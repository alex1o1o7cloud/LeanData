import Mathlib

namespace PQRS_product_eq_one_l50_50101

noncomputable def P := Real.sqrt 2011 + Real.sqrt 2012
noncomputable def Q := -Real.sqrt 2011 - Real.sqrt 2012
noncomputable def R := Real.sqrt 2011 - Real.sqrt 2012
noncomputable def S := Real.sqrt 2012 - Real.sqrt 2011

theorem PQRS_product_eq_one : P * Q * R * S = 1 := by
  sorry

end PQRS_product_eq_one_l50_50101


namespace min_distance_between_intersections_range_of_a_l50_50249

variable {a : ℝ}

/-- Given the function f(x) = x^2 - 2ax - 2(a + 1), 
1. Prove that the graph of function f(x) always intersects the x-axis at two distinct points.
2. For all x in the interval (-1, ∞), prove that f(x) + 3 ≥ 0 implies a ≤ sqrt 2 - 1. --/

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 2 * (a + 1)

theorem min_distance_between_intersections (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, (f x₁ a = 0) ∧ (f x₂ a = 0) ∧ (x₁ ≠ x₂) ∧ (dist x₁ x₂ = 2) := sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 < x → f x a + 3 ≥ 0) → a ≤ Real.sqrt 2 - 1 := sorry

end min_distance_between_intersections_range_of_a_l50_50249


namespace slower_speed_l50_50864

theorem slower_speed (f e d : ℕ) (h1 : f = 14) (h2 : e = 20) (h3 : d = 50) (x : ℕ) : 
  (50 / x : ℚ) = (50 / 14 : ℚ) + (20 / 14 : ℚ) → x = 10 := by
  sorry

end slower_speed_l50_50864


namespace simplify_expression_l50_50945

variable (a : ℝ)

theorem simplify_expression : 5 * a + 2 * a + 3 * a - 2 * a = 8 * a :=
by
  sorry

end simplify_expression_l50_50945


namespace total_pay_is_880_l50_50489

theorem total_pay_is_880 (X_pay Y_pay : ℝ) 
  (hY : Y_pay = 400)
  (hX : X_pay = 1.2 * Y_pay):
  X_pay + Y_pay = 880 :=
by
  sorry

end total_pay_is_880_l50_50489


namespace square_of_cube_of_third_smallest_prime_l50_50830

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_smallest_prime (n : ℕ) : ℕ :=
  (List.filter is_prime (List.range (n * n))).nth (n - 1).getD 0

-- The third smallest prime number
def third_smallest_prime : ℕ := nth_smallest_prime 3

-- The cube of a number
def cube (x : ℕ) : ℕ := x * x * x

-- The square of a number
def square (x : ℕ) : ℕ := x * x

theorem square_of_cube_of_third_smallest_prime : square (cube third_smallest_prime) = 15625 := by
  sorry

end square_of_cube_of_third_smallest_prime_l50_50830


namespace height_difference_l50_50167

def pine_tree_height : ℚ := 12 + 1 / 4
def maple_tree_height : ℚ := 18 + 1 / 2

theorem height_difference :
  maple_tree_height - pine_tree_height = 6 + 1 / 4 :=
by sorry

end height_difference_l50_50167


namespace alice_bob_meet_after_six_turns_l50_50365

/-
Alice and Bob play a game involving a circle whose circumference
is divided by 12 equally-spaced points. The points are numbered
clockwise, from 1 to 12. Both start on point 12. Alice moves clockwise
and Bob, counterclockwise. In a turn of the game, Alice moves 5 points 
clockwise and Bob moves 9 points counterclockwise. The game ends when they stop on
the same point. 
-/
theorem alice_bob_meet_after_six_turns (k : ℕ) :
  (5 * k) % 12 = (12 - (9 * k) % 12) % 12 -> k = 6 :=
by
  sorry

end alice_bob_meet_after_six_turns_l50_50365


namespace binom_18_6_eq_13260_l50_50549

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end binom_18_6_eq_13260_l50_50549


namespace max_distance_circle_to_line_l50_50483

open Real

theorem max_distance_circle_to_line :
  let circle_eq (x y : ℝ) := x^2 + y^2 - 2*x - 2*y + 1 = 0
  let line_eq (x y : ℝ) := x - y = 2
  ∃ (M : ℝ), (∀ x y, circle_eq x y → ∀ (d : ℝ), (line_eq x y → M ≤ d)) ∧ M = sqrt 2 + 1 :=
by
  sorry

end max_distance_circle_to_line_l50_50483


namespace number_of_20_paise_coins_l50_50985

theorem number_of_20_paise_coins (x y : ℕ) (h1 : x + y = 324) (h2 : 20 * x + 25 * y = 7000) : x = 220 :=
  sorry

end number_of_20_paise_coins_l50_50985


namespace shaded_rectangle_ratio_l50_50227

/-- Define conditions involved in the problem -/
def side_length_large_square : ℕ := 50
def num_rows_cols_grid : ℕ := 5
def rows_spanned_rect : ℕ := 2
def cols_spanned_rect : ℕ := 3

/-- Calculate the side length of a small square in the grid -/
def side_length_small_square := side_length_large_square / num_rows_cols_grid

/-- Calculate the area of the large square -/
def area_large_square := side_length_large_square * side_length_large_square

/-- Calculate the area of the shaded rectangle -/
def area_shaded_rectangle :=
  (rows_spanned_rect * side_length_small_square) *
  (cols_spanned_rect * side_length_small_square)

/-- Prove the ratio of the shaded rectangle's area to the large square's area -/
theorem shaded_rectangle_ratio : 
  (area_shaded_rectangle : ℚ) / area_large_square = 6/25 := by
  sorry

end shaded_rectangle_ratio_l50_50227


namespace evaluate_expression_l50_50392

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end evaluate_expression_l50_50392


namespace mike_spending_l50_50460

noncomputable def marbles_cost : ℝ := 9.05
noncomputable def football_cost : ℝ := 4.95
noncomputable def baseball_cost : ℝ := 6.52

noncomputable def toy_car_original_cost : ℝ := 6.50
noncomputable def toy_car_discount : ℝ := 0.20
noncomputable def toy_car_discounted_cost : ℝ := toy_car_original_cost * (1 - toy_car_discount)

noncomputable def puzzle_cost : ℝ := 3.25
noncomputable def puzzle_total_cost : ℝ := puzzle_cost -- 'buy one get one free' condition

noncomputable def action_figure_original_cost : ℝ := 15.00
noncomputable def action_figure_discounted_cost : ℝ := 10.50

noncomputable def total_cost : ℝ := marbles_cost + football_cost + baseball_cost + toy_car_discounted_cost + puzzle_total_cost + action_figure_discounted_cost

theorem mike_spending : total_cost = 39.47 := by
  sorry

end mike_spending_l50_50460


namespace original_fraction_l50_50301

theorem original_fraction (n d : ℤ) (h1 : d = n + 5) (h2 : (n + 1) / (d + 1) = 7 / 12) : (n, d) = (6, 6 + 5) := 
by
  sorry

end original_fraction_l50_50301


namespace find_m_if_extraneous_root_l50_50422

theorem find_m_if_extraneous_root :
  (∃ x : ℝ, x = 2 ∧ (∀ z : ℝ, z ≠ 2 → (m / (z-2) - 2*z / (2-z) = 1)) ∧ m = -4) :=
sorry

end find_m_if_extraneous_root_l50_50422


namespace correct_parameterization_l50_50888

noncomputable def parametrize_curve (t : ℝ) : ℝ × ℝ :=
  (t, t^2)

theorem correct_parameterization : ∀ t : ℝ, ∃ x y : ℝ, parametrize_curve t = (x, y) ∧ y = x^2 :=
by
  intro t
  use t, t^2
  dsimp [parametrize_curve]
  exact ⟨rfl, rfl⟩

end correct_parameterization_l50_50888


namespace fifteenth_term_ratio_l50_50145

noncomputable def U (n : ℕ) (c f : ℚ) := n * (2 * c + (n - 1) * f) / 2
noncomputable def V (n : ℕ) (g h : ℚ) := n * (2 * g + (n - 1) * h) / 2

theorem fifteenth_term_ratio (c f g h : ℚ)
  (h1 : ∀ n : ℕ, (n > 0) → (U n c f) / (V n g h) = (5 * (n * n) + 3 * n + 2) / (3 * (n * n) + 2 * n + 30)) :
  (c + 14 * f) / (g + 14 * h) = 125 / 99 :=
by
  sorry

end fifteenth_term_ratio_l50_50145


namespace trigonometric_identity_l50_50259

theorem trigonometric_identity (α : ℝ) (h : Real.tan (π - α) = -2) :
  (Real.cos (2 * π - α) + 2 * Real.cos (3 * π / 2 - α)) / (Real.sin (π - α) - Real.sin (-π / 2 - α)) = -1 :=
by
  sorry

end trigonometric_identity_l50_50259


namespace P_eq_Q_l50_50631

def P (m : ℝ) : Prop := -1 < m ∧ m < 0

def quadratic_inequality (m : ℝ) (x : ℝ) : Prop := m * x^2 + 4 * m * x - 4 < 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, quadratic_inequality m x

theorem P_eq_Q : ∀ m : ℝ, P m ↔ Q m := 
by 
  sorry

end P_eq_Q_l50_50631


namespace taxi_fare_distance_l50_50657

-- Define the fare calculation and distance function
def fare (x : ℕ) : ℝ :=
  if x ≤ 4 then 10
  else 10 + (x - 4) * 1.5

-- Proof statement
theorem taxi_fare_distance (x : ℕ) : fare x = 16 → x = 8 :=
by
  -- Proof skipped
  sorry

end taxi_fare_distance_l50_50657


namespace greatest_3digit_base8_divisible_by_7_l50_50669

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), n = 0b777 ∧ (base8_to_base10 0b777) % 7 = 0 ∧ ∀ m < 0o777, m % 7 = 0 → base8_to_base10 m < base8_to_base10 0b777 :=
by
  sorry

end greatest_3digit_base8_divisible_by_7_l50_50669


namespace savings_per_bagel_in_cents_l50_50807

def cost_of_one_bagel : ℝ := 2.25
def cost_of_dozen_bagels : ℝ := 24.0
def number_of_bagels_in_dozen : ℕ := 12

theorem savings_per_bagel_in_cents :
  let total_cost_individual := number_of_bagels_in_dozen * cost_of_one_bagel in
  let total_savings_dollar := total_cost_individual - cost_of_dozen_bagels in
  let savings_per_bagel_dollar := total_savings_dollar / number_of_bagels_in_dozen in
  let savings_per_bagel_cents := savings_per_bagel_dollar * 100 in
  savings_per_bagel_cents = 25 :=
begin
  sorry
end

end savings_per_bagel_in_cents_l50_50807


namespace cylindrical_coordinates_cone_shape_l50_50775

def cylindrical_coordinates := Type

def shape_description (r θ z : ℝ) : Prop :=
θ = 2 * z

theorem cylindrical_coordinates_cone_shape (r θ z : ℝ) :
  shape_description r θ z → θ = 2 * z → Prop := sorry

end cylindrical_coordinates_cone_shape_l50_50775


namespace calculate_discount_l50_50448

theorem calculate_discount
  (original_cost : ℝ)
  (amount_spent : ℝ)
  (h1 : original_cost = 35.00)
  (h2 : amount_spent = 18.00) :
  original_cost - amount_spent = 17.00 :=
by
  sorry

end calculate_discount_l50_50448


namespace find_y_l50_50896

theorem find_y :
  ∃ y : ℝ, ((0.47 * 1442) - (0.36 * y) + 65 = 5) ∧ y = 2049.28 :=
by
  sorry

end find_y_l50_50896


namespace box_width_l50_50857

variable (l h vc : ℕ)
variable (nc : ℕ)
variable (v : ℕ)

-- Given
def length_box := 8
def height_box := 5
def volume_cube := 10
def num_cubes := 60
def volume_box := num_cubes * volume_cube

-- To Prove
theorem box_width : (volume_box = l * h * w) → w = 15 :=
by
  intro h1
  sorry

end box_width_l50_50857


namespace area_ratio_of_quadrilateral_ADGJ_to_decagon_l50_50164

noncomputable def ratio_of_areas (k : ℝ) : ℝ :=
  (2 * k^2 * Real.sin (72 * Real.pi / 180)) / (5 * Real.sqrt (5 + 2 * Real.sqrt 5))

theorem area_ratio_of_quadrilateral_ADGJ_to_decagon
  (k : ℝ) :
  ∃ (n m : ℝ), m / n = ratio_of_areas k :=
  sorry

end area_ratio_of_quadrilateral_ADGJ_to_decagon_l50_50164


namespace football_game_attendance_l50_50021

-- Define the initial conditions
def saturday : ℕ := 80
def monday : ℕ := saturday - 20
def wednesday : ℕ := monday + 50
def friday : ℕ := saturday + monday
def total_week_actual : ℕ := saturday + monday + wednesday + friday
def total_week_expected : ℕ := 350

-- Define the proof statement
theorem football_game_attendance : total_week_actual - total_week_expected = 40 :=
by 
  -- Proof steps would go here
  sorry

end football_game_attendance_l50_50021


namespace quadratic_intersection_with_x_axis_l50_50479

theorem quadratic_intersection_with_x_axis :
  ∃ x : ℝ, (x^2 - 4*x + 4 = 0) ∧ (x = 2) ∧ (x, 0) = (2, 0) :=
sorry

end quadratic_intersection_with_x_axis_l50_50479


namespace correct_pythagorean_triple_l50_50070

def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

theorem correct_pythagorean_triple :
  (is_pythagorean_triple 1 2 3 = false) ∧ 
  (is_pythagorean_triple 4 5 6 = false) ∧ 
  (is_pythagorean_triple 6 8 9 = false) ∧ 
  (is_pythagorean_triple 7 24 25 = true) :=
by
  sorry

end correct_pythagorean_triple_l50_50070


namespace solve_equation_l50_50238

noncomputable def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, x = Real.arcsin (3/4) + 2 * k * Real.pi ∨ x = Real.pi - Real.arcsin (3/4) + 2 * k * Real.pi

theorem solve_equation (x : ℝ) :
  (5 * Real.sin x = 4 + 2 * Real.cos (2 * x)) ↔ solution_set x := 
sorry

end solve_equation_l50_50238


namespace required_percentage_to_pass_l50_50699

-- Definitions based on conditions
def obtained_marks : ℕ := 175
def failed_by : ℕ := 56
def max_marks : ℕ := 700
def pass_marks : ℕ := obtained_marks + failed_by

-- Theorem stating the required percentage to pass
theorem required_percentage_to_pass : 
  (pass_marks : ℚ) / max_marks * 100 = 33 := 
by 
  sorry

end required_percentage_to_pass_l50_50699


namespace product_PA_PB_eq_nine_l50_50057

theorem product_PA_PB_eq_nine 
  (P A B : ℝ × ℝ) 
  (hP : P = (3, 1)) 
  (h1 : A ≠ B)
  (h2 : ∃ L : ℝ × ℝ → Prop, L P ∧ L A ∧ L B) 
  (h3 : A.fst ^ 2 + A.snd ^ 2 = 1) 
  (h4 : B.fst ^ 2 + B.snd ^ 2 = 1) : 
  |((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2)| * |((P.1 - B.1) ^ 2 + (P.2 - B.2) ^ 2)| = 9 := 
sorry

end product_PA_PB_eq_nine_l50_50057


namespace factor_theorem_solution_l50_50889

theorem factor_theorem_solution (t : ℝ) :
  (6 * t ^ 2 - 17 * t - 7 = 0) ↔ 
  (t = (17 + Real.sqrt 457) / 12 ∨ t = (17 - Real.sqrt 457) / 12) :=
by sorry

end factor_theorem_solution_l50_50889


namespace Jake_watched_hours_on_Friday_l50_50134

theorem Jake_watched_hours_on_Friday :
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  total_show_hours - total_hours_before_Friday = 19 :=
by
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  sorry

end Jake_watched_hours_on_Friday_l50_50134


namespace example_of_four_three_digit_numbers_sum_2012_two_digits_exists_l50_50797

-- Define what it means to be a three-digit number using only two distinct digits
def two_digit_natural (d1 d2 : ℕ) (n : ℕ) : Prop :=
  (∀ (d : ℕ), d ∈ n.digits 10 → d = d1 ∨ d = d2) ∧ 100 ≤ n ∧ n < 1000

-- State the main theorem
theorem example_of_four_three_digit_numbers_sum_2012_two_digits_exists :
  ∃ a b c d : ℕ, 
    two_digit_natural 3 5 a ∧
    two_digit_natural 3 5 b ∧
    two_digit_natural 3 5 c ∧
    two_digit_natural 3 5 d ∧
    a + b + c + d = 2012 :=
by
  sorry

end example_of_four_three_digit_numbers_sum_2012_two_digits_exists_l50_50797


namespace value_of_squared_difference_l50_50423

theorem value_of_squared_difference (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) :
  (x - y)^2 = 9 :=
by
  sorry

end value_of_squared_difference_l50_50423


namespace bob_correct_answers_l50_50607

-- Define the variables, c for correct answers, w for incorrect answers, total problems 15, score 54
variables (c w : ℕ)

-- Define the conditions
axiom total_problems : c + w = 15
axiom total_score : 6 * c - 3 * w = 54

-- Prove that the number of correct answers is 11
theorem bob_correct_answers : c = 11 :=
by
  -- Here, you would provide the proof, but for the sake of the statement, we'll use sorry.
  sorry

end bob_correct_answers_l50_50607


namespace nine_a_plus_a_plus_nine_l50_50402

theorem nine_a_plus_a_plus_nine (A : Nat) (hA : 0 < A) : 
  10 * A + 9 = 9 * A + (A + 9) := 
by 
  sorry

end nine_a_plus_a_plus_nine_l50_50402


namespace find_a_range_l50_50585

noncomputable def f (x a : ℝ) := 2 ^ (x * (x - a))

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → deriv (λ x, f x a) x < 0) → 2 ≤ a :=
by sorry

end find_a_range_l50_50585


namespace algebraic_expression_value_l50_50572

theorem algebraic_expression_value (x y : ℝ) (h : x = 2 * y + 3) : 4 * x - 8 * y + 9 = 21 := by
  sorry

end algebraic_expression_value_l50_50572


namespace Jake_watched_hours_on_Friday_l50_50135

theorem Jake_watched_hours_on_Friday :
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  total_show_hours - total_hours_before_Friday = 19 :=
by
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  sorry

end Jake_watched_hours_on_Friday_l50_50135


namespace ellipse_fence_cost_is_correct_l50_50873

noncomputable def ellipse_perimeter (a b : ℝ) : ℝ :=
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

noncomputable def fence_cost_per_meter (rate : ℝ) (a b : ℝ) : ℝ :=
  rate * ellipse_perimeter a b

theorem ellipse_fence_cost_is_correct :
  fence_cost_per_meter 3 16 12 = 265.32 :=
by
  sorry

end ellipse_fence_cost_is_correct_l50_50873


namespace inverse_square_variation_l50_50841

variable (x y : ℝ)

theorem inverse_square_variation (h1 : x = 1) (h2 : y = 3) (h3 : y = 2) : x = 2.25 :=
by
  sorry

end inverse_square_variation_l50_50841


namespace second_mechanic_hours_l50_50490

theorem second_mechanic_hours (x y : ℕ) (h1 : 45 * x + 85 * y = 1100) (h2 : x + y = 20) : y = 5 :=
by
  sorry

end second_mechanic_hours_l50_50490


namespace convert_to_scientific_notation_l50_50636

theorem convert_to_scientific_notation :
  (26.62 * 10^9) = 2.662 * 10^9 :=
by
  sorry

end convert_to_scientific_notation_l50_50636


namespace area_of_square_on_RS_l50_50322

theorem area_of_square_on_RS (PQ QR PS PS_square PQ_square QR_square : ℝ)
  (hPQ : PQ_square = 25) (hQR : QR_square = 49) (hPS : PS_square = 64)
  (hPQ_eq : PQ_square = PQ^2) (hQR_eq : QR_square = QR^2) (hPS_eq : PS_square = PS^2)
  : ∃ RS_square : ℝ, RS_square = 138 := by
  let PR_square := PQ^2 + QR^2
  let RS_square := PR_square + PS^2
  use RS_square
  sorry

end area_of_square_on_RS_l50_50322


namespace sum_of_integers_is_28_l50_50766

theorem sum_of_integers_is_28 (m n p q : ℕ) (hmnpq_diff : m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q)
  (hm_pos : 0 < m) (hn_pos : 0 < n) (hp_pos : 0 < p) (hq_pos : 0 < q)
  (h_prod : (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4) : m + n + p + q = 28 :=
by
  sorry

end sum_of_integers_is_28_l50_50766


namespace sum_of_thetas_l50_50813

noncomputable def theta (k : ℕ) : ℝ := (54 + 72 * k) % 360

theorem sum_of_thetas : (theta 0 + theta 1 + theta 2 + theta 3 + theta 4) = 990 :=
by
  -- proof goes here
  sorry

end sum_of_thetas_l50_50813


namespace right_triangle_hypotenuse_45_deg_4_inradius_l50_50868

theorem right_triangle_hypotenuse_45_deg_4_inradius : 
  ∀ (R : ℝ) (hypotenuse_length : ℝ), R = 4 ∧ 
  (∀ (A B C : ℝ), A = 45 ∧ B = 45 ∧ C = 90) →
  hypotenuse_length = 8 :=
by
  sorry

end right_triangle_hypotenuse_45_deg_4_inradius_l50_50868


namespace smallest_part_division_l50_50125

theorem smallest_part_division (y : ℝ) (h1 : y > 0) :
  ∃ (x : ℝ), x = y / 9 ∧ (∃ (a b c : ℝ), a = x ∧ b = 3 * x ∧ c = 5 * x ∧ a + b + c = y) :=
sorry

end smallest_part_division_l50_50125


namespace class_students_l50_50997

theorem class_students :
  ∃ n : ℕ,
    (∃ m : ℕ, 2 * m = n) ∧
    (∃ q : ℕ, 4 * q = n) ∧
    (∃ l : ℕ, 7 * l = n) ∧
    (∀ f : ℕ, f < 6 → n - (n / 2) - (n / 4) - (n / 7) = f) ∧
    n = 28 :=
by
  sorry

end class_students_l50_50997


namespace bowling_ball_weight_l50_50723

theorem bowling_ball_weight :
  (∃ (b c : ℝ), 8 * b = 4 * c ∧ 2 * c = 64) → ∃ b : ℝ, b = 16 :=
by
  sorry

end bowling_ball_weight_l50_50723


namespace min_value_frac_sum_l50_50907

open Real

theorem min_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) :
    ∃ (z : ℝ), z = 1 + (sqrt 3) / 2 ∧ 
    (∀ t, (t > 0 → ∃ (u : ℝ), u > 0 ∧ t + u = 4 → ∀ t' (h : t' = (1 / t) + (3 / u)), t' ≥ z)) :=
by sorry

end min_value_frac_sum_l50_50907


namespace kubik_family_arrangements_l50_50528

theorem kubik_family_arrangements (n : ℕ) (h_n : n = 7) :
  let total_arrangements := (n - 1)!
  let invalid_arrangements := 2 * (n - 2)!
  let valid_arrangements := total_arrangements - invalid_arrangements
  valid_arrangements = 480 :=
by
  sorry

end kubik_family_arrangements_l50_50528


namespace volume_of_wall_is_16128_l50_50659

def wall_width : ℝ := 4
def wall_height : ℝ := 6 * wall_width
def wall_length : ℝ := 7 * wall_height

def wall_volume : ℝ := wall_length * wall_width * wall_height

theorem volume_of_wall_is_16128 :
  wall_volume = 16128 := by
  sorry

end volume_of_wall_is_16128_l50_50659


namespace enumerate_A_l50_50590

-- Define the set A according to the given conditions
def A : Set ℕ := {X : ℕ | 8 % (6 - X) = 0}

-- The equivalent proof problem
theorem enumerate_A : A = {2, 4, 5} :=
by sorry

end enumerate_A_l50_50590


namespace tan_theta_eq_sqrt_3_of_f_maximum_l50_50810

theorem tan_theta_eq_sqrt_3_of_f_maximum (θ : ℝ) 
  (h : ∀ x : ℝ, 3 * Real.sin (x + (Real.pi / 6)) ≤ 3 * Real.sin (θ + (Real.pi / 6))) : 
  Real.tan θ = Real.sqrt 3 :=
sorry

end tan_theta_eq_sqrt_3_of_f_maximum_l50_50810


namespace slower_speed_is_correct_l50_50862

/-- 
A person walks at 14 km/hr instead of a slower speed, 
and as a result, he would have walked 20 km more. 
The actual distance travelled by him is 50 km. 
What is the slower speed he usually walks at?
-/
theorem slower_speed_is_correct :
    ∃ x : ℝ, (14 * (50 / 14) - (x * (30 / x))) = 20 ∧ x = 8.4 :=
by
  sorry

end slower_speed_is_correct_l50_50862


namespace find_x_l50_50576

def f (x : ℝ) : ℝ := 2 * x - 3 -- Definition of the function f

def c : ℝ := 11 -- Definition of the constant c

theorem find_x : 
  ∃ x : ℝ, 2 * f x - c = f (x - 2) ↔ x = 5 :=
by 
  sorry

end find_x_l50_50576


namespace find_number_l50_50239

theorem find_number (x : ℕ) (h : x = 4) : x + 1 = 5 :=
by
  sorry

end find_number_l50_50239


namespace exists_x_mean_absolute_deviation_eq_half_l50_50798

theorem exists_x_mean_absolute_deviation_eq_half 
  {n : ℕ} (hn : 0 < n) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) : 
  ∃ t ∈ Icc (0 : ℝ) 1, (1 / (n : ℝ)) * (Finset.univ.sum (λ i, |t - x i|)) = 1 / 2 :=
begin
  sorry
end

end exists_x_mean_absolute_deviation_eq_half_l50_50798


namespace three_numbers_less_or_equal_than_3_l50_50321

theorem three_numbers_less_or_equal_than_3 : 
  let a := 0.8
  let b := 0.5
  let c := 0.9
  (a ≤ 3) ∧ (b ≤ 3) ∧ (c ≤ 3) → 
  3 = 3 :=
by
  intros h
  sorry

end three_numbers_less_or_equal_than_3_l50_50321


namespace grill_run_time_l50_50510

-- Define the conditions 1, 2, and 3
def burns_rate : ℕ := 15 -- coals burned every 20 minutes
def burns_time : ℕ := 20 -- minutes to burn some coals
def coals_per_bag : ℕ := 60 -- coals per bag
def num_bags : ℕ := 3 -- number of bags

-- The main theorem to prove the time taken to burn all the coals
theorem grill_run_time : 
  let total_coals := num_bags * coals_per_bag in
  let burn_time_per_coals := total_coals * burns_time / burns_rate in
  burn_time_per_coals / 60 = 4 := 
by
  sorry

end grill_run_time_l50_50510


namespace bobby_additional_candy_l50_50074

variable (initial_candy additional_candy chocolate total_candy : ℕ)
variable (bobby_initial_candy : initial_candy = 38)
variable (bobby_ate_chocolate : chocolate = 16)
variable (bobby_more_candy : initial_candy + additional_candy = 58 + chocolate)

theorem bobby_additional_candy :
  additional_candy = 36 :=
by {
  sorry
}

end bobby_additional_candy_l50_50074


namespace James_total_area_l50_50270

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase_dimension : ℕ := 2
def number_of_new_rooms : ℕ := 4
def larger_room_multiplier : ℕ := 2

noncomputable def new_length : ℕ := initial_length + increase_dimension
noncomputable def new_width : ℕ := initial_width + increase_dimension
noncomputable def area_of_one_new_room : ℕ := new_length * new_width
noncomputable def total_area_of_4_rooms : ℕ := area_of_one_new_room * number_of_new_rooms
noncomputable def area_of_larger_room : ℕ := area_of_one_new_room * larger_room_multiplier
noncomputable def total_area : ℕ := total_area_of_4_rooms + area_of_larger_room

theorem James_total_area : total_area = 1800 := 
by
  sorry

end James_total_area_l50_50270


namespace intersection_of_A_and_B_l50_50099

def setA : Set ℝ := {x | abs (x - 1) < 2}

def setB : Set ℝ := {x | x^2 + x - 2 > 0}

theorem intersection_of_A_and_B :
  (setA ∩ setB) = {x | 1 < x ∧ x < 3} :=
sorry

end intersection_of_A_and_B_l50_50099


namespace ratio_of_larger_to_smaller_l50_50965

theorem ratio_of_larger_to_smaller (x y : ℝ) (hx : x > y) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by
  sorry

end ratio_of_larger_to_smaller_l50_50965


namespace perimeter_ACFHK_is_correct_l50_50128

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the points of the pentagon within the dodecagon
def ACFHK_points : ℕ := 5

-- Define the perimeter of the pentagon ACFHK in the dodecagon
noncomputable def perimeter_of_ACFHK : ℝ :=
  let triangle_side := radius
  let isosceles_right_triangle_side := radius * Real.sqrt 2
  3 * triangle_side + 2 * isosceles_right_triangle_side

-- Verify that the calculated perimeter matches the expected value
theorem perimeter_ACFHK_is_correct : perimeter_of_ACFHK = 18 + 12 * Real.sqrt 2 :=
  sorry

end perimeter_ACFHK_is_correct_l50_50128


namespace cube_edge_length_l50_50046

theorem cube_edge_length (a : ℝ) (base_length : ℝ) (base_width : ℝ) (rise_height : ℝ) 
  (h_conditions : base_length = 20 ∧ base_width = 15 ∧ rise_height = 11.25 ∧ 
                  (base_length * base_width * rise_height) = a^3) : 
  a = 15 := 
by
  sorry

end cube_edge_length_l50_50046


namespace smallest_int_ending_in_9_divisible_by_11_l50_50193

theorem smallest_int_ending_in_9_divisible_by_11:
  ∃ x : ℕ, (∃ k : ℤ, x = 10 * k + 9) ∧ x % 11 = 0 ∧ x = 99 :=
by
  sorry

end smallest_int_ending_in_9_divisible_by_11_l50_50193


namespace equivalence_of_sum_cubed_expression_l50_50092

theorem equivalence_of_sum_cubed_expression (a b : ℝ) 
  (h₁ : a + b = 5) (h₂ : a * b = -14) : a^3 + a^2 * b + a * b^2 + b^3 = 265 :=
sorry

end equivalence_of_sum_cubed_expression_l50_50092


namespace eval_expr_ceil_floor_l50_50396

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end eval_expr_ceil_floor_l50_50396


namespace min_species_needed_l50_50437

theorem min_species_needed (num_birds : ℕ) (h1 : num_birds = 2021)
  (h2 : ∀ (s : ℤ) (x y : ℕ), x ≠ y → (between_same_species : ℕ) → (h3 : between_same_species = y - x - 1) → between_same_species % 2 = 0) :
  ∃ (species : ℕ), num_birds ≤ 2 * species ∧ species = 1011 :=
by
  sorry

end min_species_needed_l50_50437


namespace solve_congruence_l50_50948

theorem solve_congruence (n : ℤ) (h : 13 * n ≡ 9 [MOD 47]) : n ≡ 39 [MOD 47] := 
  sorry

end solve_congruence_l50_50948


namespace solution_set_of_new_inequality_l50_50917

-- Define the conditions
variable (a b c x : ℝ)

-- ax^2 + bx + c > 0 has solution set {-3 < x < 2}
def inequality_solution_set (a b c : ℝ) : Prop := ∀ x : ℝ, (-3 < x ∧ x < 2) → a * x^2 + b * x + c > 0

-- Prove that cx^2 + bx + a > 0 has solution set {x < -1/3 ∨ x > 1/2}
theorem solution_set_of_new_inequality
  (a b c : ℝ)
  (h : a < 0 ∧ inequality_solution_set a b c) :
  ∀ x : ℝ, (x < -1/3 ∨ x > 1/2) ↔ (c * x^2 + b * x + a > 0) := sorry

end solution_set_of_new_inequality_l50_50917


namespace intersection_M_N_l50_50916

-- Definitions of the domains M and N
def M := {x : ℝ | x < 1}
def N := {x : ℝ | x > 0}

-- The goal is to prove that the intersection of M and N is equal to (0, 1)
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l50_50916


namespace daily_wage_of_c_l50_50035

-- Define the conditions
variables (a b c : ℝ)
variables (h_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5)
variables (h_days : 6 * a + 9 * b + 4 * c = 1702)

-- Define the proof problem; to prove c = 115
theorem daily_wage_of_c (h_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5) (h_days : 6 * a + 9 * b + 4 * c = 1702) : 
  c = 115 :=
sorry

end daily_wage_of_c_l50_50035


namespace second_investment_amount_l50_50120

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

theorem second_investment_amount :
  ∀ (P₁ P₂ I₁ I₂ r t : ℝ), 
    P₁ = 5000 →
    I₁ = 250 →
    I₂ = 1000 →
    I₁ = simple_interest P₁ r t →
    I₂ = simple_interest P₂ r t →
    P₂ = 20000 := 
by 
  intros P₁ P₂ I₁ I₂ r t hP₁ hI₁ hI₂ hI₁_eq hI₂_eq
  sorry

end second_investment_amount_l50_50120


namespace gcd_lcm_of_300_105_l50_50975

theorem gcd_lcm_of_300_105 :
  ∃ g l : ℕ, g = Int.gcd 300 105 ∧ l = Nat.lcm 300 105 ∧ g = 15 ∧ l = 2100 :=
by
  let g := Int.gcd 300 105
  let l := Nat.lcm 300 105
  have g_def : g = 15 := sorry
  have l_def : l = 2100 := sorry
  exact ⟨g, l, ⟨g_def, ⟨l_def, ⟨g_def, l_def⟩⟩⟩⟩

end gcd_lcm_of_300_105_l50_50975


namespace N_vector_3_eq_result_vector_l50_50146

noncomputable def matrix_N : Matrix (Fin 2) (Fin 2) ℝ :=
-- The matrix N is defined such that:
-- N * (vector 3 -2) = (vector 4 1)
-- N * (vector -2 3) = (vector 1 2)
sorry

def vector_1 : Fin 2 → ℝ := fun | ⟨0,_⟩ => 3 | ⟨1,_⟩ => -2
def vector_2 : Fin 2 → ℝ := fun | ⟨0,_⟩ => -2 | ⟨1,_⟩ => 3
def vector_3 : Fin 2 → ℝ := fun | ⟨0,_⟩ => 7 | ⟨1,_⟩ => 0
def result_vector : Fin 2 → ℝ := fun | ⟨0,_⟩ => 14 | ⟨1,_⟩ => 7

theorem N_vector_3_eq_result_vector :
  matrix_N.mulVec vector_3 = result_vector := by
  -- Given conditions:
  -- matrix_N.mulVec vector_1 = vector_4
  -- and matrix_N.mulVec vector_2 = vector_5
  sorry

end N_vector_3_eq_result_vector_l50_50146


namespace largest_n_fact_product_of_four_consecutive_integers_l50_50738

theorem largest_n_fact_product_of_four_consecutive_integers :
  ∀ (n : ℕ), (∃ x : ℕ, n.factorial = x * (x + 1) * (x + 2) * (x + 3)) → n ≤ 6 :=
by
  sorry

end largest_n_fact_product_of_four_consecutive_integers_l50_50738


namespace largest_polygon_area_l50_50426

structure Polygon :=
(unit_squares : Nat)
(right_triangles : Nat)

def area (p : Polygon) : ℝ :=
p.unit_squares + 0.5 * p.right_triangles

def polygon_A : Polygon := { unit_squares := 6, right_triangles := 2 }
def polygon_B : Polygon := { unit_squares := 7, right_triangles := 1 }
def polygon_C : Polygon := { unit_squares := 8, right_triangles := 0 }
def polygon_D : Polygon := { unit_squares := 5, right_triangles := 4 }
def polygon_E : Polygon := { unit_squares := 6, right_triangles := 2 }

theorem largest_polygon_area :
  max (area polygon_A) (max (area polygon_B) (max (area polygon_C) (max (area polygon_D) (area polygon_E)))) = area polygon_C :=
by
  sorry

end largest_polygon_area_l50_50426


namespace first_number_eq_l50_50860

theorem first_number_eq (x y : ℝ) (h1 : x * 120 = 346) (h2 : y * 240 = 346) : x = 346 / 120 :=
by
  -- The final proof will be inserted here
  sorry

end first_number_eq_l50_50860


namespace probability_nan_kai_l50_50223

theorem probability_nan_kai :
  let total_outcomes := Nat.choose 6 4
  let successful_outcomes := Nat.choose 4 4
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 1 / 15 :=
by
  sorry

end probability_nan_kai_l50_50223


namespace divide_real_set_l50_50800

theorem divide_real_set (S : Finset ℝ) :
  ∃ G1 G2 : Finset ℝ, S = G1 ∪ G2 ∧ G1 ∩ G2 = ∅ ∧
  (∀ a b ∈ G1, ∀ k : ℤ, a ≠ b → |a - b| ≠ 3 ^ k) ∧
  (∀ a b ∈ G2, ∀ k : ℤ, a ≠ b → |a - b| ≠ 3 ^ k) :=
by sorry

end divide_real_set_l50_50800


namespace sec_150_eq_neg_2_sqrt_3_div_3_csc_150_eq_2_l50_50222

noncomputable def sec (x : ℝ) := 1 / Real.cos x
noncomputable def csc (x : ℝ) := 1 / Real.sin x

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := by
  sorry

theorem csc_150_eq_2 : csc (150 * Real.pi / 180) = 2 := by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_csc_150_eq_2_l50_50222


namespace arithmetic_seq_sum_l50_50923

theorem arithmetic_seq_sum {a_n : ℕ → ℤ} {d : ℤ} (S_n : ℕ → ℤ) :
  (∀ n : ℕ, S_n n = -(n * n)) →
  (∃ d, d = -2 ∧ ∀ n, a_n n = -2 * n + 1) :=
by
  -- Assuming that S_n is given as per the condition of the problem
  sorry

end arithmetic_seq_sum_l50_50923


namespace find_percentage_l50_50044

-- conditions
def N : ℕ := 160
def expected_percentage : ℕ := 35

-- statement to prove
theorem find_percentage (P : ℕ) (h : P / 100 * N = 50 / 100 * N - 24) : P = expected_percentage :=
sorry

end find_percentage_l50_50044


namespace geometric_sequence_n_l50_50444

theorem geometric_sequence_n (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 * a 2 * a 3 = 4) 
  (h2 : a 4 * a 5 * a 6 = 12) 
  (h3 : a (n-1) * a n * a (n+1) = 324) : 
  n = 14 := 
  sorry

end geometric_sequence_n_l50_50444


namespace largest_divisor_of_seven_consecutive_odd_numbers_l50_50495

theorem largest_divisor_of_seven_consecutive_odd_numbers (n : ℕ) (h : Even n) (h_pos : n > 0) :
  ∃ d, d = 45 ∧ ∀ k, k ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) → k ≤ 45 :=
sorry

end largest_divisor_of_seven_consecutive_odd_numbers_l50_50495


namespace more_people_attended_l50_50024

def saturday_attendance := 80
def monday_attendance := saturday_attendance - 20
def wednesday_attendance := monday_attendance + 50
def friday_attendance := saturday_attendance + monday_attendance
def expected_audience := 350

theorem more_people_attended :
  saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance - expected_audience = 40 :=
by
  sorry

end more_people_attended_l50_50024


namespace modulo_residue_addition_l50_50373

theorem modulo_residue_addition : 
  (368 + 3 * 78 + 8 * 242 + 6 * 22) % 11 = 8 := 
by
  have h1 : 368 % 11 = 5 := by sorry
  have h2 : 78 % 11 = 1 := by sorry
  have h3 : 242 % 11 = 0 := by sorry
  have h4 : 22 % 11 = 0 := by sorry
  sorry

end modulo_residue_addition_l50_50373


namespace product_of_three_numbers_l50_50015

theorem product_of_three_numbers
  (x y z n : ℤ)
  (h1 : x + y + z = 165)
  (h2 : n = 7 * x)
  (h3 : n = y - 9)
  (h4 : n = z + 9) :
  x * y * z = 64328 := 
by
  sorry

end product_of_three_numbers_l50_50015


namespace ratio_of_third_week_growth_l50_50073

-- Define the given conditions
def week1_growth : ℕ := 2  -- growth in week 1
def week2_growth : ℕ := 2 * week1_growth  -- growth in week 2
def total_height : ℕ := 22  -- total height after three weeks

/- 
  Statement: Prove that the growth in the third week divided by 
  the growth in the second week is 4, i.e., the ratio 4:1.
-/
theorem ratio_of_third_week_growth :
  ∃ x : ℕ, 4 * x = (total_height - week1_growth - week2_growth) ∧ x = 4 :=
by
  use 4
  sorry

end ratio_of_third_week_growth_l50_50073


namespace triangle_angle_proof_l50_50969

theorem triangle_angle_proof (α β γ : ℝ) (hα : α > 60) (hβ : β > 60) (hγ : γ > 60) (h_sum : α + β + γ = 180) : false :=
by
  sorry

end triangle_angle_proof_l50_50969


namespace g_diff_l50_50617

noncomputable section

-- Definition of g(n) as given in the problem statement
def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((1 + Real.sqrt 3) / 2)^n +
  (3 - 2 * Real.sqrt 3) / 6 * ((1 - Real.sqrt 3) / 2)^n

-- The statement to prove g(n+2) - g(n) = -1/4 * g(n)
theorem g_diff (n : ℕ) : g (n + 2) - g n = -1 / 4 * g n :=
by
  sorry

end g_diff_l50_50617


namespace non_congruent_integer_triangles_with_perimeter_20_l50_50756

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_20 (a b c : ℕ) : Prop :=
  a + b + c = 20

def distinct_triples (T : Set (ℕ × ℕ × ℕ)) (a b c : ℕ) : Prop :=
  ∀ x ∈ T, x ≠ (a, b, c)

theorem non_congruent_integer_triangles_with_perimeter_20 :
  ∃ (T : Set (ℕ × ℕ × ℕ)), (∀ (a b c : ℕ), (a, b, c) ∈ T → is_triangle a b c ∧ perimeter_20 a b c) ∧ 
  (∀ (a b c : ℕ), is_triangle a b c → perimeter_20 a b c → (a, b, c) ∈ T) ∧ 
  (∀ (x y : (ℕ × ℕ × ℕ)), x ∈ T → y ∈ T → x ≠ y) ∧ 
  T.card = 11 :=
by
  sorry

end non_congruent_integer_triangles_with_perimeter_20_l50_50756


namespace S4k_eq_32_l50_50909

-- Definition of the problem conditions
variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)
variable (k : ℕ)

-- Conditions: Arithmetic sequence sum properties
axiom sum_arithmetic_sequence : ∀ {n : ℕ}, S n = n * (a 1 + a n) / 2

-- Given conditions
axiom Sk_eq_2 : S k = 2
axiom S3k_eq_18 : S (3 * k) = 18

-- Prove the required statement
theorem S4k_eq_32 : S (4 * k) = 32 :=
by
  sorry

end S4k_eq_32_l50_50909


namespace min_triples_in_colored_complete_graph_l50_50567

theorem min_triples_in_colored_complete_graph (k: ℕ) (n: ℕ) (r: ℕ)
  (h_n: n = 36)
  (h_r: r = 5)
  (h_k: k = 3780) :
  ∃ G : simple_graph (fin n), 
  G = complete_graph (fin n) ∧
  ∀ c : edge_coloring G r,
  ∃ t : set (fin n × fin n × fin n),
  (∀ A B C : fin n, (A, B, C) ∈ t → 
    (G.edge_set (A, B) ∧ G.edge_set (B, C) ∧ 
    c (A, B) = c (B, C))) ∧
  t.card ≥ k :=
by sorry

end min_triples_in_colored_complete_graph_l50_50567


namespace profit_percentage_on_cost_price_l50_50705

theorem profit_percentage_on_cost_price (CP MP SP : ℝ)
    (h1 : CP = 100)
    (h2 : MP = 131.58)
    (h3 : SP = 0.95 * MP) :
    ((SP - CP) / CP) * 100 = 25 :=
by
  -- Sorry to skip the proof
  sorry

end profit_percentage_on_cost_price_l50_50705


namespace eval_expr_ceil_floor_l50_50393

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end eval_expr_ceil_floor_l50_50393


namespace joe_two_different_fruits_in_a_day_l50_50527

def joe_meal_event : Type := {meal : ℕ // meal = 4}
def joe_fruit_choice : Type := {fruit : ℕ // fruit ≤ 4}

noncomputable def prob_all_same_fruit : ℚ := (1 / 4) ^ 4 * 4
noncomputable def prob_at_least_two_diff_fruits : ℚ := 1 - prob_all_same_fruit

theorem joe_two_different_fruits_in_a_day :
  prob_at_least_two_diff_fruits = 63 / 64 :=
by
  sorry

end joe_two_different_fruits_in_a_day_l50_50527


namespace correct_statement_b_l50_50406

open Set 

variables {Point Line Plane : Type}
variable (m n : Line)
variable (α : Plane)
variable (perpendicular_to_plane : Line → Plane → Prop) 
variable (parallel_to_plane : Line → Plane → Prop)
variable (is_subline_of_plane : Line → Plane → Prop)
variable (perpendicular_to_line : Line → Line → Prop)

theorem correct_statement_b (hm : perpendicular_to_plane m α) (hn : is_subline_of_plane n α) : perpendicular_to_line m n :=
sorry

end correct_statement_b_l50_50406


namespace complement_of_N_is_135_l50_50110

-- Define the universal set M and subset N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 4}

-- Prove that the complement of N in M is {1, 3, 5}
theorem complement_of_N_is_135 : M \ N = {1, 3, 5} := 
by
  sorry

end complement_of_N_is_135_l50_50110


namespace isosceles_right_triangle_area_l50_50961

theorem isosceles_right_triangle_area (p : ℝ) : 
  ∃ (A : ℝ), A = (3 - 2 * Real.sqrt 2) * p^2 
  → (∃ (x : ℝ), 2 * x + x * Real.sqrt 2 = 2 * p ∧ A = 1 / 2 * x^2) := 
sorry

end isosceles_right_triangle_area_l50_50961


namespace find_p_from_conditions_l50_50251

variable (p : ℝ) (y x : ℝ)

noncomputable def parabola_eq : Prop := y^2 = 2 * p * x

noncomputable def p_positive : Prop := p > 0

noncomputable def point_on_parabola : Prop := parabola_eq p 1 (p / 4)

theorem find_p_from_conditions (hp : p_positive p) (hpp : point_on_parabola p) : p = Real.sqrt 2 :=
by 
  -- The actual proof goes here
  sorry

end find_p_from_conditions_l50_50251


namespace evaluate_nested_fraction_l50_50726

-- We start by defining the complex nested fraction
def nested_fraction : Rat :=
  1 / (3 - (1 / (3 - (1 / (3 - 1 / 3)))))

-- We assert that the value of the nested fraction is 8/21 
theorem evaluate_nested_fraction : nested_fraction = 8 / 21 := by
  sorry

end evaluate_nested_fraction_l50_50726


namespace largest_divisor_of_consecutive_odd_product_l50_50492

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h_even : n % 2 = 0) (h_pos : n > 0) :
  315 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) := 
sorry

end largest_divisor_of_consecutive_odd_product_l50_50492


namespace same_terminal_side_l50_50702

theorem same_terminal_side : ∃ k : ℤ, 36 + k * 360 = -324 :=
by
  use -1
  linarith

end same_terminal_side_l50_50702


namespace no_infinite_arithmetic_progression_divisible_l50_50778

-- Definitions based on the given condition
def is_arithmetic_progression (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a n = a 0 + n * (a 1 - a 0)

def product_divisible_by_sum (a : ℕ → ℕ) (n : ℕ) : Prop :=
(a n * a (n+1) * a (n+2) * a (n+3) * a (n+4) * a (n+5) * a (n+6) * a (n+7) * a (n+8) * a (n+9)) %
(a n + a (n+1) + a (n+2) + a (n+3) + a (n+4) + a (n+5) + a (n+6) + a (n+7) + a (n+8) + a (n+9)) = 0

-- Final statement to be proven
theorem no_infinite_arithmetic_progression_divisible :
  ¬ ∃ (a : ℕ → ℕ), is_arithmetic_progression a ∧ ∀ n : ℕ, product_divisible_by_sum a n := 
sorry

end no_infinite_arithmetic_progression_divisible_l50_50778


namespace work_rate_calculate_l50_50042

theorem work_rate_calculate (A_time B_time C_time total_time: ℕ) 
  (hA : A_time = 4) 
  (hB : B_time = 8)
  (hTotal : total_time = 2) : 
  C_time = 8 :=
by
  sorry

end work_rate_calculate_l50_50042


namespace second_investment_rate_l50_50819

theorem second_investment_rate (P : ℝ) (r₁ t : ℝ) (I_diff : ℝ) (P900 : P = 900) (r1_4_percent : r₁ = 0.04) (t7 : t = 7) (I_years : I_diff = 31.50) :
∃ r₂ : ℝ, 900 * (r₂ / 100) * 7 - 900 * 0.04 * 7 = 31.50 → r₂ = 4.5 := 
by
  sorry

end second_investment_rate_l50_50819


namespace solve_equation_l50_50833

theorem solve_equation :
  (3 * x - 6 = abs (-21 + 8 - 3)) → x = 22 / 3 :=
by
  intro h
  sorry

end solve_equation_l50_50833


namespace perfect_square_pattern_l50_50731

theorem perfect_square_pattern {a b n : ℕ} (h₁ : 1000 ≤ n ∧ n ≤ 9999)
  (h₂ : n = (10 * a + b) ^ 2)
  (h₃ : n = 1100 * a + 11 * b) : n = 7744 :=
  sorry

end perfect_square_pattern_l50_50731


namespace bank_deposit_exceeds_1000_on_saturday_l50_50786

theorem bank_deposit_exceeds_1000_on_saturday:
  ∃ n: ℕ, (2 * (3^n - 1) / 2 > 1000) ∧ ((n + 1) % 7 = 0) := by
  sorry

end bank_deposit_exceeds_1000_on_saturday_l50_50786


namespace cooper_needs_1043_bricks_l50_50553

def wall1_length := 15
def wall1_height := 6
def wall1_depth := 3

def wall2_length := 20
def wall2_height := 4
def wall2_depth := 2

def wall3_length := 25
def wall3_height := 5
def wall3_depth := 3

def wall4_length := 17
def wall4_height := 7
def wall4_depth := 2

def bricks_needed_for_wall (length height depth: Nat) : Nat :=
  length * height * depth

def total_bricks_needed : Nat :=
  bricks_needed_for_wall wall1_length wall1_height wall1_depth +
  bricks_needed_for_wall wall2_length wall2_height wall2_depth +
  bricks_needed_for_wall wall3_length wall3_height wall3_depth +
  bricks_needed_for_wall wall4_length wall4_height wall4_depth

theorem cooper_needs_1043_bricks : total_bricks_needed = 1043 := by
  sorry

end cooper_needs_1043_bricks_l50_50553


namespace factorize_expression_l50_50085

-- Define that a and b are arbitrary real numbers
variables (a b : ℝ)

-- The theorem statement claiming that 3a²b - 12b equals the factored form 3b(a + 2)(a - 2)
theorem factorize_expression : 3 * a^2 * b - 12 * b = 3 * b * (a + 2) * (a - 2) :=
by
  sorry  -- proof omitted

end factorize_expression_l50_50085


namespace sarah_speeding_tickets_l50_50712

def total_tickets (mark_speeding sarah_speeding mark_parking sarah_parking : ℕ) : ℕ :=
  mark_speeding + mark_parking + sarah_speeding + sarah_parking

theorem sarah_speeding_tickets :
  ∃ (sarah_speeding : ℕ), 
    let mark_speeding := sarah_speeding in
    let mark_parking := 8 in
    let sarah_parking := 4 in
    (2 * mark_parking + 2 * mark_speeding = 24) ∧
    mark_parking = 2 * sarah_parking ∧
    mark_speeding = sarah_speeding ∧
    total_tickets mark_speeding sarah_speeding mark_parking sarah_parking = 24 ∧
    sarah_speeding = 6 := sorry

end sarah_speeding_tickets_l50_50712


namespace find_m_l50_50752

-- Definitions from conditions
def ellipse_eq (x y : ℝ) (m : ℝ) : Prop := x^2 + m * y^2 = 1
def major_axis_twice_minor_axis (a b : ℝ) : Prop := a = 2 * b

-- Main statement
theorem find_m (m : ℝ) (h1 : ellipse_eq 0 0 m) (h2 : 0 < m) (h3 : 0 < m ∧ m < 1) :
  m = 1 / 4 :=
by
  sorry

end find_m_l50_50752


namespace geometric_sequence_a4_l50_50445

theorem geometric_sequence_a4 :
    ∀ (a : ℕ → ℝ) (n : ℕ), 
    a 1 = 2 → 
    (∀ n : ℕ, a (n + 1) = 3 * a n) → 
    a 4 = 54 :=
by
  sorry

end geometric_sequence_a4_l50_50445


namespace total_passengers_l50_50600

theorem total_passengers (P : ℕ) 
  (h1 : P = (1/12 : ℚ) * P + (1/4 : ℚ) * P + (1/9 : ℚ) * P + (1/6 : ℚ) * P + 42) :
  P = 108 :=
sorry

end total_passengers_l50_50600


namespace max_true_statements_at_most_three_l50_50453

open Real

theorem max_true_statements_at_most_three (x : ℝ) : 
  let S1 := (0 < x^3) ∧ (x^3 < 1)
  let S2 := (x^2 > 1)
  let S3 := (-1 < x) ∧ (x < 0)
  let S4 := (0 < x) ∧ (x < 1)
  let S5 := (0 < x - x^3) ∧ (x - x^3 < 1)
  ∃ n ≤ 3, (S1 → n = n + 1) ∧ (S2 → n = n + 1) ∧ (S3 → n = n + 1) ∧ (S4 → n = n + 1) ∧ (S5 → n = n + 1) := sorry

end max_true_statements_at_most_three_l50_50453


namespace selling_price_is_correct_l50_50337

-- Definitions of the given conditions

def cost_of_string_per_bracelet := 1
def cost_of_beads_per_bracelet := 3
def number_of_bracelets_sold := 25
def total_profit := 50

def cost_of_bracelet := cost_of_string_per_bracelet + cost_of_beads_per_bracelet
def total_cost := cost_of_bracelet * number_of_bracelets_sold
def total_revenue := total_profit + total_cost
def selling_price_per_bracelet := total_revenue / number_of_bracelets_sold

-- Target theorem
theorem selling_price_is_correct : selling_price_per_bracelet = 6 :=
  by
  sorry

end selling_price_is_correct_l50_50337


namespace no_common_points_l50_50898

theorem no_common_points (x0 y0 : ℝ) (h : x0^2 < 4 * y0) :
  ∀ (x y : ℝ), (x^2 = 4 * y) → (x0 * x = 2 * (y + y0)) →
  false := 
by
  sorry

end no_common_points_l50_50898


namespace intersection_A_B_l50_50452

-- Define the sets A and the function f
def A : Set ℤ := {-2, 0, 2}
def f (x : ℤ) : ℤ := |x|

-- Define the set B as the image of A under the function f
def B : Set ℤ := {b | ∃ a ∈ A, f a = b}

-- State the property that every element in B has a pre-image in A
axiom B_has_preimage : ∀ b ∈ B, ∃ a ∈ A, f a = b

-- The theorem we want to prove
theorem intersection_A_B : A ∩ B = {0, 2} :=
by sorry

end intersection_A_B_l50_50452


namespace total_area_correct_l50_50276

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase : ℕ := 2
def num_extra_rooms_same_size : ℕ := 3
def num_double_size_rooms : ℕ := 1

def increased_length : ℕ := initial_length + increase
def increased_width : ℕ := initial_width + increase

def area_of_one_room (length width : ℕ) : ℕ := length * width

def number_of_rooms : ℕ := 1 + num_extra_rooms_same_size

def total_area_same_size_rooms : ℕ := number_of_rooms * area_of_one_room increased_length increased_width

def total_area_extra_size_room : ℕ := num_double_size_rooms * 2 * area_of_one_room increased_length increased_width

def total_area : ℕ := total_area_same_size_rooms + total_area_extra_size_room

theorem total_area_correct : total_area = 1800 := by
  -- Proof omitted
  sorry

end total_area_correct_l50_50276


namespace negation_proposition_l50_50311

theorem negation_proposition (x : ℝ) :
  ¬(∀ x : ℝ, x^2 - x + 3 > 0) ↔ ∃ x : ℝ, x^2 - x + 3 ≤ 0 := 
by { sorry }

end negation_proposition_l50_50311


namespace number_from_first_group_is_6_l50_50028

-- Defining conditions
def num_students : Nat := 160
def sample_size : Nat := 20
def groups := List.range' 0 num_students (num_students / sample_size)

def num_from_group_16 (x : Nat) : Nat := 8 * 15 + x
def drawn_number_from_16 : Nat := 126

-- Main theorem
theorem number_from_first_group_is_6 : ∃ x : Nat, num_from_group_16 x = drawn_number_from_16 ∧ x = 6 := 
by
  sorry

end number_from_first_group_is_6_l50_50028


namespace maximize_area_l50_50230

-- Define the variables and constants
variables {x y p : ℝ}

-- Define the conditions
def perimeter (x y p : ℝ) := (2 * x + 2 * y = p)
def area (x y : ℝ) := x * y

-- The theorem statement with conditions
theorem maximize_area (h : perimeter x y p) : x = y → x = p / 4 :=
by
  sorry

end maximize_area_l50_50230


namespace find_number_l50_50210

theorem find_number (x : ℕ) (h1 : x - 13 = 31) : x + 11 = 55 :=
  sorry

end find_number_l50_50210


namespace man_salary_problem_l50_50353

-- Define the problem in Lean 4
theorem man_salary_problem (S : ℝ) :
  (1/3 * S) + (1/4 * S) + (1/5 * S) + 1760 = S → 
  S = 8123.08 :=
sorry

end man_salary_problem_l50_50353


namespace cows_total_l50_50852

theorem cows_total {n : ℕ} :
  (n / 3) + (n / 6) + (n / 8) + (n / 24) + 15 = n ↔ n = 45 :=
by {
  sorry
}

end cows_total_l50_50852


namespace james_age_when_john_turned_35_l50_50615

theorem james_age_when_john_turned_35 :
  ∀ (J : ℕ) (Tim : ℕ) (John : ℕ),
  (Tim = 5) →
  (Tim + 5 = 2 * John) →
  (Tim = 79) →
  (John = 35) →
  (J = John) →
  J = 35 :=
by
  intros J Tim John h1 h2 h3 h4 h5
  rw [h4] at h5
  exact h5

end james_age_when_john_turned_35_l50_50615


namespace problem_statement_l50_50748

theorem problem_statement (n : ℕ) (h : ∀ (a b : ℕ), ¬ (n ∣ (2^a * 3^b + 1))) :
  ∀ (c d : ℕ), ¬ (n ∣ (2^c + 3^d)) := by
  sorry

end problem_statement_l50_50748


namespace find_constants_l50_50742

theorem find_constants (c d : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
     (r^3 + c*r^2 + 17*r + 10 = 0) ∧ (s^3 + c*s^2 + 17*s + 10 = 0) ∧
     (r^3 + d*r^2 + 22*r + 14 = 0) ∧ (s^3 + d*s^2 + 22*s + 14 = 0)) →
  (c = 8 ∧ d = 9) :=
by
  sorry

end find_constants_l50_50742


namespace find_f_neg_one_l50_50689

open Real

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * sin x + b * tan x + 3

theorem find_f_neg_one (a b : ℝ) (h : f a b 1 = 1) : f a b (-1) = 5 :=
by
  sorry

end find_f_neg_one_l50_50689


namespace line_through_fixed_point_l50_50762

-- Define the arithmetic sequence condition
def arithmetic_sequence (k b : ℝ) : Prop :=
  k + b = -2

-- Define the line passing through a fixed point
def line_passes_through (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ (x = 1 ∧ y = -2)

-- The theorem stating the main problem
theorem line_through_fixed_point (k b : ℝ) (h : arithmetic_sequence k b) : line_passes_through k b :=
  sorry

end line_through_fixed_point_l50_50762


namespace coeff_x4y3_in_expansion_l50_50130

/-- In the expansion of (2 * x + y) * (x + 2 * y) ^ 6, the coefficient of the term x^4 * y^3 is 380. -/
theorem coeff_x4y3_in_expansion :
  let poly := (2 * x + y) * (x + 2 * y) ^ 6 in
  coeff poly (4, 3) = 380 :=
begin
  sorry
end

end coeff_x4y3_in_expansion_l50_50130


namespace area_triangle_ABC_l50_50609

theorem area_triangle_ABC (x y : ℝ) (h : x * y ≠ 0) (hAOB : 1 / 2 * |x * y| = 4) : 
  1 / 2 * |(x * (-2 * y) + x * (2 * y) + (-x) * (2 * y))| = 8 :=
by
  sorry

end area_triangle_ABC_l50_50609


namespace geometric_sequence_first_term_l50_50318

theorem geometric_sequence_first_term (a1 q : ℝ) 
  (h1 : (a1 * (1 - q^4)) / (1 - q) = 240)
  (h2 : a1 * q + a1 * q^3 = 180) : 
  a1 = 6 :=
by
  sorry

end geometric_sequence_first_term_l50_50318


namespace mike_spent_on_car_parts_l50_50568

-- Define the costs as constants
def cost_speakers : ℝ := 118.54
def cost_tires : ℝ := 106.33
def cost_cds : ℝ := 4.58

-- Define the total cost of car parts excluding the CDs
def total_cost_car_parts : ℝ := cost_speakers + cost_tires

-- The theorem we want to prove
theorem mike_spent_on_car_parts :
  total_cost_car_parts = 224.87 := 
by 
  -- Proof omitted
  sorry

end mike_spent_on_car_parts_l50_50568


namespace correct_choice_is_C_l50_50525

def is_opposite_number (a b : ℤ) : Prop := a + b = 0

def option_A : Prop := ¬is_opposite_number (2^3) (3^2)
def option_B : Prop := ¬is_opposite_number (-2) (-|-2|)
def option_C : Prop := is_opposite_number ((-3)^2) (-3^2)
def option_D : Prop := ¬is_opposite_number 2 (-(-2))

theorem correct_choice_is_C : option_C ∧ option_A ∧ option_B ∧ option_D :=
by
  sorry

end correct_choice_is_C_l50_50525


namespace least_subtracted_divisible_by_5_l50_50839

theorem least_subtracted_divisible_by_5 :
  ∃ n : ℕ, (568219 - n) % 5 = 0 ∧ n ≤ 4 ∧ (∀ m : ℕ, m < 4 → (568219 - m) % 5 ≠ 0) :=
sorry

end least_subtracted_divisible_by_5_l50_50839


namespace minimum_species_l50_50440

theorem minimum_species (n : ℕ) (h : n = 2021) 
  (even_separation : ∀ (a b : ℕ), a ≠ b → (a ≠ b) → (a % 2 = 0)) : 
  ∃ (s : ℕ), s = 1011 :=
by
  sorry

end minimum_species_l50_50440


namespace minimum_species_count_l50_50432

theorem minimum_species_count {n : ℕ} (h_n : n = 2021) 
  (h_cond : ∀ i j k : ℕ, i < j ∧ j < k → 
    birds i = birds k → birds j ≠ birds i → (j - i - 1) % 2 = 1 ∧ (k - j - 1) % 2 = 1) : 
  ∃ s : ℕ, s ≥ 1011 :=
begin
  sorry
end

end minimum_species_count_l50_50432


namespace square_of_cube_of_third_smallest_prime_l50_50828

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

end square_of_cube_of_third_smallest_prime_l50_50828


namespace smallest_x_l50_50595

theorem smallest_x (y : ℤ) (h1 : 0.9 = (y : ℚ) / (151 + x)) (h2 : 0 < x) (h3 : 0 < y) : x = 9 :=
sorry

end smallest_x_l50_50595


namespace purchase_price_l50_50957

theorem purchase_price (marked_price : ℝ) (discount_rate profit_rate x : ℝ)
  (h1 : marked_price = 126)
  (h2 : discount_rate = 0.05)
  (h3 : profit_rate = 0.05)
  (h4 : marked_price * (1 - discount_rate) - x = x * profit_rate) : 
  x = 114 :=
by 
  sorry

end purchase_price_l50_50957


namespace chocolate_bar_cost_l50_50138

theorem chocolate_bar_cost :
  ∀ (total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips : ℕ),
  total = 150 →
  gummy_bear_cost = 2 →
  chocolate_chip_cost = 5 →
  num_chocolate_bars = 10 →
  num_gummy_bears = 10 →
  num_chocolate_chips = 20 →
  ((total - (num_gummy_bears * gummy_bear_cost + num_chocolate_chips * chocolate_chip_cost)) / num_chocolate_bars = 3) := 
by
  intros total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips 
  intros htotal hgb_cost hcc_cost hncb hngb hncc
  sorry

end chocolate_bar_cost_l50_50138


namespace count_two_digit_integers_sum_seven_l50_50569

theorem count_two_digit_integers_sum_seven : 
  ∃ n : ℕ, (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a + b = 7 → n = 7) := 
by
  sorry

end count_two_digit_integers_sum_seven_l50_50569


namespace mass_of_barium_sulfate_l50_50082

-- Definitions of the chemical equation and molar masses
def barium_molar_mass : ℝ := 137.327
def sulfur_molar_mass : ℝ := 32.065
def oxygen_molar_mass : ℝ := 15.999
def molar_mass_BaSO4 : ℝ := barium_molar_mass + sulfur_molar_mass + 4 * oxygen_molar_mass

-- Given conditions
def moles_BaBr2 : ℝ := 4
def moles_BaSO4_produced : ℝ := moles_BaBr2 -- from balanced equation

-- Calculate mass of BaSO4 produced
def mass_BaSO4 : ℝ := moles_BaSO4_produced * molar_mass_BaSO4

-- Mass of Barium sulfate produced
theorem mass_of_barium_sulfate : mass_BaSO4 = 933.552 :=
by 
  -- Skip the proof
  sorry

end mass_of_barium_sulfate_l50_50082


namespace total_age_of_siblings_l50_50950

def age_total (Susan Arthur Tom Bob : ℕ) : ℕ := Susan + Arthur + Tom + Bob

theorem total_age_of_siblings :
  ∀ (Susan Arthur Tom Bob : ℕ),
    (Arthur = Susan + 2) →
    (Tom = Bob - 3) →
    (Bob = 11) →
    (Susan = 15) →
    age_total Susan Arthur Tom Bob = 51 :=
by
  intros Susan Arthur Tom Bob h1 h2 h3 h4
  rw [h4, h1, h3, h2]    -- Use the conditions
  norm_num               -- Simplify numerical expressions
  sorry                  -- Placeholder for the proof

end total_age_of_siblings_l50_50950


namespace cricket_average_increase_l50_50512

theorem cricket_average_increase :
  ∀ (x : ℝ), (11 * (33 + x) = 407) → (x = 4) :=
  by 
  intros x hx
  sorry

end cricket_average_increase_l50_50512


namespace vector_BC_coordinates_l50_50919

-- Define the given vectors
def vec_AB : ℝ × ℝ := (2, -1)
def vec_AC : ℝ × ℝ := (-4, 1)

-- Define the vector subtraction
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Define the vector BC as the result of the subtraction
def vec_BC : ℝ × ℝ := vec_sub vec_AC vec_AB

-- State the theorem
theorem vector_BC_coordinates : vec_BC = (-6, 2) := by
  sorry

end vector_BC_coordinates_l50_50919


namespace slower_speed_is_correct_l50_50861

/-- 
A person walks at 14 km/hr instead of a slower speed, 
and as a result, he would have walked 20 km more. 
The actual distance travelled by him is 50 km. 
What is the slower speed he usually walks at?
-/
theorem slower_speed_is_correct :
    ∃ x : ℝ, (14 * (50 / 14) - (x * (30 / x))) = 20 ∧ x = 8.4 :=
by
  sorry

end slower_speed_is_correct_l50_50861


namespace binom_18_6_eq_4765_l50_50547

def binom (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_18_6_eq_4765 : binom 18 6 = 4765 := by
  sorry

end binom_18_6_eq_4765_l50_50547


namespace new_volume_l50_50062

theorem new_volume (l w h : ℝ) 
  (h1 : l * w * h = 4320)
  (h2 : l * w + w * h + l * h = 852)
  (h3 : l + w + h = 52) :
  (l + 4) * (w + 4) * (h + 4) = 8624 := sorry

end new_volume_l50_50062


namespace student_correct_answers_l50_50700

theorem student_correct_answers (C I : ℕ) (h₁ : C + I = 100) (h₂ : C - 2 * I = 61) : C = 87 :=
sorry

end student_correct_answers_l50_50700


namespace sufficient_condition_for_one_positive_and_one_negative_root_l50_50814

theorem sufficient_condition_for_one_positive_and_one_negative_root (a : ℝ) (h₀ : a ≠ 0) :
  a < -1 ↔ (∃ x y : ℝ, (a * x^2 + 2 * x + 1 = 0) ∧ (a * y^2 + 2 * y + 1 = 0) ∧ x > 0 ∧ y < 0) :=
by {
  sorry
}

end sufficient_condition_for_one_positive_and_one_negative_root_l50_50814


namespace loaned_out_books_is_50_l50_50061

-- Define the conditions
def initial_books : ℕ := 75
def end_books : ℕ := 60
def percent_returned : ℝ := 0.70

-- Define the variable to represent the number of books loaned out
noncomputable def loaned_out_books := (15:ℝ) / (1 - percent_returned)

-- The target theorem statement we need to prove
theorem loaned_out_books_is_50 : loaned_out_books = 50 :=
by
  sorry

end loaned_out_books_is_50_l50_50061


namespace find_point_B_l50_50608

structure Point where
  x : Int
  y : Int

def translation (p : Point) (dx dy : Int) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem find_point_B :
  let A := Point.mk (-2) 3
  let A' := Point.mk 3 2
  let B' := Point.mk 4 0
  let dx := 5
  let dy := -1
  (translation A dx dy = A') →
  ∃ B : Point, translation B dx dy = B' ∧ B = Point.mk (-1) (-1) :=
by
  intros
  use Point.mk (-1) (-1)
  constructor
  sorry
  rfl

end find_point_B_l50_50608


namespace min_f_over_f_prime_at_1_l50_50630

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def quadratic_derivative (a b x : ℝ) : ℝ := 2 * a * x + b

theorem min_f_over_f_prime_at_1 (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b > 0) (h₂ : ∀ x, quadratic_function a b c x ≥ 0) :
  (∃ k, (∀ x, quadratic_function a b c x ≥ 0 → quadratic_function a b c ((-b)/(2*a)) ≤ x) ∧ k = 2) :=
by
  sorry

end min_f_over_f_prime_at_1_l50_50630


namespace pradeep_maximum_marks_l50_50799

theorem pradeep_maximum_marks (M : ℝ) (h1 : 0.20 * M = 185) : M = 925 :=
by
  sorry

end pradeep_maximum_marks_l50_50799


namespace chord_length_cube_l50_50176

noncomputable def diameter : ℝ := 1
noncomputable def AC (a : ℝ) : ℝ := a
noncomputable def AD (b : ℝ) : ℝ := b
noncomputable def AE (a b : ℝ) : ℝ := (a^2 + b^2).sqrt / 2
noncomputable def AF (b : ℝ) : ℝ := b^2

theorem chord_length_cube (a b : ℝ) (h : AE a b = b^2) : a = b^3 :=
by
  sorry

end chord_length_cube_l50_50176


namespace digit_is_two_l50_50900

theorem digit_is_two (d : ℕ) (h : d < 10) : (∃ k : ℤ, d - 2 = 11 * k) ↔ d = 2 := 
by sorry

end digit_is_two_l50_50900


namespace fraction_to_decimal_l50_50990

theorem fraction_to_decimal : (3 : ℚ) / 80 = 0.0375 :=
by
  sorry

end fraction_to_decimal_l50_50990


namespace find_a_l50_50081

-- Definition of * in terms of 2a - b^2
def custom_mul (a b : ℤ) := 2 * a - b^2

-- The proof statement
theorem find_a (a : ℤ) : custom_mul a 3 = 3 → a = 6 :=
by
  sorry

end find_a_l50_50081


namespace intersection_of_lines_l50_50229

theorem intersection_of_lines
    (x y : ℚ) 
    (h1 : y = 3 * x - 1)
    (h2 : y + 4 = -6 * x) :
    x = -1 / 3 ∧ y = -2 := 
sorry

end intersection_of_lines_l50_50229


namespace randy_biscuits_l50_50466

theorem randy_biscuits (initial_biscuits father_gift mother_gift brother_ate : ℕ) : 
  (initial_biscuits = 32) →
  (father_gift = 13) →
  (mother_gift = 15) →
  (brother_ate = 20) →
  initial_biscuits + father_gift + mother_gift - brother_ate = 40 := by
  sorry

end randy_biscuits_l50_50466


namespace monotonicity_f_a_eq_1_domain_condition_inequality_condition_l50_50583

noncomputable def f (x a : ℝ) := (Real.log (x^2 - 2 * x + a)) / (x - 1)

theorem monotonicity_f_a_eq_1 :
  ∀ x : ℝ, 1 < x → 
  (f x 1 < f (e + 1) 1 → 
   ∀ y, 1 < y ∧ y < e + 1 → f y 1 < f (e + 1) 1) ∧ 
  (f (e + 1) 1 < f x 1 → 
   ∀ z, e + 1 < z → f (e + 1) 1 < f z 1) :=
sorry

theorem domain_condition (a : ℝ) :
  (∀ x : ℝ, (x < 1 ∨ x > 1) → x^2 - 2 * x + a > 0) ↔ a ≥ 1 :=
sorry

theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f x a < (x - 1) * Real.exp x)) ↔ (1 + 1 / Real.exp 1 ≤ a ∧ a ≤ 2) :=
sorry

end monotonicity_f_a_eq_1_domain_condition_inequality_condition_l50_50583


namespace equal_semi_circles_radius_l50_50003

-- Define the segments and semicircles given in the problem as conditions.
def segment1 : ℝ := 12
def segment2 : ℝ := 22
def segment3 : ℝ := 22
def segment4 : ℝ := 16
def segment5 : ℝ := 22

def total_horizontal_path1 (r : ℝ) : ℝ := 2*r + segment1 + 2*r + segment1 + 2*r
def total_horizontal_path2 (r : ℝ) : ℝ := segment2 + 2*r + segment4 + 2*r + segment5

-- The theorem that proves the radius is 18.
theorem equal_semi_circles_radius : ∃ r : ℝ, total_horizontal_path1 r = total_horizontal_path2 r ∧ r = 18 := by
  use 18
  simp [total_horizontal_path1, total_horizontal_path2, segment1, segment2, segment3, segment4, segment5]
  sorry

end equal_semi_circles_radius_l50_50003


namespace lottery_probability_l50_50427

-- Definitions of the conditions
def MegaBallCount : ℕ := 30
def WinnerBallCount : ℕ := 50
def WinningCombinationCount : ℕ := Nat.choose 50 6

-- Main theorem
theorem lottery_probability :
  (1 / MegaBallCount : ℚ) * (1 / WinningCombinationCount : ℚ) = 1 / 477621000 := by
  -- Computation of the binomial coefficient
  have binom_50_6 : Nat.choose 50 6 = 15890700 := by
    rw [Nat.choose_eq_factorial_div_factorial]
    norm_num
  sorry

end lottery_probability_l50_50427


namespace isosceles_triangle_side_length_l50_50010

theorem isosceles_triangle_side_length (P : ℕ := 53) (base : ℕ := 11) (x : ℕ)
  (h1 : x + x + base = P) : x = 21 :=
by {
  -- The proof goes here.
  sorry
}

end isosceles_triangle_side_length_l50_50010


namespace num_distinct_remainders_of_prime_squared_mod_120_l50_50367

theorem num_distinct_remainders_of_prime_squared_mod_120:
  ∀ p : ℕ, Prime p → p > 5 → (p^2 % 120 = 1 ∨ p^2 % 120 = 49) := 
sorry

end num_distinct_remainders_of_prime_squared_mod_120_l50_50367


namespace train_length_approx_500_l50_50064

noncomputable def length_of_train (speed_km_per_hr : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600
  speed_m_per_s * time_sec

theorem train_length_approx_500 :
  length_of_train 120 15 = 500 :=
by
  sorry

end train_length_approx_500_l50_50064


namespace problem_statement_l50_50627

theorem problem_statement (x y z : ℝ) (hx : x + y + z = 2) (hxy : xy + xz + yz = -9) (hxyz : xyz = 1) :
  (yz / x) + (xz / y) + (xy / z) = 77 := sorry

end problem_statement_l50_50627


namespace average_price_of_towels_l50_50359

theorem average_price_of_towels :
  let total_cost := 2350
  let total_towels := 10
  total_cost / total_towels = 235 :=
by
  sorry

end average_price_of_towels_l50_50359


namespace farmer_total_acres_l50_50052

-- Define the problem conditions and the total acres
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) : 5 * x + 2 * x + 4 * x = 1034 :=
by
  -- Proof is omitted
  sorry

end farmer_total_acres_l50_50052


namespace no_burial_needed_for_survivors_l50_50804

def isSurvivor (p : Person) : Bool := sorry
def isBuried (p : Person) : Bool := sorry
variable (p : Person) (accident : Bool)

theorem no_burial_needed_for_survivors (h : accident = true) (hsurvive : isSurvivor p = true) : isBuried p = false :=
sorry

end no_burial_needed_for_survivors_l50_50804


namespace arithmetic_expression_l50_50397

theorem arithmetic_expression :
  4 * 6 * 8 + 24 / 4 - 2^3 = 190 := by
  sorry

end arithmetic_expression_l50_50397


namespace books_added_after_lunch_l50_50111

-- Definitions for the given conditions
def initial_books : Int := 100
def books_borrowed_lunch : Int := 50
def books_remaining_lunch : Int := initial_books - books_borrowed_lunch
def books_borrowed_evening : Int := 30
def books_remaining_evening : Int := 60

-- Let X be the number of books added after lunchtime
variable (X : Int)

-- The proof goal in Lean statement
theorem books_added_after_lunch (h : books_remaining_lunch + X - books_borrowed_evening = books_remaining_evening) :
  X = 40 := by
  sorry

end books_added_after_lunch_l50_50111


namespace largest_three_digit_product_l50_50181

theorem largest_three_digit_product : 
    ∃ (n : ℕ), 
    (n = 336) ∧ 
    (n > 99 ∧ n < 1000) ∧ 
    (∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ n = x * y * (5 * x + 2 * y) ∧ 
        ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ k * m = (5 * x + 2 * y)) :=
by
  sorry

end largest_three_digit_product_l50_50181


namespace eval_ceil_floor_sum_l50_50381

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end eval_ceil_floor_sum_l50_50381


namespace clock_ticks_six_times_l50_50221

-- Define the conditions
def time_between_ticks (ticks : Nat) : Nat :=
  ticks - 1

def interval_duration (total_time : Nat) (ticks : Nat) : Nat :=
  total_time / time_between_ticks ticks

def number_of_ticks (total_time : Nat) (interval_time : Nat) : Nat :=
  total_time / interval_time + 1

-- Given conditions
def specific_time_intervals : Nat := 30
def eight_oclock_intervals : Nat := 42

-- Proven result
theorem clock_ticks_six_times : number_of_ticks specific_time_intervals (interval_duration eight_oclock_intervals 8) = 6 := 
sorry

end clock_ticks_six_times_l50_50221


namespace Maria_ate_2_cookies_l50_50458

theorem Maria_ate_2_cookies : 
  ∀ (initial_cookies given_to_friend given_to_family remaining_after_eating : ℕ),
  initial_cookies = 19 →
  given_to_friend = 5 →
  given_to_family = (initial_cookies - given_to_friend) / 2 →
  remaining_after_eating = initial_cookies - given_to_friend - given_to_family - 2 →
  remaining_after_eating = 5 →
  2 = 2 := by
  intros
  sorry

end Maria_ate_2_cookies_l50_50458


namespace union_M_N_is_R_l50_50252

open Set

/-- Define the sets M and N -/
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x < 3}

/-- Main goal: prove M ∪ N = ℝ -/
theorem union_M_N_is_R : M ∪ N = univ :=
by
  sorry

end union_M_N_is_R_l50_50252


namespace largest_possible_perimeter_l50_50066

theorem largest_possible_perimeter (x : ℕ) (h1 : 1 < x) (h2 : x < 11) : 
    5 + 6 + x ≤ 21 := 
  sorry

end largest_possible_perimeter_l50_50066


namespace find_inverse_l50_50563

noncomputable def inverse_matrix_2x2 (a b c d : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  if ad_bc : (a * d - b * c) = 0 then (0, 0, 0, 0)
  else (d / (a * d - b * c), -b / (a * d - b * c), -c / (a * d - b * c), a / (a * d - b * c))

theorem find_inverse :
  inverse_matrix_2x2 5 7 2 3 = (3, -7, -2, 5) :=
by 
  sorry

end find_inverse_l50_50563


namespace apples_remaining_l50_50285

variable (initial_apples : ℕ)
variable (picked_day1 : ℕ)
variable (picked_day2 : ℕ)
variable (picked_day3 : ℕ)

-- Given conditions
def condition1 : initial_apples = 200 := sorry
def condition2 : picked_day1 = initial_apples / 5 := sorry
def condition3 : picked_day2 = 2 * picked_day1 := sorry
def condition4 : picked_day3 = picked_day1 + 20 := sorry

-- Prove the total number of apples remaining is 20
theorem apples_remaining (H1 : initial_apples = 200) 
  (H2 : picked_day1 = initial_apples / 5) 
  (H3 : picked_day2 = 2 * picked_day1)
  (H4 : picked_day3 = picked_day1 + 20) : 
  initial_apples - (picked_day1 + picked_day2 + picked_day3) = 20 := 
by
  sorry

end apples_remaining_l50_50285


namespace cost_of_flute_l50_50927

def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := 8.89
def song_book_cost : ℝ := 7
def flute_cost : ℝ := 142.46

theorem cost_of_flute :
  total_spent - (music_stand_cost + song_book_cost) = flute_cost :=
by
  sorry

end cost_of_flute_l50_50927


namespace greatest_3digit_base8_divisible_by_7_l50_50670

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), n = 0b777 ∧ (base8_to_base10 0b777) % 7 = 0 ∧ ∀ m < 0o777, m % 7 = 0 → base8_to_base10 m < base8_to_base10 0b777 :=
by
  sorry

end greatest_3digit_base8_divisible_by_7_l50_50670


namespace net_change_in_price_net_change_percentage_l50_50126

theorem net_change_in_price (P : ℝ) :
  0.80 * P * 1.55 - P = 0.24 * P :=
by sorry

theorem net_change_percentage (P : ℝ) :
  ((0.80 * P * 1.55 - P) / P) * 100 = 24 :=
by sorry


end net_change_in_price_net_change_percentage_l50_50126


namespace inequality_proof_l50_50122

theorem inequality_proof (a b c : ℝ) (h : a > b) : a / (c ^ 2 + 1) > b / (c ^ 2 + 1) :=
by
  sorry

end inequality_proof_l50_50122


namespace best_fitting_model_l50_50428

/-- Four models with different coefficients of determination -/
def model1_R2 : ℝ := 0.98
def model2_R2 : ℝ := 0.80
def model3_R2 : ℝ := 0.50
def model4_R2 : ℝ := 0.25

/-- Prove that Model 1 has the best fitting effect among the given models -/
theorem best_fitting_model :
  model1_R2 > model2_R2 ∧ model1_R2 > model3_R2 ∧ model1_R2 > model4_R2 :=
by {sorry}

end best_fitting_model_l50_50428


namespace domain_of_f_l50_50878

noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 6 * x + 10⌋

theorem domain_of_f : {x : ℝ | ∀ y, f y ≠ 0 → x ≠ 3} = {x : ℝ | x < 3 ∨ x > 3} :=
by
  sorry

end domain_of_f_l50_50878


namespace average_of_eight_twelve_and_N_is_12_l50_50661

theorem average_of_eight_twelve_and_N_is_12 (N : ℝ) (hN : 11 < N ∧ N < 19) : (8 + 12 + N) / 3 = 12 :=
by
  -- Place the complete proof step here
  sorry

end average_of_eight_twelve_and_N_is_12_l50_50661


namespace f_at_neg_2_l50_50482

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + x^2 + b * x + 2

-- Given the condition
def f_at_2_eq_3 (a b : ℝ) : Prop := f 2 a b = 3

-- Prove the value of f(-2)
theorem f_at_neg_2 (a b : ℝ) (h : f_at_2_eq_3 a b) : f (-2) a b = 1 :=
sorry

end f_at_neg_2_l50_50482


namespace point_on_x_axis_l50_50243

theorem point_on_x_axis (a : ℝ) (h₁ : 1 - a = 0) : (3 * a - 6, 1 - a) = (-3, 0) :=
by
  sorry

end point_on_x_axis_l50_50243


namespace expression_divisible_by_16_l50_50161

theorem expression_divisible_by_16 (m n : ℤ) : 
  ∃ k : ℤ, (5 * m + 3 * n + 1)^5 * (3 * m + n + 4)^4 = 16 * k :=
sorry

end expression_divisible_by_16_l50_50161


namespace unique_integer_solution_l50_50114

theorem unique_integer_solution :
  ∃! (z : ℤ), 5 * z ≤ 2 * z - 8 ∧ -3 * z ≥ 18 ∧ 7 * z ≤ -3 * z - 21 :=
by
  sorry

end unique_integer_solution_l50_50114


namespace monotonic_decreasing_condition_l50_50586

theorem monotonic_decreasing_condition {f : ℝ → ℝ} (a : ℝ) :
  (∀ x ∈ Ioo (0:ℝ) 1, (2:ℝ) ^ (x * (x - a)) < (2:ℝ) ^ ((1:ℝ) * (1 - a))) → a ≥ 2 :=
begin
  sorry
end

end monotonic_decreasing_condition_l50_50586


namespace maximum_daily_sales_l50_50012

def price (t : ℕ) : ℝ :=
if (0 < t ∧ t < 25) then t + 20
else if (25 ≤ t ∧ t ≤ 30) then -t + 100
else 0

def sales_volume (t : ℕ) : ℝ :=
if (0 < t ∧ t ≤ 30) then -t + 40
else 0

def daily_sales (t : ℕ) : ℝ :=
if (0 < t ∧ t < 25) then (t + 20) * (-t + 40)
else if (25 ≤ t ∧ t ≤ 30) then (-t + 100) * (-t + 40)
else 0

theorem maximum_daily_sales : ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_sales t = 1125 :=
sorry

end maximum_daily_sales_l50_50012


namespace equal_divided_value_l50_50623

def n : ℕ := 8^2022

theorem equal_divided_value : n / 4 = 4^3032 := 
by {
  -- We state the equivalence and details used in the proof.
  sorry
}

end equal_divided_value_l50_50623


namespace necessary_but_not_sufficient_condition_l50_50578

-- Let p be the proposition |x| < 2
def p (x : ℝ) : Prop := abs x < 2

-- Let q be the proposition x^2 - x - 2 < 0
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

-- The proof statement
theorem necessary_but_not_sufficient_condition (x : ℝ) : q x → p x ∧ ¬ (p x → q x) := 
sorry

end necessary_but_not_sufficient_condition_l50_50578


namespace zachary_more_crunches_than_pushups_l50_50034

def zachary_pushups : ℕ := 46
def zachary_crunches : ℕ := 58
def zachary_crunches_more_than_pushups : ℕ := zachary_crunches - zachary_pushups

theorem zachary_more_crunches_than_pushups : zachary_crunches_more_than_pushups = 12 := by
  sorry

end zachary_more_crunches_than_pushups_l50_50034


namespace compare_binary_digits_l50_50375

def numDigits_base2 (n : ℕ) : ℕ :=
  (Nat.log2 n) + 1

theorem compare_binary_digits :
  numDigits_base2 1600 - numDigits_base2 400 = 2 := by
  sorry

end compare_binary_digits_l50_50375


namespace bread_slices_per_loaf_l50_50519

theorem bread_slices_per_loaf (friends: ℕ) (total_loaves : ℕ) (slices_per_friend: ℕ) (total_slices: ℕ)
  (h1 : friends = 10) (h2 : total_loaves = 4) (h3 : slices_per_friend = 6) (h4 : total_slices = friends * slices_per_friend):
  total_slices / total_loaves = 15 :=
by
  sorry

end bread_slices_per_loaf_l50_50519


namespace lara_harvest_raspberries_l50_50787

-- Define measurements of the garden
def length : ℕ := 10
def width : ℕ := 7

-- Define planting and harvesting constants
def plants_per_sq_ft : ℕ := 5
def raspberries_per_plant : ℕ := 12

-- Calculate expected number of raspberries
theorem lara_harvest_raspberries :  length * width * plants_per_sq_ft * raspberries_per_plant = 4200 := 
by sorry

end lara_harvest_raspberries_l50_50787


namespace branch_fraction_remaining_l50_50616

theorem branch_fraction_remaining :
  let original_length := (3 : ℚ) in
  let third := original_length / 3 in
  let fifth := original_length / 5 in
  let removed_length := third + fifth in
  let remaining_length := original_length - removed_length in
  let remaining_fraction := remaining_length / original_length in
  remaining_fraction = (7 / 15) :=
by
  sorry

end branch_fraction_remaining_l50_50616


namespace sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0_l50_50195

theorem sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0 :
  (9^25 + 11^25) % 100 = 0 := 
sorry

end sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0_l50_50195


namespace inequality_not_hold_l50_50420

theorem inequality_not_hold (x y : ℝ) (h : x > y) : ¬ (1 - x > 1 - y) :=
by
  -- condition and given statements
  sorry

end inequality_not_hold_l50_50420


namespace train_crosses_pole_in_2point4_seconds_l50_50777

noncomputable def time_to_cross (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  length / (speed_kmh * (5/18))

theorem train_crosses_pole_in_2point4_seconds :
  time_to_cross 120 180 = 2.4 := by
  sorry

end train_crosses_pole_in_2point4_seconds_l50_50777


namespace added_number_is_five_l50_50215

def original_number := 19
def final_resultant := 129
def doubling_expression (x : ℕ) (y : ℕ) := 3 * (2 * x + y)

theorem added_number_is_five:
  ∃ y, doubling_expression original_number y = final_resultant ↔ y = 5 :=
sorry

end added_number_is_five_l50_50215


namespace apples_used_l50_50218

def initial_apples : ℕ := 43
def apples_left : ℕ := 2

theorem apples_used : initial_apples - apples_left = 41 :=
by sorry

end apples_used_l50_50218


namespace compute_expression_l50_50077

theorem compute_expression : 7^2 - 5 * 6 + 6^2 = 55 := by
  sorry

end compute_expression_l50_50077


namespace container_capacity_l50_50859

-- Definitions based on the conditions
def tablespoons_per_cup := 3
def ounces_per_cup := 8
def tablespoons_added := 15

-- Problem statement
theorem container_capacity : 
  (tablespoons_added / tablespoons_per_cup) * ounces_per_cup = 40 :=
  sorry

end container_capacity_l50_50859


namespace larger_number_is_23_l50_50821

theorem larger_number_is_23 (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 6) : a = 23 := 
by
  sorry

end larger_number_is_23_l50_50821


namespace inequality_problem_l50_50261

theorem inequality_problem
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  (b^2 / a + a^2 / b) ≥ (a + b) :=
sorry

end inequality_problem_l50_50261


namespace correct_calculation_l50_50835

-- Definition of the expressions in the problem
def exprA (a : ℝ) : Prop := 2 * a^2 + a^3 = 3 * a^5
def exprB (x y : ℝ) : Prop := ((-3 * x^2 * y)^2 / (x * y) = 9 * x^5 * y^3)
def exprC (b : ℝ) : Prop := (2 * b^2)^3 = 8 * b^6
def exprD (x : ℝ) : Prop := (2 * x * 3 * x^5 = 6 * x^5)

-- The proof problem
theorem correct_calculation (a x y b : ℝ) : exprC b ∧ ¬ exprA a ∧ ¬ exprB x y ∧ ¬ exprD x :=
by {
  sorry
}

end correct_calculation_l50_50835


namespace final_l50_50628

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ∈ [-3, -2] then 4 * x
  else sorry

lemma f_periodic (h : ∀ x : ℝ, f (x + 3) = - (1 / f x)) :
 ∀ x : ℝ, f (x + 6) = f x :=
sorry

lemma f_even (h : ∀ x : ℝ, f x = f (-x)) : ℕ := sorry

theorem final (h1 : ∀ x : ℝ, f (x + 3) = - (1 / f x))
  (h2 : ∀ x : ℝ, f x = f (-x))
  (h3 : ∀ x : ℝ, x ∈ [-3, -2] → f x = 4 * x) :
  f 107.5 = 1 / 10 :=
sorry

end final_l50_50628


namespace stuffed_animals_total_l50_50336

variable (x y z : ℕ)

theorem stuffed_animals_total :
  let initial := x
  let after_mom := initial + y
  let after_dad := z * after_mom
  let total := after_mom + after_dad
  total = (x + y) * (1 + z) := 
  by 
    let initial := x
    let after_mom := initial + y
    let after_dad := z * after_mom
    let total := after_mom + after_dad
    sorry

end stuffed_animals_total_l50_50336


namespace domain_of_function_l50_50883

noncomputable def domain_is_valid (x z : ℝ) : Prop :=
  1 < x ∧ x < 2 ∧ (|x| - z) ≠ 0

theorem domain_of_function (x z : ℝ) : domain_is_valid x z :=
by
  sorry

end domain_of_function_l50_50883


namespace base_8_to_base_10_4652_l50_50080

def convert_base_8_to_base_10 (n : ℕ) : ℕ :=
  (4 * 8^3) + (6 * 8^2) + (5 * 8^1) + (2 * 8^0)

theorem base_8_to_base_10_4652 :
  convert_base_8_to_base_10 4652 = 2474 :=
by
  -- Skipping the proof steps
  sorry

end base_8_to_base_10_4652_l50_50080


namespace walking_running_ratio_l50_50355

theorem walking_running_ratio (d_w d_r : ℝ) (h1 : d_w / 4 + d_r / 8 = 3) (h2 : d_w + d_r = 16) :
  d_w / d_r = 1 := by
  sorry

end walking_running_ratio_l50_50355


namespace set_P_equals_set_interval_l50_50790

def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x <= 1 ∨ x >= 3}
def P : Set ℝ := {x | x ∈ A ∧ ¬ (x ∈ A ∧ x ∈ B)}

theorem set_P_equals_set_interval :
  P = {x | 1 < x ∧ x < 3} :=
sorry

end set_P_equals_set_interval_l50_50790


namespace fraction_identity_l50_50876

theorem fraction_identity :
  (1721^2 - 1714^2 : ℚ) / (1728^2 - 1707^2) = 1 / 3 :=
by
  sorry

end fraction_identity_l50_50876


namespace solve_for_m_l50_50624

def z1 := Complex.mk 3 2
def z2 (m : ℝ) := Complex.mk 1 m

theorem solve_for_m (m : ℝ) (h : (z1 * z2 m).re = 0) : m = 3 / 2 :=
by
  sorry

end solve_for_m_l50_50624


namespace aria_spent_on_cookies_in_march_l50_50090

/-- Aria purchased 4 cookies each day for the entire month of March,
    each cookie costs 19 dollars, and March has 31 days.
    Prove that the total amount Aria spent on cookies in March is 2356 dollars. -/
theorem aria_spent_on_cookies_in_march :
  (4 * 31) * 19 = 2356 := 
by 
  sorry

end aria_spent_on_cookies_in_march_l50_50090


namespace initial_speed_solution_l50_50068

def initial_speed_problem : Prop :=
  ∃ V : ℝ, 
    (∀ t t_new : ℝ, 
      t = 300 / V ∧ 
      t_new = t - 4 / 5 ∧ 
      (∀ d d_remaining : ℝ, 
        d = V * (5 / 4) ∧ 
        d_remaining = 300 - d ∧ 
        t_new = (5 / 4) + d_remaining / (V + 16)) 
    ) → 
    V = 60

theorem initial_speed_solution : initial_speed_problem :=
by
  unfold initial_speed_problem
  sorry

end initial_speed_solution_l50_50068


namespace train_times_comparison_l50_50018

-- Defining the given conditions
variables (V1 T1 T2 D : ℝ)
variables (h1 : T1 = 2) (h2 : T2 = 7/3)
variables (train1_speed : V1 = D / T1)
variables (train2_speed : V2 = (3/5) * V1)

-- The proof statement to show that T2 is 1/3 hour longer than T1
theorem train_times_comparison 
  (h1 : (6/7) * V1 = D / (T1 + 1/3))
  (h2 : (3/5) * V1 = D / (T2 + 1)) :
  T2 - T1 = 1/3 :=
sorry

end train_times_comparison_l50_50018


namespace reduced_price_l50_50036

noncomputable def reduced_price_per_dozen (P : ℝ) : ℝ := 12 * (P / 2)

theorem reduced_price (X P : ℝ) (h1 : X * P = 50) (h2 : (X + 50) * (P / 2) = 50) : reduced_price_per_dozen P = 6 :=
sorry

end reduced_price_l50_50036


namespace inequality_solution_l50_50651

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4 / 3 ∨ -3 / 2 < x := 
sorry

end inequality_solution_l50_50651


namespace smallest_integer_ending_in_9_divisible_by_11_is_99_l50_50191

noncomputable def smallest_positive_integer_ending_in_9_and_divisible_by_11 : ℕ :=
  99

theorem smallest_integer_ending_in_9_divisible_by_11_is_99 :
  ∃ n : ℕ, n > 0 ∧ (n % 10 = 9) ∧ (n % 11 = 0) ∧
          (∀ m : ℕ, m > 0 → (m % 10 = 9) → (m % 11 = 0) → n ≤ m) :=
begin
  use smallest_positive_integer_ending_in_9_and_divisible_by_11,
  split,
  { -- n > 0
    exact nat.zero_lt_bit1 nat.zero_lt_one },
  split,
  { -- n % 10 = 9
    exact nat.mod_eq_of_lt (by norm_num) },
  split,
  { -- n % 11 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 99) },
  { -- ∀ m > 0, m % 10 = 9, m % 11 = 0 → n ≤ m
    intros m hm1 hm2 hm3,
    change 99 ≤ m,
    -- m % 99 = 0 → 99 ≤ m since 99 > 0
    sorry
  }
end

end smallest_integer_ending_in_9_divisible_by_11_is_99_l50_50191


namespace total_potato_weight_l50_50696

theorem total_potato_weight (bags_morning : ℕ) (bags_afternoon : ℕ) (weight_per_bag : ℕ) :
  bags_morning = 29 → 
  bags_afternoon = 17 → 
  weight_per_bag = 7 → 
  (bags_morning + bags_afternoon) * weight_per_bag = 322 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  norm_num 
  sorry

end total_potato_weight_l50_50696


namespace minimum_species_l50_50439

theorem minimum_species (n : ℕ) (h : n = 2021) 
  (even_separation : ∀ (a b : ℕ), a ≠ b → (a ≠ b) → (a % 2 = 0)) : 
  ∃ (s : ℕ), s = 1011 :=
by
  sorry

end minimum_species_l50_50439


namespace arccos_neg_half_eq_two_pi_over_three_l50_50877

theorem arccos_neg_half_eq_two_pi_over_three :
  Real.arccos (-1/2) = 2 * Real.pi / 3 := sorry

end arccos_neg_half_eq_two_pi_over_three_l50_50877


namespace right_triangle_counterexample_l50_50686

def is_acute_angle (α : ℝ) : Prop := 0 < α ∧ α < 90

def is_right_angle (α : ℝ) : Prop := α = 90

def is_triangle (α β γ : ℝ) : Prop := α + β + γ = 180

def is_acute_triangle (α β γ : ℝ) : Prop := is_acute_angle α ∧ is_acute_angle β ∧ is_acute_angle γ

def is_right_triangle (α β γ : ℝ) : Prop := 
  (is_right_angle α ∧ is_acute_angle β ∧ is_acute_angle γ) ∨ 
  (is_acute_angle α ∧ is_right_angle β ∧ is_acute_angle γ) ∨ 
  (is_acute_angle α ∧ is_acute_angle β ∧ is_right_angle γ)

theorem right_triangle_counterexample (α β γ : ℝ) : 
  is_triangle α β γ → is_right_triangle α β γ → ¬ is_acute_triangle α β γ :=
by
  intro htri hrt hacute
  sorry

end right_triangle_counterexample_l50_50686


namespace multiply_203_197_square_neg_699_l50_50972

theorem multiply_203_197 : 203 * 197 = 39991 := by
  sorry

theorem square_neg_699 : (-69.9)^2 = 4886.01 := by
  sorry

end multiply_203_197_square_neg_699_l50_50972


namespace magic_square_sum_l50_50922

theorem magic_square_sum (v w x y z : ℤ)
    (h1 : 25 + z + 23 = 25 + x + w)
    (h2 : 18 + x + y = 25 + x + w)
    (h3 : v + 22 + w = 25 + x + w)
    (h4 : 25 + 18 + v = 25 + x + w)
    (h5 : z + x + 22 = 25 + x + w)
    (h6 : 23 + y + w = 25 + x + w)
    (h7 : 25 + x + w = 25 + x + w)
    (h8 : v + x + 23 = 25 + x + w) 
:
    y + z = 45 :=
by
  sorry

end magic_square_sum_l50_50922


namespace radius_of_circle_l50_50971

theorem radius_of_circle
  (AC BD : ℝ) (h_perpendicular : AC * BD = 0)
  (h_intersect_center : AC / 2 = BD / 2)
  (AB : ℝ) (h_AB : AB = 3)
  (CD : ℝ) (h_CD : CD = 4) :
  (∃ R : ℝ, R = 5 / 2) :=
by
  sorry

end radius_of_circle_l50_50971


namespace intersection_times_l50_50987

def distance_squared_motorcyclist (t: ℝ) : ℝ := (72 * t)^2
def distance_squared_bicyclist (t: ℝ) : ℝ := (36 * (t - 1))^2
def law_of_cosines (t: ℝ) : ℝ := distance_squared_motorcyclist t +
                                      distance_squared_bicyclist t -
                                      2 * 72 * 36 * |t| * |t - 1| * (1/2)

def equation_simplified (t: ℝ) : ℝ := 4 * t^2 + t^2 - 2 * |t| * |t - 1|

theorem intersection_times :
  ∀ t: ℝ, (0 < t ∨ t < 1) → equation_simplified t = 49 → (t = 4 ∨ t = -4) := 
by
  intros t ht_eq
  intro h
  sorry

end intersection_times_l50_50987


namespace ratio_a6_b6_l50_50173

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Define the nth term of sequence a
noncomputable def b_n (n : ℕ) : ℝ := sorry -- Define the nth term of sequence b
noncomputable def S_n (n : ℕ) : ℝ := sorry -- Define the sum of the first n terms of sequence a
noncomputable def T_n (n : ℕ) : ℝ := sorry -- Define the sum of the first n terms of sequence b

axiom condition (n : ℕ) : S_n n / T_n n = (2 * n) / (3 * n + 1)

theorem ratio_a6_b6 : a_n 6 / b_n 6 = 11 / 17 :=
by
  sorry

end ratio_a6_b6_l50_50173


namespace points_distance_within_rectangle_l50_50842

theorem points_distance_within_rectangle :
  ∀ (points : Fin 6 → (ℝ × ℝ)), (∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ 3 ∧ 0 ≤ (points i).2 ∧ (points i).2 ≤ 4) →
  ∃ (i j : Fin 6), i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 2 :=
by
  sorry

end points_distance_within_rectangle_l50_50842


namespace mark_reads_1750_pages_per_week_l50_50156

def initialReadingHoursPerDay := 2
def increasePercentage := 150
def initialPagesPerDay := 100

def readingHoursPerDayAfterIncrease : Nat := initialReadingHoursPerDay + (initialReadingHoursPerDay * increasePercentage) / 100
def readingSpeedPerHour := initialPagesPerDay / initialReadingHoursPerDay
def pagesPerDayNow := readingHoursPerDayAfterIncrease * readingSpeedPerHour
def pagesPerWeekNow : Nat := pagesPerDayNow * 7

theorem mark_reads_1750_pages_per_week :
  pagesPerWeekNow = 1750 :=
sorry -- Proof omitted

end mark_reads_1750_pages_per_week_l50_50156


namespace smallest_n_exists_l50_50499

theorem smallest_n_exists :
  ∃ (a1 a2 a3 a4 a5 : ℤ), a1 + a2 + a3 + a4 + a5 = 1990 ∧ a1 * a2 * a3 * a4 * a5 = 1990 :=
sorry

end smallest_n_exists_l50_50499


namespace probability_multiple_of_4_is_2_over_5_l50_50597

-- Definitions from the conditions
def cards := {1, 2, 3, 4, 5, 6}

-- A function to determine if the product of two numbers is a multiple of 4
def is_multiple_of_4 (a b : ℕ) : Prop :=
  (a * b) % 4 = 0

-- Combinations of drawing 2 cards without replacement from a set of 6 cards
def pairs := { (a, b) : ℕ × ℕ | a ∈ cards ∧ b ∈ cards ∧ a < b }

-- Counting the total number of possible pairs
def total_pairs := pairs.size

-- Counting the favorable pairs where the product is a multiple of 4
def favorable_pairs := { p ∈ pairs | is_multiple_of_4 p.1 p.2 }.size

-- Probability calculation: favorable pairs divided by total pairs
noncomputable def probability := (favorable_pairs.toRat) / (total_pairs.toRat)

-- The main theorem stating the desired probability
theorem probability_multiple_of_4_is_2_over_5 : probability = 2 / 5 :=
  sorry

end probability_multiple_of_4_is_2_over_5_l50_50597


namespace length_AC_eq_sqrt6_l50_50268

-- Definitions for the conditions
variables {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (O1 O2 : Type*) [MetricSpace O1] [MetricSpace O2]
variables (R1 R2 : ℝ)
variable (AC : ℝ)

-- Constants given in the problem
constant angle_B_eq_pi_six : ∀ (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C], 
  ∀ (α β : ℝ), β = π / 6 → α + β = π - β → α = π - (π - β)

constant radius_circle1 : R1 = 2
constant radius_circle2 : R2 = 3

open Metric

-- Theorem to prove
theorem length_AC_eq_sqrt6 {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C]
   (O1 O2 : Type*) [MetricSpace O1] [MetricSpace O2] 
    (R1 R2 : ℝ) (AC : ℝ)
  [h1 : radius_circle1 R1]
  [h2 : radius_circle2 R2]
  (h3 : angle_B_eq_pi_six A B C (AC:ℝ) (π / 6)) : 
  AC = sqrt 6 := 
by sorry

end length_AC_eq_sqrt6_l50_50268


namespace cost_of_each_item_l50_50817

theorem cost_of_each_item 
  (x y z : ℝ) 
  (h1 : 3 * x + 5 * y + z = 32)
  (h2 : 4 * x + 7 * y + z = 40) : 
  x + y + z = 16 :=
by 
  sorry

end cost_of_each_item_l50_50817


namespace carnations_percentage_l50_50040

variables {α : Type*} [field α]

def fraction_blue (total : α) : α := total * (1/2)
def fraction_red (total : α) : α := total * (1/2)
def fraction_blue_roses (total : α) : α := fraction_blue total * (2/5)
def fraction_blue_carnations (total : α) : α := fraction_blue total * (3/5)
def fraction_red_carnations (total : α) : α := fraction_red total * (2/3)

theorem carnations_percentage (total : α) : 
  (fraction_blue_carnations total + fraction_red_carnations total) / total * 100 = 63 := 
sorry

end carnations_percentage_l50_50040


namespace function_passes_through_point_l50_50158

-- Lean 4 Statement
theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x = 1 ∧ y = 5 ∧ (a^(x-1) + 4) = y :=
by
  use 1
  use 5
  sorry

end function_passes_through_point_l50_50158


namespace calculate_expression_l50_50531

theorem calculate_expression :
  ( (128^2 - 5^2) / (72^2 - 13^2) * ((72 - 13) * (72 + 13)) / ((128 - 5) * (128 + 5)) * (128 + 5) / (72 + 13) )
  = (133 / 85) :=
by
  -- placeholder for the proof
  sorry

end calculate_expression_l50_50531


namespace discounted_price_is_correct_l50_50039

def original_price_of_cork (C : ℝ) : Prop :=
  C + (C + 2.00) = 2.10

def discounted_price_of_cork (C : ℝ) : ℝ :=
  C - (C * 0.12)

theorem discounted_price_is_correct :
  ∃ C : ℝ, original_price_of_cork C ∧ discounted_price_of_cork C = 0.044 :=
by
  sorry

end discounted_price_is_correct_l50_50039


namespace constant_and_middle_term_l50_50720

open scoped BigOperators

-- Define the expression
def expr (x : ℚ) : ℚ := (2 * x^2 - 1 / x)

-- Constants for the binomial expansion
noncomputable def binomial_coeff (n k : ℕ) : ℚ := nat.choose n k

-- The 6th power expansion of the given expression
def expand_expr := ∑ r in finset.range 7, (-1)^r * (2^(6-r)) * binomial_coeff 6 r * (λ x : ℚ, x^(12-3*r))

/-- 
    Prove that the constant term in the binomial expansion of (2x^2 - 1/x)^6 is 60
    and the middle term is -160x^3.
-/
theorem constant_and_middle_term :
    ∃ (x : ℚ), (expr x)^6 = 60 ∧ (expr x)^3 = -160 * x^3 :=
by
  sorry

end constant_and_middle_term_l50_50720


namespace evaluate_f_i_l50_50725

noncomputable def f (x : ℂ) : ℂ :=
  (x^5 + 2 * x^3 + x) / (x + 1)

theorem evaluate_f_i : f (Complex.I) = 0 := 
  sorry

end evaluate_f_i_l50_50725


namespace function_properties_l50_50721

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 3)

theorem function_properties :
  (∃ x : ℝ, f x = -1) = false ∧ 
  (∃ x_0 : ℝ, -1 < x_0 ∧ x_0 < 0 ∧ deriv f x_0 = 0) ∧ 
  (∀ x : ℝ, -3 < x → f x > -1 / 2) ∧ 
  (∃ x_0 : ℝ, -3 < x_0 ∧ ∀ x : ℝ, -3 < x → f x_0 ≤ f x) :=
by
  sorry

end function_properties_l50_50721


namespace ratio_consequent_l50_50773

theorem ratio_consequent (a b x : ℕ) (h_ratio : a = 4) (h_b : b = 6) (h_x : x = 30) :
  (a : ℚ) / b = x / 45 := 
by 
  -- add here the necessary proof steps 
  sorry

end ratio_consequent_l50_50773


namespace end_of_month_books_count_l50_50694

theorem end_of_month_books_count:
  ∀ (initial_books : ℝ) (loaned_out_books : ℝ) (return_rate : ℝ)
    (rounded_loaned_out_books : ℝ) (returned_books : ℝ)
    (not_returned_books : ℝ) (end_of_month_books : ℝ),
    initial_books = 75 →
    loaned_out_books = 60.00000000000001 →
    return_rate = 0.65 →
    rounded_loaned_out_books = 60 →
    returned_books = return_rate * rounded_loaned_out_books →
    not_returned_books = rounded_loaned_out_books - returned_books →
    end_of_month_books = initial_books - not_returned_books →
    end_of_month_books = 54 :=
by
  intros initial_books loaned_out_books return_rate
         rounded_loaned_out_books returned_books
         not_returned_books end_of_month_books
  intros h_initial_books h_loaned_out_books h_return_rate
         h_rounded_loaned_out_books h_returned_books
         h_not_returned_books h_end_of_month_books
  sorry

end end_of_month_books_count_l50_50694


namespace garbage_classification_competition_l50_50264

theorem garbage_classification_competition :
  let boy_rate_seventh := 0.4
  let boy_rate_eighth := 0.5
  let girl_rate_seventh := 0.6
  let girl_rate_eighth := 0.7
  let combined_boy_rate := (boy_rate_seventh + boy_rate_eighth) / 2
  let combined_girl_rate := (girl_rate_seventh + girl_rate_eighth) / 2
  boy_rate_seventh < boy_rate_eighth ∧ combined_boy_rate < combined_girl_rate :=
by {
  sorry
}

end garbage_classification_competition_l50_50264


namespace pyramid_volume_l50_50468

theorem pyramid_volume (VW WX VZ : ℝ) (h1 : VW = 10) (h2 : WX = 5) (h3 : VZ = 8)
  (h_perp1 : ∀ (V W Z : ℝ), V ≠ W → V ≠ Z → Z ≠ W → W = 0 ∧ Z = 0)
  (h_perp2 : ∀ (V W X : ℝ), V ≠ W → V ≠ X → X ≠ W → W = 0 ∧ X = 0) :
  let area_base := VW * WX
  let height := VZ
  let volume := 1 / 3 * area_base * height
  volume = 400 / 3 := by
  sorry

end pyramid_volume_l50_50468


namespace storybook_pages_l50_50072

theorem storybook_pages :
  (10 + 5) / (1 - (1 / 5) * 2) = 25 := by
  sorry

end storybook_pages_l50_50072


namespace negation_proposition_l50_50313

theorem negation_proposition {x : ℝ} : ¬ (x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := sorry

end negation_proposition_l50_50313


namespace range_of_a_l50_50241

noncomputable def piecewise_f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 - 2 * a * x - 2 else x + (36 / x) - 6 * a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, piecewise_f a x ≥ piecewise_f a 2) ↔ (2 ≤ a ∧ a ≤ 5) :=
by
  sorry

end range_of_a_l50_50241


namespace basis_of_R3_l50_50626

def e1 : ℝ × ℝ × ℝ := (1, 0, 0)
def e2 : ℝ × ℝ × ℝ := (0, 1, 0)
def e3 : ℝ × ℝ × ℝ := (0, 0, 1)

theorem basis_of_R3 :
  ∀ (u : ℝ × ℝ × ℝ), ∃ (α β γ : ℝ), u = α • e1 + β • e2 + γ • e3 ∧ 
  (∀ (a b c : ℝ), a • e1 + b • e2 + c • e3 = (0, 0, 0) → a = 0 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end basis_of_R3_l50_50626


namespace distance_of_point_P_to_origin_l50_50577

noncomputable def dist_to_origin (P : ℝ × ℝ) : ℝ :=
  Real.sqrt (P.1 ^ 2 + P.2 ^ 2)

theorem distance_of_point_P_to_origin :
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let y_P := 1 / 2
  ∃ x_P : ℝ, (x_P, y_P) = P ∧
    (dist_to_origin P = Real.sqrt 6 / 2) :=
by
  sorry

end distance_of_point_P_to_origin_l50_50577


namespace average_growth_rate_bing_dwen_dwen_l50_50305

noncomputable def sales_growth_rate (v0 v2 : ℕ) (x : ℝ) : Prop :=
  (1 + x) ^ 2 = (v2 : ℝ) / (v0 : ℝ)

theorem average_growth_rate_bing_dwen_dwen :
  ∀ (v0 v2 : ℕ) (x : ℝ),
    v0 = 10000 →
    v2 = 12100 →
    sales_growth_rate v0 v2 x →
    x = 0.1 :=
by
  intros v0 v2 x h₀ h₂ h_growth
  sorry

end average_growth_rate_bing_dwen_dwen_l50_50305


namespace James_total_area_l50_50272

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase_dimension : ℕ := 2
def number_of_new_rooms : ℕ := 4
def larger_room_multiplier : ℕ := 2

noncomputable def new_length : ℕ := initial_length + increase_dimension
noncomputable def new_width : ℕ := initial_width + increase_dimension
noncomputable def area_of_one_new_room : ℕ := new_length * new_width
noncomputable def total_area_of_4_rooms : ℕ := area_of_one_new_room * number_of_new_rooms
noncomputable def area_of_larger_room : ℕ := area_of_one_new_room * larger_room_multiplier
noncomputable def total_area : ℕ := total_area_of_4_rooms + area_of_larger_room

theorem James_total_area : total_area = 1800 := 
by
  sorry

end James_total_area_l50_50272


namespace shopkeeper_loss_percentage_l50_50851

theorem shopkeeper_loss_percentage
    (CP : ℝ) (profit_rate loss_percent : ℝ) 
    (SP : ℝ := CP * (1 + profit_rate)) 
    (value_after_theft : ℝ := SP * (1 - loss_percent)) 
    (goods_loss : ℝ := 100 * (1 - (value_after_theft / CP))) :
    goods_loss = 51.6 :=
by
    sorry

end shopkeeper_loss_percentage_l50_50851


namespace flute_cost_is_correct_l50_50930

-- Define the conditions
def total_spent : ℝ := 158.35
def stand_cost : ℝ := 8.89
def songbook_cost : ℝ := 7.0

-- Calculate the cost to be subtracted
def accessories_cost : ℝ := stand_cost + songbook_cost

-- Define the target cost of the flute
def flute_cost : ℝ := total_spent - accessories_cost

-- Prove that the flute cost is $142.46
theorem flute_cost_is_correct : flute_cost = 142.46 :=
by
  -- Here we would provide the proof
  sorry

end flute_cost_is_correct_l50_50930


namespace probability_of_triangle_formation_l50_50233

def sticks : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

noncomputable def valid_triangle_combinations (s : List ℕ) : ℕ :=
  (s.erase 2).erase 3.length -- This is simplified, for illustration. Properly computing valid set count is required.

def total_combinations : ℕ := (List.choose sticks 3).length

def probability_triangle : ℚ := valid_triangle_combinations sticks / total_combinations

theorem probability_of_triangle_formation : probability_triangle = 1 / 4 := by
  sorry

end probability_of_triangle_formation_l50_50233


namespace system_of_equations_solution_exists_l50_50843

theorem system_of_equations_solution_exists :
  ∃ (x y : ℚ), (x * y^2 - 2 * y^2 + 3 * x = 18) ∧ (3 * x * y + 5 * x - 6 * y = 24) ∧ 
                ((x = 3 ∧ y = 3) ∨ (x = 75 / 13 ∧ y = -3 / 7)) :=
by
  sorry

end system_of_equations_solution_exists_l50_50843


namespace total_savings_l50_50450

theorem total_savings :
  let josiah_daily := 0.25 
  let josiah_days := 24 
  let leah_daily := 0.50 
  let leah_days := 20 
  let megan_multiplier := 2
  let megan_days := 12 
  let josiah_savings := josiah_daily * josiah_days 
  let leah_savings := leah_daily * leah_days 
  let megan_daily := megan_multiplier * leah_daily 
  let megan_savings := megan_daily * megan_days 
  let total_savings := josiah_savings + leah_savings + megan_savings 
  total_savings = 28 :=
by
  sorry

end total_savings_l50_50450


namespace highest_price_more_than_lowest_l50_50867

-- Define the highest price and lowest price.
def highest_price : ℕ := 350
def lowest_price : ℕ := 250

-- Define the calculation for the percentage increase.
def percentage_increase (hp lp : ℕ) : ℕ :=
  ((hp - lp) * 100) / lp

-- The theorem to prove the required percentage increase.
theorem highest_price_more_than_lowest : percentage_increase highest_price lowest_price = 40 := 
  by sorry

end highest_price_more_than_lowest_l50_50867


namespace stratified_leader_selection_probability_of_mixed_leaders_l50_50805

theorem stratified_leader_selection :
  let num_first_grade := 150
  let num_second_grade := 100
  let total_leaders := 5
  let leaders_first_grade := (total_leaders * num_first_grade) / (num_first_grade + num_second_grade)
  let leaders_second_grade := (total_leaders * num_second_grade) / (num_first_grade + num_second_grade)
  leaders_first_grade = 3 ∧ leaders_second_grade = 2 :=
by
  sorry

theorem probability_of_mixed_leaders :
  let num_first_grade_leaders := 3
  let num_second_grade_leaders := 2
  let total_leaders := 5
  let total_ways := 10
  let favorable_ways := 6
  (favorable_ways / total_ways) = (3 / 5) :=
by
  sorry

end stratified_leader_selection_probability_of_mixed_leaders_l50_50805


namespace ball_placement_at_least_one_in_box1_l50_50942

theorem ball_placement_at_least_one_in_box1 : 
  let balls := { "A", "B", "C" }
  let boxes := { 1, 2, 3, 4 }
  ∃ f : balls → boxes, (∃ b ∈ balls, f b = 1) → 
  finset.card { f // (∃ b ∈ balls, f b = 1) } = 37 := 
sorry

end ball_placement_at_least_one_in_box1_l50_50942


namespace sum_of_three_numbers_eq_16_l50_50342

variable {a b c : ℝ}

theorem sum_of_three_numbers_eq_16
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + c * a = 50) :
  a + b + c = 16 :=
by
  sorry

end sum_of_three_numbers_eq_16_l50_50342


namespace negation_of_universal_l50_50008

theorem negation_of_universal : 
  (¬ (∀ x : ℝ, 2 * x^2 - x + 1 ≥ 0)) ↔ (∃ x : ℝ, 2 * x^2 - x + 1 < 0) :=
by
  sorry

end negation_of_universal_l50_50008


namespace angela_deliveries_l50_50071

theorem angela_deliveries
  (n_meals : ℕ)
  (h_meals : n_meals = 3)
  (n_packages : ℕ)
  (h_packages : n_packages = 8 * n_meals) :
  n_meals + n_packages = 27 := by
  sorry

end angela_deliveries_l50_50071


namespace log_properties_l50_50991

theorem log_properties :
  (Real.log 5) ^ 2 + (Real.log 2) * (Real.log 50) = 1 :=
by sorry

end log_properties_l50_50991


namespace cat_run_time_l50_50871

/-- An electronic cat runs a lap on a circular track with a perimeter of 240 meters.
It runs at a speed of 5 meters per second for the first half of the time and 3 meters per second for the second half of the time.
Prove that the cat takes 36 seconds to run the last 120 meters. -/
theorem cat_run_time
  (perimeter : ℕ)
  (speed1 speed2 : ℕ)
  (half_perimeter : ℕ)
  (half_time : ℕ)
  (last_120m_time : ℕ) :
  perimeter = 240 →
  speed1 = 5 →
  speed2 = 3 →
  half_perimeter = perimeter / 2 →
  half_time = 60 / 2 →
  (5 * half_time - half_perimeter) / speed1 + (half_perimeter - (5 * half_time - half_perimeter)) / speed2 = 36 :=
by sorry

end cat_run_time_l50_50871


namespace ronald_profit_fraction_l50_50165

theorem ronald_profit_fraction:
  let initial_units : ℕ := 200
  let total_investment : ℕ := 3000
  let selling_price_per_unit : ℕ := 20
  let total_selling_price := initial_units * selling_price_per_unit
  let total_profit := total_selling_price - total_investment
  (total_profit : ℚ) / total_investment = (1 : ℚ) / 3 :=
by
  -- here we will put the steps needed to prove the theorem.
  sorry

end ronald_profit_fraction_l50_50165


namespace alcohol_mix_problem_l50_50170

theorem alcohol_mix_problem
  (x_volume : ℕ) (y_volume : ℕ)
  (x_percentage : ℝ) (y_percentage : ℝ)
  (target_percentage : ℝ)
  (x_volume_eq : x_volume = 200)
  (x_percentage_eq : x_percentage = 0.10)
  (y_percentage_eq : y_percentage = 0.30)
  (target_percentage_eq : target_percentage = 0.14)
  (y_solution : ℝ)
  (h : y_volume = 50) :
  (20 + 0.3 * y_solution) / (200 + y_solution) = target_percentage := by sorry

end alcohol_mix_problem_l50_50170


namespace sum_of_numbers_greater_than_or_equal_to_0_1_l50_50320

def num1 : ℝ := 0.8
def num2 : ℝ := 0.5  -- converting 1/2 to 0.5
def num3 : ℝ := 0.6

def is_greater_than_or_equal_to_0_1 (n : ℝ) : Prop :=
  n ≥ 0.1

theorem sum_of_numbers_greater_than_or_equal_to_0_1 :
  is_greater_than_or_equal_to_0_1 num1 ∧ 
  is_greater_than_or_equal_to_0_1 num2 ∧ 
  is_greater_than_or_equal_to_0_1 num3 →
  num1 + num2 + num3 = 1.9 :=
by
  sorry

end sum_of_numbers_greater_than_or_equal_to_0_1_l50_50320


namespace John_walked_miles_to_park_l50_50449

theorem John_walked_miles_to_park :
  ∀ (total_skateboarded_miles skateboarded_first_leg skateboarded_return_leg walked_miles : ℕ),
    total_skateboarded_miles = 24 →
    skateboarded_first_leg = 10 →
    skateboarded_return_leg = 10 →
    total_skateboarded_miles = skateboarded_first_leg + skateboarded_return_leg + walked_miles →
    walked_miles = 4 :=
by
  intros total_skateboarded_miles skateboarded_first_leg skateboarded_return_leg walked_miles
  intro h1 h2 h3 h4
  sorry

end John_walked_miles_to_park_l50_50449


namespace greatest_3_digit_base8_num_div_by_7_eq_511_l50_50673

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end greatest_3_digit_base8_num_div_by_7_eq_511_l50_50673


namespace greatest_div_by_seven_base_eight_l50_50677

theorem greatest_div_by_seven_base_eight : ∃ n : ℕ, 
  (n < 512) ∧ (Divisibility.divides 7 n) ∧ 
  (∀ m : ℕ, (m < 512) → (Divisibility.divides 7 m) → m ≤ n) ∧ 
  nat.to_digits 8 n = [7, 7, 4] := 
sorry

end greatest_div_by_seven_base_eight_l50_50677


namespace find_special_four_digit_square_l50_50729

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ (x : ℕ), x * x = n

def same_first_two_digits (n : ℕ) : Prop := (n / 1000) = (n / 100 % 10)

def same_last_two_digits (n : ℕ) : Prop := (n % 100 / 10) = (n % 10)

theorem find_special_four_digit_square :
  ∃ (n : ℕ), is_four_digit n ∧ is_perfect_square n ∧ same_first_two_digits n ∧ same_last_two_digits n ∧ n = 7744 := 
sorry

end find_special_four_digit_square_l50_50729


namespace arithmetic_sequence_geometric_condition_l50_50407

theorem arithmetic_sequence_geometric_condition 
  (a : ℕ → ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + 3) 
  (h_geom : (a 1 + 6) ^ 2 = a 1 * (a 1 + 9)) : 
  a 2 = -9 :=
sorry

end arithmetic_sequence_geometric_condition_l50_50407


namespace inequality_true_l50_50260

theorem inequality_true (a b : ℝ) (hab : a < b) (hb : b < 0) (ha : a < 0) : (b / a) < 1 :=
by
  sorry

end inequality_true_l50_50260


namespace range_of_k_l50_50753

theorem range_of_k (k : ℝ) : 
  (∀ x, x ∈ {x | -3 ≤ x ∧ x ≤ 2} ∩ {x | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1} ↔ x ∈ {x | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1}) →
   -1 ≤ k ∧ k ≤ 1 / 2 :=
by sorry

end range_of_k_l50_50753


namespace members_even_and_divisible_l50_50266

structure ClubMember (α : Type) := 
  (friend : α) 
  (enemy : α)

def is_even (n : Nat) : Prop :=
  n % 2 = 0

def can_be_divided_into_two_subclubs (members : List (ClubMember Nat)) : Prop :=
sorry -- Definition of dividing into two subclubs here

theorem members_even_and_divisible (members : List (ClubMember Nat)) :
  is_even members.length ∧ can_be_divided_into_two_subclubs members :=
sorry

end members_even_and_divisible_l50_50266


namespace find_f2_l50_50629

-- Define the function f(x) = ax + b
def f (a b x : ℝ) : ℝ := a * x + b

-- Condition: f'(x) = a
def f_derivative (a b x : ℝ) : ℝ := a

-- Given conditions
variables (a b : ℝ)
axiom h1 : f a b 1 = 2
axiom h2 : f_derivative a b 1 = 2

theorem find_f2 : f a b 2 = 4 :=
by
  sorry

end find_f2_l50_50629


namespace find_m_l50_50253

theorem find_m (x y m : ℝ) (h1 : 2 * x + y = 1) (h2 : x + 2 * y = 2) (h3 : x + y = 2 * m - 1) : m = 1 :=
by
  sorry

end find_m_l50_50253


namespace math_class_problem_l50_50265

theorem math_class_problem
  (x a : ℝ)
  (h_mistaken : (2 * (2 * 4 - 1) + 1 = 5 * (4 + a)))
  (h_original : (2 * x - 1) / 5 + 1 = (x + a) / 2)
  : a = -1 ∧ x = 13 := by
  sorry

end math_class_problem_l50_50265


namespace arithmetic_sequence_problem_l50_50443

theorem arithmetic_sequence_problem : 
  ∀ (a : ℕ → ℕ) (d : ℕ), 
  a 1 = 1 →
  (a 3 + a 4 + a 5 + a 6 = 20) →
  a 8 = 9 :=
by
  intros a d h₁ h₂
  -- We skip the proof, leaving a placeholder.
  sorry

end arithmetic_sequence_problem_l50_50443


namespace arithm_prog_diff_max_l50_50009

noncomputable def find_most_common_difference (a b c : Int) : Prop :=
  let d := a - b
  (b = a - d) ∧ (c = a - 2 * d) ∧
  (2 * a * 2 * a - 4 * 2 * a * c ≥ 0) ∧
  (2 * a * 2 * b - 4 * 2 * a * c ≥ 0) ∧
  (2 * b * 2 * b - 4 * 2 * b * c ≥ 0) ∧
  (2 * b * c - 4 * 2 * b * a ≥ 0) ∧
  (c * c - 4 * c * 2 * b ≥ 0) ∧
  ((2 * a * c - 4 * 2 * c * b) ≥ 0)

theorem arithm_prog_diff_max (a b c Dmax: Int) : 
  find_most_common_difference 4 (-1) (-6) ∧ Dmax = -5 :=
by 
  sorry

end arithm_prog_diff_max_l50_50009


namespace power_sum_l50_50759

theorem power_sum (a b c : ℝ) (h1 : a + b + c = 1)
                  (h2 : a^2 + b^2 + c^2 = 3)
                  (h3 : a^3 + b^3 + c^3 = 4)
                  (h4 : a^4 + b^4 + c^4 = 5) :
  a^5 + b^5 + c^5 = 6 :=
  sorry

end power_sum_l50_50759


namespace slope_and_intercept_of_given_function_l50_50665

-- Defining the form of a linear function
def linear_function (m b : ℝ) (x : ℝ) : ℝ := m * x + b

-- The given linear function
def given_function (x : ℝ) : ℝ := 3 * x + 2

-- Stating the problem as a theorem
theorem slope_and_intercept_of_given_function :
  (∀ x : ℝ, given_function x = linear_function 3 2 x) :=
by
  intro x
  sorry

end slope_and_intercept_of_given_function_l50_50665


namespace necessary_but_not_sufficient_for_q_implies_range_of_a_l50_50171

variable (a : ℝ)

def p (x : ℝ) := |4*x - 3| ≤ 1
def q (x : ℝ) := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

theorem necessary_but_not_sufficient_for_q_implies_range_of_a :
  (∀ x : ℝ, q a x → p x) → (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end necessary_but_not_sufficient_for_q_implies_range_of_a_l50_50171


namespace chase_travel_time_l50_50716

-- Definitions of speeds
def chase_speed (C : ℝ) := C
def cameron_speed (C : ℝ) := 2 * C
def danielle_speed (C : ℝ) := 6 * (cameron_speed C)

-- Time taken by Danielle to cover distance
def time_taken_by_danielle (C : ℝ) := 30  
def distance_travelled (C : ℝ) := (time_taken_by_danielle C) * (danielle_speed C)  -- 180C

-- Speeds on specific stretches
def cameron_bike_speed (C : ℝ) := 0.75 * (cameron_speed C)
def chase_scooter_speed (C : ℝ) := 1.25 * (chase_speed C)

-- Prove the time Chase takes to travel the same distance D
theorem chase_travel_time (C : ℝ) : 
  (distance_travelled C) / (chase_speed C) = 180 := sorry

end chase_travel_time_l50_50716


namespace product_is_correct_l50_50201

-- Define the variables and conditions
variables {a b c d : ℚ}

-- State the conditions
def conditions (a b c d : ℚ) :=
  3 * a + 2 * b + 4 * c + 6 * d = 36 ∧
  4 * (d + c) = b ∧
  4 * b + 2 * c = a ∧
  c - 2 = d

-- The theorem statement
theorem product_is_correct (a b c d : ℚ) (h : conditions a b c d) :
  a * b * c * d = -315 / 32 :=
sorry

end product_is_correct_l50_50201


namespace farmer_total_acres_l50_50054

theorem farmer_total_acres (x : ℕ) (H1 : 4 * x = 376) : 
  5 * x + 2 * x + 4 * x = 1034 :=
by
  -- This placeholder is indicating unfinished proof
  sorry

end farmer_total_acres_l50_50054


namespace rikki_poetry_sales_l50_50641

theorem rikki_poetry_sales :
  let words_per_5min := 25
  let total_minutes := 2 * 60
  let intervals := total_minutes / 5
  let total_words := words_per_5min * intervals
  let total_earnings := 6
  let price_per_word := total_earnings / total_words
  price_per_word = 0.01 :=
by
  sorry

end rikki_poetry_sales_l50_50641


namespace sum_first_six_terms_l50_50791

variable (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)

-- Define the existence of a geometric sequence with given properties
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given Condition: a_3 = 2a_4 = 2
def cond1 (a : ℕ → ℝ) : Prop :=
  a 3 = 2 ∧ a 4 = 1

-- Define the sum of the first n terms of the sequence
def geometric_sum (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q)

-- We need to prove that under these conditions, S_6 = 63/4
theorem sum_first_six_terms 
  (hq : q = 1 / 2) 
  (ha : is_geometric_sequence a q) 
  (hcond1 : cond1 a) 
  (hS : geometric_sum a q S) : 
  S 6 = 63 / 4 := 
sorry

end sum_first_six_terms_l50_50791


namespace optimal_optimism_coefficient_l50_50713

theorem optimal_optimism_coefficient (a b : ℝ) (x : ℝ) (h_b_gt_a : b > a) (h_x : 0 < x ∧ x < 1) 
  (h_c : ∀ (c : ℝ), c = a + x * (b - a) → (c - a) * (c - a) = (b - c) * (b - a)) : 
  x = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end optimal_optimism_coefficient_l50_50713


namespace fraction_of_ponies_with_horseshoes_l50_50356

variable (P H : ℕ)
variable (F : ℚ)

theorem fraction_of_ponies_with_horseshoes 
  (h1 : H = P + 3)
  (h2 : P + H = 163)
  (h3 : (5/8 : ℚ) * F * P = 5) :
  F = 1/10 :=
  sorry

end fraction_of_ponies_with_horseshoes_l50_50356


namespace car_circuit_velocity_solution_l50_50692

theorem car_circuit_velocity_solution
    (v_s v_p v_d : ℕ)
    (h1 : v_s < v_p)
    (h2 : v_p < v_d)
    (h3 : s = d)
    (h4 : s + p + d = 600)
    (h5 : (d : ℚ) / v_s + (p : ℚ) / v_p + (d : ℚ) / v_d = 50) :
    (v_s = 7 ∧ v_p = 12 ∧ v_d = 42) ∨
    (v_s = 8 ∧ v_p = 12 ∧ v_d = 24) ∨
    (v_s = 9 ∧ v_p = 12 ∧ v_d = 18) ∨
    (v_s = 10 ∧ v_p = 12 ∧ v_d = 15) :=
by
  sorry

end car_circuit_velocity_solution_l50_50692


namespace ornithological_park_species_l50_50436

/-- In an ornithological park, there are 2021 birds arranged in a row.
Each pair of birds of the same species has an even number of birds between them.
Prove that the smallest number of bird species is 1011. -/
theorem ornithological_park_species (n : ℕ) (h1 : n = 2021) 
  (h2 : ∀ s : ℕ, s ∈ {1..n} → (∀ x y : ℕ, x < y ∧ x ≠ y → (∀ z : ℕ, z ∈ ({x, y} : set ℕ) → even (y - x - 1))) ) 
  : s ≥ 1011 :=
sorry

end ornithological_park_species_l50_50436


namespace modulus_z_eq_one_l50_50911

noncomputable def imaginary_unit : ℂ := Complex.I

noncomputable def z : ℂ := (1 - imaginary_unit) / (1 + imaginary_unit) 

theorem modulus_z_eq_one : Complex.abs z = 1 := 
sorry

end modulus_z_eq_one_l50_50911


namespace count_integers_six_times_sum_of_digits_l50_50116

theorem count_integers_six_times_sum_of_digits (n : ℕ) (h : n < 1000) 
    (digit_sum : ℕ → ℕ)
    (digit_sum_correct : ∀ (n : ℕ), digit_sum n = (n % 10) + ((n / 10) % 10) + (n / 100)) :
    ∃! n, n < 1000 ∧ n = 6 * digit_sum n :=
sorry

end count_integers_six_times_sum_of_digits_l50_50116


namespace find_s_l50_50561

theorem find_s (s : ℝ) (m : ℤ) (d : ℝ) (h_floor : ⌊s⌋ = m) (h_decompose : s = m + d) (h_fractional : 0 ≤ d ∧ d < 1) (h_equation : ⌊s⌋ - s = -10.3) : s = -9.7 :=
by
  sorry

end find_s_l50_50561


namespace range_of_x_satisfying_inequality_l50_50662

theorem range_of_x_satisfying_inequality (x : ℝ) : 
  (|x+1| + |x| < 2) ↔ (-3/2 < x ∧ x < 1/2) :=
by sorry

end range_of_x_satisfying_inequality_l50_50662


namespace product_probability_correct_l50_50637

/-- Define probabilities for spins of Paco and Dani --/
def prob_paco := 1 / 5
def prob_dani := 1 / 15

/-- Define the probability that the product of spins is less than 30 --/
def prob_product_less_than_30 : ℚ :=
  (2 / 5) + (1 / 5) * (9 / 15) + (1 / 5) * (7 / 15) + (1 / 5) * (5 / 15)

theorem product_probability_correct : prob_product_less_than_30 = 17 / 25 :=
by sorry

end product_probability_correct_l50_50637


namespace divisible_by_6_l50_50296

theorem divisible_by_6 {n : ℕ} (h2 : 2 ∣ n) (h3 : 3 ∣ n) : 6 ∣ n :=
sorry

end divisible_by_6_l50_50296


namespace binom_18_6_l50_50535

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_l50_50535


namespace percentage_less_A_than_B_l50_50872

theorem percentage_less_A_than_B :
  ∀ (full_marks A_marks D_marks C_marks B_marks : ℝ),
    full_marks = 500 →
    A_marks = 360 →
    D_marks = 0.80 * full_marks →
    C_marks = (1 - 0.20) * D_marks →
    B_marks = (1 + 0.25) * C_marks →
    ((B_marks - A_marks) / B_marks) * 100 = 10 :=
  by intros full_marks A_marks D_marks C_marks B_marks
     intros h_full h_A h_D h_C h_B
     sorry

end percentage_less_A_than_B_l50_50872


namespace min_questions_to_determine_product_50_numbers_l50_50656

/-- Prove that to uniquely determine the product of 50 numbers each either +1 or -1 
arranged on the circumference of a circle by asking for the product of three 
consecutive numbers, one must ask a minimum of 50 questions. -/
theorem min_questions_to_determine_product_50_numbers : 
  ∀ (a : ℕ → ℤ), (∀ i, a i = 1 ∨ a i = -1) → 
  (∀ i, ∃ b : ℤ, b = a i * a (i+1) * a (i+2)) → 
  ∃ n, n = 50 :=
by
  sorry

end min_questions_to_determine_product_50_numbers_l50_50656


namespace kara_uses_28_cups_of_sugar_l50_50783

theorem kara_uses_28_cups_of_sugar (S W : ℕ) (h1 : S + W = 84) (h2 : S * 2 = W) : S = 28 :=
by sorry

end kara_uses_28_cups_of_sugar_l50_50783


namespace neither_coffee_nor_tea_l50_50369

theorem neither_coffee_nor_tea (total_businesspeople coffee_drinkers tea_drinkers both_drinkers : ℕ) 
    (h_total : total_businesspeople = 35)
    (h_coffee : coffee_drinkers = 18)
    (h_tea : tea_drinkers = 15)
    (h_both : both_drinkers = 6) :
    (total_businesspeople - (coffee_drinkers + tea_drinkers - both_drinkers)) = 8 := 
by
  sorry

end neither_coffee_nor_tea_l50_50369


namespace solution_set_of_inequality_l50_50601

noncomputable def f : ℝ → ℝ := sorry 

axiom f_cond : ∀ x : ℝ, f x + deriv f x > 1
axiom f_at_zero : f 0 = 4

theorem solution_set_of_inequality : {x : ℝ | f x > 3 / Real.exp x + 1} = { x : ℝ | x > 0 } :=
by
  sorry

end solution_set_of_inequality_l50_50601


namespace find_n_l50_50707

theorem find_n 
  (a : ℝ := 9 / 15)
  (S1 : ℝ := 15 / (1 - a))
  (b : ℝ := (9 + n) / 15)
  (S2 : ℝ := 3 * S1)
  (hS1 : S1 = 37.5)
  (hS2 : S2 = 112.5)
  (hb : b = 13 / 15)
  (hn : 13 = 9 + n) : 
  n = 4 :=
by
  sorry

end find_n_l50_50707


namespace quadratic_axis_of_symmetry_is_one_l50_50006

noncomputable def quadratic_axis_of_symmetry (b c : ℝ) : ℝ :=
  (-b / (2 * 1))

theorem quadratic_axis_of_symmetry_is_one
  (b c : ℝ)
  (hA : (0:ℝ)^2 + b * 0 + c = 3)
  (hB : (2:ℝ)^2 + b * 2 + c = 3) :
  quadratic_axis_of_symmetry b c = 1 :=
by
  sorry

end quadratic_axis_of_symmetry_is_one_l50_50006


namespace sequence_and_sum_l50_50792

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 0 + n * (a 1 - a 0)

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

theorem sequence_and_sum
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_first_n_terms a S)
  (cond : a 2 + a 8 = 15 - a 5) :
  S 9 = 45 :=
sorry

end sequence_and_sum_l50_50792


namespace four_digit_perfect_square_l50_50736

theorem four_digit_perfect_square : 
  ∃ (N : ℕ), (1000 ≤ N ∧ N ≤ 9999) ∧ (∃ (a b : ℕ), a = N / 1000 ∧ b = (N % 100) / 10 ∧ a = N / 100 - (N / 100 % 10) ∧ b = (N % 100 / 10) - N % 10) ∧ (∃ (n : ℕ), N = n * n) →
  N = 7744 := 
sorry

end four_digit_perfect_square_l50_50736


namespace candy_distribution_impossible_l50_50416

theorem candy_distribution_impossible :
  ∀ (candies : Fin 6 → ℕ),
  (candies 0 = 0 ∧ candies 1 = 1 ∧ candies 2 = 0 ∧ candies 3 = 0 ∧ candies 4 = 0 ∧ candies 5 = 1) →
  (∀ t, ∃ i, (i < 6) ∧ candies ((i+t)%6) = candies ((i+t+1)%6)) →
  ∃ (i : Fin 6), candies i ≠ candies ((i + 1) % 6) :=
by
  sorry

end candy_distribution_impossible_l50_50416


namespace triangle_area_l50_50654

-- Define the line equation as a condition.
def line_equation (x : ℝ) : ℝ :=
  4 * x + 8

-- Define the y-intercept (condition 1).
def y_intercept := line_equation 0

-- Define the x-intercept (condition 2).
def x_intercept := (-8) / 4

-- Define the area of the triangle given the intercepts and prove it equals 8 (question and correct answer).
theorem triangle_area :
  (1 / 2) * abs x_intercept * y_intercept = 8 :=
by
  sorry

end triangle_area_l50_50654


namespace smallest_int_ending_in_9_divisible_by_11_l50_50194

theorem smallest_int_ending_in_9_divisible_by_11:
  ∃ x : ℕ, (∃ k : ℤ, x = 10 * k + 9) ∧ x % 11 = 0 ∧ x = 99 :=
by
  sorry

end smallest_int_ending_in_9_divisible_by_11_l50_50194


namespace f_decreasing_increasing_find_b_range_l50_50105

-- Define the function f(x) and prove its properties for x > 0 and x < 0
noncomputable def f (x a : ℝ) : ℝ := x + a / x

theorem f_decreasing_increasing (a : ℝ) (h : a > 0):
  (∀ x : ℝ, 0 < x → x ≤ Real.sqrt a → ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 ≤ Real.sqrt a) → f x1 a > f x2 a) ∧ 
  (∀ x : ℝ, 0 < Real.sqrt a → Real.sqrt a ≤ x → ∀ x1 x2 : ℝ, (Real.sqrt a ≤ x1 ∧ x1 < x2) → f x1 a < f x2 a) ∧ 
  (∀ x : ℝ, x < 0 → -Real.sqrt a ≤ x ∧ x < 0 → f x1 a > f x2 a) ∧ 
  (∀ x : ℝ, x < 0 → x < -Real.sqrt a → f x1 a < f x2 a)
:= sorry

-- Define the function h(x) and find the range of b
noncomputable def h (x : ℝ) : ℝ := x + 4 / x - 8
noncomputable def g (x b : ℝ) : ℝ := -x - 2 * b

theorem find_b_range:
  (∀ x1 : ℝ, 1 ≤ x1 ∧ x1 ≤ 3 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 3 ∧ g x2 b = h x1) ↔
  1/2 ≤ b ∧ b ≤ 1
:= sorry

end f_decreasing_increasing_find_b_range_l50_50105


namespace sequence_property_l50_50097

-- Conditions as definitions
def seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 = -(2 / 3)) ∧ (∀ n ≥ 2, S n + (1 / S n) + 2 = a n)

-- The desired property of the sequence
def S_property (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → S n = -((n + 1) / (n + 2))

-- The main theorem
theorem sequence_property (a S : ℕ → ℝ) (h_seq : seq a S) : S_property S := sorry

end sequence_property_l50_50097


namespace necessary_and_sufficient_condition_for_extreme_value_l50_50307

-- Defining the function f(x) = ax^3 + x + 1
def f (a x : ℝ) : ℝ := a * x^3 + x + 1

-- Defining the condition for f to have an extreme value
def has_extreme_value (a : ℝ) : Prop := ∃ x : ℝ, deriv (f a) x = 0

-- Stating the problem
theorem necessary_and_sufficient_condition_for_extreme_value (a : ℝ) :
  has_extreme_value a ↔ a < 0 := by
  sorry

end necessary_and_sufficient_condition_for_extreme_value_l50_50307


namespace tickets_sold_l50_50362

def advanced_purchase_tickets := ℕ
def door_purchase_tickets := ℕ

variable (A D : ℕ)

theorem tickets_sold :
  (A + D = 140) →
  (8 * A + 14 * D = 1720) →
  A = 40 :=
by
  intros h1 h2
  sorry

end tickets_sold_l50_50362


namespace hike_length_l50_50754

-- Definitions of conditions
def initial_water : ℕ := 6
def final_water : ℕ := 1
def hike_duration : ℕ := 2
def leak_rate : ℕ := 1
def last_mile_drunk : ℕ := 1
def first_part_drink_rate : ℚ := 2 / 3

-- Statement to prove
theorem hike_length (hike_duration : ℕ) (initial_water : ℕ) (final_water : ℕ) (leak_rate : ℕ) 
  (last_mile_drunk : ℕ) (first_part_drink_rate : ℚ) : 
  hike_duration = 2 → 
  initial_water = 6 → 
  final_water = 1 → 
  leak_rate = 1 → 
  last_mile_drunk = 1 → 
  first_part_drink_rate = 2 / 3 → 
  ∃ miles : ℕ, miles = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Proof placeholder
  sorry

end hike_length_l50_50754


namespace problem1_problem2_l50_50507

-- Problem 1
theorem problem1 : (2 * Real.sqrt 12 - 3 * Real.sqrt (1 / 3)) * Real.sqrt 6 = 9 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (h1 : x / (2 * x - 1) = 2 - 3 / (1 - 2 * x)) : x = -1 / 3 := by
  sorry

end problem1_problem2_l50_50507


namespace number_of_trousers_given_l50_50634

-- Define the conditions
def shirts_given : Nat := 589
def total_clothing_given : Nat := 934

-- Define the expected answer
def expected_trousers_given : Nat := 345

-- The theorem statement to prove the number of trousers given
theorem number_of_trousers_given : total_clothing_given - shirts_given = expected_trousers_given :=
by
  sorry

end number_of_trousers_given_l50_50634


namespace last_two_digits_of_7_pow_2017_l50_50159

theorem last_two_digits_of_7_pow_2017 :
  (7 ^ 2017) % 100 = 7 :=
sorry

end last_two_digits_of_7_pow_2017_l50_50159


namespace sugar_cups_used_l50_50784

def ratio_sugar_water : ℕ × ℕ := (1, 2)
def total_cups : ℕ := 84

theorem sugar_cups_used (r : ℕ × ℕ) (tc : ℕ) (hsugar : r.1 = 1) (hwater : r.2 = 2) (htotal : tc = 84) :
  (tc * r.1) / (r.1 + r.2) = 28 :=
by
  sorry

end sugar_cups_used_l50_50784


namespace free_throws_count_l50_50315

-- Given conditions:
variables (a b x : ℕ) -- α is an abbreviation for natural numbers

-- Condition: number of points from all shots
axiom points_condition : 2 * a + 3 * b + x = 79
-- Condition: three-point shots are twice the points of two-point shots
axiom three_point_condition : 3 * b = 4 * a
-- Condition: number of free throws is one more than the number of two-point shots
axiom free_throw_condition : x = a + 1

-- Prove that the number of free throws is 12
theorem free_throws_count : x = 12 :=
by {
  sorry
}

end free_throws_count_l50_50315


namespace sally_combinations_l50_50469

theorem sally_combinations :
  let wall_colors := 4
  let flooring_types := 3
  wall_colors * flooring_types = 12 := by
  sorry

end sally_combinations_l50_50469


namespace football_game_attendance_l50_50022

-- Define the initial conditions
def saturday : ℕ := 80
def monday : ℕ := saturday - 20
def wednesday : ℕ := monday + 50
def friday : ℕ := saturday + monday
def total_week_actual : ℕ := saturday + monday + wednesday + friday
def total_week_expected : ℕ := 350

-- Define the proof statement
theorem football_game_attendance : total_week_actual - total_week_expected = 40 :=
by 
  -- Proof steps would go here
  sorry

end football_game_attendance_l50_50022


namespace farmer_total_acres_l50_50051

-- Define the problem conditions and the total acres
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) : 5 * x + 2 * x + 4 * x = 1034 :=
by
  -- Proof is omitted
  sorry

end farmer_total_acres_l50_50051


namespace simplify_expression_l50_50885

variable (b : ℝ) (hb : 0 < b)

theorem simplify_expression : 
  ( ( b ^ (16 / 8) ^ (1 / 4) ) ^ 3 * ( b ^ (16 / 4) ^ (1 / 8) ) ^ 3 ) = b ^ 3 := by
  sorry

end simplify_expression_l50_50885


namespace find_a45_l50_50574

theorem find_a45 :
  ∃ (a : ℕ → ℝ), 
    a 0 = 11 ∧ a 1 = 11 ∧ 
    (∀ m n : ℕ, a (m + n) = (1/2) * (a (2 * m) + a (2 * n)) - (m - n)^2) ∧ 
    a 45 = 1991 := by
  sorry

end find_a45_l50_50574


namespace distance_between_Stockholm_and_Malmoe_l50_50952

noncomputable def actualDistanceGivenMapDistanceAndScale (mapDistance : ℕ) (scale : ℕ) : ℕ :=
  mapDistance * scale

theorem distance_between_Stockholm_and_Malmoe (mapDistance : ℕ) (scale : ℕ) :
  mapDistance = 150 → scale = 20 → actualDistanceGivenMapDistanceAndScale mapDistance scale = 3000 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end distance_between_Stockholm_and_Malmoe_l50_50952


namespace football_match_goals_even_likely_l50_50853

noncomputable def probability_even_goals (p_1 : ℝ) (q_1 : ℝ) : Prop :=
  let p := p_1^2 + q_1^2
  let q := 2 * p_1 * q_1
  p >= q

theorem football_match_goals_even_likely (p_1 : ℝ) (h : p_1 >= 0 ∧ p_1 <= 1) : probability_even_goals p_1 (1 - p_1) :=
by sorry

end football_match_goals_even_likely_l50_50853


namespace only_positive_integer_a_squared_plus_2a_is_perfect_square_l50_50088

/-- Prove that the only positive integer \( a \) for which \( a^2 + 2a \) is a perfect square is \( a = 0 \). -/
theorem only_positive_integer_a_squared_plus_2a_is_perfect_square :
  ∀ (a : ℕ), (∃ (k : ℕ), a^2 + 2*a = k^2) → a = 0 :=
by
  intro a h
  sorry

end only_positive_integer_a_squared_plus_2a_is_perfect_square_l50_50088


namespace find_p_l50_50119

variables (p q : ℚ)
variables (h1 : 2 * p + 5 * q = 10) (h2 : 5 * p + 2 * q = 20)

theorem find_p : p = 80 / 21 :=
by sorry

end find_p_l50_50119


namespace binom_18_6_eq_13260_l50_50550

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end binom_18_6_eq_13260_l50_50550


namespace johanns_path_probability_l50_50518

-- Define the necessary structures and conditions
def interior_lattice_points (n : ℕ) : list (ℕ × ℕ) := 
  (list.range n).bind (λ i, (list.range n).map (prod.mk i))

def is_even (n : ℕ) : Prop := n % 2 = 0

def contains_even_number_of_lattice_points (a b : ℕ) : Prop := 
  is_even (Nat.gcd a (2004 - b))

def probability_even_lattice_points (total : ℕ) (even : ℕ) : ℚ :=
  even / total

noncomputable def probability_johanns_path_even : ℚ :=
  let total_points := 9801
  let even_points := 2401
  probability_even_lattice_points total_points even_points

-- The main theorem to prove
theorem johanns_path_probability :
  probability_johanns_path_even = 3 / 4 :=
by sorry

end johanns_path_probability_l50_50518


namespace horner_method_v3_value_l50_50076

theorem horner_method_v3_value :
  let f (x : ℤ) := 3 * x^6 + 5 * x^5 + 6 * x^4 + 79 * x^3 - 8 * x^2 + 35 * x + 12
  let v : ℤ := 3
  let v1 (x : ℤ) : ℤ := v * x + 5
  let v2 (x : ℤ) (v1x : ℤ) : ℤ := v1x * x + 6
  let v3 (x : ℤ) (v2x : ℤ) : ℤ := v2x * x + 79
  x = -4 →
  v3 x (v2 x (v1 x)) = -57 :=
by
  sorry

end horner_method_v3_value_l50_50076


namespace alicia_tax_cents_per_hour_l50_50524

-- Define Alicia's hourly wage in dollars.
def alicia_hourly_wage_dollars : ℝ := 25
-- Define the conversion rate from dollars to cents.
def cents_per_dollar : ℝ := 100
-- Define the local tax rate as a percentage.
def tax_rate_percent : ℝ := 2

-- Convert Alicia's hourly wage to cents.
def alicia_hourly_wage_cents : ℝ := alicia_hourly_wage_dollars * cents_per_dollar

-- Define the theorem that needs to be proved.
theorem alicia_tax_cents_per_hour : alicia_hourly_wage_cents * (tax_rate_percent / 100) = 50 := by
  sorry

end alicia_tax_cents_per_hour_l50_50524


namespace student_solves_exactly_20_problems_l50_50521

theorem student_solves_exactly_20_problems :
  (∀ n, 1 ≤ (a : ℕ → ℕ) n) ∧ (∀ k, a (k + 7) ≤ a k + 12) ∧ a 77 ≤ 132 →
  ∃ i j, i < j ∧ a j - a i = 20 := sorry

end student_solves_exactly_20_problems_l50_50521


namespace quadratic_roots_l50_50894

noncomputable def solve_quadratic (a b c : ℂ) : list ℂ :=
let discriminant := b^2 - 4 * a * c in
let sqrt_discriminant := complex.sqrt discriminant in
[((-b + sqrt_discriminant) / (2 * a)), ((-b - sqrt_discriminant) / (2 * a))]

theorem quadratic_roots :
  solve_quadratic 1 2 (- (3 - 4 * complex.I)) = [-1 + complex.sqrt 2 - complex.I * complex.sqrt 2, -1 - complex.sqrt 2 + complex.I * complex.sqrt 2] :=
by
  sorry

end quadratic_roots_l50_50894


namespace JamesFlowers_l50_50779

noncomputable def numberOfFlowersJamesPlantedInADay (F : ℝ) := 0.5 * (F + 0.15 * F)

theorem JamesFlowers (F : ℝ) (H₁ : 6 * F + (F + 0.15 * F) = 315) : numberOfFlowersJamesPlantedInADay F = 25.3:=
by
  sorry

end JamesFlowers_l50_50779


namespace algebraic_expression_value_l50_50127

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x + 3 = 7) : 3 * x^2 + 3 * x + 7 = 19 :=
sorry

end algebraic_expression_value_l50_50127


namespace original_number_of_men_l50_50838

theorem original_number_of_men (x : ℕ) (h1 : x * 10 = (x - 5) * 12) : x = 30 :=
by
  sorry

end original_number_of_men_l50_50838


namespace kara_uses_28_cups_of_sugar_l50_50782

theorem kara_uses_28_cups_of_sugar (S W : ℕ) (h1 : S + W = 84) (h2 : S * 2 = W) : S = 28 :=
by sorry

end kara_uses_28_cups_of_sugar_l50_50782


namespace binom_18_6_l50_50534

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_l50_50534


namespace polynomial_bound_l50_50281

noncomputable def P (x : ℝ) : ℝ := sorry  -- Placeholder for the polynomial P(x)

theorem polynomial_bound (n : ℕ) (hP : ∀ y : ℝ, 0 ≤ y ∧ y ≤ 1 → |P y| ≤ 1) :
  P (-1 / n) ≤ 2^(n + 1) - 1 :=
sorry

end polynomial_bound_l50_50281


namespace vector_relation_condition_l50_50905

variables {V : Type*} [AddCommGroup V] (OD OE OM DO EO MO : V)

-- Given condition
theorem vector_relation_condition (h : OD + OE = OM) :

-- Option B
(OM + DO = OE) ∧ 

-- Option C
(OM - OE = OD) ∧ 

-- Option D
(DO + EO = MO) :=
by {
  -- Sorry, to focus on statement only
  sorry
}

end vector_relation_condition_l50_50905


namespace typhoon_tree_survival_l50_50112

def planted_trees : Nat := 150
def died_trees : Nat := 92
def slightly_damaged_trees : Nat := 15

def total_trees_affected : Nat := died_trees + slightly_damaged_trees
def trees_survived_without_damages : Nat := planted_trees - total_trees_affected
def more_died_than_survived : Nat := died_trees - trees_survived_without_damages

theorem typhoon_tree_survival :
  more_died_than_survived = 49 :=
by
  -- Define the necessary computations and assertions
  let total_trees_affected := 92 + 15
  let trees_survived_without_damages := 150 - total_trees_affected
  let more_died_than_survived := 92 - trees_survived_without_damages
  -- Prove the statement
  have : total_trees_affected = 107 := rfl
  have : trees_survived_without_damages = 43 := rfl
  have : more_died_than_survived = 49 := rfl
  exact this

end typhoon_tree_survival_l50_50112


namespace find_x_intervals_l50_50558

theorem find_x_intervals :
  {x : ℝ | x^3 - x^2 + 11*x - 42 < 0} = { x | -2 < x ∧ x < 3 ∨ 3 < x ∧ x < 7 } :=
by sorry

end find_x_intervals_l50_50558


namespace wolf_nobel_laureates_l50_50847

theorem wolf_nobel_laureates (W N total W_prize N_prize N_noW N_W : ℕ)
  (h1 : W_prize = 31)
  (h2 : total = 50)
  (h3 : N_prize = 27)
  (h4 : N_noW + N_W = total - W_prize)
  (h5 : N_W = N_noW + 3)
  (h6 : N_prize = W + N_W) :
  W = 16 :=
by {
  sorry
}

end wolf_nobel_laureates_l50_50847


namespace max_expr_under_condition_l50_50124

-- Define the conditions and variables
variable {x : ℝ}

-- State the theorem about the maximum value of the given expression under the given condition
theorem max_expr_under_condition (h : x < -3) : 
  ∃ M, M = -2 * Real.sqrt 2 - 3 ∧ ∀ y, y < -3 → y + 2 / (y + 3) ≤ M :=
sorry

end max_expr_under_condition_l50_50124


namespace find_triples_l50_50560

theorem find_triples (x y n : ℕ) (hx : x > 0) (hy : y > 0) (hn : n > 0) :
  (x! + y!) / n! = (3:ℕ)^n ↔ (x = 2 ∧ y = 1 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1) :=
by
  sorry

end find_triples_l50_50560


namespace sum_ages_l50_50964

variable (Bob_age Carol_age : ℕ)

theorem sum_ages (h1 : Bob_age = 16) (h2 : Carol_age = 50) (h3 : Carol_age = 3 * Bob_age + 2) :
  Bob_age + Carol_age = 66 :=
by
  sorry

end sum_ages_l50_50964


namespace sandy_total_sums_attempted_l50_50645

theorem sandy_total_sums_attempted (C I : ℕ) 
  (marks_per_correct_sum : ℕ := 3) 
  (marks_lost_per_incorrect_sum : ℕ := 2) 
  (total_marks : ℕ := 45) 
  (correct_sums : ℕ := 21) 
  (H : 3 * correct_sums - 2 * I = total_marks) 
  : C + I = 30 := 
by 
  sorry

end sandy_total_sums_attempted_l50_50645


namespace math_club_probability_l50_50667

open BigOperators

theorem math_club_probability :
  let p := (1/4 : ℚ)
  let prob_club_1 := 1 / (nat.choose 6 3 : ℚ)
  let prob_club_2 := 1 / (nat.choose 9 3 : ℚ)
  let prob_club_3 := 1 / (nat.choose 11 3 : ℚ)
  let prob_club_4 := 1 / (nat.choose 13 3 : ℚ)
  p * (prob_club_1 + prob_club_2 + prob_club_3 + prob_club_4) = (905 / 55440 : ℚ) :=
by
  sorry

end math_club_probability_l50_50667


namespace reduced_rates_start_l50_50212

theorem reduced_rates_start (reduced_fraction : ℝ) (total_hours : ℝ) (weekend_hours : ℝ) (weekday_hours : ℝ) 
  (start_time : ℝ) (end_time : ℝ) : 
  reduced_fraction = 0.6428571428571429 → 
  total_hours = 168 → 
  weekend_hours = 48 → 
  weekday_hours = 60 - weekend_hours → 
  end_time = 8 → 
  start_time = end_time - weekday_hours → 
  start_time = 20 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end reduced_rates_start_l50_50212


namespace james_hours_to_work_l50_50614

theorem james_hours_to_work :
  let meat_cost := 20 * 5
  let fruits_vegetables_cost := 15 * 4
  let bread_cost := 60 * 1.5
  let janitorial_cost := 10 * (10 * 1.5)
  let total_cost := meat_cost + fruits_vegetables_cost + bread_cost + janitorial_cost
  let hourly_wage := 8
  let hours_to_work := total_cost / hourly_wage
  hours_to_work = 50 :=
by 
  sorry

end james_hours_to_work_l50_50614


namespace eval_ceil_floor_sum_l50_50383

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end eval_ceil_floor_sum_l50_50383


namespace perfect_square_pattern_l50_50733

theorem perfect_square_pattern {a b n : ℕ} (h₁ : 1000 ≤ n ∧ n ≤ 9999)
  (h₂ : n = (10 * a + b) ^ 2)
  (h₃ : n = 1100 * a + 11 * b) : n = 7744 :=
  sorry

end perfect_square_pattern_l50_50733


namespace sandro_children_ratio_l50_50644

theorem sandro_children_ratio (d : ℕ) (h1 : d + 3 = 21) : d / 3 = 6 :=
by
  sorry

end sandro_children_ratio_l50_50644


namespace largest_divisor_of_consecutive_odd_product_l50_50493

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h_even : n % 2 = 0) (h_pos : n > 0) :
  315 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) := 
sorry

end largest_divisor_of_consecutive_odd_product_l50_50493


namespace james_total_room_area_l50_50275

theorem james_total_room_area :
  let original_length := 13
  let original_width := 18
  let increase := 2
  let new_length := original_length + increase
  let new_width := original_width + increase
  let area_of_one_room := new_length * new_width
  let number_of_rooms := 4
  let area_of_four_rooms := area_of_one_room * number_of_rooms
  let area_of_larger_room := area_of_one_room * 2
  let total_area := area_of_four_rooms + area_of_larger_room
  total_area = 1800
  := sorry

end james_total_room_area_l50_50275


namespace goose_eggs_laid_l50_50220

theorem goose_eggs_laid (E : ℕ) 
    (H1 : ∃ h, h = (2 / 5) * E)
    (H2 : ∃ m, m = (11 / 15) * h)
    (H3 : ∃ s, s = (1 / 4) * m)
    (H4 : ∃ y, y = (2 / 7) * s)
    (H5 : y = 150) : 
    E = 7160 := 
sorry

end goose_eggs_laid_l50_50220


namespace probability_meeting_twin_probability_twin_in_family_expected_twin_pairs_l50_50846

-- Problem 1
theorem probability_meeting_twin (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  (2 * p) / (p + 1) = (2 * p) / (p + 1) :=
by
  sorry

-- Problem 2
theorem probability_twin_in_family (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  (2 * p) / (2 * p + (1 - p) ^ 2) = (2 * p) / (2 * p + (1 - p) ^ 2) :=
by
  sorry

-- Problem 3
theorem expected_twin_pairs (N : ℕ) (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  N * p / (p + 1) = N * p / (p + 1) :=
by
  sorry

end probability_meeting_twin_probability_twin_in_family_expected_twin_pairs_l50_50846


namespace intersection_of_A_and_B_l50_50913

def A := Set.Ioo 1 3
def B := Set.Ioo 2 4

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 2 3 :=
by
  sorry

end intersection_of_A_and_B_l50_50913


namespace minimum_bird_species_l50_50433

theorem minimum_bird_species (total_birds : ℕ) (h : total_birds = 2021) :
  ∃ (min_species : ℕ), min_species = 1011 ∧ 
  (∀ (species_array : array total_birds ℕ),
   ∀ i j : fin total_birds, 
   species_array[i] = species_array[j] → ((i ≠ j) →
   (abs (i - j) mod 2 = 0))) :=
sorry

end minimum_bird_species_l50_50433


namespace books_per_shelf_l50_50943

theorem books_per_shelf :
  let total_books := 14
  let taken_books := 2
  let shelves := 4
  let remaining_books := total_books - taken_books
  remaining_books / shelves = 3 :=
by
  let total_books := 14
  let taken_books := 2
  let shelves := 4
  let remaining_books := total_books - taken_books
  have h1 : remaining_books = 12 := by simp [remaining_books]
  have h2 : remaining_books / shelves = 3 := by norm_num [remaining_books, shelves]
  exact h2

end books_per_shelf_l50_50943


namespace intersection_of_M_and_N_l50_50108

noncomputable def M : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2 - 1}
noncomputable def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
noncomputable def intersection : Set ℝ := {z | -1 ≤ z ∧ z ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {z | -1 ≤ z ∧ z ≤ 3} := 
sorry

end intersection_of_M_and_N_l50_50108


namespace b_is_multiple_of_5_a_plus_b_is_multiple_of_5_l50_50474

variable (a b : ℕ)

-- Conditions
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k
def is_multiple_of_10 (n : ℕ) : Prop := ∃ k : ℕ, n = 10 * k

-- Given conditions in the problem
axiom h_a : is_multiple_of_5 a
axiom h_b : is_multiple_of_10 b

-- Statements to be proved
theorem b_is_multiple_of_5 : is_multiple_of_5 b :=
sorry

theorem a_plus_b_is_multiple_of_5 : is_multiple_of_5 (a + b) :=
sorry

end b_is_multiple_of_5_a_plus_b_is_multiple_of_5_l50_50474


namespace farmer_total_acres_l50_50047

/--
A farmer used some acres of land for beans, wheat, and corn in the ratio of 5 : 2 : 4, respectively.
There were 376 acres used for corn. Prove that the farmer used a total of 1034 acres of land.
-/
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) :
    5 * x + 2 * x + 4 * x = 1034 :=
by
  sorry

end farmer_total_acres_l50_50047


namespace find_y_of_rectangle_area_l50_50300

theorem find_y_of_rectangle_area (y : ℝ) (h1 : y > 0) 
(h2 : (0, 0) = (0, 0)) (h3 : (0, 6) = (0, 6)) 
(h4 : (y, 6) = (y, 6)) (h5 : (y, 0) = (y, 0)) 
(h6 : 6 * y = 42) : y = 7 :=
by {
  sorry
}

end find_y_of_rectangle_area_l50_50300


namespace ceil_floor_eq_zero_l50_50388

theorem ceil_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (- (7 / 3)) = 0) :=
by
  sorry

end ceil_floor_eq_zero_l50_50388


namespace bananas_per_friend_l50_50491

-- Define constants and conditions
def totalBananas : Nat := 40
def totalFriends : Nat := 40

-- Define the main theorem to prove
theorem bananas_per_friend : totalBananas / totalFriends = 1 := by
  sorry

end bananas_per_friend_l50_50491


namespace number_of_initial_cans_l50_50932

theorem number_of_initial_cans (n : ℕ) (T : ℝ)
  (h1 : T = n * 36.5)
  (h2 : T - (2 * 49.5) = (n - 2) * 30) :
  n = 6 :=
sorry

end number_of_initial_cans_l50_50932


namespace slope_OA_l50_50131

-- Definitions for the given conditions
def ellipse (a b : ℝ) := {P : ℝ × ℝ | (P.1^2) / a^2 + (P.2^2) / b^2 = 1}

def C1 := ellipse 2 1  -- ∑(x^2 / 4 + y^2 = 1)
def C2 := ellipse 2 4  -- ∑(y^2 / 16 + x^2 / 4 = 1)

variable {P₁ P₂ : ℝ × ℝ}  -- Points A and B
variable (h1 : P₁ ∈ C1)
variable (h2 : P₂ ∈ C2)
variable (h_rel : P₂.1 = 2 * P₁.1 ∧ P₂.2 = 2 * P₁.2)  -- ∑(x₂ = 2x₁, y₂ = 2y₁)

-- Proof that the slope of ray OA is ±1
theorem slope_OA : ∃ (m : ℝ), (m = 1 ∨ m = -1) :=
sorry

end slope_OA_l50_50131


namespace total_length_of_segments_in_new_figure_l50_50701

-- Defining the given conditions.
def left_side := 10
def top_side := 3
def right_side := 8
def segments_removed_from_bottom := [2, 1, 2] -- List of removed segments from the bottom.

-- This is the theorem statement that confirms the total length of the new figure's sides.
theorem total_length_of_segments_in_new_figure :
  (left_side + top_side + right_side) = 21 :=
by
  -- This is where the proof would be written.
  sorry

end total_length_of_segments_in_new_figure_l50_50701


namespace translate_line_up_l50_50363

theorem translate_line_up (x y : ℝ) (h : y = 2 * x - 3) : y + 6 = 2 * x + 3 :=
by sorry

end translate_line_up_l50_50363


namespace probability_final_color_green_l50_50185

/-- 
There are initially 7 green amoeba and 3 blue amoeba.
Every minute, each amoeba splits into two identical copies.
After splitting, we randomly remove half of the amoeba so that there are always 10 amoeba.
This process continues until all amoeba are the same color.
This theorem proves the probability that the final color of the amoeba is green is 0.7.
-/
theorem probability_final_color_green 
  (initial_green : ℕ := 7) 
  (initial_blue : ℕ := 3) 
  (total_amoeba : ℕ := 10) 
  : (7 : ℛ)/10 = 0.7 := 
by sorry

end probability_final_color_green_l50_50185


namespace three_segments_form_triangle_l50_50219

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem three_segments_form_triangle :
  ¬ can_form_triangle 2 3 6 ∧
  ¬ can_form_triangle 2 4 6 ∧
  ¬ can_form_triangle 2 2 4 ∧
    can_form_triangle 6 6 6 :=
by
  repeat {sorry}

end three_segments_form_triangle_l50_50219


namespace arithmetic_mean_of_p_and_q_l50_50478

variable (p q r : ℝ)

theorem arithmetic_mean_of_p_and_q
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 22)
  (h3 : r - p = 24) :
  (p + q) / 2 = 10 :=
by
  sorry

end arithmetic_mean_of_p_and_q_l50_50478


namespace quadratic_no_real_roots_l50_50836

theorem quadratic_no_real_roots (a b c : ℝ) (h1 : a = 1) (h2 : b = 3) (h3 : c = 5) : 
  b^2 - 4 * a * c < 0 :=
by
  rw [h1, h2, h3]
  calc
    (3 : ℝ)^2 - 4 * 1 * 5 = 9 - 20 := by norm_num
    ... = -11 := by norm_num
    ... < 0 := by norm_num

end quadratic_no_real_roots_l50_50836


namespace bald_eagle_pairs_l50_50816

theorem bald_eagle_pairs (n_1963 : ℕ) (increase : ℕ) (h1 : n_1963 = 417) (h2 : increase = 6649) :
  (n_1963 + increase = 7066) :=
by
  sorry

end bald_eagle_pairs_l50_50816


namespace quadrilateral_area_is_22_5_l50_50879

-- Define the vertices of the quadrilateral
def vertex1 : ℝ × ℝ := (3, -1)
def vertex2 : ℝ × ℝ := (-1, 4)
def vertex3 : ℝ × ℝ := (2, 3)
def vertex4 : ℝ × ℝ := (9, 9)

-- Define the function to calculate the area using the Shoelace Theorem
noncomputable def shoelace_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  0.5 * (abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v1.2) 
        - (v1.2 * v2.1 + v2.2 * v3.1 + v3.2 * v4.1 + v4.2 * v1.1)))

-- State that the area of the quadrilateral with given vertices is 22.5
theorem quadrilateral_area_is_22_5 :
  shoelace_area vertex1 vertex2 vertex3 vertex4 = 22.5 :=
by 
  -- We skip the proof here.
  sorry

end quadrilateral_area_is_22_5_l50_50879


namespace square_of_cube_of_third_smallest_prime_l50_50831

theorem square_of_cube_of_third_smallest_prime : 
  let p := 5 in (p ^ 3) ^ 2 = 15625 := 
by
  sorry

end square_of_cube_of_third_smallest_prime_l50_50831


namespace alex_initial_jelly_beans_l50_50364

variable (initial : ℕ)
variable (eaten : ℕ := 6)
variable (pile_weight : ℕ := 10)
variable (piles : ℕ := 3)

theorem alex_initial_jelly_beans :
  (initial - eaten = pile_weight * piles) → initial = 36 :=
by
  -- proof will be provided here
  sorry

end alex_initial_jelly_beans_l50_50364


namespace CA_inter_B_l50_50284

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 5, 7}

theorem CA_inter_B :
  (U \ A) ∩ B = {2, 7} := by
  sorry

end CA_inter_B_l50_50284


namespace mean_of_sets_l50_50688

theorem mean_of_sets (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 62) : 
  (48 + 62 + 98 + 124 + x) / 5 = 78 :=
by
  sorry

end mean_of_sets_l50_50688


namespace domain_of_sqrt_expression_l50_50228

theorem domain_of_sqrt_expression :
  {x : ℝ | x^2 - 5 * x - 6 ≥ 0} = {x : ℝ | x ≤ -1} ∪ {x : ℝ | x ≥ 6} := by
sorry

end domain_of_sqrt_expression_l50_50228


namespace cone_base_radius_l50_50358

theorem cone_base_radius (R : ℝ) (theta : ℝ) (radius : ℝ) (hR : R = 30) (hTheta : theta = 120) :
    2 * Real.pi * radius = (theta / 360) * 2 * Real.pi * R → radius = 10 :=
by
  intros h
  sorry

end cone_base_radius_l50_50358


namespace total_balls_l50_50606

theorem total_balls (r b g : ℕ) (ratio : r = 2 * k ∧ b = 4 * k ∧ g = 6 * k) (green_balls : g = 36) : r + b + g = 72 :=
by
  sorry

end total_balls_l50_50606


namespace sum_even_102_to_600_l50_50486

def sum_first_50_even : ℕ := 2550
def sum_even_602_to_700 : ℕ := 32550

theorem sum_even_102_to_600 : sum_even_602_to_700 - sum_first_50_even = 30000 :=
by
  -- The given sum of the first 50 positive even integers is 2550
  have h1 : sum_first_50_even = 2550 := by rfl
  
  -- The given sum of the even integers from 602 to 700 inclusive is 32550
  have h2 : sum_even_602_to_700 = 32550 := by rfl
  
  -- Therefore, the sum of the even integers from 102 to 600 is:
  have h3 : sum_even_602_to_700 - sum_first_50_even = 32550 - 2550 := by
    rw [h1, h2]
  
  -- Calculate the result
  exact h3

end sum_even_102_to_600_l50_50486


namespace sum_fractions_correct_l50_50033

def sum_of_fractions (f1 f2 f3 f4 f5 f6 f7 : ℚ) : ℚ :=
  f1 + f2 + f3 + f4 + f5 + f6 + f7

theorem sum_fractions_correct : sum_of_fractions (1/3) (1/2) (-5/6) (1/5) (1/4) (-9/20) (-5/6) = -5/6 :=
by
  sorry

end sum_fractions_correct_l50_50033


namespace alice_more_than_half_sum_l50_50344

-- Conditions
def row_of_fifty_coins (denominations : List ℤ) : Prop :=
  denominations.length = 50 ∧ (List.sum denominations) % 2 = 1

def alice_starts (denominations : List ℤ) : Prop := True
def bob_follows (denominations : List ℤ) : Prop := True
def alternating_selection (denominations : List ℤ) : Prop := True

-- Question/Proof Goal
theorem alice_more_than_half_sum (denominations : List ℤ) 
  (h1 : row_of_fifty_coins denominations)
  (h2 : alice_starts denominations)
  (h3 : bob_follows denominations)
  (h4 : alternating_selection denominations) :
  ∃ s_A : ℤ, s_A > (List.sum denominations) / 2 ∧ s_A ≤ List.sum denominations :=
sorry

end alice_more_than_half_sum_l50_50344


namespace hexagon_area_correct_problem_solution_l50_50144

noncomputable def hexagon_area (b : ℝ) : ℝ :=
  8 * b + 8 * real.sqrt (b ^ 2 - 12)

noncomputable def b_value : ℝ := 10 / real.sqrt 3

theorem hexagon_area_correct :
  hexagon_area b_value = 48 * real.sqrt 3 :=
by
  sorry

def m : ℕ := 48
def n : ℕ := 3

theorem problem_solution :
  m + n = 51 :=
by
  rw [m, n]
  norm_num

end hexagon_area_correct_problem_solution_l50_50144


namespace correct_option_is_A_l50_50004

def a (n : ℕ) : ℤ :=
  match n with
  | 1 => -3
  | 2 => 7
  | _ => 0  -- This is just a placeholder for other values

def optionA (n : ℕ) : ℤ := (-1)^n * (4*n - 1)
def optionB (n : ℕ) : ℤ := (-1)^n * (4*n + 1)
def optionC (n : ℕ) : ℤ := 4*n - 7
def optionD (n : ℕ) : ℤ := (-1)^(n + 1) * (4*n - 1)

theorem correct_option_is_A :
  (a 1 = -3) ∧ (a 2 = 7) ∧
  (optionA 1 = -3 ∧ optionA 2 = 7) ∧
  ¬(optionB 1 = -3 ∧ optionB 2 = 7) ∧
  ¬(optionC 1 = -3 ∧ optionC 2 = 7) ∧
  ¬(optionD 1 = -3 ∧ optionD 2 = 7) :=
by
  sorry

end correct_option_is_A_l50_50004


namespace greatest_div_by_seven_base_eight_l50_50676

theorem greatest_div_by_seven_base_eight : ∃ n : ℕ, 
  (n < 512) ∧ (Divisibility.divides 7 n) ∧ 
  (∀ m : ℕ, (m < 512) → (Divisibility.divides 7 m) → m ≤ n) ∧ 
  nat.to_digits 8 n = [7, 7, 4] := 
sorry

end greatest_div_by_seven_base_eight_l50_50676


namespace number_of_non_congruent_triangles_l50_50100

theorem number_of_non_congruent_triangles :
  ∃ q : ℕ, q = 3 ∧ 
    (∀ (a b : ℕ), (a ≤ 2 ∧ 2 ≤ b) → (a + 2 > b) ∧ (a + b > 2) ∧ (2 + b > a) →
    (q = 3)) :=
by
  sorry

end number_of_non_congruent_triangles_l50_50100


namespace price_reduction_for_desired_profit_l50_50349

def profit_per_piece (x : ℝ) : ℝ := 40 - x
def pieces_sold_per_day (x : ℝ) : ℝ := 20 + 2 * x

theorem price_reduction_for_desired_profit (x : ℝ) :
  (profit_per_piece x) * (pieces_sold_per_day x) = 1200 ↔ (x = 10 ∨ x = 20) := by
  sorry

end price_reduction_for_desired_profit_l50_50349


namespace students_and_swimmers_l50_50998

theorem students_and_swimmers (N : ℕ) (x : ℕ) 
  (h1 : x = N / 4) 
  (h2 : x / 2 = 4) : 
  N = 32 ∧ N - x = 24 := 
by 
  sorry

end students_and_swimmers_l50_50998


namespace a_b_work_days_l50_50503

-- Definitions:
def work_days_a_b_together := 40
def work_days_a_alone := 12
def remaining_work_days_with_a := 9

-- Statement to be proven:
theorem a_b_work_days (x : ℕ) 
  (h1 : ∀ W : ℕ, W / work_days_a_b_together + remaining_work_days_with_a * (W / work_days_a_alone) = W) :
  x = 10 :=
sorry

end a_b_work_days_l50_50503


namespace jovana_shells_l50_50933

theorem jovana_shells :
  let jovana_initial := 5
  let first_friend := 15
  let second_friend := 17
  jovana_initial + first_friend + second_friend = 37 := by
  sorry

end jovana_shells_l50_50933


namespace claudia_has_three_25_cent_coins_l50_50717

def number_of_coins (x y z : ℕ) := x + y + z = 15
def number_of_combinations (x y : ℕ) := 4 * x + 3 * y = 51

theorem claudia_has_three_25_cent_coins (x y z : ℕ) 
  (h1: number_of_coins x y z) 
  (h2: number_of_combinations x y): 
  z = 3 := 
by 
sorry

end claudia_has_three_25_cent_coins_l50_50717


namespace ratio_of_students_l50_50774

-- Define the conditions
def total_students : Nat := 800
def students_spaghetti : Nat := 320
def students_fettuccine : Nat := 160

-- The proof problem
theorem ratio_of_students (h1 : students_spaghetti = 320) (h2 : students_fettuccine = 160) :
  students_spaghetti / students_fettuccine = 2 := by
  sorry

end ratio_of_students_l50_50774


namespace quadratic_has_one_solution_l50_50566

theorem quadratic_has_one_solution (n : ℤ) : 
  (n ^ 2 - 64 = 0) ↔ (n = 8 ∨ n = -8) := 
by
  sorry

end quadratic_has_one_solution_l50_50566


namespace football_attendance_l50_50019

open Nat

theorem football_attendance:
  (Saturday Monday Wednesday Friday expected_total actual_total: ℕ)
  (h₀: Saturday = 80)
  (h₁: Monday = Saturday - 20)
  (h₂: Wednesday = Monday + 50)
  (h₃: Friday = Saturday + Monday)
  (h₄: expected_total = 350)
  (h₅: actual_total = Saturday + Monday + Wednesday + Friday) :
  actual_total = expected_total + 40 :=
  sorry

end football_attendance_l50_50019


namespace magicStack_cardCount_l50_50174

-- Define the conditions and question based on a)
def isMagicStack (n : ℕ) : Prop :=
  let totalCards := 2 * n
  ∃ (A B : Finset ℕ), (A ∪ B = Finset.range totalCards) ∧
    (∀ x ∈ A, x < n) ∧ (∀ x ∈ B, x ≥ n) ∧
    (∀ i ∈ A, i % 2 = 1) ∧ (∀ j ∈ B, j % 2 = 0) ∧
    (151 ∈ A) ∧
    ∃ (newStack : Finset ℕ), (newStack = A ∪ B) ∧
    (∀ k ∈ newStack, k ∈ A ∨ k ∈ B) ∧
    (151 = 151)

-- The theorem that states the number of cards, when card 151 retains its position, is 452.
theorem magicStack_cardCount :
  isMagicStack 226 → 2 * 226 = 452 :=
by
  sorry

end magicStack_cardCount_l50_50174


namespace calc_f_2005_2007_zero_l50_50377

variable {R : Type} [LinearOrderedField R]

def odd_function (f : R → R) : Prop :=
  ∀ x, f (-x) = -f x

def periodic_function (f : R → R) (p : R) : Prop :=
  ∀ x, f (x + p) = f x

theorem calc_f_2005_2007_zero
  {f : ℝ → ℝ}
  (h_odd : odd_function f)
  (h_period : periodic_function f 4) :
  f 2005 + f 2006 + f 2007 = 0 :=
sorry

end calc_f_2005_2007_zero_l50_50377


namespace chocolate_cost_in_promotion_l50_50711

/-!
Bernie buys two chocolates every week at a local store, where one chocolate costs $3.
In a different store with a promotion, each chocolate costs some amount and Bernie would save $6 
in three weeks if he bought his chocolates there. Prove that the cost of one chocolate 
in the store with the promotion is $2.
-/

theorem chocolate_cost_in_promotion {n p_local savings : ℕ} (weeks : ℕ) (p_promo : ℕ)
  (h_n : n = 2)
  (h_local : p_local = 3)
  (h_savings : savings = 6)
  (h_weeks : weeks = 3)
  (h_promo : p_promo = (p_local * n * weeks - savings) / (n * weeks)) :
  p_promo = 2 :=
by {
  -- Proof would go here
  sorry
}

end chocolate_cost_in_promotion_l50_50711


namespace arithmetic_sequence_a4_eq_1_l50_50746

theorem arithmetic_sequence_a4_eq_1 
  (a : ℕ → ℝ)
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : a 2 ^ 2 + 2 * a 2 * a 6 + a 6 ^ 2 - 4 = 0) : 
  a 4 = 1 :=
sorry

end arithmetic_sequence_a4_eq_1_l50_50746


namespace largest_4_digit_divisible_by_98_l50_50372

theorem largest_4_digit_divisible_by_98 :
  ∃ n, (n ≤ 9999 ∧ 9999 < n + 98) ∧ 98 ∣ n :=
sorry

end largest_4_digit_divisible_by_98_l50_50372


namespace gerald_jail_time_l50_50902

theorem gerald_jail_time
    (assault_sentence : ℕ := 3) 
    (poisoning_sentence_years : ℕ := 2) 
    (third_offense_extension : ℕ := 1 / 3) 
    (months_in_year : ℕ := 12)
    : (assault_sentence + poisoning_sentence_years * months_in_year) * (1 + third_offense_extension) = 36 :=
by
  sorry

end gerald_jail_time_l50_50902


namespace below_sea_level_is_negative_l50_50760
-- Lean 4 statement


theorem below_sea_level_is_negative 
  (above_sea_pos : ∀ x : ℝ, x > 0 → x = x)
  (below_sea_neg : ∀ x : ℝ, x < 0 → x = x) : 
  (-25 = -25) :=
by 
  -- here we are supposed to provide the proof but we are skipping it with sorry
  sorry

end below_sea_level_is_negative_l50_50760


namespace equivalent_proof_problem_l50_50247

lemma condition_1 (a b : ℝ) (h : b > 0 ∧ 0 > a) : (1 / a) < (1 / b) :=
sorry

lemma condition_2 (a b : ℝ) (h : 0 > a ∧ a > b) : (1 / b) > (1 / a) :=
sorry

lemma condition_4 (a b : ℝ) (h : a > b ∧ b > 0) : (1 / b) > (1 / a) :=
sorry

theorem equivalent_proof_problem (a b : ℝ) :
  (b > 0 ∧ 0 > a → (1 / a) < (1 / b)) ∧
  (0 > a ∧ a > b → (1 / b) > (1 / a)) ∧
  (a > b ∧ b > 0 → (1 / b) > (1 / a)) :=
by {
  exact ⟨condition_1 a b, condition_2 a b, condition_4 a b⟩
}

end equivalent_proof_problem_l50_50247


namespace jenny_spent_fraction_l50_50781

theorem jenny_spent_fraction
  (x : ℝ) -- The original amount of money Jenny had
  (h_half_x : 1/2 * x = 21) -- Half of the original amount is $21
  (h_left_money : x - 24 = 24) -- Jenny had $24 left after spending
  : (x - 24) / x = 3 / 7 := sorry

end jenny_spent_fraction_l50_50781


namespace farmer_total_acres_l50_50053

theorem farmer_total_acres (x : ℕ) (H1 : 4 * x = 376) : 
  5 * x + 2 * x + 4 * x = 1034 :=
by
  -- This placeholder is indicating unfinished proof
  sorry

end farmer_total_acres_l50_50053


namespace sticks_in_100th_stage_l50_50955

theorem sticks_in_100th_stage : 
  ∀ (n a₁ d : ℕ), a₁ = 5 → d = 4 → n = 100 → a₁ + (n - 1) * d = 401 :=
by
  sorry

end sticks_in_100th_stage_l50_50955


namespace polynomial_expansion_l50_50887

-- Define the polynomial expressions
def poly1 (s : ℝ) : ℝ := 3 * s^3 - 4 * s^2 + 5 * s - 2
def poly2 (s : ℝ) : ℝ := 2 * s^2 - 3 * s + 4

-- Define the expanded form of the product of the two polynomials
def expanded_poly (s : ℝ) : ℝ :=
  6 * s^5 - 17 * s^4 + 34 * s^3 - 35 * s^2 + 26 * s - 8

-- The theorem to prove the equivalence
theorem polynomial_expansion (s : ℝ) :
  (poly1 s) * (poly2 s) = expanded_poly s :=
sorry -- proof goes here

end polynomial_expansion_l50_50887


namespace license_plate_combinations_l50_50419

theorem license_plate_combinations : 
  let letters := 26 
  let letters_and_digits := 36 
  let middle_character_choices := 2
  3 * letters * letters_and_digits * middle_character_choices = 1872 :=
by
  sorry

end license_plate_combinations_l50_50419


namespace distance_from_C_to_B_is_80_l50_50668

theorem distance_from_C_to_B_is_80
  (x : ℕ)
  (h1 : x = 60)
  (h2 : ∀ (ab cb : ℕ), ab = x → cb = x + 20  → (cb = 80))
  : x + 20 = 80 := by
  sorry

end distance_from_C_to_B_is_80_l50_50668


namespace ram_krish_together_time_l50_50639

theorem ram_krish_together_time : 
  let t_R := 36
  let t_K := t_R / 2
  let task_per_day_R := 1 / t_R
  let task_per_day_K := 1 / t_K
  let task_per_day_together := task_per_day_R + task_per_day_K
  let T := 1 / task_per_day_together
  T = 12 := 
by
  sorry

end ram_krish_together_time_l50_50639


namespace abc_geq_expression_l50_50409

variable (a b c : ℝ) -- Define variables a, b, c as real numbers
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) -- Define conditions of a, b, c being positive

theorem abc_geq_expression : 
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
by 
  sorry -- Proof goes here

end abc_geq_expression_l50_50409


namespace number_of_non_congruent_triangles_with_perimeter_20_l50_50757

theorem number_of_non_congruent_triangles_with_perimeter_20 :
  ∃ T : Finset (Finset ℕ), 
    (∀ t ∈ T, ∃ a b c : ℕ, t = {a, b, c} ∧ a + b + c = 20 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧
    T.card = 14 :=
by
  sorry

end number_of_non_congruent_triangles_with_perimeter_20_l50_50757


namespace arithmetic_sequence_n_value_l50_50918

theorem arithmetic_sequence_n_value (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) :
  a 672 = 2014 :=
sorry

end arithmetic_sequence_n_value_l50_50918


namespace total_pastries_sum_l50_50793

   theorem total_pastries_sum :
     let lola_mini_cupcakes := 13
     let lola_pop_tarts := 10
     let lola_blueberry_pies := 8
     let lola_chocolate_eclairs := 6

     let lulu_mini_cupcakes := 16
     let lulu_pop_tarts := 12
     let lulu_blueberry_pies := 14
     let lulu_chocolate_eclairs := 9

     let lila_mini_cupcakes := 22
     let lila_pop_tarts := 15
     let lila_blueberry_pies := 10
     let lila_chocolate_eclairs := 12

     lola_mini_cupcakes + lulu_mini_cupcakes + lila_mini_cupcakes +
     lola_pop_tarts + lulu_pop_tarts + lila_pop_tarts +
     lola_blueberry_pies + lulu_blueberry_pies + lila_blueberry_pies +
     lola_chocolate_eclairs + lulu_chocolate_eclairs + lila_chocolate_eclairs = 147 :=
   by
     sorry
   
end total_pastries_sum_l50_50793


namespace vector_parallel_has_value_x_l50_50593

-- Define the vectors a and b
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

-- The theorem statement
theorem vector_parallel_has_value_x :
  ∀ x : ℝ, parallel a (b x) → x = 6 :=
by
  intros x h
  sorry

end vector_parallel_has_value_x_l50_50593


namespace optionD_is_quad_eq_in_one_var_l50_50198

/-- Define a predicate for being a quadratic equation in one variable --/
def is_quad_eq_in_one_var (eq : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ ∀ x : ℕ, eq a b c

/-- Options as given predicates --/
def optionA (a b c : ℕ) : Prop := 3 * a^2 - 6 * b + 2 = 0
def optionB (a b c : ℕ) : Prop := a * a^2 - b * a + c = 0
def optionC (a b c : ℕ) : Prop := (1 / a^2) + b = c
def optionD (a b c : ℕ) : Prop := a^2 = 0

/-- Prove that Option D is a quadratic equation in one variable --/
theorem optionD_is_quad_eq_in_one_var : is_quad_eq_in_one_var optionD :=
sorry

end optionD_is_quad_eq_in_one_var_l50_50198


namespace probability_of_product_multiple_of_4_is_2_5_l50_50598

open Finset BigOperators

def all_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s \ s.diag

def num_favorable_pairs (s : Finset ℕ) : ℕ :=
  (all_pairs s).filter (λ p => (p.1 * p.2) % 4 = 0).card

def probability_multiple_of_4 : ℚ :=
  let s := (finset.range 7).filter (λ n => n ≠ 0)
  let total_pairs := (all_pairs s).card
  let favorable_pairs := num_favorable_pairs s
  favorable_pairs / total_pairs

theorem probability_of_product_multiple_of_4_is_2_5 :
  probability_multiple_of_4 = 2 / 5 := by
  -- skipping the proof
  sorry

end probability_of_product_multiple_of_4_is_2_5_l50_50598


namespace range_of_a_l50_50107

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * x + a ≤ 0) → a > 1 :=
by
  sorry

end range_of_a_l50_50107


namespace general_term_formula_l50_50102

noncomputable def S (n : ℕ) : ℕ := 2^n - 1
noncomputable def a (n : ℕ) : ℕ := 2^(n-1)

theorem general_term_formula (n : ℕ) (hn : n > 0) : 
    a n = S n - S (n - 1) := 
by 
  sorry

end general_term_formula_l50_50102


namespace non_subset_condition_l50_50306

theorem non_subset_condition (M P : Set α) (non_empty : M ≠ ∅) : 
  ¬(M ⊆ P) ↔ ∃ x ∈ M, x ∉ P := 
sorry

end non_subset_condition_l50_50306


namespace devin_basketball_chances_l50_50002

theorem devin_basketball_chances 
  (initial_chances : ℝ := 0.1) 
  (base_height : ℕ := 66) 
  (chance_increase_per_inch : ℝ := 0.1)
  (initial_height : ℕ := 65) 
  (growth : ℕ := 3) :
  initial_chances + (growth + initial_height - base_height) * chance_increase_per_inch = 0.3 := 
by 
  sorry

end devin_basketball_chances_l50_50002


namespace youngest_child_age_l50_50343

theorem youngest_child_age :
  ∃ x : ℕ, x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 65 ∧ x = 7 :=
by
  sorry

end youngest_child_age_l50_50343


namespace find_c_l50_50488

open Real

theorem find_c (a b c d : ℕ) (M : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) (h4 : d > 1) (hM : M ≠ 1) :
  (M ^ (1 / a) * (M ^ (1 / b) * (M ^ (1 / c) * (M ^ (1 / d))))) ^ (1 / a * b * c * d) = (M ^ 37) ^ (1 / 48) →
  c = 2 :=
by
  sorry

end find_c_l50_50488


namespace aluminum_weight_l50_50292

variable {weight_iron : ℝ}
variable {weight_aluminum : ℝ}
variable {difference : ℝ}

def weight_aluminum_is_correct (weight_iron weight_aluminum difference : ℝ) : Prop := 
  weight_iron = weight_aluminum + difference

theorem aluminum_weight 
  (H1 : weight_iron = 11.17)
  (H2 : difference = 10.33)
  (H3 : weight_aluminum_is_correct weight_iron weight_aluminum difference) : 
  weight_aluminum = 0.84 :=
sorry

end aluminum_weight_l50_50292


namespace store_profit_l50_50162

variables (m n : ℝ)

def total_profit (m n : ℝ) : ℝ :=
  110 * m - 50 * n

theorem store_profit (m n : ℝ) : total_profit m n = 110 * m - 50 * n :=
  by
  -- sorry indicates that the proof is skipped
  sorry

end store_profit_l50_50162


namespace Oliver_ferris_wheel_rides_l50_50709

theorem Oliver_ferris_wheel_rides :
  ∃ (F : ℕ), (4 * 7 + F * 7 = 63) ∧ (F = 5) :=
by
  sorry

end Oliver_ferris_wheel_rides_l50_50709


namespace mark_pages_per_week_l50_50154

theorem mark_pages_per_week
    (initial_hours_per_day : ℕ)
    (increase_percentage : ℕ)
    (initial_pages_per_day : ℕ) :
    initial_hours_per_day = 2 →
    increase_percentage = 150 →
    initial_pages_per_day = 100 →
    (initial_pages_per_day * (1 + increase_percentage / 100)) * 7 = 1750 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have reading_speed := 100 / 2 -- 50 pages per hour
  have increased_time := 2 * 1.5  -- 3 more hours
  have new_total_time := 2 + 3    -- 5 hours per day
  have pages_per_day := 5 * 50    -- 250 pages per day
  have pages_per_week := 250 * 7  -- 1750 pages per week
  exact eq.refl 1750

end mark_pages_per_week_l50_50154


namespace problem_statement_l50_50475

theorem problem_statement (x : ℤ) (y : ℝ) (h : y = 0.5) : 
  (⌈x + y⌉ - ⌊x + y⌋ = 1) ∧ (⌈x + y⌉ - (x + y) = 0.5) := 
by 
  sorry

end problem_statement_l50_50475


namespace pure_imaginary_z1z2_l50_50625

def z1 : ℂ := 3 + 2 * Complex.i
def z2 (m : ℝ) : ℂ := 1 + m * Complex.i

theorem pure_imaginary_z1z2 (m : ℝ) : (z1 * z2 m).re = 0 → m = 3 / 2 :=
by
  sorry

end pure_imaginary_z1z2_l50_50625


namespace evaluate_expression_l50_50083

theorem evaluate_expression : 
  (16 = 2^4) → 
  (32 = 2^5) → 
  (16^24 / 32^12 = 8^12) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end evaluate_expression_l50_50083


namespace min_species_needed_l50_50438

theorem min_species_needed (num_birds : ℕ) (h1 : num_birds = 2021)
  (h2 : ∀ (s : ℤ) (x y : ℕ), x ≠ y → (between_same_species : ℕ) → (h3 : between_same_species = y - x - 1) → between_same_species % 2 = 0) :
  ∃ (species : ℕ), num_birds ≤ 2 * species ∧ species = 1011 :=
by
  sorry

end min_species_needed_l50_50438


namespace zero_in_interval_l50_50184

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 x + 2 * x - 8

theorem zero_in_interval : (f 3 < 0) ∧ (f 4 > 0) → ∃ c, 3 < c ∧ c < 4 ∧ f c = 0 :=
by
  sorry

end zero_in_interval_l50_50184


namespace product_of_good_numbers_is_good_l50_50974

def is_good (n : ℕ) : Prop :=
  ∃ (a b c x y : ℤ), n = a * x * x + b * x * y + c * y * y ∧ b * b - 4 * a * c = -20

theorem product_of_good_numbers_is_good {n1 n2 : ℕ} (h1 : is_good n1) (h2 : is_good n2) : is_good (n1 * n2) :=
sorry

end product_of_good_numbers_is_good_l50_50974


namespace john_bought_two_shirts_l50_50279

/-- The number of shirts John bought, given the conditions:
1. The first shirt costs $6 more than the second shirt.
2. The first shirt costs $15.
3. The total cost of the shirts is $24,
is equal to 2. -/
theorem john_bought_two_shirts
  (S : ℝ) 
  (first_shirt_cost : ℝ := 15)
  (second_shirt_cost : ℝ := S)
  (cost_difference : first_shirt_cost = second_shirt_cost + 6)
  (total_cost : first_shirt_cost + second_shirt_cost = 24)
  : 2 = 2 :=
by
  sorry

end john_bought_two_shirts_l50_50279


namespace tan_sum_inequality_l50_50801

noncomputable def pi : ℝ := Real.pi

theorem tan_sum_inequality (x α : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ pi / 2) (hα1 : pi / 6 < α) (hα2 : α < pi / 3) :
  Real.tan (pi * (Real.sin x) / (4 * Real.sin α)) + Real.tan (pi * (Real.cos x) / (4 * Real.cos α)) > 1 :=
by
  sorry

end tan_sum_inequality_l50_50801


namespace star_three_five_l50_50093

def star (x y : ℕ) := x^2 + 2 * x * y + y^2

theorem star_three_five : star 3 5 = 64 :=
by
  sorry

end star_three_five_l50_50093


namespace annie_jacob_ratio_l50_50795

theorem annie_jacob_ratio :
  ∃ (a j : ℕ), ∃ (m : ℕ), (m = 2 * a) ∧ (j = 90) ∧ (m = 60) ∧ (a / j = 1 / 3) :=
by
  sorry

end annie_jacob_ratio_l50_50795


namespace tangent_line_at_0_maximum_integer_value_of_a_l50_50106

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - a*x + 2

-- Part (1)
-- Prove that the equation of the tangent line to f(x) at x = 0 is x + y - 2 = 0 when a = 2
theorem tangent_line_at_0 {a : ℝ} (h : a = 2) : ∀ x y : ℝ, (y = f x a) → (x = 0) → (y = 2 - x) :=
by 
  sorry

-- Part (2)
-- Prove that if f(x) + 2x + x log(x+1) ≥ 0 holds for all x ≥ 0, then the maximum integer value of a is 4
theorem maximum_integer_value_of_a 
  (h : ∀ x : ℝ, x ≥ 0 → f x a + 2 * x + x * Real.log (x + 1) ≥ 0) : a ≤ 4 :=
by
  sorry

end tangent_line_at_0_maximum_integer_value_of_a_l50_50106


namespace installation_quantities_l50_50182

theorem installation_quantities :
  ∃ x1 x2 x3 : ℕ, x1 = 22 ∧ x2 = 88 ∧ x3 = 22 ∧
  (x1 + x2 + x3 ≥ 100) ∧
  (x2 = 4 * x1) ∧
  (∃ k : ℕ, x3 = k * x1) ∧
  (5 * x3 = x2 + 22) :=
  by {
    -- We are simply stating the equivalence and supporting conditions.
    -- Here, we will use 'sorry' as a placeholder.
    sorry
  }

end installation_quantities_l50_50182


namespace rhombus_area_l50_50823

theorem rhombus_area (s : ℝ) (d1 d2 : ℝ) (h1 : s = Real.sqrt 145) (h2 : abs (d1 - d2) = 10) : 
  (1/2) * d1 * d2 = 100 :=
sorry

end rhombus_area_l50_50823


namespace length_of_train_is_correct_l50_50983

noncomputable def length_of_train (speed : ℕ) (time : ℕ) : ℕ :=
  (speed * (time / 3600) * 1000)

theorem length_of_train_is_correct : length_of_train 70 36 = 700 := by
  sorry

end length_of_train_is_correct_l50_50983


namespace complement_intersection_l50_50415

def U : Set ℤ := Set.univ
def A : Set ℤ := {1, 2}
def B : Set ℤ := {3, 4}

-- A ∪ B should equal {1, 2, 3, 4}
axiom AUeq : A ∪ B = {1, 2, 3, 4}

theorem complement_intersection : (U \ A) ∩ B = {3, 4} :=
by
  sorry

end complement_intersection_l50_50415


namespace infinite_superset_of_infinite_subset_l50_50768

theorem infinite_superset_of_infinite_subset {A B : Set ℕ} (h_subset : B ⊆ A) (h_infinite : Infinite B) : Infinite A := 
sorry

end infinite_superset_of_infinite_subset_l50_50768


namespace additional_ice_cubes_made_l50_50882

def original_ice_cubes : ℕ := 2
def total_ice_cubes : ℕ := 9

theorem additional_ice_cubes_made :
  (total_ice_cubes - original_ice_cubes) = 7 :=
by
  sorry

end additional_ice_cubes_made_l50_50882


namespace equation_represents_two_intersecting_lines_l50_50502

theorem equation_represents_two_intersecting_lines :
  (∀ x y : ℝ, x^3 * (x + y - 2) = y^3 * (x + y - 2) ↔
    (x = y ∨ y = 2 - x)) :=
by sorry

end equation_represents_two_intersecting_lines_l50_50502


namespace factorize_1_factorize_2_l50_50086

-- Define the variables involved
variables (a x y : ℝ)

-- Problem (1): 18a^2 - 32 = 2 * (3a + 4) * (3a - 4)
theorem factorize_1 (a : ℝ) : 
  18 * a^2 - 32 = 2 * (3 * a + 4) * (3 * a - 4) :=
sorry

-- Problem (2): y - 6xy + 9x^2y = y * (1 - 3x) ^ 2
theorem factorize_2 (x y : ℝ) : 
  y - 6 * x * y + 9 * x^2 * y = y * (1 - 3 * x) ^ 2 :=
sorry

end factorize_1_factorize_2_l50_50086


namespace smallest_value_am_hm_l50_50283

theorem smallest_value_am_hm (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * ((1 / (a + b + c)) + (1 / (b + c)) + (1 / (c + a))) ≥ 9 / 2 :=
sorry

end smallest_value_am_hm_l50_50283


namespace polynomial_coefficients_sum_and_difference_l50_50845

theorem polynomial_coefficients_sum_and_difference :
  ∀ (a_0 a_1 a_2 a_3 a_4 : ℤ),
  (∀ (x : ℤ), (2 * x - 3)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  (a_1 + a_2 + a_3 + a_4 = -80) ∧ ((a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 625) :=
by
  intros a_0 a_1 a_2 a_3 a_4 h
  sorry

end polynomial_coefficients_sum_and_difference_l50_50845


namespace no_real_solutions_l50_50650

noncomputable def original_eq (x : ℝ) : Prop := (x^2 + x + 1) / (x + 1) = x^2 + 5 * x + 6

theorem no_real_solutions (x : ℝ) : ¬ original_eq x :=
by
  sorry

end no_real_solutions_l50_50650


namespace latest_start_time_for_liz_l50_50151

def latest_start_time (weight : ℕ) (roast_time_per_pound : ℕ) (num_turkeys : ℕ) (dinner_time : ℕ) : ℕ :=
  dinner_time - (num_turkeys * weight * roast_time_per_pound) / 60

theorem latest_start_time_for_liz : 
  latest_start_time 16 15 2 18 = 10 := by
  sorry

end latest_start_time_for_liz_l50_50151


namespace sqrt_prod_simplified_l50_50075

open Real

variable (x : ℝ)

theorem sqrt_prod_simplified (hx : 0 ≤ x) : sqrt (50 * x) * sqrt (18 * x) * sqrt (8 * x) = 30 * x * sqrt (2 * x) :=
by
  sorry

end sqrt_prod_simplified_l50_50075


namespace interior_triangle_area_l50_50246

theorem interior_triangle_area (s1 s2 s3 : ℝ) (hs1 : s1 = 15) (hs2 : s2 = 6) (hs3 : s3 = 15) 
  (a1 a2 a3 : ℝ) (ha1 : a1 = 225) (ha2 : a2 = 36) (ha3 : a3 = 225) 
  (h1 : s1 * s1 = a1) (h2 : s2 * s2 = a2) (h3 : s3 * s3 = a3) :
  (1/2) * s1 * s2 = 45 :=
by
  sorry

end interior_triangle_area_l50_50246


namespace arrangement_volunteers_l50_50168

theorem arrangement_volunteers (A B : Finset ℕ) (hA : A.card = 3) (hB : B.card = 3) :
  (∀ row : Finset ℕ, row.card = 3 → ∀ x y ∈ row, x ∈ A ∧ y ∈ A ∨ x ∈ B ∧ y ∈ B → x ≠ y) →
  (∃ arr : ℕ, arr = 72) :=
sorry

end arrangement_volunteers_l50_50168


namespace ratio_A_B_l50_50812

-- Given conditions as definitions
def P_both : ℕ := 500  -- Number of people who purchased both books A and B

def P_only_B : ℕ := P_both / 2  -- Number of people who purchased only book B

def P_only_A : ℕ := 1000  -- Number of people who purchased only book A

-- Total number of people who purchased books
def P_A : ℕ := P_only_A + P_both  -- Total number of people who purchased book A

def P_B : ℕ := P_only_B + P_both  -- Total number of people who purchased book B

-- The ratio of people who purchased book A to book B
theorem ratio_A_B : P_A / P_B = 2 :=
by
  sorry

end ratio_A_B_l50_50812


namespace negation_of_proposition_l50_50309

theorem negation_of_proposition : 
    (¬ (∀ x : ℝ, x^2 - 2 * |x| ≥ 0)) ↔ (∃ x : ℝ, x^2 - 2 * |x| < 0) :=
by sorry

end negation_of_proposition_l50_50309


namespace degrees_to_radians_l50_50552

theorem degrees_to_radians (degrees : ℝ) (pi : ℝ) : 
  degrees * (pi / 180) = pi / 15 ↔ degrees = 12 :=
by 
  sorry

end degrees_to_radians_l50_50552


namespace mark_total_theater_spending_l50_50794

def week1_cost : ℝ := (3 * 5 - 0.2 * (3 * 5)) + 3
def week2_cost : ℝ := (2.5 * 6 - 0.1 * (2.5 * 6)) + 3
def week3_cost : ℝ := 4 * 4 + 3
def week4_cost : ℝ := (3 * 5 - 0.2 * (3 * 5)) + 3
def week5_cost : ℝ := (2 * (3.5 * 6 - 0.1 * (3.5 * 6))) + 6
def week6_cost : ℝ := 2 * 7 + 3

def total_cost : ℝ := week1_cost + week2_cost + week3_cost + week4_cost + week5_cost + week6_cost

theorem mark_total_theater_spending : total_cost = 126.30 := sorry

end mark_total_theater_spending_l50_50794


namespace typing_pages_l50_50769

theorem typing_pages (typists : ℕ) (pages min : ℕ) 
  (h_typists_can_type_two_pages_in_two_minutes : typists * 2 / min = pages / min) 
  (h_10_typists_type_25_pages_in_5_minutes : 10 * 25 / 5 = pages / min) :
  pages / min = 2 := 
sorry

end typing_pages_l50_50769


namespace double_persons_half_work_l50_50505

theorem double_persons_half_work :
  (∀ (n : ℕ) (d : ℕ), d = 12 → (2 * n) * (d / 2) = n * 3) :=
by
  sorry

end double_persons_half_work_l50_50505


namespace slower_speed_l50_50863

theorem slower_speed (f e d : ℕ) (h1 : f = 14) (h2 : e = 20) (h3 : d = 50) (x : ℕ) : 
  (50 / x : ℚ) = (50 / 14 : ℚ) + (20 / 14 : ℚ) → x = 10 := by
  sorry

end slower_speed_l50_50863


namespace balance_scale_weights_part_a_balance_scale_weights_part_b_l50_50487

-- Part (a)
theorem balance_scale_weights_part_a (w : List ℕ) (h : w = List.range (90 + 1) \ List.range 1) :
  ¬ ∃ (A B : List ℕ), A.length = 2 * B.length ∧ A.sum = B.sum :=
sorry

-- Part (b)
theorem balance_scale_weights_part_b (w : List ℕ) (h : w = List.range (99 + 1) \ List.range 1) :
  ∃ (A B : List ℕ), A.length = 2 * B.length ∧ A.sum = B.sum :=
sorry

end balance_scale_weights_part_a_balance_scale_weights_part_b_l50_50487


namespace bottom_price_l50_50117

open Nat

theorem bottom_price (B T : ℕ) (h1 : T = B + 300) (h2 : 3 * B + 3 * T = 21000) : B = 3350 := by
  sorry

end bottom_price_l50_50117


namespace probability_of_winning_l50_50516

noncomputable def probability_winning_prize (n_types n_bags : ℕ) : ℝ :=
let total_combinations := n_types ^ n_bags in
let not_winning_combinations := nat.choose n_types 2 * n_types ^ (n_bags - 1) - n_types in
let probability_not_winning := not_winning_combinations / total_combinations in
1 - probability_not_winning

theorem probability_of_winning :
  probability_winning_prize 3 4 = 4 / 9 := 
sorry

end probability_of_winning_l50_50516


namespace length_base_bc_l50_50446

theorem length_base_bc {A B C D : Type} [Inhabited A]
  (AB AC : ℕ)
  (BD : ℕ → ℕ → ℕ → ℕ) -- function for the median on AC
  (perimeter1 perimeter2 : ℕ)
  (h1 : AB = AC)
  (h2 : perimeter1 = 24 ∨ perimeter2 = 30)
  (AD CD : ℕ) :
  (AD = CD ∧ (∃ ab ad cd, ab + ad = perimeter1 ∧ cd + ad = perimeter2 ∧ ((AB = 2 * AD ∧ BC = 30 - CD) ∨ (AB = 2 * AD ∧ BC = 24 - CD)))) →
  (BC = 22 ∨ BC = 14) := 
sorry

end length_base_bc_l50_50446


namespace lcm_16_24_45_l50_50976

-- Define the numbers
def a : Nat := 16
def b : Nat := 24
def c : Nat := 45

-- State the theorem that the least common multiple of these numbers is 720
theorem lcm_16_24_45 : Nat.lcm (Nat.lcm 16 24) 45 = 720 := by
  sorry

end lcm_16_24_45_l50_50976


namespace measure_of_arc_BD_l50_50808

-- Definitions for conditions
def diameter (A B M : Type) : Prop := sorry -- Placeholder definition for diameter
def chord (C D M : Type) : Prop := sorry -- Placeholder definition for chord intersecting at point M
def angle_measure (A B C : Type) (angle_deg: ℝ) : Prop := sorry -- Placeholder for angle measure
def arc_measure (C B : Type) (arc_deg: ℝ) : Prop := sorry -- Placeholder for arc measure

-- Main theorem to prove
theorem measure_of_arc_BD
  (A B C D M : Type)
  (h_diameter : diameter A B M)
  (h_chord : chord C D M)
  (h_angle_CMB : angle_measure C M B 73)
  (h_arc_BC : arc_measure B C 110) :
  ∃ (arc_BD : ℝ), arc_BD = 144 :=
by
  sorry

end measure_of_arc_BD_l50_50808


namespace admittedApplicants_l50_50511

-- Definitions for the conditions in the problem
def totalApplicants : ℕ := 70
def task1Applicants : ℕ := 35
def task2Applicants : ℕ := 48
def task3Applicants : ℕ := 64
def task4Applicants : ℕ := 63

-- The proof statement
theorem admittedApplicants : 
  ∀ (totalApplicants task3Applicants task4Applicants : ℕ),
  totalApplicants = 70 →
  task3Applicants = 64 →
  task4Applicants = 63 →
  ∃ (interApplicants : ℕ), interApplicants = 57 :=
by
  intros totalApplicants task3Applicants task4Applicants
  intros h_totalApps h_task3Apps h_task4Apps
  sorry

end admittedApplicants_l50_50511


namespace total_cars_produced_l50_50211

def cars_produced_north_america : ℕ := 3884
def cars_produced_europe : ℕ := 2871
def cars_produced_asia : ℕ := 5273
def cars_produced_south_america : ℕ := 1945

theorem total_cars_produced : cars_produced_north_america + cars_produced_europe + cars_produced_asia + cars_produced_south_america = 13973 := by
  sorry

end total_cars_produced_l50_50211


namespace calculate_lower_profit_percentage_l50_50706

theorem calculate_lower_profit_percentage 
  (CP : ℕ) 
  (profitAt18Percent : ℕ) 
  (additionalProfit : ℕ)
  (hCP : CP = 800) 
  (hProfitAt18Percent : profitAt18Percent = 144) 
  (hAdditionalProfit : additionalProfit = 72) 
  (hProfitRelation : profitAt18Percent = additionalProfit + ((9 * CP) / 100)) :
  9 = ((9 * CP) / 100) :=
by
  sorry

end calculate_lower_profit_percentage_l50_50706


namespace arithmetic_sequence_geo_ratio_l50_50750

theorem arithmetic_sequence_geo_ratio
  (a_n : ℕ → ℝ)
  (d : ℝ)
  (h_nonzero : d ≠ 0)
  (S : ℕ → ℝ)
  (h_seq : ∀ n, S n = (n * (2 * a_n 1 + (n - 1) * d)) / 2)
  (h_geo : (S 2) ^ 2 = S 1 * S 4) :
  (a_n 2 + a_n 3) / a_n 1 = 8 :=
by sorry

end arithmetic_sequence_geo_ratio_l50_50750


namespace chickens_cheaper_than_eggs_l50_50642

-- Define the initial costs of the chickens
def initial_cost_chicken1 : ℝ := 25
def initial_cost_chicken2 : ℝ := 30
def initial_cost_chicken3 : ℝ := 22
def initial_cost_chicken4 : ℝ := 35

-- Define the weekly feed costs for the chickens
def weekly_feed_cost_chicken1 : ℝ := 1.50
def weekly_feed_cost_chicken2 : ℝ := 1.30
def weekly_feed_cost_chicken3 : ℝ := 1.10
def weekly_feed_cost_chicken4 : ℝ := 0.90

-- Define the weekly egg production for the chickens
def weekly_egg_prod_chicken1 : ℝ := 4
def weekly_egg_prod_chicken2 : ℝ := 3
def weekly_egg_prod_chicken3 : ℝ := 5
def weekly_egg_prod_chicken4 : ℝ := 2

-- Define the cost of a dozen eggs at the store
def cost_per_dozen_eggs : ℝ := 2

-- Define total initial costs, total weekly feed cost, and weekly savings
def total_initial_cost : ℝ := initial_cost_chicken1 + initial_cost_chicken2 + initial_cost_chicken3 + initial_cost_chicken4
def total_weekly_feed_cost : ℝ := weekly_feed_cost_chicken1 + weekly_feed_cost_chicken2 + weekly_feed_cost_chicken3 + weekly_feed_cost_chicken4
def weekly_savings : ℝ := cost_per_dozen_eggs

-- Define the condition for the number of weeks (W) when the chickens become cheaper
def breakeven_weeks : ℝ := 40

theorem chickens_cheaper_than_eggs (W : ℕ) :
  total_initial_cost + W * total_weekly_feed_cost = W * weekly_savings :=
sorry

end chickens_cheaper_than_eggs_l50_50642


namespace shorter_piece_length_l50_50508

theorem shorter_piece_length (L : ℝ) (k : ℝ) (shorter_piece : ℝ) : 
  L = 28 ∧ k = 2.00001 / 5 ∧ L = shorter_piece + k * shorter_piece → 
  shorter_piece = 20 :=
by
  sorry

end shorter_piece_length_l50_50508


namespace probability_of_forming_triangle_l50_50234

noncomputable def sticks : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triangle_combinations : List (ℕ × ℕ × ℕ) :=
  (sticks.choose 3).filter (λ t, satisfies_triangle_inequality t.1 t.2.fst t.2.snd)

theorem probability_of_forming_triangle :
  (valid_triangle_combinations.length : ℚ) / (sticks.choose 3).length = 9 / 28 :=
by
  sorry

end probability_of_forming_triangle_l50_50234


namespace bird_species_min_l50_50442

theorem bird_species_min (total_birds : ℕ) (h_total_birds : total_birds = 2021)
  (h_even_between : ∀ (species : Sort*) (a b : species), (a ≠ b) → even (nat.dist a b)) :
  ∃ species_num : ℕ, species_num = 1011 :=
by
  sorry

end bird_species_min_l50_50442


namespace incorrect_equation_l50_50698

theorem incorrect_equation (x : ℕ) (h : x + 2 * (12 - x) = 20) : 2 * (12 - x) - 20 ≠ x :=
by 
  sorry

end incorrect_equation_l50_50698


namespace combination_18_6_l50_50537

theorem combination_18_6 : nat.choose 18 6 = 18564 :=
by {
  sorry
}

end combination_18_6_l50_50537


namespace chocolate_bar_cost_l50_50141

-- Define the quantities Jessica bought
def chocolate_bars := 10
def gummy_bears_packs := 10
def chocolate_chips_bags := 20

-- Define the costs
def total_cost := 150
def gummy_bears_pack_cost := 2
def chocolate_chips_bag_cost := 5

-- Define what we want to prove (the cost of one chocolate bar)
theorem chocolate_bar_cost : 
  ∃ chocolate_bar_cost, 
    chocolate_bars * chocolate_bar_cost + 
    gummy_bears_packs * gummy_bears_pack_cost + 
    chocolate_chips_bags * chocolate_chips_bag_cost = total_cost ∧
    chocolate_bar_cost = 3 :=
by
  -- Proof goes here
  sorry

end chocolate_bar_cost_l50_50141


namespace jason_total_amount_l50_50780

def shorts_price : ℝ := 14.28
def jacket_price : ℝ := 4.74
def shoes_price : ℝ := 25.95
def socks_price : ℝ := 6.80
def tshirts_price : ℝ := 18.36
def hat_price : ℝ := 12.50
def swimsuit_price : ℝ := 22.95
def sunglasses_price : ℝ := 45.60
def wristbands_price : ℝ := 9.80

def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price - (price * discount)

def total_discounted_price : ℝ := 
  (discounted_price shorts_price discount1) + 
  (discounted_price jacket_price discount1) + 
  (discounted_price hat_price discount1) + 
  (discounted_price shoes_price discount2) + 
  (discounted_price socks_price discount2) + 
  (discounted_price tshirts_price discount2) + 
  (discounted_price swimsuit_price discount2) + 
  (discounted_price sunglasses_price discount2) + 
  (discounted_price wristbands_price discount2)

def total_with_tax : ℝ := total_discounted_price + (total_discounted_price * sales_tax_rate)

theorem jason_total_amount : total_with_tax = 153.07 := by
  sorry

end jason_total_amount_l50_50780


namespace initial_concentration_of_hydrochloric_acid_l50_50045

theorem initial_concentration_of_hydrochloric_acid
  (initial_mass : ℕ)
  (drained_mass : ℕ)
  (added_concentration : ℕ)
  (final_concentration : ℕ)
  (total_mass : ℕ)
  (initial_concentration : ℕ) :
  initial_mass = 300 ∧ drained_mass = 25 ∧ added_concentration = 80 ∧ final_concentration = 25 ∧ total_mass = 300 →
  (275 * initial_concentration / 100 + 20 = 75) →
  initial_concentration = 20 :=
by
  intros h_eq h_new_solution
  -- Rewriting the data given in h_eq and solving h_new_solution
  rcases h_eq with ⟨h_initial_mass, h_drained_mass, h_added_concentration, h_final_concentration, h_total_mass⟩
  sorry

end initial_concentration_of_hydrochloric_acid_l50_50045


namespace least_number_to_subtract_l50_50333

theorem least_number_to_subtract (n : ℕ) (h : n = 427398) : ∃ k : ℕ, (n - k) % 11 = 0 ∧ k = 4 :=
by
  sorry

end least_number_to_subtract_l50_50333


namespace ratio_a_to_c_l50_50963

theorem ratio_a_to_c (a b c d : ℚ)
  (h1 : a / b = 5 / 2)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 1 / 3) :
  a / c = 15 / 8 :=
by {
  sorry
}

end ratio_a_to_c_l50_50963


namespace find_2a_minus_3b_l50_50591

theorem find_2a_minus_3b
  (a b : ℝ)
  (h1 : a * 2 - b * 1 = 4)
  (h2 : a * 2 + b * 1 = 2) :
  2 * a - 3 * b = 6 :=
by
  sorry

end find_2a_minus_3b_l50_50591


namespace simplify_expression_l50_50640

theorem simplify_expression (a b : ℝ) (h : a ≠ b) : 
  ((a^3 - b^3) / (a * b)) - ((a * b^2 - b^3) / (a * b - a^3)) = (2 * a * (a - b)) / b :=
by
  sorry

end simplify_expression_l50_50640


namespace no_solution_to_inequalities_l50_50890

theorem no_solution_to_inequalities : 
  ∀ x : ℝ, ¬ (4 * x - 3 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x - 5) :=
by
  sorry

end no_solution_to_inequalities_l50_50890


namespace garden_perimeter_l50_50956

noncomputable def find_perimeter (l w : ℕ) : ℕ := 2 * l + 2 * w

theorem garden_perimeter :
  ∀ (l w : ℕ),
  (l = 3 * w + 2) →
  (l = 38) →
  find_perimeter l w = 100 :=
by
  intros l w H1 H2
  sorry

end garden_perimeter_l50_50956


namespace quadratic_equation_roots_l50_50413

theorem quadratic_equation_roots (m n : ℝ) 
  (h_sum : m + n = -3) 
  (h_prod : m * n = 1) 
  (h_equation : m^2 + 3 * m + 1 = 0) :
  (3 * m + 1) / (m^3 * n) = -1 := 
by sorry

end quadratic_equation_roots_l50_50413


namespace triangle_projection_inequality_l50_50844

variable (a b c t r μ : ℝ)
variable (h1 : AC_1 = 2 * t * AB)
variable (h2 : BA_1 = 2 * r * BC)
variable (h3 : CB_1 = 2 * μ * AC)
variable (h4 : AB = c)
variable (h5 : AC = b)
variable (h6 : BC = a)

theorem triangle_projection_inequality
  (h1 : AC_1 = 2 * t * AB)  -- condition AC_1 = 2t * AB
  (h2 : BA_1 = 2 * r * BC)  -- condition BA_1 = 2r * BC
  (h3 : CB_1 = 2 * μ * AC)  -- condition CB_1 = 2μ * AC
  (h4 : AB = c)             -- side AB
  (h5 : AC = b)             -- side AC
  (h6 : BC = a)             -- side BC
  : (a^2 / b^2) * (t / (1 - 2 * t))^2 
  + (b^2 / c^2) * (r / (1 - 2 * r))^2 
  + (c^2 / a^2) * (μ / (1 - 2 * μ))^2 
  + 16 * t * r * μ ≥ 1 := 
  sorry

end triangle_projection_inequality_l50_50844


namespace rays_total_grocery_bill_l50_50802

-- Conditions
def hamburger_meat_cost : ℝ := 5.0
def crackers_cost : ℝ := 3.50
def frozen_veg_cost_per_bag : ℝ := 2.0
def frozen_veg_bags : ℕ := 4
def cheese_cost : ℝ := 3.50
def discount_rate : ℝ := 0.10

-- Total cost before discount
def total_cost_before_discount : ℝ :=
  hamburger_meat_cost + crackers_cost + (frozen_veg_cost_per_bag * frozen_veg_bags) + cheese_cost

-- Discount amount
def discount_amount : ℝ := discount_rate * total_cost_before_discount

-- Total cost after discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

-- Theorem: Ray's total grocery bill
theorem rays_total_grocery_bill : total_cost_after_discount = 18.0 :=
  by
    sorry

end rays_total_grocery_bill_l50_50802


namespace rectangle_perimeter_l50_50880

theorem rectangle_perimeter (long_side short_side : ℝ) 
  (h_long : long_side = 1) 
  (h_short : short_side = long_side - 2/8) : 
  2 * long_side + 2 * short_side = 3.5 := 
by 
  sorry

end rectangle_perimeter_l50_50880


namespace charlotte_flour_cost_l50_50374

noncomputable def flour_cost 
  (flour_sugar_eggs_butter_cost blueberry_cost cherry_cost total_cost : ℝ)
  (blueberry_weight oz_per_lb blueberry_cost_per_container cherry_weight cherry_cost_per_bag : ℝ)
  (additional_cost : ℝ) : ℝ :=
  total_cost - (blueberry_cost + additional_cost)

theorem charlotte_flour_cost :
  flour_cost 2.5 13.5 14 18 3 16 2.25 4 14 2.5 = 2 :=
by
  unfold flour_cost
  sorry

end charlotte_flour_cost_l50_50374


namespace smallest_integer_y_l50_50826

theorem smallest_integer_y : ∃ y : ℤ, (8:ℚ) / 11 < y / 17 ∧ ∀ z : ℤ, ((8:ℚ) / 11 < z / 17 → y ≤ z) :=
by
  sorry

end smallest_integer_y_l50_50826


namespace find_other_root_l50_50912

theorem find_other_root (b : ℝ) (h : ∀ x : ℝ, x^2 - b * x + 3 = 0 → x = 3 ∨ ∃ y, y = 1) :
  ∃ y, y = 1 :=
by
  sorry

end find_other_root_l50_50912


namespace john_total_spent_l50_50142

noncomputable def computer_cost : ℝ := 1500
noncomputable def peripherals_cost : ℝ := (1 / 4) * computer_cost
noncomputable def base_video_card_cost : ℝ := 300
noncomputable def upgraded_video_card_cost : ℝ := 2.5 * base_video_card_cost
noncomputable def discount_on_video_card : ℝ := 0.12 * upgraded_video_card_cost
noncomputable def video_card_cost_after_discount : ℝ := upgraded_video_card_cost - discount_on_video_card
noncomputable def sales_tax_on_peripherals : ℝ := 0.05 * peripherals_cost
noncomputable def total_spent : ℝ := computer_cost + peripherals_cost + video_card_cost_after_discount + sales_tax_on_peripherals

theorem john_total_spent : total_spent = 2553.75 := by
  sorry

end john_total_spent_l50_50142


namespace arithmetic_progression_sum_l50_50603

theorem arithmetic_progression_sum (a d : ℝ)
  (h1 : 10 * (2 * a + 19 * d) = 200)
  (h2 : 25 * (2 * a + 49 * d) = 0) :
  35 * (2 * a + 69 * d) = -466.67 :=
by
  sorry

end arithmetic_progression_sum_l50_50603


namespace prove_d_minus_r_eq_1_l50_50763

theorem prove_d_minus_r_eq_1 
  (d r : ℕ) 
  (h_d1 : d > 1)
  (h1 : 1122 % d = r)
  (h2 : 1540 % d = r)
  (h3 : 2455 % d = r) :
  d - r = 1 :=
by sorry

end prove_d_minus_r_eq_1_l50_50763


namespace population_increase_l50_50429

-- Define the problem conditions
def average_birth_rate := (6 + 10) / 2 / 2  -- the average number of births per second
def average_death_rate := (4 + 8) / 2 / 2  -- the average number of deaths per second
def net_migration_day := 500  -- net migration inflow during the day
def net_migration_night := -300  -- net migration outflow during the night

-- Define the number of seconds in a day
def seconds_in_a_day := 24 * 3600

-- Define the net increase due to births and deaths
def net_increase_births_deaths := (average_birth_rate - average_death_rate) * seconds_in_a_day

-- Define the total net migration
def total_net_migration := net_migration_day + net_migration_night

-- Define the total population net increase
def total_population_net_increase :=
  net_increase_births_deaths + total_net_migration

-- The theorem to be proved
theorem population_increase (h₁ : average_birth_rate = 4)
                           (h₂ : average_death_rate = 3)
                           (h₃ : seconds_in_a_day = 86400) :
  total_population_net_increase = 86600 := by
  sorry

end population_increase_l50_50429


namespace smallest_positive_angle_l50_50410

open Real

theorem smallest_positive_angle (α : ℝ) 
  (h1 : (sin (2 * π / 3), cos (2 * π / 3)) = (sin α, cos α)) :
  α = 11 * π / 6 :=
by sorry

end smallest_positive_angle_l50_50410


namespace truck_wheels_l50_50966

theorem truck_wheels (t x : ℝ) (wheels_front : ℕ) (wheels_other : ℕ) :
  (t = 1.50 + 1.50 * (x - 2)) → (t = 6) → (wheels_front = 2) → (wheels_other = 4) → x = 5 → 
  (wheels_front + wheels_other * (x - 1) = 18) :=
by
  intros h1 h2 h3 h4 h5
  rw [h5] at *
  sorry

end truck_wheels_l50_50966


namespace find_number_l50_50354

theorem find_number (x : ℝ) (h : x / 0.04 = 25) : x = 1 := 
by 
  -- the steps for solving this will be provided here
  sorry

end find_number_l50_50354


namespace four_digit_perfect_square_l50_50735

theorem four_digit_perfect_square : 
  ∃ (N : ℕ), (1000 ≤ N ∧ N ≤ 9999) ∧ (∃ (a b : ℕ), a = N / 1000 ∧ b = (N % 100) / 10 ∧ a = N / 100 - (N / 100 % 10) ∧ b = (N % 100 / 10) - N % 10) ∧ (∃ (n : ℕ), N = n * n) →
  N = 7744 := 
sorry

end four_digit_perfect_square_l50_50735


namespace leadership_structure_ways_l50_50703

/-!
# Leadership Selection in a Tribe

Given a tribe consisting of 15 members, we prove that the number of different ways to choose:
1. One chief,
2. Three supporting chiefs,
3. Two inferior officers for each supporting chief,
is 19320300.
-/

open Finset

noncomputable def chooseLeadershipWays (total_members : ℕ) : ℕ :=
  total_members * 
  (total_members - 1) * 
  (total_members - 2) * 
  choose (total_members - 4) 2 * 
  choose (total_members - 6) 2 * 
  choose (total_members - 8) 2

theorem leadership_structure_ways (total_members : ℕ)
  (h : total_members = 15) : chooseLeadershipWays total_members = 19320300 :=
by
  rw [chooseLeadershipWays, h]
  norm_num
  sorry

end leadership_structure_ways_l50_50703


namespace perpendicular_slope_l50_50554

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 4 * y = 20) : 
  ∃ m : ℝ, m = -4 / 5 :=
sorry

end perpendicular_slope_l50_50554


namespace problem_solution_1_problem_solution_2_l50_50815

def Sn (n : ℕ) := n * (n + 2)

def a_n (n : ℕ) := 2 * n + 1

def b_n (n : ℕ) := 2 ^ (n - 1)

def c_n (n : ℕ) := if n % 2 = 1 then 2 / Sn n else b_n n

def T_n (n : ℕ) : ℤ := (Finset.range n).sum (λ i => c_n (i + 1))

theorem problem_solution_1 : 
  ∀ (n : ℕ), a_n n = 2 * n + 1 ∧ b_n n = 2 ^ (n - 1) := 
  by sorry

theorem problem_solution_2 (n : ℕ) : 
  T_n (2 * n) = (2 * n) / (2 * n + 1) + (2 / 3) * (4 ^ n - 1) := 
  by sorry

end problem_solution_1_problem_solution_2_l50_50815


namespace greatest_base8_three_digit_divisible_by_7_l50_50683

theorem greatest_base8_three_digit_divisible_by_7 :
  ∃ n : ℕ, n < 8^3 ∧ n ≥ 8^2 ∧ (n % 7 = 0) ∧ (to_base 8 n = 777) :=
sorry

end greatest_base8_three_digit_divisible_by_7_l50_50683


namespace intersection_M_N_l50_50476

-- Define the sets M and N according to the conditions given in the problem
def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

-- State the theorem to prove the intersection of M and N
theorem intersection_M_N : (M ∩ N) = {0, 1} := 
  sorry

end intersection_M_N_l50_50476


namespace sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1_l50_50556

theorem sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1 (t : ℝ) : 
  Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) :=
sorry

end sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1_l50_50556


namespace find_fx_for_neg_x_l50_50575

-- Let f be an odd function defined on ℝ 
variable {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = - f x)

-- Given condition for x > 0
variable (h_pos : ∀ x, 0 < x → f x = x^2 + x - 1)

-- Problem: Prove that f(x) = -x^2 + x + 1 for x < 0
theorem find_fx_for_neg_x (x : ℝ) (h_neg : x < 0) : f x = -x^2 + x + 1 :=
sorry

end find_fx_for_neg_x_l50_50575


namespace train_passing_time_l50_50870

noncomputable def speed_in_m_per_s : ℝ := (60 * 1000) / 3600

variable (L : ℝ) (S : ℝ)
variable (train_length : L = 500)
variable (train_speed : S = speed_in_m_per_s)

theorem train_passing_time : L / S = 30 := by
  sorry

end train_passing_time_l50_50870


namespace find_n_modulo_l50_50401

theorem find_n_modulo (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 11) (h3 : n ≡ 15827 [ZMOD 12]) : n = 11 :=
by
  sorry

end find_n_modulo_l50_50401


namespace eighth_term_of_geometric_sequence_l50_50824

def geometric_sequence_term (a₁ r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem eighth_term_of_geometric_sequence : 
  geometric_sequence_term 12 (1 / 3) 8 = 4 / 729 :=
by 
  sorry

end eighth_term_of_geometric_sequence_l50_50824


namespace ceil_floor_eq_zero_l50_50386

theorem ceil_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (- (7 / 3)) = 0) :=
by
  sorry

end ceil_floor_eq_zero_l50_50386


namespace transform_correct_l50_50526

variable {α : Type} [Mul α] [DecidableEq α]

theorem transform_correct (a b c : α) (h : a = b) : a * c = b * c :=
by sorry

end transform_correct_l50_50526


namespace desired_depth_proof_l50_50691

-- Definitions based on the conditions in Step a)
def initial_men : ℕ := 9
def initial_hours : ℕ := 8
def initial_depth : ℕ := 30
def extra_men : ℕ := 11
def total_men : ℕ := initial_men + extra_men
def new_hours : ℕ := 6

-- Total man-hours for initial setup
def initial_man_hours (days : ℕ) : ℕ := initial_men * initial_hours * days

-- Total man-hours for new setup to achieve desired depth
def new_man_hours (desired_depth : ℕ) (days : ℕ) : ℕ := total_men * new_hours * days

-- Proportional relationship between initial setup and desired depth
theorem desired_depth_proof (days : ℕ) (desired_depth : ℕ) :
  initial_man_hours days / initial_depth = new_man_hours desired_depth days / desired_depth → desired_depth = 18 :=
by
  sorry

end desired_depth_proof_l50_50691


namespace unique_solution_values_a_l50_50891

theorem unique_solution_values_a (a : ℝ) : 
  (∃ x y : ℝ, |x| + |y - 1| = 1 ∧ y = a * x + 2012) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, (|x1| + |y1 - 1| = 1 ∧ y1 = a * x1 + 2012) ∧ 
                      (|x2| + |y2 - 1| = 1 ∧ y2 = a * x2 + 2012) → 
                      (x1 = x2 ∧ y1 = y2)) ↔ 
  a = 2011 ∨ a = -2011 := 
sorry

end unique_solution_values_a_l50_50891


namespace largest_integer_m_l50_50739

theorem largest_integer_m (m : ℤ) : (m^2 - 11 * m + 28 < 0) → m = 6 :=
begin
  sorry
end

end largest_integer_m_l50_50739


namespace setD_is_pythagorean_triple_l50_50069

def is_pythagorean_triple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the sets
def setA := (1, 2, 3)
def setB := (4, 5, 6)
def setC := (6, 8, 9)
def setD := (7, 24, 25)

-- Prove that Set D is a Pythagorean triple
theorem setD_is_pythagorean_triple : is_pythagorean_triple 7 24 25 :=
by
  show 7^2 + 24^2 = 25^2,
  calc
  7^2 + 24^2 = 49 + 576 := by norm_num
  ... = 625 := by norm_num
  ... = 25^2 := by norm_num

end setD_is_pythagorean_triple_l50_50069


namespace number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime_l50_50091

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime 
  (n : ℕ) (h : n ≥ 2) : (∃ (a b : ℕ), a ≠ b ∧ is_prime (a^3 + 2) ∧ is_prime (b^3 + 2)) :=
by
  sorry

end number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime_l50_50091


namespace farmer_total_acres_l50_50055

theorem farmer_total_acres (x : ℕ) (H1 : 4 * x = 376) : 
  5 * x + 2 * x + 4 * x = 1034 :=
by
  -- This placeholder is indicating unfinished proof
  sorry

end farmer_total_acres_l50_50055


namespace sqrt_real_domain_l50_50263

theorem sqrt_real_domain (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 := 
sorry

end sqrt_real_domain_l50_50263


namespace lucy_sales_is_43_l50_50643

def total_packs : Nat := 98
def robyn_packs : Nat := 55
def lucy_packs : Nat := total_packs - robyn_packs

theorem lucy_sales_is_43 : lucy_packs = 43 :=
by
  sorry

end lucy_sales_is_43_l50_50643


namespace expected_adjacent_black_pairs_60_cards_l50_50655

noncomputable def expected_adjacent_black_pairs 
(deck_size : ℕ) (black_cards : ℕ) (red_cards : ℕ) : ℚ :=
  if h : deck_size = black_cards + red_cards 
  then (black_cards:ℚ) * (black_cards - 1) / (deck_size - 1) 
  else 0

theorem expected_adjacent_black_pairs_60_cards :
  expected_adjacent_black_pairs 60 36 24 = 1260 / 59 := by
  sorry

end expected_adjacent_black_pairs_60_cards_l50_50655


namespace equations_create_24_l50_50160

theorem equations_create_24 :
  ∃ (eq1 eq2 : ℤ),
  ((eq1 = 3 * (-6 + 4 + 10) ∧ eq1 = 24) ∧ 
   (eq2 = 4 - (-6 / 3) * 10 ∧ eq2 = 24)) ∧ 
   eq1 ≠ eq2 := 
by
  sorry

end equations_create_24_l50_50160


namespace total_boys_and_girls_sum_to_41_l50_50295

theorem total_boys_and_girls_sum_to_41 (Rs : ℕ) (amount_per_boy : ℕ) (amount_per_girl : ℕ) (total_amount : ℕ) (num_boys : ℕ) :
  Rs = 1 ∧ amount_per_boy = 12 * Rs ∧ amount_per_girl = 8 * Rs ∧ total_amount = 460 * Rs ∧ num_boys = 33 →
  ∃ num_girls : ℕ, num_boys + num_girls = 41 :=
by
  sorry

end total_boys_and_girls_sum_to_41_l50_50295


namespace solve_trig_system_l50_50653

theorem solve_trig_system
  (k n : ℤ) :
  (∃ x y : ℝ,
    (2 * Real.sin x ^ 2 + 2 * Real.sqrt 2 * Real.sin x * Real.sin (2 * x) ^ 2 + Real.sin (2 * x) ^ 2 = 0 ∧
     Real.cos x = Real.cos y) ∧
    ((x = 2 * Real.pi * k ∧ y = 2 * Real.pi * n) ∨
     (x = Real.pi + 2 * Real.pi * k ∧ y = Real.pi + 2 * Real.pi * n) ∨
     (x = -Real.pi / 4 + 2 * Real.pi * k ∧ (y = Real.pi / 4 + 2 * Real.pi * n ∨ y = -Real.pi / 4 + 2 * Real.pi * n)) ∨
     (x = -3 * Real.pi / 4 + 2 * Real.pi * k ∧ (y = 3 * Real.pi / 4 + 2 * Real.pi * n ∨ y = -3 * Real.pi / 4 + 2 * Real.pi * n)))) :=
sorry

end solve_trig_system_l50_50653


namespace quadratic_axis_of_symmetry_l50_50005

theorem quadratic_axis_of_symmetry (b c : ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A = (0, 3))
  (hB : B = (2, 3)) 
  (h_passA : 3 = 0^2 + b * 0 + c) 
  (h_passB : 3 = 2^2 + b * 2 + c) : 
  ∃ x, x = 1 :=
by {
  -- Given: The quadratic function y = x^2 + bx + c
  -- Conditions: Passes through A(0, 3) and B(2, 3)
  -- We need to prove: The axis of symmetry is x = 1
  sorry,
}

end quadratic_axis_of_symmetry_l50_50005


namespace sum_of_ages_l50_50934

theorem sum_of_ages (a b c : ℕ) (h1 : a * b * c = 72) (h2 : b = c) (h3 : a < b) : a + b + c = 14 :=
sorry

end sum_of_ages_l50_50934


namespace bernardo_larger_than_silvia_l50_50529

def bernardo_set : finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def silvia_set : finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def number_of_ways_to_choose_3_from_bernardo : ℚ :=
  ((bernardo_set.card).choose 3)

def number_of_ways_to_choose_3_from_silvia : ℚ :=
  ((silvia_set.card).choose 3)

def probability_of_bernardo_choosing_9 : ℚ :=
  ((bernardo_set.erase 9).card.choose 2) / number_of_ways_to_choose_3_from_bernardo

def probability_of_large_number_without_9 : ℚ :=
  17 / 35

theorem bernardo_larger_than_silvia :
  (probability_of_bernardo_choosing_9
    + (1 - probability_of_bernardo_choosing_9) * probability_of_large_number_without_9) 
  = 112 / 175 :=
begin
  sorry
end

end bernardo_larger_than_silvia_l50_50529


namespace simon_change_l50_50647

noncomputable def calculate_change 
  (pansies_count : ℕ) (pansies_price : ℚ) (pansies_discount : ℚ) 
  (hydrangea_count : ℕ) (hydrangea_price : ℚ) (hydrangea_discount : ℚ) 
  (petunias_count : ℕ) (petunias_price : ℚ) (petunias_discount : ℚ) 
  (lilies_count : ℕ) (lilies_price : ℚ) (lilies_discount : ℚ) 
  (orchids_count : ℕ) (orchids_price : ℚ) (orchids_discount : ℚ) 
  (sales_tax : ℚ) (payment : ℚ) : ℚ :=
  let pansies_total := (pansies_count * pansies_price) * (1 - pansies_discount)
  let hydrangea_total := (hydrangea_count * hydrangea_price) * (1 - hydrangea_discount)
  let petunias_total := (petunias_count * petunias_price) * (1 - petunias_discount)
  let lilies_total := (lilies_count * lilies_price) * (1 - lilies_discount)
  let orchids_total := (orchids_count * orchids_price) * (1 - orchids_discount)
  let total_price := pansies_total + hydrangea_total + petunias_total + lilies_total + orchids_total
  let final_price := total_price * (1 + sales_tax)
  payment - final_price

theorem simon_change : calculate_change
  5 2.50 0.10
  1 12.50 0.15
  5 1.00 0.20
  3 5.00 0.12
  2 7.50 0.08
  0.06 100 = 43.95 := by sorry

end simon_change_l50_50647


namespace four_digit_perfect_square_is_1156_l50_50244

theorem four_digit_perfect_square_is_1156 :
  ∃ (N : ℕ), (N ≥ 1000) ∧ (N < 10000) ∧ (∀ a, a ∈ [N / 1000, (N % 1000) / 100, (N % 100) / 10, N % 10] → a < 7) 
              ∧ (∃ n : ℕ, N = n * n) ∧ (∃ m : ℕ, (N + 3333 = m * m)) ∧ (N = 1156) :=
by
  sorry

end four_digit_perfect_square_is_1156_l50_50244


namespace complementary_implies_right_triangle_l50_50179

theorem complementary_implies_right_triangle (A B C : ℝ) (h : A + B = 90 ∧ A + B + C = 180) :
  C = 90 :=
by
  sorry

end complementary_implies_right_triangle_l50_50179


namespace max_a_value_l50_50618

theorem max_a_value (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 50) :
  a ≤ 2924 :=
by sorry

end max_a_value_l50_50618


namespace average_of_possible_values_l50_50421

theorem average_of_possible_values 
  (x : ℝ)
  (h : Real.sqrt (2 * x^2 + 5) = Real.sqrt 25) : 
  (x = Real.sqrt 10 ∨ x = -Real.sqrt 10) → (Real.sqrt 10 + (-Real.sqrt 10)) / 2 = 0 :=
by
  sorry

end average_of_possible_values_l50_50421


namespace calculate_value_l50_50203

theorem calculate_value (h : 2994 * 14.5 = 179) : 29.94 * 1.45 = 0.179 :=
by
  sorry

end calculate_value_l50_50203


namespace scooter_price_l50_50280

theorem scooter_price (total_cost: ℝ) (h: 0.20 * total_cost = 240): total_cost = 1200 :=
by
  sorry

end scooter_price_l50_50280


namespace parabola_focus_l50_50175

-- Define the equation of the parabola
def parabola_eq (x y : ℝ) : Prop := y^2 = -8 * x

-- Define the coordinates of the focus
def focus (x y : ℝ) : Prop := x = -2 ∧ y = 0

-- The Lean statement that needs to be proved
theorem parabola_focus : ∀ (x y : ℝ), parabola_eq x y → focus x y :=
by
  intros x y h
  sorry

end parabola_focus_l50_50175


namespace smallest_integer_ending_in_9_divisible_by_11_is_99_l50_50192

noncomputable def smallest_positive_integer_ending_in_9_and_divisible_by_11 : ℕ :=
  99

theorem smallest_integer_ending_in_9_divisible_by_11_is_99 :
  ∃ n : ℕ, n > 0 ∧ (n % 10 = 9) ∧ (n % 11 = 0) ∧
          (∀ m : ℕ, m > 0 → (m % 10 = 9) → (m % 11 = 0) → n ≤ m) :=
begin
  use smallest_positive_integer_ending_in_9_and_divisible_by_11,
  split,
  { -- n > 0
    exact nat.zero_lt_bit1 nat.zero_lt_one },
  split,
  { -- n % 10 = 9
    exact nat.mod_eq_of_lt (by norm_num) },
  split,
  { -- n % 11 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 99) },
  { -- ∀ m > 0, m % 10 = 9, m % 11 = 0 → n ≤ m
    intros m hm1 hm2 hm3,
    change 99 ≤ m,
    -- m % 99 = 0 → 99 ≤ m since 99 > 0
    sorry
  }
end

end smallest_integer_ending_in_9_divisible_by_11_is_99_l50_50192


namespace child_height_at_age_10_l50_50214

theorem child_height_at_age_10 (x y : ℝ) (h : y = 7.19 * x + 73.93) (hx : x = 10) : abs (y - 145.83) < 1 :=
by {
  sorry
}

end child_height_at_age_10_l50_50214


namespace log_function_domain_l50_50562

noncomputable def domain_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Set ℝ :=
  { x : ℝ | x < a }

theorem log_function_domain (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x, x ∈ domain_of_log_function a h1 h2 ↔ x < a :=
by
  sorry

end log_function_domain_l50_50562


namespace farmer_total_acres_l50_50050

-- Define the problem conditions and the total acres
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) : 5 * x + 2 * x + 4 * x = 1034 :=
by
  -- Proof is omitted
  sorry

end farmer_total_acres_l50_50050


namespace reduction_rate_equation_l50_50924

-- Define the given conditions
def original_price : ℝ := 23
def reduced_price : ℝ := 18.63
def monthly_reduction_rate (x : ℝ) : ℝ := (1 - x) ^ 2

-- Prove that the given equation holds
theorem reduction_rate_equation (x : ℝ) : 
  original_price * monthly_reduction_rate x = reduced_price :=
by
  sorry

end reduction_rate_equation_l50_50924


namespace range_of_a_l50_50293

open Real

namespace PropositionProof

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2*x - 8 > 0)

theorem range_of_a (a : ℝ) (h : a < 0) :
  (¬ ∀ x, ¬ p a x → ∀ x, ¬ q x) ↔ (a ≤ -4 ∨ -2/3 ≤ a ∧ a < 0) :=
sorry

end PropositionProof

end range_of_a_l50_50293


namespace jessica_attended_games_l50_50025

/-- 
Let total_games be the total number of soccer games.
Let initially_planned be the number of games Jessica initially planned to attend.
Let commitments_skipped be the number of games skipped due to other commitments.
Let rescheduled_games be the rescheduled games during the season.
Let additional_missed be the additional games missed due to rescheduling.
-/
theorem jessica_attended_games
    (total_games initially_planned commitments_skipped rescheduled_games additional_missed : ℕ)
    (h1 : total_games = 12)
    (h2 : initially_planned = 8)
    (h3 : commitments_skipped = 3)
    (h4 : rescheduled_games = 2)
    (h5 : additional_missed = 4) :
    (initially_planned - commitments_skipped) - additional_missed = 1 := by
  sorry

end jessica_attended_games_l50_50025


namespace largest_t_value_maximum_t_value_l50_50741

noncomputable def largest_t : ℚ :=
  (5 : ℚ) / 2

theorem largest_t_value (t : ℚ) :
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  t ≤ (5 : ℚ) / 2 :=
sorry

theorem maximum_t_value (t : ℚ) :
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  (5 : ℚ) / 2 = largest_t :=
sorry

end largest_t_value_maximum_t_value_l50_50741


namespace Sarah_correct_responses_l50_50298

theorem Sarah_correct_responses : ∃ x : ℕ, x ≥ 22 ∧ (7 * x - (26 - x) + 4 ≥ 150) :=
by
  sorry

end Sarah_correct_responses_l50_50298


namespace concentric_circles_radius_difference_l50_50027

theorem concentric_circles_radius_difference (r R : ℝ)
  (h : R^2 = 4 * r^2) :
  R - r = r :=
by
  sorry

end concentric_circles_radius_difference_l50_50027


namespace nested_g_of_2_l50_50149

def g (x : ℤ) : ℤ := x^2 - 4*x + 3

theorem nested_g_of_2 : g (g (g (g (g (g 2))))) = 1394486148248 := by
  sorry

end nested_g_of_2_l50_50149


namespace union_of_A_and_B_l50_50939

def setA : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def setB : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_of_A_and_B : setA ∪ setB = {x | -1 < x ∧ x ≤ 3} :=
by {
  sorry
}

end union_of_A_and_B_l50_50939


namespace min_b_factors_l50_50895

theorem min_b_factors (x r s b : ℕ) (h : r * s = 1998) (fact : (x + r) * (x + s) = x^2 + b * x + 1998) : b = 91 :=
sorry

end min_b_factors_l50_50895


namespace original_triangle_area_l50_50809

theorem original_triangle_area (A_new : ℝ) (r : ℝ) (A_original : ℝ) 
  (h1 : r = 3) 
  (h2 : A_new = 54) 
  (h3 : A_new = r^2 * A_original) : 
  A_original = 6 := 
by 
  sorry

end original_triangle_area_l50_50809


namespace sum_5n_is_630_l50_50604

variable (n : ℕ)

def sum_first_k (k : ℕ) : ℕ :=
  k * (k + 1) / 2

theorem sum_5n_is_630 (h : sum_first_k (3 * n) = sum_first_k n + 210) : sum_first_k (5 * n) = 630 := sorry

end sum_5n_is_630_l50_50604


namespace jean_business_hours_l50_50771

-- Definitions of the conditions
def weekday_hours : ℕ := 10 - 16 -- from 4 pm to 10 pm
def weekend_hours : ℕ := 10 - 18 -- from 6 pm to 10 pm
def weekdays : ℕ := 5 -- Monday through Friday
def weekends : ℕ := 2 -- Saturday and Sunday

-- Total weekly hours
def total_weekly_hours : ℕ :=
  (weekday_hours * weekdays) + (weekend_hours * weekends)

-- Proof statement
theorem jean_business_hours : total_weekly_hours = 38 :=
by
  sorry

end jean_business_hours_l50_50771


namespace largest_divisor_of_seven_consecutive_odd_numbers_l50_50494

theorem largest_divisor_of_seven_consecutive_odd_numbers (n : ℕ) (h : Even n) (h_pos : n > 0) :
  ∃ d, d = 45 ∧ ∀ k, k ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) → k ≤ 45 :=
sorry

end largest_divisor_of_seven_consecutive_odd_numbers_l50_50494


namespace how_many_oxen_c_put_l50_50360

variables (oxen_a oxen_b months_a months_b rent total_rent c_share x : ℕ)
variable (H : 10 * 7 = oxen_a)
variable (H1 : 12 * 5 = oxen_b)
variable (H2 : 3 * x = months_a)
variable (H3 : 70 + 60 + 3 * x = months_b)
variable (H4 : 280 = total_rent)
variable (H5 : 72 = c_share)

theorem how_many_oxen_c_put : x = 15 :=
  sorry

end how_many_oxen_c_put_l50_50360


namespace integers_with_abs_less_than_four_l50_50269

theorem integers_with_abs_less_than_four :
  {x : ℤ | |x| < 4} = {-3, -2, -1, 0, 1, 2, 3} :=
sorry

end integers_with_abs_less_than_four_l50_50269


namespace james_total_room_area_l50_50274

theorem james_total_room_area :
  let original_length := 13
  let original_width := 18
  let increase := 2
  let new_length := original_length + increase
  let new_width := original_width + increase
  let area_of_one_room := new_length * new_width
  let number_of_rooms := 4
  let area_of_four_rooms := area_of_one_room * number_of_rooms
  let area_of_larger_room := area_of_one_room * 2
  let total_area := area_of_four_rooms + area_of_larger_room
  total_area = 1800
  := sorry

end james_total_room_area_l50_50274


namespace hyperbola_equation_l50_50303

theorem hyperbola_equation (h : ∃ (x y : ℝ), y = 1 / 2 * x) (p : (2, 2) ∈ {p : ℝ × ℝ | ((p.snd)^2 / 3) - ((p.fst)^2 / 12) = 1}) :
  ∀ (x y : ℝ), (y^2 / 3 - x^2 / 12 = 1) ↔ (∃ (a b : ℝ), y = a * x ∧ b * y = x ^ 2) :=
sorry

end hyperbola_equation_l50_50303


namespace proof_f_of_2_add_g_of_3_l50_50148

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x^2 + 2 * x - 1

theorem proof_f_of_2_add_g_of_3 : f (2 + g 3) = 44 :=
by
  sorry

end proof_f_of_2_add_g_of_3_l50_50148


namespace least_n_divisible_by_25_and_7_l50_50506

theorem least_n_divisible_by_25_and_7 (n : ℕ) (h1 : n > 1) (h2 : n % 25 = 1) (h3 : n % 7 = 1) : n = 126 :=
by
  sorry

end least_n_divisible_by_25_and_7_l50_50506


namespace add_base_12_l50_50361

theorem add_base_12 :
  let a := 5*12^2 + 1*12^1 + 8*12^0
  let b := 2*12^2 + 7*12^1 + 6*12^0
  let result := 7*12^2 + 9*12^1 + 2*12^0
  a + b = result :=
by
  -- Placeholder for the actual proof
  sorry

end add_base_12_l50_50361


namespace mixture_percent_chemical_a_l50_50649

-- Defining the conditions
def solution_x : ℝ := 0.4
def solution_y : ℝ := 0.5
def percent_x_in_mixture : ℝ := 0.3
def percent_y_in_mixture : ℝ := 1.0 - percent_x_in_mixture

-- The goal is to prove that the mixture is 47% chemical a
theorem mixture_percent_chemical_a : (solution_x * percent_x_in_mixture + solution_y * percent_y_in_mixture) * 100 = 47 :=
by
  -- Calculation here
  sorry

end mixture_percent_chemical_a_l50_50649


namespace selling_price_correct_l50_50996

noncomputable def discount1 (price : ℝ) : ℝ := price * 0.85
noncomputable def discount2 (price : ℝ) : ℝ := price * 0.90
noncomputable def discount3 (price : ℝ) : ℝ := price * 0.95

noncomputable def final_price (initial_price : ℝ) : ℝ :=
  discount3 (discount2 (discount1 initial_price))

theorem selling_price_correct : final_price 3600 = 2616.30 := by
  sorry

end selling_price_correct_l50_50996


namespace total_potatoes_sold_is_322kg_l50_50697

-- Define the given conditions
def bags_morning := 29
def bags_afternoon := 17
def weight_per_bag := 7

-- The theorem to prove the total kilograms sold is 322kg
theorem total_potatoes_sold_is_322kg : (bags_morning + bags_afternoon) * weight_per_bag = 322 :=
by
  sorry -- Placeholder for the actual proof

end total_potatoes_sold_is_322kg_l50_50697


namespace combination_18_6_l50_50544

theorem combination_18_6 : (nat.choose 18 6) = 18564 := 
by 
  sorry

end combination_18_6_l50_50544


namespace count_four_digit_numbers_with_digit_sum_4_l50_50345

theorem count_four_digit_numbers_with_digit_sum_4 : 
  ∃ n : ℕ, (∀ (x1 x2 x3 x4 : ℕ), 
    x1 + x2 + x3 + x4 = 4 ∧ x1 ≥ 1 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0 →
    n = 20) :=
sorry

end count_four_digit_numbers_with_digit_sum_4_l50_50345


namespace gerald_jail_time_l50_50901

def jail_time_in_months (assault_months poisoning_years : ℕ) (extension_fraction : ℚ) : ℕ :=
  let poisoning_months := poisoning_years * 12
  let total_months_without_extension := assault_months + poisoning_months
  let extension := (total_months_without_extension : ℚ) * extension_fraction
  total_months_without_extension + (extension.num / extension.denom).toNat

theorem gerald_jail_time : jail_time_in_months 3 2 (1/3) = 36 := by
  sorry

end gerald_jail_time_l50_50901


namespace always_possible_to_rotate_disks_l50_50820

def labels_are_distinct (a : Fin 20 → ℕ) : Prop :=
  ∀ i j : Fin 20, i ≠ j → a i ≠ a j

def opposite_position (i : Fin 20) (r : Fin 20) : Fin 20 :=
  (i + r) % 20

def no_identical_numbers_opposite (a b : Fin 20 → ℕ) (r : Fin 20) : Prop :=
  ∀ i : Fin 20, a i ≠ b (opposite_position i r)

theorem always_possible_to_rotate_disks (a b : Fin 20 → ℕ) :
  labels_are_distinct a →
  labels_are_distinct b →
  ∃ r : Fin 20, no_identical_numbers_opposite a b r :=
sorry

end always_possible_to_rotate_disks_l50_50820


namespace arithmetic_mean_missing_digit_l50_50477

theorem arithmetic_mean_missing_digit :
  let S := {8, 88, 888, 8888, 88888} in
  let M := 17777 in
  ¬ ↑('8') ∈ (finset.digits M) :=
begin
  sorry
end

end arithmetic_mean_missing_digit_l50_50477


namespace common_root_l50_50926

theorem common_root (a b x : ℝ) (h1 : a > b) (h2 : b > 0) 
  (eq1 : x^2 + a * x + b = 0) (eq2 : x^3 + b * x + a = 0) : x = -1 :=
by
  sorry

end common_root_l50_50926


namespace integral_problem1_integral_problem2_integral_problem3_l50_50874

open Real

noncomputable def integral1 := ∫ x in (0 : ℝ)..1, x * exp (-x) = 1 - 2 / exp 1
noncomputable def integral2 := ∫ x in (1 : ℝ)..2, x * log x / log 2 = 2 - 3 / (4 * log 2)
noncomputable def integral3 := ∫ x in (1 : ℝ)..Real.exp 1, (log x) ^ 2 = exp 1 - 2

theorem integral_problem1 : integral1 := sorry
theorem integral_problem2 : integral2 := sorry
theorem integral_problem3 : integral3 := sorry

end integral_problem1_integral_problem2_integral_problem3_l50_50874


namespace minimum_bird_species_l50_50434

theorem minimum_bird_species (total_birds : ℕ) (h : total_birds = 2021) :
  ∃ (min_species : ℕ), min_species = 1011 ∧ 
  (∀ (species_array : array total_birds ℕ),
   ∀ i j : fin total_birds, 
   species_array[i] = species_array[j] → ((i ≠ j) →
   (abs (i - j) mod 2 = 0))) :=
sorry

end minimum_bird_species_l50_50434


namespace possible_values_of_m_l50_50949

open Complex

theorem possible_values_of_m (p q r s m : ℂ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0)
  (h5 : p * m^3 + q * m^2 + r * m + s = 0)
  (h6 : q * m^3 + r * m^2 + s * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I :=
sorry

end possible_values_of_m_l50_50949


namespace greatest_base8_three_digit_divisible_by_7_l50_50681

theorem greatest_base8_three_digit_divisible_by_7 :
  ∃ n : ℕ, n < 8^3 ∧ n ≥ 8^2 ∧ (n % 7 = 0) ∧ (to_base 8 n = 777) :=
sorry

end greatest_base8_three_digit_divisible_by_7_l50_50681


namespace binom_18_6_eq_18564_l50_50540

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end binom_18_6_eq_18564_l50_50540


namespace total_cranes_folded_l50_50418

-- Definitions based on conditions
def hyerinCranesPerDay : ℕ := 16
def hyerinDays : ℕ := 7
def taeyeongCranesPerDay : ℕ := 25
def taeyeongDays : ℕ := 6

-- Definition of total number of cranes folded by Hyerin and Taeyeong
def totalCranes : ℕ :=
  (hyerinCranesPerDay * hyerinDays) + (taeyeongCranesPerDay * taeyeongDays)

-- Proof statement
theorem total_cranes_folded : totalCranes = 262 := by 
  sorry

end total_cranes_folded_l50_50418


namespace pyramid_volume_correct_l50_50098

open EuclideanGeometry

noncomputable def pyramid_volume (S A B C: Point) (BC : ℝ) (B_eq_C : dist S B = dist S C) 
  (H_eq_orthocenter : orthocenter S A B C) (BC_length : BC = 2) (dihedral_angle : ∠(S, B, C) = π / 3) : ℝ :=
  volume_pyramid S A B C = sqrt 3 / 3

theorem pyramid_volume_correct (S A B C: Point) (BC : ℝ) (B_eq_C : dist S B = dist S C)
  (H_eq_orthocenter : orthocenter S A B C) (BC_length : BC = 2) (dihedral_angle : ∠(S, B, C) = π / 3) :
  pyramid_volume S A B C BC B_eq_C H_eq_orthocenter BC_length dihedral_angle = sqrt 3 / 3 :=
sorry

end pyramid_volume_correct_l50_50098


namespace sum_factorials_last_two_digits_l50_50326

/-- Prove that the last two digits of the sum of factorials of the first 15 positive integers equal to 13 --/
theorem sum_factorials_last_two_digits : 
  let f := fun n => Nat.factorial n in
  (f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 + f 11 + f 12 + f 13 + f 14 + f 15) % 100 = 13 :=
by 
  sorry

end sum_factorials_last_two_digits_l50_50326


namespace binom_18_6_eq_18564_l50_50539

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end binom_18_6_eq_18564_l50_50539


namespace ornithological_park_species_l50_50435

/-- In an ornithological park, there are 2021 birds arranged in a row.
Each pair of birds of the same species has an even number of birds between them.
Prove that the smallest number of bird species is 1011. -/
theorem ornithological_park_species (n : ℕ) (h1 : n = 2021) 
  (h2 : ∀ s : ℕ, s ∈ {1..n} → (∀ x y : ℕ, x < y ∧ x ≠ y → (∀ z : ℕ, z ∈ ({x, y} : set ℕ) → even (y - x - 1))) ) 
  : s ≥ 1011 :=
sorry

end ornithological_park_species_l50_50435


namespace tan_double_angle_l50_50256

theorem tan_double_angle (α : ℝ) (h1 : α > 0) (h2 : α < Real.pi)
  (h3 : Real.cos α + Real.sin α = -1 / 5) : Real.tan (2 * α) = -24 / 7 :=
by
  sorry

end tan_double_angle_l50_50256


namespace julie_same_hours_september_october_l50_50451

-- Define Julie's hourly rates and work hours
def rate_mowing : ℝ := 4
def rate_weeding : ℝ := 8
def september_mowing_hours : ℕ := 25
def september_weeding_hours : ℕ := 3
def total_earnings_september_october : ℤ := 248

-- Define Julie's earnings for each activity and total earnings for September
def september_earnings_mowing : ℝ := september_mowing_hours * rate_mowing
def september_earnings_weeding : ℝ := september_weeding_hours * rate_weeding
def september_total_earnings : ℝ := september_earnings_mowing + september_earnings_weeding

-- Define earnings in October
def october_earnings : ℝ := total_earnings_september_october - september_total_earnings

-- Define the theorem to prove Julie worked the same number of hours in October as in September
theorem julie_same_hours_september_october :
  october_earnings = september_total_earnings :=
by
  sorry

end julie_same_hours_september_october_l50_50451


namespace compound_carbon_atoms_l50_50850

-- Definition of data given in the problem.
def molecular_weight : ℕ := 60
def hydrogen_atoms : ℕ := 4
def oxygen_atoms : ℕ := 2
def carbon_atomic_weight : ℕ := 12
def hydrogen_atomic_weight : ℕ := 1
def oxygen_atomic_weight : ℕ := 16

-- Statement to prove the number of carbon atoms in the compound.
theorem compound_carbon_atoms : 
  (molecular_weight - (hydrogen_atoms * hydrogen_atomic_weight + oxygen_atoms * oxygen_atomic_weight)) / carbon_atomic_weight = 2 := 
by
  sorry

end compound_carbon_atoms_l50_50850


namespace square_of_cube_of_third_smallest_prime_l50_50832

theorem square_of_cube_of_third_smallest_prime :
  let p := nat.prime 5
  let cube := p ^ 3
  let square := cube ^ 2
  square = 15625 :=
by
  sorry

end square_of_cube_of_third_smallest_prime_l50_50832


namespace annie_miles_l50_50796

theorem annie_miles (x : ℝ) :
  2.50 + (0.25 * 42) = 2.50 + 5.00 + (0.25 * x) → x = 22 :=
by
  sorry

end annie_miles_l50_50796


namespace lisa_additional_marbles_l50_50457

theorem lisa_additional_marbles (n : ℕ) (f : ℕ) (m : ℕ) (current_marbles : ℕ) : 
  n = 12 ∧ f = n ∧ m = (n * (n + 1)) / 2 ∧ current_marbles = 34 → 
  m - current_marbles = 44 :=
by
  intros
  sorry

end lisa_additional_marbles_l50_50457


namespace emily_101st_card_is_10_of_Hearts_l50_50379

def number_sequence : List String := ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
def suit_sequence : List String := ["Hearts", "Diamonds", "Clubs", "Spades"]

-- Function to get the number of a specific card
def card_number (n : ℕ) : String :=
  number_sequence.get! (n % number_sequence.length)

-- Function to get the suit of a specific card
def card_suit (n : ℕ) : String :=
  suit_sequence.get! ((n / suit_sequence.length) % suit_sequence.length)

-- Definition to state the question and the answer
def emily_card (n : ℕ) : String := card_number n ++ " of " ++ card_suit n

-- Proving that the 101st card is "10 of Hearts"
theorem emily_101st_card_is_10_of_Hearts : emily_card 100 = "10 of Hearts" :=
by {
  sorry
}

end emily_101st_card_is_10_of_Hearts_l50_50379


namespace mabel_counts_sharks_l50_50940

theorem mabel_counts_sharks 
    (fish_day1 : ℕ) 
    (fish_day2 : ℕ) 
    (shark_percentage : ℚ) 
    (total_fish : ℕ) 
    (total_sharks : ℕ) 
    (h1 : fish_day1 = 15) 
    (h2 : fish_day2 = 3 * fish_day1) 
    (h3 : shark_percentage = 0.25) 
    (h4 : total_fish = fish_day1 + fish_day2) 
    (h5 : total_sharks = total_fish * shark_percentage) : 
    total_sharks = 15 := 
by {
  sorry
}

end mabel_counts_sharks_l50_50940


namespace curve_transformation_l50_50067

theorem curve_transformation (x y x' y' : ℝ) :
  (x^2 + y^2 = 1) →
  (x' = 4 * x) →
  (y' = 2 * y) →
  (x'^2 / 16 + y'^2 / 4 = 1) :=
by
  sorry

end curve_transformation_l50_50067


namespace last_two_digits_of_sum_of_factorials_l50_50325

-- Problem statement: Sum of factorials from 1 to 15
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => Nat.factorial k)

-- Define the main problem
theorem last_two_digits_of_sum_of_factorials : 
  (sum_factorials 15) % 100 = 13 :=
by 
  sorry

end last_two_digits_of_sum_of_factorials_l50_50325


namespace math_expression_evaluation_l50_50224

theorem math_expression_evaluation :
  |1 - Real.sqrt 3| + 3 * Real.tan (Real.pi / 6) - (1/2)⁻¹ + (3 - Real.pi)^0 = 3.732 + Real.sqrt 3 := by
  sorry

end math_expression_evaluation_l50_50224


namespace gauss_algorithm_sum_l50_50240

def f (x : Nat) (m : Nat) : Rat := x / (3 * m + 6054)

theorem gauss_algorithm_sum (m : Nat) :
  (Finset.sum (Finset.range (m + 2017 + 1)) (λ x => f x m)) = (m + 2017) / 6 := by
sorry

end gauss_algorithm_sum_l50_50240


namespace larger_number_is_23_l50_50822

theorem larger_number_is_23 (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 6) : a = 23 := 
by
  sorry

end larger_number_is_23_l50_50822


namespace perfect_square_pattern_l50_50732

theorem perfect_square_pattern {a b n : ℕ} (h₁ : 1000 ≤ n ∧ n ≤ 9999)
  (h₂ : n = (10 * a + b) ^ 2)
  (h₃ : n = 1100 * a + 11 * b) : n = 7744 :=
  sorry

end perfect_square_pattern_l50_50732


namespace abs_neg_three_l50_50209

theorem abs_neg_three : abs (-3) = 3 :=
by 
  sorry

end abs_neg_three_l50_50209


namespace james_needs_to_work_50_hours_l50_50612

def wasted_meat := 20
def cost_meat_per_pound := 5
def wasted_vegetables := 15
def cost_vegetables_per_pound := 4
def wasted_bread := 60
def cost_bread_per_pound := 1.5
def janitorial_hours := 10
def janitor_rate := 10
def time_and_half_multiplier := 1.5
def min_wage := 8

theorem james_needs_to_work_50_hours :
  let cost_meat := wasted_meat * cost_meat_per_pound in
  let cost_vegetables := wasted_vegetables * cost_vegetables_per_pound in
  let cost_bread := wasted_bread * cost_bread_per_pound in
  let time_and_half_rate := janitor_rate * time_and_half_multiplier in
  let cost_janitorial := janitorial_hours * time_and_half_rate in
  let total_cost := cost_meat + cost_vegetables + cost_bread + cost_janitorial in
  let hours_to_work := total_cost / min_wage in
  hours_to_work = 50 := by
  sorry

end james_needs_to_work_50_hours_l50_50612


namespace most_likely_outcome_is_draw_l50_50498

noncomputable def prob_A_win : ℝ := 0.3
noncomputable def prob_A_not_lose : ℝ := 0.7
noncomputable def prob_draw : ℝ := prob_A_not_lose - prob_A_win

theorem most_likely_outcome_is_draw :
  prob_draw = 0.4 ∧ prob_draw > prob_A_win ∧ prob_draw > (1 - prob_A_not_lose) :=
by
  -- proof goes here
  sorry

end most_likely_outcome_is_draw_l50_50498


namespace douglas_votes_in_Y_is_46_l50_50205

variable (V : ℝ)
variable (P : ℝ)

def percentage_won_in_Y :=
  let total_voters_X := 2 * V
  let total_voters_Y := V
  let votes_in_X := 0.64 * total_voters_X
  let votes_in_Y := P / 100 * total_voters_Y
  let total_votes := 1.28 * V + (P / 100 * V)
  let combined_voters := 3 * V
  let combined_votes_percentage := 0.58 * combined_voters
  P = 46

theorem douglas_votes_in_Y_is_46
  (V_pos : V > 0)
  (H : 1.28 * V + (P / 100 * V) = 0.58 * 3 * V) :
  percentage_won_in_Y V P := by
  sorry

end douglas_votes_in_Y_is_46_l50_50205


namespace max_gcd_of_consecutive_terms_seq_b_l50_50030

-- Define the sequence b_n
def sequence_b (n : ℕ) : ℕ := n.factorial + 3 * n

-- Define the gcd function for two terms in the sequence
def gcd_two_terms (n : ℕ) : ℕ := Nat.gcd (sequence_b n) (sequence_b (n + 1))

-- Define the condition of n being greater than or equal to 0
def n_ge_zero (n : ℕ) : Prop := n ≥ 0

-- The theorem statement
theorem max_gcd_of_consecutive_terms_seq_b : ∃ n : ℕ, n_ge_zero n ∧ gcd_two_terms n = 14 := 
sorry

end max_gcd_of_consecutive_terms_seq_b_l50_50030


namespace complete_square_variant_l50_50979

theorem complete_square_variant (x : ℝ) :
    3 * x^2 + 4 * x + 1 = 0 → (x + 2 / 3) ^ 2 = 1 / 9 :=
by
  intro h
  sorry

end complete_square_variant_l50_50979


namespace four_digit_perfect_square_l50_50734

theorem four_digit_perfect_square : 
  ∃ (N : ℕ), (1000 ≤ N ∧ N ≤ 9999) ∧ (∃ (a b : ℕ), a = N / 1000 ∧ b = (N % 100) / 10 ∧ a = N / 100 - (N / 100 % 10) ∧ b = (N % 100 / 10) - N % 10) ∧ (∃ (n : ℕ), N = n * n) →
  N = 7744 := 
sorry

end four_digit_perfect_square_l50_50734


namespace greatest_3_digit_base8_num_div_by_7_eq_511_l50_50672

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end greatest_3_digit_base8_num_div_by_7_eq_511_l50_50672


namespace sum_of_squares_of_consecutive_integers_l50_50484

theorem sum_of_squares_of_consecutive_integers (b : ℕ) (h : (b-1) * b * (b+1) = 12 * ((b-1) + b + (b+1))) : 
  (b - 1) * (b - 1) + b * b + (b + 1) * (b + 1) = 110 := 
by sorry

end sum_of_squares_of_consecutive_integers_l50_50484


namespace problem_statement_l50_50602

-- Definitions of the operations △ and ⊗
def triangle (a b : ℤ) : ℤ := a + b + a * b - 1
def otimes (a b : ℤ) : ℤ := a * a - a * b + b * b

-- The theorem statement
theorem problem_statement : triangle 3 (otimes 2 4) = 50 := by
  sorry

end problem_statement_l50_50602


namespace eval_f_g_at_4_l50_50789

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 12 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem eval_f_g_at_4 : f (g 4) = (25 / 7) * Real.sqrt 21 := by
  sorry

end eval_f_g_at_4_l50_50789


namespace max_days_for_same_shift_l50_50352

open BigOperators

-- We define the given conditions
def nurses : ℕ := 15
def shifts_per_day : ℕ := 24 / 8
noncomputable def total_pairs : ℕ := (nurses.choose 2)

-- The main statement to prove
theorem max_days_for_same_shift : 
  35 = total_pairs / shifts_per_day := by
  sorry

end max_days_for_same_shift_l50_50352


namespace modified_determinant_l50_50599

def determinant_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem modified_determinant (x y z w : ℝ)
  (h : determinant_2x2 x y z w = 6) :
  determinant_2x2 x (5 * x + 4 * y) z (5 * z + 4 * w) = 24 := by
  sorry

end modified_determinant_l50_50599


namespace greatest_number_of_matching_pairs_l50_50632

theorem greatest_number_of_matching_pairs 
  (original_pairs : ℕ := 27)
  (lost_shoes : ℕ := 9) 
  (remaining_pairs : ℕ := original_pairs - (lost_shoes / 1))
  : remaining_pairs = 18 := by
  sorry

end greatest_number_of_matching_pairs_l50_50632


namespace bakery_used_0_2_bags_of_wheat_flour_l50_50335

-- Define the conditions
def total_flour := 0.3
def white_flour := 0.1

-- Define the number of bags of wheat flour used
def wheat_flour := total_flour - white_flour

-- The proof statement
theorem bakery_used_0_2_bags_of_wheat_flour : wheat_flour = 0.2 := 
by
  sorry

end bakery_used_0_2_bags_of_wheat_flour_l50_50335


namespace express_x_in_terms_of_y_l50_50557

theorem express_x_in_terms_of_y (x y : ℝ) (h : 2 * x - 3 * y = 7) : x = 7 / 2 + 3 / 2 * y :=
by
  sorry

end express_x_in_terms_of_y_l50_50557


namespace initial_bedbugs_l50_50856

theorem initial_bedbugs (x : ℕ) (h_triple : ∀ n : ℕ, bedbugs (n + 1) = 3 * bedbugs n) 
  (h_fourth_day : bedbugs 4 = 810) : bedbugs 0 = 30 :=
by {
  sorry
}

end initial_bedbugs_l50_50856


namespace alyssa_photos_vacation_l50_50366

theorem alyssa_photos_vacation
  (pages_first_section : ℕ)
  (photos_per_page_first_section : ℕ)
  (pages_second_section : ℕ)
  (photos_per_page_second_section : ℕ)
  (pages_total : ℕ)
  (photos_per_page_remaining : ℕ)
  (pages_remaining : ℕ)
  (h_total_pages : pages_first_section + pages_second_section + pages_remaining = pages_total)
  (h_photos_first_section : photos_per_page_first_section = 3)
  (h_photos_second_section : photos_per_page_second_section = 4)
  (h_pages_first_section : pages_first_section = 10)
  (h_pages_second_section : pages_second_section = 10)
  (h_photos_remaining : photos_per_page_remaining = 3)
  (h_pages_total : pages_total = 30)
  (h_pages_remaining : pages_remaining = 10) :
  pages_first_section * photos_per_page_first_section +
  pages_second_section * photos_per_page_second_section +
  pages_remaining * photos_per_page_remaining = 100 := by
sorry

end alyssa_photos_vacation_l50_50366


namespace bicycles_purchased_on_Friday_l50_50290

theorem bicycles_purchased_on_Friday (F : ℕ) : (F - 10) - 4 + 2 = 3 → F = 15 := by
  intro h
  sorry

end bicycles_purchased_on_Friday_l50_50290


namespace circle_ratio_l50_50993

theorem circle_ratio (R r a c : ℝ) (hR : 0 < R) (hr : 0 < r) (h_c_lt_a : 0 < c ∧ c < a) 
  (condition : π * R^2 = (a - c) * (π * R^2 - π * r^2)) :
  R / r = Real.sqrt ((a - c) / (c + 1 - a)) :=
by
  sorry

end circle_ratio_l50_50993


namespace combinations_count_l50_50459

def colorChoices := 4
def decorationChoices := 3
def methodChoices := 3

theorem combinations_count : colorChoices * decorationChoices * methodChoices = 36 := by
  sorry

end combinations_count_l50_50459


namespace items_per_baggie_l50_50152

def num_pretzels : ℕ := 64
def num_suckers : ℕ := 32
def num_kids : ℕ := 16
def num_goldfish : ℕ := 4 * num_pretzels
def total_items : ℕ := num_pretzels + num_goldfish + num_suckers

theorem items_per_baggie : total_items / num_kids = 22 :=
by
  -- Calculation proof
  sorry

end items_per_baggie_l50_50152


namespace cyclist_motorcyclist_intersection_l50_50986

theorem cyclist_motorcyclist_intersection : 
  ∃ t : ℝ, (4 * t^2 + (t - 1)^2 - 2 * |t| * |t - 1| = 49) ∧ (t = 4 ∨ t = -4) := 
by 
  sorry

end cyclist_motorcyclist_intersection_l50_50986


namespace consecutive_odd_numbers_square_difference_l50_50109

theorem consecutive_odd_numbers_square_difference (a b : ℤ) :
  (a - b = 2 ∨ b - a = 2) → (a^2 - b^2 = 2000) → (a = 501 ∧ b = 499 ∨ a = -501 ∧ b = -499) :=
by 
  intros h1 h2
  sorry

end consecutive_odd_numbers_square_difference_l50_50109


namespace percentage_of_burpees_is_10_l50_50724

-- Definitions for each exercise count
def jumping_jacks : ℕ := 25
def pushups : ℕ := 15
def situps : ℕ := 30
def burpees : ℕ := 10
def lunges : ℕ := 20

-- Total number of exercises
def total_exercises : ℕ := jumping_jacks + pushups + situps + burpees + lunges

-- The proof statement
theorem percentage_of_burpees_is_10 :
  (burpees * 100) / total_exercises = 10 :=
by
  sorry

end percentage_of_burpees_is_10_l50_50724


namespace binom_18_6_eq_4765_l50_50545

def binom (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_18_6_eq_4765 : binom 18 6 = 4765 := by
  sorry

end binom_18_6_eq_4765_l50_50545


namespace sin_pi_add_x_pos_l50_50183

open Real
open Int

theorem sin_pi_add_x_pos (k : ℤ) (x : ℝ) :
  (2 * k * π + π < x ∧ x < 2 * k * π + 2 * π) ↔ sin (π + x) > 0 :=
sorry

end sin_pi_add_x_pos_l50_50183


namespace count_four_digit_numbers_divisible_by_17_and_end_in_17_l50_50755

theorem count_four_digit_numbers_divisible_by_17_and_end_in_17 :
  ∃ S : Finset ℕ, S.card = 5 ∧ ∀ n ∈ S, 1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0 ∧ n % 100 = 17 :=
by
  sorry

end count_four_digit_numbers_divisible_by_17_and_end_in_17_l50_50755


namespace number_of_terms_in_sequence_l50_50115

theorem number_of_terms_in_sequence :
  ∃ n : ℕ, (1 + 4 * (n - 1) = 2025) ∧ n = 507 := by
  sorry

end number_of_terms_in_sequence_l50_50115


namespace smallest_integer_ending_in_9_and_divisible_by_11_l50_50190

theorem smallest_integer_ending_in_9_and_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 9 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n :=
  sorry

end smallest_integer_ending_in_9_and_divisible_by_11_l50_50190


namespace probability_product_multiple_of_4_l50_50596

theorem probability_product_multiple_of_4 :
  let cards := {1, 2, 3, 4, 5, 6}
  let pairs := { (a, b) | a ∈ cards ∧ b ∈ cards ∧ a < b }
  let total_pairs := 15
  let valid_pairs := { (1, 4), (2, 4), (3, 4), (4, 5), (4, 6) }
  let num_valid_pairs := 5
  num_valid_pairs / total_pairs = 1 / 3 := by
  sorry

end probability_product_multiple_of_4_l50_50596


namespace smallest_denominator_of_sum_of_irreducible_fractions_l50_50480

theorem smallest_denominator_of_sum_of_irreducible_fractions :
  ∀ (a b : ℕ),
  Nat.Coprime a 600 → Nat.Coprime b 700 →
  (∃ c d : ℕ, Nat.Coprime c d ∧ d < 168 ∧ (7 * a + 6 * b) / Nat.gcd (7 * a + 6 * b) 4200 = c / d) →
  False :=
by
  sorry

end smallest_denominator_of_sum_of_irreducible_fractions_l50_50480


namespace find_value_of_p_l50_50589

-- Definition of the parabola and ellipse
def parabola (p : ℝ) : Set (ℝ × ℝ) := {xy | xy.1 ^ 2 = 2 * p * xy.2}
def ellipse : Set (ℝ × ℝ) := {xy | xy.1 ^ 2 / 6 + xy.2 ^ 2 / 4 = 1}

-- Hypotheses
variables (p : ℝ) (h_pos : p > 0)

-- Latus rectum tangent to the ellipse
theorem find_value_of_p (h_tangent : ∃ (x y : ℝ),
  (parabola p (x, y) ∧ ellipse (x, y) ∧ y = -p / 2)) : p = 4 := sorry

end find_value_of_p_l50_50589


namespace smallest_integer_y_l50_50827

theorem smallest_integer_y : ∃ (y : ℕ), (\frac{8}{11} < \frac{y}{17}) ∧ y = 13 :=
by
  sorry

end smallest_integer_y_l50_50827


namespace total_profit_l50_50217

variable (A_s B_s C_s : ℝ)
variable (A_p : ℝ := 14700)
variable (P : ℝ)

theorem total_profit
  (h1 : A_s + B_s + C_s = 50000)
  (h2 : A_s = B_s + 4000)
  (h3 : B_s = C_s + 5000)
  (h4 : A_p = 14700) :
  P = 35000 :=
sorry

end total_profit_l50_50217


namespace complex_power_identity_l50_50714

theorem complex_power_identity (z : ℂ) (i : ℂ) 
  (h1 : z = (1 + i) / Real.sqrt 2) 
  (h2 : z^2 = i) : 
  z^100 = -1 := 
  sorry

end complex_power_identity_l50_50714


namespace abs_diff_eq_l50_50016

-- Define the conditions
variables (x y : ℝ)
axiom h1 : x + y = 30
axiom h2 : x * y = 162

-- Define the problem to prove
theorem abs_diff_eq : |x - y| = 6 * Real.sqrt 7 :=
by sorry

end abs_diff_eq_l50_50016


namespace arithmetic_sequence_sum_l50_50776

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a4 : a 4 = 3) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
sorry

end arithmetic_sequence_sum_l50_50776


namespace solution_set_of_inequality_l50_50398

theorem solution_set_of_inequality :
  { x : ℝ | x ≠ 5 ∧ (x * (x + 1)) / ((x - 5) ^ 3) ≥ 25 } = 
  { x : ℝ | x ≤ 5 / 3 } ∪ { x : ℝ | x > 5 } := by
  sorry

end solution_set_of_inequality_l50_50398


namespace greatest_number_of_dimes_l50_50470

-- Definitions according to the conditions in a)
def total_value_in_cents : ℤ := 485
def dime_value_in_cents : ℤ := 10
def nickel_value_in_cents : ℤ := 5

-- The proof problem in Lean 4
theorem greatest_number_of_dimes : 
  ∃ (d : ℤ), (dime_value_in_cents * d + nickel_value_in_cents * d = total_value_in_cents) ∧ d = 32 := 
by
  sorry

end greatest_number_of_dimes_l50_50470


namespace table_ratio_l50_50357

theorem table_ratio (L W : ℝ) (h1 : L * W = 128) (h2 : L + 2 * W = 32) : L / W = 2 :=
by
  sorry

end table_ratio_l50_50357


namespace points_not_all_odd_distance_l50_50472

open Real

theorem points_not_all_odd_distance (p : Fin 4 → ℝ × ℝ) : ∃ i j : Fin 4, i ≠ j ∧ ¬ Odd (dist (p i) (p j)) := 
by
  sorry

end points_not_all_odd_distance_l50_50472


namespace residue_5_pow_1234_mod_13_l50_50497

theorem residue_5_pow_1234_mod_13 : ∃ k : ℤ, 5^1234 = 13 * k + 12 :=
by
  sorry

end residue_5_pow_1234_mod_13_l50_50497


namespace willie_final_stickers_l50_50199

-- Conditions
def willie_start_stickers : ℝ := 36.0
def emily_gives_willie : ℝ := 7.0

-- Theorem
theorem willie_final_stickers : willie_start_stickers + emily_gives_willie = 43.0 :=
by
  sorry

end willie_final_stickers_l50_50199


namespace total_area_correct_l50_50278

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase : ℕ := 2
def num_extra_rooms_same_size : ℕ := 3
def num_double_size_rooms : ℕ := 1

def increased_length : ℕ := initial_length + increase
def increased_width : ℕ := initial_width + increase

def area_of_one_room (length width : ℕ) : ℕ := length * width

def number_of_rooms : ℕ := 1 + num_extra_rooms_same_size

def total_area_same_size_rooms : ℕ := number_of_rooms * area_of_one_room increased_length increased_width

def total_area_extra_size_room : ℕ := num_double_size_rooms * 2 * area_of_one_room increased_length increased_width

def total_area : ℕ := total_area_same_size_rooms + total_area_extra_size_room

theorem total_area_correct : total_area = 1800 := by
  -- Proof omitted
  sorry

end total_area_correct_l50_50278


namespace smallest_four_digit_multiple_of_37_l50_50032

theorem smallest_four_digit_multiple_of_37 : ∃ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 ∧ 37 ∣ n ∧ (∀ m : ℕ, m ≥ 1000 ∧ m ≤ 9999 ∧ 37 ∣ m → n ≤ m) ∧ n = 1036 :=
by
  sorry

end smallest_four_digit_multiple_of_37_l50_50032


namespace quadratic_inequality_solution_l50_50262

theorem quadratic_inequality_solution (a b : ℝ)
  (h1 : ∀ x, (x > -1 ∧ x < 2) ↔ ax^2 + x + b > 0) :
  a + b = 1 :=
sorry

end quadratic_inequality_solution_l50_50262


namespace minimum_radius_third_sphere_l50_50704

-- Definitions for the problem
def height_cone := 4
def base_radius_cone := 3
def cos_alpha := 4 / 5
def radius_identical_sphere := 4 / 3
def cos_beta := 1 -- since beta is maximized

-- Define the required minimum radius for the third sphere based on the given conditions
theorem minimum_radius_third_sphere :
  ∃ x : ℝ, x = 27 / 35 ∧
    (height_cone = 4) ∧ 
    (base_radius_cone = 3) ∧ 
    (cos_alpha = 4 / 5) ∧ 
    (radius_identical_sphere = 4 / 3) ∧ 
    (cos_beta = 1) :=
sorry

end minimum_radius_third_sphere_l50_50704


namespace eval_expr_ceil_floor_l50_50395

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end eval_expr_ceil_floor_l50_50395


namespace op_4_neg3_eq_neg28_l50_50376

def op (x y : Int) : Int := x * (y + 2) + 2 * x * y

theorem op_4_neg3_eq_neg28 : op 4 (-3) = -28 := by
  sorry

end op_4_neg3_eq_neg28_l50_50376


namespace units_digit_of_fraction_l50_50197

theorem units_digit_of_fraction :
  let numer := 30 * 31 * 32 * 33 * 34 * 35
  let denom := 1000
  (numer / denom) % 10 = 6 :=
by
  sorry

end units_digit_of_fraction_l50_50197


namespace no_three_integers_exist_l50_50886

theorem no_three_integers_exist (x y z : ℤ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  ((x^2 - 1) % y = 0) ∧ ((x^2 - 1) % z = 0) ∧
  ((y^2 - 1) % x = 0) ∧ ((y^2 - 1) % z = 0) ∧
  ((z^2 - 1) % x = 0) ∧ ((z^2 - 1) % y = 0) → false :=
by
  sorry

end no_three_integers_exist_l50_50886


namespace set_equality_l50_50408

open Set

variable (A : Set ℕ)

theorem set_equality (h1 : {1, 3} ⊆ A) (h2 : {1, 3} ∪ A = {1, 3, 5}) : A = {1, 3, 5} :=
sorry

end set_equality_l50_50408


namespace expected_attempts_for_10_suitcases_l50_50994

noncomputable def expected_attempts (n : ℕ) : ℝ :=
  (1 / 2) * (n * (n + 1) / 2) + (n / 2) - (Real.log n + 0.577)

theorem expected_attempts_for_10_suitcases :
  abs (expected_attempts 10 - 29.62) < 1 :=
by
  sorry

end expected_attempts_for_10_suitcases_l50_50994


namespace area_ratio_parallelogram_to_triangle_l50_50818

variables {A B C D R E : Type*}
variables (s_AB s_AD : ℝ)

-- Given AR = 2/3 AB and AE = 1/3 AD
axiom AR_proportion : s_AB > 0 → s_AB * (2/3) = s_AB
axiom AE_proportion : s_AD > 0 → s_AD * (1/3) = s_AD

-- Given the relationship, we need to prove
theorem area_ratio_parallelogram_to_triangle (hAB : s_AB > 0) (hAD : s_AD > 0) :
  ∃ (S_ABCD S_ARE : ℝ), S_ABCD / S_ARE = 9 :=
by {
  sorry
}

end area_ratio_parallelogram_to_triangle_l50_50818


namespace amount_a_receives_l50_50340

theorem amount_a_receives (a b c : ℕ) (h1 : a + b + c = 50000) (h2 : a = b + 4000) (h3 : b = c + 5000) :
  (21000 / 50000) * 36000 = 15120 :=
by
  sorry

end amount_a_receives_l50_50340


namespace abs_neg_three_l50_50206

theorem abs_neg_three : |(-3 : ℝ)| = 3 := 
by
  -- The proof would go here, but we skip it for this exercise.
  sorry

end abs_neg_three_l50_50206


namespace tickets_required_l50_50981

theorem tickets_required (cost_ferris_wheel : ℝ) (cost_roller_coaster : ℝ) 
  (discount_multiple_rides : ℝ) (coupon_value : ℝ) 
  (total_cost_with_discounts : ℝ) : 
  cost_ferris_wheel = 2.0 ∧ 
  cost_roller_coaster = 7.0 ∧ 
  discount_multiple_rides = 1.0 ∧ 
  coupon_value = 1.0 → 
  total_cost_with_discounts = 7.0 :=
by
  sorry

end tickets_required_l50_50981


namespace ratio_of_d_to_s_l50_50226

theorem ratio_of_d_to_s (s d : ℝ) (n : ℕ) (h1 : n = 15) (h2 : (n^2 * s^2) / ((n * s + 2 * n * d)^2) = 0.75) :
  d / s = 1 / 13 :=
by
  sorry

end ratio_of_d_to_s_l50_50226


namespace probability_all_selected_l50_50186

theorem probability_all_selected (P_Ram P_Ravi P_Ritu : ℚ) 
  (h1 : P_Ram = 3 / 7) 
  (h2 : P_Ravi = 1 / 5) 
  (h3 : P_Ritu = 2 / 9) : 
  P_Ram * P_Ravi * P_Ritu = 2 / 105 := 
by
  sorry

end probability_all_selected_l50_50186


namespace segment_AB_length_l50_50611

-- Defining the conditions
def area_ratio (AB CD : ℝ) : Prop := AB / CD = 5 / 2
def length_sum (AB CD : ℝ) : Prop := AB + CD = 280

-- The theorem stating the problem
theorem segment_AB_length (AB CD : ℝ) (h₁ : area_ratio AB CD) (h₂ : length_sum AB CD) : AB = 200 :=
by {
  -- Proof step would be inserted here, but it is omitted as per instructions
  sorry
}

end segment_AB_length_l50_50611


namespace kate_candy_l50_50370

variable (K : ℕ)
variable (R : ℕ) (B : ℕ) (M : ℕ)

-- Define the conditions
def robert_pieces := R = K + 2
def mary_pieces := M = R + 2
def bill_pieces := B = M - 6
def total_pieces := K + R + M + B = 20

-- The theorem to prove
theorem kate_candy :
  ∃ (K : ℕ), robert_pieces K R ∧ mary_pieces R M ∧ bill_pieces M B ∧ total_pieces K R M B ∧ K = 4 :=
sorry

end kate_candy_l50_50370


namespace sequence_term_sequence_sum_l50_50664

def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else 3^(n-1)

def S_n (n : ℕ) : ℕ :=
  (3^n - 1) / 2

theorem sequence_term (n : ℕ) (h : n ≥ 1) :
  a_seq n = 3^(n-1) :=
sorry

theorem sequence_sum (n : ℕ) :
  S_n n = (3^n - 1) / 2 :=
sorry

end sequence_term_sequence_sum_l50_50664


namespace men_l50_50690

namespace WagesProblem

def men_women_boys_equivalence (man woman boy : ℕ) : Prop :=
  9 * man = woman ∧ woman = 7 * boy

def total_earnings (man woman boy earnings : ℕ) : Prop :=
  (9 * man + woman + woman) = earnings ∧ earnings = 216

theorem men's_wages (man woman boy : ℕ) (h1 : men_women_boys_equivalence man woman boy) (h2 : total_earnings man woman 7 216) : 9 * man = 72 :=
sorry

end WagesProblem

end men_l50_50690


namespace more_people_attended_l50_50023

def saturday_attendance := 80
def monday_attendance := saturday_attendance - 20
def wednesday_attendance := monday_attendance + 50
def friday_attendance := saturday_attendance + monday_attendance
def expected_audience := 350

theorem more_people_attended :
  saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance - expected_audience = 40 :=
by
  sorry

end more_people_attended_l50_50023


namespace eval_expr_ceil_floor_l50_50394

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end eval_expr_ceil_floor_l50_50394


namespace binom_18_6_l50_50533

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_l50_50533


namespace value_of_k_l50_50903

theorem value_of_k (a b k : ℝ) (h1 : 2 * a = k) (h2 : 3 * b = k) (h3 : 2 * a + b = a * b) (h4 : k ≠ 1) : k = 8 := 
sorry

end value_of_k_l50_50903


namespace least_possible_value_of_one_integer_l50_50840

theorem least_possible_value_of_one_integer (
  A B C D E F : ℤ
) (h1 : (A + B + C + D + E + F) / 6 = 63)
  (h2 : A ≤ 100 ∧ B ≤ 100 ∧ C ≤ 100 ∧ D ≤ 100 ∧ E ≤ 100 ∧ F ≤ 100)
  (h3 : (A + B + C) / 3 = 65) : 
  ∃ D E F, (D + E + F) = 183 ∧ min D (min E F) = 83 := sorry

end least_possible_value_of_one_integer_l50_50840


namespace simplify_complex_fraction_l50_50648

theorem simplify_complex_fraction :
  (⟨3, 5⟩ : ℂ) / (⟨-2, 7⟩ : ℂ) = (29 / 53) - (31 / 53) * I :=
by sorry

end simplify_complex_fraction_l50_50648


namespace john_tran_probability_2_9_l50_50319

def johnArrivalProbability (train_start train_end john_min john_max: ℕ) : ℚ := 
  let overlap_area := ((train_end - train_start - 15) * 15) / 2 
  let total_area := (john_max - john_min) * (train_end - train_start)
  overlap_area / total_area

theorem john_tran_probability_2_9 :
  johnArrivalProbability 30 90 0 90 = 2 / 9 := by
  sorry

end john_tran_probability_2_9_l50_50319


namespace combination_18_6_l50_50542

theorem combination_18_6 : (nat.choose 18 6) = 18564 := 
by 
  sorry

end combination_18_6_l50_50542


namespace Mark_running_speed_l50_50286

theorem Mark_running_speed
    (x : ℝ)
    (h_biking : 15 / (3 * x + 2))
    (h_running : 3 / x)
    (h_total_time : h_biking + h_running = 1.47) :
    x ≈ 5.05 :=
by
  sorry

end Mark_running_speed_l50_50286


namespace friends_count_l50_50351

-- Define the given conditions
def initial_chicken_wings := 2
def additional_chicken_wings := 25
def chicken_wings_per_person := 3

-- Define the total number of chicken wings
def total_chicken_wings := initial_chicken_wings + additional_chicken_wings

-- Define the target number of friends in the group
def number_of_friends := total_chicken_wings / chicken_wings_per_person

-- The theorem stating that the number of friends is 9
theorem friends_count : number_of_friends = 9 := by
  sorry

end friends_count_l50_50351


namespace width_of_river_l50_50869

def river_depth : ℝ := 7
def flow_rate_kmph : ℝ := 4
def volume_per_minute : ℝ := 35000

noncomputable def flow_rate_mpm : ℝ := (flow_rate_kmph * 1000) / 60

theorem width_of_river : 
  ∃ w : ℝ, 
    volume_per_minute = flow_rate_mpm * river_depth * w ∧
    w = 75 :=
by
  use 75
  field_simp [flow_rate_mpm, river_depth, volume_per_minute]
  norm_num
  sorry

end width_of_river_l50_50869


namespace bill_took_six_naps_l50_50530

def total_hours (days : Nat) : Nat := days * 24

def hours_left (total : Nat) (worked : Nat) : Nat := total - worked

def naps_taken (remaining : Nat) (duration : Nat) : Nat := remaining / duration

theorem bill_took_six_naps :
  let days := 4
  let hours_worked := 54
  let nap_duration := 7
  naps_taken (hours_left (total_hours days) hours_worked) nap_duration = 6 := 
by {
  sorry
}

end bill_took_six_naps_l50_50530


namespace perpendicular_condition_l50_50308

theorem perpendicular_condition (m : ℝ) : 
  (2 * (m + 1) * (m - 3) + 2 * (m - 3) = 0) ↔ (m = 3 ∨ m = -3) :=
by
  sorry

end perpendicular_condition_l50_50308


namespace max_sum_x_y_under_condition_l50_50454

-- Define the conditions
variables (x y : ℝ)

-- State the problem and what needs to be proven
theorem max_sum_x_y_under_condition : 
  (3 * (x^2 + y^2) = x - y) → (x + y) ≤ (1 / Real.sqrt 2) :=
by
  sorry

end max_sum_x_y_under_condition_l50_50454


namespace range_of_m_l50_50094

-- Definition of the quadratic function
def quadratic_function (m x : ℝ) : ℝ :=
  x^2 + (m - 1) * x + 1

-- Statement of the proof problem in Lean
theorem range_of_m (m : ℝ) : 
  (∀ x : ℤ, 0 ≤ x ∧ x ≤ 5 → quadratic_function m x ≥ quadratic_function m (x + 1)) ↔ m ≤ -8 :=
by
  sorry

end range_of_m_l50_50094


namespace quadratic_range_l50_50013

noncomputable def f : ℝ → ℝ := sorry -- Quadratic function with a positive coefficient for its quadratic term

axiom symmetry_condition : ∀ x : ℝ, f x = f (4 - x)

theorem quadratic_range (x : ℝ) (h1 : f (1 - 2 * x ^ 2) < f (1 + 2 * x - x ^ 2)) : -2 < x ∧ x < 0 :=
by sorry

end quadratic_range_l50_50013


namespace digit_Q_is_0_l50_50103

theorem digit_Q_is_0 (M N P Q : ℕ) (hM : M < 10) (hN : N < 10) (hP : P < 10) (hQ : Q < 10) 
  (add_eq : 10 * M + N + 10 * P + M = 10 * Q + N) 
  (sub_eq : 10 * M + N - (10 * P + M) = N) : Q = 0 := 
by
  sorry

end digit_Q_is_0_l50_50103


namespace balls_picking_l50_50316

theorem balls_picking (red_bag blue_bag : ℕ) (h_red : red_bag = 3) (h_blue : blue_bag = 5) : (red_bag * blue_bag = 15) :=
by
  sorry

end balls_picking_l50_50316


namespace distance_from_y_axis_l50_50302

theorem distance_from_y_axis (dx dy : ℝ) (h1 : dx = 8) (h2 : dx = (1/2) * dy) : dy = 16 :=
by
  sorry

end distance_from_y_axis_l50_50302


namespace ninth_term_arithmetic_sequence_l50_50430

theorem ninth_term_arithmetic_sequence :
  ∃ (a d : ℤ), (a + 2 * d = 5 ∧ a + 5 * d = 17) ∧ (a + 8 * d = 29) := 
by
  sorry

end ninth_term_arithmetic_sequence_l50_50430


namespace probability_neither_red_nor_purple_l50_50982

theorem probability_neither_red_nor_purple
  (total_balls : ℕ)
  (white_balls : ℕ)
  (green_balls : ℕ)
  (yellow_balls : ℕ)
  (red_balls : ℕ)
  (purple_balls : ℕ)
  (h_total : total_balls = 60)
  (h_white : white_balls = 22)
  (h_green : green_balls = 18)
  (h_yellow : yellow_balls = 17)
  (h_red : red_balls = 3)
  (h_purple : purple_balls = 1) :
  ((total_balls - red_balls - purple_balls) / total_balls : ℚ) = 14 / 15 :=
by
  sorry

end probability_neither_red_nor_purple_l50_50982


namespace expression_even_nat_l50_50463

theorem expression_even_nat (m n : ℕ) : 
  2 ∣ (5 * m + n + 1) * (3 * m - n + 4) := 
sorry

end expression_even_nat_l50_50463


namespace negation_example_l50_50660

theorem negation_example :
  (¬ (∀ x : ℝ, abs (x - 2) + abs (x - 4) > 3)) ↔ (∃ x : ℝ, abs (x - 2) + abs (x - 4) ≤ 3) :=
by
  sorry

end negation_example_l50_50660


namespace greatest_3digit_base8_divisible_by_7_l50_50671

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), n = 0b777 ∧ (base8_to_base10 0b777) % 7 = 0 ∧ ∀ m < 0o777, m % 7 = 0 → base8_to_base10 m < base8_to_base10 0b777 :=
by
  sorry

end greatest_3digit_base8_divisible_by_7_l50_50671


namespace sum_product_poly_roots_eq_l50_50915

theorem sum_product_poly_roots_eq (b c : ℝ) 
  (h1 : -1 + 2 = -b) 
  (h2 : (-1) * 2 = c) : c + b = -3 := 
by 
  sorry

end sum_product_poly_roots_eq_l50_50915


namespace sum_of_squares_l50_50938

theorem sum_of_squares (x y z : ℤ) (h1 : x + y + z = 3) (h2 : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 :=
sorry

end sum_of_squares_l50_50938


namespace super_rare_snake_cost_multiple_l50_50447

noncomputable def price_of_regular_snake : ℕ := 250
noncomputable def total_money_obtained : ℕ := 2250
noncomputable def number_of_snakes : ℕ := 3
noncomputable def eggs_per_snake : ℕ := 2

theorem super_rare_snake_cost_multiple :
  (total_money_obtained - (number_of_snakes * eggs_per_snake - 1) * price_of_regular_snake) / price_of_regular_snake = 4 :=
by
  sorry

end super_rare_snake_cost_multiple_l50_50447


namespace sonnets_not_read_l50_50113

-- Define the conditions in the original problem
def sonnet_lines := 14
def unheard_lines := 70

-- Define a statement that needs to be proven
-- Prove that the number of sonnets not read is 5
theorem sonnets_not_read : unheard_lines / sonnet_lines = 5 := by
  sorry

end sonnets_not_read_l50_50113


namespace car_distance_proof_l50_50348

variable (D T : ℝ)
variable (h1 : D = 70 * T)
variable (h2 : D = 105 * (T - 0.5))

theorem car_distance_proof (h1 : D = 70 * T) (h2 : D = 105 * (T - 0.5)) : D = 105 := by
  -- Begin the proof here
  sorry

end car_distance_proof_l50_50348


namespace last_two_digits_of_sum_of_first_15_factorials_is_13_l50_50327

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (k+1) => (k+1) * factorial k

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_factorials (n : ℕ) : ℕ :=
  (Nat.fold (fun acc x => acc + factorial x) 0 (List.range n))

theorem last_two_digits_of_sum_of_first_15_factorials_is_13 :
  last_two_digits (sum_of_factorials 15) = 13 :=
by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_is_13_l50_50327


namespace sqrt_meaningful_iff_l50_50772

theorem sqrt_meaningful_iff (x : ℝ) : (3 - x ≥ 0) ↔ (x ≤ 3) := by
  sorry

end sqrt_meaningful_iff_l50_50772


namespace length_of_generatrix_l50_50581

/-- Given that the base radius of a cone is sqrt(2), and its lateral surface is unfolded into a semicircle,
prove that the length of the generatrix of the cone is 2 sqrt(2). -/
theorem length_of_generatrix (r l : ℝ) (h1 : r = Real.sqrt 2)
    (h2 : 2 * Real.pi * r = Real.pi * l) : l = 2 * Real.sqrt 2 :=
by
  sorry

end length_of_generatrix_l50_50581


namespace eval_ceil_floor_sum_l50_50382

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end eval_ceil_floor_sum_l50_50382


namespace solve_ordered_pair_l50_50231

theorem solve_ordered_pair : ∃ (x y : ℚ), 3*x - 24*y = 3 ∧ x - 3*y = 4 ∧ x = 29/5 ∧ y = 3/5 := by
  sorry

end solve_ordered_pair_l50_50231


namespace decreasing_interval_range_l50_50587

theorem decreasing_interval_range (a : ℝ) :
  (∀ x y ∈ Ioo 0 1, x < y → 2^(x * (x-a)) > 2^(y * (y-a))) ↔ a ≥ 2 :=
by
  sorry

end decreasing_interval_range_l50_50587


namespace combination_18_6_l50_50543

theorem combination_18_6 : (nat.choose 18 6) = 18564 := 
by 
  sorry

end combination_18_6_l50_50543


namespace jake_watched_friday_l50_50133

theorem jake_watched_friday
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (thursday_hours : ℕ)
  (total_hours : ℕ)
  (day_hours : ℕ := 24) :
  monday_hours = (day_hours / 2) →
  tuesday_hours = 4 →
  wednesday_hours = (day_hours / 4) →
  thursday_hours = ((monday_hours + tuesday_hours + wednesday_hours) / 2) →
  total_hours = 52 →
  (total_hours - (monday_hours + tuesday_hours + wednesday_hours + thursday_hours)) = 19 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jake_watched_friday_l50_50133


namespace decreasing_on_interval_l50_50588

noncomputable def f (a x : ℝ) : ℝ := 2^(x * (x - a))

theorem decreasing_on_interval (a : ℝ) : (a ≥ 2) ↔ ∀ x ∈ Set.Ioo 0 1, (deriv (λ x, 2^(x * (x - a)))) x ≤ 0 :=
sorry

end decreasing_on_interval_l50_50588


namespace find_a_b_find_tangent_line_l50_50455

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := 2 * x ^ 3 + 3 * a * x ^ 2 + 3 * b * x + 8

-- Define the derivative of the function f(x)
def f' (a b x : ℝ) : ℝ := 6 * x ^ 2 + 6 * a * x + 3 * b

-- Define the conditions for extreme values at x=1 and x=2
def extreme_conditions (a b : ℝ) : Prop :=
  f' a b 1 = 0 ∧ f' a b 2 = 0

-- Prove the values of a and b
theorem find_a_b (a b : ℝ) (h : extreme_conditions a b) : a = -3 ∧ b = 4 :=
by sorry

-- Find the equation of the tangent line at x=0
def tangent_equation (a b : ℝ) (x y : ℝ) : Prop :=
  12 * x - y + 8 = 0

-- Prove the equation of the tangent line
theorem find_tangent_line (a b : ℝ) (h : extreme_conditions a b) : tangent_equation a b 0 8 :=
by sorry

end find_a_b_find_tangent_line_l50_50455


namespace line_eq_l50_50058

theorem line_eq (x_1 y_1 x_2 y_2 : ℝ) (h1 : x_1 + x_2 = 8) (h2 : y_1 + y_2 = 2)
  (h3 : x_1^2 - 4 * y_1^2 = 4) (h4 : x_2^2 - 4 * y_2^2 = 4) :
  ∃ l : ℝ, ∀ x y : ℝ, x - y - 3 = l :=
by sorry

end line_eq_l50_50058


namespace ellipse_hyperbola_tangent_l50_50954

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 → x^2 - m * (y - 1)^2 = 4) →
  (m = 6 ∨ m = 12) := by
  sorry

end ellipse_hyperbola_tangent_l50_50954


namespace minimum_value_is_1_l50_50825

def minimum_value_expression (x y : ℝ) : ℝ :=
  x^2 + y^2 - 8*x + 6*y + 26

theorem minimum_value_is_1 (x y : ℝ) (h : x ≥ 4) : 
  minimum_value_expression x y ≥ 1 :=
by {
  sorry
}

end minimum_value_is_1_l50_50825


namespace jay_used_zero_fraction_of_gallon_of_paint_l50_50378

theorem jay_used_zero_fraction_of_gallon_of_paint
    (dexter_used : ℝ := 3/8)
    (gallon_in_liters : ℝ := 4)
    (paint_left_liters : ℝ := 4) :
    dexter_used = 3/8 ∧ gallon_in_liters = 4 ∧ paint_left_liters = 4 →
    ∃ jay_used : ℝ, jay_used = 0 :=
by
  sorry

end jay_used_zero_fraction_of_gallon_of_paint_l50_50378


namespace intersection_single_point_l50_50485

def A (x y : ℝ) := x^2 + y^2 = 4
def B (x y : ℝ) (r : ℝ) := (x - 3)^2 + (y - 4)^2 = r^2

theorem intersection_single_point (r : ℝ) (h : r > 0) :
  (∃! p : ℝ × ℝ, A p.1 p.2 ∧ B p.1 p.2 r) → r = 3 :=
by
  apply sorry -- Proof goes here

end intersection_single_point_l50_50485


namespace scenario_a_scenario_b_scenario_c_scenario_d_scenario_e_scenario_f_l50_50968

open Nat

-- Definitions for combinations and permutations
def binomial (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))
def variations (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

-- Scenario a: Each path can be used by at most one person and at most once
theorem scenario_a : binomial 5 2 * binomial 3 2 = 30 := by sorry

-- Scenario b: Each path can be used twice but only in different directions
theorem scenario_b : binomial 5 2 * binomial 5 2 = 100 := by sorry

-- Scenario c: No restrictions
theorem scenario_c : (5 * 5) * (5 * 5) = 625 := by sorry

-- Scenario d: Same as a) with two people distinguished
theorem scenario_d : variations 5 2 * variations 3 2 = 120 := by sorry

-- Scenario e: Same as b) with two people distinguished
theorem scenario_e : variations 5 2 * variations 5 2 = 400 := by sorry

-- Scenario f: Same as c) with two people distinguished
theorem scenario_f : (5 * 5) * (5 * 5) = 625 := by sorry

end scenario_a_scenario_b_scenario_c_scenario_d_scenario_e_scenario_f_l50_50968


namespace total_length_infinite_sum_l50_50096

-- Define the infinite sums
noncomputable def S1 : ℝ := ∑' n : ℕ, (1 / (3^n))
noncomputable def S2 : ℝ := (∑' n : ℕ, (1 / (5^n))) * Real.sqrt 3
noncomputable def S3 : ℝ := (∑' n : ℕ, (1 / (7^n))) * Real.sqrt 5

-- Define the total length
noncomputable def total_length : ℝ := S1 + S2 + S3

-- The statement of the theorem
theorem total_length_infinite_sum : total_length = (3 / 2) + (Real.sqrt 3 / 4) + (Real.sqrt 5 / 6) :=
by
  sorry

end total_length_infinite_sum_l50_50096


namespace train_crosses_bridge_in_time_l50_50065

noncomputable def length_of_train : ℝ := 125
noncomputable def length_of_bridge : ℝ := 250.03
noncomputable def speed_of_train_kmh : ℝ := 45

noncomputable def speed_of_train_ms : ℝ := (speed_of_train_kmh * 1000) / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_time :
  time_to_cross_bridge = 30.0024 :=
  sorry

end train_crosses_bridge_in_time_l50_50065


namespace bird_species_min_l50_50441

theorem bird_species_min (total_birds : ℕ) (h_total_birds : total_birds = 2021)
  (h_even_between : ∀ (species : Sort*) (a b : species), (a ≠ b) → even (nat.dist a b)) :
  ∃ species_num : ℕ, species_num = 1011 :=
by
  sorry

end bird_species_min_l50_50441


namespace original_number_l50_50461

theorem original_number (y : ℚ) (h : 1 - (1 / y) = 5 / 4) : y = -4 :=
sorry

end original_number_l50_50461


namespace min_value_3x_4y_l50_50767

noncomputable def minValue (x y : ℝ) : ℝ := 3 * x + 4 * y

theorem min_value_3x_4y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x + 3 * y = 5 * x * y) : 
  minValue x y ≥ 5 :=
sorry

end min_value_3x_4y_l50_50767


namespace even_function_value_at_three_l50_50906

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- f is an even function
axiom h_even : ∀ x, f x = f (-x)

-- f(x) is defined as x^2 + x when x < 0
axiom h_neg_def : ∀ x, x < 0 → f x = x^2 + x

theorem even_function_value_at_three : f 3 = 6 :=
by {
  -- To be proved
  sorry
}

end even_function_value_at_three_l50_50906


namespace factor_theorem_l50_50765

-- Define the polynomial function f(x)
def f (k : ℚ) (x : ℚ) : ℚ := k * x^3 + 27 * x^2 - k * x + 55

-- State the theorem to find the value of k such that x+5 is a factor of f(x)
theorem factor_theorem (k : ℚ) : f k (-5) = 0 ↔ k = 73 / 12 :=
by sorry

end factor_theorem_l50_50765


namespace value_of_expression_l50_50504

theorem value_of_expression (x y : ℤ) (hx : x = -5) (hy : y = 8) : 2 * (x - y) ^ 2 - x * y = 378 :=
by
  rw [hx, hy]
  -- The proof goes here.
  sorry

end value_of_expression_l50_50504


namespace savings_per_bagel_in_cents_l50_50806

theorem savings_per_bagel_in_cents (cost_individual : ℝ) (cost_dozen : ℝ) (dozen : ℕ) (cents_per_dollar : ℕ) :
  cost_individual = 2.25 →
  cost_dozen = 24 →
  dozen = 12 →
  cents_per_dollar = 100 →
  (cost_individual * cents_per_dollar - (cost_dozen / dozen) * cents_per_dollar) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end savings_per_bagel_in_cents_l50_50806


namespace quotient_calc_l50_50978

theorem quotient_calc (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h_dividend : dividend = 139)
  (h_divisor : divisor = 19)
  (h_remainder : remainder = 6)
  (h_formula : dividend - remainder = quotient * divisor):
  quotient = 7 :=
by {
  -- Insert proof here
  sorry
}

end quotient_calc_l50_50978


namespace initial_number_of_bedbugs_l50_50855

theorem initial_number_of_bedbugs (N : ℕ) 
  (h1 : ∃ N : ℕ, True)
  (h2 : ∀ (n : ℕ), (triples_daily : ℕ → ℕ) → triples_daily n = 3 * n)
  (h3 : ∀ (n : ℕ), (N * 3^4 = n) → n = 810) : 
  N = 10 :=
sorry

end initial_number_of_bedbugs_l50_50855


namespace find_triplets_l50_50399

noncomputable def phi (t : ℝ) : ℝ := 2 * t^3 + t - 2

theorem find_triplets (x y z : ℝ) (h1 : x^5 = phi y) (h2 : y^5 = phi z) (h3 : z^5 = phi x) :
  ∃ r : ℝ, (x = r ∧ y = r ∧ z = r) ∧ (r^5 = phi r) :=
by
  sorry

end find_triplets_l50_50399


namespace tumblers_count_correct_l50_50157

section MrsPetersonsTumblers

-- Define the cost of one tumbler
def tumbler_cost : ℕ := 45

-- Define the amount paid in total by Mrs. Petersons
def total_paid : ℕ := 5 * 100

-- Define the change received by Mrs. Petersons
def change_received : ℕ := 50

-- Calculate the total amount spent
def total_spent : ℕ := total_paid - change_received

-- Calculate the number of tumblers bought
def tumblers_bought : ℕ := total_spent / tumbler_cost

-- Prove the number of tumblers bought is 10
theorem tumblers_count_correct : tumblers_bought = 10 :=
  by
    -- Proof steps will be filled here
    sorry

end MrsPetersonsTumblers

end tumblers_count_correct_l50_50157


namespace janet_daily_search_time_l50_50235

-- Define the conditions
def minutes_looking_for_keys_per_day (x : ℕ) := 
  let total_time_per_day := x + 3
  let total_time_per_week := 7 * total_time_per_day
  total_time_per_week = 77

-- State the theorem
theorem janet_daily_search_time : 
  ∃ x : ℕ, minutes_looking_for_keys_per_day x ∧ x = 8 := by
  sorry

end janet_daily_search_time_l50_50235


namespace hotel_profit_calculation_l50_50517

theorem hotel_profit_calculation
  (operations_expenses : ℝ)
  (meetings_fraction : ℝ) (events_fraction : ℝ) (rooms_fraction : ℝ)
  (meetings_tax_rate : ℝ) (meetings_commission_rate : ℝ)
  (events_tax_rate : ℝ) (events_commission_rate : ℝ)
  (rooms_tax_rate : ℝ) (rooms_commission_rate : ℝ)
  (total_profit : ℝ) :
  operations_expenses = 5000 →
  meetings_fraction = 5/8 →
  events_fraction = 3/10 →
  rooms_fraction = 11/20 →
  meetings_tax_rate = 0.10 →
  meetings_commission_rate = 0.05 →
  events_tax_rate = 0.08 →
  events_commission_rate = 0.06 →
  rooms_tax_rate = 0.12 →
  rooms_commission_rate = 0.03 →
  total_profit = (operations_expenses * (meetings_fraction + events_fraction + rooms_fraction)
                - (operations_expenses
                  + operations_expenses * (meetings_fraction * (meetings_tax_rate + meetings_commission_rate)
                  + events_fraction * (events_tax_rate + events_commission_rate)
                  + rooms_fraction * (rooms_tax_rate + rooms_commission_rate)))) ->
  total_profit = 1283.75 :=
by sorry

end hotel_profit_calculation_l50_50517


namespace find_b_for_integer_a_l50_50788

theorem find_b_for_integer_a (a : ℤ) (b : ℝ) (h1 : 0 ≤ b) (h2 : b < 1) (h3 : (a:ℝ)^2 = 2 * b * (a + b)) :
  b = 0 ∨ b = (-1 + Real.sqrt 3) / 2 :=
sorry

end find_b_for_integer_a_l50_50788


namespace find_roots_l50_50893

theorem find_roots : ∀ z : ℂ, (z^2 + 2 * z = 3 - 4 * I) → (z = 1 - I ∨ z = -3 + I) :=
by
  intro z
  intro h
  sorry

end find_roots_l50_50893


namespace equivalent_angle_l50_50400

theorem equivalent_angle (θ : ℝ) : 
  (∃ k : ℤ, θ = k * 360 + 257) ↔ θ = -463 ∨ (∃ k : ℤ, θ = k * 360 + 257) :=
by
  sorry

end equivalent_angle_l50_50400


namespace taylor_probability_l50_50084

open Nat Real

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

theorem taylor_probability :
  (binomial_probability 5 2 (3/5) = 144 / 625) :=
by
  sorry

end taylor_probability_l50_50084


namespace average_rainfall_l50_50605

theorem average_rainfall (r d h : ℕ) (rainfall_eq : r = 450) (days_eq : d = 30) (hours_eq : h = 24) :
  r / (d * h) = 25 / 16 := 
  by 
    -- Insert appropriate proof here
    sorry

end average_rainfall_l50_50605


namespace mary_score_unique_l50_50633

theorem mary_score_unique (c w : ℕ) (s : ℕ) (h_score_formula : s = 35 + 4 * c - w)
  (h_limit : c + w ≤ 35) (h_greater_90 : s > 90) :
  (∀ s' > 90, s' ≠ s → ¬ ∃ c' w', s' = 35 + 4 * c' - w' ∧ c' + w' ≤ 35) → s = 91 :=
by
  sorry

end mary_score_unique_l50_50633


namespace inequality_relations_l50_50764

variable {R : Type} [OrderedAddCommGroup R]
variables (x y z : R)

theorem inequality_relations (h1 : x - y > x + z) (h2 : x + y < y + z) : y < -z ∧ x < z :=
by
  sorry

end inequality_relations_l50_50764


namespace greatest_3_digit_base_8_divisible_by_7_l50_50678

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end greatest_3_digit_base_8_divisible_by_7_l50_50678


namespace third_discount_is_five_percent_l50_50213

theorem third_discount_is_five_percent (P F : ℝ) (D : ℝ)
  (h1: P = 9356.725146198829)
  (h2: F = 6400)
  (h3: F = (1 - D / 100) * (0.9 * (0.8 * P))) : 
  D = 5 := by
  sorry

end third_discount_is_five_percent_l50_50213


namespace pyramid_height_l50_50513

theorem pyramid_height (h : ℝ) :
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  V_cube = V_pyramid → h = 3.75 :=
by
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  intros h_eq
  sorry

end pyramid_height_l50_50513


namespace transformed_graph_passes_point_l50_50770

theorem transformed_graph_passes_point (f : ℝ → ℝ) 
  (h₁ : f 1 = 3) :
  f (-1) + 1 = 4 :=
by
  sorry

end transformed_graph_passes_point_l50_50770


namespace chocolate_bar_cost_l50_50139

theorem chocolate_bar_cost :
  ∀ (total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips : ℕ),
  total = 150 →
  gummy_bear_cost = 2 →
  chocolate_chip_cost = 5 →
  num_chocolate_bars = 10 →
  num_gummy_bears = 10 →
  num_chocolate_chips = 20 →
  ((total - (num_gummy_bears * gummy_bear_cost + num_chocolate_chips * chocolate_chip_cost)) / num_chocolate_bars = 3) := 
by
  intros total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips 
  intros htotal hgb_cost hcc_cost hncb hngb hncc
  sorry

end chocolate_bar_cost_l50_50139


namespace james_total_room_area_l50_50273

theorem james_total_room_area :
  let original_length := 13
  let original_width := 18
  let increase := 2
  let new_length := original_length + increase
  let new_width := original_width + increase
  let area_of_one_room := new_length * new_width
  let number_of_rooms := 4
  let area_of_four_rooms := area_of_one_room * number_of_rooms
  let area_of_larger_room := area_of_one_room * 2
  let total_area := area_of_four_rooms + area_of_larger_room
  total_area = 1800
  := sorry

end james_total_room_area_l50_50273


namespace man_rate_still_water_l50_50059

def speed_with_stream : ℝ := 6
def speed_against_stream : ℝ := 2

theorem man_rate_still_water : (speed_with_stream + speed_against_stream) / 2 = 4 := by
  sorry

end man_rate_still_water_l50_50059


namespace count_triangles_with_center_inside_l50_50866

theorem count_triangles_with_center_inside :
  let n := 201
  let num_triangles_with_center_inside (n : ℕ) : ℕ := 
    let half := n / 2
    let group_count := half * (half + 1) / 2
    group_count * n / 3
  num_triangles_with_center_inside n = 338350 :=
by
  sorry

end count_triangles_with_center_inside_l50_50866


namespace ceil_floor_eq_zero_l50_50387

theorem ceil_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (- (7 / 3)) = 0) :=
by
  sorry

end ceil_floor_eq_zero_l50_50387


namespace rectangle_dimensions_l50_50960

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : l = 2 * w)
  (h2 : 2 * l + 2 * w = 3 * (l * w)) : 
  w = 1 ∧ l = 2 :=
by 
  sorry

end rectangle_dimensions_l50_50960


namespace inequality_proof_l50_50121

theorem inequality_proof (a b c : ℝ) (h : a > b) : a / (c ^ 2 + 1) > b / (c ^ 2 + 1) :=
by
  sorry

end inequality_proof_l50_50121


namespace profit_percent_l50_50995

theorem profit_percent (cost_price : ℝ) (selling_price : ℝ) (marked_price : ℝ) (n_pens : ℕ) 
  (h1 : n_pens = 60) (h2 : marked_price = 1) (h3 : cost_price = (46 : ℝ) / (60 : ℝ)) 
  (h4 : selling_price = 0.99 * marked_price) : 
  (selling_price - cost_price) / cost_price * 100 = 29.11 :=
by
  sorry

end profit_percent_l50_50995


namespace quadratic_positive_range_l50_50747

theorem quadratic_positive_range (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 3 → ax^2 - 2 * a * x + 3 > 0) ↔ ((-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3)) := 
by {
  sorry
}

end quadratic_positive_range_l50_50747


namespace units_digit_k_squared_plus_three_to_the_k_mod_10_l50_50937

def k := 2025^2 + 3^2025

theorem units_digit_k_squared_plus_three_to_the_k_mod_10 : 
  (k^2 + 3^k) % 10 = 5 := by
sorry

end units_digit_k_squared_plus_three_to_the_k_mod_10_l50_50937


namespace problem_solution_l50_50749

variables (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x_0 : ℝ, x_0^2 + (a-1)*x_0 + 1 < 0

theorem problem_solution (h₁ : p a ∨ q a) (h₂ : ¬(p a ∧ q a)) :
  -1 ≤ a ∧ a ≤ 1 ∨ a > 3 :=
sorry

end problem_solution_l50_50749


namespace greatest_3_digit_base8_num_div_by_7_eq_511_l50_50674

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end greatest_3_digit_base8_num_div_by_7_eq_511_l50_50674


namespace rotated_curve_eq_l50_50166

theorem rotated_curve_eq :
  let θ := Real.pi / 4  -- Rotation angle 45 degrees in radians
  let cos_theta := Real.sqrt 2 / 2
  let sin_theta := Real.sqrt 2 / 2
  let x' := cos_theta * x - sin_theta * y
  let y' := sin_theta * x + cos_theta * y
  x + y^2 = 1 → x' ^ 2 + y' ^ 2 - 2 * x' * y' + Real.sqrt 2 * x' + Real.sqrt 2 * y' - 2 = 0 := 
sorry  -- Proof to be provided.

end rotated_curve_eq_l50_50166


namespace intersection_range_l50_50180

noncomputable def function_f (x: ℝ) : ℝ := abs (x^2 - 4 * x + 3)

theorem intersection_range (b : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ function_f x1 = b ∧ function_f x2 = b ∧ function_f x3 = b) ↔ (0 < b ∧ b ≤ 1) := 
sorry

end intersection_range_l50_50180


namespace dartboard_partitions_count_l50_50710

theorem dartboard_partitions_count : 
  ∃ (l : Multiset ℕ), l.sum = 6 ∧ l.card ≤ 5 ∧ l.sort = l ∧ Multiset.card (Multiset.Powerset l) = 10 :=
sorry

end dartboard_partitions_count_l50_50710


namespace solve_congruence_l50_50947

theorem solve_congruence (n : ℕ) (hn : n < 47) 
  (congr_13n : 13 * n ≡ 9 [MOD 47]) : n ≡ 20 [MOD 47] :=
sorry

end solve_congruence_l50_50947


namespace luggage_max_length_l50_50962

theorem luggage_max_length
  (l w h : ℕ)
  (h_eq : h = 30)
  (ratio_l_w : l = 3 * w / 2)
  (sum_leq : l + w + h ≤ 160) :
  l ≤ 78 := sorry

end luggage_max_length_l50_50962


namespace parabola_vertex_y_coord_l50_50403

theorem parabola_vertex_y_coord (a b c x y : ℝ) (h : a = 2 ∧ b = 16 ∧ c = 35 ∧ y = a*x^2 + b*x + c ∧ x = -b / (2 * a)) : y = 3 :=
by
  sorry

end parabola_vertex_y_coord_l50_50403


namespace tan_alpha_plus_pi_over_4_sin_2alpha_fraction_l50_50571

-- Question 1 (Proving tan(alpha + pi/4) = -3 given tan(alpha) = 2)
theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
sorry

-- Question 2 (Proving the given fraction equals 1 given tan(alpha) = 2)
theorem sin_2alpha_fraction (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (2 * α) / 
   (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1)) = 1 :=
sorry

end tan_alpha_plus_pi_over_4_sin_2alpha_fraction_l50_50571


namespace farmer_total_acres_l50_50048

/--
A farmer used some acres of land for beans, wheat, and corn in the ratio of 5 : 2 : 4, respectively.
There were 376 acres used for corn. Prove that the farmer used a total of 1034 acres of land.
-/
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) :
    5 * x + 2 * x + 4 * x = 1034 :=
by
  sorry

end farmer_total_acres_l50_50048


namespace find_special_four_digit_square_l50_50730

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ (x : ℕ), x * x = n

def same_first_two_digits (n : ℕ) : Prop := (n / 1000) = (n / 100 % 10)

def same_last_two_digits (n : ℕ) : Prop := (n % 100 / 10) = (n % 10)

theorem find_special_four_digit_square :
  ∃ (n : ℕ), is_four_digit n ∧ is_perfect_square n ∧ same_first_two_digits n ∧ same_last_two_digits n ∧ n = 7744 := 
sorry

end find_special_four_digit_square_l50_50730


namespace greatest_div_by_seven_base_eight_l50_50675

theorem greatest_div_by_seven_base_eight : ∃ n : ℕ, 
  (n < 512) ∧ (Divisibility.divides 7 n) ∧ 
  (∀ m : ℕ, (m < 512) → (Divisibility.divides 7 m) → m ≤ n) ∧ 
  nat.to_digits 8 n = [7, 7, 4] := 
sorry

end greatest_div_by_seven_base_eight_l50_50675


namespace evaluate_f_of_f_of_3_l50_50123

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

theorem evaluate_f_of_f_of_3 :
  f (f 3) = 2943 :=
by
  sorry

end evaluate_f_of_f_of_3_l50_50123


namespace binom_18_6_eq_4765_l50_50546

def binom (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_18_6_eq_4765 : binom 18 6 = 4765 := by
  sorry

end binom_18_6_eq_4765_l50_50546


namespace pyramid_height_l50_50514

-- Define the edge length of the cube
def cube_edge_length : ℝ := 5

-- Define the base edge length of the pyramid
def pyramid_base_edge_length : ℝ := 10

-- Define the volume of a cube with edge length 5 units
def cube_volume : ℝ := cube_edge_length ^ 3

-- Define the volume of a pyramid with a square base
def pyramid_volume (h : ℝ) : ℝ := (1 / 3) * (pyramid_base_edge_length ^ 2) * h

-- Add a theorem to prove the height of the pyramid
theorem pyramid_height : ∃ h : ℝ, cube_volume = pyramid_volume h ∧ h = 3.75 :=
by
  -- Given conditions and correct answer lead to the proof of the height being 3.75
  sorry

end pyramid_height_l50_50514


namespace total_cost_of_trick_decks_l50_50973

theorem total_cost_of_trick_decks (cost_per_deck: ℕ) (victor_decks: ℕ) (friend_decks: ℕ) (total_spent: ℕ) : 
  cost_per_deck = 8 → victor_decks = 6 → friend_decks = 2 → total_spent = cost_per_deck * victor_decks + cost_per_deck * friend_decks → total_spent = 64 :=
by 
  sorry

end total_cost_of_trick_decks_l50_50973


namespace rainfall_in_2011_l50_50424

-- Define the parameters
def avg_rainfall_2010 : ℝ := 37.2
def increase_from_2010_to_2011 : ℝ := 1.8
def months_in_a_year : ℕ := 12

-- Define the total rainfall in 2011
def total_rainfall_2011 : ℝ := 468

-- Prove that the total rainfall in Driptown in 2011 is 468 mm
theorem rainfall_in_2011 :
  avg_rainfall_2010 + increase_from_2010_to_2011 = 39.0 → 
  12 * (avg_rainfall_2010 + increase_from_2010_to_2011) = total_rainfall_2011 :=
by sorry

end rainfall_in_2011_l50_50424


namespace polygon_sides_l50_50758

theorem polygon_sides :
  ∃ (n : ℕ), (n * (n - 3) / 2) = n + 33 ∧ n = 11 :=
by
  sorry

end polygon_sides_l50_50758


namespace geometric_ratio_l50_50745

noncomputable def S (n : ℕ) : ℝ := sorry  -- Let's assume S is a function that returns the sum of the first n terms of the geometric sequence.

-- Conditions
axiom S_10_eq_S_5 : S 10 = 2 * S 5

-- Definition to be proved
theorem geometric_ratio :
  (S 5 + S 10 + S 15) / (S 10 - S 5) = -9 / 2 :=
sorry

end geometric_ratio_l50_50745


namespace initial_seashells_l50_50137

-- Definitions based on the problem conditions
def gave_to_joan : ℕ := 6
def left_with_jessica : ℕ := 2

-- Theorem statement to prove the number of seashells initially found by Jessica
theorem initial_seashells : gave_to_joan + left_with_jessica = 8 := by
  -- Proof goes here
  sorry

end initial_seashells_l50_50137


namespace evaluate_expression_l50_50391

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end evaluate_expression_l50_50391


namespace rockham_soccer_league_members_count_l50_50288

def cost_per_pair_of_socks : Nat := 4
def additional_cost_per_tshirt : Nat := 5
def cost_per_tshirt : Nat := cost_per_pair_of_socks + additional_cost_per_tshirt

def pairs_of_socks_per_member : Nat := 2
def tshirts_per_member : Nat := 2

def total_cost_per_member : Nat :=
  pairs_of_socks_per_member * cost_per_pair_of_socks + tshirts_per_member * cost_per_tshirt

def total_cost_all_members : Nat := 2366
def total_members : Nat := total_cost_all_members / total_cost_per_member

theorem rockham_soccer_league_members_count : total_members = 91 :=
by
  -- Given steps in the solution, verify each condition and calculation.
  sorry

end rockham_soccer_league_members_count_l50_50288


namespace star_operation_example_l50_50693

-- Define the operation ☆
def star (a b : ℚ) : ℚ := a - b + 1

-- The theorem to prove
theorem star_operation_example : star (star 2 3) 2 = -1 := by
  sorry

end star_operation_example_l50_50693


namespace statement_2_statement_4_l50_50935

-- Definitions and conditions
variables {Point Line Plane : Type}
variable (a b : Line)
variable (α : Plane)

def parallel (l1 l2 : Line) : Prop := sorry  -- Define parallel relation
def perp (l1 l2 : Line) : Prop := sorry  -- Define perpendicular relation
def perp_plane (l : Line) (p : Plane) : Prop := sorry  -- Define line-plane perpendicular relation
def lies_in (l : Line) (p : Plane) : Prop := sorry  -- Define line lies in plane relation

-- Problem statement 2: If a ∥ b and a ⟂ α, then b ⟂ α
theorem statement_2 (h1 : parallel a b) (h2 : perp_plane a α) : perp_plane b α := sorry

-- Problem statement 4: If a ⟂ α and b ⟂ a, then a ∥ b
theorem statement_4 (h1 : perp_plane a α) (h2 : perp b a) : parallel a b := sorry

end statement_2_statement_4_l50_50935


namespace num_letters_dot_not_straight_line_l50_50037

variable (Total : ℕ)
variable (DS : ℕ)
variable (S_only : ℕ)
variable (D_only : ℕ)

theorem num_letters_dot_not_straight_line 
  (h1 : Total = 40) 
  (h2 : DS = 11) 
  (h3 : S_only = 24) 
  (h4 : Total - S_only - DS = D_only) : 
  D_only = 5 := 
by 
  sorry

end num_letters_dot_not_straight_line_l50_50037


namespace option_b_is_factorization_l50_50334

theorem option_b_is_factorization (m : ℝ) :
  m^2 - 1 = (m + 1) * (m - 1) :=
sorry

end option_b_is_factorization_l50_50334


namespace twice_total_credits_l50_50708

theorem twice_total_credits (Aria Emily Spencer : ℕ) 
(Emily_has_20_credits : Emily = 20) 
(Aria_twice_Emily : Aria = 2 * Emily) 
(Emily_twice_Spencer : Emily = 2 * Spencer) : 
2 * (Aria + Emily + Spencer) = 140 :=
by
  sorry

end twice_total_credits_l50_50708


namespace min_value_of_frac_expr_l50_50405

theorem min_value_of_frac_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (3 / a) + (2 / b) ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end min_value_of_frac_expr_l50_50405


namespace ratio_of_third_to_second_l50_50967

-- Assume we have three numbers (a, b, c) where
-- 1. b = 2 * a
-- 2. c = k * b
-- 3. (a + b + c) / 3 = 165
-- 4. a = 45

theorem ratio_of_third_to_second (a b c k : ℝ) (h1 : b = 2 * a) (h2 : c = k * b) 
  (h3 : (a + b + c) / 3 = 165) (h4 : a = 45) : k = 4 := by 
  sorry

end ratio_of_third_to_second_l50_50967


namespace ratio_of_floors_l50_50371

-- Define the number of floors of each building
def floors_building_A := 4
def floors_building_B := 4 + 9
def floors_building_C := 59

-- Prove the ratio of floors in Building C to Building B
theorem ratio_of_floors :
  floors_building_C / floors_building_B = 59 / 13 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_floors_l50_50371


namespace factor_expression_l50_50236

variable (x : ℝ)

-- Mathematically define the expression e
def e : ℝ := 4 * x * (x + 2) + 10 * (x + 2) + 2 * (x + 2)

-- State that e is equivalent to the factored form
theorem factor_expression : e x = (x + 2) * (4 * x + 12) :=
by
  sorry

end factor_expression_l50_50236


namespace uncovered_side_length_l50_50520

theorem uncovered_side_length :
  ∃ (L : ℝ) (W : ℝ), L * W = 680 ∧ 2 * W + L = 146 ∧ L = 136 := by
  sorry

end uncovered_side_length_l50_50520


namespace determine_m_values_l50_50104

theorem determine_m_values (m : ℚ) :
  ((∃ x y : ℚ, x = -3 ∧ y = 0 ∧ (m^2 - 2 * m - 3) * x + (2 * m^2 + m - 1) * y = 2 * m - 6) ∨
  (∃ k : ℚ, k = -1 ∧ (m^2 - 2 * m - 3) + (2 * m^2 + m - 1) * k = 0)) →
  (m = -5/3 ∨ m = 4/3) :=
by
  sorry

end determine_m_values_l50_50104


namespace P_never_77_l50_50464

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_never_77 (x y : ℤ) : P x y ≠ 77 := sorry

end P_never_77_l50_50464


namespace negation_proof_l50_50314

theorem negation_proof (x : ℝ) : ¬ (x^2 - x + 3 > 0) ↔ (x^2 - x + 3 ≤ 0) :=
by sorry

end negation_proof_l50_50314


namespace even_goals_more_likely_l50_50854

-- We only define the conditions and the question as per the instructions.

variables (p1 : ℝ) (q1 : ℝ) -- Probabilities for even and odd goals in each half
variables (ind : bool) -- Independence of goals in halves

-- Definition of probabilities assuming independence
def p := p1 * p1 + q1 * q1
def q := 2 * p1 * q1

-- The theorem to prove
theorem even_goals_more_likely : p ≥ q := by
  sorry

end even_goals_more_likely_l50_50854


namespace optionD_is_quadratic_l50_50500

variable (x : ℝ)

-- Original equation in Option D
def optionDOriginal := (x^2 + 2 * x = 2 * x^2 - 1)

-- Rearranged form of Option D's equation
def optionDRearranged := (-x^2 + 2 * x + 1 = 0)

theorem optionD_is_quadratic : optionDOriginal x → optionDRearranged x :=
by
  intro h
  -- The proof steps would go here, but we use sorry to skip it
  sorry

end optionD_is_quadratic_l50_50500


namespace mark_pages_per_week_l50_50155

theorem mark_pages_per_week :
  let initial_reading_hours := 2
  let percent_increase := 150
  let initial_pages_per_day := 100
  let days_per_week := 7
  let increased_reading_hours_per_day := initial_reading_hours + (percent_increase / 100) * initial_reading_hours
  let increased_pages_per_day := increased_reading_hours_per_day / initial_reading_hours * initial_pages_per_day
  increased_pages_per_day * days_per_week = 1750 :=
by
  -- Definitions
  let initial_reading_hours := 2
  let percent_increase := 150
  let initial_pages_per_day := 100
  let days_per_week := 7
  
  -- Calculate increased reading hours per day
  let increased_reading_hours_per_day := initial_reading_hours + (percent_increase / 100.0) * initial_reading_hours
  -- Calculate increased pages per day
  let increased_pages_per_day := increased_reading_hours_per_day / initial_reading_hours * initial_pages_per_day
  
  -- Calculate pages per week
  have h : increased_pages_per_day * days_per_week = 1750 := by
    sorry

  exact h

end mark_pages_per_week_l50_50155


namespace min_expression_value_l50_50237

open Real

-- Define the conditions given in the problem: x, y, z are positive reals and their product is 32
variables {x y z : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 32)

-- Define the expression that we want to find the minimum for: x^2 + 4xy + 4y^2 + 2z^2
def expression (x y z : ℝ) : ℝ := x^2 + 4 * x * y + 4 * y^2 + 2 * z^2

-- State the theorem: proving that the minimum value of the expression given the conditions is 96
theorem min_expression_value : ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 32 ∧ expression x y z = 96 :=
sorry

end min_expression_value_l50_50237


namespace total_area_correct_l50_50277

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase : ℕ := 2
def num_extra_rooms_same_size : ℕ := 3
def num_double_size_rooms : ℕ := 1

def increased_length : ℕ := initial_length + increase
def increased_width : ℕ := initial_width + increase

def area_of_one_room (length width : ℕ) : ℕ := length * width

def number_of_rooms : ℕ := 1 + num_extra_rooms_same_size

def total_area_same_size_rooms : ℕ := number_of_rooms * area_of_one_room increased_length increased_width

def total_area_extra_size_room : ℕ := num_double_size_rooms * 2 * area_of_one_room increased_length increased_width

def total_area : ℕ := total_area_same_size_rooms + total_area_extra_size_room

theorem total_area_correct : total_area = 1800 := by
  -- Proof omitted
  sorry

end total_area_correct_l50_50277


namespace sequence_satisfies_recurrence_l50_50087

theorem sequence_satisfies_recurrence (n : ℕ) (a : ℕ → ℕ) (h : ∀ k, 2 ≤ k → k ≤ n - 1 → a (k + 1) = (a k ^ 2 + 1) / (a (k - 1) + 1) - 1) :
  n = 3 ∨ n = 4 := by
  sorry

end sequence_satisfies_recurrence_l50_50087


namespace smallest_k_for_square_l50_50570

theorem smallest_k_for_square : ∃ k : ℕ, (2016 * 2017 * 2018 * 2019 + k) = n^2 ∧ k = 1 :=
by
  use 1
  sorry

end smallest_k_for_square_l50_50570


namespace total_coins_l50_50118

theorem total_coins (total_value : ℕ) (value_2_coins : ℕ) (num_2_coins : ℕ) (num_1_coins : ℕ) : 
  total_value = 402 ∧ value_2_coins = 2 * num_2_coins ∧ num_2_coins = 148 ∧ total_value = value_2_coins + num_1_coins →
  num_1_coins + num_2_coins = 254 :=
by
  intros h
  sorry

end total_coins_l50_50118


namespace negation_proposition_l50_50310

theorem negation_proposition (x : ℝ) :
  ¬(∀ x : ℝ, x^2 - x + 3 > 0) ↔ ∃ x : ℝ, x^2 - x + 3 ≤ 0 := 
by { sorry }

end negation_proposition_l50_50310


namespace grill_runtime_l50_50509

theorem grill_runtime
    (burn_rate : ℕ)
    (burn_time : ℕ)
    (bags : ℕ)
    (coals_per_bag : ℕ)
    (total_burnt_coals : ℕ)
    (total_time : ℕ)
    (h1 : burn_rate = 15)
    (h2 : burn_time = 20)
    (h3 : bags = 3)
    (h4 : coals_per_bag = 60)
    (h5 : total_burnt_coals = bags * coals_per_bag)
    (h6 : total_time = (total_burnt_coals / burn_rate) * burn_time) :
    total_time = 240 :=
by sorry

end grill_runtime_l50_50509


namespace molecular_weight_of_7_moles_KBrO3_l50_50496

def potassium_atomic_weight : ℝ := 39.10
def bromine_atomic_weight : ℝ := 79.90
def oxygen_atomic_weight : ℝ := 16.00
def oxygen_atoms_in_KBrO3 : ℝ := 3

def KBrO3_molecular_weight : ℝ := 
  potassium_atomic_weight + bromine_atomic_weight + (oxygen_atomic_weight * oxygen_atoms_in_KBrO3)

def moles := 7

theorem molecular_weight_of_7_moles_KBrO3 : KBrO3_molecular_weight * moles = 1169.00 := 
by {
  -- The proof would be here, but it is omitted as instructed.
  sorry
}

end molecular_weight_of_7_moles_KBrO3_l50_50496


namespace axis_of_symmetry_shifted_sine_function_l50_50248

open Real

noncomputable def axisOfSymmetry (k : ℤ) : ℝ := k * π / 2 + π / 6

theorem axis_of_symmetry_shifted_sine_function (x : ℝ) (k : ℤ) :
  ∃ k : ℤ, x = axisOfSymmetry k := by
sorry

end axis_of_symmetry_shifted_sine_function_l50_50248


namespace largest_integer_x_l50_50188

theorem largest_integer_x (x : ℤ) : (x / 4 + 3 / 5 < 7 / 4) → x ≤ 4 := sorry

end largest_integer_x_l50_50188


namespace James_total_area_l50_50271

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase_dimension : ℕ := 2
def number_of_new_rooms : ℕ := 4
def larger_room_multiplier : ℕ := 2

noncomputable def new_length : ℕ := initial_length + increase_dimension
noncomputable def new_width : ℕ := initial_width + increase_dimension
noncomputable def area_of_one_new_room : ℕ := new_length * new_width
noncomputable def total_area_of_4_rooms : ℕ := area_of_one_new_room * number_of_new_rooms
noncomputable def area_of_larger_room : ℕ := area_of_one_new_room * larger_room_multiplier
noncomputable def total_area : ℕ := total_area_of_4_rooms + area_of_larger_room

theorem James_total_area : total_area = 1800 := 
by
  sorry

end James_total_area_l50_50271


namespace quadratic_function_value_l50_50056

theorem quadratic_function_value
  (p q r : ℝ)
  (h1 : p + q + r = 3)
  (h2 : 4 * p + 2 * q + r = 12) :
  p + q + 3 * r = -5 :=
by
  sorry

end quadratic_function_value_l50_50056


namespace min_value_sin_sq_l50_50904

theorem min_value_sin_sq (A B : ℝ) (h : A + B = π / 2) :
  4 / (Real.sin A)^2 + 9 / (Real.sin B)^2 ≥ 25 :=
sorry

end min_value_sin_sq_l50_50904


namespace length_of_GH_l50_50031

def EF := 180
def IJ := 120

theorem length_of_GH (EF_parallel_GH : true) (GH_parallel_IJ : true) : GH = 72 := 
sorry

end length_of_GH_l50_50031


namespace avg_age_women_is_52_l50_50001

-- Definitions
def avg_age_men (A : ℚ) := 9 * A
def total_increase := 36
def combined_age_replaced := 36 + 32
def combined_age_women := combined_age_replaced + total_increase
def avg_age_women (W : ℚ) := W / 2

-- Theorem statement
theorem avg_age_women_is_52 (A : ℚ) : avg_age_women combined_age_women = 52 :=
by
  sorry

end avg_age_women_is_52_l50_50001


namespace page_problem_insufficient_information_l50_50462

theorem page_problem_insufficient_information
  (total_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_pages : ℕ)
  (x y : ℕ)
  (O E : ℕ)
  (h1 : total_problems = 450)
  (h2 : finished_problems = 185)
  (h3 : remaining_pages = 15)
  (h4 : O + E = remaining_pages)
  (h5 : O * x + E * y = total_problems - finished_problems) :
  ∀ (x y : ℕ), O * x + E * y = 265 → x = x ∧ y = y :=
by
  sorry

end page_problem_insufficient_information_l50_50462


namespace smallest_EF_minus_DE_l50_50026

theorem smallest_EF_minus_DE (x y z : ℕ) (h1 : x < y) (h2 : y ≤ z) (h3 : x + y + z = 2050)
  (h4 : x + y > z) (h5 : y + z > x) (h6 : z + x > y) : y - x = 1 :=
by
  sorry

end smallest_EF_minus_DE_l50_50026


namespace train_can_speed_up_l50_50953

theorem train_can_speed_up (d t_reduced v_increased v_safe : ℝ) 
  (h1 : d = 1600) (h2 : t_reduced = 4) (h3 : v_increased = 20) (h4 : v_safe = 140) :
  ∃ x : ℝ, (x > 0) ∧ (d / x) = (d / (x + v_increased) + t_reduced) ∧ ((x + v_increased) < v_safe) :=
by 
  sorry

end train_can_speed_up_l50_50953


namespace proposition_true_iff_l50_50481

theorem proposition_true_iff :
  (∀ x y : ℝ, (xy = 1 → x = 1 / y ∧ y = 1 / x) → (x = 1 / y ∧ y = 1 / x → xy = 1)) ∧
  (∀ (A B : Set ℝ), (A ∩ B = B → A ⊆ B) → (A ⊆ B → A ∩ B = B)) ∧
  (∀ m : ℝ, (m > 1 → ∃ x : ℝ, x^2 - 2 * x + m = 0) → (¬(∃ x : ℝ, x^2 - 2 * x + m = 0) → m ≤ 1)) :=
by
  sorry

end proposition_true_iff_l50_50481


namespace Rockham_Soccer_League_Members_l50_50289

-- conditions
def sock_cost : ℕ := 4
def tshirt_cost : ℕ := sock_cost + 5
def total_cost_per_member : ℕ := 2 * sock_cost + 2 * tshirt_cost
def total_cost : ℕ := 2366
def number_of_members (n : ℕ) : Prop := n * total_cost_per_member = total_cost

-- statement to prove
theorem Rockham_Soccer_League_Members : ∃ n : ℕ, number_of_members n ∧ n = 91 :=
by {
  use 91,
  show number_of_members 91,
  sorry
}

end Rockham_Soccer_League_Members_l50_50289


namespace suff_cond_iff_lt_l50_50936

variable (a b : ℝ)

-- Proving that (a - b) a^2 < 0 is a sufficient but not necessary condition for a < b
theorem suff_cond_iff_lt (h : (a - b) * a^2 < 0) : a < b :=
by {
  sorry
}

end suff_cond_iff_lt_l50_50936


namespace eval_ceil_floor_sum_l50_50384

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end eval_ceil_floor_sum_l50_50384


namespace gasoline_price_increase_l50_50658

theorem gasoline_price_increase (highest_price lowest_price : ℝ) (h1 : highest_price = 24) (h2 : lowest_price = 15) : 
  ((highest_price - lowest_price) / lowest_price) * 100 = 60 :=
by
  sorry

end gasoline_price_increase_l50_50658


namespace min_expression_value_l50_50565

theorem min_expression_value :
  ∃ x y : ℝ, (9 - x^2 - 8 * x * y - 16 * y^2 > 0) ∧ 
  (∀ x y : ℝ, 9 - x^2 - 8 * x * y - 16 * y^2 > 0 →
  (13 * x^2 + 24 * x * y + 13 * y^2 + 16 * x + 14 * y + 68) / 
  (9 - x^2 - 8 * x * y - 16 * y^2)^(5/2) = (7 / 27)) :=
sorry

end min_expression_value_l50_50565


namespace no_six_digit_numbers_exists_l50_50722

theorem no_six_digit_numbers_exists :
  ¬(∃ (N : Fin 6 → Fin 720), ∀ (a b c : Fin 6), a ≠ b → a ≠ c → b ≠ c →
  (∃ (i : Fin 6), N i == 720)) := sorry

end no_six_digit_numbers_exists_l50_50722


namespace one_million_minutes_later_l50_50143

open Nat

def initial_datetime : DateTime :=
  { year := 2007, month := 4, day := 15, hour := 12, minute := 0, second := 0 }

def one_million_minutes : Nat := 1000000

def target_datetime : DateTime :=
  { year := 2009, month := 3, day := 10, hour := 10, minute := 40, second := 0 }

theorem one_million_minutes_later :
  let one_million_minutes_in_seconds := one_million_minutes * 60
  let added_seconds := initial_datetime.toSecond.toNat + one_million_minutes_in_seconds 
  DateTime.ofSecond added_seconds == target_datetime :=
by
  sorry

end one_million_minutes_later_l50_50143


namespace candy_bar_cost_correct_l50_50931

def quarters : ℕ := 4
def dimes : ℕ := 3
def nickel : ℕ := 1
def change_received : ℕ := 4

def total_paid : ℕ :=
  (quarters * 25) + (dimes * 10) + (nickel * 5)

def candy_bar_cost : ℕ :=
  total_paid - change_received

theorem candy_bar_cost_correct : candy_bar_cost = 131 := by
  sorry

end candy_bar_cost_correct_l50_50931


namespace find_a5_geometric_sequence_l50_50551

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ r > 0, ∀ n ≥ 1, a (n + 1) = r * a n

theorem find_a5_geometric_sequence :
  ∀ (a : ℕ → ℝ),
  geometric_sequence a ∧ 
  (∀ n, a n > 0) ∧ 
  (a 3 * a 11 = 16) 
  → a 5 = 1 :=
by
  sorry

end find_a5_geometric_sequence_l50_50551


namespace circle_area_conversion_l50_50330

-- Define the given diameter
def diameter (d : ℝ) := d = 8

-- Define the radius calculation
def radius (r : ℝ) := r = 4

-- Define the formula for the area of the circle in square meters
def area_sq_m (A : ℝ) := A = 16 * Real.pi

-- Define the conversion factor from square meters to square centimeters
def conversion_factor := 10000

-- Define the expected area in square centimeters
def area_sq_cm (A : ℝ) := A = 160000 * Real.pi

-- The theorem to prove
theorem circle_area_conversion (d r A_cm : ℝ) (h1 : diameter d) (h2 : radius r) (h3 : area_sq_cm A_cm) :
  A_cm = 160000 * Real.pi :=
by
  sorry

end circle_area_conversion_l50_50330


namespace flute_cost_is_correct_l50_50929

-- Define the conditions
def total_spent : ℝ := 158.35
def stand_cost : ℝ := 8.89
def songbook_cost : ℝ := 7.0

-- Calculate the cost to be subtracted
def accessories_cost : ℝ := stand_cost + songbook_cost

-- Define the target cost of the flute
def flute_cost : ℝ := total_spent - accessories_cost

-- Prove that the flute cost is $142.46
theorem flute_cost_is_correct : flute_cost = 142.46 :=
by
  -- Here we would provide the proof
  sorry

end flute_cost_is_correct_l50_50929


namespace binom_21_13_l50_50910

theorem binom_21_13 : (Nat.choose 21 13) = 203490 :=
by
  have h1 : (Nat.choose 20 13) = 77520 := by sorry
  have h2 : (Nat.choose 20 12) = 125970 := by sorry
  have pascal : (Nat.choose 21 13) = (Nat.choose 20 13) + (Nat.choose 20 12) :=
    by rw [Nat.choose_succ_succ, h1, h2]
  exact pascal

end binom_21_13_l50_50910


namespace sin_cos_acute_angle_lt_one_l50_50294

theorem sin_cos_acute_angle_lt_one (α β : ℝ) (a b c : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (h_triangle : a^2 + b^2 = c^2) (h_nonzero_c : c ≠ 0) :
  (a / c < 1) ∧ (b / c < 1) :=
by 
  sorry

end sin_cos_acute_angle_lt_one_l50_50294


namespace complementary_angles_difference_l50_50811

-- Given that the measures of two complementary angles are in the ratio 4:1,
-- we want to prove that the positive difference between the measures of the two angles is 54 degrees.

theorem complementary_angles_difference (x : ℝ) (h_complementary : 4 * x + x = 90) : 
  abs (4 * x - x) = 54 :=
by
  sorry

end complementary_angles_difference_l50_50811


namespace find_k_and_x2_l50_50899

theorem find_k_and_x2 (k : ℝ) (x2 : ℝ)
  (h1 : 2 * x2 = k)
  (h2 : 2 + x2 = 6) :
  k = 8 ∧ x2 = 4 :=
by
  sorry

end find_k_and_x2_l50_50899


namespace statement_A_correct_statement_C_correct_l50_50848

open Nat

def combinations (n r : ℕ) : ℕ := n.choose r

theorem statement_A_correct : combinations 5 3 = combinations 5 2 := sorry

theorem statement_C_correct : combinations 6 3 - combinations 4 1 = combinations 6 3 - 4 := sorry

end statement_A_correct_statement_C_correct_l50_50848


namespace price_of_magic_card_deck_l50_50858

def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3
def total_earnings : ℕ := 4
def decks_sold := initial_decks - remaining_decks
def price_per_deck := total_earnings / decks_sold

theorem price_of_magic_card_deck : price_per_deck = 2 := by
  sorry

end price_of_magic_card_deck_l50_50858


namespace minimum_packages_shipped_l50_50944

-- Definitions based on the conditions given in the problem
def Sarah_truck_capacity : ℕ := 18
def Ryan_truck_capacity : ℕ := 11

-- Minimum number of packages shipped
theorem minimum_packages_shipped :
  ∃ (n : ℕ), n = Sarah_truck_capacity * Ryan_truck_capacity :=
by sorry

end minimum_packages_shipped_l50_50944


namespace binomial_variance_eta_variance_final_variance_l50_50920

def variance_binomial {n : ℕ} {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) : ℝ := n * p * (1 - p)

noncomputable def eta (xi : ℝ) : ℝ := 5 * xi - 1

theorem binomial_variance {n : ℕ} {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) :
  variance_binomial hp = n * p * (1 - p) := 
by
  sorry

theorem eta_variance {n : ℕ} {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) :
  variance (eta (xi : ℝ)) = 25 * variance_binomial hp := 
by
  sorry

theorem final_variance {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) :
  variance (eta (xi : ℝ)) = 25 * 4 := 
by
  have h₁ := binomial_variance hp
  have h₂ : variance_binomial hp = 16 * (1 / 2) * (1 - 1 / 2) := sorry
  rw [h₂] at h₁
  have h₃ : eta_variance hp = 25 * 4 := sorry
  exact h₃

end binomial_variance_eta_variance_final_variance_l50_50920


namespace number_with_150_quarters_is_37_point_5_l50_50417

theorem number_with_150_quarters_is_37_point_5 (n : ℝ) (h : n / (1/4) = 150) : n = 37.5 := 
by 
  sorry

end number_with_150_quarters_is_37_point_5_l50_50417


namespace incenter_coordinates_l50_50267

theorem incenter_coordinates (p q r : ℝ) (h₁ : p = 8) (h₂ : q = 6) (h₃ : r = 10) :
  ∃ x y z : ℝ, x + y + z = 1 ∧ x = p / (p + q + r) ∧ y = q / (p + q + r) ∧ z = r / (p + q + r) ∧
  x = 1 / 3 ∧ y = 1 / 4 ∧ z = 5 / 12 :=
by
  sorry

end incenter_coordinates_l50_50267


namespace smallest_integer_solution_l50_50331

theorem smallest_integer_solution (x : ℤ) (h : 10 - 5 * x < -18) : x = 6 :=
sorry

end smallest_integer_solution_l50_50331


namespace alcohol_percentage_in_new_mixture_l50_50992

theorem alcohol_percentage_in_new_mixture :
  let afterShaveLotionVolume := 200
  let afterShaveLotionConcentration := 0.35
  let solutionVolume := 75
  let solutionConcentration := 0.15
  let waterVolume := 50
  let totalVolume := afterShaveLotionVolume + solutionVolume + waterVolume
  let alcoholVolume := (afterShaveLotionVolume * afterShaveLotionConcentration) + (solutionVolume * solutionConcentration)
  let alcoholPercentage := (alcoholVolume / totalVolume) * 100
  alcoholPercentage = 25 := 
  sorry

end alcohol_percentage_in_new_mixture_l50_50992


namespace least_k_9_l50_50471

open Nat

noncomputable def u : ℕ → ℝ
| 0     => 1 / 3
| (n+1) => 3 * u n - 3 * (u n) * (u n)

def M : ℝ := 0.5

def acceptable_error (n : ℕ): Prop := abs (u n - M) ≤ 1 / 2 ^ 500

theorem least_k_9 : ∃ k, 0 ≤ k ∧ acceptable_error k ∧ ∀ j, (0 ≤ j ∧ j < k) → ¬acceptable_error j ∧ k = 9 := by
  sorry

end least_k_9_l50_50471


namespace sum_cubes_l50_50875

variables (a b : ℝ)
noncomputable def calculate_sum_cubes (a b : ℝ) : ℝ :=
a^3 + b^3

theorem sum_cubes (h1 : a + b = 11) (h2 : a * b = 21) : calculate_sum_cubes a b = 638 :=
by
  sorry

end sum_cubes_l50_50875


namespace polynomial_roots_geometric_progression_q_l50_50177

theorem polynomial_roots_geometric_progression_q :
    ∃ (a r : ℝ), (a ≠ 0) ∧ (r ≠ 0) ∧
    (a + a * r + a * r ^ 2 + a * r ^ 3 = 0) ∧
    (a ^ 4 * r ^ 6 = 16) ∧
    (a ^ 2 + (a * r) ^ 2 + (a * r ^ 2) ^ 2 + (a * r ^ 3) ^ 2 = 16) :=
by
    sorry

end polynomial_roots_geometric_progression_q_l50_50177


namespace shaded_fraction_l50_50328

theorem shaded_fraction (side_length : ℝ) (base : ℝ) (height : ℝ) (H1: side_length = 4) (H2: base = 3) (H3: height = 2):
  ((side_length ^ 2) - 2 * (1 / 2 * base * height)) / (side_length ^ 2) = 5 / 8 := by
  sorry

end shaded_fraction_l50_50328


namespace rahul_and_sham_together_complete_task_in_35_days_l50_50163

noncomputable def rahul_rate (W : ℝ) : ℝ := W / 60
noncomputable def sham_rate (W : ℝ) : ℝ := W / 84
noncomputable def combined_rate (W : ℝ) := rahul_rate W + sham_rate W

theorem rahul_and_sham_together_complete_task_in_35_days (W : ℝ) :
  (W / combined_rate W) = 35 :=
by
  sorry

end rahul_and_sham_together_complete_task_in_35_days_l50_50163


namespace choose_two_items_proof_l50_50594

   def number_of_ways_to_choose_two_items (n : ℕ) : ℕ :=
     n * (n - 1) / 2

   theorem choose_two_items_proof (n : ℕ) : number_of_ways_to_choose_two_items n = (n * (n - 1)) / 2 :=
   by
     sorry
   
end choose_two_items_proof_l50_50594


namespace farmer_total_acres_l50_50049

/--
A farmer used some acres of land for beans, wheat, and corn in the ratio of 5 : 2 : 4, respectively.
There were 376 acres used for corn. Prove that the farmer used a total of 1034 acres of land.
-/
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) :
    5 * x + 2 * x + 4 * x = 1034 :=
by
  sorry

end farmer_total_acres_l50_50049


namespace train_length_l50_50522

noncomputable section

-- Define the variables involved in the problem.
def train_length_cross_signal (V : ℝ) : ℝ := V * 18
def train_speed_cross_platform (L : ℝ) (platform_length : ℝ) : ℝ := (L + platform_length) / 40

-- Define the main theorem to prove the length of the train.
theorem train_length (V L : ℝ) (platform_length : ℝ) (h1 : L = V * 18)
(h2 : L + platform_length = V * 40) (h3 : platform_length = 366.67) :
L = 300 := 
by
  sorry

end train_length_l50_50522


namespace find_special_four_digit_square_l50_50728

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ (x : ℕ), x * x = n

def same_first_two_digits (n : ℕ) : Prop := (n / 1000) = (n / 100 % 10)

def same_last_two_digits (n : ℕ) : Prop := (n % 100 / 10) = (n % 10)

theorem find_special_four_digit_square :
  ∃ (n : ℕ), is_four_digit n ∧ is_perfect_square n ∧ same_first_two_digits n ∧ same_last_two_digits n ∧ n = 7744 := 
sorry

end find_special_four_digit_square_l50_50728


namespace average_weight_of_11_children_l50_50951

theorem average_weight_of_11_children (b: ℕ) (g: ℕ) (avg_b: ℕ) (avg_g: ℕ) (hb: b = 8) (hg: g = 3) (havg_b: avg_b = 155) (havg_g: avg_g = 115) : 
  (b * avg_b + g * avg_g) / (b + g) = 144 :=
by {
  sorry
}

end average_weight_of_11_children_l50_50951


namespace smith_a_students_l50_50921

-- Definitions representing the conditions

def johnson_a_students : ℕ := 12
def johnson_total_students : ℕ := 20
def smith_total_students : ℕ := 30

def johnson_ratio := johnson_a_students / johnson_total_students

-- Statement to prove
theorem smith_a_students :
  (johnson_a_students / johnson_total_students) = (18 / smith_total_students) :=
sorry

end smith_a_students_l50_50921


namespace pyramid_height_eq_3_75_l50_50515

-- Define the edge length of the cube
def cube_edge_length : ℝ := 5

-- Define the base edge length of the pyramid
def pyramid_base_edge_length : ℝ := 10

-- Define the volume of the cube
def V_cube : ℝ := cube_edge_length ^ 3

-- Define the volume of the pyramid
def V_pyramid (h : ℝ) : ℝ := (1 / 3) * (pyramid_base_edge_length ^ 2) * h

-- Proof that the height of the pyramid is 3.75 units
theorem pyramid_height_eq_3_75 :
  ∃ h : ℝ, V_cube = V_pyramid h ∧ h = 3.75 :=
by
  use 3.75
  split
  . sorry
  . rfl

end pyramid_height_eq_3_75_l50_50515


namespace scientific_notation_of_220_billion_l50_50000

theorem scientific_notation_of_220_billion :
  220000000000 = 2.2 * 10^11 :=
by
  sorry

end scientific_notation_of_220_billion_l50_50000


namespace evaluation_of_expression_l50_50989

theorem evaluation_of_expression :
  10 * (1 / 8) - 6.4 / 8 + 1.2 * 0.125 = 0.6 :=
by sorry

end evaluation_of_expression_l50_50989


namespace max_mx_plus_ny_l50_50579

theorem max_mx_plus_ny 
  (m n x y : ℝ) 
  (h1 : m^2 + n^2 = 6) 
  (h2 : x^2 + y^2 = 24) : 
  mx + ny ≤ 12 :=
sorry

end max_mx_plus_ny_l50_50579


namespace little_twelve_conference_games_l50_50299

def teams_in_division : ℕ := 6
def divisions : ℕ :=  2

def games_within_division (t : ℕ) : ℕ := (t * (t - 1)) / 2 * 2

def games_between_divisions (d t : ℕ) : ℕ := t * t

def total_conference_games (d t : ℕ) : ℕ :=
  d * games_within_division t + games_between_divisions d t

theorem little_twelve_conference_games :
  total_conference_games divisions teams_in_division = 96 :=
by
  sorry

end little_twelve_conference_games_l50_50299


namespace remainder_of_n_l50_50897

theorem remainder_of_n {n : ℕ} (h1 : n^2 ≡ 4 [MOD 7]) (h2 : n^3 ≡ 6 [MOD 7]): 
  n ≡ 5 [MOD 7] :=
sorry

end remainder_of_n_l50_50897


namespace sugar_cups_used_l50_50785

def ratio_sugar_water : ℕ × ℕ := (1, 2)
def total_cups : ℕ := 84

theorem sugar_cups_used (r : ℕ × ℕ) (tc : ℕ) (hsugar : r.1 = 1) (hwater : r.2 = 2) (htotal : tc = 84) :
  (tc * r.1) / (r.1 + r.2) = 28 :=
by
  sorry

end sugar_cups_used_l50_50785


namespace probability_wendy_rolls_higher_l50_50291

theorem probability_wendy_rolls_higher (a b : ℕ) (h : Nat.gcd a b = 1) :
  (∃ a b : ℕ, a + b = 17 ∧ a / b = 5 / 12) :=
begin
  -- Conditions: each roll one six-sided die
  have total_outcomes : 6 * 6 = 36 := by norm_num,
  have favorable_outcomes : 1 + 2 + 3 + 4 + 5 = 15 := by norm_num,
  have probability : (15 : ℚ) / 36 = 5 / 12 := by norm_num,
  use [5, 12],
  split,
  { norm_num },
  { exact probability }
end

end probability_wendy_rolls_higher_l50_50291


namespace find_m_l50_50751

theorem find_m (S : ℕ → ℕ) (a : ℕ → ℕ) (m : ℕ) (h1 : ∀ n, S n = n^2)
  (h2 : S m = (a m + a (m + 1)) / 2)
  (h3 : ∀ n > 1, a n = S n - S (n - 1))
  (h4 : a 1 = 1) :
  m = 2 :=
sorry

end find_m_l50_50751


namespace total_exercise_hours_l50_50941

theorem total_exercise_hours (natasha_minutes_per_day : ℕ) (natasha_days : ℕ)
  (esteban_minutes_per_day : ℕ) (esteban_days : ℕ)
  (h_n : natasha_minutes_per_day = 30) (h_nd : natasha_days = 7)
  (h_e : esteban_minutes_per_day = 10) (h_ed : esteban_days = 9) :
  (natasha_minutes_per_day * natasha_days + esteban_minutes_per_day * esteban_days) / 60 = 5 :=
by
  sorry

end total_exercise_hours_l50_50941


namespace max_a2_plus_b2_l50_50908

theorem max_a2_plus_b2 (a b : ℝ) (h1 : b = 1) (h2 : 1 ≤ -a + 7) (h3 : 1 ≥ a - 3) : a^2 + b^2 = 37 :=
by {
  sorry
}

end max_a2_plus_b2_l50_50908


namespace vector_add_sub_eq_l50_50884

-- Define the vectors involved in the problem
def v1 : ℝ×ℝ×ℝ := (4, -3, 7)
def v2 : ℝ×ℝ×ℝ := (-1, 5, 2)
def v3 : ℝ×ℝ×ℝ := (2, -4, 9)

-- Define the result of the given vector operations
def result : ℝ×ℝ×ℝ := (1, 6, 0)

-- State the theorem we want to prove
theorem vector_add_sub_eq :
  v1 + v2 - v3 = result :=
sorry

end vector_add_sub_eq_l50_50884


namespace evaluate_expression_l50_50389

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end evaluate_expression_l50_50389


namespace simplify_and_evaluate_l50_50946

noncomputable def given_expression (a : ℝ) : ℝ :=
  (a-4) / a / ((a+2) / (a^2 - 2 * a) - (a-1) / (a^2 - 4 * a + 4))

theorem simplify_and_evaluate (a : ℝ) (h : a^2 - 4 * a + 3 = 0) : given_expression a = 1 := by
  sorry

end simplify_and_evaluate_l50_50946


namespace fraction_simplification_l50_50225

theorem fraction_simplification : 
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = (1 / 3) := 
sorry

end fraction_simplification_l50_50225


namespace count_valid_arrangements_l50_50324

-- Definitions for colors
inductive Color | orange | red | blue | yellow | green | purple

open Color

def valid_arrangements : Finset (List Color) :=
  Finset.filter (λ arrangement,
    (∃ g b, g < b ∧ arrangement.nth g = some green ∧ arrangement.nth b = some blue) ∧
    (∃ o p, o < p ∧ arrangement.nth o = some orange ∧ arrangement.nth p = some purple) ∧
    (∀ g b, arrangement.nth g = some green → arrangement.nth b = some blue → (g + 1 ≠ b ∧ b + 1 ≠ g))
  )
  (Finset.map (Equiv.ListEquiv.apply (Finₙ 5 ↪ Color)) (Finset.List.finPerms 5))

theorem count_valid_arrangements :
  valid_arrangements.card = 7 :=
sorry

end count_valid_arrangements_l50_50324


namespace randy_biscuits_l50_50467

theorem randy_biscuits : 
  let initial := 32 
  let father_gift := 13 
  let mother_gift := 15 
  let brother_ate := 20 
  let total := initial + father_gift + mother_gift 
  total - brother_ate = 40 := 
by
  let initial := 32 
  let father_gift := 13 
  let mother_gift := 15 
  let brother_ate := 20 
  let total := initial + father_gift + mother_gift 
  show total - brother_ate = 40 from sorry

end randy_biscuits_l50_50467


namespace binom_18_6_eq_13260_l50_50548

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end binom_18_6_eq_13260_l50_50548


namespace problem_solution_set_l50_50411

theorem problem_solution_set (a b : ℝ) (h : ∀ x, (1 < x ∧ x < 2) ↔ ax^2 + x + b > 0) : a + b = -1 :=
sorry

end problem_solution_set_l50_50411


namespace length_of_AB_l50_50610

noncomputable def height (t : Type) [LinearOrderedField t] := sorry
def area (a b h : ℝ) : ℝ := 0.5 * a * h

theorem length_of_AB (AB CD : ℝ) (h : ℝ) (ratio : area AB h / area CD h = 5 / 2) (sum : AB + CD = 280) :
  AB = 200 :=
begin
  sorry
end

end length_of_AB_l50_50610


namespace ceil_floor_eq_zero_l50_50385

theorem ceil_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (- (7 / 3)) = 0) :=
by
  sorry

end ceil_floor_eq_zero_l50_50385


namespace simplify_expression_l50_50881

variable (b : ℝ)

theorem simplify_expression (h : b ≠ 2) : (2 - 1 / (1 + b / (2 - b))) = 1 + b / 2 := 
sorry

end simplify_expression_l50_50881


namespace abs_neg_three_l50_50207

theorem abs_neg_three : |(-3 : ℝ)| = 3 := 
by
  -- The proof would go here, but we skip it for this exercise.
  sorry

end abs_neg_three_l50_50207


namespace chocolate_bar_cost_l50_50140

-- Define the quantities Jessica bought
def chocolate_bars := 10
def gummy_bears_packs := 10
def chocolate_chips_bags := 20

-- Define the costs
def total_cost := 150
def gummy_bears_pack_cost := 2
def chocolate_chips_bag_cost := 5

-- Define what we want to prove (the cost of one chocolate bar)
theorem chocolate_bar_cost : 
  ∃ chocolate_bar_cost, 
    chocolate_bars * chocolate_bar_cost + 
    gummy_bears_packs * gummy_bears_pack_cost + 
    chocolate_chips_bags * chocolate_chips_bag_cost = total_cost ∧
    chocolate_bar_cost = 3 :=
by
  -- Proof goes here
  sorry

end chocolate_bar_cost_l50_50140


namespace mans_rate_in_still_water_l50_50338

/-- The man's rowing speed in still water given his rowing speeds with and against the stream. -/
theorem mans_rate_in_still_water (v_with_stream v_against_stream : ℝ) (h1 : v_with_stream = 6) (h2 : v_against_stream = 2) : (v_with_stream + v_against_stream) / 2 = 4 := by
  sorry

end mans_rate_in_still_water_l50_50338


namespace bug_total_distance_l50_50041

theorem bug_total_distance : 
  let start := 3
  let first_point := 9
  let second_point := -4
  let distance_1 := abs (first_point - start)
  let distance_2 := abs (second_point - first_point)
  distance_1 + distance_2 = 19 := 
by
  sorry

end bug_total_distance_l50_50041


namespace sum_of_numbers_l50_50958

theorem sum_of_numbers (a b c : ℕ) 
  (h1 : a ≤ b ∧ b ≤ c) 
  (h2 : b = 10) 
  (h3 : (a + b + c) / 3 = a + 15) 
  (h4 : (a + b + c) / 3 = c - 20) 
  (h5 : c = 2 * a)
  : a + b + c = 115 := by
  sorry

end sum_of_numbers_l50_50958


namespace solve_expression_l50_50803

theorem solve_expression :
  ( (12.05 * 5.4 + 0.6) / (2.3 - 1.8) * (7/3) - (4.07 * 3.5 + 0.45) ^ 2) = 90.493 := 
by 
  sorry

end solve_expression_l50_50803


namespace problem_l50_50620

theorem problem (n : ℕ) (h : n = 8 ^ 2022) : n / 4 = 4 ^ 3032 := 
sorry

end problem_l50_50620


namespace maximum_value_of_rocks_l50_50980

theorem maximum_value_of_rocks (R6_val R3_val R2_val : ℕ)
  (R6_wt R3_wt R2_wt : ℕ)
  (num6 num3 num2 : ℕ) :
  R6_val = 16 →
  R3_val = 9 →
  R2_val = 3 →
  R6_wt = 6 →
  R3_wt = 3 →
  R2_wt = 2 →
  30 ≤ num6 →
  30 ≤ num3 →
  30 ≤ num2 →
  ∃ (x6 x3 x2 : ℕ),
    x6 ≤ 4 ∧
    x3 ≤ 4 ∧
    x2 ≤ 4 ∧
    (x6 * R6_wt + x3 * R3_wt + x2 * R2_wt ≤ 24) ∧
    (x6 * R6_val + x3 * R3_val + x2 * R2_val = 68) :=
by
  sorry

end maximum_value_of_rocks_l50_50980


namespace terminal_side_third_quadrant_l50_50412

theorem terminal_side_third_quadrant (α : ℝ) (k : ℤ) 
  (hα : (π / 2) + 2 * k * π < α ∧ α < π + 2 * k * π) : 
  ¬(π + 2 * k * π < α / 3 ∧ α / 3 < (3 / 2) * π + 2 * k * π) :=
by
  sorry

end terminal_side_third_quadrant_l50_50412


namespace counterexample_to_conjecture_l50_50078

theorem counterexample_to_conjecture (n : ℕ) (h : n > 5) : 
  ¬ (∃ a b c : ℕ, (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (a + b + c = n)) ∨
  ¬ (∃ a b c : ℕ, (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (a + b + c = n)) :=
sorry

end counterexample_to_conjecture_l50_50078


namespace problem_l50_50621

theorem problem (n : ℕ) (h : n = 8 ^ 2022) : n / 4 = 4 ^ 3032 := 
sorry

end problem_l50_50621


namespace polynomial_divisible_m_l50_50719

theorem polynomial_divisible_m (m : ℤ) : (∀ x, (4 * x^2 - 6 * x + m) % (x - 3) = 0) → m = -18 :=
by
  assume h : ∀ x, (4 * x^2 - 6 * x + m) % (x - 3) = 0
  let f := λ x, 4 * x^2 - 6 * x + m
  have : f 3 = 0, from by simp [h]
  simp [f] at this
  sorry

end polynomial_divisible_m_l50_50719


namespace value_of_F_l50_50079

theorem value_of_F (D E F : ℕ) (hD : D < 10) (hE : E < 10) (hF : F < 10)
    (h1 : (8 + 5 + D + 7 + 3 + E + 2) % 3 = 0)
    (h2 : (4 + 1 + 7 + D + E + 6 + F) % 3 = 0) : 
    F = 6 :=
by
  sorry

end value_of_F_l50_50079


namespace number_of_truthful_monkeys_l50_50129

-- Define the conditions of the problem
def num_tigers : ℕ := 100
def num_foxes : ℕ := 100
def num_monkeys : ℕ := 100
def total_groups : ℕ := 100
def animals_per_group : ℕ := 3
def yes_tiger : ℕ := 138
def yes_fox : ℕ := 188

-- Problem statement to be proved
theorem number_of_truthful_monkeys :
  ∃ m : ℕ, m = 76 ∧
  ∃ (x y z m n : ℕ),
    -- The number of monkeys mixed with tigers
    x + 2 * (74 - y) = num_monkeys ∧

    -- Given constraints
    m ∈ {n : ℕ | n ≤ x} ∧
    n ∈ {n : ℕ | n ≤ (num_foxes - x)} ∧

    -- Equation setup and derived equations
    (x - m) + (num_foxes - y) + n = yes_tiger ∧
    m + (num_tigers - x - n) + (num_tigers - z) = yes_fox ∧
    y + z = 74 ∧
    
    -- ensuring the groups are valid
    2 * (74 - y) = z :=

sorry

end number_of_truthful_monkeys_l50_50129


namespace pow_mod_equiv_l50_50200

theorem pow_mod_equiv (h : 5^500 ≡ 1 [MOD 1250]) : 5^15000 ≡ 1 [MOD 1250] := 
by 
  sorry

end pow_mod_equiv_l50_50200


namespace greatest_3_digit_base_8_divisible_by_7_l50_50679

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end greatest_3_digit_base_8_divisible_by_7_l50_50679


namespace distance_between_chords_l50_50687

theorem distance_between_chords (R AB CD : ℝ) (hR : R = 15) (hAB : AB = 18) (hCD : CD = 24) : 
  ∃ d : ℝ, d = 21 :=
by 
  sorry

end distance_between_chords_l50_50687


namespace min_value_inequality_l50_50147

theorem min_value_inequality (a b c : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (h : a + b + c = 2) : 
  (1 / (a + 3 * b) + 1 / (b + 3 * c) + 1 / (c + 3 * a)) ≥ 27 / 8 :=
sorry

end min_value_inequality_l50_50147


namespace solve_for_x_l50_50473

theorem solve_for_x : (2 / 5 : ℚ) - (1 / 7) = 1 / (35 / 9) :=
by
  sorry

end solve_for_x_l50_50473


namespace moles_of_KOH_combined_l50_50892

theorem moles_of_KOH_combined 
  (moles_NH4Cl : ℕ)
  (moles_KCl : ℕ)
  (balanced_reaction : ℕ → ℕ → ℕ)
  (h_NH4Cl : moles_NH4Cl = 3)
  (h_KCl : moles_KCl = 3)
  (reaction_ratio : ∀ n, balanced_reaction n n = n) :
  balanced_reaction moles_NH4Cl moles_KCl = 3 * balanced_reaction 1 1 := 
by
  sorry

end moles_of_KOH_combined_l50_50892


namespace sum_due_l50_50984

theorem sum_due (BD TD S : ℝ) (hBD : BD = 18) (hTD : TD = 15) (hRel : BD = TD + (TD^2 / S)) : S = 75 :=
by
  sorry

end sum_due_l50_50984


namespace least_possible_value_of_x_minus_y_plus_z_l50_50341

theorem least_possible_value_of_x_minus_y_plus_z : 
  ∃ (x y z : ℕ), 3 * x = 4 * y ∧ 4 * y = 7 * z ∧ x - y + z = 19 :=
by
  sorry

end least_possible_value_of_x_minus_y_plus_z_l50_50341


namespace fruit_bowl_apples_l50_50347

theorem fruit_bowl_apples (A : ℕ) (total_oranges initial_oranges remaining_oranges : ℕ) (percentage_apples : ℝ) :
  total_oranges = 20 →
  initial_oranges = total_oranges →
  remaining_oranges = initial_oranges - 14 →
  percentage_apples = 0.70 →
  percentage_apples * (A + remaining_oranges) = A →
  A = 14 :=
by 
  intro h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end fruit_bowl_apples_l50_50347


namespace marc_trip_equation_l50_50153

theorem marc_trip_equation (t : ℝ) 
  (before_stop_speed : ℝ := 90)
  (stop_time : ℝ := 0.5)
  (after_stop_speed : ℝ := 110)
  (total_distance : ℝ := 300)
  (total_trip_time : ℝ := 3.5) :
  before_stop_speed * t + after_stop_speed * (total_trip_time - stop_time - t) = total_distance :=
by 
  sorry

end marc_trip_equation_l50_50153


namespace picking_time_l50_50038

theorem picking_time (x : ℝ) 
  (h_wang : x * 8 - 0.25 = x * 7) : 
  x = 0.25 := 
by
  sorry

end picking_time_l50_50038


namespace range_of_a_l50_50172

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (|4 * x - 3| ≤ 1)) → 
  (∀ x : ℝ, (x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
begin
  sorry
end

end range_of_a_l50_50172


namespace evaluate_expression_l50_50390

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end evaluate_expression_l50_50390


namespace trapezium_second_side_length_l50_50737

theorem trapezium_second_side_length (a b h : ℕ) (Area : ℕ) 
  (h_area : Area = (1 / 2 : ℚ) * (a + b) * h)
  (ha : a = 20) (hh : h = 12) (hA : Area = 228) : b = 18 := by
  sorry

end trapezium_second_side_length_l50_50737


namespace Randy_biscuits_l50_50465

theorem Randy_biscuits
  (biscuits_initial : ℕ)
  (father_gift : ℕ)
  (mother_gift : ℕ)
  (brother_eat : ℕ) :
  biscuits_initial = 32 →
  father_gift = 13 →
  mother_gift = 15 →
  brother_eat = 20 →
  biscuits_initial + father_gift + mother_gift - brother_eat = 40 :=
by
  intros h_initial h_father h_mother h_brother
  rw [h_initial, h_father, h_mother, h_brother]
  norm_num
  sorry

end Randy_biscuits_l50_50465


namespace impossible_digit_placement_l50_50925

-- Define the main variables and assumptions
variable (A B C : ℕ)
variable (h_sum : A + B = 45)
variable (h_segmentSum : 3 * A + B = 6 * C)

-- Define the impossible placement problem
theorem impossible_digit_placement :
  ¬(∃ A B C, A + B = 45 ∧ 3 * A + B = 6 * C ∧ 2 * A = 6 * C - 45) :=
by
  sorry

end impossible_digit_placement_l50_50925


namespace value_of_expression_l50_50744

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6 * b = 9 :=
  sorry

end value_of_expression_l50_50744


namespace hyperbola_eqn_l50_50250

-- Definitions of given conditions
def a := 4
def b := 3
def c := 5

def hyperbola (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Hypotheses derived from conditions
axiom asymptotes : b / a = 3 / 4
axiom right_focus : a^2 + b^2 = c^2

-- Main theorem statement
theorem hyperbola_eqn : (forall x y, hyperbola x y ↔ x^2 / 16 - y^2 / 9 = 1) :=
by
  intros
  sorry

end hyperbola_eqn_l50_50250


namespace greatest_prime_factor_is_5_l50_50187

-- Define the expression
def expr : Nat := (3^8 + 9^5)

-- State the theorem
theorem greatest_prime_factor_is_5 : ∃ p : Nat, Prime p ∧ p = 5 ∧ ∀ q : Nat, Prime q ∧ q ∣ expr → q ≤ 5 := by
  sorry

end greatest_prime_factor_is_5_l50_50187


namespace find_natural_number_A_l50_50959

theorem find_natural_number_A (A : ℕ) : 
  (A * 1000 ≤ (A * (A + 1)) / 2 ∧ (A * (A + 1)) / 2 ≤ A * 1000 + 999) → A = 1999 :=
by
  sorry

end find_natural_number_A_l50_50959


namespace find_a_l50_50282

def A : Set ℤ := {-1, 1, 3}
def B (a : ℤ) : Set ℤ := {a + 1, a^2 + 4}
def intersection (a : ℤ) : Set ℤ := A ∩ B a

theorem find_a : ∃ a : ℤ, intersection a = {3} ∧ a = 2 :=
by
  sorry

end find_a_l50_50282


namespace simplify_fraction_l50_50685

theorem simplify_fraction : (3 ^ 2016 - 3 ^ 2014) / (3 ^ 2016 + 3 ^ 2014) = 4 / 5 :=
by
  sorry

end simplify_fraction_l50_50685


namespace eight_girls_circle_least_distance_l50_50232

theorem eight_girls_circle_least_distance :
  let r := 50
  let num_girls := 8
  let total_distance := (8 * (3 * (r * Real.sqrt 2) + 2 * (2 * r)))
  total_distance = 1200 * Real.sqrt 2 + 1600 :=
by
  sorry

end eight_girls_circle_least_distance_l50_50232


namespace simplify_expression_correct_l50_50169

-- Defining the problem conditions and required proof
def simplify_expression (x : ℝ) (h : x ≠ 2) : Prop :=
  (x / (x - 2) + 2 / (2 - x) = 1)

-- Stating the theorem
theorem simplify_expression_correct (x : ℝ) (h : x ≠ 2) : simplify_expression x h :=
  by sorry

end simplify_expression_correct_l50_50169


namespace football_attendance_l50_50020

open Nat

theorem football_attendance:
  (Saturday Monday Wednesday Friday expected_total actual_total: ℕ)
  (h₀: Saturday = 80)
  (h₁: Monday = Saturday - 20)
  (h₂: Wednesday = Monday + 50)
  (h₃: Friday = Saturday + Monday)
  (h₄: expected_total = 350)
  (h₅: actual_total = Saturday + Monday + Wednesday + Friday) :
  actual_total = expected_total + 40 :=
  sorry

end football_attendance_l50_50020


namespace total_number_of_students_l50_50014

theorem total_number_of_students 
  (b g : ℕ) 
  (ratio_condition : 5 * g = 8 * b) 
  (girls_count : g = 160) : 
  b + g = 260 := by
  sorry

end total_number_of_students_l50_50014


namespace solve_inequality_l50_50652

theorem solve_inequality (x : ℝ) : 
  3 - (1 / (3 * x + 4)) < 5 ↔ x ∈ set.Ioo (-∞ : ℝ) (-7 / 6) ∪ set.Ioo (-4 / 3) ∞ := 
sorry

end solve_inequality_l50_50652


namespace primes_solution_l50_50089

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_solution (p : ℕ) (hp : is_prime p) :
  is_prime (p^2 + 2007 * p - 1) ↔ p = 3 :=
by
  sorry

end primes_solution_l50_50089


namespace visual_range_increase_percent_l50_50849

theorem visual_range_increase_percent :
  let original_visual_range := 100
  let new_visual_range := 150
  ((new_visual_range - original_visual_range) / original_visual_range) * 100 = 50 :=
by
  sorry

end visual_range_increase_percent_l50_50849


namespace smallest_integer_ending_in_9_and_divisible_by_11_l50_50189

theorem smallest_integer_ending_in_9_and_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 9 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n :=
  sorry

end smallest_integer_ending_in_9_and_divisible_by_11_l50_50189


namespace largest_gcd_value_l50_50029

open Nat

theorem largest_gcd_value (n : ℕ) : ∃ m ∈ {k | gcd (n^2 + 3) ((n + 1)^2 + 3) = k}, k = 13 :=
by
  sorry

end largest_gcd_value_l50_50029


namespace determine_m_l50_50414

theorem determine_m (x y m : ℝ) :
  (3 * x - y = 4 * m + 1) ∧ (x + y = 2 * m - 5) ∧ (x - y = 4) → m = 1 :=
by sorry

end determine_m_l50_50414


namespace find_first_term_l50_50666

theorem find_first_term
  (S : ℝ) (a r : ℝ)
  (h1 : S = 10)
  (h2 : a + a * r = 6)
  (h3 : a = 2 * r) :
  a = -1 + Real.sqrt 13 ∨ a = -1 - Real.sqrt 13 := by
  sorry

end find_first_term_l50_50666


namespace initial_water_in_hole_l50_50255

theorem initial_water_in_hole (total_needed additional_needed initial : ℕ) (h1 : total_needed = 823) (h2 : additional_needed = 147) :
  initial = total_needed - additional_needed :=
by
  sorry

end initial_water_in_hole_l50_50255


namespace no_difference_410_l50_50346

theorem no_difference_410 (n : ℕ) (R L a : ℕ) (h1 : R + L = 300)
  (h2 : L = 300 - R)
  (h3 : a ≤ 2 * R)
  (h4 : n = L + a)  :
  ¬ (n = 410) :=
by
  sorry

end no_difference_410_l50_50346


namespace polygon_side_possibilities_l50_50865

theorem polygon_side_possibilities (n : ℕ) (h : (n-2) * 180 = 1620) :
  n = 10 ∨ n = 11 ∨ n = 12 :=
by
  sorry

end polygon_side_possibilities_l50_50865


namespace solve_for_x_l50_50761

theorem solve_for_x (x : ℝ) (h : 3 * x + 8 = -4 * x - 16) : x = -24 / 7 :=
sorry

end solve_for_x_l50_50761


namespace zoo_problem_l50_50216

theorem zoo_problem :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := monkeys - 35
  elephants - zebras = 3 :=
by
  sorry

end zoo_problem_l50_50216


namespace speed_of_second_train_l50_50523

/-- 
Given:
1. A train leaves Mumbai at 9 am at a speed of 40 kmph.
2. After one hour, another train leaves Mumbai in the same direction at an unknown speed.
3. The two trains meet at a distance of 80 km from Mumbai.

Prove that the speed of the second train is 80 kmph.
-/
theorem speed_of_second_train (v : ℝ) :
  (∃ (distance_first : ℝ) (distance_meet : ℝ) (initial_speed_first : ℝ) (hours_later : ℤ),
    distance_first = 40 ∧ distance_meet = 80 ∧ initial_speed_first = 40 ∧ hours_later = 1 ∧
    v = distance_meet / (distance_meet / initial_speed_first - hours_later)) → v = 80 := by
  sorry

end speed_of_second_train_l50_50523


namespace remainder_zero_when_x_divided_by_y_l50_50834

theorem remainder_zero_when_x_divided_by_y :
  ∀ (x y : ℝ), 
    0 < x ∧ 0 < y ∧ x / y = 6.12 ∧ y = 49.99999999999996 → 
      x % y = 0 := by
  sorry

end remainder_zero_when_x_divided_by_y_l50_50834


namespace ratio_expression_l50_50257

theorem ratio_expression (p q s u : ℚ) (h1 : p / q = 3 / 5) (h2 : s / u = 8 / 11) : 
  (4 * p * s - 3 * q * u) / (5 * q * u - 8 * p * s) = -69 / 83 :=
by
  sorry

end ratio_expression_l50_50257


namespace negation_proposition_l50_50312

theorem negation_proposition {x : ℝ} : ¬ (x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := sorry

end negation_proposition_l50_50312


namespace sum_of_geometric_sequence_l50_50404

noncomputable def geometric_sequence_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem sum_of_geometric_sequence (a₁ q : ℝ) (n : ℕ) 
  (h1 : a₁ + a₁ * q^3 = 10) 
  (h2 : a₁ * q + a₁ * q^4 = 20) : 
  geometric_sequence_sum a₁ q n = (10 / 9) * (2^n - 1) :=
by 
  sorry

end sum_of_geometric_sequence_l50_50404


namespace perpendicular_vectors_X_value_l50_50456

open Real

-- Define vectors a and b, and their perpendicularity condition
def vector_a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def vector_b : ℝ × ℝ := (1, 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The theorem statement
theorem perpendicular_vectors_X_value (x : ℝ) 
  (h : dot_product (vector_a x) vector_b = 0) : 
    x = -2 / 3 :=
by sorry

end perpendicular_vectors_X_value_l50_50456


namespace progress_regress_ratio_l50_50988

theorem progress_regress_ratio :
  let progress_rate := 1.2
  let regress_rate := 0.8
  let log2 := 0.3010
  let log3 := 0.4771
  let target_ratio := 10000
  (progress_rate / regress_rate) ^ 23 = target_ratio :=
by
  sorry

end progress_regress_ratio_l50_50988


namespace james_needs_50_hours_l50_50613

-- Define the given conditions
def meat_cost_per_pound : ℝ := 5
def meat_pounds_wasted : ℝ := 20
def fruits_veg_cost_per_pound : ℝ := 4
def fruits_veg_pounds_wasted : ℝ := 15
def bread_cost_per_pound : ℝ := 1.5
def bread_pounds_wasted : ℝ := 60
def janitorial_hourly_rate : ℝ := 10
def janitorial_hours_worked : ℝ := 10
def james_hourly_rate : ℝ := 8

-- Calculate the total costs separately
def total_meat_cost : ℝ := meat_cost_per_pound * meat_pounds_wasted
def total_fruits_veg_cost : ℝ := fruits_veg_cost_per_pound * fruits_veg_pounds_wasted
def total_bread_cost : ℝ := bread_cost_per_pound * bread_pounds_wasted
def janitorial_time_and_a_half_rate : ℝ := janitorial_hourly_rate * 1.5
def total_janitorial_cost : ℝ := janitorial_time_and_a_half_rate * janitorial_hours_worked

-- Calculate the total cost James has to pay
def total_cost : ℝ := total_meat_cost + total_fruits_veg_cost + total_bread_cost + total_janitorial_cost

-- Calculate the number of hours James needs to work
def james_work_hours : ℝ := total_cost / james_hourly_rate

-- The theorem to be proved
theorem james_needs_50_hours : james_work_hours = 50 :=
by
  sorry

end james_needs_50_hours_l50_50613


namespace square_of_cube_of_third_smallest_prime_l50_50829

theorem square_of_cube_of_third_smallest_prime :
  let p := 5 in ((p ^ 3) ^ 2) = 15625 :=
by
  let p := 5
  sorry

end square_of_cube_of_third_smallest_prime_l50_50829


namespace cost_of_flute_l50_50928

def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := 8.89
def song_book_cost : ℝ := 7
def flute_cost : ℝ := 142.46

theorem cost_of_flute :
  total_spent - (music_stand_cost + song_book_cost) = flute_cost :=
by
  sorry

end cost_of_flute_l50_50928


namespace percentage_cut_away_in_second_week_l50_50063

theorem percentage_cut_away_in_second_week :
  ∃(x : ℝ), (x / 100) * 142.5 * 0.9 = 109.0125 ∧ x = 15 :=
by
  sorry

end percentage_cut_away_in_second_week_l50_50063


namespace king_then_ten_prob_l50_50970

def num_kings : ℕ := 4
def num_tens : ℕ := 4
def deck_size : ℕ := 52
def first_card_draw_prob := (num_kings : ℚ) / (deck_size : ℚ)
def second_card_draw_prob := (num_tens : ℚ) / (deck_size - 1 : ℚ)

theorem king_then_ten_prob : 
  first_card_draw_prob * second_card_draw_prob = 4 / 663 := by
  sorry

end king_then_ten_prob_l50_50970


namespace combination_18_6_l50_50536

theorem combination_18_6 : nat.choose 18 6 = 18564 :=
by {
  sorry
}

end combination_18_6_l50_50536


namespace Jana_new_walking_speed_l50_50136

variable (minutes : ℕ) (distance1 distance2 : ℝ)

-- Given conditions
def minutes_taken_to_walk := 30
def current_distance := 2
def new_distance := 3
def time_in_hours := minutes / 60

-- Define outcomes
def current_speed_per_minute := current_distance / minutes
def current_speed_per_hour := current_speed_per_minute * 60
def required_speed_per_minute := new_distance / minutes
def required_speed_per_hour := required_speed_per_minute * 60

-- Final statement to prove
theorem Jana_new_walking_speed : required_speed_per_hour = 6 := by
  sorry

end Jana_new_walking_speed_l50_50136


namespace dogs_in_garden_l50_50017

theorem dogs_in_garden (D : ℕ) (ducks : ℕ) (total_feet : ℕ) (dogs_feet : ℕ) (ducks_feet : ℕ) 
  (h1 : ducks = 2) 
  (h2 : total_feet = 28)
  (h3 : dogs_feet = 4)
  (h4 : ducks_feet = 2) 
  (h_eq : dogs_feet * D + ducks_feet * ducks = total_feet) : 
  D = 6 := by
  sorry

end dogs_in_garden_l50_50017


namespace arcsin_neg_one_half_l50_50532

theorem arcsin_neg_one_half : Real.arcsin (-1 / 2) = -Real.pi / 6 :=
by
  sorry

end arcsin_neg_one_half_l50_50532


namespace binom_18_6_eq_18564_l50_50541

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end binom_18_6_eq_18564_l50_50541


namespace minimum_species_count_l50_50431

theorem minimum_species_count {n : ℕ} (h_n : n = 2021) 
  (h_cond : ∀ i j k : ℕ, i < j ∧ j < k → 
    birds i = birds k → birds j ≠ birds i → (j - i - 1) % 2 = 1 ∧ (k - j - 1) % 2 = 1) : 
  ∃ s : ℕ, s ≥ 1011 :=
begin
  sorry
end

end minimum_species_count_l50_50431


namespace farmland_acres_l50_50287

theorem farmland_acres (x y : ℝ) 
  (h1 : x + y = 100) 
  (h2 : 300 * x + (500 / 7) * y = 10000) : 
  true :=
sorry

end farmland_acres_l50_50287


namespace rational_solutions_equation_l50_50663

theorem rational_solutions_equation :
  ∃ x : ℚ, (|x - 19| + |x - 93| = 74 ∧ x ∈ {y : ℚ | 19 ≤ y ∨ 19 < y ∧ y < 93 ∨ y ≥ 93}) :=
sorry

end rational_solutions_equation_l50_50663


namespace dodecahedron_decagon_area_sum_l50_50999

theorem dodecahedron_decagon_area_sum {a b c : ℕ} (h1 : Nat.Coprime a c) (h2 : b ≠ 0) (h3 : ¬ ∃ p : ℕ, p.Prime ∧ p * p ∣ b) 
  (area_eq : (5 + 5 * Real.sqrt 5) / 4 = (a * Real.sqrt b) / c) : a + b + c = 14 :=
sorry

end dodecahedron_decagon_area_sum_l50_50999


namespace largest_n_satisfying_inequality_l50_50564

theorem largest_n_satisfying_inequality : 
  ∃ (n : ℕ), (∀ k : ℕ, (8 : ℚ) / 15 < n / (n + k) ∧ n / (n + k) < (7 : ℚ) / 13) ∧ 
  ∀ n' : ℕ, (∀ k : ℕ, (8 : ℚ) / 15 < n' / (n' + k) ∧ n' / (n' + k) < (7 : ℚ) / 13) → n' ≤ n :=
sorry

end largest_n_satisfying_inequality_l50_50564


namespace equal_divided_value_l50_50622

def n : ℕ := 8^2022

theorem equal_divided_value : n / 4 = 4^3032 := 
by {
  -- We state the equivalence and details used in the proof.
  sorry
}

end equal_divided_value_l50_50622


namespace greatest_base8_three_digit_divisible_by_7_l50_50682

theorem greatest_base8_three_digit_divisible_by_7 :
  ∃ n : ℕ, n < 8^3 ∧ n ≥ 8^2 ∧ (n % 7 = 0) ∧ (to_base 8 n = 777) :=
sorry

end greatest_base8_three_digit_divisible_by_7_l50_50682


namespace greatest_3_digit_base_8_divisible_by_7_l50_50680

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end greatest_3_digit_base_8_divisible_by_7_l50_50680


namespace EventB_is_random_l50_50501

-- Define the events A, B, C, and D as propositions
def EventA : Prop := ∀ (x : ℕ), true -- A coin thrown will fall due to gravity (certain event)
def EventB : Prop := ∃ (n : ℕ), n > 0 -- Hitting the target with a score of 10 points (random event)
def EventC : Prop := ∀ (x : ℕ), true -- The sun rises from the east (certain event)
def EventD : Prop := ∀ (x : ℕ), false -- Horse runs at 70 meters per second (impossible event)

-- Prove that EventB is random, we can use a custom predicate for random events
def is_random_event (e : Prop) : Prop := (∃ (n : ℕ), n > 1) ∧ ¬ ∀ (x : ℕ), e

-- Main statement
theorem EventB_is_random :
  is_random_event EventB :=
by sorry -- The proof will be written here

end EventB_is_random_l50_50501


namespace guard_team_size_l50_50646

theorem guard_team_size (b n s : ℕ) (h_total : b * s * n = 1001) (h_condition : s < n ∧ n < b) : s = 7 := 
by
  sorry

end guard_team_size_l50_50646


namespace monotonic_intervals_l50_50095

noncomputable def f (a x : ℝ) : ℝ := x^2 * Real.exp (a * x)

theorem monotonic_intervals (a : ℝ) :
  (a = 0 → (∀ x : ℝ, (x < 0 → f a x < f a (-1)) ∧ (x > 0 → f a x > f a 1))) ∧
  (a > 0 → (∀ x : ℝ, (x < -2 / a → f a x < f a (-2 / a - 1)) ∧ (x > 0 → f a x > f a 1) ∧ 
                  ((-2 / a) < x ∧ x < 0 → f a x < f a (-2 / a + 1)))) ∧
  (a < 0 → (∀ x : ℝ, (x < 0 → f a x < f a (-1)) ∧ (x > -2 / a → f a x < f a (-2 / a - 1)) ∧
                  (0 < x ∧ x < -2 / a → f a x > f a (-2 / a + 1))))
:= sorry

end monotonic_intervals_l50_50095


namespace ratio_of_speeds_l50_50555

def eddy_time := 3
def eddy_distance := 480
def freddy_time := 4
def freddy_distance := 300

def eddy_speed := eddy_distance / eddy_time
def freddy_speed := freddy_distance / freddy_time

theorem ratio_of_speeds : (eddy_speed / freddy_speed) = 32 / 15 :=
by
  sorry

end ratio_of_speeds_l50_50555


namespace transformation_is_rotation_l50_50196

-- Define the 90 degree rotation matrix
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

-- Define the transformation matrix to be proven
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

-- The theorem that proves they are equivalent
theorem transformation_is_rotation :
  transformation_matrix = rotation_matrix :=
by
  sorry

end transformation_is_rotation_l50_50196


namespace numbers_sum_prod_l50_50582

theorem numbers_sum_prod (x y : ℝ) (h_sum : x + y = 10) (h_prod : x * y = 24) : (x = 4 ∧ y = 6) ∨ (x = 6 ∧ y = 4) :=
by
  sorry

end numbers_sum_prod_l50_50582


namespace simple_interest_rate_l50_50202

theorem simple_interest_rate (P A T : ℝ) (R : ℝ) (hP : P = 750) (hA : A = 900) (hT : T = 5) :
    (A - P) = (P * R * T) / 100 → R = 4 := by
  sorry

end simple_interest_rate_l50_50202


namespace single_bacteria_colony_days_to_limit_l50_50043

theorem single_bacteria_colony_days_to_limit (n : ℕ) (h : ∀ t : ℕ, t ≤ 21 → (2 ^ t = 2 * 2 ^ (t - 1))) : n = 22 :=
by
  sorry

end single_bacteria_colony_days_to_limit_l50_50043


namespace evaluate_expression_l50_50380

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem evaluate_expression : (factorial (factorial 4)) / factorial 4 = factorial 23 :=
by sorry

end evaluate_expression_l50_50380


namespace lcm_16_24_45_l50_50977

-- Define the numbers
def a : Nat := 16
def b : Nat := 24
def c : Nat := 45

-- State the theorem that the least common multiple of these numbers is 720
theorem lcm_16_24_45 : Nat.lcm (Nat.lcm 16 24) 45 = 720 := by
  sorry

end lcm_16_24_45_l50_50977


namespace combination_18_6_l50_50538

theorem combination_18_6 : nat.choose 18 6 = 18564 :=
by {
  sorry
}

end combination_18_6_l50_50538


namespace central_angle_of_sector_l50_50573

/-- The central angle of the sector obtained by unfolding the lateral surface of a cone with
    base radius 1 and slant height 2 is \(\pi\). -/
theorem central_angle_of_sector (r_base : ℝ) (r_slant : ℝ) (α : ℝ)
  (h1 : r_base = 1) (h2 : r_slant = 2) (h3 : 2 * π = α * r_slant) : α = π :=
by
  sorry

end central_angle_of_sector_l50_50573


namespace ticTacToeWinningDiagonals_l50_50425

-- Define the tic-tac-toe board and the conditions
def ticTacToeBoard : Type := Fin 3 × Fin 3
inductive Player | X | O

def isWinningDiagonal (board : ticTacToeBoard → Option Player) : Prop :=
  (board (0, 0) = some Player.O ∧ board (1, 1) = some Player.O ∧ board (2, 2) = some Player.O) ∨
  (board (0, 2) = some Player.O ∧ board (1, 1) = some Player.O ∧ board (2, 0) = some Player.O)

-- Define the main problem statement
theorem ticTacToeWinningDiagonals : ∃ (n : ℕ), n = 40 :=
  sorry

end ticTacToeWinningDiagonals_l50_50425


namespace percentage_increase_l50_50011

noncomputable def price_increase (d new_price : ℝ) : ℝ :=
  ((new_price - d) / d) * 100

theorem percentage_increase 
  (d new_price : ℝ)
  (h1 : 2 * d = 585)
  (h2 : new_price = 351) :
  price_increase d new_price = 20 :=
by
  sorry

end percentage_increase_l50_50011


namespace abs_neg_three_l50_50208

theorem abs_neg_three : abs (-3) = 3 :=
by 
  sorry

end abs_neg_three_l50_50208


namespace fraction_students_study_japanese_l50_50368

theorem fraction_students_study_japanese (J S : ℕ) (h1 : S = 3 * J) 
(h2 : ∃ k : ℕ, k = (1/3 : ℚ) * S) (h3 : ∃ l : ℕ, l = (3/4 : ℚ) * J) :
  (∃ f : ℚ, f = ((1/3 : ℚ) * S + (3/4 : ℚ) * J) / (S + J) ∧ f = 7/16) :=
by
  sorry

end fraction_students_study_japanese_l50_50368


namespace intersect_at_two_points_l50_50254

theorem intersect_at_two_points (a : ℝ) :
  (∃ p q : ℝ × ℝ, 
    (p.1 - p.2 + 1 = 0) ∧ (2 * p.1 + p.2 - 4 = 0) ∧ (a * p.1 - p.2 + 2 = 0) ∧
    (q.1 - q.2 + 1 = 0) ∧ (2 * q.1 + q.2 - 4 = 0) ∧ (a * q.1 - q.2 + 2 = 0) ∧ p ≠ q) →
  (a = 1 ∨ a = -2) :=
by 
  sorry

end intersect_at_two_points_l50_50254


namespace sum_of_coefficients_l50_50150

def u (n : ℕ) : ℕ := 
  match n with
  | 0 => 6 -- Assume the sequence starts at u_0 for easier indexing
  | n + 1 => u n + 5 + 2 * n

theorem sum_of_coefficients (u : ℕ → ℕ) : 
  (∀ n, u (n + 1) = u n + 5 + 2 * n) ∧ u 1 = 6 → 
  (∃ a b c : ℕ, (∀ n, u n = a * n^2 + b * n + c) ∧ a + b + c = 6) := 
by
  sorry

end sum_of_coefficients_l50_50150


namespace yard_fraction_occupied_by_flowerbeds_l50_50695

noncomputable def rectangular_yard_area (length width : ℕ) : ℕ :=
  length * width

noncomputable def triangle_area (leg_length : ℕ) : ℕ :=
  2 * (1 / 2 * leg_length ^ 2)

theorem yard_fraction_occupied_by_flowerbeds :
  let length := 30
  let width := 7
  let parallel_side_short := 20
  let parallel_side_long := 30
  let flowerbed_leg := 7
  rectangular_yard_area length width ≠ 0 ∧
  triangle_area flowerbed_leg * 2 = 49 →
  (triangle_area flowerbed_leg * 2) / rectangular_yard_area length width = 7 / 30 :=
sorry

end yard_fraction_occupied_by_flowerbeds_l50_50695


namespace bus_stops_for_4_minutes_per_hour_l50_50727

theorem bus_stops_for_4_minutes_per_hour
  (V_excluding_stoppages V_including_stoppages : ℝ)
  (h1 : V_excluding_stoppages = 90)
  (h2 : V_including_stoppages = 84) :
  (60 * (V_excluding_stoppages - V_including_stoppages)) / V_excluding_stoppages = 4 :=
by
  sorry

end bus_stops_for_4_minutes_per_hour_l50_50727


namespace percentage_born_in_september_l50_50178

theorem percentage_born_in_september (total famous : ℕ) (born_in_september : ℕ) (h1 : total = 150) (h2 : born_in_september = 12) :
  (born_in_september * 100 / total) = 8 :=
by
  sorry

end percentage_born_in_september_l50_50178


namespace smallest_possible_students_group_l50_50635

theorem smallest_possible_students_group 
  (students : ℕ) :
  (∀ n, 2 ≤ n ∧ n ≤ 15 → ∃ k, students = k * n) ∧
  ¬∃ k, students = k * 10 ∧ ¬∃ k, students = k * 25 ∧ ¬∃ k, students = k * 50 ∧
  ∀ m n, 1 ≤ m ∧ m ≤ 15 ∧ 1 ≤ n ∧ n ≤ 15 ∧ (students ≠ m * n) → (m = n ∨ m ≠ n)
  → students = 120 := sorry

end smallest_possible_students_group_l50_50635


namespace solve_equation_l50_50297

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x^2 + x + 1) / (x + 2) = x + 1 → x = -1 / 2 := 
by
  intro h1
  sorry

end solve_equation_l50_50297


namespace arith_seq_ratio_l50_50592

theorem arith_seq_ratio {a b : ℕ → ℕ} {S T : ℕ → ℕ}
  (h₁ : ∀ n, S n = (n * (2 * a n - a 1)) / 2)
  (h₂ : ∀ n, T n = (n * (2 * b n - b 1)) / 2)
  (h₃ : ∀ n, S n / T n = (5 * n + 3) / (2 * n + 7)) :
  (a 9 / b 9 = 88 / 41) :=
sorry

end arith_seq_ratio_l50_50592


namespace candidates_appeared_in_each_state_equals_7900_l50_50204

theorem candidates_appeared_in_each_state_equals_7900 (x : ℝ) (h : 0.07 * x = 0.06 * x + 79) : x = 7900 :=
sorry

end candidates_appeared_in_each_state_equals_7900_l50_50204


namespace jake_watched_friday_l50_50132

theorem jake_watched_friday
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (thursday_hours : ℕ)
  (total_hours : ℕ)
  (day_hours : ℕ := 24) :
  monday_hours = (day_hours / 2) →
  tuesday_hours = 4 →
  wednesday_hours = (day_hours / 4) →
  thursday_hours = ((monday_hours + tuesday_hours + wednesday_hours) / 2) →
  total_hours = 52 →
  (total_hours - (monday_hours + tuesday_hours + wednesday_hours + thursday_hours)) = 19 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jake_watched_friday_l50_50132


namespace length_of_MN_l50_50638

-- Define the lengths and trapezoid properties
variables (a b: ℝ)

-- Define the problem statement
theorem length_of_MN (a b: ℝ) :
  ∃ (MN: ℝ), ∀ (M N: ℝ) (is_trapezoid : True),
  (MN = 3 * a * b / (a + 2 * b)) :=
sorry

end length_of_MN_l50_50638


namespace find_m_l50_50619

noncomputable def g (d e f x : ℤ) : ℤ := d * x * x + e * x + f

theorem find_m (d e f m : ℤ) (h₁ : g d e f 2 = 0)
    (h₂ : 60 < g d e f 6 ∧ g d e f 6 < 70) 
    (h₃ : 80 < g d e f 9 ∧ g d e f 9 < 90)
    (h₄ : 10000 * m < g d e f 100 ∧ g d e f 100 < 10000 * (m + 1)) :
  m = -1 :=
sorry

end find_m_l50_50619


namespace intersection_point_correct_l50_50304

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Line :=
(p1 : Point3D) (p2 : Point3D)

structure Plane :=
(trace : Line) (point : Point3D)

noncomputable def intersection_point (l : Line) (β : Plane) : Point3D := sorry

theorem intersection_point_correct (l : Line) (β : Plane) (P : Point3D) :
  let res := intersection_point l β
  res = P :=
sorry

end intersection_point_correct_l50_50304


namespace train_speed_l50_50339

theorem train_speed (L : ℝ) (T : ℝ) (L_pos : 0 < L) (T_pos : 0 < T) (L_eq : L = 150) (T_eq : T = 3) : L / T = 50 := by
  sorry

end train_speed_l50_50339


namespace min_value_abs_diff_l50_50245

-- Definitions of conditions
def is_in_interval (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 4

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  (b^2 - a^2 = 2) ∧ (c^2 - b^2 = 2)

-- Main statement
theorem min_value_abs_diff (x y z : ℝ)
  (h1 : is_in_interval x)
  (h2 : is_in_interval y)
  (h3 : is_in_interval z)
  (h4 : is_arithmetic_progression x y z) :
  |x - y| + |y - z| = 4 - 2 * Real.sqrt 3 :=
sorry

end min_value_abs_diff_l50_50245


namespace max_intersections_circle_quadrilateral_max_intersections_correct_l50_50684

-- Define the intersection property of a circle and a line segment
def max_intersections_per_side (circle : Type) (line_segment : Type) : ℕ := 2

-- Define a quadrilateral as a shape having four sides
def sides_of_quadrilateral : ℕ := 4

-- The theorem stating the maximum number of intersection points between a circle and a quadrilateral
theorem max_intersections_circle_quadrilateral (circle : Type) (quadrilateral : Type) : Prop :=
  max_intersections_per_side circle quadrilateral * sides_of_quadrilateral = 8

-- Proof is skipped with 'sorry'
theorem max_intersections_correct (circle : Type) (quadrilateral : Type) :
  max_intersections_circle_quadrilateral circle quadrilateral :=
by
  sorry

end max_intersections_circle_quadrilateral_max_intersections_correct_l50_50684


namespace complex_fraction_simplification_l50_50715

theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) : (2 : ℂ) / (1 + i)^2 = i :=
by 
-- this will be filled when proving the theorem in Lean
sorry

end complex_fraction_simplification_l50_50715


namespace parabola_standard_eq_l50_50317

theorem parabola_standard_eq (p p' : ℝ) (h₁ : p > 0) (h₂ : p' > 0) :
  (∀ (x y : ℝ), (x^2 = 2 * p * y ∨ y^2 = -2 * p' * x) → 
  (x = -2 ∧ y = 4 → (x^2 = y ∨ y^2 = -8 * x))) :=
by
  sorry

end parabola_standard_eq_l50_50317


namespace pass_rate_eq_l50_50007

theorem pass_rate_eq (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : (1 - a) * (1 - b) = ab - a - b + 1 :=
by
  sorry

end pass_rate_eq_l50_50007


namespace largest_integer_value_neg_quadratic_l50_50740

theorem largest_integer_value_neg_quadratic :
  ∃ m : ℤ, (4 < m ∧ m < 7) ∧ (m^2 - 11 * m + 28 < 0) ∧ ∀ n : ℤ, (4 < n ∧ n < 7 ∧ (n^2 - 11 * n + 28 < 0)) → n ≤ m :=
sorry

end largest_integer_value_neg_quadratic_l50_50740


namespace min_omega_l50_50584

noncomputable def f (ω φ : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem min_omega (ω φ : ℝ) (hω : ω > 0)
  (h_sym : ∀ x : ℝ, f ω φ (2 * (π / 3) - x) = f ω φ x)
  (h_val : f ω φ (π / 12) = 0) :
  ω = 2 :=
sorry

end min_omega_l50_50584


namespace problem_value_l50_50332

theorem problem_value :
  (1 / 3 * 9 * 1 / 27 * 81 * 1 / 243 * 729 * 1 / 2187 * 6561 * 1 / 19683 * 59049) = 243 := 
sorry

end problem_value_l50_50332


namespace no_integers_satisfy_eq_l50_50559

theorem no_integers_satisfy_eq (m n : ℤ) : ¬ (m^3 = 4 * n + 2) := 
  sorry

end no_integers_satisfy_eq_l50_50559


namespace ads_ratio_l50_50323

theorem ads_ratio 
  (first_ads : ℕ := 12)
  (second_ads : ℕ)
  (third_ads := second_ads + 24)
  (fourth_ads := (3 / 4) * second_ads)
  (clicked_ads := 68)
  (total_ads := (3 / 2) * clicked_ads == 102)
  (ads_eq : first_ads + second_ads + third_ads + fourth_ads = total_ads) :
  second_ads / first_ads = 2 :=
by sorry

end ads_ratio_l50_50323


namespace ratio_surface_area_l50_50350

noncomputable def side_length (a : ℝ) := a
noncomputable def radius (R : ℝ) := R

theorem ratio_surface_area (a R : ℝ) (h : a^3 = (4/3) * Real.pi * R^3) : 
  (6 * a^2) / (4 * Real.pi * R^2) = (3 * (6 / Real.pi)) :=
by sorry

end ratio_surface_area_l50_50350


namespace hyperbola_and_line_properties_l50_50242

open Real

def hyperbola (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def asymptote1 (x y : ℝ) : Prop := y = sqrt 3 * x
def asymptote2 (x y : ℝ) : Prop := y = -sqrt 3 * x
def line (x y t : ℝ) : Prop := y = x + t

theorem hyperbola_and_line_properties :
  ∃ a b t : ℝ,
  a > 0 ∧ b > 0 ∧ a = 1 ∧ b^2 = 3 ∧
  (∀ x y, hyperbola x y a b ↔ (x^2 - y^2 / 3 = 1)) ∧
  (∀ x y, asymptote1 x y ↔ y = sqrt 3 * x) ∧
  (∀ x y, asymptote2 x y ↔ y = -sqrt 3 * x) ∧
  (∀ x y, (line x y t ↔ (y = x + sqrt 3) ∨ (y = x - sqrt 3))) := sorry

end hyperbola_and_line_properties_l50_50242


namespace least_positive_int_to_next_multiple_l50_50329

theorem least_positive_int_to_next_multiple (x : ℕ) (n : ℕ) (h : x = 365 ∧ n > 0) 
  (hm : (x + n) % 5 = 0) : n = 5 :=
by
  sorry

end least_positive_int_to_next_multiple_l50_50329


namespace find_eccentricity_l50_50914

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

def eccentricity_conic_section (m : ℝ) (e : ℝ) : Prop :=
  (m = 6 → e = (Real.sqrt 30) / 6) ∧
  (m = -6 → e = Real.sqrt 7)

theorem find_eccentricity (m : ℝ) :
  geometric_sequence 4 m 9 →
  eccentricity_conic_section m ((Real.sqrt 30) / 6) ∨
  eccentricity_conic_section m (Real.sqrt 7) :=
by
  sorry

end find_eccentricity_l50_50914


namespace no_real_roots_equationD_l50_50837

def discriminant (a b c : ℕ) : ℤ := b^2 - 4 * a * c

def equationA := (1, -2, -4)
def equationB := (1, -4, 4)
def equationC := (1, -2, -5)
def equationD := (1, 3, 5)

theorem no_real_roots_equationD :
  discriminant (1 : ℕ) 3 5 < 0 :=
by
  show discriminant 1 3 5 < 0
  sorry

end no_real_roots_equationD_l50_50837


namespace find_a_l50_50718

def star (a b : ℕ) : ℕ := 3 * a - b ^ 2

theorem find_a (a : ℕ) (b : ℕ) (h : star a b = 14) : a = 10 :=
by sorry

end find_a_l50_50718


namespace verify_trees_in_other_row_l50_50060

-- Definition of a normal lemon tree lemon production per year
def normalLemonTreeProduction : ℕ := 60

-- Definition of the percentage increase in lemon production for specially engineered lemon trees
def percentageIncrease : ℕ := 50

-- Definition of lemon production for specially engineered lemon trees
def specialLemonTreeProduction : ℕ := normalLemonTreeProduction * (1 + percentageIncrease / 100)

-- Number of trees in one row of the grove
def treesInOneRow : ℕ := 50

-- Total lemon production in 5 years
def totalLemonProduction : ℕ := 675000

-- Number of years
def years : ℕ := 5

-- Total number of trees in the grove
def totalNumberOfTrees : ℕ := totalLemonProduction / (specialLemonTreeProduction * years)

-- Number of trees in the other row
def treesInOtherRow : ℕ := totalNumberOfTrees - treesInOneRow

-- Theorem: Verification of the number of trees in the other row
theorem verify_trees_in_other_row : treesInOtherRow = 1450 :=
  by
  -- Proof logic is omitted, leaving as sorry
  sorry

end verify_trees_in_other_row_l50_50060


namespace solve_for_x_l50_50258

theorem solve_for_x 
  (y : ℚ) (x : ℚ)
  (h : x / (x - 1) = (y^3 + 2 * y^2 - 2) / (y^3 + 2 * y^2 - 3)) :
  x = (y^3 + 2 * y^2 - 2) / 2 :=
sorry

end solve_for_x_l50_50258


namespace slope_of_line_l50_50580

theorem slope_of_line (θ : ℝ) (h_cosθ : (Real.cos θ) = 4/5) : (Real.sin θ) / (Real.cos θ) = 3/4 :=
by
  sorry

end slope_of_line_l50_50580


namespace range_of_g_l50_50743

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x / 3))^2 + (Real.pi / 4) * (Real.arcsin (x / 3)) 
    - (Real.arcsin (x / 3))^2 + (Real.pi^2 / 16) * (x^2 + 2 * x + 3)

theorem range_of_g : 
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → 
  ∃ y, y = g x ∧ y ∈ (Set.Icc (Real.pi^2 / 4) (15 * Real.pi^2 / 16 + Real.pi / 4 * Real.arcsin 1)) :=
by
  sorry

end range_of_g_l50_50743
