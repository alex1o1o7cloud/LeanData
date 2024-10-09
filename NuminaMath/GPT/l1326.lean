import Mathlib

namespace find_A_l1326_132606

theorem find_A (A : ℕ) (B : ℕ) (h₀ : 0 ≤ B) (h₁ : B ≤ 999) :
  1000 * A + B = (A * (A + 1)) / 2 → A = 1999 := sorry

end find_A_l1326_132606


namespace cos_identity_15_30_degrees_l1326_132661

theorem cos_identity_15_30_degrees (a b : ℝ) (h : b = 2 * a^2 - 1) : 2 * a^2 - b = 1 :=
by
  sorry

end cos_identity_15_30_degrees_l1326_132661


namespace triangle_inequality_sqrt_sum_three_l1326_132676

theorem triangle_inequality_sqrt_sum_three
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  3 ≤ (Real.sqrt (a / (-a + b + c)) + 
       Real.sqrt (b / (a - b + c)) + 
       Real.sqrt (c / (a + b - c))) := 
sorry

end triangle_inequality_sqrt_sum_three_l1326_132676


namespace no_positive_integers_abc_l1326_132685

theorem no_positive_integers_abc :
  ¬ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (a * b + b * c = a * c) ∧ (a * b * c = Nat.factorial 10) :=
sorry

end no_positive_integers_abc_l1326_132685


namespace scientific_notation_of_4370000_l1326_132608

theorem scientific_notation_of_4370000 :
  4370000 = 4.37 * 10^6 :=
sorry

end scientific_notation_of_4370000_l1326_132608


namespace cos_2beta_correct_l1326_132650

open Real

theorem cos_2beta_correct (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
    (h3 : tan α = 1 / 7) (h4 : cos (α + β) = 2 * sqrt 5 / 5) :
    cos (2 * β) = 4 / 5 := 
  sorry

end cos_2beta_correct_l1326_132650


namespace petrol_expense_l1326_132681

theorem petrol_expense 
  (rent milk groceries education misc savings petrol total_salary : ℝ)
  (H1 : rent = 5000)
  (H2 : milk = 1500)
  (H3 : groceries = 4500)
  (H4 : education = 2500)
  (H5 : misc = 6100)
  (H6 : savings = 2400)
  (H7 : total_salary = savings / 0.10)
  (H8 : total_salary = rent + milk + groceries + education + misc + petrol + savings) :
  petrol = 2000 :=
by
  sorry

end petrol_expense_l1326_132681


namespace max_notebooks_15_dollars_l1326_132667

noncomputable def max_notebooks (money : ℕ) : ℕ :=
  let cost_individual   := 2
  let cost_pack_4       := 6
  let cost_pack_7       := 9
  let notebooks_budget  := 15
  if money >= 9 then 
    7 + max_notebooks (money - 9)
  else if money >= 6 then 
    4 + max_notebooks (money - 6)
  else 
    money / 2

theorem max_notebooks_15_dollars : max_notebooks 15 = 11 :=
by
  sorry

end max_notebooks_15_dollars_l1326_132667


namespace max_value_of_abc_expression_l1326_132683

noncomputable def max_abc_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) : ℝ :=
  a^3 * b^2 * c^2

theorem max_value_of_abc_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  max_abc_expression a b c h1 h2 h3 h4 ≤ 432 / 7^7 :=
sorry

end max_value_of_abc_expression_l1326_132683


namespace tan_5pi_over_4_l1326_132669

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_5pi_over_4_l1326_132669


namespace solve_for_x_l1326_132665

theorem solve_for_x (x : ℝ) (h1 : 3 * x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 / 3 :=
by
  sorry

end solve_for_x_l1326_132665


namespace solve_for_x_l1326_132630

theorem solve_for_x (x : ℝ) (h : (8 - x)^2 = x^2) : x = 4 := 
by 
  sorry

end solve_for_x_l1326_132630


namespace proof_problems_l1326_132624

def otimes (a b : ℝ) : ℝ :=
  a * (1 - b)

theorem proof_problems :
  (otimes 2 (-2) = 6) ∧
  ¬ (∀ (a b : ℝ), otimes a b = otimes b a) ∧
  (∀ (a b : ℝ), a + b = 0 → otimes a a + otimes b b = 2 * a * b) ∧
  ¬ (∀ (a b : ℝ), otimes a b = 0 → a = 0) :=
by
  sorry
 
end proof_problems_l1326_132624


namespace sin_neg_1290_l1326_132636

theorem sin_neg_1290 : Real.sin (-(1290 : ℝ) * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_neg_1290_l1326_132636


namespace determine_ratio_l1326_132633

def p (x : ℝ) : ℝ := (x - 4) * (x + 3)
def q (x : ℝ) : ℝ := (x - 4) * (x + 3)

theorem determine_ratio : q 1 ≠ 0 ∧ p 1 / q 1 = 1 := by
  have hq : q 1 ≠ 0 := by
    simp [q]
    norm_num
  have hpq : p 1 / q 1 = 1 := by
    simp [p, q]
    norm_num
  exact ⟨hq, hpq⟩

end determine_ratio_l1326_132633


namespace final_price_wednesday_l1326_132627

theorem final_price_wednesday :
  let coffee_price := 6
  let cheesecake_price := 10
  let sandwich_price := 8
  let coffee_discount := 0.25
  let cheesecake_discount_wednesday := 0.10
  let additional_discount := 3
  let sales_tax := 0.05
  let discounted_coffee_price := coffee_price - coffee_price * coffee_discount
  let discounted_cheesecake_price := cheesecake_price - cheesecake_price * cheesecake_discount_wednesday
  let total_price_before_additional_discount := discounted_coffee_price + discounted_cheesecake_price + sandwich_price
  let total_price_after_additional_discount := total_price_before_additional_discount - additional_discount
  let total_price_with_tax := total_price_after_additional_discount + total_price_after_additional_discount * sales_tax
  let final_price := total_price_with_tax.round
  final_price = 19.43 :=
by
  sorry

end final_price_wednesday_l1326_132627


namespace tromino_covering_l1326_132687

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def chessboard_black_squares (n : ℕ) : ℕ := (n^2 + 1) / 2

def minimum_trominos (n : ℕ) : ℕ := (n^2 + 1) / 6

theorem tromino_covering (n : ℕ) (h_odd : is_odd n) (h_ge7 : n ≥ 7) :
  ∃ k : ℕ, chessboard_black_squares n = 3 * k ∧ (k = minimum_trominos n) :=
sorry

end tromino_covering_l1326_132687


namespace correct_option_is_B_l1326_132671

-- Define the conditions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (m : ℝ) : Prop := (-2 * m^2)^3 = -8 * m^6
def optionC (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2
def optionD (a b : ℝ) : Prop := 2 * a * b + 3 * a^2 * b = 5 * a^3 * b^2

-- The proof problem: which option is correct
theorem correct_option_is_B (m : ℝ) : optionB m := by
  sorry

end correct_option_is_B_l1326_132671


namespace ratio_five_to_one_l1326_132680

theorem ratio_five_to_one (x : ℕ) : (5 : ℕ) * 13 = 1 * x → x = 65 := 
by 
  intro h
  linarith

end ratio_five_to_one_l1326_132680


namespace monotonic_iff_m_ge_one_third_l1326_132659

-- Define the function f(x) = x^3 + x^2 + mx + 1
def f (x m : ℝ) : ℝ := x^3 + x^2 + m * x + 1

-- Define the derivative of the function f w.r.t x
def f' (x m : ℝ) : ℝ := 3 * x^2 + 2 * x + m

-- State the main theorem: f is monotonic on ℝ if and only if m ≥ 1/3
theorem monotonic_iff_m_ge_one_third (m : ℝ) :
  (∀ x y : ℝ, x < y → f x m ≤ f y m) ↔ (m ≥ 1 / 3) :=
sorry

end monotonic_iff_m_ge_one_third_l1326_132659


namespace smallest_value_square_l1326_132619

theorem smallest_value_square (z : ℂ) (hz : z.re > 0) (A : ℝ) :
  (A = 24 / 25) →
  abs ((Complex.abs z + 1 / Complex.abs z)^2 - (2 - 14 / 25)) = 0 :=
by
  sorry

end smallest_value_square_l1326_132619


namespace range_of_a_l1326_132648

-- Define the quadratic function f
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- State the theorem that describes the condition and proves the answer
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < 4 → x₂ < 4 → f a x₁ ≥ f a x₂) → a ≤ -3 :=
by
  -- The proof would go here; for now, we skip it
  sorry

end range_of_a_l1326_132648


namespace find_value_of_y_l1326_132686

noncomputable def angle_sum_triangle (A B C : ℝ) : Prop :=
A + B + C = 180

noncomputable def triangle_ABC : angle_sum_triangle 80 60 x := by
  sorry

noncomputable def triangle_CDE (x y : ℝ) : Prop :=
(x = 40) ∧ (90 + x + y = 180)

theorem find_value_of_y (x y : ℝ) 
  (h1 : angle_sum_triangle 80 60 x)
  (h2 : triangle_CDE x y) : 
  y = 50 := 
by
  sorry

end find_value_of_y_l1326_132686


namespace total_fruits_on_display_l1326_132634

-- Declare the variables and conditions as hypotheses
variables (apples oranges bananas : ℕ)
variables (h1 : apples = 2 * oranges)
variables (h2 : oranges = 2 * bananas)
variables (h3 : bananas = 5)

-- Define what we want to prove
theorem total_fruits_on_display : apples + oranges + bananas = 35 :=
by sorry

end total_fruits_on_display_l1326_132634


namespace dividend_ratio_l1326_132615

theorem dividend_ratio
  (expected_earnings_per_share : ℝ)
  (actual_earnings_per_share : ℝ)
  (dividend_per_share_increase : ℝ)
  (threshold_earnings_increase : ℝ)
  (shares_owned : ℕ)
  (h_expected_earnings : expected_earnings_per_share = 0.8)
  (h_actual_earnings : actual_earnings_per_share = 1.1)
  (h_dividend_increase : dividend_per_share_increase = 0.04)
  (h_threshold_increase : threshold_earnings_increase = 0.1)
  (h_shares_owned : shares_owned = 100)
  : (shares_owned * (expected_earnings_per_share + 
      (actual_earnings_per_share - expected_earnings_per_share) / threshold_earnings_increase * dividend_per_share_increase)) /
    (shares_owned * actual_earnings_per_share) = 46 / 55 :=
by
  sorry

end dividend_ratio_l1326_132615


namespace range_of_a_l1326_132644

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (|x-2| + |x+3| < a) → false) → a ≤ 5 :=
sorry

end range_of_a_l1326_132644


namespace intersection_points_zero_l1326_132614

theorem intersection_points_zero (a b c: ℝ) (h1: b^2 = a * c) (h2: a * c > 0) : 
  ∀ x: ℝ, ¬ (a * x^2 + b * x + c = 0) := 
by 
  sorry

end intersection_points_zero_l1326_132614


namespace wasting_water_notation_l1326_132600

theorem wasting_water_notation (saving_wasting : ℕ → ℤ)
  (h_pos : saving_wasting 30 = 30) :
  saving_wasting 10 = -10 :=
by
  sorry

end wasting_water_notation_l1326_132600


namespace total_area_correct_l1326_132639

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

end total_area_correct_l1326_132639


namespace total_geese_l1326_132698

/-- Definition of the number of geese that remain flying after each lake, 
    based on the given conditions. -/
def geese_after_lake (G : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then G else 2^(n : ℕ) - 1

/-- Main theorem stating the total number of geese in the flock. -/
theorem total_geese (n : ℕ) : ∃ (G : ℕ), geese_after_lake G n = 2^n - 1 :=
by
  sorry

end total_geese_l1326_132698


namespace minimum_value_problem_l1326_132603

theorem minimum_value_problem (a b c : ℝ) (hb : a > 0 ∧ b > 0 ∧ c > 0)
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) : 
  ∃ x, (x = 47) ∧ (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ x :=
by
  sorry

end minimum_value_problem_l1326_132603


namespace probability_all_white_balls_drawn_l1326_132635

theorem probability_all_white_balls_drawn (total_balls white_balls black_balls drawn_balls : ℕ) 
  (h_total : total_balls = 15) (h_white : white_balls = 7) (h_black : black_balls = 8) (h_drawn : drawn_balls = 7) :
  (Nat.choose 7 7 : ℚ) / (Nat.choose 15 7 : ℚ) = 1 / 6435 := by
sorry

end probability_all_white_balls_drawn_l1326_132635


namespace find_xy_l1326_132625

theorem find_xy :
  ∃ (x y : ℝ), (x - 14)^2 + (y - 15)^2 + (x - y)^2 = 1/3 ∧ x = 14 + 1/3 ∧ y = 14 + 2/3 :=
by
  sorry

end find_xy_l1326_132625


namespace cos_two_sum_l1326_132684

theorem cos_two_sum {α β : ℝ} 
  (h1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h2 : 3 * (Real.sin α + Real.cos α) ^ 2 - 2 * (Real.sin β + Real.cos β) ^ 2 = 1) :
  Real.cos (2 * (α + β)) = -1 / 3 :=
sorry

end cos_two_sum_l1326_132684


namespace replace_asterisks_l1326_132631

theorem replace_asterisks (x : ℝ) (h : (x / 21) * (x / 84) = 1) : x = 42 :=
sorry

end replace_asterisks_l1326_132631


namespace product_value_l1326_132616

noncomputable def product_of_integers (A B C D : ℕ) : ℕ :=
  A * B * C * D

theorem product_value :
  ∃ (A B C D : ℕ), A + B + C + D = 72 ∧ 
                    A + 2 = B - 2 ∧ 
                    A + 2 = C * 2 ∧ 
                    A + 2 = D / 2 ∧ 
                    product_of_integers A B C D = 64512 :=
by
  sorry

end product_value_l1326_132616


namespace perimeter_of_square_l1326_132617

theorem perimeter_of_square
  (s : ℝ) -- s is the side length of the square
  (h_divided_rectangles : ∀ r, r ∈ {r : ℝ × ℝ | r = (s, s / 6)} → true) -- the square is divided into six congruent rectangles
  (h_perimeter_rect : 2 * (s + s / 6) = 42) -- the perimeter of each of these rectangles is 42 inches
  : 4 * s = 72 := 
sorry

end perimeter_of_square_l1326_132617


namespace cake_fractions_l1326_132612

theorem cake_fractions (x y z : ℚ) 
  (h1 : x + y + z = 1)
  (h2 : 2 * z = x)
  (h3 : z = 1 / 2 * (y + 2 / 3 * x)) :
  x = 6 / 11 ∧ y = 2 / 11 ∧ z = 3 / 11 :=
sorry

end cake_fractions_l1326_132612


namespace solution_set_of_abs_inequality_l1326_132672

theorem solution_set_of_abs_inequality (x : ℝ) : 
  (x < 5 ↔ |x - 8| - |x - 4| > 2) :=
sorry

end solution_set_of_abs_inequality_l1326_132672


namespace part_one_part_two_l1326_132657

section part_one
variables {x : ℝ}

def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

theorem part_one : ∀ x : ℝ, f x ≥ 3 ↔ (x ≤ 1 ∨ x ≥ 4) := by
  sorry
end part_one

section part_two
variables {a x : ℝ}

def g (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

theorem part_two : (∀ x ∈ (Set.Icc 1 2), g a x ≤ |x - 4|) → (a ∈ Set.Icc (-3) 0) := by
  sorry
end part_two

end part_one_part_two_l1326_132657


namespace simplest_common_denominator_fraction_exist_l1326_132679

variable (x y : ℝ)

theorem simplest_common_denominator_fraction_exist :
  let d1 := x + y
  let d2 := x - y
  let d3 := x^2 - y^2
  (d3 = d1 * d2) → 
    ∀ n, (n = d1 * d2) → 
      (∃ m, (d1 * m = n) ∧ (d2 * m = n) ∧ (d3 * m = n)) :=
by
  sorry

end simplest_common_denominator_fraction_exist_l1326_132679


namespace sector_area_max_sector_area_l1326_132675

-- Definitions based on the given conditions
def perimeter : ℝ := 8
def central_angle (α : ℝ) : Prop := α = 2

-- Question 1: Find the area of the sector given the central angle is 2 rad
theorem sector_area (r l : ℝ) (h1 : 2 * r + l = perimeter) (h2 : l = 2 * r) : 
  (1/2) * r * l = 4 := 
by sorry

-- Question 2: Find the maximum area of the sector and the corresponding central angle
theorem max_sector_area (r l : ℝ) (h1 : 2 * r + l = perimeter) : 
  ∃ r, 0 < r ∧ r < 4 ∧ l = 8 - 2 * r ∧ 
  (1/2) * r * l = 4 ∧ l = 2 * r := 
by sorry

end sector_area_max_sector_area_l1326_132675


namespace melted_ice_cream_depth_l1326_132654

theorem melted_ice_cream_depth :
  ∀ (r_sphere r_cylinder : ℝ) (h : ℝ),
    r_sphere = 3 →
    r_cylinder = 10 →
    (4 / 3) * π * r_sphere^3 = 100 * π * h →
    h = 9 / 25 :=
  by
    intros r_sphere r_cylinder h
    intros hr_sphere hr_cylinder
    intros h_volume_eq
    sorry

end melted_ice_cream_depth_l1326_132654


namespace arctan_sum_in_right_triangle_l1326_132697

theorem arctan_sum_in_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : 
  (Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4) :=
sorry

end arctan_sum_in_right_triangle_l1326_132697


namespace solvable_eq_l1326_132645

theorem solvable_eq (x : ℝ) :
    Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6 →
    (x = 2 ∨ x = -2) :=
by
  sorry

end solvable_eq_l1326_132645


namespace compute_expression_l1326_132682

theorem compute_expression :
  (-9 * 5) - (-7 * -2) + (11 * -4) = -103 :=
by
  sorry

end compute_expression_l1326_132682


namespace fraction_calculation_l1326_132640

theorem fraction_calculation :
  ( (3 / 7 + 5 / 8 + 1 / 3) / (5 / 12 + 2 / 9) = 2097 / 966 ) :=
by
  sorry

end fraction_calculation_l1326_132640


namespace trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13_l1326_132699

theorem trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13 :
  (Real.cos (58 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) +
   Real.sin (58 * Real.pi / 180) * Real.sin (13 * Real.pi / 180) =
   Real.cos (45 * Real.pi / 180)) :=
sorry

end trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13_l1326_132699


namespace correct_statements_count_l1326_132658

theorem correct_statements_count (x : ℝ) :
  let inverse := (x > 0) → (x^2 > 0)
  let converse := (x^2 ≤ 0) → (x ≤ 0)
  let contrapositive := (x ≤ 0) → (x^2 ≤ 0)
  (∃ p : Prop, p = inverse ∨ p = converse ∧ p) ↔ 
  ¬ contrapositive →
  2 = 2 :=
by
  sorry

end correct_statements_count_l1326_132658


namespace speed_with_current_l1326_132691

-- Define the constants
def speed_of_current : ℝ := 2.5
def speed_against_current : ℝ := 20

-- Define the man's speed in still water
axiom speed_in_still_water : ℝ
axiom speed_against_current_eq : speed_in_still_water - speed_of_current = speed_against_current

-- The statement we need to prove
theorem speed_with_current : speed_in_still_water + speed_of_current = 25 := sorry

end speed_with_current_l1326_132691


namespace carnations_count_l1326_132696

theorem carnations_count (total_flowers : ℕ) (fract_rose : ℚ) (num_tulips : ℕ) (h1 : total_flowers = 40) (h2 : fract_rose = 2 / 5) (h3 : num_tulips = 10) :
  total_flowers - ((fract_rose * total_flowers) + num_tulips) = 14 := 
by
  sorry

end carnations_count_l1326_132696


namespace vectors_opposite_direction_l1326_132690

noncomputable def a : ℝ × ℝ := (-2, 4)
noncomputable def b : ℝ × ℝ := (1, -2)

theorem vectors_opposite_direction : a = (-2 : ℝ) • b :=
by
  sorry

end vectors_opposite_direction_l1326_132690


namespace weight_ratio_l1326_132655

noncomputable def weight_ratio_proof : Prop :=
  ∃ (R S : ℝ), 
  (R + S = 72) ∧ 
  (1.10 * R + 1.17 * S = 82.8) ∧ 
  (R / S = 1 / 2.5)

theorem weight_ratio : weight_ratio_proof := 
  by
    sorry

end weight_ratio_l1326_132655


namespace inscribed_sphere_radius_l1326_132620

theorem inscribed_sphere_radius 
  (V : ℝ) (S1 S2 S3 S4 : ℝ) (R : ℝ) :
  (1 / 3) * R * (S1 + S2 + S3 + S4) = V ↔ R = (3 * V) / (S1 + S2 + S3 + S4) := 
by
  sorry

end inscribed_sphere_radius_l1326_132620


namespace exists_m_for_division_l1326_132643

theorem exists_m_for_division (n : ℕ) (h : 0 < n) : ∃ m : ℕ, n ∣ (2016 ^ m + m) := by
  sorry

end exists_m_for_division_l1326_132643


namespace sqrt_inequality_sum_inverse_ge_9_l1326_132688

-- (1) Prove that \(\sqrt{3} + \sqrt{8} < 2 + \sqrt{7}\)
theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 8 < 2 + Real.sqrt 7 := sorry

-- (2) Prove that given \(a > 0, b > 0, c > 0\) and \(a + b + c = 1\), \(\frac{1}{a} + \frac{1}{b} + \frac{1}{c} \geq 9\)
theorem sum_inverse_ge_9 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) : 
    1 / a + 1 / b + 1 / c ≥ 9 := sorry

end sqrt_inequality_sum_inverse_ge_9_l1326_132688


namespace ratio_of_spinsters_to_cats_l1326_132673

theorem ratio_of_spinsters_to_cats (S C : ℕ) (hS : S = 12) (hC : C = S + 42) : S / gcd S C = 2 ∧ C / gcd S C = 9 :=
by
  -- skip proof (use sorry)
  sorry

end ratio_of_spinsters_to_cats_l1326_132673


namespace sum_first_40_terms_l1326_132649

-- Defining the sequence a_n following the given conditions
noncomputable def a : ℕ → ℝ
| 0 => 1
| 1 => 3
| n + 2 => a (n + 1) * a (n - 1)

-- Defining the sum of the first 40 terms of the sequence
noncomputable def S40 := (Finset.range 40).sum a

-- The theorem stating the desired property
theorem sum_first_40_terms : S40 = 60 :=
sorry

end sum_first_40_terms_l1326_132649


namespace find_y_l1326_132663

theorem find_y : ∃ y : ℕ, y > 0 ∧ (y + 3050) % 15 = 1234 % 15 ∧ y = 14 := 
by
  sorry

end find_y_l1326_132663


namespace proportion_exists_x_l1326_132674

theorem proportion_exists_x : ∃ x : ℕ, 1 * x = 3 * 4 :=
by
  sorry

end proportion_exists_x_l1326_132674


namespace variance_transformation_example_l1326_132694

def variance (X : List ℝ) : ℝ := sorry -- Assuming some definition of variance

theorem variance_transformation_example {n : ℕ} (X : List ℝ) (h_len : X.length = 2021) (h_var : variance X = 3) :
  variance (X.map (fun x => 3 * (x - 2))) = 27 := 
sorry

end variance_transformation_example_l1326_132694


namespace cafeteria_ordered_red_apples_l1326_132609

theorem cafeteria_ordered_red_apples
  (R : ℕ) 
  (h : (R + 17) - 10 = 32) : 
  R = 25 :=
sorry

end cafeteria_ordered_red_apples_l1326_132609


namespace sum_of_nonneg_real_numbers_inequality_l1326_132670

open BigOperators

variables {α : Type*} [LinearOrderedField α]

theorem sum_of_nonneg_real_numbers_inequality 
  (a : ℕ → α) (n : ℕ)
  (h_nonneg : ∀ i : ℕ, 0 ≤ a i) : 
  (∑ i in Finset.range n, (a i * (∑ j in Finset.range (i + 1), a j) * (∑ j in Finset.Icc i (n - 1), a j ^ 2))) 
  ≤ (∑ i in Finset.range n, (a i * (∑ j in Finset.range (i + 1), a j)) ^ 2) :=
sorry

end sum_of_nonneg_real_numbers_inequality_l1326_132670


namespace least_number_to_subtract_l1326_132693

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (h : n = 427398) (k : d = 13) (r_val : r = 2) : 
  ∃ x : ℕ, (n - x) % d = 0 ∧ r = x :=
by sorry

end least_number_to_subtract_l1326_132693


namespace vasya_correct_l1326_132668

theorem vasya_correct (x : ℝ) (h : x^2 + x + 1 = 0) : 
  x^2000 + x^1999 + x^1998 + 1000*x^1000 + 1000*x^999 + 1000*x^998 + 2000*x^3 + 2000*x^2 + 2000*x + 3000 = 3000 :=
by 
  sorry

end vasya_correct_l1326_132668


namespace luca_loss_years_l1326_132622

variable (months_in_year : ℕ := 12)
variable (barbi_kg_per_month : ℚ := 1.5)
variable (luca_kg_per_year : ℚ := 9)
variable (luca_additional_kg : ℚ := 81)

theorem luca_loss_years (barbi_yearly_loss : ℚ :=
                          barbi_kg_per_month * months_in_year) :
  (81 + barbi_yearly_loss) / luca_kg_per_year = 11 := by
  let total_loss_by_luca := 81 + barbi_yearly_loss
  sorry

end luca_loss_years_l1326_132622


namespace cannot_be_correct_average_l1326_132660

theorem cannot_be_correct_average (a : ℝ) (h_pos : a > 0) (h_median : a ≤ 12) : 
  ∀ avg, avg = (12 + a + 8 + 15 + 23) / 5 → avg ≠ 71 / 5 := 
by
  intro avg h_avg
  sorry

end cannot_be_correct_average_l1326_132660


namespace Janice_earnings_l1326_132638

theorem Janice_earnings (days_worked_per_week : ℕ) (earnings_per_day : ℕ) (overtime_shifts : ℕ) (overtime_earnings_per_shift : ℕ)
  (h1 : days_worked_per_week = 5)
  (h2 : earnings_per_day = 30)
  (h3 : overtime_shifts = 3)
  (h4 : overtime_earnings_per_shift = 15) :
  (days_worked_per_week * earnings_per_day) + (overtime_shifts * overtime_earnings_per_shift) = 195 :=
by {
  sorry
}

end Janice_earnings_l1326_132638


namespace math_problem_proof_l1326_132641

variable {a : ℝ} (ha : a > 0)

theorem math_problem_proof : ((36 * a^9)^4 * (63 * a^9)^4 = a^(72)) :=
by sorry

end math_problem_proof_l1326_132641


namespace min_value_of_f_min_value_achieved_min_value_f_l1326_132611

noncomputable def f (x : ℝ) := x + 2 / (2 * x + 1) - 1

theorem min_value_of_f : ∀ x : ℝ, x > 0 → f x ≥ 1/2 := 
by sorry

theorem min_value_achieved : f (1/2) = 1/2 := 
by sorry

theorem min_value_f : ∃ x : ℝ, x > 0 ∧ f x = 1/2 := 
⟨1/2, by norm_num, by sorry⟩

end min_value_of_f_min_value_achieved_min_value_f_l1326_132611


namespace subtract_rational_from_zero_yields_additive_inverse_l1326_132652

theorem subtract_rational_from_zero_yields_additive_inverse (a : ℚ) : 0 - a = -a := by
  sorry

end subtract_rational_from_zero_yields_additive_inverse_l1326_132652


namespace find_y_parallel_l1326_132678

-- Definitions
def a : ℝ × ℝ := (2, 3)
def b (y : ℝ) : ℝ × ℝ := (4, -1 + y)

-- Parallel condition implies proportional components
def parallel_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- The proof problem
theorem find_y_parallel : ∀ y : ℝ, parallel_vectors a (b y) → y = 7 :=
by
  sorry

end find_y_parallel_l1326_132678


namespace find_a_l1326_132602

theorem find_a (a : ℚ) (h : ∃ r s : ℚ, (r*x + s)^2 = ax^2 + 18*x + 16) : a = 81 / 16 := 
by sorry 

end find_a_l1326_132602


namespace total_admission_cost_l1326_132662

-- Define the standard admission cost
def standard_admission_cost : ℕ := 8

-- Define the discount if watched before 6 P.M.
def discount : ℕ := 3

-- Define the number of people in Kath's group
def number_of_people : ℕ := 1 + 2 + 3

-- Define the movie starting time (not directly used in calculation but part of the conditions)
def movie_start_time : ℕ := 16

-- Define the discounted admission cost
def discounted_admission_cost : ℕ := standard_admission_cost - discount

-- Prove that the total cost Kath will pay is $30
theorem total_admission_cost : number_of_people * discounted_admission_cost = 30 := by
  -- sorry is used to skip the proof
  sorry

end total_admission_cost_l1326_132662


namespace mitch_total_scoops_l1326_132647

theorem mitch_total_scoops :
  (3 : ℝ) / (1/3 : ℝ) + (2 : ℝ) / (1/3 : ℝ) = 15 :=
by
  sorry

end mitch_total_scoops_l1326_132647


namespace gcd_18_30_is_6_l1326_132632

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l1326_132632


namespace simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l1326_132605

variable (a b : ℤ)
def A : ℤ := b^2 - a^2 + 5 * a * b
def B : ℤ := 3 * a * b + 2 * b^2 - a^2

theorem simplify_2A_minus_B : 2 * A a b - B a b = -a^2 + 7 * a * b := by
  sorry

theorem evaluate_2A_minus_B_at_1_2 : 2 * A 1 2 - B 1 2 = 13 := by
  sorry

end simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l1326_132605


namespace center_of_circle_point_not_on_circle_l1326_132642

-- Definitions and conditions
def circle_eq (x y : ℝ) := x^2 - 6 * x + y^2 + 2 * y - 11 = 0

-- The problem statement split into two separate theorems

-- Proving the center of the circle is (3, -1)
theorem center_of_circle : 
  ∃ h k : ℝ, (∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 21) ∧ (h, k) = (3, -1) := sorry

-- Proving the point (5, -1) does not lie on the circle
theorem point_not_on_circle : ¬ circle_eq 5 (-1) := sorry

end center_of_circle_point_not_on_circle_l1326_132642


namespace contest_end_time_l1326_132695

def start_time : ℕ := 15 * 60 -- 3:00 p.m. in minutes from midnight
def duration : ℕ := 765 -- duration of the contest in minutes

theorem contest_end_time : start_time + duration = 3 * 60 + 45 := by
  -- start_time is 15 * 60 (3:00 p.m. in minutes)
  -- duration is 765 minutes
  -- end_time should be 3:45 a.m. which is 3 * 60 + 45 minutes from midnight
  sorry

end contest_end_time_l1326_132695


namespace quadratic_passing_point_l1326_132653

theorem quadratic_passing_point :
  ∃ (m : ℝ), (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = 8 → x = 0) →
  (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = -10 → x = -1) →
  (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = m → x = 5) →
  m = 638 := by
  sorry

end quadratic_passing_point_l1326_132653


namespace total_points_is_400_l1326_132689

-- Define the conditions as definitions in Lean 4 
def pointsPerEnemy : ℕ := 15
def bonusPoints : ℕ := 50
def totalEnemies : ℕ := 25
def enemiesLeftUndestroyed : ℕ := 5
def bonusesEarned : ℕ := 2

-- Calculate the total number of enemies defeated
def enemiesDefeated : ℕ := totalEnemies - enemiesLeftUndestroyed

-- Calculate the points from defeating enemies
def pointsFromEnemies := enemiesDefeated * pointsPerEnemy

-- Calculate the total bonus points
def totalBonusPoints := bonusesEarned * bonusPoints

-- The total points earned is the sum of points from enemies and bonus points
def totalPointsEarned := pointsFromEnemies + totalBonusPoints

-- Prove that the total points earned is equal to 400
theorem total_points_is_400 : totalPointsEarned = 400 := by
    sorry

end total_points_is_400_l1326_132689


namespace company_fund_initial_amount_l1326_132607

theorem company_fund_initial_amount (n : ℕ) :
  (70 * n + 75 = 80 * n - 20) →
  (n = 9) →
  (80 * n - 20 = 700) :=
by
  intros h1 h2
  rw [h2] at h1
  linarith

end company_fund_initial_amount_l1326_132607


namespace total_points_needed_l1326_132610

def num_students : ℕ := 25
def num_weeks : ℕ := 2
def vegetables_per_student_per_week : ℕ := 2
def points_per_vegetable : ℕ := 2

theorem total_points_needed : 
  (num_students * (vegetables_per_student_per_week * num_weeks) * points_per_vegetable) = 200 := by
  sorry

end total_points_needed_l1326_132610


namespace estimate_fish_population_l1326_132629

theorem estimate_fish_population :
  ∀ (initial_tagged: ℕ) (august_sample: ℕ) (tagged_in_august: ℕ) (leaving_rate: ℝ) (new_rate: ℝ),
  initial_tagged = 50 →
  august_sample = 80 →
  tagged_in_august = 4 →
  leaving_rate = 0.30 →
  new_rate = 0.45 →
  ∃ (april_population : ℕ),
  april_population = 550 :=
by
  intros initial_tagged august_sample tagged_in_august leaving_rate new_rate
  intros h_initial_tagged h_august_sample h_tagged_in_august h_leaving_rate h_new_rate
  existsi 550
  sorry

end estimate_fish_population_l1326_132629


namespace sum_gcd_lcm_60_429_l1326_132666

theorem sum_gcd_lcm_60_429 : 
  let a := 60
  let b := 429
  gcd a b + lcm a b = 8583 :=
by
  -- Definitions of a and b
  let a := 60
  let b := 429
  
  -- The GCD and LCM calculations would go here
  
  -- Proof body (skipped with 'sorry')
  sorry

end sum_gcd_lcm_60_429_l1326_132666


namespace sqrt_9025_squared_l1326_132618

-- Define the square root function and its properties
noncomputable def sqrt (x : ℕ) : ℕ := sorry

axiom sqrt_def (n : ℕ) (hn : 0 ≤ n) : (sqrt n) ^ 2 = n

-- Prove the specific case
theorem sqrt_9025_squared : (sqrt 9025) ^ 2 = 9025 :=
sorry

end sqrt_9025_squared_l1326_132618


namespace tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence_l1326_132601

open Real

theorem tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence (α β γ : ℝ) 
  (h1 : α + β + γ = π)  -- Assuming α, β, γ are angles in a triangle
  (h2 : tan α + tan γ = 2 * tan β) :
  sin (2 * α) + sin (2 * γ) = 2 * sin (2 * β) :=
by
  sorry

end tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence_l1326_132601


namespace num_females_math_not_english_is_15_l1326_132621

-- Define the conditions
def male_math := 120
def female_math := 80
def female_english := 120
def male_english := 80
def total_students := 260
def both_male := 75

def female_math_not_english : Nat :=
  female_math - (female_english + female_math - (total_students - (male_math + male_english - both_male)))

theorem num_females_math_not_english_is_15 :
  female_math_not_english = 15 :=
by
  -- This is where the proof will be, but for now, we use 'sorry' to skip it.
  sorry

end num_females_math_not_english_is_15_l1326_132621


namespace tangent_line_at_a1_one_zero_per_interval_l1326_132646

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

theorem tangent_line_at_a1 (a : ℝ) (h : a = 1) : 
  (∃ (m b : ℝ), ∀ x, f a x = m * x + b ∧ m = 2 ∧ b = 0) :=
by
  sorry

theorem one_zero_per_interval (a : ℝ) :
  (∃ x : ℝ, -1 < x ∧ x < 0 ∧ f a x = 0) ∧ (∃ x : ℝ, 0 < x ∧ f a x = 0) ↔ a < -1 :=
by
  sorry

end tangent_line_at_a1_one_zero_per_interval_l1326_132646


namespace total_ladders_climbed_in_inches_l1326_132692

-- Define the conditions as hypotheses
def keaton_ladder_length := 30
def keaton_climbs := 20
def reece_ladder_difference := 4
def reece_climbs := 15
def feet_to_inches := 12

-- Define the lengths climbed by Keaton and Reece
def keaton_total_feet := keaton_ladder_length * keaton_climbs
def reece_ladder_length := keaton_ladder_length - reece_ladder_difference
def reece_total_feet := reece_ladder_length * reece_climbs

-- Calculate the total feet climbed and convert to inches
def total_feet := keaton_total_feet + reece_total_feet
def total_inches := total_feet * feet_to_inches

-- Prove the final result
theorem total_ladders_climbed_in_inches : total_inches = 11880 := by
  sorry

end total_ladders_climbed_in_inches_l1326_132692


namespace alice_no_guarantee_win_when_N_is_18_l1326_132664

noncomputable def alice_cannot_guarantee_win : Prop :=
  ∀ (B : ℝ × ℝ) (P : ℕ → ℝ × ℝ),
    (∀ k, 0 ≤ k → k ≤ 18 → 
         dist (P (k + 1)) B < dist (P k) B ∨ dist (P (k + 1)) B ≥ dist (P k) B) →
    ∀ A : ℝ × ℝ, dist A B > 1 / 2020

theorem alice_no_guarantee_win_when_N_is_18 : alice_cannot_guarantee_win :=
sorry

end alice_no_guarantee_win_when_N_is_18_l1326_132664


namespace fewer_ducks_than_chickens_and_geese_l1326_132623

/-- There are 42 chickens and 48 ducks on the farm, and there are as many geese as there are chickens. 
Prove that there are 36 fewer ducks than the number of chickens and geese combined. -/
theorem fewer_ducks_than_chickens_and_geese (chickens ducks geese : ℕ)
  (h_chickens : chickens = 42)
  (h_ducks : ducks = 48)
  (h_geese : geese = chickens):
  ducks + 36 = chickens + geese :=
by
  sorry

end fewer_ducks_than_chickens_and_geese_l1326_132623


namespace shaded_area_l1326_132651

theorem shaded_area (PQ : ℝ) (n_squares : ℕ) (d_intersect : ℝ)
  (h1 : PQ = 8) (h2 : n_squares = 20) (h3 : d_intersect = 8) : ∃ (A : ℝ), A = 160 := 
by {
  sorry
}

end shaded_area_l1326_132651


namespace transistors_2004_l1326_132656

-- Definition of Moore's law specifying the initial amount and the doubling period
def moores_law (initial : ℕ) (years : ℕ) (doubling_period : ℕ) : ℕ :=
  initial * 2 ^ (years / doubling_period)

-- Condition: The number of transistors in 1992
def initial_1992 : ℕ := 2000000

-- Condition: The number of years between 1992 and 2004
def years_between : ℕ := 2004 - 1992

-- Condition: Doubling period every 2 years
def doubling_period : ℕ := 2

-- Goal: Prove the number of transistors in 2004 using the conditions above
theorem transistors_2004 : moores_law initial_1992 years_between doubling_period = 128000000 :=
by
  sorry

end transistors_2004_l1326_132656


namespace tan_sum_identity_l1326_132626

noncomputable def tan_25 := Real.tan (Real.pi / 180 * 25)
noncomputable def tan_35 := Real.tan (Real.pi / 180 * 35)
noncomputable def sqrt_3 := Real.sqrt 3

theorem tan_sum_identity :
  tan_25 + tan_35 + sqrt_3 * tan_25 * tan_35 = 1 :=
by
  sorry

end tan_sum_identity_l1326_132626


namespace rectangular_diagonal_length_l1326_132604

theorem rectangular_diagonal_length (x y z : ℝ) 
  (h_surface_area : 2 * (x * y + y * z + z * x) = 11)
  (h_edge_sum : x + y + z = 6) :
  Real.sqrt (x^2 + y^2 + z^2) = 5 := 
by
  sorry

end rectangular_diagonal_length_l1326_132604


namespace framing_feet_required_l1326_132628

noncomputable def original_width := 5
noncomputable def original_height := 7
noncomputable def enlargement_factor := 4
noncomputable def border_width := 3
noncomputable def inches_per_foot := 12

theorem framing_feet_required :
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (final_width + final_height)
  let framing_feet := perimeter / inches_per_foot
  framing_feet = 10 :=
by
  sorry

end framing_feet_required_l1326_132628


namespace common_root_implies_remaining_roots_l1326_132677

variables {R : Type*} [LinearOrderedField R]

theorem common_root_implies_remaining_roots
  (a b c x1 x2 x3 : R) 
  (h_non_zero_a : a ≠ 0)
  (h_non_zero_b : b ≠ 0)
  (h_non_zero_c : c ≠ 0)
  (h_a_ne_b : a ≠ b)
  (h_common_root1 : x1^2 + a*x1 + b*c = 0)
  (h_common_root2 : x1^2 + b*x1 + c*a = 0)
  (h_root2_eq : x2^2 + a*x2 + b*c = 0)
  (h_root3_eq : x3^2 + b*x3 + c*a = 0)
  : x2^2 + c*x2 + a*b = 0 ∧ x3^2 + c*x3 + a*b = 0 :=
sorry

end common_root_implies_remaining_roots_l1326_132677


namespace problem_l1326_132637

namespace arithmetic_sequence

def is_arithmetic_sequence (a : ℕ → ℚ) := ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem problem 
  (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_cond : a 1 + a 7 + a 13 = 4) : a 2 + a 12 = 8 / 3 :=
sorry

end arithmetic_sequence

end problem_l1326_132637


namespace good_number_sum_l1326_132613

def is_good (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem good_number_sum (a : ℕ) (h1 : a > 6) (h2 : is_good a) :
  ∃ x y : ℕ, is_good x ∧ is_good y ∧ a * (a + 1) = x * (x + 1) + 3 * y * (y + 1) :=
sorry

end good_number_sum_l1326_132613
