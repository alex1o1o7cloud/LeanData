import Mathlib

namespace inscribed_circle_radius_l87_8794

theorem inscribed_circle_radius :
  ∀ (a b c : ℝ), a = 3 → b = 6 → c = 18 → (∃ (r : ℝ), (1 / r) = (1 / a) + (1 / b) + (1 / c) + 4 * Real.sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c))) ∧ r = 9 / (5 + 6 * Real.sqrt 3)) :=
by
  intros a b c h₁ h₂ h₃
  sorry

end inscribed_circle_radius_l87_8794


namespace inequality_solution_l87_8704

theorem inequality_solution (x : ℝ) (h : x > -4/3) : 2 - 1 / (3 * x + 4) < 5 :=
sorry

end inequality_solution_l87_8704


namespace cube_root_squared_l87_8713

noncomputable def solve_for_x (x : ℝ) : Prop :=
  (x^(1/3))^2 = 81 → x = 729

theorem cube_root_squared (x : ℝ) :
  solve_for_x x :=
by
  sorry

end cube_root_squared_l87_8713


namespace distribute_books_l87_8771

theorem distribute_books : 
  let total_ways := 4^5
  let subtract_one_student_none := 4 * 3^5
  let add_two_students_none := 6 * 2^5
  total_ways - subtract_one_student_none + add_two_students_none = 240 :=
by
  -- Definitions based on conditions in a)
  let total_ways := 4^5
  let subtract_one_student_none := 4 * 3^5
  let add_two_students_none := 6 * 2^5

  -- The final calculation
  have h : total_ways - subtract_one_student_none + add_two_students_none = 240 := by sorry
  exact h

end distribute_books_l87_8771


namespace solve_equation_l87_8731

theorem solve_equation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 → (x + 1) / (x - 1) = 1 / (x - 2) + 1 → x = 3 := by
  sorry

end solve_equation_l87_8731


namespace find_c_l87_8700

theorem find_c (a b c : ℝ) : 
  (a * x^2 + b * x - 5) * (a * x^2 + b * x + 25) + c = (a * x^2 + b * x + 10)^2 → 
  c = 225 :=
by sorry

end find_c_l87_8700


namespace union_sets_S_T_l87_8779

open Set Int

def S : Set Int := { s : Int | ∃ n : Int, s = 2 * n + 1 }
def T : Set Int := { t : Int | ∃ n : Int, t = 4 * n + 1 }

theorem union_sets_S_T : S ∪ T = S := 
by sorry

end union_sets_S_T_l87_8779


namespace volume_of_cube_l87_8751

theorem volume_of_cube (a : ℕ) (h : (a^3 - a = a^3 - 5)) : a^3 = 125 :=
by {
  -- The necessary algebraic manipulation follows
  sorry
}

end volume_of_cube_l87_8751


namespace backup_settings_required_l87_8720

-- Definitions for the given conditions
def weight_of_silverware_piece : ℕ := 4
def pieces_of_silverware_per_setting : ℕ := 3
def weight_of_plate : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def total_weight_ounces : ℕ := 5040

-- Statement to prove
theorem backup_settings_required :
  (total_weight_ounces - 
     (tables * settings_per_table) * 
       (pieces_of_silverware_per_setting * weight_of_silverware_piece + 
        plates_per_setting * weight_of_plate)) /
  (pieces_of_silverware_per_setting * weight_of_silverware_piece + 
   plates_per_setting * weight_of_plate) = 20 := 
by sorry

end backup_settings_required_l87_8720


namespace inequality_of_distinct_positives_l87_8785

variable {a b c : ℝ}

theorem inequality_of_distinct_positives (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
(habc : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) :=
by
  sorry

end inequality_of_distinct_positives_l87_8785


namespace sum_difference_20_l87_8701

def sum_of_even_integers (n : ℕ) : ℕ := (n / 2) * (2 + 2 * (n - 1))

def sum_of_odd_integers (n : ℕ) : ℕ := (n / 2) * (1 + 2 * (n - 1))

theorem sum_difference_20 : sum_of_even_integers (20) - sum_of_odd_integers (20) = 20 := by
  sorry

end sum_difference_20_l87_8701


namespace fraction_addition_l87_8775

variable {a b : ℚ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4)

theorem fraction_addition (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4) : (a + b) / b = 7 / 4 :=
  sorry

end fraction_addition_l87_8775


namespace vector_line_form_to_slope_intercept_l87_8758

variable (x y : ℝ)

theorem vector_line_form_to_slope_intercept :
  (∀ (x y : ℝ), ((-1) * (x - 3) + 2 * (y + 4) = 0) ↔ (y = (-1/2) * x - 11/2)) :=
by
  sorry

end vector_line_form_to_slope_intercept_l87_8758


namespace Nina_money_l87_8727

theorem Nina_money : ∃ (M : ℝ) (W : ℝ), M = 10 * W ∧ M = 14 * (W - 3) ∧ M = 105 :=
by
  sorry

end Nina_money_l87_8727


namespace line_through_point_equal_intercepts_l87_8759

theorem line_through_point_equal_intercepts (P : ℝ × ℝ) (x y a : ℝ) (k : ℝ) 
  (hP : P = (2, 3))
  (hx : x / a + y / a = 1 ∨ (P.fst * k - P.snd = 0)) :
  (x + y - 5 = 0 ∨ 3 * P.fst - 2 * P.snd = 0) := by
  sorry

end line_through_point_equal_intercepts_l87_8759


namespace binom_10_3_l87_8749

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l87_8749


namespace roses_left_unsold_l87_8732

def price_per_rose : ℕ := 4
def initial_roses : ℕ := 13
def total_earned : ℕ := 36

theorem roses_left_unsold : (initial_roses - (total_earned / price_per_rose) = 4) :=
by
  sorry

end roses_left_unsold_l87_8732


namespace total_cards_is_56_l87_8742

-- Let n be the number of Pokemon cards each person has
def n : Nat := 14

-- Let k be the number of people
def k : Nat := 4

-- Total number of Pokemon cards
def total_cards : Nat := n * k

-- Prove that the total number of Pokemon cards is 56
theorem total_cards_is_56 : total_cards = 56 := by
  sorry

end total_cards_is_56_l87_8742


namespace polynomial_value_at_minus_2_l87_8783

-- Define the polynomial f(x)
def f (x : ℤ) := x^6 - 5 * x^5 + 6 * x^4 + x^2 + 3 * x + 2

-- Define the evaluation point
def x_val : ℤ := -2

-- State the theorem we want to prove
theorem polynomial_value_at_minus_2 : f x_val = 320 := 
by sorry

end polynomial_value_at_minus_2_l87_8783


namespace expression_eval_l87_8719

theorem expression_eval (a b c d : ℝ) :
  a * b + c - d = a * (b + c - d) :=
sorry

end expression_eval_l87_8719


namespace incorrect_conclusions_l87_8740

theorem incorrect_conclusions
  (h1 : ∃ (y x : ℝ), (¬∃ a b : ℝ, a < 0 ∧ y = a * x + b) ∧ ∃ a b : ℝ, y = 2.347 * x - 6.423)
  (h2 : ∃ (y x : ℝ), (∃ a b : ℝ, a < 0 ∧ y = a * x + b) ∧ y = -3.476 * x + 5.648)
  (h3 : ∃ (y x : ℝ), (∃ a b : ℝ, a > 0 ∧ y = a * x + b) ∧ y = 5.437 * x + 8.493)
  (h4 : ∃ (y x : ℝ), (¬∃ a b : ℝ, a > 0 ∧ y = a * x + b) ∧ y = -4.326 * x - 4.578) :
  (∃ (y x : ℝ), y = 2.347 * x - 6.423 ∧ (¬∃ a b : ℝ, a < 0 ∧ y = a * x + b)) ∧
  (∃ (y x : ℝ), y = -4.326 * x - 4.578 ∧ (¬∃ a b : ℝ, a > 0 ∧ y = a * x + b)) :=
by {
  sorry
}

end incorrect_conclusions_l87_8740


namespace problem1_problem2_l87_8776

theorem problem1 (α : ℝ) (hα : Real.tan α = 2) :
    (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := 
sorry

theorem problem2 (α : ℝ) (hα : Real.tan α = 2) :
    (Real.sin (↑(π/2) + α) * Real.cos (↑(5*π/2) - α) * Real.tan (↑(-π) + α)) / 
    (Real.tan (↑(7*π) - α) * Real.sin (↑π + α)) = Real.cos α := 
sorry

end problem1_problem2_l87_8776


namespace calculate_price_l87_8772

-- Define variables for prices
def sugar_price_in_terms_of_salt (T : ℝ) : ℝ := 2 * T
def rice_price_in_terms_of_salt (T : ℝ) : ℝ := 3 * T
def apple_price : ℝ := 1.50
def pepper_price : ℝ := 1.25

-- Define pricing conditions
def condition_1 (T : ℝ) : Prop :=
  5 * (sugar_price_in_terms_of_salt T) + 3 * T + 2 * (rice_price_in_terms_of_salt T) + 3 * apple_price + 4 * pepper_price = 35

def condition_2 (T : ℝ) : Prop :=
  4 * (sugar_price_in_terms_of_salt T) + 2 * T + 1 * (rice_price_in_terms_of_salt T) + 2 * apple_price + 3 * pepper_price = 24

-- Define final price calculation with discounts
def total_price (T : ℝ) : ℝ :=
  8 * (sugar_price_in_terms_of_salt T) * 0.9 +
  5 * T +
  (rice_price_in_terms_of_salt T + 3 * (rice_price_in_terms_of_salt T - 0.5)) +
  -- adding two free apples to the count
  5 * apple_price +
  6 * pepper_price

-- Main theorem to prove
theorem calculate_price (T : ℝ) (h1 : condition_1 T) (h2 : condition_2 T) :
  total_price T = 55.64 :=
sorry -- proof omitted

end calculate_price_l87_8772


namespace recycling_money_l87_8755

theorem recycling_money (cans_per_unit : ℕ) (payment_per_unit_cans : ℝ) 
  (newspapers_per_unit : ℕ) (payment_per_unit_newspapers : ℝ) 
  (total_cans : ℕ) (total_newspapers : ℕ) : 
  cans_per_unit = 12 → payment_per_unit_cans = 0.50 → 
  newspapers_per_unit = 5 → payment_per_unit_newspapers = 1.50 → 
  total_cans = 144 → total_newspapers = 20 → 
  (total_cans / cans_per_unit) * payment_per_unit_cans + 
  (total_newspapers / newspapers_per_unit) * payment_per_unit_newspapers = 12 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  done

end recycling_money_l87_8755


namespace arithmetic_sequence_25th_term_l87_8796

theorem arithmetic_sequence_25th_term (a1 a2 : ℕ) (d : ℕ) (n : ℕ) (h1 : a1 = 2) (h2 : a2 = 5) (h3 : d = a2 - a1) (h4 : n = 25) :
  a1 + (n - 1) * d = 74 :=
by
  sorry

end arithmetic_sequence_25th_term_l87_8796


namespace min_cost_per_student_is_80_l87_8788

def num_students : ℕ := 48
def swims_per_student : ℕ := 8
def cost_per_card : ℕ := 240
def cost_per_bus : ℕ := 40

def total_swims : ℕ := num_students * swims_per_student

def min_cost_per_student : ℕ :=
  let n := 8
  let c := total_swims / n
  let total_cost := cost_per_card * n + cost_per_bus * c
  total_cost / num_students

theorem min_cost_per_student_is_80 :
  min_cost_per_student = 80 :=
sorry

end min_cost_per_student_is_80_l87_8788


namespace watch_cost_price_l87_8773

theorem watch_cost_price (cost_price : ℝ)
  (h1 : SP_loss = 0.90 * cost_price)
  (h2 : SP_gain = 1.08 * cost_price)
  (h3 : SP_gain - SP_loss = 540) :
  cost_price = 3000 := 
sorry

end watch_cost_price_l87_8773


namespace find_b_l87_8729

theorem find_b (b x : ℝ) (h₁ : 5 * x + 3 = b * x - 22) (h₂ : x = 5) : b = 10 := 
by 
  sorry

end find_b_l87_8729


namespace condition_p_neither_sufficient_nor_necessary_l87_8712

theorem condition_p_neither_sufficient_nor_necessary
  (x : ℝ) :
  (1/x ≤ 1 → x^2 - 2 * x ≥ 0) = false ∧ 
  (x^2 - 2 * x ≥ 0 → 1/x ≤ 1) = false := 
by 
  sorry

end condition_p_neither_sufficient_nor_necessary_l87_8712


namespace chocolate_discount_l87_8767

theorem chocolate_discount :
    let original_cost : ℝ := 2
    let final_price : ℝ := 1.43
    let discount := original_cost - final_price
    discount = 0.57 := by
  sorry

end chocolate_discount_l87_8767


namespace price_of_adult_ticket_l87_8716

theorem price_of_adult_ticket (total_payment : ℕ) (child_price : ℕ) (difference : ℕ) (children : ℕ) (adults : ℕ) (A : ℕ)
  (h1 : total_payment = 720) 
  (h2 : child_price = 8) 
  (h3 : difference = 25) 
  (h4 : children = 15)
  (h5 : adults = children + difference)
  (h6 : total_payment = children * child_price + adults * A) :
  A = 15 :=
by
  sorry

end price_of_adult_ticket_l87_8716


namespace candy_necklaces_left_l87_8799

theorem candy_necklaces_left (total_packs : ℕ) (candy_per_pack : ℕ) 
  (opened_packs : ℕ) (candy_necklaces : ℕ)
  (h1 : total_packs = 9) 
  (h2 : candy_per_pack = 8) 
  (h3 : opened_packs = 4)
  (h4 : candy_necklaces = total_packs * candy_per_pack) :
  (total_packs - opened_packs) * candy_per_pack = 40 :=
by
  sorry

end candy_necklaces_left_l87_8799


namespace digit_A_in_comb_60_15_correct_l87_8781

-- Define the combination function
def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main theorem we want to prove
theorem digit_A_in_comb_60_15_correct : 
  ∃ (A : ℕ), (660 * 10^9 + A * 10^8 + B * 10^7 + 5 * 10^6 + A * 10^4 + 640 * 10^1 + A) = comb 60 15 ∧ A = 6 :=
by
  sorry

end digit_A_in_comb_60_15_correct_l87_8781


namespace problem_l87_8710

noncomputable def f : ℝ → ℝ := sorry

variables (x : ℝ) (a b c : ℝ)
  (h1 : ∀ x, f (x + 1) = f (-x + 1))
  (h2 : ∀ x, 1 < x → f x ≤ f (x - 1))
  (ha : a = f 2)
  (hb : b = f (Real.log 2 / Real.log 3))
  (hc : c = f (1 / 2))

theorem problem (h : a = f 2 ∧ b = f (Real.log 2 / Real.log 3) ∧ c = f (1 / 2)) : 
  a < c ∧ c < b := sorry

end problem_l87_8710


namespace speed_first_hour_l87_8739

variable (x : ℕ)

-- Definitions based on conditions
def total_distance (x : ℕ) : ℕ := x + 50
def average_speed (x : ℕ) : Prop := (total_distance x) / 2 = 70

-- Theorem statement
theorem speed_first_hour : ∃ x, average_speed x ∧ x = 90 := by
  sorry

end speed_first_hour_l87_8739


namespace num_special_matrices_l87_8770

open Matrix

theorem num_special_matrices :
  ∃ (M : Matrix (Fin 4) (Fin 4) ℕ), 
    (∀ i j, 1 ≤ M i j ∧ M i j ≤ 16) ∧ 
    (∀ i j, i < j → M i j < M i (j + 1)) ∧ 
    (∀ i j, i < j → M i j < M (i + 1) j) ∧ 
    (∀ i, i < 3 → M i i < M (i + 1) (i + 1)) ∧ 
    (∀ i, i < 3 → M i (3 - i) < M (i + 1) (2 - i)) ∧ 
    (∃ n, n = 144) :=
sorry

end num_special_matrices_l87_8770


namespace find_parameters_l87_8761

theorem find_parameters (s h : ℝ) :
  (∀ (x y t : ℝ), (x = s + 3 * t) ∧ (y = 2 + h * t) ∧ (y = 5 * x - 7)) → (s = 9 / 5 ∧ h = 15) :=
by
  sorry

end find_parameters_l87_8761


namespace ellipse_properties_l87_8738

-- Define the ellipse E with its given properties
def is_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define properties related to the intersection points and lines
def intersects (l : ℝ → ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  l (-1) = 0 ∧ 
  is_ellipse x₁ (l x₁) ∧ 
  is_ellipse x₂ (l x₂) ∧ 
  y₁ = l x₁ ∧ 
  y₂ = l x₂

def perpendicular_lines (l1 l2 : ℝ → ℝ) : Prop :=
  ∀ x, l1 x * l2 x = -1

-- Define the main theorem
theorem ellipse_properties :
  (∀ (x y : ℝ), is_ellipse x y) →
  (∀ (l1 l2 : ℝ → ℝ) 
     (A B C D : ℝ × ℝ),
      intersects l1 A.1 A.2 B.1 B.2 → 
      intersects l2 C.1 C.2 D.1 D.2 → 
      perpendicular_lines l1 l2 → 
      12 * (|A.1 - B.1| + |C.1 - D.1|) = 7 * |A.1 - B.1| * |C.1 - D.1|) :=
by 
  sorry

end ellipse_properties_l87_8738


namespace more_larger_boxes_l87_8709

theorem more_larger_boxes (S L : ℕ) 
  (h1 : 12 * S + 16 * L = 480)
  (h2 : S + L = 32)
  (h3 : L > S) : L - S = 16 := 
sorry

end more_larger_boxes_l87_8709


namespace product_of_factors_l87_8743

theorem product_of_factors : (2.1 * (53.2 - 0.2) = 111.3) := by
  sorry

end product_of_factors_l87_8743


namespace base_conversion_min_sum_l87_8703

theorem base_conversion_min_sum : ∃ a b : ℕ, a > 6 ∧ b > 6 ∧ (6 * a + 3 = 3 * b + 6) ∧ (a + b = 20) :=
by
  sorry

end base_conversion_min_sum_l87_8703


namespace total_skips_l87_8748

-- Definitions based on conditions
def fifth_throw := 8
def fourth_throw := fifth_throw - 1
def third_throw := fourth_throw + 3
def second_throw := third_throw / 2
def first_throw := second_throw - 2

-- Statement of the proof problem
theorem total_skips : first_throw + second_throw + third_throw + fourth_throw + fifth_throw = 33 := by
  sorry

end total_skips_l87_8748


namespace axisymmetric_and_centrally_symmetric_l87_8728

def Polygon := String

def EquilateralTriangle : Polygon := "EquilateralTriangle"
def Square : Polygon := "Square"
def RegularPentagon : Polygon := "RegularPentagon"
def RegularHexagon : Polygon := "RegularHexagon"

def is_axisymmetric (p : Polygon) : Prop := 
  p = EquilateralTriangle ∨ p = Square ∨ p = RegularPentagon ∨ p = RegularHexagon

def is_centrally_symmetric (p : Polygon) : Prop := 
  p = Square ∨ p = RegularHexagon

theorem axisymmetric_and_centrally_symmetric :
  {p : Polygon | is_axisymmetric p ∧ is_centrally_symmetric p} = {Square, RegularHexagon} :=
by
  sorry

end axisymmetric_and_centrally_symmetric_l87_8728


namespace evaluate_expression_l87_8786

theorem evaluate_expression : (20^40) / (40^20) = 10^20 := by
  sorry

end evaluate_expression_l87_8786


namespace solution_set_of_inequality_min_value_of_expression_l87_8752

def f (x : ℝ) : ℝ := |x + 1| - |2 * x - 2|

-- (I) Prove that the solution set of the inequality f(x) ≥ x - 1 is [0, 2]
theorem solution_set_of_inequality 
  (x : ℝ) : f x ≥ x - 1 ↔ 0 ≤ x ∧ x ≤ 2 := 
sorry

-- (II) Given the maximum value m of f(x) is 2 and a + b + c = 2, prove the minimum value of b^2/a + c^2/b + a^2/c is 2
theorem min_value_of_expression
  (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 2) :
  b^2 / a + c^2 / b + a^2 / c ≥ 2 :=
sorry

end solution_set_of_inequality_min_value_of_expression_l87_8752


namespace second_man_start_time_l87_8769

theorem second_man_start_time (P Q : Type) (departure_time_P departure_time_Q meeting_time arrival_time_P arrival_time_Q : ℕ) 
(distance speed : ℝ) (first_man_speed second_man_speed : ℕ → ℝ)
(h1 : departure_time_P = 6) 
(h2 : arrival_time_Q = 10) 
(h3 : arrival_time_P = 12) 
(h4 : meeting_time = 9) 
(h5 : ∀ t, 0 ≤ t ∧ t ≤ 4 → first_man_speed t = distance / 4)
(h6 : ∀ t, second_man_speed t = distance / 4)
(h7 : ∀ t, second_man_speed t * (meeting_time - t) = (3 * distance / 4))
: departure_time_Q = departure_time_P :=
by 
  sorry

end second_man_start_time_l87_8769


namespace symmetric_point_correct_l87_8707

-- Define the point P in a three-dimensional Cartesian coordinate system.
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the function to find the symmetric point with respect to the x-axis.
def symmetricWithRespectToXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Given point P(1, -2, 3).
def P : Point3D := { x := 1, y := -2, z := 3 }

-- The expected symmetric point
def symmetricP : Point3D := { x := 1, y := 2, z := -3 }

-- The proposition we need to prove
theorem symmetric_point_correct :
  symmetricWithRespectToXAxis P = symmetricP :=
by
  sorry

end symmetric_point_correct_l87_8707


namespace initial_processing_capacity_l87_8754

variable (x y z : ℕ)

-- Conditions
def initial_condition : Prop := x * y = 38880
def after_modernization : Prop := (x + 3) * z = 44800
def capacity_increased : Prop := y < z
def minimum_machines : Prop := x ≥ 20

-- Prove that the initial daily processing capacity y is 1215
theorem initial_processing_capacity
  (h1 : initial_condition x y)
  (h2 : after_modernization x z)
  (h3 : capacity_increased y z)
  (h4 : minimum_machines x) :
  y = 1215 := by
  sorry

end initial_processing_capacity_l87_8754


namespace sum_remainder_l87_8765

theorem sum_remainder (m : ℤ) : ((9 - m) + (m + 5)) % 8 = 6 :=
by
  sorry

end sum_remainder_l87_8765


namespace range_of_f_when_a_0_range_of_a_for_three_zeros_l87_8733

noncomputable def f_part1 (x : ℝ) : ℝ :=
if h : x ≤ 0 then 2 ^ x else x ^ 2

theorem range_of_f_when_a_0 : Set.range f_part1 = {y : ℝ | 0 < y} := by
  sorry

noncomputable def f_part2 (a : ℝ) (x : ℝ) : ℝ :=
if h : x ≤ 0 then 2 ^ x - a else x ^ 2 - 3 * a * x + a

def discriminant (a : ℝ) (x : ℝ) : ℝ := (3 * a) ^ 2 - 4 * a

theorem range_of_a_for_three_zeros (a : ℝ) :
  (∀ x : ℝ, f_part2 a x = 0) → (4 / 9 < a ∧ a ≤ 1) := by
  sorry

end range_of_f_when_a_0_range_of_a_for_three_zeros_l87_8733


namespace arithmetic_sequence_value_l87_8735

theorem arithmetic_sequence_value 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : 4 * a 3 + a 11 - 3 * a 5 = 10) : 
  (1 / 5 * a 4 = 1) := 
by
  sorry

end arithmetic_sequence_value_l87_8735


namespace hypotenuse_length_l87_8711

theorem hypotenuse_length (a b c : ℝ) (h₀ : a^2 + b^2 + c^2 = 1800) (h₁ : c^2 = a^2 + b^2) : c = 30 :=
by
  sorry

end hypotenuse_length_l87_8711


namespace minimum_value_inequality_l87_8795

def minimum_value_inequality_problem : Prop :=
∀ (a b : ℝ), (0 < a) → (0 < b) → (a + 3 * b = 1) → (1 / a + 1 / (3 * b)) = 4

theorem minimum_value_inequality : minimum_value_inequality_problem :=
sorry

end minimum_value_inequality_l87_8795


namespace hens_count_l87_8798

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 136) : H = 28 := by
  sorry

end hens_count_l87_8798


namespace fraction_simplification_l87_8736

theorem fraction_simplification (a b d : ℝ) (h : a^2 + d^2 - b^2 + 2 * a * d ≠ 0) :
  (a^2 + b^2 + d^2 + 2 * b * d) / (a^2 + d^2 - b^2 + 2 * a * d) = (a^2 + (b + d)^2) / ((a + d)^2 + a^2 - b^2) :=
sorry

end fraction_simplification_l87_8736


namespace garden_area_increase_l87_8717

-- Definitions corresponding to the conditions
def length := 40
def width := 20
def original_perimeter := 2 * (length + width)

-- Definition of the correct answer calculation
def original_area := length * width
def side_length := original_perimeter / 4
def new_area := side_length * side_length
def area_increase := new_area - original_area

-- The statement to be proven
theorem garden_area_increase : area_increase = 100 :=
by sorry

end garden_area_increase_l87_8717


namespace smallest_positive_angle_terminal_side_eq_l87_8730

theorem smallest_positive_angle_terminal_side_eq (n : ℤ) :
  (0 ≤ n % 360 ∧ n % 360 < 360) → (∃ k : ℤ, n = -2015 + k * 360 ) → n % 360 = 145 :=
by
  sorry

end smallest_positive_angle_terminal_side_eq_l87_8730


namespace rectangle_area_problem_l87_8702

theorem rectangle_area_problem (l w l1 l2 w1 w2 : ℝ) (h1 : l = l1 + l2) (h2 : w = w1 + w2) 
  (h3 : l1 * w1 = 12) (h4 : l2 * w1 = 15) (h5 : l1 * w2 = 12) 
  (h6 : l2 * w2 = 8) (h7 : w1 * l2 = 18) (h8 : l1 * w2 = 20) :
  l2 * w1 = 18 :=
sorry

end rectangle_area_problem_l87_8702


namespace find_m_l87_8724

theorem find_m : ∃ m : ℤ, 2^5 - 7 = 3^3 + m ∧ m = -2 :=
by
  use -2
  sorry

end find_m_l87_8724


namespace positional_relationship_l87_8706

theorem positional_relationship 
  (m n : ℝ) 
  (h_points_on_ellipse : (m^2 / 4) + (n^2 / 3) = 1)
  (h_relation : n^2 = 3 - (3/4) * m^2) : 
  (∃ x y : ℝ, (x^2 + y^2 = 1/3) ∧ (m * x + n * y + 1 = 0)) ∨ 
  (∀ x y : ℝ, (x^2 + y^2 = 1/3) → (m * x + n * y + 1 ≠ 0)) :=
sorry

end positional_relationship_l87_8706


namespace delta_max_success_ratio_l87_8741

theorem delta_max_success_ratio :
  ∃ (x y z w : ℕ),
  (0 < x ∧ x < (7 * y) / 12) ∧
  (0 < z ∧ z < (5 * w) / 8) ∧
  (y + w = 600) ∧
  (35 * x + 28 * z < 4200) ∧
  (x + z = 150) ∧ 
  (x + z) / 600 = 1 / 4 :=
by sorry

end delta_max_success_ratio_l87_8741


namespace cos_alpha_value_l87_8745

theorem cos_alpha_value (θ α : Real) (P : Real × Real)
  (hP : P = (-3/5, 4/5))
  (hθ : θ = Real.arccos (-3/5))
  (hαθ : α = θ - Real.pi / 3) :
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 := 
by 
  sorry

end cos_alpha_value_l87_8745


namespace intersection_points_calculation_l87_8756

-- Define the quadratic function and related functions
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def u (a b c x : ℝ) : ℝ := - f a b c (-x)
def v (a b c x : ℝ) : ℝ := f a b c (x + 1)

-- Define the number of intersection points
def m : ℝ := 1
def n : ℝ := 0

-- The proof goal
theorem intersection_points_calculation (a b c : ℝ) : 7 * m + 3 * n = 7 :=
by sorry

end intersection_points_calculation_l87_8756


namespace clock_overlap_24_hours_l87_8722

theorem clock_overlap_24_hours (hour_rotations : ℕ) (minute_rotations : ℕ) 
  (h_hour_rotations: hour_rotations = 2) 
  (h_minute_rotations: minute_rotations = 24) : 
  ∃ (overlaps : ℕ), overlaps = 22 := 
by 
  sorry

end clock_overlap_24_hours_l87_8722


namespace max_consecutive_sum_l87_8723

theorem max_consecutive_sum (N a : ℕ) (h : N * (2 * a + N - 1) = 240) : N ≤ 15 :=
by
  -- proof goes here
  sorry

end max_consecutive_sum_l87_8723


namespace abs_diff_squares_l87_8718

theorem abs_diff_squares (a b : ℤ) (ha : a = 103) (hb : b = 97) : |a^2 - b^2| = 1200 :=
by
  sorry

end abs_diff_squares_l87_8718


namespace no_symmetry_line_for_exponential_l87_8753

theorem no_symmetry_line_for_exponential : ¬ ∃ l : ℝ → ℝ, ∀ x : ℝ, (2 ^ x) = l (2 ^ (2 * l x - x)) := 
sorry

end no_symmetry_line_for_exponential_l87_8753


namespace termite_ridden_fraction_l87_8708

theorem termite_ridden_fraction (T : ℝ) 
    (h1 : 5/8 * T > 0)
    (h2 : 3/8 * T = 0.125) : T = 1/8 :=
by
  sorry

end termite_ridden_fraction_l87_8708


namespace parallel_vectors_m_value_l87_8747

theorem parallel_vectors_m_value :
  ∀ (m : ℝ), (∀ k : ℝ, (1 : ℝ) = k * m ∧ (-2) = k * (-1)) -> m = (1 / 2) :=
by
  intros m h
  sorry

end parallel_vectors_m_value_l87_8747


namespace inequality_equivalence_l87_8778

theorem inequality_equivalence (x : ℝ) : 
  (x + 2) / (x - 1) ≥ 0 ↔ (x + 2) * (x - 1) ≥ 0 :=
sorry

end inequality_equivalence_l87_8778


namespace simplify_and_evaluate_expression_l87_8715

theorem simplify_and_evaluate_expression (x : ℝ) (hx : x = 4) :
  (1 / (x + 2) + 1) / ((x^2 + 6 * x + 9) / (x^2 - 4)) = 2 / 7 :=
by
  sorry

end simplify_and_evaluate_expression_l87_8715


namespace sin_alpha_cos_alpha_l87_8746

theorem sin_alpha_cos_alpha (α : ℝ) (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) : 
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

end sin_alpha_cos_alpha_l87_8746


namespace egg_laying_hens_l87_8726

theorem egg_laying_hens (total_chickens : ℕ) (num_roosters : ℕ) (non_egg_laying_hens : ℕ)
  (h1 : total_chickens = 325)
  (h2 : num_roosters = 28)
  (h3 : non_egg_laying_hens = 20) :
  total_chickens - num_roosters - non_egg_laying_hens = 277 :=
by sorry

end egg_laying_hens_l87_8726


namespace intersection_M_N_l87_8784

def M : Set ℝ := { y | ∃ x, y = 2^x ∧ x > 0 }
def N : Set ℝ := { y | ∃ z, y = Real.log z ∧ z ∈ M }

theorem intersection_M_N : M ∩ N = { y | y > 1 } := sorry

end intersection_M_N_l87_8784


namespace converse_of_statement_l87_8760

variables (a b : ℝ)

theorem converse_of_statement :
  (a + b ≤ 2) → (a ≤ 1 ∨ b ≤ 1) :=
by
  sorry

end converse_of_statement_l87_8760


namespace find_M_l87_8777

theorem find_M (a b c M : ℝ) (h1 : a + b + c = 120) (h2 : a - 9 = M) (h3 : b + 9 = M) (h4 : 9 * c = M) : 
  M = 1080 / 19 :=
by sorry

end find_M_l87_8777


namespace largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative_l87_8734

theorem largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative :
  ∃ (n : ℤ), (4 < n) ∧ (n < 7) ∧ (n = 6) :=
by
  sorry

end largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative_l87_8734


namespace son_age_is_9_l87_8790

-- Definitions for the conditions in the problem
def son_age (S F : ℕ) : Prop := S = (1 / 4 : ℝ) * F - 1
def father_age (S F : ℕ) : Prop := F = 5 * S - 5

-- Main statement of the equivalent problem
theorem son_age_is_9 : ∃ S F : ℕ, son_age S F ∧ father_age S F ∧ S = 9 :=
by
  -- We will leave the proof as an exercise
  sorry

end son_age_is_9_l87_8790


namespace magnitude_of_sum_l87_8764

variables (a b : ℝ × ℝ)
variables (h1 : a.1 * b.1 + a.2 * b.2 = 0)
variables (h2 : a = (4, 3))
variables (h3 : (b.1 ^ 2 + b.2 ^ 2) = 1)

theorem magnitude_of_sum (a b : ℝ × ℝ) (h1 : a.1 * b.1 + a.2 * b.2 = 0) 
  (h2 : a = (4, 3)) (h3 : (b.1 ^ 2 + b.2 ^ 2) = 1) : 
  (a.1 + 2 * b.1) ^ 2 + (a.2 + 2 * b.2) ^ 2 = 29 :=
by sorry

end magnitude_of_sum_l87_8764


namespace oscar_leap_more_than_piper_hop_l87_8763

noncomputable def difference_leap_hop : ℝ :=
let number_of_poles := 51
let total_distance := 7920 -- in feet
let Elmer_strides_per_gap := 44
let Oscar_leaps_per_gap := 15
let Piper_hops_per_gap := 22
let number_of_gaps := number_of_poles - 1
let Elmer_total_strides := Elmer_strides_per_gap * number_of_gaps
let Oscar_total_leaps := Oscar_leaps_per_gap * number_of_gaps
let Piper_total_hops := Piper_hops_per_gap * number_of_gaps
let Elmer_stride_length := total_distance / Elmer_total_strides
let Oscar_leap_length := total_distance / Oscar_total_leaps
let Piper_hop_length := total_distance / Piper_total_hops
Oscar_leap_length - Piper_hop_length

theorem oscar_leap_more_than_piper_hop :
  difference_leap_hop = 3.36 := by
  sorry

end oscar_leap_more_than_piper_hop_l87_8763


namespace product_of_roots_l87_8725

open Real

theorem product_of_roots : (sqrt (Real.exp (1 / 4 * log (16)))) * (sqrt (Real.exp (1 / 6 * log (64)))) = 4 :=
by
  -- sorry is used to bypass the actual proof implementation
  sorry

end product_of_roots_l87_8725


namespace weeds_in_rice_l87_8787

-- Define the conditions
def total_weight_of_rice := 1536
def sample_size := 224
def weeds_in_sample := 28

-- State the main proof
theorem weeds_in_rice (total_rice : ℕ) (sample_size : ℕ) (weeds_sample : ℕ) 
  (H1 : total_rice = total_weight_of_rice) (H2 : sample_size = sample_size) (H3 : weeds_sample = weeds_in_sample) :
  total_rice * weeds_sample / sample_size = 192 := 
by
  -- Evidence of calculations and external assumptions, translated initial assumptions into mathematical format
  sorry

end weeds_in_rice_l87_8787


namespace find_A_l87_8737

theorem find_A (A B : ℕ) (hcfAB lcmAB : ℕ)
  (hcf_cond : Nat.gcd A B = hcfAB)
  (lcm_cond : Nat.lcm A B = lcmAB)
  (B_val : B = 169)
  (hcf_val : hcfAB = 13)
  (lcm_val : lcmAB = 312) :
  A = 24 :=
by 
  sorry

end find_A_l87_8737


namespace tan_add_pi_div_four_sine_cosine_ratio_l87_8789

-- Definition of the tangent function and trigonometric identities
variable {α : ℝ}

-- Given condition: tan(α) = 2
axiom tan_alpha_eq_2 : Real.tan α = 2

-- Problem 1: Prove that tan(α + π/4) = -3
theorem tan_add_pi_div_four : Real.tan ( α + Real.pi / 4 ) = -3 :=
by
  sorry

-- Problem 2: Prove that (6 * sin(α) + cos(α)) / (3 * sin(α) - cos(α)) = 13 / 5
theorem sine_cosine_ratio : 
  ( 6 * Real.sin α + Real.cos α ) / ( 3 * Real.sin α - Real.cos α ) = 13 / 5 :=
by
  sorry

end tan_add_pi_div_four_sine_cosine_ratio_l87_8789


namespace batsman_average_after_12_innings_l87_8792

theorem batsman_average_after_12_innings
  (score_12th: ℕ) (increase_avg: ℕ) (initial_innings: ℕ) (final_innings: ℕ) 
  (initial_avg: ℕ) (final_avg: ℕ) :
  score_12th = 48 ∧ increase_avg = 2 ∧ initial_innings = 11 ∧ final_innings = 12 ∧
  final_avg = initial_avg + increase_avg ∧
  12 * final_avg = initial_innings * initial_avg + score_12th →
  final_avg = 26 :=
by 
  sorry

end batsman_average_after_12_innings_l87_8792


namespace jogger_ahead_distance_l87_8768

/-- The jogger is running at a constant speed of 9 km/hr, the train at a speed of 45 km/hr,
    it is 210 meters long and passes the jogger in 41 seconds.
    Prove the jogger is 200 meters ahead of the train. -/
theorem jogger_ahead_distance 
  (v_j : ℝ) (v_t : ℝ) (L : ℝ) (t : ℝ) (d : ℝ) 
  (hv_j : v_j = 9) (hv_t : v_t = 45) (hL : L = 210) (ht : t = 41) :
  d = 200 :=
by {
  -- The conditions and the final proof step, 
  -- actual mathematical proofs steps are not necessary according to the problem statement.
  sorry
}

end jogger_ahead_distance_l87_8768


namespace initial_velocity_l87_8721

noncomputable def displacement (t : ℝ) : ℝ := 3 * t - t^2

theorem initial_velocity :
  (deriv displacement 0) = 3 :=
by
  sorry

end initial_velocity_l87_8721


namespace infinitely_many_digitally_divisible_integers_l87_8780

theorem infinitely_many_digitally_divisible_integers :
  ∀ n : ℕ, ∃ k : ℕ, k = (10 ^ (3 ^ n) - 1) / 9 ∧ (3 ^ n ∣ k) :=
by
  sorry

end infinitely_many_digitally_divisible_integers_l87_8780


namespace data_transmission_time_l87_8774

def packet_size : ℕ := 256
def num_packets : ℕ := 100
def transmission_rate : ℕ := 200
def total_data : ℕ := num_packets * packet_size
def transmission_time_in_seconds : ℚ := total_data / transmission_rate
def transmission_time_in_minutes : ℚ := transmission_time_in_seconds / 60

theorem data_transmission_time :
  transmission_time_in_minutes = 2 :=
  sorry

end data_transmission_time_l87_8774


namespace original_recipe_pasta_l87_8797

noncomputable def pasta_per_person (total_pasta : ℕ) (total_people : ℕ) : ℚ :=
  total_pasta / total_people

noncomputable def original_pasta (pasta_per_person : ℚ) (people_served : ℕ) : ℚ :=
  pasta_per_person * people_served

theorem original_recipe_pasta (total_pasta : ℕ) (total_people : ℕ) (people_served : ℕ) (required_pasta : ℚ) :
  total_pasta = 10 → total_people = 35 → people_served = 7 → required_pasta = 2 →
  pasta_per_person total_pasta total_people * people_served = required_pasta :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end original_recipe_pasta_l87_8797


namespace larger_number_is_1634_l87_8750

theorem larger_number_is_1634 (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 20) : L = 1634 := 
sorry

end larger_number_is_1634_l87_8750


namespace trigonometric_expression_evaluation_l87_8762

theorem trigonometric_expression_evaluation
  (α : ℝ)
  (h1 : Real.tan α = -3 / 4) :
  (3 * Real.sin (α / 2) ^ 2 + 
   2 * Real.sin (α / 2) * Real.cos (α / 2) + 
   Real.cos (α / 2) ^ 2 - 2) / 
  (Real.sin (π / 2 + α) * Real.tan (-3 * π + α) + 
   Real.cos (6 * π - α)) = -7 := 
by 
  sorry
  -- This will skip the proof and ensure the Lean code can be built successfully.

end trigonometric_expression_evaluation_l87_8762


namespace solve_inequality_l87_8744

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  3 - 1 / (3 * x + 4) < 5 ↔ -3 / 2 < x :=
by
  sorry

end solve_inequality_l87_8744


namespace number_of_whole_numbers_in_intervals_l87_8757

theorem number_of_whole_numbers_in_intervals : 
  let interval_start := (5 / 3 : ℝ)
  let interval_end := 2 * Real.pi
  ∃ n : ℕ, interval_start < ↑n ∧ ↑n < interval_end ∧ (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) ∧ 
  (∀ m : ℕ, interval_start < ↑m ∧ ↑m < interval_end → (m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6)) :=
sorry

end number_of_whole_numbers_in_intervals_l87_8757


namespace bug_probability_nine_moves_l87_8791

noncomputable def bug_cube_probability (moves : ℕ) : ℚ := sorry

/-- 
The probability that after exactly 9 moves, a bug starting at one vertex of a cube 
and moving randomly along the edges will have visited every vertex exactly once and 
revisited one vertex once more. 
-/
theorem bug_probability_nine_moves : bug_cube_probability 9 = 16 / 6561 := by
  sorry

end bug_probability_nine_moves_l87_8791


namespace roman_numeral_calculation_l87_8782

def I : ℕ := 1
def V : ℕ := 5
def X : ℕ := 10
def L : ℕ := 50
def C : ℕ := 100
def D : ℕ := 500
def M : ℕ := 1000

theorem roman_numeral_calculation : 2 * M + 5 * L + 7 * X + 9 * I = 2329 := by
  sorry

end roman_numeral_calculation_l87_8782


namespace sin_double_angle_ratio_l87_8714

theorem sin_double_angle_ratio (α : ℝ) (h : Real.sin α = 3 * Real.cos α) : 
  Real.sin (2 * α) / (Real.cos α)^2 = 6 :=
by 
  sorry

end sin_double_angle_ratio_l87_8714


namespace n_squared_plus_d_not_perfect_square_l87_8766

theorem n_squared_plus_d_not_perfect_square (n d : ℕ) (h1 : n > 0)
  (h2 : d > 0) (h3 : d ∣ 2 * n^2) : ¬ ∃ x : ℕ, n^2 + d = x^2 := 
sorry

end n_squared_plus_d_not_perfect_square_l87_8766


namespace total_journey_time_l87_8793

theorem total_journey_time
  (river_speed : ℝ)
  (boat_speed_still_water : ℝ)
  (distance_upstream : ℝ)
  (total_journey_time : ℝ) :
  river_speed = 2 → 
  boat_speed_still_water = 6 → 
  distance_upstream = 48 → 
  total_journey_time = (distance_upstream / (boat_speed_still_water - river_speed) + distance_upstream / (boat_speed_still_water + river_speed)) → 
  total_journey_time = 18 := 
by
  intros h1 h2 h3 h4
  sorry

end total_journey_time_l87_8793


namespace subtract_value_is_34_l87_8705

theorem subtract_value_is_34 
    (x y : ℤ) 
    (h1 : (x - 5) / 7 = 7) 
    (h2 : (x - y) / 10 = 2) : 
    y = 34 := 
sorry

end subtract_value_is_34_l87_8705
