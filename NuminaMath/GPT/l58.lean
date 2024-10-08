import Mathlib

namespace tamara_total_earnings_l58_58630

-- Definitions derived from the conditions in the problem statement.
def pans : ℕ := 2
def pieces_per_pan : ℕ := 8
def price_per_piece : ℕ := 2

-- Theorem stating the required proof problem.
theorem tamara_total_earnings : 
  (pans * pieces_per_pan * price_per_piece) = 32 :=
by
  sorry

end tamara_total_earnings_l58_58630


namespace product_of_four_integers_l58_58656

theorem product_of_four_integers (A B C D : ℕ) (h_pos_A : 0 < A) (h_pos_B : 0 < B) (h_pos_C : 0 < C) (h_pos_D : 0 < D)
  (h_sum : A + B + C + D = 36)
  (h_eq1 : A + 2 = B - 2)
  (h_eq2 : B - 2 = C * 2)
  (h_eq3 : C * 2 = D / 2) :
  A * B * C * D = 3840 :=
by
  sorry

end product_of_four_integers_l58_58656


namespace move_line_up_l58_58227

/-- Define the original line equation as y = 3x - 2 -/
def original_line (x : ℝ) : ℝ := 3 * x - 2

/-- Define the resulting line equation as y = 3x + 4 -/
def resulting_line (x : ℝ) : ℝ := 3 * x + 4

theorem move_line_up (x : ℝ) : resulting_line x = original_line x + 6 :=
by
  sorry

end move_line_up_l58_58227


namespace maximum_profit_l58_58084

noncomputable def profit (x : ℝ) : ℝ :=
  5.06 * x - 0.15 * x^2 + 2 * (15 - x)

theorem maximum_profit : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 15 ∧ profit x = 45.6 :=
by
  sorry

end maximum_profit_l58_58084


namespace indeterminate_C_l58_58009

variable (m n C : ℝ)

theorem indeterminate_C (h1 : m = 8 * n + C)
                      (h2 : m + 2 = 8 * (n + 0.25) + C) : 
                      False :=
by
  sorry

end indeterminate_C_l58_58009


namespace largest_number_is_A_l58_58572

-- Definitions of the numbers
def numA := 8.45678
def numB := 8.456777777 -- This should be represented properly with an infinite sequence in a real formal proof
def numC := 8.456767676 -- This should be represented properly with an infinite sequence in a real formal proof
def numD := 8.456756756 -- This should be represented properly with an infinite sequence in a real formal proof
def numE := 8.456745674 -- This should be represented properly with an infinite sequence in a real formal proof

-- Lean statement to prove that numA is the largest number
theorem largest_number_is_A : numA > numB ∧ numA > numC ∧ numA > numD ∧ numA > numE :=
by
  -- Proof not provided, sorry to skip
  sorry

end largest_number_is_A_l58_58572


namespace number_of_solutions_l58_58442

theorem number_of_solutions : 
  ∃ n : ℕ, n = 5 ∧ (∃ (x y : ℕ), 1 ≤ x ∧ 1 ≤ y ∧ 4 * x + 5 * y = 98) :=
sorry

end number_of_solutions_l58_58442


namespace find_x_plus_y_l58_58494

-- Define the initial assumptions and conditions
variables {x y : ℝ}
axiom geom_sequence : 1 > 0 ∧ x > 0 ∧ y > 0 ∧ 3 > 0 ∧ 1 * x = y
axiom arith_sequence : 2 * y = x + 3

-- Prove that x + y = 15 / 4
theorem find_x_plus_y : x + y = 15 / 4 := sorry

end find_x_plus_y_l58_58494


namespace probability_of_pink_flower_is_five_over_nine_l58_58091

-- Definitions as per the conditions
def flowersInBagA := 9
def pinkFlowersInBagA := 3
def flowersInBagB := 9
def pinkFlowersInBagB := 7
def probChoosingBag := (1:ℚ) / 2

-- Definition of the probabilities
def probPinkFromA := pinkFlowersInBagA / flowersInBagA
def probPinkFromB := pinkFlowersInBagB / flowersInBagB

-- Total probability calculation using the law of total probability
def probPink := probPinkFromA * probChoosingBag + probPinkFromB * probChoosingBag

-- Statement to be proved
theorem probability_of_pink_flower_is_five_over_nine : probPink = (5:ℚ) / 9 := 
by
  sorry

end probability_of_pink_flower_is_five_over_nine_l58_58091


namespace find_f_neg1_l58_58751

theorem find_f_neg1 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (x - 1) = x^2 + 1) : f (-1) = 1 := 
by 
  -- skipping the proof: 
  sorry

end find_f_neg1_l58_58751


namespace min_y_in_quadratic_l58_58134

theorem min_y_in_quadratic (x : ℝ) : ∃ y : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ y', (y' = x^2 + 16 * x + 20) → y ≤ y' := 
sorry

end min_y_in_quadratic_l58_58134


namespace sequence_a1_l58_58525

variable (S : ℕ → ℤ) (a : ℕ → ℤ)

def Sn_formula (n : ℕ) (a₁ : ℤ) : ℤ := (a₁ * (4^n - 1)) / 3

theorem sequence_a1 (h1 : ∀ n : ℕ, S n = Sn_formula n (a 1))
                    (h2 : a 4 = 32) :
  a 1 = 1 / 2 :=
by
  sorry

end sequence_a1_l58_58525


namespace goldfish_feeding_l58_58347

theorem goldfish_feeding (g : ℕ) (h : g = 8) : 4 * g = 32 :=
by
  sorry

end goldfish_feeding_l58_58347


namespace find_m_l58_58001

variable (m : ℝ)
def vector_a : ℝ × ℝ := (1, 3)
def vector_b : ℝ × ℝ := (m, -2)

theorem find_m (h : (1 + m) + 3 = 0) : m = -4 := by
  sorry

end find_m_l58_58001


namespace scientific_notation_of_116_million_l58_58565

theorem scientific_notation_of_116_million : 116000000 = 1.16 * 10^7 :=
sorry

end scientific_notation_of_116_million_l58_58565


namespace parabola_axis_of_symmetry_l58_58945

theorem parabola_axis_of_symmetry (p : ℝ) :
  (∀ x : ℝ, x = 3 → -x^2 - p*x + 2 = -x^2 - (-6)*x + 2) → p = -6 :=
by sorry

end parabola_axis_of_symmetry_l58_58945


namespace sqrt_cubic_sqrt_decimal_l58_58277

theorem sqrt_cubic_sqrt_decimal : 
  (Real.sqrt (0.0036 : ℝ))^(1/3) = 0.3912 :=
sorry

end sqrt_cubic_sqrt_decimal_l58_58277


namespace inequality_for_average_daily_work_l58_58877

-- Given
def total_earthwork : ℕ := 300
def completed_earthwork_first_day : ℕ := 60
def scheduled_days : ℕ := 6
def days_ahead : ℕ := 2

-- To Prove
theorem inequality_for_average_daily_work (x : ℕ) :
  scheduled_days - days_ahead - 1 > 0 →
  (total_earthwork - completed_earthwork_first_day) ≤ x * (scheduled_days - days_ahead - 1) :=
by
  sorry

end inequality_for_average_daily_work_l58_58877


namespace find_f6_l58_58305

variable {R : Type*} [AddGroup R] [Semiring R]

def functional_equation (f : R → R) :=
∀ x y : R, f (x + y) = f x + f y

theorem find_f6 (f : ℝ → ℝ) (h1 : functional_equation f) (h2 : f 4 = 10) : f 6 = 10 :=
sorry

end find_f6_l58_58305


namespace inequality_proof_l58_58942

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (h : a + b + c = 1)

theorem inequality_proof :
  (1 + a) / (1 - a) + (1 + b) / (1 - a) + (1 + c) / (1 - c) ≤ 2 * ((b / a) + (c / b) + (a / c)) :=
by sorry

end inequality_proof_l58_58942


namespace b_contribution_is_correct_l58_58486

-- Definitions based on the conditions
def A_investment : ℕ := 35000
def B_join_after_months : ℕ := 5
def profit_ratio_A_B : ℕ := 2
def profit_ratio_B_A : ℕ := 3
def A_total_months : ℕ := 12
def B_total_months : ℕ := 7
def profit_ratio := (profit_ratio_A_B, profit_ratio_B_A)
def total_investment_time_ratio : ℕ := 12 * 35000 / 7

-- The property to be proven
theorem b_contribution_is_correct (X : ℕ) (h : 35000 * 12 / (X * 7) = 2 / 3) : X = 90000 :=
by
  sorry

end b_contribution_is_correct_l58_58486


namespace black_to_white_area_ratio_l58_58788

noncomputable def radius1 : ℝ := 2
noncomputable def radius2 : ℝ := 4
noncomputable def radius3 : ℝ := 6
noncomputable def radius4 : ℝ := 8
noncomputable def radius5 : ℝ := 10

noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def black_area : ℝ :=
  area radius1 + (area radius3 - area radius2) + (area radius5 - area radius4)

noncomputable def white_area : ℝ :=
  (area radius2 - area radius1) + (area radius4 - area radius3)

theorem black_to_white_area_ratio :
  black_area / white_area = 3 / 2 := by
  sorry

end black_to_white_area_ratio_l58_58788


namespace proposition_3_correct_l58_58749

open Real

def is_obtuse (A B C : ℝ) : Prop :=
  A + B + C = π ∧ (A > π / 2 ∨ B > π / 2 ∨ C > π / 2)

theorem proposition_3_correct (A B C : ℝ) (h₀ : 0 < A) (h₁ : 0 < B) (h₂ : 0 < C) (h₃ : A + B + C = π)
  (h : sin A ^ 2 + sin B ^ 2 + cos C ^ 2 < 1) : is_obtuse A B C :=
by
  sorry

end proposition_3_correct_l58_58749


namespace negation_of_p_l58_58695

def f (a x : ℝ) : ℝ := a * x - x - a

theorem negation_of_p :
  (¬ ∀ a > 0, a ≠ 1 → ∃ x : ℝ, f a x = 0) ↔ (∃ a > 0, a ≠ 1 ∧ ¬ ∃ x : ℝ, f a x = 0) :=
by {
  sorry
}

end negation_of_p_l58_58695


namespace part_a_l58_58284

theorem part_a (b c: ℤ) : ∃ (n : ℕ) (a : ℕ → ℤ), 
  (a 0 = b) ∧ (a n = c) ∧ (∀ i, 1 ≤ i ∧ i ≤ n → |a i - a (i - 1)| = i^2) :=
sorry

end part_a_l58_58284


namespace complementary_angles_positive_difference_l58_58553

/-- Two angles are complementary if their sum is 90 degrees.
    If the measures of these angles are in the ratio 3:1,
    then their positive difference is 45 degrees. -/
theorem complementary_angles_positive_difference (x : ℝ) (h1 : (3 * x) + x = 90) :
  abs ((3 * x) - x) = 45 :=
by
  sorry

end complementary_angles_positive_difference_l58_58553


namespace find_b_in_geometric_sequence_l58_58496

theorem find_b_in_geometric_sequence (a_1 : ℤ) :
  ∀ (n : ℕ), ∃ (b : ℤ), (3^n - b = (a_1 * (3^n - 1)) / 2) :=
by
  sorry

example (a_1 : ℤ) :
  ∃ (b : ℤ), ∀ (n : ℕ), 3^n - b = (a_1 * (3^n - 1)) / 2 :=
by
  use 1
  sorry

end find_b_in_geometric_sequence_l58_58496


namespace scott_runs_84_miles_in_a_month_l58_58575

-- Define the number of miles Scott runs from Monday to Wednesday in a week.
def milesMonToWed : ℕ := 3 * 3

-- Define the number of miles Scott runs on Thursday and Friday in a week.
def milesThuFri : ℕ := 3 * 2 * 2

-- Define the total number of miles Scott runs in a week.
def totalMilesPerWeek : ℕ := milesMonToWed + milesThuFri

-- Define the number of weeks in a month.
def weeksInMonth : ℕ := 4

-- Define the total number of miles Scott runs in a month.
def totalMilesInMonth : ℕ := totalMilesPerWeek * weeksInMonth

-- Statement to prove that Scott runs 84 miles in a month with 4 weeks.
theorem scott_runs_84_miles_in_a_month : totalMilesInMonth = 84 := by
  -- The proof is omitted for this example.
  sorry

end scott_runs_84_miles_in_a_month_l58_58575


namespace cardinality_union_l58_58171

open Finset

theorem cardinality_union (A B : Finset ℕ) (h : 2 ^ A.card + 2 ^ B.card - 2 ^ (A ∩ B).card = 144) : (A ∪ B).card = 8 := 
by 
  sorry

end cardinality_union_l58_58171


namespace minimum_value_of_sum_of_squares_l58_58010

variable {x y : ℝ}

theorem minimum_value_of_sum_of_squares (h : x^2 + 2*x*y - y^2 = 7) : 
  x^2 + y^2 ≥ 7 * Real.sqrt 2 / 2 := by 
    sorry

end minimum_value_of_sum_of_squares_l58_58010


namespace find_ellipse_eq_product_of_tangent_slopes_l58_58715

variables {a b : ℝ} {x y x0 y0 : ℝ}

-- Given conditions
def ellipse (a b : ℝ) := a > 0 ∧ b > 0 ∧ a > b ∧ (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → y = 1 ∧ y = 3 / 2)

def eccentricity (a b : ℝ) := b = (1 / 2) * a

def passes_through (x y : ℝ) := x = 1 ∧ y = 3 / 2

-- Part 1: Prove the equation of the ellipse
theorem find_ellipse_eq (a b : ℝ) (h_ellipse : ellipse a b) (h_eccentricity : eccentricity a b) (h_point : passes_through 1 (3/2)) :
    (x^2) / 4 + (y^2) / 3 = 1 :=
sorry

-- Circle equation definition
def circle (x y : ℝ) := x^2 + y^2 = 7

-- Part 2: Prove the product of the slopes of the tangent lines is constant
theorem product_of_tangent_slopes (P : ℝ × ℝ) (h_circle : circle P.1 P.2) : 
    ∀ k1 k2 : ℝ, (4 - P.1^2) * k1^2 + 6 * P.1 * P.2 * k1 + 3 - P.2^2 = 0 → 
    (4 - P.1^2) * k2^2 + 6 * P.1 * P.2 * k2 + 3 - P.2^2 = 0 → k1 * k2 = -1 :=
sorry

end find_ellipse_eq_product_of_tangent_slopes_l58_58715


namespace mn_min_l58_58086

noncomputable def min_mn_value (m n : ℝ) : ℝ := m * n

theorem mn_min : 
  (∃ m n, m = Real.sin (2 * (π / 12)) ∧ n > 0 ∧ 
            Real.cos (2 * (π / 12 + n) - π / 4) = m ∧ 
            min_mn_value m n = π * 5 / 48) := by
  sorry

end mn_min_l58_58086


namespace average_marks_correct_l58_58551

-- Definitions used in the Lean 4 statement, reflecting conditions in the problem
def total_students_class1 : ℕ := 25 
def average_marks_class1 : ℕ := 40 
def total_students_class2 : ℕ := 30 
def average_marks_class2 : ℕ := 60 

-- Calculate the total marks for both classes
def total_marks_class1 : ℕ := total_students_class1 * average_marks_class1 
def total_marks_class2 : ℕ := total_students_class2 * average_marks_class2 
def total_marks : ℕ := total_marks_class1 + total_marks_class2 

-- Calculate the total number of students
def total_students : ℕ := total_students_class1 + total_students_class2 

-- Define the average of all students
noncomputable def average_marks_all_students : ℚ := total_marks / total_students 

-- The theorem to be proved
theorem average_marks_correct : average_marks_all_students = (2800 : ℚ) / 55 := 
by 
  sorry

end average_marks_correct_l58_58551


namespace toms_expense_l58_58737

def cost_per_square_foot : ℝ := 5
def square_feet_per_seat : ℝ := 12
def number_of_seats : ℝ := 500
def partner_coverage : ℝ := 0.40

def total_square_feet : ℝ := square_feet_per_seat * number_of_seats
def land_cost : ℝ := cost_per_square_foot * total_square_feet
def construction_cost : ℝ := 2 * land_cost
def total_cost : ℝ := land_cost + construction_cost
def tom_coverage_percentage : ℝ := 1 - partner_coverage
def toms_share : ℝ := tom_coverage_percentage * total_cost

theorem toms_expense :
  toms_share = 54000 :=
by
  sorry

end toms_expense_l58_58737


namespace remaining_amount_is_9_l58_58511

-- Define the original prices of the books
def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00

-- Define the discount rate for the first two books
def discount_rate : ℝ := 0.25

-- Define the total cost without discount
def total_cost_without_discount := book1_price + book2_price + book3_price + book4_price

-- Calculate the discounts for the first two books
def book1_discount := book1_price * discount_rate
def book2_discount := book2_price * discount_rate

-- Calculate the discounted prices for the first two books
def discounted_book1_price := book1_price - book1_discount
def discounted_book2_price := book2_price - book2_discount

-- Calculate the total cost of the books with discounts applied
def total_cost_with_discount := discounted_book1_price + discounted_book2_price + book3_price + book4_price

-- Define the free shipping threshold
def free_shipping_threshold : ℝ := 50.00

-- Calculate the remaining amount Connor needs to spend
def remaining_amount_to_spend := free_shipping_threshold - total_cost_with_discount

-- State the theorem
theorem remaining_amount_is_9 : remaining_amount_to_spend = 9.00 := by
  -- we would provide the proof here
  sorry

end remaining_amount_is_9_l58_58511


namespace complement_intersection_l58_58541

open Set

-- Define the universal set I, and sets M and N
def I : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {3, 4, 5}

-- Lean statement to prove the desired result
theorem complement_intersection : (I \ N) ∩ M = {1, 2} := by
  sorry

end complement_intersection_l58_58541


namespace find_m_of_pure_imaginary_l58_58187

theorem find_m_of_pure_imaginary (m : ℝ) (h1 : (m^2 + m - 2) = 0) (h2 : (m^2 - 1) ≠ 0) : m = -2 :=
by
  sorry

end find_m_of_pure_imaginary_l58_58187


namespace smallest_base_converted_l58_58106

def convert_to_decimal_base_3 (n : ℕ) : ℕ :=
  1 * 3^3 + 0 * 3^2 + 0 * 3^1 + 2 * 3^0

def convert_to_decimal_base_6 (n : ℕ) : ℕ :=
  2 * 6^2 + 1 * 6^1 + 0 * 6^0

def convert_to_decimal_base_4 (n : ℕ) : ℕ :=
  1 * 4^3 + 0 * 4^2 + 0 * 4^1 + 0 * 4^0

def convert_to_decimal_base_2 (n : ℕ) : ℕ :=
  1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem smallest_base_converted :
  min (convert_to_decimal_base_3 1002) 
      (min (convert_to_decimal_base_6 210) 
           (min (convert_to_decimal_base_4 1000) 
                (convert_to_decimal_base_2 111111))) = convert_to_decimal_base_3 1002 :=
by sorry

end smallest_base_converted_l58_58106


namespace thomas_savings_years_l58_58328

def weekly_allowance : ℕ := 50
def weekly_coffee_shop_earning : ℕ := 9 * 30
def weekly_spending : ℕ := 35
def car_cost : ℕ := 15000
def additional_amount_needed : ℕ := 2000
def weeks_in_a_year : ℕ := 52

def first_year_savings : ℕ := weeks_in_a_year * (weekly_allowance - weekly_spending)
def second_year_savings : ℕ := weeks_in_a_year * (weekly_coffee_shop_earning - weekly_spending)

noncomputable def total_savings_needed : ℕ := car_cost - additional_amount_needed

theorem thomas_savings_years : 
  first_year_savings + second_year_savings = total_savings_needed → 2 = 2 :=
by
  sorry

end thomas_savings_years_l58_58328


namespace find_vector_c_l58_58020

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (1, 2)
def c : ℝ × ℝ := (2, 1)

def perp (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, w = (k * v.1, k * v.2)

theorem find_vector_c : 
  perp (c.1 + b.1, c.2 + b.2) a ∧ parallel (c.1 - a.1, c.2 + a.2) b :=
by 
  sorry

end find_vector_c_l58_58020


namespace ThreeDigitEvenNumbersCount_l58_58435

theorem ThreeDigitEvenNumbersCount : 
  let a := 100
  let max := 998
  let d := 2
  let n := (max - a) / d + 1
  100 < 999 ∧ 100 % 2 = 0 ∧ max % 2 = 0 
  → d > 0 
  → n = 450 :=
by
  sorry

end ThreeDigitEvenNumbersCount_l58_58435


namespace octahedron_cut_area_l58_58028

theorem octahedron_cut_area:
  let a := 9
  let b := 3
  let c := 8
  a + b + c = 20 :=
by
  sorry

end octahedron_cut_area_l58_58028


namespace parabola_directrix_l58_58095

theorem parabola_directrix (x y : ℝ) (h_parabola : x^2 = (1/2) * y) : y = - (1/8) :=
sorry

end parabola_directrix_l58_58095


namespace pauls_score_is_91_l58_58925

theorem pauls_score_is_91 (q s c w : ℕ) 
  (h1 : q = 35)
  (h2 : s = 35 + 5 * c - 2 * w)
  (h3 : s > 90)
  (h4 : c + w ≤ 35)
  (h5 : ∀ s', 90 < s' ∧ s' < s → ¬ (∃ c' w', s' = 35 + 5 * c' - 2 * w' ∧ c' + w' ≤ 35 ∧ c' ≠ c)) : 
  s = 91 := 
sorry

end pauls_score_is_91_l58_58925


namespace friend_c_spent_26_l58_58286

theorem friend_c_spent_26 :
  let you_spent := 12
  let friend_a_spent := you_spent + 4
  let friend_b_spent := friend_a_spent - 3
  let friend_c_spent := friend_b_spent * 2
  friend_c_spent = 26 :=
by
  sorry

end friend_c_spent_26_l58_58286


namespace min_value_l58_58359

theorem min_value (x : ℝ) (h : x > 1) : (x + 4 / (x - 1)) ≥ 5 :=
by sorry

end min_value_l58_58359


namespace sum_of_intercepts_l58_58090

theorem sum_of_intercepts (x y : ℝ) (h : 3 * x - 4 * y - 12 = 0) :
    (y = -3 ∧ x = 4) → x + y = 1 :=
by
  intro h'
  obtain ⟨hy, hx⟩ := h'
  rw [hy, hx]
  norm_num
  done

end sum_of_intercepts_l58_58090


namespace sum_x_y_eq_two_l58_58099

theorem sum_x_y_eq_two (x y : ℝ) 
  (h1 : (x-1)^3 + 2003*(x-1) = -1) 
  (h2 : (y-1)^3 + 2003*(y-1) = 1) : 
  x + y = 2 :=
by
  sorry

end sum_x_y_eq_two_l58_58099


namespace hash_op_is_100_l58_58760

def hash_op (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

theorem hash_op_is_100 (a b : ℕ) (h1 : a + b = 5) : hash_op a b = 100 :=
sorry

end hash_op_is_100_l58_58760


namespace recruits_total_l58_58145

theorem recruits_total (P N D : ℕ) (total_recruits : ℕ) 
  (h1 : P = 50) 
  (h2 : N = 100) 
  (h3 : D = 170)
  (h4 : (∃ x y, (x = 50) ∧ (y = 100) ∧ (x = 4 * y))
        ∨ (∃ x z, (x = 50) ∧ (z = 170) ∧ (x = 4 * z))
        ∨ (∃ y z, (y = 100) ∧ (z = 170) ∧ (y = 4 * z))) : 
  total_recruits = 211 :=
by
  sorry

end recruits_total_l58_58145


namespace find_last_number_l58_58807

theorem find_last_number (A B C D E F G : ℝ)
    (h1 : (A + B + C + D) / 4 = 13)
    (h2 : (D + E + F + G) / 4 = 15)
    (h3 : E + F + G = 55)
    (h4 : D^2 = G) :
  G = 25 := by 
  sorry

end find_last_number_l58_58807


namespace find_min_difference_l58_58787

theorem find_min_difference (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h₁ : 3 * q < 5 * p)
  (h₂ : 8 * p < 5 * q)
  (h₃ : ∀ r s : ℤ, 0 < s → (3 * s < 5 * r ∧ 8 * r < 5 * s) → q ≤ s) :
  q - p = 5 :=
sorry

end find_min_difference_l58_58787


namespace greatest_drop_in_price_is_august_l58_58550

-- Define the months and their respective price changes
def price_changes : List (String × ℝ) :=
  [("January", -1.00), ("February", 1.50), ("March", -3.00), ("April", 2.50), 
   ("May", -0.75), ("June", -2.25), ("July", 1.00), ("August", -4.00)]

-- Define the statement that August has the greatest drop in price
theorem greatest_drop_in_price_is_august :
  ∀ month ∈ price_changes, month.snd ≤ -4.00 → month.fst = "August" :=
by
  sorry

end greatest_drop_in_price_is_august_l58_58550


namespace period_of_sin3x_plus_cos3x_l58_58158

noncomputable def period_of_trig_sum (x : ℝ) : ℝ := 
  let y := (fun x => Real.sin (3 * x) + Real.cos (3 * x))
  (2 * Real.pi) / 3

theorem period_of_sin3x_plus_cos3x : (fun x => Real.sin (3 * x) + Real.cos (3 * x)) =
  (fun x => Real.sin (3 * (x + period_of_trig_sum x)) + Real.cos (3 * (x + period_of_trig_sum x))) :=
by
  sorry

end period_of_sin3x_plus_cos3x_l58_58158


namespace Irene_age_is_46_l58_58970

-- Definitions based on the given conditions
def Eddie_age : ℕ := 92
def Becky_age : ℕ := Eddie_age / 4
def Irene_age : ℕ := 2 * Becky_age

-- Theorem we aim to prove that Irene's age is 46
theorem Irene_age_is_46 : Irene_age = 46 := by
  sorry

end Irene_age_is_46_l58_58970


namespace trash_picked_outside_l58_58049

theorem trash_picked_outside (T_tot : ℕ) (C1 C2 C3 C4 C5 C6 C7 C8 : ℕ)
  (hT_tot : T_tot = 1576)
  (hC1 : C1 = 124) (hC2 : C2 = 98) (hC3 : C3 = 176) (hC4 : C4 = 212)
  (hC5 : C5 = 89) (hC6 : C6 = 241) (hC7 : C7 = 121) (hC8 : C8 = 102) :
  T_tot - (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8) = 413 :=
by sorry

end trash_picked_outside_l58_58049


namespace trig_expression_evaluation_l58_58434

theorem trig_expression_evaluation
  (α : ℝ)
  (h : Real.tan α = 2) :
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := 
by 
  sorry

end trig_expression_evaluation_l58_58434


namespace peasant_initial_money_l58_58943

theorem peasant_initial_money :
  ∃ (x1 x2 x3 : ℕ), 
    (x1 / 2 + 1 = x2) ∧ 
    (x2 / 2 + 2 = x3) ∧ 
    (x3 / 2 + 1 = 0) ∧ 
    x1 = 18 := 
by
  sorry

end peasant_initial_money_l58_58943


namespace b_range_given_conditions_l58_58237

theorem b_range_given_conditions 
    (b c : ℝ)
    (roots_in_interval : ∀ x, x^2 + b * x + c = 0 → -1 ≤ x ∧ x ≤ 1)
    (ineq : 0 ≤ 3 * b + c ∧ 3 * b + c ≤ 3) :
    0 ≤ b ∧ b ≤ 2 :=
sorry

end b_range_given_conditions_l58_58237


namespace general_term_arithmetic_sequence_l58_58470

theorem general_term_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (a1 : a 1 = -1) 
  (d : ℤ) 
  (h : d = 4) : 
  ∀ n : ℕ, a n = 4 * n - 5 :=
by
  sorry

end general_term_arithmetic_sequence_l58_58470


namespace swimming_pool_width_l58_58617

theorem swimming_pool_width 
  (V : ℝ) (L : ℝ) (B1 : ℝ) (B2 : ℝ) (h : ℝ)
  (h_volume : V = (h / 2) * (B1 + B2) * L) 
  (h_V : V = 270) 
  (h_L : L = 12) 
  (h_B1 : B1 = 1) 
  (h_B2 : B2 = 4) : 
  h = 9 :=
  sorry

end swimming_pool_width_l58_58617


namespace probability_white_ball_second_draw_l58_58303

noncomputable def probability_white_given_red (red_white_yellow_balls : Nat × Nat × Nat) : ℚ :=
  let (r, w, y) := red_white_yellow_balls
  let total_balls := r + w + y
  let p_A := (r : ℚ) / total_balls
  let p_AB := (r : ℚ) / total_balls * (w : ℚ) / (total_balls - 1)
  p_AB / p_A

theorem probability_white_ball_second_draw (r w y : Nat) (h_r : r = 2) (h_w : w = 3) (h_y : y = 1) :
  probability_white_given_red (r, w, y) = 3 / 5 :=
by
  rw [h_r, h_w, h_y]
  unfold probability_white_given_red
  simp
  sorry

end probability_white_ball_second_draw_l58_58303


namespace zongzi_packing_l58_58147

theorem zongzi_packing (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (8 * x + 10 * y = 200) ↔ (x, y) = (5, 16) ∨ (x, y) = (10, 12) ∨ (x, y) = (15, 8) ∨ (x, y) = (20, 4) := 
sorry

end zongzi_packing_l58_58147


namespace sequence_term_is_square_l58_58862

noncomputable def sequence_term (n : ℕ) : ℕ :=
  let part1 := (10 ^ (n + 1) - 1) / 9
  let part2 := (10 ^ (2 * n + 2) - 10 ^ (n + 1)) / 9
  1 + 4 * part1 + 4 * part2

theorem sequence_term_is_square (n : ℕ) : ∃ k : ℕ, k^2 = sequence_term n :=
by
  sorry

end sequence_term_is_square_l58_58862


namespace roots_of_quadratic_eq_l58_58419

theorem roots_of_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_quadratic_eq_l58_58419


namespace maximize_volume_l58_58421

-- Define the given dimensions
def length := 90
def width := 48

-- Define the volume function based on the height h
def volume (h : ℝ) : ℝ := h * (length - 2 * h) * (width - 2 * h)

-- Define the height that maximizes the volume
def optimal_height := 10

-- Define the maximum volume obtained at the optimal height
def max_volume := 19600

-- State the proof problem
theorem maximize_volume : 
  (∃ h : ℝ, volume h ≤ volume optimal_height) ∧
  volume optimal_height = max_volume := 
by
  sorry

end maximize_volume_l58_58421


namespace original_area_of_triangle_l58_58240

theorem original_area_of_triangle (A : ℝ) (h1 : 4 * A * 16 = 64) : A = 4 :=
by
  sorry

end original_area_of_triangle_l58_58240


namespace quadratic_roots_condition_l58_58561

theorem quadratic_roots_condition (k : ℝ) : 
  (∀ (r s : ℝ), r + s = -k ∧ r * s = 12 → (r + 3) + (s + 3) = k) → k = 3 := 
by 
  sorry

end quadratic_roots_condition_l58_58561


namespace max_3cosx_4sinx_l58_58073

theorem max_3cosx_4sinx (x : ℝ) : (3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧ (∃ y : ℝ, 3 * Real.cos y + 4 * Real.sin y = 5) :=
  sorry

end max_3cosx_4sinx_l58_58073


namespace age_ratio_in_2_years_is_2_1_l58_58364

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

end age_ratio_in_2_years_is_2_1_l58_58364


namespace isosceles_triangle_base_length_l58_58466

-- Define the isosceles triangle problem
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  perimeter : ℝ
  isIsosceles : (side1 = side2 ∨ side1 = base ∨ side2 = base)
  sideLengthCondition : (side1 = 3 ∨ side2 = 3 ∨ base = 3)
  perimeterCondition : side1 + side2 + base = 13
  triangleInequality1 : side1 + side2 > base
  triangleInequality2 : side1 + base > side2
  triangleInequality3 : side2 + base > side1

-- Define the theorem to prove
theorem isosceles_triangle_base_length (T : IsoscelesTriangle) :
  T.base = 3 := by
  sorry

end isosceles_triangle_base_length_l58_58466


namespace linear_function_is_C_l58_58376

theorem linear_function_is_C :
  ∀ (f : ℤ → ℤ), (f = (λ x => 2 * x^2 - 1) ∨ f = (λ x => -1/x) ∨ f = (λ x => (x+1)/3) ∨ f = (λ x => 3 * x + 2 * x^2 - 1)) →
  (f = (λ x => (x+1)/3)) ↔ 
  (∃ (m b : ℤ), ∀ x : ℤ, f x = m * x + b) :=
by
  sorry

end linear_function_is_C_l58_58376


namespace find_c_l58_58066

def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

theorem find_c (c : ℝ) :
  (∀ x, f x c ≤ f 2 c) → c = 6 :=
sorry

end find_c_l58_58066


namespace limit_calculation_l58_58307

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

theorem limit_calculation :
  (Real.exp (-1) * Real.exp 0 - Real.exp (-1) * Real.exp 0) / 0 = -3 / Real.exp 1 := by
  sorry

end limit_calculation_l58_58307


namespace multiply_expand_l58_58634

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l58_58634


namespace solution_of_two_quadratics_l58_58327

theorem solution_of_two_quadratics (x : ℝ) (h1 : 8 * x^2 + 7 * x - 1 = 0) (h2 : 24 * x^2 + 53 * x - 7 = 0) : x = 1 / 8 := 
by 
  sorry

end solution_of_two_quadratics_l58_58327


namespace color_preference_l58_58716

-- Define the conditions
def total_students := 50
def girls := 30
def boys := 20

def girls_pref_pink := girls / 3
def girls_pref_purple := 2 * girls / 5
def girls_pref_blue := girls - girls_pref_pink - girls_pref_purple

def boys_pref_red := 2 * boys / 5
def boys_pref_green := 3 * boys / 10
def boys_pref_orange := boys - boys_pref_red - boys_pref_green

-- Proof statement
theorem color_preference :
  girls_pref_pink = 10 ∧
  girls_pref_purple = 12 ∧
  girls_pref_blue = 8 ∧
  boys_pref_red = 8 ∧
  boys_pref_green = 6 ∧
  boys_pref_orange = 6 :=
by
  sorry

end color_preference_l58_58716


namespace floor_sum_even_l58_58650

theorem floor_sum_even (a b c : ℕ) (h1 : a^2 + b^2 + 1 = c^2) : 
    ((a / 2) + (c / 2)) % 2 = 0 := 
  sorry

end floor_sum_even_l58_58650


namespace triangle_acute_angle_sufficient_condition_triangle_acute_angle_not_necessary_condition_l58_58984

theorem triangle_acute_angle_sufficient_condition
  (a b c : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a ≤ (b + c) / 2 → b^2 + c^2 > a^2 :=
sorry

theorem triangle_acute_angle_not_necessary_condition
  (a b c : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  b^2 + c^2 > a^2 → ¬ (a ≤ (b + c) / 2) :=
sorry

end triangle_acute_angle_sufficient_condition_triangle_acute_angle_not_necessary_condition_l58_58984


namespace chinaman_change_possible_l58_58802

def pence (x : ℕ) := x -- defining the value of pence as a natural number

def ching_chang_by_value (d : ℕ) := 
  (2 * pence d) + (4 * (2 * pence d) / 15)

def equivalent_value_of_half_crown (d : ℕ) := 30 * pence d

def coin_value_with_holes (holes_value : ℕ) (value_per_eleven : ℕ) := 
  (value_per_eleven * ching_chang_by_value 1) / 11

theorem chinaman_change_possible :
  ∃ (x y z : ℕ), 
  (7 * coin_value_with_holes 15 11) + (1 * coin_value_with_holes 16 11) + (0 * coin_value_with_holes 17 11) = 
  equivalent_value_of_half_crown 1 :=
sorry

end chinaman_change_possible_l58_58802


namespace cakes_bought_l58_58743

theorem cakes_bought (initial : ℕ) (left : ℕ) (bought : ℕ) :
  initial = 169 → left = 32 → bought = initial - left → bought = 137 :=
by
  intros h_initial h_left h_bought
  rw [h_initial, h_left] at h_bought
  exact h_bought

end cakes_bought_l58_58743


namespace inverse_of_matrix_C_l58_58814

-- Define the given matrix C
def C : Matrix (Fin 3) (Fin 3) ℚ := ![
  ![1, 2, 1],
  ![3, -5, 3],
  ![2, 7, -1]
]

-- Define the claimed inverse of the matrix C
def C_inv : Matrix (Fin 3) (Fin 3) ℚ := (1 / 33 : ℚ) • ![
  ![-16,  9,  11],
  ![  9, -3,   0],
  ![ 31, -3, -11]
]

-- Statement to prove that C_inv is the inverse of C
theorem inverse_of_matrix_C : C * C_inv = 1 ∧ C_inv * C = 1 := by
  sorry

end inverse_of_matrix_C_l58_58814


namespace M_eq_N_l58_58128

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r}

theorem M_eq_N : M = N :=
by
  sorry

end M_eq_N_l58_58128


namespace estimate_students_height_at_least_165_l58_58837

theorem estimate_students_height_at_least_165 
  (sample_size : ℕ)
  (total_school_size : ℕ)
  (students_165_170 : ℕ)
  (students_170_175 : ℕ)
  (h_sample : sample_size = 100)
  (h_total_school : total_school_size = 1000)
  (h_students_165_170 : students_165_170 = 20)
  (h_students_170_175 : students_170_175 = 30)
  : (students_165_170 + students_170_175) * (total_school_size / sample_size) = 500 := 
by
  sorry

end estimate_students_height_at_least_165_l58_58837


namespace system_of_equations_solution_exists_l58_58202

theorem system_of_equations_solution_exists :
  ∃ (x y : ℝ), 
    (4 * x^2 + 8 * x * y + 16 * y^2 + 2 * x + 20 * y = -7) ∧
    (2 * x^2 - 16 * x * y + 8 * y^2 - 14 * x + 20 * y = -11) ∧
    (x = 1/2) ∧ (y = -3/4) :=
by
  sorry

end system_of_equations_solution_exists_l58_58202


namespace dante_final_coconuts_l58_58892

theorem dante_final_coconuts
  (Paolo_coconuts : ℕ) (Dante_init_coconuts : ℝ)
  (Bianca_coconuts : ℕ) (Dante_final_coconuts : ℕ):
  Paolo_coconuts = 14 →
  Dante_init_coconuts = 1.5 * Real.sqrt Paolo_coconuts →
  Bianca_coconuts = 2 * (Paolo_coconuts + Int.floor Dante_init_coconuts) →
  Dante_final_coconuts = (Int.floor (Dante_init_coconuts) - (Int.floor (Dante_init_coconuts) / 3)) - 
    (25 * (Int.floor (Dante_init_coconuts) - (Int.floor (Dante_init_coconuts) / 3)) / 100) →
  Dante_final_coconuts = 3 :=
by
  sorry

end dante_final_coconuts_l58_58892


namespace minimum_days_to_pay_back_l58_58663

theorem minimum_days_to_pay_back (x : ℕ) : 
  (50 + 5 * x ≥ 150) → x = 20 :=
sorry

end minimum_days_to_pay_back_l58_58663


namespace tessa_needs_more_apples_l58_58355

/-- Tessa starts with 4 apples.
    Anita gives her 5 more apples.
    She needs 10 apples to make a pie.
    Prove that she needs 1 more apple to make the pie.
-/
theorem tessa_needs_more_apples:
  ∀ initial_apples extra_apples total_needed extra_needed: ℕ,
    initial_apples = 4 → extra_apples = 5 → total_needed = 10 →
    extra_needed = total_needed - (initial_apples + extra_apples) →
    extra_needed = 1 :=
by
  intros initial_apples extra_apples total_needed extra_needed hi he ht heq
  rw [hi, he, ht] at heq
  simp at heq
  assumption

end tessa_needs_more_apples_l58_58355


namespace parabola_directrix_eq_l58_58827

theorem parabola_directrix_eq (x y : ℝ) : x^2 + 12 * y = 0 → y = 3 := 
by sorry

end parabola_directrix_eq_l58_58827


namespace initial_people_count_l58_58535

theorem initial_people_count (x : ℕ) 
  (h1 : (x + 15) % 5 = 0)
  (h2 : (x + 15) / 5 = 12) : 
  x = 45 := 
by
  sorry

end initial_people_count_l58_58535


namespace sin_alpha_cos_squared_beta_range_l58_58078

theorem sin_alpha_cos_squared_beta_range (α β : ℝ) 
  (h : Real.sin α + Real.sin β = 1) : 
  ∃ y, y = Real.sin α - Real.cos β ^ 2 ∧ (-1/4 ≤ y ∧ y ≤ 0) :=
sorry

end sin_alpha_cos_squared_beta_range_l58_58078


namespace total_area_for_building_l58_58188

theorem total_area_for_building (num_sections : ℕ) (area_per_section : ℝ) (open_space_percentage : ℝ) :
  num_sections = 7 →
  area_per_section = 9473 →
  open_space_percentage = 0.15 →
  (num_sections * (area_per_section * (1 - open_space_percentage))) = 56364.35 :=
by
  intros h1 h2 h3
  sorry

end total_area_for_building_l58_58188


namespace estimate_height_of_student_l58_58260

theorem estimate_height_of_student
  (x_values : List ℝ)
  (y_values : List ℝ)
  (h_sum_x : x_values.sum = 225)
  (h_sum_y : y_values.sum = 1600)
  (h_length : x_values.length = 10 ∧ y_values.length = 10)
  (b : ℝ := 4) :
  ∃ a : ℝ, ∀ x : ℝ, x = 24 → (b * x + a = 166) :=
by
  have avg_x := (225 / 10 : ℝ)
  have avg_y := (1600 / 10 : ℝ)
  have a := avg_y - b * avg_x
  use a
  intro x h
  rw [h]
  sorry

end estimate_height_of_student_l58_58260


namespace smallest_integer_solution_l58_58000

theorem smallest_integer_solution : ∃ x : ℤ, (x^2 = 3 * x + 78) ∧ x = -6 :=
by {
  sorry
}

end smallest_integer_solution_l58_58000


namespace exists_x_quadratic_eq_zero_iff_le_one_l58_58864

variable (a : ℝ)

theorem exists_x_quadratic_eq_zero_iff_le_one : (∃ x : ℝ, x^2 - 2 * x + a = 0) ↔ a ≤ 1 :=
sorry

end exists_x_quadratic_eq_zero_iff_le_one_l58_58864


namespace correct_choice_2point5_l58_58583

def set_M : Set ℝ := {x | -2 < x ∧ x < 3}

theorem correct_choice_2point5 : 2.5 ∈ set_M :=
by {
  -- sorry is added to close the proof for now
  sorry
}

end correct_choice_2point5_l58_58583


namespace definite_integral_eval_l58_58778

theorem definite_integral_eval :
  ∫ x in (1:ℝ)..(3:ℝ), (2 * x - 1 / x ^ 2) = 22 / 3 :=
by
  sorry

end definite_integral_eval_l58_58778


namespace imaginary_unit_problem_l58_58238

theorem imaginary_unit_problem (i : ℂ) (h : i^2 = -1) :
  ( (1 + i) / i )^2014 = 2^(1007 : ℤ) * i :=
by sorry

end imaginary_unit_problem_l58_58238


namespace range_f_does_not_include_zero_l58_58478

noncomputable def f (x : ℝ) : ℤ :=
if x > 0 then ⌈1 / (x + 1)⌉ else if x < 0 then ⌈1 / (x - 1)⌉ else 0 -- this will be used only as a formal definition

theorem range_f_does_not_include_zero : ¬ (0 ∈ {y : ℤ | ∃ x : ℝ, x ≠ 0 ∧ y = f x}) :=
by sorry

end range_f_does_not_include_zero_l58_58478


namespace interval_first_bell_l58_58735

theorem interval_first_bell (x : ℕ) : (Nat.lcm (Nat.lcm (Nat.lcm x 10) 14) 18 = 630) → x = 1 := by
  sorry

end interval_first_bell_l58_58735


namespace compute_t_minus_s_l58_58331

noncomputable def t : ℚ := (40 + 30 + 30 + 20) / 4

noncomputable def s : ℚ := (40 * (40 / 120) + 30 * (30 / 120) + 30 * (30 / 120) + 20 * (20 / 120))

theorem compute_t_minus_s : t - s = -1.67 := by
  sorry

end compute_t_minus_s_l58_58331


namespace arithmetic_progression_l58_58615

theorem arithmetic_progression (a b c : ℝ) (h : a + c = 2 * b) :
  3 * (a^2 + b^2 + c^2) = 6 * (a - b)^2 + (a + b + c)^2 :=
by
  sorry

end arithmetic_progression_l58_58615


namespace part_one_part_two_l58_58756

namespace ProofProblem

def setA (a : ℝ) := {x : ℝ | a - 1 < x ∧ x < 2 * a + 1}
def setB := {x : ℝ | 0 < x ∧ x < 1}

theorem part_one (a : ℝ) (h : a = 1/2) : 
  setA a ∩ setB = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

theorem part_two (a : ℝ) (h_subset : setB ⊆ setA a) : 
  0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end ProofProblem

end part_one_part_two_l58_58756


namespace simplify_expression_l58_58437

theorem simplify_expression (n : ℕ) : 
  (3^(n + 3) - 3 * 3^n) / (3 * 3^(n + 2)) = 8 / 3 := 
sorry

end simplify_expression_l58_58437


namespace rate_of_interest_l58_58360

theorem rate_of_interest (P T SI: ℝ) (h1 : P = 2500) (h2 : T = 5) (h3 : SI = P - 2000) (h4 : SI = (P * R * T) / 100):
  R = 4 :=
by
  sorry

end rate_of_interest_l58_58360


namespace tissue_pallets_ratio_l58_58700

-- Define the total number of pallets received
def total_pallets : ℕ := 20

-- Define the number of pallets of each type
def paper_towels_pallets : ℕ := total_pallets / 2
def paper_plates_pallets : ℕ := total_pallets / 5
def paper_cups_pallets : ℕ := 1

-- Calculate the number of pallets of tissues
def tissues_pallets : ℕ := total_pallets - (paper_towels_pallets + paper_plates_pallets + paper_cups_pallets)

-- Prove the ratio of pallets of tissues to total pallets is 1/4
theorem tissue_pallets_ratio : (tissues_pallets : ℚ) / total_pallets = 1 / 4 :=
by
  -- Proof goes here
  sorry

end tissue_pallets_ratio_l58_58700


namespace johns_number_l58_58023

theorem johns_number (n : ℕ) :
  64 ∣ n ∧ 45 ∣ n ∧ 1000 < n ∧ n < 3000 -> n = 2880 :=
by
  sorry

end johns_number_l58_58023


namespace inequality_solution_l58_58135

theorem inequality_solution (x : ℝ) : 
  (3 / 20 + abs (2 * x - 5 / 40) < 9 / 40) → (1 / 40 < x ∧ x < 1 / 10) :=
by
  sorry

end inequality_solution_l58_58135


namespace compute_b_l58_58948

theorem compute_b (x y b : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 7 * y = 3 * b) (hx : x = 3) : b = 66 :=
sorry

end compute_b_l58_58948


namespace contribution_is_6_l58_58246

-- Defining the earnings of each friend
def earning_1 : ℕ := 18
def earning_2 : ℕ := 22
def earning_3 : ℕ := 30
def earning_4 : ℕ := 35
def earning_5 : ℕ := 45

-- Defining the modified contribution for the highest earner
def modified_earning_5 : ℕ := 40

-- Calculate the total adjusted earnings
def total_earnings : ℕ := earning_1 + earning_2 + earning_3 + earning_4 + modified_earning_5

-- Calculate the equal share each friend should receive
def equal_share : ℕ := total_earnings / 5

-- Calculate the contribution needed from the friend who earned $35 to match the equal share
def contribution_from_earning_4 : ℕ := earning_4 - equal_share

-- Stating the proof problem
theorem contribution_is_6 : contribution_from_earning_4 = 6 := by
  sorry

end contribution_is_6_l58_58246


namespace eval_polynomial_at_3_l58_58568

def f (x : ℝ) : ℝ := 2 * x^5 + 5 * x^4 + 8 * x^3 + 7 * x^2 - 6 * x + 11

theorem eval_polynomial_at_3 : f 3 = 130 :=
by
  -- proof can be completed here following proper steps or using Horner's method
  sorry

end eval_polynomial_at_3_l58_58568


namespace total_monthly_sales_l58_58483

-- Definitions and conditions
def num_customers_per_month : ℕ := 500
def lettuce_per_customer : ℕ := 2
def price_per_lettuce : ℕ := 1
def tomatoes_per_customer : ℕ := 4
def price_per_tomato : ℕ := 1 / 2

-- Statement to prove
theorem total_monthly_sales : num_customers_per_month * (lettuce_per_customer * price_per_lettuce + tomatoes_per_customer * price_per_tomato) = 2000 := 
by 
  sorry

end total_monthly_sales_l58_58483


namespace football_field_width_l58_58382

theorem football_field_width (length : ℕ) (total_distance : ℕ) (laps : ℕ) (width : ℕ) 
  (h1 : length = 100) (h2 : total_distance = 1800) (h3 : laps = 6) :
  width = 50 :=
by 
  -- Proof omitted
  sorry

end football_field_width_l58_58382


namespace album_count_l58_58847

theorem album_count (A B S : ℕ) (hA : A = 23) (hB : B = 9) (hS : S = 15) : 
  (A - S) + B = 17 :=
by
  -- Variables and conditions
  have Andrew_unique : ℕ := A - S
  have Bella_unique : ℕ := B
  -- Proof starts here
  sorry

end album_count_l58_58847


namespace solution_set_inequality_l58_58262

theorem solution_set_inequality (x : ℝ) (h1 : x < -3) (h2 : x < 2) : x < -3 :=
by
  exact h1

end solution_set_inequality_l58_58262


namespace total_books_l58_58973

-- Definitions for the conditions
def SandyBooks : Nat := 10
def BennyBooks : Nat := 24
def TimBooks : Nat := 33

-- Stating the theorem we need to prove
theorem total_books : SandyBooks + BennyBooks + TimBooks = 67 := by
  sorry

end total_books_l58_58973


namespace city_map_distance_example_l58_58517

variable (distance_on_map : ℝ)
variable (scale : ℝ)
variable (actual_distance : ℝ)

theorem city_map_distance_example
  (h1 : distance_on_map = 16)
  (h2 : scale = 1 / 10000)
  (h3 : actual_distance = distance_on_map / scale) :
  actual_distance = 1.6 * 10^3 :=
by
  sorry

end city_map_distance_example_l58_58517


namespace sasha_added_num_l58_58896

theorem sasha_added_num (a b c : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a / b = 5 * (a + c) / (b * c)) : c = 6 ∨ c = -20 := 
sorry

end sasha_added_num_l58_58896


namespace movie_box_office_revenue_l58_58138

variable (x : ℝ)

theorem movie_box_office_revenue (h : 300 + 300 * (1 + x) + 300 * (1 + x)^2 = 1000) :
  3 + 3 * (1 + x) + 3 * (1 + x)^2 = 10 :=
by
  sorry

end movie_box_office_revenue_l58_58138


namespace x7_value_l58_58338

theorem x7_value
  (x : ℕ → ℕ)
  (h1 : x 6 = 144)
  (h2 : ∀ n, 1 ≤ n ∧ n ≤ 4 → x (n + 3) = x (n + 2) * (x (n + 1) + x n))
  (h3 : ∀ m, m < 1 → 0 < x m) : x 7 = 3456 :=
by
  sorry

end x7_value_l58_58338


namespace limping_rook_adjacent_sum_not_divisible_by_4_l58_58671

/-- Problem statement: A limping rook traversed a 10 × 10 board,
visiting each square exactly once with numbers 1 through 100
written in the order visited.
Prove that the sum of the numbers in any two adjacent cells
is not divisible by 4. -/
theorem limping_rook_adjacent_sum_not_divisible_by_4 :
  ∀ (board : Fin 10 → Fin 10 → ℕ), 
  (∀ (i j : Fin 10), 1 ≤ board i j ∧ board i j ≤ 100) →
  (∀ (i j : Fin 10), (∃ (i' : Fin 10), i = i' + 1 ∨ i = i' - 1)
                 ∨ (∃ (j' : Fin 10), j = j' + 1 ∨ j = j' - 1)) →
  ((∀ (i j : Fin 10) (k l : Fin 10),
      (i = k ∧ (j = l + 1 ∨ j = l - 1) ∨ j = l ∧ (i = k + 1 ∨ i = k - 1)) →
      (board i j + board k l) % 4 ≠ 0)) :=
by
  sorry

end limping_rook_adjacent_sum_not_divisible_by_4_l58_58671


namespace longer_diagonal_is_116_l58_58144

-- Given conditions
def side_length : ℕ := 65
def short_diagonal : ℕ := 60

-- Prove that the length of the longer diagonal in the rhombus is 116 units.
theorem longer_diagonal_is_116 : 
  let s := side_length
  let d1 := short_diagonal / 2
  let d2 := (s^2 - d1^2).sqrt
  (2 * d2) = 116 :=
by
  sorry

end longer_diagonal_is_116_l58_58144


namespace cyclic_sequence_u_16_eq_a_l58_58748

-- Sequence definition and recurrence relation
def cyclic_sequence (u : ℕ → ℝ) (a : ℝ) : Prop :=
  u 1 = a ∧ ∀ n : ℕ, u (n + 1) = -1 / (u n + 1)

-- Proof that u_{16} = a under given conditions
theorem cyclic_sequence_u_16_eq_a (a : ℝ) (h : 0 < a) : ∃ (u : ℕ → ℝ), cyclic_sequence u a ∧ u 16 = a :=
by
  sorry

end cyclic_sequence_u_16_eq_a_l58_58748


namespace sum_non_solutions_is_neg21_l58_58758

noncomputable def sum_of_non_solutions (A B C : ℝ) (h1 : ∀ x : ℝ, ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) ≠ 2) : ℝ :=
  -21

theorem sum_non_solutions_is_neg21 (A B C : ℝ) (h1 : ∀ x : ℝ, ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) = 2) : 
  ∃! (x1 x2 : ℝ), ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) ≠ 2 → x = x1 ∨ x = x2 ∧ x1 + x2 = -21 :=
sorry

end sum_non_solutions_is_neg21_l58_58758


namespace distance_travelled_l58_58164

theorem distance_travelled (t : ℝ) (h : 15 * t = 10 * t + 20) : 10 * t = 40 :=
by
  have ht : t = 4 := by linarith
  rw [ht]
  norm_num

end distance_travelled_l58_58164


namespace remainder_is_correct_l58_58755

def P (x : ℝ) : ℝ := x^6 + 2 * x^5 - 3 * x^4 + x^2 - 8
def D (x : ℝ) : ℝ := x^2 - 1

theorem remainder_is_correct : 
  ∃ q : ℝ → ℝ, ∀ x : ℝ, P x = D x * q x + (2.5 * x - 9.5) :=
by
  sorry

end remainder_is_correct_l58_58755


namespace part_a_part_b_l58_58033

noncomputable def same_start_digit (n x : ℕ) : Prop :=
  ∃ d : ℕ, ∀ k : ℕ, (k ≤ n) → (x * 10^(k-1) ≤ d * 10^(k-1) + 10^(k-1) - 1) ∧ ((d * 10^(k-1)) < x * 10^(k-1))

theorem part_a (x : ℕ) : 
  (same_start_digit 3 x) → ¬(∃ d : ℕ, d = 1) → false :=
  sorry

theorem part_b (x : ℕ) : 
  (same_start_digit 2015 x) → ¬(∃ d : ℕ, d = 1) → false :=
  sorry

end part_a_part_b_l58_58033


namespace reciprocal_of_neg_two_thirds_l58_58055

-- Definition for finding the reciprocal
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The proof problem statement
theorem reciprocal_of_neg_two_thirds : reciprocal (-2 / 3) = -3 / 2 :=
sorry

end reciprocal_of_neg_two_thirds_l58_58055


namespace toilet_paper_packs_needed_l58_58958

-- Definitions based on conditions
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def weeks : ℕ := 4
def rolls_per_pack : ℕ := 12
def daily_stock : ℕ := 1

-- The main theorem statement
theorem toilet_paper_packs_needed : 
  (bathrooms * days_per_week * weeks) / rolls_per_pack = 14 := by
sorry

end toilet_paper_packs_needed_l58_58958


namespace share_per_person_is_135k_l58_58141

noncomputable def calculate_share : ℝ :=
  (0.90 * (500000 * 1.20)) / 4

theorem share_per_person_is_135k : calculate_share = 135000 :=
by
  sorry

end share_per_person_is_135k_l58_58141


namespace arithmetic_expression_l58_58267

theorem arithmetic_expression : 8 / 4 + 5 * 2 ^ 2 - (3 + 7) = 12 := by
  sorry

end arithmetic_expression_l58_58267


namespace sum_of_two_numbers_l58_58362

theorem sum_of_two_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 16) (h2 : (1 / x) = 3 * (1 / y)) : 
  x + y = (16 * Real.sqrt 3) / 3 := 
sorry

end sum_of_two_numbers_l58_58362


namespace calculate_t_minus_d_l58_58585

def tom_pays : ℕ := 150
def dorothy_pays : ℕ := 190
def sammy_pays : ℕ := 240
def nancy_pays : ℕ := 320
def total_expenses := tom_pays + dorothy_pays + sammy_pays + nancy_pays
def individual_share := total_expenses / 4
def tom_needs_to_pay := individual_share - tom_pays
def dorothy_needs_to_pay := individual_share - dorothy_pays
def sammy_should_receive := sammy_pays - individual_share
def nancy_should_receive := nancy_pays - individual_share
def t := tom_needs_to_pay
def d := dorothy_needs_to_pay

theorem calculate_t_minus_d : t - d = 40 :=
by
  sorry

end calculate_t_minus_d_l58_58585


namespace find_a_l58_58704

/-- 
Given sets A and B defined by specific quadratic equations, 
if A ∪ B = A, then a ∈ (-∞, 0).
-/
theorem find_a :
  ∀ (a : ℝ),
    (A = {x : ℝ | x^2 - 3 * x + 2 = 0}) →
    (B = {x : ℝ | x^2 - 2 * a * x + a^2 - a = 0}) →
    (A ∪ B = A) →
    a < 0 :=
by
  sorry

end find_a_l58_58704


namespace heroes_on_the_back_l58_58052

theorem heroes_on_the_back (total_heroes front_heroes : ℕ) (h1 : total_heroes = 9) (h2 : front_heroes = 2) :
  total_heroes - front_heroes = 7 := by
  sorry

end heroes_on_the_back_l58_58052


namespace students_not_playing_games_l58_58150

theorem students_not_playing_games 
  (total_students : ℕ)
  (basketball_players : ℕ)
  (volleyball_players : ℕ)
  (both_players : ℕ)
  (h1 : total_students = 20)
  (h2 : basketball_players = (1 / 2) * total_students)
  (h3 : volleyball_players = (2 / 5) * total_students)
  (h4 : both_players = (1 / 10) * total_students) :
  total_students - ((basketball_players + volleyball_players) - both_players) = 4 :=
by
  sorry

end students_not_playing_games_l58_58150


namespace biscuits_per_guest_correct_l58_58997

def flour_per_batch : ℚ := 5 / 4
def biscuits_per_batch : ℕ := 9
def flour_needed : ℚ := 5
def guests : ℕ := 18

theorem biscuits_per_guest_correct :
  (flour_needed * biscuits_per_batch / flour_per_batch) / guests = 2 := by
  sorry

end biscuits_per_guest_correct_l58_58997


namespace calculate_product_l58_58446

theorem calculate_product : 3^6 * 4^3 = 46656 := by
  sorry

end calculate_product_l58_58446


namespace max_P_l58_58195

noncomputable def P (a b : ℝ) : ℝ :=
  (a^2 + 6*b + 1) / (a^2 + a)

theorem max_P (a b x1 x2 x3 : ℝ) (h1 : a = x1 + x2 + x3) (h2 : a = x1 * x2 * x3) (h3 : ab = x1 * x2 + x2 * x3 + x3 * x1) 
    (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
    P a b ≤ (9 + Real.sqrt 3) / 9 := 
sorry

end max_P_l58_58195


namespace bottle_cost_l58_58668

-- Definitions of the conditions
def total_cost := 30
def wine_extra_cost := 26

-- Statement of the problem in Lean 4
theorem bottle_cost : 
  ∃ x : ℕ, (x + (x + wine_extra_cost) = total_cost) ∧ x = 2 :=
by
  sorry

end bottle_cost_l58_58668


namespace hyperbola_asymptotes_l58_58893

theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, x^2 / 16 - y^2 / 9 = -1 → (y = 3/4 * x ∨ y = -3/4 * x)) :=
by
  sorry

end hyperbola_asymptotes_l58_58893


namespace no_tangential_triangle_exists_l58_58884

-- Define the first circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the ellipse C2
def C2 (a b x y : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Additional condition that the point (1, 1) lies on C2
def point_on_C2 (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (1^2) / (a^2) + (1^2) / (b^2) = 1

-- The theorem to prove
theorem no_tangential_triangle_exists (a b : ℝ) (h : a > b ∧ b > 0) :
  point_on_C2 a b h →
  ¬ ∃ (A B C : ℝ × ℝ), 
    (C1 A.1 A.2 ∧ C1 B.1 B.2 ∧ C1 C.1 C.2) ∧ 
    (C2 a b A.1 A.2 h ∧ C2 a b B.1 B.2 h ∧ C2 a b C.1 C.2 h) :=
by sorry

end no_tangential_triangle_exists_l58_58884


namespace school_dance_boys_count_l58_58889

theorem school_dance_boys_count :
  let total_attendees := 100
  let faculty_and_staff := total_attendees * 10 / 100
  let students := total_attendees - faculty_and_staff
  let girls := 2 * students / 3
  let boys := students - girls
  boys = 30 := by
  sorry

end school_dance_boys_count_l58_58889


namespace number_of_solutions_l58_58177

noncomputable def g_n (n : ℕ) (x : ℝ) := (Real.sin x)^(2 * n) + (Real.cos x)^(2 * n)

theorem number_of_solutions : ∀ (x : ℝ), x ∈ Set.Icc 0 (2 * Real.pi) -> 
  8 * g_n 3 x - 6 * g_n 2 x = 3 * g_n 1 x -> false :=
by sorry

end number_of_solutions_l58_58177


namespace floor_diff_l58_58015

theorem floor_diff {x : ℝ} (h : x = 12.7) : 
  (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * (⌊x⌋ : ℤ) = 17 :=
by
  have h1 : x = 12.7 := h
  have hx2 : x^2 = 161.29 := by sorry
  have hfloor : ⌊x⌋ = 12 := by sorry
  have hfloor2 : ⌊x^2⌋ = 161 := by sorry
  sorry

end floor_diff_l58_58015


namespace total_cost_pencils_and_pens_l58_58881

def pencil_cost : ℝ := 2.50
def pen_cost : ℝ := 3.50
def num_pencils : ℕ := 38
def num_pens : ℕ := 56

theorem total_cost_pencils_and_pens :
  (pencil_cost * ↑num_pencils + pen_cost * ↑num_pens) = 291 :=
sorry

end total_cost_pencils_and_pens_l58_58881


namespace inequality_proof_l58_58687

theorem inequality_proof (a b c : ℝ) : 
  1 < (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + 
  c / (Real.sqrt (c^2 + a^2))) ∧ 
  (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + 
  c / (Real.sqrt (c^2 + a^2))) ≤ (3 * Real.sqrt 2 / 2) :=
by
  sorry

end inequality_proof_l58_58687


namespace Martha_needs_54_cakes_l58_58559

theorem Martha_needs_54_cakes :
  let n_children := 3
  let n_cakes_per_child := 18
  let n_cakes_total := 54
  n_cakes_total = n_children * n_cakes_per_child :=
by
  sorry

end Martha_needs_54_cakes_l58_58559


namespace relay_scheme_count_l58_58731

theorem relay_scheme_count
  (num_segments : ℕ)
  (num_torchbearers : ℕ)
  (first_choices : ℕ)
  (last_choices : ℕ) :
  num_segments = 6 ∧
  num_torchbearers = 6 ∧
  first_choices = 3 ∧
  last_choices = 2 →
  ∃ num_schemes : ℕ, num_schemes = 7776 :=
by
  intro h
  obtain ⟨h_segments, h_torchbearers, h_first_choices, h_last_choices⟩ := h
  exact ⟨7776, sorry⟩

end relay_scheme_count_l58_58731


namespace origin_moves_3sqrt5_under_dilation_l58_58843

/--
Given:
1. The original circle has radius 3 centered at point B(3, 3).
2. The dilated circle has radius 6 centered at point B'(9, 12).

Prove that the distance moved by the origin O(0, 0) under this dilation is 3 * sqrt(5).
-/
theorem origin_moves_3sqrt5_under_dilation:
  let B := (3, 3)
  let B' := (9, 12)
  let radius_B := 3
  let radius_B' := 6
  let dilation_center := (-3, -6)
  let origin := (0, 0)
  let k := radius_B' / radius_B
  let d_0 := Real.sqrt ((-3 : ℝ)^2 + (-6 : ℝ)^2)
  let d_1 := k * d_0
  d_1 - d_0 = 3 * Real.sqrt (5 : ℝ) := by sorry

end origin_moves_3sqrt5_under_dilation_l58_58843


namespace monotonically_increasing_function_l58_58406

open Function

theorem monotonically_increasing_function (f : ℝ → ℝ) (h_mono : ∀ x y, x < y → f x < f y) (t : ℝ) (h_t : t ≠ 0) :
    f (t^2 + t) > f t :=
by
  sorry

end monotonically_increasing_function_l58_58406


namespace cows_with_no_spots_l58_58275

-- Definitions of conditions
def total_cows : Nat := 140
def cows_with_red_spot : Nat := (40 * total_cows) / 100
def cows_without_red_spot : Nat := total_cows - cows_with_red_spot
def cows_with_blue_spot : Nat := (25 * cows_without_red_spot) / 100

-- Theorem statement asserting the number of cows with no spots
theorem cows_with_no_spots : (total_cows - cows_with_red_spot - cows_with_blue_spot) = 63 := by
  -- Proof would go here
  sorry

end cows_with_no_spots_l58_58275


namespace framed_painting_ratio_l58_58245

/-- A rectangular painting measuring 20" by 30" is to be framed, with the longer dimension vertical.
The width of the frame at the top and bottom is three times the width of the frame on the sides.
Given that the total area of the frame equals the area of the painting, the ratio of the smaller to the 
larger dimension of the framed painting is 4:7. -/
theorem framed_painting_ratio : 
  ∀ (w h : ℝ) (side_frame_width : ℝ), 
    w = 20 ∧ h = 30 ∧ 3 * side_frame_width * (2 * (w + 2 * side_frame_width) + 2 * (h + 6 * side_frame_width) - w * h) = w * h 
    → side_frame_width = 2 
    → (w + 2 * side_frame_width) / (h + 6 * side_frame_width) = 4 / 7 :=
sorry

end framed_painting_ratio_l58_58245


namespace Kenneth_money_left_l58_58302

def initial_amount : ℕ := 50
def number_of_baguettes : ℕ := 2
def cost_per_baguette : ℕ := 2
def number_of_bottles : ℕ := 2
def cost_per_bottle : ℕ := 1

-- This theorem states that Kenneth has $44 left after his purchases.
theorem Kenneth_money_left : initial_amount - (number_of_baguettes * cost_per_baguette + number_of_bottles * cost_per_bottle) = 44 := by
  sorry

end Kenneth_money_left_l58_58302


namespace functional_equation_solution_l58_58530

-- Define the conditions of the problem.
variable (f : ℝ → ℝ) 
variable (h : ∀ x y u v : ℝ, (f x + f y) * (f u + f v) = f (x * u - y * v) + f (x * v + y * u))

-- Formalize the statement that no other functions satisfy the conditions except f(x) = x^2.
theorem functional_equation_solution : (∀ x : ℝ, f x = x^2) :=
by
  -- The proof goes here, but since the proof is not required, we skip it.
  sorry

end functional_equation_solution_l58_58530


namespace division_quotient_less_dividend_l58_58528

theorem division_quotient_less_dividend
  (a1 : (6 : ℝ) > 0)
  (a2 : (5 / 7 : ℝ) > 0)
  (a3 : (3 / 8 : ℝ) > 0)
  (h1 : (3 / 5 : ℝ) < 1)
  (h2 : (5 / 4 : ℝ) > 1)
  (h3 : (5 / 12 : ℝ) < 1):
  (6 / (3 / 5) > 6) ∧ (5 / 7 / (5 / 4) < 5 / 7) ∧ (3 / 8 / (5 / 12) > 3 / 8) :=
by
  sorry

end division_quotient_less_dividend_l58_58528


namespace arun_age_l58_58549

variable (A S G M : ℕ)

theorem arun_age (h1 : A - 6 = 18 * G)
                 (h2 : G + 2 = M)
                 (h3 : M = 5)
                 (h4 : S = A - 8) : A = 60 :=
by sorry

end arun_age_l58_58549


namespace relation_between_x_and_y_l58_58300

variable (t : ℝ)
variable (x : ℝ := t ^ (2 / (t - 1))) (y : ℝ := t ^ ((t + 1) / (t - 1)))

theorem relation_between_x_and_y (h1 : t > 0) (h2 : t ≠ 1) : y ^ (1 / x) = x ^ y :=
by sorry

end relation_between_x_and_y_l58_58300


namespace FO_greater_DI_l58_58037

-- The quadrilateral FIDO is assumed to be convex with specified properties
variables {F I D O E : Type*}

variables (length_FI length_DI length_DO length_FO : ℝ)
variables (angle_FIO angle_DIO : ℝ)
variables (E : I)

-- Given conditions
variables (convex_FIDO : Prop) -- FIDO is convex
variables (h1 : length_FI = length_DO)
variables (h2 : length_FI > length_DI)
variables (h3 : angle_FIO = angle_DIO)

-- Use given identity IE = ID
variables (length_IE : ℝ) (h4 : length_IE = length_DI)

theorem FO_greater_DI 
    (length_FI length_DI length_DO length_FO : ℝ)
    (angle_FIO angle_DIO : ℝ)
    (convex_FIDO : Prop)
    (h1 : length_FI = length_DO)
    (h2 : length_FI > length_DI)
    (h3 : angle_FIO = angle_DIO)
    (length_IE : ℝ)
    (h4 : length_IE = length_DI) : 
    length_FO > length_DI :=
sorry

end FO_greater_DI_l58_58037


namespace liams_numbers_l58_58278

theorem liams_numbers (x y : ℤ) 
  (h1 : 3 * x + 2 * y = 75)
  (h2 : x = 15)
  (h3 : ∃ k : ℕ, x * y = 5 * k) : 
  y = 15 := 
by
  sorry

end liams_numbers_l58_58278


namespace range_of_x_in_tight_sequence_arithmetic_tight_sequence_geometric_tight_sequence_l58_58295

-- Problem (1)
theorem range_of_x_in_tight_sequence (a : ℕ → ℝ) (x : ℝ) (h : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = x ∧ a 4 = 4 → 2 ≤ x ∧ x ≤ 3 :=
sorry

-- Problem (2)
theorem arithmetic_tight_sequence (a : ℕ → ℝ) (a1 d : ℝ) (h : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2) :
  ∀ n : ℕ, a n = a1 + ↑n * d → 0 < d ∧ d ≤ a1 →
  ∀ n : ℕ, 1 / 2 ≤ (a (n + 1) / a n) ∧ (a (n + 1) / a n) ≤ 2 :=
sorry

-- Problem (3)
theorem geometric_tight_sequence (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) (h_seq : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2)
(S : ℕ → ℝ) (h_sum_seq : ∀ n : ℕ, 1 / 2 ≤ S (n + 1) / S n ∧ S (n + 1) / S n ≤ 2) :
  (∀ n : ℕ, a n = a1 * q ^ n ∧ S n = a1 * (1 - q ^ (n + 1)) / (1 - q)) → 
  1 / 2 ≤ q ∧ q ≤ 1 :=
sorry

end range_of_x_in_tight_sequence_arithmetic_tight_sequence_geometric_tight_sequence_l58_58295


namespace complex_number_solution_l58_58315

theorem complex_number_solution (z : ℂ) (i : ℂ) (h_i : i * i = -1) (h : z + z * i = 1 + 5 * i) : z = 3 + 2 * i :=
sorry

end complex_number_solution_l58_58315


namespace isosceles_triangle_aacute_l58_58742

theorem isosceles_triangle_aacute (a b c : ℝ) (h1 : a = b) (h2 : a + b + c = 180) (h3 : c = 108)
  : ∃ x y z : ℝ, x + y + z = 180 ∧ x < 90 ∧ y < 90 ∧ z < 90 ∧ x > 0 ∧ y > 0 ∧ z > 0 :=
by {
  sorry
}

end isosceles_triangle_aacute_l58_58742


namespace circumcircle_radius_of_right_triangle_l58_58554

theorem circumcircle_radius_of_right_triangle (r : ℝ) (BC : ℝ) (R : ℝ) 
  (h1 : r = 3) (h2 : BC = 10) : R = 7.25 := 
by
  sorry

end circumcircle_radius_of_right_triangle_l58_58554


namespace find_a_l58_58582

theorem find_a (x a : ℝ) (h₁ : x = 2) (h₂ : (4 - x) / 2 + a = 4) : a = 3 :=
by
  -- Proof steps will go here
  sorry

end find_a_l58_58582


namespace math_problem_l58_58397

theorem math_problem 
  (x y z : ℚ)
  (h1 : 4 * x - 5 * y - z = 0)
  (h2 : x + 5 * y - 18 * z = 0)
  (hz : z ≠ 0) :
  (x^2 + 4 * x * y) / (y^2 + z^2) = 3622 / 9256 := 
sorry

end math_problem_l58_58397


namespace target_average_income_l58_58608

variable (past_incomes : List ℕ) (next_average : ℕ)

def total_past_income := past_incomes.sum
def total_next_income := next_average * 5
def total_ten_week_income := total_past_income past_incomes + total_next_income next_average

theorem target_average_income (h1 : past_incomes = [406, 413, 420, 436, 395])
                              (h2 : next_average = 586) :
  total_ten_week_income past_incomes next_average / 10 = 500 := by
  sorry

end target_average_income_l58_58608


namespace find_D_l58_58017

noncomputable def point := (ℝ × ℝ)

def vector_add (u v : point) : point := (u.1 + v.1, u.2 + v.2)
def vector_sub (u v : point) : point := (u.1 - v.1, u.2 - v.2)
def scalar_multiplication (k : ℝ) (u : point) : point := (k * u.1, k * u.2)

namespace GeometryProblem

def A : point := (2, 3)
def B : point := (-1, 5)

def D : point := 
  let AB := vector_sub B A
  vector_add A (scalar_multiplication 3 AB)

theorem find_D : D = (-7, 9) := by
  sorry

end GeometryProblem

end find_D_l58_58017


namespace any_nat_representation_as_fraction_l58_58769

theorem any_nat_representation_as_fraction (n : ℕ) : 
    ∃ x y : ℕ, y ≠ 0 ∧ (x^3 : ℚ) / (y^4 : ℚ) = n := by
  sorry

end any_nat_representation_as_fraction_l58_58769


namespace number_of_squares_l58_58828

theorem number_of_squares (total_streetlights squares_streetlights unused_streetlights : ℕ) 
  (h1 : total_streetlights = 200) 
  (h2 : squares_streetlights = 12) 
  (h3 : unused_streetlights = 20) : 
  (∃ S : ℕ, total_streetlights = squares_streetlights * S + unused_streetlights ∧ S = 15) :=
by
  sorry

end number_of_squares_l58_58828


namespace evaluate_expression_at_2_l58_58276

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 9) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := 2 * x - 3

theorem evaluate_expression_at_2 : f (g 2) + g (f 2) = 331 / 20 :=
by
  sorry

end evaluate_expression_at_2_l58_58276


namespace factor_quadratic_l58_58795

theorem factor_quadratic (x : ℝ) (m n : ℝ) 
  (hm : m^2 = 16) (hn : n^2 = 25) (hmn : 2 * m * n = 40) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := 
by sorry

end factor_quadratic_l58_58795


namespace power_mod_l58_58034

theorem power_mod (h1: 5^2 % 17 = 8) (h2: 5^4 % 17 = 13) (h3: 5^8 % 17 = 16) (h4: 5^16 % 17 = 1):
  5^2024 % 17 = 16 :=
by
  sorry

end power_mod_l58_58034


namespace heartsuit_fraction_l58_58910

def heartsuit (n m : ℕ) : ℕ := n ^ 4 * m ^ 3

theorem heartsuit_fraction :
  (heartsuit 3 5) / (heartsuit 5 3) = 3 / 5 :=
by
  sorry

end heartsuit_fraction_l58_58910


namespace reading_club_coordinator_selection_l58_58333

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem reading_club_coordinator_selection :
  let total_ways := choose 18 4
  let no_former_ways := choose 10 4
  total_ways - no_former_ways = 2850 := by
  sorry

end reading_club_coordinator_selection_l58_58333


namespace least_possible_lcm_l58_58079

-- Definitions of the least common multiples given the conditions
variable (a b c : ℕ)
variable (h₁ : Nat.lcm a b = 20)
variable (h₂ : Nat.lcm b c = 28)

-- The goal is to prove the least possible value of lcm(a, c) given the conditions
theorem least_possible_lcm (a b c : ℕ) (h₁ : Nat.lcm a b = 20) (h₂ : Nat.lcm b c = 28) : Nat.lcm a c = 35 :=
by
  sorry

end least_possible_lcm_l58_58079


namespace simplify_expression_l58_58922

-- Define the given expression
def given_expr (x y : ℝ) := 3 * x + 4 * y + 5 * x^2 + 2 - (8 - 5 * x - 3 * y - 2 * x^2)

-- Define the expected simplified expression
def simplified_expr (x y : ℝ) := 7 * x^2 + 8 * x + 7 * y - 6

-- Theorem statement to prove the equivalence of the expressions
theorem simplify_expression (x y : ℝ) : 
  given_expr x y = simplified_expr x y := sorry

end simplify_expression_l58_58922


namespace domain_range_a_l58_58820

theorem domain_range_a (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x + a > 0) ↔ 1 < a :=
by
  sorry

end domain_range_a_l58_58820


namespace triangle_type_is_isosceles_l58_58467

theorem triangle_type_is_isosceles {A B C : ℝ}
  (h1 : A + B + C = π)
  (h2 : ∀ x : ℝ, x^2 - x * (Real.cos A * Real.cos B) + 2 * Real.sin (C / 2)^2 = 0)
  (h3 : ∃ x1 x2 : ℝ, x1 + x2 = Real.cos A * Real.cos B ∧ x1 * x2 = 2 * Real.sin (C / 2)^2 ∧ (x1 + x2 = (x1 * x2) / 2)) :
  A = B ∨ B = C ∨ C = A := 
sorry

end triangle_type_is_isosceles_l58_58467


namespace ratio_won_to_lost_l58_58068

-- Define the total number of games and the number of games won
def total_games : Nat := 30
def games_won : Nat := 18

-- Define the number of games lost
def games_lost : Nat := total_games - games_won

-- Define the ratio of games won to games lost as a pair
def ratio : Nat × Nat := (games_won / Nat.gcd games_won games_lost, games_lost / Nat.gcd games_won games_lost)

-- The theorem to be proved
theorem ratio_won_to_lost : ratio = (3, 2) :=
  by
    -- Skipping the proof here
    sorry

end ratio_won_to_lost_l58_58068


namespace correct_transformation_l58_58148

theorem correct_transformation (x y : ℤ) (h : x = y) : x - 2 = y - 2 :=
by
  sorry

end correct_transformation_l58_58148


namespace opposite_of_one_third_l58_58885

theorem opposite_of_one_third : -(1/3) = -1/3 := by
  sorry

end opposite_of_one_third_l58_58885


namespace card_probability_ratio_l58_58741

theorem card_probability_ratio :
  let total_cards := 40
  let numbers := 10
  let cards_per_number := 4
  let choose (n k : ℕ) := Nat.choose n k
  let p := 10 / choose total_cards 4
  let q := 1440 / choose total_cards 4
  (q / p) = 144 :=
by
  sorry

end card_probability_ratio_l58_58741


namespace max_band_members_l58_58808

theorem max_band_members (k n m : ℕ) : m = k^2 + 11 → m = n * (n + 9) → m ≤ 112 :=
by
  sorry

end max_band_members_l58_58808


namespace find_a_l58_58216

theorem find_a (a : ℝ) (M : Set ℝ) (N : Set ℝ) : 
  M = {1, 3} → N = {1 - a, 3} → (M ∪ N) = {1, 2, 3} → a = -1 :=
by
  intros hM hN hUnion
  sorry

end find_a_l58_58216


namespace fraction_of_planted_area_l58_58417

-- Definitions of the conditions
def right_triangle (a b : ℕ) : Prop :=
  a * a + b * b = (Int.sqrt (a ^ 2 + b ^ 2))^2

def unplanted_square_distance (dist : ℕ) : Prop :=
  dist = 3

-- The main theorem to be proved
theorem fraction_of_planted_area (a b : ℕ) (dist : ℕ) (h_triangle : right_triangle a b) (h_square_dist : unplanted_square_distance dist) :
  (a = 5) → (b = 12) → ((a * b - dist ^ 2) / (a * b) = 412 / 1000) :=
by
  sorry

end fraction_of_planted_area_l58_58417


namespace Emily_beads_l58_58250

-- Define the conditions and question
theorem Emily_beads (n k : ℕ) (h1 : k = 4) (h2 : n = 5) : n * k = 20 := by
  -- Sorry: this is a placeholder for the actual proof
  sorry

end Emily_beads_l58_58250


namespace converse_proposition_inverse_proposition_contrapositive_proposition_l58_58764

theorem converse_proposition (x y : ℝ) : (xy = 0 → x^2 + y^2 = 0) = false :=
sorry

theorem inverse_proposition (x y : ℝ) : (x^2 + y^2 ≠ 0 → xy ≠ 0) = false :=
sorry

theorem contrapositive_proposition (x y : ℝ) : (xy ≠ 0 → x^2 + y^2 ≠ 0) = true :=
sorry

end converse_proposition_inverse_proposition_contrapositive_proposition_l58_58764


namespace suzie_reads_pages_hour_l58_58092

-- Declaration of the variables and conditions
variables (S : ℕ) -- S is the number of pages Suzie reads in an hour
variables (L : ℕ) -- L is the number of pages Liza reads in an hour

-- Conditions given in the problem
def reads_per_hour_Liza : L = 20 := sorry
def reads_more_pages : L * 3 = S * 3 + 15 := sorry

-- The statement we want to prove:
theorem suzie_reads_pages_hour : S = 15 :=
by
  -- Proof steps needed here (omitted due to the instruction)
  sorry

end suzie_reads_pages_hour_l58_58092


namespace customers_in_other_countries_l58_58610

-- Definitions for conditions
def total_customers : ℕ := 7422
def us_customers : ℕ := 723

-- Statement to prove
theorem customers_in_other_countries : total_customers - us_customers = 6699 :=
by
  sorry

end customers_in_other_countries_l58_58610


namespace simplify_expression_l58_58956

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ( (x^6 - 1) / (3 * x^3) )^2) = Real.sqrt (x^12 + 7 * x^6 + 1) / (3 * x^3) :=
by sorry

end simplify_expression_l58_58956


namespace find_a_from_function_l58_58004

theorem find_a_from_function (f : ℝ → ℝ) (h_f : ∀ x, f x = Real.sqrt (2 * x + 1)) (a : ℝ) (h_a : f a = 5) : a = 12 :=
by
  sorry

end find_a_from_function_l58_58004


namespace large_seat_capacity_l58_58692

-- Definition of conditions
def num_large_seats : ℕ := 7
def total_capacity_large_seats : ℕ := 84

-- Theorem to prove
theorem large_seat_capacity : total_capacity_large_seats / num_large_seats = 12 :=
by
  sorry

end large_seat_capacity_l58_58692


namespace circle_radius_l58_58670

/-
  Given:
  - The area of the circle x = π r^2
  - The circumference of the circle y = 2π r
  - The sum x + y = 72π

  Prove:
  The radius r = 6
-/
theorem circle_radius (r : ℝ) (x : ℝ) (y : ℝ) 
  (h₁ : x = π * r ^ 2) 
  (h₂ : y = 2 * π * r) 
  (h₃ : x + y = 72 * π) : 
  r = 6 := 
sorry

end circle_radius_l58_58670


namespace graph_of_equation_is_two_lines_l58_58518

theorem graph_of_equation_is_two_lines : 
  ∀ (x y : ℝ), (x - y)^2 = x^2 - y^2 ↔ (x = 0 ∨ y = 0) := 
by
  sorry

end graph_of_equation_is_two_lines_l58_58518


namespace gain_is_rs_150_l58_58520

noncomputable def P : ℝ := 5000
noncomputable def R_borrow : ℝ := 4
noncomputable def R_lend : ℝ := 7
noncomputable def T : ℝ := 2

noncomputable def SI (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (P * R * T) / 100

noncomputable def interest_paid := SI P R_borrow T
noncomputable def interest_earned := SI P R_lend T

noncomputable def gain_per_year : ℝ :=
  (interest_earned / T) - (interest_paid / T)

theorem gain_is_rs_150 : gain_per_year = 150 :=
by
  sorry

end gain_is_rs_150_l58_58520


namespace total_rainfall_in_2011_l58_58044

-- Define the given conditions
def avg_monthly_rainfall_2010 : ℝ := 36.8
def increase_2011 : ℝ := 3.5

-- Define the resulting average monthly rainfall in 2011
def avg_monthly_rainfall_2011 : ℝ := avg_monthly_rainfall_2010 + increase_2011

-- Calculate the total annual rainfall
def total_rainfall_2011 : ℝ := avg_monthly_rainfall_2011 * 12

-- State the proof problem
theorem total_rainfall_in_2011 :
  total_rainfall_2011 = 483.6 := by
  sorry

end total_rainfall_in_2011_l58_58044


namespace value_of_a10_l58_58495

/-- Define arithmetic sequence and properties -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = (n * (a 1 + a n) / 2)

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}
axiom arith_seq : arithmetic_sequence a d
axiom sum_formula : sum_of_first_n_terms a 5 S
axiom sum_condition : S 5 = 60
axiom term_condition : a 1 + a 2 + a 3 = a 4 + a 5

theorem value_of_a10 : a 10 = 26 :=
sorry

end value_of_a10_l58_58495


namespace range_of_x2_plus_y2_l58_58097

theorem range_of_x2_plus_y2 (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f x = -f (-x))
  (h_increasing : ∀ x y : ℝ, x < y → f x < f y)
  (x y : ℝ)
  (h_inequality : f (x^2 - 6 * x) + f (y^2 - 8 * y + 24) < 0) :
  16 < x^2 + y^2 ∧ x^2 + y^2 < 36 :=
sorry

end range_of_x2_plus_y2_l58_58097


namespace neg_exists_le_zero_iff_forall_gt_zero_l58_58506

variable (m : ℝ)

theorem neg_exists_le_zero_iff_forall_gt_zero :
  (¬ ∃ x : ℤ, (x:ℝ)^2 + 2 * x + m ≤ 0) ↔ ∀ x : ℤ, (x:ℝ)^2 + 2 * x + m > 0 :=
by
  sorry

end neg_exists_le_zero_iff_forall_gt_zero_l58_58506


namespace loan_difference_is_979_l58_58719

noncomputable def compounded_interest (P r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

noncomputable def loan_difference (P : ℝ) : ℝ :=
  let compounded_7_years := compounded_interest P 0.08 12 7
  let half_payment := compounded_7_years / 2
  let remaining_balance := compounded_interest half_payment 0.08 12 8
  let total_compounded := half_payment + remaining_balance
  let total_simple := simple_interest P 0.10 15
  abs (total_compounded - total_simple)

theorem loan_difference_is_979 : loan_difference 15000 = 979 := sorry

end loan_difference_is_979_l58_58719


namespace convert_spherical_to_rectangular_l58_58809

noncomputable def spherical_to_rectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin phi * Real.cos theta,
   rho * Real.sin phi * Real.sin theta,
   rho * Real.cos phi)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 4) = (2 * Real.sqrt 3, Real.sqrt 2, 2 * Real.sqrt 2) :=
by
  -- Define the spherical coordinates
  let rho := 4
  let theta := Real.pi / 6
  let phi := Real.pi / 4

  -- Calculate x, y, z using conversion formulas
  sorry

end convert_spherical_to_rectangular_l58_58809


namespace distance_x_intercepts_correct_l58_58980

noncomputable def distance_between_x_intercepts : ℝ :=
  let slope1 : ℝ := 4
  let slope2 : ℝ := -3
  let intercept_point : Prod ℝ ℝ := (8, 20)
  let line1 (x : ℝ) : ℝ := slope1 * (x - intercept_point.1) + intercept_point.2
  let line2 (x : ℝ) : ℝ := slope2 * (x - intercept_point.1) + intercept_point.2
  let x_intercept1 : ℝ := (0 - intercept_point.2) / slope1 + intercept_point.1
  let x_intercept2 : ℝ := (0 - intercept_point.2) / slope2 + intercept_point.1
  abs (x_intercept2 - x_intercept1)

theorem distance_x_intercepts_correct :
  distance_between_x_intercepts = 35 / 3 :=
sorry

end distance_x_intercepts_correct_l58_58980


namespace right_triangle_side_length_l58_58067

theorem right_triangle_side_length (x : ℝ) (hx : x > 0) (h_area : (1 / 2) * x * (3 * x) = 108) :
  x = 6 * Real.sqrt 2 :=
sorry

end right_triangle_side_length_l58_58067


namespace sample_size_l58_58088

-- Define the given conditions
def number_of_male_athletes : Nat := 42
def number_of_female_athletes : Nat := 30
def sampled_female_athletes : Nat := 5

-- Define the target total sample size
def total_sample_size (male_athletes female_athletes sample_females : Nat) : Nat :=
  sample_females * male_athletes / female_athletes + sample_females

-- State the theorem to prove
theorem sample_size (h1: number_of_male_athletes = 42) 
                    (h2: number_of_female_athletes = 30)
                    (h3: sampled_female_athletes = 5) :
  total_sample_size number_of_male_athletes number_of_female_athletes sampled_female_athletes = 12 :=
by
  -- Proof is omitted
  sorry

end sample_size_l58_58088


namespace mass_percentage_of_O_in_dichromate_l58_58993

noncomputable def molar_mass_Cr : ℝ := 52.00
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def molar_mass_Cr2O7_2_minus : ℝ := (2 * molar_mass_Cr) + (7 * molar_mass_O)

theorem mass_percentage_of_O_in_dichromate :
  (7 * molar_mass_O / molar_mass_Cr2O7_2_minus) * 100 = 51.85 := 
by
  sorry

end mass_percentage_of_O_in_dichromate_l58_58993


namespace last_two_non_zero_digits_of_75_factorial_l58_58846

theorem last_two_non_zero_digits_of_75_factorial : 
  ∃ (d : ℕ), d = 32 := sorry

end last_two_non_zero_digits_of_75_factorial_l58_58846


namespace rectangle_area_with_inscribed_circle_l58_58430

theorem rectangle_area_with_inscribed_circle (w h r : ℝ)
  (hw : ∀ O : ℝ × ℝ, dist O (w/2, h/2) = r)
  (hw_eq_h : w = h) :
  w * h = 2 * r^2 := 
by
  sorry

end rectangle_area_with_inscribed_circle_l58_58430


namespace plane_coloring_l58_58804

-- Define a type for colors to represent red and blue
inductive Color
| red
| blue

-- The main statement
theorem plane_coloring (x : ℝ) (h_pos : 0 < x) (coloring : ℝ × ℝ → Color) :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ coloring p1 = coloring p2 ∧ dist p1 p2 = x :=
sorry

end plane_coloring_l58_58804


namespace arccos_sqrt_half_l58_58563

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l58_58563


namespace sum_of_ages_3_years_ago_l58_58883

noncomputable def siblings_age_3_years_ago (R D S J : ℕ) : Prop :=
  R = D + 6 ∧
  D = S + 8 ∧
  J = R - 5 ∧
  R + 8 = 2 * (S + 8) ∧
  J + 10 = (D + 10) / 2 + 4 ∧
  S + 24 + J = 60 →
  (R - 3) + (D - 3) + (S - 3) + (J - 3) = 43

theorem sum_of_ages_3_years_ago (R D S J : ℕ) :
  siblings_age_3_years_ago R D S J :=
by
  intros
  sorry

end sum_of_ages_3_years_ago_l58_58883


namespace N_cannot_be_sum_of_three_squares_l58_58757

theorem N_cannot_be_sum_of_three_squares (K : ℕ) (L : ℕ) (N : ℕ) (h1 : N = 4^K * L) (h2 : L % 8 = 7) : ¬ ∃ (a b c : ℕ), N = a^2 + b^2 + c^2 := 
sorry

end N_cannot_be_sum_of_three_squares_l58_58757


namespace first_vessel_milk_water_l58_58747

variable (V : ℝ)

def vessel_ratio (v1 v2 : ℝ) : Prop := 
  v1 / v2 = 3 / 5

def vessel1_milk_water_ratio (milk water : ℝ) : Prop :=
  milk / water = 1 / 2

def vessel2_milk_water_ratio (milk water : ℝ) : Prop :=
  milk / water = 3 / 2

def mix_ratio (milk1 water1 milk2 water2 : ℝ) : Prop :=
  (milk1 + milk2) / (water1 + water2) = 1

theorem first_vessel_milk_water (V : ℝ) (v1 v2 : ℝ) (m1 w1 m2 w2 : ℝ)
  (hv : vessel_ratio v1 v2)
  (hv1 : vessel1_milk_water_ratio m1 w1)
  (hv2 : vessel2_milk_water_ratio m2 w2)
  (hmix : mix_ratio m1 w1 m2 w2) :
  vessel1_milk_water_ratio m1 w1 :=
  sorry

end first_vessel_milk_water_l58_58747


namespace max_value_of_f_l58_58251

noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem max_value_of_f : ∃ M : ℝ, M = 1 / 3 ∧ ∀ x : ℝ, f x ≤ M :=
by
  sorry

end max_value_of_f_l58_58251


namespace time_to_run_up_and_down_l58_58924

/-- Problem statement: Prove that the time it takes Vasya to run up and down a moving escalator 
which moves upwards is 468 seconds, given these conditions:
1. Vasya runs down twice as fast as he runs up.
2. When the escalator is not working, it takes Vasya 6 minutes to run up and down.
3. When the escalator is moving down, it takes Vasya 13.5 minutes to run up and down.
--/
theorem time_to_run_up_and_down (up_speed down_speed : ℝ) (escalator_speed : ℝ) 
  (h1 : down_speed = 2 * up_speed) 
  (h2 : (1 / up_speed + 1 / down_speed) = 6) 
  (h3 : (1 / (up_speed + escalator_speed) + 1 / (down_speed - escalator_speed)) = 13.5) : 
  (1 / (up_speed - escalator_speed) + 1 / (down_speed + escalator_speed)) * 60 = 468 := 
sorry

end time_to_run_up_and_down_l58_58924


namespace prove_math_problem_l58_58480

noncomputable def ellipse_foci : Prop := 
  ∃ (a b : ℝ), 
  a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ),
  (x^2 / a^2 + y^2 / b^2 = 1) → 
  a = 2 ∧ b^2 = 3)

noncomputable def intersect_and_rhombus : Prop :=
  ∃ (m : ℝ) (t : ℝ),
  (3 * m^2 + 4) > 0 ∧ 
  t = 1 / (3 * m^2 + 4) ∧ 
  0 < t ∧ t < 1 / 4

theorem prove_math_problem : ellipse_foci ∧ intersect_and_rhombus :=
by sorry

end prove_math_problem_l58_58480


namespace jeff_boxes_filled_l58_58624

def donuts_each_day : ℕ := 10
def days : ℕ := 12
def jeff_eats_per_day : ℕ := 1
def chris_eats : ℕ := 8
def donuts_per_box : ℕ := 10

theorem jeff_boxes_filled : 
  (donuts_each_day * days - jeff_eats_per_day * days - chris_eats) / donuts_per_box = 10 :=
by
  sorry

end jeff_boxes_filled_l58_58624


namespace first_day_exceeds_target_l58_58558

-- Definitions based on the conditions
def initial_count : ℕ := 5
def daily_growth_factor : ℕ := 3
def target_count : ℕ := 200

-- The proof problem in Lean
theorem first_day_exceeds_target : ∃ n : ℕ, 5 * 3 ^ n > 200 ∧ ∀ m < n, ¬ (5 * 3 ^ m > 200) :=
by
  sorry

end first_day_exceeds_target_l58_58558


namespace evaluate_expression_l58_58127

noncomputable def absoluteValue (x : ℝ) : ℝ := |x|

noncomputable def ceilingFunction (x : ℝ) : ℤ := ⌈x⌉

theorem evaluate_expression : ceilingFunction (absoluteValue (-52.7)) = 53 :=
by
  sorry

end evaluate_expression_l58_58127


namespace inequality_satisfaction_l58_58840

theorem inequality_satisfaction (a b : ℝ) (h : 0 < a ∧ a < b) : 
  a < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < b :=
by
  sorry

end inequality_satisfaction_l58_58840


namespace trains_meet_in_time_l58_58626

noncomputable def time_to_meet (length1 length2 distance_between speed1 speed2 : ℝ) : ℝ :=
  let speed1_mps := speed1 * 1000 / 3600
  let speed2_mps := speed2 * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := distance_between + length1 + length2
  total_distance / relative_speed

theorem trains_meet_in_time :
  time_to_meet 150 250 850 110 130 = 18.75 :=
by 
  -- here would go the proof steps, but since we are not required,
  sorry

end trains_meet_in_time_l58_58626


namespace isosceles_triangle_legs_length_l58_58184

-- Define the given conditions in Lean
def perimeter (L B: ℕ) : ℕ := 2 * L + B
def base_length : ℕ := 8
def given_perimeter : ℕ := 20

-- State the theorem to be proven
theorem isosceles_triangle_legs_length :
  ∃ (L : ℕ), perimeter L base_length = given_perimeter ∧ L = 6 :=
by
  sorry

end isosceles_triangle_legs_length_l58_58184


namespace minimize_theta_abs_theta_val_l58_58056

noncomputable def theta (k : ℤ) : ℝ := -11 / 4 * Real.pi + 2 * k * Real.pi

theorem minimize_theta_abs (k : ℤ) :
  ∃ θ : ℝ, (θ = -11 / 4 * Real.pi + 2 * k * Real.pi) ∧
           (∀ η : ℝ, (η = -11 / 4 * Real.pi + 2 * (k + 1) * Real.pi) →
             |θ| ≤ |η|) :=
  sorry

theorem theta_val : ∃ θ : ℝ, θ = -3 / 4 * Real.pi :=
  ⟨ -3 / 4 * Real.pi, rfl ⟩

end minimize_theta_abs_theta_val_l58_58056


namespace litter_collection_total_weight_l58_58294

/-- Gina collected 8 bags of litter: 5 bags of glass bottles weighing 7 pounds each and 3 bags of plastic waste weighing 4 pounds each. The 25 neighbors together collected 120 times as much glass as Gina and 80 times as much plastic as Gina. Prove that the total weight of all the collected litter is 5207 pounds. -/
theorem litter_collection_total_weight
  (glass_bags_gina : ℕ)
  (glass_weight_per_bag : ℕ)
  (plastic_bags_gina : ℕ)
  (plastic_weight_per_bag : ℕ)
  (neighbors_glass_multiplier : ℕ)
  (neighbors_plastic_multiplier : ℕ)
  (total_weight : ℕ)
  (h1 : glass_bags_gina = 5)
  (h2 : glass_weight_per_bag = 7)
  (h3 : plastic_bags_gina = 3)
  (h4 : plastic_weight_per_bag = 4)
  (h5 : neighbors_glass_multiplier = 120)
  (h6 : neighbors_plastic_multiplier = 80)
  (h_total_weight : total_weight = 5207) : total_weight = 
  glass_bags_gina * glass_weight_per_bag + 
  plastic_bags_gina * plastic_weight_per_bag + 
  neighbors_glass_multiplier * (glass_bags_gina * glass_weight_per_bag) + 
  neighbors_plastic_multiplier * (plastic_bags_gina * plastic_weight_per_bag) := 
by {
  /- Proof omitted -/
  sorry
}

end litter_collection_total_weight_l58_58294


namespace distance_to_base_is_42_l58_58913

theorem distance_to_base_is_42 (x : ℕ) (hx : 4 * x + 3 * (x + 3) = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) :
  4 * x = 36 ∨ 4 * x + 6 = 42 := 
by
  sorry

end distance_to_base_is_42_l58_58913


namespace smallest_y_value_l58_58310

-- Define the original equation
def original_eq (y : ℝ) := 3 * y^2 + 36 * y - 90 = y * (y + 18)

-- Define the problem statement
theorem smallest_y_value : ∃ (y : ℝ), original_eq y ∧ y = -15 :=
by
  sorry

end smallest_y_value_l58_58310


namespace correct_option_is_B_l58_58071

-- Define the Pythagorean theorem condition for right-angled triangles
def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Conditions given in the problem
def option_A : Prop := ¬is_right_angled_triangle 1 2 2
def option_B : Prop := is_right_angled_triangle 1 (Real.sqrt 3) 2
def option_C : Prop := ¬is_right_angled_triangle 4 5 6
def option_D : Prop := ¬is_right_angled_triangle 1 1 (Real.sqrt 3)

-- The formal proof problem statement
theorem correct_option_is_B : option_A ∧ option_B ∧ option_C ∧ option_D :=
by
  sorry

end correct_option_is_B_l58_58071


namespace number_of_small_companies_l58_58043

theorem number_of_small_companies
  (large_companies : ℕ)
  (medium_companies : ℕ)
  (inspected_companies : ℕ)
  (inspected_medium_companies : ℕ)
  (total_inspected_companies : ℕ)
  (small_companies : ℕ)
  (inspection_fraction : ℕ → ℚ)
  (proportion : inspection_fraction 20 = 1 / 4)
  (H1 : large_companies = 4)
  (H2 : medium_companies = 20)
  (H3 : inspected_medium_companies = 5)
  (H4 : total_inspected_companies = 40)
  (H5 : inspected_companies = total_inspected_companies - large_companies - inspected_medium_companies)
  (H6 : small_companies = inspected_companies * 4)
  (correct_result : small_companies = 136) :
  small_companies = 136 :=
by sorry

end number_of_small_companies_l58_58043


namespace first_applicant_earnings_l58_58172

def first_applicant_salary : ℕ := 42000
def first_applicant_training_cost_per_month : ℕ := 1200
def first_applicant_training_months : ℕ := 3
def second_applicant_salary : ℕ := 45000
def second_applicant_bonus_percentage : ℕ := 1
def company_earnings_from_second_applicant : ℕ := 92000
def earnings_difference : ℕ := 850

theorem first_applicant_earnings 
  (salary1 : first_applicant_salary = 42000)
  (train_cost_per_month : first_applicant_training_cost_per_month = 1200)
  (train_months : first_applicant_training_months = 3)
  (salary2 : second_applicant_salary = 45000)
  (bonus_percentage : second_applicant_bonus_percentage = 1)
  (earnings2 : company_earnings_from_second_applicant = 92000)
  (earning_diff : earnings_difference = 850) :
  (company_earnings_from_second_applicant - (second_applicant_salary + (second_applicant_salary * second_applicant_bonus_percentage / 100)) - earnings_difference) = 45700 := 
by 
  sorry

end first_applicant_earnings_l58_58172


namespace minimize_AB_l58_58314

-- Definition of the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 2 * y - 3 = 0

-- Definition of the point P
def P : ℝ × ℝ := (-1, 2)

-- Definition of the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- The goal is to prove that line_l is the line through P minimizing |AB|
theorem minimize_AB : 
  ∀ l : ℝ → ℝ → Prop, 
  (∀ x y, l x y → (∃ a b, circleC a b ∧ l a b ∧ circleC x y ∧ l x y ∧ (x ≠ a ∨ y ≠ b)) → False) 
  → l = line_l :=
by
  sorry

end minimize_AB_l58_58314


namespace min_value_proof_l58_58784

noncomputable def min_value (x y : ℝ) : ℝ :=
x^3 + y^3 - x^2 - y^2

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 15 * x - y = 22) : 
min_value x y ≥ 1 := by
  sorry

end min_value_proof_l58_58784


namespace john_less_than_anna_l58_58733

theorem john_less_than_anna (J A L T : ℕ) (h1 : A = 50) (h2: L = 3) (h3: T = 82) (h4: T + L = A + J) : A - J = 15 :=
by
  sorry

end john_less_than_anna_l58_58733


namespace opposite_face_A_is_E_l58_58911

-- Axiomatically defining the basic conditions from the problem statement.

-- We have six labels for the faces of a net
inductive Face : Type
| A | B | C | D | E | F

open Face

-- Define the adjacency relation
def adjacent (x y : Face) : Prop :=
  (x = A ∧ y = B) ∨ (x = A ∧ y = D) ∨ (x = B ∧ y = A) ∨ (x = D ∧ y = A)

-- Define the "not directly attached" relationship
def not_adjacent (x y : Face) : Prop :=
  ¬adjacent x y

-- Given the conditions in the problem statement
axiom condition1 : adjacent A B
axiom condition2 : adjacent A D
axiom condition3 : not_adjacent A E

-- The proof objective is to show that E is the face opposite to A
theorem opposite_face_A_is_E : ∃ (F : Face), 
  (∀ x : Face, adjacent A x ∨ not_adjacent A x) → (∀ y : Face, adjacent A y ↔ y ≠ E) → E = F :=
sorry

end opposite_face_A_is_E_l58_58911


namespace basketball_court_length_difference_l58_58131

theorem basketball_court_length_difference :
  ∃ (l w : ℕ), l = 31 ∧ w = 17 ∧ l - w = 14 := by
  sorry

end basketball_court_length_difference_l58_58131


namespace number_of_questionnaires_drawn_from_15_to_16_is_120_l58_58445

variable (x : ℕ)
variable (H1 : 120 + 180 + 240 + x = 900)
variable (H2 : 60 = (bit0 90) / 180)
variable (H3 : (bit0 (bit0 (bit0 15))) = (bit0 (bit0 (bit0 15))) * (900 / 300))

theorem number_of_questionnaires_drawn_from_15_to_16_is_120 :
  ((900 - 120 - 180 - 240) * (300 / 900)) = 120 :=
sorry

end number_of_questionnaires_drawn_from_15_to_16_is_120_l58_58445


namespace find_pairs_l58_58752

/-
Define the conditions:
1. The number of three-digit phone numbers consisting of only odd digits.
2. The number of three-digit phone numbers consisting of only even digits excluding 0.
3. Revenue difference is given by a specific equation.
4. \(X\) and \(Y\) are integers less than 250.
-/
def N₁ : ℕ := 5 * 5 * 5  -- Number of combinations with odd digits (1, 3, 5, 7, 9)
def N₂ : ℕ := 4 * 4 * 4  -- Number of combinations with even digits (2, 4, 6, 8)

-- Main theorem: finding pairs (X, Y) that satisfy the given conditions.
theorem find_pairs (X Y : ℕ) (hX : X < 250) (hY : Y < 250) :
  N₁ * X - N₂ * Y = 5 ↔ (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) := 
by {
  sorry
}

end find_pairs_l58_58752


namespace quarters_for_soda_l58_58750

def quarters_for_chips := 4
def total_dollars := 4

theorem quarters_for_soda :
  (total_dollars * 4) - quarters_for_chips = 12 :=
by
  sorry

end quarters_for_soda_l58_58750


namespace project_total_hours_l58_58944

def pat_time (k : ℕ) : ℕ := 2 * k
def mark_time (k : ℕ) : ℕ := k + 120

theorem project_total_hours (k : ℕ) (H1 : 3 * 2 * k = k + 120) :
  k + pat_time k + mark_time k = 216 :=
by
  sorry

end project_total_hours_l58_58944


namespace intersection_of_A_and_B_l58_58192

open Set

def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}
def B : Set ℝ := {y | 0 ≤ y ∧ y < 4}

theorem intersection_of_A_and_B : A ∩ B = {z | 2 ≤ z ∧ z < 4} :=
by
  sorry

end intersection_of_A_and_B_l58_58192


namespace seashells_count_l58_58775

theorem seashells_count : 18 + 47 = 65 := by
  sorry

end seashells_count_l58_58775


namespace unique_root_iff_l58_58233

def has_unique_solution (a : ℝ) : Prop :=
  ∃ (x : ℝ), ∀ (y : ℝ), (a * y^2 + 2 * y - 1 = 0 ↔ y = x)

theorem unique_root_iff (a : ℝ) : has_unique_solution a ↔ (a = 0 ∨ a = 1) := 
sorry

end unique_root_iff_l58_58233


namespace domain_of_f_l58_58019

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (Real.log x / Real.log 2 - 1))

theorem domain_of_f :
  {x : ℝ | x > 2} = {x : ℝ | x > 0 ∧ Real.log x / Real.log 2 - 1 > 0} := 
by
  sorry

end domain_of_f_l58_58019


namespace find_value_of_k_l58_58266

theorem find_value_of_k (k x : ℝ) 
  (h : 1 / (4 - x ^ 2) + 2 = k / (x - 2)) : 
  k = -1 / 4 :=
by
  sorry

end find_value_of_k_l58_58266


namespace friends_contribution_l58_58274

theorem friends_contribution (x : ℝ) 
  (h1 : 4 * (x - 5) = 0.75 * 4 * x) : 
  0.75 * 4 * x = 60 :=
by 
  sorry

end friends_contribution_l58_58274


namespace range_of_a_l58_58317

theorem range_of_a (a b : ℝ) (h1 : 0 ≤ a - b ∧ a - b ≤ 1) (h2 : 1 ≤ a + b ∧ a + b ≤ 4) : 
  1 / 2 ≤ a ∧ a ≤ 5 / 2 := 
sorry

end range_of_a_l58_58317


namespace range_of_a_l58_58977

theorem range_of_a (a : ℝ) : 
(∀ x : ℝ, |x - 1| + |x - 3| > a ^ 2 - 2 * a - 1) ↔ -1 < a ∧ a < 3 := 
sorry

end range_of_a_l58_58977


namespace pupils_like_both_l58_58469

theorem pupils_like_both (total_pupils : ℕ) (likes_pizza : ℕ) (likes_burgers : ℕ)
  (total := 200) (P := 125) (B := 115) :
  (P + B - total_pupils) = 40 :=
by
  sorry

end pupils_like_both_l58_58469


namespace range_of_m_l58_58567

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x then (1 / 3)^(-x) - 2 
  else 2 * Real.log x / Real.log 3

theorem range_of_m :
  {m : ℝ | f m > 1} = {m : ℝ | m < -Real.sqrt 3} ∪ {m : ℝ | 1 < m} :=
by
  sorry

end range_of_m_l58_58567


namespace Y_minus_X_eq_92_l58_58611

def arithmetic_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def X : ℕ := arithmetic_sum 10 2 46
def Y : ℕ := arithmetic_sum 12 2 46

theorem Y_minus_X_eq_92 : Y - X = 92 := by
  sorry

end Y_minus_X_eq_92_l58_58611


namespace intersection_slopes_l58_58618

theorem intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ 
  m ∈ Set.Iic (-Real.sqrt (4 / 41)) ∨ m ∈ Set.Ici (Real.sqrt (4 / 41)) := 
sorry

end intersection_slopes_l58_58618


namespace vehicle_distance_traveled_l58_58377

theorem vehicle_distance_traveled 
  (perimeter_back : ℕ) (perimeter_front : ℕ) (revolution_difference : ℕ)
  (R : ℕ)
  (h1 : perimeter_back = 9)
  (h2 : perimeter_front = 7)
  (h3 : revolution_difference = 10)
  (h4 : (R * perimeter_back) = ((R + revolution_difference) * perimeter_front)) :
  (R * perimeter_back) = 315 :=
by
  -- Prove that the distance traveled by the vehicle is 315 feet
  -- given the conditions and the hypothesis.
  sorry

end vehicle_distance_traveled_l58_58377


namespace initial_number_of_machines_l58_58271

theorem initial_number_of_machines
  (x : ℕ)
  (h1 : x * 270 = 1080)
  (h2 : 20 * 3600 = 144000)
  (h3 : ∀ y, (20 * y * 4 = 3600) → y = 45) :
  x = 6 :=
by
  sorry

end initial_number_of_machines_l58_58271


namespace diagonals_of_hexadecagon_l58_58321

-- Define the function to calculate number of diagonals in a convex polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- State the theorem for the number of diagonals in a convex hexadecagon
theorem diagonals_of_hexadecagon : num_diagonals 16 = 104 := by
  -- sorry is used to indicate the proof is skipped
  sorry

end diagonals_of_hexadecagon_l58_58321


namespace rectangle_area_l58_58902

theorem rectangle_area
  (L B : ℕ)
  (h1 : L - B = 23)
  (h2 : 2 * L + 2 * B = 186) : L * B = 2030 :=
sorry

end rectangle_area_l58_58902


namespace solution_set_correct_l58_58674

theorem solution_set_correct (a b c : ℝ) (h : a < 0) (h1 : ∀ x, (ax^2 + bx + c < 0) ↔ ((x < 1) ∨ (x > 3))) :
  ∀ x, (cx^2 + bx + a > 0) ↔ (1 / 3 < x ∧ x < 1) :=
sorry

end solution_set_correct_l58_58674


namespace range_of_a_l58_58592

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (4 * (x - 1) > 3 * x - 1) ∧ (5 * x > 3 * x + 2 * a) ↔ (x > 3)) ↔ (a ≤ 3) :=
by
  sorry

end range_of_a_l58_58592


namespace bob_age_is_725_l58_58894

theorem bob_age_is_725 (n : ℕ) (h1 : ∃ k : ℤ, n - 3 = k^2) (h2 : ∃ j : ℤ, n + 4 = j^3) : n = 725 :=
sorry

end bob_age_is_725_l58_58894


namespace total_amount_in_account_after_two_years_l58_58689

-- Initial definitions based on conditions in the problem
def initial_investment : ℝ := 76800
def annual_interest_rate : ℝ := 0.125
def annual_contribution : ℝ := 5000

-- Function to calculate amount after n years with annual contributions
def total_amount_after_years (P : ℝ) (r : ℝ) (A : ℝ) (n : ℕ) : ℝ :=
  let rec helper (P : ℝ) (n : ℕ) :=
    if n = 0 then P
    else 
      let previous_amount := helper P (n - 1)
      (previous_amount * (1 + r) + A)
  helper P n

-- Theorem to prove the final total amount after 2 years
theorem total_amount_in_account_after_two_years :
  total_amount_after_years initial_investment annual_interest_rate annual_contribution 2 = 107825 :=
  by 
  -- proof goes here
  sorry

end total_amount_in_account_after_two_years_l58_58689


namespace factorization_6x2_minus_24x_plus_18_l58_58205

theorem factorization_6x2_minus_24x_plus_18 :
    ∀ x : ℝ, 6 * x^2 - 24 * x + 18 = 6 * (x - 1) * (x - 3) :=
by
  intro x
  sorry

end factorization_6x2_minus_24x_plus_18_l58_58205


namespace calculation_correct_l58_58600

def expression : ℝ := 200 * 375 * 0.0375 * 5

theorem calculation_correct : expression = 14062.5 := 
by
  sorry

end calculation_correct_l58_58600


namespace Ivan_pays_1_point_5_times_more_l58_58867

theorem Ivan_pays_1_point_5_times_more (x y : ℝ) (h : x = 2 * y) : 1.5 * (0.6 * x + 0.8 * y) = x + y :=
by
  sorry

end Ivan_pays_1_point_5_times_more_l58_58867


namespace parabola_properties_l58_58833

theorem parabola_properties (a b c: ℝ) (ha : a ≠ 0) (hc : c > 1) (h1 : 4 * a + 2 * b + c = 0) (h2 : -b / (2 * a) = 1/2):
  a * b * c < 0 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = a ∧ a * x2^2 + b * x2 + c = a) ∧ a < -1/2 :=
by {
    sorry
}

end parabola_properties_l58_58833


namespace stewart_farm_sheep_l58_58703

theorem stewart_farm_sheep (S H : ℕ)
  (h1 : S / H = 2 / 7)
  (h2 : H * 230 = 12880) :
  S = 16 :=
by sorry

end stewart_farm_sheep_l58_58703


namespace diophantine_solution_l58_58981

theorem diophantine_solution :
  ∃ (x y k : ℤ), 1990 * x - 173 * y = 11 ∧ x = -22 + 173 * k ∧ y = 253 - 1990 * k :=
by {
  sorry
}

end diophantine_solution_l58_58981


namespace f_2017_eq_2018_l58_58819

def f (n : ℕ) : ℕ := sorry

theorem f_2017_eq_2018 (f : ℕ → ℕ) (h1 : ∀ n, f (f n) + f n = 2 * n + 3) (h2 : f 0 = 1) : f 2017 = 2018 :=
sorry

end f_2017_eq_2018_l58_58819


namespace duration_of_loan_l58_58772

namespace SimpleInterest

variables (P SI R : ℝ) (T : ℝ)

-- Defining the conditions
def principal := P = 1500
def simple_interest := SI = 735
def rate := R = 7 / 100

-- The question: Prove the duration (T) of the loan
theorem duration_of_loan (hP : principal P) (hSI : simple_interest SI) (hR : rate R) :
  T = 7 :=
sorry

end SimpleInterest

end duration_of_loan_l58_58772


namespace tallest_building_model_height_l58_58588

def height_campus : ℝ := 120
def volume_campus : ℝ := 30000
def volume_model : ℝ := 0.03
def height_model : ℝ := 1.2

theorem tallest_building_model_height :
  (volume_campus / volume_model)^(1/3) = (height_campus / height_model) :=
by
  sorry

end tallest_building_model_height_l58_58588


namespace fraction_decomposition_roots_sum_l58_58651

theorem fraction_decomposition_roots_sum :
  ∀ (p q r A B C : ℝ),
  p ≠ q → p ≠ r → q ≠ r →
  (∀ (s : ℝ), s ≠ p → s ≠ q → s ≠ r →
          1 / (s^3 - 15 * s^2 + 50 * s - 56) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 225 :=
by
  intros p q r A B C hpq hpr hqr hDecomp
  -- Skip proof
  sorry

end fraction_decomposition_roots_sum_l58_58651


namespace clock_malfunction_fraction_correct_l58_58510

theorem clock_malfunction_fraction_correct : 
  let hours_total := 24
  let hours_incorrect := 6
  let minutes_total := 60
  let minutes_incorrect := 6
  let fraction_correct_hours := (hours_total - hours_incorrect) / hours_total
  let fraction_correct_minutes := (minutes_total - minutes_incorrect) / minutes_total
  (fraction_correct_hours * fraction_correct_minutes) = 27 / 40
:= 
by
  sorry

end clock_malfunction_fraction_correct_l58_58510


namespace amaya_movie_watching_time_l58_58667

theorem amaya_movie_watching_time :
  let t1 := 30 + 5
  let t2 := 20 + 7
  let t3 := 10 + 12
  let t4 := 15 + 8
  let t5 := 25 + 15
  let t6 := 15 + 10
  t1 + t2 + t3 + t4 + t5 + t6 = 172 :=
by
  sorry

end amaya_movie_watching_time_l58_58667


namespace length_of_scale_parts_l58_58754

theorem length_of_scale_parts (total_length_ft : ℕ) (remaining_inches : ℕ) (parts : ℕ) : 
  total_length_ft = 6 ∧ remaining_inches = 8 ∧ parts = 2 →
  ∃ ft inches, ft = 3 ∧ inches = 4 :=
by
  sorry

end length_of_scale_parts_l58_58754


namespace probability_blue_face_up_l58_58289

-- Definitions of the conditions
def dodecahedron_faces : ℕ := 12
def blue_faces : ℕ := 10
def red_faces : ℕ := 2

-- Expected probability
def probability_blue_face : ℚ := 5 / 6

-- Theorem to prove the probability of rolling a blue face on a dodecahedron
theorem probability_blue_face_up (total_faces blue_count red_count : ℕ)
    (h1 : total_faces = dodecahedron_faces)
    (h2 : blue_count = blue_faces)
    (h3 : red_count = red_faces) :
  blue_count / total_faces = probability_blue_face :=
by sorry

end probability_blue_face_up_l58_58289


namespace remainder_of_repeated_23_l58_58959

theorem remainder_of_repeated_23 {n : ℤ} (n : ℤ) (hn : n = 23 * 10^(2*23)) : 
  (n % 32) = 19 :=
sorry

end remainder_of_repeated_23_l58_58959


namespace find_j_value_l58_58390

variable {R : Type*} [LinearOrderedField R]

-- Definitions based on conditions
def polynomial_has_four_distinct_real_roots_in_arithmetic_progression
(p : Polynomial R) : Prop :=
∃ a d : R, p.roots.toFinset = {a, a + d, a + 2*d, a + 3*d} ∧
a ≠ a + d ∧ a ≠ a + 2*d ∧ a ≠ a + 3*d ∧ a + d ≠ a + 2*d ∧
a + d ≠ a + 3*d ∧ a + 2*d ≠ a + 3*d

-- The main theorem statement
theorem find_j_value (k : R) 
  (h : polynomial_has_four_distinct_real_roots_in_arithmetic_progression 
  (Polynomial.X^4 + Polynomial.C j * Polynomial.X^2 + Polynomial.C k * Polynomial.X + Polynomial.C 900)) :
  j = -900 :=
sorry

end find_j_value_l58_58390


namespace compute_expression_l58_58124

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 + 1012^2 - 988^2 = 68000 := 
by
  sorry

end compute_expression_l58_58124


namespace third_offense_percentage_increase_l58_58962

theorem third_offense_percentage_increase 
    (base_per_5000 : ℕ)
    (goods_stolen : ℕ)
    (additional_years : ℕ)
    (total_sentence : ℕ) :
    base_per_5000 = 1 →
    goods_stolen = 40000 →
    additional_years = 2 →
    total_sentence = 12 →
    100 * (total_sentence - additional_years - goods_stolen / 5000) / (goods_stolen / 5000) = 25 :=
by
  intros h_base h_goods h_additional h_total
  sorry

end third_offense_percentage_increase_l58_58962


namespace tailor_time_calculation_l58_58263

-- Define the basic quantities and their relationships
def time_ratio_shirt : ℕ := 1
def time_ratio_pants : ℕ := 2
def time_ratio_jacket : ℕ := 3

-- Given conditions
def shirts_made := 2
def pants_made := 3
def jackets_made := 4
def total_time_initial : ℝ := 10

-- Unknown time per shirt
noncomputable def time_per_shirt := total_time_initial / (shirts_made * time_ratio_shirt 
  + pants_made * time_ratio_pants 
  + jackets_made * time_ratio_jacket)

-- Future quantities
def future_shirts := 14
def future_pants := 10
def future_jackets := 2

-- Calculate the future total time required
noncomputable def future_time_required := (future_shirts * time_ratio_shirt 
  + future_pants * time_ratio_pants 
  + future_jackets * time_ratio_jacket) * time_per_shirt

-- State the theorem to prove
theorem tailor_time_calculation : future_time_required = 20 := by
  sorry

end tailor_time_calculation_l58_58263


namespace weird_fraction_implies_weird_power_fraction_l58_58232

theorem weird_fraction_implies_weird_power_fraction 
  (a b c : ℝ) (k : ℕ) 
  (h1 : (1/a) + (1/b) + (1/c) = (1/(a + b + c))) 
  (h2 : Odd k) : 
  (1 / (a^k) + 1 / (b^k) + 1 / (c^k) = 1 / (a^k + b^k + c^k)) := 
by 
  sorry

end weird_fraction_implies_weird_power_fraction_l58_58232


namespace train_problem_l58_58850

variables (x : ℝ) (p q : ℝ)
variables (speed_p speed_q : ℝ) (dist_diff : ℝ)

theorem train_problem
  (speed_p : speed_p = 50)
  (speed_q : speed_q = 40)
  (dist_diff : ∀ x, x = 500 → p = 50 * x ∧ q = 40 * (500 - 100)) :
  p + q = 900 :=
by
sorry

end train_problem_l58_58850


namespace range_of_a1_l58_58441

theorem range_of_a1 (a : ℕ → ℕ) (S : ℕ → ℕ) (h_seq : ∀ n, 12 * S n = 4 * a (n + 1) + 5^n - 13)
  (h_S4 : ∀ n, S n ≤ S 4):
  13 / 48 ≤ a 1 ∧ a 1 ≤ 59 / 64 :=
sorry

end range_of_a1_l58_58441


namespace projectile_height_reaches_45_at_t_0_5_l58_58888

noncomputable def quadratic (a b c : ℝ) : ℝ → ℝ :=
  λ t => a * t^2 + b * t + c

theorem projectile_height_reaches_45_at_t_0_5 :
  ∃ t : ℝ, quadratic (-16) 98.5 (-45) t = 45 ∧ 0 ≤ t ∧ t = 0.5 :=
by
  sorry

end projectile_height_reaches_45_at_t_0_5_l58_58888


namespace diameter_of_circular_ground_l58_58688

noncomputable def radius_of_garden_condition (area_garden : ℝ) (broad_garden : ℝ) : ℝ :=
  let pi_val := Real.pi
  (area_garden / pi_val - broad_garden * broad_garden) / (2 * broad_garden)

-- Given conditions
variable (area_garden : ℝ := 226.19467105846502)
variable (broad_garden : ℝ := 2)

-- Goal to prove: diameter of the circular ground is 34 metres
theorem diameter_of_circular_ground : 2 * radius_of_garden_condition area_garden broad_garden = 34 :=
  sorry

end diameter_of_circular_ground_l58_58688


namespace intersection_M_N_l58_58891

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l58_58891


namespace f_2008_eq_zero_l58_58342

noncomputable def f : ℝ → ℝ := sorry

-- f is odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- f satisfies f(x + 2) = -f(x)
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x

theorem f_2008_eq_zero : f 2008 = 0 :=
by
  sorry

end f_2008_eq_zero_l58_58342


namespace length_of_purple_part_l58_58653

variables (P : ℝ) (black : ℝ) (blue : ℝ) (total_len : ℝ)

-- The conditions
def conditions := 
  black = 0.5 ∧ 
  blue = 2 ∧ 
  total_len = 4 ∧ 
  P + black + blue = total_len

-- The proof problem statement
theorem length_of_purple_part (h : conditions P 0.5 2 4) : P = 1.5 :=
sorry

end length_of_purple_part_l58_58653


namespace sum_of_first_3n_terms_l58_58727

def arithmetic_geometric_sequence (n : ℕ) (s : ℕ → ℕ) :=
  (s n = 10) ∧ (s (2 * n) = 30)

theorem sum_of_first_3n_terms (n : ℕ) (s : ℕ → ℕ) :
  arithmetic_geometric_sequence n s → s (3 * n) = 70 :=
by
  intro h
  sorry

end sum_of_first_3n_terms_l58_58727


namespace import_tax_paid_l58_58657

theorem import_tax_paid (total_value excess_value tax_rate tax_paid : ℝ)
  (h₁ : total_value = 2590)
  (h₂ : excess_value = total_value - 1000)
  (h₃ : tax_rate = 0.07)
  (h₄ : tax_paid = excess_value * tax_rate) : 
  tax_paid = 111.30 := by
  -- variables
  sorry

end import_tax_paid_l58_58657


namespace smallest_solution_l58_58826

theorem smallest_solution (x : ℝ) :
  (∃ x, (3 * x) / (x - 3) + (3 * x^2 - 36) / (x + 3) = 15) →
  x = -1 := 
sorry

end smallest_solution_l58_58826


namespace number_of_white_balls_l58_58513

-- Definition of conditions
def red_balls : ℕ := 4
def frequency_of_red_balls : ℝ := 0.25
def total_balls (white_balls : ℕ) : ℕ := red_balls + white_balls

-- Proving the number of white balls given the conditions
theorem number_of_white_balls (x : ℕ) :
  (red_balls : ℝ) / total_balls x = frequency_of_red_balls → x = 12 :=
by
  sorry

end number_of_white_balls_l58_58513


namespace no_descending_digits_multiple_of_111_l58_58865

theorem no_descending_digits_multiple_of_111 (n : ℕ) (h_desc : (∀ i j, i < j → (n % 10 ^ (i + 1)) / 10 ^ i ≥ (n % 10 ^ (j + 1)) / 10 ^ j)) :
  ¬(111 ∣ n) :=
sorry

end no_descending_digits_multiple_of_111_l58_58865


namespace line_through_points_l58_58072

theorem line_through_points (a b : ℝ) (h₁ : 1 = a * 3 + b) (h₂ : 13 = a * 7 + b) : a - b = 11 := 
  sorry

end line_through_points_l58_58072


namespace tree_ratio_l58_58502

theorem tree_ratio (A P C : ℕ) 
  (hA : A = 58)
  (hP : P = 3 * A)
  (hC : C = 5 * P) : (A, P, C) = (1, 3 * 58, 15 * 58) :=
by
  sorry

end tree_ratio_l58_58502


namespace solve_equation_l58_58642

theorem solve_equation (x : ℝ) : x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by
  sorry

end solve_equation_l58_58642


namespace friend_owns_10_bicycles_l58_58861

variable (ignatius_bicycles : ℕ)
variable (tires_per_bicycle : ℕ)
variable (friend_tires_ratio : ℕ)
variable (unicycle_tires : ℕ)
variable (tricycle_tires : ℕ)

def friend_bicycles (friend_bicycle_tires : ℕ) : ℕ :=
  friend_bicycle_tires / tires_per_bicycle

theorem friend_owns_10_bicycles :
  ignatius_bicycles = 4 →
  tires_per_bicycle = 2 →
  friend_tires_ratio = 3 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_bicycles (friend_tires_ratio * (ignatius_bicycles * tires_per_bicycle) - unicycle_tires - tricycle_tires) = 10 :=
by
  intros
  -- Proof goes here
  sorry

end friend_owns_10_bicycles_l58_58861


namespace least_divisor_for_perfect_square_l58_58707

theorem least_divisor_for_perfect_square : 
  ∃ d : ℕ, (∀ n : ℕ, n > 0 → 16800 / d = n * n) ∧ d = 21 := 
sorry

end least_divisor_for_perfect_square_l58_58707


namespace find_nonnegative_solutions_l58_58594

theorem find_nonnegative_solutions :
  ∀ (x y z : ℕ), 5^x + 7^y = 2^z ↔ (x = 0 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 5) :=
by
  sorry

end find_nonnegative_solutions_l58_58594


namespace inheritance_division_l58_58119

variables {M P Q R : ℝ} {p q r : ℕ}

theorem inheritance_division (hP : P < 99 * (p : ℝ))
                             (hR : R > 10000 * (r : ℝ))
                             (hM : M = P + Q + R)
                             (hRichPoor : R ≥ P) : 
                             R ≥ 100 * P := 
sorry

end inheritance_division_l58_58119


namespace john_free_throws_l58_58105

theorem john_free_throws 
  (hit_rate : ℝ) 
  (shots_per_foul : ℕ) 
  (fouls_per_game : ℕ) 
  (total_games : ℕ) 
  (percentage_played : ℝ) 
  : hit_rate = 0.7 → 
    shots_per_foul = 2 → 
    fouls_per_game = 5 → 
    total_games = 20 → 
    percentage_played = 0.8 → 
    ∃ (total_free_throws : ℕ), total_free_throws = 112 := 
by
  intros
  sorry

end john_free_throws_l58_58105


namespace symmetrical_point_correct_l58_58878

variables (x₁ y₁ : ℝ)

def symmetrical_point_x_axis (x y : ℝ) : ℝ × ℝ :=
(x, -y)

theorem symmetrical_point_correct : symmetrical_point_x_axis 3 2 = (3, -2) :=
by
  -- This is where we would provide the proof
  sorry

end symmetrical_point_correct_l58_58878


namespace rotated_line_x_intercept_l58_58684

theorem rotated_line_x_intercept (x y : ℝ) :
  (∃ (k : ℝ), y = (3 * Real.sqrt 3 + 5) / (2 * Real.sqrt 3) * x) →
  (∃ y : ℝ, 3 * x - 5 * y + 40 = 0) →
  (∃ (x_intercept : ℝ), x_intercept = 0) := 
by
  sorry

end rotated_line_x_intercept_l58_58684


namespace tables_in_conference_hall_l58_58129

theorem tables_in_conference_hall (c t : ℕ) 
  (h1 : c = 8 * t) 
  (h2 : 4 * c + 4 * t = 648) : 
  t = 18 :=
by sorry

end tables_in_conference_hall_l58_58129


namespace double_luckiness_l58_58181

variable (oats marshmallows : ℕ)
variable (initial_luckiness doubled_luckiness : ℚ)

def luckiness (marshmallows total_pieces : ℕ) : ℚ :=
  marshmallows / total_pieces

theorem double_luckiness (h_oats : oats = 90) (h_marshmallows : marshmallows = 9)
  (h_initial : initial_luckiness = luckiness marshmallows (oats + marshmallows))
  (h_doubled : doubled_luckiness = 2 * initial_luckiness) :
  ∃ x : ℕ, doubled_luckiness = luckiness (marshmallows + x) (oats + marshmallows + x) :=
  sorry

#check double_luckiness

end double_luckiness_l58_58181


namespace number_of_true_statements_l58_58548

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m
def is_odd (n : ℕ) : Prop := ∃ m : ℕ, n = 2 * m + 1
def is_even (n : ℕ) : Prop := ∃ m : ℕ, n = 2 * m

theorem number_of_true_statements : 3 = (ite ((∀ p q : ℕ, is_prime p → is_prime q → is_prime (p * q)) = false) 0 1) +
                                     (ite ((∀ a b : ℕ, is_square a → is_square b → is_square (a * b)) = true) 1 0) +
                                     (ite ((∀ x y : ℕ, is_odd x → is_odd y → is_odd (x * y)) = true) 1 0) +
                                     (ite ((∀ u v : ℕ, is_even u → is_even v → is_even (u * v)) = true) 1 0) :=
by
  sorry

end number_of_true_statements_l58_58548


namespace quadratic_cubic_inequalities_l58_58839

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (x : ℝ) : ℝ := -x ^ 3 + 5 * x - 3

variable (x : ℝ)

theorem quadratic_cubic_inequalities (h : 0 < x) : 
  (f x ≥ 2 * x - 1) ∧ (g x ≤ 2 * x - 1) := 
sorry

end quadratic_cubic_inequalities_l58_58839


namespace number_is_43_l58_58330

theorem number_is_43 (m : ℕ) : (m > 30 ∧ m < 50) ∧ Nat.Prime m ∧ m % 12 = 7 ↔ m = 43 :=
by
  sorry

end number_is_43_l58_58330


namespace salmon_total_l58_58591

def num_male : ℕ := 712261
def num_female : ℕ := 259378
def num_total : ℕ := 971639

theorem salmon_total :
  num_male + num_female = num_total :=
by
  -- proof will be provided here
  sorry

end salmon_total_l58_58591


namespace sum_mod_9_l58_58770

theorem sum_mod_9 : (7155 + 7156 + 7157 + 7158 + 7159) % 9 = 1 :=
by sorry

end sum_mod_9_l58_58770


namespace joshua_needs_more_cents_l58_58547

-- Definitions of inputs
def cost_of_pen_dollars : ℕ := 6
def joshua_money_dollars : ℕ := 5
def borrowed_cents : ℕ := 68

-- Convert dollar amounts to cents
def dollar_to_cents (d : ℕ) : ℕ := d * 100

def cost_of_pen_cents := dollar_to_cents cost_of_pen_dollars
def joshua_money_cents := dollar_to_cents joshua_money_dollars

-- Total amount Joshua has in cents
def total_cents := joshua_money_cents + borrowed_cents

-- Calculation of the required amount
def needed_cents := cost_of_pen_cents - total_cents

theorem joshua_needs_more_cents : needed_cents = 32 := by 
  sorry

end joshua_needs_more_cents_l58_58547


namespace print_colored_pages_l58_58440

theorem print_colored_pages (cost_per_page : ℕ) (dollars : ℕ) (conversion_rate : ℕ) 
    (h_cost : cost_per_page = 4) (h_dollars : dollars = 30) (h_conversion : conversion_rate = 100) :
    (dollars * conversion_rate) / cost_per_page = 750 := 
by
  sorry

end print_colored_pages_l58_58440


namespace abs_inequality_solution_set_l58_58994

theorem abs_inequality_solution_set (x : ℝ) : (|x - 1| ≥ 5) ↔ (x ≥ 6 ∨ x ≤ -4) := 
by sorry

end abs_inequality_solution_set_l58_58994


namespace inverse_matrix_l58_58783

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 7], ![-1, -1]]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![-(1/3 : ℚ), -(7/3 : ℚ)], ![1/3, 4/3]]

theorem inverse_matrix : A.det ≠ 0 → A⁻¹ = A_inv := by
  sorry

end inverse_matrix_l58_58783


namespace convex_polygon_max_interior_angles_l58_58168

theorem convex_polygon_max_interior_angles (n : ℕ) (h1 : n ≥ 3) (h2 : n < 360) :
  ∃ x, x ≤ 4 ∧ ∀ k, k > 4 → False :=
by
  sorry

end convex_polygon_max_interior_angles_l58_58168


namespace max_area_rectangle_min_area_rectangle_l58_58024

theorem max_area_rectangle (n : ℕ) (x y : ℕ → ℕ)
  (S : ℕ → ℕ) (H1 : ∀ k, S k = 2^(2*k)) 
  (H2 : ∀ k, (1 ≤ k ∧ k ≤ n) → x k * y k = S k) 
  : (n - 1 + 2^(2*n)) * (4 * 2^(2*(n-1)) - 1/3) = 1/3 * (4^n - 1) * (4^n + n - 1) := sorry

theorem min_area_rectangle (n : ℕ) (x y : ℕ → ℕ)
  (S : ℕ → ℕ) (H1 : ∀ k, S k = 2^(2*k)) 
  (H2 : ∀ k, (1 ≤ k ∧ k ≤ n) → x k * y k = S k)
  : (2^n - 1)^2 = 4 * (2^n - 1)^2 := sorry

end max_area_rectangle_min_area_rectangle_l58_58024


namespace repeating_decimal_eq_fraction_l58_58841

noncomputable def repeating_decimal_to_fraction (x : ℝ) : ℝ :=
  let x : ℝ := 4.5656565656 -- * 0.5656... repeating
  (100*x - x) / (100 - 1)

-- Define the theorem we want to prove
theorem repeating_decimal_eq_fraction : 
  ∀ x : ℝ, x = 4.565656 -> x = (452 : ℝ) / (99 : ℝ) :=
by
  intro x h
  -- here we would provide the proof steps, but since it's omitted
  -- we'll use sorry to skip it.
  sorry

end repeating_decimal_eq_fraction_l58_58841


namespace cos_double_angle_l58_58368

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 1 / 2) : Real.cos (2 * α) = 3 / 5 :=
by
  sorry

end cos_double_angle_l58_58368


namespace triangles_with_positive_integer_area_count_l58_58456

theorem triangles_with_positive_integer_area_count :
  let points := { p : (ℕ × ℕ) // 41 * p.1 + p.2 = 2017 }
  ∃ count, count = 600 ∧ ∀ (P Q : points), P ≠ Q →
    let area := (P.val.1 * Q.val.2 - Q.val.1 * P.val.2 : ℤ)
    0 < area ∧ (area % 2 = 0) := sorry

end triangles_with_positive_integer_area_count_l58_58456


namespace expiry_time_correct_l58_58649

def factorial (n : Nat) : Nat := match n with
| 0 => 1
| n + 1 => (n + 1) * factorial n

def seconds_in_a_day : Nat := 86400
def seconds_in_an_hour : Nat := 3600
def donation_time_seconds : Nat := 8 * seconds_in_an_hour
def expiry_seconds : Nat := factorial 8

def time_of_expiry (donation_time : Nat) (expiry_time : Nat) : Nat :=
  (donation_time + expiry_time) % seconds_in_a_day

def time_to_HM (time_seconds : Nat) : Nat × Nat :=
  let hours := time_seconds / seconds_in_an_hour
  let minutes := (time_seconds % seconds_in_an_hour) / 60
  (hours, minutes)

def is_correct_expiry_time : Prop :=
  let (hours, minutes) := time_to_HM (time_of_expiry donation_time_seconds expiry_seconds)
  hours = 19 ∧ minutes = 12

theorem expiry_time_correct : is_correct_expiry_time := by
  sorry

end expiry_time_correct_l58_58649


namespace vectors_orthogonal_dot_product_l58_58603

theorem vectors_orthogonal_dot_product (y : ℤ) :
  (3 * -2) + (4 * y) + (-1 * 5) = 0 → y = 11 / 4 :=
by
  sorry

end vectors_orthogonal_dot_product_l58_58603


namespace kate_retirement_fund_value_l58_58203

theorem kate_retirement_fund_value 
(initial_value decrease final_value : ℝ) 
(h1 : initial_value = 1472)
(h2 : decrease = 12)
(h3 : final_value = initial_value - decrease) : 
final_value = 1460 := 
by
  sorry

end kate_retirement_fund_value_l58_58203


namespace number_of_red_candies_is_4_l58_58720

-- Define the parameters as given in the conditions
def number_of_green_candies : ℕ := 5
def number_of_blue_candies : ℕ := 3
def likelihood_of_blue_candy : ℚ := 25 / 100

-- Define the total number of candies
def total_number_of_candies (number_of_red_candies : ℕ) : ℕ :=
  number_of_green_candies + number_of_blue_candies + number_of_red_candies

-- Define the proof statement
theorem number_of_red_candies_is_4 (R : ℕ) :
  (3 / total_number_of_candies R = 25 / 100) → R = 4 :=
sorry

end number_of_red_candies_is_4_l58_58720


namespace distance_from_reflected_point_l58_58125

theorem distance_from_reflected_point
  (P : ℝ × ℝ) (P' : ℝ × ℝ)
  (hP : P = (3, 2))
  (hP' : P' = (3, -2))
  : dist P P' = 4 := sorry

end distance_from_reflected_point_l58_58125


namespace manny_gave_2_marbles_l58_58212

-- Define the total number of marbles
def total_marbles : ℕ := 36

-- Define the ratio parts for Mario and Manny
def mario_ratio : ℕ := 4
def manny_ratio : ℕ := 5

-- Define the total ratio parts
def total_ratio : ℕ := mario_ratio + manny_ratio

-- Define the number of marbles Manny has after giving some away
def manny_marbles_now : ℕ := 18

-- Calculate the marbles per part based on the ratio and total marbles
def marbles_per_part : ℕ := total_marbles / total_ratio

-- Calculate the number of marbles Manny originally had
def manny_marbles_original : ℕ := manny_ratio * marbles_per_part

-- Formulate the theorem
theorem manny_gave_2_marbles : manny_marbles_original - manny_marbles_now = 2 := by
  sorry

end manny_gave_2_marbles_l58_58212


namespace exists_indices_l58_58932

open Nat List

theorem exists_indices (m n : ℕ) (a : Fin m → ℕ) (b : Fin n → ℕ) 
  (h1 : ∀ i : Fin m, a i ≤ n) (h2 : ∀ i j : Fin m, i ≤ j → a i ≤ a j)
  (h3 : ∀ j : Fin n, b j ≤ m) (h4 : ∀ i j : Fin n, i ≤ j → b i ≤ b j) :
  ∃ i : Fin m, ∃ j : Fin n, a i + i.val + 1 = b j + j.val + 1 := by
  sorry

end exists_indices_l58_58932


namespace find_least_number_l58_58698

theorem find_least_number (x : ℕ) :
  (∀ k, 24 ∣ k + 7 → 32 ∣ k + 7 → 36 ∣ k + 7 → 54 ∣ k + 7 → x = k) → 
  x + 7 = Nat.lcm (Nat.lcm (Nat.lcm 24 32) 36) 54 → x = 857 :=
by
  sorry

end find_least_number_l58_58698


namespace find_valid_tax_range_l58_58682

noncomputable def valid_tax_range (t : ℝ) : Prop :=
  let initial_consumption := 200000
  let price_per_cubic_meter := 240
  let consumption_reduction := 2.5 * t * 10^4
  let tax_revenue := (initial_consumption - consumption_reduction) * price_per_cubic_meter * (t / 100)
  tax_revenue >= 900000

theorem find_valid_tax_range (t : ℝ) : 3 ≤ t ∧ t ≤ 5 ↔ valid_tax_range t :=
sorry

end find_valid_tax_range_l58_58682


namespace library_books_l58_58013

theorem library_books (A : Prop) (B : Prop) (C : Prop) (D : Prop) :
  (¬A) → (B ∧ D) :=
by
  -- Assume the statement "All books in this library are available for lending." is represented by A.
  -- A is false.
  intro h_notA
  -- Show that statement II ("There is some book in this library not available for lending.")
  -- and statement IV ("Not all books in this library are available for lending.") are both true.
  -- These are represented as B and D, respectively.
  sorry

end library_books_l58_58013


namespace domain_intersection_l58_58838

theorem domain_intersection (A B : Set ℝ) 
    (h1 : A = {x | x < 1})
    (h2 : B = {y | y ≥ 0}) : A ∩ B = {z | 0 ≤ z ∧ z < 1} := 
by
  sorry

end domain_intersection_l58_58838


namespace simplify_expression_l58_58351

theorem simplify_expression : 4 * (8 - 2 + 3) - 7 = 29 := 
by {
  sorry
}

end simplify_expression_l58_58351


namespace fraction_simplification_l58_58154

def numerator : Int := 5^4 + 5^2 + 5
def denominator : Int := 5^3 - 2 * 5

theorem fraction_simplification :
  (numerator : ℚ) / (denominator : ℚ) = 27 + (14 / 23) := by
  sorry

end fraction_simplification_l58_58154


namespace teamAPointDifferenceTeamB_l58_58685

-- Definitions for players' scores and penalties
structure Player where
  name : String
  points : ℕ
  penalties : List ℕ

def TeamA : List Player := [
  { name := "Beth", points := 12, penalties := [1, 2] },
  { name := "Jan", points := 18, penalties := [1, 2, 3] },
  { name := "Mike", points := 5, penalties := [] },
  { name := "Kim", points := 7, penalties := [1, 2] },
  { name := "Chris", points := 6, penalties := [1] }
]

def TeamB : List Player := [
  { name := "Judy", points := 10, penalties := [1, 2] },
  { name := "Angel", points := 9, penalties := [1] },
  { name := "Nick", points := 12, penalties := [] },
  { name := "Steve", points := 8, penalties := [1, 2, 3] },
  { name := "Mary", points := 5, penalties := [1, 2] },
  { name := "Vera", points := 4, penalties := [1] }
]

-- Helper function to calculate total points for a player considering penalties
def Player.totalPoints (p : Player) : ℕ :=
  p.points - p.penalties.sum

-- Helper function to calculate total points for a team
def totalTeamPoints (team : List Player) : ℕ :=
  team.foldr (λ p acc => acc + p.totalPoints) 0

def teamAPoints : ℕ := totalTeamPoints TeamA
def teamBPoints : ℕ := totalTeamPoints TeamB

theorem teamAPointDifferenceTeamB :
  teamAPoints - teamBPoints = 1 :=
  sorry

end teamAPointDifferenceTeamB_l58_58685


namespace marginal_cost_proof_l58_58920

theorem marginal_cost_proof (fixed_cost : ℕ) (total_cost : ℕ) (n : ℕ) (MC : ℕ)
  (h1 : fixed_cost = 12000)
  (h2 : total_cost = 16000)
  (h3 : n = 20)
  (h4 : total_cost = fixed_cost + MC * n) :
  MC = 200 :=
  sorry

end marginal_cost_proof_l58_58920


namespace right_triangle_third_side_l58_58979

theorem right_triangle_third_side (a b : ℝ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :
  c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2) :=
by 
  sorry

end right_triangle_third_side_l58_58979


namespace evaluate_expression_l58_58776

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15

theorem evaluate_expression : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 240 / 961 := 
by 
  sorry

end evaluate_expression_l58_58776


namespace price_of_orange_is_60_l58_58306

-- Definitions from the conditions
def price_of_apple : ℕ := 40 -- The price of each apple is 40 cents
def total_fruits : ℕ := 10 -- Mary selects a total of 10 apples and oranges
def avg_price_initial : ℕ := 48 -- The average price of the 10 pieces of fruit is 48 cents
def put_back_oranges : ℕ := 2 -- Mary puts back 2 oranges
def avg_price_remaining : ℕ := 45 -- The average price of the remaining fruits is 45 cents

-- Variable definition for the price of an orange which will be solved for
variable (price_of_orange : ℕ)

-- Theorem: proving the price of each orange is 60 cents given the conditions
theorem price_of_orange_is_60 : 
  (∀ a o : ℕ, a + o = total_fruits →
  40 * a + price_of_orange * o = total_fruits * avg_price_initial →
  40 * a + price_of_orange * (o - put_back_oranges) = (total_fruits - put_back_oranges) * avg_price_remaining)
  → price_of_orange = 60 :=
by
  -- Proof is omitted
  sorry

end price_of_orange_is_60_l58_58306


namespace problem1_problem2_l58_58855

theorem problem1 : (1 : ℤ) - (2 : ℤ)^3 / 8 - ((1 / 4 : ℚ) * (-2)^2) = (-2 : ℤ) := by
  sorry

theorem problem2 : (-(1 / 12 : ℚ) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = (-21 : ℤ) := by
  sorry

end problem1_problem2_l58_58855


namespace inscribed_square_area_l58_58247

theorem inscribed_square_area :
  (∃ (t : ℝ), (2*t)^2 = 4 * (t^2) ∧ ∀ (x y : ℝ), (x = t ∧ y = t ∨ x = -t ∧ y = t ∨ x = t ∧ y = -t ∨ x = -t ∧ y = -t) 
  → (x^2 / 4 + y^2 / 8 = 1) ) 
  → (∃ (a : ℝ), a = 32 / 3) := 
by
  sorry

end inscribed_square_area_l58_58247


namespace neg_09_not_in_integers_l58_58919

def negative_numbers : Set ℝ := {x | x < 0}
def fractions : Set ℝ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}
def integers : Set ℝ := {x | ∃ (n : ℤ), x = n}
def rational_numbers : Set ℝ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

theorem neg_09_not_in_integers : -0.9 ∉ integers :=
by {
  sorry
}

end neg_09_not_in_integers_l58_58919


namespace squares_below_16x_144y_1152_l58_58781

noncomputable def count_squares_below_line (a b c : ℝ) (x_max y_max : ℝ) : ℝ :=
  let total_squares := x_max * y_max
  let line_slope := -a/b
  let squares_crossed_by_diagonal := x_max + y_max - 1
  (total_squares - squares_crossed_by_diagonal) / 2

theorem squares_below_16x_144y_1152 : 
  count_squares_below_line 16 144 1152 72 8 = 248.5 := 
by
  sorry

end squares_below_16x_144y_1152_l58_58781


namespace total_lunch_bill_l58_58041

theorem total_lunch_bill (hotdog salad : ℝ) (h1 : hotdog = 5.36) (h2 : salad = 5.10) : hotdog + salad = 10.46 := 
by
  rw [h1, h2]
  norm_num
  

end total_lunch_bill_l58_58041


namespace undefined_integer_count_l58_58785

noncomputable def expression (x : ℤ) : ℚ := (x^2 - 16) / ((x^2 - x - 6) * (x - 4))

theorem undefined_integer_count : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x^2 - x - 6) * (x - 4) = 0) ∧ S.card = 3 :=
  sorry

end undefined_integer_count_l58_58785


namespace comp_figure_perimeter_l58_58152

-- Given conditions
def side_length_square : ℕ := 2
def side_length_triangle : ℕ := 1
def number_of_squares : ℕ := 4
def number_of_triangles : ℕ := 3

-- Define the perimeter calculation
def perimeter_of_figure : ℕ :=
  let perimeter_squares := (2 * (number_of_squares - 2) + 2 * 2 + 2 * 1) * side_length_square
  let perimeter_triangles := number_of_triangles * side_length_triangle
  perimeter_squares + perimeter_triangles

-- Target theorem
theorem comp_figure_perimeter : perimeter_of_figure = 17 := by
  sorry

end comp_figure_perimeter_l58_58152


namespace total_leaves_correct_l58_58102

-- Definitions based on conditions
def basil_pots := 3
def rosemary_pots := 9
def thyme_pots := 6

def basil_leaves_per_pot := 4
def rosemary_leaves_per_pot := 18
def thyme_leaves_per_pot := 30

-- Calculate the total number of leaves
def total_leaves : Nat :=
  (basil_pots * basil_leaves_per_pot) +
  (rosemary_pots * rosemary_leaves_per_pot) +
  (thyme_pots * thyme_leaves_per_pot)

-- The statement to prove
theorem total_leaves_correct : total_leaves = 354 := by
  sorry

end total_leaves_correct_l58_58102


namespace integer_pairs_satisfying_condition_l58_58298

theorem integer_pairs_satisfying_condition :
  { (m, n) : ℤ × ℤ | ∃ k : ℤ, (n^3 + 1) = k * (m * n - 1) } =
  { (1, 2), (1, 3), (2, 1), (3, 1), (2, 5), (3, 5), (5, 2), (5, 3), (2, 2) } :=
sorry

end integer_pairs_satisfying_condition_l58_58298


namespace spring_festival_scientific_notation_l58_58054

noncomputable def scientific_notation := (260000000: ℝ) = (2.6 * 10^8)

theorem spring_festival_scientific_notation : scientific_notation :=
by
  -- proof logic goes here
  sorry

end spring_festival_scientific_notation_l58_58054


namespace simplify_expression_l58_58971

theorem simplify_expression : 
  (1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3 + 1 / (1 / 3)^4)) = 1 / 120 := 
by 
  sorry

end simplify_expression_l58_58971


namespace claire_photos_eq_10_l58_58273

variable (C L R : Nat)

theorem claire_photos_eq_10
  (h1: L = 3 * C)
  (h2: R = C + 20)
  (h3: L = R)
  : C = 10 := by
  sorry

end claire_photos_eq_10_l58_58273


namespace arrangement_count_l58_58969

-- Definitions corresponding to the conditions in a)
def num_students : ℕ := 8
def max_per_activity : ℕ := 5

-- Lean statement reflecting the target theorem in c)
theorem arrangement_count (n : ℕ) (max : ℕ) 
  (h1 : n = num_students)
  (h2 : max = max_per_activity) :
  ∃ total : ℕ, total = 182 :=
sorry

end arrangement_count_l58_58969


namespace man_saves_percentage_of_salary_l58_58285

variable (S : ℝ) (P : ℝ) (S_s : ℝ)

def problem_statement (S : ℝ) (S_s : ℝ) (P : ℝ) : Prop :=
  S_s = S - 1.2 * (S - (P / 100) * S)

theorem man_saves_percentage_of_salary
  (h1 : S = 6250)
  (h2 : S_s = 250) :
  problem_statement S S_s 20 :=
by
  sorry

end man_saves_percentage_of_salary_l58_58285


namespace remainder_of_N_mod_45_l58_58252

def concatenated_num_from_1_to_52 : ℕ := 
  -- This represents the concatenated number from 1 to 52.
  -- We define here in Lean as a placeholder 
  -- since Lean cannot concatenate numbers directly.
  12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152

theorem remainder_of_N_mod_45 : 
  concatenated_num_from_1_to_52 % 45 = 37 := 
sorry

end remainder_of_N_mod_45_l58_58252


namespace nina_earnings_l58_58405

/-- 
Problem: Calculate the total earnings from selling various types of jewelry.
Conditions:
- Necklace price: $25 each
- Bracelet price: $15 each
- Earring price: $10 per pair
- Complete jewelry ensemble price: $45 each
- Number of necklaces sold: 5
- Number of bracelets sold: 10
- Number of earrings sold: 20
- Number of complete jewelry ensembles sold: 2
Question: How much money did Nina make over the weekend?
Answer: Nina made $565.00
-/
theorem nina_earnings
  (necklace_price : ℕ)
  (bracelet_price : ℕ)
  (earring_price : ℕ)
  (ensemble_price : ℕ)
  (necklaces_sold : ℕ)
  (bracelets_sold : ℕ)
  (earrings_sold : ℕ)
  (ensembles_sold : ℕ) :
  necklace_price = 25 → 
  bracelet_price = 15 → 
  earring_price = 10 → 
  ensemble_price = 45 → 
  necklaces_sold = 5 → 
  bracelets_sold = 10 → 
  earrings_sold = 20 → 
  ensembles_sold = 2 →
  (necklace_price * necklaces_sold) + 
  (bracelet_price * bracelets_sold) + 
  (earring_price * earrings_sold) +
  (ensemble_price * ensembles_sold) = 565 := by
  sorry

end nina_earnings_l58_58405


namespace find_Q_digit_l58_58420

theorem find_Q_digit (P Q R S T U : ℕ) (h1 : P ≠ Q) (h2 : P ≠ R) (h3 : P ≠ S)
  (h4 : P ≠ T) (h5 : P ≠ U) (h6 : Q ≠ R) (h7 : Q ≠ S) (h8 : Q ≠ T)
  (h9 : Q ≠ U) (h10 : R ≠ S) (h11 : R ≠ T) (h12 : R ≠ U) (h13 : S ≠ T)
  (h14 : S ≠ U) (h15 : T ≠ U) (h_range_P : 4 ≤ P ∧ P ≤ 9)
  (h_range_Q : 4 ≤ Q ∧ Q ≤ 9) (h_range_R : 4 ≤ R ∧ R ≤ 9)
  (h_range_S : 4 ≤ S ∧ S ≤ 9) (h_range_T : 4 ≤ T ∧ T ≤ 9)
  (h_range_U : 4 ≤ U ∧ U ≤ 9) 
  (h_sum_lines : 3 * P + 2 * Q + 3 * S + R + T + 2 * U = 100)
  (h_sum_digits : P + Q + S + R + T + U = 39) : Q = 6 :=
sorry  -- proof to be provided

end find_Q_digit_l58_58420


namespace total_stops_traveled_l58_58527

-- Definitions based on the conditions provided
def yoojeong_stops : ℕ := 3
def namjoon_stops : ℕ := 2

-- Theorem statement to prove the total number of stops
theorem total_stops_traveled : yoojeong_stops + namjoon_stops = 5 := by
  -- Proof omitted
  sorry

end total_stops_traveled_l58_58527


namespace find_number_l58_58863

theorem find_number (x : ℝ) (h : x - (3/5) * x = 60) : x = 150 :=
by
  sorry

end find_number_l58_58863


namespace prove_correct_option_C_l58_58374

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end prove_correct_option_C_l58_58374


namespace farm_cows_l58_58120

theorem farm_cows (c h : ℕ) 
  (legs_eq : 5 * c + 2 * h = 20 + 2 * (c + h)) : 
  c = 6 :=
by 
  sorry

end farm_cows_l58_58120


namespace tree_growth_per_two_weeks_l58_58912

-- Definitions based on conditions
def initial_height_meters : ℕ := 2
def initial_height_centimeters : ℕ := initial_height_meters * 100
def final_height_centimeters : ℕ := 600
def total_growth : ℕ := final_height_centimeters - initial_height_centimeters
def weeks_in_4_months : ℕ := 16
def number_of_two_week_periods : ℕ := weeks_in_4_months / 2

-- Objective: Prove that the growth every two weeks is 50 centimeters
theorem tree_growth_per_two_weeks :
  (total_growth / number_of_two_week_periods) = 50 :=
  by
  sorry

end tree_growth_per_two_weeks_l58_58912


namespace area_of_rectangle_l58_58529

-- Definitions and conditions
def side_of_square : ℕ := 50
def radius_of_circle : ℕ := side_of_square
def length_of_rectangle : ℕ := (2 * radius_of_circle) / 5
def breadth_of_rectangle : ℕ := 10

-- Theorem statement
theorem area_of_rectangle :
  (length_of_rectangle * breadth_of_rectangle = 200) := by
  sorry

end area_of_rectangle_l58_58529


namespace a_b_work_days_l58_58344

-- Definitions:
def work_days_a_b_together := 40
def work_days_a_alone := 12
def remaining_work_days_with_a := 9

-- Statement to be proven:
theorem a_b_work_days (x : ℕ) 
  (h1 : ∀ W : ℕ, W / work_days_a_b_together + remaining_work_days_with_a * (W / work_days_a_alone) = W) :
  x = 10 :=
sorry

end a_b_work_days_l58_58344


namespace kate_average_speed_correct_l58_58928

noncomputable def kate_average_speed : ℝ :=
  let biking_time_hours := 20 / 60
  let walking_time_hours := 60 / 60
  let jogging_time_hours := 40 / 60
  let biking_distance := 20 * biking_time_hours
  let walking_distance := 4 * walking_time_hours
  let jogging_distance := 6 * jogging_time_hours
  let total_distance := biking_distance + walking_distance + jogging_distance
  let total_time_hours := biking_time_hours + walking_time_hours + jogging_time_hours
  total_distance / total_time_hours

theorem kate_average_speed_correct : kate_average_speed = 9 :=
by
  sorry

end kate_average_speed_correct_l58_58928


namespace range_of_a_l58_58658

/-- 
Proof problem statement derived from the given math problem and solution:
Prove that if the conditions:
1. ∀ x > 0, x + 1/x > a
2. ∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0
3. ¬ (∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0) is false
4. (∀ x > 0, x + 1/x > a) ∧ (∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0) is false
hold, then a ≥ 2.
-/
theorem range_of_a (a : ℝ)
  (h1 : ∀ x : ℝ, x > 0 → x + 1 / x > a)
  (h2 : ∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0)
  (h3 : ¬ (¬ (∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0)))
  (h4 : ¬ ((∀ x : ℝ, x > 0 → x + 1 / x > a) ∧ (∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0))) :
  a ≥ 2 :=
sorry

end range_of_a_l58_58658


namespace work_done_in_one_day_by_A_and_B_l58_58849

noncomputable def A_days : ℕ := 12
noncomputable def B_days : ℕ := A_days / 2

theorem work_done_in_one_day_by_A_and_B : 1 / (A_days : ℚ) + 1 / (B_days : ℚ) = 1 / 4 := by
  sorry

end work_done_in_one_day_by_A_and_B_l58_58849


namespace new_person_weight_l58_58488

theorem new_person_weight (avg_increase : ℝ) (num_people : ℕ) (weight_replaced : ℝ) (new_weight : ℝ) : 
    num_people = 8 → avg_increase = 1.5 → weight_replaced = 65 → 
    new_weight = weight_replaced + num_people * avg_increase → 
    new_weight = 77 :=
by
  intros h1 h2 h3 h4
  sorry

end new_person_weight_l58_58488


namespace average_marks_math_chem_l58_58242

-- Definitions to capture the conditions
variables (M P C : ℕ)
variable (cond1 : M + P = 32)
variable (cond2 : C = P + 20)

-- The theorem to prove
theorem average_marks_math_chem (M P C : ℕ) 
  (cond1 : M + P = 32) 
  (cond2 : C = P + 20) : 
  (M + C) / 2 = 26 := 
sorry

end average_marks_math_chem_l58_58242


namespace total_typing_cost_l58_58907

def typingCost (totalPages revisedOncePages revisedTwicePages : ℕ) (firstTimeCost revisionCost : ℕ) : ℕ := 
  let initialCost := totalPages * firstTimeCost
  let firstRevisionCost := revisedOncePages * revisionCost
  let secondRevisionCost := revisedTwicePages * (revisionCost * 2)
  initialCost + firstRevisionCost + secondRevisionCost

theorem total_typing_cost : typingCost 200 80 20 5 3 = 1360 := 
  by 
    rfl

end total_typing_cost_l58_58907


namespace num_quadricycles_l58_58946

theorem num_quadricycles (b t q : ℕ) (h1 : b + t + q = 10) (h2 : 2 * b + 3 * t + 4 * q = 30) : q = 2 :=
by sorry

end num_quadricycles_l58_58946


namespace find_side_b_of_triangle_l58_58425

theorem find_side_b_of_triangle
  (A B : Real) (a b : Real)
  (hA : A = Real.pi / 6)
  (hB : B = Real.pi / 4)
  (ha : a = 2) :
  b = 2 * Real.sqrt 2 :=
sorry

end find_side_b_of_triangle_l58_58425


namespace difference_in_pages_l58_58533

def purple_pages_per_book : ℕ := 230
def orange_pages_per_book : ℕ := 510
def purple_books_read : ℕ := 5
def orange_books_read : ℕ := 4

theorem difference_in_pages : 
  orange_books_read * orange_pages_per_book - purple_books_read * purple_pages_per_book = 890 :=
by
  sorry

end difference_in_pages_l58_58533


namespace divisor_is_36_l58_58968

theorem divisor_is_36
  (Dividend Quotient Remainder : ℕ)
  (h1 : Dividend = 690)
  (h2 : Quotient = 19)
  (h3 : Remainder = 6)
  (h4 : Dividend = (Divisor * Quotient) + Remainder) :
  Divisor = 36 :=
sorry

end divisor_is_36_l58_58968


namespace ab_product_eq_four_l58_58219

theorem ab_product_eq_four (a b : ℝ) (h1: 0 < a) (h2: 0 < b) 
  (h3: (1/2) * (4 / a) * (6 / b) = 3) : 
  a * b = 4 :=
by 
  sorry

end ab_product_eq_four_l58_58219


namespace tangent_addition_tangent_subtraction_l58_58771

theorem tangent_addition (a b : ℝ) : 
  Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
sorry

theorem tangent_subtraction (a b : ℝ) : 
  Real.tan (a - b) = (Real.tan a - Real.tan b) / (1 + Real.tan a * Real.tan b) :=
sorry

end tangent_addition_tangent_subtraction_l58_58771


namespace can_reach_4_white_l58_58211

/-
We define the possible states and operations on the urn as described.
-/

structure Urn :=
  (white : ℕ)
  (black : ℕ)

def operation1 (u : Urn) : Urn :=
  { white := u.white, black := u.black - 2 }

def operation2 (u : Urn) : Urn :=
  { white := u.white, black := u.black - 2 }

def operation3 (u : Urn) : Urn :=
  { white := u.white - 1, black := u.black - 1 }

def operation4 (u : Urn) : Urn :=
  { white := u.white - 2, black := u.black + 1 }

theorem can_reach_4_white : ∃ (u : Urn), u.white = 4 ∧ u.black > 0 :=
  sorry

end can_reach_4_white_l58_58211


namespace initial_concentration_l58_58734

theorem initial_concentration (f : ℚ) (C : ℚ) (h₀ : f = 0.7142857142857143) (h₁ : (1 - f) * C + f * 0.25 = 0.35) : C = 0.6 :=
by
  rw [h₀] at h₁
  -- The proof will follow the steps to solve for C
  sorry

end initial_concentration_l58_58734


namespace yogurt_price_is_5_l58_58813

theorem yogurt_price_is_5
  (yogurt_pints : ℕ)
  (gum_packs : ℕ)
  (shrimp_trays : ℕ)
  (total_cost : ℝ)
  (shrimp_cost : ℝ)
  (gum_fraction : ℝ)
  (price_frozen_yogurt : ℝ) :
  yogurt_pints = 5 →
  gum_packs = 2 →
  shrimp_trays = 5 →
  total_cost = 55 →
  shrimp_cost = 5 →
  gum_fraction = 0.5 →
  5 * price_frozen_yogurt + 2 * (gum_fraction * price_frozen_yogurt) + 5 * shrimp_cost = total_cost →
  price_frozen_yogurt = 5 :=
by
  intro hp hg hs ht hc hf h_formula
  sorry

end yogurt_price_is_5_l58_58813


namespace geometric_sequence_root_product_l58_58810

theorem geometric_sequence_root_product
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (a1_pos : 0 < a 1)
  (a19_root : a 1 * r^18 = (1 : ℝ))
  (h_poly : ∀ x, x^2 - 10 * x + 16 = 0) :
  a 8 * a 12 = 16  :=
sorry

end geometric_sequence_root_product_l58_58810


namespace gcd_840_1764_l58_58673

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l58_58673


namespace reciprocal_of_one_twentieth_l58_58200

theorem reciprocal_of_one_twentieth : (1 / (1 / 20 : ℝ)) = 20 := 
by
  sorry

end reciprocal_of_one_twentieth_l58_58200


namespace range_of_a_squared_minus_2b_l58_58226

variable (a b : ℝ)

def quadratic_has_two_real_roots_in_01 (a b : ℝ) : Prop :=
  b ≥ 0 ∧ 1 + a + b ≥ 0 ∧ -2 ≤ a ∧ a ≤ 0 ∧ a^2 - 4 * b ≥ 0

theorem range_of_a_squared_minus_2b (a b : ℝ)
  (h : quadratic_has_two_real_roots_in_01 a b) : 0 ≤ a^2 - 2 * b ∧ a^2 - 2 * b ≤ 2 :=
sorry

end range_of_a_squared_minus_2b_l58_58226


namespace base_7_divisibility_l58_58218

theorem base_7_divisibility (y : ℕ) :
  (934 + 7 * y) % 19 = 0 ↔ y = 3 :=
by
  sorry

end base_7_divisibility_l58_58218


namespace union_complements_l58_58356

open Set

variable (U : Set ℕ) (A B : Set ℕ)

-- Define the conditions
def condition_U : U = {1, 2, 3, 4, 5} := by
  sorry

def condition_A : A = {1, 2, 3} := by
  sorry

def condition_B : B = {2, 3, 4} := by
  sorry

-- Prove that (complement_U A) ∪ (complement_U B) = {1, 4, 5}
theorem union_complements :
  (U \ A) ∪ (U \ B) = {1, 4, 5} := by
  sorry

end union_complements_l58_58356


namespace ratio_of_m1_and_m2_l58_58794

theorem ratio_of_m1_and_m2 (m a b m1 m2 : ℝ) (h1 : a^2 * m - 3 * a * m + 2 * a + 7 = 0) (h2 : b^2 * m - 3 * b * m + 2 * b + 7 = 0) 
  (h3 : (a / b) + (b / a) = 2) (h4 : m1^2 * 9 - m1 * 28 + 4 = 0) (h5 : m2^2 * 9 - m2 * 28 + 4 = 0) : 
  (m1 / m2) + (m2 / m1) = 194 / 9 := 
sorry

end ratio_of_m1_and_m2_l58_58794


namespace hypotenuse_is_2_l58_58904

noncomputable def quadratic_trinomial_hypotenuse (a b c : ℝ) : ℝ :=
  let x1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let xv := -b / (2 * a)
  let yv := a * xv^2 + b * xv + c
  if xv = (x1 + x2) / 2 then
    Real.sqrt 2 * abs (-b / a)
  else 0

theorem hypotenuse_is_2 {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  quadratic_trinomial_hypotenuse a b c = 2 := by
  sorry

end hypotenuse_is_2_l58_58904


namespace cylindrical_to_rectangular_conversion_l58_58939

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular 6 (5 * Real.pi / 4) (-3) = (-3 * Real.sqrt 2, -3 * Real.sqrt 2, -3) :=
by
  sorry

end cylindrical_to_rectangular_conversion_l58_58939


namespace max_photo_area_correct_l58_58163

def frame_area : ℝ := 59.6
def num_photos : ℕ := 4
def max_photo_area : ℝ := 14.9

theorem max_photo_area_correct : frame_area / num_photos = max_photo_area :=
by sorry

end max_photo_area_correct_l58_58163


namespace find_m_from_decomposition_l58_58490

theorem find_m_from_decomposition (m : ℕ) (h : m > 0) : (m^2 - m + 1 = 73) → (m = 9) :=
by
  sorry

end find_m_from_decomposition_l58_58490


namespace square_area_l58_58418

theorem square_area (side_length : ℝ) (h : side_length = 11) : side_length * side_length = 121 := 
by 
  simp [h]
  sorry

end square_area_l58_58418


namespace triangle_inequality_product_l58_58042

theorem triangle_inequality_product (x y z : ℝ) (h1 : x + y > z) (h2 : x + z > y) (h3 : y + z > x) :
  (x + y + z) * (x + y - z) * (x + z - y) * (z + y - x) ≤ 4 * x^2 * y^2 := 
by
  sorry

end triangle_inequality_product_l58_58042


namespace greatest_possible_d_l58_58280

noncomputable def point_2d_units_away_origin (d : ℝ) : Prop :=
  2 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d + 5)^2)

theorem greatest_possible_d : 
  ∃ d : ℝ, point_2d_units_away_origin d ∧ d = (5 + Real.sqrt 244) / 3 :=
sorry

end greatest_possible_d_l58_58280


namespace ratio_third_second_l58_58963

theorem ratio_third_second (k : ℝ) (x y z : ℝ) (h1 : y = 4 * x) (h2 : x = 18) (h3 : z = k * y) (h4 : (x + y + z) / 3 = 78) :
  z = 2 * y :=
by
  sorry

end ratio_third_second_l58_58963


namespace geometric_sequence_problem_l58_58133

theorem geometric_sequence_problem (a : ℕ → ℝ) (r : ℝ) 
  (h_geo : ∀ n, a (n + 1) = r * a n) 
  (h_cond: a 4 + a 6 = 8) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 :=
  sorry

end geometric_sequence_problem_l58_58133


namespace sum_of_center_coordinates_l58_58793

theorem sum_of_center_coordinates : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 = 6*x - 10*y + 24) -> 
  (∃ (cx cy : ℝ), (x^2 - 6*x + y^2 + 10*y = (cx - 3)^2 + (cy + 5)^2 + 58) ∧ (cx + cy = -2)) :=
  sorry

end sum_of_center_coordinates_l58_58793


namespace sum_of_radii_l58_58853

noncomputable def tangency_equation (r : ℝ) : Prop :=
  (r - 5)^2 + r^2 = (r + 1.5)^2

theorem sum_of_radii : ∀ (r1 r2 : ℝ), tangency_equation r1 ∧ tangency_equation r2 →
  r1 + r2 = 13 :=
by
  intros r1 r2 h
  sorry

end sum_of_radii_l58_58853


namespace probability_all_five_dice_even_l58_58664

-- Definitions of conditions
def standard_six_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6}
def even_numbers : Set ℕ := {2, 4, 6}

-- The statement to be proven
theorem probability_all_five_dice_even : 
  (∀ die ∈ standard_six_sided_die, (∃ n ∈ even_numbers, die = n)) → (1 / 32) = (1 / 2) ^ 5 :=
by
  intro h
  sorry

end probability_all_five_dice_even_l58_58664


namespace fill_in_square_l58_58386

theorem fill_in_square (x y : ℝ) (h : 4 * x^2 * (81 / 4 * x * y) = 81 * x^3 * y) : (81 / 4 * x * y) = (81 / 4 * x * y) :=
by
  sorry

end fill_in_square_l58_58386


namespace find_a_l58_58713

theorem find_a (α β : ℝ) (h1 : α + β = 10) (h2 : α * β = 20) : (1 / α + 1 / β) = 1 / 2 :=
sorry

end find_a_l58_58713


namespace sum_of_reciprocals_is_3_over_8_l58_58064

theorem sum_of_reciprocals_is_3_over_8 (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  (1 / x + 1 / y) = 3 / 8 := 
by 
  sorry

end sum_of_reciprocals_is_3_over_8_l58_58064


namespace value_of_transformed_product_of_roots_l58_58693

theorem value_of_transformed_product_of_roots 
  (a b : ℚ)
  (h1 : 3 * a^2 + 4 * a - 7 = 0)
  (h2 : 3 * b^2 + 4 * b - 7 = 0)
  (h3 : a ≠ b) : 
  (a - 2) * (b - 2) = 13 / 3 :=
by
  -- The exact proof would be completed here.
  sorry

end value_of_transformed_product_of_roots_l58_58693


namespace quadratic_no_real_roots_l58_58830

theorem quadratic_no_real_roots 
  (p q a b c : ℝ) 
  (h1 : 0 < p) (h2 : 0 < q) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c)
  (h6 : p ≠ q)
  (h7 : a^2 = p * q)
  (h8 : b + c = p + q)
  (h9 : b = (2 * p + q) / 3)
  (h10 : c = (p + 2 * q) / 3) :
  (∀ x : ℝ, ¬ (b * x^2 - 2 * a * x + c = 0)) := 
by
  sorry

end quadratic_no_real_roots_l58_58830


namespace sum_of_squares_of_sides_l58_58411

-- Definition: A cyclic quadrilateral with perpendicular diagonals inscribed in a circle
structure CyclicQuadrilateral (R : ℝ) :=
  (m n k t : ℝ) -- sides of the quadrilateral
  (perpendicular_diagonals : true) -- diagonals are perpendicular (trivial placeholder)
  (radius : ℝ := R) -- Radius of the circumscribed circle

-- The theorem to prove: The sum of the squares of the sides of the quadrilateral is 8R^2
theorem sum_of_squares_of_sides (R : ℝ) (quad : CyclicQuadrilateral R) :
  quad.m ^ 2 + quad.n ^ 2 + quad.k ^ 2 + quad.t ^ 2 = 8 * R^2 := 
by sorry

end sum_of_squares_of_sides_l58_58411


namespace blue_chairs_fewer_than_yellow_l58_58710

theorem blue_chairs_fewer_than_yellow :
  ∀ (red_chairs yellow_chairs chairs_left total_chairs blue_chairs : ℕ),
    red_chairs = 4 →
    yellow_chairs = 2 * red_chairs →
    chairs_left = 15 →
    total_chairs = chairs_left + 3 →
    blue_chairs = total_chairs - (red_chairs + yellow_chairs) →
    yellow_chairs - blue_chairs = 2 :=
by sorry

end blue_chairs_fewer_than_yellow_l58_58710


namespace denise_travel_l58_58694

theorem denise_travel (a b c : ℕ) (h₀ : a ≥ 1) (h₁ : a + b + c = 8) (h₂ : 90 * (b - a) % 48 = 0) : a^2 + b^2 + c^2 = 26 :=
sorry

end denise_travel_l58_58694


namespace inequality_amgm_l58_58628

variable {a b c : ℝ}

theorem inequality_amgm (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) : 
  (1 / 2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) <= a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1 / 3) ∧ 
  a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1 / 3) <= (a - b)^2 + (b - c)^2 + (c - a)^2 := 
by 
  sorry

end inequality_amgm_l58_58628


namespace line_intersects_circle_and_angle_conditions_l58_58637

noncomputable def line_circle_intersection_condition (k : ℝ) : Prop :=
  - (Real.sqrt 3) / 3 ≤ k ∧ k ≤ (Real.sqrt 3) / 3

noncomputable def inclination_angle_condition (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi)

theorem line_intersects_circle_and_angle_conditions (k θ : ℝ) :
  line_circle_intersection_condition k →
  inclination_angle_condition θ →
  ∃ x y : ℝ, (y = k * (x + 1)) ∧ ((x - 1)^2 + y^2 = 1) :=
by
  sorry

end line_intersects_circle_and_angle_conditions_l58_58637


namespace other_root_is_neg_2_l58_58283

theorem other_root_is_neg_2 (k : ℝ) (h : Polynomial.eval 0 (Polynomial.C k + Polynomial.X * 2 + Polynomial.X ^ 2) = 0) : 
  ∃ t : ℝ, (Polynomial.eval t (Polynomial.C k + Polynomial.X * 2 + Polynomial.X ^ 2) = 0) ∧ t = -2 :=
by
  sorry

end other_root_is_neg_2_l58_58283


namespace tan_sin_cos_proof_l58_58153

theorem tan_sin_cos_proof (h1 : Real.sin (Real.pi / 6) = 1 / 2)
    (h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2)
    (h3 : Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6)) :
    ((Real.tan (Real.pi / 6))^2 - (Real.sin (Real.pi / 6))^2) / ((Real.tan (Real.pi / 6))^2 * (Real.cos (Real.pi / 6))^2) = 1 / 3 := by
  sorry

end tan_sin_cos_proof_l58_58153


namespace total_amount_paid_l58_58162

-- Definitions based on the conditions in part (a)
def cost_of_manicure : ℝ := 30
def tip_percentage : ℝ := 0.30

-- Proof statement based on conditions and answer in part (c)
theorem total_amount_paid : cost_of_manicure + (cost_of_manicure * tip_percentage) = 39 := by
  sorry

end total_amount_paid_l58_58162


namespace money_left_eq_l58_58595

theorem money_left_eq :
  let initial_money := 56
  let notebooks := 7
  let cost_per_notebook := 4
  let books := 2
  let cost_per_book := 7
  let money_left := initial_money - (notebooks * cost_per_notebook + books * cost_per_book)
  money_left = 14 :=
by
  sorry

end money_left_eq_l58_58595


namespace dot_product_value_l58_58296

-- Define vectors a and b, and the condition of their linear combination
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def a : Vector2D := ⟨-1, 2⟩
def b (m : ℝ) : Vector2D := ⟨m, 1⟩

-- Define the condition that vector a + 2b is parallel to 2a - b
def parallel (v w : Vector2D) : Prop := ∃ k : ℝ, v.x = k * w.x ∧ v.y = k * w.y

def vector_add (v w : Vector2D) : Vector2D := ⟨v.x + w.x, v.y + w.y⟩
def scalar_mul (c : ℝ) (v : Vector2D) : Vector2D := ⟨c * v.x, c * v.y⟩

-- Dot product definition
def dot_product (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

-- The theorem to prove
theorem dot_product_value (m : ℝ)
  (h : parallel (vector_add a (scalar_mul 2 (b m))) (vector_add (scalar_mul 2 a) (scalar_mul (-1) (b m)))) :
  dot_product a (b m) = 5 / 2 :=
sorry

end dot_product_value_l58_58296


namespace printer_time_ratio_l58_58679

theorem printer_time_ratio
  (X_time : ℝ) (Y_time : ℝ) (Z_time : ℝ)
  (hX : X_time = 15)
  (hY : Y_time = 10)
  (hZ : Z_time = 20) :
  (X_time / (Y_time * Z_time / (Y_time + Z_time))) = 9 / 4 :=
by
  sorry

end printer_time_ratio_l58_58679


namespace cube_volume_l58_58578

-- Define the condition: the surface area of the cube is 54
def surface_area_of_cube (x : ℝ) : Prop := 6 * x^2 = 54

-- Define the theorem that states the volume of the cube given the surface area condition
theorem cube_volume : ∃ (x : ℝ), surface_area_of_cube x ∧ x^3 = 27 := by
  sorry

end cube_volume_l58_58578


namespace count_whole_numbers_in_interval_l58_58380

open Real

theorem count_whole_numbers_in_interval : 
  ∃ n : ℕ, n = 5 ∧ ∀ x : ℕ, (sqrt 7 < x ∧ x < exp 2) ↔ (3 ≤ x ∧ x ≤ 7) :=
by
  sorry

end count_whole_numbers_in_interval_l58_58380


namespace tan_sum_identity_l58_58070

-- Definitions
def quadratic_eq (x : ℝ) : Prop := 6 * x^2 - 5 * x + 1 = 0
def tan_roots (α β : ℝ) : Prop := quadratic_eq (Real.tan α) ∧ quadratic_eq (Real.tan β)

-- Problem statement
theorem tan_sum_identity (α β : ℝ) (hαβ : tan_roots α β) : Real.tan (α + β) = 1 :=
sorry

end tan_sum_identity_l58_58070


namespace city_grid_sinks_l58_58094

-- Define the main conditions of the grid city
def cell_side_meter : Int := 500
def max_travel_km : Int := 1

-- Total number of intersections in a 100x100 grid
def total_intersections : Int := (100 + 1) * (100 + 1)

-- Number of sinks that need to be proven
def required_sinks : Int := 1300

-- Lean theorem statement to prove that given the conditions,
-- there are at least 1300 sinks (intersections that act as sinks)
theorem city_grid_sinks :
  ∀ (city_grid : Matrix (Fin 101) (Fin 101) IntersectionType),
  (∀ i j, i < 100 → j < 100 → cell_side_meter ≤ max_travel_km * 1000) →
  ∃ (sinks : Finset (Fin 101 × Fin 101)), 
  (sinks.card ≥ required_sinks) := sorry

end city_grid_sinks_l58_58094


namespace expected_BBR_sequences_l58_58243

theorem expected_BBR_sequences :
  let total_cards := 52
  let black_cards := 26
  let red_cards := 26
  let probability_of_next_black := (25 / 51)
  let probability_of_third_red := (26 / 50)
  let probability_of_BBR := probability_of_next_black * probability_of_third_red
  let possible_start_positions := 26
  let expected_BBR := possible_start_positions * probability_of_BBR
  expected_BBR = (338 / 51) :=
by
  sorry

end expected_BBR_sequences_l58_58243


namespace probability_no_absolute_winner_l58_58773

def no_absolute_winner_prob (P_AB : ℝ) (P_BV : ℝ) (P_VA : ℝ) : ℝ :=
  0.24 * P_VA + 0.36 * (1 - P_VA)

theorem probability_no_absolute_winner :
  (∀ P_VA : ℝ, P_VA >= 0 ∧ P_VA <= 1 → no_absolute_winner_prob 0.6 0.4 P_VA == 0.24) :=
sorry

end probability_no_absolute_winner_l58_58773


namespace purchase_price_l58_58996

-- Define the context and conditions 
variables (P S : ℝ)
-- Define the conditions
axiom cond1 : S = P + 0.5 * S
axiom cond2 : S - P = 100

-- Define the main theorem
theorem purchase_price : P = 100 :=
by sorry

end purchase_price_l58_58996


namespace students_without_glasses_l58_58040

theorem students_without_glasses (total_students : ℕ) (perc_glasses : ℕ) (students_with_glasses students_without_glasses : ℕ) 
  (h1 : total_students = 325) (h2 : perc_glasses = 40) (h3 : students_with_glasses = perc_glasses * total_students / 100)
  (h4 : students_without_glasses = total_students - students_with_glasses) : students_without_glasses = 195 := 
by
  sorry

end students_without_glasses_l58_58040


namespace arithmetic_expression_l58_58209

theorem arithmetic_expression : (5 * 7 - 6 + 2 * 12 + 2 * 6 + 7 * 3) = 86 :=
by
  sorry

end arithmetic_expression_l58_58209


namespace minimum_cost_to_buy_additional_sheets_l58_58831

def total_sheets : ℕ := 98
def students : ℕ := 12
def cost_per_sheet : ℕ := 450

theorem minimum_cost_to_buy_additional_sheets : 
  (students * (1 + total_sheets / students) - total_sheets) * cost_per_sheet = 4500 :=
by {
  sorry
}

end minimum_cost_to_buy_additional_sheets_l58_58831


namespace inequality_solution_l58_58167

theorem inequality_solution (x y : ℝ) (h1 : y ≥ x^2 + 1) :
    2^y - 2 * Real.cos x + Real.sqrt (y - x^2 - 1) ≤ 0 ↔ x = 0 ∧ y = 1 :=
by
  sorry

end inequality_solution_l58_58167


namespace triangle_angle_sum_l58_58985

theorem triangle_angle_sum (CD CB : ℝ) 
    (isosceles_triangle: CD = CB)
    (interior_pentagon_angle: 108 = 180 * (5 - 2) / 5)
    (interior_triangle_angle: 60 = 180 / 3)
    (triangle_angle_sum: ∀ (a b c : ℝ), a + b + c = 180) :
    mangle_CDB = 6 :=
by
  have x : ℝ := 6
  sorry

end triangle_angle_sum_l58_58985


namespace x_cubed_plus_y_cubed_l58_58887

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := 
by 
  sorry

end x_cubed_plus_y_cubed_l58_58887


namespace man_work_days_l58_58379

variable (W : ℝ) -- Denoting the amount of work by W

-- Defining the work rate variables
variables (M Wm B : ℝ)

-- Conditions from the problem:
-- Combined work rate of man, woman, and boy together completes the work in 3 days
axiom combined_work_rate : M + Wm + B = W / 3
-- Woman completes the work alone in 18 days
axiom woman_work_rate : Wm = W / 18
-- Boy completes the work alone in 9 days
axiom boy_work_rate : B = W / 9

-- The goal is to prove the man takes 6 days to complete the work alone
theorem man_work_days : (W / M) = 6 :=
by
  sorry

end man_work_days_l58_58379


namespace Joan_initial_money_l58_58661

def cost_hummus (containers : ℕ) (price_per_container : ℕ) : ℕ := containers * price_per_container
def cost_apple (quantity : ℕ) (price_per_apple : ℕ) : ℕ := quantity * price_per_apple

theorem Joan_initial_money 
  (containers_of_hummus : ℕ)
  (price_per_hummus : ℕ)
  (cost_chicken : ℕ)
  (cost_bacon : ℕ)
  (cost_vegetables : ℕ)
  (quantity_apple : ℕ)
  (price_per_apple : ℕ)
  (total_cost : ℕ)
  (remaining_money : ℕ):
  containers_of_hummus = 2 →
  price_per_hummus = 5 →
  cost_chicken = 20 →
  cost_bacon = 10 →
  cost_vegetables = 10 →
  quantity_apple = 5 →
  price_per_apple = 2 →
  remaining_money = cost_apple quantity_apple price_per_apple →
  total_cost = cost_hummus containers_of_hummus price_per_hummus + cost_chicken + cost_bacon + cost_vegetables + remaining_money →
  total_cost = 60 :=
by
  intros
  sorry

end Joan_initial_money_l58_58661


namespace mike_games_l58_58156

theorem mike_games (init_money spent_money game_cost : ℕ) (h1 : init_money = 42) (h2 : spent_money = 10) (h3 : game_cost = 8) :
  (init_money - spent_money) / game_cost = 4 :=
by
  sorry

end mike_games_l58_58156


namespace fill_sink_time_l58_58854

theorem fill_sink_time {R1 R2 R T: ℝ} (h1: R1 = 1 / 210) (h2: R2 = 1 / 214) (h3: R = R1 + R2) (h4: T = 1 / R):
  T = 105.75 :=
by 
  sorry

end fill_sink_time_l58_58854


namespace abs_diff_simplification_l58_58021

theorem abs_diff_simplification (a b : ℝ) (h1 : a < 0) (h2 : a * b < 0) : |b - a + 1| - |a - b - 5| = -4 :=
  sorry

end abs_diff_simplification_l58_58021


namespace find_derivative_l58_58988

theorem find_derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f' 1 = -1 := 
by
  sorry

end find_derivative_l58_58988


namespace total_coughs_after_20_minutes_l58_58571

def coughs_in_n_minutes (rate_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  rate_per_minute * minutes

def total_coughs (georgia_rate_per_minute : ℕ) (minutes : ℕ) (multiplier : ℕ) : ℕ :=
  let georgia_coughs := coughs_in_n_minutes georgia_rate_per_minute minutes
  let robert_rate_per_minute := georgia_rate_per_minute * multiplier
  let robert_coughs := coughs_in_n_minutes robert_rate_per_minute minutes
  georgia_coughs + robert_coughs

theorem total_coughs_after_20_minutes :
  total_coughs 5 20 2 = 300 :=
by
  sorry

end total_coughs_after_20_minutes_l58_58571


namespace find_mass_of_water_vapor_l58_58022

noncomputable def heat_balance_problem : Prop :=
  ∃ (m_s : ℝ), m_s * 536 + m_s * 80 = 
  (50 * 80 + 50 * 20 + 300 * 20 + 100 * 0.5 * 20)
  ∧ m_s = 19.48

theorem find_mass_of_water_vapor : heat_balance_problem := by
  sorry

end find_mass_of_water_vapor_l58_58022


namespace base_9_units_digit_of_sum_l58_58180

def base_n_units_digit (n : ℕ) (a : ℕ) : ℕ :=
a % n

theorem base_9_units_digit_of_sum : base_n_units_digit 9 (45 + 76) = 2 :=
by
  sorry

end base_9_units_digit_of_sum_l58_58180


namespace ratio_Pat_Mark_l58_58717

-- Total hours charged by all three
def total_hours (P K M : ℕ) : Prop :=
  P + K + M = 144

-- Pat charged twice as much time as Kate
def pat_hours (P K : ℕ) : Prop :=
  P = 2 * K

-- Mark charged 80 hours more than Kate
def mark_hours (M K : ℕ) : Prop :=
  M = K + 80

-- The ratio of Pat's hours to Mark's hours
def ratio (P M : ℕ) : ℚ :=
  (P : ℚ) / (M : ℚ)

theorem ratio_Pat_Mark (P K M : ℕ)
  (h1 : total_hours P K M)
  (h2 : pat_hours P K)
  (h3 : mark_hours M K) :
  ratio P M = (1 : ℚ) / (3 : ℚ) :=
by
  sorry

end ratio_Pat_Mark_l58_58717


namespace merchant_loss_l58_58875

theorem merchant_loss (n m : ℝ) (h₁ : n ≠ m) : 
  let x := n / m
  let y := m / n
  x + y > 2 := by
sorry

end merchant_loss_l58_58875


namespace isosceles_triangle_angle_B_l58_58220

theorem isosceles_triangle_angle_B :
  ∀ (A B C : ℝ), (B = C) → (C = 3 * A) → (A + B + C = 180) → (B = 540 / 7) :=
by
  intros A B C h1 h2 h3
  sorry

end isosceles_triangle_angle_B_l58_58220


namespace selling_price_41_l58_58951

-- Purchase price per item
def purchase_price : ℝ := 30

-- Government restriction on pice increase: selling price cannot be more than 40% increase of the purchase price
def price_increase_restriction (a : ℝ) : Prop :=
  a <= purchase_price * 1.4

-- Profit condition equation
def profit_condition (a : ℝ) : Prop :=
  (a - purchase_price) * (112 - 2 * a) = 330

-- The selling price of each item that satisfies all conditions is 41 yuan  
theorem selling_price_41 (a : ℝ) (h1 : profit_condition a) (h2 : price_increase_restriction a) :
  a = 41 := sorry

end selling_price_41_l58_58951


namespace kimberly_bought_skittles_l58_58874

-- Conditions
def initial_skittles : ℕ := 5
def total_skittles : ℕ := 12

-- Prove
theorem kimberly_bought_skittles : ∃ bought_skittles : ℕ, (total_skittles = initial_skittles + bought_skittles) ∧ bought_skittles = 7 :=
by
  sorry

end kimberly_bought_skittles_l58_58874


namespace contractor_total_engaged_days_l58_58961

-- Definitions based on conditions
def earnings_per_work_day : ℝ := 25
def fine_per_absent_day : ℝ := 7.5
def total_earnings : ℝ := 425
def days_absent : ℝ := 10

-- The proof problem statement
theorem contractor_total_engaged_days :
  ∃ (x y : ℝ), y = days_absent ∧ total_earnings = earnings_per_work_day * x - fine_per_absent_day * y ∧ x + y = 30 :=
by
  -- let x be the number of working days
  -- let y be the number of absent days
  -- y is given as 10
  -- total_earnings = 25 * x - 7.5 * 10
  -- solve for x and sum x and y to get 30
  sorry

end contractor_total_engaged_days_l58_58961


namespace measure_of_angle_f_l58_58383

theorem measure_of_angle_f (angle_D angle_E angle_F : ℝ)
  (h1 : angle_D = 75)
  (h2 : angle_E = 4 * angle_F + 30)
  (h3 : angle_D + angle_E + angle_F = 180) : 
  angle_F = 15 :=
by
  sorry

end measure_of_angle_f_l58_58383


namespace quotient_base6_division_l58_58393

theorem quotient_base6_division :
  let a := 2045
  let b := 14
  let base := 6
  a / b = 51 :=
by
  sorry

end quotient_base6_division_l58_58393


namespace range_of_a_l58_58179

namespace ProofProblem

theorem range_of_a (a : ℝ) (H1 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → ∃ y : ℝ, y = a * x + 2 * a + 1 ∧ y > 0 ∧ y < 0) : 
  -1 < a ∧ a < -1/3 := 
sorry

end ProofProblem

end range_of_a_l58_58179


namespace Josh_lost_marbles_l58_58834

theorem Josh_lost_marbles :
  let original_marbles := 9.5
  let current_marbles := 4.25
  original_marbles - current_marbles = 5.25 :=
by
  sorry

end Josh_lost_marbles_l58_58834


namespace litter_patrol_total_pieces_l58_58146

theorem litter_patrol_total_pieces :
  let glass_bottles := 25
  let aluminum_cans := 18
  let plastic_bags := 12
  let paper_cups := 7
  let cigarette_packs := 5
  let discarded_face_masks := 3
  glass_bottles + aluminum_cans + plastic_bags + paper_cups + cigarette_packs + discarded_face_masks = 70 :=
by
  sorry

end litter_patrol_total_pieces_l58_58146


namespace parabola_distance_to_focus_l58_58543

theorem parabola_distance_to_focus :
  ∀ (P : ℝ × ℝ), P.1 = 2 ∧ P.2^2 = 4 * P.1 → dist P (1, 0) = 3 :=
by
  intro P h
  have h₁ : P.1 = 2 := h.1
  have h₂ : P.2^2 = 4 * P.1 := h.2
  sorry

end parabola_distance_to_focus_l58_58543


namespace external_tangency_sum_internal_tangency_diff_converse_sum_of_radii_converse_diff_of_radii_l58_58427

variables {O₁ O₂ : ℝ} {r R : ℝ}

-- External tangency implies sum of radii equals distance between centers
theorem external_tangency_sum {O₁ O₂ r R : ℝ} (h1 : O₁ ≠ O₂) (h2 : ∀ M, (dist O₁ M = r) ∧ (dist O₂ M = R) → dist O₁ O₂ = r + R) : 
  dist O₁ O₂ = r + R :=
sorry

-- Internal tangency implies difference of radii equals distance between centers
theorem internal_tangency_diff {O₁ O₂ r R : ℝ} 
  (h1 : O₁ ≠ O₂) 
  (h2 : ∀ M, (dist O₁ M = r) ∧ (dist O₂ M = R) → dist O₁ O₂ = abs (R - r)) : 
  dist O₁ O₂ = abs (R - r) :=
sorry

-- Converse for sum of radii equals distance between centers
theorem converse_sum_of_radii {O₁ O₂ r R : ℝ}
  (h1 : O₁ ≠ O₂) 
  (h2 : dist O₁ O₂ = r + R) : 
  ∃ M, (dist O₁ M = r) ∧ (dist O₂ M = R) ∧ (dist O₁ O₂ = r + R) :=
sorry

-- Converse for difference of radii equals distance between centers
theorem converse_diff_of_radii {O₁ O₂ r R : ℝ}
  (h1 : O₁ ≠ O₂) 
  (h2 : dist O₁ O₂ = abs (R - r)) : 
  ∃ M, (dist O₁ M = r) ∧ (dist O₂ M = R) ∧ (dist O₁ O₂ = abs (R - r)) :=
sorry

end external_tangency_sum_internal_tangency_diff_converse_sum_of_radii_converse_diff_of_radii_l58_58427


namespace num_boys_in_circle_l58_58149

theorem num_boys_in_circle (n : ℕ) 
  (h : ∃ k, n = 2 * k ∧ k = 40 - 10) : n = 60 :=
by
  sorry

end num_boys_in_circle_l58_58149


namespace bob_age_sum_digits_l58_58598

theorem bob_age_sum_digits
  (A B C : ℕ)  -- Define ages for Alice (A), Bob (B), and Carl (C)
  (h1 : C = 2)  -- Carl's age is 2
  (h2 : B = A + 2)  -- Bob is 2 years older than Alice
  (h3 : ∃ n, A = 2 * n ∧ n > 0 ∧ n ≤ 8 )  -- Alice's age is a multiple of Carl's age today, marking the second of the 8 such multiples 
  : ∃ n, (B + n) % (C + n) = 0 ∧ (B + n) = 50 :=  -- Prove that the next time Bob's age is a multiple of Carl's, Bob's age will be 50
sorry

end bob_age_sum_digits_l58_58598


namespace percent_within_one_standard_deviation_l58_58449

variable (m d : ℝ)
variable (distribution : ℝ → ℝ)
variable (symmetric_about_mean : ∀ x, distribution (m + x) = distribution (m - x))
variable (percent_less_than_m_plus_d : distribution (m + d) = 0.84)

theorem percent_within_one_standard_deviation :
  distribution (m + d) - distribution (m - d) = 0.68 :=
sorry

end percent_within_one_standard_deviation_l58_58449


namespace find_number_l58_58933

theorem find_number (N Q : ℕ) (h1 : N = 11 * Q) (h2 : Q + N + 11 = 71) : N = 55 :=
by {
  sorry
}

end find_number_l58_58933


namespace C_work_completion_l58_58493

theorem C_work_completion (A_completion_days B_completion_days AB_completion_days : ℕ)
  (A_cond : A_completion_days = 8)
  (B_cond : B_completion_days = 12)
  (AB_cond : AB_completion_days = 4) :
  ∃ (C_completion_days : ℕ), C_completion_days = 24 := 
by
  sorry

end C_work_completion_l58_58493


namespace units_digit_1_to_99_is_5_l58_58039

noncomputable def units_digit_of_product_of_odds : ℕ :=
  let seq := List.range' 1 99;
  (seq.filter (λ n => n % 2 = 1)).prod % 10

theorem units_digit_1_to_99_is_5 : units_digit_of_product_of_odds = 5 :=
by sorry

end units_digit_1_to_99_is_5_l58_58039


namespace value_of_g_at_neg3_l58_58465

def g (x : ℚ) : ℚ := (6 * x + 2) / (x - 2)

theorem value_of_g_at_neg3 : g (-3) = 16 / 5 := by
  sorry

end value_of_g_at_neg3_l58_58465


namespace simplify_expression_l58_58108

-- We need to prove that the simplified expression is equal to the expected form
theorem simplify_expression (y : ℝ) : (3 * y - 7 * y^2 + 4 - (5 + 3 * y - 7 * y^2)) = (0 * y^2 + 0 * y - 1) :=
by
  -- The detailed proof steps will go here
  sorry

end simplify_expression_l58_58108


namespace maximum_value_minimum_value_l58_58337

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def check_digits (N M : ℕ) (a b c d e f g h : ℕ) : Prop :=
  N = 1000 * a + 100 * b + 10 * c + d ∧
  M = 1000 * e + 100 * f + 10 * g + h ∧
  a ≠ e ∧
  b ≠ f ∧
  c ≠ g ∧
  d ≠ h ∧
  a ≠ f ∧
  a ≠ g ∧
  a ≠ h ∧
  b ≠ e ∧
  b ≠ g ∧
  b ≠ h ∧
  c ≠ e ∧
  c ≠ f ∧
  c ≠ h ∧
  d ≠ e ∧
  d ≠ f ∧
  d ≠ g

theorem maximum_value (N M a b c d e f g h : ℕ) :
  is_four_digit_number N →
  is_four_digit_number M →
  check_digits N M a b c d e f g h →
  N - M = 1994 →
  N + M = 15000 :=
by
  intros
  sorry

theorem minimum_value (N M a b c d e f g h : ℕ) :
  is_four_digit_number N →
  is_four_digit_number M →
  check_digits N M a b c d e f g h →
  N - M = 1994 →
  N + M = 4998 :=
by
  intros
  sorry

end maximum_value_minimum_value_l58_58337


namespace largest_four_digit_number_divisible_by_2_5_9_11_l58_58629

theorem largest_four_digit_number_divisible_by_2_5_9_11 : ∃ n : ℤ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∀ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (n % 2 = 0) ∧ 
  (n % 5 = 0) ∧ 
  (n % 9 = 0) ∧ 
  (n % 11 = 0) ∧ 
  (n = 8910) := 
by
  sorry

end largest_four_digit_number_divisible_by_2_5_9_11_l58_58629


namespace james_muffins_correct_l58_58507

-- Arthur baked 115 muffins
def arthur_muffins : ℕ := 115

-- James baked 12 times as many muffins as Arthur
def james_multiplier : ℕ := 12

-- The number of muffins James baked
def james_muffins : ℕ := arthur_muffins * james_multiplier

-- The expected result
def expected_james_muffins : ℕ := 1380

-- The statement we want to prove
theorem james_muffins_correct : james_muffins = expected_james_muffins := by
  sorry

end james_muffins_correct_l58_58507


namespace find_k_l58_58005

noncomputable def line1_slope : ℝ := -1
noncomputable def line2_slope (k : ℝ) : ℝ := -k / 3

theorem find_k (k : ℝ) : 
  (line2_slope k) * line1_slope = -1 → k = -3 := 
by
  sorry

end find_k_l58_58005


namespace intersection_of_M_and_N_l58_58586

open Set

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2 - |x|}

theorem intersection_of_M_and_N : M ∩ N = {y | 0 ≤ y ∧ y ≤ 2} :=
by sorry

end intersection_of_M_and_N_l58_58586


namespace average_first_21_multiples_of_6_l58_58587

-- Define the arithmetic sequence and its conditions.
def arithmetic_sequence (a1 d : ℕ) (n : ℕ) := a1 + (n - 1) * d

-- Define the problem statement.
theorem average_first_21_multiples_of_6 :
  let a1 := 6
  let d := 6
  let n := 21
  let an := arithmetic_sequence a1 d n
  (a1 + an) / 2 = 66 := by
  sorry

end average_first_21_multiples_of_6_l58_58587


namespace students_qualifying_percentage_l58_58738

theorem students_qualifying_percentage (N B G : ℕ) (boy_percent : ℝ) (girl_percent : ℝ) :
  N = 400 →
  G = 100 →
  B = N - G →
  boy_percent = 0.60 →
  girl_percent = 0.80 →
  (boy_percent * B + girl_percent * G) / N * 100 = 65 :=
by
  intros hN hG hB hBoy hGirl
  simp [hN, hG, hB, hBoy, hGirl]
  sorry

end students_qualifying_percentage_l58_58738


namespace toys_per_week_l58_58745

-- Define the number of days the workers work in a week
def days_per_week : ℕ := 4

-- Define the number of toys produced each day
def toys_per_day : ℕ := 1140

-- State the proof problem: workers produce 4560 toys per week
theorem toys_per_week : (toys_per_day * days_per_week) = 4560 :=
by
  -- Proof goes here
  sorry

end toys_per_week_l58_58745


namespace inequality_proof_l58_58476

variable (a b c d e f : Real)

theorem inequality_proof (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l58_58476


namespace volume_of_parallelepiped_l58_58882

theorem volume_of_parallelepiped 
  (l w h : ℝ)
  (h1 : l * w / Real.sqrt (l^2 + w^2) = 2 * Real.sqrt 5)
  (h2 : h * w / Real.sqrt (h^2 + w^2) = 30 / Real.sqrt 13)
  (h3 : h * l / Real.sqrt (h^2 + l^2) = 15 / Real.sqrt 10) 
  : l * w * h = 750 :=
sorry

end volume_of_parallelepiped_l58_58882


namespace man_profit_doubled_l58_58821

noncomputable def percentage_profit (C SP1 SP2 : ℝ) : ℝ :=
  (SP2 - C) / C * 100

theorem man_profit_doubled (C SP1 SP2 : ℝ) (h1 : SP1 = 1.30 * C) (h2 : SP2 = 2 * SP1) :
  percentage_profit C SP1 SP2 = 160 := by
  sorry

end man_profit_doubled_l58_58821


namespace sequence_properties_l58_58724

open BigOperators

-- Given conditions
def is_geometric_sequence (a : ℕ → ℝ) := ∃ q > 0, ∀ n, a (n + 1) = a n * q
def sequence_a (n : ℕ) : ℝ := 2^(n - 1)

-- Definitions for b_n and S_n
def sequence_b (n : ℕ) : ℕ := n - 1
def sequence_c (n : ℕ) : ℝ := sequence_a n * (sequence_b n) -- c_n = a_n * b_n

-- Statement of the problem
theorem sequence_properties (a : ℕ → ℝ) (hgeo : is_geometric_sequence a) (h1 : a 1 = 1) (h2 : a 2 * a 4 = 16) : 
 (∀ n, sequence_b n = n - 1 ) ∧ S_n = ∑ i in Finset.range n, sequence_c (i + 1) := sorry

end sequence_properties_l58_58724


namespace equal_commissions_implies_list_price_l58_58304

theorem equal_commissions_implies_list_price (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  intro h
  sorry

end equal_commissions_implies_list_price_l58_58304


namespace find_coords_C_l58_58373

-- Define the coordinates of given points
def A : ℝ × ℝ := (13, 7)
def B : ℝ × ℝ := (5, -1)
def D : ℝ × ℝ := (2, 2)

-- The proof problem wrapped in a lean theorem
theorem find_coords_C (C : ℝ × ℝ) 
  (h1 : AB = AC) (h2 : (D.1, D.2) = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) :
  C = (-1, 5) :=
sorry

end find_coords_C_l58_58373


namespace find_n_l58_58616

theorem find_n : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 11] ∧ n = 3 :=
by
  sorry

end find_n_l58_58616


namespace total_fish_caught_l58_58905

theorem total_fish_caught (C_trips : ℕ) (B_fish_per_trip : ℕ) (C_fish_per_trip : ℕ) (D_fish_per_trip : ℕ) (B_trips D_trips : ℕ) :
  C_trips = 10 →
  B_trips = 2 * C_trips →
  B_fish_per_trip = 400 →
  C_fish_per_trip = B_fish_per_trip * (1 + 2/5) →
  D_trips = 3 * C_trips →
  D_fish_per_trip = C_fish_per_trip * (1 + 50/100) →
  B_trips * B_fish_per_trip + C_trips * C_fish_per_trip + D_trips * D_fish_per_trip = 38800 := 
by
  sorry

end total_fish_caught_l58_58905


namespace family_members_l58_58991

theorem family_members (cost_purify : ℝ) (water_per_person : ℝ) (total_cost : ℝ) 
  (h1 : cost_purify = 1) (h2 : water_per_person = 1 / 2) (h3 : total_cost = 3) : 
  total_cost / (cost_purify * water_per_person) = 6 :=
by
  sorry

end family_members_l58_58991


namespace sugar_per_bar_l58_58812

theorem sugar_per_bar (bars_per_minute : ℕ) (sugar_per_2_minutes : ℕ)
  (h1 : bars_per_minute = 36)
  (h2 : sugar_per_2_minutes = 108) :
  (sugar_per_2_minutes / (bars_per_minute * 2) : ℚ) = 1.5 := 
by 
  sorry

end sugar_per_bar_l58_58812


namespace domain_of_function_l58_58431

theorem domain_of_function :
  { x : ℝ | x + 2 ≥ 0 ∧ x - 1 ≠ 0 } = { x : ℝ | x ≥ -2 ∧ x ≠ 1 } :=
by
  sorry

end domain_of_function_l58_58431


namespace correct_sampling_methods_l58_58832

-- Defining the conditions
def high_income_families : ℕ := 50
def middle_income_families : ℕ := 300
def low_income_families : ℕ := 150
def total_residents : ℕ := 500
def sample_size : ℕ := 100
def worker_group_size : ℕ := 10
def selected_workers : ℕ := 3

-- Definitions of sampling methods
inductive SamplingMethod
| random
| systematic
| stratified

open SamplingMethod

-- Problem statement in Lean 4
theorem correct_sampling_methods :
  (total_residents = high_income_families + middle_income_families + low_income_families) →
  (sample_size = 100) →
  (worker_group_size = 10) →
  (selected_workers = 3) →
  (chosen_method_for_task1 = SamplingMethod.stratified) →
  (chosen_method_for_task2 = SamplingMethod.random) →
  (chosen_method_for_task1, chosen_method_for_task2) = (SamplingMethod.stratified, SamplingMethod.random) :=
by
  intros
  sorry -- Proof to be filled in

end correct_sampling_methods_l58_58832


namespace solution_l58_58768

noncomputable def problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 4/y = 1) : Prop :=
  x + y ≥ 9

theorem solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 4/y = 1) : problem x y h1 h2 h3 :=
  sorry

end solution_l58_58768


namespace product_of_two_numbers_l58_58174

theorem product_of_two_numbers (a b : ℝ) 
  (h1 : a - b = 2 * k)
  (h2 : a + b = 8 * k)
  (h3 : 2 * a * b = 30 * k) : a * b = 15 :=
by
  sorry

end product_of_two_numbers_l58_58174


namespace impossible_rearrange_reverse_l58_58190

theorem impossible_rearrange_reverse :
  ∀ (tokens : ℕ → ℕ), 
    (∀ i, (i % 2 = 1 ∧ i < 99 → tokens i = tokens (i + 2)) 
      ∧ (i % 2 = 0 ∧ i < 99 → tokens i = tokens (i + 2))) → ¬(∀ i, tokens i = 100 + 1 - tokens (i - 1)) :=
by
  intros tokens h
  sorry

end impossible_rearrange_reverse_l58_58190


namespace common_tangents_count_l58_58972

-- Define the first circle Q1
def Q1 (x y : ℝ) := x^2 + y^2 = 9

-- Define the second circle Q2
def Q2 (x y : ℝ) := (x - 3)^2 + (y - 4)^2 = 1

-- Prove the number of common tangents between Q1 and Q2
theorem common_tangents_count :
  ∃ n : ℕ, n = 4 ∧ ∀ x y : ℝ, Q1 x y ∧ Q2 x y -> n = 4 := sorry

end common_tangents_count_l58_58972


namespace geometric_sequence_ratio_l58_58622

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h0 : q ≠ 1) 
  (h1 : ∀ n, S n = a 0 * (1 - q^n) / (1 - q)) 
  (h2 : ∀ n, a n = a 0 * q^n) 
  (h3 : 2 * S 3 = 7 * a 2) :
  (S 5 / a 2 = 31 / 2) ∨ (S 5 / a 2 = 31 / 8) :=
by sorry

end geometric_sequence_ratio_l58_58622


namespace noah_class_size_l58_58215

theorem noah_class_size :
  ∀ n : ℕ, (n = 39 + 39 + 1) → n = 79 :=
by
  intro n
  intro h
  exact h

end noah_class_size_l58_58215


namespace find_distance_l58_58921

theorem find_distance (T D : ℝ) 
  (h1 : D = 5 * (T + 0.2)) 
  (h2 : D = 6 * (T - 0.25)) : 
  D = 13.5 :=
by
  sorry

end find_distance_l58_58921


namespace u_2023_is_4_l58_58816

def f (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 2
  | 4 => 1
  | 5 => 4
  | _ => 0  -- f is only defined for x in {1, 2, 3, 4, 5}

def u : ℕ → ℕ
| 0 => 5
| (n + 1) => f (u n)

theorem u_2023_is_4 : u 2023 = 4 := by
  sorry

end u_2023_is_4_l58_58816


namespace lines_do_not_form_triangle_l58_58557

noncomputable def line1 (x y : ℝ) := 3 * x - y + 2 = 0
noncomputable def line2 (x y : ℝ) := 2 * x + y + 3 = 0
noncomputable def line3 (m x y : ℝ) := m * x + y = 0

theorem lines_do_not_form_triangle (m : ℝ) :
  (∃ x y : ℝ, line1 x y ∧ line2 x y) →
  (∀ x y : ℝ, (line1 x y → line3 m x y) ∨ (line2 x y → line3 m x y) ∨ 
    (line1 x y ∧ line2 x y → line3 m x y)) →
  (m = -3 ∨ m = 2 ∨ m = -1) :=
by
  sorry

end lines_do_not_form_triangle_l58_58557


namespace no_square_remainder_2_infinitely_many_squares_remainder_3_l58_58204

theorem no_square_remainder_2 :
  ∀ n : ℤ, (n * n) % 6 ≠ 2 :=
by sorry

theorem infinitely_many_squares_remainder_3 :
  ∀ k : ℤ, ∃ n : ℤ, n = 6 * k + 3 ∧ (n * n) % 6 = 3 :=
by sorry

end no_square_remainder_2_infinitely_many_squares_remainder_3_l58_58204


namespace triangle_side_identity_l58_58725

theorem triangle_side_identity
  (a b c : ℝ)
  (alpha beta gamma : ℝ)
  (h1 : alpha = 60)
  (h2 : a^2 = b^2 + c^2 - b * c) :
  a^2 = (a^3 + b^3 + c^3) / (a + b + c) := 
by
  sorry

end triangle_side_identity_l58_58725


namespace returning_players_count_l58_58429

def total_players_in_team (groups : ℕ) (players_per_group : ℕ): ℕ := groups * players_per_group
def returning_players (total_players : ℕ) (new_players : ℕ): ℕ := total_players - new_players

theorem returning_players_count
    (new_players : ℕ)
    (groups : ℕ)
    (players_per_group : ℕ)
    (total_players : ℕ := total_players_in_team groups players_per_group)
    (returning_players_count : ℕ := returning_players total_players new_players):
    new_players = 4 ∧
    groups = 2 ∧
    players_per_group = 5 → 
    returning_players_count = 6 := by
    intros h
    sorry

end returning_players_count_l58_58429


namespace tileable_if_and_only_if_l58_58038

def is_tileable (n : ℕ) : Prop :=
  ∃ k : ℕ, 15 * n = 4 * k

theorem tileable_if_and_only_if (n : ℕ) :
  (n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7) ↔ is_tileable n :=
sorry

end tileable_if_and_only_if_l58_58038


namespace polyhedron_volume_l58_58544

-- Define the polyhedron and its properties
def polyhedron (P : Type) : Prop :=
∃ (C : Type), 
  (∀ (p : P) (e : ℝ), e = 2) ∧ 
  (∃ (octFaces triFaces : ℕ), octFaces = 6 ∧ triFaces = 8) ∧
  (∀ (vol : ℝ), vol = (56 + (112 * Real.sqrt 2) / 3))
  
-- A theorem stating the volume of the polyhedron
theorem polyhedron_volume : ∀ (P : Type), polyhedron P → ∃ (vol : ℝ), vol = 56 + (112 * Real.sqrt 2) / 3 :=
by
  intros P hP
  sorry

end polyhedron_volume_l58_58544


namespace area_of_rectangle_l58_58392

def length : ℕ := 4
def width : ℕ := 2

theorem area_of_rectangle : length * width = 8 :=
by
  sorry

end area_of_rectangle_l58_58392


namespace prime_1021_n_unique_l58_58978

theorem prime_1021_n_unique :
  ∃! (n : ℕ), n ≥ 2 ∧ Prime (n^3 + 2 * n + 1) :=
sorry

end prime_1021_n_unique_l58_58978


namespace negation_of_universal_prop_l58_58248

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
by 
  sorry

end negation_of_universal_prop_l58_58248


namespace problem1_problem2_problem3_problem4_l58_58579

theorem problem1 : 12 - (-1) + (-7) = 6 := by
  sorry

theorem problem2 : -3.5 * (-3 / 4) / (7 / 8) = 3 := by
  sorry

theorem problem3 : (1 / 3 - 1 / 6 - 1 / 12) * (-12) = -1 := by
  sorry

theorem problem4 : (-2)^4 / (-4) * (-1/2)^2 - 1^2 = -2 := by
  sorry

end problem1_problem2_problem3_problem4_l58_58579


namespace negation_of_existential_l58_58858

def divisible_by (n x : ℤ) := ∃ k : ℤ, x = k * n
def odd (x : ℤ) := ∃ k : ℤ, x = 2 * k + 1

def P (x : ℤ) := divisible_by 7 x ∧ ¬ odd x

theorem negation_of_existential :
  (¬ ∃ x : ℤ, P x) ↔ ∀ x : ℤ, divisible_by 7 x → odd x :=
by
  sorry

end negation_of_existential_l58_58858


namespace dexter_total_cards_l58_58074

theorem dexter_total_cards 
  (boxes_basketball : ℕ) 
  (cards_per_basketball_box : ℕ) 
  (boxes_football : ℕ) 
  (cards_per_football_box : ℕ) 
   (h1 : boxes_basketball = 15)
   (h2 : cards_per_basketball_box = 20)
   (h3 : boxes_football = boxes_basketball - 7)
   (h4 : cards_per_football_box = 25) 
   : boxes_basketball * cards_per_basketball_box + boxes_football * cards_per_football_box = 500 := by 
sorry

end dexter_total_cards_l58_58074


namespace minimal_storing_capacity_required_l58_58343

theorem minimal_storing_capacity_required (k : ℕ) (h1 : k > 0)
    (bins : ℕ → ℕ → ℕ → Prop)
    (h_initial : bins 0 0 0)
    (h_laundry_generated : ∀ n, bins (10 * n) (10 * n) (10 * n))
    (h_heaviest_bin_emptied : ∀ n r b g, (r + b + g = 10 * n) → max r (max b g) + 10 * n - max r (max b g) = 10 * n)
    : ∀ (capacity : ℕ), capacity = 25 :=
sorry

end minimal_storing_capacity_required_l58_58343


namespace find_value_of_A_l58_58312

theorem find_value_of_A (x : ℝ) (h₁ : x - 3 * (x - 2) ≥ 2) (h₂ : 4 * x - 2 < 5 * x - 1) (h₃ : x ≠ 1) (h₄ : x ≠ -1) (h₅ : x ≠ 0) (hx : x = 2) :
  let A := (3 * x / (x - 1) - x / (x + 1)) / (x / (x^2 - 1))
  A = 8 :=
by
  -- Proof will be filled in
  sorry

end find_value_of_A_l58_58312


namespace days_B_can_finish_alone_l58_58244

theorem days_B_can_finish_alone (x : ℚ) : 
  (1 / 3 : ℚ) + (1 / x) = (1 / 2 : ℚ) → x = 6 := 
by
  sorry

end days_B_can_finish_alone_l58_58244


namespace average_weight_l58_58589

theorem average_weight (Ishmael Ponce Jalen : ℝ) 
  (h1 : Ishmael = Ponce + 20) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Jalen = 160) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
by 
  sorry

end average_weight_l58_58589


namespace quadratic_graph_y1_lt_y2_l58_58538

theorem quadratic_graph_y1_lt_y2 (x1 x2 : ℝ) (h1 : -x1^2 = y1) (h2 : -x2^2 = y2) (h3 : x1 * x2 > x2^2) : y1 < y2 :=
  sorry

end quadratic_graph_y1_lt_y2_l58_58538


namespace bus_speed_excluding_stoppages_l58_58264

theorem bus_speed_excluding_stoppages (v : ℕ): (45 : ℝ) = (5 / 6 * v) → v = 54 :=
by
  sorry

end bus_speed_excluding_stoppages_l58_58264


namespace perfect_square_proof_l58_58652

theorem perfect_square_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := 
sorry

end perfect_square_proof_l58_58652


namespace smallest_value_l58_58705

theorem smallest_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ (v : ℝ), (∀ x y : ℝ, 0 < x → 0 < y → v ≤ (16 / x + 108 / y + x * y)) ∧ v = 36 :=
sorry

end smallest_value_l58_58705


namespace possible_values_of_AD_l58_58194

-- Define the conditions as variables
variables {A B C D : ℝ}
variables {AB BC CD : ℝ}

-- Assume the given conditions
def conditions (A B C D : ℝ) (AB BC CD : ℝ) : Prop :=
  AB = 1 ∧ BC = 2 ∧ CD = 4

-- Define the proof goal: proving the possible values of AD
theorem possible_values_of_AD (h : conditions A B C D AB BC CD) :
  ∃ AD, AD = 1 ∨ AD = 3 ∨ AD = 5 ∨ AD = 7 :=
sorry

end possible_values_of_AD_l58_58194


namespace continuous_stripe_probability_is_3_16_l58_58223

-- Define the stripe orientation enumeration
inductive StripeOrientation
| diagonal
| straight

-- Define the face enumeration
inductive Face
| front
| back
| left
| right
| top
| bottom

-- Total number of stripe combinations (2^6 for each face having 2 orientations)
def total_combinations : ℕ := 2^6

-- Number of combinations for continuous stripes along length, width, and height
def length_combinations : ℕ := 2^2 -- 4 combinations
def width_combinations : ℕ := 2^2  -- 4 combinations
def height_combinations : ℕ := 2^2 -- 4 combinations

-- Total number of continuous stripe combinations across all dimensions
def total_continuous_stripe_combinations : ℕ :=
  length_combinations + width_combinations + height_combinations

-- Probability calculation
def continuous_stripe_probability : ℚ :=
  total_continuous_stripe_combinations / total_combinations

-- Final theorem statement
theorem continuous_stripe_probability_is_3_16 :
  continuous_stripe_probability = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_is_3_16_l58_58223


namespace parallelepiped_volume_l58_58361

noncomputable def volume_of_parallelepiped (a : ℝ) : ℝ :=
  (a^3 * Real.sqrt 2) / 2

theorem parallelepiped_volume (a : ℝ) (h_pos : 0 < a) :
  volume_of_parallelepiped a = (a^3 * Real.sqrt 2) / 2 :=
by
  sorry

end parallelepiped_volume_l58_58361


namespace cos_beta_zero_l58_58556

theorem cos_beta_zero (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) (h3 : Real.cos α = 1 / 2) (h4 : Real.cos (α + β) = -1 / 2) : Real.cos β = 0 :=
sorry

end cos_beta_zero_l58_58556


namespace fraction_operation_correct_l58_58228

theorem fraction_operation_correct {a b : ℝ} :
  (0.2 * a + 0.5 * b) ≠ 0 →
  (2 * a + 5 * b) ≠ 0 →
  (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) :=
by
  intros h1 h2
  sorry

end fraction_operation_correct_l58_58228


namespace max_clouds_crossed_by_plane_l58_58675

-- Define the conditions
def plane_region_divide (num_planes : ℕ) : ℕ :=
  num_planes + 1

-- Hypotheses/Conditions
variable (num_planes : ℕ)
variable (initial_region_clouds : ℕ)
variable (max_crosses : ℕ)

-- The primary statement to be proved
theorem max_clouds_crossed_by_plane : 
  num_planes = 10 → initial_region_clouds = 1 → max_crosses = num_planes + initial_region_clouds →
  max_crosses = 11 := 
by
  -- Placeholder for the actual proof
  intros
  sorry

end max_clouds_crossed_by_plane_l58_58675


namespace primes_eq_condition_l58_58607

theorem primes_eq_condition (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) : 
  p + q^2 = r^4 → p = 7 ∧ q = 3 ∧ r = 2 := 
by
  sorry

end primes_eq_condition_l58_58607


namespace directrix_of_parabola_l58_58208

theorem directrix_of_parabola (y x : ℝ) (h_eq : y^2 = 8 * x) :
  x = -2 :=
sorry

end directrix_of_parabola_l58_58208


namespace train_actual_speed_l58_58334
-- Import necessary libraries

-- Define the given conditions and question
def departs_time := 6
def planned_speed := 100
def scheduled_arrival_time := 18
def actual_arrival_time := 16
def distance (t₁ t₂ : ℕ) (s : ℕ) : ℕ := s * (t₂ - t₁)
def actual_speed (d t₁ t₂ : ℕ) : ℕ := d / (t₂ - t₁)

-- Proof problem statement
theorem train_actual_speed:
  actual_speed (distance departs_time scheduled_arrival_time planned_speed) departs_time actual_arrival_time = 120 := by sorry

end train_actual_speed_l58_58334


namespace even_function_and_monotonicity_l58_58584

noncomputable def f (x : ℝ) : ℝ := sorry

theorem even_function_and_monotonicity (f_symm : ∀ x : ℝ, f x = f (-x))
  (f_inc_neg : ∀ ⦃x1 x2 : ℝ⦄, x1 < x2 → x1 ≤ 0 → x2 ≤ 0 → f x1 < f x2)
  (n : ℕ) (hn : n > 0) :
  f (n + 1) < f (-n) ∧ f (-n) < f (n - 1) := 
sorry

end even_function_and_monotonicity_l58_58584


namespace calculate_ggg1_l58_58817

def g (x : ℕ) : ℕ := 7 * x + 3

theorem calculate_ggg1 : g (g (g 1)) = 514 := 
by
  sorry

end calculate_ggg1_l58_58817


namespace correct_operation_l58_58489

theorem correct_operation (a b : ℝ) : 
  (a+2)*(a-2) = a^2 - 4 :=
by
  sorry

end correct_operation_l58_58489


namespace necessarily_negative_b_plus_3b_squared_l58_58947

theorem necessarily_negative_b_plus_3b_squared
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 1) :
  b + 3 * b^2 < 0 :=
sorry

end necessarily_negative_b_plus_3b_squared_l58_58947


namespace find_k_l58_58609

noncomputable def a_squared : ℝ := 9
noncomputable def b_squared (k : ℝ) : ℝ := 4 + k
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

noncomputable def c_squared_1 (k : ℝ) : ℝ := 5 - k
noncomputable def c_squared_2 (k : ℝ) : ℝ := k - 5

theorem find_k (k : ℝ) :
  (eccentricity (Real.sqrt (c_squared_1 k)) (Real.sqrt a_squared) = 4 / 5 →
   k = -19 / 25) ∨ 
  (eccentricity (Real.sqrt (c_squared_2 k)) (Real.sqrt (b_squared k)) = 4 / 5 →
   k = 21) :=
sorry

end find_k_l58_58609


namespace find_k_l58_58711

theorem find_k (k : ℝ) (h1 : k > 0) (h2 : |3 * (k^2 - 9) - 2 * (4 * k - 15) + 2 * (12 - 5 * k)| = 20) : k = 4 := by
  sorry

end find_k_l58_58711


namespace quadratic_equation_solution_l58_58159

theorem quadratic_equation_solution : ∀ x : ℝ, x^2 - 9 = 0 ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end quadratic_equation_solution_l58_58159


namespace part1_part2_l58_58699

-- Definitions
def p (t : ℝ) := ∀ x : ℝ, x^2 + 2 * x + 2 * t - 4 ≠ 0
def q (t : ℝ) := (4 - t > 0) ∧ (t - 2 > 0)

-- Theorem statements
theorem part1 (t : ℝ) (hp : p t) : t > 5 / 2 := sorry

theorem part2 (t : ℝ) (h : p t ∨ q t) (h_and : ¬ (p t ∧ q t)) : (2 < t ∧ t ≤ 5 / 2) ∨ (t ≥ 3) := sorry

end part1_part2_l58_58699


namespace area_of_circle_diameter_7_5_l58_58257

theorem area_of_circle_diameter_7_5 :
  ∃ (A : ℝ), (A = 14.0625 * Real.pi) ↔ (∃ (d : ℝ), d = 7.5 ∧ A = Real.pi * (d / 2) ^ 2) :=
by
  sorry

end area_of_circle_diameter_7_5_l58_58257


namespace imo_42nd_inequality_l58_58866

theorem imo_42nd_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 := by
  sorry

end imo_42nd_inequality_l58_58866


namespace jogger_distance_ahead_l58_58035

noncomputable def jogger_speed_kmph : ℤ := 9
noncomputable def train_speed_kmph : ℤ := 45
noncomputable def train_length_m : ℤ := 120
noncomputable def time_to_pass_seconds : ℤ := 38

theorem jogger_distance_ahead
  (jogger_speed_kmph : ℤ)
  (train_speed_kmph : ℤ)
  (train_length_m : ℤ)
  (time_to_pass_seconds : ℤ) :
  jogger_speed_kmph = 9 →
  train_speed_kmph = 45 →
  train_length_m = 120 →
  time_to_pass_seconds = 38 →
  ∃ distance_ahead : ℤ, distance_ahead = 260 :=
by 
  -- the proof would go here
  sorry  

end jogger_distance_ahead_l58_58035


namespace jason_pokemon_cards_l58_58353

-- Conditions
def initial_cards : ℕ := 13
def cards_given : ℕ := 9

-- Proof Statement
theorem jason_pokemon_cards (initial_cards cards_given : ℕ) : initial_cards - cards_given = 4 :=
by
  sorry

end jason_pokemon_cards_l58_58353


namespace zongzi_unit_price_l58_58198

theorem zongzi_unit_price (uA uB : ℝ) (pA pB : ℝ) : 
  pA = 1200 → pB = 800 → uA = 2 * uB → pA / uA = pB / uB - 50 → uB = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end zongzi_unit_price_l58_58198


namespace a2017_value_l58_58917

def seq (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) = a n / (a n + 1)

theorem a2017_value :
  ∃ (a : ℕ → ℝ),
  seq a ∧ a 1 = 1 / 2 ∧ a 2017 = 1 / 2018 :=
by
  sorry

end a2017_value_l58_58917


namespace problem_statement_l58_58848

noncomputable def A : Set ℝ := {x | 2 < x ∧ x ≤ 6}
noncomputable def B : Set ℝ := {x | x^2 - 4 * x < 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 1}

theorem problem_statement (m : ℝ) :
    (A ∩ B = {x | 2 < x ∧ x < 4}) ∧
    (¬(A ∪ B) = {x | x ≤ 0 ∨ x > 6}) ∧
    (C m ⊆ B → m ∈ Set.Iic (5/2)) := 
by
  sorry

end problem_statement_l58_58848


namespace sufficient_but_not_necessary_l58_58900

variable (x : ℝ)

def condition1 : Prop := x > 2
def condition2 : Prop := x^2 > 4

theorem sufficient_but_not_necessary :
  (condition1 x → condition2 x) ∧ (¬ (condition2 x → condition1 x)) :=
by 
  sorry

end sufficient_but_not_necessary_l58_58900


namespace tan_condition_then_expression_value_l58_58324

theorem tan_condition_then_expression_value (θ : ℝ) (h : Real.tan θ = 2) :
  (2 * Real.sin θ) / (Real.sin θ + 2 * Real.cos θ) = 1 :=
sorry

end tan_condition_then_expression_value_l58_58324


namespace proof_inequality_l58_58058

noncomputable def proof_problem (a b c d : ℝ) (h_ab : a * b + b * c + c * d + d * a = 1) : Prop :=
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1 / 3

theorem proof_inequality (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (h_ab : a * b + b * c + c * d + d * a = 1) : 
  proof_problem a b c d h_ab := 
by
  sorry

end proof_inequality_l58_58058


namespace P_eq_Q_at_x_l58_58358

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 2
def Q (x : ℝ) : ℝ := 0

theorem P_eq_Q_at_x :
  ∃ x : ℝ, P x = Q x ∧ x = 1 :=
by
  sorry

end P_eq_Q_at_x_l58_58358


namespace problem_statement_l58_58766

open Real

namespace MathProblem

def p₁ := ∃ x : ℝ, x^2 + x + 1 < 0
def p₂ := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - 1 ≥ 0

theorem problem_statement : (¬p₁) ∨ (¬p₂) :=
by
  sorry

end MathProblem

end problem_statement_l58_58766


namespace square_numbers_divisible_by_5_between_20_and_110_l58_58625

theorem square_numbers_divisible_by_5_between_20_and_110 :
  ∃ (y : ℕ), (y = 25 ∨ y = 100) ∧ (∃ (n : ℕ), y = n^2) ∧ 5 ∣ y ∧ 20 < y ∧ y < 110 :=
by
  sorry

end square_numbers_divisible_by_5_between_20_and_110_l58_58625


namespace texts_sent_total_l58_58909

def texts_sent_on_monday_to_allison_and_brittney : Nat := 5 + 5
def texts_sent_on_tuesday_to_allison_and_brittney : Nat := 15 + 15

def total_texts_sent (texts_monday : Nat) (texts_tuesday : Nat) : Nat := texts_monday + texts_tuesday

theorem texts_sent_total :
  total_texts_sent texts_sent_on_monday_to_allison_and_brittney texts_sent_on_tuesday_to_allison_and_brittney = 40 :=
by
  sorry

end texts_sent_total_l58_58909


namespace inequality_solution_l58_58825

theorem inequality_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 6) : x^3 - 12 * x^2 + 36 * x > 0 :=
sorry

end inequality_solution_l58_58825


namespace range_of_a_l58_58903

theorem range_of_a (a : ℝ) : (a < 0 → (∀ x : ℝ, 3 * a < x ∧ x < a → x^2 - 4 * a * x + 3 * a^2 < 0)) ∧ 
                              (∀ x : ℝ, (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2 * x - 8 > 0) ↔ (x < -4 ∨ x ≥ -2)) ∧ 
                              ((¬(∀ x : ℝ, 3 * a < x ∧ x < a → x^2 - 4 * a * x + 3 * a^2 < 0)) 
                                → (¬(∀ x : ℝ, (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2 * x - 8 > 0))))
                            → (a ≤ -4 ∨ (a < 0 ∧ 3 * a >= -2)) :=
by
  intros h
  sorry

end range_of_a_l58_58903


namespace total_bill_calculation_l58_58570

theorem total_bill_calculation (n : ℕ) (amount_per_person : ℝ) (total_amount : ℝ) :
  n = 9 → amount_per_person = 514.19 → total_amount = 4627.71 → 
  n * amount_per_person = total_amount :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_bill_calculation_l58_58570


namespace find_sin_cos_of_perpendicular_vectors_l58_58085

theorem find_sin_cos_of_perpendicular_vectors 
  (θ : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (h_a : a = (Real.sin θ, -2)) 
  (h_b : b = (1, Real.cos θ)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) 
  (h_theta_range : 0 < θ ∧ θ < Real.pi / 2) : 
  Real.sin θ = 2 * Real.sqrt 5 / 5 ∧ Real.cos θ = Real.sqrt 5 / 5 := 
by 
  sorry

end find_sin_cos_of_perpendicular_vectors_l58_58085


namespace sum_possible_values_l58_58708

theorem sum_possible_values (x : ℤ) (h : ∃ y : ℤ, y = (3 * x + 13) / (x + 6)) :
  ∃ s : ℤ, s = -2 + 8 + 2 + 4 :=
sorry

end sum_possible_values_l58_58708


namespace union_A_B_intersection_A_complement_B_l58_58811

def setA (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2
def setB (x : ℝ) : Prop := x * (x - 4) ≤ 0

theorem union_A_B : {x : ℝ | setA x} ∪ {x : ℝ | setB x} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem intersection_A_complement_B : {x : ℝ | setA x} ∩ {x : ℝ | ¬ setB x} = {x : ℝ | -1 ≤ x ∧ x < 0} :=
by
  sorry

end union_A_B_intersection_A_complement_B_l58_58811


namespace total_onions_l58_58075

-- Define the number of onions grown by each individual
def nancy_onions : ℕ := 2
def dan_onions : ℕ := 9
def mike_onions : ℕ := 4

-- Proposition: The total number of onions grown is 15
theorem total_onions : (nancy_onions + dan_onions + mike_onions) = 15 := 
by sorry

end total_onions_l58_58075


namespace karl_savings_l58_58836

noncomputable def cost_per_notebook : ℝ := 3.75
noncomputable def notebooks_bought : ℕ := 8
noncomputable def discount_rate : ℝ := 0.25
noncomputable def original_total_cost : ℝ := notebooks_bought * cost_per_notebook
noncomputable def discount_per_notebook : ℝ := cost_per_notebook * discount_rate
noncomputable def discounted_price_per_notebook : ℝ := cost_per_notebook - discount_per_notebook
noncomputable def discounted_total_cost : ℝ := notebooks_bought * discounted_price_per_notebook
noncomputable def total_savings : ℝ := original_total_cost - discounted_total_cost

theorem karl_savings : total_savings = 7.50 := by 
  sorry

end karl_savings_l58_58836


namespace simple_interest_rate_problem_l58_58926

noncomputable def simple_interest_rate (P : ℝ) (T : ℝ) (final_amount : ℝ) : ℝ :=
  (final_amount - P) * 100 / (P * T)

theorem simple_interest_rate_problem
  (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : T = 2)
  (h2 : final_amount = (7 / 6) * P)
  (h3 : simple_interest_rate P T final_amount = R) : 
  R = 100 / 12 := sorry

end simple_interest_rate_problem_l58_58926


namespace cost_per_ream_is_27_l58_58389

-- Let ream_sheets be the number of sheets in one ream.
def ream_sheets : ℕ := 500

-- Let total_sheets be the total number of sheets needed.
def total_sheets : ℕ := 5000

-- Let total_cost be the total cost to buy the total number of sheets.
def total_cost : ℕ := 270

-- We need to prove that the cost per ream (in dollars) is 27.
theorem cost_per_ream_is_27 : (total_cost / (total_sheets / ream_sheets)) = 27 := 
by
  sorry

end cost_per_ream_is_27_l58_58389


namespace bottom_row_bricks_l58_58632

theorem bottom_row_bricks (n : ℕ) 
  (h1 : (n + (n-1) + (n-2) + (n-3) + (n-4) = 200)) : 
  n = 42 := 
by sorry

end bottom_row_bricks_l58_58632


namespace a_5_value_l58_58444

noncomputable def seq : ℕ → ℤ
| 0       => 1
| (n + 1) => (seq n) ^ 2 - 1

theorem a_5_value : seq 4 = -1 :=
by
  sorry

end a_5_value_l58_58444


namespace quadratic_eq_a_val_l58_58029

theorem quadratic_eq_a_val (a : ℝ) (h : a - 6 = 0) :
  a = 6 :=
by
  sorry

end quadratic_eq_a_val_l58_58029


namespace minimum_value_l58_58623

open Real

theorem minimum_value (a : ℝ) (m n : ℝ) (h_a : a > 0) (h_a_not_one : a ≠ 1) 
                      (h_mn : m * n > 0) (h_point : -m - n + 1 = 0) :
  (1 / m + 2 / n) = 3 + 2 * sqrt 2 :=
by
  -- proof should go here
  sorry

end minimum_value_l58_58623


namespace cos_seven_pi_over_six_l58_58822

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  -- Place the proof here
  sorry

end cos_seven_pi_over_six_l58_58822


namespace triangle_area_is_six_l58_58132

-- Conditions
def line_equation (Q : ℝ) : Prop :=
  ∀ (x y : ℝ), 12 * x - 4 * y + (Q - 305) = 0

def area_of_triangle (Q R : ℝ) : Prop :=
  R = (305 - Q) ^ 2 / 96

-- Question: Given a line equation forming a specific triangle, prove the area R equals 6.
theorem triangle_area_is_six (Q : ℝ) (h1 : Q = 281 ∨ Q = 329) :
  ∃ R : ℝ, line_equation Q → area_of_triangle Q R → R = 6 :=
by {
  sorry -- Proof to be provided
}

end triangle_area_is_six_l58_58132


namespace chord_square_length_l58_58370

/-- Given three circles with radii 4, 8, and 16, such that the first two are externally tangent to each other and both are internally tangent to the third, if a chord in the circle with radius 16 is a common external tangent to the other two circles, then the square of the length of this chord is 7616/9. -/
theorem chord_square_length (r1 r2 r3 : ℝ) (h1 : r1 = 4) (h2 : r2 = 8) (h3 : r3 = 16)
  (tangent_condition : ∀ (O4 O8 O16 : ℝ), O4 = r1 + r2 ∧ O8 = r2 + r3 ∧ O16 = r1 + r3) :
  (16^2 - (20/3)^2) * 4 = 7616 / 9 :=
by
  sorry

end chord_square_length_l58_58370


namespace AM_GM_Inequality_equality_condition_l58_58394

-- Given conditions
variables (n : ℕ) (a b : ℝ)

-- Assumptions
lemma condition_n : 0 < n := sorry
lemma condition_a : 0 < a := sorry
lemma condition_b : 0 < b := sorry

-- Statement
theorem AM_GM_Inequality :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) :=
sorry

-- Equality condition
theorem equality_condition :
  (1 + a / b) ^ n + (1 + b / a) ^ n = 2 ^ (n + 1) ↔ a = b :=
sorry

end AM_GM_Inequality_equality_condition_l58_58394


namespace possible_values_of_N_l58_58197

theorem possible_values_of_N (N : ℤ) (h : N^2 - N = 12) : N = 4 ∨ N = -3 :=
sorry

end possible_values_of_N_l58_58197


namespace largest_divisor_of_difference_of_squares_l58_58982

theorem largest_divisor_of_difference_of_squares (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) (h : n < m) :
  ∃ k, (∀ m n : ℤ, m % 2 = 1 → n % 2 = 1 → n < m → k ∣ (m^2 - n^2)) ∧ (∀ j : ℤ, (∀ m n : ℤ, m % 2 = 1 → n % 2 = 1 → n < m → j ∣ (m^2 - n^2)) → j ≤ k) ∧ k = 8 :=
sorry

end largest_divisor_of_difference_of_squares_l58_58982


namespace count_total_shells_l58_58443

theorem count_total_shells 
  (purple_shells : ℕ := 13)
  (pink_shells : ℕ := 8)
  (yellow_shells : ℕ := 18)
  (blue_shells : ℕ := 12)
  (orange_shells : ℕ := 14) :
  purple_shells + pink_shells + yellow_shells + blue_shells + orange_shells = 65 :=
by
  -- Calculation
  sorry

end count_total_shells_l58_58443


namespace Ivan_can_safely_make_the_journey_l58_58790

def eruption_cycle_first_crater (t : ℕ) : Prop :=
  ∃ n : ℕ, t = 1 + 18 * n

def eruption_cycle_second_crater (t : ℕ) : Prop :=
  ∃ m : ℕ, t = 1 + 10 * m

def is_safe (start_time : ℕ) : Prop :=
  ∀ t, start_time ≤ t ∧ t < start_time + 16 → 
    ¬ eruption_cycle_first_crater t ∧ 
    ¬ (t ≥ start_time + 12 ∧ eruption_cycle_second_crater t)

theorem Ivan_can_safely_make_the_journey : ∃ t : ℕ, is_safe (38 + t) :=
sorry

end Ivan_can_safely_make_the_journey_l58_58790


namespace cubic_polynomial_solution_l58_58797

noncomputable def q (x : ℚ) : ℚ := (51/13) * x^3 + (-31/13) * x^2 + (16/13) * x + (3/13)

theorem cubic_polynomial_solution : 
  q 1 = 3 ∧ q 2 = 23 ∧ q 3 = 81 ∧ q 5 = 399 :=
by {
  sorry
}

end cubic_polynomial_solution_l58_58797


namespace tan_alpha_eq_one_l58_58515

open Real

theorem tan_alpha_eq_one (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h_cos_sin_eq : cos (α + β) = sin (α - β)) : tan α = 1 :=
by
  sorry

end tan_alpha_eq_one_l58_58515


namespace derivative_at_zero_l58_58880

-- Define the function f(x)
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

-- State the theorem with the given conditions and expected result
theorem derivative_at_zero :
  (deriv f 0 = -120) :=
by
  sorry

end derivative_at_zero_l58_58880


namespace arithmetic_geometric_seq_l58_58779

noncomputable def a (n : ℕ) : ℤ := 2 * n - 4 -- General form of the arithmetic sequence

def is_geometric_sequence (s : ℕ → ℤ) : Prop := 
  ∀ n : ℕ, (n > 1) → s (n+1) * s (n-1) = s n ^ 2

theorem arithmetic_geometric_seq:
  (∃ (d : ℤ) (a : ℕ → ℤ), a 5 = 6 ∧ 
  (∀ n, a n = 6 + (n - 5) * d) ∧ a (3) * a (11) = a (5) ^ 2 ∧
  (∀ k, 5 < k → is_geometric_sequence (fun n => a (k + n - 1)))) → 
  ∃ t : ℕ, ∀ n : ℕ, n <= 2015 → 
  (a n = 2 * n - 4 →  n = 7) := 
sorry

end arithmetic_geometric_seq_l58_58779


namespace least_possible_value_of_z_l58_58301

theorem least_possible_value_of_z (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : y - x > 5) 
  (h2 : z - x = 9) : 
  z = 11 := 
by
  sorry

end least_possible_value_of_z_l58_58301


namespace soja_book_page_count_l58_58508

theorem soja_book_page_count (P : ℕ) (h1 : P > 0) (h2 : (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 100) : P = 300 :=
by
  -- The Lean proof is not required, so we just add sorry to skip the proof
  sorry

end soja_book_page_count_l58_58508


namespace proof_problem_l58_58798

theorem proof_problem
  (x y z : ℤ)
  (h1 : x = 11 * y + 4)
  (h2 : 2 * x = 3 * y * z + 3)
  (h3 : 13 * y - x = 1) :
  z = 8 := by
  sorry

end proof_problem_l58_58798


namespace no_valid_two_digit_N_exists_l58_58645

def is_two_digit_number (N : ℕ) : Prop :=
  10 ≤ N ∧ N < 100

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ (n : ℕ), n ^ 3 = x

def reverse_digits (N : ℕ) : ℕ :=
  match N / 10, N % 10 with
  | a, b => 10 * b + a

theorem no_valid_two_digit_N_exists : ∀ N : ℕ,
  is_two_digit_number N →
  (is_perfect_cube (N - reverse_digits N) ∧ (N - reverse_digits N) ≠ 27) → false :=
by sorry

end no_valid_two_digit_N_exists_l58_58645


namespace proof_problem_l58_58574

-- Definitions based on the conditions from the problem
def optionA (A : Set α) : Prop := ∅ ∩ A = ∅

def optionC : Prop := { y | ∃ x, y = 1 / x } = { z | ∃ t, z = 1 / t }

-- The main theorem statement
theorem proof_problem (A : Set α) : optionA A ∧ optionC := by
  -- Placeholder for the proof
  sorry

end proof_problem_l58_58574


namespace students_second_scenario_l58_58728

def total_students (R : ℕ) : ℕ := 5 * R + 6
def effective_students (R : ℕ) : ℕ := 6 * (R - 3)
def filled_rows (R : ℕ) : ℕ := R - 3
def students_per_row := 6

theorem students_second_scenario:
  ∀ (R : ℕ), R = 24 → total_students R = effective_students R → students_per_row = 6
:= by
  intro R h_eq h_total_eq_effective
  -- Insert proof steps here
  sorry

end students_second_scenario_l58_58728


namespace geometric_mean_of_roots_l58_58254

theorem geometric_mean_of_roots (x : ℝ) (h : x^2 = (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1)) : x = 1 ∨ x = -1 := 
by
  sorry

end geometric_mean_of_roots_l58_58254


namespace boys_at_reunion_l58_58590

theorem boys_at_reunion (n : ℕ) (h : n * (n - 1) = 56) : n = 8 :=
sorry

end boys_at_reunion_l58_58590


namespace male_student_number_l58_58191

theorem male_student_number (year class_num student_num : ℕ) (h_year : year = 2011) (h_class : class_num = 6) (h_student : student_num = 23) : 
  (100000 * year + 1000 * class_num + 10 * student_num + 1 = 116231) :=
by
  sorry

end male_student_number_l58_58191


namespace simple_interest_double_in_4_years_interest_25_percent_l58_58964

theorem simple_interest_double_in_4_years_interest_25_percent :
  ∀ {P : ℕ} (h : P > 0), ∃ (R : ℕ), R = 25 ∧ P + P * R * 4 / 100 = 2 * P :=
by
  sorry

end simple_interest_double_in_4_years_interest_25_percent_l58_58964


namespace tom_watching_days_l58_58176

noncomputable def total_watch_time : ℕ :=
  30 * 22 + 28 * 25 + 27 * 29 + 20 * 31 + 25 * 27 + 20 * 35

noncomputable def daily_watch_time : ℕ := 2 * 60

theorem tom_watching_days : ⌈(total_watch_time / daily_watch_time : ℚ)⌉ = 35 := by
  sorry

end tom_watching_days_l58_58176


namespace probability_l58_58107

def total_chips : ℕ := 15
def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4

def probability_of_different_colors : ℚ :=
  (blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips) +
  (red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips) +
  (yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips)

theorem probability : probability_of_different_colors = 148 / 225 :=
by
  unfold probability_of_different_colors
  sorry

end probability_l58_58107


namespace intersection_eq_l58_58665

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_eq : M ∩ N = {5, 7, 9} := sorry

end intersection_eq_l58_58665


namespace lemuel_total_points_l58_58385

theorem lemuel_total_points (two_point_shots : ℕ) (three_point_shots : ℕ) (points_from_two : ℕ) (points_from_three : ℕ) :
  two_point_shots = 7 →
  three_point_shots = 3 →
  points_from_two = 2 →
  points_from_three = 3 →
  two_point_shots * points_from_two + three_point_shots * points_from_three = 23 :=
by
  sorry

end lemuel_total_points_l58_58385


namespace apples_in_basket_l58_58352

-- Define the conditions in Lean
def four_times_as_many_apples (O A : ℕ) : Prop :=
  A = 4 * O

def emiliano_consumes (O A : ℕ) : Prop :=
  (2/3 : ℚ) * O + (2/3 : ℚ) * A = 50

-- Formulate the main proposition to prove there are 60 apples
theorem apples_in_basket (O A : ℕ) (h1 : four_times_as_many_apples O A) (h2 : emiliano_consumes O A) : A = 60 := 
by
  sorry

end apples_in_basket_l58_58352


namespace probability_visible_l58_58309

-- Definitions of the conditions
def lap_time_sarah : ℕ := 120
def lap_time_sam : ℕ := 100
def start_to_photo_min : ℕ := 15
def start_to_photo_max : ℕ := 16
def photo_fraction : ℚ := 1/3
def shadow_start_interval : ℕ := 45
def shadow_duration : ℕ := 15

-- The theorem to prove
theorem probability_visible :
  let total_time := 60
  let valid_overlap_time := 13.33
  valid_overlap_time / total_time = 1333 / 6000 :=
by {
  sorry
}

end probability_visible_l58_58309


namespace find_original_number_l58_58498

theorem find_original_number (r : ℝ) (h : 1.15 * r - 0.7 * r = 40) : r = 88.88888888888889 :=
by
  sorry

end find_original_number_l58_58498


namespace bob_day3_miles_l58_58613

noncomputable def total_miles : ℕ := 70
noncomputable def day1_miles : ℕ := total_miles * 20 / 100
noncomputable def remaining_after_day1 : ℕ := total_miles - day1_miles
noncomputable def day2_miles : ℕ := remaining_after_day1 * 50 / 100
noncomputable def remaining_after_day2 : ℕ := remaining_after_day1 - day2_miles
noncomputable def day3_miles : ℕ := remaining_after_day2

theorem bob_day3_miles : day3_miles = 28 :=
by
  -- Insert proof here
  sorry

end bob_day3_miles_l58_58613


namespace rod_division_segments_l58_58050

theorem rod_division_segments (L : ℕ) (K : ℕ) (hL : L = 72 * K) :
  let red_divisions := 7
  let blue_divisions := 11
  let black_divisions := 17
  let overlap_9_6 := 4
  let overlap_6_4 := 6
  let overlap_9_4 := 2
  let overlap_all := 2
  let total_segments := red_divisions + blue_divisions + black_divisions - overlap_9_6 - overlap_6_4 - overlap_9_4 + overlap_all
  (total_segments = 28) ∧ ((L / 72) = K)
:=
by
  sorry

end rod_division_segments_l58_58050


namespace find_a_l58_58966

def new_operation (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a (a : ℝ) (b : ℝ) (h : b = 4) (h2 : new_operation a b = 10) : a = 14 := by
  have h' : new_operation a 4 = 10 := by rw [h] at h2; exact h2
  unfold new_operation at h'
  linarith

end find_a_l58_58966


namespace number_of_ways_to_select_president_and_vice_president_l58_58792

-- Define the given conditions
def num_candidates : Nat := 4

-- Define the problem to prove
theorem number_of_ways_to_select_president_and_vice_president : (num_candidates * (num_candidates - 1)) = 12 :=
by
  -- This is where the proof would go, but we are skipping it
  sorry

end number_of_ways_to_select_president_and_vice_president_l58_58792


namespace range_of_a_l58_58545

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x >= 2 then (a - 1 / 2) * x 
  else a^x - 4

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) ↔ (1 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l58_58545


namespace star_7_2_l58_58562

def star (a b : ℕ) := 4 * a - 4 * b

theorem star_7_2 : star 7 2 = 20 := 
by
  sorry

end star_7_2_l58_58562


namespace solve_trig_eqn_solution_set_l58_58234

theorem solve_trig_eqn_solution_set :
  {x : ℝ | ∃ k : ℤ, x = 3 * k * Real.pi + Real.pi / 4 ∨ x = 3 * k * Real.pi + 5 * Real.pi / 4} =
  {x : ℝ | 2 * Real.sin ((2 / 3) * x) = 1} :=
by
  sorry

end solve_trig_eqn_solution_set_l58_58234


namespace number_multiplies_xz_l58_58183

theorem number_multiplies_xz (x y z w A B : ℝ) (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  A * B = 4 :=
sorry

end number_multiplies_xz_l58_58183


namespace license_plate_count_l58_58796

def num_license_plates : Nat :=
  26 * 10 * 36

theorem license_plate_count : num_license_plates = 9360 :=
by
  sorry

end license_plate_count_l58_58796


namespace abigail_initial_money_l58_58974

variables (A : ℝ) -- Initial amount of money Abigail had.

-- Conditions
variables (food_rate : ℝ := 0.60) -- 60% spent on food
variables (phone_rate : ℝ := 0.25) -- 25% of the remainder spent on phone bill
variables (entertainment_spent : ℝ := 20) -- $20 spent on entertainment
variables (final_amount : ℝ := 40) -- $40 left after all expenditures

theorem abigail_initial_money :
  (A - (A * food_rate)) * (1 - phone_rate) - entertainment_spent = final_amount → A = 200 :=
by
  intro h
  sorry

end abigail_initial_money_l58_58974


namespace quadratic_root_value_l58_58648

theorem quadratic_root_value {m : ℝ} (h : m^2 + m - 1 = 0) : 2 * m^2 + 2 * m + 2025 = 2027 :=
sorry

end quadratic_root_value_l58_58648


namespace line_through_point_parallel_l58_58709

/-
Given the point P(2, 0) and a line x - 2y + 3 = 0,
prove that the equation of the line passing through 
P and parallel to the given line is 2y - x + 2 = 0.
-/
theorem line_through_point_parallel
  (P : ℝ × ℝ)
  (x y : ℝ)
  (line_eq : x - 2*y + 3 = 0)
  (P_eq : P = (2, 0)) :
  ∃ (a b c : ℝ), a * y - b * x + c = 0 :=
sorry

end line_through_point_parallel_l58_58709


namespace tan_alpha_l58_58950

theorem tan_alpha {α : ℝ} (h : 3 * Real.sin α + 4 * Real.cos α = 5) : Real.tan α = 3 / 4 :=
by
  -- Proof goes here
  sorry

end tan_alpha_l58_58950


namespace tax_free_amount_l58_58646

theorem tax_free_amount (X : ℝ) (total_value : ℝ) (tax_paid : ℝ) (tax_rate : ℝ) 
(h1 : total_value = 1720) 
(h2 : tax_paid = 134.4) 
(h3 : tax_rate = 0.12) 
(h4 : tax_paid = tax_rate * (total_value - X)) 
: X = 600 := 
sorry

end tax_free_amount_l58_58646


namespace Murtha_pebbles_l58_58408

-- Definition of the geometric series sum formula
noncomputable def sum_geometric_series (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

-- Constants for the problem
def a : ℕ := 1
def r : ℕ := 2
def n : ℕ := 10

-- The theorem to be proven
theorem Murtha_pebbles : sum_geometric_series a r n = 1023 :=
by
  -- Our condition setup implies the formula
  sorry

end Murtha_pebbles_l58_58408


namespace stickers_decorate_l58_58621

theorem stickers_decorate (initial_stickers bought_stickers birthday_stickers given_stickers remaining_stickers stickers_used : ℕ)
    (h1 : initial_stickers = 20)
    (h2 : bought_stickers = 12)
    (h3 : birthday_stickers = 20)
    (h4 : given_stickers = 5)
    (h5 : remaining_stickers = 39) :
    (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers = stickers_used) →
    stickers_used = 8 
:= by {sorry}

end stickers_decorate_l58_58621


namespace mean_eq_median_of_set_l58_58143

theorem mean_eq_median_of_set (x : ℕ) (hx : 0 < x) :
  let s := [1, 2, 4, 5, x]
  let mean := (1 + 2 + 4 + 5 + x) / 5
  let median := if x ≤ 2 then 2 else if x ≤ 4 then x else 4
  mean = median → (x = 3 ∨ x = 8) :=
by {
  sorry
}

end mean_eq_median_of_set_l58_58143


namespace Z_is_all_positive_integers_l58_58222

theorem Z_is_all_positive_integers (Z : Set ℕ) (h_nonempty : Z.Nonempty)
(h1 : ∀ x ∈ Z, 4 * x ∈ Z)
(h2 : ∀ x ∈ Z, (Nat.sqrt x) ∈ Z) : 
Z = { n : ℕ | n > 0 } :=
sorry

end Z_is_all_positive_integers_l58_58222


namespace answer_keys_count_l58_58564

theorem answer_keys_count 
  (test_questions : ℕ)
  (true_answers : ℕ)
  (false_answers : ℕ)
  (min_score : ℕ)
  (conditions : test_questions = 10 ∧ true_answers = 5 ∧ false_answers = 5 ∧ min_score >= 4) :
  ∃ (count : ℕ), count = 22 := by
  sorry

end answer_keys_count_l58_58564


namespace distance_between_P_and_Q_l58_58160

theorem distance_between_P_and_Q : 
  let initial_speed := 40  -- Speed in kmph
  let increment := 20      -- Speed increment in kmph after every 12 minutes
  let segment_duration := 12 / 60 -- Duration of each segment in hours (12 minutes in hours)
  let total_duration := 48 / 60    -- Total duration in hours (48 minutes in hours)
  let total_segments := total_duration / segment_duration -- Number of segments
  (total_segments = 4) ∧ 
  (∀ n : ℕ, n ≥ 0 → n < total_segments → 
    let speed := initial_speed + n * increment
    let distance := speed * segment_duration
    distance = speed * (12 / 60)) 
  → (40 * (12 / 60) + 60 * (12 / 60) + 80 * (12 / 60) + 100 * (12 / 60)) = 56 :=
by
  sorry

end distance_between_P_and_Q_l58_58160


namespace rectangle_ratio_l58_58193

theorem rectangle_ratio (s x y : ℝ) 
  (h1 : 4 * (x * y) + s^2 = 9 * s^2)
  (h2 : x + s = 3 * s)
  (h3 : s + 2 * y = 3 * s) :
  x / y = 2 :=
by
  sorry

end rectangle_ratio_l58_58193


namespace root_in_interval_l58_58998

noncomputable def f (x : ℝ) : ℝ := 3^x + 3*x - 8

theorem root_in_interval :
  f 1 < 0 ∧ f 1.5 > 0 ∧ f 1.25 < 0 → ∃ c, 1.25 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  sorry

end root_in_interval_l58_58998


namespace convert_neg_300_deg_to_rad_l58_58366

theorem convert_neg_300_deg_to_rad :
  -300 * (Real.pi / 180) = - (5 / 3) * Real.pi :=
by
  sorry

end convert_neg_300_deg_to_rad_l58_58366


namespace find_larger_number_l58_58077

theorem find_larger_number (x y : ℕ) (h1 : y - x = 1365) (h2 : y = 4 * x + 15) : y = 1815 :=
sorry

end find_larger_number_l58_58077


namespace simplify_fractional_expression_l58_58581

variable {a b c : ℝ}

theorem simplify_fractional_expression 
  (h_nonzero_a : a ≠ 0)
  (h_nonzero_b : b ≠ 0)
  (h_nonzero_c : c ≠ 0)
  (h_sum : a + b + c = 1) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 
  3 / (2 * (-b - c + b * c)) :=
sorry

end simplify_fractional_expression_l58_58581


namespace circles_C1_C2_intersect_C1_C2_l58_58322

noncomputable def center1 : (ℝ × ℝ) := (5, 3)
noncomputable def radius1 : ℝ := 3

noncomputable def center2 : (ℝ × ℝ) := (2, -1)
noncomputable def radius2 : ℝ := Real.sqrt 14

noncomputable def distance : ℝ := Real.sqrt ((5 - 2)^2 + (3 + 1)^2)

def circles_intersect : Prop :=
  radius2 - radius1 < distance ∧ distance < radius2 + radius1

theorem circles_C1_C2_intersect_C1_C2 : circles_intersect :=
by
  -- The proof of this theorem is to be worked out using the given conditions and steps.
  sorry

end circles_C1_C2_intersect_C1_C2_l58_58322


namespace plumber_total_cost_l58_58375

variable (copperLength : ℕ) (plasticLength : ℕ) (costPerMeter : ℕ)
variable (condition1 : copperLength = 10)
variable (condition2 : plasticLength = copperLength + 5)
variable (condition3 : costPerMeter = 4)

theorem plumber_total_cost (copperLength plasticLength costPerMeter : ℕ)
  (condition1 : copperLength = 10)
  (condition2 : plasticLength = copperLength + 5)
  (condition3 : costPerMeter = 4) :
  copperLength * costPerMeter + plasticLength * costPerMeter = 100 := by
  sorry

end plumber_total_cost_l58_58375


namespace juice_cost_l58_58690

theorem juice_cost (J : ℝ) (h1 : 15 * 3 + 25 * 1 + 12 * J = 88) : J = 1.5 :=
by
  sorry

end juice_cost_l58_58690


namespace game_show_prizes_count_l58_58416

theorem game_show_prizes_count:
  let digits := [1, 1, 1, 1, 3, 3, 3, 3]
  let is_valid_prize (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9999
  let is_three_digit_or_more (n : ℕ) : Prop := 100 ≤ n
  ∃ (A B C : ℕ), 
    is_valid_prize A ∧ is_valid_prize B ∧ is_valid_prize C ∧
    is_three_digit_or_more C ∧
    (A + B + C = digits.sum) ∧
    (A + B + C = 1260) := sorry

end game_show_prizes_count_l58_58416


namespace school_profit_calc_l58_58844

-- Definitions based on the conditions provided
def pizza_slices : Nat := 8
def slices_per_pizza : ℕ := 8
def slice_price : ℝ := 1.0 -- Defining price per slice
def pizzas_bought : ℕ := 55
def cost_per_pizza : ℝ := 6.85
def total_revenue : ℝ := pizzas_bought * slices_per_pizza * slice_price
def total_cost : ℝ := pizzas_bought * cost_per_pizza

-- The lean mathematical statement we need to prove
theorem school_profit_calc :
  total_revenue - total_cost = 63.25 := by
  sorry

end school_profit_calc_l58_58844


namespace percentage_less_than_l58_58082

theorem percentage_less_than (T F S : ℝ) 
  (hF : F = 0.70 * T) 
  (hS : S = 0.63 * T) : 
  ((T - S) / T) * 100 = 37 := 
by
  sorry

end percentage_less_than_l58_58082


namespace projectile_first_reaches_70_feet_l58_58605

theorem projectile_first_reaches_70_feet :
  ∃ t : ℝ , (t > 0) ∧ (-16 * t^2 + 80 * t = 70) ∧ (∀ t' : ℝ, (t' > 0) ∧ (-16 * t'^2 + 80 * t' = 70) → t ≤ t') :=
sorry

end projectile_first_reaches_70_feet_l58_58605


namespace gcd_8154_8640_l58_58059

theorem gcd_8154_8640 : Nat.gcd 8154 8640 = 6 := by
  sorry

end gcd_8154_8640_l58_58059


namespace clothing_probability_l58_58256

/-- I have a drawer with 6 shirts, 8 pairs of shorts, 7 pairs of socks, and 3 jackets in it.
    If I reach in and randomly remove four articles of clothing, what is the probability that 
    I get one shirt, one pair of shorts, one pair of socks, and one jacket? -/
theorem clothing_probability :
  let total_articles := 6 + 8 + 7 + 3
  let total_combinations := Nat.choose total_articles 4
  let favorable_combinations := 6 * 8 * 7 * 3
  (favorable_combinations : ℚ) / total_combinations = 144 / 1815 :=
by
  let total_articles := 6 + 8 + 7 + 3
  let total_combinations := Nat.choose total_articles 4
  let favorable_combinations := 6 * 8 * 7 * 3
  suffices (favorable_combinations : ℚ) / total_combinations = 144 / 1815
  by
    sorry
  sorry

end clothing_probability_l58_58256


namespace min_value_l58_58217

noncomputable def min_res (a b c : ℝ) : ℝ := 
  if h : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + 4 * b^2 + 9 * c^2 = 4 * b + 12 * c - 2) 
  then (1 / a + 2 / b + 3 / c) 
  else 0

theorem min_value (a b c : ℝ) : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + 4 * b^2 + 9 * c^2 = 4 * b + 12 * c - 2) 
    → min_res a b c = 6 := 
sorry

end min_value_l58_58217


namespace sum_of_roots_l58_58938

theorem sum_of_roots :
  let a := 1
  let b := 10
  let c := -25
  let sum_of_roots := -b / a
  (∀ x, 25 - 10 * x - x ^ 2 = 0 ↔ x ^ 2 + 10 * x - 25 = 0) →
  sum_of_roots = -10 :=
by
  intros
  sorry

end sum_of_roots_l58_58938


namespace arithmetic_sequence_solution_l58_58363

theorem arithmetic_sequence_solution (a : ℕ → ℝ) (q : ℝ) (S_n : ℕ → ℝ)
    (h1 : q > 0)
    (h2 : 2 * a 3 = a 5 - 3 * a 4) 
    (h3 : a 2 * a 4 * a 6 = 64) 
    (h4 : ∀ n, S_n n = (1 - q^n) / (1 - q) * a 1) :
    q = 2 ∧ (∀ n, S_n n = (2^n - 1) / 2) := 
  by
  sorry

end arithmetic_sequence_solution_l58_58363


namespace fraction_not_simplifiable_l58_58329

theorem fraction_not_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_not_simplifiable_l58_58329


namespace determine_k_a_l58_58916

theorem determine_k_a (k a : ℝ) (h : k - a ≠ 0) : (k = 0 ∧ a = 1 / 2) ↔ 
  (∀ x : ℝ, (x + 2) / (kx - ax - 1) = x → x = -2) :=
by
  sorry

end determine_k_a_l58_58916


namespace parabola_focus_l58_58387

theorem parabola_focus (a b c : ℝ) (h_eq : ∀ x : ℝ, 2 * x^2 + 8 * x - 1 = a * (x + b)^2 + c) :
  ∃ focus : ℝ × ℝ, focus = (-2, -71 / 8) :=
sorry

end parabola_focus_l58_58387


namespace yanna_kept_apples_l58_58560

-- Define the given conditions
def initial_apples : ℕ := 60
def percentage_given_to_zenny : ℝ := 0.40
def percentage_given_to_andrea : ℝ := 0.25

-- Prove the main statement
theorem yanna_kept_apples : 
  let apples_given_to_zenny := (percentage_given_to_zenny * initial_apples)
  let apples_remaining_after_zenny := (initial_apples - apples_given_to_zenny)
  let apples_given_to_andrea := (percentage_given_to_andrea * apples_remaining_after_zenny)
  let apples_kept := (apples_remaining_after_zenny - apples_given_to_andrea)
  apples_kept = 27 :=
by
  sorry

end yanna_kept_apples_l58_58560


namespace students_per_bus_l58_58341

def total_students : ℕ := 360
def number_of_buses : ℕ := 8

theorem students_per_bus : total_students / number_of_buses = 45 :=
by
  sorry

end students_per_bus_l58_58341


namespace Z_real_axis_Z_first_quadrant_Z_on_line_l58_58746

-- Definitions based on the problem conditions
def Z_real (m : ℝ) : ℝ := m^2 + 5*m + 6
def Z_imag (m : ℝ) : ℝ := m^2 - 2*m - 15

-- Lean statement for the equivalent proof problem

theorem Z_real_axis (m : ℝ) :
  Z_imag m = 0 ↔ (m = -3 ∨ m = 5) := sorry

theorem Z_first_quadrant (m : ℝ) :
  (Z_real m > 0 ∧ Z_imag m > 0) ↔ (m > 5) := sorry

theorem Z_on_line (m : ℝ) :
  (Z_real m + Z_imag m + 5 = 0) ↔ (m = (-5 + Real.sqrt 41) / 2) := sorry

end Z_real_axis_Z_first_quadrant_Z_on_line_l58_58746


namespace simplify_expression_l58_58320

theorem simplify_expression :
  (16 / 54) * (27 / 8) * (64 / 81) = 64 / 9 :=
by sorry

end simplify_expression_l58_58320


namespace implicit_derivative_l58_58026

noncomputable section

open Real

section ImplicitDifferentiation

variable {x : ℝ} {y : ℝ → ℝ}

def f (x y : ℝ) : ℝ := y^2 + x^2 - 1

theorem implicit_derivative (h : f x (y x) = 0) :
  deriv y x = -x / y x :=
  sorry

end ImplicitDifferentiation

end implicit_derivative_l58_58026


namespace toothpaste_last_day_l58_58660

theorem toothpaste_last_day (total_toothpaste : ℝ)
  (dad_use_per_brush : ℝ) (dad_brushes_per_day : ℕ)
  (mom_use_per_brush : ℝ) (mom_brushes_per_day : ℕ)
  (anne_use_per_brush : ℝ) (anne_brushes_per_day : ℕ)
  (brother_use_per_brush : ℝ) (brother_brushes_per_day : ℕ)
  (sister_use_per_brush : ℝ) (sister_brushes_per_day : ℕ)
  (grandfather_use_per_brush : ℝ) (grandfather_brushes_per_day : ℕ)
  (guest_use_per_brush : ℝ) (guest_brushes_per_day : ℕ) (guest_days : ℕ)
  (total_usage_per_day : ℝ) :
  total_toothpaste = 80 →
  dad_use_per_brush * dad_brushes_per_day = 16 →
  mom_use_per_brush * mom_brushes_per_day = 12 →
  anne_use_per_brush * anne_brushes_per_day = 8 →
  brother_use_per_brush * brother_brushes_per_day = 4 →
  sister_use_per_brush * sister_brushes_per_day = 2 →
  grandfather_use_per_brush * grandfather_brushes_per_day = 6 →
  guest_use_per_brush * guest_brushes_per_day * guest_days = 6 * 4 →
  total_usage_per_day = 54 →
  80 / 54 = 1 → 
  total_toothpaste / total_usage_per_day = 1 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end toothpaste_last_day_l58_58660


namespace glove_ratio_l58_58640

theorem glove_ratio (P : ℕ) (G : ℕ) (hf : P = 43) (hg : G = 2 * P) : G / P = 2 := by
  rw [hf, hg]
  norm_num
  sorry

end glove_ratio_l58_58640


namespace total_bottle_caps_in_collection_l58_58436

-- Statements of given conditions
def small_box_caps : ℕ := 35
def large_box_caps : ℕ := 75
def num_small_boxes : ℕ := 7
def num_large_boxes : ℕ := 3
def individual_caps : ℕ := 23

-- Theorem statement that needs to be proved
theorem total_bottle_caps_in_collection :
  small_box_caps * num_small_boxes + large_box_caps * num_large_boxes + individual_caps = 493 :=
by sorry

end total_bottle_caps_in_collection_l58_58436


namespace smallest_s_triangle_l58_58782

theorem smallest_s_triangle (s : ℕ) :
  (7 + s > 11) ∧ (7 + 11 > s) ∧ (11 + s > 7) → s = 5 :=
sorry

end smallest_s_triangle_l58_58782


namespace sufficient_condition_for_A_l58_58182

variables {A B C : Prop}

theorem sufficient_condition_for_A (h1 : A ↔ B) (h2 : C → B) : C → A :=
sorry

end sufficient_condition_for_A_l58_58182


namespace cos_three_theta_l58_58929

open Complex

theorem cos_three_theta (θ : ℝ) (h : cos θ = 1 / 2) : cos (3 * θ) = -1 / 2 :=
by
  sorry

end cos_three_theta_l58_58929


namespace smallest_interior_angle_l58_58206

open Real

theorem smallest_interior_angle (A B C : ℝ) (hA : 0 < A ∧ A < π)
    (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π)
    (h_sum_angles : A + B + C = π)
    (h_ratio : sin A / sin B = 2 / sqrt 6 ∧ sin A / sin C = 2 / (sqrt 3 + 1)) :
    min A (min B C) = π / 4 := 
  by sorry

end smallest_interior_angle_l58_58206


namespace coin_flip_probability_l58_58104

theorem coin_flip_probability :
  let total_outcomes := 2^5
  let successful_outcomes := 2 * 2^2
  total_outcomes > 0 → (successful_outcomes / total_outcomes) = (1 / 4) :=
by
  intros
  sorry

end coin_flip_probability_l58_58104


namespace tetrahedron_pairs_l58_58422

theorem tetrahedron_pairs (tetra_edges : ℕ) (h_tetra : tetra_edges = 6) :
  ∀ (num_pairs : ℕ), num_pairs = (tetra_edges * (tetra_edges - 1)) / 2 → num_pairs = 15 :=
by
  sorry

end tetrahedron_pairs_l58_58422


namespace total_dollar_amount_l58_58644

/-- Definitions of base 5 numbers given in the problem -/
def pearls := 1 * 5^0 + 2 * 5^1 + 3 * 5^2 + 4 * 5^3
def silk := 1 * 5^0 + 1 * 5^1 + 1 * 5^2 + 1 * 5^3
def spices := 1 * 5^0 + 2 * 5^1 + 2 * 5^2
def maps := 0 * 5^0 + 1 * 5^1

/-- The theorem to prove the total dollar amount in base 10 -/
theorem total_dollar_amount : pearls + silk + spices + maps = 808 :=
by
  sorry

end total_dollar_amount_l58_58644


namespace annie_journey_time_l58_58485

noncomputable def total_time_journey (walk_speed1 bus_speed train_speed walk_speed2 blocks_walk1 blocks_bus blocks_train blocks_walk2 : ℝ) : ℝ :=
  let time_walk1 := blocks_walk1 / walk_speed1
  let time_bus := blocks_bus / bus_speed
  let time_train := blocks_train / train_speed
  let time_walk2 := blocks_walk2 / walk_speed2
  let time_back := time_walk2
  time_walk1 + time_bus + time_train + time_walk2 + time_back + time_train + time_bus + time_walk1

theorem annie_journey_time :
  total_time_journey 2 4 5 2 5 7 10 4 = 16.5 := by 
  sorry

end annie_journey_time_l58_58485


namespace question_solution_l58_58114

variable (a b : ℝ)

theorem question_solution : 2 * a - 3 * (a - b) = -a + 3 * b := by
  sorry

end question_solution_l58_58114


namespace tabitha_honey_nights_l58_58117

def servings_per_cup := 1
def cups_per_night := 2
def ounces_per_container := 16
def servings_per_ounce := 6
def total_servings := servings_per_ounce * ounces_per_container
def servings_per_night := servings_per_cup * cups_per_night
def number_of_nights := total_servings / servings_per_night

theorem tabitha_honey_nights : number_of_nights = 48 :=
by
  -- Proof to be provided.
  sorry

end tabitha_honey_nights_l58_58117


namespace meet_time_correct_l58_58045

variable (circumference : ℕ) (speed_yeonjeong speed_donghun : ℕ)

def meet_time (circumference speed_yeonjeong speed_donghun : ℕ) : ℕ :=
  circumference / (speed_yeonjeong + speed_donghun)

theorem meet_time_correct
  (h_circumference : circumference = 3000)
  (h_speed_yeonjeong : speed_yeonjeong = 100)
  (h_speed_donghun : speed_donghun = 150) :
  meet_time circumference speed_yeonjeong speed_donghun = 12 :=
by
  rw [h_circumference, h_speed_yeonjeong, h_speed_donghun]
  norm_num
  sorry

end meet_time_correct_l58_58045


namespace edric_hourly_rate_l58_58230

theorem edric_hourly_rate
  (monthly_salary : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (H1 : monthly_salary = 576)
  (H2 : hours_per_day = 8)
  (H3 : days_per_week = 6)
  (H4 : weeks_per_month = 4) :
  monthly_salary / weeks_per_month / days_per_week / hours_per_day = 3 := by
  sorry

end edric_hourly_rate_l58_58230


namespace total_training_hours_l58_58011

-- Define Thomas's training conditions
def hours_per_day := 5
def days_initial := 30
def days_additional := 12
def total_days := days_initial + days_additional

-- State the theorem to be proved
theorem total_training_hours : total_days * hours_per_day = 210 :=
by
  sorry

end total_training_hours_l58_58011


namespace max_abs_sum_l58_58372

theorem max_abs_sum (a b c : ℝ) (h : ∀ x, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  |a| + |b| + |c| ≤ 3 :=
sorry

end max_abs_sum_l58_58372


namespace max_b_lattice_free_line_l58_58253

theorem max_b_lattice_free_line : 
  ∃ b : ℚ, (∀ (m : ℚ), (1 / 3) < m ∧ m < b → 
  ∀ x : ℤ, 0 < x ∧ x ≤ 150 → ¬ (∃ y : ℤ, y = m * x + 4)) ∧ 
  b = 50 / 147 :=
sorry

end max_b_lattice_free_line_l58_58253


namespace stratified_sampling_total_results_l58_58906

theorem stratified_sampling_total_results :
  let junior_students := 400
  let senior_students := 200
  let total_students_to_sample := 60
  let junior_sample := 40
  let senior_sample := 20
  (Nat.choose junior_students junior_sample) * (Nat.choose senior_students senior_sample) = Nat.choose 400 40 * Nat.choose 200 20 :=
  sorry

end stratified_sampling_total_results_l58_58906


namespace inequality_am_gm_l58_58641

theorem inequality_am_gm (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 := by
  sorry

end inequality_am_gm_l58_58641


namespace range_of_m_l58_58136

variable (m : ℝ)

def p : Prop := ∀ x : ℝ, 2 * x > m * (x^2 + 1)
def q : Prop := ∃ x0 : ℝ, x0^2 + 2 * x0 - m - 1 = 0

theorem range_of_m (hp : p m) (hq : q m) : -2 ≤ m ∧ m < -1 :=
sorry

end range_of_m_l58_58136


namespace remainder_of_x_l58_58111

theorem remainder_of_x (x : ℤ) (h : 2 * x - 3 = 7) : x % 2 = 1 := by
  sorry

end remainder_of_x_l58_58111


namespace graph_intersect_points_l58_58482

-- Define f as a function defined on all real numbers and invertible
variable (f : ℝ → ℝ) (hf : Function.Injective f)

-- Define the theorem to find the number of intersection points
theorem graph_intersect_points : 
  ∃ (n : ℕ), n = 3 ∧ ∃ (x : ℝ), (f (x^2) = f (x^6)) :=
  by
    -- Outline sketch: We aim to show there are 3 real solutions satisfying the equation
    -- The proof here is skipped, hence we put sorry
    sorry

end graph_intersect_points_l58_58482


namespace vehicles_travelled_last_year_l58_58580

theorem vehicles_travelled_last_year (V : ℕ) : 
  (∀ (x : ℕ), (96 : ℕ) * (V / 100000000) = 2880) → V = 3000000000 := 
by 
  sorry

end vehicles_travelled_last_year_l58_58580


namespace rows_of_roses_l58_58898

variable (rows total_roses_per_row roses_per_row_red roses_per_row_non_red roses_per_row_white roses_per_row_pink total_pink_roses : ℕ)
variable (half_two_fifth three_fifth : ℚ)

-- Assume the conditions
axiom h1 : total_roses_per_row = 20
axiom h2 : roses_per_row_red = total_roses_per_row / 2
axiom h3 : roses_per_row_non_red = total_roses_per_row - roses_per_row_red
axiom h4 : roses_per_row_white = (3 / 5 : ℚ) * roses_per_row_non_red
axiom h5 : roses_per_row_pink = (2 / 5 : ℚ) * roses_per_row_non_red
axiom h6 : total_pink_roses = 40

-- Prove the number of rows in the garden
theorem rows_of_roses : rows = total_pink_roses / (roses_per_row_pink) :=
by
  sorry

end rows_of_roses_l58_58898


namespace fraction_of_40_l58_58983

theorem fraction_of_40 : (3 / 4) * 40 = 30 :=
by
  -- We'll add the 'sorry' here to indicate that this is the proof part which is not required.
  sorry

end fraction_of_40_l58_58983


namespace correct_operation_l58_58576

theorem correct_operation :
  (∀ a : ℝ, (a^5 * a^3 = a^15) = false) ∧
  (∀ a : ℝ, (a^5 - a^3 = a^2) = false) ∧
  (∀ a : ℝ, ((-a^5)^2 = a^10) = true) ∧
  (∀ a : ℝ, (a^6 / a^3 = a^2) = false) :=
by
  sorry

end correct_operation_l58_58576


namespace galya_overtakes_sasha_l58_58643

variable {L : ℝ} -- Length of the track
variable (Sasha_uphill_speed : ℝ := 8)
variable (Sasha_downhill_speed : ℝ := 24)
variable (Galya_uphill_speed : ℝ := 16)
variable (Galya_downhill_speed : ℝ := 18)

noncomputable def average_speed (uphill_speed: ℝ) (downhill_speed: ℝ) : ℝ :=
  1 / ((1 / (4 * uphill_speed)) + (3 / (4 * downhill_speed)))

noncomputable def time_for_one_lap (L: ℝ) (speed: ℝ) : ℝ :=
  L / speed

theorem galya_overtakes_sasha 
  (L_pos : 0 < L) :
  let v_Sasha := average_speed Sasha_uphill_speed Sasha_downhill_speed
  let v_Galya := average_speed Galya_uphill_speed Galya_downhill_speed
  let t_Sasha := time_for_one_lap L v_Sasha
  let t_Galya := time_for_one_lap L v_Galya
  (L * 11 / v_Galya) < (L * 10 / v_Sasha) :=
by
  sorry

end galya_overtakes_sasha_l58_58643


namespace saree_sale_price_l58_58448

def initial_price : Real := 150
def discount1 : Real := 0.20
def tax1 : Real := 0.05
def discount2 : Real := 0.15
def tax2 : Real := 0.04
def discount3 : Real := 0.10
def tax3 : Real := 0.03
def final_price : Real := 103.25

theorem saree_sale_price :
  let price_after_discount1 : Real := initial_price * (1 - discount1)
  let price_after_tax1 : Real := price_after_discount1 * (1 + tax1)
  let price_after_discount2 : Real := price_after_tax1 * (1 - discount2)
  let price_after_tax2 : Real := price_after_discount2 * (1 + tax2)
  let price_after_discount3 : Real := price_after_tax2 * (1 - discount3)
  let price_after_tax3 : Real := price_after_discount3 * (1 + tax3)
  abs (price_after_tax3 - final_price) < 0.01 :=
by
  sorry

end saree_sale_price_l58_58448


namespace express_y_in_terms_of_x_l58_58268

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + 3 * y = 4) : y = (4 - 2 * x) / 3 := 
by
  sorry

end express_y_in_terms_of_x_l58_58268


namespace emily_spending_l58_58369

theorem emily_spending : ∀ {x : ℝ}, (x + 2 * x + 3 * x = 120) → (x = 20) :=
by
  intros x h
  sorry

end emily_spending_l58_58369


namespace cube_surface_area_sum_of_edges_l58_58339

noncomputable def edge_length (sum_of_edges : ℝ) (num_of_edges : ℝ) : ℝ :=
  sum_of_edges / num_of_edges

noncomputable def surface_area (edge_length : ℝ) : ℝ :=
  6 * edge_length ^ 2

theorem cube_surface_area_sum_of_edges (sum_of_edges : ℝ) (num_of_edges : ℝ) (expected_area : ℝ) :
  num_of_edges = 12 → sum_of_edges = 72 → surface_area (edge_length sum_of_edges num_of_edges) = expected_area :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end cube_surface_area_sum_of_edges_l58_58339


namespace servant_position_for_28_purses_servant_position_for_27_purses_l58_58818

-- Definitions based on problem conditions
def total_wealthy_men: ℕ := 7

def valid_purse_placement (n: ℕ): Prop := 
  (n ≤ total_wealthy_men * (total_wealthy_men + 1) / 2)

def get_servant_position (n: ℕ): ℕ := 
  if n = 28 then total_wealthy_men else if n = 27 then 6 else 0

-- Proof statements to equate conditions with the answers
theorem servant_position_for_28_purses : 
  get_servant_position 28 = 7 :=
sorry

theorem servant_position_for_27_purses : 
  get_servant_position 27 = 6 ∨ get_servant_position 27 = 7 :=
sorry

end servant_position_for_28_purses_servant_position_for_27_purses_l58_58818


namespace maximum_value_a3_b3_c3_d3_l58_58672

noncomputable def max_value (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

theorem maximum_value_a3_b3_c3_d3
  (a b c d : ℝ)
  (h1 : a^2 + b^2 + c^2 + d^2 = 20)
  (h2 : a + b + c + d = 10) :
  max_value a b c d ≤ 500 :=
sorry

end maximum_value_a3_b3_c3_d3_l58_58672


namespace tennis_racket_price_l58_58941

theorem tennis_racket_price (P : ℝ) : 
    (0.8 * P + 515) * 1.10 + 20 = 800 → 
    P = 242.61 :=
by
  sorry

end tennis_racket_price_l58_58941


namespace non_shaded_area_l58_58291

theorem non_shaded_area (s : ℝ) (hex_area : ℝ) (tri_area : ℝ) (non_shaded_area : ℝ) :
  s = 12 →
  hex_area = (3 * Real.sqrt 3 / 2) * s^2 →
  tri_area = (Real.sqrt 3 / 4) * (2 * s)^2 →
  non_shaded_area = hex_area - tri_area →
  non_shaded_area = 288 * Real.sqrt 3 :=
by
  intros hs hhex htri hnon
  sorry

end non_shaded_area_l58_58291


namespace abs_lt_2_sufficient_not_necessary_l58_58860

theorem abs_lt_2_sufficient_not_necessary (x : ℝ) :
  (|x| < 2 → x^2 - x - 6 < 0) ∧ ¬ (x^2 - x - 6 < 0 → |x| < 2) :=
by {
  sorry
}

end abs_lt_2_sufficient_not_necessary_l58_58860


namespace shinyoung_initial_candies_l58_58492

theorem shinyoung_initial_candies : 
  ∀ (C : ℕ), 
    (C / 2) - ((C / 6) + 5) = 5 → 
    C = 30 := by
  intros C h
  sorry

end shinyoung_initial_candies_l58_58492


namespace cube_root_of_neg_125_l58_58404

theorem cube_root_of_neg_125 : (-5)^3 = -125 := 
by sorry

end cube_root_of_neg_125_l58_58404


namespace daisy_dog_toys_l58_58619

-- Given conditions
def dog_toys_monday : ℕ := 5
def dog_toys_tuesday_left : ℕ := 3
def dog_toys_tuesday_bought : ℕ := 3
def dog_toys_wednesday_all_found : ℕ := 13

-- The question we need to answer
def dog_toys_bought_wednesday : ℕ := 7

-- Statement to prove
theorem daisy_dog_toys :
  (dog_toys_monday - dog_toys_tuesday_left + dog_toys_tuesday_left + dog_toys_tuesday_bought + dog_toys_bought_wednesday = dog_toys_wednesday_all_found) :=
sorry

end daisy_dog_toys_l58_58619


namespace christine_wander_time_l58_58697

-- Definitions based on conditions
def distance : ℝ := 50.0
def speed : ℝ := 6.0

-- The statement to prove
theorem christine_wander_time : (distance / speed) = 8 + 20/60 :=
by
  sorry

end christine_wander_time_l58_58697


namespace range_of_a_l58_58415

variable (x a : ℝ)

-- Definition of α: x > a
def α : Prop := x > a

-- Definition of β: (x - 1) / x > 0
def β : Prop := (x - 1) / x > 0

-- Theorem to prove the range of a
theorem range_of_a (h : α x a → β x) : 1 ≤ a :=
  sorry

end range_of_a_l58_58415


namespace find_m_l58_58087

theorem find_m (x y m : ℝ) (hx : x = 1) (hy : y = 2) (h : m * x + 2 * y = 6) : m = 2 :=
by sorry

end find_m_l58_58087


namespace larger_angle_of_nonagon_l58_58282

theorem larger_angle_of_nonagon : 
  ∀ (n : ℕ) (x : ℝ), 
  n = 9 → 
  (∃ a b : ℕ, a + b = n ∧ a * x + b * (3 * x) = 180 * (n - 2)) → 
  3 * (180 * (n - 2) / 15) = 252 :=
by
  sorry

end larger_angle_of_nonagon_l58_58282


namespace smallest_possible_sum_l58_58915

-- Defining the conditions for x and y.
variables (x y : ℕ)

-- We need a theorem to formalize our question with the given conditions.
theorem smallest_possible_sum (hx : x > 0) (hy : y > 0) (hne : x ≠ y) (hxy : 1/x + 1/y = 1/24) : x + y = 100 :=
by
  sorry

end smallest_possible_sum_l58_58915


namespace people_per_pizza_l58_58080

def pizza_cost := 12 -- dollars per pizza
def babysitting_earnings_per_night := 4 -- dollars per night
def nights_babysitting := 15
def total_people := 15

theorem people_per_pizza : (babysitting_earnings_per_night * nights_babysitting / pizza_cost) = (total_people / ((babysitting_earnings_per_night * nights_babysitting / pizza_cost))) := 
by
  sorry

end people_per_pizza_l58_58080


namespace total_length_of_sticks_l58_58396

-- Definitions of stick lengths based on the conditions
def length_first_stick : ℕ := 3
def length_second_stick : ℕ := 2 * length_first_stick
def length_third_stick : ℕ := length_second_stick - 1

-- Proof statement
theorem total_length_of_sticks : length_first_stick + length_second_stick + length_third_stick = 14 :=
by
  sorry

end total_length_of_sticks_l58_58396


namespace school_badminton_rackets_l58_58207

theorem school_badminton_rackets :
  ∃ (x y : ℕ), x + y = 30 ∧ 50 * x + 40 * y = 1360 ∧ x = 16 ∧ y = 14 :=
by
  sorry

end school_badminton_rackets_l58_58207


namespace find_n_l58_58221

theorem find_n (x y : ℝ) (n : ℝ) (h1 : x / (2 * y) = 3 / n) (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) : n = 2 := by
  sorry

end find_n_l58_58221


namespace mod_neg_result_l58_58007

-- Define the hypothesis as the residue equivalence and positive range constraint.
theorem mod_neg_result : 
  ∀ (a b : ℤ), (-1277 : ℤ) % 32 = 3 := by
  sorry

end mod_neg_result_l58_58007


namespace inequality_not_always_hold_l58_58018

theorem inequality_not_always_hold (a b : ℝ) (h : a > -b) : ¬ (∀ a b : ℝ, a > -b → (1 / a + 1 / b > 0)) :=
by
  intro h2
  have h3 := h2 a b h
  sorry

end inequality_not_always_hold_l58_58018


namespace figure_surface_area_calculation_l58_58801

-- Define the surface area of one bar
def bar_surface_area : ℕ := 18

-- Define the surface area lost at the junctions
def surface_area_lost : ℕ := 2

-- Define the effective surface area of one bar after accounting for overlaps
def effective_bar_surface_area : ℕ := bar_surface_area - surface_area_lost

-- Define the number of bars used in the figure
def number_of_bars : ℕ := 4

-- Define the total surface area of the figure
def total_surface_area : ℕ := number_of_bars * effective_bar_surface_area

-- The theorem stating the total surface area of the figure
theorem figure_surface_area_calculation : total_surface_area = 64 := by
  sorry

end figure_surface_area_calculation_l58_58801


namespace sufficient_but_not_necessary_decreasing_l58_58701

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f y ≤ f x

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 6 * m * x + 6

theorem sufficient_but_not_necessary_decreasing (m : ℝ) :
  m = 1 → is_decreasing_on (f m) (Set.Iic 3) :=
by
  intros h
  rw [h]
  sorry

end sufficient_but_not_necessary_decreasing_l58_58701


namespace elgin_money_l58_58462

theorem elgin_money {A B C D E : ℤ} 
  (h1 : |A - B| = 19) 
  (h2 : |B - C| = 9) 
  (h3 : |C - D| = 5) 
  (h4 : |D - E| = 4) 
  (h5 : |E - A| = 11) 
  (h6 : A + B + C + D + E = 60) : 
  E = 10 := 
sorry

end elgin_money_l58_58462


namespace shifted_parabola_sum_l58_58428

theorem shifted_parabola_sum (a b c : ℝ) :
  (∃ (a b c : ℝ), ∀ x : ℝ, 3 * x^2 + 2 * x - 5 = 3 * (x - 6)^2 + 2 * (x - 6) - 5 → y = a * x^2 + b * x + c) → a + b + c = 60 :=
sorry

end shifted_parabola_sum_l58_58428


namespace number_of_integers_between_cubed_values_l58_58522

theorem number_of_integers_between_cubed_values :
  ∃ n : ℕ, n = (1278 - 1122 + 1) ∧ 
  ∀ x : ℤ, (1122 < x ∧ x < 1278) → (1123 ≤ x ∧ x ≤ 1277) := 
by
  sorry

end number_of_integers_between_cubed_values_l58_58522


namespace buns_problem_l58_58113

theorem buns_problem (N : ℕ) (x y u v : ℕ) 
  (h1 : 3 * x + 5 * y = 25)
  (h2 : 3 * u + 5 * v = 35)
  (h3 : x + y = N)
  (h4 : u + v = N) : 
  N = 7 := 
sorry

end buns_problem_l58_58113


namespace find_second_sum_l58_58354

theorem find_second_sum (x : ℝ) (total_sum : ℝ) (h : total_sum = 2691) 
  (h1 : (24 * x) / 100 = 15 * (total_sum - x) / 100) : total_sum - x = 1656 :=
by
  sorry

end find_second_sum_l58_58354


namespace find_radius_l58_58597

def setA : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def setB (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

theorem find_radius (r : ℝ) (h : r > 0) :
  (∃! p : ℝ × ℝ, p ∈ setA ∧ p ∈ setB r) ↔ (r = 3 ∨ r = 7) :=
by
  sorry

end find_radius_l58_58597


namespace total_bananas_in_collection_l58_58123

theorem total_bananas_in_collection (g b T : ℕ) (h₀ : g = 196) (h₁ : b = 2) (h₂ : T = 392) : g * b = T :=
by
  sorry

end total_bananas_in_collection_l58_58123


namespace range_of_c_l58_58365

theorem range_of_c (c : ℝ) :
  (c^2 - 5 * c + 7 > 1 ∧ (|2 * c - 1| ≤ 1)) ∨ ((c^2 - 5 * c + 7 ≤ 1) ∧ |2 * c - 1| > 1) ↔ (0 ≤ c ∧ c ≤ 1) ∨ (2 ≤ c ∧ c ≤ 3) :=
sorry

end range_of_c_l58_58365


namespace max_sum_unit_hexagons_l58_58057

theorem max_sum_unit_hexagons (k : ℕ) (hk : k ≥ 3) : 
  ∃ S, S = 6 + (3 * k - 9) * k * (k + 1) / 2 + (3 * (k^2 - 2)) * (k * (k + 1) * (2 * k + 1) / 6) / 6 ∧
       S = 3 * (k * k - 14 * k + 33 * k - 28) / 2 :=
by
  sorry

end max_sum_unit_hexagons_l58_58057


namespace solution_set_f_l58_58122

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^(x - 1) - 2 else 2^(1 - x) - 2

theorem solution_set_f (x : ℝ) : 
  (1 ≤ x ∧ x ≤ 3) ↔ (f (x - 1) ≤ 0) :=
sorry

end solution_set_f_l58_58122


namespace new_rectangle_area_l58_58451

theorem new_rectangle_area (a b : ℝ) : 
  let base := b + 2 * a
  let height := b - a
  let area := base * height
  area = b^2 + b * a - 2 * a^2 :=
by
  let base := b + 2 * a
  let height := b - a
  let area := base * height
  show area = b^2 + b * a - 2 * a^2
  sorry

end new_rectangle_area_l58_58451


namespace manicure_cost_l58_58683

noncomputable def cost_of_manicure : ℝ := 30

theorem manicure_cost
    (cost_hair_updo : ℝ)
    (total_cost_with_tips : ℝ)
    (tip_rate : ℝ)
    (M : ℝ) :
  cost_hair_updo = 50 →
  total_cost_with_tips = 96 →
  tip_rate = 0.20 →
  (cost_hair_updo + M + tip_rate * cost_hair_updo + tip_rate * M = total_cost_with_tips) →
  M = cost_of_manicure :=
by
  intros h1 h2 h3 h4
  sorry

end manicure_cost_l58_58683


namespace break_even_production_volume_l58_58288

theorem break_even_production_volume :
  ∃ Q : ℝ, 300 = 100 + 100000 / Q ∧ Q = 500 :=
by
  use 500
  sorry

end break_even_production_volume_l58_58288


namespace satisfy_third_eq_l58_58297

theorem satisfy_third_eq 
  (x y : ℝ) 
  (h1 : x^2 - 3 * x * y + 2 * y^2 + x - y = 0)
  (h2 : x^2 - 2 * x * y + y^2 - 5 * x + 7 * y = 0) 
  : x * y - 12 * x + 15 * y = 0 :=
by
  sorry

end satisfy_third_eq_l58_58297


namespace find_daily_rate_of_first_company_l58_58008

-- Define the daily rate of the first car rental company
def daily_rate_first_company (x : ℝ) : ℝ :=
  x + 0.18 * 48.0

-- Define the total cost for City Rentals
def total_cost_city_rentals : ℝ :=
  18.95 + 0.16 * 48.0

-- Prove the daily rate of the first car rental company
theorem find_daily_rate_of_first_company (x : ℝ) (h : daily_rate_first_company x = total_cost_city_rentals) : 
  x = 17.99 := 
by
  sorry

end find_daily_rate_of_first_company_l58_58008


namespace domain_of_f_l58_58706

theorem domain_of_f : 
  ∀ x, (2 - x ≥ 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x ≤ 2) := by
  sorry

end domain_of_f_l58_58706


namespace determine_a_l58_58546

-- Define the sets A and B
def A : Set ℝ := { -1, 0, 2 }
def B (a : ℝ) : Set ℝ := { 2^a }

-- State the main theorem
theorem determine_a (a : ℝ) (h : B a ⊆ A) : a = 1 :=
by
  sorry

end determine_a_l58_58546


namespace smallest_n_terminating_decimal_l58_58051

-- Define the given condition: n + 150 must be expressible as 2^a * 5^b.
def has_terminating_decimal_property (n : ℕ) := ∃ a b : ℕ, n + 150 = 2^a * 5^b

-- We want to prove that the smallest n satisfying the property is 50.
theorem smallest_n_terminating_decimal :
  (∀ n : ℕ, n > 0 ∧ has_terminating_decimal_property n → n ≥ 50) ∧ (has_terminating_decimal_property 50) :=
by
  sorry

end smallest_n_terminating_decimal_l58_58051


namespace range_of_m_for_distance_l58_58046

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (|x1 - x2|) + 2 * (|y1 - y2|)

theorem range_of_m_for_distance (m : ℝ) : 
  distance 2 1 (-1) m ≤ 5 ↔ 0 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_for_distance_l58_58046


namespace counterexample_to_conjecture_l58_58236

theorem counterexample_to_conjecture (n : ℕ) (h : n > 5) : 
  ¬ (∃ a b c : ℕ, (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (a + b + c = n)) ∨
  ¬ (∃ a b c : ℕ, (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (a + b + c = n)) :=
sorry

end counterexample_to_conjecture_l58_58236


namespace find_k_l58_58901

theorem find_k (m n k : ℝ) (h1 : m = 2 * n + 5) (h2 : m + 3 = 2 * (n + k) + 5) : k = 3 / 2 := 
by 
  sorry

end find_k_l58_58901


namespace polygon_area_correct_l58_58995

noncomputable def polygonArea : ℝ :=
  let x1 := 1
  let y1 := 1
  let x2 := 4
  let y2 := 3
  let x3 := 5
  let y3 := 1
  let x4 := 6
  let y4 := 4
  let x5 := 3
  let y5 := 6
  (1 / 2 : ℝ) * 
  abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y5 + x5 * y1) -
       (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x5 + y5 * x1))

theorem polygon_area_correct : polygonArea = 11.5 := by
  sorry

end polygon_area_correct_l58_58995


namespace find_power_of_7_l58_58918

theorem find_power_of_7 :
  (7^(1/4)) / (7^(1/6)) = 7^(1/12) :=
by
  sorry

end find_power_of_7_l58_58918


namespace exponentiation_rule_l58_58453

theorem exponentiation_rule (a m : ℕ) (h : (a^2)^m = a^6) : m = 3 :=
by
  sorry

end exponentiation_rule_l58_58453


namespace magnitude_difference_l58_58723

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (norm_a : ‖a‖ = 2) (norm_b : ‖b‖ = 1) (norm_a_plus_b : ‖a + b‖ = Real.sqrt 3)

theorem magnitude_difference :
  ‖a - b‖ = Real.sqrt 7 :=
by
  sorry

end magnitude_difference_l58_58723


namespace total_amount_l58_58930

noncomputable def x_share : ℝ := 60
noncomputable def y_share : ℝ := 27
noncomputable def z_share : ℝ := 0.30 * x_share

theorem total_amount (hx : y_share = 0.45 * x_share) : x_share + y_share + z_share = 105 :=
by
  have hx_val : x_share = 27 / 0.45 := by
  { -- Proof that x_share is indeed 60 as per the given problem
    sorry }
  sorry

end total_amount_l58_58930


namespace height_of_fourth_person_l58_58414

theorem height_of_fourth_person
  (h : ℝ)
  (H1 : h + (h + 2) + (h + 4) + (h + 10) = 4 * 79) :
  h + 10 = 85 :=
by
  have H2 : h + 4 = 79 := by linarith
  linarith


end height_of_fourth_person_l58_58414


namespace joan_seashells_initially_l58_58014

variable (mikeGave joanTotal : ℕ)

theorem joan_seashells_initially (h : mikeGave = 63) (t : joanTotal = 142) : joanTotal - mikeGave = 79 := 
by
  sorry

end joan_seashells_initially_l58_58014


namespace sufficient_but_not_necessary_l58_58003

theorem sufficient_but_not_necessary (a : ℝ) : ((a = 2) → ((a - 1) * (a - 2) = 0)) ∧ (¬(((a - 1) * (a - 2) = 0) → (a = 2))) := 
by 
sorry

end sufficient_but_not_necessary_l58_58003


namespace platform_length_is_correct_l58_58526

def speed_kmph : ℝ := 72
def seconds_to_cross_platform : ℝ := 26
def train_length_m : ℝ := 270.0416

noncomputable def length_of_platform : ℝ :=
  let speed_mps := speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * seconds_to_cross_platform
  total_distance - train_length_m

theorem platform_length_is_correct : 
  length_of_platform = 249.9584 := 
by
  sorry

end platform_length_is_correct_l58_58526


namespace hexagon_largest_angle_measure_l58_58225

theorem hexagon_largest_angle_measure (x : ℝ) (a b c d e f : ℝ)
  (h_ratio: a = 2 * x) (h_ratio2: b = 3 * x)
  (h_ratio3: c = 3 * x) (h_ratio4: d = 4 * x)
  (h_ratio5: e = 4 * x) (h_ratio6: f = 6 * x)
  (h_sum: a + b + c + d + e + f = 720) :
  f = 2160 / 11 :=
by
  -- Proof is not required
  sorry

end hexagon_largest_angle_measure_l58_58225


namespace largest_possible_value_of_m_l58_58960

theorem largest_possible_value_of_m :
  ∃ (X Y Z : ℕ), 0 ≤ X ∧ X ≤ 7 ∧ 0 ≤ Y ∧ Y ≤ 7 ∧ 0 ≤ Z ∧ Z ≤ 7 ∧
                 (64 * X + 8 * Y + Z = 475) ∧ 
                 (144 * Z + 12 * Y + X = 475) := 
sorry

end largest_possible_value_of_m_l58_58960


namespace perpendicular_vector_l58_58155

theorem perpendicular_vector {a : ℝ × ℝ} (h : a = (1, -2)) : ∃ (b : ℝ × ℝ), b = (2, 1) ∧ (a.1 * b.1 + a.2 * b.2 = 0) :=
by 
  sorry

end perpendicular_vector_l58_58155


namespace wood_burned_afternoon_l58_58599

theorem wood_burned_afternoon (burned_morning burned_afternoon bundles_start bundles_end : ℕ) 
  (h_burned_morning : burned_morning = 4)
  (h_bundles_start : bundles_start = 10) 
  (h_bundles_end : bundles_end = 3)
  (total_burned : bundles_start - bundles_end = burned_morning + burned_afternoon) :
  burned_afternoon = 3 :=
by {
  -- Proof placeholder
  sorry
}

end wood_burned_afternoon_l58_58599


namespace tangent_line_equation_l58_58025

noncomputable def circle_eq1 (x y : ℝ) := x^2 + (y - 2)^2 - 4
noncomputable def circle_eq2 (x y : ℝ) := (x - 3)^2 + (y + 2)^2 - 21
noncomputable def line_eq (x y : ℝ) := 3*x - 4*y - 4

theorem tangent_line_equation :
  ∀ (x y : ℝ), (circle_eq1 x y = 0 ∧ circle_eq2 x y = 0) ↔ line_eq x y = 0 :=
sorry

end tangent_line_equation_l58_58025


namespace expression_not_defined_l58_58473

theorem expression_not_defined (x : ℝ) :
    ¬(x^2 - 22*x + 121 = 0) ↔ ¬(x - 11 = 0) :=
by sorry

end expression_not_defined_l58_58473


namespace inequality_a4_b4_c4_l58_58976

theorem inequality_a4_b4_c4 (a b c : Real) : a^4 + b^4 + c^4 ≥ abc * (a + b + c) := 
by
  sorry

end inequality_a4_b4_c4_l58_58976


namespace solve_for_n_l58_58169

theorem solve_for_n :
  ∃ n : ℤ, -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.sin (750 * Real.pi / 180) ∧ n = 30 :=
by
  sorry

end solve_for_n_l58_58169


namespace calculate_decimal_l58_58899

theorem calculate_decimal : 3.59 + 2.4 - 1.67 = 4.32 := 
  by
  sorry

end calculate_decimal_l58_58899


namespace annie_total_distance_traveled_l58_58030

-- Definitions of conditions
def walk_distance : ℕ := 5
def bus_distance : ℕ := 7
def total_distance_one_way : ℕ := walk_distance + bus_distance
def total_distance_round_trip : ℕ := total_distance_one_way * 2

-- Theorem statement to prove the total number of blocks traveled
theorem annie_total_distance_traveled : total_distance_round_trip = 24 :=
by
  sorry

end annie_total_distance_traveled_l58_58030


namespace gcd_1729_867_l58_58791

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by 
  sorry

end gcd_1729_867_l58_58791


namespace factor_quadratic_l58_58455

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 16 * x^2 - 56 * x + 49

-- The goal is to prove that the quadratic expression is equal to (4x - 7)^2
theorem factor_quadratic (x : ℝ) : quadratic_expr x = (4 * x - 7)^2 :=
by
  sorry

end factor_quadratic_l58_58455


namespace garden_roller_length_l58_58165

theorem garden_roller_length
  (diameter : ℝ)
  (total_area : ℝ)
  (revolutions : ℕ)
  (pi : ℝ)
  (circumference : ℝ)
  (area_per_revolution : ℝ)
  (length : ℝ)
  (h1 : diameter = 1.4)
  (h2 : total_area = 44)
  (h3 : revolutions = 5)
  (h4 : pi = (22 / 7))
  (h5 : circumference = pi * diameter)
  (h6 : area_per_revolution = total_area / (revolutions : ℝ))
  (h7 : area_per_revolution = circumference * length) :
  length = 7 := by
  sorry

end garden_roller_length_l58_58165


namespace sachin_is_younger_than_rahul_by_18_years_l58_58027

-- Definitions based on conditions
def sachin_age : ℕ := 63
def ratio_of_ages : ℚ := 7 / 9

-- Assertion that based on the given conditions, Sachin is 18 years younger than Rahul
theorem sachin_is_younger_than_rahul_by_18_years (R : ℕ) (h1 : (sachin_age : ℚ) / R = ratio_of_ages) : R - sachin_age = 18 :=
by
  sorry

end sachin_is_younger_than_rahul_by_18_years_l58_58027


namespace sum_of_corners_10x10_l58_58718

theorem sum_of_corners_10x10 : 
  let top_left := 1
  let top_right := 10
  let bottom_left := 91
  let bottom_right := 100
  (top_left + top_right + bottom_left + bottom_right) = 202 :=
by
  let top_left := 1
  let top_right := 10
  let bottom_left := 91
  let bottom_right := 100
  show top_left + top_right + bottom_left + bottom_right = 202
  sorry

end sum_of_corners_10x10_l58_58718


namespace person_a_work_days_l58_58799

theorem person_a_work_days (x : ℝ) (h1 : 1 / 6 + 1 / x = 1 / 3.75) : x = 10 := 
sorry

end person_a_work_days_l58_58799


namespace plan_Y_cheaper_l58_58308

theorem plan_Y_cheaper (y : ℤ) :
  (15 * (y : ℚ) > 2500 + 8 * (y : ℚ)) ↔ y > 358 :=
by
  sorry

end plan_Y_cheaper_l58_58308


namespace missing_number_l58_58047

theorem missing_number (n : ℝ) (h : (0.0088 * 4.5) / (0.05 * n * 0.008) = 990) : n = 0.1 :=
sorry

end missing_number_l58_58047


namespace integral_2x_plus_3_squared_l58_58371

open Real

-- Define the function to be integrated
def f (x : ℝ) := (2 * x + 3) ^ 2

-- State the theorem for the indefinite integral
theorem integral_2x_plus_3_squared :
  ∃ C : ℝ, ∫ x, f x = (1 / 6) * (2 * x + 3) ^ 3 + C :=
by
  sorry

end integral_2x_plus_3_squared_l58_58371


namespace part_I_part_II_l58_58736

/-- (I) -/
theorem part_I (x : ℝ) (a : ℝ) (h_a : a = -1) :
  (|2 * x| + |x - 1| ≤ 4) → x ∈ Set.Icc (-1) (5 / 3) :=
by sorry

/-- (II) -/
theorem part_II (x : ℝ) (a : ℝ) (h_eq : |2 * x| + |x + a| = |x - a|) :
  (a > 0 → x ∈ Set.Icc (-a) 0) ∧ (a < 0 → x ∈ Set.Icc 0 (-a)) :=
by sorry

end part_I_part_II_l58_58736


namespace two_colonies_reach_limit_l58_58170

noncomputable def bacteria_growth (n : ℕ) : ℕ := 2^n

theorem two_colonies_reach_limit (days : ℕ) (h : bacteria_growth days = (2^20)) : 
  bacteria_growth days = bacteria_growth 20 := 
by sorry

end two_colonies_reach_limit_l58_58170


namespace find_y_l58_58458

theorem find_y (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := 
by sorry

end find_y_l58_58458


namespace target_hit_probability_l58_58142

-- Define the probabilities of Person A and Person B hitting the target
def prob_A_hits := 0.8
def prob_B_hits := 0.7

-- Define the probability that the target is hit when both shoot independently at the same time
def prob_target_hit := 1 - (1 - prob_A_hits) * (1 - prob_B_hits)

theorem target_hit_probability : prob_target_hit = 0.94 := 
by
  sorry

end target_hit_probability_l58_58142


namespace average_infection_rate_l58_58659

theorem average_infection_rate (x : ℝ) : 
  (1 + x + x * (1 + x) = 196) → x = 13 :=
by
  intro h
  sorry

end average_infection_rate_l58_58659


namespace remainder_of_3_pow_244_mod_5_l58_58908

theorem remainder_of_3_pow_244_mod_5 : (3^244) % 5 = 1 := by
  sorry

end remainder_of_3_pow_244_mod_5_l58_58908


namespace correct_chart_for_percentage_representation_l58_58336

def bar_chart_characteristic := "easily shows the quantity"
def line_chart_characteristic := "shows the quantity and reflects the changes in quantity"
def pie_chart_characteristic := "reflects the relationship between a part and the whole"

def representation_requirement := "represents the percentage of students in each grade level in the fifth grade's physical education test scores out of the total number of students in the grade"

theorem correct_chart_for_percentage_representation : 
  (representation_requirement = pie_chart_characteristic) := 
by 
   -- The proof follows from the prior definition of characteristics.
   sorry

end correct_chart_for_percentage_representation_l58_58336


namespace total_decorations_l58_58491

theorem total_decorations 
  (skulls : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) (pumpkins : ℕ) 
  (cauldron : ℕ) (budget_decorations : ℕ) (left_decorations : ℕ)
  (h_skulls : skulls = 12)
  (h_broomsticks : broomsticks = 4)
  (h_spiderwebs : spiderwebs = 12)
  (h_pumpkins : pumpkins = 2 * spiderwebs)
  (h_cauldron : cauldron = 1)
  (h_budget_decorations : budget_decorations = 20)
  (h_left_decorations : left_decorations = 10) : 
  skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget_decorations + left_decorations = 83 := 
by 
  sorry

end total_decorations_l58_58491


namespace pencils_count_l58_58989

theorem pencils_count (P L : ℕ) 
  (h1 : P * 6 = L * 5) 
  (h2 : L = P + 7) : 
  L = 42 :=
by
  sorry

end pencils_count_l58_58989


namespace pay_nineteen_rubles_l58_58677

/-- 
Given a purchase cost of 19 rubles, a customer with only three-ruble bills, 
and a cashier with only five-ruble bills, both having 15 bills each,
prove that it is possible for the customer to pay exactly 19 rubles.
-/
theorem pay_nineteen_rubles (purchase_cost : ℕ) (customer_bills cashier_bills : ℕ) 
  (customer_denomination cashier_denomination : ℕ) (customer_count cashier_count : ℕ) :
  purchase_cost = 19 →
  customer_denomination = 3 →
  cashier_denomination = 5 →
  customer_count = 15 →
  cashier_count = 15 →
  (∃ m n : ℕ, m * customer_denomination - n * cashier_denomination = purchase_cost 
  ∧ m ≤ customer_count ∧ n ≤ cashier_count) :=
by
  intros
  sorry

end pay_nineteen_rubles_l58_58677


namespace expected_value_of_unfair_die_l58_58345

noncomputable def seven_sided_die_expected_value : ℝ :=
  let p7 := 1 / 3
  let p_other := (2 / 3) / 6
  ((1 + 2 + 3 + 4 + 5 + 6) * p_other + 7 * p7)

theorem expected_value_of_unfair_die :
  seven_sided_die_expected_value = 14 / 3 :=
by
  sorry

end expected_value_of_unfair_die_l58_58345


namespace sum_50_to_75_l58_58895

-- Conditionally sum the series from 50 to 75
def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_50_to_75 : sum_integers 50 75 = 1625 :=
by
  sorry

end sum_50_to_75_l58_58895


namespace trains_clear_each_other_in_11_seconds_l58_58101

-- Define the lengths of the trains
def length_train1 := 100  -- in meters
def length_train2 := 120  -- in meters

-- Define the speeds of the trains (in km/h), converted to m/s
def speed_train1 := 42 * 1000 / 3600  -- 42 km/h to m/s
def speed_train2 := 30 * 1000 / 3600  -- 30 km/h to m/s

-- Calculate the total distance to be covered
def total_distance := length_train1 + length_train2  -- in meters

-- Calculate the relative speed when they are moving towards each other
def relative_speed := speed_train1 + speed_train2  -- in m/s

-- Calculate the time required for the trains to be clear of each other (in seconds)
noncomputable def clear_time := total_distance / relative_speed

-- Theorem stating the above
theorem trains_clear_each_other_in_11_seconds :
  clear_time = 11 :=
by
  -- Proof would go here
  sorry

end trains_clear_each_other_in_11_seconds_l58_58101


namespace g_at_neg_two_is_fifteen_l58_58803

def g (x : ℤ) : ℤ := 2 * x^2 - 3 * x + 1

theorem g_at_neg_two_is_fifteen : g (-2) = 15 :=
by 
  -- proof is skipped
  sorry

end g_at_neg_two_is_fifteen_l58_58803


namespace johns_quarters_l58_58477

variable (x : ℕ)  -- Number of quarters John has

def number_of_dimes : ℕ := x + 3  -- Number of dimes
def number_of_nickels : ℕ := x - 6  -- Number of nickels

theorem johns_quarters (h : x + (x + 3) + (x - 6) = 63) : x = 22 :=
by
  sorry

end johns_quarters_l58_58477


namespace sum_of_coefficients_evaluated_l58_58934

theorem sum_of_coefficients_evaluated 
  (x y : ℤ) (h1 : x = 2) (h2 : y = -1)
  : (3 * x + 4 * y)^9 + (2 * x - 5 * y)^9 = 387420501 := 
by
  rw [h1, h2]
  sorry

end sum_of_coefficients_evaluated_l58_58934


namespace initial_books_count_l58_58121

-- Definitions of the given conditions
def shelves : ℕ := 9
def books_per_shelf : ℕ := 9
def books_remaining : ℕ := shelves * books_per_shelf
def books_sold : ℕ := 39

-- Statement of the proof problem
theorem initial_books_count : books_remaining + books_sold = 120 := 
by {
  sorry
}

end initial_books_count_l58_58121


namespace train_crossing_time_l58_58118

noncomputable def relative_speed_kmh (speed_train : ℕ) (speed_man : ℕ) : ℕ := speed_train + speed_man

noncomputable def kmh_to_mps (speed_kmh : ℕ) : ℝ := speed_kmh * 1000 / 3600

noncomputable def crossing_time (length_train : ℕ) (speed_train_kmh : ℕ) (speed_man_kmh : ℕ) : ℝ :=
  let relative_speed_kmh := relative_speed_kmh speed_train_kmh speed_man_kmh
  let relative_speed_mps := kmh_to_mps relative_speed_kmh
  length_train / relative_speed_mps

theorem train_crossing_time :
  crossing_time 210 25 2 = 28 :=
  by
  sorry

end train_crossing_time_l58_58118


namespace common_ratio_of_geometric_series_l58_58412

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l58_58412


namespace sum_of_inverses_inequality_l58_58631

theorem sum_of_inverses_inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum_eq : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 :=
sorry

end sum_of_inverses_inequality_l58_58631


namespace problem1_problem2_l58_58081

-- Define the base types and expressions
variables (x m : ℝ)

-- Proofs of the given expressions
theorem problem1 : (x^7 / x^3) * x^4 = x^8 :=
by sorry

theorem problem2 : m * m^3 + ((-m^2)^3 / m^2) = 0 :=
by sorry

end problem1_problem2_l58_58081


namespace find_m_value_l58_58856

theorem find_m_value (m : ℝ) (h₀ : m > 0) (h₁ : (4 - m) / (m - 2) = m) : m = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end find_m_value_l58_58856


namespace strip_width_l58_58992

theorem strip_width (w : ℝ) (h_floor : ℝ := 10) (b_floor : ℝ := 8) (area_rug : ℝ := 24) :
  (h_floor - 2 * w) * (b_floor - 2 * w) = area_rug → w = 2 := 
by 
  sorry

end strip_width_l58_58992


namespace find_b_for_inf_solutions_l58_58185

theorem find_b_for_inf_solutions (x : ℝ) (b : ℝ) : 5 * (3 * x - b) = 3 * (5 * x + 15) → b = -9 :=
by
  intro h
  sorry

end find_b_for_inf_solutions_l58_58185


namespace solution_interval_l58_58139

-- Define the differentiable function f over the interval (-∞, 0)
variable {f : ℝ → ℝ}
variable (hf : ∀ x < 0, HasDerivAt f (f' x) x)
variable (hx_cond : ∀ x < 0, 2 * f x + x * (deriv f x) > x^2)

-- Proof statement to show the solution interval
theorem solution_interval :
  {x : ℝ | (x + 2018)^2 * f (x + 2018) - 4 * f (-2) > 0} = {x | x < -2020} :=
sorry

end solution_interval_l58_58139


namespace lizard_eye_difference_l58_58503

def jan_eye : ℕ := 3
def jan_wrinkle : ℕ := 3 * jan_eye
def jan_spot : ℕ := 7 * jan_wrinkle

def cousin_eye : ℕ := 3
def cousin_wrinkle : ℕ := 2 * cousin_eye
def cousin_spot : ℕ := 5 * cousin_wrinkle

def total_eyes : ℕ := jan_eye + cousin_eye
def total_wrinkles : ℕ := jan_wrinkle + cousin_wrinkle
def total_spots : ℕ := jan_spot + cousin_spot
def total_spots_and_wrinkles : ℕ := total_wrinkles + total_spots

theorem lizard_eye_difference : total_spots_and_wrinkles - total_eyes = 102 := by
  sorry

end lizard_eye_difference_l58_58503


namespace unique_third_rectangle_exists_l58_58065

-- Define the given rectangles.
def rect1_length : ℕ := 3
def rect1_width : ℕ := 8
def rect2_length : ℕ := 2
def rect2_width : ℕ := 5

-- Define the areas of the given rectangles.
def area_rect1 : ℕ := rect1_length * rect1_width
def area_rect2 : ℕ := rect2_length * rect2_width

-- Define the total area covered by the two given rectangles.
def total_area_without_third : ℕ := area_rect1 + area_rect2

-- We need to prove that there exists one unique configuration for the third rectangle.
theorem unique_third_rectangle_exists (a b : ℕ) : 
  (total_area_without_third + a * b = 34) → 
  (a * b = 4) → 
  (a = 4 ∧ b = 1 ∨ a = 1 ∧ b = 4) :=
by sorry

end unique_third_rectangle_exists_l58_58065


namespace school_B_saving_l58_58604

def cost_A (kg_price : ℚ) (kg_amount : ℚ) : ℚ :=
  kg_price * kg_amount

def effective_kg_B (total_kg : ℚ) (extra_percentage : ℚ) : ℚ :=
  total_kg / (1 + extra_percentage)

def cost_B (kg_price : ℚ) (effective_kg : ℚ) : ℚ :=
  kg_price * effective_kg

theorem school_B_saving
  (kg_amount : ℚ) (price_A: ℚ) (discount: ℚ) (extra_percentage : ℚ) 
  (expected_saving : ℚ)
  (h1 : kg_amount = 56)
  (h2 : price_A = 8.06)
  (h3 : discount = 0.56)
  (h4 : extra_percentage = 0.05)
  (h5 : expected_saving = 51.36) :
  cost_A price_A kg_amount - cost_B (price_A - discount) (effective_kg_B kg_amount extra_percentage) = expected_saving := 
by 
  sorry

end school_B_saving_l58_58604


namespace prime_gt_three_modulus_l58_58633

theorem prime_gt_three_modulus (p : ℕ) (hp : Prime p) (hp_gt3 : p > 3) : (p^2 + 12) % 12 = 1 := by
  sorry

end prime_gt_three_modulus_l58_58633


namespace remainder_of_division_l58_58413

noncomputable def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 1
noncomputable def g (x : ℝ) : ℝ := (x - 3) ^ 2
noncomputable def remainder (x : ℝ) : ℝ := 324 * x - 488

theorem remainder_of_division :
  ∀ (x : ℝ), (f x) % (g x) = remainder x :=
sorry

end remainder_of_division_l58_58413


namespace triangles_in_figure_l58_58927

-- Definitions for the figure
def number_of_triangles : ℕ :=
  -- The number of triangles in a figure composed of a rectangle with three vertical lines and two horizontal lines
  50

-- The theorem we want to prove
theorem triangles_in_figure : number_of_triangles = 50 :=
by
  sorry

end triangles_in_figure_l58_58927


namespace time_for_A_l58_58681

theorem time_for_A (A B C : ℝ) 
  (h1 : 1/B + 1/C = 1/3) 
  (h2 : 1/A + 1/C = 1/2) 
  (h3 : 1/B = 1/30) : 
  A = 5/2 := 
by
  sorry

end time_for_A_l58_58681


namespace grayson_travels_further_l58_58702

noncomputable def grayson_first_part_distance : ℝ := 25 * 1
noncomputable def grayson_second_part_distance : ℝ := 20 * 0.5
noncomputable def total_distance_grayson : ℝ := grayson_first_part_distance + grayson_second_part_distance

noncomputable def total_distance_rudy : ℝ := 10 * 3

theorem grayson_travels_further : (total_distance_grayson - total_distance_rudy) = 5 := by
  sorry

end grayson_travels_further_l58_58702


namespace dogs_sold_l58_58069

theorem dogs_sold (cats_sold : ℕ) (h1 : cats_sold = 16) (ratio : ℕ × ℕ) (h2 : ratio = (2, 1)) : ∃ dogs_sold : ℕ, dogs_sold = 8 := by
  sorry

end dogs_sold_l58_58069


namespace number_of_packs_of_cake_l58_58012

-- Define the total number of packs of groceries
def total_packs : ℕ := 14

-- Define the number of packs of cookies
def packs_of_cookies : ℕ := 2

-- Define the number of packs of cake as total packs minus packs of cookies
def packs_of_cake : ℕ := total_packs - packs_of_cookies

theorem number_of_packs_of_cake :
  packs_of_cake = 12 := by
  -- Placeholder for the proof
  sorry

end number_of_packs_of_cake_l58_58012


namespace arithmetic_sequence_common_difference_l58_58851

theorem arithmetic_sequence_common_difference (a_1 a_4 a_5 d : ℤ) 
  (h1 : a_1 + a_5 = 10) 
  (h2 : a_4 = 7) 
  (h3 : a_4 = a_1 + 3 * d) 
  (h4 : a_5 = a_1 + 4 * d) : 
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l58_58851


namespace num_students_third_section_l58_58500

-- Define the conditions
def num_students_first_section : ℕ := 65
def num_students_second_section : ℕ := 35
def num_students_fourth_section : ℕ := 42
def mean_marks_first_section : ℝ := 50
def mean_marks_second_section : ℝ := 60
def mean_marks_third_section : ℝ := 55
def mean_marks_fourth_section : ℝ := 45
def overall_average_marks : ℝ := 51.95

-- Theorem stating the number of students in the third section
theorem num_students_third_section
  (x : ℝ)
  (h : (num_students_first_section * mean_marks_first_section
       + num_students_second_section * mean_marks_second_section
       + x * mean_marks_third_section
       + num_students_fourth_section * mean_marks_fourth_section)
       = overall_average_marks * (num_students_first_section + num_students_second_section + x + num_students_fourth_section)) :
  x = 45 :=
by
  -- Proof will go here
  sorry

end num_students_third_section_l58_58500


namespace intersection_of_A_and_B_l58_58287

def A : Set ℤ := {-3, -1, 2, 6}
def B : Set ℤ := {x | x > 0}

theorem intersection_of_A_and_B : A ∩ B = {2, 6} :=
by
  sorry

end intersection_of_A_and_B_l58_58287


namespace point_on_graph_find_k_shifted_graph_passing_y_axis_intercept_range_l58_58555

-- Question (1): Proving that the point (-2,0) lies on the graph
theorem point_on_graph (k : ℝ) (hk : k ≠ 0) : k * (-2 + 2) = 0 := 
by sorry

-- Question (2): Finding the value of k given a shifted graph passing through a point
theorem find_k_shifted_graph_passing (k : ℝ) : (k * (1 + 2) + 2 = -2) → k = -4/3 := 
by sorry

-- Question (3): Proving the range of k for the function's y-intercept within given limits
theorem y_axis_intercept_range (k : ℝ) (hk : -2 < 2 * k ∧ 2 * k < 0) : -1 < k ∧ k < 0 := 
by sorry

end point_on_graph_find_k_shifted_graph_passing_y_axis_intercept_range_l58_58555


namespace train_speed_l58_58780

theorem train_speed (length_of_train time_to_cross : ℕ) (h_length : length_of_train = 50) (h_time : time_to_cross = 3) : 
  (length_of_train / time_to_cross : ℝ) * 3.6 = 60 := by
  sorry

end train_speed_l58_58780


namespace M_gt_N_l58_58722

-- Define the variables and conditions
variables (a : ℝ)
def M : ℝ := 5 * a^2 - a + 1
def N : ℝ := 4 * a^2 + a - 1

-- Statement to prove
theorem M_gt_N : M a > N a := by
  -- Placeholder for the actual proof
  sorry

end M_gt_N_l58_58722


namespace original_grain_amount_l58_58346

def grain_spilled : ℕ := 49952
def grain_remaining : ℕ := 918

theorem original_grain_amount : grain_spilled + grain_remaining = 50870 :=
by
  sorry

end original_grain_amount_l58_58346


namespace find_values_l58_58213

theorem find_values (x y z : ℝ) :
  (x + y + z = 1) →
  (x^2 * y + y^2 * z + z^2 * x = x * y^2 + y * z^2 + z * x^2) →
  (x^3 + y^2 + z = y^3 + z^2 + x) →
  ( (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ 
    (x = 0 ∧ y = 0 ∧ z = 1) ∨
    (x = 2/3 ∧ y = -1/3 ∧ z = 2/3) ∨
    (x = 0 ∧ y = 1 ∧ z = 0) ∨
    (x = 1 ∧ y = 0 ∧ z = 0) ∨
    (x = -1 ∧ y = 1 ∧ z = 1) ) := 
sorry

end find_values_l58_58213


namespace surface_area_of_z_eq_xy_over_a_l58_58871

noncomputable def surface_area (a : ℝ) (h : a > 0) : ℝ :=
  (2 * Real.pi / 3) * a^2 * (2 * Real.sqrt 2 - 1)

theorem surface_area_of_z_eq_xy_over_a (a : ℝ) (h : a > 0) :
  surface_area a h = (2 * Real.pi / 3) * a^2 * (2 * Real.sqrt 2 - 1) := 
sorry

end surface_area_of_z_eq_xy_over_a_l58_58871


namespace math_problem_l58_58089

theorem math_problem
  (m : ℕ) (h₁ : m = 8^126) :
  (m * 16) / 64 = 16^94 :=
by
  sorry

end math_problem_l58_58089


namespace negation_of_at_most_one_obtuse_is_at_least_two_obtuse_l58_58313

-- Definition of a triangle with a property on the angles.
def triangle (a b c : ℝ) : Prop := a + b + c = 180 ∧ 0 < a ∧ 0 < b ∧ 0 < c

-- Definition of an obtuse angle.
def obtuse (x : ℝ) : Prop := x > 90

-- Proposition: In a triangle, at most one angle is obtuse.
def at_most_one_obtuse (a b c : ℝ) : Prop := 
  triangle a b c ∧ (obtuse a → ¬ obtuse b ∧ ¬ obtuse c) ∧ (obtuse b → ¬ obtuse a ∧ ¬ obtuse c) ∧ (obtuse c → ¬ obtuse a ∧ ¬ obtuse b)

-- Negation: In a triangle, there are at least two obtuse angles.
def at_least_two_obtuse (a b c : ℝ) : Prop := 
  triangle a b c ∧ (obtuse a ∧ obtuse b) ∨ (obtuse a ∧ obtuse c) ∨ (obtuse b ∧ obtuse c)

-- Prove the negation equivalence
theorem negation_of_at_most_one_obtuse_is_at_least_two_obtuse (a b c : ℝ) :
  (¬ at_most_one_obtuse a b c) ↔ at_least_two_obtuse a b c :=
sorry

end negation_of_at_most_one_obtuse_is_at_least_two_obtuse_l58_58313


namespace fraction_to_decimal_l58_58062

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end fraction_to_decimal_l58_58062


namespace quadratic_inequality_solution_set_l58_58512

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < -2) : 
  { x : ℝ | ax^2 + (a - 2)*x - 2 ≥ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 2/a } := 
by
  sorry

end quadratic_inequality_solution_set_l58_58512


namespace roots_of_Q_are_fifth_powers_of_roots_of_P_l58_58765

def P (x : ℝ) : ℝ := x^3 - 3 * x + 1

noncomputable def Q (y : ℝ) : ℝ := y^3 + 15 * y^2 - 198 * y + 1

theorem roots_of_Q_are_fifth_powers_of_roots_of_P : 
  ∀ α β γ : ℝ, (P α = 0) ∧ (P β = 0) ∧ (P γ = 0) →
  (Q (α^5) = 0) ∧ (Q (β^5) = 0) ∧ (Q (γ^5) = 0) := 
by 
  intros α β γ h
  sorry

end roots_of_Q_are_fifth_powers_of_roots_of_P_l58_58765


namespace parrots_are_red_l58_58573

-- Definitions for fractions.
def total_parrots : ℕ := 160
def green_fraction : ℚ := 5 / 8
def blue_fraction : ℚ := 1 / 4

-- Definition for calculating the number of parrots.
def number_of_green_parrots : ℚ := green_fraction * total_parrots
def number_of_blue_parrots : ℚ := blue_fraction * total_parrots
def number_of_red_parrots : ℚ := total_parrots - number_of_green_parrots - number_of_blue_parrots

-- The theorem to prove.
theorem parrots_are_red : number_of_red_parrots = 20 := by
  -- Proof is omitted.
  sorry

end parrots_are_red_l58_58573


namespace chocolate_ratio_l58_58161

theorem chocolate_ratio (N A : ℕ) (h1 : N = 10) (h2 : A - 5 = N + 15) : A / N = 3 :=
by {
  sorry
}

end chocolate_ratio_l58_58161


namespace average_velocity_eq_l58_58666

noncomputable def motion_eq : ℝ → ℝ := λ t => 1 - t + t^2

theorem average_velocity_eq (Δt : ℝ) :
  (motion_eq (3 + Δt) - motion_eq 3) / Δt = 5 + Δt :=
by
  sorry

end average_velocity_eq_l58_58666


namespace ab_value_l58_58061

theorem ab_value (a b : ℝ) (h1 : a + b = 4) (h2 : a^2 + b^2 = 18) : a * b = -1 :=
by sorry

end ab_value_l58_58061


namespace factor_expression_l58_58006

theorem factor_expression (y : ℝ) : 5 * y * (y + 2) + 9 * (y + 2) + 2 * (y + 2) = (y + 2) * (5 * y + 11) := by
  sorry

end factor_expression_l58_58006


namespace nonagon_diagonals_l58_58459

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nonagon_diagonals : number_of_diagonals 9 = 27 := 
by
  sorry

end nonagon_diagonals_l58_58459


namespace volume_of_one_gram_l58_58999

theorem volume_of_one_gram (mass_per_cubic_meter : ℕ)
  (kilo_to_grams : ℕ)
  (cubic_meter_to_cubic_centimeters : ℕ)
  (substance_mass : mass_per_cubic_meter = 300)
  (kilo_conv : kilo_to_grams = 1000)
  (cubic_conv : cubic_meter_to_cubic_centimeters = 1000000)
  :
  ∃ v : ℝ, v = cubic_meter_to_cubic_centimeters / (mass_per_cubic_meter * kilo_to_grams) ∧ v = 10 / 3 := 
by 
  sorry

end volume_of_one_gram_l58_58999


namespace min_value_expression_l58_58196

/--
  Prove that the minimum value of the expression (xy - 2)^2 + (x + y - 1)^2 
  for real numbers x and y is 2.
--/
theorem min_value_expression : 
  ∃ x y : ℝ, (∀ a b : ℝ, (a * b - 2)^2 + (a + b - 1)^2 ≥ (x * y - 2)^2 + (x + y - 1)^2 ) ∧ 
  (x * y - 2)^2 + (x + y - 1)^2 = 2 :=
by
  sorry

end min_value_expression_l58_58196


namespace total_chickens_after_purchase_l58_58953

def initial_chickens : ℕ := 400
def percentage_died : ℕ := 40
def times_to_buy : ℕ := 10

noncomputable def chickens_died : ℕ := (percentage_died * initial_chickens) / 100
noncomputable def chickens_remaining : ℕ := initial_chickens - chickens_died
noncomputable def chickens_bought : ℕ := times_to_buy * chickens_died
noncomputable def total_chickens : ℕ := chickens_remaining + chickens_bought

theorem total_chickens_after_purchase : total_chickens = 1840 :=
by
  sorry

end total_chickens_after_purchase_l58_58953


namespace katherine_fruit_count_l58_58100

variables (apples pears bananas total_fruit : ℕ)

theorem katherine_fruit_count (h1 : apples = 4) 
  (h2 : pears = 3 * apples)
  (h3 : total_fruit = 21) 
  (h4 : total_fruit = apples + pears + bananas) : bananas = 5 := 
by sorry

end katherine_fruit_count_l58_58100


namespace people_at_the_beach_l58_58292

-- Conditions
def initial : ℕ := 3  -- Molly and her parents
def joined : ℕ := 100 -- 100 people joined at the beach
def left : ℕ := 40    -- 40 people left at 5:00

-- Proof statement
theorem people_at_the_beach : initial + joined - left = 63 :=
by
  sorry

end people_at_the_beach_l58_58292


namespace coordinates_reflect_y_axis_l58_58638

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem coordinates_reflect_y_axis (p : ℝ × ℝ) (h : p = (5, 2)) : reflect_y_axis p = (-5, 2) :=
by
  rw [h]
  rfl

end coordinates_reflect_y_axis_l58_58638


namespace robotics_club_students_l58_58514

theorem robotics_club_students (total cs e both neither : ℕ) 
  (h1 : total = 80)
  (h2 : cs = 52)
  (h3 : e = 38)
  (h4 : both = 25)
  (h5 : neither = total - (cs - both + e - both + both)) :
  neither = 15 :=
by
  sorry

end robotics_club_students_l58_58514


namespace find_other_root_l58_58937

theorem find_other_root 
  (m : ℚ) 
  (h : 3 * 3^2 + m * 3 - 5 = 0) :
  (1 - 3) * (x : ℚ) = 0 :=
sorry

end find_other_root_l58_58937


namespace odd_function_ln_negx_l58_58475

theorem odd_function_ln_negx (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_positive : ∀ x, x > 0 → f x = Real.log x) :
  ∀ x, x < 0 → f x = -Real.log (-x) :=
by 
  intros x hx_neg
  have hx_pos : -x > 0 := by linarith
  rw [← h_positive (-x) hx_pos, h_odd x]
  sorry

end odd_function_ln_negx_l58_58475


namespace positive_integer_a_l58_58432

theorem positive_integer_a (a : ℕ) (h1 : 0 < a) (h2 : ∃ (k : ℤ), (2 * a + 8) = k * (a + 1)) :
  a = 1 ∨ a = 2 ∨ a = 5 :=
by sorry

end positive_integer_a_l58_58432


namespace polygon_sides_l58_58229

theorem polygon_sides (n : ℕ) (h₁ : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 156 = (180 * (n - 2)) / n) : n = 15 := sorry

end polygon_sides_l58_58229


namespace platform_length_correct_l58_58678

noncomputable def platform_length (train_speed_kmph : ℝ) (crossing_time_s : ℝ) (train_length_m : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let distance_covered := train_speed_mps * crossing_time_s
  distance_covered - train_length_m

theorem platform_length_correct :
  platform_length 72 26 260.0416 = 259.9584 :=
by
  sorry

end platform_length_correct_l58_58678


namespace problem_solution_l58_58424

theorem problem_solution (x : ℝ) (h : x * Real.log 4 / Real.log 3 = 1) : 
  2^x + 4^(-x) = 1 / 3 + Real.sqrt 3 :=
by 
  sorry

end problem_solution_l58_58424


namespace Eleanor_books_l58_58235

theorem Eleanor_books (h p : ℕ) : 
    h + p = 12 ∧ 28 * h + 18 * p = 276 → h = 6 :=
by
  intro hp
  sorry

end Eleanor_books_l58_58235


namespace digit_a_for_divisibility_l58_58214

theorem digit_a_for_divisibility (a : ℕ) (h1 : (8 * 10^3 + 7 * 10^2 + 5 * 10 + a) % 6 = 0) : a = 4 :=
sorry

end digit_a_for_divisibility_l58_58214


namespace simplify_sqrt_25000_l58_58279

theorem simplify_sqrt_25000 : Real.sqrt 25000 = 50 * Real.sqrt 10 := 
by
  sorry

end simplify_sqrt_25000_l58_58279


namespace largest_value_p_l58_58350

theorem largest_value_p 
  (p q r : ℝ) 
  (h1 : p + q + r = 10) 
  (h2 : p * q + p * r + q * r = 25) :
  p ≤ 20 / 3 :=
sorry

end largest_value_p_l58_58350


namespace area_of_triangle_ABC_l58_58647

open Real

noncomputable def triangle_area (b c : ℝ) : ℝ :=
  (sqrt 2 / 4) * (sqrt (4 + b^2)) * (sqrt (4 + c^2))

theorem area_of_triangle_ABC (b c : ℝ) :
  let O : ℝ × ℝ × ℝ := (0, 0, 0)
  let A : ℝ × ℝ × ℝ := (2, 0, 0)
  let B : ℝ × ℝ × ℝ := (0, b, 0)
  let C : ℝ × ℝ × ℝ := (0, 0, c)
  let angle_BAC : ℝ := 45
  (cos (angle_BAC * π / 180) = sqrt 2 / 2) →
  (sin (angle_BAC * π / 180) = sqrt 2 / 2) →
  let AB := sqrt (2^2 + b^2)
  let AC := sqrt (2^2 + c^2)
  let area := (1/2) * AB * AC * (sin (45 * π / 180))
  area = triangle_area b c :=
sorry

end area_of_triangle_ABC_l58_58647


namespace polynomial_value_given_cond_l58_58367

variable (x : ℝ)
theorem polynomial_value_given_cond :
  (x^2 - (5/2) * x = 6) →
  2 * x^2 - 5 * x + 6 = 18 :=
by
  sorry

end polynomial_value_given_cond_l58_58367


namespace gcd_8_10_l58_58002

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_8_10 : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := 
by
  sorry

end gcd_8_10_l58_58002


namespace fencing_rate_l58_58468

/-- Given a circular field of diameter 20 meters and a total cost of fencing of Rs. 94.24777960769379,
    prove that the rate per meter for the fencing is Rs. 1.5. -/
theorem fencing_rate 
  (d : ℝ) (cost : ℝ) (π : ℝ) (rate : ℝ)
  (hd : d = 20)
  (hcost : cost = 94.24777960769379)
  (hπ : π = 3.14159)
  (Circumference : ℝ := π * d)
  (Rate : ℝ := cost / Circumference) : 
  rate = 1.5 :=
sorry

end fencing_rate_l58_58468


namespace smallest_multiple_14_15_16_l58_58258

theorem smallest_multiple_14_15_16 : 
  Nat.lcm (Nat.lcm 14 15) 16 = 1680 := by
  sorry

end smallest_multiple_14_15_16_l58_58258


namespace percentage_of_first_pay_cut_l58_58290

theorem percentage_of_first_pay_cut
  (x : ℝ)
  (h1 : ∃ y z w : ℝ, y = 1 - x/100 ∧ z = 0.86 ∧ w = 0.82 ∧ y * z * w = 0.648784):
  x = 8.04 := by
-- The proof will be added here, this is just the statement
sorry

end percentage_of_first_pay_cut_l58_58290


namespace M_inter_N_is_empty_l58_58463

-- Definition conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | (x - 1) / x < 0}

-- Theorem statement
theorem M_inter_N_is_empty : M ∩ N = ∅ := by
  sorry

end M_inter_N_is_empty_l58_58463


namespace nine_points_unit_square_l58_58401

theorem nine_points_unit_square :
  ∀ (points : List (ℝ × ℝ)), points.length = 9 → 
  (∀ (x : ℝ × ℝ), x ∈ points → 0 ≤ x.1 ∧ x.1 ≤ 1 ∧ 0 ≤ x.2 ∧ x.2 ≤ 1) → 
  ∃ (A B C : ℝ × ℝ), A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ 
  (1 / 8 : ℝ) ≤ abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 :=
by
  sorry

end nine_points_unit_square_l58_58401


namespace remainder_cd_mod_40_l58_58325

theorem remainder_cd_mod_40 (c d : ℤ) (hc : c % 80 = 75) (hd : d % 120 = 117) : (c + d) % 40 = 32 :=
by
  sorry

end remainder_cd_mod_40_l58_58325


namespace tan_alpha_add_pi_div_four_l58_58636

theorem tan_alpha_add_pi_div_four {α : ℝ} (h1 : α ∈ Set.Ioo 0 (Real.pi)) (h2 : Real.cos α = -4/5) :
  Real.tan (α + Real.pi / 4) = 1 / 7 :=
sorry

end tan_alpha_add_pi_div_four_l58_58636


namespace range_of_a_solution_set_of_inequality_l58_58800

-- Lean statement for Part 1
theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, x^2 - 2 * a * x + a > 0 :=
by
  sorry

-- Lean statement for Part 2
theorem solution_set_of_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  { x : ℝ | a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1 } = { x : ℝ | x > 3 } :=
by
  sorry

end range_of_a_solution_set_of_inequality_l58_58800


namespace least_number_to_make_divisible_by_9_l58_58739

theorem least_number_to_make_divisible_by_9 (n : ℕ) :
  ∃ m : ℕ, (228712 + m) % 9 = 0 ∧ n = 5 :=
by
  sorry

end least_number_to_make_divisible_by_9_l58_58739


namespace solve_for_x_l58_58201

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 5) * x = 2) : x = 17.5 :=
by
  -- Here we acknowledge the initial condition and conclusion without proving
  sorry

end solve_for_x_l58_58201


namespace no_valid_pairs_l58_58635

open Nat

theorem no_valid_pairs (l y : ℕ) (h1 : y % 30 = 0) (h2 : l > 1) :
  (∃ n m : ℕ, 180 - 360 / n = y ∧ 180 - 360 / m = l * y ∧ y * l ≤ 180) → False := 
by
  intro h
  sorry

end no_valid_pairs_l58_58635


namespace common_difference_l58_58269

def arith_seq_common_difference (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference {a : ℕ → ℤ} (h₁ : a 5 = 3) (h₂ : a 6 = -2) : arith_seq_common_difference a (-5) :=
by
  intros n
  cases n with
  | zero => sorry -- base case: a 1 = a 0 + (-5), requires additional initial condition
  | succ n' => sorry -- inductive step

end common_difference_l58_58269


namespace river_length_GSA_AWRA_l58_58098

-- Define the main problem statement
noncomputable def river_length_estimate (GSA_length AWRA_length GSA_error AWRA_error error_prob : ℝ) : Prop :=
  (GSA_length = 402) ∧ (AWRA_length = 403) ∧ 
  (GSA_error = 0.5) ∧ (AWRA_error = 0.5) ∧ 
  (error_prob = 0.04) ∧ 
  (abs (402.5 - GSA_length) ≤ GSA_error) ∧ 
  (abs (402.5 - AWRA_length) ≤ AWRA_error) ∧ 
  (error_prob = 1 - (2 * 0.02))

-- The main theorem statement
theorem river_length_GSA_AWRA :
  river_length_estimate 402 403 0.5 0.5 0.04 :=
by
  sorry

end river_length_GSA_AWRA_l58_58098


namespace cos_270_eq_zero_l58_58897

-- Defining the cosine value for the given angle
theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 :=
by
  sorry

end cos_270_eq_zero_l58_58897


namespace sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022_l58_58835

theorem sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022:
  ( (Real.sqrt 10 + 3) ^ 2023 * (Real.sqrt 10 - 3) ^ 2022 = Real.sqrt 10 + 3 ) :=
by {
  sorry
}

end sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022_l58_58835


namespace find_m_l58_58662

-- Define vectors a and b
def a (m : ℝ) : ℝ × ℝ := (2, -m)
def b : ℝ × ℝ := (1, 3)

-- Define the condition for perpendicular vectors
def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the problem
theorem find_m (m : ℝ) (h : is_perpendicular (a m + b) b) : m = 4 :=
sorry -- proof omitted

end find_m_l58_58662


namespace fraction_difference_l58_58474

theorem fraction_difference:
  let f1 := 2 / 3
  let f2 := 3 / 4
  let f3 := 4 / 5
  let f4 := 5 / 7
  (max f1 (max f2 (max f3 f4)) - min f1 (min f2 (min f3 f4))) = 2 / 15 :=
by
  sorry

end fraction_difference_l58_58474


namespace area_of_rhombus_l58_58952

/-- Given the radii of the circles circumscribed around triangles EFG and EGH
    are 10 and 20, respectively, then the area of rhombus EFGH is 30.72√3. -/
theorem area_of_rhombus (R1 R2 : ℝ) (A : ℝ) :
  R1 = 10 → R2 = 20 → A = 30.72 * Real.sqrt 3 :=
by sorry

end area_of_rhombus_l58_58952


namespace moving_circle_passes_focus_l58_58096

noncomputable def parabola (x : ℝ) : Set (ℝ × ℝ) := {p | p.2 ^ 2 = 8 * p.1}
def is_tangent (c : ℝ × ℝ) (r : ℝ) : Prop := c.1 = -2 ∨ c.1 = -2 + 2 * r

theorem moving_circle_passes_focus
  (center : ℝ × ℝ) (H1 : center ∈ parabola center.1)
  (H2 : is_tangent center 2) :
  ∃ focus : ℝ × ℝ, focus = (2, 0) ∧ ∃ r : ℝ, ∀ p ∈ parabola center.1, dist center p = r := sorry

end moving_circle_passes_focus_l58_58096


namespace total_amount_l58_58063

theorem total_amount {B C : ℝ} 
  (h1 : C = 1600) 
  (h2 : 4 * B = 16 * C) : 
  B + C = 2000 :=
sorry

end total_amount_l58_58063


namespace jake_first_test_score_l58_58454

theorem jake_first_test_score 
  (avg_score : ℕ)
  (n_tests : ℕ)
  (second_test_extra : ℕ)
  (third_test_score : ℕ)
  (x : ℕ) : 
  avg_score = 75 → 
  n_tests = 4 → 
  second_test_extra = 10 → 
  third_test_score = 65 →
  (x + (x + second_test_extra) + third_test_score + third_test_score) / n_tests = avg_score →
  x = 80 := by
  intros h1 h2 h3 h4 h5
  sorry

end jake_first_test_score_l58_58454


namespace find_cost_price_l58_58857

theorem find_cost_price (SP : ℤ) (profit_percent : ℚ) (CP : ℤ) (h1 : SP = CP + (profit_percent * CP)) (h2 : SP = 240) (h3 : profit_percent = 0.25) : CP = 192 :=
by
  sorry

end find_cost_price_l58_58857


namespace exponential_function_decreasing_l58_58537

theorem exponential_function_decreasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (0 < a ∧ a < 1) → ¬ (∀ x : ℝ, x > 0 → a ^ x > 0) :=
by
  sorry

end exponential_function_decreasing_l58_58537


namespace initial_value_exists_l58_58048

theorem initial_value_exists (x : ℕ) (h : ∃ k : ℕ, x + 7 = k * 456) : x = 449 :=
sorry

end initial_value_exists_l58_58048


namespace tower_surface_area_l58_58103

noncomputable def total_visible_surface_area (volumes : List ℕ) : ℕ := sorry

theorem tower_surface_area :
  total_visible_surface_area [512, 343, 216, 125, 64, 27, 8, 1] = 882 :=
sorry

end tower_surface_area_l58_58103


namespace class_a_winning_probability_best_of_three_l58_58259

theorem class_a_winning_probability_best_of_three :
  let p := (3 : ℚ) / 5
  let win_first_two := p * p
  let win_first_and_third := p * ((1 - p) * p)
  let win_last_two := (1 - p) * (p * p)
  p * p + p * ((1 - p) * p) + (1 - p) * (p * p) = 81 / 125 :=
by
  sorry

end class_a_winning_probability_best_of_three_l58_58259


namespace math_problem_l58_58472

/-- The proof problem: Calculate -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11. -/
theorem math_problem : -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11 :=
by
  sorry

end math_problem_l58_58472


namespace determine_a_l58_58596

theorem determine_a (a b c : ℕ) (h_b : b = 5) (h_c : c = 6) (h_order : c > b ∧ b > a ∧ a > 2) :
(a - 2) * (b - 2) * (c - 2) = 4 * (b - 2) + 4 * (c - 2) → a = 4 :=
by 
  sorry

end determine_a_l58_58596


namespace find_f_of_three_l58_58031

variable {f : ℝ → ℝ}

theorem find_f_of_three (h : ∀ x : ℝ, f (1 - 2 * x) = x^2 + x) : f 3 = 0 :=
by
  sorry

end find_f_of_three_l58_58031


namespace blue_eyed_among_blondes_l58_58378

variable (l g b a : ℝ)

-- Given: The proportion of blondes among blue-eyed people is greater than the proportion of blondes among all people.
axiom given_condition : a / g > b / l

-- Prove: The proportion of blue-eyed people among blondes is greater than the proportion of blue-eyed people among all people.
theorem blue_eyed_among_blondes (l g b a : ℝ) (h : a / g > b / l) : a / b > g / l :=
by
  sorry

end blue_eyed_among_blondes_l58_58378


namespace a_term_b_value_c_value_d_value_l58_58606

theorem a_term (a x : ℝ) (h1 : a * (x + 1) = x^3 + 3 * x^2 + 3 * x + 1) : a = x^2 + 2 * x + 1 :=
sorry

theorem b_value (a x b : ℝ) (h1 : a - 1 = 0) (h2 : x = 0 ∨ x = b) : b = -2 :=
sorry

theorem c_value (p c b : ℝ) (h1 : p * c^4 = 32) (h2 : p * c = b^2) (h3 : 0 < c) : c = 2 :=
sorry

theorem d_value (A B d : ℝ) (P : ℝ → ℝ) (c : ℝ) (h1 : P (A * B) = P A + P B) (h2 : P A = 1) (h3 : P B = c) (h4 : A = 10^ P A) (h5 : B = 10^ P B) (h6 : d = A * B) : d = 1000 :=
sorry

end a_term_b_value_c_value_d_value_l58_58606


namespace volume_of_pyramid_l58_58311

-- Define the conditions
def pyramid_conditions : Prop :=
  ∃ (s h : ℝ),
  s^2 = 256 ∧
  ∃ (h_A h_C h_B : ℝ),
  ((∃ h_A, 128 = 1/2 * s * h_A) ∧
  (∃ h_C, 112 = 1/2 * s * h_C) ∧
  (∃ h_B, 96 = 1/2 * s * h_B)) ∧
  h^2 + (s/2)^2 = h_A^2 ∧
  h^2 = 256 - (s/2)^2 ∧
  h^2 + (s/4)^2 = h_B^2

-- Define the theorem
theorem volume_of_pyramid :
  pyramid_conditions → 
  ∃ V : ℝ, V = 682.67 * Real.sqrt 3 :=
sorry

end volume_of_pyramid_l58_58311


namespace value_of_m_l58_58438

theorem value_of_m (m : ℤ) (h : m + 1 = - (-2)) : m = 1 :=
sorry

end value_of_m_l58_58438


namespace power_function_monotonic_incr_l58_58931

theorem power_function_monotonic_incr (m : ℝ) (h₁ : m^2 - 5 * m + 7 = 1) (h₂ : m^2 - 6 > 0) : m = 3 := 
by
  sorry

end power_function_monotonic_incr_l58_58931


namespace distinct_units_digits_of_squares_mod_6_l58_58593

theorem distinct_units_digits_of_squares_mod_6 : 
  ∃ (s : Finset ℕ), s = {0, 1, 4, 3} ∧ s.card = 4 :=
by
  sorry

end distinct_units_digits_of_squares_mod_6_l58_58593


namespace smallest_positive_number_is_option_B_l58_58255

theorem smallest_positive_number_is_option_B :
  let A := 8 - 2 * Real.sqrt 17
  let B := 2 * Real.sqrt 17 - 8
  let C := 25 - 7 * Real.sqrt 5
  let D := 40 - 9 * Real.sqrt 2
  let E := 9 * Real.sqrt 2 - 40
  0 < B ∧ (A ≤ 0 ∨ B < A) ∧ (C ≤ 0 ∨ B < C) ∧ (D ≤ 0 ∨ B < D) ∧ (E ≤ 0 ∨ B < E) :=
by
  sorry

end smallest_positive_number_is_option_B_l58_58255


namespace real_roots_if_and_only_if_m_leq_5_l58_58532

theorem real_roots_if_and_only_if_m_leq_5 (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x + 1 = 0) ↔ m ≤ 5 :=
by
  sorry

end real_roots_if_and_only_if_m_leq_5_l58_58532


namespace passing_probability_l58_58879

def probability_of_passing (p : ℝ) : ℝ :=
  p^3 + p^2 * (1 - p) + (1 - p) * p^2

theorem passing_probability :
  probability_of_passing 0.6 = 0.504 :=
by {
  sorry
}

end passing_probability_l58_58879


namespace minimum_spend_on_boxes_l58_58036

def box_dimensions : ℕ × ℕ × ℕ := (20, 20, 12)
def cost_per_box : ℝ := 0.40
def total_volume : ℕ := 2400000
def volume_of_box (l w h : ℕ) : ℕ := l * w * h
def number_of_boxes (total_vol vol_per_box : ℕ) : ℕ := total_vol / vol_per_box
def total_cost (num_boxes : ℕ) (cost_box : ℝ) : ℝ := num_boxes * cost_box

theorem minimum_spend_on_boxes : total_cost (number_of_boxes total_volume (volume_of_box 20 20 12)) cost_per_box = 200 := by
  sorry

end minimum_spend_on_boxes_l58_58036


namespace find_g_l58_58654

noncomputable def g (x : ℝ) : ℝ := 2 - 4 * x

theorem find_g :
  g 0 = 2 ∧ (∀ x y : ℝ, g (x * y) = g ((3 * x ^ 2 + y ^ 2) / 4) + 3 * (x - y) ^ 2) → ∀ x : ℝ, g x = 2 - 4 * x :=
by
  sorry

end find_g_l58_58654


namespace identify_quadratic_equation_l58_58241

theorem identify_quadratic_equation :
  (¬(∃ x y : ℝ, x^2 - 2*x*y + y^2 = 0) ∧  -- Condition A is not a quadratic equation
   ¬(∃ x : ℝ, x*(x + 3) = x^2 - 1) ∧      -- Condition B is not a quadratic equation
   (∃ x : ℝ, x^2 - 2*x - 3 = 0) ∧         -- Condition C is a quadratic equation
   ¬(∃ x : ℝ, x + (1/x) = 0)) →           -- Condition D is not a quadratic equation
  (true) := sorry

end identify_quadratic_equation_l58_58241


namespace equilibrium_shift_if_K_changes_l58_58987

-- Define the equilibrium constant and its relation to temperature
def equilibrium_constant (T : ℝ) : ℝ := sorry

-- Define the conditions
axiom K_related_to_temp (T₁ T₂ : ℝ) (K₁ K₂ : ℝ) :
  equilibrium_constant T₁ = K₁ ∧ equilibrium_constant T₂ = K₂ → T₁ = T₂ ↔ K₁ = K₂

axiom K_constant_with_concentration_change (T : ℝ) (K : ℝ) (c₁ c₂ : ℝ) :
  equilibrium_constant T = K → equilibrium_constant T = K

axiom K_squared_with_stoichiometric_double (T : ℝ) (K : ℝ) :
  equilibrium_constant (2 * T) = K * K

-- Define the problem to be proved
theorem equilibrium_shift_if_K_changes (T₁ T₂ : ℝ) (K₁ K₂ : ℝ) :
  equilibrium_constant T₁ = K₁ ∧ equilibrium_constant T₂ = K₂ → K₁ ≠ K₂ → T₁ ≠ T₂ := 
sorry

end equilibrium_shift_if_K_changes_l58_58987


namespace competition_results_l58_58332

namespace Competition

-- Define the probabilities for each game
def prob_win_game_A : ℚ := 2 / 3
def prob_win_game_B : ℚ := 1 / 2

-- Define the probability of winning each project (best of five format)
def prob_win_project_A : ℚ := (8 / 27) + (8 / 27) + (16 / 81)
def prob_win_project_B : ℚ := (1 / 8) + (3 / 16) + (3 / 16)

-- Define the distribution of the random variable X (number of projects won by player A)
def P_X_0 : ℚ := (17 / 81) * (1 / 2)
def P_X_2 : ℚ := (64 / 81) * (1 / 2)
def P_X_1 : ℚ := 1 - P_X_0 - P_X_2

-- Define the mathematical expectation of X
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2

-- Theorem stating the results
theorem competition_results :
  prob_win_project_A = 64 / 81 ∧
  prob_win_project_B = 1 / 2 ∧
  P_X_0 = 17 / 162 ∧
  P_X_1 = 81 / 162 ∧
  P_X_2 = 64 / 162 ∧
  E_X = 209 / 162 :=
by sorry

end Competition

end competition_results_l58_58332


namespace train_A_reaches_destination_in_6_hours_l58_58452

noncomputable def t : ℕ := 
  let tA := 110
  let tB := 165
  let tB_time := 4
  (tB * tB_time) / tA

theorem train_A_reaches_destination_in_6_hours :
  t = 6 := by
  sorry

end train_A_reaches_destination_in_6_hours_l58_58452


namespace valid_votes_correct_l58_58627

noncomputable def Total_votes : ℕ := 560000
noncomputable def Percentages_received : Fin 4 → ℚ 
| 0 => 0.4
| 1 => 0.35
| 2 => 0.15
| 3 => 0.1

noncomputable def Percentages_invalid : Fin 4 → ℚ 
| 0 => 0.12
| 1 => 0.18
| 2 => 0.25
| 3 => 0.3

noncomputable def Votes_received (i : Fin 4) : ℚ := Total_votes * Percentages_received i

noncomputable def Invalid_votes (i : Fin 4) : ℚ := Votes_received i * Percentages_invalid i

noncomputable def Valid_votes (i : Fin 4) : ℚ := Votes_received i - Invalid_votes i

def A_valid_votes := 197120
def B_valid_votes := 160720
def C_valid_votes := 63000
def D_valid_votes := 39200

theorem valid_votes_correct :
  Valid_votes 0 = A_valid_votes ∧
  Valid_votes 1 = B_valid_votes ∧
  Valid_votes 2 = C_valid_votes ∧
  Valid_votes 3 = D_valid_votes := by
  sorry

end valid_votes_correct_l58_58627


namespace total_amount_paid_l58_58210

def apples_kg := 8
def apples_rate := 70
def mangoes_kg := 9
def mangoes_rate := 65
def oranges_kg := 5
def oranges_rate := 50
def bananas_kg := 3
def bananas_rate := 30

def total_amount := (apples_kg * apples_rate) + (mangoes_kg * mangoes_rate) + (oranges_kg * oranges_rate) + (bananas_kg * bananas_rate)

theorem total_amount_paid : total_amount = 1485 := by
  sorry

end total_amount_paid_l58_58210


namespace intersection_points_count_l58_58173

theorem intersection_points_count (A : ℝ) (hA : A > 0) :
  ((A > 1 / 4) → ∃! (x y : ℝ), (y = A * x^2) ∧ (x^2 + y^2 = 4 * y) ∧
                              (x ≠ 0 ∨ y ≠ 0)) ∧
  ((A ≤ 1 / 4) → ∃! (x y : ℝ), (y = A * x^2) ∧ (x^2 + y^2 = 4 * y)) :=
by
  sorry

end intersection_points_count_l58_58173


namespace distance_is_3_l58_58967

-- define the distance between Masha's and Misha's homes
def distance_between_homes (d : ℝ) : Prop :=
  -- Masha and Misha meet 1 kilometer from Masha's home in the first occasion
  (∃ v_m v_i : ℝ, v_m > 0 ∧ v_i > 0 ∧
  1 / v_m = (d - 1) / v_i) ∧

  -- On the second occasion, Masha walked at twice her original speed,
  -- and Misha walked at half his original speed, and they met 1 kilometer away from Misha's home.
  (∃ v_m v_i : ℝ, v_m > 0 ∧ v_i > 0 ∧
  1 / (2 * v_m) = 2 * (d - 1) / (0.5 * v_i))

-- The theorem to prove the distance is 3
theorem distance_is_3 : distance_between_homes 3 :=
  sorry

end distance_is_3_l58_58967


namespace find_magnitude_of_z_l58_58759

open Complex

theorem find_magnitude_of_z
    (z : ℂ)
    (h : z^4 = 80 - 96 * I) : abs z = 5^(3/4) :=
by sorry

end find_magnitude_of_z_l58_58759


namespace speed_ratio_l58_58753

theorem speed_ratio (v_A v_B : ℝ) (h : 71 / v_B = 142 / v_A) : v_A / v_B = 2 :=
by
  sorry

end speed_ratio_l58_58753


namespace probability_at_least_two_tails_l58_58763

def fair_coin_prob (n : ℕ) : ℚ :=
  (1 / 2 : ℚ)^n

def at_least_two_tails_in_next_three_flips : ℚ :=
  1 - (fair_coin_prob 3 + 3 * fair_coin_prob 3)

theorem probability_at_least_two_tails :
  at_least_two_tails_in_next_three_flips = 1 / 2 := 
by
  sorry

end probability_at_least_two_tails_l58_58763


namespace smallest_second_term_l58_58569

theorem smallest_second_term (a d : ℕ) (h1 : 5 * a + 10 * d = 95) (h2 : a > 0) (h3 : d > 0) : 
  a + d = 10 :=
sorry

end smallest_second_term_l58_58569


namespace green_peaches_eq_three_l58_58886

theorem green_peaches_eq_three (p r g : ℕ) (h1 : p = r + g) (h2 : r + 2 * g = p + 3) : g = 3 := 
by 
  sorry

end green_peaches_eq_three_l58_58886


namespace min_value_x2_y2_l58_58957

theorem min_value_x2_y2 (x y : ℝ) (h : 2 * x + y + 5 = 0) : x^2 + y^2 ≥ 5 :=
by
  sorry

end min_value_x2_y2_l58_58957


namespace largest_value_of_c_l58_58224

theorem largest_value_of_c : ∃ c, (∀ x : ℝ, x^2 - 6 * x + c = 1 → c ≤ 10) :=
sorry

end largest_value_of_c_l58_58224


namespace find_A_l58_58721

variable (A B x : ℝ)
variable (hB : B ≠ 0)
variable (h : f (g 2) = 0)
def f := λ x => A * x^3 - B
def g := λ x => B * x^2

theorem find_A (hB : B ≠ 0) (h : (λ x => A * x^3 - B) ((λ x => B * x^2) 2) = 0) : 
  A = 1 / (64 * B^2) :=
  sorry

end find_A_l58_58721


namespace proof_aim_l58_58126

variables (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + (2 - a) = 0

theorem proof_aim (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 :=
sorry

end proof_aim_l58_58126


namespace order_a_c_b_l58_58016

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 4 / Real.log 3
noncomputable def c : ℝ := Real.log 8 / Real.log 5

theorem order_a_c_b : a > c ∧ c > b := 
by {
  sorry
}

end order_a_c_b_l58_58016


namespace circumscribed_quadrilateral_converse_arithmetic_progression_l58_58566

theorem circumscribed_quadrilateral (a b c d : ℝ) (k : ℝ) (h1 : b = a + k) (h2 : d = a + 2 * k) (h3 : c = a + 3 * k) :
  a + c = b + d :=
by
  sorry

theorem converse_arithmetic_progression (a b c d : ℝ) (h : a + c = b + d) :
  ∃ k : ℝ, b = a + k ∧ d = a + 2 * k ∧ c = a + 3 * k :=
by
  sorry

end circumscribed_quadrilateral_converse_arithmetic_progression_l58_58566


namespace final_score_l58_58400

-- Definitions based on the conditions
def bullseye_points : ℕ := 50
def miss_points : ℕ := 0
def half_bullseye_points : ℕ := bullseye_points / 2

-- Statement to prove
theorem final_score : bullseye_points + miss_points + half_bullseye_points = 75 :=
by
  sorry

end final_score_l58_58400


namespace books_in_final_category_l58_58740

-- Define the number of initial books
def initial_books : ℕ := 400

-- Define the number of divisions
def num_divisions : ℕ := 4

-- Define the iterative division process
def final_books (initial : ℕ) (divisions : ℕ) : ℕ :=
  initial / (2 ^ divisions)

-- State the theorem
theorem books_in_final_category : final_books initial_books num_divisions = 25 := by
  sorry

end books_in_final_category_l58_58740


namespace exp_inequality_solution_l58_58391

theorem exp_inequality_solution (x : ℝ) (h : 1 < Real.exp x ∧ Real.exp x < 2) : 0 < x ∧ x < Real.log 2 :=
by
  sorry

end exp_inequality_solution_l58_58391


namespace function_domain_exclusion_l58_58872

theorem function_domain_exclusion (x : ℝ) :
  (∃ y, y = 2 / (x - 8)) ↔ x ≠ 8 :=
sorry

end function_domain_exclusion_l58_58872


namespace isosceles_triangle_base_length_l58_58272

theorem isosceles_triangle_base_length (b : ℕ) (h₁ : 6 + 6 + b = 20) : b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l58_58272


namespace second_number_is_72_l58_58293

-- Define the necessary variables and conditions
variables (x y : ℕ)
variables (h_first_num : x = 48)
variables (h_ratio : 48 / 8 = x / y)
variables (h_LCM : Nat.lcm x y = 432)

-- State the problem as a theorem
theorem second_number_is_72 : y = 72 :=
by
  sorry

end second_number_is_72_l58_58293


namespace antonella_purchase_l58_58115

theorem antonella_purchase
  (total_coins : ℕ)
  (coin_value : ℕ → ℕ)
  (num_toonies : ℕ)
  (initial_loonies : ℕ)
  (initial_toonies : ℕ)
  (total_value : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (H1 : total_coins = 10)
  (H2 : coin_value 1 = 1)
  (H3 : coin_value 2 = 2)
  (H4 : initial_toonies = 4)
  (H5 : initial_loonies = total_coins - initial_toonies)
  (H6 : total_value = initial_loonies * coin_value 1 + initial_toonies * coin_value 2)
  (H7 : amount_spent = 3)
  (H8 : amount_left = total_value - amount_spent)
  (H9 : amount_left = 11) :
  ∃ (used_loonies used_toonies : ℕ), used_loonies = 1 ∧ used_toonies = 1 ∧ (used_loonies * coin_value 1 + used_toonies * coin_value 2 = amount_spent) :=
by
  sorry

end antonella_purchase_l58_58115


namespace minions_mistake_score_l58_58873

theorem minions_mistake_score :
  (minions_left_phone_on_untrusted_website ∧
   downloaded_file_from_untrusted_source ∧
   guidelines_by_cellular_operators ∧
   avoid_sharing_personal_info ∧
   unverified_files_may_be_harmful ∧
   double_extensions_signify_malicious_software) →
  score = 21 :=
by
  -- Here we would provide the proof steps which we skip with sorry
  sorry

end minions_mistake_score_l58_58873


namespace norb_age_is_47_l58_58199

section NorbAge

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def exactlyHalfGuessesTooLow (guesses : List ℕ) (age : ℕ) : Prop :=
  (guesses.filter (λ x => x < age)).length = (guesses.length / 2)

def oneGuessOffByTwo (guesses : List ℕ) (age : ℕ) : Prop :=
  guesses.any (λ x => x = age + 2 ∨ x = age - 2)

def validAge (guesses : List ℕ) (age : ℕ) : Prop :=
  exactlyHalfGuessesTooLow guesses age ∧ oneGuessOffByTwo guesses age ∧ isPrime age

theorem norb_age_is_47 : validAge [23, 29, 33, 35, 39, 41, 46, 48, 50, 54] 47 :=
sorry

end NorbAge

end norb_age_is_47_l58_58199


namespace problem1_problem2_l58_58669

open Real -- Open the Real namespace for trigonometric functions

-- Part 1: Prove cos(5π + α) * tan(α - 7π) = 4/5 given π < α < 2π and cos α = 3/5
theorem problem1 (α : ℝ) (hα1 : π < α) (hα2 : α < 2 * π) (hcos : cos α = 3 / 5) : 
  cos (5 * π + α) * tan (α - 7 * π) = 4 / 5 := sorry

-- Part 2: Prove sin(π/3 + α) = √3/3 given cos (π/6 - α) = √3/3
theorem problem2 (α : ℝ) (hcos : cos (π / 6 - α) = sqrt 3 / 3) : 
  sin (π / 3 + α) = sqrt 3 / 3 := sorry

end problem1_problem2_l58_58669


namespace exists_super_number_B_l58_58806

-- Define a function is_super_number to identify super numbers.
def is_super_number (A : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 ≤ A n ∧ A n < 10

-- Define a function zero_super_number to represent the super number with all digits zero.
def zero_super_number (n : ℕ) := 0

-- Task: Prove the existence of B such that A + B = zero_super_number.
theorem exists_super_number_B (A : ℕ → ℕ) (hA : is_super_number A) :
  ∃ B : ℕ → ℕ, is_super_number B ∧ (∀ n : ℕ, (A n + B n) % 10 = zero_super_number n) :=
sorry

end exists_super_number_B_l58_58806


namespace suraya_picked_more_apples_l58_58986

theorem suraya_picked_more_apples (k c s : ℕ)
  (h_kayla : k = 20)
  (h_caleb : c = k - 5)
  (h_suraya : s = k + 7) :
  s - c = 12 :=
by
  -- Mark this as a place where the proof can be provided
  sorry

end suraya_picked_more_apples_l58_58986


namespace min_value_g_range_of_m_l58_58318

section
variable (x : ℝ)
noncomputable def g (x : ℝ) := Real.exp x - x

theorem min_value_g :
  (∀ x : ℝ, g x ≥ g 0) ∧ g 0 = 1 := 
by 
  sorry

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2 * x - m) / g x > x) → m < Real.log 2 ^ 2 := 
by 
  sorry
end

end min_value_g_range_of_m_l58_58318


namespace solution_set_of_inequality_l58_58060

theorem solution_set_of_inequality (x : ℝ) : 
  (x ≠ 0 ∧ (x * (x - 1)) ≤ 0) ↔ 0 < x ∧ x ≤ 1 :=
sorry

end solution_set_of_inequality_l58_58060


namespace trigonometric_identity_l58_58620

theorem trigonometric_identity 
  (deg7 deg37 deg83 : ℝ)
  (h7 : deg7 = 7) 
  (h37 : deg37 = 37) 
  (h83 : deg83 = 83) 
  : (Real.sin (deg7 * Real.pi / 180) * Real.cos (deg37 * Real.pi / 180) - Real.sin (deg83 * Real.pi / 180) * Real.sin (deg37 * Real.pi / 180) = -1/2) :=
sorry

end trigonometric_identity_l58_58620


namespace quiz_true_false_questions_l58_58859

theorem quiz_true_false_questions (n : ℕ) 
  (h1 : 2^n - 2 ≠ 0) 
  (h2 : (2^n - 2) * 16 = 224) : 
  n = 4 := 
sorry

end quiz_true_false_questions_l58_58859


namespace dance_pairs_exist_l58_58116

variable {Boy Girl : Type} 

-- Define danced_with relation
variable (danced_with : Boy → Girl → Prop)

-- Given conditions
variable (H1 : ∀ (b : Boy), ∃ (g : Girl), ¬ danced_with b g)
variable (H2 : ∀ (g : Girl), ∃ (b : Boy), danced_with b g)

-- Proof that desired pairs exist
theorem dance_pairs_exist :
  ∃ (M1 M2 : Boy) (D1 D2 : Girl),
    danced_with M1 D1 ∧
    danced_with M2 D2 ∧
    ¬ danced_with M1 D2 ∧
    ¬ danced_with M2 D1 :=
sorry

end dance_pairs_exist_l58_58116


namespace delacroix_band_max_members_l58_58534

theorem delacroix_band_max_members :
  ∃ n : ℕ, 30 * n % 28 = 6 ∧ 30 * n < 1200 ∧ 30 * n = 930 :=
by
  sorry

end delacroix_band_max_members_l58_58534


namespace digits_to_replace_l58_58577

theorem digits_to_replace (a b c d e f : ℕ) :
  (a = 1) →
  (b < 5) →
  (c = 8) →
  (d = 1) →
  (e = 0) →
  (f = 4) →
  (100 * a + 10 * b + c)^2 = 10000 * d + 1000 * e + 100 * f + 10 * f + f :=
  by
    intros ha hb hc hd he hf 
    sorry

end digits_to_replace_l58_58577


namespace karlson_word_count_l58_58726

def single_word_count : Nat := 9
def ten_to_nineteen_count : Nat := 10
def two_word_count (num_tens_units : Nat) : Nat := 2 * num_tens_units

def count_words_1_to_99 : Nat :=
  let single_word := single_word_count + ten_to_nineteen_count
  let two_word := two_word_count (99 - (single_word_count + ten_to_nineteen_count))
  single_word + two_word

def prefix_hundred (count_1_to_99 : Nat) : Nat := 9 * count_1_to_99
def extra_prefix (num_two_word_transformed : Nat) : Nat := 9 * num_two_word_transformed

def total_words : Nat :=
  let first_99 := count_words_1_to_99
  let nine_hundreds := prefix_hundred count_words_1_to_99 + extra_prefix 72
  first_99 + nine_hundreds + 37

theorem karlson_word_count : total_words = 2611 :=
  by
    sorry

end karlson_word_count_l58_58726


namespace max_sum_abs_values_l58_58935

-- Define the main problem in Lean
theorem max_sum_abs_values (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  |a| + |b| + |c| ≤ 3 :=
by
  intros h
  sorry

end max_sum_abs_values_l58_58935


namespace value_of_x_plus_y_l58_58399

theorem value_of_x_plus_y 
  (x y : ℝ) 
  (h1 : -x = 3) 
  (h2 : |y| = 5) : 
  x + y = 2 ∨ x + y = -8 := 
  sorry

end value_of_x_plus_y_l58_58399


namespace add_ten_to_certain_number_l58_58505

theorem add_ten_to_certain_number (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 :=
by
  sorry

end add_ten_to_certain_number_l58_58505


namespace arithmetic_geometric_sequence_formula_l58_58403

theorem arithmetic_geometric_sequence_formula :
  ∃ (a d : ℝ), (3 * a = 6) ∧
  ((5 - d) * (15 + d) = 64) ∧
  (∀ (n : ℕ), n ≥ 3 → (∃ (b_n : ℝ), b_n = 2 ^ (n - 1))) :=
by
  sorry

end arithmetic_geometric_sequence_formula_l58_58403


namespace clock_angle_5_30_l58_58281

theorem clock_angle_5_30 (h_degree : ℕ → ℝ) (m_degree : ℕ → ℝ) (hours_pos : ℕ → ℝ) :
  (h_degree 12 = 360) →
  (m_degree 60 = 360) →
  (hours_pos 5 + h_degree 1 - (m_degree 30 / 2) = 165) →
  (m_degree 30 = 180) →
  ∃ θ : ℝ, θ = abs (m_degree 30 - (hours_pos 5 + h_degree 1 - (m_degree 30 / 2))) ∧ θ = 15 :=
by
  sorry

end clock_angle_5_30_l58_58281


namespace one_in_set_A_l58_58384

theorem one_in_set_A : 1 ∈ {x | x ≥ -1} :=
sorry

end one_in_set_A_l58_58384


namespace ordered_pair_correct_l58_58381

def find_ordered_pair (s m : ℚ) : Prop :=
  (∀ t : ℚ, (∃ x y : ℚ, x = -3 + t * m ∧ y = s + t * (-7) ∧ y = (3/4) * x + 5))
  ∧ s = 11/4 ∧ m = -28/3

theorem ordered_pair_correct :
  find_ordered_pair (11/4) (-28/3) :=
by
  sorry

end ordered_pair_correct_l58_58381


namespace symmetric_coordinates_l58_58032

-- Define the point A as a tuple of its coordinates
def A : Prod ℤ ℤ := (-1, 2)

-- Define what it means for point A' to be symmetric to the origin
def symmetric_to_origin (p : Prod ℤ ℤ) : Prod ℤ ℤ :=
  (-p.1, -p.2)

-- The theorem we need to prove
theorem symmetric_coordinates :
  symmetric_to_origin A = (1, -2) :=
by
  sorry

end symmetric_coordinates_l58_58032


namespace number_of_classmates_ate_cake_l58_58239

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l58_58239


namespace decimal_fraction_error_l58_58166

theorem decimal_fraction_error (A B C D E : ℕ) (hA : A < 100) 
    (h10B : 10 * B = A + C) (h10C : 10 * C = 6 * A + D) (h10D : 10 * D = 7 * A + E) 
    (hBCDE_lt_A : B < A ∧ C < A ∧ D < A ∧ E < A) : 
    false :=
sorry

end decimal_fraction_error_l58_58166


namespace final_number_proof_l58_58789

/- Define the symbols and their corresponding values -/
def cat := 1
def chicken := 5
def crab := 2
def bear := 4
def goat := 3

/- Define the equations from the conditions -/
axiom row4_eq : 5 * crab = 10
axiom col5_eq : 4 * crab + goat = 11
axiom row2_eq : 2 * goat + crab + 2 * bear = 16
axiom col2_eq : cat + bear + 2 * goat + crab = 13
axiom col3_eq : 2 * crab + 2 * chicken + goat = 17

/- Final number is derived by concatenating digits -/
def final_number := cat * 10000 + chicken * 1000 + crab * 100 + bear * 10 + goat

/- Theorem to prove the final number is 15243 -/
theorem final_number_proof : final_number = 15243 := by
  -- Proof steps to be provided here.
  sorry

end final_number_proof_l58_58789


namespace tickets_sold_l58_58326

theorem tickets_sold (T : ℕ) (h1 : 3 * T / 4 > 0)
    (h2 : 5 * (T / 4) / 9 > 0)
    (h3 : 80 > 0)
    (h4 : 20 > 0) :
    (1 / 4 * T - 5 / 36 * T = 100) -> T = 900 :=
by
  sorry

end tickets_sold_l58_58326


namespace cake_recipe_l58_58265

theorem cake_recipe (flour : ℕ) (milk_per_200ml : ℕ) (egg_per_200ml : ℕ) (total_flour : ℕ)
  (h1 : milk_per_200ml = 60)
  (h2 : egg_per_200ml = 1)
  (h3 : total_flour = 800) :
  (total_flour / 200 * milk_per_200ml = 240) ∧ (total_flour / 200 * egg_per_200ml = 4) :=
by
  sorry

end cake_recipe_l58_58265


namespace camilla_blueberry_jelly_beans_l58_58349

theorem camilla_blueberry_jelly_beans (b c : ℕ) 
  (h1 : b = 3 * c)
  (h2 : b - 20 = 2 * (c - 5)) : 
  b = 30 := 
sorry

end camilla_blueberry_jelly_beans_l58_58349


namespace dot_product_calculation_l58_58540

def vector := (ℤ × ℤ)

def dot_product (v1 v2 : vector) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : vector := (1, 3)
def b : vector := (-1, 2)

def scalar_mult (c : ℤ) (v : vector) : vector :=
  (c * v.1, c * v.2)

def vector_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

theorem dot_product_calculation :
  dot_product (vector_add (scalar_mult 2 a) b) b = 15 := by
  sorry

end dot_product_calculation_l58_58540


namespace radius_of_circle_l58_58433

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

-- Prove that given the circle's equation, the radius is 1
theorem radius_of_circle (x y : ℝ) :
  circle_equation x y → ∃ (r : ℝ), r = 1 :=
by
  sorry

end radius_of_circle_l58_58433


namespace part_a_part_b_l58_58402

-- Part (a): Proving that 91 divides n^37 - n for all integers n
theorem part_a (n : ℤ) : 91 ∣ (n ^ 37 - n) := 
sorry

-- Part (b): Finding the largest k that divides n^37 - n for all integers n is 3276
theorem part_b (n : ℤ) : ∀ k : ℤ, (k > 0) → (∀ n : ℤ, k ∣ (n ^ 37 - n)) → k ≤ 3276 :=
sorry

end part_a_part_b_l58_58402


namespace decrease_is_75_86_percent_l58_58868

noncomputable def decrease_percent (x y z : ℝ) : ℝ :=
  let x' := 0.8 * x
  let y' := 0.75 * y
  let z' := 0.9 * z
  let original_value := x^2 * y^3 * z
  let new_value := (x')^2 * (y')^3 * z'
  let decrease_value := original_value - new_value
  decrease_value / original_value

theorem decrease_is_75_86_percent (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) :
  decrease_percent x y z = 0.7586 :=
sorry

end decrease_is_75_86_percent_l58_58868


namespace friend_reading_time_l58_58348

-- Define the conditions
def my_reading_time : ℝ := 1.5 * 60 -- 1.5 hours converted to minutes
def friend_speed_multiplier : ℝ := 5 -- Friend reads 5 times faster than I do
def distraction_time : ℝ := 15 -- Friend is distracted for 15 minutes

-- Define the time taken for my friend to read the book accounting for distraction
theorem friend_reading_time :
  (my_reading_time / friend_speed_multiplier) + distraction_time = 33 := by
  sorry

end friend_reading_time_l58_58348


namespace fraction_of_men_collected_dues_l58_58774

theorem fraction_of_men_collected_dues
  (M W : ℕ)
  (x : ℚ)
  (h1 : 45 * x * M + 5 * W = 17760)
  (h2 : M + W = 3552)
  (h3 : 1 / 12 * W = W / 12) :
  x = 1 / 9 :=
by
  -- Proof steps would go here
  sorry

end fraction_of_men_collected_dues_l58_58774


namespace farmer_tomatoes_l58_58407

theorem farmer_tomatoes (t p l : ℕ) (H1 : t = 97) (H2 : p = 83) : l = t - p → l = 14 :=
by {
  sorry
}

end farmer_tomatoes_l58_58407


namespace dragons_at_meeting_l58_58231

def dragon_meeting : Prop :=
  ∃ (x y : ℕ), 
    (2 * x + 7 * y = 26) ∧ 
    (x + y = 8)

theorem dragons_at_meeting : dragon_meeting :=
by
  sorry

end dragons_at_meeting_l58_58231


namespace area_of_circle_r_is_16_percent_of_circle_s_l58_58761

open Real

variables (Ds Dr Rs Rr As Ar : ℝ)

def circle_r_is_40_percent_of_circle_s (Ds Dr : ℝ) := Dr = 0.40 * Ds
def radius_of_circle (D : ℝ) (R : ℝ) := R = D / 2
def area_of_circle (R : ℝ) (A : ℝ) := A = π * R^2
def percentage_area (As Ar : ℝ) (P : ℝ) := P = (Ar / As) * 100

theorem area_of_circle_r_is_16_percent_of_circle_s :
  ∀ (Ds Dr Rs Rr As Ar : ℝ),
    circle_r_is_40_percent_of_circle_s Ds Dr →
    radius_of_circle Ds Rs →
    radius_of_circle Dr Rr →
    area_of_circle Rs As →
    area_of_circle Rr Ar →
    percentage_area As Ar 16 := by
  intros Ds Dr Rs Rr As Ar H1 H2 H3 H4 H5
  sorry

end area_of_circle_r_is_16_percent_of_circle_s_l58_58761


namespace gcf_lcm_60_72_l58_58499

def gcf_lcm_problem (a b : ℕ) : Prop :=
  gcd a b = 12 ∧ lcm a b = 360

theorem gcf_lcm_60_72 : gcf_lcm_problem 60 72 :=
by {
  sorry
}

end gcf_lcm_60_72_l58_58499


namespace first_complete_row_cover_l58_58691

def is_shaded_square (n : ℕ) : ℕ := n ^ 2

def row_number (square_number : ℕ) : ℕ :=
  (square_number + 9) / 10 -- ceiling of square_number / 10

theorem first_complete_row_cover : ∃ n, ∀ r : ℕ, 1 ≤ r ∧ r ≤ 10 → ∃ k : ℕ, is_shaded_square k ≤ n ∧ row_number (is_shaded_square k) = r :=
by
  use 100
  intros r h
  sorry

end first_complete_row_cover_l58_58691


namespace concert_cost_l58_58410

noncomputable def ticket_price : ℝ := 50.0
noncomputable def processing_fee_rate : ℝ := 0.15
noncomputable def parking_fee : ℝ := 10.0
noncomputable def entrance_fee : ℝ := 5.0
def number_of_people : ℕ := 2

noncomputable def processing_fee_per_ticket : ℝ := processing_fee_rate * ticket_price
noncomputable def total_cost_per_ticket : ℝ := ticket_price + processing_fee_per_ticket
noncomputable def total_ticket_cost : ℝ := number_of_people * total_cost_per_ticket
noncomputable def total_cost_with_parking : ℝ := total_ticket_cost + parking_fee
noncomputable def total_entrance_fee : ℝ := number_of_people * entrance_fee
noncomputable def total_cost : ℝ := total_cost_with_parking + total_entrance_fee

theorem concert_cost : total_cost = 135.0 := by
  sorry

end concert_cost_l58_58410


namespace value_of_c7_l58_58744

def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

theorem value_of_c7 : c 7 = 448 := by
  sorry

end value_of_c7_l58_58744


namespace population_total_l58_58602

theorem population_total (total_population layers : ℕ) (ratio_A ratio_B ratio_C : ℕ) 
(sample_capacity : ℕ) (prob_ab_in_C : ℚ) 
(h1 : ratio_A = 3)
(h2 : ratio_B = 6)
(h3 : ratio_C = 1)
(h4 : sample_capacity = 20)
(h5 : prob_ab_in_C = 1 / 21)
(h6 : total_population = 10 * ratio_C) :
  total_population = 70 := 
by 
  sorry

end population_total_l58_58602


namespace g_one_fourth_l58_58426

noncomputable def g : ℝ → ℝ := sorry

theorem g_one_fourth :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1) ∧  -- g(x) is defined for 0 ≤ x ≤ 1
  g 0 = 0 ∧                                    -- g(0) = 0
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧ -- g is non-decreasing
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧ -- symmetric property
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2)   -- scaling property
  → g (1/4) = 1/2 :=
sorry

end g_one_fourth_l58_58426


namespace half_angle_quadrant_l58_58870

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 / 2 * Real.pi)
  : (k % 2 = 0 → k * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < k * Real.pi + 3 / 4 * Real.pi) ∨
    (k % 2 = 1 → (k + 1) * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < (k + 1) * Real.pi + 3 / 4 * Real.pi) :=
by
  sorry

end half_angle_quadrant_l58_58870


namespace find_integers_l58_58484

theorem find_integers (x : ℤ) : x^2 < 3 * x → x = 1 ∨ x = 2 := by
  sorry

end find_integers_l58_58484


namespace proof_equivalent_problem_l58_58936

-- Definition of conditions
def cost_condition_1 (x y : ℚ) : Prop := 500 * x + 40 * y = 1250
def cost_condition_2 (x y : ℚ) : Prop := 1000 * x + 20 * y = 1000
def budget_condition (a b : ℕ) (total_masks : ℕ) (budget : ℕ) : Prop := 2 * a + (total_masks - a) / 2 + 25 * b = budget

-- Main theorem
theorem proof_equivalent_problem : 
  ∃ (x y : ℚ) (a b : ℕ), 
    cost_condition_1 x y ∧
    cost_condition_2 x y ∧
    (x = 1 / 2) ∧ 
    (y = 25) ∧
    (budget_condition a b 200 400) ∧
    ((a = 150 ∧ b = 3) ∨
     (a = 100 ∧ b = 6) ∨
     (a = 50 ∧ b = 9)) :=
by {
  sorry -- The proof steps are not required
}

end proof_equivalent_problem_l58_58936


namespace bushes_needed_for_octagon_perimeter_l58_58140

theorem bushes_needed_for_octagon_perimeter
  (side_length : ℝ) (spacing : ℝ)
  (octagonal : ∀ (s : ℝ), s = 8 → 8 * s = 64)
  (spacing_condition : ∀ (p : ℝ), p = 64 → p / spacing = 32) :
  spacing = 2 → side_length = 8 → (64 / 2 = 32) := 
by
  sorry

end bushes_needed_for_octagon_perimeter_l58_58140


namespace inequality_not_true_l58_58501

variable (a b c : ℝ)

theorem inequality_not_true (h : a < b) : ¬ (-3 * a < -3 * b) :=
by
  sorry

end inequality_not_true_l58_58501


namespace area_of_shaded_region_l58_58824

noncomputable def r2 : ℝ := Real.sqrt 20
noncomputable def r1 : ℝ := 3 * r2

theorem area_of_shaded_region :
  let area := π * (r1 ^ 2) - π * (r2 ^ 2)
  area = 160 * π :=
by
  sorry

end area_of_shaded_region_l58_58824


namespace polygonal_pyramid_faces_l58_58395

/-- A polygonal pyramid is a three-dimensional solid. Its base is a regular polygon. Each of the vertices of the polygonal base is connected to a single point, called the apex. The sum of the number of edges and the number of vertices of a particular polygonal pyramid is 1915. This theorem states that the number of faces of this pyramid is 639. -/
theorem polygonal_pyramid_faces (n : ℕ) (hn : 2 * n + (n + 1) = 1915) : n + 1 = 639 :=
by
  sorry

end polygonal_pyramid_faces_l58_58395


namespace simplify_expression_l58_58729

noncomputable def proof_problem (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : Prop :=
  (1 / (1 + a + a * b) + 1 / (1 + b + b * c) + 1 / (1 + c + c * a)) = 1

theorem simplify_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) :
  proof_problem a b c h h_abc :=
by sorry

end simplify_expression_l58_58729


namespace period_is_3_years_l58_58186

def gain_of_B_per_annum (principal : ℕ) (rate_A rate_B : ℚ) : ℚ := 
  (rate_B - rate_A) * principal

def period (principal : ℕ) (rate_A rate_B : ℚ) (total_gain : ℚ) : ℚ := 
  total_gain / gain_of_B_per_annum principal rate_A rate_B

theorem period_is_3_years :
  period 1500 (10 / 100) (11.5 / 100) 67.5 = 3 :=
by
  sorry

end period_is_3_years_l58_58186


namespace cuboid_ratio_l58_58655

theorem cuboid_ratio (length breadth height: ℕ) (h_length: length = 90) (h_breadth: breadth = 75) (h_height: height = 60) : 
(length / Nat.gcd length (Nat.gcd breadth height) = 6) ∧ 
(breadth / Nat.gcd length (Nat.gcd breadth height) = 5) ∧ 
(height / Nat.gcd length (Nat.gcd breadth height) = 4) := by 
  -- intentionally skipped proof 
  sorry

end cuboid_ratio_l58_58655


namespace polynomial_division_l58_58497

theorem polynomial_division (a b c : ℤ) :
  (∀ x : ℝ, (17 * x^2 - 3 * x + 4) - (a * x^2 + b * x + c) = (5 * x + 6) * (2 * x + 1)) →
  a - b - c = 29 := by
  sorry

end polynomial_division_l58_58497


namespace man_work_days_l58_58914

theorem man_work_days (M : ℕ) (h1 : (1 : ℝ)/M + (1 : ℝ)/10 = 1/5) : M = 10 :=
sorry

end man_work_days_l58_58914


namespace trig_problem_1_trig_problem_2_l58_58676

-- Problem (1)
theorem trig_problem_1 (α : ℝ) (h1 : Real.tan (π + α) = -4 / 3) (h2 : 3 * Real.sin α / 4 = -Real.cos α)
  : Real.sin α = -4 / 5 ∧ Real.cos α = 3 / 5 := by
  sorry

-- Problem (2)
theorem trig_problem_2 : Real.sin (25 * π / 6) + Real.cos (26 * π / 3) + Real.tan (-25 * π / 4) = -1 := by
  sorry

end trig_problem_1_trig_problem_2_l58_58676


namespace bob_age_l58_58542

theorem bob_age (a b : ℝ) 
    (h1 : b = 3 * a - 20)
    (h2 : b + a = 70) : 
    b = 47.5 := by
    sorry

end bob_age_l58_58542


namespace trapezoid_area_l58_58130

variables (R₁ R₂ : ℝ)

theorem trapezoid_area (h_eq : h = 4 * R₁ * R₂ / (R₁ + R₂)) (mn_eq : mn = 2 * Real.sqrt (R₁ * R₂)) :
  S_ABCD = 8 * R₁ * R₂ * Real.sqrt (R₁ * R₂) / (R₁ + R₂) :=
sorry

end trapezoid_area_l58_58130


namespace remainder_mod_68_l58_58178

theorem remainder_mod_68 (n : ℕ) (h : 67^67 + 67 ≡ 66 [MOD n]) : n = 68 := 
by 
  sorry

end remainder_mod_68_l58_58178


namespace correct_operation_l58_58450

-- Define the conditions
def cond1 (m : ℝ) : Prop := m^2 + m^3 ≠ m^5
def cond2 (m : ℝ) : Prop := m^2 * m^3 = m^5
def cond3 (m : ℝ) : Prop := (m^2)^3 = m^6

-- Main statement that checks the correct operation
theorem correct_operation (m : ℝ) : cond1 m → cond2 m → cond3 m → (m^2 * m^3 = m^5) :=
by
  intros h1 h2 h3
  exact h2

end correct_operation_l58_58450


namespace decreasing_geometric_sequence_l58_58539

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * q ^ n

theorem decreasing_geometric_sequence (a₁ q : ℝ) (aₙ : ℕ → ℝ) (hₙ : ∀ n, aₙ n = geometric_sequence a₁ q n) 
  (h_condition : 0 < q ∧ q < 1) : ¬(0 < q ∧ q < 1 ↔ ∀ n, aₙ n > aₙ (n + 1)) :=
sorry

end decreasing_geometric_sequence_l58_58539


namespace distance_between_lines_is_sqrt2_l58_58270

noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  |c1 - c2| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines_is_sqrt2 :
  distance_between_parallel_lines 1 1 (-1) 1 = Real.sqrt 2 := 
by 
  sorry

end distance_between_lines_is_sqrt2_l58_58270


namespace problem_statement_l58_58261

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem problem_statement (a b c : ℝ) (h : f a b c (-5) = 3) : f a b c 5 + f a b c (-5) = 4 :=
by
  sorry

end problem_statement_l58_58261


namespace solve_inequality_l58_58536

variable {a x : ℝ}

theorem solve_inequality (h : a > 0) : 
  (ax^2 - (a + 1)*x + 1 < 0) ↔ 
    (if 0 < a ∧ a < 1 then 1 < x ∧ x < 1/a else 
     if a = 1 then false else 
     if a > 1 then 1/a < x ∧ x < 1 else true) :=
  sorry

end solve_inequality_l58_58536


namespace exists_prime_q_l58_58447

theorem exists_prime_q (p : ℕ) (hp : Nat.Prime p) :
  ∃ q, Nat.Prime q ∧ ∀ n, ¬ (q ∣ n^p - p) := by
  sorry

end exists_prime_q_l58_58447


namespace divisibility_condition_of_exponents_l58_58975

theorem divisibility_condition_of_exponents (n : ℕ) (h : n ≥ 1) :
  (∀ a b : ℕ, (11 ∣ a^n + b^n) → (11 ∣ a ∧ 11 ∣ b)) ↔ (n % 2 = 0) :=
sorry

end divisibility_condition_of_exponents_l58_58975


namespace validate_triangle_count_l58_58249

noncomputable def count_valid_triangles : ℕ :=
  let total_points := 25
  let total_triples := (Nat.choose total_points 3)
  let collinear_rows := 5 * (Nat.choose 5 3)
  let collinear_columns := 5 * (Nat.choose 5 3)
  let main_diagonals := 2 * (Nat.choose 5 3)
  let secondary_diagonals := 8 * (Nat.choose 4 3)
  let invalid_triangles := collinear_rows + collinear_columns + main_diagonals + secondary_diagonals
  total_triples - invalid_triangles

theorem validate_triangle_count : count_valid_triangles = 2148 :=
by
  sorry

end validate_triangle_count_l58_58249


namespace range_of_a_l58_58323

theorem range_of_a
  (h : ∀ x : ℝ, |x - 1| + |x - 2| > Real.log (a ^ 2) / Real.log 4) :
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2 :=
sorry

end range_of_a_l58_58323


namespace problem1_problem2_l58_58457
noncomputable section

-- Problem (1) Lean Statement
theorem problem1 : |-4| - (2021 - Real.pi)^0 + (Real.cos (Real.pi / 3))⁻¹ - (-Real.sqrt 3)^2 = 2 :=
by 
  sorry

-- Problem (2) Lean Statement
theorem problem2 (a : ℝ) (h : a ≠ 2 ∧ a ≠ -2) : 
  (1 + 4 / (a^2 - 4)) / (a / (a + 2)) = a / (a - 2) := 
by 
  sorry

end problem1_problem2_l58_58457


namespace fruit_seller_apples_l58_58786

theorem fruit_seller_apples (original_apples : ℝ) (sold_percent : ℝ) (remaining_apples : ℝ)
  (h1 : sold_percent = 0.40)
  (h2 : remaining_apples = 420)
  (h3 : original_apples * (1 - sold_percent) = remaining_apples) :
  original_apples = 700 :=
by
  sorry

end fruit_seller_apples_l58_58786


namespace y_coordinate_midpoint_l58_58175

theorem y_coordinate_midpoint : 
  let L : (ℝ → ℝ) := λ x => x - 1
  let P : (ℝ → ℝ) := λ y => 8 * (y^2)
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    P (L x₁) = y₁ ∧ P (L x₂) = y₂ ∧ 
    L x₁ = y₁ ∧ L x₂ = y₂ ∧ 
    x₁ + x₂ = 10 ∧ y₁ + y₂ = 8 ∧
    (y₁ + y₂) / 2 = 4 := sorry

end y_coordinate_midpoint_l58_58175


namespace appropriate_sampling_method_l58_58189

theorem appropriate_sampling_method (total_staff teachers admin_staff logistics_personnel sample_size : ℕ)
  (h1 : total_staff = 160)
  (h2 : teachers = 120)
  (h3 : admin_staff = 16)
  (h4 : logistics_personnel = 24)
  (h5 : sample_size = 20) :
  (sample_method : String) -> sample_method = "Stratified sampling" :=
sorry

end appropriate_sampling_method_l58_58189


namespace circle_not_pass_second_quadrant_l58_58680

theorem circle_not_pass_second_quadrant (a : ℝ) : ¬(∃ x y : ℝ, x < 0 ∧ y > 0 ∧ (x - a)^2 + y^2 = 4) → a ≥ 2 :=
by
  intro h
  by_contra
  sorry

end circle_not_pass_second_quadrant_l58_58680


namespace infinite_equal_pairs_l58_58409

theorem infinite_equal_pairs
  (a : ℤ → ℝ)
  (h : ∀ k : ℤ, a k = 1/4 * (a (k - 1) + a (k + 1)))
  (k p : ℤ) (hne : k ≠ p) (heq : a k = a p) :
  ∃ infinite_pairs : ℕ → (ℤ × ℤ), 
  (∀ n : ℕ, (infinite_pairs n).1 ≠ (infinite_pairs n).2) ∧
  (∀ n : ℕ, a (infinite_pairs n).1 = a (infinite_pairs n).2) :=
sorry

end infinite_equal_pairs_l58_58409


namespace sufficient_but_not_necessary_condition_l58_58552

theorem sufficient_but_not_necessary_condition :
  (∀ (x : ℝ), x^2 - 2 * x < 0 → 0 < x ∧ x < 4)
  ∧ ¬(∀ (x : ℝ), 0 < x ∧ x < 4 → x^2 - 2 * x < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l58_58552


namespace volume_of_wedge_l58_58521

theorem volume_of_wedge (d : ℝ) (angle : ℝ) (V : ℝ) (n : ℕ) 
  (h_d : d = 18) 
  (h_angle : angle = 60)
  (h_radius_height : ∀ r h, r = d / 2 ∧ h = d) 
  (h_volume_cylinder : V = π * (d / 2) ^ 2 * d) 
  : n = 729 ↔ V / 2 = n * π :=
by
  sorry

end volume_of_wedge_l58_58521


namespace prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29_l58_58523

theorem prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29 
  (n : ℕ) (h1 : Prime n) (h2 : 20 < n) (h3 : n < 30) (h4 : n % 8 = 5) : n = 29 := 
by
  sorry

end prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29_l58_58523


namespace burn_time_for_structure_l58_58686

noncomputable def time_to_burn_structure (total_toothpicks : ℕ) (burn_time_per_toothpick : ℕ) (adjacent_corners : Bool) : ℕ :=
  if total_toothpicks = 38 ∧ burn_time_per_toothpick = 10 ∧ adjacent_corners = true then 65 else 0

theorem burn_time_for_structure :
  time_to_burn_structure 38 10 true = 65 :=
sorry

end burn_time_for_structure_l58_58686


namespace determine_a_b_l58_58955

theorem determine_a_b (a b : ℤ) :
  (∀ x : ℤ, x^2 + a * x + b = (x - 1) * (x + 4)) → (a = 3 ∧ b = -4) :=
by
  intro h
  sorry

end determine_a_b_l58_58955


namespace johnny_years_ago_l58_58516

theorem johnny_years_ago 
  (J : ℕ) (hJ : J = 8) (X : ℕ) 
  (h : J + 2 = 2 * (J - X)) : 
  X = 3 := by
  sorry

end johnny_years_ago_l58_58516


namespace factor_expression_l58_58876

theorem factor_expression (x : ℝ) : 18 * x^2 + 9 * x - 3 = 3 * (6 * x^2 + 3 * x - 1) :=
by
  sorry

end factor_expression_l58_58876


namespace min_value_function_l58_58823

theorem min_value_function (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (∀ x y : ℝ, x > 1 ∧ y > 1 → (min ((x^2 + y) / (y^2 - 1) + (y^2 + x) / (x^2 - 1)) = 8 / 3)) := 
sorry

end min_value_function_l58_58823


namespace circle_center_coordinates_l58_58714

open Real

noncomputable def circle_center (x y : Real) : Prop := 
  x^2 + y^2 - 4*x + 6*y = 0

theorem circle_center_coordinates :
  ∃ (a b : Real), circle_center a b ∧ a = 2 ∧ b = -3 :=
by
  use 2, -3
  sorry

end circle_center_coordinates_l58_58714


namespace largest_int_mod_6_less_than_100_l58_58524

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l58_58524


namespace roses_remain_unchanged_l58_58299

variable (initial_roses : ℕ) (initial_orchids : ℕ) (final_orchids : ℕ)

def unchanged_roses (roses_now : ℕ) : Prop :=
  roses_now = initial_roses

theorem roses_remain_unchanged :
  initial_roses = 13 → 
  initial_orchids = 84 → 
  final_orchids = 91 →
  ∀ (roses_now : ℕ), unchanged_roses initial_roses roses_now :=
by
  intros _ _ _ _
  simp [unchanged_roses]
  sorry

end roses_remain_unchanged_l58_58299


namespace polynomial_div_simplify_l58_58829

theorem polynomial_div_simplify (x : ℝ) (hx : x ≠ 0) :
  (6 * x ^ 4 - 4 * x ^ 3 + 2 * x ^ 2) / (2 * x ^ 2) = 3 * x ^ 2 - 2 * x + 1 :=
by sorry

end polynomial_div_simplify_l58_58829


namespace least_four_digit_perfect_square_and_fourth_power_l58_58639

theorem least_four_digit_perfect_square_and_fourth_power : 
    ∃ (n : ℕ), (1000 ≤ n) ∧ (n < 10000) ∧ (∃ a : ℕ, n = a^2) ∧ (∃ b : ℕ, n = b^4) ∧ 
    (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (∃ a : ℕ, m = a^2) ∧ (∃ b : ℕ, m = b^4) → n ≤ m) ∧ n = 6561 :=
by
  sorry

end least_four_digit_perfect_square_and_fourth_power_l58_58639


namespace concyclic_projections_of_concyclic_quad_l58_58869

variables {A B C D A' B' C' D' : Type*}

def are_concyclic (p1 p2 p3 p4: Type*) : Prop :=
  sorry -- Assume we have a definition for concyclic property of points

def are_orthogonal_projection (x y : Type*) (l : Type*) : Type* :=
  sorry -- Assume we have a definition for orthogonal projection of a point on line

theorem concyclic_projections_of_concyclic_quad
  (hABCD : are_concyclic A B C D)
  (hA'_proj : are_orthogonal_projection A A' (BD))
  (hC'_proj : are_orthogonal_projection C C' (BD))
  (hB'_proj : are_orthogonal_projection B B' (AC))
  (hD'_proj : are_orthogonal_projection D D' (AC)) :
  are_concyclic A' B' C' D' :=
sorry

end concyclic_projections_of_concyclic_quad_l58_58869


namespace eq_of_nonzero_real_x_l58_58340

theorem eq_of_nonzero_real_x (x : ℝ) (hx : x ≠ 0) (a b : ℝ) (ha : a = 9) (hb : b = 18) :
  ((a * x) ^ 10 = (b * x) ^ 5) → x = 2 / 9 :=
by
  sorry

end eq_of_nonzero_real_x_l58_58340


namespace monotonic_increasing_interval_f_l58_58954

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2 * x + 8)

theorem monotonic_increasing_interval_f :
  ∃ I : Set ℝ, (I = Set.Icc (-2) 1) ∧ (∀x1 ∈ I, ∀x2 ∈ I, x1 ≤ x2 → f x1 ≤ f x2) :=
sorry

end monotonic_increasing_interval_f_l58_58954


namespace group_C_both_axis_and_central_l58_58319

def is_axisymmetric (shape : Type) : Prop := sorry
def is_centrally_symmetric (shape : Type) : Prop := sorry

def square : Type := sorry
def rhombus : Type := sorry
def rectangle : Type := sorry
def parallelogram : Type := sorry
def equilateral_triangle : Type := sorry
def isosceles_triangle : Type := sorry

def group_A := [square, rhombus, rectangle, parallelogram]
def group_B := [equilateral_triangle, square, rhombus, rectangle]
def group_C := [square, rectangle, rhombus]
def group_D := [parallelogram, square, isosceles_triangle]

def all_axisymmetric (group : List Type) : Prop :=
  ∀ shape ∈ group, is_axisymmetric shape

def all_centrally_symmetric (group : List Type) : Prop :=
  ∀ shape ∈ group, is_centrally_symmetric shape

theorem group_C_both_axis_and_central :
  (all_axisymmetric group_C ∧ all_centrally_symmetric group_C) ∧
  (∀ (group : List Type), (all_axisymmetric group ∧ all_centrally_symmetric group) →
    group = group_C) :=
by sorry

end group_C_both_axis_and_central_l58_58319


namespace sum_fractions_correct_l58_58815

def sum_of_fractions (f1 f2 f3 f4 f5 f6 f7 : ℚ) : ℚ :=
  f1 + f2 + f3 + f4 + f5 + f6 + f7

theorem sum_fractions_correct : sum_of_fractions (1/3) (1/2) (-5/6) (1/5) (1/4) (-9/20) (-5/6) = -5/6 :=
by
  sorry

end sum_fractions_correct_l58_58815


namespace solve_x_minus_y_l58_58076

theorem solve_x_minus_y :
  (2 = 0.25 * x) → (2 = 0.1 * y) → (x - y = -12) :=
by
  sorry

end solve_x_minus_y_l58_58076


namespace find_number_l58_58614

theorem find_number (x : ℤ) (h : 27 + 2 * x = 39) : x = 6 :=
sorry

end find_number_l58_58614


namespace pool_water_left_l58_58519

theorem pool_water_left 
  (h1_rate: ℝ) (h1_time: ℝ)
  (h2_rate: ℝ) (h2_time: ℝ)
  (h4_rate: ℝ) (h4_time: ℝ)
  (leak_loss: ℝ)
  (h1_rate_eq: h1_rate = 8)
  (h1_time_eq: h1_time = 1)
  (h2_rate_eq: h2_rate = 10)
  (h2_time_eq: h2_time = 2)
  (h4_rate_eq: h4_rate = 14)
  (h4_time_eq: h4_time = 1)
  (leak_loss_eq: leak_loss = 8) :
  (h1_rate * h1_time) + (h2_rate * h2_time) + (h2_rate * h2_time) + (h4_rate * h4_time) - leak_loss = 34 :=
by
  rw [h1_rate_eq, h1_time_eq, h2_rate_eq, h2_time_eq, h4_rate_eq, h4_time_eq, leak_loss_eq]
  norm_num
  sorry

end pool_water_left_l58_58519


namespace bicycle_trip_length_l58_58357

def total_distance (days1 day1 miles1 day2 miles2: ℕ) : ℕ :=
  days1 * miles1 + day2 * miles2

theorem bicycle_trip_length :
  total_distance 12 12 1 6 = 150 :=
by
  sorry

end bicycle_trip_length_l58_58357


namespace calc_problem1_calc_problem2_calc_problem3_calc_problem4_l58_58842

theorem calc_problem1 : (-3 + 8 - 15 - 6 = -16) :=
by
  sorry

theorem calc_problem2 : (-4/13 - (-4/17) + 4/13 + (-13/17) = -9/17) :=
by
  sorry

theorem calc_problem3 : (-25 - (5/4 * 4/5) - (-16) = -10) :=
by
  sorry

theorem calc_problem4 : (-2^4 - (1/2 * (5 - (-3)^2)) = -14) :=
by
  sorry

end calc_problem1_calc_problem2_calc_problem3_calc_problem4_l58_58842


namespace find_a_l58_58712

noncomputable def A (a : ℝ) : Set ℝ := {1, 2, a}
noncomputable def B (a : ℝ) : Set ℝ := {1, a^2 - a}

theorem find_a (a : ℝ) : A a ⊇ B a → a = -1 ∨ a = 0 :=
by
  sorry

end find_a_l58_58712


namespace whitney_greatest_sets_l58_58335

-- Define the conditions: Whitney has 4 T-shirts and 20 buttons.
def num_tshirts := 4
def num_buttons := 20

-- The problem statement: Prove that the greatest number of sets Whitney can make is 4.
theorem whitney_greatest_sets : Nat.gcd num_tshirts num_buttons = 4 := by
  sorry

end whitney_greatest_sets_l58_58335


namespace greatest_possible_value_of_n_l58_58460

theorem greatest_possible_value_of_n (n : ℤ) (h : 101 * n^2 ≤ 6400) : n ≤ 7 :=
by
  sorry

end greatest_possible_value_of_n_l58_58460


namespace total_seats_theater_l58_58461

theorem total_seats_theater (a1 an d n Sn : ℕ) 
    (h1 : a1 = 12) 
    (h2 : d = 2) 
    (h3 : an = 48) 
    (h4 : an = a1 + (n - 1) * d) 
    (h5 : Sn = n * (a1 + an) / 2) : 
    Sn = 570 := 
sorry

end total_seats_theater_l58_58461


namespace radius_of_inscribed_circle_l58_58053

noncomputable def radius_inscribed_circle (AB BC AC : ℝ) (s : ℝ) (K : ℝ) : ℝ := K / s

theorem radius_of_inscribed_circle (AB BC AC : ℝ) (h1: AB = 8) (h2: BC = 8) (h3: AC = 10) :
  radius_inscribed_circle AB BC AC 13 (5 * Real.sqrt 39) = (5 * Real.sqrt 39) / 13 :=
  by
  sorry

end radius_of_inscribed_circle_l58_58053


namespace total_company_pay_monthly_l58_58732

-- Define the given conditions
def hours_josh_works_daily : ℕ := 8
def days_josh_works_weekly : ℕ := 5
def weeks_josh_works_monthly : ℕ := 4
def hourly_rate_josh : ℕ := 9

-- Define Carl's working hours and rate based on the conditions
def hours_carl_works_daily : ℕ := hours_josh_works_daily - 2
def hourly_rate_carl : ℕ := hourly_rate_josh / 2

-- Calculate total hours worked monthly by Josh and Carl
def total_hours_josh_monthly : ℕ := hours_josh_works_daily * days_josh_works_weekly * weeks_josh_works_monthly
def total_hours_carl_monthly : ℕ := hours_carl_works_daily * days_josh_works_weekly * weeks_josh_works_monthly

-- Calculate monthly pay for Josh and Carl
def monthly_pay_josh : ℕ := total_hours_josh_monthly * hourly_rate_josh
def monthly_pay_carl : ℕ := total_hours_carl_monthly * hourly_rate_carl

-- Theorem to prove the total pay for both Josh and Carl in one month
theorem total_company_pay_monthly : monthly_pay_josh + monthly_pay_carl = 1980 := by
  sorry

end total_company_pay_monthly_l58_58732


namespace numerator_of_fraction_l58_58464

-- Define the conditions
def y_pos (y : ℝ) : Prop := y > 0

-- Define the equation
def equation (x y : ℝ) : Prop := x + (3 * y) / 10 = (1 / 2) * y

-- Prove that x = (1/5) * y given the conditions
theorem numerator_of_fraction {y x : ℝ} (h1 : y_pos y) (h2 : equation x y) : x = (1/5) * y :=
  sorry

end numerator_of_fraction_l58_58464


namespace ineq_a3b3c3_l58_58137

theorem ineq_a3b3c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^3 + b^3 + c^3 ≥ a^2 * b + b^2 * c + c^2 * a ∧ (a^3 + b^3 + c^3 = a^2 * b + b^2 * c + c^2 * a ↔ a = b ∧ b = c) :=
by
  sorry

end ineq_a3b3c3_l58_58137


namespace tan_alpha_eq_3_l58_58110

theorem tan_alpha_eq_3 (α : ℝ) (h1 : 0 < α ∧ α < (π / 2))
  (h2 : (Real.sin α)^2 + Real.cos ((π / 2) + 2 * α) = 3 / 10) : Real.tan α = 3 := by
  sorry

end tan_alpha_eq_3_l58_58110


namespace member_age_greater_than_zero_l58_58762

def num_members : ℕ := 23
def avg_age : ℤ := 0
def age_range : Set ℤ := {x | x ≥ -20 ∧ x ≤ 20}
def num_negative_members : ℕ := 5

theorem member_age_greater_than_zero :
  ∃ n : ℕ, n ≤ 18 ∧ (avg_age = 0 ∧ num_members = 23 ∧ num_negative_members = 5 ∧ ∀ age ∈ age_range, age ≥ -20 ∧ age ≤ 20) :=
sorry

end member_age_greater_than_zero_l58_58762


namespace g_x_squared_plus_2_l58_58601

namespace PolynomialProof

open Polynomial

noncomputable def g (x : ℚ) : ℚ := sorry

theorem g_x_squared_plus_2 (x : ℚ) (h : g (x^2 - 2) = x^4 - 6*x^2 + 8) :
  g (x^2 + 2) = x^4 + 2*x^2 + 2 :=
sorry

end PolynomialProof

end g_x_squared_plus_2_l58_58601


namespace initial_number_of_angelfish_l58_58083

/-- The initial number of fish in the tank. -/
def initial_total_fish (A : ℕ) := 94 + A + 89 + 58

/-- The remaining number of fish for each species after sale. -/
def remaining_fish (A : ℕ) := 64 + (A - 48) + 72 + 34

/-- Given: 
1. The total number of remaining fish in the tank is 198.
2. The initial number of fish for each species: 94 guppies, A angelfish, 89 tiger sharks, 58 Oscar fish.
3. The number of fish sold: 30 guppies, 48 angelfish, 17 tiger sharks, 24 Oscar fish.
Prove: The initial number of angelfish is 76. -/
theorem initial_number_of_angelfish (A : ℕ) (h : remaining_fish A = 198) : A = 76 :=
sorry

end initial_number_of_angelfish_l58_58083


namespace quadratic_function_is_parabola_l58_58949

theorem quadratic_function_is_parabola (a : ℝ) (b : ℝ) (c : ℝ) :
  ∃ k h, ∀ x, (y = a * (x - h)^2 + k) ∧ a ≠ 0 → (y = 3 * (x - 2)^2 + 6) → (a = 3 ∧ h = 2 ∧ k = 6) → ∀ x, (y = 3 * (x - 2)^2 + 6) := 
by
  sorry

end quadratic_function_is_parabola_l58_58949


namespace cost_of_item_is_200_l58_58845

noncomputable def cost_of_each_item (x : ℕ) : ℕ :=
  let before_discount := 7 * x -- Total cost before discount
  let discount_part := before_discount - 1000 -- Part of the cost over $1000
  let discount := discount_part / 10 -- 10% of the part over $1000
  let after_discount := before_discount - discount -- Total cost after discount
  after_discount

theorem cost_of_item_is_200 :
  (∃ x : ℕ, cost_of_each_item x = 1360) ↔ x = 200 :=
by
  sorry

end cost_of_item_is_200_l58_58845


namespace union_A_B_inter_A_B_inter_compA_B_l58_58481

-- Extend the universal set U to be the set of all real numbers ℝ
def U : Set ℝ := Set.univ

-- Define set A as the set of all real numbers x such that -3 ≤ x ≤ 4
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}

-- Define set B as the set of all real numbers x such that -1 < x < 5
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}

-- Prove that A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 5}
theorem union_A_B : A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 5} := by
  sorry

-- Prove that A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 4}
theorem inter_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 4} := by
  sorry

-- Define the complement of A in U
def comp_A : Set ℝ := {x : ℝ | x < -3 ∨ x > 4}

-- Prove that (complement_U A) ∩ B = {x : ℝ | 4 < x ∧ x < 5}
theorem inter_compA_B : comp_A ∩ B = {x : ℝ | 4 < x ∧ x < 5} := by
  sorry

end union_A_B_inter_A_B_inter_compA_B_l58_58481


namespace expression_for_f_in_positive_domain_l58_58093

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def given_f (x : ℝ) : ℝ :=
  if x < 0 then 3 * Real.sin x + 4 * Real.cos x + 1 else 0 -- temp def for Lean proof

theorem expression_for_f_in_positive_domain (f : ℝ → ℝ) (h_odd : is_odd_function f)
  (h_neg : ∀ x : ℝ, x < 0 → f x = 3 * Real.sin x + 4 * Real.cos x + 1) :
  ∀ x : ℝ, x > 0 → f x = 3 * Real.sin x - 4 * Real.cos x - 1 :=
by
  intros x hx_pos
  sorry

end expression_for_f_in_positive_domain_l58_58093


namespace speed_of_man_rowing_upstream_l58_58890

theorem speed_of_man_rowing_upstream (Vm Vdownstream Vupstream : ℝ) (hVm : Vm = 40) (hVdownstream : Vdownstream = 45) : Vupstream = 35 :=
by
  sorry

end speed_of_man_rowing_upstream_l58_58890


namespace probability_neither_l58_58805

variable (P : Set ℕ → ℝ) -- Use ℕ as a placeholder for the event space
variables (A B : Set ℕ)
variables (hA : P A = 0.25) (hB : P B = 0.35) (hAB : P (A ∩ B) = 0.15)

theorem probability_neither :
  P (Aᶜ ∩ Bᶜ) = 0.55 :=
by
  sorry

end probability_neither_l58_58805


namespace trip_length_l58_58612

theorem trip_length 
  (total_time : ℝ) (canoe_speed : ℝ) (hike_speed : ℝ) (hike_distance : ℝ)
  (hike_time_eq : hike_distance / hike_speed = 5.4) 
  (canoe_time_eq : total_time - hike_distance / hike_speed = 0.1)
  (canoe_distance_eq : canoe_speed * (total_time - hike_distance / hike_speed) = 1.2)
  (total_time_val : total_time = 5.5)
  (canoe_speed_val : canoe_speed = 12)
  (hike_speed_val : hike_speed = 5)
  (hike_distance_val : hike_distance = 27) :
  total_time = 5.5 → canoe_speed = 12 → hike_speed = 5 → hike_distance = 27 → hike_distance + canoe_speed * (total_time - hike_distance / hike_speed) = 28.2 := 
by
  intro h_total_time h_canoe_speed h_hike_speed h_hike_distance
  rw [h_total_time, h_canoe_speed, h_hike_speed, h_hike_distance]
  sorry

end trip_length_l58_58612


namespace find_a_l58_58696

theorem find_a (a : ℝ) (h : 3 ∈ ({1, a, a - 2} : Set ℝ)) : a = 5 :=
sorry

end find_a_l58_58696


namespace general_term_formula_sum_of_geometric_sequence_l58_58109

-- Definitions for the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = 3

def conditions_1 (a : ℕ → ℤ) : Prop :=
  a 2 + a 4 = 14

-- Definitions for the geometric sequence
def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ q, ∀ n, b (n + 1) = b n * q

def conditions_2 (a b : ℕ → ℤ) : Prop := 
  b 2 = a 2 ∧ 
  b 4 = a 6

-- The main theorem statements for part (I) and part (II)
theorem general_term_formula (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : conditions_1 a) : 
  ∀ n, a n = 3 * n - 2 := 
sorry

theorem sum_of_geometric_sequence (a b : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) - a n = 3)
  (h2 : a 2 + a 4 = 14)
  (h3 : b 2 = a 2)
  (h4 : b 4 = a 6)
  (h5 : geometric_sequence b) :
  ∃ (S7 : ℤ), S7 = 254 ∨ S7 = -86 :=
sorry

end general_term_formula_sum_of_geometric_sequence_l58_58109


namespace full_price_ticket_revenue_l58_58990

theorem full_price_ticket_revenue (f d : ℕ) (p : ℝ) : 
  f + d = 200 → 
  f * p + d * (p / 3) = 3000 → 
  d = 200 - f → 
  (f * p) = 1500 := 
by
  intros h1 h2 h3
  sorry

end full_price_ticket_revenue_l58_58990


namespace initial_percentage_proof_l58_58439

-- Defining the initial percentage of water filled in the container
def initial_percentage (capacity add amount_filled : ℕ) : ℕ :=
  (amount_filled * 100) / capacity

-- The problem constraints
theorem initial_percentage_proof : initial_percentage 120 48 (3 * 120 / 4 - 48) = 35 := by
  -- We need to show that the initial percentage is 35%
  sorry

end initial_percentage_proof_l58_58439


namespace intersection_M_N_l58_58852

-- Definitions of sets M and N
def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Proof statement showing the intersection of M and N
theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l58_58852


namespace math_problem_l58_58316

open Set

noncomputable def A : Set ℝ := { x | x < 1 }
noncomputable def B : Set ℝ := { x | x * (x - 1) > 6 }
noncomputable def C (m : ℝ) : Set ℝ := { x | -1 + m < x ∧ x < 2 * m }

theorem math_problem (m : ℝ) (m_range : C m ≠ ∅) :
  (A ∪ B = { x | x > 3 ∨ x < 1 }) ∧
  (A ∩ (compl B) = { x | -2 ≤ x ∧ x < 1 }) ∧
  (-1 < m ∧ m ≤ 0.5) :=
by
  sorry

end math_problem_l58_58316


namespace central_angle_of_sector_l58_58940

theorem central_angle_of_sector (R r n : ℝ) (h_lateral_area : 2 * π * r^2 = π * r * R) 
  (h_arc_length : (n * π * R) / 180 = 2 * π * r) : n = 180 :=
by 
  sorry

end central_angle_of_sector_l58_58940


namespace intersection_A_B_l58_58151

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end intersection_A_B_l58_58151


namespace adam_earnings_l58_58471

theorem adam_earnings
  (earn_per_lawn : ℕ) (total_lawns : ℕ) (forgot_lawns : ℕ)
  (h1 : earn_per_lawn = 9) (h2 : total_lawns = 12) (h3 : forgot_lawns = 8) :
  (total_lawns - forgot_lawns) * earn_per_lawn = 36 :=
by
  sorry

end adam_earnings_l58_58471


namespace gcd_765432_654321_l58_58398

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l58_58398


namespace sector_central_angle_l58_58531

theorem sector_central_angle (r θ : ℝ) (h1 : 2 * r + r * θ = 6) (h2 : 0.5 * r * r * θ = 2) : θ = 1 ∨ θ = 4 :=
sorry

end sector_central_angle_l58_58531


namespace find_other_person_money_l58_58923

noncomputable def other_person_money (mias_money : ℕ) : ℕ :=
  let x := (mias_money - 20) / 2
  x

theorem find_other_person_money (mias_money : ℕ) (h_mias_money : mias_money = 110) : 
  other_person_money mias_money = 45 := by
  sorry

end find_other_person_money_l58_58923


namespace length_first_train_l58_58730

/-- Let the speeds of two trains be 120 km/hr and 80 km/hr, respectively. 
These trains cross each other in 9 seconds, and the length of the second train is 250.04 meters. 
Prove that the length of the first train is 250 meters. -/
theorem length_first_train
  (FirstTrainSpeed : ℝ := 120)  -- speed of the first train in km/hr
  (SecondTrainSpeed : ℝ := 80)  -- speed of the second train in km/hr
  (TimeToCross : ℝ := 9)        -- time to cross each other in seconds
  (LengthSecondTrain : ℝ := 250.04) -- length of the second train in meters
  : FirstTrainSpeed / 0.36 + SecondTrainSpeed / 0.36 * TimeToCross - LengthSecondTrain = 250 :=
by
  -- omitted proof
  sorry

end length_first_train_l58_58730


namespace area_of_quadrilateral_l58_58423

theorem area_of_quadrilateral (d h1 h2 : ℝ) (h1_pos : h1 = 9) (h2_pos : h2 = 6) (d_pos : d = 30) : 
  let area1 := (1/2 : ℝ) * d * h1
  let area2 := (1/2 : ℝ) * d * h2
  (area1 + area2) = 225 :=
by
  sorry

end area_of_quadrilateral_l58_58423


namespace original_pencil_count_l58_58487

-- Defining relevant constants and assumptions based on the problem conditions
def pencilsRemoved : ℕ := 4
def pencilsLeft : ℕ := 83

-- Theorem to prove the original number of pencils is 87
theorem original_pencil_count : pencilsLeft + pencilsRemoved = 87 := by
  sorry

end original_pencil_count_l58_58487


namespace side_length_sum_area_l58_58777

theorem side_length_sum_area (a b c d : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 12) :
  d = 13 :=
by
  -- Proof is not required
  sorry

end side_length_sum_area_l58_58777


namespace original_cube_volume_l58_58965

theorem original_cube_volume (V₂ : ℝ) (s : ℝ) (h₀ : V₂ = 216) (h₁ : (2 * s) ^ 3 = V₂) : s ^ 3 = 27 := by
  sorry

end original_cube_volume_l58_58965


namespace remainder_of_sum_div_11_is_9_l58_58504

def seven_times_ten_pow_twenty : ℕ := 7 * 10 ^ 20
def two_pow_twenty : ℕ := 2 ^ 20
def sum : ℕ := seven_times_ten_pow_twenty + two_pow_twenty

theorem remainder_of_sum_div_11_is_9 : sum % 11 = 9 := by
  sorry

end remainder_of_sum_div_11_is_9_l58_58504


namespace tan_identity_proof_l58_58509

noncomputable def tan_add_pi_over_3 (α β : ℝ) : ℝ :=
  Real.tan (α + Real.pi / 3)

theorem tan_identity_proof 
  (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3 / 5)
  (h2 : Real.tan (β - Real.pi / 3) = 1 / 4) :
  tan_add_pi_over_3 α β = 7 / 23 := 
sorry

end tan_identity_proof_l58_58509


namespace not_possible_to_get_105_single_stone_piles_l58_58157

noncomputable def piles : List Nat := [51, 49, 5]
def combine (a b : Nat) : Nat := a + b
def split (a : Nat) : List Nat := if a % 2 = 0 then [a / 2, a / 2] else [a]

theorem not_possible_to_get_105_single_stone_piles 
  (initial_piles : List Nat := piles) 
  (combine : Nat → Nat → Nat := combine) 
  (split : Nat → List Nat := split) :
  ¬ ∃ (final_piles : List Nat), final_piles.length = 105 ∧ (∀ n ∈ final_piles, n = 1) :=
by
  sorry

end not_possible_to_get_105_single_stone_piles_l58_58157


namespace investor_amount_after_two_years_l58_58767

noncomputable def compound_interest
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem investor_amount_after_two_years :
  compound_interest 3000 0.10 1 2 = 3630 :=
by
  -- Calculation goes here
  sorry

end investor_amount_after_two_years_l58_58767


namespace binom_1294_2_l58_58388

def combination (n k : Nat) := n.choose k

theorem binom_1294_2 : combination 1294 2 = 836161 := by
  sorry

end binom_1294_2_l58_58388


namespace triangle_area_is_180_l58_58479

theorem triangle_area_is_180 {a b c : ℕ} (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) 
  (h_right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2 : ℚ) * a * b = 180 :=
by
  sorry

end triangle_area_is_180_l58_58479


namespace collinearity_necessary_but_not_sufficient_l58_58112

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (u v : V) : Prop := ∃ (a : ℝ), v = a • u

def equal (u v : V) : Prop := u = v

theorem collinearity_necessary_but_not_sufficient (u v : V) :
  (collinear u v → equal u v) ∧ (equal u v → collinear u v) → collinear u v ∧ ¬(collinear u v ↔ equal u v) :=
sorry

end collinearity_necessary_but_not_sufficient_l58_58112
