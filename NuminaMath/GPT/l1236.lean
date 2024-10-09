import Mathlib

namespace paint_canvas_cost_ratio_l1236_123620

theorem paint_canvas_cost_ratio (C P : ℝ) (hc : 0.6 * C = C - 0.4 * C) (hp : 0.4 * P = P - 0.6 * P)
 (total_cost_reduction : 0.4 * P + 0.6 * C = 0.44 * (P + C)) :
  P / C = 4 :=
by
  sorry

end paint_canvas_cost_ratio_l1236_123620


namespace no_non_congruent_right_triangles_l1236_123675

theorem no_non_congruent_right_triangles (a b : ℝ) (c : ℝ) (h_right_triangle : c = Real.sqrt (a^2 + b^2)) (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = 2 * Real.sqrt (a^2 + b^2)) : a = 0 ∨ b = 0 :=
by
  sorry

end no_non_congruent_right_triangles_l1236_123675


namespace book_price_l1236_123600

theorem book_price (x : ℕ) (h1 : x - 1 = 1 + (x - 1)) : x = 2 :=
by
  sorry

end book_price_l1236_123600


namespace B_subsetneq_A_l1236_123670

def A : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }
def B : Set ℝ := { x : ℝ | 1 - x^2 > 0 }

theorem B_subsetneq_A : B ⊂ A :=
by
  sorry

end B_subsetneq_A_l1236_123670


namespace economist_winning_strategy_l1236_123615

-- Conditions setup
variables {n a b x1 x2 y1 y2 : ℕ}

-- Definitions according to the conditions
def valid_initial_division (n a b : ℕ) : Prop :=
  n > 4 ∧ n % 2 = 1 ∧ 2 ≤ a ∧ 2 ≤ b ∧ a + b = n ∧ a < b

def valid_further_division (a b x1 x2 y1 y2 : ℕ) : Prop :=
  x1 + x2 = a ∧ x1 ≥ 1 ∧ x2 ≥ 1 ∧ y1 + y2 = b ∧ y1 ≥ 1 ∧ y2 ≥ 1 ∧ x1 ≤ x2 ∧ y1 ≤ y2

-- Methods defined: Assumptions about which parts the economist takes
def method_1 (x1 x2 y1 y2 : ℕ) : ℕ :=
  max x2 y2 + min x1 y1

def method_2 (x1 x2 y1 y2 : ℕ) : ℕ :=
  (x1 + y1) / 2 + (x2 + y2) / 2

def method_3 (x1 x2 y1 y2 : ℕ) : ℕ :=
  max (method_1 x1 x2 y1 y2 - 1) (method_2 x1 x2 y1 y2 - 1) + 1

-- The statement to prove that the economist would choose method 1
theorem economist_winning_strategy :
  ∀ n a b x1 x2 y1 y2,
    valid_initial_division n a b →
    valid_further_division a b x1 x2 y1 y2 →
    n > 4 → n % 2 = 1 →
    (method_1 x1 x2 y1 y2) > (method_2 x1 x2 y1 y2) →
    (method_1 x1 x2 y1 y2) > (method_3 x1 x2 y1 y2) →
    method_1 x1 x2 y1 y2 = max (method_1 x1 x2 y1 y2) (method_2 x1 x2 y1 y2) :=
by
  -- Placeholder for the actual proof
  sorry

end economist_winning_strategy_l1236_123615


namespace last_score_entered_is_75_l1236_123647

theorem last_score_entered_is_75 (scores : List ℕ) (h : scores = [62, 75, 83, 90]) :
  ∃ last_score, last_score ∈ scores ∧ 
    (∀ (num list : List ℕ), list ≠ [] → list.length ≤ scores.length → 
    ¬ list.sum % list.length ≠ 0) → 
  last_score = 75 :=
by
  sorry

end last_score_entered_is_75_l1236_123647


namespace rectangular_plot_area_l1236_123613

theorem rectangular_plot_area (Breadth Length Area : ℕ): 
  (Length = 3 * Breadth) → 
  (Breadth = 30) → 
  (Area = Length * Breadth) → 
  Area = 2700 :=
by 
  intros h_length h_breadth h_area
  rw [h_breadth] at h_length
  rw [h_length, h_breadth] at h_area
  exact h_area

end rectangular_plot_area_l1236_123613


namespace meiosis_and_fertilization_outcome_l1236_123692

-- Definitions corresponding to the conditions:
def increases_probability_of_genetic_mutations (x : Type) := 
  ∃ (p : x), false -- Placeholder for the actual mutation rate being low

def inherits_all_genetic_material (x : Type) :=
  ∀ (p : x), false -- Parents do not pass all genes to offspring

def receives_exactly_same_genetic_information (x : Type) :=
  ∀ (p : x), false -- Offspring do not receive exact genetic information from either parent

def produces_genetic_combination_different (x : Type) :=
  ∃ (o : x), true -- The offspring has different genetic information from either parent

-- The main statement to be proven:
theorem meiosis_and_fertilization_outcome (x : Type) 
  (cond1 : ¬ increases_probability_of_genetic_mutations x)
  (cond2 : ¬ inherits_all_genetic_material x)
  (cond3 : ¬ receives_exactly_same_genetic_information x) :
  produces_genetic_combination_different x :=
sorry

end meiosis_and_fertilization_outcome_l1236_123692


namespace sin_double_angle_l1236_123656

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ = 2) : Real.sin (2 * θ) = 4 / 5 :=
by
  sorry

end sin_double_angle_l1236_123656


namespace min_cost_of_packaging_l1236_123623

def packaging_problem : Prop :=
  ∃ (x y : ℕ), 35 * x + 24 * y = 106 ∧ 140 * x + 120 * y = 500

theorem min_cost_of_packaging : packaging_problem :=
sorry

end min_cost_of_packaging_l1236_123623


namespace negation_proposition_l1236_123668

theorem negation_proposition : (¬ ∀ x : ℝ, (1 < x) → x - 1 ≥ Real.log x) ↔ (∃ x_0 : ℝ, (1 < x_0) ∧ x_0 - 1 < Real.log x_0) :=
by
  sorry

end negation_proposition_l1236_123668


namespace ratio_of_age_difference_l1236_123616

theorem ratio_of_age_difference (R J K : ℕ) 
  (h1 : R = J + 6) 
  (h2 : R + 4 = 2 * (J + 4)) 
  (h3 : (R + 4) * (K + 4) = 108) : 
  (R - J) / (R - K) = 2 :=
by 
  sorry

end ratio_of_age_difference_l1236_123616


namespace regular_milk_cartons_l1236_123640

variable (R C : ℕ)
variable (h1 : C + R = 24)
variable (h2 : C = 7 * R)

theorem regular_milk_cartons : R = 3 :=
by
  sorry

end regular_milk_cartons_l1236_123640


namespace ratio_of_perimeters_of_similar_triangles_l1236_123658

theorem ratio_of_perimeters_of_similar_triangles (A1 A2 P1 P2 : ℝ) (h : A1 / A2 = 16 / 9) : P1 / P2 = 4 / 3 :=
sorry

end ratio_of_perimeters_of_similar_triangles_l1236_123658


namespace remainder_when_12_plus_a_div_by_31_l1236_123633

open Int

theorem remainder_when_12_plus_a_div_by_31 (a : ℤ) (ha : 0 < a) (h : 17 * a % 31 = 1) : (12 + a) % 31 = 23 := by
  sorry

end remainder_when_12_plus_a_div_by_31_l1236_123633


namespace gift_card_amount_l1236_123654

theorem gift_card_amount (original_price final_price : ℝ) 
  (discount1 discount2 : ℝ) 
  (discounted_price1 discounted_price2 : ℝ) :
  original_price = 2000 →
  discount1 = 0.15 →
  discount2 = 0.10 →
  discounted_price1 = original_price - (discount1 * original_price) →
  discounted_price2 = discounted_price1 - (discount2 * discounted_price1) →
  final_price = 1330 →
  discounted_price2 - final_price = 200 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end gift_card_amount_l1236_123654


namespace students_suggested_bacon_l1236_123649

-- Defining the conditions
def total_students := 310
def mashed_potatoes_students := 185

-- Lean statement for proving the equivalent problem
theorem students_suggested_bacon : total_students - mashed_potatoes_students = 125 := by
  sorry -- Proof is omitted

end students_suggested_bacon_l1236_123649


namespace problem_statement_l1236_123697

def f (x : ℕ) : ℕ := x^2 + x + 4
def g (x : ℕ) : ℕ := 3 * x^3 + 2

theorem problem_statement : g (f 3) = 12290 := by
  sorry

end problem_statement_l1236_123697


namespace find_s_of_2_l1236_123619

def t (x : ℝ) : ℝ := 4 * x - 9
def s (x : ℝ) : ℝ := x^2 + 4 * x - 1

theorem find_s_of_2 : s (2) = 281 / 16 :=
by
  sorry

end find_s_of_2_l1236_123619


namespace max_point_h_l1236_123677

-- Definitions of the linear functions f and g
def f (x : ℝ) : ℝ := 2 * x + 2
def g (x : ℝ) : ℝ := -x - 3

-- The product of f(x) and g(x)
def h (x : ℝ) : ℝ := f x * g x

-- Statement: Prove that x = -2 is the maximum point of h(x)
theorem max_point_h : ∃ x_max : ℝ, h x_max = (-2) :=
by
  -- skipping the proof
  sorry

end max_point_h_l1236_123677


namespace percent_in_second_part_l1236_123696

-- Defining the conditions and the proof statement
theorem percent_in_second_part (x y P : ℝ) 
  (h1 : 0.25 * (x - y) = (P / 100) * (x + y))
  (h2 : y = 0.25 * x) : 
  P = 15 :=
by
  sorry

end percent_in_second_part_l1236_123696


namespace sum_of_percentages_l1236_123607

theorem sum_of_percentages : 
  let x := 80 + (0.2 * 80)
  let y := 60 - (0.3 * 60)
  let z := 40 + (0.5 * 40)
  x + y + z = 198 := by
  sorry

end sum_of_percentages_l1236_123607


namespace fraction_doubling_unchanged_l1236_123694

theorem fraction_doubling_unchanged (x y : ℝ) (h : x ≠ y) : 
  (3 * (2 * x)) / (2 * x - 2 * y) = (3 * x) / (x - y) :=
by
  sorry

end fraction_doubling_unchanged_l1236_123694


namespace Aiyanna_has_more_cookies_l1236_123664

theorem Aiyanna_has_more_cookies (cookies_Alyssa : ℕ) (cookies_Aiyanna : ℕ) (h1 : cookies_Alyssa = 129) (h2 : cookies_Aiyanna = cookies_Alyssa + 11) : cookies_Aiyanna = 140 := by
  sorry

end Aiyanna_has_more_cookies_l1236_123664


namespace distinct_solutions_diff_l1236_123682

theorem distinct_solutions_diff {r s : ℝ} (h_eq_r : (6 * r - 18) / (r^2 + 3 * r - 18) = r + 3)
  (h_eq_s : (6 * s - 18) / (s^2 + 3 * s - 18) = s + 3)
  (h_distinct : r ≠ s) (h_r_gt_s : r > s) : r - s = 11 := 
sorry

end distinct_solutions_diff_l1236_123682


namespace toms_score_l1236_123672

theorem toms_score (T J : ℝ) (h1 : T = J + 30) (h2 : (T + J) / 2 = 90) : T = 105 := by
  sorry

end toms_score_l1236_123672


namespace remainder_when_divided_by_l1236_123687

def P (x : ℤ) : ℤ := 5 * x^8 - 2 * x^7 - 8 * x^6 + 3 * x^4 + 5 * x^3 - 13
def D (x : ℤ) : ℤ := 3 * (x - 3)

theorem remainder_when_divided_by (x : ℤ) : P 3 = 23364 :=
by {
  -- This is where the calculation steps would go, but we're omitting them.
  sorry
}

end remainder_when_divided_by_l1236_123687


namespace actual_diameter_of_tissue_l1236_123605

variable (magnified_diameter : ℝ) (magnification_factor : ℝ)

theorem actual_diameter_of_tissue 
    (h1 : magnified_diameter = 0.2) 
    (h2 : magnification_factor = 1000) : 
    magnified_diameter / magnification_factor = 0.0002 := 
  by
    sorry

end actual_diameter_of_tissue_l1236_123605


namespace checkerboard_probability_l1236_123627

-- Define the number of squares in the checkerboard and the number on the perimeter
def total_squares : Nat := 10 * 10
def perimeter_squares : Nat := 10 + 10 + (10 - 2) + (10 - 2)

-- The number of squares not on the perimeter
def inner_squares : Nat := total_squares - perimeter_squares

-- The probability that a randomly chosen square does not touch the outer edge
def probability_not_on_perimeter : ℚ := inner_squares / total_squares

theorem checkerboard_probability :
  probability_not_on_perimeter = 16 / 25 :=
by
  -- proof goes here
  sorry

end checkerboard_probability_l1236_123627


namespace total_revenue_is_correct_l1236_123684

def category_a_price : ℝ := 65
def category_b_price : ℝ := 45
def category_c_price : ℝ := 25

def category_a_discounted_price : ℝ := category_a_price - 0.55 * category_a_price
def category_b_discounted_price : ℝ := category_b_price - 0.35 * category_b_price
def category_c_discounted_price : ℝ := category_c_price - 0.20 * category_c_price

def category_a_full_price_quantity : ℕ := 100
def category_b_full_price_quantity : ℕ := 50
def category_c_full_price_quantity : ℕ := 60

def category_a_discounted_quantity : ℕ := 20
def category_b_discounted_quantity : ℕ := 30
def category_c_discounted_quantity : ℕ := 40

def revenue_from_category_a : ℝ :=
  category_a_discounted_quantity * category_a_discounted_price +
  category_a_full_price_quantity * category_a_price

def revenue_from_category_b : ℝ :=
  category_b_discounted_quantity * category_b_discounted_price +
  category_b_full_price_quantity * category_b_price

def revenue_from_category_c : ℝ :=
  category_c_discounted_quantity * category_c_discounted_price +
  category_c_full_price_quantity * category_c_price

def total_revenue : ℝ :=
  revenue_from_category_a + revenue_from_category_b + revenue_from_category_c

theorem total_revenue_is_correct :
  total_revenue = 12512.50 :=
by
  unfold total_revenue
  unfold revenue_from_category_a
  unfold revenue_from_category_b
  unfold revenue_from_category_c
  unfold category_a_discounted_price
  unfold category_b_discounted_price
  unfold category_c_discounted_price
  sorry

end total_revenue_is_correct_l1236_123684


namespace honor_students_count_l1236_123606

noncomputable def number_of_students_in_class_is_less_than_30 := ∃ n, n < 30
def probability_girl_honor_student (G E_G : ℕ) := E_G / G = (3 : ℚ) / 13
def probability_boy_honor_student (B E_B : ℕ) := E_B / B = (4 : ℚ) / 11

theorem honor_students_count (G B E_G E_B : ℕ) 
  (hG_cond : probability_girl_honor_student G E_G) 
  (hB_cond : probability_boy_honor_student B E_B) 
  (h_total_students : G + B < 30) 
  (hE_G_def : E_G = 3 * G / 13) 
  (hE_B_def : E_B = 4 * B / 11) 
  (hG_nonneg : G >= 13)
  (hB_nonneg : B >= 11):
  E_G + E_B = 7 := 
sorry

end honor_students_count_l1236_123606


namespace num_ways_to_pay_l1236_123676

theorem num_ways_to_pay (n : ℕ) : 
  ∃ a_n : ℕ, a_n = (n / 2) + 1 :=
sorry

end num_ways_to_pay_l1236_123676


namespace money_inequalities_l1236_123680

theorem money_inequalities (a b : ℝ) (h₁ : 5 * a + b > 51) (h₂ : 3 * a - b = 21) : a > 9 ∧ b > 6 := 
by
  sorry

end money_inequalities_l1236_123680


namespace solve_inequality_l1236_123609

theorem solve_inequality (x : ℝ) : 
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
by
  sorry

end solve_inequality_l1236_123609


namespace shortest_altitude_l1236_123608

theorem shortest_altitude (a b c : ℝ) (h : a = 9 ∧ b = 12 ∧ c = 15) (h_right : a^2 + b^2 = c^2) : 
  ∃ x : ℝ, x = 7.2 ∧ (1/2) * c * x = (1/2) * a * b := 
by
  sorry

end shortest_altitude_l1236_123608


namespace find_three_numbers_l1236_123662

-- Define the conditions
def condition1 (X : ℝ) : Prop := X = 0.35 * X + 60
def condition2 (X Y : ℝ) : Prop := X = 0.7 * (1 / 2) * Y + (1 / 2) * Y
def condition3 (Y Z : ℝ) : Prop := Y = 2 * Z ^ 2

-- Define the final result that we need to prove
def final_result (X Y Z : ℝ) : Prop := X = 92 ∧ Y = 108 ∧ Z = 7

-- The main theorem statement
theorem find_three_numbers :
  ∃ (X Y Z : ℝ), condition1 X ∧ condition2 X Y ∧ condition3 Y Z ∧ final_result X Y Z :=
by
  sorry

end find_three_numbers_l1236_123662


namespace find_x_parallel_l1236_123603

def m : ℝ × ℝ := (-2, 4)
def n (x : ℝ) : ℝ × ℝ := (x, -1)

def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ u.1 = k * v.1 ∧ u.2 = k * v.2

theorem find_x_parallel :
  parallel m (n x) → x = 1 / 2 := by 
sorry

end find_x_parallel_l1236_123603


namespace always_exists_triangle_l1236_123659

variable (a1 a2 a3 a4 d : ℕ)

def arithmetic_sequence (a1 a2 a3 a4 d : ℕ) :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d

def positive_terms (a1 a2 a3 a4 : ℕ) :=
  a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0

theorem always_exists_triangle (a1 a2 a3 a4 d : ℕ)
  (h1 : arithmetic_sequence a1 a2 a3 a4 d)
  (h2 : d > 0)
  (h3 : positive_terms a1 a2 a3 a4) :
  a2 + a3 > a4 ∧ a2 + a4 > a3 ∧ a3 + a4 > a2 :=
sorry

end always_exists_triangle_l1236_123659


namespace average_weight_of_children_l1236_123685

theorem average_weight_of_children (avg_weight_boys avg_weight_girls : ℕ)
                                   (num_boys num_girls : ℕ)
                                   (h1 : avg_weight_boys = 160)
                                   (h2 : avg_weight_girls = 110)
                                   (h3 : num_boys = 8)
                                   (h4 : num_girls = 5) :
                                   (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 141 :=
by
    sorry

end average_weight_of_children_l1236_123685


namespace cubic_identity_l1236_123643

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cubic_identity_l1236_123643


namespace disjoint_subsets_mod_1000_l1236_123644

open Nat

theorem disjoint_subsets_mod_1000 :
  let T := Finset.range 13
  let m := (3^12 - 2 * 2^12 + 1) / 2
  m % 1000 = 625 := 
by
  let T := Finset.range 13
  let m := (3^12 - 2 * 2^12 + 1) / 2
  have : m % 1000 = 625 := sorry
  exact this

end disjoint_subsets_mod_1000_l1236_123644


namespace Darnel_sprinted_further_l1236_123679

-- Define the distances sprinted and jogged
def sprinted : ℝ := 0.88
def jogged : ℝ := 0.75

-- State the theorem to prove the main question
theorem Darnel_sprinted_further : sprinted - jogged = 0.13 :=
by
  sorry

end Darnel_sprinted_further_l1236_123679


namespace max_value_ineq_l1236_123666

theorem max_value_ineq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y)^2 / (x^2 + y^2 + xy) ≤ 4 / 3 :=
sorry

end max_value_ineq_l1236_123666


namespace find_x_value_l1236_123667

theorem find_x_value (x : ℝ) (hx : x ≠ 0) : 
    (1/x) + (3/x) / (6/x) = 1 → x = 2 := 
by 
    intro h
    sorry

end find_x_value_l1236_123667


namespace arithmetic_sequence_of_condition_l1236_123634

variables {R : Type*} [LinearOrderedRing R]

theorem arithmetic_sequence_of_condition (x y z : R) (h : (z-x)^2 - 4*(x-y)*(y-z) = 0) : 2*y = x + z :=
sorry

end arithmetic_sequence_of_condition_l1236_123634


namespace zachary_additional_money_needed_l1236_123639

noncomputable def total_cost : ℝ := 3.756 + 2 * 2.498 + 11.856 + 4 * 1.329 + 7.834
noncomputable def zachary_money : ℝ := 24.042
noncomputable def money_needed : ℝ := total_cost - zachary_money

theorem zachary_additional_money_needed : money_needed = 9.716 := 
by 
  sorry

end zachary_additional_money_needed_l1236_123639


namespace ball_bounce_height_l1236_123642

theorem ball_bounce_height (n : ℕ) : (512 * (1/2)^n < 20) → n = 8 := 
sorry

end ball_bounce_height_l1236_123642


namespace tan_315_eq_neg1_l1236_123617

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l1236_123617


namespace range_of_a_l1236_123628

variable {f : ℝ → ℝ}
variable {a : ℝ}

-- Define the conditions given:
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def monotonic_increasing_on_nonnegative_reals (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, (0 ≤ x1) → (0 ≤ x2) → (x1 < x2) → (f x1 < f x2)

def inequality_in_interval (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x, (1 / 2 ≤ x) → (x ≤ 1) → (f (a * x + 1) ≤ f (x - 2))

-- The theorem we want to prove
theorem range_of_a (h1 : even_function f)
                   (h2 : monotonic_increasing_on_nonnegative_reals f)
                   (h3 : inequality_in_interval f a) :
  -2 ≤ a ∧ a ≤ 0 := sorry

end range_of_a_l1236_123628


namespace fraction_to_decimal_l1236_123660

theorem fraction_to_decimal : (7 / 12 : ℝ) = 0.5833 + (3 / 10000) * (1 / (1 - (1 / 10))) := 
by sorry

end fraction_to_decimal_l1236_123660


namespace blue_pill_cost_l1236_123674

variable (cost_blue_pill : ℕ) (cost_red_pill : ℕ) (daily_cost : ℕ) 
variable (num_days : ℕ) (total_cost : ℕ)
variable (cost_diff : ℕ)

theorem blue_pill_cost :
  num_days = 21 ∧
  total_cost = 966 ∧
  cost_diff = 4 ∧
  daily_cost = total_cost / num_days ∧
  daily_cost = cost_blue_pill + cost_red_pill ∧
  cost_blue_pill = cost_red_pill + cost_diff ∧
  daily_cost = 46 →
  cost_blue_pill = 25 := by
  sorry

end blue_pill_cost_l1236_123674


namespace total_amount_paid_correct_l1236_123657

-- Define variables for prices of the pizzas
def first_pizza_price : ℝ := 8
def second_pizza_price : ℝ := 12
def third_pizza_price : ℝ := 10

-- Define variables for discount rate and tax rate
def discount_rate : ℝ := 0.20
def sales_tax_rate : ℝ := 0.05

-- Define the total amount paid by Mrs. Hilt
def total_amount_paid : ℝ :=
  let total_cost := first_pizza_price + second_pizza_price + third_pizza_price
  let discount := total_cost * discount_rate
  let discounted_total := total_cost - discount
  let sales_tax := discounted_total * sales_tax_rate
  discounted_total + sales_tax

-- Prove that the total amount paid is $25.20
theorem total_amount_paid_correct : total_amount_paid = 25.20 := 
  by
  sorry

end total_amount_paid_correct_l1236_123657


namespace calculation_is_zero_l1236_123648

theorem calculation_is_zero : 
  20062006 * 2007 + 20072007 * 2008 - 2006 * 20072007 - 2007 * 20082008 = 0 := 
by 
  sorry

end calculation_is_zero_l1236_123648


namespace product_ABC_sol_l1236_123690

theorem product_ABC_sol (A B C : ℚ) : 
  (∀ x : ℚ, x^2 - 20 = A * (x + 2) * (x - 3) + B * (x - 2) * (x - 3) + C * (x - 2) * (x + 2)) → 
  A * B * C = 2816 / 35 := 
by 
  intro h
  sorry

end product_ABC_sol_l1236_123690


namespace operation_B_correct_operation_C_correct_l1236_123629

theorem operation_B_correct (x y : ℝ) : (-3 * x * y) ^ 2 = 9 * x ^ 2 * y ^ 2 :=
  sorry

theorem operation_C_correct (x y : ℝ) (h : x ≠ y) : 
  (x - y) / (2 * x * y - x ^ 2 - y ^ 2) = 1 / (y - x) :=
  sorry

end operation_B_correct_operation_C_correct_l1236_123629


namespace equation_of_circle_l1236_123632

-- Defining the problem conditions directly
variables (a : ℝ) (x y: ℝ)

-- Assume a ≠ 0
variable (h : a ≠ 0)

-- Prove that the circle passing through the origin with center (a, a) has the equation (x - a)^2 + (y - a)^2 = 2a^2.
theorem equation_of_circle (h : a ≠ 0) :
  (x - a)^2 + (y - a)^2 = 2 * a^2 :=
sorry

end equation_of_circle_l1236_123632


namespace cubic_polynomial_a_value_l1236_123604

theorem cubic_polynomial_a_value (a b c d y₁ y₂ : ℝ)
  (h₁ : y₁ = a + b + c + d)
  (h₂ : y₂ = -a + b - c + d)
  (h₃ : y₁ - y₂ = -8) : a = -4 :=
by
  sorry

end cubic_polynomial_a_value_l1236_123604


namespace first_term_of_new_ratio_l1236_123645

-- Given conditions as definitions
def original_ratio : ℚ := 6 / 7
def x (n : ℕ) : Prop := n ≥ 3

-- Prove that the first term of the ratio that the new ratio should be less than is 4
theorem first_term_of_new_ratio (n : ℕ) (h1 : x n) : ∃ b, (6 - n) / (7 - n) < 4 / b :=
by
  exists 5
  sorry

end first_term_of_new_ratio_l1236_123645


namespace campers_afternoon_l1236_123653

noncomputable def campers_morning : ℕ := 35
noncomputable def campers_total : ℕ := 62

theorem campers_afternoon :
  campers_total - campers_morning = 27 :=
by
  sorry

end campers_afternoon_l1236_123653


namespace apartments_in_each_complex_l1236_123663

variable {A : ℕ}

theorem apartments_in_each_complex
    (h1 : ∀ (locks_per_apartment : ℕ), locks_per_apartment = 3)
    (h2 : ∀ (num_complexes : ℕ), num_complexes = 2)
    (h3 : 3 * 2 * A = 72) :
    A = 12 :=
by
  sorry

end apartments_in_each_complex_l1236_123663


namespace find_number_l1236_123698

theorem find_number (a p x : ℕ) (h1 : p = 36) (h2 : 6 * a = 6 * (2 * p + x)) : x = 9 :=
by
  sorry

end find_number_l1236_123698


namespace range_of_x_in_second_quadrant_l1236_123637

theorem range_of_x_in_second_quadrant (x : ℝ) (h1 : x - 2 < 0) (h2 : x > 0) : 0 < x ∧ x < 2 :=
sorry

end range_of_x_in_second_quadrant_l1236_123637


namespace profit_increase_l1236_123681

theorem profit_increase (x y : ℝ) (a : ℝ)
  (h1 : x = (57 / 20) * y)
  (h2 : (x - y) / y = a / 100)
  (h3 : (x - 0.95 * y) / (0.95 * y) = (a + 15) / 100) :
  a = 185 := sorry

end profit_increase_l1236_123681


namespace year_with_greatest_temp_increase_l1236_123688

def avg_temp (year : ℕ) : ℝ :=
  match year with
  | 2000 => 2.0
  | 2001 => 2.3
  | 2002 => 2.5
  | 2003 => 2.7
  | 2004 => 3.9
  | 2005 => 4.1
  | 2006 => 4.2
  | 2007 => 4.4
  | 2008 => 3.9
  | 2009 => 3.1
  | _    => 0.0

theorem year_with_greatest_temp_increase : ∃ year, year = 2004 ∧
  (∀ y, 2000 < y ∧ y ≤ 2009 → avg_temp y - avg_temp (y - 1) ≤ avg_temp 2004 - avg_temp 2003) := by
  sorry

end year_with_greatest_temp_increase_l1236_123688


namespace number_of_female_students_l1236_123686

noncomputable def total_students : ℕ := 1600
noncomputable def sample_size : ℕ := 200
noncomputable def sampled_males : ℕ := 110
noncomputable def sampled_females := sample_size - sampled_males
noncomputable def total_males := (sampled_males * total_students) / sample_size
noncomputable def total_females := total_students - total_males

theorem number_of_female_students : total_females = 720 := 
sorry

end number_of_female_students_l1236_123686


namespace squares_difference_sum_l1236_123678

theorem squares_difference_sum : 
  19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 200 :=
by
  sorry

end squares_difference_sum_l1236_123678


namespace max_value_expression_l1236_123612

theorem max_value_expression (x y : ℝ) (h : x * y > 0) : 
  ∃ (max_val : ℝ), max_val = 4 - 2 * Real.sqrt 2 ∧ 
  (∀ a b : ℝ, a * b > 0 → (a / (a + b) + 2 * b / (a + 2 * b)) ≤ max_val) := 
sorry

end max_value_expression_l1236_123612


namespace sum_of_square_and_divisor_not_square_l1236_123624

theorem sum_of_square_and_divisor_not_square {A B : ℕ} (hA : A ≠ 0) (hA_square : ∃ k : ℕ, A = k * k) (hB_divisor : B ∣ A) : ¬ (∃ m : ℕ, A + B = m * m) := by
  -- Proof is omitted
  sorry

end sum_of_square_and_divisor_not_square_l1236_123624


namespace polly_age_is_33_l1236_123665

theorem polly_age_is_33 
  (x : ℕ) 
  (h1 : ∀ y, y = 20 → x - y = x - 20)
  (h2 : ∀ y, y = 22 → x - y = x - 22)
  (h3 : ∀ y, y = 24 → x - y = x - 24) : 
  x = 33 :=
by 
  sorry

end polly_age_is_33_l1236_123665


namespace counterexample_proof_l1236_123655

theorem counterexample_proof :
  ∃ a : ℝ, |a - 1| > 1 ∧ ¬ (a > 2) :=
  sorry

end counterexample_proof_l1236_123655


namespace equal_contribution_expense_split_l1236_123621

theorem equal_contribution_expense_split (Mitch_expense Jam_expense Jay_expense Jordan_expense total_expense each_contribution : ℕ)
  (hmitch : Mitch_expense = 4 * 7)
  (hjam : Jam_expense = (2 * 15) / 10 + 4) -- note: 1.5 dollar per box interpreted as 15/10 to avoid float in Lean
  (hjay : Jay_expense = 3 * 3)
  (hjordan : Jordan_expense = 4 * 2)
  (htotal : total_expense = Mitch_expense + Jam_expense + Jay_expense + Jordan_expense)
  (hequal_split : each_contribution = total_expense / 4) :
  each_contribution = 13 :=
by
  sorry

end equal_contribution_expense_split_l1236_123621


namespace area_of_triangle_LMN_l1236_123630

-- Define the vertices
def point := ℝ × ℝ
def L: point := (2, 3)
def M: point := (5, 1)
def N: point := (3, 5)

-- Shoelace formula for the area of a triangle
noncomputable def triangle_area (A B C : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2))

-- Statement to prove the area
theorem area_of_triangle_LMN : triangle_area L M N = 4 := by
  -- Proof would go here
  sorry

end area_of_triangle_LMN_l1236_123630


namespace mean_of_samantha_scores_l1236_123622

noncomputable def arithmetic_mean (l : List ℝ) : ℝ := l.sum / l.length

theorem mean_of_samantha_scores :
  arithmetic_mean [93, 87, 90, 96, 88, 94] = 91.333 :=
by
  sorry

end mean_of_samantha_scores_l1236_123622


namespace interest_rate_difference_l1236_123636

theorem interest_rate_difference:
  ∀ (R H: ℝ),
    (300 * (H / 100) * 5 = 300 * (R / 100) * 5 + 90) →
    (H - R = 6) :=
by
  intros R H h
  sorry

end interest_rate_difference_l1236_123636


namespace t_mobile_additional_line_cost_l1236_123651

variable (T : ℕ)

def t_mobile_cost (n : ℕ) : ℕ :=
  if n ≤ 2 then 50 else 50 + (n - 2) * T

def m_mobile_cost (n : ℕ) : ℕ :=
  if n ≤ 2 then 45 else 45 + (n - 2) * 14

theorem t_mobile_additional_line_cost
  (h : t_mobile_cost 5 = m_mobile_cost 5 + 11) :
  T = 16 :=
by
  sorry

end t_mobile_additional_line_cost_l1236_123651


namespace integer_solutions_of_polynomial_l1236_123610

theorem integer_solutions_of_polynomial :
  ∀ n : ℤ, n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0 → n = -1 ∨ n = 3 := 
by 
  sorry

end integer_solutions_of_polynomial_l1236_123610


namespace parabola_focus_directrix_distance_l1236_123650

theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y = (1 / 4) * x^2 → 
  (∃ p : ℝ, p = 2 ∧ x^2 = 4 * p * y) →
  ∃ d : ℝ, d = 2 :=
by
  sorry

end parabola_focus_directrix_distance_l1236_123650


namespace coins_distribution_l1236_123673

theorem coins_distribution :
  ∃ (x y z : ℕ), x + y + z = 1000 ∧ x + 2 * y + 5 * z = 2000 ∧ Nat.Prime x ∧ x = 3 ∧ y = 996 ∧ z = 1 :=
by
  sorry

end coins_distribution_l1236_123673


namespace solve_tangents_equation_l1236_123641

open Real

def is_deg (x : ℝ) : Prop := ∃ k : ℤ, x = 30 + 180 * k

theorem solve_tangents_equation (x : ℝ) (h : tan (x * π / 180) * tan (20 * π / 180) + tan (20 * π / 180) * tan (40 * π / 180) + tan (40 * π / 180) * tan (x * π / 180) = 1) :
  is_deg x :=
sorry

end solve_tangents_equation_l1236_123641


namespace Megan_total_earnings_two_months_l1236_123602

-- Define the conditions
def hours_per_day : ℕ := 8
def wage_per_hour : ℝ := 7.50
def days_per_month : ℕ := 20

-- Define the main question and correct answer
theorem Megan_total_earnings_two_months : 
  (2 * (days_per_month * (hours_per_day * wage_per_hour))) = 2400 := 
by
  -- In the problem statement, we are given conditions so we just state sorry because the focus is on the statement, not the solution steps.
  sorry

end Megan_total_earnings_two_months_l1236_123602


namespace cone_height_l1236_123625

theorem cone_height (r : ℝ) (n : ℕ) (circumference : ℝ) 
  (sector_circumference : ℝ) (base_radius : ℝ) (slant_height : ℝ) 
  (h : ℝ) : 
  r = 8 →
  n = 4 →
  circumference = 2 * Real.pi * r →
  sector_circumference = circumference / n →
  base_radius = sector_circumference / (2 * Real.pi) →
  slant_height = r →
  h = Real.sqrt (slant_height^2 - base_radius^2) →
  h = 2 * Real.sqrt 15 := 
by
  intros
  sorry

end cone_height_l1236_123625


namespace find_y_coordinate_of_P_l1236_123661

theorem find_y_coordinate_of_P (P Q : ℝ × ℝ)
  (h1 : ∀ x, y = 0.8 * x) -- line equation
  (h2 : P.1 = 4) -- x-coordinate of P
  (h3 : P = Q) -- P and Q are equidistant from the line
  : P.2 = 3.2 := sorry

end find_y_coordinate_of_P_l1236_123661


namespace measure_15_minutes_l1236_123691

/-- Given a timer setup with a 7-minute hourglass and an 11-minute hourglass, show that we can measure exactly 15 minutes. -/
theorem measure_15_minutes (h7 : ∃ t : ℕ, t = 7) (h11 : ∃ t : ℕ, t = 11) : ∃ t : ℕ, t = 15 := 
  by 
    sorry

end measure_15_minutes_l1236_123691


namespace allyn_total_expense_in_june_l1236_123689

/-- We have a house with 40 bulbs, each using 60 watts of power daily.
Allyn pays 0.20 dollars per watt used. June has 30 days.
We need to calculate Allyn's total monthly expense on electricity in June,
which should be \$14400. -/
theorem allyn_total_expense_in_june
    (daily_watt_per_bulb : ℕ := 60)
    (num_bulbs : ℕ := 40)
    (cost_per_watt : ℝ := 0.20)
    (days_in_june : ℕ := 30)
    : num_bulbs * daily_watt_per_bulb * days_in_june * cost_per_watt = 14400 := 
by
  sorry

end allyn_total_expense_in_june_l1236_123689


namespace movie_production_cost_l1236_123626

-- Definitions based on the conditions
def opening_revenue : ℝ := 120 -- in million dollars
def total_revenue : ℝ := 3.5 * opening_revenue -- movie made during its entire run
def kept_revenue : ℝ := 0.60 * total_revenue -- production company keeps 60% of total revenue
def profit : ℝ := 192 -- in million dollars

-- Theorem stating the cost to produce the movie
theorem movie_production_cost : 
  (kept_revenue - 60) = profit :=
by
  sorry

end movie_production_cost_l1236_123626


namespace line_translation_upwards_units_l1236_123683

theorem line_translation_upwards_units:
  ∀ (x : ℝ), (y = x / 3) → (y = (x + 5) / 3) → (y' = y + 5 / 3) :=
by
  sorry

end line_translation_upwards_units_l1236_123683


namespace binomial_coefficient_7_5_permutation_7_5_l1236_123695

-- Define function for binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define function for permutation calculation
def permutation (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

theorem binomial_coefficient_7_5 : binomial_coefficient 7 5 = 21 :=
by
  sorry

theorem permutation_7_5 : permutation 7 5 = 2520 :=
by
  sorry

end binomial_coefficient_7_5_permutation_7_5_l1236_123695


namespace unique_root_of_increasing_l1236_123652

variable {R : Type} [LinearOrderedField R] [DecidableEq R]

def increasing (f : R → R) : Prop :=
  ∀ x1 x2 : R, x1 < x2 → f x1 < f x2

theorem unique_root_of_increasing (f : R → R)
  (h_inc : increasing f) :
  ∃! x : R, f x = 0 :=
sorry

end unique_root_of_increasing_l1236_123652


namespace garden_fencing_cost_l1236_123693

theorem garden_fencing_cost (x y : ℝ) (h1 : x^2 + y^2 = 900) (h2 : x * y = 200)
    (cost_per_meter : ℝ) (h3 : cost_per_meter = 15) : 
    cost_per_meter * (2 * x + y) = 300 * Real.sqrt 7 + 150 * Real.sqrt 2 :=
by
  sorry

end garden_fencing_cost_l1236_123693


namespace final_configuration_l1236_123635

def initial_configuration : (String × String) :=
  ("bottom-right", "bottom-left")

def first_transformation (conf : (String × String)) : (String × String) :=
  match conf with
  | ("bottom-right", "bottom-left") => ("top-right", "top-left")
  | _ => conf

def second_transformation (conf : (String × String)) : (String × String) :=
  match conf with
  | ("top-right", "top-left") => ("top-left", "top-right")
  | _ => conf

theorem final_configuration :
  second_transformation (first_transformation initial_configuration) =
  ("top-left", "top-right") :=
by
  sorry

end final_configuration_l1236_123635


namespace value_of_a_l1236_123669

noncomputable def f : ℝ → ℝ 
| x => if x > 0 then 2^x else x + 1

theorem value_of_a (a : ℝ) (h : f a + f 1 = 0) : a = -3 :=
by
  sorry

end value_of_a_l1236_123669


namespace lana_trip_longer_by_25_percent_l1236_123671

-- Define the dimensions of the rectangular field
def length_field : ℕ := 3
def width_field : ℕ := 1

-- Define Tom's path distance
def tom_path_distance : ℕ := length_field + width_field

-- Define Lana's path distance
def lana_path_distance : ℕ := 2 + 1 + 1 + 1

-- Define the percentage increase calculation
def percentage_increase (initial final : ℕ) : ℕ :=
  (final - initial) * 100 / initial

-- Define the theorem to be proven
theorem lana_trip_longer_by_25_percent :
  percentage_increase tom_path_distance lana_path_distance = 25 :=
by
  sorry

end lana_trip_longer_by_25_percent_l1236_123671


namespace george_stickers_l1236_123646

theorem george_stickers :
  let bob_stickers := 12
  let tom_stickers := 3 * bob_stickers
  let dan_stickers := 2 * tom_stickers
  let george_stickers := 5 * dan_stickers
  george_stickers = 360 := by
  sorry

end george_stickers_l1236_123646


namespace number_of_B_eq_l1236_123631

variable (a b : ℝ)
variable (B : ℝ)

theorem number_of_B_eq : 3 * B = a + b → B = (a + b) / 3 :=
by sorry

end number_of_B_eq_l1236_123631


namespace factorize_poly_l1236_123601

open Polynomial

theorem factorize_poly : 
  (X ^ 15 + X ^ 7 + 1 : Polynomial ℤ) =
    (X^2 + X + 1) * (X^13 - X^12 + X^10 - X^9 + X^7 - X^6 + X^4 - X^3 + X - 1) := 
  by
  sorry

end factorize_poly_l1236_123601


namespace root_of_equation_l1236_123699

theorem root_of_equation (x : ℝ) :
  (∃ u : ℝ, u = Real.sqrt (x + 15) ∧ u - 7 / u = 6) → x = 34 :=
by
  sorry

end root_of_equation_l1236_123699


namespace sin_pi_plus_alpha_l1236_123618

open Real

-- Define the given conditions
variable (α : ℝ) (hα1 : sin (π / 2 + α) = 3 / 5) (hα2 : 0 < α ∧ α < π / 2)

-- The theorem statement that must be proved
theorem sin_pi_plus_alpha : sin (π + α) = -4 / 5 :=
by
  sorry

end sin_pi_plus_alpha_l1236_123618


namespace total_apples_picked_l1236_123638

theorem total_apples_picked (benny_apples : ℕ) (dan_apples : ℕ) (h_benny : benny_apples = 2) (h_dan : dan_apples = 9) :
  benny_apples + dan_apples = 11 :=
by
  sorry

end total_apples_picked_l1236_123638


namespace angle_between_apothems_correct_l1236_123611

noncomputable def angle_between_apothems (n : ℕ) (α : ℝ) : ℝ :=
  2 * Real.arcsin (Real.cos (Real.pi / n) * Real.tan (α / 2))

theorem angle_between_apothems_correct (n : ℕ) (α : ℝ) (h1 : 0 < n) (h2 : 0 < α) (h3 : α < 2 * Real.pi) :
  angle_between_apothems n α = 2 * Real.arcsin (Real.cos (Real.pi / n) * Real.tan (α / 2)) :=
by
  sorry

end angle_between_apothems_correct_l1236_123611


namespace machines_needed_l1236_123614

theorem machines_needed (original_machines : ℕ) (original_days : ℕ) (additional_machines : ℕ) :
  original_machines = 12 → original_days = 40 → 
  additional_machines = ((original_machines * original_days) / (3 * original_days / 4)) - original_machines →
  additional_machines = 4 :=
by
  intros h_machines h_days h_additional
  rw [h_machines, h_days] at h_additional
  sorry

end machines_needed_l1236_123614
