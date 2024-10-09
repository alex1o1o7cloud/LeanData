import Mathlib

namespace benny_birthday_money_l1179_117914

def money_spent_on_gear : ℕ := 34
def money_left_over : ℕ := 33

theorem benny_birthday_money : money_spent_on_gear + money_left_over = 67 :=
by
  sorry

end benny_birthday_money_l1179_117914


namespace two_digit_number_is_24_l1179_117903

-- Defining the two-digit number conditions

variables (x y : ℕ)

noncomputable def condition1 := y = x + 2
noncomputable def condition2 := (10 * x + y) * (x + y) = 144

-- The statement of the proof problem
theorem two_digit_number_is_24 (h1 : condition1 x y) (h2 : condition2 x y) : 10 * x + y = 24 :=
sorry

end two_digit_number_is_24_l1179_117903


namespace subset_to_union_eq_l1179_117994

open Set

variable {α : Type*} (A B : Set α)

theorem subset_to_union_eq (h : A ∩ B = A) : A ∪ B = B :=
by
  sorry

end subset_to_union_eq_l1179_117994


namespace sum_of_distinct_integers_l1179_117944

theorem sum_of_distinct_integers 
  (p q r s : ℕ) 
  (h1 : p * q = 6) 
  (h2 : r * s = 8) 
  (h3 : p * r = 4) 
  (h4 : q * s = 12) 
  (hpqrs : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) : 
  p + q + r + s = 13 :=
sorry

end sum_of_distinct_integers_l1179_117944


namespace no_solution_for_floor_x_plus_x_eq_15_point_3_l1179_117900

theorem no_solution_for_floor_x_plus_x_eq_15_point_3 : ¬ ∃ (x : ℝ), (⌊x⌋ : ℝ) + x = 15.3 := by
  sorry

end no_solution_for_floor_x_plus_x_eq_15_point_3_l1179_117900


namespace correct_option_D_l1179_117995

theorem correct_option_D (defect_rate_products : ℚ)
                         (rain_probability : ℚ)
                         (cure_rate_hospital : ℚ)
                         (coin_toss_heads_probability : ℚ)
                         (coin_toss_tails_probability : ℚ):
  defect_rate_products = 1/10 →
  rain_probability = 0.9 →
  cure_rate_hospital = 0.1 →
  coin_toss_heads_probability = 0.5 →
  coin_toss_tails_probability = 0.5 →
  coin_toss_tails_probability = 0.5 :=
by
  intros h1 h2 h3 h4 h5
  exact h5

end correct_option_D_l1179_117995


namespace fraction_zero_numerator_l1179_117965

theorem fraction_zero_numerator (x : ℝ) (h₁ : (x^2 - 9) / (x + 3) = 0) (h₂ : x + 3 ≠ 0) : x = 3 :=
sorry

end fraction_zero_numerator_l1179_117965


namespace businessmen_neither_coffee_nor_tea_l1179_117918

/-- Definitions of conditions -/
def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 13
def both_drinkers : ℕ := 6

/-- Statement of the problem -/
theorem businessmen_neither_coffee_nor_tea : 
  (total_businessmen - (coffee_drinkers + tea_drinkers - both_drinkers)) = 8 := 
by
  sorry

end businessmen_neither_coffee_nor_tea_l1179_117918


namespace concentric_but_different_radius_l1179_117983

noncomputable def circleF (x y : ℝ) : ℝ :=
  x^2 + y^2 - 1

def pointP (x : ℝ) : ℝ × ℝ :=
  (x, x)

def circleEquation (x y : ℝ) : Prop :=
  circleF x y = 0

def circleEquation' (x y : ℝ) : Prop :=
  circleF x y - circleF x y = 0

theorem concentric_but_different_radius (x : ℝ) (hP : circleF x x ≠ 0) (hCenter : x ≠ 0):
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧
    ∀ x y, (circleEquation x y ↔ x^2 + y^2 = 1) ∧ 
           (circleEquation' x y ↔ x^2 + y^2 = 2) :=
by
  sorry

end concentric_but_different_radius_l1179_117983


namespace find_f2009_l1179_117952

noncomputable def f : ℝ → ℝ :=
sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom functional_equation (x : ℝ) : f (2 + x) = -f (2 - x)
axiom initial_condition : f (-3) = -2

theorem find_f2009 : f 2009 = 2 :=
sorry

end find_f2009_l1179_117952


namespace zack_group_size_l1179_117940

theorem zack_group_size (total_students : Nat) (groups : Nat) (group_size : Nat)
  (H1 : total_students = 70)
  (H2 : groups = 7)
  (H3 : total_students = group_size * groups) :
  group_size = 10 := by
  sorry

end zack_group_size_l1179_117940


namespace sum_of_decimals_l1179_117943

-- Defining the specific decimal values as constants
def x : ℝ := 5.47
def y : ℝ := 4.26

-- Noncomputable version for addition to allow Lean to handle real number operations safely
noncomputable def sum : ℝ := x + y

-- Theorem statement asserting the sum of x and y
theorem sum_of_decimals : sum = 9.73 := 
by
  -- This is where the proof would go
  sorry

end sum_of_decimals_l1179_117943


namespace pants_original_price_l1179_117913

theorem pants_original_price (P : ℝ) (h1 : P * 0.6 = 50.40) : P = 84 :=
sorry

end pants_original_price_l1179_117913


namespace sodium_bicarbonate_moles_needed_l1179_117910

-- Definitions for the problem.
def balanced_reaction : Prop := 
  ∀ (NaHCO₃ HCl NaCl H₂O CO₂ : Type) (moles_NaHCO₃ moles_HCl moles_NaCl moles_H₂O moles_CO₂ : Nat),
  (moles_NaHCO₃ = moles_HCl) → 
  (moles_NaCl = moles_HCl) → 
  (moles_H₂O = moles_HCl) → 
  (moles_CO₂ = moles_HCl)

-- Given condition: 3 moles of HCl
def moles_HCl : Nat := 3

-- The theorem statement
theorem sodium_bicarbonate_moles_needed : 
  balanced_reaction → moles_HCl = 3 → ∃ moles_NaHCO₃, moles_NaHCO₃ = 3 :=
by 
  -- Proof will be provided here.
  sorry

end sodium_bicarbonate_moles_needed_l1179_117910


namespace text_messages_in_march_l1179_117992

/-
Jared sent text messages each month according to the formula:
  T_n = n^3 - n^2 + n
We need to prove that the number of text messages Jared will send in March
(which is the 5th month) is given by T_5 = 105.
-/

def T (n : ℕ) : ℕ := n^3 - n^2 + n

theorem text_messages_in_march : T 5 = 105 :=
by
  -- proof goes here
  sorry

end text_messages_in_march_l1179_117992


namespace div_by_7_l1179_117991

theorem div_by_7 (k : ℕ) : (2^(6*k + 1) + 3^(6*k + 1) + 5^(6*k + 1)) % 7 = 0 := by
  sorry

end div_by_7_l1179_117991


namespace point_in_plane_region_l1179_117935

theorem point_in_plane_region :
  let P := (0, 0)
  let Q := (2, 4)
  let R := (-1, 4)
  let S := (1, 8)
  (P.1 + P.2 - 1 < 0) ∧ ¬(Q.1 + Q.2 - 1 < 0) ∧ ¬(R.1 + R.2 - 1 < 0) ∧ ¬(S.1 + S.2 - 1 < 0) :=
by
  sorry

end point_in_plane_region_l1179_117935


namespace correct_quotient_l1179_117939

-- Define number N based on given conditions
def N : ℕ := 9 * 8 + 6

-- Prove that the correct quotient when N is divided by 6 is 13
theorem correct_quotient : N / 6 = 13 := 
by {
  sorry
}

end correct_quotient_l1179_117939


namespace geometric_sequence_fraction_l1179_117908

open Classical

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_fraction {a : ℕ → ℝ} {q : ℝ}
  (h₀ : ∀ n, 0 < a n)
  (h₁ : geometric_seq a q)
  (h₂ : 2 * (1 / 2 * a 2) = a 0 + 2 * a 1) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_fraction_l1179_117908


namespace find_m_plus_b_l1179_117980

-- Define the given equation
def given_line (x y : ℝ) : Prop := x - 3 * y + 11 = 0

-- Define the reflection of the given line about the x-axis
def reflected_line (x y : ℝ) : Prop := x + 3 * y + 11 = 0

-- Define the slope-intercept form of the reflected line
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- State the theorem to prove
theorem find_m_plus_b (m b : ℝ) :
  (∀ x y : ℝ, reflected_line x y ↔ slope_intercept_form m b x y) → m + b = -4 :=
by
  sorry

end find_m_plus_b_l1179_117980


namespace pat_peano_maximum_pages_l1179_117990

noncomputable def count_fives_in_range : ℕ → ℕ := sorry

theorem pat_peano_maximum_pages (n : ℕ) : 
  (count_fives_in_range 54) = 15 → n ≤ 54 :=
sorry

end pat_peano_maximum_pages_l1179_117990


namespace wendy_first_day_miles_l1179_117977

-- Define the variables for the problem
def total_miles : ℕ := 493
def miles_day2 : ℕ := 223
def miles_day3 : ℕ := 145

-- Define the proof problem
theorem wendy_first_day_miles :
  total_miles = miles_day2 + miles_day3 + 125 :=
sorry

end wendy_first_day_miles_l1179_117977


namespace pumpkin_weight_difference_l1179_117905

variable (Brad_weight Jessica_weight Betty_weight : ℕ)

theorem pumpkin_weight_difference :
  Brad_weight = 54 →
  Jessica_weight = Brad_weight / 2 →
  Betty_weight = 4 * Jessica_weight →
  Betty_weight - Jessica_weight = 81 := by
  sorry

end pumpkin_weight_difference_l1179_117905


namespace dance_problem_l1179_117973

theorem dance_problem :
  ∃ (G : ℝ) (B T : ℝ),
    B / G = 3 / 4 ∧
    T = 0.20 * B ∧
    B + G + T = 114 ∧
    G = 60 :=
by
  sorry

end dance_problem_l1179_117973


namespace find_third_angle_l1179_117953

variable (A B C : ℝ)

theorem find_third_angle
  (hA : A = 32)
  (hB : B = 3 * A)
  (hC : C = 2 * A - 12) :
  C = 52 := by
  sorry

end find_third_angle_l1179_117953


namespace students_in_class_l1179_117912

theorem students_in_class (N : ℕ) 
  (avg_age_class : ℕ) (avg_age_4 : ℕ) (avg_age_10 : ℕ) (age_15th : ℕ) 
  (total_age_class : ℕ) (total_age_4 : ℕ) (total_age_10 : ℕ)
  (h1 : avg_age_class = 15)
  (h2 : avg_age_4 = 14)
  (h3 : avg_age_10 = 16)
  (h4 : age_15th = 9)
  (h5 : total_age_class = avg_age_class * N)
  (h6 : total_age_4 = 4 * avg_age_4)
  (h7 : total_age_10 = 10 * avg_age_10)
  (h8 : total_age_class = total_age_4 + total_age_10 + age_15th) :
  N = 15 :=
by
  sorry

end students_in_class_l1179_117912


namespace first_math_festival_divisibility_largest_ordinal_number_divisibility_l1179_117932

-- Definition of the conditions for part (a)
def first_math_festival_year : ℕ := 1990
def first_ordinal_number : ℕ := 1

-- Statement for part (a)
theorem first_math_festival_divisibility : first_math_festival_year % first_ordinal_number = 0 :=
sorry

-- Definition of the conditions for part (b)
def nth_math_festival_year (N : ℕ) : ℕ := 1989 + N

-- Statement for part (b)
theorem largest_ordinal_number_divisibility : ∀ N : ℕ, 
  (nth_math_festival_year N) % N = 0 → N ≤ 1989 :=
sorry

end first_math_festival_divisibility_largest_ordinal_number_divisibility_l1179_117932


namespace tangent_line_characterization_l1179_117989

theorem tangent_line_characterization 
  (α β m n : ℝ) 
  (h_pos_α : 0 < α) 
  (h_pos_β : 0 < β) 
  (h_alpha_beta : 1/α + 1/β = 1)
  (h_pos_m : 0 < m)
  (h_pos_n : 0 < n) :
  (∀ (x y : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ x^α + y^α = 1 → mx + ny = 1) ↔ (m^β + n^β = 1) := 
sorry

end tangent_line_characterization_l1179_117989


namespace triangle_area_l1179_117961

theorem triangle_area (area_WXYZ : ℝ) (side_small_squares : ℝ) 
  (AB_eq_AC : (AB = AC)) (A_on_center : (A = O)) :
  area_WXYZ = 64 ∧ side_small_squares = 2 →
  ∃ (area_triangle_ABC : ℝ), area_triangle_ABC = 8 :=
by
  intros h
  sorry

end triangle_area_l1179_117961


namespace algebraic_expression_value_l1179_117969

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 2 * a - 1 = 0) : 2 * a^2 - 4 * a + 2022 = 2024 := 
by 
  sorry

end algebraic_expression_value_l1179_117969


namespace production_equation_l1179_117948

-- Define the conditions as per the problem
variables (workers : ℕ) (x : ℕ) 

-- The number of total workers is fixed
def total_workers := 44

-- Production rates per worker
def bodies_per_worker := 50
def bottoms_per_worker := 120

-- The problem statement as a Lean theorem
theorem production_equation (h : workers = total_workers) (hx : x ≤ workers) :
  2 * bottoms_per_worker * (total_workers - x) = bodies_per_worker * x :=
by
  sorry

end production_equation_l1179_117948


namespace overtime_hours_correct_l1179_117974

def regular_pay_rate : ℕ := 3
def max_regular_hours : ℕ := 40
def total_pay_received : ℕ := 192
def overtime_pay_rate : ℕ := 2 * regular_pay_rate
def regular_earnings : ℕ := regular_pay_rate * max_regular_hours
def additional_earnings : ℕ := total_pay_received - regular_earnings
def calculated_overtime_hours : ℕ := additional_earnings / overtime_pay_rate

theorem overtime_hours_correct :
  calculated_overtime_hours = 12 :=
by
  sorry

end overtime_hours_correct_l1179_117974


namespace find_real_numbers_l1179_117934

theorem find_real_numbers (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x^2 + y^2 + z^2 = 6) 
  (h3 : x^3 + y^3 + z^3 = 8) :
  (x = 1 ∧ y = 2 ∧ z = -1) ∨ 
  (x = 1 ∧ y = -1 ∧ z = 2) ∨
  (x = 2 ∧ y = 1 ∧ z = -1) ∨ 
  (x = 2 ∧ y = -1 ∧ z = 1) ∨
  (x = -1 ∧ y = 1 ∧ z = 2) ∨
  (x = -1 ∧ y = 2 ∧ z = 1) := 
sorry

end find_real_numbers_l1179_117934


namespace monochromatic_triangle_probability_l1179_117920

-- Define the coloring of the edges
inductive Color
| Red : Color
| Blue : Color

-- Define an edge
structure Edge :=
(v1 v2 : Nat)
(color : Color)

-- Define the hexagon with its sides and diagonals
def hexagonEdges : List Edge := [
  -- Sides of the hexagon
  { v1 := 1, v2 := 2, color := sorry }, { v1 := 2, v2 := 3, color := sorry },
  { v1 := 3, v2 := 4, color := sorry }, { v1 := 4, v2 := 5, color := sorry },
  { v1 := 5, v2 := 6, color := sorry }, { v1 := 6, v2 := 1, color := sorry },
  -- Diagonals of the hexagon
  { v1 := 1, v2 := 3, color := sorry }, { v1 := 1, v2 := 4, color := sorry },
  { v1 := 1, v2 := 5, color := sorry }, { v1 := 2, v2 := 4, color := sorry },
  { v1 := 2, v2 := 5, color := sorry }, { v1 := 2, v2 := 6, color := sorry },
  { v1 := 3, v2 := 5, color := sorry }, { v1 := 3, v2 := 6, color := sorry },
  { v1 := 4, v2 := 6, color := sorry }
]

-- Define what a triangle is
structure Triangle :=
(v1 v2 v3 : Nat)

-- List all possible triangles formed by vertices of the hexagon
def hexagonTriangles : List Triangle := [
  { v1 := 1, v2 := 2, v3 := 3 }, { v1 := 1, v2 := 2, v3 := 4 },
  { v1 := 1, v2 := 2, v3 := 5 }, { v1 := 1, v2 := 2, v3 := 6 },
  { v1 := 1, v2 := 3, v3 := 4 }, { v1 := 1, v2 := 3, v3 := 5 },
  { v1 := 1, v2 := 3, v3 := 6 }, { v1 := 1, v2 := 4, v3 := 5 },
  { v1 := 1, v2 := 4, v3 := 6 }, { v1 := 1, v2 := 5, v3 := 6 },
  { v1 := 2, v2 := 3, v3 := 4 }, { v1 := 2, v2 := 3, v3 := 5 },
  { v1 := 2, v2 := 3, v3 := 6 }, { v1 := 2, v2 := 4, v3 := 5 },
  { v1 := 2, v2 := 4, v3 := 6 }, { v1 := 2, v2 := 5, v3 := 6 },
  { v1 := 3, v2 := 4, v3 := 5 }, { v1 := 3, v2 := 4, v3 := 6 },
  { v1 := 3, v2 := 5, v3 := 6 }, { v1 := 4, v2 := 5, v3 := 6 }
]

-- Define the probability calculation, with placeholders for terms that need proving
noncomputable def probabilityMonochromaticTriangle : ℚ :=
  1 - (3 / 4) ^ 20

-- The theorem to prove the probability matches the given answer
theorem monochromatic_triangle_probability :
  probabilityMonochromaticTriangle = 253 / 256 :=
by sorry

end monochromatic_triangle_probability_l1179_117920


namespace percent_increase_first_quarter_l1179_117984

theorem percent_increase_first_quarter (S : ℝ) (P : ℝ) :
  (S * 1.75 = (S + (P / 100) * S) * 1.346153846153846) → P = 30 :=
by
  intro h
  sorry

end percent_increase_first_quarter_l1179_117984


namespace total_books_l1179_117947

-- Define the number of books each person has
def books_beatrix : ℕ := 30
def books_alannah : ℕ := books_beatrix + 20
def books_queen : ℕ := books_alannah + (books_alannah / 5)

-- State the theorem to be proved
theorem total_books (h_beatrix : books_beatrix = 30)
                    (h_alannah : books_alannah = books_beatrix + 20)
                    (h_queen : books_queen = books_alannah + (books_alannah / 5)) :
  books_alannah + books_beatrix + books_queen = 140 :=
sorry

end total_books_l1179_117947


namespace fraction_of_ponies_with_horseshoes_l1179_117957

variable (P H : ℕ)
variable (F : ℚ)

theorem fraction_of_ponies_with_horseshoes 
  (h1 : H = P + 3)
  (h2 : P + H = 163)
  (h3 : (5/8 : ℚ) * F * P = 5) :
  F = 1/10 :=
  sorry

end fraction_of_ponies_with_horseshoes_l1179_117957


namespace minimize_sum_of_squares_l1179_117954

noncomputable def sum_of_squares (x : ℝ) : ℝ := x^2 + (18 - x)^2

theorem minimize_sum_of_squares : ∃ x : ℝ, x = 9 ∧ (18 - x) = 9 ∧ ∀ y : ℝ, sum_of_squares y ≥ sum_of_squares 9 :=
by
  sorry

end minimize_sum_of_squares_l1179_117954


namespace fish_disappeared_l1179_117936

theorem fish_disappeared (g : ℕ) (c : ℕ) (left : ℕ) (disappeared : ℕ) (h₁ : g = 7) (h₂ : c = 12) (h₃ : left = 15) (h₄ : g + c - left = disappeared) : disappeared = 4 :=
by
  sorry

end fish_disappeared_l1179_117936


namespace vertex_of_parabola_l1179_117978

theorem vertex_of_parabola :
  ∀ (x y : ℝ), y = (1 / 3) * (x - 7) ^ 2 + 5 → ∃ h k : ℝ, h = 7 ∧ k = 5 ∧ y = (1 / 3) * (x - h) ^ 2 + k :=
by
  intro x y h
  sorry

end vertex_of_parabola_l1179_117978


namespace woodworker_tables_l1179_117968

theorem woodworker_tables (L C_leg C T_leg : ℕ) (hL : L = 40) (hC_leg : C_leg = 4) (hC : C = 6) (hT_leg : T_leg = 4) :
  T = (L - C * C_leg) / T_leg := by
  sorry

end woodworker_tables_l1179_117968


namespace ways_to_make_50_cents_without_dimes_or_quarters_l1179_117902

theorem ways_to_make_50_cents_without_dimes_or_quarters : 
  ∃ (n : ℕ), n = 1024 := 
by
  let num_ways := (2 ^ 10)
  existsi num_ways
  sorry

end ways_to_make_50_cents_without_dimes_or_quarters_l1179_117902


namespace k_even_l1179_117982

theorem k_even (n a b k : ℕ) (h1 : 2^n - 1 = a * b) (h2 : 2^k ∣ 2^(n-2) + a - b):
  k % 2 = 0 :=
sorry

end k_even_l1179_117982


namespace quadratic_has_two_distinct_real_roots_l1179_117922

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  (x : ℝ) -> x^2 + m * x + 1 = 0 → (m < -2 ∨ m > 2) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l1179_117922


namespace fraction_relation_l1179_117958

theorem fraction_relation (a b : ℝ) (h : a / b = 2 / 3) : (a - b) / b = -1 / 3 :=
by
  sorry

end fraction_relation_l1179_117958


namespace son_age_is_14_l1179_117937

-- Definition of Sandra's age and the condition about the ages 3 years ago.
def Sandra_age : ℕ := 36
def son_age_3_years_ago (son_age_now : ℕ) : ℕ := son_age_now - 3 
def Sandra_age_3_years_ago := 36 - 3
def condition_3_years_ago (son_age_now : ℕ) : Prop := Sandra_age_3_years_ago = 3 * (son_age_3_years_ago son_age_now)

-- The goal: proving Sandra's son's age is 14
theorem son_age_is_14 (son_age_now : ℕ) (h : condition_3_years_ago son_age_now) : son_age_now = 14 :=
by {
  sorry
}

end son_age_is_14_l1179_117937


namespace ratio_nonupgraded_to_upgraded_l1179_117979

-- Define the initial conditions and properties
variable (S : ℝ) (N : ℝ)
variable (h1 : ∀ N, N = S / 32)
variable (h2 : ∀ S, 0.25 * S = 0.25 * S)
variable (h3 : S > 0)

-- Define the theorem to show the required ratio
theorem ratio_nonupgraded_to_upgraded (h3 : 24 * N = 0.75 * S) : (N / (0.25 * S) = 1 / 8) :=
by
  sorry

end ratio_nonupgraded_to_upgraded_l1179_117979


namespace find_a_l1179_117950

theorem find_a (a : ℝ) : (∃ b : ℝ, ∀ x : ℝ, (4 * x^2 + 12 * x + a = (2 * x + b) ^ 2)) → a = 9 :=
by
  intro h
  sorry

end find_a_l1179_117950


namespace polygon_area_correct_l1179_117971

def AreaOfPolygon : Real := 37.5

def polygonVertices : List (Real × Real) :=
  [(0, 0), (5, 0), (5, 5), (0, 5), (5, 10), (0, 10), (0, 0)]

theorem polygon_area_correct :
  (∃ (A : Real) (verts : List (Real × Real)),
    verts = polygonVertices ∧ A = AreaOfPolygon ∧ 
    A = 37.5) := by
  sorry

end polygon_area_correct_l1179_117971


namespace snack_cost_inequality_l1179_117916

variables (S : ℝ)

def cost_water : ℝ := 0.50
def cost_fruit : ℝ := 0.25
def bundle_price : ℝ := 4.60
def special_price : ℝ := 2.00

theorem snack_cost_inequality (h : bundle_price = 4.60 ∧ special_price = 2.00 ∧
  cost_water = 0.50 ∧ cost_fruit = 0.25) : S < 15.40 / 16 := sorry

end snack_cost_inequality_l1179_117916


namespace min_value_of_f_at_sqrt2_l1179_117959

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x + (1 / x)))

theorem min_value_of_f_at_sqrt2 :
  f (Real.sqrt 2) = (11 * Real.sqrt 2) / 6 :=
sorry

end min_value_of_f_at_sqrt2_l1179_117959


namespace solve_case1_solve_case2_l1179_117942

variables (a b c A B C x y z : ℝ)

-- Define the conditions for the first special case
def conditions_case1 := (A = b + c) ∧ (B = c + a) ∧ (C = a + b)

-- State the proposition to prove for the first special case
theorem solve_case1 (h : conditions_case1 a b c A B C) :
  z = 0 ∧ y = -1 ∧ x = A + b := by
  sorry

-- Define the conditions for the second special case
def conditions_case2 := (A = b * c) ∧ (B = c * a) ∧ (C = a * b)

-- State the proposition to prove for the second special case
theorem solve_case2 (h : conditions_case2 a b c A B C) :
  z = 1 ∧ y = -(a + b + c) ∧ x = a * b * c := by
  sorry

end solve_case1_solve_case2_l1179_117942


namespace exist_positive_abc_with_nonzero_integer_roots_l1179_117997

theorem exist_positive_abc_with_nonzero_integer_roots :
  ∃ (a b c : ℤ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (∀ x y : ℤ, (a * x^2 + b * x + c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 + b * x - c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 - b * x + c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 - b * x - c = 0 → x ≠ 0 ∧ y ≠ 0)) :=
sorry

end exist_positive_abc_with_nonzero_integer_roots_l1179_117997


namespace winning_candidate_percentage_l1179_117909

theorem winning_candidate_percentage (votes1 votes2 votes3 : ℕ) (h1 : votes1 = 1256) (h2 : votes2 = 7636) (h3 : votes3 = 11628) 
    : (votes3 : ℝ) / (votes1 + votes2 + votes3) * 100 = 56.67 := by
  sorry

end winning_candidate_percentage_l1179_117909


namespace soda_cost_l1179_117929

theorem soda_cost (S P W : ℝ) (h1 : P = 3 * S) (h2 : W = 3 * P) (h3 : 3 * S + 2 * P + W = 18) : S = 1 :=
by
  sorry

end soda_cost_l1179_117929


namespace problem_1_problem_2_l1179_117956

-- Problem 1 proof statement
theorem problem_1 (x : ℝ) (h : x = -1) : 
  (1 * (-x^2 + 5 * x) - (x - 3) - 4 * x) = 2 := by
  -- Placeholder for the proof
  sorry

-- Problem 2 proof statement
theorem problem_2 (m n : ℝ) (h_m : m = -1/2) (h_n : n = 1/3) : 
  (5 * (3 * m^2 * n - m * n^2) - (m * n^2 + 3 * m^2 * n)) = 4/3 := by
  -- Placeholder for the proof
  sorry

end problem_1_problem_2_l1179_117956


namespace ordering_of_a_b_c_l1179_117960

noncomputable def a := 1 / Real.exp 1
noncomputable def b := Real.log 3 / 3
noncomputable def c := Real.log 4 / 4

-- We need to prove that the ordering is a > b > c.

theorem ordering_of_a_b_c : a > b ∧ b > c :=
by 
  sorry

end ordering_of_a_b_c_l1179_117960


namespace find_initial_money_l1179_117998
 
theorem find_initial_money (x : ℕ) (gift_grandma gift_aunt_uncle gift_parents total_money : ℕ) 
  (h1 : gift_grandma = 25) 
  (h2 : gift_aunt_uncle = 20) 
  (h3 : gift_parents = 75) 
  (h4 : total_money = 279) 
  (h : x + (gift_grandma + gift_aunt_uncle + gift_parents) = total_money) : 
  x = 159 :=
by
  sorry

end find_initial_money_l1179_117998


namespace exchange_rate_decrease_l1179_117917

theorem exchange_rate_decrease
  (x y z : ℝ)
  (hx : 0 < |x| ∧ |x| < 1)
  (hy : 0 < |y| ∧ |y| < 1)
  (hz : 0 < |z| ∧ |z| < 1)
  (h_eq : (1 + x) * (1 + y) * (1 + z) = (1 - x) * (1 - y) * (1 - z)) :
  (1 - x^2) * (1 - y^2) * (1 - z^2) < 1 :=
by
  sorry

end exchange_rate_decrease_l1179_117917


namespace fifth_dog_is_older_than_fourth_l1179_117962

theorem fifth_dog_is_older_than_fourth :
  ∀ (age_1 age_2 age_3 age_4 age_5 : ℕ),
  (age_1 = 10) →
  (age_2 = age_1 - 2) →
  (age_3 = age_2 + 4) →
  (age_4 = age_3 / 2) →
  (age_5 = age_4 + 20) →
  ((age_1 + age_5) / 2 = 18) →
  (age_5 - age_4 = 20) :=
by
  intros age_1 age_2 age_3 age_4 age_5 h1 h2 h3 h4 h5 h_avg
  sorry

end fifth_dog_is_older_than_fourth_l1179_117962


namespace total_earnings_l1179_117906

theorem total_earnings :
  (15 * 2) + (12 * 1.5) = 48 := by
  sorry

end total_earnings_l1179_117906


namespace problem_l1179_117911

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x >= 0 then Real.log x / Real.log 3 + m else 1 / 2017

theorem problem (m := -2) (h_root : f 3 m = 0):
  f (f 6 m - 2) m = 1 / 2017 :=
by
  sorry

end problem_l1179_117911


namespace unique_positive_integer_solution_l1179_117963

theorem unique_positive_integer_solution (p : ℕ) (hp : Nat.Prime p) (hop : p % 2 = 1) :
  ∃! (x y : ℕ), x^2 + p * x = y^2 ∧ x > 0 ∧ y > 0 :=
sorry

end unique_positive_integer_solution_l1179_117963


namespace range_of_a_l1179_117904

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * x + a * Real.log x

theorem range_of_a (a : ℝ) : 
  (∀ t : ℝ, t ≥ 1 → f (2 * t - 1) a ≥ 2 * f t a - 3) ↔ a < 2 := 
by 
  sorry

end range_of_a_l1179_117904


namespace area_fraction_of_square_hole_l1179_117919

theorem area_fraction_of_square_hole (A B C M N : ℝ)
  (h1 : B = C)
  (h2 : M = 0.5 * A)
  (h3 : N = 0.5 * A) :
  (M * N) / (B * C) = 1 / 4 :=
by
  sorry

end area_fraction_of_square_hole_l1179_117919


namespace positive_difference_of_numbers_l1179_117986

theorem positive_difference_of_numbers :
  ∃ x y : ℕ, x + y = 50 ∧ 3 * y - 4 * x = 10 ∧ y - x = 10 :=
by
  sorry

end positive_difference_of_numbers_l1179_117986


namespace largest_y_coordinate_of_degenerate_ellipse_l1179_117938

theorem largest_y_coordinate_of_degenerate_ellipse :
  ∀ x y : ℝ, (x^2 / 49 + (y - 3)^2 / 25 = 0) → y ≤ 3 := by
  sorry

end largest_y_coordinate_of_degenerate_ellipse_l1179_117938


namespace staffing_correct_l1179_117975

-- The number of ways to staff a battle station with constraints.
def staffing_ways (total_applicants unsuitable_fraction: ℕ) (job_openings: ℕ): ℕ :=
  let suitable_candidates := total_applicants * (1 - unsuitable_fraction)
  if suitable_candidates < job_openings then
    0 
  else
    (List.range' (suitable_candidates - job_openings + 1) job_openings).prod

-- Definitions of the problem conditions
def total_applicants := 30
def unsuitable_fraction := 2/3
def job_openings := 5
-- Expected result
def expected_result := 30240

-- The theorem to prove the number of ways to staff the battle station equals the given result.
theorem staffing_correct : staffing_ways total_applicants unsuitable_fraction job_openings = expected_result := by
  sorry

end staffing_correct_l1179_117975


namespace g_neg_eleven_eq_neg_two_l1179_117996

def f (x : ℝ) : ℝ := 2 * x - 7
def g (y : ℝ) : ℝ := 3 * y^2 + 4 * y - 6

theorem g_neg_eleven_eq_neg_two : g (-11) = -2 := by
  sorry

end g_neg_eleven_eq_neg_two_l1179_117996


namespace tan_alpha_plus_pi_over_12_l1179_117931

theorem tan_alpha_plus_pi_over_12 (α : ℝ) (h : Real.sin α = 3 * Real.sin (α + π / 6)) :
  Real.tan (α + π / 12) = 2 * Real.sqrt 3 - 4 :=
by
  sorry

end tan_alpha_plus_pi_over_12_l1179_117931


namespace contrapositive_example_contrapositive_proof_l1179_117928

theorem contrapositive_example (x : ℝ) (h : x > 1) : x^2 > 1 := 
sorry

theorem contrapositive_proof (x : ℝ) (h : x^2 ≤ 1) : x ≤ 1 :=
sorry

end contrapositive_example_contrapositive_proof_l1179_117928


namespace find_k_l1179_117949

theorem find_k (k : ℤ)
  (h : ∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ ∀ x, ((k^2 - 1) * x^2 - 3 * (3 * k - 1) * x + 18 = 0) ↔ (x = x₁ ∨ x = x₂)
       ∧ x₁ > 0 ∧ x₂ > 0) : k = 2 :=
by
  sorry

end find_k_l1179_117949


namespace find_y_l1179_117946

theorem find_y (y : ℕ) (hy_mult_of_7 : ∃ k, y = 7 * k) (hy_pos : 0 < y) (hy_square : y^2 > 225) (hy_upper_bound : y < 30) : y = 21 :=
sorry

end find_y_l1179_117946


namespace cost_of_soccer_ball_l1179_117921

theorem cost_of_soccer_ball
  (F S : ℝ)
  (h1 : 3 * F + S = 155)
  (h2 : 2 * F + 3 * S = 220) :
  S = 50 :=
sorry

end cost_of_soccer_ball_l1179_117921


namespace small_stick_length_l1179_117970

theorem small_stick_length 
  (x : ℝ) 
  (hx1 : 3 < x) 
  (hx2 : x < 9) 
  (hx3 : 3 + 6 > x) : 
  x = 4 := 
by 
  sorry

end small_stick_length_l1179_117970


namespace find_m_n_l1179_117907

theorem find_m_n : ∃ (m n : ℕ), m > n ∧ m^3 - n^3 = 999 ∧ ((m = 10 ∧ n = 1) ∨ (m = 12 ∧ n = 9)) :=
by
  sorry

end find_m_n_l1179_117907


namespace gcd_8251_6105_l1179_117964

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 :=
by
  sorry

end gcd_8251_6105_l1179_117964


namespace banknotes_sum_divisible_by_101_l1179_117930

theorem banknotes_sum_divisible_by_101 (a b : ℕ) (h₀ : a ≠ b % 101) : 
  ∃ (m n : ℕ), m + n = 100 ∧ ∃ k l : ℕ, k ≤ m ∧ l ≤ n ∧ (k * a + l * b) % 101 = 0 :=
sorry

end banknotes_sum_divisible_by_101_l1179_117930


namespace car_sharing_problem_l1179_117915

theorem car_sharing_problem 
  (x : ℕ)
  (cond1 : ∃ c : ℕ, x = 4 * c + 4)
  (cond2 : ∃ c : ℕ, x = 3 * c + 9):
  (x / 4 + 1 = (x - 9) / 3) :=
by sorry

end car_sharing_problem_l1179_117915


namespace possible_case_l1179_117924

-- Define the logical propositions P and Q
variables (P Q : Prop)

-- State the conditions given in the problem
axiom h1 : P ∨ Q     -- P ∨ Q is true
axiom h2 : ¬ (P ∧ Q) -- P ∧ Q is false

-- Formulate the proof problem in Lean
theorem possible_case : P ∧ ¬Q :=
by
  sorry -- Proof to be filled in later

end possible_case_l1179_117924


namespace trapezoid_area_l1179_117941

theorem trapezoid_area 
  (area_ABE area_ADE : ℝ)
  (DE BE : ℝ)
  (h1 : area_ABE = 40)
  (h2 : area_ADE = 30)
  (h3 : DE = 2 * BE) : 
  area_ABE + area_ADE + area_ADE + 4 * area_ABE = 260 :=
by
  -- sorry admits the goal without providing the actual proof
  sorry

end trapezoid_area_l1179_117941


namespace find_n_values_l1179_117981

-- Define a function to sum the first n consecutive natural numbers starting from k
def sum_consecutive_numbers (n k : ℕ) : ℕ :=
  n * k + (n * (n - 1)) / 2

-- Define a predicate to check if a number is a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the theorem statement
theorem find_n_values (n : ℕ) (k : ℕ) :
  is_prime (sum_consecutive_numbers n k) →
  n = 1 ∨ n = 2 :=
sorry

end find_n_values_l1179_117981


namespace sum_f_always_negative_l1179_117988

noncomputable def f (x : ℝ) : ℝ := -x - x^3

theorem sum_f_always_negative
  (α β γ : ℝ)
  (h1 : α + β > 0)
  (h2 : β + γ > 0)
  (h3 : γ + α > 0) :
  f α + f β + f γ < 0 :=
by
  unfold f
  sorry

end sum_f_always_negative_l1179_117988


namespace number_of_sides_of_polygon_l1179_117985

theorem number_of_sides_of_polygon (n : ℕ) (h1 : (n * (n - 3)) = 340) : n = 20 :=
by
  sorry

end number_of_sides_of_polygon_l1179_117985


namespace determinant_scalar_multiplication_l1179_117967

theorem determinant_scalar_multiplication (x y z w : ℝ) (h : abs (x * w - y * z) = 10) :
  abs (3*x * 3*w - 3*y * 3*z) = 90 :=
by
  sorry

end determinant_scalar_multiplication_l1179_117967


namespace fraction_of_weight_kept_l1179_117923

-- Definitions of the conditions
def hunting_trips_per_month := 6
def months_in_season := 3
def deers_per_trip := 2
def weight_per_deer := 600
def weight_kept_per_year := 10800

-- Definition calculating total weight caught in the hunting season
def total_trips := hunting_trips_per_month * months_in_season
def weight_per_trip := deers_per_trip * weight_per_deer
def total_weight_caught := total_trips * weight_per_trip

-- The theorem to prove the fraction
theorem fraction_of_weight_kept : (weight_kept_per_year : ℚ) / (total_weight_caught : ℚ) = 1 / 2 := by
  -- Proof goes here
  sorry

end fraction_of_weight_kept_l1179_117923


namespace legendre_polynomial_expansion_l1179_117927

noncomputable def f (α β γ : ℝ) (θ : ℝ) : ℝ := α + β * Real.cos θ + γ * Real.cos θ ^ 2

noncomputable def P0 (x : ℝ) : ℝ := 1
noncomputable def P1 (x : ℝ) : ℝ := x
noncomputable def P2 (x : ℝ) : ℝ := (3 * x ^ 2 - 1) / 2

theorem legendre_polynomial_expansion (α β γ : ℝ) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
    f α β γ θ = (α + γ / 3) * P0 (Real.cos θ) + β * P1 (Real.cos θ) + (2 * γ / 3) * P2 (Real.cos θ) := by
  sorry

end legendre_polynomial_expansion_l1179_117927


namespace output_value_is_16_l1179_117987

def f (x : ℤ) : ℤ :=
  if x < 0 then (x + 1) * (x + 1) else (x - 1) * (x - 1)

theorem output_value_is_16 : f 5 = 16 := by
  sorry

end output_value_is_16_l1179_117987


namespace Chloe_total_score_l1179_117972

-- Definitions
def points_per_treasure : ℕ := 9
def treasures_first_level : ℕ := 6
def treasures_second_level : ℕ := 3

-- Statement of the theorem
theorem Chloe_total_score : (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level) = 81 := by
  sorry

end Chloe_total_score_l1179_117972


namespace complex_expression_evaluation_l1179_117901

theorem complex_expression_evaluation (i : ℂ) (h : i^2 = -1) : i^3 * (1 - i)^2 = -2 :=
by
  -- Placeholder for the actual proof which is skipped here
  sorry

end complex_expression_evaluation_l1179_117901


namespace problem_l1179_117966

theorem problem (x : ℝ) : (x^2 + 2 * x - 3 ≤ 0) → ¬(abs x > 3) :=
by sorry

end problem_l1179_117966


namespace best_trip_representation_l1179_117926

structure TripConditions where
  initial_walk_moderate : Prop
  main_road_speed_up : Prop
  bird_watching : Prop
  return_same_route : Prop
  coffee_stop : Prop
  final_walk_moderate : Prop

theorem best_trip_representation (conds : TripConditions) : 
  conds.initial_walk_moderate →
  conds.main_road_speed_up →
  conds.bird_watching →
  conds.return_same_route →
  conds.coffee_stop →
  conds.final_walk_moderate →
  True := 
by 
  intros 
  exact True.intro

end best_trip_representation_l1179_117926


namespace compute_expression_l1179_117925

-- Defining notation for the problem expression and answer simplification
theorem compute_expression : 
    9 * (2/3)^4 = 16/9 := by 
  sorry

end compute_expression_l1179_117925


namespace time_walking_each_day_l1179_117976

variable (days : Finset ℕ) (d1 : ℕ) (d2 : ℕ) (W : ℕ)

def time_spent_parking (days : Finset ℕ) : ℕ :=
  5 * days.card

def time_spent_metal_detector : ℕ :=
  2 * 30 + 3 * 10

def total_timespent (d1 d2 W : ℕ) : ℕ :=
  d1 + d2 + W

theorem time_walking_each_day (total_minutes : ℕ) (total_days : ℕ):
  total_timespent (time_spent_parking days) (time_spent_metal_detector) (total_minutes - time_spent_metal_detector - 5 * total_days)
  = total_minutes → W = 3 := by
  sorry

end time_walking_each_day_l1179_117976


namespace problem_solution_l1179_117951

theorem problem_solution :
  50000 - ((37500 / 62.35) ^ 2 + Real.sqrt 324) = -311752.222 :=
by
  sorry

end problem_solution_l1179_117951


namespace identify_conic_section_l1179_117993

theorem identify_conic_section (x y : ℝ) :
  (x + 7)^2 = (5 * y - 6)^2 + 125 →
  ∃ a b c d e f : ℝ, a * x^2 + b * y^2 + c * x + d * y + e * x * y + f = 0 ∧
  (a > 0) ∧ (b < 0) := sorry

end identify_conic_section_l1179_117993


namespace sum_of_roots_of_quadratic_l1179_117999

theorem sum_of_roots_of_quadratic : 
  ∀ x1 x2 : ℝ, 
  (3 * x1^2 - 6 * x1 - 7 = 0 ∧ 3 * x2^2 - 6 * x2 - 7 = 0) → 
  (x1 + x2 = 2) := by
  sorry

end sum_of_roots_of_quadratic_l1179_117999


namespace monotonicity_x_pow_2_over_3_l1179_117945

noncomputable def x_pow_2_over_3 (x : ℝ) : ℝ := x^(2/3)

theorem monotonicity_x_pow_2_over_3 : ∀ x y : ℝ, 0 < x → x < y → x_pow_2_over_3 x < x_pow_2_over_3 y :=
by
  intros x y hx hxy
  sorry

end monotonicity_x_pow_2_over_3_l1179_117945


namespace ratio_of_larger_to_smaller_l1179_117955

theorem ratio_of_larger_to_smaller (x y : ℝ) (h_pos : 0 < y) (h_ineq : x > y) (h_eq : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by 
  sorry

end ratio_of_larger_to_smaller_l1179_117955


namespace smaller_circle_radius_is_6_l1179_117933

-- Define the conditions of the problem
def large_circle_radius : ℝ := 2

def smaller_circles_touching_each_other (r : ℝ) : Prop :=
  let oa := large_circle_radius + r
  let ob := large_circle_radius + r
  let ab := 2 * r
  (oa^2 + ob^2 = ab^2)

def problem_statement : Prop :=
  ∃ r : ℝ, smaller_circles_touching_each_other r ∧ r = 6

theorem smaller_circle_radius_is_6 : problem_statement :=
sorry

end smaller_circle_radius_is_6_l1179_117933
