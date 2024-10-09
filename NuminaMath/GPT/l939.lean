import Mathlib

namespace problem_l939_93996

theorem problem (p q r : ℂ)
  (h1 : p + q + r = 0)
  (h2 : p * q + q * r + r * p = -2)
  (h3 : p * q * r = 2)
  (hp : p ^ 3 = 2 * p + 2)
  (hq : q ^ 3 = 2 * q + 2)
  (hr : r ^ 3 = 2 * r + 2) :
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = -18 := by
  sorry

end problem_l939_93996


namespace fraction_red_marbles_after_doubling_l939_93966

theorem fraction_red_marbles_after_doubling (x : ℕ) (h : x > 0) :
  let blue_fraction : ℚ := 3 / 5
  let red_fraction := 1 - blue_fraction
  let initial_blue_marbles := blue_fraction * x
  let initial_red_marbles := red_fraction * x
  let new_red_marbles := 2 * initial_red_marbles
  let new_total_marbles := initial_blue_marbles + new_red_marbles
  let new_red_fraction := new_red_marbles / new_total_marbles
  new_red_fraction = 4 / 7 :=
sorry

end fraction_red_marbles_after_doubling_l939_93966


namespace sum_squares_reciprocal_l939_93971

variable (x y : ℝ)

theorem sum_squares_reciprocal (h₁ : x + y = 12) (h₂ : x * y = 32) :
  (1/x)^2 + (1/y)^2 = 5/64 := by
  sorry

end sum_squares_reciprocal_l939_93971


namespace average_score_correct_l939_93915

-- Define the conditions
def simplified_scores : List Int := [10, -5, 0, 8, -3]
def base_score : Int := 90

-- Translate simplified score to actual score
def actual_score (s : Int) : Int :=
  base_score + s

-- Calculate the average of the actual scores
def average_score : Int :=
  (simplified_scores.map actual_score).sum / simplified_scores.length

-- The proof statement
theorem average_score_correct : average_score = 92 := 
by 
  -- Steps to compute the average score
  -- sorry is used since the proof steps are not required
  sorry

end average_score_correct_l939_93915


namespace g_sum_1_2_3_2_l939_93934

def g (a b : ℚ) : ℚ :=
  if a + b ≤ 4 then
    (a * b + a - 3) / (3 * a)
  else
    (a * b + b + 3) / (-3 * b)

theorem g_sum_1_2_3_2 : g 1 2 + g 3 2 = -11 / 6 :=
by sorry

end g_sum_1_2_3_2_l939_93934


namespace sum_of_coefficients_l939_93946

def polynomial (x y : ℕ) : ℕ := (x^2 - 3*x*y + y^2)^8

theorem sum_of_coefficients : polynomial 1 1 = 1 :=
sorry

end sum_of_coefficients_l939_93946


namespace integer_solutions_inequality_system_l939_93901

theorem integer_solutions_inequality_system :
  {x : ℤ | (x + 2 > 0) ∧ (2 * x - 1 ≤ 0)} = {-1, 0} := 
by
  -- proof goes here
  sorry

end integer_solutions_inequality_system_l939_93901


namespace abs_diff_two_numbers_l939_93988

theorem abs_diff_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : |x - y| = 10 := by
  sorry

end abs_diff_two_numbers_l939_93988


namespace find_missing_number_l939_93917

theorem find_missing_number (square boxplus boxtimes boxminus : ℕ) :
  square = 423 / 47 ∧
  1448 = 282 * boxminus + (boxminus * 10 + boxtimes) ∧
  423 * (boxplus / 3) = 282 →
  square = 9 ∧
  boxminus = 5 ∧
  boxtimes = 8 ∧
  boxplus = 2 ∧
  9 = 9 :=
by
  intro h
  sorry

end find_missing_number_l939_93917


namespace square_sides_product_l939_93904

theorem square_sides_product (a : ℝ) : 
  (∃ s : ℝ, s = 5 ∧ (a = -3 + s ∨ a = -3 - s)) → (a = 2 ∨ a = -8) → -8 * 2 = -16 :=
by
  intro _ _
  exact rfl

end square_sides_product_l939_93904


namespace line_l_passes_through_fixed_point_intersecting_lines_find_k_l939_93982

-- Define the lines
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0
def line_l1 (x y : ℝ) : Prop := 2 * x + 3 * y + 8 = 0
def line_l2 (x y : ℝ) : Prop := x - y - 1 = 0

-- 1. Prove line l passes through the point (-2, 1)
theorem line_l_passes_through_fixed_point (k : ℝ) :
  line_l k (-2) 1 :=
by sorry

-- 2. Given lines l, l1, and l2 intersect at a single point, find k
theorem intersecting_lines_find_k (k : ℝ) :
  (∃ x y : ℝ, line_l k x y ∧ line_l1 x y ∧ line_l2 x y) ↔ k = -3 :=
by sorry

end line_l_passes_through_fixed_point_intersecting_lines_find_k_l939_93982


namespace xy_product_l939_93968

noncomputable def f (t : ℝ) : ℝ := Real.sqrt (t^2 + 1) - t + 1

theorem xy_product (x y : ℝ)
  (h : (Real.sqrt (x^2 + 1) - x + 1) * (Real.sqrt (y^2 + 1) - y + 1) = 2) :
  x * y = 1 := by
  sorry

end xy_product_l939_93968


namespace fraction_product_108_l939_93939

theorem fraction_product_108 : (1/2 : ℚ) * (1/3) * (1/6) * 108 = 3 := by
  sorry

end fraction_product_108_l939_93939


namespace slower_speed_percentage_l939_93957

noncomputable def usual_speed_time : ℕ := 16  -- usual time in minutes
noncomputable def additional_time : ℕ := 24  -- additional time in minutes

theorem slower_speed_percentage (S S_slow : ℝ) (D : ℝ) 
  (h1 : D = S * usual_speed_time) 
  (h2 : D = S_slow * (usual_speed_time + additional_time)) : 
  (S_slow / S) * 100 = 40 :=
by 
  sorry

end slower_speed_percentage_l939_93957


namespace Debby_bought_bottles_l939_93942

theorem Debby_bought_bottles :
  (5 : ℕ) * (71 : ℕ) = 355 :=
by
  -- Math proof goes here
  sorry

end Debby_bought_bottles_l939_93942


namespace unique_very_set_on_line_l939_93984

def very_set (S : Finset (ℝ × ℝ)) : Prop :=
  ∀ X ∈ S, ∃ (r : ℝ), 
  ∀ Y ∈ S, Y ≠ X → ∃ Z ∈ S, Z ≠ X ∧ r * r = dist X Y * dist X Z

theorem unique_very_set_on_line (n : ℕ) (A B : ℝ × ℝ) (S1 S2 : Finset (ℝ × ℝ))
  (h : 2 ≤ n) (hA1 : A ∈ S1) (hB1 : B ∈ S1) (hA2 : A ∈ S2) (hB2 : B ∈ S2)
  (hS1 : S1.card = n) (hS2 : S2.card = n) (hV1 : very_set S1) (hV2 : very_set S2) :
  S1 = S2 := 
sorry

end unique_very_set_on_line_l939_93984


namespace cost_of_fencing_each_side_l939_93979

theorem cost_of_fencing_each_side (x : ℝ) (h : 4 * x = 316) : x = 79 :=
by
  sorry

end cost_of_fencing_each_side_l939_93979


namespace probability_all_selected_l939_93943

theorem probability_all_selected (P_Ram P_Ravi P_Ritu : ℚ) 
  (h1 : P_Ram = 3 / 7) 
  (h2 : P_Ravi = 1 / 5) 
  (h3 : P_Ritu = 2 / 9) : 
  P_Ram * P_Ravi * P_Ritu = 2 / 105 := 
by
  sorry

end probability_all_selected_l939_93943


namespace men_absent_l939_93990

theorem men_absent (x : ℕ) (H1 : 10 * 6 = 60) (H2 : (10 - x) * 10 = 60) : x = 4 :=
by
  sorry

end men_absent_l939_93990


namespace XiaoMing_selection_l939_93947

def final_positions (n : Nat) : List Nat :=
  if n <= 2 then
    List.range n
  else
    final_positions (n / 2) |>.filter (λ k => k % 2 = 0) |>.map (λ k => k / 2)

theorem XiaoMing_selection (n : Nat) (h : n = 32) : final_positions n = [16, 32] :=
  by
  sorry

end XiaoMing_selection_l939_93947


namespace point_not_in_second_quadrant_l939_93958

theorem point_not_in_second_quadrant (a : ℝ) :
  (∃ b : ℝ, b = 2 * a - 1) ∧ ¬(a < 0 ∧ (2 * a - 1 > 0)) := 
by sorry

end point_not_in_second_quadrant_l939_93958


namespace original_number_exists_l939_93967

theorem original_number_exists 
  (N: ℤ)
  (h1: ∃ (k: ℤ), N - 6 = 16 * k)
  (h2: ∀ (m: ℤ), (N - m) % 16 = 0 → m ≥ 6) : 
  N = 22 :=
sorry

end original_number_exists_l939_93967


namespace find_y_l939_93903

theorem find_y (y : ℝ) : 3 + 1 / (2 - y) = 2 * (1 / (2 - y)) → y = 5 / 3 := 
by
  sorry

end find_y_l939_93903


namespace find_a_l939_93916

-- Define the hyperbola equation and the asymptote conditions
def hyperbola (a : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / 9) = 1

def asymptote1 (x y : ℝ) : Prop := 3 * x + 2 * y = 0
def asymptote2 (x y : ℝ) : Prop := 3 * x - 2 * y = 0

-- Prove that if asymptote conditions hold, a = 2
theorem find_a (a : ℝ) (ha : a > 0) :
  (∀ x y, asymptote1 x y) ∧ (∀ x y, asymptote2 x y) → a = 2 :=
sorry

end find_a_l939_93916


namespace quadratic_inequality_a_value_l939_93999

theorem quadratic_inequality_a_value (a t : ℝ)
  (h_a1 : ∀ x : ℝ, t * x ^ 2 - 6 * x + t ^ 2 = 0 → (x = a ∨ x = 1))
  (h_t : t < 0) :
  a = -3 :=
by
  sorry

end quadratic_inequality_a_value_l939_93999


namespace boys_and_girls_l939_93998

theorem boys_and_girls (x y : ℕ) (h1 : x + y = 21) (h2 : 5 * x + 2 * y = 69) : x = 9 ∧ y = 12 :=
by 
  sorry

end boys_and_girls_l939_93998


namespace area_six_layers_l939_93986

theorem area_six_layers
  (A : ℕ → ℕ)
  (h1 : A 1 + A 2 + A 3 = 280)
  (h2 : A 2 = 54)
  (h3 : A 3 = 28)
  (h4 : A 4 = 14)
  (h5 : A 1 + 2 * A 2 + 3 * A 3 + 4 * A 4 + 6 * A 6 = 500)
  : A 6 = 9 := 
sorry

end area_six_layers_l939_93986


namespace solve_x_values_l939_93977

theorem solve_x_values : ∀ (x : ℝ), (x + 45 / (x - 4) = -10) ↔ (x = -1 ∨ x = -5) :=
by
  intro x
  sorry

end solve_x_values_l939_93977


namespace solve_system_of_inequalities_l939_93948

variable {R : Type*} [LinearOrderedField R]

theorem solve_system_of_inequalities (x1 x2 x3 x4 x5 : R)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0) (h5 : x5 > 0) :
  (x1^2 - x3^2) * (x2^2 - x3^2) ≤ 0 ∧ 
  (x3^2 - x1^2) * (x3^2 - x1^2) ≤ 0 ∧ 
  (x3^2 - x3 * x2) * (x1^2 - x3 * x2) ≤ 0 ∧ 
  (x1^2 - x1 * x3) * (x3^2 - x1 * x3) ≤ 0 ∧ 
  (x3^2 - x2 * x1) * (x1^2 - x2 * x1) ≤ 0 →
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5 :=
sorry

end solve_system_of_inequalities_l939_93948


namespace units_digit_sum_cubes_l939_93933

theorem units_digit_sum_cubes (n1 n2 : ℕ) 
  (h1 : n1 = 24) (h2 : n2 = 42) : 
  (n1 ^ 3 + n2 ^ 3) % 10 = 6 :=
by 
  -- substitution based on h1 and h2 can be done here.
  sorry

end units_digit_sum_cubes_l939_93933


namespace car_fuel_efficiency_in_city_l939_93973

theorem car_fuel_efficiency_in_city 
    (H C T : ℝ) 
    (h1 : H * T = 462) 
    (h2 : (H - 15) * T = 336) : 
    C = 40 :=
by 
    sorry

end car_fuel_efficiency_in_city_l939_93973


namespace original_class_size_l939_93921

theorem original_class_size
  (N : ℕ)
  (h1 : 40 * N = T)
  (h2 : T + 15 * 32 = 36 * (N + 15)) :
  N = 15 := by
  sorry

end original_class_size_l939_93921


namespace ratio_boysGradeA_girlsGradeB_l939_93920

variable (S G B : ℕ)

-- Given conditions
axiom h1 : (1 / 3 : ℚ) * G = (1 / 4 : ℚ) * S
axiom h2 : S = B + G

-- Definitions based on conditions
def boys_in_GradeA (B : ℕ) := (2 / 5 : ℚ) * B
def girls_in_GradeB (G : ℕ) := (3 / 5 : ℚ) * G

-- The proof goal
theorem ratio_boysGradeA_girlsGradeB (S G B : ℕ) (h1 : (1 / 3 : ℚ) * G = (1 / 4 : ℚ) * S) (h2 : S = B + G) :
    boys_in_GradeA B / girls_in_GradeB G = 2 / 9 :=
by
  sorry

end ratio_boysGradeA_girlsGradeB_l939_93920


namespace directors_dividends_correct_l939_93923

theorem directors_dividends_correct :
  let net_profit : ℝ := (1500000 - 674992) - 0.2 * (1500000 - 674992)
  let total_loan_payments : ℝ := 23914 * 12 - 74992
  let profit_for_dividends : ℝ := net_profit - total_loan_payments
  let dividend_per_share : ℝ := profit_for_dividends / 1000
  let total_dividends_director : ℝ := dividend_per_share * 550
  total_dividends_director = 246400.0 :=
by
  sorry

end directors_dividends_correct_l939_93923


namespace find_multiplier_l939_93960

theorem find_multiplier (n x : ℝ) (h1 : n = 1.0) (h2 : 3 * n - 1 = x * n) : x = 2 :=
by
  sorry

end find_multiplier_l939_93960


namespace delivery_newspapers_15_houses_l939_93961

-- State the problem using Lean 4 syntax

noncomputable def delivery_sequences (n : ℕ) : ℕ :=
  if h : n < 3 then 2^n
  else if n = 3 then 6
  else delivery_sequences (n-1) + delivery_sequences (n-2) + delivery_sequences (n-3)

theorem delivery_newspapers_15_houses :
  delivery_sequences 15 = 849 :=
sorry

end delivery_newspapers_15_houses_l939_93961


namespace min_value_is_8_plus_4_sqrt_3_l939_93919

noncomputable def min_value_of_expression (a b : ℝ) : ℝ :=
  2 / a + 1 / b

theorem min_value_is_8_plus_4_sqrt_3 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a + 2 * b = 1) :
  min_value_of_expression a b = 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_is_8_plus_4_sqrt_3_l939_93919


namespace cash_price_of_tablet_l939_93975

-- Define the conditions
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 4 * 40
def next_4_months_payment : ℕ := 4 * 35
def last_4_months_payment : ℕ := 4 * 30
def savings : ℕ := 70

-- Define the total installment payments
def total_installment_payments : ℕ := down_payment + first_4_months_payment + next_4_months_payment + last_4_months_payment

-- The statement to prove
theorem cash_price_of_tablet : total_installment_payments - savings = 450 := by
  -- proof goes here
  sorry

end cash_price_of_tablet_l939_93975


namespace chocolates_not_in_box_initially_l939_93927

theorem chocolates_not_in_box_initially 
  (total_chocolates : ℕ) 
  (chocolates_friend_brought : ℕ) 
  (initial_boxes : ℕ) 
  (additional_boxes : ℕ)
  (total_after_friend : ℕ)
  (chocolates_each_box : ℕ)
  (total_chocolates_initial : ℕ) :
  total_chocolates = 50 ∧ initial_boxes = 3 ∧ chocolates_friend_brought = 25 ∧ total_after_friend = 75 
  ∧ additional_boxes = 2 ∧ chocolates_each_box = 15 ∧ total_chocolates_initial = 50
  → (total_chocolates_initial - (initial_boxes * chocolates_each_box)) = 5 :=
by
  sorry

end chocolates_not_in_box_initially_l939_93927


namespace probability_red_or_white_correct_l939_93962

-- Define the conditions
def totalMarbles : ℕ := 30
def blueMarbles : ℕ := 5
def redMarbles : ℕ := 9
def whiteMarbles : ℕ := totalMarbles - (blueMarbles + redMarbles)

-- Define the calculated probability
def probabilityRedOrWhite : ℚ := (redMarbles + whiteMarbles) / totalMarbles

-- Verify the probability is equal to 5 / 6
theorem probability_red_or_white_correct :
  probabilityRedOrWhite = 5 / 6 := by
  sorry

end probability_red_or_white_correct_l939_93962


namespace johnny_needs_45_planks_l939_93941

theorem johnny_needs_45_planks
  (legs_per_table : ℕ)
  (planks_per_leg : ℕ)
  (surface_planks_per_table : ℕ)
  (number_of_tables : ℕ)
  (h1 : legs_per_table = 4)
  (h2 : planks_per_leg = 1)
  (h3 : surface_planks_per_table = 5)
  (h4 : number_of_tables = 5) :
  number_of_tables * (legs_per_table * planks_per_leg + surface_planks_per_table) = 45 :=
by
  sorry

end johnny_needs_45_planks_l939_93941


namespace perfect_squares_count_in_range_l939_93932

theorem perfect_squares_count_in_range :
  ∃ (n : ℕ), (
    (∀ (k : ℕ), (50 < k^2 ∧ k^2 < 500) → (8 ≤ k ∧ k ≤ 22)) ∧
    (15 = 22 - 8 + 1)
  ) := sorry

end perfect_squares_count_in_range_l939_93932


namespace triangle_angle_sum_l939_93945

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l939_93945


namespace reciprocal_of_5_is_1_div_5_l939_93910

-- Define the concept of reciprocal
def is_reciprocal (a b : ℚ) : Prop := a * b = 1

-- The problem statement: Prove that the reciprocal of 5 is 1/5
theorem reciprocal_of_5_is_1_div_5 : is_reciprocal 5 (1 / 5) :=
by
  sorry

end reciprocal_of_5_is_1_div_5_l939_93910


namespace pears_for_twenty_apples_l939_93911

-- Definitions based on given conditions
variables (a o p : ℕ) -- represent the number of apples, oranges, and pears respectively
variables (k1 k2 : ℕ) -- scaling factors 

-- Conditions as given
axiom ten_apples_five_oranges : 10 * a = 5 * o
axiom three_oranges_four_pears : 3 * o = 4 * p

-- Proving the number of pears Mia can buy for 20 apples
theorem pears_for_twenty_apples : 13 * p ≤ (20 * a) :=
by
  -- Actual proof would go here
  sorry

end pears_for_twenty_apples_l939_93911


namespace car_mpg_in_city_l939_93963

theorem car_mpg_in_city 
    (miles_per_tank_highway : Real)
    (miles_per_tank_city : Real)
    (mpg_difference : Real)
    : True := by
  let H := 21.05
  let T := 720 / H
  let C := H - 10
  have h1 : 720 = H * T := by
    sorry
  have h2 : 378 = C * T := by
    sorry
  exact True.intro

end car_mpg_in_city_l939_93963


namespace interest_calculation_years_l939_93929

theorem interest_calculation_years (P n : ℝ) (r : ℝ) (SI CI : ℝ)
  (h₁ : SI = P * r * n / 100)
  (h₂ : r = 5)
  (h₃ : SI = 50)
  (h₄ : CI = P * ((1 + r / 100)^n - 1))
  (h₅ : CI = 51.25) :
  n = 2 := by
  sorry

end interest_calculation_years_l939_93929


namespace a_friend_gcd_l939_93987

theorem a_friend_gcd (a b : ℕ) (d : ℕ) (hab : a * b = d * d) (hd : d = Nat.gcd a b) : ∃ k : ℕ, a * d = k * k := by
  sorry

end a_friend_gcd_l939_93987


namespace student_correct_answers_l939_93935

theorem student_correct_answers (C I : ℕ) (h₁ : C + I = 100) (h₂ : C - 2 * I = 61) : C = 87 :=
sorry

end student_correct_answers_l939_93935


namespace net_profit_calc_l939_93980

theorem net_profit_calc (purchase_price : ℕ) (overhead_percentage : ℝ) (markup : ℝ) 
  (h_pp : purchase_price = 48) (h_op : overhead_percentage = 0.10) (h_markup : markup = 35) :
  let overhead := overhead_percentage * purchase_price
  let net_profit := markup - overhead
  net_profit = 30.20 := by
    sorry

end net_profit_calc_l939_93980


namespace find_n_l939_93972

variable {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
variable (h : 1/a + 1/b + 1/c = 1/(a + b + c))

theorem find_n (n : ℤ) : (∃ k : ℕ, n = 2 * k - 1) → 
  (1 / a^n + 1 / b^n + 1 / c^n = 1 / (a^n + b^n + c^n)) :=
by
  sorry

end find_n_l939_93972


namespace x3_plus_y3_values_l939_93936

noncomputable def x_y_satisfy_eqns (x y : ℝ) : Prop :=
  y^2 - 3 = (x - 3)^3 ∧ x^2 - 3 = (y - 3)^2 ∧ x ≠ y

theorem x3_plus_y3_values (x y : ℝ) (h : x_y_satisfy_eqns x y) :
  x^3 + y^3 = 27 + 3 * Real.sqrt 3 ∨ x^3 + y^3 = 27 - 3 * Real.sqrt 3 :=
  sorry

end x3_plus_y3_values_l939_93936


namespace number_of_groups_of_oranges_l939_93937

-- Defining the conditions
def total_oranges : ℕ := 356
def oranges_per_group : ℕ := 2

-- The proof statement
theorem number_of_groups_of_oranges : total_oranges / oranges_per_group = 178 := 
by 
  sorry

end number_of_groups_of_oranges_l939_93937


namespace tank_ratio_two_l939_93953

variable (T1 : ℕ) (F1 : ℕ) (F2 : ℕ) (T2 : ℕ)

-- Assume the given conditions
axiom h1 : T1 = 48
axiom h2 : F1 = T1 / 3
axiom h3 : F1 - 1 = F2 + 3
axiom h4 : T2 = F2 * 2

-- The theorem to prove
theorem tank_ratio_two (h1 : T1 = 48) (h2 : F1 = T1 / 3) (h3 : F1 - 1 = F2 + 3) (h4 : T2 = F2 * 2) : T1 / T2 = 2 := by
  sorry

end tank_ratio_two_l939_93953


namespace ab_neither_sufficient_nor_necessary_l939_93985

theorem ab_neither_sufficient_nor_necessary (a b : ℝ) (h : a * b ≠ 0) :
  (¬ ((a * b > 1) → (a > 1 / b))) ∧ (¬ ((a > 1 / b) → (a * b > 1))) :=
by
  sorry

end ab_neither_sufficient_nor_necessary_l939_93985


namespace line_through_center_and_perpendicular_l939_93905

theorem line_through_center_and_perpendicular 
(C : ℝ × ℝ) 
(HC : ∀ (x y : ℝ), x ^ 2 + (y - 1) ^ 2 = 4 → C = (0, 1))
(l : ℝ → ℝ)
(Hl : ∀ x y : ℝ, 3 * x + 2 * y + 1 = 0 → y = l x)
: ∃ k b : ℝ, (∀ x : ℝ, y = k * x + b ↔ 2 * x - 3 * y + 3 = 0) :=
by 
  sorry

end line_through_center_and_perpendicular_l939_93905


namespace sequence_all_ones_l939_93954

theorem sequence_all_ones (k : ℕ) (n : ℕ → ℕ) (h_k : 2 ≤ k)
  (h1 : ∀ i, 1 ≤ i → i ≤ k → 1 ≤ n i) 
  (h2 : n 2 ∣ 2^(n 1) - 1) 
  (h3 : n 3 ∣ 2^(n 2) - 1) 
  (h4 : n 4 ∣ 2^(n 3) - 1)
  (h5 : ∀ i, 2 ≤ i → i < k → n (i + 1) ∣ 2^(n i) - 1)
  (h6 : n 1 ∣ 2^(n k) - 1) : 
  ∀ i, 1 ≤ i → i ≤ k → n i = 1 := 
by 
  sorry

end sequence_all_ones_l939_93954


namespace pete_total_miles_l939_93931

-- Definitions based on conditions
def flip_step_count : ℕ := 89999
def steps_full_cycle : ℕ := 90000
def total_flips : ℕ := 52
def end_year_reading : ℕ := 55555
def steps_per_mile : ℕ := 1900

-- Total steps Pete walked
def total_steps_pete_walked (flips : ℕ) (end_reading : ℕ) : ℕ :=
  flips * steps_full_cycle + end_reading

-- Total miles Pete walked
def total_miles_pete_walked (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

-- Given the parameters, closest number of miles Pete walked should be 2500
theorem pete_total_miles : total_miles_pete_walked (total_steps_pete_walked total_flips end_year_reading) steps_per_mile = 2500 :=
by
  sorry

end pete_total_miles_l939_93931


namespace min_value_expr_l939_93976

open Real

theorem min_value_expr(p q r : ℝ)(hp : 0 < p)(hq : 0 < q)(hr : 0 < r) :
  (5 * r / (3 * p + q) + 5 * p / (q + 3 * r) + 4 * q / (2 * p + 2 * r)) ≥ 5 / 2 :=
sorry

end min_value_expr_l939_93976


namespace solve_for_x_l939_93950

-- Define the new operation m ※ n
def operation (m n : ℤ) : ℤ :=
  if m ≥ 0 then m + n else m / n

-- Define the condition given in the problem
def condition (x : ℤ) : Prop :=
  operation (-9) (-x) = x

-- The main theorem to prove
theorem solve_for_x (x : ℤ) : condition x ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end solve_for_x_l939_93950


namespace average_last_4_matches_l939_93940

theorem average_last_4_matches (avg_10: ℝ) (avg_6: ℝ) (total_matches: ℕ) (first_matches: ℕ) :
  avg_10 = 38.9 → avg_6 = 42 → total_matches = 10 → first_matches = 6 → 
  (avg_10 * total_matches - avg_6 * first_matches) / (total_matches - first_matches) = 34.25 :=
by 
  intros h1 h2 h3 h4
  sorry

end average_last_4_matches_l939_93940


namespace number_of_puppies_sold_l939_93994

variables (P : ℕ) (p_0 : ℕ) (k_0 : ℕ) (r : ℕ) (k_s : ℕ)

theorem number_of_puppies_sold 
  (h1 : p_0 = 7) 
  (h2 : k_0 = 6) 
  (h3 : r = 8) 
  (h4 : k_s = 3) : 
  P = p_0 - (r - (k_0 - k_s)) :=
by sorry

end number_of_puppies_sold_l939_93994


namespace negation_of_p_l939_93902

def p := ∀ x : ℝ, x^2 ≥ 0

theorem negation_of_p : ¬p = (∃ x : ℝ, x^2 < 0) :=
  sorry

end negation_of_p_l939_93902


namespace percent_of_x_eq_to_y_l939_93989

variable {x y : ℝ}

theorem percent_of_x_eq_to_y (h: 0.5 * (x - y) = 0.3 * (x + y)) : y = 0.25 * x :=
by
  sorry

end percent_of_x_eq_to_y_l939_93989


namespace union_A_B_union_complement_A_B_l939_93970

open Set

-- Definitions for sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {3, 5}

-- Statement 1: Prove that A ∪ B = {1, 3, 5, 7}
theorem union_A_B : A ∪ B = {1, 3, 5, 7} := by
  sorry

-- Definition for complement of A in U
def complement_A_U : Set ℕ := {x ∈ U | x ∉ A}

-- Statement 2: Prove that (complement of A in U) ∪ B = {2, 3, 4, 5, 6}
theorem union_complement_A_B : complement_A_U ∪ B = {2, 3, 4, 5, 6} := by
  sorry

end union_A_B_union_complement_A_B_l939_93970


namespace fraction_relevant_quarters_l939_93969

-- Define the total number of quarters and the number of relevant quarters
def total_quarters : ℕ := 50
def relevant_quarters : ℕ := 10

-- Define the theorem that states the fraction of relevant quarters is 1/5
theorem fraction_relevant_quarters : (relevant_quarters : ℚ) / total_quarters = 1 / 5 := by
  sorry

end fraction_relevant_quarters_l939_93969


namespace vacation_days_proof_l939_93951

-- Define the conditions
def family_vacation (total_days rain_days clear_afternoons : ℕ) : Prop :=
  total_days = 18 ∧ rain_days = 13 ∧ clear_afternoons = 12

-- State the theorem to be proved
theorem vacation_days_proof : family_vacation 18 13 12 → 18 = 18 :=
by
  -- Skip the proof
  intro h
  sorry

end vacation_days_proof_l939_93951


namespace problem_equiv_proof_l939_93983

-- Define the universe set U
def U : Set ℝ := Set.univ

-- Define the set A based on the given condition
def A : Set ℝ := { x | x^2 + x - 2 ≤ 0 }

-- Define the set B based on the given condition
def B : Set ℝ := { y | ∃ x : ℝ, x ∈ A ∧ y = Real.log (x + 3) / Real.log 2 }

-- Define the complement of B in the universal set U
def complement_B : Set ℝ := { y | y < 0 ∨ y ≥ 2 }

-- Define the set C that is the intersection of A and complement of B
def C : Set ℝ := A ∩ complement_B

-- State the theorem we need to prove
theorem problem_equiv_proof : C = { x | -2 ≤ x ∧ x < 0 } :=
sorry

end problem_equiv_proof_l939_93983


namespace common_measure_angle_l939_93964

theorem common_measure_angle (α β : ℝ) (m n : ℕ) (h : α = β * (m / n)) : α / m = β / n :=
by 
  sorry

end common_measure_angle_l939_93964


namespace minimum_y_squared_l939_93926

theorem minimum_y_squared :
  let consecutive_sum (x : ℤ) := (x-5)^2 + (x-4)^2 + (x-3)^2 + (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2
  ∃ y : ℤ, y^2 = 11 * (1^2 + 10) ∧ ∀ z : ℤ, z^2 = 11 * consecutive_sum z → y^2 ≤ z^2 := by
sorry

end minimum_y_squared_l939_93926


namespace pqrs_product_l939_93912

noncomputable def P : ℝ := Real.sqrt 2012 + Real.sqrt 2013
noncomputable def Q : ℝ := -Real.sqrt 2012 - Real.sqrt 2013
noncomputable def R : ℝ := Real.sqrt 2012 - Real.sqrt 2013
noncomputable def S : ℝ := Real.sqrt 2013 - Real.sqrt 2012

theorem pqrs_product : P * Q * R * S = 1 := 
by 
  sorry

end pqrs_product_l939_93912


namespace inscribed_quadrilateral_inradius_l939_93909

noncomputable def calculate_inradius (a b c d: ℝ) (A: ℝ) : ℝ := (A / ((a + c + b + d) / 2))

theorem inscribed_quadrilateral_inradius {a b c d: ℝ} (h1: a + c = 10) (h2: b + d = 10) (h3: a + b + c + d = 20) (hA: 12 = 12):
  calculate_inradius a b c d 12 = 6 / 5 :=
by
  sorry

end inscribed_quadrilateral_inradius_l939_93909


namespace typing_time_together_l939_93925

theorem typing_time_together 
  (jonathan_time : ℝ)
  (susan_time : ℝ)
  (jack_time : ℝ)
  (document_pages : ℝ)
  (combined_time : ℝ) :
  jonathan_time = 40 →
  susan_time = 30 →
  jack_time = 24 →
  document_pages = 10 →
  combined_time = document_pages / ((document_pages / jonathan_time) + (document_pages / susan_time) + (document_pages / jack_time)) →
  combined_time = 10 :=
by sorry

end typing_time_together_l939_93925


namespace regression_is_appropriate_l939_93956

-- Definitions for the different analysis methods
inductive AnalysisMethod
| ResidualAnalysis : AnalysisMethod
| RegressionAnalysis : AnalysisMethod
| IsoplethBarChart : AnalysisMethod
| IndependenceTest : AnalysisMethod

-- Relating height and weight with an appropriate analysis method
def appropriateMethod (method : AnalysisMethod) : Prop :=
  method = AnalysisMethod.RegressionAnalysis

-- Stating the theorem that regression analysis is the appropriate method
theorem regression_is_appropriate : appropriateMethod AnalysisMethod.RegressionAnalysis :=
by sorry

end regression_is_appropriate_l939_93956


namespace multiple_of_bees_l939_93900

theorem multiple_of_bees (b₁ b₂ : ℕ) (h₁ : b₁ = 144) (h₂ : b₂ = 432) : b₂ / b₁ = 3 := 
by
  sorry

end multiple_of_bees_l939_93900


namespace handshake_count_l939_93995

theorem handshake_count (n_total n_group1 n_group2 : ℕ) 
  (h_total : n_total = 40) (h_group1 : n_group1 = 25) (h_group2 : n_group2 = 15) 
  (h_sum : n_group1 + n_group2 = n_total) : 
  (15 * 39) / 2 = 292 := 
by sorry

end handshake_count_l939_93995


namespace remainder_of_large_number_l939_93993

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l939_93993


namespace solve_for_x_l939_93908

theorem solve_for_x (x : ℝ) (h : x / 6 = 15 / 10) : x = 9 :=
by
  sorry

end solve_for_x_l939_93908


namespace find_radius_of_sphere_l939_93965

noncomputable def radius_of_sphere (R : ℝ) : Prop :=
  ∃ a b c : ℝ, 
  (R = |a| ∧ R = |b| ∧ R = |c|) ∧ 
  ((3 - R)^2 + (2 - R)^2 + (1 - R)^2 = R^2)

theorem find_radius_of_sphere : radius_of_sphere (3 + Real.sqrt 2) ∨ radius_of_sphere (3 - Real.sqrt 2) :=
sorry

end find_radius_of_sphere_l939_93965


namespace simplify_expression_l939_93978

theorem simplify_expression (x : ℝ) : (3 * x) ^ 5 - (4 * x) * (x ^ 4) = 239 * x ^ 5 := 
by
  sorry

end simplify_expression_l939_93978


namespace num_12_digit_with_consecutive_ones_l939_93955

theorem num_12_digit_with_consecutive_ones :
  let total := 3^12
  let F12 := 985
  total - F12 = 530456 :=
by
  let total := 3^12
  let F12 := 985
  have h : total - F12 = 530456
  sorry
  exact h

end num_12_digit_with_consecutive_ones_l939_93955


namespace find_total_roses_l939_93974

open Nat

theorem find_total_roses 
  (a : ℕ)
  (h1 : 300 ≤ a)
  (h2 : a ≤ 400)
  (h3 : a % 21 = 13)
  (h4 : a % 15 = 7) : 
  a = 307 := 
sorry

end find_total_roses_l939_93974


namespace train_speed_l939_93949

theorem train_speed (D T : ℝ) (h1 : D = 160) (h2 : T = 16) : D / T = 10 :=
by 
  -- given D = 160 and T = 16, we need to prove D / T = 10
  sorry

end train_speed_l939_93949


namespace problem_l939_93991

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 2}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def compN : Set ℝ := {x | x < -1 ∨ 1 < x}
def intersection : Set ℝ := {x | x < -1 ∨ (1 < x ∧ x ≤ 2)}

theorem problem (x : ℝ) : x ∈ (M ∩ compN) ↔ x ∈ intersection := by
  sorry

end problem_l939_93991


namespace alec_votes_l939_93928

variable (students totalVotes goalVotes neededVotes : ℕ)

theorem alec_votes (h1 : students = 60)
                   (h2 : goalVotes = 3 * students / 4)
                   (h3 : totalVotes = students / 2 + 5 + (students - (students / 2 + 5)) / 5)
                   (h4 : neededVotes = goalVotes - totalVotes) :
                   neededVotes = 5 :=
by sorry

end alec_votes_l939_93928


namespace pages_per_sheet_is_one_l939_93907

-- Definition of conditions
def stories_per_week : Nat := 3
def pages_per_story : Nat := 50
def num_weeks : Nat := 12
def reams_bought : Nat := 3
def sheets_per_ream : Nat := 500

-- Calculate total pages written over num_weeks (short stories only)
def total_pages : Nat := stories_per_week * pages_per_story * num_weeks

-- Calculate total sheets available
def total_sheets : Nat := reams_bought * sheets_per_ream

-- Calculate pages per sheet, rounding to nearest whole number
def pages_per_sheet : Nat := (total_pages / total_sheets)

-- The main statement to prove
theorem pages_per_sheet_is_one : pages_per_sheet = 1 :=
by
  sorry

end pages_per_sheet_is_one_l939_93907


namespace probability_correct_l939_93913

-- Definition for the total number of ways to select topics
def total_ways : ℕ := 6 * 6

-- Definition for the number of ways two students select different topics
def different_topics_ways : ℕ := 6 * 5

-- Definition for the probability of selecting different topics
def probability_different_topics : ℚ := different_topics_ways / total_ways

-- The statement to be proved in Lean
theorem probability_correct :
  probability_different_topics = 5 / 6 := 
sorry

end probability_correct_l939_93913


namespace log_sum_correct_l939_93906

noncomputable def log_sum : ℝ := 
  Real.log 8 / Real.log 10 + 
  3 * Real.log 4 / Real.log 10 + 
  4 * Real.log 2 / Real.log 10 +
  2 * Real.log 5 / Real.log 10 +
  5 * Real.log 25 / Real.log 10

theorem log_sum_correct : abs (log_sum - 12.301) < 0.001 :=
by sorry

end log_sum_correct_l939_93906


namespace division_addition_correct_l939_93944

-- Define a function that performs the arithmetic operations described
def calculateResult : ℕ :=
  let division := 12 * 4 -- dividing 12 by 1/4 is the same as multiplying by 4
  division + 5 -- then add 5 to the result

-- The theorem statement to prove
theorem division_addition_correct : calculateResult = 53 := by
  sorry

end division_addition_correct_l939_93944


namespace roots_of_transformed_quadratic_l939_93997

variable {a b c : ℝ}

theorem roots_of_transformed_quadratic
    (h₁: a ≠ 0)
    (h₂: ∀ x, a * (x - 1)^2 - 1 = ax^2 + bx + c - 1)
    (h₃: ax^2 + bx + c = -1) :
    (x = 1) ∧ (x = 1) := 
  sorry

end roots_of_transformed_quadratic_l939_93997


namespace change_received_l939_93924

def totalCostBeforeDiscount : ℝ :=
  5.75 + 2.50 + 3.25 + 3.75 + 4.20

def discount : ℝ :=
  (3.75 + 4.20) * 0.10

def totalCostAfterDiscount : ℝ :=
  totalCostBeforeDiscount - discount

def salesTax : ℝ :=
  totalCostAfterDiscount * 0.06

def finalTotalCost : ℝ :=
  totalCostAfterDiscount + salesTax

def amountPaid : ℝ :=
  50.00

def change : ℝ :=
  amountPaid - finalTotalCost

theorem change_received (h : change = 30.34) : change = 30.34 := by
  sorry

end change_received_l939_93924


namespace tangent_identity_l939_93959

theorem tangent_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2)
  = ((3 * x - x^3) / (1 - 3 * x^2)) * ((3 * y - y^3) / (1 - 3 * y^2)) * ((3 * z - z^3) / (1 - 3 * z^2)) :=
sorry

end tangent_identity_l939_93959


namespace time_before_Car_Y_started_in_minutes_l939_93922

noncomputable def timeBeforeCarYStarted (speedX speedY distanceX : ℝ) : ℝ :=
  let t := distanceX / speedX
  (speedY * t - distanceX) / speedX

theorem time_before_Car_Y_started_in_minutes 
  (speedX speedY distanceX : ℝ)
  (h_speedX : speedX = 35)
  (h_speedY : speedY = 70)
  (h_distanceX : distanceX = 42) : 
  (timeBeforeCarYStarted speedX speedY distanceX) * 60 = 72 :=
by
  sorry

end time_before_Car_Y_started_in_minutes_l939_93922


namespace sum_seven_terms_l939_93952

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) - a n = d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

-- Given condition
def given_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 = 42

-- Proof statement
theorem sum_seven_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : sum_of_first_n_terms a S) 
  (h_cond : given_condition a) : 
  S 7 = 98 := 
sorry

end sum_seven_terms_l939_93952


namespace blue_pieces_correct_l939_93918

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_pieces_correct : blue_pieces = 3264 := by
  sorry

end blue_pieces_correct_l939_93918


namespace find_initial_divisor_l939_93938

theorem find_initial_divisor (N D : ℤ) (h1 : N = 2 * D) (h2 : N % 4 = 2) : D = 3 :=
by
  sorry

end find_initial_divisor_l939_93938


namespace angle_bisector_theorem_l939_93930

noncomputable def ratio_of_segments (x y z p q : ℝ) :=
  q / x = y / (y + x)

theorem angle_bisector_theorem (x y z p q : ℝ) (h1 : p / x = q / y)
  (h2 : p + q = z) : ratio_of_segments x y z p q :=
by
  sorry

end angle_bisector_theorem_l939_93930


namespace intersection_is_empty_l939_93914

-- Define sets M and N
def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {0, 3, 4}

-- Define isolated elements for a set
def is_isolated (A : Set ℕ) (x : ℕ) : Prop :=
  x ∈ A ∧ (x - 1 ∉ A) ∧ (x + 1 ∉ A)

-- Define isolated sets
def isolated_set (A : Set ℕ) : Set ℕ :=
  {x | is_isolated A x}

-- Define isolated sets for M and N
def M' := isolated_set M
def N' := isolated_set N

-- The intersection of the isolated sets
theorem intersection_is_empty : M' ∩ N' = ∅ := 
  sorry

end intersection_is_empty_l939_93914


namespace reflection_line_slope_l939_93992

/-- Given two points (1, -2) and (7, 4), and the reflection line y = mx + b. 
    The image of (1, -2) under the reflection is (7, 4). Prove m + b = 4. -/
theorem reflection_line_slope (m b : ℝ)
    (h1: (∀ (x1 y1 x2 y2: ℝ), 
        (x1, y1) = (1, -2) → 
        (x2, y2) = (7, 4) → 
        (y2 - y1) / (x2 - x1) = 1)) 
    (h2: ∀ (x1 y1 x2 y2: ℝ),
        (x1, y1) = (1, -2) → 
        (x2, y2) = (7, 4) →
        (x1 + x2) / 2 = 4 ∧ (y1 + y2) / 2 = 1) 
    (h3: y = mx + b → m = -1 → (4, 1).1 = 4 ∧ (4, 1).2 = 1 → b = 5) : 
    m + b = 4 := by 
  -- No Proof Required
  sorry

end reflection_line_slope_l939_93992


namespace problem_statement_l939_93981

variables {Line Plane : Type}
variables {m n : Line} {alpha beta : Plane}

-- Define parallel and perpendicular relations
def parallel (l1 l2 : Line) : Prop := sorry
def perp (l : Line) (p : Plane) : Prop := sorry

-- Define that m and n are different lines
axiom diff_lines (m n : Line) : m ≠ n 

-- Define that alpha and beta are different planes
axiom diff_planes (alpha beta : Plane) : alpha ≠ beta

-- Statement to prove: If m ∥ n and m ⟂ α, then n ⟂ α
theorem problem_statement (h1 : parallel m n) (h2 : perp m alpha) : perp n alpha := 
sorry

end problem_statement_l939_93981
