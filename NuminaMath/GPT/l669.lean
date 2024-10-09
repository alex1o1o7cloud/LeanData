import Mathlib

namespace units_digit_42_3_plus_27_2_l669_66913

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_42_3_plus_27_2 : units_digit (42^3 + 27^2) = 7 :=
by
  sorry

end units_digit_42_3_plus_27_2_l669_66913


namespace pascal_row_12_sum_pascal_row_12_middle_l669_66930

open Nat

/-- Definition of the sum of all numbers in a given row of Pascal's Triangle -/
def pascal_sum (n : ℕ) : ℕ :=
  2^n

/-- Definition of the binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Pascal Triangle Row 12 sum -/
theorem pascal_row_12_sum : pascal_sum 12 = 4096 :=
by
  sorry

/-- Pascal Triangle Row 12 middle number -/
theorem pascal_row_12_middle : binomial 12 6 = 924 :=
by
  sorry

end pascal_row_12_sum_pascal_row_12_middle_l669_66930


namespace race_track_cost_l669_66927

def toy_car_cost : ℝ := 0.95
def num_toy_cars : ℕ := 4
def total_money : ℝ := 17.80
def money_left : ℝ := 8.00

theorem race_track_cost :
  total_money - num_toy_cars * toy_car_cost - money_left = 6.00 :=
by
  sorry

end race_track_cost_l669_66927


namespace find_number_l669_66981

variable (x : ℝ)

theorem find_number (hx : 5100 - (102 / x) = 5095) : x = 20.4 := 
by
  sorry

end find_number_l669_66981


namespace simplify_fraction_l669_66973

theorem simplify_fraction :
  (4 * 6) / (12 * 15) * (5 * 12 * 15^2) / (2 * 6 * 5) = 2.5 := by
  sorry

end simplify_fraction_l669_66973


namespace fraction_addition_l669_66961

theorem fraction_addition :
  (3 / 4) / (5 / 8) + (1 / 2) = 17 / 10 :=
by
  sorry

end fraction_addition_l669_66961


namespace exists_a_b_divisible_l669_66942

theorem exists_a_b_divisible (n : ℕ) (hn : 0 < n) : 
  ∃ a b : ℤ, (4 * a^2 + 9 * b^2 - 1) % n = 0 := 
sorry

end exists_a_b_divisible_l669_66942


namespace nat_no_solution_x3_plus_5y_eq_y3_plus_5x_positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x_l669_66965

theorem nat_no_solution_x3_plus_5y_eq_y3_plus_5x (x y : ℕ) (h₁ : x ≠ y) : 
  x^3 + 5 * y ≠ y^3 + 5 * x :=
sorry

theorem positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + 5 * y = y^3 + 5 * x :=
sorry

end nat_no_solution_x3_plus_5y_eq_y3_plus_5x_positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x_l669_66965


namespace tan_cot_theta_l669_66922

theorem tan_cot_theta 
  (θ : ℝ) 
  (h1 : Real.sin θ + Real.cos θ = (Real.sqrt 2) / 3) 
  (h2 : Real.pi / 2 < θ ∧ θ < Real.pi) : 
  Real.tan θ - (1 / Real.tan θ) = - (8 * Real.sqrt 2) / 7 := 
sorry

end tan_cot_theta_l669_66922


namespace part1_part2_l669_66938

variable (a : ℝ)

-- Proposition A
def propA (a : ℝ) := ∀ x : ℝ, ¬ (x^2 + (2*a-1)*x + a^2 ≤ 0)

-- Proposition B
def propB (a : ℝ) := 0 < a^2 - 1 ∧ a^2 - 1 < 1

theorem part1 (ha : propA a ∨ propB a) : 
  (-Real.sqrt 2 < a ∧ a < -1) ∨ (a > 1/4) :=
  sorry

theorem part2 (ha : ¬ propA a) (hb : propB a) : 
  (-Real.sqrt 2 < a ∧ a < -1) → (a^3 + 1 < a^2 + a) :=
  sorry

end part1_part2_l669_66938


namespace minimum_value_of_x_plus_y_l669_66987

noncomputable def minValueSatisfies (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y + x * y = 2 → x + y ≥ 2 * Real.sqrt 3 - 2

theorem minimum_value_of_x_plus_y (x y : ℝ) : minValueSatisfies x y :=
by sorry

end minimum_value_of_x_plus_y_l669_66987


namespace roots_poly_eq_l669_66985

theorem roots_poly_eq (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : d = 0) (root1_eq : 64 * a + 16 * b + 4 * c = 0) (root2_eq : -27 * a + 9 * b - 3 * c = 0) :
  (b + c) / a = -13 :=
by {
  sorry
}

end roots_poly_eq_l669_66985


namespace correct_answer_is_ln_abs_l669_66992

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, (0 < x ∧ x < y) → f x ≤ f y

theorem correct_answer_is_ln_abs :
  is_even_function (fun x => Real.log (abs x)) ∧ is_monotonically_increasing_on_pos (fun x => Real.log (abs x)) ∧
  ¬ is_even_function (fun x => x^3) ∧
  ¬ is_monotonically_increasing_on_pos (fun x => Real.cos x) :=
by
  sorry

end correct_answer_is_ln_abs_l669_66992


namespace replace_with_30_digit_nat_number_l669_66976

noncomputable def is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000

theorem replace_with_30_digit_nat_number (a : Fin 10 → ℕ) (h : ∀ i, is_three_digit (a i)) :
  ∃ b : ℕ, (b < 10^30 ∧ ∃ x : ℤ, (a 9) * x^9 + (a 8) * x^8 + (a 7) * x^7 + (a 6) * x^6 + (a 5) * x^5 + 
           (a 4) * x^4 + (a 3) * x^3 + (a 2) * x^2 + (a 1) * x + (a 0) = b) :=
by
  sorry

end replace_with_30_digit_nat_number_l669_66976


namespace exists_invisible_square_l669_66914

def invisible (p q : ℤ) : Prop := Int.gcd p q > 1

theorem exists_invisible_square (n : ℤ) (h : 0 < n) : 
  ∃ (a b : ℤ), ∀ i j : ℤ, (0 ≤ i) ∧ (i < n) ∧ (0 ≤ j) ∧ (j < n) → invisible (a + i) (b + j) :=
by {
  sorry
}

end exists_invisible_square_l669_66914


namespace painted_cells_l669_66969

theorem painted_cells (k l : ℕ) (h : k * l = 74) :
    (2 * k + 1) * (2 * l + 1) - k * l = 301 ∨ 
    (2 * k + 1) * (2 * l + 1) - k * l = 373 :=
sorry

end painted_cells_l669_66969


namespace range_x_satisfies_inequality_l669_66957

theorem range_x_satisfies_inequality (x : ℝ) : (x^2 < |x|) ↔ (-1 < x ∧ x < 1 ∧ x ≠ 0) :=
sorry

end range_x_satisfies_inequality_l669_66957


namespace trader_sells_cloth_l669_66921

theorem trader_sells_cloth
  (total_SP : ℝ := 4950)
  (profit_per_meter : ℝ := 15)
  (cost_price_per_meter : ℝ := 51)
  (SP_per_meter : ℝ := cost_price_per_meter + profit_per_meter)
  (x : ℝ := total_SP / SP_per_meter) :
  x = 75 :=
by
  sorry

end trader_sells_cloth_l669_66921


namespace average_speed_l669_66902

theorem average_speed (D T : ℝ) (h1 : D = 100) (h2 : T = 6) : (D / T) = 50 / 3 := by
  sorry

end average_speed_l669_66902


namespace angle_A_measure_l669_66966

theorem angle_A_measure 
  (B : ℝ) 
  (angle_in_smaller_triangle : ℝ) 
  (sum_of_triangle_angles_eq_180 : ∀ (x y z : ℝ), x + y + z = 180)
  (C : ℝ) 
  (angle_pair_linear : ∀ (x y : ℝ), x + y = 180) 
  (A : ℝ) 
  (C_eq_180_minus_B : C = 180 - B) 
  (A_eq_180_minus_angle_in_smaller_triangle_minus_C : 
    A = 180 - angle_in_smaller_triangle - C) :
  A = 70 :=
by
  sorry

end angle_A_measure_l669_66966


namespace tan_beta_minus_pi_over_4_l669_66977

theorem tan_beta_minus_pi_over_4 (α β : ℝ) 
  (h1 : Real.tan (α + β) = 1/2) 
  (h2 : Real.tan (α + π/4) = -1/3) : 
  Real.tan (β - π/4) = 1 := 
sorry

end tan_beta_minus_pi_over_4_l669_66977


namespace shortest_minor_arc_line_equation_l669_66947

noncomputable def pointM : (ℝ × ℝ) := (1, -2)
noncomputable def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

theorem shortest_minor_arc_line_equation :
  (∀ x y : ℝ, (x + 2 * y + 3 = 0) ↔ 
  ((x = 1 ∧ y = -2) ∨ ∃ (k_l : ℝ), (k_l * (2) = -1) ∧ (y + 2 = -k_l * (x - 1)))) :=
sorry

end shortest_minor_arc_line_equation_l669_66947


namespace square_area_l669_66955

theorem square_area {d : ℝ} (h : d = 12 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 144 ∧ ( ∃ s : ℝ, s = d / Real.sqrt 2 ∧ A = s^2 ) :=
by
  sorry

end square_area_l669_66955


namespace triangle_area_l669_66962

def point := ℝ × ℝ

def A : point := (2, -3)
def B : point := (8, 1)
def C : point := (2, 3)

def area_triangle (A B C : point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area : area_triangle A B C = 18 :=
  sorry

end triangle_area_l669_66962


namespace brick_height_calculation_l669_66972

theorem brick_height_calculation :
  ∀ (num_bricks : ℕ) (brick_length brick_width brick_height : ℝ)
    (wall_length wall_height wall_width : ℝ),
    num_bricks = 1600 →
    brick_length = 100 →
    brick_width = 11.25 →
    wall_length = 800 →
    wall_height = 600 →
    wall_width = 22.5 →
    wall_length * wall_height * wall_width = 
    num_bricks * brick_length * brick_width * brick_height →
    brick_height = 60 :=
by
  sorry

end brick_height_calculation_l669_66972


namespace coefficient_of_x4_l669_66982

theorem coefficient_of_x4 (a : ℝ) (h : 15 * a^4 = 240) : a = 2 ∨ a = -2 := 
sorry

end coefficient_of_x4_l669_66982


namespace pine_tree_next_one_in_between_l669_66967

theorem pine_tree_next_one_in_between (n : ℕ) (p s : ℕ) (trees : n = 2019) (pines : p = 1009) (spruces : s = 1010)
    (equal_intervals : true) : 
    ∃ (i : ℕ), (i < n) ∧ ((i + 1) % n ∈ {j | j < p}) ∧ ((i + 3) % n ∈ {j | j < p}) :=
  sorry

end pine_tree_next_one_in_between_l669_66967


namespace parallel_lines_value_of_m_l669_66901

theorem parallel_lines_value_of_m (m : ℝ) 
  (h1 : ∀ x y : ℝ, x + m * y - 2 = 0 = (2 * x + (1 - m) * y + 2 = 0)) : 
  m = 1 / 3 :=
by {
  sorry
}

end parallel_lines_value_of_m_l669_66901


namespace ab_div_c_eq_2_l669_66912

variable (a b c : ℝ)

def condition1 (a b c : ℝ) : Prop := a * b - c = 3
def condition2 (a b c : ℝ) : Prop := a * b * c = 18

theorem ab_div_c_eq_2 (h1 : condition1 a b c) (h2 : condition2 a b c) : a * b / c = 2 :=
by sorry

end ab_div_c_eq_2_l669_66912


namespace equilateral_triangle_side_length_l669_66963

theorem equilateral_triangle_side_length (perimeter : ℕ) (h_perimeter : perimeter = 69) : 
  ∃ (side_length : ℕ), side_length = perimeter / 3 := 
by
  sorry

end equilateral_triangle_side_length_l669_66963


namespace bananas_to_oranges_l669_66908

theorem bananas_to_oranges (B A O : ℕ) 
    (h1 : 4 * B = 3 * A) 
    (h2 : 7 * A = 5 * O) : 
    28 * B = 15 * O :=
by
  sorry

end bananas_to_oranges_l669_66908


namespace magic_square_sum_l669_66954

theorem magic_square_sum (a b c d e f S : ℕ) 
  (h1 : 30 + b + 22 = S) 
  (h2 : 19 + c + d = S) 
  (h3 : a + 28 + f = S)
  (h4 : 30 + 19 + a = S)
  (h5 : b + c + 28 = S)
  (h6 : 22 + d + f = S)
  (h7 : 30 + c + f = S)
  (h8 : 22 + c + a = S)
  (h9 : e = b) :
  d + e = 54 := 
by 
  sorry

end magic_square_sum_l669_66954


namespace integer_values_of_x_for_positive_star_l669_66926

-- Definition of the operation star
def star (a b : ℕ) : ℚ := (a^2 : ℕ) / b

-- Problem statement
theorem integer_values_of_x_for_positive_star :
  ∃ (count : ℕ), count = 9 ∧ (∀ x : ℕ, (10^2 % x = 0) → (∃ n : ℕ, star 10 x = n)) :=
sorry

end integer_values_of_x_for_positive_star_l669_66926


namespace opposite_of_neg3_l669_66934

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
  sorry

end opposite_of_neg3_l669_66934


namespace rectangle_coloring_problem_l669_66971

theorem rectangle_coloring_problem :
  let n := 3
  let m := 4
  ∃ n, ∃ m, n = 3 ∧ m = 4 := sorry

end rectangle_coloring_problem_l669_66971


namespace find_remainder_l669_66959

theorem find_remainder : 
    ∃ (d q r : ℕ), 472 = d * q + r ∧ 427 = d * (q - 5) + r ∧ r = 4 :=
by
  sorry

end find_remainder_l669_66959


namespace find_f_20_l669_66997

theorem find_f_20 (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = (1/2) * f (x + 2))
  (h2 : f 2 = 1) :
  f 20 = 512 :=
sorry

end find_f_20_l669_66997


namespace graph_passes_through_point_l669_66928

theorem graph_passes_through_point (a : ℝ) (ha : 0 < a) (ha_ne_one : a ≠ 1) :
  let f := fun x : ℝ => a^(x - 3) + 2
  f 3 = 3 := by
  sorry

end graph_passes_through_point_l669_66928


namespace dan_money_left_l669_66998

def initial_money : ℝ := 50.00
def candy_bar_price : ℝ := 1.75
def candy_bar_count : ℕ := 3
def gum_price : ℝ := 0.85
def soda_price : ℝ := 2.25
def sales_tax_rate : ℝ := 0.08

theorem dan_money_left : 
  initial_money - (candy_bar_count * candy_bar_price + gum_price + soda_price) * (1 + sales_tax_rate) = 40.98 :=
by
  sorry

end dan_money_left_l669_66998


namespace equation_result_l669_66948

theorem equation_result : 
  ∀ (n : ℝ), n = 5.0 → (4 * n + 7 * n) = 55.0 :=
by
  intro n h
  rw [h]
  norm_num

end equation_result_l669_66948


namespace soap_box_missing_dimension_l669_66983

theorem soap_box_missing_dimension
  (x : ℕ) -- The missing dimension of the soap box
  (Volume_carton : ℕ := 25 * 48 * 60)
  (Volume_soap_box : ℕ := 8 * x * 5)
  (Max_soap_boxes : ℕ := 300)
  (condition : Max_soap_boxes * Volume_soap_box ≤ Volume_carton) :
  x ≤ 6 := by
sorry

end soap_box_missing_dimension_l669_66983


namespace comparison_theorem_l669_66984

open Real

noncomputable def comparison (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : Prop :=
  let a := log (sin x)
  let b := sin x
  let c := exp (sin x)
  a < b ∧ b < c

theorem comparison_theorem (x : ℝ) (h : 0 < x ∧ x < π / 2) : comparison x h.1 h.2 :=
by { sorry }

end comparison_theorem_l669_66984


namespace duration_of_time_l669_66915

variable (A B C : String)
variable {a1 : A = "Get up at 6:30"}
variable {b1 : B = "School ends at 3:40"}
variable {c1 : C = "It took 30 minutes to do the homework"}

theorem duration_of_time : C = "It took 30 minutes to do the homework" :=
  sorry

end duration_of_time_l669_66915


namespace simplify_sqrt_product_l669_66924

theorem simplify_sqrt_product (x : ℝ) : 
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 120 * x * Real.sqrt (2 * x) := 
by
  sorry

end simplify_sqrt_product_l669_66924


namespace third_intermission_served_l669_66906

def total_served : ℚ :=  0.9166666666666666
def first_intermission : ℚ := 0.25
def second_intermission : ℚ := 0.4166666666666667

theorem third_intermission_served : first_intermission + second_intermission ≤ total_served →
  (total_served - (first_intermission + second_intermission)) = 0.25 :=
by
  sorry

end third_intermission_served_l669_66906


namespace arccos_cos_eight_l669_66990

theorem arccos_cos_eight : Real.arccos (Real.cos 8) = 8 - 2 * Real.pi :=
by sorry

end arccos_cos_eight_l669_66990


namespace percent_of_class_received_50_to_59_l669_66999

-- Define the frequencies for each score range
def freq_90_to_100 := 5
def freq_80_to_89 := 7
def freq_70_to_79 := 9
def freq_60_to_69 := 8
def freq_50_to_59 := 4
def freq_below_50 := 3

-- Define the total number of students
def total_students := freq_90_to_100 + freq_80_to_89 + freq_70_to_79 + freq_60_to_69 + freq_50_to_59 + freq_below_50

-- Define the frequency of students scoring in the 50%-59% range
def freq_50_to_59_ratio := (freq_50_to_59 : ℚ) / total_students

-- Define the percentage calculation
def percent_50_to_59 := freq_50_to_59_ratio * 100

theorem percent_of_class_received_50_to_59 :
  percent_50_to_59 = 100 / 9 := 
by {
  sorry
}

end percent_of_class_received_50_to_59_l669_66999


namespace C_share_correct_l669_66905

noncomputable def C_share (B_invest: ℝ) (total_profit: ℝ) : ℝ :=
  let A_invest := 3 * B_invest
  let C_invest := (3 * B_invest) * (3/2)
  let total_invest := (3 * B_invest + B_invest + C_invest)
  (C_invest / total_invest) * total_profit

theorem C_share_correct (B_invest total_profit: ℝ) 
  (hA : ∀ x: ℝ, A_invest = 3 * x)
  (hC : ∀ x: ℝ, C_invest = (3 * x) * (3/2)) :
  C_share B_invest 12375 = 6551.47 :=
by
  sorry

end C_share_correct_l669_66905


namespace find_values_of_c_x1_x2_l669_66911

theorem find_values_of_c_x1_x2 (x₁ x₂ c : ℝ)
    (h1 : x₁ + x₂ = -2)
    (h2 : x₁ * x₂ = c)
    (h3 : x₁^2 + x₂^2 = c^2 - 2 * c) :
    c = -2 ∧ x₁ = -1 + Real.sqrt 3 ∧ x₂ = -1 - Real.sqrt 3 :=
by
  sorry

end find_values_of_c_x1_x2_l669_66911


namespace ben_fraction_of_taxes_l669_66950

theorem ben_fraction_of_taxes 
  (gross_income : ℝ) (car_payment : ℝ) (fraction_spend_on_car : ℝ) (after_tax_income_fraction : ℝ) 
  (h1 : gross_income = 3000) (h2 : car_payment = 400) (h3 : fraction_spend_on_car = 0.2) :
  after_tax_income_fraction = (1 / 3) :=
by
  sorry

end ben_fraction_of_taxes_l669_66950


namespace min_value_of_squares_l669_66946

theorem min_value_of_squares (a b : ℝ) 
  (h_cond : a^2 - 2015 * a = b^2 - 2015 * b)
  (h_neq : a ≠ b)
  (h_positive_a : 0 < a)
  (h_positive_b : 0 < b) : 
  a^2 + b^2 ≥ 2015^2 / 2 :=
sorry

end min_value_of_squares_l669_66946


namespace correct_car_selection_l669_66920

-- Define the production volumes
def production_emgrand : ℕ := 1600
def production_king_kong : ℕ := 6000
def production_freedom_ship : ℕ := 2000

-- Define the total number of cars produced
def total_production : ℕ := production_emgrand + production_king_kong + production_freedom_ship

-- Define the number of cars selected for inspection
def cars_selected_for_inspection : ℕ := 48

-- Calculate the sampling ratio
def sampling_ratio : ℚ := cars_selected_for_inspection / total_production

-- Define the expected number of cars to be selected from each model using the sampling ratio
def cars_selected_emgrand : ℚ := sampling_ratio * production_emgrand
def cars_selected_king_kong : ℚ := sampling_ratio * production_king_kong
def cars_selected_freedom_ship : ℚ := sampling_ratio * production_freedom_ship

theorem correct_car_selection :
  cars_selected_emgrand = 8 ∧ cars_selected_king_kong = 30 ∧ cars_selected_freedom_ship = 10 := by
  sorry

end correct_car_selection_l669_66920


namespace roots_form_parallelogram_l669_66956

theorem roots_form_parallelogram :
  let polynomial := fun (z : ℂ) (a : ℝ) =>
    z^4 - 8*z^3 + 13*a*z^2 - 2*(3*a^2 + 2*a - 4)*z - 2
  let a1 := 7.791
  let a2 := -8.457
  ∀ z1 z2 z3 z4 : ℂ,
    ( (polynomial z1 a1 = 0) ∧ (polynomial z2 a1 = 0) ∧ (polynomial z3 a1 = 0) ∧ (polynomial z4 a1 = 0)
    ∨ (polynomial z1 a2 = 0) ∧ (polynomial z2 a2 = 0) ∧ (polynomial z3 a2 = 0) ∧ (polynomial z4 a2 = 0) )
    → ( (z1 + z2 + z3 + z4) / 4 = 2 )
    → ( Complex.abs (z1 - z2) = Complex.abs (z3 - z4) 
      ∧ Complex.abs (z1 - z3) = Complex.abs (z2 - z4) ) := sorry

end roots_form_parallelogram_l669_66956


namespace new_container_volume_l669_66933

def volume_of_cube (s : ℝ) : ℝ := s^3

theorem new_container_volume (s : ℝ) (h : volume_of_cube s = 4) : 
  volume_of_cube (2 * s) * volume_of_cube (3 * s) * volume_of_cube (4 * s) = 96 :=
by
  sorry

end new_container_volume_l669_66933


namespace number_of_sides_of_regular_polygon_l669_66931

theorem number_of_sides_of_regular_polygon (h: ∀ (n: ℕ), (180 * (n - 2) / n) = 135) : ∃ n, n = 8 :=
by
  sorry

end number_of_sides_of_regular_polygon_l669_66931


namespace road_construction_days_l669_66939

theorem road_construction_days
  (length_of_road : ℝ)
  (initial_men : ℕ)
  (completed_length : ℝ)
  (completed_days : ℕ)
  (extra_men : ℕ)
  (initial_days : ℕ)
  (remaining_length : ℝ)
  (remaining_days : ℕ)
  (total_men : ℕ) :
  length_of_road = 15 →
  initial_men = 30 →
  completed_length = 2.5 →
  completed_days = 100 →
  extra_men = 45 →
  initial_days = initial_days →
  remaining_length = length_of_road - completed_length →
  remaining_days = initial_days - completed_days →
  total_men = initial_men + extra_men →
  initial_days = 700 :=
by
  intros
  sorry

end road_construction_days_l669_66939


namespace cost_of_first_15_kgs_l669_66989

def cost_33_kg := 333
def cost_36_kg := 366
def kilo_33 := 33
def kilo_36 := 36
def first_limit := 30
def extra_3kg := 3  -- 33 - 30
def extra_6kg := 6  -- 36 - 30

theorem cost_of_first_15_kgs (l q : ℕ) 
  (h1 : first_limit * l + extra_3kg * q = cost_33_kg)
  (h2 : first_limit * l + extra_6kg * q = cost_36_kg) :
  15 * l = 150 :=
by
  sorry

end cost_of_first_15_kgs_l669_66989


namespace area_difference_l669_66929

noncomputable def speed_ratio_A_B : ℚ := 3 / 2
noncomputable def side_length : ℝ := 100
noncomputable def perimeter : ℝ := 4 * side_length

noncomputable def distance_A := (3 / 5) * perimeter
noncomputable def distance_B := perimeter - distance_A

noncomputable def EC := distance_A - 2 * side_length
noncomputable def DE := distance_B - side_length

noncomputable def area_ADE := 0.5 * DE * side_length
noncomputable def area_BCE := 0.5 * EC * side_length

theorem area_difference :
  (area_ADE - area_BCE) = 1000 :=
by
  sorry

end area_difference_l669_66929


namespace initial_position_l669_66909

variable (x : Int)

theorem initial_position 
  (h: x - 5 + 4 + 2 - 3 + 1 = 6) : x = 7 := 
  by 
  sorry

end initial_position_l669_66909


namespace optimal_response_l669_66944

theorem optimal_response (n : ℕ) (m : ℕ) (s : ℕ) (a_1 : ℕ) (a_2 : ℕ -> ℕ) (a_opt : ℕ):
  n = 100 → 
  m = 107 →
  (∀ i, i ≥ 1 ∧ i ≤ 99 → a_2 i = a_opt) →
  a_1 = 7 :=
by
  sorry

end optimal_response_l669_66944


namespace N_def_M_intersection_CU_N_def_M_union_N_def_l669_66958

section Sets

variable {α : Type}

-- Declarations of conditions
def U := {x : ℝ | -3 ≤ x ∧ x ≤ 3}
def M := {x : ℝ | -1 < x ∧ x < 1}
def CU (N : Set ℝ) := {x : ℝ | 0 < x ∧ x < 2}

-- Problem statements
theorem N_def (N : Set ℝ) : N = {x : ℝ | (-3 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 3)} ↔ CU N = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

theorem M_intersection_CU_N_def (N : Set ℝ) : (M ∩ CU N) = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

theorem M_union_N_def (N : Set ℝ) : (M ∪ N) = {x : ℝ | (-3 ≤ x ∧ x < 1) ∨ (2 ≤ x ∧ x ≤ 3)} :=
by sorry

end Sets

end N_def_M_intersection_CU_N_def_M_union_N_def_l669_66958


namespace initial_speed_l669_66952

variable (D T : ℝ) -- Total distance D and total time T
variable (S : ℝ)   -- Initial speed S

theorem initial_speed :
  (2 * D / 3) = (S * T / 3) →
  (35 = (D / (2 * T))) →
  S = 70 :=
by
  intro h1 h2
  -- Skipping the proof with 'sorry'
  sorry

end initial_speed_l669_66952


namespace sin_315_eq_neg_sqrt2_over_2_l669_66979

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l669_66979


namespace compare_times_l669_66904

variable {v : ℝ} (h_v_pos : 0 < v)

/-- 
  Jones covered a distance of 80 miles on his first trip at speed v.
  On a later trip, he traveled 360 miles at four times his original speed.
  Prove that his new time is (9/8) times his original time.
-/
theorem compare_times :
  let t1 := 80 / v
  let t2 := 360 / (4 * v)
  t2 = (9 / 8) * t1 :=
by
  sorry

end compare_times_l669_66904


namespace number_of_solutions_is_3_l669_66917

noncomputable def count_solutions : Nat :=
  Nat.card {x : Nat // x < 150 ∧ (x + 15) % 45 = 75 % 45}

theorem number_of_solutions_is_3 : count_solutions = 3 := by
  sorry

end number_of_solutions_is_3_l669_66917


namespace votes_cast_l669_66980

theorem votes_cast (candidate_percentage : ℝ) (vote_difference : ℝ) (total_votes : ℝ) 
  (h1 : candidate_percentage = 0.30) 
  (h2 : vote_difference = 1760) 
  (h3 : total_votes = vote_difference / (1 - 2 * candidate_percentage)) 
  : total_votes = 4400 := by
  sorry

end votes_cast_l669_66980


namespace expression_increase_l669_66978

variable {x y : ℝ}

theorem expression_increase (hx : x > 0) (hy : y > 0) :
  let original_expr := 3 * x^2 * y
  let new_x := 1.2 * x
  let new_y := 2.4 * y
  let new_expr := 3 * new_x ^ 2 * new_y
  (new_expr / original_expr) = 3.456 :=
by
-- original_expr is 3 * x^2 * y
-- new_x = 1.2 * x
-- new_y = 2.4 * y
-- new_expr = 3 * (1.2 * x)^2 * (2.4 * y)
-- (new_expr / original_expr) = (10.368 * x^2 * y) / (3 * x^2 * y)
-- (new_expr / original_expr) = 10.368 / 3
-- (new_expr / original_expr) = 3.456
sorry

end expression_increase_l669_66978


namespace aarti_completes_work_multiple_l669_66994

-- Define the condition that Aarti can complete one piece of work in 9 days.
def aarti_work_rate (work_size : ℕ) : ℕ := 9

-- Define the task to find how many times she will complete the work in 27 days.
def aarti_work_multiple (total_days : ℕ) (work_size: ℕ) : ℕ :=
  total_days / (aarti_work_rate work_size)

-- The theorem to prove the number of times Aarti will complete the work.
theorem aarti_completes_work_multiple : aarti_work_multiple 27 1 = 3 := by
  sorry

end aarti_completes_work_multiple_l669_66994


namespace integer_sequence_count_l669_66918

theorem integer_sequence_count (a₀ : ℕ) (step : ℕ → ℕ) (n : ℕ) 
  (h₀ : a₀ = 5184)
  (h_step : ∀ k, k < n → step k = (a₀ / 4^k))
  (h_stop : a₀ = (4 ^ (n - 1)) * 81) :
  n = 4 := 
sorry

end integer_sequence_count_l669_66918


namespace find_a12_l669_66970

variable (a : ℕ → ℝ) (q : ℝ)
variable (h1 : ∀ n, a (n + 1) = a n * q)
variable (h2 : abs q > 1)
variable (h3 : a 1 + a 6 = 2)
variable (h4 : a 3 * a 4 = -15)

theorem find_a12 : a 11 = -25 / 3 :=
by sorry

end find_a12_l669_66970


namespace blue_whale_tongue_weight_in_tons_l669_66936

-- Define the conditions
def weight_of_tongue_pounds : ℕ := 6000
def pounds_per_ton : ℕ := 2000

-- Define the theorem stating the question and its answer
theorem blue_whale_tongue_weight_in_tons :
  (weight_of_tongue_pounds / pounds_per_ton) = 3 :=
by sorry

end blue_whale_tongue_weight_in_tons_l669_66936


namespace twenty_first_term_is_4641_l669_66991

def nthGroupStart (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

def sumGroup (start n : ℕ) : ℕ :=
  (n * (start + (start + n - 1))) / 2

theorem twenty_first_term_is_4641 : sumGroup (nthGroupStart 21) 21 = 4641 := by
  sorry

end twenty_first_term_is_4641_l669_66991


namespace heat_of_reaction_correct_l669_66993

def delta_H_f_NH4Cl : ℝ := -314.43  -- Enthalpy of formation of NH4Cl in kJ/mol
def delta_H_f_H2O : ℝ := -285.83    -- Enthalpy of formation of H2O in kJ/mol
def delta_H_f_HCl : ℝ := -92.31     -- Enthalpy of formation of HCl in kJ/mol
def delta_H_f_NH4OH : ℝ := -80.29   -- Enthalpy of formation of NH4OH in kJ/mol

def delta_H_rxn : ℝ :=
  ((2 * delta_H_f_NH4OH) + (2 * delta_H_f_HCl)) -
  ((2 * delta_H_f_NH4Cl) + (2 * delta_H_f_H2O))

theorem heat_of_reaction_correct :
  delta_H_rxn = 855.32 :=
  by
    -- Calculation and proof steps go here
    sorry

end heat_of_reaction_correct_l669_66993


namespace radius_of_arch_bridge_l669_66903

theorem radius_of_arch_bridge :
  ∀ (AB CD AD r : ℝ),
    AB = 12 →
    CD = 4 →
    AD = AB / 2 →
    r^2 = AD^2 + (r - CD)^2 →
    r = 6.5 :=
by
  intros AB CD AD r hAB hCD hAD h_eq
  sorry

end radius_of_arch_bridge_l669_66903


namespace elizabeth_needs_to_borrow_more_money_l669_66960

-- Define the costs of the items
def pencil_cost : ℝ := 6.00 
def notebook_cost : ℝ := 3.50 
def pen_cost : ℝ := 2.25 

-- Define the amount of money Elizabeth initially has and what she borrowed
def elizabeth_money : ℝ := 5.00 
def borrowed_money : ℝ := 0.53 

-- Define the total cost of the items
def total_cost : ℝ := pencil_cost + notebook_cost + pen_cost

-- Define the total amount of money Elizabeth has
def total_money : ℝ := elizabeth_money + borrowed_money

-- Define the additional amount Elizabeth needs to borrow
def amount_needed_to_borrow : ℝ := total_cost - total_money

-- The theorem to prove that Elizabeth needs to borrow an additional $6.22
theorem elizabeth_needs_to_borrow_more_money : 
  amount_needed_to_borrow = 6.22 := by 
    -- Proof goes here
    sorry

end elizabeth_needs_to_borrow_more_money_l669_66960


namespace fractions_sum_to_one_l669_66964

theorem fractions_sum_to_one :
  ∃ (a b c : ℕ), (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 1) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ ((a, b, c) = (2, 3, 6) ∨ (a, b, c) = (2, 6, 3) ∨ (a, b, c) = (3, 2, 6) ∨ (a, b, c) = (3, 6, 2) ∨ (a, b, c) = (6, 2, 3) ∨ (a, b, c) = (6, 3, 2)) :=
by
  sorry

end fractions_sum_to_one_l669_66964


namespace problem1_proof_problem2_proof_l669_66919

noncomputable def problem1_statement : Prop :=
  (2 * Real.sin (Real.pi / 6) - Real.sin (Real.pi / 4) * Real.cos (Real.pi / 4) = 1 / 2)

noncomputable def problem2_statement : Prop :=
  ((-1)^2023 + 2 * Real.sin (Real.pi / 4) - Real.cos (Real.pi / 6) + Real.sin (Real.pi / 3) + Real.tan (Real.pi / 3)^2 = 2 + Real.sqrt 2)

theorem problem1_proof : problem1_statement :=
by
  sorry

theorem problem2_proof : problem2_statement :=
by
  sorry

end problem1_proof_problem2_proof_l669_66919


namespace sum_of_products_two_at_a_time_l669_66949

theorem sum_of_products_two_at_a_time (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a + b + c = 23) : 
  ab + bc + ac = 131 :=
by
  sorry

end sum_of_products_two_at_a_time_l669_66949


namespace largest_possible_s_l669_66996

theorem largest_possible_s 
  (r s : ℕ) 
  (hr : r ≥ s) 
  (hs : s ≥ 3) 
  (hangles : (r - 2) * 60 * s = (s - 2) * 61 * r) : 
  s = 121 := 
sorry

end largest_possible_s_l669_66996


namespace find_a_l669_66937

theorem find_a (a b c d : ℕ) (h1 : a + b = d) (h2 : b + c = 6) (h3 : c + d = 7) : a = 1 :=
by
  sorry

end find_a_l669_66937


namespace cricket_innings_l669_66943

theorem cricket_innings (n : ℕ) 
  (avg_run_inn : n * 36 = n * 36)  -- average runs is 36 (initially true for any n)
  (increase_avg_by_4 : (36 * n + 120) / (n + 1) = 40) : 
  n = 20 := 
sorry

end cricket_innings_l669_66943


namespace intersection_complement_M_and_N_l669_66986
open Set

def U := @univ ℝ
def M := {x : ℝ | x^2 + 2*x - 8 ≤ 0}
def N := {x : ℝ | -1 < x ∧ x < 3}
def complement_M := {x : ℝ | ¬ (x ∈ M)}

theorem intersection_complement_M_and_N :
  (complement_M ∩ N) = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_complement_M_and_N_l669_66986


namespace find_x2_plus_y2_l669_66925

theorem find_x2_plus_y2
  (x y : ℝ)
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := 
by
  sorry

end find_x2_plus_y2_l669_66925


namespace least_integer_x_l669_66995

theorem least_integer_x (x : ℤ) : (2 * |x| + 7 < 17) → x = -4 := by
  sorry

end least_integer_x_l669_66995


namespace James_has_43_Oreos_l669_66932

variable (J : ℕ)
variable (James_Oreos : ℕ)

-- Conditions
def condition1 : Prop := James_Oreos = 4 * J + 7
def condition2 : Prop := J + James_Oreos = 52

-- The statement to prove: James has 43 Oreos given the conditions
theorem James_has_43_Oreos (h1 : condition1 J James_Oreos) (h2 : condition2 J James_Oreos) : James_Oreos = 43 :=
by
  sorry

end James_has_43_Oreos_l669_66932


namespace find_supplementary_angle_l669_66941

noncomputable def degree (x : ℝ) : ℝ := x
noncomputable def complementary_angle (x : ℝ) : ℝ := 90 - x
noncomputable def supplementary_angle (x : ℝ) : ℝ := 180 - x

theorem find_supplementary_angle
  (x : ℝ)
  (h1 : degree x / complementary_angle x = 1 / 8) :
  supplementary_angle x = 170 :=
by
  sorry

end find_supplementary_angle_l669_66941


namespace rabbit_speed_correct_l669_66910

-- Define the conditions given in the problem
def rabbit_speed (x : ℝ) : Prop :=
2 * (2 * x + 4) = 188

-- State the main theorem using the defined conditions
theorem rabbit_speed_correct : ∃ x : ℝ, rabbit_speed x ∧ x = 45 :=
by
  sorry

end rabbit_speed_correct_l669_66910


namespace seventh_term_in_geometric_sequence_l669_66923

theorem seventh_term_in_geometric_sequence :
  ∃ r, (4 * r^8 = 2097152) ∧ (4 * r^6 = 1048576) :=
by
  sorry

end seventh_term_in_geometric_sequence_l669_66923


namespace real_condition_complex_condition_pure_imaginary_condition_l669_66975

-- Definitions for our conditions
def is_real (z : ℂ) : Prop := z.im = 0
def is_complex (z : ℂ) : Prop := z.im ≠ 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- The given complex number definition
def z (m : ℝ) : ℂ := { re := m^2 + m, im := m^2 - 1 }

-- Prove that for z to be a real number, m must be ±1
theorem real_condition (m : ℝ) : is_real (z m) ↔ m = 1 ∨ m = -1 := 
sorry

-- Prove that for z to be a complex number, m must not be ±1 
theorem complex_condition (m : ℝ) : is_complex (z m) ↔ m ≠ 1 ∧ m ≠ -1 := 
sorry 

-- Prove that for z to be a pure imaginary number, m must be 0
theorem pure_imaginary_condition (m : ℝ) : is_pure_imaginary (z m) ↔ m = 0 := 
sorry 

end real_condition_complex_condition_pure_imaginary_condition_l669_66975


namespace find_a_l669_66907

theorem find_a (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 3) (h₂ : 3 / a + 6 / b = 2 / 3) : 
  a = 9 * b / (2 * b - 18) :=
by
  sorry

end find_a_l669_66907


namespace exists_coprime_less_than_100_l669_66968

theorem exists_coprime_less_than_100 (a b c : ℕ) (ha : a < 1000000) (hb : b < 1000000) (hc : c < 1000000) :
  ∃ d, d < 100 ∧ gcd d a = 1 ∧ gcd d b = 1 ∧ gcd d c = 1 :=
by sorry

end exists_coprime_less_than_100_l669_66968


namespace probability_of_hitting_10_or_9_probability_of_hitting_at_least_7_probability_of_hitting_less_than_8_l669_66916

-- Definitions of the probabilities
def P_A := 0.24
def P_B := 0.28
def P_C := 0.19
def P_D := 0.16
def P_E := 0.13

-- Prove that the probability of hitting the 10 or 9 rings is 0.52
theorem probability_of_hitting_10_or_9 : P_A + P_B = 0.52 :=
  by sorry

-- Prove that the probability of hitting at least the 7 ring is 0.87
theorem probability_of_hitting_at_least_7 : P_A + P_B + P_C + P_D = 0.87 :=
  by sorry

-- Prove that the probability of hitting less than 8 rings is 0.29
theorem probability_of_hitting_less_than_8 : P_D + P_E = 0.29 :=
  by sorry

end probability_of_hitting_10_or_9_probability_of_hitting_at_least_7_probability_of_hitting_less_than_8_l669_66916


namespace quadratic_inequality_solution_set_l669_66945

theorem quadratic_inequality_solution_set (m : ℝ) (h : m * (m - 1) < 0) : 
  ∀ x : ℝ, (x^2 - (m + 1/m) * x + 1 < 0) ↔ m < x ∧ x < 1/m :=
by
  sorry

end quadratic_inequality_solution_set_l669_66945


namespace max_value_of_expression_l669_66951

theorem max_value_of_expression (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ 7 * Real.sqrt 2 :=
sorry

end max_value_of_expression_l669_66951


namespace total_collected_funds_l669_66974

theorem total_collected_funds (A B T : ℕ) (hA : A = 5) (hB : B = 3 * A + 3) (h_quotient : B / 3 = 6) (hT : T = B * (B / 3) + A) : 
  T = 113 := 
by 
  sorry

end total_collected_funds_l669_66974


namespace quadratic_eq1_solution_quadratic_eq2_solution_l669_66900

-- Define the first problem and its conditions
theorem quadratic_eq1_solution :
  ∀ x : ℝ, 4 * x^2 + x - (1 / 2) = 0 ↔ (x = -1 / 2 ∨ x = 1 / 4) :=
by
  -- The proof is omitted
  sorry

-- Define the second problem and its conditions
theorem quadratic_eq2_solution :
  ∀ y : ℝ, (y - 2) * (y + 3) = 6 ↔ (y = -4 ∨ y = 3) :=
by
  -- The proof is omitted
  sorry

end quadratic_eq1_solution_quadratic_eq2_solution_l669_66900


namespace supplementary_angle_ratio_l669_66988

theorem supplementary_angle_ratio (x : ℝ) (hx : 4 * x + x = 180) : x = 36 :=
by sorry

end supplementary_angle_ratio_l669_66988


namespace molecular_weight_BaCl2_l669_66953

def molecular_weight_one_mole (w_four_moles : ℕ) (n : ℕ) : ℕ := 
    w_four_moles / n

theorem molecular_weight_BaCl2 
    (w_four_moles : ℕ)
    (H : w_four_moles = 828) :
  molecular_weight_one_mole w_four_moles 4 = 207 :=
by
  -- sorry to skip the proof
  sorry

end molecular_weight_BaCl2_l669_66953


namespace part1_minimum_b_over_a_l669_66935

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

-- Prove part 1
theorem part1 (x : ℝ) : (0 < x ∧ x < 1 → (f x 1 / (1/x - 1) > 0)) ∧ (1 < x → (f x 1 / (1/x - 1) < 0)) := sorry

-- Prove part 2
lemma part2 (a b : ℝ) (h : ∀ x > 0, f x a ≤ b - a) (ha : a ≠ 0) : ∃ x > 0, f x a = b - a := sorry

theorem minimum_b_over_a (a : ℝ) (ha : a ≠ 0) (h : ∀ x > 0, f x a ≤ b - a) : b/a ≥ 0 := sorry

end part1_minimum_b_over_a_l669_66935


namespace find_m_given_slope_condition_l669_66940

variable (m : ℝ)

theorem find_m_given_slope_condition
  (h : (m - 4) / (3 - 2) = 1) : m = 5 :=
sorry

end find_m_given_slope_condition_l669_66940
