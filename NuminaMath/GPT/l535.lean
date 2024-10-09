import Mathlib

namespace find_number_of_white_balls_l535_53574

theorem find_number_of_white_balls (n : ℕ) (h : 6 / (6 + n) = 2 / 5) : n = 9 :=
sorry

end find_number_of_white_balls_l535_53574


namespace add_base6_l535_53588

def base6_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  6 * d1 + d0

theorem add_base6 (a b : Nat) (ha : base6_to_base10 a = 23) (hb : base6_to_base10 b = 10) : 
  base6_to_base10 (53 : Nat) = 33 :=
by
  sorry

end add_base6_l535_53588


namespace least_value_of_b_l535_53540

variable {x y b : ℝ}

noncomputable def condition_inequality (x y b : ℝ) : Prop :=
  (x^2 + y^2)^2 ≤ b * (x^4 + y^4)

theorem least_value_of_b (h : ∀ x y : ℝ, condition_inequality x y b) : b ≥ 2 := 
sorry

end least_value_of_b_l535_53540


namespace intersection_A_B_l535_53545

def A := {x : ℝ | x^2 - x - 2 ≤ 0}
def B := {x : ℝ | ∃ y : ℝ, y = Real.log (1 - x)}

theorem intersection_A_B : (A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_A_B_l535_53545


namespace VivianMailApril_l535_53543

variable (piecesMailApril piecesMailMay piecesMailJune piecesMailJuly piecesMailAugust : ℕ)

-- Conditions
def condition_double_monthly (a b : ℕ) : Prop := b = 2 * a

axiom May : piecesMailMay = 10
axiom June : piecesMailJune = 20
axiom July : piecesMailJuly = 40
axiom August : piecesMailAugust = 80

axiom patternMay : condition_double_monthly piecesMailApril piecesMailMay
axiom patternJune : condition_double_monthly piecesMailMay piecesMailJune
axiom patternJuly : condition_double_monthly piecesMailJune piecesMailJuly
axiom patternAugust : condition_double_monthly piecesMailJuly piecesMailAugust

-- Statement to prove
theorem VivianMailApril :
  piecesMailApril = 5 :=
by
  sorry

end VivianMailApril_l535_53543


namespace fraction_meaningful_l535_53558

theorem fraction_meaningful (x : ℝ) : (∃ y, y = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end fraction_meaningful_l535_53558


namespace math_problem_l535_53516

-- Define constants and conversions from decimal/mixed numbers to fractions
def thirteen_and_three_quarters : ℚ := 55 / 4
def nine_and_sixth : ℚ := 55 / 6
def one_point_two : ℚ := 1.2
def ten_point_three : ℚ := 103 / 10
def eight_and_half : ℚ := 17 / 2
def six_point_eight : ℚ := 34 / 5
def three_and_three_fifths : ℚ := 18 / 5
def five_and_five_sixths : ℚ := 35 / 6
def three_and_two_thirds : ℚ := 11 / 3
def three_and_one_sixth : ℚ := 19 / 6
def fifty_six : ℚ := 56
def twenty_seven_and_sixth : ℚ := 163 / 6

def E : ℚ := 
  ((thirteen_and_three_quarters + nine_and_sixth) * one_point_two) / ((ten_point_three - eight_and_half) * (5 / 9)) + 
  ((six_point_eight - three_and_three_fifths) * five_and_five_sixths) / ((three_and_two_thirds - three_and_one_sixth) * fifty_six) - 
  twenty_seven_and_sixth

theorem math_problem : E = 29 / 3 := by
  sorry

end math_problem_l535_53516


namespace zionsDadX_l535_53594

section ZionProblem

-- Define the conditions
variables (Z : ℕ) (D : ℕ) (X : ℕ)

-- Zion's current age
def ZionAge : Prop := Z = 8

-- Zion's dad's age in terms of Zion's age and X
def DadsAge : Prop := D = 4 * Z + X

-- Zion's dad's age in 10 years compared to Zion's age in 10 years
def AgeInTenYears : Prop := D + 10 = (Z + 10) + 27

-- The theorem statement to be proved
theorem zionsDadX :
  ZionAge Z →  
  DadsAge Z D X →  
  AgeInTenYears Z D →  
  X = 3 := 
sorry

end ZionProblem

end zionsDadX_l535_53594


namespace solution_set_of_inequality_l535_53575

theorem solution_set_of_inequality : { x : ℝ | x^2 - 2 * x + 1 ≤ 0 } = {1} :=
sorry

end solution_set_of_inequality_l535_53575


namespace diameter_percentage_l535_53586

theorem diameter_percentage (d_R d_S : ℝ) (h : π * (d_R / 2)^2 = 0.16 * π * (d_S / 2)^2) :
  (d_R / d_S) * 100 = 40 :=
by {
  sorry
}

end diameter_percentage_l535_53586


namespace infinite_n_dividing_a_pow_n_plus_1_l535_53547

theorem infinite_n_dividing_a_pow_n_plus_1 (a : ℕ) (h1 : 1 < a) (h2 : a % 2 = 0) :
  ∃ (S : Set ℕ), S.Infinite ∧ ∀ n ∈ S, n ∣ a^n + 1 := 
sorry

end infinite_n_dividing_a_pow_n_plus_1_l535_53547


namespace min_value_expression_l535_53508

variable {m n : ℝ}

theorem min_value_expression (hm : m > 0) (hn : n > 0) (hperp : m + n = 1) :
  ∃ (m n : ℝ), (1 / m + 2 / n = 3 + 2 * Real.sqrt 2) :=
by 
  sorry

end min_value_expression_l535_53508


namespace arithmetic_sequence_sum_19_l535_53592

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_19 (h1 : is_arithmetic_sequence a)
  (h2 : a 9 = 11) (h3 : a 11 = 9) (h4 : ∀ n, S n = n / 2 * (a 1 + a n)) :
  S 19 = 190 :=
sorry

end arithmetic_sequence_sum_19_l535_53592


namespace value_of_each_baseball_card_l535_53571

theorem value_of_each_baseball_card (x : ℝ) (h : 2 * x + 3 = 15) : x = 6 := by
  sorry

end value_of_each_baseball_card_l535_53571


namespace solve_for_x_l535_53501

theorem solve_for_x (x : ℝ) :
  let area_square1 := (2 * x) ^ 2
  let area_square2 := (5 * x) ^ 2
  let area_triangle := 0.5 * (2 * x) * (5 * x)
  (area_square1 + area_square2 + area_triangle = 850) → x = 5 := by
  sorry

end solve_for_x_l535_53501


namespace RS_plus_ST_l535_53503

theorem RS_plus_ST {a b c d e : ℕ} 
  (h1 : a = 68) 
  (h2 : b = 10) 
  (h3 : c = 7) 
  (h4 : d = 6) 
  : e = 3 :=
sorry

end RS_plus_ST_l535_53503


namespace appropriate_line_chart_for_temperature_l535_53584

-- Define the assumption that line charts are effective in displaying changes in data over time
axiom effective_line_chart_display (changes_over_time : Prop) : Prop

-- Define the statement to be proved, using the assumption above
theorem appropriate_line_chart_for_temperature (changes_over_time : Prop) 
  (line_charts_effective : effective_line_chart_display changes_over_time) : Prop :=
  sorry

end appropriate_line_chart_for_temperature_l535_53584


namespace max_sum_first_n_terms_l535_53548

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem max_sum_first_n_terms (a1 : ℝ) (h1 : a1 > 0)
  (h2 : 5 * a_n a1 d 8 = 8 * a_n a1 d 13) :
  ∃ n : ℕ, n = 21 ∧ ∀ m : ℕ, S_n a1 d m ≤ S_n a1 d n :=
by
  sorry

end max_sum_first_n_terms_l535_53548


namespace most_frequent_digit_100000_l535_53518

/- Define the digital root function -/
def digital_root (n : ℕ) : ℕ :=
  if n == 0 then 0 else if n % 9 == 0 then 9 else n % 9

/- Define the problem statement -/
theorem most_frequent_digit_100000 : 
  ∃ digit : ℕ, 
  digit = 1 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → ∃ k : ℕ, k = digital_root n ∧ k = digit) →
  digit = 1 :=
sorry

end most_frequent_digit_100000_l535_53518


namespace find_ages_l535_53570

-- Definitions of the conditions
def cond1 (D S : ℕ) : Prop := D = 3 * S
def cond2 (D S : ℕ) : Prop := D + 5 = 2 * (S + 5)

-- Theorem statement
theorem find_ages (D S : ℕ) 
  (h1 : cond1 D S) 
  (h2 : cond2 D S) : 
  D = 15 ∧ S = 5 :=
by 
  sorry

end find_ages_l535_53570


namespace bottles_not_placed_in_crate_l535_53560

-- Defining the constants based on the conditions
def bottles_per_crate : Nat := 12
def total_bottles : Nat := 130
def crates : Nat := 10

-- Theorem statement based on the question and the correct answer
theorem bottles_not_placed_in_crate :
  total_bottles - (bottles_per_crate * crates) = 10 :=
by
  -- Proof will be here
  sorry

end bottles_not_placed_in_crate_l535_53560


namespace find_a_3_l535_53598

noncomputable def a_n (n : ℕ) : ℤ := 2 + (n - 1)  -- Definition of the arithmetic sequence

theorem find_a_3 (d : ℤ) (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 5 + a 7 = 2 * a 4 + 4) : a 3 = 4 :=
by 
  sorry

end find_a_3_l535_53598


namespace polyhedron_faces_l535_53550

theorem polyhedron_faces (V E : ℕ) (F T P : ℕ) (h1 : F = 40) (h2 : V - E + F = 2) (h3 : T + P = 40) 
  (h4 : E = (3 * T + 4 * P) / 2) (h5 : V = (160 - T) / 2 - 38) (h6 : P = 3) (h7 : T = 1) :
  100 * P + 10 * T + V = 351 :=
by
  sorry

end polyhedron_faces_l535_53550


namespace circle_diameter_mn_origin_l535_53566

-- Definitions based on conditions in (a)
def circle_equation (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y + m = 0
def line_equation (x y : ℝ) : Prop := x + 2 * y - 4 = 0
def orthogonal (x1 x2 y1 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem to prove (based on conditions and correct answer in (b))
theorem circle_diameter_mn_origin 
  (m : ℝ) 
  (x1 y1 x2 y2 : ℝ)
  (h1: circle_equation m x1 y1) 
  (h2: circle_equation m x2 y2)
  (h3: line_equation x1 y1)
  (h4: line_equation x2 y2)
  (h5: orthogonal x1 x2 y1 y2) :
  m = 8 / 5 := 
sorry

end circle_diameter_mn_origin_l535_53566


namespace find_x_l535_53569

theorem find_x 
  (x : ℕ)
  (h : 3^x = 3^(20) * 3^(20) * 3^(18) + 3^(19) * 3^(20) * 3^(19) + 3^(18) * 3^(21) * 3^(19)) :
  x = 59 :=
sorry

end find_x_l535_53569


namespace smallest_value_expression_geq_three_l535_53577

theorem smallest_value_expression_geq_three :
  ∀ (x y : ℝ), 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2 ≥ 3 := 
by
  sorry

end smallest_value_expression_geq_three_l535_53577


namespace height_of_pyramid_l535_53552

theorem height_of_pyramid :
  let edge_cube := 6
  let edge_base_square_pyramid := 10
  let cube_volume := edge_cube ^ 3
  let sphere_volume := cube_volume
  let pyramid_volume := 2 * sphere_volume
  let base_area_square_pyramid := edge_base_square_pyramid ^ 2
  let height_pyramid := 12.96
  pyramid_volume = (1 / 3) * base_area_square_pyramid * height_pyramid :=
by
  sorry

end height_of_pyramid_l535_53552


namespace anna_final_stamp_count_l535_53537

theorem anna_final_stamp_count (anna_initial : ℕ) (alison_initial : ℕ) (jeff_initial : ℕ)
  (anna_receive_from_alison : ℕ) (anna_give_jeff : ℕ) (anna_receive_jeff : ℕ) :
  anna_initial = 37 →
  alison_initial = 28 →
  jeff_initial = 31 →
  anna_receive_from_alison = alison_initial / 2 →
  anna_give_jeff = 2 →
  anna_receive_jeff = 1 →
  ∃ result : ℕ, result = 50 :=
by
  intros
  sorry

end anna_final_stamp_count_l535_53537


namespace functional_equation_solutions_l535_53549

theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)) : 
  (∀ x : ℝ, f x = 0) ∨
  (∀ x : ℝ, f x = x - 1) ∨
  (∀ x : ℝ, f x = 1 - x) :=
sorry

end functional_equation_solutions_l535_53549


namespace train_speed_l535_53507

theorem train_speed (length : ℝ) (time : ℝ) (conversion_factor : ℝ)
  (h1 : length = 500) (h2 : time = 5) (h3 : conversion_factor = 3.6) :
  (length / time) * conversion_factor = 360 :=
by
  sorry

end train_speed_l535_53507


namespace journeymen_percentage_after_layoff_l535_53530

noncomputable def total_employees : ℝ := 20210
noncomputable def fraction_journeymen : ℝ := 2 / 7
noncomputable def total_journeymen : ℝ := total_employees * fraction_journeymen
noncomputable def laid_off_journeymen : ℝ := total_journeymen / 2
noncomputable def remaining_journeymen : ℝ := total_journeymen / 2
noncomputable def remaining_employees : ℝ := total_employees - laid_off_journeymen
noncomputable def journeymen_percentage : ℝ := (remaining_journeymen / remaining_employees) * 100

theorem journeymen_percentage_after_layoff : journeymen_percentage = 16.62 := by
  sorry

end journeymen_percentage_after_layoff_l535_53530


namespace original_price_l535_53553

theorem original_price (P : ℝ) (h1 : P + 0.10 * P = 330) : P = 300 := 
by
  sorry

end original_price_l535_53553


namespace HephaestusCharges_l535_53521

variable (x : ℕ)

theorem HephaestusCharges :
  3 * x + 6 * (12 - x) = 54 -> x = 6 :=
by
  intros h
  sorry

end HephaestusCharges_l535_53521


namespace value_of_f_3_and_f_neg_7_point_5_l535_53564

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 1) = -f x
axiom definition_f : ∀ x : ℝ, -1 < x → x < 1 → f x = x

theorem value_of_f_3_and_f_neg_7_point_5 :
  f 3 + f (-7.5) = 0.5 :=
sorry

end value_of_f_3_and_f_neg_7_point_5_l535_53564


namespace ones_digit_7_pow_35_l535_53563

theorem ones_digit_7_pow_35 : (7^35) % 10 = 3 := 
by
  sorry

end ones_digit_7_pow_35_l535_53563


namespace largest_four_digit_number_with_property_l535_53591

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l535_53591


namespace find_line_equation_l535_53556

theorem find_line_equation (k x y x₁ y₁ x₂ y₂ : ℝ) (h_parabola : y ^ 2 = 2 * x) 
  (h_line_ny_eq : y = k * x + 2) (h_intersect_1 : (y₁ - (k * x₁ + 2)) = 0)
  (h_intersect_2 : (y₂ - (k * x₂ + 2)) = 0) 
  (h_y_intercept : (0,2) = (x,y))-- the line has y-intercept 2 
  (h_origin : (0,0) = (x, y)) -- origin 
  (h_orthogonal : x₁ * x₂ + y₁ * y₂ = 0): 
  y = -x + 2 :=
by {
  sorry
}

end find_line_equation_l535_53556


namespace last_four_digits_5_pow_2015_l535_53557

theorem last_four_digits_5_pow_2015 :
  (5^2015) % 10000 = 8125 :=
by
  sorry

end last_four_digits_5_pow_2015_l535_53557


namespace total_distance_walked_l535_53582

variables
  (distance1 : ℝ := 1.2)
  (distance2 : ℝ := 0.8)
  (distance3 : ℝ := 1.5)
  (distance4 : ℝ := 0.6)
  (distance5 : ℝ := 2)

theorem total_distance_walked :
  distance1 + distance2 + distance3 + distance4 + distance5 = 6.1 :=
sorry

end total_distance_walked_l535_53582


namespace mary_balloons_correct_l535_53544

-- Define the number of black balloons Nancy has
def nancy_balloons : ℕ := 7

-- Define the multiplier that represents how many times more balloons Mary has compared to Nancy
def multiplier : ℕ := 4

-- Define the number of black balloons Mary has in terms of Nancy's balloons and the multiplier
def mary_balloons : ℕ := nancy_balloons * multiplier

-- The statement we want to prove
theorem mary_balloons_correct : mary_balloons = 28 :=
by
  sorry

end mary_balloons_correct_l535_53544


namespace find_angle_y_l535_53561

open Real

theorem find_angle_y 
    (angle_ABC angle_BAC : ℝ)
    (h1 : angle_ABC = 70)
    (h2 : angle_BAC = 50)
    (triangle_sum : ∀ {A B C : ℝ}, A + B + C = 180)
    (right_triangle_sum : ∀ D E : ℝ, D + E = 90) :
    30 = 30 :=
by
    -- Given, conditions, and intermediate results (skipped)
    sorry

end find_angle_y_l535_53561


namespace problem_inequality_solution_l535_53535

theorem problem_inequality_solution (x : ℝ) :
  5 ≤ (x - 1) / (3 * x - 7) ∧ (x - 1) / (3 * x - 7) < 10 ↔ (69 / 29) < x ∧ x ≤ (17 / 7) :=
by sorry

end problem_inequality_solution_l535_53535


namespace find_a_value_l535_53589

noncomputable def A (a : ℝ) : Set ℝ := {x | x = a}
noncomputable def B (a : ℝ) : Set ℝ := if a = 0 then ∅ else {x | a * x = 1}

theorem find_a_value (a : ℝ) :
  (A a ∩ B a = B a) → (a = 1 ∨ a = -1 ∨ a = 0) :=
by
  intro h
  sorry

end find_a_value_l535_53589


namespace x_add_y_add_one_is_composite_l535_53533

theorem x_add_y_add_one_is_composite (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (k : ℕ) (h : x^2 + x * y - y = k^2) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (x + y + 1 = a * b) :=
by
  sorry

end x_add_y_add_one_is_composite_l535_53533


namespace problem_solution_l535_53595

theorem problem_solution
  (k : ℝ)
  (y : ℝ → ℝ)
  (quadratic_fn : ∀ x, y x = (k + 2) * x^(k^2 + k - 4))
  (increase_for_neg_x : ∀ x : ℝ, x < 0 → y (x + 1) > y x) :
  k = -3 ∧ (∀ m n : ℝ, -2 ≤ m ∧ m ≤ 1 → y m = n → -4 ≤ n ∧ n ≤ 0) := 
sorry

end problem_solution_l535_53595


namespace angle_D_measure_l535_53551

theorem angle_D_measure (E D F : ℝ) (h1 : E + D + F = 180) (h2 : E = 30) (h3 : D = 2 * F) : D = 100 :=
by
  -- The proof is not required, only the statement
  sorry

end angle_D_measure_l535_53551


namespace q_minus_p_897_l535_53581

def smallest_three_digit_integer_congruent_7_mod_13 := ∃ p : ℕ, p ≥ 100 ∧ p < 1000 ∧ p % 13 = 7
def smallest_four_digit_integer_congruent_7_mod_13 := ∃ q : ℕ, q ≥ 1000 ∧ q < 10000 ∧ q % 13 = 7

theorem q_minus_p_897 : 
  (∃ p : ℕ, p ≥ 100 ∧ p < 1000 ∧ p % 13 = 7) → 
  (∃ q : ℕ, q ≥ 1000 ∧ q < 10000 ∧ q % 13 = 7) → 
  ∀ p q : ℕ, 
    (p = 8*13+7) → 
    (q = 77*13+7) → 
    q - p = 897 :=
by
  intros h1 h2 p q hp hq
  sorry

end q_minus_p_897_l535_53581


namespace range_of_a_no_fixed_points_l535_53510

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 1

theorem range_of_a_no_fixed_points : 
  ∀ a : ℝ, ¬∃ x : ℝ, f x a = x ↔ -1 < a ∧ a < 3 :=
by sorry

end range_of_a_no_fixed_points_l535_53510


namespace toms_initial_investment_l535_53523

theorem toms_initial_investment (t j k : ℕ) (hj_neq_ht : t ≠ j) (hk_neq_ht : t ≠ k) (hj_neq_hk : j ≠ k) 
  (h1 : t + j + k = 1200) 
  (h2 : t - 150 + 3 * j + 3 * k = 1800) : 
  t = 825 := 
sorry

end toms_initial_investment_l535_53523


namespace find_a3_l535_53522

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem find_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : geometric_sequence a r)
  (h2 : a 0 * a 1 * a 2 * a 3 * a 4 = 32):
  a 2 = 2 :=
sorry

end find_a3_l535_53522


namespace minimum_value_of_f_l535_53572

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x + 1)

theorem minimum_value_of_f (x : ℝ) (h : x > -1) : f x = 1 ↔ x = 0 :=
by
  sorry

end minimum_value_of_f_l535_53572


namespace find_a200_l535_53525

def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = a n + 2 * a n / n

theorem find_a200 (a : ℕ → ℕ) (h : seq a) : a 200 = 20100 :=
sorry

end find_a200_l535_53525


namespace Kath_payment_l535_53539

noncomputable def reducedPrice (standardPrice discount : ℝ) : ℝ :=
  standardPrice - discount

noncomputable def totalCost (numPeople price : ℝ) : ℝ :=
  numPeople * price

theorem Kath_payment :
  let standardPrice := 8
  let discount := 3
  let numPeople := 6
  let movieTime := 16 -- 4 P.M. in 24-hour format
  let reduced := reducedPrice standardPrice discount
  totalCost numPeople reduced = 30 :=
by
  sorry

end Kath_payment_l535_53539


namespace right_triangle_third_side_l535_53538

/-- In a right triangle, given the lengths of two sides are 4 and 5, prove that the length of the
third side is either sqrt 41 or 3. -/
theorem right_triangle_third_side (a b : ℕ) (h1 : a = 4 ∨ a = 5) (h2 : b = 4 ∨ b = 5) (h3 : a ≠ b) :
  ∃ c, c = Real.sqrt 41 ∨ c = 3 :=
by
  sorry

end right_triangle_third_side_l535_53538


namespace no_solutions_to_cubic_sum_l535_53502

theorem no_solutions_to_cubic_sum (x y z : ℤ) : 
    ¬ (x^3 + y^3 = z^3 + 4) :=
by 
  sorry

end no_solutions_to_cubic_sum_l535_53502


namespace average_is_700_l535_53531

-- Define the list of known numbers
def numbers_without_x : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755]

-- Define the value of x
def x : ℕ := 755

-- Define the list of all numbers including x
def all_numbers : List ℕ := numbers_without_x.append [x]

-- Define the total length of the list containing x
def n : ℕ := all_numbers.length

-- Define the sum of the numbers in the list including x
noncomputable def sum_all_numbers : ℕ := all_numbers.sum

-- Define the average formula
noncomputable def average : ℕ := sum_all_numbers / n

-- State the theorem
theorem average_is_700 : average = 700 := by
  sorry

end average_is_700_l535_53531


namespace not_perfect_square_infinitely_many_l535_53534

theorem not_perfect_square_infinitely_many (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_gt : b > a) (h_prime : Prime (b - a)) :
  ∃ᶠ n in at_top, ¬ IsSquare ((a ^ n + a + 1) * (b ^ n + b + 1)) :=
sorry

end not_perfect_square_infinitely_many_l535_53534


namespace exponent_calculation_l535_53587

theorem exponent_calculation (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : 
  a^(2 * m - 3 * n) = 9 / 8 := 
by
  sorry

end exponent_calculation_l535_53587


namespace fraction_sum_identity_l535_53580

variable (a b c : ℝ)

theorem fraction_sum_identity (h1 : a + b + c = 0) (h2 : a / b + b / c + c / a = 100) : 
  b / a + c / b + a / c = -103 :=
by {
  -- Proof goes here
  sorry
}

end fraction_sum_identity_l535_53580


namespace abcd_solution_l535_53532

-- Define the problem statement
theorem abcd_solution (a b c d : ℤ) (h1 : a + c = -2) (h2 : a * c + b + d = 3) (h3 : a * d + b * c = 4) (h4 : b * d = -10) : 
  a + b + c + d = 1 := by 
  sorry

end abcd_solution_l535_53532


namespace concentration_after_dilution_l535_53528

-- Definitions and conditions
def initial_volume : ℝ := 5
def initial_concentration : ℝ := 0.06
def poured_out_volume : ℝ := 1
def added_water_volume : ℝ := 2

-- Theorem statement
theorem concentration_after_dilution : 
  (initial_volume * initial_concentration - poured_out_volume * initial_concentration) / 
  (initial_volume - poured_out_volume + added_water_volume) = 0.04 :=
by 
  sorry

end concentration_after_dilution_l535_53528


namespace number_of_sides_of_polygon_l535_53526

theorem number_of_sides_of_polygon (exterior_angle : ℝ) (h : exterior_angle = 40) : 
  (360 / exterior_angle) = 9 :=
by
  sorry

end number_of_sides_of_polygon_l535_53526


namespace six_inch_cube_value_eq_844_l535_53513

-- Definition of the value of a cube in lean
noncomputable def cube_value (s₁ s₂ : ℕ) (value₁ : ℕ) : ℕ :=
  let volume₁ := s₁ ^ 3
  let volume₂ := s₂ ^ 3
  (value₁ * volume₂) / volume₁

-- Theorem stating the equivalence between the volumes and values.
theorem six_inch_cube_value_eq_844 :
  cube_value 4 6 250 = 844 :=
by
  sorry

end six_inch_cube_value_eq_844_l535_53513


namespace winnie_keeps_balloons_l535_53541

theorem winnie_keeps_balloons :
  let blueBalloons := 15
  let yellowBalloons := 40
  let purpleBalloons := 70
  let orangeBalloons := 90
  let friends := 9
  let totalBalloons := blueBalloons + yellowBalloons + purpleBalloons + orangeBalloons
  (totalBalloons % friends) = 8 := 
by 
  -- Definitions
  let blueBalloons := 15
  let yellowBalloons := 40
  let purpleBalloons := 70
  let orangeBalloons := 90
  let friends := 9
  let totalBalloons := blueBalloons + yellowBalloons + purpleBalloons + orangeBalloons
  -- Conclusion
  show totalBalloons % friends = 8
  sorry

end winnie_keeps_balloons_l535_53541


namespace solve_for_x_l535_53583

theorem solve_for_x (x : ℚ) (h : (x + 10) / (x - 4) = (x + 3) / (x - 6)) : x = 48 / 5 :=
sorry

end solve_for_x_l535_53583


namespace intersection_is_integer_for_m_l535_53573

noncomputable def intersects_at_integer_point (m : ℤ) : Prop :=
∃ x y : ℤ, y = x - 4 ∧ y = m * x + 2 * m

theorem intersection_is_integer_for_m :
  intersects_at_integer_point 8 :=
by
  -- The proof would go here
  sorry

end intersection_is_integer_for_m_l535_53573


namespace perpendicular_vectors_l535_53576

open scoped BigOperators

noncomputable def i : ℝ × ℝ := (1, 0)
noncomputable def j : ℝ × ℝ := (0, 1)
noncomputable def u : ℝ × ℝ := (1, 3)
noncomputable def v : ℝ × ℝ := (3, -1)

theorem perpendicular_vectors :
  (u.1 * v.1 + u.2 * v.2) = 0 :=
by
  have hi : i = (1, 0) := rfl
  have hj : j = (0, 1) := rfl
  have hu : u = (1, 3) := rfl
  have hv : v = (3, -1) := rfl
  -- using the dot product definition for perpendicularity
  sorry

end perpendicular_vectors_l535_53576


namespace stormi_needs_more_money_to_afford_bicycle_l535_53506

-- Definitions from conditions
def money_washed_cars : ℕ := 3 * 10
def money_mowed_lawns : ℕ := 2 * 13
def bicycle_cost : ℕ := 80
def total_earnings : ℕ := money_washed_cars + money_mowed_lawns

-- The goal to prove 
theorem stormi_needs_more_money_to_afford_bicycle :
  (bicycle_cost - total_earnings) = 24 := by
  sorry

end stormi_needs_more_money_to_afford_bicycle_l535_53506


namespace black_cars_count_l535_53554

theorem black_cars_count
    (r b : ℕ)
    (r_ratio : r = 33)
    (ratio_condition : r / b = 3 / 8) :
    b = 88 :=
by 
  sorry

end black_cars_count_l535_53554


namespace find_added_number_l535_53504

def S₁₅ := 15 * 17
def S₁₆ := 16 * 20
def added_number := S₁₆ - S₁₅

theorem find_added_number : added_number = 65 :=
by
  sorry

end find_added_number_l535_53504


namespace hilda_loan_compounding_difference_l535_53590

noncomputable def difference_due_to_compounding (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  let A_monthly := P * (1 + r / 12)^(12 * t)
  let A_annually := P * (1 + r)^t
  A_monthly - A_annually

theorem hilda_loan_compounding_difference :
  difference_due_to_compounding 8000 0.10 5 = 376.04 :=
sorry

end hilda_loan_compounding_difference_l535_53590


namespace pineapple_rings_per_pineapple_l535_53555

def pineapples_purchased : Nat := 6
def cost_per_pineapple : Nat := 3
def rings_sold_per_set : Nat := 4
def price_per_set_of_4_rings : Nat := 5
def profit_made : Nat := 72

theorem pineapple_rings_per_pineapple : (90 / 5 * 4 / 6) = 12 := 
by 
  sorry

end pineapple_rings_per_pineapple_l535_53555


namespace total_students_l535_53597

-- Given conditions
variable (A B : ℕ)
noncomputable def M_A := 80 * A
noncomputable def M_B := 70 * B

axiom classA_condition1 : M_A - 160 = 90 * (A - 8)
axiom classB_condition1 : M_B - 180 = 85 * (B - 6)

-- Required proof in Lean 4 statement
theorem total_students : A + B = 78 :=
by
  sorry

end total_students_l535_53597


namespace minimum_value_y_l535_53542

variable {x y : ℝ}

theorem minimum_value_y (h : y * Real.log y = Real.exp (2 * x) - y * Real.log (2 * x)) : y ≥ Real.exp 1 :=
sorry

end minimum_value_y_l535_53542


namespace second_and_third_finish_job_together_in_8_days_l535_53511

theorem second_and_third_finish_job_together_in_8_days
  (x y : ℕ)
  (h1 : 1/24 + 1/x + 1/y = 1/6) :
  1/x + 1/y = 1/8 :=
by sorry

end second_and_third_finish_job_together_in_8_days_l535_53511


namespace remainder_of_polynomial_l535_53527

theorem remainder_of_polynomial (x : ℕ) :
  (x + 1) ^ 2021 % (x ^ 2 + x + 1) = 1 + x ^ 2 := 
by
  sorry

end remainder_of_polynomial_l535_53527


namespace range_of_a_l535_53520

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4*x + a

theorem range_of_a 
  (f : ℝ → ℝ → ℝ)
  (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x a ≥ 0) : 
  3 ≤ a :=
sorry

end range_of_a_l535_53520


namespace find_d_l535_53514

theorem find_d (y d : ℝ) (hy : y > 0) (h : (8 * y) / 20 + (3 * y) / d = 0.7 * y) : d = 10 :=
by
  sorry

end find_d_l535_53514


namespace cardinality_bound_l535_53500

theorem cardinality_bound {m n : ℕ} (hm : m > 1) (hn : n > 1)
  (S : Finset ℕ) (hS : S.card = n)
  (A : Fin m → Finset ℕ)
  (h : ∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → ∃ i, (x ∈ A i ∧ y ∉ (A i)) ∨ (x ∉ (A i) ∧ y ∈ A i)) :
  n ≤ 2^m :=
sorry

end cardinality_bound_l535_53500


namespace find_four_digit_number_l535_53565

def is_four_digit_number (k : ℕ) : Prop :=
  1000 ≤ k ∧ k < 10000

def appended_number (k : ℕ) : ℕ :=
  4000000 + k

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_four_digit_number (k : ℕ) (hk : is_four_digit_number k) :
  is_perfect_square (appended_number k) ↔ k = 4001 ∨ k = 8004 :=
sorry

end find_four_digit_number_l535_53565


namespace problem_solution_l535_53515

def otimes (a b : ℚ) : ℚ := (a ^ 3) / (b ^ 2)

theorem problem_solution :
  (otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4)) = (-2016) / 729 := by
  sorry

end problem_solution_l535_53515


namespace ivanov_entitled_to_12_million_rubles_l535_53585

def equal_contributions (x : ℝ) : Prop :=
  let ivanov_contribution := 70 * x
  let petrov_contribution := 40 * x
  let sidorov_contribution := 44
  ivanov_contribution = 44 ∧ petrov_contribution = 44 ∧ (ivanov_contribution + petrov_contribution + sidorov_contribution) / 3 = 44

def money_ivanov_receives (x : ℝ) : ℝ :=
  let ivanov_contribution := 70 * x
  ivanov_contribution - 44

theorem ivanov_entitled_to_12_million_rubles :
  ∃ x : ℝ, equal_contributions x → money_ivanov_receives x = 12 :=
sorry

end ivanov_entitled_to_12_million_rubles_l535_53585


namespace b_a_range_l535_53593
open Real

-- Definitions of angles A, B, and sides a, b in an acute triangle ABC we assume that these are given.
variables {A B C a b c : ℝ}
variable {ABC_acute : A + B + C = π}
variable {angle_condition : B = 2 * A}
variable {sides : a = b * (sin A / sin B)}

theorem b_a_range (h₁ : 0 < A) (h₂ : A < π/2) (h₃ : 0 < C) (h₄ : C < π/2) :
  (∃ A, 30 * (π/180) < A ∧ A < 45 * (π/180)) → 
  (∃ b a, b / a = 2 * cos A) → 
  (∃ x : ℝ, x = b / a ∧ sqrt 2 < x ∧ x < sqrt 3) :=
sorry

end b_a_range_l535_53593


namespace cost_of_each_fish_is_four_l535_53524

-- Definitions according to the conditions
def number_of_fish_given_to_dog := 40
def number_of_fish_given_to_cat := number_of_fish_given_to_dog / 2
def total_fish := number_of_fish_given_to_dog + number_of_fish_given_to_cat
def total_cost := 240
def cost_per_fish := total_cost / total_fish

-- The main statement / theorem that needs to be proved
theorem cost_of_each_fish_is_four :
  cost_per_fish = 4 :=
by
  sorry

end cost_of_each_fish_is_four_l535_53524


namespace least_gumballs_to_get_four_same_color_l535_53567

theorem least_gumballs_to_get_four_same_color
  (R W B : ℕ)
  (hR : R = 9)
  (hW : W = 7)
  (hB : B = 8) : 
  ∃ n, n = 10 ∧ (∀ m < n, ∀ r w b : ℕ, r + w + b = m → r < 4 ∧ w < 4 ∧ b < 4) ∧ 
  (∀ r w b : ℕ, r + w + b = n → r = 4 ∨ w = 4 ∨ b = 4) :=
sorry

end least_gumballs_to_get_four_same_color_l535_53567


namespace problem1_problem2_l535_53546

-- Definitions for the conditions
variables {A B C : ℝ}
variables {a b c S : ℝ}

-- Problem 1: Proving the value of side "a" given certain conditions
theorem problem1 (h₁ : S = (1 / 2) * a * b * Real.sin C) (h₂ : a^2 = 4 * Real.sqrt 3 * S)
  (h₃ : C = Real.pi / 3) (h₄ : b = 1) : a = 3 := by
  sorry

-- Problem 2: Proving the measure of angle "A" given certain conditions
theorem problem2 (h₁ : S = (1 / 2) * a * b * Real.sin C) (h₂ : a^2 = 4 * Real.sqrt 3 * S)
  (h₃ : c / b = 2 + Real.sqrt 3) : A = Real.pi / 3 := by
  sorry

end problem1_problem2_l535_53546


namespace arccos_sin_three_l535_53599

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 :=
by
  sorry

end arccos_sin_three_l535_53599


namespace quotient_of_division_l535_53529

theorem quotient_of_division (L S Q : ℕ) (h1 : L - S = 2500) (h2 : L = 2982) (h3 : L = Q * S + 15) : Q = 6 := 
sorry

end quotient_of_division_l535_53529


namespace value_of_f_at_minus_point_two_l535_53568

noncomputable def f (x : ℝ) : ℝ := 1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem value_of_f_at_minus_point_two : f (-0.2) = 0.81873 :=
by {
  sorry
}

end value_of_f_at_minus_point_two_l535_53568


namespace record_cost_calculation_l535_53562

theorem record_cost_calculation :
  ∀ (books_owned book_price records_bought money_left total_selling_price money_spent_per_record record_cost : ℕ),
  books_owned = 200 →
  book_price = 3 / 2 →
  records_bought = 75 →
  money_left = 75 →
  total_selling_price = books_owned * book_price →
  money_spent_per_record = total_selling_price - money_left →
  record_cost = money_spent_per_record / records_bought →
  record_cost = 3 :=
by
  intros books_owned book_price records_bought money_left total_selling_price money_spent_per_record record_cost
  sorry

end record_cost_calculation_l535_53562


namespace convex_polygon_diagonals_25_convex_polygon_triangles_25_l535_53536

-- Define a convex polygon with 25 sides
def convex_polygon_sides : ℕ := 25

-- Define the number of diagonals in a convex polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Define the number of triangles that can be formed by choosing any three vertices from n vertices
def number_of_triangles (n : ℕ) : ℕ := n.choose 3

-- Theorem to prove the number of diagonals is 275 for a convex polygon with 25 sides
theorem convex_polygon_diagonals_25 : number_of_diagonals convex_polygon_sides = 275 :=
by sorry

-- Theorem to prove the number of triangles is 2300 for a convex polygon with 25 sides
theorem convex_polygon_triangles_25 : number_of_triangles convex_polygon_sides = 2300 :=
by sorry

end convex_polygon_diagonals_25_convex_polygon_triangles_25_l535_53536


namespace trips_to_collect_all_trays_l535_53596

-- Definition of conditions
def trays_at_once : ℕ := 7
def trays_one_table : ℕ := 23
def trays_other_table : ℕ := 5

-- Theorem statement
theorem trips_to_collect_all_trays : 
  (trays_one_table / trays_at_once) + (if trays_one_table % trays_at_once = 0 then 0 else 1) + 
  (trays_other_table / trays_at_once) + (if trays_other_table % trays_at_once = 0 then 0 else 1) = 5 := 
by
  sorry

end trips_to_collect_all_trays_l535_53596


namespace number_of_children_at_reunion_l535_53517

theorem number_of_children_at_reunion (A C : ℕ) 
    (h1 : 3 * A = C)
    (h2 : 2 * A / 3 = 10) : 
  C = 45 :=
by
  sorry

end number_of_children_at_reunion_l535_53517


namespace inverse_proportional_l535_53509

-- Define the variables and the condition
variables {R : Type*} [CommRing R] {x y k : R}
-- Assuming x and y are non-zero
variables (hx : x ≠ 0) (hy : y ≠ 0)

-- Define the constant product relationship
def product_constant (x y k : R) : Prop := x * y = k

-- The main statement that needs to be proved
theorem inverse_proportional (h : product_constant x y k) : 
  ∃ k, x * y = k :=
by sorry

end inverse_proportional_l535_53509


namespace Marilyn_end_caps_l535_53519

def starting_caps := 51
def shared_caps := 36
def ending_caps := starting_caps - shared_caps

theorem Marilyn_end_caps : ending_caps = 15 := by
  -- proof omitted
  sorry

end Marilyn_end_caps_l535_53519


namespace flat_rate_first_night_l535_53512

-- Definitions of conditions
def total_cost_sarah (f n : ℕ) := f + 3 * n = 210
def total_cost_mark (f n : ℕ) := f + 7 * n = 450

-- Main theorem to be proven
theorem flat_rate_first_night : 
  ∃ f n : ℕ, total_cost_sarah f n ∧ total_cost_mark f n ∧ f = 30 :=
by
  sorry

end flat_rate_first_night_l535_53512


namespace units_digit_of_2_to_the_10_l535_53559

theorem units_digit_of_2_to_the_10 : ∃ d : ℕ, (d < 10) ∧ (2^10 % 10 = d) ∧ (d == 4) :=
by {
  -- sorry to skip the proof
  sorry
}

end units_digit_of_2_to_the_10_l535_53559


namespace average_transformation_l535_53578

theorem average_transformation (a b c : ℝ) (h : (a + b + c) / 3 = 12) : ((2 * a + 1) + (2 * b + 2) + (2 * c + 3) + 2) / 4 = 20 :=
by
  sorry

end average_transformation_l535_53578


namespace c_share_l535_53579

theorem c_share (x y z a b c : ℝ) 
  (H1 : b = (65/100) * a)
  (H2 : c = (40/100) * a)
  (H3 : a + b + c = 328) : 
  c = 64 := 
sorry

end c_share_l535_53579


namespace granger_total_amount_l535_53505

-- Define the constants for the problem
def cost_spam := 3
def cost_peanut_butter := 5
def cost_bread := 2
def quantity_spam := 12
def quantity_peanut_butter := 3
def quantity_bread := 4

-- Define the total cost calculation
def total_cost := (quantity_spam * cost_spam) + (quantity_peanut_butter * cost_peanut_butter) + (quantity_bread * cost_bread)

-- The theorem we need to prove
theorem granger_total_amount : total_cost = 59 := by
  sorry

end granger_total_amount_l535_53505
