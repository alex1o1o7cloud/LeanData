import Mathlib

namespace river_length_l719_71967

theorem river_length :
  let still_water_speed := 10 -- Karen's paddling speed on still water in miles per hour
  let current_speed      := 4  -- River's current speed in miles per hour
  let time               := 2  -- Time it takes Karen to paddle up the river in hours
  let effective_speed    := still_water_speed - current_speed -- Karen's effective speed against the current
  effective_speed * time = 12 -- Length of the river in miles
:= by
  sorry

end river_length_l719_71967


namespace square_in_semicircle_l719_71929

theorem square_in_semicircle (Q : ℝ) (h1 : ∃ Q : ℝ, (Q^2 / 4) + Q^2 = 4) : Q = 4 * Real.sqrt 5 / 5 := sorry

end square_in_semicircle_l719_71929


namespace first_class_product_probability_l719_71996

theorem first_class_product_probability
  (defective_rate : ℝ) (first_class_rate_qualified : ℝ)
  (H_def_rate : defective_rate = 0.04)
  (H_first_class_rate_qualified : first_class_rate_qualified = 0.75) :
  (1 - defective_rate) * first_class_rate_qualified = 0.72 :=
by
  sorry

end first_class_product_probability_l719_71996


namespace incorrect_conclusion_l719_71935

-- Define the given parabola.
def parabola (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the conditions for the parabola.
def parabola_opens_upwards : Prop := ∀ x y : ℝ, parabola (x + y) = (x + y - 2)^2 + 1
def axis_of_symmetry : Prop := ∀ x : ℝ, parabola x = parabola (4 - x)
def vertex_coordinates : Prop := parabola 2 = 1 ∧ (parabola 2, 2) = (1, 2)
def behavior_when_x_less_than_2 : Prop := ∀ x : ℝ, x < 2 → parabola x < parabola (x + 1)

-- The statement that needs to be proven in Lean 4.
theorem incorrect_conclusion : ¬ behavior_when_x_less_than_2 :=
  by
  sorry

end incorrect_conclusion_l719_71935


namespace b_share_220_l719_71934

theorem b_share_220 (A B C : ℝ) (h1 : A = B + 40) (h2 : C = A + 30) (h3 : A + B + C = 770) : 
  B = 220 :=
by
  sorry

end b_share_220_l719_71934


namespace minimize_fence_perimeter_l719_71927

-- Define the area of the pen
def area (L W : ℝ) : ℝ := L * W

-- Define that only three sides of the fence need to be fenced
def perimeter (L W : ℝ) : ℝ := 2 * W + L

-- Given conditions
def A : ℝ := 54450  -- Area in square meters

-- The proof statement
theorem minimize_fence_perimeter :
  ∃ (L W : ℝ), 
  area L W = A ∧ 
  ∀ (L' W' : ℝ), area L' W' = A → perimeter L W ≤ perimeter L' W' ∧ L = 330 ∧ W = 165 :=
sorry

end minimize_fence_perimeter_l719_71927


namespace mean_correct_and_no_seven_l719_71933

-- Define the set of numbers.
def numbers : List ℕ := 
  [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

-- Define the arithmetic mean of the numbers in the set.
def arithmetic_mean (l : List ℕ) : ℕ := (l.sum / l.length)

-- Specify the mean value
def mean_value : ℕ := 109629012

-- State the theorem that the mean value is correct and does not contain the digit 7.
theorem mean_correct_and_no_seven : arithmetic_mean numbers = mean_value ∧ ¬ 7 ∈ (mean_value.digits 10) :=
  sorry

end mean_correct_and_no_seven_l719_71933


namespace true_false_question_count_l719_71977

theorem true_false_question_count (n : ℕ) (h : (1 / 3) * (1 / 2)^n = 1 / 12) : n = 2 := by
  sorry

end true_false_question_count_l719_71977


namespace furthest_distance_l719_71956

-- Definitions of point distances as given conditions
def PQ : ℝ := 13
def QR : ℝ := 11
def RS : ℝ := 14
def SP : ℝ := 12

-- Statement of the problem in Lean
theorem furthest_distance :
  ∃ (P Q R S : ℝ),
    |P - Q| = PQ ∧
    |Q - R| = QR ∧
    |R - S| = RS ∧
    |S - P| = SP ∧
    ∀ (a b : ℝ), a ≠ b →
      |a - b| ≤ 25 :=
sorry

end furthest_distance_l719_71956


namespace sequence_sum_l719_71902

theorem sequence_sum {A B C D E F G H I J : ℤ} (hD : D = 8)
    (h_sum1 : A + B + C + D = 45)
    (h_sum2 : B + C + D + E = 45)
    (h_sum3 : C + D + E + F = 45)
    (h_sum4 : D + E + F + G = 45)
    (h_sum5 : E + F + G + H = 45)
    (h_sum6 : F + G + H + I = 45)
    (h_sum7 : G + H + I + J = 45)
    (h_sum8 : H + I + J + A = 45)
    (h_sum9 : I + J + A + B = 45)
    (h_sum10 : J + A + B + C = 45) :
  A + J = 0 := 
sorry

end sequence_sum_l719_71902


namespace dice_probability_l719_71915

noncomputable def probability_same_face (throws : ℕ) (dice : ℕ) : ℚ :=
  1 - (1 - (1 / 6) ^ dice) ^ throws

theorem dice_probability : 
  probability_same_face 5 10 = 1 - (1 - (1 / 6) ^ 10) ^ 5 :=
by 
  sorry

end dice_probability_l719_71915


namespace max_discount_rate_l719_71990

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l719_71990


namespace min_x_y_sum_l719_71998

theorem min_x_y_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/(x+1) + 1/y = 1/2) : x + y ≥ 7 := 
by 
  sorry

end min_x_y_sum_l719_71998


namespace collinear_vectors_x_value_l719_71908

theorem collinear_vectors_x_value (x : ℝ) (a b : ℝ × ℝ) (h₁: a = (2, x)) (h₂: b = (1, 2))
  (h₃: ∃ k : ℝ, a = k • b) : x = 4 :=
by
  sorry

end collinear_vectors_x_value_l719_71908


namespace consecutive_odds_base_eqn_l719_71954

-- Given conditions
def isOdd (n : ℕ) : Prop := n % 2 = 1

variables {C D : ℕ}

theorem consecutive_odds_base_eqn (C_odd : isOdd C) (D_odd : isOdd D) (consec : D = C + 2)
    (base_eqn : 2 * C^2 + 4 * C + 3 + 6 * D + 5 = 10 * (C + D) + 7) :
    C + D = 16 :=
sorry

end consecutive_odds_base_eqn_l719_71954


namespace roots_sum_of_squares_l719_71983

theorem roots_sum_of_squares
  (p q r : ℝ)
  (h_roots : ∀ x, (3 * x^3 - 4 * x^2 + 3 * x + 7 = 0) → (x = p ∨ x = q ∨ x = r))
  (h_sum : p + q + r = 4 / 3)
  (h_prod_sum : p * q + q * r + r * p = 1)
  (h_prod : p * q * r = -7 / 3) :
  p^2 + q^2 + r^2 = -2 / 9 := 
sorry

end roots_sum_of_squares_l719_71983


namespace complex_division_l719_71920

theorem complex_division (z : ℂ) (hz : (3 + 4 * I) * z = 25) : z = 3 - 4 * I :=
sorry

end complex_division_l719_71920


namespace Sara_has_3194_quarters_in_the_end_l719_71981

theorem Sara_has_3194_quarters_in_the_end
  (initial_quarters : ℕ)
  (borrowed_quarters : ℕ)
  (initial_quarters_eq : initial_quarters = 4937)
  (borrowed_quarters_eq : borrowed_quarters = 1743) :
  initial_quarters - borrowed_quarters = 3194 := by
  sorry

end Sara_has_3194_quarters_in_the_end_l719_71981


namespace simplify_expression_l719_71930

variable {R : Type*} [CommRing R] (x y : R)

theorem simplify_expression :
  (x - 2 * y) * (x + 2 * y) - x * (x - y) = -4 * y ^ 2 + x * y :=
by
  sorry

end simplify_expression_l719_71930


namespace point_not_on_line_l719_71972

theorem point_not_on_line (a c : ℝ) (h : a * c > 0) : ¬(0 = 2500 * a + c) := by
  sorry

end point_not_on_line_l719_71972


namespace smallest_N_l719_71962

theorem smallest_N (l m n N : ℕ) (hl : l > 1) (hm : m > 1) (hn : n > 1) :
  (l - 1) * (m - 1) * (n - 1) = 231 → l * m * n = N → N = 384 :=
sorry

end smallest_N_l719_71962


namespace four_digit_number_divisible_by_36_l719_71936

theorem four_digit_number_divisible_by_36 (n : ℕ) (h₁ : ∃ k : ℕ, 6130 + n = 36 * k) 
  (h₂ : ∃ k : ℕ, 130 + n = 4 * k) 
  (h₃ : ∃ k : ℕ, (10 + n) = 9 * k) : n = 6 :=
sorry

end four_digit_number_divisible_by_36_l719_71936


namespace simplify_expression_l719_71924

theorem simplify_expression : (Real.sqrt (9 / 4) - Real.sqrt (4 / 9)) = 5 / 6 :=
by
  sorry

end simplify_expression_l719_71924


namespace sculpture_and_base_height_l719_71980

def height_sculpture : ℕ := 2 * 12 + 10
def height_base : ℕ := 8
def total_height : ℕ := 42

theorem sculpture_and_base_height :
  height_sculpture + height_base = total_height :=
by
  -- provide the necessary proof steps here
  sorry

end sculpture_and_base_height_l719_71980


namespace quadratic_distinct_real_roots_l719_71931

theorem quadratic_distinct_real_roots (m : ℝ) : 
  (m ≠ 0 ∧ m < 1 / 5) ↔ ∃ (x y : ℝ), x ≠ y ∧ m * x^2 - 2 * x + 5 = 0 ∧ m * y^2 - 2 * y + 5 = 0 :=
sorry

end quadratic_distinct_real_roots_l719_71931


namespace second_machine_time_l719_71919

theorem second_machine_time
  (machine1_rate : ℕ)
  (machine2_rate : ℕ)
  (combined_rate12 : ℕ)
  (combined_rate123 : ℕ)
  (rate3 : ℕ)
  (time3 : ℚ) :
  machine1_rate = 60 →
  machine2_rate = 120 →
  combined_rate12 = 200 →
  combined_rate123 = 600 →
  rate3 = 420 →
  time3 = 10 / 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_machine_time_l719_71919


namespace total_snowfall_l719_71947

theorem total_snowfall (morning_snowfall : ℝ) (afternoon_snowfall : ℝ) (h_morning : morning_snowfall = 0.125) (h_afternoon : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.625 :=
by 
  sorry

end total_snowfall_l719_71947


namespace card_total_l719_71938

theorem card_total (Brenda Janet Mara : ℕ)
  (h1 : Janet = Brenda + 9)
  (h2 : Mara = 2 * Janet)
  (h3 : Mara = 150 - 40) :
  Brenda + Janet + Mara = 211 := by
  sorry

end card_total_l719_71938


namespace axis_of_symmetry_parabola_l719_71960

/-- If a parabola passes through points A(-2,0) and B(4,0), then the axis of symmetry of the parabola is the line x = 1. -/
theorem axis_of_symmetry_parabola (x : ℝ → ℝ) (hA : x (-2) = 0) (hB : x 4 = 0) : 
  ∃ c : ℝ, c = 1 ∧ ∀ y : ℝ, x y = x (2 * c - y) :=
sorry

end axis_of_symmetry_parabola_l719_71960


namespace parallelogram_proof_l719_71974

noncomputable def parallelogram_ratio (AP AB AQ AD AC AT : ℝ) (hP : AP / AB = 61 / 2022) (hQ : AQ / AD = 61 / 2065) (h_intersect : true) : ℕ :=
if h : AC / AT = 4087 / 61 then 67 else 0

theorem parallelogram_proof :
  ∀ (ABCD : Type) (P : Type) (Q : Type) (T : Type) 
     (AP AB AQ AD AC AT : ℝ) 
     (hP : AP / AB = 61 / 2022) 
     (hQ : AQ / AD = 61 / 2065)
     (h_intersect : true),
  parallelogram_ratio AP AB AQ AD AC AT hP hQ h_intersect = 67 :=
by
  sorry

end parallelogram_proof_l719_71974


namespace investment_c_is_correct_l719_71986

-- Define the investments of a and b
def investment_a : ℕ := 45000
def investment_b : ℕ := 63000
def profit_c : ℕ := 24000
def total_profit : ℕ := 60000

-- Define the equation to find the investment of c
def proportional_share (x y total : ℕ) : Prop :=
  2 * (x + y + total) = 5 * total

-- The theorem to prove c's investment given the conditions
theorem investment_c_is_correct (c : ℕ) (h_proportional: proportional_share investment_a investment_b c) :
  c = 72000 :=
by
  sorry

end investment_c_is_correct_l719_71986


namespace log_mul_l719_71942

theorem log_mul (a M N : ℝ) (ha_pos : 0 < a) (hM_pos : 0 < M) (hN_pos : 0 < N) (ha_ne_one : a ≠ 1) :
    Real.log (M * N) / Real.log a = Real.log M / Real.log a + Real.log N / Real.log a := by
  sorry

end log_mul_l719_71942


namespace megatek_employees_in_manufacturing_l719_71949

theorem megatek_employees_in_manufacturing :
  let total_degrees := 360
  let manufacturing_degrees := 108
  (manufacturing_degrees / total_degrees.toFloat) * 100 = 30 := 
by
  sorry

end megatek_employees_in_manufacturing_l719_71949


namespace final_price_after_discounts_l719_71939

noncomputable def initial_price : ℝ := 9795.3216374269
noncomputable def discount_20 (p : ℝ) : ℝ := p * 0.80
noncomputable def discount_10 (p : ℝ) : ℝ := p * 0.90
noncomputable def discount_5 (p : ℝ) : ℝ := p * 0.95

theorem final_price_after_discounts : discount_5 (discount_10 (discount_20 initial_price)) = 6700 := 
by
  sorry

end final_price_after_discounts_l719_71939


namespace totalCandies_l719_71917

def bobCandies : Nat := 10
def maryCandies : Nat := 5
def sueCandies : Nat := 20
def johnCandies : Nat := 5
def samCandies : Nat := 10

theorem totalCandies : bobCandies + maryCandies + sueCandies + johnCandies + samCandies = 50 := 
by
  sorry

end totalCandies_l719_71917


namespace gain_percentage_l719_71961

theorem gain_percentage (MP CP : ℝ) (h1 : 0.90 * MP = 1.17 * CP) :
  (((MP - CP) / CP) * 100) = 30 := 
by
  sorry

end gain_percentage_l719_71961


namespace intersection_correct_l719_71952

def A (x : ℝ) : Prop := |x| > 4
def B (x : ℝ) : Prop := -2 < x ∧ x ≤ 6
def intersection (x : ℝ) : Prop := B x ∧ A x

theorem intersection_correct :
  ∀ x : ℝ, intersection x ↔ 4 < x ∧ x ≤ 6 := 
by
  sorry

end intersection_correct_l719_71952


namespace find_correct_average_of_numbers_l719_71971

variable (nums : List ℝ)
variable (n : ℕ) (avg_wrong avg_correct : ℝ) (wrong_val correct_val : ℝ)

noncomputable def correct_average (nums : List ℝ) (wrong_val correct_val : ℝ) : ℝ :=
  let correct_sum := nums.sum - wrong_val + correct_val
  correct_sum / nums.length

theorem find_correct_average_of_numbers
  (h₀ : n = 10)
  (h₁ : avg_wrong = 15)
  (h₂ : wrong_val = 26)
  (h₃ : correct_val = 36)
  (h₄ : avg_correct = 16)
  (nums : List ℝ) :
  avg_wrong * n - wrong_val + correct_val = avg_correct * n := 
sorry

end find_correct_average_of_numbers_l719_71971


namespace counterexample_exists_l719_71973

theorem counterexample_exists : ∃ n : ℕ, n ≥ 2 ∧ ¬ ∃ k : ℕ, 2 ^ 2 ^ n % (2 ^ n - 1) = 4 ^ k := 
by
  sorry

end counterexample_exists_l719_71973


namespace find_special_numbers_l719_71978

theorem find_special_numbers (N : ℕ) (hN : 100 ≤ N ∧ N < 1000) :
  (∀ k : ℕ, N^k % 1000 = N % 1000) ↔ (N = 376 ∨ N = 625) :=
by
  sorry

end find_special_numbers_l719_71978


namespace vector_subtraction_l719_71975

variable (a b : ℝ × ℝ)

def vector_calc (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem vector_subtraction :
  a = (2, 4) → b = (-1, 1) → vector_calc a b = (5, 7) := by
  intros ha hb
  simp [vector_calc]
  rw [ha, hb]
  simp
  sorry

end vector_subtraction_l719_71975


namespace treadmill_discount_percentage_l719_71993

theorem treadmill_discount_percentage
  (p_t : ℝ) -- original price of the treadmill
  (t_p : ℝ) -- total amount paid for treadmill and plates
  (p_plate : ℝ) -- price of each plate
  (n_plate : ℕ) -- number of plates
  (h_t : p_t = 1350)
  (h_tp : t_p = 1045)
  (h_p_plate : p_plate = 50)
  (h_n_plate : n_plate = 2) :
  ((p_t - (t_p - n_plate * p_plate)) / p_t) * 100 = 30 :=
by
  sorry

end treadmill_discount_percentage_l719_71993


namespace candy_division_l719_71916

def pieces_per_bag (total_candies : ℕ) (bags : ℕ) : ℕ :=
total_candies / bags

theorem candy_division : pieces_per_bag 42 2 = 21 :=
by
  sorry

end candy_division_l719_71916


namespace equilateral_triangle_stack_impossible_l719_71948

theorem equilateral_triangle_stack_impossible :
  ¬ ∃ n : ℕ, 3 * 55 = 6 * n :=
by
  sorry

end equilateral_triangle_stack_impossible_l719_71948


namespace value_of_f_2_plus_g_3_l719_71968

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 - 1

theorem value_of_f_2_plus_g_3 : f (2 + g 3) = 26 :=
by
  sorry

end value_of_f_2_plus_g_3_l719_71968


namespace solve_for_x_l719_71958

theorem solve_for_x (x : ℝ) (h : 2 * x - 5 = 15) : x = 10 :=
sorry

end solve_for_x_l719_71958


namespace conference_handshakes_l719_71905

-- Define the number of attendees at the conference
def attendees : ℕ := 10

-- Define the number of ways to choose 2 people from the attendees
-- This is equivalent to the combination formula C(10, 2)
def handshakes (n : ℕ) : ℕ := n.choose 2

-- Prove that the number of handshakes at the conference is 45
theorem conference_handshakes : handshakes attendees = 45 := by
  sorry

end conference_handshakes_l719_71905


namespace sequence_remainder_prime_l719_71906

theorem sequence_remainder_prime (p : ℕ) (hp : Nat.Prime p) (x : ℕ → ℕ)
  (h1 : ∀ i, 0 ≤ i ∧ i < p → x i = i)
  (h2 : ∀ n, n ≥ p → x n = x (n-1) + x (n-p)) :
  (x (p^3) % p) = p - 1 :=
sorry

end sequence_remainder_prime_l719_71906


namespace original_time_40_l719_71940

theorem original_time_40
  (S T : ℝ)
  (h1 : ∀ D : ℝ, D = S * T)
  (h2 : ∀ D : ℝ, D = 0.8 * S * (T + 10)) :
  T = 40 :=
by
  sorry

end original_time_40_l719_71940


namespace number_of_possible_winning_scores_l719_71918

noncomputable def sum_of_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem number_of_possible_winning_scores : 
  let total_sum := sum_of_first_n_integers 12
  let max_possible_score := total_sum / 2
  let min_possible_score := sum_of_first_n_integers 6
  39 - 21 + 1 = 19 := 
by
  sorry

end number_of_possible_winning_scores_l719_71918


namespace incorrect_option_D_l719_71987

theorem incorrect_option_D (x y : ℝ) : y = (x - 2) ^ 2 + 1 → ¬ (∀ (x : ℝ), x < 2 → y < (x - 1) ^ 2 + 1) :=
by
  intro h
  sorry

end incorrect_option_D_l719_71987


namespace minimum_value_function_l719_71925

theorem minimum_value_function (x : ℝ) (h : x > 1) : 
  ∃ y, y = (16 - 2 * Real.sqrt 7) / 3 ∧ ∀ x > 1, (4*x^2 + 2*x + 5) / (x^2 + x + 1) ≥ y :=
sorry

end minimum_value_function_l719_71925


namespace fishing_boat_should_go_out_to_sea_l719_71976

def good_weather_profit : ℤ := 6000
def bad_weather_loss : ℤ := -8000
def stay_at_port_loss : ℤ := -1000

def prob_good_weather : ℚ := 0.6
def prob_bad_weather : ℚ := 0.4

def expected_profit_going : ℚ :=  prob_good_weather * good_weather_profit + prob_bad_weather * bad_weather_loss
def expected_profit_staying : ℚ := stay_at_port_loss

theorem fishing_boat_should_go_out_to_sea : 
  expected_profit_going > expected_profit_staying :=
  sorry

end fishing_boat_should_go_out_to_sea_l719_71976


namespace directrix_of_parabola_l719_71999

theorem directrix_of_parabola (p : ℝ) (hp : 2 * p = 4) : 
  (∃ x : ℝ, x = -1) :=
by
  sorry

end directrix_of_parabola_l719_71999


namespace boys_from_school_a_not_study_science_l719_71922

theorem boys_from_school_a_not_study_science (total_boys : ℕ) (boys_from_school_a_percentage : ℝ) (science_study_percentage : ℝ)
  (total_boys_in_camp : total_boys = 250) (school_a_percent : boys_from_school_a_percentage = 0.20) 
  (science_percent : science_study_percentage = 0.30) :
  ∃ (boys_from_school_a_not_science : ℕ), boys_from_school_a_not_science = 35 :=
by
  sorry

end boys_from_school_a_not_study_science_l719_71922


namespace people_with_uncool_parents_l719_71964

theorem people_with_uncool_parents :
  ∀ (total cool_dads cool_moms cool_both : ℕ),
    total = 50 →
    cool_dads = 25 →
    cool_moms = 30 →
    cool_both = 15 →
    (total - (cool_dads + cool_moms - cool_both)) = 10 := 
by
  intros total cool_dads cool_moms cool_both h1 h2 h3 h4
  sorry

end people_with_uncool_parents_l719_71964


namespace curved_surface_area_cone_l719_71903

-- Define the necessary values
def r := 8  -- radius of the base of the cone in centimeters
def l := 18 -- slant height of the cone in centimeters

-- Prove the curved surface area of the cone
theorem curved_surface_area_cone :
  (π * r * l = 144 * π) :=
by sorry

end curved_surface_area_cone_l719_71903


namespace value_of_p_l719_71932

noncomputable def third_term (x y : ℝ) := 45 * x^8 * y^2
noncomputable def fourth_term (x y : ℝ) := 120 * x^7 * y^3

theorem value_of_p (p q : ℝ) (h1 : third_term p q = fourth_term p q) (h2 : p + 2 * q = 1) (h3 : 0 < p) (h4 : 0 < q) : p = 4 / 7 :=
by
  have h : third_term p q = 45 * p^8 * q^2 := rfl
  have h' : fourth_term p q = 120 * p^7 * q^3 := rfl
  rw [h, h'] at h1
  sorry

end value_of_p_l719_71932


namespace product_of_roots_l719_71966

theorem product_of_roots (p q r : ℝ)
  (h1 : ∀ x : ℝ, (3 * x^3 - 9 * x^2 + 5 * x - 15 = 0) → (x = p ∨ x = q ∨ x = r)) :
  p * q * r = 5 := by
  sorry

end product_of_roots_l719_71966


namespace remainder_of_266_div_33_and_8_is_2_l719_71985

theorem remainder_of_266_div_33_and_8_is_2 :
  (266 % 33 = 2) ∧ (266 % 8 = 2) := by
  sorry

end remainder_of_266_div_33_and_8_is_2_l719_71985


namespace fill_in_blanks_problem1_fill_in_blanks_problem2_fill_in_blanks_problem3_l719_71951

def problem1_seq : List ℕ := [102, 101, 100, 99, 98, 97, 96]
def problem2_seq : List ℕ := [190, 180, 170, 160, 150, 140, 130, 120, 110, 100]
def problem3_seq : List ℕ := [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]

theorem fill_in_blanks_problem1 :
  ∃ (a b c d : ℕ), [102, a, 100, b, c, 97, d] = [102, 101, 100, 99, 98, 97, 96] :=
by
  exact ⟨101, 99, 98, 96, rfl⟩ -- Proof omitted with exact values

theorem fill_in_blanks_problem2 :
  ∃ (a b c d e f g : ℕ), [190, a, b, 160, c, d, e, 120, f, g] = [190, 180, 170, 160, 150, 140, 130, 120, 110, 100] :=
by
  exact ⟨180, 170, 150, 140, 130, 110, 100, rfl⟩ -- Proof omitted with exact values

theorem fill_in_blanks_problem3 :
  ∃ (a b c d e f : ℕ), [5000, a, 6000, b, 7000, c, d, e, f, 9500] = [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500] :=
by
  exact ⟨5500, 6500, 7500, 8000, 8500, 9000, rfl⟩ -- Proof omitted with exact values

end fill_in_blanks_problem1_fill_in_blanks_problem2_fill_in_blanks_problem3_l719_71951


namespace one_third_of_7_times_9_l719_71992

theorem one_third_of_7_times_9 : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_7_times_9_l719_71992


namespace roots_of_star_equation_l719_71913

def star (m n : ℝ) : ℝ := m * n^2 - m * n - 1

theorem roots_of_star_equation :
  ∀ x : ℝ, (star 1 x = 0) → (∃ a b : ℝ, a ≠ b ∧ x = a ∨ x = b) := 
by
  sorry

end roots_of_star_equation_l719_71913


namespace total_stamps_l719_71979

def c : ℕ := 578833
def bw : ℕ := 523776
def total : ℕ := 1102609

theorem total_stamps : c + bw = total := 
by 
  sorry

end total_stamps_l719_71979


namespace imaginary_part_of_z_l719_71944

-- Define the complex number z
def z : ℂ :=
  3 - 2 * Complex.I

-- Lean theorem statement to prove the imaginary part of z is -2
theorem imaginary_part_of_z :
  Complex.im z = -2 :=
by
  sorry

end imaginary_part_of_z_l719_71944


namespace geometric_sequence_b_value_l719_71995

theorem geometric_sequence_b_value (a b c : ℝ) (h : 1 * a = a * b ∧ a * b = b * c ∧ b * c = c * 5) : b = Real.sqrt 5 :=
sorry

end geometric_sequence_b_value_l719_71995


namespace paint_per_large_canvas_l719_71904

-- Define the conditions
variables (L : ℕ) (paint_large paint_small total_paint : ℕ)

-- Given conditions
def large_canvas_paint := 3 * L
def small_canvas_paint := 4 * 2
def total_paint_used := large_canvas_paint + small_canvas_paint

-- Statement that needs to be proven
theorem paint_per_large_canvas :
  total_paint_used = 17 → L = 3 :=
by
  intro h
  sorry

end paint_per_large_canvas_l719_71904


namespace sum_of_roots_l719_71970

theorem sum_of_roots (m n : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : ∀ x : ℝ, x^2 + m * x + n = 0 → (x = m ∨ x = n)) :
  m + n = -1 :=
sorry

end sum_of_roots_l719_71970


namespace impossible_digit_placement_l719_71953

-- Define the main variables and assumptions
variable (A B C : ℕ)
variable (h_sum : A + B = 45)
variable (h_segmentSum : 3 * A + B = 6 * C)

-- Define the impossible placement problem
theorem impossible_digit_placement :
  ¬(∃ A B C, A + B = 45 ∧ 3 * A + B = 6 * C ∧ 2 * A = 6 * C - 45) :=
by
  sorry

end impossible_digit_placement_l719_71953


namespace find_y_l719_71911

theorem find_y (x y : ℝ) (h1 : 0.5 * x = 0.25 * y - 30) (h2 : x = 690) : y = 1500 :=
by
  sorry

end find_y_l719_71911


namespace find_other_number_l719_71946

open BigOperators

noncomputable def other_number (n : ℕ) : Prop := n = 12

theorem find_other_number (n : ℕ) (h_lcm : Nat.lcm 8 n = 24) (h_hcf : Nat.gcd 8 n = 4) : other_number n := 
by
  sorry

end find_other_number_l719_71946


namespace quadratic_eq_m_neg1_l719_71984

theorem quadratic_eq_m_neg1 (m : ℝ) (h1 : (m - 3) ≠ 0) (h2 : m^2 - 2*m - 3 = 0) : m = -1 :=
sorry

end quadratic_eq_m_neg1_l719_71984


namespace cost_per_bag_l719_71928

theorem cost_per_bag (total_bags : ℕ) (sale_price_per_bag : ℕ) (desired_profit : ℕ) (total_revenue : ℕ)
  (total_cost : ℕ) (cost_per_bag : ℕ) :
  total_bags = 100 → sale_price_per_bag = 10 → desired_profit = 300 →
  total_revenue = total_bags * sale_price_per_bag →
  total_cost = total_revenue - desired_profit →
  cost_per_bag = total_cost / total_bags →
  cost_per_bag = 7 := by
  sorry

end cost_per_bag_l719_71928


namespace no_real_roots_x_squared_minus_x_plus_nine_l719_71926

theorem no_real_roots_x_squared_minus_x_plus_nine :
  ∀ x : ℝ, ¬ (x^2 - x + 9 = 0) :=
by 
  intro x 
  sorry

end no_real_roots_x_squared_minus_x_plus_nine_l719_71926


namespace kim_driving_speed_l719_71957

open Nat
open Real

noncomputable def driving_speed (distance there distance_back time_spent traveling_time total_time: ℝ) : ℝ :=
  (distance + distance_back) / traveling_time

theorem kim_driving_speed:
  ∀ (distance there distance_back time_spent traveling_time total_time: ℝ),
  distance = 30 →
  distance_back = 30 * 1.20 →
  total_time = 2 →
  time_spent = 0.5 →
  traveling_time = total_time - time_spent →
  driving_speed distance there distance_back time_spent traveling_time total_time = 44 :=
by
  intros
  simp only [driving_speed]
  sorry

end kim_driving_speed_l719_71957


namespace fraction_eggs_given_to_Sofia_l719_71997

variables (m : ℕ) -- Number of eggs Mia has
def Sofia_eggs := 3 * m
def Pablo_eggs := 4 * Sofia_eggs
def Lucas_eggs := 0

theorem fraction_eggs_given_to_Sofia (h1 : Pablo_eggs = 12 * m) :
  (1 : ℚ) / (12 : ℚ) = 1 / 12 := by sorry

end fraction_eggs_given_to_Sofia_l719_71997


namespace integer_solution_l719_71965

theorem integer_solution (n : ℤ) (h1 : n + 15 > 16) (h2 : -3 * n > -9) : n = 2 :=
by
  sorry

end integer_solution_l719_71965


namespace solve_fraction_l719_71912

variable (x y : ℝ)
variable (h1 : y > x)
variable (h2 : x > 0)
variable (h3 : x / y + y / x = 8)

theorem solve_fraction : (x + y) / (x - y) = Real.sqrt (5 / 3) :=
by
  sorry

end solve_fraction_l719_71912


namespace third_term_of_sequence_l719_71937

theorem third_term_of_sequence :
  (3 - (1 / 3) = 8 / 3) :=
by
  sorry

end third_term_of_sequence_l719_71937


namespace hoot_difference_l719_71988

def owl_hoot_rate : ℕ := 5
def heard_hoots_per_min : ℕ := 20
def owls_count : ℕ := 3

theorem hoot_difference :
  heard_hoots_per_min - (owls_count * owl_hoot_rate) = 5 := by
  sorry

end hoot_difference_l719_71988


namespace solve_quadratic_equation_l719_71945

noncomputable def f (x : ℝ) := 
  5 / (Real.sqrt (x - 9) - 8) - 
  2 / (Real.sqrt (x - 9) - 5) + 
  6 / (Real.sqrt (x - 9) + 5) - 
  9 / (Real.sqrt (x - 9) + 8)

theorem solve_quadratic_equation :
  ∀ (x : ℝ), x ≥ 9 → f x = 0 → 
  x = 19.2917 ∨ x = 8.9167 :=
by sorry

end solve_quadratic_equation_l719_71945


namespace second_difference_is_quadratic_l719_71989

theorem second_difference_is_quadratic (f : ℕ → ℝ) 
  (h : ∀ n : ℕ, (f (n + 2) - 2 * f (n + 1) + f n) = 2) :
  ∃ (a b : ℝ), ∀ (n : ℕ), f n = n^2 + a * n + b :=
by
  sorry

end second_difference_is_quadratic_l719_71989


namespace high_temp_three_years_same_l719_71950

theorem high_temp_three_years_same
  (T : ℝ)                               -- The high temperature for the three years with the same temperature
  (temp2017 : ℝ := 79)                   -- The high temperature for 2017
  (temp2016 : ℝ := 71)                   -- The high temperature for 2016
  (average_temp : ℝ := 84)               -- The average high temperature for 5 years
  (num_years : ℕ := 5)                   -- The number of years to consider
  (years_with_same_temp : ℕ := 3)        -- The number of years with the same high temperature
  (total_temp : ℝ := average_temp * num_years) -- The sum of the high temperatures for the 5 years
  (total_known_temp : ℝ := temp2017 + temp2016) -- The known high temperatures for 2016 and 2017
  (total_for_three_years : ℝ := total_temp - total_known_temp) -- Total high temperatures for the three years
  (high_temp_per_year : ℝ := total_for_three_years / years_with_same_temp) -- High temperature per year for three years
  :
  T = 90 :=
sorry

end high_temp_three_years_same_l719_71950


namespace find_positive_value_of_A_l719_71921

variable (A : ℝ)

-- Given conditions
def relation (A B : ℝ) : ℝ := A^2 + B^2

-- The proof statement
theorem find_positive_value_of_A (h : relation A 7 = 200) : A = Real.sqrt 151 := sorry

end find_positive_value_of_A_l719_71921


namespace triangle_constructibility_l719_71963

noncomputable def constructible_triangle (a b w_c : ℝ) : Prop :=
  (2 * a * b) / (a + b) > w_c

theorem triangle_constructibility {a b w_c : ℝ} (h : (a > 0) ∧ (b > 0) ∧ (w_c > 0)) :
  constructible_triangle a b w_c ↔ True :=
by
  sorry

end triangle_constructibility_l719_71963


namespace find_pq_l719_71901

noncomputable def find_k_squared (x y : ℝ) : ℝ :=
  let u1 := x^2 + y^2 - 12 * x + 16 * y - 160
  let u2 := x^2 + y^2 + 12 * x + 16 * y - 36
  let k_sq := 741 / 324
  k_sq

theorem find_pq : (741 + 324) = 1065 := by
  sorry

end find_pq_l719_71901


namespace find_constant_c_l719_71969

theorem find_constant_c : ∃ (c : ℝ), (∀ n : ℤ, c * (n:ℝ)^2 ≤ 3600) ∧ (∀ n : ℤ, n ≤ 5) ∧ (c = 144) :=
by
  sorry

end find_constant_c_l719_71969


namespace carla_order_cost_l719_71914

theorem carla_order_cost (base_cost : ℝ) (coupon : ℝ) (senior_discount_rate : ℝ)
  (additional_charge : ℝ) (tax_rate : ℝ) (conversion_rate : ℝ) :
  base_cost = 7.50 →
  coupon = 2.50 →
  senior_discount_rate = 0.20 →
  additional_charge = 1.00 →
  tax_rate = 0.08 →
  conversion_rate = 0.85 →
  (2 * (base_cost - coupon) * (1 - senior_discount_rate) + additional_charge) * (1 + tax_rate) * conversion_rate = 4.59 :=
by
  sorry

end carla_order_cost_l719_71914


namespace books_jerry_added_l719_71959

def initial_action_figures : ℕ := 7
def initial_books : ℕ := 2

theorem books_jerry_added (B : ℕ) (h : initial_action_figures = initial_books + B + 1) : B = 4 :=
by
  sorry

end books_jerry_added_l719_71959


namespace ratio_planes_bisect_volume_l719_71943

-- Definitions
def n : ℕ := 6
def m : ℕ := 20

-- Statement to prove
theorem ratio_planes_bisect_volume : (n / m : ℚ) = 3 / 10 := by
  sorry

end ratio_planes_bisect_volume_l719_71943


namespace smallest_integer_form_l719_71991

theorem smallest_integer_form (m n : ℤ) : ∃ (a : ℤ), a = 2011 * m + 55555 * n ∧ a > 0 → a = 1 :=
by
  sorry

end smallest_integer_form_l719_71991


namespace gemstones_needed_for_sets_l719_71982

-- Define the number of magnets per earring
def magnets_per_earring : ℕ := 2

-- Define the number of buttons per earring as half the number of magnets
def buttons_per_earring (magnets : ℕ) : ℕ := magnets / 2

-- Define the number of gemstones per earring as three times the number of buttons
def gemstones_per_earring (buttons : ℕ) : ℕ := 3 * buttons

-- Define the number of earrings per set
def earrings_per_set : ℕ := 2

-- Define the number of sets
def sets : ℕ := 4

-- Prove that Rebecca needs 24 gemstones for 4 sets of earrings given the conditions
theorem gemstones_needed_for_sets :
  gemstones_per_earring (buttons_per_earring magnets_per_earring) * earrings_per_set * sets = 24 :=
by
  sorry

end gemstones_needed_for_sets_l719_71982


namespace non_juniors_play_instrument_l719_71994

theorem non_juniors_play_instrument (total_students juniors non_juniors play_instrument_juniors play_instrument_non_juniors total_do_not_play : ℝ) :
  total_students = 600 →
  play_instrument_juniors = 0.3 * juniors →
  play_instrument_non_juniors = 0.65 * non_juniors →
  total_do_not_play = 0.4 * total_students →
  0.7 * juniors + 0.35 * non_juniors = total_do_not_play →
  juniors + non_juniors = total_students →
  non_juniors * 0.65 = 334 :=
by
  sorry

end non_juniors_play_instrument_l719_71994


namespace intersection_M_N_l719_71955

open Set

variable (x : ℝ)
def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def N : Set ℝ := {-2, 0, 2}

theorem intersection_M_N : M ∩ N = {0, 2} := sorry

end intersection_M_N_l719_71955


namespace solve_for_x_l719_71923

theorem solve_for_x : ∀ (x : ℂ) (i : ℂ), i^2 = -1 → 3 - 2 * i * x = 6 + i * x → x = i :=
by
  intros x i hI2 hEq
  sorry

end solve_for_x_l719_71923


namespace product_of_two_even_numbers_is_even_product_of_two_odd_numbers_is_odd_product_of_even_and_odd_number_is_even_product_of_odd_and_even_number_is_even_l719_71907

-- Definition of even and odd numbers
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Theorem statements for each condition

-- Prove that the product of two even numbers is even
theorem product_of_two_even_numbers_is_even (a b : ℤ) :
  is_even a → is_even b → is_even (a * b) :=
by sorry

-- Prove that the product of two odd numbers is odd
theorem product_of_two_odd_numbers_is_odd (c d : ℤ) :
  is_odd c → is_odd d → is_odd (c * d) :=
by sorry

-- Prove that the product of one even and one odd number is even
theorem product_of_even_and_odd_number_is_even (e f : ℤ) :
  is_even e → is_odd f → is_even (e * f) :=
by sorry

-- Prove that the product of one odd and one even number is even
theorem product_of_odd_and_even_number_is_even (g h : ℤ) :
  is_odd g → is_even h → is_even (g * h) :=
by sorry

end product_of_two_even_numbers_is_even_product_of_two_odd_numbers_is_odd_product_of_even_and_odd_number_is_even_product_of_odd_and_even_number_is_even_l719_71907


namespace line_equation_l719_71909

theorem line_equation (a b : ℝ) 
  (h1 : -4 = (a + 0) / 2)
  (h2 : 6 = (0 + b) / 2) :
  (∀ x y : ℝ, y = (3 / 2) * (x + 4) → 3 * x - 2 * y + 24 = 0) :=
by
  sorry

end line_equation_l719_71909


namespace sum_fractions_l719_71941

theorem sum_fractions:
  (Finset.range 16).sum (λ k => (k + 1) / 7) = 136 / 7 := by
  sorry

end sum_fractions_l719_71941


namespace find_x_l719_71900

-- define initial quantities of apples and oranges
def initial_apples (x : ℕ) : ℕ := 3 * x + 1
def initial_oranges (x : ℕ) : ℕ := 4 * x + 12

-- define the condition that the number of oranges is twice the number of apples
def condition (x : ℕ) : Prop := initial_oranges x = 2 * initial_apples x

-- define the final state
def final_apples : ℕ := 1
def final_oranges : ℕ := 12

-- theorem to prove that the number of times is 5
theorem find_x : ∃ x : ℕ, condition x ∧ final_apples = 1 ∧ final_oranges = 12 :=
by
  use 5
  sorry

end find_x_l719_71900


namespace molecular_weight_l719_71910

noncomputable def molecular_weight_of_one_mole : ℕ → ℝ :=
  fun n => if n = 1 then 78 else n * 78

theorem molecular_weight (n: ℕ) (hn: n > 0) (condition: ∃ k: ℕ, k = 4 ∧ 312 = k * 78) :
  molecular_weight_of_one_mole n = 78 * n :=
by
  sorry

end molecular_weight_l719_71910
