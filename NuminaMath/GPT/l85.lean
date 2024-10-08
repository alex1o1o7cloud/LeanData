import Mathlib

namespace sum_of_factors_1656_l85_85148

theorem sum_of_factors_1656 : ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 1656 ∧ a + b = 110 := by
  sorry

end sum_of_factors_1656_l85_85148


namespace find_other_solution_l85_85842

theorem find_other_solution (x : ℚ) (hx : 45 * (2 / 5 : ℚ)^2 + 22 = 56 * (2 / 5 : ℚ) - 9) : x = 7 / 9 :=
by 
  sorry

end find_other_solution_l85_85842


namespace find_num_white_balls_l85_85033

theorem find_num_white_balls
  (W : ℕ)
  (total_balls : ℕ := 15 + W)
  (prob_black : ℚ := 7 / total_balls)
  (given_prob : ℚ := 0.38095238095238093) :
  prob_black = given_prob → W = 3 :=
by
  intro h
  sorry

end find_num_white_balls_l85_85033


namespace first_term_of_arithmetic_series_l85_85711

theorem first_term_of_arithmetic_series 
  (a d : ℝ)
  (h1 : 20 * (2 * a + 39 * d) = 600)
  (h2 : 20 * (2 * a + 119 * d) = 1800) :
  a = 0.375 :=
by
  sorry

end first_term_of_arithmetic_series_l85_85711


namespace daily_sales_volume_relationship_maximize_daily_sales_profit_l85_85627

variables (x : ℝ) (y : ℝ) (P : ℝ)

-- Conditions
def cost_per_box : ℝ := 40
def min_selling_price : ℝ := 45
def initial_selling_price : ℝ := 45
def initial_sales_volume : ℝ := 700
def decrease_in_sales_volume_per_dollar : ℝ := 20

-- The functional relationship between y and x
theorem daily_sales_volume_relationship (hx : min_selling_price ≤ x ∧ x < 80) : y = -20 * x + 1600 := by
  sorry

-- The profit function
def profit_function (x : ℝ) := (x - cost_per_box) * (initial_sales_volume - decrease_in_sales_volume_per_dollar * (x - initial_selling_price))

-- Maximizing the profit
theorem maximize_daily_sales_profit : ∃ x_max, x_max = 60 ∧ P = profit_function 60 ∧ P = 8000 := by
  sorry

end daily_sales_volume_relationship_maximize_daily_sales_profit_l85_85627


namespace smallest_possible_value_of_c_l85_85328

theorem smallest_possible_value_of_c (b c : ℝ) (h1 : 1 < b) (h2 : b < c)
    (h3 : ¬∃ (u v w : ℝ), u = 1 ∧ v = b ∧ w = c ∧ u + v > w ∧ u + w > v ∧ v + w > u)
    (h4 : ¬∃ (x y z : ℝ), x = 1 ∧ y = 1/b ∧ z = 1/c ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
    c = (5 + Real.sqrt 5) / 2 :=
by
  sorry

end smallest_possible_value_of_c_l85_85328


namespace Sues_necklace_total_beads_l85_85176

theorem Sues_necklace_total_beads 
  (purple_beads : ℕ)
  (blue_beads : ℕ)
  (green_beads : ℕ)
  (h1 : purple_beads = 7)
  (h2 : blue_beads = 2 * purple_beads)
  (h3 : green_beads = blue_beads + 11) :
  purple_beads + blue_beads + green_beads = 46 :=
by
  sorry

end Sues_necklace_total_beads_l85_85176


namespace distance_from_point_to_asymptote_l85_85584

theorem distance_from_point_to_asymptote :
  ∃ (d : ℝ), ∀ (x₀ y₀ : ℝ), (x₀, y₀) = (3, 0) ∧ 3 * x₀ - 4 * y₀ = 0 →
  d = 9 / 5 :=
by
  sorry

end distance_from_point_to_asymptote_l85_85584


namespace total_score_is_248_l85_85684

def geography_score : ℕ := 50
def math_score : ℕ := 70
def english_score : ℕ := 66

def history_score : ℕ := (geography_score + math_score + english_score) / 3

theorem total_score_is_248 : geography_score + math_score + english_score + history_score = 248 := by
  -- proofs go here
  sorry

end total_score_is_248_l85_85684


namespace production_volume_bounds_l85_85230

theorem production_volume_bounds:
  ∀ (x : ℕ),
  (10 * x ≤ 800 * 2400) ∧ 
  (10 * x ≤ 4000000 + 16000000) ∧
  (x ≥ 1800000) →
  (1800000 ≤ x ∧ x ≤ 1920000) :=
by
  sorry

end production_volume_bounds_l85_85230


namespace find_function_satisfying_condition_l85_85282

theorem find_function_satisfying_condition :
  ∃ c : ℝ, ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (f x + 2 * y) = 6 * x + f (f y - x)) → 
                          (∀ x : ℝ, f x = 2 * x + c) :=
sorry

end find_function_satisfying_condition_l85_85282


namespace jake_has_fewer_peaches_than_steven_l85_85350

theorem jake_has_fewer_peaches_than_steven :
  ∀ (jillPeaches jakePeaches stevenPeaches : ℕ),
    jillPeaches = 12 →
    jakePeaches = jillPeaches - 1 →
    stevenPeaches = jillPeaches + 15 →
    stevenPeaches - jakePeaches = 16 :=
  by
    intros jillPeaches jakePeaches stevenPeaches
    intro h_jill
    intro h_jake
    intro h_steven
    sorry

end jake_has_fewer_peaches_than_steven_l85_85350


namespace remainder_of_n_div_11_is_1_l85_85345

def A : ℕ := 20072009
def n : ℕ := 100 * A

theorem remainder_of_n_div_11_is_1 :
  (n % 11) = 1 :=
sorry

end remainder_of_n_div_11_is_1_l85_85345


namespace probability_of_stock_price_increase_l85_85310

namespace StockPriceProbability

variables (P_A P_B P_C P_D_given_A P_D_given_B P_D_given_C : ℝ)

def P_D : ℝ := P_A * P_D_given_A + P_B * P_D_given_B + P_C * P_D_given_C

theorem probability_of_stock_price_increase :
    P_A = 0.6 → P_B = 0.3 → P_C = 0.1 → 
    P_D_given_A = 0.7 → P_D_given_B = 0.2 → P_D_given_C = 0.1 → 
    P_D P_A P_B P_C P_D_given_A P_D_given_B P_D_given_C = 0.49 :=
by intros h₁ h₂ h₃ h₄ h₅ h₆; sorry

end StockPriceProbability

end probability_of_stock_price_increase_l85_85310


namespace equivalent_contrapositive_l85_85269

-- Given definitions
variables {Person : Type} (possess : Person → Prop) (happy : Person → Prop)

-- The original statement: "If someone is happy, then they possess it."
def original_statement : Prop := ∀ p : Person, happy p → possess p

-- The contrapositive: "If someone does not possess it, then they are not happy."
def contrapositive_statement : Prop := ∀ p : Person, ¬ possess p → ¬ happy p

-- The theorem to prove logical equivalence
theorem equivalent_contrapositive : original_statement possess happy ↔ contrapositive_statement possess happy := 
by sorry

end equivalent_contrapositive_l85_85269


namespace increase_function_a_seq_increasing_b_seq_decreasing_seq_relation_l85_85117

open Real

-- Defining the sequences
noncomputable def a_seq (n : ℕ) : ℝ := (1 + (1 : ℝ) / n) ^ n
noncomputable def b_seq (n : ℕ) : ℝ := (1 + (1 : ℝ) / n) ^ (n + 1)

theorem increase_function (x : ℝ) (hx : 0 < x) : 
  ((1:ℝ) + 1 / x) ^ x < (1 + 1 / (x + 1)) ^ (x + 1) := sorry

theorem a_seq_increasing (n : ℕ) (hn : 0 < n) : 
  a_seq n < a_seq (n + 1) := sorry

theorem b_seq_decreasing (n : ℕ) (hn : 0 < n) : 
  b_seq (n + 1) < b_seq n := sorry

theorem seq_relation (n : ℕ) (hn : 0 < n) : 
  a_seq n < b_seq n := sorry

end increase_function_a_seq_increasing_b_seq_decreasing_seq_relation_l85_85117


namespace simple_interest_difference_l85_85243

theorem simple_interest_difference :
  let P : ℝ := 900
  let R1 : ℝ := 4
  let R2 : ℝ := 4.5
  let T : ℝ := 7
  let SI1 := P * R1 * T / 100
  let SI2 := P * R2 * T / 100
  SI2 - SI1 = 31.50 := by
  sorry

end simple_interest_difference_l85_85243


namespace total_students_in_school_l85_85042

theorem total_students_in_school (s : ℕ) (below_8 above_8 : ℕ) (students_8 : ℕ)
  (h1 : below_8 = 20 * s / 100) 
  (h2 : above_8 = 2 * students_8 / 3) 
  (h3 : students_8 = 48) 
  (h4 : s = students_8 + above_8 + below_8) : 
  s = 100 := 
by 
  sorry 

end total_students_in_school_l85_85042


namespace actual_distance_traveled_l85_85806

-- Definitions based on conditions
def original_speed : ℕ := 12
def increased_speed : ℕ := 20
def distance_difference : ℕ := 24

-- We need to prove the actual distance traveled by the person.
theorem actual_distance_traveled : 
  ∃ t : ℕ, increased_speed * t = original_speed * t + distance_difference → original_speed * t = 36 :=
by
  sorry

end actual_distance_traveled_l85_85806


namespace compare_neg_fractions_l85_85876

theorem compare_neg_fractions : (-2 / 3 : ℚ) < -3 / 5 :=
by
  sorry

end compare_neg_fractions_l85_85876


namespace mass_percentage_H_calculation_l85_85456

noncomputable def molar_mass_CaH2 : ℝ := 42.09
noncomputable def molar_mass_H2O : ℝ := 18.015
noncomputable def molar_mass_H2SO4 : ℝ := 98.079

noncomputable def moles_CaH2 : ℕ := 3
noncomputable def moles_H2O : ℕ := 4
noncomputable def moles_H2SO4 : ℕ := 2

noncomputable def mass_H_CaH2 : ℝ := 3 * 2 * 1.008
noncomputable def mass_H_H2O : ℝ := 4 * 2 * 1.008
noncomputable def mass_H_H2SO4 : ℝ := 2 * 2 * 1.008

noncomputable def total_mass_H : ℝ :=
  mass_H_CaH2 + mass_H_H2O + mass_H_H2SO4

noncomputable def total_mass_mixture : ℝ :=
  (moles_CaH2 * molar_mass_CaH2) + (moles_H2O * molar_mass_H2O) + (moles_H2SO4 * molar_mass_H2SO4)

noncomputable def mass_percentage_H : ℝ :=
  (total_mass_H / total_mass_mixture) * 100

theorem mass_percentage_H_calculation :
  abs (mass_percentage_H - 4.599) < 0.001 :=
by
  sorry

end mass_percentage_H_calculation_l85_85456


namespace remaining_area_correct_l85_85100

-- Define the side lengths of the large rectangle
def large_rectangle_length1 (x : ℝ) := 2 * x + 5
def large_rectangle_length2 (x : ℝ) := x + 8

-- Define the side lengths of the rectangular hole
def hole_length1 (x : ℝ) := 3 * x - 2
def hole_length2 (x : ℝ) := x + 1

-- Define the area of the large rectangle
def large_rectangle_area (x : ℝ) := (large_rectangle_length1 x) * (large_rectangle_length2 x)

-- Define the area of the hole
def hole_area (x : ℝ) := (hole_length1 x) * (hole_length2 x)

-- Prove the remaining area after accounting for the hole
theorem remaining_area_correct (x : ℝ) : 
  large_rectangle_area x - hole_area x = -x^2 + 20 * x + 42 := 
  by 
    sorry

end remaining_area_correct_l85_85100


namespace matches_length_l85_85743

-- Definitions and conditions
def area_shaded_figure : ℝ := 300 -- given in cm^2
def num_small_squares : ℕ := 8
def large_square_area_coefficient : ℕ := 4
def area_small_square (a : ℝ) : ℝ := num_small_squares * a + large_square_area_coefficient * a

-- Question and answer to be proven
theorem matches_length (a : ℝ) (side_length: ℝ) :
  area_shaded_figure = 300 → 
  area_small_square a = area_shaded_figure →
  (a = 25) →
  (side_length = 5) →
  4 * 7 * side_length = 140 :=
by
  intros h1 h2 h3 h4
  sorry

end matches_length_l85_85743


namespace rect_tiling_l85_85425

theorem rect_tiling (a b : ℕ) : ∃ (w h : ℕ), w = max 1 (2 * a) ∧ h = 2 * b ∧ (∃ f : ℕ → ℕ → (ℕ × ℕ), ∀ i j, (i < w ∧ j < h → f i j = (a, b))) := sorry

end rect_tiling_l85_85425


namespace factor_expression_l85_85647

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) := 
by
  sorry

end factor_expression_l85_85647


namespace vasya_result_correct_l85_85330

def num : ℕ := 10^1990 + (10^1989 * 6 - 1)
def denom : ℕ := 10 * (10^1989 * 6 - 1) + 4

theorem vasya_result_correct : (num / denom) = (1 / 4) := 
  sorry

end vasya_result_correct_l85_85330


namespace triangle_angles_l85_85111

-- Define the problem and the conditions as Lean statements.
theorem triangle_angles (x y z : ℝ) 
  (h1 : y + 150 + 160 = 360)
  (h2 : z + 150 + 160 = 360)
  (h3 : x + y + z = 180) : 
  x = 80 ∧ y = 50 ∧ z = 50 := 
by 
  sorry

end triangle_angles_l85_85111


namespace train_length_l85_85772

-- Definitions based on conditions
def train_speed_kmh := 54 -- speed of the train in km/h
def time_to_cross_sec := 16 -- time to cross the telegraph post in seconds
def kmh_to_ms (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 5 / 18 -- conversion factor from km/h to m/s

-- Prove that the length of the train is 240 meters
theorem train_length (h1 : train_speed_kmh = 54) (h2 : time_to_cross_sec = 16) : 
  (kmh_to_ms train_speed_kmh * time_to_cross_sec) = 240 := by
  sorry

end train_length_l85_85772


namespace find_z_solutions_l85_85862

theorem find_z_solutions (r : ℚ) (z : ℤ) (h : 2^z + 2 = r^2) : 
  (r = 2 ∧ z = 1) ∨ (r = -2 ∧ z = 1) ∨ (r = 3/2 ∧ z = -2) ∨ (r = -3/2 ∧ z = -2) :=
sorry

end find_z_solutions_l85_85862


namespace ratio_men_to_women_on_team_l85_85422

theorem ratio_men_to_women_on_team (M W : ℕ) 
  (h1 : W = M + 6) 
  (h2 : M + W = 24) : 
  M / W = 3 / 5 := 
by 
  sorry

end ratio_men_to_women_on_team_l85_85422


namespace sum_primes_upto_20_l85_85607

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def primes_upto_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_upto_20 : (primes_upto_20.sum = 77) :=
by
  sorry

end sum_primes_upto_20_l85_85607


namespace quadratic_equation_solution_l85_85974

noncomputable def findOrderPair (b d : ℝ) : Prop :=
  (b + d = 7) ∧ (b < d) ∧ (36 - 4 * b * d = 0)

theorem quadratic_equation_solution :
  ∃ b d : ℝ, findOrderPair b d ∧ (b, d) = ( (7 - Real.sqrt 13) / 2, (7 + Real.sqrt 13) / 2 ) :=
by
  sorry

end quadratic_equation_solution_l85_85974


namespace ellipse_equation_l85_85001

def major_axis_length (a : ℝ) := 2 * a = 8
def eccentricity (c a : ℝ) := c / a = 3 / 4

theorem ellipse_equation (a b c x y : ℝ) (h1 : major_axis_length a)
    (h2 : eccentricity c a) (h3 : b^2 = a^2 - c^2) :
    (x^2 / 16 + y^2 / 7 = 1 ∨ x^2 / 7 + y^2 / 16 = 1) :=
by
  sorry

end ellipse_equation_l85_85001


namespace five_letter_words_with_at_least_one_vowel_l85_85484

theorem five_letter_words_with_at_least_one_vowel :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F']
  let vowels := ['A', 'E', 'F']
  (6 ^ 5) - (3 ^ 5) = 7533 := by 
  sorry

end five_letter_words_with_at_least_one_vowel_l85_85484


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l85_85699

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l85_85699


namespace handshakes_max_number_of_men_l85_85659

theorem handshakes_max_number_of_men (n : ℕ) (h: n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end handshakes_max_number_of_men_l85_85659


namespace flower_shop_types_l85_85882

variable (C V T R F : ℕ)

-- Define the conditions
def condition1 : Prop := V = C / 3
def condition2 : Prop := T = V / 4
def condition3 : Prop := R = T
def condition4 : Prop := C = (2 / 3) * F

-- The main statement we need to prove: the shop stocks 4 types of flowers
theorem flower_shop_types
  (h1 : condition1 C V)
  (h2 : condition2 V T)
  (h3 : condition3 T R)
  (h4 : condition4 C F) :
  4 = 4 :=
by 
  sorry

end flower_shop_types_l85_85882


namespace math_solution_l85_85412

noncomputable def math_problem (x y z : ℝ) : Prop :=
  (0 ≤ x) ∧ (0 ≤ y) ∧ (0 ≤ z) ∧ (x + y + z = 1) → 
  (x^2 * y^2 + y^2 * z^2 + z^2 * x^2 + x^2 * y^2 * z^2 ≤ 1 / 16)

theorem math_solution (x y z : ℝ) :
  math_problem x y z := 
by
  sorry

end math_solution_l85_85412


namespace string_length_correct_l85_85938

noncomputable def cylinder_circumference : ℝ := 6
noncomputable def cylinder_height : ℝ := 18
noncomputable def number_of_loops : ℕ := 6

noncomputable def height_per_loop : ℝ := cylinder_height / number_of_loops
noncomputable def hypotenuse_per_loop : ℝ := Real.sqrt (cylinder_circumference ^ 2 + height_per_loop ^ 2)
noncomputable def total_string_length : ℝ := number_of_loops * hypotenuse_per_loop

theorem string_length_correct :
  total_string_length = 18 * Real.sqrt 5 := by
  sorry

end string_length_correct_l85_85938


namespace discriminant_of_quadratic_l85_85687

theorem discriminant_of_quadratic :
  let a := (5 : ℚ)
  let b := (5 + 1/5 : ℚ)
  let c := (1/5 : ℚ)
  let Δ := b^2 - 4 * a * c
  Δ = 576 / 25 :=
by
  sorry

end discriminant_of_quadratic_l85_85687


namespace stamp_book_gcd_l85_85958

theorem stamp_book_gcd (total1 total2 total3 : ℕ) 
    (h1 : total1 = 945) (h2 : total2 = 1260) (h3 : total3 = 630) : 
    ∃ d, d = Nat.gcd (Nat.gcd total1 total2) total3 ∧ d = 315 := 
by
  sorry

end stamp_book_gcd_l85_85958


namespace find_tan_G_l85_85231

def right_triangle (FG GH FH : ℕ) : Prop :=
  FG^2 = GH^2 + FH^2

def tan_ratio (GH FH : ℕ) : ℚ :=
  FH / GH

theorem find_tan_G
  (FG GH : ℕ)
  (H1 : FG = 13)
  (H2 : GH = 12)
  (FH : ℕ)
  (H3 : right_triangle FG GH FH) :
  tan_ratio GH FH = 5 / 12 :=
by
  sorry

end find_tan_G_l85_85231


namespace Jovana_final_addition_l85_85857

theorem Jovana_final_addition 
  (initial_amount added_initial removed final_amount x : ℕ)
  (h1 : initial_amount = 5)
  (h2 : added_initial = 9)
  (h3 : removed = 2)
  (h4 : final_amount = 28) :
  final_amount = initial_amount + added_initial - removed + x → x = 16 :=
by
  intros h
  sorry

end Jovana_final_addition_l85_85857


namespace domain_of_sqrt_function_l85_85701

theorem domain_of_sqrt_function (f : ℝ → ℝ) (x : ℝ) 
  (h1 : ∀ x, (1 / (Real.log x) - 2) ≥ 0) 
  (h2 : ∀ x, Real.log x ≠ 0) : 
  (1 < x ∧ x ≤ Real.sqrt 10) ↔ (∀ x, 0 < Real.log x ∧ Real.log x ≤ 1 / 2) := 
  sorry

end domain_of_sqrt_function_l85_85701


namespace find_f_zero_l85_85172

theorem find_f_zero (f : ℝ → ℝ) (h : ∀ x, f ((x + 1) / (x - 1)) = x^2 + 3) : f 0 = 4 :=
by
  -- The proof goes here.
  sorry

end find_f_zero_l85_85172


namespace rhombus_area_l85_85831

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 18) (h2 : d2 = 14) : 
  (d1 * d2) / 2 = 126 := 
  by sorry

end rhombus_area_l85_85831


namespace simplify_complex_fraction_l85_85897

theorem simplify_complex_fraction : 
  (6 - 3 * Complex.I) / (-2 + 5 * Complex.I) = (-27 / 29) - (24 / 29) * Complex.I := 
by 
  sorry

end simplify_complex_fraction_l85_85897


namespace grill_run_time_l85_85239

def time_burn (coals : ℕ) (burn_rate : ℕ) (interval : ℕ) : ℚ :=
  (coals / burn_rate) * interval

theorem grill_run_time :
  let time_a1 := time_burn 60 15 20
  let time_a2 := time_burn 75 12 20
  let time_a3 := time_burn 45 15 20
  let time_b1 := time_burn 50 10 30
  let time_b2 := time_burn 70 8 30
  let time_b3 := time_burn 40 10 30
  let time_b4 := time_burn 80 8 30
  time_a1 + time_a2 + time_a3 + time_b1 + time_b2 + time_b3 + time_b4 = 1097.5 := sorry

end grill_run_time_l85_85239


namespace root_equivalence_l85_85475

theorem root_equivalence (a_1 a_2 a_3 b : ℝ) :
  (∃ c_1 c_2 c_3 : ℝ, c_1 ≠ c_2 ∧ c_2 ≠ c_3 ∧ c_1 ≠ c_3 ∧
    (∀ x : ℝ, (x - a_1) * (x - a_2) * (x - a_3) = b ↔ (x = c_1 ∨ x = c_2 ∨ x = c_3))) →
  (∀ x : ℝ, (x + c_1) * (x + c_2) * (x + c_3) = b ↔ (x = -a_1 ∨ x = -a_2 ∨ x = -a_3)) :=
by 
  sorry

end root_equivalence_l85_85475


namespace value_of_ab_l85_85901

theorem value_of_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : ab = 8 :=
by
  sorry

end value_of_ab_l85_85901


namespace fraction_power_l85_85119

theorem fraction_power (a b : ℕ) (ha : a = 5) (hb : b = 6) : (a / b : ℚ) ^ 4 = 625 / 1296 := by
  sorry

end fraction_power_l85_85119


namespace find_angle_C_find_side_c_l85_85864

noncomputable section

-- Definitions and conditions for Part 1
def vectors_dot_product_sin_2C (A B C : ℝ) (m : ℝ × ℝ) (n : ℝ × ℝ) : Prop :=
  m = (Real.sin A, Real.cos A) ∧ n = (Real.cos B, Real.sin B) ∧ 
  ((m.1 * n.1 + m.2 * n.2) = Real.sin (2 * C))

def angles_of_triangle (A B C : ℝ) : Prop := 
  A + B + C = Real.pi

theorem find_angle_C (A B C : ℝ) (m n : ℝ × ℝ) :
  vectors_dot_product_sin_2C A B C m n → angles_of_triangle A B C → C = Real.pi / 3 :=
sorry

-- Definitions and conditions for Part 2
def sin_in_arithmetic_sequence (x y z : ℝ) : Prop :=
  x + z = 2 * y

def product_of_sides_cos_C (a b c : ℝ) (C : ℝ) : Prop :=
  (a * b * Real.cos C = 18) ∧ (Real.cos C = 1 / 2)

theorem find_side_c (A B C a b c : ℝ) (m n : ℝ × ℝ) :
  sin_in_arithmetic_sequence (Real.sin A) (Real.sin C) (Real.sin B) → 
  angles_of_triangle A B C → 
  product_of_sides_cos_C a b c C → 
  C = Real.pi / 3 → 
  c = 6 :=
sorry

end find_angle_C_find_side_c_l85_85864


namespace first_four_cards_all_red_l85_85779

noncomputable def probability_first_four_red_cards : ℚ :=
  (26 / 52) * (25 / 51) * (24 / 50) * (23 / 49)

theorem first_four_cards_all_red :
  probability_first_four_red_cards = 276 / 9801 :=
by
  -- The proof itself is not required; we are only stating it.
  sorry

end first_four_cards_all_red_l85_85779


namespace arithmetic_seq_a2_a8_a5_l85_85273

-- Define the sequence and sum conditions
variable {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Define the given conditions
axiom seq_condition (n : ℕ) : (1 - q) * S n + q * a n = 1
axiom q_nonzero : q * (q - 1) ≠ 0
axiom geom_seq : ∀ n, a n = q^(n - 1)

-- Main theorem (consistent with both parts (Ⅰ) and (Ⅱ) results)
theorem arithmetic_seq_a2_a8_a5 (S_arith : S 3 + S 6 = 2 * S 9) : a 2 + a 5 = 2 * a 8 :=
by
    sorry

end arithmetic_seq_a2_a8_a5_l85_85273


namespace last_digit_one_over_three_pow_neg_ten_l85_85525

theorem last_digit_one_over_three_pow_neg_ten : (3^10) % 10 = 9 := by
  sorry

end last_digit_one_over_three_pow_neg_ten_l85_85525


namespace min_value_of_polynomial_l85_85500

theorem min_value_of_polynomial (a : ℝ) : 
  (∀ x : ℝ, (2 * x^3 - 3 * x^2 + a) ≥ 5) → a = 6 :=
by
  sorry   -- Proof omitted

end min_value_of_polynomial_l85_85500


namespace angle_C_eq_pi_over_3_l85_85366

theorem angle_C_eq_pi_over_3 (a b c A B C : ℝ)
  (h : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)) :
  C = Real.pi / 3 :=
sorry

end angle_C_eq_pi_over_3_l85_85366


namespace measure_exterior_angle_BAC_l85_85587

-- Define the interior angle of a regular nonagon
def nonagon_interior_angle := (180 * (9 - 2)) / 9

-- Define the exterior angle of the nonagon
def nonagon_exterior_angle := 360 - nonagon_interior_angle

-- The square's interior angle
def square_interior_angle := 90

-- The question to be proven
theorem measure_exterior_angle_BAC :
  nonagon_exterior_angle - square_interior_angle = 130 :=
  by
  sorry

end measure_exterior_angle_BAC_l85_85587


namespace rectangle_dimension_l85_85250

theorem rectangle_dimension (x : ℝ) (h : (x^2) * (x + 5) = 3 * (2 * (x^2) + 2 * (x + 5))) : x = 3 :=
by
  have eq1 : (x^2) * (x + 5) = x^3 + 5 * x^2 := by ring
  have eq2 : 3 * (2 * (x^2) + 2 * (x + 5)) = 6 * x^2 + 6 * x + 30 := by ring
  rw [eq1, eq2] at h
  sorry  -- Proof details omitted

end rectangle_dimension_l85_85250


namespace common_rational_root_l85_85288

theorem common_rational_root (a b c d e f g : ℚ) (p : ℚ) :
  (48 * p^4 + a * p^3 + b * p^2 + c * p + 16 = 0) ∧
  (16 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 48 = 0) ∧
  (∃ m n : ℤ, p = m / n ∧ Int.gcd m n = 1 ∧ n ≠ 1 ∧ p < 0 ∧ n > 0) →
  p = -1/2 :=
by
  sorry

end common_rational_root_l85_85288


namespace solution_set_of_inequality_system_l85_85596

theorem solution_set_of_inequality_system (x : ℝ) : (x + 1 > 0) ∧ (-2 * x ≤ 6) ↔ (x > -1) := 
by 
  sorry

end solution_set_of_inequality_system_l85_85596


namespace quadratic_equiv_original_correct_transformation_l85_85442

theorem quadratic_equiv_original :
  (5 + 3*Real.sqrt 2) * x^2 + (3 + Real.sqrt 2) * x - 3 = 
  (7 + 4 * Real.sqrt 3) * x^2 + (2 + Real.sqrt 3) * x - 2 :=
sorry

theorem correct_transformation :
  ∃ r : ℝ, r = (9 / 7) - (4 * Real.sqrt 2 / 7) ∧ 
  ((5 + 3 * Real.sqrt 2) * x^2 + (3 + Real.sqrt 2) * x - 3) = 0 :=
sorry

end quadratic_equiv_original_correct_transformation_l85_85442


namespace candy_box_price_increase_l85_85816

theorem candy_box_price_increase
  (C : ℝ) -- Original price of the candy box
  (S : ℝ := 12) -- Original price of a can of soda
  (combined_price : C + S = 16) -- Combined price before increase
  (candy_box_increase : C + 0.25 * C = 1.25 * C) -- Price increase definition
  (soda_increase : S + 0.50 * S = 18) -- New price of soda after increase
  : 1.25 * C = 5 := sorry

end candy_box_price_increase_l85_85816


namespace terminating_decimal_expansion_l85_85803

theorem terminating_decimal_expansion (a b : ℝ) :
  (13 / 200 = a / 10^b) → a = 52 ∧ b = 3 ∧ a / 10^b = 0.052 :=
by sorry

end terminating_decimal_expansion_l85_85803


namespace find_x_for_vectors_l85_85708

theorem find_x_for_vectors
  (x : ℝ)
  (h1 : x ∈ Set.Icc 0 Real.pi)
  (a : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2)))
  (b : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2)))
  (h2 : (a.1 + b.1)^2 + (a.2 + b.2)^2 = 1) :
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 :=
by
  sorry

end find_x_for_vectors_l85_85708


namespace find_de_over_ef_l85_85960

-- Definitions based on problem conditions
variables {A B C D E F : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E] [AddCommGroup F] 
variables (a b c d e f : A) 
variables (α β γ δ : ℝ)

-- Conditions
-- AD:DB = 2:3
def d_def : A := (3 / 5) • a + (2 / 5) • b
-- BE:EC = 1:4
def e_def : A := (4 / 5) • b + (1 / 5) • c
-- Intersection F of DE and AC
def f_def : A := (5 • d) - (10 • e)

-- Target Proof
theorem find_de_over_ef (h_d: d = d_def a b) (h_e: e = e_def b c) (h_f: f = f_def d e):
  DE / EF = 1 / 5 := 
sorry

end find_de_over_ef_l85_85960


namespace relationship_among_values_l85_85339

-- Assume there exists a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Condition 1: f is strictly increasing on (0, 3)
def increasing_on_0_to_3 : Prop :=
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 3 → f x < f y

-- Condition 2: f(x + 3) is an even function
def even_function_shifted : Prop :=
  ∀ x : ℝ, f (x + 3) = f (-(x + 3))

-- The theorem we need to prove
theorem relationship_among_values 
  (h1 : increasing_on_0_to_3 f)
  (h2 : even_function_shifted f) :
  f (9/2) < f 2 ∧ f 2 < f (7/2) :=
sorry

end relationship_among_values_l85_85339


namespace fraction_of_credit_extended_l85_85092

noncomputable def C_total : ℝ := 342.857
noncomputable def P_auto : ℝ := 0.35
noncomputable def C_company : ℝ := 40

theorem fraction_of_credit_extended :
  (C_company / (C_total * P_auto)) = (1 / 3) :=
  by
    sorry

end fraction_of_credit_extended_l85_85092


namespace second_smallest_five_digit_in_pascal_l85_85322

theorem second_smallest_five_digit_in_pascal :
  ∃ (x : ℕ), (x > 10000) ∧ (∀ y : ℕ, (y ≠ 10000) → (y < x) → (y < 10000)) ∧ (x = 10001) :=
sorry

end second_smallest_five_digit_in_pascal_l85_85322


namespace future_cup_defensive_analysis_l85_85492

variables (avg_A : ℝ) (std_dev_A : ℝ) (avg_B : ℝ) (std_dev_B : ℝ)

-- Statement translations:
-- A: On average, Class B has better defensive skills than Class A.
def stat_A : Prop := avg_B < avg_A

-- C: Class B sometimes performs very well in defense, while other times it performs relatively poorly.
def stat_C : Prop := std_dev_B > std_dev_A

-- D: Class A rarely concedes goals.
def stat_D : Prop := avg_A <= 1.9 -- It's implied that 'rarely' indicates consistency and a lower average threshold, so this represents that.

theorem future_cup_defensive_analysis (h_avg_A : avg_A = 1.9) (h_std_dev_A : std_dev_A = 0.3) 
  (h_avg_B : avg_B = 1.3) (h_std_dev_B : std_dev_B = 1.2) :
  stat_A avg_A avg_B ∧ stat_C std_dev_A std_dev_B ∧ stat_D avg_A :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end future_cup_defensive_analysis_l85_85492


namespace expected_value_eight_l85_85327

-- Define the 10-sided die roll outcomes
def outcomes := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the value function for a roll outcome
def value (x : ℕ) : ℕ :=
  if x % 2 = 0 then x  -- even value
  else 2 * x  -- odd value

-- Calculate the expected value
def expected_value : ℚ :=
  (1 / 10 : ℚ) * (2 + 2 + 6 + 4 + 10 + 6 + 14 + 8 + 18 + 10)

-- The theorem stating the expected value equals 8
theorem expected_value_eight :
  expected_value = 8 := by
  sorry

end expected_value_eight_l85_85327


namespace complex_division_l85_85581

theorem complex_division (i : ℂ) (hi : i^2 = -1) : (2 * i) / (1 - i) = -1 + i :=
by sorry

end complex_division_l85_85581


namespace total_profit_l85_85693

-- Definitions based on the conditions
def tom_investment : ℝ := 30000
def tom_duration : ℝ := 12
def jose_investment : ℝ := 45000
def jose_duration : ℝ := 10
def jose_share_profit : ℝ := 25000

-- Theorem statement
theorem total_profit (tom_investment tom_duration jose_investment jose_duration jose_share_profit : ℝ) :
  (jose_share_profit / (jose_investment * jose_duration / (tom_investment * tom_duration + jose_investment * jose_duration)) = 5 / 9) →
  ∃ P : ℝ, P = 45000 :=
by
  sorry

end total_profit_l85_85693


namespace dima_age_l85_85936

variable (x : ℕ)

-- Dima's age is x years
def age_of_dima := x

-- Dima's age is twice his brother's age
def age_of_brother := x / 2

-- Dima's age is three times his sister's age
def age_of_sister := x / 3

-- The average age of Dima, his sister, and his brother is 11 years
def average_age := (x + age_of_brother x + age_of_sister x) / 3 = 11

theorem dima_age (h1 : age_of_brother x = x / 2) 
                 (h2 : age_of_sister x = x / 3) 
                 (h3 : average_age x) : x = 18 := 
by sorry

end dima_age_l85_85936


namespace symmetric_linear_functions_l85_85247

theorem symmetric_linear_functions :
  (∃ (a b : ℝ), ∀ x y : ℝ, (y = a * x + 2 ∧ y = 3 * x - b) → a = 1 / 3 ∧ b = 6) :=
by
  sorry

end symmetric_linear_functions_l85_85247


namespace beka_distance_l85_85151

theorem beka_distance (jackson_distance : ℕ) (beka_more_than_jackson : ℕ) :
  jackson_distance = 563 → beka_more_than_jackson = 310 → 
  (jackson_distance + beka_more_than_jackson = 873) :=
by
  sorry

end beka_distance_l85_85151


namespace profit_when_sold_at_double_price_l85_85751

-- Define the problem parameters

-- Assume cost price (CP)
def CP : ℕ := 100

-- Define initial selling price (SP) with 50% profit
def SP : ℕ := CP + (CP / 2)

-- Define new selling price when sold at double the initial selling price
def SP2 : ℕ := 2 * SP

-- Define profit when sold at SP2
def profit : ℕ := SP2 - CP

-- Define the percentage profit
def profit_percentage : ℕ := (profit * 100) / CP

-- The proof goal: if selling at double the price, percentage profit is 200%
theorem profit_when_sold_at_double_price : profit_percentage = 200 :=
by {sorry}

end profit_when_sold_at_double_price_l85_85751


namespace validate_assignment_l85_85909

-- Define the statements as conditions
def S1 := "x = x + 1"
def S2 := "b ="
def S3 := "x = y = 10"
def S4 := "x + y = 10"

-- A function to check if a statement is a valid assignment
def is_valid_assignment (s : String) : Prop :=
  s = S1

-- The theorem statement proving that S1 is the only valid assignment
theorem validate_assignment : is_valid_assignment S1 ∧
                              ¬is_valid_assignment S2 ∧
                              ¬is_valid_assignment S3 ∧
                              ¬is_valid_assignment S4 :=
by
  sorry

end validate_assignment_l85_85909


namespace prime_divides_3np_minus_3n1_l85_85236

theorem prime_divides_3np_minus_3n1 (p n : ℕ) (hp : Prime p) : p ∣ (3^(n + p) - 3^(n + 1)) :=
sorry

end prime_divides_3np_minus_3n1_l85_85236


namespace lisa_caffeine_l85_85780

theorem lisa_caffeine (caffeine_per_cup : ℕ) (daily_goal : ℕ) (cups_drank : ℕ) : caffeine_per_cup = 80 → daily_goal = 200 → cups_drank = 3 → (caffeine_per_cup * cups_drank - daily_goal) = 40 :=
by
  -- This is a theorem statement, thus no proof is provided here.
  sorry

end lisa_caffeine_l85_85780


namespace solve_equation_l85_85710

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * (x^2020)^(1/202) - 1 = 2020 * x → x = 1 :=
by
  sorry

end solve_equation_l85_85710


namespace largest_systematic_sample_l85_85187

theorem largest_systematic_sample {n_products interval start second_smallest max_sample : ℕ} 
  (h1 : n_products = 300) 
  (h2 : start = 2) 
  (h3 : second_smallest = 17) 
  (h4 : interval = second_smallest - start) 
  (h5 : n_products % interval = 0) 
  (h6 : max_sample = start + (interval * ((n_products / interval) - 1))) : 
  max_sample = 287 := 
by
  -- This is where the proof would go if required.
  sorry

end largest_systematic_sample_l85_85187


namespace bottles_stolen_at_dance_l85_85302

-- Define the initial conditions
def initial_bottles := 10
def bottles_lost_at_school := 2
def total_stickers := 21
def stickers_per_bottle := 3

-- Calculate remaining bottles after loss at school
def remaining_bottles_after_school := initial_bottles - bottles_lost_at_school

-- Calculate the remaining bottles after the theft
def remaining_bottles_after_theft := total_stickers / stickers_per_bottle

-- Prove the number of bottles stolen
theorem bottles_stolen_at_dance : remaining_bottles_after_school - remaining_bottles_after_theft = 1 :=
by
  sorry

end bottles_stolen_at_dance_l85_85302


namespace number_of_shirts_l85_85749

theorem number_of_shirts (ratio_pants_shirts: ℕ) (num_pants: ℕ) (S: ℕ) : 
  ratio_pants_shirts = 7 ∧ num_pants = 14 → S = 20 :=
by
  sorry

end number_of_shirts_l85_85749


namespace simplify_expression_l85_85834

theorem simplify_expression (x : ℝ) : 
  (x^3 * x^2 * x + (x^3)^2 + (-2 * x^2)^3) = -6 * x^6 := 
by 
  sorry

end simplify_expression_l85_85834


namespace john_got_80_percent_of_value_l85_85733

noncomputable def percentage_of_value (P : ℝ) : Prop :=
  let old_system_cost := 250
  let new_system_cost := 600
  let discount_percentage := 0.25
  let pocket_spent := 250
  let discount_amount := discount_percentage * new_system_cost
  let price_after_discount := new_system_cost - discount_amount
  let value_for_old_system := (P / 100) * old_system_cost
  value_for_old_system + pocket_spent = price_after_discount

theorem john_got_80_percent_of_value : percentage_of_value 80 :=
by
  sorry

end john_got_80_percent_of_value_l85_85733


namespace sum_of_possible_values_l85_85558

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 10) = -7) :
  ∃ N1 N2 : ℝ, (N1 * (N1 - 10) = -7 ∧ N2 * (N2 - 10) = -7) ∧ (N1 + N2 = 10) :=
sorry

end sum_of_possible_values_l85_85558


namespace fraction_subtraction_l85_85107

theorem fraction_subtraction : (1 / 6 : ℚ) - (5 / 12) = -1 / 4 := 
by sorry

end fraction_subtraction_l85_85107


namespace sum_of_coeff_l85_85401

theorem sum_of_coeff (x y : ℕ) (n : ℕ) (h : 2 * x + y = 3) : (2 * x + y) ^ n = 3^n := 
by
  sorry

end sum_of_coeff_l85_85401


namespace total_chairs_calculation_l85_85200

theorem total_chairs_calculation
  (chairs_per_trip : ℕ)
  (trips_per_student : ℕ)
  (total_students : ℕ)
  (h1 : chairs_per_trip = 5)
  (h2 : trips_per_student = 10)
  (h3 : total_students = 5) :
  total_students * (chairs_per_trip * trips_per_student) = 250 :=
by
  sorry

end total_chairs_calculation_l85_85200


namespace decagonal_pyramid_volume_l85_85296

noncomputable def volume_of_decagonal_pyramid (m : ℝ) (apex_angle : ℝ) : ℝ :=
  let sin18 := Real.sin (18 * Real.pi / 180)
  let sin36 := Real.sin (36 * Real.pi / 180)
  let cos18 := Real.cos (18 * Real.pi / 180)
  (5 * m^3 * sin36) / (3 * (1 + 2 * cos18))

theorem decagonal_pyramid_volume : volume_of_decagonal_pyramid 39 (18 * Real.pi / 180) = 20023 :=
  sorry

end decagonal_pyramid_volume_l85_85296


namespace range_of_x_l85_85696

theorem range_of_x (x : ℝ) (h : 4 * x - 12 ≥ 0) : x ≥ 3 := 
sorry

end range_of_x_l85_85696


namespace perimeter_change_l85_85953

theorem perimeter_change (s h : ℝ) 
  (h1 : 2 * (1.3 * s + 0.8 * h) = 2 * (s + h)) :
  (2 * (0.8 * s + 1.3 * h) = 1.1 * (2 * (s + h))) :=
by
  sorry

end perimeter_change_l85_85953


namespace handshake_problem_l85_85399

-- Defining the necessary elements:
def num_people : Nat := 12
def num_handshakes_per_person : Nat := num_people - 2

-- Defining the total number of handshakes. Each handshake is counted twice.
def total_handshakes : Nat := (num_people * num_handshakes_per_person) / 2

-- The theorem statement:
theorem handshake_problem : total_handshakes = 60 :=
by
  sorry

end handshake_problem_l85_85399


namespace initial_weight_of_cheese_l85_85356

theorem initial_weight_of_cheese :
  let initial_weight : Nat := 850
  -- final state after 3 bites
  let final_weight1 : Nat := 25
  let final_weight2 : Nat := 25
  -- third state
  let third_weight1 : Nat := final_weight1 + final_weight2
  let third_weight2 : Nat := final_weight1
  -- second state
  let second_weight1 : Nat := third_weight1 + third_weight2
  let second_weight2 : Nat := third_weight1
  -- first state
  let first_weight1 : Nat := second_weight1 + second_weight2
  let first_weight2 : Nat := second_weight1
  -- initial state
  let initial_weight1 : Nat := first_weight1 + first_weight2
  let initial_weight2 : Nat := first_weight1
  initial_weight = initial_weight1 + initial_weight2 :=
by
  sorry

end initial_weight_of_cheese_l85_85356


namespace fraction_of_phone_numbers_begin_with_8_and_end_with_5_l85_85946

theorem fraction_of_phone_numbers_begin_with_8_and_end_with_5 :
  let total_numbers := 7 * 10^7
  let specific_numbers := 10^6
  specific_numbers / total_numbers = 1 / 70 := by
  sorry

end fraction_of_phone_numbers_begin_with_8_and_end_with_5_l85_85946


namespace original_number_is_19_l85_85812

theorem original_number_is_19 (x : ℤ) (h : (x + 4) % 23 = 0) : x = 19 := 
by 
  sorry

end original_number_is_19_l85_85812


namespace triangle_side_length_b_l85_85656

theorem triangle_side_length_b (a b c : ℝ) (A B C : ℝ)
  (hB : B = 30) 
  (h_area : 1/2 * a * c * Real.sin (B * Real.pi/180) = 3/2) 
  (h_sine : Real.sin (A * Real.pi/180) + Real.sin (C * Real.pi/180) = 2 * Real.sin (B * Real.pi/180)) :
  b = Real.sqrt 3 + 1 :=
by
  sorry

end triangle_side_length_b_l85_85656


namespace paid_more_than_free_l85_85529

def num_men : ℕ := 194
def num_women : ℕ := 235
def free_admission : ℕ := 68
def total_people (num_men num_women : ℕ) : ℕ := num_men + num_women
def paid_admission (total_people free_admission : ℕ) : ℕ := total_people - free_admission
def paid_over_free (paid_admission free_admission : ℕ) : ℕ := paid_admission - free_admission

theorem paid_more_than_free :
  paid_over_free (paid_admission (total_people num_men num_women) free_admission) free_admission = 293 := 
by
  sorry

end paid_more_than_free_l85_85529


namespace train_distance_after_braking_l85_85988

theorem train_distance_after_braking : 
  (∃ t : ℝ, (27 * t - 0.45 * t^2 = 0) ∧ (∀ s : ℝ, s = 27 * t - 0.45 * t^2) ∧ s = 405) :=
sorry

end train_distance_after_braking_l85_85988


namespace machines_complete_job_in_12_days_l85_85555

-- Given the conditions
variable (D : ℕ) -- The number of days for 12 machines to complete the job
variable (h1 : (1 : ℚ) / ((12 : ℚ) * D) = (1 : ℚ) / ((18 : ℚ) * 8))

-- Prove the number of days for 12 machines to complete the job
theorem machines_complete_job_in_12_days (h1 : (1 : ℚ) / ((12 : ℚ) * D) = (1 : ℚ) / ((18 : ℚ) * 8)) : D = 12 :=
by
  sorry

end machines_complete_job_in_12_days_l85_85555


namespace neg_exists_equiv_forall_l85_85458

theorem neg_exists_equiv_forall :
  (¬ ∃ x : ℝ, x^2 - x + 4 < 0) ↔ (∀ x : ℝ, x^2 - x + 4 ≥ 0) :=
by
  sorry

end neg_exists_equiv_forall_l85_85458


namespace pyarelal_loss_l85_85360

/-
Problem statement:
Given the following conditions:
1. Ashok's capital is 1/9 of Pyarelal's.
2. Ashok experienced a loss of 12% on his investment.
3. Pyarelal's loss was 9% of his investment.
4. Their total combined loss is Rs. 2,100.

Prove that the loss incurred by Pyarelal is Rs. 1,829.32.
-/

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (ashok_ratio : ℝ) (ashok_loss_percent : ℝ) (pyarelal_loss_percent : ℝ)
  (h1 : ashok_ratio = (1 : ℝ) / 9)
  (h2 : ashok_loss_percent = 0.12)
  (h3 : pyarelal_loss_percent = 0.09)
  (h4 : total_loss = 2100)
  (h5 : total_loss = ashok_loss_percent * (P * ashok_ratio) + pyarelal_loss_percent * P) :
  pyarelal_loss_percent * P = 1829.32 :=
by
  sorry

end pyarelal_loss_l85_85360


namespace average_TV_sets_in_shops_l85_85907

def shop_a := 20
def shop_b := 30
def shop_c := 60
def shop_d := 80
def shop_e := 50
def total_shops := 5

theorem average_TV_sets_in_shops : (shop_a + shop_b + shop_c + shop_d + shop_e) / total_shops = 48 :=
by
  have h1 : shop_a + shop_b + shop_c + shop_d + shop_e = 240
  { sorry }
  have h2 : 240 / total_shops = 48
  { sorry }
  exact Eq.trans (congrArg (fun x => x / total_shops) h1) h2

end average_TV_sets_in_shops_l85_85907


namespace magnitude_of_root_of_quadratic_eq_l85_85881

open Complex

theorem magnitude_of_root_of_quadratic_eq (z : ℂ) 
  (h : z^2 - (2 : ℂ) * z + 2 = 0) : abs z = Real.sqrt 2 :=
by 
  sorry

end magnitude_of_root_of_quadratic_eq_l85_85881


namespace fraction_increase_by_five_l85_85346

variable (x y : ℝ)

theorem fraction_increase_by_five :
  let f := fun x y => (x * y) / (2 * x - 3 * y)
  f (5 * x) (5 * y) = 5 * (f x y) :=
by
  sorry

end fraction_increase_by_five_l85_85346


namespace number_of_terms_in_arithmetic_sequence_l85_85559

-- Define the first term, common difference, and the nth term of the sequence
def a : ℤ := -3
def d : ℤ := 4
def a_n : ℤ := 45

-- Define the number of terms in the arithmetic sequence
def num_of_terms : ℤ := 13

-- The theorem states that for the given arithmetic sequence, the number of terms n satisfies the sequence equation
theorem number_of_terms_in_arithmetic_sequence :
  a + (num_of_terms - 1) * d = a_n :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l85_85559


namespace project_completion_time_l85_85443

-- Definitions for conditions
def a_rate : ℚ := 1 / 20
def b_rate : ℚ := 1 / 30
def combined_rate : ℚ := a_rate + b_rate

-- Total days to complete the project
def total_days (x : ℚ) : Prop :=
  (x - 5) * a_rate + x * b_rate = 1

-- The theorem to be proven
theorem project_completion_time : ∃ (x : ℚ), total_days x ∧ x = 15 := by
  sorry

end project_completion_time_l85_85443


namespace number_of_ways_to_fulfill_order_l85_85791

open Finset Nat

/-- Bill must buy exactly eight donuts from a shop offering five types, 
with at least two of the first type and one of each of the other four types. 
Prove that there are exactly 15 different ways to fulfill this order. -/
theorem number_of_ways_to_fulfill_order : 
  let total_donuts := 8
  let types_of_donuts := 5
  let mandatory_first_type := 2
  let mandatory_each_other_type := 1
  let remaining_donuts := total_donuts - (mandatory_first_type + 4 * mandatory_each_other_type)
  let combinations := (remaining_donuts + types_of_donuts - 1).choose (types_of_donuts - 1)
  combinations = 15 := 
by
  sorry

end number_of_ways_to_fulfill_order_l85_85791


namespace calculation_l85_85590

theorem calculation (a b c d e : ℤ)
  (h1 : a = (-4)^6)
  (h2 : b = 4^4)
  (h3 : c = 2^5)
  (h4 : d = 7^2)
  (h5 : e = (a / b) + c - d) :
  e = -1 := by
  sorry

end calculation_l85_85590


namespace proportion_equation_l85_85613

theorem proportion_equation (x y : ℝ) (h : 3 * x = 4 * y) (hy : y ≠ 0) : (x / 4 = y / 3) :=
by
  sorry

end proportion_equation_l85_85613


namespace candidate_majority_votes_l85_85560

theorem candidate_majority_votes (total_votes : ℕ) (candidate_percentage other_percentage : ℕ) 
  (h_total_votes : total_votes = 5200)
  (h_candidate_percentage : candidate_percentage = 60)
  (h_other_percentage : other_percentage = 40) :
  (candidate_percentage * total_votes / 100) - (other_percentage * total_votes / 100) = 1040 := 
by
  sorry

end candidate_majority_votes_l85_85560


namespace find_points_per_enemy_l85_85096

def points_per_enemy (x : ℕ) : Prop :=
  let points_from_enemies := 6 * x
  let additional_points := 8
  let total_points := points_from_enemies + additional_points
  total_points = 62

theorem find_points_per_enemy (x : ℕ) (h : points_per_enemy x) : x = 9 :=
  by sorry

end find_points_per_enemy_l85_85096


namespace expenses_denoted_as_negative_l85_85855

theorem expenses_denoted_as_negative (income_yuan expenses_yuan : Int) (h : income_yuan = 6) : 
  expenses_yuan = -4 :=
by
  sorry

end expenses_denoted_as_negative_l85_85855


namespace incorrect_statements_count_l85_85955

-- Definitions of the statements
def statement1 : Prop := "The diameter perpendicular to the chord bisects the chord" = "incorrect"

def statement2 : Prop := "A circle is a symmetrical figure, and any diameter is its axis of symmetry" = "incorrect"

def statement3 : Prop := "Two arcs of equal length are congruent" = "incorrect"

-- Theorem stating that the number of incorrect statements is 3
theorem incorrect_statements_count : 
  (statement1 → False) → (statement2 → False) → (statement3 → False) → 3 = 3 :=
by sorry

end incorrect_statements_count_l85_85955


namespace sum_of_odd_base4_digits_of_152_and_345_l85_85283

def base_4_digit_count (n : ℕ) : ℕ :=
    n.digits 4 |>.filter (λ x => x % 2 = 1) |>.length

theorem sum_of_odd_base4_digits_of_152_and_345 :
    base_4_digit_count 152 + base_4_digit_count 345 = 6 :=
by
    sorry

end sum_of_odd_base4_digits_of_152_and_345_l85_85283


namespace problem_statement_l85_85175

theorem problem_statement (p q : Prop) :
  ¬(p ∧ q) ∧ ¬¬p → ¬q := 
by 
  sorry

end problem_statement_l85_85175


namespace max_value_of_g_l85_85047

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_of_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 23 :=
by
  sorry

end max_value_of_g_l85_85047


namespace smallest_solution_of_quartic_equation_l85_85321

theorem smallest_solution_of_quartic_equation :
  ∃ x : ℝ, (x^4 - 50 * x^2 + 625 = 0) ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := sorry

end smallest_solution_of_quartic_equation_l85_85321


namespace expression_evaluation_l85_85792

theorem expression_evaluation :
  5 * 423 + 4 * 423 + 3 * 423 + 421 = 5497 := by
  sorry

end expression_evaluation_l85_85792


namespace washing_machine_capacity_l85_85499

theorem washing_machine_capacity 
  (shirts : ℕ) (sweaters : ℕ) (loads : ℕ) (total_clothing : ℕ) (n : ℕ)
  (h1 : shirts = 43) (h2 : sweaters = 2) (h3 : loads = 9)
  (h4 : total_clothing = shirts + sweaters)
  (h5 : total_clothing / loads = n) :
  n = 5 :=
sorry

end washing_machine_capacity_l85_85499


namespace deer_families_initial_count_l85_85651

theorem deer_families_initial_count (stayed moved_out : ℕ) (h_stayed : stayed = 45) (h_moved_out : moved_out = 34) :
  stayed + moved_out = 79 :=
by
  sorry

end deer_families_initial_count_l85_85651


namespace range_of_a_l85_85818

def A (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 + a * x - y + 2 = 0}
def B : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ 2 * x - y + 1 = 0 ∧ x > 0}

theorem range_of_a (a : ℝ) : (∃ p, p ∈ A a ∧ p ∈ B) ↔ a ∈ Set.Iic 0 := by
  sorry

end range_of_a_l85_85818


namespace dice_probability_green_l85_85311

theorem dice_probability_green :
  let total_faces := 6
  let green_faces := 3
  let probability := green_faces / total_faces
  probability = 1 / 2 :=
by
  let total_faces := 6
  let green_faces := 3
  let probability := green_faces / total_faces
  have h : probability = 1 / 2 := by sorry
  exact h

end dice_probability_green_l85_85311


namespace find_b_minus_c_l85_85891

variable (a b c: ℤ)

theorem find_b_minus_c (h1: a - b - c = 1) (h2: a - (b - c) = 13) (h3: (b - c) - a = -9) : b - c = 1 :=
by {
  sorry
}

end find_b_minus_c_l85_85891


namespace units_digit_of_17_pow_2025_l85_85312

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_2025 :
  units_digit (17 ^ 2025) = 7 :=
by sorry

end units_digit_of_17_pow_2025_l85_85312


namespace measure_of_one_exterior_angle_l85_85163

theorem measure_of_one_exterior_angle (n : ℕ) (h : n > 2) : 
  n > 2 → ∃ (angle : ℝ), angle = 360 / n :=
by 
  sorry

end measure_of_one_exterior_angle_l85_85163


namespace amc12a_2006_p24_l85_85719

theorem amc12a_2006_p24 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 :=
by
  sorry

end amc12a_2006_p24_l85_85719


namespace c_sq_minus_a_sq_divisible_by_48_l85_85160

theorem c_sq_minus_a_sq_divisible_by_48
  (a b c : ℤ) (h_ac : a < c) (h_eq : a^2 + c^2 = 2 * b^2) : 48 ∣ (c^2 - a^2) := 
  sorry

end c_sq_minus_a_sq_divisible_by_48_l85_85160


namespace rectangle_area_l85_85275

theorem rectangle_area (side_of_square := 45)
  (radius_of_circle := side_of_square)
  (length_of_rectangle := (2/5 : ℚ) * radius_of_circle)
  (breadth_of_rectangle := 10) :
  breadth_of_rectangle * length_of_rectangle = 180 := 
by
  sorry

end rectangle_area_l85_85275


namespace number_of_true_propositions_l85_85746

-- Define the original proposition
def prop (x: Real) : Prop := x^2 > 1 → x > 1

-- Define converse, inverse, contrapositive
def converse (x: Real) : Prop := x > 1 → x^2 > 1
def inverse (x: Real) : Prop := x^2 ≤ 1 → x ≤ 1
def contrapositive (x: Real) : Prop := x ≤ 1 → x^2 ≤ 1

-- Define the proposition we want to prove: the number of true propositions among them
theorem number_of_true_propositions :
  (converse 2 = True) ∧ (inverse 2 = True) ∧ (contrapositive 2 = False) → 2 = 2 :=
by sorry

end number_of_true_propositions_l85_85746


namespace quadratic_equation_completing_square_l85_85079

theorem quadratic_equation_completing_square :
  (∃ m n : ℝ, (∀ x : ℝ, 15 * x^2 - 30 * x - 45 = 15 * ((x + m)^2 - m^2 - 3) + 45 ∧ (m + n = 3))) :=
sorry

end quadratic_equation_completing_square_l85_85079


namespace initial_men_in_camp_l85_85242

theorem initial_men_in_camp (M F : ℕ) 
  (h1 : F = M * 50)
  (h2 : F = (M + 10) * 25) : 
  M = 10 :=
by
  sorry

end initial_men_in_camp_l85_85242


namespace angle_A_range_l85_85597

theorem angle_A_range (a b : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) :
  ∃ A : ℝ, 0 < A ∧ A ≤ Real.pi / 4 :=
sorry

end angle_A_range_l85_85597


namespace max_value_l85_85671

noncomputable def satisfies_equation (x y : ℝ) : Prop :=
  x + 4 * y - x * y = 0

theorem max_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : satisfies_equation x y) :
  ∃ m, m = (4 / (x + y)) ∧ m ≤ (4 / 9) :=
by
  sorry

end max_value_l85_85671


namespace range_of_values_for_a_l85_85143

theorem range_of_values_for_a (a : ℝ) :
  (∀ x : ℝ, (x + 2) / 3 - x / 2 > 1 → 2 * (x - a) ≤ 0) → a ≥ -2 :=
by
  intro h
  sorry

end range_of_values_for_a_l85_85143


namespace triangle_is_obtuse_l85_85125

def is_obtuse_triangle (a b c : ℕ) : Prop := a^2 + b^2 < c^2

theorem triangle_is_obtuse :
    is_obtuse_triangle 4 6 8 :=
by
    sorry

end triangle_is_obtuse_l85_85125


namespace speed_of_rest_distance_l85_85502

theorem speed_of_rest_distance (D V : ℝ) (h1 : D = 26.67)
                                (h2 : (D / 2) / 5 + (D / 2) / V = 6) : 
  V = 20 :=
by
  sorry

end speed_of_rest_distance_l85_85502


namespace length_of_train_l85_85213

-- Definitions based on the conditions in the problem
def time_to_cross_signal_pole : ℝ := 18
def time_to_cross_platform : ℝ := 54
def length_of_platform : ℝ := 600.0000000000001

-- Prove that the length of the train is 300.00000000000005 meters
theorem length_of_train
    (L V : ℝ)
    (h1 : L = V * time_to_cross_signal_pole)
    (h2 : L + length_of_platform = V * time_to_cross_platform) :
    L = 300.00000000000005 :=
by
  sorry

end length_of_train_l85_85213


namespace abs_neg_one_over_2023_l85_85849

theorem abs_neg_one_over_2023 : abs (-1 / 2023) = 1 / 2023 :=
by
  sorry

end abs_neg_one_over_2023_l85_85849


namespace max_working_groups_l85_85325

theorem max_working_groups (teachers groups : ℕ) (memberships_per_teacher group_size : ℕ) 
  (h_teachers : teachers = 36) (h_memberships_per_teacher : memberships_per_teacher = 2)
  (h_group_size : group_size = 4) 
  (h_max_memberships : teachers * memberships_per_teacher = 72) :
  groups ≤ 18 :=
by
  sorry

end max_working_groups_l85_85325


namespace reciprocal_neg_two_l85_85028

theorem reciprocal_neg_two : 1 / (-2) = - (1 / 2) :=
by
  sorry

end reciprocal_neg_two_l85_85028


namespace min_height_box_l85_85426

noncomputable def min_height (x : ℝ) : ℝ :=
  if h : x ≥ (5 : ℝ) then x + 5 else 0

theorem min_height_box (x : ℝ) (hx : 3*x^2 + 10*x - 65 ≥ 0) : min_height x = 10 :=
by
  sorry

end min_height_box_l85_85426


namespace washes_per_bottle_l85_85683

def bottle_cost : ℝ := 4.0
def total_weeks : ℕ := 20
def total_cost : ℝ := 20.0

theorem washes_per_bottle : (total_weeks / (total_cost / bottle_cost)) = 4 := by
  sorry

end washes_per_bottle_l85_85683


namespace carolyn_fewer_stickers_l85_85904

theorem carolyn_fewer_stickers :
  let belle_stickers := 97
  let carolyn_stickers := 79
  carolyn_stickers < belle_stickers →
  belle_stickers - carolyn_stickers = 18 :=
by
  intros
  sorry

end carolyn_fewer_stickers_l85_85904


namespace smallest_clock_equiv_to_square_greater_than_10_l85_85286

def clock_equiv (h k : ℕ) : Prop :=
  (h % 12) = (k % 12)

theorem smallest_clock_equiv_to_square_greater_than_10 : ∃ h > 10, clock_equiv h (h * h) ∧ ∀ h' > 10, clock_equiv h' (h' * h') → h ≤ h' :=
by
  sorry

end smallest_clock_equiv_to_square_greater_than_10_l85_85286


namespace intersection_eq_inter_l85_85990

noncomputable def M : Set ℝ := { x | x^2 < 4 }
noncomputable def N : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
noncomputable def inter : Set ℝ := { x | -1 < x ∧ x < 2 }

theorem intersection_eq_inter : M ∩ N = inter :=
by sorry

end intersection_eq_inter_l85_85990


namespace jack_total_damage_costs_l85_85040

def cost_per_tire := 250
def number_of_tires := 3
def cost_of_window := 700

def total_cost_of_tires := cost_per_tire * number_of_tires
def total_cost_of_damages := total_cost_of_tires + cost_of_window

theorem jack_total_damage_costs : total_cost_of_damages = 1450 := 
by
  -- Using the definitions provided
  -- total_cost_of_tires = 250 * 3 = 750
  -- total_cost_of_damages = 750 + 700 = 1450
  sorry

end jack_total_damage_costs_l85_85040


namespace common_divisors_count_l85_85521

-- Define the numbers
def num1 := 9240
def num2 := 13860

-- Define the gcd of the numbers
def gcdNum := Nat.gcd num1 num2

-- Prove the number of divisors of the gcd is 48
theorem common_divisors_count : (Nat.divisors gcdNum).card = 48 :=
by
  -- Normally we would provide a detailed proof here
  sorry

end common_divisors_count_l85_85521


namespace fraction_of_fractions_l85_85459

theorem fraction_of_fractions : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_fractions_l85_85459


namespace lucas_notation_sum_l85_85405

-- Define what each representation in Lucas's notation means
def lucasValue : String → Int
| "0" => 0
| s => -((s.length) - 1)

-- Define the question as a Lean theorem
theorem lucas_notation_sum :
  lucasValue "000" + lucasValue "0000" = lucasValue "000000" :=
by
  sorry

end lucas_notation_sum_l85_85405


namespace no_equilateral_triangle_on_grid_regular_tetrahedron_on_grid_l85_85359

-- Define the context for part (a)
theorem no_equilateral_triangle_on_grid (x1 y1 x2 y2 x3 y3 : ℤ) :
  ¬ (x1 = x2 ∧ y1 = y2) ∧ (x2 = x3 ∧ y2 = y3) ∧ (x3 = x1 ∧ y3 = y1) ∧ -- vertices must not be the same
  ((x2 - x1)^2 + (y2 - y1)^2 = (x3 - x2)^2 + (y3 - y2)^2) ∧ -- sides must be equal
  ((x3 - x1)^2 + (y3 - y1)^2 = (x2 - x1)^2 + (y2 - y1)^2) ->
  false := 
sorry

-- Define the context for part (b)
theorem regular_tetrahedron_on_grid (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℤ) :
  ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2 = (x3 - x2)^2 + (y3 - y2)^2 + (z3 - z2)^2) ∧ -- first condition: edge lengths equal
  ((x3 - x1)^2 + (y3 - y1)^2 + (z3 - z1)^2 = (x4 - x3)^2 + (y4 - y3)^2 + (z4 - z3)^2) ∧ -- second condition: edge lengths equal
  ((x4 - x1)^2 + (y4 - y1)^2 + (z4 - z1)^2 = (x2 - x4)^2 + (y2 - y4)^2 + (z2 - z4)^2) -> -- third condition: edge lengths equal
  true := 
sorry

end no_equilateral_triangle_on_grid_regular_tetrahedron_on_grid_l85_85359


namespace boxes_contain_fruits_l85_85880

-- Define the weights of the boxes
def box_weights : List ℕ := [15, 16, 18, 19, 20, 31]

-- Define the weight requirement for apples and pears
def weight_rel (apple_weight pear_weight : ℕ) : Prop := apple_weight = pear_weight / 2

-- Define the statement with the constraints, given conditions and assignments.
theorem boxes_contain_fruits (h1 : box_weights = [15, 16, 18, 19, 20, 31])
                             (h2 : ∃ apple_weight pear_weight, 
                                   weight_rel apple_weight pear_weight ∧ 
                                   pear_weight ∈ box_weights ∧ apple_weight ∈ box_weights)
                             (h3 : ∃ orange_weight, orange_weight ∈ box_weights ∧ 
                                   ∀ w, w ∈ box_weights → w ≠ orange_weight)
                             : (15 = 2 ∧ 19 = 3 ∧ 20 = 1 ∧ 31 = 3) := 
                             sorry

end boxes_contain_fruits_l85_85880


namespace smallest_x_value_l85_85713

-- Definitions based on given problem conditions
def is_solution (x y : ℕ) : Prop :=
  0 < x ∧ 0 < y ∧ (3 : ℝ) / 4 = y / (252 + x)

theorem smallest_x_value : ∃ x : ℕ, ∀ y : ℕ, is_solution x y → x = 0 :=
by
  sorry

end smallest_x_value_l85_85713


namespace total_windows_l85_85994

theorem total_windows (installed: ℕ) (hours_per_window: ℕ) (remaining_hours: ℕ) : installed = 8 → hours_per_window = 8 → remaining_hours = 48 → 
  (installed + remaining_hours / hours_per_window) = 14 := by 
  intros h1 h2 h3
  sorry

end total_windows_l85_85994


namespace arithmetic_seq_a₄_l85_85263

-- Definitions for conditions in the given problem
def S₅ (a₁ a₅ : ℕ) : ℕ := ((a₁ + a₅) * 5) / 2
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Final proof statement to show that a₄ = 9
theorem arithmetic_seq_a₄ (a₁ a₅ : ℕ) (d : ℕ) (h₁ : S₅ a₁ a₅ = 35) (h₂ : a₅ = 11) (h₃ : d = (a₅ - a₁) / 4) :
  arithmetic_sequence a₁ d 4 = 9 :=
sorry

end arithmetic_seq_a₄_l85_85263


namespace find_r_l85_85438

variable (n : ℕ) (q r : ℝ)

-- n must be a positive natural number
axiom n_pos : n > 0

-- q is a positive real number and not equal to 1
axiom q_pos : q > 0
axiom q_ne_one : q ≠ 1

-- Define the sequence sum S_n according to the problem statement
def S_n (n : ℕ) (q r : ℝ) : ℝ := q^n + r

-- The goal is to prove that the correct value of r is -1
theorem find_r : r = -1 :=
sorry

end find_r_l85_85438


namespace intersection_P_Q_l85_85066

def P : Set ℝ := {x | x^2 - 9 < 0}
def Q : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

theorem intersection_P_Q :
  {x : ℤ | (x : ℝ) ∈ P} ∩ Q = {-1, 0, 1, 2} := 
by
  sorry

end intersection_P_Q_l85_85066


namespace sin_810_cos_neg60_l85_85095

theorem sin_810_cos_neg60 :
  Real.sin (810 * Real.pi / 180) + Real.cos (-60 * Real.pi / 180) = 3 / 2 :=
by
  sorry

end sin_810_cos_neg60_l85_85095


namespace a_minus_b_eq_zero_l85_85579

-- Definitions from the conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- The point (0, b)
def point_b (b : ℝ) : (ℝ × ℝ) := (0, b)

-- Slope condition at point (0, b)
def slope_of_f_at_0 (a : ℝ) : ℝ := a
def slope_of_tangent_line : ℝ := 1

-- Prove a - b = 0 given the conditions
theorem a_minus_b_eq_zero (a b : ℝ) 
    (h1 : f 0 a b = b)
    (h2 : tangent_line 0 b) 
    (h3 : slope_of_f_at_0 a = slope_of_tangent_line) : a - b = 0 :=
by
  sorry

end a_minus_b_eq_zero_l85_85579


namespace initial_people_in_line_l85_85479

theorem initial_people_in_line (X : ℕ) 
  (h1 : X - 6 + 3 = 18) : X = 21 :=
  sorry

end initial_people_in_line_l85_85479


namespace completing_the_square_l85_85255

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l85_85255


namespace num_five_letter_words_correct_l85_85260

noncomputable def num_five_letter_words : ℕ := 1889568

theorem num_five_letter_words_correct :
  let a := 3
  let e := 4
  let i := 2
  let o := 5
  let u := 4
  (a + e + i + o + u) ^ 5 = num_five_letter_words :=
by
  sorry

end num_five_letter_words_correct_l85_85260


namespace monomial_properties_l85_85972

def coefficient (m : ℝ) := -3
def degree (x_exp y_exp : ℕ) := x_exp + y_exp

theorem monomial_properties :
  ∀ (x_exp y_exp : ℕ), coefficient (-3) = -3 ∧ degree 2 1 = 3 :=
by
  sorry

end monomial_properties_l85_85972


namespace greatest_possible_perimeter_l85_85548

def triangle_side_lengths (x : ℤ) : Prop :=
  (x > 0) ∧ (5 * x > 18) ∧ (x < 6)

def perimeter (x : ℤ) : ℤ :=
  x + 4 * x + 18

theorem greatest_possible_perimeter :
  ∃ x : ℤ, triangle_side_lengths x ∧ (perimeter x = 38) :=
by
  sorry

end greatest_possible_perimeter_l85_85548


namespace B_finish_in_54_days_l85_85894

-- Definitions based on conditions
variables (A B : ℝ) -- A and B are the amount of work done in one day
axiom h1 : A = 2 * B -- A is twice as good as workman as B
axiom h2 : (A + B) * 18 = 1 -- Together, A and B finish the piece of work in 18 days

-- Prove that B alone will finish the work in 54 days.
theorem B_finish_in_54_days : (1 / B) = 54 :=
by 
  sorry

end B_finish_in_54_days_l85_85894


namespace necessary_but_not_sufficient_l85_85697

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (b < -1 → |a| + |b| > 1) ∧ (∃ a b : ℝ, |a| + |b| > 1 ∧ b >= -1) :=
by
  sorry

end necessary_but_not_sufficient_l85_85697


namespace apples_sold_l85_85782

theorem apples_sold (a1 a2 a3 : ℕ) (h1 : a3 = a2 / 4 + 8) (h2 : a2 = a1 / 4 + 8) (h3 : a3 = 18) : a1 = 128 :=
by
  sorry

end apples_sold_l85_85782


namespace rectangle_area_l85_85787

theorem rectangle_area
  (x : ℝ)
  (perimeter_eq_160 : 10 * x = 160) :
  4 * (4 * x * x) = 1024 :=
by
  -- We would solve the problem and show the steps here
  sorry

end rectangle_area_l85_85787


namespace b_2018_eq_5043_l85_85487

def b (n : Nat) : Nat :=
  if n % 2 = 1 then 5 * ((n + 1) / 2) - 3 else 5 * (n / 2) - 2

theorem b_2018_eq_5043 : b 2018 = 5043 := by
  sorry

end b_2018_eq_5043_l85_85487


namespace largest_divisor_60_36_divisible_by_3_l85_85439

theorem largest_divisor_60_36_divisible_by_3 : 
  ∃ x, (x ∣ 60) ∧ (x ∣ 36) ∧ (3 ∣ x) ∧ (∀ y, (y ∣ 60) → (y ∣ 36) → (3 ∣ y) → y ≤ x) ∧ x = 12 :=
sorry

end largest_divisor_60_36_divisible_by_3_l85_85439


namespace normal_line_equation_at_x0_l85_85706

def curve (x : ℝ) : ℝ := x - x^3
noncomputable def x0 : ℝ := -1
noncomputable def y0 : ℝ := curve x0

theorem normal_line_equation_at_x0 :
  ∀ (y : ℝ), y = (1/2 : ℝ) * x + 1/2 ↔ (∃ (x : ℝ), y = curve x ∧ x = x0) :=
by
  sorry

end normal_line_equation_at_x0_l85_85706


namespace combined_perimeter_l85_85496

theorem combined_perimeter (side_square : ℝ) (a b c : ℝ) (diameter : ℝ) 
  (h_square : side_square = 7) 
  (h_triangle : a = 5 ∧ b = 6 ∧ c = 7) 
  (h_diameter : diameter = 4) : 
  4 * side_square + (a + b + c) + (2 * Real.pi * (diameter / 2) + diameter) = 50 + 2 * Real.pi := 
by 
  sorry

end combined_perimeter_l85_85496


namespace extreme_values_for_f_when_a_is_one_number_of_zeros_of_h_l85_85540

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / Real.exp x + x^2 / 2 - x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := max (f a x) (g x)

theorem extreme_values_for_f_when_a_is_one :
  (∀ x : ℝ, (f 1 x) ≤ 0) ∧ f 1 0 = 0 ∧ f 1 1 = (1 / Real.exp 1) - 1 / 2 :=
sorry

theorem number_of_zeros_of_h (a : ℝ) :
  (0 ≤ a → 
   if 1 < a ∧ a < Real.exp 1 / 2 then
     ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ h a x1 = 0 ∧ h a x2 = 0
   else if 0 ≤ a ∧ a ≤ 1 ∨ a = Real.exp 1 / 2 then
     ∃ x : ℝ, 0 < x ∧ x < 1 ∧ h a x = 0
   else
     ∀ x : ℝ, x > 0 → h a x ≠ 0) :=
sorry

end extreme_values_for_f_when_a_is_one_number_of_zeros_of_h_l85_85540


namespace speed_of_second_half_l85_85449

theorem speed_of_second_half (total_time : ℕ) (speed_first_half : ℕ) (total_distance : ℕ)
  (h1 : total_time = 15) (h2 : speed_first_half = 21) (h3 : total_distance = 336) :
  2 * total_distance / total_time - speed_first_half * (total_time / 2) / (total_time / 2) = 24 :=
by
  -- Proof omitted
  sorry

end speed_of_second_half_l85_85449


namespace ones_digit_largest_power_of_three_divides_factorial_3_pow_3_l85_85485

theorem ones_digit_largest_power_of_three_divides_factorial_3_pow_3 :
  (3 ^ 13) % 10 = 3 := by
  sorry

end ones_digit_largest_power_of_three_divides_factorial_3_pow_3_l85_85485


namespace problem1_solution_set_problem2_a_range_l85_85755

-- Define the function f
def f (x a : ℝ) := |x - a| + 5 * x

-- Problem Part 1: Prove for a = -1, the solution set for f(x) ≤ 5x + 3 is [-4, 2]
theorem problem1_solution_set :
  ∀ (x : ℝ), f x (-1) ≤ 5 * x + 3 ↔ -4 ≤ x ∧ x ≤ 2 := by
  sorry

-- Problem Part 2: Prove that if f(x) ≥ 0 for all x ≥ -1, then a ≥ 4 or a ≤ -6
theorem problem2_a_range (a : ℝ) :
  (∀ (x : ℝ), x ≥ -1 → f x a ≥ 0) ↔ a ≥ 4 ∨ a ≤ -6 := by
  sorry

end problem1_solution_set_problem2_a_range_l85_85755


namespace min_inverse_sum_l85_85956

theorem min_inverse_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 4) : 1 ≤ (1/a) + (1/b) :=
by
  sorry

end min_inverse_sum_l85_85956


namespace apples_distribution_count_l85_85218

theorem apples_distribution_count : 
  ∃ (count : ℕ), count = 249 ∧ 
  (∃ (a b c : ℕ), a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3 ∧ a ≤ 20) →
  (a' + 3 + b' + 3 + c' + 3 = 30 ∧ a' + b' + c' = 21) → 
  (∃ (a' b' c' : ℕ), a' + b' + c' = 21 ∧ a' ≤ 17) :=
by
  sorry

end apples_distribution_count_l85_85218


namespace t1_eq_t2_l85_85995

variable (n : ℕ)
variable (s₁ s₂ s₃ : ℝ)
variable (t₁ t₂ : ℝ)
variable (S1 S2 S3 : ℝ)

-- Conditions
axiom h1 : S1 = s₁
axiom h2 : S2 = s₂
axiom h3 : S3 = s₃
axiom h4 : t₁ = s₂^2 - s₁ * s₃
axiom h5 : t₂ = ( (s₁ - s₃) / 2 )^2
axiom h6 : s₁ + s₃ = 2 * s₂

theorem t1_eq_t2 : t₁ = t₂ := by
  sorry

end t1_eq_t2_l85_85995


namespace problem_statement_l85_85868

noncomputable def x : ℕ := 4
noncomputable def y : ℤ := 3  -- alternatively, we could define y as -3 and the equality would still hold

theorem problem_statement : x^2 + y^2 + x + 2023 = 2052 := by
  sorry  -- Proof goes here

end problem_statement_l85_85868


namespace alex_ate_more_pears_than_sam_l85_85614

namespace PearEatingContest

def number_of_pears_eaten (Alex Sam : ℕ) : ℕ :=
  Alex - Sam

theorem alex_ate_more_pears_than_sam :
  number_of_pears_eaten 8 2 = 6 := by
  -- proof
  sorry

end PearEatingContest

end alex_ate_more_pears_than_sam_l85_85614


namespace min_sum_rect_box_l85_85037

-- Define the main theorem with the given constraints
theorem min_sum_rect_box (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_vol : a * b * c = 2002) : a + b + c ≥ 38 :=
  sorry

end min_sum_rect_box_l85_85037


namespace range_of_b_for_monotonic_function_l85_85158

theorem range_of_b_for_monotonic_function :
  (∀ x : ℝ, (x^2 + 2 * b * x + b + 2) ≥ 0) ↔ (-1 ≤ b ∧ b ≤ 2) :=
by sorry

end range_of_b_for_monotonic_function_l85_85158


namespace exists_consecutive_nat_with_integer_quotient_l85_85768

theorem exists_consecutive_nat_with_integer_quotient :
  ∃ n : ℕ, (n + 1) / n = 2 :=
by
  sorry

end exists_consecutive_nat_with_integer_quotient_l85_85768


namespace evaluate_expression_l85_85976

noncomputable def given_expression : ℝ :=
  |8 - 8 * (3 - 12)^2| - |5 - Real.sin 11| + |2^(4 - 2 * 3) / ((3^2) - 7)|

theorem evaluate_expression : given_expression = 634.125009794 := 
  sorry

end evaluate_expression_l85_85976


namespace positive_integer_cases_l85_85726

theorem positive_integer_cases (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℤ, (abs (x^2 - abs x)) / x = n ∧ n > 0) ↔ (∃ m : ℤ, (x = m) ∧ (m > 1 ∨ m < -1)) :=
by
  sorry

end positive_integer_cases_l85_85726


namespace pumac_grader_remainder_l85_85887

/-- A PUMaC grader is grading the submissions of forty students s₁, s₂, ..., s₄₀ for the
    individual finals round, which has three problems.
    After grading a problem of student sᵢ, the grader either:
    * grades another problem of the same student, or
    * grades the same problem of the student sᵢ₋₁ or sᵢ₊₁ (if i > 1 and i < 40, respectively).
    He grades each problem exactly once, starting with the first problem of s₁
    and ending with the third problem of s₄₀.
    Let N be the number of different orders the grader may grade the students’ problems in this way.
    Prove: N ≡ 78 [MOD 100] -/

noncomputable def grading_orders_mod : ℕ := 2 * (3 ^ 38) % 100

theorem pumac_grader_remainder :
  grading_orders_mod = 78 :=
by
  sorry

end pumac_grader_remainder_l85_85887


namespace greatest_integer_value_l85_85769

theorem greatest_integer_value (x : ℤ) : 7 - 3 * x > 20 → x ≤ -5 :=
by
  intros h
  sorry

end greatest_integer_value_l85_85769


namespace domain_of_function_l85_85889

section
variable (x : ℝ)

def condition_1 := x + 4 ≥ 0
def condition_2 := x + 2 ≠ 0
def domain := { x : ℝ | x ≥ -4 ∧ x ≠ -2 }

theorem domain_of_function : (condition_1 x ∧ condition_2 x) ↔ (x ∈ domain) :=
by
  sorry
end

end domain_of_function_l85_85889


namespace spoons_in_set_l85_85290

def number_of_spoons_in_set (total_cost_set : ℕ) (cost_five_spoons : ℕ) : ℕ :=
  let c := cost_five_spoons / 5
  let s := total_cost_set / c
  s

theorem spoons_in_set (total_cost_set : ℕ) (cost_five_spoons : ℕ) (h1 : total_cost_set = 21) (h2 : cost_five_spoons = 15) : 
  number_of_spoons_in_set total_cost_set cost_five_spoons = 7 :=
by
  sorry

end spoons_in_set_l85_85290


namespace find_line_eq_l85_85093

theorem find_line_eq (x y : ℝ) (h : x^2 + y^2 - 4 * x - 5 = 0) 
(mid_x mid_y : ℝ) (mid_point : mid_x = 3 ∧ mid_y = 1) : 
x + y - 4 = 0 := 
sorry

end find_line_eq_l85_85093


namespace problem_statement_l85_85536

theorem problem_statement (x y : ℝ) (h : x - 2 * y = -2) : 3 + 2 * x - 4 * y = -1 :=
  sorry

end problem_statement_l85_85536


namespace max_items_sum_l85_85464

theorem max_items_sum (m n : ℕ) (h : 5 * m + 17 * n = 203) : m + n ≤ 31 :=
sorry

end max_items_sum_l85_85464


namespace archer_scores_distribution_l85_85396

structure ArcherScores where
  hits_40 : ℕ
  hits_39 : ℕ
  hits_24 : ℕ
  hits_23 : ℕ
  hits_17 : ℕ
  hits_16 : ℕ
  total_score : ℕ

theorem archer_scores_distribution
  (dora : ArcherScores)
  (reggie : ArcherScores)
  (finch : ArcherScores)
  (h1 : dora.total_score = 120)
  (h2 : reggie.total_score = 110)
  (h3 : finch.total_score = 100)
  (h4 : dora.hits_40 + dora.hits_39 + dora.hits_24 + dora.hits_23 + dora.hits_17 + dora.hits_16 = 6)
  (h5 : reggie.hits_40 + reggie.hits_39 + reggie.hits_24 + reggie.hits_23 + reggie.hits_17 + reggie.hits_16 = 6)
  (h6 : finch.hits_40 + finch.hits_39 + finch.hits_24 + finch.hits_23 + finch.hits_17 + finch.hits_16 = 6)
  (h7 : 40 * dora.hits_40 + 39 * dora.hits_39 + 24 * dora.hits_24 + 23 * dora.hits_23 + 17 * dora.hits_17 + 16 * dora.hits_16 = 120)
  (h8 : 40 * reggie.hits_40 + 39 * reggie.hits_39 + 24 * reggie.hits_24 + 23 * reggie.hits_23 + 17 * reggie.hits_17 + 16 * reggie.hits_16 = 110)
  (h9 : 40 * finch.hits_40 + 39 * finch.hits_39 + 24 * finch.hits_24 + 23 * finch.hits_23 + 17 * finch.hits_17 + 16 * finch.hits_16 = 100)
  (h10 : dora.hits_40 = 1)
  (h11 : dora.hits_39 = 0)
  (h12 : dora.hits_24 = 0) :
  dora.hits_40 = 1 ∧ dora.hits_16 = 5 ∧ 
  reggie.hits_23 = 2 ∧ reggie.hits_16 = 4 ∧ 
  finch.hits_17 = 4 ∧ finch.hits_16 = 2 :=
sorry

end archer_scores_distribution_l85_85396


namespace find_d_l85_85858

noncomputable def median (x : ℕ) : ℕ := x + 4
noncomputable def mean (x d : ℕ) : ℕ := x + (13 + d) / 5

theorem find_d (x d : ℕ) (h : mean x d = median x + 5) : d = 32 := by
  sorry

end find_d_l85_85858


namespace no_integer_n_exists_l85_85272

theorem no_integer_n_exists (n : ℤ) : ¬(∃ n : ℤ, ∃ k : ℤ, ∃ m : ℤ, (n - 6) = 15 * k ∧ (n - 5) = 24 * m) :=
by
  sorry

end no_integer_n_exists_l85_85272


namespace median_on_hypotenuse_length_l85_85267

theorem median_on_hypotenuse_length
  (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) (right_triangle : (a ^ 2 + b ^ 2) = c ^ 2) :
  (1 / 2) * c = 5 :=
  sorry

end median_on_hypotenuse_length_l85_85267


namespace tennis_to_soccer_ratio_l85_85323

theorem tennis_to_soccer_ratio
  (total_balls : ℕ)
  (soccer_balls : ℕ)
  (basketball_offset : ℕ)
  (baseball_offset : ℕ)
  (volleyballs : ℕ)
  (tennis_balls : ℕ)
  (total_balls_eq : total_balls = 145)
  (soccer_balls_eq : soccer_balls = 20)
  (basketball_count : soccer_balls + basketball_offset = 20 + 5)
  (baseball_count : soccer_balls + baseball_offset = 20 + 10)
  (volleyballs_eq : volleyballs = 30)
  (accounted_balls : soccer_balls + (soccer_balls + basketball_offset) + (soccer_balls + baseball_offset) + volleyballs = 105)
  (tennis_balls_eq : tennis_balls = 145 - 105) :
  tennis_balls / soccer_balls = 2 :=
sorry

end tennis_to_soccer_ratio_l85_85323


namespace least_number_of_stamps_l85_85598

def min_stamps (x y : ℕ) : ℕ := x + y

theorem least_number_of_stamps {x y : ℕ} (h : 5 * x + 7 * y = 50) 
  : min_stamps x y = 8 :=
sorry

end least_number_of_stamps_l85_85598


namespace proof_l85_85189

-- Define the expression
def expr : ℕ :=
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128)

-- Define the conjectured result
def result : ℕ := 5^128 - 4^128

-- Assert their equality
theorem proof : expr = result :=
by
    sorry

end proof_l85_85189


namespace a0_a2_a4_sum_l85_85089

theorem a0_a2_a4_sum (a0 a1 a2 a3 a4 a5 : ℝ) :
  (∀ x : ℝ, (2 * x - 3) ^ 5 = a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 + a5 * (x - 1) ^ 5) →
  a0 + a2 + a4 = -121 :=
by
  intros h
  sorry

end a0_a2_a4_sum_l85_85089


namespace least_possible_product_of_primes_l85_85556

-- Define a prime predicate for a number greater than 20
def is_prime_over_20 (p : Nat) : Prop := Nat.Prime p ∧ p > 20

-- Define the two primes
def prime1 := 23
def prime2 := 29

-- Given the conditions, prove the least possible product of two distinct primes greater than 20 is 667
theorem least_possible_product_of_primes :
  ∃ p1 p2 : Nat, is_prime_over_20 p1 ∧ is_prime_over_20 p2 ∧ p1 ≠ p2 ∧ (p1 * p2 = 667) :=
by
  -- Theorem statement without proof
  existsi (prime1)
  existsi (prime2)
  have h1 : is_prime_over_20 prime1 := by sorry
  have h2 : is_prime_over_20 prime2 := by sorry
  have h3 : prime1 ≠ prime2 := by sorry
  have h4 : prime1 * prime2 = 667 := by sorry
  exact ⟨h1, h2, h3, h4⟩

end least_possible_product_of_primes_l85_85556


namespace max_value_F_l85_85650

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * x^2
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x

noncomputable def F (x : ℝ) : ℝ :=
if f x ≥ g x then f x else g x

theorem max_value_F : ∃ x : ℝ, ∀ y : ℝ, F y ≤ F x ∧ F x = 7 / 9 := 
sorry

end max_value_F_l85_85650


namespace algebraic_notation_correct_l85_85981

def exprA : String := "a * 5"
def exprB : String := "a7"
def exprC : String := "3 1/2 x"
def exprD : String := "-7/8 x"

theorem algebraic_notation_correct :
  exprA ≠ "correct" ∧
  exprB ≠ "correct" ∧
  exprC ≠ "correct" ∧
  exprD = "correct" :=
by
  sorry

end algebraic_notation_correct_l85_85981


namespace range_of_x_l85_85204

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the relevant conditions
axiom decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) < 0
axiom symmetry : ∀ x : ℝ, f (1 - x) = -f (1 + x)
axiom f_one : f 1 = -1

-- Define the statement to be proved
theorem range_of_x : ∀ x : ℝ, -1 ≤ f (0.5 * x - 1) ∧ f (0.5 * x - 1) ≤ 1 → 0 ≤ x ∧ x ≤ 4 :=
sorry

end range_of_x_l85_85204


namespace find_value_of_a_l85_85802

theorem find_value_of_a (a : ℝ) :
  (∀ x y : ℝ, (ax + y - 1 = 0) → (x - y + 3 = 0) → (-a) * 1 = -1) → a = 1 :=
by
  sorry

end find_value_of_a_l85_85802


namespace find_larger_number_l85_85137

theorem find_larger_number 
  (L S : ℕ) 
  (h1 : L - S = 2342) 
  (h2 : L = 9 * S + 23) : 
  L = 2624 := 
sorry

end find_larger_number_l85_85137


namespace solution_set_l85_85515

theorem solution_set (x : ℝ) : (3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9) ↔ (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) :=
sorry

end solution_set_l85_85515


namespace min_value_y_l85_85430

theorem min_value_y (x : ℝ) (hx : x > 2) : 
  ∃ x, x > 2 ∧ (∀ y, y = (x^2 - 4*x + 8) / (x - 2) → y ≥ 4 ∧ y = 4 ↔ x = 4) :=
sorry

end min_value_y_l85_85430


namespace min_value_of_squares_attains_min_value_l85_85415

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) :
  (a^2 + b^2 + c^2) ≥ (t^2 / 3) :=
sorry

theorem attains_min_value (a b c t : ℝ) (h : a = t / 3 ∧ b = t / 3 ∧ c = t / 3) :
  (a^2 + b^2 + c^2) = (t^2 / 3) :=
sorry

end min_value_of_squares_attains_min_value_l85_85415


namespace constant_term_in_expansion_l85_85382

-- Given conditions
def eq_half_n_minus_m_zero (n m : ℕ) : Prop := 1/2 * n = m
def eq_n_plus_m_ten (n m : ℕ) : Prop := n + m = 10
noncomputable def binom (n k : ℕ) : ℝ := Real.exp (Real.log (Nat.factorial n) - Real.log (Nat.factorial k) - Real.log (Nat.factorial (n - k)))

-- Main theorem
theorem constant_term_in_expansion : 
  ∃ (n m : ℕ), eq_half_n_minus_m_zero n m ∧ eq_n_plus_m_ten n m ∧ 
  binom 10 m * (3^4 : ℝ) = 17010 :=
by
  -- Definitions translation
  sorry

end constant_term_in_expansion_l85_85382


namespace trigonometric_identity_l85_85940

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.tan α = 2) :
  (4 * Real.sin α ^ 3 - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 2 / 5 :=
by
  sorry

end trigonometric_identity_l85_85940


namespace evaluate_expression_l85_85872

theorem evaluate_expression : 
  (Int.ceil ((Int.floor ((15 / 8 : Rat) ^ 2) : Rat) - (19 / 5 : Rat) : Rat) : Int) = 0 :=
sorry

end evaluate_expression_l85_85872


namespace g_minus_1001_l85_85957

def g (x : ℝ) : ℝ := sorry

theorem g_minus_1001 :
  (∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x) →
  g 1 = 3 →
  g (-1001) = 1005 :=
by
  intros h1 h2
  sorry

end g_minus_1001_l85_85957


namespace differential_of_y_l85_85048

variable (x : ℝ) (dx : ℝ)

noncomputable def y := x * (Real.sin (Real.log x) - Real.cos (Real.log x))

theorem differential_of_y : (deriv y x * dx) = 2 * Real.sin (Real.log x) * dx := by
  sorry

end differential_of_y_l85_85048


namespace min_ratio_area_of_incircle_circumcircle_rt_triangle_l85_85915

variables (a b: ℝ)
variables (a' b' c: ℝ)

-- Conditions
def area_of_right_triangle (a b : ℝ) : ℝ := 
    0.5 * a * b

def incircle_radius (a' b' c : ℝ) : ℝ := 
    0.5 * (a' + b' - c)

def circumcircle_radius (c : ℝ) : ℝ := 
    0.5 * c

-- Condition of the problem
def condition (a b a' b' c : ℝ) : Prop :=
    incircle_radius a' b' c = circumcircle_radius c ∧ 
    a' + b' = 2 * c

-- The final proof problem
theorem min_ratio_area_of_incircle_circumcircle_rt_triangle (a b a' b' c : ℝ)
    (h_area_a : a = area_of_right_triangle a' b')
    (h_area_b : b = area_of_right_triangle a b)
    (h_condition : condition a b a' b' c) :
    (a / b ≥ 3 + 2 * Real.sqrt 2) :=
by
  sorry

end min_ratio_area_of_incircle_circumcircle_rt_triangle_l85_85915


namespace isabella_non_yellow_houses_l85_85341

variable (Green Yellow Red Blue Pink : ℕ)

axiom h1 : 3 * Yellow = Green
axiom h2 : Red = Yellow + 40
axiom h3 : Green = 90
axiom h4 : Blue = (Green + Yellow) / 2
axiom h5 : Pink = (Red / 2) + 15

theorem isabella_non_yellow_houses : (Green + Red + Blue + Pink - Yellow) = 270 :=
by 
  sorry

end isabella_non_yellow_houses_l85_85341


namespace purchasing_options_count_l85_85235

theorem purchasing_options_count : ∃ (s : Finset (ℕ × ℕ)), s.card = 4 ∧
  ∀ (a : ℕ × ℕ), a ∈ s ↔ 
    (80 * a.1 + 120 * a.2 = 1000) 
    ∧ (a.1 > 0) ∧ (a.2 > 0) :=
by
  sorry

end purchasing_options_count_l85_85235


namespace number_of_square_tiles_l85_85411

/-- A box contains a collection of triangular tiles, square tiles, and pentagonal tiles. 
    There are a total of 30 tiles in the box and a total of 100 edges. 
    We need to show that the number of square tiles is 10. --/
theorem number_of_square_tiles (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 3 * a + 4 * b + 5 * c = 100) : b = 10 := by
  sorry

end number_of_square_tiles_l85_85411


namespace angle_of_inclination_range_l85_85416

theorem angle_of_inclination_range (a : ℝ) :
  (∃ m : ℝ, ax + (a + 1)*m + 2 = 0 ∧ (m < 0 ∨ m > 1)) ↔ (a < -1/2 ∨ a > 0) := sorry

end angle_of_inclination_range_l85_85416


namespace delaney_left_home_at_7_50_l85_85141

theorem delaney_left_home_at_7_50 :
  (bus_time = 8 * 60 ∧ travel_time = 30 ∧ miss_time = 20) →
  (delaney_leave_time = bus_time + miss_time - travel_time) →
  delaney_leave_time = 7 * 60 + 50 :=
by
  intros
  sorry

end delaney_left_home_at_7_50_l85_85141


namespace R2_perfect_fit_l85_85294

variables {n : ℕ} (x y : Fin n → ℝ) (b a : ℝ)

-- Condition: Observations \( (x_i, y_i) \) such that \( y_i = bx_i + a \)
def observations (i : Fin n) : Prop :=
  y i = b * x i + a

-- Condition: \( e_i = 0 \) for all \( i \)
def no_error (i : Fin n) : Prop := (b * x i + a + 0 = y i)

theorem R2_perfect_fit (h_obs: ∀ i, observations x y b a i)
                       (h_no_error: ∀ i, no_error x y b a i) : R_squared = 1 := by
  sorry

end R2_perfect_fit_l85_85294


namespace positive_difference_l85_85852

theorem positive_difference:
  let a := (7^3 + 7^3) / 7
  let b := (7^3)^2 / 7
  b - a = 16709 :=
by
  sorry

end positive_difference_l85_85852


namespace general_formula_sequence_less_than_zero_maximum_sum_value_l85_85805

variable (n : ℕ)

-- Helper definition
def arithmetic_seq (d : ℤ) (a₁ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Conditions given in the problem
def a₁ : ℤ := 31
def a₄ : ℤ := 7
def d : ℤ := (a₄ - a₁) / 3

-- Definitions extracted from problem conditions
def an (n : ℕ) : ℤ := arithmetic_seq d a₁ n
def Sn (n : ℕ) : ℤ := n * a₁ + (n * (n - 1) / 2) * d

-- Proving the general formula aₙ = -8n + 39
theorem general_formula :
  ∀ (n : ℕ), an n = -8 * n + 39 :=
by
  sorry

-- Proving when the sequence starts to be less than 0
theorem sequence_less_than_zero :
  ∀ (n : ℕ), n ≥ 5 → an n < 0 :=
by
  sorry

-- Proving that the sum Sn has a maximum value
theorem maximum_sum_value :
  Sn 4 = 76 ∧ ∀ (n : ℕ), Sn n ≤ 76 :=
by
  sorry

end general_formula_sequence_less_than_zero_maximum_sum_value_l85_85805


namespace find_x2_x1_add_x3_l85_85220

-- Definition of the polynomial
def polynomial (x : ℝ) : ℝ := (10*x^3 - 210*x^2 + 3)

-- Statement including conditions and the question we need to prove
theorem find_x2_x1_add_x3 :
  ∃ x₁ x₂ x₃ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ 
    polynomial x₁ = 0 ∧ 
    polynomial x₂ = 0 ∧ 
    polynomial x₃ = 0 ∧ 
    x₂ * (x₁ + x₃) = 21 :=
by sorry

end find_x2_x1_add_x3_l85_85220


namespace c_share_l85_85747

theorem c_share (a b c : ℕ) (k : ℕ) 
    (h1 : a + b + c = 1010)
    (h2 : a - 25 = 3 * k) 
    (h3 : b - 10 = 2 * k) 
    (h4 : c - 15 = 5 * k) 
    : c = 495 := 
sorry

end c_share_l85_85747


namespace intersection_unique_one_point_l85_85838

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 7 * x + a
noncomputable def g (x : ℝ) : ℝ := -3 * x^2 + 5 * x - 6

theorem intersection_unique_one_point (a : ℝ) :
  (∃ x y, y = f x a ∧ y = g x) ↔ a = 3 := by
  sorry

end intersection_unique_one_point_l85_85838


namespace vans_needed_l85_85919

-- Given Conditions
def van_capacity : ℕ := 4
def students : ℕ := 2
def adults : ℕ := 6
def total_people : ℕ := students + adults

-- Theorem to prove
theorem vans_needed : total_people / van_capacity = 2 :=
by
  -- Proof will be added here
  sorry

end vans_needed_l85_85919


namespace trigonometric_identity_l85_85931

theorem trigonometric_identity (α : ℝ) 
  (h : Real.cos (5 * Real.pi / 12 - α) = Real.sqrt 2 / 3) :
  Real.sqrt 3 * Real.cos (2 * α) - Real.sin (2 * α) = 10 / 9 := sorry

end trigonometric_identity_l85_85931


namespace find_RS_length_l85_85741

-- Define the given conditions
def tetrahedron_edges (a b c d e f : ℕ) : Prop :=
  (a = 8 ∨ a = 14 ∨ a = 19 ∨ a = 28 ∨ a = 37 ∨ a = 42) ∧
  (b = 8 ∨ b = 14 ∨ b = 19 ∨ b = 28 ∨ b = 37 ∨ b = 42) ∧
  (c = 8 ∨ c = 14 ∨ c = 19 ∨ c = 28 ∨ c = 37 ∨ c = 42) ∧
  (d = 8 ∨ d = 14 ∨ d = 19 ∨ d = 28 ∨ d = 37 ∨ d = 42) ∧
  (e = 8 ∨ e = 14 ∨ e = 19 ∨ e = 28 ∨ e = 37 ∨ e = 42) ∧
  (f = 8 ∨ f = 14 ∨ f = 19 ∨ f = 28 ∨ f = 37 ∨ f = 42) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def length_of_PQ (pq : ℕ) : Prop := pq = 42

def length_of_RS (rs : ℕ) (a b c d e f pq : ℕ) : Prop :=
  tetrahedron_edges a b c d e f ∧ length_of_PQ pq →
  (rs = 14)

-- The theorem statement
theorem find_RS_length (a b c d e f pq rs : ℕ) :
  tetrahedron_edges a b c d e f ∧ length_of_PQ pq →
  length_of_RS rs a b c d e f pq :=
by sorry

end find_RS_length_l85_85741


namespace simplify_polynomials_l85_85365

theorem simplify_polynomials :
  (3 * x^3 + 4 * x^2 + 6 * x - 5) - (2 * x^3 + 2 * x^2 + 3 * x - 8) = x^3 + 2 * x^2 + 3 * x + 3 :=
by
  sorry

end simplify_polynomials_l85_85365


namespace smallest_common_multiple_l85_85685

theorem smallest_common_multiple (n : ℕ) (h8 : n % 8 = 0) (h15 : n % 15 = 0) : n = 120 :=
sorry

end smallest_common_multiple_l85_85685


namespace perimeter_of_triangle_l85_85942

theorem perimeter_of_triangle (side_length : ℕ) (num_sides : ℕ) (h1 : side_length = 7) (h2 : num_sides = 3) : 
  num_sides * side_length = 21 :=
by
  sorry

end perimeter_of_triangle_l85_85942


namespace find_number_l85_85256

theorem find_number (x : ℝ) : (x * 12) / (180 / 3) + 80 = 81 → x = 5 :=
by
  sorry

end find_number_l85_85256


namespace value_of_expression_l85_85705

theorem value_of_expression 
  (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 + 49 * x7 = 1) 
  (h2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 + 64 * x7 = 12) 
  (h3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 + 81 * x7 = 123) 
  : 16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 + 100 * x7 = 334 :=
by
  sorry

end value_of_expression_l85_85705


namespace solve_system_equations_l85_85550

noncomputable def system_equations : Prop :=
  ∃ x y : ℝ,
    (8 * x^2 - 26 * x * y + 15 * y^2 + 116 * x - 150 * y + 360 = 0) ∧
    (8 * x^2 + 18 * x * y - 18 * y^2 + 60 * x + 45 * y + 108 = 0) ∧
    ((x = 0 ∧ y = 4) ∨ (x = -7.5 ∧ y = 1) ∨ (x = -4.5 ∧ y = 0))

theorem solve_system_equations : system_equations := 
by
  sorry

end solve_system_equations_l85_85550


namespace identify_triangle_centers_l85_85206

variable (P : Fin 7 → Type)
variable (I O H L G N K : Type)
variable (P1 P2 P3 P4 P5 P6 P7 : Type)
variable (cond : (P 1 = K) ∧ (P 2 = O) ∧ (P 3 = L) ∧ (P 4 = I) ∧ (P 5 = N) ∧ (P 6 = G) ∧ (P 7 = H))

theorem identify_triangle_centers :
  (P 1 = K) ∧ (P 2 = O) ∧ (P 3 = L) ∧ (P 4 = I) ∧ (P 5 = N) ∧ (P 6 = G) ∧ (P 7 = H) :=
by sorry

end identify_triangle_centers_l85_85206


namespace min_stool_height_l85_85351

/-
Alice needs to reach a ceiling fan switch located 15 centimeters below a 3-meter-tall ceiling.
Alice is 160 centimeters tall and can reach 50 centimeters above her head. She uses a stack of books
12 centimeters tall to assist her reach. We aim to show that the minimum height of the stool she needs is 63 centimeters.
-/

def ceiling_height_cm : ℕ := 300
def alice_height_cm : ℕ := 160
def reach_above_head_cm : ℕ := 50
def books_height_cm : ℕ := 12
def switch_below_ceiling_cm : ℕ := 15

def total_reach_with_books := alice_height_cm + reach_above_head_cm + books_height_cm
def switch_height_from_floor := ceiling_height_cm - switch_below_ceiling_cm

theorem min_stool_height : total_reach_with_books + 63 = switch_height_from_floor := by
  unfold total_reach_with_books switch_height_from_floor
  sorry

end min_stool_height_l85_85351


namespace emily_sixth_score_needed_l85_85183

def emily_test_scores : List ℕ := [88, 92, 85, 90, 97]

def needed_sixth_score (scores : List ℕ) (target_mean : ℕ) : ℕ :=
  let current_sum := scores.sum
  let total_sum_needed := target_mean * (scores.length + 1)
  total_sum_needed - current_sum

theorem emily_sixth_score_needed :
  needed_sixth_score emily_test_scores 91 = 94 := by
  sorry

end emily_sixth_score_needed_l85_85183


namespace part_I_part_II_l85_85571

noncomputable def f (x : ℝ) : ℝ := 5 + Real.log x
noncomputable def g (x k : ℝ) : ℝ := k * x / (x + 1)

theorem part_I (k : ℝ) : 
  (∃ x0, g x0 k = x0 + 4 ∧ (k / (x0 + 1)^2) = 1) ↔ (k = 1 ∨ k = 9) :=
by
  sorry

theorem part_II (k : ℕ) : (∀ x : ℝ, 1 < x → f x > g x k) → k ≤ 7 :=
by
  sorry

end part_I_part_II_l85_85571


namespace log_comparison_l85_85008

theorem log_comparison (a b c : ℝ) (h₁ : a = Real.log 6 / Real.log 4) (h₂ : b = Real.log 3 / Real.log 2) (h₃ : c = 3/2) : b > c ∧ c > a := 
by 
  sorry

end log_comparison_l85_85008


namespace avg_height_and_weight_of_class_l85_85790

-- Defining the given conditions
def num_students : ℕ := 70
def num_girls : ℕ := 40
def num_boys : ℕ := 30

def avg_height_30_girls : ℕ := 160
def avg_height_10_girls : ℕ := 156
def avg_height_15_boys_high : ℕ := 170
def avg_height_15_boys_low : ℕ := 160
def avg_weight_girls : ℕ := 55
def avg_weight_boys : ℕ := 60

-- Theorem stating the given question
theorem avg_height_and_weight_of_class :
  ∃ (avg_height avg_weight : ℚ),
    avg_height = (30 * 160 + 10 * 156 + 15 * 170 + 15 * 160) / num_students ∧
    avg_weight = (40 * 55 + 30 * 60) / num_students ∧
    avg_height = 161.57 ∧
    avg_weight = 57.14 :=
by
  -- include the solution steps here if required
  -- examples using appropriate constructs like ring, norm_num, etc.
  sorry

end avg_height_and_weight_of_class_l85_85790


namespace find_tan_α_l85_85127

variable (α : ℝ) (h1 : Real.sin (α - Real.pi / 3) = 3 / 5)
variable (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2)

theorem find_tan_α (h1 : Real.sin (α - Real.pi / 3) = 3 / 5) (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2) : 
  Real.tan α = - (48 + 25 * Real.sqrt 3) / 11 :=
sorry

end find_tan_α_l85_85127


namespace triangle_median_length_l85_85455

variable (XY XZ XM YZ : ℝ)

theorem triangle_median_length :
  XY = 6 →
  XZ = 8 →
  XM = 5 →
  YZ = 10 := by
  sorry

end triangle_median_length_l85_85455


namespace water_consumed_is_correct_l85_85354

def water_consumed (traveler_ounces : ℕ) (camel_multiplier : ℕ) (ounces_per_gallon : ℕ) : ℕ :=
  let camel_ounces := traveler_ounces * camel_multiplier
  let total_ounces := traveler_ounces + camel_ounces
  total_ounces / ounces_per_gallon

theorem water_consumed_is_correct :
  water_consumed 32 7 128 = 2 :=
by
  -- add proof here
  sorry

end water_consumed_is_correct_l85_85354


namespace fish_lives_longer_than_dog_l85_85961

-- Definitions based on conditions
def hamster_lifespan : ℝ := 2.5
def dog_lifespan : ℝ := 4 * hamster_lifespan
def fish_lifespan : ℝ := 12

-- Theorem stating the desired proof
theorem fish_lives_longer_than_dog :
  fish_lifespan - dog_lifespan = 2 := 
sorry

end fish_lives_longer_than_dog_l85_85961


namespace sin_neg_45_l85_85005

theorem sin_neg_45 :
  Real.sin (-45 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
sorry

end sin_neg_45_l85_85005


namespace math_players_count_l85_85969

-- Define the conditions given in the problem.
def total_players : ℕ := 25
def physics_players : ℕ := 9
def both_subjects_players : ℕ := 5

-- Statement to be proven
theorem math_players_count :
  total_players = physics_players + both_subjects_players + (total_players - physics_players - both_subjects_players) → 
  total_players - physics_players + both_subjects_players = 21 := 
sorry

end math_players_count_l85_85969


namespace negate_proposition_l85_85291

theorem negate_proposition (x y : ℝ) :
  (¬ (x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔ (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0)) :=
by
  sorry

end negate_proposition_l85_85291


namespace average_of_numbers_eq_x_l85_85890

theorem average_of_numbers_eq_x (x : ℝ) (h : (2 + x + 10) / 3 = x) : x = 6 := 
by sorry

end average_of_numbers_eq_x_l85_85890


namespace length_of_string_C_l85_85722

theorem length_of_string_C (A B C : ℕ) (h1 : A = 6 * C) (h2 : A = 5 * B) (h3 : B = 12) : C = 10 :=
sorry

end length_of_string_C_l85_85722


namespace right_triangular_pyramid_property_l85_85060

theorem right_triangular_pyramid_property
  (S1 S2 S3 S : ℝ)
  (right_angle_face1_area : S1 = S1) 
  (right_angle_face2_area : S2 = S2) 
  (right_angle_face3_area : S3 = S3) 
  (oblique_face_area : S = S) :
  S1^2 + S2^2 + S3^2 = S^2 := 
sorry

end right_triangular_pyramid_property_l85_85060


namespace monotonically_decreasing_interval_l85_85009

noncomputable def f (x : ℝ) : ℝ :=
  (2 * Real.exp 2) * Real.exp (x - 2) - 2 * x + 1/2 * x^2

theorem monotonically_decreasing_interval :
  ∀ x : ℝ, x < 0 → ((2 * Real.exp x - 2 + x) < 0) :=
by
  sorry

end monotonically_decreasing_interval_l85_85009


namespace projection_plane_right_angle_l85_85167

-- Given conditions and definitions
def is_right_angle (α β : ℝ) : Prop := α = 90 ∧ β = 90
def is_parallel_to_side (plane : ℝ → ℝ → Prop) (side : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, plane x y ↔ a * x + b * y = c ∧ ∃ d e : ℝ, ∀ x y : ℝ, side x y ↔ d * x + e * y = 90

theorem projection_plane_right_angle (plane : ℝ → ℝ → Prop) (side1 side2 : ℝ → ℝ → Prop) :
  is_right_angle (90 : ℝ) (90 : ℝ) →
  (is_parallel_to_side plane side1 ∨ is_parallel_to_side plane side2) →
  ∃ α β : ℝ, is_right_angle α β :=
by 
  sorry

end projection_plane_right_angle_l85_85167


namespace provisions_initial_days_l85_85763

theorem provisions_initial_days (D : ℕ) (P : ℕ) (Q : ℕ) (X : ℕ) (Y : ℕ)
  (h1 : P = 300) 
  (h2 : X = 30) 
  (h3 : Y = 90) 
  (h4 : Q = 200) 
  (h5 : P * D = P * X + Q * Y) : D + X = 120 :=
by
  -- We need to prove that the initial number of days the provisions were meant to last is 120.
  sorry

end provisions_initial_days_l85_85763


namespace probability_calculation_l85_85730

open Classical

def probability_odd_sum_given_even_product :=
  let num_even := 4  -- even numbers: 2, 4, 6, 8
  let num_odd := 4   -- odd numbers: 1, 3, 5, 7
  let total_outcomes := 8^5
  let prob_all_odd := (num_odd / 8)^5
  let prob_even_product := 1 - prob_all_odd

  let ways_one_odd := 5 * num_odd * num_even^4
  let ways_three_odd := Nat.choose 5 3 * num_odd^3 * num_even^2
  let ways_five_odd := num_odd^5

  let favorable_outcomes := ways_one_odd + ways_three_odd + ways_five_odd
  let total_even_product_outcomes := total_outcomes * prob_even_product

  favorable_outcomes / total_even_product_outcomes

theorem probability_calculation :
  probability_odd_sum_given_even_product = rational_result := sorry

end probability_calculation_l85_85730


namespace tangent_line_equation_l85_85714

theorem tangent_line_equation (e x y : ℝ) (h_curve : y = x^3 / e) (h_point : x = e ∧ y = e^2) :
  3 * e * x - y - 2 * e^2 = 0 :=
sorry

end tangent_line_equation_l85_85714


namespace vacation_cost_l85_85725

theorem vacation_cost (n : ℕ) (h : 480 / n + 40 = 120) : n = 6 :=
sorry

end vacation_cost_l85_85725


namespace probability_length_error_in_interval_l85_85303

noncomputable def normal_dist_prob (μ σ : ℝ) (a b : ℝ) : ℝ :=
∫ x in a..b, (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ) ^ 2) / (2 * σ ^ 2))

theorem probability_length_error_in_interval :
  normal_dist_prob 0 3 3 6 = 0.1359 :=
by
  sorry

end probability_length_error_in_interval_l85_85303


namespace no_integer_solutions_l85_85072

theorem no_integer_solutions (x y z : ℤ) :
  x^2 - 3 * x * y + 2 * y^2 - z^2 = 31 ∧
  -x^2 + 6 * y * z + 2 * z^2 = 44 ∧
  x^2 + x * y + 8 * z^2 = 100 →
  false :=
by
  sorry

end no_integer_solutions_l85_85072


namespace brenda_spay_cats_l85_85221

theorem brenda_spay_cats (c d : ℕ) (h1 : c + d = 21) (h2 : d = 2 * c) : c = 7 :=
sorry

end brenda_spay_cats_l85_85221


namespace Koschei_no_equal_coins_l85_85408

theorem Koschei_no_equal_coins (a : Fin 6 → ℕ)
  (initial_condition : a 0 = 1 ∧ a 1 = 0 ∧ a 2 = 0 ∧ a 3 = 0 ∧ a 4 = 0 ∧ a 5 = 0) :
  ¬ ( ∃ k : ℕ, ( ( ∀ i : Fin 6, a i = k ) ) ) :=
by
  sorry

end Koschei_no_equal_coins_l85_85408


namespace tangent_line_equation_at_1_l85_85463

-- Define the function f and the point of tangency
def f (x : ℝ) : ℝ := x^2 + 2 * x
def p : ℝ × ℝ := (1, f 1)

-- Statement of the theorem
theorem tangent_line_equation_at_1 :
  ∃ a b c : ℝ, (∀ x y : ℝ, y = f x → y - p.2 = a * (x - p.1)) ∧
               4 * (p.1 : ℝ) - (p.2 : ℝ) - 1 = 0 :=
by
  -- Skipping the proof
  sorry

end tangent_line_equation_at_1_l85_85463


namespace sum_of_row_10_pascals_triangle_sum_of_squares_of_row_10_pascals_triangle_l85_85278

def row_10_pascals_triangle : List ℕ := [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]

theorem sum_of_row_10_pascals_triangle :
  (List.sum row_10_pascals_triangle) = 1024 := by
  sorry

theorem sum_of_squares_of_row_10_pascals_triangle :
  (List.sum (List.map (fun x => x * x) row_10_pascals_triangle)) = 183756 := by
  sorry

end sum_of_row_10_pascals_triangle_sum_of_squares_of_row_10_pascals_triangle_l85_85278


namespace negation_statement_l85_85227

variable {α : Type} (S : Set α)

theorem negation_statement (P : α → Prop) :
  (∀ x ∈ S, ¬ P x) ↔ (∃ x ∈ S, P x) :=
by
  sorry

end negation_statement_l85_85227


namespace sum_digit_product_1001_to_2011_l85_85385

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).foldr (λ d acc => d * acc) 1

theorem sum_digit_product_1001_to_2011 :
  (Finset.range 1011).sum (λ k => digit_product (1001 + k)) = 91125 :=
by
  sorry

end sum_digit_product_1001_to_2011_l85_85385


namespace lightest_pumpkin_weight_l85_85913

theorem lightest_pumpkin_weight 
  (A B C : ℕ)
  (h₁ : A + B = 12)
  (h₂ : B + C = 15)
  (h₃ : A + C = 13) :
  A = 5 :=
by
  sorry

end lightest_pumpkin_weight_l85_85913


namespace calc_derivative_at_pi_over_2_l85_85911

noncomputable def f (x: ℝ) : ℝ := Real.exp x * Real.cos x

theorem calc_derivative_at_pi_over_2 : (deriv f) (Real.pi / 2) = -Real.exp (Real.pi / 2) :=
by
  sorry

end calc_derivative_at_pi_over_2_l85_85911


namespace marathons_yards_l85_85996

theorem marathons_yards
  (miles_per_marathon : ℕ)
  (yards_per_marathon : ℕ)
  (miles_in_yard : ℕ)
  (marathons_run : ℕ)
  (total_miles : ℕ)
  (total_yards : ℕ)
  (y : ℕ) :
  miles_per_marathon = 30
  → yards_per_marathon = 520
  → miles_in_yard = 1760
  → marathons_run = 8
  → total_miles = (miles_per_marathon * marathons_run) + (yards_per_marathon * marathons_run) / miles_in_yard
  → total_yards = (yards_per_marathon * marathons_run) % miles_in_yard
  → y = 640 := 
by
  intros
  sorry

end marathons_yards_l85_85996


namespace no_real_roots_eq_xsq_abs_x_plus_1_eq_0_l85_85622

theorem no_real_roots_eq_xsq_abs_x_plus_1_eq_0 :
  ¬ ∃ x : ℝ, x^2 + abs x + 1 = 0 :=
by
  sorry

end no_real_roots_eq_xsq_abs_x_plus_1_eq_0_l85_85622


namespace intersection_eq_l85_85436

def M := {x : ℝ | x < 1}
def N := {x : ℝ | Real.log x / Real.log 2 < 1}

theorem intersection_eq : {x : ℝ | x ∈ M ∧ x ∈ N} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l85_85436


namespace evaluate_polynomial_at_3_l85_85592

def f (x : ℕ) : ℕ := 3 * x^7 + 2 * x^5 + 4 * x^3 + x

theorem evaluate_polynomial_at_3 : f 3 = 7158 := by
  sorry

end evaluate_polynomial_at_3_l85_85592


namespace expression_is_minus_two_l85_85837

noncomputable def A : ℝ := (Real.sqrt 6 + Real.sqrt 2) * (Real.sqrt 3 - 2) * Real.sqrt (Real.sqrt 3 + 2)

theorem expression_is_minus_two : A = -2 := by
  sorry

end expression_is_minus_two_l85_85837


namespace solve_abs_inequality_l85_85373

theorem solve_abs_inequality (x : ℝ) :
  (|x - 2| + |x - 4| > 6) ↔ (x < 0 ∨ 12 < x) :=
by
  sorry

end solve_abs_inequality_l85_85373


namespace leak_time_to_empty_tank_l85_85447

theorem leak_time_to_empty_tank :
  let rateA := 1 / 2  -- rate at which pipe A fills the tank (tanks per hour)
  let rateB := 2 / 3  -- rate at which pipe B fills the tank (tanks per hour)
  let combined_rate_without_leak := rateA + rateB  -- combined rate without leak
  let combined_rate_with_leak := 1 / 1.75  -- combined rate with leak (tanks per hour)
  let leak_rate := combined_rate_without_leak - combined_rate_with_leak  -- rate of the leak (tanks per hour)
  60 / leak_rate = 100.8 :=  -- time to empty the tank by the leak (minutes)
    by sorry

end leak_time_to_empty_tank_l85_85447


namespace false_p_and_q_l85_85202

variable {a : ℝ} 

def p (a : ℝ) := 3 * a / 2 ≤ 1
def q (a : ℝ) := 0 < 2 * a - 1 ∧ 2 * a - 1 < 1

theorem false_p_and_q (a : ℝ) :
  ¬ (p a ∧ q a) ↔ (a ≤ (1 : ℝ) / 2 ∨ a > (2 : ℝ) / 3) :=
by
  sorry

end false_p_and_q_l85_85202


namespace shaded_region_area_l85_85860

noncomputable def area_of_shaded_region (r : ℝ) (oa : ℝ) (ab_length : ℝ) : ℝ :=
  18 * (Real.sqrt 2) - 9 - (9 * Real.pi / 4)

theorem shaded_region_area (r : ℝ) (oa : ℝ) (ab_length : ℝ) : 
  r = 3 ∧ oa = 3 * Real.sqrt 2 ∧ ab_length = 6 * Real.sqrt 2 → 
  area_of_shaded_region r oa ab_length = 18 * (Real.sqrt 2) - 9 - (9 * Real.pi / 4) :=
by
  intro h
  obtain ⟨hr, hoa, hab⟩ := h
  rw [hr, hoa, hab]
  exact rfl

end shaded_region_area_l85_85860


namespace percent_gain_is_5_333_l85_85203

noncomputable def calculate_percent_gain (total_sheep : ℕ) 
                                         (sold_sheep : ℕ) 
                                         (price_paid_sheep : ℕ) 
                                         (sold_remaining_sheep : ℕ)
                                         (remaining_sheep : ℕ) 
                                         (total_cost : ℝ) 
                                         (initial_revenue : ℝ) 
                                         (remaining_revenue : ℝ) : ℝ :=
  (remaining_revenue + initial_revenue - total_cost) / total_cost * 100

theorem percent_gain_is_5_333
  (x : ℝ)
  (total_sheep : ℕ := 800)
  (sold_sheep : ℕ := 750)
  (price_paid_sheep : ℕ := 790)
  (remaining_sheep : ℕ := 50)
  (total_cost : ℝ := (800 : ℝ) * x)
  (initial_revenue : ℝ := (790 : ℝ) * x)
  (remaining_revenue : ℝ := (50 : ℝ) * ((790 : ℝ) * x / 750)) :
  calculate_percent_gain total_sheep sold_sheep price_paid_sheep remaining_sheep 50 total_cost initial_revenue remaining_revenue = 5.333 := by
  sorry

end percent_gain_is_5_333_l85_85203


namespace words_per_page_l85_85352

theorem words_per_page (p : ℕ) (h1 : p ≤ 150) (h2 : 120 * p ≡ 172 [MOD 221]) : p = 114 := by
  sorry

end words_per_page_l85_85352


namespace inequality_one_inequality_two_l85_85547

-- First Inequality Problem
theorem inequality_one (a b c d : ℝ) (h : a + b + c + d = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) + (1 / d^2) ≤ (1 / (a^2 * b^2 * c^2 * d^2)) :=
sorry

-- Second Inequality Problem
theorem inequality_two (a b c d : ℝ) (h : a + b + c + d = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) + (1 / d^3) ≤ (1 / (a^3 * b^3 * c^3 * d^3)) :=
sorry

end inequality_one_inequality_two_l85_85547


namespace value_of_expression_l85_85320

theorem value_of_expression (x : ℝ) (h : x = 1 + Real.sqrt 2) : x^4 - 4 * x^3 + 4 * x^2 + 4 = 5 := by
  sorry

end value_of_expression_l85_85320


namespace problem_1_problem_2_l85_85030

def condition_p (x : ℝ) : Prop := 4 * x ^ 2 + 12 * x - 7 ≤ 0
def condition_q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Problem 1: When a=0, if p is true and q is false, the range of real numbers x
theorem problem_1 (x : ℝ) :
  condition_p x ∧ ¬ condition_q 0 x ↔ -7/2 ≤ x ∧ x < -3 := sorry

-- Problem 2: If p is a sufficient condition for q, the range of real numbers a
theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, condition_p x → condition_q a x) ↔ -5/2 ≤ a ∧ a ≤ -1/2 := sorry

end problem_1_problem_2_l85_85030


namespace fraction_of_married_men_is_two_fifths_l85_85561

noncomputable def fraction_of_married_men (W : ℕ) (p : ℚ) (h : p = 1 / 3) : ℚ :=
  let W_s := p * W
  let W_m := W - W_s
  let M_m := W_m
  let T := W + M_m
  M_m / T

theorem fraction_of_married_men_is_two_fifths (W : ℕ) (p : ℚ) (h : p = 1 / 3) (hW : W = 6) : fraction_of_married_men W p h = 2 / 5 :=
by
  sorry

end fraction_of_married_men_is_two_fifths_l85_85561


namespace problem1_problem2_l85_85851

-- Define the function f(x)
def f (x m : ℝ) : ℝ := abs (x - m) - abs (x + 3 * m)

-- Condition that m must be greater than 0
variable {m : ℝ} (hm : m > 0)

-- First problem statement: When m=1, the solution set for f(x) ≥ 1 is x ≤ -3/2.
theorem problem1 (x : ℝ) (h : f x 1 ≥ 1) : x ≤ -3 / 2 :=
sorry

-- Second problem statement: The range of values for m such that f(x) < |2 + t| + |t - 1| holds for all x and t is 0 < m < 3/4.
theorem problem2 (m : ℝ) : (∀ (x t : ℝ), f x m < abs (2 + t) + abs (t - 1)) ↔ (0 < m ∧ m < 3 / 4) :=
sorry

end problem1_problem2_l85_85851


namespace factorize_l85_85482

variables (a b x y : ℝ)

theorem factorize : (a * x - b * y)^2 + (a * y + b * x)^2 = (x^2 + y^2) * (a^2 + b^2) :=
by
  sorry

end factorize_l85_85482


namespace set_equality_implies_a_value_l85_85963

theorem set_equality_implies_a_value (a : ℤ) : ({2, 3} : Set ℤ) = {2, 2 * a - 1} → a = 2 := 
by
  intro h
  sorry

end set_equality_implies_a_value_l85_85963


namespace A_eq_B_l85_85514

variables (α : Type) (Q : α → Prop)
variables (A B C : α → Prop)

-- Conditions
-- 1. For the questions where both B and C answered "yes", A also answered "yes".
axiom h1 : ∀ q, B q ∧ C q → A q
-- 2. For the questions where A answered "yes", B also answered "yes".
axiom h2 : ∀ q, A q → B q
-- 3. For the questions where B answered "yes", at least one of A and C answered "yes".
axiom h3 : ∀ q, B q → (A q ∨ C q)

-- Prove that A and B gave the same answer to all questions
theorem A_eq_B : ∀ q, A q ↔ B q :=
sorry

end A_eq_B_l85_85514


namespace largest_common_value_l85_85431

theorem largest_common_value :
  ∃ (a : ℕ), (∃ (n m : ℕ), a = 4 + 5 * n ∧ a = 5 + 10 * m) ∧ a < 1000 ∧ a = 994 :=
by {
  sorry
}

end largest_common_value_l85_85431


namespace interest_rate_l85_85692

theorem interest_rate (SI P T R : ℝ) (h1 : SI = 100) (h2 : P = 500) (h3 : T = 4) (h4 : SI = (P * R * T) / 100) :
  R = 5 :=
by
  sorry

end interest_rate_l85_85692


namespace ben_paints_area_l85_85843

variable (allen_ratio : ℕ) (ben_ratio : ℕ) (total_area : ℕ)
variable (total_ratio : ℕ := allen_ratio + ben_ratio)
variable (part_size : ℕ := total_area / total_ratio)

theorem ben_paints_area 
  (h1 : allen_ratio = 2)
  (h2 : ben_ratio = 6)
  (h3 : total_area = 360) : 
  ben_ratio * part_size = 270 := sorry

end ben_paints_area_l85_85843


namespace inequality_proof_l85_85716

theorem inequality_proof (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^4 / (y-1)^2) + (y^4 / (z-1)^2) + (z^4 / (x-1)^2) ≥ 48 := 
by
  sorry -- The actual proof is omitted

end inequality_proof_l85_85716


namespace shopping_center_expense_l85_85159

theorem shopping_center_expense
    (films_count : ℕ := 9)
    (films_original_price : ℝ := 7)
    (film_discount : ℝ := 2)
    (books_full_price : ℝ := 10)
    (books_count : ℕ := 5)
    (books_discount_rate : ℝ := 0.25)
    (cd_price : ℝ := 4.50)
    (cd_count : ℕ := 6)
    (tax_rate : ℝ := 0.06)
    (total_amount_spent : ℝ := 109.18) :
    let films_total := films_count * (films_original_price - film_discount)
    let remaining_books := books_count - 1
    let discounted_books_total := remaining_books * (books_full_price * (1 - books_discount_rate))
    let books_total := books_full_price + discounted_books_total
    let cds_paid_count := cd_count - (cd_count / 3)
    let cds_total := cds_paid_count * cd_price
    let total_before_tax := films_total + books_total + cds_total
    let tax := total_before_tax * tax_rate
    let total_with_tax := total_before_tax + tax
    total_with_tax = total_amount_spent :=
by
  sorry

end shopping_center_expense_l85_85159


namespace larger_number_is_299_l85_85526

theorem larger_number_is_299 {a b : ℕ} (hcf : Nat.gcd a b = 23) (lcm_factors : ∃ k1 k2 : ℕ, Nat.lcm a b = 23 * k1 * k2 ∧ k1 = 12 ∧ k2 = 13) :
  max a b = 299 :=
by
  sorry

end larger_number_is_299_l85_85526


namespace math_problem_l85_85166

theorem math_problem (a b c d e : ℤ) (x : ℤ) (hx : x > 196)
  (h1 : a + b = 183) (h2 : a + c = 186) (h3 : d + e = x) (h4 : c + e = 196)
  (h5 : 183 < 186) (h6 : 186 < 187) (h7 : 187 < 190) (h8 : 190 < 191) (h9 : 191 < 192)
  (h10 : 192 < 193) (h11 : 193 < 194) (h12 : 194 < 196) (h13 : 196 < x) :
  (a = 91 ∧ b = 92 ∧ c = 95 ∧ d = 99 ∧ e = 101 ∧ x = 200) ∧ (∃ y, y = 10 * x + 3 ∧ y = 2003) :=
by
  sorry

end math_problem_l85_85166


namespace simplify_expression_l85_85025

theorem simplify_expression (θ : ℝ) : 
  ((1 + Real.sin θ ^ 2) ^ 2 - Real.cos θ ^ 4) * ((1 + Real.cos θ ^ 2) ^ 2 - Real.sin θ ^ 4) = 4 * Real.sin (2 * θ) ^ 2 :=
by 
  sorry

end simplify_expression_l85_85025


namespace units_digit_of_p_is_6_l85_85193

-- Given conditions
variable (p : ℕ)
variable (h1 : p % 2 = 0)                -- p is a positive even integer
variable (h2 : (p^3 % 10) - (p^2 % 10) = 0)  -- The units digit of p^3 minus the units digit of p^2 is 0
variable (h3 : (p + 2) % 10 = 8)         -- The units digit of p + 2 is 8

-- Prove the units digit of p is 6
theorem units_digit_of_p_is_6 : p % 10 = 6 :=
sorry

end units_digit_of_p_is_6_l85_85193


namespace infinite_squares_in_arithmetic_progression_l85_85388

theorem infinite_squares_in_arithmetic_progression
  (a d : ℕ) (hposd : 0 < d) (hpos : 0 < a) (k n : ℕ)
  (hk : a + k * d = n^2) :
  ∃ (t : ℕ), ∃ (m : ℕ), (a + (k + t) * d = m^2) := by
  sorry

end infinite_squares_in_arithmetic_progression_l85_85388


namespace find_special_numbers_l85_85565

/-- Define the sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Define the main statement to be proved -/
theorem find_special_numbers :
  { n : ℕ | sum_of_digits n * (sum_of_digits n - 1) = n - 1 } = {1, 13, 43, 91, 157} :=
by
  sorry

end find_special_numbers_l85_85565


namespace tire_price_l85_85534

-- Definitions based on given conditions
def tire_cost (T : ℝ) (n : ℕ) : Prop :=
  n * T + 56 = 224

-- The equivalence we want to prove
theorem tire_price (T : ℝ) (n : ℕ) (h : tire_cost T n) : n * T = 168 :=
by
  sorry

end tire_price_l85_85534


namespace ribbons_jane_uses_l85_85372

-- Given conditions
def dresses_sewn_first_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def dresses_sewn_second_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def total_dresses_sewn (dresses_first_period : ℕ) (dresses_second_period : ℕ) : ℕ :=
  dresses_first_period + dresses_second_period

def total_ribbons_used (total_dresses : ℕ) (ribbons_per_dress : ℕ) : ℕ :=
  total_dresses * ribbons_per_dress

-- Theorem to prove
theorem ribbons_jane_uses :
  total_ribbons_used (total_dresses_sewn (dresses_sewn_first_period 2 7) (dresses_sewn_second_period 3 2)) 2 = 40 :=
  sorry

end ribbons_jane_uses_l85_85372


namespace son_age_l85_85507

theorem son_age {M S : ℕ} 
  (h1 : M = S + 18) 
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 16 := 
by
  sorry

end son_age_l85_85507


namespace average_weight_l85_85796

def weights (A B C : ℝ) : Prop :=
  (A + B + C = 135) ∧
  (B + C = 86) ∧
  (B = 31)

theorem average_weight (A B C : ℝ) (h : weights A B C) :
  (A + B) / 2 = 40 :=
by
  sorry

end average_weight_l85_85796


namespace cara_total_bread_l85_85813

theorem cara_total_bread 
  (d : ℕ) (L : ℕ) (B : ℕ) (S : ℕ) 
  (h_dinner : d = 240) 
  (h_lunch : d = 8 * L) 
  (h_breakfast : d = 6 * B) 
  (h_snack : d = 4 * S) : 
  d + L + B + S = 370 := 
sorry

end cara_total_bread_l85_85813


namespace five_wednesdays_implies_five_saturdays_in_august_l85_85031

theorem five_wednesdays_implies_five_saturdays_in_august (N : ℕ) (H1 : ∃ ws : Finset ℕ, ws.card = 5 ∧ ∀ w ∈ ws, w < 32 ∧ (w % 7 = 3)) (H2 : July_days = 31) (H3 : August_days = 31):
  ∀ w : ℕ, w < 7 → ∃ ws : Finset ℕ, ws.card = 5 ∧ ∀ sat ∈ ws, (sat % 7 = 6) :=
by
  sorry

end five_wednesdays_implies_five_saturdays_in_august_l85_85031


namespace triangle_perimeter_l85_85543

theorem triangle_perimeter {a b c : ℕ} (ha : a = 10) (hb : b = 6) (hc : c = 7) :
    a + b + c = 23 := by
  sorry

end triangle_perimeter_l85_85543


namespace determine_speeds_l85_85639

structure Particle :=
  (speed : ℝ)

def distance : ℝ := 3.01 -- meters

def initial_distance (m1_speed : ℝ) : ℝ :=
  301 - 11 * m1_speed -- converted to cm

theorem determine_speeds :
  ∃ (m1 m2 : Particle), 
  m1.speed = 11 ∧ m2.speed = 7 ∧ 
  ∀ t : ℝ, (t = 10 ∨ t = 45) →
  (initial_distance m1.speed) = t * (m1.speed + m2.speed) ∧
  20 * m2.speed = 35 * (m1.speed - m2.speed) :=
by {
  sorry 
}

end determine_speeds_l85_85639


namespace sufficient_but_not_necessary_condition_l85_85888

noncomputable def P := {x : ℝ | 0 < x ∧ x < 3}
noncomputable def Q := {x : ℝ | -3 < x ∧ x < 3}

theorem sufficient_but_not_necessary_condition :
  (∀ x, x ∈ P → x ∈ Q) ∧ ¬(∀ x, x ∈ Q → x ∈ P) := by
  sorry

end sufficient_but_not_necessary_condition_l85_85888


namespace amy_uploaded_photos_l85_85052

theorem amy_uploaded_photos (albums photos_per_album : ℕ) (h1 : albums = 9) (h2 : photos_per_album = 20) :
  albums * photos_per_album = 180 :=
by {
  sorry
}

end amy_uploaded_photos_l85_85052


namespace P_intersect_Q_empty_l85_85454

def is_element_of_P (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k / 2 + 1 / 4

def is_element_of_Q (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k / 2 + 1 / 2

theorem P_intersect_Q_empty : ∀ x, is_element_of_P x → is_element_of_Q x → false :=
by
  intro x hP hQ
  sorry

end P_intersect_Q_empty_l85_85454


namespace tour_groups_and_savings_minimum_people_for_savings_l85_85921

theorem tour_groups_and_savings (x y : ℕ) (m : ℕ):
  (x + y = 102) ∧ (45 * x + 50 * y - 40 * 102 = 730) → 
  (x = 58 ∧ y = 44) :=
by
  sorry

theorem minimum_people_for_savings (m : ℕ):
  (∀ m, m < 50 → 50 * m > 45 * 51) → 
  (m ≥ 46) :=
by
  sorry

end tour_groups_and_savings_minimum_people_for_savings_l85_85921


namespace projection_matrix_determinant_l85_85128

theorem projection_matrix_determinant (a c : ℚ) (h : (a^2 + (20 / 49 : ℚ) * c = a) ∧ ((20 / 49 : ℚ) * a + 580 / 2401 = 20 / 49) ∧ (a * c + (29 / 49 : ℚ) * c = c) ∧ ((20 / 49 : ℚ) * c + 841 / 2401 = 29 / 49)) :
  (a = 41 / 49) ∧ (c = 204 / 1225) := 
by {
  sorry
}

end projection_matrix_determinant_l85_85128


namespace condition_2_3_implies_f_x1_greater_f_x2_l85_85371

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.cos x

theorem condition_2_3_implies_f_x1_greater_f_x2 
(x1 x2 : ℝ) (h1 : -2 * Real.pi / 3 ≤ x1 ∧ x1 ≤ 2 * Real.pi / 3) 
(h2 : -2 * Real.pi / 3 ≤ x2 ∧ x2 ≤ 2 * Real.pi / 3) 
(hx1_sq_gt_x2_sq : x1^2 > x2^2) (hx1_gt_abs_x2 : x1 > |x2|) : 
  f x1 > f x2 := 
sorry

end condition_2_3_implies_f_x1_greater_f_x2_l85_85371


namespace cube_sum_divisible_by_six_l85_85630

theorem cube_sum_divisible_by_six
  (a b c : ℤ)
  (h1 : 6 ∣ (a^2 + b^2 + c^2))
  (h2 : 3 ∣ (a * b + b * c + c * a))
  : 6 ∣ (a^3 + b^3 + c^3) := 
sorry

end cube_sum_divisible_by_six_l85_85630


namespace number_of_rectangular_arrays_of_chairs_l85_85212

/-- 
Given a classroom that contains 45 chairs, prove that 
the number of rectangular arrays of chairs that can be made such that 
each row contains at least 3 chairs and each column contains at least 3 chairs is 4.
-/
theorem number_of_rectangular_arrays_of_chairs : 
  ∃ (n : ℕ), n = 4 ∧ 
    ∀ (a b : ℕ), (a * b = 45) → 
      (a ≥ 3) → (b ≥ 3) → 
      (n = 4) := 
sorry

end number_of_rectangular_arrays_of_chairs_l85_85212


namespace retailer_profit_percentage_l85_85914

theorem retailer_profit_percentage
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (h_wholesale_price : wholesale_price = 108)
  (h_retail_price : retail_price = 144)
  (h_discount_rate : discount_rate = 0.10) :
  (retail_price * (1 - discount_rate) - wholesale_price) / wholesale_price * 100 = 20 :=
by
  sorry

end retailer_profit_percentage_l85_85914


namespace exists_sum_of_divisibles_l85_85421

theorem exists_sum_of_divisibles : ∃ (a b: ℕ), a + b = 316 ∧ (13 ∣ a) ∧ (11 ∣ b) :=
by
  existsi 52
  existsi 264
  sorry

end exists_sum_of_divisibles_l85_85421


namespace hexagon_theorem_l85_85554

-- Define a structure for the hexagon with its sides
structure Hexagon :=
(side1 side2 side3 side4 side5 side6 : ℕ)

-- Define the conditions of the problem
def hexagon_conditions (h : Hexagon) : Prop :=
  h.side1 = 5 ∧ h.side2 = 6 ∧ h.side3 = 7 ∧
  (h.side1 + h.side2 + h.side3 + h.side4 + h.side5 + h.side6 = 38)

-- Define the proposition that we need to prove
def hexagon_proposition (h : Hexagon) : Prop :=
  (h.side3 = 7 ∨ h.side4 = 7 ∨ h.side5 = 7 ∨ h.side6 = 7) → 
  (h.side1 = 5 ∧ h.side2 = 6 ∧ h.side3 = 7 ∧ h.side4 = 7 ∧ h.side5 = 7 ∧ h.side6 = 7 → 3 = 3)

-- The proof statement combining conditions and the to-be-proven proposition
theorem hexagon_theorem (h : Hexagon) (hc : hexagon_conditions h) : hexagon_proposition h :=
by
  sorry -- No proof is required

end hexagon_theorem_l85_85554


namespace food_remaining_l85_85240

-- Definitions for conditions
def first_week_donations : ℕ := 40
def second_week_donations := 2 * first_week_donations
def total_donations := first_week_donations + second_week_donations
def percentage_given_out : ℝ := 0.70
def amount_given_out := percentage_given_out * total_donations

-- Proof goal
theorem food_remaining (h1 : first_week_donations = 40)
                      (h2 : second_week_donations = 2 * first_week_donations)
                      (h3 : percentage_given_out = 0.70) :
                      total_donations - amount_given_out = 36 := by
  sorry

end food_remaining_l85_85240


namespace amount_of_money_l85_85545

variable (x : ℝ)

-- Conditions
def condition1 : Prop := x < 2000
def condition2 : Prop := 4 * x > 2000
def condition3 : Prop := 4 * x - 2000 = 2000 - x

theorem amount_of_money (h1 : condition1 x) (h2 : condition2 x) (h3 : condition3 x) : x = 800 :=
by
  sorry

end amount_of_money_l85_85545


namespace no_integer_solutions_l85_85391

theorem no_integer_solutions :
  ∀ x y : ℤ, x^3 + 4 * x^2 - 11 * x + 30 ≠ 8 * y^3 + 24 * y^2 + 18 * y + 7 :=
by sorry

end no_integer_solutions_l85_85391


namespace first_book_cost_correct_l85_85061

noncomputable def cost_of_first_book (x : ℝ) : Prop :=
  let total_cost := x + 6.5
  let given_amount := 20
  let change_received := 8
  total_cost = given_amount - change_received → x = 5.5

theorem first_book_cost_correct : cost_of_first_book 5.5 :=
by
  sorry

end first_book_cost_correct_l85_85061


namespace vertex_of_parabola_is_max_and_correct_l85_85600

theorem vertex_of_parabola_is_max_and_correct (x y : ℝ) (h : y = -3 * x^2 + 6 * x + 1) :
  (x, y) = (1, 4) ∧ ∃ ε > 0, ∀ z : ℝ, abs (z - x) < ε → y ≥ -3 * z^2 + 6 * z + 1 :=
by
  sorry

end vertex_of_parabola_is_max_and_correct_l85_85600


namespace triangle_area_l85_85827

def right_triangle_area (hypotenuse leg1 : ℕ) : ℕ :=
  if (hypotenuse ^ 2 - leg1 ^ 2) > 0 then (1 / 2) * leg1 * (hypotenuse ^ 2 - leg1 ^ 2).sqrt else 0

theorem triangle_area (hypotenuse leg1 : ℕ) (h_hypotenuse : hypotenuse = 13) (h_leg1 : leg1 = 5) :
  right_triangle_area hypotenuse leg1 = 30 :=
by
  rw [h_hypotenuse, h_leg1]
  sorry

end triangle_area_l85_85827


namespace days_to_complete_work_l85_85668

-- Let's define the conditions as Lean definitions based on the problem.

variables (P D : ℕ)
noncomputable def original_work := P * D
noncomputable def half_work_by_double_people := 2 * P * 3

-- Here is our theorem statement
theorem days_to_complete_work : original_work P D = 2 * half_work_by_double_people P :=
by sorry

end days_to_complete_work_l85_85668


namespace first_divisibility_second_divisibility_l85_85049

variable {n : ℕ}
variable (h : n > 0)

theorem first_divisibility :
  17 ∣ (5 * 3^(4*n+1) + 2^(6*n+1)) :=
sorry

theorem second_divisibility :
  32 ∣ (25 * 7^(2*n+1) + 3^(4*n)) :=
sorry

end first_divisibility_second_divisibility_l85_85049


namespace tteokbokki_cost_l85_85418

theorem tteokbokki_cost (P : ℝ) (h1 : P / 2 - P * (3 / 16) = 2500) : P / 2 = 4000 :=
by
  sorry

end tteokbokki_cost_l85_85418


namespace sufficient_but_not_necessary_l85_85859

noncomputable def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

def z (a : ℝ) : ℂ := ⟨a^2 - 4, a + 1⟩

theorem sufficient_but_not_necessary (a : ℝ) (h : a = -2) : 
  is_purely_imaginary (z a) ∧ ¬(∀ a, is_purely_imaginary (z a) → a = -2) :=
by
  sorry

end sufficient_but_not_necessary_l85_85859


namespace find_b_c_l85_85253

theorem find_b_c (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 40) 
  (h2 : a + d = 6)
  (h3 : a * b + b * c + c * d + d * a = 28) : 
  b + c = 17 / 3 := 
by
  sorry

end find_b_c_l85_85253


namespace pet_store_total_birds_l85_85642

def total_birds_in_pet_store (bird_cages parrots_per_cage parakeets_per_cage : ℕ) : ℕ :=
  bird_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_total_birds :
  total_birds_in_pet_store 4 8 2 = 40 :=
by
  sorry

end pet_store_total_birds_l85_85642


namespace radius_of_isosceles_tangent_circle_l85_85041

noncomputable def R : ℝ := 2 * Real.sqrt 3

variables (x : ℝ) (AB AC BD AD DC r : ℝ)

def is_isosceles (AB BC : ℝ) : Prop := AB = BC
def is_tangent (r : ℝ) (x : ℝ) : Prop := r = 2.4 * x

theorem radius_of_isosceles_tangent_circle
  (h_isosceles: is_isosceles AB BC)
  (h_area: 1/2 * AC * BD = 25)
  (h_height_ratio: BD / AC = 3 / 8)
  (h_AD_DC: AD = DC)
  (h_AC: AC = 8 * x)
  (h_BD: BD = 3 * x)
  (h_radius: is_tangent r x):
  r = R :=
sorry

end radius_of_isosceles_tangent_circle_l85_85041


namespace sarah_total_distance_walked_l85_85307

noncomputable def total_distance : ℝ :=
  let rest_time : ℝ := 1 / 3
  let total_time : ℝ := 3.5
  let time_spent_walking : ℝ := total_time - rest_time -- time spent walking
  let uphill_speed : ℝ := 3 -- in mph
  let downhill_speed : ℝ := 4 -- in mph
  let d := time_spent_walking * (uphill_speed * downhill_speed) / (uphill_speed + downhill_speed) -- half distance D
  2 * d

theorem sarah_total_distance_walked :
  total_distance = 10.858 := sorry

end sarah_total_distance_walked_l85_85307


namespace circle_radius_increase_l85_85744

theorem circle_radius_increase (r r' : ℝ) (h : π * r'^2 = (25.44 / 100 + 1) * π * r^2) : 
  (r' - r) / r * 100 = 12 :=
by sorry

end circle_radius_increase_l85_85744


namespace dad_steps_l85_85652

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l85_85652


namespace triangle_side_ineq_l85_85832

theorem triangle_side_ineq (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a^2 * c + b^2 * a + c^2 * b < 1 / 8 := 
by 
  sorry

end triangle_side_ineq_l85_85832


namespace boxcar_capacity_ratio_l85_85470

-- The known conditions translated into Lean definitions
def red_boxcar_capacity (B : ℕ) : ℕ := 3 * B
def blue_boxcar_count : ℕ := 4
def red_boxcar_count : ℕ := 3
def black_boxcar_count : ℕ := 7
def black_boxcar_capacity : ℕ := 4000
def total_capacity : ℕ := 132000

-- The mathematical condition as a Lean theorem statement.
theorem boxcar_capacity_ratio 
  (B : ℕ)
  (h_condition : (red_boxcar_count * red_boxcar_capacity B + 
                  blue_boxcar_count * B + 
                  black_boxcar_count * black_boxcar_capacity = 
                  total_capacity)) : 
  black_boxcar_capacity / B = 1 / 2 := 
sorry

end boxcar_capacity_ratio_l85_85470


namespace infinite_B_l85_85112

open Set Function

variable (A B : Type) 

theorem infinite_B (hA_inf : Infinite A) (f : A → B) : Infinite B :=
by
  sorry

end infinite_B_l85_85112


namespace range_of_a_l85_85108

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 ≤ a ∧ a < 3) :=
by
  sorry

end range_of_a_l85_85108


namespace rectangle_area_l85_85364

theorem rectangle_area (r : ℝ) (L W : ℝ) (h₀ : r = 7) (h₁ : 2 * r = W) (h₂ : L / W = 3) : 
  L * W = 588 :=
by sorry

end rectangle_area_l85_85364


namespace quadratic_roots_expression_l85_85944

theorem quadratic_roots_expression :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 1 = 0) ∧ (x2^2 - 2 * x2 - 1 = 0) →
  (x1 + x2 - x1 * x2 = 3) :=
by
  intros x1 x2 h
  sorry

end quadratic_roots_expression_l85_85944


namespace cake_divided_into_equal_parts_l85_85495

theorem cake_divided_into_equal_parts (cake_weight : ℕ) (pierre : ℕ) (nathalie : ℕ) (parts : ℕ) 
  (hw_eq : cake_weight = 400)
  (hp_eq : pierre = 100)
  (pn_eq : pierre = 2 * nathalie)
  (parts_eq : cake_weight / nathalie = parts)
  (hparts_eq : parts = 8) :
  cake_weight / nathalie = 8 := 
by
  sorry

end cake_divided_into_equal_parts_l85_85495


namespace tangent_circle_locus_l85_85723

-- Definitions for circle C1 and circle C2
def Circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def Circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Definition of being tangent to a circle
def ExternallyTangent (cx cy cr : ℝ) : Prop := (cx - 0)^2 + (cy - 0)^2 = (cr + 1)^2
def InternallyTangent (cx cy cr : ℝ) : Prop := (cx - 3)^2 + (cy - 0)^2 = (3 - cr)^2

-- Definition of locus L where (a,b) are centers of circles tangent to both C1 and C2
def Locus (a b : ℝ) : Prop := 28 * a^2 + 64 * b^2 - 84 * a - 49 = 0

-- The theorem to be proved
theorem tangent_circle_locus (a b r : ℝ) :
  (ExternallyTangent a b r) → (InternallyTangent a b r) → Locus a b :=
by {
  sorry
}

end tangent_circle_locus_l85_85723


namespace julias_total_spending_l85_85863

def adoption_fee : ℝ := 20.00
def dog_food_cost : ℝ := 20.00
def treat_cost_per_bag : ℝ := 2.50
def num_treat_bags : ℝ := 2
def toy_box_cost : ℝ := 15.00
def crate_cost : ℝ := 20.00
def bed_cost : ℝ := 20.00
def collar_leash_cost : ℝ := 15.00
def discount_rate : ℝ := 0.20

def total_items_cost : ℝ :=
  dog_food_cost + (treat_cost_per_bag * num_treat_bags) + toy_box_cost +
  crate_cost + bed_cost + collar_leash_cost

def discount_amount : ℝ := total_items_cost * discount_rate
def discounted_items_cost : ℝ := total_items_cost - discount_amount
def total_expenditure : ℝ := adoption_fee + discounted_items_cost

theorem julias_total_spending :
  total_expenditure = 96.00 := by
  sorry

end julias_total_spending_l85_85863


namespace solve_abs_inequality_l85_85986

theorem solve_abs_inequality (x : ℝ) : x + |2 * x + 3| ≥ 2 ↔ (x ≤ -5 ∨ x ≥ -1/3) :=
by {
  sorry
}

end solve_abs_inequality_l85_85986


namespace missed_questions_l85_85925

theorem missed_questions (F U : ℕ) (h1 : U = 5 * F) (h2 : F + U = 216) : U = 180 :=
by
  sorry

end missed_questions_l85_85925


namespace four_fives_to_hundred_case1_four_fives_to_hundred_case2_l85_85078

theorem four_fives_to_hundred_case1 : (5 + 5) * (5 + 5) = 100 :=
by sorry

theorem four_fives_to_hundred_case2 : (5 * 5 - 5) * 5 = 100 :=
by sorry

end four_fives_to_hundred_case1_four_fives_to_hundred_case2_l85_85078


namespace large_rectangle_perimeter_l85_85471

-- Definitions from the conditions
def side_length_of_square (perimeter_square : ℕ) : ℕ := perimeter_square / 4
def width_of_small_rectangle (perimeter_rect : ℕ) (side_length : ℕ) : ℕ := (perimeter_rect / 2) - side_length

-- Given conditions
def perimeter_square := 24
def perimeter_rect := 16
def side_length := side_length_of_square perimeter_square
def rect_width := width_of_small_rectangle perimeter_rect side_length
def large_rectangle_height := side_length + rect_width
def large_rectangle_width := 3 * side_length

-- Perimeter calculation
def perimeter_large_rectangle (width height : ℕ) : ℕ := 2 * (width + height)

-- Proof problem statement
theorem large_rectangle_perimeter : 
  perimeter_large_rectangle large_rectangle_width large_rectangle_height = 52 :=
sorry

end large_rectangle_perimeter_l85_85471


namespace new_person_weight_l85_85750

-- The conditions from part (a)
variables (average_increase: ℝ) (num_people: ℕ) (weight_lost_person: ℝ)
variables (total_increase: ℝ) (new_weight: ℝ)

-- Assigning the given conditions
axiom h1 : average_increase = 2.5
axiom h2 : num_people = 8
axiom h3 : weight_lost_person = 45
axiom h4 : total_increase = num_people * average_increase
axiom h5 : new_weight = weight_lost_person + total_increase

-- The proof goal: proving that the new person's weight is 65 kg
theorem new_person_weight : new_weight = 65 :=
by
  -- Proof steps go here
  sorry

end new_person_weight_l85_85750


namespace not_divisible_by_81_l85_85505

theorem not_divisible_by_81 (n : ℤ) : ¬ (81 ∣ n^3 - 9 * n + 27) :=
sorry

end not_divisible_by_81_l85_85505


namespace train_pass_jogger_in_41_seconds_l85_85494

-- Definitions based on conditions
def jogger_speed_kmh := 9 -- in km/hr
def train_speed_kmh := 45 -- in km/hr
def initial_distance_jogger := 200 -- in meters
def train_length := 210 -- in meters

-- Converting speeds from km/hr to m/s
def kmh_to_ms (kmh: ℕ) : ℕ := (kmh * 1000) / 3600

def jogger_speed_ms := kmh_to_ms jogger_speed_kmh -- in m/s
def train_speed_ms := kmh_to_ms train_speed_kmh -- in m/s

-- Relative speed of the train with respect to the jogger
def relative_speed := train_speed_ms - jogger_speed_ms -- in m/s

-- Total distance to be covered by the train to pass the jogger
def total_distance := initial_distance_jogger + train_length -- in meters

-- Time taken to pass the jogger
def time_to_pass (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem train_pass_jogger_in_41_seconds : time_to_pass total_distance relative_speed = 41 :=
by
  sorry

end train_pass_jogger_in_41_seconds_l85_85494


namespace molecular_weight_correct_l85_85589

-- Define atomic weights
def atomic_weight_aluminium : Float := 26.98
def atomic_weight_oxygen : Float := 16.00
def atomic_weight_hydrogen : Float := 1.01
def atomic_weight_silicon : Float := 28.09
def atomic_weight_nitrogen : Float := 14.01

-- Define the number of each atom in the compound
def num_aluminium : Nat := 2
def num_oxygen : Nat := 6
def num_hydrogen : Nat := 3
def num_silicon : Nat := 2
def num_nitrogen : Nat := 4

-- Calculate the expected molecular weight
def expected_molecular_weight : Float :=
  (2 * atomic_weight_aluminium) + 
  (6 * atomic_weight_oxygen) + 
  (3 * atomic_weight_hydrogen) + 
  (2 * atomic_weight_silicon) + 
  (4 * atomic_weight_nitrogen)

-- Prove that the expected molecular weight is 265.21 amu
theorem molecular_weight_correct : expected_molecular_weight = 265.21 :=
by
  sorry

end molecular_weight_correct_l85_85589


namespace elderly_in_sample_l85_85577

variable (A E M : ℕ)
variable (total_employees : ℕ)
variable (total_young : ℕ)
variable (sample_size_young : ℕ)
variable (sampling_ratio : ℚ)
variable (sample_elderly : ℕ)

axiom condition_1 : total_young = 160
axiom condition_2 : total_employees = 430
axiom condition_3 : M = 2 * E
axiom condition_4 : A + M + E = total_employees
axiom condition_5 : sampling_ratio = sample_size_young / total_young
axiom sampling : sample_size_young = 32
axiom elderly_employees : sample_elderly = 18

theorem elderly_in_sample : sample_elderly = sampling_ratio * E := by
  -- Proof steps are not provided
  sorry

end elderly_in_sample_l85_85577


namespace one_third_sugar_amount_l85_85595

-- Define the original amount of sugar as a mixed number
def original_sugar_mixed : ℚ := 6 + 1 / 3

-- Define the fraction representing one-third of the recipe
def one_third : ℚ := 1 / 3

-- Define the expected amount of sugar for one-third of the recipe
def expected_sugar_mixed : ℚ := 2 + 1 / 9

-- The theorem stating the proof problem
theorem one_third_sugar_amount : (one_third * original_sugar_mixed) = expected_sugar_mixed :=
sorry

end one_third_sugar_amount_l85_85595


namespace sheela_deposit_amount_l85_85147

theorem sheela_deposit_amount (monthly_income : ℕ) (deposit_percentage : ℕ) :
  monthly_income = 25000 → deposit_percentage = 20 → (deposit_percentage / 100 * monthly_income) = 5000 :=
  by
    intros h_income h_percentage
    rw [h_income, h_percentage]
    sorry

end sheela_deposit_amount_l85_85147


namespace part1_part2_l85_85869

section Problem

open Real

noncomputable def f (x : ℝ) := exp x
noncomputable def g (x : ℝ) := log x - 2

theorem part1 (x : ℝ) (hx : x > 0) : g x ≥ - (exp 1) / x :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x, x ≥ 0 → f x - 1 / (f x) ≥ a * x) : a ≤ 2 :=
by sorry

end Problem

end part1_part2_l85_85869


namespace polynomial_identity_sum_l85_85672

theorem polynomial_identity_sum (A B C D : ℤ) (h : (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 := 
by 
  sorry

end polynomial_identity_sum_l85_85672


namespace isosceles_triangle_base_length_l85_85276

noncomputable def equilateral_side_length (p_eq : ℕ) : ℕ := p_eq / 3

theorem isosceles_triangle_base_length (p_eq p_iso s b : ℕ) 
  (h1 : p_eq = 45)
  (h2 : p_iso = 40)
  (h3 : s = equilateral_side_length p_eq)
  (h4 : p_iso = s + s + b)
  : b = 10 :=
by
  simp [h1, h2, h3] at h4
  -- steps to solve for b would be written here
  sorry

end isosceles_triangle_base_length_l85_85276


namespace expression_equality_l85_85800

theorem expression_equality :
  - (2^3) = (-2)^3 :=
by sorry

end expression_equality_l85_85800


namespace construct_rhombus_l85_85518

-- Define data structure representing a point in a 2-dimensional Euclidean space.
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for four points to form a rhombus.
def isRhombus (A B C D : Point) : Prop :=
  (A.x - B.x) ^ 2 + (A.y - B.y) ^ 2 = (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 ∧
  (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 = (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 ∧
  (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 = (D.x - A.x) ^ 2 + (D.y - A.y) ^ 2

-- Define circumradius condition for triangle ABC
def circumradius (A B C : Point) (R : ℝ) : Prop := sorry -- Detailed definition would be added here.

-- Define inradius condition for triangle BCD
def inradius (B C D : Point) (r : ℝ) : Prop := sorry -- Detailed definition would be added here.

-- The proposition to be proved: We can construct the rhombus ABCD given R and r.
theorem construct_rhombus (A B C D : Point) (R r : ℝ) :
  (circumradius A B C R) →
  (inradius B C D r) →
  isRhombus A B C D :=
by
  sorry

end construct_rhombus_l85_85518


namespace balloon_totals_l85_85866

-- Definitions
def Joan_blue := 40
def Joan_red := 30
def Joan_green := 0
def Joan_yellow := 0

def Melanie_blue := 41
def Melanie_red := 0
def Melanie_green := 20
def Melanie_yellow := 0

def Eric_blue := 0
def Eric_red := 25
def Eric_green := 0
def Eric_yellow := 15

-- Total counts
def total_blue := Joan_blue + Melanie_blue + Eric_blue
def total_red := Joan_red + Melanie_red + Eric_red
def total_green := Joan_green + Melanie_green + Eric_green
def total_yellow := Joan_yellow + Melanie_yellow + Eric_yellow

-- Statement of the problem
theorem balloon_totals :
  total_blue = 81 ∧
  total_red = 55 ∧
  total_green = 20 ∧
  total_yellow = 15 :=
by
  -- Proof omitted
  sorry

end balloon_totals_l85_85866


namespace good_numbers_l85_85029

def is_good (n : ℕ) : Prop :=
  ∀ d, d ∣ n → (d + 1) ∣ (n + 1)

theorem good_numbers (n : ℕ) :
  is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ Odd n) :=
by
  sorry

end good_numbers_l85_85029


namespace mice_population_l85_85506

theorem mice_population :
  ∃ (mice_initial : ℕ) (pups_per_mouse : ℕ) (survival_rate_first_gen : ℕ → ℕ) 
    (survival_rate_second_gen : ℕ → ℕ) (num_dead_first_gen : ℕ) (pups_eaten_per_adult : ℕ)
    (total_mice : ℕ),
    mice_initial = 8 ∧ pups_per_mouse = 7 ∧
    (∀ n, survival_rate_first_gen n = (n * 80) / 100) ∧
    (∀ n, survival_rate_second_gen n = (n * 60) / 100) ∧
    num_dead_first_gen = 2 ∧ pups_eaten_per_adult = 3 ∧
    total_mice = mice_initial + (survival_rate_first_gen (mice_initial * pups_per_mouse)) - num_dead_first_gen + (survival_rate_second_gen ((mice_initial + (survival_rate_first_gen (mice_initial * pups_per_mouse))) * pups_per_mouse)) - ((mice_initial - num_dead_first_gen) * pups_eaten_per_adult) :=
  sorry

end mice_population_l85_85506


namespace triangle_perimeter_l85_85004

theorem triangle_perimeter (x : ℕ) (hx1 : x % 2 = 1) (hx2 : 5 < x) (hx3 : x < 11) : 
  (3 + 8 + x = 18) ∨ (3 + 8 + x = 20) :=
sorry

end triangle_perimeter_l85_85004


namespace three_a_ge_two_b_plus_two_l85_85982

theorem three_a_ge_two_b_plus_two (a b : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : (a! * b!) % (a! + b!) = 0) :
  3 * a ≥ 2 * b + 2 :=
sorry

end three_a_ge_two_b_plus_two_l85_85982


namespace ukuleles_and_violins_l85_85179

theorem ukuleles_and_violins (U V : ℕ) : 
  (4 * U + 6 * 4 + 4 * V = 40) → (U + V = 4) :=
by
  intro h
  sorry

end ukuleles_and_violins_l85_85179


namespace largest_is_B_l85_85478

noncomputable def A := Real.sqrt (Real.sqrt (56 ^ (1 / 3)))
noncomputable def B := Real.sqrt (Real.sqrt (3584 ^ (1 / 3)))
noncomputable def C := Real.sqrt (Real.sqrt (2744 ^ (1 / 3)))
noncomputable def D := Real.sqrt (Real.sqrt (392 ^ (1 / 3)))
noncomputable def E := Real.sqrt (Real.sqrt (448 ^ (1 / 3)))

theorem largest_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end largest_is_B_l85_85478


namespace abs_ineq_sol_set_l85_85071

theorem abs_ineq_sol_set (x : ℝ) : (|x - 2| + |x - 1| ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) :=
by
  sorry

end abs_ineq_sol_set_l85_85071


namespace remainder_when_divided_by_x_minus_1_is_minus_2_l85_85461

def p (x : ℝ) : ℝ := x^4 - 2*x^2 + 4*x - 5

theorem remainder_when_divided_by_x_minus_1_is_minus_2 : (p 1) = -2 := 
by 
  -- Proof not required
  sorry

end remainder_when_divided_by_x_minus_1_is_minus_2_l85_85461


namespace rachel_hw_diff_l85_85920

-- Definitions based on the conditions of the problem
def math_hw_pages := 15
def reading_hw_pages := 6

-- The statement we need to prove, including the conditions
theorem rachel_hw_diff : 
  math_hw_pages - reading_hw_pages = 9 := 
by
  sorry

end rachel_hw_diff_l85_85920


namespace sequence_arith_l85_85257

theorem sequence_arith {a : ℕ → ℕ} (h_initial : a 2 = 2) (h_recursive : ∀ n ≥ 2, a (n + 1) = a n + 1) :
  ∀ n ≥ 2, a n = n :=
by
  sorry

end sequence_arith_l85_85257


namespace jonathan_needs_more_money_l85_85395

def cost_dictionary : ℕ := 11
def cost_dinosaur_book : ℕ := 19
def cost_childrens_cookbook : ℕ := 7
def saved_money : ℕ := 8

def total_cost : ℕ := cost_dictionary + cost_dinosaur_book + cost_childrens_cookbook
def amount_needed : ℕ := total_cost - saved_money

theorem jonathan_needs_more_money : amount_needed = 29 := by
  have h1 : total_cost = 37 := by
    show 11 + 19 + 7 = 37
    sorry
  show 37 - 8 = 29
  sorry

end jonathan_needs_more_money_l85_85395


namespace apples_in_basket_l85_85608

theorem apples_in_basket
  (total_rotten : ℝ := 12 / 100)
  (total_spots : ℝ := 7 / 100)
  (total_insects : ℝ := 5 / 100)
  (total_varying_rot : ℝ := 3 / 100)
  (perfect_apples : ℝ := 66) :
  (perfect_apples / ((1 - (total_rotten + total_spots + total_insects + total_varying_rot))) = 90) :=
by
  sorry

end apples_in_basket_l85_85608


namespace average_paper_tape_length_l85_85248

-- Define the lengths of the paper tapes as given in the conditions
def red_tape_length : ℝ := 20
def purple_tape_length : ℝ := 16

-- State the proof problem
theorem average_paper_tape_length : 
  (red_tape_length + purple_tape_length) / 2 = 18 := 
by
  sorry

end average_paper_tape_length_l85_85248


namespace smallest_positive_whole_number_divisible_by_first_five_primes_l85_85026

def is_prime (n : Nat) : Prop := Nat.Prime n

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

def smallest_positive_divisible (lst : List Nat) : Nat :=
  List.foldl (· * ·) 1 lst

theorem smallest_positive_whole_number_divisible_by_first_five_primes :
  smallest_positive_divisible first_five_primes = 2310 := by
  sorry

end smallest_positive_whole_number_divisible_by_first_five_primes_l85_85026


namespace find_set_A_l85_85835

def M : Set ℤ := {1, 3, 5, 7, 9}

def satisfiesCondition (A : Set ℤ) : Prop :=
  A ≠ ∅ ∧
  (∀ a ∈ A, a + 4 ∈ M) ∧
  (∀ a ∈ A, a - 4 ∈ M)

theorem find_set_A : ∃ A : Set ℤ, satisfiesCondition A ∧ A = {5} :=
  by
    sorry

end find_set_A_l85_85835


namespace decimal_to_base5_equiv_l85_85539

def base5_representation (n : ℕ) : ℕ := -- Conversion function (implementation to be filled later)
  sorry

theorem decimal_to_base5_equiv : base5_representation 88 = 323 :=
by
  -- Proof steps go here.
  sorry

end decimal_to_base5_equiv_l85_85539


namespace prime_sum_remainder_l85_85207

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l85_85207


namespace distribute_pencils_l85_85410

theorem distribute_pencils (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  (∃ ways : ℕ, ways = Nat.choose (n - 1) (k - 1) ∧ ways = 35) :=
by
  sorry

end distribute_pencils_l85_85410


namespace Ryan_dig_time_alone_l85_85335

theorem Ryan_dig_time_alone :
  ∃ R : ℝ, ∀ Castel_time together_time,
    Castel_time = 6 ∧ together_time = 30 / 11 →
    (1 / R + 1 / Castel_time = 11 / 30) →
    R = 5 :=
by 
  sorry

end Ryan_dig_time_alone_l85_85335


namespace yellow_tint_percentage_new_mixture_l85_85393

def original_volume : ℝ := 40
def yellow_tint_percentage : ℝ := 0.35
def additional_yellow_tint : ℝ := 10
def new_volume : ℝ := original_volume + additional_yellow_tint
def original_yellow_tint : ℝ := yellow_tint_percentage * original_volume
def new_yellow_tint : ℝ := original_yellow_tint + additional_yellow_tint

theorem yellow_tint_percentage_new_mixture : 
  (new_yellow_tint / new_volume) * 100 = 48 := 
by
  sorry

end yellow_tint_percentage_new_mixture_l85_85393


namespace expand_product_l85_85252

theorem expand_product (x : ℝ) : 
  5 * (x + 6) * (x^2 + 2 * x + 3) = 5 * x^3 + 40 * x^2 + 75 * x + 90 := 
by 
  sorry

end expand_product_l85_85252


namespace original_number_l85_85601

theorem original_number (n : ℕ) (h1 : 100000 ≤ n ∧ n < 1000000) (h2 : n / 100000 = 7) (h3 : (n % 100000) * 10 + 7 = n / 5) : n = 714285 :=
sorry

end original_number_l85_85601


namespace hardware_contract_probability_l85_85516

noncomputable def P_S' : ℚ := 3 / 5
noncomputable def P_at_least_one : ℚ := 5 / 6
noncomputable def P_H_and_S : ℚ := 0.31666666666666654 -- 19 / 60 in fraction form
noncomputable def P_S : ℚ := 1 - P_S'

theorem hardware_contract_probability :
  (P_at_least_one = P_H + P_S - P_H_and_S) →
  P_H = 0.75 :=
by
  sorry

end hardware_contract_probability_l85_85516


namespace favorite_number_l85_85678

theorem favorite_number (S₁ S₂ S₃ : ℕ) (total_sum : ℕ) (adjacent_sum : ℕ) 
  (h₁ : S₁ = 8) (h₂ : S₂ = 14) (h₃ : S₃ = 12) 
  (h_total_sum : total_sum = 17) 
  (h_adjacent_sum : adjacent_sum = 12) : 
  ∃ x : ℕ, x = 5 := 
by 
  sorry

end favorite_number_l85_85678


namespace fraction_decomposition_l85_85058

theorem fraction_decomposition (P Q : ℚ) :
  (∀ x : ℚ, 4 * x ^ 3 - 5 * x ^ 2 - 26 * x + 24 = (2 * x ^ 2 - 5 * x + 3) * (2 * x - 3))
  → P / (2 * x ^ 2 - 5 * x + 3) + Q / (2 * x - 3) = (8 * x ^ 2 - 9 * x + 20) / (4 * x ^ 3 - 5 * x ^ 2 - 26 * x + 24)
  → P = 4 / 9 ∧ Q = 68 / 9 := by 
  sorry

end fraction_decomposition_l85_85058


namespace solve_xyz_l85_85899

theorem solve_xyz (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y + x) : x + y + z = 14 * x :=
by
  sorry

end solve_xyz_l85_85899


namespace Carla_more_miles_than_Daniel_after_5_hours_l85_85814

theorem Carla_more_miles_than_Daniel_after_5_hours (Carla_distance : ℝ) (Daniel_distance : ℝ) (h_Carla : Carla_distance = 100) (h_Daniel : Daniel_distance = 75) : 
  Carla_distance - Daniel_distance = 25 := 
by
  sorry

end Carla_more_miles_than_Daniel_after_5_hours_l85_85814


namespace select_pencils_l85_85736

theorem select_pencils (boxes : Fin 10 → ℕ) (colors : ∀ (i : Fin 10), Fin (boxes i) → Fin 10) :
  (∀ i : Fin 10, 1 ≤ boxes i) → -- Each box is non-empty
  (∀ i j : Fin 10, i ≠ j → boxes i ≠ boxes j) → -- Different number of pencils in each box
  ∃ (selection : Fin 10 → Fin 10), -- Function to select a pencil color from each box
  Function.Injective selection := -- All selected pencils have different colors
sorry

end select_pencils_l85_85736


namespace problem_1_problem_2_l85_85270

def is_in_solution_set (x : ℝ) : Prop := -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0

variables {a b : ℝ}

theorem problem_1 (ha : is_in_solution_set a) (hb : is_in_solution_set b) :
  |(1 / 3) * a + (1 / 6) * b| < 1 / 4 :=
sorry

theorem problem_2 (ha : is_in_solution_set a) (hb : is_in_solution_set b) :
  |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end problem_1_problem_2_l85_85270


namespace find_S_l85_85140

variable {R k : ℝ}

theorem find_S (h : |k + R| / |R| = 0) : S = 1 :=
by
  let S := |k + 2*R| / |2*k + R|
  have h1 : k + R = 0 := by sorry
  have h2 : k = -R := by sorry
  sorry

end find_S_l85_85140


namespace sin_inequality_l85_85530

theorem sin_inequality (x : ℝ) (hx1 : 0 < x) (hx2 : x < Real.pi / 4) : 
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) :=
by 
  sorry

end sin_inequality_l85_85530


namespace sale_in_fifth_month_l85_85821

theorem sale_in_fifth_month 
  (sale_month_1 : ℕ) (sale_month_2 : ℕ) (sale_month_3 : ℕ) (sale_month_4 : ℕ) 
  (sale_month_6 : ℕ) (average_sale : ℕ) 
  (h1 : sale_month_1 = 5266) (h2 : sale_month_2 = 5744) (h3 : sale_month_3 = 5864) 
  (h4 : sale_month_4 = 6122) (h6 : sale_month_6 = 4916) (h_avg : average_sale = 5750) :
  ∃ sale_month_5, sale_month_5 = 6588 :=
by
  sorry

end sale_in_fifth_month_l85_85821


namespace black_white_difference_l85_85640

theorem black_white_difference (m n : ℕ) (h_dim : m = 7 ∧ n = 9) (h_first_black : m % 2 = 1 ∧ n % 2 = 1) :
  let black_count := (5 * 4 + 4 * 3)
  let white_count := (4 * 4 + 5 * 3)
  black_count - white_count = 1 := 
by
  -- We start with known dimensions and conditions
  let ⟨hm, hn⟩ := h_dim
  have : m = 7 := by rw [hm]
  have : n = 9 := by rw [hn]
  
  -- Calculate the number of black and white squares 
  let black_count := (5 * 4 + 4 * 3)
  let white_count := (4 * 4 + 5 * 3)
  
  -- Use given formulas to calculate the difference
  have diff : black_count - white_count = 1 := by
    sorry -- proof to be provided
  
  exact diff

end black_white_difference_l85_85640


namespace close_time_for_pipe_b_l85_85191

-- Define entities and rates
def rate_fill (A_rate B_rate : ℝ) (t_fill t_empty t_fill_target t_close : ℝ) : Prop :=
  A_rate = 1 / t_fill ∧
  B_rate = 1 / t_empty ∧
  t_fill_target = 30 ∧
  A_rate * (t_close + (t_fill_target - t_close)) - B_rate * t_close = 1

-- Declare the theorem statement
theorem close_time_for_pipe_b (A_rate B_rate t_fill_target t_fill t_empty t_close: ℝ) :
   rate_fill A_rate B_rate t_fill t_empty t_fill_target t_close → t_close = 26.25 :=
by have h1 : A_rate = 1 / 15 := by sorry
   have h2 : B_rate = 1 / 24 := by sorry
   have h3 : t_fill_target = 30 := by sorry
   sorry

end close_time_for_pipe_b_l85_85191


namespace find_specific_n_l85_85950

theorem find_specific_n :
  ∀ (n : ℕ), (∃ (a b : ℤ), n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n = 0 ∨ n = 1 ∨ n = 2 :=
by {
  sorry
}

end find_specific_n_l85_85950


namespace sum_of_tens_l85_85975

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l85_85975


namespace treasures_found_second_level_l85_85775

theorem treasures_found_second_level:
  ∀ (P T1 S T2 : ℕ), 
    P = 4 → 
    T1 = 6 → 
    S = 32 → 
    S = P * T1 + P * T2 → 
    T2 = 2 := 
by
  intros P T1 S T2 hP hT1 hS hTotal
  sorry

end treasures_found_second_level_l85_85775


namespace erasers_per_friend_l85_85783

variable (erasers friends : ℕ)

theorem erasers_per_friend (h1 : erasers = 3840) (h2 : friends = 48) :
  erasers / friends = 80 :=
by sorry

end erasers_per_friend_l85_85783


namespace set_intersection_complement_l85_85980

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem set_intersection_complement :
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end set_intersection_complement_l85_85980


namespace number_of_games_is_15_l85_85053

-- Definition of the given conditions
def total_points : ℕ := 345
def avg_points_per_game : ℕ := 4 + 10 + 9
def number_of_games (total_points : ℕ) (avg_points_per_game : ℕ) := total_points / avg_points_per_game

-- The theorem stating the proof problem
theorem number_of_games_is_15 : number_of_games total_points avg_points_per_game = 15 :=
by
  -- Skipping the proof as only the statement is required
  sorry

end number_of_games_is_15_l85_85053


namespace parabola_intersection_l85_85965

theorem parabola_intersection :
  (∀ x y : ℝ, y = 3 * x^2 - 4 * x + 2 ↔ y = 9 * x^2 + 6 * x + 2) →
  (∃ x1 y1 x2 y2 : ℝ,
    (x1 = 0 ∧ y1 = 2) ∧ (x2 = -5 / 3 ∧ y2 = 17)) :=
by
  intro h
  sorry

end parabola_intersection_l85_85965


namespace area_large_square_l85_85964

theorem area_large_square (a b c : ℝ) 
  (h1 : a^2 = b^2 + 32) 
  (h2 : 4*a = 4*c + 16) : a^2 = 100 := 
by {
  sorry
}

end area_large_square_l85_85964


namespace find_x_l85_85224

theorem find_x (x y : ℕ) (h1 : x / y = 6 / 3) (h2 : y = 27) : x = 54 :=
sorry

end find_x_l85_85224


namespace fixed_point_l85_85675

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  2 + a^(1-1) = 3 :=
by
  sorry

end fixed_point_l85_85675


namespace saturated_function_2014_l85_85237

def saturated (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f^[f^[f n] n] n = n

theorem saturated_function_2014 (f : ℕ → ℕ) (m : ℕ) (h : saturated f) :
  (m ∣ 2014) ↔ (f^[2014] m = m) :=
sorry

end saturated_function_2014_l85_85237


namespace math_problem_solution_l85_85326

theorem math_problem_solution : 8 / 4 - 3 - 9 + 3 * 9 = 17 := 
by 
  sorry

end math_problem_solution_l85_85326


namespace rectangle_area_is_correct_l85_85567

-- Define the conditions
def length : ℕ := 135
def breadth (l : ℕ) : ℕ := l / 3

-- Define the area of the rectangle
def area (l b : ℕ) : ℕ := l * b

-- The statement to prove
theorem rectangle_area_is_correct : area length (breadth length) = 6075 := by
  -- Proof goes here, this is just the statement
  sorry

end rectangle_area_is_correct_l85_85567


namespace o_l85_85673

theorem o'hara_triple_example (a b x : ℕ) (h₁ : a = 49) (h₂ : b = 16) (h₃ : x = (Int.sqrt a).toNat + (Int.sqrt b).toNat) : x = 11 := 
by
  sorry

end o_l85_85673


namespace lesser_fraction_l85_85828

theorem lesser_fraction 
  (x y : ℚ)
  (h_sum : x + y = 13 / 14)
  (h_prod : x * y = 1 / 5) :
  min x y = 87 / 700 := sorry

end lesser_fraction_l85_85828


namespace todd_initial_gum_l85_85966

-- Define the conditions and the final result
def initial_gum (final_gum: Nat) (given_gum: Nat) : Nat := final_gum - given_gum

theorem todd_initial_gum :
  initial_gum 54 16 = 38 :=
by
  -- Use the initial_gum definition to state the problem
  -- The proof is skipped with sorry
  sorry

end todd_initial_gum_l85_85966


namespace max_pairs_of_corner_and_squares_l85_85903

def rectangle : ℕ := 3 * 100
def unit_squares_per_pair : ℕ := 4 + 3

-- Given conditions
def conditions := rectangle = 300 ∧ unit_squares_per_pair = 7

-- Proof statement
theorem max_pairs_of_corner_and_squares (h: conditions) : ∃ n, n = 33 ∧ n * unit_squares_per_pair ≤ rectangle := 
sorry

end max_pairs_of_corner_and_squares_l85_85903


namespace correct_transformation_l85_85131

theorem correct_transformation (a b m : ℝ) (h : m ≠ 0) : (am / bm) = (a / b) :=
by sorry

end correct_transformation_l85_85131


namespace no_possible_seating_arrangement_l85_85152

theorem no_possible_seating_arrangement : 
  ¬(∃ (students : Fin 11 → Fin 4),
    ∀ (i : Fin 11),
    ∃ (s1 s2 s3 s4 s5 : Fin 11),
      s1 = i ∧ 
      (s2 = (i + 1) % 11) ∧ 
      (s3 = (i + 2) % 11) ∧ 
      (s4 = (i + 3) % 11) ∧ 
      (s5 = (i + 4) % 11) ∧
      ∃ (g1 g2 g3 g4 : Fin 4),
        (students s1 = g1) ∧ 
        (students s2 = g2) ∧ 
        (students s3 = g3) ∧ 
        (students s4 = g4) ∧ 
        (students s5).val ≠ (students s1).val ∧ 
        (students s5).val ≠ (students s2).val ∧ 
        (students s5).val ≠ (students s3).val ∧ 
        (students s5).val ≠ (students s4).val) :=
sorry

end no_possible_seating_arrangement_l85_85152


namespace smallest_number_divisible_l85_85588

/-- The smallest number which, when diminished by 20, is divisible by 15, 30, 45, and 60 --/
theorem smallest_number_divisible (n : ℕ) (h : ∀ k : ℕ, n - 20 = k * Int.lcm 15 (Int.lcm 30 (Int.lcm 45 60))) : n = 200 :=
sorry

end smallest_number_divisible_l85_85588


namespace range_of_m_l85_85453

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ (m + 3) / (x - 1) = 1) ↔ m > -4 ∧ m ≠ -3 := 
by 
  sorry

end range_of_m_l85_85453


namespace binomial_coeff_coprime_l85_85731

def binom (a b : ℕ) : ℕ := Nat.factorial a / (Nat.factorial b * Nat.factorial (a - b))

theorem binomial_coeff_coprime (p a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hp : Nat.Prime p) 
  (hbase_p_a : ∀ i, (a / p^i % p) ≥ (b / p^i % p)) 
  : Nat.gcd (binom a b) p = 1 :=
by sorry

end binomial_coeff_coprime_l85_85731


namespace f_sq_add_g_sq_eq_one_f_even_f_periodic_l85_85342

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom g_odd : ∀ x : ℝ, g (-x) = - g x
axiom f_0 : f 0 = 1
axiom f_eq : ∀ x y : ℝ, f (x - y) = f x * f y + g x * g y

theorem f_sq_add_g_sq_eq_one (x : ℝ) : f x ^ 2 + g x ^ 2 = 1 :=
sorry

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
sorry

theorem f_periodic (a : ℝ) (ha : a ≠ 0) (hfa : f a = 1) : ∀ x : ℝ, f (x + a) = f x :=
sorry

end f_sq_add_g_sq_eq_one_f_even_f_periodic_l85_85342


namespace least_number_to_subtract_l85_85122

theorem least_number_to_subtract :
  ∃ k : ℕ, k = 45 ∧ (568219 - k) % 89 = 0 :=
by
  sorry

end least_number_to_subtract_l85_85122


namespace train_A_start_time_l85_85451

theorem train_A_start_time :
  let distance := 155 -- km
  let speed_A := 20 -- km/h
  let speed_B := 25 -- km/h
  let start_B := 8 -- a.m.
  let meet_time := 11 -- a.m.
  let travel_time_B := meet_time - start_B -- time in hours for train B from 8 a.m. to 11 a.m.
  let distance_B := speed_B * travel_time_B -- distance covered by train B
  let distance_A := distance - distance_B -- remaining distance covered by train A
  let travel_time_A := distance_A / speed_A -- time for train A to cover its distance
  let start_A := meet_time - travel_time_A -- start time for train A
  start_A = 7 := by
  sorry

end train_A_start_time_l85_85451


namespace q_joins_after_2_days_l85_85313

-- Define the conditions
def work_rate_p := 1 / 10
def work_rate_q := 1 / 6
def total_days := 5

-- Define the proof problem
theorem q_joins_after_2_days (a b : ℝ) (t x : ℕ) : 
  a = work_rate_p → b = work_rate_q → t = total_days →
  x * a + (t - x) * (a + b) = 1 → 
  x = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end q_joins_after_2_days_l85_85313


namespace gain_percent_40_l85_85854

theorem gain_percent_40 (cost_price selling_price : ℝ) (h1 : cost_price = 900) (h2 : selling_price = 1260) :
  ((selling_price - cost_price) / cost_price) * 100 = 40 :=
by
  sorry

end gain_percent_40_l85_85854


namespace recreation_percentage_l85_85735

def wages_last_week (W : ℝ) : ℝ := W
def spent_on_recreation_last_week (W : ℝ) : ℝ := 0.15 * W
def wages_this_week (W : ℝ) : ℝ := 0.90 * W
def spent_on_recreation_this_week (W : ℝ) : ℝ := 0.30 * (wages_this_week W)

theorem recreation_percentage (W : ℝ) (hW: W > 0) :
  (spent_on_recreation_this_week W) / (spent_on_recreation_last_week W) * 100 = 180 := by
  sorry

end recreation_percentage_l85_85735


namespace question_solution_l85_85407

noncomputable def segment_ratio : (ℝ × ℝ) :=
  let m := 7
  let n := 2
  let x := - (2 / (m - n))
  let y := 7 / (m - n)
  (x, y)

theorem question_solution : segment_ratio = (-2/5, 7/5) :=
  by
  -- prove that the pair (x, y) calculated using given m and n equals (-2/5, 7/5)
  sorry

end question_solution_l85_85407


namespace maximize_integral_l85_85926
open Real

noncomputable def integral_to_maximize (a b : ℝ) : ℝ :=
  ∫ x in a..b, exp (cos x) * (380 - x - x^2)

theorem maximize_integral :
  ∀ (a b : ℝ), a ≤ b → integral_to_maximize a b ≤ integral_to_maximize (-20) 19 :=
by
  intros a b h
  sorry

end maximize_integral_l85_85926


namespace percentage_reduction_in_price_of_oil_l85_85951

theorem percentage_reduction_in_price_of_oil :
  ∀ P : ℝ, ∀ R : ℝ, P = 800 / (800 / R - 5) ∧ R = 40 →
  (P - R) / P * 100 = 25 := by
  -- Assumptions
  intros P R h
  have hP : P = 800 / (800 / R - 5) := h.1
  have hR : R = 40 := h.2
  -- Result to be proved
  sorry

end percentage_reduction_in_price_of_oil_l85_85951


namespace correct_statement_C_l85_85760

theorem correct_statement_C : ∀ y : ℝ, y^2 = 25 ↔ (y = 5 ∨ y = -5) := 
by sorry

end correct_statement_C_l85_85760


namespace readers_in_group_l85_85615

theorem readers_in_group (S L B T : ℕ) (hS : S = 120) (hL : L = 90) (hB : B = 60) :
  T = S + L - B → T = 150 :=
by
  intro h₁
  rw [hS, hL, hB] at h₁
  linarith

end readers_in_group_l85_85615


namespace adam_and_simon_time_to_be_80_miles_apart_l85_85129

theorem adam_and_simon_time_to_be_80_miles_apart :
  ∃ x : ℝ, (10 * x)^2 + (8 * x)^2 = 80^2 ∧ x = 6.25 :=
by
  sorry

end adam_and_simon_time_to_be_80_miles_apart_l85_85129


namespace Laticia_knitted_socks_l85_85481

theorem Laticia_knitted_socks (x : ℕ) (cond1 : x ≥ 0)
  (cond2 : ∃ y, y = x + 4)
  (cond3 : ∃ z, z = (x + (x + 4)) / 2)
  (cond4 : ∃ w, w = z - 3)
  (cond5 : x + (x + 4) + z + w = 57) : x = 13 := by
  sorry

end Laticia_knitted_socks_l85_85481


namespace triangle_area_bounded_by_lines_l85_85625

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let base := 16
  let height := 8
  let area := base * height / 2
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l85_85625


namespace product_p_yi_eq_neg26_l85_85739

-- Definitions of the polynomials h and p.
def h (y : ℂ) : ℂ := y^3 - 3 * y + 1
def p (y : ℂ) : ℂ := y^3 + 2

-- Given that y1, y2, y3 are roots of h(y)
variables (y1 y2 y3 : ℂ) (H1 : h y1 = 0) (H2 : h y2 = 0) (H3 : h y3 = 0)

-- State the theorem to show p(y1) * p(y2) * p(y3) = -26
theorem product_p_yi_eq_neg26 : p y1 * p y2 * p y3 = -26 :=
sorry

end product_p_yi_eq_neg26_l85_85739


namespace heptagon_angle_in_arithmetic_progression_l85_85962

theorem heptagon_angle_in_arithmetic_progression (a d : ℝ) :
  a + 3 * d = 128.57 → 
  (7 * a + 21 * d = 900) → 
  ∃ angle : ℝ, angle = 128.57 :=
by
  sorry

end heptagon_angle_in_arithmetic_progression_l85_85962


namespace sample_size_survey_l85_85347

theorem sample_size_survey (students_selected : ℕ) (h : students_selected = 200) : students_selected = 200 :=
by
  assumption

end sample_size_survey_l85_85347


namespace lighting_candles_correct_l85_85085

noncomputable def time_to_light_candles (initial_length : ℝ) : ℝ :=
  let burn_rate_1 := initial_length / 300
  let burn_rate_2 := initial_length / 240
  let t := (5 * 60 + 43) - (5 * 60) -- 11:17 AM is 342.857 minutes before 5 PM
  if ((initial_length - burn_rate_2 * t) = 3 * (initial_length - burn_rate_1 * t)) then 11 + 17 / 60 else 0 -- Check if the condition is met

theorem lighting_candles_correct :
  ∀ (initial_length : ℝ), time_to_light_candles initial_length = 11 + 17 / 60 :=
by
  intros initial_length
  sorry  -- Proof goes here

end lighting_candles_correct_l85_85085


namespace sons_age_l85_85379

theorem sons_age (S M : ℕ) (h1 : M = 3 * S) (h2 : M + 12 = 2 * (S + 12)) : S = 12 :=
by 
  sorry

end sons_age_l85_85379


namespace barrels_are_1360_l85_85641

-- Defining the top layer dimensions and properties
def a : ℕ := 2
def b : ℕ := 1
def n : ℕ := 15

-- Defining the dimensions of the bottom layer based on given properties
def c : ℕ := a + n
def d : ℕ := b + n

-- Formula for the total number of barrels
def total_barrels : ℕ := n * ((2 * a + c) * b + (2 * c + a) * d + (d - b)) / 6

-- Theorem to prove
theorem barrels_are_1360 : total_barrels = 1360 :=
by
  sorry

end barrels_are_1360_l85_85641


namespace loss_percent_l85_85916

theorem loss_percent (CP SP : ℝ) (h₁ : CP = 600) (h₂ : SP = 300) : 
  (CP - SP) / CP * 100 = 50 :=
by
  rw [h₁, h₂]
  norm_num

end loss_percent_l85_85916


namespace daphne_two_visits_in_365_days_l85_85136

def visits_in_days (d1 d2 : ℕ) (days : ℕ) : ℕ :=
  days / Nat.lcm d1 d2

theorem daphne_two_visits_in_365_days :
  let days := 365
  let lcm_all := Nat.lcm 4 (Nat.lcm 6 (Nat.lcm 8 10))
  (visits_in_days 4 6 lcm_all + 
   visits_in_days 4 8 lcm_all + 
   visits_in_days 4 10 lcm_all + 
   visits_in_days 6 8 lcm_all + 
   visits_in_days 6 10 lcm_all + 
   visits_in_days 8 10 lcm_all) * 
   (days / lcm_all) = 129 :=
by
  sorry

end daphne_two_visits_in_365_days_l85_85136


namespace find_a_b_range_of_a_l85_85130

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * a * x + 2

-- Problem 1
theorem find_a_b (a b : ℝ) :
  f a 1 = 0 ∧ f a b = 0 ∧ (∀ x, f a x > 0 ↔ x < 1 ∨ x > b) → a = 1 ∧ b = 2 := sorry

-- Problem 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x > 0) → (0 ≤ a ∧ a < 8/9) := sorry

end find_a_b_range_of_a_l85_85130


namespace hoseok_divides_number_l85_85156

theorem hoseok_divides_number (x : ℕ) (h : x / 6 = 11) : x = 66 := by
  sorry

end hoseok_divides_number_l85_85156


namespace max_c_magnitude_l85_85380

variables {a b c : ℝ × ℝ}

-- Definitions of the given conditions
def unit_vector (v : ℝ × ℝ) : Prop := ‖v‖ = 1
def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0
def satisfied_c (c a b : ℝ × ℝ) : Prop := ‖c - (a + b)‖ = 2

-- Main theorem to prove
theorem max_c_magnitude (ha : unit_vector a) (hb : unit_vector b) (hab : orthogonal a b) (hc : satisfied_c c a b) : ‖c‖ ≤ 2 + Real.sqrt 2 := 
sorry

end max_c_magnitude_l85_85380


namespace alice_vs_bob_payment_multiple_l85_85301

theorem alice_vs_bob_payment_multiple :
  let alice_acorns := 3600
  let price_per_acorn := 15
  let bob_payment := 6000
  let total_alice_payment := alice_acorns * price_per_acorn
  total_alice_payment / bob_payment = 9 := by
  -- define the variables as per the conditions
  let alice_acorns := 3600
  let price_per_acorn := 15
  let bob_payment := 6000
  let total_alice_payment := alice_acorns * price_per_acorn
  -- define the target statement
  show total_alice_payment / bob_payment = 9
  sorry

end alice_vs_bob_payment_multiple_l85_85301


namespace max_intersections_l85_85569

-- Define the conditions
def num_points_x : ℕ := 15
def num_points_y : ℕ := 10

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the problem statement
theorem max_intersections (I : ℕ) :
  (15 : ℕ) == num_points_x →
  (10 : ℕ) == num_points_y →
  (I = binom 15 2 * binom 10 2) →
  I = 4725 := by
  -- We add sorry to skip the proof
  sorry

end max_intersections_l85_85569


namespace transformed_curve_is_circle_l85_85019

open Real

def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * cos θ^2 + 4 * sin θ^2)

def cartesian_curve (x y: ℝ) : Prop :=
  3 * x^2 + 4 * y^2 = 12

def transformation (x y x' y' : ℝ) : Prop :=
  x' = x / 2 ∧ y' = y * sqrt (3 / 3)

theorem transformed_curve_is_circle (x y x' y' : ℝ) 
  (h1: cartesian_curve x y) (h2: transformation x y x' y') : 
  (x'^2 + y'^2 = 1) :=
sorry

end transformed_curve_is_circle_l85_85019


namespace N_square_solutions_l85_85698

theorem N_square_solutions :
  ∀ N : ℕ, (N > 0 → ∃ k : ℕ, 2^N - 2 * N = k^2) → (N = 1 ∨ N = 2) :=
by
  sorry

end N_square_solutions_l85_85698


namespace construct_triangle_l85_85617

variable (h_a h_b h_c : ℝ)

noncomputable def triangle_exists_and_similar :=
  ∃ (a b c : ℝ), (a = h_b) ∧ (b = h_a) ∧ (c = h_a * h_b / h_c) ∧
  (∃ (area : ℝ), area = 1/2 * a * (h_a * h_c / h_b) ∧ area = 1/2 * b * (h_b * h_c / h_a) ∧ area = 1/2 * c * h_c)

theorem construct_triangle (h_a h_b h_c : ℝ) :
  ∃ a b c, a = h_b ∧ b = h_a ∧ c = h_a * h_b / h_c ∧
  ∃ area, area = 1/2 * a * (h_a * h_c / h_b) ∧ area = 1/2 * b * (h_b * h_c / h_a) ∧ area = 1/2 * c * h_c := 
  sorry

end construct_triangle_l85_85617


namespace parallel_lines_sufficient_not_necessary_condition_l85_85715

theorem parallel_lines_sufficient_not_necessary_condition {a : ℝ} :
  (a = 4) → (∀ x y : ℝ, (a * x + 8 * y - 3 = 0) ↔ (2 * x + a * y - a = 0)) :=
by sorry

end parallel_lines_sufficient_not_necessary_condition_l85_85715


namespace books_before_grant_correct_l85_85171

-- Definitions based on the given conditions
def books_purchased : ℕ := 2647
def total_books_now : ℕ := 8582

-- Definition and the proof statement
def books_before_grant : ℕ := 5935

-- Proof statement: The number of books before the grant plus the books purchased equals the total books now
theorem books_before_grant_correct :
  books_before_grant + books_purchased = total_books_now :=
by
  -- Predictably, no need to complete proof, 'sorry' is used.
  sorry

end books_before_grant_correct_l85_85171


namespace solution_set_of_inequality_l85_85075

theorem solution_set_of_inequality :
  { x : ℝ | x ^ 2 - 5 * x + 6 ≤ 0 } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } :=
by 
  sorry

end solution_set_of_inequality_l85_85075


namespace tunnel_length_is_correct_l85_85645

-- Define the conditions given in the problem
def length_of_train : ℕ := 90
def speed_of_train : ℕ := 160
def time_to_pass_tunnel : ℕ := 3

-- Define the length of the tunnel to be proven
def length_of_tunnel : ℕ := 480 - length_of_train

-- Define the statement to be proven
theorem tunnel_length_is_correct : length_of_tunnel = 390 := by
  sorry

end tunnel_length_is_correct_l85_85645


namespace markup_percentage_l85_85197

-- Define the purchase price and the gross profit
def purchase_price : ℝ := 54
def gross_profit : ℝ := 18

-- Define the sale price after discount
def sale_discount : ℝ := 0.8

-- Given that the sale price after the discount is purchase_price + gross_profit
theorem markup_percentage (M : ℝ) (SP : ℝ) : 
  SP = purchase_price * (1 + M / 100) → -- selling price as function of markup
  (SP * sale_discount = purchase_price + gross_profit) → -- sale price after 20% discount
  M = 66.67 := 
by
  -- sorry to skip the proof
  sorry

end markup_percentage_l85_85197


namespace diameter_of_large_circle_is_19_312_l85_85281

noncomputable def diameter_large_circle (r_small : ℝ) (n : ℕ) : ℝ :=
  let side_length_inner_octagon := 2 * r_small
  let radius_inner_octagon := side_length_inner_octagon / (2 * Real.sin (Real.pi / n)) / 2
  let radius_large_circle := radius_inner_octagon + r_small
  2 * radius_large_circle

theorem diameter_of_large_circle_is_19_312 :
  diameter_large_circle 4 8 = 19.312 :=
by
  sorry

end diameter_of_large_circle_is_19_312_l85_85281


namespace tan_theta_plus_pi_over_eight_sub_inv_l85_85841

/-- Given the trigonometric identity, we can prove the tangent calculation -/
theorem tan_theta_plus_pi_over_eight_sub_inv (θ : ℝ)
  (h : 3 * Real.sin θ + Real.cos θ = Real.sqrt 10) :
  Real.tan (θ + Real.pi / 8) - 1 / Real.tan (θ + Real.pi / 8) = -14 := 
sorry

end tan_theta_plus_pi_over_eight_sub_inv_l85_85841


namespace find_z_l85_85764

theorem find_z (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : z * i = 2 - i) : z = -1 - 2 * i := 
by
  sorry

end find_z_l85_85764


namespace tile_count_difference_l85_85629

theorem tile_count_difference :
  let red_initial := 15
  let yellow_initial := 10
  let yellow_added := 18
  let yellow_total := yellow_initial + yellow_added
  let red_total := red_initial
  yellow_total - red_total = 13 :=
by
  sorry

end tile_count_difference_l85_85629


namespace arithmetic_sequence_problem_l85_85017

noncomputable def a1 := 3
noncomputable def S (n : ℕ) (a1 d : ℕ) : ℕ := n * (a1 + (n - 1) * d / 2)

theorem arithmetic_sequence_problem (d : ℕ) 
  (h1 : S 1 a1 d = 3) 
  (h2 : S 1 a1 d / 2 + S 4 a1 d / 4 = 18) : 
  S 5 a1 d = 75 :=
sorry

end arithmetic_sequence_problem_l85_85017


namespace total_production_l85_85503

theorem total_production (S : ℝ) 
  (h1 : 4 * S = 4400) : 
  4400 + S = 5500 := 
by
  sorry

end total_production_l85_85503


namespace monotonically_increasing_interval_l85_85348

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

theorem monotonically_increasing_interval :
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) :=
sorry

end monotonically_increasing_interval_l85_85348


namespace fraction_identity_l85_85300

noncomputable def simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : ℝ :=
  (1 / (2 * a * b)) + (b / (4 * a))

theorem fraction_identity (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  simplify_fraction a b h₁ h₂ = (2 + b^2) / (4 * a * b) :=
by sorry

end fraction_identity_l85_85300


namespace minimum_value_x3_plus_y3_minus_5xy_l85_85896

theorem minimum_value_x3_plus_y3_minus_5xy (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  x^3 + y^3 - 5 * x * y ≥ -125 / 27 := 
sorry

end minimum_value_x3_plus_y3_minus_5xy_l85_85896


namespace num_sides_polygon_l85_85059

theorem num_sides_polygon (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end num_sides_polygon_l85_85059


namespace gain_percentage_calculation_l85_85023

theorem gain_percentage_calculation 
  (C S : ℝ)
  (h1 : 30 * S = 40 * C) :
  (10 * S / (30 * C)) * 100 = 44.44 :=
by
  sorry

end gain_percentage_calculation_l85_85023


namespace total_sand_arrived_l85_85691

theorem total_sand_arrived :
  let truck1_carry := 4.1
  let truck1_loss := 2.4
  let truck2_carry := 5.7
  let truck2_loss := 3.6
  let truck3_carry := 8.2
  let truck3_loss := 1.9
  (truck1_carry - truck1_loss) + 
  (truck2_carry - truck2_loss) + 
  (truck3_carry - truck3_loss) = 10.1 :=
by
  sorry

end total_sand_arrived_l85_85691


namespace correct_statements_l85_85758

variables {n : ℕ}
noncomputable def S (n : ℕ) : ℝ := (n + 1) / n
noncomputable def T (n : ℕ) : ℝ := (n + 1)
noncomputable def a (n : ℕ) : ℝ := if n = 1 then 2 else (-(1:ℝ)) / (n * (n - 1))

theorem correct_statements (n : ℕ) (hn : n ≠ 0) :
  (S n + T n = S n * T n) ∧ (a 1 = 2) ∧ (∀ n, ∃ d, ∀ m, T (n + m) - T n = m * d) ∧ (S n = (n + 1) / n) :=
by
  sorry

end correct_statements_l85_85758


namespace problem1_problem2_l85_85488

-- Problem 1: If a is parallel to b, then x = 4
theorem problem1 (x : ℝ) (u v : ℝ × ℝ) : 
  let a := (1, 1)
  let b := (4, x)
  (a.1 / b.1 = a.2 / b.2) → x = 4 := 
by 
  intros a b h
  dsimp [a, b] at h
  sorry

-- Problem 2: If (u - 2 * v) is perpendicular to (u + v), then x = -6
theorem problem2 (x : ℝ) (a u v : ℝ × ℝ) : 
  let a := (1, 1)
  let b := (4, x)
  let u := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let v := (2 * a.1 + b.1, 2 * a.2 + b.2)
  ((u.1 - 2 * v.1) * (u.1 + v.1) + (u.2 - 2 * v.2) * (u.2 + v.2) = 0) → x = -6 := 
by 
  intros a b u v h
  dsimp [a, b, u, v] at h
  sorry

end problem1_problem2_l85_85488


namespace james_ride_time_l85_85249

theorem james_ride_time :
  let distance := 80 
  let speed := 16 
  distance / speed = 5 := 
by
  -- sorry to skip the proof
  sorry

end james_ride_time_l85_85249


namespace probability_interval_l85_85279

noncomputable def Phi : ℝ → ℝ := sorry -- assuming Φ is a given function for CDF of a standard normal distribution

theorem probability_interval (h : Phi 1.98 = 0.9762) : 
  2 * Phi 1.98 - 1 = 0.9524 :=
by
  sorry

end probability_interval_l85_85279


namespace range_of_k_intersection_l85_85927

theorem range_of_k_intersection (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (k^2 - 1) * x1^2 + 4 * k * x1 + 10 = 0 ∧ (k^2 - 1) * x2^2 + 4 * k * x2 + 10 = 0) ↔ (-1 < k ∧ k < 1) :=
by
  sorry

end range_of_k_intersection_l85_85927


namespace sum_abcd_eq_16_l85_85653

variable (a b c d : ℝ)

def cond1 : Prop := a^2 + b^2 + c^2 + d^2 = 250
def cond2 : Prop := a * b + b * c + c * a + a * d + b * d + c * d = 3

theorem sum_abcd_eq_16 (h1 : cond1 a b c d) (h2 : cond2 a b c d) : a + b + c + d = 16 := 
by 
  sorry

end sum_abcd_eq_16_l85_85653


namespace simplify_expression_l85_85190

theorem simplify_expression :
  let a := 7
  let b := 11
  let c := 19
  (49 * (1 / 11 - 1 / 19) + 121 * (1 / 19 - 1 / 7) + 361 * (1 / 7 - 1 / 11)) /
  (7 * (1 / 11 - 1 / 19) + 11 * (1 / 19 - 1 / 7) + 19 * (1 / 7 - 1 / 11)) = 37 := by
  sorry

end simplify_expression_l85_85190


namespace smallest_C_l85_85788

-- Defining the problem and the conditions
theorem smallest_C (k : ℕ) (C : ℕ) :
  (∀ n : ℕ, n ≥ k → (C * Nat.choose (2 * n) (n + k)) % (n + k + 1) = 0) ↔
  C = 2 * k + 1 :=
by sorry

end smallest_C_l85_85788


namespace sqrt_12_eq_2_sqrt_3_sqrt_1_div_2_eq_sqrt_2_div_2_l85_85563

theorem sqrt_12_eq_2_sqrt_3 : Real.sqrt 12 = 2 * Real.sqrt 3 := sorry

theorem sqrt_1_div_2_eq_sqrt_2_div_2 : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 := sorry

end sqrt_12_eq_2_sqrt_3_sqrt_1_div_2_eq_sqrt_2_div_2_l85_85563


namespace sum_primes_between_20_and_40_l85_85985

theorem sum_primes_between_20_and_40 :
  23 + 29 + 31 + 37 = 120 :=
by
  sorry

end sum_primes_between_20_and_40_l85_85985


namespace clients_select_two_cars_l85_85084

theorem clients_select_two_cars (cars clients selections : ℕ) (total_selections : ℕ)
  (h1 : cars = 10) (h2 : clients = 15) (h3 : total_selections = cars * 3) (h4 : total_selections = clients * selections) :
  selections = 2 :=
by 
  sorry

end clients_select_two_cars_l85_85084


namespace find_abc_l85_85034

theorem find_abc
  (a b c : ℝ)
  (h : ∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|):
  (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) ∨ 
  (a = 0 ∧ b = 1 ∧ c = 0) ∨ (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 1) ∨ (a = 0 ∧ b = 0 ∧ c = -1) :=
sorry

end find_abc_l85_85034


namespace series_product_solution_l85_85688

theorem series_product_solution (y : ℚ) :
  ( (∑' n, (1 / 2) * (1 / 3) ^ n) * (∑' n, (1 / 3) * (-1 / 3) ^ n) ) = ∑' n, (1 / y) ^ (n + 1) → y = 19 / 3 :=
by
  sorry

end series_product_solution_l85_85688


namespace solve_x_eq_40_l85_85609

theorem solve_x_eq_40 : ∀ (x : ℝ), x + 2 * x = 400 - (3 * x + 4 * x) → x = 40 :=
by
  intro x
  intro h
  sorry

end solve_x_eq_40_l85_85609


namespace find_operation_l85_85977

theorem find_operation (a b : ℝ) (h_a : a = 0.137) (h_b : b = 0.098) :
  ((a + b) ^ 2 - (a - b) ^ 2) / (a * b) = 4 :=
by
  sorry

end find_operation_l85_85977


namespace cost_of_4_bags_of_ice_l85_85663

theorem cost_of_4_bags_of_ice (
  cost_per_2_bags : ℝ := 1.46
) 
  (h : cost_per_2_bags / 2 = 0.73)
  :
  4 * (cost_per_2_bags / 2) = 2.92 :=
by 
  sorry

end cost_of_4_bags_of_ice_l85_85663


namespace sail_pressure_l85_85308

def pressure (k A V : ℝ) : ℝ := k * A * V^2

theorem sail_pressure (k : ℝ)
  (h_k : k = 1 / 800) 
  (A : ℝ) 
  (V : ℝ) 
  (P : ℝ)
  (h_initial : A = 1 ∧ V = 20 ∧ P = 0.5) 
  (A2 : ℝ) 
  (V2 : ℝ) 
  (h_doubled : A2 = 2 ∧ V2 = 30) :
  pressure k A2 V2 = 2.25 :=
by
  sorry

end sail_pressure_l85_85308


namespace claudia_coins_l85_85378

theorem claudia_coins (x y : ℕ) (h1 : x + y = 15) (h2 : 29 - x = 26) :
  y = 12 :=
by
  sorry

end claudia_coins_l85_85378


namespace rhombus_diagonals_l85_85517

theorem rhombus_diagonals (p d1 d2 : ℝ) (h1 : p = 100) (h2 : abs (d1 - d2) = 34) :
  ∃ d1 d2 : ℝ, d1 = 14 ∧ d2 = 48 :=
by
  -- proof omitted
  sorry

end rhombus_diagonals_l85_85517


namespace range_of_a_l85_85786

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), 0 < x ∧ x < 4 → |x - 1| < a) ↔ 3 ≤ a :=
sorry

end range_of_a_l85_85786


namespace watermelon_cost_l85_85856

-- Define the problem conditions
def container_full_conditions (w m : ℕ) : Prop :=
  w + m = 150 ∧ (w / 160) + (m / 120) = 1

def equal_total_values (w m w_value m_value : ℕ) : Prop :=
  w * w_value = m * m_value ∧ w * w_value + m * m_value = 24000

-- Define the proof problem
theorem watermelon_cost (w m w_value m_value : ℕ) (hw : container_full_conditions w m) (hv : equal_total_values w m w_value m_value) :
  w_value = 100 :=
by
  -- precise proof goes here
  sorry

end watermelon_cost_l85_85856


namespace complex_number_is_real_implies_m_eq_3_l85_85759

open Complex

theorem complex_number_is_real_implies_m_eq_3 (m : ℝ) :
  (∃ (z : ℂ), z = (1 / (m + 5) : ℝ) + (m^2 + 2 * m - 15) * I ∧ z.im = 0) →
  m = 3 :=
by
  sorry

end complex_number_is_real_implies_m_eq_3_l85_85759


namespace fair_hair_percentage_l85_85829

-- Define the main entities
variables (E F W : ℝ)

-- Define the conditions given in the problem
def women_with_fair_hair : Prop := W = 0.32 * E
def fair_hair_women_ratio : Prop := W = 0.40 * F

-- Define the theorem to prove
theorem fair_hair_percentage
  (hwf: women_with_fair_hair E W)
  (fhr: fair_hair_women_ratio W F) :
  (F / E) * 100 = 80 :=
by
  sorry

end fair_hair_percentage_l85_85829


namespace modulus_of_complex_l85_85081

noncomputable def modulus (z : Complex) : Real :=
  Complex.abs z

theorem modulus_of_complex :
  ∀ (i : Complex) (z : Complex), i = Complex.I → z = i * (2 - i) → modulus z = Real.sqrt 5 :=
by
  intros i z hi hz
  -- Proof omitted
  sorry

end modulus_of_complex_l85_85081


namespace train_speed_l85_85542

def train_length : ℕ := 180
def crossing_time : ℕ := 12

theorem train_speed :
  train_length / crossing_time = 15 := sorry

end train_speed_l85_85542


namespace find_n_divisible_by_35_l85_85444

-- Define the five-digit number for some digit n
def num (n : ℕ) : ℕ := 80000 + n * 1000 + 975

-- Define the conditions
def divisible_by_5 (d : ℕ) : Prop := d % 5 = 0
def divisible_by_7 (d : ℕ) : Prop := d % 7 = 0
def divisible_by_35 (d : ℕ) : Prop := divisible_by_5 d ∧ divisible_by_7 d

-- Statement of the problem for proving given conditions and the correct answer
theorem find_n_divisible_by_35 : ∃ (n : ℕ), (num n % 35 = 0) ∧ n = 6 := by
  sorry

end find_n_divisible_by_35_l85_85444


namespace Z_real_iff_m_eq_neg3_or_5_Z_pure_imaginary_iff_m_eq_neg2_Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5_l85_85316

open Complex

noncomputable def Z (m : ℝ) : ℂ :=
  (m ^ 2 + 5 * m + 6) + (m ^ 2 - 2 * m - 15) * Complex.I

namespace ComplexNumbersProofs

-- Prove that Z is a real number if and only if m = -3 or m = 5
theorem Z_real_iff_m_eq_neg3_or_5 (m : ℝ) :
  (Z m).im = 0 ↔ (m = -3 ∨ m = 5) := 
by
  sorry

-- Prove that Z is a pure imaginary number if and only if m = -2
theorem Z_pure_imaginary_iff_m_eq_neg2 (m : ℝ) :
  (Z m).re = 0 ↔ (m = -2) := 
by
  sorry

-- Prove that the point corresponding to Z lies in the fourth quadrant if and only if -2 < m < 5
theorem Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5 (m : ℝ) :
  (Z m).re > 0 ∧ (Z m).im < 0 ↔ (-2 < m ∧ m < 5) :=
by
  sorry

end ComplexNumbersProofs

end Z_real_iff_m_eq_neg3_or_5_Z_pure_imaginary_iff_m_eq_neg2_Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5_l85_85316


namespace combined_average_l85_85063

-- Given Conditions
def num_results_1 : ℕ := 30
def avg_results_1 : ℝ := 20
def num_results_2 : ℕ := 20
def avg_results_2 : ℝ := 30
def num_results_3 : ℕ := 25
def avg_results_3 : ℝ := 40

-- Helper Definitions
def total_sum_1 : ℝ := num_results_1 * avg_results_1
def total_sum_2 : ℝ := num_results_2 * avg_results_2
def total_sum_3 : ℝ := num_results_3 * avg_results_3
def total_sum_all : ℝ := total_sum_1 + total_sum_2 + total_sum_3
def total_number_results : ℕ := num_results_1 + num_results_2 + num_results_3

-- Problem Statement
theorem combined_average : 
  (total_sum_all / (total_number_results:ℝ)) = 29.33 := 
by 
  sorry

end combined_average_l85_85063


namespace infinitely_many_composite_numbers_l85_85807

-- We define n in a specialized form.
def n (m : ℕ) : ℕ := (3 * m) ^ 3

-- We state that m is an odd positive integer.
def odd_positive_integer (m : ℕ) : Prop := m > 0 ∧ (m % 2 = 1)

-- The main statement: for infinitely many odd values of n, 2^n + n - 1 is composite.
theorem infinitely_many_composite_numbers : 
  ∃ (m : ℕ), odd_positive_integer m ∧ Nat.Prime (n m) ∧ ∃ d : ℕ, d > 1 ∧ d < n m ∧ (2^(n m) + n m - 1) % d = 0 :=
by
  sorry

end infinitely_many_composite_numbers_l85_85807


namespace smallest_nonprime_greater_than_with_no_prime_factors_less_than_15_l85_85893

-- Definitions for the given conditions.
def is_prime (p : ℕ) : Prop := (p > 1) ∧ ∀ d : ℕ, d ∣ p → (d = 1 ∨ d = p)

def has_no_prime_factors_less_than (n : ℕ) (m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

def is_nonprime (n : ℕ) : Prop := ¬ is_prime n

-- The main theorem based on the proof problem.
theorem smallest_nonprime_greater_than_with_no_prime_factors_less_than_15 
  (n : ℕ) (h1 : n > 1) (h2 : has_no_prime_factors_less_than n 15) (h3 : is_nonprime n) : 
  280 < n ∧ n ≤ 290 :=
by
  sorry

end smallest_nonprime_greater_than_with_no_prime_factors_less_than_15_l85_85893


namespace carl_weight_l85_85101

variable (C R B : ℕ)

theorem carl_weight (h1 : B = R + 9) (h2 : R = C + 5) (h3 : B = 159) : C = 145 :=
by
  sorry

end carl_weight_l85_85101


namespace real_roots_range_of_m_l85_85570

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem real_roots_range_of_m :
  (∃ x : ℝ, x^2 + 4 * m * x + 4 * m^2 + 2 * m + 3 = 0) ∨ 
  (∃ x : ℝ, x^2 + (2 * m + 1) * x + m^2 = 0) ↔ 
  m ≤ -3 / 2 ∨ m ≥ -1 / 4 :=
by
  sorry

end real_roots_range_of_m_l85_85570


namespace parabola_vertex_b_l85_85983

theorem parabola_vertex_b (a b c p : ℝ) (h₁ : p ≠ 0)
  (h₂ : ∀ x, (x = p → -p = a * (p^2) + b * p + c) ∧ (x = 0 → p = c)) :
  b = - (4 / p) :=
sorry

end parabola_vertex_b_l85_85983


namespace sum_of_values_l85_85132

def r (x : ℝ) : ℝ := abs (x + 1) - 3
def s (x : ℝ) : ℝ := -(abs (x + 2))

theorem sum_of_values :
  (s (r (-5)) + s (r (-4)) + s (r (-3)) + s (r (-2)) + s (r (-1)) + s (r (0)) + s (r (1)) + s (r (2)) + s (r (3))) = -37 :=
by {
  sorry
}

end sum_of_values_l85_85132


namespace line_through_point_area_T_l85_85935

variable (a T : ℝ)

def equation_of_line (x y : ℝ) : Prop := 2 * T * x - a^2 * y + 2 * a * T = 0

theorem line_through_point_area_T :
  ∃ (x y : ℝ), equation_of_line a T x y ∧ x = -a ∧ y = (2 * T) / a :=
by
  sorry

end line_through_point_area_T_l85_85935


namespace determine_h_l85_85509

theorem determine_h (x : ℝ) : 
  ∃ h : ℝ → ℝ, (4*x^4 + 11*x^3 + h x = 10*x^3 - x^2 + 4*x - 7) ↔ (h x = -4*x^4 - x^3 - x^2 + 4*x - 7) :=
by
  sorry

end determine_h_l85_85509


namespace remaining_days_to_finish_coke_l85_85490

def initial_coke_in_ml : ℕ := 2000
def daily_consumption_in_ml : ℕ := 200
def days_already_drunk : ℕ := 3

theorem remaining_days_to_finish_coke : 
  (initial_coke_in_ml / daily_consumption_in_ml) - days_already_drunk = 7 := 
by
  sorry -- Proof placeholder

end remaining_days_to_finish_coke_l85_85490


namespace symmetry_condition_l85_85448

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 3)

theorem symmetry_condition (ϕ : ℝ) (hϕ : |ϕ| ≤ π / 2)
    (hxy: ∀ x : ℝ, f (x + ϕ) = f (-x + ϕ)) : ϕ = π / 6 :=
by
  -- Since the problem specifically asks for the statement only and not the proof steps,
  -- a "sorry" is used to skip the proof content.
  sorry

end symmetry_condition_l85_85448


namespace identify_stolen_bag_with_two_weighings_l85_85338

-- Definition of the weights of the nine bags
def weights : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Statement of the problem: Using two weighings on a balance scale without weights,
-- prove that it is possible to identify the specific bag from which the treasure was stolen.
theorem identify_stolen_bag_with_two_weighings (stolen_bag : {n // n < 9}) :
  ∃ (group1 group2 : List ℕ), group1 ≠ group2 ∧ (group1.sum = 11 ∨ group1.sum = 15) ∧ (group2.sum = 11 ∨ group2.sum = 15) →
  ∃ (b1 b2 b3 : ℕ), b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3 ∧ b1 + b2 + b3 = 6 ∧ (b1 + b2 = 11 ∨ b1 + b2 = 15) := sorry

end identify_stolen_bag_with_two_weighings_l85_85338


namespace middle_number_is_11_l85_85679

theorem middle_number_is_11 (a b c : ℕ) (h1 : a + b = 18) (h2 : a + c = 22) (h3 : b + c = 26) (h4 : c - a = 10) :
  b = 11 :=
by
  sorry

end middle_number_is_11_l85_85679


namespace prove_billy_age_l85_85194

-- Define B and J as real numbers representing the ages of Billy and Joe respectively
variables (B J : ℝ)

-- State the conditions
def billy_triple_of_joe : Prop := B = 3 * J
def sum_of_ages : Prop := B + J = 63

-- State the proposition to prove
def billy_age_proof : Prop := B = 47.25

-- Main theorem combining the conditions and the proof statement
theorem prove_billy_age (h1 : billy_triple_of_joe B J) (h2 : sum_of_ages B J) : billy_age_proof B :=
by
  sorry

end prove_billy_age_l85_85194


namespace sufficient_but_not_necessary_condition_l85_85844

theorem sufficient_but_not_necessary_condition (h1 : 1^2 - 1 = 0) (h2 : ∀ x, x^2 - 1 = 0 → (x = 1 ∨ x = -1)) :
  (∀ x, x = 1 → x^2 - 1 = 0) ∧ ¬ (∀ x, x^2 - 1 = 0 → x = 1) := by
  sorry

end sufficient_but_not_necessary_condition_l85_85844


namespace solve_for_a_l85_85898

theorem solve_for_a (a : ℕ) (h : a^3 = 21 * 25 * 35 * 63) : a = 105 :=
sorry

end solve_for_a_l85_85898


namespace part_a_part_b_l85_85314

-- Given distinct primes p and q
variables (p q : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] (h : p ≠ q)

-- Prove p^q + q^p ≡ p + q (mod pq)
theorem part_a (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q) :
  (p^q + q^p) % (p * q) = (p + q) % (p * q) := by
  sorry

-- Given distinct primes p and q, and neither are 2
theorem part_b (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q) (hp2 : p ≠ 2) (hq2 : q ≠ 2) :
  Even (Nat.floor ((p^q + q^p) / (p * q))) := by
  sorry

end part_a_part_b_l85_85314


namespace hockey_players_l85_85161

theorem hockey_players (n : ℕ) (h1 : n < 30) (h2 : n % 2 = 0) (h3 : n % 4 = 0) (h4 : n % 7 = 0) :
  (n / 4 = 7) :=
by
  sorry

end hockey_players_l85_85161


namespace bead_count_l85_85770

variable (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)

theorem bead_count : total_beads = 40 ∧ blue_beads = 5 ∧ red_beads = 2 * blue_beads ∧ white_beads = blue_beads + red_beads ∧ silver_beads = total_beads - (blue_beads + red_beads + white_beads) → silver_beads = 10 :=
by
  intro h
  sorry

end bead_count_l85_85770


namespace correct_relation_l85_85186

def satisfies_relation : Prop :=
  (∀ x y, (x = 0 ∧ y = 200) ∨ (x = 1 ∧ y = 170) ∨ (x = 2 ∧ y = 120) ∨ (x = 3 ∧ y = 50) ∨ (x = 4 ∧ y = 0) →
  y = 200 - 10 * x - 10 * x^2) 

theorem correct_relation : satisfies_relation :=
sorry

end correct_relation_l85_85186


namespace evaluate_at_minus_three_l85_85610

def g (x : ℝ) : ℝ := 3 * x^5 - 5 * x^4 + 9 * x^3 - 6 * x^2 + 15 * x - 210

theorem evaluate_at_minus_three : g (-3) = -1686 :=
by
  sorry

end evaluate_at_minus_three_l85_85610


namespace range_of_m_l85_85666

theorem range_of_m (m x y : ℝ) 
  (h1 : x + y = -1) 
  (h2 : 5 * x + 2 * y = 6 * m + 7) 
  (h3 : 2 * x - y < 19) : 
  m < 3 / 2 := 
sorry

end range_of_m_l85_85666


namespace larger_triangle_perimeter_l85_85501

theorem larger_triangle_perimeter 
    (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
    (h1 : a = 6) (h2 : b = 8)
    (hypo_large : ∀ c : ℝ, c = 20) : 
    (2 * a + 2 * b + 20 = 48) :=
by {
  sorry
}

end larger_triangle_perimeter_l85_85501


namespace unique_real_solution_k_eq_35_over_4_l85_85340

theorem unique_real_solution_k_eq_35_over_4 :
  ∃ k : ℚ, (∀ x : ℝ, (x + 5) * (x + 3) = k + 3 * x) ↔ (k = 35 / 4) :=
by
  sorry

end unique_real_solution_k_eq_35_over_4_l85_85340


namespace M_gt_N_l85_85324

-- Define the variables and conditions
variables (x y : ℝ)
noncomputable def M := x^2 + y^2
noncomputable def N := 2*x + 6*y - 11

-- State the theorem
theorem M_gt_N : M x y > N x y := by
  sorry -- Placeholder for the proof

end M_gt_N_l85_85324


namespace total_ladybugs_and_ants_l85_85214

def num_leaves : ℕ := 84
def ladybugs_per_leaf : ℕ := 139
def ants_per_leaf : ℕ := 97

def total_ladybugs := ladybugs_per_leaf * num_leaves
def total_ants := ants_per_leaf * num_leaves
def total_insects := total_ladybugs + total_ants

theorem total_ladybugs_and_ants : total_insects = 19824 := by
  sorry

end total_ladybugs_and_ants_l85_85214


namespace pam_total_apples_l85_85789

theorem pam_total_apples (pam_bags : ℕ) (gerald_bags_apples : ℕ) (gerald_bags_factor : ℕ) 
  (pam_bags_count : pam_bags = 10)
  (gerald_apples_count : gerald_bags_apples = 40)
  (gerald_bags_ratio : gerald_bags_factor = 3) : 
  pam_bags * gerald_bags_factor * gerald_bags_apples = 1200 := by
  sorry

end pam_total_apples_l85_85789


namespace peter_read_more_books_l85_85265

/-
Given conditions:
  Peter has 20 books.
  Peter has read 40% of them.
  Peter's brother has read 10% of them.
We aim to prove that Peter has read 6 more books than his brother.
-/

def total_books : ℕ := 20
def peter_read_fraction : ℚ := 0.4
def brother_read_fraction : ℚ := 0.1

def books_read_by_peter := total_books * peter_read_fraction
def books_read_by_brother := total_books * brother_read_fraction

theorem peter_read_more_books :
  books_read_by_peter - books_read_by_brother = 6 := by
  sorry

end peter_read_more_books_l85_85265


namespace eval_expression_l85_85823

theorem eval_expression : (256 : ℝ) ^ ((-2 : ℝ) ^ (-3 : ℝ)) = 1 / 2 := by
  sorry

end eval_expression_l85_85823


namespace unique_ordered_triple_l85_85223

theorem unique_ordered_triple (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ab : Nat.lcm a b = 500) (h_bc : Nat.lcm b c = 2000) (h_ca : Nat.lcm c a = 2000) :
  (a = 100 ∧ b = 2000 ∧ c = 2000) :=
by
  sorry

end unique_ordered_triple_l85_85223


namespace sequence_value_is_correct_l85_85551

theorem sequence_value_is_correct (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 2) : a 8 = 15 :=
sorry

end sequence_value_is_correct_l85_85551


namespace power_sum_greater_than_linear_l85_85535

theorem power_sum_greater_than_linear (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0) (hn : n ≥ 2) :
  (1 + x) ^ n > 1 + n * x :=
sorry

end power_sum_greater_than_linear_l85_85535


namespace valid_tickets_percentage_l85_85285

theorem valid_tickets_percentage (cars : ℕ) (people_without_payment : ℕ) (P : ℚ) 
  (h_cars : cars = 300) (h_people_without_payment : people_without_payment = 30) 
  (h_total_valid_or_passes : (cars - people_without_payment = 270)) :
  P + (P / 5) = 90 → P = 75 :=
by
  sorry

end valid_tickets_percentage_l85_85285


namespace minimum_value_inequality_equality_condition_exists_l85_85413

theorem minimum_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  6 * c / (3 * a + b) + 6 * a / (b + 3 * c) + 2 * b / (a + c) ≥ 12 := by
  sorry

theorem equality_condition_exists : 
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (6 * c / (3 * a + b) + 6 * a / (b + 3 * c) + 2 * b / (a + c) = 12) := by
  sorry

end minimum_value_inequality_equality_condition_exists_l85_85413


namespace angle_A_in_quadrilateral_l85_85181

noncomputable def degree_measure_A (A B C D : ℝ) := A

theorem angle_A_in_quadrilateral 
  (A B C D : ℝ)
  (hA : A = 3 * B)
  (hC : A = 4 * C)
  (hD : A = 6 * D)
  (sum_angles : A + B + C + D = 360) :
  degree_measure_A A B C D = 206 :=
by
  sorry

end angle_A_in_quadrilateral_l85_85181


namespace triangular_array_sum_digits_l85_85133

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 2080) : 
  (N.digits 10).sum = 10 :=
sorry

end triangular_array_sum_digits_l85_85133


namespace sqrt_a_plus_sqrt_b_eq_3_l85_85045

theorem sqrt_a_plus_sqrt_b_eq_3 (a b : ℝ) (h : (Real.sqrt a + Real.sqrt b) * (Real.sqrt a + Real.sqrt b - 2) = 3) : Real.sqrt a + Real.sqrt b = 3 :=
sorry

end sqrt_a_plus_sqrt_b_eq_3_l85_85045


namespace total_boxes_l85_85853

theorem total_boxes (w1 w2 : ℕ) (h1 : w1 = 400) (h2 : w1 = 2 * w2) : w1 + w2 = 600 := 
by
  sorry

end total_boxes_l85_85853


namespace professional_tax_correct_l85_85469

-- Define the total income and professional deductions
def total_income : ℝ := 50000
def professional_deductions : ℝ := 35000

-- Define the tax rates
def tax_rate_ndfl : ℝ := 0.13
def tax_rate_simplified_income : ℝ := 0.06
def tax_rate_simplified_income_minus_exp : ℝ := 0.15
def tax_rate_professional_income : ℝ := 0.04

-- Define the expected tax amount
def expected_tax_professional_income : ℝ := 2000

-- Define a function to calculate the professional income tax for self-employed individuals
def calculate_professional_income_tax (income : ℝ) (rate : ℝ) : ℝ :=
  income * rate

-- Define the main theorem to assert the correctness of the tax calculation
theorem professional_tax_correct :
  calculate_professional_income_tax total_income tax_rate_professional_income = expected_tax_professional_income :=
by
  sorry

end professional_tax_correct_l85_85469


namespace find_second_divisor_l85_85195

theorem find_second_divisor (k : ℕ) (d : ℕ) 
  (h1 : k % 5 = 2)
  (h2 : k < 42)
  (h3 : k % 7 = 3)
  (h4 : k % d = 5) : d = 12 := 
sorry

end find_second_divisor_l85_85195


namespace linear_dependent_vectors_l85_85702

variable (m : ℝ) (a b : ℝ) 

theorem linear_dependent_vectors :
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
    a • (⟨2, 3⟩ : ℝ × ℝ) + b • (⟨5, m⟩ : ℝ × ℝ) = (⟨0, 0⟩ : ℝ × ℝ)) ↔ m = 15 / 2 :=
sorry

end linear_dependent_vectors_l85_85702


namespace water_needed_to_fill_glasses_l85_85376

theorem water_needed_to_fill_glasses :
  let glasses := 10
  let capacity_per_glass := 6
  let filled_fraction := 4 / 5
  let total_capacity := glasses * capacity_per_glass
  let total_water := glasses * (capacity_per_glass * filled_fraction)
  let water_needed := total_capacity - total_water
  water_needed = 12 :=
by
  sorry

end water_needed_to_fill_glasses_l85_85376


namespace value_of_expression_l85_85384

theorem value_of_expression : (15 + 5)^2 - (15^2 + 5^2) = 150 := by
  sorry

end value_of_expression_l85_85384


namespace mark_second_part_playtime_l85_85712

theorem mark_second_part_playtime (total_time initial_time sideline_time : ℕ) 
  (h1 : total_time = 90) (h2 : initial_time = 20) (h3 : sideline_time = 35) :
  total_time - initial_time - sideline_time = 35 :=
sorry

end mark_second_part_playtime_l85_85712


namespace geometric_sequence_l85_85468

theorem geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) 
  (h2 : (3 * S 1, 2 * S 2, S 3) = (3 * S 1, 2 * S 2, S 3) ∧ (4 * S 2 = 3 * S 1 + S 3)) 
  (hq_pos : q ≠ 0) 
  (hq : ∀ n, a (n + 1) = a n * q):
  ∀ n, a n = 3^(n-1) :=
by
  sorry

end geometric_sequence_l85_85468


namespace average_runs_l85_85633

theorem average_runs (games : ℕ) (runs1 matches1 runs2 matches2 runs3 matches3 : ℕ)
  (h1 : runs1 = 1) 
  (h2 : matches1 = 1) 
  (h3 : runs2 = 4) 
  (h4 : matches2 = 2)
  (h5 : runs3 = 5) 
  (h6 : matches3 = 3) 
  (h_games : games = matches1 + matches2 + matches3) :
  (runs1 * matches1 + runs2 * matches2 + runs3 * matches3) / games = 4 :=
by
  sorry

end average_runs_l85_85633


namespace total_students_count_l85_85700

-- Definitions for the conditions
def ratio_girls_to_boys (g b : ℕ) : Prop := g * 4 = b * 3
def boys_count : ℕ := 28

-- Theorem to prove the total number of students
theorem total_students_count {g : ℕ} (h : ratio_girls_to_boys g boys_count) : g + boys_count = 49 :=
sorry

end total_students_count_l85_85700


namespace part1_combined_time_part2_copier_A_insufficient_part3_combined_after_repair_l85_85121

-- Definitions for times needed by copiers A and B
def time_A : ℕ := 90
def time_B : ℕ := 60

-- (1) Combined time for both copiers
theorem part1_combined_time : 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 36 = 1 := 
by sorry

-- (2) Time left for copier A alone
theorem part2_copier_A_insufficient (mins_combined : ℕ) (time_left : ℕ) : 
  mins_combined = 30 → time_left = 13 → 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 30 + time_left / (time_A : ℝ) ≠ 1 := 
by sorry

-- (3) Combined time with B after repair is sufficient
theorem part3_combined_after_repair (mins_combined : ℕ) (mins_repair_B : ℕ) (time_left : ℕ) : 
  mins_combined = 30 → mins_repair_B = 9 → time_left = 13 →
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 30 + 9 / (time_A : ℝ) + 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 2.4 = 1 := 
by sorry

end part1_combined_time_part2_copier_A_insufficient_part3_combined_after_repair_l85_85121


namespace circle_O₁_equation_sum_of_squares_constant_l85_85245

-- Given conditions
def circle_O (x y : ℝ) := x^2 + y^2 = 25
def center_O₁ (m : ℝ) : ℝ × ℝ := (m, 0) 
def intersect_point := (3, 4)
def is_intersection (x y : ℝ) := circle_O x y ∧ (x - intersect_point.1)^2 + (y - intersect_point.2)^2 = 0
def line_passing_P (k : ℝ) (x y : ℝ) := y - intersect_point.2 = k * (x - intersect_point.1)
def point_on_circle (circle : ℝ × ℝ → Prop) (x y : ℝ) := circle (x, y)
def distance_squared (A B : ℝ × ℝ) := (A.1 - B.1)^2 + (A.2 - B.2)^2

-- Problem statements
theorem circle_O₁_equation (k : ℝ) (m : ℝ) (x y : ℝ) (h : k = 1) (h_intersect: is_intersection 3 4)
  (h_BP_distance : distance_squared (3, 4) (x, y) = (7 * Real.sqrt 2)^2) : 
  (x - 14)^2 + y^2 = 137 := sorry

theorem sum_of_squares_constant (k m : ℝ) (h : k ≠ 0) (h_perpendicular : line_passing_P (-1/k) 3 4)
  (A B C D : ℝ × ℝ) (h_AB_distance : distance_squared A B = 4 * m^2 / (1 + k^2)) 
  (h_CD_distance : distance_squared C D = 4 * m^2 * k^2 / (1 + k^2)) : 
  distance_squared A B + distance_squared C D = 4 * m^2 := sorry

end circle_O₁_equation_sum_of_squares_constant_l85_85245


namespace geometric_sum_of_first_five_terms_l85_85885

theorem geometric_sum_of_first_five_terms (a_1 l : ℝ)
  (h₁ : ∀ r : ℝ, (2 * l = a_1 * (r - 1) ^ 2)) 
  (h₂ : ∀ (r : ℝ), a_1 * r ^ 3 = 8 * a_1):
  (a_1 + a_1 * (2 : ℝ) + a_1 * (2 : ℝ)^2 + a_1 * (2 : ℝ)^3 + a_1 * (2 : ℝ)^4) = 62 :=
by
  sorry

end geometric_sum_of_first_five_terms_l85_85885


namespace dryer_cost_l85_85086

theorem dryer_cost (W D : ℕ) (h1 : W + D = 600) (h2 : W = 3 * D) : D = 150 :=
by
  sorry

end dryer_cost_l85_85086


namespace percentage_of_september_authors_l85_85767

def total_authors : ℕ := 120
def september_authors : ℕ := 15

theorem percentage_of_september_authors : 
  (september_authors / total_authors : ℚ) * 100 = 12.5 :=
by
  sorry

end percentage_of_september_authors_l85_85767


namespace abs_ineq_solution_set_l85_85562

theorem abs_ineq_solution_set (x : ℝ) : |x + 1| - |x - 5| < 4 ↔ x < 4 :=
sorry

end abs_ineq_solution_set_l85_85562


namespace f_recurrence_l85_85124

noncomputable def f (n : ℕ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem f_recurrence (n : ℕ) : f (n + 1) - f (n - 1) = (3 * Real.sqrt 7 / 14) * f n := 
  sorry

end f_recurrence_l85_85124


namespace sum_of_interior_angles_l85_85135

theorem sum_of_interior_angles (n : ℕ) (h : 180 * (n - 2) = 1800) : 180 * ((n - 3) - 2) = 1260 :=
by
  sorry

end sum_of_interior_angles_l85_85135


namespace charlie_older_than_bobby_by_three_l85_85644

variable (J C B x : ℕ)

def jenny_older_charlie_by_five (J C : ℕ) := J = C + 5
def charlie_age_when_jenny_twice_bobby_age (C x : ℕ) := C + x = 11
def jenny_twice_bobby (J B x : ℕ) := J + x = 2 * (B + x)

theorem charlie_older_than_bobby_by_three
  (h1 : jenny_older_charlie_by_five J C)
  (h2 : charlie_age_when_jenny_twice_bobby_age C x)
  (h3 : jenny_twice_bobby J B x) :
  (C = B + 3) :=
by
  sorry

end charlie_older_than_bobby_by_three_l85_85644


namespace find_x_l85_85497

-- Given condition that x is 11 percent greater than 90
def eleven_percent_greater (x : ℝ) : Prop := x = 90 + (11 / 100) * 90

-- Theorem statement
theorem find_x (x : ℝ) (h: eleven_percent_greater x) : x = 99.9 :=
  sorry

end find_x_l85_85497


namespace find_a_l85_85820

theorem find_a (a : ℝ) : (dist (⟨-2, -1⟩ : ℝ × ℝ) (⟨a, 3⟩ : ℝ × ℝ) = 5) ↔ (a = 1 ∨ a = -5) :=
by
  sorry

end find_a_l85_85820


namespace quadratic_roots_sum_cubes_l85_85020

theorem quadratic_roots_sum_cubes (k : ℚ) (a b : ℚ) 
  (h1 : 4 * a^2 + 5 * a + k = 0) 
  (h2 : 4 * b^2 + 5 * b + k = 0) 
  (h3 : a^3 + b^3 = a + b) :
  k = 9 / 4 :=
by {
  -- Lean code requires the proof, here we use sorry to skip it
  sorry
}

end quadratic_roots_sum_cubes_l85_85020


namespace correct_equation_l85_85574

-- Define conditions as variables in Lean
def cost_price (x : ℝ) : Prop := x > 0
def markup_percentage : ℝ := 0.40
def discount_percentage : ℝ := 0.80
def selling_price : ℝ := 240

-- Define the theorem
theorem correct_equation (x : ℝ) (hx : cost_price x) :
  x * (1 + markup_percentage) * discount_percentage = selling_price :=
by
  sorry

end correct_equation_l85_85574


namespace no_three_natural_numbers_l85_85538

theorem no_three_natural_numbers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
    (h4 : b ∣ a^2 - 1) (h5 : a ∣ c^2 - 1) (h6 : b ∣ c^2 - 1) : false :=
by
  sorry

end no_three_natural_numbers_l85_85538


namespace point_in_third_quadrant_cos_sin_l85_85658

theorem point_in_third_quadrant_cos_sin (P : ℝ × ℝ) (hP : P = (Real.cos (2009 * Real.pi / 180), Real.sin (2009 * Real.pi / 180))) :
  P.1 < 0 ∧ P.2 < 0 :=
by
  sorry

end point_in_third_quadrant_cos_sin_l85_85658


namespace reflection_about_x_axis_l85_85349

theorem reflection_about_x_axis (a : ℝ) : 
  (A : ℝ × ℝ) = (3, a) → (B : ℝ × ℝ) = (3, 4) → A = (3, -4) → a = -4 :=
by
  intros A_eq B_eq reflection_eq
  sorry

end reflection_about_x_axis_l85_85349


namespace width_of_wall_is_two_l85_85968

noncomputable def volume_of_brick : ℝ := 20 * 10 * 7.5 / 10^6 -- Volume in cubic meters
def number_of_bricks : ℕ := 27000
noncomputable def volume_of_wall (width : ℝ) : ℝ := 27 * width * 0.75

theorem width_of_wall_is_two :
  ∃ (W : ℝ), volume_of_wall W = number_of_bricks * volume_of_brick ∧ W = 2 :=
by
  sorry

end width_of_wall_is_two_l85_85968


namespace cos_120_eq_neg_one_half_l85_85967

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l85_85967


namespace minimum_value_l85_85557

theorem minimum_value (x : ℝ) (hx : 0 ≤ x) : ∃ y : ℝ, y = x^2 - 6 * x + 8 ∧ (∀ t : ℝ, 0 ≤ t → y ≤ t^2 - 6 * t + 8) :=
sorry

end minimum_value_l85_85557


namespace jennifer_dogs_l85_85381

theorem jennifer_dogs (D : ℕ) (groom_time_per_dog : ℕ) (groom_days : ℕ) (total_groom_time : ℕ) :
  groom_time_per_dog = 20 →
  groom_days = 30 →
  total_groom_time = 1200 →
  groom_days * (groom_time_per_dog * D) = total_groom_time →
  D = 2 :=
by
  intro h1 h2 h3 h4
  sorry

end jennifer_dogs_l85_85381


namespace evaluate_expression_l85_85329

theorem evaluate_expression : 
  101^3 + 3 * (101^2) * 2 + 3 * 101 * (2^2) + 2^3 = 1092727 := 
by 
  sorry

end evaluate_expression_l85_85329


namespace value_of_expression_l85_85082

variable (x y : ℝ)

theorem value_of_expression (h1 : x + y = 3) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 849 := by sorry

end value_of_expression_l85_85082


namespace problem_solution_l85_85512

noncomputable def root1 : ℝ := (3 + Real.sqrt 105) / 4
noncomputable def root2 : ℝ := (3 - Real.sqrt 105) / 4

theorem problem_solution :
  (∀ x : ℝ, x ≠ -2 → x ≠ -3 → (x^3 - x^2 - 4 * x) / (x^2 + 5 * x + 6) + x = -4
    → x = root1 ∨ x = root2) := 
by
  sorry

end problem_solution_l85_85512


namespace find_range_of_k_l85_85513

-- Define the conditions and the theorem
def is_ellipse (k : ℝ) : Prop :=
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

theorem find_range_of_k :
  {k : ℝ | is_ellipse k} = {k : ℝ | (-3 < k ∧ k < -1/2) ∨ (-1/2 < k ∧ k < 2)} :=
by
  sorry

end find_range_of_k_l85_85513


namespace evaluate_power_l85_85055

theorem evaluate_power :
  (64 : ℝ) = 2^6 →
  64^(3/4 : ℝ) = 16 * Real.sqrt 2 :=
by
  intro h₁
  rw [h₁]
  sorry

end evaluate_power_l85_85055


namespace cone_volume_l85_85875

theorem cone_volume (d h : ℝ) (V : ℝ) (hd : d = 12) (hh : h = 8) :
  V = (1 / 3) * Real.pi * (d / 2) ^ 2 * h → V = 96 * Real.pi :=
by
  rw [hd, hh]
  sorry

end cone_volume_l85_85875


namespace point_P_location_l85_85087

theorem point_P_location (a b : ℝ) : (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) → a^2 + b^2 > 1 :=
by sorry

end point_P_location_l85_85087


namespace evaluate_expression_l85_85738

-- Definitions based on conditions
def a : ℤ := 5
def b : ℤ := -3
def c : ℤ := 2

-- Theorem to be proved: evaluate the expression
theorem evaluate_expression : (3 : ℚ) / (a + b + c) = 3 / 4 := by
  sorry

end evaluate_expression_l85_85738


namespace first_number_in_proportion_is_60_l85_85878

theorem first_number_in_proportion_is_60 : 
  ∀ (x : ℝ), (x / 6 = 2 / 0.19999999999999998) → x = 60 :=
by
  intros x hx
  sorry

end first_number_in_proportion_is_60_l85_85878


namespace dollar_triple_60_l85_85817

-- Define the function $N
def dollar (N : Real) : Real :=
  0.4 * N + 2

-- Proposition proving that $$(($60)) = 6.96
theorem dollar_triple_60 : dollar (dollar (dollar 60)) = 6.96 := by
  sorry

end dollar_triple_60_l85_85817


namespace ratio_of_areas_l85_85065

variable (A B : ℝ)

-- Conditions
def total_area := A + B = 700
def smaller_part_area := B = 315

-- Problem Statement
theorem ratio_of_areas (h_total : total_area A B) (h_small : smaller_part_area B) :
  (A - B) / ((A + B) / 2) = 1 / 5 := by
sorry

end ratio_of_areas_l85_85065


namespace problem_1_problem_2_l85_85624

-- Define the function f(x) = |x + a| + |x|
def f (x : ℝ) (a : ℝ) : ℝ := abs (x + a) + abs x

-- (Ⅰ) Prove that for a = 1, the solution set for f(x) ≥ 2 is (-∞, -1/2] ∪ [3/2, +∞)
theorem problem_1 : 
  ∀ (x : ℝ), f x 1 ≥ 2 ↔ (x ≤ -1/2 ∨ x ≥ 3/2) :=
by
  intro x
  sorry

-- (Ⅱ) Prove that if there exists x ∈ ℝ such that f(x) < 2, then -2 < a < 2
theorem problem_2 :
  (∃ (x : ℝ), f x a < 2) → -2 < a ∧ a < 2 :=
by
  intro h
  sorry

end problem_1_problem_2_l85_85624


namespace possible_values_of_n_l85_85991

-- Definitions for the problem
def side_ab (n : ℕ) := 3 * n + 3
def side_ac (n : ℕ) := 2 * n + 10
def side_bc (n : ℕ) := 2 * n + 16

-- Triangle inequality conditions
def triangle_inequality_1 (n : ℕ) : Prop := side_ab n + side_ac n > side_bc n
def triangle_inequality_2 (n : ℕ) : Prop := side_ab n + side_bc n > side_ac n
def triangle_inequality_3 (n : ℕ) : Prop := side_ac n + side_bc n > side_ab n

-- Angle condition simplified (since the more complex one was invalid)
def angle_condition (n : ℕ) : Prop := side_ac n > side_ab n

-- Combined valid n range
def valid_n_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 12

-- The theorem to prove
theorem possible_values_of_n (n : ℕ) : triangle_inequality_1 n ∧
                                        triangle_inequality_2 n ∧
                                        triangle_inequality_3 n ∧
                                        angle_condition n ↔
                                        valid_n_range n :=
by
  sorry

end possible_values_of_n_l85_85991


namespace fraction_value_unchanged_l85_85510

theorem fraction_value_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / (x + y) = (2 * x) / (2 * (x + y))) :=
by sorry

end fraction_value_unchanged_l85_85510


namespace total_trip_cost_l85_85319

-- Definitions for the problem
def price_per_person : ℕ := 147
def discount : ℕ := 14
def number_of_people : ℕ := 2

-- Statement to prove
theorem total_trip_cost :
  (price_per_person - discount) * number_of_people = 266 :=
by
  sorry

end total_trip_cost_l85_85319


namespace surface_area_of_segmented_part_l85_85292

theorem surface_area_of_segmented_part (h_prism : ∀ (base_height prism_height : ℝ), base_height = 9 ∧ prism_height = 20)
  (isosceles_triangle : ∀ (a b c : ℝ), a = 18 ∧ b = 15 ∧ c = 15 ∧ b = c)
  (midpoints : ∀ (X Y Z : ℝ), X = 9 ∧ Y = 10 ∧ Z = 9) 
  : let triangle_CZX_area := 45
    let triangle_CZY_area := 45
    let triangle_CXY_area := 9
    let triangle_XYZ_area := 9
    (triangle_CZX_area + triangle_CZY_area + triangle_CXY_area + triangle_XYZ_area = 108) :=
sorry

end surface_area_of_segmented_part_l85_85292


namespace range_of_a_l85_85945

-- Definition for set A
def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = -|x| - 2 }

-- Definition for set B
def B (a : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x - a)^2 + y^2 = a^2 }

-- Statement of the problem in Lean
theorem range_of_a (a : ℝ) : (∀ p, p ∈ A → p ∉ B a) → -2 * Real.sqrt 2 - 2 < a ∧ a < 2 * Real.sqrt 2 + 2 := by
  sorry

end range_of_a_l85_85945


namespace simplify_expression_l85_85225

theorem simplify_expression (x : ℝ) (h : x^2 - x - 1 = 0) :
  ( ( (x - 1) / x - (x - 2) / (x + 1) ) / ( (2 * x^2 - x) / (x^2 + 2 * x + 1) ) ) = 1 := 
by
  sorry

end simplify_expression_l85_85225


namespace positive_diff_of_squares_l85_85022

theorem positive_diff_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 10) : a^2 - b^2 = 400 := by
  sorry

end positive_diff_of_squares_l85_85022


namespace right_triangle_properties_l85_85056

theorem right_triangle_properties (a b c : ℕ) (h1 : c = 13) (h2 : a = 5) (h3 : a^2 + b^2 = c^2) :
  ∃ (area perimeter : ℕ), area = 30 ∧ perimeter = 30 ∧ (a < c ∧ b < c) :=
by
  let area := 1 / 2 * a * b
  let perimeter := a + b + c
  have acute_angles : a < c ∧ b < c := by sorry
  exact ⟨area, perimeter, ⟨sorry, sorry, acute_angles⟩⟩

end right_triangle_properties_l85_85056


namespace jaylen_charge_per_yard_l85_85987

def total_cost : ℝ := 250
def number_of_yards : ℝ := 6
def charge_per_yard : ℝ := 41.67

theorem jaylen_charge_per_yard :
  total_cost / number_of_yards = charge_per_yard :=
sorry

end jaylen_charge_per_yard_l85_85987


namespace rational_sum_abs_ratios_l85_85305

theorem rational_sum_abs_ratios (a b c : ℚ) (h : |a * b * c| / (a * b * c) = 1) : (|a| / a + |b| / b + |c| / c = 3) ∨ (|a| / a + |b| / b + |c| / c = -1) := 
sorry

end rational_sum_abs_ratios_l85_85305


namespace min_wins_required_l85_85575

theorem min_wins_required 
  (total_matches initial_matches remaining_matches : ℕ)
  (points_for_win points_for_draw points_for_defeat current_points target_points : ℕ)
  (matches_played_points : ℕ)
  (h_total : total_matches = 20)
  (h_initial : initial_matches = 5)
  (h_remaining : remaining_matches = total_matches - initial_matches)
  (h_win_points : points_for_win = 3)
  (h_draw_points : points_for_draw = 1)
  (h_defeat_points : points_for_defeat = 0)
  (h_current_points : current_points = 8)
  (h_target_points : target_points = 40)
  (h_matches_played_points : matches_played_points = current_points)
  :
  (∃ min_wins : ℕ, min_wins * points_for_win + (remaining_matches - min_wins) * points_for_defeat >= target_points - matches_played_points ∧ min_wins ≤ remaining_matches) ∧
  (∀ other_wins : ℕ, other_wins < min_wins → (other_wins * points_for_win + (remaining_matches - other_wins) * points_for_defeat < target_points - matches_played_points)) :=
sorry

end min_wins_required_l85_85575


namespace general_term_of_c_l85_85924

theorem general_term_of_c (a b : ℕ → ℕ) (c : ℕ → ℕ) : 
  (∀ n, a n = 2 ^ n) →
  (∀ n, b n = 3 * n + 2) →
  (∀ n, ∃ m k, a n = b m ∧ n = 2 * k + 1 → c k = a n) →
  ∀ n, c n = 2 ^ (2 * n + 1) :=
by
  intros ha hb hc n
  have h' := hc n
  sorry

end general_term_of_c_l85_85924


namespace square_mirror_side_length_l85_85229

theorem square_mirror_side_length :
  ∃ (side_length : ℝ),
  let wall_width := 42
  let wall_length := 27.428571428571427
  let wall_area := wall_width * wall_length
  let mirror_area := wall_area / 2
  (side_length * side_length = mirror_area) → side_length = 24 :=
by
  use 24
  intro h
  sorry

end square_mirror_side_length_l85_85229


namespace arithmetic_sequence_ratio_l85_85362

theorem arithmetic_sequence_ratio
  (d : ℕ) (h₀ : d ≠ 0)
  (a : ℕ → ℕ)
  (h₁ : ∀ n, a (n + 1) = a n + d)
  (h₂ : (a 3)^2 = (a 1) * (a 9)) :
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5 / 8 :=
  sorry

end arithmetic_sequence_ratio_l85_85362


namespace age_of_B_l85_85952

variables (A B C : ℝ)

theorem age_of_B :
  (A + B + C) / 3 = 26 →
  (A + C) / 2 = 29 →
  B = 20 :=
by
  intro h1 h2
  sorry

end age_of_B_l85_85952


namespace chess_competition_players_l85_85564

theorem chess_competition_players (J H : ℕ) (total_points : ℕ) (junior_points : ℕ) (high_school_points : ℕ → ℕ)
  (plays : ℕ → ℕ)
  (H_junior_points : junior_points = 8)
  (H_total_points : total_points = (J + H) * (J + H - 1) / 2)
  (H_total_points_contribution : total_points = junior_points + H * high_school_points H)
  (H_even_distribution : ∀ x : ℕ, 0 ≤ x ∧ x ≤ J → high_school_points H = x * (x - 1) / 2)
  (H_H_cases : H = 7 ∨ H = 9 ∨ H = 14) :
  H = 7 ∨ H = 14 :=
by
  have H_cases : H = 7 ∨ H = 14 :=
    by
      sorry
  exact H_cases

end chess_competition_players_l85_85564


namespace product_213_16_l85_85776

theorem product_213_16 :
  (213 * 16 = 3408) :=
by
  have h1 : (0.16 * 2.13 = 0.3408) := by sorry
  sorry

end product_213_16_l85_85776


namespace find_sum_l85_85619

variables (x y : ℝ)

def condition1 : Prop := x^3 - 3 * x^2 + 5 * x = 1
def condition2 : Prop := y^3 - 3 * y^2 + 5 * y = 5

theorem find_sum : condition1 x → condition2 y → x + y = 2 := 
by 
  sorry -- The proof goes here

end find_sum_l85_85619


namespace find_n_l85_85157

theorem find_n (n : ℕ) (k : ℕ) (x : ℝ) (h1 : k = 1) (h2 : x = 180 - 360 / n) (h3 : 1.5 * x = 180 - 360 / (n + 1)) :
    n = 3 :=
by
  -- proof steps will be provided here
  sorry

end find_n_l85_85157


namespace additional_savings_is_300_l85_85727

-- Define constants
def price_per_window : ℕ := 120
def discount_threshold : ℕ := 10
def discount_per_window : ℕ := 10
def free_window_threshold : ℕ := 5

-- Define the number of windows Alice needs
def alice_windows : ℕ := 9

-- Define the number of windows Bob needs
def bob_windows : ℕ := 12

-- Define the function to calculate total cost without discount
def cost_without_discount (n : ℕ) : ℕ := n * price_per_window

-- Define the function to calculate cost with discount
def cost_with_discount (n : ℕ) : ℕ :=
  let full_windows := n - n / free_window_threshold
  let discounted_price := if n > discount_threshold then price_per_window - discount_per_window else price_per_window
  full_windows * discounted_price

-- Define the function to calculate savings when windows are bought separately
def savings_separately : ℕ :=
  (cost_without_discount alice_windows + cost_without_discount bob_windows) 
  - (cost_with_discount alice_windows + cost_with_discount bob_windows)

-- Define the function to calculate savings when windows are bought together
def savings_together : ℕ :=
  let combined_windows := alice_windows + bob_windows
  cost_without_discount combined_windows - cost_with_discount combined_windows

-- Prove that the additional savings when buying together is $300
theorem additional_savings_is_300 : savings_together - savings_separately = 300 := by
  -- missing proof
  sorry

end additional_savings_is_300_l85_85727


namespace factorize_expr_l85_85848

theorem factorize_expr (a b : ℝ) : a^2 - 2 * a * b = a * (a - 2 * b) := 
by 
  sorry

end factorize_expr_l85_85848


namespace Jerry_weekly_earnings_l85_85552

def hours_per_task : ℕ := 2
def pay_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

theorem Jerry_weekly_earnings : pay_per_task * (hours_per_day / hours_per_task) * days_per_week = 1400 :=
by
  -- Carry out the proof here
  sorry

end Jerry_weekly_earnings_l85_85552


namespace rectangular_field_area_l85_85437

theorem rectangular_field_area (a b c : ℕ) (h1 : a = 15) (h2 : c = 17)
  (h3 : a * a + b * b = c * c) : a * b = 120 := by
  sorry

end rectangular_field_area_l85_85437


namespace min_y_value_l85_85120

theorem min_y_value (x : ℝ) : 
  ∃ y : ℝ, y = 4 * x^2 + 8 * x + 12 ∧ ∀ z, (z = 4 * x^2 + 8 * x + 12) → y ≤ z := sorry

end min_y_value_l85_85120


namespace cost_per_pumpkin_pie_l85_85971

theorem cost_per_pumpkin_pie
  (pumpkin_pies : ℕ)
  (cherry_pies : ℕ)
  (cost_cherry_pie : ℕ)
  (total_profit : ℕ)
  (selling_price : ℕ)
  (total_revenue : ℕ)
  (total_cost : ℕ)
  (cost_pumpkin_pie : ℕ)
  (H1 : pumpkin_pies = 10)
  (H2 : cherry_pies = 12)
  (H3 : cost_cherry_pie = 5)
  (H4 : total_profit = 20)
  (H5 : selling_price = 5)
  (H6 : total_revenue = (pumpkin_pies + cherry_pies) * selling_price)
  (H7 : total_cost = total_revenue - total_profit)
  (H8 : total_cost = pumpkin_pies * cost_pumpkin_pie + cherry_pies * cost_cherry_pie) :
  cost_pumpkin_pie = 3 :=
by
  -- Placeholder for proof
  sorry

end cost_per_pumpkin_pie_l85_85971


namespace find_number_l85_85368

theorem find_number (x : ℝ) (h : x / 5 = 70 + x / 6) : x = 2100 := by
  sorry

end find_number_l85_85368


namespace triangle_side_length_c_l85_85993

theorem triangle_side_length_c (a b : ℝ) (α β γ : ℝ) (h_angle_sum : α + β + γ = 180) (h_angle_eq : 3 * α + 2 * β = 180) (h_a : a = 2) (h_b : b = 3) : 
∃ c : ℝ, c = 4 :=
by
  sorry

end triangle_side_length_c_l85_85993


namespace interest_for_20000_l85_85000

-- Definition of simple interest
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

variables (P1 P2 I1 I2 r : ℝ)
-- Given conditions
def h1 := (P1 = 5000)
def h2 := (I1 = 250)
def h3 := (r = I1 / P1)
-- Question condition
def h4 := (P2 = 20000)
def t := 1

theorem interest_for_20000 :
  P1 = 5000 →
  I1 = 250 →
  P2 = 20000 →
  r = I1 / P1 →
  simple_interest P2 r t = 1000 :=
by
  intros
  -- Proof goes here
  sorry

end interest_for_20000_l85_85000


namespace greta_hours_worked_l85_85054

-- Define the problem conditions
def greta_hourly_rate := 12
def lisa_hourly_rate := 15
def lisa_hours_to_equal_greta_earnings := 32
def greta_earnings (hours_worked : ℕ) := greta_hourly_rate * hours_worked
def lisa_earnings := lisa_hourly_rate * lisa_hours_to_equal_greta_earnings

-- Problem statement
theorem greta_hours_worked (G : ℕ) (H : greta_earnings G = lisa_earnings) : G = 40 := by
  sorry

end greta_hours_worked_l85_85054


namespace median_and_mode_l85_85476

theorem median_and_mode (data : List ℝ) (h : data = [6, 7, 4, 7, 5, 2]) :
  ∃ median mode, median = 5.5 ∧ mode = 7 := 
by {
  sorry
}

end median_and_mode_l85_85476


namespace evaluate_expression_l85_85761

-- Definition of the function f
def f (x : ℤ) : ℤ := 3 * x^2 - 5 * x + 8

-- Theorems and lemmas
theorem evaluate_expression : 3 * f 4 + 2 * f (-4) = 260 := by
  sorry

end evaluate_expression_l85_85761


namespace ratio_of_rectangles_l85_85826

noncomputable def rect_ratio (a b c d e f : ℝ) 
  (h1: a / c = 3 / 5) 
  (h2: b / d = 3 / 5) 
  (h3: a / e = 7 / 4) 
  (h4: b / f = 7 / 4) : ℝ :=
  let A_A := a * b
  let A_B := (a * 5 / 3) * (b * 5 / 3)
  let A_C := (a * 4 / 7) * (b * 4 / 7)
  let A_BC := A_B + A_C
  A_A / A_BC

theorem ratio_of_rectangles (a b c d e f : ℝ) 
  (h1: a / c = 3 / 5) 
  (h2: b / d = 3 / 5) 
  (h3: a / e = 7 / 4) 
  (h4: b / f = 7 / 4) : 
  rect_ratio a b c d e f h1 h2 h3 h4 = 441 / 1369 :=
by
  sorry

end ratio_of_rectangles_l85_85826


namespace train_crosses_platform_in_25_002_seconds_l85_85620

noncomputable def time_to_cross_platform 
  (length_train : ℝ) 
  (length_platform : ℝ) 
  (speed_kmph : ℝ) : ℝ := 
  let total_distance := length_train + length_platform
  let speed_mps := speed_kmph * (1000 / 3600)
  total_distance / speed_mps

theorem train_crosses_platform_in_25_002_seconds :
  time_to_cross_platform 225 400.05 90 = 25.002 := by
  sorry

end train_crosses_platform_in_25_002_seconds_l85_85620


namespace ratio_of_dancers_l85_85483

theorem ratio_of_dancers (total_kids total_dancers slow_dance non_slow_dance : ℕ)
  (h1 : total_kids = 140)
  (h2 : slow_dance = 25)
  (h3 : non_slow_dance = 10)
  (h4 : total_dancers = slow_dance + non_slow_dance) :
  (total_dancers : ℚ) / total_kids = 1 / 4 :=
by
  sorry

end ratio_of_dancers_l85_85483


namespace zack_marbles_number_l85_85867

-- Define the conditions as Lean definitions
def zack_initial_marbles (x : ℕ) :=
  (∃ k : ℕ, x = 3 * k + 5) ∧ (3 * 20 + 5 = 65)

-- State the theorem using the conditions
theorem zack_marbles_number : ∃ x : ℕ, zack_initial_marbles x ∧ x = 65 :=
by
  sorry

end zack_marbles_number_l85_85867


namespace quadrilateral_area_l85_85778

theorem quadrilateral_area {ABCQ : ℝ} 
  (side_length : ℝ) 
  (D P E N : ℝ → Prop) 
  (midpoints : ℝ) 
  (W X Y Z : ℝ → Prop) :
  side_length = 4 → 
  (∀ a b : ℝ, D a ∧ P b → a = 1 ∧ b = 1) → 
  (∀ c d : ℝ, E c ∧ N d → c = 1 ∧ d = 1) →
  (∀ w x y z : ℝ, W w ∧ X x ∧ Y y ∧ Z z → w = 0.5 ∧ x = 0.5 ∧ y = 0.5 ∧ z = 0.5) →
  ∃ (area : ℝ), area = 0.25 :=
by
  sorry

end quadrilateral_area_l85_85778


namespace probability_y_eq_2x_l85_85232

/-- Two fair cubic dice each have six faces labeled with the numbers 1, 2, 3, 4, 5, and 6. 
Rolling these dice sequentially, find the probability that the number on the top face 
of the second die (y) is twice the number on the top face of the first die (x). --/
noncomputable def dice_probability : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

theorem probability_y_eq_2x : dice_probability = 1 / 12 :=
  by sorry

end probability_y_eq_2x_l85_85232


namespace cinematic_academy_members_l85_85386

theorem cinematic_academy_members (h1 : ∀ x, x / 4 ≥ 196.25 → x ≥ 785) : 
  ∃ n : ℝ, 1 / 4 * n = 196.25 ∧ n = 785 :=
by
  sorry

end cinematic_academy_members_l85_85386


namespace sin_maximum_value_l85_85884

theorem sin_maximum_value (c : ℝ) :
  (∀ x : ℝ, x = -π/4 → 3 * Real.sin (2 * x + c) = 3) → c = π :=
by
 sorry

end sin_maximum_value_l85_85884


namespace pirate_schooner_problem_l85_85785

theorem pirate_schooner_problem (p : ℕ) (h1 : 10 < p) 
  (h2 : 0.54 * (p - 10) = (54 : ℝ) / 100 * (p - 10)) 
  (h3 : 0.34 * (p - 10) = (34 : ℝ) / 100 * (p - 10)) 
  (h4 : 2 / 3 * p = (2 : ℝ) / 3 * p) : 
  p = 60 := 
sorry

end pirate_schooner_problem_l85_85785


namespace candies_bought_is_18_l85_85289

-- Define the original number of candies
def original_candies : ℕ := 9

-- Define the total number of candies after buying more
def total_candies : ℕ := 27

-- Define the function to calculate the number of candies bought
def candies_bought (o t : ℕ) : ℕ := t - o

-- The main theorem stating that the number of candies bought is 18
theorem candies_bought_is_18 : candies_bought original_candies total_candies = 18 := by
  -- This is where the proof would go
  sorry

end candies_bought_is_18_l85_85289


namespace simplify_and_evaluate_expression_l85_85676

-- Define the condition
def condition (x y : ℝ) := (x - 2) ^ 2 + |y + 1| = 0

-- Define the expression
def expression (x y : ℝ) := 3 * x ^ 2 * y - (2 * x ^ 2 * y - 3 * (2 * x * y - x ^ 2 * y) + 5 * x * y)

-- State the theorem
theorem simplify_and_evaluate_expression (x y : ℝ) (h : condition x y) : expression x y = 6 :=
by
  sorry

end simplify_and_evaluate_expression_l85_85676


namespace problem1_problem2_l85_85428

theorem problem1 (x : ℝ) (a : ℝ) (h : a = 1) (hp : a < x ∧ x < 3 * a) (hq : 2 < x ∧ x < 3) : 2 < x ∧ x < 3 := 
by
  sorry

theorem problem2 (x : ℝ) (a : ℝ) (hp : 0 < a ∧ a < x ∧ x < 3 * a) (hq : 2 < x ∧ x < 3) (hsuff : ∀ (a x : ℝ), (2 < x ∧ x < 3) → a < x ∧ x < 3 * a) : 1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end problem1_problem2_l85_85428


namespace ratio_A_to_B_l85_85398

theorem ratio_A_to_B (A B C : ℕ) (h1 : A + B + C = 406) (h2 : C = 232) (h3 : B = C / 2) : A / gcd A B = 1 ∧ B / gcd A B = 2 := 
by sorry

end ratio_A_to_B_l85_85398


namespace museum_wings_paintings_l85_85208

theorem museum_wings_paintings (P A : ℕ) (h1: P + A = 8) (h2: P = 1 + 2) : P = 3 :=
by
  -- Proof here
  sorry

end museum_wings_paintings_l85_85208


namespace union_complement_eq_l85_85299

open Set

def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1}
def B : Set ℕ := {1, 2}

theorem union_complement_eq : A ∪ (U \ B) = {1, 3} := by
  sorry

end union_complement_eq_l85_85299


namespace digit_D_value_l85_85717

/- The main conditions are:
1. A, B, C, D are digits (0 through 9)
2. Addition equation: AB + CA = D0
3. Subtraction equation: AB - CA = 00
-/

theorem digit_D_value (A B C D : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10) (hD : D < 10)
  (add_eq : 10 * A + B + 10 * C + A = 10 * D + 0)
  (sub_eq : 10 * A + B - (10 * C + A) = 0) :
  D = 1 :=
sorry

end digit_D_value_l85_85717


namespace no_solution_for_inequalities_l85_85064

theorem no_solution_for_inequalities (m : ℝ) :
  (∀ x : ℝ, x - m ≤ 2 * m + 3 ∧ (x - 1) / 2 ≥ m → false) ↔ m < -2 :=
by
  sorry

end no_solution_for_inequalities_l85_85064


namespace initial_avg_income_l85_85905

theorem initial_avg_income (A : ℝ) :
  (4 * A - 990 = 3 * 650) → (A = 735) :=
by
  sorry

end initial_avg_income_l85_85905


namespace intersecting_lines_l85_85520

theorem intersecting_lines (n c : ℝ) 
  (h1 : (15 : ℝ) = n * 5 + 5)
  (h2 : (15 : ℝ) = 4 * 5 + c) : 
  c + n = -3 := 
by
  sorry

end intersecting_lines_l85_85520


namespace factor_x12_minus_729_l85_85068

theorem factor_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^6 + 27) * (x^3 + 3 * Real.sqrt 3) * (x^3 - 3 * Real.sqrt 3) := 
by
  sorry

end factor_x12_minus_729_l85_85068


namespace power_function_not_origin_l85_85568

theorem power_function_not_origin (m : ℝ) 
  (h1 : m^2 - 3 * m + 3 = 1) 
  (h2 : m^2 - m - 2 ≤ 0) : 
  m = 1 ∨ m = 2 :=
sorry

end power_function_not_origin_l85_85568


namespace num_subsets_with_even_is_24_l85_85655

def A : Set ℕ := {1, 2, 3, 4, 5}
def odd_subsets_count : ℕ := 2^3

theorem num_subsets_with_even_is_24 : 
  let total_subsets := 2^5
  total_subsets - odd_subsets_count = 24 := by
  sorry

end num_subsets_with_even_is_24_l85_85655


namespace someone_received_grade_D_or_F_l85_85602

theorem someone_received_grade_D_or_F (m x : ℕ) (hboys : ∃ n : ℕ, n = m + 3) 
  (hgrades_B : ∃ k : ℕ, k = x + 2) (hgrades_C : ∃ l : ℕ, l = 2 * (x + 2)) :
  ∃ p : ℕ, p = 1 ∨ p = 2 :=
by
  sorry

end someone_received_grade_D_or_F_l85_85602


namespace price_per_cake_l85_85073

def number_of_cakes_per_day := 4
def number_of_working_days_per_week := 5
def total_amount_collected := 640
def number_of_weeks := 4

theorem price_per_cake :
  let total_cakes_per_week := number_of_cakes_per_day * number_of_working_days_per_week
  let total_cakes_in_four_weeks := total_cakes_per_week * number_of_weeks
  let price_per_cake := total_amount_collected / total_cakes_in_four_weeks
  price_per_cake = 8 := by
sorry

end price_per_cake_l85_85073


namespace hiker_speed_calculation_l85_85979

theorem hiker_speed_calculation :
  ∃ (h_speed : ℝ),
    let c_speed := 10
    let c_time := 5.0 / 60.0
    let c_wait := 7.5 / 60.0
    let c_distance := c_speed * c_time
    let h_distance := c_distance
    h_distance = h_speed * c_wait ∧ h_speed = 10 * (5 / 7.5) := by
  sorry

end hiker_speed_calculation_l85_85979


namespace chocolate_syrup_amount_l85_85754

theorem chocolate_syrup_amount (x : ℝ) (H1 : 2 * x + 6 = 14) : x = 4 :=
by
  sorry

end chocolate_syrup_amount_l85_85754


namespace inequality_transformation_l85_85585

variable (x y : ℝ)

theorem inequality_transformation (h : x > y) : x - 2 > y - 2 :=
by
  sorry

end inequality_transformation_l85_85585


namespace mirror_full_body_view_l85_85777

theorem mirror_full_body_view (AB MN : ℝ) (h : AB > 0): 
  (MN = 1/2 * AB) ↔
  ∀ (P : ℝ), (0 < P) → (P < AB) → 
    (P < MN + (AB - P)) ∧ (P > AB - MN + P) := 
by
  sorry

end mirror_full_body_view_l85_85777


namespace solution_set_of_inequality_system_l85_85937

theorem solution_set_of_inequality_system (x : ℝ) :
  (4 * x + 1 > 2) ∧ (1 - 2 * x < 7) ↔ (x > 1 / 4) :=
by
  sorry

end solution_set_of_inequality_system_l85_85937


namespace solution_to_diff_eq_l85_85720

def y (x C : ℝ) : ℝ := x^2 + x + C

theorem solution_to_diff_eq (C : ℝ) : ∀ x : ℝ, 
  (dy = (2 * x + 1) * dx) :=
by
  sorry

end solution_to_diff_eq_l85_85720


namespace hotel_fee_original_flat_fee_l85_85657

theorem hotel_fee_original_flat_fee
  (f n : ℝ)
  (H1 : 0.85 * (f + 3 * n) = 210)
  (H2 : f + 6 * n = 400) :
  f = 94.12 :=
by
  -- Sorry is used to indicate that the proof is not provided
  sorry

end hotel_fee_original_flat_fee_l85_85657


namespace find_x_value_l85_85793

theorem find_x_value (x : ℝ) (h : 3 * x + 6 * x + x + 2 * x = 360) : x = 30 :=
by sorry

end find_x_value_l85_85793


namespace find_b_l85_85549

theorem find_b 
  (a b c x : ℝ)
  (h : (3 * x^2 - 4 * x + 5 / 2) * (a * x^2 + b * x + c) 
       = 6 * x^4 - 17 * x^3 + 11 * x^2 - 7 / 2 * x + 5 / 3) 
  (ha : 3 * a = 6) : b = -3 := 
by 
  sorry

end find_b_l85_85549


namespace expr_D_is_diff_of_squares_l85_85403

-- Definitions for the expressions
def expr_A (a b : ℤ) : ℤ := (a + 2 * b) * (-a - 2 * b)
def expr_B (m n : ℤ) : ℤ := (2 * m - 3 * n) * (3 * n - 2 * m)
def expr_C (x y : ℤ) : ℤ := (2 * x - 3 * y) * (3 * x + 2 * y)
def expr_D (a b : ℤ) : ℤ := (a - b) * (-b - a)

-- Theorem stating that Expression D can be calculated using the difference of squares formula
theorem expr_D_is_diff_of_squares (a b : ℤ) : expr_D a b = a^2 - b^2 :=
by sorry

end expr_D_is_diff_of_squares_l85_85403


namespace solve_equations_l85_85486

theorem solve_equations :
  (∃ x1 x2 : ℝ, (x1 = 1 ∧ x2 = 3) ∧ (x1^2 - 4 * x1 + 3 = 0) ∧ (x2^2 - 4 * x2 + 3 = 0)) ∧
  (∃ y1 y2 : ℝ, (y1 = 9 ∧ y2 = 11 / 7) ∧ (4 * (2 * y1 - 5)^2 = (3 * y1 - 1)^2) ∧ (4 * (2 * y2 - 5)^2 = (3 * y2 - 1)^2)) :=
by
  sorry

end solve_equations_l85_85486


namespace zoe_recycled_correctly_l85_85304

-- Let Z be the number of pounds recycled by Zoe
def pounds_by_zoe (total_points : ℕ) (friends_pounds : ℕ) (pounds_per_point : ℕ) : ℕ :=
  total_points * pounds_per_point - friends_pounds

-- Given conditions
def total_points : ℕ := 6
def friends_pounds : ℕ := 23
def pounds_per_point : ℕ := 8

-- Lean statement for the proof problem
theorem zoe_recycled_correctly : pounds_by_zoe total_points friends_pounds pounds_per_point = 25 :=
by
  -- proof to be provided here
  sorry

end zoe_recycled_correctly_l85_85304


namespace minimize_f_at_a_l85_85244

def f (x a : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

theorem minimize_f_at_a (a : ℝ) (h : a = 82 / 43) :
  ∃ x, ∀ y, f x a ≤ f y a :=
sorry

end minimize_f_at_a_l85_85244


namespace percentage_area_covered_by_pentagons_l85_85262

theorem percentage_area_covered_by_pentagons :
  ∀ (a : ℝ), (∃ (large_square_area small_square_area pentagon_area : ℝ),
    large_square_area = 16 * a^2 ∧
    small_square_area = a^2 ∧
    pentagon_area = 10 * small_square_area ∧
    (pentagon_area / large_square_area) * 100 = 62.5) :=
sorry

end percentage_area_covered_by_pentagons_l85_85262


namespace part1_part2_l85_85414

def unitPrices (x : ℕ) (y : ℕ) : Prop :=
  (20 * x = 16 * (y + 20)) ∧ (x = y + 20)

def maxBoxes (a : ℕ) : Prop :=
  ∀ b, (100 * a + 80 * b ≤ 4600) → (a + b = 50)

theorem part1 (x : ℕ) :
  unitPrices x (x - 20) → x = 100 ∧ (x - 20 = 80) :=
by
  sorry

theorem part2 :
  maxBoxes 30 :=
by
  sorry

end part1_part2_l85_85414


namespace weight_of_newcomer_l85_85457

theorem weight_of_newcomer (avg_old W_initial : ℝ) 
  (h_weight_range : 400 ≤ W_initial ∧ W_initial ≤ 420)
  (h_avg_increase : avg_old + 3.5 = (W_initial - 47 + W_new) / 6)
  (h_person_replaced : 47 = 47) :
  W_new = 68 := 
sorry

end weight_of_newcomer_l85_85457


namespace specific_values_exist_l85_85465

def expr_equal_for_specific_values (a b c : ℝ) : Prop :=
  a + b^2 * c = (a^2 + b) * (a + c)

theorem specific_values_exist :
  ∃ a b c : ℝ, expr_equal_for_specific_values a b c :=
sorry

end specific_values_exist_l85_85465


namespace solve_x_squared_plus_y_squared_l85_85003

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l85_85003


namespace largest_number_is_56_l85_85006

-- Definitions based on the conditions
def ratio_three_five_seven (a b c : ℕ) : Prop :=
  3 * c = a ∧ 5 * c = b ∧ 7 * c = c

def difference_is_32 (a c : ℕ) : Prop :=
  c - a = 32

-- Statement of the proof
theorem largest_number_is_56 (a b c : ℕ) (h1 : ratio_three_five_seven a b c) (h2 : difference_is_32 a c) : c = 56 :=
by
  sorry

end largest_number_is_56_l85_85006


namespace find_a4_b4_l85_85636

theorem find_a4_b4 :
  ∃ (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ),
    a₁ * b₁ + a₂ * b₃ = 1 ∧
    a₁ * b₂ + a₂ * b₄ = 0 ∧
    a₃ * b₁ + a₄ * b₃ = 0 ∧
    a₃ * b₂ + a₄ * b₄ = 1 ∧
    a₂ * b₃ = 7 ∧
    a₄ * b₄ = -6 :=
by
  sorry

end find_a4_b4_l85_85636


namespace number_of_teachers_l85_85477

theorem number_of_teachers
    (number_of_students : ℕ)
    (classes_per_student : ℕ)
    (classes_per_teacher : ℕ)
    (students_per_class : ℕ)
    (total_teachers : ℕ)
    (h1 : number_of_students = 2400)
    (h2 : classes_per_student = 5)
    (h3 : classes_per_teacher = 4)
    (h4 : students_per_class = 30)
    (h5 : total_teachers * classes_per_teacher * students_per_class = number_of_students * classes_per_student) :
    total_teachers = 100 :=
by
  sorry

end number_of_teachers_l85_85477


namespace intersection_M_N_l85_85427

-- Definitions of the sets M and N
def M : Set ℤ := {-3, -2, -1}
def N : Set ℤ := { x | -2 < x ∧ x < 3 }

-- The theorem stating that the intersection of M and N is {-1}
theorem intersection_M_N : M ∩ N = {-1} := by
  sorry

end intersection_M_N_l85_85427


namespace percentage_non_honda_red_cars_l85_85467

theorem percentage_non_honda_red_cars 
  (total_cars : ℕ)
  (honda_cars : ℕ)
  (toyota_cars : ℕ)
  (ford_cars : ℕ)
  (other_cars : ℕ)
  (perc_red_honda : ℕ)
  (perc_red_toyota : ℕ)
  (perc_red_ford : ℕ)
  (perc_red_other : ℕ)
  (perc_total_red : ℕ)
  (hyp_total_cars : total_cars = 900)
  (hyp_honda_cars : honda_cars = 500)
  (hyp_toyota_cars : toyota_cars = 200)
  (hyp_ford_cars : ford_cars = 150)
  (hyp_other_cars : other_cars = 50)
  (hyp_perc_red_honda : perc_red_honda = 90)
  (hyp_perc_red_toyota : perc_red_toyota = 75)
  (hyp_perc_red_ford : perc_red_ford = 30)
  (hyp_perc_red_other : perc_red_other = 20)
  (hyp_perc_total_red : perc_total_red = 60) :
  (205 / 400) * 100 = 51.25 := 
by {
  sorry
}

end percentage_non_honda_red_cars_l85_85467


namespace ducks_and_chickens_l85_85762

theorem ducks_and_chickens : 
  (∃ ducks chickens : ℕ, ducks = 7 ∧ chickens = 6 ∧ ducks + chickens = 13) :=
by
  sorry

end ducks_and_chickens_l85_85762


namespace thm1_thm2_thm3_thm4_l85_85799

variables {Point Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Definitions relating lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p q : Plane) : Prop := sorry
def perpendicular_planes (p q : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Theorem 1: This statement is false, so we negate its for proof.
theorem thm1 (h1 : parallel_line_plane m α) (h2 : parallel_line_plane n β) (h3 : parallel_planes α β) :
  ¬ parallel_lines m n :=
sorry

-- Theorem 2: This statement is true, we need to prove it.
theorem thm2 (h1 : perpendicular_line_plane m α) (h2 : perpendicular_line_plane n β) (h3 : perpendicular_planes α β) :
  perpendicular_lines m n :=
sorry

-- Theorem 3: This statement is true, we need to prove it.
theorem thm3 (h1 : perpendicular_line_plane m α) (h2 : parallel_line_plane n β) (h3 : parallel_planes α β) :
  perpendicular_lines m n :=
sorry

-- Theorem 4: This statement is false, so we negate its for proof.
theorem thm4 (h1 : parallel_line_plane m α) (h2 : perpendicular_line_plane n β) (h3 : perpendicular_planes α β) :
  ¬ parallel_lines m n :=
sorry

end thm1_thm2_thm3_thm4_l85_85799


namespace min_dist_sum_l85_85462

theorem min_dist_sum (x y : ℝ) :
  let M := (1, 3)
  let N := (7, 5)
  let P_on_M := (x - 1)^2 + (y - 3)^2 = 1
  let Q_on_N := (x - 7)^2 + (y - 5)^2 = 4
  let A_on_x_axis := y = 0
  ∃ (P Q : ℝ × ℝ), P_on_M ∧ Q_on_N ∧ ∀ A : ℝ × ℝ, A_on_x_axis → (|dist A P| + |dist A Q|) = 7 := 
sorry

end min_dist_sum_l85_85462


namespace compute_product_l85_85865

variables (x1 y1 x2 y2 x3 y3 : ℝ)

def condition1 (x y : ℝ) : Prop :=
  x^3 - 3 * x * y^2 = 2017

def condition2 (x y : ℝ) : Prop :=
  y^3 - 3 * x^2 * y = 2016

theorem compute_product :
  condition1 x1 y1 → condition2 x1 y1 →
  condition1 x2 y2 → condition2 x2 y2 →
  condition1 x3 y3 → condition2 x3 y3 →
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 1008 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end compute_product_l85_85865


namespace train_average_speed_l85_85295

-- Define the variables used in the conditions
variables (D V : ℝ)
-- Condition: Distance D in 50 minutes at average speed V kmph
-- 50 minutes to hours conversion
def condition1 : D = V * (50 / 60) := sorry
-- Condition: Distance D in 40 minutes at speed 60 kmph
-- 40 minutes to hours conversion
def condition2 : D = 60 * (40 / 60) := sorry

-- Claim: Current average speed V
theorem train_average_speed : V = 48 :=
by
  -- Using the conditions to prove the claim
  sorry

end train_average_speed_l85_85295


namespace tan_value_l85_85623

theorem tan_value (α : ℝ) 
  (h : (2 * Real.cos α ^ 2 + Real.cos (π / 2 + 2 * α) - 1) / (Real.sqrt 2 * Real.sin (2 * α + π / 4)) = 4) : 
  Real.tan (2 * α + π / 4) = 1 / 4 :=
by
  sorry

end tan_value_l85_85623


namespace complex_modulus_proof_l85_85998

noncomputable def complex_modulus_example : ℝ := 
  Complex.abs ⟨3/4, -3⟩

theorem complex_modulus_proof : complex_modulus_example = Real.sqrt 153 / 4 := 
by 
  unfold complex_modulus_example
  sorry

end complex_modulus_proof_l85_85998


namespace area_sum_eq_l85_85798

-- Define the conditions given in the problem
variables {A B C P Q R M N : Type*}

-- Define the properties of the points
variables (triangle_ABC : Triangle A B C)
          (point_P : OnSegment P A B)
          (point_Q : OnSegment Q B C)
          (point_R : OnSegment R A C)
          (parallelogram_PQCR : Parallelogram P Q C R)
          (intersection_M : Intersection M (LineSegment AQ) (LineSegment PR))
          (intersection_N : Intersection N (LineSegment BR) (LineSegment PQ))

-- Define the areas of the triangles involved
variables (area_AMP area_BNP area_CQR : ℝ)

-- Define the conditions for the areas of the triangles
variables (h_area_AMP : area_AMP = Area (Triangle A M P))
          (h_area_BNP : area_BNP = Area (Triangle B N P))
          (h_area_CQR : area_CQR = Area (Triangle C Q R))

-- The theorem to be proved
theorem area_sum_eq :
  area_AMP + area_BNP = area_CQR :=
sorry

end area_sum_eq_l85_85798


namespace sufficient_not_necessary_condition_l85_85646

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ (a : ℝ), a = 2 → (-(a) * (a / 4) = -1)) ∧ ∀ (a : ℝ), (-(a) * (a / 4) = -1 → a = 2 ∨ a = -2) :=
by
  sorry

end sufficient_not_necessary_condition_l85_85646


namespace rewrite_sum_l85_85689

theorem rewrite_sum (S_b S : ℕ → ℕ) (n S_1 : ℕ) (a b c : ℕ) :
  b = 4 → (a + b + c) / 3 = 6 →
  S_b n = b * n + (a + b + c) / 3 * (S n - n * S_1) →
  S_b n = 4 * n + 6 * (S n - n * S_1) := by
sorry

end rewrite_sum_l85_85689


namespace inequality_proof_l85_85527

theorem inequality_proof {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hsum : x + y + z = 1) :
    (2 * x^2 / (y + z)) + (2 * y^2 / (z + x)) + (2 * z^2 / (x + y)) ≥ 1 := sorry

end inequality_proof_l85_85527


namespace part1_part2_part3_l85_85357

def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)
def g (x : ℝ) : ℝ := f x - abs (x - 2)

theorem part1 : ∀ x : ℝ, f x ≤ 8 ↔ (-11 ≤ x ∧ x ≤ 5) := by sorry

theorem part2 : ∃ x : ℝ, g x = 5 := by sorry

theorem part3 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 5) : 
  1 / a + 9 / b = 16 / 5 := by sorry

end part1_part2_part3_l85_85357


namespace lives_after_bonus_l85_85808

variable (X Y Z : ℕ)

theorem lives_after_bonus (X Y Z : ℕ) : (X - Y + 3 * Z) = (X - Y + 3 * Z) :=
sorry

end lives_after_bonus_l85_85808


namespace simplified_value_l85_85664

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log (3) / Real.log (20) + 1) + 
  1 / (Real.log (4) / Real.log (15) + 1) + 
  1 / (Real.log (7) / Real.log (12) + 1)

theorem simplified_value : simplify_expression = 2 :=
by {
  sorry
}

end simplified_value_l85_85664


namespace circle_radius_l85_85522

theorem circle_radius (r : ℝ) (hr : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 :=
by 
  sorry

end circle_radius_l85_85522


namespace binomial_7_2_l85_85773

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l85_85773


namespace solve_for_m_l85_85169

theorem solve_for_m (x y m : ℝ) 
  (h1 : 2 * x + y = 3 * m) 
  (h2 : x - 4 * y = -2 * m)
  (h3 : y + 2 * m = 1 + x) :
  m = 3 / 5 := 
by 
  sorry

end solve_for_m_l85_85169


namespace number_of_restaurants_l85_85077

theorem number_of_restaurants
  (total_units : ℕ)
  (residential_units : ℕ)
  (non_residential_units : ℕ)
  (restaurants : ℕ)
  (h1 : total_units = 300)
  (h2 : residential_units = total_units / 2)
  (h3 : non_residential_units = total_units - residential_units)
  (h4 : restaurants = non_residential_units / 2)
  : restaurants = 75 := 
by
  sorry

end number_of_restaurants_l85_85077


namespace sum_of_x_coords_f_eq_3_l85_85677

section
-- Define the piecewise linear function, splits into five segments
def f1 (x : ℝ) : ℝ := 2 * x + 6
def f2 (x : ℝ) : ℝ := -2 * x + 6
def f3 (x : ℝ) : ℝ := 2 * x + 2
def f4 (x : ℝ) : ℝ := -x + 2
def f5 (x : ℝ) : ℝ := 2 * x - 4

-- The sum of x-coordinates where f(x) = 3
noncomputable def x_coords_3_sum : ℝ := -1.5 + 0.5 + 3.5

-- Goal statement
theorem sum_of_x_coords_f_eq_3 : -1.5 + 0.5 + 3.5 = 2.5 := by
  sorry
end

end sum_of_x_coords_f_eq_3_l85_85677


namespace picture_area_l85_85367

theorem picture_area (x y : ℕ) (h1 : 1 < x) (h2 : 1 < y)
  (h3 : (3*x + 3) * (y + 2) = 110) : x * y = 28 :=
by {
  sorry
}

end picture_area_l85_85367


namespace max_gold_coins_l85_85015

theorem max_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 110) : n ≤ 107 :=
by
  sorry

end max_gold_coins_l85_85015


namespace johns_monthly_earnings_l85_85012

variable (work_days : ℕ) (hours_per_day : ℕ) (former_wage : ℝ) (raise_percentage : ℝ) (days_in_month : ℕ)

def johns_earnings (work_days hours_per_day : ℕ) (former_wage raise_percentage : ℝ) (days_in_month : ℕ) : ℝ :=
  let days_worked := days_in_month / 2
  let total_hours := days_worked * hours_per_day
  let raise := former_wage * raise_percentage
  let new_wage := former_wage + raise
  total_hours * new_wage

theorem johns_monthly_earnings (work_days : ℕ := 15) (hours_per_day : ℕ := 12) (former_wage : ℝ := 20) (raise_percentage : ℝ := 0.3) (days_in_month : ℕ := 30) :
  johns_earnings work_days hours_per_day former_wage raise_percentage days_in_month = 4680 :=
by
  sorry

end johns_monthly_earnings_l85_85012


namespace n_n_plus_1_divisible_by_2_l85_85667

theorem n_n_plus_1_divisible_by_2 (n : ℤ) (h1 : 1 ≤ n) (h2 : n ≤ 99) : (n * (n + 1)) % 2 = 0 := 
sorry

end n_n_plus_1_divisible_by_2_l85_85667


namespace last_two_digits_28_l85_85877

theorem last_two_digits_28 (n : ℕ) (h1 : n > 0) (h2 : n % 2 = 1) : 
  (2^(2*n) * (2^(2*n+1) - 1)) % 100 = 28 :=
by
  sorry

end last_two_digits_28_l85_85877


namespace sanjay_homework_fraction_l85_85954

theorem sanjay_homework_fraction (x : ℚ) :
  (2 * x + 1) / 3 + 4 / 15 = 1 ↔ x = 3 / 5 :=
by
  sorry

end sanjay_homework_fraction_l85_85954


namespace wrapping_paper_each_present_l85_85604

theorem wrapping_paper_each_present (total_paper : ℚ) (num_presents : ℕ)
  (h1 : total_paper = 1 / 2) (h2 : num_presents = 5) :
  (total_paper / num_presents = 1 / 10) :=
by
  sorry

end wrapping_paper_each_present_l85_85604


namespace solve_system_l85_85886

variable (x y z : ℝ)

def equation1 : Prop := x^2 + 25 * y + 19 * z = -471
def equation2 : Prop := y^2 + 23 * x + 21 * z = -397
def equation3 : Prop := z^2 + 21 * x + 21 * y = -545

theorem solve_system : equation1 (-22) (-23) (-20) ∧ equation2 (-22) (-23) (-20) ∧ equation3 (-22) (-23) (-20) := by
  sorry

end solve_system_l85_85886


namespace toothpick_sequence_l85_85959

theorem toothpick_sequence (a d n : ℕ) (h1 : a = 6) (h2 : d = 4) (h3 : n = 150) : a + (n - 1) * d = 602 := by
  sorry

end toothpick_sequence_l85_85959


namespace bike_ride_time_good_l85_85906

theorem bike_ride_time_good (x : ℚ) :
  (20 * x + 12 * (8 - x) = 122) → x = 13 / 4 :=
by
  intro h
  sorry

end bike_ride_time_good_l85_85906


namespace eel_jellyfish_ratio_l85_85353

noncomputable def combined_cost : ℝ := 200
noncomputable def eel_cost : ℝ := 180
noncomputable def jellyfish_cost : ℝ := combined_cost - eel_cost

theorem eel_jellyfish_ratio : eel_cost / jellyfish_cost = 9 :=
by
  sorry

end eel_jellyfish_ratio_l85_85353


namespace factors_of_48_multiples_of_8_l85_85233

theorem factors_of_48_multiples_of_8 : 
  ∃ count : ℕ, count = 4 ∧ (∀ d ∈ {d | d ∣ 48 ∧ (∃ k, d = 8 * k)}, true) :=
by {
  sorry  -- This is a placeholder for the actual proof
}

end factors_of_48_multiples_of_8_l85_85233


namespace third_side_length_l85_85383

def is_odd (n : ℕ) := n % 2 = 1

theorem third_side_length (x : ℕ) (h1 : 2 + 5 > x) (h2 : x + 2 > 5) (h3 : is_odd x) : x = 5 :=
by
  sorry

end third_side_length_l85_85383


namespace a6_value_l85_85375

theorem a6_value
  (a : ℕ → ℤ)
  (h1 : a 2 = 3)
  (h2 : a 4 = 15)
  (geo : ∃ q : ℤ, ∀ n : ℕ, n > 0 → a (n + 1) = q^n * (a 1 + 1) - 1):
  a 6 = 63 :=
by
  sorry

end a6_value_l85_85375


namespace child_ticket_cost_is_2_l85_85102

-- Define the conditions
def adult_ticket_cost : ℕ := 5
def total_tickets_sold : ℕ := 85
def total_revenue : ℕ := 275
def adult_tickets_sold : ℕ := 35

-- Define the function to calculate child ticket cost
noncomputable def child_ticket_cost (adult_ticket_cost : ℕ) (total_tickets_sold : ℕ) (total_revenue : ℕ) (adult_tickets_sold : ℕ) : ℕ :=
  let total_adult_revenue := adult_tickets_sold * adult_ticket_cost
  let total_child_revenue := total_revenue - total_adult_revenue
  let child_tickets_sold := total_tickets_sold - adult_tickets_sold
  total_child_revenue / child_tickets_sold

theorem child_ticket_cost_is_2 : child_ticket_cost adult_ticket_cost total_tickets_sold total_revenue adult_tickets_sold = 2 := 
by
  -- This is a placeholder for the actual proof which we can fill in separately.
  sorry

end child_ticket_cost_is_2_l85_85102


namespace number_of_friends_l85_85845

-- Let n be the number of friends
-- Given the conditions:
-- 1. 9 chicken wings initially.
-- 2. 7 more chicken wings cooked.
-- 3. Each friend gets 4 chicken wings.

theorem number_of_friends :
  let initial_wings := 9
  let additional_wings := 7
  let wings_per_friend := 4
  let total_wings := initial_wings + additional_wings
  let n := total_wings / wings_per_friend
  n = 4 :=
by
  sorry

end number_of_friends_l85_85845


namespace train_speed_approx_kmph_l85_85185

noncomputable def length_of_train : ℝ := 150
noncomputable def time_to_cross_pole : ℝ := 4.425875438161669

theorem train_speed_approx_kmph :
  (length_of_train / time_to_cross_pole) * 3.6 = 122.03 :=
by sorry

end train_speed_approx_kmph_l85_85185


namespace find_unknown_rate_of_blankets_l85_85297

theorem find_unknown_rate_of_blankets (x : ℕ) 
  (h1 : 3 * 100 = 300) 
  (h2 : 5 * 150 = 750)
  (h3 : 3 + 5 + 2 = 10) 
  (h4 : 10 * 160 = 1600) 
  (h5 : 300 + 750 + 2 * x = 1600) : 
  x = 275 := 
sorry

end find_unknown_rate_of_blankets_l85_85297


namespace percent_sugar_in_resulting_solution_l85_85874

theorem percent_sugar_in_resulting_solution (W : ℝ) (hW : W > 0) :
  let original_sugar_percent := 22 / 100
  let second_solution_sugar_percent := 74 / 100
  let remaining_original_weight := (3 / 4) * W
  let removed_weight := (1 / 4) * W
  let sugar_from_remaining_original := (original_sugar_percent * remaining_original_weight)
  let sugar_from_added_second_solution := (second_solution_sugar_percent * removed_weight)
  let total_sugar := sugar_from_remaining_original + sugar_from_added_second_solution
  let resulting_sugar_percent := total_sugar / W
  resulting_sugar_percent = 35 / 100 :=
by
  sorry

end percent_sugar_in_resulting_solution_l85_85874


namespace greatest_integer_y_l85_85466

theorem greatest_integer_y (y : ℤ) : abs (3 * y - 4) ≤ 21 → y ≤ 8 :=
by
  sorry

end greatest_integer_y_l85_85466


namespace min_platforms_needed_l85_85811

theorem min_platforms_needed :
  let slabs_7_tons := 120
  let slabs_9_tons := 80
  let weight_7_tons := 7
  let weight_9_tons := 9
  let max_weight_per_platform := 40
  let total_weight := slabs_7_tons * weight_7_tons + slabs_9_tons * weight_9_tons
  let platforms_needed_per_7_tons := slabs_7_tons / 3
  let platforms_needed_per_9_tons := slabs_9_tons / 2
  platforms_needed_per_7_tons = 40 ∧ platforms_needed_per_9_tons = 40 ∧ 3 * platforms_needed_per_7_tons = slabs_7_tons ∧ 2 * platforms_needed_per_9_tons = slabs_9_tons →
  platforms_needed_per_7_tons = 40 ∧ platforms_needed_per_9_tons = 40 :=
by
  sorry

end min_platforms_needed_l85_85811


namespace locomotive_distance_l85_85098

theorem locomotive_distance 
  (speed_train : ℝ) (speed_sound : ℝ) (time_diff : ℝ)
  (h_train : speed_train = 20) 
  (h_sound : speed_sound = 340) 
  (h_time : time_diff = 4) : 
  ∃ x : ℝ, x = 85 := 
by 
  sorry

end locomotive_distance_l85_85098


namespace businesses_can_apply_l85_85771

-- Define conditions
def total_businesses : ℕ := 72
def businesses_fired : ℕ := 36 -- Half of total businesses (72 / 2)
def businesses_quit : ℕ := 24 -- One third of total businesses (72 / 3)

-- Theorem: Number of businesses Brandon can still apply to
theorem businesses_can_apply : (total_businesses - (businesses_fired + businesses_quit)) = 12 := 
by
  sorry

end businesses_can_apply_l85_85771


namespace different_sets_l85_85170

theorem different_sets (a b c : ℤ) (h1 : 0 < a) (h2 : a < c - 1) (h3 : 1 < b) (h4 : b < c)
  (rk : ∀ (k : ℤ), 0 ≤ k ∧ k ≤ a → ∃ (r : ℤ), 0 ≤ r ∧ r < c ∧ k * b % c = r) :
  {r | ∃ k, 0 ≤ k ∧ k ≤ a ∧ r = k * b % c} ≠ {k | 0 ≤ k ∧ k ≤ a} :=
sorry

end different_sets_l85_85170


namespace length_of_AC_l85_85424

-- Define the conditions: lengths and angle
def AB : ℝ := 10
def BC : ℝ := 10
def CD : ℝ := 15
def DA : ℝ := 15
def angle_ADC : ℝ := 120

-- Prove the length of diagonal AC is 15*sqrt(3)
theorem length_of_AC : 
  (CD ^ 2 + DA ^ 2 - 2 * CD * DA * Real.cos (angle_ADC * Real.pi / 180)) = (15 * Real.sqrt 3) ^ 2 :=
by
  sorry

end length_of_AC_l85_85424


namespace roger_current_money_l85_85433

def roger_initial_money : ℕ := 16
def roger_birthday_money : ℕ := 28
def roger_spent_money : ℕ := 25

theorem roger_current_money : roger_initial_money + roger_birthday_money - roger_spent_money = 19 := 
by sorry

end roger_current_money_l85_85433


namespace find_m_l85_85180

def vector_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (m : ℝ) : vector_perpendicular (3, 1) (m, -3) → m = 1 :=
by
  sorry

end find_m_l85_85180


namespace rational_t_l85_85606

variable (A B t : ℚ)

theorem rational_t (A B : ℚ) (hA : A = 2 * t / (1 + t^2)) (hB : B = (1 - t^2) / (1 + t^2)) : ∃ t' : ℚ, t = t' :=
by
  sorry

end rational_t_l85_85606


namespace C_increases_with_n_l85_85654

variables (n e R r : ℝ)
variables (h_pos_e : e > 0) (h_pos_R : R > 0)
variables (h_pos_r : r > 0) (h_R_nr : R > n * r)
noncomputable def C : ℝ := (e * n) / (R - n * r)

theorem C_increases_with_n (h_pos_e : e > 0) (h_pos_R : R > 0)
(h_pos_r : r > 0) (h_R_nr : R > n * r) (hn1 hn2 : ℝ)
(h_inequality : hn1 < hn2) : 
((e*hn1) / (R - hn1*r)) < ((e*hn2) / (R - hn2*r)) :=
by sorry

end C_increases_with_n_l85_85654


namespace single_elimination_games_l85_85268

theorem single_elimination_games (n : ℕ) (h : n = 23) : 
  ∃ g : ℕ, g = n - 1 :=
by
  sorry

end single_elimination_games_l85_85268


namespace simplify_expression_l85_85228

theorem simplify_expression (y : ℝ) :
  (2 * y^6 + 3 * y^5 + y^3 + 15) - (y^6 + 4 * y^5 - 2 * y^4 + 17) = 
  (y^6 - y^5 + 2 * y^4 + y^3 - 2) :=
by 
  sorry

end simplify_expression_l85_85228


namespace anne_cleaning_time_l85_85830

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end anne_cleaning_time_l85_85830


namespace negation_of_proposition_l85_85238

theorem negation_of_proposition :
    (¬ ∃ (x : ℝ), (Real.exp x - x - 1 < 0)) ↔ (∀ (x : ℝ), Real.exp x - x - 1 ≥ 0) :=
by
  sorry

end negation_of_proposition_l85_85238


namespace stratified_sampling_l85_85912

theorem stratified_sampling (teachers male_students female_students total_pop sample_female_students proportion_total n : ℕ)
    (h_teachers : teachers = 200)
    (h_male_students : male_students = 1200)
    (h_female_students : female_students = 1000)
    (h_total_pop : total_pop = teachers + male_students + female_students)
    (h_sample_female_students : sample_female_students = 80)
    (h_proportion_total : proportion_total = female_students / total_pop)
    (h_proportion_equation : sample_female_students = proportion_total * n) :
  n = 192 :=
by
  sorry

end stratified_sampling_l85_85912


namespace minimum_common_perimeter_l85_85947

namespace IsoscelesTriangles

def integer_sided_isosceles_triangles (a b x : ℕ) :=
  2 * a + 10 * x = 2 * b + 8 * x ∧
  5 * Real.sqrt (a^2 - 25 * x^2) = 4 * Real.sqrt (b^2 - 16 * x^2) ∧
  5 * b = 4 * (b + x)

theorem minimum_common_perimeter : ∃ (a b x : ℕ), 
  integer_sided_isosceles_triangles a b x ∧
  2 * a + 10 * x = 192 :=
by
  sorry

end IsoscelesTriangles

end minimum_common_perimeter_l85_85947


namespace arrangements_three_events_l85_85757

theorem arrangements_three_events (volunteers : ℕ) (events : ℕ) (h_vol : volunteers = 5) (h_events : events = 3) : 
  ∃ n : ℕ, n = (events^volunteers - events * 2^volunteers + events * 1^volunteers) ∧ n = 150 := 
by
  sorry

end arrangements_three_events_l85_85757


namespace problem_solving_example_l85_85908

theorem problem_solving_example (α β : ℝ) (h1 : α + β = 3) (h2 : α * β = 1) (h3 : α^2 - 3 * α + 1 = 0) (h4 : β^2 - 3 * β + 1 = 0) :
  7 * α^5 + 8 * β^4 = 1448 :=
sorry

end problem_solving_example_l85_85908


namespace symmetric_point_origin_l85_85949

-- Define the coordinates of point A and the relation of symmetry about the origin
def A : ℝ × ℝ := (2, -1)
def symm_origin (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- Theorem statement: Point B is the symmetric point of A about the origin
theorem symmetric_point_origin : symm_origin A = (-2, 1) :=
  sorry

end symmetric_point_origin_l85_85949


namespace chicken_coop_problem_l85_85932

-- Definitions of conditions
def available_area : ℝ := 240
def area_per_chicken : ℝ := 4
def area_per_chick : ℝ := 2
def max_daily_feed : ℝ := 8000
def feed_per_chicken : ℝ := 160
def feed_per_chick : ℝ := 40

-- Variables representing the number of chickens and chicks
variables (x y : ℕ)

-- Condition expressions
def space_condition (x y : ℕ) : Prop := 
  (2 * x + y = (available_area / area_per_chick))

def feed_condition (x y : ℕ) : Prop := 
  ((4 * x + y) * feed_per_chick <= max_daily_feed / feed_per_chick)

-- Given conditions and queries proof problem
theorem chicken_coop_problem : 
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 20 ∧ y = 80)) 
  ∧
  (¬ ∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 30 ∧ y = 100))
  ∧
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 40 ∧ y = 40))
  ∧
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 0 ∧ y = 120)) :=
by
  sorry  -- The proof will be provided here.


end chicken_coop_problem_l85_85932


namespace find_parallel_lines_l85_85973

open Real

-- Definitions for the problem conditions
def line1 (a x y : ℝ) : Prop := x + 2 * a * y - 1 = 0
def line2 (a x y : ℝ) : Prop := (2 * a - 1) * x - a * y - 1 = 0

-- Definition of when two lines are parallel in ℝ²
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, (l1 x y → ∃ k, ∀ x' y', l2 x' y' → x = k * x' ∧ y = k * y')

-- Main theorem statement
theorem find_parallel_lines:
  ∀ a : ℝ, (parallel (line1 a) (line2 a)) → (a = 0 ∨ a = 1 / 4) :=
by sorry

end find_parallel_lines_l85_85973


namespace average_weight_of_three_l85_85631

theorem average_weight_of_three :
  ∀ A B C : ℝ,
  (A + B) / 2 = 40 →
  (B + C) / 2 = 43 →
  B = 31 →
  (A + B + C) / 3 = 45 :=
by
  intros A B C h1 h2 h3
  sorry

end average_weight_of_three_l85_85631


namespace ticket_cost_per_ride_l85_85690

theorem ticket_cost_per_ride (total_tickets : ℕ) (spent_tickets : ℕ) (rides : ℕ) (remaining_tickets : ℕ) (cost_per_ride : ℕ) 
  (h1 : total_tickets = 79) 
  (h2 : spent_tickets = 23) 
  (h3 : rides = 8) 
  (h4 : remaining_tickets = total_tickets - spent_tickets) 
  (h5 : remaining_tickets / rides = cost_per_ride) 
  : cost_per_ride = 7 := 
sorry

end ticket_cost_per_ride_l85_85690


namespace eval_expression_l85_85011

theorem eval_expression : 8 / 4 - 3^2 - 10 + 5 * 2 = -7 :=
by
  sorry

end eval_expression_l85_85011


namespace find_angle_l85_85669

-- Given definitions:
def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

-- Condition:
def condition (α : ℝ) : Prop :=
  supplement α = 3 * complement α + 10

-- Statement to prove:
theorem find_angle (α : ℝ) (h : condition α) : α = 50 :=
sorry

end find_angle_l85_85669


namespace quad_function_intersects_x_axis_l85_85032

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quad_function_intersects_x_axis (m : ℝ) :
  (discriminant (2 * m) (8 * m + 1) (8 * m) ≥ 0) ↔ (m ≥ -1/16 ∧ m ≠ 0) :=
by
  sorry

end quad_function_intersects_x_axis_l85_85032


namespace ratio_pentagon_rectangle_l85_85753

theorem ratio_pentagon_rectangle (P: ℝ) (a w: ℝ) (h1: 5 * a = P) (h2: 6 * w = P) (h3: P = 75) : a / w = 6 / 5 := 
by 
  -- Proof steps will be provided to conclude this result 
  sorry

end ratio_pentagon_rectangle_l85_85753


namespace second_month_sale_l85_85566

theorem second_month_sale (S : ℝ) :
  (S + 5420 + 6200 + 6350 + 6500 = 30000) → S = 5530 :=
by
  sorry

end second_month_sale_l85_85566


namespace boxes_per_hand_l85_85216

theorem boxes_per_hand (total_people : ℕ) (total_boxes : ℕ) (boxes_per_person : ℕ) (hands_per_person : ℕ) 
  (h1: total_people = 10) (h2: total_boxes = 20) (h3: boxes_per_person = total_boxes / total_people) 
  (h4: hands_per_person = 2) : boxes_per_person / hands_per_person = 1 := 
by
  sorry

end boxes_per_hand_l85_85216


namespace Mr_Kishore_Savings_l85_85941

noncomputable def total_expenses := 
  5000 + 1500 + 4500 + 2500 + 2000 + 6100 + 3500 + 2700

noncomputable def monthly_salary (S : ℝ) := 
  total_expenses + 0.10 * S = S

noncomputable def savings (S : ℝ) := 
  0.10 * S

theorem Mr_Kishore_Savings : 
  ∃ S : ℝ, monthly_salary S ∧ savings S = 3422.22 :=
by
  sorry

end Mr_Kishore_Savings_l85_85941


namespace xiao_cong_math_score_l85_85226

theorem xiao_cong_math_score :
  ∀ (C M E : ℕ),
    (C + M + E) / 3 = 122 → C = 118 → E = 125 → M = 123 :=
by
  intros C M E h1 h2 h3
  sorry

end xiao_cong_math_score_l85_85226


namespace algebraic_expression_value_l85_85930

theorem algebraic_expression_value
  (a b x y : ℤ)
  (h1 : x = a)
  (h2 : y = b)
  (h3 : x - 2 * y = 7) :
  -a + 2 * b + 1 = -6 :=
by
  -- the proof steps are omitted as instructed
  sorry

end algebraic_expression_value_l85_85930


namespace find_denominator_l85_85002

-- Define the conditions given in the problem
variables (p q : ℚ)
variable (denominator : ℚ)

-- Assuming the conditions
variables (h1 : p / q = 4 / 5)
variables (h2 : 11 / 7 + (2 * q - p) / denominator = 2)

-- State the theorem we want to prove
theorem find_denominator : denominator = 14 :=
by
  -- The proof will be constructed later
  sorry

end find_denominator_l85_85002


namespace certain_number_is_50_l85_85076

theorem certain_number_is_50 (x : ℝ) (h : 4 = 0.08 * x) : x = 50 :=
by {
    sorry
}

end certain_number_is_50_l85_85076


namespace average_of_seven_consecutive_l85_85150

variable (a : ℕ) 

def average_of_consecutive_integers (x : ℕ) : ℕ :=
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) / 7

theorem average_of_seven_consecutive (a : ℕ) :
  average_of_consecutive_integers (average_of_consecutive_integers a) = a + 6 :=
by
  sorry

end average_of_seven_consecutive_l85_85150


namespace total_length_of_ropes_l85_85014

theorem total_length_of_ropes 
  (L : ℕ)
  (first_used second_used : ℕ)
  (h1 : first_used = 42) 
  (h2 : second_used = 12) 
  (h3 : (L - second_used) = 4 * (L - first_used)) :
  2 * L = 104 :=
by
  -- We skip the proof for now
  sorry

end total_length_of_ropes_l85_85014


namespace sum_of_consecutive_integers_with_product_812_l85_85429

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l85_85429


namespace find_b_l85_85873

-- Define the curve and the line equations
def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x + 1
def line (k : ℝ) (b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the conditions in the problem
def passes_through_point (a : ℝ) : Prop := curve a 2 = 3
def is_tangent_at_point (a k b : ℝ) : Prop :=
  ∀ x : ℝ, curve a x = 3 → line k b 2 = 3

-- Main theorem statement
theorem find_b (a k b : ℝ) (h1 : passes_through_point a) (h2 : is_tangent_at_point a k b) : b = -15 :=
by sorry

end find_b_l85_85873


namespace Mike_height_l85_85182

theorem Mike_height (h_mark: 5 * 12 + 3 = 63) (h_mark_mike:  63 + 10 = 73) (h_foot: 12 = 12)
: 73 / 12 = 6 ∧ 73 % 12 = 1 := 
sorry

end Mike_height_l85_85182


namespace students_neither_art_nor_music_l85_85928

def total_students := 75
def art_students := 45
def music_students := 50
def both_art_and_music := 30

theorem students_neither_art_nor_music : 
  total_students - (art_students - both_art_and_music + music_students - both_art_and_music + both_art_and_music) = 10 :=
by 
  sorry

end students_neither_art_nor_music_l85_85928


namespace zero_in_interval_l85_85246

noncomputable def f (x : ℝ) := Real.log x - 6 + 2 * x

theorem zero_in_interval : ∃ c ∈ Set.Ioo 2 3, f c = 0 := by
  sorry

end zero_in_interval_l85_85246


namespace sandy_paint_area_l85_85491

-- Define the dimensions of the wall
def wall_height : ℕ := 10
def wall_length : ℕ := 15

-- Define the dimensions of the decorative region
def deco_height : ℕ := 3
def deco_length : ℕ := 5

-- Calculate the areas and prove the required area to paint
theorem sandy_paint_area :
  wall_height * wall_length - deco_height * deco_length = 135 := by
  sorry

end sandy_paint_area_l85_85491


namespace compare_f_values_l85_85138

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem compare_f_values (a : ℝ) (h_pos : 0 < a) :
  (a > 2 * Real.sqrt 2 → f a > f (a / 2) * f (a / 2)) ∧
  (a = 2 * Real.sqrt 2 → f a = f (a / 2) * f (a / 2)) ∧
  (0 < a ∧ a < 2 * Real.sqrt 2 → f a < f (a / 2) * f (a / 2)) :=
by
  sorry

end compare_f_values_l85_85138


namespace highest_more_than_lowest_by_37_5_percent_l85_85094

variables (highest_price lowest_price : ℝ)

theorem highest_more_than_lowest_by_37_5_percent
  (h_highest : highest_price = 22)
  (h_lowest : lowest_price = 16) :
  ((highest_price - lowest_price) / lowest_price) * 100 = 37.5 :=
by
  sorry

end highest_more_than_lowest_by_37_5_percent_l85_85094


namespace find_a_l85_85083

noncomputable def f (a x : ℝ) : ℝ := Real.log (x + 1) / Real.log a

theorem find_a (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) 
  (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f a x ∧ f a x ≤ 1) : a = 2 :=
sorry

end find_a_l85_85083


namespace total_profit_correct_l85_85997

variables (x y : ℝ) -- B's investment and period
variables (B_profit : ℝ) -- profit received by B
variable (A_investment : ℝ) -- A's investment

-- Given conditions
def A_investment_cond := A_investment = 3 * x
def period_cond := 2 * y
def B_profit_given := B_profit = 4500
def total_profit := 7 * B_profit

theorem total_profit_correct :
  (A_investment = 3 * x)
  ∧ (B_profit = 4500)
  ∧ ((6 * x * 2 * y) / (x * y) = 6)
  → total_profit = 31500 :=
by sorry

end total_profit_correct_l85_85997


namespace mike_passing_percentage_l85_85544

theorem mike_passing_percentage :
  ∀ (score shortfall max_marks : ℕ), 
    score = 212 ∧ shortfall = 25 ∧ max_marks = 790 →
    (score + shortfall) / max_marks * 100 = 30 :=
by
  intros score shortfall max_marks h
  have h1 : score = 212 := h.1
  have h2 : shortfall = 25 := h.2.1
  have h3 : max_marks = 790 := h.2.2
  rw [h1, h2, h3]
  sorry

end mike_passing_percentage_l85_85544


namespace yellow_more_than_green_l85_85450

-- Given conditions
def G : ℕ := 90               -- Number of green buttons
def B : ℕ := 85               -- Number of blue buttons
def T : ℕ := 275              -- Total number of buttons
def Y : ℕ := 100              -- Number of yellow buttons (derived from conditions)

-- Mathematically equivalent proof problem
theorem yellow_more_than_green : (90 + 100 + 85 = 275) → (100 - 90 = 10) :=
by sorry

end yellow_more_than_green_l85_85450


namespace find_m_l85_85306

theorem find_m (S : ℕ → ℝ) (a : ℕ → ℝ) (m : ℝ) (hS : ∀ n, S n = m * 2^(n-1) - 3) 
               (ha1 : a 1 = S 1) (han : ∀ n > 1, a n = S n - S (n - 1)) 
               (ratio : ∀ n > 1, a (n+1) / a n = 1/2): 
  m = 6 := 
sorry

end find_m_l85_85306


namespace solve_exp_eq_l85_85394

theorem solve_exp_eq (x : ℝ) (h : Real.sqrt ((1 + Real.sqrt 2)^x) + Real.sqrt ((1 - Real.sqrt 2)^x) = 2) : 
  x = 0 := 
sorry

end solve_exp_eq_l85_85394


namespace determine_a_l85_85088

noncomputable def f (x : ℝ) : ℝ := Real.exp (abs (x - 1)) + 1

theorem determine_a (a : ℝ) (h : f a = 2) : a = 1 :=
by
  sorry

end determine_a_l85_85088


namespace sqrt_of_4_l85_85315

theorem sqrt_of_4 : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_of_4_l85_85315


namespace perpendicular_distance_is_8_cm_l85_85361

theorem perpendicular_distance_is_8_cm :
  ∀ (side_length distance_from_corner cut_angle : ℝ),
    side_length = 100 →
    distance_from_corner = 8 →
    cut_angle = 45 →
    (∃ h : ℝ, h = 8) :=
by
  intros side_length distance_from_corner cut_angle hms d8 a45
  sorry

end perpendicular_distance_is_8_cm_l85_85361


namespace sampling_methods_l85_85266
-- Import the necessary library

-- Definitions for the conditions of the problem:
def NumberOfFamilies := 500
def HighIncomeFamilies := 125
def MiddleIncomeFamilies := 280
def LowIncomeFamilies := 95
def SampleSize := 100

def FemaleStudentAthletes := 12
def NumberToChoose := 3

-- Define the appropriate sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Stating the proof problem in Lean 4
theorem sampling_methods :
  SamplingMethod.Stratified = SamplingMethod.Stratified ∧
  SamplingMethod.SimpleRandom = SamplingMethod.SimpleRandom :=
by
  -- Proof is omitted in this theorem statement
  sorry

end sampling_methods_l85_85266


namespace polygon_problem_l85_85018

theorem polygon_problem 
  (D : ℕ → ℕ) (m x : ℕ) 
  (H1 : ∀ n, D n = n * (n - 3) / 2)
  (H2 : D m = 3 * D (m - 3))
  (H3 : D (m + x) = 7 * D m) :
  m = 9 ∧ x = 12 ∧ (m + x) - m = 12 :=
by {
  -- the proof would go here, skipped as per the instructions.
  sorry
}

end polygon_problem_l85_85018


namespace letters_symmetry_l85_85795

theorem letters_symmetry (people : Fin 20) (sends : Fin 20 → Finset (Fin 20)) (h : ∀ p, (sends p).card = 10) :
  ∃ i j : Fin 20, i ≠ j ∧ j ∈ sends i ∧ i ∈ sends j :=
by
  sorry

end letters_symmetry_l85_85795


namespace ordered_pairs_count_l85_85046

theorem ordered_pairs_count : 
  ∃ n : ℕ, n = 6 ∧ ∀ A B : ℕ, (0 < A ∧ 0 < B) → (A * B = 32 ↔ A = 1 ∧ B = 32 ∨ A = 32 ∧ B = 1 ∨ A = 2 ∧ B = 16 ∨ A = 16 ∧ B = 2 ∨ A = 4 ∧ B = 8 ∨ A = 8 ∧ B = 4) := 
sorry

end ordered_pairs_count_l85_85046


namespace student_weight_loss_l85_85917

theorem student_weight_loss {S R L : ℕ} (h1 : S = 90) (h2 : S + R = 132) (h3 : S - L = 2 * R) : L = 6 := by
  sorry

end student_weight_loss_l85_85917


namespace tangent_line_parabola_l85_85251

theorem tangent_line_parabola (P : ℝ × ℝ) (hP : P = (-1, 0)) :
  ∀ x y : ℝ, (y^2 = 4 * x) ∧ (P = (-1, 0)) → (x + y + 1 = 0) ∨ (x - y + 1 = 0) := by
  sorry

end tangent_line_parabola_l85_85251


namespace correct_operation_l85_85261

theorem correct_operation :
  (∀ a : ℝ, (a^4)^2 ≠ a^6) ∧
  (∀ a b : ℝ, (a - b)^2 ≠ a^2 - ab + b^2) ∧
  (∀ a b : ℝ, 6 * a^2 * b / (2 * a * b) = 3 * a) ∧
  (∀ a : ℝ, a^2 + a^4 ≠ a^6) :=
by {
  sorry
}

end correct_operation_l85_85261


namespace sheep_count_l85_85616

theorem sheep_count (S H : ℕ) (h1 : S / H = 2 / 7) (h2 : H * 230 = 12880) : S = 16 :=
by 
  -- Lean proof goes here
  sorry

end sheep_count_l85_85616


namespace not_equivalent_expression_l85_85752

/--
Let A, B, C, D be expressions defined as follows:
A := 3 * (x + 2)
B := (-9 * x - 18) / -3
C := (1/3) * (3 * x) + (2/3) * 9
D := (1/3) * (9 * x + 18)

Prove that only C is not equivalent to 3 * x + 6.
-/
theorem not_equivalent_expression (x : ℝ) :
  let A := 3 * (x + 2)
  let B := (-9 * x - 18) / -3
  let C := (1/3) * (3 * x) + (2/3) * 9
  let D := (1/3) * (9 * x + 18)
  C ≠ 3 * x + 6 :=
by
  intros A B C D
  sorry

end not_equivalent_expression_l85_85752


namespace common_ratio_of_geometric_sequence_l85_85337

variable (a : ℕ → ℝ) (d : ℝ)
variable (a1 : ℝ) (h_d : d ≠ 0)
variable (h_arith : ∀ n, a (n + 1) = a n + d)

theorem common_ratio_of_geometric_sequence :
  (a 0 = a1) →
  (a 4 = a1 + 4 * d) →
  (a 16 = a1 + 16 * d) →
  (a1 + 4 * d) / a1 = (a1 + 16 * d) / (a1 + 4 * d) →
  (a1 + 16 * d) / (a1 + 4 * d) = 3 :=
by
  sorry

end common_ratio_of_geometric_sequence_l85_85337


namespace determine_gallons_l85_85434

def current_amount : ℝ := 7.75
def desired_total : ℝ := 14.75
def needed_to_add (x : ℝ) : Prop := desired_total = current_amount + x

theorem determine_gallons : needed_to_add 7 :=
by
  sorry

end determine_gallons_l85_85434


namespace max_sin_x_value_l85_85162

theorem max_sin_x_value (x y z : ℝ) (h1 : Real.sin x = Real.cos y) (h2 : Real.sin y = Real.cos z) (h3 : Real.sin z = Real.cos x) : Real.sin x ≤ Real.sqrt 2 / 2 :=
by
  sorry

end max_sin_x_value_l85_85162


namespace most_stable_student_l85_85531

-- Define the variances for the four students
def variance_A (SA2 : ℝ) : Prop := SA2 = 0.15
def variance_B (SB2 : ℝ) : Prop := SB2 = 0.32
def variance_C (SC2 : ℝ) : Prop := SC2 = 0.5
def variance_D (SD2 : ℝ) : Prop := SD2 = 0.25

-- Theorem proving that the most stable student is A
theorem most_stable_student {SA2 SB2 SC2 SD2 : ℝ} 
  (hA : variance_A SA2) 
  (hB : variance_B SB2)
  (hC : variance_C SC2)
  (hD : variance_D SD2) : 
  SA2 < SB2 ∧ SA2 < SC2 ∧ SA2 < SD2 :=
by
  rw [variance_A, variance_B, variance_C, variance_D] at *
  sorry

end most_stable_student_l85_85531


namespace john_pays_total_l85_85493

-- Definitions based on conditions
def total_cans : ℕ := 30
def price_per_can : ℝ := 0.60

-- Main statement to be proven
theorem john_pays_total : (total_cans / 2) * price_per_can = 9 := 
by
  sorry

end john_pays_total_l85_85493


namespace balls_in_boxes_l85_85115

theorem balls_in_boxes :
  let balls := 5
  let boxes := 4
  boxes ^ balls = 1024 :=
by
  sorry

end balls_in_boxes_l85_85115


namespace leo_third_part_time_l85_85397

-- Definitions to represent the conditions
def total_time : ℕ := 120
def first_part_time : ℕ := 25
def second_part_time : ℕ := 2 * first_part_time

-- Proposition to prove
theorem leo_third_part_time :
  total_time - (first_part_time + second_part_time) = 45 :=
by
  sorry

end leo_third_part_time_l85_85397


namespace range_of_m_l85_85174

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 > 0
def B (x : ℝ) (m : ℝ) : Prop := 2 * m - 1 ≤ x ∧ x ≤ m + 3
def subset (B A : ℝ → Prop) : Prop := ∀ x, B x → A x

theorem range_of_m (m : ℝ) : (∀ x, B x m → A x) ↔ (m < -4 ∨ m > 2) :=
by 
  sorry

end range_of_m_l85_85174


namespace sequence_first_equals_last_four_l85_85682

theorem sequence_first_equals_last_four (n : ℕ) (S : ℕ → ℕ) (h_length : ∀ i < n, S i = 0 ∨ S i = 1)
  (h_condition : ∀ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n - 4 → 
    (S i = S j ∧ S (i + 1) = S (j + 1) ∧ S (i + 2) = S (j + 2) ∧ S (i + 3) = S (j + 3) ∧ S (i + 4) = S (j + 4)) → false) :
  S 1 = S (n - 3) ∧ S 2 = S (n - 2) ∧ S 3 = S (n - 1) ∧ S 4 = S n :=
sorry

end sequence_first_equals_last_four_l85_85682


namespace proof_expression_C_equals_negative_one_l85_85643

def A : ℤ := abs (-1)
def B : ℤ := -(-1)
def C : ℤ := -(1^2)
def D : ℤ := (-1)^2

theorem proof_expression_C_equals_negative_one : C = -1 :=
by 
  sorry

end proof_expression_C_equals_negative_one_l85_85643


namespace number_of_friends_l85_85280

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l85_85280


namespace value_of_k_l85_85164

-- Define the conditions of the quartic equation and the product of two roots
variable (a b c d k : ℝ)
variable (hx : (Polynomial.X ^ 4 - 18 * Polynomial.X ^ 3 + k * Polynomial.X ^ 2 + 200 * Polynomial.X - 1984).rootSet ℝ = {a, b, c, d})
variable (hprod_ab : a * b = -32)

-- The statement to prove: the value of k is 86
theorem value_of_k :
  k = 86 :=
by sorry

end value_of_k_l85_85164


namespace probability_of_qualification_l85_85801

-- Define the probability of hitting a target and the number of shots
def probability_hit : ℝ := 0.4
def number_of_shots : ℕ := 3

-- Define the probability of hitting a specific number of targets
noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- Define the event of qualifying by hitting at least 2 targets
noncomputable def probability_qualify (n : ℕ) (p : ℝ) : ℝ :=
  binomial n 2 p + binomial n 3 p

-- The theorem we want to prove
theorem probability_of_qualification : probability_qualify number_of_shots probability_hit = 0.352 :=
  by sorry

end probability_of_qualification_l85_85801


namespace income_of_deceased_member_l85_85533

theorem income_of_deceased_member
  (A B C : ℝ) -- Incomes of the three members
  (h1 : (A + B + C) / 3 = 735)
  (h2 : (A + B) / 2 = 650) :
  C = 905 :=
by
  sorry

end income_of_deceased_member_l85_85533


namespace count_integers_log_condition_l85_85528

theorem count_integers_log_condition :
  (∃! n : ℕ, n = 54 ∧ (∀ x : ℕ, x > 30 ∧ x < 90 ∧ ((x - 30) * (90 - x) < 1000) ↔ (31 <= x ∧ x <= 84))) :=
sorry

end count_integers_log_condition_l85_85528


namespace arithmetic_sequence_general_term_l85_85618

noncomputable def an (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + (n - 1) * d
def bn (a_n : ℤ) : ℚ := (1 / 2)^a_n

theorem arithmetic_sequence_general_term
  (a_n : ℕ → ℤ)
  (b_1 b_2 b_3 : ℚ)
  (a_1 d : ℤ)
  (h_seq : ∀ n, a_n n = a_1 + (n - 1) * d)
  (h_b1 : b_1 = (1 / 2)^(a_n 1))
  (h_b2 : b_2 = (1 / 2)^(a_n 2))
  (h_b3 : b_3 = (1 / 2)^(a_n 3))
  (h_sum : b_1 + b_2 + b_3 = 21 / 8)
  (h_prod : b_1 * b_2 * b_3 = 1 / 8)
  : (∀ n, a_n n = 2 * n - 3) ∨ (∀ n, a_n n = 5 - 2 * n) :=
sorry

end arithmetic_sequence_general_term_l85_85618


namespace find_a8_l85_85192

def seq (a : Nat → Int) := a 1 = -1 ∧ ∀ n, a (n + 1) = a n - 3

theorem find_a8 (a : Nat → Int) (h : seq a) : a 8 = -22 :=
by {
  sorry
}

end find_a8_l85_85192


namespace sequences_zero_at_2_l85_85201

theorem sequences_zero_at_2
  (a b c d : ℕ → ℝ)
  (h1 : ∀ n, a (n+1) = a n + b n)
  (h2 : ∀ n, b (n+1) = b n + c n)
  (h3 : ∀ n, c (n+1) = c n + d n)
  (h4 : ∀ n, d (n+1) = d n + a n)
  (k m : ℕ)
  (hk : 1 ≤ k)
  (hm : 1 ≤ m)
  (h5 : a (k + m) = a m)
  (h6 : b (k + m) = b m)
  (h7 : c (k + m) = c m)
  (h8 : d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 :=
by sorry

end sequences_zero_at_2_l85_85201


namespace inequality_proof_l85_85508

theorem inequality_proof (a b : ℝ) (h_a : a > 0) (h_b : 3 + b = a) : 
  3 / b + 1 / a >= 3 :=
sorry

end inequality_proof_l85_85508


namespace robot_steps_difference_zero_l85_85815

/-- Define the robot's position at second n --/
def robot_position (n : ℕ) : ℤ :=
  let cycle_length := 7
  let cycle_steps := 4 - 3
  let full_cycles := n / cycle_length
  let remainder := n % cycle_length
  full_cycles + if remainder = 0 then 0 else
    if remainder ≤ 4 then remainder else 4 - (remainder - 4)

/-- The main theorem to prove x_2007 - x_2011 = 0 --/
theorem robot_steps_difference_zero : 
  robot_position 2007 - robot_position 2011 = 0 :=
by sorry

end robot_steps_difference_zero_l85_85815


namespace people_own_only_cats_and_dogs_l85_85106

-- Define the given conditions
def total_people : ℕ := 59
def only_dogs : ℕ := 15
def only_cats : ℕ := 10
def cats_dogs_snakes : ℕ := 3
def total_snakes : ℕ := 29

-- Define the proof problem
theorem people_own_only_cats_and_dogs : ∃ x : ℕ, 15 + 10 + x + 3 + (29 - 3) = 59 ∧ x = 5 :=
by {
  sorry
}

end people_own_only_cats_and_dogs_l85_85106


namespace minimal_inverse_presses_l85_85153

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem minimal_inverse_presses (x : ℚ) (h : x = 50) : 
  ∃ n, n = 2 ∧ (reciprocal^[n] x = x) :=
by
  sorry

end minimal_inverse_presses_l85_85153


namespace universal_inequality_l85_85847

theorem universal_inequality (x y : ℝ) : x^2 + y^2 ≥ 2 * x * y := 
by 
  sorry

end universal_inequality_l85_85847


namespace tangent_line_eq_l85_85745

theorem tangent_line_eq (f : ℝ → ℝ) (f' : ℝ → ℝ) (x y : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, f' x = Real.exp x) →
  f 0 = 1 →
  f' 0 = 1 →
  x = 0 →
  y = 1 →
  x - y + 1 = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end tangent_line_eq_l85_85745


namespace find_unique_digit_sets_l85_85833

theorem find_unique_digit_sets (a b c : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c)
 (h4 : 22 * (a + b + c) = 462) :
  (a = 4 ∧ b = 8 ∧ c = 9) ∨ 
  (a = 4 ∧ b = 9 ∧ c = 8) ∨ 
  (a = 8 ∧ b = 4 ∧ c = 9) ∨
  (a = 8 ∧ b = 9 ∧ c = 4) ∨ 
  (a = 9 ∧ b = 4 ∧ c = 8) ∨ 
  (a = 9 ∧ b = 8 ∧ c = 4) ∨
  (a = 5 ∧ b = 7 ∧ c = 9) ∨ 
  (a = 5 ∧ b = 9 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 5 ∧ c = 9) ∨
  (a = 7 ∧ b = 9 ∧ c = 5) ∨ 
  (a = 9 ∧ b = 5 ∧ c = 7) ∨ 
  (a = 9 ∧ b = 7 ∧ c = 5) ∨
  (a = 6 ∧ b = 7 ∧ c = 8) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 6 ∧ c = 8) ∨
  (a = 7 ∧ b = 8 ∧ c = 6) ∨ 
  (a = 8 ∧ b = 6 ∧ c = 7) ∨ 
  (a = 8 ∧ b = 7 ∧ c = 6) :=
sorry

end find_unique_digit_sets_l85_85833


namespace slower_train_speed_l85_85021

theorem slower_train_speed (v : ℝ) (L : ℝ) (faster_speed_km_hr : ℝ) (time_sec : ℝ) (relative_speed : ℝ) 
  (hL : L = 70) (hfaster_speed_km_hr : faster_speed_km_hr = 50)
  (htime_sec : time_sec = 36) (hrelative_speed : relative_speed = (faster_speed_km_hr - v) * (1000 / 3600)) :
  140 = relative_speed * time_sec → v = 36 := 
by
  -- Proof omitted
  sorry

end slower_train_speed_l85_85021


namespace number_of_students_l85_85511

theorem number_of_students (n S : ℕ) 
  (h1 : S = 15 * n) 
  (h2 : (S + 36) / (n + 1) = 16) : 
  n = 20 :=
by 
  sorry

end number_of_students_l85_85511


namespace eval_complex_div_l85_85612

theorem eval_complex_div : 
  (i / (Real.sqrt 7 + 3 * I) = (3 / 16) + (Real.sqrt 7 / 16) * I) := 
by 
  sorry

end eval_complex_div_l85_85612


namespace smallest_prime_divisor_of_3_pow_19_add_11_pow_23_l85_85784

theorem smallest_prime_divisor_of_3_pow_19_add_11_pow_23 :
  ∀ (n : ℕ), Prime n → n ∣ 3^19 + 11^23 → n = 2 :=
by
  sorry

end smallest_prime_divisor_of_3_pow_19_add_11_pow_23_l85_85784


namespace max_cookies_Andy_could_have_eaten_l85_85681

theorem max_cookies_Andy_could_have_eaten (cookies : ℕ) (Andy Alexa : ℕ) 
  (h1 : cookies = 24) 
  (h2 : Alexa = k * Andy) 
  (h3 : k > 0) 
  (h4 : Andy + Alexa = cookies) 
  : Andy ≤ 12 := 
sorry

end max_cookies_Andy_could_have_eaten_l85_85681


namespace matrix_determinant_transformation_l85_85298

theorem matrix_determinant_transformation (p q r s : ℝ) (h : p * s - q * r = -3) :
  (p * (5 * r + 4 * s) - r * (5 * p + 4 * q)) = -12 :=
sorry

end matrix_determinant_transformation_l85_85298


namespace shirt_wallet_ratio_l85_85573

theorem shirt_wallet_ratio
  (F W S : ℕ)
  (hF : F = 30)
  (hW : W = F + 60)
  (h_total : S + W + F = 150) :
  S / W = 1 / 3 := by
  sorry

end shirt_wallet_ratio_l85_85573


namespace no_monochromatic_ap_11_l85_85686

open Function

theorem no_monochromatic_ap_11 :
  ∃ (coloring : ℕ → Fin 4), (∀ a r : ℕ, r > 0 → a + 10 * r ≤ 2014 → ∃ i j : ℕ, (i ≠ j) ∧ (a + i * r < 1 ∨ a + j * r > 2014 ∨ coloring (a + i * r) ≠ coloring (a + j * r))) :=
sorry

end no_monochromatic_ap_11_l85_85686


namespace describe_difference_of_squares_l85_85219

def description_of_a_squared_minus_b_squared : Prop :=
  ∃ (a b : ℝ), (a^2 - b^2) = (a^2 - b^2)

theorem describe_difference_of_squares :
  description_of_a_squared_minus_b_squared :=
by sorry

end describe_difference_of_squares_l85_85219


namespace original_number_is_16_l85_85822

theorem original_number_is_16 (x : ℤ) (h1 : 3 * (2 * x + 5) = 111) : x = 16 :=
by
  sorry

end original_number_is_16_l85_85822


namespace net_distance_from_start_total_distance_driven_fuel_consumption_l85_85970

def driving_distances : List Int := [14, -3, 7, -3, 11, -4, -3, 11, 6, -7, 9]

theorem net_distance_from_start : List.sum driving_distances = 38 := by
  sorry

theorem total_distance_driven : List.sum (List.map Int.natAbs driving_distances) = 78 := by
  sorry

theorem fuel_consumption (fuel_rate : Float) (total_distance : Nat) : total_distance = 78 → total_distance.toFloat * fuel_rate = 7.8 := by
  intros h_total_distance
  rw [h_total_distance]
  norm_num
  sorry

end net_distance_from_start_total_distance_driven_fuel_consumption_l85_85970


namespace sum_distances_between_l85_85583

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2).sqrt

theorem sum_distances_between (A B D : ℝ × ℝ)
  (hB : B = (0, 5))
  (hD : D = (8, 0))
  (hA : A = (20, 0)) :
  21 < distance A D + distance B D ∧ distance A D + distance B D < 22 :=
by
  sorry

end sum_distances_between_l85_85583


namespace triangle_proportion_l85_85317

theorem triangle_proportion (p q r x y : ℝ)
  (h1 : x / q = y / r)
  (h2 : x + y = p) :
  y / r = p / (q + r) := sorry

end triangle_proportion_l85_85317


namespace left_handed_ratio_l85_85074

-- Given the conditions:
-- total number of players
def total_players : ℕ := 70
-- number of throwers who are all right-handed 
def throwers : ℕ := 37 
-- total number of right-handed players
def right_handed : ℕ := 59

-- Define the necessary variables based on the given conditions.
def non_throwers : ℕ := total_players - throwers
def non_throwing_right_handed : ℕ := right_handed - throwers
def left_handed_non_throwers : ℕ := non_throwers - non_throwing_right_handed

-- State the theorem to prove that the ratio of 
-- left-handed non-throwers to the rest of the team (excluding throwers) is 1:3
theorem left_handed_ratio : 
  (left_handed_non_throwers : ℚ) / (non_throwers : ℚ) = 1 / 3 := by
    sorry

end left_handed_ratio_l85_85074


namespace total_kayaks_built_by_April_l85_85211

theorem total_kayaks_built_by_April
    (a : Nat := 9) (r : Nat := 3) (n : Nat := 4) :
    let S := a * (r ^ n - 1) / (r - 1)
    S = 360 := by
  sorry

end total_kayaks_built_by_April_l85_85211


namespace sum_of_first_2015_digits_l85_85580

noncomputable def repeating_decimal : List ℕ := [1, 4, 2, 8, 5, 7]

def sum_first_n_digits (digits : List ℕ) (n : ℕ) : ℕ :=
  let repeat_length := digits.length
  let full_cycles := n / repeat_length
  let remaining_digits := n % repeat_length
  full_cycles * (digits.sum) + (digits.take remaining_digits).sum

theorem sum_of_first_2015_digits :
  sum_first_n_digits repeating_decimal 2015 = 9065 :=
by
  sorry

end sum_of_first_2015_digits_l85_85580


namespace overall_weighted_defective_shipped_percentage_l85_85069

theorem overall_weighted_defective_shipped_percentage
  (defective_A : ℝ := 0.06) (shipped_A : ℝ := 0.04) (prod_A : ℝ := 0.30)
  (defective_B : ℝ := 0.09) (shipped_B : ℝ := 0.06) (prod_B : ℝ := 0.50)
  (defective_C : ℝ := 0.12) (shipped_C : ℝ := 0.07) (prod_C : ℝ := 0.20) :
  prod_A * defective_A * shipped_A + prod_B * defective_B * shipped_B + prod_C * defective_C * shipped_C = 0.00510 :=
by
  sorry

end overall_weighted_defective_shipped_percentage_l85_85069


namespace number_exceeds_fraction_l85_85918

theorem number_exceeds_fraction (x : ℝ) (hx : x = 0.45 * x + 1000) : x = 1818.18 := 
by
  sorry

end number_exceeds_fraction_l85_85918


namespace oliver_first_coupon_redeem_on_friday_l85_85177

-- Definitions of conditions in the problem
def has_coupons (n : ℕ) := n = 8
def uses_coupon_every_9_days (days : ℕ) := days = 9
def is_closed_on_monday (day : ℕ) := day % 7 = 1  -- Assuming 1 represents Monday
def does_not_redeem_on_closed_day (redemption_days : List ℕ) :=
  ∀ day ∈ redemption_days, day % 7 ≠ 1

-- Main theorem statement
theorem oliver_first_coupon_redeem_on_friday : 
  ∃ (first_redeem_day: ℕ), 
  has_coupons 8 ∧ uses_coupon_every_9_days 9 ∧
  is_closed_on_monday 1 ∧ 
  does_not_redeem_on_closed_day [first_redeem_day, first_redeem_day + 9, first_redeem_day + 18, first_redeem_day + 27, first_redeem_day + 36, first_redeem_day + 45, first_redeem_day + 54, first_redeem_day + 63] ∧ 
  first_redeem_day % 7 = 5 := sorry

end oliver_first_coupon_redeem_on_friday_l85_85177


namespace problem_am_hm_l85_85734

open Real

theorem problem_am_hm (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 2) :
  ∃ S : Set ℝ, (∀ s ∈ S, (2 ≤ s)) ∧ (∀ z, (2 ≤ z) → (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 ∧ z = 1/x + 1/y))
  ∧ (S = {z | 2 ≤ z}) := sorry

end problem_am_hm_l85_85734


namespace sin_double_angle_ineq_l85_85892

theorem sin_double_angle_ineq (α : ℝ) (h1 : Real.sin (π - α) = 1 / 3) (h2 : π / 2 ≤ α) (h3 : α ≤ π) :
  Real.sin (2 * α) = - (4 * Real.sqrt 2) / 9 := 
by
  sorry

end sin_double_angle_ineq_l85_85892


namespace exponent_logarithm_simplifies_l85_85662

theorem exponent_logarithm_simplifies :
  (1/2 : ℝ) ^ (Real.log 3 / Real.log 2 - 1) = 2 / 3 :=
by sorry

end exponent_logarithm_simplifies_l85_85662


namespace solution_set_of_x_sq_gt_x_l85_85217

theorem solution_set_of_x_sq_gt_x :
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} := 
sorry

end solution_set_of_x_sq_gt_x_l85_85217


namespace xiao_ming_final_score_correct_l85_85900

/-- Xiao Ming's scores in image, content, and effect are 9, 8, and 8 points, respectively.
    The weights (ratios) for these scores are 3:4:3.
    Prove that Xiao Ming's final competition score is 8.3 points. -/
def xiao_ming_final_score : Prop :=
  let image_score := 9
  let content_score := 8
  let effect_score := 8
  let image_weight := 3
  let content_weight := 4
  let effect_weight := 3
  let total_weight := image_weight + content_weight + effect_weight
  let weighted_score := (image_score * image_weight) + (content_score * content_weight) + (effect_score * effect_weight)
  weighted_score / total_weight = 8.3

theorem xiao_ming_final_score_correct : xiao_ming_final_score := by
  sorry

end xiao_ming_final_score_correct_l85_85900


namespace madeline_has_five_boxes_l85_85419

theorem madeline_has_five_boxes 
    (total_crayons_per_box : ℕ)
    (not_used_fraction1 : ℚ)
    (not_used_fraction2 : ℚ)
    (used_fraction2 : ℚ)
    (total_boxes_not_used : ℚ)
    (total_unused_crayons : ℕ)
    (unused_in_last_box : ℚ)
    (total_boxes : ℕ) :
    total_crayons_per_box = 24 →
    not_used_fraction1 = 5 / 8 →
    not_used_fraction2 = 1 / 3 →
    used_fraction2 = 2 / 3 →
    total_boxes_not_used = 4 →
    total_unused_crayons = 70 →
    total_boxes = 5 :=
by
  -- Insert proof here
  sorry

end madeline_has_five_boxes_l85_85419


namespace find_number_l85_85110

theorem find_number : ∃ x : ℝ, x + 5 * 12 / (180 / 3) = 51 ∧ x = 50 :=
by
  sorry

end find_number_l85_85110


namespace frisbee_sales_total_receipts_l85_85402

theorem frisbee_sales_total_receipts 
  (total_frisbees : ℕ) 
  (price_3_frisbee : ℕ) 
  (price_4_frisbee : ℕ) 
  (sold_3 : ℕ) 
  (sold_4 : ℕ) 
  (total_receipts : ℕ) 
  (h1 : total_frisbees = 60) 
  (h2 : price_3_frisbee = 3)
  (h3 : price_4_frisbee = 4) 
  (h4 : sold_3 + sold_4 = total_frisbees) 
  (h5 : sold_4 ≥ 24)
  (h6 : total_receipts = sold_3 * price_3_frisbee + sold_4 * price_4_frisbee) :
  total_receipts = 204 :=
sorry

end frisbee_sales_total_receipts_l85_85402


namespace hexagon_angle_sum_l85_85680

theorem hexagon_angle_sum 
  (mA mB mC x y : ℝ)
  (hA : mA = 34)
  (hB : mB = 80)
  (hC : mC = 30)
  (hx' : x = 36 - y) : x + y = 36 :=
by
  sorry

end hexagon_angle_sum_l85_85680


namespace yellow_jelly_bean_probability_l85_85781

theorem yellow_jelly_bean_probability :
  let p_red := 0.15
  let p_orange := 0.35
  let p_green := 0.25
  let p_yellow := 1 - (p_red + p_orange + p_green)
  p_yellow = 0.25 := by
    let p_red := 0.15
    let p_orange := 0.35
    let p_green := 0.25
    let p_yellow := 1 - (p_red + p_orange + p_green)
    show p_yellow = 0.25
    sorry

end yellow_jelly_bean_probability_l85_85781


namespace parallelogram_angle_bisector_l85_85114

theorem parallelogram_angle_bisector (a b S Q : ℝ) (α : ℝ) 
  (hS : S = a * b * Real.sin α)
  (hQ : Q = (1 / 2) * (a - b) ^ 2 * Real.sin α) :
  (2 * a * b) / (a - b) ^ 2 = (S + Q + Real.sqrt (Q ^ 2 + 2 * Q * S)) / S :=
by
  sorry

end parallelogram_angle_bisector_l85_85114


namespace proof_problem_l85_85277

-- Given conditions
variables {a b c : ℕ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variables (h4 : a > b) (h5 : a^2 - a * b - a * c + b * c = 7)

-- Statement to prove
theorem proof_problem : a - c = 1 ∨ a - c = 7 :=
sorry

end proof_problem_l85_85277


namespace trigonometric_identity_l85_85839

theorem trigonometric_identity :
  let cos60 := (1 / 2)
  let sin30 := (1 / 2)
  let tan45 := (1 : ℝ)
  4 * cos60 + 8 * sin30 - 5 * tan45 = 1 :=
by
  let cos60 := (1 / 2 : ℝ)
  let sin30 := (1 / 2 : ℝ)
  let tan45 := (1 : ℝ)
  show 4 * cos60 + 8 * sin30 - 5 * tan45 = 1
  sorry

end trigonometric_identity_l85_85839


namespace alex_min_additional_coins_l85_85648

theorem alex_min_additional_coins (n m k : ℕ) (h_n : n = 15) (h_m : m = 120) :
  k = 0 ↔ m = (n * (n + 1)) / 2 :=
by
  sorry

end alex_min_additional_coins_l85_85648


namespace even_not_divisible_by_4_not_sum_of_two_consecutive_odds_l85_85593

theorem even_not_divisible_by_4_not_sum_of_two_consecutive_odds (x n : ℕ) (h₁ : Even x) (h₂ : ¬ ∃ k, x = 4 * k) : x ≠ (2 * n + 1) + (2 * n + 3) := by
  sorry

end even_not_divisible_by_4_not_sum_of_two_consecutive_odds_l85_85593


namespace sum_of_roots_l85_85109

open Real

theorem sum_of_roots (x1 x2 k c : ℝ) (h1 : 4 * x1^2 - k * x1 = c) (h2 : 4 * x2^2 - k * x2 = c) (h3 : x1 ≠ x2) :
  x1 + x2 = k / 4 :=
by
  sorry

end sum_of_roots_l85_85109


namespace second_tap_fills_in_15_hours_l85_85626

theorem second_tap_fills_in_15_hours 
  (r1 r3 : ℝ) 
  (x : ℝ) 
  (H1 : r1 = 1 / 10) 
  (H2 : r3 = 1 / 6) 
  (H3 : r1 + 1 / x + r3 = 1 / 3) : 
  x = 15 :=
sorry

end second_tap_fills_in_15_hours_l85_85626


namespace hotel_room_count_l85_85331

theorem hotel_room_count {total_lamps lamps_per_room : ℕ} (h_total_lamps : total_lamps = 147) (h_lamps_per_room : lamps_per_room = 7) : total_lamps / lamps_per_room = 21 := by
  -- We will insert this placeholder auto-proof, as the actual arithmetic proof isn't the focus.
  sorry

end hotel_room_count_l85_85331


namespace total_monthly_bill_working_from_home_l85_85871

def original_monthly_bill : ℝ := 60
def percentage_increase : ℝ := 0.30

theorem total_monthly_bill_working_from_home :
  original_monthly_bill + (original_monthly_bill * percentage_increase) = 78 := by
  sorry

end total_monthly_bill_working_from_home_l85_85871


namespace cost_of_each_television_l85_85923

-- Define the conditions
def number_of_televisions : Nat := 5
def number_of_figurines : Nat := 10
def cost_per_figurine : Nat := 1
def total_spent : Nat := 260

-- Define the proof problem
theorem cost_of_each_television (T : Nat) :
  (number_of_televisions * T + number_of_figurines * cost_per_figurine = total_spent) → (T = 50) :=
by
  sorry

end cost_of_each_television_l85_85923


namespace find_x9_y9_l85_85432

theorem find_x9_y9 (x y : ℝ) (h1 : x^3 + y^3 = 7) (h2 : x^6 + y^6 = 49) : x^9 + y^9 = 343 :=
by
  sorry

end find_x9_y9_l85_85432


namespace dog_speed_is_16_kmh_l85_85091

variable (man's_speed : ℝ := 4) -- man's speed in km/h
variable (total_path_length : ℝ := 625) -- total path length in meters
variable (remaining_distance : ℝ := 81) -- remaining distance in meters

theorem dog_speed_is_16_kmh :
  let total_path_length_km := total_path_length / 1000
  let remaining_distance_km := remaining_distance / 1000
  let man_covered_distance_km := total_path_length_km - remaining_distance_km
  let time := man_covered_distance_km / man's_speed
  let dog_total_distance_km := 4 * (2 * total_path_length_km)
  let dog_speed := dog_total_distance_km / time
  dog_speed = 16 :=
by
  sorry

end dog_speed_is_16_kmh_l85_85091


namespace arc_length_of_octagon_side_l85_85222

-- Define the conditions
def is_regular_octagon (side_length : ℝ) (angle_subtended : ℝ) := side_length = 5 ∧ angle_subtended = 2 * Real.pi / 8

-- Define the property to be proved
theorem arc_length_of_octagon_side :
  ∀ (side_length : ℝ) (angle_subtended : ℝ), 
    is_regular_octagon side_length angle_subtended →
    (angle_subtended / (2 * Real.pi)) * (2 * Real.pi * side_length) = 5 * Real.pi / 4 :=
by
  intros side_length angle_subtended h
  unfold is_regular_octagon at h
  sorry

end arc_length_of_octagon_side_l85_85222


namespace sufficient_but_not_necessary_condition_l85_85634

def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3
def q (x : ℝ) : Prop := x ≠ 0

theorem sufficient_but_not_necessary_condition (h: ∀ x : ℝ, p x → q x) : (∀ x : ℝ, q x → p x) → false := sorry

end sufficient_but_not_necessary_condition_l85_85634


namespace sale_decrease_by_20_percent_l85_85205

theorem sale_decrease_by_20_percent (P Q : ℝ)
  (h1 : P > 0) (h2 : Q > 0)
  (price_increased : ∀ P', P' = 1.30 * P)
  (revenue_increase : ∀ R, R = P * Q → ∀ R', R' = 1.04 * R)
  (new_revenue : ∀ P' Q' R', P' = 1.30 * P → Q' = Q * (1 - x / 100) → R' = P' * Q' → R' = 1.04 * (P * Q)) :
  1 - (20 / 100) = 0.8 :=
by sorry

end sale_decrease_by_20_percent_l85_85205


namespace three_digit_numbers_condition_l85_85989

theorem three_digit_numbers_condition (a b c : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c = 2 * ((10 * a + b) + (10 * a + c) + (10 * b + a) + (10 * b + c) + (10 * c + a) + (10 * c + b)))
  ↔ (100 * a + 10 * b + c = 132 ∨ 100 * a + 10 * b + c = 264 ∨ 100 * a + 10 * b + c = 396) :=
by
  sorry

end three_digit_numbers_condition_l85_85989


namespace part_one_part_two_i_part_two_ii_l85_85948

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem part_one (a b : ℝ) : 
  f (-a / 2 + 1) a b ≤ f (a^2 + 5 / 4) a b :=
sorry

theorem part_two_i (a b : ℝ) : 
  f 1 a b + f 3 a b - 2 * f 2 a b = 2 :=
sorry

theorem part_two_ii (a b : ℝ) : 
  ¬((|f 1 a b| < 1/2) ∧ (|f 2 a b| < 1/2) ∧ (|f 3 a b| < 1/2)) :=
sorry

end part_one_part_two_i_part_two_ii_l85_85948


namespace simplify_equation_l85_85155

theorem simplify_equation (x : ℝ) : 
  (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) -> 
  (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by 
  sorry

end simplify_equation_l85_85155


namespace daisy_lunch_vs_breakfast_spending_l85_85632

noncomputable def breakfast_cost : ℝ := 2 + 3 + 4 + 3.5 + 1.5
noncomputable def lunch_base_cost : ℝ := 3.5 + 4 + 5.25 + 6 + 1 + 3
noncomputable def service_charge : ℝ := 0.10 * lunch_base_cost
noncomputable def lunch_cost_with_service_charge : ℝ := lunch_base_cost + service_charge
noncomputable def food_tax : ℝ := 0.05 * lunch_cost_with_service_charge
noncomputable def total_lunch_cost : ℝ := lunch_cost_with_service_charge + food_tax
noncomputable def difference : ℝ := total_lunch_cost - breakfast_cost

theorem daisy_lunch_vs_breakfast_spending :
  difference = 12.28 :=
by 
  sorry

end daisy_lunch_vs_breakfast_spending_l85_85632


namespace count_L_shapes_l85_85400

theorem count_L_shapes (m n : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) : 
  ∃ k, k = 4 * (m - 1) * (n - 1) :=
by
  sorry

end count_L_shapes_l85_85400


namespace regular_polygon_sides_l85_85810

-- Define the measure of each exterior angle
def exterior_angle (n : ℕ) (angle : ℝ) : Prop :=
  angle = 40.0

-- Define the sum of exterior angles of any polygon
def sum_exterior_angles (n : ℕ) (total_angle : ℝ) : Prop :=
  total_angle = 360.0

-- Theorem to prove
theorem regular_polygon_sides (n : ℕ) :
  (exterior_angle n 40.0) ∧ (sum_exterior_angles n 360.0) → n = 9 :=
by
  sorry

end regular_polygon_sides_l85_85810


namespace simplify_exponent_l85_85099

variable {x : ℝ} {m n : ℕ}

theorem simplify_exponent (x : ℝ) : (3 * x ^ 5) * (4 * x ^ 3) = 12 * x ^ 8 := by
  sorry

end simplify_exponent_l85_85099


namespace each_persons_final_share_l85_85080

theorem each_persons_final_share
  (total_dining_bill : ℝ)
  (number_of_people : ℕ)
  (tip_percentage : ℝ) :
  total_dining_bill = 211.00 →
  tip_percentage = 0.15 →
  number_of_people = 5 →
  ((total_dining_bill + total_dining_bill * tip_percentage) / number_of_people) = 48.53 :=
by
  intros
  sorry

end each_persons_final_share_l85_85080


namespace minimum_rubles_to_reverse_chips_l85_85389

theorem minimum_rubles_to_reverse_chips (n : ℕ) (h : n = 100)
  (adjacent_cost : ℕ → ℕ → ℕ)
  (free_cost : ℕ → ℕ → Prop)
  (reverse_cost : ℕ) :
  (∀ i j, i + 1 = j → adjacent_cost i j = 1) →
  (∀ i j, i + 5 = j → free_cost i j) →
  reverse_cost = 61 :=
by
  sorry

end minimum_rubles_to_reverse_chips_l85_85389


namespace range_of_x_l85_85809

theorem range_of_x (x a1 a2 y : ℝ) (d r : ℝ) (hx : x ≠ 0) 
  (h_arith : a1 = x + d ∧ a2 = x + 2 * d ∧ y = x + 3 * d)
  (h_geom : b1 = x * r ∧ b2 = x * r^2 ∧ y = x * r^3) : 4 ≤ x :=
by
  -- Assume x ≠ 0 as given and the sequences are arithmetic and geometric
  have hx3d := h_arith.2.2
  have hx3r := h_geom.2.2
  -- Substituting y in both sequences
  simp only [hx3d, hx3r] at *
  -- Solving for d and determining constraints
  sorry

end range_of_x_l85_85809


namespace tricycle_count_l85_85188

theorem tricycle_count
    (total_children : ℕ) (total_wheels : ℕ) (walking_children : ℕ)
    (h1 : total_children - walking_children = 8)
    (h2 : 2 * (total_children - walking_children - (total_wheels - 16) / 3) + 3 * ((total_wheels - 16) / 3) = total_wheels) :
    (total_wheels - 16) / 3 = 8 :=
by
    intros
    sorry

end tricycle_count_l85_85188


namespace custom_operation_example_l85_85707

-- Define the custom operation
def custom_operation (a b : ℕ) : ℕ := a * b + (a - b)

-- State the theorem
theorem custom_operation_example : custom_operation (custom_operation 3 2) 4 = 31 :=
by
  -- the proof will go here, but we skip it for now
  sorry

end custom_operation_example_l85_85707


namespace man_walking_rate_is_12_l85_85199

theorem man_walking_rate_is_12 (M : ℝ) (woman_speed : ℝ) (time_waiting : ℝ) (catch_up_time : ℝ) 
  (woman_speed_eq : woman_speed = 12) (time_waiting_eq : time_waiting = 1 / 6) 
  (catch_up_time_eq : catch_up_time = 1 / 6): 
  (M * catch_up_time = woman_speed * time_waiting) → M = 12 := by
  intro h
  rw [woman_speed_eq, time_waiting_eq, catch_up_time_eq] at h
  sorry

end man_walking_rate_is_12_l85_85199


namespace time_elephants_l85_85638

def total_time := 130
def time_seals := 13
def time_penguins := 8 * time_seals

theorem time_elephants : total_time - (time_seals + time_penguins) = 13 :=
by
  sorry

end time_elephants_l85_85638


namespace ratio_of_r_to_pq_l85_85363

theorem ratio_of_r_to_pq (p q r : ℕ) (h₁ : p + q + r = 7000) (h₂ : r = 2800) :
  r / (p + q) = 2 / 3 :=
by sorry

end ratio_of_r_to_pq_l85_85363


namespace g_neither_even_nor_odd_l85_85721

noncomputable def g (x : ℝ) : ℝ := ⌊x⌋ + 1/2 + Real.sin x

theorem g_neither_even_nor_odd : ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) := sorry

end g_neither_even_nor_odd_l85_85721


namespace second_number_l85_85334

theorem second_number (x : ℕ) (h1 : ∃ k : ℕ, 1428 = 129 * k + 9)
  (h2 : ∃ m : ℕ, x = 129 * m + 13) (h_gcd : ∀ (d : ℕ), d ∣ (1428 - 9 : ℕ) ∧ d ∣ (x - 13 : ℕ) → d ≤ 129) :
  x = 1561 :=
by
  sorry

end second_number_l85_85334


namespace laptop_sticker_price_l85_85440

theorem laptop_sticker_price (x : ℝ) (h1 : 0.8 * x - 120 = y) (h2 : 0.7 * x = z) (h3 : y + 25 = z) : x = 950 :=
sorry

end laptop_sticker_price_l85_85440


namespace fenced_yard_area_l85_85258

theorem fenced_yard_area :
  let yard := 20 * 18
  let cutout1 := 3 * 3
  let cutout2 := 4 * 2
  yard - cutout1 - cutout2 = 343 := by
  let yard := 20 * 18
  let cutout1 := 3 * 3
  let cutout2 := 4 * 2
  have h : yard - cutout1 - cutout2 = 343 := sorry
  exact h

end fenced_yard_area_l85_85258


namespace min_m_plus_n_l85_85498

theorem min_m_plus_n (m n : ℕ) (h₁ : m > n) (h₂ : 4^m + 4^n % 100 = 0) : m + n = 7 :=
sorry

end min_m_plus_n_l85_85498


namespace product_of_roots_l85_85165

-- Define the coefficients of the cubic equation
def a : ℝ := 2
def d : ℝ := 12

-- Define the cubic equation
def cubic_eq (x : ℝ) : ℝ := a * x^3 - 3 * x^2 - 8 * x + d

-- Prove the product of the roots is -6 using Vieta's formulas
theorem product_of_roots : -d / a = -6 := by
  sorry

end product_of_roots_l85_85165


namespace find_n_l85_85264

noncomputable def arithmeticSequenceTerm (a b : ℝ) (n : ℕ) : ℝ :=
  let A := Real.log a
  let B := Real.log b
  6 * B + (n - 1) * 11 * B

theorem find_n 
  (a b : ℝ) 
  (h1 : Real.log (a^2 * b^4) = 2 * Real.log a + 4 * Real.log b)
  (h2 : Real.log (a^6 * b^11) = 6 * Real.log a + 11 * Real.log b)
  (h3 : Real.log (a^12 * b^20) = 12 * Real.log a + 20 * Real.log b) 
  (h_diff : (6 * Real.log a + 11 * Real.log b) - (2 * Real.log a + 4 * Real.log b) = 
            (12 * Real.log a + 20 * Real.log b) - (6 * Real.log a + 11 * Real.log b))
  : ∃ n : ℕ, arithmeticSequenceTerm a b 15 = Real.log (b^n) ∧ n = 160 :=
by
  use 160
  sorry

end find_n_l85_85264


namespace cells_after_one_week_l85_85649

theorem cells_after_one_week : (3 ^ 7) = 2187 :=
by sorry

end cells_after_one_week_l85_85649


namespace find_rate_percent_l85_85404

theorem find_rate_percent
  (P : ℝ) (SI : ℝ) (T : ℝ) (R : ℝ) 
  (hP : P = 1600)
  (hSI : SI = 200)
  (hT : T = 4)
  (hSI_eq : SI = (P * R * T) / 100) :
  R = 3.125 :=
by {
  sorry
}

end find_rate_percent_l85_85404


namespace lasso_success_probability_l85_85553

-- Let p be the probability of successfully placing a lasso in a single throw
def p := 1 / 2

-- Let q be the probability of failure in a single throw
def q := 1 - p

-- Let n be the number of attempts
def n := 4

-- The probability of failing all n times
def probFailAll := q ^ n

-- The probability of succeeding at least once
def probSuccessAtLeastOnce := 1 - probFailAll

-- Theorem statement
theorem lasso_success_probability : probSuccessAtLeastOnce = 15 / 16 := by
  sorry

end lasso_success_probability_l85_85553


namespace sqrt_x_minus_1_meaningful_l85_85819

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end sqrt_x_minus_1_meaningful_l85_85819


namespace S_7_is_28_l85_85146

-- Define the arithmetic sequence and sum of first n terms
def a : ℕ → ℝ := sorry  -- placeholder for arithmetic sequence
def S (n : ℕ) : ℝ := sorry  -- placeholder for the sum of first n terms

-- Given conditions
def a_3 : ℝ := 3
def a_10 : ℝ := 10

-- Define properties of the arithmetic sequence
axiom a_n_property (n : ℕ) : a n = a 1 + (n - 1) * (a 10 - a 3) / (10 - 3)

-- Define the sum of first n terms
axiom sum_property (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Given specific elements of the sequence
axiom a_3_property : a 3 = 3
axiom a_10_property : a 10 = 10

-- The statement to prove
theorem S_7_is_28 : S 7 = 28 :=
sorry

end S_7_is_28_l85_85146


namespace figure_100_squares_l85_85332

theorem figure_100_squares : (∃ f : ℕ → ℕ, f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 ∧ f 100 = 30301) :=
  sorry

end figure_100_squares_l85_85332


namespace simplify_and_evaluate_l85_85504

theorem simplify_and_evaluate (x : ℝ) (h : x = 2 + Real.sqrt 2) :
  (1 - 3 / (x + 1)) / ((x^2 - 4*x + 4) / (x + 1)) = Real.sqrt 2 / 2 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l85_85504


namespace boy_running_time_l85_85737

theorem boy_running_time :
  let side_length := 60
  let speed1 := 9 * 1000 / 3600       -- 9 km/h to m/s
  let speed2 := 6 * 1000 / 3600       -- 6 km/h to m/s
  let speed3 := 8 * 1000 / 3600       -- 8 km/h to m/s
  let speed4 := 7 * 1000 / 3600       -- 7 km/h to m/s
  let hurdle_time := 5 * 3 * 4        -- 3 hurdles per side, 4 sides
  let time1 := side_length / speed1
  let time2 := side_length / speed2
  let time3 := side_length / speed3
  let time4 := side_length / speed4
  let total_time := time1 + time2 + time3 + time4 + hurdle_time
  total_time = 177.86 := by
{
  -- actual proof would be provided here
  sorry
}

end boy_running_time_l85_85737


namespace cake_remaining_after_4_trips_l85_85870

theorem cake_remaining_after_4_trips :
  ∀ (cake_portion_left_after_trip : ℕ → ℚ), 
    cake_portion_left_after_trip 0 = 1 ∧
    (∀ n, cake_portion_left_after_trip (n + 1) = cake_portion_left_after_trip n / 2) →
    cake_portion_left_after_trip 4 = 1 / 16 :=
by
  intros cake_portion_left_after_trip h
  have h0 : cake_portion_left_after_trip 0 = 1 := h.1
  have h1 : ∀ n, cake_portion_left_after_trip (n + 1) = cake_portion_left_after_trip n / 2 := h.2
  sorry

end cake_remaining_after_4_trips_l85_85870


namespace sequence_geometric_proof_l85_85934

theorem sequence_geometric_proof (a : ℕ → ℕ) (h1 : a 1 = 5) (h2 : ∀ n, a (n + 1) = 2 * a n) :
  ∀ n, a n = 5 * 2 ^ (n - 1) :=
by
  sorry

end sequence_geometric_proof_l85_85934


namespace int_solution_exists_l85_85010

theorem int_solution_exists (x y : ℤ) (h : x + y = 5) : x = 2 ∧ y = 3 := 
by
  sorry

end int_solution_exists_l85_85010


namespace point_B_coordinates_l85_85113

/-
Problem Statement:
Given a point A(2, 4) which is symmetric to point B with respect to the origin,
we need to prove the coordinates of point B.
-/

structure Point where
  x : ℝ
  y : ℝ

def symmetric_wrt_origin (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = -A.y

noncomputable def point_A : Point := ⟨2, 4⟩
noncomputable def point_B : Point := ⟨-2, -4⟩

theorem point_B_coordinates : symmetric_wrt_origin point_A point_B :=
  by
    -- Proof is omitted
    sorry

end point_B_coordinates_l85_85113


namespace find_y_l85_85999

theorem find_y (x y : ℤ)
  (h1 : (100 + 200300 + x) / 3 = 250)
  (h2 : (300 + 150100 + x + y) / 4 = 200) :
  y = -4250 :=
sorry

end find_y_l85_85999


namespace units_digit_42_pow_5_add_27_pow_5_l85_85774

theorem units_digit_42_pow_5_add_27_pow_5 :
  (42 ^ 5 + 27 ^ 5) % 10 = 9 :=
by
  sorry

end units_digit_42_pow_5_add_27_pow_5_l85_85774


namespace rectangle_area_l85_85027

-- Define the conditions as hypotheses in Lean 4
variable (x : ℤ)
variable (area : ℤ := 864)
variable (width : ℤ := x - 12)

-- State the theorem to prove the relation between length and area
theorem rectangle_area (h : x * width = area) : x * (x - 12) = 864 :=
by 
  sorry

end rectangle_area_l85_85027


namespace find_four_real_numbers_l85_85586

theorem find_four_real_numbers
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 + x2 * x3 * x4 = 2)
  (h2 : x2 + x1 * x3 * x4 = 2)
  (h3 : x3 + x1 * x2 * x4 = 2)
  (h4 : x4 + x1 * x2 * x3 = 2) :
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) :=
sorry

end find_four_real_numbers_l85_85586


namespace fuel_consumption_l85_85293

open Real

theorem fuel_consumption (initial_fuel : ℝ) (final_fuel : ℝ) (distance_covered : ℝ) (consumption_rate : ℝ) (fuel_left : ℝ) (x : ℝ) :
  initial_fuel = 60 ∧ final_fuel = 50 ∧ distance_covered = 100 ∧ 
  consumption_rate = (initial_fuel - final_fuel) / distance_covered ∧ consumption_rate = 0.1 ∧ 
  fuel_left = initial_fuel - consumption_rate * x ∧ x = 260 →
  fuel_left = 34 :=
by
  sorry

end fuel_consumption_l85_85293


namespace find_x_l85_85846

-- Define the initial point A with coordinates A(x, -2)
def A (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the transformation of moving 5 units up and 3 units to the right to obtain point B
def transform (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 3, p.2 + 5)

-- Define the final point B with coordinates B(1, y)
def B (y : ℝ) : ℝ × ℝ := (1, y)

-- Define the proof problem
theorem find_x (x y : ℝ) (h : transform (A x) = B y) : x = -2 :=
by sorry

end find_x_l85_85846


namespace geometric_progression_x_l85_85145

theorem geometric_progression_x :
  ∃ x : ℝ, (70 + x) ^ 2 = (30 + x) * (150 + x) ∧ x = 10 :=
by sorry

end geometric_progression_x_l85_85145


namespace aarti_bina_work_l85_85910

theorem aarti_bina_work (days_aarti : ℚ) (days_bina : ℚ) (D : ℚ)
  (ha : days_aarti = 5) (hb : days_bina = 8)
  (rate_aarti : 1 / days_aarti = 1/5) 
  (rate_bina : 1 / days_bina = 1/8)
  (combine_rate : (1 / days_aarti) + (1 / days_bina) = 13 / 40) :
  3 / (13 / 40) = 120 / 13 := 
by
  sorry

end aarti_bina_work_l85_85910


namespace goldfish_cost_discrete_points_l85_85978

def goldfish_cost (n : ℕ) : ℝ :=
  0.25 * n + 5

theorem goldfish_cost_discrete_points :
  ∀ n : ℕ, 5 ≤ n ∧ n ≤ 20 → ∃ k : ℕ, goldfish_cost n = goldfish_cost k ∧ 5 ≤ k ∧ k ≤ 20 :=
by sorry

end goldfish_cost_discrete_points_l85_85978


namespace perfect_game_points_l85_85765

theorem perfect_game_points (points_per_game games_played total_points : ℕ) 
  (h1 : points_per_game = 21) 
  (h2 : games_played = 11) 
  (h3 : total_points = points_per_game * games_played) : 
  total_points = 231 := 
by 
  sorry

end perfect_game_points_l85_85765


namespace kitten_length_doubling_l85_85344

theorem kitten_length_doubling (initial_length : ℕ) (week2_length : ℕ) (current_length : ℕ) 
  (h1 : initial_length = 4) 
  (h2 : week2_length = 2 * initial_length) 
  (h3 : current_length = 2 * week2_length) : 
    current_length = 16 := 
by 
  sorry

end kitten_length_doubling_l85_85344


namespace kanul_cash_spending_percentage_l85_85922

theorem kanul_cash_spending_percentage :
  ∀ (spent_raw_materials spent_machinery total_amount spent_cash : ℝ),
    spent_raw_materials = 500 →
    spent_machinery = 400 →
    total_amount = 1000 →
    spent_cash = total_amount - (spent_raw_materials + spent_machinery) →
    (spent_cash / total_amount) * 100 = 10 :=
by
  intros spent_raw_materials spent_machinery total_amount spent_cash
  intro h1 h2 h3 h4
  sorry

end kanul_cash_spending_percentage_l85_85922


namespace remainder_five_n_minus_eleven_l85_85489

theorem remainder_five_n_minus_eleven (n : ℤ) (h : n % 7 = 3) : (5 * n - 11) % 7 = 4 := 
    sorry

end remainder_five_n_minus_eleven_l85_85489


namespace five_million_squared_l85_85748

theorem five_million_squared : (5 * 10^6)^2 = 25 * 10^12 := by
  sorry

end five_million_squared_l85_85748


namespace findYears_l85_85196

def totalInterest (n : ℕ) : ℕ :=
  24 * n + 70 * n

theorem findYears (n : ℕ) : totalInterest n = 350 → n = 4 := 
sorry

end findYears_l85_85196


namespace problem_M_l85_85406

theorem problem_M (M : ℤ) (h : 1989 + 1991 + 1993 + 1995 + 1997 + 1999 + 2001 = 14000 - M) : M = 35 :=
by
  sorry

end problem_M_l85_85406


namespace intersecting_lines_find_m_l85_85939

theorem intersecting_lines_find_m : ∃ m : ℚ, 
  (∃ x y : ℚ, y = 4*x + 2 ∧ y = -3*x - 18 ∧ y = 2*x + m) ↔ m = -26/7 :=
by
  sorry

end intersecting_lines_find_m_l85_85939


namespace dot_product_a_b_equals_neg5_l85_85309

-- Defining vectors and conditions
structure vector2 := (x : ℝ) (y : ℝ)

def a : vector2 := ⟨2, 1⟩
def b (x : ℝ) : vector2 := ⟨x, -1⟩

-- Collinearity condition
def parallel (v w : vector2) : Prop :=
  v.x * w.y = v.y * w.x

-- Dot product definition
def dot_product (v w : vector2) : ℝ :=
  v.x * w.x + v.y * w.y

-- Given condition
theorem dot_product_a_b_equals_neg5 (x : ℝ) (h : parallel a ⟨a.x - x, a.y - (-1)⟩) : dot_product a (b x) = -5 :=
sorry

end dot_product_a_b_equals_neg5_l85_85309


namespace combined_area_is_correct_l85_85836

def tract1_length := 300
def tract1_width  := 500
def tract2_length := 250
def tract2_width  := 630
def tract3_length := 350
def tract3_width  := 450
def tract4_length := 275
def tract4_width  := 600
def tract5_length := 325
def tract5_width  := 520

def area (length width : ℕ) : ℕ := length * width

theorem combined_area_is_correct :
  area tract1_length tract1_width +
  area tract2_length tract2_width +
  area tract3_length tract3_width +
  area tract4_length tract4_width +
  area tract5_length tract5_width = 799000 :=
by
  sorry

end combined_area_is_correct_l85_85836


namespace a_2n_is_square_l85_85474

def a_n (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else a_n (n - 1) + a_n (n - 3) + a_n (n - 4)

theorem a_2n_is_square (n : ℕ) : ∃ k : ℕ, a_n (2 * n) = k * k := by
  sorry

end a_2n_is_square_l85_85474


namespace percent_equivalence_l85_85825

theorem percent_equivalence (y : ℝ) (h : y ≠ 0) : 0.21 * y = 0.21 * y :=
by sorry

end percent_equivalence_l85_85825


namespace range_a_l85_85740

theorem range_a (a : ℝ) (x : ℝ) : 
    (∀ x, (x = 1 → x - a ≥ 1) ∧ (x = -1 → ¬(x - a ≥ 1))) ↔ (-2 < a ∧ a ≤ 0) :=
by
  sorry

end range_a_l85_85740


namespace eval_expression_l85_85024

theorem eval_expression (x : ℝ) (h : x = Real.sqrt 2 + 1) : (x + 1) / (x - 1) = 1 + Real.sqrt 2 := 
by
  sorry

end eval_expression_l85_85024


namespace watermelon_ratio_l85_85794

theorem watermelon_ratio (michael_weight : ℕ) (john_weight : ℕ) (clay_weight : ℕ)
  (h₁ : michael_weight = 8) 
  (h₂ : john_weight = 12) 
  (h₃ : john_weight * 2 = clay_weight) :
  clay_weight / michael_weight = 3 :=
by {
  sorry
}

end watermelon_ratio_l85_85794


namespace mk97_x_eq_one_l85_85704

noncomputable def mk97_initial_number (x : ℝ) : Prop := 
  x ≠ 0 ∧ 4 * (x^2 - x) = 0

theorem mk97_x_eq_one (x : ℝ) (h : mk97_initial_number x) : x = 1 := by
  sorry

end mk97_x_eq_one_l85_85704


namespace set_intersection_complement_l85_85343

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
noncomputable def M : Set ℕ := {2, 3, 4, 5}
noncomputable def N : Set ℕ := {1, 4, 5, 7}

theorem set_intersection_complement :
  M ∩ (U \ N) = {2, 3} :=
by
  sorry

end set_intersection_complement_l85_85343


namespace library_average_visitors_l85_85370

theorem library_average_visitors (V : ℝ) (h1 : (4 * 1000 + 26 * V = 750 * 30)) : V = 18500 / 26 := 
by 
  -- The actual proof is omitted and replaced by sorry.
  sorry

end library_average_visitors_l85_85370


namespace simplify_fraction_l85_85154

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by 
  sorry

end simplify_fraction_l85_85154


namespace repair_cost_l85_85103

theorem repair_cost (C : ℝ) (repair_cost : ℝ) (profit : ℝ) (selling_price : ℝ)
  (h1 : repair_cost = 0.10 * C)
  (h2 : profit = 1100)
  (h3 : selling_price = 1.20 * C)
  (h4 : profit = selling_price - C) :
  repair_cost = 550 :=
by
  sorry

end repair_cost_l85_85103


namespace boundary_of_shadow_of_sphere_l85_85139

theorem boundary_of_shadow_of_sphere (x y : ℝ) :
  let O := (0, 0, 2)
  let P := (1, -2, 3)
  let r := 2
  (∃ T : ℝ × ℝ × ℝ,
    T = (0, -2, 2) ∧
    (∃ g : ℝ → ℝ,
      y = g x ∧
      g x = (x^2 - 2 * x - 11) / 6)) → 
  y = (x^2 - 2 * x - 11) / 6 :=
by
  sorry

end boundary_of_shadow_of_sphere_l85_85139


namespace images_per_memory_card_l85_85840

-- Define the constants based on the conditions given in the problem
def daily_pictures : ℕ := 10
def years : ℕ := 3
def days_per_year : ℕ := 365
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140

-- Define the properties to be proved
theorem images_per_memory_card :
  (years * days_per_year * daily_pictures) / (total_spent / cost_per_card) = 50 :=
by
  sorry

end images_per_memory_card_l85_85840


namespace club_population_after_five_years_l85_85879

noncomputable def a : ℕ → ℕ
| 0     => 18
| (n+1) => 3 * (a n - 5) + 5

theorem club_population_after_five_years : a 5 = 3164 := by
  sorry

end club_population_after_five_years_l85_85879


namespace matt_total_score_l85_85271

-- Definitions from the conditions
def num_2_point_shots : ℕ := 4
def num_3_point_shots : ℕ := 2
def score_per_2_point_shot : ℕ := 2
def score_per_3_point_shot : ℕ := 3

-- Proof statement
theorem matt_total_score : 
  (num_2_point_shots * score_per_2_point_shot) + 
  (num_3_point_shots * score_per_3_point_shot) = 14 := 
by 
  sorry  -- placeholder for the actual proof

end matt_total_score_l85_85271


namespace reciprocal_of_one_is_one_l85_85234

def is_reciprocal (x y : ℝ) : Prop := x * y = 1

theorem reciprocal_of_one_is_one : is_reciprocal 1 1 := 
by
  sorry

end reciprocal_of_one_is_one_l85_85234


namespace rhombus_area_l85_85480

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 5) (h_d2 : d2 = 8) : 
  (1 / 2) * d1 * d2 = 20 :=
by
  sorry

end rhombus_area_l85_85480


namespace least_subtracted_number_l85_85369

theorem least_subtracted_number (a b c d e : ℕ) 
  (h₁ : a = 2590) 
  (h₂ : b = 9) 
  (h₃ : c = 11) 
  (h₄ : d = 13) 
  (h₅ : e = 6) 
  : ∃ (x : ℕ), a - x % b = e ∧ a - x % c = e ∧ a - x % d = e := by
  sorry

end least_subtracted_number_l85_85369


namespace books_loaned_l85_85902

theorem books_loaned (L : ℕ)
  (initial_books : ℕ := 150)
  (end_year_books : ℕ := 100)
  (return_rate : ℝ := 0.60)
  (loan_rate : ℝ := 0.40)
  (returned_books : ℕ := (initial_books - end_year_books)) :
  loan_rate * (L : ℝ) = (returned_books : ℝ) → L = 125 := by
  intro h
  sorry

end books_loaned_l85_85902


namespace base_number_of_equation_l85_85594

theorem base_number_of_equation (y : ℕ) (b : ℕ) (h1 : 16 ^ y = b ^ 14) (h2 : y = 7) : b = 4 := 
by 
  sorry

end base_number_of_equation_l85_85594


namespace matrix_C_power_50_l85_85460

open Matrix

theorem matrix_C_power_50 (C : Matrix (Fin 2) (Fin 2) ℤ) 
  (hC : C = !![3, 2; -8, -5]) : 
  C^50 = !![1, 0; 0, 1] :=
by {
  -- External proof omitted.
  sorry
}

end matrix_C_power_50_l85_85460


namespace total_balloons_l85_85070

theorem total_balloons (A_initial : Nat) (A_additional : Nat) (J_initial : Nat) 
  (h1 : A_initial = 3) (h2 : J_initial = 5) (h3 : A_additional = 2) : 
  A_initial + A_additional + J_initial = 10 := by
  sorry

end total_balloons_l85_85070


namespace transformation_invariant_l85_85284

-- Define the initial and transformed parabolas
def initial_parabola (x : ℝ) : ℝ := 2 * x^2
def transformed_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 3

-- Define the transformation process
def move_right_1 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 1)
def move_up_3 (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 3

-- Concatenate transformations to form the final transformation
def combined_transformation (x : ℝ) : ℝ :=
  move_up_3 (move_right_1 initial_parabola) x

-- Statement to prove
theorem transformation_invariant :
  ∀ x : ℝ, combined_transformation x = transformed_parabola x := 
by {
  sorry
}

end transformation_invariant_l85_85284


namespace minimum_value_l85_85943

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 6) :
    6 ≤ (x^2 + 2*y^2) / (x + y) + (x^2 + 2*z^2) / (x + z) + (y^2 + 2*z^2) / (y + z) :=
by
  sorry

end minimum_value_l85_85943


namespace fraction_equality_l85_85621

noncomputable def x := (4 : ℚ) / 6
noncomputable def y := (8 : ℚ) / 12

theorem fraction_equality : (6 * x + 8 * y) / (48 * x * y) = (7 : ℚ) / 16 := 
by 
  sorry

end fraction_equality_l85_85621


namespace circle_center_tangent_lines_l85_85423

theorem circle_center_tangent_lines 
    (center : ℝ × ℝ)
    (h1 : 3 * center.1 + 4 * center.2 = 10)
    (h2 : center.1 = 3 * center.2) : 
    center = (30 / 13, 10 / 13) := 
by {
  sorry
}

end circle_center_tangent_lines_l85_85423


namespace jerry_age_proof_l85_85933

variable (J : ℝ)

/-- Mickey's age is 4 years less than 400% of Jerry's age. Mickey is 18 years old. Prove that Jerry is 5.5 years old. -/
theorem jerry_age_proof (h : 18 = 4 * J - 4) : J = 5.5 :=
by
  sorry

end jerry_age_proof_l85_85933


namespace flower_seedlings_pots_l85_85392

theorem flower_seedlings_pots (x y z : ℕ) :
  (1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z) →
  (x + y + z = 16) →
  (2 * x + 4 * y + 10 * z = 50) →
  (x = 10 ∨ x = 13) :=
by
  intros h1 h2 h3
  sorry

end flower_seedlings_pots_l85_85392


namespace area_of_rhombus_l85_85472

theorem area_of_rhombus (x y : ℝ) (d1 d2 : ℝ) (hx : x^2 + y^2 = 130) (hy : d1 = 2 * x) (hz : d2 = 2 * y) (h_diff : abs (d1 - d2) = 4) : 
  4 * 0.5 * x * y = 126 :=
by
  sorry

end area_of_rhombus_l85_85472


namespace daria_weeks_needed_l85_85523

-- Defining the parameters and conditions
def initial_amount : ℕ := 20
def weekly_savings : ℕ := 10
def cost_of_vacuum_cleaner : ℕ := 120

-- Defining the total money Daria needs to add to her initial amount
def additional_amount_needed : ℕ := cost_of_vacuum_cleaner - initial_amount

-- Defining the number of weeks needed to save the additional amount, given weekly savings
def weeks_needed : ℕ := additional_amount_needed / weekly_savings

-- The theorem stating that Daria needs exactly 10 weeks to cover the expense of the vacuum cleaner
theorem daria_weeks_needed : weeks_needed = 10 := by
  sorry

end daria_weeks_needed_l85_85523


namespace barbara_needs_more_weeks_l85_85603

/-
  Problem Statement:
  Barbara wants to save up for a new wristwatch that costs $100. Her parents give her an allowance
  of $5 a week and she can either save it all up or spend it as she wishes. 10 weeks pass and
  due to spending some of her money, Barbara currently only has $20. How many more weeks does she need
  to save for a watch if she stops spending on other things right now?
-/

def wristwatch_cost : ℕ := 100
def allowance_per_week : ℕ := 5
def current_savings : ℕ := 20
def amount_needed : ℕ := wristwatch_cost - current_savings
def weeks_needed : ℕ := amount_needed / allowance_per_week

theorem barbara_needs_more_weeks :
  weeks_needed = 16 :=
by
  -- proof goes here
  sorry

end barbara_needs_more_weeks_l85_85603


namespace Lisa_types_correctly_l85_85718

-- Given conditions
def Rudy_wpm : ℕ := 64
def Joyce_wpm : ℕ := 76
def Gladys_wpm : ℕ := 91
def Mike_wpm : ℕ := 89
def avg_wpm : ℕ := 80
def num_employees : ℕ := 5

-- Define the hypothesis about Lisa's typing speaking
def Lisa_wpm : ℕ := (num_employees * avg_wpm) - Rudy_wpm - Joyce_wpm - Gladys_wpm - Mike_wpm

-- The statement to prove
theorem Lisa_types_correctly :
  Lisa_wpm = 140 := by
  sorry

end Lisa_types_correctly_l85_85718


namespace repeating_decimal_to_fraction_l85_85287

theorem repeating_decimal_to_fraction :
  (0.512341234123412341234 : ℝ) = (51229 / 99990 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l85_85287


namespace common_denominator_step1_error_in_step3_simplified_expression_l85_85126

theorem common_denominator_step1 (x : ℝ) (h1: x ≠ 2) (h2: x ≠ -2):
  (3 * x / (x - 2) - x / (x + 2)) = (3 * x * (x + 2)) / ((x - 2) * (x + 2)) - (x * (x - 2)) / ((x - 2) * (x + 2)) :=
sorry

theorem error_in_step3 (x : ℝ) (h1: x ≠ 2) (h2: x ≠ -2) :
  (3 * x^2 + 6 * x - (x^2 - 2 * x)) / ((x - 2) * (x + 2)) ≠ (3 * x^2 + 6 * x * (x^2 - 2 * x)) / ((x - 2) * (x + 2)) :=
sorry

theorem simplified_expression (x : ℝ) (h1: x ≠ 0) (h2: x ≠ 2) (h3: x ≠ -2) :
  ((3 * x / (x - 2) - x / (x + 2)) * (x^2 - 4) / x) = 2 * x + 8 :=
sorry

end common_denominator_step1_error_in_step3_simplified_expression_l85_85126


namespace constant_two_l85_85728

theorem constant_two (p : ℕ) (h_prime : Nat.Prime p) (h_gt_two : p > 2) (c : ℕ) (n : ℕ) (h_n : n = c * p) (h_even_divisors : ∀ d : ℕ, d ∣ n → (d % 2 = 0) → d = 2) : c = 2 := by
  sorry

end constant_two_l85_85728


namespace car_travel_distance_l85_85144

theorem car_travel_distance:
  (∃ r, r = 3 / 4 ∧ ∀ t, t = 2 → ((r * 60) * t = 90)) :=
by
  sorry

end car_travel_distance_l85_85144


namespace cubic_function_properties_l85_85992

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 9 * x

theorem cubic_function_properties :
  (∀ (x : ℝ), deriv f x = 3 * x^2 - 12 * x + 9) ∧
  (f 1 = 4) ∧ 
  (deriv f 1 = 0) ∧
  (f 3 = 0) ∧ 
  (deriv f 3 = 0) ∧
  (f 0 = 0) :=
by
  sorry

end cubic_function_properties_l85_85992


namespace range_of_x_l85_85336

theorem range_of_x (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (x + 2) / (x - 3) :=
by {
  sorry
}

end range_of_x_l85_85336


namespace cost_price_of_article_l85_85210

theorem cost_price_of_article :
  ∃ (CP : ℝ), (616 = 1.10 * (1.17 * CP)) → CP = 478.77 :=
by
  sorry

end cost_price_of_article_l85_85210


namespace find_noon_temperature_l85_85007

theorem find_noon_temperature (T T₄₀₀ T₈₀₀ : ℝ) 
  (h1 : T₄₀₀ = T + 8)
  (h2 : T₈₀₀ = T₄₀₀ - 11)
  (h3 : T₈₀₀ = T + 1) : 
  T = 4 :=
by
  sorry

end find_noon_temperature_l85_85007


namespace animals_left_in_barn_l85_85417

-- Define the conditions
def num_pigs : Nat := 156
def num_cows : Nat := 267
def num_sold : Nat := 115

-- Define the question
def num_left := num_pigs + num_cows - num_sold

-- State the theorem
theorem animals_left_in_barn : num_left = 308 :=
by
  sorry

end animals_left_in_barn_l85_85417


namespace divide_and_add_l85_85105

theorem divide_and_add (x : ℤ) (h1 : x = 95) : (x / 5) + 23 = 42 := by
  sorry

end divide_and_add_l85_85105


namespace find_x_l85_85409

theorem find_x (x : ℝ) : 
  let a := (4, 2)
  let b := (x, 3)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -3 / 2 :=
by
  intros a b h
  sorry

end find_x_l85_85409


namespace paint_needed_l85_85537

-- Definitions from conditions
def total_needed_paint := 70
def initial_paint := 36
def bought_paint := 23

-- The main statement to prove
theorem paint_needed : total_needed_paint - (initial_paint + bought_ppaint) = 11 :=
by
  -- Definitions are already imported and stated
  -- Just need to refer these to the theorem assertion correctly
  sorry

end paint_needed_l85_85537


namespace total_bottles_l85_85173

theorem total_bottles (n : ℕ) (h1 : ∃ one_third two_third: ℕ, one_third = n / 3 ∧ two_third = 2 * (n / 3) ∧ 3 * one_third = n)
    (h2 : 25 ≤ n)
    (h3 : ∃ damage1 damage2 damage_diff : ℕ, damage1 = 25 * 160 ∧ damage2 = (n / 3) * 160 + ((2 * (n / 3) - 25) * 130) ∧ damage1 - damage2 = 660) :
    n = 36 :=
by
  sorry

end total_bottles_l85_85173


namespace find_angle_between_altitude_and_median_l85_85044

noncomputable def angle_between_altitude_and_median 
  (a b S : ℝ) (h1 : a > b) (h2 : S > 0) : ℝ :=
  Real.arctan ((a^2 - b^2) / (4 * S))

theorem find_angle_between_altitude_and_median 
  (a b S : ℝ) (h1 : a > b) (h2 : S > 0) : 
  angle_between_altitude_and_median a b S h1 h2 = 
    Real.arctan ((a^2 - b^2) / (4 * S)) := 
  sorry

end find_angle_between_altitude_and_median_l85_85044


namespace floss_leftover_l85_85732

noncomputable def leftover_floss
    (students : ℕ)
    (floss_per_student : ℚ)
    (floss_per_packet : ℚ) :
    ℚ :=
  let total_needed := students * floss_per_student
  let packets_needed := (total_needed / floss_per_packet).ceil
  let total_floss := packets_needed * floss_per_packet
  total_floss - total_needed

theorem floss_leftover {students : ℕ} {floss_per_student floss_per_packet : ℚ}
    (h_students : students = 20)
    (h_floss_per_student : floss_per_student = 3 / 2)
    (h_floss_per_packet : floss_per_packet = 35) :
    leftover_floss students floss_per_student floss_per_packet = 5 :=
by
  rw [h_students, h_floss_per_student, h_floss_per_packet]
  simp only [leftover_floss]
  norm_num
  sorry

end floss_leftover_l85_85732


namespace fraction_value_l85_85599

theorem fraction_value (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (1 / (y : ℚ) / (1 / (x : ℚ))) = 3 / 4 :=
by
  rw [hx, hy]
  norm_num

end fraction_value_l85_85599


namespace regular_polygons_enclosing_hexagon_l85_85661

theorem regular_polygons_enclosing_hexagon (m n : ℕ) 
  (hm : m = 6)
  (h_exterior_angle_central : 180 - ((m - 2) * 180 / m) = 60)
  (h_exterior_angle_enclosing : 2 * 60 = 120): 
  n = 3 := sorry

end regular_polygons_enclosing_hexagon_l85_85661


namespace find_angle_l85_85895

theorem find_angle (x : ℝ) (h : 180 - x = 6 * (90 - x)) : x = 72 := 
by 
    sorry

end find_angle_l85_85895


namespace correct_multiplication_l85_85036

variable {a : ℕ} -- Assume 'a' to be a natural number for simplicity in this example

theorem correct_multiplication : (3 * a) * (4 * a^2) = 12 * a^3 := by
  sorry

end correct_multiplication_l85_85036


namespace min_value_one_over_a_plus_one_over_b_point_P_outside_ellipse_l85_85038

variable (a b : ℝ)
-- Conditions: a and b are positive real numbers and (a + b)x - 1 ≤ x^2 for all x > 0
variables (ha : a > 0) (hb : b > 0) (h : ∀ x : ℝ, 0 < x → (a + b) * x - 1 ≤ x^2)

-- Question 1: Prove that the minimum value of 1/a + 1/b is 2
theorem min_value_one_over_a_plus_one_over_b : (1 : ℝ) / a + (1 : ℝ) / b = 2 := 
sorry

-- Question 2: Determine point P(1, -1) relative to the ellipse x^2/a^2 + y^2/b^2 = 1
theorem point_P_outside_ellipse : (1 : ℝ)^2 / (a^2) + (-1 : ℝ)^2 / (b^2) > 1 :=
sorry

end min_value_one_over_a_plus_one_over_b_point_P_outside_ellipse_l85_85038


namespace math_problem_l85_85709

theorem math_problem :
  (- (1 / 8)) ^ 2007 * (- 8) ^ 2008 = -8 :=
by
  sorry

end math_problem_l85_85709


namespace john_amount_share_l85_85274

theorem john_amount_share {total_amount : ℕ} {total_parts john_share : ℕ} (h1 : total_amount = 4200) (h2 : total_parts = 2 + 4 + 6) (h3 : john_share = 2) :
  john_share * (total_amount / total_parts) = 700 :=
by
  sorry

end john_amount_share_l85_85274


namespace original_cost_price_l85_85241

-- Define the conditions
def selling_price : ℝ := 24000
def discount_rate : ℝ := 0.15
def tax_rate : ℝ := 0.02
def profit_rate : ℝ := 0.12

-- Define the necessary calculations
def discounted_price (sp : ℝ) (dr : ℝ) : ℝ := sp * (1 - dr)
def total_tax (sp : ℝ) (tr : ℝ) : ℝ := sp * tr
def profit (c : ℝ) (pr : ℝ) : ℝ := c * (1 + pr)

-- The problem is to prove that the original cost price is $17,785.71
theorem original_cost_price : 
  ∃ (C : ℝ), C = 17785.71 ∧ 
  selling_price * (1 - discount_rate - tax_rate) = (1 + profit_rate) * C :=
sorry

end original_cost_price_l85_85241


namespace water_level_height_l85_85104

/-- Problem: An inverted frustum with a bottom diameter of 12 and height of 18, filled with water, 
    is emptied into another cylindrical container with a bottom diameter of 24. Assuming the 
    cylindrical container is sufficiently tall, the height of the water level in the cylindrical container -/
theorem water_level_height
  (V_cone : ℝ := (1 / 3) * π * (12 / 2) ^ 2 * 18)
  (R_cyl : ℝ := 24 / 2)
  (H_cyl : ℝ) :
  V_cone = π * R_cyl ^ 2 * H_cyl →
  H_cyl = 1.5 :=
by 
  sorry

end water_level_height_l85_85104


namespace largest_value_among_expressions_l85_85445

def expA : ℕ := 3 + 1 + 2 + 4
def expB : ℕ := 3 * 1 + 2 + 4
def expC : ℕ := 3 + 1 * 2 + 4
def expD : ℕ := 3 + 1 + 2 * 4
def expE : ℕ := 3 * 1 * 2 * 4

theorem largest_value_among_expressions :
  expE > expA ∧ expE > expB ∧ expE > expC ∧ expE > expD :=
by
  -- Proof will go here
  sorry

end largest_value_among_expressions_l85_85445


namespace parabola_tangent_circle_l85_85724

noncomputable def parabola_focus (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p / 2, 0)

theorem parabola_tangent_circle (p : ℝ) (hp : p > 0)
  (x0 : ℝ) (hx0 : x0 = p)
  (M : ℝ × ℝ) (hM : M = (x0, 2 * (Real.sqrt 2)))
  (MA AF : ℝ) (h_ratio : MA / AF = 2) :
  p = 2 :=
by
  sorry

end parabola_tangent_circle_l85_85724


namespace proof_goal_l85_85209

noncomputable def proof_problem (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a^2 + b^2 + c^2 = 1) : Prop :=
  (1 / a) + (1 / b) + (1 / c) > 4

theorem proof_goal (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a^2 + b^2 + c^2 = 1) : 
  (1 / a) + (1 / b) + (1 / c) > 4 :=
sorry

end proof_goal_l85_85209


namespace smallest_b_for_34b_perfect_square_is_4_l85_85635

theorem smallest_b_for_34b_perfect_square_is_4 :
  ∃ n : ℕ, ∀ b : ℤ, b > 3 → (3 * b + 4 = n * n → b = 4) :=
by
  existsi 4
  intros b hb
  intro h
  sorry

end smallest_b_for_34b_perfect_square_is_4_l85_85635


namespace find_x_l85_85473

theorem find_x 
  (x : ℝ) 
  (angle_PQS angle_QSR angle_SRQ : ℝ) 
  (h1 : angle_PQS = 2 * x)
  (h2 : angle_QSR = 50)
  (h3 : angle_SRQ = x) :
  x = 50 :=
sorry

end find_x_l85_85473


namespace empty_subset_singleton_l85_85695

theorem empty_subset_singleton : (∅ ⊆ ({0} : Set ℕ)) = true :=
by sorry

end empty_subset_singleton_l85_85695


namespace GregsAgeIs16_l85_85804

def CindyAge := 5
def JanAge := CindyAge + 2
def MarciaAge := 2 * JanAge
def GregAge := MarciaAge + 2

theorem GregsAgeIs16 : GregAge = 16 := by
  sorry

end GregsAgeIs16_l85_85804


namespace sum_geometric_series_l85_85039

theorem sum_geometric_series :
  ∑' n : ℕ+, (3 : ℝ)⁻¹ ^ (n : ℕ) = (1 / 2 : ℝ) := by
  sorry

end sum_geometric_series_l85_85039


namespace prime_condition_l85_85756

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_condition (p : ℕ) (h1 : is_prime p) (h2 : is_prime (8 * p^2 + 1)) : 
  p = 3 ∧ is_prime (8 * p^2 - p + 2) :=
by
  sorry

end prime_condition_l85_85756


namespace seven_digit_number_subtraction_l85_85546

theorem seven_digit_number_subtraction 
  (n : ℕ)
  (d1 d2 d3 d4 d5 d6 d7 : ℕ)
  (h1 : n = d1 * 10^6 + d2 * 10^5 + d3 * 10^4 + d4 * 10^3 + d5 * 10^2 + d6 * 10 + d7)
  (h2 : d1 < 10 ∧ d2 < 10 ∧ d3 < 10 ∧ d4 < 10 ∧ d5 < 10 ∧ d6 < 10 ∧ d7 < 10)
  (h3 : n - (d1 + d3 + d4 + d5 + d6 + d7) = 9875352) :
  n - (d1 + d3 + d4 + d5 + d6 + d7 - d2) = 9875357 :=
sorry

end seven_digit_number_subtraction_l85_85546


namespace law_school_student_count_l85_85446

theorem law_school_student_count 
    (business_students : ℕ)
    (sibling_pairs : ℕ)
    (selection_probability : ℚ)
    (L : ℕ)
    (h1 : business_students = 500)
    (h2 : sibling_pairs = 30)
    (h3 : selection_probability = 7.500000000000001e-5) :
    L = 8000 :=
by
  sorry

end law_school_student_count_l85_85446


namespace seats_filled_percentage_l85_85067

theorem seats_filled_percentage (total_seats vacant_seats : ℕ) (h1 : total_seats = 600) (h2 : vacant_seats = 228) :
  ((total_seats - vacant_seats) / total_seats * 100 : ℝ) = 62 := by
  sorry

end seats_filled_percentage_l85_85067


namespace cube_side_length_l85_85628

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 / (6 * n^3) = 1 / 3) : n = 3 :=
by
  sorry

end cube_side_length_l85_85628


namespace total_turnover_in_first_quarter_l85_85674

theorem total_turnover_in_first_quarter (x : ℝ) : 
  200 + 200 * (1 + x) + 200 * (1 + x) ^ 2 = 1000 :=
sorry

end total_turnover_in_first_quarter_l85_85674


namespace ratio_man_to_son_in_two_years_l85_85118

-- Define the conditions
def son_current_age : ℕ := 32
def man_current_age : ℕ := son_current_age + 34

-- Define the ages in two years
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- The theorem to prove the ratio in two years
theorem ratio_man_to_son_in_two_years : 
  (man_age_in_two_years : ℚ) / son_age_in_two_years = 2 :=
by
  -- Skip the proof
  sorry

end ratio_man_to_son_in_two_years_l85_85118


namespace sum_first_4_terms_l85_85090

theorem sum_first_4_terms 
  (a_1 : ℚ) 
  (q : ℚ) 
  (h1 : a_1 * q - a_1 * q^2 = -2) 
  (h2 : a_1 + a_1 * q^2 = 10 / 3) 
  : a_1 * (1 + q + q^2 + q^3) = 40 / 3 := sorry

end sum_first_4_terms_l85_85090


namespace loss_percentage_is_nine_percent_l85_85168

theorem loss_percentage_is_nine_percent
    (C S : ℝ)
    (h1 : 15 * C = 20 * S)
    (discount_rate : ℝ := 0.10)
    (tax_rate : ℝ := 0.08) :
    (((0.9 * C) - (1.08 * S)) / C) * 100 = 9 :=
by
  sorry

end loss_percentage_is_nine_percent_l85_85168


namespace cream_cheese_volume_l85_85134

theorem cream_cheese_volume
  (raw_spinach : ℕ)
  (spinach_reduction : ℕ)
  (eggs_volume : ℕ)
  (total_volume : ℕ)
  (cooked_spinach : ℕ)
  (cream_cheese : ℕ) :
  raw_spinach = 40 →
  spinach_reduction = 20 →
  eggs_volume = 4 →
  total_volume = 18 →
  cooked_spinach = raw_spinach * spinach_reduction / 100 →
  cream_cheese = total_volume - cooked_spinach - eggs_volume →
  cream_cheese = 6 :=
by
  intros h_raw_spinach h_spinach_reduction h_eggs_volume h_total_volume h_cooked_spinach h_cream_cheese
  sorry

end cream_cheese_volume_l85_85134


namespace average_speed_of_car_l85_85861

noncomputable def avgSpeed (Distance_uphill Speed_uphill Distance_downhill Speed_downhill : ℝ) : ℝ :=
  let Time_uphill := Distance_uphill / Speed_uphill
  let Time_downhill := Distance_downhill / Speed_downhill
  let Total_time := Time_uphill + Time_downhill
  let Total_distance := Distance_uphill + Distance_downhill
  Total_distance / Total_time

theorem average_speed_of_car:
  avgSpeed 100 30 50 60 = 36 := by
  sorry

end average_speed_of_car_l85_85861


namespace gasoline_price_increase_l85_85797

theorem gasoline_price_increase
  (P Q : ℝ)
  (h1 : (P * Q) * 1.10 = P * (1 + X / 100) * Q * 0.88) :
  X = 25 :=
by
  -- proof here
  sorry

end gasoline_price_increase_l85_85797


namespace total_animal_eyes_l85_85215

def num_snakes := 18
def num_alligators := 10
def eyes_per_snake := 2
def eyes_per_alligator := 2

theorem total_animal_eyes : 
  (num_snakes * eyes_per_snake) + (num_alligators * eyes_per_alligator) = 56 :=
by 
  sorry

end total_animal_eyes_l85_85215


namespace abs_val_problem_l85_85524

variable (a b : ℝ)

theorem abs_val_problem (h_abs_a : |a| = 2) (h_abs_b : |b| = 4) (h_sum_neg : a + b < 0) : a - b = 2 ∨ a - b = 6 :=
sorry

end abs_val_problem_l85_85524


namespace fruits_total_l85_85452

def remaining_fruits (frank_apples susan_blueberries henry_apples karen_grapes : ℤ) : ℤ :=
  let frank_remaining := 36 - (36 / 3)
  let susan_remaining := 120 - (120 / 2)
  let henry_collected := 2 * 120
  let henry_after_eating := henry_collected - (henry_collected / 4)
  let henry_remaining := henry_after_eating - (henry_after_eating / 10)
  let karen_collected := henry_collected / 2
  let karen_after_spoilage := karen_collected - (15 * karen_collected / 100)
  let karen_after_giving_away := karen_after_spoilage - (karen_after_spoilage / 3)
  let karen_remaining := karen_after_giving_away - (Int.sqrt karen_after_giving_away)
  frank_remaining + susan_remaining + henry_remaining + karen_remaining

theorem fruits_total : remaining_fruits 36 120 240 120 = 254 :=
by sorry

end fruits_total_l85_85452


namespace electricity_fee_l85_85572

theorem electricity_fee (a b : ℝ) : 
  let base_usage := 100
  let additional_usage := 160 - base_usage
  let base_cost := base_usage * a
  let additional_cost := additional_usage * b
  base_cost + additional_cost = 100 * a + 60 * b :=
by
  sorry

end electricity_fee_l85_85572


namespace inequality_a3_minus_b3_l85_85532

theorem inequality_a3_minus_b3 (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a^3 - b^3 < 0 :=
by sorry

end inequality_a3_minus_b3_l85_85532


namespace yellow_yellow_pairs_l85_85390

variable (students_total : ℕ := 150)
variable (blue_students : ℕ := 65)
variable (yellow_students : ℕ := 85)
variable (total_pairs : ℕ := 75)
variable (blue_blue_pairs : ℕ := 30)

theorem yellow_yellow_pairs : 
  (yellow_students - (blue_students - blue_blue_pairs * 2)) / 2 = 40 :=
by 
  -- proof goes here
  sorry

end yellow_yellow_pairs_l85_85390


namespace polygon_sides_and_diagonals_l85_85254

theorem polygon_sides_and_diagonals (n : ℕ) (h : (n-2) * 180 / 360 = 13 / 2) : 
  n = 15 ∧ (n * (n - 3) / 2 = 90) :=
by {
  sorry
}

end polygon_sides_and_diagonals_l85_85254


namespace sin_and_tan_alpha_in_second_quadrant_expression_value_for_given_tan_l85_85198

theorem sin_and_tan_alpha_in_second_quadrant 
  (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (hcos : Real.cos α = -8 / 17) :
  Real.sin α = 15 / 17 ∧ Real.tan α = -15 / 8 := 
  sorry

theorem expression_value_for_given_tan 
  (α : ℝ) (htan : Real.tan α = 2) :
  (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7 := 
  sorry

end sin_and_tan_alpha_in_second_quadrant_expression_value_for_given_tan_l85_85198


namespace find_fraction_l85_85703

theorem find_fraction (F N : ℝ) 
  (h1 : F * (1 / 4 * N) = 15)
  (h2 : (3 / 10) * N = 54) : 
  F = 1 / 3 := 
by
  sorry

end find_fraction_l85_85703


namespace initial_bacteria_count_l85_85374

theorem initial_bacteria_count (doubling_interval : ℕ) (initial_count four_minutes_final_count : ℕ)
  (h1 : doubling_interval = 30)
  (h2 : four_minutes_final_count = 524288)
  (h3 : ∀ t : ℕ, initial_count * 2 ^ (t / doubling_interval) = four_minutes_final_count) :
  initial_count = 2048 :=
sorry

end initial_bacteria_count_l85_85374


namespace sampling_probability_equal_l85_85611

theorem sampling_probability_equal :
  let total_people := 2014
  let first_sample := 14
  let remaining_people := total_people - first_sample
  let sample_size := 50
  let probability := sample_size / total_people
  50 / 2014 = 25 / 1007 :=
by
  sorry

end sampling_probability_equal_l85_85611


namespace area_of_triangle_l85_85318

theorem area_of_triangle (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (angle : ℝ) (h_side_ratio : side2 / side3 = 8 / 5)
  (h_side_opposite : side1 = 14)
  (h_angle_opposite : angle = 60) :
  (1/2 * side2 * side3 * Real.sin (angle * Real.pi / 180)) = 40 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l85_85318


namespace value_of_x_for_real_y_l85_85582

theorem value_of_x_for_real_y (x y : ℝ) (h : 4 * y^2 + 2 * x * y + |x| + 8 = 0) :
  (x ≤ -10) ∨ (x ≥ 10) :=
sorry

end value_of_x_for_real_y_l85_85582


namespace number_of_pencil_boxes_l85_85519

-- Define the total number of pencils and pencils per box as given conditions
def total_pencils : ℝ := 2592
def pencils_per_box : ℝ := 648.0

-- Problem statement: To prove the number of pencil boxes is 4
theorem number_of_pencil_boxes : total_pencils / pencils_per_box = 4 := by
  sorry

end number_of_pencil_boxes_l85_85519


namespace twenty_one_less_than_sixty_thousand_l85_85605

theorem twenty_one_less_than_sixty_thousand : 60000 - 21 = 59979 :=
by
  sorry

end twenty_one_less_than_sixty_thousand_l85_85605


namespace directrix_of_parabola_l85_85441

-- Define the given condition: the equation of the parabola
def given_parabola (x : ℝ) : ℝ := 4 * x ^ 2

-- State the theorem to be proven
theorem directrix_of_parabola : 
  (∀ x : ℝ, given_parabola x = 4 * x ^ 2) → 
  (y = -1 / 16) :=
sorry

end directrix_of_parabola_l85_85441


namespace intersection_condition_l85_85435

noncomputable def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
noncomputable def N (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem intersection_condition (k : ℝ) (h : M ⊆ N k) : k ≥ 2 :=
  sorry

end intersection_condition_l85_85435


namespace inscribed_sphere_radius_l85_85660

theorem inscribed_sphere_radius (a α : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) :
  ∃ (ρ : ℝ), ρ = a * (1 - Real.cos α) / (2 * Real.sqrt (1 + Real.cos α) * (1 + Real.sqrt (- Real.cos α))) :=
  sorry

end inscribed_sphere_radius_l85_85660


namespace right_triangle_of_medians_l85_85576

theorem right_triangle_of_medians
  (a b c m1 m2 m3 : ℝ)
  (h1 : 4 * m1^2 = 2 * (b^2 + c^2) - a^2)
  (h2 : 4 * m2^2 = 2 * (a^2 + c^2) - b^2)
  (h3 : 4 * m3^2 = 2 * (a^2 + b^2) - c^2)
  (h4 : m1^2 + m2^2 = 5 * m3^2) :
  c^2 = a^2 + b^2 :=
by
  sorry

end right_triangle_of_medians_l85_85576


namespace complement_U_A_union_B_is_1_and_9_l85_85123

-- Define the universe set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define set A according to the given condition
def is_elem_of_A (x : ℕ) : Prop := 2 < x ∧ x ≤ 6
def A : Set ℕ := {x | is_elem_of_A x}

-- Define set B explicitly
def B : Set ℕ := {0, 2, 4, 5, 7, 8}

-- Define the union A ∪ B
def A_union_B : Set ℕ := A ∪ B

-- Define the complement of A ∪ B in U
def complement_U_A_union_B : Set ℕ := {x ∈ U | x ∉ A_union_B}

-- State the theorem
theorem complement_U_A_union_B_is_1_and_9 :
  complement_U_A_union_B = {1, 9} :=
by
  sorry

end complement_U_A_union_B_is_1_and_9_l85_85123


namespace shirt_cost_l85_85665

def george_initial_money : ℕ := 100
def total_spent_on_clothes (initial_money remaining_money : ℕ) : ℕ := initial_money - remaining_money
def socks_cost : ℕ := 11
def remaining_money_after_purchase : ℕ := 65

theorem shirt_cost
  (initial_money : ℕ)
  (remaining_money : ℕ)
  (total_spent : ℕ)
  (socks_cost : ℕ)
  (remaining_money_after_purchase : ℕ) :
  initial_money = 100 →
  remaining_money = 65 →
  total_spent = initial_money - remaining_money →
  total_spent = 35 →
  socks_cost = 11 →
  remaining_money_after_purchase = remaining_money →
  (total_spent - socks_cost = 24) :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h4] at *
  exact sorry

end shirt_cost_l85_85665


namespace problem_solution_l85_85377

theorem problem_solution (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : x^11 - 7 * x^7 + x^3 = 0 := 
sorry

end problem_solution_l85_85377


namespace perpendicular_vector_x_value_l85_85729

-- Definitions based on the given problem conditions
def dot_product_perpendicular (a1 a2 b1 b2 x : ℝ) : Prop :=
  (a1 * b1 + a2 * b2 = 0)

-- Statement to be proved
theorem perpendicular_vector_x_value (x : ℝ) :
  dot_product_perpendicular 4 x 2 4 x → x = -2 :=
by
  intros h
  sorry

end perpendicular_vector_x_value_l85_85729


namespace madhav_rank_from_last_is_15_l85_85541

-- Defining the conditions
def class_size : ℕ := 31
def madhav_rank_from_start : ℕ := 17

-- Statement to be proved
theorem madhav_rank_from_last_is_15 :
  (class_size - madhav_rank_from_start + 1) = 15 := by
  sorry

end madhav_rank_from_last_is_15_l85_85541


namespace clothing_store_gross_profit_l85_85062

theorem clothing_store_gross_profit :
  ∃ S : ℝ, S = 81 + 0.25 * S ∧
  ∃ new_price : ℝ,
    new_price = S - 0.20 * S ∧
    ∃ profit : ℝ,
      profit = new_price - 81 ∧
      profit = 5.40 :=
by
  sorry

end clothing_store_gross_profit_l85_85062


namespace distinct_cubes_meet_condition_l85_85742

theorem distinct_cubes_meet_condition :
  ∃ (a b c d e f : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a + b + c + d + e + f = 60) ∧
    ∃ (k : ℕ), 
        ((a = k) ∧ (b = k) ∧ (c = k) ∧ (d = k) ∧ (e = k) ∧ (f = k)) ∧
        -- Number of distinct ways
        (∃ (num_ways : ℕ), num_ways = 84) :=
sorry

end distinct_cubes_meet_condition_l85_85742


namespace minimum_value_of_x_plus_y_l85_85420

theorem minimum_value_of_x_plus_y
  (x y : ℝ)
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : (1 / y) + (4 / x) = 1) : 
  x + y = 9 :=
sorry

end minimum_value_of_x_plus_y_l85_85420


namespace petya_coloring_failure_7_petya_coloring_failure_10_l85_85035

theorem petya_coloring_failure_7 :
  ¬ ∀ (points : Fin 200 → Fin 7) (segments : ∀ (i j : Fin 200), i ≠ j → Fin 7),
  ∃ (colors : ∀ (i j : Fin 200), i ≠ j → Fin 7),
  ∀ (i j : Fin 200) (h : i ≠ j),
    (segments i j h ≠ points i) ∧ (segments i j h ≠ points j) :=
sorry

theorem petya_coloring_failure_10 :
  ¬ ∀ (points : Fin 200 → Fin 10) (segments : ∀ (i j : Fin 200), i ≠ j → Fin 10),
  ∃ (colors : ∀ (i j : Fin 200), i ≠ j → Fin 10),
  ∀ (i j : Fin 200) (h : i ≠ j),
    (segments i j h ≠ points i) ∧ (segments i j h ≠ points j) :=
sorry

end petya_coloring_failure_7_petya_coloring_failure_10_l85_85035


namespace chord_length_l85_85850

theorem chord_length
  (a b c A B C : ℝ)
  (h₁ : c * Real.sin C = 3 * a * Real.sin A + 3 * b * Real.sin B)
  (O : ℝ → ℝ → Prop)
  (hO : ∀ x y, O x y ↔ x^2 + y^2 = 12)
  (l : ℝ → ℝ → Prop)
  (hl : ∀ x y, l x y ↔ a * x - b * y + c = 0) :
  (2 * Real.sqrt ( (2 * Real.sqrt 3)^2 - (Real.sqrt 3)^2 )) = 6 :=
by
  sorry

end chord_length_l85_85850


namespace determine_f_value_l85_85694

noncomputable def f (t : ℝ) : ℝ := t^2 + 2

theorem determine_f_value : f 3 = 11 := by
  sorry

end determine_f_value_l85_85694


namespace efficacy_rate_is_80_percent_l85_85097

-- Define the total number of people surveyed
def total_people : ℕ := 20

-- Define the number of people who find the new drug effective
def effective_people : ℕ := 16

-- Calculate the efficacy rate
def efficacy_rate (effective : ℕ) (total : ℕ) : ℚ := effective / total

-- The theorem to be proved
theorem efficacy_rate_is_80_percent : efficacy_rate effective_people total_people = 0.8 :=
by
  sorry

end efficacy_rate_is_80_percent_l85_85097


namespace lcm_division_l85_85013

open Nat

-- Define the LCM function for a list of integers
def list_lcm (l : List Nat) : Nat := l.foldr (fun a b => Nat.lcm a b) 1

-- Define the sequence ranges
def range1 := List.range' 20 21 -- From 20 to 40 inclusive
def range2 := List.range' 41 10 -- From 41 to 50 inclusive

-- Define P and Q
def P : Nat := list_lcm range1
def Q : Nat := Nat.lcm P (list_lcm range2)

-- The theorem statement
theorem lcm_division : (Q / P) = 55541 := by
  sorry

end lcm_division_l85_85013


namespace total_students_in_class_l85_85116

theorem total_students_in_class (S R : ℕ)
  (h1 : S = 2 + 12 + 4 + R)
  (h2 : 0 * 2 + 1 * 12 + 2 * 4 + 3 * R = 2 * S) : S = 34 :=
by { sorry }

end total_students_in_class_l85_85116


namespace ab_value_l85_85184

theorem ab_value (a b : ℝ) (h1 : a + b = 7) (h2 : a^3 + b^3 = 91) : a * b = 12 :=
by
  sorry

end ab_value_l85_85184


namespace man_speed_km_per_hr_l85_85043

noncomputable def train_length : ℝ := 110
noncomputable def train_speed_km_per_hr : ℝ := 82
noncomputable def time_to_pass_man_sec : ℝ := 4.499640028797696

theorem man_speed_km_per_hr :
  ∃ (Vm_km_per_hr : ℝ), Vm_km_per_hr = 6.0084 :=
sorry

end man_speed_km_per_hr_l85_85043


namespace expenditure_recording_l85_85578

def income : ℕ := 200
def recorded_income : ℤ := 200
def expenditure (e : ℕ) : ℤ := -(e : ℤ)

theorem expenditure_recording (e : ℕ) :
  expenditure 150 = -150 := by
  sorry

end expenditure_recording_l85_85578


namespace connie_grandma_birth_year_l85_85984

theorem connie_grandma_birth_year :
  ∀ (B S G : ℕ),
  B = 1932 →
  S = 1936 →
  (S - B) * 2 = (S - G) →
  G = 1928 := 
by
  intros B S G hB hS hGap
  -- Proof goes here
  sorry

end connie_grandma_birth_year_l85_85984


namespace range_of_m_l85_85824

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, e ≤ x ∧ x ≤ e^2 ∧ f x - m * x - 1/2 + m ≤ 0) →
  1/2 ≤ m := by
  sorry

end range_of_m_l85_85824


namespace green_yarn_length_l85_85355

/-- The length of the green piece of yarn given the red yarn is 8 cm more 
than three times the length of the green yarn and the total length 
for 2 pieces of yarn is 632 cm. -/
theorem green_yarn_length (G R : ℕ) 
  (h1 : R = 3 * G + 8)
  (h2 : G + R = 632) : 
  G = 156 := 
by
  sorry

end green_yarn_length_l85_85355


namespace frequency_total_students_l85_85387

noncomputable def total_students (known : ℕ) (freq : ℝ) : ℝ :=
known / freq

theorem frequency_total_students (known : ℕ) (freq : ℝ) (h1 : known = 40) (h2 : freq = 0.8) :
  total_students known freq = 50 :=
by
  rw [total_students, h1, h2]
  norm_num

end frequency_total_students_l85_85387


namespace hyperbola_eccentricity_l85_85637

theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (h4 : c = 3 * b) 
  (h5 : c * c = a * a + b * b)
  (h6 : e = c / a) :
  e = 3 * Real.sqrt 2 / 4 :=
by
  sorry

end hyperbola_eccentricity_l85_85637


namespace div_by_19_l85_85050

theorem div_by_19 (n : ℕ) (h : n > 0) : (3^(3*n+2) + 5 * 2^(3*n+1)) % 19 = 0 := by
  sorry

end div_by_19_l85_85050


namespace functional_equation_solution_l85_85142

open Nat

theorem functional_equation_solution (f : ℕ+ → ℕ+) 
  (H : ∀ (m n : ℕ+), f (f (f m) * f (f m) + 2 * f (f n) * f (f n)) = m * m + 2 * n * n) : 
  ∀ n : ℕ+, f n = n := 
sorry

end functional_equation_solution_l85_85142


namespace angle_of_inclination_l85_85016

theorem angle_of_inclination (m : ℝ) (h : m = -1) : 
  ∃ α : ℝ, α = 3 * Real.pi / 4 := 
sorry

end angle_of_inclination_l85_85016


namespace necessary_but_not_sufficient_l85_85670

theorem necessary_but_not_sufficient (x : ℝ) : (x > 1 → x > 2) = (false) ∧ (x > 2 → x > 1) = (true) := by
  sorry

end necessary_but_not_sufficient_l85_85670


namespace sophomores_selected_l85_85929

variables (total_students freshmen sophomores juniors selected_students : ℕ)
def high_school_data := total_students = 2800 ∧ freshmen = 970 ∧ sophomores = 930 ∧ juniors = 900 ∧ selected_students = 280

theorem sophomores_selected (h : high_school_data total_students freshmen sophomores juniors selected_students) :
  (930 / 2800 : ℚ) * 280 = 93 := by
  sorry

end sophomores_selected_l85_85929


namespace num_students_third_school_l85_85051

variable (x : ℕ)

def num_students_condition := (2 * (x + 40) + (x + 40) + x = 920)

theorem num_students_third_school (h : num_students_condition x) : x = 200 :=
sorry

end num_students_third_school_l85_85051


namespace ann_total_fare_for_100_miles_l85_85358

-- Conditions
def base_fare : ℕ := 20
def fare_per_distance (distance : ℕ) : ℕ := 180 * distance / 80

-- Question: How much would Ann be charged if she traveled 100 miles?
def total_fare (distance : ℕ) : ℕ := (fare_per_distance distance) + base_fare

-- Prove that the total fare for 100 miles is 245 dollars
theorem ann_total_fare_for_100_miles : total_fare 100 = 245 :=
by
  -- Adding your proof here
  sorry

end ann_total_fare_for_100_miles_l85_85358


namespace find_a_l85_85766

def mul_op (a b : ℝ) : ℝ := 2 * a - b^2

theorem find_a (a : ℝ) (h : mul_op a 3 = 7) : a = 8 :=
sorry

end find_a_l85_85766


namespace alpha_beta_square_eq_eight_l85_85178

theorem alpha_beta_square_eq_eight (α β : ℝ) 
  (hα : α^2 = 2*α + 1) 
  (hβ : β^2 = 2*β + 1) 
  (h_distinct : α ≠ β) : 
  (α - β)^2 = 8 := 
sorry

end alpha_beta_square_eq_eight_l85_85178


namespace lcm_of_12_and_15_l85_85259
-- Import the entire Mathlib library

-- Define the given conditions
def HCF (a b : ℕ) : ℕ := gcd a b
def LCM (a b : ℕ) : ℕ := (a * b) / (gcd a b)

-- Given the values
def a := 12
def b := 15
def hcf := 3

-- State the proof problem
theorem lcm_of_12_and_15 : LCM a b = 60 :=
by
  -- Proof goes here (skipped)
  sorry

end lcm_of_12_and_15_l85_85259


namespace riley_pawns_lost_l85_85883

theorem riley_pawns_lost (initial_pawns : ℕ) (kennedy_lost : ℕ) (total_pawns_left : ℕ)
  (kennedy_initial_pawns : ℕ) (riley_initial_pawns : ℕ) : 
  kennedy_initial_pawns = initial_pawns ∧
  riley_initial_pawns = initial_pawns ∧
  kennedy_lost = 4 ∧
  total_pawns_left = 11 →
  riley_initial_pawns - (total_pawns_left - (kennedy_initial_pawns - kennedy_lost)) = 1 :=
by
  sorry

end riley_pawns_lost_l85_85883


namespace evaluate_f_at_3_l85_85591

-- Function definition
def f (x : ℚ) : ℚ := (x - 2) / (4 * x + 5)

-- Problem statement
theorem evaluate_f_at_3 : f 3 = 1 / 17 := by
  sorry

end evaluate_f_at_3_l85_85591


namespace equilateral_triangle_distances_l85_85333

-- Defining the necessary conditions
variables {h x y z : ℝ}
variables (hx : 0 < h) (hx_cond : x + y + z = h)
variables (triangle_ineqs : x + y > z ∧ y + z > x ∧ z + x > y)

-- Lean 4 statement to express the proof problem
theorem equilateral_triangle_distances (hx : 0 < h) (hx_cond : x + y + z = h) (triangle_ineqs : x + y > z ∧ y + z > x ∧ z + x > y) : 
  x < h / 2 ∧ y < h / 2 ∧ z < h / 2 :=
sorry

end equilateral_triangle_distances_l85_85333


namespace find_f2_l85_85057

def f (x : ℝ) (a b : ℝ) := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 3) : f 2 a b = -19 :=
by sorry

end find_f2_l85_85057


namespace adi_change_l85_85149

theorem adi_change : 
  let pencil := 0.35
  let notebook := 1.50
  let colored_pencils := 2.75
  let discount := 0.05
  let tax := 0.10
  let payment := 20.00
  let total_cost_before_discount := pencil + notebook + colored_pencils
  let discount_amount := discount * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let tax_amount := tax * total_cost_after_discount
  let total_cost := total_cost_after_discount + tax_amount
  let change := payment - total_cost
  change = 15.19 :=
by
  sorry

end adi_change_l85_85149
