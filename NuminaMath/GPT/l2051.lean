import Mathlib

namespace NUMINAMATH_GPT_transformation_result_l2051_205199

theorem transformation_result (a b : ℝ) 
  (h1 : ∃ P : ℝ × ℝ, P = (a, b))
  (h2 : ∃ Q : ℝ × ℝ, Q = (b, a))
  (h3 : ∃ R : ℝ × ℝ, R = (2 - b, 10 - a))
  (h4 : (2 - b, 10 - a) = (-8, 2)) : 
  a - b = -2 := 
by 
  sorry

end NUMINAMATH_GPT_transformation_result_l2051_205199


namespace NUMINAMATH_GPT_find_x_l2051_205120

noncomputable def x : ℝ :=
  0.49

theorem find_x (h : (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 0.81) / (Real.sqrt x) = 2.507936507936508) : 
  x = 0.49 :=
sorry

end NUMINAMATH_GPT_find_x_l2051_205120


namespace NUMINAMATH_GPT_number_of_correct_statements_l2051_205111

-- Definitions of the conditions from the problem
def seq_is_graphical_points := true  -- Statement 1
def seq_is_finite (s : ℕ → ℝ) := ∀ n, s n = 0 -- Statement 2
def seq_decreasing_implies_finite (s : ℕ → ℝ) := (∀ n, s (n + 1) ≤ s n) → seq_is_finite s -- Statement 3

-- Prove the number of correct statements is 1
theorem number_of_correct_statements : (seq_is_graphical_points = true ∧ ¬(∃ s: ℕ → ℝ, ¬seq_is_finite s) ∧ ∃ s : ℕ → ℝ, ¬seq_decreasing_implies_finite s) → 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_correct_statements_l2051_205111


namespace NUMINAMATH_GPT_percentage_increase_l2051_205119

theorem percentage_increase 
  (distance : ℝ) (time_q : ℝ) (time_y : ℝ) 
  (speed_q : ℝ) (speed_y : ℝ) 
  (percentage_increase : ℝ) 
  (h_distance : distance = 80)
  (h_time_q : time_q = 2)
  (h_time_y : time_y = 1.3333333333333333)
  (h_speed_q : speed_q = distance / time_q)
  (h_speed_y : speed_y = distance / time_y)
  (h_faster : speed_y > speed_q)
  : percentage_increase = ((speed_y - speed_q) / speed_q) * 100 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l2051_205119


namespace NUMINAMATH_GPT_remainder_div_357_l2051_205104

theorem remainder_div_357 (N : ℤ) (h : N % 17 = 2) : N % 357 = 2 :=
sorry

end NUMINAMATH_GPT_remainder_div_357_l2051_205104


namespace NUMINAMATH_GPT_simplify_fraction_l2051_205157

def a : ℕ := 2016
def b : ℕ := 2017

theorem simplify_fraction :
  (a^4 - 2 * a^3 * b + 3 * a^2 * b^2 - a * b^3 + 1) / (a^2 * b^2) = 1 - 1 / b^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2051_205157


namespace NUMINAMATH_GPT_find_e_value_l2051_205196

theorem find_e_value : 
  ∃ e : ℝ, 12 / (-12 + 2 * e) = -11 - 2 * e ∧ e = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_find_e_value_l2051_205196


namespace NUMINAMATH_GPT_george_earnings_l2051_205118

theorem george_earnings (cars_sold : ℕ) (price_per_car : ℕ) (lego_set_price : ℕ) (h1 : cars_sold = 3) (h2 : price_per_car = 5) (h3 : lego_set_price = 30) :
  cars_sold * price_per_car + lego_set_price = 45 :=
by
  sorry

end NUMINAMATH_GPT_george_earnings_l2051_205118


namespace NUMINAMATH_GPT_hot_air_balloon_height_l2051_205150

theorem hot_air_balloon_height (altitude_temp_decrease_per_1000m : ℝ) 
  (ground_temp : ℝ) (high_altitude_temp : ℝ) :
  altitude_temp_decrease_per_1000m = 6 →
  ground_temp = 8 →
  high_altitude_temp = -1 →
  ∃ (height : ℝ), height = 1500 :=
by
  intro h1 h2 h3
  have temp_change := ground_temp - high_altitude_temp
  have height := (temp_change / altitude_temp_decrease_per_1000m) * 1000
  exact Exists.intro height sorry -- height needs to be computed here

end NUMINAMATH_GPT_hot_air_balloon_height_l2051_205150


namespace NUMINAMATH_GPT_friends_division_ways_l2051_205187

theorem friends_division_ways : (4 ^ 8 = 65536) :=
by
  sorry

end NUMINAMATH_GPT_friends_division_ways_l2051_205187


namespace NUMINAMATH_GPT_distance_between_pathway_lines_is_5_l2051_205180

-- Define the conditions
def parallel_lines_distance (distance : ℤ) : Prop :=
  distance = 30

def pathway_length_between_lines (length : ℤ) : Prop :=
  length = 10

def pathway_line_length (length : ℤ) : Prop :=
  length = 60

-- Main proof problem
theorem distance_between_pathway_lines_is_5:
  ∀ (d : ℤ), parallel_lines_distance 30 → 
  pathway_length_between_lines 10 → 
  pathway_line_length 60 → 
  d = 5 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_pathway_lines_is_5_l2051_205180


namespace NUMINAMATH_GPT_find_A_l2051_205151

noncomputable def f (A B x : ℝ) : ℝ := A * x - 3 * B ^ 2
def g (B x : ℝ) : ℝ := B * x
variable (B : ℝ) (hB : B ≠ 0)

theorem find_A (h : f (A := A) B (g B 2) = 0) : A = 3 * B / 2 := by
  sorry

end NUMINAMATH_GPT_find_A_l2051_205151


namespace NUMINAMATH_GPT_find_value_of_f_l2051_205160

noncomputable def f (ω φ : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem find_value_of_f (ω φ : ℝ) (h_symmetry : ∀ x : ℝ, f ω φ (π/4 + x) = f ω φ (π/4 - x)) :
  f ω φ (π/4) = 2 ∨ f ω φ (π/4) = -2 := 
sorry

end NUMINAMATH_GPT_find_value_of_f_l2051_205160


namespace NUMINAMATH_GPT_isosceles_triangle_relationship_l2051_205113

theorem isosceles_triangle_relationship (x y : ℝ) (h1 : 2 * x + y = 30) (h2 : 7.5 < x) (h3 : x < 15) : 
  y = 30 - 2 * x :=
  by sorry

end NUMINAMATH_GPT_isosceles_triangle_relationship_l2051_205113


namespace NUMINAMATH_GPT_square_area_from_diagonal_l2051_205158

theorem square_area_from_diagonal (d : ℝ) (h_d : d = 12) : ∃ (A : ℝ), A = 72 :=
by
  -- we will use the given diagonal to derive the result
  sorry

end NUMINAMATH_GPT_square_area_from_diagonal_l2051_205158


namespace NUMINAMATH_GPT_lines_are_coplanar_l2051_205184

-- Define the first line
def line1 (t : ℝ) (m : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2 * t, 2 - m * t, 6 + t)

-- Define the second line
def line2 (u : ℝ) (m : ℝ) : ℝ × ℝ × ℝ :=
  (4 + m * u, 5 + 3 * u, 8 + 2 * u)

-- Define the vector connecting points on the lines when t=0 and u=0
def connecting_vector : ℝ × ℝ × ℝ :=
  (1, 3, 2)

-- Define the cross product of the direction vectors
def cross_product (m : ℝ) : ℝ × ℝ × ℝ :=
  ((-2 * m - 3), (m + 2), (6 + 2 * m))

-- Prove that lines are coplanar when m = -9/4
theorem lines_are_coplanar : ∃ k : ℝ, ∀ m : ℝ,
  cross_product m = (k * 1, k * 3, k * 2) → m = -9/4 :=
by
  sorry

end NUMINAMATH_GPT_lines_are_coplanar_l2051_205184


namespace NUMINAMATH_GPT_maximum_value_existence_l2051_205142

open Real

theorem maximum_value_existence (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
    8 * a + 3 * b + 5 * c ≤ sqrt (373 / 36) := by
  sorry

end NUMINAMATH_GPT_maximum_value_existence_l2051_205142


namespace NUMINAMATH_GPT_solve_for_x_l2051_205149

theorem solve_for_x : ∃ x : ℤ, 24 - 5 = 3 + x ∧ x = 16 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2051_205149


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l2051_205127

theorem equilateral_triangle_side_length 
  (x1 y1 : ℝ) 
  (hx1y1 : y1 = - (1 / 4) * x1^2)
  (h_eq_tri: ∃ (x2 y2 : ℝ), x2 = -x1 ∧ y2 = y1 ∧ (x2, y2) ≠ (x1, y1) ∧ ((x1 - x2)^2 + (y1 - y2)^2 = x1^2 + y1^2 ∧ (x1 - 0)^2 + (y1 - 0)^2 = (x1 - x2)^2 + (y1 - y2)^2)):
  2 * x1 = 8 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_equilateral_triangle_side_length_l2051_205127


namespace NUMINAMATH_GPT_initial_acorns_l2051_205166

theorem initial_acorns (T : ℝ) (h1 : 0.35 * T = 7) (h2 : 0.45 * T = 9) : T = 20 :=
sorry

end NUMINAMATH_GPT_initial_acorns_l2051_205166


namespace NUMINAMATH_GPT_minimum_value_S_l2051_205107

noncomputable def S (x a : ℝ) : ℝ := (x - a)^2 + (Real.log x - a)^2

theorem minimum_value_S : ∃ x a : ℝ, x > 0 ∧ (S x a = 1 / 2) := by
  sorry

end NUMINAMATH_GPT_minimum_value_S_l2051_205107


namespace NUMINAMATH_GPT_max_integer_k_l2051_205194

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x - 1)) / (x - 2)

theorem max_integer_k (x : ℝ) (k : ℕ) (hx : x > 2) :
  (∀ x, x > 2 → f x > (k : ℝ) / (x - 1)) ↔ k ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_integer_k_l2051_205194


namespace NUMINAMATH_GPT_angle_bisector_ratio_l2051_205139

theorem angle_bisector_ratio (XY XZ YZ : ℝ) (hXY : XY = 8) (hXZ : XZ = 6) (hYZ : YZ = 4) :
  ∃ (Q : Point) (YQ QV : ℝ), YQ / QV = 2 :=
by
  sorry

end NUMINAMATH_GPT_angle_bisector_ratio_l2051_205139


namespace NUMINAMATH_GPT_number_of_customers_l2051_205185

theorem number_of_customers 
  (offices sandwiches_per_office total_sandwiches group_sandwiches_per_customer half_group_sandwiches : ℕ)
  (h1 : offices = 3)
  (h2 : sandwiches_per_office = 10)
  (h3 : total_sandwiches = 54)
  (h4 : group_sandwiches_per_customer = 4)
  (h5 : half_group_sandwiches = 54 - (3 * 10))
  : half_group_sandwiches = 24 → 2 * 12 = 24 :=
by
  sorry

end NUMINAMATH_GPT_number_of_customers_l2051_205185


namespace NUMINAMATH_GPT_avg_of_multiples_of_10_eq_305_l2051_205133

theorem avg_of_multiples_of_10_eq_305 (N : ℕ) (h : N % 10 = 0) (h_avg : (10 + N) / 2 = 305) : N = 600 :=
sorry

end NUMINAMATH_GPT_avg_of_multiples_of_10_eq_305_l2051_205133


namespace NUMINAMATH_GPT_oranges_weight_l2051_205147

theorem oranges_weight (A O : ℕ) (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 := 
by 
  sorry

end NUMINAMATH_GPT_oranges_weight_l2051_205147


namespace NUMINAMATH_GPT_arrange_snow_leopards_l2051_205176

theorem arrange_snow_leopards :
  let n := 9 -- number of leopards
  let factorial x := (Nat.factorial x) -- definition for factorial
  let tall_short_perm := 2 -- there are 2 ways to arrange the tallest and shortest leopards at the ends
  tall_short_perm * factorial (n - 2) = 10080 := by sorry

end NUMINAMATH_GPT_arrange_snow_leopards_l2051_205176


namespace NUMINAMATH_GPT_evaluate_difference_of_squares_l2051_205154

theorem evaluate_difference_of_squares : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_GPT_evaluate_difference_of_squares_l2051_205154


namespace NUMINAMATH_GPT_total_combinations_l2051_205121

def varieties_of_wrapping_paper : Nat := 10
def colors_of_ribbon : Nat := 4
def types_of_gift_cards : Nat := 5
def kinds_of_decorative_stickers : Nat := 2

theorem total_combinations : varieties_of_wrapping_paper * colors_of_ribbon * types_of_gift_cards * kinds_of_decorative_stickers = 400 := by
  sorry

end NUMINAMATH_GPT_total_combinations_l2051_205121


namespace NUMINAMATH_GPT_angle_coloring_min_colors_l2051_205112

  theorem angle_coloring_min_colors (n : ℕ) : 
    (∃ c : ℕ, (c = 2 ↔ n % 2 = 0) ∧ (c = 3 ↔ n % 2 = 1)) :=
  by
    sorry
  
end NUMINAMATH_GPT_angle_coloring_min_colors_l2051_205112


namespace NUMINAMATH_GPT_exponent_proof_l2051_205169

theorem exponent_proof (n m : ℕ) (h1 : 4^n = 3) (h2 : 8^m = 5) : 2^(2*n + 3*m) = 15 :=
by
  -- Proof steps
  sorry

end NUMINAMATH_GPT_exponent_proof_l2051_205169


namespace NUMINAMATH_GPT_ten_times_product_is_2010_l2051_205189

theorem ten_times_product_is_2010 (n : ℕ) (hn : 10 ≤ n ∧ n < 100) : 
  (∃ k : ℤ, 4.02 * (n : ℝ) = k) → (10 * k = 2010) :=
by
  sorry

end NUMINAMATH_GPT_ten_times_product_is_2010_l2051_205189


namespace NUMINAMATH_GPT_solution_exists_l2051_205167

noncomputable def find_A_and_B : Prop :=
  ∃ A B : ℚ, 
    (A, B) = (75 / 16, 21 / 16) ∧ 
    ∀ x : ℚ, x ≠ 12 ∧ x ≠ -4 → 
    (6 * x + 3) / ((x - 12) * (x + 4)) = A / (x - 12) + B / (x + 4)

theorem solution_exists : find_A_and_B :=
sorry

end NUMINAMATH_GPT_solution_exists_l2051_205167


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2051_205102

open Real

theorem sufficient_but_not_necessary_condition :
  ∀ (m : ℝ),
  (∀ x, (x^2 - 3*x - 4 ≤ 0) → (x^2 - 6*x + 9 - m^2 ≤ 0)) ∧
  (∃ x, ¬(x^2 - 3*x - 4 ≤ 0) ∧ (x^2 - 6*x + 9 - m^2 ≤ 0)) ↔
  m ∈ Set.Iic (-4) ∪ Set.Ici 4 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2051_205102


namespace NUMINAMATH_GPT_mass_percentage_H_in_C4H8O2_l2051_205106

theorem mass_percentage_H_in_C4H8O2 (molar_mass_C : Real := 12.01) 
                                    (molar_mass_H : Real := 1.008) 
                                    (molar_mass_O : Real := 16.00) 
                                    (num_C_atoms : Nat := 4)
                                    (num_H_atoms : Nat := 8)
                                    (num_O_atoms : Nat := 2) :
    (num_H_atoms * molar_mass_H) / ((num_C_atoms * molar_mass_C) + (num_H_atoms * molar_mass_H) + (num_O_atoms * molar_mass_O)) * 100 = 9.15 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_H_in_C4H8O2_l2051_205106


namespace NUMINAMATH_GPT_correct_value_of_wrongly_read_number_l2051_205193

theorem correct_value_of_wrongly_read_number 
  (avg_wrong : ℝ) (n : ℕ) (wrong_value : ℝ) (avg_correct : ℝ) :
  avg_wrong = 5 →
  n = 10 →
  wrong_value = 26 →
  avg_correct = 6 →
  let sum_wrong := avg_wrong * n
  let correct_sum := avg_correct * n
  let difference := correct_sum - sum_wrong
  let correct_value := wrong_value + difference
  correct_value = 36 :=
by
  intros h_avg_wrong h_n h_wrong_value h_avg_correct
  let sum_wrong := avg_wrong * n
  let correct_sum := avg_correct * n
  let difference := correct_sum - sum_wrong
  let correct_value := wrong_value + difference
  sorry

end NUMINAMATH_GPT_correct_value_of_wrongly_read_number_l2051_205193


namespace NUMINAMATH_GPT_david_total_course_hours_l2051_205172

-- Definitions based on the conditions
def course_weeks : ℕ := 24
def three_hour_classes_per_week : ℕ := 2
def hours_per_three_hour_class : ℕ := 3
def four_hour_classes_per_week : ℕ := 1
def hours_per_four_hour_class : ℕ := 4
def homework_hours_per_week : ℕ := 4

-- Sum of weekly hours
def weekly_hours : ℕ := (three_hour_classes_per_week * hours_per_three_hour_class) +
                         (four_hour_classes_per_week * hours_per_four_hour_class) +
                         homework_hours_per_week

-- Total hours spent on the course
def total_hours : ℕ := weekly_hours * course_weeks

-- Prove that the total number of hours spent on the course is 336 hours
theorem david_total_course_hours : total_hours = 336 := by
  sorry

end NUMINAMATH_GPT_david_total_course_hours_l2051_205172


namespace NUMINAMATH_GPT_remaining_fruits_l2051_205156

theorem remaining_fruits (initial_apples initial_oranges initial_mangoes taken_apples twice_taken_apples taken_mangoes) : 
  initial_apples = 7 → 
  initial_oranges = 8 → 
  initial_mangoes = 15 → 
  taken_apples = 2 → 
  twice_taken_apples = 2 * taken_apples → 
  taken_mangoes = 2 * initial_mangoes / 3 → 
  initial_apples - taken_apples + initial_oranges - twice_taken_apples + initial_mangoes - taken_mangoes = 14 :=
by
  sorry

end NUMINAMATH_GPT_remaining_fruits_l2051_205156


namespace NUMINAMATH_GPT_monthly_rent_of_shop_l2051_205186

theorem monthly_rent_of_shop
  (length width : ℕ) (rent_per_sqft : ℕ)
  (h_length : length = 20) (h_width : width = 18) (h_rent : rent_per_sqft = 48) :
  (length * width * rent_per_sqft) / 12 = 1440 := 
by
  sorry

end NUMINAMATH_GPT_monthly_rent_of_shop_l2051_205186


namespace NUMINAMATH_GPT_sum_of_percentages_l2051_205125

theorem sum_of_percentages :
  let percent1 := 7.35 / 100
  let percent2 := 13.6 / 100
  let percent3 := 21.29 / 100
  let num1 := 12658
  let num2 := 18472
  let num3 := 29345
  let result := percent1 * num1 + percent2 * num2 + percent3 * num3
  result = 9689.9355 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_percentages_l2051_205125


namespace NUMINAMATH_GPT_inscribed_square_ratios_l2051_205181

theorem inscribed_square_ratios (a b c x y : ℝ) (h_right_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sides : a^2 + b^2 = c^2) 
  (h_leg_square : x = a) 
  (h_hyp_square : y = 5 / 18 * c) : 
  x / y = 18 / 13 := by
  sorry

end NUMINAMATH_GPT_inscribed_square_ratios_l2051_205181


namespace NUMINAMATH_GPT_gordon_total_cost_l2051_205137

noncomputable def DiscountA (price : ℝ) : ℝ :=
if price > 22.00 then price * 0.70 else price

noncomputable def DiscountB (price : ℝ) : ℝ :=
if 10.00 < price ∧ price <= 20.00 then price * 0.80 else price

noncomputable def DiscountC (price : ℝ) : ℝ :=
if price < 10.00 then price * 0.85 else price

noncomputable def apply_discount (price : ℝ) : ℝ :=
if price > 22.00 then DiscountA price
else if price > 10.00 then DiscountB price
else DiscountC price

noncomputable def total_price (prices : List ℝ) : ℝ :=
(prices.map apply_discount).sum

noncomputable def total_with_tax_and_fee (prices : List ℝ) (tax_rate extra_fee : ℝ) : ℝ :=
let total := total_price prices
let tax := total * tax_rate
total + tax + extra_fee

theorem gordon_total_cost :
  total_with_tax_and_fee
    [25.00, 18.00, 21.00, 35.00, 12.00, 10.00, 8.50, 23.00, 6.00, 15.50, 30.00, 9.50]
    0.05 2.00
  = 171.27 :=
  sorry

end NUMINAMATH_GPT_gordon_total_cost_l2051_205137


namespace NUMINAMATH_GPT_fg_eval_l2051_205124

def f (x : ℤ) : ℤ := x^3
def g (x : ℤ) : ℤ := 4 * x + 5

theorem fg_eval : f (g (-2)) = -27 := by
  sorry

end NUMINAMATH_GPT_fg_eval_l2051_205124


namespace NUMINAMATH_GPT_probability_of_all_female_l2051_205131

noncomputable def probability_all_females_final (females males total chosen : ℕ) : ℚ :=
  (females.choose chosen) / (total.choose chosen)

theorem probability_of_all_female:
  probability_all_females_final 5 3 8 3 = 5 / 28 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_all_female_l2051_205131


namespace NUMINAMATH_GPT_product_abc_l2051_205108

theorem product_abc 
  (a b c : ℝ)
  (h1 : a + b + c = 1) 
  (h2 : 3 * (4 * a + 2 * b + c) = 15) 
  (h3 : 5 * (9 * a + 3 * b + c) = 65) :
  a * b * c = -4 :=
by
  sorry

end NUMINAMATH_GPT_product_abc_l2051_205108


namespace NUMINAMATH_GPT_part_a_part_b_l2051_205138

-- Let p_k represent the probability that at the moment of completing the first collection, the second collection is missing exactly k crocodiles.
def p (k : ℕ) : ℝ := sorry

-- The conditions 
def totalCrocodiles : ℕ := 10
def probabilityEachEgg : ℝ := 0.1

-- Problems to prove:

-- (a) Prove that p_1 = p_2
theorem part_a : p 1 = p 2 := sorry

-- (b) Prove that p_2 > p_3 > p_4 > ... > p_10
theorem part_b : ∀ k, 2 ≤ k ∧ k < totalCrocodiles → p k > p (k + 1) := sorry

end NUMINAMATH_GPT_part_a_part_b_l2051_205138


namespace NUMINAMATH_GPT_finite_tasty_integers_l2051_205153

def is_terminating_decimal (a b : ℕ) : Prop :=
  ∃ (c : ℕ), (b = c * 2^a * 5^a)

def is_tasty (n : ℕ) : Prop :=
  n > 2 ∧ ∀ (a b : ℕ), a + b = n → (is_terminating_decimal a b ∨ is_terminating_decimal b a)

theorem finite_tasty_integers : 
  ∃ (N : ℕ), ∀ (n : ℕ), n > N → ¬ is_tasty n :=
sorry

end NUMINAMATH_GPT_finite_tasty_integers_l2051_205153


namespace NUMINAMATH_GPT_find_number_l2051_205132

theorem find_number (N : ℝ) (h : 0.1 * 0.3 * 0.5 * N = 90) : N = 6000 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2051_205132


namespace NUMINAMATH_GPT_parabola_vertex_value_of_a_l2051_205164

-- Define the conditions as given in the math problem
variables (a b c : ℤ)
def quadratic_fun (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

-- Given conditions about the vertex and a point on the parabola
def vertex_condition : Prop := (quadratic_fun a b c 2 = 3)
def point_condition : Prop := (quadratic_fun a b c 1 = 0)

-- Statement to prove
theorem parabola_vertex_value_of_a : vertex_condition a b c ∧ point_condition a b c → a = -3 :=
sorry

end NUMINAMATH_GPT_parabola_vertex_value_of_a_l2051_205164


namespace NUMINAMATH_GPT_tan_theta_eq_neg_sqrt_3_l2051_205117

theorem tan_theta_eq_neg_sqrt_3 (theta : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h_a : a = (Real.cos theta, Real.sin theta))
  (h_b : b = (Real.sqrt 3, 1))
  (h_perpendicular : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.tan theta = -Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_theta_eq_neg_sqrt_3_l2051_205117


namespace NUMINAMATH_GPT_surface_area_difference_l2051_205168

theorem surface_area_difference
  (larger_cube_volume : ℝ)
  (num_smaller_cubes : ℝ)
  (smaller_cube_volume : ℝ)
  (h1 : larger_cube_volume = 125)
  (h2 : num_smaller_cubes = 125)
  (h3 : smaller_cube_volume = 1) :
  (6 * (smaller_cube_volume)^(2/3) * num_smaller_cubes) - (6 * (larger_cube_volume)^(2/3)) = 600 :=
by {
  sorry
}

end NUMINAMATH_GPT_surface_area_difference_l2051_205168


namespace NUMINAMATH_GPT_no_solution_eqn_l2051_205165

theorem no_solution_eqn (m : ℝ) :
  ¬ ∃ x : ℝ, (3 - 2 * x) / (x - 3) - (m * x - 2) / (3 - x) = -1 ↔ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_eqn_l2051_205165


namespace NUMINAMATH_GPT_deriv_prob1_deriv_prob2_l2051_205115

noncomputable def prob1 (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem deriv_prob1 : ∀ x, deriv prob1 x = -x * Real.sin x :=
by 
  sorry

noncomputable def prob2 (x : ℝ) : ℝ := x / (Real.exp x - 1)

theorem deriv_prob2 : ∀ x, x ≠ 0 → deriv prob2 x = (Real.exp x * (1 - x) - 1) / (Real.exp x - 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_deriv_prob1_deriv_prob2_l2051_205115


namespace NUMINAMATH_GPT_red_large_toys_count_l2051_205161

def percentage_red : ℝ := 0.25
def percentage_green : ℝ := 0.20
def percentage_blue : ℝ := 0.15
def percentage_yellow : ℝ := 0.25
def percentage_orange : ℝ := 0.15

def red_small : ℝ := 0.06
def red_medium : ℝ := 0.08
def red_large : ℝ := 0.07
def red_extra_large : ℝ := 0.04

def green_small : ℝ := 0.04
def green_medium : ℝ := 0.07
def green_large : ℝ := 0.05
def green_extra_large : ℝ := 0.04

def blue_small : ℝ := 0.06
def blue_medium : ℝ := 0.03
def blue_large : ℝ := 0.04
def blue_extra_large : ℝ := 0.02

def yellow_small : ℝ := 0.08
def yellow_medium : ℝ := 0.10
def yellow_large : ℝ := 0.05
def yellow_extra_large : ℝ := 0.02

def orange_small : ℝ := 0.09
def orange_medium : ℝ := 0.06
def orange_large : ℝ := 0.05
def orange_extra_large : ℝ := 0.05

def green_large_count : ℕ := 47

noncomputable def total_green_toys := green_large_count / green_large

noncomputable def total_toys := total_green_toys / percentage_green

noncomputable def red_large_toys := total_toys * red_large

theorem red_large_toys_count : red_large_toys = 329 := by
  sorry

end NUMINAMATH_GPT_red_large_toys_count_l2051_205161


namespace NUMINAMATH_GPT_rectangle_area_l2051_205173

theorem rectangle_area
    (w l : ℕ)
    (h₁ : 28 = 2 * (l + w))
    (h₂ : w = 6) : l * w = 48 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2051_205173


namespace NUMINAMATH_GPT_simplified_equation_has_solution_l2051_205183

theorem simplified_equation_has_solution (n : ℤ) :
  (∃ x y z : ℤ, x^2 + y^2 + z^2 - x * y - y * z - z * x = n) →
  (∃ x y : ℤ, x^2 + y^2 - x * y = n) :=
by
  intros h
  exact sorry

end NUMINAMATH_GPT_simplified_equation_has_solution_l2051_205183


namespace NUMINAMATH_GPT_gain_percent_l2051_205103

theorem gain_percent (CP SP : ℝ) (hCP : CP = 20) (hSP : SP = 35) : 
  (SP - CP) / CP * 100 = 75 :=
by
  rw [hCP, hSP]
  sorry

end NUMINAMATH_GPT_gain_percent_l2051_205103


namespace NUMINAMATH_GPT_sum_of_cubes_four_consecutive_integers_l2051_205175

theorem sum_of_cubes_four_consecutive_integers (n : ℕ) (h : (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 = 11534) :
  (n-1)^3 + n^3 + (n+1)^3 + (n+2)^3 = 74836 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_four_consecutive_integers_l2051_205175


namespace NUMINAMATH_GPT_ratio_shorter_to_longer_l2051_205182

theorem ratio_shorter_to_longer (total_length shorter_length longer_length : ℕ) (h1 : total_length = 40) 
(h2 : shorter_length = 16) (h3 : longer_length = total_length - shorter_length) : 
(shorter_length / Nat.gcd shorter_length longer_length) / (longer_length / Nat.gcd shorter_length longer_length) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_shorter_to_longer_l2051_205182


namespace NUMINAMATH_GPT_min_value_expression_l2051_205171

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (c : ℝ), c = 216 ∧
    ∀ (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c), 
      ( (a^2 + 3*a + 2) * (b^2 + 3*b + 2) * (c^2 + 3*c + 2) / (a * b * c) ) ≥ 216 := 
sorry

end NUMINAMATH_GPT_min_value_expression_l2051_205171


namespace NUMINAMATH_GPT_find_g_inverse_sum_l2051_205129

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 2 then x^2 - 2 * x + 2 else 3 - x

theorem find_g_inverse_sum :
  (∃ x, g x = -2 ∧ x = 5) ∧
  (∃ x, g x = 0 ∧ x = 3) ∧
  (∃ x, g x = 2 ∧ x = 0) ∧
  (5 + 3 + 0 = 8) := by
  sorry

end NUMINAMATH_GPT_find_g_inverse_sum_l2051_205129


namespace NUMINAMATH_GPT_hanoi_tower_l2051_205140

noncomputable def move_all_disks (n : ℕ) : Prop := 
  ∀ (A B C : Type), 
  (∃ (move : A → B), move = sorry) ∧ -- Only one disk can be moved
  (∃ (can_place : A → A → Prop), can_place = sorry) -- A disk cannot be placed on top of a smaller disk 
  → ∃ (u_n : ℕ), u_n = 2^n - 1 -- Formula for minimum number of steps

theorem hanoi_tower : ∀ n : ℕ, move_all_disks n :=
by sorry

end NUMINAMATH_GPT_hanoi_tower_l2051_205140


namespace NUMINAMATH_GPT_coloring_problem_l2051_205134

def condition (m n : ℕ) : Prop :=
  2 ≤ m ∧ m ≤ 31 ∧ 2 ≤ n ∧ n ≤ 31 ∧ m ≠ n ∧ m % n = 0

def color (f : ℕ → ℕ) : Prop :=
  ∀ m n, condition m n → f m ≠ f n

theorem coloring_problem :
  ∃ (k : ℕ) (f : ℕ → ℕ), (∀ n, 2 ≤ n ∧ n ≤ 31 → f n ≤ k) ∧ color f ∧ k = 4 :=
by
  sorry

end NUMINAMATH_GPT_coloring_problem_l2051_205134


namespace NUMINAMATH_GPT_greatest_divisor_of_546_smaller_than_30_and_factor_of_126_l2051_205116

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem greatest_divisor_of_546_smaller_than_30_and_factor_of_126 :
  ∃ (d : ℕ), d < 30 ∧ is_factor d 546 ∧ is_factor d 126 ∧ ∀ e : ℕ, e < 30 ∧ is_factor e 546 ∧ is_factor e 126 → e ≤ d := 
sorry

end NUMINAMATH_GPT_greatest_divisor_of_546_smaller_than_30_and_factor_of_126_l2051_205116


namespace NUMINAMATH_GPT_total_chairs_calc_l2051_205146

-- Defining the condition of having 27 rows
def rows : ℕ := 27

-- Defining the condition of having 16 chairs per row
def chairs_per_row : ℕ := 16

-- Stating the theorem that the total number of chairs is 432
theorem total_chairs_calc : rows * chairs_per_row = 432 :=
by
  sorry

end NUMINAMATH_GPT_total_chairs_calc_l2051_205146


namespace NUMINAMATH_GPT_initial_total_quantity_l2051_205191

theorem initial_total_quantity(milk_ratio water_ratio : ℕ) (W : ℕ) (x : ℕ) (h1 : milk_ratio = 3) (h2 : water_ratio = 1) (h3 : W = 100) (h4 : 3 * x / (x + 100) = 1 / 3) :
    4 * x = 50 :=
by
  sorry

end NUMINAMATH_GPT_initial_total_quantity_l2051_205191


namespace NUMINAMATH_GPT_find_principal_l2051_205195

theorem find_principal (R : ℝ) : ∃ P : ℝ, (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 100 :=
by {
  use 200,
  sorry
}

end NUMINAMATH_GPT_find_principal_l2051_205195


namespace NUMINAMATH_GPT_fraction_ordering_l2051_205159

theorem fraction_ordering (x y : ℝ) (hx : x < 0) (hy : 0 < y ∧ y < 1) :
  (1 / x) < (y / x) ∧ (y / x) < (y^2 / x) :=
by
  sorry

end NUMINAMATH_GPT_fraction_ordering_l2051_205159


namespace NUMINAMATH_GPT_books_total_correct_l2051_205163

-- Define the constants for the number of books obtained each day
def books_day1 : ℕ := 54
def books_day2_total : ℕ := 23
def books_day2_kept : ℕ := 12
def books_day3_multiplier : ℕ := 3

-- Calculate the total number of books obtained each day
def books_day3 := books_day3_multiplier * books_day2_total
def total_books := books_day1 + books_day2_kept + books_day3

-- The theorem to prove
theorem books_total_correct : total_books = 135 := by
  sorry

end NUMINAMATH_GPT_books_total_correct_l2051_205163


namespace NUMINAMATH_GPT_max_value_x1_x2_l2051_205197

noncomputable def f (x : ℝ) := 1 - Real.sqrt (2 - 3 * x)
noncomputable def g (x : ℝ) := 2 * Real.log x

theorem max_value_x1_x2 (x1 x2 : ℝ) (h1 : x1 ≤ 2 / 3) (h2 : x2 > 0) (h3 : x1 - x2 = (1 - Real.sqrt (2 - 3 * x1)) - (2 * Real.log x2)) :
  x1 - x2 ≤ -25 / 48 :=
sorry

end NUMINAMATH_GPT_max_value_x1_x2_l2051_205197


namespace NUMINAMATH_GPT_Nicki_runs_30_miles_per_week_in_second_half_l2051_205162

/-
  Nicki ran 20 miles per week for the first half of the year.
  There are 26 weeks in each half of the year.
  She ran a total of 1300 miles for the year.
  Prove that Nicki ran 30 miles per week in the second half of the year.
-/

theorem Nicki_runs_30_miles_per_week_in_second_half (weekly_first_half : ℕ) (weeks_per_half : ℕ) (total_miles : ℕ) :
  weekly_first_half = 20 → weeks_per_half = 26 → total_miles = 1300 → 
  (total_miles - (weekly_first_half * weeks_per_half)) / weeks_per_half = 30 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_Nicki_runs_30_miles_per_week_in_second_half_l2051_205162


namespace NUMINAMATH_GPT_car_speed_l2051_205170

theorem car_speed (uses_one_gallon_per_30_miles : ∀ d : ℝ, d = 30 → d / 30 ≥ 1)
    (full_tank : ℝ := 10)
    (travel_time : ℝ := 5)
    (fraction_of_tank_used : ℝ := 0.8333333333333334)
    (speed : ℝ := 50) :
  let amount_of_gasoline_used := fraction_of_tank_used * full_tank
  let distance_traveled := amount_of_gasoline_used * 30
  speed = distance_traveled / travel_time :=
by
  sorry

end NUMINAMATH_GPT_car_speed_l2051_205170


namespace NUMINAMATH_GPT_gcd_equivalence_l2051_205109

theorem gcd_equivalence : 
  let m := 2^2100 - 1
  let n := 2^2091 + 31
  gcd m n = gcd (2^2091 + 31) 511 :=
by
  sorry

end NUMINAMATH_GPT_gcd_equivalence_l2051_205109


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l2051_205155

-- Define the parameters and conditions
def speed_in_still_water (v_m : ℝ) (v_s : ℝ) : Prop :=
    (v_m + v_s = 5) ∧  -- downstream condition
    (v_m - v_s = 7)    -- upstream condition

-- The theorem statement
theorem speed_of_man_in_still_water : 
  ∃ v_m v_s : ℝ, speed_in_still_water v_m v_s ∧ v_m = 6 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l2051_205155


namespace NUMINAMATH_GPT_ed_total_pets_l2051_205123

theorem ed_total_pets (num_dogs num_cats : ℕ) (h_dogs : num_dogs = 2) (h_cats : num_cats = 3) :
  ∃ num_fish : ℕ, (num_fish = 2 * (num_dogs + num_cats)) ∧ (num_dogs + num_cats + num_fish) = 15 :=
by
  sorry

end NUMINAMATH_GPT_ed_total_pets_l2051_205123


namespace NUMINAMATH_GPT_largest_x_eq_neg5_l2051_205190

theorem largest_x_eq_neg5 (x : ℝ) (h : x ≠ 7) : (x^2 - 5*x - 84)/(x - 7) = 2/(x + 6) → x ≤ -5 := 
sorry

end NUMINAMATH_GPT_largest_x_eq_neg5_l2051_205190


namespace NUMINAMATH_GPT_average_rate_of_change_is_7_l2051_205144

-- Define the function
def f (x : ℝ) : ℝ := x^3 + 1

-- Define the interval
def a : ℝ := 1
def b : ℝ := 2

-- Define the proof problem
theorem average_rate_of_change_is_7 : 
  ((f b - f a) / (b - a)) = 7 :=
by 
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_average_rate_of_change_is_7_l2051_205144


namespace NUMINAMATH_GPT_width_of_jordan_rectangle_l2051_205122

def carol_length := 5
def carol_width := 24
def jordan_length := 2
def jordan_area := carol_length * carol_width

theorem width_of_jordan_rectangle : ∃ (w : ℝ), jordan_length * w = jordan_area ∧ w = 60 :=
by
  use 60
  simp [carol_length, carol_width, jordan_length, jordan_area]
  sorry

end NUMINAMATH_GPT_width_of_jordan_rectangle_l2051_205122


namespace NUMINAMATH_GPT_paintings_in_four_weeks_l2051_205178

def weekly_hours := 30
def hours_per_painting := 3
def weeks := 4

theorem paintings_in_four_weeks (w_hours : ℕ) (h_per_painting : ℕ) (n_weeks : ℕ) (result : ℕ) :
  w_hours = weekly_hours →
  h_per_painting = hours_per_painting →
  n_weeks = weeks →
  result = (w_hours / h_per_painting) * n_weeks →
  result = 40 :=
by
  intros
  sorry

end NUMINAMATH_GPT_paintings_in_four_weeks_l2051_205178


namespace NUMINAMATH_GPT_otimes_2_5_l2051_205188

def otimes (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem otimes_2_5 : otimes 2 5 = 23 :=
by
  sorry

end NUMINAMATH_GPT_otimes_2_5_l2051_205188


namespace NUMINAMATH_GPT_solution_set_of_absolute_value_inequality_l2051_205143

theorem solution_set_of_absolute_value_inequality :
  { x : ℝ | |x + 1| - |x - 2| > 1 } = { x : ℝ | 1 < x } :=
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_absolute_value_inequality_l2051_205143


namespace NUMINAMATH_GPT_triangle_equilateral_of_angles_and_intersecting_segments_l2051_205100

theorem triangle_equilateral_of_angles_and_intersecting_segments
    (A B C : Type) (angle_A : ℝ) (intersect_at_one_point : Prop)
    (angle_M_bisects : Prop) (N_is_median : Prop) (L_is_altitude : Prop) :
  angle_A = 60 ∧ angle_M_bisects ∧ N_is_median ∧ L_is_altitude ∧ intersect_at_one_point → 
  ∀ (angle_B angle_C : ℝ), angle_B = 60 ∧ angle_C = 60 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_triangle_equilateral_of_angles_and_intersecting_segments_l2051_205100


namespace NUMINAMATH_GPT_horse_catch_up_l2051_205152

theorem horse_catch_up :
  ∀ (x : ℕ), (240 * x = 150 * (x + 12)) → x = 20 :=
by
  intros x h
  have : 240 * x = 150 * x + 1800 := by sorry
  have : 240 * x - 150 * x = 1800 := by sorry
  have : 90 * x = 1800 := by sorry
  have : x = 1800 / 90 := by sorry
  have : x = 20 := by sorry
  exact this

end NUMINAMATH_GPT_horse_catch_up_l2051_205152


namespace NUMINAMATH_GPT_smallest_n_leq_l2051_205101

theorem smallest_n_leq (n : ℤ) : (n ^ 2 - 13 * n + 40 ≤ 0) → (n = 5) :=
sorry

end NUMINAMATH_GPT_smallest_n_leq_l2051_205101


namespace NUMINAMATH_GPT_julia_internet_speed_l2051_205192

theorem julia_internet_speed
  (songs : ℕ) (song_size : ℕ) (time_sec : ℕ)
  (h_songs : songs = 7200)
  (h_song_size : song_size = 5)
  (h_time_sec : time_sec = 1800) :
  songs * song_size / time_sec = 20 := by
  sorry

end NUMINAMATH_GPT_julia_internet_speed_l2051_205192


namespace NUMINAMATH_GPT_unique_sequence_and_a_2002_l2051_205141

-- Define the sequence (a_n)
noncomputable def a : ℕ → ℕ := -- define the correct sequence based on conditions
  -- we would define a such as in the constructive steps in the solution, but here's a placeholder
  sorry

-- Prove the uniqueness and finding a_2002
theorem unique_sequence_and_a_2002 :
  (∀ n : ℕ, ∃! (i j k : ℕ), n = a i + 2 * a j + 4 * a k) ∧ a 2002 = 1227132168 :=
by
  sorry

end NUMINAMATH_GPT_unique_sequence_and_a_2002_l2051_205141


namespace NUMINAMATH_GPT_ratio_avg_eq_42_l2051_205177

theorem ratio_avg_eq_42 (a b c d : ℕ)
  (h1 : ∃ k : ℕ, a = 2 * k ∧ b = 3 * k ∧ c = 4 * k ∧ d = 5 * k)
  (h2 : (a + b + c + d) / 4 = 42) : a = 24 :=
by sorry

end NUMINAMATH_GPT_ratio_avg_eq_42_l2051_205177


namespace NUMINAMATH_GPT_find_f_2_l2051_205105

theorem find_f_2 (f : ℝ → ℝ) (h : ∀ x, f (1 / x + 1) = 2 * x + 3) : f 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2_l2051_205105


namespace NUMINAMATH_GPT_problem_div_expansion_l2051_205174

theorem problem_div_expansion (m : ℝ) : ((2 * m^2 - m)^2) / (-m^2) = -4 * m^2 + 4 * m - 1 := 
by sorry

end NUMINAMATH_GPT_problem_div_expansion_l2051_205174


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2051_205126

theorem geometric_sequence_sum (a_1 q n S : ℕ) (h1 : a_1 = 2) (h2 : q = 2) (h3 : S = 126) 
    (h4 : S = (a_1 * (1 - q^n)) / (1 - q)) : 
    n = 6 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2051_205126


namespace NUMINAMATH_GPT_trajectory_equation_circle_equation_l2051_205148

-- Define the variables
variables {x y r : ℝ}

-- Prove the trajectory equation of the circle center P
theorem trajectory_equation (h1 : x^2 + r^2 = 2) (h2 : y^2 + r^2 = 3) : y^2 - x^2 = 1 :=
sorry

-- Prove the equation of the circle P given the distance to the line y = x
theorem circle_equation (h : (|x - y| / Real.sqrt 2) = (Real.sqrt 2) / 2) : 
  (x = y + 1 ∨ x = y - 1) → 
  ((y + 1)^2 + x^2 = 3 ∨ (y - 1)^2 + x^2 = 3) :=
sorry

end NUMINAMATH_GPT_trajectory_equation_circle_equation_l2051_205148


namespace NUMINAMATH_GPT_area_triangle_ABC_given_conditions_l2051_205145

variable (a b c : ℝ) (A B C : ℝ)

noncomputable def area_of_triangle_ABC (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem area_triangle_ABC_given_conditions
  (habc : a = 4)
  (hbc : b + c = 5)
  (htan : Real.tan B + Real.tan C + Real.sqrt 3 = Real.sqrt 3 * (Real.tan B * Real.tan C))
  : area_of_triangle_ABC a b c (Real.pi / 3) B C = 3 * Real.sqrt 3 / 4 := 
sorry

end NUMINAMATH_GPT_area_triangle_ABC_given_conditions_l2051_205145


namespace NUMINAMATH_GPT_theorem_227_l2051_205110

theorem theorem_227 (a b c d : ℤ) (k : ℤ) (h : b ≡ c [ZMOD d]) :
  (a + b ≡ a + c [ZMOD d]) ∧
  (a - b ≡ a - c [ZMOD d]) ∧
  (a * b ≡ a * c [ZMOD d]) :=
by
  sorry

end NUMINAMATH_GPT_theorem_227_l2051_205110


namespace NUMINAMATH_GPT_domain_of_function_l2051_205135

noncomputable def domain : Set ℝ := {x | x ≥ 1/2 ∧ x ≠ 1}

theorem domain_of_function : ∀ (x : ℝ), (2 * x - 1 ≥ 0) ∧ (x ^ 2 + x - 2 ≠ 0) ↔ (x ∈ domain) :=
by 
  sorry

end NUMINAMATH_GPT_domain_of_function_l2051_205135


namespace NUMINAMATH_GPT_difference_of_squares_l2051_205136

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := 
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l2051_205136


namespace NUMINAMATH_GPT_antonio_weight_l2051_205179

-- Let A be the weight of Antonio
variable (A : ℕ)

-- Conditions:
-- 1. Antonio's sister weighs A - 12 kilograms.
-- 2. The total weight of Antonio and his sister is 88 kilograms.

theorem antonio_weight (A: ℕ) (h1: A - 12 >= 0) (h2: A + (A - 12) = 88) : A = 50 := by
  sorry

end NUMINAMATH_GPT_antonio_weight_l2051_205179


namespace NUMINAMATH_GPT_right_triangle_perimeter_l2051_205198

theorem right_triangle_perimeter
  (a b c : ℝ)
  (h_right: a^2 + b^2 = c^2)
  (h_area: (1/2) * a * b = (1/2) * c) :
  a + b + c = 2 * (Real.sqrt 2 + 1) :=
sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l2051_205198


namespace NUMINAMATH_GPT_only_solution_l2051_205130

theorem only_solution (a : ℤ) : 
  (∀ x : ℤ, x > 0 → 2 * x > 4 * x - 8 → 3 * x - a > -9 → x = 2) →
  (12 ≤ a ∧ a < 15) :=
by
  sorry

end NUMINAMATH_GPT_only_solution_l2051_205130


namespace NUMINAMATH_GPT_sum_of_first_39_natural_numbers_l2051_205114

theorem sum_of_first_39_natural_numbers : (39 * (39 + 1)) / 2 = 780 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_39_natural_numbers_l2051_205114


namespace NUMINAMATH_GPT_cos_angle_B_bounds_l2051_205128

theorem cos_angle_B_bounds {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ)
  (angle_ADC : ℝ) (angle_B : ℝ)
  (h1 : AB = 2) (h2 : BC = 3) (h3 : CD = 2) (h4 : angle_ADC = 180 - angle_B) :
  (1 / 4) < Real.cos angle_B ∧ Real.cos angle_B < (3 / 4) := 
sorry -- Proof to be provided

end NUMINAMATH_GPT_cos_angle_B_bounds_l2051_205128
