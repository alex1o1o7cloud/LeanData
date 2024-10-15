import Mathlib

namespace NUMINAMATH_GPT_rectangle_dimensions_l1480_148097

theorem rectangle_dimensions (w l : ℕ) (h : l = w + 5) (hp : 2 * l + 2 * w = 34) : w = 6 ∧ l = 11 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1480_148097


namespace NUMINAMATH_GPT_resultant_after_trebled_l1480_148051

variable (x : ℕ)

theorem resultant_after_trebled (h : x = 7) : 3 * (2 * x + 9) = 69 := by
  sorry

end NUMINAMATH_GPT_resultant_after_trebled_l1480_148051


namespace NUMINAMATH_GPT_reduced_price_is_55_l1480_148005

variables (P R : ℝ) (X : ℕ)

-- Conditions
def condition1 : R = 0.75 * P := sorry
def condition2 : P * X = 1100 := sorry
def condition3 : 0.75 * P * (X + 5) = 1100 := sorry

-- Theorem
theorem reduced_price_is_55 (P R : ℝ) (X : ℕ) (h1 : R = 0.75 * P) (h2 : P * X = 1100) (h3 : 0.75 * P * (X + 5) = 1100) :
  R = 55 :=
sorry

end NUMINAMATH_GPT_reduced_price_is_55_l1480_148005


namespace NUMINAMATH_GPT_mr_willson_friday_work_time_l1480_148080

theorem mr_willson_friday_work_time :
  let monday := 3 / 4
  let tuesday := 1 / 2
  let wednesday := 2 / 3
  let thursday := 5 / 6
  let total_work := 4
  let time_monday_to_thursday := monday + tuesday + wednesday + thursday
  let time_friday := total_work - time_monday_to_thursday
  time_friday * 60 = 75 :=
by
  sorry

end NUMINAMATH_GPT_mr_willson_friday_work_time_l1480_148080


namespace NUMINAMATH_GPT_Alex_age_l1480_148043

theorem Alex_age : ∃ (x : ℕ), (∃ (y : ℕ), x - 2 = y^2) ∧ (∃ (z : ℕ), x + 2 = z^3) ∧ x = 6 := by
  sorry

end NUMINAMATH_GPT_Alex_age_l1480_148043


namespace NUMINAMATH_GPT_ferris_wheel_time_l1480_148034

theorem ferris_wheel_time (R T : ℝ) (t : ℝ) (h : ℝ → ℝ) :
  R = 30 → T = 90 → (∀ t, h t = R * Real.cos ((2 * Real.pi / T) * t) + R) → h t = 45 → t = 15 :=
by
  intros hR hT hFunc hHt
  sorry

end NUMINAMATH_GPT_ferris_wheel_time_l1480_148034


namespace NUMINAMATH_GPT_boxes_sold_l1480_148052

theorem boxes_sold (start_boxes sold_boxes left_boxes : ℕ) (h1 : start_boxes = 10) (h2 : left_boxes = 5) (h3 : start_boxes - sold_boxes = left_boxes) : sold_boxes = 5 :=
by
  sorry

end NUMINAMATH_GPT_boxes_sold_l1480_148052


namespace NUMINAMATH_GPT_half_product_unique_l1480_148021

theorem half_product_unique (x : ℕ) (n k : ℕ) 
  (hn : x = n * (n + 1) / 2) (hk : x = k * (k + 1) / 2) : 
  n = k := 
sorry

end NUMINAMATH_GPT_half_product_unique_l1480_148021


namespace NUMINAMATH_GPT_sum_of_fractions_as_decimal_l1480_148068

theorem sum_of_fractions_as_decimal : (3 / 8 : ℝ) + (5 / 32) = 0.53125 := by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_as_decimal_l1480_148068


namespace NUMINAMATH_GPT_major_premise_wrong_l1480_148042

-- Definitions of the given conditions in Lean
def is_parallel_to_plane (line : Type) (plane : Type) : Prop := sorry -- Provide an appropriate definition
def contains_line (plane : Type) (line : Type) : Prop := sorry -- Provide an appropriate definition
def is_parallel_to_line (line1 : Type) (line2 : Type) : Prop := sorry -- Provide an appropriate definition

-- Given conditions
variables (b α a : Type)
variable (H1 : ¬ contains_line α b)  -- Line b is not contained in plane α
variable (H2 : contains_line α a)    -- Line a is contained in plane α
variable (H3 : is_parallel_to_plane b α) -- Line b is parallel to plane α

-- Proposition to prove: The major premise is wrong
theorem major_premise_wrong : ¬(∀ (a b : Type), is_parallel_to_plane b α → contains_line α a → is_parallel_to_line b a) :=
by
  sorry

end NUMINAMATH_GPT_major_premise_wrong_l1480_148042


namespace NUMINAMATH_GPT_trucks_needed_for_coal_transport_l1480_148088

def number_of_trucks (total_coal : ℕ) (capacity_per_truck : ℕ) (x : ℕ) : Prop :=
  capacity_per_truck * x = total_coal

theorem trucks_needed_for_coal_transport :
  number_of_trucks 47500 2500 19 :=
by
  sorry

end NUMINAMATH_GPT_trucks_needed_for_coal_transport_l1480_148088


namespace NUMINAMATH_GPT_allocation_schemes_l1480_148075

theorem allocation_schemes :
  ∃ (n : ℕ), n = 240 ∧ (∀ (volunteers : Fin 5 → Fin 4), 
    (∃ (assign : Fin 5 → Fin 4), 
      (∀ (i : Fin 4), ∃ (j : Fin 5), assign j = i)
      ∧ (∀ (k : Fin 5), assign k ≠ assign k)) 
    → true) := sorry

end NUMINAMATH_GPT_allocation_schemes_l1480_148075


namespace NUMINAMATH_GPT_union_P_Q_l1480_148022

def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | x^2 - 2*x < 0}

theorem union_P_Q : P ∪ Q = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_union_P_Q_l1480_148022


namespace NUMINAMATH_GPT_sugar_in_lollipop_l1480_148099

-- Definitions based on problem conditions
def chocolate_bars := 14
def sugar_per_bar := 10
def total_sugar := 177

-- The theorem to prove
theorem sugar_in_lollipop : total_sugar - (chocolate_bars * sugar_per_bar) = 37 :=
by
  -- we are not providing the proof, hence using sorry
  sorry

end NUMINAMATH_GPT_sugar_in_lollipop_l1480_148099


namespace NUMINAMATH_GPT_reciprocal_sqrt5_minus_2_l1480_148098

theorem reciprocal_sqrt5_minus_2 : 1 / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_sqrt5_minus_2_l1480_148098


namespace NUMINAMATH_GPT_each_child_gets_twelve_cupcakes_l1480_148012

def total_cupcakes := 96
def children := 8
def cupcakes_per_child : ℕ := total_cupcakes / children

theorem each_child_gets_twelve_cupcakes :
  cupcakes_per_child = 12 :=
by
  sorry

end NUMINAMATH_GPT_each_child_gets_twelve_cupcakes_l1480_148012


namespace NUMINAMATH_GPT_tomato_price_l1480_148073

theorem tomato_price (P : ℝ) (W : ℝ) :
  (0.9956 * 0.9 * W = P * W + 0.12 * (P * W)) → P = 0.8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tomato_price_l1480_148073


namespace NUMINAMATH_GPT_jacks_speed_is_7_l1480_148037

-- Define the constants and speeds as given in conditions
def initial_distance : ℝ := 150
def christina_speed : ℝ := 8
def lindy_speed : ℝ := 10
def lindy_total_distance : ℝ := 100

-- Hypothesis stating when the three meet
theorem jacks_speed_is_7 :
  ∃ (jack_speed : ℝ), (∃ (time : ℝ), 
    time = lindy_total_distance / lindy_speed
    ∧ christina_speed * time + jack_speed * time = initial_distance) 
  → jack_speed = 7 :=
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_jacks_speed_is_7_l1480_148037


namespace NUMINAMATH_GPT_maximum_distance_l1480_148078

-- Defining the conditions
def highway_mileage : ℝ := 12.2
def city_mileage : ℝ := 7.6
def gasoline_amount : ℝ := 22

-- Mathematical equivalent proof statement
theorem maximum_distance (h_mileage : ℝ) (g_amount : ℝ) : h_mileage = 12.2 ∧ g_amount = 22 → g_amount * h_mileage = 268.4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_maximum_distance_l1480_148078


namespace NUMINAMATH_GPT_prime_power_seven_l1480_148044

theorem prime_power_seven (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (eqn : p + 25 = q^7) : p = 103 := by
  sorry

end NUMINAMATH_GPT_prime_power_seven_l1480_148044


namespace NUMINAMATH_GPT_bases_to_make_equality_l1480_148072

theorem bases_to_make_equality (a b : ℕ) (h : 3 * a^2 + 4 * a + 2 = 9 * b + 7) : 
  (3 * a^2 + 4 * a + 2 = 342) ∧ (9 * b + 7 = 97) :=
by
  sorry

end NUMINAMATH_GPT_bases_to_make_equality_l1480_148072


namespace NUMINAMATH_GPT_ratio_of_ages_in_two_years_l1480_148023

theorem ratio_of_ages_in_two_years (S M : ℕ) 
  (h1 : M = S + 37) 
  (h2 : S = 35) : 
  (M + 2) / (S + 2) = 2 := 
by 
  -- We skip the proof steps as instructed
  sorry

end NUMINAMATH_GPT_ratio_of_ages_in_two_years_l1480_148023


namespace NUMINAMATH_GPT_price_25_bag_l1480_148010

noncomputable def price_per_bag_25 : ℝ := 28.97

def price_per_bag_5 : ℝ := 13.85
def price_per_bag_10 : ℝ := 20.42

def total_cost (p5 p10 p25 : ℝ) (n5 n10 n25 : ℕ) : ℝ :=
  n5 * p5 + n10 * p10 + n25 * p25

theorem price_25_bag :
  ∃ (p5 p10 p25 : ℝ) (n5 n10 n25 : ℕ),
    p5 = price_per_bag_5 ∧
    p10 = price_per_bag_10 ∧
    p25 = price_per_bag_25 ∧
    65 ≤ 5 * n5 + 10 * n10 + 25 * n25 ∧
    5 * n5 + 10 * n10 + 25 * n25 ≤ 80 ∧
    total_cost p5 p10 p25 n5 n10 n25 = 98.77 :=
by
  sorry

end NUMINAMATH_GPT_price_25_bag_l1480_148010


namespace NUMINAMATH_GPT_incorrect_statement_C_l1480_148039

/-- 
  Prove that the function y = -1/2 * x + 3 does not intersect the y-axis at (6,0).
-/
theorem incorrect_statement_C 
: ∀ (x y : ℝ), y = -1/2 * x + 3 → (x, y) ≠ (6, 0) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_incorrect_statement_C_l1480_148039


namespace NUMINAMATH_GPT_polynomial_divisibility_l1480_148090

theorem polynomial_divisibility (n : ℕ) (h : 0 < n) : 
  ∃ g : Polynomial ℚ, 
    (Polynomial.X + 1)^(2*n + 1) + Polynomial.X^(n + 2) = g * (Polynomial.X^2 + Polynomial.X + 1) := 
by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1480_148090


namespace NUMINAMATH_GPT_problem_l1480_148081

open Set

theorem problem (M : Set ℤ) (N : Set ℤ) (hM : M = {1, 2, 3, 4}) (hN : N = {-2, 2}) : 
  M ∩ N = {2} :=
by
  sorry

end NUMINAMATH_GPT_problem_l1480_148081


namespace NUMINAMATH_GPT_rectangle_y_coordinate_l1480_148046

theorem rectangle_y_coordinate (x1 x2 y1 A : ℝ) (h1 : x1 = -8) (h2 : x2 = 1) (h3 : y1 = 1) (h4 : A = 72)
    (hL : x2 - x1 = 9) (hA : A = 9 * (y - y1)) :
    (y = 9) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_y_coordinate_l1480_148046


namespace NUMINAMATH_GPT_french_fries_cost_is_correct_l1480_148082

def burger_cost : ℝ := 5
def soft_drink_cost : ℝ := 3
def special_burger_meal_cost : ℝ := 9.5

def french_fries_cost : ℝ :=
  special_burger_meal_cost - (burger_cost + soft_drink_cost)

theorem french_fries_cost_is_correct :
  french_fries_cost = 1.5 :=
by
  unfold french_fries_cost
  unfold special_burger_meal_cost
  unfold burger_cost
  unfold soft_drink_cost
  sorry

end NUMINAMATH_GPT_french_fries_cost_is_correct_l1480_148082


namespace NUMINAMATH_GPT_angle_sum_around_point_l1480_148006

theorem angle_sum_around_point {x : ℝ} (h : 2 * x + 210 = 360) : x = 75 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_around_point_l1480_148006


namespace NUMINAMATH_GPT_number_of_people_tasting_apple_pies_l1480_148050

/-- Sedrach's apple pie problem -/
def apple_pies : ℕ := 13
def halves_per_apple_pie : ℕ := 2
def bite_size_samples_per_half : ℕ := 5

theorem number_of_people_tasting_apple_pies :
    (apple_pies * halves_per_apple_pie * bite_size_samples_per_half) = 130 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_tasting_apple_pies_l1480_148050


namespace NUMINAMATH_GPT_value_of_expression_l1480_148065

theorem value_of_expression {a b c : ℝ} (h_eqn : a + b + c = 15)
  (h_ab_bc_ca : ab + bc + ca = 13) (h_abc : abc = 8)
  (h_roots : Polynomial.roots (Polynomial.X^3 - 15 * Polynomial.X^2 + 13 * Polynomial.X - 8) = {a, b, c}) :
  (a / (1/a + b*c)) + (b / (1/b + c*a)) + (c / (1/c + a*b)) = 199/9 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l1480_148065


namespace NUMINAMATH_GPT_least_possible_value_of_z_minus_w_l1480_148040

variable (x y z w k m : Int)
variable (h1 : Even x)
variable (h2 : Odd y)
variable (h3 : Odd z)
variable (h4 : ∃ n : Int, w = - (2 * n + 1) / 3)
variable (h5 : w < x)
variable (h6 : x < y)
variable (h7 : y < z)
variable (h8 : 0 < k)
variable (h9 : (y - x) > k)
variable (h10 : 0 < m)
variable (h11 : (z - w) > m)
variable (h12 : k > m)

theorem least_possible_value_of_z_minus_w
  : z - w = 6 := sorry

end NUMINAMATH_GPT_least_possible_value_of_z_minus_w_l1480_148040


namespace NUMINAMATH_GPT_smallest_possible_n_l1480_148045

theorem smallest_possible_n (n : ℕ) (h : ∃ k : ℕ, 15 * n - 2 = 11 * k) : n % 11 = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_n_l1480_148045


namespace NUMINAMATH_GPT_calculate_wholesale_price_l1480_148053

noncomputable def retail_price : ℝ := 108

noncomputable def selling_price (retail_price : ℝ) : ℝ := retail_price * 0.90

noncomputable def selling_price_alt (wholesale_price : ℝ) : ℝ := wholesale_price * 1.20

theorem calculate_wholesale_price (W : ℝ) (R : ℝ) (SP : ℝ)
  (hR : R = 108)
  (hSP1 : SP = selling_price R)
  (hSP2 : SP = selling_price_alt W) : W = 81 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_calculate_wholesale_price_l1480_148053


namespace NUMINAMATH_GPT_two_digit_numbers_sum_reversed_l1480_148014

theorem two_digit_numbers_sum_reversed (a b : ℕ) (h₁ : 0 ≤ a) (h₂ : a ≤ 9) (h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : a + b = 12) :
  ∃ n : ℕ, n = 7 := 
sorry

end NUMINAMATH_GPT_two_digit_numbers_sum_reversed_l1480_148014


namespace NUMINAMATH_GPT_triangle_obtuse_l1480_148041

theorem triangle_obtuse (a b c : ℝ) (A B C : ℝ) 
  (hBpos : 0 < B) 
  (hBpi : B < Real.pi) 
  (sin_C_lt_cos_A_sin_B : Real.sin C / Real.sin B < Real.cos A) 
  (hC_eq : C = A + B) 
  (ha2 : A + B + C = Real.pi) :
  B > Real.pi / 2 := 
sorry

end NUMINAMATH_GPT_triangle_obtuse_l1480_148041


namespace NUMINAMATH_GPT_average_temperature_second_to_fifth_days_l1480_148030

variable (T1 T2 T3 T4 T5 : ℝ)

theorem average_temperature_second_to_fifth_days 
  (h1 : (T1 + T2 + T3 + T4) / 4 = 58)
  (h2 : T1 / T5 = 7 / 8)
  (h3 : T5 = 32) :
  (T2 + T3 + T4 + T5) / 4 = 59 :=
by
  sorry

end NUMINAMATH_GPT_average_temperature_second_to_fifth_days_l1480_148030


namespace NUMINAMATH_GPT_jason_messages_l1480_148002

theorem jason_messages :
  ∃ M : ℕ, (M + M / 2 + 150) / 5 = 96 ∧ M = 220 := by
  sorry

end NUMINAMATH_GPT_jason_messages_l1480_148002


namespace NUMINAMATH_GPT_range_m_l1480_148025

open Real

theorem range_m (m : ℝ)
  (hP : ¬ (∃ x : ℝ, m * x^2 + 1 ≤ 0))
  (hQ : ¬ (∃ x : ℝ, x^2 + m * x + 1 < 0)) :
  0 ≤ m ∧ m ≤ 2 := 
sorry

end NUMINAMATH_GPT_range_m_l1480_148025


namespace NUMINAMATH_GPT_magnitude_of_z_l1480_148056

theorem magnitude_of_z (z : ℂ) (h : z * (1 + 2 * Complex.I) + Complex.I = 0) : 
  Complex.abs z = Real.sqrt (5) / 5 := 
sorry

end NUMINAMATH_GPT_magnitude_of_z_l1480_148056


namespace NUMINAMATH_GPT_L_shape_area_and_perimeter_l1480_148064

def rectangle1_length := 0.5
def rectangle1_width := 0.3
def rectangle2_length := 0.2
def rectangle2_width := 0.5

def area_rectangle1 := rectangle1_length * rectangle1_width
def area_rectangle2 := rectangle2_length * rectangle2_width
def total_area := area_rectangle1 + area_rectangle2

def perimeter_L_shape := rectangle1_length + rectangle1_width + rectangle1_width + rectangle2_length + rectangle2_length + rectangle2_width

theorem L_shape_area_and_perimeter :
  total_area = 0.25 ∧ perimeter_L_shape = 2.0 :=
by
  sorry

end NUMINAMATH_GPT_L_shape_area_and_perimeter_l1480_148064


namespace NUMINAMATH_GPT_min_abs_sum_of_products_l1480_148011

noncomputable def g (x : ℝ) : ℝ := x^4 + 10*x^3 + 29*x^2 + 30*x + 9

theorem min_abs_sum_of_products (w : Fin 4 → ℝ) (h_roots : ∀ i, g (w i) = 0)
  : ∃ a b c d : Fin 4, a ≠ b ∧ c ≠ d ∧ (∀ i j, i ≠ j → a ≠ i ∧ b ≠ i ∧ c ≠ i ∧ d ≠ i → a ≠ j ∧ b ≠ j ∧ c ≠ j ∧ d ≠ j) ∧
    |w a * w b + w c * w d| = 6 :=
sorry

end NUMINAMATH_GPT_min_abs_sum_of_products_l1480_148011


namespace NUMINAMATH_GPT_xy_sufficient_not_necessary_l1480_148028

theorem xy_sufficient_not_necessary (x y : ℝ) :
  (xy ≠ 6) → (x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → xy ≠ 6) := by
  sorry

end NUMINAMATH_GPT_xy_sufficient_not_necessary_l1480_148028


namespace NUMINAMATH_GPT_math_problem_l1480_148018

theorem math_problem : 2 - (-3)^2 - 4 - (-5) - 6^2 - (-7) = -35 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l1480_148018


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l1480_148032

theorem symmetric_point_coordinates (a b : ℝ) (hp : (3, 4) = (a + 3, b + 4)) :
  (a, b) = (5, 2) :=
  sorry

end NUMINAMATH_GPT_symmetric_point_coordinates_l1480_148032


namespace NUMINAMATH_GPT_find_x_l1480_148048

theorem find_x (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x)
    (h3 : 1 / a + 1 / b = 1) : x = 6 :=
sorry

end NUMINAMATH_GPT_find_x_l1480_148048


namespace NUMINAMATH_GPT_adjustments_to_equal_boys_and_girls_l1480_148060

theorem adjustments_to_equal_boys_and_girls (n : ℕ) :
  let initial_boys := 40
  let initial_girls := 0
  let boys_after_n := initial_boys - 3 * n
  let girls_after_n := initial_girls + 2 * n
  boys_after_n = girls_after_n → n = 8 :=
by
  sorry

end NUMINAMATH_GPT_adjustments_to_equal_boys_and_girls_l1480_148060


namespace NUMINAMATH_GPT_Mark_less_than_Craig_l1480_148087

-- Definitions for the conditions
def Dave_weight : ℕ := 175
def Dave_bench_press : ℕ := Dave_weight * 3
def Craig_bench_press : ℕ := (20 * Dave_bench_press) / 100
def Mark_bench_press : ℕ := 55

-- The theorem to be proven
theorem Mark_less_than_Craig : Craig_bench_press - Mark_bench_press = 50 :=
by
  sorry

end NUMINAMATH_GPT_Mark_less_than_Craig_l1480_148087


namespace NUMINAMATH_GPT_problems_per_hour_l1480_148057

theorem problems_per_hour :
  ∀ (mathProblems spellingProblems totalHours problemsPerHour : ℕ), 
    mathProblems = 36 →
    spellingProblems = 28 →
    totalHours = 8 →
    (mathProblems + spellingProblems) / totalHours = problemsPerHour →
    problemsPerHour = 8 :=
by
  intros
  subst_vars
  sorry

end NUMINAMATH_GPT_problems_per_hour_l1480_148057


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1480_148031

noncomputable def fifth_term_of_arithmetic_sequence (x y : ℝ)
  (h1 : 2 * x + y = 2 * x + y)
  (h2 : 2 * x - y = 2 * x + y - 2 * y)
  (h3 : 2 * x * y = 2 * x - 2 * y - 2 * y)
  (h4 : 2 * x / y = 2 * x * y - 5 * y^2 - 2 * y)
  : ℝ :=
(2 * x / y) - 2 * y

theorem arithmetic_sequence_fifth_term (x y : ℝ)
  (h1 : 2 * x + y = 2 * x + y)
  (h2 : 2 * x - y = 2 * x + y - 2 * y)
  (h3 : 2 * x * y = 2 * x - 2 * y - 2 * y)
  (h4 : 2 * x / y = 2 * x * y - 5 * y^2 - 2 * y)
  : fifth_term_of_arithmetic_sequence x y h1 h2 h3 h4 = -77 / 10 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1480_148031


namespace NUMINAMATH_GPT_cary_needs_6_weekends_l1480_148067

variable (shoe_cost : ℕ)
variable (current_savings : ℕ)
variable (earn_per_lawn : ℕ)
variable (lawns_per_weekend : ℕ)
variable (w : ℕ)

theorem cary_needs_6_weekends
    (h1 : shoe_cost = 120)
    (h2 : current_savings = 30)
    (h3 : earn_per_lawn = 5)
    (h4 : lawns_per_weekend = 3)
    (h5 : w * (earn_per_lawn * lawns_per_weekend) = shoe_cost - current_savings) :
    w = 6 :=
by sorry

end NUMINAMATH_GPT_cary_needs_6_weekends_l1480_148067


namespace NUMINAMATH_GPT_factorize_expression_l1480_148001

theorem factorize_expression (a b : ℝ) : 
  a^3 + 2 * a^2 * b + a * b^2 = a * (a + b)^2 := by sorry

end NUMINAMATH_GPT_factorize_expression_l1480_148001


namespace NUMINAMATH_GPT_complementary_angle_of_60_l1480_148020

theorem complementary_angle_of_60 (a : ℝ) : 
  (∀ (a b : ℝ), a + b = 180 → a = 60 → b = 120) := 
by
  sorry

end NUMINAMATH_GPT_complementary_angle_of_60_l1480_148020


namespace NUMINAMATH_GPT_wrapping_paper_needs_l1480_148095

theorem wrapping_paper_needs :
  let first_present := 2
  let second_present := (3 / 4) * first_present
  let third_present := first_present + second_present
  first_present + second_present + third_present = 7 := by
  let first_present := 2
  let second_present := (3 / 4) * first_present
  let third_present := first_present + second_present
  sorry

end NUMINAMATH_GPT_wrapping_paper_needs_l1480_148095


namespace NUMINAMATH_GPT_positive_integer_solution_exists_l1480_148015

theorem positive_integer_solution_exists (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h_eq : x^2 = y^2 + 7 * y + 6) : (x, y) = (6, 3) := 
sorry

end NUMINAMATH_GPT_positive_integer_solution_exists_l1480_148015


namespace NUMINAMATH_GPT_total_GDP_l1480_148061

noncomputable def GDP_first_quarter : ℝ := 232
noncomputable def GDP_fourth_quarter : ℝ := 241

theorem total_GDP (x y : ℝ) (h1 : GDP_first_quarter < x)
                  (h2 : x < y) (h3 : y < GDP_fourth_quarter)
                  (h4 : (x + y) / 2 = (GDP_first_quarter + x + y + GDP_fourth_quarter) / 4) :
  GDP_first_quarter + x + y + GDP_fourth_quarter = 946 :=
by
  sorry

end NUMINAMATH_GPT_total_GDP_l1480_148061


namespace NUMINAMATH_GPT_quadratic_real_roots_range_l1480_148007

theorem quadratic_real_roots_range (m : ℝ) : (∃ x y : ℝ, x ≠ y ∧ mx^2 + 2*x + 1 = 0 ∧ yx^2 + 2*y + 1 = 0) → m ≤ 1 ∧ m ≠ 0 :=
by 
sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_l1480_148007


namespace NUMINAMATH_GPT_correct_equation_l1480_148077

theorem correct_equation (x : ℝ) (hx : x > 80) : 
  353 / (x - 80) - 353 / x = 5 / 3 :=
sorry

end NUMINAMATH_GPT_correct_equation_l1480_148077


namespace NUMINAMATH_GPT_decimal_sum_difference_l1480_148085

theorem decimal_sum_difference :
  (0.5 - 0.03 + 0.007 + 0.0008 = 0.4778) :=
by
  sorry

end NUMINAMATH_GPT_decimal_sum_difference_l1480_148085


namespace NUMINAMATH_GPT_find_x_l1480_148055

theorem find_x (x : ℝ) (a b : ℝ × ℝ) (h : a = (Real.cos (3 * x / 2), Real.sin (3 * x / 2)) ∧ b = (Real.cos (x / 2), -Real.sin (x / 2)) ∧ (a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2 = 1 ∧ 0 ≤ x ∧ x ≤ Real.pi)  :
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1480_148055


namespace NUMINAMATH_GPT_count_right_triangles_l1480_148079

theorem count_right_triangles: 
  ∃ n : ℕ, n = 9 ∧ ∃ (a b : ℕ), a^2 + b^2 = (b+2)^2 ∧ b < 100 ∧ a > 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_GPT_count_right_triangles_l1480_148079


namespace NUMINAMATH_GPT_calculate_a_plus_b_l1480_148035

theorem calculate_a_plus_b (a b : ℝ) (h1 : 3 = a + b / 2) (h2 : 2 = a + b / 4) : a + b = 5 :=
by
  sorry

end NUMINAMATH_GPT_calculate_a_plus_b_l1480_148035


namespace NUMINAMATH_GPT_probability_at_least_one_passes_l1480_148036

theorem probability_at_least_one_passes (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 :=
by sorry

end NUMINAMATH_GPT_probability_at_least_one_passes_l1480_148036


namespace NUMINAMATH_GPT_cloves_of_garlic_needed_l1480_148013

def cloves_needed_for_vampires (vampires : ℕ) : ℕ :=
  (vampires * 3) / 2

def cloves_needed_for_wights (wights : ℕ) : ℕ :=
  (wights * 3) / 3

def cloves_needed_for_vampire_bats (vampire_bats : ℕ) : ℕ :=
  (vampire_bats * 3) / 8

theorem cloves_of_garlic_needed (vampires wights vampire_bats : ℕ) :
  cloves_needed_for_vampires 30 + cloves_needed_for_wights 12 + 
  cloves_needed_for_vampire_bats 40 = 72 :=
by
  sorry

end NUMINAMATH_GPT_cloves_of_garlic_needed_l1480_148013


namespace NUMINAMATH_GPT_find_second_month_sale_l1480_148096

/-- Given sales for specific months and required sales goal -/
def sales_1 := 4000
def sales_3 := 5689
def sales_4 := 7230
def sales_5 := 6000
def sales_6 := 12557
def avg_goal := 7000
def months := 6

theorem find_second_month_sale (x2 : ℕ) :
  (sales_1 + x2 + sales_3 + sales_4 + sales_5 + sales_6) / months = avg_goal →
  x2 = 6524 :=
by
  sorry

end NUMINAMATH_GPT_find_second_month_sale_l1480_148096


namespace NUMINAMATH_GPT_fraction_value_l1480_148024

theorem fraction_value : (1 - 1 / 4) / (1 - 1 / 3) = 9 / 8 := 
by sorry

end NUMINAMATH_GPT_fraction_value_l1480_148024


namespace NUMINAMATH_GPT_point_inside_circle_l1480_148089

theorem point_inside_circle (a : ℝ) :
  ((1 - a) ^ 2 + (1 + a) ^ 2 < 4) → (-1 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_point_inside_circle_l1480_148089


namespace NUMINAMATH_GPT_more_time_in_swamp_l1480_148016

theorem more_time_in_swamp (a b c : ℝ) 
  (h1 : a + b + c = 4) 
  (h2 : 2 * a + 4 * b + 6 * c = 15) : a > c :=
by {
  sorry
}

end NUMINAMATH_GPT_more_time_in_swamp_l1480_148016


namespace NUMINAMATH_GPT_store_sells_2_kg_per_week_l1480_148047

def packets_per_week := 20
def grams_per_packet := 100
def grams_per_kg := 1000
def kg_per_week (p : Nat) (gr_per_pkt : Nat) (gr_per_kg : Nat) : Nat :=
  (p * gr_per_pkt) / gr_per_kg

theorem store_sells_2_kg_per_week :
  kg_per_week packets_per_week grams_per_packet grams_per_kg = 2 :=
  sorry

end NUMINAMATH_GPT_store_sells_2_kg_per_week_l1480_148047


namespace NUMINAMATH_GPT_hulk_first_jump_more_than_500_l1480_148086

def hulk_jumping_threshold : Prop :=
  ∃ n : ℕ, (3^n > 500) ∧ (∀ m < n, 3^m ≤ 500)

theorem hulk_first_jump_more_than_500 : ∃ n : ℕ, n = 6 ∧ hulk_jumping_threshold :=
  sorry

end NUMINAMATH_GPT_hulk_first_jump_more_than_500_l1480_148086


namespace NUMINAMATH_GPT_no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_no_terms_of_form_x_pow_2m_infinitely_many_terms_of_form_x_pow_2m_plus_1_l1480_148000

theorem no_perfect_squares_in_ap (n x : ℤ) : ¬(3 * n + 2 = x^2) :=
sorry

theorem infinitely_many_perfect_cubes_in_ap : ∃ᶠ n in Filter.atTop, ∃ x : ℤ, 3 * n + 2 = x^3 :=
sorry

theorem no_terms_of_form_x_pow_2m (n x : ℤ) (m : ℕ) : 3 * n + 2 ≠ x^(2 * m) :=
sorry

theorem infinitely_many_terms_of_form_x_pow_2m_plus_1 (m : ℕ) : ∃ᶠ n in Filter.atTop, ∃ x : ℤ, 3 * n + 2 = x^(2 * m + 1) :=
sorry

end NUMINAMATH_GPT_no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_no_terms_of_form_x_pow_2m_infinitely_many_terms_of_form_x_pow_2m_plus_1_l1480_148000


namespace NUMINAMATH_GPT_find_m_n_pairs_l1480_148004

theorem find_m_n_pairs (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (∀ᶠ a in Filter.atTop, (a^m + a - 1) % (a^n + a^2 - 1) = 0) → m = n + 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_n_pairs_l1480_148004


namespace NUMINAMATH_GPT_sequence_divisibility_count_l1480_148054

theorem sequence_divisibility_count :
  ∀ (f : ℕ → ℕ), (∀ n, n ≥ 2 → f n = 10^n - 1) → 
  (∃ count, count = 504 ∧ ∀ i, 2 ≤ i ∧ i ≤ 2023 → (101 ∣ f i ↔ i % 4 = 0)) :=
by { sorry }

end NUMINAMATH_GPT_sequence_divisibility_count_l1480_148054


namespace NUMINAMATH_GPT_inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc_l1480_148058

theorem inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc (a b c : ℝ) :
  5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * a * c + 4 * b * c ∧
  (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * a * c + 4 * b * c → a = 0 ∧ b = 0 ∧ c = 0) := sorry

end NUMINAMATH_GPT_inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc_l1480_148058


namespace NUMINAMATH_GPT_bertha_no_daughters_count_l1480_148083

open Nat

-- Definitions for the conditions
def daughters : ℕ := 8
def total_women : ℕ := 42
def granddaughters : ℕ := total_women - daughters
def daughters_who_have_daughters := granddaughters / 6
def daughters_without_daughters := daughters - daughters_who_have_daughters
def total_without_daughters := granddaughters + daughters_without_daughters

-- The theorem to prove
theorem bertha_no_daughters_count : total_without_daughters = 37 := by
  sorry

end NUMINAMATH_GPT_bertha_no_daughters_count_l1480_148083


namespace NUMINAMATH_GPT_find_a6_l1480_148003

def is_arithmetic_sequence (b : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem find_a6 :
  ∀ (a b : ℕ → ℕ),
    a 1 = 3 →
    b 1 = 2 →
    b 3 = 6 →
    is_arithmetic_sequence b →
    (∀ n, b n = a (n + 1) - a n) →
    a 6 = 33 :=
by
  intros a b h_a1 h_b1 h_b3 h_arith h_diff
  sorry

end NUMINAMATH_GPT_find_a6_l1480_148003


namespace NUMINAMATH_GPT_bus_stop_time_l1480_148063

theorem bus_stop_time (speed_without_stoppages speed_with_stoppages : ℕ) 
(distance : ℕ) (time_without_stoppages time_with_stoppages : ℝ) :
  speed_without_stoppages = 80 ∧ speed_with_stoppages = 40 ∧ distance = 80 ∧
  time_without_stoppages = distance / speed_without_stoppages ∧
  time_with_stoppages = distance / speed_with_stoppages →
  (time_with_stoppages - time_without_stoppages) * 60 = 30 :=
by
  sorry

end NUMINAMATH_GPT_bus_stop_time_l1480_148063


namespace NUMINAMATH_GPT_fraction_is_integer_l1480_148009

theorem fraction_is_integer (b t : ℤ) (hb : b ≠ 1) :
  ∃ (k : ℤ), (t^5 - 5 * b + 4) = k * (b^2 - 2 * b + 1) :=
by 
  sorry

end NUMINAMATH_GPT_fraction_is_integer_l1480_148009


namespace NUMINAMATH_GPT_textbook_weight_difference_l1480_148091

theorem textbook_weight_difference :
  let chem_weight := 7.125
  let geom_weight := 0.625
  chem_weight - geom_weight = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_textbook_weight_difference_l1480_148091


namespace NUMINAMATH_GPT_arithmetic_geometric_sequences_l1480_148084

theorem arithmetic_geometric_sequences :
  ∀ (a₁ a₂ b₁ b₂ b₃ : ℤ), 
  (a₂ = a₁ + (a₁ - (-1))) ∧ 
  (-4 = -1 + 3 * (a₂ - a₁)) ∧ 
  (-4 = -1 * (b₃/b₁)^4) ∧ 
  (b₂ = b₁ * (b₂/b₁)^2) →
  (a₂ - a₁) / b₂ = 1 / 2 := 
by
  intros a₁ a₂ b₁ b₂ b₃ h
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequences_l1480_148084


namespace NUMINAMATH_GPT_time_left_after_council_room_is_zero_l1480_148071

-- Define the conditions
def totalTimeAllowed : ℕ := 30
def travelToSchoolTime : ℕ := 25
def walkToLibraryTime : ℕ := 3
def returnBooksTime : ℕ := 4
def walkToCouncilRoomTime : ℕ := 5
def submitProjectTime : ℕ := 3

-- Calculate time spent up to the student council room
def timeSpentUpToCouncilRoom : ℕ :=
  travelToSchoolTime + walkToLibraryTime + returnBooksTime + walkToCouncilRoomTime + submitProjectTime

-- Question: How much time is left after leaving the student council room to reach the classroom without being late?
theorem time_left_after_council_room_is_zero (totalTimeAllowed travelToSchoolTime walkToLibraryTime returnBooksTime walkToCouncilRoomTime submitProjectTime : ℕ):
  totalTimeAllowed - timeSpentUpToCouncilRoom = 0 := by
  sorry

end NUMINAMATH_GPT_time_left_after_council_room_is_zero_l1480_148071


namespace NUMINAMATH_GPT_parabola_focus_l1480_148093

theorem parabola_focus (a : ℝ) (h : a ≠ 0) (h_directrix : ∀ x y : ℝ, y^2 = a * x → x = -1) : 
    ∃ x y : ℝ, (y = 0 ∧ x = 1 ∧ y^2 = a * x) :=
sorry

end NUMINAMATH_GPT_parabola_focus_l1480_148093


namespace NUMINAMATH_GPT_sum_faces_of_pentahedron_l1480_148008

def pentahedron := {f : ℕ // f = 5}

theorem sum_faces_of_pentahedron (p : pentahedron) : p.val = 5 := 
by
  sorry

end NUMINAMATH_GPT_sum_faces_of_pentahedron_l1480_148008


namespace NUMINAMATH_GPT_parabola_directrix_l1480_148074

theorem parabola_directrix (a : ℝ) : 
  (∃ y, (y ^ 2 = 4 * a * (-2))) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1480_148074


namespace NUMINAMATH_GPT_simplifiedtown_path_difference_l1480_148062

/-- In Simplifiedtown, all streets are 30 feet wide. Each enclosed block forms a square with 
each side measuring 400 feet. Sarah runs exactly next to the block on a path that is 400 feet 
from the block's inner edge while Maude runs on the outer edge of the street opposite to 
Sarah. Prove that Maude runs 120 feet more than Sarah for each lap around the block. -/
theorem simplifiedtown_path_difference :
  let street_width := 30
  let block_side := 400
  let sarah_path := block_side
  let maude_path := block_side + street_width
  let sarah_lap := 4 * sarah_path
  let maude_lap := 4 * maude_path
  maude_lap - sarah_lap = 120 :=
by
  let street_width := 30
  let block_side := 400
  let sarah_path := block_side
  let maude_path := block_side + street_width
  let sarah_lap := 4 * sarah_path
  let maude_lap := 4 * maude_path
  show maude_lap - sarah_lap = 120
  sorry

end NUMINAMATH_GPT_simplifiedtown_path_difference_l1480_148062


namespace NUMINAMATH_GPT_height_difference_petronas_empire_state_l1480_148070

theorem height_difference_petronas_empire_state :
  let esb_height := 443
  let pt_height := 452
  pt_height - esb_height = 9 := by
  sorry

end NUMINAMATH_GPT_height_difference_petronas_empire_state_l1480_148070


namespace NUMINAMATH_GPT_walkway_area_correct_l1480_148049

/-- Define the dimensions of a single flower bed. --/
def flower_bed_length : ℝ := 8
def flower_bed_width : ℝ := 3

/-- Define the number of flower beds in rows and columns. --/
def rows : ℕ := 4
def cols : ℕ := 3

/-- Define the width of the walkways surrounding the flower beds. --/
def walkway_width : ℝ := 2

/-- Calculate the total dimensions of the garden including walkways. --/
def total_garden_width : ℝ := (cols * flower_bed_length) + ((cols + 1) * walkway_width)
def total_garden_height : ℝ := (rows * flower_bed_width) + ((rows + 1) * walkway_width)

/-- Calculate the total area of the garden including walkways. --/
def total_garden_area : ℝ := total_garden_width * total_garden_height

/-- Calculate the total area of the flower beds. --/
def flower_bed_area : ℝ := flower_bed_length * flower_bed_width
def total_flower_beds_area : ℝ := rows * cols * flower_bed_area

/-- Calculate the total area of the walkways. --/
def walkway_area := total_garden_area - total_flower_beds_area

theorem walkway_area_correct : walkway_area = 416 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_walkway_area_correct_l1480_148049


namespace NUMINAMATH_GPT_total_amount_paid_l1480_148059

theorem total_amount_paid (cost_of_manicure : ℝ) (tip_percentage : ℝ) (total : ℝ) 
  (h1 : cost_of_manicure = 30) (h2 : tip_percentage = 0.3) (h3 : total = cost_of_manicure + cost_of_manicure * tip_percentage) : 
  total = 39 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1480_148059


namespace NUMINAMATH_GPT_sqrt_15_estimate_l1480_148076

theorem sqrt_15_estimate : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_15_estimate_l1480_148076


namespace NUMINAMATH_GPT_proof_inequalities_l1480_148029

variable {R : Type} [LinearOrder R] [Ring R]

def odd_function (f : R → R) : Prop :=
∀ x : R, f (-x) = -f x

def decreasing_function (f : R → R) : Prop :=
∀ x y : R, x ≤ y → f y ≤ f x

theorem proof_inequalities (f : R → R) (a b : R) 
  (h_odd : odd_function f)
  (h_decr : decreasing_function f)
  (h : a + b ≤ 0) :
  (f a * f (-a) ≤ 0) ∧ (f a + f b ≥ f (-a) + f (-b)) :=
by
  sorry

end NUMINAMATH_GPT_proof_inequalities_l1480_148029


namespace NUMINAMATH_GPT_numbers_product_l1480_148066

theorem numbers_product (x y : ℝ) (h1 : x + y = 24) (h2 : x - y = 8) : x * y = 128 := by
  sorry

end NUMINAMATH_GPT_numbers_product_l1480_148066


namespace NUMINAMATH_GPT_northbound_vehicle_count_l1480_148019

theorem northbound_vehicle_count :
  ∀ (southbound_speed northbound_speed : ℝ) (vehicles_passed : ℕ) 
  (time_minutes : ℝ) (section_length : ℝ), 
  southbound_speed = 70 → northbound_speed = 50 → vehicles_passed = 30 → time_minutes = 10
  → section_length = 150
  → (vehicles_passed / ((southbound_speed + northbound_speed) * (time_minutes / 60))) * section_length = 270 :=
by sorry

end NUMINAMATH_GPT_northbound_vehicle_count_l1480_148019


namespace NUMINAMATH_GPT_total_students_l1480_148092

theorem total_students (rank_right rank_left : ℕ) (h_right : rank_right = 18) (h_left : rank_left = 12) : rank_right + rank_left - 1 = 29 := 
by
  sorry

end NUMINAMATH_GPT_total_students_l1480_148092


namespace NUMINAMATH_GPT_find_quadruples_l1480_148038

def quadrupleSolution (a b c d : ℝ): Prop :=
  (a * (b + c) = b * (c + d) ∧ b * (c + d) = c * (d + a) ∧ c * (d + a) = d * (a + b))

def isSolution (a b c d : ℝ): Prop :=
  (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 1 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 1 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 1) ∨
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 1 ∧ b = -1 ∧ c = 1 ∧ d = -1) ∨
  (a = 1 ∧ b = -1 + Real.sqrt 2 ∧ c = -1 ∧ d = 1 - Real.sqrt 2) ∨
  (a = 1 ∧ b = -1 - Real.sqrt 2 ∧ c = -1 ∧ d = 1 + Real.sqrt 2)

theorem find_quadruples (a b c d : ℝ) :
  quadrupleSolution a b c d ↔ isSolution a b c d :=
sorry

end NUMINAMATH_GPT_find_quadruples_l1480_148038


namespace NUMINAMATH_GPT_product_of_last_two_digits_of_divisible_by_6_l1480_148094

-- Definitions
def is_divisible_by_6 (n : ℤ) : Prop := n % 6 = 0
def sum_of_last_two_digits (n : ℤ) (a b : ℤ) : Prop := (n % 100) = 10 * a + b

-- Theorem statement
theorem product_of_last_two_digits_of_divisible_by_6 (x a b : ℤ)
  (h1 : is_divisible_by_6 x)
  (h2 : sum_of_last_two_digits x a b)
  (h3 : a + b = 15) :
  (a * b = 54 ∨ a * b = 56) := 
sorry

end NUMINAMATH_GPT_product_of_last_two_digits_of_divisible_by_6_l1480_148094


namespace NUMINAMATH_GPT_cards_from_around_country_l1480_148033

-- Define the total number of cards and the number from home
def total_cards : ℝ := 403.0
def home_cards : ℝ := 287.0

-- Define the expected number of cards from around the country
def expected_country_cards : ℝ := 116.0

-- Theorem statement
theorem cards_from_around_country :
  total_cards - home_cards = expected_country_cards :=
by
  -- Since this only requires the statement, the proof is omitted
  sorry

end NUMINAMATH_GPT_cards_from_around_country_l1480_148033


namespace NUMINAMATH_GPT_abigail_lost_money_l1480_148069

-- Conditions
def initial_money := 11
def money_spent := 2
def money_left := 3

-- Statement of the problem as a Lean theorem
theorem abigail_lost_money : initial_money - money_spent - money_left = 6 := by
  sorry

end NUMINAMATH_GPT_abigail_lost_money_l1480_148069


namespace NUMINAMATH_GPT_ellipse_slope_ratio_l1480_148027

theorem ellipse_slope_ratio (a b : ℝ) (k1 k2 : ℝ) 
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a > 2)
  (h4 : k2 = k1 * (a^2 + 5) / (a^2 - 1)) : 
  1 < (k2 / k1) ∧ (k2 / k1) < 3 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_slope_ratio_l1480_148027


namespace NUMINAMATH_GPT_total_amount_shared_l1480_148017

theorem total_amount_shared (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) 
  (portion_a : ℕ) (portion_b : ℕ) (portion_c : ℕ)
  (h_ratio : ratio_a = 3 ∧ ratio_b = 4 ∧ ratio_c = 9)
  (h_portion_a : portion_a = 30)
  (h_portion_b : portion_b = 2 * portion_a + 10)
  (h_portion_c : portion_c = (ratio_c / ratio_a) * portion_a) :
  portion_a + portion_b + portion_c = 190 :=
by sorry

end NUMINAMATH_GPT_total_amount_shared_l1480_148017


namespace NUMINAMATH_GPT_members_play_both_eq_21_l1480_148026

-- Given definitions
def TotalMembers := 80
def MembersPlayBadminton := 48
def MembersPlayTennis := 46
def MembersPlayNeither := 7

-- Inclusion-Exclusion Principle application to solve the problem
def MembersPlayBoth : ℕ := MembersPlayBadminton + MembersPlayTennis - (TotalMembers - MembersPlayNeither)

-- The theorem we want to prove
theorem members_play_both_eq_21 : MembersPlayBoth = 21 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_members_play_both_eq_21_l1480_148026
