import Mathlib

namespace lightning_distance_l845_84517

/--
Linus observed a flash of lightning and then heard the thunder 15 seconds later.
Given:
- speed of sound: 1088 feet/second
- 1 mile = 5280 feet
Prove that the distance from Linus to the lightning strike is 3.25 miles.
-/
theorem lightning_distance (time_seconds : ℕ) (speed_sound : ℕ) (feet_per_mile : ℕ) (distance_miles : ℚ) :
  time_seconds = 15 →
  speed_sound = 1088 →
  feet_per_mile = 5280 →
  distance_miles = 3.25 :=
by
  sorry

end lightning_distance_l845_84517


namespace find_angle_B_l845_84552

theorem find_angle_B (a b c : ℝ) (h : a^2 + c^2 - b^2 = a * c) : 
  ∃ B : ℝ, 0 < B ∧ B < 180 ∧ B = 60 :=
by 
  sorry

end find_angle_B_l845_84552


namespace min_value_of_sum_l845_84555

theorem min_value_of_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 1 / x + 1 / y + 1 / z = 1) :
  x + 4 * y + 9 * z ≥ 36 ∧ (x + 4 * y + 9 * z = 36 ↔ x = 6 ∧ y = 3 ∧ z = 2) := 
sorry

end min_value_of_sum_l845_84555


namespace scientific_notation_of_274000000_l845_84519

theorem scientific_notation_of_274000000 :
  (274000000 : ℝ) = 2.74 * 10 ^ 8 :=
by
    sorry

end scientific_notation_of_274000000_l845_84519


namespace average_book_width_l845_84559

noncomputable def bookWidths : List ℝ := [5, 0.75, 1.5, 3, 12, 2, 7.5]

theorem average_book_width :
  (bookWidths.sum / bookWidths.length = 4.54) :=
by
  sorry

end average_book_width_l845_84559


namespace perimeter_of_semi_circle_region_l845_84566

theorem perimeter_of_semi_circle_region (side_length : ℝ) (h : side_length = 1/π) : 
  let radius := side_length / 2
  let circumference_of_half_circle := (1 / 2) * π * side_length
  3 * circumference_of_half_circle = 3 / 2
  := by
  sorry

end perimeter_of_semi_circle_region_l845_84566


namespace problem1_equation_of_line_intersection_perpendicular_problem2_equation_of_line_point_equal_intercepts_l845_84553

/-- Lean statement for the math proof problem -/

/- First problem -/
theorem problem1_equation_of_line_intersection_perpendicular :
  ∃ k, 3 * k - 2 * ( - (5 - 3 * k) / 2) - 11 = 0 :=
sorry

/- Second problem -/
theorem problem2_equation_of_line_point_equal_intercepts :
  (∃ a, (1, 2) ∈ {(x, y) | x + y = a}) ∧ a = 3
  ∨ (∃ b, (1, 2) ∈ {(x, y) | y = b * x}) ∧ b = 2 :=
sorry

end problem1_equation_of_line_intersection_perpendicular_problem2_equation_of_line_point_equal_intercepts_l845_84553


namespace identify_irrational_number_l845_84579

theorem identify_irrational_number :
  (∀ a b : ℤ, (-1 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (0 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (1 : ℚ) / (2 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  ¬(∃ a b : ℤ, (Real.sqrt 3) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) :=
sorry

end identify_irrational_number_l845_84579


namespace percentage_of_number_l845_84549

theorem percentage_of_number (X P : ℝ) (h1 : 0.20 * X = 80) (h2 : (P / 100) * X = 160) : P = 40 := by
  sorry

end percentage_of_number_l845_84549


namespace fraction_of_field_planted_l845_84561

theorem fraction_of_field_planted (a b : ℕ) (d : ℝ) :
  a = 5 → b = 12 → d = 3 →
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let side_square := (d * hypotenuse - d^2)/(a + b - 2 * d)
  let area_square := side_square^2
  let area_triangle : ℝ := 1/2 * a * b
  let planted_area := area_triangle - area_square
  let fraction_planted := planted_area / area_triangle
  fraction_planted = 9693/10140 := by
  sorry

end fraction_of_field_planted_l845_84561


namespace largest_int_square_3_digits_base_7_l845_84550

theorem largest_int_square_3_digits_base_7 :
  ∃ (N : ℕ), (7^2 ≤ N^2) ∧ (N^2 < 7^3) ∧ 
  ∃ k : ℕ, N = k ∧ k^2 ≥ 7^2 ∧ k^2 < 7^3 ∧
  N = 45 := sorry

end largest_int_square_3_digits_base_7_l845_84550


namespace pencils_left_l845_84598

-- Define the initial quantities
def MondayPencils := 35
def TuesdayPencils := 42
def WednesdayPencils := 3 * TuesdayPencils
def WednesdayLoss := 20
def ThursdayPencils := WednesdayPencils / 2
def FridayPencils := 2 * MondayPencils
def WeekendLoss := 50

-- Define the total number of pencils Sarah has at the end of each day
def TotalMonday := MondayPencils
def TotalTuesday := TotalMonday + TuesdayPencils
def TotalWednesday := TotalTuesday + WednesdayPencils - WednesdayLoss
def TotalThursday := TotalWednesday + ThursdayPencils
def TotalFriday := TotalThursday + FridayPencils
def TotalWeekend := TotalFriday - WeekendLoss

-- The proof statement
theorem pencils_left : TotalWeekend = 266 :=
by
  sorry

end pencils_left_l845_84598


namespace find_a1_l845_84541

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {a_n} is a geometric sequence with a common ratio q > 0
axiom geom_seq : (∀ n, a (n + 1) = a n * q)

-- Given conditions of the problem
def condition1 : q > 0 := sorry
def condition2 : a 5 * a 7 = 4 * (a 4) ^ 2 := sorry
def condition3 : a 2 = 1 := sorry

-- Prove that a_1 = sqrt 2 / 2
theorem find_a1 : a 1 = (Real.sqrt 2) / 2 := sorry

end find_a1_l845_84541


namespace h_of_j_of_3_l845_84591

def h (x : ℝ) : ℝ := 4 * x + 3
def j (x : ℝ) : ℝ := (x + 2) ^ 2

theorem h_of_j_of_3 : h (j 3) = 103 := by
  sorry

end h_of_j_of_3_l845_84591


namespace percentageReduction_l845_84525

variable (R P : ℝ)

def originalPrice (R : ℝ) (P : ℝ) : Prop :=
  2400 / R - 2400 / P = 8 ∧ R = 120

theorem percentageReduction : 
  originalPrice 120 P → ((P - 120) / P) * 100 = 40 := 
by
  sorry

end percentageReduction_l845_84525


namespace leesburg_population_l845_84574

theorem leesburg_population (salem_population leesburg_population half_salem_population number_moved_out : ℕ)
  (h1 : half_salem_population * 2 = salem_population)
  (h2 : salem_population - number_moved_out = 754100)
  (h3 : salem_population = 15 * leesburg_population)
  (h4 : half_salem_population = 377050)
  (h5 : number_moved_out = 130000) :
  leesburg_population = 58940 :=
by
  sorry

end leesburg_population_l845_84574


namespace Raine_steps_to_school_l845_84536

-- Define Raine's conditions
variable (steps_total : ℕ) (days : ℕ) (round_trip_steps : ℕ)

-- Given conditions
def Raine_conditions := steps_total = 1500 ∧ days = 5 ∧ round_trip_steps = steps_total / days

-- Prove that the steps to school is 150 given Raine's conditions
theorem Raine_steps_to_school (h : Raine_conditions 1500 5 300) : (300 / 2) = 150 :=
by
  sorry

end Raine_steps_to_school_l845_84536


namespace net_effect_on_sale_value_l845_84534

theorem net_effect_on_sale_value (P Q : ℝ) (hP : P > 0) (hQ : Q > 0) :
  let original_sale_value := P * Q
  let new_price := 0.82 * P
  let new_quantity := 1.88 * Q
  let new_sale_value := new_price * new_quantity
  let net_effect := (new_sale_value / original_sale_value - 1) * 100
  net_effect = 54.16 :=
by
  sorry

end net_effect_on_sale_value_l845_84534


namespace add_fractions_l845_84528

theorem add_fractions : (2 / 3 : ℚ) + (7 / 8) = 37 / 24 := 
by sorry

end add_fractions_l845_84528


namespace micrometer_conversion_l845_84514

theorem micrometer_conversion :
  (0.01 * (1 * 10 ^ (-6))) = (1 * 10 ^ (-8)) :=
by 
  -- sorry is used to skip the actual proof but ensure the theorem is recognized
  sorry

end micrometer_conversion_l845_84514


namespace range_a_for_increasing_f_l845_84520

theorem range_a_for_increasing_f :
  (∀ (x : ℝ), 1 ≤ x → (2 * x - 2 * a) ≥ 0) → a ≤ 1 := by
  intro h
  sorry

end range_a_for_increasing_f_l845_84520


namespace intersection_A_B_l845_84533

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a}

theorem intersection_A_B : A ∩ B = {0, 2} :=
by
  sorry

end intersection_A_B_l845_84533


namespace multiplicative_inverse_modulo_l845_84595

noncomputable def A := 123456
noncomputable def B := 153846
noncomputable def N := 500000

theorem multiplicative_inverse_modulo :
  (A * B * N) % 1000000 = 1 % 1000000 :=
by
  sorry

end multiplicative_inverse_modulo_l845_84595


namespace probability_of_multiples_of_4_l845_84539

def number_of_multiples_of_4 (n : ℕ) : ℕ :=
  n / 4

def number_not_multiples_of_4 (n : ℕ) (m : ℕ) : ℕ :=
  n - m

def probability_neither_multiples_of_4 (n : ℕ) (m : ℕ) : ℚ :=
  (m / n : ℚ) * (m / n)

def probability_at_least_one_multiple_of_4 (n : ℕ) (m : ℕ) : ℚ :=
  1 - probability_neither_multiples_of_4 n m

theorem probability_of_multiples_of_4 :
  probability_at_least_one_multiple_of_4 60 45 = 7 / 16 :=
by
  sorry

end probability_of_multiples_of_4_l845_84539


namespace decreasing_intervals_tangent_line_eq_l845_84562

-- Define the function f and its derivative.
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + 1
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Part 1: Prove intervals of monotonic decreasing.
theorem decreasing_intervals :
  (∀ x, f' x < 0 → x < -1 ∨ x > 3) := 
sorry

-- Part 2: Prove the tangent line equation.
theorem tangent_line_eq :
  15 * (-2) + (-13) + 27 = 0 :=
sorry

end decreasing_intervals_tangent_line_eq_l845_84562


namespace geometric_sequence_property_l845_84501

noncomputable def S_n (n : ℕ) (a_n : ℕ → ℕ) : ℕ := 3 * 2^n - 3

noncomputable def a_n (n : ℕ) : ℕ := 3 * 2^(n-1)

noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1)

noncomputable def T_n (n : ℕ) : ℕ := 2^n - 1

theorem geometric_sequence_property (n : ℕ) (hn : n ≥ 0) :
  T_n n < b_n (n+1) :=
by
  sorry

end geometric_sequence_property_l845_84501


namespace find_x_if_delta_phi_eq_3_l845_84503

variable (x : ℚ)

def delta (x : ℚ) := 4 * x + 9
def phi (x : ℚ) := 9 * x + 6

theorem find_x_if_delta_phi_eq_3 : 
  delta (phi x) = 3 → x = -5 / 6 := by 
  sorry

end find_x_if_delta_phi_eq_3_l845_84503


namespace beka_flew_more_l845_84560

def bekaMiles := 873
def jacksonMiles := 563

theorem beka_flew_more : bekaMiles - jacksonMiles = 310 := by
  -- proof here
  sorry

end beka_flew_more_l845_84560


namespace triangle_inequality_l845_84545

variable {α β γ a b c : ℝ}

theorem triangle_inequality (h1: α ≥ β) (h2: β ≥ γ) (h3: a ≥ b) (h4: b ≥ c) (h5: α ≥ γ) (h6: a ≥ c) :
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α :=
by
  sorry

end triangle_inequality_l845_84545


namespace average_income_A_B_l845_84518

theorem average_income_A_B (A B C : ℝ)
  (h1 : (B + C) / 2 = 5250)
  (h2 : (A + C) / 2 = 4200)
  (h3 : A = 3000) : (A + B) / 2 = 4050 :=
by
  sorry

end average_income_A_B_l845_84518


namespace mn_minus_n_values_l845_84548

theorem mn_minus_n_values (m n : ℝ) (h1 : |m| = 4) (h2 : |n| = 2.5) (h3 : m * n < 0) :
  m * n - n = -7.5 ∨ m * n - n = -12.5 :=
sorry

end mn_minus_n_values_l845_84548


namespace descending_order_of_numbers_l845_84529

theorem descending_order_of_numbers :
  let a := 62
  let b := 78
  let c := 64
  let d := 59
  b > c ∧ c > a ∧ a > d :=
by
  let a := 62
  let b := 78
  let c := 64
  let d := 59
  sorry

end descending_order_of_numbers_l845_84529


namespace total_number_of_coins_l845_84526

theorem total_number_of_coins (x : ℕ) :
  5 * x + 10 * x + 25 * x = 120 → 3 * x = 9 :=
by
  intro h
  sorry

end total_number_of_coins_l845_84526


namespace distance_to_conference_l845_84568

theorem distance_to_conference (t d : ℝ) 
  (h1 : d = 40 * (t + 0.75))
  (h2 : d - 40 = 60 * (t - 1.25)) :
  d = 160 :=
by
  sorry

end distance_to_conference_l845_84568


namespace river_bank_bottom_width_l845_84507

/-- 
The cross-section of a river bank is a trapezium with a 12 m wide top and 
a certain width at the bottom. The area of the cross-section is 500 sq m 
and the depth is 50 m. Prove that the width at the bottom is 8 m.
-/
theorem river_bank_bottom_width (area height top_width : ℝ) (h_area: area = 500) 
(h_height : height = 50) (h_top_width : top_width = 12) : ∃ b : ℝ, (1 / 2) * (top_width + b) * height = area ∧ b = 8 :=
by
  use 8
  sorry

end river_bank_bottom_width_l845_84507


namespace janet_percentage_of_snowballs_l845_84593

-- Define the number of snowballs made by Janet
def janet_snowballs : ℕ := 50

-- Define the number of snowballs made by Janet's brother
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage calculation function
def calculate_percentage (part whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- Proof statement
theorem janet_percentage_of_snowballs : calculate_percentage janet_snowballs total_snowballs = 25 := 
by
  sorry

end janet_percentage_of_snowballs_l845_84593


namespace vector_dot_product_value_l845_84535

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_dot_product_value : dot_product (add (scalar_mul 2 a) b) c = -3 := by
  sorry

end vector_dot_product_value_l845_84535


namespace asia_discount_problem_l845_84597

theorem asia_discount_problem
  (originalPrice : ℝ)
  (storeDiscount : ℝ)
  (memberDiscount : ℝ)
  (finalPriceUSD : ℝ)
  (exchangeRate : ℝ)
  (finalDiscountPercentage : ℝ) :
  originalPrice = 300 →
  storeDiscount = 0.20 →
  memberDiscount = 0.10 →
  finalPriceUSD = 224 →
  exchangeRate = 1.10 →
  finalDiscountPercentage = 28 :=
by
  sorry

end asia_discount_problem_l845_84597


namespace average_visitors_remaining_days_l845_84531

-- Definitions
def visitors_monday := 50
def visitors_tuesday := 2 * visitors_monday
def total_week_visitors := 250
def days_remaining := 5
def remaining_visitors := total_week_visitors - (visitors_monday + visitors_tuesday)
def average_remaining_visitors_per_day := remaining_visitors / days_remaining

-- Theorem statement
theorem average_visitors_remaining_days : average_remaining_visitors_per_day = 20 :=
by
  -- Proof is skipped
  sorry

end average_visitors_remaining_days_l845_84531


namespace always_real_roots_range_of_b_analytical_expression_parabola_l845_84589

-- Define the quadratic equation with parameter m
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (5 * m - 1) * x + 4 * m - 4

-- Part 1: Prove the equation always has real roots
theorem always_real_roots (m : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 := 
sorry

-- Part 2: Find the range of b such that the line intersects the parabola at two distinct points
theorem range_of_b (b : ℝ) : 
  (∀ m : ℝ, m = 1 → (b > -25/4 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq m x1 = (x1 + b) ∧ quadratic_eq m x2 = (x2 + b)))) :=
sorry

-- Part 3: Find the analytical expressions of the parabolas given the distance condition
theorem analytical_expression_parabola (m : ℝ) : 
  (∀ x1 x2 : ℝ, (|x1 - x2| = 2 → quadratic_eq m x1 = 0 → quadratic_eq m x2 = 0) → 
  (m = -1 ∨ m = -1/5) → 
  ((quadratic_eq (-1) x = -x^2 + 6*x - 8) ∨ (quadratic_eq (-1/5) x = -1/5*x^2 + 2*x - 24/5))) :=
sorry

end always_real_roots_range_of_b_analytical_expression_parabola_l845_84589


namespace trains_crossing_time_l845_84516

theorem trains_crossing_time
  (L1 : ℕ) (L2 : ℕ) (T1 : ℕ) (T2 : ℕ)
  (H1 : L1 = 150) (H2 : L2 = 180)
  (H3 : T1 = 10) (H4 : T2 = 15) :
  (L1 + L2) / ((L1 / T1) + (L2 / T2)) = 330 / 27 := sorry

end trains_crossing_time_l845_84516


namespace order_large_pizzas_sufficient_l845_84576

def pizza_satisfaction (gluten_free_slices_per_large : ℕ) (medium_slices : ℕ) (small_slices : ℕ) 
                       (gluten_free_needed : ℕ) (dairy_free_needed : ℕ) :=
  let slices_gluten_free := small_slices
  let slices_dairy_free := 2 * medium_slices
  (slices_gluten_free < gluten_free_needed) → 
  let additional_slices_gluten_free := gluten_free_needed - slices_gluten_free
  let large_pizzas_gluten_free := (additional_slices_gluten_free + gluten_free_slices_per_large - 1) / gluten_free_slices_per_large
  large_pizzas_gluten_free = 1

theorem order_large_pizzas_sufficient :
  pizza_satisfaction 14 10 8 15 15 :=
by
  unfold pizza_satisfaction
  sorry

end order_large_pizzas_sufficient_l845_84576


namespace calculate_neg_pow_mul_l845_84584

theorem calculate_neg_pow_mul (a : ℝ) : -a^4 * a^3 = -a^7 := by
  sorry

end calculate_neg_pow_mul_l845_84584


namespace polynomial_expansion_l845_84509

theorem polynomial_expansion (x : ℝ) :
  (3 * x^3 + 4 * x - 7) * (2 * x^4 - 3 * x^2 + 5) =
  6 * x^7 + 12 * x^5 - 9 * x^4 - 21 * x^3 - 11 * x + 35 :=
by
  sorry

end polynomial_expansion_l845_84509


namespace consecutive_integers_avg_l845_84575

theorem consecutive_integers_avg (n x : ℤ) (h_avg : (2*x + n - 1 : ℝ)/2 = 20.5) (h_10th : x + 9 = 25) :
  n = 10 :=
by
  sorry

end consecutive_integers_avg_l845_84575


namespace alcohol_water_ratio_l845_84542

theorem alcohol_water_ratio (alcohol water : ℝ) (h_alcohol : alcohol = 3 / 5) (h_water : water = 2 / 5) :
  alcohol / water = 3 / 2 :=
by 
  sorry

end alcohol_water_ratio_l845_84542


namespace find_g_two_l845_84546

variable (g : ℝ → ℝ)

-- Condition 1: Functional equation
axiom g_eq : ∀ x y : ℝ, g (x - y) = g x * g y

-- Condition 2: Non-zero property
axiom g_ne_zero : ∀ x : ℝ, g x ≠ 0

-- Proof statement
theorem find_g_two : g 2 = 1 := 
by sorry

end find_g_two_l845_84546


namespace garden_area_increase_l845_84500

-- Definitions derived directly from the conditions
def length := 50
def width := 10
def perimeter := 2 * (length + width)
def side_length_square := perimeter / 4
def area_rectangle := length * width
def area_square := side_length_square * side_length_square

-- The proof statement
theorem garden_area_increase :
  area_square - area_rectangle = 400 := 
by
  sorry

end garden_area_increase_l845_84500


namespace inequality_holds_l845_84587

theorem inequality_holds (a : ℝ) : 
  (∀ x : ℝ, a*x^2 + 2*a*x - 2 < 0) ↔ a ∈ Set.Icc (-2 : ℝ) (0 : ℝ) :=
sorry

end inequality_holds_l845_84587


namespace math_problem_l845_84515

/-
Two mathematicians take a morning coffee break each day.
They arrive at the cafeteria independently, at random times between 9 a.m. and 10:30 a.m.,
and stay for exactly m minutes.
Given the probability that either one arrives while the other is in the cafeteria is 30%,
and m = a - b√c, where a, b, and c are positive integers, and c is not divisible by the square of any prime,
prove that a + b + c = 127.

-/

noncomputable def is_square_free (c : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p * p ∣ c → False

theorem math_problem
  (m a b c : ℕ)
  (h1 : 0 < m)
  (h2 : m = a - b * Real.sqrt c)
  (h3 : is_square_free c)
  (h4 : 30 * (90 * 90) / 100 = (90 - m) * (90 - m)) :
  a + b + c = 127 :=
sorry

end math_problem_l845_84515


namespace complement_union_l845_84547

open Set

def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 2}
def N : Set ℕ := {0, 2, 3}

theorem complement_union :
  compl (M ∪ N) = {1} :=
by
  sorry

end complement_union_l845_84547


namespace average_speed_trip_l845_84537

-- Conditions: Definitions
def distance_north_feet : ℝ := 5280
def speed_north_mpm : ℝ := 2
def speed_south_mpm : ℝ := 1

-- Question and Equivalent Proof Problem
theorem average_speed_trip :
  let distance_north_miles := distance_north_feet / 5280
  let distance_south_miles := 2 * distance_north_miles
  let total_distance_miles := distance_north_miles + distance_south_miles + distance_south_miles
  let time_north_hours := distance_north_miles / speed_north_mpm / 60
  let time_south_hours := distance_south_miles / speed_south_mpm / 60
  let time_return_hours := distance_south_miles / speed_south_mpm / 60
  let total_time_hours := time_north_hours + time_south_hours + time_return_hours
  let average_speed_mph := total_distance_miles / total_time_hours
  average_speed_mph = 76.4 := by
    sorry

end average_speed_trip_l845_84537


namespace floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2_l845_84510

theorem floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2 (n : ℕ) (hn : n > 0) :
  Int.floor (Real.sqrt n + Real.sqrt (n + 1)) = Int.floor (Real.sqrt (4 * n + 2)) := 
  sorry

end floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2_l845_84510


namespace initial_acidic_liquid_quantity_l845_84573

theorem initial_acidic_liquid_quantity
  (A : ℝ) -- initial quantity of the acidic liquid in liters
  (W : ℝ) -- quantity of water to be removed in liters
  (h1 : W = 6)
  (h2 : (0.40 * A) = 0.60 * (A - W)) : 
  A = 18 :=
by sorry

end initial_acidic_liquid_quantity_l845_84573


namespace problem_solution_l845_84523

theorem problem_solution (x y : ℝ) (h₁ : (4 * y^2 + 1) * (x^4 + 2 * x^2 + 2) = 8 * |y| * (x^2 + 1))
  (h₂ : y ≠ 0) :
  (x = 0 ∧ (y = 1/2 ∨ y = -1/2)) :=
by {
  sorry -- Proof required
}

end problem_solution_l845_84523


namespace m_not_equal_n_possible_l845_84512

-- Define the touching relation on an infinite chessboard
structure Chessboard :=
(colored_square : ℤ × ℤ → Prop)
(touches : ℤ × ℤ → ℤ × ℤ → Prop)

-- Define the properties
def colors_square (board : Chessboard) : Prop :=
∃ i j : ℤ, board.colored_square (i, j) ∧ board.colored_square (i + 1, j + 1)

def black_square_touches_m_black_squares (board : Chessboard) (m : ℕ) : Prop :=
∀ i j : ℤ, board.colored_square (i, j) →
    (board.touches (i, j) (i + 1, j) → board.colored_square (i + 1, j)) ∧ 
    (board.touches (i, j) (i - 1, j) → board.colored_square (i - 1, j)) ∧
    (board.touches (i, j) (i, j + 1) → board.colored_square (i, j + 1)) ∧
    (board.touches (i, j) (i, j - 1) → board.colored_square (i, j - 1))
    -- Add additional conditions to ensure exactly m black squares are touched

def white_square_touches_n_white_squares (board : Chessboard) (n : ℕ) : Prop :=
∀ i j : ℤ, ¬board.colored_square (i, j) →
    (board.touches (i, j) (i + 1, j) → ¬board.colored_square (i + 1, j)) ∧ 
    (board.touches (i, j) (i - 1, j) → ¬board.colored_square (i - 1, j)) ∧
    (board.touches (i, j) (i, j + 1) → ¬board.colored_square (i, j + 1)) ∧
    (board.touches (i, j) (i, j - 1) → ¬board.colored_square (i, j - 1))
    -- Add additional conditions to ensure exactly n white squares are touched

theorem m_not_equal_n_possible (board : Chessboard) (m n : ℕ) :
colors_square board →
black_square_touches_m_black_squares board m →
white_square_touches_n_white_squares board n →
m ≠ n :=
by {
    sorry
}

end m_not_equal_n_possible_l845_84512


namespace at_least_two_even_l845_84554

theorem at_least_two_even (x y z : ℤ) (u : ℤ)
  (h : x^2 + y^2 + z^2 = u^2) : (↑x % 2 = 0) ∨ (↑y % 2 = 0) → (↑x % 2 = 0) ∨ (↑z % 2 = 0) ∨ (↑y % 2 = 0) := 
by
  sorry

end at_least_two_even_l845_84554


namespace beautifulEquations_1_find_n_l845_84585

def isBeautifulEquations (eq1 eq2 : ℝ → Prop) : Prop :=
  ∃ x y : ℝ, eq1 x ∧ eq2 y ∧ x + y = 1

def eq1a (x : ℝ) : Prop := 4 * x - (x + 5) = 1
def eq2a (y : ℝ) : Prop := -2 * y - y = 3

theorem beautifulEquations_1 : isBeautifulEquations eq1a eq2a :=
sorry

def eq1b (x : ℝ) (n : ℝ) : Prop := 2 * x - n + 3 = 0
def eq2b (x : ℝ) (n : ℝ) : Prop := x + 5 * n - 1 = 0

theorem find_n (n : ℝ) : (∀ x1 x2 : ℝ, eq1b x1 n ∧ eq2b x2 n ∧ x1 + x2 = 1) → n = -1 / 3 :=
sorry

end beautifulEquations_1_find_n_l845_84585


namespace prob_top_three_cards_all_hearts_l845_84592

-- Define the total numbers of cards and suits
def total_cards := 52
def hearts_count := 13

-- Define the probability calculation as per the problem statement
def prob_top_three_hearts : ℚ :=
  (13 * 12 * 11 : ℚ) / (52 * 51 * 50 : ℚ)

-- The theorem states that the probability of the top three cards being all hearts is 11/850
theorem prob_top_three_cards_all_hearts : prob_top_three_hearts = 11 / 850 := by
  -- The proof details are not required, just stating the structure
  sorry

end prob_top_three_cards_all_hearts_l845_84592


namespace problem_l845_84543

theorem problem
  (x y : ℝ)
  (h1 : x + 3 * y = 9)
  (h2 : x * y = -27) :
  x^2 + 9 * y^2 = 243 :=
sorry

end problem_l845_84543


namespace banana_cantaloupe_cost_l845_84564

theorem banana_cantaloupe_cost {a b c d : ℕ} 
  (h1 : a + b + c + d = 20) 
  (h2 : d = 2 * a)
  (h3 : c = a - b) : b + c = 5 :=
sorry

end banana_cantaloupe_cost_l845_84564


namespace tiles_needed_l845_84508

/-- 
Given:
- The cafeteria is tiled with the same floor tiles.
- It takes 630 tiles to cover an area of 18 square decimeters of tiles.
- We switch to square tiles with a side length of 6 decimeters.

Prove:
- The number of new tiles needed to cover the same area is 315.
--/
theorem tiles_needed (n_tiles : ℕ) (area_per_tile : ℕ) (new_tile_side_length : ℕ) 
  (h1 : n_tiles = 630) (h2 : area_per_tile = 18) (h3 : new_tile_side_length = 6) :
  (630 * 18) / (6 * 6) = 315 :=
by
  sorry

end tiles_needed_l845_84508


namespace find_d_l845_84569

theorem find_d (c : ℝ) (d : ℝ) (h1 : c = 7)
  (h2 : (2, 6) ∈ { p : ℝ × ℝ | ∃ d, (p = (2, 6) ∨ p = (5, c) ∨ p = (d, 0)) ∧
           ∃ m, m = (0 - 6) / (d - 2) ∧ m = (c - 6) / (5 - 2) }) : 
  d = -16 :=
by
  sorry

end find_d_l845_84569


namespace find_x_l845_84511

def custom_op (a b : ℝ) : ℝ :=
  a^2 - 3 * b

theorem find_x (x : ℝ) : 
  (custom_op (custom_op 7 x) 3 = 18) ↔ (x = 17.71 ∨ x = 14.96) := 
by
  sorry

end find_x_l845_84511


namespace n_squared_plus_3n_is_perfect_square_iff_l845_84524

theorem n_squared_plus_3n_is_perfect_square_iff (n : ℕ) : 
  ∃ k : ℕ, n^2 + 3 * n = k^2 ↔ n = 1 :=
by 
  sorry

end n_squared_plus_3n_is_perfect_square_iff_l845_84524


namespace chess_tournament_no_804_games_l845_84577

/-- Statement of the problem: 
    Under the given conditions, prove that it is impossible for exactly 804 games to have been played in the chess tournament.
--/
theorem chess_tournament_no_804_games :
  ¬ ∃ n : ℕ, n * (n - 4) = 1608 :=
by
  sorry

end chess_tournament_no_804_games_l845_84577


namespace integral_sqrt_1_minus_x_sq_plus_2x_l845_84556

theorem integral_sqrt_1_minus_x_sq_plus_2x :
  ∫ x in (0 : Real)..1, (Real.sqrt (1 - x^2) + 2 * x) = (Real.pi + 4) / 4 := by
  sorry

end integral_sqrt_1_minus_x_sq_plus_2x_l845_84556


namespace triangle_angle_contradiction_l845_84502

theorem triangle_angle_contradiction (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (h_sum : A + B + C = 180) :
  false :=
by
  -- Here "A > 60, B > 60, C > 60 and A + B + C = 180" leads to a contradiction
  sorry

end triangle_angle_contradiction_l845_84502


namespace find_h_l845_84581

theorem find_h (h : ℝ) (r s : ℝ) (h_eq : ∀ x : ℝ, x^2 - 4 * h * x - 8 = 0)
  (sum_of_squares : r^2 + s^2 = 20) (roots_eq : x^2 - 4 * h * x - 8 = (x - r) * (x - s)) :
  h = 1 / 2 ∨ h = -1 / 2 := 
sorry

end find_h_l845_84581


namespace noah_small_paintings_sold_last_month_l845_84521

theorem noah_small_paintings_sold_last_month
  (large_painting_price small_painting_price : ℕ)
  (large_paintings_sold_last_month : ℕ)
  (total_sales_this_month : ℕ)
  (sale_multiplier : ℕ)
  (x : ℕ)
  (h1 : large_painting_price = 60)
  (h2 : small_painting_price = 30)
  (h3 : large_paintings_sold_last_month = 8)
  (h4 : total_sales_this_month = 1200)
  (h5 : sale_multiplier = 2) :
  (2 * ((large_paintings_sold_last_month * large_painting_price) + (x * small_painting_price)) = total_sales_this_month) → x = 4 :=
by
  sorry

end noah_small_paintings_sold_last_month_l845_84521


namespace sum_of_reversed_base_digits_eq_zero_l845_84563

theorem sum_of_reversed_base_digits_eq_zero : ∃ n : ℕ, 
  (∀ a₁ a₀ : ℕ, n = 5 * a₁ + a₀ ∧ n = 12 * a₀ + a₁ ∧ 0 ≤ a₁ ∧ a₁ < 5 ∧ 0 ≤ a₀ ∧ a₀ < 12 
  ∧ n > 0 → n = 0)
:= sorry

end sum_of_reversed_base_digits_eq_zero_l845_84563


namespace robert_monthly_expenses_l845_84582

def robert_basic_salary : ℝ := 1250
def robert_sales : ℝ := 23600
def first_tier_limit : ℝ := 10000
def second_tier_limit : ℝ := 20000
def first_tier_rate : ℝ := 0.10
def second_tier_rate : ℝ := 0.12
def third_tier_rate : ℝ := 0.15
def savings_rate : ℝ := 0.20

def first_tier_commission : ℝ :=
  first_tier_limit * first_tier_rate

def second_tier_commission : ℝ :=
  (second_tier_limit - first_tier_limit) * second_tier_rate

def third_tier_commission : ℝ :=
  (robert_sales - second_tier_limit) * third_tier_rate

def total_commission : ℝ :=
  first_tier_commission + second_tier_commission + third_tier_commission

def total_earnings : ℝ :=
  robert_basic_salary + total_commission

def savings : ℝ :=
  total_earnings * savings_rate

def monthly_expenses : ℝ :=
  total_earnings - savings

theorem robert_monthly_expenses :
  monthly_expenses = 3192 := by
  sorry

end robert_monthly_expenses_l845_84582


namespace propositions_A_and_D_true_l845_84567

theorem propositions_A_and_D_true :
  (∀ x : ℝ, x^2 - 4*x + 5 > 0) ∧ (∃ x : ℤ, 3*x^2 - 2*x - 1 = 0) :=
by
  sorry

end propositions_A_and_D_true_l845_84567


namespace goldie_earnings_l845_84599

theorem goldie_earnings
  (hourly_wage : ℕ := 5)
  (hours_last_week : ℕ := 20)
  (hours_this_week : ℕ := 30) :
  hourly_wage * hours_last_week + hourly_wage * hours_this_week = 250 :=
by
  sorry

end goldie_earnings_l845_84599


namespace problem_l845_84590

noncomputable def f : ℝ → ℝ := sorry

theorem problem (f_decreasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x)
                (h : ∀ x : ℝ, 0 < x → f (f x - 1 / Real.exp x) = 1 / Real.exp 1 + 1) :
  f (Real.log 2) = 3 / 2 :=
sorry

end problem_l845_84590


namespace pipe_fill_time_with_leak_l845_84527

theorem pipe_fill_time_with_leak (A L : ℝ) (hA : A = 1 / 2) (hL : L = 1 / 6) :
  (1 / (A - L)) = 3 :=
by
  sorry

end pipe_fill_time_with_leak_l845_84527


namespace wednesday_tips_value_l845_84557

-- Definitions for the conditions
def hourly_wage : ℕ := 10
def monday_hours : ℕ := 7
def tuesday_hours : ℕ := 5
def wednesday_hours : ℕ := 7
def monday_tips : ℕ := 18
def tuesday_tips : ℕ := 12
def total_earnings : ℕ := 240

-- Hourly earnings
def monday_earnings := monday_hours * hourly_wage
def tuesday_earnings := tuesday_hours * hourly_wage
def wednesday_earnings := wednesday_hours * hourly_wage

-- Total wage earnings
def total_wage_earnings := monday_earnings + tuesday_earnings + wednesday_earnings

-- Total earnings with known tips
def known_earnings := total_wage_earnings + monday_tips + tuesday_tips

-- Prove that Wednesday tips is $20
theorem wednesday_tips_value : (total_earnings - known_earnings) = 20 := by
  sorry

end wednesday_tips_value_l845_84557


namespace jasper_drinks_more_than_hot_dogs_l845_84505

-- Definition of conditions based on the problem
def bags_of_chips := 27
def fewer_hot_dogs_than_chips := 8
def drinks_sold := 31

-- Definition to compute the number of hot dogs
def hot_dogs_sold := bags_of_chips - fewer_hot_dogs_than_chips

-- Lean 4 statement to prove the final result
theorem jasper_drinks_more_than_hot_dogs : drinks_sold - hot_dogs_sold = 12 :=
by
  -- skipping the proof
  sorry

end jasper_drinks_more_than_hot_dogs_l845_84505


namespace investment_of_c_l845_84513

theorem investment_of_c (P_b P_a P_c C_a C_b C_c : ℝ)
  (h1 : P_b = 2000) 
  (h2 : P_a - P_c = 799.9999999999998)
  (h3 : C_a = 8000)
  (h4 : C_b = 10000)
  (h5 : P_b / C_b = P_a / C_a)
  (h6 : P_c / C_c = P_a / C_a)
  : C_c = 4000 :=
by 
  sorry

end investment_of_c_l845_84513


namespace lunch_break_duration_l845_84594

theorem lunch_break_duration :
  ∃ L : ℝ, 
    ∀ (p h : ℝ),
      (9 - L) * (p + h) = 0.4 ∧
      (7 - L) * h = 0.3 ∧
      (12 - L) * p = 0.3 →
      L = 0.5 := by
  sorry

end lunch_break_duration_l845_84594


namespace divisible_by_120_l845_84506

theorem divisible_by_120 (n : ℕ) (hn_pos : n > 0) : 120 ∣ n * (n^2 - 1) * (n^2 - 5 * n + 26) := 
by
  sorry

end divisible_by_120_l845_84506


namespace triangle_area_rational_l845_84578

-- Define the conditions
def satisfies_eq (x y : ℤ) : Prop := x - y = 1

-- Define the points
variables (x1 y1 x2 y2 x3 y3 : ℤ)

-- Assume each point satisfies the equation
axiom point1 : satisfies_eq x1 y1
axiom point2 : satisfies_eq x2 y2
axiom point3 : satisfies_eq x3 y3

-- Statement that we need to prove
theorem triangle_area_rational :
  ∃ (area : ℚ), 
    ∃ (triangle_points : ∃ (x1 y1 x2 y2 x3 y3 : ℤ), satisfies_eq x1 y1 ∧ satisfies_eq x2 y2 ∧ satisfies_eq x3 y3), 
      true :=
sorry

end triangle_area_rational_l845_84578


namespace smallest_m_l845_84558

theorem smallest_m (m : ℤ) :
  (∀ x : ℝ, (3 * x * (m * x - 5) - x^2 + 8) = 0) → (257 - 96 * m < 0) → (m = 3) :=
sorry

end smallest_m_l845_84558


namespace no_solution_equation_l845_84572

theorem no_solution_equation (x : ℝ) : (x + 1) / (x - 1) + 4 / (1 - x^2) ≠ 1 :=
  sorry

end no_solution_equation_l845_84572


namespace shifted_parabola_eq_l845_84580

def initial_parabola (x : ℝ) : ℝ := 5 * x^2

def shifted_parabola (x : ℝ) : ℝ := 5 * (x + 2)^2 + 3

theorem shifted_parabola_eq :
  ∀ x : ℝ, shifted_parabola x = 5 * (x + 2)^2 + 3 :=
by
  intro x
  sorry

end shifted_parabola_eq_l845_84580


namespace car_mileage_city_l845_84544

theorem car_mileage_city (h c t : ℕ) 
  (h_eq_tank_mileage : 462 = h * t) 
  (c_eq_tank_mileage : 336 = c * t) 
  (mileage_diff : c = h - 3) : 
  c = 8 := 
by
  sorry

end car_mileage_city_l845_84544


namespace jill_draws_spade_probability_l845_84530

noncomputable def probability_jill_draws_spade : ℚ :=
  ∑' (k : ℕ), ((3 / 4) * (3 / 4))^k * ((3 / 4) * (1 / 4))

theorem jill_draws_spade_probability : probability_jill_draws_spade = 3 / 7 :=
sorry

end jill_draws_spade_probability_l845_84530


namespace solve_equation_1_solve_equation_2_solve_equation_3_l845_84565

theorem solve_equation_1 (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 := 
sorry

theorem solve_equation_2 (x : ℝ) : (2 * x - 1)^2 = (3 - x)^2 ↔ x = -2 ∨ x = 4 / 3 := 
sorry

theorem solve_equation_3 (x : ℝ) : 3 * x * (x - 2) = x - 2 ↔ x = 2 ∨ x = 1 / 3 :=
sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l845_84565


namespace triangle_shape_l845_84522

theorem triangle_shape (a b : ℝ) (A B : ℝ) (hA : 0 < A) (hB : A < π) (h : a * Real.cos A = b * Real.cos B) :
  (A = B ∨ A + B = π / 2 ∨ a = b) :=
by
  sorry

end triangle_shape_l845_84522


namespace triangle_angle_split_l845_84504

-- Conditions
variables (A B C C1 C2 : ℝ)
-- Axioms/Assumptions
axiom angle_order : A < B
axiom angle_partition : A + C1 = 90 ∧ B + C2 = 90

-- The theorem to prove
theorem triangle_angle_split : C1 - C2 = B - A :=
by {
  sorry
}

end triangle_angle_split_l845_84504


namespace rectangular_eq_of_C_slope_of_l_l845_84583

noncomputable section

/-- Parametric equations for curve C -/
def parametric_eq (θ : ℝ) : ℝ × ℝ :=
⟨4 * Real.cos θ, 3 * Real.sin θ⟩

/-- Question 1: Prove that the rectangular coordinate equation of curve C is (x^2)/16 + (y^2)/9 = 1. -/
theorem rectangular_eq_of_C (x y θ : ℝ) (h₁ : x = 4 * Real.cos θ) (h₂ : y = 3 * Real.sin θ) : 
  x^2 / 16 + y^2 / 9 = 1 := 
sorry

/-- Line passing through point M(2, 2) with parametric equations -/
def line_through_M (t α : ℝ) : ℝ × ℝ :=
⟨2 + t * Real.cos α, 2 + t * Real.sin α⟩ 

/-- Question 2: Prove that the slope of line l passing M(2, 2) which intersects curve C at points A and B is -9/16 -/
theorem slope_of_l (t₁ t₂ α : ℝ) (t₁_t₂_sum_zero : (9 * Real.sin α + 36 * Real.cos α) = 0) :
  Real.tan α = -9 / 16 :=
sorry

end rectangular_eq_of_C_slope_of_l_l845_84583


namespace multiplication_decomposition_l845_84586

theorem multiplication_decomposition :
  100 * 3 = 100 + 100 + 100 :=
sorry

end multiplication_decomposition_l845_84586


namespace joining_fee_per_person_l845_84538

variables (F : ℝ)
variables (family_members : ℕ) (monthly_cost_per_person : ℝ) (john_yearly_payment : ℝ)

def total_cost (F : ℝ) (family_members : ℕ) (monthly_cost_per_person : ℝ) : ℝ :=
  family_members * (F + 12 * monthly_cost_per_person)

theorem joining_fee_per_person :
  (family_members = 4) →
  (monthly_cost_per_person = 1000) →
  (john_yearly_payment = 32000) →
  john_yearly_payment = 0.5 * total_cost F family_members monthly_cost_per_person →
  F = 4000 :=
by
  intros h_family h_monthly_cost h_yearly_payment h_eq
  sorry

end joining_fee_per_person_l845_84538


namespace fraction_simplification_l845_84540

theorem fraction_simplification :
  (3100 - 3037)^2 / 81 = 49 := by
  sorry

end fraction_simplification_l845_84540


namespace lily_pads_cover_half_l845_84551

theorem lily_pads_cover_half (P D : ℕ) (cover_entire : P * (2 ^ 25) = D) : P * (2 ^ 24) = D / 2 :=
by sorry

end lily_pads_cover_half_l845_84551


namespace absolute_value_inequality_solution_l845_84532

theorem absolute_value_inequality_solution (x : ℝ) : abs (x - 3) < 2 ↔ 1 < x ∧ x < 5 :=
by
  sorry

end absolute_value_inequality_solution_l845_84532


namespace james_music_BPM_l845_84588

theorem james_music_BPM 
  (hours_per_day : ℕ)
  (beats_per_week : ℕ)
  (days_per_week : ℕ)
  (minutes_per_hour : ℕ)
  (minutes_per_day : ℕ)
  (total_minutes_per_week : ℕ)
  (BPM : ℕ)
  (h1 : hours_per_day = 2)
  (h2 : beats_per_week = 168000)
  (h3 : days_per_week = 7)
  (h4 : minutes_per_hour = 60)
  (h5 : minutes_per_day = hours_per_day * minutes_per_hour)
  (h6 : total_minutes_per_week = minutes_per_day * days_per_week)
  (h7 : BPM = beats_per_week / total_minutes_per_week)
  : BPM = 200 :=
sorry

end james_music_BPM_l845_84588


namespace parallel_vectors_l845_84570

theorem parallel_vectors (m : ℝ) :
  let a : (ℝ × ℝ × ℝ) := (2, -1, 2)
  let b : (ℝ × ℝ × ℝ) := (-4, 2, m)
  (∀ k : ℝ, a = (k * -4, k * 2, k * m)) →
  m = -4 :=
by
  sorry

end parallel_vectors_l845_84570


namespace find_legs_of_triangle_l845_84571

-- Definition of the problem conditions
def right_triangle (x y : ℝ) := x * y = 200 ∧ 4 * (y - 4) = 8 * (x - 8)

-- Theorem we want to prove
theorem find_legs_of_triangle : 
  ∃ (x y : ℝ), right_triangle x y ∧ ((x = 40 ∧ y = 5) ∨ (x = 10 ∧ y = 20)) :=
by
  sorry

end find_legs_of_triangle_l845_84571


namespace average_age_decrease_l845_84596

theorem average_age_decrease (N T : ℕ) (h₁ : (T : ℝ) / N - 3 = (T - 30 : ℝ) / N) : N = 10 :=
sorry

end average_age_decrease_l845_84596
