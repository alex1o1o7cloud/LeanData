import Mathlib

namespace NUMINAMATH_GPT_students_on_playground_l974_97403

theorem students_on_playground (rows_left : ℕ) (rows_right : ℕ) (rows_front : ℕ) (rows_back : ℕ) (h1 : rows_left = 12) (h2 : rows_right = 11) (h3 : rows_front = 18) (h4 : rows_back = 8) :
    (rows_left + rows_right - 1) * (rows_front + rows_back - 1) = 550 := 
by
  sorry

end NUMINAMATH_GPT_students_on_playground_l974_97403


namespace NUMINAMATH_GPT_arithmetic_sequence_a3_l974_97443

variable (a : ℕ → ℕ)
variable (S5 : ℕ)
variable (arithmetic_seq : Prop)

def is_arithmetic_seq (a : ℕ → ℕ) : Prop := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_a3 (h1 : is_arithmetic_seq a) (h2 : (a 1 + a 2 + a 3 + a 4 + a 5) = 25) : a 3 = 5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a3_l974_97443


namespace NUMINAMATH_GPT_ababab_divisible_by_7_l974_97416

theorem ababab_divisible_by_7 (a b : ℕ) (ha : a < 10) (hb : b < 10) : (101010 * a + 10101 * b) % 7 = 0 :=
by sorry

end NUMINAMATH_GPT_ababab_divisible_by_7_l974_97416


namespace NUMINAMATH_GPT_sqrt_2_plus_x_nonnegative_l974_97412

theorem sqrt_2_plus_x_nonnegative (x : ℝ) : (2 + x ≥ 0) → (x ≥ -2) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_2_plus_x_nonnegative_l974_97412


namespace NUMINAMATH_GPT_sum_of_three_distinct_integers_l974_97463

theorem sum_of_three_distinct_integers (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c) 
  (h₄ : a * b * c = 5^3) (h₅ : a > 0) (h₆ : b > 0) (h₇ : c > 0) : a + b + c = 31 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_distinct_integers_l974_97463


namespace NUMINAMATH_GPT_pradeep_maximum_marks_l974_97426

theorem pradeep_maximum_marks (M : ℝ) (h1 : 0.20 * M = 185) : M = 925 :=
by
  sorry

end NUMINAMATH_GPT_pradeep_maximum_marks_l974_97426


namespace NUMINAMATH_GPT_lateral_surface_area_of_pyramid_l974_97423

theorem lateral_surface_area_of_pyramid
  (sin_alpha : ℝ)
  (A_section : ℝ)
  (h1 : sin_alpha = 15 / 17)
  (h2 : A_section = 3 * Real.sqrt 34) :
  ∃ A_lateral : ℝ, A_lateral = 68 :=
sorry

end NUMINAMATH_GPT_lateral_surface_area_of_pyramid_l974_97423


namespace NUMINAMATH_GPT_steve_oranges_count_l974_97402

variable (Brian_oranges Marcie_oranges Shawn_oranges Steve_oranges : ℝ)

def oranges_conditions : Prop :=
  (Marcie_oranges = 12) ∧
  (Brian_oranges = Marcie_oranges) ∧
  (Shawn_oranges = 1.075 * (Brian_oranges + Marcie_oranges)) ∧
  (Steve_oranges = 3 * (Marcie_oranges + Brian_oranges + Shawn_oranges))

theorem steve_oranges_count (h : oranges_conditions Brian_oranges Marcie_oranges Shawn_oranges Steve_oranges) :
  Steve_oranges = 149.4 :=
sorry

end NUMINAMATH_GPT_steve_oranges_count_l974_97402


namespace NUMINAMATH_GPT_cara_cats_correct_l974_97452

def martha_cats_rats : ℕ := 3
def martha_cats_birds : ℕ := 7
def martha_cats_animals : ℕ := martha_cats_rats + martha_cats_birds

def cara_cats_animals : ℕ := 5 * martha_cats_animals - 3

theorem cara_cats_correct : cara_cats_animals = 47 :=
by
  -- Proof omitted
  -- Here's where the actual calculation steps would go, but we'll just use sorry for now.
  sorry

end NUMINAMATH_GPT_cara_cats_correct_l974_97452


namespace NUMINAMATH_GPT_hostel_food_duration_l974_97494

noncomputable def food_last_days (total_food_units daily_consumption_new: ℝ) : ℝ :=
  total_food_units / daily_consumption_new

theorem hostel_food_duration:
  let x : ℝ := 1 -- assuming x is a positive real number
  let men_initial := 100
  let women_initial := 100
  let children_initial := 50
  let total_days := 40
  let consumption_man := 3 * x
  let consumption_woman := 2 * x
  let consumption_child := 1 * x
  let food_sufficient_for := 250
  let total_food_units := 550 * x * 40
  let men_leave := 30
  let women_leave := 20
  let children_leave := 10
  let men_new := men_initial - men_leave
  let women_new := women_initial - women_leave
  let children_new := children_initial - children_leave
  let daily_consumption_new := 210 * x + 160 * x + 40 * x 
  (food_last_days total_food_units daily_consumption_new) = 22000 / 410 := 
by
  sorry

end NUMINAMATH_GPT_hostel_food_duration_l974_97494


namespace NUMINAMATH_GPT_main_theorem_l974_97454

theorem main_theorem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * c * a) + c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_main_theorem_l974_97454


namespace NUMINAMATH_GPT_regular_polygon_sides_l974_97493

theorem regular_polygon_sides (h : ∀ n : ℕ, 140 * n = 180 * (n - 2)) : n = 9 :=
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l974_97493


namespace NUMINAMATH_GPT_product_of_fractions_l974_97472

theorem product_of_fractions : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 :=
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_l974_97472


namespace NUMINAMATH_GPT_combined_solid_volume_l974_97432

open Real

noncomputable def volume_truncated_cone (R r h : ℝ) :=
  (1 / 3) * π * h * (R^2 + R * r + r^2)

noncomputable def volume_cylinder (r h : ℝ): ℝ :=
  π * r^2 * h

theorem combined_solid_volume :
  let R := 10
  let r := 3
  let h_cone := 8
  let h_cyl := 10
  volume_truncated_cone R r h_cone + volume_cylinder r h_cyl = (1382 * π) / 3 :=
  by
  sorry

end NUMINAMATH_GPT_combined_solid_volume_l974_97432


namespace NUMINAMATH_GPT_calculate_profit_l974_97420

def additional_cost (purchase_cost : ℕ) : ℕ := (purchase_cost * 20) / 100

def total_feeding_cost (purchase_cost : ℕ) : ℕ := purchase_cost + additional_cost purchase_cost

def total_cost (purchase_cost : ℕ) (feeding_cost : ℕ) : ℕ := purchase_cost + feeding_cost

def selling_price_per_cow (weight : ℕ) (price_per_pound : ℕ) : ℕ := weight * price_per_pound

def total_revenue (price_per_cow : ℕ) (number_of_cows : ℕ) : ℕ := price_per_cow * number_of_cows

def profit (revenue : ℕ) (total_cost : ℕ) : ℕ := revenue - total_cost

def purchase_cost : ℕ := 40000
def number_of_cows : ℕ := 100
def weight_per_cow : ℕ := 1000
def price_per_pound : ℕ := 2

-- The theorem to prove
theorem calculate_profit : 
  profit (total_revenue (selling_price_per_cow weight_per_cow price_per_pound) number_of_cows) 
         (total_cost purchase_cost (total_feeding_cost purchase_cost)) = 112000 := by
  sorry

end NUMINAMATH_GPT_calculate_profit_l974_97420


namespace NUMINAMATH_GPT_perimeter_of_plot_l974_97421

theorem perimeter_of_plot
  (width : ℝ) 
  (cost_per_meter : ℝ)
  (total_cost : ℝ)
  (h1 : cost_per_meter = 6.5)
  (h2 : total_cost = 1170)
  (h3 : total_cost = (2 * (width + (width + 10))) * cost_per_meter) 
  :
  (2 * ((width + 10) + width)) = 180 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_plot_l974_97421


namespace NUMINAMATH_GPT_total_earnings_l974_97431

-- Define the constants and conditions.
def regular_hourly_rate : ℕ := 5
def overtime_hourly_rate : ℕ := 6
def regular_hours_per_week : ℕ := 40
def first_week_hours : ℕ := 44
def second_week_hours : ℕ := 48

-- Define the proof problem in Lean 4.
theorem total_earnings : (regular_hours_per_week * 2 * regular_hourly_rate + 
                         ((first_week_hours - regular_hours_per_week) + 
                          (second_week_hours - regular_hours_per_week)) * overtime_hourly_rate) = 472 := 
by 
  exact sorry -- Detailed proof steps would go here.

end NUMINAMATH_GPT_total_earnings_l974_97431


namespace NUMINAMATH_GPT_sum_of_roots_l974_97406

theorem sum_of_roots (g : ℝ → ℝ) 
  (h_symmetry : ∀ x : ℝ, g (3 + x) = g (3 - x))
  (h_roots : ∃ s1 s2 s3 s4 : ℝ, 
               g s1 = 0 ∧ 
               g s2 = 0 ∧ 
               g s3 = 0 ∧ 
               g s4 = 0 ∧ 
               s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ 
               s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4) :
  s1 + s2 + s3 + s4 = 12 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_roots_l974_97406


namespace NUMINAMATH_GPT_ezekiel_shoes_l974_97404

theorem ezekiel_shoes (pairs : ℕ) (shoes_per_pair : ℕ) (bought_pairs : pairs = 3) (pair_contains : shoes_per_pair = 2) : pairs * shoes_per_pair = 6 := by
  sorry

end NUMINAMATH_GPT_ezekiel_shoes_l974_97404


namespace NUMINAMATH_GPT_percentage_increase_l974_97445

theorem percentage_increase (L : ℕ) (h1 : L + 450 = 1350) :
  (450 / L : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l974_97445


namespace NUMINAMATH_GPT_distance_is_12_l974_97439

def distance_to_Mount_Overlook (D : ℝ) : Prop :=
  let T1 := D / 4
  let T2 := D / 6
  T1 + T2 = 5

theorem distance_is_12 : ∃ D : ℝ, distance_to_Mount_Overlook D ∧ D = 12 :=
by
  use 12
  rw [distance_to_Mount_Overlook]
  sorry

end NUMINAMATH_GPT_distance_is_12_l974_97439


namespace NUMINAMATH_GPT_probability_C_and_D_l974_97481

theorem probability_C_and_D (P_A P_B : ℚ) (H1 : P_A = 1/4) (H2 : P_B = 1/3) :
  P_C + P_D = 5/12 :=
by
  sorry

end NUMINAMATH_GPT_probability_C_and_D_l974_97481


namespace NUMINAMATH_GPT_range_of_a_l974_97479

variable {x a : ℝ}

theorem range_of_a (hx : 1 ≤ x ∧ x ≤ 2) (h : 2 * x > a - x^2) : a < 8 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l974_97479


namespace NUMINAMATH_GPT_find_sum_l974_97408

theorem find_sum (I r1 r2 r3 r4 r5: ℝ) (t1 t2 t3 t4 t5 : ℝ) (P: ℝ) 
  (hI: I = 6016.75)
  (hr1: r1 = 0.06) (hr2: r2 = 0.075) (hr3: r3 = 0.08) (hr4: r4 = 0.085) (hr5: r5 = 0.09)
  (ht: ∀ i, (i = t1 ∨ i = t2 ∨ i = t3 ∨ i = t4 ∨ i = t5) → i = 1): 
  I = P * (r1 * t1 + r2 * t2 + r3 * t3 + r4 * t4 + r5 * t5) → P = 15430 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_l974_97408


namespace NUMINAMATH_GPT_find_three_numbers_l974_97415

theorem find_three_numbers (x y z : ℝ)
  (h1 : x - y = (1 / 3) * z)
  (h2 : y - z = (1 / 3) * x)
  (h3 : z - 10 = (1 / 3) * y) :
  x = 45 ∧ y = 37.5 ∧ z = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_find_three_numbers_l974_97415


namespace NUMINAMATH_GPT_triangle_inequality_l974_97497

theorem triangle_inequality
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : c > 0)
  (h5 : a + b > c)
  (h6 : a + c > b)
  (h7 : b + c > a) :
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l974_97497


namespace NUMINAMATH_GPT_total_amount_l974_97433

-- Declare the variables
variables (A B C : ℕ)

-- Introduce the conditions as hypotheses
theorem total_amount (h1 : A = B + 40) (h2 : C = A + 30) (h3 : B = 290) : 
  A + B + C = 980 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_amount_l974_97433


namespace NUMINAMATH_GPT_am_gm_four_vars_l974_97427

theorem am_gm_four_vars {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / a + 1 / b + 1 / c + 1 / d) ≥ 16 :=
by
  sorry

end NUMINAMATH_GPT_am_gm_four_vars_l974_97427


namespace NUMINAMATH_GPT_initial_invited_people_l974_97448

theorem initial_invited_people (not_showed_up : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) 
  (H1 : not_showed_up = 12) (H2 : table_capacity = 3) (H3 : tables_needed = 2) :
  not_showed_up + (table_capacity * tables_needed) = 18 :=
by
  sorry

end NUMINAMATH_GPT_initial_invited_people_l974_97448


namespace NUMINAMATH_GPT_fraction_of_juniors_l974_97488

theorem fraction_of_juniors (J S : ℕ) (h1 : 0 < J) (h2 : 0 < S) (h3 : J = (4 / 3) * S) :
  (J : ℚ) / (J + S) = 4 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_juniors_l974_97488


namespace NUMINAMATH_GPT_xiaoming_minimum_time_l974_97444

theorem xiaoming_minimum_time :
  let review_time := 30
  let rest_time := 30
  let boil_time := 15
  let homework_time := 25
  (boil_time ≤ rest_time) → 
  (review_time + rest_time + homework_time = 85) :=
by
  intros review_time rest_time boil_time homework_time h_boil_le_rest
  sorry

end NUMINAMATH_GPT_xiaoming_minimum_time_l974_97444


namespace NUMINAMATH_GPT_part_one_part_two_l974_97425

variable {x m : ℝ}

theorem part_one (h1 : ∀ x : ℝ, ¬(m * x^2 - (m + 1) * x + (m + 1) ≥ 0)) : m < -1 := sorry

theorem part_two (h2 : ∀ x : ℝ, 1 < x → m * x^2 - (m + 1) * x + (m + 1) ≥ 0) : m ≥ 1 / 3 := sorry

end NUMINAMATH_GPT_part_one_part_two_l974_97425


namespace NUMINAMATH_GPT_find_x_l974_97473

variable (x : ℝ)
variable (y : ℝ := x * 3.5)
variable (z : ℝ := y / 0.00002)

theorem find_x (h : z = 840) : x = 0.0048 :=
sorry

end NUMINAMATH_GPT_find_x_l974_97473


namespace NUMINAMATH_GPT_breadth_of_rectangular_plot_l974_97422

theorem breadth_of_rectangular_plot :
  ∃ b : ℝ, (∃ l : ℝ, l = 3 * b ∧ l * b = 867) ∧ b = 17 :=
by
  sorry

end NUMINAMATH_GPT_breadth_of_rectangular_plot_l974_97422


namespace NUMINAMATH_GPT_greatest_two_digit_multiple_of_17_l974_97400

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end NUMINAMATH_GPT_greatest_two_digit_multiple_of_17_l974_97400


namespace NUMINAMATH_GPT_chickens_at_stacy_farm_l974_97477
-- Importing the necessary library

-- Defining the provided conditions and correct answer in Lean 4.
theorem chickens_at_stacy_farm (C : ℕ) (piglets : ℕ) (goats : ℕ) : 
  piglets = 40 → 
  goats = 34 → 
  (C + piglets + goats) = 2 * 50 → 
  C = 26 :=
by
  intros h_piglets h_goats h_animals
  sorry

end NUMINAMATH_GPT_chickens_at_stacy_farm_l974_97477


namespace NUMINAMATH_GPT_university_admission_l974_97466

def students_ratio (x y z : ℕ) : Prop :=
  x * 5 = y * 2 ∧ y * 3 = z * 5

def third_tier_students : ℕ := 1500

theorem university_admission :
  ∀ x y z : ℕ, students_ratio x y z → z = third_tier_students → y - x = 1500 :=
by
  intros x y z hratio hthird
  sorry

end NUMINAMATH_GPT_university_admission_l974_97466


namespace NUMINAMATH_GPT_find_n_l974_97496

theorem find_n : ∃ n : ℤ, 3^3 - 5 = 2^5 + n ∧ n = -10 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l974_97496


namespace NUMINAMATH_GPT_percentage_of_black_marbles_l974_97495

variable (T : ℝ) -- Total number of marbles
variable (C : ℝ) -- Number of clear marbles
variable (B : ℝ) -- Number of black marbles
variable (O : ℝ) -- Number of other colored marbles

-- Conditions
def condition1 := C = 0.40 * T
def condition2 := O = (2 / 5) * T
def condition3 := C + B + O = T

-- Proof statement
theorem percentage_of_black_marbles :
  C = 0.40 * T → O = (2 / 5) * T → C + B + O = T → B = 0.20 * T :=
by
  intros hC hO hTotal
  -- Intermediate steps would go here, but we use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_percentage_of_black_marbles_l974_97495


namespace NUMINAMATH_GPT_johns_age_l974_97451

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end NUMINAMATH_GPT_johns_age_l974_97451


namespace NUMINAMATH_GPT_determine_constant_l974_97417

theorem determine_constant (c : ℝ) :
  (∃ d : ℝ, 9 * x^2 - 24 * x + c = (3 * x + d)^2) ↔ c = 16 :=
by
  sorry

end NUMINAMATH_GPT_determine_constant_l974_97417


namespace NUMINAMATH_GPT_strictly_decreasing_interval_l974_97464

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem strictly_decreasing_interval :
  ∀ x, (0 < x) ∧ (x < 2) → (deriv f x < 0) := by
sorry

end NUMINAMATH_GPT_strictly_decreasing_interval_l974_97464


namespace NUMINAMATH_GPT_range_of_a_l974_97475

noncomputable def f (x a : ℝ) := 2^(2*x) - a * 2^x + 4

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l974_97475


namespace NUMINAMATH_GPT_trapezoid_height_l974_97461

theorem trapezoid_height (A : ℝ) (d1 d2 : ℝ) (h : ℝ) :
  A = 2 ∧ d1 + d2 = 4 → h = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_height_l974_97461


namespace NUMINAMATH_GPT_linear_decreasing_sequence_l974_97471

theorem linear_decreasing_sequence 
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (h_func1 : y1 = -3 * x1 + 1)
  (h_func2 : y2 = -3 * x2 + 1)
  (h_func3 : y3 = -3 * x3 + 1)
  (hx_seq : x1 < x2 ∧ x2 < x3)
  : y3 < y2 ∧ y2 < y1 := 
sorry

end NUMINAMATH_GPT_linear_decreasing_sequence_l974_97471


namespace NUMINAMATH_GPT_tan_simplification_l974_97442

theorem tan_simplification 
  (θ : ℝ) 
  (h : Real.tan θ = 3) : 
  (1 - Real.sin θ) / (Real.cos θ) - (Real.cos θ) / (1 + Real.sin θ) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_tan_simplification_l974_97442


namespace NUMINAMATH_GPT_fraction_value_l974_97489

theorem fraction_value (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : (x + y) / 3 = 5 / 3 :=
by sorry

end NUMINAMATH_GPT_fraction_value_l974_97489


namespace NUMINAMATH_GPT_minimum_value_of_y_l974_97456

-- Define the function y
noncomputable def y (x : ℝ) := 2 + 4 * x + 1 / x

-- Prove that the minimum value is 6 for x > 0
theorem minimum_value_of_y : ∃ (x : ℝ), x > 0 ∧ (∀ (y : ℝ), (2 + 4 * x + 1 / x) ≤ y) ∧ (2 + 4 * x + 1 / x) = 6 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_y_l974_97456


namespace NUMINAMATH_GPT_race_head_start_l974_97457

-- This statement defines the problem in Lean 4
theorem race_head_start (Va Vb L H : ℝ) 
(h₀ : Va = 51 / 44 * Vb) 
(h₁ : L / Va = (L - H) / Vb) : 
H = 7 / 51 * L := 
sorry

end NUMINAMATH_GPT_race_head_start_l974_97457


namespace NUMINAMATH_GPT_find_a_value_l974_97434

-- Definitions of conditions
def eq_has_positive_root (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (x / (x - 5) = 3 - (a / (x - 5)))

-- Statement of the theorem
theorem find_a_value (a : ℝ) (h : eq_has_positive_root a) : a = -5 := 
  sorry

end NUMINAMATH_GPT_find_a_value_l974_97434


namespace NUMINAMATH_GPT_tree_height_end_of_2_years_l974_97446

theorem tree_height_end_of_2_years (h4 : ℕ → ℕ)
  (h_tripling : ∀ n, h4 (n + 1) = 3 * h4 n)
  (h4_at_4 : h4 4 = 81) :
  h4 2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_tree_height_end_of_2_years_l974_97446


namespace NUMINAMATH_GPT_range_of_a_proof_l974_97467

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

theorem range_of_a_proof (a : ℝ) : range_of_a a ↔ 0 ≤ a ∧ a < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_proof_l974_97467


namespace NUMINAMATH_GPT_example_problem_l974_97458

variables (a b : ℕ)

def HCF (m n : ℕ) : ℕ := m.gcd n
def LCM (m n : ℕ) : ℕ := m.lcm n

theorem example_problem (hcf_ab : HCF 385 180 = 30) (a_def: a = 385) (b_def: b = 180) :
  LCM 385 180 = 2310 := 
by
  sorry

end NUMINAMATH_GPT_example_problem_l974_97458


namespace NUMINAMATH_GPT_blue_balls_needed_l974_97490

theorem blue_balls_needed 
  (G B Y W : ℝ)
  (h1 : G = 2 * B)
  (h2 : Y = (8 / 3) * B)
  (h3 : W = (4 / 3) * B) :
  5 * G + 3 * Y + 4 * W = (70 / 3) * B :=
by
  sorry

end NUMINAMATH_GPT_blue_balls_needed_l974_97490


namespace NUMINAMATH_GPT_minimizing_reciprocal_sum_l974_97455

theorem minimizing_reciprocal_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a + 4 * b = 30) :
  a = 10 ∧ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_minimizing_reciprocal_sum_l974_97455


namespace NUMINAMATH_GPT_snow_on_Monday_l974_97437

def snow_on_Tuesday : ℝ := 0.21
def snow_on_Monday_and_Tuesday : ℝ := 0.53

theorem snow_on_Monday : snow_on_Monday_and_Tuesday - snow_on_Tuesday = 0.32 :=
by
  sorry

end NUMINAMATH_GPT_snow_on_Monday_l974_97437


namespace NUMINAMATH_GPT_div_expression_l974_97470

theorem div_expression : (124 : ℝ) / (8 + 14 * 3) = 2.48 := by
  sorry

end NUMINAMATH_GPT_div_expression_l974_97470


namespace NUMINAMATH_GPT_alice_total_distance_correct_l974_97405

noncomputable def alice_daily_morning_distance : ℕ := 10

noncomputable def alice_daily_afternoon_distance : ℕ := 12

noncomputable def alice_daily_distance : ℕ :=
  alice_daily_morning_distance + alice_daily_afternoon_distance

noncomputable def alice_weekly_distance : ℕ :=
  5 * alice_daily_distance

theorem alice_total_distance_correct :
  alice_weekly_distance = 110 :=
by
  unfold alice_weekly_distance alice_daily_distance alice_daily_morning_distance alice_daily_afternoon_distance
  norm_num

end NUMINAMATH_GPT_alice_total_distance_correct_l974_97405


namespace NUMINAMATH_GPT_sqrt_14_bounds_l974_97480

theorem sqrt_14_bounds : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end NUMINAMATH_GPT_sqrt_14_bounds_l974_97480


namespace NUMINAMATH_GPT_discount_percentage_l974_97499

theorem discount_percentage 
    (original_price : ℝ) 
    (total_paid : ℝ) 
    (sales_tax_rate : ℝ) 
    (sale_price_before_tax : ℝ) 
    (discount_amount : ℝ) 
    (discount_percentage : ℝ) :
    original_price = 200 → total_paid = 165 → sales_tax_rate = 0.10 →
    total_paid = sale_price_before_tax * (1 + sales_tax_rate) →
    sale_price_before_tax = original_price - discount_amount →
    discount_percentage = (discount_amount / original_price) * 100 →
    discount_percentage = 25 :=
by
  intros h_original h_total h_tax h_eq1 h_eq2 h_eq3
  sorry

end NUMINAMATH_GPT_discount_percentage_l974_97499


namespace NUMINAMATH_GPT_sushil_marks_ratio_l974_97491

theorem sushil_marks_ratio
  (E M Science : ℕ)
  (h1 : E + M + Science = 170)
  (h2 : E = M / 4)
  (h3 : Science = 17) :
  E = 31 :=
by
  sorry

end NUMINAMATH_GPT_sushil_marks_ratio_l974_97491


namespace NUMINAMATH_GPT_geometric_sequence_a7_l974_97462

-- Define the geometric sequence
def geometic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Conditions
def a1 (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def a2a4 (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 = 16

-- The statement to prove
theorem geometric_sequence_a7 (a : ℕ → ℝ) (h1 : a1 a) (h2 : a2a4 a) (gs : geometic_sequence a) :
  a 7 = 64 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a7_l974_97462


namespace NUMINAMATH_GPT_domain_of_inverse_l974_97468

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x - 1) + 1

theorem domain_of_inverse :
  ∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ y = f x) → (y ∈ Set.Icc (3/2) 3) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_inverse_l974_97468


namespace NUMINAMATH_GPT_max_of_three_diff_pos_int_with_mean_7_l974_97401

theorem max_of_three_diff_pos_int_with_mean_7 (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_mean : (a + b + c) / 3 = 7) :
  max a (max b c) = 18 := 
sorry

end NUMINAMATH_GPT_max_of_three_diff_pos_int_with_mean_7_l974_97401


namespace NUMINAMATH_GPT_largest_divisor_of_seven_consecutive_odd_numbers_l974_97411

theorem largest_divisor_of_seven_consecutive_odd_numbers (n : ℕ) (h : Even n) (h_pos : n > 0) :
  ∃ d, d = 45 ∧ ∀ k, k ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) → k ≤ 45 :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_seven_consecutive_odd_numbers_l974_97411


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l974_97459

noncomputable def geometric_sequence_pos (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a n > 0 ∧ a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h : geometric_sequence_pos a q) (h_q : q^2 = 4) :
  (a 2 + a 3) / (a 3 + a 4) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l974_97459


namespace NUMINAMATH_GPT_fraction_sum_l974_97476

theorem fraction_sum :
  (7 : ℚ) / 12 + (3 : ℚ) / 8 = 23 / 24 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_fraction_sum_l974_97476


namespace NUMINAMATH_GPT_ellen_painted_roses_l974_97409

theorem ellen_painted_roses :
  ∀ (r : ℕ),
    (5 * 17 + 7 * r + 3 * 6 + 2 * 20 = 213) → (r = 10) :=
by
  intros r h
  sorry

end NUMINAMATH_GPT_ellen_painted_roses_l974_97409


namespace NUMINAMATH_GPT_cody_steps_l974_97460

theorem cody_steps (S steps_week1 steps_week2 steps_week3 steps_week4 total_steps_4weeks : ℕ) 
  (h1 : steps_week1 = 7 * S) 
  (h2 : steps_week2 = 7 * (S + 1000)) 
  (h3 : steps_week3 = 7 * (S + 2000)) 
  (h4 : steps_week4 = 7 * (S + 3000)) 
  (h5 : total_steps_4weeks = steps_week1 + steps_week2 + steps_week3 + steps_week4) 
  (h6 : total_steps_4weeks = 70000) : 
  S = 1000 := 
    sorry

end NUMINAMATH_GPT_cody_steps_l974_97460


namespace NUMINAMATH_GPT_Aarti_work_days_l974_97436

theorem Aarti_work_days (x : ℕ) : (3 * x = 24) → x = 8 := by
  intro h
  linarith

end NUMINAMATH_GPT_Aarti_work_days_l974_97436


namespace NUMINAMATH_GPT_bike_average_speed_l974_97474

theorem bike_average_speed (distance time : ℕ)
    (h1 : distance = 48)
    (h2 : time = 6) :
    distance / time = 8 := 
  by
    sorry

end NUMINAMATH_GPT_bike_average_speed_l974_97474


namespace NUMINAMATH_GPT_solution_set_A_solution_set_B_subset_A_l974_97450

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem solution_set_A :
  {x : ℝ | f x > 6} = {x : ℝ | x < -1 ∨ x > 2} :=
sorry

theorem solution_set_B_subset_A {a : ℝ} :
  (∀ x, f x > |a-1| → x < -1 ∨ x > 2) → a ≤ -5 ∨ a ≥ 7 :=
sorry

end NUMINAMATH_GPT_solution_set_A_solution_set_B_subset_A_l974_97450


namespace NUMINAMATH_GPT_units_digit_of_m3_plus_2m_l974_97483

def m : ℕ := 2021^2 + 2^2021

theorem units_digit_of_m3_plus_2m : (m^3 + 2^m) % 10 = 5 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_m3_plus_2m_l974_97483


namespace NUMINAMATH_GPT_closely_related_interval_unique_l974_97485

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
noncomputable def g (x : ℝ) : ℝ := 2 * x - 3

def closely_related (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

theorem closely_related_interval_unique :
  closely_related f g 2 3 :=
sorry

end NUMINAMATH_GPT_closely_related_interval_unique_l974_97485


namespace NUMINAMATH_GPT_NoahClosetsFit_l974_97435

-- Declare the conditions as Lean variables and proofs
variable (AliClosetCapacity : ℕ) (NoahClosetsRatio : ℕ) (NoahClosetsCount : ℕ)
variable (H1 : AliClosetCapacity = 200)
variable (H2 : NoahClosetsRatio = 1 / 4)
variable (H3 : NoahClosetsCount = 2)

-- Define the total number of jeans both of Noah's closets can fit
noncomputable def NoahTotalJeans : ℕ := (AliClosetCapacity * NoahClosetsRatio) * NoahClosetsCount

-- Theorem to prove
theorem NoahClosetsFit (AliClosetCapacity : ℕ) (NoahClosetsRatio : ℕ) (NoahClosetsCount : ℕ)
  (H1 : AliClosetCapacity = 200) 
  (H2 : NoahClosetsRatio = 1 / 4) 
  (H3 : NoahClosetsCount = 2) 
  : NoahTotalJeans AliClosetCapacity NoahClosetsRatio NoahClosetsCount = 100 := 
  by 
    sorry

end NUMINAMATH_GPT_NoahClosetsFit_l974_97435


namespace NUMINAMATH_GPT_floor_sqrt_27_square_l974_97469

theorem floor_sqrt_27_square : (Int.floor (Real.sqrt 27))^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_floor_sqrt_27_square_l974_97469


namespace NUMINAMATH_GPT_range_of_m_l974_97419

variable (m : ℝ)
def p := ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → x^2 - 2*x - 4*m^2 + 8*m - 2 ≥ 0
def q := ∃ x : ℝ, x ∈ Set.Icc (1 : ℝ) 2 ∧ Real.log (x^2 - m*x + 1) / Real.log (1/2) < -1

theorem range_of_m (hp : p m) (hq : q m) (hl : (p m) ∨ (q m)) (hf : ¬ ((p m) ∧ (q m))) :
  m < 1/2 ∨ m = 3/2 := sorry

end NUMINAMATH_GPT_range_of_m_l974_97419


namespace NUMINAMATH_GPT_find_first_train_length_l974_97418

theorem find_first_train_length
  (length_second_train : ℝ)
  (initial_distance : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_minutes : ℝ) :
  length_second_train = 200 →
  initial_distance = 100 →
  speed_first_train_kmph = 54 →
  speed_second_train_kmph = 72 →
  time_minutes = 2.856914303998537 →
  ∃ (L : ℝ), L = 5699.52 :=
by
  sorry

end NUMINAMATH_GPT_find_first_train_length_l974_97418


namespace NUMINAMATH_GPT_probability_red_or_blue_is_713_l974_97492

-- Definition of area ratios
def area_ratio_red : ℕ := 6
def area_ratio_yellow : ℕ := 2
def area_ratio_blue : ℕ := 1
def area_ratio_black : ℕ := 4

-- Total area ratio
def total_area_ratio := area_ratio_red + area_ratio_yellow + area_ratio_blue + area_ratio_black

-- Probability of stopping on either red or blue
def probability_red_or_blue := (area_ratio_red + area_ratio_blue) / total_area_ratio

-- Theorem stating the probability is 7/13
theorem probability_red_or_blue_is_713 : probability_red_or_blue = 7 / 13 :=
by
  unfold probability_red_or_blue total_area_ratio area_ratio_red area_ratio_blue
  simp
  sorry

end NUMINAMATH_GPT_probability_red_or_blue_is_713_l974_97492


namespace NUMINAMATH_GPT_brianna_initial_marbles_l974_97413

-- Defining the variables and constants
def initial_marbles : Nat := 24
def marbles_lost : Nat := 4
def marbles_given : Nat := 2 * marbles_lost
def marbles_ate : Nat := marbles_lost / 2
def marbles_remaining : Nat := 10

-- The main statement to prove
theorem brianna_initial_marbles :
  marbles_remaining + marbles_ate + marbles_given + marbles_lost = initial_marbles :=
by
  sorry

end NUMINAMATH_GPT_brianna_initial_marbles_l974_97413


namespace NUMINAMATH_GPT_minimal_n_is_40_l974_97428

def sequence_minimal_n (p : ℝ) (a : ℕ → ℝ) : Prop :=
  a 1 = p ∧
  a 2 = p + 1 ∧
  (∀ n, n ≥ 1 → a (n + 2) - 2 * a (n + 1) + a n = n - 20) ∧
  (∀ n, a n ≥ p) -- Since minimal \(a_n\) implies non-negative with given \(a_1, a_2\)

theorem minimal_n_is_40 (p : ℝ) (a : ℕ → ℝ) (h : sequence_minimal_n p a) : ∃ n, n = 40 ∧ (∀ m, n ≠ m → a n ≤ a m) :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_minimal_n_is_40_l974_97428


namespace NUMINAMATH_GPT_average_after_17th_inning_l974_97482

-- Define the conditions.
variable (A : ℚ) -- The initial average after 16 innings

-- Define the score in the 17th inning and the increment in the average.
def runs_in_17th_inning : ℚ := 87
def increment_in_average : ℚ := 3

-- Define the equation derived from the conditions.
theorem average_after_17th_inning :
  (16 * A + runs_in_17th_inning) / 17 = A + increment_in_average →
  A + increment_in_average = 39 :=
sorry

end NUMINAMATH_GPT_average_after_17th_inning_l974_97482


namespace NUMINAMATH_GPT_locus_of_point_P_l974_97465

theorem locus_of_point_P (P : ℝ × ℝ) (M N : ℝ × ℝ)
  (hxM : M = (-2, 0))
  (hxN : N = (2, 0))
  (hxPM : P.fst ^ 2 + (P.snd - 0) ^ 2 = xPM)
  (hxPN : P.fst ^ 2 + (P.snd - 0) ^ 2 = xPN)
  : P.fst ^ 2 + P.snd ^ 2 = 4 ∧ P.fst ≠ 2 ∧ P.fst ≠ -2 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_locus_of_point_P_l974_97465


namespace NUMINAMATH_GPT_average_age_of_two_new_men_l974_97486

theorem average_age_of_two_new_men :
  ∀ (A N : ℕ), 
    (∀ n : ℕ, n = 12) → 
    (N = 21 + 23 + 12) → 
    (A = N / 2) → 
    A = 28 :=
by
  intros A N twelve men_replace_eq_avg men_avg_eq
  sorry

end NUMINAMATH_GPT_average_age_of_two_new_men_l974_97486


namespace NUMINAMATH_GPT_larger_triangle_perimeter_l974_97430

def is_similar (a b c : ℕ) (x y z : ℕ) : Prop :=
  x * c = z * a ∧
  x * c = z * b ∧
  y * c = z * a ∧
  y * c = z * c ∧
  a ≠ b ∧ c ≠ b

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∧ a ≠ c

theorem larger_triangle_perimeter (a b c x y z : ℕ) 
  (h1 : is_isosceles a b c) 
  (h2 : is_similar a b c x y z) 
  (h3 : c = 12) 
  (h4 : z = 36)
  (h5 : a = 7) 
  (h6 : b = 7) : 
  x + y + z = 78 :=
sorry

end NUMINAMATH_GPT_larger_triangle_perimeter_l974_97430


namespace NUMINAMATH_GPT_Jamie_owns_2_Maine_Coons_l974_97414

-- Definitions based on conditions
variables (Jamie_MaineCoons Gordon_MaineCoons Hawkeye_MaineCoons Jamie_Persians Gordon_Persians Hawkeye_Persians : ℕ)

-- Conditions
axiom Jamie_owns_4_Persians : Jamie_Persians = 4
axiom Gordon_owns_half_as_many_Persians_as_Jamie : Gordon_Persians = Jamie_Persians / 2
axiom Gordon_owns_one_more_Maine_Coon_than_Jamie : Gordon_MaineCoons = Jamie_MaineCoons + 1
axiom Hawkeye_owns_one_less_Maine_Coon_than_Gordon : Hawkeye_MaineCoons = Gordon_MaineCoons - 1
axiom Hawkeye_owns_no_Persian_cats : Hawkeye_Persians = 0
axiom total_number_of_cats_is_13 : Jamie_Persians + Jamie_MaineCoons + Gordon_Persians + Gordon_MaineCoons + Hawkeye_Persians + Hawkeye_MaineCoons = 13

-- Theorem statement
theorem Jamie_owns_2_Maine_Coons : Jamie_MaineCoons = 2 :=
by {
  -- Ideally, you would provide the proof here, stepping through algebraically as shown in the solution,
  -- but we are skipping the proof as specified in the instructions.
  sorry
}

end NUMINAMATH_GPT_Jamie_owns_2_Maine_Coons_l974_97414


namespace NUMINAMATH_GPT_least_value_a2000_l974_97453

theorem least_value_a2000 (a : ℕ → ℕ)
  (h1 : ∀ m n, (m ∣ n) → (m < n) → (a m ∣ a n))
  (h2 : ∀ m n, (m ∣ n) → (m < n) → (a m < a n)) :
  a 2000 >= 128 :=
sorry

end NUMINAMATH_GPT_least_value_a2000_l974_97453


namespace NUMINAMATH_GPT_length_of_MN_l974_97487

theorem length_of_MN (b : ℝ) (h_focus : ∃ b : ℝ, (3/2, b).1 > 0 ∧ (3/2, b).2 * (3/2, b).2 = 6 * (3 / 2)) : 
  |2 * b| = 6 :=
by sorry

end NUMINAMATH_GPT_length_of_MN_l974_97487


namespace NUMINAMATH_GPT_investment_calculation_l974_97478

theorem investment_calculation
  (face_value : ℝ)
  (market_price : ℝ)
  (rate_of_dividend : ℝ)
  (annual_income : ℝ)
  (h1 : face_value = 10)
  (h2 : market_price = 8.25)
  (h3 : rate_of_dividend = 12)
  (h4 : annual_income = 648) :
  ∃ investment : ℝ, investment = 4455 :=
by
  sorry

end NUMINAMATH_GPT_investment_calculation_l974_97478


namespace NUMINAMATH_GPT_circle_center_and_sum_l974_97410

/-- Given the equation of a circle x^2 + y^2 - 6x + 14y = -28,
    prove that the coordinates (h, k) of the center of the circle are (3, -7)
    and compute h + k. -/
theorem circle_center_and_sum (x y : ℝ) :
  (∃ h k, (x^2 + y^2 - 6*x + 14*y = -28) ∧ (h = 3) ∧ (k = -7) ∧ (h + k = -4)) :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_center_and_sum_l974_97410


namespace NUMINAMATH_GPT_pyramid_distance_to_larger_cross_section_l974_97438

theorem pyramid_distance_to_larger_cross_section
  (A1 A2 : ℝ) (d : ℝ)
  (h : ℝ)
  (hA1 : A1 = 256 * Real.sqrt 2)
  (hA2 : A2 = 576 * Real.sqrt 2)
  (hd : d = 12)
  (h_ratio : (Real.sqrt (A1 / A2)) = 2 / 3) :
  h = 36 := 
  sorry

end NUMINAMATH_GPT_pyramid_distance_to_larger_cross_section_l974_97438


namespace NUMINAMATH_GPT_min_value_z_l974_97498

theorem min_value_z : ∃ (min_z : ℝ), min_z = 24.1 ∧ 
  ∀ (x y : ℝ), (3 * x ^ 2 + 4 * y ^ 2 + 8 * x - 6 * y + 30) ≥ min_z :=
sorry

end NUMINAMATH_GPT_min_value_z_l974_97498


namespace NUMINAMATH_GPT_jar_total_value_l974_97429

def total_value_in_jar (p n q : ℕ) (total_coins : ℕ) (value : ℝ) : Prop :=
  p + n + q = total_coins ∧
  n = 3 * p ∧
  q = 4 * n ∧
  value = p * 0.01 + n * 0.05 + q * 0.25

theorem jar_total_value (p : ℕ) (h₁ : 16 * p = 240) : 
  ∃ value, total_value_in_jar p (3 * p) (12 * p) 240 value ∧ value = 47.4 :=
by
  sorry

end NUMINAMATH_GPT_jar_total_value_l974_97429


namespace NUMINAMATH_GPT_time_to_send_data_in_minutes_l974_97407

def blocks := 100
def chunks_per_block := 256
def transmission_rate := 100 -- chunks per second
def seconds_per_minute := 60

theorem time_to_send_data_in_minutes :
    (blocks * chunks_per_block) / transmission_rate / seconds_per_minute = 4 := by
  sorry

end NUMINAMATH_GPT_time_to_send_data_in_minutes_l974_97407


namespace NUMINAMATH_GPT_calculation_l974_97440

theorem calculation (a b : ℕ) (h1 : a = 7) (h2 : b = 5) : (a^2 - b^2) ^ 2 = 576 :=
by
  sorry

end NUMINAMATH_GPT_calculation_l974_97440


namespace NUMINAMATH_GPT_increasing_function_unique_root_proof_l974_97441

noncomputable def increasing_function_unique_root (f : ℝ → ℝ) :=
  (∀ x y : ℝ, x < y → f x ≤ f y) -- condition for increasing function
  ∧ ∃! x : ℝ, f x = 0 -- exists exactly one root

theorem increasing_function_unique_root_proof
  (f : ℝ → ℝ)
  (h_inc : ∀ x y : ℝ, x < y → f x ≤ f y)
  (h_ex : ∃ x : ℝ, f x = 0) :
  ∃! x : ℝ, f x = 0 := sorry

end NUMINAMATH_GPT_increasing_function_unique_root_proof_l974_97441


namespace NUMINAMATH_GPT_number_of_scoops_l974_97449

/-- Pierre gets 3 scoops of ice cream given the conditions described -/
theorem number_of_scoops (P : ℕ) (cost_per_scoop total_bill : ℝ) (mom_scoops : ℕ)
  (h1 : cost_per_scoop = 2) 
  (h2 : mom_scoops = 4) 
  (h3 : total_bill = 14) 
  (h4 : cost_per_scoop * P + cost_per_scoop * mom_scoops = total_bill) :
  P = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_scoops_l974_97449


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l974_97447

variables (a_n : Nat → Int) (S_n : Nat → Int)
variable (n : Nat)

-- Definitions based on given conditions:
def is_arithmetic_sequence (a_n : Nat → Int) : Prop :=
∀ n, a_n (n + 1) = a_n n + a_n 1 - a_n 0

def a_1 : Int := -2018

def arithmetic_sequence_sum (S_n : Nat → Int) (a_n : Nat → Int) (n : Nat) : Prop :=
S_n n = n * a_n 0 + (n * (n - 1) / 2 * (a_n 1 - a_n 0))

-- Given condition S_12 / 12 - S_10 / 10 = 2
def condition (S_n : Nat → Int) : Prop :=
S_n 12 / 12 - S_n 10 / 10 = 2

-- Goal: Prove S_2018 = -2018
theorem sum_of_arithmetic_sequence (a_n S_n : Nat → Int)
  (h1 : a_n 1 = -2018)
  (h2 : is_arithmetic_sequence a_n)
  (h3 : ∀ n, arithmetic_sequence_sum S_n a_n n)
  (h4 : condition S_n) :
  S_n 2018 = -2018 :=
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l974_97447


namespace NUMINAMATH_GPT_calc_eq_neg_ten_thirds_l974_97484

theorem calc_eq_neg_ten_thirds :
  (7 / 4 - 7 / 8 - 7 / 12) / (-7 / 8) + (-7 / 8) / (7 / 4 - 7 / 8 - 7 / 12) = -10 / 3 := by 
sorry

end NUMINAMATH_GPT_calc_eq_neg_ten_thirds_l974_97484


namespace NUMINAMATH_GPT_roman_created_171_roman_created_1513_m1_roman_created_1513_m2_roman_created_largest_l974_97424

-- Lean 4 statements to capture the proofs without computation.
theorem roman_created_171 (a b : ℕ) (h_sum : a + b = 17) (h_diff : a - b = 1) : 
  a = 9 ∧ b = 8 ∨ a = 8 ∧ b = 9 := 
  sorry

theorem roman_created_1513_m1 (a b : ℕ) (h_sum : a + b = 15) (h_diff : a - b = 13) : 
  a = 14 ∧ b = 1 ∨ a = 1 ∧ b = 14 := 
  sorry

theorem roman_created_1513_m2 (a b : ℕ) (h_sum : a + b = 151) (h_diff : a - b = 3) : 
  a = 77 ∧ b = 74 ∨ a = 74 ∧ b = 77 := 
  sorry

theorem roman_created_largest (a b : ℕ) (h_sum : a + b = 188) (h_diff : a - b = 10) : 
  a = 99 ∧ b = 89 ∨ a = 89 ∧ b = 99 := 
  sorry

end NUMINAMATH_GPT_roman_created_171_roman_created_1513_m1_roman_created_1513_m2_roman_created_largest_l974_97424
