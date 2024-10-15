import Mathlib

namespace NUMINAMATH_GPT_ball_probability_l1254_125456

theorem ball_probability (n : ℕ) (h : (n : ℚ) / (n + 2) = 1 / 3) : n = 1 :=
sorry

end NUMINAMATH_GPT_ball_probability_l1254_125456


namespace NUMINAMATH_GPT_range_sin_cos_two_x_is_minus2_to_9_over_8_l1254_125437

noncomputable def range_of_function : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = Real.sin x + Real.cos (2 * x) }

theorem range_sin_cos_two_x_is_minus2_to_9_over_8 :
  range_of_function = Set.Icc (-2) (9 / 8) := 
by
  sorry

end NUMINAMATH_GPT_range_sin_cos_two_x_is_minus2_to_9_over_8_l1254_125437


namespace NUMINAMATH_GPT_min_value_expression_l1254_125435

theorem min_value_expression : ∃ (x y : ℝ), x^2 + 2*x*y + 3*y^2 - 6*x - 2*y = -11 := by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1254_125435


namespace NUMINAMATH_GPT_range_of_a_for_f_zero_l1254_125428

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_f_zero (a : ℝ) :
  (∃ x : ℝ, f x a = 0) ↔ a ≤ 2 * Real.log 2 - 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_f_zero_l1254_125428


namespace NUMINAMATH_GPT_expandProduct_l1254_125405

theorem expandProduct (x : ℝ) : 4 * (x - 5) * (x + 8) = 4 * x^2 + 12 * x - 160 := 
by 
  sorry

end NUMINAMATH_GPT_expandProduct_l1254_125405


namespace NUMINAMATH_GPT_fraction_proof_l1254_125446

-- Define the fractions as constants
def a := 1 / 3
def b := 1 / 4
def c := 1 / 2
def d := 1 / 3

-- Prove the main statement
theorem fraction_proof : (a - b) / (c - d) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_fraction_proof_l1254_125446


namespace NUMINAMATH_GPT_jacob_river_water_collection_l1254_125412

/-- Definitions: 
1. Capacity of the tank in milliliters
2. Daily water collected from the rain in milliliters
3. Number of days to fill the tank
4. To be proved: Daily water collected from the river in milliliters
-/
def tank_capacity_ml : Int := 50000
def daily_rain_ml : Int := 800
def days_to_fill : Int := 20
def daily_river_ml : Int := 1700

/-- Prove that the amount of water Jacob collects from the river every day equals 1700 milliliters.
-/
theorem jacob_river_water_collection (total_water: Int) 
  (rain_water: Int) (days: Int) (correct_river_water: Int) : 
  total_water = tank_capacity_ml → 
  rain_water = daily_rain_ml → 
  days = days_to_fill → 
  correct_river_water = daily_river_ml → 
  (total_water - rain_water * days) / days = correct_river_water := 
by 
  intros; 
  sorry

end NUMINAMATH_GPT_jacob_river_water_collection_l1254_125412


namespace NUMINAMATH_GPT_find_a9_l1254_125459

variable {a_n : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable {d a₁ : ℤ}

-- Conditions
def arithmetic_sequence := ∀ n : ℕ, a_n n = a₁ + n * d
def sum_first_n_terms := ∀ n : ℕ, S n = (n * (2 * a₁ + (n - 1) * d)) / 2

-- Specific Conditions for the problem
axiom condition1 : S 8 = 4 * a₁
axiom condition2 : a_n 6 = -2 -- Note that a_n is 0-indexed here.

theorem find_a9 : a_n 8 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a9_l1254_125459


namespace NUMINAMATH_GPT_book_shelf_arrangement_l1254_125401

-- Definitions for the problem conditions
def math_books := 3
def english_books := 4
def science_books := 2

-- The total number of ways to arrange the books
def total_arrangements :=
  (Nat.factorial (math_books + english_books + science_books - 6)) * -- For the groups
  (Nat.factorial math_books) * -- For math books within the group
  (Nat.factorial english_books) * -- For English books within the group
  (Nat.factorial science_books) -- For science books within the group

theorem book_shelf_arrangement :
  total_arrangements = 1728 := by
  -- Proof starts here
  sorry

end NUMINAMATH_GPT_book_shelf_arrangement_l1254_125401


namespace NUMINAMATH_GPT_geometric_sequence_12th_term_l1254_125460

theorem geometric_sequence_12th_term 
  (a_4 a_8 : ℕ) (h4 : a_4 = 2) (h8 : a_8 = 162) :
  ∃ a_12 : ℕ, a_12 = 13122 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_12th_term_l1254_125460


namespace NUMINAMATH_GPT_unit_circle_inequality_l1254_125494

theorem unit_circle_inequality 
  (a b c d : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (habcd : a * b + c * d = 1) 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) 
  (hx1 : x1^2 + y1^2 = 1)
  (hx2 : x2^2 + y2^2 = 1)
  (hx3 : x3^2 + y3^2 = 1)
  (hx4 : x4^2 + y4^2 = 1) :
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2 ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := 
sorry

end NUMINAMATH_GPT_unit_circle_inequality_l1254_125494


namespace NUMINAMATH_GPT_no_negative_roots_l1254_125482

theorem no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_no_negative_roots_l1254_125482


namespace NUMINAMATH_GPT_calories_in_250_grams_is_106_l1254_125429

noncomputable def total_calories_apple : ℝ := 150 * (46 / 100)
noncomputable def total_calories_orange : ℝ := 50 * (45 / 100)
noncomputable def total_calories_carrot : ℝ := 300 * (40 / 100)
noncomputable def total_calories_mix : ℝ := total_calories_apple + total_calories_orange + total_calories_carrot
noncomputable def total_weight_mix : ℝ := 150 + 50 + 300
noncomputable def caloric_density : ℝ := total_calories_mix / total_weight_mix
noncomputable def calories_in_250_grams : ℝ := 250 * caloric_density

theorem calories_in_250_grams_is_106 : calories_in_250_grams = 106 :=
by
  sorry

end NUMINAMATH_GPT_calories_in_250_grams_is_106_l1254_125429


namespace NUMINAMATH_GPT_polynomial_remainder_l1254_125462

noncomputable def f (x : ℝ) : ℝ := x^4 + 2 * x^2 - 3
noncomputable def g (x : ℝ) : ℝ := x^2 + x - 2
noncomputable def r (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 3

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = g x * q x + r x :=
sorry

end NUMINAMATH_GPT_polynomial_remainder_l1254_125462


namespace NUMINAMATH_GPT_find_x_l1254_125448

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 6 * x^2 + 12 * x * y + 6 * y^2 = x^3 + 3 * x^2 * y + 3 * x * y^2) : x = 24 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1254_125448


namespace NUMINAMATH_GPT_triangle_side_lengths_l1254_125486

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end NUMINAMATH_GPT_triangle_side_lengths_l1254_125486


namespace NUMINAMATH_GPT_smallest_positive_period_minimum_value_of_f_l1254_125458

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos x * sin (x + π / 3) - sqrt 3 * sin x ^ 2 + sin x * cos x

theorem smallest_positive_period :
  ∀ x, f (x + π) = f x :=
sorry

theorem minimum_value_of_f :
  ∀ k : ℤ, f (k * π - 5 * π / 12) = -2 :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_minimum_value_of_f_l1254_125458


namespace NUMINAMATH_GPT_g_is_even_l1254_125468

noncomputable def g (x : ℝ) : ℝ := 5^(x^2 - 4) - |x|

theorem g_is_even : ∀ x : ℝ, g x = g (-x) :=
by
  sorry

end NUMINAMATH_GPT_g_is_even_l1254_125468


namespace NUMINAMATH_GPT_expected_worth_coin_flip_l1254_125457

def prob_head : ℚ := 2 / 3
def prob_tail : ℚ := 1 / 3
def gain_head : ℚ := 5
def loss_tail : ℚ := -12

theorem expected_worth_coin_flip : ∃ E : ℚ, E = round (((prob_head * gain_head) + (prob_tail * loss_tail)) * 100) / 100 ∧ E = - (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_expected_worth_coin_flip_l1254_125457


namespace NUMINAMATH_GPT_exists_set_with_property_l1254_125403

theorem exists_set_with_property (n : ℕ) (h : n > 0) :
  ∃ S : Finset ℕ, S.card = n ∧
  (∀ {a b}, a ∈ S → b ∈ S → a ≠ b → (a - b) ∣ a ∧ (a - b) ∣ b) ∧
  (∀ {a b c}, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → ¬ ((a - b) ∣ c)) :=
sorry

end NUMINAMATH_GPT_exists_set_with_property_l1254_125403


namespace NUMINAMATH_GPT_cheese_left_after_10_customers_l1254_125495

theorem cheese_left_after_10_customers :
  ∀ (S : ℕ → ℚ), (∀ n, S n = (20 * n) / (n + 10)) →
  20 - S 10 = 10 := by
  sorry

end NUMINAMATH_GPT_cheese_left_after_10_customers_l1254_125495


namespace NUMINAMATH_GPT_no_positive_abc_exists_l1254_125434

theorem no_positive_abc_exists 
  (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h1 : b^2 ≥ 4 * a * c)
  (h2 : c^2 ≥ 4 * b * a)
  (h3 : a^2 ≥ 4 * b * c)
  : false :=
sorry

end NUMINAMATH_GPT_no_positive_abc_exists_l1254_125434


namespace NUMINAMATH_GPT_verify_base_case_l1254_125496

theorem verify_base_case : 1 + (1 / 2) + (1 / 3) < 2 :=
sorry

end NUMINAMATH_GPT_verify_base_case_l1254_125496


namespace NUMINAMATH_GPT_midpoint_of_segment_l1254_125474

def z1 : ℂ := 2 + 4 * Complex.I  -- Define the first endpoint
def z2 : ℂ := -6 + 10 * Complex.I  -- Define the second endpoint

theorem midpoint_of_segment :
  (z1 + z2) / 2 = -2 + 7 * Complex.I := by
  sorry

end NUMINAMATH_GPT_midpoint_of_segment_l1254_125474


namespace NUMINAMATH_GPT_vartan_spent_on_recreation_last_week_l1254_125473

variable (W P : ℝ)
variable (h1 : P = 0.20)
variable (h2 : W > 0)

theorem vartan_spent_on_recreation_last_week :
  (P * W) = 0.20 * W :=
by
  sorry

end NUMINAMATH_GPT_vartan_spent_on_recreation_last_week_l1254_125473


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1254_125406

open BigOperators

theorem common_ratio_of_geometric_sequence
  (a1 : ℝ) (q : ℝ)
  (h1 : 2 * (a1 * q^5) = 3 * (a1 * (1 - q^4) / (1 - q)) + 1)
  (h2 : a1 * q^6 = 3 * (a1 * (1 - q^5) / (1 - q)) + 1)
  (h_pos : a1 > 0) :
  q = 3 :=
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1254_125406


namespace NUMINAMATH_GPT_no_integers_satisfy_eq_l1254_125483

theorem no_integers_satisfy_eq (m n : ℤ) : ¬ (m^3 = 4 * n + 2) := 
  sorry

end NUMINAMATH_GPT_no_integers_satisfy_eq_l1254_125483


namespace NUMINAMATH_GPT_integer_solutions_count_l1254_125410

theorem integer_solutions_count (x : ℤ) : 
  (x^2 - 3 * x + 2)^2 - 3 * (x^2 - 3 * x) - 4 = 0 ↔ 0 = 0 :=
by sorry

end NUMINAMATH_GPT_integer_solutions_count_l1254_125410


namespace NUMINAMATH_GPT_number_line_y_l1254_125424

theorem number_line_y (step_length : ℕ) (steps_total : ℕ) (total_distance : ℕ) (y_step : ℕ) (y : ℕ) 
    (H1 : steps_total = 6) 
    (H2 : total_distance = 24) 
    (H3 : y_step = 4)
    (H4 : step_length = total_distance / steps_total) 
    (H5 : y = step_length * y_step) : 
  y = 16 := 
  by 
    sorry

end NUMINAMATH_GPT_number_line_y_l1254_125424


namespace NUMINAMATH_GPT_two_planes_divide_at_most_4_parts_l1254_125499

-- Definitions related to the conditions
def Plane := ℝ × ℝ × ℝ → Prop -- Representing a plane in ℝ³ by an equation

-- Axiom: Two given planes
axiom plane1 : Plane
axiom plane2 : Plane

-- Conditions about their relationship
def are_parallel (p1 p2 : Plane) : Prop := 
  ∀ x y z, p1 (x, y, z) → p2 (x, y, z)

def intersect (p1 p2 : Plane) : Prop :=
  ∃ x y z, p1 (x, y, z) ∧ p2 (x, y, z)

-- Main theorem to state
theorem two_planes_divide_at_most_4_parts :
  (∃ p1 p2 : Plane, are_parallel p1 p2 ∨ intersect p1 p2) →
  (exists n : ℕ, n <= 4) :=
sorry

end NUMINAMATH_GPT_two_planes_divide_at_most_4_parts_l1254_125499


namespace NUMINAMATH_GPT_find_digit_e_l1254_125432

theorem find_digit_e (A B C D E F : ℕ) (h1 : A * 10 + B + (C * 10 + D) = A * 10 + E) (h2 : A * 10 + B - (D * 10 + C) = A * 10 + F) : E = 9 :=
sorry

end NUMINAMATH_GPT_find_digit_e_l1254_125432


namespace NUMINAMATH_GPT_divisible_by_6_l1254_125438

theorem divisible_by_6 (n : ℤ) (h1 : n % 3 = 0) (h2 : n % 2 = 0) : n % 6 = 0 :=
sorry

end NUMINAMATH_GPT_divisible_by_6_l1254_125438


namespace NUMINAMATH_GPT_broken_crayons_l1254_125439

theorem broken_crayons (total new used : Nat) (h1 : total = 14) (h2 : new = 2) (h3 : used = 4) :
  total = new + used + 8 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_broken_crayons_l1254_125439


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1254_125433

def A : Set ℝ := Set.Icc (-1) 1
def B : Set ℝ := Set.Icc (-2) 2
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + m * x - 1
def g (a m x : ℝ) : ℝ := 2 * abs (x - a) - x^2 - m * x

theorem problem1 (m : ℝ) : (∀ x, f m x ≤ 0 → x ∈ A) → m ∈ Set.Icc (-1) 1 :=
sorry

theorem problem2 (f_eq : ∀ x, f (-4) (1-x) = f (-4) (1+x)) : 
  Set.range (f (-4) ∘ id) ⊆ Set.Icc (-3) 15 :=
sorry

theorem problem3 (a : ℝ) (m : ℝ) :
  (a ≤ -1 → ∃ x, f m x + g a m x = -2*a - 2) ∧
  (-1 < a ∧ a < 1 → ∃ x, f m x + g a m x = a^2 - 1) ∧
  (a ≥ 1 → ∃ x, f m x + g a m x = 2*a - 2) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1254_125433


namespace NUMINAMATH_GPT_triangle_sides_inequality_l1254_125477

theorem triangle_sides_inequality
  {a b c : ℝ} (h₁ : a + b + c = 1) (h₂ : a > 0) (h₃ : b > 0) (h₄ : c > 0)
  (h₅ : a + b > c) (h₆ : a + c > b) (h₇ : b + c > a) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
by
  -- We would place the proof here if it were required
  sorry

end NUMINAMATH_GPT_triangle_sides_inequality_l1254_125477


namespace NUMINAMATH_GPT_ratio_bound_exceeds_2023_power_l1254_125441

theorem ratio_bound_exceeds_2023_power (a b : ℕ → ℝ) (h_pos : ∀ n, 0 < a n ∧ 0 < b n)
  (h1 : ∀ n, (a (n + 1)) * (b (n + 1)) = (a n)^2 + (b n)^2)
  (h2 : ∀ n, (a (n + 1)) + (b (n + 1)) = (a n) * (b n))
  (h3 : ∀ n, a n ≥ b n) :
  ∃ n, (a n) / (b n) > 2023^2023 :=
by
  sorry

end NUMINAMATH_GPT_ratio_bound_exceeds_2023_power_l1254_125441


namespace NUMINAMATH_GPT_same_root_a_eq_3_l1254_125444

theorem same_root_a_eq_3 {x a : ℝ} (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_same_root_a_eq_3_l1254_125444


namespace NUMINAMATH_GPT_printer_cost_l1254_125478

theorem printer_cost (num_keyboards : ℕ) (num_printers : ℕ) (total_cost : ℕ) (keyboard_cost : ℕ) (printer_cost : ℕ) :
  num_keyboards = 15 →
  num_printers = 25 →
  total_cost = 2050 →
  keyboard_cost = 20 →
  (total_cost - (num_keyboards * keyboard_cost)) / num_printers = printer_cost →
  printer_cost = 70 :=
by
  sorry

end NUMINAMATH_GPT_printer_cost_l1254_125478


namespace NUMINAMATH_GPT_midpoint_product_l1254_125431

theorem midpoint_product (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 4) (hy1 : y1 = -3) (hx2 : x2 = -8) (hy2 : y2 = 7) :
  let midx := (x1 + x2) / 2
  let midy := (y1 + y2) / 2
  midx * midy = -4 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_product_l1254_125431


namespace NUMINAMATH_GPT_smallest_number_divisible_by_618_3648_60_inc_l1254_125467

theorem smallest_number_divisible_by_618_3648_60_inc :
  ∃ N : ℕ, (N + 1) % 618 = 0 ∧ (N + 1) % 3648 = 0 ∧ (N + 1) % 60 = 0 ∧ N = 1038239 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_618_3648_60_inc_l1254_125467


namespace NUMINAMATH_GPT_y_coordinate_sum_of_circle_on_y_axis_l1254_125475

-- Define the properties of the circle
def center := (-3, 1)
def radius := 8

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + (y - 1) ^ 2 = 64

-- Define the Lean theorem statement
theorem y_coordinate_sum_of_circle_on_y_axis 
  (h₁ : center = (-3, 1)) 
  (h₂ : radius = 8) 
  (h₃ : ∀ y : ℝ, circle_eq 0 y → (∃ y1 y2 : ℝ, y = y1 ∨ y = y2) ) : 
  ∃ y1 y2 : ℝ, (y1 + y2 = 2) ∧ (circle_eq 0 y1) ∧ (circle_eq 0 y2) := 
by 
  sorry

end NUMINAMATH_GPT_y_coordinate_sum_of_circle_on_y_axis_l1254_125475


namespace NUMINAMATH_GPT_range_of_m_l1254_125420

theorem range_of_m (x y m : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 / x + 1 / y = 1) (h4 : x + 2 * y > m^2 + 2 * m) :
  -4 < m ∧ m < 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1254_125420


namespace NUMINAMATH_GPT_y_is_one_y_is_neg_two_thirds_l1254_125445

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (1, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove y = 1 given dot_product(vector_a, vector_b(y)) = 5
theorem y_is_one (h : dot_product vector_a (vector_b y) = 5) : y = 1 :=
by
  -- We assume the proof (otherwise it would go here)
  sorry

-- Prove y = -2/3 given |vector_a + vector_b(y)| = |vector_a - vector_b(y)|
theorem y_is_neg_two_thirds (h : (vector_a.1 + (vector_b y).1)^2 + (vector_a.2 + (vector_b y).2)^2 =
                                (vector_a.1 - (vector_b y).1)^2 + (vector_a.2 - (vector_b y).2)^2) : y = -2/3 :=
by
  -- We assume the proof (otherwise it would go here)
  sorry

end NUMINAMATH_GPT_y_is_one_y_is_neg_two_thirds_l1254_125445


namespace NUMINAMATH_GPT_largest_T_l1254_125463

theorem largest_T (T : ℝ) (a b c d e : ℝ) 
  (h1: a ≥ 0) (h2: b ≥ 0) (h3: c ≥ 0) (h4: d ≥ 0) (h5: e ≥ 0)
  (h_sum : a + b = c + d + e)
  (h_T : T ≤ (Real.sqrt 30) / (30 + 12 * Real.sqrt 6)) : 
  Real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2) ≥ T * (Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d + Real.sqrt e)^2 :=
sorry

end NUMINAMATH_GPT_largest_T_l1254_125463


namespace NUMINAMATH_GPT_total_pencils_sold_l1254_125466

theorem total_pencils_sold (price_reduced: Bool)
  (day1_students : ℕ) (first4_d1 : ℕ) (next3_d1 : ℕ) (last3_d1 : ℕ)
  (day2_students : ℕ) (first5_d2 : ℕ) (next6_d2 : ℕ) (last4_d2 : ℕ)
  (day3_students : ℕ) (first10_d3 : ℕ) (next10_d3 : ℕ) (last10_d3 : ℕ)
  (day1_total : day1_students = 10 ∧ first4_d1 = 4 ∧ next3_d1 = 3 ∧ last3_d1 = 3 ∧
    (first4_d1 * 5) + (next3_d1 * 7) + (last3_d1 * 3) = 50)
  (day2_total : day2_students = 15 ∧ first5_d2 = 5 ∧ next6_d2 = 6 ∧ last4_d2 = 4 ∧
    (first5_d2 * 4) + (next6_d2 * 9) + (last4_d2 * 6) = 98)
  (day3_total : day3_students = 2 * day2_students ∧ first10_d3 = 10 ∧ next10_d3 = 10 ∧ last10_d3 = 10 ∧
    (first10_d3 * 2) + (next10_d3 * 8) + (last10_d3 * 4) = 140) :
  (50 + 98 + 140 = 288) :=
sorry

end NUMINAMATH_GPT_total_pencils_sold_l1254_125466


namespace NUMINAMATH_GPT_cost_of_goat_l1254_125480

theorem cost_of_goat (G : ℝ) (goat_count : ℕ) (llama_count : ℕ) (llama_multiplier : ℝ) (total_cost : ℝ) 
    (h1 : goat_count = 3)
    (h2 : llama_count = 2 * goat_count)
    (h3 : llama_multiplier = 1.5)
    (h4 : total_cost = 4800) : G = 400 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_goat_l1254_125480


namespace NUMINAMATH_GPT_distance_bob_walked_when_met_l1254_125422

theorem distance_bob_walked_when_met (distance_XY walk_rate_Yolanda walk_rate_Bob : ℕ)
  (start_time_Yolanda start_time_Bob : ℕ) (y_distance b_distance : ℕ) (t : ℕ)
  (h1 : distance_XY = 65)
  (h2 : walk_rate_Yolanda = 5)
  (h3 : walk_rate_Bob = 7)
  (h4 : start_time_Yolanda = 0)
  (h5 : start_time_Bob = 1)
  (h6 : y_distance = walk_rate_Yolanda * (t + start_time_Bob))
  (h7 : b_distance = walk_rate_Bob * t)
  (h8 : y_distance + b_distance = distance_XY) : 
  b_distance = 35 := 
sorry

end NUMINAMATH_GPT_distance_bob_walked_when_met_l1254_125422


namespace NUMINAMATH_GPT_units_digit_product_composites_l1254_125423

theorem units_digit_product_composites :
  (4 * 6 * 8 * 9 * 10) % 10 = 0 :=
sorry

end NUMINAMATH_GPT_units_digit_product_composites_l1254_125423


namespace NUMINAMATH_GPT_smallest_difference_l1254_125485

-- Definition for the given problem conditions.
def side_lengths (AB BC AC : ℕ) : Prop := 
  AB + BC + AC = 2023 ∧ AB < BC ∧ BC ≤ AC ∧ 
  AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB

theorem smallest_difference (AB BC AC : ℕ) 
  (h: side_lengths AB BC AC) : 
  ∃ (AB BC AC : ℕ), side_lengths AB BC AC ∧ (BC - AB = 1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_difference_l1254_125485


namespace NUMINAMATH_GPT_probability_one_project_not_selected_l1254_125443

noncomputable def calc_probability : ℚ :=
  let n := 4 ^ 4
  let m := Nat.choose 4 2 * Nat.factorial 4
  let p := m / n
  p

theorem probability_one_project_not_selected :
  calc_probability = 9 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_project_not_selected_l1254_125443


namespace NUMINAMATH_GPT_min_value_proof_l1254_125489

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  2 / a + 2 / b + 2 / c

theorem min_value_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_abc : a + b + c = 9) : 
  minimum_value a b c ≥ 2 := 
by 
  sorry

end NUMINAMATH_GPT_min_value_proof_l1254_125489


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1254_125461

noncomputable def expression (a : ℝ) : ℝ :=
  ((a^2 - 1) / (a - 3) - a - 1) / ((a + 1) / (a^2 - 6 * a + 9))

theorem simplify_and_evaluate_expression : expression (3 - Real.sqrt 2) = -2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1254_125461


namespace NUMINAMATH_GPT_second_ball_probability_l1254_125400

-- Definitions and conditions
def red_balls := 3
def white_balls := 2
def black_balls := 5
def total_balls := red_balls + white_balls + black_balls

def first_ball_white_condition : Prop := (white_balls / total_balls) = (2 / 10)
def second_ball_red_given_first_white (first_ball_white : Prop) : Prop :=
  (first_ball_white → (red_balls / (total_balls - 1)) = (1 / 3))

-- Mathematical equivalence proof problem statement in Lean
theorem second_ball_probability : 
  first_ball_white_condition ∧ second_ball_red_given_first_white first_ball_white_condition :=
by
  sorry

end NUMINAMATH_GPT_second_ball_probability_l1254_125400


namespace NUMINAMATH_GPT_ram_birthday_l1254_125417

theorem ram_birthday
    (L : ℕ) (L1 : ℕ) (Llast : ℕ) (d : ℕ) (languages_learned_per_day : ℕ) (days_in_month : ℕ) :
    (L = 1000) →
    (L1 = 820) →
    (Llast = 1100) →
    (days_in_month = 28 ∨ days_in_month = 29 ∨ days_in_month = 30 ∨ days_in_month = 31) →
    (d = days_in_month - 1) →
    (languages_learned_per_day = (Llast - L1) / d) →
    ∃ n : ℕ, n = 19 :=
by
  intros hL hL1 hLlast hDays hm_d hLearned
  existsi 19
  sorry

end NUMINAMATH_GPT_ram_birthday_l1254_125417


namespace NUMINAMATH_GPT_floor_sqrt_12_squared_l1254_125426

theorem floor_sqrt_12_squared : (Int.floor (Real.sqrt 12))^2 = 9 := by
  sorry

end NUMINAMATH_GPT_floor_sqrt_12_squared_l1254_125426


namespace NUMINAMATH_GPT_min_length_QR_l1254_125487

theorem min_length_QR (PQ PR SR QS QR : ℕ) (hPQ : PQ = 7) (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
  QR > PR - PQ ∧ QR > QS - SR ↔ QR = 16 :=
by
  sorry

end NUMINAMATH_GPT_min_length_QR_l1254_125487


namespace NUMINAMATH_GPT_proposition_correctness_l1254_125436

theorem proposition_correctness :
  (∀ a b : ℝ, a < b ∧ b < 0 → ¬ (1 / a < 1 / b)) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 → (a + b) / 2 ≥ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≥ a * b / (a + b)) ∧
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) ∧
  (Real.log 9 * Real.log 11 < 1) ∧
  (∀ a b : ℝ, a > b ∧ 1 / a > 1 / b → a > 0 ∧ b < 0) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = 1 → ¬(x + 2 * y = 6)) :=
sorry

end NUMINAMATH_GPT_proposition_correctness_l1254_125436


namespace NUMINAMATH_GPT_gyeongyeon_total_path_l1254_125488

theorem gyeongyeon_total_path (D : ℝ) :
  (D / 4 + 250 = D / 2 - 300) -> D = 2200 :=
by
  intro h
  -- We would now proceed to show that D must equal 2200
  sorry

end NUMINAMATH_GPT_gyeongyeon_total_path_l1254_125488


namespace NUMINAMATH_GPT_weight_lifting_ratio_l1254_125450

theorem weight_lifting_ratio :
  ∀ (F S : ℕ), F + S = 600 ∧ F = 300 ∧ 2 * F = S + 300 → F / S = 1 :=
by
  intro F S
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_weight_lifting_ratio_l1254_125450


namespace NUMINAMATH_GPT_quadrilateral_area_is_two_l1254_125416

def A : (Int × Int) := (0, 0)
def B : (Int × Int) := (2, 0)
def C : (Int × Int) := (2, 3)
def D : (Int × Int) := (0, 2)

noncomputable def area (p1 p2 p3 p4 : (Int × Int)) : ℚ :=
  (1 / 2 : ℚ) * (abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p4.2 + p4.1 * p1.2) - 
                      (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p4.1 + p4.2 * p1.1)))

theorem quadrilateral_area_is_two : 
  area A B C D = 2 := by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_is_two_l1254_125416


namespace NUMINAMATH_GPT_range_of_m_l1254_125470

def A := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
def B (m : ℝ) := { x : ℝ | x^2 - (2 * m + 1) * x + 2 * m < 0 }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → (-1 / 2 ≤ m ∧ m ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1254_125470


namespace NUMINAMATH_GPT_original_sticker_price_l1254_125402

-- Define the conditions in Lean
variables {x : ℝ} -- x is the original sticker price of the laptop

-- Definitions based on the problem conditions
def store_A_price (x : ℝ) : ℝ := 0.80 * x - 50
def store_B_price (x : ℝ) : ℝ := 0.70 * x
def heather_saves (x : ℝ) : Prop := store_B_price x - store_A_price x = 30

-- The theorem to prove
theorem original_sticker_price (x : ℝ) (h : heather_saves x) : x = 200 :=
by
  sorry

end NUMINAMATH_GPT_original_sticker_price_l1254_125402


namespace NUMINAMATH_GPT_solve_olympics_problem_max_large_sets_l1254_125418

-- Definitions based on the conditions
variables (x y : ℝ)

-- Condition 1: 2 small sets cost $20 less than 1 large set
def condition1 : Prop := y - 2 * x = 20

-- Condition 2: 3 small sets and 2 large sets cost $390
def condition2 : Prop := 3 * x + 2 * y = 390

-- Finding unit prices
def unit_prices : Prop := x = 50 ∧ y = 120

-- Condition 3: Budget constraint for purchasing sets
def budget_constraint (m : ℕ) : Prop := m ≤ 7

-- Prove unit prices and purchasing constraints
theorem solve_olympics_problem :
  condition1 x y ∧ condition2 x y → unit_prices x y :=
by
  sorry

theorem max_large_sets :
  budget_constraint 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_olympics_problem_max_large_sets_l1254_125418


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t_2_l1254_125414

def y (t : ℝ) : ℝ := 3 * t^2 + 4

theorem instantaneous_velocity_at_t_2 :
  deriv y 2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t_2_l1254_125414


namespace NUMINAMATH_GPT_savanna_total_animals_l1254_125476

def num_lions_safari := 100
def num_snakes_safari := num_lions_safari / 2
def num_giraffes_safari := num_snakes_safari - 10
def num_elephants_safari := num_lions_safari / 4

def num_lions_savanna := num_lions_safari * 2
def num_snakes_savanna := num_snakes_safari * 3
def num_giraffes_savanna := num_giraffes_safari + 20
def num_elephants_savanna := num_elephants_safari * 5
def num_zebras_savanna := (num_lions_savanna + num_snakes_savanna) / 2

def total_animals_savanna := 
  num_lions_savanna 
  + num_snakes_savanna 
  + num_giraffes_savanna 
  + num_elephants_savanna 
  + num_zebras_savanna

open Nat
theorem savanna_total_animals : total_animals_savanna = 710 := by
  sorry

end NUMINAMATH_GPT_savanna_total_animals_l1254_125476


namespace NUMINAMATH_GPT_find_a_l1254_125484

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a {x0 a : ℝ} (h : f x0 a - g x0 a = 3) : a = -Real.log 2 - 1 :=
sorry

end NUMINAMATH_GPT_find_a_l1254_125484


namespace NUMINAMATH_GPT_probability_of_selecting_3_co_captains_is_correct_l1254_125491

def teams : List ℕ := [4, 6, 7, 9]

def probability_of_selecting_3_co_captains (n : ℕ) : ℚ :=
  if n = 4 then 1/4
  else if n = 6 then 1/20
  else if n = 7 then 1/35
  else if n = 9 then 1/84
  else 0

def total_probability : ℚ :=
  (1/4) * (probability_of_selecting_3_co_captains 4 +
            probability_of_selecting_3_co_captains 6 +
            probability_of_selecting_3_co_captains 7 +
            probability_of_selecting_3_co_captains 9)

theorem probability_of_selecting_3_co_captains_is_correct :
  total_probability = 143 / 1680 :=
by
  -- The proof will be inserted here
  sorry

end NUMINAMATH_GPT_probability_of_selecting_3_co_captains_is_correct_l1254_125491


namespace NUMINAMATH_GPT_painted_cubes_count_l1254_125471

/-- A theorem to prove the number of painted small cubes in a larger cube. -/
theorem painted_cubes_count (total_cubes unpainted_cubes : ℕ) (a b : ℕ) :
  total_cubes = a * a * a →
  unpainted_cubes = (a - 2) * (a - 2) * (a - 2) →
  22 = unpainted_cubes →
  64 = total_cubes →
  ∃ m, m = total_cubes - unpainted_cubes ∧ m = 42 :=
by
  sorry

end NUMINAMATH_GPT_painted_cubes_count_l1254_125471


namespace NUMINAMATH_GPT_alexandra_magazines_l1254_125425

theorem alexandra_magazines :
  let friday_magazines := 8
  let saturday_magazines := 12
  let sunday_magazines := 4 * friday_magazines
  let dog_chewed_magazines := 4
  let total_magazines_before_dog := friday_magazines + saturday_magazines + sunday_magazines
  let total_magazines_now := total_magazines_before_dog - dog_chewed_magazines
  total_magazines_now = 48 := by
  sorry

end NUMINAMATH_GPT_alexandra_magazines_l1254_125425


namespace NUMINAMATH_GPT_solve_equation_l1254_125421

theorem solve_equation (x : ℝ) : 
  (9 - x - 2 * (31 - x) = 27) → (x = 80) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1254_125421


namespace NUMINAMATH_GPT_find_divisor_l1254_125440

def div_remainder (a b r : ℕ) : Prop :=
  ∃ k : ℕ, a = k * b + r

theorem find_divisor :
  ∃ D : ℕ, (div_remainder 242 D 15) ∧ (div_remainder 698 D 27) ∧ (div_remainder (242 + 698) D 5) ∧ D = 37 := 
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1254_125440


namespace NUMINAMATH_GPT_annual_profits_l1254_125481

-- Define the profits of each quarter
def P1 : ℕ := 1500
def P2 : ℕ := 1500
def P3 : ℕ := 3000
def P4 : ℕ := 2000

-- State the annual profit theorem
theorem annual_profits : P1 + P2 + P3 + P4 = 8000 := by
  sorry

end NUMINAMATH_GPT_annual_profits_l1254_125481


namespace NUMINAMATH_GPT_find_k_value_l1254_125490

theorem find_k_value
  (k : ℤ)
  (h : 3 * 2^2001 - 3 * 2^2000 - 2^1999 + 2^1998 = k * 2^1998) : k = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_k_value_l1254_125490


namespace NUMINAMATH_GPT_solve_diamond_l1254_125407

theorem solve_diamond (d : ℕ) (h : d * 6 + 5 = d * 7 + 2) : d = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_diamond_l1254_125407


namespace NUMINAMATH_GPT_probability_four_collinear_dots_l1254_125454

noncomputable def probability_collinear_four_dots : ℚ :=
  let total_dots := 25
  let choose_4 := (total_dots.choose 4)
  let successful_outcomes := 60
  successful_outcomes / choose_4

theorem probability_four_collinear_dots :
  probability_collinear_four_dots = 12 / 2530 :=
by
  sorry

end NUMINAMATH_GPT_probability_four_collinear_dots_l1254_125454


namespace NUMINAMATH_GPT_problem1_problem2_l1254_125465

-- Definitions for permutation and combination
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problems statements
theorem problem1 : 
  (2 * A 8 5 + 7 * A 8 4) / (A 8 8 - A 9 5) = 1 / 15 := by 
  sorry

theorem problem2 :
  C 200 198 + C 200 196 + 2 * C 200 197 = C 202 4 := by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1254_125465


namespace NUMINAMATH_GPT_SandySpentTotal_l1254_125411

theorem SandySpentTotal :
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 := by
  sorry

end NUMINAMATH_GPT_SandySpentTotal_l1254_125411


namespace NUMINAMATH_GPT_arithmetic_series_sum_l1254_125453

theorem arithmetic_series_sum :
  let first_term := -25
  let common_difference := 2
  let last_term := 19
  let n := (last_term - first_term) / common_difference + 1
  let sum := n * (first_term + last_term) / 2
  sum = -69 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_l1254_125453


namespace NUMINAMATH_GPT_toilet_paper_duration_l1254_125452

theorem toilet_paper_duration :
  let bill_weekday := 3 * 5
  let wife_weekday := 4 * 8
  let kid_weekday := 5 * 6
  let total_weekday := bill_weekday + wife_weekday + 2 * kid_weekday
  let bill_weekend := 4 * 6
  let wife_weekend := 5 * 10
  let kid_weekend := 6 * 5
  let total_weekend := bill_weekend + wife_weekend + 2 * kid_weekend
  let total_week := 5 * total_weekday + 2 * total_weekend
  let total_squares := 1000 * 300
  let weeks_last := total_squares / total_week
  let days_last := weeks_last * 7
  days_last = 2615 :=
sorry

end NUMINAMATH_GPT_toilet_paper_duration_l1254_125452


namespace NUMINAMATH_GPT_nhai_highway_construction_l1254_125415

/-- Problem definition -/
def total_man_hours (men1 men2 days1 days2 hours1 hours2 : Nat) : Nat := 
  (men1 * days1 * hours1) + (men2 * days2 * hours2)

theorem nhai_highway_construction :
  let men := 100
  let days1 := 25
  let days2 := 25
  let hours1 := 8
  let hours2 := 10
  let additional_men := 60
  let total_days := 50
  total_man_hours men (men + additional_men) total_days total_days hours1 hours2 = 
  2 * total_man_hours men men days1 days2 hours1 hours1 :=
  sorry

end NUMINAMATH_GPT_nhai_highway_construction_l1254_125415


namespace NUMINAMATH_GPT_fraction_meaningful_range_l1254_125449

theorem fraction_meaningful_range (x : ℝ) : 5 - x ≠ 0 ↔ x ≠ 5 :=
by sorry

end NUMINAMATH_GPT_fraction_meaningful_range_l1254_125449


namespace NUMINAMATH_GPT_intersection_M_N_l1254_125409

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : (M ∩ N) = {x | 0 ≤ x ∧ x < 1} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_M_N_l1254_125409


namespace NUMINAMATH_GPT_Dan_work_hours_l1254_125408

theorem Dan_work_hours (x : ℝ) :
  (1 / 15) * x + 3 / 5 = 1 → x = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Dan_work_hours_l1254_125408


namespace NUMINAMATH_GPT_matrix_sum_correct_l1254_125464

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![4, -3],
  ![2, 5]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-6, 8],
  ![-3, 7]
]

def C : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-2, 5],
  ![-1, 12]
]

theorem matrix_sum_correct : A + B = C := by
  sorry

end NUMINAMATH_GPT_matrix_sum_correct_l1254_125464


namespace NUMINAMATH_GPT_probability_of_odd_sum_l1254_125442

open Nat

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_odd_sum :
  (binomial 11 3) / (binomial 12 4) = 1 / 3 := by
sorry

end NUMINAMATH_GPT_probability_of_odd_sum_l1254_125442


namespace NUMINAMATH_GPT_pow_product_l1254_125451

theorem pow_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := 
by {
  sorry
}

end NUMINAMATH_GPT_pow_product_l1254_125451


namespace NUMINAMATH_GPT_min_value_of_expression_l1254_125492

theorem min_value_of_expression (x y : ℝ) (h : x^2 + x * y + y^2 = 3) : x^2 - x * y + y^2 ≥ 1 :=
by 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1254_125492


namespace NUMINAMATH_GPT_square_side_length_l1254_125447

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end NUMINAMATH_GPT_square_side_length_l1254_125447


namespace NUMINAMATH_GPT_find_k_l1254_125427

def g (a b c x : ℤ) := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (h1 : g a b c 1 = 0)
  (h2 : 20 < g a b c 5 ∧ g a b c 5 < 30)
  (h3 : 40 < g a b c 6 ∧ g a b c 6 < 50)
  (h4 : ∃ k : ℤ, 3000 * k < g a b c 100 ∧ g a b c 100 < 3000 * (k + 1)) :
  ∃ k : ℤ, k = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1254_125427


namespace NUMINAMATH_GPT_cost_of_one_hockey_stick_l1254_125413

theorem cost_of_one_hockey_stick (x : ℝ)
    (h1 : x * 2 + 25 = 68) : x = 21.50 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_hockey_stick_l1254_125413


namespace NUMINAMATH_GPT_total_people_l1254_125497

theorem total_people (M W C : ℕ) (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : C = 30) : M + W + C = 300 :=
by
  sorry

end NUMINAMATH_GPT_total_people_l1254_125497


namespace NUMINAMATH_GPT_inequality_solution_l1254_125469

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (2^x + 1)

lemma monotone_decreasing (a : ℝ) : ∀ x y : ℝ, x < y → f a y < f a x := 
sorry

lemma odd_function (a : ℝ) (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 0 := 
sorry

theorem inequality_solution (t : ℝ) (a : ℝ) (h_monotone : ∀ x y : ℝ, x < y → f a y < f a x)
    (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : t ≥ 4 / 3 ↔ f a (2 * t + 1) + f a (t - 5) ≤ 0 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1254_125469


namespace NUMINAMATH_GPT_sin_cos_equation_solution_l1254_125404

open Real

theorem sin_cos_equation_solution (x : ℝ): 
  (∃ n : ℤ, x = (π / 4050) + (π * n / 2025)) ∨ (∃ k : ℤ, x = (π * k / 9)) ↔ 
  sin (2025 * x) ^ 4 + (cos (2016 * x) ^ 2019) * (cos (2025 * x) ^ 2018) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_sin_cos_equation_solution_l1254_125404


namespace NUMINAMATH_GPT_part_a_part_b_l1254_125419

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def valid_permutation (P : Fin 16 → ℕ) : Prop :=
  (∀ i : Fin 15, is_perfect_square (P i + P (i + 1))) ∧
  ∀ i, P i ∈ (Finset.range 16).image (λ x => x + 1)

def valid_cyclic_permutation (C : Fin 16 → ℕ) : Prop :=
  (∀ i : Fin 15, is_perfect_square (C i + C (i + 1))) ∧
  is_perfect_square (C 15 + C 0) ∧
  ∀ i, C i ∈ (Finset.range 16).image (λ x => x + 1)

theorem part_a :
  ∃ P : Fin 16 → ℕ, valid_permutation P := sorry

theorem part_b :
  ¬ ∃ C : Fin 16 → ℕ, valid_cyclic_permutation C := sorry

end NUMINAMATH_GPT_part_a_part_b_l1254_125419


namespace NUMINAMATH_GPT_eq_x_minus_y_l1254_125498

theorem eq_x_minus_y (x y : ℝ) : (x - y) * (x - y) = x^2 - 2 * x * y + y^2 :=
by
  sorry

end NUMINAMATH_GPT_eq_x_minus_y_l1254_125498


namespace NUMINAMATH_GPT_relation_among_a_b_c_l1254_125479

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 3
noncomputable def c : ℝ := Real.sqrt 6 - Real.sqrt 2

theorem relation_among_a_b_c : a > c ∧ c > b :=
by {
  sorry
}

end NUMINAMATH_GPT_relation_among_a_b_c_l1254_125479


namespace NUMINAMATH_GPT_color_of_241st_marble_l1254_125455

def sequence_color (n : ℕ) : String :=
  if n % 14 < 6 then "blue"
  else if n % 14 < 11 then "red"
  else "green"

theorem color_of_241st_marble : sequence_color 240 = "blue" :=
  by
  sorry

end NUMINAMATH_GPT_color_of_241st_marble_l1254_125455


namespace NUMINAMATH_GPT_midpoint_coordinates_l1254_125472

theorem midpoint_coordinates :
  let A := (7, 8)
  let B := (1, 2)
  let midpoint (p1 p2 : ℕ × ℕ) : ℕ × ℕ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint A B = (4, 5) :=
by
  sorry

end NUMINAMATH_GPT_midpoint_coordinates_l1254_125472


namespace NUMINAMATH_GPT_range_of_m_for_inversely_proportional_function_l1254_125493

theorem range_of_m_for_inversely_proportional_function 
  (m : ℝ)
  (h : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > x₁ → (m - 1) / x₂ < (m - 1) / x₁) : 
  m > 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_for_inversely_proportional_function_l1254_125493


namespace NUMINAMATH_GPT_candidate_a_valid_votes_l1254_125430

/-- In an election, candidate A got 80% of the total valid votes.
If 15% of the total votes were declared invalid and the total number of votes is 560,000,
find the number of valid votes polled in favor of candidate A. -/
theorem candidate_a_valid_votes :
  let total_votes := 560000
  let invalid_percentage := 0.15
  let valid_percentage := 0.85
  let candidate_a_percentage := 0.80
  let valid_votes := (valid_percentage * total_votes : ℝ)
  let candidate_a_votes := (candidate_a_percentage * valid_votes : ℝ)
  candidate_a_votes = 380800 :=
by
  sorry

end NUMINAMATH_GPT_candidate_a_valid_votes_l1254_125430
