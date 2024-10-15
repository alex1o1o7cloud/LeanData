import Mathlib

namespace NUMINAMATH_GPT_point_outside_circle_l1589_158944

theorem point_outside_circle {a b : ℝ} (h : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ a * x + b * y = 1) : a^2 + b^2 > 1 :=
by sorry

end NUMINAMATH_GPT_point_outside_circle_l1589_158944


namespace NUMINAMATH_GPT_mean_goals_is_correct_l1589_158922

theorem mean_goals_is_correct :
  let goals5 := 5
  let players5 := 4
  let goals6 := 6
  let players6 := 3
  let goals7 := 7
  let players7 := 2
  let goals8 := 8
  let players8 := 1
  let total_goals := goals5 * players5 + goals6 * players6 + goals7 * players7 + goals8 * players8
  let total_players := players5 + players6 + players7 + players8
  (total_goals / total_players : ℝ) = 6 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_mean_goals_is_correct_l1589_158922


namespace NUMINAMATH_GPT_sqrt_expression_result_l1589_158933

theorem sqrt_expression_result :
  (Real.sqrt (16 - 8 * Real.sqrt 3) - Real.sqrt (16 + 8 * Real.sqrt 3)) ^ 2 = 48 := 
sorry

end NUMINAMATH_GPT_sqrt_expression_result_l1589_158933


namespace NUMINAMATH_GPT_initial_books_correct_l1589_158981

def sold_books : ℕ := 78
def left_books : ℕ := 37
def initial_books : ℕ := sold_books + left_books

theorem initial_books_correct : initial_books = 115 := by
  sorry

end NUMINAMATH_GPT_initial_books_correct_l1589_158981


namespace NUMINAMATH_GPT_circle_equation_l1589_158936

/-- Given that point C is above the x-axis and
    the circle C with center C is tangent to the x-axis at point A(1,0) and
    intersects with circle O: x² + y² = 4 at points P and Q such that
    the length of PQ is sqrt(14)/2, the standard equation of circle C
    is (x - 1)² + (y - 1)² = 1. -/
theorem circle_equation {C : ℝ × ℝ} (hC : C.2 > 0) (tangent_at_A : C = (1, C.2))
  (intersect_with_O : ∃ P Q : ℝ × ℝ, (P ≠ Q) ∧ (P.1 ^ 2 + P.2 ^ 2 = 4) ∧ 
  (Q.1 ^ 2 + Q.2 ^ 2 = 4) ∧ ((P.1 - 1)^2 + (P.2 - C.2)^2 = C.2^2) ∧ 
  ((Q.1 - 1)^2 + (Q.2 - C.2)^2 = C.2^2) ∧ ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 14/4)) :
  (C.2 = 1) ∧ ((x - 1)^2 + (y - 1)^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l1589_158936


namespace NUMINAMATH_GPT_pages_copied_l1589_158953

theorem pages_copied (cost_per_page : ℕ) (amount_in_dollars : ℕ)
    (cents_per_dollar : ℕ) (total_cents : ℕ) 
    (pages : ℕ)
    (h1 : cost_per_page = 3)
    (h2 : amount_in_dollars = 25)
    (h3 : cents_per_dollar = 100)
    (h4 : total_cents = amount_in_dollars * cents_per_dollar)
    (h5 : total_cents = 2500)
    (h6 : pages = total_cents / cost_per_page) :
  pages = 833 := 
sorry

end NUMINAMATH_GPT_pages_copied_l1589_158953


namespace NUMINAMATH_GPT_fish_tank_ratio_l1589_158968

theorem fish_tank_ratio :
  ∀ (F1 F2 F3: ℕ),
  F1 = 15 →
  F3 = 10 →
  (F3 = (1 / 3 * F2)) →
  F2 / F1 = 2 :=
by
  intros F1 F2 F3 hF1 hF3 hF2
  sorry

end NUMINAMATH_GPT_fish_tank_ratio_l1589_158968


namespace NUMINAMATH_GPT_positive_integers_N_segment_condition_l1589_158913

theorem positive_integers_N_segment_condition (N : ℕ) (x : ℕ) (n : ℕ)
  (h1 : 10 ≤ N ∧ N ≤ 10^20)
  (h2 : N = x * (10^n - 1) / 9) (h3 : 1 ≤ n ∧ n ≤ 20) : 
  N + 1 = (x + 1) * (9 + 1)^n ∧ x < 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_positive_integers_N_segment_condition_l1589_158913


namespace NUMINAMATH_GPT_quadratic_roots_l1589_158988

theorem quadratic_roots (m n p : ℕ) (h : m.gcd p = 1) 
  (h1 : 3 * m^2 - 8 * m * p + p^2 = p^2 * n) : n = 13 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_l1589_158988


namespace NUMINAMATH_GPT_price_of_expensive_feed_l1589_158969

theorem price_of_expensive_feed
  (total_weight : ℝ)
  (mix_price_per_pound : ℝ)
  (cheaper_feed_weight : ℝ)
  (cheaper_feed_price_per_pound : ℝ)
  (expensive_feed_price_per_pound : ℝ) :
  total_weight = 27 →
  mix_price_per_pound = 0.26 →
  cheaper_feed_weight = 14.2105263158 →
  cheaper_feed_price_per_pound = 0.17 →
  expensive_feed_price_per_pound = 0.36 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_price_of_expensive_feed_l1589_158969


namespace NUMINAMATH_GPT_subtraction_of_decimals_l1589_158901

theorem subtraction_of_decimals : 58.3 - 0.45 = 57.85 := by
  sorry

end NUMINAMATH_GPT_subtraction_of_decimals_l1589_158901


namespace NUMINAMATH_GPT_max_value_xy_l1589_158973

open Real

theorem max_value_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + 5 * y < 100) :
  ∃ (c : ℝ), c = 3703.7 ∧ ∀ (x' y' : ℝ), 0 < x' → 0 < y' → 2 * x' + 5 * y' < 100 → x' * y' * (100 - 2 * x' - 5 * y') ≤ c :=
sorry

end NUMINAMATH_GPT_max_value_xy_l1589_158973


namespace NUMINAMATH_GPT_total_parts_in_order_l1589_158931

theorem total_parts_in_order (total_cost : ℕ) (cost_20 : ℕ) (cost_50 : ℕ) (num_50_dollar_parts : ℕ) (num_20_dollar_parts : ℕ) :
  total_cost = 2380 → cost_20 = 20 → cost_50 = 50 → num_50_dollar_parts = 40 → (total_cost = num_50_dollar_parts * cost_50 + num_20_dollar_parts * cost_20) → (num_50_dollar_parts + num_20_dollar_parts = 59) :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end NUMINAMATH_GPT_total_parts_in_order_l1589_158931


namespace NUMINAMATH_GPT_max_sub_min_value_l1589_158974

variable {x y : ℝ}

noncomputable def expression (x y : ℝ) : ℝ :=
  (abs (x + y))^2 / ((abs x)^2 + (abs y)^2)

theorem max_sub_min_value :
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 
  (expression x y ≤ 2 ∧ 0 ≤ expression x y) → 
  (∃ m M, m = 0 ∧ M = 2 ∧ M - m = 2) :=
by
  sorry

end NUMINAMATH_GPT_max_sub_min_value_l1589_158974


namespace NUMINAMATH_GPT_average_donation_l1589_158947

theorem average_donation (d : ℕ) (n : ℕ) (r : ℕ) (average_donation : ℕ) 
  (h1 : d = 10)   -- $10 donated by customers
  (h2 : r = 2)    -- $2 donated by restaurant
  (h3 : n = 40)   -- number of customers
  (h4 : (r : ℕ) * n / d = 24) -- total donation by restaurant is $24
  : average_donation = 3 := 
by
  sorry

end NUMINAMATH_GPT_average_donation_l1589_158947


namespace NUMINAMATH_GPT_sam_paint_cans_l1589_158937

theorem sam_paint_cans : 
  ∀ (cans_per_room : ℝ) (initial_cans remaining_cans : ℕ),
    initial_cans * cans_per_room = 40 ∧
    remaining_cans * cans_per_room = 30 ∧
    initial_cans - remaining_cans = 4 →
    remaining_cans = 12 :=
by sorry

end NUMINAMATH_GPT_sam_paint_cans_l1589_158937


namespace NUMINAMATH_GPT_find_second_term_geometric_sequence_l1589_158954

noncomputable def second_term_geometric_sequence (a r : ℝ) : ℝ :=
  a * r

theorem find_second_term_geometric_sequence:
  ∀ (a r : ℝ),
    a * r^2 = 12 →
    a * r^3 = 18 →
    second_term_geometric_sequence a r = 8 :=
by
  intros a r h1 h2
  sorry

end NUMINAMATH_GPT_find_second_term_geometric_sequence_l1589_158954


namespace NUMINAMATH_GPT_counted_integer_twice_l1589_158966

theorem counted_integer_twice (x n : ℕ) (hn : n = 100) 
  (h_sum : (n * (n + 1)) / 2 + x = 5053) : x = 3 := by
  sorry

end NUMINAMATH_GPT_counted_integer_twice_l1589_158966


namespace NUMINAMATH_GPT_geometric_sequence_a3_l1589_158928

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a3 
  (a : ℕ → ℝ) (h1 : a 1 = -2) (h5 : a 5 = -8)
  (h : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) : 
  a 3 = -4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a3_l1589_158928


namespace NUMINAMATH_GPT_find_other_number_l1589_158965

theorem find_other_number (a b : ℕ) (h_lcm: Nat.lcm a b = 2310) (h_hcf: Nat.gcd a b = 55) (h_a: a = 210) : b = 605 := by
  sorry

end NUMINAMATH_GPT_find_other_number_l1589_158965


namespace NUMINAMATH_GPT_john_spent_l1589_158904

/-- John bought 9.25 meters of cloth at a cost price of $44 per meter.
    Prove that the total amount John spent on the cloth is $407. -/
theorem john_spent :
  let length_of_cloth := 9.25
  let cost_per_meter := 44
  let total_cost := length_of_cloth * cost_per_meter
  total_cost = 407 := by
  sorry

end NUMINAMATH_GPT_john_spent_l1589_158904


namespace NUMINAMATH_GPT_average_of_P_and_R_l1589_158950

theorem average_of_P_and_R (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (Q + R) / 2 = 5250)
  (h3 : P = 3000)
  : (P + R) / 2 = 6200 := by
  sorry

end NUMINAMATH_GPT_average_of_P_and_R_l1589_158950


namespace NUMINAMATH_GPT_inequality_range_of_a_l1589_158970

theorem inequality_range_of_a (a : ℝ) :
  (∀ x y : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (1 ≤ y ∧ y ≤ 3) → 2 * x^2 - a * x * y + y^2 ≥ 0) →
  a ≤ 2 * Real.sqrt 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_inequality_range_of_a_l1589_158970


namespace NUMINAMATH_GPT_ratio_of_men_to_women_l1589_158939

-- Define constants
def total_people : ℕ := 60
def men_in_meeting : ℕ := 4
def women_in_meeting : ℕ := 6
def women_reduction_percentage : ℕ := 20

-- Statement of the problem
theorem ratio_of_men_to_women (total_people men_in_meeting women_in_meeting women_reduction_percentage: ℕ)
  (total_people_eq : total_people = 60)
  (men_in_meeting_eq : men_in_meeting = 4)
  (women_in_meeting_eq : women_in_meeting = 6)
  (women_reduction_percentage_eq : women_reduction_percentage = 20) :
  (men_in_meeting + ((total_people - men_in_meeting - women_in_meeting) * women_reduction_percentage / 100)) 
  = total_people / 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_men_to_women_l1589_158939


namespace NUMINAMATH_GPT_nancy_tortilla_chips_l1589_158998

theorem nancy_tortilla_chips :
  ∀ (total_chips chips_brother chips_herself chips_sister : ℕ),
    total_chips = 22 →
    chips_brother = 7 →
    chips_herself = 10 →
    chips_sister = total_chips - chips_brother - chips_herself →
    chips_sister = 5 :=
by
  intros total_chips chips_brother chips_herself chips_sister
  intro h_total h_brother h_herself h_sister
  rw [h_total, h_brother, h_herself] at h_sister
  simp at h_sister
  assumption

end NUMINAMATH_GPT_nancy_tortilla_chips_l1589_158998


namespace NUMINAMATH_GPT_sqrt_approximation_l1589_158915

theorem sqrt_approximation :
  (2^2 < 5) ∧ (5 < 3^2) ∧ 
  (2.2^2 < 5) ∧ (5 < 2.3^2) ∧ 
  (2.23^2 < 5) ∧ (5 < 2.24^2) ∧ 
  (2.236^2 < 5) ∧ (5 < 2.237^2) →
  (Float.ceil (Float.sqrt 5 * 100) / 100) = 2.24 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_sqrt_approximation_l1589_158915


namespace NUMINAMATH_GPT_length_BD_l1589_158971

noncomputable def length_segments (CB : ℝ) : ℝ := 4 * CB

noncomputable def circle_radius_AC (CB : ℝ) : ℝ := (4 * CB) / 2

noncomputable def circle_radius_CB (CB : ℝ) : ℝ := CB / 2

noncomputable def tangent_touch_point (CB BD : ℝ) : Prop :=
  ∃ x, CB = x ∧ BD = x

theorem length_BD (CB BD : ℝ) (h : tangent_touch_point CB BD) : BD = CB :=
by
  sorry

end NUMINAMATH_GPT_length_BD_l1589_158971


namespace NUMINAMATH_GPT_inequality_proof_l1589_158934

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y + y * z + z * x = 1) :
  3 - Real.sqrt 3 + (x^2 / y) + (y^2 / z) + (z^2 / x) ≥ (x + y + z)^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1589_158934


namespace NUMINAMATH_GPT_quadrangular_prism_volume_l1589_158916

theorem quadrangular_prism_volume
  (perimeter : ℝ)
  (side_length : ℝ)
  (height : ℝ)
  (volume : ℝ)
  (H1 : perimeter = 32)
  (H2 : side_length = perimeter / 4)
  (H3 : height = side_length)
  (H4 : volume = side_length * side_length * height) :
  volume = 512 := by
    sorry

end NUMINAMATH_GPT_quadrangular_prism_volume_l1589_158916


namespace NUMINAMATH_GPT_op_7_3_eq_70_l1589_158923

noncomputable def op (x y : ℝ) : ℝ := sorry

axiom ax1 : ∀ x : ℝ, op x 0 = x
axiom ax2 : ∀ x y : ℝ, op x y = op y x
axiom ax3 : ∀ x y : ℝ, op (x + 1) y = (op x y) + y + 2

theorem op_7_3_eq_70 : op 7 3 = 70 := by
  sorry

end NUMINAMATH_GPT_op_7_3_eq_70_l1589_158923


namespace NUMINAMATH_GPT_calc_root_difference_l1589_158993

theorem calc_root_difference :
  ((81: ℝ)^(1/4) + (32: ℝ)^(1/5) - (49: ℝ)^(1/2)) = -2 :=
by
  have h1 : (81: ℝ)^(1/4) = 3 := by sorry
  have h2 : (32: ℝ)^(1/5) = 2 := by sorry
  have h3 : (49: ℝ)^(1/2) = 7 := by sorry
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_calc_root_difference_l1589_158993


namespace NUMINAMATH_GPT_paul_money_last_weeks_l1589_158985

theorem paul_money_last_weeks (a b c: ℕ) (h1: a = 68) (h2: b = 13) (h3: c = 9) : 
  (a + b) / c = 9 := 
by 
  sorry

end NUMINAMATH_GPT_paul_money_last_weeks_l1589_158985


namespace NUMINAMATH_GPT_cleaning_time_if_anne_doubled_l1589_158996

-- Definitions based on conditions
def anne_rate := 1 / 12
def combined_rate := 1 / 4
def bruce_rate := combined_rate - anne_rate
def double_anne_rate := 2 * anne_rate
def doubled_combined_rate := bruce_rate + double_anne_rate

-- Statement of the problem
theorem cleaning_time_if_anne_doubled :  1 / doubled_combined_rate = 3 :=
by sorry

end NUMINAMATH_GPT_cleaning_time_if_anne_doubled_l1589_158996


namespace NUMINAMATH_GPT_smallest_solution_for_quartic_eq_l1589_158906

theorem smallest_solution_for_quartic_eq :
  let f (x : ℝ) := x^4 - 40*x^2 + 144
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y :=
sorry

end NUMINAMATH_GPT_smallest_solution_for_quartic_eq_l1589_158906


namespace NUMINAMATH_GPT_problem_eval_expression_l1589_158991

theorem problem_eval_expression :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by
  sorry

end NUMINAMATH_GPT_problem_eval_expression_l1589_158991


namespace NUMINAMATH_GPT_find_matrix_N_l1589_158972

open Matrix

variable (u : Fin 3 → ℝ)

def cross_product (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![a 1 * b 2 - a 2 * b 1, a 2 * b 0 - a 0 * b 2, a 0 * b 1 - a 1 * b 0]

-- Define vector v as the fixed vector in the problem
def v : Fin 3 → ℝ := ![7, 3, -9]

-- Define matrix N as the matrix to be found
def N : Matrix (Fin 3) (Fin 3) ℝ := ![![0, 9, 3], ![-9, 0, -7], ![-3, 7, 0]]

-- Define the requirement condition
theorem find_matrix_N :
  ∀ (u : Fin 3 → ℝ), (N.mulVec u) = cross_product v u :=
by
  sorry

end NUMINAMATH_GPT_find_matrix_N_l1589_158972


namespace NUMINAMATH_GPT_negation_of_exists_leq_zero_l1589_158932

theorem negation_of_exists_leq_zero (x : ℝ) : ¬(∃ x ≥ 1, 2^x ≤ 0) ↔ ∀ x ≥ 1, 2^x > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_leq_zero_l1589_158932


namespace NUMINAMATH_GPT_range_of_sine_l1589_158999

theorem range_of_sine {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.sin x ≥ Real.sqrt 2 / 2) :
  Real.pi / 4 ≤ x ∧ x ≤ 3 * Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_sine_l1589_158999


namespace NUMINAMATH_GPT_blue_parrots_count_l1589_158943

theorem blue_parrots_count (P : ℕ) (red green blue : ℕ) (h₁ : red = P / 2) (h₂ : green = P / 4) (h₃ : blue = P - red - green) (h₄ :  P + 30 = 150) : blue = 38 :=
by {
-- We will write the proof here
sorry
}

end NUMINAMATH_GPT_blue_parrots_count_l1589_158943


namespace NUMINAMATH_GPT_total_cost_is_correct_l1589_158914

noncomputable def total_cost_of_gifts : ℝ :=
  let polo_shirts := 3 * 26
  let necklaces := 2 * 83
  let computer_game := 90
  let socks := 4 * 7
  let books := 3 * 15
  let scarves := 2 * 22
  let mugs := 5 * 8
  let sneakers := 65

  let cost_before_discounts := polo_shirts + necklaces + computer_game + socks + books + scarves + mugs + sneakers

  let discount_books := 0.10 * books
  let discount_sneakers := 0.15 * sneakers
  let cost_after_discounts := cost_before_discounts - discount_books - discount_sneakers

  let sales_tax := 0.065 * cost_after_discounts
  let cost_after_tax := cost_after_discounts + sales_tax

  let final_cost := cost_after_tax - 12

  final_cost

theorem total_cost_is_correct :
  total_cost_of_gifts = 564.96 := by
sorry

end NUMINAMATH_GPT_total_cost_is_correct_l1589_158914


namespace NUMINAMATH_GPT_simplify_expression_l1589_158982

theorem simplify_expression :
  (2 + Real.sqrt 3)^2 - Real.sqrt 18 * Real.sqrt (2 / 3) = 7 + 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1589_158982


namespace NUMINAMATH_GPT_min_sum_of_3_digit_numbers_l1589_158995

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_3_digit (n : ℕ) := 100 ≤ n ∧ n ≤ 999

theorem min_sum_of_3_digit_numbers : 
  ∃ (a b c : ℕ), 
    a ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    b ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    c ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    a + b = c ∧ 
    a + b + c = 459 := 
sorry

end NUMINAMATH_GPT_min_sum_of_3_digit_numbers_l1589_158995


namespace NUMINAMATH_GPT_total_students_in_school_l1589_158958

noncomputable def small_school_students (boys girls : ℕ) (total_students : ℕ) : Prop :=
boys = 42 ∧ 
(girls : ℕ) = boys / 7 ∧
total_students = boys + girls

theorem total_students_in_school : small_school_students 42 6 48 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_school_l1589_158958


namespace NUMINAMATH_GPT_sqrt_ratio_simplify_l1589_158910

theorem sqrt_ratio_simplify :
  ( (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 75 = 12 / 5 ) :=
by
  let sqrt27 := Real.sqrt 27
  let sqrt243 := Real.sqrt 243
  let sqrt75 := Real.sqrt 75
  have h_sqrt27 : sqrt27 = Real.sqrt (3^2 * 3) := by sorry
  have h_sqrt243 : sqrt243 = Real.sqrt (3^5) := by sorry
  have h_sqrt75 : sqrt75 = Real.sqrt (3 * 5^2) := by sorry
  have h_simplified : (sqrt27 + sqrt243) / sqrt75 = 12 / 5 := by sorry
  exact h_simplified

end NUMINAMATH_GPT_sqrt_ratio_simplify_l1589_158910


namespace NUMINAMATH_GPT_sqrt_fraction_subtraction_l1589_158957

theorem sqrt_fraction_subtraction :
  (Real.sqrt (9 / 2) - Real.sqrt (2 / 9)) = (7 * Real.sqrt 2 / 6) :=
by sorry

end NUMINAMATH_GPT_sqrt_fraction_subtraction_l1589_158957


namespace NUMINAMATH_GPT_interest_rate_per_annum_l1589_158948

theorem interest_rate_per_annum (P A : ℝ) (T : ℝ)
  (principal_eq : P = 973.913043478261)
  (amount_eq : A = 1120)
  (time_eq : T = 3):
  (A - P) / (T * P) * 100 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_interest_rate_per_annum_l1589_158948


namespace NUMINAMATH_GPT_inequality_2_inequality_1_9_l1589_158921

variables {a : ℕ → ℝ}

-- Conditions
def non_negative (a : ℕ → ℝ) : Prop := ∀ n, a n ≥ 0
def boundary_zero (a : ℕ → ℝ) : Prop := a 1 = 0 ∧ a 9 = 0
def non_zero_interior (a : ℕ → ℝ) : Prop := ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a i ≠ 0

-- Proof problems
theorem inequality_2 (a : ℕ → ℝ) (h1 : non_negative a) (h2 : boundary_zero a) (h3 : non_zero_interior a) :
  ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i - 1) + a (i + 1) < 2 * a i := sorry

theorem inequality_1_9 (a : ℕ → ℝ) (h1 : non_negative a) (h2 : boundary_zero a) (h3 : non_zero_interior a) :
  ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i - 1) + a (i + 1) < 1.9 * a i := sorry

end NUMINAMATH_GPT_inequality_2_inequality_1_9_l1589_158921


namespace NUMINAMATH_GPT_sin_theta_value_l1589_158907

theorem sin_theta_value (a : ℝ) (h : a ≠ 0) (h_tan : Real.tan θ = -a) (h_point : P = (a, -1)) : Real.sin θ = -Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_sin_theta_value_l1589_158907


namespace NUMINAMATH_GPT_binom_12_9_is_220_l1589_158942

def choose (n k : ℕ) : ℕ := n.choose k

theorem binom_12_9_is_220 :
  choose 12 9 = 220 :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_binom_12_9_is_220_l1589_158942


namespace NUMINAMATH_GPT_find_remainder_l1589_158987

noncomputable def q (x : ℝ) : ℝ := (x^2010 + x^2009 + x^2008 + x + 1)
noncomputable def s (x : ℝ) := (q x) % (x^3 + 2*x^2 + 3*x + 1)

theorem find_remainder (x : ℝ) : (|s 2011| % 500) = 357 := by
    sorry

end NUMINAMATH_GPT_find_remainder_l1589_158987


namespace NUMINAMATH_GPT_qt_q_t_neq_2_l1589_158979

theorem qt_q_t_neq_2 (q t : ℕ) (hq : 0 < q) (ht : 0 < t) : q * t + q + t ≠ 2 :=
  sorry

end NUMINAMATH_GPT_qt_q_t_neq_2_l1589_158979


namespace NUMINAMATH_GPT_proportion_equivalence_l1589_158925

variable {x y : ℝ}

theorem proportion_equivalence (h : 3 * x = 5 * y) (hy : y ≠ 0) : 
  x / 5 = y / 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_proportion_equivalence_l1589_158925


namespace NUMINAMATH_GPT_crayons_total_l1589_158951

theorem crayons_total (Billy_crayons : ℝ) (Jane_crayons : ℝ)
  (h1 : Billy_crayons = 62.0) (h2 : Jane_crayons = 52.0) :
  Billy_crayons + Jane_crayons = 114.0 := 
by
  sorry

end NUMINAMATH_GPT_crayons_total_l1589_158951


namespace NUMINAMATH_GPT_colleen_paid_more_l1589_158929

-- Define the number of pencils Joy has
def joy_pencils : ℕ := 30

-- Define the number of pencils Colleen has
def colleen_pencils : ℕ := 50

-- Define the cost per pencil
def pencil_cost : ℕ := 4

-- The proof problem: Colleen paid $80 more for her pencils than Joy
theorem colleen_paid_more : 
  (colleen_pencils - joy_pencils) * pencil_cost = 80 := by
  sorry

end NUMINAMATH_GPT_colleen_paid_more_l1589_158929


namespace NUMINAMATH_GPT_vincent_total_packs_l1589_158909

noncomputable def total_packs (yesterday today_addition: ℕ) : ℕ :=
  let today := yesterday + today_addition
  yesterday + today

theorem vincent_total_packs
  (yesterday_packs : ℕ)
  (today_addition: ℕ)
  (hyesterday: yesterday_packs = 15)
  (htoday_addition: today_addition = 10) :
  total_packs yesterday_packs today_addition = 40 :=
by
  rw [hyesterday, htoday_addition]
  unfold total_packs
  -- at this point it simplifies to 15 + (15 + 10) = 40
  sorry

end NUMINAMATH_GPT_vincent_total_packs_l1589_158909


namespace NUMINAMATH_GPT_inequality_solution_set_range_of_a_l1589_158984

noncomputable def f (x : ℝ) := abs (2 * x - 1) - abs (x + 2)

theorem inequality_solution_set :
  { x : ℝ | f x > 0 } = { x : ℝ | x < -1 / 3 ∨ x > 3 } :=
sorry

theorem range_of_a (x0 : ℝ) (h : f x0 + 2 * a ^ 2 < 4 * a) :
  -1 / 2 < a ∧ a < 5 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_range_of_a_l1589_158984


namespace NUMINAMATH_GPT_positive_integer_root_k_l1589_158911

theorem positive_integer_root_k (k : ℕ) :
  (∃ x : ℕ, x > 0 ∧ x * x - 34 * x + 34 * k - 1 = 0) ↔ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_root_k_l1589_158911


namespace NUMINAMATH_GPT_x_gt_zero_sufficient_but_not_necessary_l1589_158912

theorem x_gt_zero_sufficient_but_not_necessary (x : ℝ): 
  (x > 0 → x ≠ 0) ∧ (x ≠ 0 → ¬ (x > 0)) → 
  ((x > 0 ↔ x ≠ 0) = false) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_x_gt_zero_sufficient_but_not_necessary_l1589_158912


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l1589_158976

theorem sum_of_consecutive_integers (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l1589_158976


namespace NUMINAMATH_GPT_intersection_M_N_l1589_158975

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N :
  M ∩ N = {x | 1 < x ∧ x ≤ 2} := 
sorry

end NUMINAMATH_GPT_intersection_M_N_l1589_158975


namespace NUMINAMATH_GPT_ratio_of_books_on_each_table_l1589_158986

-- Define the conditions
variables (number_of_tables number_of_books : ℕ)
variables (R : ℕ) -- Ratio we need to find

-- State the conditions
def conditions := (number_of_tables = 500) ∧ (number_of_books = 100000)

-- Mathematical Problem Statement
theorem ratio_of_books_on_each_table (h : conditions number_of_tables number_of_books) :
    100000 = 500 * R → R = 200 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_books_on_each_table_l1589_158986


namespace NUMINAMATH_GPT_probability_two_dice_same_l1589_158930

def fair_dice_probability (dice : ℕ) (sides : ℕ) : ℚ :=
  1 - ((sides.factorial / (sides - dice).factorial) / sides^dice)

theorem probability_two_dice_same (dice : ℕ) (sides : ℕ) (h1 : dice = 5) (h2 : sides = 10) :
  fair_dice_probability dice sides = 1744 / 2500 := by
  sorry

end NUMINAMATH_GPT_probability_two_dice_same_l1589_158930


namespace NUMINAMATH_GPT_estimate_total_balls_l1589_158917

theorem estimate_total_balls (red_balls : ℕ) (frequency : ℝ) (total_balls : ℕ) 
  (h_red : red_balls = 12) (h_freq : frequency = 0.6) 
  (h_eq : (red_balls : ℝ) / total_balls = frequency) : 
  total_balls = 20 :=
by
  sorry

end NUMINAMATH_GPT_estimate_total_balls_l1589_158917


namespace NUMINAMATH_GPT_count_arithmetic_sequence_l1589_158908

theorem count_arithmetic_sequence :
  let a1 := 2.5
  let an := 68.5
  let d := 6.0
  let offset := 0.5
  let adjusted_a1 := a1 + offset
  let adjusted_an := an + offset
  let n := (adjusted_an - adjusted_a1) / d + 1
  n = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_count_arithmetic_sequence_l1589_158908


namespace NUMINAMATH_GPT_distance_from_origin_l1589_158927

theorem distance_from_origin (A : ℝ) (h : |A - 0| = 4) : A = 4 ∨ A = -4 :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_from_origin_l1589_158927


namespace NUMINAMATH_GPT_total_puzzle_pieces_l1589_158956

theorem total_puzzle_pieces : 
  ∀ (p1 p2 p3 : ℕ), 
  p1 = 1000 → 
  p2 = p1 + p1 / 2 → 
  p3 = p1 + p1 / 2 → 
  p1 + p2 + p3 = 4000 := 
by 
  intros p1 p2 p3 
  intro h1 
  intro h2 
  intro h3 
  rw [h1, h2, h3] 
  norm_num
  sorry

end NUMINAMATH_GPT_total_puzzle_pieces_l1589_158956


namespace NUMINAMATH_GPT_initial_cookies_l1589_158919

variable (andys_cookies : ℕ)

def total_cookies_andy_ate : ℕ := 3
def total_cookies_brother_ate : ℕ := 5

def arithmetic_sequence_sum (n : ℕ) : ℕ := n * (2 * n - 1)

def total_cookies_team_ate : ℕ := arithmetic_sequence_sum 8

theorem initial_cookies :
  andys_cookies = total_cookies_andy_ate + total_cookies_brother_ate + total_cookies_team_ate :=
  by
    -- Here the missing proof would go
    sorry

end NUMINAMATH_GPT_initial_cookies_l1589_158919


namespace NUMINAMATH_GPT_fixed_point_l1589_158903

noncomputable def f (a : ℝ) (x : ℝ) := a^(x - 2) - 3

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_l1589_158903


namespace NUMINAMATH_GPT_small_denominator_difference_l1589_158955

theorem small_denominator_difference :
  ∃ (p q : ℕ), 0 < p ∧ 0 < q ∧
               (5 : ℚ) / 9 < (p : ℚ) / q ∧
               (p : ℚ) / q < 4 / 7 ∧
               (∀ r, 0 < r → (5 : ℚ) / 9 < (p : ℚ) / r → (p : ℚ) / r < 4 / 7 → q ≤ r) ∧
               q - p = 7 := 
  by
  sorry

end NUMINAMATH_GPT_small_denominator_difference_l1589_158955


namespace NUMINAMATH_GPT_percentage_of_females_l1589_158994

theorem percentage_of_females (total_passengers : ℕ)
  (first_class_percentage : ℝ) (male_fraction_first_class : ℝ)
  (females_coach_class : ℕ) (h1 : total_passengers = 120)
  (h2 : first_class_percentage = 0.10)
  (h3 : male_fraction_first_class = 1/3)
  (h4 : females_coach_class = 40) :
  (females_coach_class + (first_class_percentage * total_passengers - male_fraction_first_class * (first_class_percentage * total_passengers))) / total_passengers * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_females_l1589_158994


namespace NUMINAMATH_GPT_circle_equation_tangent_y_axis_center_on_line_chord_length_condition_l1589_158978

theorem circle_equation_tangent_y_axis_center_on_line_chord_length_condition :
  ∃ (x₀ y₀ r : ℝ), 
  (x₀ - 3 * y₀ = 0) ∧ 
  (r = |3 * y₀|) ∧ 
  ((x₀ + 3)^2 + (y₀ - 1)^2 = r^2 ∨ (x₀ - 3)^2 + (y₀ + 1)^2 = r^2) :=
sorry

end NUMINAMATH_GPT_circle_equation_tangent_y_axis_center_on_line_chord_length_condition_l1589_158978


namespace NUMINAMATH_GPT_silverware_probability_l1589_158990

-- Definitions based on the problem conditions
def total_silverware : ℕ := 8 + 10 + 7
def total_combinations : ℕ := Nat.choose total_silverware 4

def fork_combinations : ℕ := Nat.choose 8 2
def spoon_combinations : ℕ := Nat.choose 10 1
def knife_combinations : ℕ := Nat.choose 7 1

def favorable_combinations : ℕ := fork_combinations * spoon_combinations * knife_combinations
def specific_combination_probability : ℚ := favorable_combinations / total_combinations

-- The statement to prove the given probability
theorem silverware_probability :
  specific_combination_probability = 392 / 2530 :=
by
  sorry

end NUMINAMATH_GPT_silverware_probability_l1589_158990


namespace NUMINAMATH_GPT_problem_statement_l1589_158924

variable (a b c d x : ℕ)

theorem problem_statement
  (h1 : a + b = x)
  (h2 : b + c = 9)
  (h3 : c + d = 3)
  (h4 : a + d = 6) :
  x = 12 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1589_158924


namespace NUMINAMATH_GPT_simplify_polynomial_l1589_158918

theorem simplify_polynomial (x : ℝ) :
  (14 * x ^ 12 + 8 * x ^ 9 + 3 * x ^ 8) + (2 * x ^ 14 - x ^ 12 + 2 * x ^ 9 + 5 * x ^ 5 + 7 * x ^ 2 + 6) =
  2 * x ^ 14 + 13 * x ^ 12 + 10 * x ^ 9 + 3 * x ^ 8 + 5 * x ^ 5 + 7 * x ^ 2 + 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1589_158918


namespace NUMINAMATH_GPT_snack_eaters_remaining_l1589_158967

noncomputable def initial_snack_eaters := 5000 * 60 / 100
noncomputable def snack_eaters_after_1_hour := initial_snack_eaters + 25
noncomputable def snack_eaters_after_70_percent_left := snack_eaters_after_1_hour * 30 / 100
noncomputable def snack_eaters_after_2_hour := snack_eaters_after_70_percent_left + 50
noncomputable def snack_eaters_after_800_left := snack_eaters_after_2_hour - 800
noncomputable def snack_eaters_after_2_thirds_left := snack_eaters_after_800_left * 1 / 3
noncomputable def final_snack_eaters := snack_eaters_after_2_thirds_left + 100

theorem snack_eaters_remaining : final_snack_eaters = 153 :=
by
  have h1 : initial_snack_eaters = 3000 := by sorry
  have h2 : snack_eaters_after_1_hour = initial_snack_eaters + 25 := by sorry
  have h3 : snack_eaters_after_70_percent_left = snack_eaters_after_1_hour * 30 / 100 := by sorry
  have h4 : snack_eaters_after_2_hour = snack_eaters_after_70_percent_left + 50 := by sorry
  have h5 : snack_eaters_after_800_left = snack_eaters_after_2_hour - 800 := by sorry
  have h6 : snack_eaters_after_2_thirds_left = snack_eaters_after_800_left * 1 / 3 := by sorry
  have h7 : final_snack_eaters = snack_eaters_after_2_thirds_left + 100 := by sorry
  -- Prove that these equal 153 overall
  sorry

end NUMINAMATH_GPT_snack_eaters_remaining_l1589_158967


namespace NUMINAMATH_GPT_binom_20_5_l1589_158926

-- Definition of the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Problem statement
theorem binom_20_5 : binomial_coefficient 20 5 = 7752 := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_binom_20_5_l1589_158926


namespace NUMINAMATH_GPT_max_strips_cut_l1589_158949

-- Definitions: dimensions of the paper and the strips
def length_paper : ℕ := 14
def width_paper : ℕ := 11
def length_strip : ℕ := 4
def width_strip : ℕ := 1

-- States the main theorem: Maximum number of strips that can be cut from the rectangular piece of paper
theorem max_strips_cut (L W l w : ℕ) (H1 : L = 14) (H2 : W = 11) (H3 : l = 4) (H4 : w = 1) :
  ∃ n : ℕ, n = 33 :=
by
  sorry

end NUMINAMATH_GPT_max_strips_cut_l1589_158949


namespace NUMINAMATH_GPT_fiona_weekly_earnings_l1589_158963

theorem fiona_weekly_earnings :
  let monday_hours := 1.5
  let tuesday_hours := 1.25
  let wednesday_hours := 3.1667
  let thursday_hours := 0.75
  let hourly_wage := 4
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours
  let total_earnings := total_hours * hourly_wage
  total_earnings = 26.67 := by
  sorry

end NUMINAMATH_GPT_fiona_weekly_earnings_l1589_158963


namespace NUMINAMATH_GPT_eric_bike_speed_l1589_158980

def swim_distance : ℝ := 0.5
def swim_speed : ℝ := 1
def run_distance : ℝ := 2
def run_speed : ℝ := 8
def bike_distance : ℝ := 12
def total_time_limit : ℝ := 2

theorem eric_bike_speed :
  (swim_distance / swim_speed) + (run_distance / run_speed) + (bike_distance / (48/5)) < total_time_limit :=
by
  sorry

end NUMINAMATH_GPT_eric_bike_speed_l1589_158980


namespace NUMINAMATH_GPT_max_pens_l1589_158946

theorem max_pens (total_money notebook_cost pen_cost num_notebooks : ℝ) (notebook_qty pen_qty : ℕ):
  total_money = 18 ∧ notebook_cost = 3.6 ∧ pen_cost = 3 ∧ num_notebooks = 2 →
  (pen_qty = 1 ∨ pen_qty = 2 ∨ pen_qty = 3) ↔ (2 * notebook_cost + pen_qty * pen_cost ≤ total_money) :=
by {
  sorry
}

end NUMINAMATH_GPT_max_pens_l1589_158946


namespace NUMINAMATH_GPT_avg_speed_last_40_min_is_70_l1589_158941

noncomputable def avg_speed_last_interval
  (total_distance : ℝ) (total_time : ℝ)
  (speed_first_40_min : ℝ) (time_first_40_min : ℝ)
  (speed_second_40_min : ℝ) (time_second_40_min : ℝ) : ℝ :=
  let time_last_40_min := total_time - (time_first_40_min + time_second_40_min)
  let distance_first_40_min := speed_first_40_min * time_first_40_min
  let distance_second_40_min := speed_second_40_min * time_second_40_min
  let distance_last_40_min := total_distance - (distance_first_40_min + distance_second_40_min)
  distance_last_40_min / time_last_40_min

theorem avg_speed_last_40_min_is_70
  (h_total_distance : total_distance = 120)
  (h_total_time : total_time = 2)
  (h_speed_first_40_min : speed_first_40_min = 50)
  (h_time_first_40_min : time_first_40_min = 2 / 3)
  (h_speed_second_40_min : speed_second_40_min = 60)
  (h_time_second_40_min : time_second_40_min = 2 / 3) :
  avg_speed_last_interval 120 2 50 (2 / 3) 60 (2 / 3) = 70 :=
by
  sorry

end NUMINAMATH_GPT_avg_speed_last_40_min_is_70_l1589_158941


namespace NUMINAMATH_GPT_simplify_expression_l1589_158905

variable (a b : ℝ)

theorem simplify_expression (a b : ℝ) :
  (6 * a^5 * b^2) / (3 * a^3 * b^2) + ((2 * a * b^3)^2) / ((-b^2)^3) = -2 * a^2 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1589_158905


namespace NUMINAMATH_GPT_train_crossing_pole_time_l1589_158900

theorem train_crossing_pole_time :
  ∀ (length_of_train : ℝ) (speed_km_per_hr : ℝ) (t : ℝ),
    length_of_train = 45 →
    speed_km_per_hr = 108 →
    t = 1.5 →
    t = length_of_train / (speed_km_per_hr * 1000 / 3600) := 
  sorry

end NUMINAMATH_GPT_train_crossing_pole_time_l1589_158900


namespace NUMINAMATH_GPT_watermelon_count_l1589_158992

theorem watermelon_count (seeds_per_watermelon : ℕ) (total_seeds : ℕ)
  (h1 : seeds_per_watermelon = 100) (h2 : total_seeds = 400) : total_seeds / seeds_per_watermelon = 4 :=
by
  sorry

end NUMINAMATH_GPT_watermelon_count_l1589_158992


namespace NUMINAMATH_GPT_cost_of_camel_l1589_158940

theorem cost_of_camel
  (C H O E : ℝ)
  (h1 : 10 * C = 24 * H)
  (h2 : 16 * H = 4 * O)
  (h3 : 6 * O = 4 * E)
  (h4 : 10 * E = 140000) :
  C = 5600 :=
by
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_cost_of_camel_l1589_158940


namespace NUMINAMATH_GPT_eval_expr_at_x_eq_neg6_l1589_158997

-- Define the given condition
def x : ℤ := -4

-- Define the expression to be simplified and evaluated
def expr (x y : ℤ) : ℤ := ((x + y)^2 - y * (2 * x + y) - 8 * x) / (2 * x)

-- The theorem stating the result of the evaluated expression
theorem eval_expr_at_x_eq_neg6 (y : ℤ) : expr (-4) y = -6 := 
by
  sorry

end NUMINAMATH_GPT_eval_expr_at_x_eq_neg6_l1589_158997


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_for_circle_l1589_158935

theorem sufficient_but_not_necessary_for_circle (m : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y + m = 0) → (m = 0) → (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = r^2)) ∧
  ¬(∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y + m = 0) → (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = r^2) → (m = 0)) := sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_for_circle_l1589_158935


namespace NUMINAMATH_GPT_range_of_a_if_slope_is_obtuse_l1589_158959

theorem range_of_a_if_slope_is_obtuse : 
  ∀ a : ℝ, (a^2 + 2 * a < 0) → -2 < a ∧ a < 0 :=
by
  intro a
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_if_slope_is_obtuse_l1589_158959


namespace NUMINAMATH_GPT_gcd_7854_13843_l1589_158983

theorem gcd_7854_13843 : Nat.gcd 7854 13843 = 1 := 
  sorry

end NUMINAMATH_GPT_gcd_7854_13843_l1589_158983


namespace NUMINAMATH_GPT_ellipse_circle_parallelogram_condition_l1589_158977

theorem ellipse_circle_parallelogram_condition
  (a b : ℝ)
  (C₀ : ∀ x y : ℝ, x^2 + y^2 = 1)
  (C₁ : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (h : a > 0 ∧ b > 0 ∧ a > b) :
  1 / a^2 + 1 / b^2 = 1 := by
  sorry

end NUMINAMATH_GPT_ellipse_circle_parallelogram_condition_l1589_158977


namespace NUMINAMATH_GPT_combined_cost_is_450_l1589_158902

-- Given conditions
def bench_cost : ℕ := 150
def table_cost : ℕ := 2 * bench_cost

-- The statement we want to prove
theorem combined_cost_is_450 : bench_cost + table_cost = 450 :=
by
  sorry

end NUMINAMATH_GPT_combined_cost_is_450_l1589_158902


namespace NUMINAMATH_GPT_minimum_xy_l1589_158938

theorem minimum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : xy ≥ 64 :=
sorry

end NUMINAMATH_GPT_minimum_xy_l1589_158938


namespace NUMINAMATH_GPT_four_digit_number_exists_l1589_158989

-- Definitions corresponding to the conditions in the problem
def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def follows_scheme (n : ℕ) (d : ℕ) : Prop :=
  -- Placeholder for the scheme condition
  sorry

-- The Lean statement for the proof problem
theorem four_digit_number_exists :
  ∃ n d1 d2 : ℕ, is_four_digit_number n ∧ follows_scheme n d1 ∧ follows_scheme n d2 ∧ 
  (n = 1014 ∨ n = 1035 ∨ n = 1512) :=
by {
  -- Placeholder for proof steps
  sorry
}

end NUMINAMATH_GPT_four_digit_number_exists_l1589_158989


namespace NUMINAMATH_GPT_seven_expression_one_seven_expression_two_l1589_158961

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end NUMINAMATH_GPT_seven_expression_one_seven_expression_two_l1589_158961


namespace NUMINAMATH_GPT_greatest_integer_less_than_or_equal_to_l1589_158962

theorem greatest_integer_less_than_or_equal_to (x : ℝ) (h : x = 2 + Real.sqrt 3) : 
  ⌊x^3⌋ = 51 :=
by
  have h' : x ^ 3 = (2 + Real.sqrt 3) ^ 3 := by rw [h]
  sorry

end NUMINAMATH_GPT_greatest_integer_less_than_or_equal_to_l1589_158962


namespace NUMINAMATH_GPT_fraction_ratio_l1589_158964

theorem fraction_ratio (x : ℚ) (h1 : 2 / 5 / (3 / 7) = x / (1 / 2)) :
  x = 7 / 15 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_fraction_ratio_l1589_158964


namespace NUMINAMATH_GPT_combined_rocket_height_l1589_158945

theorem combined_rocket_height :
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 :=
by
  sorry

end NUMINAMATH_GPT_combined_rocket_height_l1589_158945


namespace NUMINAMATH_GPT_A_investment_is_100_l1589_158960

-- Definitions directly from the conditions in a)
def A_investment (X : ℝ) := X * 12
def B_investment : ℝ := 200 * 6
def total_profit : ℝ := 100
def A_share_of_profit : ℝ := 50

-- Prove that given these conditions, A's initial investment X is 100
theorem A_investment_is_100 (X : ℝ) (h : A_share_of_profit / total_profit = A_investment X / B_investment) : X = 100 :=
by
  sorry

end NUMINAMATH_GPT_A_investment_is_100_l1589_158960


namespace NUMINAMATH_GPT_arthur_walks_distance_l1589_158952

theorem arthur_walks_distance :
  ∀ (blocks_east blocks_north blocks_first blocks_other distance_first distance_other : ℕ)
  (fraction_first fraction_other : ℚ),
    blocks_east = 8 →
    blocks_north = 16 →
    blocks_first = 10 →
    blocks_other = (blocks_east + blocks_north) - blocks_first →
    fraction_first = 1 / 3 →
    fraction_other = 1 / 4 →
    distance_first = blocks_first * fraction_first →
    distance_other = blocks_other * fraction_other →
    (distance_first + distance_other) = 41 / 6 :=
by
  intros blocks_east blocks_north blocks_first blocks_other distance_first distance_other fraction_first fraction_other
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_arthur_walks_distance_l1589_158952


namespace NUMINAMATH_GPT_books_fill_shelf_l1589_158920

theorem books_fill_shelf
  (A H S M E : ℕ)
  (h1 : A ≠ H) (h2 : S ≠ M) (h3 : M ≠ H) (h4 : E > 0)
  (Eq1 : A > 0) (Eq2 : H > 0) (Eq3 : S > 0) (Eq4 : M > 0)
  (h5 : A ≠ S) (h6 : E ≠ A) (h7 : E ≠ H) (h8 : E ≠ S) (h9 : E ≠ M) :
  E = (A * M - S * H) / (M - H) :=
by
  sorry

end NUMINAMATH_GPT_books_fill_shelf_l1589_158920
