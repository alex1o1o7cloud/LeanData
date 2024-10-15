import Mathlib

namespace NUMINAMATH_GPT_monica_cookies_left_l1334_133435

theorem monica_cookies_left 
  (father_cookies : ℕ) 
  (mother_cookies : ℕ) 
  (brother_cookies : ℕ) 
  (sister_cookies : ℕ) 
  (aunt_cookies : ℕ) 
  (cousin_cookies : ℕ) 
  (total_cookies : ℕ)
  (father_cookies_eq : father_cookies = 12)
  (mother_cookies_eq : mother_cookies = father_cookies / 2)
  (brother_cookies_eq : brother_cookies = mother_cookies + 2)
  (sister_cookies_eq : sister_cookies = brother_cookies * 3)
  (aunt_cookies_eq : aunt_cookies = father_cookies * 2)
  (cousin_cookies_eq : cousin_cookies = aunt_cookies - 5)
  (total_cookies_eq : total_cookies = 120) : 
  total_cookies - (father_cookies + mother_cookies + brother_cookies + sister_cookies + aunt_cookies + cousin_cookies) = 27 :=
by
  sorry

end NUMINAMATH_GPT_monica_cookies_left_l1334_133435


namespace NUMINAMATH_GPT_min_abs_sum_l1334_133498

theorem min_abs_sum (x : ℝ) : (∃ x : ℝ, ∀ y : ℝ, (|y - 2| + |y - 47| ≥ |x - 2| + |x - 47|)) → (|x - 2| + |x - 47| = 45) :=
by
  sorry

end NUMINAMATH_GPT_min_abs_sum_l1334_133498


namespace NUMINAMATH_GPT_value_of_m_div_x_l1334_133450

variable (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ratio : a / b = 4 / 5)

def x := a + 0.25 * a
def m := b - 0.40 * b

theorem value_of_m_div_x (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ratio : a / b = 4 / 5) :
    m / x = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_div_x_l1334_133450


namespace NUMINAMATH_GPT_problem1_problem2_l1334_133476

-- Proof Problem 1:

theorem problem1 : (5 / 3) ^ 2004 * (3 / 5) ^ 2003 = 5 / 3 := by
  sorry

-- Proof Problem 2:

theorem problem2 (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1334_133476


namespace NUMINAMATH_GPT_min_a2_plus_b2_quartic_eq_l1334_133445

theorem min_a2_plus_b2_quartic_eq (a b : ℝ) (x : ℝ) 
  (h : x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  a^2 + b^2 ≥ 4/5 := 
sorry

end NUMINAMATH_GPT_min_a2_plus_b2_quartic_eq_l1334_133445


namespace NUMINAMATH_GPT_square_side_length_l1334_133443

theorem square_side_length (x : ℝ) (h : 4 * x = 8 * Real.pi) : x = 6.28 := 
by {
  -- proof will go here
  sorry
}

end NUMINAMATH_GPT_square_side_length_l1334_133443


namespace NUMINAMATH_GPT_triangle_angle_tangent_ratio_triangle_tan_A_minus_B_maximum_l1334_133496

theorem triangle_angle_tangent_ratio (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = 3 / 5 * c) :
  Real.tan A / Real.tan B = 4 := sorry

theorem triangle_tan_A_minus_B_maximum (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = 3 / 5 * c)
  (h2 : Real.tan A / Real.tan B = 4) : Real.tan (A - B) ≤ 3 / 4 := sorry

end NUMINAMATH_GPT_triangle_angle_tangent_ratio_triangle_tan_A_minus_B_maximum_l1334_133496


namespace NUMINAMATH_GPT_total_distance_AD_l1334_133465

theorem total_distance_AD :
  let d_AB := 100
  let d_BC := d_AB + 50
  let d_CD := 2 * d_BC
  d_AB + d_BC + d_CD = 550 := by
  sorry

end NUMINAMATH_GPT_total_distance_AD_l1334_133465


namespace NUMINAMATH_GPT_members_count_l1334_133475

theorem members_count (n : ℕ) (h : n * n = 2025) : n = 45 :=
sorry

end NUMINAMATH_GPT_members_count_l1334_133475


namespace NUMINAMATH_GPT_ratio_of_chickens_in_run_to_coop_l1334_133489

def chickens_in_coop : ℕ := 14
def free_ranging_chickens : ℕ := 52
def run_condition (R : ℕ) : Prop := 2 * R - 4 = 52

theorem ratio_of_chickens_in_run_to_coop (R : ℕ) (hR : run_condition R) :
  R / chickens_in_coop = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_chickens_in_run_to_coop_l1334_133489


namespace NUMINAMATH_GPT_order_b_gt_c_gt_a_l1334_133495

noncomputable def a : ℝ := Real.log 2.6
def b : ℝ := 0.5 * 1.8^2
noncomputable def c : ℝ := 1.1^5

theorem order_b_gt_c_gt_a : b > c ∧ c > a := by
  sorry

end NUMINAMATH_GPT_order_b_gt_c_gt_a_l1334_133495


namespace NUMINAMATH_GPT_find_number_l1334_133471

theorem find_number (x : ℝ) (h : 75 = 0.6 * x) : x = 125 :=
sorry

end NUMINAMATH_GPT_find_number_l1334_133471


namespace NUMINAMATH_GPT_earnings_per_widget_l1334_133422

theorem earnings_per_widget (W_h : ℝ) (H_w : ℕ) (W_t : ℕ) (E_w : ℝ) (E : ℝ) :
  W_h = 12.50 ∧ H_w = 40 ∧ W_t = 1000 ∧ E_w = 660 →
  E = 0.16 :=
by
  sorry

end NUMINAMATH_GPT_earnings_per_widget_l1334_133422


namespace NUMINAMATH_GPT_value_of_ak_l1334_133416

noncomputable def Sn (n : ℕ) : ℤ := n^2 - 9 * n
noncomputable def a (n : ℕ) : ℤ := Sn n - Sn (n - 1)

theorem value_of_ak (k : ℕ) (hk : 5 < a k ∧ a k < 8) : a k = 6 := by
  sorry

end NUMINAMATH_GPT_value_of_ak_l1334_133416


namespace NUMINAMATH_GPT_factor_expression_zero_l1334_133409

theorem factor_expression_zero (a b c : ℝ) (h : a + b + c ≠ 0) :
  (a^3 - b^3)^2 + (b^3 - c^3)^2 + (c^3 - a^3)^2 = 0 :=
sorry

end NUMINAMATH_GPT_factor_expression_zero_l1334_133409


namespace NUMINAMATH_GPT_sum_products_roots_l1334_133497

theorem sum_products_roots :
  (∃ p q r : ℂ, (3 * p^3 - 5 * p^2 + 12 * p - 10 = 0) ∧
                  (3 * q^3 - 5 * q^2 + 12 * q - 10 = 0) ∧
                  (3 * r^3 - 5 * r^2 + 12 * r - 10 = 0) ∧
                  (p ≠ q) ∧ (q ≠ r) ∧ (p ≠ r)) →
  ∀ p q r : ℂ, (3 * p) * (q * r) + (3 * q) * (r * p) + (3 * r) * (p * q) =
    (3 * p * q * r) :=
sorry

end NUMINAMATH_GPT_sum_products_roots_l1334_133497


namespace NUMINAMATH_GPT_relationship_between_x_and_y_l1334_133499

variable (u : ℝ)

theorem relationship_between_x_and_y (h : u > 0) (hx : x = (u + 1)^(1 / u)) (hy : y = (u + 1)^((u + 1) / u)) :
  y^x = x^y :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_x_and_y_l1334_133499


namespace NUMINAMATH_GPT_distinct_arrangements_of_pebbles_in_octagon_l1334_133429

noncomputable def number_of_distinct_arrangements : ℕ :=
  (Nat.factorial 8) / 16

theorem distinct_arrangements_of_pebbles_in_octagon : 
  number_of_distinct_arrangements = 2520 :=
by
  sorry

end NUMINAMATH_GPT_distinct_arrangements_of_pebbles_in_octagon_l1334_133429


namespace NUMINAMATH_GPT_button_remainders_l1334_133421

theorem button_remainders 
  (a : ℤ)
  (h1 : a % 2 = 1)
  (h2 : a % 3 = 1)
  (h3 : a % 4 = 3)
  (h4 : a % 5 = 3) :
  a % 12 = 7 := 
sorry

end NUMINAMATH_GPT_button_remainders_l1334_133421


namespace NUMINAMATH_GPT_sum_and_product_of_roots_l1334_133441

theorem sum_and_product_of_roots (m n : ℝ) (h1 : (m / 3) = 9) (h2 : (n / 3) = 20) : m + n = 87 :=
by
  sorry

end NUMINAMATH_GPT_sum_and_product_of_roots_l1334_133441


namespace NUMINAMATH_GPT_pen_defect_probability_l1334_133404

theorem pen_defect_probability :
  ∀ (n m : ℕ) (k : ℚ), n = 12 → m = 4 → k = 2 → 
  (8 / 12) * (7 / 11) = 141 / 330 := 
by
  intros n m k h1 h2 h3
  sorry

end NUMINAMATH_GPT_pen_defect_probability_l1334_133404


namespace NUMINAMATH_GPT_courtyard_length_l1334_133466

/-- Given the following conditions:
  1. The width of the courtyard is 16.5 meters.
  2. 66 paving stones are required.
  3. Each paving stone measures 2.5 meters by 2 meters.
  Prove that the length of the rectangular courtyard is 20 meters. -/
theorem courtyard_length :
  ∃ L : ℝ, L = 20 ∧ 
           (∃ W : ℝ, W = 16.5) ∧ 
           (∃ n : ℕ, n = 66) ∧ 
           (∃ A : ℝ, A = 2.5 * 2) ∧
           n * A = L * W :=
by
  sorry

end NUMINAMATH_GPT_courtyard_length_l1334_133466


namespace NUMINAMATH_GPT_tom_dimes_now_l1334_133401

-- Define the initial number of dimes and the number of dimes given by dad
def initial_dimes : ℕ := 15
def dimes_given_by_dad : ℕ := 33

-- Define the final count of dimes Tom has now
def final_dimes (initial_dimes dimes_given_by_dad : ℕ) : ℕ :=
  initial_dimes + dimes_given_by_dad

-- The main theorem to prove "how many dimes Tom has now"
theorem tom_dimes_now : initial_dimes + dimes_given_by_dad = 48 :=
by
  -- The proof can be skipped using sorry
  sorry

end NUMINAMATH_GPT_tom_dimes_now_l1334_133401


namespace NUMINAMATH_GPT_complex_solution_l1334_133474

theorem complex_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) (hz : (3 - 4 * i) * z = 5 * i) : z = (4 / 5) + (3 / 5) * i :=
by {
  sorry
}

end NUMINAMATH_GPT_complex_solution_l1334_133474


namespace NUMINAMATH_GPT_prove_inequality_l1334_133449

noncomputable def valid_x (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ (1-Real.sqrt 5)/2 ∧ x ≠ (1+Real.sqrt 5)/2

noncomputable def valid_intervals (x : ℝ) : Prop :=
  (x ≥ -1 ∧ x < (1 - Real.sqrt 5) / 2) ∨
  ((1 - Real.sqrt 5) / 2 < x ∧ x < 0) ∨
  (0 < x ∧ x < (1 + Real.sqrt 5) / 2) ∨
  (x > (1 + Real.sqrt 5) / 2)

theorem prove_inequality (x : ℝ) (hx : valid_x x) :
  (x^2 + x^3 - x^4) / (x + x^2 - x^3) ≥ -1 ↔ valid_intervals x := by
  sorry

end NUMINAMATH_GPT_prove_inequality_l1334_133449


namespace NUMINAMATH_GPT_perpendicular_lines_l1334_133451

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, (a * x - y + 2 * a = 0) → ((2 * a - 1) * x + a * y + a = 0) -> 
  (a ≠ 0 → ∃ k : ℝ, k = (a * ((1 - 2 * a) / a)) ∧ k = -1) -> a * ((1 - 2 * a) / a) = -1) →
  a = 0 ∨ a = 1 := by sorry

end NUMINAMATH_GPT_perpendicular_lines_l1334_133451


namespace NUMINAMATH_GPT_unique_vector_a_l1334_133425

-- Defining the vectors
def vector_a (x y : ℝ) : ℝ × ℝ := (x, y)
def vector_b (x y : ℝ) : ℝ × ℝ := (x^2, y^2)
def vector_c : ℝ × ℝ := (1, 1)
def vector_d : ℝ × ℝ := (2, 2)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The Lean statement to prove
theorem unique_vector_a (x y : ℝ) 
  (h1 : dot_product (vector_a x y) vector_c = 1)
  (h2 : dot_product (vector_b x y) vector_d = 1) : 
  vector_a x y = vector_a (1/2) (1/2) :=
by {
  sorry 
}

end NUMINAMATH_GPT_unique_vector_a_l1334_133425


namespace NUMINAMATH_GPT_original_water_depth_in_larger_vase_l1334_133419

-- Definitions based on the conditions
noncomputable def largerVaseDiameter := 20 -- in cm
noncomputable def smallerVaseDiameter := 10 -- in cm
noncomputable def smallerVaseHeight := 16 -- in cm

-- Proving the original depth of the water in the larger vase
theorem original_water_depth_in_larger_vase :
  ∃ depth : ℝ, depth = 14 :=
by
  sorry

end NUMINAMATH_GPT_original_water_depth_in_larger_vase_l1334_133419


namespace NUMINAMATH_GPT_fraction_sum_lt_one_l1334_133453

theorem fraction_sum_lt_one (n : ℕ) (h_pos : n > 0) : 
  (1 / 2 + 1 / 3 + 1 / 10 + 1 / n < 1) ↔ (n > 15) :=
sorry

end NUMINAMATH_GPT_fraction_sum_lt_one_l1334_133453


namespace NUMINAMATH_GPT_base7_to_base10_of_645_l1334_133490

theorem base7_to_base10_of_645 :
  (6 * 7^2 + 4 * 7^1 + 5 * 7^0) = 327 := 
by 
  sorry

end NUMINAMATH_GPT_base7_to_base10_of_645_l1334_133490


namespace NUMINAMATH_GPT_four_distinct_real_solutions_l1334_133485

noncomputable def polynomial (a b c d e x : ℝ) : ℝ :=
  (x - a) * (x - b) * (x - c) * (x - d) * (x - e)

noncomputable def derivative (a b c d e x : ℝ) : ℝ :=
  (x - b) * (x - c) * (x - d) * (x - e) + 
  (x - a) * (x - c) * (x - d) * (x - e) + 
  (x - a) * (x - b) * (x - d) * (x - e) +
  (x - a) * (x - b) * (x - c) * (x - e) +
  (x - a) * (x - b) * (x - c) * (x - d)

theorem four_distinct_real_solutions (a b c d e : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃ x1 x2 x3 x4 : ℝ, 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
    (derivative a b c d e x1 = 0 ∧ derivative a b c d e x2 = 0 ∧ derivative a b c d e x3 = 0 ∧ derivative a b c d e x4 = 0) :=
sorry

end NUMINAMATH_GPT_four_distinct_real_solutions_l1334_133485


namespace NUMINAMATH_GPT_completing_square_transformation_l1334_133493

theorem completing_square_transformation (x : ℝ) :
  x^2 - 2 * x - 5 = 0 -> (x - 1)^2 = 6 :=
by {
  sorry -- Proof to be completed
}

end NUMINAMATH_GPT_completing_square_transformation_l1334_133493


namespace NUMINAMATH_GPT_trapezoid_area_l1334_133426

theorem trapezoid_area (A B : ℝ) (n : ℕ) (hA : A = 36) (hB : B = 4) (hn : n = 6) :
    (A - B) / n = 5.33 := 
by 
  -- Given conditions and the goal
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1334_133426


namespace NUMINAMATH_GPT_chameleons_color_change_l1334_133458

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end NUMINAMATH_GPT_chameleons_color_change_l1334_133458


namespace NUMINAMATH_GPT_certain_number_is_48_l1334_133468

theorem certain_number_is_48 (x : ℕ) (h : x = 4) : 36 + 3 * x = 48 := by
  sorry

end NUMINAMATH_GPT_certain_number_is_48_l1334_133468


namespace NUMINAMATH_GPT_minimum_value_l1334_133487

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem minimum_value (a m n : ℝ)
    (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1)
    (h_a_on_graph : ∀ x, log_a a (x + 3) - 1 = 0 → x = -2)
    (h_on_line : 2 * m + n = 2)
    (h_mn_pos : m * n > 0) :
    (1 / m) + (2 / n) = 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l1334_133487


namespace NUMINAMATH_GPT_puzzles_sold_correct_l1334_133494

def science_kits_sold : ℕ := 45
def puzzles_sold : ℕ := science_kits_sold - 9

theorem puzzles_sold_correct : puzzles_sold = 36 := by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_puzzles_sold_correct_l1334_133494


namespace NUMINAMATH_GPT_april_plant_arrangement_l1334_133427

theorem april_plant_arrangement :
    let nBasil := 5
    let nTomato := 4
    let nPairs := nTomato / 2
    let nUnits := nBasil + nPairs
    let totalWays := (Nat.factorial nUnits) * (Nat.factorial nPairs) * (Nat.factorial (nPairs - 1))
    totalWays = 20160 := by
{
  let nBasil := 5
  let nTomato := 4
  let nPairs := nTomato / 2
  let nUnits := nBasil + nPairs
  let totalWays := (Nat.factorial nUnits) * (Nat.factorial nPairs) * (Nat.factorial (nPairs - 1))
  sorry
}

end NUMINAMATH_GPT_april_plant_arrangement_l1334_133427


namespace NUMINAMATH_GPT_pos_int_fraction_iff_l1334_133457

theorem pos_int_fraction_iff (p : ℕ) (hp : p > 0) : (∃ k : ℕ, 4 * p + 11 = k * (2 * p - 7)) ↔ (p = 4 ∨ p = 5) := 
sorry

end NUMINAMATH_GPT_pos_int_fraction_iff_l1334_133457


namespace NUMINAMATH_GPT_average_first_20_multiples_of_17_l1334_133463

theorem average_first_20_multiples_of_17 :
  (20 / 2 : ℝ) * (17 + 17 * 20) / 20 = 178.5 := by
  sorry

end NUMINAMATH_GPT_average_first_20_multiples_of_17_l1334_133463


namespace NUMINAMATH_GPT_part_i_part_ii_l1334_133467

variable {b c : ℤ}

theorem part_i (hb : b ≠ 0) (hc : c ≠ 0) 
    (cond1 : ∃ m n : ℤ, m ≠ n ∧ (m + n = -b) ∧ (m * n = c))
    (cond2 : ∃ r s : ℤ, r ≠ s ∧ (r + s = -b) ∧ (r * s = -c)) :
    ∃ (p q : ℤ), p > 0 ∧ q > 0 ∧ p ≠ q ∧ 2 * b ^ 2 = p ^ 2 + q ^ 2 :=
sorry

theorem part_ii (hb : b ≠ 0) (hc : c ≠ 0) 
    (cond1 : ∃ m n : ℤ, m ≠ n ∧ (m + n = -b) ∧ (m * n = c))
    (cond2 : ∃ r s : ℤ, r ≠ s ∧ (r + s = -b) ∧ (r * s = -c)) :
    ∃ (r s : ℤ), r > 0 ∧ s > 0 ∧ r ≠ s ∧ b ^ 2 = r ^ 2 + s ^ 2 :=
sorry

end NUMINAMATH_GPT_part_i_part_ii_l1334_133467


namespace NUMINAMATH_GPT_theta_digit_l1334_133484

theorem theta_digit (Θ : ℕ) (h : Θ ≠ 0) (h1 : 252 / Θ = 10 * 4 + Θ + Θ) : Θ = 5 :=
  sorry

end NUMINAMATH_GPT_theta_digit_l1334_133484


namespace NUMINAMATH_GPT_daniel_practices_total_minutes_in_week_l1334_133488

theorem daniel_practices_total_minutes_in_week :
  let school_minutes_per_day := 15
  let school_days := 5
  let weekend_minutes_per_day := 2 * school_minutes_per_day
  let weekend_days := 2
  let total_school_week_minutes := school_minutes_per_day * school_days
  let total_weekend_minutes := weekend_minutes_per_day * weekend_days
  total_school_week_minutes + total_weekend_minutes = 135 :=
by
  sorry

end NUMINAMATH_GPT_daniel_practices_total_minutes_in_week_l1334_133488


namespace NUMINAMATH_GPT_arithmetic_geom_sequence_a2_l1334_133448

theorem arithmetic_geom_sequence_a2 :
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n+1) = a n + 2) →  -- Arithmetic sequence with common difference of 2
    a 1 * a 4 = a 3 ^ 2 →  -- Geometric sequence property for a_1, a_3, a_4
    a 2 = -6 :=             -- The value of a_2
by
  intros a h_arith h_geom
  sorry

end NUMINAMATH_GPT_arithmetic_geom_sequence_a2_l1334_133448


namespace NUMINAMATH_GPT_total_turnips_l1334_133431

-- Conditions
def turnips_keith : ℕ := 6
def turnips_alyssa : ℕ := 9

-- Statement to be proved
theorem total_turnips : turnips_keith + turnips_alyssa = 15 := by
  -- Proof is not required for this prompt, so we use sorry
  sorry

end NUMINAMATH_GPT_total_turnips_l1334_133431


namespace NUMINAMATH_GPT_problem1_problem2_l1334_133492

noncomputable def A : Set ℝ := Set.Icc 1 4
noncomputable def B (a : ℝ) : Set ℝ := Set.Iio a

-- Problem 1
theorem problem1 (A := A) (B := B 4) : A ∩ B = Set.Icc 1 4 := by
  sorry 

-- Problem 2
theorem problem2 (A := A) : ∀ a : ℝ, (A ⊆ B a) → (4 ≤ a) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1334_133492


namespace NUMINAMATH_GPT_athletes_same_color_probability_l1334_133491

theorem athletes_same_color_probability :
  let colors := ["red", "white", "blue"]
  let total_ways := 3 * 3
  let same_color_ways := 3
  total_ways > 0 → 
  (same_color_ways : ℚ) / (total_ways : ℚ) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_athletes_same_color_probability_l1334_133491


namespace NUMINAMATH_GPT_slope_of_perpendicular_line_l1334_133439

theorem slope_of_perpendicular_line (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ m : ℝ, a * x - b * y = c → m = - (b / a) :=
by
  -- Here we state the definition and conditions provided in the problem
  -- And indicate what we want to prove (that the slope is -b/a in this case)
  sorry

end NUMINAMATH_GPT_slope_of_perpendicular_line_l1334_133439


namespace NUMINAMATH_GPT_divisor_of_condition_l1334_133479

theorem divisor_of_condition {d z : ℤ} (h1 : ∃ k : ℤ, z = k * d + 6)
  (h2 : ∃ m : ℤ, (z + 3) = d * m) : d = 9 := 
sorry

end NUMINAMATH_GPT_divisor_of_condition_l1334_133479


namespace NUMINAMATH_GPT_bricks_required_to_pave_courtyard_l1334_133411

theorem bricks_required_to_pave_courtyard :
  let courtyard_length_m := 24
  let courtyard_width_m := 14
  let brick_length_cm := 25
  let brick_width_cm := 15
  let courtyard_area_m2 := courtyard_length_m * courtyard_width_m
  let courtyard_area_cm2 := courtyard_area_m2 * 10000
  let brick_area_cm2 := brick_length_cm * brick_width_cm
  let num_bricks := courtyard_area_cm2 / brick_area_cm2
  num_bricks = 8960 := by
  {
    -- Additional context not needed for theorem statement, mock proof omitted
    sorry
  }

end NUMINAMATH_GPT_bricks_required_to_pave_courtyard_l1334_133411


namespace NUMINAMATH_GPT_gum_candy_ratio_l1334_133436

theorem gum_candy_ratio
  (g c : ℝ)  -- let g be the cost of a stick of gum and c be the cost of a candy bar.
  (hc : c = 1.5)  -- the cost of each candy bar is $1.5
  (h_total_cost : 2 * g + 3 * c = 6)  -- total cost of 2 sticks of gum and 3 candy bars is $6
  : g / c = 1 / 2 := -- the ratio of the cost of gum to candy is 1:2
sorry

end NUMINAMATH_GPT_gum_candy_ratio_l1334_133436


namespace NUMINAMATH_GPT_max_height_of_projectile_l1334_133438

def projectile_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height_of_projectile : 
  ∃ t : ℝ, projectile_height t = 161 :=
sorry

end NUMINAMATH_GPT_max_height_of_projectile_l1334_133438


namespace NUMINAMATH_GPT_find_ab_l1334_133470

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l1334_133470


namespace NUMINAMATH_GPT_find_x_plus_3y_l1334_133413

variables {α : Type*} {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (x y : ℝ)
variables (OA OB OC OD OE : V)

-- Defining the conditions
def condition1 := OA = (1/2) • OB + x • OC + y • OD
def condition2 := OB = 2 • x • OC + (1/3) • OD + y • OE

-- Writing the theorem statement
theorem find_x_plus_3y (h1 : condition1 x y OA OB OC OD) (h2 : condition2 x y OB OC OD OE) : 
  x + 3 * y = 7 / 6 := 
sorry

end NUMINAMATH_GPT_find_x_plus_3y_l1334_133413


namespace NUMINAMATH_GPT_right_triangle_leg_square_l1334_133402

theorem right_triangle_leg_square (a b c : ℝ) 
  (h1 : c = a + 2) 
  (h2 : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_leg_square_l1334_133402


namespace NUMINAMATH_GPT_price_of_eraser_l1334_133454

variables (x y : ℝ)

theorem price_of_eraser : 
  (3 * x + 5 * y = 10.6) ∧ (4 * x + 4 * y = 12) → x = 2.2 :=
by
  sorry

end NUMINAMATH_GPT_price_of_eraser_l1334_133454


namespace NUMINAMATH_GPT_thickness_of_wall_l1334_133481

theorem thickness_of_wall 
    (brick_length cm : ℝ)
    (brick_width cm : ℝ)
    (brick_height cm : ℝ)
    (num_bricks : ℝ)
    (wall_length cm : ℝ)
    (wall_height cm : ℝ)
    (wall_thickness cm : ℝ) :
    brick_length = 25 → 
    brick_width = 11.25 → 
    brick_height = 6 →
    num_bricks = 7200 → 
    wall_length = 900 → 
    wall_height = 600 →
    wall_length * wall_height * wall_thickness = num_bricks * (brick_length * brick_width * brick_height) →
    wall_thickness = 22.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_thickness_of_wall_l1334_133481


namespace NUMINAMATH_GPT_eval_expression_l1334_133405

theorem eval_expression :
  2^0 + 9^5 / 9^3 = 82 :=
by
  have h1 : 2^0 = 1 := by sorry
  have h2 : 9^5 / 9^3 = 9^(5-3) := by sorry
  have h3 : 9^(5-3) = 9^2 := by sorry
  have h4 : 9^2 = 81 := by sorry
  sorry

end NUMINAMATH_GPT_eval_expression_l1334_133405


namespace NUMINAMATH_GPT_symmetric_point_line_l1334_133437

theorem symmetric_point_line (a b : ℝ) :
  (∀ (x y : ℝ), (y - 2) / (x - 1) = -2 → (x + 1)/2 + 2 * (y + 2)/2 - 10 = 0) →
  a = 3 ∧ b = 6 := by
  intro h
  sorry

end NUMINAMATH_GPT_symmetric_point_line_l1334_133437


namespace NUMINAMATH_GPT_focus_of_parabola_l1334_133456

theorem focus_of_parabola (a : ℝ) (h : ℝ) (k : ℝ) (x y : ℝ) :
  (∀ x, y = a * (x - h) ^ 2 + k) →
  a = -2 ∧ h = 0 ∧ k = 4 →
  (0, y - (1 / (4 * a))) = (0, 31 / 8) := by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l1334_133456


namespace NUMINAMATH_GPT_solve_equation_l1334_133410

theorem solve_equation (x : ℝ) (floor : ℝ → ℤ) 
  (h_floor : ∀ y, floor y ≤ y ∧ y < floor y + 1) :
  (floor (20 * x + 23) = 20 + 23 * x) ↔ 
  (∃ n : ℤ, 20 ≤ n ∧ n ≤ 43 ∧ x = (n - 23) / 20) := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1334_133410


namespace NUMINAMATH_GPT_base_b_not_divisible_by_5_l1334_133469

theorem base_b_not_divisible_by_5 (b : ℕ) : b = 4 ∨ b = 7 ∨ b = 8 → ¬ (5 ∣ (2 * b^2 * (b - 1))) :=
by
  sorry

end NUMINAMATH_GPT_base_b_not_divisible_by_5_l1334_133469


namespace NUMINAMATH_GPT_steven_card_count_l1334_133428

theorem steven_card_count (num_groups : ℕ) (cards_per_group : ℕ) (h_groups : num_groups = 5) (h_cards : cards_per_group = 6) : num_groups * cards_per_group = 30 := by
  sorry

end NUMINAMATH_GPT_steven_card_count_l1334_133428


namespace NUMINAMATH_GPT_inlet_pipe_rate_l1334_133452

-- Conditions definitions
def tank_capacity : ℕ := 4320
def leak_empty_time : ℕ := 6
def full_empty_time_with_inlet : ℕ := 8

-- Question translated into a theorem
theorem inlet_pipe_rate : 
  (tank_capacity / leak_empty_time) = 720 →
  (tank_capacity / full_empty_time_with_inlet) = 540 →
  ∀ R : ℕ, 
    R - 720 = 540 →
    (R / 60) = 21 :=
by
  intros h_leak h_net R h_R
  sorry

end NUMINAMATH_GPT_inlet_pipe_rate_l1334_133452


namespace NUMINAMATH_GPT_min_binary_questions_to_determine_number_l1334_133455

theorem min_binary_questions_to_determine_number (x : ℕ) (h : 10 ≤ x ∧ x ≤ 19) : 
  ∃ (n : ℕ), n = 3 := 
sorry

end NUMINAMATH_GPT_min_binary_questions_to_determine_number_l1334_133455


namespace NUMINAMATH_GPT_corveus_lack_of_sleep_l1334_133434

def daily_sleep_actual : ℕ := 4
def daily_sleep_recommended : ℕ := 6
def days_in_week : ℕ := 7

theorem corveus_lack_of_sleep : (daily_sleep_recommended - daily_sleep_actual) * days_in_week = 14 := 
by 
  sorry

end NUMINAMATH_GPT_corveus_lack_of_sleep_l1334_133434


namespace NUMINAMATH_GPT_mean_equality_l1334_133472

theorem mean_equality (y z : ℝ)
  (h : (14 + y + z) / 3 = (8 + 15 + 21) / 3)
  (hyz : y = z) :
  y = 15 ∧ z = 15 :=
by sorry

end NUMINAMATH_GPT_mean_equality_l1334_133472


namespace NUMINAMATH_GPT_additional_bureaus_needed_correct_l1334_133464

-- The number of bureaus the company has
def total_bureaus : ℕ := 192

-- The number of offices
def total_offices : ℕ := 36

-- The additional bureaus needed to ensure each office gets an equal number
def additional_bureaus_needed (bureaus : ℕ) (offices : ℕ) : ℕ :=
  let bureaus_per_office := bureaus / offices
  let rounded_bureaus_per_office := bureaus_per_office + if bureaus % offices = 0 then 0 else 1
  let total_bureaus_needed := offices * rounded_bureaus_per_office
  total_bureaus_needed - bureaus

-- Problem Statement: Prove that at least 24 more bureaus are needed
theorem additional_bureaus_needed_correct : 
  additional_bureaus_needed total_bureaus total_offices = 24 := 
by
  sorry

end NUMINAMATH_GPT_additional_bureaus_needed_correct_l1334_133464


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1334_133477

theorem arithmetic_sequence_common_difference
  (a1 a4 : ℤ) (d : ℤ) 
  (h1 : a1 + (a1 + 4 * d) = 10)
  (h2 : a1 + 3 * d = 7) : 
  d = 2 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1334_133477


namespace NUMINAMATH_GPT_simplify_expression_l1334_133486

theorem simplify_expression : 
  (1 / (64^(1/3))^9) * 8^6 = 1 := by 
  have h1 : 64 = 2^6 := by rfl
  have h2 : 8 = 2^3 := by rfl
  sorry

end NUMINAMATH_GPT_simplify_expression_l1334_133486


namespace NUMINAMATH_GPT_sarah_total_height_in_cm_l1334_133446

def sarah_height_in_inches : ℝ := 54
def book_thickness_in_inches : ℝ := 2
def conversion_factor : ℝ := 2.54

def total_height_in_inches : ℝ := sarah_height_in_inches + book_thickness_in_inches
def total_height_in_cm : ℝ := total_height_in_inches * conversion_factor

theorem sarah_total_height_in_cm : total_height_in_cm = 142.2 :=
by
  -- Skip the proof for now
  sorry

end NUMINAMATH_GPT_sarah_total_height_in_cm_l1334_133446


namespace NUMINAMATH_GPT_marco_paints_8_15_in_32_minutes_l1334_133459

-- Define the rates at which Marco and Carla paint
def marco_rate : ℚ := 1 / 60
def combined_rate : ℚ := 1 / 40

-- Define the function to calculate the fraction of the room painted by Marco alone in a given time
def fraction_painted_by_marco (time: ℚ) : ℚ := time * marco_rate

-- State the theorem to prove
theorem marco_paints_8_15_in_32_minutes :
  (marco_rate + (combined_rate - marco_rate) = combined_rate) →
  fraction_painted_by_marco 32 = 8 / 15 := by
  sorry

end NUMINAMATH_GPT_marco_paints_8_15_in_32_minutes_l1334_133459


namespace NUMINAMATH_GPT_solution_range_l1334_133482

-- Define the polynomial function and given values at specific points
def polynomial (x : ℝ) (b : ℝ) : ℝ := x^2 - b * x - 5

-- Given conditions as values of the polynomial at specific points
axiom h1 : ∀ b : ℝ, polynomial (-2) b = 5
axiom h2 : ∀ b : ℝ, polynomial (-1) b = -1
axiom h3 : ∀ b : ℝ, polynomial 4 b = -1
axiom h4 : ∀ b : ℝ, polynomial 5 b = 5

-- The range of solutions for the polynomial equation
theorem solution_range (b : ℝ) : 
  (∃ x : ℝ, -2 < x ∧ x < -1 ∧ polynomial x b = 0) ∨ 
  (∃ x : ℝ, 4 < x ∧ x < 5 ∧ polynomial x b = 0) :=
sorry

end NUMINAMATH_GPT_solution_range_l1334_133482


namespace NUMINAMATH_GPT_smallest_positive_m_l1334_133473

theorem smallest_positive_m (m : ℕ) :
  (∃ (r s : ℤ), 18 * r * s = 252 ∧ m = 18 * (r + s) ∧ r ≠ s) ∧ m > 0 →
  m = 162 := 
sorry

end NUMINAMATH_GPT_smallest_positive_m_l1334_133473


namespace NUMINAMATH_GPT_min_value_fraction_l1334_133414

variable (x y : ℝ)

theorem min_value_fraction (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ (m : ℝ), (∀ z, (z = (1/x) + (9/y)) → z ≥ 16) ∧ ((1/x) + (9/y) = m) :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l1334_133414


namespace NUMINAMATH_GPT_fraction_same_ratio_l1334_133423

theorem fraction_same_ratio (x : ℚ) : 
  (x / (2 / 5)) = (3 / 7) / (6 / 5) ↔ x = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_same_ratio_l1334_133423


namespace NUMINAMATH_GPT_number_of_girls_l1334_133400

theorem number_of_girls (sections : ℕ) (boys_per_section : ℕ) (total_boys : ℕ) (total_sections : ℕ) (boys_sections girls : ℕ) :
  total_boys = 408 → 
  total_sections = 27 → 
  total_boys / total_sections = boys_per_section → 
  boys_sections = total_boys / boys_per_section → 
  total_sections - boys_sections = girls / boys_per_section → 
  girls = 324 :=
by sorry

end NUMINAMATH_GPT_number_of_girls_l1334_133400


namespace NUMINAMATH_GPT_sum_two_angles_greater_third_l1334_133478

-- Definitions of the angles and the largest angle condition
variables {P A B C} -- Points defining the trihedral angle
variables {α β γ : ℝ} -- Angles α, β, γ
variables (h1 : γ ≥ α) (h2 : γ ≥ β)

-- Statement of the theorem
theorem sum_two_angles_greater_third (P A B C : Type*) (α β γ : ℝ)
  (h1 : γ ≥ α) (h2 : γ ≥ β) : α + β > γ :=
sorry  -- Proof is omitted

end NUMINAMATH_GPT_sum_two_angles_greater_third_l1334_133478


namespace NUMINAMATH_GPT_min_y_value_l1334_133442

theorem min_y_value :
  ∃ c : ℝ, ∀ x : ℝ, (5 * x^2 + 20 * x + 25) >= c ∧ (∀ x : ℝ, (5 * x^2 + 20 * x + 25 = c) → x = -2) ∧ c = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_y_value_l1334_133442


namespace NUMINAMATH_GPT_find_p_l1334_133412

theorem find_p (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (5 * p^2 - 2)) : p = 3 :=
sorry

end NUMINAMATH_GPT_find_p_l1334_133412


namespace NUMINAMATH_GPT_dwarfs_truthful_count_l1334_133483

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end NUMINAMATH_GPT_dwarfs_truthful_count_l1334_133483


namespace NUMINAMATH_GPT_rectangular_solid_surface_area_l1334_133415

theorem rectangular_solid_surface_area (a b c : ℕ) (h₁ : Prime a ∨ ∃ p : ℕ, Prime p ∧ a = p + (p + 1))
                                         (h₂ : Prime b ∨ ∃ q : ℕ, Prime q ∧ b = q + (q + 1))
                                         (h₃ : Prime c ∨ ∃ r : ℕ, Prime r ∧ c = r + (r + 1))
                                         (h₄ : a * b * c = 399) :
  2 * (a * b + b * c + c * a) = 422 := 
sorry

end NUMINAMATH_GPT_rectangular_solid_surface_area_l1334_133415


namespace NUMINAMATH_GPT_car_parking_arrangements_l1334_133461

theorem car_parking_arrangements : 
  let grid_size := 6
  let group_size := 3
  let red_car_positions := Nat.choose grid_size group_size
  let arrange_black_cars := Nat.factorial grid_size
  (red_car_positions * arrange_black_cars) = 14400 := 
by
  let grid_size := 6
  let group_size := 3
  let red_car_positions := Nat.choose grid_size group_size
  let arrange_black_cars := Nat.factorial grid_size
  sorry

end NUMINAMATH_GPT_car_parking_arrangements_l1334_133461


namespace NUMINAMATH_GPT_customers_total_l1334_133403

theorem customers_total 
  (initial : ℝ) 
  (added_lunch_rush : ℝ) 
  (added_after_lunch_rush : ℝ) :
  initial = 29.0 →
  added_lunch_rush = 20.0 →
  added_after_lunch_rush = 34.0 →
  initial + added_lunch_rush + added_after_lunch_rush = 83.0 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_customers_total_l1334_133403


namespace NUMINAMATH_GPT_tayzia_tip_l1334_133406

theorem tayzia_tip (haircut_women : ℕ) (haircut_children : ℕ) (num_women : ℕ) (num_children : ℕ) (tip_percentage : ℕ) :
  ((num_women * haircut_women + num_children * haircut_children) * tip_percentage / 100) = 24 :=
by
  -- Given conditions
  let haircut_women := 48
  let haircut_children := 36
  let num_women := 1
  let num_children := 2
  let tip_percentage := 20
  -- Perform the calculations as shown in the solution steps
  sorry

end NUMINAMATH_GPT_tayzia_tip_l1334_133406


namespace NUMINAMATH_GPT_scientific_notation_l1334_133420

theorem scientific_notation : (20160 : ℝ) = 2.016 * 10^4 := 
  sorry

end NUMINAMATH_GPT_scientific_notation_l1334_133420


namespace NUMINAMATH_GPT_intersection_unique_element_l1334_133424

noncomputable def A := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
noncomputable def B (r : ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

theorem intersection_unique_element (r : ℝ) (hr : r > 0) :
  (∃! p : ℝ × ℝ, p ∈ A ∧ p ∈ B r) → (r = 3 ∨ r = 7) :=
sorry

end NUMINAMATH_GPT_intersection_unique_element_l1334_133424


namespace NUMINAMATH_GPT_part1_part2_l1334_133408

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end NUMINAMATH_GPT_part1_part2_l1334_133408


namespace NUMINAMATH_GPT_solve_equation_solutions_count_l1334_133460

open Real

theorem solve_equation_solutions_count :
  (∃ (x_plural : List ℝ), (∀ x ∈ x_plural, 2 * sqrt 2 * (sin (π * x / 4)) ^ 3 = cos (π / 4 * (1 - x)) ∧ 0 ≤ x ∧ x ≤ 2020) ∧ x_plural.length = 505) :=
sorry

end NUMINAMATH_GPT_solve_equation_solutions_count_l1334_133460


namespace NUMINAMATH_GPT_petya_finishes_earlier_than_masha_l1334_133417

variable (t_P t_M t_K : ℕ)

-- Given conditions
def condition1 := t_K = 2 * t_P
def condition2 := t_P + 12 = t_K
def condition3 := t_M = 3 * t_P

-- The proof goal: Petya finishes 24 seconds earlier than Masha
theorem petya_finishes_earlier_than_masha
    (h1 : condition1 t_P t_K)
    (h2 : condition2 t_P t_K)
    (h3 : condition3 t_P t_M) :
    t_M - t_P = 24 := by
  sorry

end NUMINAMATH_GPT_petya_finishes_earlier_than_masha_l1334_133417


namespace NUMINAMATH_GPT_solve_for_x_l1334_133462

theorem solve_for_x (x : ℕ) (h : 2^12 = 64^x) : x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1334_133462


namespace NUMINAMATH_GPT_greatest_prime_factor_3_8_plus_6_7_l1334_133440

theorem greatest_prime_factor_3_8_plus_6_7 : ∃ p, p = 131 ∧ Prime p ∧ ∀ q, Prime q ∧ q ∣ (3^8 + 6^7) → q ≤ 131 :=
by
  sorry


end NUMINAMATH_GPT_greatest_prime_factor_3_8_plus_6_7_l1334_133440


namespace NUMINAMATH_GPT_min_club_members_l1334_133433

theorem min_club_members (n : ℕ) :
  (∀ k : ℕ, k = 8 ∨ k = 9 ∨ k = 11 → n % k = 0) ∧ (n ≥ 300) → n = 792 :=
sorry

end NUMINAMATH_GPT_min_club_members_l1334_133433


namespace NUMINAMATH_GPT_add_to_fraction_eq_l1334_133407

theorem add_to_fraction_eq (n : ℤ) (h : (4 + n) / (7 + n) = 3 / 4) : n = 5 :=
by sorry

end NUMINAMATH_GPT_add_to_fraction_eq_l1334_133407


namespace NUMINAMATH_GPT_height_of_building_l1334_133430

def flagpole_height : ℝ := 18
def flagpole_shadow_length : ℝ := 45

def building_shadow_length : ℝ := 65
def building_height : ℝ := 26

theorem height_of_building
  (hflagpole : flagpole_height / flagpole_shadow_length = building_height / building_shadow_length) :
  building_height = 26 :=
sorry

end NUMINAMATH_GPT_height_of_building_l1334_133430


namespace NUMINAMATH_GPT_width_of_bottom_trapezium_l1334_133444

theorem width_of_bottom_trapezium (top_width : ℝ) (area : ℝ) (depth : ℝ) (bottom_width : ℝ) 
  (h_top_width : top_width = 10)
  (h_area : area = 640)
  (h_depth : depth = 80) :
  bottom_width = 6 :=
by
  -- Problem description: calculating the width of the bottom of the trapezium given the conditions.
  sorry

end NUMINAMATH_GPT_width_of_bottom_trapezium_l1334_133444


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l1334_133432

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 1) : 1 ≤ x :=
by
sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_l1334_133432


namespace NUMINAMATH_GPT_fraction_of_peaches_l1334_133418

-- Define the number of peaches each person has
def Benjy_peaches : ℕ := 5
def Martine_peaches : ℕ := 16
def Gabrielle_peaches : ℕ := 15

-- Condition that Martine has 6 more than twice Benjy's peaches
def Martine_cond : Prop := Martine_peaches = 2 * Benjy_peaches + 6

-- The goal is to prove the fraction of Gabrielle's peaches that Benjy has
theorem fraction_of_peaches :
  Martine_cond → (Benjy_peaches : ℚ) / (Gabrielle_peaches : ℚ) = 1 / 3 :=
by
  -- Assuming the condition holds
  intro h
  rw [Martine_cond] at h
  -- Use the condition directly, since Martine_cond implies Benjy_peaches = 5
  exact sorry

end NUMINAMATH_GPT_fraction_of_peaches_l1334_133418


namespace NUMINAMATH_GPT_solution_y_chemical_A_percentage_l1334_133480

def percent_chemical_A_in_x : ℝ := 0.30
def percent_chemical_A_in_mixture : ℝ := 0.32
def percent_solution_x_in_mixture : ℝ := 0.80
def percent_solution_y_in_mixture : ℝ := 0.20

theorem solution_y_chemical_A_percentage
  (P : ℝ) 
  (h : percent_solution_x_in_mixture * percent_chemical_A_in_x + percent_solution_y_in_mixture * P = percent_chemical_A_in_mixture) :
  P = 0.40 :=
sorry

end NUMINAMATH_GPT_solution_y_chemical_A_percentage_l1334_133480


namespace NUMINAMATH_GPT_horse_goat_sheep_consumption_l1334_133447

theorem horse_goat_sheep_consumption :
  (1 / (1 / (1 : ℝ) + 1 / 2 + 1 / 3)) = 6 / 11 :=
by
  sorry

end NUMINAMATH_GPT_horse_goat_sheep_consumption_l1334_133447
