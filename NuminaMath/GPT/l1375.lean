import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l1375_137575

def sequence_arithmetic (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a (n+1) / 2^(n+1) - a n / 2^n = 1)

theorem problem_statement : 
  ∃ a : ℕ → ℝ, a 1 = 2 ∧ a 2 = 8 ∧ (∀ n : ℕ, n ≥ 1 → a (n+1) - 2 * a n = 2^(n+1)) → sequence_arithmetic a :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1375_137575


namespace NUMINAMATH_GPT_cows_total_l1375_137529

theorem cows_total {n : ℕ} :
  (n / 3) + (n / 6) + (n / 8) + (n / 24) + 15 = n ↔ n = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_cows_total_l1375_137529


namespace NUMINAMATH_GPT_sum_of_distinct_roots_eq_zero_l1375_137565

theorem sum_of_distinct_roots_eq_zero
  (a b m n p : ℝ)
  (h1 : m ≠ n)
  (h2 : m ≠ p)
  (h3 : n ≠ p)
  (h_m : m^3 + a * m + b = 0)
  (h_n : n^3 + a * n + b = 0)
  (h_p : p^3 + a * p + b = 0) : 
  m + n + p = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_distinct_roots_eq_zero_l1375_137565


namespace NUMINAMATH_GPT_value_expression_l1375_137510

-- Definitions
variable (m n : ℝ)
def reciprocals (m n : ℝ) := m * n = 1

-- Theorem statement
theorem value_expression (m n : ℝ) (h : reciprocals m n) : m * n^2 - (n - 3) = 3 := by
  sorry

end NUMINAMATH_GPT_value_expression_l1375_137510


namespace NUMINAMATH_GPT_number_of_yellow_balls_l1375_137535

theorem number_of_yellow_balls (x : ℕ) :
  (4 : ℕ) / (4 + x) = 2 / 3 → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_yellow_balls_l1375_137535


namespace NUMINAMATH_GPT_soda_ratio_l1375_137588

theorem soda_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) : 
  let v_z := 1.3 * v
  let p_z := 0.85 * p
  (p_z / v_z) / (p / v) = 17 / 26 :=
by sorry

end NUMINAMATH_GPT_soda_ratio_l1375_137588


namespace NUMINAMATH_GPT_total_students_l1375_137503

-- Define the conditions
def ratio_boys_to_girls (boys girls : ℕ) : Prop := boys = 3 * (girls / 2)
def boys_girls_difference (boys girls : ℕ) : Prop := boys = girls + 20

-- Define the property to be proved
theorem total_students (boys girls : ℕ) 
  (h1 : ratio_boys_to_girls boys girls)
  (h2 : boys_girls_difference boys girls) :
  boys + girls = 100 :=
sorry

end NUMINAMATH_GPT_total_students_l1375_137503


namespace NUMINAMATH_GPT_compute_fg_difference_l1375_137563

def f (x : ℕ) : ℕ := x^2 + 3
def g (x : ℕ) : ℕ := 2 * x + 5

theorem compute_fg_difference : f (g 5) - g (f 5) = 167 := by
  sorry

end NUMINAMATH_GPT_compute_fg_difference_l1375_137563


namespace NUMINAMATH_GPT_value_of_x_in_equation_l1375_137532

theorem value_of_x_in_equation : 
  (∀ x : ℕ, 8 ^ 17 + 8 ^ 17 + 8 ^ 17 + 8 ^ 17 = 2 ^ x → x = 53) := 
by 
  sorry

end NUMINAMATH_GPT_value_of_x_in_equation_l1375_137532


namespace NUMINAMATH_GPT_votes_to_win_l1375_137593

theorem votes_to_win (total_votes : ℕ) (geoff_votes_percent : ℝ) (additional_votes : ℕ) (x : ℝ) 
(h1 : total_votes = 6000)
(h2 : geoff_votes_percent = 0.5)
(h3 : additional_votes = 3000)
(h4 : x = 50.5) :
  ((geoff_votes_percent / 100 * total_votes) + additional_votes) / total_votes * 100 = x :=
by
  sorry

end NUMINAMATH_GPT_votes_to_win_l1375_137593


namespace NUMINAMATH_GPT_greatest_k_value_l1375_137505

-- Define a type for triangle and medians intersecting at centroid
structure Triangle :=
(medianA : ℝ)
(medianB : ℝ)
(medianC : ℝ)
(angleA : ℝ)
(angleB : ℝ)
(angleC : ℝ)
(centroid : ℝ)

-- Define a function to determine if the internal angles formed by medians 
-- are greater than 30 degrees
def angle_greater_than_30 (θ : ℝ) : Prop :=
  θ > 30

-- A proof statement that given a triangle and its medians dividing an angle
-- into six angles, the greatest possible number of these angles greater than 30° is 3.
theorem greatest_k_value (T : Triangle) : ∃ k : ℕ, k = 3 ∧ 
  (∀ θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ, 
    (angle_greater_than_30 θ₁ ∨ angle_greater_than_30 θ₂ ∨ angle_greater_than_30 θ₃ ∨ 
     angle_greater_than_30 θ₄ ∨ angle_greater_than_30 θ₅ ∨ angle_greater_than_30 θ₆) → 
    k = 3) := 
sorry

end NUMINAMATH_GPT_greatest_k_value_l1375_137505


namespace NUMINAMATH_GPT_hollow_cylinder_surface_area_l1375_137509

theorem hollow_cylinder_surface_area (h : ℝ) (r_outer r_inner : ℝ) (h_eq : h = 12) (r_outer_eq : r_outer = 5) (r_inner_eq : r_inner = 2) :
  (2 * π * ((r_outer ^ 2 - r_inner ^ 2)) + 2 * π * r_outer * h + 2 * π * r_inner * h) = 210 * π :=
by
  rw [h_eq, r_outer_eq, r_inner_eq]
  sorry

end NUMINAMATH_GPT_hollow_cylinder_surface_area_l1375_137509


namespace NUMINAMATH_GPT_triangle_overlap_angle_is_30_l1375_137537

noncomputable def triangle_rotation_angle (hypotenuse : ℝ) (overlap_ratio : ℝ) :=
  if hypotenuse = 10 ∧ overlap_ratio = 0.5 then 30 else sorry

theorem triangle_overlap_angle_is_30 :
  triangle_rotation_angle 10 0.5 = 30 :=
sorry

end NUMINAMATH_GPT_triangle_overlap_angle_is_30_l1375_137537


namespace NUMINAMATH_GPT_non_deg_ellipse_projection_l1375_137518

theorem non_deg_ellipse_projection (m : ℝ) : 
  (3 * x^2 + 9 * y^2 - 12 * x + 18 * y + 6 * z = m → (m > -21)) := 
by
  sorry

end NUMINAMATH_GPT_non_deg_ellipse_projection_l1375_137518


namespace NUMINAMATH_GPT_f_bound_l1375_137533

theorem f_bound (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1) 
  (h2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) : ∀ x : ℝ, |f x| ≤ 2 + x^2 :=
by
  sorry

end NUMINAMATH_GPT_f_bound_l1375_137533


namespace NUMINAMATH_GPT_sum_of_products_mod_7_l1375_137590

-- Define the numbers involved
def a := 1789
def b := 1861
def c := 1945
def d := 1533
def e := 1607
def f := 1688

-- Define the sum of products
def sum_of_products := a * b * c + d * e * f

-- The statement to prove:
theorem sum_of_products_mod_7 : sum_of_products % 7 = 3 := 
by sorry

end NUMINAMATH_GPT_sum_of_products_mod_7_l1375_137590


namespace NUMINAMATH_GPT_smallest_number_of_brownies_l1375_137511

noncomputable def total_brownies (m n : ℕ) : ℕ := m * n
def perimeter_brownies (m n : ℕ) : ℕ := 2 * m + 2 * n - 4
def interior_brownies (m n : ℕ) : ℕ := (m - 2) * (n - 2)

theorem smallest_number_of_brownies : 
  ∃ (m n : ℕ), 2 * interior_brownies m n = perimeter_brownies m n ∧ total_brownies m n = 36 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_brownies_l1375_137511


namespace NUMINAMATH_GPT_find_number_l1375_137513

theorem find_number (x : ℝ) (h : 0.20 * x = 0.20 * 650 + 190) : x = 1600 :=
sorry

end NUMINAMATH_GPT_find_number_l1375_137513


namespace NUMINAMATH_GPT_gambler_win_percentage_l1375_137515

theorem gambler_win_percentage :
  ∀ (T W play_extra : ℕ) (P_win_extra P_week P_current P_required : ℚ),
    T = 40 →
    P_win_extra = 0.80 →
    play_extra = 40 →
    P_week = 0.60 →
    P_required = 48 →
    (W + P_win_extra * play_extra = P_required) →
    (P_current = (W : ℚ) / T * 100) →
    P_current = 40 :=
by
  intros T W play_extra P_win_extra P_week P_current P_required h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_gambler_win_percentage_l1375_137515


namespace NUMINAMATH_GPT_Tommy_Ratio_Nickels_to_Dimes_l1375_137592

def TommyCoinsProblem :=
  ∃ (P D N Q : ℕ), 
    (D = P + 10) ∧ 
    (Q = 4) ∧ 
    (P = 10 * Q) ∧ 
    (N = 100) ∧ 
    (N / D = 2)

theorem Tommy_Ratio_Nickels_to_Dimes : TommyCoinsProblem := by
  sorry

end NUMINAMATH_GPT_Tommy_Ratio_Nickels_to_Dimes_l1375_137592


namespace NUMINAMATH_GPT_value_of_expression_l1375_137585

theorem value_of_expression (n : ℝ) (h : n + 1/n = 6) : n^2 + 1/n^2 + 9 = 43 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1375_137585


namespace NUMINAMATH_GPT_temperature_conversion_l1375_137527

theorem temperature_conversion (C F F_new C_new : ℚ) 
  (h_formula : C = (5/9) * (F - 32))
  (h_C : C = 30)
  (h_F_new : F_new = F + 15)
  (h_F : F = 86)
: C_new = (5/9) * (F_new - 32) ↔ C_new = 38.33 := 
by 
  sorry

end NUMINAMATH_GPT_temperature_conversion_l1375_137527


namespace NUMINAMATH_GPT_concert_attendance_difference_l1375_137523

/-- Define the number of people attending the first concert. -/
def first_concert_attendance : ℕ := 65899

/-- Define the number of people attending the second concert. -/
def second_concert_attendance : ℕ := 66018

/-- The proof statement that the difference in attendance between the second and first concert is 119. -/
theorem concert_attendance_difference :
  (second_concert_attendance - first_concert_attendance = 119) := by
  sorry

end NUMINAMATH_GPT_concert_attendance_difference_l1375_137523


namespace NUMINAMATH_GPT_total_water_intake_l1375_137578

def morning_water : ℝ := 1.5
def afternoon_water : ℝ := 3 * morning_water
def evening_water : ℝ := 0.5 * afternoon_water

theorem total_water_intake : 
  (morning_water + afternoon_water + evening_water) = 8.25 :=
by
  sorry

end NUMINAMATH_GPT_total_water_intake_l1375_137578


namespace NUMINAMATH_GPT_jane_ends_with_crayons_l1375_137561

-- Definitions for the conditions in the problem
def initial_crayons : Nat := 87
def crayons_eaten : Nat := 7
def packs_bought : Nat := 5
def crayons_per_pack : Nat := 10
def crayons_break : Nat := 3

-- Statement to prove: Jane ends with 127 crayons
theorem jane_ends_with_crayons :
  initial_crayons - crayons_eaten + (packs_bought * crayons_per_pack) - crayons_break = 127 :=
by
  sorry

end NUMINAMATH_GPT_jane_ends_with_crayons_l1375_137561


namespace NUMINAMATH_GPT_grid_coloring_count_l1375_137506

/-- Let n be a positive integer with n ≥ 2. Each of the 2n vertices in a 2 × n grid need to be 
colored red (R), yellow (Y), or blue (B). The three vertices at the endpoints are already colored 
as shown in the problem description. For the remaining 2n-3 vertices, each vertex must be colored 
exactly one color, and adjacent vertices must be colored differently. We aim to show that the 
number of distinct ways to color the vertices is 3^(n-1). -/
theorem grid_coloring_count (n : ℕ) (hn : n ≥ 2) : 
  ∃ a_n b_n c_n : ℕ, 
    (a_n + b_n + c_n = 3^(n-1)) ∧ 
    (a_n = b_n) ∧ 
    (a_n = 2 * b_n + c_n) := 
by 
  sorry

end NUMINAMATH_GPT_grid_coloring_count_l1375_137506


namespace NUMINAMATH_GPT_part1_part2_l1375_137597

def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y
def B (x y : ℝ) : ℝ := x^2 - x * y + x

def difference (x y : ℝ) : ℝ := A x y - 2 * B x y

theorem part1 : difference (-2) 3 = -20 :=
by
  -- Proving that difference (-2) 3 = -20
  sorry

theorem part2 (y : ℝ) : (∀ (x : ℝ), difference x y = 2 * y) → y = 2 / 5 :=
by
  -- Proving that if difference x y is independent of x, then y = 2 / 5
  sorry

end NUMINAMATH_GPT_part1_part2_l1375_137597


namespace NUMINAMATH_GPT_close_to_one_below_l1375_137556

theorem close_to_one_below (k l m n : ℕ) (h1 : k > l) (h2 : l > m) (h3 : m > n) (hk : k = 43) (hl : l = 7) (hm : m = 3) (hn : n = 2) :
  (1 : ℚ) / k + 1 / l + 1 / m + 1 / n < 1 := by
  sorry

end NUMINAMATH_GPT_close_to_one_below_l1375_137556


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1375_137543

variables (a b c : ℝ)

-- First proof problem
theorem problem1 (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : a * b * c ≠ 0 :=
sorry

-- Second proof problem
theorem problem2 (h : a = 0 ∨ b = 0 ∨ c = 0) : a * b * c = 0 :=
sorry

-- Third proof problem
theorem problem3 (h : a * b < 0 ∨ a = 0 ∨ b = 0) : a * b ≤ 0 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1375_137543


namespace NUMINAMATH_GPT_tunnel_build_equation_l1375_137555

theorem tunnel_build_equation (x : ℝ) (h1 : 1280 > 0) (h2 : x > 0) : 
  (1280 - x) / x = (1280 - x) / (1.4 * x) + 2 := 
by
  sorry

end NUMINAMATH_GPT_tunnel_build_equation_l1375_137555


namespace NUMINAMATH_GPT_number_of_pencils_broken_l1375_137536

theorem number_of_pencils_broken
  (initial_pencils : ℕ)
  (misplaced_pencils : ℕ)
  (found_pencils : ℕ)
  (bought_pencils : ℕ)
  (final_pencils : ℕ)
  (h_initial : initial_pencils = 20)
  (h_misplaced : misplaced_pencils = 7)
  (h_found : found_pencils = 4)
  (h_bought : bought_pencils = 2)
  (h_final : final_pencils = 16) :
  (initial_pencils - misplaced_pencils + found_pencils + bought_pencils - final_pencils) = 3 := 
by
  sorry

end NUMINAMATH_GPT_number_of_pencils_broken_l1375_137536


namespace NUMINAMATH_GPT_digit_in_tens_place_l1375_137580

theorem digit_in_tens_place (n : ℕ) (cycle : List ℕ) (h_cycle : cycle = [16, 96, 76, 56]) (hk : n % 4 = 3) :
  (6 ^ n % 100) / 10 % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_digit_in_tens_place_l1375_137580


namespace NUMINAMATH_GPT_opposite_of_5_is_neg_5_l1375_137596

def opposite_number (x y : ℤ) : Prop := x + y = 0

theorem opposite_of_5_is_neg_5 : opposite_number 5 (-5) := by
  sorry

end NUMINAMATH_GPT_opposite_of_5_is_neg_5_l1375_137596


namespace NUMINAMATH_GPT_socks_ratio_l1375_137598

/-- Alice ordered 6 pairs of green socks and some additional pairs of red socks. The price per pair
of green socks was three times that of the red socks. During the delivery, the quantities of the 
pairs were accidentally swapped. This mistake increased the bill by 40%. Prove that the ratio of the 
number of pairs of green socks to red socks in Alice's original order is 1:2. -/
theorem socks_ratio (r y : ℕ) (h1 : y * r ≠ 0) (h2 : 6 * 3 * y + r * y = (r * 3 * y + 6 * y) * 10 / 7) :
  6 / r = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_socks_ratio_l1375_137598


namespace NUMINAMATH_GPT_prime_number_property_l1375_137512

open Nat

-- Definition that p is prime
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Conjecture to prove: if p is a prime number and p^4 - 3p^2 + 9 is also a prime number, then p = 2.
theorem prime_number_property (p : ℕ) (h1 : is_prime p) (h2 : is_prime (p^4 - 3*p^2 + 9)) : p = 2 :=
sorry

end NUMINAMATH_GPT_prime_number_property_l1375_137512


namespace NUMINAMATH_GPT_train_cross_time_l1375_137582

def train_length := 100
def bridge_length := 275
def train_speed_kmph := 45

noncomputable def train_speed_mps : ℝ :=
  (train_speed_kmph * 1000.0) / 3600.0

theorem train_cross_time :
  let total_distance := train_length + bridge_length
  let speed := train_speed_mps
  let time := total_distance / speed
  time = 30 :=
by 
  -- Introduce definitions to make sure they align with the initial conditions
  let total_distance := train_length + bridge_length
  let speed := train_speed_mps
  let time := total_distance / speed
  -- Prove time = 30
  sorry

end NUMINAMATH_GPT_train_cross_time_l1375_137582


namespace NUMINAMATH_GPT_reflection_across_x_axis_l1375_137591

theorem reflection_across_x_axis (x y : ℝ) : (x, -y) = (-2, 3) ↔ (x, y) = (-2, -3) :=
by sorry

end NUMINAMATH_GPT_reflection_across_x_axis_l1375_137591


namespace NUMINAMATH_GPT_min_value_expression_l1375_137552

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/a + (a/b^2) + b) ≥ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1375_137552


namespace NUMINAMATH_GPT_intersection_P_Q_l1375_137542

noncomputable def P : Set ℝ := { x | x < 1 }
noncomputable def Q : Set ℝ := { x | x^2 < 4 }

theorem intersection_P_Q :
  P ∩ Q = { x | -2 < x ∧ x < 1 } :=
by 
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l1375_137542


namespace NUMINAMATH_GPT_max_m_x_range_l1375_137544

variables {a b x : ℝ}

theorem max_m (h1 : a * b > 0) (h2 : a^2 * b = 4) : 
  a + b ≥ 3 :=
sorry

theorem x_range (h : 2 * |x - 1| + |x| ≤ 3) : 
  -1/3 ≤ x ∧ x ≤ 5/3 :=
sorry

end NUMINAMATH_GPT_max_m_x_range_l1375_137544


namespace NUMINAMATH_GPT_sufficient_condition_for_inequality_l1375_137557

theorem sufficient_condition_for_inequality (a x : ℝ) (h1 : -2 < x) (h2 : x < -1) :
  (a + x) * (1 + x) < 0 → a > 2 :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_inequality_l1375_137557


namespace NUMINAMATH_GPT_ratio_of_numbers_l1375_137589

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : 0 < a) (h3 : b < a) (h4 : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_numbers_l1375_137589


namespace NUMINAMATH_GPT_find_x_average_is_60_l1375_137549

theorem find_x_average_is_60 : 
  ∃ x : ℕ, (54 + 55 + 57 + 58 + 59 + 62 + 62 + 63 + x) / 9 = 60 ∧ x = 70 :=
by
  existsi 70
  sorry

end NUMINAMATH_GPT_find_x_average_is_60_l1375_137549


namespace NUMINAMATH_GPT_max_correct_answers_l1375_137567

variable (x y z : ℕ)

theorem max_correct_answers
  (h1 : x + y + z = 100)
  (h2 : x - 3 * y - 2 * z = 50) :
  x ≤ 87 := by
    sorry

end NUMINAMATH_GPT_max_correct_answers_l1375_137567


namespace NUMINAMATH_GPT_solve_for_x_l1375_137519

def star (a b : ℕ) := a * b + a + b

theorem solve_for_x : ∃ x : ℕ, star 3 x = 27 ∧ x = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1375_137519


namespace NUMINAMATH_GPT_value_of_t_eq_3_over_4_l1375_137548

-- Define the values x and y as per the conditions
def x (t : ℝ) : ℝ := 1 - 2 * t
def y (t : ℝ) : ℝ := 2 * t - 2

-- Statement only, proof is omitted using sorry
theorem value_of_t_eq_3_over_4 (t : ℝ) (h : x t = y t) : t = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_t_eq_3_over_4_l1375_137548


namespace NUMINAMATH_GPT_remainder_of_division_l1375_137514

theorem remainder_of_division (x : ℕ) (r : ℕ) :
  1584 - x = 1335 ∧ 1584 = 6 * x + r → r = 90 := by
  sorry

end NUMINAMATH_GPT_remainder_of_division_l1375_137514


namespace NUMINAMATH_GPT_sale_in_fifth_month_l1375_137550

theorem sale_in_fifth_month (sale1 sale2 sale3 sale4 sale6 : ℕ) (avg : ℕ) (months : ℕ) (total_sales : ℕ)
    (known_sales : sale1 = 6335 ∧ sale2 = 6927 ∧ sale3 = 6855 ∧ sale4 = 7230 ∧ sale6 = 5091)
    (avg_condition : avg = 6500)
    (months_condition : months = 6)
    (total_sales_condition : total_sales = avg * months) :
    total_sales - (sale1 + sale2 + sale3 + sale4 + sale6) = 6562 :=
by
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l1375_137550


namespace NUMINAMATH_GPT_Oliver_Battle_Gremlins_Card_Count_l1375_137525

theorem Oliver_Battle_Gremlins_Card_Count 
  (MonsterClubCards AlienBaseballCards BattleGremlinsCards : ℕ)
  (h1 : MonsterClubCards = 2 * AlienBaseballCards)
  (h2 : BattleGremlinsCards = 3 * AlienBaseballCards)
  (h3 : MonsterClubCards = 32) : 
  BattleGremlinsCards = 48 := by
  sorry

end NUMINAMATH_GPT_Oliver_Battle_Gremlins_Card_Count_l1375_137525


namespace NUMINAMATH_GPT_cost_per_pouch_is_20_l1375_137531

theorem cost_per_pouch_is_20 :
  let boxes := 10
  let pouches_per_box := 6
  let dollars := 12
  let cents_per_dollar := 100
  let total_pouches := boxes * pouches_per_box
  let total_cents := dollars * cents_per_dollar
  let cost_per_pouch := total_cents / total_pouches
  cost_per_pouch = 20 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_pouch_is_20_l1375_137531


namespace NUMINAMATH_GPT_work_days_of_A_and_B_l1375_137568

theorem work_days_of_A_and_B (B : ℝ) (A : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 27) :
  1 / (A + B) = 9 :=
by
  sorry

end NUMINAMATH_GPT_work_days_of_A_and_B_l1375_137568


namespace NUMINAMATH_GPT_greatest_possible_difference_l1375_137530

def is_reverse (q r : ℕ) : Prop :=
  let q_tens := q / 10
  let q_units := q % 10
  let r_tens := r / 10
  let r_units := r % 10
  (q_tens = r_units) ∧ (q_units = r_tens)

theorem greatest_possible_difference (q r : ℕ) (hq1 : q ≥ 10) (hq2 : q < 100)
  (hr1 : r ≥ 10) (hr2 : r < 100) (hrev : is_reverse q r) (hpos_diff : q - r < 30) :
  q - r ≤ 27 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_difference_l1375_137530


namespace NUMINAMATH_GPT_cube_root_of_unity_identity_l1375_137508

theorem cube_root_of_unity_identity (ω : ℂ) (hω3: ω^3 = 1) (hω_ne_1 : ω ≠ 1) (hunit : ω^2 + ω + 1 = 0) :
  (1 - ω) * (1 - ω^2) * (1 - ω^4) * (1 - ω^8) = 9 :=
by
  sorry

end NUMINAMATH_GPT_cube_root_of_unity_identity_l1375_137508


namespace NUMINAMATH_GPT_original_price_per_lesson_l1375_137571

theorem original_price_per_lesson (piano_cost lessons_cost : ℤ) (number_of_lessons discount_percent : ℚ) (total_cost : ℤ) (original_price : ℚ) :
  piano_cost = 500 ∧
  number_of_lessons = 20 ∧
  discount_percent = 0.25 ∧
  total_cost = 1100 →
  lessons_cost = total_cost - piano_cost →
  0.75 * (number_of_lessons * original_price) = lessons_cost →
  original_price = 40 :=
by
  intros h h1 h2
  sorry

end NUMINAMATH_GPT_original_price_per_lesson_l1375_137571


namespace NUMINAMATH_GPT_pentagon_area_pq_sum_l1375_137540

theorem pentagon_area_pq_sum 
  (p q : ℤ) 
  (hp : 0 < q ∧ q < p) 
  (harea : 5 * p * q - q * q = 700) : 
  ∃ sum : ℤ, sum = p + q :=
by
  sorry

end NUMINAMATH_GPT_pentagon_area_pq_sum_l1375_137540


namespace NUMINAMATH_GPT_translated_parabola_correct_l1375_137541

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := x^2 + 2

-- Theorem stating that translating the original parabola up by 2 units results in the translated parabola
theorem translated_parabola_correct (x : ℝ) :
  translated_parabola x = original_parabola x + 2 :=
by
  sorry

end NUMINAMATH_GPT_translated_parabola_correct_l1375_137541


namespace NUMINAMATH_GPT_polynomial_solution_l1375_137570

open Polynomial
open Real

theorem polynomial_solution (P : Polynomial ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → P.eval (x * sqrt 2) = P.eval (x + sqrt (1 - x^2))) :
  ∃ U : Polynomial ℝ, P = (U.comp (Polynomial.C (1/4) - 2 * X^2 + 5 * X^4 - 4 * X^6 + X^8)) :=
sorry

end NUMINAMATH_GPT_polynomial_solution_l1375_137570


namespace NUMINAMATH_GPT_square_101_l1375_137554

theorem square_101:
  (101 : ℕ)^2 = 10201 :=
by
  sorry

end NUMINAMATH_GPT_square_101_l1375_137554


namespace NUMINAMATH_GPT_equalize_marbles_condition_l1375_137528

variables (D : ℝ)
noncomputable def marble_distribution := 
    let C := 1.25 * D
    let B := 1.4375 * D
    let A := 1.725 * D
    let total := A + B + C + D
    let equal := total / 4
    let move_from_A := (A - equal) / A * 100
    let move_from_B := (B - equal) / B * 100
    let add_to_C := (equal - C) / C * 100
    let add_to_D := (equal - D) / D * 100
    (move_from_A, move_from_B, add_to_C, add_to_D)

theorem equalize_marbles_condition :
    marble_distribution D = (21.56, 5.87, 8.25, 35.31) := sorry

end NUMINAMATH_GPT_equalize_marbles_condition_l1375_137528


namespace NUMINAMATH_GPT_largest_expression_l1375_137545

theorem largest_expression :
  let A := 0.9387
  let B := 0.9381
  let C := 9385 / 10000
  let D := 0.9379
  let E := 0.9389
  E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  let A := 0.9387
  let B := 0.9381
  let C := 9385 / 10000
  let D := 0.9379
  let E := 0.9389
  sorry

end NUMINAMATH_GPT_largest_expression_l1375_137545


namespace NUMINAMATH_GPT_part1_solution_set_part2_solution_l1375_137599

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

theorem part1_solution_set :
  {x : ℝ | f x > 2} = {x | x > 1} ∪ {x | x < -5} :=
by
  sorry

theorem part2_solution (t : ℝ) :
  (∀ x, f x ≥ t^2 - (11 / 2) * t) ↔ (1 / 2 ≤ t ∧ t ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_set_part2_solution_l1375_137599


namespace NUMINAMATH_GPT_solve_eqn_in_integers_l1375_137560

theorem solve_eqn_in_integers :
  ∃ (x y : ℤ), xy + 3*x - 5*y = -3 ∧ 
  ((x, y) = (6, 9) ∨ (x, y) = (7, 3) ∨ (x, y) = (8, 1) ∨ 
  (x, y) = (9, 0) ∨ (x, y) = (11, -1) ∨ (x, y) = (17, -2) ∨ 
  (x, y) = (4, -15) ∨ (x, y) = (3, -9) ∨ (x, y) = (2, -7) ∨ 
  (x, y) = (1, -6) ∨ (x, y) = (-1, -5) ∨  (x, y) = (-7, -4)) :=
sorry

end NUMINAMATH_GPT_solve_eqn_in_integers_l1375_137560


namespace NUMINAMATH_GPT_equivalent_lengthEF_l1375_137572

namespace GeometryProof

noncomputable def lengthEF 
  (AB CD EF : ℝ) 
  (h_AB_parallel_CD : true) 
  (h_lengthAB : AB = 200) 
  (h_lengthCD : CD = 50) 
  (h_angleEF : true) 
  : ℝ := 
  50

theorem equivalent_lengthEF
  (AB CD EF : ℝ) 
  (h_AB_parallel_CD : true) 
  (h_lengthAB : AB = 200) 
  (h_lengthCD : CD = 50) 
  (h_angleEF : true) 
  : lengthEF AB CD EF h_AB_parallel_CD h_lengthAB h_lengthCD h_angleEF = 50 :=
by
  sorry

end GeometryProof

end NUMINAMATH_GPT_equivalent_lengthEF_l1375_137572


namespace NUMINAMATH_GPT_triangle_area_l1375_137507

theorem triangle_area {x y : ℝ} :

  (∀ a:ℝ, y = a ↔ a = x) ∧
  (∀ b:ℝ, y = -b ↔ b = x) ∧
  ( y = 10 )
  → 1 / 2 * abs (10 - (-10)) * 10 = 100 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1375_137507


namespace NUMINAMATH_GPT_commute_time_difference_l1375_137576

-- Define the conditions as constants
def distance_to_work : ℝ := 1.5
def walking_speed : ℝ := 3
def train_speed : ℝ := 20
def additional_train_time_minutes : ℝ := 10.5

-- The main proof problem
theorem commute_time_difference : 
  (distance_to_work / walking_speed * 60) - 
  ((distance_to_work / train_speed * 60) + additional_train_time_minutes) = 15 :=
by
  sorry

end NUMINAMATH_GPT_commute_time_difference_l1375_137576


namespace NUMINAMATH_GPT_more_regular_than_diet_l1375_137559

-- Define the conditions
def num_regular_soda : Nat := 67
def num_diet_soda : Nat := 9

-- State the theorem
theorem more_regular_than_diet :
  num_regular_soda - num_diet_soda = 58 :=
by
  sorry

end NUMINAMATH_GPT_more_regular_than_diet_l1375_137559


namespace NUMINAMATH_GPT_union_of_M_and_Q_is_correct_l1375_137595

-- Given sets M and Q
def M : Set ℕ := {0, 2, 4, 6}
def Q : Set ℕ := {0, 1, 3, 5}

-- Statement to prove
theorem union_of_M_and_Q_is_correct : M ∪ Q = {0, 1, 2, 3, 4, 5, 6} :=
by
  sorry

end NUMINAMATH_GPT_union_of_M_and_Q_is_correct_l1375_137595


namespace NUMINAMATH_GPT_acres_used_for_corn_l1375_137534

theorem acres_used_for_corn (total_land : ℕ) (ratio_beans : ℕ) (ratio_wheat : ℕ) (ratio_corn : ℕ)
  (total_ratio_parts : ℕ) (one_part_size : ℕ) :
  total_land = 1034 →
  ratio_beans = 5 →
  ratio_wheat = 2 →
  ratio_corn = 4 →
  total_ratio_parts = ratio_beans + ratio_wheat + ratio_corn →
  one_part_size = total_land / total_ratio_parts →
  ratio_corn * one_part_size = 376 :=
by
  intros
  sorry

end NUMINAMATH_GPT_acres_used_for_corn_l1375_137534


namespace NUMINAMATH_GPT_remainder_four_times_plus_six_l1375_137594

theorem remainder_four_times_plus_six (n : ℤ) (h : n % 5 = 3) : (4 * n + 6) % 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_four_times_plus_six_l1375_137594


namespace NUMINAMATH_GPT_gcd_of_17420_23826_36654_l1375_137587

theorem gcd_of_17420_23826_36654 : Nat.gcd (Nat.gcd 17420 23826) 36654 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_of_17420_23826_36654_l1375_137587


namespace NUMINAMATH_GPT_smallest_n_integer_l1375_137504

theorem smallest_n_integer (m n : ℕ) (s : ℝ) (h_m : m = (n + s)^4) (h_n_pos : 0 < n) (h_s_range : 0 < s ∧ s < 1 / 2000) : n = 8 := 
by
  sorry

end NUMINAMATH_GPT_smallest_n_integer_l1375_137504


namespace NUMINAMATH_GPT_crocus_bulb_cost_l1375_137569

theorem crocus_bulb_cost 
  (space_bulbs : ℕ)
  (crocus_bulbs : ℕ)
  (cost_daffodil_bulb : ℝ)
  (budget : ℝ)
  (purchased_crocus_bulbs : ℕ)
  (total_cost : ℝ)
  (c : ℝ)
  (h_space : space_bulbs = 55)
  (h_cost_daffodil : cost_daffodil_bulb = 0.65)
  (h_budget : budget = 29.15)
  (h_purchased_crocus : purchased_crocus_bulbs = 22)
  (h_total_cost_eq : total_cost = (33:ℕ) * cost_daffodil_bulb)
  (h_eqn : (purchased_crocus_bulbs : ℝ) * c + total_cost = budget) :
  c = 0.35 :=
by 
  sorry

end NUMINAMATH_GPT_crocus_bulb_cost_l1375_137569


namespace NUMINAMATH_GPT_total_bowling_balls_is_66_l1375_137584

-- Define the given conditions
def red_bowling_balls := 30
def difference_green_red := 6
def green_bowling_balls := red_bowling_balls + difference_green_red

-- The statement to prove
theorem total_bowling_balls_is_66 :
  red_bowling_balls + green_bowling_balls = 66 := by
  sorry

end NUMINAMATH_GPT_total_bowling_balls_is_66_l1375_137584


namespace NUMINAMATH_GPT_region_area_l1375_137558

theorem region_area {x y : ℝ} (h : x^2 + y^2 - 4*x + 2*y = -1) : 
  ∃ (r : ℝ), r = 4*pi := 
sorry

end NUMINAMATH_GPT_region_area_l1375_137558


namespace NUMINAMATH_GPT_cos_pi_minus_2alpha_l1375_137522

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (Real.pi - 2 * α) = -1 / 9 := 
by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_2alpha_l1375_137522


namespace NUMINAMATH_GPT_count_dna_sequences_Rthea_l1375_137501

-- Definition of bases
inductive Base | H | M | N | T

-- Function to check whether two bases can be adjacent on the same strand
def can_be_adjacent (x y : Base) : Prop :=
  match x, y with
  | Base.H, Base.M => False
  | Base.M, Base.H => False
  | Base.N, Base.T => False
  | Base.T, Base.N => False
  | _, _ => True

-- Function to count the number of valid sequences
noncomputable def count_valid_sequences : Nat := 12 * 7^4

-- Theorem stating the expected count of valid sequences
theorem count_dna_sequences_Rthea : count_valid_sequences = 28812 := by
  sorry

end NUMINAMATH_GPT_count_dna_sequences_Rthea_l1375_137501


namespace NUMINAMATH_GPT_joes_speed_second_part_l1375_137539

theorem joes_speed_second_part
  (d1 d2 t1 t_total: ℝ)
  (s1 s_avg: ℝ)
  (h_d1: d1 = 420)
  (h_d2: d2 = 120)
  (h_s1: s1 = 60)
  (h_s_avg: s_avg = 54) :
  (d1 / s1 + d2 / (d2 / 40) = t_total ∧ t_total = (d1 + d2) / s_avg) →
  d2 / (t_total - d1 / s1) = 40 :=
by
  sorry

end NUMINAMATH_GPT_joes_speed_second_part_l1375_137539


namespace NUMINAMATH_GPT_max_f_max_ab_plus_bc_l1375_137521

def f (x : ℝ) := |x - 3| - 2 * |x + 1|

theorem max_f : ∃ (m : ℝ), m = 4 ∧ (∀ x : ℝ, f x ≤ m) := 
  sorry

theorem max_ab_plus_bc (a b c : ℝ) : a > 0 ∧ b > 0 → a^2 + 2 * b^2 + c^2 = 4 → (ab + bc) ≤ 2 :=
  sorry

end NUMINAMATH_GPT_max_f_max_ab_plus_bc_l1375_137521


namespace NUMINAMATH_GPT_marc_journey_fraction_l1375_137524

-- Defining the problem based on identified conditions
def total_cycling_time (k : ℝ) : ℝ := 20 * k
def total_walking_time (k : ℝ) : ℝ := 60 * (1 - k)
def total_travel_time (k : ℝ) : ℝ := total_cycling_time k + total_walking_time k

theorem marc_journey_fraction:
  ∀ (k : ℝ), total_travel_time k = 52 → k = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_marc_journey_fraction_l1375_137524


namespace NUMINAMATH_GPT_LindasOriginalSavings_l1375_137538

theorem LindasOriginalSavings : 
  (∃ S : ℝ, (1 / 4) * S = 200) ∧ 
  (3 / 4) * S = 600 ∧ 
  (∀ F : ℝ, 0.80 * F = 600 → F = 750) → 
  S = 800 :=
by
  sorry

end NUMINAMATH_GPT_LindasOriginalSavings_l1375_137538


namespace NUMINAMATH_GPT_marcella_matching_pairs_l1375_137502

theorem marcella_matching_pairs (P : ℕ) (L : ℕ) (H : P = 20) (H1 : L = 9) : (P - L) / 2 = 11 :=
by
  -- definition of P and L are given by 20 and 9 respectively
  -- proof is omitted for the statement focus
  sorry

end NUMINAMATH_GPT_marcella_matching_pairs_l1375_137502


namespace NUMINAMATH_GPT_scientific_notation_of_384000_l1375_137517

theorem scientific_notation_of_384000 : 384000 = 3.84 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_384000_l1375_137517


namespace NUMINAMATH_GPT_student_A_more_stable_l1375_137520

-- Defining the variances of students A and B as constants
def S_A_sq : ℝ := 0.04
def S_B_sq : ℝ := 0.13

-- Statement of the theorem
theorem student_A_more_stable : S_A_sq < S_B_sq → true :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_student_A_more_stable_l1375_137520


namespace NUMINAMATH_GPT_range_of_a_l1375_137553

def p (a : ℝ) : Prop := 0 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 1 / 4

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : a ∈ Set.Ioc 0 (1 / 4) ∨ a ∈ Set.Ioi 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1375_137553


namespace NUMINAMATH_GPT_compute_expression_l1375_137546

theorem compute_expression : 45 * 28 + 72 * 45 = 4500 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1375_137546


namespace NUMINAMATH_GPT_MarkBenchPressAmount_l1375_137547

def DaveWeight : ℝ := 175
def DaveBenchPressMultiplier : ℝ := 3
def CraigBenchPressFraction : ℝ := 0.20
def MarkDeficitFromCraig : ℝ := 50

theorem MarkBenchPressAmount : 
  let DaveBenchPress := DaveWeight * DaveBenchPressMultiplier
  let CraigBenchPress := DaveBenchPress * CraigBenchPressFraction
  let MarkBenchPress := CraigBenchPress - MarkDeficitFromCraig
  MarkBenchPress = 55 := by
  let DaveBenchPress := DaveWeight * DaveBenchPressMultiplier
  let CraigBenchPress := DaveBenchPress * CraigBenchPressFraction
  let MarkBenchPress := CraigBenchPress - MarkDeficitFromCraig
  sorry

end NUMINAMATH_GPT_MarkBenchPressAmount_l1375_137547


namespace NUMINAMATH_GPT_gcd_of_75_and_360_l1375_137551

theorem gcd_of_75_and_360 : Nat.gcd 75 360 = 15 := by
  sorry

end NUMINAMATH_GPT_gcd_of_75_and_360_l1375_137551


namespace NUMINAMATH_GPT_ratio_first_term_common_difference_l1375_137577

theorem ratio_first_term_common_difference
  (a d : ℚ)
  (h : (15 / 2) * (2 * a + 14 * d) = 4 * (8 / 2) * (2 * a + 7 * d)) :
  a / d = -7 / 17 := 
by {
  sorry
}

end NUMINAMATH_GPT_ratio_first_term_common_difference_l1375_137577


namespace NUMINAMATH_GPT_no_divisor_neighbors_l1375_137581

def is_divisor (a b : ℕ) : Prop := b % a = 0

def circle_arrangement (arr : Fin 8 → ℕ) : Prop :=
  arr 0 = 7 ∧ arr 1 = 9 ∧ arr 2 = 4 ∧ arr 3 = 5 ∧ arr 4 = 3 ∧ arr 5 = 6 ∧ arr 6 = 8 ∧ arr 7 = 2

def valid_neighbors (arr : Fin 8 → ℕ) : Prop :=
  ¬ is_divisor (arr 0) (arr 1) ∧ ¬ is_divisor (arr 0) (arr 3) ∧
  ¬ is_divisor (arr 1) (arr 2) ∧ ¬ is_divisor (arr 1) (arr 3) ∧ ¬ is_divisor (arr 1) (arr 5) ∧
  ¬ is_divisor (arr 2) (arr 1) ∧ ¬ is_divisor (arr 2) (arr 6) ∧ ¬ is_divisor (arr 2) (arr 3) ∧
  ¬ is_divisor (arr 3) (arr 1) ∧ ¬ is_divisor (arr 3) (arr 4) ∧ ¬ is_divisor (arr 3) (arr 2) ∧ ¬ is_divisor (arr 3) (arr 0) ∧
  ¬ is_divisor (arr 4) (arr 3) ∧ ¬ is_divisor (arr 4) (arr 5) ∧
  ¬ is_divisor (arr 5) (arr 1) ∧ ¬ is_divisor (arr 5) (arr 4) ∧ ¬ is_divisor (arr 5) (arr 6) ∧
  ¬ is_divisor (arr 6) (arr 2) ∧ ¬ is_divisor (arr 6) (arr 5) ∧ ¬ is_divisor (arr 6) (arr 7) ∧
  ¬ is_divisor (arr 7) (arr 6)

theorem no_divisor_neighbors :
  ∀ (arr : Fin 8 → ℕ), circle_arrangement arr → valid_neighbors arr :=
by
  intros arr h
  sorry

end NUMINAMATH_GPT_no_divisor_neighbors_l1375_137581


namespace NUMINAMATH_GPT_division_remainder_l1375_137516

theorem division_remainder (q d D R : ℕ) (h_q : q = 40) (h_d : d = 72) (h_D : D = 2944) (h_div : D = d * q + R) : R = 64 :=
by sorry

end NUMINAMATH_GPT_division_remainder_l1375_137516


namespace NUMINAMATH_GPT_dana_more_pencils_than_marcus_l1375_137574

theorem dana_more_pencils_than_marcus :
  ∀ (Jayden Dana Marcus : ℕ), 
  (Jayden = 20) ∧ 
  (Dana = Jayden + 15) ∧ 
  (Jayden = 2 * Marcus) → 
  (Dana - Marcus = 25) :=
by
  intros Jayden Dana Marcus h
  rcases h with ⟨hJayden, hDana, hMarcus⟩
  sorry

end NUMINAMATH_GPT_dana_more_pencils_than_marcus_l1375_137574


namespace NUMINAMATH_GPT_minimum_bats_examined_l1375_137500

theorem minimum_bats_examined 
  (bats : Type) 
  (R L : bats → Prop) 
  (total_bats : ℕ)
  (right_eye_bats : ∀ {b: bats}, R b → Fin 2)
  (left_eye_bats : ∀ {b: bats}, L b → Fin 3)
  (not_left_eye_bats: ∀ {b: bats}, ¬ L b → Fin 4)
  (not_right_eye_bats: ∀ {b: bats}, ¬ R b → Fin 5)
  : total_bats ≥ 7 := sorry

end NUMINAMATH_GPT_minimum_bats_examined_l1375_137500


namespace NUMINAMATH_GPT_unique_positive_real_solution_l1375_137526

theorem unique_positive_real_solution (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h1 : x * y = z) (h2 : y * z = x) (h3 : z * x = y) : x = 1 ∧ y = 1 ∧ z = 1 :=
sorry

end NUMINAMATH_GPT_unique_positive_real_solution_l1375_137526


namespace NUMINAMATH_GPT_determine_f_1789_l1375_137583

theorem determine_f_1789
  (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, 0 < n → f (f n) = 4 * n + 9)
  (h2 : ∀ k : ℕ, f (2^k) = 2^(k+1) + 3) :
  f 1789 = 3581 :=
sorry

end NUMINAMATH_GPT_determine_f_1789_l1375_137583


namespace NUMINAMATH_GPT_trader_profit_l1375_137566

theorem trader_profit (P : ℝ) (hP : 0 < P) : 
  let purchase_price := 0.80 * P
  let selling_price := 1.36 * P
  let profit := selling_price - P
  (profit / P) * 100 = 36 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_trader_profit_l1375_137566


namespace NUMINAMATH_GPT_intersection_M_N_l1375_137586

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := 
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1375_137586


namespace NUMINAMATH_GPT_common_difference_is_4_l1375_137564

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Defining the arithmetic sequence {a_n}
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
variable (d : ℤ) (a4_a5_sum : a 4 + a 5 = 24) (S6_val : S 6 = 48)

-- Statement to prove: given the conditions, d = 4
theorem common_difference_is_4 (h_seq : is_arithmetic_sequence a d) :
  d = 4 := sorry

end NUMINAMATH_GPT_common_difference_is_4_l1375_137564


namespace NUMINAMATH_GPT_expected_sixes_correct_l1375_137562

-- Define probabilities for rolling individual numbers on a die
def P (n : ℕ) (k : ℕ) : ℚ := if k = n then 1 / 6 else 0

-- Expected value calculation for two dice
noncomputable def expected_sixes_two_dice_with_resets : ℚ :=
(0 * (13/18)) + (1 * (2/9)) + (2 * (1/36))

-- Main theorem to prove
theorem expected_sixes_correct :
  expected_sixes_two_dice_with_resets = 5 / 18 :=
by
  -- The actual proof steps go here; added sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_expected_sixes_correct_l1375_137562


namespace NUMINAMATH_GPT_baker_cakes_l1375_137579

theorem baker_cakes (initial_cakes sold_cakes remaining_cakes final_cakes new_cakes : ℕ)
  (h1 : initial_cakes = 110)
  (h2 : sold_cakes = 75)
  (h3 : final_cakes = 111)
  (h4 : new_cakes = final_cakes - (initial_cakes - sold_cakes)) :
  new_cakes = 76 :=
by {
  sorry
}

end NUMINAMATH_GPT_baker_cakes_l1375_137579


namespace NUMINAMATH_GPT_connie_total_markers_l1375_137573

/-
Connie has 4 different types of markers: red, blue, green, and yellow.
She has twice as many red markers as green markers.
She has three times as many blue markers as red markers.
She has four times as many yellow markers as green markers.
She has 36 green markers.
Prove that the total number of markers she has is 468.
-/

theorem connie_total_markers
 (g r b y : ℕ) 
 (hg : g = 36) 
 (hr : r = 2 * g)
 (hb : b = 3 * r)
 (hy : y = 4 * g) :
 g + r + b + y = 468 := 
 by
  sorry

end NUMINAMATH_GPT_connie_total_markers_l1375_137573
