import Mathlib

namespace NUMINAMATH_GPT_four_digit_number_2010_l1845_184545

theorem four_digit_number_2010 (a b c d : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 1000 ≤ 1000 * a + 100 * b + 10 * c + d ∧
        1000 * a + 100 * b + 10 * c + d < 10000)
  (h_eq : a * (a + b + c + d) * (a^2 + b^2 + c^2 + d^2) * (a^6 + 2 * b^6 + 3 * c^6 + 4 * d^6)
          = 1000 * a + 100 * b + 10 * c + d)
  : 1000 * a + 100 * b + 10 * c + d = 2010 :=
sorry

end NUMINAMATH_GPT_four_digit_number_2010_l1845_184545


namespace NUMINAMATH_GPT_domain_of_function_l1845_184501

theorem domain_of_function (x : ℝ) : (|x - 2| + |x + 2| ≠ 0) := 
sorry

end NUMINAMATH_GPT_domain_of_function_l1845_184501


namespace NUMINAMATH_GPT_pure_imaginary_complex_l1845_184505

theorem pure_imaginary_complex (a : ℝ) (i : ℂ) (h : i * i = -1) (p : (1 + a * i) / (1 - i) = (0 : ℂ) + b * i) :
  a = 1 := 
sorry

end NUMINAMATH_GPT_pure_imaginary_complex_l1845_184505


namespace NUMINAMATH_GPT_hypotenuse_of_right_angle_triangle_l1845_184595

theorem hypotenuse_of_right_angle_triangle {a b c : ℕ} (h1 : a^2 + b^2 = c^2) 
  (h2 : a > 0) (h3 : b > 0) 
  (h4 : a + b + c = (a * b) / 2): 
  c = 10 ∨ c = 13 :=
sorry

end NUMINAMATH_GPT_hypotenuse_of_right_angle_triangle_l1845_184595


namespace NUMINAMATH_GPT_marshmallow_ratio_l1845_184592

theorem marshmallow_ratio:
  (∀ h m b, 
    h = 8 ∧ 
    m = 3 * h ∧ 
    h + m + b = 44
  ) → (1 / 2 = b / m) :=
by
sorry

end NUMINAMATH_GPT_marshmallow_ratio_l1845_184592


namespace NUMINAMATH_GPT_min_value_of_fraction_l1845_184522

theorem min_value_of_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 3 * y = 3) : 
  (3 / x + 2 / y) = 8 :=
sorry

end NUMINAMATH_GPT_min_value_of_fraction_l1845_184522


namespace NUMINAMATH_GPT_B_share_correct_l1845_184534

noncomputable def total_share : ℕ := 120
noncomputable def B_share : ℕ := 20
noncomputable def A_share (x : ℕ) : ℕ := x + 20
noncomputable def C_share (x : ℕ) : ℕ := x + 40

theorem B_share_correct : ∃ x : ℕ, total_share = (A_share x) + x + (C_share x) ∧ x = B_share := by
  sorry

end NUMINAMATH_GPT_B_share_correct_l1845_184534


namespace NUMINAMATH_GPT_gcd_eq_gcd_of_division_l1845_184546

theorem gcd_eq_gcd_of_division (a b q r : ℕ) (h1 : a = b * q + r) (h2 : 0 < r) (h3 : r < b) (h4 : a > b) : 
  Nat.gcd a b = Nat.gcd b r :=
by
  sorry

end NUMINAMATH_GPT_gcd_eq_gcd_of_division_l1845_184546


namespace NUMINAMATH_GPT_initial_oranges_l1845_184589

theorem initial_oranges (left_oranges taken_oranges : ℕ) (h1 : left_oranges = 25) (h2 : taken_oranges = 35) : 
  left_oranges + taken_oranges = 60 := 
by 
  sorry

end NUMINAMATH_GPT_initial_oranges_l1845_184589


namespace NUMINAMATH_GPT_find_number_l1845_184587

theorem find_number (n : ℕ) (h1 : n % 5 = 0) (h2 : 70 ≤ n ∧ n ≤ 90) (h3 : Nat.Prime n) : n = 85 := 
sorry

end NUMINAMATH_GPT_find_number_l1845_184587


namespace NUMINAMATH_GPT_angle_measure_l1845_184574

theorem angle_measure (x : ℝ) (h1 : 180 - x = 6 * (90 - x)) : x = 72 := by
  sorry

end NUMINAMATH_GPT_angle_measure_l1845_184574


namespace NUMINAMATH_GPT_sandy_shopping_l1845_184554

theorem sandy_shopping (T : ℝ) (h : 0.70 * T = 217) : T = 310 := sorry

end NUMINAMATH_GPT_sandy_shopping_l1845_184554


namespace NUMINAMATH_GPT_find_p_l1845_184591

theorem find_p 
  (a : ℝ) (p : ℕ) 
  (h1 : 12345 * 6789 = a * 10^p)
  (h2 : 1 ≤ a) (h3 : a < 10) (h4 : 0 < p) 
  : p = 7 := 
sorry

end NUMINAMATH_GPT_find_p_l1845_184591


namespace NUMINAMATH_GPT_toms_remaining_speed_l1845_184556

-- Defining the constants and conditions
def total_distance : ℝ := 100
def first_leg_distance : ℝ := 50
def first_leg_speed : ℝ := 20
def avg_speed : ℝ := 28.571428571428573

-- Proving Tom's speed during the remaining part of the trip
theorem toms_remaining_speed :
  ∃ (remaining_leg_speed : ℝ),
    (remaining_leg_speed = 50) ∧
    (total_distance = first_leg_distance + 50) ∧
    ((first_leg_distance / first_leg_speed + 50 / remaining_leg_speed) = total_distance / avg_speed) :=
by
  sorry

end NUMINAMATH_GPT_toms_remaining_speed_l1845_184556


namespace NUMINAMATH_GPT_f_2021_l1845_184563

noncomputable def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)
axiom period_f : ∀ x : ℝ, f (x) = f (2 - x)
axiom f_neg1 : f (-1) = 1

theorem f_2021 : f (2021) = -1 :=
by
  sorry

end NUMINAMATH_GPT_f_2021_l1845_184563


namespace NUMINAMATH_GPT_numbers_must_be_equal_l1845_184558

theorem numbers_must_be_equal
  (n : ℕ) (nums : Fin n → ℕ)
  (hn_pos : n = 99)
  (hbound : ∀ i, nums i < 100)
  (hdiv : ∀ (s : Finset (Fin n)) (hs : 2 ≤ s.card), ¬ 100 ∣ s.sum nums) :
  ∀ i j, nums i = nums j := 
sorry

end NUMINAMATH_GPT_numbers_must_be_equal_l1845_184558


namespace NUMINAMATH_GPT_cos_of_sin_given_l1845_184570

theorem cos_of_sin_given (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_cos_of_sin_given_l1845_184570


namespace NUMINAMATH_GPT_collinear_condition_l1845_184550

variable {R : Type*} [LinearOrderedField R]
variable {x1 y1 x2 y2 x3 y3 : R}

theorem collinear_condition : 
  x1 * y2 + x2 * y3 + x3 * y1 = y1 * x2 + y2 * x3 + y3 * x1 →
  ∃ k l m : R, k * (x2 - x1) = l * (y2 - y1) ∧ k * (x3 - x1) = m * (y3 - y1) :=
by
  sorry

end NUMINAMATH_GPT_collinear_condition_l1845_184550


namespace NUMINAMATH_GPT_Q_share_of_profit_l1845_184544

theorem Q_share_of_profit (P Q T : ℕ) (hP : P = 54000) (hQ : Q = 36000) (hT : T = 18000) : Q's_share = 7200 :=
by
  -- Definitions and conditions
  let P := 54000
  let Q := 36000
  let T := 18000
  have P_ratio := 3
  have Q_ratio := 2
  have ratio_sum := P_ratio + Q_ratio
  have Q's_share := (T * Q_ratio) / ratio_sum
  
  -- Q's share of the profit
  sorry

end NUMINAMATH_GPT_Q_share_of_profit_l1845_184544


namespace NUMINAMATH_GPT_correct_inequality_l1845_184529

-- Define the conditions
variables (a b : ℝ)
variable (h : a > 1 ∧ 1 > b ∧ b > 0)

-- State the theorem to prove
theorem correct_inequality (h : a > 1 ∧ 1 > b ∧ b > 0) : 
  (1 / Real.log a) > (1 / Real.log b) :=
sorry

end NUMINAMATH_GPT_correct_inequality_l1845_184529


namespace NUMINAMATH_GPT_compute_expression_l1845_184523

theorem compute_expression :
  21 * 47 + 21 * 53 = 2100 := 
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1845_184523


namespace NUMINAMATH_GPT_max_value_range_l1845_184585

theorem max_value_range (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_deriv : ∀ x, f' x = a * (x - 1) * (x - a))
  (h_max : ∀ x, (x = a → (∀ y, f y ≤ f x))) : 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_max_value_range_l1845_184585


namespace NUMINAMATH_GPT_temperature_difference_l1845_184530

theorem temperature_difference 
  (lowest: ℤ) (highest: ℤ) 
  (h_lowest : lowest = -4)
  (h_highest : highest = 5) :
  highest - lowest = 9 := 
by
  --relies on the correctness of problem and given simplyifying
  sorry

end NUMINAMATH_GPT_temperature_difference_l1845_184530


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l1845_184590

theorem solve_system_of_inequalities 
  (x : ℝ) 
  (h1 : x - 3 * (x - 2) ≥ 4)
  (h2 : (1 + 2 * x) / 3 > x - 1) : 
  x ≤ 1 := 
sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l1845_184590


namespace NUMINAMATH_GPT_quadratic_has_real_root_l1845_184598

theorem quadratic_has_real_root (a b : ℝ) : ¬ (∀ x : ℝ, x^2 + a * x + b ≠ 0) → ∃ x : ℝ, x^2 + a * x + b = 0 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_root_l1845_184598


namespace NUMINAMATH_GPT_am_gm_inequality_l1845_184531

theorem am_gm_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (x + y + z)^2 / 3 ≥ x * Real.sqrt (y * z) + y * Real.sqrt (z * x) + z * Real.sqrt (x * y) := 
by sorry

end NUMINAMATH_GPT_am_gm_inequality_l1845_184531


namespace NUMINAMATH_GPT_base_b_cube_l1845_184525

theorem base_b_cube (b : ℕ) : (b > 4) → (∃ n : ℕ, (b^2 + 4 * b + 4 = n^3)) ↔ (b = 5 ∨ b = 6) :=
by
  sorry

end NUMINAMATH_GPT_base_b_cube_l1845_184525


namespace NUMINAMATH_GPT_more_than_half_millet_on_day_5_l1845_184573

noncomputable def millet_amount (n : ℕ) : ℚ :=
  1 - (3 / 4)^n

theorem more_than_half_millet_on_day_5 : millet_amount 5 > 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_more_than_half_millet_on_day_5_l1845_184573


namespace NUMINAMATH_GPT_average_rst_l1845_184518

theorem average_rst (r s t : ℝ) (h : (5 / 4) * (r + s + t - 2) = 15) : (r + s + t) / 3 = 14 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_rst_l1845_184518


namespace NUMINAMATH_GPT_fifth_month_sale_correct_l1845_184500

noncomputable def fifth_month_sale
  (sales : Fin 4 → ℕ)
  (sixth_month_sale : ℕ)
  (average_sale : ℕ) : ℕ :=
  let total_sales := average_sale * 6
  let known_sales := sales 0 + sales 1 + sales 2 + sales 3 + sixth_month_sale
  total_sales - known_sales

theorem fifth_month_sale_correct :
  ∀ (sales : Fin 4 → ℕ) (sixth_month_sale : ℕ) (average_sale : ℕ),
    sales 0 = 6435 →
    sales 1 = 6927 →
    sales 2 = 6855 →
    sales 3 = 7230 →
    sixth_month_sale = 5591 →
    average_sale = 6600 →
    fifth_month_sale sales sixth_month_sale average_sale = 13562 :=
by
  intros sales sixth_month_sale average_sale h0 h1 h2 h3 h4 h5
  unfold fifth_month_sale
  sorry

end NUMINAMATH_GPT_fifth_month_sale_correct_l1845_184500


namespace NUMINAMATH_GPT_combined_age_in_years_l1845_184567

theorem combined_age_in_years (years : ℕ) (adam_age : ℕ) (tom_age : ℕ) (target_age : ℕ) :
  adam_age = 8 → tom_age = 12 → target_age = 44 → (adam_age + tom_age) + 2 * years = target_age → years = 12 :=
by
  intros h_adam h_tom h_target h_combined
  rw [h_adam, h_tom, h_target] at h_combined
  linarith

end NUMINAMATH_GPT_combined_age_in_years_l1845_184567


namespace NUMINAMATH_GPT_sum_of_tripled_numbers_l1845_184506

theorem sum_of_tripled_numbers (a b S : ℤ) (h : a + b = S) : 3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_tripled_numbers_l1845_184506


namespace NUMINAMATH_GPT_simplify_frac_l1845_184513

theorem simplify_frac (b : ℤ) (hb : b = 2) : (15 * b^4) / (45 * b^3) = 2 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_frac_l1845_184513


namespace NUMINAMATH_GPT_vertices_after_cut_off_four_corners_l1845_184509

-- Definitions for the conditions
def regular_tetrahedron.num_vertices : ℕ := 4

def new_vertices_per_cut : ℕ := 3

def total_vertices_after_cut : ℕ := 
  regular_tetrahedron.num_vertices + regular_tetrahedron.num_vertices * new_vertices_per_cut

-- The theorem to prove the question
theorem vertices_after_cut_off_four_corners :
  total_vertices_after_cut = 12 :=
by
  -- sorry is used to skip the proof steps, as per instructions
  sorry

end NUMINAMATH_GPT_vertices_after_cut_off_four_corners_l1845_184509


namespace NUMINAMATH_GPT_angle_B_in_parallelogram_l1845_184583

theorem angle_B_in_parallelogram (ABCD : Parallelogram) (angle_A angle_C : ℝ) 
  (h : angle_A + angle_C = 100) : 
  angle_B = 130 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_angle_B_in_parallelogram_l1845_184583


namespace NUMINAMATH_GPT_find_x1_value_l1845_184555

theorem find_x1_value (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3) (h2 : x3 ≤ x2) (h3 : x2 ≤ x1) (h4 : x1 ≤ 1) 
  (h_eq : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1 / 3) : 
  x1 = 2 / 3 := 
sorry

end NUMINAMATH_GPT_find_x1_value_l1845_184555


namespace NUMINAMATH_GPT_g_at_neg_1001_l1845_184559

-- Defining the function g and the conditions
def g (x : ℝ) : ℝ := 2.5 * x - 0.5

-- Defining the main theorem to be proved
theorem g_at_neg_1001 : g (-1001) = -2503 := by
  sorry

end NUMINAMATH_GPT_g_at_neg_1001_l1845_184559


namespace NUMINAMATH_GPT_quadrilateral_parallelogram_iff_l1845_184542

variable (a b c d e f MN : ℝ)

-- Define a quadrilateral as a structure with sides and diagonals 
structure Quadrilateral :=
  (a b c d e f : ℝ)

-- Define the condition: sum of squares of diagonals equals sum of squares of sides
def sum_of_squares_condition (q : Quadrilateral) : Prop :=
  q.e ^ 2 + q.f ^ 2 = q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2

-- Define what it means for a quadrilateral to be a parallelogram:
-- Midpoints of the diagonals coincide (MN = 0)
def is_parallelogram (q : Quadrilateral) (MN : ℝ) : Prop :=
  MN = 0

-- Main theorem to prove
theorem quadrilateral_parallelogram_iff (q : Quadrilateral) (MN : ℝ) :
  is_parallelogram q MN ↔ sum_of_squares_condition q :=
sorry

end NUMINAMATH_GPT_quadrilateral_parallelogram_iff_l1845_184542


namespace NUMINAMATH_GPT_abs_eq_neg_iff_nonpositive_l1845_184565

theorem abs_eq_neg_iff_nonpositive (x : ℝ) : |x| = -x ↔ x ≤ 0 := by
  sorry

end NUMINAMATH_GPT_abs_eq_neg_iff_nonpositive_l1845_184565


namespace NUMINAMATH_GPT_problem1_problem2_l1845_184581

theorem problem1 : -24 - (-15) + (-1) + (-15) = -25 := 
by 
  sorry

theorem problem2 : -27 / (3 / 2) * (2 / 3) = -12 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1845_184581


namespace NUMINAMATH_GPT_total_toothpicks_480_l1845_184599

/- Define the number of toothpicks per side -/
def toothpicks_per_side : ℕ := 15

/- Define the number of horizontal lines in the grid -/
def horizontal_lines (sides : ℕ) : ℕ := sides + 1

/- Define the number of vertical lines in the grid -/
def vertical_lines (sides : ℕ) : ℕ := sides + 1

/- Define the total number of toothpicks used -/
def total_toothpicks (sides : ℕ) : ℕ :=
  (horizontal_lines sides * toothpicks_per_side) + (vertical_lines sides * toothpicks_per_side)

/- Theorem statement: Prove that for a grid with 15 toothpicks per side, the total number of toothpicks is 480 -/
theorem total_toothpicks_480 : total_toothpicks 15 = 480 :=
  sorry

end NUMINAMATH_GPT_total_toothpicks_480_l1845_184599


namespace NUMINAMATH_GPT_wendy_chocolates_l1845_184535

theorem wendy_chocolates (h : ℕ) : 
  let chocolates_per_4_hours := 1152
  let chocolates_per_hour := chocolates_per_4_hours / 4
  (chocolates_per_hour * h) = 288 * h :=
by
  sorry

end NUMINAMATH_GPT_wendy_chocolates_l1845_184535


namespace NUMINAMATH_GPT_minimum_doors_to_safety_l1845_184543

-- Definitions in Lean 4 based on the conditions provided
def spaceship (corridors : ℕ) : Prop := corridors = 23

def command_closes (N : ℕ) (corridors : ℕ) : Prop := N ≤ corridors

-- Theorem based on the question and conditions
theorem minimum_doors_to_safety (N : ℕ) (corridors : ℕ)
  (h_corridors : spaceship corridors)
  (h_command : command_closes N corridors) :
  N = 22 :=
sorry

end NUMINAMATH_GPT_minimum_doors_to_safety_l1845_184543


namespace NUMINAMATH_GPT_find_m_l1845_184597

theorem find_m (m : ℝ) (h1 : m > 0) (h2 : (4 - m) / (m - 2) = 2 * m) : 
  m = (3 + Real.sqrt 41) / 4 := by
  sorry

end NUMINAMATH_GPT_find_m_l1845_184597


namespace NUMINAMATH_GPT_ratio_water_to_orange_juice_l1845_184561

variable (O W : ℝ)

-- Conditions:
-- 1. Amount of orange juice is O for both days.
-- 2. Amount of water is W on the first day and 2W on the second day.
-- 3. Price per glass is $0.60 on the first day and $0.40 on the second day.

theorem ratio_water_to_orange_juice 
  (h : (O + W) * 0.60 = (O + 2 * W) * 0.40) : 
  W / O = 1 := 
by 
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_ratio_water_to_orange_juice_l1845_184561


namespace NUMINAMATH_GPT_amount_after_two_years_l1845_184578

theorem amount_after_two_years (P : ℝ) (r1 r2 : ℝ) : 
  P = 64000 → 
  r1 = 0.12 → 
  r2 = 0.15 → 
  (P + P * r1) + (P + P * r1) * r2 = 82432 := by
  sorry

end NUMINAMATH_GPT_amount_after_two_years_l1845_184578


namespace NUMINAMATH_GPT_Catriona_goldfish_count_l1845_184579

theorem Catriona_goldfish_count (G : ℕ) (A : ℕ) (U : ℕ) 
    (h1 : A = G + 4) 
    (h2 : U = 2 * A) 
    (h3 : G + A + U = 44) : G = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Catriona_goldfish_count_l1845_184579


namespace NUMINAMATH_GPT_digit_makes_57A2_divisible_by_9_l1845_184557

theorem digit_makes_57A2_divisible_by_9 (A : ℕ) (h : 0 ≤ A ∧ A ≤ 9) : 
  (5 + 7 + A + 2) % 9 = 0 ↔ A = 4 :=
by
  sorry

end NUMINAMATH_GPT_digit_makes_57A2_divisible_by_9_l1845_184557


namespace NUMINAMATH_GPT_sum_of_squares_l1845_184527

-- Define conditions
def condition1 (a b : ℝ) : Prop := a - b = 6
def condition2 (a b : ℝ) : Prop := a * b = 7

-- Define what we want to prove
def target (a b : ℝ) : Prop := a^2 + b^2 = 50

-- Main theorem stating the required proof
theorem sum_of_squares (a b : ℝ) (h1 : condition1 a b) (h2 : condition2 a b) : target a b :=
by sorry

end NUMINAMATH_GPT_sum_of_squares_l1845_184527


namespace NUMINAMATH_GPT_complex_num_sum_l1845_184562

def is_complex_num (a b : ℝ) (z : ℂ) : Prop :=
  z = a + b * Complex.I

theorem complex_num_sum (a b : ℝ) (z : ℂ) (h : is_complex_num a b z) :
  z = (1 - Complex.I) ^ 2 / (1 + Complex.I) → a + b = -2 :=
by
  sorry

end NUMINAMATH_GPT_complex_num_sum_l1845_184562


namespace NUMINAMATH_GPT_slope_of_arithmetic_sequence_l1845_184552

variable {α : Type*} [LinearOrderedField α]

noncomputable def S (a_1 d n : α) : α := n * a_1 + n * (n-1) / 2 * d

theorem slope_of_arithmetic_sequence (a_1 d n : α) 
  (hS2 : S a_1 d 2 = 10)
  (hS5 : S a_1 d 5 = 55)
  : (a_1 + 2 * d - a_1) / 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_arithmetic_sequence_l1845_184552


namespace NUMINAMATH_GPT_right_triangle_distance_midpoint_l1845_184586

noncomputable def distance_from_F_to_midpoint_DE
  (D E F : ℝ × ℝ)
  (right_triangle : ∃ A B C, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
                    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧
                    D = A ∧ E = B ∧ F = C) 
  (DE : ℝ)
  (DF : ℝ)
  (EF : ℝ)
  : ℝ :=
  if hD : (D.1 - E.1)^2 + (D.2 - E.2)^2 = DE^2 then
    if hF : (D.1 - F.1)^2 + (D.2 - F.2)^2 = DF^2 then
      if hDE : DE = 15 then
        (15 / 2) --distance from F to midpoint of DE
      else
        0 -- This will never be executed since DE = 15 is a given condition
    else
      0 -- This will never be executed since DF = 9 is a given condition
  else
    0 -- This will never be executed since EF = 12 is a given condition

theorem right_triangle_distance_midpoint
  (D E F : ℝ × ℝ)
  (h_triangle : ∃ A B C, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
                    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧
                    D = A ∧ E = B ∧ F = C)
  (hDE : (D.1 - E.1)^2 + (D.2 - E.2)^2 = 15^2)
  (hDF : (D.1 - F.1)^2 + (D.2 - F.2)^2 = 9^2)
  (hEF : (E.1 - F.1)^2 + (E.2 - F.2)^2 = 12^2) :
  distance_from_F_to_midpoint_DE D E F h_triangle 15 9 12 = 7.5 :=
by sorry

end NUMINAMATH_GPT_right_triangle_distance_midpoint_l1845_184586


namespace NUMINAMATH_GPT_mary_hourly_wage_l1845_184516

-- Defining the conditions as given in the problem
def hours_per_day_MWF : ℕ := 9
def hours_per_day_TTh : ℕ := 5
def days_MWF : ℕ := 3
def days_TTh : ℕ := 2
def weekly_earnings : ℕ := 407

-- Total hours worked in a week by Mary
def total_hours_worked : ℕ := (days_MWF * hours_per_day_MWF) + (days_TTh * hours_per_day_TTh)

-- The hourly wage calculation
def hourly_wage : ℕ := weekly_earnings / total_hours_worked

-- The statement to prove
theorem mary_hourly_wage : hourly_wage = 11 := by
  sorry

end NUMINAMATH_GPT_mary_hourly_wage_l1845_184516


namespace NUMINAMATH_GPT_science_and_technology_group_total_count_l1845_184564

theorem science_and_technology_group_total_count 
  (number_of_girls : ℕ)
  (number_of_boys : ℕ)
  (h1 : number_of_girls = 18)
  (h2 : number_of_girls = 2 * number_of_boys - 2)
  : number_of_girls + number_of_boys = 28 := 
by
  sorry

end NUMINAMATH_GPT_science_and_technology_group_total_count_l1845_184564


namespace NUMINAMATH_GPT_price_of_stock_l1845_184507

-- Defining the conditions
def income : ℚ := 650
def dividend_rate : ℚ := 10
def investment : ℚ := 6240

-- Defining the face value calculation from income and dividend rate
def face_value (i : ℚ) (d_rate : ℚ) : ℚ := (i * 100) / d_rate

-- Calculating the price of the stock
def stock_price (inv : ℚ) (fv : ℚ) : ℚ := (inv / fv) * 100

-- Main theorem to be proved
theorem price_of_stock : stock_price investment (face_value income dividend_rate) = 96 := by
  sorry

end NUMINAMATH_GPT_price_of_stock_l1845_184507


namespace NUMINAMATH_GPT_abs_eq_condition_l1845_184588

theorem abs_eq_condition (x : ℝ) (h : |x - 1| + x = 1) : x ≤ 1 :=
by sorry

end NUMINAMATH_GPT_abs_eq_condition_l1845_184588


namespace NUMINAMATH_GPT_figure_50_squares_l1845_184575

-- Define the quadratic function with the given number of squares for figures 0, 1, 2, and 3.
def g (n : ℕ) : ℕ := 2 * n ^ 2 + 4 * n + 2

-- Prove that the number of nonoverlapping unit squares in figure 50 is 5202.
theorem figure_50_squares : g 50 = 5202 := 
by 
  sorry

end NUMINAMATH_GPT_figure_50_squares_l1845_184575


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_extremum_l1845_184519

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 6 * x^2 + (a - 1) * x - 5

theorem necessary_and_sufficient_condition_extremum (a : ℝ) :
  (∃ x, f a x = 0) ↔ -3 < a ∧ a < 4 :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_extremum_l1845_184519


namespace NUMINAMATH_GPT_alicia_candies_problem_l1845_184553

theorem alicia_candies_problem :
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ (n % 9 = 7) ∧ (n % 7 = 5) ∧ n = 124 :=
by
  sorry

end NUMINAMATH_GPT_alicia_candies_problem_l1845_184553


namespace NUMINAMATH_GPT_midpoint_of_intersection_l1845_184571

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 2 * t)

noncomputable def polar_curve (θ : ℝ) : ℝ :=
  2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2)

theorem midpoint_of_intersection :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, 
      A = parametric_line t₁ ∧ 
      B = parametric_line t₂ ∧ 
      (A.1 ^ 2 / 4 + A.2 ^ 2 = 1) ∧ 
      (B.1 ^ 2 / 4 + B.2 ^ 2 = 1)) ∧
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (4 / 5, -1 / 5) :=
sorry

end NUMINAMATH_GPT_midpoint_of_intersection_l1845_184571


namespace NUMINAMATH_GPT_johns_groceries_cost_l1845_184508

noncomputable def calculate_total_cost : ℝ := 
  let bananas_cost := 6 * 2
  let bread_cost := 2 * 3
  let butter_cost := 3 * 5
  let cereal_cost := 4 * (6 - 0.25 * 6)
  let subtotal := bananas_cost + bread_cost + butter_cost + cereal_cost
  if subtotal >= 50 then
    subtotal - 10
  else
    subtotal

-- The statement to prove
theorem johns_groceries_cost : calculate_total_cost = 41 := by
  sorry

end NUMINAMATH_GPT_johns_groceries_cost_l1845_184508


namespace NUMINAMATH_GPT_order_of_abc_l1845_184547

noncomputable def a : ℝ := (1 / 3) * Real.logb 2 (1 / 4)
noncomputable def b : ℝ := 1 - Real.logb 2 3
noncomputable def c : ℝ := Real.cos (5 * Real.pi / 6)

theorem order_of_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_GPT_order_of_abc_l1845_184547


namespace NUMINAMATH_GPT_watch_loss_percentage_l1845_184593

noncomputable def loss_percentage (CP SP_gain : ℝ) : ℝ :=
  100 * (CP - SP_gain) / CP

theorem watch_loss_percentage (CP : ℝ) (SP_gain : ℝ) :
  (SP_gain = CP + 0.04 * CP) →
  (CP = 700) →
  (CP - (SP_gain - 140) = CP * (16 / 100)) :=
by
  intros h_SP_gain h_CP
  rw [h_SP_gain, h_CP]
  simp
  sorry

end NUMINAMATH_GPT_watch_loss_percentage_l1845_184593


namespace NUMINAMATH_GPT_problem_statement_l1845_184502

-- Universal set U is the set of all real numbers
def U : Set ℝ := Set.univ

-- Definition of set M
def M : Set ℝ := { y | ∃ x : ℝ, y = 2 ^ (Real.sqrt (2 * x - x ^ 2 + 3)) }

-- Complement of M in U
def C_U_M : Set ℝ := { y | y < 1 ∨ y > 4 }

-- Definition of set N
def N : Set ℝ := { x | -3 < x ∧ x < 2 }

-- Theorem stating (C_U_M) ∩ N = (-3, 1)
theorem problem_statement : (C_U_M ∩ N) = { x | -3 < x ∧ x < 1 } :=
sorry

end NUMINAMATH_GPT_problem_statement_l1845_184502


namespace NUMINAMATH_GPT_factorize_polynomial_l1845_184539

theorem factorize_polynomial (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3) ^ 2 :=
by sorry

end NUMINAMATH_GPT_factorize_polynomial_l1845_184539


namespace NUMINAMATH_GPT_upper_limit_of_x_l1845_184568

theorem upper_limit_of_x 
  {x : ℤ} 
  (h1 : 0 < x) 
  (h2 : x < 15) 
  (h3 : -1 < x) 
  (h4 : x < 5) 
  (h5 : 0 < x) 
  (h6 : x < 3) 
  (h7 : x + 2 < 4) 
  (h8 : x = 1) : 
  0 < x ∧ x < 2 := 
by 
  sorry

end NUMINAMATH_GPT_upper_limit_of_x_l1845_184568


namespace NUMINAMATH_GPT_david_marks_in_english_l1845_184580

variable (E : ℕ)
variable (marks_in_math : ℕ := 98)
variable (marks_in_physics : ℕ := 99)
variable (marks_in_chemistry : ℕ := 100)
variable (marks_in_biology : ℕ := 98)
variable (average_marks : ℚ := 98.2)
variable (num_subjects : ℕ := 5)

theorem david_marks_in_english 
  (H1 : average_marks = (E + marks_in_math + marks_in_physics + marks_in_chemistry + marks_in_biology) / num_subjects) :
  E = 96 :=
sorry

end NUMINAMATH_GPT_david_marks_in_english_l1845_184580


namespace NUMINAMATH_GPT_count_integer_radii_l1845_184594

theorem count_integer_radii (r : ℕ) (h : r < 150) :
  (∃ n : ℕ, n = 11 ∧ (∀ r, 0 < r ∧ r < 150 → (150 % r = 0)) ∧ (r ≠ 150)) := sorry

end NUMINAMATH_GPT_count_integer_radii_l1845_184594


namespace NUMINAMATH_GPT_sum_geometric_terms_l1845_184584

noncomputable def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^n

theorem sum_geometric_terms (a q : ℝ) :
  a * (1 + q) = 3 → a * (1 + q) * q^2 = 6 → 
  a * (1 + q) * q^6 = 24 :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_sum_geometric_terms_l1845_184584


namespace NUMINAMATH_GPT_gas_volumes_correct_l1845_184514

noncomputable def west_gas_vol_per_capita : ℝ := 21428
noncomputable def non_west_gas_vol : ℝ := 185255
noncomputable def non_west_population : ℝ := 6.9
noncomputable def non_west_gas_vol_per_capita : ℝ := non_west_gas_vol / non_west_population

noncomputable def russia_gas_vol_68_percent : ℝ := 30266.9
noncomputable def russia_gas_vol : ℝ := russia_gas_vol_68_percent * 100 / 68
noncomputable def russia_population : ℝ := 0.147
noncomputable def russia_gas_vol_per_capita : ℝ := russia_gas_vol / russia_population

theorem gas_volumes_correct :
  west_gas_vol_per_capita = 21428 ∧
  non_west_gas_vol_per_capita = 26848.55 ∧
  russia_gas_vol_per_capita = 302790.13 := by
    sorry

end NUMINAMATH_GPT_gas_volumes_correct_l1845_184514


namespace NUMINAMATH_GPT_no_solutions_l1845_184528

theorem no_solutions (x y : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) : ¬ (x^5 = y^2 + 4) :=
by sorry

end NUMINAMATH_GPT_no_solutions_l1845_184528


namespace NUMINAMATH_GPT_curve_intersections_l1845_184560

theorem curve_intersections (m : ℝ) :
  (∃ x y : ℝ, ((x-1)^2 + y^2 = 1) ∧ (y = mx + m) ∧ (y ≠ 0) ∧ (y^2 = 0)) =
  ((m > -Real.sqrt 3 / 3) ∧ (m < 0)) ∨ ((m > 0) ∧ (m < Real.sqrt 3 / 3)) := 
sorry

end NUMINAMATH_GPT_curve_intersections_l1845_184560


namespace NUMINAMATH_GPT_subtraction_example_l1845_184549

theorem subtraction_example : -1 - 3 = -4 := 
  sorry

end NUMINAMATH_GPT_subtraction_example_l1845_184549


namespace NUMINAMATH_GPT_unique_point_graph_eq_l1845_184511

theorem unique_point_graph_eq (c : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + c = 0 → x = -1 ∧ y = 6) ↔ c = 39 :=
sorry

end NUMINAMATH_GPT_unique_point_graph_eq_l1845_184511


namespace NUMINAMATH_GPT_tan_3theta_eq_9_13_l1845_184512

open Real

noncomputable def tan3theta (θ : ℝ) (h : tan θ = 3) : Prop :=
  tan (3 * θ) = (9 / 13)

theorem tan_3theta_eq_9_13 (θ : ℝ) (h : tan θ = 3) : tan3theta θ h :=
by
  sorry

end NUMINAMATH_GPT_tan_3theta_eq_9_13_l1845_184512


namespace NUMINAMATH_GPT_average_weight_of_class_is_61_67_l1845_184540

noncomputable def totalWeightA (avgWeightA : ℝ) (numStudentsA : ℕ) : ℝ := avgWeightA * numStudentsA
noncomputable def totalWeightB (avgWeightB : ℝ) (numStudentsB : ℕ) : ℝ := avgWeightB * numStudentsB
noncomputable def totalWeightClass (totalWeightA : ℝ) (totalWeightB : ℝ) : ℝ := totalWeightA + totalWeightB
noncomputable def totalStudentsClass (numStudentsA : ℕ) (numStudentsB : ℕ) : ℕ := numStudentsA + numStudentsB
noncomputable def averageWeightClass (totalWeightClass : ℝ) (totalStudentsClass : ℕ) : ℝ := totalWeightClass / totalStudentsClass

theorem average_weight_of_class_is_61_67 :
  averageWeightClass (totalWeightClass (totalWeightA 50 50) (totalWeightB 70 70))
    (totalStudentsClass 50 70) = 61.67 := by
  sorry

end NUMINAMATH_GPT_average_weight_of_class_is_61_67_l1845_184540


namespace NUMINAMATH_GPT_milburg_children_count_l1845_184577

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 → grown_ups = 5256 → 
  (total_population - grown_ups) = 2987 :=
by
  intros total_population grown_ups h1 h2
  sorry

end NUMINAMATH_GPT_milburg_children_count_l1845_184577


namespace NUMINAMATH_GPT_suraj_innings_count_l1845_184572

theorem suraj_innings_count
  (A : ℕ := 24)  -- average before the last innings
  (new_average : ℕ := 28)  -- Suraj’s average after the last innings
  (last_score : ℕ := 92)  -- Suraj’s score in the last innings
  (avg_increase : ℕ := 4)  -- the increase in average after the last innings
  (n : ℕ)  -- number of innings before the last one
  (h_avg : A + avg_increase = new_average)  -- A + 4 = 28
  (h_eqn : n * A + last_score = (n + 1) * new_average) :  -- n * 24 + 92 = (n + 1) * 28
  n = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_suraj_innings_count_l1845_184572


namespace NUMINAMATH_GPT_songs_before_camp_l1845_184548

theorem songs_before_camp (total_songs : ℕ) (learned_at_camp : ℕ) (songs_before_camp : ℕ) (h1 : total_songs = 74) (h2 : learned_at_camp = 18) : songs_before_camp = 56 :=
by
  sorry

end NUMINAMATH_GPT_songs_before_camp_l1845_184548


namespace NUMINAMATH_GPT_inequality_nonnegative_reals_l1845_184566

theorem inequality_nonnegative_reals (a b c : ℝ) (h_a : 0 ≤ a) (h_b : 0 ≤ b) (h_c : 0 ≤ c) :
  |(c * a - a * b)| + |(a * b - b * c)| + |(b * c - c * a)| ≤ |(b^2 - c^2)| + |(c^2 - a^2)| + |(a^2 - b^2)| :=
by
  sorry

end NUMINAMATH_GPT_inequality_nonnegative_reals_l1845_184566


namespace NUMINAMATH_GPT_find_parallelogram_base_length_l1845_184576

variable (A h b : ℕ)
variable (parallelogram_area : A = 240)
variable (parallelogram_height : h = 10)
variable (area_formula : A = b * h)

theorem find_parallelogram_base_length : b = 24 :=
by
  have h₁ : A = 240 := parallelogram_area
  have h₂ : h = 10 := parallelogram_height
  have h₃ : A = b * h := area_formula
  sorry

end NUMINAMATH_GPT_find_parallelogram_base_length_l1845_184576


namespace NUMINAMATH_GPT_investment_period_l1845_184520

theorem investment_period (P A : ℝ) (r n t : ℝ)
  (hP : P = 4000)
  (hA : A = 4840.000000000001)
  (hr : r = 0.10)
  (hn : n = 1)
  (hC : A = P * (1 + r / n) ^ (n * t)) :
  t = 2 := by
-- Adding a sorry to skip the actual proof.
sorry

end NUMINAMATH_GPT_investment_period_l1845_184520


namespace NUMINAMATH_GPT_part1_l1845_184537

theorem part1 (a n : ℕ) (hne : a % 2 = 1) : (4 ∣ a^n - 1) → (n % 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_part1_l1845_184537


namespace NUMINAMATH_GPT_similar_triangle_side_length_l1845_184533

theorem similar_triangle_side_length
  (A_1 A_2 : ℕ)
  (area_diff : A_1 - A_2 = 32)
  (area_ratio : A_1 = 9 * A_2)
  (side_small_triangle : ℕ)
  (side_small_triangle_eq : side_small_triangle = 5)
  (side_ratio : ∃ r : ℕ, r = 3) :
  ∃ side_large_triangle : ℕ, side_large_triangle = side_small_triangle * 3 := by
sorry

end NUMINAMATH_GPT_similar_triangle_side_length_l1845_184533


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1845_184510

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : 3 * a 0 + 2 * a 1 = a 2 / 0.5) :
  q = 3 :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1845_184510


namespace NUMINAMATH_GPT_bhanu_house_rent_expenditure_l1845_184541

variable (Income house_rent_expenditure petrol_expenditure remaining_income : ℝ)
variable (h1 : petrol_expenditure = (30 / 100) * Income)
variable (h2 : remaining_income = Income - petrol_expenditure)
variable (h3 : house_rent_expenditure = (20 / 100) * remaining_income)
variable (h4 : petrol_expenditure = 300)

theorem bhanu_house_rent_expenditure :
  house_rent_expenditure = 140 :=
by sorry

end NUMINAMATH_GPT_bhanu_house_rent_expenditure_l1845_184541


namespace NUMINAMATH_GPT_complex_roots_equilateral_l1845_184538

noncomputable def omega : ℂ := -1/2 + Complex.I * Real.sqrt 3 / 2

theorem complex_roots_equilateral (z1 z2 p q : ℂ) (h₁ : z2 = omega * z1) (h₂ : -p = (1 + omega) * z1) (h₃ : q = omega * z1 ^ 2) :
  p^2 / q = 1 + Complex.I * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_complex_roots_equilateral_l1845_184538


namespace NUMINAMATH_GPT_race_position_problem_l1845_184504

theorem race_position_problem 
  (Cara Bruno Emily David Fiona Alan: ℕ)
  (participants : Finset ℕ)
  (participants_card : participants.card = 12)
  (hCara_Bruno : Cara = Bruno - 3)
  (hEmily_David : Emily = David + 1)
  (hAlan_Bruno : Alan = Bruno + 4)
  (hDavid_Fiona : David = Fiona + 3)
  (hFiona_Cara : Fiona = Cara - 2)
  (hBruno : Bruno = 9)
  (Cara_in_participants : Cara ∈ participants)
  (Bruno_in_participants : Bruno ∈ participants)
  (Emily_in_participants : Emily ∈ participants)
  (David_in_participants : David ∈ participants)
  (Fiona_in_participants : Fiona ∈ participants)
  (Alan_in_participants : Alan ∈ participants)
  : David = 7 := 
sorry

end NUMINAMATH_GPT_race_position_problem_l1845_184504


namespace NUMINAMATH_GPT_num_integers_satisfying_inequality_l1845_184536

theorem num_integers_satisfying_inequality :
  ∃ (x : ℕ), ∀ (y: ℤ), (-3 ≤ 3 * y + 2 → 3 * y + 2 ≤ 8) ↔ 4 = x :=
by
  sorry

end NUMINAMATH_GPT_num_integers_satisfying_inequality_l1845_184536


namespace NUMINAMATH_GPT_average_salary_8800_l1845_184515

theorem average_salary_8800 
  (average_salary_start : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_salary : ℝ)
  (avg_specific_months : ℝ)
  (jan_salary_rate : average_salary_start * 4 = total_salary)
  (may_salary_rate : total_salary - salary_jan = total_salary - 3300)
  (final_salary_rate : total_salary - salary_jan + salary_may = 35200)
  (specific_avg_calculation : 35200 / 4 = avg_specific_months)
  : avg_specific_months = 8800 :=
sorry -- Proof steps will be filled in later

end NUMINAMATH_GPT_average_salary_8800_l1845_184515


namespace NUMINAMATH_GPT_final_number_not_perfect_square_l1845_184526

theorem final_number_not_perfect_square :
  (∃ final_number : ℕ, 
    ∀ a b : ℕ, a ∈ Finset.range 101 ∧ b ∈ Finset.range 101 ∧ a ≠ b → 
    gcd (a^2 + b^2 + 2) (a^2 * b^2 + 3) = final_number) →
  ∀ final_number : ℕ, ¬ ∃ k : ℕ, final_number = k ^ 2 :=
sorry

end NUMINAMATH_GPT_final_number_not_perfect_square_l1845_184526


namespace NUMINAMATH_GPT_max_geometric_sequence_sum_l1845_184569

theorem max_geometric_sequence_sum (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a * b * c = 216) (h4 : ∃ r : ℕ, b = a * r ∧ c = b * r) : 
  a + b + c ≤ 43 :=
sorry

end NUMINAMATH_GPT_max_geometric_sequence_sum_l1845_184569


namespace NUMINAMATH_GPT_inequality_satisfaction_l1845_184596

theorem inequality_satisfaction (a b : ℝ) (h : a < 0) : (a < b) ∧ (a^2 + b^2 > 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_satisfaction_l1845_184596


namespace NUMINAMATH_GPT_polynomial_value_l1845_184551

theorem polynomial_value (x y : ℝ) (h : x - 2 * y + 3 = 8) : x - 2 * y = 5 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l1845_184551


namespace NUMINAMATH_GPT_probability_both_girls_l1845_184517

def club_probability (total_members girls chosen_members : ℕ) : ℚ :=
  (Nat.choose girls chosen_members : ℚ) / (Nat.choose total_members chosen_members : ℚ)

theorem probability_both_girls (H1 : total_members = 12) (H2 : girls = 7) (H3 : chosen_members = 2) :
  club_probability 12 7 2 = 7 / 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_both_girls_l1845_184517


namespace NUMINAMATH_GPT_hannah_spent_65_l1845_184582

-- Definitions based on the conditions
def sweatshirts_count : ℕ := 3
def t_shirts_count : ℕ := 2
def sweatshirt_cost : ℕ := 15
def t_shirt_cost : ℕ := 10

-- The total amount spent
def total_spent : ℕ := sweatshirts_count * sweatshirt_cost + t_shirts_count * t_shirt_cost

-- The theorem stating the problem
theorem hannah_spent_65 : total_spent = 65 :=
by
  sorry

end NUMINAMATH_GPT_hannah_spent_65_l1845_184582


namespace NUMINAMATH_GPT_lily_patch_cover_entire_lake_l1845_184532

noncomputable def days_to_cover_half (initial_days : ℕ) := 33

theorem lily_patch_cover_entire_lake (initial_days : ℕ) (h : days_to_cover_half initial_days = 33) :
  initial_days + 1 = 34 :=
by
  sorry

end NUMINAMATH_GPT_lily_patch_cover_entire_lake_l1845_184532


namespace NUMINAMATH_GPT_squirrel_nuts_l1845_184524

theorem squirrel_nuts :
  ∃ (a b c d : ℕ), 103 ≤ a ∧ 103 ≤ b ∧ 103 ≤ c ∧ 103 ≤ d ∧
                   a ≥ b ∧ a ≥ c ∧ a ≥ d ∧
                   a + b + c + d = 2020 ∧
                   b + c = 1277 ∧
                   a = 640 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_squirrel_nuts_l1845_184524


namespace NUMINAMATH_GPT_unique_non_congruent_rectangle_with_conditions_l1845_184503

theorem unique_non_congruent_rectangle_with_conditions :
  ∃! (w h : ℕ), 2 * (w + h) = 80 ∧ w * h = 400 :=
by
  sorry

end NUMINAMATH_GPT_unique_non_congruent_rectangle_with_conditions_l1845_184503


namespace NUMINAMATH_GPT_intercept_sum_modulo_l1845_184521

theorem intercept_sum_modulo (x_0 y_0 : ℤ) (h1 : 0 ≤ x_0) (h2 : x_0 < 17) (h3 : 0 ≤ y_0) (h4 : y_0 < 17)
                       (hx : 5 * x_0 ≡ 2 [ZMOD 17])
                       (hy : 3 * y_0 ≡ 15 [ZMOD 17]) :
    x_0 + y_0 = 19 := 
by
  sorry

end NUMINAMATH_GPT_intercept_sum_modulo_l1845_184521
