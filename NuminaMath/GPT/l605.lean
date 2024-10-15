import Mathlib

namespace NUMINAMATH_GPT_sum_digits_18_to_21_l605_60570

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_18_to_21 :
  sum_of_digits 18 + sum_of_digits 19 + sum_of_digits 20 + sum_of_digits 21 = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_digits_18_to_21_l605_60570


namespace NUMINAMATH_GPT_max_mean_weight_BC_l605_60575

theorem max_mean_weight_BC
  (A_n B_n C_n : ℕ)
  (w_A w_B : ℕ)
  (mean_A mean_B mean_AB mean_AC : ℤ)
  (hA : mean_A = 30)
  (hB : mean_B = 55)
  (hAB : mean_AB = 35)
  (hAC : mean_AC = 32)
  (h1 : mean_A * A_n + mean_B * B_n = mean_AB * (A_n + B_n))
  (h2 : mean_A * A_n + mean_AC * C_n = mean_AC * (A_n + C_n)) :
  ∃ n : ℕ, n ≤ 62 ∧ (mean_B * B_n + w_A * C_n) / (B_n + C_n) = n := 
sorry

end NUMINAMATH_GPT_max_mean_weight_BC_l605_60575


namespace NUMINAMATH_GPT_remainder_is_correct_l605_60542

def dividend : ℕ := 725
def divisor : ℕ := 36
def quotient : ℕ := 20

theorem remainder_is_correct : ∃ (remainder : ℕ), dividend = (divisor * quotient) + remainder ∧ remainder = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_is_correct_l605_60542


namespace NUMINAMATH_GPT_geom_seq_a12_value_l605_60576

-- Define the geometric sequence as a function from natural numbers to real numbers
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geom_seq_a12_value (a : ℕ → ℝ) 
  (H_geom : geom_seq a) 
  (H_7_9 : a 7 * a 9 = 4) 
  (H_4 : a 4 = 1) : 
  a 12 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_geom_seq_a12_value_l605_60576


namespace NUMINAMATH_GPT_three_digit_integers_congruent_to_2_mod_4_l605_60580

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end NUMINAMATH_GPT_three_digit_integers_congruent_to_2_mod_4_l605_60580


namespace NUMINAMATH_GPT_triangle_perimeter_l605_60566

theorem triangle_perimeter (m : ℝ) (a b : ℝ) (h1 : 3 ^ 2 - 3 * (m + 1) + 2 * m = 0)
  (h2 : a ^ 2 - (m + 1) * a + 2 * m = 0)
  (h3 : b ^ 2 - (m + 1) * b + 2 * m = 0)
  (h4 : a = 3 ∨ b = 3)
  (h5 : a ≠ b ∨ a = b)
  (hAB : a ≠ b ∨ a = b) :
  (∀ s₁ s₂ : ℝ, s₁ = a ∨ s₁ = b ∧ s₂ = a ∨ s₂ = b ∧ s₁ ≠ s₂ → s₁ + s₁ + s₂ = 10 ∨ s₁ + s₁ + s₂ = 11) ∨
  (∀ s₁ s₂ : ℝ, s₁ = a ∨ s₁ = b ∧ s₂ = a ∨ s₂ = b ∧ s₁ = s₂ → b + b + a = 10 ∨ b + b + a = 11) := by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l605_60566


namespace NUMINAMATH_GPT_area_of_square_l605_60557

-- Define the conditions given in the problem
def radius_circle := 7 -- radius of each circle in inches

def diameter_circle := 2 * radius_circle -- diameter of each circle

def side_length_square := 2 * diameter_circle -- side length of the square

-- State the theorem we want to prove
theorem area_of_square : side_length_square ^ 2 = 784 := 
by
  sorry

end NUMINAMATH_GPT_area_of_square_l605_60557


namespace NUMINAMATH_GPT_appointment_duration_l605_60547

-- Define the given conditions
def total_workday_hours : ℕ := 8
def permits_per_hour : ℕ := 50
def total_permits : ℕ := 100
def stamping_time : ℕ := total_permits / permits_per_hour
def appointment_time : ℕ := (total_workday_hours - stamping_time) / 2

-- State the theorem and ignore the proof part by adding sorry
theorem appointment_duration : appointment_time = 3 := by
  -- skipping the proof steps
  sorry

end NUMINAMATH_GPT_appointment_duration_l605_60547


namespace NUMINAMATH_GPT_smallest_integer_larger_than_expression_l605_60546

theorem smallest_integer_larger_than_expression :
  ∃ n : ℤ, n = 248 ∧ (↑n > ((Real.sqrt 5 + Real.sqrt 3) ^ 4 : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_larger_than_expression_l605_60546


namespace NUMINAMATH_GPT_coat_total_selling_price_l605_60520

theorem coat_total_selling_price :
  let original_price := 120
  let discount_percent := 30
  let tax_percent := 8
  let discount_amount := (discount_percent / 100) * original_price
  let sale_price := original_price - discount_amount
  let tax_amount := (tax_percent / 100) * sale_price
  let total_selling_price := sale_price + tax_amount
  total_selling_price = 90.72 :=
by
  sorry

end NUMINAMATH_GPT_coat_total_selling_price_l605_60520


namespace NUMINAMATH_GPT_curve_intersects_x_axis_at_4_over_5_l605_60518

-- Define the function for the curve
noncomputable def curve (x : ℝ) : ℝ :=
  (3 * x - 1) * (Real.sqrt (9 * x ^ 2 - 6 * x + 5) + 1) +
  (2 * x - 3) * (Real.sqrt (4 * x ^ 2 - 12 * x + 13) + 1)

-- Prove that curve(x) = 0 when x = 4 / 5
theorem curve_intersects_x_axis_at_4_over_5 :
  curve (4 / 5) = 0 :=
by
  sorry

end NUMINAMATH_GPT_curve_intersects_x_axis_at_4_over_5_l605_60518


namespace NUMINAMATH_GPT_smallest_possible_number_of_students_l605_60584

theorem smallest_possible_number_of_students :
  ∃ n : ℕ, (n % 200 = 0) ∧ (∀ m : ℕ, (m < n → 
    75 * m ≤ 100 * n) ∧
    (∃ a b c : ℕ, a = m / 4 ∧ b = a / 10 ∧ 
    ∃ y z : ℕ, y = 3 * z ∧ (y + z - b = a) ∧ y * c = n / 4)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_number_of_students_l605_60584


namespace NUMINAMATH_GPT_football_players_count_l605_60502

def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def softball_players : ℕ := 13
def total_players : ℕ := 59

theorem football_players_count :
  total_players - (cricket_players + hockey_players + softball_players) = 18 :=
by 
  sorry

end NUMINAMATH_GPT_football_players_count_l605_60502


namespace NUMINAMATH_GPT_number_divisible_by_45_and_6_l605_60549

theorem number_divisible_by_45_and_6 (k : ℕ) (h1 : 1 ≤ k) (h2 : ∃ n : ℕ, 190 + 90 * (k - 1) ≤  n ∧ n < 190 + 90 * k) 
: 190 + 90 * 5 = 720 := by
  sorry

end NUMINAMATH_GPT_number_divisible_by_45_and_6_l605_60549


namespace NUMINAMATH_GPT_absolute_value_expression_evaluation_l605_60548

theorem absolute_value_expression_evaluation : abs (-2) * (abs (-Real.sqrt 25) - abs (Real.sin (5 * Real.pi / 2))) = 8 := by
  sorry

end NUMINAMATH_GPT_absolute_value_expression_evaluation_l605_60548


namespace NUMINAMATH_GPT_infinitely_many_divisors_l605_60511

theorem infinitely_many_divisors (a : ℕ) : ∃ᶠ n in at_top, n ∣ a ^ (n - a + 1) - 1 :=
sorry

end NUMINAMATH_GPT_infinitely_many_divisors_l605_60511


namespace NUMINAMATH_GPT_find_value_of_a_l605_60597

-- Let a, b, and c be different numbers from {1, 2, 4}
def a_b_c_valid (a b c : ℕ) : Prop := 
  (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
  (a = 1 ∨ a = 2 ∨ a = 4) ∧ 
  (b = 1 ∨ b = 2 ∨ b = 4) ∧ 
  (c = 1 ∨ c = 2 ∨ c = 4)

-- The condition that (a / 2) / (b / c) equals 4 when evaluated
def expr_eq_four (a b c : ℕ) : Prop :=
  (a / 2 : ℚ) / (b / c : ℚ) = 4

-- Given the above conditions, prove that the value of 'a' is 4
theorem find_value_of_a (a b c : ℕ) (h_valid : a_b_c_valid a b c) (h_expr : expr_eq_four a b c) : a = 4 := 
  sorry

end NUMINAMATH_GPT_find_value_of_a_l605_60597


namespace NUMINAMATH_GPT_prove_geomSeqSumFirst3_l605_60541

noncomputable def geomSeqSumFirst3 {a₁ a₆ : ℕ} (h₁ : a₁ = 1) (h₂ : a₆ = 32) : ℕ :=
  let r := 2 -- since r^5 = 32 which means r = 2
  let S3 := a₁ * (1 - r^3) / (1 - r)
  S3

theorem prove_geomSeqSumFirst3 : 
  geomSeqSumFirst3 (h₁ : 1 = 1) (h₂ : 32 = 32) = 7 := by
  sorry

end NUMINAMATH_GPT_prove_geomSeqSumFirst3_l605_60541


namespace NUMINAMATH_GPT_acrobats_count_l605_60506

theorem acrobats_count (a g : ℕ) 
  (h1 : 2 * a + 4 * g = 32) 
  (h2 : a + g = 10) : 
  a = 4 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_acrobats_count_l605_60506


namespace NUMINAMATH_GPT_tim_bought_two_appetizers_l605_60585

-- Definitions of the conditions.
def total_spending : ℝ := 50
def portion_spent_on_entrees : ℝ := 0.80
def entree_cost : ℝ := total_spending * portion_spent_on_entrees
def appetizer_cost : ℝ := 5
def appetizer_spending : ℝ := total_spending - entree_cost

-- The statement to prove: that Tim bought 2 appetizers.
theorem tim_bought_two_appetizers :
  appetizer_spending / appetizer_cost = 2 := 
by
  sorry

end NUMINAMATH_GPT_tim_bought_two_appetizers_l605_60585


namespace NUMINAMATH_GPT_cube_loop_probability_l605_60519

-- Define the number of faces and alignments for a cube
def total_faces := 6
def stripe_orientations_per_face := 2

-- Define the total possible stripe combinations
def total_stripe_combinations := stripe_orientations_per_face ^ total_faces

-- Define the combinations for both vertical and horizontal loops
def vertical_and_horizontal_loop_combinations := 64

-- Define the probability space
def probability_at_least_one_each := vertical_and_horizontal_loop_combinations / total_stripe_combinations

-- The main theorem to state the probability of having at least one vertical and one horizontal loop
theorem cube_loop_probability : probability_at_least_one_each = 1 := by
  sorry

end NUMINAMATH_GPT_cube_loop_probability_l605_60519


namespace NUMINAMATH_GPT_intersection_M_N_l605_60583

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := 3^x - 2

def M : Set ℝ := {x | f (g x) > 0}
def N : Set ℝ := {x | g x < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | x < 1} :=
by sorry

end NUMINAMATH_GPT_intersection_M_N_l605_60583


namespace NUMINAMATH_GPT_part1_part2_l605_60522

def f (x : ℝ) : ℝ := abs (x - 1)
def g (x : ℝ) : ℝ := abs (2 * x - 1)

theorem part1 (x : ℝ) :
  abs (x + 4) ≤ x * abs (2 * x - 1) ↔ x ≥ 2 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, abs ((x + 2) - 1) + abs (x - 1) + a = 0 → False) ↔ a ≤ -2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l605_60522


namespace NUMINAMATH_GPT_find_x_when_fx_eq_3_l605_60513

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ -1 then x + 2 else
if x < 2 then x^2 else
2 * x

theorem find_x_when_fx_eq_3 : ∃ x : ℝ, f x = 3 ∧ x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_find_x_when_fx_eq_3_l605_60513


namespace NUMINAMATH_GPT_verify_total_bill_l605_60565

def fixed_charge : ℝ := 20
def daytime_rate : ℝ := 0.10
def evening_rate : ℝ := 0.05
def free_evening_minutes : ℕ := 200

def daytime_minutes : ℕ := 200
def evening_minutes : ℕ := 300

noncomputable def total_bill : ℝ :=
  fixed_charge + (daytime_minutes * daytime_rate) +
  ((evening_minutes - free_evening_minutes) * evening_rate)

theorem verify_total_bill : total_bill = 45 := by
  sorry

end NUMINAMATH_GPT_verify_total_bill_l605_60565


namespace NUMINAMATH_GPT_linear_system_sum_l605_60596

theorem linear_system_sum (x y : ℝ) 
  (h1: x - y = 2) 
  (h2: y = 2): 
  x + y = 6 := 
sorry

end NUMINAMATH_GPT_linear_system_sum_l605_60596


namespace NUMINAMATH_GPT_coeffs_equal_implies_a_plus_b_eq_4_l605_60525

theorem coeffs_equal_implies_a_plus_b_eq_4 (a b : ℕ) (h_rel_prime : Nat.gcd a b = 1) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq_coeffs : (Nat.choose 2000 1998) * (a ^ 2) * (b ^ 1998) = (Nat.choose 2000 1997) * (a ^ 3) * (b ^ 1997)) :
  a + b = 4 := 
sorry

end NUMINAMATH_GPT_coeffs_equal_implies_a_plus_b_eq_4_l605_60525


namespace NUMINAMATH_GPT_find_rate_percent_l605_60593

theorem find_rate_percent (SI P T : ℝ) (h : SI = (P * R * T) / 100) (H_SI : SI = 250) 
  (H_P : P = 1500) (H_T : T = 5) : R = 250 / 75 := by
  sorry

end NUMINAMATH_GPT_find_rate_percent_l605_60593


namespace NUMINAMATH_GPT_real_roots_in_intervals_l605_60507

theorem real_roots_in_intervals (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ x1 x2 : ℝ, (x1 = a / 3 ∨ x1 = -2 * b / 3) ∧ (x2 = a / 3 ∨ x2 = -2 * b / 3) ∧ x1 ≠ x2 ∧
  (a / 3 ≤ x1 ∧ x1 ≤ 2 * a / 3) ∧ (-2 * b / 3 ≤ x2 ∧ x2 ≤ -b / 3) ∧
  (x1 > 0 ∧ x2 < 0) ∧ (1 / x1 + 1 / (x1 - a) + 1 / (x1 + b) = 0) ∧
  (1 / x2 + 1 / (x2 - a) + 1 / (x2 + b) = 0) :=
sorry

end NUMINAMATH_GPT_real_roots_in_intervals_l605_60507


namespace NUMINAMATH_GPT_sum_of_digits_2_1989_and_5_1989_l605_60577

theorem sum_of_digits_2_1989_and_5_1989 
  (m n : ℕ) 
  (h1 : 10^(m-1) < 2^1989 ∧ 2^1989 < 10^m) 
  (h2 : 10^(n-1) < 5^1989 ∧ 5^1989 < 10^n) 
  (h3 : 2^1989 * 5^1989 = 10^1989) : 
  m + n = 1990 := 
sorry

end NUMINAMATH_GPT_sum_of_digits_2_1989_and_5_1989_l605_60577


namespace NUMINAMATH_GPT_find_fff_l605_60521

def f (x : ℚ) : ℚ :=
  if x ≥ 2 then x + 2 else x * x

theorem find_fff : f (f (3/2)) = 17/4 := by
  sorry

end NUMINAMATH_GPT_find_fff_l605_60521


namespace NUMINAMATH_GPT_subtraction_division_l605_60599

theorem subtraction_division : 3550 - (1002 / 20.04) = 3499.9501 := by
  sorry

end NUMINAMATH_GPT_subtraction_division_l605_60599


namespace NUMINAMATH_GPT_race_distance_l605_60578

theorem race_distance (d x y z : ℝ) 
  (h1: d / x = (d - 25) / y)
  (h2: d / y = (d - 15) / z)
  (h3: d / x = (d - 35) / z) :
  d = 75 :=
sorry

end NUMINAMATH_GPT_race_distance_l605_60578


namespace NUMINAMATH_GPT_combined_mpg_l605_60594

def ray_mpg := 50
def tom_mpg := 20
def ray_miles := 100
def tom_miles := 200

theorem combined_mpg : 
  let ray_gallons := ray_miles / ray_mpg
  let tom_gallons := tom_miles / tom_mpg
  let total_gallons := ray_gallons + tom_gallons
  let total_miles := ray_miles + tom_miles
  total_miles / total_gallons = 25 :=
by
  sorry

end NUMINAMATH_GPT_combined_mpg_l605_60594


namespace NUMINAMATH_GPT_m_intersects_at_least_one_of_a_or_b_l605_60589

-- Definitions based on given conditions
variables {Plane : Type} {Line : Type} (α β : Plane) (a b m : Line)

-- Assume necessary conditions
axiom skew_lines (a b : Line) : Prop
axiom line_in_plane (l : Line) (p : Plane) : Prop
axiom plane_intersection_is_line (p1 p2 : Plane) : Line
axiom intersects (l1 l2 : Line) : Prop

-- Given conditions
variables
  (h1 : skew_lines a b)               -- a and b are skew lines
  (h2 : line_in_plane a α)            -- a is contained in plane α
  (h3 : line_in_plane b β)            -- b is contained in plane β
  (h4 : plane_intersection_is_line α β = m)  -- α ∩ β = m

-- The theorem to prove the correct answer
theorem m_intersects_at_least_one_of_a_or_b :
  intersects m a ∨ intersects m b :=
sorry -- proof to be provided

end NUMINAMATH_GPT_m_intersects_at_least_one_of_a_or_b_l605_60589


namespace NUMINAMATH_GPT_inequality_proof_l605_60558

-- Definitions for the conditions
variable (x y : ℝ)

-- Conditions
def conditions : Prop := 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Problem statement to be proven
theorem inequality_proof (h : conditions x y) : 
  x^3 + x * y^2 + 2 * x * y ≤ 2 * x^2 * y + x^2 + x + y := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l605_60558


namespace NUMINAMATH_GPT_percentage_of_copper_is_correct_l605_60573

-- Defining the conditions
def total_weight := 100.0
def weight_20_percent_alloy := 30.0
def weight_27_percent_alloy := total_weight - weight_20_percent_alloy

def percentage_20 := 0.20
def percentage_27 := 0.27

def copper_20 := percentage_20 * weight_20_percent_alloy
def copper_27 := percentage_27 * weight_27_percent_alloy
def total_copper := copper_20 + copper_27

-- The statement to be proved
def percentage_copper := (total_copper / total_weight) * 100

-- The theorem to prove
theorem percentage_of_copper_is_correct : percentage_copper = 24.9 := by sorry

end NUMINAMATH_GPT_percentage_of_copper_is_correct_l605_60573


namespace NUMINAMATH_GPT_quotient_base5_l605_60598

theorem quotient_base5 (a b quotient : ℕ) 
  (ha : a = 2 * 5^3 + 4 * 5^2 + 3 * 5^1 + 1) 
  (hb : b = 2 * 5^1 + 3) 
  (hquotient : quotient = 1 * 5^2 + 0 * 5^1 + 3) :
  a / b = quotient :=
by sorry

end NUMINAMATH_GPT_quotient_base5_l605_60598


namespace NUMINAMATH_GPT_units_digit_of_expression_l605_60508

noncomputable def C : ℝ := 7 + Real.sqrt 50
noncomputable def D : ℝ := 7 - Real.sqrt 50

theorem units_digit_of_expression (C D : ℝ) (hC : C = 7 + Real.sqrt 50) (hD : D = 7 - Real.sqrt 50) : 
  ((C ^ 21 + D ^ 21) % 10) = 4 :=
  sorry

end NUMINAMATH_GPT_units_digit_of_expression_l605_60508


namespace NUMINAMATH_GPT_additional_time_required_l605_60564

-- Definitions based on conditions
def time_to_clean_three_sections : ℕ := 24
def total_sections : ℕ := 27

-- Rate of cleaning
def cleaning_rate_per_section (t : ℕ) (n : ℕ) : ℕ := t / n

-- Total time required to clean all sections
def total_cleaning_time (n : ℕ) (r : ℕ) : ℕ := n * r

-- Additional time required to clean the remaining sections
def additional_cleaning_time (t_total : ℕ) (t_spent : ℕ) : ℕ := t_total - t_spent

-- Theorem statement
theorem additional_time_required 
  (t3 : ℕ) (n : ℕ) (t_spent : ℕ) 
  (h₁ : t3 = time_to_clean_three_sections)
  (h₂ : n = total_sections)
  (h₃ : t_spent = time_to_clean_three_sections)
  : additional_cleaning_time (total_cleaning_time n (cleaning_rate_per_section t3 3)) t_spent = 192 :=
by
  sorry

end NUMINAMATH_GPT_additional_time_required_l605_60564


namespace NUMINAMATH_GPT_dart_probability_l605_60543

noncomputable def area_hexagon (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

noncomputable def area_circle (s : ℝ) : ℝ := Real.pi * s^2

noncomputable def probability (s : ℝ) : ℝ := 
  (area_circle s) / (area_hexagon s)

theorem dart_probability (s : ℝ) (hs : s > 0) :
  probability s = (2 * Real.pi) / (3 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_dart_probability_l605_60543


namespace NUMINAMATH_GPT_fraction_power_simplification_l605_60523

theorem fraction_power_simplification:
  (81000/9000)^3 = 729 → (81000^3) / (9000^3) = 729 :=
by 
  intro h
  rw [<- h]
  sorry

end NUMINAMATH_GPT_fraction_power_simplification_l605_60523


namespace NUMINAMATH_GPT_inequality_property_l605_60537

theorem inequality_property (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : (a / b) > (b / a) := 
sorry

end NUMINAMATH_GPT_inequality_property_l605_60537


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_and_sum_max_l605_60555

theorem arithmetic_sequence_general_formula_and_sum_max :
  ∀ (a : ℕ → ℤ), 
  (a 7 = -8) → (a 17 = -28) → 
  (∀ n, a n = -2 * n + 6) ∧ 
  (∀ S : ℕ → ℤ, (∀ n, S n = -n^2 + 5 * n) → ∀ n, S n ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_and_sum_max_l605_60555


namespace NUMINAMATH_GPT_time_differences_l605_60581

def malcolm_speed := 6 -- minutes per mile
def joshua_speed := 8 -- minutes per mile
def lila_speed := 7 -- minutes per mile
def race_distance := 12 -- miles

noncomputable def malcolm_time := malcolm_speed * race_distance
noncomputable def joshua_time := joshua_speed * race_distance
noncomputable def lila_time := lila_speed * race_distance

theorem time_differences :
  joshua_time - malcolm_time = 24 ∧
  lila_time - malcolm_time = 12 :=
by
  sorry

end NUMINAMATH_GPT_time_differences_l605_60581


namespace NUMINAMATH_GPT_find_original_number_l605_60572

theorem find_original_number (x : ℕ) :
  (43 * x - 34 * x = 1251) → x = 139 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l605_60572


namespace NUMINAMATH_GPT_radius_of_base_circle_of_cone_l605_60586

theorem radius_of_base_circle_of_cone (θ : ℝ) (r_sector : ℝ) (L : ℝ) (C : ℝ) (r_base : ℝ) :
  θ = 120 ∧ r_sector = 6 ∧ L = (θ / 360) * 2 * Real.pi * r_sector ∧ C = L ∧ C = 2 * Real.pi * r_base → r_base = 2 := by
  sorry

end NUMINAMATH_GPT_radius_of_base_circle_of_cone_l605_60586


namespace NUMINAMATH_GPT_nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3_l605_60531

theorem nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3
  (a b : ℤ)
  (h : 9 ∣ (a^2 + a * b + b^2)) :
  3 ∣ a ∧ 3 ∣ b :=
sorry

end NUMINAMATH_GPT_nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3_l605_60531


namespace NUMINAMATH_GPT_remaining_dogs_eq_200_l605_60568

def initial_dogs : ℕ := 200
def additional_dogs : ℕ := 100
def first_adoption : ℕ := 40
def second_adoption : ℕ := 60

def total_dogs_after_adoption : ℕ :=
  initial_dogs + additional_dogs - first_adoption - second_adoption

theorem remaining_dogs_eq_200 : total_dogs_after_adoption = 200 :=
by
  -- Omitted the proof as requested
  sorry

end NUMINAMATH_GPT_remaining_dogs_eq_200_l605_60568


namespace NUMINAMATH_GPT_rhomboid_toothpicks_l605_60505

/-- 
Given:
- The rhomboid consists of two sections, each similar to half of a large equilateral triangle split along its height.
- The longest diagonal of the rhomboid contains 987 small equilateral triangles.
- The effective fact that each small equilateral triangle contributes on average 1.5 toothpicks due to shared sides.

Prove:
- The number of toothpicks required to construct the rhomboid is 1463598.
-/

-- Defining the number of small triangles along the base of the rhomboid
def base_triangles : ℕ := 987

-- Calculating the number of triangles in one section of the rhomboid
def triangles_in_section : ℕ := (base_triangles * (base_triangles + 1)) / 2

-- Calculating the total number of triangles in the rhomboid
def total_triangles : ℕ := 2 * triangles_in_section

-- Given the effective sides per triangle contributing to toothpicks is on average 1.5
def avg_sides_per_triangle : ℚ := 1.5

-- Calculating the total number of toothpicks required
def total_toothpicks : ℚ := avg_sides_per_triangle * total_triangles

theorem rhomboid_toothpicks (h : base_triangles = 987) : total_toothpicks = 1463598 := by
  sorry

end NUMINAMATH_GPT_rhomboid_toothpicks_l605_60505


namespace NUMINAMATH_GPT_percentage_received_certificates_l605_60538

theorem percentage_received_certificates (boys girls : ℕ) (pct_boys pct_girls : ℝ) :
    boys = 30 ∧ girls = 20 ∧ pct_boys = 0.1 ∧ pct_girls = 0.2 →
    ((pct_boys * boys + pct_girls * girls) / (boys + girls) * 100) = 14 := by
  sorry

end NUMINAMATH_GPT_percentage_received_certificates_l605_60538


namespace NUMINAMATH_GPT_jordan_final_weight_l605_60571

-- Defining the initial weight and the weight losses over the specified weeks
def initial_weight := 250
def loss_first_4_weeks := 4 * 3
def loss_next_8_weeks := 8 * 2
def total_loss := loss_first_4_weeks + loss_next_8_weeks

-- Theorem stating the final weight of Jordan
theorem jordan_final_weight : initial_weight - total_loss = 222 := by
  -- We skip the proof as requested
  sorry

end NUMINAMATH_GPT_jordan_final_weight_l605_60571


namespace NUMINAMATH_GPT_sum_of_integers_remainder_l605_60536

-- Definitions of the integers and their properties
variables (a b c : ℕ)

-- Conditions
axiom h1 : a % 53 = 31
axiom h2 : b % 53 = 17
axiom h3 : c % 53 = 8
axiom h4 : a % 5 = 0

-- The proof goal
theorem sum_of_integers_remainder :
  (a + b + c) % 53 = 3 :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_sum_of_integers_remainder_l605_60536


namespace NUMINAMATH_GPT_triangle_area_30_l605_60503

theorem triangle_area_30 (h : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 5 ∧ c = 13 ∧ b > 0) : 
  ∃ area : ℝ, area = 1 / 2 * 5 * (b : ℝ) ∧ area = 30 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_30_l605_60503


namespace NUMINAMATH_GPT_distance_between_B_and_D_l605_60579

theorem distance_between_B_and_D (a b c d : ℝ) (h1 : |2 * a - 3 * c| = 1) (h2 : |2 * b - 3 * c| = 1) (h3 : |(2/3) * (d - a)| = 1) (h4 : a ≠ b) :
  |d - b| = 0.5 ∨ |d - b| = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_B_and_D_l605_60579


namespace NUMINAMATH_GPT_committee_combinations_l605_60529

-- We use a broader import to ensure all necessary libraries are included.
-- Definitions and theorem

def club_member_count : ℕ := 20
def committee_member_count : ℕ := 3

theorem committee_combinations : 
  (Nat.choose club_member_count committee_member_count) = 1140 := by
sorry

end NUMINAMATH_GPT_committee_combinations_l605_60529


namespace NUMINAMATH_GPT_stating_area_trapezoid_AMBQ_is_18_l605_60551

/-- Definition of the 20-sided polygon configuration with 2 unit sides and right-angle turns. -/
structure Polygon20 where
  sides : ℕ → ℝ
  units : ∀ i, sides i = 2
  right_angles : ∀ i, (i + 1) % 20 ≠ i -- Right angles between consecutive sides

/-- Intersection point of AJ and DP, named M, under the given polygon configuration. -/
def intersection_point (p : Polygon20) : ℝ × ℝ :=
  (5 * p.sides 0, 5 * p.sides 1)  -- Assuming relevant distances for simplicity

/-- Area of the trapezoid AMBQ formed given the defined Polygon20. -/
noncomputable def area_trapezoid_AMBQ (p : Polygon20) : ℝ :=
  let base1 := 10 * p.sides 0
  let base2 := 8 * p.sides 0
  let height := p.sides 0
  (base1 + base2) * height / 2

/-- 
  Theorem stating the area of the trapezoid AMBQ in the given configuration.
  We prove that the area is 18 units.
-/
theorem area_trapezoid_AMBQ_is_18 (p : Polygon20) :
  area_trapezoid_AMBQ p = 18 :=
sorry -- Proof to be done

end NUMINAMATH_GPT_stating_area_trapezoid_AMBQ_is_18_l605_60551


namespace NUMINAMATH_GPT_unique_geometric_progression_12_a_b_ab_l605_60512

noncomputable def geometric_progression_12_a_b_ab : Prop :=
  ∃ (a b : ℝ), ∃ r : ℝ, a = 12 * r ∧ b = 12 * r^2 ∧ 12 * r * (12 * r^2) = 144 * r^3

theorem unique_geometric_progression_12_a_b_ab :
  ∃! (a b : ℝ), ∃ r : ℝ, a = 12 * r ∧ b = 12 * r^2 ∧ 12 * r * (12 * r^2) = 144 * r^3 :=
by
  sorry

end NUMINAMATH_GPT_unique_geometric_progression_12_a_b_ab_l605_60512


namespace NUMINAMATH_GPT_classmates_ate_cake_l605_60528

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end NUMINAMATH_GPT_classmates_ate_cake_l605_60528


namespace NUMINAMATH_GPT_fraction_decomposition_l605_60527

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ -8/3 → (7 * x - 19) / (3 * x^2 + 5 * x - 8) = A / (x - 1) + B / (3 * x + 8)) →
  A = -12 / 11 ∧ B = 113 / 11 :=
by
  sorry

end NUMINAMATH_GPT_fraction_decomposition_l605_60527


namespace NUMINAMATH_GPT_find_common_ratio_l605_60500

-- We need to state that q is the common ratio of the geometric sequence

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first three terms for the geometric sequence
def S_3 (a : ℕ → ℝ) := a 0 + a 1 + a 2

-- State the Lean 4 declaration of the proof problem
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : (S_3 a) / (a 2) = 3) :
  q = 1 := 
sorry

end NUMINAMATH_GPT_find_common_ratio_l605_60500


namespace NUMINAMATH_GPT_remaining_units_correct_l605_60556

-- Definitions based on conditions
def total_units : ℕ := 2000
def fraction_built_in_first_half : ℚ := 3/5
def additional_units_by_october : ℕ := 300

-- Calculate units built in the first half of the year
def units_built_in_first_half : ℚ := fraction_built_in_first_half * total_units

-- Remaining units after the first half of the year
def remaining_units_after_first_half : ℚ := total_units - units_built_in_first_half

-- Remaining units after building additional units by October
def remaining_units_to_be_built : ℚ := remaining_units_after_first_half - additional_units_by_october

-- Theorem statement: Prove remaining units to be built is 500
theorem remaining_units_correct : remaining_units_to_be_built = 500 := by
  sorry

end NUMINAMATH_GPT_remaining_units_correct_l605_60556


namespace NUMINAMATH_GPT_real_inequality_l605_60574

theorem real_inequality
  (a1 a2 a3 : ℝ)
  (h1 : 1 < a1)
  (h2 : 1 < a2)
  (h3 : 1 < a3)
  (S : ℝ)
  (hS : S = a1 + a2 + a3)
  (h4 : ∀ i ∈ [a1, a2, a3], (i^2 / (i - 1) > S)) :
  (1 / (a1 + a2) + 1 / (a2 + a3) + 1 / (a3 + a1) > 1) := 
by
  sorry

end NUMINAMATH_GPT_real_inequality_l605_60574


namespace NUMINAMATH_GPT_two_pow_n_plus_one_divisible_by_three_l605_60591

-- defining what it means to be an odd number
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- stating the main theorem in Lean
theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h_pos : 0 < n) : (2^n + 1) % 3 = 0 ↔ is_odd n :=
by sorry

end NUMINAMATH_GPT_two_pow_n_plus_one_divisible_by_three_l605_60591


namespace NUMINAMATH_GPT_cubes_sum_to_91_l605_60582

theorem cubes_sum_to_91
  (a b : ℤ)
  (h : a^3 + b^3 = 91) : a * b = 12 :=
sorry

end NUMINAMATH_GPT_cubes_sum_to_91_l605_60582


namespace NUMINAMATH_GPT_roots_are_reciprocals_eq_a_minus_one_l605_60554

theorem roots_are_reciprocals_eq_a_minus_one (a : ℝ) :
  (∀ x y : ℝ, x + y = -(a - 1) ∧ x * y = a^2 → x * y = 1) → a = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_roots_are_reciprocals_eq_a_minus_one_l605_60554


namespace NUMINAMATH_GPT_find_f_2011_l605_60592

def f: ℝ → ℝ :=
sorry

axiom f_periodicity (x : ℝ) : f (x + 3) = -f x
axiom f_initial_value : f 4 = -2

theorem find_f_2011 : f 2011 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2011_l605_60592


namespace NUMINAMATH_GPT_find_y_value_l605_60560

theorem find_y_value : (12 ^ 3 * 6 ^ 4) / 432 = 5184 := by
  sorry

end NUMINAMATH_GPT_find_y_value_l605_60560


namespace NUMINAMATH_GPT_sum_of_elements_in_M_l605_60533

theorem sum_of_elements_in_M (m : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + m = 0) :
  (∀ x : ℝ, x ∈ {x | x^2 - 2 * x + m = 0} → x = 1) ∧ m = 1 ∨
  (∃ x1 x2 : ℝ, x1 ∈ {x | x^2 - 2 * x + m = 0} ∧ x2 ∈ {x | x^2 - 2 * x + m = 0} ∧ x1 ≠ x2 ∧
   x1 + x2 = 2 ∧ m < 1) :=
sorry

end NUMINAMATH_GPT_sum_of_elements_in_M_l605_60533


namespace NUMINAMATH_GPT_m_power_of_prime_no_m_a_k_l605_60552

-- Part (i)
theorem m_power_of_prime (m : ℕ) (p : ℕ) (k : ℕ) (h1 : m ≥ 1) (h2 : Prime p) (h3 : m * (m + 1) = p^k) : m = 1 :=
by sorry

-- Part (ii)
theorem no_m_a_k (m a k : ℕ) (h1 : m ≥ 1) (h2 : a ≥ 1) (h3 : k ≥ 2) (h4 : m * (m + 1) = a^k) : False :=
by sorry

end NUMINAMATH_GPT_m_power_of_prime_no_m_a_k_l605_60552


namespace NUMINAMATH_GPT_ratio_of_wins_l605_60595

-- Definitions based on conditions
def W1 : ℕ := 15  -- Number of wins before first loss
def L : ℕ := 2    -- Total number of losses
def W2 : ℕ := 30 - W1  -- Calculate W2 based on W1 and total wins being 28 more than losses

-- Theorem statement: Prove the ratio of wins after her first loss to wins before her first loss is 1:1
theorem ratio_of_wins (h : W1 = 15 ∧ L = 2) : W2 / W1 = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_of_wins_l605_60595


namespace NUMINAMATH_GPT_cost_of_first_shipment_1100_l605_60514

variables (S J : ℝ)
-- conditions
def second_shipment (S J : ℝ) := 5 * S + 15 * J = 550
def first_shipment (S J : ℝ) := 10 * S + 20 * J

-- goal
theorem cost_of_first_shipment_1100 (S J : ℝ) (h : second_shipment S J) : first_shipment S J = 1100 :=
sorry

end NUMINAMATH_GPT_cost_of_first_shipment_1100_l605_60514


namespace NUMINAMATH_GPT_parallel_lines_implies_a_eq_one_l605_60559

theorem parallel_lines_implies_a_eq_one 
(h_parallel: ∀ (a : ℝ), ∀ (x y : ℝ), (x + a * y = 2 * a + 2) → (a * x + y = a + 1) → -1/a = -a) :
  ∀ (a : ℝ), a = 1 := by
  sorry

end NUMINAMATH_GPT_parallel_lines_implies_a_eq_one_l605_60559


namespace NUMINAMATH_GPT_positive_rational_as_sum_of_cubes_l605_60562

theorem positive_rational_as_sum_of_cubes (q : ℚ) (h_q_pos : q > 0) : 
  ∃ (a b c d : ℤ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ q = ((a^3 + b^3) / (c^3 + d^3)) :=
sorry

end NUMINAMATH_GPT_positive_rational_as_sum_of_cubes_l605_60562


namespace NUMINAMATH_GPT_find_a1_l605_60516

noncomputable def seq (a : ℕ → ℝ) : Prop :=
a 8 = 2 ∧ ∀ n, a (n + 1) = 1 / (1 - a n)

theorem find_a1 (a : ℕ → ℝ) (h : seq a) : a 1 = 1/2 := by
sorry

end NUMINAMATH_GPT_find_a1_l605_60516


namespace NUMINAMATH_GPT_percentage_of_passengers_in_first_class_l605_60545

theorem percentage_of_passengers_in_first_class (total_passengers : ℕ) (percentage_female : ℝ) (females_coach : ℕ) 
  (males_perc_first_class : ℝ) (Perc_first_class : ℝ) : 
  total_passengers = 120 → percentage_female = 0.45 → females_coach = 46 → males_perc_first_class = (1/3) → 
  Perc_first_class = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_of_passengers_in_first_class_l605_60545


namespace NUMINAMATH_GPT_solutions_in_nat_solutions_in_non_neg_int_l605_60544

-- Definitions for Part A
def nat_sol_count (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem solutions_in_nat (x1 x2 x3 : ℕ) : 
  (x1 > 0) → (x2 > 0) → (x3 > 0) → (x1 + x2 + x3 = 1000) → 
  nat_sol_count 997 3 = Nat.choose 999 2 := sorry

-- Definitions for Part B
theorem solutions_in_non_neg_int (x1 x2 x3 : ℕ) : 
  (x1 + x2 + x3 = 1000) → 
  nat_sol_count 1000 3 = Nat.choose 1002 2 := sorry

end NUMINAMATH_GPT_solutions_in_nat_solutions_in_non_neg_int_l605_60544


namespace NUMINAMATH_GPT_locus_of_right_angle_vertex_l605_60587

variables {x y : ℝ}

/-- Given points M(-2,0) and N(2,0), if P(x,y) is the right-angled vertex of
  a right-angled triangle with MN as its hypotenuse, then the locus equation
  of P is given by x^2 + y^2 = 4 with the condition x ≠ ±2. -/
theorem locus_of_right_angle_vertex (h : x ≠ 2 ∧ x ≠ -2) :
  x^2 + y^2 = 4 :=
sorry

end NUMINAMATH_GPT_locus_of_right_angle_vertex_l605_60587


namespace NUMINAMATH_GPT_Robin_needs_to_buy_more_bottles_l605_60550

/-- Robin wants to drink exactly nine bottles of water each day.
    She initially bought six hundred seventeen bottles.
    Prove that she will need to buy 4 more bottles on the last day
    to meet her goal of drinking exactly nine bottles each day. -/
theorem Robin_needs_to_buy_more_bottles :
  ∀ total_bottles bottles_per_day : ℕ, total_bottles = 617 → bottles_per_day = 9 → 
  ∃ extra_bottles : ℕ, (617 % 9) + extra_bottles = 9 ∧ extra_bottles = 4 :=
by
  sorry

end NUMINAMATH_GPT_Robin_needs_to_buy_more_bottles_l605_60550


namespace NUMINAMATH_GPT_greening_investment_equation_l605_60524

theorem greening_investment_equation:
  ∃ (x : ℝ), 20 * (1 + x)^2 = 25 := 
sorry

end NUMINAMATH_GPT_greening_investment_equation_l605_60524


namespace NUMINAMATH_GPT_polynomial_evaluation_l605_60534

theorem polynomial_evaluation :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l605_60534


namespace NUMINAMATH_GPT_sum_of_cubes_of_consecutive_even_integers_l605_60535

theorem sum_of_cubes_of_consecutive_even_integers (x : ℤ) (h : x^2 + (x+2)^2 + (x+4)^2 = 2960) :
  x^3 + (x + 2)^3 + (x + 4)^3 = 90117 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_of_consecutive_even_integers_l605_60535


namespace NUMINAMATH_GPT_average_discount_rate_l605_60563

theorem average_discount_rate
  (bag_marked_price : ℝ) (bag_sold_price : ℝ)
  (shoes_marked_price : ℝ) (shoes_sold_price : ℝ)
  (jacket_marked_price : ℝ) (jacket_sold_price : ℝ)
  (h_bag : bag_marked_price = 80) (h_bag_sold : bag_sold_price = 68)
  (h_shoes : shoes_marked_price = 120) (h_shoes_sold : shoes_sold_price = 96)
  (h_jacket : jacket_marked_price = 150) (h_jacket_sold : jacket_sold_price = 135) : 
  (15 : ℝ) =
  (((bag_marked_price - bag_sold_price) / bag_marked_price * 100) + 
   ((shoes_marked_price - shoes_sold_price) / shoes_marked_price * 100) + 
   ((jacket_marked_price - jacket_sold_price) / jacket_marked_price * 100)) / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_discount_rate_l605_60563


namespace NUMINAMATH_GPT_triangle_angle_not_greater_than_60_l605_60501

theorem triangle_angle_not_greater_than_60 (A B C : ℝ) (h1 : A + B + C = 180) :
  A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
sorry -- proof by contradiction to be implemented here

end NUMINAMATH_GPT_triangle_angle_not_greater_than_60_l605_60501


namespace NUMINAMATH_GPT_squares_difference_l605_60517

theorem squares_difference (n : ℕ) : (n + 1)^2 - n^2 = 2 * n + 1 := by
  sorry

end NUMINAMATH_GPT_squares_difference_l605_60517


namespace NUMINAMATH_GPT_chi_square_hypothesis_test_l605_60530

-- Definitions based on the conditions
def males_like_sports := "Males like to participate in sports activities"
def females_dislike_sports := "Females do not like to participate in sports activities"
def activities_related_to_gender := "Liking to participate in sports activities is related to gender"
def activities_not_related_to_gender := "Liking to participate in sports activities is not related to gender"

-- Statement to prove that D is the correct null hypothesis
theorem chi_square_hypothesis_test :
  activities_not_related_to_gender = "H₀: Liking to participate in sports activities is not related to gender" :=
sorry

end NUMINAMATH_GPT_chi_square_hypothesis_test_l605_60530


namespace NUMINAMATH_GPT_expression_evaluation_l605_60515

variable (x y : ℤ)

theorem expression_evaluation (h₁ : x = -1) (h₂ : y = 1) : 
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 2 :=
by
  rw [h₁, h₂]
  have h₃ : (-1 + 1) * (-1 - 1) - (4 * (-1)^3 * 1 - 8 * (-1) * 1^3) / (2 * (-1) * 1) = (-2) - (-10 / -2) := by sorry
  have h₄ : (-2) - 5 = 2 := by sorry
  sorry

end NUMINAMATH_GPT_expression_evaluation_l605_60515


namespace NUMINAMATH_GPT_elisa_improvement_l605_60539

theorem elisa_improvement (cur_laps cur_minutes prev_laps prev_minutes : ℕ) 
  (h1 : cur_laps = 15) (h2 : cur_minutes = 30) 
  (h3 : prev_laps = 20) (h4 : prev_minutes = 50) : 
  ((prev_minutes / prev_laps : ℚ) - (cur_minutes / cur_laps : ℚ) = 0.5) :=
by
  sorry

end NUMINAMATH_GPT_elisa_improvement_l605_60539


namespace NUMINAMATH_GPT_ranking_emily_olivia_nicole_l605_60567

noncomputable def Emily_score : ℝ := sorry
noncomputable def Olivia_score : ℝ := sorry
noncomputable def Nicole_score : ℝ := sorry

theorem ranking_emily_olivia_nicole :
  (Emily_score > Olivia_score) ∧ (Emily_score > Nicole_score) → 
  (Emily_score > Olivia_score) ∧ (Olivia_score > Nicole_score) := 
by sorry

end NUMINAMATH_GPT_ranking_emily_olivia_nicole_l605_60567


namespace NUMINAMATH_GPT_circle_m_range_l605_60553

theorem circle_m_range (m : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 - 2 * x + 6 * y + m = 0 → m < 10) :=
sorry

end NUMINAMATH_GPT_circle_m_range_l605_60553


namespace NUMINAMATH_GPT_find_a_for_inequality_l605_60588

theorem find_a_for_inequality (a : ℚ) :
  (∀ x : ℚ, (ax / (x - 1)) < 1 ↔ (x < 1 ∨ x > 2)) → a = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_inequality_l605_60588


namespace NUMINAMATH_GPT_value_of_3y_l605_60532

theorem value_of_3y (x y z : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 11) (h4 : z = 3) :
  3 * y = 12 :=
by
  sorry

end NUMINAMATH_GPT_value_of_3y_l605_60532


namespace NUMINAMATH_GPT_calculate_fraction_l605_60509

theorem calculate_fraction: (1 / (2 + 1 / (3 + 1 / 4))) = 13 / 30 := by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l605_60509


namespace NUMINAMATH_GPT_who_received_q_first_round_l605_60504

-- Define the variables and conditions
variables (p q r : ℕ) (A B C : ℕ → ℕ) (n : ℕ)

-- Conditions
axiom h1 : 0 < p
axiom h2 : p < q
axiom h3 : q < r
axiom h4 : n ≥ 3
axiom h5 : A n = 20
axiom h6 : B n = 10
axiom h7 : C n = 9
axiom h8 : ∀ k, k > 0 → (B k = r → B (k-1) ≠ r)
axiom h9 : p + q + r = 13

-- Theorem to prove
theorem who_received_q_first_round : C 1 = q :=
sorry

end NUMINAMATH_GPT_who_received_q_first_round_l605_60504


namespace NUMINAMATH_GPT_sample_size_of_survey_l605_60510

theorem sample_size_of_survey (total_students : ℕ) (analyzed_students : ℕ)
  (h1 : total_students = 4000) (h2 : analyzed_students = 500) :
  analyzed_students = 500 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_of_survey_l605_60510


namespace NUMINAMATH_GPT_minimum_cost_is_correct_l605_60590

noncomputable def rectangular_area (length width : ℝ) : ℝ :=
  length * width

def flower_cost_per_sqft (flower : String) : ℝ :=
  match flower with
  | "Marigold" => 1.00
  | "Sunflower" => 1.75
  | "Tulip" => 1.25
  | "Orchid" => 2.75
  | "Iris" => 3.25
  | _ => 0.00

def min_garden_cost : ℝ :=
  let areas := [rectangular_area 5 2, rectangular_area 7 3, rectangular_area 5 5, rectangular_area 2 4, rectangular_area 5 4]
  let costs := [flower_cost_per_sqft "Orchid" * 8, 
                flower_cost_per_sqft "Iris" * 10, 
                flower_cost_per_sqft "Sunflower" * 20, 
                flower_cost_per_sqft "Tulip" * 21, 
                flower_cost_per_sqft "Marigold" * 25]
  costs.sum

theorem minimum_cost_is_correct :
  min_garden_cost = 140.75 :=
  by
    -- Proof omitted
    sorry

end NUMINAMATH_GPT_minimum_cost_is_correct_l605_60590


namespace NUMINAMATH_GPT_min_value_inequality_l605_60540

noncomputable def minValue : ℝ := 17 / 2

theorem min_value_inequality (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_cond : a + 2 * b = 1) :
  a^2 + 4 * b^2 + 1 / (a * b) = minValue := 
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l605_60540


namespace NUMINAMATH_GPT_price_white_stamp_l605_60561

variable (price_per_white_stamp : ℝ)

theorem price_white_stamp (simon_red_stamps : ℕ)
                          (peter_white_stamps : ℕ)
                          (price_per_red_stamp : ℝ)
                          (money_difference : ℝ)
                          (h1 : simon_red_stamps = 30)
                          (h2 : peter_white_stamps = 80)
                          (h3 : price_per_red_stamp = 0.50)
                          (h4 : money_difference = 1) :
    money_difference = peter_white_stamps * price_per_white_stamp - simon_red_stamps * price_per_red_stamp →
    price_per_white_stamp = 1 / 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_price_white_stamp_l605_60561


namespace NUMINAMATH_GPT_shipCargoCalculation_l605_60526

def initialCargo : Int := 5973
def cargoLoadedInBahamas : Int := 8723
def totalCargo (initial : Int) (loaded : Int) : Int := initial + loaded

theorem shipCargoCalculation : totalCargo initialCargo cargoLoadedInBahamas = 14696 := by
  sorry

end NUMINAMATH_GPT_shipCargoCalculation_l605_60526


namespace NUMINAMATH_GPT_part1_part2_l605_60569

theorem part1 (a b h3 : ℝ) (C : ℝ) (h : 1 / h3 = 1 / a + 1 / b) : C ≤ 120 :=
sorry

theorem part2 (a b m3 : ℝ) (C : ℝ) (h : 1 / m3 = 1 / a + 1 / b) : C ≥ 120 :=
sorry

end NUMINAMATH_GPT_part1_part2_l605_60569
