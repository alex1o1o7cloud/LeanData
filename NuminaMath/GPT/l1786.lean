import Mathlib

namespace NUMINAMATH_GPT_beads_per_bracelet_l1786_178653

-- Definitions for the conditions
def Nancy_metal_beads : ℕ := 40
def Nancy_pearl_beads : ℕ := Nancy_metal_beads + 20
def Rose_crystal_beads : ℕ := 20
def Rose_stone_beads : ℕ := Rose_crystal_beads * 2
def total_beads : ℕ := Nancy_metal_beads + Nancy_pearl_beads + Rose_crystal_beads + Rose_stone_beads
def bracelets : ℕ := 20

-- Statement to prove
theorem beads_per_bracelet :
  total_beads / bracelets = 8 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_beads_per_bracelet_l1786_178653


namespace NUMINAMATH_GPT_nth_wise_number_1990_l1786_178650

/--
A natural number that can be expressed as the difference of squares 
of two other natural numbers is called a "wise number".
-/
def is_wise_number (n : ℕ) : Prop :=
  ∃ x y : ℕ, x^2 - y^2 = n

/--
The 1990th "wise number" is 2659.
-/
theorem nth_wise_number_1990 : ∃ n : ℕ, is_wise_number n ∧ n = 2659 :=
  sorry

end NUMINAMATH_GPT_nth_wise_number_1990_l1786_178650


namespace NUMINAMATH_GPT_initial_amount_correct_l1786_178630

-- Definitions
def spent_on_fruits : ℝ := 15.00
def left_to_spend : ℝ := 85.00
def initial_amount_given (spent: ℝ) (left: ℝ) : ℝ := spent + left

-- Theorem stating the problem
theorem initial_amount_correct :
  initial_amount_given spent_on_fruits left_to_spend = 100.00 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_correct_l1786_178630


namespace NUMINAMATH_GPT_find_multiple_l1786_178610

variables (total_questions correct_answers score : ℕ)
variable (m : ℕ)
variable (incorrect_answers : ℕ := total_questions - correct_answers)

-- Given conditions
axiom total_questions_eq : total_questions = 100
axiom correct_answers_eq : correct_answers = 92
axiom score_eq : score = 76

-- Define the scoring method
def score_formula : ℕ := correct_answers - m * incorrect_answers

-- Statement to prove
theorem find_multiple : score = 76 → correct_answers = 92 → total_questions = 100 → score_formula total_questions correct_answers m = score → m = 2 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_multiple_l1786_178610


namespace NUMINAMATH_GPT_AB_eq_B_exp_V_l1786_178664

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end NUMINAMATH_GPT_AB_eq_B_exp_V_l1786_178664


namespace NUMINAMATH_GPT_determine_k_and_solution_l1786_178600

theorem determine_k_and_solution :
  ∃ (k : ℚ), (5 * k * x^2 + 30 * x + 10 = 0 → k = 9/2) ∧
    (∃ (x : ℚ), (5 * (9/2) * x^2 + 30 * x + 10 = 0) ∧ x = -2/3) := by
  sorry

end NUMINAMATH_GPT_determine_k_and_solution_l1786_178600


namespace NUMINAMATH_GPT_aqua_park_earnings_l1786_178654

def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def meal_fee : ℕ := 10
def souvenir_fee : ℕ := 8

def group1_admission_count : ℕ := 10
def group1_tour_count : ℕ := 10
def group1_meal_count : ℕ := 10
def group1_souvenir_count : ℕ := 10
def group1_discount : ℚ := 0.10

def group2_admission_count : ℕ := 15
def group2_meal_count : ℕ := 15
def group2_meal_discount : ℚ := 0.05

def group3_admission_count : ℕ := 8
def group3_tour_count : ℕ := 8
def group3_souvenir_count : ℕ := 8

-- total cost for group 1 before discount
def group1_total_before_discount : ℕ := 
  (group1_admission_count * admission_fee) +
  (group1_tour_count * tour_fee) +
  (group1_meal_count * meal_fee) +
  (group1_souvenir_count * souvenir_fee)

-- group 1 total cost after discount
def group1_total_after_discount : ℚ :=
  group1_total_before_discount * (1 - group1_discount)

-- total cost for group 2 before discount
def group2_admission_total_before_discount : ℕ := 
  group2_admission_count * admission_fee
def group2_meal_total_before_discount : ℕ := 
  group2_meal_count * meal_fee

-- group 2 total cost after discount
def group2_meal_total_after_discount : ℚ :=
  group2_meal_total_before_discount * (1 - group2_meal_discount)
def group2_total_after_discount : ℚ :=
  group2_admission_total_before_discount + group2_meal_total_after_discount

-- total cost for group 3 before discount
def group3_total_before_discount : ℕ := 
  (group3_admission_count * admission_fee) +
  (group3_tour_count * tour_fee) +
  (group3_souvenir_count * souvenir_fee)

-- group 3 total cost after discount (no discount applied)
def group3_total_after_discount : ℕ := group3_total_before_discount

-- total earnings from all groups
def total_earnings : ℚ :=
  group1_total_after_discount +
  group2_total_after_discount +
  group3_total_after_discount

theorem aqua_park_earnings : total_earnings = 854.50 := by
  sorry

end NUMINAMATH_GPT_aqua_park_earnings_l1786_178654


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1786_178649

-- Definitions of sets A and B
def set_A : Set ℝ := { x | x^2 - x - 6 < 0 }
def set_B : Set ℝ := { x | (x + 4) * (x - 2) > 0 }

-- Theorem statement for the intersection of A and B
theorem intersection_of_A_and_B : set_A ∩ set_B = { x | 2 < x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1786_178649


namespace NUMINAMATH_GPT_shaded_area_calculation_l1786_178661

-- Define the grid and the side length conditions
def grid_size : ℕ := 5 * 4
def side_length : ℕ := 1
def total_squares : ℕ := 5 * 4

-- Define the area of one small square
def area_of_square (side: ℕ) : ℕ := side * side

-- Define the shaded region in terms of number of small squares fully or partially occupied
def shaded_squares : ℕ := 11

-- By analyzing the grid based on given conditions, prove that the area of the shaded region is 11
theorem shaded_area_calculation : (shaded_squares * side_length * side_length) = 11 := sorry

end NUMINAMATH_GPT_shaded_area_calculation_l1786_178661


namespace NUMINAMATH_GPT_difference_between_mean_and_median_l1786_178673

def percent_students := {p : ℝ // 0 ≤ p ∧ p ≤ 1}

def students_scores_distribution (p60 p75 p85 p95 : percent_students) : Prop :=
  p60.val + p75.val + p85.val + p95.val = 1 ∧
  p60.val = 0.15 ∧
  p75.val = 0.20 ∧
  p85.val = 0.40 ∧
  p95.val = 0.25

noncomputable def weighted_mean (p60 p75 p85 p95 : percent_students) : ℝ :=
  60 * p60.val + 75 * p75.val + 85 * p85.val + 95 * p95.val

noncomputable def median_score (p60 p75 p85 p95 : percent_students) : ℝ :=
  if p60.val + p75.val < 0.5 then 85 else if p60.val + p75.val < 0.9 then 95 else 60

theorem difference_between_mean_and_median :
  ∀ (p60 p75 p85 p95 : percent_students),
    students_scores_distribution p60 p75 p85 p95 →
    abs (median_score p60 p75 p85 p95 - weighted_mean p60 p75 p85 p95) = 3.25 :=
by
  intro p60 p75 p85 p95
  intro h
  sorry

end NUMINAMATH_GPT_difference_between_mean_and_median_l1786_178673


namespace NUMINAMATH_GPT_average_of_roots_l1786_178627

theorem average_of_roots (c : ℝ) (h : ∃ x1 x2 : ℝ, 2 * x1^2 - 6 * x1 + c = 0 ∧ 2 * x2^2 - 6 * x2 + c = 0 ∧ x1 ≠ x2) :
    (∃ p q : ℝ, (2 : ℝ) * (p : ℝ)^2 + (-6 : ℝ) * p + c = 0 ∧ (2 : ℝ) * (q : ℝ)^2 + (-6 : ℝ) * q + c = 0 ∧ p ≠ q) →
    (p + q) / 2 = 3 / 2 := 
sorry

end NUMINAMATH_GPT_average_of_roots_l1786_178627


namespace NUMINAMATH_GPT_sin_neg_30_eq_neg_one_half_l1786_178637

theorem sin_neg_30_eq_neg_one_half : Real.sin (-30 / 180 * Real.pi) = -1 / 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sin_neg_30_eq_neg_one_half_l1786_178637


namespace NUMINAMATH_GPT_remaining_watermelons_l1786_178612

-- Define the given conditions
def initial_watermelons : ℕ := 35
def watermelons_eaten : ℕ := 27

-- Define the question as a theorem
theorem remaining_watermelons : 
  initial_watermelons - watermelons_eaten = 8 :=
by
  sorry

end NUMINAMATH_GPT_remaining_watermelons_l1786_178612


namespace NUMINAMATH_GPT_speed_of_water_is_10_l1786_178683

/-- Define the conditions -/
def swimming_speed_in_still_water : ℝ := 12 -- km/h
def time_to_swim_against_current : ℝ := 4 -- hours
def distance_against_current : ℝ := 8 -- km

/-- Define the effective speed against the current and the proof goal -/
def speed_of_water (v : ℝ) : Prop :=
  (swimming_speed_in_still_water - v) = distance_against_current / time_to_swim_against_current

theorem speed_of_water_is_10 : speed_of_water 10 :=
by
  unfold speed_of_water
  sorry

end NUMINAMATH_GPT_speed_of_water_is_10_l1786_178683


namespace NUMINAMATH_GPT_hyperbola_focal_coordinates_l1786_178628

theorem hyperbola_focal_coordinates:
  ∀ (x y : ℝ), x^2 / 16 - y^2 / 9 = 1 → ∃ c : ℝ, c = 5 ∧ (x = -c ∨ x = c) ∧ y = 0 :=
by
  intro x y
  sorry

end NUMINAMATH_GPT_hyperbola_focal_coordinates_l1786_178628


namespace NUMINAMATH_GPT_find_positive_real_solution_l1786_178679

theorem find_positive_real_solution (x : ℝ) (h1 : x > 0) (h2 : (x - 5) / 8 = 5 / (x - 8)) : x = 13 := 
sorry

end NUMINAMATH_GPT_find_positive_real_solution_l1786_178679


namespace NUMINAMATH_GPT_exists_divisible_by_3_l1786_178640

open Nat

-- Definitions used in Lean 4 statement to represent conditions from part a)
def neighbors (n m : ℕ) : Prop := (m = n + 1) ∨ (m = n + 2) ∨ (2 * m = n) ∨ (m = 2 * n)

def circle_arrangement (ns : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, (neighbors (ns i) (ns ((i + 1) % 99)))

-- Proof problem:
theorem exists_divisible_by_3 (ns : Fin 99 → ℕ) (h : circle_arrangement ns) :
  ∃ i : Fin 99, 3 ∣ ns i :=
sorry

end NUMINAMATH_GPT_exists_divisible_by_3_l1786_178640


namespace NUMINAMATH_GPT_min_value_sin_cos_l1786_178646

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_sin_cos_l1786_178646


namespace NUMINAMATH_GPT_polynomial_factorization_l1786_178687

-- Definitions used in the conditions
def given_polynomial (a b c : ℝ) : ℝ :=
  a^3 * (b^2 - c^2) + b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2)

def p (a b c : ℝ) : ℝ := -(a * b + a * c + b * c)

-- The Lean 4 statement to be proved
theorem polynomial_factorization (a b c : ℝ) :
  given_polynomial a b c = (a - b) * (b - c) * (c - a) * p a b c :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l1786_178687


namespace NUMINAMATH_GPT_rectangle_properties_l1786_178676

noncomputable def diagonal (x1 y1 x2 y2 : ℕ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def area (length width : ℕ) : ℕ :=
  length * width

theorem rectangle_properties :
  diagonal 1 1 9 7 = 10 ∧ area (9 - 1) (7 - 1) = 48 := by
  sorry

end NUMINAMATH_GPT_rectangle_properties_l1786_178676


namespace NUMINAMATH_GPT_largest_mersenne_prime_lt_500_l1786_178669

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_mersenne_prime (p : ℕ) : Prop :=
  is_prime p ∧ is_prime (2^p - 1)

theorem largest_mersenne_prime_lt_500 : 
  ∀ n, is_mersenne_prime n → 2^n - 1 < 500 → 2^n - 1 ≤ 127 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_largest_mersenne_prime_lt_500_l1786_178669


namespace NUMINAMATH_GPT_simplify_expression_l1786_178611

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (yz + xz + xy) / (xyz * (x + y + z)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1786_178611


namespace NUMINAMATH_GPT_sqrt_S_n_arithmetic_seq_seq_sqrt_S_n_condition_l1786_178625

-- (1)
theorem sqrt_S_n_arithmetic_seq (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d) (h3 : S n = (n * (2 * a 1 + (n - 1) * (2 : ℝ))) / 2) :
  ∃ d, ∀ n, Real.sqrt (S (n + 1)) = Real.sqrt (S n) + d :=
by sorry

-- (2)
theorem seq_sqrt_S_n_condition (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ) :
  (∃ d, ∀ n, S n / 2 = n * (a1 + (n - 1) * d)) ↔ (∀ n, S n = a1 * n^2) :=
by sorry

end NUMINAMATH_GPT_sqrt_S_n_arithmetic_seq_seq_sqrt_S_n_condition_l1786_178625


namespace NUMINAMATH_GPT_reciprocal_of_neg_three_l1786_178623

theorem reciprocal_of_neg_three : ∃ x : ℚ, (-3) * x = 1 ∧ x = (-1) / 3 := sorry

end NUMINAMATH_GPT_reciprocal_of_neg_three_l1786_178623


namespace NUMINAMATH_GPT_perfect_square_polynomial_l1786_178666

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2) ↔ (x = -1 ∨ x = 0 ∨ x = 3) :=
sorry

end NUMINAMATH_GPT_perfect_square_polynomial_l1786_178666


namespace NUMINAMATH_GPT_intersection_of_AB_CD_l1786_178642

def point (α : Type*) := (α × α × α)

def A : point ℚ := (5, -8, 9)
def B : point ℚ := (15, -18, 14)
def C : point ℚ := (1, 4, -7)
def D : point ℚ := (3, -4, 11)

def parametric_AB (t : ℚ) : point ℚ :=
  (5 + 10 * t, -8 - 10 * t, 9 + 5 * t)

def parametric_CD (s : ℚ) : point ℚ :=
  (1 + 2 * s, 4 - 8 * s, -7 + 18 * s)

def intersection_point (pi : point ℚ) :=
  ∃ t s : ℚ, parametric_AB t = pi ∧ parametric_CD s = pi

theorem intersection_of_AB_CD : intersection_point (76/15, -118/15, 170/15) :=
  sorry

end NUMINAMATH_GPT_intersection_of_AB_CD_l1786_178642


namespace NUMINAMATH_GPT_wang_trip_duration_xiao_travel_times_l1786_178695

variables (start_fee : ℝ) (time_fee_per_min : ℝ) (mileage_fee_per_km : ℝ) (long_distance_fee_per_km : ℝ)

-- Conditions
def billing_rules := 
  start_fee = 12 ∧ 
  time_fee_per_min = 0.5 ∧ 
  mileage_fee_per_km = 2.0 ∧ 
  long_distance_fee_per_km = 1.0

-- Proof for Mr. Wang's trip duration
theorem wang_trip_duration
  (x : ℝ) 
  (total_fare : ℝ)
  (distance : ℝ) 
  (h : billing_rules start_fee time_fee_per_min mileage_fee_per_km long_distance_fee_per_km) : 
  total_fare = 69.5 ∧ distance = 20 → 0.5 * x = 12.5 :=
by 
  sorry

-- Proof for Xiao Hong's and Xiao Lan's travel times
theorem xiao_travel_times 
  (x : ℝ) 
  (travel_time_multiplier : ℝ)
  (distance_hong : ℝ)
  (distance_lan : ℝ)
  (equal_fares : Prop)
  (h : billing_rules start_fee time_fee_per_min mileage_fee_per_km long_distance_fee_per_km)
  (p1 : distance_hong = 14 ∧ distance_lan = 16 ∧ travel_time_multiplier = 1.5) :
  equal_fares → 0.25 * x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_wang_trip_duration_xiao_travel_times_l1786_178695


namespace NUMINAMATH_GPT_kelly_carrot_weight_l1786_178633

-- Define the number of carrots harvested from each bed
def carrots_bed1 : ℕ := 55
def carrots_bed2 : ℕ := 101
def carrots_bed3 : ℕ := 78
def carrots_per_pound : ℕ := 6

-- Define the total number of carrots
def total_carrots := carrots_bed1 + carrots_bed2 + carrots_bed3

-- Define the total weight in pounds
def total_weight := total_carrots / carrots_per_pound

-- The theorem to prove the total weight is 39 pounds
theorem kelly_carrot_weight : total_weight = 39 := by
  sorry

end NUMINAMATH_GPT_kelly_carrot_weight_l1786_178633


namespace NUMINAMATH_GPT_sin_double_angle_l1786_178659

theorem sin_double_angle (h1 : Real.pi / 2 < β)
    (h2 : β < α)
    (h3 : α < 3 * Real.pi / 4)
    (h4 : Real.cos (α - β) = 12 / 13)
    (h5 : Real.sin (α + β) = -3 / 5) :
    Real.sin (2 * α) = -56 / 65 := 
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1786_178659


namespace NUMINAMATH_GPT_mary_friends_count_l1786_178607

-- Definitions based on conditions
def total_stickers := 50
def stickers_left := 8
def total_students := 17
def classmates := total_students - 1 -- excluding Mary

-- Defining the proof problem
theorem mary_friends_count (F : ℕ) (h1 : 4 * F + 2 * (classmates - F) = total_stickers - stickers_left) :
  F = 5 :=
by sorry

end NUMINAMATH_GPT_mary_friends_count_l1786_178607


namespace NUMINAMATH_GPT_product_513_12_l1786_178638

theorem product_513_12 : 513 * 12 = 6156 := 
  by
    sorry

end NUMINAMATH_GPT_product_513_12_l1786_178638


namespace NUMINAMATH_GPT_sum_minimum_values_l1786_178629

def P (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
def Q (x : ℝ) (d e f : ℝ) : ℝ := x^3 + d * x^2 + e * x + f

theorem sum_minimum_values (a b c d e f : ℝ)
  (hPQ : ∀ x, P (Q x d e f) a b c = 0 → x = -4 ∨ x = -2 ∨ x = 0 ∨ x = 2 ∨ x = 4)
  (hQP : ∀ x, Q (P x a b c) d e f = 0 → x = -3 ∨ x = -1 ∨ x = 1 ∨ x = 3) :
  P 0 a b c + Q 0 d e f = -20 := sorry

end NUMINAMATH_GPT_sum_minimum_values_l1786_178629


namespace NUMINAMATH_GPT_factorize1_factorize2_factorize3_factorize4_l1786_178604

-- Statement for the first equation
theorem factorize1 (a x : ℝ) : 
  a * x^2 - 7 * a * x + 6 * a = a * (x - 6) * (x - 1) :=
sorry

-- Statement for the second equation
theorem factorize2 (x y : ℝ) : 
  x * y^2 - 9 * x = x * (y + 3) * (y - 3) :=
sorry

-- Statement for the third equation
theorem factorize3 (x y : ℝ) : 
  1 - x^2 + 2 * x * y - y^2 = (1 + x - y) * (1 - x + y) :=
sorry

-- Statement for the fourth equation
theorem factorize4 (x y : ℝ) : 
  8 * (x^2 - 2 * y^2) - x * (7 * x + y) + x * y = (x + 4 * y) * (x - 4 * y) :=
sorry

end NUMINAMATH_GPT_factorize1_factorize2_factorize3_factorize4_l1786_178604


namespace NUMINAMATH_GPT_fraction_sum_product_roots_of_quadratic_l1786_178601

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_fraction_sum_product_roots_of_quadratic_l1786_178601


namespace NUMINAMATH_GPT_drawing_time_total_l1786_178657

theorem drawing_time_total
  (bianca_school : ℕ)
  (bianca_home : ℕ)
  (lucas_school : ℕ)
  (lucas_home : ℕ)
  (h_bianca_school : bianca_school = 22)
  (h_bianca_home : bianca_home = 19)
  (h_lucas_school : lucas_school = 10)
  (h_lucas_home : lucas_home = 35) :
  bianca_school + bianca_home + lucas_school + lucas_home = 86 := 
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_drawing_time_total_l1786_178657


namespace NUMINAMATH_GPT_Shell_Ratio_l1786_178603

-- Definitions of the number of shells collected by Alan, Ben, and Laurie.
variable (A B L : ℕ)

-- Hypotheses based on the given conditions:
-- 1. Alan collected four times as many shells as Ben did.
-- 2. Laurie collected 36 shells.
-- 3. Alan collected 48 shells.
theorem Shell_Ratio (h1 : A = 4 * B) (h2 : L = 36) (h3 : A = 48) : B / Nat.gcd B L = 1 ∧ L / Nat.gcd B L = 3 :=
by
  sorry

end NUMINAMATH_GPT_Shell_Ratio_l1786_178603


namespace NUMINAMATH_GPT_range_of_a_l1786_178643

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 1 / x

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem range_of_a (a : ℝ) :
  is_increasing_on (λ x => x^2 + a * x + 1 / x) (Set.Ioi (1 / 2)) ↔ 3 ≤ a := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1786_178643


namespace NUMINAMATH_GPT_monotonically_increasing_a_range_l1786_178606

noncomputable def f (a x : ℝ) : ℝ := (a * x - 1) * Real.exp x

theorem monotonically_increasing_a_range :
  ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≥ 0) ↔ 1 ≤ a  :=
by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_a_range_l1786_178606


namespace NUMINAMATH_GPT_bettys_herb_garden_l1786_178662

theorem bettys_herb_garden :
  ∀ (basil oregano thyme rosemary total : ℕ),
    oregano = 2 * basil + 2 →
    thyme = 3 * basil - 3 →
    rosemary = (basil + thyme) / 2 →
    basil = 5 →
    total = basil + oregano + thyme + rosemary →
    total ≤ 50 →
    total = 37 :=
by
  intros basil oregano thyme rosemary total h_oregano h_thyme h_rosemary h_basil h_total h_le_total
  sorry

end NUMINAMATH_GPT_bettys_herb_garden_l1786_178662


namespace NUMINAMATH_GPT_joan_travel_time_correct_l1786_178697

noncomputable def joan_travel_time (distance rate : ℕ) (lunch_break bathroom_breaks : ℕ) : ℕ := 
  let driving_time := distance / rate
  let break_time := lunch_break + 2 * bathroom_breaks
  driving_time + break_time / 60

theorem joan_travel_time_correct : joan_travel_time 480 60 30 15 = 9 := by
  sorry

end NUMINAMATH_GPT_joan_travel_time_correct_l1786_178697


namespace NUMINAMATH_GPT_sequence_monotonically_decreasing_l1786_178619

theorem sequence_monotonically_decreasing (t : ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, a n = -↑n^2 + t * ↑n) →
  (∀ n : ℕ, a (n + 1) < a n) →
  t < 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sequence_monotonically_decreasing_l1786_178619


namespace NUMINAMATH_GPT_emily_wrong_questions_l1786_178691

variable (E F G H : ℕ)

theorem emily_wrong_questions (h1 : E + F + 4 = G + H) 
                             (h2 : E + H = F + G + 8) 
                             (h3 : G = 6) : 
                             E = 8 :=
sorry

end NUMINAMATH_GPT_emily_wrong_questions_l1786_178691


namespace NUMINAMATH_GPT_polynomial_remainder_l1786_178622

theorem polynomial_remainder :
  let f := X^2023 + 1
  let g := X^6 - X^4 + X^2 - 1
  ∃ (r : Polynomial ℤ), (r = -X^3 + 1) ∧ (∃ q : Polynomial ℤ, f = q * g + r) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1786_178622


namespace NUMINAMATH_GPT_percentage_needed_to_pass_l1786_178677

def MikeScore : ℕ := 212
def Shortfall : ℕ := 19
def MaxMarks : ℕ := 770

theorem percentage_needed_to_pass :
  (231.0 / (770.0 : ℝ)) * 100 = 30 := by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_percentage_needed_to_pass_l1786_178677


namespace NUMINAMATH_GPT_not_on_line_l1786_178671

-- Defining the point (0,20)
def pt : ℝ × ℝ := (0, 20)

-- Defining the line equation
def line (m b : ℝ) (p : ℝ × ℝ) : Prop := p.2 = m * p.1 + b

-- The proof problem stating that for all real numbers m and b, if m + b < 0, 
-- then the point (0, 20) cannot be on the line y = mx + b
theorem not_on_line (m b : ℝ) (h : m + b < 0) : ¬line m b pt := by
  sorry

end NUMINAMATH_GPT_not_on_line_l1786_178671


namespace NUMINAMATH_GPT_inequality_of_fractions_l1786_178613

theorem inequality_of_fractions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x / (x + y)) + (y / (y + z)) + (z / (z + x)) ≤ 2 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_of_fractions_l1786_178613


namespace NUMINAMATH_GPT_sin_2alpha_plus_sin_squared_l1786_178624

theorem sin_2alpha_plus_sin_squared (α : ℝ) (h : Real.tan α = 1 / 2) : Real.sin (2 * α) + Real.sin α ^ 2 = 1 :=
sorry

end NUMINAMATH_GPT_sin_2alpha_plus_sin_squared_l1786_178624


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1786_178614

variable {a_n : ℕ → ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∀ (m n k : ℕ), m < n → (n - m) = k → a_n n = a_n m + k * (a_n 1 - a_n 0)

theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a_n →
  a_n 2 = 5 →
  a_n 6 = 33 →
  a_n 3 + a_n 5 = 38 :=
by
  intros h_seq h_a2 h_a6
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1786_178614


namespace NUMINAMATH_GPT_ratio_of_volumes_l1786_178690

theorem ratio_of_volumes (A B : ℝ) (h : (3 / 4) * A = (2 / 3) * B) : A / B = 8 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_l1786_178690


namespace NUMINAMATH_GPT_tangent_line_equation_l1786_178652

theorem tangent_line_equation
  (x y : ℝ)
  (h₁ : x^2 + y^2 = 5)
  (hM : x = -1 ∧ y = 2) :
  x - 2 * y + 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l1786_178652


namespace NUMINAMATH_GPT_div_by_3_iff_n_form_l1786_178641

theorem div_by_3_iff_n_form (n : ℕ) : (3 ∣ (n * 2^n + 1)) ↔ (∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k + 2) :=
by
  sorry

end NUMINAMATH_GPT_div_by_3_iff_n_form_l1786_178641


namespace NUMINAMATH_GPT_price_relation_l1786_178698

-- Defining the conditions
variable (TotalPrice : ℕ) (NumberOfPens : ℕ)
variable (total_price_val : TotalPrice = 24) (number_of_pens_val : NumberOfPens = 16)

-- Statement of the problem
theorem price_relation (y x : ℕ) (h_y : y = 3 / 2) : y = 3 / 2 * x := 
  sorry

end NUMINAMATH_GPT_price_relation_l1786_178698


namespace NUMINAMATH_GPT_cos_double_angle_l1786_178699

theorem cos_double_angle (α : ℝ) (h : Real.sin (π + α) = 2 / 3) : Real.cos (2 * α) = 1 / 9 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_l1786_178699


namespace NUMINAMATH_GPT_cassidy_grounded_days_l1786_178686

-- Definitions for the conditions
def days_for_lying : Nat := 14
def extra_days_per_grade : Nat := 3
def grades_below_B : Nat := 4

-- Definition for the total days grounded
def total_days_grounded : Nat :=
  days_for_lying + extra_days_per_grade * grades_below_B

-- The theorem statement
theorem cassidy_grounded_days :
  total_days_grounded = 26 := by
  sorry

end NUMINAMATH_GPT_cassidy_grounded_days_l1786_178686


namespace NUMINAMATH_GPT_solve_system_l1786_178609

noncomputable def sqrt_cond (x y : ℝ) : Prop :=
  Real.sqrt ((3 * x - 2 * y) / (2 * x)) + Real.sqrt ((2 * x) / (3 * x - 2 * y)) = 2

noncomputable def quad_cond (x y : ℝ) : Prop :=
  x^2 - 18 = 2 * y * (4 * y - 9)

theorem solve_system (x y : ℝ) : sqrt_cond x y ∧ quad_cond x y ↔ (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 1.5) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1786_178609


namespace NUMINAMATH_GPT_L_shaped_figure_perimeter_is_14_l1786_178668

-- Define the side length of each square as a constant
def side_length : ℕ := 2

-- Define the horizontal base length
def base_length : ℕ := 3 * side_length

-- Define the height of the vertical stack
def vertical_stack_height : ℕ := 2 * side_length

-- Define the total perimeter of the "L" shaped figure
def L_shaped_figure_perimeter : ℕ :=
  base_length + side_length + vertical_stack_height + side_length + side_length + vertical_stack_height

-- The theorem that states the perimeter of the L-shaped figure is 14 units
theorem L_shaped_figure_perimeter_is_14 : L_shaped_figure_perimeter = 14 := sorry

end NUMINAMATH_GPT_L_shaped_figure_perimeter_is_14_l1786_178668


namespace NUMINAMATH_GPT_train_length_in_terms_of_james_cycle_l1786_178645

/-- Define the mathematical entities involved: L (train length), J (James's cycle length), T (train length per cycle) -/
theorem train_length_in_terms_of_james_cycle 
  (L J T : ℝ) 
  (h1 : 130 * J = L + 130 * T) 
  (h2 : 26 * J = L - 26 * T) 
    : L = 58 * J := 
by 
  sorry

end NUMINAMATH_GPT_train_length_in_terms_of_james_cycle_l1786_178645


namespace NUMINAMATH_GPT_functional_equation_solution_l1786_178626

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x) + f (f y) = 2 * y + f (x - y)) ↔ (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1786_178626


namespace NUMINAMATH_GPT_amount_allocated_to_food_l1786_178621

theorem amount_allocated_to_food (total_amount : ℝ) (household_ratio food_ratio misc_ratio : ℝ) 
  (h₁ : total_amount = 1800) (h₂ : household_ratio = 5) (h₃ : food_ratio = 4) (h₄ : misc_ratio = 1) :
  food_ratio / (household_ratio + food_ratio + misc_ratio) * total_amount = 720 :=
by
  sorry

end NUMINAMATH_GPT_amount_allocated_to_food_l1786_178621


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l1786_178651

variable {a b : ℝ}

noncomputable def A (a b : ℝ) := (a + b) / 2
noncomputable def B (a b : ℝ) := Real.sqrt (a * b)

theorem arithmetic_geometric_mean_inequality (h₀ : a > 0) (h₁ : b > 0) (h₂ : a ≠ b) : A a b > B a b := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l1786_178651


namespace NUMINAMATH_GPT_jane_buys_four_bagels_l1786_178675

-- Define Jane's 7-day breakfast choices
def number_of_items (b m : ℕ) := b + m = 7

-- Define the total weekly cost condition
def total_cost_divisible_by_100 (b : ℕ) := (90 * b + 40 * (7 - b)) % 100 = 0

-- The statement to prove
theorem jane_buys_four_bagels (b : ℕ) (m : ℕ) (h1 : number_of_items b m) (h2 : total_cost_divisible_by_100 b) : b = 4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_jane_buys_four_bagels_l1786_178675


namespace NUMINAMATH_GPT_largest_divisor_of_exp_and_linear_combination_l1786_178692

theorem largest_divisor_of_exp_and_linear_combination :
  ∃ x : ℕ, (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧ x = 18 :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_exp_and_linear_combination_l1786_178692


namespace NUMINAMATH_GPT_flat_terrain_length_l1786_178617

noncomputable def terrain_distance_equation (x y z : ℝ) : Prop :=
  (x + y + z = 11.5) ∧
  (x / 3 + y / 4 + z / 5 = 2.9) ∧
  (z / 3 + y / 4 + x / 5 = 3.1)

theorem flat_terrain_length (x y z : ℝ) 
  (h : terrain_distance_equation x y z) :
  y = 4 :=
sorry

end NUMINAMATH_GPT_flat_terrain_length_l1786_178617


namespace NUMINAMATH_GPT_min_product_ab_l1786_178644

theorem min_product_ab (a b : ℝ) (h : 20 * a * b = 13 * a + 14 * b) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  a * b = 1.82 :=
sorry

end NUMINAMATH_GPT_min_product_ab_l1786_178644


namespace NUMINAMATH_GPT_number_of_yellow_marbles_l1786_178635

theorem number_of_yellow_marbles (total_marbles blue_marbles red_marbles green_marbles yellow_marbles : ℕ)
    (h_total : total_marbles = 164) 
    (h_blue : blue_marbles = total_marbles / 2)
    (h_red : red_marbles = total_marbles / 4)
    (h_green : green_marbles = 27) :
    yellow_marbles = total_marbles - (blue_marbles + red_marbles + green_marbles) →
    yellow_marbles = 14 := by
  sorry

end NUMINAMATH_GPT_number_of_yellow_marbles_l1786_178635


namespace NUMINAMATH_GPT_abs_neg_two_eq_two_l1786_178660

theorem abs_neg_two_eq_two : |(-2 : ℤ)| = 2 := 
by 
  sorry

end NUMINAMATH_GPT_abs_neg_two_eq_two_l1786_178660


namespace NUMINAMATH_GPT_linear_equation_solution_l1786_178615

theorem linear_equation_solution (x y b : ℝ) (h1 : x - 2*y + b = 0) (h2 : y = (1/2)*x + b - 1) :
  b = 2 :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_solution_l1786_178615


namespace NUMINAMATH_GPT_min_value_of_m_l1786_178670

theorem min_value_of_m : (2 ∈ {x | ∃ (m : ℤ), x * (x - m) < 0}) → ∃ (m : ℤ), m = 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_m_l1786_178670


namespace NUMINAMATH_GPT_intersection_points_relation_l1786_178678

-- Suppressing noncomputable theory to focus on the structure
-- of the Lean statement rather than computability aspects.

noncomputable def intersection_points (k : ℕ) : ℕ :=
sorry -- This represents the function f(k)

axiom no_parallel (k : ℕ) : Prop
axiom no_three_intersect (k : ℕ) : Prop

theorem intersection_points_relation (k : ℕ) (h1 : no_parallel k) (h2 : no_three_intersect k) :
  intersection_points (k + 1) = intersection_points k + k :=
sorry

end NUMINAMATH_GPT_intersection_points_relation_l1786_178678


namespace NUMINAMATH_GPT_moles_HCl_combination_l1786_178688

-- Define the conditions:
def moles_HCl (C5H12O: ℕ) (H2O: ℕ) : ℕ :=
  if H2O = 18 then 18 else 0

-- The main statement to prove:
theorem moles_HCl_combination :
  moles_HCl 1 18 = 18 :=
sorry

end NUMINAMATH_GPT_moles_HCl_combination_l1786_178688


namespace NUMINAMATH_GPT_max_xy_of_perpendicular_l1786_178663

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, x - 1)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ := (y, 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 

theorem max_xy_of_perpendicular (x y : ℝ) 
  (h_perp : dot_product (vector_a x) (vector_b y) = 0) : xy ≤ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_max_xy_of_perpendicular_l1786_178663


namespace NUMINAMATH_GPT_matches_in_round1_group1_matches_in_round1_group2_matches_in_round2_l1786_178685

def num_teams_group1 : ℕ := 3
def num_teams_group2 : ℕ := 4

def num_matches_round1_group1 (n : ℕ) : ℕ := n * (n - 1) / 2
def num_matches_round1_group2 (n : ℕ) : ℕ := n * (n - 1) / 2

def num_matches_round2 (n1 n2 : ℕ) : ℕ := n1 * n2

theorem matches_in_round1_group1 : num_matches_round1_group1 num_teams_group1 = 3 := 
by
  -- Exact proof steps should be filled in here.
  sorry

theorem matches_in_round1_group2 : num_matches_round1_group2 num_teams_group2 = 6 := 
by
  -- Exact proof steps should be filled in here.
  sorry

theorem matches_in_round2 : num_matches_round2 num_teams_group1 num_teams_group2 = 12 := 
by
  -- Exact proof steps should be filled in here.
  sorry

end NUMINAMATH_GPT_matches_in_round1_group1_matches_in_round1_group2_matches_in_round2_l1786_178685


namespace NUMINAMATH_GPT_bezdikov_population_l1786_178656

variable (W M : ℕ) -- original number of women and men
variable (W_current M_current : ℕ) -- current number of women and men

theorem bezdikov_population (h1 : W = M + 30)
                          (h2 : W_current = W / 4)
                          (h3 : M_current = M - 196)
                          (h4 : W_current = M_current + 10) : W_current + M_current = 134 :=
by
  sorry

end NUMINAMATH_GPT_bezdikov_population_l1786_178656


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_inequality_l1786_178608

theorem necessary_but_not_sufficient_for_inequality : 
  ∀ x : ℝ, (-2 < x ∧ x < 4) → (x < 5) ∧ (¬(x < 5) → (-2 < x ∧ x < 4) ) :=
by 
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_inequality_l1786_178608


namespace NUMINAMATH_GPT_simplify_expression_l1786_178693

theorem simplify_expression (x : ℝ) (h : x ≠ 0) : 
    (Real.sqrt (4 + ( (x^3 - 2) / (3 * x) ) ^ 2)) = 
    (Real.sqrt (x^6 - 4 * x^3 + 36 * x^2 + 4) / (3 * x)) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1786_178693


namespace NUMINAMATH_GPT_fraction_multiplication_l1786_178665

theorem fraction_multiplication :
  ((2 / 5) * (5 / 7) * (7 / 3) * (3 / 8) = 1 / 4) :=
sorry

end NUMINAMATH_GPT_fraction_multiplication_l1786_178665


namespace NUMINAMATH_GPT_solution_set_for_f_ge_0_range_of_a_l1786_178680

def f (x : ℝ) : ℝ := |3 * x + 1| - |2 * x + 2|

theorem solution_set_for_f_ge_0 : {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -3/5} ∪ {x : ℝ | x ≥ 1} :=
sorry

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x - |x + 1| ≤ |a + 1|) ↔ (a ≤ -3 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_GPT_solution_set_for_f_ge_0_range_of_a_l1786_178680


namespace NUMINAMATH_GPT_quadratic_equation_solution_l1786_178674

theorem quadratic_equation_solution (m : ℝ) (h : m ≠ 1) : 
  (m^2 - 3 * m + 2 = 0) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_solution_l1786_178674


namespace NUMINAMATH_GPT_range_of_a_l1786_178616

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x + a| + |x - 2| + a < 2010) ↔ a < 1006 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1786_178616


namespace NUMINAMATH_GPT_third_root_of_polynomial_l1786_178618

theorem third_root_of_polynomial (a b : ℚ) 
  (h₁ : a*(-1)^3 + (a + 3*b)*(-1)^2 + (2*b - 4*a)*(-1) + (10 - a) = 0)
  (h₂ : a*(4)^3 + (a + 3*b)*(4)^2 + (2*b - 4*a)*(4) + (10 - a) = 0) :
  ∃ (r : ℚ), r = -24 / 19 :=
by
  sorry

end NUMINAMATH_GPT_third_root_of_polynomial_l1786_178618


namespace NUMINAMATH_GPT_smallest_n_is_29_l1786_178667

noncomputable def smallest_possible_n (r g b : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm (10 * r) (16 * g)) (18 * b) / 25

theorem smallest_n_is_29 (r g b : ℕ) (h : 10 * r = 16 * g ∧ 16 * g = 18 * b) :
  smallest_possible_n r g b = 29 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_is_29_l1786_178667


namespace NUMINAMATH_GPT_find_multiplier_l1786_178602

theorem find_multiplier (n m : ℕ) (h1 : 2 * n = (26 - n) + 19) (h2 : n = 15) : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_multiplier_l1786_178602


namespace NUMINAMATH_GPT_arithmetic_sqrt_of_9_l1786_178605

def arithmetic_sqrt (n : ℕ) : ℕ :=
  Nat.sqrt n

theorem arithmetic_sqrt_of_9 : arithmetic_sqrt 9 = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sqrt_of_9_l1786_178605


namespace NUMINAMATH_GPT_binary_to_decimal_l1786_178648

theorem binary_to_decimal : 
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0) = 54 :=
by 
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l1786_178648


namespace NUMINAMATH_GPT_correct_operation_l1786_178696

-- Defining the options as hypotheses
variable {a b : ℕ}

theorem correct_operation (hA : 4*a + 3*b ≠ 7*a*b)
    (hB : a^4 * a^3 = a^7)
    (hC : (3*a)^3 ≠ 9*a^3)
    (hD : a^6 / a^2 ≠ a^3) :
    a^4 * a^3 = a^7 := by
  sorry

end NUMINAMATH_GPT_correct_operation_l1786_178696


namespace NUMINAMATH_GPT_percent_paddyfield_warblers_l1786_178689

variable (B : ℝ) -- The total number of birds.
variable (N_h : ℝ := 0.30 * B) -- Number of hawks.
variable (N_non_hawks : ℝ := 0.70 * B) -- Number of non-hawks.
variable (N_not_hpwk : ℝ := 0.35 * B) -- 35% are not hawks, paddyfield-warblers, or kingfishers.
variable (N_hpwk : ℝ := 0.65 * B) -- 65% are hawks, paddyfield-warblers, or kingfishers.
variable (P : ℝ) -- Percentage of non-hawks that are paddyfield-warblers, to be found.
variable (N_pw : ℝ := P * 0.70 * B) -- Number of paddyfield-warblers.
variable (N_k : ℝ := 0.25 * N_pw) -- Number of kingfishers.

theorem percent_paddyfield_warblers (h_eq : N_h + N_pw + N_k = 0.65 * B) : P = 0.5714 := by
  sorry

end NUMINAMATH_GPT_percent_paddyfield_warblers_l1786_178689


namespace NUMINAMATH_GPT_markup_rate_correct_l1786_178681

noncomputable def selling_price : ℝ := 10.00
noncomputable def profit_percentage : ℝ := 0.20
noncomputable def expenses_percentage : ℝ := 0.15
noncomputable def cost (S : ℝ) : ℝ := S - (profit_percentage * S + expenses_percentage * S)
noncomputable def markup_rate (S C : ℝ) : ℝ := (S - C) / C * 100

theorem markup_rate_correct :
  markup_rate selling_price (cost selling_price) = 53.85 := 
by
  sorry

end NUMINAMATH_GPT_markup_rate_correct_l1786_178681


namespace NUMINAMATH_GPT_sqrt_expression_eq_1720_l1786_178647

theorem sqrt_expression_eq_1720 : Real.sqrt ((43 * 42 * 41 * 40) + 1) = 1720 := by
  sorry

end NUMINAMATH_GPT_sqrt_expression_eq_1720_l1786_178647


namespace NUMINAMATH_GPT_base_addition_l1786_178636

theorem base_addition (R1 R3 : ℕ) (F1 F2 : ℚ)
    (hF1_baseR1 : F1 = 45 / (R1^2 - 1))
    (hF2_baseR1 : F2 = 54 / (R1^2 - 1))
    (hF1_baseR3 : F1 = 36 / (R3^2 - 1))
    (hF2_baseR3 : F2 = 63 / (R3^2 - 1)) :
  R1 + R3 = 20 :=
sorry

end NUMINAMATH_GPT_base_addition_l1786_178636


namespace NUMINAMATH_GPT_num_of_terms_in_arith_seq_l1786_178655

-- Definitions of the conditions
def a : Int := -5 -- Start of the arithmetic sequence
def l : Int := 85 -- End of the arithmetic sequence
def d : Nat := 5  -- Common difference

-- The theorem that needs to be proved
theorem num_of_terms_in_arith_seq : (l - a) / d + 1 = 19 := sorry

end NUMINAMATH_GPT_num_of_terms_in_arith_seq_l1786_178655


namespace NUMINAMATH_GPT_range_of_m_l1786_178658

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), x > -5 ∧ x ≤ m + 1) ∧ (∀ x, x > -5 → x ≤ m + 1 → x = -4 ∨ x = -3 ∨ x = -2) →
  (-3 ≤ m ∧ m < -2) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1786_178658


namespace NUMINAMATH_GPT_possible_sets_l1786_178632

theorem possible_sets 
  (A B C : Set ℕ) 
  (U : Set ℕ := {a, b, c, d, e, f}) 
  (H1 : A ∪ B ∪ C = U) 
  (H2 : A ∩ B = {a, b, c, d}) 
  (H3 : c ∈ A ∩ B ∩ C) : 
  ∃ (n : ℕ), n = 200 :=
sorry

end NUMINAMATH_GPT_possible_sets_l1786_178632


namespace NUMINAMATH_GPT_rita_bought_5_dresses_l1786_178634

def pants_cost := 3 * 12
def jackets_cost := 4 * 30
def total_cost_pants_jackets := pants_cost + jackets_cost
def amount_spent := 400 - 139
def total_cost_dresses := amount_spent - total_cost_pants_jackets - 5
def number_of_dresses := total_cost_dresses / 20

theorem rita_bought_5_dresses : number_of_dresses = 5 :=
by sorry

end NUMINAMATH_GPT_rita_bought_5_dresses_l1786_178634


namespace NUMINAMATH_GPT_line_equation_l1786_178639

theorem line_equation {a b c : ℝ} (x : ℝ) (y : ℝ)
  (point : ∃ p: ℝ × ℝ, p = (-1, 0))
  (perpendicular : ∀ k: ℝ, k = 1 → 
    ∀ m: ℝ, m = -1 → 
      ∀ b1: ℝ, b1 = 0 → 
        ∀ x1: ℝ, x1 = -1 →
          ∀ y1: ℝ, y1 = 0 →
            ∀ l: ℝ, l = b1 + k * (x1 - (-1)) + m * (y1 - 0) → 
              x - y + 1 = 0) :
  x - y + 1 = 0 :=
sorry

end NUMINAMATH_GPT_line_equation_l1786_178639


namespace NUMINAMATH_GPT_coordinates_of_points_l1786_178631

theorem coordinates_of_points
  (R : ℝ) (a b : ℝ)
  (hR : R = 10)
  (h_area : 1/2 * a * b = 600)
  (h_a_gt_b : a > b) :
  (a, 0) = (40, 0) ∧ (0, b) = (0, 30) ∧ (16, 18) = (16, 18) :=
  sorry

end NUMINAMATH_GPT_coordinates_of_points_l1786_178631


namespace NUMINAMATH_GPT_katie_total_marbles_l1786_178684

def pink_marbles := 13
def orange_marbles := pink_marbles - 9
def purple_marbles := 4 * orange_marbles
def blue_marbles := 2 * purple_marbles
def total_marbles := pink_marbles + orange_marbles + purple_marbles + blue_marbles

theorem katie_total_marbles : total_marbles = 65 := 
by
  -- The proof is omitted here.
  sorry

end NUMINAMATH_GPT_katie_total_marbles_l1786_178684


namespace NUMINAMATH_GPT_pqrs_product_l1786_178694

noncomputable def product_of_area_and_perimeter :=
  let P := (1, 3)
  let Q := (4, 4)
  let R := (3, 1)
  let S := (0, 0)
  let side_length := Real.sqrt ((1 - 0)^2 * 4 + (3 - 0)^2 * 4)
  let area := side_length ^ 2
  let perimeter := 4 * side_length
  area * perimeter

theorem pqrs_product : product_of_area_and_perimeter = 208 * Real.sqrt 52 := 
  by 
    sorry

end NUMINAMATH_GPT_pqrs_product_l1786_178694


namespace NUMINAMATH_GPT_kindergarteners_line_up_probability_l1786_178672

theorem kindergarteners_line_up_probability :
  let total_line_up := Nat.choose 20 9
  let first_scenario := Nat.choose 14 9
  let second_scenario_single := Nat.choose 13 8
  let second_scenario := 6 * second_scenario_single
  let valid_arrangements := first_scenario + second_scenario
  valid_arrangements / total_line_up = 9724 / 167960 := by
  sorry

end NUMINAMATH_GPT_kindergarteners_line_up_probability_l1786_178672


namespace NUMINAMATH_GPT_problem_2011_Mentougou_l1786_178620

theorem problem_2011_Mentougou 
  (f : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (H2 : ∀ x : ℝ, 0 < x → 0 < f x) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :=
sorry

end NUMINAMATH_GPT_problem_2011_Mentougou_l1786_178620


namespace NUMINAMATH_GPT_smallest_positive_integer_divisible_by_15_16_18_l1786_178682

theorem smallest_positive_integer_divisible_by_15_16_18 : 
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) ∧ n = 720 := 
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_divisible_by_15_16_18_l1786_178682
