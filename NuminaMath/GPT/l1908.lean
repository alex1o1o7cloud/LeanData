import Mathlib

namespace NUMINAMATH_GPT_math_problem_l1908_190893

noncomputable def f : ℝ → ℝ := sorry

theorem math_problem (h_decreasing : ∀ x y : ℝ, 2 < x → x < y → f y < f x)
  (h_even : ∀ x : ℝ, f (-x + 2) = f (x + 2)) :
  f 2 < f 3 ∧ f 3 < f 0 ∧ f 0 < f (-1) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1908_190893


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l1908_190867

variables {a : ℕ → ℝ} -- 'a' is a sequence of positive real numbers
variable {q : ℝ} -- 'q' is the common ratio of the geometric sequence

-- Definition of a geometric sequence with common ratio 'q'
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Condition from the problem statement
def condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  2 * a 5 - 3 * a 4 = 2 * a 3

-- Main theorem: If the sequence {a_n} is a geometric sequence with positive terms and satisfies the condition, 
-- then the common ratio q = 2
theorem common_ratio_geometric_sequence :
  (∀ n, 0 < a n) → geometric_sequence a q → condition a q → q = 2 :=
by
  intro h_pos h_geom h_cond
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l1908_190867


namespace NUMINAMATH_GPT_length_PC_l1908_190881

-- Define lengths of the sides of triangle ABC.
def AB := 10
def BC := 8
def CA := 7

-- Define the similarity condition
def similar_triangles (PA PC : ℝ) : Prop :=
  PA / PC = AB / CA

-- Define the extension of side BC to point P
def extension_condition (PA PC : ℝ) : Prop :=
  PA = PC + BC

theorem length_PC (PC : ℝ) (PA : ℝ) :
  similar_triangles PA PC → extension_condition PA PC → PC = 56 / 3 :=
by
  intro h_sim h_ext
  sorry

end NUMINAMATH_GPT_length_PC_l1908_190881


namespace NUMINAMATH_GPT_fraction_product_eq_l1908_190831
-- Import the necessary library

-- Define the fractions and the product
def fraction_product : ℚ :=
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8)

-- State the theorem we want to prove
theorem fraction_product_eq : fraction_product = 3 / 8 := 
sorry

end NUMINAMATH_GPT_fraction_product_eq_l1908_190831


namespace NUMINAMATH_GPT_number_of_panes_l1908_190801

theorem number_of_panes (length width total_area : ℕ) (h_length : length = 12) (h_width : width = 8) (h_total_area : total_area = 768) :
  total_area / (length * width) = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_panes_l1908_190801


namespace NUMINAMATH_GPT_speed_equation_l1908_190847

theorem speed_equation
  (dA dB : ℝ)
  (sB : ℝ)
  (sA : ℝ)
  (time_difference : ℝ)
  (h1 : dA = 800)
  (h2 : dB = 400)
  (h3 : sA = 1.2 * sB)
  (h4 : time_difference = 4) :
  (dA / sA - dB / sB = time_difference) :=
by
  sorry

end NUMINAMATH_GPT_speed_equation_l1908_190847


namespace NUMINAMATH_GPT_max_diff_consecutive_slightly_unlucky_l1908_190879

def is_slightly_unlucky (n : ℕ) : Prop := (n.digits 10).sum % 13 = 0

theorem max_diff_consecutive_slightly_unlucky :
  ∃ n m : ℕ, is_slightly_unlucky n ∧ is_slightly_unlucky m ∧ (m > n) ∧ ∀ k, (is_slightly_unlucky k ∧ k > n ∧ k < m) → false → (m - n) = 79 :=
sorry

end NUMINAMATH_GPT_max_diff_consecutive_slightly_unlucky_l1908_190879


namespace NUMINAMATH_GPT_f2011_eq_two_l1908_190876

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom periodicity_eqn : ∀ x : ℝ, f (x + 6) = f (x) + f 3
axiom f1_eq_two : f 1 = 2

theorem f2011_eq_two : f 2011 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_f2011_eq_two_l1908_190876


namespace NUMINAMATH_GPT_gcd_difference_5610_210_10_l1908_190840

theorem gcd_difference_5610_210_10 : Int.gcd 5610 210 - 10 = 20 := by
  sorry

end NUMINAMATH_GPT_gcd_difference_5610_210_10_l1908_190840


namespace NUMINAMATH_GPT_total_quantities_l1908_190854

theorem total_quantities (n S S₃ S₂ : ℕ) (h₁ : S = 6 * n) (h₂ : S₃ = 4 * 3) (h₃ : S₂ = 33 * 2) (h₄ : S = S₃ + S₂) : n = 13 :=
by
  sorry

end NUMINAMATH_GPT_total_quantities_l1908_190854


namespace NUMINAMATH_GPT_remainder_276_l1908_190862

theorem remainder_276 (y : ℤ) (k : ℤ) (hk : y = 23 * k + 19) : y % 276 = 180 :=
sorry

end NUMINAMATH_GPT_remainder_276_l1908_190862


namespace NUMINAMATH_GPT_part_I_part_II_l1908_190808

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |x - 1|

theorem part_I (a : ℝ) (h : ∃ x : ℝ, f x a ≤ 2 - |x - 1|) : 0 ≤ a ∧ a ≤ 4 := 
sorry

theorem part_II (a : ℝ) (h₁ : a < 2) (h₂ : ∀ x : ℝ, f x a ≥ 3) : a = -4 := 
sorry

end NUMINAMATH_GPT_part_I_part_II_l1908_190808


namespace NUMINAMATH_GPT_sum_lent_eq_1100_l1908_190853

def interest_rate : ℚ := 6 / 100

def period : ℕ := 8

def interest_amount (P : ℚ) : ℚ :=
  period * interest_rate * P

def total_interest_eq_principal_minus_572 (P: ℚ) : Prop :=
  interest_amount P = P - 572

theorem sum_lent_eq_1100 : ∃ P : ℚ, total_interest_eq_principal_minus_572 P ∧ P = 1100 :=
by
  use 1100
  sorry

end NUMINAMATH_GPT_sum_lent_eq_1100_l1908_190853


namespace NUMINAMATH_GPT_time_in_still_water_l1908_190869

-- Define the conditions
variable (S x y : ℝ)
axiom condition1 : S / (x + y) = 6
axiom condition2 : S / (x - y) = 8

-- Define the proof statement
theorem time_in_still_water : S / x = 48 / 7 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_time_in_still_water_l1908_190869


namespace NUMINAMATH_GPT_fractions_inequality_l1908_190838

variable {a b c d : ℝ}
variable (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0)

theorem fractions_inequality : 
  (a > b) → (b > 0) → (c < d) → (d < 0) → (a / d < b / c) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_fractions_inequality_l1908_190838


namespace NUMINAMATH_GPT_repeating_decimal_product_l1908_190809

-- Define the repeating decimal 0.\overline{137} as a fraction
def repeating_decimal_137 : ℚ := 137 / 999

-- Define the repeating decimal 0.\overline{6} as a fraction
def repeating_decimal_6 : ℚ := 2 / 3

-- The problem is to prove that the product of these fractions is 274 / 2997
theorem repeating_decimal_product : repeating_decimal_137 * repeating_decimal_6 = 274 / 2997 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_product_l1908_190809


namespace NUMINAMATH_GPT_angle_is_50_l1908_190868

-- Define the angle, supplement, and complement
def angle (x : ℝ) := x
def supplement (x : ℝ) := 180 - x
def complement (x : ℝ) := 90 - x
def condition (x : ℝ) := supplement x = 3 * (complement x) + 10

theorem angle_is_50 :
  ∃ x : ℝ, condition x ∧ x = 50 :=
by
  -- Here we show the existence of x that satisfies the condition and is equal to 50
  sorry

end NUMINAMATH_GPT_angle_is_50_l1908_190868


namespace NUMINAMATH_GPT_rope_fold_length_l1908_190858

theorem rope_fold_length (L : ℝ) (hL : L = 1) :
  (L / 2 / 2 / 2) = (1 / 8) :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_rope_fold_length_l1908_190858


namespace NUMINAMATH_GPT_bridge_extension_length_l1908_190857

theorem bridge_extension_length (width_of_river length_of_existing_bridge additional_length_needed : ℕ)
  (h1 : width_of_river = 487)
  (h2 : length_of_existing_bridge = 295)
  (h3 : additional_length_needed = width_of_river - length_of_existing_bridge) :
  additional_length_needed = 192 :=
by {
  -- The steps of the proof would go here, but we use sorry for now.
  sorry
}

end NUMINAMATH_GPT_bridge_extension_length_l1908_190857


namespace NUMINAMATH_GPT_tomato_plant_relationship_l1908_190815

theorem tomato_plant_relationship :
  ∃ (T1 T2 T3 : ℕ), T1 = 24 ∧ T3 = T2 + 2 ∧ T1 + T2 + T3 = 60 ∧ T1 - T2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_tomato_plant_relationship_l1908_190815


namespace NUMINAMATH_GPT_harmonic_mean_is_54_div_11_l1908_190811

-- Define lengths of sides
def a : ℕ := 3
def b : ℕ := 6
def c : ℕ := 9

-- Define the harmonic mean calculation function
def harmonic_mean (x y z : ℕ) : ℚ :=
  let reciprocals_sum : ℚ := (1 / x + 1 / y + 1 / z)
  let average_reciprocal : ℚ := reciprocals_sum / 3
  1 / average_reciprocal

-- Prove that the harmonic mean of the given lengths is 54/11
theorem harmonic_mean_is_54_div_11 : harmonic_mean a b c = 54 / 11 := by
  sorry

end NUMINAMATH_GPT_harmonic_mean_is_54_div_11_l1908_190811


namespace NUMINAMATH_GPT_circle_area_l1908_190899

theorem circle_area (C : ℝ) (hC : C = 31.4) : 
  ∃ (A : ℝ), A = 246.49 / π := 
by
  sorry -- proof not required

end NUMINAMATH_GPT_circle_area_l1908_190899


namespace NUMINAMATH_GPT_quadratic_equation_proof_l1908_190833

def is_quadratic_equation (eqn : String) : Prop :=
  eqn = "x^2 + 2x - 1 = 0"

theorem quadratic_equation_proof :
  is_quadratic_equation "x^2 + 2x - 1 = 0" :=
sorry

end NUMINAMATH_GPT_quadratic_equation_proof_l1908_190833


namespace NUMINAMATH_GPT_min_ab_l1908_190875

theorem min_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1 / a + 1 / b = 1) : ab = 4 :=
  sorry

end NUMINAMATH_GPT_min_ab_l1908_190875


namespace NUMINAMATH_GPT_minimum_boxes_to_eliminate_50_percent_chance_l1908_190829

def total_boxes : Nat := 30
def high_value_boxes : Nat := 6
def minimum_boxes_to_eliminate (total_boxes high_value_boxes : Nat) : Nat :=
  total_boxes - high_value_boxes - high_value_boxes

theorem minimum_boxes_to_eliminate_50_percent_chance :
  minimum_boxes_to_eliminate total_boxes high_value_boxes = 18 :=
by
  sorry

end NUMINAMATH_GPT_minimum_boxes_to_eliminate_50_percent_chance_l1908_190829


namespace NUMINAMATH_GPT_Billy_current_age_l1908_190880

variable (B : ℕ)

theorem Billy_current_age 
  (h1 : ∃ B, 4 * B - B = 12) : B = 4 := by
  sorry

end NUMINAMATH_GPT_Billy_current_age_l1908_190880


namespace NUMINAMATH_GPT_train_pass_station_time_l1908_190890

-- Define the lengths of the train and station
def length_train : ℕ := 250
def length_station : ℕ := 200

-- Define the speed of the train in km/hour
def speed_kmh : ℕ := 36

-- Convert the speed to meters per second
def speed_mps : ℕ := speed_kmh * 1000 / 3600

-- Calculate the total distance the train needs to cover
def total_distance : ℕ := length_train + length_station

-- Define the expected time to pass the station
def expected_time : ℕ := 45

-- State the theorem that needs to be proven
theorem train_pass_station_time :
  total_distance / speed_mps = expected_time := by
  sorry

end NUMINAMATH_GPT_train_pass_station_time_l1908_190890


namespace NUMINAMATH_GPT_correct_conclusion_l1908_190889

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 2 else n * 2^n

theorem correct_conclusion (n : ℕ) (h₁ : ∀ k : ℕ, k > 0 → a_n (k + 1) - 2 * a_n k = 2^(k + 1)) :
  a_n n = n * 2 ^ n :=
by
  sorry

end NUMINAMATH_GPT_correct_conclusion_l1908_190889


namespace NUMINAMATH_GPT_Mario_savings_percentage_l1908_190895

theorem Mario_savings_percentage 
  (P : ℝ) -- Normal price of a single ticket 
  (h_campaign : 5 * P = 3 * P) -- Campaign condition: 5 tickets for the price of 3
  : (2 * P) / (5 * P) * 100 = 40 := 
by
  -- Below this, we would write the actual automated proof, but we leave it as sorry.
  sorry

end NUMINAMATH_GPT_Mario_savings_percentage_l1908_190895


namespace NUMINAMATH_GPT_range_of_m_l1908_190812

open Real

noncomputable def complex_modulus_log_condition (m : ℝ) : Prop :=
  Complex.abs (Complex.log (m : ℂ) / Complex.log 2 + Complex.I * 4) ≤ 5

theorem range_of_m (m : ℝ) (h : complex_modulus_log_condition m) : 
  (1 / 8 : ℝ) ≤ m ∧ m ≤ (8 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1908_190812


namespace NUMINAMATH_GPT_algorithm_find_GCD_Song_Yuan_l1908_190800

theorem algorithm_find_GCD_Song_Yuan :
  (∀ method, method = "continuous subtraction" → method_finds_GCD_Song_Yuan) :=
sorry

end NUMINAMATH_GPT_algorithm_find_GCD_Song_Yuan_l1908_190800


namespace NUMINAMATH_GPT_complement_of_M_in_U_l1908_190851

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U :
  U \ M = {3, 5, 6} := by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l1908_190851


namespace NUMINAMATH_GPT_sarah_saves_5_dollars_l1908_190828

noncomputable def price_per_pair : ℕ := 40

noncomputable def promotion_A_price (n : ℕ) : ℕ :=
if n % 2 = 0 then price_per_pair * n / 2 else price_per_pair

noncomputable def promotion_B_price (n : ℕ) : ℕ :=
if n % 2 = 0 then price_per_pair * n - (15 * (n / 2)) else price_per_pair

noncomputable def total_price_promotion_A : ℕ :=
price_per_pair + (price_per_pair / 2)

noncomputable def total_price_promotion_B : ℕ :=
price_per_pair + (price_per_pair - 15)

theorem sarah_saves_5_dollars : total_price_promotion_B - total_price_promotion_A = 5 :=
by
  rw [total_price_promotion_B, total_price_promotion_A]
  norm_num
  sorry

end NUMINAMATH_GPT_sarah_saves_5_dollars_l1908_190828


namespace NUMINAMATH_GPT_percentage_number_l1908_190885

theorem percentage_number (b : ℕ) (h : b = 100) : (320 * b / 100) = 320 :=
by
  sorry

end NUMINAMATH_GPT_percentage_number_l1908_190885


namespace NUMINAMATH_GPT_quad_side_difference_l1908_190834

theorem quad_side_difference (a b c d s x y : ℝ)
  (h1 : a = 80) (h2 : b = 100) (h3 : c = 150) (h4 : d = 120)
  (semiperimeter : s = (a + b + c + d) / 2)
  (h5 : x + y = c) 
  (h6 : (|x - y| = 30)) : 
  |x - y| = 30 :=
sorry

end NUMINAMATH_GPT_quad_side_difference_l1908_190834


namespace NUMINAMATH_GPT_problem_I_problem_II_l1908_190850

noncomputable def f (x m : ℝ) : ℝ := |x + m^2| + |x - 2*m - 3|

theorem problem_I (x m : ℝ) : f x m ≥ 2 :=
by 
  sorry

theorem problem_II (m : ℝ) : f 2 m ≤ 16 ↔ -3 ≤ m ∧ m ≤ Real.sqrt 14 - 1 :=
by 
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l1908_190850


namespace NUMINAMATH_GPT_binom_factorial_eq_120_factorial_l1908_190897

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binom_factorial_eq_120_factorial : (factorial (binomial 10 3)) = factorial 120 := by
  sorry

end NUMINAMATH_GPT_binom_factorial_eq_120_factorial_l1908_190897


namespace NUMINAMATH_GPT_oak_taller_than_shortest_l1908_190820

noncomputable def pine_tree_height : ℚ := 14 + 1 / 2
noncomputable def elm_tree_height : ℚ := 13 + 1 / 3
noncomputable def oak_tree_height : ℚ := 19 + 1 / 2

theorem oak_taller_than_shortest : 
  oak_tree_height - elm_tree_height = 6 + 1 / 6 := 
  sorry

end NUMINAMATH_GPT_oak_taller_than_shortest_l1908_190820


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1908_190870

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1908_190870


namespace NUMINAMATH_GPT_remainder_45_to_15_l1908_190844

theorem remainder_45_to_15 : ∀ (N : ℤ) (k : ℤ), N = 45 * k + 31 → N % 15 = 1 :=
by
  intros N k h
  sorry

end NUMINAMATH_GPT_remainder_45_to_15_l1908_190844


namespace NUMINAMATH_GPT_tiffany_bags_on_monday_l1908_190898

theorem tiffany_bags_on_monday : 
  ∃ M : ℕ, M = 8 ∧ ∃ T : ℕ, T = 7 ∧ M = T + 1 :=
by
  sorry

end NUMINAMATH_GPT_tiffany_bags_on_monday_l1908_190898


namespace NUMINAMATH_GPT_quad_eq_pos_neg_root_l1908_190877

theorem quad_eq_pos_neg_root (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ + x₂ = 2 ∧ x₁ * x₂ = a + 1) ↔ a < -1 :=
by sorry

end NUMINAMATH_GPT_quad_eq_pos_neg_root_l1908_190877


namespace NUMINAMATH_GPT_probability_longer_piece_l1908_190856

theorem probability_longer_piece {x y : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) :
  (∃ (p : ℝ), p = 2 / (x * y + 1)) :=
by
  sorry

end NUMINAMATH_GPT_probability_longer_piece_l1908_190856


namespace NUMINAMATH_GPT_find_positive_integer_n_l1908_190817

theorem find_positive_integer_n (n : ℕ) (hpos : 0 < n) : 
  (n + 1) ∣ (2 * n^2 + 5 * n) ↔ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integer_n_l1908_190817


namespace NUMINAMATH_GPT_average_marks_l1908_190894

-- Define the conditions
variables (M P C : ℕ)
axiom condition1 : M + P = 30
axiom condition2 : C = P + 20

-- Define the target statement
theorem average_marks : (M + C) / 2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_l1908_190894


namespace NUMINAMATH_GPT_borrowed_movie_price_correct_l1908_190823

def ticket_price : ℝ := 5.92
def number_of_tickets : ℕ := 2
def total_paid : ℝ := 20.00
def change_received : ℝ := 1.37
def tickets_cost : ℝ := number_of_tickets * ticket_price
def total_spent : ℝ := total_paid - change_received
def borrowed_movie_cost : ℝ := total_spent - tickets_cost

theorem borrowed_movie_price_correct : borrowed_movie_cost = 6.79 := by
  sorry

end NUMINAMATH_GPT_borrowed_movie_price_correct_l1908_190823


namespace NUMINAMATH_GPT_find_digit_D_l1908_190873

theorem find_digit_D (A B C D : ℕ)
  (h_add : 100 + 10 * A + B + 100 * C + 10 * A + A = 100 * D + 10 * A + B)
  (h_sub : 100 + 10 * A + B - (100 * C + 10 * A + A) = 100 + 10 * A) :
  D = 1 :=
by
  -- Since we're skipping the proof and focusing on the statement only
  sorry

end NUMINAMATH_GPT_find_digit_D_l1908_190873


namespace NUMINAMATH_GPT_last_even_distribution_l1908_190882

theorem last_even_distribution (n : ℕ) (h : n = 590490) :
  ∃ k : ℕ, (k ≤ n ∧ (n = 3^k + 3^k + 3^k) ∧ (∀ m : ℕ, m < k → ¬(n = 3^m + 3^m + 3^m))) ∧ k = 1 := 
by 
  sorry

end NUMINAMATH_GPT_last_even_distribution_l1908_190882


namespace NUMINAMATH_GPT_counterexample_not_prime_implies_prime_l1908_190883

theorem counterexample_not_prime_implies_prime (n : ℕ) (h₁ : ¬Nat.Prime n) (h₂ : n = 27) : ¬Nat.Prime (n - 2) :=
by
  sorry

end NUMINAMATH_GPT_counterexample_not_prime_implies_prime_l1908_190883


namespace NUMINAMATH_GPT_part1_solution_set_part2_inequality_l1908_190836

-- Part (1)
theorem part1_solution_set (x : ℝ) : |x| < 2 * x - 1 ↔ 1 < x := by
  sorry

-- Part (2)
theorem part2_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + 2 * b + c = 1) :
  (1 / (a + b)) + (1 / (b + c)) ≥ 4 := by
  sorry

end NUMINAMATH_GPT_part1_solution_set_part2_inequality_l1908_190836


namespace NUMINAMATH_GPT_constant_sequence_if_and_only_if_arith_geo_progression_l1908_190827

/-- A sequence a_n is both an arithmetic and geometric progression if and only if it is constant --/
theorem constant_sequence_if_and_only_if_arith_geo_progression (a : ℕ → ℝ) :
  (∃ q d : ℝ, (∀ n : ℕ, a (n+1) - a n = d) ∧ (∀ n : ℕ, a n = a 0 * q ^ n)) ↔ (∃ c : ℝ, ∀ n : ℕ, a n = c) := 
sorry

end NUMINAMATH_GPT_constant_sequence_if_and_only_if_arith_geo_progression_l1908_190827


namespace NUMINAMATH_GPT_differential_savings_is_4830_l1908_190887

-- Defining the conditions
def initial_tax_rate : ℝ := 0.42
def new_tax_rate : ℝ := 0.28
def annual_income : ℝ := 34500

-- Defining the calculation of tax before and after the tax rate change
def tax_before : ℝ := annual_income * initial_tax_rate
def tax_after : ℝ := annual_income * new_tax_rate

-- Defining the differential savings
def differential_savings : ℝ := tax_before - tax_after

-- Statement asserting that the differential savings is $4830
theorem differential_savings_is_4830 : differential_savings = 4830 := by sorry

end NUMINAMATH_GPT_differential_savings_is_4830_l1908_190887


namespace NUMINAMATH_GPT_canoe_total_weight_calculation_canoe_maximum_weight_limit_l1908_190842

def canoe_max_people : ℕ := 8
def people_with_pets_ratio : ℚ := 3 / 4
def adult_weight : ℚ := 150
def child_weight : ℚ := adult_weight / 2
def dog_weight : ℚ := adult_weight / 3
def cat1_weight : ℚ := adult_weight / 10
def cat2_weight : ℚ := adult_weight / 8

def canoe_capacity_with_pets : ℚ := people_with_pets_ratio * canoe_max_people

def total_weight_adults_and_children : ℚ := 4 * adult_weight + 2 * child_weight
def total_weight_pets : ℚ := dog_weight + cat1_weight + cat2_weight
def total_weight : ℚ := total_weight_adults_and_children + total_weight_pets

def max_weight_limit : ℚ := canoe_max_people * adult_weight

theorem canoe_total_weight_calculation :
  total_weight = 833 + 3 / 4 := by
  sorry

theorem canoe_maximum_weight_limit :
  max_weight_limit = 1200 := by
  sorry

end NUMINAMATH_GPT_canoe_total_weight_calculation_canoe_maximum_weight_limit_l1908_190842


namespace NUMINAMATH_GPT_solutions_to_equation_l1908_190855

theorem solutions_to_equation :
  ∀ x : ℝ, (x + 1) * (x - 2) = x + 1 ↔ x = -1 ∨ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solutions_to_equation_l1908_190855


namespace NUMINAMATH_GPT_train_usual_time_l1908_190866

theorem train_usual_time (S T_new T : ℝ) (h_speed : T_new = 7 / 6 * T) (h_delay : T_new = T + 1 / 6) : T = 1 := by
  sorry

end NUMINAMATH_GPT_train_usual_time_l1908_190866


namespace NUMINAMATH_GPT_multiply_polynomials_l1908_190806

variable {x y : ℝ}

theorem multiply_polynomials (x y : ℝ) :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 :=
by
  sorry

end NUMINAMATH_GPT_multiply_polynomials_l1908_190806


namespace NUMINAMATH_GPT_brie_clothes_washer_l1908_190803

theorem brie_clothes_washer (total_blouses total_skirts total_slacks : ℕ)
  (blouses_pct skirts_pct slacks_pct : ℝ)
  (h_blouses : total_blouses = 12)
  (h_skirts : total_skirts = 6)
  (h_slacks : total_slacks = 8)
  (h_blouses_pct : blouses_pct = 0.75)
  (h_skirts_pct : skirts_pct = 0.5)
  (h_slacks_pct : slacks_pct = 0.25) :
  let blouses_in_hamper := total_blouses * blouses_pct
  let skirts_in_hamper := total_skirts * skirts_pct
  let slacks_in_hamper := total_slacks * slacks_pct
  blouses_in_hamper + skirts_in_hamper + slacks_in_hamper = 14 := 
by
  sorry

end NUMINAMATH_GPT_brie_clothes_washer_l1908_190803


namespace NUMINAMATH_GPT_find_n_for_2013_in_expansion_l1908_190824

/-- Define the pattern for the last term of the expansion of n^3 -/
def last_term (n : ℕ) : ℕ :=
  n^2 + n - 1

/-- The main problem statement -/
theorem find_n_for_2013_in_expansion :
  ∃ n : ℕ, last_term (n - 1) ≤ 2013 ∧ 2013 < last_term n ∧ n = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_n_for_2013_in_expansion_l1908_190824


namespace NUMINAMATH_GPT_green_balloons_count_l1908_190860

-- Define the conditions
def total_balloons : Nat := 50
def red_balloons : Nat := 12
def blue_balloons : Nat := 7

-- Define the proof problem
theorem green_balloons_count : 
  let green_balloons := total_balloons - (red_balloons + blue_balloons)
  green_balloons = 31 :=
by
  sorry

end NUMINAMATH_GPT_green_balloons_count_l1908_190860


namespace NUMINAMATH_GPT_boys_in_class_l1908_190837

theorem boys_in_class (students : ℕ) (ratio_girls_boys : ℕ → Prop)
  (h1 : students = 56)
  (h2 : ratio_girls_boys 4 ∧ ratio_girls_boys 3) :
  ∃ k : ℕ, 4 * k + 3 * k = students ∧ 3 * k = 24 :=
by
  sorry

end NUMINAMATH_GPT_boys_in_class_l1908_190837


namespace NUMINAMATH_GPT_tan_alpha_third_quadrant_l1908_190816

theorem tan_alpha_third_quadrant (α : ℝ) 
  (h_eq: Real.sin α = Real.cos α) 
  (h_third: π < α ∧ α < 3 * π / 2) : Real.tan α = 1 := 
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_third_quadrant_l1908_190816


namespace NUMINAMATH_GPT_vectors_coplanar_l1908_190839

def vector3 := ℝ × ℝ × ℝ

def scalar_triple_product (a b c : vector3) : ℝ :=
  match a, b, c with
  | (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) =>
    a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

theorem vectors_coplanar : scalar_triple_product (-3, 3, 3) (-4, 7, 6) (3, 0, -1) = 0 :=
by
  sorry

end NUMINAMATH_GPT_vectors_coplanar_l1908_190839


namespace NUMINAMATH_GPT_polygon_angle_pairs_l1908_190835

theorem polygon_angle_pairs
  {r k : ℕ}
  (h_ratio : (180 * r - 360) / r = (4 / 3) * (180 * k - 360) / k)
  (h_k_lt_15 : k < 15)
  (h_r_ge_3 : r ≥ 3) :
  (k = 7 ∧ r = 42) ∨ (k = 6 ∧ r = 18) ∨ (k = 5 ∧ r = 10) ∨ (k = 4 ∧ r = 6) :=
sorry

end NUMINAMATH_GPT_polygon_angle_pairs_l1908_190835


namespace NUMINAMATH_GPT_dogwood_trees_total_is_100_l1908_190807

def initial_dogwood_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20
def total_dogwood_trees : ℕ := initial_dogwood_trees + trees_planted_today + trees_planted_tomorrow

theorem dogwood_trees_total_is_100 : total_dogwood_trees = 100 := by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_dogwood_trees_total_is_100_l1908_190807


namespace NUMINAMATH_GPT_combined_sleep_hours_l1908_190830

theorem combined_sleep_hours :
  let connor_sleep_hours := 6
  let luke_sleep_hours := connor_sleep_hours + 2
  let emma_sleep_hours := connor_sleep_hours - 1
  let ava_sleep_hours :=
    2 * 5 + 
    2 * (5 + 1) + 
    2 * (5 + 2) + 
    (5 + 3)
  let puppy_sleep_hours := 2 * luke_sleep_hours
  let cat_sleep_hours := 4 + 7
  7 * connor_sleep_hours +
  7 * luke_sleep_hours +
  7 * emma_sleep_hours +
  ava_sleep_hours +
  7 * puppy_sleep_hours +
  7 * cat_sleep_hours = 366 :=
by
  sorry

end NUMINAMATH_GPT_combined_sleep_hours_l1908_190830


namespace NUMINAMATH_GPT_pascal_sixth_element_row_20_l1908_190863

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
by 
  sorry

end NUMINAMATH_GPT_pascal_sixth_element_row_20_l1908_190863


namespace NUMINAMATH_GPT_sum_of_integers_l1908_190845

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_l1908_190845


namespace NUMINAMATH_GPT_find_f_one_l1908_190814

-- Define the function f(x-3) = 2x^2 - 3x + 1
noncomputable def f (x : ℤ) := 2 * (x+3)^2 - 3 * (x+3) + 1

-- Declare the theorem we intend to prove
theorem find_f_one : f 1 = 21 :=
by
  -- The proof goes here (saying "sorry" because the detailed proof is skipped)
  sorry

end NUMINAMATH_GPT_find_f_one_l1908_190814


namespace NUMINAMATH_GPT_negation_of_proposition_l1908_190804

variable (x y : ℝ)

theorem negation_of_proposition :
  (¬ (∀ x y : ℝ, (x^2 + y^2 = 0) → (x = 0 ∧ y = 0))) ↔ 
  (∃ x y : ℝ, (x^2 + y^2 ≠ 0) ∧ (x ≠ 0 ∨ y ≠ 0)) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1908_190804


namespace NUMINAMATH_GPT_range_of_a_l1908_190846

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x = 1 → ¬ ((x + 1) / (x + a) < 2))) ↔ -1 ≤ a ∧ a ≤ 0 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1908_190846


namespace NUMINAMATH_GPT_negation_of_p_is_universal_l1908_190865

-- Define the proposition p
def p : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

-- The proof statement for the negation of p
theorem negation_of_p_is_universal : ¬p ↔ ∀ x : ℝ, Real.exp x - x - 1 > 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_p_is_universal_l1908_190865


namespace NUMINAMATH_GPT_oblique_area_l1908_190859

theorem oblique_area (side_length : ℝ) (A_ratio : ℝ) (S_original : ℝ) (S_oblique : ℝ) 
  (h1 : side_length = 1) 
  (h2 : A_ratio = (Real.sqrt 2) / 4) 
  (h3 : S_original = side_length ^ 2) 
  (h4 : S_oblique / S_original = A_ratio) : 
  S_oblique = (Real.sqrt 2) / 4 :=
by 
  sorry

end NUMINAMATH_GPT_oblique_area_l1908_190859


namespace NUMINAMATH_GPT_ratio_of_Patrick_to_Joseph_l1908_190848

def countries_traveled_by_George : Nat := 6
def countries_traveled_by_Joseph : Nat := countries_traveled_by_George / 2
def countries_traveled_by_Zack : Nat := 18
def countries_traveled_by_Patrick : Nat := countries_traveled_by_Zack / 2

theorem ratio_of_Patrick_to_Joseph : countries_traveled_by_Patrick / countries_traveled_by_Joseph = 3 :=
by
  -- The definition conditions have already been integrated above
  sorry

end NUMINAMATH_GPT_ratio_of_Patrick_to_Joseph_l1908_190848


namespace NUMINAMATH_GPT_ryan_fish_count_l1908_190821

theorem ryan_fish_count
  (R : ℕ)
  (J : ℕ)
  (Jeffery_fish : ℕ)
  (h1 : Jeffery_fish = 60)
  (h2 : Jeffery_fish = 2 * R)
  (h3 : J + R + Jeffery_fish = 100)
  : R = 30 :=
by
  sorry

end NUMINAMATH_GPT_ryan_fish_count_l1908_190821


namespace NUMINAMATH_GPT_five_ones_make_100_l1908_190843

noncomputable def concatenate (a b c : Nat) : Nat :=
  a * 100 + b * 10 + c

theorem five_ones_make_100 :
  let one := 1
  let x := concatenate one one one -- 111
  let y := concatenate one one 0 / 10 -- 11, concatenation of 1 and 1 treated as 110, divided by 10
  x - y = 100 :=
by
  sorry

end NUMINAMATH_GPT_five_ones_make_100_l1908_190843


namespace NUMINAMATH_GPT_paint_for_smaller_statues_l1908_190871

open Real

theorem paint_for_smaller_statues :
  ∀ (paint_needed : ℝ) (height_big_statue height_small_statue : ℝ) (num_small_statues : ℝ),
  height_big_statue = 10 → height_small_statue = 2 → paint_needed = 5 → num_small_statues = 200 →
  (paint_needed / (height_big_statue / height_small_statue) ^ 2) * num_small_statues = 40 :=
by
  intros paint_needed height_big_statue height_small_statue num_small_statues
  intros h_big_height h_small_height h_paint_needed h_num_small
  rw [h_big_height, h_small_height, h_paint_needed, h_num_small]
  sorry

end NUMINAMATH_GPT_paint_for_smaller_statues_l1908_190871


namespace NUMINAMATH_GPT_algebraic_expression_value_l1908_190872

theorem algebraic_expression_value (x y : ℝ) (h1 : x + 2 * y = 4) (h2 : x - 2 * y = -1) :
  x^2 - 4 * y^2 + 1 = -3 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1908_190872


namespace NUMINAMATH_GPT_compute_cos_2_sum_zero_l1908_190892

theorem compute_cos_2_sum_zero (x y z : ℝ)
  (h1 : Real.cos (x + Real.pi / 4) + Real.cos (y + Real.pi / 4) + Real.cos (z + Real.pi / 4) = 0)
  (h2 : Real.sin (x + Real.pi / 4) + Real.sin (y + Real.pi / 4) + Real.sin (z + Real.pi / 4) = 0) :
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0 :=
by
  sorry

end NUMINAMATH_GPT_compute_cos_2_sum_zero_l1908_190892


namespace NUMINAMATH_GPT_largest_even_number_l1908_190864

theorem largest_even_number (x : ℤ) (h1 : 3 * x + 6 = (x + (x + 2) + (x + 4)) / 3 + 44) : 
  x + 4 = 24 := 
by 
  sorry

end NUMINAMATH_GPT_largest_even_number_l1908_190864


namespace NUMINAMATH_GPT_nominal_rate_of_interest_annual_l1908_190888

theorem nominal_rate_of_interest_annual (EAR nominal_rate : ℝ) (n : ℕ) (h1 : EAR = 0.0816) (h2 : n = 2) : 
  nominal_rate = 0.0796 :=
by 
  sorry

end NUMINAMATH_GPT_nominal_rate_of_interest_annual_l1908_190888


namespace NUMINAMATH_GPT_stadium_length_in_yards_l1908_190884

theorem stadium_length_in_yards (length_in_feet : ℕ) (conversion_factor : ℕ) : ℕ :=
    length_in_feet / conversion_factor

example : stadium_length_in_yards 240 3 = 80 :=
by sorry

end NUMINAMATH_GPT_stadium_length_in_yards_l1908_190884


namespace NUMINAMATH_GPT_maximum_value_l1908_190825

-- Define the variables as positive real numbers
variables (a b c : ℝ)

-- Define the conditions
def condition (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 2*a*b*c + 1

-- Define the expression
def expr (a b c : ℝ) : ℝ := (a - 2*b*c) * (b - 2*c*a) * (c - 2*a*b)

-- The theorem stating that under the given conditions, the expression has a maximum value of 1/8
theorem maximum_value : ∀ (a b c : ℝ), condition a b c → expr a b c ≤ 1/8 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_l1908_190825


namespace NUMINAMATH_GPT_number_of_good_numbers_lt_1000_l1908_190813

def is_good_number (n : ℕ) : Prop :=
  let sum := n + (n + 1) + (n + 2)
  sum % 10 < 10 ∧
  (sum / 10) % 10 < 10 ∧
  (sum / 100) % 10 < 10 ∧
  (sum < 1000)

theorem number_of_good_numbers_lt_1000 : ∃ n : ℕ, n = 48 ∧
  (forall k, k < 1000 → k < 1000 → is_good_number k → k = 48) := sorry

end NUMINAMATH_GPT_number_of_good_numbers_lt_1000_l1908_190813


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1908_190849

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a = 2 → |a| = 2) ∧ (|a| = 2 → a = 2 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1908_190849


namespace NUMINAMATH_GPT_complement_intersection_l1908_190841

open Set

def U : Set ℕ := {2, 3, 4, 5, 6}
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem complement_intersection :
  (U \ A) ∩ B = {3} :=
sorry

end NUMINAMATH_GPT_complement_intersection_l1908_190841


namespace NUMINAMATH_GPT_find_common_ratio_l1908_190832

variable (a₃ a₂ : ℝ)
variable (S₁ S₂ : ℝ)

-- Conditions
def condition1 : Prop := 3 * S₂ = a₃ - 2
def condition2 : Prop := 3 * S₁ = a₂ - 2

-- Theorem statement
theorem find_common_ratio (h1 : condition1 a₃ S₂)
                          (h2 : condition2 a₂ S₁) : 
                          (a₃ / a₂ = 4) :=
by 
  sorry

end NUMINAMATH_GPT_find_common_ratio_l1908_190832


namespace NUMINAMATH_GPT_smallest_a1_l1908_190896

noncomputable def is_sequence (a : ℕ → ℝ) : Prop :=
∀ n > 1, a n = 7 * a (n - 1) - 2 * n

noncomputable def is_positive_sequence (a : ℕ → ℝ) : Prop :=
∀ n > 0, a n > 0

theorem smallest_a1 (a : ℕ → ℝ)
  (h_seq : is_sequence a)
  (h_pos : is_positive_sequence a) :
  a 1 ≥ 13 / 18 :=
sorry

end NUMINAMATH_GPT_smallest_a1_l1908_190896


namespace NUMINAMATH_GPT_number_of_valid_pairs_l1908_190805

theorem number_of_valid_pairs :
  (∃! S : ℕ, S = 1250 ∧ ∀ (m n : ℕ), (1 ≤ m ∧ m ≤ 1000) →
  (3^n < 4^m ∧ 4^m < 4^(m+1) ∧ 4^(m+1) < 3^(n+1))) :=
sorry

end NUMINAMATH_GPT_number_of_valid_pairs_l1908_190805


namespace NUMINAMATH_GPT_primes_between_30_and_60_l1908_190810

theorem primes_between_30_and_60 (list_of_primes : List ℕ) 
  (H1 : list_of_primes = [31, 37, 41, 43, 47, 53, 59]) :
  (list_of_primes.headI * list_of_primes.reverse.headI) = 1829 := by
  sorry

end NUMINAMATH_GPT_primes_between_30_and_60_l1908_190810


namespace NUMINAMATH_GPT_model_A_selected_count_l1908_190878

def production_A := 1200
def production_B := 6000
def production_C := 2000
def total_selected := 46

def total_production := production_A + production_B + production_C

theorem model_A_selected_count :
  (production_A / total_production) * total_selected = 6 := by
  sorry

end NUMINAMATH_GPT_model_A_selected_count_l1908_190878


namespace NUMINAMATH_GPT_equation1_solutions_equation2_solutions_l1908_190861

theorem equation1_solutions (x : ℝ) :
  x ^ 2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 := by
  sorry

theorem equation2_solutions (x : ℝ) :
  2 * x ^ 2 - 2 * x = 1 ↔ x = (1 + Real.sqrt 3) / 2 ∨ x = (1 - Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_GPT_equation1_solutions_equation2_solutions_l1908_190861


namespace NUMINAMATH_GPT_max_m_divides_f_l1908_190819

noncomputable def f (n : ℕ) : ℤ :=
  (2 * n + 7) * 3^n + 9

theorem max_m_divides_f (m n : ℕ) (h1 : n > 0) (h2 : ∀ n : ℕ, n > 0 → m ∣ ((2 * n + 7) * 3^n + 9)) : m = 36 :=
sorry

end NUMINAMATH_GPT_max_m_divides_f_l1908_190819


namespace NUMINAMATH_GPT_third_year_students_sampled_correct_l1908_190891

-- The given conditions
def first_year_students := 700
def second_year_students := 670
def third_year_students := 630
def total_samples := 200
def total_students := first_year_students + second_year_students + third_year_students

-- The proportion of third-year students
def third_year_proportion := third_year_students / total_students

-- The number of third-year students to be selected
def samples_third_year := total_samples * third_year_proportion

theorem third_year_students_sampled_correct :
  samples_third_year = 63 :=
by
  -- We skip the actual proof for this statement with sorry
  sorry

end NUMINAMATH_GPT_third_year_students_sampled_correct_l1908_190891


namespace NUMINAMATH_GPT_number_of_oxygen_atoms_l1908_190852

theorem number_of_oxygen_atoms 
  (M_weight : ℝ)
  (H_weight : ℝ)
  (Cl_weight : ℝ)
  (O_weight : ℝ)
  (MW_formula : M_weight = H_weight + Cl_weight + n * O_weight)
  (M_weight_eq : M_weight = 68)
  (H_weight_eq : H_weight = 1)
  (Cl_weight_eq : Cl_weight = 35.5)
  (O_weight_eq : O_weight = 16)
  : n = 2 := 
  by sorry

end NUMINAMATH_GPT_number_of_oxygen_atoms_l1908_190852


namespace NUMINAMATH_GPT_stones_in_courtyard_l1908_190802

theorem stones_in_courtyard (S T B : ℕ) (h1 : T = S + 3 * S) (h2 : B = 2 * (T + S)) (h3 : B = 400) : S = 40 :=
by
  sorry

end NUMINAMATH_GPT_stones_in_courtyard_l1908_190802


namespace NUMINAMATH_GPT_find_side_b_l1908_190886

theorem find_side_b
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b * Real.sin A = 3 * c * Real.sin B)
  (h2 : a = 3)
  (h3 : Real.cos B = 2 / 3) :
  b = Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_find_side_b_l1908_190886


namespace NUMINAMATH_GPT_smallest_N_divisible_l1908_190874

theorem smallest_N_divisible (N x : ℕ) (H: N - 24 = 84 * Nat.lcm x 60) : N = 5064 :=
by
  sorry

end NUMINAMATH_GPT_smallest_N_divisible_l1908_190874


namespace NUMINAMATH_GPT_maximum_value_expression_l1908_190822

theorem maximum_value_expression (x y : ℝ) (h : x + y = 5) :
  ∃ p, p = x * y ∧ (4 * p^3 - 92 * p^2 + 754 * p) = 441 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_maximum_value_expression_l1908_190822


namespace NUMINAMATH_GPT_option_d_correct_l1908_190826

theorem option_d_correct (a b : ℝ) (h : a > b) : -b > -a :=
sorry

end NUMINAMATH_GPT_option_d_correct_l1908_190826


namespace NUMINAMATH_GPT_surface_area_of_cylinder_with_square_cross_section_l1908_190818

theorem surface_area_of_cylinder_with_square_cross_section
  (side_length : ℝ) (h1 : side_length = 2) : 
  (2 * Real.pi * 2 + 2 * Real.pi * 1^2) = 6 * Real.pi :=
by
  rw [←h1]
  sorry

end NUMINAMATH_GPT_surface_area_of_cylinder_with_square_cross_section_l1908_190818
