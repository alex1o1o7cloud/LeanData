import Mathlib

namespace NUMINAMATH_GPT_gcd_sum_and_lcm_eq_gcd_l5_563

theorem gcd_sum_and_lcm_eq_gcd (a b : ℤ) :  Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
sorry

end NUMINAMATH_GPT_gcd_sum_and_lcm_eq_gcd_l5_563


namespace NUMINAMATH_GPT_problem1_problem2_l5_594

theorem problem1 : (-1 / 2) * (-8) + (-6) = -2 := by
  sorry

theorem problem2 : -(1^4) - 2 / (-1 / 3) - abs (-9) = -4 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l5_594


namespace NUMINAMATH_GPT_triangle_subsegment_length_l5_557

noncomputable def length_of_shorter_subsegment (PQ QR PR PS SR : ℝ) :=
  PQ < QR ∧ 
  PR = 15 ∧ 
  PQ / QR = 1 / 5 ∧ 
  PS + SR = PR ∧ 
  PS = PQ / QR * SR → 
  PS = 5 / 2

theorem triangle_subsegment_length (PQ QR PR PS SR : ℝ) 
  (h1 : PQ < QR) 
  (h2 : PR = 15) 
  (h3 : PQ / QR = 1 / 5) 
  (h4 : PS + SR = PR) 
  (h5 : PS = PQ / QR * SR) : 
  length_of_shorter_subsegment PQ QR PR PS SR := 
sorry

end NUMINAMATH_GPT_triangle_subsegment_length_l5_557


namespace NUMINAMATH_GPT_geometric_progression_sum_eq_l5_528

theorem geometric_progression_sum_eq
  (a q b : ℝ) (n : ℕ)
  (hq : q ≠ 1)
  (h : (a * (q^2^n - 1)) / (q - 1) = (b * (q^(2*n) - 1)) / (q^2 - 1)) :
  b = a + a * q :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_sum_eq_l5_528


namespace NUMINAMATH_GPT_find_absolute_cd_l5_502

noncomputable def polynomial_solution (c d : ℤ) (root1 root2 root3 : ℤ) : Prop :=
  c ≠ 0 ∧ d ≠ 0 ∧ 
  root1 = root2 ∧
  (root3 ≠ root1 ∨ root3 ≠ root2) ∧
  (root1^3 + root2^2 * root3 + (c * root1^2) + (d * root1) + 16 * c = 0) ∧ 
  (root2^3 + root1^2 * root3 + (c * root2^2) + (d * root2) + 16 * c = 0) ∧
  (root3^3 + root1^2 * root3 + (c * root3^2) + (d * root3) + 16 * c = 0)

theorem find_absolute_cd : ∃ c d root1 root2 root3 : ℤ,
  polynomial_solution c d root1 root2 root3 ∧ (|c * d| = 2560) :=
sorry

end NUMINAMATH_GPT_find_absolute_cd_l5_502


namespace NUMINAMATH_GPT_fgf_one_l5_509

/-- Define the function f(x) = 5x + 2 --/
def f (x : ℝ) := 5 * x + 2

/-- Define the function g(x) = 3x - 1 --/
def g (x : ℝ) := 3 * x - 1

/-- Prove that f(g(f(1))) = 102 given the definitions of f and g --/
theorem fgf_one : f (g (f 1)) = 102 := by
  sorry

end NUMINAMATH_GPT_fgf_one_l5_509


namespace NUMINAMATH_GPT_more_non_product_eight_digit_numbers_l5_577

def num_eight_digit_numbers := 10^8 - 10^7
def num_four_digit_numbers := 9999 - 1000 + 1
def num_unique_products := (num_four_digit_numbers.choose 2) + num_four_digit_numbers

theorem more_non_product_eight_digit_numbers :
  (num_eight_digit_numbers - num_unique_products) > num_unique_products := by sorry

end NUMINAMATH_GPT_more_non_product_eight_digit_numbers_l5_577


namespace NUMINAMATH_GPT_find_sqrt_abc_sum_l5_520

theorem find_sqrt_abc_sum (a b c : ℝ) (h1 : b + c = 20) (h2 : c + a = 22) (h3 : a + b = 24) :
    Real.sqrt (a * b * c * (a + b + c)) = 206.1 := by
  sorry

end NUMINAMATH_GPT_find_sqrt_abc_sum_l5_520


namespace NUMINAMATH_GPT_distance_from_origin_is_correct_l5_525

-- Define the point (x, y) with given conditions
variables (x y : ℝ)

-- Given conditions
axiom h1 : y = 20
axiom h2 : dist (x, y) (2, 15) = 15
axiom h3 : x > 2

-- The theorem to prove
theorem distance_from_origin_is_correct :
  dist (x, y) (0, 0) = Real.sqrt (604 + 40 * Real.sqrt 2) :=
by
  -- Set h1, h2, and h3 as our constraints
  sorry

end NUMINAMATH_GPT_distance_from_origin_is_correct_l5_525


namespace NUMINAMATH_GPT_unique_non_overtaken_city_l5_531

structure City :=
(size_left : ℕ)
(size_right : ℕ)

def canOvertake (A B : City) : Prop :=
  A.size_right > B.size_left 

theorem unique_non_overtaken_city (n : ℕ) (H : n > 0) (cities : Fin n → City) : 
  ∃! i : Fin n, ∀ j : Fin n, ¬ canOvertake (cities j) (cities i) :=
by
  sorry

end NUMINAMATH_GPT_unique_non_overtaken_city_l5_531


namespace NUMINAMATH_GPT_mix_ratios_l5_550

theorem mix_ratios (milk1 water1 milk2 water2 : ℕ) 
  (h1 : milk1 = 7) (h2 : water1 = 2)
  (h3 : milk2 = 8) (h4 : water2 = 1) :
  (milk1 + milk2) / (water1 + water2) = 5 :=
by
  -- Proof required here
  sorry

end NUMINAMATH_GPT_mix_ratios_l5_550


namespace NUMINAMATH_GPT_inverse_proportion_increasing_implication_l5_512

theorem inverse_proportion_increasing_implication (m x : ℝ) (h1 : x > 0) (h2 : ∀ x1 x2, x1 > 0 → x2 > 0 → x1 < x2 → (m + 3) / x1 < (m + 3) / x2) : m < -3 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_increasing_implication_l5_512


namespace NUMINAMATH_GPT_lowest_price_is_six_l5_503

def single_package_cost : ℝ := 7
def eight_oz_package_cost : ℝ := 4
def four_oz_package_original_cost : ℝ := 2
def discount_rate : ℝ := 0.5

theorem lowest_price_is_six
  (cost_single : single_package_cost = 7)
  (cost_eight : eight_oz_package_cost = 4)
  (cost_four : four_oz_package_original_cost = 2)
  (discount : discount_rate = 0.5) :
  min single_package_cost (eight_oz_package_cost + 2 * (four_oz_package_original_cost * discount_rate)) = 6 := by
  sorry

end NUMINAMATH_GPT_lowest_price_is_six_l5_503


namespace NUMINAMATH_GPT_factorize_expression_l5_517

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_factorize_expression_l5_517


namespace NUMINAMATH_GPT_original_amount_of_solution_y_l5_598

theorem original_amount_of_solution_y (Y : ℝ) 
  (h1 : 0 < Y) -- We assume Y > 0 
  (h2 : 0.3 * (Y - 4) + 1.2 = 0.45 * Y) :
  Y = 8 := 
sorry

end NUMINAMATH_GPT_original_amount_of_solution_y_l5_598


namespace NUMINAMATH_GPT_exists_x_for_integer_conditions_l5_524

-- Define the conditions as functions in Lean
def is_int_div (a b : Int) : Prop := ∃ k : Int, a = b * k

-- The target statement in Lean 4
theorem exists_x_for_integer_conditions :
  ∃ t_1 : Int, ∃ x : Int, (x = 105 * t_1 + 52) ∧ 
    (is_int_div (x - 3) 7) ∧ 
    (is_int_div (x - 2) 5) ∧ 
    (is_int_div (x - 4) 3) :=
by 
  sorry

end NUMINAMATH_GPT_exists_x_for_integer_conditions_l5_524


namespace NUMINAMATH_GPT_beka_flies_more_l5_576

-- Definitions
def beka_flight_distance : ℕ := 873
def jackson_flight_distance : ℕ := 563

-- The theorem we need to prove
theorem beka_flies_more : beka_flight_distance - jackson_flight_distance = 310 :=
by
  sorry

end NUMINAMATH_GPT_beka_flies_more_l5_576


namespace NUMINAMATH_GPT_percentage_passed_eng_students_l5_508

variable (total_male_students : ℕ := 120)
variable (total_female_students : ℕ := 100)
variable (total_international_students : ℕ := 70)
variable (total_disabilities_students : ℕ := 30)

variable (male_eng_percentage : ℕ := 25)
variable (female_eng_percentage : ℕ := 20)
variable (intern_eng_percentage : ℕ := 15)
variable (disab_eng_percentage : ℕ := 10)

variable (male_pass_percentage : ℕ := 20)
variable (female_pass_percentage : ℕ := 25)
variable (intern_pass_percentage : ℕ := 30)
variable (disab_pass_percentage : ℕ := 35)

def total_engineering_students : ℕ :=
  (total_male_students * male_eng_percentage / 100) +
  (total_female_students * female_eng_percentage / 100) +
  (total_international_students * intern_eng_percentage / 100) +
  (total_disabilities_students * disab_eng_percentage / 100)

def total_passed_engineering_students : ℕ :=
  (total_male_students * male_eng_percentage / 100 * male_pass_percentage / 100) +
  (total_female_students * female_eng_percentage / 100 * female_pass_percentage / 100) +
  (total_international_students * intern_eng_percentage / 100 * intern_pass_percentage / 100) +
  (total_disabilities_students * disab_eng_percentage / 100 * disab_pass_percentage / 100)

def passed_eng_students_percentage : ℕ :=
  total_passed_engineering_students * 100 / total_engineering_students

theorem percentage_passed_eng_students :
  passed_eng_students_percentage = 23 :=
sorry

end NUMINAMATH_GPT_percentage_passed_eng_students_l5_508


namespace NUMINAMATH_GPT_travel_from_A_to_C_l5_555

def num_ways_A_to_B : ℕ := 5 + 2  -- 5 buses and 2 trains
def num_ways_B_to_C : ℕ := 3 + 2  -- 3 buses and 2 ferries

theorem travel_from_A_to_C :
  num_ways_A_to_B * num_ways_B_to_C = 35 :=
by
  -- The proof environment will be added here. 
  -- We include 'sorry' here for now.
  sorry

end NUMINAMATH_GPT_travel_from_A_to_C_l5_555


namespace NUMINAMATH_GPT_fraction_sum_condition_l5_586

theorem fraction_sum_condition 
  (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0)
  (h : x + y = x * y): 
  (1/x + 1/y = 1) :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_condition_l5_586


namespace NUMINAMATH_GPT_find_angle_A_l5_575

theorem find_angle_A (a b : ℝ) (B A : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 2) 
  (h3 : B = Real.pi / 4) : A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_find_angle_A_l5_575


namespace NUMINAMATH_GPT_max_magnitude_z3_plus_3z_plus_2i_l5_595

open Complex

theorem max_magnitude_z3_plus_3z_plus_2i (z : ℂ) (h : Complex.abs z = 1) :
  ∃ M, M = 3 * Real.sqrt 3 ∧ ∀ (z : ℂ), Complex.abs z = 1 → Complex.abs (z^3 + 3 * z + 2 * Complex.I) ≤ M :=
by
  sorry

end NUMINAMATH_GPT_max_magnitude_z3_plus_3z_plus_2i_l5_595


namespace NUMINAMATH_GPT_percentage_calculation_l5_567

theorem percentage_calculation 
  (number : ℝ)
  (h1 : 0.035 * number = 700) :
  0.024 * (1.5 * number) = 720 := 
by
  sorry

end NUMINAMATH_GPT_percentage_calculation_l5_567


namespace NUMINAMATH_GPT_not_unique_equilateral_by_one_angle_and_opposite_side_l5_529

-- Definitions related to triangles
structure Triangle :=
  (a b c : ℝ) -- sides
  (alpha beta gamma : ℝ) -- angles

-- Definition of triangle types
def isIsosceles (t : Triangle) : Prop :=
  (t.a = t.b ∨ t.b = t.c ∨ t.a = t.c)

def isRight (t : Triangle) : Prop :=
  (t.alpha = 90 ∨ t.beta = 90 ∨ t.gamma = 90)

def isEquilateral (t : Triangle) : Prop :=
  (t.a = t.b ∧ t.b = t.c ∧ t.alpha = 60 ∧ t.beta = 60 ∧ t.gamma = 60)

def isScalene (t : Triangle) : Prop :=
  (t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c)

-- Proof that having one angle and the side opposite it does not determine an equilateral triangle.
theorem not_unique_equilateral_by_one_angle_and_opposite_side :
  ¬ ∀ (t1 t2 : Triangle), (isEquilateral t1 ∧ isEquilateral t2 →
    t1.alpha = t2.alpha ∧ t1.a = t2.a →
    t1 = t2) := sorry

end NUMINAMATH_GPT_not_unique_equilateral_by_one_angle_and_opposite_side_l5_529


namespace NUMINAMATH_GPT_CDs_per_rack_l5_535

theorem CDs_per_rack (racks_on_shelf : ℕ) (CDs_on_shelf : ℕ) (h1 : racks_on_shelf = 4) (h2 : CDs_on_shelf = 32) : 
  CDs_on_shelf / racks_on_shelf = 8 :=
by
  sorry

end NUMINAMATH_GPT_CDs_per_rack_l5_535


namespace NUMINAMATH_GPT_k_range_l5_533

noncomputable def valid_k (k : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → x / Real.exp x < 1 / (k + 2 * x - x^2)

theorem k_range : {k : ℝ | valid_k k} = {k : ℝ | 0 ≤ k ∧ k < Real.exp 1 - 1} :=
by sorry

end NUMINAMATH_GPT_k_range_l5_533


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l5_540

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 - 2 * x + 3 ≥ 0) ↔ ∃ x : ℝ, x^2 - 2 * x + 3 < 0 := 
sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l5_540


namespace NUMINAMATH_GPT_perimeter_ratio_l5_513

theorem perimeter_ratio (w l : ℕ) (hfold : w = 8) (lfold : l = 6) 
(folded_w : w / 2 = 4) (folded_l : l / 2 = 3) 
(hcut : w / 4 = 1) (lcut : l / 2 = 3) 
(perimeter_small : ℕ) (perimeter_large : ℕ)
(hperim_small : perimeter_small = 2 * (3 + 4)) 
(hperim_large : perimeter_large = 2 * (6 + 4)) :
(perimeter_small : ℕ) / (perimeter_large : ℕ) = 7 / 10 := sorry

end NUMINAMATH_GPT_perimeter_ratio_l5_513


namespace NUMINAMATH_GPT_equation_D_is_linear_l5_504

-- Definitions according to the given conditions
def equation_A (x y : ℝ) := x + 2 * y = 3
def equation_B (x : ℝ) := 3 * x - 2
def equation_C (x : ℝ) := x^2 + x = 6
def equation_D (x : ℝ) := (1 / 3) * x - 2 = 3

-- Properties of a linear equation
def is_linear (eq : ℝ → Prop) : Prop :=
∃ a b c : ℝ, (∃ x : ℝ, eq x = (a * x + b = c)) ∧ a ≠ 0

-- Specifying that equation_D is linear
theorem equation_D_is_linear : is_linear equation_D :=
by
  sorry

end NUMINAMATH_GPT_equation_D_is_linear_l5_504


namespace NUMINAMATH_GPT_find_a_minus_b_l5_537

-- Given definitions for conditions
variables (a b : ℤ)

-- Given conditions as hypotheses
def condition1 := a + 2 * b = 5
def condition2 := a * b = -12

theorem find_a_minus_b (h1 : condition1 a b) (h2 : condition2 a b) : a - b = -7 :=
sorry

end NUMINAMATH_GPT_find_a_minus_b_l5_537


namespace NUMINAMATH_GPT_quotient_remainder_l5_523

theorem quotient_remainder (x y : ℕ) (hx : 0 ≤ x) (hy : 0 < y) : 
  ∃ q r : ℕ, q ≥ 0 ∧ 0 ≤ r ∧ r < y ∧ x = q * y + r := by
  sorry

end NUMINAMATH_GPT_quotient_remainder_l5_523


namespace NUMINAMATH_GPT_julie_initial_savings_l5_568

def calculate_earnings (lawns newspapers dogs : ℕ) (price_lawn price_newspaper price_dog : ℝ) : ℝ :=
  (lawns * price_lawn) + (newspapers * price_newspaper) + (dogs * price_dog)

def calculate_total_spent_bike (earnings remaining_money : ℝ) : ℝ :=
  earnings + remaining_money

def calculate_initial_savings (cost_bike total_spent : ℝ) : ℝ :=
  cost_bike - total_spent

theorem julie_initial_savings :
  let cost_bike := 2345
  let lawns := 20
  let newspapers := 600
  let dogs := 24
  let price_lawn := 20
  let price_newspaper := 0.40
  let price_dog := 15
  let remaining_money := 155
  let earnings := calculate_earnings lawns newspapers dogs price_lawn price_newspaper price_dog
  let total_spent := calculate_total_spent_bike earnings remaining_money
  calculate_initial_savings cost_bike total_spent = 1190 :=
by
  -- Although the proof is not required, this setup assumes correctness.
  sorry

end NUMINAMATH_GPT_julie_initial_savings_l5_568


namespace NUMINAMATH_GPT_at_least_one_not_less_than_neg_two_l5_574

theorem at_least_one_not_less_than_neg_two (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1/b ≥ -2 ∨ b + 1/c ≥ -2 ∨ c + 1/a ≥ -2) :=
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_neg_two_l5_574


namespace NUMINAMATH_GPT_probability_of_shaded_section_l5_562

theorem probability_of_shaded_section 
  (total_sections : ℕ)
  (shaded_sections : ℕ)
  (H1 : total_sections = 8)
  (H2 : shaded_sections = 4)
  : (shaded_sections / total_sections : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_shaded_section_l5_562


namespace NUMINAMATH_GPT_geometric_series_proof_l5_585

theorem geometric_series_proof (y : ℝ) :
  ((1 + (1/3) + (1/9) + (1/27) + ∑' n : ℕ, (1 / 3^(n+1))) * 
   (1 - (1/3) + (1/9) - (1/27) + ∑' n : ℕ, ((-1)^n * (1 / 3^(n+1)))) = 
   1 + (1/y) + (1/y^2) + (∑' n : ℕ, (1 / y^(n+1)))) → y = 9 := by
  sorry

end NUMINAMATH_GPT_geometric_series_proof_l5_585


namespace NUMINAMATH_GPT_stratified_sampling_example_l5_510

noncomputable def sample_proportion := 70 / 3500
noncomputable def total_students := 3500 + 1500
noncomputable def sample_size := total_students * sample_proportion

theorem stratified_sampling_example 
  (high_school_students : ℕ := 3500)
  (junior_high_students : ℕ := 1500)
  (sampled_high_school_students : ℕ := 70)
  (proportion_of_sampling : ℝ := sampled_high_school_students / high_school_students)
  (total_number_of_students : ℕ := high_school_students + junior_high_students)
  (calculated_sample_size : ℝ := total_number_of_students * proportion_of_sampling) :
  calculated_sample_size = 100 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_example_l5_510


namespace NUMINAMATH_GPT_quadratic_has_one_solution_positive_value_of_n_l5_554

theorem quadratic_has_one_solution_positive_value_of_n :
  ∃ n : ℝ, (4 * x ^ 2 + n * x + 1 = 0 → n ^ 2 - 16 = 0) ∧ n > 0 ∧ n = 4 :=
sorry

end NUMINAMATH_GPT_quadratic_has_one_solution_positive_value_of_n_l5_554


namespace NUMINAMATH_GPT_third_of_ten_l5_564

theorem third_of_ten : (1/3 : ℝ) * 10 = 8 / 3 :=
by
  have h : (1/4 : ℝ) * 20 = 4 := by sorry
  sorry

end NUMINAMATH_GPT_third_of_ten_l5_564


namespace NUMINAMATH_GPT__l5_530

noncomputable def X := 0
noncomputable def Y := 1
noncomputable def Z := 2

noncomputable def angle_XYZ (X Y Z : ℝ) : ℝ := 90 -- Triangle XYZ where ∠X = 90°

noncomputable def length_YZ := 10 -- YZ = 10 units
noncomputable def length_XY := 6 -- XY = 6 units
noncomputable def length_XZ : ℝ := Real.sqrt (length_YZ^2 - length_XY^2) -- Pythagorean theorem to find XZ
noncomputable def cos_Z : ℝ := length_XZ / length_YZ -- cos Z = adjacent/hypotenuse

example : cos_Z = 0.8 :=
by {
  sorry
}

end NUMINAMATH_GPT__l5_530


namespace NUMINAMATH_GPT_gcd_13642_19236_34176_l5_573

theorem gcd_13642_19236_34176 : Int.gcd (Int.gcd 13642 19236) 34176 = 2 := 
sorry

end NUMINAMATH_GPT_gcd_13642_19236_34176_l5_573


namespace NUMINAMATH_GPT_area_under_cos_l5_589

theorem area_under_cos :
  ∫ x in (0 : ℝ)..(3 * Real.pi / 2), |Real.cos x| = 3 :=
by
  sorry

end NUMINAMATH_GPT_area_under_cos_l5_589


namespace NUMINAMATH_GPT_percentage_of_water_in_mixture_l5_552

-- Definitions based on conditions from a)
def original_price : ℝ := 1 -- assuming $1 per liter for pure dairy
def selling_price : ℝ := 1.25 -- 25% profit means selling at $1.25
def profit_percentage : ℝ := 0.25 -- 25% profit

-- Theorem statement based on the equivalent problem in c)
theorem percentage_of_water_in_mixture : 
  (selling_price - original_price) / selling_price * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_water_in_mixture_l5_552


namespace NUMINAMATH_GPT_smallest_PR_minus_QR_l5_505

theorem smallest_PR_minus_QR :
  ∃ (PQ QR PR : ℤ), 
    PQ + QR + PR = 2023 ∧ PQ ≤ QR ∧ QR < PR ∧ PR - QR = 13 :=
by
  sorry

end NUMINAMATH_GPT_smallest_PR_minus_QR_l5_505


namespace NUMINAMATH_GPT_negation_of_existence_l5_561

theorem negation_of_existence (T : Type) (triangle : T → Prop) (sum_interior_angles : T → ℝ) :
  (¬ ∃ t : T, sum_interior_angles t ≠ 180) ↔ (∀ t : T, sum_interior_angles t = 180) :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_existence_l5_561


namespace NUMINAMATH_GPT_toothpick_pattern_15th_stage_l5_553

theorem toothpick_pattern_15th_stage :
  let a₁ := 5
  let d := 3
  let n := 15
  a₁ + (n - 1) * d = 47 :=
by
  sorry

end NUMINAMATH_GPT_toothpick_pattern_15th_stage_l5_553


namespace NUMINAMATH_GPT_find_x_l5_572

noncomputable def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

noncomputable def vec_dot (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt ((v.1)^2 + (v.2)^2)

theorem find_x (x : ℝ) (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (1, x)) 
  (h3 : magnitude (vec_sub a b) = vec_dot a b) : 
  x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l5_572


namespace NUMINAMATH_GPT_max_value_of_exp_diff_l5_581

open Real

theorem max_value_of_exp_diff : ∀ x : ℝ, ∃ y : ℝ, y = 2^x - 4^x ∧ y ≤ 1/4 := sorry

end NUMINAMATH_GPT_max_value_of_exp_diff_l5_581


namespace NUMINAMATH_GPT_sufficient_condition_for_inequality_l5_507

theorem sufficient_condition_for_inequality (x : ℝ) : (1 - 1/x > 0) → (x > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_inequality_l5_507


namespace NUMINAMATH_GPT_power_expression_l5_501

variable {a b : ℝ}

theorem power_expression : (-2 * a^2 * b^3)^3 = -8 * a^6 * b^9 := 
by 
  sorry

end NUMINAMATH_GPT_power_expression_l5_501


namespace NUMINAMATH_GPT_exists_overlapping_pairs_l5_566

-- Definition of conditions:
def no_boy_danced_with_all_girls (B : Type) (G : Type) (danced : B → G → Prop) :=
  ∀ b : B, ∃ g : G, ¬ danced b g

def each_girl_danced_with_at_least_one_boy (B : Type) (G : Type) (danced : B → G → Prop) :=
  ∀ g : G, ∃ b : B, danced b g

-- The main theorem to prove:
theorem exists_overlapping_pairs
  (B : Type) (G : Type) (danced : B → G → Prop)
  (h1 : no_boy_danced_with_all_girls B G danced)
  (h2 : each_girl_danced_with_at_least_one_boy B G danced) :
  ∃ (b1 b2 : B) (g1 g2 : G), b1 ≠ b2 ∧ g1 ≠ g2 ∧ danced b1 g1 ∧ danced b2 g2 :=
sorry

end NUMINAMATH_GPT_exists_overlapping_pairs_l5_566


namespace NUMINAMATH_GPT_platform_length_1000_l5_590

open Nat Real

noncomputable def length_of_platform (train_length : ℝ) (time_pole : ℝ) (time_platform : ℝ) : ℝ :=
  let speed := train_length / time_pole
  let platform_length := (speed * time_platform) - train_length
  platform_length

theorem platform_length_1000 :
  length_of_platform 300 9 39 = 1000 := by
  sorry

end NUMINAMATH_GPT_platform_length_1000_l5_590


namespace NUMINAMATH_GPT_outlet_pipe_empties_2_over_3_in_16_min_l5_596

def outlet_pipe_part_empty_in_t (t : ℕ) (part_per_8_min : ℚ) : ℚ :=
  (part_per_8_min / 8) * t

theorem outlet_pipe_empties_2_over_3_in_16_min (
  part_per_8_min : ℚ := 1/3
) : outlet_pipe_part_empty_in_t 16 part_per_8_min = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_outlet_pipe_empties_2_over_3_in_16_min_l5_596


namespace NUMINAMATH_GPT_quadratic_one_solution_m_l5_556

theorem quadratic_one_solution_m (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - 7 * x + m = 0) → 
  (∀ (x y : ℝ), 3 * x^2 - 7 * x + m = 0 → 3 * y^2 - 7 * y + m = 0 → x = y) → 
  m = 49 / 12 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_one_solution_m_l5_556


namespace NUMINAMATH_GPT_StacyBoughtPacks_l5_506

theorem StacyBoughtPacks (sheets_per_pack days daily_printed_sheets total_packs : ℕ) 
  (h1 : sheets_per_pack = 240)
  (h2 : days = 6)
  (h3 : daily_printed_sheets = 80) 
  (h4 : total_packs = (days * daily_printed_sheets) / sheets_per_pack) : total_packs = 2 :=
by 
  sorry

end NUMINAMATH_GPT_StacyBoughtPacks_l5_506


namespace NUMINAMATH_GPT_plywood_cut_difference_l5_522

theorem plywood_cut_difference :
  ∀ (length width : ℕ) (n : ℕ) (perimeter_greatest perimeter_least : ℕ),
    length = 8 ∧ width = 4 ∧ n = 4 ∧
    (∀ l w, (l = (length / 2) ∧ w = width) ∨ (l = length ∧ w = (width / 2)) → (perimeter_greatest = 2 * (l + w))) ∧
    (∀ l w, (l = (length / n) ∧ w = width) ∨ (l = length ∧ w = (width / n)) → (perimeter_least = 2 * (l + w))) →
    length = 8 ∧ width = 4 ∧ n = 4 ∧ perimeter_greatest = 18 ∧ perimeter_least = 12 →
    (perimeter_greatest - perimeter_least) = 6 :=
by
  intros length width n perimeter_greatest perimeter_least h1 h2
  sorry

end NUMINAMATH_GPT_plywood_cut_difference_l5_522


namespace NUMINAMATH_GPT_minimum_force_to_submerge_cube_l5_500

-- Definitions and given conditions
def volume_cube : ℝ := 10e-6 -- 10 cm^3 in m^3
def density_cube : ℝ := 700 -- in kg/m^3
def density_water : ℝ := 1000 -- in kg/m^3
def gravity : ℝ := 10 -- in m/s^2

-- Prove the minimum force required to submerge the cube completely
theorem minimum_force_to_submerge_cube : 
  (density_water * volume_cube * gravity - density_cube * volume_cube * gravity) = 0.03 :=
by
  sorry

end NUMINAMATH_GPT_minimum_force_to_submerge_cube_l5_500


namespace NUMINAMATH_GPT_pages_revised_only_once_l5_545

variable (x : ℕ)

def rate_first_time_typing := 6
def rate_revision := 4
def total_pages := 100
def pages_revised_twice := 15
def total_cost := 860

theorem pages_revised_only_once : 
  rate_first_time_typing * total_pages 
  + rate_revision * x 
  + rate_revision * pages_revised_twice * 2 
  = total_cost 
  → x = 35 :=
by
  sorry

end NUMINAMATH_GPT_pages_revised_only_once_l5_545


namespace NUMINAMATH_GPT_linear_regression_intercept_l5_580

theorem linear_regression_intercept :
  let x_values := [1, 2, 3, 4, 5]
  let y_values := [0.5, 0.8, 1.0, 1.2, 1.5]
  let x_mean := (x_values.sum / x_values.length : ℝ)
  let y_mean := (y_values.sum / y_values.length : ℝ)
  let slope := 0.24
  (x_mean = 3) →
  (y_mean = 1) →
  y_mean = slope * x_mean + 0.28 :=
by
  sorry

end NUMINAMATH_GPT_linear_regression_intercept_l5_580


namespace NUMINAMATH_GPT_sum_first_n_terms_l5_511

variable (a : ℕ → ℕ)

axiom a1_condition : a 1 = 2
axiom diff_condition : ∀ n : ℕ, a (n + 1) - a n = 2^n

-- Define the sum of the first n terms of the sequence
noncomputable def S : ℕ → ℕ
| 0 => 0
| (n + 1) => S n + a (n + 1)

theorem sum_first_n_terms (n : ℕ) : S a n = 2^(n + 1) - 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_n_terms_l5_511


namespace NUMINAMATH_GPT_larger_rectangle_area_l5_578

/-- Given a smaller rectangle made out of three squares each of area 25 cm²,
    where two vertices of the smaller rectangle lie on the midpoints of the
    shorter sides of the larger rectangle and the other two vertices lie on
    the longer sides, prove the area of the larger rectangle is 150 cm². -/
theorem larger_rectangle_area (s : ℝ) (l W S_Larger W_Larger : ℝ)
  (h_s : s^2 = 25) 
  (h_small_dim : l = 3 * s ∧ W = s ∧ l * W = 3 * s^2) 
  (h_vertices : 2 * W = W_Larger ∧ l = S_Larger) :
  (S_Larger * W_Larger = 150) := 
by
  sorry

end NUMINAMATH_GPT_larger_rectangle_area_l5_578


namespace NUMINAMATH_GPT_production_increase_l5_519

theorem production_increase (h1 : ℝ) (h2 : ℝ) (h3 : h1 = 0.75) (h4 : h2 = 0.5) :
  (h1 + h2 - 1) = 0.25 := by
  sorry

end NUMINAMATH_GPT_production_increase_l5_519


namespace NUMINAMATH_GPT_ratio_equivalence_l5_582

theorem ratio_equivalence (a b : ℝ) (hb : b ≠ 0) (h : a / b = 5 / 4) : (4 * a + 3 * b) / (4 * a - 3 * b) = 4 :=
sorry

end NUMINAMATH_GPT_ratio_equivalence_l5_582


namespace NUMINAMATH_GPT_solve_equation_frac_l5_597

theorem solve_equation_frac (x : ℝ) (h : x ≠ 2) : (3 / (x - 2) = 1) ↔ (x = 5) :=
by
  sorry -- proof is to be constructed

end NUMINAMATH_GPT_solve_equation_frac_l5_597


namespace NUMINAMATH_GPT_diesel_fuel_usage_l5_547

theorem diesel_fuel_usage (weekly_spending : ℝ) (cost_per_gallon : ℝ) (weeks : ℝ) (result : ℝ): 
  weekly_spending = 36 → cost_per_gallon = 3 → weeks = 2 → result = 24 → 
  (weekly_spending / cost_per_gallon) * weeks = result :=
by
  intros
  sorry

end NUMINAMATH_GPT_diesel_fuel_usage_l5_547


namespace NUMINAMATH_GPT_Barons_theorem_correct_l5_549

theorem Barons_theorem_correct (a b : ℕ) (ha: 0 < a) (hb: 0 < b) : 
  ∃ n : ℕ, 0 < n ∧ ∃ k1 k2 : ℕ, an = k1 ^ 2 ∧ bn = k2 ^ 3 := 
sorry

end NUMINAMATH_GPT_Barons_theorem_correct_l5_549


namespace NUMINAMATH_GPT_spring_length_function_l5_543

noncomputable def spring_length (x : ℝ) : ℝ :=
  12 + 3 * x

theorem spring_length_function :
  ∀ (x : ℝ), spring_length x = 12 + 3 * x :=
by
  intro x
  rfl

end NUMINAMATH_GPT_spring_length_function_l5_543


namespace NUMINAMATH_GPT_price_reduction_2100_yuan_l5_587

-- Definitions based on conditions
def initial_sales : ℕ := 30
def initial_profit_per_item : ℕ := 50
def additional_sales_per_yuan (x : ℕ) : ℕ := 2 * x
def new_profit_per_item (x : ℕ) : ℕ := 50 - x
def target_profit : ℕ := 2100

-- Final proof statement, showing the price reduction needed
theorem price_reduction_2100_yuan (x : ℕ) 
  (h : (50 - x) * (30 + 2 * x) = 2100) : 
  x = 20 := 
by 
  sorry

end NUMINAMATH_GPT_price_reduction_2100_yuan_l5_587


namespace NUMINAMATH_GPT_total_erasers_is_35_l5_534

def Celine : ℕ := 10

def Gabriel : ℕ := Celine / 2

def Julian : ℕ := Celine * 2

def total_erasers : ℕ := Celine + Gabriel + Julian

theorem total_erasers_is_35 : total_erasers = 35 :=
  by
  sorry

end NUMINAMATH_GPT_total_erasers_is_35_l5_534


namespace NUMINAMATH_GPT_find_F_l5_591

-- Define the condition and the equation
def C (F : ℤ) : ℤ := (5 * (F - 30)) / 9

-- Define the assumption that C = 25
def C_condition : ℤ := 25

-- The theorem to prove that F = 75 given the conditions
theorem find_F (F : ℤ) (h : C F = C_condition) : F = 75 :=
sorry

end NUMINAMATH_GPT_find_F_l5_591


namespace NUMINAMATH_GPT_annual_yield_range_l5_527

-- Here we set up the conditions as definitions in Lean 4
def last_year_range : ℝ := 10000
def improvement_rate : ℝ := 0.15

-- Theorems that are based on the conditions and need proving
theorem annual_yield_range (last_year_range : ℝ) (improvement_rate : ℝ) : 
  last_year_range * (1 + improvement_rate) = 11500 := 
sorry

end NUMINAMATH_GPT_annual_yield_range_l5_527


namespace NUMINAMATH_GPT_ratio_larva_to_cocoon_l5_559

theorem ratio_larva_to_cocoon (total_days : ℕ) (cocoon_days : ℕ)
  (h1 : total_days = 120) (h2 : cocoon_days = 30) :
  (total_days - cocoon_days) / cocoon_days = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_larva_to_cocoon_l5_559


namespace NUMINAMATH_GPT_solution_l5_544

noncomputable def problem (x : ℝ) : Prop :=
  (Real.sqrt (Real.sqrt (53 - 3 * x)) + Real.sqrt (Real.sqrt (39 + 3 * x))) = 5

theorem solution :
  ∀ x : ℝ, problem x → x = -23 / 3 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solution_l5_544


namespace NUMINAMATH_GPT_intersection_point_l5_592

noncomputable def g (x : ℝ) : ℝ := x^3 + 3 * x^2 + 9 * x + 15

theorem intersection_point :
  ∃ a : ℝ, g a = a ∧ a = -3 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_l5_592


namespace NUMINAMATH_GPT_jason_total_spent_l5_565

-- Conditions
def shorts_cost : ℝ := 14.28
def jacket_cost : ℝ := 4.74

-- Statement to prove
theorem jason_total_spent : shorts_cost + jacket_cost = 19.02 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_jason_total_spent_l5_565


namespace NUMINAMATH_GPT_ribbon_tying_length_l5_526

theorem ribbon_tying_length :
  let l1 := 36
  let l2 := 42
  let l3 := 48
  let cut1 := l1 / 6
  let cut2 := l2 / 6
  let cut3 := l3 / 6
  let rem1 := l1 - cut1
  let rem2 := l2 - cut2
  let rem3 := l3 - cut3
  let total_rem := rem1 + rem2 + rem3
  let final_length := 97
  let tying_length := total_rem - final_length
  tying_length = 8 :=
by
  sorry

end NUMINAMATH_GPT_ribbon_tying_length_l5_526


namespace NUMINAMATH_GPT_rational_number_div_eq_l5_593

theorem rational_number_div_eq :
  ∃ x : ℚ, (-2 : ℚ) / x = 8 ∧ x = -1 / 4 :=
by
  existsi (-1 / 4 : ℚ)
  sorry

end NUMINAMATH_GPT_rational_number_div_eq_l5_593


namespace NUMINAMATH_GPT_symmetric_point_with_respect_to_x_axis_l5_518

-- Definition of point M
def point_M : ℝ × ℝ := (3, -4)

-- Define the symmetry condition with respect to the x-axis
def symmetric_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Statement that the symmetric point to point M with respect to the x-axis is (3, 4)
theorem symmetric_point_with_respect_to_x_axis : symmetric_x point_M = (3, 4) :=
by
  -- This is the statement of the theorem; the proof will be added here.
  sorry

end NUMINAMATH_GPT_symmetric_point_with_respect_to_x_axis_l5_518


namespace NUMINAMATH_GPT_total_number_of_eyes_l5_551

theorem total_number_of_eyes (n_spiders n_ants eyes_per_spider eyes_per_ant : ℕ)
  (h1 : n_spiders = 3) (h2 : n_ants = 50) (h3 : eyes_per_spider = 8) (h4 : eyes_per_ant = 2) :
  (n_spiders * eyes_per_spider + n_ants * eyes_per_ant) = 124 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_eyes_l5_551


namespace NUMINAMATH_GPT_max_principals_in_10_years_l5_546

theorem max_principals_in_10_years (h : ∀ p : ℕ, 4 * p ≤ 10) :
  ∃ n : ℕ, n ≤ 3 ∧ n = 3 :=
sorry

end NUMINAMATH_GPT_max_principals_in_10_years_l5_546


namespace NUMINAMATH_GPT_ratio_of_medians_to_sides_l5_521

theorem ratio_of_medians_to_sides (a b c : ℝ) (m_a m_b m_c : ℝ) 
  (h1: m_a = 1/2 * (2 * b^2 + 2 * c^2 - a^2)^(1/2))
  (h2: m_b = 1/2 * (2 * a^2 + 2 * c^2 - b^2)^(1/2))
  (h3: m_c = 1/2 * (2 * a^2 + 2 * b^2 - c^2)^(1/2)) :
  (m_a*m_a + m_b*m_b + m_c*m_c) / (a*a + b*b + c*c) = 3/4 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_medians_to_sides_l5_521


namespace NUMINAMATH_GPT_injectivity_of_composition_l5_542

variable {R : Type*} [LinearOrderedField R]

def injective (f : R → R) := ∀ a b, f a = f b → a = b

theorem injectivity_of_composition {f g : R → R} (h : injective (g ∘ f)) : injective f :=
by
  sorry

end NUMINAMATH_GPT_injectivity_of_composition_l5_542


namespace NUMINAMATH_GPT_bobby_jumps_per_second_as_adult_l5_548

-- Define the conditions as variables
def child_jumps_per_minute : ℕ := 30
def additional_jumps_as_adult : ℕ := 30

-- Theorem statement
theorem bobby_jumps_per_second_as_adult :
  (child_jumps_per_minute + additional_jumps_as_adult) / 60 = 1 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_bobby_jumps_per_second_as_adult_l5_548


namespace NUMINAMATH_GPT_count_zeros_in_10000_power_50_l5_558

theorem count_zeros_in_10000_power_50 :
  10000^50 = 10^200 :=
by
  have h1 : 10000 = 10^4 := by sorry
  have h2 : (10^4)^50 = 10^(4 * 50) := by sorry
  exact h2.trans (by norm_num)

end NUMINAMATH_GPT_count_zeros_in_10000_power_50_l5_558


namespace NUMINAMATH_GPT_negation_of_exists_prop_l5_514

variable (n : ℕ)

theorem negation_of_exists_prop :
  (¬ (∃ n : ℕ, n^2 > 2^n)) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_prop_l5_514


namespace NUMINAMATH_GPT_sum_of_ages_l5_579

theorem sum_of_ages {a b c : ℕ} (h1 : a * b * c = 72) (h2 : b < a) (h3 : a < c) : a + b + c = 13 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l5_579


namespace NUMINAMATH_GPT_polynomial_divisible_by_cube_l5_538

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := 
  n^2 * x^(n+2) - (2 * n^2 + 2 * n - 1) * x^(n+1) + (n + 1)^2 * x^n - x - 1

theorem polynomial_divisible_by_cube (n : ℕ) (h : n > 0) : 
  ∃ Q, P n x = (x - 1)^3 * Q :=
sorry

end NUMINAMATH_GPT_polynomial_divisible_by_cube_l5_538


namespace NUMINAMATH_GPT_no_correct_option_l5_536

-- Define the given table as a list of pairs
def table :=
  [(1, -2), (2, 0), (3, 2), (4, 6), (5, 12), (6, 20)]

-- Define the given functions as potential options
def optionA (x : ℕ) : ℤ := x^2 - 5 * x + 4
def optionB (x : ℕ) : ℤ := x^2 - 3 * x
def optionC (x : ℕ) : ℤ := x^3 - 3 * x^2 + 2 * x
def optionD (x : ℕ) : ℤ := 2 * x^2 - 4 * x - 2
def optionE (x : ℕ) : ℤ := x^2 - 4 * x + 2

-- Prove that there is no correct option among the given options that matches the table
theorem no_correct_option : 
  ¬(∀ p ∈ table, p.snd = optionA p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionB p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionC p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionD p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionE p.fst) :=
by sorry

end NUMINAMATH_GPT_no_correct_option_l5_536


namespace NUMINAMATH_GPT_condition_equiv_l5_541

theorem condition_equiv (p q : Prop) : (¬ (p ∧ q) ∧ (p ∨ q)) ↔ ((p ∨ q) ∧ (¬ p ↔ q)) :=
  sorry

end NUMINAMATH_GPT_condition_equiv_l5_541


namespace NUMINAMATH_GPT_cost_of_parakeet_l5_599

theorem cost_of_parakeet
  (P Py K : ℕ) -- defining the costs of parakeet, puppy, and kitten
  (h1 : Py = 3 * P) -- puppy is three times the cost of parakeet
  (h2 : P = K / 2) -- parakeet is half the cost of kitten
  (h3 : 2 * Py + 2 * K + 3 * P = 130) -- total cost equation
  : P = 10 := 
sorry

end NUMINAMATH_GPT_cost_of_parakeet_l5_599


namespace NUMINAMATH_GPT_ThreeStudentsGotA_l5_560

-- Definitions of students receiving A grades
variable (Edward Fiona George Hannah Ian : Prop)

-- Conditions given in the problem
axiom H1 : Edward → Fiona
axiom H2 : Fiona → George
axiom H3 : George → Hannah
axiom H4 : Hannah → Ian
axiom H5 : (Edward → False) ∧ (Fiona → False)

-- Theorem stating the final result
theorem ThreeStudentsGotA : (George ∧ Hannah ∧ Ian) ∧ 
                            (¬Edward ∧ ¬Fiona) ∧ 
                            (Edward ∨ Fiona ∨ George ∨ Hannah ∨ Ian) :=
by
  sorry

end NUMINAMATH_GPT_ThreeStudentsGotA_l5_560


namespace NUMINAMATH_GPT_value_of_expression_when_x_is_3_l5_571

theorem value_of_expression_when_x_is_3 :
  (3^6 - 6*3 = 711) :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_when_x_is_3_l5_571


namespace NUMINAMATH_GPT_possible_values_of_a₁_l5_583

-- Define arithmetic progression with first term a₁ and common difference d
def arithmetic_progression (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

-- Define the sum of the first 7 terms of the arithmetic progression
def sum_first_7_terms (a₁ d : ℤ) : ℤ := 7 * a₁ + 21 * d

-- Define the conditions given
def condition1 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 7) * (arithmetic_progression a₁ d 12) > (sum_first_7_terms a₁ d) + 20

def condition2 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 9) * (arithmetic_progression a₁ d 10) < (sum_first_7_terms a₁ d) + 44

-- The main problem to prove
def problem (a₁ : ℤ) (d : ℤ) : Prop := 
  condition1 a₁ d ∧ condition2 a₁ d

-- The theorem statement to prove
theorem possible_values_of_a₁ (a₁ d : ℤ) : problem a₁ d → a₁ = -9 ∨ a₁ = -8 ∨ a₁ = -7 ∨ a₁ = -6 ∨ a₁ = -4 ∨ a₁ = -3 ∨ a₁ = -2 ∨ a₁ = -1 := 
by sorry

end NUMINAMATH_GPT_possible_values_of_a₁_l5_583


namespace NUMINAMATH_GPT_student_number_choice_l5_570

theorem student_number_choice (x : ℤ) (h : 3 * x - 220 = 110) : x = 110 :=
by sorry

end NUMINAMATH_GPT_student_number_choice_l5_570


namespace NUMINAMATH_GPT_cylinder_ellipse_eccentricity_l5_516

noncomputable def eccentricity_of_ellipse (diameter : ℝ) (angle : ℝ) : ℝ :=
  let r := diameter / 2
  let b := r
  let a := r / (Real.cos angle)
  let c := Real.sqrt (a^2 - b^2)
  c / a

theorem cylinder_ellipse_eccentricity :
  eccentricity_of_ellipse 12 (Real.pi / 6) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_ellipse_eccentricity_l5_516


namespace NUMINAMATH_GPT_prove_odd_function_definition_l5_532

theorem prove_odd_function_definition (f : ℝ → ℝ) 
  (odd : ∀ x : ℝ, f (-x) = -f x)
  (pos_def : ∀ x : ℝ, 0 < x → f x = 2 * x ^ 2 - x + 1) :
  ∀ x : ℝ, x < 0 → f x = -2 * x ^ 2 - x - 1 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_prove_odd_function_definition_l5_532


namespace NUMINAMATH_GPT_remainder_when_7n_divided_by_4_l5_569

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end NUMINAMATH_GPT_remainder_when_7n_divided_by_4_l5_569


namespace NUMINAMATH_GPT_jose_share_of_profit_l5_584

def investment_months (amount : ℕ) (months : ℕ) : ℕ := amount * months

def profit_share (investment_months : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment_months * total_profit) / total_investment_months

theorem jose_share_of_profit :
  let tom_investment := 30000
  let jose_investment := 45000
  let total_profit := 36000
  let tom_months := 12
  let jose_months := 10
  let tom_investment_months := investment_months tom_investment tom_months
  let jose_investment_months := investment_months jose_investment jose_months
  let total_investment_months := tom_investment_months + jose_investment_months
  profit_share jose_investment_months total_investment_months total_profit = 20000 :=
by
  sorry

end NUMINAMATH_GPT_jose_share_of_profit_l5_584


namespace NUMINAMATH_GPT_value_spent_more_than_l5_539

theorem value_spent_more_than (x : ℕ) (h : 8 * 12 + (x + 8) = 117) : x = 13 :=
by
  sorry

end NUMINAMATH_GPT_value_spent_more_than_l5_539


namespace NUMINAMATH_GPT_minimum_value_of_f_l5_515

def f (x a : ℝ) : ℝ := abs (x + 1) + abs (a * x + 1)

theorem minimum_value_of_f (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3 / 2) →
  (∃ x : ℝ, f x a = 3 / 2) →
  (a = -1 / 2 ∨ a = -2) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l5_515


namespace NUMINAMATH_GPT_problem_inequality_l5_588

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem problem_inequality (a x : ℝ) (h : a ∈ Set.Iic (-1/Real.exp 2)) :
  f a x ≥ 2 * a * x - x * Real.exp (a * x - 1) := 
sorry

end NUMINAMATH_GPT_problem_inequality_l5_588
