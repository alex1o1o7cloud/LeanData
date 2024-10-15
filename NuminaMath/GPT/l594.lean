import Mathlib

namespace NUMINAMATH_GPT_max_sum_prod_48_l594_59401

theorem max_sum_prod_48 (spadesuit heartsuit : Nat) (h: spadesuit * heartsuit = 48) : spadesuit + heartsuit ≤ 49 :=
sorry

end NUMINAMATH_GPT_max_sum_prod_48_l594_59401


namespace NUMINAMATH_GPT_mass_percentage_of_Br_in_BaBr2_l594_59448

theorem mass_percentage_of_Br_in_BaBr2 :
  let Ba_molar_mass := 137.33
  let Br_molar_mass := 79.90
  let BaBr2_molar_mass := Ba_molar_mass + 2 * Br_molar_mass
  let mass_percentage_Br := (2 * Br_molar_mass / BaBr2_molar_mass) * 100
  mass_percentage_Br = 53.80 :=
by
  let Ba_molar_mass := 137.33
  let Br_molar_mass := 79.90
  let BaBr2_molar_mass := Ba_molar_mass + 2 * Br_molar_mass
  let mass_percentage_Br := (2 * Br_molar_mass / BaBr2_molar_mass) * 100
  sorry

end NUMINAMATH_GPT_mass_percentage_of_Br_in_BaBr2_l594_59448


namespace NUMINAMATH_GPT_largest_band_members_l594_59485

theorem largest_band_members :
  ∃ (r x : ℕ), r * x + 3 = 107 ∧ (r - 3) * (x + 2) = 107 ∧ r * x < 147 :=
sorry

end NUMINAMATH_GPT_largest_band_members_l594_59485


namespace NUMINAMATH_GPT_probability_of_drawing_ball_1_is_2_over_5_l594_59417

noncomputable def probability_of_drawing_ball_1 : ℚ :=
  let total_balls := [1, 2, 3, 4, 5]
  let draw_pairs := [ (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5) ]
  let favorable_pairs := [ (1, 2), (1, 3), (1, 4), (1, 5) ]
  (favorable_pairs.length : ℚ) / (draw_pairs.length : ℚ)

theorem probability_of_drawing_ball_1_is_2_over_5 :
  probability_of_drawing_ball_1 = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_probability_of_drawing_ball_1_is_2_over_5_l594_59417


namespace NUMINAMATH_GPT_triangle_problem_l594_59438

/--
Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C, respectively, 
where:
1. b * (sin B - sin C) = a * sin A - c * sin C
2. a = 2 * sqrt 3
3. the area of triangle ABC is 2 * sqrt 3

Prove:
1. A = π / 3
2. The perimeter of triangle ABC is 2 * sqrt 3 + 6
-/
theorem triangle_problem 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : b * (Real.sin B - Real.sin C) = a * Real.sin A - c * Real.sin C)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : 0.5 * b * c * Real.sin A = 2 * Real.sqrt 3) :
  A = Real.pi / 3 ∧ a + b + c = 2 * Real.sqrt 3 + 6 := 
sorry

end NUMINAMATH_GPT_triangle_problem_l594_59438


namespace NUMINAMATH_GPT_direct_proportion_l594_59454

theorem direct_proportion : 
  ∃ k, (∀ x, y = k * x) ↔ (y = -2 * x) :=
by
  sorry

end NUMINAMATH_GPT_direct_proportion_l594_59454


namespace NUMINAMATH_GPT_hike_length_l594_59458

-- Definitions of conditions
def initial_water : ℕ := 6
def final_water : ℕ := 1
def hike_duration : ℕ := 2
def leak_rate : ℕ := 1
def last_mile_drunk : ℕ := 1
def first_part_drink_rate : ℚ := 2 / 3

-- Statement to prove
theorem hike_length (hike_duration : ℕ) (initial_water : ℕ) (final_water : ℕ) (leak_rate : ℕ) 
  (last_mile_drunk : ℕ) (first_part_drink_rate : ℚ) : 
  hike_duration = 2 → 
  initial_water = 6 → 
  final_water = 1 → 
  leak_rate = 1 → 
  last_mile_drunk = 1 → 
  first_part_drink_rate = 2 / 3 → 
  ∃ miles : ℕ, miles = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_hike_length_l594_59458


namespace NUMINAMATH_GPT_intersection_A_B_l594_59456

def A : Set ℤ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℤ := {-2, -1, 0, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 2, 3} :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l594_59456


namespace NUMINAMATH_GPT_xyz_value_l594_59465

theorem xyz_value (x y z : ℝ) (h1 : 2 * x + 3 * y + z = 13) 
                              (h2 : 4 * x^2 + 9 * y^2 + z^2 - 2 * x + 15 * y + 3 * z = 82) : 
  x * y * z = 12 := 
by 
  sorry

end NUMINAMATH_GPT_xyz_value_l594_59465


namespace NUMINAMATH_GPT_rhombus_diagonal_l594_59423

theorem rhombus_diagonal (d2 : ℝ) (area : ℝ) (d1 : ℝ) : d2 = 15 → area = 127.5 → d1 = 17 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_l594_59423


namespace NUMINAMATH_GPT_inequality_bounds_of_xyz_l594_59493

theorem inequality_bounds_of_xyz
  (x y z : ℝ)
  (h1 : x < y) (h2 : y < z)
  (h3 : x + y + z = 6)
  (h4 : x * y + y * z + z * x = 9) :
  0 < x ∧ x < 1 ∧ 1 < y ∧ y < 3 ∧ 3 < z ∧ z < 4 := 
sorry

end NUMINAMATH_GPT_inequality_bounds_of_xyz_l594_59493


namespace NUMINAMATH_GPT_leila_toys_l594_59408

theorem leila_toys:
  ∀ (x : ℕ),
  (∀ l m : ℕ, l = 2 * x ∧ m = 3 * 19 ∧ m = l + 7 → x = 25) :=
by
  sorry

end NUMINAMATH_GPT_leila_toys_l594_59408


namespace NUMINAMATH_GPT_math_problem_l594_59479

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def g' : ℝ → ℝ := sorry

def condition1 (x : ℝ) : Prop := f (x + 3) = g (-x) + 4
def condition2 (x : ℝ) : Prop := f' x + g' (1 + x) = 0
def even_function (x : ℝ) : Prop := g (2 * x + 1) = g (- (2 * x + 1))

theorem math_problem (x : ℝ) :
  (∀ x, condition1 x) →
  (∀ x, condition2 x) →
  (∀ x, even_function x) →
  (g' 1 = 0) ∧
  (∀ x, f (1 - x) = f (x + 3)) ∧
  (∀ x, f' x = f' (-x + 2)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_math_problem_l594_59479


namespace NUMINAMATH_GPT_luke_initial_stickers_l594_59419

theorem luke_initial_stickers (x : ℕ) (h : x + 12 + 20 - 5 - 8 = 39) : x = 20 := 
by 
  sorry

end NUMINAMATH_GPT_luke_initial_stickers_l594_59419


namespace NUMINAMATH_GPT_student_average_always_less_l594_59415

theorem student_average_always_less (w x y z: ℝ) (hwx: w < x) (hxy: x < y) (hyz: y < z) :
  let A' := (w + x + y + z) / 4
  let B' := (2 * w + 2 * x + y + z) / 6
  B' < A' :=
by
  intro A' B'
  sorry

end NUMINAMATH_GPT_student_average_always_less_l594_59415


namespace NUMINAMATH_GPT_jason_borrowed_amount_l594_59490

def earning_per_six_hours : ℤ :=
  2 + 4 + 6 + 2 + 4 + 6

def total_hours_worked : ℤ :=
  48

def cycle_length : ℤ :=
  6

def total_cycles : ℤ :=
  total_hours_worked / cycle_length

def total_amount_borrowed : ℤ :=
  total_cycles * earning_per_six_hours

theorem jason_borrowed_amount : total_amount_borrowed = 192 :=
  by
    -- Here we use the definition and conditions to prove the equivalence
    -- of the calculation to the problem statement.
    sorry

end NUMINAMATH_GPT_jason_borrowed_amount_l594_59490


namespace NUMINAMATH_GPT_correct_option_is_c_l594_59453

variable {x y : ℕ}

theorem correct_option_is_c (hx : (x^2)^3 = x^6) :
  (∀ x : ℕ, x * x^2 ≠ x^2) →
  (∀ x y : ℕ, (x + y)^2 ≠ x^2 + y^2) →
  (∃ x : ℕ, x^2 + x^2 ≠ x^4) →
  (x^2)^3 = x^6 :=
by
  intros h1 h2 h3
  exact hx

end NUMINAMATH_GPT_correct_option_is_c_l594_59453


namespace NUMINAMATH_GPT_closest_pressure_reading_l594_59477

theorem closest_pressure_reading (x : ℝ) (h : 102.4 ≤ x ∧ x ≤ 102.8) :
    (|x - 102.5| > |x - 102.6| ∧ |x - 102.6| < |x - 102.7| ∧ |x - 102.6| < |x - 103.0|) → x = 102.6 :=
by
  sorry

end NUMINAMATH_GPT_closest_pressure_reading_l594_59477


namespace NUMINAMATH_GPT_petya_mistake_l594_59494

theorem petya_mistake :
  (35 + 10 - 41 = 42 + 12 - 50) →
  (35 + 10 - 45 = 42 + 12 - 54) →
  (5 * (7 + 2 - 9) = 6 * (7 + 2 - 9)) →
  False :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_petya_mistake_l594_59494


namespace NUMINAMATH_GPT_wife_weekly_savings_correct_l594_59422

-- Define constants
def monthly_savings_husband := 225
def num_months := 4
def weeks_per_month := 4
def num_weeks := num_months * weeks_per_month
def stocks_per_share := 50
def num_shares := 25
def invested_amount := num_shares * stocks_per_share
def total_savings := 2 * invested_amount

-- Weekly savings amount to prove
def weekly_savings_wife := 100

-- Total savings calculation condition
theorem wife_weekly_savings_correct :
  (monthly_savings_husband * num_months + weekly_savings_wife * num_weeks) = total_savings :=
by
  sorry

end NUMINAMATH_GPT_wife_weekly_savings_correct_l594_59422


namespace NUMINAMATH_GPT_factorization_solution_1_factorization_solution_2_factorization_solution_3_l594_59410

noncomputable def factorization_problem_1 (m : ℝ) : Prop :=
  -3 * m^3 + 12 * m = -3 * m * (m + 2) * (m - 2)

noncomputable def factorization_problem_2 (x y : ℝ) : Prop :=
  2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2

noncomputable def factorization_problem_3 (a : ℝ) : Prop :=
  a^4 + 3 * a^2 - 4 = (a^2 + 4) * (a + 1) * (a - 1)

-- Lean statements for the proofs
theorem factorization_solution_1 (m : ℝ) : factorization_problem_1 m :=
  by sorry

theorem factorization_solution_2 (x y : ℝ) : factorization_problem_2 x y :=
  by sorry

theorem factorization_solution_3 (a : ℝ) : factorization_problem_3 a :=
  by sorry

end NUMINAMATH_GPT_factorization_solution_1_factorization_solution_2_factorization_solution_3_l594_59410


namespace NUMINAMATH_GPT_xyz_sum_eq_eleven_l594_59445

theorem xyz_sum_eq_eleven (x y z : ℝ) (h : (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0) : x + y + z = 11 :=
sorry

end NUMINAMATH_GPT_xyz_sum_eq_eleven_l594_59445


namespace NUMINAMATH_GPT_sum_even_integers_102_to_200_l594_59497

theorem sum_even_integers_102_to_200 :
  let S := (List.range' 102 (200 - 102 + 1)).filter (λ x => x % 2 = 0)
  List.sum S = 7550 := by
{
  sorry
}

end NUMINAMATH_GPT_sum_even_integers_102_to_200_l594_59497


namespace NUMINAMATH_GPT_least_number_remainder_5_l594_59416

theorem least_number_remainder_5 (n : ℕ) : 
  n % 12 = 5 ∧ n % 15 = 5 ∧ n % 20 = 5 ∧ n % 54 = 5 → n = 545 := 
  by
  sorry

end NUMINAMATH_GPT_least_number_remainder_5_l594_59416


namespace NUMINAMATH_GPT_expression_value_l594_59424

theorem expression_value : (100 - (1000 - 300)) - (1000 - (300 - 100)) = -1400 := by
  sorry

end NUMINAMATH_GPT_expression_value_l594_59424


namespace NUMINAMATH_GPT_function_range_l594_59475

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x^2 - 1) * (x^2 + a * x + b)

theorem function_range (a b : ℝ) (h_symm : ∀ x : ℝ, f (6 - x) a b = f x a b) :
  a = -12 ∧ b = 35 ∧ (∀ y, ∃ x : ℝ, f x (-12) 35 = y ↔ -36 ≤ y) :=
by
  sorry

end NUMINAMATH_GPT_function_range_l594_59475


namespace NUMINAMATH_GPT_no_2014_ambiguous_integer_exists_l594_59467

theorem no_2014_ambiguous_integer_exists :
  ∀ k : ℕ, (∃ m : ℤ, k^2 - 8056 = m^2) → (∃ n : ℤ, k^2 + 8056 = n^2) → false :=
by
  -- Proof is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_no_2014_ambiguous_integer_exists_l594_59467


namespace NUMINAMATH_GPT_find_sum_of_numbers_l594_59491

variables (a b c : ℕ) (h_ratio : a * 7 = b * 5 ∧ b * 9 = c * 7) (h_lcm : Nat.lcm a (Nat.lcm b c) = 6300)

theorem find_sum_of_numbers (h_ratio : a * 7 = b * 5 ∧ b * 9 = c * 7) (h_lcm : Nat.lcm a (Nat.lcm b c) = 6300) :
  a + b + c = 14700 :=
sorry

end NUMINAMATH_GPT_find_sum_of_numbers_l594_59491


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l594_59492

theorem simplify_and_evaluate_expression
    (a b : ℤ)
    (h1 : a = -1/3)
    (h2 : b = -2) :
  ((3 * a + b)^2 - (3 * a + b) * (3 * a - b)) / (2 * b) = -3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l594_59492


namespace NUMINAMATH_GPT_rahul_and_sham_together_complete_task_in_35_days_l594_59446

noncomputable def rahul_rate (W : ℝ) : ℝ := W / 60
noncomputable def sham_rate (W : ℝ) : ℝ := W / 84
noncomputable def combined_rate (W : ℝ) := rahul_rate W + sham_rate W

theorem rahul_and_sham_together_complete_task_in_35_days (W : ℝ) :
  (W / combined_rate W) = 35 :=
by
  sorry

end NUMINAMATH_GPT_rahul_and_sham_together_complete_task_in_35_days_l594_59446


namespace NUMINAMATH_GPT_net_emails_received_l594_59433

-- Define the conditions
def emails_received_morning : ℕ := 3
def emails_sent_morning : ℕ := 2
def emails_received_afternoon : ℕ := 5
def emails_sent_afternoon : ℕ := 1

-- Define the problem statement
theorem net_emails_received :
  emails_received_morning - emails_sent_morning + emails_received_afternoon - emails_sent_afternoon = 5 := by
  sorry

end NUMINAMATH_GPT_net_emails_received_l594_59433


namespace NUMINAMATH_GPT_problem_solution_l594_59403

theorem problem_solution (u v : ℤ) (h₁ : 0 < v) (h₂ : v < u) (h₃ : u^2 + 3 * u * v = 451) : u + v = 21 :=
sorry

end NUMINAMATH_GPT_problem_solution_l594_59403


namespace NUMINAMATH_GPT_total_cost_of_groceries_l594_59451

noncomputable def M (R : ℝ) : ℝ := 24 * R / 10
noncomputable def F : ℝ := 22

theorem total_cost_of_groceries (R : ℝ) (hR : 2 * R = 22) :
  10 * M R = 24 * R ∧ F = 2 * R ∧ F = 22 →
  4 * M R + 3 * R + 5 * F = 248.6 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_groceries_l594_59451


namespace NUMINAMATH_GPT_find_a_l594_59478

theorem find_a {a : ℝ} :
  (∀ x : ℝ, (ax - 1) / (x + 1) < 0 → (x < -1 ∨ x > -1 / 2)) → a = -2 :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_find_a_l594_59478


namespace NUMINAMATH_GPT_find_girls_l594_59425

theorem find_girls (n : ℕ) (h : 1 - (1 / Nat.choose (3 + n) 3) = 34 / 35) : n = 4 :=
  sorry

end NUMINAMATH_GPT_find_girls_l594_59425


namespace NUMINAMATH_GPT_smallest_other_integer_l594_59413

-- Definitions of conditions
def gcd_condition (a b : ℕ) (x : ℕ) : Prop := 
  Nat.gcd a b = x + 5

def lcm_condition (a b : ℕ) (x : ℕ) : Prop := 
  Nat.lcm a b = x * (x + 5)

def sum_condition (a b : ℕ) : Prop := 
  a + b < 100

-- Main statement incorporating all conditions
theorem smallest_other_integer {x b : ℕ} (hx_pos : x > 0)
  (h_gcd : gcd_condition 45 b x)
  (h_lcm : lcm_condition 45 b x)
  (h_sum : sum_condition 45 b) :
  b = 12 :=
sorry

end NUMINAMATH_GPT_smallest_other_integer_l594_59413


namespace NUMINAMATH_GPT_total_amount_is_24_l594_59435

-- Define the original price of a tub of ice cream
def original_price_ice_cream : ℕ := 12

-- Define the discount per tub of ice cream
def discount_per_tub : ℕ := 2

-- Define the discounted price of a tub of ice cream
def discounted_price_ice_cream : ℕ := original_price_ice_cream - discount_per_tub

-- Define the price for 5 cans of juice
def price_per_5_cans_of_juice : ℕ := 2

-- Define the number of cans of juice bought
def cans_of_juice_bought : ℕ := 10

-- Calculate the total cost for two tubs of ice cream and 10 cans of juice
def total_cost (p1 p2 : ℕ) : ℕ := 2 * p1 + (price_per_5_cans_of_juice * (cans_of_juice_bought / 5))

-- Prove that the total cost is $24
theorem total_amount_is_24 : total_cost discounted_price_ice_cream price_per_5_cans_of_juice = 24 := by
  sorry

end NUMINAMATH_GPT_total_amount_is_24_l594_59435


namespace NUMINAMATH_GPT_find_k_l594_59430

theorem find_k (x y k : ℝ)
  (h1 : x - 4 * y + 3 ≤ 0)
  (h2 : 3 * x + 5 * y - 25 ≤ 0)
  (h3 : x ≥ 1)
  (h4 : ∃ z, z = k * x + y ∧ z = 12)
  (h5 : ∃ z', z' = k * x + y ∧ z' = 3) :
  k = 2 :=
by sorry

end NUMINAMATH_GPT_find_k_l594_59430


namespace NUMINAMATH_GPT_calculate_f_at_5_l594_59488

noncomputable def g (y : ℝ) : ℝ := (1 / 2) * y^2

noncomputable def f (x y : ℝ) : ℝ := 2 * x^2 + g y

theorem calculate_f_at_5 (y : ℝ) (h1 : f 2 y = 50) (h2 : y = 2*Real.sqrt 21) :
  f 5 y = 92 :=
by
  sorry

end NUMINAMATH_GPT_calculate_f_at_5_l594_59488


namespace NUMINAMATH_GPT_total_spider_legs_l594_59418

theorem total_spider_legs (num_legs_single_spider group_spider_count: ℕ) 
      (h1: num_legs_single_spider = 8) 
      (h2: group_spider_count = (num_legs_single_spider / 2) + 10) :
      group_spider_count * num_legs_single_spider = 112 := 
by
  sorry

end NUMINAMATH_GPT_total_spider_legs_l594_59418


namespace NUMINAMATH_GPT_ten_crates_probability_l594_59476

theorem ten_crates_probability (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) :
  let num_crates := 10
  let crate_dimensions := [3, 4, 6]
  let target_height := 41

  -- Definition of the generating function coefficients and constraints will be complex,
  -- so stating the specific problem directly.
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ m = 190 ∧ n = 2187 →
  let probability := (m : ℚ) / (n : ℚ)
  probability = (190 : ℚ) / 2187 := 
by
  sorry

end NUMINAMATH_GPT_ten_crates_probability_l594_59476


namespace NUMINAMATH_GPT_continuous_stripe_probability_l594_59405

noncomputable def probability_continuous_stripe_encircle_cube : ℚ :=
  let total_combinations : ℕ := 2^6
  let favor_combinations : ℕ := 3 * 4 -- 3 pairs of parallel faces, with 4 valid combinations each
  favor_combinations / total_combinations

theorem continuous_stripe_probability :
  probability_continuous_stripe_encircle_cube = 3 / 16 :=
by
  sorry

end NUMINAMATH_GPT_continuous_stripe_probability_l594_59405


namespace NUMINAMATH_GPT_smallest_interesting_number_l594_59469

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end NUMINAMATH_GPT_smallest_interesting_number_l594_59469


namespace NUMINAMATH_GPT_coefficients_of_polynomial_l594_59437

theorem coefficients_of_polynomial (a_5 a_4 a_3 a_2 a_1 a_0 : ℝ) :
  (∀ x : ℝ, x^5 = a_5 * (2*x + 1)^5 + a_4 * (2*x + 1)^4 + a_3 * (2*x + 1)^3 + a_2 * (2*x + 1)^2 + a_1 * (2*x + 1) + a_0) →
  a_5 = 1/32 ∧ a_4 = -5/32 :=
by sorry

end NUMINAMATH_GPT_coefficients_of_polynomial_l594_59437


namespace NUMINAMATH_GPT_total_cost_is_correct_l594_59414

def goldfish_price := 3
def goldfish_quantity := 15
def blue_fish_price := 6
def blue_fish_quantity := 7
def neon_tetra_price := 2
def neon_tetra_quantity := 10
def angelfish_price := 8
def angelfish_quantity := 5

def total_cost := goldfish_quantity * goldfish_price 
                 + blue_fish_quantity * blue_fish_price 
                 + neon_tetra_quantity * neon_tetra_price 
                 + angelfish_quantity * angelfish_price

theorem total_cost_is_correct : total_cost = 147 :=
by
  -- Summary of the proof steps goes here
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l594_59414


namespace NUMINAMATH_GPT_max_togs_possible_l594_59459

def tag_cost : ℕ := 3
def tig_cost : ℕ := 4
def tog_cost : ℕ := 8
def total_budget : ℕ := 100
def min_tags : ℕ := 1
def min_tigs : ℕ := 1
def min_togs : ℕ := 1

theorem max_togs_possible : 
  ∃ (tags tigs togs : ℕ), tags ≥ min_tags ∧ tigs ≥ min_tigs ∧ togs ≥ min_togs ∧ 
  tag_cost * tags + tig_cost * tigs + tog_cost * togs = total_budget ∧ togs = 11 :=
sorry

end NUMINAMATH_GPT_max_togs_possible_l594_59459


namespace NUMINAMATH_GPT_real_number_unique_l594_59429

variable (a x : ℝ)

theorem real_number_unique (h1 : (a + 3) * (a + 3) = x)
  (h2 : (2 * a - 9) * (2 * a - 9) = x) : x = 25 := by
  sorry

end NUMINAMATH_GPT_real_number_unique_l594_59429


namespace NUMINAMATH_GPT_bills_needed_can_pay_groceries_l594_59427

theorem bills_needed_can_pay_groceries 
  (cans_of_soup : ℕ := 6) (price_per_can : ℕ := 2)
  (loaves_of_bread : ℕ := 3) (price_per_loaf : ℕ := 5)
  (boxes_of_cereal : ℕ := 4) (price_per_box : ℕ := 3)
  (gallons_of_milk : ℕ := 2) (price_per_gallon : ℕ := 4)
  (apples : ℕ := 7) (price_per_apple : ℕ := 1)
  (bags_of_cookies : ℕ := 5) (price_per_bag : ℕ := 3)
  (bottles_of_olive_oil : ℕ := 1) (price_per_bottle : ℕ := 8)
  : ∃ (bills_needed : ℕ), bills_needed = 4 :=
by
  let total_cost := (cans_of_soup * price_per_can) + 
                    (loaves_of_bread * price_per_loaf) +
                    (boxes_of_cereal * price_per_box) +
                    (gallons_of_milk * price_per_gallon) +
                    (apples * price_per_apple) +
                    (bags_of_cookies * price_per_bag) +
                    (bottles_of_olive_oil * price_per_bottle)
  let bills_needed := (total_cost + 19) / 20   -- Calculating ceiling of total_cost / 20
  sorry

end NUMINAMATH_GPT_bills_needed_can_pay_groceries_l594_59427


namespace NUMINAMATH_GPT_john_frank_age_ratio_l594_59432

theorem john_frank_age_ratio
  (F J : ℕ)
  (h1 : F + 4 = 16)
  (h2 : J - F = 15)
  (h3 : ∃ k : ℕ, J + 3 = k * (F + 3)) :
  (J + 3) / (F + 3) = 2 :=
by
  sorry

end NUMINAMATH_GPT_john_frank_age_ratio_l594_59432


namespace NUMINAMATH_GPT_playground_area_l594_59464

theorem playground_area (L B : ℕ) (h1 : B = 6 * L) (h2 : B = 420)
  (A_total A_playground : ℕ) (h3 : A_total = L * B) 
  (h4 : A_playground = A_total / 7) :
  A_playground = 4200 :=
by sorry

end NUMINAMATH_GPT_playground_area_l594_59464


namespace NUMINAMATH_GPT_segment_AC_length_l594_59444

-- Define segments AB and BC
def AB : ℝ := 4
def BC : ℝ := 3

-- Define segment AC in terms of the conditions given
def AC_case1 : ℝ := AB - BC
def AC_case2 : ℝ := AB + BC

-- The proof problem statement
theorem segment_AC_length : AC_case1 = 1 ∨ AC_case2 = 7 := by
  sorry

end NUMINAMATH_GPT_segment_AC_length_l594_59444


namespace NUMINAMATH_GPT_problem_a4_inv_a4_eq_seven_l594_59468

theorem problem_a4_inv_a4_eq_seven (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^4 + (1/a)^4 = 7 :=
sorry

end NUMINAMATH_GPT_problem_a4_inv_a4_eq_seven_l594_59468


namespace NUMINAMATH_GPT_clock_angle_34030_l594_59470

noncomputable def calculate_angle (h m s : ℕ) : ℚ :=
  abs ((60 * h - 11 * (m + s / 60)) / 2)

theorem clock_angle_34030 : calculate_angle 3 40 30 = 130 :=
by
  sorry

end NUMINAMATH_GPT_clock_angle_34030_l594_59470


namespace NUMINAMATH_GPT_no_three_digit_numbers_divisible_by_30_l594_59443

def digits_greater_than_6 (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d > 6

theorem no_three_digit_numbers_divisible_by_30 :
  ∀ n, (100 ≤ n ∧ n < 1000 ∧ digits_greater_than_6 n ∧ n % 30 = 0) → false :=
by
  sorry

end NUMINAMATH_GPT_no_three_digit_numbers_divisible_by_30_l594_59443


namespace NUMINAMATH_GPT_reversed_number_increase_l594_59402

theorem reversed_number_increase (a b c : ℕ) 
  (h1 : a + b + c = 10) 
  (h2 : b = a + c)
  (h3 : a = 2 ∧ b = 5 ∧ c = 3) :
  (c * 100 + b * 10 + a) - (a * 100 + b * 10 + c) = 99 :=
by
  sorry

end NUMINAMATH_GPT_reversed_number_increase_l594_59402


namespace NUMINAMATH_GPT_three_digit_number_condition_l594_59466

theorem three_digit_number_condition (x y z : ℕ) (h₀ : 1 ≤ x ∧ x ≤ 9) (h₁ : 0 ≤ y ∧ y ≤ 9) (h₂ : 0 ≤ z ∧ z ≤ 9)
(h₃ : 100 * x + 10 * y + z = 34 * (x + y + z)) : 
100 * x + 10 * y + z = 102 ∨ 100 * x + 10 * y + z = 204 ∨ 100 * x + 10 * y + z = 306 ∨ 100 * x + 10 * y + z = 408 :=
sorry

end NUMINAMATH_GPT_three_digit_number_condition_l594_59466


namespace NUMINAMATH_GPT_number_of_b_values_l594_59434

-- Let's define the conditions and the final proof required.
def inequations (x b : ℤ) : Prop := 
  (3 * x > 4 * x - 4) ∧
  (4 * x - b > -8) ∧
  (5 * x < b + 13)

theorem number_of_b_values :
  (∀ x : ℤ, 1 ≤ x → x ≠ 3 → ¬ inequations x b) →
  (∃ (b_values : Finset ℤ), 
      (∀ b ∈ b_values, inequations 3 b) ∧ 
      (b_values.card = 7)) :=
sorry

end NUMINAMATH_GPT_number_of_b_values_l594_59434


namespace NUMINAMATH_GPT_polynomial_quotient_l594_59441

theorem polynomial_quotient : 
  (12 * x^3 + 20 * x^2 - 7 * x + 4) / (3 * x + 4) = 4 * x^2 + (4/3) * x - 37/9 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_quotient_l594_59441


namespace NUMINAMATH_GPT_smallest_sum_p_q_l594_59472

theorem smallest_sum_p_q (p q : ℕ) (h_pos : 1 < p) (h_cond : (p^2 * q - 1) = (2021 * p * q) / 2021) : p + q = 44 :=
sorry

end NUMINAMATH_GPT_smallest_sum_p_q_l594_59472


namespace NUMINAMATH_GPT_probability_of_two_red_two_green_l594_59499

def red_balls : ℕ := 10
def green_balls : ℕ := 8
def total_balls : ℕ := red_balls + green_balls
def drawn_balls : ℕ := 4

def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def prob_two_red_two_green : ℚ :=
  (combination red_balls 2 * combination green_balls 2 : ℚ) / combination total_balls drawn_balls

theorem probability_of_two_red_two_green :
  prob_two_red_two_green = 7 / 17 := 
sorry

end NUMINAMATH_GPT_probability_of_two_red_two_green_l594_59499


namespace NUMINAMATH_GPT_sam_paid_amount_l594_59449

theorem sam_paid_amount (F : ℝ) (Joe Peter Sam : ℝ) 
  (h1 : Joe = (1/4)*F + 7) 
  (h2 : Peter = (1/3)*F - 7) 
  (h3 : Sam = (1/2)*F - 12)
  (h4 : Joe + Peter + Sam = F) : 
  Sam = 60 := 
by 
  sorry

end NUMINAMATH_GPT_sam_paid_amount_l594_59449


namespace NUMINAMATH_GPT_sum_of_three_consecutive_even_integers_l594_59421

theorem sum_of_three_consecutive_even_integers : 
  ∃ (n : ℤ), n * (n + 2) * (n + 4) = 480 → n + (n + 2) + (n + 4) = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_even_integers_l594_59421


namespace NUMINAMATH_GPT_total_theme_parks_l594_59442

-- Define the constants based on the problem's conditions
def Jamestown := 20
def Venice := Jamestown + 25
def MarinaDelRay := Jamestown + 50

-- Theorem statement: Total number of theme parks in all three towns is 135
theorem total_theme_parks : Jamestown + Venice + MarinaDelRay = 135 := by
  sorry

end NUMINAMATH_GPT_total_theme_parks_l594_59442


namespace NUMINAMATH_GPT_exists_equidistant_point_l594_59496

-- Define three points A, B, and C in 2D space
variables {A B C P: ℝ × ℝ}

-- Assume the points A, B, and C are not collinear
def not_collinear (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.2 - A.2) ≠ (C.1 - A.1) * (B.2 - A.2)

-- Define the concept of a point being equidistant from three given points
def equidistant (P A B C : ℝ × ℝ) : Prop :=
  dist P A = dist P B ∧ dist P B = dist P C

-- Define the intersection of the perpendicular bisectors of the sides of the triangle formed by A, B, and C
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- placeholder for the actual construction

-- The main theorem statement: If A, B, and C are not collinear, then there exists a unique point P that is equidistant from A, B, and C
theorem exists_equidistant_point (h: not_collinear A B C) :
  ∃! P, equidistant P A B C := 
sorry

end NUMINAMATH_GPT_exists_equidistant_point_l594_59496


namespace NUMINAMATH_GPT_velocity_is_zero_at_t_equals_2_l594_59486

def displacement (t : ℝ) : ℝ := -2 * t^2 + 8 * t

theorem velocity_is_zero_at_t_equals_2 : (deriv displacement 2 = 0) :=
by
  -- The definition step from (a). 
  let v := deriv displacement
  -- This would skip the proof itself, as instructed.
  sorry

end NUMINAMATH_GPT_velocity_is_zero_at_t_equals_2_l594_59486


namespace NUMINAMATH_GPT_range_of_a_l594_59411

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + (a^2 + 1) * x + a - 2

theorem range_of_a (a : ℝ) :
  (f a 1 < 0) ∧ (f a (-1) < 0) → -1 < a ∧ a < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l594_59411


namespace NUMINAMATH_GPT_abs_sum_eq_two_l594_59409

theorem abs_sum_eq_two (a b c : ℤ) (h : (a - b) ^ 10 + (a - c) ^ 10 = 1) : 
  abs (a - b) + abs (b - c) + abs (c - a) = 2 := 
sorry

end NUMINAMATH_GPT_abs_sum_eq_two_l594_59409


namespace NUMINAMATH_GPT_minValue_expression_l594_59431

noncomputable def minValue (x y : ℝ) : ℝ :=
  4 / x^2 + 4 / (x * y) + 1 / y^2

theorem minValue_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : (x - 2 * y)^2 = (x * y)^3) :
  minValue x y = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minValue_expression_l594_59431


namespace NUMINAMATH_GPT_smallest_x_remainder_l594_59473

theorem smallest_x_remainder : ∃ x : ℕ, x > 0 ∧ 
    x % 6 = 5 ∧
    x % 7 = 6 ∧
    x % 8 = 7 ∧
    x = 167 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_remainder_l594_59473


namespace NUMINAMATH_GPT_trigonometric_identity_l594_59462

open Real

variable (α : ℝ)

theorem trigonometric_identity (h : tan (π - α) = 2) :
  (sin (π / 2 + α) + sin (π - α)) / (cos (3 * π / 2 + α) + 2 * cos (π + α)) = 1 / 4 :=
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l594_59462


namespace NUMINAMATH_GPT_fraction_of_25_exists_l594_59487

theorem fraction_of_25_exists :
  ∃ x : ℚ, 0.60 * 40 = x * 25 + 4 ∧ x = 4 / 5 :=
by
  simp
  sorry

end NUMINAMATH_GPT_fraction_of_25_exists_l594_59487


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l594_59455

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, x^2 - (m - 4) * x - m + 7 > 0) ↔ m ∈ Set.Ioo (-2 : ℝ) 6 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l594_59455


namespace NUMINAMATH_GPT_Nigel_initial_amount_l594_59461

-- Defining the initial amount Olivia has
def Olivia_initial : ℕ := 112

-- Defining the amount left after buying the tickets
def amount_left : ℕ := 83

-- Defining the cost per ticket and the number of tickets bought
def cost_per_ticket : ℕ := 28
def number_of_tickets : ℕ := 6

-- Calculating the total cost of the tickets
def total_cost : ℕ := cost_per_ticket * number_of_tickets

-- Calculating the total amount Olivia spent
def Olivia_spent : ℕ := Olivia_initial - amount_left

-- Defining the total amount they spent
def total_spent : ℕ := total_cost

-- Main theorem to prove that Nigel initially had $139
theorem Nigel_initial_amount : ∃ (n : ℕ), (n + Olivia_initial - Olivia_spent = total_spent) → n = 139 :=
by {
  sorry
}

end NUMINAMATH_GPT_Nigel_initial_amount_l594_59461


namespace NUMINAMATH_GPT_garden_length_l594_59463

theorem garden_length (w l : ℝ) (h1 : l = 2 + 3 * w) (h2 : 2 * l + 2 * w = 100) : l = 38 :=
sorry

end NUMINAMATH_GPT_garden_length_l594_59463


namespace NUMINAMATH_GPT_eval_composition_l594_59439

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 - 2

theorem eval_composition : f (g 2) = -7 := 
by {
  sorry
}

end NUMINAMATH_GPT_eval_composition_l594_59439


namespace NUMINAMATH_GPT_number_of_always_true_inequalities_l594_59480

theorem number_of_always_true_inequalities (a b c d : ℝ) (h1 : a > b) (h2 : c > d) :
  (a + c > b + d) ∧
  (¬(a - c > b - d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 - 3 > -2 - (-2))) ∧
  (¬(a * c > b * d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 * 3 > -2 * (-2))) ∧
  (¬(a / c > b / d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 / 3 > (-2) / (-2))) :=
by
  sorry

end NUMINAMATH_GPT_number_of_always_true_inequalities_l594_59480


namespace NUMINAMATH_GPT_train_to_platform_ratio_l594_59489

-- Define the given conditions as assumptions
def speed_kmh : ℕ := 54 -- speed of the train in km/hr
def train_length_m : ℕ := 450 -- length of the train in meters
def crossing_time_min : ℕ := 1 -- time to cross the platform in minutes

-- Conversion from km/hr to m/min
def speed_mpm : ℕ := (speed_kmh * 1000) / 60

-- Calculate the total distance covered in one minute
def total_distance_m : ℕ := speed_mpm * crossing_time_min

-- Define the length of the platform
def platform_length_m : ℕ := total_distance_m - train_length_m

-- The proof statement to show the ratio of the lengths
theorem train_to_platform_ratio : train_length_m = platform_length_m :=
by 
  -- following from the definition of platform_length_m
  sorry

end NUMINAMATH_GPT_train_to_platform_ratio_l594_59489


namespace NUMINAMATH_GPT_sqrt_two_irrational_l594_59406

theorem sqrt_two_irrational :
  ¬ ∃ (a b : ℕ), (a.gcd b = 1) ∧ (b ≠ 0) ∧ (a^2 = 2 * b^2) :=
sorry

end NUMINAMATH_GPT_sqrt_two_irrational_l594_59406


namespace NUMINAMATH_GPT_max_proj_area_of_regular_tetrahedron_l594_59498

theorem max_proj_area_of_regular_tetrahedron (a : ℝ) (h_a : a > 0) : 
    ∃ max_area : ℝ, max_area = a^2 / 2 :=
by
  existsi (a^2 / 2)
  sorry

end NUMINAMATH_GPT_max_proj_area_of_regular_tetrahedron_l594_59498


namespace NUMINAMATH_GPT_sum_of_c_n_l594_59481

variable {a_n : ℕ → ℕ}    -- Sequence {a_n}
variable {b_n : ℕ → ℕ}    -- Sequence {b_n}
variable {c_n : ℕ → ℕ}    -- Sequence {c_n}
variable {S_n : ℕ → ℕ}    -- Sum of the first n terms of sequence {a_n}
variable {T_n : ℕ → ℕ}    -- Sum of the first n terms of sequence {c_n}

axiom a3 : a_n 3 = 7
axiom S6 : S_n 6 = 48
axiom b_recur : ∀ n : ℕ, 2 * b_n (n + 1) = b_n n + 2
axiom b1 : b_n 1 = 3
axiom c_def : ∀ n : ℕ, c_n n = a_n n * (b_n n - 2)

theorem sum_of_c_n : ∀ n : ℕ, T_n n = 10 - (2*n + 5) * (1 / (2^(n-1))) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_of_c_n_l594_59481


namespace NUMINAMATH_GPT_compass_legs_cannot_swap_l594_59450

-- Define the problem conditions: compass legs on infinite grid, constant distance d.
def on_grid (p q : ℤ × ℤ) : Prop := 
  ∃ d : ℕ, d * d = (p.1 - q.1) * (p.1 - q.1) + (p.2 - q.2) * (p.2 - q.2) ∧ d > 0

-- Define the main theorem as a Lean 4 statement
theorem compass_legs_cannot_swap (p q : ℤ × ℤ) (h : on_grid p q) : 
  ¬ ∃ r s : ℤ × ℤ, on_grid r p ∧ on_grid s p ∧ p ≠ q ∧ r = q ∧ s = p :=
sorry

end NUMINAMATH_GPT_compass_legs_cannot_swap_l594_59450


namespace NUMINAMATH_GPT_cubes_in_fig_6_surface_area_fig_10_l594_59457

-- Define the function to calculate the number of unit cubes in Fig. n
def cubes_in_fig (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

-- Define the function to calculate the surface area of the solid figure for Fig. n
def surface_area_fig (n : ℕ) : ℕ := 6 * n * n

-- Theorem statements
theorem cubes_in_fig_6 : cubes_in_fig 6 = 91 :=
by sorry

theorem surface_area_fig_10 : surface_area_fig 10 = 600 :=
by sorry

end NUMINAMATH_GPT_cubes_in_fig_6_surface_area_fig_10_l594_59457


namespace NUMINAMATH_GPT_max_of_2x_plus_y_l594_59428

theorem max_of_2x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y / 2 + 1 / x + 8 / y = 10) : 
  2 * x + y ≤ 18 :=
sorry

end NUMINAMATH_GPT_max_of_2x_plus_y_l594_59428


namespace NUMINAMATH_GPT_evaluate_expression_l594_59407

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) :
  2 * a^2 - 3 * b^2 + 4 * a * b = -43 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l594_59407


namespace NUMINAMATH_GPT_quadratic_has_negative_root_iff_l594_59440

theorem quadratic_has_negative_root_iff (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 4 * x + 1 = 0) ↔ a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_negative_root_iff_l594_59440


namespace NUMINAMATH_GPT_calculation_of_product_l594_59495

theorem calculation_of_product : (0.09)^3 * 0.0007 = 0.0000005103 := 
by
  sorry

end NUMINAMATH_GPT_calculation_of_product_l594_59495


namespace NUMINAMATH_GPT_minimum_bail_rate_l594_59471

theorem minimum_bail_rate
  (distance : ℝ) (leak_rate : ℝ) (rain_rate : ℝ) (sink_threshold : ℝ) (rowing_speed : ℝ) (time_in_minutes : ℝ) (bail_rate : ℝ) : 
  (distance = 2) → 
  (leak_rate = 15) → 
  (rain_rate = 5) →
  (sink_threshold = 60) →
  (rowing_speed = 3) →
  (time_in_minutes = (2 / 3) * 60) →
  (bail_rate = sink_threshold / (time_in_minutes) - (rain_rate + leak_rate)) →
  bail_rate ≥ 18.5 :=
by
  intros h_distance h_leak_rate h_rain_rate h_sink_threshold h_rowing_speed h_time_in_minutes h_bail_rate
  sorry

end NUMINAMATH_GPT_minimum_bail_rate_l594_59471


namespace NUMINAMATH_GPT_expected_losses_correct_l594_59420

def game_probabilities : List (ℕ × ℝ) := [
  (5, 0.6), (10, 0.75), (15, 0.4), (12, 0.85), (20, 0.5),
  (30, 0.2), (10, 0.9), (25, 0.7), (35, 0.65), (10, 0.8)
]

def expected_losses : ℝ :=
  (1 - 0.6) + (1 - 0.75) + (1 - 0.4) + (1 - 0.85) +
  (1 - 0.5) + (1 - 0.2) + (1 - 0.9) + (1 - 0.7) +
  (1 - 0.65) + (1 - 0.8)

theorem expected_losses_correct :
  expected_losses = 3.55 :=
by {
  -- Skipping the actual proof and inserting a sorry as instructed
  sorry
}

end NUMINAMATH_GPT_expected_losses_correct_l594_59420


namespace NUMINAMATH_GPT_sara_spent_on_hotdog_l594_59400

-- Define variables for the costs
def costSalad : ℝ := 5.1
def totalLunchBill : ℝ := 10.46

-- Define the cost of the hotdog
def costHotdog : ℝ := totalLunchBill - costSalad

-- The theorem we need to prove
theorem sara_spent_on_hotdog : costHotdog = 5.36 := by
  -- Proof would go here (if required)
  sorry

end NUMINAMATH_GPT_sara_spent_on_hotdog_l594_59400


namespace NUMINAMATH_GPT_city_of_archimedes_schools_l594_59483

noncomputable def numberOfSchools : ℕ := 32

theorem city_of_archimedes_schools :
  ∃ n : ℕ, (∀ s : Set ℕ, s = {45, 68, 113} →
  (∀ x ∈ s, x > 1 → 4 * n = x + 1 → (2 * n ≤ x ∧ 2 * n + 1 ≥ x) ))
  ∧ n = numberOfSchools :=
sorry

end NUMINAMATH_GPT_city_of_archimedes_schools_l594_59483


namespace NUMINAMATH_GPT_derek_age_l594_59484

theorem derek_age (aunt_beatrice_age : ℕ) (emily_age : ℕ) (derek_age : ℕ)
  (h1 : aunt_beatrice_age = 54)
  (h2 : emily_age = aunt_beatrice_age / 2)
  (h3 : derek_age = emily_age - 7) : derek_age = 20 :=
by
  sorry

end NUMINAMATH_GPT_derek_age_l594_59484


namespace NUMINAMATH_GPT_columbian_coffee_price_is_correct_l594_59426

-- Definitions based on the conditions
def total_mix_weight : ℝ := 100
def brazilian_coffee_price_per_pound : ℝ := 3.75
def final_mix_price_per_pound : ℝ := 6.35
def columbian_coffee_weight : ℝ := 52

-- Let C be the price per pound of the Columbian coffee
noncomputable def columbian_coffee_price_per_pound : ℝ := sorry

-- Define the Lean 4 proof problem
theorem columbian_coffee_price_is_correct :
  columbian_coffee_price_per_pound = 8.75 :=
by
  -- Total weight and calculation based on conditions
  let brazilian_coffee_weight := total_mix_weight - columbian_coffee_weight
  let total_value_of_columbian := columbian_coffee_weight * columbian_coffee_price_per_pound
  let total_value_of_brazilian := brazilian_coffee_weight * brazilian_coffee_price_per_pound
  let total_value_of_mix := total_mix_weight * final_mix_price_per_pound
  
  -- Main equation based on the mix
  have main_eq : total_value_of_columbian + total_value_of_brazilian = total_value_of_mix :=
    by sorry

  -- Solve for C (columbian coffee price per pound)
  sorry

end NUMINAMATH_GPT_columbian_coffee_price_is_correct_l594_59426


namespace NUMINAMATH_GPT_unique_quotient_is_9742_l594_59452

theorem unique_quotient_is_9742 :
  ∃ (d4 d3 d2 d1 : ℕ),
    (d2 = d1 + 2) ∧
    (d4 = d3 + 2) ∧
    (0 ≤ d1 ∧ d1 ≤ 9) ∧
    (0 ≤ d2 ∧ d2 ≤ 9) ∧
    (0 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d4 * 1000 + d3 * 100 + d2 * 10 + d1 = 9742) :=
by sorry

end NUMINAMATH_GPT_unique_quotient_is_9742_l594_59452


namespace NUMINAMATH_GPT_equal_amounts_hot_and_cold_water_l594_59436

theorem equal_amounts_hot_and_cold_water (time_to_fill_cold : ℕ) (time_to_fill_hot : ℕ) (t_c : ℤ) : 
  time_to_fill_cold = 19 → 
  time_to_fill_hot = 23 → 
  t_c = 2 :=
by
  intros h_c h_h
  sorry

end NUMINAMATH_GPT_equal_amounts_hot_and_cold_water_l594_59436


namespace NUMINAMATH_GPT_max_wrestlers_more_than_131_l594_59474

theorem max_wrestlers_more_than_131
  (n : ℤ)
  (total_wrestlers : ℤ := 20)
  (average_weight : ℕ := 125)
  (min_weight : ℕ := 90)
  (constraint1 : n ≥ 0)
  (constraint2 : n ≤ total_wrestlers)
  (total_weight := 2500) :
  n ≤ 17 :=
by
  sorry

end NUMINAMATH_GPT_max_wrestlers_more_than_131_l594_59474


namespace NUMINAMATH_GPT_find_n_l594_59447

noncomputable def b_0 : ℝ := Real.cos (Real.pi / 18) ^ 2

noncomputable def b_n (n : ℕ) : ℝ :=
if n = 0 then b_0 else 4 * (b_n (n - 1)) * (1 - (b_n (n - 1)))

theorem find_n : ∀ n : ℕ, b_n n = b_0 → n = 24 := 
sorry

end NUMINAMATH_GPT_find_n_l594_59447


namespace NUMINAMATH_GPT_find_fraction_l594_59412

-- Define the given variables and conditions
variables (x y : ℝ)
-- Assume x and y are nonzero
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
-- Assume the given condition
variable (h : (4*x + 2*y) / (2*x - 8*y) = 3)

-- Define the theorem to be proven
theorem find_fraction (h : (4*x + 2*y) / (2*x - 8*y) = 3) : (x + 4 * y) / (4 * x - y) = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_find_fraction_l594_59412


namespace NUMINAMATH_GPT_mean_of_remaining_three_numbers_l594_59404

variable {a b c : ℝ}

theorem mean_of_remaining_three_numbers (h1 : (a + b + c + 103) / 4 = 90) : (a + b + c) / 3 = 85.7 :=
by
  -- Sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_mean_of_remaining_three_numbers_l594_59404


namespace NUMINAMATH_GPT_fractionD_is_unchanged_l594_59482

-- Define variables x and y
variable (x y : ℚ)

-- Define the fractions
def fractionA := x / (y + 1)
def fractionB := (x + y) / (x + 1)
def fractionC := (x * y) / (x + y)
def fractionD := (2 * x) / (3 * x - y)

-- Define the transformation
def transform (a b : ℚ) : ℚ × ℚ := (3 * a, 3 * b)

-- Define the new fractions after transformation
def newFractionA := (3 * x) / (3 * y + 1)
def newFractionB := (3 * x + 3 * y) / (3 * x + 1)
def newFractionC := (9 * x * y) / (3 * x + 3 * y)
def newFractionD := (6 * x) / (9 * x - 3 * y)

-- The proof problem statement
theorem fractionD_is_unchanged :
  fractionD x y = newFractionD x y ∧
  fractionA x y ≠ newFractionA x y ∧
  fractionB x y ≠ newFractionB x y ∧
  fractionC x y ≠ newFractionC x y := sorry

end NUMINAMATH_GPT_fractionD_is_unchanged_l594_59482


namespace NUMINAMATH_GPT_notebooks_if_students_halved_l594_59460

-- Definitions based on the problem conditions
def totalNotebooks: ℕ := 512
def notebooksPerStudent (students: ℕ) : ℕ := students / 8
def notebooksWhenStudentsHalved (students notebooks: ℕ) : ℕ := notebooks / (students / 2)

-- Theorem statement
theorem notebooks_if_students_halved (S : ℕ) (h : S * (S / 8) = totalNotebooks) :
    notebooksWhenStudentsHalved S totalNotebooks = 16 :=
by
  sorry

end NUMINAMATH_GPT_notebooks_if_students_halved_l594_59460
