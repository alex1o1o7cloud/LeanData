import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l1093_109362

theorem range_of_a (f : ℝ → ℝ) (a : ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_mono_inc : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_ineq : f (a - 3) < f 4) : -1 < a ∧ a < 7 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1093_109362


namespace NUMINAMATH_GPT_billy_has_62_crayons_l1093_109388

noncomputable def billy_crayons (total_crayons : ℝ) (jane_crayons : ℝ) : ℝ :=
  total_crayons - jane_crayons

theorem billy_has_62_crayons : billy_crayons 114 52.0 = 62 := by
  sorry

end NUMINAMATH_GPT_billy_has_62_crayons_l1093_109388


namespace NUMINAMATH_GPT_inverse_value_l1093_109346

def f (x : ℤ) : ℤ := 5 * x ^ 3 - 3

theorem inverse_value : ∀ y, (f y) = 4 → y = 317 :=
by
  intros
  sorry

end NUMINAMATH_GPT_inverse_value_l1093_109346


namespace NUMINAMATH_GPT_f_one_minus_a_l1093_109394

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 2) = f x
axiom f_one_plus_a {a : ℝ} : f (1 + a) = 1

theorem f_one_minus_a (a : ℝ) : f (1 - a) = -1 :=
by
  sorry

end NUMINAMATH_GPT_f_one_minus_a_l1093_109394


namespace NUMINAMATH_GPT_euclid1976_partb_problem2_l1093_109364

theorem euclid1976_partb_problem2
  (x y : ℝ)
  (geo_prog : y^2 = 2 * x)
  (arith_prog : 2 / y = 1 / x + 9 / x^2) :
  x * y = 27 / 2 := by 
  sorry

end NUMINAMATH_GPT_euclid1976_partb_problem2_l1093_109364


namespace NUMINAMATH_GPT_ratio_proof_l1093_109326

theorem ratio_proof (a b c : ℝ) (ha : b / a = 3) (hb : c / b = 4) :
    (a + 2 * b) / (b + 2 * c) = 7 / 27 := by
  sorry

end NUMINAMATH_GPT_ratio_proof_l1093_109326


namespace NUMINAMATH_GPT_hours_per_day_initial_l1093_109317

-- Definition of the problem and conditions
def initial_men : ℕ := 75
def depth1 : ℕ := 50
def additional_men : ℕ := 65
def total_men : ℕ := initial_men + additional_men
def depth2 : ℕ := 70
def hours_per_day2 : ℕ := 6
def work1 (H : ℝ) := initial_men * H * depth1
def work2 := total_men * hours_per_day2 * depth2

-- Statement to prove
theorem hours_per_day_initial (H : ℝ) (h1 : work1 H = work2) : H = 15.68 :=
by
  sorry

end NUMINAMATH_GPT_hours_per_day_initial_l1093_109317


namespace NUMINAMATH_GPT_e_is_dq_sequence_l1093_109322

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d a₀, ∀ n, a n = a₀ + n * d

def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∃ q b₀, q > 0 ∧ ∀ n, b n = b₀ * q^n

def is_dq_sequence (c : ℕ → ℕ) : Prop :=
  ∃ a b, is_arithmetic_sequence a ∧ is_geometric_sequence b ∧ ∀ n, c n = a n + b n

def e (n : ℕ) : ℕ :=
  n + 2^n

theorem e_is_dq_sequence : is_dq_sequence e :=
  sorry

end NUMINAMATH_GPT_e_is_dq_sequence_l1093_109322


namespace NUMINAMATH_GPT_negation_equiv_l1093_109331

theorem negation_equiv (p : Prop) : 
  (p = (∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0)) → 
  (¬ p = (∀ x : ℝ, x^2 + 2*x + 2 > 0)) :=
by
  sorry

end NUMINAMATH_GPT_negation_equiv_l1093_109331


namespace NUMINAMATH_GPT_joann_third_day_lollipops_l1093_109321

theorem joann_third_day_lollipops
  (a b c d e : ℕ)
  (h1 : b = a + 6)
  (h2 : c = b + 6)
  (h3 : d = c + 6)
  (h4 : e = d + 6)
  (h5 : a + b + c + d + e = 100) :
  c = 20 :=
by
  sorry

end NUMINAMATH_GPT_joann_third_day_lollipops_l1093_109321


namespace NUMINAMATH_GPT_gcd_1020_multiple_38962_l1093_109319

-- Define that x is a multiple of 38962
def multiple_of (x n : ℤ) : Prop := ∃ k : ℤ, x = k * n

-- The main theorem statement
theorem gcd_1020_multiple_38962 (x : ℤ) (h : multiple_of x 38962) : Int.gcd 1020 x = 6 := 
sorry

end NUMINAMATH_GPT_gcd_1020_multiple_38962_l1093_109319


namespace NUMINAMATH_GPT_cube_negative_iff_l1093_109333

theorem cube_negative_iff (x : ℝ) : x < 0 ↔ x^3 < 0 :=
sorry

end NUMINAMATH_GPT_cube_negative_iff_l1093_109333


namespace NUMINAMATH_GPT_quadratic_root_shift_l1093_109302

theorem quadratic_root_shift (A B p : ℤ) (α β : ℤ) 
  (h1 : ∀ x, x^2 + p * x + 19 = 0 → x = α + 1 ∨ x = β + 1)
  (h2 : ∀ x, x^2 - A * x + B = 0 → x = α ∨ x = β)
  (h3 : α + β = A)
  (h4 : α * β = B) :
  A + B = 18 := 
sorry

end NUMINAMATH_GPT_quadratic_root_shift_l1093_109302


namespace NUMINAMATH_GPT_cube_root_solutions_l1093_109365

theorem cube_root_solutions (p : ℕ) (hp : p > 3) :
    (∃ (k : ℤ) (h1 : k^2 ≡ -3 [ZMOD p]), ∀ x, x^3 ≡ 1 [ZMOD p] → 
        (x = 1 ∨ (x^2 + x + 1 ≡ 0 [ZMOD p])) )
    ∨ 
    (∀ x, x^3 ≡ 1 [ZMOD p] → x = 1) := 
sorry

end NUMINAMATH_GPT_cube_root_solutions_l1093_109365


namespace NUMINAMATH_GPT_clowns_per_mobile_28_l1093_109389

def clowns_in_each_mobile (total_clowns num_mobiles : Nat) (h : total_clowns = 140 ∧ num_mobiles = 5) : Nat :=
  total_clowns / num_mobiles

theorem clowns_per_mobile_28 (total_clowns num_mobiles : Nat) (h : total_clowns = 140 ∧ num_mobiles = 5) :
  clowns_in_each_mobile total_clowns num_mobiles h = 28 :=
by
  sorry

end NUMINAMATH_GPT_clowns_per_mobile_28_l1093_109389


namespace NUMINAMATH_GPT_harry_pencils_remaining_l1093_109345

def num_pencils_anna : ℕ := 50
def num_pencils_harry_initial := 2 * num_pencils_anna
def num_pencils_lost_harry := 19

def pencils_left_harry (pencils_anna : ℕ) (pencils_harry_initial : ℕ) (pencils_lost : ℕ) : ℕ :=
  pencils_harry_initial - pencils_lost

theorem harry_pencils_remaining : pencils_left_harry num_pencils_anna num_pencils_harry_initial num_pencils_lost_harry = 81 :=
by
  sorry

end NUMINAMATH_GPT_harry_pencils_remaining_l1093_109345


namespace NUMINAMATH_GPT_shaded_region_perimeter_l1093_109392

theorem shaded_region_perimeter (C : Real) (r : Real) (L : Real) (P : Real)
  (h0 : C = 48)
  (h1 : r = C / (2 * Real.pi))
  (h2 : L = (90 / 360) * C)
  (h3 : P = 3 * L) :
  P = 36 := by
  sorry

end NUMINAMATH_GPT_shaded_region_perimeter_l1093_109392


namespace NUMINAMATH_GPT_complex_div_l1093_109359

theorem complex_div (i : ℂ) (hi : i^2 = -1) : (1 + i) / i = 1 - i := by
  sorry

end NUMINAMATH_GPT_complex_div_l1093_109359


namespace NUMINAMATH_GPT_units_digit_of_8_pow_47_l1093_109337

theorem units_digit_of_8_pow_47 : (8 ^ 47) % 10 = 2 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_8_pow_47_l1093_109337


namespace NUMINAMATH_GPT_eagles_min_additional_wins_l1093_109312

theorem eagles_min_additional_wins {N : ℕ} (eagles_initial_wins falcons_initial_wins : ℕ) (initial_games : ℕ)
  (total_games_won_fraction : ℚ) (required_fraction : ℚ) :
  eagles_initial_wins = 3 →
  falcons_initial_wins = 4 →
  initial_games = eagles_initial_wins + falcons_initial_wins →
  total_games_won_fraction = (3 + N) / (7 + N) →
  required_fraction = 9 / 10 →
  total_games_won_fraction = required_fraction →
  N = 33 :=
by
  sorry

end NUMINAMATH_GPT_eagles_min_additional_wins_l1093_109312


namespace NUMINAMATH_GPT_find_fake_coin_l1093_109363

def coin_value (n : Nat) : Nat :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | _ => 0

def coin_weight (n : Nat) : Nat :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | _ => 0

def is_fake (weight : Nat) : Prop :=
  weight ≠ coin_weight 1 ∧ weight ≠ coin_weight 2 ∧ weight ≠ coin_weight 3 ∧ weight ≠ coin_weight 4

theorem find_fake_coin :
  ∃ (n : Nat) (w : Nat), (is_fake w) → ∃! (m : Nat), m ≠ w ∧ (m = coin_weight 1 ∨ m = coin_weight 2 ∨ m = coin_weight 3 ∨ m = coin_weight 4) := 
sorry

end NUMINAMATH_GPT_find_fake_coin_l1093_109363


namespace NUMINAMATH_GPT_part1_part2_l1093_109325

-- Define the conditions for p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 6) <= 0
def q (x m : ℝ) : Prop := (2 - m <= x) ∧ (x <= 2 + m)

-- Proof statement for part (1)
theorem part1 (m: ℝ) : 
  (∀ x : ℝ, p x → q x m) → 4 <= m :=
sorry

-- Proof statement for part (2)
theorem part2 (x : ℝ) (m : ℝ) : 
  (m = 5) → (p x ∨ q x m) ∧ ¬(p x ∧ q x m) → x ∈ Set.Ico (-3) (-2) ∪ Set.Ioc 6 7 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1093_109325


namespace NUMINAMATH_GPT_final_customer_boxes_l1093_109320

theorem final_customer_boxes (f1 f2 f3 f4 goal left boxes_first : ℕ) 
  (h1 : boxes_first = 5) 
  (h2 : f2 = 4 * boxes_first) 
  (h3 : f3 = f2 / 2) 
  (h4 : f4 = 3 * f3)
  (h5 : goal = 150) 
  (h6 : left = 75) 
  (h7 : goal - left = f1 + f2 + f3 + f4) : 
  (goal - left - (f1 + f2 + f3 + f4) = 10) := 
sorry

end NUMINAMATH_GPT_final_customer_boxes_l1093_109320


namespace NUMINAMATH_GPT_valid_division_l1093_109367

theorem valid_division (A B C E F G H K : ℕ) (hA : A = 7) (hB : B = 1) (hC : C = 2)
    (hE : E = 6) (hF : F = 8) (hG : G = 5) (hH : H = 4) (hK : K = 9) :
    (A * 10 + B) / ((C * 100 + A * 10 + B) / 100 + E + B * F * D) = 71 / 271 :=
by {
  sorry
}

end NUMINAMATH_GPT_valid_division_l1093_109367


namespace NUMINAMATH_GPT_jello_mix_needed_per_pound_l1093_109398

variable (bathtub_volume : ℝ) (gallons_per_cubic_foot : ℝ) 
          (pounds_per_gallon : ℝ) (cost_per_tablespoon : ℝ) 
          (total_cost : ℝ)

theorem jello_mix_needed_per_pound :
  bathtub_volume = 6 ∧
  gallons_per_cubic_foot = 7.5 ∧
  pounds_per_gallon = 8 ∧
  cost_per_tablespoon = 0.50 ∧
  total_cost = 270 →
  (total_cost / cost_per_tablespoon) / 
  (bathtub_volume * gallons_per_cubic_foot * pounds_per_gallon) = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_jello_mix_needed_per_pound_l1093_109398


namespace NUMINAMATH_GPT_online_store_commission_l1093_109366

theorem online_store_commission (cost : ℝ) (desired_profit_pct : ℝ) (online_price : ℝ) (commission_pct : ℝ) :
  cost = 19 →
  desired_profit_pct = 0.20 →
  online_price = 28.5 →
  commission_pct = 25 :=
by
  sorry

end NUMINAMATH_GPT_online_store_commission_l1093_109366


namespace NUMINAMATH_GPT_sum_of_squares_eq_expansion_l1093_109330

theorem sum_of_squares_eq_expansion (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_eq_expansion_l1093_109330


namespace NUMINAMATH_GPT_last_three_digits_of_5_pow_9000_l1093_109358

theorem last_three_digits_of_5_pow_9000 (h : 5^300 ≡ 1 [MOD 800]) : 5^9000 ≡ 1 [MOD 800] :=
by
  -- The proof is omitted here according to the instruction
  sorry

end NUMINAMATH_GPT_last_three_digits_of_5_pow_9000_l1093_109358


namespace NUMINAMATH_GPT_smallest_x_for_three_digit_product_l1093_109378

theorem smallest_x_for_three_digit_product : ∃ x : ℕ, (27 * x >= 100) ∧ (∀ y < x, 27 * y < 100) :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_for_three_digit_product_l1093_109378


namespace NUMINAMATH_GPT_rectangle_is_possible_l1093_109354

def possibleToFormRectangle (stick_lengths : List ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ (a + b) * 2 = List.sum stick_lengths

noncomputable def sticks : List ℕ := List.range' 1 99

theorem rectangle_is_possible : possibleToFormRectangle sticks :=
sorry

end NUMINAMATH_GPT_rectangle_is_possible_l1093_109354


namespace NUMINAMATH_GPT_root_sum_greater_than_one_l1093_109393

noncomputable def f (x a : ℝ) : ℝ := (x * Real.log x) / (x - 1) - a

noncomputable def h (x a : ℝ) : ℝ := (x^2 - x) * f x a

theorem root_sum_greater_than_one {a m x1 x2 : ℝ} (ha : a < 0)
  (h_eq_m : ∀ x, h x a = m) (hx1_root : h x1 a = m) (hx2_root : h x2 a = m)
  (hx1x2_distinct : x1 ≠ x2) :
  x1 + x2 > 1 := 
sorry

end NUMINAMATH_GPT_root_sum_greater_than_one_l1093_109393


namespace NUMINAMATH_GPT_difference_of_squares_l1093_109336

theorem difference_of_squares (a b c : ℤ) (h₁ : a < b) (h₂ : b < c) (h₃ : a % 2 = 0) (h₄ : b % 2 = 0) (h₅ : c % 2 = 0) (h₆ : a + b + c = 1992) :
  c^2 - a^2 = 5312 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1093_109336


namespace NUMINAMATH_GPT_min_volume_for_cone_l1093_109343

noncomputable def min_cone_volume (V1 : ℝ) : Prop :=
  ∀ V2 : ℝ, (V1 = 1) → 
    V2 ≥ (4 / 3)

-- The statement without proof
theorem min_volume_for_cone : 
  min_cone_volume 1 :=
sorry

end NUMINAMATH_GPT_min_volume_for_cone_l1093_109343


namespace NUMINAMATH_GPT_fraction_always_defined_l1093_109329

theorem fraction_always_defined (y : ℝ) : (y^2 + 1) ≠ 0 := 
by
  -- proof is not required
  sorry

end NUMINAMATH_GPT_fraction_always_defined_l1093_109329


namespace NUMINAMATH_GPT_eight_div_repeat_three_l1093_109383

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end NUMINAMATH_GPT_eight_div_repeat_three_l1093_109383


namespace NUMINAMATH_GPT_total_students_l1093_109340

-- Condition 1: 20% of students are below 8 years of age.
-- Condition 2: The number of students of 8 years of age is 72.
-- Condition 3: The number of students above 8 years of age is 2/3 of the number of students of 8 years of age.

variable {T : ℝ} -- Total number of students

axiom cond1 : 0.20 * T = (T - (72 + (2 / 3) * 72))
axiom cond2 : 72 = 72
axiom cond3 : (T - 72 - (2 / 3) * 72) = 0

theorem total_students : T = 150 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_students_l1093_109340


namespace NUMINAMATH_GPT_part1_part2_l1093_109390

def U : Set ℝ := {x : ℝ | True}

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part 1: Prove the range of m when 4 ∈ B(m) is [5/2, 3]
theorem part1 (m : ℝ) : (4 ∈ B m) → (5/2 ≤ m ∧ m ≤ 3) := by
  sorry

-- Part 2: Prove the range of m when x ∈ A is a necessary but not sufficient condition for x ∈ B(m) 
theorem part2 (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ∧ ¬(∀ x, x ∈ A → x ∈ B m) → (m ≤ 3) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1093_109390


namespace NUMINAMATH_GPT_triangle_perimeter_l1093_109379

theorem triangle_perimeter : 
  ∀ (f : ℝ → ℝ), 
  (∀ x, f x = 4 * (1 - x / 3)) →
  ∃ (A B C : ℝ × ℝ), 
  A = (3, 0) ∧ 
  B = (0, 4) ∧ 
  C = (0, 0) ∧ 
  dist A B + dist B C + dist C A = 12 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1093_109379


namespace NUMINAMATH_GPT_part_a_part_b_l1093_109375

theorem part_a (x y : ℂ) : (3 * y + 5 * x * Complex.I = 15 - 7 * Complex.I) ↔ (x = -7/5 ∧ y = 5) := by
  sorry

theorem part_b (x y : ℝ) : (2 * x + 3 * y + (x - y) * Complex.I = 7 + 6 * Complex.I) ↔ (x = 5 ∧ y = -1) := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1093_109375


namespace NUMINAMATH_GPT_triangle_area_ratio_l1093_109303

theorem triangle_area_ratio :
  let base_jihye := 3
  let height_jihye := 2
  let base_donggeon := 3
  let height_donggeon := 6.02
  let area_jihye := (base_jihye * height_jihye) / 2
  let area_donggeon := (base_donggeon * height_donggeon) / 2
  (area_donggeon / area_jihye) = 3.01 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_ratio_l1093_109303


namespace NUMINAMATH_GPT_find_Pete_original_number_l1093_109350

noncomputable def PeteOriginalNumber (x : ℝ) : Prop :=
  5 * (3 * x + 15) = 200

theorem find_Pete_original_number : ∃ x : ℝ, PeteOriginalNumber x ∧ x = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_Pete_original_number_l1093_109350


namespace NUMINAMATH_GPT_find_n_l1093_109338

theorem find_n (n : ℕ) (h1 : ∃ k : ℕ, 12 - n = k * k) : n = 11 := 
by sorry

end NUMINAMATH_GPT_find_n_l1093_109338


namespace NUMINAMATH_GPT_billy_sisters_count_l1093_109385

theorem billy_sisters_count 
  (S B : ℕ) -- S is the number of sisters, B is the number of brothers
  (h1 : B = 2 * S) -- Billy has twice as many brothers as sisters
  (h2 : 2 * (B + S) = 12) -- Billy gives 2 sodas to each sibling to give out the 12 pack
  : S = 2 := 
  by sorry

end NUMINAMATH_GPT_billy_sisters_count_l1093_109385


namespace NUMINAMATH_GPT_solution_set_of_fraction_inequality_l1093_109328

theorem solution_set_of_fraction_inequality (a b x : ℝ) (h1: ∀ x, ax - b > 0 ↔ x ∈ Set.Iio 1) (h2: a < 0) (h3: a - b = 0) :
  ∀ x, (a * x + b) / (x - 2) > 0 ↔ x ∈ Set.Ioo (-1 : ℝ) 2 := 
sorry

end NUMINAMATH_GPT_solution_set_of_fraction_inequality_l1093_109328


namespace NUMINAMATH_GPT_math_problem_l1093_109395

noncomputable def problem_statement (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hxyz : x * y * z = 1) : Prop :=
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + x) * (1 + z)) + z^3 / ((1 + x) * (1 + y))) ≥ 3 / 4

theorem math_problem (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hxyz : x * y * z = 1) :
  problem_statement x y z hx hy hz hxyz :=
sorry

end NUMINAMATH_GPT_math_problem_l1093_109395


namespace NUMINAMATH_GPT_ceil_minus_floor_eq_one_imp_ceil_minus_x_l1093_109347

variable {x : ℝ}

theorem ceil_minus_floor_eq_one_imp_ceil_minus_x (H : ⌈x⌉ - ⌊x⌋ = 1) : ∃ (n : ℤ) (f : ℝ), (x = n + f) ∧ (0 < f) ∧ (f < 1) ∧ (⌈x⌉ - x = 1 - f) := sorry

end NUMINAMATH_GPT_ceil_minus_floor_eq_one_imp_ceil_minus_x_l1093_109347


namespace NUMINAMATH_GPT_greatest_n_and_k_l1093_109372

-- (condition): k is a positive integer
def isPositive (k : Nat) : Prop :=
  k > 0

-- (condition): k < n
def lessThan (k n : Nat) : Prop :=
  k < n

/-- Let m = 3^n and k be a positive integer such that k < n.
     Determine the greatest value of n for which 3^n divides 25!,
     and the greatest value of k such that 3^k divides (25! - 3^n). -/
theorem greatest_n_and_k :
  ∃ (n k : Nat), (3^n ∣ Nat.factorial 25) ∧ (isPositive k) ∧ (lessThan k n) ∧ (3^k ∣ (Nat.factorial 25 - 3^n)) ∧ n = 10 ∧ k = 9 := by
    sorry

end NUMINAMATH_GPT_greatest_n_and_k_l1093_109372


namespace NUMINAMATH_GPT_number_B_expression_l1093_109301

theorem number_B_expression (A B : ℝ) (h : A = B - (4/5) * B) : B = (A + B) / (4 / 5) :=
sorry

end NUMINAMATH_GPT_number_B_expression_l1093_109301


namespace NUMINAMATH_GPT_number_of_routes_600_l1093_109373

-- Define the problem conditions
def number_of_routes (total_cities : Nat) (pick_cities : Nat) (selected_cities : List Nat) : Nat := sorry

-- The number of ways to pick and order 3 cities from remaining 5
def num_ways_pick_three (total_cities : Nat) (pick_cities : Nat) : Nat :=
  Nat.factorial total_cities / Nat.factorial (total_cities - pick_cities)

-- The number of ways to choose positions for M and N
def num_ways_positions (total_positions : Nat) (pick_positions : Nat) : Nat :=
  Nat.choose total_positions pick_positions

-- The main theorem to prove
theorem number_of_routes_600 :
  number_of_routes 7 5 [M, N] = num_ways_pick_three 5 3 * num_ways_positions 4 2 :=
  by sorry

end NUMINAMATH_GPT_number_of_routes_600_l1093_109373


namespace NUMINAMATH_GPT_hexagon_angle_D_135_l1093_109371

theorem hexagon_angle_D_135 
  (A B C D E F : ℝ)
  (h1 : A = B ∧ B = C)
  (h2 : D = E ∧ E = F)
  (h3 : A = D - 30)
  (h4 : A + B + C + D + E + F = 720) :
  D = 135 :=
by {
  sorry
}

end NUMINAMATH_GPT_hexagon_angle_D_135_l1093_109371


namespace NUMINAMATH_GPT_joan_total_spent_on_clothing_l1093_109306

theorem joan_total_spent_on_clothing :
  let shorts_cost := 15.00
  let jacket_cost := 14.82
  let shirt_cost := 12.51
  let shoes_cost := 21.67
  let hat_cost := 8.75
  let belt_cost := 6.34
  shorts_cost + jacket_cost + shirt_cost + shoes_cost + hat_cost + belt_cost = 79.09 :=
by
  sorry

end NUMINAMATH_GPT_joan_total_spent_on_clothing_l1093_109306


namespace NUMINAMATH_GPT_product_of_distances_is_one_l1093_109357

theorem product_of_distances_is_one (k : ℝ) (x1 x2 : ℝ)
  (h1 : x1^2 - k*x1 - 1 = 0)
  (h2 : x2^2 - k*x2 - 1 = 0)
  (h3 : x1 ≠ x2) :
  (|x1| * |x2| = 1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_product_of_distances_is_one_l1093_109357


namespace NUMINAMATH_GPT_order_of_arrival_l1093_109305

noncomputable def position_order (P S O E R : ℕ) : Prop :=
  S = O - 10 ∧ S = R + 25 ∧ R = E - 5 ∧ E = P - 25

theorem order_of_arrival (P S O E R : ℕ) (h : position_order P S O E R) :
  P > (S + 10) ∧ S > (O - 10) ∧ O > (E + 5) ∧ E > R :=
sorry

end NUMINAMATH_GPT_order_of_arrival_l1093_109305


namespace NUMINAMATH_GPT_company_workers_count_l1093_109315

-- Definitions
def num_supervisors := 13
def team_leads_per_supervisor := 3
def workers_per_team_lead := 10

-- Hypothesis
def team_leads := num_supervisors * team_leads_per_supervisor
def workers := team_leads * workers_per_team_lead

-- Theorem to prove
theorem company_workers_count : workers = 390 :=
by
  sorry

end NUMINAMATH_GPT_company_workers_count_l1093_109315


namespace NUMINAMATH_GPT_simplify_fraction_l1093_109382

variable {a b c k : ℝ}
variable (h : a * b = c * k ∧ a * b ≠ 0)

theorem simplify_fraction (h : a * b = c * k ∧ a * b ≠ 0) : 
  (a - b - c + k) / (a + b + c + k) = (a - c) / (a + c) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1093_109382


namespace NUMINAMATH_GPT_jose_profit_share_l1093_109369

theorem jose_profit_share :
  ∀ (Tom_investment Jose_investment total_profit month_investment_tom month_investment_jose total_month_investment: ℝ),
    Tom_investment = 30000 →
    ∃ (months_tom months_jose : ℝ), months_tom = 12 ∧ months_jose = 10 →
      Jose_investment = 45000 →
      total_profit = 72000 →
      month_investment_tom = Tom_investment * months_tom →
      month_investment_jose = Jose_investment * months_jose →
      total_month_investment = month_investment_tom + month_investment_jose →
      (Jose_investment * months_jose / total_month_investment) * total_profit = 40000 :=
by
  sorry

end NUMINAMATH_GPT_jose_profit_share_l1093_109369


namespace NUMINAMATH_GPT_John_gave_the_store_20_dollars_l1093_109339

def slurpee_cost : ℕ := 2
def change_received : ℕ := 8
def slurpees_bought : ℕ := 6
def total_money_given : ℕ := slurpee_cost * slurpees_bought + change_received

theorem John_gave_the_store_20_dollars : total_money_given = 20 := 
by 
  sorry

end NUMINAMATH_GPT_John_gave_the_store_20_dollars_l1093_109339


namespace NUMINAMATH_GPT_intersection_A_B_l1093_109318

open Set

def isInSetA (x : ℕ) : Prop := ∃ n : ℕ, x = 3 * n + 2
def A : Set ℕ := { x | isInSetA x }
def B : Set ℕ := {6, 8, 10, 12, 14}

theorem intersection_A_B :
  A ∩ B = {8, 14} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1093_109318


namespace NUMINAMATH_GPT_solve_system_b_zero_solve_system_b_nonzero_solve_second_system_l1093_109384

section B_zero

variables {x y z b : ℝ}

-- Given conditions for the first system when b = 0
variables (hb_zero : b = 0)
variables (h1 : x + y + z = 0)
variables (h2 : x^2 + y^2 - z^2 = 0)
variables (h3 : 3 * x * y * z - x^3 - y^3 - z^3 = b^3)

theorem solve_system_b_zero :
  ∃ x y z, 3 * x * y * z - x^3 - y^3 - z^3 = b^3 :=
by { sorry }

end B_zero

section B_nonzero

variables {x y z b : ℝ}

-- Given conditions for the first system when b ≠ 0
variables (hb_nonzero : b ≠ 0)
variables (h1 : x + y + z = 2 * b)
variables (h2 : x^2 + y^2 - z^2 = b^2)
variables (h3 : 3 * x * y * z - x^3 - y^3 - z^3 = b^3)

theorem solve_system_b_nonzero :
  ∃ x y z, 3 * x * y * z - x^3 - y^3 - z^3 = b^3 :=
by { sorry }

end B_nonzero

section Second_System

variables {x y z a : ℝ}

-- Given conditions for the second system
variables (h4 : x^2 + y^2 - 2 * z^2 = 2 * a^2)
variables (h5 : x + y + 2 * z = 4 * (a^2 + 1))
variables (h6 : z^2 - x * y = a^2)

theorem solve_second_system :
  ∃ x y z, z^2 - x * y = a^2 :=
by { sorry }

end Second_System

end NUMINAMATH_GPT_solve_system_b_zero_solve_system_b_nonzero_solve_second_system_l1093_109384


namespace NUMINAMATH_GPT_a_must_be_negative_l1093_109370

variable (a b c d e : ℝ)

theorem a_must_be_negative
  (h1 : a / b < -c / d)
  (hb : b > 0)
  (hd : d > 0)
  (he : e > 0)
  (h2 : a + e > 0) : a < 0 := by
  sorry

end NUMINAMATH_GPT_a_must_be_negative_l1093_109370


namespace NUMINAMATH_GPT_find_a2019_l1093_109355

-- Arithmetic sequence
def a (n : ℕ) : ℤ := sorry -- to be defined later

-- Given conditions
def sum_first_five_terms (a: ℕ → ℤ) : Prop := a 1 + a 2 + a 3 + a 4 + a 5 = 15
def term_six (a: ℕ → ℤ) : Prop := a 6 = 6

-- Question (statement to be proved)
def term_2019 (a: ℕ → ℤ) : Prop := a 2019 = 2019

-- Main theorem to be proved
theorem find_a2019 (a: ℕ → ℤ) 
  (h1 : sum_first_five_terms a)
  (h2 : term_six a) : 
  term_2019 a := 
by
  sorry

end NUMINAMATH_GPT_find_a2019_l1093_109355


namespace NUMINAMATH_GPT_ratio_of_circumscribed_areas_l1093_109374

noncomputable def rect_pentagon_circ_ratio (P : ℝ) : ℝ :=
  let s : ℝ := P / 8
  let r_circle : ℝ := (P * Real.sqrt 10) / 16
  let A : ℝ := Real.pi * (r_circle ^ 2)
  let pentagon_side : ℝ := P / 5
  let R_pentagon : ℝ := P / (10 * Real.sin (Real.pi / 5))
  let B : ℝ := Real.pi * (R_pentagon ^ 2)
  A / B

theorem ratio_of_circumscribed_areas (P : ℝ) : rect_pentagon_circ_ratio P = (5 * (5 - Real.sqrt 5)) / 64 :=
by sorry

end NUMINAMATH_GPT_ratio_of_circumscribed_areas_l1093_109374


namespace NUMINAMATH_GPT_proof_problem_l1093_109335

theorem proof_problem (p q : Prop) (hnpq : ¬ (p ∧ q)) (hnp : ¬ p) : ¬ p :=
by
  exact hnp

end NUMINAMATH_GPT_proof_problem_l1093_109335


namespace NUMINAMATH_GPT_smallest_total_students_l1093_109300

theorem smallest_total_students :
  (∃ (n : ℕ), 4 * n + (n + 2) > 50 ∧ ∀ m, 4 * m + (m + 2) > 50 → m ≥ n) → 4 * 10 + (10 + 2) = 52 :=
by
  sorry

end NUMINAMATH_GPT_smallest_total_students_l1093_109300


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1093_109341

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1093_109341


namespace NUMINAMATH_GPT_maxwell_meets_brad_l1093_109396

variable (t : ℝ) -- time in hours
variable (distance_between_homes : ℝ) -- total distance
variable (maxwell_speed : ℝ) -- Maxwell's walking speed
variable (brad_speed : ℝ) -- Brad's running speed
variable (brad_delay : ℝ) -- Brad's start time delay

theorem maxwell_meets_brad 
  (hb: brad_delay = 1)
  (d: distance_between_homes = 34)
  (v_m: maxwell_speed = 4)
  (v_b: brad_speed = 6)
  (h : 4 * t + 6 * (t - 1) = distance_between_homes) :
  t = 4 := 
  sorry

end NUMINAMATH_GPT_maxwell_meets_brad_l1093_109396


namespace NUMINAMATH_GPT_angle_F_measure_l1093_109349

-- Define angle B
def angle_B := 120

-- Define angle C being supplementary to angle B on a straight line
def angle_C := 180 - angle_B

-- Define angle D
def angle_D := 45

-- Define angle E
def angle_E := 30

-- Define the vertically opposite angle F to angle C
def angle_F := angle_C

theorem angle_F_measure : angle_F = 60 :=
by
  -- Provide a proof by specifying sorry to indicate the proof is not complete
  sorry

end NUMINAMATH_GPT_angle_F_measure_l1093_109349


namespace NUMINAMATH_GPT_garden_table_bench_cost_l1093_109397

theorem garden_table_bench_cost (B T : ℕ) (h1 : T + B = 750) (h2 : T = 2 * B) : B = 250 :=
by
  sorry

end NUMINAMATH_GPT_garden_table_bench_cost_l1093_109397


namespace NUMINAMATH_GPT_larger_solution_of_quadratic_l1093_109387

theorem larger_solution_of_quadratic :
  ∀ x y : ℝ, x^2 - 19 * x - 48 = 0 ∧ y^2 - 19 * y - 48 = 0 ∧ x ≠ y →
  max x y = 24 :=
by
  sorry

end NUMINAMATH_GPT_larger_solution_of_quadratic_l1093_109387


namespace NUMINAMATH_GPT_form_x2_sub_2y2_l1093_109361

theorem form_x2_sub_2y2 (x y : ℤ) (hx : x % 2 = 1) : (x^2 - 2*y^2) % 8 = 1 ∨ (x^2 - 2*y^2) % 8 = -1 := 
sorry

end NUMINAMATH_GPT_form_x2_sub_2y2_l1093_109361


namespace NUMINAMATH_GPT_equal_lead_concentration_l1093_109344

theorem equal_lead_concentration (x : ℝ) (h1 : 0 < x) (h2 : x < 6) (h3 : x < 12) 
: (x / 6 = (12 - x) / 12) → x = 4 := by
  sorry

end NUMINAMATH_GPT_equal_lead_concentration_l1093_109344


namespace NUMINAMATH_GPT_find_x_l1093_109309

theorem find_x (x : ℝ) (h : 128/x + 75/x + 57/x = 6.5) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1093_109309


namespace NUMINAMATH_GPT_hotel_rolls_l1093_109313

theorem hotel_rolls (m n : ℕ) (rel_prime : Nat.gcd m n = 1) : 
  let num_nut_rolls := 3
  let num_cheese_rolls := 3
  let num_fruit_rolls := 3
  let total_rolls := 9
  let num_guests := 3
  let rolls_per_guest := 3
  let probability_first_guest := (3 / 9) * (3 / 8) * (3 / 7)
  let probability_second_guest := (2 / 6) * (2 / 5) * (2 / 4)
  let probability_third_guest := 1
  let overall_probability := probability_first_guest * probability_second_guest * probability_third_guest
  overall_probability = (9 / 70) → m = 9 ∧ n = 70 → m + n = 79 :=
by
  intros
  sorry

end NUMINAMATH_GPT_hotel_rolls_l1093_109313


namespace NUMINAMATH_GPT_generic_packages_needed_eq_2_l1093_109324

-- Define parameters
def tees_per_generic_package : ℕ := 12
def tees_per_aero_package : ℕ := 2
def members_foursome : ℕ := 4
def tees_needed_per_member : ℕ := 20
def aero_packages_purchased : ℕ := 28

-- Calculate total tees needed and total tees obtained from aero packages
def total_tees_needed : ℕ := members_foursome * tees_needed_per_member
def aero_tees_obtained : ℕ := aero_packages_purchased * tees_per_aero_package
def generic_tees_needed : ℕ := total_tees_needed - aero_tees_obtained

-- Prove the number of generic packages needed is 2
theorem generic_packages_needed_eq_2 : 
  generic_tees_needed / tees_per_generic_package = 2 :=
  sorry

end NUMINAMATH_GPT_generic_packages_needed_eq_2_l1093_109324


namespace NUMINAMATH_GPT_radius_of_unique_circle_l1093_109386

noncomputable def circle_radius (z : ℂ) (h k : ℝ) : ℝ :=
  if z = 2 then 1/4 else 0  -- function that determines the circle

def unique_circle_radius : Prop :=
  let x1 := 2
  let y1 := 0
  
  let x2 := 3 / 2
  let y2 := Real.sqrt 11 / 2

  let h := 7 / 4 -- x-coordinate of the circle's center
  let k := 0    -- y-coordinate of the circle's center

  let r := 1 / 4 -- Radius of the circle
  
  -- equation of the circle passing through (x1, y1) and (x2, y2) should satisfy
  -- the radius of the resulting circle is r

  (x1 - h)^2 + y1^2 = r^2 ∧ (x2 - h)^2 + y2^2 = r^2

theorem radius_of_unique_circle :
  unique_circle_radius :=
sorry

end NUMINAMATH_GPT_radius_of_unique_circle_l1093_109386


namespace NUMINAMATH_GPT_h_in_terms_of_f_l1093_109314

-- Definitions based on conditions in a)
def reflect_y_axis (f : ℝ → ℝ) (x : ℝ) := f (-x)
def shift_left (f : ℝ → ℝ) (x : ℝ) (c : ℝ) := f (x + c)

-- Express h(x) in terms of f(x) based on conditions
theorem h_in_terms_of_f (f : ℝ → ℝ) (x : ℝ) :
  reflect_y_axis (shift_left f 2) x = f (-x - 2) :=
by
  sorry

end NUMINAMATH_GPT_h_in_terms_of_f_l1093_109314


namespace NUMINAMATH_GPT_probability_of_meeting_l1093_109380

noncomputable def meeting_probability : ℝ :=
  let total_area := 10 * 10
  let favorable_area := 51
  favorable_area / total_area

theorem probability_of_meeting : meeting_probability = 51 / 100 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_meeting_l1093_109380


namespace NUMINAMATH_GPT_probability_selecting_A_l1093_109353

theorem probability_selecting_A :
  let total_people := 4
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_people
  probability = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_selecting_A_l1093_109353


namespace NUMINAMATH_GPT_total_rooms_to_paint_l1093_109399

-- Definitions based on conditions
def hours_per_room : ℕ := 8
def rooms_already_painted : ℕ := 8
def hours_to_paint_rest : ℕ := 16

-- Theorem statement
theorem total_rooms_to_paint :
  rooms_already_painted + hours_to_paint_rest / hours_per_room = 10 :=
  sorry

end NUMINAMATH_GPT_total_rooms_to_paint_l1093_109399


namespace NUMINAMATH_GPT_quadratic_eqn_a_range_l1093_109342

variable {a : ℝ}

theorem quadratic_eqn_a_range (a : ℝ) : (∃ x : ℝ, (a - 3) * x^2 - 4 * x + 1 = 0) ↔ a ≠ 3 :=
by sorry

end NUMINAMATH_GPT_quadratic_eqn_a_range_l1093_109342


namespace NUMINAMATH_GPT_goblins_return_l1093_109310

theorem goblins_return (n : ℕ) (f : Fin n → Fin n) (h1 : ∀ a, ∃! b, f a = b) (h2 : ∀ b, ∃! a, f a = b) : 
  ∃ k : ℕ, ∀ x : Fin n, (f^[k]) x = x := 
sorry

end NUMINAMATH_GPT_goblins_return_l1093_109310


namespace NUMINAMATH_GPT_Anton_thought_number_is_729_l1093_109307

theorem Anton_thought_number_is_729 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  ((n / 100 = 1 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 9) ∨
   (n / 100 = 7 ∧ (n / 10) % 10 = 0 ∧ n % 10 = 4) ∨
   (n / 100 = 1 ∧ (n / 10) % 10 = 2 ∧ n % 10 = 4)) → n = 729 :=
by sorry

end NUMINAMATH_GPT_Anton_thought_number_is_729_l1093_109307


namespace NUMINAMATH_GPT_find_number_l1093_109316

theorem find_number (x : ℕ) (h : x * 99999 = 65818408915) : x = 658185 :=
sorry

end NUMINAMATH_GPT_find_number_l1093_109316


namespace NUMINAMATH_GPT_total_cost_correct_l1093_109376

-- Define the cost of each category of items
def cost_of_book : ℕ := 16
def cost_of_binders : ℕ := 3 * 2
def cost_of_notebooks : ℕ := 6 * 1

-- Define the total cost calculation
def total_cost : ℕ := cost_of_book + cost_of_binders + cost_of_notebooks

-- Prove that the total cost of Léa's purchases is 28
theorem total_cost_correct : total_cost = 28 :=
by {
  -- This is where the proof would go, but it's omitted for now.
  sorry
}

end NUMINAMATH_GPT_total_cost_correct_l1093_109376


namespace NUMINAMATH_GPT_sum_of_fully_paintable_numbers_l1093_109308

def is_fully_paintable (h t u : ℕ) : Prop :=
  (∀ n : ℕ, (∀ k1 : ℕ, n ≠ 1 + k1 * h) ∧ (∀ k2 : ℕ, n ≠ 3 + k2 * t) ∧ (∀ k3 : ℕ, n ≠ 2 + k3 * u)) → False

theorem sum_of_fully_paintable_numbers :  ∃ L : List ℕ, (∀ x ∈ L, ∃ (h t u : ℕ), is_fully_paintable h t u ∧ 100 * h + 10 * t + u = x) ∧ L.sum = 944 :=
sorry

end NUMINAMATH_GPT_sum_of_fully_paintable_numbers_l1093_109308


namespace NUMINAMATH_GPT_greatest_AB_CBA_div_by_11_l1093_109332

noncomputable def AB_CBA_max_value (A B C : ℕ) : ℕ := 10001 * A + 1010 * B + 100 * C + 10 * B + A

theorem greatest_AB_CBA_div_by_11 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
  2 * A - 2 * B + C % 11 = 0 ∧ 
  ∀ (A' B' C' : ℕ),
    A' ≠ B' ∧ B' ≠ C' ∧ C' ≠ A' ∧ 
    2 * A' - 2 * B' + C' % 11 = 0 → 
    AB_CBA_max_value A B C ≥ AB_CBA_max_value A' B' C' :=
  by sorry

end NUMINAMATH_GPT_greatest_AB_CBA_div_by_11_l1093_109332


namespace NUMINAMATH_GPT_prove_value_of_custom_ops_l1093_109356

-- Define custom operations to match problem statement
def custom_op1 (x : ℤ) : ℤ := 7 - x
def custom_op2 (x : ℤ) : ℤ := x - 10

-- The main proof statement
theorem prove_value_of_custom_ops : custom_op2 (custom_op1 12) = -15 :=
by sorry

end NUMINAMATH_GPT_prove_value_of_custom_ops_l1093_109356


namespace NUMINAMATH_GPT_ratio_of_areas_l1093_109351

-- Definitions and conditions
variables (s r : ℝ)
variables (h1 : 4 * s = 4 * π * r)

-- Statement to prove
theorem ratio_of_areas (h1 : 4 * s = 4 * π * r) : s^2 / (π * r^2) = π := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1093_109351


namespace NUMINAMATH_GPT_largest_number_l1093_109323

theorem largest_number
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ)
  (ha : a = 0.883) (hb : b = 0.8839) (hc : c = 0.88) (hd : d = 0.839) (he : e = 0.889) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_number_l1093_109323


namespace NUMINAMATH_GPT_Eiffel_Tower_model_scale_l1093_109360

theorem Eiffel_Tower_model_scale
  (h_tower : ℝ := 324)
  (h_model_cm : ℝ := 18) :
  (h_tower / (h_model_cm / 100)) / 100 = 18 :=
by
  sorry

end NUMINAMATH_GPT_Eiffel_Tower_model_scale_l1093_109360


namespace NUMINAMATH_GPT_JerryAge_l1093_109368

-- Given definitions
def MickeysAge : ℕ := 20
def AgeRelationship (M J : ℕ) : Prop := M = 2 * J + 10

-- Proof statement
theorem JerryAge : ∃ J : ℕ, AgeRelationship MickeysAge J ∧ J = 5 :=
by
  sorry

end NUMINAMATH_GPT_JerryAge_l1093_109368


namespace NUMINAMATH_GPT_symmetric_point_correct_l1093_109381

def symmetric_point (P A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x₁, y₁, z₁) := P
  let (x₀, y₀, z₀) := A
  (2 * x₀ - x₁, 2 * y₀ - y₁, 2 * z₀ - z₁)

def P : ℝ × ℝ × ℝ := (3, -2, 4)
def A : ℝ × ℝ × ℝ := (0, 1, -2)
def expected_result : ℝ × ℝ × ℝ := (-3, 4, -8)

theorem symmetric_point_correct : symmetric_point P A = expected_result :=
  by
    sorry

end NUMINAMATH_GPT_symmetric_point_correct_l1093_109381


namespace NUMINAMATH_GPT_f_le_one_l1093_109304

open Real

theorem f_le_one (x : ℝ) (hx : 0 < x) : (1 + log x) / x ≤ 1 := 
sorry

end NUMINAMATH_GPT_f_le_one_l1093_109304


namespace NUMINAMATH_GPT_bernie_savings_l1093_109334

-- Defining conditions
def chocolates_per_week : ℕ := 2
def weeks : ℕ := 3
def chocolates_total : ℕ := chocolates_per_week * weeks
def local_store_cost_per_chocolate : ℕ := 3
def different_store_cost_per_chocolate : ℕ := 2

-- Defining the costs in both stores
def local_store_total_cost : ℕ := chocolates_total * local_store_cost_per_chocolate
def different_store_total_cost : ℕ := chocolates_total * different_store_cost_per_chocolate

-- The statement we want to prove
theorem bernie_savings : local_store_total_cost - different_store_total_cost = 6 :=
by
  sorry

end NUMINAMATH_GPT_bernie_savings_l1093_109334


namespace NUMINAMATH_GPT_range_of_a_l1093_109352

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1093_109352


namespace NUMINAMATH_GPT_train_cross_bridge_time_l1093_109348

def train_length : ℕ := 170
def train_speed_kmph : ℕ := 45
def bridge_length : ℕ := 205

def total_distance : ℕ := train_length + bridge_length
def train_speed_mps : ℕ := (train_speed_kmph * 1000) / 3600

theorem train_cross_bridge_time : (total_distance / train_speed_mps) = 30 := 
sorry

end NUMINAMATH_GPT_train_cross_bridge_time_l1093_109348


namespace NUMINAMATH_GPT_combination_sum_eq_l1093_109311

theorem combination_sum_eq :
  ∀ (n : ℕ), (2 * n ≥ 10 - 2 * n) ∧ (3 + n ≥ 2 * n) →
  Nat.choose (2 * n) (10 - 2 * n) + Nat.choose (3 + n) (2 * n) = 16 :=
by
  intro n h
  cases' h with h1 h2
  sorry

end NUMINAMATH_GPT_combination_sum_eq_l1093_109311


namespace NUMINAMATH_GPT_quadratic_function_relation_l1093_109391

theorem quadratic_function_relation 
  (y : ℝ → ℝ) 
  (y_def : ∀ x : ℝ, y x = x^2 + x + 1) 
  (y1 y2 y3 : ℝ) 
  (hA : y (-3) = y1) 
  (hB : y 2 = y2) 
  (hC : y (1/2) = y3) : 
  y3 < y1 ∧ y1 = y2 := 
sorry

end NUMINAMATH_GPT_quadratic_function_relation_l1093_109391


namespace NUMINAMATH_GPT_max_value_x_minus_2y_l1093_109377

open Real

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) : 
  x - 2*y ≤ 10 :=
sorry

end NUMINAMATH_GPT_max_value_x_minus_2y_l1093_109377


namespace NUMINAMATH_GPT_line_equation_l1093_109327

variable (x y : ℝ)

theorem line_equation (x1 y1 m : ℝ) (h : x1 = -2 ∧ y1 = 3 ∧ m = 2) :
    -2 * x + y = 1 := by
  sorry

end NUMINAMATH_GPT_line_equation_l1093_109327
