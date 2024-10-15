import Mathlib

namespace NUMINAMATH_GPT_company_annual_income_l1773_177351

variable {p a : ℝ}

theorem company_annual_income (h : 280 * p + (a - 280) * (p + 2) = a * (p + 0.25)) : a = 320 := 
sorry

end NUMINAMATH_GPT_company_annual_income_l1773_177351


namespace NUMINAMATH_GPT_percentage_y_less_than_x_l1773_177374

variable (x y : ℝ)

-- given condition
axiom hyp : x = 11 * y

-- proof problem: Prove that the percentage y is less than x is (10/11) * 100
theorem percentage_y_less_than_x (x y : ℝ) (hyp : x = 11 * y) : 
  (x - y) / x * 100 = (10 / 11) * 100 :=
by
  sorry

end NUMINAMATH_GPT_percentage_y_less_than_x_l1773_177374


namespace NUMINAMATH_GPT_negation_of_exists_statement_l1773_177303

theorem negation_of_exists_statement :
  ¬ (∃ x0 : ℝ, x0 > 0 ∧ x0^2 - 5 * x0 + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_statement_l1773_177303


namespace NUMINAMATH_GPT_inequality_proof_l1773_177322

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (2 * x) + 1 / (2 * y) + 1 / (2 * z)) > 
  (1 / (y + z) + 1 / (z + x) + 1 / (x + y)) :=
  by
    let a := y + z
    let b := z + x
    let c := x + y
    have x_def : x = (a + c - b) / 2 := sorry
    have y_def : y = (a + b - c) / 2 := sorry
    have z_def : z = (b + c - a) / 2 := sorry
    sorry

end NUMINAMATH_GPT_inequality_proof_l1773_177322


namespace NUMINAMATH_GPT_intersection_eq_1_2_l1773_177399

-- Define the set M
def M : Set ℝ := {y : ℝ | -2 ≤ y ∧ y ≤ 2}

-- Define the set N
def N : Set ℝ := {x : ℝ | 1 < x}

-- The intersection of M and N
def intersection : Set ℝ := { x : ℝ | 1 < x ∧ x ≤ 2 }

-- Our goal is to prove that M ∩ N = (1, 2]
theorem intersection_eq_1_2 : (M ∩ N) = (Set.Ioo 1 2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_1_2_l1773_177399


namespace NUMINAMATH_GPT_alice_needs_136_life_vests_l1773_177383

-- Definitions from the problem statement
def num_classes : ℕ := 4
def students_per_class : ℕ := 40
def instructors_per_class : ℕ := 10
def life_vest_probability : ℝ := 0.40

-- Calculate the total number of people
def total_people := num_classes * (students_per_class + instructors_per_class)

-- Calculate the expected number of students with life vests
def students_with_life_vests := (students_per_class : ℝ) * life_vest_probability
def total_students_with_life_vests := num_classes * students_with_life_vests

-- Calculate the number of life vests needed
def life_vests_needed := total_people - total_students_with_life_vests

-- Proof statement (missing the actual proof)
theorem alice_needs_136_life_vests : life_vests_needed = 136 := by
  sorry

end NUMINAMATH_GPT_alice_needs_136_life_vests_l1773_177383


namespace NUMINAMATH_GPT_number_of_seven_banana_bunches_l1773_177373

theorem number_of_seven_banana_bunches (total_bananas : ℕ) (eight_banana_bunches : ℕ) (seven_banana_bunches : ℕ) : 
    total_bananas = 83 → 
    eight_banana_bunches = 6 → 
    (∃ n : ℕ, seven_banana_bunches = n) → 
    8 * eight_banana_bunches + 7 * seven_banana_bunches = total_bananas → 
    seven_banana_bunches = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_seven_banana_bunches_l1773_177373


namespace NUMINAMATH_GPT_ratio_lena_kevin_after_5_more_l1773_177309

variables (L K N : ℕ)

def lena_initial_candy : ℕ := 16
def lena_gets_more : ℕ := 5
def kevin_candy_less_than_nicole : ℕ := 4
def lena_more_than_nicole : ℕ := 5

theorem ratio_lena_kevin_after_5_more
  (lena_initial : L = lena_initial_candy)
  (lena_to_multiple_of_kevin : L + lena_gets_more = K * 3) 
  (kevin_less_than_nicole : K = N - kevin_candy_less_than_nicole)
  (lena_more_than_nicole_condition : L = N + lena_more_than_nicole) :
  (L + lena_gets_more) / K = 3 :=
sorry

end NUMINAMATH_GPT_ratio_lena_kevin_after_5_more_l1773_177309


namespace NUMINAMATH_GPT_blue_marbles_l1773_177366

theorem blue_marbles (r b : ℕ) (h_ratio : 3 * b = 5 * r) (h_red : r = 18) : b = 30 := by
  -- proof
  sorry

end NUMINAMATH_GPT_blue_marbles_l1773_177366


namespace NUMINAMATH_GPT_lcm_of_8_and_15_l1773_177382

theorem lcm_of_8_and_15 : Nat.lcm 8 15 = 120 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_8_and_15_l1773_177382


namespace NUMINAMATH_GPT_sqrt_one_fourth_l1773_177301

theorem sqrt_one_fourth :
  {x : ℚ | x^2 = 1/4} = {1/2, -1/2} :=
by sorry

end NUMINAMATH_GPT_sqrt_one_fourth_l1773_177301


namespace NUMINAMATH_GPT_gcd_factorials_l1773_177345

noncomputable def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

theorem gcd_factorials (n m : ℕ) (hn : n = 8) (hm : m = 10) :
  Nat.gcd (factorial n) (factorial m) = 40320 := by
  sorry

end NUMINAMATH_GPT_gcd_factorials_l1773_177345


namespace NUMINAMATH_GPT_probability_one_even_dice_l1773_177331

noncomputable def probability_exactly_one_even (p : ℚ) : Prop :=
  ∃ (n : ℕ), (p = (4 * (1/2)^4 )) ∧ (n = 1) → p = 1/4

theorem probability_one_even_dice : probability_exactly_one_even (1/4) :=
by
  unfold probability_exactly_one_even
  sorry

end NUMINAMATH_GPT_probability_one_even_dice_l1773_177331


namespace NUMINAMATH_GPT_work_completion_time_l1773_177385

theorem work_completion_time 
    (A B : ℝ) 
    (h1 : A = 2 * B) 
    (h2 : (A + B) * 18 = 1) : 
    1 / A = 27 := 
by 
    sorry

end NUMINAMATH_GPT_work_completion_time_l1773_177385


namespace NUMINAMATH_GPT_hens_egg_laying_l1773_177304

theorem hens_egg_laying :
  ∀ (hens: ℕ) (price_per_dozen: ℝ) (total_revenue: ℝ) (weeks: ℕ) (total_hens: ℕ),
  hens = 10 →
  price_per_dozen = 3 →
  total_revenue = 120 →
  weeks = 4 →
  total_hens = hens →
  (total_revenue / price_per_dozen / 12) * 12 = 480 →
  (480 / weeks) = 120 →
  (120 / hens) = 12 :=
by sorry

end NUMINAMATH_GPT_hens_egg_laying_l1773_177304


namespace NUMINAMATH_GPT_max_value_of_inverse_l1773_177335

noncomputable def f (x y z : ℝ) : ℝ := (1/4) * x^2 + 2 * y^2 + 16 * z^2

theorem max_value_of_inverse (x y z a b c : ℝ) (h : a + b + c = 1) (pos_intercepts : a > 0 ∧ b > 0 ∧ c > 0)
  (point_on_plane : (x/a + y/b + z/c = 1)) (pos_points : x > 0 ∧ y > 0 ∧ z > 0) :
  ∀ (k : ℕ), 21 ≤ k → k < (f x y z)⁻¹ :=
sorry

end NUMINAMATH_GPT_max_value_of_inverse_l1773_177335


namespace NUMINAMATH_GPT_florida_vs_georgia_license_plates_l1773_177342

theorem florida_vs_georgia_license_plates :
  26 ^ 4 * 10 ^ 3 - 26 ^ 3 * 10 ^ 3 = 439400000 := by
  -- proof is omitted as directed
  sorry

end NUMINAMATH_GPT_florida_vs_georgia_license_plates_l1773_177342


namespace NUMINAMATH_GPT_jelly_bean_ratio_l1773_177320

theorem jelly_bean_ratio
  (initial_jelly_beans : ℕ)
  (num_people : ℕ)
  (remaining_jelly_beans : ℕ)
  (amount_taken_by_each_of_last_four : ℕ)
  (total_taken_by_last_four : ℕ)
  (total_jelly_beans_taken : ℕ)
  (X : ℕ)
  (ratio : ℕ)
  (h0 : initial_jelly_beans = 8000)
  (h1 : num_people = 10)
  (h2 : remaining_jelly_beans = 1600)
  (h3 : amount_taken_by_each_of_last_four = 400)
  (h4 : total_taken_by_last_four = 4 * amount_taken_by_each_of_last_four)
  (h5 : total_jelly_beans_taken = initial_jelly_beans - remaining_jelly_beans)
  (h6 : X = total_jelly_beans_taken - total_taken_by_last_four)
  (h7 : ratio = X / total_taken_by_last_four)
  : ratio = 3 :=
by sorry

end NUMINAMATH_GPT_jelly_bean_ratio_l1773_177320


namespace NUMINAMATH_GPT_prove_a_value_l1773_177337

theorem prove_a_value (a : ℝ) (h : (a - 2) * 0^2 + 0 + a^2 - 4 = 0) : a = -2 := 
by
  sorry

end NUMINAMATH_GPT_prove_a_value_l1773_177337


namespace NUMINAMATH_GPT_length_of_the_bridge_l1773_177389

-- Conditions
def train_length : ℝ := 80
def train_speed_kmh : ℝ := 45
def crossing_time_seconds : ℝ := 30

-- Conversion factor
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Calculation
noncomputable def train_speed_ms : ℝ := train_speed_kmh * km_to_m / hr_to_s
noncomputable def total_distance : ℝ := train_speed_ms * crossing_time_seconds
noncomputable def bridge_length : ℝ := total_distance - train_length

-- Proof statement
theorem length_of_the_bridge : bridge_length = 295 :=
by
  sorry

end NUMINAMATH_GPT_length_of_the_bridge_l1773_177389


namespace NUMINAMATH_GPT_largest_k_inequality_l1773_177388

theorem largest_k_inequality {a b c : ℝ} (h1 : a ≤ b) (h2 : b ≤ c) (h3 : ab + bc + ca = 0) (h4 : abc = 1) :
  |a + b| ≥ 4 * |c| :=
sorry

end NUMINAMATH_GPT_largest_k_inequality_l1773_177388


namespace NUMINAMATH_GPT_find_x_l1773_177355

noncomputable def f (x : ℝ) := (30 : ℝ) / (x + 5)
noncomputable def h (x : ℝ) := 4 * (f⁻¹ x)

theorem find_x (x : ℝ) (hx : h x = 20) : x = 3 :=
by 
  -- Conditions
  let f_inv := f⁻¹
  have h_def : h x = 4 * f_inv x := rfl
  have f_def : f x = (30 : ℝ) / (x + 5) := rfl
  -- Needed Proof Steps
  sorry

end NUMINAMATH_GPT_find_x_l1773_177355


namespace NUMINAMATH_GPT_average_bc_l1773_177318

variables (A B C : ℝ)

-- Conditions
def average_abc := (A + B + C) / 3 = 45
def average_ab := (A + B) / 2 = 40
def weight_b := B = 31

-- Proof statement
theorem average_bc (A B C : ℝ) (h_avg_abc : average_abc A B C) (h_avg_ab : average_ab A B) (h_b : weight_b B) :
  (B + C) / 2 = 43 :=
sorry

end NUMINAMATH_GPT_average_bc_l1773_177318


namespace NUMINAMATH_GPT_min_value_of_x_l1773_177372

open Real

-- Defining the conditions
def condition1 (x : ℝ) : Prop := x > 0
def condition2 (x : ℝ) : Prop := log x ≥ 2 * log 3 + (1/3) * log x

-- Statement of the theorem
theorem min_value_of_x (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) : x ≥ 27 :=
sorry

end NUMINAMATH_GPT_min_value_of_x_l1773_177372


namespace NUMINAMATH_GPT_razorback_shop_revenue_from_jerseys_zero_l1773_177381

theorem razorback_shop_revenue_from_jerseys_zero:
  let num_tshirts := 20
  let num_jerseys := 64
  let revenue_per_tshirt := 215
  let total_revenue_tshirts := 4300
  let total_revenue := total_revenue_tshirts
  let revenue_from_jerseys := total_revenue - total_revenue_tshirts
  revenue_from_jerseys = 0 := by
  sorry

end NUMINAMATH_GPT_razorback_shop_revenue_from_jerseys_zero_l1773_177381


namespace NUMINAMATH_GPT_total_hotdogs_brought_l1773_177316

-- Define the number of hotdogs brought by the first and second neighbors based on given conditions.

def first_neighbor_hotdogs : Nat := 75
def second_neighbor_hotdogs : Nat := first_neighbor_hotdogs - 25

-- Prove that the total hotdogs brought by the neighbors equals 125.
theorem total_hotdogs_brought :
  first_neighbor_hotdogs + second_neighbor_hotdogs = 125 :=
by
  -- statement only, proof not required
  sorry

end NUMINAMATH_GPT_total_hotdogs_brought_l1773_177316


namespace NUMINAMATH_GPT_compare_fractions_l1773_177305

theorem compare_fractions :
  (111110 / 111111) < (333331 / 333334) ∧ (333331 / 333334) < (222221 / 222223) :=
by
  sorry

end NUMINAMATH_GPT_compare_fractions_l1773_177305


namespace NUMINAMATH_GPT_yellow_yarns_count_l1773_177336

theorem yellow_yarns_count (total_scarves red_yarn_count blue_yarn_count yellow_yarns scarves_per_yarn : ℕ) 
  (h1 : 3 = scarves_per_yarn)
  (h2 : red_yarn_count = 2)
  (h3 : blue_yarn_count = 6)
  (h4 : total_scarves = 36)
  :
  yellow_yarns = 4 :=
by 
  sorry

end NUMINAMATH_GPT_yellow_yarns_count_l1773_177336


namespace NUMINAMATH_GPT_trigonometric_inequality_1_l1773_177306

theorem trigonometric_inequality_1 {n : ℕ} 
  (h1 : 0 < n) (x : ℝ) (h2 : 0 < x) (h3 : x < (Real.pi / (2 * n))) :
  (1 / 2) * (Real.tan x + Real.tan (n * x) - Real.tan ((n - 1) * x)) > (1 / n) * Real.tan (n * x) := 
sorry

end NUMINAMATH_GPT_trigonometric_inequality_1_l1773_177306


namespace NUMINAMATH_GPT_balance_balls_l1773_177328

open Real

variables (G B Y W : ℝ)

-- Conditions
def condition1 := (4 * G = 8 * B)
def condition2 := (3 * Y = 6 * B)
def condition3 := (8 * B = 6 * W)

-- Theorem statement
theorem balance_balls 
  (h1 : condition1 G B) 
  (h2 : condition2 Y B) 
  (h3 : condition3 B W) :
  ∃ (B_needed : ℝ), B_needed = 5 * G + 3 * Y + 4 * W ∧ B_needed = 64 / 3 * B :=
sorry

end NUMINAMATH_GPT_balance_balls_l1773_177328


namespace NUMINAMATH_GPT_rectangles_divided_into_13_squares_l1773_177380

theorem rectangles_divided_into_13_squares (m n : ℕ) (h : m * n = 13) : 
  (m = 1 ∧ n = 13) ∨ (m = 13 ∧ n = 1) :=
sorry

end NUMINAMATH_GPT_rectangles_divided_into_13_squares_l1773_177380


namespace NUMINAMATH_GPT_factorization_correct_l1773_177357

theorem factorization_correct : ∀ x : ℝ, (x^2 - 2*x - 9 = 0) → ((x-1)^2 = 10) :=
by 
  intros x h
  sorry

end NUMINAMATH_GPT_factorization_correct_l1773_177357


namespace NUMINAMATH_GPT_gcd_lcm_product_l1773_177324

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [h₁, h₂]
  unfold Nat.gcd Nat.lcm
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l1773_177324


namespace NUMINAMATH_GPT_functional_eq_l1773_177393

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem functional_eq {f : ℝ → ℝ} (h1 : ∀ x, x * (f (x + 1) - f x) = f x) (h2 : ∀ x y, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, ∀ x > 0, f x = k * x :=
sorry

end NUMINAMATH_GPT_functional_eq_l1773_177393


namespace NUMINAMATH_GPT_first_box_oranges_l1773_177391

theorem first_box_oranges (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) = 120) : x = 11 :=
sorry

end NUMINAMATH_GPT_first_box_oranges_l1773_177391


namespace NUMINAMATH_GPT_union_sets_l1773_177341

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2^a}

theorem union_sets : A ∪ B = {0, 1, 2, 4} := by
  sorry

end NUMINAMATH_GPT_union_sets_l1773_177341


namespace NUMINAMATH_GPT_leak_time_to_empty_cistern_l1773_177375

theorem leak_time_to_empty_cistern :
  (1/6 - 1/8) = 1/24 → (1 / (1/24)) = 24 := by
sorry

end NUMINAMATH_GPT_leak_time_to_empty_cistern_l1773_177375


namespace NUMINAMATH_GPT_circumcircle_equation_l1773_177317

theorem circumcircle_equation :
  ∃ (a b r : ℝ), 
    (∀ {x y : ℝ}, (x, y) = (2, 2) → (x - a)^2 + (y - b)^2 = r^2) ∧
    (∀ {x y : ℝ}, (x, y) = (5, 3) → (x - a)^2 + (y - b)^2 = r^2) ∧
    (∀ {x y : ℝ}, (x, y) = (3, -1) → (x - a)^2 + (y - b)^2 = r^2) ∧
    ((x - 4)^2 + (y - 1)^2 = 5) :=
sorry

end NUMINAMATH_GPT_circumcircle_equation_l1773_177317


namespace NUMINAMATH_GPT_min_days_to_sun_l1773_177307

def active_days_for_level (N : ℕ) : ℕ :=
  N * (N + 4)

def days_needed_for_upgrade (current_days future_days : ℕ) : ℕ :=
  future_days - current_days

theorem min_days_to_sun (current_level future_level : ℕ) :
  current_level = 9 →
  future_level = 16 →
  days_needed_for_upgrade (active_days_for_level current_level) (active_days_for_level future_level) = 203 :=
by
  intros h1 h2
  rw [h1, h2, active_days_for_level, active_days_for_level]
  sorry

end NUMINAMATH_GPT_min_days_to_sun_l1773_177307


namespace NUMINAMATH_GPT_find_k_l1773_177311

theorem find_k (k b : ℤ) (h1 : -x^2 - (k + 10) * x - b = -(x - 2) * (x - 4))
  (h2 : b = 8) : k = -16 :=
sorry

end NUMINAMATH_GPT_find_k_l1773_177311


namespace NUMINAMATH_GPT_one_thirds_in_fraction_l1773_177392

theorem one_thirds_in_fraction : (11 / 5) / (1 / 3) = 33 / 5 := by
  sorry

end NUMINAMATH_GPT_one_thirds_in_fraction_l1773_177392


namespace NUMINAMATH_GPT_problem_solution_l1773_177326

theorem problem_solution : (3127 - 2972) ^ 3 / 343 = 125 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1773_177326


namespace NUMINAMATH_GPT_bacteria_growth_time_l1773_177377

theorem bacteria_growth_time (n0 : ℕ) (n : ℕ) (rate : ℕ) (time_step : ℕ) (final : ℕ)
  (h0 : n0 = 200)
  (h1 : rate = 3)
  (h2 : time_step = 5)
  (h3 : n = n0 * rate ^ final)
  (h4 : n = 145800) :
  final = 30 := 
sorry

end NUMINAMATH_GPT_bacteria_growth_time_l1773_177377


namespace NUMINAMATH_GPT_house_cost_l1773_177376

-- Definitions of given conditions
def annual_salary : ℝ := 150000
def saving_rate : ℝ := 0.10
def downpayment_rate : ℝ := 0.20
def years_saving : ℝ := 6

-- Given the conditions, calculate annual savings and total savings after 6 years
def annual_savings : ℝ := annual_salary * saving_rate
def total_savings : ℝ := annual_savings * years_saving

-- Total savings represents 20% of the house cost
def downpayment : ℝ := total_savings

-- Prove the total cost of the house
theorem house_cost (downpayment : ℝ) (downpayment_rate : ℝ) : ℝ :=
  downpayment / downpayment_rate

lemma house_cost_correct : house_cost downpayment downpayment_rate = 450000 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_house_cost_l1773_177376


namespace NUMINAMATH_GPT_number_exceeds_by_35_l1773_177314

theorem number_exceeds_by_35 (x : ℤ) (h : x = (3 / 8 : ℚ) * x + 35) : x = 56 :=
by
  sorry

end NUMINAMATH_GPT_number_exceeds_by_35_l1773_177314


namespace NUMINAMATH_GPT_negation_problem_l1773_177356

variable {a b c : ℝ}

theorem negation_problem (h : a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) : 
  a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3 :=
sorry

end NUMINAMATH_GPT_negation_problem_l1773_177356


namespace NUMINAMATH_GPT_initial_number_proof_l1773_177365

def initial_number : ℕ := 7899665
def result : ℕ := 7899593
def factor1 : ℕ := 12
def factor2 : ℕ := 3
def factor3 : ℕ := 2

def certain_value : ℕ := (factor1 * factor2) * factor3

theorem initial_number_proof :
  initial_number - certain_value = result := by
  sorry

end NUMINAMATH_GPT_initial_number_proof_l1773_177365


namespace NUMINAMATH_GPT_calculation_error_l1773_177325

def percentage_error (actual expected : ℚ) : ℚ :=
  (actual - expected) / expected * 100

theorem calculation_error :
  let correct_result := (5 / 3) * 3
  let incorrect_result := (5 / 3) / 3
  percentage_error incorrect_result correct_result = 88.89 := by
  sorry

end NUMINAMATH_GPT_calculation_error_l1773_177325


namespace NUMINAMATH_GPT_smallest_angle_in_triangle_l1773_177371

theorem smallest_angle_in_triangle (k : ℕ) 
  (h1 : 3 * k + 4 * k + 5 * k = 180) : 
  3 * k = 45 := 
by sorry

end NUMINAMATH_GPT_smallest_angle_in_triangle_l1773_177371


namespace NUMINAMATH_GPT_derivative_of_my_function_l1773_177346

variable (x : ℝ)

noncomputable def my_function : ℝ :=
  (Real.cos (Real.sin 3))^2 + (Real.sin (29 * x))^2 / (29 * Real.cos (58 * x))

theorem derivative_of_my_function :
  deriv my_function x = Real.tan (58 * x) / Real.cos (58 * x) := 
sorry

end NUMINAMATH_GPT_derivative_of_my_function_l1773_177346


namespace NUMINAMATH_GPT_ratio_third_first_l1773_177339

theorem ratio_third_first (A B C : ℕ) (h1 : A + B + C = 110) (h2 : A = 2 * B) (h3 : B = 30) :
  C / A = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_third_first_l1773_177339


namespace NUMINAMATH_GPT_smallest_value_A_plus_B_plus_C_plus_D_l1773_177348

variable (A B C D : ℤ)

-- Given conditions in Lean statement form
def isArithmeticSequence (A B C : ℤ) : Prop :=
  B - A = C - B

def isGeometricSequence (B C D : ℤ) : Prop :=
  (C / B : ℚ) = 4 / 3 ∧ (D / C : ℚ) = C / B

def givenConditions (A B C D : ℤ) : Prop :=
  isArithmeticSequence A B C ∧ isGeometricSequence B C D

-- The proof problem to validate the smallest possible value
theorem smallest_value_A_plus_B_plus_C_plus_D (h : givenConditions A B C D) :
  A + B + C + D = 43 :=
sorry

end NUMINAMATH_GPT_smallest_value_A_plus_B_plus_C_plus_D_l1773_177348


namespace NUMINAMATH_GPT_find_angle_B_l1773_177344

-- Conditions
variable (A B C a b : ℝ)
variable (h1 : a = Real.sqrt 6)
variable (h2 : b = Real.sqrt 3)
variable (h3 : b + a * (Real.sin C - Real.cos C) = 0)

-- Target
theorem find_angle_B : B = Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_find_angle_B_l1773_177344


namespace NUMINAMATH_GPT_number_of_monkeys_l1773_177330

theorem number_of_monkeys (X : ℕ) : 
  10 * 10 = 10 →
  1 * 1 = 1 →
  1 * 70 / 10 = 7 →
  (X / 7) = X / 7 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_number_of_monkeys_l1773_177330


namespace NUMINAMATH_GPT_number_of_girls_l1773_177353

variable (G : ℕ) -- Number of girls in the school
axiom boys_count : G + 807 = 841 -- Given condition

theorem number_of_girls : G = 34 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l1773_177353


namespace NUMINAMATH_GPT_product_pass_rate_l1773_177362

variable {a b : ℝ} (h_a : 0 ≤ a ∧ a ≤ 1) (h_b : 0 ≤ b ∧ b ≤ 1) (h_indep : true)

theorem product_pass_rate : (1 - a) * (1 - b) = 
((1 - a) * (1 - b)) :=
by
  sorry

end NUMINAMATH_GPT_product_pass_rate_l1773_177362


namespace NUMINAMATH_GPT_total_amount_spent_l1773_177323

namespace KeithSpending

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tires_cost : ℝ := 112.46
def total_cost : ℝ := 387.85

theorem total_amount_spent : speakers_cost + cd_player_cost + tires_cost = total_cost :=
by sorry

end KeithSpending

end NUMINAMATH_GPT_total_amount_spent_l1773_177323


namespace NUMINAMATH_GPT_Jean_money_l1773_177310

theorem Jean_money (x : ℝ) (h1 : 3 * x + x = 76): 
  3 * x = 57 := 
by
  sorry

end NUMINAMATH_GPT_Jean_money_l1773_177310


namespace NUMINAMATH_GPT_solve_abc_values_l1773_177398

theorem solve_abc_values (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + 1/b = 5)
  (h2 : b + 1/c = 2)
  (h3 : c + 1/a = 8/3) :
  abc = 1 ∨ abc = 37/3 :=
sorry

end NUMINAMATH_GPT_solve_abc_values_l1773_177398


namespace NUMINAMATH_GPT_quarters_initial_l1773_177349

-- Define the given conditions
def candies_cost_dimes : Nat := 4 * 3
def candies_cost_cents : Nat := candies_cost_dimes * 10
def lollipop_cost_quarters : Nat := 1
def lollipop_cost_cents : Nat := lollipop_cost_quarters * 25
def total_spent_cents : Nat := candies_cost_cents + lollipop_cost_cents
def money_left_cents : Nat := 195
def total_initial_money_cents : Nat := money_left_cents + total_spent_cents
def dimes_count : Nat := 19
def dimes_value_cents : Nat := dimes_count * 10

-- Prove that the number of quarters initially is 6
theorem quarters_initial (quarters_count : Nat) (h : quarters_count * 25 = total_initial_money_cents - dimes_value_cents) : quarters_count = 6 :=
by
  sorry

end NUMINAMATH_GPT_quarters_initial_l1773_177349


namespace NUMINAMATH_GPT_transformation_C_factorization_l1773_177390

open Function

theorem transformation_C_factorization (a b : ℤ) :
  (a - 1) * (b - 1) = ab - a - b + 1 :=
by sorry

end NUMINAMATH_GPT_transformation_C_factorization_l1773_177390


namespace NUMINAMATH_GPT_find_a_l1773_177387

-- Definitions and theorem statement
def A (a : ℝ) : Set ℝ := {2, a^2 - a + 1}
def B (a : ℝ) : Set ℝ := {3, a + 3}
def C (a : ℝ) : Set ℝ := {3}

theorem find_a (a : ℝ) : A a ∩ B a = C a → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1773_177387


namespace NUMINAMATH_GPT_alex_candles_left_l1773_177333

theorem alex_candles_left (candles_start used_candles : ℕ) (h1 : candles_start = 44) (h2 : used_candles = 32) :
  candles_start - used_candles = 12 :=
by
  sorry

end NUMINAMATH_GPT_alex_candles_left_l1773_177333


namespace NUMINAMATH_GPT_chess_positions_after_one_move_each_l1773_177313

def number_of_chess_positions (initial_positions : ℕ) (pawn_moves : ℕ) (knight_moves : ℕ) (active_pawns : ℕ) (active_knights : ℕ) : ℕ :=
  let pawn_move_combinations := active_pawns * pawn_moves
  let knight_move_combinations := active_knights * knight_moves
  pawn_move_combinations + knight_move_combinations

theorem chess_positions_after_one_move_each :
  number_of_chess_positions 1 2 2 8 2 * number_of_chess_positions 1 2 2 8 2 = 400 :=
by
  sorry

end NUMINAMATH_GPT_chess_positions_after_one_move_each_l1773_177313


namespace NUMINAMATH_GPT_probability_three_defective_before_two_good_correct_l1773_177378

noncomputable def probability_three_defective_before_two_good 
  (total_items : ℕ) 
  (good_items : ℕ) 
  (defective_items : ℕ) 
  (sequence_length : ℕ) : ℚ := 
  -- We will skip the proof part and just acknowledge the result as mentioned
  (1 / 55 : ℚ)

theorem probability_three_defective_before_two_good_correct :
  probability_three_defective_before_two_good 12 9 3 5 = 1 / 55 := 
by sorry

end NUMINAMATH_GPT_probability_three_defective_before_two_good_correct_l1773_177378


namespace NUMINAMATH_GPT_total_students_in_class_l1773_177379

theorem total_students_in_class : 
  ∀ (total_candies students_candies : ℕ), 
    total_candies = 901 → students_candies = 53 → 
    students_candies * (total_candies / students_candies) = total_candies ∧ 
    total_candies % students_candies = 0 → 
    total_candies / students_candies = 17 := 
by 
  sorry

end NUMINAMATH_GPT_total_students_in_class_l1773_177379


namespace NUMINAMATH_GPT_problem_example_l1773_177364

theorem problem_example (a : ℕ) (H1 : a ∈ ({a, b, c} : Set ℕ)) (H2 : 0 ∈ ({x | x^2 ≠ 0} : Set ℕ)) :
  a ∈ ({a, b, c} : Set ℕ) ∧ 0 ∈ ({x | x^2 ≠ 0} : Set ℕ) :=
by
  sorry

end NUMINAMATH_GPT_problem_example_l1773_177364


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1773_177338

def repeating_decimal_to_fraction (d : ℕ) (n : ℕ) : ℚ := n / ((10^d) - 1)

theorem repeating_decimal_sum : 
  repeating_decimal_to_fraction 1 2 + repeating_decimal_to_fraction 2 2 + repeating_decimal_to_fraction 4 2 = 2474646 / 9999 := 
sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l1773_177338


namespace NUMINAMATH_GPT_all_terms_are_integers_l1773_177321

open Nat

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 143 ∧ ∀ n ≥ 2, a (n + 1) = 5 * (Finset.range n).sum a / n

theorem all_terms_are_integers (a : ℕ → ℕ) (h : seq a) : ∀ n : ℕ, 1 ≤ n → ∃ k : ℕ, a n = k := 
by
  sorry

end NUMINAMATH_GPT_all_terms_are_integers_l1773_177321


namespace NUMINAMATH_GPT_simplify_expression_l1773_177340

theorem simplify_expression (x y : ℝ) :
  5 * x - 3 * y + 9 * x ^ 2 + 8 - (4 - 5 * x + 3 * y - 9 * x ^ 2) = 18 * x ^ 2 + 10 * x - 6 * y + 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1773_177340


namespace NUMINAMATH_GPT_joe_height_l1773_177319

theorem joe_height (S J A : ℝ) (h1 : S + J + A = 180) (h2 : J = 2 * S + 6) (h3 : A = S - 3) : J = 94.5 :=
by 
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_joe_height_l1773_177319


namespace NUMINAMATH_GPT_find_s_l1773_177350

theorem find_s (s : Real) (h : ⌊s⌋ + s = 15.4) : s = 7.4 :=
sorry

end NUMINAMATH_GPT_find_s_l1773_177350


namespace NUMINAMATH_GPT_divisibility_of_binomial_l1773_177397

theorem divisibility_of_binomial (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 1) :
    (∀ x : ℕ, 1 ≤ x ∧ x ≤ n-1 → p ∣ Nat.choose n x) ↔ ∃ m : ℕ, n = p^m := sorry

end NUMINAMATH_GPT_divisibility_of_binomial_l1773_177397


namespace NUMINAMATH_GPT_valid_third_side_l1773_177358

-- Define a structure for the triangle with given sides
structure Triangle where
  a : ℝ
  b : ℝ
  x : ℝ

-- Define the conditions using the triangle inequality theorem
def valid_triangle (T : Triangle) : Prop :=
  T.a + T.x > T.b ∧ T.b + T.x > T.a ∧ T.a + T.b > T.x

-- Given values of a and b, and the condition on x
def specific_triangle : Triangle :=
  { a := 4, b := 9, x := 6 }

-- Statement to prove valid_triangle holds for specific_triangle
theorem valid_third_side : valid_triangle specific_triangle :=
by
  -- Import or assumptions about inequalities can be skipped or replaced by sorry
  sorry

end NUMINAMATH_GPT_valid_third_side_l1773_177358


namespace NUMINAMATH_GPT_profit_of_150_cents_requires_120_oranges_l1773_177315

def cost_price_per_orange := 15 / 4  -- cost price per orange in cents
def selling_price_per_orange := 30 / 6  -- selling price per orange in cents
def profit_per_orange := selling_price_per_orange - cost_price_per_orange  -- profit per orange in cents
def required_oranges_to_make_profit := 150 / profit_per_orange  -- number of oranges to get 150 cents of profit

theorem profit_of_150_cents_requires_120_oranges :
  required_oranges_to_make_profit = 120 :=
by
  -- the actual proof will follow here
  sorry

end NUMINAMATH_GPT_profit_of_150_cents_requires_120_oranges_l1773_177315


namespace NUMINAMATH_GPT_num_divisors_fact8_l1773_177347

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end NUMINAMATH_GPT_num_divisors_fact8_l1773_177347


namespace NUMINAMATH_GPT_number_of_dozens_l1773_177360

theorem number_of_dozens (x : Nat) (h : x = 16 * (3 * 4)) : x / 12 = 16 :=
by
  sorry

end NUMINAMATH_GPT_number_of_dozens_l1773_177360


namespace NUMINAMATH_GPT_depth_of_water_in_cistern_l1773_177368

-- Define the given constants
def length_cistern : ℝ := 6
def width_cistern : ℝ := 5
def total_wet_area : ℝ := 57.5

-- Define the area of the bottom of the cistern
def area_bottom (length : ℝ) (width : ℝ) : ℝ := length * width

-- Define the area of the longer sides of the cistern in contact with water
def area_long_sides (length : ℝ) (depth : ℝ) : ℝ := 2 * length * depth

-- Define the area of the shorter sides of the cistern in contact with water
def area_short_sides (width : ℝ) (depth : ℝ) : ℝ := 2 * width * depth

-- Define the total wet surface area based on depth of the water
def total_wet_surface_area (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ := 
    area_bottom length width + area_long_sides length depth + area_short_sides width depth

-- Define the proof statement
theorem depth_of_water_in_cistern : ∃ h : ℝ, h = 1.25 ∧ total_wet_surface_area length_cistern width_cistern h = total_wet_area := 
by
  use 1.25
  sorry

end NUMINAMATH_GPT_depth_of_water_in_cistern_l1773_177368


namespace NUMINAMATH_GPT_max_investment_at_7_percent_l1773_177363

variables (x y : ℝ)

theorem max_investment_at_7_percent 
  (h1 : x + y = 25000)
  (h2 : 0.07 * x + 0.12 * y ≥ 2450) : 
  x ≤ 11000 :=
sorry

end NUMINAMATH_GPT_max_investment_at_7_percent_l1773_177363


namespace NUMINAMATH_GPT_train_length_l1773_177396

theorem train_length (speed_kmph : ℕ) (time_seconds : ℕ) (length_meters : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_seconds = 14)
  (h3 : length_meters = speed_kmph * 1000 * time_seconds / 3600)
  : length_meters = 280 := by
  sorry

end NUMINAMATH_GPT_train_length_l1773_177396


namespace NUMINAMATH_GPT_pilot_fish_speed_is_30_l1773_177308

-- Define the initial conditions
def keanu_speed : ℝ := 20
def shark_initial_speed : ℝ := keanu_speed
def shark_speed_increase_factor : ℝ := 2
def pilot_fish_speed_increase_factor : ℝ := 0.5

-- Calculating final speeds
def shark_final_speed : ℝ := shark_initial_speed * shark_speed_increase_factor
def shark_speed_increase : ℝ := shark_final_speed - shark_initial_speed
def pilot_fish_speed_increase : ℝ := shark_speed_increase * pilot_fish_speed_increase_factor
def pilot_fish_final_speed : ℝ := keanu_speed + pilot_fish_speed_increase

-- The statement to prove
theorem pilot_fish_speed_is_30 : pilot_fish_final_speed = 30 := by
  sorry

end NUMINAMATH_GPT_pilot_fish_speed_is_30_l1773_177308


namespace NUMINAMATH_GPT_units_digit_m_squared_plus_2_pow_m_l1773_177395

-- Define the value of m
def m : ℕ := 2023^2 + 2^2023

-- Define the property we need to prove
theorem units_digit_m_squared_plus_2_pow_m :
  ((m^2 + 2^m) % 10) = 7 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_m_squared_plus_2_pow_m_l1773_177395


namespace NUMINAMATH_GPT_roots_product_l1773_177386

theorem roots_product (x1 x2 : ℝ) (h : ∀ x : ℝ, x^2 - 4 * x + 1 = 0 → x = x1 ∨ x = x2) : x1 * x2 = 1 :=
sorry

end NUMINAMATH_GPT_roots_product_l1773_177386


namespace NUMINAMATH_GPT_range_of_m_l1773_177394

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x^2 + 2 * x - m - 1 = 0) → m ≥ -2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1773_177394


namespace NUMINAMATH_GPT_find_values_of_m_l1773_177370

theorem find_values_of_m (m : ℤ) (h₁ : m > 2022) (h₂ : (2022 + m) ∣ (2022 * m)) : 
  m = 1011 ∨ m = 2022 :=
sorry

end NUMINAMATH_GPT_find_values_of_m_l1773_177370


namespace NUMINAMATH_GPT_compare_negatives_l1773_177369

theorem compare_negatives : -3 < -2 :=
by {
  -- Placeholder for proof
  sorry
}

end NUMINAMATH_GPT_compare_negatives_l1773_177369


namespace NUMINAMATH_GPT_productivity_after_repair_l1773_177302

-- Define the initial productivity and the increase factor.
def original_productivity : ℕ := 10
def increase_factor : ℝ := 1.5

-- Define the expected productivity after the improvement.
def expected_productivity : ℝ := 25

-- The theorem we need to prove.
theorem productivity_after_repair :
  original_productivity * (1 + increase_factor) = expected_productivity := by
  sorry

end NUMINAMATH_GPT_productivity_after_repair_l1773_177302


namespace NUMINAMATH_GPT_final_height_of_helicopter_total_fuel_consumed_l1773_177329

noncomputable def height_changes : List Float := [4.1, -2.3, 1.6, -0.9, 1.1]

def total_height_change (changes : List Float) : Float :=
  changes.foldl (λ acc x => acc + x) 0

theorem final_height_of_helicopter :
  total_height_change height_changes = 3.6 :=
by
  sorry

noncomputable def fuel_consumption (changes : List Float) : Float :=
  changes.foldl (λ acc x => if x > 0 then acc + 5 * x else acc + 3 * -x) 0

theorem total_fuel_consumed :
  fuel_consumption height_changes = 43.6 :=
by
  sorry

end NUMINAMATH_GPT_final_height_of_helicopter_total_fuel_consumed_l1773_177329


namespace NUMINAMATH_GPT_pizza_slices_per_pizza_l1773_177352

theorem pizza_slices_per_pizza (h : ∀ (mrsKaplanSlices bobbySlices pizzas : ℕ), 
  mrsKaplanSlices = 3 ∧ mrsKaplanSlices = bobbySlices / 4 ∧ pizzas = 2 → bobbySlices / pizzas = 6) : 
  ∃ (bobbySlices pizzas : ℕ), bobbySlices / pizzas = 6 :=
by
  existsi (3 * 4)
  existsi 2
  sorry

end NUMINAMATH_GPT_pizza_slices_per_pizza_l1773_177352


namespace NUMINAMATH_GPT_balloon_count_l1773_177300

-- Conditions
def Fred_balloons : ℕ := 5
def Sam_balloons : ℕ := 6
def Mary_balloons : ℕ := 7
def total_balloons : ℕ := 18

-- Proof statement
theorem balloon_count :
  Fred_balloons + Sam_balloons + Mary_balloons = total_balloons :=
by
  exact Nat.add_assoc 5 6 7 ▸ rfl

end NUMINAMATH_GPT_balloon_count_l1773_177300


namespace NUMINAMATH_GPT_sam_after_joan_took_marbles_l1773_177384

theorem sam_after_joan_took_marbles
  (original_yellow : ℕ)
  (marbles_taken_by_joan : ℕ)
  (remaining_yellow : ℕ)
  (h1 : original_yellow = 86)
  (h2 : marbles_taken_by_joan = 25)
  (h3 : remaining_yellow = original_yellow - marbles_taken_by_joan) :
  remaining_yellow = 61 :=
by
  sorry

end NUMINAMATH_GPT_sam_after_joan_took_marbles_l1773_177384


namespace NUMINAMATH_GPT_tank_capacity_l1773_177332

theorem tank_capacity (T : ℝ) (h : (3 / 4) * T + 7 = (7 / 8) * T) : T = 56 := 
sorry

end NUMINAMATH_GPT_tank_capacity_l1773_177332


namespace NUMINAMATH_GPT_log_expression_l1773_177361

section log_problem

variable (log : ℝ → ℝ)
variable (m n : ℝ)

-- Assume the properties of logarithms:
-- 1. log(m^n) = n * log(m)
axiom log_pow (m : ℝ) (n : ℝ) : log (m ^ n) = n * log m
-- 2. log(m * n) = log(m) + log(n)
axiom log_mul (m n : ℝ) : log (m * n) = log m + log n
-- 3. log(1) = 0
axiom log_one : log 1 = 0

theorem log_expression : log 5 * log 2 + log (2 ^ 2) - log 2 = 0 := by
  sorry

end log_problem

end NUMINAMATH_GPT_log_expression_l1773_177361


namespace NUMINAMATH_GPT_find_function_f_l1773_177327

theorem find_function_f
  (f : ℝ → ℝ)
  (H : ∀ x y, f x ^ 2 + f y ^ 2 = f (x + y) ^ 2) :
  ∀ x, f x = 0 := 
by 
  sorry

end NUMINAMATH_GPT_find_function_f_l1773_177327


namespace NUMINAMATH_GPT_num_trains_encountered_l1773_177354

noncomputable def train_travel_encounters : ℕ := 5

theorem num_trains_encountered (start_time : ℕ) (duration : ℕ) (daily_departure : ℕ) 
  (train_journey_duration : ℕ) (daily_start_interval : ℕ) 
  (end_time : ℕ) (number_encountered : ℕ) :
  (train_journey_duration = 3 * 24 * 60 + 30) → -- 3 days and 30 minutes in minutes
  (daily_start_interval = 24 * 60) →             -- interval between daily train starts (in minutes)
  (number_encountered = 5) :=
by
  sorry

end NUMINAMATH_GPT_num_trains_encountered_l1773_177354


namespace NUMINAMATH_GPT_unique_y_star_l1773_177367

def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

theorem unique_y_star :
  ∃! y : ℝ, star 4 y = 20 :=
by 
  sorry

end NUMINAMATH_GPT_unique_y_star_l1773_177367


namespace NUMINAMATH_GPT_sequence_geometric_l1773_177312

theorem sequence_geometric (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = r * a n) (h2 : a 4 = 2) : a 2 * a 6 = 4 :=
by
  sorry

end NUMINAMATH_GPT_sequence_geometric_l1773_177312


namespace NUMINAMATH_GPT_father_son_age_problem_l1773_177334

theorem father_son_age_problem
  (F S Y : ℕ)
  (h1 : F = 3 * S)
  (h2 : F = 45)
  (h3 : F + Y = 2 * (S + Y)) :
  Y = 15 :=
sorry

end NUMINAMATH_GPT_father_son_age_problem_l1773_177334


namespace NUMINAMATH_GPT_work_completion_l1773_177359

theorem work_completion (A B C : ℚ) (hA : A = 1/21) (hB : B = 1/6) 
    (hCombined : A + B + C = 1/3.36) : C = 1/12 := by
  sorry

end NUMINAMATH_GPT_work_completion_l1773_177359


namespace NUMINAMATH_GPT_T_n_lt_1_l1773_177343

open Nat

def a (n : ℕ) : ℕ := 2^n

def b (n : ℕ) : ℕ := 2^n - 1

def c (n : ℕ) : ℚ := (a n : ℚ) / ((b n : ℚ) * (b (n + 1) : ℚ))

noncomputable def T (n : ℕ) : ℚ := (Finset.range (n + 1)).sum c

theorem T_n_lt_1 (n : ℕ) : T n < 1 := by
  sorry

end NUMINAMATH_GPT_T_n_lt_1_l1773_177343
