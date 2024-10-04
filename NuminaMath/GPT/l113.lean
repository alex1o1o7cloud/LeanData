import Mathlib

namespace no_real_solutions_of_quadratic_eq_l113_113024

theorem no_real_solutions_of_quadratic_eq
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  ∀ x : ℝ, ¬ (b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 = 0) :=
by
  sorry

end no_real_solutions_of_quadratic_eq_l113_113024


namespace sequence_term_3001_exists_exactly_4_values_l113_113925

theorem sequence_term_3001_exists_exactly_4_values (x : ℝ) (h_pos : x > 0) :
  (∃ n, (1 < n) ∧ n < 6 ∧ (sequence n) = 3001) ↔ 
  (x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999) :=
by 
  -- Definition of the sequence a_n based on the provided recursive relationship
  let a : ℕ → ℝ
  | 1 => x
  | 2 => 3000
  | 3 => 3001 / x
  | 4 => (x + 3001) / (3000 * x)
  | 5 => (x + 1) / 3000
  | 6 => x
  | 7 => 3000
  | n => a ((n - 1) % 5 + 1)  -- Applying periodicity
  exactly[4] sorry

end sequence_term_3001_exists_exactly_4_values_l113_113925


namespace inequality_proof_l113_113379

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ((b + c - a)^2) / (a^2 + (b + c)^2) + ((c + a - b)^2) / (b^2 + (c + a)^2) + ((a + b - c)^2) / (c^2 + (a + b)^2) ≥ 3 / 5 :=
  sorry

end inequality_proof_l113_113379


namespace find_f_three_l113_113849

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l113_113849


namespace smallest_positive_multiple_45_l113_113250

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l113_113250


namespace total_gift_money_l113_113590

-- Definitions based on the conditions given in the problem
def initialAmount : ℕ := 159
def giftFromGrandmother : ℕ := 25
def giftFromAuntAndUncle : ℕ := 20
def giftFromParents : ℕ := 75

-- Lean statement to prove the total amount of money Chris has after receiving his birthday gifts
theorem total_gift_money : 
    initialAmount + giftFromGrandmother + giftFromAuntAndUncle + giftFromParents = 279 := by
sorry

end total_gift_money_l113_113590


namespace maria_chairs_l113_113796

variable (C : ℕ) -- Number of chairs Maria bought
variable (tables : ℕ := 2) -- Number of tables Maria bought is 2
variable (time_per_furniture : ℕ := 8) -- Time spent on each piece of furniture in minutes
variable (total_time : ℕ := 32) -- Total time spent assembling furniture

theorem maria_chairs :
  (time_per_furniture * C + time_per_furniture * tables = total_time) → C = 2 :=
by
  intro h
  sorry

end maria_chairs_l113_113796


namespace sum_of_integers_l113_113666

theorem sum_of_integers (x y : ℕ) (hxy_diff : x - y = 8) (hxy_prod : x * y = 240) (hx_gt_hy : x > y) : x + y = 32 := by
  sorry

end sum_of_integers_l113_113666


namespace find_f_3_l113_113847

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l113_113847


namespace green_socks_count_l113_113001

theorem green_socks_count: 
  ∀ (total_socks : ℕ) (white_socks : ℕ) (blue_socks : ℕ) (red_socks : ℕ) (green_socks : ℕ),
  total_socks = 900 →
  white_socks = total_socks / 3 →
  blue_socks = total_socks / 4 →
  red_socks = total_socks / 5 →
  green_socks = total_socks - (white_socks + blue_socks + red_socks) →
  green_socks = 195 :=
by
  intros total_socks white_socks blue_socks red_socks green_socks
  sorry

end green_socks_count_l113_113001


namespace income_of_sixth_member_l113_113702

def income_member1 : ℝ := 11000
def income_member2 : ℝ := 15000
def income_member3 : ℝ := 10000
def income_member4 : ℝ := 9000
def income_member5 : ℝ := 13000
def number_of_members : ℕ := 6
def average_income : ℝ := 12000
def total_income_of_five_members := income_member1 + income_member2 + income_member3 + income_member4 + income_member5

theorem income_of_sixth_member :
  6 * average_income - total_income_of_five_members = 14000 := by
  sorry

end income_of_sixth_member_l113_113702


namespace solve_equation_l113_113801

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) (h3 : x ≠ -1) :
  -x^2 = (4 * x + 2) / (x^2 + 3 * x + 2) ↔ x = -1 :=
by
  sorry

end solve_equation_l113_113801


namespace find_f_of_3_l113_113813

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l113_113813


namespace painting_cost_conversion_l113_113648

def paintingCostInCNY (paintingCostNAD : ℕ) (usd_to_nad : ℕ) (usd_to_cny : ℕ) : ℕ :=
  paintingCostNAD * (1 / usd_to_nad) * usd_to_cny

theorem painting_cost_conversion :
  (paintingCostInCNY 105 7 6 = 90) :=
by
  sorry

end painting_cost_conversion_l113_113648


namespace no_snow_five_days_l113_113679

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l113_113679


namespace convert_and_compute_l113_113464

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  if n = 231 then 2 * 4^2 + 3 * 4^1 + 1 * 4^0
  else if n = 21 then 2 * 4^1 + 1 * 4^0
  else if n = 3 then 3
  else 0

noncomputable def base10_to_base4 (n : ℕ) : ℕ :=
  if n = 135 then 2 * 4^2 + 1 * 4^1 + 3 * 4^0
  else 0

theorem convert_and_compute :
  base10_to_base4 ((base4_to_base10 231 / base4_to_base10 3) * base4_to_base10 21) = 213 :=
by {
  sorry
}

end convert_and_compute_l113_113464


namespace div_by_frac_eq_mult_multiply_12_by_4_equals_48_l113_113466

theorem div_by_frac_eq_mult (a b : ℝ) (hb : b ≠ 0) : a / (1 / b) = a * b :=
by
  sorry

theorem multiply_12_by_4_equals_48 : 12 * 4 = 48 :=
by
  have h : 12 / (1 / 4) = 12 * 4 := div_by_frac_eq_mult 12 4 (by norm_num)
  exact h.trans (by norm_num)

end div_by_frac_eq_mult_multiply_12_by_4_equals_48_l113_113466


namespace max_value_of_f_on_interval_exists_x_eq_min_1_l113_113677

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 + 6*x + 13)

theorem max_value_of_f_on_interval :
  ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → f x ≤ 1 / 4 := sorry

theorem exists_x_eq_min_1 : 
  ∃ x, -2 ≤ x ∧ x ≤ 2 ∧ f x = 1 / 4 := sorry

end max_value_of_f_on_interval_exists_x_eq_min_1_l113_113677


namespace math_problem_l113_113320

theorem math_problem :
  let numerator := (15^4 + 400) * (30^4 + 400) * (45^4 + 400) * (60^4 + 400) * (75^4 + 400)
  let denominator := (5^4 + 400) * (20^4 + 400) * (35^4 + 400) * (50^4 + 400) * (65^4 + 400)
  numerator / denominator = 301 :=
by 
  sorry

end math_problem_l113_113320


namespace repeatingDecimalSum_is_fraction_l113_113937

noncomputable def repeatingDecimalSum : ℚ :=
  (0.3333...).val + (0.040404...).val + (0.005005...).val

theorem repeatingDecimalSum_is_fraction : repeatingDecimalSum = 1134 / 2997 := by
  sorry

end repeatingDecimalSum_is_fraction_l113_113937


namespace original_cube_volume_eq_216_l113_113429

theorem original_cube_volume_eq_216 (a : ℕ)
  (h1 : ∀ (a : ℕ), ∃ V_orig V_new : ℕ, 
    V_orig = a^3 ∧ 
    V_new = (a + 1) * (a + 1) * (a - 2) ∧ 
    V_orig = V_new + 10) : 
  a = 6 → a^3 = 216 := 
by
  sorry

end original_cube_volume_eq_216_l113_113429


namespace fitness_club_alpha_is_more_advantageous_l113_113062

-- Define the costs and attendance pattern constants
def yearly_cost_alpha : ℕ := 11988
def monthly_cost_beta : ℕ := 1299
def weeks_per_month : ℕ := 4

-- Define the attendance pattern
def attendance_pattern : List ℕ := [3 * weeks_per_month, 2 * weeks_per_month, 1 * weeks_per_month, 0 * weeks_per_month]

-- Compute the total visits in a year for regular attendance
def total_visits (patterns : List ℕ) : ℕ :=
  patterns.sum * 3

-- Compute the total yearly cost for Beta when considering regular attendance
def yearly_cost_beta (monthly_cost : ℕ) : ℕ :=
  monthly_cost * 12

-- Calculate cost per visit for each club with given attendance
def cost_per_visit (total_cost : ℕ) (total_visits : ℕ) : ℚ :=
  total_cost / total_visits

theorem fitness_club_alpha_is_more_advantageous :
  cost_per_visit yearly_cost_alpha (total_visits attendance_pattern) <
  cost_per_visit (yearly_cost_beta monthly_cost_beta) (total_visits attendance_pattern) :=
by
  sorry

end fitness_club_alpha_is_more_advantageous_l113_113062


namespace find_f_of_3_l113_113819

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l113_113819


namespace smallest_positive_multiple_of_45_l113_113276

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l113_113276


namespace limes_remaining_l113_113129

-- Definitions based on conditions
def initial_limes : ℕ := 9
def limes_given_to_Sara : ℕ := 4

-- Theorem to prove
theorem limes_remaining : initial_limes - limes_given_to_Sara = 5 :=
by
  -- Sorry keyword to skip the actual proof
  sorry

end limes_remaining_l113_113129


namespace area_of_triangle_l113_113068

def point (α : Type*) := (α × α)

def x_and_y_lines (p : point ℝ) : Prop :=
  p.1 = p.2 ∨ p.1 = -p.2

def horizontal_line (y_val : ℝ) (p : point ℝ) : Prop :=
  p.2 = y_val

def vertices_of_triangle (p₁ p₂ p₃: point ℝ) : Prop :=
  horizontal_line 8 p₁ ∧ horizontal_line 8 p₂ ∧ x_and_y_lines p₃ ∧
  p₁ = (8, 8) ∧ p₂ = (-8, 8) ∧ p₃ = (0, 0)

theorem area_of_triangle : 
  ∃ (p₁ p₂ p₃ : point ℝ), vertices_of_triangle p₁ p₂ p₃ → 
  let base := abs (p₁.1 - p₂.1),
      height := abs (p₃.2 - p₁.2)
  in (1 / 2) * base * height = 64 := 
sorry

end area_of_triangle_l113_113068


namespace find_f_3_l113_113838

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l113_113838


namespace max_value_correct_l113_113790

noncomputable def max_value_ineq (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 3 * x + 2 * y + 6 * z = 1) : Prop :=
  x ^ 4 * y ^ 3 * z ^ 2 ≤ 1 / 372008

theorem max_value_correct (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 3 * x + 2 * y + 6 * z = 1) :
  max_value_ineq x y z h1 h2 h3 h4 :=
sorry

end max_value_correct_l113_113790


namespace four_possible_x_values_l113_113929

noncomputable def sequence_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a 1
  else if n = 2 then a 2
  else (a (n - 1) + 1) / (a (n - 2))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + 1) / (a (n - 2))

noncomputable def possible_values (x : ℝ) : Prop :=
  ∃ a : ℕ → ℝ, a 1 = x ∧ a 2 = 3000 ∧ satisfies_condition a ∧ 
                ∃ n, a n = 3001

theorem four_possible_x_values : 
  { x : ℝ | possible_values x }.finite ∧ 
  (card { x : ℝ | possible_values x } = 4) :=
  sorry

end four_possible_x_values_l113_113929


namespace smallest_positive_multiple_of_45_is_45_l113_113269

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113269


namespace find_number_l113_113768

theorem find_number (n : ℝ) (x : ℕ) (h1 : x = 4) (h2 : n^(2*x) = 3^(12-x)) : n = 3 := by
  sorry

end find_number_l113_113768


namespace jellybean_count_l113_113058

variable (initial_count removed1 added_back removed2 : ℕ)

theorem jellybean_count :
  initial_count = 37 →
  removed1 = 15 →
  added_back = 5 →
  removed2 = 4 →
  initial_count - removed1 + added_back - removed2 = 23 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jellybean_count_l113_113058


namespace cost_of_each_book_l113_113506

noncomputable def cost_of_book (money_given money_left notebook_cost notebook_count book_count : ℕ) : ℕ :=
  (money_given - money_left - (notebook_count * notebook_cost)) / book_count

-- Conditions
def money_given : ℕ := 56
def money_left : ℕ := 14
def notebook_cost : ℕ := 4
def notebook_count : ℕ := 7
def book_count : ℕ := 2

-- Theorem stating that the cost of each book is $7 under given conditions
theorem cost_of_each_book : cost_of_book money_given money_left notebook_cost notebook_count book_count = 7 := by
  sorry

end cost_of_each_book_l113_113506


namespace square_line_product_l113_113041

theorem square_line_product (b : ℝ) (h1 : y = 3) (h2 : y = 7) (h3 := x = 2) (h4 : x = b) : 
  (b = 6 ∨ b = -2) → (6 * -2 = -12) :=
by
  sorry

end square_line_product_l113_113041


namespace prime_square_remainders_l113_113583

theorem prime_square_remainders (p : ℕ) (hp : Nat.Prime p) (hgt : p > 5) : 
    {r | ∃ k : ℕ, p^2 = 180 * k + r}.finite ∧ 
    {r | ∃ k : ℕ, p^2 = 180 * k + r} = {1, 145} := 
by
  sorry

end prime_square_remainders_l113_113583


namespace twelfth_term_l113_113480

noncomputable def a (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else (n * (n + 2)) - ((n - 1) * (n + 1))

theorem twelfth_term : a 12 = 25 :=
by sorry

end twelfth_term_l113_113480


namespace candidate_a_votes_l113_113596

theorem candidate_a_votes (x : ℕ) (h : 2 * x + x = 21) : 2 * x = 14 :=
by sorry

end candidate_a_votes_l113_113596


namespace smallest_positive_multiple_45_l113_113248

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l113_113248


namespace solve_for_x_l113_113131

theorem solve_for_x (x : ℚ) (h : 5 * (x - 6) = 3 * (3 - 3 * x) + 9) : x = 24 / 7 :=
sorry

end solve_for_x_l113_113131


namespace recurring_sum_fractions_l113_113936

theorem recurring_sum_fractions :
  let x := (1 / 3) in
  let y := (4 / 99) in
  let z := (5 / 999) in
  x + y + z = (742 / 999) :=
by 
  sorry

end recurring_sum_fractions_l113_113936


namespace non_monotonic_piecewise_l113_113354

theorem non_monotonic_piecewise (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ (x t : ℝ),
    (f x = if x ≤ t then (4 * a - 3) * x + (2 * a - 4) else (2 * x^3 - 6 * x)))
  : a ≤ 3 / 4 := 
sorry

end non_monotonic_piecewise_l113_113354


namespace sum_of_reciprocals_l113_113540

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  (1 / x) + (1 / y) = 3 / 8 := 
sorry

end sum_of_reciprocals_l113_113540


namespace find_incorrect_value_of_observation_l113_113859

noncomputable def incorrect_observation_value (mean1 : ℝ) (mean2 : ℝ) (n : ℕ) : ℝ :=
  let old_sum := mean1 * n
  let new_sum := mean2 * n
  let correct_value := 45
  let incorrect_value := (old_sum - new_sum + correct_value)
  (incorrect_value / -1)

theorem find_incorrect_value_of_observation :
  incorrect_observation_value 36 36.5 50 = 20 :=
by
  -- By the problem setup, incorrect_observation_value 36 36.5 50 is as defined in the proof steps.
  -- As per the proof steps and calculation, incorrect_observation_value 36 36.5 50 should evaluate to 20.
  sorry

end find_incorrect_value_of_observation_l113_113859


namespace sufficient_but_not_necessary_l113_113180

variable {a : ℝ}

theorem sufficient_but_not_necessary (h : a > 1) : a^2 > a :=
by
  sorry

end sufficient_but_not_necessary_l113_113180


namespace fraction_to_decimal_l113_113593

theorem fraction_to_decimal : (3 : ℝ) / 50 = 0.06 := by
  sorry

end fraction_to_decimal_l113_113593


namespace part1_part2_l113_113355

noncomputable def f (x a : ℝ) : ℝ := |(x - a)| + |(x + 2)|

-- Part (1)
theorem part1 (x : ℝ) (h : f x 1 ≤ 7) : -4 ≤ x ∧ x ≤ 3 :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2 * a + 1) : a ≤ 1 :=
by
  sorry

end part1_part2_l113_113355


namespace paintings_per_gallery_l113_113311

theorem paintings_per_gallery (pencils_total: ℕ) (pictures_initial: ℕ) (galleries_new: ℕ) (pencils_per_signing: ℕ) (pencils_per_picture: ℕ) (pencils_for_signature: ℕ) :
  pencils_total = 88 ∧ pictures_initial = 9 ∧ galleries_new = 5 ∧ pencils_per_picture = 4 ∧ pencils_per_signing = 2 → 
  (pencils_total - (galleries_new + 1) * pencils_per_signing) / pencils_per_picture - pictures_initial = galleries_new * 2 :=
by
  intros h,
  cases h with ha hb,
  sorry

end paintings_per_gallery_l113_113311


namespace meaning_of_a2_add_b2_ne_zero_l113_113420

theorem meaning_of_a2_add_b2_ne_zero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
by
  sorry

end meaning_of_a2_add_b2_ne_zero_l113_113420


namespace digit_in_thousandths_place_l113_113079

theorem digit_in_thousandths_place :
  (decimals (7 / 32)).nth 3 = some 8 := 
sorry

end digit_in_thousandths_place_l113_113079


namespace longest_side_is_48_l113_113047

noncomputable def longest_side_of_triangle (a b c : ℝ) (ha : a/b = 1/2) (hb : b/c = 1/3) (hc : c/a = 1/4) (hp : a + b + c = 104) : ℝ :=
  a

theorem longest_side_is_48 {a b c : ℝ} (ha : a/b = 1/2) (hb : b/c = 1/3) (hc : c/a = 1/4) (hp : a + b + c = 104) : 
  longest_side_of_triangle a b c ha hb hc hp = 48 :=
sorry

end longest_side_is_48_l113_113047


namespace barium_oxide_amount_l113_113600

theorem barium_oxide_amount (BaO H2O BaOH₂ : ℕ) 
  (reaction : BaO + H2O = BaOH₂) 
  (molar_ratio : BaOH₂ = BaO) 
  (required_BaOH₂ : BaOH₂ = 2) :
  BaO = 2 :=
by 
  sorry

end barium_oxide_amount_l113_113600


namespace base_8_to_base_4_l113_113930

theorem base_8_to_base_4 (n : ℕ) (h : n = 6 * 8^2 + 5 * 8^1 + 3 * 8^0) : 
  (n : ℕ) = 1 * 4^4 + 2 * 4^3 + 2 * 4^2 + 2 * 4^1 + 3 * 4^0 :=
by
  -- Conversion proof goes here
  sorry

end base_8_to_base_4_l113_113930


namespace number_added_after_division_is_5_l113_113424

noncomputable def number_thought_of : ℕ := 72
noncomputable def result_after_division (n : ℕ) : ℕ := n / 6
noncomputable def final_result (n x : ℕ) : ℕ := result_after_division n + x

theorem number_added_after_division_is_5 :
  ∃ x : ℕ, final_result number_thought_of x = 17 ∧ x = 5 :=
by
  sorry

end number_added_after_division_is_5_l113_113424


namespace range_of_a1_of_arithmetic_sequence_l113_113482

theorem range_of_a1_of_arithmetic_sequence
  {a : ℕ → ℝ} (S : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum: ∀ n, S n = (n + 1) * (a 0 + a n) / 2)
  (h_min: ∀ n > 0, S n ≥ S 0)
  (h_S1: S 0 = 10) :
  -30 < a 0 ∧ a 0 < -27 := 
sorry

end range_of_a1_of_arithmetic_sequence_l113_113482


namespace evening_temperature_l113_113972

-- Definitions based on conditions
def noon_temperature : ℤ := 2
def temperature_drop : ℤ := 3

-- The theorem statement
theorem evening_temperature : noon_temperature - temperature_drop = -1 := 
by
  -- The proof is omitted
  sorry

end evening_temperature_l113_113972


namespace original_smallest_element_l113_113043

theorem original_smallest_element (x : ℤ) 
  (h1 : x < -1) 
  (h2 : x + 14 + 0 + 6 + 9 = 2 * (2 + 3 + 0 + 6 + 9)) : 
  x = -4 :=
by sorry

end original_smallest_element_l113_113043


namespace find_f_three_l113_113852

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l113_113852


namespace james_bought_400_fish_l113_113782

theorem james_bought_400_fish
  (F : ℝ)
  (h1 : 0.80 * F = 320)
  (h2 : F / 0.80 = 400) :
  F = 400 :=
by
  sorry

end james_bought_400_fish_l113_113782


namespace triangle_area_bounded_by_lines_l113_113075

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := dist A B
  let height := 8
  triangle_area A B O = 64 :=
sorry

end triangle_area_bounded_by_lines_l113_113075


namespace not_perfect_square_l113_113027

open Nat

theorem not_perfect_square (m n : ℕ) : ¬∃ k : ℕ, k^2 = 1 + 3^m + 3^n :=
by
  sorry

end not_perfect_square_l113_113027


namespace no_snow_five_days_l113_113685

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l113_113685


namespace prob_playerA_wins_4_to_1_prob_playerB_wins_and_match_lasts_more_than_5_games_dist_of_number_of_games_played_l113_113728

-- Define the conditions as assumptions
def playerA_wins_4_to_1 (p_win : ℝ) : Prop := comb 3 1 * (p_win ^ 4) * ((1 - p_win) ^ 1) = 1 / 8

def playerB_wins_and_match_lasts_more_than_5_games (p_win : ℝ) : Prop :=
  (comb 4 2 * (p_win ^ 4) * ((1 - p_win) ^ 2) + comb 5 3 * (p_win ^ 4) * ((1 - p_win) ^ 3)) = 15 / 64

def distribution_of_number_of_games (p_win : ℝ) : Prop :=
  (comb 3 0 * (p_win ^ 4) = 1 / 16) ∧
  (comb 4 1 * (p_win ^ 4) * ((1 - p_win) ^ 1) = 1 / 8) ∧
  (comb 5 2 * (p_win ^ 4) * ((1 - p_win) ^ 2) = 5 / 32) ∧
  (comb 6 3 * (p_win ^ 4) * ((1 - p_win) ^ 3) = 5 / 32)

-- Lean 4 statements that need proving
theorem prob_playerA_wins_4_to_1 (p_win : ℝ) (h : p_win = 1 / 2) : playerA_wins_4_to_1 p_win :=
by
  sorry

theorem prob_playerB_wins_and_match_lasts_more_than_5_games (p_win : ℝ) (h : p_win = 1 / 2) : 
  playerB_wins_and_match_lasts_more_than_5_games p_win :=
by 
  sorry

theorem dist_of_number_of_games_played (p_win : ℝ) (h : p_win = 1 / 2) : 
  distribution_of_number_of_games p_win :=
by 
  sorry

end prob_playerA_wins_4_to_1_prob_playerB_wins_and_match_lasts_more_than_5_games_dist_of_number_of_games_played_l113_113728


namespace no_snow_probability_l113_113690

theorem no_snow_probability (p_snow : ℚ) (h : p_snow = 2 / 3) :
  let p_no_snow := 1 - p_snow in
  (p_no_snow ^ 5 = 1 / 243) :=
by
  sorry

end no_snow_probability_l113_113690


namespace water_needed_quarts_l113_113059

-- Definitions from conditions
def ratio_water : ℕ := 8
def ratio_lemon : ℕ := 1
def total_gallons : ℚ := 1.5
def gallons_to_quarts : ℚ := 4

-- State what needs to be proven
theorem water_needed_quarts : 
  (total_gallons * gallons_to_quarts * (ratio_water / (ratio_water + ratio_lemon))) = 16 / 3 :=
by
  sorry

end water_needed_quarts_l113_113059


namespace teresa_ahmad_equation_l113_113031

theorem teresa_ahmad_equation (b c : ℝ) :
  (∀ x : ℝ, |x - 4| = 3 ↔ x = 7 ∨ x = 1) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ x = 7 ∨ x = 1) →
  (b, c) = (-8, 7) :=
by
  intro h1 h2
  sorry

end teresa_ahmad_equation_l113_113031


namespace smallest_b_gt_4_perfect_square_l113_113226

theorem smallest_b_gt_4_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ k : ℕ, 4 * b + 5 = k^2 ∧ b = 5 :=
by
  sorry

end smallest_b_gt_4_perfect_square_l113_113226


namespace smallest_n_common_factor_l113_113432

theorem smallest_n_common_factor :
  ∃ n : ℕ, n > 0 ∧ (∀ d : ℕ, d > 1 → d ∣ (11 * n - 4) → d ∣ (8 * n - 5)) ∧ n = 15 :=
by {
  -- Define the conditions as given in the problem
  sorry
}

end smallest_n_common_factor_l113_113432


namespace ab_root_of_Q_l113_113803

theorem ab_root_of_Q (a b : ℝ) (h : a ≠ b) (ha : a^4 + a^3 - 1 = 0) (hb : b^4 + b^3 - 1 = 0) :
  (ab : ℝ)^6 + (ab : ℝ)^4 + (ab : ℝ)^3 - (ab : ℝ)^2 - 1 = 0 := 
sorry

end ab_root_of_Q_l113_113803


namespace bounded_variation_l113_113184

theorem bounded_variation {f : ℝ → ℝ}
  (h1 : ∀ x ≥ 1, f x = ∫ t in (x - 1)..x, f t)
  (h2 : Differentiable ℝ f)
  : ∫ x in set.Ici (1:ℝ), |deriv f x| < ⊤ :=
begin
  sorry
end

end bounded_variation_l113_113184


namespace clive_money_l113_113318

noncomputable def clive_initial_money : ℝ  :=
  let total_olives := 80
  let olives_per_jar := 20
  let cost_per_jar := 1.5
  let change := 4
  let jars_needed := total_olives / olives_per_jar
  let total_cost := jars_needed * cost_per_jar
  total_cost + change

theorem clive_money (h1 : clive_initial_money = 10) : clive_initial_money = 10 :=
by sorry

end clive_money_l113_113318


namespace prism_lateral_edges_correct_cone_axial_section_equilateral_l113_113435

/-- Defining the lateral edges of a prism and its properties --/
structure Prism (r : ℝ) :=
(lateral_edges_equal : ∀ (e1 e2 : ℝ), e1 = r ∧ e2 = r)

/-- Defining the axial section of a cone with properties of base radius and generatrix length --/
structure Cone (r : ℝ) :=
(base_radius : ℝ := r)
(generatrix_length : ℝ := 2 * r)
(is_equilateral : base_radius * 2 = generatrix_length)

theorem prism_lateral_edges_correct (r : ℝ) (P : Prism r) : 
 ∃ e, e = r ∧ ∀ e', e' = r :=
by {
  sorry
}

theorem cone_axial_section_equilateral (r : ℝ) (C : Cone r) : 
 base_radius * 2 = generatrix_length :=
by {
  sorry
}

end prism_lateral_edges_correct_cone_axial_section_equilateral_l113_113435


namespace smallest_b_for_perfect_square_l113_113224

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ (∃ (n : ℤ), 4 * b + 5 = n ^ 2) ∧ b = 5 := 
sorry

end smallest_b_for_perfect_square_l113_113224


namespace jigsaw_puzzle_pieces_l113_113386

theorem jigsaw_puzzle_pieces
  (P : ℝ)
  (h1 : ∃ P, P = 0.90 * P + 0.72 * 0.10 * P + 0.504 * 0.08 * P + 504)
  (h2 : 0.504 * P = 504) :
  P = 1000 :=
by
  sorry

end jigsaw_puzzle_pieces_l113_113386


namespace least_positive_value_of_cubic_eq_l113_113644

theorem least_positive_value_of_cubic_eq (x y z w : ℕ) 
  (hx : Nat.Prime x) (hy : Nat.Prime y) 
  (hz : Nat.Prime z) (hw : Nat.Prime w) 
  (sum_lt_50 : x + y + z + w < 50) : 
  24 * x^3 + 16 * y^3 - 7 * z^3 + 5 * w^3 = 1464 :=
by
  sorry

end least_positive_value_of_cubic_eq_l113_113644


namespace emily_necklaces_l113_113474

theorem emily_necklaces (total_beads : ℤ) (beads_per_necklace : ℤ) 
(h_total_beads : total_beads = 16) (h_beads_per_necklace : beads_per_necklace = 8) : 
  total_beads / beads_per_necklace = 2 := 
by
  sorry

end emily_necklaces_l113_113474


namespace B_greater_than_A_l113_113327

def A := (54 : ℚ) / (5^7 * 11^4 : ℚ)
def B := (55 : ℚ) / (5^7 * 11^4 : ℚ)

theorem B_greater_than_A : B > A := by
  sorry

end B_greater_than_A_l113_113327


namespace find_y_from_x_squared_l113_113497

theorem find_y_from_x_squared (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = 6) : y = 29 :=
by
  sorry

end find_y_from_x_squared_l113_113497


namespace sum_of_x_and_y_l113_113964

theorem sum_of_x_and_y (x y : ℚ) (h1 : 1/x + 1/y = 3) (h2 : 1/x - 1/y = -7) : x + y = -3/10 :=
by
  sorry

end sum_of_x_and_y_l113_113964


namespace dwarf_diamond_distribution_l113_113133

-- Definitions for conditions
def dwarves : Type := Fin 8
structure State :=
  (diamonds : dwarves → ℕ)

-- Initial condition: Each dwarf has 3 diamonds
def initial_state : State := 
  { diamonds := fun _ => 3 }

-- Transition function: Each dwarf divides diamonds into two piles and passes them to neighbors
noncomputable def transition (s : State) : State := sorry

-- Proof goal: At a certain point in time, 3 specific dwarves have 24 diamonds in total,
-- with one dwarf having 7 diamonds, then prove the other two dwarves have 12 and 5 diamonds.
theorem dwarf_diamond_distribution (s : State)
  (h1 : ∃ t, s = (transition^[t]) initial_state ∧ ∃ i j k : dwarves, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    s.diamonds i + s.diamonds j + s.diamonds k = 24 ∧
    s.diamonds i = 7)
  : ∃ a b : dwarves, a ≠ b ∧ s.diamonds a = 12 ∧ s.diamonds b = 5 := sorry

end dwarf_diamond_distribution_l113_113133


namespace solve_system_l113_113168

theorem solve_system :
  ∃ (x y : ℕ), 
    (∃ d : ℕ, d ∣ 42 ∧ x^2 + y^2 = 468 ∧ d + (x * y) / d = 42) ∧ 
    (x = 12 ∧ y = 18) ∨ (x = 18 ∧ y = 12) :=
sorry

end solve_system_l113_113168


namespace math_problem_l113_113147

noncomputable def problem_statement (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hxyz : x * y * z = 1) : Prop :=
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + x) * (1 + z)) + z^3 / ((1 + x) * (1 + y))) ≥ 3 / 4

theorem math_problem (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hxyz : x * y * z = 1) :
  problem_statement x y z hx hy hz hxyz :=
sorry

end math_problem_l113_113147


namespace cookies_in_each_bag_l113_113647

-- Definitions based on the conditions
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41
def baggies : ℕ := 6

-- Assertion of the correct answer
theorem cookies_in_each_bag : 
  (chocolate_chip_cookies + oatmeal_cookies) / baggies = 9 := by
  sorry

end cookies_in_each_bag_l113_113647


namespace revenue_percentage_l113_113042

theorem revenue_percentage (R C : ℝ) (hR_pos : R > 0) (hC_pos : C > 0) :
  let projected_revenue := 1.20 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100 = 62.5 := by
  sorry

end revenue_percentage_l113_113042


namespace wall_length_eq_800_l113_113099

theorem wall_length_eq_800 
  (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_width : ℝ) (wall_height : ℝ)
  (num_bricks : ℝ) 
  (brick_volume : ℝ) 
  (total_brick_volume : ℝ)
  (wall_volume : ℝ) :
  brick_length = 25 → 
  brick_width = 11.25 → 
  brick_height = 6 → 
  wall_width = 600 → 
  wall_height = 22.5 → 
  num_bricks = 6400 → 
  brick_volume = brick_length * brick_width * brick_height → 
  total_brick_volume = brick_volume * num_bricks → 
  total_brick_volume = wall_volume →
  wall_volume = (800 : ℝ) * wall_width * wall_height :=
by
  sorry

end wall_length_eq_800_l113_113099


namespace num_divisors_36_l113_113164

theorem num_divisors_36 : ∃ n, n = 9 ∧ ∀ d : ℕ, d ∣ 36 → (d > 0 ∧ d ≤ 36) → ∃ k : ℕ, d = (2^k or 3^k or (2^p * 3^q) where p, q <= 2):
begin
  have factorization : 36 = 2^2 * 3^2 := by norm_num,
  have exponents : (2+1)*(2+1) = 9 := by norm_num,
  use 9,
  split,
  { exact exponents },
  {
    intros d hd hpos_range,
    cases hpos_range with hpos hrange,
    sorry -- Proof showing that there are exactly 9 positive divisors.
  },
end

end num_divisors_36_l113_113164


namespace walking_west_negation_l113_113974

theorem walking_west_negation (distance_east distance_west : Int) (h_east : distance_east = 6) (h_west : distance_west = -10) : 
    (10 : Int) = - distance_west := by
  sorry

end walking_west_negation_l113_113974


namespace minimum_value_side_c_l113_113502

open Real

noncomputable def minimum_side_c (a b c : ℝ) (B : ℝ) (S : ℝ) : ℝ := c

theorem minimum_value_side_c (a b c B : ℝ) (h1 : c * cos B = a + 1 / 2 * b)
  (h2 : S = sqrt 3 / 12 * c) :
  minimum_side_c a b c B S >= 1 :=
by
  -- Precise translation of mathematical conditions and required proof. 
  -- The actual steps to prove the theorem would be here.
  sorry

end minimum_value_side_c_l113_113502


namespace museum_pictures_l113_113580

theorem museum_pictures (P : ℕ) (h1 : ¬ (∃ k, P = 2 * k)) (h2 : ∃ k, P + 1 = 2 * k) : P = 3 := 
by 
  sorry

end museum_pictures_l113_113580


namespace decimal_expansion_2023rd_digit_l113_113331

theorem decimal_expansion_2023rd_digit 
  (x : ℚ) 
  (hx : x = 7 / 26) 
  (decimal_expansion : ℕ → ℕ)
  (hdecimal : ∀ n : ℕ, decimal_expansion n = if n % 12 = 0 
                        then 2 
                        else if n % 12 = 1 
                          then 7 
                          else if n % 12 = 2 
                            then 9 
                            else if n % 12 = 3 
                              then 2 
                              else if n % 12 = 4 
                                then 3 
                                else if n % 12 = 5 
                                  then 0 
                                  else if n % 12 = 6 
                                    then 7 
                                    else if n % 12 = 7 
                                      then 6 
                                      else if n % 12 = 8 
                                        then 9 
                                        else if n % 12 = 9 
                                          then 2 
                                          else if n % 12 = 10 
                                            then 3 
                                            else 0) :
  decimal_expansion 2023 = 0 :=
sorry

end decimal_expansion_2023rd_digit_l113_113331


namespace equation_D_is_quadratic_l113_113550

def is_quadratic_in_one_variable (eq : ℕ → Prop) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ eq = λ x, a * x ^ 2 + b * x + c

def equation_D : ℕ → Prop := λ x, x ^ 2 - 1 = 0

theorem equation_D_is_quadratic :
  is_quadratic_in_one_variable equation_D :=
sorry

end equation_D_is_quadratic_l113_113550


namespace seventyFifthTermInSequence_l113_113169

/-- Given a sequence that starts at 2 and increases by 4 each term, 
prove that the 75th term in this sequence is 298. -/
theorem seventyFifthTermInSequence : 
  (∃ a : ℕ → ℤ, (∀ n : ℕ, a n = 2 + 4 * n) ∧ a 74 = 298) :=
by
  sorry

end seventyFifthTermInSequence_l113_113169


namespace salem_size_comparison_l113_113400

theorem salem_size_comparison (S L : ℕ) (hL: L = 58940)
  (hSalem: S - 130000 = 2 * 377050) :
  (S / L = 15) :=
sorry

end salem_size_comparison_l113_113400


namespace x_lt_y_l113_113518

theorem x_lt_y (n : ℕ) (h_n : n > 2) (x y : ℝ) (h_x_pos : 0 < x) (h_y_pos : 0 < y) (h_x : x ^ n = x + 1) (h_y : y ^ (n + 1) = y ^ 3 + 1) : x < y :=
sorry

end x_lt_y_l113_113518


namespace sequence_arithmetic_progression_l113_113380

theorem sequence_arithmetic_progression (b : ℕ → ℕ) (b1_eq : b 1 = 1) (recurrence : ∀ n, b (n + 2) = b (n + 1) * b n + 1) : b 2 = 1 ↔ 
  ∃ d : ℕ, ∀ n, b (n + 1) - b n = d :=
sorry

end sequence_arithmetic_progression_l113_113380


namespace correct_quadratic_equation_l113_113551

-- The main statement to prove.
theorem correct_quadratic_equation :
  (∀ (x y a : ℝ), (3 * x + 2 * y - 1 ≠ 0) ∧ (5 * x^2 - 6 * y - 3 ≠ 0) ∧ (a * x^2 - x + 2 ≠ 0) ∧ (x^2 - 1 = 0) → (x^2 - 1 = 0)) :=
by
  sorry

end correct_quadratic_equation_l113_113551


namespace parallelepiped_face_areas_l113_113415

theorem parallelepiped_face_areas
    (h₁ : ℝ := 2)  -- height corresponding to face area x
    (h₂ : ℝ := 3)  -- height corresponding to face area y
    (h₃ : ℝ := 4)  -- height corresponding to face area z
    (total_surface_area : ℝ := 36) : 
    ∃ (x y z : ℝ), 
    2 * x + 2 * y + 2 * z = total_surface_area ∧
    (∃ V : ℝ, V = h₁ * x ∧ V = h₂ * y ∧ V = h₃ * z) ∧
    x = 108 / 13 ∧ y = 72 / 13 ∧ z = 54 / 13 := 
by 
  sorry

end parallelepiped_face_areas_l113_113415


namespace smallest_positive_multiple_of_45_l113_113228

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l113_113228


namespace analogical_reasoning_correct_l113_113291

theorem analogical_reasoning_correct (a b c : ℝ) (hc : c ≠ 0) : (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c :=
by
  sorry

end analogical_reasoning_correct_l113_113291


namespace jovana_shells_l113_113640

def initial_weight : ℕ := 5
def added_weight : ℕ := 23
def total_weight : ℕ := 28

theorem jovana_shells :
  initial_weight + added_weight = total_weight :=
by
  sorry

end jovana_shells_l113_113640


namespace last_three_digits_2021st_term_l113_113998

noncomputable def s (n : ℕ) : ℕ := n^3
noncomputable def x (n : ℕ) : ℕ := s n - s (n - 1)

theorem last_three_digits_2021st_term : (x 2021) % 1000 = 261 :=
by 
  -- Add the proof steps here.
  sorry

end last_three_digits_2021st_term_l113_113998


namespace absent_children_l113_113021

theorem absent_children (total_children bananas_per_child_if_present bananas_per_child_if_absent children_present absent_children : ℕ) 
  (H1 : total_children = 740)
  (H2 : bananas_per_child_if_present = 2)
  (H3 : bananas_per_child_if_absent = 4)
  (H4 : children_present * bananas_per_child_if_absent = total_children * bananas_per_child_if_present)
  (H5 : children_present = total_children - absent_children) : 
  absent_children = 370 :=
sorry

end absent_children_l113_113021


namespace paving_cost_l113_113676

def length : Real := 5.5
def width : Real := 3.75
def rate : Real := 700
def area : Real := length * width
def cost : Real := area * rate

theorem paving_cost :
  cost = 14437.50 :=
by
  -- Proof steps go here
  sorry

end paving_cost_l113_113676


namespace find_f_of_3_l113_113826

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l113_113826


namespace tom_and_jerry_same_speed_l113_113315

noncomputable def speed_of_tom (y : ℝ) : ℝ :=
  y^2 - 14*y + 45

noncomputable def speed_of_jerry (y : ℝ) : ℝ :=
  (y^2 - 2*y - 35) / (y - 5)

theorem tom_and_jerry_same_speed (y : ℝ) (h₁ : y ≠ 5) (h₂ : speed_of_tom y = speed_of_jerry y) :
  speed_of_tom y = 6 :=
by
  sorry

end tom_and_jerry_same_speed_l113_113315


namespace four_digit_numbers_with_conditions_l113_113767

theorem four_digit_numbers_with_conditions :
  ∃ n : ℕ, n = 126 ∧
    (∃ a b c d : ℕ,
      1000 ≤ a * 10^3 + b * 10^2 + c * 10 + d ∧ 
      a * 10^3 + b * 10^2 + c * 10 + d < 10000 ∧
      a ∈ {2, 6, 8} ∧ 
      d = 4 ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
      b ≠ c ∧ b ≠ d ∧
      c ≠ d) := 
begin
  use 126,
  split,
  { refl },
  { sorry }
end

end four_digit_numbers_with_conditions_l113_113767


namespace find_f_of_3_l113_113827

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l113_113827


namespace engineers_crimson_meet_in_tournament_l113_113713

noncomputable def probability_engineers_crimson_meet : ℝ := 
  1 - Real.exp (-1)

theorem engineers_crimson_meet_in_tournament :
  (∃ (n : ℕ), n = 128) → 
  (∀ (i : ℕ), i < 128 → (∀ (j : ℕ), j < 128 → i ≠ j → ∃ (p : ℝ), p = probability_engineers_crimson_meet)) :=
sorry

end engineers_crimson_meet_in_tournament_l113_113713


namespace sum_factorial_mod_21_l113_113501

open BigOperators
open Nat

theorem sum_factorial_mod_21 : 
  (∑ n in Finset.range 21, n.factorial) % 21 = 12 := 
sorry

end sum_factorial_mod_21_l113_113501


namespace smaller_solution_of_quadratic_eq_l113_113338

theorem smaller_solution_of_quadratic_eq : 
  (exists x y : ℝ, x < y ∧ x^2 - 13 * x + 36 = 0 ∧ y^2 - 13 * y + 36 = 0 ∧ x = 4) :=
by sorry

end smaller_solution_of_quadratic_eq_l113_113338


namespace angle_same_terminal_side_l113_113805

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, 290 = k * 360 - 70 :=
by
  sorry

end angle_same_terminal_side_l113_113805


namespace smallest_seating_l113_113446

theorem smallest_seating (N : ℕ) (h: ∀ (chairs : ℕ) (occupants : ℕ), 
  chairs = 100 ∧ occupants = 25 → 
  ∃ (adjacent_occupied: ℕ), adjacent_occupied > 0 ∧ adjacent_occupied < chairs ∧
  adjacent_occupied ≠ occupants) : 
  N = 25 :=
sorry

end smallest_seating_l113_113446


namespace equivalent_operation_l113_113725

theorem equivalent_operation (x : ℚ) :
  (x * (2/3)) / (5/6) = x * (4/5) :=
by
  -- Normal proof steps might follow here
  sorry

end equivalent_operation_l113_113725


namespace solution_set_of_inequality_l113_113942

theorem solution_set_of_inequality (x : ℝ) : (1 / x ≤ x) ↔ (-1 ≤ x ∧ x < 0) ∨ (x ≥ 1) := sorry

end solution_set_of_inequality_l113_113942


namespace digit_in_thousandths_place_of_7_over_32_is_8_l113_113077

theorem digit_in_thousandths_place_of_7_over_32_is_8 :
  (decimal_expansion_digit (7 / 32) 3) = 8 :=
sorry

end digit_in_thousandths_place_of_7_over_32_is_8_l113_113077


namespace find_f_prime_2_l113_113486

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f x / x

axiom tangent_coincide_at_points : ∀ f : ℝ → ℝ,
  (deriv f 0 = 1 / 2) ∧ (deriv (λ x, f x / x) 2 = 1 / 2)

theorem find_f_prime_2 :
  ∀ f : ℝ → ℝ, (f 2 = 2) ∧ ((∀ x, f 0 = 0)) ∧ tangent_coincide_at_points f → deriv f 2 = 2 :=
begin
  sorry
end

end find_f_prime_2_l113_113486


namespace find_f_3_l113_113836

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l113_113836


namespace smallest_positive_multiple_of_45_l113_113261

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l113_113261


namespace rational_solution_for_k_is_6_l113_113944

theorem rational_solution_for_k_is_6 (k : ℕ) (h : 0 < k) :
  (∃ x : ℚ, k * x ^ 2 + 12 * x + k = 0) ↔ k = 6 :=
by { sorry }

end rational_solution_for_k_is_6_l113_113944


namespace matrix_inverse_exists_and_correct_l113_113332

theorem matrix_inverse_exists_and_correct : 
  let A := matrix ([[3, 4], [-2, 9]] : matrix (fin 2) (fin 2) ℚ)
  let detA := (3 * 9) - (4 * -2)
  detA ≠ 0 →
  matrix.inv A = matrix ([[9/35, -4/35], [2/35, 3/35]] : matrix (fin 2) (fin 2) ℚ) :=
by
  let A := matrix ([[3, 4], [-2, 9]] : matrix (fin 2) (fin 2) ℚ)
  let detA := (3 * 9) - (4 * -2)
  have detA_nz : detA ≠ 0 := by simp [detA]
  have invA := (matrix.inv_of_det_ne_zero _ detA_nz)
  have matrix_inv_eq := (invA A)
  let expected_inv := (matrix ([[9/35, -4/35], [2/35, 3/35]] : matrix (fin 2) (fin 2) ℚ))
  exact matrix_inv_eq = expected_inv
  sorry

end matrix_inverse_exists_and_correct_l113_113332


namespace number_of_divisors_of_36_l113_113158

theorem number_of_divisors_of_36 : Nat.totient 36 = 9 := 
by sorry

end number_of_divisors_of_36_l113_113158


namespace no_positive_integer_solutions_l113_113538

theorem no_positive_integer_solutions : ¬∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ^ 4004 + y ^ 4004 = z ^ 2002 :=
by
  sorry

end no_positive_integer_solutions_l113_113538


namespace solve_divisor_problem_l113_113189

def divisor_problem : Prop :=
  ∃ D : ℕ, 12401 = (D * 76) + 13 ∧ D = 163

theorem solve_divisor_problem : divisor_problem :=
sorry

end solve_divisor_problem_l113_113189


namespace find_k_l113_113788

theorem find_k (a b c k : ℤ)
  (g : ℤ → ℤ)
  (h1 : ∀ x, g x = a * x^2 + b * x + c)
  (h2 : g 2 = 0)
  (h3 : 60 < g 6 ∧ g 6 < 70)
  (h4 : 90 < g 9 ∧ g 9 < 100)
  (h5 : 10000 * k < g 50 ∧ g 50 < 10000 * (k + 1)) :
  k = 0 :=
sorry

end find_k_l113_113788


namespace average_of_distinct_s_values_l113_113620

theorem average_of_distinct_s_values : 
  (1 + 5 + 2 + 4 + 3 + 3 + 4 + 2 + 5 + 1) / 3 = 7.33 :=
by
  sorry

end average_of_distinct_s_values_l113_113620


namespace smallest_positive_multiple_of_45_l113_113252

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l113_113252


namespace closest_point_to_line_l113_113605

theorem closest_point_to_line {x y : ℝ} :
  (y = 2 * x - 7) → (∃ p : ℝ × ℝ, p.1 = 5 ∧ p.2 = 3 ∧ (p.1, p.2) ∈ {q : ℝ × ℝ | q.2 = 2 * q.1 - 7} ∧ (∀ q : ℝ × ℝ, q ∈ {q : ℝ × ℝ | q.2 = 2 * q.1 - 7} → dist ⟨x, y⟩ p ≤ dist ⟨x, y⟩ q)) :=
by
  -- proof goes here
  sorry

end closest_point_to_line_l113_113605


namespace coordinates_F_l113_113427

-- Definition of point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Reflection over the y-axis
def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

-- Reflection over the x-axis
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Original point F
def F : Point := { x := 3, y := 3 }

-- First reflection over the y-axis
def F' := reflect_y F

-- Second reflection over the x-axis
def F'' := reflect_x F'

-- Goal: Coordinates of F'' after both reflections
theorem coordinates_F'' : F'' = { x := -3, y := -3 } :=
by
  -- Proof would go here
  sorry

end coordinates_F_l113_113427


namespace walking_time_l113_113891

theorem walking_time (v : ℕ) (d : ℕ) (h1 : v = 10) (h2 : d = 4) : 
    ∃ (T : ℕ), T = 24 := 
by
  sorry

end walking_time_l113_113891


namespace pascal_triangle_tenth_number_l113_113709

theorem pascal_triangle_tenth_number :
  let n := 50 in
  nat.choose n 9 = 2586948580 := 
by
  let n := 50
  have h : nat.choose n 9 = 2586948580 := sorry
  exact h

end pascal_triangle_tenth_number_l113_113709


namespace probability_white_ball_from_first_urn_correct_l113_113056

noncomputable def probability_white_ball_from_first_urn : ℝ :=
  let p_H1 : ℝ := 0.5
  let p_H2 : ℝ := 0.5
  let p_A_given_H1 : ℝ := 0.7
  let p_A_given_H2 : ℝ := 0.6
  let p_A : ℝ := p_H1 * p_A_given_H1 + p_H2 * p_A_given_H2
  p_H1 * p_A_given_H1 / p_A

theorem probability_white_ball_from_first_urn_correct :
  probability_white_ball_from_first_urn = 0.538 :=
sorry

end probability_white_ball_from_first_urn_correct_l113_113056


namespace smallest_b_gt_4_perfect_square_l113_113225

theorem smallest_b_gt_4_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ k : ℕ, 4 * b + 5 = k^2 ∧ b = 5 :=
by
  sorry

end smallest_b_gt_4_perfect_square_l113_113225


namespace binomial_19_10_l113_113914

theorem binomial_19_10 :
  ∀ (binom : ℕ → ℕ → ℕ),
  binom 17 7 = 19448 → binom 17 9 = 24310 →
  binom 19 10 = 92378 :=
by
  intros
  sorry

end binomial_19_10_l113_113914


namespace tangent_line_circle_l113_113496

theorem tangent_line_circle (r : ℝ) (h : r > 0) :
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = r^2 → x + y = 2 * r) ↔ r = 2 + Real.sqrt 2 :=
by
  sorry

end tangent_line_circle_l113_113496


namespace infinite_product_eq_cbrt_27_l113_113599

noncomputable def product_series : ℝ :=
  ∏' n : ℕ+ , (3 : ℝ)^((n : ℝ) / 3^(n : ℝ))

theorem infinite_product_eq_cbrt_27 : 
  product_series = real.exp ((3 : ℝ) / 4 * real.log 3) :=
by
  sorry

end infinite_product_eq_cbrt_27_l113_113599


namespace sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l113_113288

theorem sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30 :
  (7^30 + 13^30) % 100 = 0 := 
sorry

end sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l113_113288


namespace original_price_l113_113218

theorem original_price (price_paid original_price : ℝ) 
  (h₁ : price_paid = 5) 
  (h₂ : price_paid = original_price / 10) : 
  original_price = 50 := by
  sorry

end original_price_l113_113218


namespace total_unique_handshakes_l113_113541

def num_couples := 8
def num_individuals := num_couples * 2
def potential_handshakes_per_person := num_individuals - 1 - 1
def total_handshakes := num_individuals * potential_handshakes_per_person / 2

theorem total_unique_handshakes : total_handshakes = 112 := sorry

end total_unique_handshakes_l113_113541


namespace a_gt_b_l113_113943

variable (n : ℕ) (a b : ℝ)
variable (n_pos : n > 1) (a_pos : 0 < a) (b_pos : 0 < b)
variable (a_eqn : a^n = a + 1)
variable (b_eqn : b^{2 * n} = b + 3 * a)

theorem a_gt_b : a > b :=
by {
  -- Proof is needed here
  sorry
}

end a_gt_b_l113_113943


namespace ratio_comparison_l113_113637

-- Define the ratios in the standard and sport formulations
def ratio_flavor_corn_standard : ℚ := 1 / 12
def ratio_flavor_water_standard : ℚ := 1 / 30
def ratio_flavor_water_sport : ℚ := 1 / 60

-- Define the amounts of corn syrup and water in the sport formulation
def corn_syrup_sport : ℚ := 2
def water_sport : ℚ := 30

-- Calculate the amount of flavoring in the sport formulation
def flavoring_sport : ℚ := water_sport / 60

-- Calculate the ratio of flavoring to corn syrup in the sport formulation
def ratio_flavor_corn_sport : ℚ := flavoring_sport / corn_syrup_sport

-- Define the theorem to prove the ratio comparison
theorem ratio_comparison :
  (ratio_flavor_corn_sport / ratio_flavor_corn_standard) = 3 :=
by
  -- Using the given conditions and definitions, prove the theorem
  sorry

end ratio_comparison_l113_113637


namespace initial_sheep_count_l113_113026

theorem initial_sheep_count (S : ℕ) :
  let S1 := S - (S / 3 + 1 / 3)
  let S2 := S1 - (S1 / 4 + 1 / 4)
  let S3 := S2 - (S2 / 5 + 3 / 5)
  S3 = 409
  → S = 1025 := 
by 
  sorry

end initial_sheep_count_l113_113026


namespace P_subset_Q_l113_113511

def P (x : ℝ) := abs x < 2
def Q (x : ℝ) := x < 2

theorem P_subset_Q : ∀ x : ℝ, P x → Q x := by
  sorry

end P_subset_Q_l113_113511


namespace find_f_3_l113_113834

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l113_113834


namespace compute_expression_l113_113591

theorem compute_expression :
  -9 * 5 - (-(7 * -2)) + (-(11 * -6)) = 7 :=
by
  sorry

end compute_expression_l113_113591


namespace function_even_and_monotonically_increasing_l113_113579

-- Definition: Even Function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Definition: Monotonically Increasing on (0, ∞)
def is_monotonically_increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- Given Function
def f (x : ℝ) : ℝ := |x| + 1

-- Theorem to prove
theorem function_even_and_monotonically_increasing :
  is_even_function f ∧ is_monotonically_increasing_on_pos f := by
  sorry

end function_even_and_monotonically_increasing_l113_113579


namespace smallest_positive_multiple_of_45_is_45_l113_113270

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113270


namespace find_roots_of_parabola_l113_113500

-- Define the conditions given in the problem
variables (a b c : ℝ)
variable (a_nonzero : a ≠ 0)
variable (passes_through_1_0 : a * 1^2 + b * 1 + c = 0)
variable (axis_of_symmetry : -b / (2 * a) = -2)

-- Lean theorem statement
theorem find_roots_of_parabola (a b c : ℝ) (a_nonzero : a ≠ 0)
(passes_through_1_0 : a * 1^2 + b * 1 + c = 0) (axis_of_symmetry : -b / (2 * a) = -2) :
  (a * (-5)^2 + b * (-5) + c = 0) ∧ (a * 1^2 + b * 1 + c = 0) :=
by
  -- Placeholder for the proof
  sorry

end find_roots_of_parabola_l113_113500


namespace cindy_correct_answer_l113_113120

theorem cindy_correct_answer (x : ℝ) (h₀ : (x - 12) / 4 = 32) : (x - 7) / 5 = 27 :=
by
  sorry

end cindy_correct_answer_l113_113120


namespace pats_password_length_l113_113190

-- Definitions based on conditions
def num_lowercase_letters := 8
def num_uppercase_numbers := num_lowercase_letters / 2
def num_symbols := 2

-- Translate the math proof problem to Lean 4 statement
theorem pats_password_length : 
  num_lowercase_letters + num_uppercase_numbers + num_symbols = 14 := by
  sorry

end pats_password_length_l113_113190


namespace team_average_typing_speed_l113_113810

-- Definitions of typing speeds of each team member
def typing_speed_rudy := 64
def typing_speed_joyce := 76
def typing_speed_gladys := 91
def typing_speed_lisa := 80
def typing_speed_mike := 89

-- Number of team members
def number_of_team_members := 5

-- Total typing speed calculation
def total_typing_speed := typing_speed_rudy + typing_speed_joyce + typing_speed_gladys + typing_speed_lisa + typing_speed_mike

-- Average typing speed calculation
def average_typing_speed := total_typing_speed / number_of_team_members

-- Theorem statement
theorem team_average_typing_speed : average_typing_speed = 80 := by
  sorry

end team_average_typing_speed_l113_113810


namespace bookmarks_sold_l113_113595

-- Definitions pertaining to the problem
def total_books_sold : ℕ := 72
def books_ratio : ℕ := 9
def bookmarks_ratio : ℕ := 2

-- Statement of the theorem
theorem bookmarks_sold :
  (total_books_sold / books_ratio) * bookmarks_ratio = 16 :=
by
  sorry

end bookmarks_sold_l113_113595


namespace smallest_b_for_perfect_square_l113_113222

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ (∃ (n : ℤ), 4 * b + 5 = n ^ 2) ∧ b = 5 := 
sorry

end smallest_b_for_perfect_square_l113_113222


namespace smallest_n_for_divisibility_l113_113549

theorem smallest_n_for_divisibility (n : ℕ) (h : 2 ∣ 3^(2*n) - 1) (k : ℕ) : n = 2^(2007) := by
  sorry

end smallest_n_for_divisibility_l113_113549


namespace Lena_stops_in_X_l113_113797

def circumference : ℕ := 60
def distance_run : ℕ := 7920
def starting_point : String := "T"
def quarter_stops : String := "X"

theorem Lena_stops_in_X :
  (distance_run / circumference) * circumference + (distance_run % circumference) = distance_run →
  distance_run % circumference = 0 →
  (distance_run % circumference = 0 → starting_point = quarter_stops) →
  quarter_stops = "X" :=
sorry

end Lena_stops_in_X_l113_113797


namespace total_toys_correct_l113_113994

-- Define the conditions
def JaxonToys : ℕ := 15
def GabrielToys : ℕ := 2 * JaxonToys
def JerryToys : ℕ := GabrielToys + 8

-- Define the total number of toys
def TotalToys : ℕ := JaxonToys + GabrielToys + JerryToys

-- State the theorem
theorem total_toys_correct : TotalToys = 83 :=
by
  -- Skipping the proof as per instruction
  sorry

end total_toys_correct_l113_113994


namespace quadrilateral_area_correct_l113_113383

noncomputable def area_of_quadrilateral (n : ℕ) (hn : n > 0) : ℚ :=
  (2 * n^3) / (4 * n^2 - 1)

theorem quadrilateral_area_correct (n : ℕ) (hn : n > 0) :
  ∃ area : ℚ, area = (2 * n^3) / (4 * n^2 - 1) :=
by
  use area_of_quadrilateral n hn
  sorry

end quadrilateral_area_correct_l113_113383


namespace perfect_square_k_l113_113498

theorem perfect_square_k (a b k : ℝ) (h : ∃ c : ℝ, a^2 + 2*(k-3)*a*b + 9*b^2 = (a + c*b)^2) : 
  k = 6 ∨ k = 0 := 
sorry

end perfect_square_k_l113_113498


namespace X_is_N_l113_113999

theorem X_is_N (X : Set ℕ) (h_nonempty : ∃ x, x ∈ X)
  (h_condition1 : ∀ x ∈ X, 4 * x ∈ X)
  (h_condition2 : ∀ x ∈ X, Nat.floor (Real.sqrt x) ∈ X) : 
  X = Set.univ := 
sorry

end X_is_N_l113_113999


namespace nine_x_plus_twenty_seven_y_l113_113421

theorem nine_x_plus_twenty_seven_y (x y : ℤ) (h : 17 * x + 51 * y = 102) : 9 * x + 27 * y = 54 := 
by sorry

end nine_x_plus_twenty_seven_y_l113_113421


namespace smallest_positive_multiple_l113_113280

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l113_113280


namespace bottles_left_l113_113105

-- Define initial conditions
def bottlesInRefrigerator : Nat := 4
def bottlesInPantry : Nat := 4
def bottlesBought : Nat := 5
def bottlesDrank : Nat := 3

-- Goal: Prove the total number of bottles left
theorem bottles_left : bottlesInRefrigerator + bottlesInPantry + bottlesBought - bottlesDrank = 10 :=
by
  sorry

end bottles_left_l113_113105


namespace multiplication_difference_is_1242_l113_113303

theorem multiplication_difference_is_1242 (a b c : ℕ) (h1 : a = 138) (h2 : b = 43) (h3 : c = 34) :
  a * b - a * c = 1242 :=
by
  sorry

end multiplication_difference_is_1242_l113_113303


namespace unique_solution_l113_113739

theorem unique_solution (n : ℕ) (h1 : n > 0) (h2 : n^2 ∣ 3^n + 1) : n = 1 :=
sorry

end unique_solution_l113_113739


namespace initial_y_percentage_proof_l113_113296

variable (initial_volume : ℝ) (added_volume : ℝ) (initial_percentage_x : ℝ) (result_percentage_x : ℝ)

-- Conditions
def initial_volume_condition : Prop := initial_volume = 80
def added_volume_condition : Prop := added_volume = 20
def initial_percentage_x_condition : Prop := initial_percentage_x = 0.30
def result_percentage_x_condition : Prop := result_percentage_x = 0.44

-- Question
def initial_percentage_y (initial_volume added_volume initial_percentage_x result_percentage_x : ℝ) : ℝ :=
  1 - initial_percentage_x

-- Theorem
theorem initial_y_percentage_proof 
  (h1 : initial_volume_condition initial_volume)
  (h2 : added_volume_condition added_volume)
  (h3 : initial_percentage_x_condition initial_percentage_x)
  (h4 : result_percentage_x_condition result_percentage_x) :
  initial_percentage_y initial_volume added_volume initial_percentage_x result_percentage_x = 0.70 := 
sorry

end initial_y_percentage_proof_l113_113296


namespace lena_more_than_nicole_l113_113377

theorem lena_more_than_nicole (L K N : ℕ) 
  (h1 : L = 23)
  (h2 : 4 * K = L + 7)
  (h3 : K = N - 6) : L - N = 10 := sorry

end lena_more_than_nicole_l113_113377


namespace pets_remaining_is_correct_l113_113116

-- Definitions for the initial conditions and actions taken
def initial_puppies : Nat := 7
def initial_kittens : Nat := 6
def puppies_sold : Nat := 2
def kittens_sold : Nat := 3

-- Definition that calculates the remaining number of pets
def remaining_pets : Nat := initial_puppies + initial_kittens - (puppies_sold + kittens_sold)

-- The theorem to prove
theorem pets_remaining_is_correct : remaining_pets = 8 := by sorry

end pets_remaining_is_correct_l113_113116


namespace a_minus_b_perfect_square_l113_113183

theorem a_minus_b_perfect_square (a b : ℕ) (h : 2 * a^2 + a = 3 * b^2 + b) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℕ, a - b = k^2 :=
by sorry

end a_minus_b_perfect_square_l113_113183


namespace larger_number_is_450_l113_113094

-- Given conditions
def HCF := 30
def Factor1 := 10
def Factor2 := 15

-- Derived definitions needed for the proof
def LCM := HCF * Factor1 * Factor2

def Number1 := LCM / Factor1
def Number2 := LCM / Factor2

-- The goal is to prove the larger of the two numbers is 450
theorem larger_number_is_450 : max Number1 Number2 = 450 :=
by
  sorry

end larger_number_is_450_l113_113094


namespace lights_on_after_2011_toggles_l113_113204

-- Definitions for light states and index of lights
inductive Light : Type
| A | B | C | D | E | F | G
deriving DecidableEq

-- Initial light state: function from Light to Bool (true means the light is on)
def initialState : Light → Bool
| Light.A => true
| Light.B => false
| Light.C => true
| Light.D => false
| Light.E => true
| Light.F => false
| Light.G => true

-- Toggling function: toggles the state of a given light
def toggleState (state : Light → Bool) (light : Light) : Light → Bool :=
  fun l => if l = light then ¬ (state l) else state l

-- Toggling sequence: sequentially toggle lights in the given list
def toggleSequence (state : Light → Bool) (seq : List Light) : Light → Bool :=
  seq.foldl toggleState state

-- Toggles the sequence n times
def toggleNTimes (state : Light → Bool) (seq : List Light) (n : Nat) : Light → Bool :=
  let rec aux (state : Light → Bool) (n : Nat) : Light → Bool :=
    if n = 0 then state
    else aux (toggleSequence state seq) (n - 1)
  aux state n

-- Toggling sequence: A, B, C, D, E, F, G
def toggleSeq : List Light := [Light.A, Light.B, Light.C, Light.D, Light.E, Light.F, Light.G]

-- Determine the final state after 2011 toggles
def finalState : Light → Bool := toggleNTimes initialState toggleSeq 2011

-- Proof statement: the state of the lights after 2011 toggles is such that lights A, D, F are on
theorem lights_on_after_2011_toggles :
  finalState Light.A = true ∧
  finalState Light.D = true ∧
  finalState Light.F = true ∧
  finalState Light.B = false ∧
  finalState Light.C = false ∧
  finalState Light.E = false ∧
  finalState Light.G = false :=
by
  sorry

end lights_on_after_2011_toggles_l113_113204


namespace wire_around_field_l113_113939

theorem wire_around_field 
  (area_square : ℕ)
  (total_length_wire : ℕ)
  (h_area : area_square = 69696)
  (h_total_length : total_length_wire = 15840) :
  (total_length_wire / (4 * Int.natAbs (Int.sqrt area_square))) = 15 :=
  sorry

end wire_around_field_l113_113939


namespace total_toys_l113_113989

theorem total_toys 
  (jaxon_toys : ℕ)
  (gabriel_toys : ℕ)
  (jerry_toys : ℕ)
  (h1 : jaxon_toys = 15)
  (h2 : gabriel_toys = 2 * jaxon_toys)
  (h3 : jerry_toys = gabriel_toys + 8) : 
  jaxon_toys + gabriel_toys + jerry_toys = 83 :=
  by sorry

end total_toys_l113_113989


namespace range_of_b_l113_113765

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x + 2
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := f (f x b) b

theorem range_of_b (b : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f x b = y) → (∀ z : ℝ, ∃ x : ℝ, g x b = z) → b ≥ 4 ∨ b ≤ -2 :=
sorry

end range_of_b_l113_113765


namespace find_f_of_3_l113_113814

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l113_113814


namespace xyz_value_l113_113871

theorem xyz_value (x y z : ℝ) (h1 : 2 * x + 3 * y + z = 13) 
                              (h2 : 4 * x^2 + 9 * y^2 + z^2 - 2 * x + 15 * y + 3 * z = 82) : 
  x * y * z = 12 := 
by 
  sorry

end xyz_value_l113_113871


namespace a5_b5_sum_l113_113010

-- Definitions of arithmetic sequences
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable
def a : ℕ → ℝ := sorry -- defining the arithmetic sequences
noncomputable
def b : ℕ → ℝ := sorry

-- Common differences for the sequences
noncomputable
def d_a : ℝ := sorry
noncomputable
def d_b : ℝ := sorry

-- Conditions given in the problem
axiom a1_b1_sum : a 1 + b 1 = 7
axiom a3_b3_sum : a 3 + b 3 = 21
axiom a_is_arithmetic : arithmetic_seq a d_a
axiom b_is_arithmetic : arithmetic_seq b d_b

-- Theorem to be proved
theorem a5_b5_sum : a 5 + b 5 = 35 := 
by sorry

end a5_b5_sum_l113_113010


namespace car_speed_correct_l113_113405

noncomputable def car_speed (d v_bike t_delay : ℝ) (h1 : v_bike > 0) (h2 : t_delay > 0): ℝ := 2 * v_bike

theorem car_speed_correct:
  ∀ (d v_bike : ℝ) (t_delay : ℝ) (h1 : v_bike > 0) (h2 : t_delay > 0),
    (d / v_bike - t_delay = d / (car_speed d v_bike t_delay h1 h2)) → 
    car_speed d v_bike t_delay h1 h2 = 0.6 :=
by
  intros
  -- The proof would go here
  sorry

end car_speed_correct_l113_113405


namespace find_f_of_3_l113_113833

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l113_113833


namespace find_f_of_3_l113_113815

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l113_113815


namespace probability_no_snow_l113_113694

theorem probability_no_snow {
  let p_snow : ℚ := 2/3,     -- Probability of snow on any one day
  let p_no_snow : ℚ := 1 - p_snow,  -- Probability of no snow on any one day
  let p_no_snow_5_days : ℚ := p_no_snow ^ 5  -- Probability of no snow on any of the five days
} : p_no_snow_5_days = 1 / 243 := by
  sorry

end probability_no_snow_l113_113694


namespace smallest_positive_multiple_of_45_l113_113286

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l113_113286


namespace lines_do_not_form_triangle_l113_113359

noncomputable def line1 (x y : ℝ) := 3 * x - y + 2 = 0
noncomputable def line2 (x y : ℝ) := 2 * x + y + 3 = 0
noncomputable def line3 (m x y : ℝ) := m * x + y = 0

theorem lines_do_not_form_triangle (m : ℝ) :
  (∃ x y : ℝ, line1 x y ∧ line2 x y) →
  (∀ x y : ℝ, (line1 x y → line3 m x y) ∨ (line2 x y → line3 m x y) ∨ 
    (line1 x y ∧ line2 x y → line3 m x y)) →
  (m = -3 ∨ m = 2 ∨ m = -1) :=
by
  sorry

end lines_do_not_form_triangle_l113_113359


namespace smallest_positive_multiple_of_45_l113_113278

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l113_113278


namespace cos_C_in_triangle_l113_113983

theorem cos_C_in_triangle
  (A B C : ℝ) (sin_A : ℝ) (cos_B : ℝ)
  (h1 : sin_A = 4 / 5)
  (h2 : cos_B = 12 / 13) :
  cos (π - A - B) = -16 / 65 :=
by
  -- Proof steps would be included here
  sorry

end cos_C_in_triangle_l113_113983


namespace compare_negatives_l113_113912

theorem compare_negatives : - (1 : ℝ) / 2 < - (1 : ℝ) / 3 :=
by
  -- We start with the fact that 1/2 > 1/3
  have h : (1 : ℝ) / 2 > (1 : ℝ) / 3 := by
    linarith
    
  -- Negating the inequality reverses the sign
  exact neg_lt_neg_iff.mpr h

end compare_negatives_l113_113912


namespace square_line_product_l113_113040

theorem square_line_product (b : ℝ) (h1 : y = 3) (h2 : y = 7) (h3 := x = 2) (h4 : x = b) : 
  (b = 6 ∨ b = -2) → (6 * -2 = -12) :=
by
  sorry

end square_line_product_l113_113040


namespace not_snow_probability_l113_113681

theorem not_snow_probability :
  let p_snow : ℚ := 2 / 3,
      p_not_snow : ℚ := 1 - p_snow in
  (p_not_snow ^ 5) = 1 / 243 := by
  sorry

end not_snow_probability_l113_113681


namespace distance_to_x_axis_l113_113052

theorem distance_to_x_axis (x y : ℝ) :
  (x^2 / 9 - y^2 / 16 = 1) →
  (x^2 + y^2 = 25) →
  abs y = 16 / 5 :=
by
  -- Conditions: x^2 / 9 - y^2 / 16 = 1, x^2 + y^2 = 25
  -- Conclusion: abs y = 16 / 5 
  intro h1 h2
  sorry

end distance_to_x_axis_l113_113052


namespace pets_remaining_is_correct_l113_113115

-- Definitions for the initial conditions and actions taken
def initial_puppies : Nat := 7
def initial_kittens : Nat := 6
def puppies_sold : Nat := 2
def kittens_sold : Nat := 3

-- Definition that calculates the remaining number of pets
def remaining_pets : Nat := initial_puppies + initial_kittens - (puppies_sold + kittens_sold)

-- The theorem to prove
theorem pets_remaining_is_correct : remaining_pets = 8 := by sorry

end pets_remaining_is_correct_l113_113115


namespace divide_plane_into_regions_l113_113124

theorem divide_plane_into_regions :
  (∀ (x y : ℝ), y = 3 * x ∨ y = x / 3) →
  ∃ (regions : ℕ), regions = 4 :=
by
  sorry

end divide_plane_into_regions_l113_113124


namespace maximum_value_of_k_minus_b_l113_113352

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x - a * x + b

theorem maximum_value_of_k_minus_b (b : ℝ) (k : ℝ) (x : ℝ) 
  (h₀ : 0 ≤ b ∧ b ≤ 2) 
  (h₁ : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h₂ : ∀ x ∈ Set.Icc 1 (Real.exp 1), f x 1 b ≥ (k * x - x * Real.log x - 1)) :
  k - b ≤ 0 :=
sorry

end maximum_value_of_k_minus_b_l113_113352


namespace selling_price_l113_113308

theorem selling_price (cost_price : ℕ) (profit_percent : ℕ) (selling_price : ℕ) : 
  cost_price = 2400 ∧ profit_percent = 6 → selling_price = 2544 := by
  sorry

end selling_price_l113_113308


namespace arithmetic_sequence_problem_l113_113979

variable (a : ℕ → ℝ) (d : ℝ) (m : ℕ)

noncomputable def a_seq := ∀ n, a n = a 1 + (n - 1) * d

theorem arithmetic_sequence_problem
  (h1 : a 1 = 0)
  (h2 : d ≠ 0)
  (h3 : a m = a 1 + a 2 + a 3 + a 4 + a 5) :
  m = 11 :=
sorry

end arithmetic_sequence_problem_l113_113979


namespace area_triangle_OCD_l113_113064

open Real

-- Define points C and D based on the given conditions.
def C := (8, 8)
def D := (-8, 8)

-- Define the base length between points C and D.
def base := dist C D

-- Height from the origin to the line y = 8.
def height := 8

-- Statement to prove the area of triangle OCD.
theorem area_triangle_OCD : (1 / 2) * base * height = 64 := by
  sorry

end area_triangle_OCD_l113_113064


namespace original_game_start_player_wins_modified_game_start_player_wins_l113_113704

def divisor_game_condition (num : ℕ) := ∀ d : ℕ, d ∣ num → ∀ x : ℕ, x ∣ d → x = d ∨ x = 1
def modified_divisor_game_condition (num d_prev : ℕ) := ∀ d : ℕ, d ∣ num → d ≠ d_prev → ∃ k l : ℕ, d = k * l ∧ k ≠ 1 ∧ l ≠ 1 ∧ k ≤ l

/-- Prove that if the starting player plays wisely, they will always win the original game. -/
theorem original_game_start_player_wins : ∀ d : ℕ, divisor_game_condition 1000 → d = 100 → (∃ p : ℕ, p != 1000) := 
sorry

/-- What happens if the game is modified such that a divisor cannot be mentioned if it has fewer divisors than any previously mentioned number? -/
theorem modified_game_start_player_wins : ∀ d_prev : ℕ, modified_divisor_game_condition 1000 d_prev → d_prev = 100 → (∃ p : ℕ, p != 1000) := 
sorry

end original_game_start_player_wins_modified_game_start_player_wins_l113_113704


namespace union_cardinality_inequality_l113_113959

open Set

/-- Given three finite sets A, B, and C such that A ∩ B ∩ C = ∅,
prove that |A ∪ B ∪ C| ≥ 1/2 (|A| + |B| + |C|) -/
theorem union_cardinality_inequality (A B C : Finset ℕ)
  (h : (A ∩ B ∩ C) = ∅) : (A ∪ B ∪ C).card ≥ (A.card + B.card + C.card) / 2 := sorry

end union_cardinality_inequality_l113_113959


namespace pedro_plums_l113_113523

theorem pedro_plums :
  ∃ P Q : ℕ, P + Q = 32 ∧ 2 * P + Q = 52 ∧ P = 20 :=
by
  sorry

end pedro_plums_l113_113523


namespace A_inter_B_empty_l113_113358

def Z_plus := { n : ℤ // 0 < n }

def A : Set ℤ := { x | ∃ n : Z_plus, x = 2 * (n.1) - 1 }
def B : Set ℤ := { y | ∃ x ∈ A, y = 3 * x - 1 }

theorem A_inter_B_empty : A ∩ B = ∅ :=
by {
  sorry
}

end A_inter_B_empty_l113_113358


namespace square_nonnegative_for_rat_l113_113436

theorem square_nonnegative_for_rat (x : ℚ) : x^2 ≥ 0 :=
sorry

end square_nonnegative_for_rat_l113_113436


namespace no_snow_five_days_l113_113678

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l113_113678


namespace digit_in_thousandths_place_l113_113707

theorem digit_in_thousandths_place : (3 / 16 : ℚ) = 0.1875 :=
by sorry

end digit_in_thousandths_place_l113_113707


namespace julia_song_download_l113_113006

theorem julia_song_download : 
  let internet_speed := 20 -- in MBps
  let half_hour_in_minutes := 30
  let size_per_song := 5 -- in MB
  (internet_speed * 60 * half_hour_in_minutes) / size_per_song = 7200 :=
by
  sorry

end julia_song_download_l113_113006


namespace arrangement_count_l113_113564

theorem arrangement_count (n m : ℕ) (h : n = 7 ∧ m = 3) :
  (n + m).choose m = 120 :=
by
  rw [h.1, h.2]
  unfold nat.choose
  sorry

end arrangement_count_l113_113564


namespace student_tickets_sold_l113_113314

theorem student_tickets_sold (S NS : ℕ) (h1 : S + NS = 150) (h2 : 5 * S + 8 * NS = 930) : S = 90 :=
by
  sorry

end student_tickets_sold_l113_113314


namespace moles_of_Na2SO4_formed_l113_113470

/-- 
Given the following conditions:
1. 1 mole of H2SO4 reacts with 2 moles of NaOH.
2. In the presence of 0.5 moles of HCl and 0.5 moles of KOH.
3. At a temperature of 25°C and a pressure of 1 atm.
Prove that the moles of Na2SO4 formed is 1 mole.
-/

theorem moles_of_Na2SO4_formed
  (H2SO4 : ℝ) -- moles of H2SO4
  (NaOH : ℝ) -- moles of NaOH
  (HCl : ℝ) -- moles of HCl
  (KOH : ℝ) -- moles of KOH
  (T : ℝ) -- temperature in °C
  (P : ℝ) -- pressure in atm
  : H2SO4 = 1 ∧ NaOH = 2 ∧ HCl = 0.5 ∧ KOH = 0.5 ∧ T = 25 ∧ P = 1 → 
  ∃ Na2SO4 : ℝ, Na2SO4 = 1 :=
by
  sorry

end moles_of_Na2SO4_formed_l113_113470


namespace correct_propositions_l113_113753

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * (Real.sin x + Real.cos x)

-- Proposition 2: Symmetry about the line x = -3π/4
def proposition_2 : Prop := ∀ x, f (x + 3 * Real.pi / 4) = f (-x)

-- Proposition 3: There exists φ ∈ ℝ, such that the graph of the function f(x + φ) is centrally symmetric about the origin
def proposition_3 : Prop := ∃ φ : ℝ, ∀ x, f (x + φ) = -f (-x)

theorem correct_propositions :
  (proposition_2 ∧ proposition_3) := by
  sorry

end correct_propositions_l113_113753


namespace charity_ticket_revenue_l113_113562

noncomputable def full_price_ticket_revenue
  (f h p : ℕ) -- number of full-price tickets, number of half-price tickets, price of a full-price ticket
  (tickets_sold : f + h = 140)
  (total_revenue : f * p + h * (p / 2) = 2001) : ℕ :=
  f * p

theorem charity_ticket_revenue :
  ∃ (f h p : ℕ), f + h = 140 ∧ f * p + h * (p / 2) = 2001 ∧ f * p = 782 :=
begin
  sorry
end

end charity_ticket_revenue_l113_113562


namespace binomial_expansion_sum_zero_l113_113560

open BigOperators

theorem binomial_expansion_sum_zero (a b m n : ℕ) (k : ℕ) (h1 : n ≥ 2) (h2 : ab ≠ 0) (h3 : a = m * b) (h4 : m = k + 2) (h5 : ∀ (i : ℕ), i = 3 ∨ i = 4 → ∑ i in range n, binomial n i * b^(n-i) * a^i = 0) : n = 2 * m + 3 := 
sorry

end binomial_expansion_sum_zero_l113_113560


namespace smallest_positive_multiple_of_45_l113_113235

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l113_113235


namespace sin_neg_nine_pi_div_two_l113_113053

theorem sin_neg_nine_pi_div_two : Real.sin (-9 * Real.pi / 2) = -1 := by
  sorry

end sin_neg_nine_pi_div_two_l113_113053


namespace repetitive_decimals_subtraction_correct_l113_113749

noncomputable def repetitive_decimals_subtraction : Prop :=
  let a : ℚ := 4567 / 9999
  let b : ℚ := 1234 / 9999
  let c : ℚ := 2345 / 9999
  a - b - c = 988 / 9999

theorem repetitive_decimals_subtraction_correct : repetitive_decimals_subtraction :=
  by sorry

end repetitive_decimals_subtraction_correct_l113_113749


namespace prob_bashers_win_at_least_4_out_of_5_l113_113663

-- Define the probability p that the Bashers win a single game.
def p := 4 / 5

-- Define the number of games n.
def n := 5

-- Define the random trial outcome space.
def trials : Type := Fin n → Bool

-- Define the number of wins (true means a win, false means a loss).
def wins (t : trials) : ℕ := (Finset.univ.filter (λ i => t i = true)).card

-- Define winning exactly k games.
def win_exactly (t : trials) (k : ℕ) : Prop := wins t = k

-- Define the probability of winning exactly k games.
noncomputable def prob_win_exactly (k : ℕ) : ℚ :=
  (Nat.descFactorial n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the event of winning at least 4 out of 5 games.
def event_win_at_least (t : trials) := (wins t ≥ 4)

-- Define the probability of winning at least k out of n games.
noncomputable def prob_win_at_least (k : ℕ) : ℚ :=
  prob_win_exactly k + prob_win_exactly (k + 1)

-- Theorem to prove: Probability of winning at least 4 out of 5 games is 3072/3125.
theorem prob_bashers_win_at_least_4_out_of_5 :
  prob_win_at_least 4 = 3072 / 3125 :=
by
  sorry

end prob_bashers_win_at_least_4_out_of_5_l113_113663


namespace f_cos_x_l113_113630

theorem f_cos_x (f : ℝ → ℝ) (x : ℝ) (h : f (Real.sin x) = 2 - Real.cos x ^ 2) : f (Real.cos x) = 2 + Real.sin x ^ 2 := by
  sorry

end f_cos_x_l113_113630


namespace smallest_perfect_square_5336100_l113_113434

def smallestPerfectSquareDivisibleBy (a b c d : Nat) (s : Nat) : Prop :=
  ∃ k : Nat, s = k * k ∧ s % a = 0 ∧ s % b = 0 ∧ s % c = 0 ∧ s % d = 0

theorem smallest_perfect_square_5336100 :
  smallestPerfectSquareDivisibleBy 6 14 22 30 5336100 :=
sorry

end smallest_perfect_square_5336100_l113_113434


namespace distinct_real_numbers_inequality_l113_113018

theorem distinct_real_numbers_inequality
  (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  ( (2 * a - b) / (a - b) )^2 + ( (2 * b - c) / (b - c) )^2 + ( (2 * c - a) / (c - a) )^2 ≥ 5 :=
by {
    sorry
}

end distinct_real_numbers_inequality_l113_113018


namespace subset_of_positive_reals_l113_113378

def M := { x : ℝ | x > -1 }

theorem subset_of_positive_reals : {0} ⊆ M :=
by
  sorry

end subset_of_positive_reals_l113_113378


namespace original_denominator_l113_113572

theorem original_denominator (d : ℤ) : 
  (∀ n : ℤ, n = 3 → (n + 8) / (d + 8) = 1 / 3) → d = 25 :=
by
  intro h
  specialize h 3 rfl
  sorry

end original_denominator_l113_113572


namespace total_toys_l113_113991

theorem total_toys 
  (jaxon_toys : ℕ)
  (gabriel_toys : ℕ)
  (jerry_toys : ℕ)
  (h1 : jaxon_toys = 15)
  (h2 : gabriel_toys = 2 * jaxon_toys)
  (h3 : jerry_toys = gabriel_toys + 8) : 
  jaxon_toys + gabriel_toys + jerry_toys = 83 :=
  by sorry

end total_toys_l113_113991


namespace divisor_is_seventeen_l113_113544

theorem divisor_is_seventeen (D x : ℕ) (h1 : D = 7 * x) (h2 : D + x = 136) : x = 17 :=
by
  sorry

end divisor_is_seventeen_l113_113544


namespace field_area_l113_113892

theorem field_area
  (L : ℕ) (W : ℕ) (A : ℕ)
  (h₁ : L = 20)
  (h₂ : 2 * W + L = 100)
  (h₃ : A = L * W) :
  A = 800 := by
  sorry

end field_area_l113_113892


namespace ordered_triple_unique_l113_113013

theorem ordered_triple_unique (a b c : ℝ) (h2 : a > 2) (h3 : b > 2) (h4 : c > 2)
    (h : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 49) :
    a = 7 ∧ b = 5 ∧ c = 3 :=
sorry

end ordered_triple_unique_l113_113013


namespace maria_needs_nuts_l113_113020

theorem maria_needs_nuts (total_cookies nuts_per_cookie : ℕ) 
  (nuts_fraction : ℚ) (chocolate_fraction : ℚ) 
  (H1 : nuts_fraction = 1 / 4) 
  (H2 : chocolate_fraction = 0.4) 
  (H3 : total_cookies = 60) 
  (H4 : nuts_per_cookie = 2) :
  (total_cookies * nuts_fraction + (total_cookies - total_cookies * nuts_fraction - total_cookies * chocolate_fraction) * nuts_per_cookie) = 72 := 
by
  sorry

end maria_needs_nuts_l113_113020


namespace smallest_positive_multiple_of_45_l113_113265

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l113_113265


namespace arithmetic_sequence_75th_term_diff_l113_113896

noncomputable def sum_arith_sequence (n : ℕ) (a d : ℚ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_75th_term_diff {n : ℕ} {a d : ℚ}
  (hn : n = 150)
  (sum_seq : sum_arith_sequence n a d = 15000)
  (term_range : ∀ k, 0 ≤ k ∧ k < n → 20 ≤ a + k * d ∧ a + k * d ≤ 150)
  (t75th : ∃ L G, L = a + 74 * d ∧ G = a + 74 * d) :
  G - L = (7500 / 149) :=
sorry

end arithmetic_sequence_75th_term_diff_l113_113896


namespace sibling_discount_is_correct_l113_113428

-- Defining the given conditions
def tuition_per_person : ℕ := 45
def total_cost_with_discount : ℕ := 75

-- Defining the calculation of sibling discount
def sibling_discount : ℕ :=
  let original_cost := 2 * tuition_per_person
  let discount := original_cost - total_cost_with_discount
  discount

-- Statement to prove
theorem sibling_discount_is_correct : sibling_discount = 15 :=
by
  unfold sibling_discount
  simp
  sorry

end sibling_discount_is_correct_l113_113428


namespace orchid_bushes_planted_l113_113214

theorem orchid_bushes_planted (b1 b2 : ℕ) (h1 : b1 = 22) (h2 : b2 = 35) : b2 - b1 = 13 :=
by 
  sorry

end orchid_bushes_planted_l113_113214


namespace triangle_area_bounded_by_lines_l113_113073

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let base := 16
  let height := 8
  let area := base * height / 2
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l113_113073


namespace sin_minus_cos_sqrt_l113_113361

theorem sin_minus_cos_sqrt (θ : ℝ) (b : ℝ) (h₁ : 0 < θ ∧ θ < π / 2) (h₂ : Real.cos (2 * θ) = b) :
  Real.sin θ - Real.cos θ = Real.sqrt (1 - b) :=
sorry

end sin_minus_cos_sqrt_l113_113361


namespace race_head_start_l113_113714

/-- A's speed is 22/19 times that of B. If A and B run a race, A should give B a head start of (3 / 22) of the race length so the race ends in a dead heat. -/
theorem race_head_start {Va Vb L H : ℝ} (hVa : Va = (22 / 19) * Vb) (hL_Va : L / Va = (L - H) / Vb) : 
  H = (3 / 22) * L :=
by
  sorry

end race_head_start_l113_113714


namespace number_of_subjects_l113_113573

variable (P C M : ℝ)

-- Given conditions
def conditions (P C M : ℝ) : Prop :=
  (P + C + M) / 3 = 75 ∧
  (P + M) / 2 = 90 ∧
  (P + C) / 2 = 70 ∧
  P = 95

-- Proposition with given conditions and the conclusion
theorem number_of_subjects (P C M : ℝ) (h : conditions P C M) : 
  (∃ n : ℕ, n = 3) :=
by
  sorry

end number_of_subjects_l113_113573


namespace hexagon_coloring_l113_113132

def valid_coloring_hexagon : Prop :=
  ∃ (A B C D E F : Fin 8), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ A ≠ D ∧ B ≠ D ∧ C ≠ D ∧
    B ≠ E ∧ C ≠ E ∧ D ≠ E ∧ A ≠ F ∧ C ≠ F ∧ E ≠ F

theorem hexagon_coloring : ∃ (n : Nat), valid_coloring_hexagon ∧ n = 20160 := 
sorry

end hexagon_coloring_l113_113132


namespace smallest_positive_multiple_of_45_l113_113234

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l113_113234


namespace dice_tower_even_n_l113_113657

/-- Given that n standard dice are stacked in a vertical tower,
and the total visible dots on each of the four vertical walls are all odd,
prove that n must be even.
-/
theorem dice_tower_even_n (n : ℕ)
  (h : ∀ (S T : ℕ), (S + T = 7 * n → (S % 2 = 1 ∧ T % 2 = 1))) : n % 2 = 0 :=
by sorry

end dice_tower_even_n_l113_113657


namespace mod_17_residue_l113_113081

theorem mod_17_residue : (255 + 7 * 51 + 9 * 187 + 5 * 34) % 17 = 0 := 
  by sorry

end mod_17_residue_l113_113081


namespace triangle_area_is_64_l113_113066

noncomputable def area_triangle_lines (y : ℝ → ℝ) (x : ℝ → ℝ) (neg_x : ℝ → ℝ) : ℝ :=
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := 16
  let height := 8
  1 / 2 * base * height

theorem triangle_area_is_64 :
  let y := fun x => 8
  let x := fun y => y
  let neg_x := fun y => -y
  area_triangle_lines y x neg_x = 64 := 
by 
  sorry

end triangle_area_is_64_l113_113066


namespace flower_problem_l113_113388

def totalFlowers (n_rows n_per_row : Nat) : Nat :=
  n_rows * n_per_row

def flowersCut (total percent_cut : Nat) : Nat :=
  total * percent_cut / 100

def flowersRemaining (total cut : Nat) : Nat :=
  total - cut

theorem flower_problem :
  let n_rows := 50
  let n_per_row := 400
  let percent_cut := 60
  let total := totalFlowers n_rows n_per_row
  let cut := flowersCut total percent_cut
  flowersRemaining total cut = 8000 :=
by
  sorry

end flower_problem_l113_113388


namespace gcd_14568_78452_l113_113740

theorem gcd_14568_78452 : Nat.gcd 14568 78452 = 4 :=
sorry

end gcd_14568_78452_l113_113740


namespace find_XY_length_l113_113490

variables (a b c : ℝ) -- sides of triangle ABC
variables (s : ℝ) -- semi-perimeter s = (a + b + c) / 2

-- Definition of similar triangles and perimeter condition
noncomputable def XY_length
  (AX : ℝ) (XY : ℝ) (AY : ℝ) (BX : ℝ) 
  (BC : ℝ) (CY : ℝ) 
  (h1 : AX + AY + XY = BX + BC + CY)
  (h2 : AX = c * XY / a) 
  (h3 : AY = b * XY / a) : ℝ :=
  s * a / (b + c) -- by the given solution

-- The theorem statement
theorem find_XY_length
  (a b c : ℝ) (s : ℝ) -- given sides and semi-perimeter
  (AX : ℝ) (XY : ℝ) (AY : ℝ) (BX : ℝ)
  (BC : ℝ) (CY : ℝ) 
  (h1 : AX + AY + XY = BX + BC + CY)
  (h2 : AX = c * XY / a) 
  (h3 : AY = b * XY / a) :
  XY = s * a / (b + c) :=
sorry -- proof


end find_XY_length_l113_113490


namespace find_m_l113_113488

theorem find_m 
  (h : ∀ x, (0 < x ∧ x < 2) ↔ ( - (1 / 2) * x^2 + 2 * x > m * x )) :
  m = 1 :=
sorry

end find_m_l113_113488


namespace randy_initial_money_l113_113527

theorem randy_initial_money (X : ℕ) (h : X + 200 - 1200 = 2000) : X = 3000 :=
by {
  sorry
}

end randy_initial_money_l113_113527


namespace symmetric_scanning_codes_count_l113_113452

-- Definition of a symmetric 8x8 scanning code grid under given conditions
def is_symmetric_code (grid : Fin 8 → Fin 8 → Bool) : Prop :=
  ∀ i j : Fin 8, grid i j = grid (7 - i) (7 - j) ∧ grid i j = grid j i

def at_least_one_each_color (grid : Fin 8 → Fin 8 → Bool) : Prop :=
  ∃ i j k l : Fin 8, grid i j = true ∧ grid k l = false

def total_symmetric_scanning_codes : Nat :=
  1022

theorem symmetric_scanning_codes_count :
  ∀ (grid : Fin 8 → Fin 8 → Bool), is_symmetric_code grid ∧ at_least_one_each_color grid → 
  1022 = total_symmetric_scanning_codes :=
by
  sorry

end symmetric_scanning_codes_count_l113_113452


namespace blue_hat_cost_is_6_l113_113063

-- Total number of hats is 85
def total_hats : ℕ := 85

-- Number of green hats
def green_hats : ℕ := 20

-- Number of blue hats
def blue_hats : ℕ := total_hats - green_hats

-- Cost of each green hat
def cost_per_green_hat : ℕ := 7

-- Total cost for all hats
def total_cost : ℕ := 530

-- Total cost of green hats
def total_cost_green_hats : ℕ := green_hats * cost_per_green_hat

-- Total cost of blue hats
def total_cost_blue_hats : ℕ := total_cost - total_cost_green_hats

-- Cost per blue hat
def cost_per_blue_hat : ℕ := total_cost_blue_hats / blue_hats 

-- Prove that the cost of each blue hat is $6
theorem blue_hat_cost_is_6 : cost_per_blue_hat = 6 :=
by
  sorry

end blue_hat_cost_is_6_l113_113063


namespace S8_value_l113_113780

theorem S8_value (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 5 / 5 + S 11 / 11 = 12) (h2 : S 11 = S 8 + 1 / a 9 + 1 / a 10 + 1 / a 11) : S 8 = 48 :=
sorry

end S8_value_l113_113780


namespace integer_square_mod_4_l113_113654

theorem integer_square_mod_4 (N : ℤ) : (N^2 % 4 = 0) ∨ (N^2 % 4 = 1) :=
by sorry

end integer_square_mod_4_l113_113654


namespace Sam_scored_points_l113_113202

theorem Sam_scored_points (total_points friend_points S: ℕ) (h1: friend_points = 12) (h2: total_points = 87) (h3: total_points = S + friend_points) : S = 75 :=
by
  sorry

end Sam_scored_points_l113_113202


namespace problem_statement_l113_113137

theorem problem_statement (n m : ℕ) (hn : n ≠ 0) (hm : m ≠ 0) : 
  (n * 5^n)^n = m * 5^9 ↔ n = 3 ∧ m = 27 :=
by {
  sorry
}

end problem_statement_l113_113137


namespace triangle_area_bounded_by_lines_l113_113072

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let base := 16
  let height := 8
  let area := base * height / 2
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l113_113072


namespace xiao_li_estimate_l113_113438

variable (x y z : ℝ)

theorem xiao_li_estimate (h1 : x > y) (h2 : y > 0) (h3 : 0 < z):
    (x + z) + (y - z) = x + y := 
by 
sorry

end xiao_li_estimate_l113_113438


namespace motorist_gas_problem_l113_113724

noncomputable def original_price_per_gallon (P : ℝ) : Prop :=
  12 * P = 10 * (P + 0.30)

def fuel_efficiency := 25

def new_distance_travelled (P : ℝ) : ℝ :=
  10 * fuel_efficiency

theorem motorist_gas_problem :
  ∃ P : ℝ, original_price_per_gallon P ∧ P = 1.5 ∧ new_distance_travelled P = 250 :=
by
  use 1.5
  sorry

end motorist_gas_problem_l113_113724


namespace base_b_conversion_l113_113326

theorem base_b_conversion (b : ℝ) (h₁ : 1 * 5^2 + 3 * 5^1 + 2 * 5^0 = 42) (h₂ : 2 * b^2 + 2 * b + 1 = 42) :
  b = (-1 + Real.sqrt 83) / 2 := 
  sorry

end base_b_conversion_l113_113326


namespace age_of_son_l113_113554

theorem age_of_son (S M : ℕ) (h1 : M = S + 28) (h2 : M + 2 = 2 * (S + 2)) : S = 26 := by
  sorry

end age_of_son_l113_113554


namespace value_is_6_l113_113418

-- We know the conditions that the least number which needs an increment is 858
def least_number : ℕ := 858

-- Define the numbers 24, 32, 36, and 54
def num1 : ℕ := 24
def num2 : ℕ := 32
def num3 : ℕ := 36
def num4 : ℕ := 54

-- Define the LCM function to compute the least common multiple
def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

-- Define the LCM of the four numbers
def lcm_all : ℕ := lcm (lcm num1 num2) (lcm num3 num4)

-- Compute the value that needs to be added
def value_to_be_added : ℕ := lcm_all - least_number

-- Prove that this value equals to 6
theorem value_is_6 : value_to_be_added = 6 := by
  -- Proof would go here
  sorry

end value_is_6_l113_113418


namespace mechanism_completion_times_l113_113217

theorem mechanism_completion_times :
  ∃ (x y : ℝ), (1 / x + 1 / y = 1 / 30) ∧ (6 * (1 / x + 1 / y) + 40 * (1 / y) = 1) ∧ x = 75 ∧ y = 50 :=
by {
  sorry
}

end mechanism_completion_times_l113_113217


namespace dan_money_left_l113_113737

theorem dan_money_left
  (initial_amount : ℝ := 45)
  (cost_per_candy_bar : ℝ := 4)
  (num_candy_bars : ℕ := 4)
  (price_toy_car : ℝ := 15)
  (discount_rate_toy_car : ℝ := 0.10)
  (sales_tax_rate : ℝ := 0.05) :
  initial_amount - ((num_candy_bars * cost_per_candy_bar) + ((price_toy_car - (price_toy_car * discount_rate_toy_car)) * (1 + sales_tax_rate))) = 14.02 :=
by
  sorry

end dan_money_left_l113_113737


namespace propA_necessary_but_not_sufficient_l113_113619

variable {a : ℝ}

-- Proposition A: ∀ x ∈ ℝ, ax² + 2ax + 1 > 0
def propA (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0

-- Proposition B: 0 < a < 1
def propB (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Theorem statement: Proposition A is necessary but not sufficient for Proposition B
theorem propA_necessary_but_not_sufficient (a : ℝ) :
  (propB a → propA a) ∧
  (propA a → propB a → False) :=
by
  sorry

end propA_necessary_but_not_sufficient_l113_113619


namespace insurance_compensation_zero_l113_113670

noncomputable def insured_amount : ℝ := 500000
noncomputable def deductible : ℝ := 0.01
noncomputable def actual_damage : ℝ := 4000

theorem insurance_compensation_zero :
  actual_damage < insured_amount * deductible → 0 = 0 := by
sorry

end insurance_compensation_zero_l113_113670


namespace domain_log_function_l113_113669

open Real

def quadratic_term (x : ℝ) : ℝ := 4 - 3 * x - x^2

def valid_argument (x : ℝ) : Prop := quadratic_term x > 0

theorem domain_log_function : { x : ℝ | valid_argument x } = Set.Ioo (-4 : ℝ) (1 : ℝ) :=
by
  sorry

end domain_log_function_l113_113669


namespace product_of_b_values_is_neg_12_l113_113038

theorem product_of_b_values_is_neg_12 (b : ℝ) (y1 y2 x1 : ℝ) (h1 : y1 = 3) (h2 : y2 = 7) (h3 : x1 = 2) (h4 : y2 - y1 = 4) (h5 : ∃ b1 b2, b1 = x1 - 4 ∧ b2 = x1 + 4) : 
  (b1 * b2 = -12) :=
by
  sorry

end product_of_b_values_is_neg_12_l113_113038


namespace smallest_positive_multiple_45_l113_113251

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l113_113251


namespace width_of_rectangle_l113_113450

theorem width_of_rectangle (w l : ℝ) (h1 : l = 2 * w) (h2 : l * w = 1) : w = Real.sqrt 2 / 2 :=
sorry

end width_of_rectangle_l113_113450


namespace lemonade_percentage_correct_l113_113309
noncomputable def lemonade_percentage (first_lemonade first_carbon second_carbon mixture_carbon first_portion : ℝ) : ℝ :=
  100 - second_carbon

theorem lemonade_percentage_correct :
  let first_lemonade := 20
  let first_carbon := 80
  let second_carbon := 55
  let mixture_carbon := 60
  let first_portion := 19.99999999999997
  lemonade_percentage first_lemonade first_carbon second_carbon mixture_carbon first_portion = 45 :=
by
  -- Proof to be completed.
  sorry

end lemonade_percentage_correct_l113_113309


namespace Rebecca_tent_stakes_l113_113528

theorem Rebecca_tent_stakes : 
  ∃ T D W : ℕ, 
    D = 3 * T ∧ 
    W = T + 2 ∧ 
    T + D + W = 22 ∧ 
    T = 4 := 
by
  sorry

end Rebecca_tent_stakes_l113_113528


namespace john_computers_fixed_count_l113_113373

-- Define the problem conditions.
variables (C : ℕ)
variables (unfixable_ratio spare_part_ratio fixable_ratio : ℝ)
variables (fixed_right_away : ℕ)
variables (h1 : unfixable_ratio = 0.20)
variables (h2 : spare_part_ratio = 0.40)
variables (h3 : fixable_ratio = 0.40)
variables (h4 : fixed_right_away = 8)
variables (h5 : fixable_ratio * ↑C = fixed_right_away)

-- The theorem to prove.
theorem john_computers_fixed_count (h1 : C > 0) : C = 20 := by
  sorry

end john_computers_fixed_count_l113_113373


namespace vector_c_expression_l113_113492

-- Define the vectors a, b, c
def vector_a : ℤ × ℤ := (1, 2)
def vector_b : ℤ × ℤ := (-1, 1)
def vector_c : ℤ × ℤ := (1, 5)

-- Define the addition of vectors in ℤ × ℤ
def vec_add (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define the scalar multiplication of vectors in ℤ × ℤ
def scalar_mul (k : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (k * v.1, k * v.2)

-- Given the conditions
def condition1 := vector_a = (1, 2)
def condition2 := vec_add vector_a vector_b = (0, 3)

-- The goal is to prove that vector_c = 2 * vector_a + vector_b
theorem vector_c_expression : vec_add (scalar_mul 2 vector_a) vector_b = vector_c := by
  sorry

end vector_c_expression_l113_113492


namespace area_of_figure_l113_113624

theorem area_of_figure : 
  let S := { p : ℝ × ℝ | (|p.1| + p.1)^2 + (|p.2| - p.2)^2 ≤ 16 ∧ p.2 - 3 * p.1 ≤ 0 } in
  measure_theory.measure.inter_volume S = (20/3 + real.pi) :=
sorry

end area_of_figure_l113_113624


namespace smallest_positive_multiple_45_l113_113249

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l113_113249


namespace boat_speed_in_still_water_l113_113864

/--
The speed of the stream is 6 kmph.
The boat can cover 48 km downstream or 32 km upstream in the same time.
We want to prove that the speed of the boat in still water is 30 kmph.
-/
theorem boat_speed_in_still_water (x : ℝ)
  (h1 : ∃ t : ℝ, t = 48 / (x + 6) ∧ t = 32 / (x - 6)) : x = 30 :=
by
  sorry

end boat_speed_in_still_water_l113_113864


namespace arithmetic_sequence_example_l113_113634

theorem arithmetic_sequence_example (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h2 : a 2 = 2) (h14 : a 14 = 18) : a 8 = 10 :=
by
  sorry

end arithmetic_sequence_example_l113_113634


namespace inverse_of_A_l113_113334

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  matrix.of ![![3, 4], ![-2, 9]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  matrix.of ![![9/35, -4/35], ![2/35, 3/35]]

theorem inverse_of_A :
  A⁻¹ = A_inv := by
  sorry

end inverse_of_A_l113_113334


namespace expected_value_binomial_l113_113484

theorem expected_value_binomial :
  (∃ ξ : ℕ → ℝ, (∀ n p. ξ n p = n * p) → ξ 6 (1/3) = 2) :=
by
  sorry

end expected_value_binomial_l113_113484


namespace angle_C_max_perimeter_l113_113785

def triangle_ABC (A B C a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 

def circumradius_2 (r : ℝ) : Prop :=
  r = 2

def satisfies_condition (a b c A B C : ℝ) : Prop :=
  (a - c)*(Real.sin A + Real.sin C) = b*(Real.sin A - Real.sin B)

theorem angle_C (A B C a b c : ℝ) (h₁ : triangle_ABC A B C a b c) 
                 (h₂ : satisfies_condition a b c A B C)
                 (h₃ : circumradius_2 (2 : ℝ)) : 
  C = Real.pi / 3 :=
sorry

theorem max_perimeter (A B C a b c r : ℝ) (h₁ : triangle_ABC A B C a b c)
                      (h₂ : satisfies_condition a b c A B C)
                      (h₃ : circumradius_2 r) : 
  4 * Real.sqrt 3 + 2 * Real.sqrt 3 = 6 * Real.sqrt 3 :=
sorry

end angle_C_max_perimeter_l113_113785


namespace inequality_abc_l113_113618

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / (a * b * c) + 1 ≥ 3 * (1 / (a^2 + b^2 + c^2) + 1 / (a + b + c)) :=
by
  sorry

end inequality_abc_l113_113618


namespace reciprocal_of_neg6_l113_113696

theorem reciprocal_of_neg6 : 1 / (-6 : ℝ) = -1 / 6 := 
sorry

end reciprocal_of_neg6_l113_113696


namespace smallest_positive_multiple_of_45_l113_113259

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l113_113259


namespace milk_production_days_l113_113495

variable (x : ℕ)
def cows := 2 * x
def cans := 2 * x + 2
def days := 2 * x + 1
def total_cows := 2 * x + 4
def required_cans := 2 * x + 10

theorem milk_production_days :
  (total_cows * required_cans) = ((2 * x) * (2 * x + 1) * required_cans) / ((2 * x + 2) * (2 * x + 4)) :=
sorry

end milk_production_days_l113_113495


namespace find_M_l113_113628

theorem find_M :
  (∃ (M : ℕ), (5 + 7 + 9) / 3 = (4020 + 4021 + 4022) / M) → M = 1723 :=
  by
  sorry

end find_M_l113_113628


namespace minimum_value_expression_l113_113350

theorem minimum_value_expression {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) : 
  (x / y + y) * (y / x + x) ≥ 4 :=
sorry

end minimum_value_expression_l113_113350


namespace charity_tickets_l113_113563

theorem charity_tickets (f h p : ℕ) (H1 : f + h = 140) (H2 : f * p + h * (p / 2) = 2001) : f * p = 782 := 
sorry

end charity_tickets_l113_113563


namespace range_of_a_l113_113396

def condition1 (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def condition2 (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (3 - 2 * a)^x < (3 - 2 * a)^y

def exclusive_or (p q : Prop) : Prop :=
  (p ∧ ¬q) ∨ (¬p ∧ q)

theorem range_of_a (a : ℝ) :
  exclusive_or (condition1 a) (condition2 a) → (1 ≤ a ∧ a < 2) ∨ a ≤ -2 :=
by
  -- Proof omitted
  sorry

end range_of_a_l113_113396


namespace unique_solution_l113_113136

theorem unique_solution :
  ∀ (x y z n : ℕ), n ≥ 2 → z ≤ 5 * 2^(2 * n) → (x^ (2 * n + 1) - y^ (2 * n + 1) = x * y * z + 2^(2 * n + 1)) → (x, y, z, n) = (3, 1, 70, 2) :=
by
  intros x y z n hn hzn hxyz
  sorry

end unique_solution_l113_113136


namespace alice_cell_phone_cost_l113_113175

theorem alice_cell_phone_cost
  (base_cost : ℕ)
  (included_hours : ℕ)
  (text_cost_per_message : ℕ)
  (extra_minute_cost : ℕ)
  (messages_sent : ℕ)
  (hours_spent : ℕ) :
  base_cost = 25 →
  included_hours = 40 →
  text_cost_per_message = 4 →
  extra_minute_cost = 5 →
  messages_sent = 150 →
  hours_spent = 42 →
  (base_cost + (messages_sent * text_cost_per_message) / 100 + ((hours_spent - included_hours) * 60 * extra_minute_cost) / 100) = 37 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end alice_cell_phone_cost_l113_113175


namespace smallest_a_for_5880_to_be_cube_l113_113139

theorem smallest_a_for_5880_to_be_cube : ∃ (a : ℕ), a > 0 ∧ (∃ (k : ℕ), 5880 * a = k ^ 3) ∧
  (∀ (b : ℕ), b > 0 ∧ (∃ (k : ℕ), 5880 * b = k ^ 3) → a ≤ b) ∧ a = 1575 :=
sorry

end smallest_a_for_5880_to_be_cube_l113_113139


namespace solve_eq1_solve_eq2_l113_113802

-- Proof problem 1: Prove that under the condition 6x - 4 = 3x + 2, x = 2
theorem solve_eq1 : ∀ x : ℝ, 6 * x - 4 = 3 * x + 2 → x = 2 :=
by
  intro x
  intro h
  sorry

-- Proof problem 2: Prove that under the condition (x / 4) - (3 / 5) = (x + 1) / 2, x = -22/5
theorem solve_eq2 : ∀ x : ℝ, (x / 4) - (3 / 5) = (x + 1) / 2 → x = -(22 / 5) :=
by
  intro x
  intro h
  sorry

end solve_eq1_solve_eq2_l113_113802


namespace total_toys_l113_113990

theorem total_toys 
  (jaxon_toys : ℕ)
  (gabriel_toys : ℕ)
  (jerry_toys : ℕ)
  (h1 : jaxon_toys = 15)
  (h2 : gabriel_toys = 2 * jaxon_toys)
  (h3 : jerry_toys = gabriel_toys + 8) : 
  jaxon_toys + gabriel_toys + jerry_toys = 83 :=
  by sorry

end total_toys_l113_113990


namespace correct_divisor_l113_113369

variable (D X : ℕ)

-- Conditions
def condition1 : Prop := X = D * 24
def condition2 : Prop := X = (D - 12) * 42

theorem correct_divisor (D X : ℕ) (h1 : condition1 D X) (h2 : condition2 D X) : D = 28 := by
  sorry

end correct_divisor_l113_113369


namespace areas_of_isosceles_triangles_l113_113199

theorem areas_of_isosceles_triangles (A B C : ℝ) (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13)
  (hA : A = 1/2 * a * a) (hB : B = 1/2 * b * b) (hC : C = 1/2 * c * c) :
  A + B = C :=
by
  sorry

end areas_of_isosceles_triangles_l113_113199


namespace marilyn_total_caps_l113_113520

def marilyn_initial_caps : ℝ := 51.0
def nancy_gives_caps : ℝ := 36.0
def total_caps (initial: ℝ) (given: ℝ) : ℝ := initial + given

theorem marilyn_total_caps : total_caps marilyn_initial_caps nancy_gives_caps = 87.0 :=
by
  sorry

end marilyn_total_caps_l113_113520


namespace insuranceCompensationIsZero_l113_113671

-- Define the insurance problem parameters
def insuredAmount : ℕ := 500000
def deductibleRate : ℚ := 1 / 100
def actualDamage : ℕ := 4000

-- Define what the threshold value for the deductible is
def thresholdValue : ℚ := insuredAmount * deductibleRate

def insuranceCompensation (insuredAmount : ℕ) (deductibleRate : ℚ) (actualDamage : ℕ) : ℕ := 
  if actualDamage < thresholdValue.to_nat then 0 else sorry -- Use sorry to handle the else part.

-- Prove that given the conditions, the insurance compensation amount is 0 rubles.
theorem insuranceCompensationIsZero :
  insuranceCompensation insuredAmount deductibleRate actualDamage = 0 := by
  -- Since the actual damage (4000) is less than the threshold (5000), the deduction applies.
  have h_threshold : actualDamage < thresholdValue.to_nat := by 
    calc
      actualDamage = 4000 : rfl
      ... < 5000 : by norm_num
  -- Thus, the compensation amount should be zero.
  rw [insuranceCompensation]
  simp [h_threshold]
  exact rfl

end insuranceCompensationIsZero_l113_113671


namespace smallest_positive_multiple_of_45_is_45_l113_113237

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113237


namespace smallest_positive_multiple_45_l113_113242

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l113_113242


namespace not_snow_probability_l113_113682

theorem not_snow_probability :
  let p_snow : ℚ := 2 / 3,
      p_not_snow : ℚ := 1 - p_snow in
  (p_not_snow ^ 5) = 1 / 243 := by
  sorry

end not_snow_probability_l113_113682


namespace find_angle_A_find_perimeter_l113_113174

noncomputable def cos_rule (b c a : ℝ) (h : b^2 + c^2 - a^2 = b * c) : ℝ :=
(b^2 + c^2 - a^2) / (2 * b * c)

theorem find_angle_A (A B C : ℝ) (a b c : ℝ)
  (h1 : b^2 + c^2 - a^2 = b * c) (hA : cos_rule b c a h1 = 1 / 2) :
  A = Real.arccos (1 / 2) :=
by sorry

theorem find_perimeter (a b c : ℝ)
  (h_a : a = Real.sqrt 2) (hA : Real.sin (Real.arccos (1 / 2))^2 = (Real.sqrt 3 / 2)^2)
  (hBC : Real.sin (Real.arccos (1 / 2))^2 = Real.sin (Real.arccos (1 / 2)) * Real.sin (Real.arccos (1 / 2)))
  (h_bc : b * c = 2)
  (h_bc_eq : b^2 + c^2 - a^2 = b * c) :
  a + b + c = 3 * Real.sqrt 2 :=
by sorry

end find_angle_A_find_perimeter_l113_113174


namespace find_const_s_l113_113381

noncomputable def g (x : ℝ) (a b c d : ℝ) := (x + 2 * a) * (x + 2 * b) * (x + 2 * c) * (x + 2 * d)

theorem find_const_s (a b c d : ℝ) (p q r s : ℝ) (h1 : 1 + p + q + r + s = 4041)
  (h2 : g 1 a b c d = 1 + p + q + r + s) :
  s = 3584 := 
sorry

end find_const_s_l113_113381


namespace percent_germinated_is_31_l113_113340

-- Define given conditions
def seeds_first_plot : ℕ := 300
def seeds_second_plot : ℕ := 200
def germination_rate_first_plot : ℝ := 0.25
def germination_rate_second_plot : ℝ := 0.40

-- Calculate the number of germinated seeds in each plot
def germinated_first_plot : ℝ := germination_rate_first_plot * seeds_first_plot
def germinated_second_plot : ℝ := germination_rate_second_plot * seeds_second_plot

-- Calculate total number of seeds and total number of germinated seeds
def total_seeds : ℕ := seeds_first_plot + seeds_second_plot
def total_germinated : ℝ := germinated_first_plot + germinated_second_plot

-- Prove the percentage of the total number of seeds that germinated
theorem percent_germinated_is_31 :
  ((total_germinated / total_seeds) * 100) = 31 := 
by
  sorry

end percent_germinated_is_31_l113_113340


namespace surprise_shop_daily_revenue_l113_113742

def closed_days_per_year : ℕ := 3
def years_active : ℕ := 6
def total_revenue_lost : ℚ := 90000

def total_closed_days : ℕ :=
  closed_days_per_year * years_active

def daily_revenue : ℚ :=
  total_revenue_lost / total_closed_days

theorem surprise_shop_daily_revenue :
  daily_revenue = 5000 := by
  sorry

end surprise_shop_daily_revenue_l113_113742


namespace percentage_increase_first_year_l113_113302

theorem percentage_increase_first_year (P : ℝ) (x : ℝ) :
  (1 + x / 100) * 0.7 = 1.0499999999999998 → x = 50 := 
by
  sorry

end percentage_increase_first_year_l113_113302


namespace three_digit_sum_of_factorials_is_145_l113_113440

theorem three_digit_sum_of_factorials_is_145 :
  ∀ (x y z : ℕ),
    x < 10 ∧ y < 10 ∧ z < 10 ∧ 100x + 10y + z = x! + y! + z! →
    100x + 10y + z = 145 :=
by
  sorry

end three_digit_sum_of_factorials_is_145_l113_113440


namespace total_distance_covered_l113_113084

theorem total_distance_covered :
  ∀ (r j w total : ℝ),
    r = 40 →
    j = (3 / 5) * r →
    w = 5 * j →
    total = r + j + w →
    total = 184 := by
  sorry

end total_distance_covered_l113_113084


namespace time_difference_l113_113625

-- Definitions for the conditions
def blocks_to_office : Nat := 12
def walk_time_per_block : Nat := 1 -- time in minutes
def bike_time_per_block : Nat := 20 / 60 -- time in minutes, converted from seconds

-- Definitions for the total times
def walk_time : Nat := blocks_to_office * walk_time_per_block
def bike_time : Nat := blocks_to_office * bike_time_per_block

-- Theorem statement
theorem time_difference : walk_time - bike_time = 8 := by
  -- Proof omitted
  sorry

end time_difference_l113_113625


namespace find_f_2_l113_113621

def f (x : ℕ) : ℤ := sorry

axiom func_def : ∀ x : ℕ, f (x + 1) = x^2 - x

theorem find_f_2 : f 2 = 0 :=
by
  sorry

end find_f_2_l113_113621


namespace num_x_for_3001_in_sequence_l113_113922

theorem num_x_for_3001_in_sequence :
  let seq : ℕ → ℝ := λ n, if n = 0 then x else if n = 1 then 3000 else seq (n - 2) * seq (n - 1) - 1
  let appears (a : ℝ) : Prop := ∃ n : ℕ, seq n = a
  ∃ (x : ℝ), appears seq 3001 :=
      sorry

  ∃ (x : ℝ→ ℕ) (hx : (∑ x. appear 3001) = 4 :=  ∑ sorry

end num_x_for_3001_in_sequence_l113_113922


namespace average_speed_x_to_z_l113_113524

theorem average_speed_x_to_z 
  (d : ℝ)
  (h1 : d > 0)
  (distance_xy : ℝ := 2 * d)
  (distance_yz : ℝ := d)
  (speed_xy : ℝ := 100)
  (speed_yz : ℝ := 75)
  (total_distance : ℝ := distance_xy + distance_yz)
  (time_xy : ℝ := distance_xy / speed_xy)
  (time_yz : ℝ := distance_yz / speed_yz)
  (total_time : ℝ := time_xy + time_yz) :
  total_distance / total_time = 90 :=
by
  sorry

end average_speed_x_to_z_l113_113524


namespace square_perimeter_is_64_l113_113453

-- Given conditions
variables (s : ℕ)
def is_square_divided_into_four_congruent_rectangles : Prop :=
  ∀ (r : ℕ), r = 4 → (∀ (p : ℕ), p = (5 * s) / 2 → p = 40)

-- Lean 4 statement for the proof problem
theorem square_perimeter_is_64 
  (h : is_square_divided_into_four_congruent_rectangles s) 
  (hs : (5 * s) / 2 = 40) : 
  4 * s = 64 :=
by
  sorry

end square_perimeter_is_64_l113_113453


namespace odd_four_digit_strictly_decreasing_count_l113_113494

theorem odd_four_digit_strictly_decreasing_count : 
  ∃ count : ℕ, count = 105 ∧ 
  count = (finset.univ.filter (λ x : finset (fin (10)), ∃ (d1 d2 d3 d4 : ℕ),
    d1 ∈ finset.range 10 ∧ d2 ∈ finset.range 10 ∧ d3 ∈ finset.range 10 ∧ d4 ∈ finset.range 10 ∧
    d1 > d2 ∧ d2 > d3 ∧ d3 > d4 ∧ d4 % 2 = 1 ∧ 
    1000 * d1 + 100 * d2 + 10 * d3 + d4 < 10000 ∧ 
    1000 * d1 + 100 * d2 + 10 * d3 + d4 ≥ 1000)).card :=
begin
  -- Proof is skipped
  sorry
end

end odd_four_digit_strictly_decreasing_count_l113_113494


namespace pats_password_length_l113_113191

/-- Pat’s computer password is made up of several kinds of alphanumeric and symbol characters for security.
  He uses:
  1. A string of eight random lowercase letters.
  2. A string half that length of alternating upper case letters and numbers.
  3. One symbol on each end of the password.

  Prove that the total number of characters in Pat's computer password is 14.
-/ 
theorem pats_password_length : 
  let lowercase_len := 8 in
  let alternating_len := lowercase_len / 2 in
  let symbols := 2 in
  lowercase_len + alternating_len + symbols = 14 := 
by 
  -- definitions
  let lowercase_len : Nat := 8
  let alternating_len : Nat := lowercase_len / 2
  let symbols : Nat := 2
  
  -- calculation
  have total_length := lowercase_len + alternating_len + symbols
  
  -- assertion
  show total_length = 14 from sorry

end pats_password_length_l113_113191


namespace smallest_positive_multiple_of_45_l113_113264

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l113_113264


namespace flip_ratio_l113_113504

theorem flip_ratio (jen_triple_flips tyler_double_flips : ℕ)
  (hjen : jen_triple_flips = 16)
  (htyler : tyler_double_flips = 12)
  : 2 * tyler_double_flips / 3 * jen_triple_flips = 1 / 2 := 
by
  rw [hjen, htyler]
  norm_num
  sorry

end flip_ratio_l113_113504


namespace donut_selection_count_l113_113650

theorem donut_selection_count : (∃ (g c p : ℕ), g + c + p = 5) ↔ finset.card (finset.filter (λ (k : fin 8), k.val ≤ 5) (finset.Ico 0 8)) = 21 := 
by
  sorry

end donut_selection_count_l113_113650


namespace quadratic_function_min_value_l113_113638

noncomputable def f (a h k : ℝ) (x : ℝ) : ℝ :=
  a * (x - h) ^ 2 + k

theorem quadratic_function_min_value :
  ∀ (f : ℝ → ℝ) (n : ℕ),
  (f n = 13) ∧ (f (n + 1) = 13) ∧ (f (n + 2) = 35) →
  (∃ k, k = 2) :=
  sorry

end quadratic_function_min_value_l113_113638


namespace proportion_equation_l113_113166

theorem proportion_equation (x y : ℝ) (h : 3 * x = 4 * y) (hy : y ≠ 0) : (x / 4 = y / 3) :=
by
  sorry

end proportion_equation_l113_113166


namespace sandy_carrots_l113_113656

-- Definitions and conditions
def total_carrots : ℕ := 14
def mary_carrots : ℕ := 6

-- Proof statement
theorem sandy_carrots : (total_carrots - mary_carrots) = 8 :=
by
  -- sorry is used to bypass the actual proof steps
  sorry

end sandy_carrots_l113_113656


namespace negative_half_less_than_negative_third_l113_113907

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end negative_half_less_than_negative_third_l113_113907


namespace ages_proof_l113_113976

def hans_now : ℕ := 8

def sum_ages (annika_now emil_now frida_now : ℕ) :=
  hans_now + annika_now + emil_now + frida_now = 58

def annika_age_in_4_years (annika_now : ℕ) : ℕ :=
  3 * (hans_now + 4)

def emil_age_in_4_years (emil_now : ℕ) : ℕ :=
  2 * (hans_now + 4)

def frida_age_in_4_years (frida_now : ℕ) :=
  2 * 12

def annika_frida_age_difference (annika_now frida_now : ℕ) : Prop :=
  annika_now = frida_now + 5

theorem ages_proof :
  ∃ (annika_now emil_now frida_now : ℕ),
    sum_ages annika_now emil_now frida_now ∧
    annika_age_in_4_years annika_now = 36 ∧
    emil_age_in_4_years emil_now = 24 ∧
    frida_age_in_4_years frida_now = 24 ∧
    annika_frida_age_difference annika_now frida_now :=
by
  sorry

end ages_proof_l113_113976


namespace sequence_x_values_3001_l113_113926

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end sequence_x_values_3001_l113_113926


namespace circumradius_eq_l113_113643

noncomputable def circumradius (r : ℂ) (t1 t2 t3 : ℂ) : ℂ :=
  (2 * r ^ 4) / Complex.abs ((t1 + t2) * (t2 + t3) * (t3 + t1))

theorem circumradius_eq (r t1 t2 t3 : ℂ) (h_pos_r : r ≠ 0) :
  circumradius r t1 t2 t3 = (2 * r ^ 4) / Complex.abs ((t1 + t2) * (t2 + t3) * (t3 + t1)) :=
  by sorry

end circumradius_eq_l113_113643


namespace crayons_difference_l113_113394

noncomputable def initial_crayons : ℕ := 250
noncomputable def gave_crayons : ℕ := 150
noncomputable def lost_crayons : ℕ := 512
noncomputable def broke_crayons : ℕ := 75
noncomputable def traded_crayons : ℕ := 35

theorem crayons_difference :
  lost_crayons - (gave_crayons + broke_crayons + traded_crayons) = 252 := by
  sorry

end crayons_difference_l113_113394


namespace train_platform_length_l113_113575

noncomputable def kmph_to_mps (v : ℕ) : ℕ := v * 1000 / 3600

theorem train_platform_length :
  ∀ (train_length speed_kmph time_sec : ℕ),
    speed_kmph = 36 →
    train_length = 175 →
    time_sec = 40 →
    let speed_mps := kmph_to_mps speed_kmph
    let total_distance := speed_mps * time_sec
    let platform_length := total_distance - train_length
    platform_length = 225 :=
by
  intros train_length speed_kmph time_sec h_speed h_train h_time
  let speed_mps := kmph_to_mps speed_kmph
  let total_distance := speed_mps * time_sec
  let platform_length := total_distance - train_length
  sorry

end train_platform_length_l113_113575


namespace john_saves_7680_per_year_l113_113507

-- Define old rent cost
def oldRent := 1200

-- Define the increase percentage for new apartment costs
def increasePercent := 0.40

-- Define the new rent cost
def newRent := oldRent + oldRent * increasePercent

-- Define the number of people sharing the cost
def numPeople := 3

-- Define John's share of the new apartment cost
def johnsShare := newRent / numPeople

-- Define monthly savings
def monthlySavings := oldRent - johnsShare

-- Define yearly savings
def yearlySavings := monthlySavings * 12

-- Theorem statement
theorem john_saves_7680_per_year : yearlySavings = 7680 := 
by {
  -- Placeholder for actual proof,
  -- but statement builds successfully.
  sorry
}

end john_saves_7680_per_year_l113_113507


namespace green_ball_probability_l113_113126

-- Defining the number of red and green balls in each container
def containerI_red : ℕ := 10
def containerI_green : ℕ := 5

def containerII_red : ℕ := 3
def containerII_green : ℕ := 6

def containerIII_red : ℕ := 3
def containerIII_green : ℕ := 6

-- Probability of selecting any container
def prob_container : ℚ := 1 / 3

-- Defining the probabilities of drawing a green ball from each container
def prob_green_I : ℚ := containerI_green / (containerI_red + containerI_green)
def prob_green_II : ℚ := containerII_green / (containerII_red + containerII_green)
def prob_green_III : ℚ := containerIII_green / (containerIII_red + containerIII_green)

-- Law of total probability
def prob_green_total : ℚ :=
  prob_container * prob_green_I +
  prob_container * prob_green_II +
  prob_container * prob_green_III

-- The mathematical statement to be proven
theorem green_ball_probability :
  prob_green_total = 5 / 9 := by
  sorry

end green_ball_probability_l113_113126


namespace time_spent_on_each_piece_l113_113398

def chairs : Nat := 7
def tables : Nat := 3
def total_time : Nat := 40
def total_pieces := chairs + tables
def time_per_piece := total_time / total_pieces

theorem time_spent_on_each_piece : time_per_piece = 4 :=
by
  sorry

end time_spent_on_each_piece_l113_113398


namespace initial_tanks_hold_fifteen_fish_l113_113988

theorem initial_tanks_hold_fifteen_fish (t : Nat) (additional_tanks : Nat) (fish_per_additional_tank : Nat) (total_fish : Nat) :
  t = 3 ∧ additional_tanks = 3 ∧ fish_per_additional_tank = 10 ∧ total_fish = 75 → 
  ∀ (F : Nat), (F * t) = 45 → F = 15 :=
by
  sorry

end initial_tanks_hold_fifteen_fish_l113_113988


namespace sum_last_two_digits_l113_113289

theorem sum_last_two_digits (a b m n : ℕ) (h7 : a = 7) (h13 : b = 13) (h100 : m = 100) (h30 : n = 30) : 
 ((a ^ n) + (b ^ n)) % m = 98 :=
by
  have h₁ : 7 ^ 30 % 100 = (49 : ℕ) := by sorry
  have h₂ : 13 ^ 30 % 100 = 49 := by sorry
  calc
    (7 ^ 30 + 13 ^ 30) % 100
      = (49 + 49) % 100 : by { rw [h₁, h₂] }
  ... = 98 % 100 : by rfl
  ... = 98 : by rfl

end sum_last_two_digits_l113_113289


namespace onions_left_on_shelf_l113_113545

def initial_onions : ℕ := 98
def sold_onions : ℕ := 65
def remaining_onions : ℕ := initial_onions - sold_onions

theorem onions_left_on_shelf : remaining_onions = 33 :=
by 
  -- Proof would go here
  sorry

end onions_left_on_shelf_l113_113545


namespace curve_max_value_ratio_l113_113762

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem curve_max_value_ratio (a b : ℝ) 
  (h1 : f a b 1 = 10) 
  (h2 : deriv (f a b) 1 = 0) : 
  a / b = -2/3 := 
by 
  sorry

end curve_max_value_ratio_l113_113762


namespace solve_system_of_equations_l113_113622

theorem solve_system_of_equations (x y m : ℝ) 
  (h1 : 2 * x + y = 7)
  (h2 : x + 2 * y = m - 3) 
  (h3 : x - y = 2) : m = 8 :=
by
  -- Proof part is replaced with sorry as mentioned
  sorry

end solve_system_of_equations_l113_113622


namespace vector_linear_combination_l113_113468

open Matrix

theorem vector_linear_combination :
  let v1 := ![3, -9]
  let v2 := ![2, -8]
  let v3 := ![1, -6]
  4 • v1 - 3 • v2 + 2 • v3 = ![8, -24] :=
by sorry

end vector_linear_combination_l113_113468


namespace bob_cookie_price_same_as_jane_l113_113141

theorem bob_cookie_price_same_as_jane
  (r_jane : ℝ)
  (s_bob : ℝ)
  (dough_jane : ℝ)
  (num_jane_cookies : ℕ)
  (price_jane_cookie : ℝ)
  (total_earning_jane : ℝ)
  (num_cookies_bob : ℝ)
  (price_bob_cookie : ℝ) :
  r_jane = 4 ∧
  s_bob = 6 ∧
  dough_jane = 18 * (Real.pi * r_jane^2) ∧
  price_jane_cookie = 0.50 ∧
  total_earning_jane = 18 * 50 ∧
  num_cookies_bob = dough_jane / s_bob^2 ∧
  total_earning_jane = num_cookies_bob * price_bob_cookie →
  price_bob_cookie = 36 :=
by
  intros
  sorry

end bob_cookie_price_same_as_jane_l113_113141


namespace max_m_x_range_l113_113144

variables {a b x : ℝ}

theorem max_m (h1 : a * b > 0) (h2 : a^2 * b = 4) : 
  a + b ≥ 3 :=
sorry

theorem x_range (h : 2 * |x - 1| + |x| ≤ 3) : 
  -1/3 ≤ x ∧ x ≤ 5/3 :=
sorry

end max_m_x_range_l113_113144


namespace no_six_digit_starting_with_five_12_digit_square_six_digit_starting_with_one_12_digit_square_smallest_k_for_n_digit_number_square_l113_113295

-- Part (a)
theorem no_six_digit_starting_with_five_12_digit_square : ∀ (x y : ℕ), (5 * 10^5 ≤ x) → (x < 6 * 10^5) → (10^5 ≤ y) → (y < 10^6) → ¬∃ z : ℕ, (10^11 ≤ z) ∧ (z < 10^12) ∧ x * 10^6 + y = z^2 := sorry

-- Part (b)
theorem six_digit_starting_with_one_12_digit_square : ∀ (x y : ℕ), (10^5 ≤ x) → (x < 2 * 10^5) → (10^5 ≤ y) → (y < 10^6) → ∃ z : ℕ, (10^11 ≤ z) ∧ (z < 2 * 10^11) ∧ x * 10^6 + y = z^2 := sorry

-- Part (c)
theorem smallest_k_for_n_digit_number_square : ∀ (n : ℕ), ∃ (k : ℕ), k = n + 1 ∧ ∀ (x : ℕ), (10^(n-1) ≤ x) → (x < 10^n) → ∃ y : ℕ, (10^(n + k - 1) ≤ x * 10^k + y) ∧ (x * 10^k + y) < 10^(n + k) ∧ ∃ z : ℕ, x * 10^k + y = z^2 := sorry

end no_six_digit_starting_with_five_12_digit_square_six_digit_starting_with_one_12_digit_square_smallest_k_for_n_digit_number_square_l113_113295


namespace find_number_of_elements_l113_113806

theorem find_number_of_elements (n S : ℕ) (h1: (S + 26) / n = 15) (h2: (S + 36) / n = 16) : n = 10 := by
  sorry

end find_number_of_elements_l113_113806


namespace ArianaBoughtTulips_l113_113582

theorem ArianaBoughtTulips (total_flowers : ℕ) (fraction_roses : ℚ) (carnations : ℕ) 
    (h_total : total_flowers = 40) (h_fraction : fraction_roses = 2/5) (h_carnations : carnations = 14) : 
    total_flowers - (total_flowers * fraction_roses + carnations) = 10 := by
  sorry

end ArianaBoughtTulips_l113_113582


namespace ascorbic_acid_oxygen_mass_percentage_l113_113603

noncomputable def mass_percentage_oxygen_in_ascorbic_acid : Float := 54.49

theorem ascorbic_acid_oxygen_mass_percentage :
  let C_mass := 12.01
  let H_mass := 1.01
  let O_mass := 16.00
  let ascorbic_acid_formula := (6, 8, 6) -- (number of C, number of H, number of O)
  let total_mass := 6 * C_mass + 8 * H_mass + 6 * O_mass
  let O_mass_total := 6 * O_mass
  mass_percentage_oxygen_in_ascorbic_acid = (O_mass_total / total_mass) * 100 := by
  sorry

end ascorbic_acid_oxygen_mass_percentage_l113_113603


namespace xyz_inequality_l113_113787

-- Definitions for the conditions and the statement of the problem
theorem xyz_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h_ineq : x * y * z ≥ x * y + y * z + z * x) : 
  x * y * z ≥ 3 * (x + y + z) :=
by
  sorry

end xyz_inequality_l113_113787


namespace system_solution_unique_l113_113938

theorem system_solution_unique (w x y z : ℝ) (h1 : w + x + y + z = 12)
  (h2 : w * x * y * z = w * x + w * y + w * z + x * y + x * z + y * z + 27) :
  w = 3 ∧ x = 3 ∧ y = 3 ∧ z = 3 := 
sorry

end system_solution_unique_l113_113938


namespace not_snowing_next_five_days_l113_113688

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end not_snowing_next_five_days_l113_113688


namespace liters_per_bottle_l113_113660

-- Condition statements
def price_per_liter : ℕ := 1
def total_cost : ℕ := 12
def num_bottles : ℕ := 6

-- Desired result statement
theorem liters_per_bottle : (total_cost / price_per_liter) / num_bottles = 2 := by
  sorry

end liters_per_bottle_l113_113660


namespace system_solution_l113_113879

theorem system_solution (x y : ℝ) 
  (h1 : 0 < x + y) 
  (h2 : x + y ≠ 1) 
  (h3 : 2 * x - y ≠ 0)
  (eq1 : (x + y) * 2^(y - 2 * x) = 6.25) 
  (eq2 : (x + y) * (1 / (2 * x - y)) = 5) :
x = 9 ∧ y = 16 := 
sorry

end system_solution_l113_113879


namespace time_per_employee_updating_payroll_records_l113_113376

-- Define the conditions
def minutes_making_coffee : ℕ := 5
def minutes_per_employee_status_update : ℕ := 2
def num_employees : ℕ := 9
def total_morning_routine_minutes : ℕ := 50

-- Define the proof statement encapsulating the problem
theorem time_per_employee_updating_payroll_records :
  (total_morning_routine_minutes - (minutes_making_coffee + minutes_per_employee_status_update * num_employees)) / num_employees = 3 := by
  sorry

end time_per_employee_updating_payroll_records_l113_113376


namespace max_non_intersecting_segments_l113_113950

theorem max_non_intersecting_segments (n m : ℕ) (hn: 1 < n) (hm: m ≥ 3): 
  ∃ L, L = 3 * n - m - 3 :=
by
  sorry

end max_non_intersecting_segments_l113_113950


namespace find_integer_mod_condition_l113_113477

theorem find_integer_mod_condition (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 4) (h3 : n ≡ -998 [ZMOD 5]) : n = 2 :=
sorry

end find_integer_mod_condition_l113_113477


namespace roberto_outfits_l113_113200

theorem roberto_outfits : 
  let trousers := 5
  let shirts := 5
  let jackets := 3
  (trousers * shirts * jackets = 75) :=
by sorry

end roberto_outfits_l113_113200


namespace percent_of_class_received_50_to_59_l113_113720

-- Define the frequencies for each score range
def freq_90_to_100 := 5
def freq_80_to_89 := 7
def freq_70_to_79 := 9
def freq_60_to_69 := 8
def freq_50_to_59 := 4
def freq_below_50 := 3

-- Define the total number of students
def total_students := freq_90_to_100 + freq_80_to_89 + freq_70_to_79 + freq_60_to_69 + freq_50_to_59 + freq_below_50

-- Define the frequency of students scoring in the 50%-59% range
def freq_50_to_59_ratio := (freq_50_to_59 : ℚ) / total_students

-- Define the percentage calculation
def percent_50_to_59 := freq_50_to_59_ratio * 100

theorem percent_of_class_received_50_to_59 :
  percent_50_to_59 = 100 / 9 := 
by {
  sorry
}

end percent_of_class_received_50_to_59_l113_113720


namespace binary_to_decimal_l113_113736

theorem binary_to_decimal : 
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0) = 54 :=
by 
  sorry

end binary_to_decimal_l113_113736


namespace distance_PQ_is_12_miles_l113_113799

-- Define the conditions
def average_speed_PQ := 40 -- mph
def average_speed_QP := 45 -- mph
def time_difference := 2 -- minutes

-- Main proof statement to show that the distance is 12 miles
theorem distance_PQ_is_12_miles 
    (x : ℝ) 
    (h1 : average_speed_PQ > 0) 
    (h2 : average_speed_QP > 0) 
    (h3 : abs ((x / average_speed_PQ * 60) - (x / average_speed_QP * 60)) = time_difference) 
    : x = 12 := 
by
  sorry

end distance_PQ_is_12_miles_l113_113799


namespace inequality_a_b_c_l113_113610

theorem inequality_a_b_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^3 / (a^2 + a * b + b^2)) + (b^3 / (b^2 + b * c + c^2)) + (c^3 / (c^2 + c * a + a^2)) ≥ (a + b + c) / 3 :=
sorry

end inequality_a_b_c_l113_113610


namespace equation_of_line_l113_113955

theorem equation_of_line (l : ℝ → ℝ) :
  (∀ (P : ℝ × ℝ), P = (4, 2) → 
    ∃ (a b : ℝ), ((P = ( (4 - a), (2 - b)) ∨ P = ( (4 + a), (2 + b))) ∧ 
    ((4 - a)^2 / 36 + (2 - b)^2 / 9 = 1) ∧ ((4 + a)^2 / 36 + (2 + b)^2 / 9 = 1)) ∧
    (P.2 = l P.1)) →
  (∀ (x y : ℝ), y = l x ↔ 2 * x + 3 * y - 16 = 0) :=
by
  intros h P hp
  sorry -- Placeholder for the proof

end equation_of_line_l113_113955


namespace each_bug_ate_1_5_flowers_l113_113389

-- Define the conditions given in the problem
def bugs : ℝ := 2.0
def flowers : ℝ := 3.0

-- The goal is to prove that the number of flowers each bug ate is 1.5
theorem each_bug_ate_1_5_flowers : (flowers / bugs) = 1.5 :=
by
  sorry

end each_bug_ate_1_5_flowers_l113_113389


namespace lorraine_initial_brownies_l113_113385

theorem lorraine_initial_brownies (B : ℝ) 
(h1: (0.375 * B - 1 = 5)) : B = 16 := 
sorry

end lorraine_initial_brownies_l113_113385


namespace point_on_graph_of_inverse_proportion_l113_113460

theorem point_on_graph_of_inverse_proportion :
  ∃ x y : ℝ, (x = 2 ∧ y = 4) ∧ y = 8 / x :=
by
  sorry

end point_on_graph_of_inverse_proportion_l113_113460


namespace range_of_magnitudes_l113_113755

theorem range_of_magnitudes (AB AC BC : ℝ) (hAB : AB = 8) (hAC : AC = 5) :
  3 ≤ BC ∧ BC ≤ 13 :=
by
  sorry

end range_of_magnitudes_l113_113755


namespace known_number_is_24_l113_113697

noncomputable def HCF (a b : ℕ) : ℕ := sorry
noncomputable def LCM (a b : ℕ) : ℕ := sorry

theorem known_number_is_24 (A B : ℕ) (h1 : B = 182)
  (h2 : HCF A B = 14)
  (h3 : LCM A B = 312) : A = 24 := by
  sorry

end known_number_is_24_l113_113697


namespace A_subscribed_fraction_l113_113894

theorem A_subscribed_fraction 
  (total_profit : ℝ) (A_share : ℝ) 
  (B_fraction : ℝ) (C_fraction : ℝ) 
  (A_fraction : ℝ) :
  total_profit = 2430 →
  A_share = 810 →
  B_fraction = 1/4 →
  C_fraction = 1/5 →
  A_fraction = A_share / total_profit →
  A_fraction = 1/3 :=
by
  intros h_total_profit h_A_share h_B_fraction h_C_fraction h_A_fraction
  sorry

end A_subscribed_fraction_l113_113894


namespace ott_fraction_of_total_money_l113_113387

-- Definitions for the conditions
def Moe_initial_money (x : ℕ) : ℕ := 3 * x
def Loki_initial_money (x : ℕ) : ℕ := 5 * x
def Nick_initial_money (x : ℕ) : ℕ := 4 * x
def Total_initial_money (x : ℕ) : ℕ := Moe_initial_money x + Loki_initial_money x + Nick_initial_money x
def Ott_received_money (x : ℕ) : ℕ := 3 * x

-- Making the statement we want to prove
theorem ott_fraction_of_total_money (x : ℕ) : 
  (Ott_received_money x) / (Total_initial_money x) = 1 / 4 := by
  sorry

end ott_fraction_of_total_money_l113_113387


namespace james_hours_worked_l113_113744

variable (x : ℝ) (y : ℝ)

theorem james_hours_worked (h1: 18 * x + 16 * (1.5 * x) = 40 * x + (y - 40) * (2 * x)) : y = 41 :=
by
  sorry

end james_hours_worked_l113_113744


namespace exist_distinct_indices_l113_113014

theorem exist_distinct_indices (n : ℕ) (h1 : n > 3)
  (a : Fin n.succ → ℕ) 
  (h2 : StrictMono a) 
  (h3 : a n ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n.succ), 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ 
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ 
    k ≠ l ∧ k ≠ m ∧ l ≠ m ∧ 
    a i + a j = a k + a l ∧ 
    a k + a l = a m := 
sorry

end exist_distinct_indices_l113_113014


namespace find_f_three_l113_113848

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l113_113848


namespace sum_is_zero_l113_113384

theorem sum_is_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / |a|) + (b / |b|) + (c / |c|) + ((a * b * c) / |a * b * c|) = 0 :=
by
  sorry

end sum_is_zero_l113_113384


namespace range_of_a_l113_113791

def p (x : ℝ) : Prop := abs (2 * x - 1) ≤ 3

def q (x a : ℝ) : Prop := x^2 - (2*a + 1) * x + a*(a + 1) ≤ 0

theorem range_of_a : 
  (∀ x a, (¬ q x a) → (¬ p x))
  ∧ (∃ x a, (¬ q x a) ∧ (¬ p x))
  → (-1 : ℝ) ≤ a ∧ a ≤ (1 : ℝ) :=
sorry

end range_of_a_l113_113791


namespace count_invitations_l113_113577

theorem count_invitations (teachers : Finset ℕ) (A B : ℕ) (hA : A ∈ teachers) (hB : B ∈ teachers) (h_size : teachers.card = 10):
  ∃ (ways : ℕ), ways = 140 ∧ ∀ (S : Finset ℕ), S.card = 6 → ((A ∈ S ∧ B ∉ S) ∨ (A ∉ S ∧ B ∈ S) ∨ (A ∉ S ∧ B ∉ S)) ↔ ways = 140 := 
sorry

end count_invitations_l113_113577


namespace emily_jumping_game_l113_113473

def tiles_number (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 5 = 2

theorem emily_jumping_game : tiles_number 47 :=
by
  unfold tiles_number
  sorry

end emily_jumping_game_l113_113473


namespace smallest_positive_multiple_45_l113_113243

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l113_113243


namespace solution_set_inequality_l113_113698

theorem solution_set_inequality
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, (1 < x ∧ x < 2) → ax^2 + bx + c > 0) :
  ∃ s : Set ℝ, s = {x | (1/2) < x ∧ x < 1} ∧ ∀ x : ℝ, x ∈ s → cx^2 + bx + a > 0 := by
sorry

end solution_set_inequality_l113_113698


namespace dartboard_area_ratio_l113_113322

theorem dartboard_area_ratio
  (side_length : ℝ)
  (h_side_length : side_length = 2)
  (t : ℝ)
  (q : ℝ)
  (h_t : t = (1 / 2) * (1 / (Real.sqrt 2)) * (1 / (Real.sqrt 2)))
  (h_q : q = ((side_length * side_length) - (8 * t)) / 4) :
  q / t = 2 := by
  sorry

end dartboard_area_ratio_l113_113322


namespace no_snow_five_days_l113_113684

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l113_113684


namespace neg_p_l113_113653

theorem neg_p : ∀ (m : ℝ), ∀ (x : ℝ), (x^2 + m*x + 1 ≠ 0) :=
by
  sorry

end neg_p_l113_113653


namespace find_f_three_l113_113851

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l113_113851


namespace market_value_calculation_l113_113558

variables (annual_dividend_per_share face_value yield market_value : ℝ)

axiom annual_dividend_definition : annual_dividend_per_share = 0.09 * face_value
axiom face_value_definition : face_value = 100
axiom yield_definition : yield = 0.25

theorem market_value_calculation (annual_dividend_per_share face_value yield market_value : ℝ) 
  (h1: annual_dividend_per_share = 0.09 * face_value)
  (h2: face_value = 100)
  (h3: yield = 0.25):
  market_value = annual_dividend_per_share / yield :=
sorry

end market_value_calculation_l113_113558


namespace find_f_three_l113_113850

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l113_113850


namespace smallest_positive_multiple_of_45_l113_113253

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l113_113253


namespace find_m_for_parallel_lines_l113_113419

noncomputable def parallel_lines_x_plus_1_plus_m_y_eq_2_minus_m_and_m_x_plus_2_y_plus_8_eq_0 (m : ℝ) : Prop :=
  let l1_slope := -(1 + m) / 1
  let l2_slope := -m / 2
  l1_slope = l2_slope

theorem find_m_for_parallel_lines :
  parallel_lines_x_plus_1_plus_m_y_eq_2_minus_m_and_m_x_plus_2_y_plus_8_eq_0 m →
  m = 1 :=
by
  intro h_parallel
  -- Here we would present the proof steps to show that m = 1 under the given conditions.
  sorry

end find_m_for_parallel_lines_l113_113419


namespace percentage_of_students_wearing_red_shirts_l113_113775

/-- In a school of 700 students:
    - 45% of students wear blue shirts.
    - 15% of students wear green shirts.
    - 119 students wear shirts of other colors.
    We are proving that the percentage of students wearing red shirts is 23%. --/
theorem percentage_of_students_wearing_red_shirts:
  let total_students := 700
  let blue_shirt_percentage := 45 / 100
  let green_shirt_percentage := 15 / 100
  let other_colors_students := 119
  let students_with_blue_shirts := blue_shirt_percentage * total_students
  let students_with_green_shirts := green_shirt_percentage * total_students
  let students_with_other_colors := other_colors_students
  let students_with_blue_green_or_red_shirts := total_students - students_with_other_colors
  let students_with_red_shirts := students_with_blue_green_or_red_shirts - students_with_blue_shirts - students_with_green_shirts
  (students_with_red_shirts / total_students) * 100 = 23 := by
  sorry

end percentage_of_students_wearing_red_shirts_l113_113775


namespace price_of_feed_corn_l113_113449

theorem price_of_feed_corn :
  ∀ (num_sheep : ℕ) (num_cows : ℕ) (grass_per_cow : ℕ) (grass_per_sheep : ℕ)
    (feed_corn_duration_cow : ℕ) (feed_corn_duration_sheep : ℕ)
    (total_grass : ℕ) (total_expenditure : ℕ) (months_in_year : ℕ),
  num_sheep = 8 →
  num_cows = 5 →
  grass_per_cow = 2 →
  grass_per_sheep = 1 →
  feed_corn_duration_cow = 1 →
  feed_corn_duration_sheep = 2 →
  total_grass = 144 →
  total_expenditure = 360 →
  months_in_year = 12 →
  ((total_expenditure : ℝ) / (((num_cows * feed_corn_duration_cow * 4) + (num_sheep * (4 / feed_corn_duration_sheep))) : ℝ)) = 10 :=
by
  intros
  sorry

end price_of_feed_corn_l113_113449


namespace prime_squares_mod_180_l113_113584

theorem prime_squares_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) : 
  ∃ r ∈ {1, 49}, ∀ q ∈ {1, 49}, (q ≡ p^2 [MOD 180]) → q = r :=
by
  sorry

end prime_squares_mod_180_l113_113584


namespace min_knights_in_village_l113_113776

theorem min_knights_in_village :
  ∃ (K L : ℕ), K + L = 7 ∧ 2 * K * L = 24 ∧ K ≥ 3 :=
by
  sorry

end min_knights_in_village_l113_113776


namespace smallest_positive_multiple_of_45_is_45_l113_113239

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113239


namespace percentage_short_l113_113210

def cost_of_goldfish : ℝ := 0.25
def sale_price_of_goldfish : ℝ := 0.75
def tank_price : ℝ := 100
def goldfish_sold : ℕ := 110

theorem percentage_short : ((tank_price - (sale_price_of_goldfish - cost_of_goldfish) * goldfish_sold) / tank_price) * 100 = 45 := 
by
  sorry

end percentage_short_l113_113210


namespace recurring_decimals_sum_l113_113934

theorem recurring_decimals_sum :
  (0.333333333333 : ℚ) + (0.040404040404 : ℚ) + (0.005005005005 : ℚ) = 42 / 111 :=
by
  sorry

end recurring_decimals_sum_l113_113934


namespace min_floor_sum_l113_113012

-- Definitions of the conditions
variables (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 24)

-- Our main theorem statement
theorem min_floor_sum (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 24) :
  (Nat.floor ((a+b) / c) + Nat.floor ((b+c) / a) + Nat.floor ((c+a) / b)) = 6 := 
sorry

end min_floor_sum_l113_113012


namespace find_f_of_3_l113_113818

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l113_113818


namespace total_wheels_computation_probability_bicycle_or_tricycle_l113_113751

def transportation_data := {
  cars : ℕ,
  bicycles : ℕ,
  trucks : ℕ,
  tricycles : ℕ,
  motorcycles : ℕ,
  skateboards : ℕ,
  unicycles : ℕ
}

def wheels_per_vehicle : transportation_data → ℕ
| ⟨cars, bicycles, trucks, tricycles, motorcycles, skateboards, unicycles⟩ :=
  (cars * 4) + (bicycles * 2) + (trucks * 4) + (tricycles * 3) + (motorcycles * 2) + (skateboards * 4) + (unicycles * 1)

theorem total_wheels_computation:
  let t := ⟨15, 3, 8, 1, 4, 2, 1⟩ in wheels_per_vehicle t = 118 :=
by {
  intro t,
  unfold wheels_per_vehicle,
  norm_num,
  have h : ((((15 * 4) + (3 * 2) + (8 * 4)) + (1 * 3) + (4 * 2)) + (2 * 4) + (1 * 1) = 118), by norm_num,
  exact h,
  sorry
}

theorem probability_bicycle_or_tricycle:
  let total_units := 15 + 3 + 8 + 1 + 4 + 2 + 1 in
  let bicycles_and_tricycles := 3 + 1 in
  (bicycles_and_tricycles : rat) / (total_units : rat) ≈ 11.76 / 100 :=
by {
  rw [total_units, bicycles_and_tricycles],
  norm_num,
  have h : ((4 : rat) / (34 : rat)) ≈ (11.76 / 100), sorry,
  exact h,
  sorry
}

end total_wheels_computation_probability_bicycle_or_tricycle_l113_113751


namespace smallest_positive_multiple_l113_113283

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l113_113283


namespace trains_clear_time_l113_113442

noncomputable def time_to_clear (length_train1 length_train2 speed_train1 speed_train2 : ℕ) : ℝ :=
  (length_train1 + length_train2) / ((speed_train1 + speed_train2) * 1000 / 3600)

theorem trains_clear_time :
  time_to_clear 121 153 80 65 = 6.803 :=
by
  -- This is a placeholder for the proof
  sorry

end trains_clear_time_l113_113442


namespace find_incorrect_value_of_observation_l113_113858

noncomputable def incorrect_observation_value (mean1 : ℝ) (mean2 : ℝ) (n : ℕ) : ℝ :=
  let old_sum := mean1 * n
  let new_sum := mean2 * n
  let correct_value := 45
  let incorrect_value := (old_sum - new_sum + correct_value)
  (incorrect_value / -1)

theorem find_incorrect_value_of_observation :
  incorrect_observation_value 36 36.5 50 = 20 :=
by
  -- By the problem setup, incorrect_observation_value 36 36.5 50 is as defined in the proof steps.
  -- As per the proof steps and calculation, incorrect_observation_value 36 36.5 50 should evaluate to 20.
  sorry

end find_incorrect_value_of_observation_l113_113858


namespace find_f_3_l113_113835

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l113_113835


namespace oak_grove_libraries_total_books_l113_113869

theorem oak_grove_libraries_total_books :
  let publicLibraryBooks := 1986
  let schoolLibrariesBooks := 5106
  let communityCollegeLibraryBooks := 3294.5
  let medicalLibraryBooks := 1342.25
  let lawLibraryBooks := 2785.75
  publicLibraryBooks + schoolLibrariesBooks + communityCollegeLibraryBooks + medicalLibraryBooks + lawLibraryBooks = 15514.5 :=
by
  sorry

end oak_grove_libraries_total_books_l113_113869


namespace find_a_l113_113763

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then (1/2)^x - 7 else x^2

theorem find_a (a : ℝ) (h : f a = 1) : a = -3 ∨ a = 1 := 
by
  sorry

end find_a_l113_113763


namespace pyramid_x_value_l113_113044

theorem pyramid_x_value (x y : ℝ) 
  (h1 : 150 = 10 * x)
  (h2 : 225 = x * 15)
  (h3 : 1800 = 150 * y * 225) :
  x = 15 :=
sorry

end pyramid_x_value_l113_113044


namespace total_days_2010_to_2013_l113_113626

theorem total_days_2010_to_2013 :
  let year2010_days := 365
  let year2011_days := 365
  let year2012_days := 366
  let year2013_days := 365
  year2010_days + year2011_days + year2012_days + year2013_days = 1461 := by
  sorry

end total_days_2010_to_2013_l113_113626


namespace find_f_3_l113_113843

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l113_113843


namespace count_true_statements_l113_113096

open Set

variable {M P : Set α}

theorem count_true_statements (h : ¬ ∀ x ∈ M, x ∈ P) (hne : Nonempty M) :
  (¬ ∃ x, x ∈ M ∧ x ∈ P ∨ ∀ x, x ∈ M → x ∈ P) ∧ (∃ x, x ∈ M ∧ x ∉ P) ∧ 
  ¬ (∃ x, x ∈ M ∧ x ∈ P) ∧ (¬ ∀ x, x ∈ M → x ∈ P) :=
sorry

end count_true_statements_l113_113096


namespace fill_bathtub_time_l113_113172

theorem fill_bathtub_time (V : ℝ) (cold_rate hot_rate drain_rate net_rate : ℝ) 
  (hcold : cold_rate = V / 10) 
  (hhot : hot_rate = V / 15) 
  (hdrain : drain_rate = -V / 12) 
  (hnet : net_rate = cold_rate + hot_rate + drain_rate) 
  (V_eq : V = 1) : 
  1 / net_rate = 12 :=
by {
  -- placeholder for proof steps
  sorry
}

end fill_bathtub_time_l113_113172


namespace sector_area_l113_113185

theorem sector_area (r : ℝ) (θ : ℝ) (arc_area : ℝ) : 
  r = 24 ∧ θ = 110 ∧ arc_area = 176 * Real.pi → 
  arc_area = (θ / 360) * (Real.pi * r ^ 2) :=
by
  intros
  sorry

end sector_area_l113_113185


namespace age_difference_l113_113890

-- Defining the current age of the son
def S : ℕ := 26

-- Defining the current age of the man
def M : ℕ := 54

-- Defining the condition that in two years, the man's age is twice the son's age
def condition : Prop := (M + 2) = 2 * (S + 2)

-- The theorem that states how much older the man is than the son
theorem age_difference : condition → M - S = 28 := by
  sorry

end age_difference_l113_113890


namespace value_of_a_l113_113356

theorem value_of_a (a : ℝ) (A : Set ℝ) (hA : A = {a^2, 1}) (h : 3 ∈ A) : 
  a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by
  sorry

end value_of_a_l113_113356


namespace odd_numbers_square_division_l113_113623

theorem odd_numbers_square_division (m n : ℤ) (hm : Odd m) (hn : Odd n) (h : m^2 - n^2 + 1 ∣ n^2 - 1) : ∃ k : ℤ, m^2 - n^2 + 1 = k^2 := 
sorry

end odd_numbers_square_division_l113_113623


namespace total_number_of_toys_l113_113995

def jaxon_toys := 15
def gabriel_toys := 2 * jaxon_toys
def jerry_toys := gabriel_toys + 8

theorem total_number_of_toys : jaxon_toys + gabriel_toys + jerry_toys = 83 := by
  sorry

end total_number_of_toys_l113_113995


namespace arithmetic_sequence_common_difference_l113_113143

noncomputable def common_difference (a b : ℝ) : ℝ := a - 1

theorem arithmetic_sequence_common_difference :
  ∀ (a b : ℝ), 
    (a - 1 = b - a) → 
    ((a + 2) ^ 2 = 3 * (b + 5)) → 
    common_difference a b = 3 := by
  intros a b h1 h2
  sorry

end arithmetic_sequence_common_difference_l113_113143


namespace find_b_l113_113220

-- Define what it means for b to be a solution
def is_solution (b : ℤ) : Prop :=
  b > 4 ∧ ∃ k : ℤ, 4 * b + 5 = k * k

-- State the problem
theorem find_b : ∃ b : ℤ, is_solution b ∧ ∀ b' : ℤ, is_solution b' → b' ≥ 5 := by
  sorry

end find_b_l113_113220


namespace river_bank_depth_l113_113093

-- Definitions related to the problem
def is_trapezium (top_width bottom_width height area : ℝ) :=
  area = 1 / 2 * (top_width + bottom_width) * height

-- The theorem we want to prove
theorem river_bank_depth :
  ∀ (top_width bottom_width area : ℝ), 
    top_width = 12 → 
    bottom_width = 8 → 
    area = 500 → 
    ∃ h : ℝ, is_trapezium top_width bottom_width h area ∧ h = 50 :=
by
  intros top_width bottom_width area ht hb ha
  sorry

end river_bank_depth_l113_113093


namespace number_of_smoothies_l113_113182

-- Definitions of the given conditions
def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def total_cost : ℕ := 17

-- Statement of the proof problem
theorem number_of_smoothies (S : ℕ) : burger_cost + sandwich_cost + S * smoothie_cost = total_cost → S = 2 :=
by
  intro h
  sorry

end number_of_smoothies_l113_113182


namespace quad_sin_theorem_l113_113778

-- Define the necessary entities in Lean
structure Quadrilateral (A B C D : Type) :=
(angleB : ℝ)
(angleD : ℝ)
(angleA : ℝ)

-- Define the main theorem
theorem quad_sin_theorem {A B C D : Type} (quad : Quadrilateral A B C D) (AC AD : ℝ) (α : ℝ) :
  quad.angleB = 90 ∧ quad.angleD = 90 ∧ quad.angleA = α → AD = AC * Real.sin α := 
sorry

end quad_sin_theorem_l113_113778


namespace handshake_max_l113_113632

theorem handshake_max (N : ℕ) (hN : N > 4) (pN pNm1 : ℕ) 
    (hpN : pN ≠ pNm1) (h1 : ∃ p1, pN ≠ p1) (h2 : ∃ p2, pNm1 ≠ p2) :
    ∀ (i : ℕ), i ≤ N - 2 → i ≤ N - 2 :=
sorry

end handshake_max_l113_113632


namespace fraction_simplification_l113_113878

theorem fraction_simplification (a : ℝ) (h1 : a > 1) (h2 : a ≠ 2 / Real.sqrt 3) : 
  (a^3 - 3 * a^2 + 4 + (a^2 - 4) * Real.sqrt (a^2 - 1)) / 
  (a^3 + 3 * a^2 - 4 + (a^2 - 4) * Real.sqrt (a^2 - 1)) = 
  ((a - 2) * Real.sqrt (a + 1)) / ((a + 2) * Real.sqrt (a - 1)) :=
by
  sorry

end fraction_simplification_l113_113878


namespace sum_of_consecutive_integers_product_is_negative_336_l113_113211

theorem sum_of_consecutive_integers_product_is_negative_336 :
  ∃ (n : ℤ), (n - 1) * n * (n + 1) = -336 ∧ (n - 1) + n + (n + 1) = -21 :=
by
  sorry

end sum_of_consecutive_integers_product_is_negative_336_l113_113211


namespace spinner_points_east_l113_113781

-- Definitions for the conditions
def initial_direction := "north"

-- Clockwise and counterclockwise movements as improper fractions
def clockwise_move := (7 : ℚ) / 2
def counterclockwise_move := (17 : ℚ) / 4

-- Compute the net movement (negative means counterclockwise)
def net_movement := clockwise_move - counterclockwise_move

-- Translate net movement into a final direction (using modulo arithmetic with 1 revolution = 360 degrees equivalent)
def final_position : ℚ := (net_movement + 1) % 1

-- The goal is to prove that the final direction is east (which corresponds to 1/4 revolution)
theorem spinner_points_east :
  final_position = (1 / 4 : ℚ) :=
by
  sorry

end spinner_points_east_l113_113781


namespace smaller_solution_of_quadratic_eq_l113_113479

noncomputable def smaller_solution (a b c : ℝ) : ℝ :=
  if a ≠ 0 then min ((-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a))
              ((-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a))
  else if b ≠ 0 then -c / b else 0 

theorem smaller_solution_of_quadratic_eq :
  smaller_solution 1 (-13) (-30) = -2 := 
by
  sorry

end smaller_solution_of_quadratic_eq_l113_113479


namespace average_typing_speed_l113_113809

theorem average_typing_speed :
  let rudy := 64
  let joyce := 76
  let gladys := 91
  let lisa := 80
  let mike := 89
  let total := rudy + joyce + gladys + lisa + mike
  let average := total / 5
  in average = 80 := by
  sorry

end average_typing_speed_l113_113809


namespace root_in_interval_2_3_l113_113208

noncomputable def f (x : ℝ) : ℝ := -|x - 5| + 2^(x - 1)

theorem root_in_interval_2_3 :
  (f 2) * (f 3) < 0 → ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 := by sorry

end root_in_interval_2_3_l113_113208


namespace max_omega_l113_113353

noncomputable def f (ω φ x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem max_omega (ω φ : ℝ) (k k' : ℤ) (hω_pos : ω > 0) (hφ1 : 0 < φ)
  (hφ2 : φ < Real.pi / 2) (h1 : f ω φ (-Real.pi / 4) = 0)
  (h2 : ∀ x, f ω φ (Real.pi / 4 - x) = f ω φ (Real.pi / 4 + x))
  (h3 : ∀ x, x ∈ Set.Ioo (Real.pi / 18) (2 * Real.pi / 9) →
    Monotone (f ω φ)) :
  ω = 5 :=
sorry

end max_omega_l113_113353


namespace multiplication_in_A_l113_113949

def A : Set ℤ :=
  {x | ∃ a b : ℤ, x = a^2 + b^2}

theorem multiplication_in_A (x1 x2 : ℤ) (h1 : x1 ∈ A) (h2 : x2 ∈ A) :
  x1 * x2 ∈ A :=
sorry

end multiplication_in_A_l113_113949


namespace sam_initial_balloons_l113_113800

theorem sam_initial_balloons:
  ∀ (S : ℝ), (S - 5.0 + 7.0 = 8) → S = 6.0 :=
by
  intro S h
  sorry

end sam_initial_balloons_l113_113800


namespace crayons_count_l113_113393

-- Define the initial number of crayons
def initial_crayons : ℕ := 1453

-- Define the number of crayons given away
def crayons_given_away : ℕ := 563

-- Define the number of crayons lost
def crayons_lost : ℕ := 558

-- Define the final number of crayons left
def final_crayons_left : ℕ := initial_crayons - crayons_given_away - crayons_lost

-- State that the final number of crayons left is 332
theorem crayons_count : final_crayons_left = 332 :=
by
    -- This is where the proof would go, which we're skipping with sorry
    sorry

end crayons_count_l113_113393


namespace find_f_of_3_l113_113832

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l113_113832


namespace simplify_expression_l113_113659

theorem simplify_expression (a : ℝ) (h : a = Real.sqrt 3 - 3) : 
  (a^2 - 4 * a + 4) / (a^2 - 4) / ((a - 2) / (a^2 + 2 * a)) + 3 = Real.sqrt 3 :=
by
  sorry

end simplify_expression_l113_113659


namespace digit_in_tens_place_l113_113076

theorem digit_in_tens_place (n : ℕ) (cycle : List ℕ) (h_cycle : cycle = [16, 96, 76, 56]) (hk : n % 4 = 3) :
  (6 ^ n % 100) / 10 % 10 = 7 := by
  sorry

end digit_in_tens_place_l113_113076


namespace ordered_triples_count_l113_113752

theorem ordered_triples_count :
  {n : ℕ // n = 4} :=
sorry

end ordered_triples_count_l113_113752


namespace smallest_number_is_neg1_l113_113730

-- Defining the list of numbers
def numbers := [0, -1, 1, 2]

-- Theorem statement to prove that the smallest number in the list is -1
theorem smallest_number_is_neg1 :
  ∀ x ∈ numbers, x ≥ -1 := 
sorry

end smallest_number_is_neg1_l113_113730


namespace problem_l113_113087

def isRightTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

def CannotFormRightTriangle (lst : List ℝ) : Prop :=
  ¬isRightTriangle lst.head! lst.tail.head! lst.tail.tail.head!

theorem problem :
  (¬isRightTriangle 3 4 5 ∧ ¬isRightTriangle 5 12 13 ∧ ¬isRightTriangle 2 3 (Real.sqrt 13)) ∧ CannotFormRightTriangle [4, 6, 8] :=
by
  sorry

end problem_l113_113087


namespace smallest_two_digit_integer_l113_113877

theorem smallest_two_digit_integer (n a b : ℕ) (h1 : n = 10 * a + b) (h2 : 2 * n = 10 * b + a + 5) (h3 : 1 ≤ a) (h4 : a ≤ 9) (h5 : 0 ≤ b) (h6 : b ≤ 9) : n = 69 := 
by 
  sorry

end smallest_two_digit_integer_l113_113877


namespace find_f_3_l113_113837

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l113_113837


namespace exists_three_digit_number_l113_113134

theorem exists_three_digit_number : ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ (100 * a + 10 * b + c = a^3 + b^3 + c^3) ∧ (100 * a + 10 * b + c ≥ 100 ∧ 100 * a + 10 * b + c < 1000) := 
sorry

end exists_three_digit_number_l113_113134


namespace calculate_expression_l113_113898

theorem calculate_expression : (3.65 - 1.25) * 2 = 4.80 := 
by 
  sorry

end calculate_expression_l113_113898


namespace remove_two_terms_sum_one_l113_113456

theorem remove_two_terms_sum_one :
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let total_sum := (60/120) + (30/120) + (20/120) + (15/120) + (12/120) + (10/120)
  let target_sum := 1
  ∃ (a b : ℚ), a ∈ fractions ∧ b ∈ fractions ∧ 
    (a ≠ b) ∧ 
    (total_sum - (a + b) = target_sum) 
    ∧ ((a = 1/8 ∧ b = 1/10) ∨ (a = 1/10 ∧ b = 1/8)) := 
by
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let fractions_set := fractions.to_finset
  let total_sum := 147 / 120
  let target := 120 / 120
  use [1/8, 1/10]
  simp [Fractions, fractions_set, total_sum, target]
  split
  { exact sorry }
  { split
    { rw [1/8, 1/10, total_sum, target], norm_num }
    { simp, norm_num }}
  sorry

end remove_two_terms_sum_one_l113_113456


namespace SamLastPage_l113_113366

theorem SamLastPage (total_pages : ℕ) (Sam_read_time : ℕ) (Lily_read_time : ℕ) (last_page : ℕ) :
  total_pages = 920 ∧ Sam_read_time = 30 ∧ Lily_read_time = 50 → last_page = 575 :=
by
  intros h
  sorry

end SamLastPage_l113_113366


namespace bus_stop_time_l113_113294

theorem bus_stop_time 
  (bus_speed_without_stoppages : ℤ)
  (bus_speed_with_stoppages : ℤ)
  (h1 : bus_speed_without_stoppages = 54)
  (h2 : bus_speed_with_stoppages = 36) :
  ∃ t : ℕ, t = 20 :=
by
  sorry

end bus_stop_time_l113_113294


namespace product_of_large_integers_l113_113403

theorem product_of_large_integers :
  ∃ A B : ℤ, A > 10^2009 ∧ B > 10^2009 ∧ A * B = 3^(4^5) + 4^(5^6) :=
by
  sorry

end product_of_large_integers_l113_113403


namespace smallest_positive_multiple_of_45_l113_113266

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l113_113266


namespace Kaarel_wins_l113_113399

theorem Kaarel_wins (p : ℕ) (hp : Prime p) (hp_gt3 : p > 3) :
  ∃ (x y a : ℕ), x ∈ Finset.range (p-1) ∧ y ∈ Finset.range (p-1) ∧ a ∈ Finset.range (p-1) ∧ 
  x ≠ y ∧ y ≠ (p - x) ∧ a ≠ x ∧ a ≠ (p - x) ∧ a ≠ y ∧ 
  (x * (p - x) + y * a) % p = 0 :=
sorry

end Kaarel_wins_l113_113399


namespace number_and_its_square_root_l113_113710

theorem number_and_its_square_root (x : ℝ) (h : x + 10 * Real.sqrt x = 39) : x = 9 :=
sorry

end number_and_its_square_root_l113_113710


namespace problem_statement_l113_113738

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 5}
def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem problem_statement : star A B = {1, 7} := by
  sorry

end problem_statement_l113_113738


namespace compare_negatives_l113_113904

theorem compare_negatives : -4 < -2.1 := 
sorry

end compare_negatives_l113_113904


namespace product_of_roots_eq_neg_125_over_4_l113_113138

theorem product_of_roots_eq_neg_125_over_4 :
  (∀ x y : ℝ, (24 * x^2 + 60 * x - 750 = 0 ∧ 24 * y^2 + 60 * y - 750 = 0 ∧ x ≠ y) → x * y = -125 / 4) :=
by
  intro x y h
  sorry

end product_of_roots_eq_neg_125_over_4_l113_113138


namespace final_student_count_l113_113588

def initial_students := 150
def students_joined := 30
def students_left := 15

theorem final_student_count : initial_students + students_joined - students_left = 165 := by
  sorry

end final_student_count_l113_113588


namespace cupcake_cost_l113_113475

def initialMoney : ℝ := 20
def moneyFromMother : ℝ := 2 * initialMoney
def totalMoney : ℝ := initialMoney + moneyFromMother
def costPerBoxOfCookies : ℝ := 3
def numberOfBoxesOfCookies : ℝ := 5
def costOfCookies : ℝ := costPerBoxOfCookies * numberOfBoxesOfCookies
def moneyAfterCookies : ℝ := totalMoney - costOfCookies
def moneyLeftAfterCupcakes : ℝ := 30
def numberOfCupcakes : ℝ := 10

noncomputable def costPerCupcake : ℝ := 
  (moneyAfterCookies - moneyLeftAfterCupcakes) / numberOfCupcakes

theorem cupcake_cost :
  costPerCupcake = 1.50 :=
by 
  sorry

end cupcake_cost_l113_113475


namespace increasing_inverse_relation_l113_113416

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry -- This is the inverse function f^-1

theorem increasing_inverse_relation {a b c : ℝ} 
  (h_inc_f : ∀ x y, x < y → f x < f y)
  (h_inc_f_inv : ∀ x y, x < y → f_inv x < f_inv y)
  (h_f3 : f 3 = 0)
  (h_f2 : f 2 = a)
  (h_f_inv2 : f_inv 2 = b)
  (h_f_inv0 : f_inv 0 = c) :
  b > c ∧ c > a := sorry

end increasing_inverse_relation_l113_113416


namespace probability_of_green_ball_l113_113128

-- Define the number of balls in each container
def number_balls_I := (10, 5)  -- (red, green)
def number_balls_II := (3, 6)  -- (red, green)
def number_balls_III := (3, 6)  -- (red, green)

-- Define the probability of selecting each container
noncomputable def probability_container_selected := (1 / 3 : ℝ)

-- Define the probability of drawing a green ball from each container
noncomputable def probability_green_I := (number_balls_I.snd : ℝ) / ((number_balls_I.fst + number_balls_I.snd) : ℝ)
noncomputable def probability_green_II := (number_balls_II.snd : ℝ) / ((number_balls_II.fst + number_balls_II.snd) : ℝ)
noncomputable def probability_green_III := (number_balls_III.snd : ℝ) / ((number_balls_III.fst + number_balls_III.snd) : ℝ)

-- Define the combined probabilities for drawing a green ball and selecting each container
noncomputable def combined_probability_I := probability_container_selected * probability_green_I
noncomputable def combined_probability_II := probability_container_selected * probability_green_II
noncomputable def combined_probability_III := probability_container_selected * probability_green_III

-- Define the total probability of drawing a green ball
noncomputable def total_probability_green := combined_probability_I + combined_probability_II + combined_probability_III

-- The theorem to be proved
theorem probability_of_green_ball : total_probability_green = (5 / 9 : ℝ) :=
by
  sorry

end probability_of_green_ball_l113_113128


namespace find_x_l113_113970

theorem find_x (x : ℕ) : (x % 7 = 0) ∧ (x^2 > 200) ∧ (x < 30) ↔ (x = 21 ∨ x = 28) :=
by
  sorry

end find_x_l113_113970


namespace share_total_l113_113110

theorem share_total (A B C : ℕ) (A_ratio B_ratio C_ratio : ℕ) (Amanda_share : ℕ) 
  (h_ratio : A_ratio = 2) (h_ratio_B : B_ratio = 3) (h_ratio_C : C_ratio = 8) (h_A_share : Amanda_share = 30) 
  (h_A : A = Amanda_share / A_ratio) : A_ratio * A + B_ratio * A + C_ratio * (A) = 195 :=
by
  have hA : A = 15 := by sorry
  have hB : B = 3 * A := by sorry
  have hC : C = 8 * A := by sorry
  have h_sum : 2 * A + 3 * A + 8 * A = 13 * A := by sorry
  rw [hA],
  exact h_ratio_C

end share_total_l113_113110


namespace triangle_area_is_64_l113_113067

noncomputable def area_triangle_lines (y : ℝ → ℝ) (x : ℝ → ℝ) (neg_x : ℝ → ℝ) : ℝ :=
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := 16
  let height := 8
  1 / 2 * base * height

theorem triangle_area_is_64 :
  let y := fun x => 8
  let x := fun y => y
  let neg_x := fun y => -y
  area_triangle_lines y x neg_x = 64 := 
by 
  sorry

end triangle_area_is_64_l113_113067


namespace inequality_solutions_l113_113761

theorem inequality_solutions (a : ℝ) (h_pos : 0 < a) 
  (h_ineq_1 : ∃! x : ℕ, 10 < a ^ x ∧ a ^ x < 100) : ∃! x : ℕ, 100 < a ^ x ∧ a ^ x < 1000 :=
by
  sorry

end inequality_solutions_l113_113761


namespace arrangement_of_volunteers_l113_113462

theorem arrangement_of_volunteers :
  let volunteers := {A, B, C, D, E, F}
  let elders := {甲, 乙, 丙}
  let pairs := { (s : elders × fin 2 → volunteers) //
    ∀ (a : elders), ∃ (v1 v2 : volunteers), v1 ≠ v2 ∧ (s a 0 = v1 ∧ s a 1 = v2)}
  -- Given conditions
  (∀ s : pairs, s 甲 0 ≠ A ∧ s 甲 1 ≠ A) ∧
  (∀ s : pairs, s 乙 0 ≠ B ∧ s 乙 1 ≠ B)
  -- Total arrangements
  → finset.card pairs = 42 := 
sorry

end arrangement_of_volunteers_l113_113462


namespace mutually_exclusive_pair2_complementary_pair2_mutually_exclusive_pair4_complementary_pair4_l113_113754

def card_is_heart (c : ℕ) := c ≥ 1 ∧ c ≤ 13

def card_is_diamond (c : ℕ) := c ≥ 14 ∧ c ≤ 26

def card_is_red (c : ℕ) := c ≥ 1 ∧ c ≤ 26

def card_is_black (c : ℕ) := c ≥ 27 ∧ c ≤ 52

def card_is_face_234610 (c : ℕ) := c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 6 ∨ c = 10

def card_is_face_2345678910 (c : ℕ) :=
  c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6 ∨ c = 7 ∨ c = 8 ∨ c = 9 ∨ c = 10

def card_is_face_AKQJ (c : ℕ) :=
  c = 1 ∨ c = 11 ∨ c = 12 ∨ c = 13

def card_is_ace_king_queen_jack (c : ℕ) := c = 1 ∨ (c ≥ 11 ∧ c ≤ 13)

theorem mutually_exclusive_pair2 : ∀ c : ℕ, card_is_red c ≠ card_is_black c := by
  sorry

theorem complementary_pair2 : ∀ c : ℕ, card_is_red c ∨ card_is_black c := by
  sorry

theorem mutually_exclusive_pair4 : ∀ c : ℕ, card_is_face_2345678910 c ≠ card_is_ace_king_queen_jack c := by
  sorry

theorem complementary_pair4 : ∀ c : ℕ, card_is_face_2345678910 c ∨ card_is_ace_king_queen_jack c := by
  sorry

end mutually_exclusive_pair2_complementary_pair2_mutually_exclusive_pair4_complementary_pair4_l113_113754


namespace least_number_to_subtract_l113_113443

theorem least_number_to_subtract (n : ℕ) (h : n = 42739) : 
    ∃ k, k = 4 ∧ (n - k) % 15 = 0 := by
  sorry

end least_number_to_subtract_l113_113443


namespace sphere_surface_area_of_circumscribing_cuboid_l113_113887

theorem sphere_surface_area_of_circumscribing_cuboid :
  ∀ (a b c : ℝ), a = 5 ∧ b = 4 ∧ c = 3 → 4 * Real.pi * ((Real.sqrt ((a^2 + b^2 + c^2)) / 2) ^ 2) = 50 * Real.pi :=
by
  -- introduction of variables and conditions
  intros a b c h
  obtain ⟨_, _, _⟩ := h -- decomposing the conditions
  -- the proof is skipped
  sorry

end sphere_surface_area_of_circumscribing_cuboid_l113_113887


namespace point_P_coordinates_l113_113178

-- Definitions based on conditions
def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs P.2 = d

def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs P.1 = d

-- The theorem statement based on the proof problem
theorem point_P_coordinates :
  ∃ P : ℝ × ℝ, 
    in_fourth_quadrant P ∧ 
    distance_to_x_axis P 2 ∧ 
    distance_to_y_axis P 3 ∧ 
    P = (3, -2) :=
by
  sorry

end point_P_coordinates_l113_113178


namespace problem_statement_l113_113513

noncomputable def a : ℝ := (Real.tan 23) / (1 - (Real.tan 23) ^ 2)
noncomputable def b : ℝ := 2 * Real.sin 13 * Real.cos 13
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos 50) / 2)

theorem problem_statement : c < b ∧ b < a :=
by
  -- Proof omitted
  sorry

end problem_statement_l113_113513


namespace three_digit_diff_l113_113034

theorem three_digit_diff (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) :
  ∃ d : ℕ, d = a - b ∧ (d < 10 ∨ (10 ≤ d ∧ d < 100) ∨ (100 ≤ d ∧ d < 1000)) :=
sorry

end three_digit_diff_l113_113034


namespace equation_of_line_l_equations_of_line_m_l113_113954

-- Define the point P and condition for line l
def P := (2, (7 : ℚ)/4)
def l_slope : ℚ := 3 / 4

-- Define the given equation form and conditions for line l
def condition_l (x y : ℚ) : Prop := y - (7 / 4) = (3 / 4) * (x - 2)
def equation_l (x y : ℚ) : Prop := 3 * x - 4 * y = 5

theorem equation_of_line_l :
  ∀ x y : ℚ, condition_l x y → equation_l x y :=
sorry

-- Define the distance condition for line m
def equation_m (x y n : ℚ) : Prop := 3 * x - 4 * y + n = 0
def distance_condition_m (n : ℚ) : Prop := 
  |(-1 + n : ℚ)| / 5 = 3

theorem equations_of_line_m :
  ∃ n : ℚ, distance_condition_m n ∧ (equation_m 2 (7/4) n) ∨ 
            equation_m 2 (7/4) (-14) :=
sorry

end equation_of_line_l_equations_of_line_m_l113_113954


namespace total_candies_count_l113_113367

variable (purple_candies orange_candies yellow_candies : ℕ)

theorem total_candies_count
  (ratio_condition : purple_candies / orange_candies = 2 / 4 ∧ purple_candies / yellow_candies = 2 / 5)
  (yellow_candies_count : yellow_candies = 40) :
  purple_candies + orange_candies + yellow_candies = 88 :=
by
  sorry

end total_candies_count_l113_113367


namespace winner_beats_by_16_secons_l113_113723

-- Definitions of the times for mathematician and physicist
variables (x y : ℕ)

-- Conditions based on the given problem
def condition1 := 2 * y - x = 24
def condition2 := 2 * x - y = 72

-- The statement to prove
theorem winner_beats_by_16_secons (h1 : condition1 x y) (h2 : condition2 x y) : 2 * x - 2 * y = 16 := 
sorry

end winner_beats_by_16_secons_l113_113723


namespace coefficient_x3_expansion_l113_113412

open Finset -- To use binomial coefficients and summation

theorem coefficient_x3_expansion (x : ℝ) : 
  (2 + x) ^ 3 = 8 + 12 * x + 6 * x^2 + 1 * x^3 :=
by
  sorry

end coefficient_x3_expansion_l113_113412


namespace find_a_and_max_value_l113_113956

noncomputable def f (x a : ℝ) := 2 * x^3 - 6 * x^2 + a

theorem find_a_and_max_value :
  (∃ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f x a ≥ 0) ∧ (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f x a ≤ 3)) :=
by
  sorry

end find_a_and_max_value_l113_113956


namespace find_f_3_l113_113840

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l113_113840


namespace probability_event_A_occurs_1400_times_2400_trials_l113_113606

noncomputable theory

-- Definitions for the problem
def num_trials : ℕ := 2400
def successes : ℕ := 1400
def prob_success : ℝ := 0.6
def prob_failure : ℝ := 1 - prob_success

-- Standard normal density function
def std_normal_density (x : ℝ) : ℝ :=
  (1 / (real.sqrt (2 * real.pi))) * real.exp (- (x ^ 2 / 2))

-- De Moivre-Laplace theorem formula for binomial distribution
def de_moivre_laplace (n k : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - p in
  let x := (k - n * p) / real.sqrt (n * p * q) in
  (1 / real.sqrt (n * p * q)) * std_normal_density x

theorem probability_event_A_occurs_1400_times_2400_trials :
  de_moivre_laplace num_trials successes prob_success = 0.0041 := by
  sorry

end probability_event_A_occurs_1400_times_2400_trials_l113_113606


namespace find_b_l113_113219

-- Define what it means for b to be a solution
def is_solution (b : ℤ) : Prop :=
  b > 4 ∧ ∃ k : ℤ, 4 * b + 5 = k * k

-- State the problem
theorem find_b : ∃ b : ℤ, is_solution b ∧ ∀ b' : ℤ, is_solution b' → b' ≥ 5 := by
  sorry

end find_b_l113_113219


namespace binom_19_10_l113_113917

theorem binom_19_10 : 
  (nat.choose 19 10) = 92378 :=
by
  have h1 : (nat.choose 17 7) = 19448 := by sorry
  have h2 : (nat.choose 17 9) = 24310 := by sorry
  sorry

end binom_19_10_l113_113917


namespace circle_tangent_l113_113611

theorem circle_tangent {m : ℝ} (h : ∃ (x y : ℝ), (x - 3)^2 + (y - 4)^2 = 25 - m ∧ x^2 + y^2 = 1) :
  m = 9 :=
sorry

end circle_tangent_l113_113611


namespace compare_negatives_l113_113910

theorem compare_negatives : - (1 : ℝ) / 2 < - (1 : ℝ) / 3 :=
by
  -- We start with the fact that 1/2 > 1/3
  have h : (1 : ℝ) / 2 > (1 : ℝ) / 3 := by
    linarith
    
  -- Negating the inequality reverses the sign
  exact neg_lt_neg_iff.mpr h

end compare_negatives_l113_113910


namespace min_ratio_ax_l113_113642

theorem min_ratio_ax (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) 
: y^2 - 1 = a^2 * (x^2 - 1) → ∃ (k : ℕ), k = 2 ∧ (a = k * x) := 
sorry

end min_ratio_ax_l113_113642


namespace smallest_positive_multiple_of_45_is_45_l113_113245

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113245


namespace smallest_positive_multiple_of_45_l113_113255

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l113_113255


namespace obtuse_triangle_k_values_l113_113539

theorem obtuse_triangle_k_values (k : ℕ) (h : k > 0) :
  (∃ k, (5 < k ∧ k ≤ 12) ∨ (21 ≤ k ∧ k < 29)) → ∃ n : ℕ, n = 15 :=
by
  sorry

end obtuse_triangle_k_values_l113_113539


namespace data_plan_comparison_l113_113458

theorem data_plan_comparison : ∃ (m : ℕ), 500 < m :=
by
  let cost_plan_x (m : ℕ) : ℕ := 15 * m
  let cost_plan_y (m : ℕ) : ℕ := 2500 + 10 * m
  use 501
  have h : 500 < 501 := by norm_num
  exact h

end data_plan_comparison_l113_113458


namespace lines_parallel_iff_a_eq_neg2_l113_113349

def line₁_eq (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def line₂_eq (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y - 1 = 0

theorem lines_parallel_iff_a_eq_neg2 (a : ℝ) :
  (∀ x y : ℝ, line₁_eq a x y → line₂_eq a x y) ↔ a = -2 :=
by sorry

end lines_parallel_iff_a_eq_neg2_l113_113349


namespace term_sequence_10th_l113_113963

theorem term_sequence_10th :
  let a (n : ℕ) := (-1:ℚ)^(n+1) * (2*n)/(2*n + 1)
  a 10 = -20/21 := 
by
  sorry

end term_sequence_10th_l113_113963


namespace adam_spent_on_ferris_wheel_l113_113884

-- Define the conditions
def ticketsBought : Nat := 13
def ticketsLeft : Nat := 4
def costPerTicket : Nat := 9

-- Define the question and correct answer as a proof goal
theorem adam_spent_on_ferris_wheel : (ticketsBought - ticketsLeft) * costPerTicket = 81 := by
  sorry

end adam_spent_on_ferris_wheel_l113_113884


namespace count_x_values_l113_113921

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x 
  else if n = 1 then 3000
  else 1 / (sequence x (n - 1)) - 1 / (sequence x (n - 2))

theorem count_x_values (x : ℝ) (n : ℕ) : ℕ :=
  ∃ x : ℝ, ∃ n : ℕ, x > 0 ∧ sequence x n = 3001 :=
begin
  sorry  -- Proof to be filled in later
end

end count_x_values_l113_113921


namespace area_of_pentagon_eq_fraction_l113_113855

theorem area_of_pentagon_eq_fraction (w : ℝ) (h : ℝ) (fold_x : ℝ) (fold_y : ℝ)
    (hw3 : h = 3 * w)
    (hfold : fold_x = fold_y)
    (hx : fold_x ^ 2 + fold_y ^ 2 = 3 ^ 2)
    (hx_dist : fold_x = 4 / 3) :
  (3 * (1 / 2) + fold_x / 2) / (3 * w) = 13 / 18 := 
by 
  sorry

end area_of_pentagon_eq_fraction_l113_113855


namespace find_f_3_l113_113844

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l113_113844


namespace standard_deviation_is_2point5_l113_113410

noncomputable def mean : ℝ := 17.5
noncomputable def given_value : ℝ := 12.5

theorem standard_deviation_is_2point5 :
  ∀ (σ : ℝ), mean - 2 * σ = given_value → σ = 2.5 := by
  sorry

end standard_deviation_is_2point5_l113_113410


namespace skylar_starting_age_l113_113028

-- Conditions of the problem
def annual_donation : ℕ := 8000
def current_age : ℕ := 71
def total_amount_donated : ℕ := 440000

-- Question and proof statement
theorem skylar_starting_age :
  (current_age - total_amount_donated / annual_donation) = 16 := 
by
  sorry

end skylar_starting_age_l113_113028


namespace sequence_x_values_3001_l113_113927

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end sequence_x_values_3001_l113_113927


namespace binom_19_10_l113_113916

theorem binom_19_10 : 
  (nat.choose 19 10) = 92378 :=
by
  have h1 : (nat.choose 17 7) = 19448 := by sorry
  have h2 : (nat.choose 17 9) = 24310 := by sorry
  sorry

end binom_19_10_l113_113916


namespace machine_A_sprockets_per_hour_l113_113794

-- Definitions based on the problem conditions
def MachineP_time (A : ℝ) (T : ℝ) : ℝ := T + 10
def MachineQ_rate (A : ℝ) : ℝ := 1.1 * A
def MachineP_sprockets (A : ℝ) (T : ℝ) : ℝ := A * (T + 10)
def MachineQ_sprockets (A : ℝ) (T : ℝ) : ℝ := 1.1 * A * T

-- Lean proof statement to prove that Machine A produces 8 sprockets per hour
theorem machine_A_sprockets_per_hour :
  ∀ A T : ℝ, 
  880 = MachineP_sprockets A T ∧
  880 = MachineQ_sprockets A T →
  A = 8 :=
by
  intros A T h
  have h1 : 880 = MachineP_sprockets A T := h.left
  have h2 : 880 = MachineQ_sprockets A T := h.right
  sorry

end machine_A_sprockets_per_hour_l113_113794


namespace find_first_term_and_difference_l113_113051

-- Define the two conditions given in the problem.
def a3 : ℕ → ℝ := λ n, 3
def a11 : ℕ → ℝ := λ n, 15

-- Define the arithmetic sequence using the formula a_n = a_1 + (n-1) * d
def arithmetic_seq (a1 d : ℝ) : ℕ → ℝ := λ n, a1 + (n - 1) * d

-- State the theorem with the conditions and the goal (proof not provided, using sorry).
theorem find_first_term_and_difference :
  (∃ a1 d : ℝ, arithmetic_seq a1 d 3 = 3 ∧ arithmetic_seq a1 d 11 = 15 ∧ a1 = 0 ∧ d = 3 / 2) := 
sorry

end find_first_term_and_difference_l113_113051


namespace sqrt_condition_iff_l113_113413

theorem sqrt_condition_iff (x : ℝ) : (∃ y : ℝ, y = (2 * x + 3) ∧ (0 ≤ y)) ↔ (x ≥ -3 / 2) :=
by sorry

end sqrt_condition_iff_l113_113413


namespace smallest_positive_multiple_of_45_l113_113285

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l113_113285


namespace fractional_expression_evaluation_l113_113965

theorem fractional_expression_evaluation
  (m n r t : ℚ)
  (h1 : m / n = 4 / 3)
  (h2 : r / t = 9 / 14) :
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -11 / 14 := by
  sorry

end fractional_expression_evaluation_l113_113965


namespace solve_for_s_l113_113404

theorem solve_for_s {x : ℝ} (h : 4 * x^2 - 8 * x - 320 = 0) : ∃ s, s = 81 :=
by 
  -- Introduce the conditions and the steps
  sorry

end solve_for_s_l113_113404


namespace brookdale_avg_temp_l113_113104

def highs : List ℤ := [51, 64, 60, 59, 48, 55]
def lows : List ℤ := [42, 49, 47, 43, 41, 44]

def average_temperature : ℚ :=
  let total_sum := highs.sum + lows.sum
  let count := (highs.length + lows.length : ℚ)
  total_sum / count

theorem brookdale_avg_temp :
  average_temperature = 49.4 :=
by
  -- The proof goes here
  sorry

end brookdale_avg_temp_l113_113104


namespace probability_non_adjacent_sum_l113_113888

-- Definitions and conditions from the problem
def total_trees := 13
def maple_trees := 4
def oak_trees := 3
def birch_trees := 6

-- Total possible arrangements of 13 trees
def total_arrangements := Nat.choose total_trees birch_trees

-- Number of ways to arrange birch trees with no two adjacent
def favorable_arrangements := Nat.choose (maple_trees + oak_trees + 1) birch_trees

-- Probability calculation
def probability_non_adjacent := (favorable_arrangements : ℚ) / (total_arrangements : ℚ)

-- This value should be simplified to form m/n in lowest terms
def fraction_part_m := 7
def fraction_part_n := 429

-- Verify m + n
def sum_m_n := fraction_part_m + fraction_part_n

-- Check that sum_m_n is equal to 436
theorem probability_non_adjacent_sum :
  sum_m_n = 436 := by {
    -- Placeholder proof
    sorry
}

end probability_non_adjacent_sum_l113_113888


namespace stools_count_l113_113870

theorem stools_count : ∃ x y : ℕ, 3 * x + 4 * y = 39 ∧ x = 3 := 
by
  sorry

end stools_count_l113_113870


namespace hotel_r_charge_percentage_l113_113556

-- Let P, R, and G be the charges for a single room at Hotels P, R, and G respectively
variables (P R G : ℝ)

-- Given conditions:
-- 1. The charge for a single room at Hotel P is 55% less than the charge for a single room at Hotel R.
-- 2. The charge for a single room at Hotel P is 10% less than the charge for a single room at Hotel G.
axiom h1 : P = 0.45 * R
axiom h2 : P = 0.90 * G

-- The charge for a single room at Hotel R is what percent greater than the charge for a single room at Hotel G.
theorem hotel_r_charge_percentage : (R - G) / G * 100 = 100 :=
sorry

end hotel_r_charge_percentage_l113_113556


namespace roots_difference_l113_113463

theorem roots_difference :
  let a := 2 
  let b := 5 
  let c := -12
  let disc := b*b - 4*a*c
  let root1 := (-b + Real.sqrt disc) / (2 * a)
  let root2 := (-b - Real.sqrt disc) / (2 * a)
  let larger_root := max root1 root2
  let smaller_root := min root1 root2
  larger_root - smaller_root = 5.5 := by
  sorry

end roots_difference_l113_113463


namespace necklace_cost_l113_113985

def bead_necklaces := 3
def gemstone_necklaces := 3
def total_necklaces := bead_necklaces + gemstone_necklaces
def total_earnings := 36

theorem necklace_cost :
  (total_earnings / total_necklaces) = 6 :=
by
  -- Proof goes here
  sorry

end necklace_cost_l113_113985


namespace simpsons_hats_l113_113733

variable (S : ℕ)
variable (O : ℕ)

-- Define the conditions: O'Brien's hats before losing one
def obriens_hats_before : Prop := O = 2 * S + 5

-- Define the current number of O'Brien's hats
def obriens_current_hats : Prop := O = 34 + 1

-- Main theorem statement
theorem simpsons_hats : obriens_hats_before S O ∧ obriens_current_hats O → S = 15 := 
by
  sorry

end simpsons_hats_l113_113733


namespace function_form_l113_113747

theorem function_form (f : ℕ → ℕ) (H : ∀ (x y z : ℕ), x ≠ y → y ≠ z → z ≠ x → (∃ k : ℕ, x + y + z = k^2 ↔ ∃ m : ℕ, f x + f y + f z = m^2)) : ∃ k : ℕ, ∀ n : ℕ, f n = k^2 * n :=
by
  sorry

end function_form_l113_113747


namespace binomial_coeff_arith_seq_expansion_l113_113947

open BigOperators

-- Given the binomial expansion of (sqrt(x) + 2/sqrt(x))^n
-- we need to prove that the condition on binomial coefficients
-- implies that n = 7, and the expansion contains no constant term.
theorem binomial_coeff_arith_seq_expansion (x : ℝ) (n : ℕ) :
  (2 * Nat.choose n 2 = Nat.choose n 1 + Nat.choose n 3) ↔ n = 7 ∧ ∀ r : ℕ, x ^ (7 - 2 * r) / 2 ≠ x ^ 0 := by
  sorry

end binomial_coeff_arith_seq_expansion_l113_113947


namespace intersection_A_B_l113_113614

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | x < -1 ∨ x > 4}

theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | 4 < x ∧ x < 7} :=
by
  sorry

end intersection_A_B_l113_113614


namespace molecular_weight_4_benzoic_acid_l113_113431

def benzoic_acid_molecular_weight : Float := (7 * 12.01) + (6 * 1.008) + (2 * 16.00)

def molecular_weight_4_moles_benzoic_acid (molecular_weight : Float) : Float := molecular_weight * 4

theorem molecular_weight_4_benzoic_acid :
  molecular_weight_4_moles_benzoic_acid benzoic_acid_molecular_weight = 488.472 :=
by
  unfold molecular_weight_4_moles_benzoic_acid benzoic_acid_molecular_weight
  -- rest of the proof
  sorry

end molecular_weight_4_benzoic_acid_l113_113431


namespace arithmetic_sequence_problem_l113_113148

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h1 : a 2 + a 3 + a 4 = 15)
  (h2 : (a 1 + 2) * (a 6 + 16) = (a 3 + 4) ^ 2)
  (h_positive : ∀ n, 0 < a n) :
  a 10 = 19 :=
sorry

end arithmetic_sequence_problem_l113_113148


namespace factorial_division_l113_113121

theorem factorial_division :
  52! / 50! = 2652 := by
  sorry

end factorial_division_l113_113121


namespace arithmetic_sequence_a5_l113_113633

variable (a : ℕ → ℝ) (h : a 1 + a 9 = 10)

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : a 1 + a 9 = 10) : 
  a 5 = 5 :=
by sorry

end arithmetic_sequence_a5_l113_113633


namespace statement_A_statement_B_statement_C_statement_D_statement_E_l113_113130

def diamond (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem statement_A : ∀ (x y : ℝ), diamond x y = diamond y x := sorry

theorem statement_B : ∀ (x y : ℝ), 2 * (diamond x y) ≠ diamond (2 * x) (2 * y) := sorry

theorem statement_C : ∀ (x : ℝ), diamond x 0 = x^2 := sorry

theorem statement_D : ∀ (x : ℝ), diamond x x = 0 := sorry

theorem statement_E : ∀ (x y : ℝ), x = y → diamond x y = 0 := sorry

end statement_A_statement_B_statement_C_statement_D_statement_E_l113_113130


namespace part_a_part_b_l113_113328

noncomputable def triangle_exists (h1 h2 h3 : ℕ) : Prop :=
  ∃ a b c, 2 * a = h1 * (b + c) ∧ 2 * b = h2 * (a + c) ∧ 2 * c = h3 * (a + b)

theorem part_a : ¬ triangle_exists 2 3 6 :=
sorry

theorem part_b : triangle_exists 2 3 5 :=
sorry

end part_a_part_b_l113_113328


namespace restaurant_sodas_l113_113727

theorem restaurant_sodas (M : ℕ) (h1 : M + 19 = 96) : M = 77 :=
by
  sorry

end restaurant_sodas_l113_113727


namespace find_f_of_3_l113_113830

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l113_113830


namespace relative_magnitude_of_reciprocal_l113_113167

theorem relative_magnitude_of_reciprocal 
  (a b : ℝ) (hab : a < 1 / b) :
  (a > 0 ∧ b > 0 ∧ 1 / a > b) ∨ (a < 0 ∧ b < 0 ∧ 1 / a > b)
   ∨ (a > 0 ∧ b < 0 ∧ 1 / a < b) ∨ (a < 0 ∧ b > 0 ∧ 1 / a < b) :=
by sorry

end relative_magnitude_of_reciprocal_l113_113167


namespace number_of_divisors_of_36_l113_113160

theorem number_of_divisors_of_36 :  
  let n := 36
  number_of_divisors n = 9 := 
by 
  sorry

end number_of_divisors_of_36_l113_113160


namespace min_f_value_l113_113786

open Real

theorem min_f_value (a b c d e : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) :
    ∃ (x : ℝ), (∀ y : ℝ, (|y - a| + |y - b| + |y - c| + |y - d| + |y - e|) ≥ -a - b + d + e) ∧ 
    (|x - a| + |x - b| + |x - c| + |x - d| + |x - e| = -a - b + d + e) :=
sorry

end min_f_value_l113_113786


namespace sum_reciprocal_l113_113422

-- Definition of the problem
theorem sum_reciprocal (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 4 * x * y) : 
  (1 / x) + (1 / y) = 1 :=
sorry

end sum_reciprocal_l113_113422


namespace expansion_no_x2_term_l113_113149

theorem expansion_no_x2_term (n : ℕ) (h1 : 5 ≤ n) (h2 : n ≤ 8) :
  ¬ ∃ (r : ℕ), 0 ≤ r ∧ r ≤ n ∧ n - 4 * r = 2 → n = 7 := by
  sorry

end expansion_no_x2_term_l113_113149


namespace total_cans_given_away_l113_113903

noncomputable def total_cans_initial : ℕ := 2000
noncomputable def cans_taken_first_day : ℕ := 500
noncomputable def restocked_first_day : ℕ := 1500
noncomputable def people_second_day : ℕ := 1000
noncomputable def cans_per_person_second_day : ℕ := 2
noncomputable def restocked_second_day : ℕ := 3000

theorem total_cans_given_away :
  (cans_taken_first_day + (people_second_day * cans_per_person_second_day) = 2500) :=
by
  sorry

end total_cans_given_away_l113_113903


namespace circle_center_sum_l113_113594

theorem circle_center_sum (x y : ℝ) (h : (x - 5)^2 + (y - 2)^2 = 38) : x + y = 7 := 
  sorry

end circle_center_sum_l113_113594


namespace prod_is_96_l113_113941

noncomputable def prod_of_nums (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : ℝ := x * y

theorem prod_is_96 (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : prod_of_nums x y h1 h2 = 96 :=
by
  sorry

end prod_is_96_l113_113941


namespace angle_coincides_with_graph_y_eq_neg_abs_x_l113_113973

noncomputable def angle_set (α : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + 225 ∨ α = k * 360 + 315}

theorem angle_coincides_with_graph_y_eq_neg_abs_x (α : ℝ) :
  α ∈ angle_set α ↔ 
  ∃ k : ℤ, (α = k * 360 + 225 ∨ α = k * 360 + 315) :=
by
  sorry

end angle_coincides_with_graph_y_eq_neg_abs_x_l113_113973


namespace sought_circle_equation_l113_113748

def circle_passing_through_point (D E F : ℝ) : Prop :=
  ∀ (x y : ℝ), (x = 0) → (y = 2) → x^2 + y^2 + D * x + E * y + F = 0

def chord_lies_on_line (D E F : ℝ) : Prop :=
  (D + 1) / 5 = (E - 2) / 2 ∧ (D + 1) / 5 = (F + 3)

theorem sought_circle_equation :
  ∃ (D E F : ℝ), 
  circle_passing_through_point D E F ∧ 
  chord_lies_on_line D E F ∧
  (D = -6) ∧ (E = 0) ∧ (F = -4) ∧ 
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 6 * x - 4 = 0 :=
by
  sorry

end sought_circle_equation_l113_113748


namespace volume_ratio_of_smaller_snowball_l113_113301

theorem volume_ratio_of_smaller_snowball (r : ℝ) (k : ℝ) :
  let V₀ := (4/3) * π * r^3
  let S := 4 * π * r^2
  let V_large := (4/3) * π * (2 * r)^3
  let V_large_half := V_large / 2
  let new_r := (V_large_half / ((4/3) * π))^(1/3)
  let reduction := 2*r - new_r
  let remaining_r := r - reduction
  let remaining_V := (4/3) * π * remaining_r^3
  let volume_ratio := remaining_V / V₀ 
  volume_ratio = 1/5 :=
by
  -- Proof goes here
  sorry

end volume_ratio_of_smaller_snowball_l113_113301


namespace number_of_sides_of_regular_polygon_l113_113569

theorem number_of_sides_of_regular_polygon (P s n : ℕ) (hP : P = 150) (hs : s = 15) (hP_formula : P = n * s) : n = 10 :=
  by {
    -- proof goes here
    sorry
  }

end number_of_sides_of_regular_polygon_l113_113569


namespace marcy_pets_cat_time_l113_113795

theorem marcy_pets_cat_time (P : ℝ) (h1 : P + (1/3)*P = 16) : P = 12 :=
by
  sorry

end marcy_pets_cat_time_l113_113795


namespace find_2n_plus_m_l113_113672

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 2 * n + m = 36 :=
sorry

end find_2n_plus_m_l113_113672


namespace min_sum_ab_72_l113_113017

theorem min_sum_ab_72 (a b : ℤ) (h : a * b = 72) : a + b ≥ -17 := sorry

end min_sum_ab_72_l113_113017


namespace martha_bottles_l113_113108

def total_bottles_left (a b c d : ℕ) : ℕ :=
  a + b + c - d

theorem martha_bottles : total_bottles_left 4 4 5 3 = 10 :=
by
  sorry

end martha_bottles_l113_113108


namespace find_minimum_value_l113_113940

theorem find_minimum_value (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : x > 0): 
  (∃ m : ℝ, ∀ x > 0, (a^2 + x^2) / x ≥ m ∧ ∃ x₀ > 0, (a^2 + x₀^2) / x₀ = m) :=
sorry

end find_minimum_value_l113_113940


namespace camden_dogs_fraction_l113_113119

def number_of_dogs (Justins_dogs : ℕ) (extra_dogs : ℕ) : ℕ := Justins_dogs + extra_dogs
def dogs_from_legs (total_legs : ℕ) (legs_per_dog : ℕ) : ℕ := total_legs / legs_per_dog
def fraction_of_dogs (dogs_camden : ℕ) (dogs_rico : ℕ) : ℚ := dogs_camden / dogs_rico

theorem camden_dogs_fraction (Justins_dogs : ℕ) (extra_dogs : ℕ) (total_legs_camden : ℕ) (legs_per_dog : ℕ) :
  Justins_dogs = 14 →
  extra_dogs = 10 →
  total_legs_camden = 72 →
  legs_per_dog = 4 →
  fraction_of_dogs (dogs_from_legs total_legs_camden legs_per_dog) (number_of_dogs Justins_dogs extra_dogs) = 3 / 4 :=
by
  sorry

end camden_dogs_fraction_l113_113119


namespace f_prime_at_2_l113_113487

noncomputable def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := f(x) / x

-- Condition 1: The point (2, 1) lies on y = f(x) / x which gives f(2) = 2 
def condition1 : Prop := f(2) = 2

-- Condition 2: The lines tangent to y = f(x) at (0,0) and y = f(x) / x at (2,1) have the same slope == 1/2
def tangent_slope : ℝ := 1/2
def g' := λ x, (x * (Deriv f x) - f x) / (x^2)
def condition2 : Prop := g'(2) = tangent_slope

-- Proving f'(2) = 2
theorem f_prime_at_2 : condition1 ∧ condition2 → (Deriv f 2) = 2 :=
by
  sorry

end f_prime_at_2_l113_113487


namespace angle_between_slant_height_and_base_l113_113103

theorem angle_between_slant_height_and_base (R : ℝ) (diam_base_upper diam_base_lower : ℝ) 
(h1 : diam_base_upper + diam_base_lower = 5 * R)
: ∃ θ : ℝ, θ = Real.arcsin (4 / 5) := 
sorry

end angle_between_slant_height_and_base_l113_113103


namespace quadratic_equal_roots_iff_a_eq_4_l113_113771

theorem quadratic_equal_roots_iff_a_eq_4 (a : ℝ) (h : ∃ x : ℝ, (a * x^2 - 4 * x + 1 = 0) ∧ (a * x^2 - 4 * x + 1 = 0)) :
  a = 4 :=
by
  sorry

end quadratic_equal_roots_iff_a_eq_4_l113_113771


namespace smallest_positive_multiple_of_45_l113_113254

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ 45 * x = 45 ∧ ∀ y, (y > 0 → 45 * y = 45 * y → y ≥ x) :=
begin
  -- Proof goes here
  sorry
end

end smallest_positive_multiple_of_45_l113_113254


namespace area_of_circle_O_l113_113652

open Real EuclideanGeometry

noncomputable def area_of_circle (O : Circle) (D E F P : Point) (h1 : D ∈ O.circumference)
  (h2 : E ∈ O.circumference) (h3 : F ∈ O.circumference) (h4 : Tangent O D P)
  (h5 : OnRay P E F) (PD : Real) (PF : Real) (angle_FPD : Real) : Real :=
  let PD := 4
  let PF := 2
  let angle_FPD := π / 3 -- 60 degrees in radians
  12 * π

theorem area_of_circle_O {O : Circle} {D E F P : Point} 
  (h1 : D ∈ O.circumference)
  (h2 : E ∈ O.circumference)
  (h3 : F ∈ O.circumference)
  (h4 : Tangent O D P)
  (h5 : OnRay P E F)
  (PD := 4)
  (PF := 2)
  (angle_FPD := π / 3) : area_of_circle O D E F P h1 h2 h3 h4 h5 PD PF angle_FPD = 12 * π := by 
  sorry

end area_of_circle_O_l113_113652


namespace train_speed_in_kmph_l113_113576

theorem train_speed_in_kmph
  (train_length : ℝ) (bridge_length : ℝ) (time_seconds : ℝ)
  (H1: train_length = 200) (H2: bridge_length = 150) (H3: time_seconds = 34.997200223982084) :
  train_length + bridge_length = 200 + 150 →
  (train_length + bridge_length) / time_seconds * 3.6 = 36 :=
sorry

end train_speed_in_kmph_l113_113576


namespace quadratic_min_value_unique_l113_113735

theorem quadratic_min_value_unique {a b c : ℝ} (h : a > 0) :
  (∀ x : ℝ, 3 * x^2 - 8 * x + 7 ≥ 3 * (4 / 3)^2 - 8 * (4 / 3) + 7) → 
  ∃ x : ℝ, x = 4 / 3 :=
by
  sorry

end quadratic_min_value_unique_l113_113735


namespace certain_number_exists_l113_113953

theorem certain_number_exists (a b : ℝ) (C : ℝ) (h1 : a ≠ b) (h2 : a + b = 4) (h3 : a * (a - 4) = C) (h4 : b * (b - 4) = C) : 
  C = -3 := 
sorry

end certain_number_exists_l113_113953


namespace inversely_directly_proportional_l113_113408

theorem inversely_directly_proportional (m n z : ℝ) (x : ℝ) (h₁ : x = 4) (hz₁ : z = 16) (hz₂ : z = 64) (hy : ∃ y : ℝ, y = n * Real.sqrt z) (hx : ∃ m y : ℝ, x = m / y^2)
: x = 1 :=
by
  sorry

end inversely_directly_proportional_l113_113408


namespace probability_drawing_black_piece_l113_113054

/-- There are 7 Go pieces in a pocket, 3 white and 4 black.
    The probability of drawing a black piece is 4/7. -/
theorem probability_drawing_black_piece (total_pieces: ℕ) (white_pieces: ℕ) (black_pieces: ℕ) :
  total_pieces = 7 → white_pieces = 3 → black_pieces = 4 → (black_pieces : ℚ) / total_pieces = 4 / 7 :=
by
  intros h1 h2 h3
  rw [h1, h3]
  norm_cast
  norm_num
  sorry

end probability_drawing_black_piece_l113_113054


namespace intersection_of_lines_l113_113478

theorem intersection_of_lines :
  ∃ x y : ℚ, 12 * x - 5 * y = 8 ∧ 10 * x + 2 * y = 20 ∧ x = 58 / 37 ∧ y = 667 / 370 :=
by
  sorry

end intersection_of_lines_l113_113478


namespace remove_two_fractions_sum_is_one_l113_113457

theorem remove_two_fractions_sum_is_one :
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let total_sum := (fractions.sum : ℚ)
  let remaining_sum := total_sum - (1/8 + 1/10)
  remaining_sum = 1 := by
    sorry

end remove_two_fractions_sum_is_one_l113_113457


namespace compare_negatives_l113_113909

theorem compare_negatives : - (1 : ℝ) / 2 < - (1 : ℝ) / 3 :=
by
  -- We start with the fact that 1/2 > 1/3
  have h : (1 : ℝ) / 2 > (1 : ℝ) / 3 := by
    linarith
    
  -- Negating the inequality reverses the sign
  exact neg_lt_neg_iff.mpr h

end compare_negatives_l113_113909


namespace ratio_P_K_is_2_l113_113522

theorem ratio_P_K_is_2 (P K M : ℝ) (r : ℝ)
  (h1: P + K + M = 153)
  (h2: P = r * K)
  (h3: P = (1/3) * M)
  (h4: M = K + 85) : r = 2 :=
  sorry

end ratio_P_K_is_2_l113_113522


namespace find_f_three_l113_113854

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l113_113854


namespace gcd_of_ten_digit_same_five_digit_l113_113574

def ten_digit_same_five_digit (n : ℕ) : Prop :=
  n > 9999 ∧ n < 100000 ∧ ∃ k : ℕ, k = n * (10^10 + 10^5 + 1)

theorem gcd_of_ten_digit_same_five_digit :
  (∀ n : ℕ, ten_digit_same_five_digit n → ∃ d : ℕ, d = 10000100001 ∧ ∀ m : ℕ, m ∣ d) := 
sorry

end gcd_of_ten_digit_same_five_digit_l113_113574


namespace basketball_scores_l113_113559

theorem basketball_scores (n : ℕ) (h : n = 7) : 
  ∃ (k : ℕ), k = 8 :=
by {
  sorry
}

end basketball_scores_l113_113559


namespace smallest_positive_multiple_of_45_l113_113230

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l113_113230


namespace tile_chessboard_2n_l113_113946

theorem tile_chessboard_2n (n : ℕ) (board : Fin (2^n) → Fin (2^n) → Prop) (i j : Fin (2^n)) 
  (h : board i j = false) : ∃ tile : Fin (2^n) → Fin (2^n) → Bool, 
  (∀ i j, board i j = true ↔ tile i j = true) :=
sorry

end tile_chessboard_2n_l113_113946


namespace travel_time_between_resorts_l113_113542

theorem travel_time_between_resorts
  (num_cars : ℕ)
  (car_interval : ℕ)
  (opposing_encounter_time : ℕ)
  (travel_time : ℕ) :
  num_cars = 80 →
  car_interval = 15 →
  (opposing_encounter_time * 2 * car_interval / travel_time) = num_cars →
  travel_time = 20 :=
by
  sorry

end travel_time_between_resorts_l113_113542


namespace households_both_brands_l113_113297

theorem households_both_brands
  (T : ℕ) (N : ℕ) (A : ℕ) (B : ℕ)
  (hT : T = 300) (hN : N = 80) (hA : A = 60) (hB : ∃ X : ℕ, B = 3 * X ∧ T = N + A + B + X) :
  ∃ X : ℕ, X = 40 :=
by
  -- Upon extracting values from conditions, solving for both brand users X = 40
  sorry

end households_both_brands_l113_113297


namespace simplified_factorial_fraction_l113_113117

theorem simplified_factorial_fraction :
  (5 * Nat.factorial 7 + 35 * Nat.factorial 6) / Nat.factorial 8 = 5 / 4 :=
by
  sorry

end simplified_factorial_fraction_l113_113117


namespace exists_three_digit_number_cube_ends_in_777_l113_113932

theorem exists_three_digit_number_cube_ends_in_777 :
  ∃ x : ℤ, 100 ≤ x ∧ x < 1000 ∧ x^3 % 1000 = 777 := 
sorry

end exists_three_digit_number_cube_ends_in_777_l113_113932


namespace arithmetic_sequence_geometric_subsequence_l113_113612

theorem arithmetic_sequence_geometric_subsequence (a : ℕ → ℤ) (a1 a3 a4 : ℤ) (d : ℤ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d = 2)
  (h3 : a1 = a 1)
  (h4 : a3 = a 3)
  (h5 : a4 = a 4)
  (h6 : a3^2 = a1 * a4) :
  a 6 = 2 := 
by
  sorry

end arithmetic_sequence_geometric_subsequence_l113_113612


namespace find_x_values_for_3001_l113_113919

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
match n with
| 0 => x
| 1 => 3000
| 2 => 3001 / x
| 3 => (3000 + x) / (3000 * x)
| 4 => (x + 1) / 3000
| 5 => x
| _ => 3000

theorem find_x_values_for_3001 :
  {x : ℝ | x > 0 ∧ x = 3001 ∨ x = 1 ∨ x = 9002999}.card = 3 :=
begin
  sorry
end

end find_x_values_for_3001_l113_113919


namespace ponchik_ate_62_cubes_l113_113346

def has_odd_neighbors (neighbors : ℕ) : Prop := neighbors % 2 = 1

def cube_positions : List (ℕ × ℕ × ℕ) := 
  List.product (List.finRange 5) (List.product (List.finRange 5) (List.finRange 5))

def count_odd_neighbor_cubes : ℕ :=
  cube_positions.foldr (λ pos acc, 
    let ⟨x, (y, z)⟩ := pos in
    let neighbors := 
      (if x > 0 then 1 else 0) + (if x < 4 then 1 else 0) + 
      (if y > 0 then 1 else 0) + (if y < 4 then 1 else 0) +
      (if z > 0 then 1 else 0) + (if z < 4 then 1 else 0) 
    in if has_odd_neighbors neighbors then acc + 1 else acc) 0

theorem ponchik_ate_62_cubes : count_odd_neighbor_cubes = 62 := by 
  sorry

end ponchik_ate_62_cubes_l113_113346


namespace negative_half_less_than_negative_third_l113_113908

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end negative_half_less_than_negative_third_l113_113908


namespace jori_water_left_l113_113639

theorem jori_water_left (initial used : ℚ) (h1 : initial = 3) (h2 : used = 4 / 3) :
  initial - used = 5 / 3 :=
by
  sorry

end jori_water_left_l113_113639


namespace sixth_bar_placement_l113_113055

theorem sixth_bar_placement (f : ℕ → ℕ) (h1 : f 1 = 1) (h2 : f 2 = 121) :
  (∃ n, f 6 = n ∧ (n = 16 ∨ n = 46 ∨ n = 76 ∨ n = 106)) :=
sorry

end sixth_bar_placement_l113_113055


namespace not_snowing_next_five_days_l113_113689

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end not_snowing_next_five_days_l113_113689


namespace determine_percentage_of_second_mixture_l113_113298

-- Define the given conditions and question
def mixture_problem (P : ℝ) : Prop :=
  ∃ (V1 V2 : ℝ) (A1 A2 A_final : ℝ),
  V1 = 2.5 ∧ A1 = 0.30 ∧
  V2 = 7.5 ∧ A2 = P / 100 ∧
  A_final = 0.45 ∧
  (V1 * A1 + V2 * A2) / (V1 + V2) = A_final

-- State the theorem
theorem determine_percentage_of_second_mixture : mixture_problem 50 := sorry

end determine_percentage_of_second_mixture_l113_113298


namespace initial_percentage_of_salt_l113_113112

theorem initial_percentage_of_salt (P : ℝ) :
  (P / 100) * 80 = 8 → P = 10 :=
by
  intro h
  sorry

end initial_percentage_of_salt_l113_113112


namespace bill_difference_proof_l113_113895

variable (a b c : ℝ)

def alice_condition := (25/100) * a = 5
def bob_condition := (20/100) * b = 6
def carol_condition := (10/100) * c = 7

theorem bill_difference_proof (ha : alice_condition a) (hb : bob_condition b) (hc : carol_condition c) :
  max a (max b c) - min a (min b c) = 50 :=
by sorry

end bill_difference_proof_l113_113895


namespace proportion_red_MMs_l113_113897

theorem proportion_red_MMs (R B : ℝ) (h1 : R + B = 1) 
  (h2 : R * (4 / 5) = B * (1 / 6)) :
  R = 5 / 29 :=
by
  sorry

end proportion_red_MMs_l113_113897


namespace cost_per_meter_l113_113856

def length_of_plot : ℝ := 75
def cost_of_fencing : ℝ := 5300

-- Define breadth as a variable b
def breadth_of_plot (b : ℝ) : Prop := length_of_plot = b + 50

-- Calculate the perimeter given the known breadth
def perimeter (b : ℝ) : ℝ := 2 * length_of_plot + 2 * b

-- Define the proof problem
theorem cost_per_meter (b : ℝ) (hb : breadth_of_plot b) : 5300 / (perimeter b) = 26.5 := by
  -- Given hb: length_of_plot = b + 50, perimeter calculation follows
  sorry

end cost_per_meter_l113_113856


namespace sum_of_integers_with_even_product_l113_113364

theorem sum_of_integers_with_even_product (a b : ℤ) (h : ∃ k, a * b = 2 * k) : 
∃ k1 k2, a = 2 * k1 ∨ a = 2 * k1 + 1 ∧ (a + b = 2 * k2 ∨ a + b = 2 * k2 + 1) :=
by
  sorry

end sum_of_integers_with_even_product_l113_113364


namespace smallest_positive_multiple_of_45_l113_113274

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l113_113274


namespace ratio_of_books_on_each_table_l113_113718

-- Define the conditions
variables (number_of_tables number_of_books : ℕ)
variables (R : ℕ) -- Ratio we need to find

-- State the conditions
def conditions := (number_of_tables = 500) ∧ (number_of_books = 100000)

-- Mathematical Problem Statement
theorem ratio_of_books_on_each_table (h : conditions number_of_tables number_of_books) :
    100000 = 500 * R → R = 200 :=
by
  sorry

end ratio_of_books_on_each_table_l113_113718


namespace smallest_positive_multiple_of_45_is_45_l113_113244

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113244


namespace green_ball_probability_l113_113125

-- Defining the number of red and green balls in each container
def containerI_red : ℕ := 10
def containerI_green : ℕ := 5

def containerII_red : ℕ := 3
def containerII_green : ℕ := 6

def containerIII_red : ℕ := 3
def containerIII_green : ℕ := 6

-- Probability of selecting any container
def prob_container : ℚ := 1 / 3

-- Defining the probabilities of drawing a green ball from each container
def prob_green_I : ℚ := containerI_green / (containerI_red + containerI_green)
def prob_green_II : ℚ := containerII_green / (containerII_red + containerII_green)
def prob_green_III : ℚ := containerIII_green / (containerIII_red + containerIII_green)

-- Law of total probability
def prob_green_total : ℚ :=
  prob_container * prob_green_I +
  prob_container * prob_green_II +
  prob_container * prob_green_III

-- The mathematical statement to be proven
theorem green_ball_probability :
  prob_green_total = 5 / 9 := by
  sorry

end green_ball_probability_l113_113125


namespace chess_games_won_l113_113717

theorem chess_games_won (W L : ℕ) (h1 : W + L = 44) (h2 : 4 * L = 7 * W) : W = 16 :=
by
  sorry

end chess_games_won_l113_113717


namespace smallest_positive_integer_ends_in_7_and_divisible_by_5_l113_113433

theorem smallest_positive_integer_ends_in_7_and_divisible_by_5 : 
  ∃ n : ℤ, n > 0 ∧ n % 10 = 7 ∧ n % 5 = 0 ∧ n = 37 := 
by 
  sorry

end smallest_positive_integer_ends_in_7_and_divisible_by_5_l113_113433


namespace graph_symmetric_monotonicity_interval_false_max_value_one_min_value_a_l113_113957

noncomputable def f (a x : ℝ) : ℝ :=
  a * (sqrt (-((x + 2) * (x - 6))) / (sqrt (x + 2) + sqrt (6 - x)))

theorem graph_symmetric {a : ℝ} : (∀ x ∈ set.Icc (-2 : ℝ) 6, f a x = f a (4 - x)) :=
by sorry

theorem monotonicity_interval_false {a : ℝ} (h : ∀ x ∈ set.Icc (-2 : ℝ) 2, monotone_on (f a) (set.Icc (-2 : ℝ) 2)) : a < 0 = false :=
by sorry

theorem max_value_one {a : ℝ} (ha : a = 1) : (∀ x ∈ set.Icc (-2 : ℝ) 6, f a x ≤ 1) ∧ (∃ x ∈ set.Icc (-2 : ℝ) 6, f a x = 1) :=
by sorry

theorem min_value_a {a : ℝ} (ha : a < 0) : (∀ x ∈ set.Icc (-2 : ℝ) 6, f a x ≥ a) ∧ (∃ x ∈ set.Icc (-2 : ℝ) 6, f a x = a) :=
by sorry

end graph_symmetric_monotonicity_interval_false_max_value_one_min_value_a_l113_113957


namespace evaluate_fraction_sum_l113_113145

-- Define the problem conditions and target equation
theorem evaluate_fraction_sum
    (p q r : ℝ)
    (h : p / (30 - p) + q / (75 - q) + r / (45 - r) = 8) :
    6 / (30 - p) + 15 / (75 - q) + 9 / (45 - r) = 11 / 5 := by
  sorry

end evaluate_fraction_sum_l113_113145


namespace length_AB_is_2sqrt3_l113_113151

open Real

-- Definitions of circle C and line l, point A
def circle_C := {x : ℝ × ℝ | (x.1 - 3)^2 + (x.2 + 1)^2 = 1}
def line_l (k : ℝ) := {p : ℝ × ℝ | k * p.1 + p.2 - 2 = 0}
def point_A (k : ℝ) := (0, k)

-- Conditions: line l passes through the center of the circle and is the axis of symmetry
def is_axis_of_symmetry_l (k : ℝ) := ∀ p: ℝ × ℝ, p ∈ circle_C → line_l k p

-- Main theorem to be proved
theorem length_AB_is_2sqrt3 (k : ℝ) (h_sym: is_axis_of_symmetry_l k) : 
  let A := point_A 1 in 
  let C := (3, -1) in 
  let radius := 1 in 
  let AC := sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) in
  sqrt (AC^2 - radius^2) = 2 * sqrt 3 :=
sorry -- proof not required

end length_AB_is_2sqrt3_l113_113151


namespace range_of_x8_l113_113348

theorem range_of_x8 (x : ℕ → ℝ) (h1 : 0 ≤ x 1 ∧ x 1 ≤ x 2)
  (h_recurrence : ∀ n ≥ 1, x (n+2) = x (n+1) + x n)
  (h_x7 : 1 ≤ x 7 ∧ x 7 ≤ 2) : 
  (21/13 : ℝ) ≤ x 8 ∧ x 8 ≤ (13/4) :=
sorry

end range_of_x8_l113_113348


namespace initial_persons_count_l113_113661

theorem initial_persons_count (P : ℕ) (H1 : 18 * P = 1) (H2 : 6 * P = 1/3) (H3 : 9 * (P + 4) = 2/3) : P = 12 :=
by
  sorry

end initial_persons_count_l113_113661


namespace no_snow_probability_l113_113692

theorem no_snow_probability (p_snow : ℚ) (h : p_snow = 2 / 3) :
  let p_no_snow := 1 - p_snow in
  (p_no_snow ^ 5 = 1 / 243) :=
by
  sorry

end no_snow_probability_l113_113692


namespace repeating_decimal_sum_l113_113935

theorem repeating_decimal_sum : (0.\overline{3} : ℚ) + (0.\overline{04} : ℚ) + (0.\overline{005} : ℚ) = 1135 / 2997 := 
sorry

end repeating_decimal_sum_l113_113935


namespace probability_no_snow_l113_113693

theorem probability_no_snow {
  let p_snow : ℚ := 2/3,     -- Probability of snow on any one day
  let p_no_snow : ℚ := 1 - p_snow,  -- Probability of no snow on any one day
  let p_no_snow_5_days : ℚ := p_no_snow ^ 5  -- Probability of no snow on any of the five days
} : p_no_snow_5_days = 1 / 243 := by
  sorry

end probability_no_snow_l113_113693


namespace sam_dad_gave_39_nickels_l113_113401

-- Define the initial conditions
def initial_pennies : ℕ := 49
def initial_nickels : ℕ := 24
def given_quarters : ℕ := 31
def dad_given_nickels : ℕ := 63 - initial_nickels

-- Statement to prove
theorem sam_dad_gave_39_nickels 
    (total_nickels_after : ℕ) 
    (initial_nickels : ℕ) 
    (final_nickels : ℕ := total_nickels_after - initial_nickels) : 
    final_nickels = 39 :=
sorry

end sam_dad_gave_39_nickels_l113_113401


namespace laptop_price_reduction_l113_113304

-- Conditions definitions
def initial_price (P : ℝ) : ℝ := P
def seasonal_sale (P : ℝ) : ℝ := 0.7 * P
def special_promotion (seasonal_price : ℝ) : ℝ := 0.8 * seasonal_price
def clearance_event (promotion_price : ℝ) : ℝ := 0.9 * promotion_price

-- Proof statement
theorem laptop_price_reduction (P : ℝ) (h1 : seasonal_sale P = 0.7 * P) 
    (h2 : special_promotion (seasonal_sale P) = 0.8 * (seasonal_sale P)) 
    (h3 : clearance_event (special_promotion (seasonal_sale P)) = 0.9 * (special_promotion (seasonal_sale P))) : 
    (initial_price P - clearance_event (special_promotion (seasonal_sale P))) / (initial_price P) = 0.496 := 
by 
  sorry

end laptop_price_reduction_l113_113304


namespace area_of_triangle_l113_113069

def point (α : Type*) := (α × α)

def x_and_y_lines (p : point ℝ) : Prop :=
  p.1 = p.2 ∨ p.1 = -p.2

def horizontal_line (y_val : ℝ) (p : point ℝ) : Prop :=
  p.2 = y_val

def vertices_of_triangle (p₁ p₂ p₃: point ℝ) : Prop :=
  horizontal_line 8 p₁ ∧ horizontal_line 8 p₂ ∧ x_and_y_lines p₃ ∧
  p₁ = (8, 8) ∧ p₂ = (-8, 8) ∧ p₃ = (0, 0)

theorem area_of_triangle : 
  ∃ (p₁ p₂ p₃ : point ℝ), vertices_of_triangle p₁ p₂ p₃ → 
  let base := abs (p₁.1 - p₂.1),
      height := abs (p₃.2 - p₁.2)
  in (1 / 2) * base * height = 64 := 
sorry

end area_of_triangle_l113_113069


namespace urn_marbles_100_white_l113_113984

theorem urn_marbles_100_white 
(initial_white initial_black final_white final_black : ℕ) 
(h_initial : initial_white = 150 ∧ initial_black = 50)
(h_operations : 
  (∀ n, (initial_white - 3 * n + 2 * n = final_white ∧ initial_black + n = final_black) ∨
  (initial_white - 2 * n - 1 = initial_white ∧ initial_black = final_black) ∨
  (initial_white - 1 * n - 2 = final_white ∧ initial_black - 1 * n = final_black) ∨
  (initial_white - 3 * n + 2 = final_white ∧ initial_black + 1 * n = final_black)) →
  ((initial_white = 150 ∧ initial_black = 50) →
   ∃ m: ℕ, final_white = 100)) :
∃ n: ℕ, initial_white - 3 * n + 2 * n = 100 ∧ initial_black + n = final_black :=
sorry

end urn_marbles_100_white_l113_113984


namespace distance_between_city_centers_l113_113035

theorem distance_between_city_centers (d_map : ℝ) (scale : ℝ) (d_real : ℝ) (h1 : d_map = 112) (h2 : scale = 10) (h3 : d_real = d_map * scale) : d_real = 1120 := by
  sorry

end distance_between_city_centers_l113_113035


namespace binomial_coefficient_divisibility_l113_113206

theorem binomial_coefficient_divisibility (n k : ℕ) (hkn : k ≤ n - 1) :
  ((n.prime) ∨ (¬ (n.prime) ∧ ∃ p, Nat.Prime p ∧ p ∣ n ∧ ¬ (n ∣ Nat.choose n p))) :=
by sorry

end binomial_coefficient_divisibility_l113_113206


namespace base_of_second_term_l113_113881

theorem base_of_second_term (h : ℕ) (a b c : ℕ) (H1 : h > 0) 
  (H2 : 225 ∣ h) (H3 : 216 ∣ h) 
  (H4 : h = (2^a) * (some_number^b) * (5^c)) 
  (H5 : a + b + c = 8) : some_number = 3 :=
by
  sorry

end base_of_second_term_l113_113881


namespace find_f_of_3_l113_113825

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l113_113825


namespace area_of_triangle_bounded_by_lines_l113_113070

theorem area_of_triangle_bounded_by_lines :
  let y1 := λ x : ℝ, x
  let y2 := λ x : ℝ, -x
  let y3 := λ x : ℝ, 8
  ∀ A B O : (ℝ × ℝ), 
  (A = (8, 8)) → 
  (B = (-8, 8)) → 
  (O = (0, 0)) →
  (triangle_area A B O = 64) :=
by
  intros y1 y2 y3 A B O hA hB hO
  have hA : A = (8, 8) := hA
  have hB : B = (-8, 8) := hB
  have hO : O = (0, 0) := hO
  sorry

end area_of_triangle_bounded_by_lines_l113_113070


namespace percent_absent_l113_113530

-- Given conditions
def total_students : ℕ := 120
def boys : ℕ := 72
def girls : ℕ := 48
def absent_boys_fraction : ℚ := 1 / 8
def absent_girls_fraction : ℚ := 1 / 4

-- Theorem to prove
theorem percent_absent : 100 * ((absent_boys_fraction * boys + absent_girls_fraction * girls) / total_students) = 17.5 := 
sorry

end percent_absent_l113_113530


namespace john_saving_yearly_l113_113508

def old_monthly_cost : ℕ := 1200
def increase_percentage : ℕ := 40
def split_count : ℕ := 3

def old_annual_cost (monthly_cost : ℕ) := monthly_cost * 12
def new_monthly_cost (monthly_cost : ℕ) (percentage : ℕ) := monthly_cost * (100 + percentage) / 100
def new_monthly_share (new_cost : ℕ) (split : ℕ) := new_cost / split
def new_annual_cost (monthly_share : ℕ) := monthly_share * 12
def annual_savings (old_annual : ℕ) (new_annual : ℕ) := old_annual - new_annual

theorem john_saving_yearly 
  (old_cost : ℕ := old_monthly_cost)
  (increase : ℕ := increase_percentage)
  (split : ℕ := split_count) :
  annual_savings (old_annual_cost old_cost) 
                 (new_annual_cost (new_monthly_share (new_monthly_cost old_cost increase) split)) 
  = 7680 :=
by
  sorry

end john_saving_yearly_l113_113508


namespace combined_height_is_9_l113_113007

def barrys_reach : ℝ := 5 -- Barry can reach apples that are 5 feet high

def larrys_full_height : ℝ := 5 -- Larry's full height is 5 feet

def larrys_shoulder_height : ℝ := larrys_full_height * 0.8 -- Larry's shoulder height is 20% less than his full height

def combined_reach (b_reach : ℝ) (l_shoulder : ℝ) : ℝ := b_reach + l_shoulder

theorem combined_height_is_9 : combined_reach barrys_reach larrys_shoulder_height = 9 := by
  sorry

end combined_height_is_9_l113_113007


namespace find_a_l113_113789

theorem find_a (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end find_a_l113_113789


namespace transformed_line_theorem_l113_113037

theorem transformed_line_theorem (k b : ℝ) (h₁ : k = 1) (h₂ : b = 1) (x : ℝ) :
  (k * x + b > 0) ↔ (x > -1) :=
by sorry

end transformed_line_theorem_l113_113037


namespace problem_expression_value_l113_113515

theorem problem_expression_value {a b c k1 k2 : ℂ} 
  (h_root : ∀ x, x^3 - k1 * x - k2 = 0 → x = a ∨ x = b ∨ x = c) 
  (h_condition : k1 + k2 ≠ 1)
  (h_vieta1 : a + b + c = 0)
  (h_vieta2 : a * b + b * c + c * a = -k1)
  (h_vieta3 : a * b * c = k2) :
  (1 + a)/(1 - a) + (1 + b)/(1 - b) + (1 + c)/(1 - c) = 
  (3 + k1 + 3 * k2)/(1 - k1 - k2) :=
by
  sorry

end problem_expression_value_l113_113515


namespace find_f_of_3_l113_113831

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l113_113831


namespace smallest_b_gt_4_perfect_square_l113_113227

theorem smallest_b_gt_4_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ k : ℕ, 4 * b + 5 = k^2 ∧ b = 5 :=
by
  sorry

end smallest_b_gt_4_perfect_square_l113_113227


namespace proof_problem_l113_113525

def problem : Prop :=
  ∃ (x y z t : ℤ), 
    0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ t ∧ 
    x^2 + y^2 + z^2 + t^2 = 2^2004

theorem proof_problem : 
  problem → 
  ∃! (x y z t : ℤ), 
    0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ t ∧ 
    x^2 + y^2 + z^2 + t^2 = 2^2004 :=
sorry

end proof_problem_l113_113525


namespace christopher_more_than_karen_l113_113509

-- Define the number of quarters Karen and Christopher have
def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64

-- Define the value of a quarter in dollars
def value_of_quarter : ℚ := 0.25

-- Define the amount of money Christopher has more than Karen in dollars
def christopher_more_money : ℚ := (christopher_quarters - karen_quarters) * value_of_quarter

-- Theorem to prove that Christopher has $8.00 more than Karen
theorem christopher_more_than_karen : christopher_more_money = 8 := by
  sorry

end christopher_more_than_karen_l113_113509


namespace general_term_formula_l113_113417

def seq (n : ℕ) : ℤ :=
match n with
| 0       => 1
| 1       => -3
| 2       => 5
| 3       => -7
| 4       => 9
| (n + 1) => (-1)^(n+1) * (2*n + 1) -- extends indefinitely for general natural number

theorem general_term_formula (n : ℕ) : 
  seq n = (-1)^(n+1) * (2*n-1) :=
sorry

end general_term_formula_l113_113417


namespace photographer_max_photos_l113_113454

-- The initial number of birds of each species
def total_birds : ℕ := 20
def starlings : ℕ := 8
def wagtails : ℕ := 7
def woodpeckers : ℕ := 5

-- Define a function to count the remaining birds of each species after n photos
def remaining_birds (n : ℕ) (species : ℕ) : ℕ := species - (if species ≤ n then species else n)

-- Define the main theorem we want to prove
theorem photographer_max_photos (n : ℕ) (h1 : remaining_birds n starlings ≥ 4) (h2 : remaining_birds n wagtails ≥ 3) : 
  n ≤ 7 :=
by
  sorry

end photographer_max_photos_l113_113454


namespace same_side_probability_l113_113711

theorem same_side_probability :
  (∀ (n : ℕ), n = 4 → ∀ (p : ℚ), p = 1/2 → (p ^ n = 1/16)) :=
by
  sorry

end same_side_probability_l113_113711


namespace dog_treats_cost_l113_113003

theorem dog_treats_cost
  (treats_per_day : ℕ)
  (cost_per_treat : ℚ)
  (days_in_month : ℕ)
  (H1 : treats_per_day = 2)
  (H2 : cost_per_treat = 0.1)
  (H3 : days_in_month = 30) :
  treats_per_day * days_in_month * cost_per_treat = 6 :=
by sorry

end dog_treats_cost_l113_113003


namespace tile_border_ratio_l113_113722

theorem tile_border_ratio (n : ℕ) (t w : ℝ) (H1 : n = 30)
  (H2 : 900 * t^2 / (30 * t + 30 * w)^2 = 0.81) :
  w / t = 1 / 9 :=
by
  sorry

end tile_border_ratio_l113_113722


namespace cos_C_in_triangle_l113_113982

theorem cos_C_in_triangle 
  (A B C : ℝ) 
  (h_triangle : A + B + C = 180)
  (sin_A : Real.sin A = 4 / 5) 
  (cos_B : Real.cos B = 12 / 13) : 
  Real.cos C = -16 / 65 :=
by
  sorry

end cos_C_in_triangle_l113_113982


namespace problem_statement_l113_113179

noncomputable def ratio_AD_AB (AB AD : ℝ) (angle_A angle_B angle_ADE : ℝ) : Prop :=
  angle_A = 60 ∧ angle_B = 45 ∧ angle_ADE = 45 ∧
  AD / AB = (Real.sqrt 6 + Real.sqrt 2) / (4 * Real.sqrt 2)

theorem problem_statement {AB AD : ℝ} (angle_A angle_B angle_ADE : ℝ) 
  (h1 : angle_A = 60)
  (h2 : angle_B = 45)
  (h3 : angle_ADE = 45) : 
  ratio_AD_AB AB AD angle_A angle_B angle_ADE := by {
    sorry
}

end problem_statement_l113_113179


namespace total_cans_given_away_l113_113900

-- Define constants
def initial_stock : ℕ := 2000

-- Define conditions day 1
def people_day1 : ℕ := 500
def cans_per_person_day1 : ℕ := 1
def restock_day1 : ℕ := 1500

-- Define conditions day 2
def people_day2 : ℕ := 1000
def cans_per_person_day2 : ℕ := 2
def restock_day2 : ℕ := 3000

-- Define the question as a theorem
theorem total_cans_given_away : (people_day1 * cans_per_person_day1 + people_day2 * cans_per_person_day2) = 2500 := by
  sorry

end total_cans_given_away_l113_113900


namespace ratio_brown_eyes_l113_113213

theorem ratio_brown_eyes (total_people : ℕ) (blue_eyes : ℕ) (black_eyes : ℕ) (green_eyes : ℕ) (brown_eyes : ℕ) 
    (h1 : total_people = 100) 
    (h2 : blue_eyes = 19) 
    (h3 : black_eyes = total_people / 4) 
    (h4 : green_eyes = 6) 
    (h5 : brown_eyes = total_people - (blue_eyes + black_eyes + green_eyes)) : 
    brown_eyes / total_people = 1 / 2 :=
by sorry

end ratio_brown_eyes_l113_113213


namespace smallest_positive_multiple_45_l113_113240

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l113_113240


namespace number_of_valid_partitions_l113_113757

-- Define the condition to check if a list of integers has all elements same or exactly differ by 1
def validPartition (l : List ℕ) : Prop :=
  l ≠ [] ∧ (∀ (a b : ℕ), a ∈ l → b ∈ l → a = b ∨ a = b + 1 ∨ b = a + 1)

-- Count valid partitions of n (integer partitions meeting the given condition)
noncomputable def countValidPartitions (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

-- Main theorem
theorem number_of_valid_partitions (n : ℕ) : countValidPartitions n = n :=
by
  sorry

end number_of_valid_partitions_l113_113757


namespace divisors_of_36_l113_113163

theorem divisors_of_36 : ∀ n : ℕ, n = 36 → (∃ k : ℕ, k = 9) :=
by
  intro n hn
  have h_prime_factors : (n = 2^2 * 3^2) := by rw hn; norm_num
  -- Using the formula for the number of divisors based on prime factorization
  have h_num_divisors : (2 + 1) * (2 + 1) = 9 := by norm_num
  use 9
  rw h_num_divisors
  sorry

end divisors_of_36_l113_113163


namespace new_average_is_ten_l113_113537

-- Define the initial conditions
def initial_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) : Prop :=
  x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ + x₈ + x₉ = 9 * 7

-- Define the transformation on the nine numbers
def transformed_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) : ℝ :=
  (x₁ - 3) + (x₂ - 3) + (x₃ - 3) +
  (x₄ + 5) + (x₅ + 5) + (x₆ + 5) +
  (2 * x₇) + (2 * x₈) + (2 * x₉)

-- The theorem to prove the new average is 10
theorem new_average_is_ten (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) 
  (h : initial_sum x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉) :
  transformed_sum x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ / 9 = 10 :=
by 
  sorry

end new_average_is_ten_l113_113537


namespace base7_divisibility_l113_113342

/-
Given a base-7 number represented as 3dd6_7, where the first digit is 3, the last digit is 6, 
and the middle two digits are equal to d, prove that the base-7 digit d which makes this 
number divisible by 13 is 5.
-/
theorem base7_divisibility (d : ℕ) (hdig : d ∈ {0, 1, 2, 3, 4, 5, 6}) : 
  (3 * 7^3 + d * 7^2 + d * 7 + 6) % 13 = 0 ↔ d = 5 := 
sorry

end base7_divisibility_l113_113342


namespace smallest_positive_multiple_of_45_is_45_l113_113271

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113271


namespace fraction_of_capital_contributed_by_a_l113_113090

theorem fraction_of_capital_contributed_by_a
  (A B : ℚ)
  (h1 : 15 * A / (10 * B) = 1 / 2)
  (h2 : A + B = 1) :
  A = 1 / 4 :=
sorry

end fraction_of_capital_contributed_by_a_l113_113090


namespace expenditure_increase_36_percent_l113_113447

theorem expenditure_increase_36_percent
  (m : ℝ) -- mass of the bread
  (p_bread : ℝ) -- price of the bread
  (p_crust : ℝ) -- price of the crust
  (h1 : p_crust = 1.2 * p_bread) -- condition: crust is 20% more expensive
  (h2 : p_crust = 1.2 * p_bread) -- condition: crust is 20% more expensive
  (h3 : ∃ (m_crust : ℝ), m_crust = 0.75 * m) -- condition: crust is 25% lighter in weight
  (h4 : ∃ (m_consumed_bread : ℝ), m_consumed_bread = 0.85 * m) -- condition: 15% of bread dries out
  (h5 : ∃ (m_consumed_crust : ℝ), m_consumed_crust = 0.75 * m) -- condition: crust is consumed completely
  : (17 / 15) * (1.2 : ℝ) = 1.36 := 
by sorry

end expenditure_increase_36_percent_l113_113447


namespace term_containing_x3_l113_113533

-- Define the problem statement in Lean 4
theorem term_containing_x3 (a : ℝ) (x : ℝ) (hx : x ≠ 0) 
(h_sum_coeff : (2 + a) ^ 5 = 0) :
  (2 * x + a / x) ^ 5 = -160 * x ^ 3 :=
sorry

end term_containing_x3_l113_113533


namespace find_n_l113_113362

theorem find_n (x k m n : ℤ) 
  (h1 : x = 82 * k + 5)
  (h2 : x + n = 41 * m + 18) :
  n = 5 :=
by
  sorry

end find_n_l113_113362


namespace prove_conditions_l113_113552

theorem prove_conditions :
  (a ∈ {a, b, c}) ∧ ({0, 1} ⊆ (set_of nat)) :=
by
  apply and.intro
  {
    exact set.mem_insert a {b, c}
  }
  {
    apply set.subset.trans (set.singleton_subset_iff.mpr (nat.cast_id ..))
    sorry -- Requires proof that {0, 1} ⊆ ℕ
  }

end prove_conditions_l113_113552


namespace fraction_product_l113_113548

theorem fraction_product : (1 / 4 : ℚ) * (2 / 5) * (3 / 6) = (1 / 20) := by
  sorry

end fraction_product_l113_113548


namespace parallel_vectors_l113_113961

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (P : a = (1, m) ∧ b = (m, 2) ∧ (a.1 / m = b.1 / 2)) :
  m = -Real.sqrt 2 ∨ m = Real.sqrt 2 :=
by
  sorry

end parallel_vectors_l113_113961


namespace max_omega_l113_113764

open Real

-- Define the function f(x) = sin(ωx + φ)
noncomputable def f (ω φ x : ℝ) := sin (ω * x + φ)

-- ω > 0 and |φ| ≤ π / 2
def condition_omega_pos (ω : ℝ) := ω > 0
def condition_phi_bound (φ : ℝ) := abs φ ≤ π / 2

-- x = -π/4 is a zero of f(x)
def condition_zero (ω φ : ℝ) := f ω φ (-π/4) = 0

-- x = π/4 is the axis of symmetry for the graph of y = f(x)
def condition_symmetry (ω φ : ℝ) := 
  ∀ x : ℝ, f ω φ (π/4 - x) = f ω φ (π/4 + x)

-- f(x) is monotonic in the interval (π/18, 5π/36)
def condition_monotonic (ω φ : ℝ) := 
  ∀ x₁ x₂ : ℝ, π/18 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5 * π / 36 
  → f ω φ x₁ ≤ f ω φ x₂

-- Prove that the maximum value of ω satisfying all the conditions is 9
theorem max_omega (ω : ℝ) (φ : ℝ)
  (h1 : condition_omega_pos ω)
  (h2 : condition_phi_bound φ)
  (h3 : condition_zero ω φ)
  (h4 : condition_symmetry ω φ)
  (h5 : condition_monotonic ω φ) :
  ω ≤ 9 :=
sorry

end max_omega_l113_113764


namespace gcd_polynomial_is_25_l113_113616

theorem gcd_polynomial_is_25 (b : ℕ) (h : ∃ k : ℕ, b = 2700 * k) :
  Nat.gcd (b^2 + 27 * b + 75) (b + 25) = 25 :=
by 
    sorry

end gcd_polynomial_is_25_l113_113616


namespace quadratic_has_real_root_l113_113873

theorem quadratic_has_real_root (a b : ℝ) : ¬ (∀ x : ℝ, x^2 + a * x + b ≠ 0) → ∃ x : ℝ, x^2 + a * x + b = 0 := 
by
  sorry

end quadratic_has_real_root_l113_113873


namespace polygon_sides_l113_113305

-- Define the conditions
def side_length : ℝ := 7
def perimeter : ℝ := 42

-- The statement to prove: number of sides is 6
theorem polygon_sides : (perimeter / side_length) = 6 := by
  sorry

end polygon_sides_l113_113305


namespace prove_PoincareObservation_l113_113883

noncomputable def PoincareObservation 
  (X : ℕ → ℕ → ℝ)
  (ξ : ℕ → ℝ)
  (S : ℕ → ℝ)
  (n m : ℕ)
  (hn : n ≥ 1)
  (hm : m ≥ 1) :
  Prop :=
  tendsto (λ (n : ℕ), (λ i, X n i) '' {1..m}) 
          (𝓝 (λ i, ξ i '' {1..m}))

theorem prove_PoincareObservation 
  (X : ℕ → ℕ → ℝ)
  (ξ : ℕ → ℝ)
  (S : ℕ → ℝ)
  (n m : ℕ)
  (hn : n ≥ 1)
  (hm : m ≥ 1)
  (h_uniform : ∀ n, ∀ i ∈ finset.range n, MeasureTheory.measure_probability_measure.map (MeasureTheory.probability_measure_of_measurable_set (λ x, X n x) finset.range n) = spherical_measure (S n))
  (h_gaussian : ∀ i, MeasureTheory.probability_measure (λ x, ξ i x)) :
  PoincareObservation X ξ S n m hn hm :=
begin
  sorry
end

end prove_PoincareObservation_l113_113883


namespace no_snow_probability_l113_113691

theorem no_snow_probability (p_snow : ℚ) (h : p_snow = 2 / 3) :
  let p_no_snow := 1 - p_snow in
  (p_no_snow ^ 5 = 1 / 243) :=
by
  sorry

end no_snow_probability_l113_113691


namespace smallest_positive_multiple_l113_113281

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l113_113281


namespace non_zero_digits_fraction_l113_113899

def count_non_zero_digits (n : ℚ) : ℕ :=
  -- A placeholder for the actual implementation.
  sorry

theorem non_zero_digits_fraction : count_non_zero_digits (120 / (2^4 * 5^9 : ℚ)) = 3 :=
  sorry

end non_zero_digits_fraction_l113_113899


namespace smallest_positive_multiple_of_45_l113_113273

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l113_113273


namespace arrange_chairs_and_stools_l113_113565

-- Definition of the mathematical entities based on the conditions
def num_ways_to_arrange (women men : ℕ) : ℕ :=
  let total := women + men
  (total.factorial) / (women.factorial * men.factorial)

-- Prove that the arrangement yields the correct number of ways
theorem arrange_chairs_and_stools :
  num_ways_to_arrange 7 3 = 120 := by
  -- The specific definitions and steps are not to be included in the Lean statement;
  -- hence, adding a placeholder for the proof.
  sorry

end arrange_chairs_and_stools_l113_113565


namespace smallest_positive_multiple_of_45_l113_113232

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l113_113232


namespace find_fifth_integer_l113_113140

theorem find_fifth_integer (x y : ℤ) (h_pos : x > 0)
  (h_mean_median : (x + 2 + x + 7 + x + y) / 5 = x + 7) :
  y = 22 :=
sorry

end find_fifth_integer_l113_113140


namespace find_f_of_3_l113_113820

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l113_113820


namespace only_n1_makes_n4_plus4_prime_l113_113608

theorem only_n1_makes_n4_plus4_prime (n : ℕ) (h : n > 0) : (n = 1) ↔ Prime (n^4 + 4) :=
sorry

end only_n1_makes_n4_plus4_prime_l113_113608


namespace arithmetic_sequence_expression_l113_113779

variable (a : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, (n < m) → (a (m + 1) - a m = a (n + 1) - a n)

theorem arithmetic_sequence_expression
  (h_arith_seq : is_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = -3) :
  ∀ n : ℕ, a n = -2 * n + 3 :=
  sorry

end arithmetic_sequence_expression_l113_113779


namespace cost_of_cheaper_feed_l113_113426

theorem cost_of_cheaper_feed (C : ℝ)
  (total_weight : ℝ) (weight_cheaper : ℝ) (price_expensive : ℝ) (total_value : ℝ) : 
  total_weight = 35 → 
  total_value = 0.36 * total_weight → 
  weight_cheaper = 17 → 
  price_expensive = 0.53 →
  (total_value = weight_cheaper * C + (total_weight - weight_cheaper) * price_expensive) →
  C = 0.18 := 
by
  sorry

end cost_of_cheaper_feed_l113_113426


namespace oscar_cookie_baking_time_l113_113745

theorem oscar_cookie_baking_time : 
  (1 / 5) + (1 / 6) + (1 / o) - (1 / 4) = (1 / 8) → o = 120 := by
  sorry

end oscar_cookie_baking_time_l113_113745


namespace find_total_income_l113_113880

theorem find_total_income (I : ℝ)
  (h1 : 0.6 * I + 0.3 * I + 0.005 * (I - (0.6 * I + 0.3 * I)) + 50000 = I) : 
  I = 526315.79 :=
by
  sorry

end find_total_income_l113_113880


namespace triple_composition_f_3_l113_113968

def f (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition_f_3 : f (f (f 3)) = 107 :=
by
  sorry

end triple_composition_f_3_l113_113968


namespace opposite_pairs_l113_113086

theorem opposite_pairs :
  (3^2 = 9) ∧ (-3^2 = -9) ∧
  ¬ ((3^2 = 9 ∧ -2^3 = -8) ∧ 9 = -(-8)) ∧
  ¬ ((3^2 = 9 ∧ (-3)^2 = 9) ∧ 9 = -9) ∧
  ¬ ((-3^2 = -9 ∧ -(-3)^2 = -9) ∧ -9 = -(-9)) :=
by
  sorry

end opposite_pairs_l113_113086


namespace factorial_quotient_52_50_l113_113122

theorem factorial_quotient_52_50 : (Nat.factorial 52) / (Nat.factorial 50) = 2652 := 
by 
  sorry

end factorial_quotient_52_50_l113_113122


namespace parts_production_equation_l113_113445

theorem parts_production_equation (x : ℝ) : 
  let apr := 50
  let may := 50 * (1 + x)
  let jun := 50 * (1 + x) * (1 + x)
  (apr + may + jun = 182) :=
sorry

end parts_production_equation_l113_113445


namespace member_pays_48_percent_of_SRP_l113_113300

theorem member_pays_48_percent_of_SRP
  (P : ℝ)
  (h₀ : P > 0)
  (basic_discount : ℝ := 0.40)
  (additional_discount : ℝ := 0.20) :
  ((1 - additional_discount) * (1 - basic_discount) * P) / P * 100 = 48 := by
  sorry

end member_pays_48_percent_of_SRP_l113_113300


namespace john_spends_on_dog_treats_l113_113004

def number_of_days_in_month := 30
def treats_per_day := 2
def cost_per_treat := 0.1

theorem john_spends_on_dog_treats : 
  (number_of_days_in_month * treats_per_day * cost_per_treat) = 6 := 
by 
  sorry

end john_spends_on_dog_treats_l113_113004


namespace new_galleries_receive_two_pictures_l113_113312

theorem new_galleries_receive_two_pictures :
  ∀ (total_pencils : ℕ) (orig_gallery_pictures : ℕ) (new_galleries : ℕ) 
    (pencils_per_picture : ℕ) (signature_pencils_per_gallery : ℕ),
    (total_pencils = 88) →
    (orig_gallery_pictures = 9) →
    (new_galleries = 5) →
    (pencils_per_picture = 4) →
    (signature_pencils_per_gallery = 2) →
    let orig_gallery_signature_pencils := signature_pencils_per_gallery in
    let total_galleries := 1 + new_galleries in
    let total_signature_pencils := total_galleries * signature_pencils_per_gallery in
    let total_drawing_pencils := total_pencils - total_signature_pencils in
    let orig_gallery_drawing_pencils := orig_gallery_pictures * pencils_per_picture in
    let new_galleries_drawing_pencils := total_drawing_pencils - orig_gallery_drawing_pencils in
    let total_new_gallery_pictures := new_galleries_drawing_pencils / pencils_per_picture in
    let pictures_per_new_gallery := total_new_gallery_pictures / new_galleries in
    pictures_per_new_gallery = 2 :=
begin
  intros total_pencils orig_gallery_pictures new_galleries pencils_per_picture signature_pencils_per_gallery
          H_total_pencils H_orig_gallery_pictures H_new_galleries H_pencils_per_picture H_signature_pencils_per_gallery,

  let orig_gallery_signature_pencils := signature_pencils_per_gallery,
  let total_galleries := 1 + new_galleries,
  let total_signature_pencils := total_galleries * signature_pencils_per_gallery,
  let total_drawing_pencils := total_pencils - total_signature_pencils,
  let orig_gallery_drawing_pencils := orig_gallery_pictures * pencils_per_picture,
  let new_galleries_drawing_pencils := total_drawing_pencils - orig_gallery_drawing_pencils,
  let total_new_gallery_pictures := new_galleries_drawing_pencils / pencils_per_picture,
  let pictures_per_new_gallery := total_new_gallery_pictures / new_galleries,

  exact sorry
end

end new_galleries_receive_two_pictures_l113_113312


namespace value_of_a_plus_c_l113_113969

variables {a b c : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

def f_inv (x : ℝ) : ℝ := c * x^2 + b * x + a

theorem value_of_a_plus_c : a + c = -1 :=
sorry

end value_of_a_plus_c_l113_113969


namespace shaded_region_perimeter_l113_113371

theorem shaded_region_perimeter (r : ℝ) (h : r = 12 / Real.pi) :
  3 * (24 / 6) = 12 := 
by
  sorry

end shaded_region_perimeter_l113_113371


namespace k_range_condition_l113_113329

theorem k_range_condition (k : ℝ) :
    (∀ x : ℝ, x^2 - (2 * k - 6) * x + k - 3 > 0) ↔ (3 < k ∧ k < 4) :=
by
  sorry

end k_range_condition_l113_113329


namespace find_two_digit_number_with_cubic_ending_in_9_l113_113636

theorem find_two_digit_number_with_cubic_ending_in_9:
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n^3 % 10 = 9 ∧ n = 19 := 
by
  sorry

end find_two_digit_number_with_cubic_ending_in_9_l113_113636


namespace tens_digit_of_M_l113_113008

def P (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a * b

def S (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a + b

theorem tens_digit_of_M {M : ℕ} (h : 10 ≤ M ∧ M < 100) (h_eq : M = P M + S M + 6) :
  M / 10 = 1 ∨ M / 10 = 2 :=
sorry

end tens_digit_of_M_l113_113008


namespace total_number_of_toys_l113_113997

def jaxon_toys := 15
def gabriel_toys := 2 * jaxon_toys
def jerry_toys := gabriel_toys + 8

theorem total_number_of_toys : jaxon_toys + gabriel_toys + jerry_toys = 83 := by
  sorry

end total_number_of_toys_l113_113997


namespace minimum_shirts_to_save_money_l113_113662

-- Definitions for the costs
def EliteCost (n : ℕ) : ℕ := 30 + 8 * n
def OmegaCost (n : ℕ) : ℕ := 10 + 12 * n

-- Theorem to prove the given solution
theorem minimum_shirts_to_save_money : ∃ n : ℕ, 30 + 8 * n < 10 + 12 * n ∧ n = 6 :=
by {
  sorry
}

end minimum_shirts_to_save_money_l113_113662


namespace composite_integer_divisors_l113_113324

theorem composite_integer_divisors (n : ℕ) (k : ℕ) (d : ℕ → ℕ) 
  (h_composite : 1 < n ∧ ¬Prime n)
  (h_divisors : ∀ i, 1 ≤ i ∧ i ≤ k → d i ∣ n)
  (h_distinct : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → d i < d j)
  (h_range : d 1 = 1 ∧ d k = n)
  (h_ratio : ∀ i, 1 ≤ i ∧ i < k → (d (i + 1) - d i) = (i * (d 2 - d 1))) : n = 6 :=
by sorry

end composite_integer_divisors_l113_113324


namespace find_k_l113_113491

/--
Given a system of linear equations:
1) x + 2 * y = -a + 1
2) x - 3 * y = 4 * a + 6
If the expression k * x - y remains unchanged regardless of the value of the constant a, 
show that k = -1.
-/
theorem find_k 
  (a x y k : ℝ) 
  (h1 : x + 2 * y = -a + 1) 
  (h2 : x - 3 * y = 4 * a + 6)
  (h3 : ∀ a₁ a₂ x₁ x₂ y₁ y₂, (x₁ + 2 * y₁ = -a₁ + 1) → (x₁ - 3 * y₁ = 4 * a₁ + 6) → 
                               (x₂ + 2 * y₂ = -a₂ + 1) → (x₂ - 3 * y₂ = 4 * a₂ + 6) → 
                               (k * x₁ - y₁ = k * x₂ - y₂)) : 
  k = -1 :=
  sorry

end find_k_l113_113491


namespace purchase_price_of_furniture_l113_113209

theorem purchase_price_of_furniture (marked_price discount_rate profit_rate : ℝ) 
(h_marked_price : marked_price = 132) 
(h_discount_rate : discount_rate = 0.1)
(h_profit_rate : profit_rate = 0.1)
: ∃ a : ℝ, (marked_price * (1 - discount_rate) - a = profit_rate * a) ∧ a = 108 := by
  sorry

end purchase_price_of_furniture_l113_113209


namespace two_colonies_reach_limit_in_same_time_l113_113293

theorem two_colonies_reach_limit_in_same_time (d : ℕ) (h : 16 = d): 
  d = 16 :=
by
  /- Asserting that if one colony takes 16 days, two starting together will also take 16 days -/
  sorry

end two_colonies_reach_limit_in_same_time_l113_113293


namespace f_f_f_3_l113_113967

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_f_f_3 : f(f(f(3))) = 107 := 
by
  sorry

end f_f_f_3_l113_113967


namespace first_term_of_arithmetic_seq_l113_113009

theorem first_term_of_arithmetic_seq (S : ℕ → ℚ) (a : ℚ) (d : ℚ) (h_d : d = 4)
  (hS_n : ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2)
  (h_ratio_const : ∀ n, (S (2 * n)) / (S n) = 4)
  : a = 2 :=
by
  sorry

end first_term_of_arithmetic_seq_l113_113009


namespace scientific_notation_40_9_billion_l113_113977

theorem scientific_notation_40_9_billion :
  (40.9 * 10^9) = 4.09 * 10^9 :=
by
  sorry

end scientific_notation_40_9_billion_l113_113977


namespace mirror_side_length_l113_113893

theorem mirror_side_length
  (width_wall : ℝ)
  (length_wall : ℝ)
  (area_wall : ℝ)
  (area_mirror : ℝ)
  (side_length_mirror : ℝ)
  (h1 : width_wall = 32)
  (h2 : length_wall = 20.25)
  (h3 : area_wall = width_wall * length_wall)
  (h4 : area_mirror = area_wall / 2)
  (h5 : side_length_mirror * side_length_mirror = area_mirror)
  : side_length_mirror = 18 := by
  sorry

end mirror_side_length_l113_113893


namespace math_proof_problem_l113_113510

theorem math_proof_problem
  (n m k l : ℕ)
  (hpos_n : n > 0)
  (hpos_m : m > 0)
  (hpos_k : k > 0)
  (hpos_l : l > 0)
  (hneq_n : n ≠ 1)
  (hdiv : n^k + m*n^l + 1 ∣ n^(k+l) - 1) :
  (m = 1 ∧ l = 2*k) ∨ (l ∣ k ∧ m = (n^(k-l) - 1) / (n^l - 1)) :=
by 
  sorry

end math_proof_problem_l113_113510


namespace part1_part2_l113_113766

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x * (Real.sin x + Real.cos x)) - 1 / 2

theorem part1 (α : ℝ) (hα1 : 0 < α ∧ α < Real.pi / 2) (hα2 : Real.sin α = Real.sqrt 2 / 2) :
  f α = 1 / 2 :=
sorry

theorem part2 :
  ∀ (k : ℤ), ∀ (x : ℝ),
  -((3 : ℝ) * Real.pi / 8) + k * Real.pi ≤ x ∧ x ≤ (Real.pi / 8) + k * Real.pi →
  MonotoneOn f (Set.Icc (-((3 : ℝ) * Real.pi / 8) + k * Real.pi) ((Real.pi / 8) + k * Real.pi)) :=
sorry

end part1_part2_l113_113766


namespace problem_extraneous_root_l113_113770

theorem problem_extraneous_root (m : ℤ) :
  (∃ x, x = -4 ∧ (x + 4 = 0) ∧ ((x-1)/(x+4) = m/(x+4)) ∧ (m = -5)) :=
sorry

end problem_extraneous_root_l113_113770


namespace new_oranges_added_l113_113571

-- Defining the initial conditions
def initial_oranges : Nat := 40
def thrown_away_oranges : Nat := 37
def total_oranges_now : Nat := 10
def remaining_oranges : Nat := initial_oranges - thrown_away_oranges
def new_oranges := total_oranges_now - remaining_oranges

-- The theorem we want to prove
theorem new_oranges_added : new_oranges = 7 := by
  sorry

end new_oranges_added_l113_113571


namespace inverse_matrix_l113_113602

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 7], ![-1, -1]]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![-(1/3 : ℚ), -(7/3 : ℚ)], ![1/3, 4/3]]

theorem inverse_matrix : A.det ≠ 0 → A⁻¹ = A_inv := by
  sorry

end inverse_matrix_l113_113602


namespace probability_spade_heart_diamond_l113_113347

-- Condition: Definition of probability functions and a standard deck
def probability_of_first_spade (deck : Finset ℕ) : ℚ := 13 / 52
def probability_of_second_heart (deck : Finset ℕ) (first_card_spade : Prop) : ℚ := 13 / 51
def probability_of_third_diamond (deck : Finset ℕ) (first_card_spade : Prop) (second_card_heart : Prop) : ℚ := 13 / 50

-- Combined probability calculation
def probability_sequence_spade_heart_diamond (deck : Finset ℕ) : ℚ := 
  probability_of_first_spade deck * 
  probability_of_second_heart deck (true) * 
  probability_of_third_diamond deck (true) (true)

-- Lean statement proving the problem
theorem probability_spade_heart_diamond :
  probability_sequence_spade_heart_diamond (Finset.range 52) = 2197 / 132600 :=
by
  -- Proof steps will go here
  sorry

end probability_spade_heart_diamond_l113_113347


namespace sign_of_b_l113_113216

variable (a b : ℝ)

theorem sign_of_b (h1 : (a + b > 0 ∨ a - b > 0) ∧ (a + b < 0 ∨ a - b < 0)) 
                  (h2 : (ab > 0 ∨ a / b > 0) ∧ (ab < 0 ∨ a / b < 0))
                  (h3 : (ab > 0 → a > 0 ∧ b > 0) ∨ (ab < 0 → (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0))) :
  b < 0 :=
sorry

end sign_of_b_l113_113216


namespace number_of_unique_intersections_l113_113772

-- Definitions for the given lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 3
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 5 * x - 3 * y = 6

-- The problem is to show the number of unique intersection points is 2
theorem number_of_unique_intersections : ∃ p1 p2 : ℝ × ℝ,
  p1 ≠ p2 ∧
  (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
  (line1 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
  (p1 ≠ p2 → ∀ p : ℝ × ℝ, (line1 p.1 p.2 ∨ line2 p.1 p.2 ∨ line3 p.1 p.2) →
    (p = p1 ∨ p = p2)) :=
sorry

end number_of_unique_intersections_l113_113772


namespace find_fixed_point_on_ellipse_l113_113759

theorem find_fixed_point_on_ellipse (a b c : ℝ) (h_gt_zero : a > b ∧ b > 0)
    (h_ellipse : ∀ (P : ℝ × ℝ), (P.1 ^ 2 / a ^ 2) + (P.2 ^ 2 / b ^ 2) = 1)
    (A1 A2 : ℝ × ℝ)
    (h_A1 : A1 = (-a, 0))
    (h_A2 : A2 = (a, 0))
    (MC : ℝ) (h_MC : MC = (a^2 + b^2) / c) :
  ∃ (M : ℝ × ℝ), M = (MC, 0) := 
sorry

end find_fixed_point_on_ellipse_l113_113759


namespace smaller_solution_of_quadratic_eq_l113_113339

theorem smaller_solution_of_quadratic_eq : 
  (exists x y : ℝ, x < y ∧ x^2 - 13 * x + 36 = 0 ∧ y^2 - 13 * y + 36 = 0 ∧ x = 4) :=
by sorry

end smaller_solution_of_quadratic_eq_l113_113339


namespace find_length_BF_l113_113101

-- Define the conditions
structure Rectangle :=
  (short_side : ℝ)
  (long_side : ℝ)

def folded_paper (rect : Rectangle) : Prop :=
  rect.short_side = 12

def congruent_triangles (rect : Rectangle) : Prop :=
  rect.short_side = 12

-- Define the length of BF to prove
def length_BF (rect : Rectangle) : ℝ := 10

-- The theorem statement
theorem find_length_BF (rect : Rectangle) (h1 : folded_paper rect) (h2 : congruent_triangles rect) :
  length_BF rect = 10 := 
  sorry

end find_length_BF_l113_113101


namespace root_k_value_l113_113499

theorem root_k_value
  (k : ℝ)
  (h : Polynomial.eval 4 (Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 3 * Polynomial.X - Polynomial.C k) = 0) :
  k = 44 :=
sorry

end root_k_value_l113_113499


namespace largest_divisor_of_square_divisible_by_24_l113_113423

theorem largest_divisor_of_square_divisible_by_24 (n : ℕ) (h₁ : n > 0) (h₂ : 24 ∣ n^2) (h₃ : ∀ k : ℕ, k ∣ n → k ≤ 8) : n = 24 := 
sorry

end largest_divisor_of_square_divisible_by_24_l113_113423


namespace fill_time_l113_113649

-- Definition of the conditions
def faster_pipe_rate (t : ℕ) := 1 / t
def slower_pipe_rate (t : ℕ) := 1 / (4 * t)
def combined_rate (t : ℕ) := faster_pipe_rate t + slower_pipe_rate t
def time_to_fill_tank (t : ℕ) := 1 / combined_rate t

-- Given t = 50, prove the combined fill time is 40 minutes which is equal to the target time to fill the tank.
theorem fill_time (t : ℕ) (h : 4 * t = 200) : t = 50 → time_to_fill_tank t = 40 :=
by
  intros ht
  rw [ht]
  sorry

end fill_time_l113_113649


namespace smallest_positive_multiple_of_45_l113_113258

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l113_113258


namespace find_f_of_3_l113_113817

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l113_113817


namespace find_x_values_for_3001_l113_113918

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
match n with
| 0 => x
| 1 => 3000
| 2 => 3001 / x
| 3 => (3000 + x) / (3000 * x)
| 4 => (x + 1) / 3000
| 5 => x
| _ => 3000

theorem find_x_values_for_3001 :
  {x : ℝ | x > 0 ∧ x = 3001 ∨ x = 1 ∨ x = 9002999}.card = 3 :=
begin
  sorry
end

end find_x_values_for_3001_l113_113918


namespace cricket_average_increase_l113_113566

theorem cricket_average_increase (runs_mean : ℕ) (innings : ℕ) (runs : ℕ) (new_runs : ℕ) (x : ℕ) :
  runs_mean = 35 → innings = 10 → runs = 79 → (total_runs : ℕ) = runs_mean * innings → 
  (new_total : ℕ) = total_runs + runs → (new_mean : ℕ) = new_total / (innings + 1) ∧ new_mean = runs_mean + x → x = 4 :=
by
  sorry

end cricket_average_increase_l113_113566


namespace equal_distribution_of_drawings_l113_113402

theorem equal_distribution_of_drawings (total_drawings : ℕ) (neighbors : ℕ) (drawings_per_neighbor : ℕ)
  (h1 : total_drawings = 54)
  (h2 : neighbors = 6)
  (h3 : total_drawings = neighbors * drawings_per_neighbor) :
  drawings_per_neighbor = 9 :=
by
  rw [h1, h2] at h3
  linarith

end equal_distribution_of_drawings_l113_113402


namespace smaller_solution_of_quadratic_l113_113337

theorem smaller_solution_of_quadratic :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 13 * x + 36 = 0) ∧ (y^2 - 13 * y + 36 = 0) ∧ min x y = 4) :=
sorry

end smaller_solution_of_quadratic_l113_113337


namespace solve_trig_eq_l113_113532

theorem solve_trig_eq {x : ℝ} (hx : cos x ≠ 0) :
  (sin (2 * x))^2 - 4 * (sin x)^2) / ((sin (2 * x))^2 + 4 * (sin x)^2 - 4) + 1 = 2 * (tan x)^2 ↔
  ∃ m : ℤ, x = π / 4 * (2 * m + 1) :=
sorry

end solve_trig_eq_l113_113532


namespace arithmetic_sequence_properties_l113_113050

variable (a_n : ℕ → ℚ)
variable (a_3 a_11 : ℚ)

notation "a₃" => a_3
notation "a₁₁" => a_11

theorem arithmetic_sequence_properties :
  a₃ = a_n 3 → a₁₁ = a_n 11 → 
  (∃ (a₁ d : ℚ), a_n n = a₁ + (n - 1) * d ∧ a₁ = 0 ∧ d = 3/2) := sorry

end arithmetic_sequence_properties_l113_113050


namespace find_f_of_3_l113_113816

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l113_113816


namespace calc_val_l113_113467

theorem calc_val : 
  (3 + 5 + 7) / (2 + 4 + 6) * (4 + 8 + 12) / (1 + 3 + 5) = 10 / 3 :=
by 
  -- Calculation proof
  sorry

end calc_val_l113_113467


namespace fill_bathtub_time_l113_113170

theorem fill_bathtub_time
  (r_cold : ℚ := 1/10)
  (r_hot : ℚ := 1/15)
  (r_empty : ℚ := -1/12)
  (net_rate : ℚ := r_cold + r_hot + r_empty) :
  net_rate = 1/12 → 
  t = 12 :=
by
  sorry

end fill_bathtub_time_l113_113170


namespace smallest_positive_multiple_of_45_l113_113287

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l113_113287


namespace most_cost_effective_80_oranges_l113_113392

noncomputable def cost_of_oranges (p1 p2 p3 : ℕ) (q1 q2 q3 : ℕ) : ℕ :=
  let cost_per_orange_p1 := p1 / q1
  let cost_per_orange_p2 := p2 / q2
  let cost_per_orange_p3 := p3 / q3
  if cost_per_orange_p3 ≤ cost_per_orange_p2 ∧ cost_per_orange_p3 ≤ cost_per_orange_p1 then
    (80 / q3) * p3
  else if cost_per_orange_p2 ≤ cost_per_orange_p1 then
    (80 / q2) * p2
  else
    (80 / q1) * p1

theorem most_cost_effective_80_oranges :
  cost_of_oranges 35 45 95 6 9 20 = 380 :=
by sorry

end most_cost_effective_80_oranges_l113_113392


namespace problem_statement_l113_113150

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) :=
  ∀ x y, x ≤ y → f x ≤ f y

noncomputable def isOddFunction (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

noncomputable def isArithmeticSeq (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem problem_statement (f : ℝ → ℝ) (a : ℕ → ℝ) (d : ℝ) (a3 : ℝ):
  isMonotonicIncreasing f →
  isOddFunction f →
  isArithmeticSeq a →
  a 3 = a3 →
  a3 > 0 →
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by
  -- proof will go here
  sorry

end problem_statement_l113_113150


namespace sara_quarters_eq_l113_113025

-- Define the initial conditions and transactions
def initial_quarters : ℕ := 21
def dad_quarters : ℕ := 49
def spent_quarters : ℕ := 15
def mom_dollars : ℕ := 2
def quarters_per_dollar : ℕ := 4
def amy_quarters (x : ℕ) := x

-- Define the function to compute total quarters
noncomputable def total_quarters (x : ℕ) : ℕ :=
initial_quarters + dad_quarters - spent_quarters + mom_dollars * quarters_per_dollar + amy_quarters x

-- Prove that the total number of quarters matches the expected value
theorem sara_quarters_eq (x : ℕ) : total_quarters x = 63 + x :=
by
  sorry

end sara_quarters_eq_l113_113025


namespace magic_card_profit_l113_113645

theorem magic_card_profit (purchase_price : ℝ) (multiplier : ℝ) (selling_price : ℝ) (profit : ℝ) 
                          (h1 : purchase_price = 100) 
                          (h2 : multiplier = 3) 
                          (h3 : selling_price = purchase_price * multiplier) 
                          (h4 : profit = selling_price - purchase_price) : 
                          profit = 200 :=
by 
  -- Here, you can introduce intermediate steps if needed.
  sorry

end magic_card_profit_l113_113645


namespace Maria_needs_72_nuts_l113_113019

theorem Maria_needs_72_nuts
    (fraction_nuts : ℚ := 1 / 4)
    (percentage_chocolate_chips : ℚ := 40 / 100)
    (nuts_per_cookie : ℕ := 2)
    (total_cookies : ℕ := 60) :
    (total_cookies * ((fraction_nuts + (1 - fraction_nuts - percentage_chocolate_chips)) * nuts_per_cookie).toRat) = 72 :=
by
    sorry

end Maria_needs_72_nuts_l113_113019


namespace smallest_positive_multiple_of_45_l113_113279

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l113_113279


namespace diagonal_not_perpendicular_l113_113635

open Real

theorem diagonal_not_perpendicular (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_a_ne_b : a ≠ b) (h_c_ne_d : c ≠ d) (h_a_ne_c : a ≠ c) (h_b_ne_d : b ≠ d): 
  ¬ ((d - b) * (b - a) = - (c - a) * (d - c)) :=
by
  sorry

end diagonal_not_perpendicular_l113_113635


namespace number_one_seventh_equals_five_l113_113774

theorem number_one_seventh_equals_five (n : ℕ) (h : n / 7 = 5) : n = 35 :=
sorry

end number_one_seventh_equals_five_l113_113774


namespace sum_of_integers_l113_113665

theorem sum_of_integers (x y : ℕ) (hxy_diff : x - y = 8) (hxy_prod : x * y = 240) (hx_gt_hy : x > y) : x + y = 32 := by
  sorry

end sum_of_integers_l113_113665


namespace circle_centers_connection_line_eq_l113_113812

-- Define the first circle equation
def circle1 (x y : ℝ) := (x^2 + y^2 - 4*x + 6*y = 0)

-- Define the second circle equation
def circle2 (x y : ℝ) := (x^2 + y^2 - 6*x = 0)

-- Given the centers of the circles, prove the equation of the line connecting them
theorem circle_centers_connection_line_eq (x y : ℝ) :
  (∀ (x y : ℝ), circle1 x y → (x = 2 ∧ y = -3)) →
  (∀ (x y : ℝ), circle2 x y → (x = 3 ∧ y = 0)) →
  (3 * x - y - 9 = 0) :=
by
  -- Here we would sketch the proof but skip it with sorry
  sorry

end circle_centers_connection_line_eq_l113_113812


namespace plain_b_area_l113_113046

theorem plain_b_area : 
  ∃ x : ℕ, (x + (x - 50) = 350) ∧ x = 200 :=
by
  sorry

end plain_b_area_l113_113046


namespace safe_unlockable_by_five_l113_113036

def min_total_keys (num_locks : ℕ) (num_people : ℕ) (key_distribution : (Fin num_locks) → (Fin num_people) → Prop) : ℕ :=
  num_locks * ((num_people + 1) / 2)

theorem safe_unlockable_by_five (num_locks : ℕ) (num_people : ℕ) 
  (key_distribution : (Fin num_locks) → (Fin num_people) → Prop) :
  (∀ (P : Finset (Fin num_people)), P.card = 5 → (∀ k : Fin num_locks, ∃ p ∈ P, key_distribution k p)) →
  min_total_keys num_locks num_people key_distribution = 20 := 
by
  sorry

end safe_unlockable_by_five_l113_113036


namespace find_f_3_l113_113846

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l113_113846


namespace zero_points_C_exist_l113_113975

theorem zero_points_C_exist (A B C : ℝ × ℝ) (hAB_dist : dist A B = 12) (h_perimeter : dist A B + dist A C + dist B C = 52)
    (h_area : abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 100) : 
    false :=
by
  sorry

end zero_points_C_exist_l113_113975


namespace max_neg_integers_l113_113555

-- Definitions for the conditions
def areIntegers (a b c d e f : Int) : Prop := True
def sumOfProductsNeg (a b c d e f : Int) : Prop := (a * b + c * d * e * f) < 0

-- The theorem to prove
theorem max_neg_integers (a b c d e f : Int) (h1 : areIntegers a b c d e f) (h2 : sumOfProductsNeg a b c d e f) : 
  ∃ s : Nat, s = 4 := 
sorry

end max_neg_integers_l113_113555


namespace probability_no_snow_l113_113695

theorem probability_no_snow {
  let p_snow : ℚ := 2/3,     -- Probability of snow on any one day
  let p_no_snow : ℚ := 1 - p_snow,  -- Probability of no snow on any one day
  let p_no_snow_5_days : ℚ := p_no_snow ^ 5  -- Probability of no snow on any of the five days
} : p_no_snow_5_days = 1 / 243 := by
  sorry

end probability_no_snow_l113_113695


namespace more_crayons_than_erasers_l113_113651

theorem more_crayons_than_erasers
  (E : ℕ) (C : ℕ) (C_left : ℕ) (E_left : ℕ)
  (hE : E = 457) (hC : C = 617) (hC_left : C_left = 523) (hE_left : E_left = E) :
  C_left - E_left = 66 := 
by
  sorry

end more_crayons_than_erasers_l113_113651


namespace arithmetic_sequence_general_term_l113_113978

theorem arithmetic_sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = 5 * n^2 + 3 * n)
  (hS₁ : a 1 = S 1)
  (hS₂ : ∀ n, a (n + 1) = S (n + 1) - S n) :
  ∀ n, a n = 10 * n - 2 :=
by
  sorry

end arithmetic_sequence_general_term_l113_113978


namespace book_arrangement_count_l113_113176

-- Conditions
def num_math_books := 4
def num_history_books := 5

-- The number of arrangements is
def arrangements (n m : Nat) : Nat :=
  let choose_end_books := n * (n - 1)
  let choose_middle_book := (n - 2)
  let remaining_books := (n - 3) + m
  choose_end_books * choose_middle_book * Nat.factorial remaining_books

theorem book_arrangement_count (n m : Nat) (h1 : n = num_math_books) (h2 : m = num_history_books) :
  arrangements n m = 120960 :=
by
  rw [h1, h2, arrangements]
  norm_num
  sorry

end book_arrangement_count_l113_113176


namespace greatest_k_dividing_n_l113_113306

theorem greatest_k_dividing_n (n : ℕ) 
  (h1 : Nat.totient n = 72) 
  (h2 : Nat.totient (3 * n) = 96) : ∃ k : ℕ, 3^k ∣ n ∧ ∀ j : ℕ, 3^j ∣ n → j ≤ 2 := 
by {
  sorry
}

end greatest_k_dividing_n_l113_113306


namespace repeating_decimal_sum_l113_113597

def repeating_decimal_to_fraction (d : ℕ) (n : ℕ) : ℚ := n / ((10^d) - 1)

theorem repeating_decimal_sum : 
  repeating_decimal_to_fraction 1 2 + repeating_decimal_to_fraction 2 2 + repeating_decimal_to_fraction 4 2 = 2474646 / 9999 := 
sorry

end repeating_decimal_sum_l113_113597


namespace union_of_A_and_B_l113_113156

def set_A : Set Int := {0, 1}
def set_B : Set Int := {0, -1}

theorem union_of_A_and_B : set_A ∪ set_B = {-1, 0, 1} := by
  sorry

end union_of_A_and_B_l113_113156


namespace boat_speed_in_still_water_l113_113370

-- Boat's speed in still water in km/hr
variable (B S : ℝ)

-- Conditions given for the boat's speed along and against the stream
axiom cond1 : B + S = 11
axiom cond2 : B - S = 5

-- Prove that the speed of the boat in still water is 8 km/hr
theorem boat_speed_in_still_water : B = 8 :=
by
  sorry

end boat_speed_in_still_water_l113_113370


namespace square_area_720_l113_113391

noncomputable def length_squared {α : Type*} [EuclideanDomain α] (a b : α) := a * a + b * b

theorem square_area_720
  (side x : ℝ)
  (h1 : BE = 20) (h2 : EF = 20) (h3 : FD = 20)
  (h4 : AE = 2 * ED) (h5 : BF = 2 * FC)
  : x * x = 720 :=
by
  let AE := 2/3 * side
  let ED := 1/3 * side
  let BF := 2/3 * side
  let FC := 1/3 * side
  have h6 : length_squared BF EF = BE * BE := sorry
  have h7 : x * x = 720 := sorry
  exact h7

end square_area_720_l113_113391


namespace sin_60_equiv_l113_113472

theorem sin_60_equiv : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := 
by
  sorry

end sin_60_equiv_l113_113472


namespace percy_bound_longer_martha_step_l113_113521

theorem percy_bound_longer_martha_step (steps_per_gap_martha: ℕ) (bounds_per_gap_percy: ℕ)
  (gaps: ℕ) (total_distance: ℕ) 
  (step_length_martha: ℝ) (bound_length_percy: ℝ) :
  steps_per_gap_martha = 50 →
  bounds_per_gap_percy = 15 →
  gaps = 50 →
  total_distance = 10560 →
  step_length_martha = total_distance / (steps_per_gap_martha * gaps) →
  bound_length_percy = total_distance / (bounds_per_gap_percy * gaps) →
  (bound_length_percy - step_length_martha) = 10 :=
by
  sorry

end percy_bound_longer_martha_step_l113_113521


namespace value_of_x_in_logarithm_equation_l113_113177

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem value_of_x_in_logarithm_equation (n : ℝ) (h1 : n = 343) : 
  ∃ (x : ℝ), log_base x n + log_base 7 n = log_base 1 n :=
by
  sorry

end value_of_x_in_logarithm_equation_l113_113177


namespace problem1_problem2_l113_113734

-- Problem 1: Prove \( \sqrt{10} \times \sqrt{2} + \sqrt{15} \div \sqrt{3} = 3\sqrt{5} \)
theorem problem1 : Real.sqrt 10 * Real.sqrt 2 + Real.sqrt 15 / Real.sqrt 3 = 3 * Real.sqrt 5 := 
by sorry

-- Problem 2: Prove \( \sqrt{27} - (\sqrt{12} - \sqrt{\frac{1}{3}}) = \frac{4\sqrt{3}}{3} \)
theorem problem2 : Real.sqrt 27 - (Real.sqrt 12 - Real.sqrt (1 / 3)) = (4 * Real.sqrt 3) / 3 :=
by sorry

end problem1_problem2_l113_113734


namespace skirt_price_is_13_l113_113987

-- Definitions based on conditions
def skirts_cost (S : ℝ) : ℝ := 2 * S
def blouses_cost : ℝ := 3 * 6
def total_cost (S : ℝ) : ℝ := skirts_cost S + blouses_cost
def amount_spent : ℝ := 100 - 56

-- The statement we want to prove
theorem skirt_price_is_13 (S : ℝ) (h : total_cost S = amount_spent) : S = 13 :=
by sorry

end skirt_price_is_13_l113_113987


namespace alpha_parallel_to_beta_l113_113615

variables (a b : ℝ → ℝ → ℝ) (α β : ℝ → ℝ)

-- Definitions based on conditions
def are_distinct_lines : a ≠ b := sorry
def are_distinct_planes : α ≠ β := sorry

def line_parallel_to_plane (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ) : Prop := sorry -- Define parallel relation
def line_perpendicular_to_plane (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ) : Prop := sorry -- Define perpendicular relation
def planes_parallel (p1 p2 : ℝ → ℝ) : Prop := sorry -- Define planes being parallel

-- Given as conditions
axiom a_perpendicular_to_alpha : line_perpendicular_to_plane a α
axiom b_perpendicular_to_beta : line_perpendicular_to_plane b β
axiom a_parallel_to_b : a = b

-- The proposition to prove
theorem alpha_parallel_to_beta : planes_parallel α β :=
by {
  -- Placeholder for the logic provided through the previous solution steps.
  sorry
}

end alpha_parallel_to_beta_l113_113615


namespace part1_part2_l113_113155

-- Definitions and assumptions based on the problem
def f (x a : ℝ) : ℝ := abs (x - a)

-- Condition (1) with given function and inequality solution set
theorem part1 (a : ℝ) :
  (∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  sorry

-- Condition (2) with the range of m under the previously found value of a
theorem part2 (m : ℝ) :
  (∃ x, f x 2 + f (x + 5) 2 < m) → m > 5 :=
by
  sorry

end part1_part2_l113_113155


namespace present_age_ratio_l113_113212

-- Define the variables and the conditions
variable (S M : ℕ)

-- Condition 1: Sandy's present age is 84 because she was 78 six years ago
def present_age_sandy := S = 84

-- Condition 2: Sixteen years from now, the ratio of their ages is 5:2
def age_ratio_16_years := (S + 16) * 2 = 5 * (M + 16)

-- The goal: The present age ratio of Sandy to Molly is 7:2
theorem present_age_ratio {S M : ℕ} (h1 : S = 84) (h2 : (S + 16) * 2 = 5 * (M + 16)) : S / M = 7 / 2 :=
by
  -- Integrating conditions
  have hS : S = 84 := h1
  have hR : (S + 16) * 2 = 5 * (M + 16) := h2
  -- We need a proof here, but we'll skip it for now
  sorry

end present_age_ratio_l113_113212


namespace range_of_k_for_circle_l113_113769

theorem range_of_k_for_circle (x y : ℝ) (k : ℝ) : 
  (x^2 + y^2 - 4*x + 2*y + 5*k = 0) → k < 1 :=
by 
  sorry

end range_of_k_for_circle_l113_113769


namespace linda_savings_l113_113792

theorem linda_savings (S : ℝ) (h : (1 / 2) * S = 300) : S = 600 :=
sorry

end linda_savings_l113_113792


namespace decreasing_function_range_l113_113471

theorem decreasing_function_range (k : ℝ) : (∀ x : ℝ, k + 2 < 0) ↔ k < -2 :=
by
  sorry

end decreasing_function_range_l113_113471


namespace triangle_side_relationship_l113_113483

theorem triangle_side_relationship
  (a b c : ℝ)
  (habc : a < b + c)
  (ha_pos : a > 0) :
  a^2 < a * b + a * c :=
by
  sorry

end triangle_side_relationship_l113_113483


namespace lionsAfterOneYear_l113_113868

-- Definitions based on problem conditions
def initialLions : Nat := 100
def birthRate : Nat := 5
def deathRate : Nat := 1
def monthsInYear : Nat := 12

-- Theorem statement
theorem lionsAfterOneYear :
  initialLions + birthRate * monthsInYear - deathRate * monthsInYear = 148 :=
by
  sorry

end lionsAfterOneYear_l113_113868


namespace smallest_positive_multiple_45_l113_113241

theorem smallest_positive_multiple_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  simp
  split
  · exact Nat.one_pos
  · ring

end smallest_positive_multiple_45_l113_113241


namespace binomial_19_10_l113_113915

theorem binomial_19_10 :
  ∀ (binom : ℕ → ℕ → ℕ),
  binom 17 7 = 19448 → binom 17 9 = 24310 →
  binom 19 10 = 92378 :=
by
  intros
  sorry

end binomial_19_10_l113_113915


namespace calculate_weight_difference_l113_113290

noncomputable def joe_weight := 43 -- Joe's weight in kg
noncomputable def original_avg_weight := 30 -- Original average weight in kg
noncomputable def new_avg_weight := 31 -- New average weight in kg after Joe joins
noncomputable def final_avg_weight := 30 -- Final average weight after two students leave

theorem calculate_weight_difference :
  ∃ (n : ℕ) (x : ℝ), 
  (original_avg_weight * n + joe_weight) / (n + 1) = new_avg_weight ∧
  (new_avg_weight * (n + 1) - 2 * x) / (n - 1) = final_avg_weight →
  x - joe_weight = -6.5 :=
by
  sorry

end calculate_weight_difference_l113_113290


namespace functional_identity_l113_113016

-- Define the set of non-negative integers
def S : Set ℕ := {n | n ≥ 0}

-- Define the function f with the required domain and codomain
def f (n : ℕ) : ℕ := n

-- The hypothesis: the functional equation satisfied by f
axiom functional_equation :
  ∀ m n : ℕ, f (m + f n) = f (f m) + f n

-- The theorem we want to prove
theorem functional_identity (n : ℕ) : f n = n :=
  sorry

end functional_identity_l113_113016


namespace decreasing_function_on_real_l113_113732

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x + y) = f x + f y
axiom f_negative (x : ℝ) : x > 0 → f x < 0
axiom f_not_identically_zero : ∃ x, f x ≠ 0

theorem decreasing_function_on_real :
  ∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2 :=
sorry

end decreasing_function_on_real_l113_113732


namespace dried_grapes_weight_l113_113344

def fresh_grapes_weight : ℝ := 30
def fresh_grapes_water_percentage : ℝ := 0.60
def dried_grapes_water_percentage : ℝ := 0.20

theorem dried_grapes_weight :
  let non_water_content := fresh_grapes_weight * (1 - fresh_grapes_water_percentage)
  let dried_grapes := non_water_content / (1 - dried_grapes_water_percentage)
  dried_grapes = 15 :=
by
  let non_water_content := fresh_grapes_weight * (1 - fresh_grapes_water_percentage)
  let dried_grapes := non_water_content / (1 - dried_grapes_water_percentage)
  show dried_grapes = 15
  sorry

end dried_grapes_weight_l113_113344


namespace factorize_expression_l113_113746

variable (x y : ℝ)

theorem factorize_expression : xy^2 + 6*xy + 9*x = x*(y + 3)^2 := by
  sorry

end factorize_expression_l113_113746


namespace abs_diff_expr_l113_113095

theorem abs_diff_expr :
  let a := -3 * (7 - 15)
  let b := (5 - 7)^2 + (-4)^2
  |a| - |b| = 4 :=
by
  let a := -3 * (7 - 15)
  let b := (5 - 7)^2 + (-4)^2
  sorry

end abs_diff_expr_l113_113095


namespace john_heroes_on_large_sheets_front_l113_113374

noncomputable def num_pictures_on_large_sheets_front : ℕ :=
  let total_pictures := 20
  let minutes_spent := 75 - 5
  let average_time_per_picture := 5
  let front_pictures := total_pictures / 2
  let x := front_pictures / 3
  2 * x

theorem john_heroes_on_large_sheets_front : num_pictures_on_large_sheets_front = 6 :=
by
  sorry

end john_heroes_on_large_sheets_front_l113_113374


namespace ordered_pair_and_sum_of_squares_l113_113604

theorem ordered_pair_and_sum_of_squares :
  ∃ x y : ℚ, 
    6 * x - 48 * y = 2 ∧ 
    3 * y - x = 4 ∧ 
    x ^ 2 + y ^ 2 = 442 / 25 :=
by
  sorry

end ordered_pair_and_sum_of_squares_l113_113604


namespace digit_2023_in_7_div_26_is_3_l113_113330

-- We define the decimal expansion of 7/26 as a repeating sequence of "269230769"
def repeating_block : string := "269230769"

-- Verify that the 2023rd digit in the sequence is "3"
theorem digit_2023_in_7_div_26_is_3 :
  (repeating_block.str.to_list.nth ((2023 % 9) - 1)).iget = '3' :=
by
  sorry

end digit_2023_in_7_div_26_is_3_l113_113330


namespace find_f_of_3_l113_113824

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l113_113824


namespace margarita_vs_ricciana_l113_113198

-- Ricciana's distances
def ricciana_run : ℕ := 20
def ricciana_jump : ℕ := 4
def ricciana_total : ℕ := ricciana_run + ricciana_jump

-- Margarita's distances
def margarita_run : ℕ := 18
def margarita_jump : ℕ := (2 * ricciana_jump) - 1
def margarita_total : ℕ := margarita_run + margarita_jump

-- Statement to prove Margarita ran and jumped 1 more foot than Ricciana
theorem margarita_vs_ricciana : margarita_total = ricciana_total + 1 := by
  sorry

end margarita_vs_ricciana_l113_113198


namespace parallelogram_area_find_perpendicular_vector_l113_113360

noncomputable def pointA : (ℝ × ℝ × ℝ) := (0, 2, 3)
noncomputable def pointB : (ℝ × ℝ × ℝ) := (-2, 1, 6)
noncomputable def pointC : (ℝ × ℝ × ℝ) := (1, -1, 5)

noncomputable def vecAB := (pointB.1 - pointA.1, pointB.2 - pointA.2, pointB.3 - pointA.3)
noncomputable def vecAC := (pointC.1 - pointA.1, pointC.2 - pointA.2, pointC.3 - pointA.3)

noncomputable def crossProd (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

noncomputable def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2 + u.3^2)

theorem parallelogram_area :
  magnitude (crossProd vecAB vecAC) = Real.sqrt 123 :=
  sorry

theorem find_perpendicular_vector (a : ℝ × ℝ × ℝ) :
  (vecAB.1 * a.1 + vecAB.2 * a.2 + vecAB.3 * a.3 = 0) ∧
  (vecAC.1 * a.1 + vecAC.2 * a.2 + vecAC.3 * a.3 = 0) ∧
  (magnitude a = 3) →
  (a = (1, 1, 1) ∨ a = (-1, -1, -1)) :=
  sorry

end parallelogram_area_find_perpendicular_vector_l113_113360


namespace find_f_3_l113_113842

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l113_113842


namespace find_larger_number_l113_113061

theorem find_larger_number :
  ∃ x y : ℤ, x + y = 30 ∧ 2 * y - x = 6 ∧ x > y ∧ x = 18 :=
by
  sorry

end find_larger_number_l113_113061


namespace greatest_number_dividing_1642_and_1856_l113_113875

theorem greatest_number_dividing_1642_and_1856 (a b r1 r2 k : ℤ) (h_intro : a = 1642) (h_intro2 : b = 1856) 
    (h_r1 : r1 = 6) (h_r2 : r2 = 4) (h_k1 : k = Int.gcd (a - r1) (b - r2)) :
    k = 4 :=
by
  sorry

end greatest_number_dividing_1642_and_1856_l113_113875


namespace probability_two_point_distribution_l113_113153

theorem probability_two_point_distribution 
  (P : ℕ → ℚ)
  (two_point_dist : P 0 + P 1 = 1)
  (condition : P 1 = (3 / 2) * P 0) :
  P 1 = 3 / 5 :=
by
  sorry

end probability_two_point_distribution_l113_113153


namespace smallest_positive_multiple_l113_113282

theorem smallest_positive_multiple (x : ℕ) (hx : x > 0) :
  (45 * x) = 45 ↔ x = 1 :=
by {
  split;
  intro h;
  sorry;
}

end smallest_positive_multiple_l113_113282


namespace smallest_positive_multiple_of_45_l113_113277

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * (x : ℕ) = 45 ∧ (∀ y : ℕ+, 45 * (y : ℕ) ≥ 45) :=
by
  have h : 45 * (1 : ℕ) = 45 := by norm_num,
  use 1,
  split,
  exact h,
  intro y,
  exact mul_le_mul_of_nonneg_left (y.2) (by norm_num)

end smallest_positive_multiple_of_45_l113_113277


namespace pow_two_gt_cube_l113_113193

theorem pow_two_gt_cube (n : ℕ) (h : 10 ≤ n) : 2^n > n^3 := sorry

end pow_two_gt_cube_l113_113193


namespace trapezoid_leg_length_l113_113535

theorem trapezoid_leg_length (S : ℝ) (h₁ : S > 0) : 
  ∃ x : ℝ, x = Real.sqrt (2 * S) ∧ x > 0 :=
by
  sorry

end trapezoid_leg_length_l113_113535


namespace percentage_shoes_polished_l113_113587

theorem percentage_shoes_polished (total_pairs : ℕ) (shoes_to_polish : ℕ)
  (total_individual_shoes : ℕ := total_pairs * 2)
  (shoes_polished : ℕ := total_individual_shoes - shoes_to_polish)
  (percentage_polished : ℚ := (shoes_polished : ℚ) / total_individual_shoes * 100) :
  total_pairs = 10 → shoes_to_polish = 11 → percentage_polished = 45 :=
by
  intros hpairs hleft
  sorry

end percentage_shoes_polished_l113_113587


namespace pair_not_product_48_l113_113088

theorem pair_not_product_48:
  (∀(a b : ℤ), (a, b) = (-6, -8)                    → a * b = 48) ∧
  (∀(a b : ℤ), (a, b) = (-4, -12)                   → a * b = 48) ∧
  (∀(a b : ℚ), (a, b) = (3/4, -64)                  → a * b ≠ 48) ∧
  (∀(a b : ℤ), (a, b) = (3, 16)                     → a * b = 48) ∧
  (∀(a b : ℚ), (a, b) = (4/3, 36)                   → a * b = 48)
  :=
by
  sorry

end pair_not_product_48_l113_113088


namespace inverse_sum_is_minus_two_l113_113519

variable (f : ℝ → ℝ)
variable (h_injective : Function.Injective f)
variable (h_surjective : Function.Surjective f)
variable (h_eq : ∀ x : ℝ, f (x + 1) + f (-x - 3) = 2)

theorem inverse_sum_is_minus_two (x : ℝ) : f⁻¹ (2009 - x) + f⁻¹ (x - 2007) = -2 := 
  sorry

end inverse_sum_is_minus_two_l113_113519


namespace min_passengers_on_vehicle_with_no_adjacent_seats_l113_113743

-- Define the seating arrangement and adjacency rules

structure Seat :=
(row : Fin 2) (col : Fin 5)

def adjacent (a b : Seat) : Prop :=
(a.row = b.row ∧ (a.col = b.col + 1 ∨ a.col + 1 = b.col)) ∨
(a.col = b.col ∧ (a.row = b.row + 1 ∨ a.row + 1 = b.row))

def valid_seating (seated : List Seat) : Prop :=
∀ (i j : Seat), i ∈ seated → j ∈ seated → adjacent i j → false

def min_passengers : ℕ :=
5

theorem min_passengers_on_vehicle_with_no_adjacent_seats :
∃ seated : List Seat, valid_seating seated ∧ List.length seated = min_passengers :=
sorry

end min_passengers_on_vehicle_with_no_adjacent_seats_l113_113743


namespace solution_set_of_inequality_l113_113863

theorem solution_set_of_inequality :
  { x : ℝ | x * (x - 1) ≤ 0 } = { x : ℝ | 0 ≤ x ∧ x ≤ 1 } :=
by
sorry

end solution_set_of_inequality_l113_113863


namespace triangle_area_bounded_by_lines_l113_113074

theorem triangle_area_bounded_by_lines :
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := dist A B
  let height := 8
  triangle_area A B O = 64 :=
sorry

end triangle_area_bounded_by_lines_l113_113074


namespace compute_expression_l113_113553

theorem compute_expression :
  (1 / 36) / ((1 / 4) + (1 / 12) - (7 / 18) - (1 / 36)) + 
  (((1 / 4) + (1 / 12) - (7 / 18) - (1 / 36)) / (1 / 36)) = -10 / 3 :=
by
  sorry

end compute_expression_l113_113553


namespace total_students_l113_113368

theorem total_students (ratio_boys : ℕ) (ratio_girls : ℕ) (num_girls : ℕ) 
  (h_ratio : ratio_boys = 8) (h_ratio_girls : ratio_girls = 5) (h_num_girls : num_girls = 175) : 
  ratio_boys * (num_girls / ratio_girls) + num_girls = 455 :=
by
  sorry

end total_students_l113_113368


namespace log_equality_l113_113933

theorem log_equality (x : ℝ) : (8 : ℝ)^x = 16 ↔ x = 4 / 3 :=
by
  sorry

end log_equality_l113_113933


namespace smallest_positive_multiple_of_45_l113_113256

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l113_113256


namespace maximum_smallest_triplet_sum_l113_113646

theorem maximum_smallest_triplet_sum (circle : Fin 10 → ℕ) (h : ∀ i : Fin 10, 1 ≤ circle i ∧ circle i ≤ 10 ∧ ∀ j k, j ≠ k → circle j ≠ circle k):
  ∃ (i : Fin 10), ∀ j ∈ ({i, i + 1, i + 2} : Finset (Fin 10)), circle i + circle (i + 1) + circle (i + 2) ≤ 15 :=
sorry

end maximum_smallest_triplet_sum_l113_113646


namespace hours_per_day_l113_113181

theorem hours_per_day
  (num_warehouse : ℕ := 4)
  (num_managers : ℕ := 2)
  (rate_warehouse : ℝ := 15)
  (rate_manager : ℝ := 20)
  (tax_rate : ℝ := 0.10)
  (days_worked : ℕ := 25)
  (total_cost : ℝ := 22000) :
  ∃ h : ℝ, 6 * h * days_worked * (rate_warehouse + rate_manager) * (1 + tax_rate) = total_cost ∧ h = 8 :=
by
  sorry

end hours_per_day_l113_113181


namespace gcd_repeated_three_digit_integers_l113_113102

theorem gcd_repeated_three_digit_integers : 
  ∀ m ∈ {n | 100 ≤ n ∧ n < 1000}, 
  gcd (1001 * m) (1001 * (m + 1)) = 1001 :=
by
  sorry

end gcd_repeated_three_digit_integers_l113_113102


namespace probability_of_green_ball_l113_113127

-- Define the number of balls in each container
def number_balls_I := (10, 5)  -- (red, green)
def number_balls_II := (3, 6)  -- (red, green)
def number_balls_III := (3, 6)  -- (red, green)

-- Define the probability of selecting each container
noncomputable def probability_container_selected := (1 / 3 : ℝ)

-- Define the probability of drawing a green ball from each container
noncomputable def probability_green_I := (number_balls_I.snd : ℝ) / ((number_balls_I.fst + number_balls_I.snd) : ℝ)
noncomputable def probability_green_II := (number_balls_II.snd : ℝ) / ((number_balls_II.fst + number_balls_II.snd) : ℝ)
noncomputable def probability_green_III := (number_balls_III.snd : ℝ) / ((number_balls_III.fst + number_balls_III.snd) : ℝ)

-- Define the combined probabilities for drawing a green ball and selecting each container
noncomputable def combined_probability_I := probability_container_selected * probability_green_I
noncomputable def combined_probability_II := probability_container_selected * probability_green_II
noncomputable def combined_probability_III := probability_container_selected * probability_green_III

-- Define the total probability of drawing a green ball
noncomputable def total_probability_green := combined_probability_I + combined_probability_II + combined_probability_III

-- The theorem to be proved
theorem probability_of_green_ball : total_probability_green = (5 / 9 : ℝ) :=
by
  sorry

end probability_of_green_ball_l113_113127


namespace primes_squared_mod_180_l113_113586

theorem primes_squared_mod_180 (p : ℕ) (hp_prime : Prime p) (hp_gt_5 : p > 5) :
  {r | ∃ k : ℕ, p^2 = 180 * k + r}.card = 2 :=
by
  sorry

end primes_squared_mod_180_l113_113586


namespace total_cans_given_away_l113_113901

-- Define constants
def initial_stock : ℕ := 2000

-- Define conditions day 1
def people_day1 : ℕ := 500
def cans_per_person_day1 : ℕ := 1
def restock_day1 : ℕ := 1500

-- Define conditions day 2
def people_day2 : ℕ := 1000
def cans_per_person_day2 : ℕ := 2
def restock_day2 : ℕ := 3000

-- Define the question as a theorem
theorem total_cans_given_away : (people_day1 * cans_per_person_day1 + people_day2 * cans_per_person_day2) = 2500 := by
  sorry

end total_cans_given_away_l113_113901


namespace spends_at_arcade_each_weekend_l113_113705

def vanessa_savings : ℕ := 20
def parents_weekly_allowance : ℕ := 30
def dress_cost : ℕ := 80
def weeks : ℕ := 3

theorem spends_at_arcade_each_weekend (arcade_weekend_expense : ℕ) :
  (vanessa_savings + weeks * parents_weekly_allowance - dress_cost = weeks * parents_weekly_allowance - arcade_weekend_expense * weeks) →
  arcade_weekend_expense = 30 :=
by
  intro h
  sorry

end spends_at_arcade_each_weekend_l113_113705


namespace exam_question_combinations_l113_113777

theorem exam_question_combinations : 
  let questions := Finset.range 9 in
  let first_five := Finset.range 5 in
  let count_ways_to_choose := 
    (@Finset.choose (Fin 5) _ _ ⟨3, by norm_num⟩).card * 
    (@Finset.choose (Fin 4) _ _ ⟨3, by norm_num⟩).card +
    (@Finset.choose (Fin 5) _ _ ⟨4, by norm_num⟩).card * 
    (@Finset.choose (Fin 4) _ _ ⟨2, by norm_num⟩).card +
    (@Finset.choose (Fin 5) _ _ ⟨5, by norm_num⟩).card * 
    (@Finset.choose (Fin 4) _ _ ⟨1, by norm_num⟩).card
  in count_ways_to_choose = 74 :=
by sorry

end exam_question_combinations_l113_113777


namespace unique_combinations_bathing_suits_l113_113299

theorem unique_combinations_bathing_suits
  (men_styles : ℕ) (men_sizes : ℕ) (men_colors : ℕ)
  (women_styles : ℕ) (women_sizes : ℕ) (women_colors : ℕ)
  (h_men_styles : men_styles = 5) (h_men_sizes : men_sizes = 3) (h_men_colors : men_colors = 4)
  (h_women_styles : women_styles = 4) (h_women_sizes : women_sizes = 4) (h_women_colors : women_colors = 5) :
  men_styles * men_sizes * men_colors + women_styles * women_sizes * women_colors = 140 :=
by
  sorry

end unique_combinations_bathing_suits_l113_113299


namespace find_f_of_3_l113_113822

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l113_113822


namespace divisible_by_120_l113_113517

theorem divisible_by_120 (n : ℕ) (hn_pos : n > 0) : 120 ∣ n * (n^2 - 1) * (n^2 - 5 * n + 26) := 
by
  sorry

end divisible_by_120_l113_113517


namespace min_reciprocal_sum_l113_113186

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (1/x) + (1/y) = 3 + 2 * Real.sqrt 2 :=
sorry

end min_reciprocal_sum_l113_113186


namespace find_f_of_3_l113_113829

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l113_113829


namespace find_x_given_scores_l113_113951

theorem find_x_given_scores : 
  ∃ x : ℝ, (9.1 + 9.3 + x + 9.2 + 9.4) / 5 = 9.3 ∧ x = 9.5 :=
by {
  sorry
}

end find_x_given_scores_l113_113951


namespace mental_math_competition_l113_113203

theorem mental_math_competition :
  -- The number of teams that participated is 4
  (∃ (teams : ℕ) (numbers : List ℕ),
     -- Each team received a number that can be written as 15M + 11m where M is the largest odd divisor
     -- and m is the smallest odd divisor greater than 1.
     teams = 4 ∧ 
     numbers = [528, 880, 1232, 1936] ∧
     ∀ n ∈ numbers,
       ∃ M m, M > 1 ∧ m > 1 ∧
       M % 2 = 1 ∧ m % 2 = 1 ∧
       (∀ d, d ∣ n → (d % 2 = 1 → M ≥ d)) ∧ 
       (∀ d, d ∣ n → (d % 2 = 1 ∧ d > 1 → m ≤ d)) ∧
       n = 15 * M + 11 * m) :=
sorry

end mental_math_competition_l113_113203


namespace coffee_price_increase_l113_113207

variable (C : ℝ) -- cost per pound of green tea and coffee in June
variable (P_green_tea_july : ℝ := 0.1) -- price of green tea per pound in July
variable (mixture_cost : ℝ := 3.15) -- cost of mixture of equal quantities of green tea and coffee for 3 lbs
variable (green_tea_cost_per_lb_july : ℝ := 0.1) -- cost per pound of green tea in July
variable (green_tea_weight : ℝ := 1.5) -- weight of green tea in the mixture in lbs
variable (coffee_weight : ℝ := 1.5) -- weight of coffee in the mixture in lbs
variable (coffee_cost_per_lb_july : ℝ := 2.0) -- cost per pound of coffee in July

theorem coffee_price_increase :
  C = 1 → mixture_cost = 3.15 →
  P_green_tea_july * C = green_tea_cost_per_lb_july →
  green_tea_weight * green_tea_cost_per_lb_july + coffee_weight * coffee_cost_per_lb_july = mixture_cost →
  (coffee_cost_per_lb_july - C) / C * 100 = 100 :=
by
  intros
  sorry

end coffee_price_increase_l113_113207


namespace pyramid_volume_l113_113568

-- Definitions based on the given conditions
def AB : ℝ := 15
def AD : ℝ := 8
def Area_Δ_ABE : ℝ := 120
def Area_Δ_CDE : ℝ := 64
def h : ℝ := 16
def Base_Area : ℝ := AB * AD

-- Statement to prove the volume of the pyramid is 640
theorem pyramid_volume : (1 / 3) * Base_Area * h = 640 :=
sorry

end pyramid_volume_l113_113568


namespace geom_seq_thm_l113_113758

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
a 1 ≠ 0 ∧ ∀ n, a (n + 1) = (a n ^ 2) / (a (n - 1))

theorem geom_seq_thm (a : ℕ → ℝ) (h : geom_seq a) (h_neg : ∀ n, a n < 0) 
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) : a 3 + a 5 = -6 :=
by
  sorry

end geom_seq_thm_l113_113758


namespace Aiyanna_more_than_Alyssa_Brady_fewer_than_Aiyanna_Brady_more_than_Alyssa_l113_113578

-- Defining the number of cookies each person had
def Alyssa_cookies : ℕ := 1523
def Aiyanna_cookies : ℕ := 3720
def Brady_cookies : ℕ := 2265

-- Proving the statements
theorem Aiyanna_more_than_Alyssa : Aiyanna_cookies - Alyssa_cookies = 2197 := by
  sorry

theorem Brady_fewer_than_Aiyanna : Aiyanna_cookies - Brady_cookies = 1455 := by
  sorry

theorem Brady_more_than_Alyssa : Brady_cookies - Alyssa_cookies = 742 := by
  sorry

end Aiyanna_more_than_Alyssa_Brady_fewer_than_Aiyanna_Brady_more_than_Alyssa_l113_113578


namespace find_f_of_3_l113_113821

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l113_113821


namespace find_f_of_3_l113_113828

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l113_113828


namespace simplify_expr_l113_113658

theorem simplify_expr (a : ℝ) (h : a > 1) : (1 - a) * (1 / (a - 1)).sqrt = -(a - 1).sqrt :=
sorry

end simplify_expr_l113_113658


namespace area_triangle_OCD_l113_113065

open Real

-- Define points C and D based on the given conditions.
def C := (8, 8)
def D := (-8, 8)

-- Define the base length between points C and D.
def base := dist C D

-- Height from the origin to the line y = 8.
def height := 8

-- Statement to prove the area of triangle OCD.
theorem area_triangle_OCD : (1 / 2) * base * height = 64 := by
  sorry

end area_triangle_OCD_l113_113065


namespace negative_half_less_than_negative_third_l113_113905

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end negative_half_less_than_negative_third_l113_113905


namespace player2_winning_strategy_l113_113455

-- Definitions of the game setup
def initial_position_player1 := (1, 1)
def initial_position_player2 := (998, 1998)

def adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 - 1 ∨ p1.2 = p2.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 - 1 ∨ p1.1 = p2.1 + 1))

-- A function defining the winning condition for Player 2
def player2_wins (p1 p2 : ℕ × ℕ) : Prop :=
  p1 = p2 ∨ p1.1 = (initial_position_player2.1)

-- Theorem stating the pair (998, 1998) guarantees a win for Player 2
theorem player2_winning_strategy : player2_wins (998, 0) (998, 1998) :=
sorry

end player2_winning_strategy_l113_113455


namespace prime_square_mod_180_l113_113585

theorem prime_square_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : 5 < p) :
  ∃ (s : Finset ℕ), s.card = 2 ∧ ∀ r ∈ s, ∃ n : ℕ, p^2 % 180 = r :=
sorry

end prime_square_mod_180_l113_113585


namespace probability_winning_ticket_l113_113192

open Finset

theorem probability_winning_ticket : 
  let s := ({1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40} : Finset ℕ) in
  (∃ (chosen : Finset ℕ), chosen.card = 6 ∧ chosen ⊆ s ∧ 
    ∃ (win : Finset ℕ), win.card = 6 ∧ win ⊆ s ∧ 
    (∑ x in chosen, Real.log x / Real.log 10).denom = 1 ∧
    (∑ x in win, Real.log x / Real.log 10).denom = 1 ∧
    chosen = win) →
  (1 / 4 : ℝ) :=
by
  sorry

end probability_winning_ticket_l113_113192


namespace matrix_inv_correct_l113_113333

open Matrix

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  !![3, 4;
   -2, 9]

noncomputable def matrix_A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![
    (9 : ℚ)/35, (-4 : ℚ)/35;
    (2 : ℚ)/35, (3 : ℚ)/35]

theorem matrix_inv_correct : matrix_A⁻¹ = matrix_A_inv := by
  sorry

end matrix_inv_correct_l113_113333


namespace gcd_7654321_6789012_l113_113316

theorem gcd_7654321_6789012 : Nat.gcd 7654321 6789012 = 3 := by
  sorry

end gcd_7654321_6789012_l113_113316


namespace smallest_value_geq_4_l113_113011

noncomputable def smallest_value (a b c d : ℝ) : ℝ :=
  (a + b + c + d) * ((1 / (a + b + d)) + (1 / (a + c + d)) + (1 / (b + c + d)))

theorem smallest_value_geq_4 (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  smallest_value a b c d ≥ 4 :=
by
  sorry

end smallest_value_geq_4_l113_113011


namespace no_real_solution_arctan_eqn_l113_113029

theorem no_real_solution_arctan_eqn :
  ¬∃ x : ℝ, 0 < x ∧ (Real.arctan (1 / x ^ 2) + Real.arctan (1 / x ^ 4) = (Real.pi / 4)) :=
by
  sorry

end no_real_solution_arctan_eqn_l113_113029


namespace probability_face_cards_l113_113703

theorem probability_face_cards :
  let first_card_hearts_face := 3 / 52
  let second_card_clubs_face_after_hearts := 3 / 51
  let combined_probability := first_card_hearts_face * second_card_clubs_face_after_hearts
  combined_probability = 1 / 294 :=
by 
  sorry

end probability_face_cards_l113_113703


namespace players_without_cautions_l113_113319

theorem players_without_cautions (Y N : ℕ) (h1 : Y + N = 11) (h2 : Y = 6) : N = 5 :=
by
  sorry

end players_without_cautions_l113_113319


namespace tangent_lines_coincide_l113_113485

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := f x / x

theorem tangent_lines_coincide : 
  f(2) = 2 ∧ 
  (∀ x, g x = f x / x) ∧ 
  (2 * deriv f 2 - f 2) / 4 = 1/2 → 
  deriv f 2 = 2 :=
by
  intros h,
  cases' h with h1 h2,
  cases' h2 with h2 h3,
  have : f 2 = 2 := h1,
  have : (2 * deriv f 2 - f 2) / 4 = 1 / 2 := h3,
  sorry

end tangent_lines_coincide_l113_113485


namespace steve_final_height_l113_113406

-- Define the initial height of Steve in inches.
def initial_height : ℕ := 5 * 12 + 6

-- Define how many inches Steve grew.
def growth : ℕ := 6

-- Define Steve's final height after growing.
def final_height : ℕ := initial_height + growth

-- The final height should be 72 inches.
theorem steve_final_height : final_height = 72 := by
  -- we don't provide the proof here
  sorry

end steve_final_height_l113_113406


namespace find_A_l113_113188

theorem find_A (A : ℝ) (h : (12 + 3) * (12 - A) = 120) : A = 4 :=
by sorry

end find_A_l113_113188


namespace difference_between_two_numbers_l113_113866

theorem difference_between_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 10) 
  (h3 : x^2 - y^2 = 200) : 
  x - y = 10 :=
by 
  sorry

end difference_between_two_numbers_l113_113866


namespace complement_of_intersection_l113_113357

theorem complement_of_intersection (U M N : Set ℕ)
  (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3})
  (hN : N = {2, 3, 4}) :
  (U \ (M ∩ N)) = {1, 4} :=
by
  rw [hU, hM, hN]
  sorry

end complement_of_intersection_l113_113357


namespace number_of_paintings_per_new_gallery_l113_113313

-- Define all the conditions as variables/constants
def pictures_original : Nat := 9
def new_galleries : Nat := 5
def pencils_per_picture : Nat := 4
def pencils_per_exhibition : Nat := 2
def total_pencils : Nat := 88

-- Define the proof problem in Lean
theorem number_of_paintings_per_new_gallery (pictures_original new_galleries pencils_per_picture pencils_per_exhibition total_pencils : Nat) :
(pictures_original = 9) → (new_galleries = 5) → (pencils_per_picture = 4) → (pencils_per_exhibition = 2) → (total_pencils = 88) → 
∃ (pictures_per_gallery : Nat), pictures_per_gallery = 2 :=
by
  intros
  sorry

end number_of_paintings_per_new_gallery_l113_113313


namespace smallest_positive_multiple_of_45_is_45_l113_113236

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113236


namespace compare_negatives_l113_113911

theorem compare_negatives : - (1 : ℝ) / 2 < - (1 : ℝ) / 3 :=
by
  -- We start with the fact that 1/2 > 1/3
  have h : (1 : ℝ) / 2 > (1 : ℝ) / 3 := by
    linarith
    
  -- Negating the inequality reverses the sign
  exact neg_lt_neg_iff.mpr h

end compare_negatives_l113_113911


namespace smallest_positive_multiple_of_45_l113_113272

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l113_113272


namespace four_consecutive_integers_divisible_by_24_l113_113080

noncomputable def product_of_consecutive_integers (n : ℤ) : ℤ :=
  n * (n + 1) * (n + 2) * (n + 3)

theorem four_consecutive_integers_divisible_by_24 (n : ℤ) :
  24 ∣ product_of_consecutive_integers n :=
by
  sorry

end four_consecutive_integers_divisible_by_24_l113_113080


namespace find_2n_plus_m_l113_113673

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 2 * n + m = 36 :=
sorry

end find_2n_plus_m_l113_113673


namespace sum_of_x_coordinates_on_parabola_l113_113760

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1

-- Define the points P and Q on the parabola
variables {x1 x2 : ℝ}

-- The Lean theorem statement: 
theorem sum_of_x_coordinates_on_parabola 
  (h1 : parabola x1 = 1) 
  (h2 : parabola x2 = 1) : 
  x1 + x2 = 2 :=
sorry

end sum_of_x_coordinates_on_parabola_l113_113760


namespace cost_of_six_dozen_l113_113529

variable (cost_of_four_dozen : ℕ)
variable (dozens_to_purchase : ℕ)

theorem cost_of_six_dozen :
  cost_of_four_dozen = 24 →
  dozens_to_purchase = 6 →
  (dozens_to_purchase * (cost_of_four_dozen / 4)) = 36 :=
by
  intros h1 h2
  sorry

end cost_of_six_dozen_l113_113529


namespace fill_bathtub_time_l113_113808

def rate_cold_water : ℚ := 3 / 20
def rate_hot_water : ℚ := 1 / 8
def rate_drain : ℚ := 3 / 40
def net_rate : ℚ := rate_cold_water + rate_hot_water - rate_drain

theorem fill_bathtub_time :
  net_rate = 1/5 → (1 / net_rate) = 5 := by
  sorry

end fill_bathtub_time_l113_113808


namespace product_of_possible_b_values_l113_113857

theorem product_of_possible_b_values : 
  ∀ b : ℝ, 
    (abs (b - 2) = 2 * (4 - 1)) → 
    (b = 8 ∨ b = -4) → 
    (8 * (-4) = -32) := by
  sorry

end product_of_possible_b_values_l113_113857


namespace total_distance_covered_l113_113085

-- Definitions based on conditions
def distance_ran : ℝ := 40
def distance_walked : ℝ := (3 / 5) * distance_ran
def distance_jogged : ℝ := distance_walked / 5

-- Theorem stating the total distance covered
theorem total_distance_covered : distance_ran + distance_walked + distance_jogged = 64.8 := by
  -- You can place the formal proof steps here
  sorry

end total_distance_covered_l113_113085


namespace gcd_pow_sub_l113_113874

theorem gcd_pow_sub (h1001 h1012 : ℕ) (h : 1001 ≤ 1012) : 
  (Nat.gcd (2 ^ 1001 - 1) (2 ^ 1012 - 1)) = 2047 := sorry

end gcd_pow_sub_l113_113874


namespace number_of_cutlery_pieces_added_l113_113701

-- Define the initial conditions
def forks_initial := 6
def knives_initial := forks_initial + 9
def spoons_initial := 2 * knives_initial
def teaspoons_initial := forks_initial / 2
def total_initial_cutlery := forks_initial + knives_initial + spoons_initial + teaspoons_initial
def total_final_cutlery := 62

-- Define the total number of cutlery pieces added
def cutlery_added := total_final_cutlery - total_initial_cutlery

-- Define the theorem to prove
theorem number_of_cutlery_pieces_added : cutlery_added = 8 := by
  sorry

end number_of_cutlery_pieces_added_l113_113701


namespace negative_half_less_than_negative_third_l113_113906

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end negative_half_less_than_negative_third_l113_113906


namespace sum_reciprocal_transformation_l113_113503

theorem sum_reciprocal_transformation 
  (a b c d S : ℝ) 
  (h1 : a + b + c + d = S)
  (h2 : 1 / a + 1 / b + 1 / c + 1 / d = S)
  (h3 : a ≠ 0 ∧ a ≠ 1)
  (h4 : b ≠ 0 ∧ b ≠ 1)
  (h5 : c ≠ 0 ∧ c ≠ 1)
  (h6 : d ≠ 0 ∧ d ≠ 1) :
  S = -2 :=
by
  sorry

end sum_reciprocal_transformation_l113_113503


namespace cost_price_of_one_ball_l113_113091

theorem cost_price_of_one_ball (x : ℝ) (h : 11 * x - 720 = 5 * x) : x = 120 :=
sorry

end cost_price_of_one_ball_l113_113091


namespace expected_value_two_consecutive_red_balls_l113_113773

noncomputable def expected_draws_until_consecutive_red : ℝ :=
  let E : ℝ := sorry
  in E

theorem expected_value_two_consecutive_red_balls : expected_draws_until_consecutive_red = 4 := sorry

end expected_value_two_consecutive_red_balls_l113_113773


namespace smallest_b_for_perfect_square_l113_113223

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ (∃ (n : ℤ), 4 * b + 5 = n ^ 2) ∧ b = 5 := 
sorry

end smallest_b_for_perfect_square_l113_113223


namespace symmetric_point_line_eq_l113_113049

theorem symmetric_point_line_eq (A B : ℝ × ℝ) (l : ℝ → ℝ) (x1 y1 x2 y2 : ℝ)
  (hA : A = (4, 5))
  (hB : B = (-2, 7))
  (hSymmetric : ∀ x y, B = (2 * l x - A.1, 2 * l y - A.2)) :
  ∀ x y, l x = 3 * x - 5 ∧ l y = 3 * y + 6 :=
by
  sorry

end symmetric_point_line_eq_l113_113049


namespace probability_max_3_l113_113886

-- Define the set of cards and the drawn cards configuration
def cards := ({1, 2, 3} : Finset ℕ).product ({1, 2} : Finset ℕ)

-- Define a draw to be a subset of two cards
def draw (c : Finset (ℕ × ℕ)) : Finset (Finset (ℕ × ℕ)) :=
  c.powerset.filter (λ s, s.card = 2)

-- Event "the maximum number on the two drawn cards is 3"
def event_max_is_3 (s : Finset (ℕ × ℕ)) : Prop :=
  ∃ a b, (a, b) ∈ s ∧ ((a = 3) ∨ (b = 3))

-- Calculate the number of favorable outcomes
def favorable_outcomes : Finset (Finset (ℕ × ℕ)) :=
  (draw cards).filter (λ s, event_max_is_3 s)

-- Calculate the total number of outcomes
def total_outcomes : Finset (Finset (ℕ × ℕ)) :=
  draw cards

-- Calculate the probability
noncomputable def probability : ℚ :=
  favorable_outcomes.card / total_outcomes.card

-- Main theorem to prove
theorem probability_max_3 : probability = 3 / 5 :=
  sorry

end probability_max_3_l113_113886


namespace wilson_fraction_l113_113089

theorem wilson_fraction (N : ℝ) (result : ℝ) (F : ℝ) (h1 : N = 8) (h2 : result = 16 / 3) (h3 : N - F * N = result) : F = 1 / 3 := 
by
  sorry

end wilson_fraction_l113_113089


namespace simplify_fraction_l113_113321

variable (x : ℕ)

theorem simplify_fraction (h : x = 3) : (x^10 + 15 * x^5 + 125) / (x^5 + 5) = 248 + 25 / 62 := by
  sorry

end simplify_fraction_l113_113321


namespace student_age_is_24_l113_113889

/-- A man is 26 years older than his student. In two years, his age will be twice the age of his student.
    Prove that the present age of the student is 24 years old. -/
theorem student_age_is_24 (S M : ℕ) (h1 : M = S + 26) (h2 : M + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end student_age_is_24_l113_113889


namespace total_toys_correct_l113_113993

-- Define the conditions
def JaxonToys : ℕ := 15
def GabrielToys : ℕ := 2 * JaxonToys
def JerryToys : ℕ := GabrielToys + 8

-- Define the total number of toys
def TotalToys : ℕ := JaxonToys + GabrielToys + JerryToys

-- State the theorem
theorem total_toys_correct : TotalToys = 83 :=
by
  -- Skipping the proof as per instruction
  sorry

end total_toys_correct_l113_113993


namespace lcm_9_14_l113_113335

/-- Given the definition of the least common multiple (LCM) and the prime factorizations,
    prove that the LCM of 9 and 14 is 126. -/
theorem lcm_9_14 : Int.lcm 9 14 = 126 := by
  sorry

end lcm_9_14_l113_113335


namespace four_possible_x_values_l113_113928

noncomputable def sequence_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a 1
  else if n = 2 then a 2
  else (a (n - 1) + 1) / (a (n - 2))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + 1) / (a (n - 2))

noncomputable def possible_values (x : ℝ) : Prop :=
  ∃ a : ℕ → ℝ, a 1 = x ∧ a 2 = 3000 ∧ satisfies_condition a ∧ 
                ∃ n, a n = 3001

theorem four_possible_x_values : 
  { x : ℝ | possible_values x }.finite ∧ 
  (card { x : ℝ | possible_values x } = 4) :=
  sorry

end four_possible_x_values_l113_113928


namespace seashells_total_l113_113531

def seashells :=
  let sam_seashells := 18
  let mary_seashells := 47
  sam_seashells + mary_seashells

theorem seashells_total : seashells = 65 := by
  sorry

end seashells_total_l113_113531


namespace repeating_decimal_sum_l113_113598

theorem repeating_decimal_sum :
  (0.\overline{2} : ℝ) + (0.\overline{02} : ℝ) + (0.\overline{0002} : ℝ) = 2426/9999 :=
begin
  sorry
end

end repeating_decimal_sum_l113_113598


namespace LovelyCakeSlices_l113_113187

/-- Lovely cuts her birthday cake into some equal pieces.
    One-fourth of the cake was eaten by her visitors.
    Nine slices of cake were kept, representing three-fourths of the total number of slices.
    Prove: Lovely cut her birthday cake into 12 equal pieces. -/
theorem LovelyCakeSlices (totalSlices : ℕ) 
  (h1 : (3 / 4 : ℚ) * totalSlices = 9) : totalSlices = 12 := by
  sorry

end LovelyCakeSlices_l113_113187


namespace smallest_positive_multiple_of_45_l113_113231

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l113_113231


namespace jellybeans_final_count_l113_113057

-- Defining the initial number of jellybeans and operations
def initial_jellybeans : ℕ := 37
def removed_first : ℕ := 15
def added_back : ℕ := 5
def removed_second : ℕ := 4

-- Defining the final number of jellybeans to prove it equals 23
def final_jellybeans : ℕ := (initial_jellybeans - removed_first) + added_back - removed_second

-- The theorem that states the final number of jellybeans is 23
theorem jellybeans_final_count : final_jellybeans = 23 :=
by
  -- The proof will be provided here if needed
  sorry

end jellybeans_final_count_l113_113057


namespace max_concentration_at_2_l113_113060

noncomputable def concentration (t : ℝ) : ℝ := (20 * t) / (t^2 + 4)

theorem max_concentration_at_2 : ∃ t : ℝ, 0 ≤ t ∧ ∀ s : ℝ, (0 ≤ s → concentration s ≤ concentration t) ∧ t = 2 := 
by 
  sorry -- we add sorry to skip the actual proof

end max_concentration_at_2_l113_113060


namespace no_solution_for_inequality_l113_113469

theorem no_solution_for_inequality (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end no_solution_for_inequality_l113_113469


namespace max_additional_hours_l113_113023

/-- Define the additional hours of studying given the investments in dorms, food, and parties -/
def additional_hours (a b c : ℝ) : ℝ :=
  5 * a + 3 * b + (11 * c - c^2)

/-- Define the total investment constraint -/
def investment_constraint (a b c : ℝ) : Prop :=
  a + b + c = 5

/-- Prove the maximal additional hours of studying -/
theorem max_additional_hours : ∃ (a b c : ℝ), investment_constraint a b c ∧ additional_hours a b c = 34 :=
by
  sorry

end max_additional_hours_l113_113023


namespace find_b_l113_113221

-- Define what it means for b to be a solution
def is_solution (b : ℤ) : Prop :=
  b > 4 ∧ ∃ k : ℤ, 4 * b + 5 = k * k

-- State the problem
theorem find_b : ∃ b : ℤ, is_solution b ∧ ∀ b' : ℤ, is_solution b' → b' ≥ 5 := by
  sorry

end find_b_l113_113221


namespace conjugate_in_fourth_quadrant_l113_113045

def complex_conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

-- Given complex number
def z : ℂ := ⟨5, 3⟩

-- Conjugate of z
def z_conjugate : ℂ := complex_conjugate z

-- Cartesian coordinates of the conjugate
def z_conjugate_coordinates : ℝ × ℝ := (z_conjugate.re, z_conjugate.im)

-- Definition of the Fourth Quadrant
def is_in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem conjugate_in_fourth_quadrant :
  is_in_fourth_quadrant z_conjugate_coordinates :=
by sorry

end conjugate_in_fourth_quadrant_l113_113045


namespace award_medals_at_most_one_canadian_l113_113700

/-- Definition of conditions -/
def sprinter_count := 10 -- Total number of sprinters
def canadian_sprinter_count := 4 -- Number of Canadian sprinters
def medals := ["Gold", "Silver", "Bronze"] -- Types of medals

/-- Definition stating the requirement of the problem -/
def atMostOneCanadianMedal (total_sprinters : Nat) (canadian_sprinters : Nat) 
    (medal_types : List String) : Bool := 
  if total_sprinters = sprinter_count ∧ canadian_sprinters = canadian_sprinter_count ∧ medal_types = medals then
    true
  else
    false

/-- Statement to prove the number of ways to award the medals -/
theorem award_medals_at_most_one_canadian :
  (atMostOneCanadianMedal sprinter_count canadian_sprinter_count medals) →
  ∃ (ways : Nat), ways = 480 :=
by
  sorry

end award_medals_at_most_one_canadian_l113_113700


namespace rightmost_three_digits_of_3_pow_1987_l113_113547

theorem rightmost_three_digits_of_3_pow_1987 :
  3^1987 % 2000 = 187 :=
by sorry

end rightmost_three_digits_of_3_pow_1987_l113_113547


namespace find_k_l113_113962

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (k : ℝ) : ℝ × ℝ := (2 * k, 3)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
noncomputable def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

theorem find_k : ∃ k : ℝ, dot_product a (vector_add (scalar_mult 2 a) (b k)) = 0 ∧ k = -8 :=
by
  sorry

end find_k_l113_113962


namespace rational_solution_l113_113609

theorem rational_solution (m n : ℤ) (h : a = (m^4 + n^4 + m^2 * n^2) / (4 * m^2 * n^2)) : 
  ∃ a : ℚ, a = (m^4 + n^4 + m^2 * n^2) / (4 * m^2 * n^2) :=
by {
  sorry
}

end rational_solution_l113_113609


namespace sum_eq_sum_l113_113862

theorem sum_eq_sum {a b c d : ℝ} (h1 : a + b = c + d) (h2 : ac = bd) (h3 : a + b ≠ 0) : a + c = b + d := 
by
  sorry

end sum_eq_sum_l113_113862


namespace num_divisors_36_l113_113165

theorem num_divisors_36 : ∃ n, n = 9 ∧ ∀ d : ℕ, d ∣ 36 → (d > 0 ∧ d ≤ 36) → ∃ k : ℕ, d = (2^k or 3^k or (2^p * 3^q) where p, q <= 2):
begin
  have factorization : 36 = 2^2 * 3^2 := by norm_num,
  have exponents : (2+1)*(2+1) = 9 := by norm_num,
  use 9,
  split,
  { exact exponents },
  {
    intros d hd hpos_range,
    cases hpos_range with hpos hrange,
    sorry -- Proof showing that there are exactly 9 positive divisors.
  },
end

end num_divisors_36_l113_113165


namespace batsman_average_after_17th_l113_113716

def runs_17th_inning : ℕ := 87
def increase_in_avg : ℕ := 4
def num_innings : ℕ := 17

theorem batsman_average_after_17th (A : ℕ) (H : A + increase_in_avg = (16 * A + runs_17th_inning) / num_innings) : 
  (A + increase_in_avg) = 23 := sorry

end batsman_average_after_17th_l113_113716


namespace hex_B1C_base10_l113_113931

theorem hex_B1C_base10 : (11 * 16^2 + 1 * 16^1 + 12 * 16^0) = 2844 :=
by
  sorry

end hex_B1C_base10_l113_113931


namespace not_snow_probability_l113_113683

theorem not_snow_probability :
  let p_snow : ℚ := 2 / 3,
      p_not_snow : ℚ := 1 - p_snow in
  (p_not_snow ^ 5) = 1 / 243 := by
  sorry

end not_snow_probability_l113_113683


namespace converse_negation_contrapositive_l113_113292

variable {x : ℝ}

def P (x : ℝ) : Prop := x^2 - 3 * x + 2 ≠ 0
def Q (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ 2

theorem converse (h : Q x) : P x := by
  sorry

theorem negation (h : ¬ P x) : ¬ Q x := by
  sorry

theorem contrapositive (h : ¬ Q x) : ¬ P x := by
  sorry

end converse_negation_contrapositive_l113_113292


namespace notebook_pen_cost_l113_113581

theorem notebook_pen_cost :
  ∃ (n p : ℕ), 15 * n + 4 * p = 160 ∧ n > p ∧ n + p = 18 := 
sorry

end notebook_pen_cost_l113_113581


namespace divisors_of_36_l113_113162

theorem divisors_of_36 : ∀ n : ℕ, n = 36 → (∃ k : ℕ, k = 9) :=
by
  intro n hn
  have h_prime_factors : (n = 2^2 * 3^2) := by rw hn; norm_num
  -- Using the formula for the number of divisors based on prime factorization
  have h_num_divisors : (2 + 1) * (2 + 1) = 9 := by norm_num
  use 9
  rw h_num_divisors
  sorry

end divisors_of_36_l113_113162


namespace smallest_positive_multiple_of_45_l113_113263

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l113_113263


namespace kendra_fish_count_l113_113375

variable (K : ℕ) -- Number of fish Kendra caught
variable (Ken_fish : ℕ) -- Number of fish Ken brought home

-- Conditions
axiom twice_as_many : Ken_fish = 2 * K - 3
axiom total_fish : K + Ken_fish = 87

-- The theorem we need to prove
theorem kendra_fish_count : K = 30 :=
by
  -- Lean proof goes here
  sorry

end kendra_fish_count_l113_113375


namespace calculate_speed_of_stream_l113_113441

noncomputable def speed_of_stream (boat_speed : ℕ) (downstream_distance : ℕ) (upstream_distance : ℕ) : ℕ :=
  let x := (downstream_distance * boat_speed - boat_speed * upstream_distance) / (downstream_distance + upstream_distance)
  x

theorem calculate_speed_of_stream :
  speed_of_stream 20 26 14 = 6 := by
  sorry

end calculate_speed_of_stream_l113_113441


namespace worked_days_proof_l113_113721

theorem worked_days_proof (W N : ℕ) (hN : N = 24) (h0 : 100 * W = 25 * N) : W + N = 30 :=
by
  sorry

end worked_days_proof_l113_113721


namespace peter_situps_eq_24_l113_113341

noncomputable def situps_peter_did : ℕ :=
  let ratio_peter_greg := 3 / 4
  let situps_greg := 32
  let situps_peter := (3 * situps_greg) / 4
  situps_peter

theorem peter_situps_eq_24 : situps_peter_did = 24 := 
by 
  let h := situps_peter_did
  show h = 24
  sorry

end peter_situps_eq_24_l113_113341


namespace combined_average_age_l113_113411

-- Definitions based on given conditions
def num_fifth_graders : ℕ := 28
def avg_age_fifth_graders : ℝ := 10
def num_parents : ℕ := 45
def avg_age_parents : ℝ := 40

-- The statement to prove
theorem combined_average_age : (num_fifth_graders * avg_age_fifth_graders + num_parents * avg_age_parents) / (num_fifth_graders + num_parents) = 28.49 :=
  by
  sorry

end combined_average_age_l113_113411


namespace exists_n_for_sin_l113_113397

theorem exists_n_for_sin (x : ℝ) (h : Real.sin x ≠ 0) :
  ∃ n : ℕ, |Real.sin (n * x)| ≥ Real.sqrt 3 / 2 :=
sorry

end exists_n_for_sin_l113_113397


namespace find_value_l113_113536

-- Define the mean, standard deviation, and the number of standard deviations
def mean : ℝ := 17.5
def std_dev : ℝ := 2.5
def num_std_dev : ℝ := 2.7

-- The theorem to prove that the value is exactly 10.75
theorem find_value : mean - (num_std_dev * std_dev) = 10.75 := 
by
  sorry

end find_value_l113_113536


namespace volume_of_prism_l113_113807

theorem volume_of_prism (a : ℝ) (h_pos : 0 < a) (h_lat : ∀ S_lat, S_lat = a ^ 2) : 
  ∃ V, V = (a ^ 3 * (Real.sqrt 2 - 1)) / 4 :=
by
  sorry

end volume_of_prism_l113_113807


namespace pets_remaining_l113_113113

-- Definitions based on conditions
def initial_puppies : ℕ := 7
def initial_kittens : ℕ := 6
def sold_puppies : ℕ := 2
def sold_kittens : ℕ := 3

-- Theorem statement
theorem pets_remaining : initial_puppies + initial_kittens - (sold_puppies + sold_kittens) = 8 :=
by
  sorry

end pets_remaining_l113_113113


namespace rahim_average_price_per_book_l113_113655

noncomputable section

open BigOperators

def store_A_price_per_book : ℝ := 
  let original_total := 1600
  let discount := original_total * 0.15
  let discounted_total := original_total - discount
  let sales_tax := discounted_total * 0.05
  let final_total := discounted_total + sales_tax
  final_total / 25

def store_B_price_per_book : ℝ := 
  let original_total := 3200
  let effective_books_paid := 35 - (35 / 4)
  original_total / effective_books_paid

def store_C_price_per_book : ℝ := 
  let original_total := 3800
  let discount := 0.10 * (4 * (original_total / 40))
  let discounted_total := original_total - discount
  let service_charge := discounted_total * 0.07
  let final_total := discounted_total + service_charge
  final_total / 40

def store_D_price_per_book : ℝ := 
  let original_total := 2400
  let discount := 0.50 * (original_total / 30)
  let discounted_total := original_total - discount
  let sales_tax := discounted_total * 0.06
  let final_total := discounted_total + sales_tax
  final_total / 30

def store_E_price_per_book : ℝ := 
  let original_total := 1800
  let discount := original_total * 0.08
  let discounted_total := original_total - discount
  let additional_fee := discounted_total * 0.04
  let final_total := discounted_total + additional_fee
  final_total / 20

def total_books : ℝ := 25 + 35 + 40 + 30 + 20

def total_amount : ℝ := 
  store_A_price_per_book * 25 + 
  store_B_price_per_book * 35 + 
  store_C_price_per_book * 40 + 
  store_D_price_per_book * 30 + 
  store_E_price_per_book * 20

def average_price_per_book : ℝ := total_amount / total_books

theorem rahim_average_price_per_book : average_price_per_book = 85.85 :=
sorry

end rahim_average_price_per_book_l113_113655


namespace optimal_perimeter_proof_l113_113215

-- Definition of conditions
def fencing_length : Nat := 400
def min_width : Nat := 50
def area : Nat := 8000

-- Definition of the perimeter to be proven as optimal
def optimal_perimeter : Nat := 360

-- Theorem statement to be proven
theorem optimal_perimeter_proof (l w : Nat) (h1 : l * w = area) (h2 : 2 * l + 2 * w <= fencing_length) (h3 : w >= min_width) :
  2 * l + 2 * w = optimal_perimeter :=
sorry

end optimal_perimeter_proof_l113_113215


namespace base7_digit_divisibility_l113_113343

-- Define base-7 digit integers
notation "digit" => Fin 7

-- Define conversion from base-7 to base-10 for the form 3dd6_7
def base7_to_base10 (d : digit) : ℤ := 3 * (7^3) + (d:ℤ) * (7^2) + (d:ℤ) * 7 + 6

-- Define the property of being divisible by 13
def is_divisible_by_13 (n : ℤ) : Prop := ∃ k : ℤ, n = 13 * k

-- Formalize the theorem
theorem base7_digit_divisibility (d : digit) :
  is_divisible_by_13 (base7_to_base10 d) ↔ d = 4 :=
sorry

end base7_digit_divisibility_l113_113343


namespace smallest_positive_multiple_of_45_l113_113262

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l113_113262


namespace no_snow_five_days_l113_113680

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l113_113680


namespace not_snowing_next_five_days_l113_113687

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end not_snowing_next_five_days_l113_113687


namespace num_ways_to_pay_l113_113750

theorem num_ways_to_pay (n : ℕ) : 
  ∃ a_n : ℕ, a_n = (n / 2) + 1 :=
sorry

end num_ways_to_pay_l113_113750


namespace zero_in_interval_l113_113699

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem zero_in_interval (h_mono : ∀ x y, 0 < x → x < y → f x < f y) (h_f2 : f 2 < 0) (h_f3 : 0 < f 3) :
  ∃ x₀ ∈ (Set.Ioo 2 3), f x₀ = 0 :=
by
  sorry

end zero_in_interval_l113_113699


namespace total_books_l113_113589

theorem total_books (b1 b2 b3 b4 b5 b6 b7 b8 b9 : ℕ) :
  b1 = 56 →
  b2 = b1 + 2 →
  b3 = b2 + 2 →
  b4 = b3 + 2 →
  b5 = b4 + 2 →
  b6 = b5 + 2 →
  b7 = b6 - 4 →
  b8 = b7 - 4 →
  b9 = b8 - 4 →
  b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 = 490 :=
by
  sorry

end total_books_l113_113589


namespace p_is_necessary_but_not_sufficient_for_q_l113_113613

variable (x : ℝ)

def p := x > 4
def q := 4 < x ∧ x < 10

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) := by
  sorry

end p_is_necessary_but_not_sufficient_for_q_l113_113613


namespace solve_eq1_solve_eq2_l113_113030

-- Define the first problem statement and the correct answers
theorem solve_eq1 (x : ℝ) (h : (x - 2) ^ 2 = 169) : x = 15 ∨ x = -11 := 
  by sorry

-- Define the second problem statement and the correct answer
theorem solve_eq2 (x : ℝ) (h : 3 * (x - 3) ^ 3 - 24 = 0) : x = 5 := 
  by sorry

end solve_eq1_solve_eq2_l113_113030


namespace lcm_5_7_10_14_l113_113876

theorem lcm_5_7_10_14 : Nat.lcm (Nat.lcm 5 7) (Nat.lcm 10 14) = 70 := by
  sorry

end lcm_5_7_10_14_l113_113876


namespace margarita_jumps_farther_l113_113195

-- Definitions for conditions
def RiccianaTotalDistance := 24
def RiccianaRunDistance := 20
def RiccianaJumpDistance := 4

def MargaritaRunDistance := 18
def MargaritaJumpDistance := 2 * RiccianaJumpDistance - 1

-- Theorem statement to prove the question
theorem margarita_jumps_farther :
  (MargaritaRunDistance + MargaritaJumpDistance) - RiccianaTotalDistance = 1 :=
by
  -- Proof will be written here
  sorry

end margarita_jumps_farther_l113_113195


namespace smallest_three_digit_number_l113_113100

theorem smallest_three_digit_number :
  ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧
  (x % 2 = 0) ∧
  ((x + 1) % 3 = 0) ∧
  ((x + 2) % 4 = 0) ∧
  ((x + 3) % 5 = 0) ∧
  ((x + 4) % 6 = 0) ∧
  x = 122 :=
by
  sorry

end smallest_three_digit_number_l113_113100


namespace julia_tuesday_kids_l113_113784

-- Definitions based on the given conditions in the problem.
def monday_kids : ℕ := 15
def monday_tuesday_kids : ℕ := 33

-- The problem statement to prove the number of kids played with on Tuesday.
theorem julia_tuesday_kids :
  (∃ tuesday_kids : ℕ, tuesday_kids = monday_tuesday_kids - monday_kids) →
  18 = monday_tuesday_kids - monday_kids :=
by
  intro h
  sorry

end julia_tuesday_kids_l113_113784


namespace inequality_proof_l113_113948

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  b + (1 / a) > a + (1 / b) := 
by sorry

end inequality_proof_l113_113948


namespace find_a_l113_113481

theorem find_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : x - a * y = 3) : a = -1 :=
by
  -- Solution steps go here
  sorry

end find_a_l113_113481


namespace find_f_3_l113_113841

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l113_113841


namespace team_leader_prize_l113_113409

theorem team_leader_prize 
    (number_of_students : ℕ := 10)
    (number_of_team_members : ℕ := 9)
    (team_member_prize : ℕ := 200)
    (additional_leader_prize : ℕ := 90)
    (total_prize : ℕ)
    (leader_prize : ℕ := total_prize - (number_of_team_members * team_member_prize))
    (average_prize : ℕ := (total_prize + additional_leader_prize) / number_of_students)
: leader_prize = 300 := 
by {
  sorry  -- Proof omitted
}

end team_leader_prize_l113_113409


namespace values_of_2n_plus_m_l113_113674

theorem values_of_2n_plus_m (n m : ℤ) (h1 : 3 * n - m ≤ 4) (h2 : n + m ≥ 27) (h3 : 3 * m - 2 * n ≤ 45) 
  (h4 : n = 8) (h5 : m = 20) : 2 * n + m = 36 := by
  sorry

end values_of_2n_plus_m_l113_113674


namespace smallest_positive_multiple_of_45_l113_113284

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ, n > 0 ∧ 45 * n = 45 :=
by
  use 1
  split
  · exact Nat.succ_pos 0
  · exact Nat.mul_one 45

end smallest_positive_multiple_of_45_l113_113284


namespace correct_reflection_l113_113712

section
variable {Point : Type}
variables (PQ : Point → Point → Prop) (shaded_figure : Point → Prop)
variables (A B C D E : Point → Prop)

-- Condition: The line segment PQ is the axis of reflection.
-- Condition: The shaded figure is positioned above the line PQ and touches it at two points.
-- Define the reflection operation (assuming definitions for points and reflections are given).

def reflected (fig : Point → Prop) (axis : Point → Point → Prop) : Point → Prop := sorry  -- Define properly

-- The correct answer: The reflected figure should match figure (A).
theorem correct_reflection :
  reflected shaded_figure PQ = A :=
sorry
end

end correct_reflection_l113_113712


namespace travel_time_in_minutes_l113_113444

def bird_speed : ℝ := 8 -- Speed of the bird in miles per hour
def distance_to_travel : ℝ := 3 -- Distance to be traveled in miles

theorem travel_time_in_minutes : (distance_to_travel / bird_speed) * 60 = 22.5 :=
by
  sorry

end travel_time_in_minutes_l113_113444


namespace arithmetic_sequence_ninth_term_l113_113205

theorem arithmetic_sequence_ninth_term (a d : ℤ) 
    (h5 : a + 4 * d = 23) (h7 : a + 6 * d = 37) : 
    a + 8 * d = 51 := 
by 
  sorry

end arithmetic_sequence_ninth_term_l113_113205


namespace sum_of_104th_parenthesis_is_correct_l113_113022

def b (n : ℕ) : ℕ := 2 * n + 1

def sumOf104thParenthesis : ℕ :=
  let cycleCount := 104 / 4
  let numbersBefore104 := 260
  let firstNumIndex := numbersBefore104 + 1
  let firstNum := b firstNumIndex
  let secondNum := b (firstNumIndex + 1)
  let thirdNum := b (firstNumIndex + 2)
  let fourthNum := b (firstNumIndex + 3)
  firstNum + secondNum + thirdNum + fourthNum

theorem sum_of_104th_parenthesis_is_correct : sumOf104thParenthesis = 2104 :=
  by
    sorry

end sum_of_104th_parenthesis_is_correct_l113_113022


namespace calculate_constants_l113_113971

noncomputable def parabola_tangent_to_line (a b : ℝ) : Prop :=
  let discriminant := (b - 2) ^ 2 + 28 * a
  discriminant = 0

theorem calculate_constants
  (a b : ℝ)
  (h_tangent : parabola_tangent_to_line a b) :
  a = -((b - 2) ^ 2) / 28 ∧ b ≠ 2 :=
by
  sorry

end calculate_constants_l113_113971


namespace candy_cost_proof_l113_113719

theorem candy_cost_proof (x : ℝ) (h1 : 10 ≤ 30) (h2 : 0 ≤ 5) (h3 : 0 ≤ 6) 
(h4 : 10 * x + 20 * 5 = 6 * 30) : x = 8 := by
  sorry

end candy_cost_proof_l113_113719


namespace smallest_positive_multiple_of_45_l113_113267

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l113_113267


namespace cos_C_given_sin_A_and_cos_B_l113_113980

theorem cos_C_given_sin_A_and_cos_B (A B C : ℝ) (h_triangle : A + B + C = real.pi)
  (h_sinA : real.sin A = 4 / 5) (h_cosB : real.cos B = 12 / 13) :
  real.cos C = -16 / 65 :=
sorry

end cos_C_given_sin_A_and_cos_B_l113_113980


namespace find_f_3_l113_113845

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l113_113845


namespace fill_bathtub_time_l113_113171

theorem fill_bathtub_time
  (r_cold : ℚ := 1/10)
  (r_hot : ℚ := 1/15)
  (r_empty : ℚ := -1/12)
  (net_rate : ℚ := r_cold + r_hot + r_empty) :
  net_rate = 1/12 → 
  t = 12 :=
by
  sorry

end fill_bathtub_time_l113_113171


namespace Samuel_fraction_spent_l113_113097

variable (totalAmount receivedRatio remainingAmount : ℕ)
variable (h1 : totalAmount = 240)
variable (h2 : receivedRatio = 3 / 4)
variable (h3 : remainingAmount = 132)

theorem Samuel_fraction_spent (spend : ℚ) : 
  (spend = (1 / 5)) :=
by
  sorry

end Samuel_fraction_spent_l113_113097


namespace smallest_positive_multiple_of_45_is_45_l113_113246

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113246


namespace problem1_problem2_l113_113118

-- Proof problem 1
theorem problem1 (x : ℝ) : (x - 1)^2 + x * (3 - x) = x + 1 := sorry

-- Proof problem 2
theorem problem2 (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -2) : (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1)) = 1 / (a + 2) := sorry

end problem1_problem2_l113_113118


namespace price_Ramesh_paid_l113_113526

-- Define the conditions
def labelled_price_sold (P : ℝ) := 1.10 * P
def discount_price_paid (P : ℝ) := 0.80 * P
def additional_costs := 125 + 250
def total_cost (P : ℝ) := discount_price_paid P + additional_costs

-- The main theorem stating that given the conditions,
-- the price Ramesh paid for the refrigerator is Rs. 13175.
theorem price_Ramesh_paid (P : ℝ) (H : labelled_price_sold P = 17600) :
  total_cost P = 13175 :=
by
  -- Providing a placeholder, as we do not need to provide the proof steps in the problem formulation
  sorry

end price_Ramesh_paid_l113_113526


namespace no_snow_five_days_l113_113686

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l113_113686


namespace inverse_h_l113_113407

-- Define the functions f, g, and h as given in the conditions
def f (x : ℝ) := 4 * x - 3
def g (x : ℝ) := 3 * x + 2
def h (x : ℝ) := f (g x)

-- State the problem of proving the inverse of h
theorem inverse_h : ∀ x, h⁻¹ (x : ℝ) = (x - 5) / 12 :=
sorry

end inverse_h_l113_113407


namespace twelve_div_one_fourth_eq_48_l113_113465

theorem twelve_div_one_fourth_eq_48 : 12 / (1 / 4) = 48 := by
  -- We know that dividing by a fraction is equivalent to multiplying by its reciprocal
  sorry

end twelve_div_one_fourth_eq_48_l113_113465


namespace smallest_positive_multiple_of_45_l113_113229

theorem smallest_positive_multiple_of_45 :
  ∃ x : ℕ, x > 0 ∧ 45 * x = 45 := 
by
  use 1
  simp
  sorry

end smallest_positive_multiple_of_45_l113_113229


namespace profit_percentage_A_is_20_l113_113570

-- Definitions of conditions
def cost_price_A := 156 -- Cost price of the cricket bat for A
def selling_price_C := 234 -- Selling price of the cricket bat to C
def profit_percent_B := 25 / 100 -- Profit percentage for B

-- Calculations
def cost_price_B := selling_price_C / (1 + profit_percent_B) -- Cost price of the cricket bat for B
def selling_price_A := cost_price_B -- Selling price of the cricket bat for A

-- Profit and profit percentage calculations
def profit_A := selling_price_A - cost_price_A -- Profit for A
def profit_percent_A := profit_A / cost_price_A * 100 -- Profit percentage for A

-- Statement to prove
theorem profit_percentage_A_is_20 : profit_percent_A = 20 :=
by
  sorry

end profit_percentage_A_is_20_l113_113570


namespace probability_second_roll_odd_given_first_roll_odd_l113_113201

theorem probability_second_roll_odd_given_first_roll_odd :
  let A := (fun (ω : Fin 6) => ω % 2 = 1)
  let B := (fun (ω : Fin 6) => ω % 2 = 1)
  let P := (fun (A : Set (Fin 6)) => (Set.size A).toReal / (Set.size (Fin 6)).toReal)
  P(B | A) = 1 / 2 :=
by
  sorry

end probability_second_roll_odd_given_first_roll_odd_l113_113201


namespace parallel_lines_slope_l113_113351

theorem parallel_lines_slope {m : ℝ} : 
  (∃ m, (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 → 6 * x + m * y + 14 = 0)) ↔ m = 8 :=
by
  sorry

end parallel_lines_slope_l113_113351


namespace find_functional_solution_l113_113135

theorem find_functional_solution (c : ℝ) (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) :
  ∀ x : ℝ, f x = x ^ 3 + c * x := by
  sorry

end find_functional_solution_l113_113135


namespace pentagon_perimeter_even_l113_113372

noncomputable def dist_sq (A B : ℤ × ℤ) : ℤ :=
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2

theorem pentagon_perimeter_even (A B C D E : ℤ × ℤ) (h1 : dist_sq A B % 2 = 1) (h2 : dist_sq B C % 2 = 1) 
  (h3 : dist_sq C D % 2 = 1) (h4 : dist_sq D E % 2 = 1) (h5 : dist_sq E A % 2 = 1) : 
  (dist_sq A B + dist_sq B C + dist_sq C D + dist_sq D E + dist_sq E A) % 2 = 0 := 
by 
  sorry

end pentagon_perimeter_even_l113_113372


namespace margarita_jumps_farther_l113_113196

-- Definitions for conditions
def RiccianaTotalDistance := 24
def RiccianaRunDistance := 20
def RiccianaJumpDistance := 4

def MargaritaRunDistance := 18
def MargaritaJumpDistance := 2 * RiccianaJumpDistance - 1

-- Theorem statement to prove the question
theorem margarita_jumps_farther :
  (MargaritaRunDistance + MargaritaJumpDistance) - RiccianaTotalDistance = 1 :=
by
  -- Proof will be written here
  sorry

end margarita_jumps_farther_l113_113196


namespace total_number_of_toys_l113_113996

def jaxon_toys := 15
def gabriel_toys := 2 * jaxon_toys
def jerry_toys := gabriel_toys + 8

theorem total_number_of_toys : jaxon_toys + gabriel_toys + jerry_toys = 83 := by
  sorry

end total_number_of_toys_l113_113996


namespace total_flowers_eaten_l113_113390

theorem total_flowers_eaten (bugs : ℕ) (flowers_per_bug : ℕ) (h_bugs : bugs = 3) (h_flowers_per_bug : flowers_per_bug = 2) :
  (bugs * flowers_per_bug) = 6 :=
by
  sorry

end total_flowers_eaten_l113_113390


namespace find_f_of_3_l113_113823

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l113_113823


namespace intersection_M_N_l113_113489

def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | (x ∈ U) ∧ ¬(x ∈ complement_U_N)}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
sorry

end intersection_M_N_l113_113489


namespace english_homework_correct_time_l113_113395

-- Define the given conditions as constants
def total_time : ℕ := 180 -- 3 hours in minutes
def math_homework_time : ℕ := 45
def science_homework_time : ℕ := 50
def history_homework_time : ℕ := 25
def special_project_time : ℕ := 30

-- Define the function to compute english homework time
def english_homework_time : ℕ :=
  total_time - (math_homework_time + science_homework_time + history_homework_time + special_project_time)

-- The theorem to show the English homework time is 30 minutes
theorem english_homework_correct_time :
  english_homework_time = 30 :=
  by
    sorry

end english_homework_correct_time_l113_113395


namespace bottles_left_l113_113106

-- Define initial conditions
def bottlesInRefrigerator : Nat := 4
def bottlesInPantry : Nat := 4
def bottlesBought : Nat := 5
def bottlesDrank : Nat := 3

-- Goal: Prove the total number of bottles left
theorem bottles_left : bottlesInRefrigerator + bottlesInPantry + bottlesBought - bottlesDrank = 10 :=
by
  sorry

end bottles_left_l113_113106


namespace jorge_spent_amount_l113_113005

theorem jorge_spent_amount
  (num_tickets : ℕ)
  (price_per_ticket : ℕ)
  (discount_percentage : ℚ)
  (h1 : num_tickets = 24)
  (h2 : price_per_ticket = 7)
  (h3 : discount_percentage = 0.5) :
  num_tickets * price_per_ticket * (1 - discount_percentage) = 84 := 
by
  simp [h1, h2, h3]
  sorry

end jorge_spent_amount_l113_113005


namespace sum_mod_7_l113_113607

/-- Define the six numbers involved. -/
def a := 102345
def b := 102346
def c := 102347
def d := 102348
def e := 102349
def f := 102350

/-- State the theorem to prove the remainder of their sum when divided by 7. -/
theorem sum_mod_7 : 
  (a + b + c + d + e + f) % 7 = 5 := 
by sorry

end sum_mod_7_l113_113607


namespace smallest_positive_multiple_of_45_l113_113260

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ+, 45 * x = 45 :=
by
  use 1
  exact (Nat.mul_one 45).symm

end smallest_positive_multiple_of_45_l113_113260


namespace find_a_l113_113966

open Complex

theorem find_a (a : ℝ) (h : (2 + Complex.I * a) / (1 + Complex.I * Real.sqrt 2) = -Complex.I * Real.sqrt 2) :
  a = Real.sqrt 2 := by
  sorry

end find_a_l113_113966


namespace lisa_total_miles_flown_l113_113793

-- Definitions based on given conditions
def distance_per_trip : ℝ := 256.0
def number_of_trips : ℝ := 32.0
def total_miles_flown : ℝ := 8192.0

-- Lean statement asserting the equivalence
theorem lisa_total_miles_flown : 
    (distance_per_trip * number_of_trips = total_miles_flown) :=
by 
    sorry

end lisa_total_miles_flown_l113_113793


namespace problem_inequality_l113_113617

theorem problem_inequality (k m n : ℕ) (hk1 : 1 < k) (hkm : k ≤ m) (hmn : m < n) :
  (1 + m) ^ 2 > (1 + n) ^ m :=
  sorry

end problem_inequality_l113_113617


namespace sum_of_integers_l113_113667

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
by
  sorry

end sum_of_integers_l113_113667


namespace factorial_division_l113_113123

theorem factorial_division (n m : ℕ) (h : n = 52) (g : m = 50) : (n! / m!) = 2652 :=
by
  sorry

end factorial_division_l113_113123


namespace mixed_numbers_sum_l113_113867

-- Declare the mixed numbers as fraction equivalents
def mixed1 : ℚ := 2 + 1/10
def mixed2 : ℚ := 3 + 11/100
def mixed3 : ℚ := 4 + 111/1000

-- Assert that the sum of mixed1, mixed2, and mixed3 is equal to 9.321
theorem mixed_numbers_sum : mixed1 + mixed2 + mixed3 = 9321 / 1000 := by
  sorry

end mixed_numbers_sum_l113_113867


namespace sum_of_integers_l113_113668

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
by
  sorry

end sum_of_integers_l113_113668


namespace arithmetic_identity_l113_113913

theorem arithmetic_identity : 45 * 27 + 73 * 45 = 4500 := by sorry

end arithmetic_identity_l113_113913


namespace smallest_positive_multiple_of_45_is_45_l113_113238

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113238


namespace sum_of_integers_l113_113811

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 288) : x + y = 35 :=
sorry

end sum_of_integers_l113_113811


namespace total_cans_given_away_l113_113902

noncomputable def total_cans_initial : ℕ := 2000
noncomputable def cans_taken_first_day : ℕ := 500
noncomputable def restocked_first_day : ℕ := 1500
noncomputable def people_second_day : ℕ := 1000
noncomputable def cans_per_person_second_day : ℕ := 2
noncomputable def restocked_second_day : ℕ := 3000

theorem total_cans_given_away :
  (cans_taken_first_day + (people_second_day * cans_per_person_second_day) = 2500) :=
by
  sorry

end total_cans_given_away_l113_113902


namespace number_of_divisors_of_36_l113_113159

theorem number_of_divisors_of_36 : Nat.totient 36 = 9 := 
by sorry

end number_of_divisors_of_36_l113_113159


namespace color_of_last_bead_l113_113561

-- Define the sequence and length of repeated pattern
def bead_pattern : List String := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "green", "blue"]
def pattern_length : Nat := bead_pattern.length

-- Define the total number of beads in the bracelet
def total_beads : Nat := 85

-- State the theorem to prove the color of the last bead
theorem color_of_last_bead : bead_pattern.get? ((total_beads - 1) % pattern_length) = some "yellow" :=
by
  sorry

end color_of_last_bead_l113_113561


namespace gp_sum_l113_113325

theorem gp_sum (x : ℕ) (h : (30 + x) / (10 + x) = (60 + x) / (30 + x)) :
  x = 30 ∧ (10 + x) + (30 + x) + (60 + x) + (120 + x) = 340 :=
by {
  sorry
}

end gp_sum_l113_113325


namespace pencils_purchased_l113_113098

theorem pencils_purchased 
  (total_cost : ℝ)
  (num_pens : ℕ)
  (price_per_pen : ℝ)
  (price_per_pencil : ℝ)
  (total_cost_condition : total_cost = 510)
  (num_pens_condition : num_pens = 30)
  (price_per_pen_condition : price_per_pen = 12)
  (price_per_pencil_condition : price_per_pencil = 2) :
  num_pens * price_per_pen + sorry = total_cost →
  150 / price_per_pencil = 75 :=
by
  sorry

end pencils_purchased_l113_113098


namespace find_f_neg2_l113_113142

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ a b : ℝ, f (a + b) = f a * f b
axiom cond2 : ∀ x : ℝ, f x > 0
axiom cond3 : f 1 = 1 / 3

theorem find_f_neg2 : f (-2) = 9 := sorry

end find_f_neg2_l113_113142


namespace smaller_solution_of_quadratic_l113_113336

theorem smaller_solution_of_quadratic :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 13 * x + 36 = 0) ∧ (y^2 - 13 * y + 36 = 0) ∧ min x y = 4) :=
sorry

end smaller_solution_of_quadratic_l113_113336


namespace max_f_of_polynomial_l113_113516

theorem max_f_of_polynomial (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (h_poly : ∃ p : Polynomial ℝ, ∀ x, f x = Polynomial.eval x p)
    (h1 : f 4 = 16)
    (h2 : f 16 = 512) :
    f 8 ≤ 64 :=
by
  sorry

end max_f_of_polynomial_l113_113516


namespace river_flow_rate_l113_113451

theorem river_flow_rate
  (h : ℝ) (h_eq : h = 3)
  (w : ℝ) (w_eq : w = 36)
  (V : ℝ) (V_eq : V = 3600)
  (conversion_factor : ℝ) (conversion_factor_eq : conversion_factor = 3.6) :
  (60 / (w * h)) * conversion_factor = 2 := by
  sorry

end river_flow_rate_l113_113451


namespace distance_AC_in_terms_of_M_l113_113032

-- Define the given constants and the relevant equations
variables (M x : ℝ) (AB BC AC : ℝ)
axiom distance_eq_add : AB = M + BC
axiom time_AB : (M + x) / 7 = x / 5
axiom time_BC : BC = x
axiom time_S : (M + x + x) = AC

theorem distance_AC_in_terms_of_M : AC = 6 * M :=
by
  sorry

end distance_AC_in_terms_of_M_l113_113032


namespace Jenny_reading_days_l113_113505

theorem Jenny_reading_days :
  let words_per_hour := 100
  let book1_words := 200
  let book2_words := 400
  let book3_words := 300
  let total_words := book1_words + book2_words + book3_words
  let total_hours := total_words / words_per_hour
  let minutes_per_day := 54
  let hours_per_day := minutes_per_day / 60
  total_hours / hours_per_day = 10 :=
by
  sorry

end Jenny_reading_days_l113_113505


namespace find_k_l113_113146

open Real

variables (a b : ℝ × ℝ) (k : ℝ)

def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

theorem find_k (ha : a = (1, 2)) (hb : b = (-2, 4)) (perpendicular : dot_product (k • a + b) b = 0) :
  k = - (10 / 3) :=
by
  sorry

end find_k_l113_113146


namespace evaluate_expression_l113_113631

theorem evaluate_expression (x y : ℤ) (h₁ : x = 3) (h₂ : y = 4) : 
  (x^4 + 3 * x^2 - 2 * y + 2 * y^2) / 6 = 22 :=
by
  -- Conditions from the problem
  rw [h₁, h₂]
  -- Sorry is used to skip the proof
  sorry

end evaluate_expression_l113_113631


namespace smallest_sum_squares_edges_is_cube_l113_113111

theorem smallest_sum_squares_edges_is_cube (V : ℝ) (a b c : ℝ)
  (h_vol : a * b * c = V) :
  a^2 + b^2 + c^2 ≥ 3 * (V^(2/3)) := 
sorry

end smallest_sum_squares_edges_is_cube_l113_113111


namespace total_cost_after_discount_l113_113092

noncomputable def mango_cost : ℝ := sorry
noncomputable def rice_cost : ℝ := sorry
noncomputable def flour_cost : ℝ := 21

theorem total_cost_after_discount :
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  (flour_cost = 21) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost) * 0.9 = 808.92 :=
by
  intros h1 h2 h3
  -- sorry as placeholder for actual proof
  sorry

end total_cost_after_discount_l113_113092


namespace distance_AF_l113_113641

theorem distance_AF (A B C D E F : ℝ×ℝ)
  (h1 : A = (0, 0))
  (h2 : B = (5, 0))
  (h3 : C = (5, 5))
  (h4 : D = (0, 5))
  (h5 : E = (2.5, 5))
  (h6 : ∃ k : ℝ, F = (k, 2 * k) ∧ dist F C = 5) :
  dist A F = Real.sqrt 5 :=
by
  sorry

end distance_AF_l113_113641


namespace fraction_of_satisfactory_is_15_over_23_l113_113664

def num_students_with_grade_A : ℕ := 6
def num_students_with_grade_B : ℕ := 5
def num_students_with_grade_C : ℕ := 4
def num_students_with_grade_D : ℕ := 2
def num_students_with_grade_F : ℕ := 6

def num_satisfactory_students : ℕ := 
  num_students_with_grade_A + num_students_with_grade_B + num_students_with_grade_C

def total_students : ℕ := 
  num_satisfactory_students + num_students_with_grade_D + num_students_with_grade_F

def fraction_satisfactory : ℚ := 
  (num_satisfactory_students : ℚ) / (total_students : ℚ)

theorem fraction_of_satisfactory_is_15_over_23 : 
  fraction_satisfactory = 15/23 :=
by
  -- proof omitted
  sorry

end fraction_of_satisfactory_is_15_over_23_l113_113664


namespace fill_bathtub_time_l113_113173

theorem fill_bathtub_time (V : ℝ) (cold_rate hot_rate drain_rate net_rate : ℝ) 
  (hcold : cold_rate = V / 10) 
  (hhot : hot_rate = V / 15) 
  (hdrain : drain_rate = -V / 12) 
  (hnet : net_rate = cold_rate + hot_rate + drain_rate) 
  (V_eq : V = 1) : 
  1 / net_rate = 12 :=
by {
  -- placeholder for proof steps
  sorry
}

end fill_bathtub_time_l113_113173


namespace parallelogram_area_is_correct_l113_113960

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨0, 2, 3⟩
def B : Point3D := ⟨2, 5, 2⟩
def C : Point3D := ⟨-2, 3, 6⟩

noncomputable def vectorAB (A B : Point3D) : Point3D :=
  { x := B.x - A.x
  , y := B.y - A.y
  , z := B.z - A.z 
  }

noncomputable def vectorAC (A C : Point3D) : Point3D :=
  { x := C.x - A.x
  , y := C.y - A.y
  , z := C.z - A.z 
  }

noncomputable def dotProduct (u v : Point3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

noncomputable def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

noncomputable def sinAngle (u v : Point3D) : ℝ :=
  Real.sqrt (1 - (dotProduct u v / (magnitude u * magnitude v)) ^ 2)

noncomputable def parallelogramArea (u v : Point3D) : ℝ :=
  magnitude u * magnitude v * sinAngle u v

theorem parallelogram_area_is_correct :
  parallelogramArea (vectorAB A B) (vectorAC A C) = 6 * Real.sqrt 5 := by
  sorry

end parallelogram_area_is_correct_l113_113960


namespace values_of_2n_plus_m_l113_113675

theorem values_of_2n_plus_m (n m : ℤ) (h1 : 3 * n - m ≤ 4) (h2 : n + m ≥ 27) (h3 : 3 * m - 2 * n ≤ 45) 
  (h4 : n = 8) (h5 : m = 20) : 2 * n + m = 36 := by
  sorry

end values_of_2n_plus_m_l113_113675


namespace right_triangle_hypotenuse_l113_113307

theorem right_triangle_hypotenuse (a b c : ℕ) (h1 : a^2 + b^2 = c^2) 
  (h2 : b = c - 1575) (h3 : b < 1991) : c = 1800 :=
sorry

end right_triangle_hypotenuse_l113_113307


namespace midlines_tangent_fixed_circle_l113_113882

-- Definitions of geometric objects and properties
structure Point :=
(x : ℝ) (y : ℝ)

structure Circle :=
(center : Point) (radius : ℝ)

-- Assumptions (conditions)
variable (ω1 ω2 : Circle)
variable (l1 l2 : Point → Prop) -- Representing line equations in terms of points
variable (angle : Point → Prop) -- Representing the given angle sides

-- Tangency conditions
axiom tangency1 : ∀ p : Point, l1 p → p ≠ ω1.center ∧ (ω1.center.x - p.x) ^ 2 + (ω1.center.y - p.y) ^ 2 = ω1.radius ^ 2
axiom tangency2 : ∀ p : Point, l2 p → p ≠ ω2.center ∧ (ω2.center.x - p.x) ^ 2 + (ω2.center.y - p.y) ^ 2 = ω2.radius ^ 2

-- Non-intersecting condition for circles
axiom nonintersecting : (ω1.center.x - ω2.center.x) ^ 2 + (ω1.center.y - ω2.center.y) ^ 2 > (ω1.radius + ω2.radius) ^ 2

-- Conditions for tangent circles and middle line being between them
axiom betweenness : ∀ p, angle p → (ω1.center.y < p.y ∧ p.y < ω2.center.y)

-- Midline definition and fixed circle condition
theorem midlines_tangent_fixed_circle :
  ∃ (O : Point) (d : ℝ), ∀ (T : Point → Prop), 
  (∃ (p1 p2 : Point), l1 p1 ∧ l2 p2 ∧ T p1 ∧ T p2) →
  (∀ (m : Point), T m ↔ ∃ (p1 p2 p3 p4 : Point), T p1 ∧ T p2 ∧ angle p3 ∧ angle p4 ∧ 
  m.x = (p1.x + p2.x + p3.x + p4.x) / 4 ∧ m.y = (p1.y + p2.y + p3.y + p4.y) / 4) → 
  (∀ (m : Point), (m.x - O.x) ^ 2 + (m.y - O.y) ^ 2 = d^2)
:= 
sorry

end midlines_tangent_fixed_circle_l113_113882


namespace calculate_total_money_made_l113_113534

def original_price : ℕ := 51
def discount : ℕ := 8
def num_tshirts_sold : ℕ := 130
def discounted_price : ℕ := original_price - discount
def total_money_made : ℕ := discounted_price * num_tshirts_sold

theorem calculate_total_money_made :
  total_money_made = 5590 := 
sorry

end calculate_total_money_made_l113_113534


namespace count_x_values_l113_113920

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then x 
  else if n = 1 then 3000
  else 1 / (sequence x (n - 1)) - 1 / (sequence x (n - 2))

theorem count_x_values (x : ℝ) (n : ℕ) : ℕ :=
  ∃ x : ℝ, ∃ n : ℕ, x > 0 ∧ sequence x n = 3001 :=
begin
  sorry  -- Proof to be filled in later
end

end count_x_values_l113_113920


namespace thousandths_place_digit_of_7_div_32_l113_113078

noncomputable def decimal_thousandths_digit : ℚ := 7 / 32

theorem thousandths_place_digit_of_7_div_32 :
  (decimal_thousandths_digit * 1000) % 10 = 8 :=
sorry

end thousandths_place_digit_of_7_div_32_l113_113078


namespace volume_ratio_of_trapezoidal_pyramids_l113_113033

theorem volume_ratio_of_trapezoidal_pyramids 
  (V U : ℝ) (m n m₁ n₁ : ℝ)
  (hV : V > 0) (hU : U > 0) (hm : m > 0) (hn : n > 0) (hm₁ : m₁ > 0) (hn₁ : n₁ > 0)
  (h_ratio : U / V = (m₁ + n₁)^2 / (m + n)^2) :
  U / V = (m₁ + n₁)^2 / (m + n)^2 :=
sorry

end volume_ratio_of_trapezoidal_pyramids_l113_113033


namespace distinct_integers_division_l113_113194

theorem distinct_integers_division (n : ℤ) (h : n > 1) :
  ∃ (a b c : ℤ), a = n^2 + n + 1 ∧ b = n^2 + 2 ∧ c = n^2 + 1 ∧
  n^2 < a ∧ a < (n + 1)^2 ∧ 
  n^2 < b ∧ b < (n + 1)^2 ∧ 
  n^2 < c ∧ c < (n + 1)^2 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ c ∣ (a ^ 2 + b ^ 2) := 
by
  sorry

end distinct_integers_division_l113_113194


namespace ratio_of_hypothetical_to_actual_children_l113_113945

theorem ratio_of_hypothetical_to_actual_children (C H : ℕ) 
  (h1 : H = 16 * 8)
  (h2 : ∀ N : ℕ, N = C / 8 → C * N = 512) 
  (h3 : C^2 = 512 * 8) : H / C = 2 := 
by 
  sorry

end ratio_of_hypothetical_to_actual_children_l113_113945


namespace find_f_three_l113_113853

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l113_113853


namespace smallest_positive_multiple_of_45_l113_113257

theorem smallest_positive_multiple_of_45 :
  ∃ (x : ℕ), x > 0 ∧ (x * 45 = 45) :=
by
  existsi 1
  split
  · exact Nat.one_pos
  · simp

end smallest_positive_multiple_of_45_l113_113257


namespace area_of_triangle_bounded_by_lines_l113_113071

theorem area_of_triangle_bounded_by_lines :
  let y1 := λ x : ℝ, x
  let y2 := λ x : ℝ, -x
  let y3 := λ x : ℝ, 8
  ∀ A B O : (ℝ × ℝ), 
  (A = (8, 8)) → 
  (B = (-8, 8)) → 
  (O = (0, 0)) →
  (triangle_area A B O = 64) :=
by
  intros y1 y2 y3 A B O hA hB hO
  have hA : A = (8, 8) := hA
  have hB : B = (-8, 8) := hB
  have hO : O = (0, 0) := hO
  sorry

end area_of_triangle_bounded_by_lines_l113_113071


namespace min_omega_l113_113015

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega (ω φ T : ℝ) (hω : ω > 0) (hφ1 : 0 < φ) (hφ2 : φ < Real.pi / 2)
  (hT : f ω φ T = Real.sqrt 3 / 2)
  (hx : f ω φ (Real.pi / 6) = 0) :
  ω = 4 := by
  sorry

end min_omega_l113_113015


namespace tate_initial_tickets_l113_113804

theorem tate_initial_tickets (T : ℕ) (h1 : T + 2 + (T + 2)/2 = 51) : T = 32 := 
by
  sorry

end tate_initial_tickets_l113_113804


namespace smallest_divisor_l113_113002

noncomputable def even_four_digit_number (m : ℕ) : Prop :=
  1000 ≤ m ∧ m < 10000 ∧ m % 2 = 0

def divisor_ordered (m : ℕ) (d : ℕ) : Prop :=
  d ∣ m

theorem smallest_divisor (m : ℕ) (h1 : even_four_digit_number m) (h2 : divisor_ordered m 437) :
  ∃ d,  d > 437 ∧ divisor_ordered m d ∧ (∀ e, e > 437 → divisor_ordered m e → d ≤ e) ∧ d = 874 :=
sorry

end smallest_divisor_l113_113002


namespace ones_digit_of_3_pow_52_l113_113708

theorem ones_digit_of_3_pow_52 : (3 ^ 52 % 10) = 1 := 
by sorry

end ones_digit_of_3_pow_52_l113_113708


namespace number_of_divisors_of_36_l113_113161

theorem number_of_divisors_of_36 :  
  let n := 36
  number_of_divisors n = 9 := 
by 
  sorry

end number_of_divisors_of_36_l113_113161


namespace triangle_inequality_proof_l113_113952

variable (a b c : ℝ)

-- Condition that a, b, c are side lengths of a triangle
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : b + c > a
axiom triangle_inequality3 : c + a > b

-- Theorem stating the required inequality and the condition for equality
theorem triangle_inequality_proof :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c ∧ c = a) :=
sorry

end triangle_inequality_proof_l113_113952


namespace smallest_bob_number_l113_113109

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : Set ℕ := { p | is_prime p ∧ p ∣ n }

def alice_number := 36
def bob_number (m : ℕ) : Prop := prime_factors alice_number ⊆ prime_factors m

-- Proof problem statement
theorem smallest_bob_number :
  ∃ m, bob_number m ∧ m = 6 :=
sorry

end smallest_bob_number_l113_113109


namespace find_fourth_term_geometric_progression_l113_113592

theorem find_fourth_term_geometric_progression (x : ℝ) (a1 a2 a3 : ℝ) (r : ℝ)
  (h1 : a1 = x)
  (h2 : a2 = 3 * x + 6)
  (h3 : a3 = 7 * x + 21)
  (h4 : ∃ r, a2 / a1 = r ∧ a3 / a2 = r)
  (hx : x = 3 / 2) :
  7 * (7 * x + 21) = 220.5 :=
by
  sorry

end find_fourth_term_geometric_progression_l113_113592


namespace common_tangent_exists_l113_113512

theorem common_tangent_exists:
  ∃ (a b c : ℕ), (a + b + c = 11) ∧
  ( ∀ (x y : ℝ),
      (y = x^2 + 12/5) ∧ 
      (x = y^2 + 99/10) ∧ 
      (a*x + b*y = c) ∧ 
      0 < a ∧ 0 < b ∧ 0 < c ∧ 
      Int.gcd (Int.gcd a b) c = 1
  ) := 
by
  sorry

end common_tangent_exists_l113_113512


namespace molecular_weight_proof_l113_113082

/-- Atomic weights in atomic mass units (amu) --/
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_P : ℝ := 30.97

/-- Number of atoms in the compound --/
def num_Al : ℝ := 2
def num_O : ℝ := 4
def num_H : ℝ := 6
def num_N : ℝ := 3
def num_P : ℝ := 1

/-- calculating the molecular weight --/
def molecular_weight : ℝ := 
  (num_Al * atomic_weight_Al) +
  (num_O * atomic_weight_O) +
  (num_H * atomic_weight_H) +
  (num_N * atomic_weight_N) +
  (num_P * atomic_weight_P)

-- The proof statement
theorem molecular_weight_proof : molecular_weight = 197.02 := 
by
  sorry

end molecular_weight_proof_l113_113082


namespace exp_product_correct_l113_113083

def exp_1 := (2 : ℕ) ^ 4
def exp_2 := (3 : ℕ) ^ 2
def exp_3 := (5 : ℕ) ^ 2
def exp_4 := (7 : ℕ)
def exp_5 := (11 : ℕ)
def final_value := exp_1 * exp_2 * exp_3 * exp_4 * exp_5

theorem exp_product_correct : final_value = 277200 := by
  sorry

end exp_product_correct_l113_113083


namespace smallest_positive_multiple_of_45_is_45_l113_113247

noncomputable def smallest_positive_multiple_of_45 : ℕ :=
  if h : ∃ x : ℕ, x > 0 then 45 else 0

theorem smallest_positive_multiple_of_45_is_45
  (h : ∃ x : ℕ, x > 0) : smallest_positive_multiple_of_45 = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113247


namespace solve_inequalities_solve_linear_system_l113_113885

-- System of Inequalities
theorem solve_inequalities (x : ℝ) (h1 : x + 2 > 1) (h2 : 2 * x < x + 3) : -1 < x ∧ x < 3 :=
by
  sorry

-- System of Linear Equations
theorem solve_linear_system (x y : ℝ) (h1 : 3 * x + 2 * y = 12) (h2 : 2 * x - y = 1) : x = 2 ∧ y = 3 :=
by
  sorry

end solve_inequalities_solve_linear_system_l113_113885


namespace difference_of_numbers_l113_113865

theorem difference_of_numbers (x y : ℕ) (h₁ : x + y = 50) (h₂ : Nat.gcd x y = 5) :
  (x - y = 20 ∨ y - x = 20 ∨ x - y = 40 ∨ y - x = 40) :=
sorry

end difference_of_numbers_l113_113865


namespace purely_imaginary_z_eq_a2_iff_a2_l113_113363

theorem purely_imaginary_z_eq_a2_iff_a2 (a : Real) : 
(∃ (b : Real), a^2 - a - 2 = 0 ∧ a + 1 ≠ 0) → a = 2 :=
by
  sorry

end purely_imaginary_z_eq_a2_iff_a2_l113_113363


namespace cubes_with_odd_neighbors_in_5x5x5_l113_113345

theorem cubes_with_odd_neighbors_in_5x5x5 (unit_cubes : Fin 125 → ℕ) 
  (neighbors : ∀ (i : Fin 125), Fin 125 → Prop) : ∃ n, n = 62 := 
by
  sorry

end cubes_with_odd_neighbors_in_5x5x5_l113_113345


namespace customers_total_l113_113310

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

end customers_total_l113_113310


namespace largest_of_20_consecutive_even_integers_l113_113048

theorem largest_of_20_consecutive_even_integers (x : ℕ) 
  (h : 20 * (x + 19) = 8000) : (x + 38) = 419 :=
  sorry

end largest_of_20_consecutive_even_integers_l113_113048


namespace median_line_eqn_l113_113414

theorem median_line_eqn (A B C : ℝ × ℝ)
  (hA : A = (3, 7)) (hB : B = (5, -1)) (hC : C = (-2, -5)) :
  ∃ m b : ℝ, (4, -3, -7) = (m, b, 0) :=
by sorry

end median_line_eqn_l113_113414


namespace james_earnings_l113_113986

-- Define the conditions
def rain_gallons_per_inch : ℕ := 15
def rain_monday : ℕ := 4
def rain_tuesday : ℕ := 3
def price_per_gallon : ℝ := 1.2

-- State the theorem to be proved
theorem james_earnings : (rain_monday * rain_gallons_per_inch + rain_tuesday * rain_gallons_per_inch) * price_per_gallon = 126 :=
by
  sorry

end james_earnings_l113_113986


namespace range_of_a_l113_113154

noncomputable def f (x : ℤ) (a : ℝ) := (3 * x^2 + a * x + 26) / (x + 1)

theorem range_of_a (a : ℝ) :
  (∃ x : ℕ+, f x a ≤ 2) → a ≤ -15 :=
by
  sorry

end range_of_a_l113_113154


namespace martha_bottles_l113_113107

def total_bottles_left (a b c d : ℕ) : ℕ :=
  a + b + c - d

theorem martha_bottles : total_bottles_left 4 4 5 3 = 10 :=
by
  sorry

end martha_bottles_l113_113107


namespace pets_remaining_l113_113114

-- Definitions based on conditions
def initial_puppies : ℕ := 7
def initial_kittens : ℕ := 6
def sold_puppies : ℕ := 2
def sold_kittens : ℕ := 3

-- Theorem statement
theorem pets_remaining : initial_puppies + initial_kittens - (sold_puppies + sold_kittens) = 8 :=
by
  sorry

end pets_remaining_l113_113114


namespace mentorship_arrangement_is_90_l113_113872

def mentorship_arrangement_ways : Nat :=
  let total_students := 5
  let teachers := 3
  -- Calculation of groups arrangement
  let ways_to_group := (Nat.choose total_students 2 * Nat.choose (total_students - 2) 2 * Nat.choose (total_students - 4) 1) / Nat.factorial 2
  -- Ways to assign the groups to the teachers
  let ways_to_assign := ways_to_group * Nat.factorial teachers
  ways_to_assign

theorem mentorship_arrangement_is_90 : mentorship_arrangement_ways = 90 := by
  sorry

end mentorship_arrangement_is_90_l113_113872


namespace knight_probability_sum_l113_113425

def num_knights := 30
def chosen_knights := 4

-- Calculate valid placements where no knights are adjacent
def valid_placements : ℕ := 26 * 24 * 22 * 20
-- Calculate total unrestricted placements
def total_placements : ℕ := 26 * 27 * 28 * 29
-- Calculate probability
def P : ℚ := 1 - (valid_placements : ℚ) / total_placements

-- Simplify the fraction P to its lowest terms: 553/1079
def simplified_num := 553
def simplified_denom := 1079

-- Sum of the numerator and denominator of simplified P
def sum_numer_denom := simplified_num + simplified_denom

theorem knight_probability_sum :
  sum_numer_denom = 1632 :=
by
  -- Proof is omitted
  sorry

end knight_probability_sum_l113_113425


namespace max_visible_unit_cubes_from_corner_l113_113461

theorem max_visible_unit_cubes_from_corner :
  let n := 11
  let faces_visible := 3 * n^2
  let edges_shared := 3 * (n - 1)
  let corner_cube := 1
  faces_visible - edges_shared + corner_cube = 331 := by
  let n := 11
  let faces_visible := 3 * n^2
  let edges_shared := 3 * (n - 1)
  let corner_cube := 1
  have result : faces_visible - edges_shared + corner_cube = 331 := by
    sorry
  exact result

end max_visible_unit_cubes_from_corner_l113_113461


namespace margarita_vs_ricciana_l113_113197

-- Ricciana's distances
def ricciana_run : ℕ := 20
def ricciana_jump : ℕ := 4
def ricciana_total : ℕ := ricciana_run + ricciana_jump

-- Margarita's distances
def margarita_run : ℕ := 18
def margarita_jump : ℕ := (2 * ricciana_jump) - 1
def margarita_total : ℕ := margarita_run + margarita_jump

-- Statement to prove Margarita ran and jumped 1 more foot than Ricciana
theorem margarita_vs_ricciana : margarita_total = ricciana_total + 1 := by
  sorry

end margarita_vs_ricciana_l113_113197


namespace smallest_positive_multiple_of_45_is_45_l113_113268

theorem smallest_positive_multiple_of_45_is_45 :
  ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * 45 = n ∧ m = 1 :=
by
  existsi 45
  existsi 1
  sorry

end smallest_positive_multiple_of_45_is_45_l113_113268


namespace incorrect_value_of_observation_l113_113860

theorem incorrect_value_of_observation
  (mean_initial : ℝ) (n : ℕ) (sum_initial: ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mean_corrected : ℝ)
  (h1 : mean_initial = 36) 
  (h2 : n = 50) 
  (h3 : sum_initial = n * mean_initial) 
  (h4 : correct_value = 45) 
  (h5 : mean_corrected = 36.5) 
  (sum_corrected : ℝ) 
  (h6 : sum_corrected = n * mean_corrected) : 
  incorrect_value = 20 := 
by 
  sorry

end incorrect_value_of_observation_l113_113860


namespace hcf_36_84_l113_113601

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := by
  sorry

end hcf_36_84_l113_113601


namespace inequality_square_l113_113629

theorem inequality_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 :=
sorry

end inequality_square_l113_113629


namespace points_on_line_eqdist_quadrants_l113_113323

theorem points_on_line_eqdist_quadrants :
  ∀ (x y : ℝ), 4 * x - 3 * y = 12 ∧ |x| = |y| → 
  (x > 0 ∧ y > 0 ∨ x > 0 ∧ y < 0) :=
by
  sorry

end points_on_line_eqdist_quadrants_l113_113323


namespace solve_for_p_l113_113365

theorem solve_for_p (a b c p t : ℝ) (h1 : a + b + c + p = 360) (h2 : t = 180 - c) : 
  p = 180 - a - b + t :=
by
  sorry

end solve_for_p_l113_113365


namespace inverse_variation_l113_113557

theorem inverse_variation (x y : ℝ) (h1 : 7 * y = 1400 / x^3) (h2 : x = 4) : y = 25 / 8 :=
  by
  sorry

end inverse_variation_l113_113557


namespace log_expression_eq_l113_113706

theorem log_expression_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x^2 / Real.log (y^4)) * 
  (Real.log (y^3) / Real.log (x^6)) * 
  (Real.log (x^4) / Real.log (y^3)) * 
  (Real.log (y^4) / Real.log (x^2)) * 
  (Real.log (x^6) / Real.log y) = 
  16 * Real.log x / Real.log y := 
sorry

end log_expression_eq_l113_113706


namespace jeanne_additional_tickets_l113_113000

-- Define the costs
def ferris_wheel_cost : ℕ := 5
def roller_coaster_cost : ℕ := 4
def bumper_cars_cost : ℕ := 4
def jeanne_tickets : ℕ := 5

-- Calculate the total cost
def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost

-- Define the proof problem
theorem jeanne_additional_tickets : total_cost - jeanne_tickets = 8 :=
by sorry

end jeanne_additional_tickets_l113_113000


namespace b2009_value_l113_113514

noncomputable def b (n : ℕ) : ℝ := sorry

axiom b_recursion (n : ℕ) (hn : 2 ≤ n) : b n = b (n - 1) * b (n + 1)

axiom b1_value : b 1 = 2 + Real.sqrt 3
axiom b1776_value : b 1776 = 10 + Real.sqrt 3

theorem b2009_value : b 2009 = -4 + 8 * Real.sqrt 3 := 
by sorry

end b2009_value_l113_113514


namespace students_tried_out_l113_113543

theorem students_tried_out (x : ℕ) (h1 : 8 * (x - 17) = 384) : x = 65 := 
by
  sorry

end students_tried_out_l113_113543


namespace max_b_integer_l113_113430

theorem max_b_integer (b : ℤ) : (∀ x : ℝ, x^2 + (b : ℝ) * x + 20 ≠ -10) → b ≤ 10 :=
by
  sorry

end max_b_integer_l113_113430


namespace incorrect_value_of_observation_l113_113861

theorem incorrect_value_of_observation
  (mean_initial : ℝ) (n : ℕ) (sum_initial: ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mean_corrected : ℝ)
  (h1 : mean_initial = 36) 
  (h2 : n = 50) 
  (h3 : sum_initial = n * mean_initial) 
  (h4 : correct_value = 45) 
  (h5 : mean_corrected = 36.5) 
  (sum_corrected : ℝ) 
  (h6 : sum_corrected = n * mean_corrected) : 
  incorrect_value = 20 := 
by 
  sorry

end incorrect_value_of_observation_l113_113861


namespace smallest_positive_multiple_of_45_l113_113233

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ ∃ y : ℕ, x = 45 * y ∧ x = 45 := by
  exists 45
  use 1
  split
  . exact lt_add_one 0
  . split
  . exact (by simp [mul_one])
  . rfl

end smallest_positive_multiple_of_45_l113_113233


namespace max_inscribed_triangle_area_sum_l113_113726

noncomputable def inscribed_triangle_area (a b : ℝ) (h_a : a = 12) (h_b : b = 13) : ℝ :=
  let s := min (a / (Real.sqrt 3 / 2)) (b / (1 / 2))
  (Real.sqrt 3 / 4) * s^2

theorem max_inscribed_triangle_area_sum :
  inscribed_triangle_area 12 13 (by rfl) (by rfl) = 48 * Real.sqrt 3 - 0 :=
by
  sorry

#eval 48 + 3 + 0
-- Expected Result: 51

end max_inscribed_triangle_area_sum_l113_113726


namespace linear_equation_l113_113627

noncomputable def is_linear (k : ℝ) : Prop :=
  2 * (|k|) = 1 ∧ k ≠ 1

theorem linear_equation (k : ℝ) : is_linear k ↔ k = -1 :=
by
  sorry

end linear_equation_l113_113627


namespace translate_quadratic_vertex_right_l113_113546

theorem translate_quadratic_vertex_right : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = 2 * (x - 4)^2 - 3) ∧ 
  (∃ (g : ℝ → ℝ), (∀ x, g x = 2 * ((x - 1) - 3)^2 - 3))) → 
  (∃ v : ℝ × ℝ, v = (4, -3)) :=
sorry

end translate_quadratic_vertex_right_l113_113546


namespace problem_not_true_equation_l113_113756

theorem problem_not_true_equation
  (a b : ℝ)
  (h : a / b = 2 / 3) : a / b ≠ (a + 2) / (b + 2) := 
sorry

end problem_not_true_equation_l113_113756


namespace harmonic_mean_1999_2001_is_2000_l113_113317

def harmonic_mean (a b : ℕ) : ℚ := (2 * a * b) / (a + b)

theorem harmonic_mean_1999_2001_is_2000 :
  abs (harmonic_mean 1999 2001 - 2000 : ℚ) < 1 := by
  -- Actual proof omitted
  sorry

end harmonic_mean_1999_2001_is_2000_l113_113317


namespace total_toys_correct_l113_113992

-- Define the conditions
def JaxonToys : ℕ := 15
def GabrielToys : ℕ := 2 * JaxonToys
def JerryToys : ℕ := GabrielToys + 8

-- Define the total number of toys
def TotalToys : ℕ := JaxonToys + GabrielToys + JerryToys

-- State the theorem
theorem total_toys_correct : TotalToys = 83 :=
by
  -- Skipping the proof as per instruction
  sorry

end total_toys_correct_l113_113992


namespace distance_between_consecutive_trees_l113_113459

noncomputable def distance_between_trees (yard_length : ℝ) (num_trees : ℕ) (obstacle_pos : ℝ) (obstacle_gap : ℝ) : ℝ :=
  let planting_distance := yard_length - obstacle_gap
  let num_gaps := num_trees - 1
  planting_distance / num_gaps

theorem distance_between_consecutive_trees :
  distance_between_trees 600 36 250 10 = 16.857 := by
  sorry

end distance_between_consecutive_trees_l113_113459


namespace product_of_b_values_is_neg_12_l113_113039

theorem product_of_b_values_is_neg_12 (b : ℝ) (y1 y2 x1 : ℝ) (h1 : y1 = 3) (h2 : y2 = 7) (h3 : x1 = 2) (h4 : y2 - y1 = 4) (h5 : ∃ b1 b2, b1 = x1 - 4 ∧ b2 = x1 + 4) : 
  (b1 * b2 = -12) :=
by
  sorry

end product_of_b_values_is_neg_12_l113_113039


namespace problem_solution_l113_113152

noncomputable def length_segment_AB : ℝ :=
  let k : ℝ := 1 -- derived from 3k - 3 = 0
  let A : ℝ × ℝ := (0, k) -- point (0, k)
  let C : ℝ × ℝ := (3, -1) -- center of the circle
  let r : ℝ := 1 -- radius of the circle
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) -- distance formula
  Real.sqrt (AC^2 - r^2)

theorem problem_solution :
  length_segment_AB = 2 * Real.sqrt 3 :=
by
  sorry

end problem_solution_l113_113152


namespace intersection_A_B_l113_113157

def A := {x : ℤ | ∃ k : ℤ, x = 2 * k + 1}
def B := {x : ℤ | 0 < x ∧ x < 5}

theorem intersection_A_B : A ∩ B = {1, 3} :=
by
  sorry

end intersection_A_B_l113_113157


namespace sequence_term_3001_exists_exactly_4_values_l113_113924

theorem sequence_term_3001_exists_exactly_4_values (x : ℝ) (h_pos : x > 0) :
  (∃ n, (1 < n) ∧ n < 6 ∧ (sequence n) = 3001) ↔ 
  (x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999) :=
by 
  -- Definition of the sequence a_n based on the provided recursive relationship
  let a : ℕ → ℝ
  | 1 => x
  | 2 => 3000
  | 3 => 3001 / x
  | 4 => (x + 3001) / (3000 * x)
  | 5 => (x + 1) / 3000
  | 6 => x
  | 7 => 3000
  | n => a ((n - 1) % 5 + 1)  -- Applying periodicity
  exactly[4] sorry

end sequence_term_3001_exists_exactly_4_values_l113_113924


namespace correct_mark_l113_113567

theorem correct_mark (x : ℕ) (h1 : 73 - x = 10) : x = 63 :=
by
  sorry

end correct_mark_l113_113567


namespace central_number_l113_113493

theorem central_number (C : ℕ) (verts : Finset ℕ) (h : verts = {1, 2, 7, 8, 9, 13, 14}) :
  (∀ T ∈ {t | ∃ a b c, (a + b + c) % 3 = 0 ∧ a ∈ verts ∧ b ∈ verts ∧ c ∈ verts}, (T + C) % 3 = 0) →
  C = 9 :=
by
  sorry

end central_number_l113_113493


namespace max_three_numbers_condition_l113_113798

theorem max_three_numbers_condition (n : ℕ) 
  (x : Fin n → ℝ) 
  (h : ∀ (i j k : Fin n), i ≠ j → i ≠ k → j ≠ k → (x i)^2 > (x j) * (x k)) : n ≤ 3 := 
sorry

end max_three_numbers_condition_l113_113798


namespace savings_after_expense_increase_l113_113448

-- Define constants and initial conditions
def salary : ℝ := 7272.727272727273
def savings_rate : ℝ := 0.10
def expense_increase_rate : ℝ := 0.05

-- Define initial savings, expenses, and new expenses
def initial_savings : ℝ := savings_rate * salary
def initial_expenses : ℝ := salary - initial_savings
def new_expenses : ℝ := initial_expenses * (1 + expense_increase_rate)
def new_savings : ℝ := salary - new_expenses

-- The theorem statement
theorem savings_after_expense_increase : new_savings = 400 := by
  sorry

end savings_after_expense_increase_l113_113448


namespace find_f_3_l113_113839

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l113_113839


namespace equilateral_triangle_perimeter_isosceles_triangle_leg_length_l113_113731

-- Definitions for equilateral triangle problem
def side_length_equilateral : ℕ := 12
def perimeter_equilateral := side_length_equilateral * 3

-- Definitions for isosceles triangle problem
def perimeter_isosceles : ℕ := 72
def base_length_isosceles : ℕ := 28
def leg_length_isosceles := (perimeter_isosceles - base_length_isosceles) / 2

-- Theorem statement
theorem equilateral_triangle_perimeter : perimeter_equilateral = 36 := 
by
  sorry

theorem isosceles_triangle_leg_length : leg_length_isosceles = 22 := 
by
  sorry

end equilateral_triangle_perimeter_isosceles_triangle_leg_length_l113_113731


namespace correct_algorithm_statement_l113_113437

def reversible : Prop := false -- Algorithms are generally not reversible.
def endless : Prop := false -- Algorithms should not run endlessly.
def unique_algo : Prop := false -- Not always one single algorithm for a task.
def simple_convenient : Prop := true -- Algorithms should be simple and convenient.

theorem correct_algorithm_statement : simple_convenient = true :=
by
  sorry

end correct_algorithm_statement_l113_113437


namespace FDI_in_rural_AndhraPradesh_l113_113729

-- Definitions from conditions
def total_FDI : ℝ := 300 -- Total FDI calculated
def FDI_Gujarat : ℝ := 0.30 * total_FDI
def FDI_Gujarat_Urban : ℝ := 0.80 * FDI_Gujarat
def FDI_AndhraPradesh : ℝ := 0.20 * total_FDI
def FDI_AndhraPradesh_Rural : ℝ := 0.50 * FDI_AndhraPradesh 

-- Given the conditions, prove the size of FDI in rural Andhra Pradesh is 30 million
theorem FDI_in_rural_AndhraPradesh :
  FDI_Gujarat_Urban = 72 → FDI_AndhraPradesh_Rural = 30 :=
by
  sorry

end FDI_in_rural_AndhraPradesh_l113_113729


namespace exists_negative_root_of_P_l113_113741

def P(x : ℝ) : ℝ := x^7 - 2 * x^6 - 7 * x^4 - x^2 + 10

theorem exists_negative_root_of_P : ∃ x : ℝ, x < 0 ∧ P x = 0 :=
sorry

end exists_negative_root_of_P_l113_113741


namespace f_leq_binom_l113_113382

-- Define the function f with given conditions
def f (m n : ℕ) : ℕ := if m = 1 ∨ n = 1 then 1 else sorry

-- State the property to be proven
theorem f_leq_binom (m n : ℕ) (h2 : 2 ≤ m) (h2' : 2 ≤ n) :
  f m n ≤ Nat.choose (m + n) n := 
sorry

end f_leq_binom_l113_113382


namespace num_x_for_3001_in_sequence_l113_113923

theorem num_x_for_3001_in_sequence :
  let seq : ℕ → ℝ := λ n, if n = 0 then x else if n = 1 then 3000 else seq (n - 2) * seq (n - 1) - 1
  let appears (a : ℝ) : Prop := ∃ n : ℕ, seq n = a
  ∃ (x : ℝ), appears seq 3001 :=
      sorry

  ∃ (x : ℝ→ ℕ) (hx : (∑ x. appear 3001) = 4 :=  ∑ sorry

end num_x_for_3001_in_sequence_l113_113923


namespace intersection_of_M_and_N_l113_113958

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x^2 = 2 * x}

theorem intersection_of_M_and_N : M ∩ N = {0} := 
by sorry

end intersection_of_M_and_N_l113_113958


namespace smallest_positive_multiple_of_45_l113_113275

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l113_113275


namespace coin_game_goal_l113_113715

theorem coin_game_goal (a b : ℕ) (h_diff : a ≤ 3 * b ∧ b ≤ 3 * a) (h_sum : (a + b) % 4 = 0) :
  ∃ x y p q : ℕ, (a + 2 * x - 2 * y = 3 * (b + 2 * p - 2 * q)) ∨ (a + 2 * y - 2 * x = 3 * (b + 2 * q - 2 * p)) :=
sorry

end coin_game_goal_l113_113715


namespace total_spent_on_clothing_l113_113783

-- Define the individual costs
def shorts_cost : ℝ := 15
def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51

-- Define the proof problem to show the total cost
theorem total_spent_on_clothing : shorts_cost + jacket_cost + shirt_cost = 42.33 := by
  sorry

end total_spent_on_clothing_l113_113783


namespace cos_C_of_triangle_l113_113981

theorem cos_C_of_triangle (A B C : ℝ) (hA : sin A = 4 / 5) (hB : cos B = 12 / 13) :
  cos C = -16 / 65 :=
sorry

end cos_C_of_triangle_l113_113981


namespace find_parabola_focus_l113_113476

theorem find_parabola_focus : 
  ∀ (x y : ℝ), (y = 2 * x ^ 2 + 4 * x - 1) → (∃ p q : ℝ, p = -1 ∧ q = -(23:ℝ) / 8 ∧ (y = 2 * x ^ 2 + 4 * x - 1) → (x, y) = (p, q)) :=
by
  sorry

end find_parabola_focus_l113_113476


namespace total_sales_l113_113439

theorem total_sales (S : ℝ) (remitted : ℝ) : 
  (∀ S, remitted = S - (0.05 * 10000 + 0.04 * (S - 10000)) → remitted = 31100) → S = 32500 :=
by
  sorry

end total_sales_l113_113439
