import Mathlib

namespace committee_probability_l1012_101296

/--
Suppose there are 24 members in a club: 12 boys and 12 girls.
A 5-person committee is chosen at random.
Prove that the probability of having at least 2 boys and at least 2 girls in the committee is 121/177.
-/
theorem committee_probability :
  let boys := 12
  let girls := 12
  let total_members := 24
  let committee_size := 5
  let all_ways := Nat.choose total_members committee_size
  let invalid_ways := 2 * Nat.choose boys committee_size + 2 * (Nat.choose boys 1 * Nat.choose girls 4)
  let valid_ways := all_ways - invalid_ways
  let probability := valid_ways / all_ways
  probability = 121 / 177 :=
by
  sorry

end committee_probability_l1012_101296


namespace complement_intersection_l1012_101286

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2 * x > 0}

def B : Set ℝ := {x | -3 < x ∧ x < 1}

def compA : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem complement_intersection :
  (compA ∩ B) = {x | 0 ≤ x ∧ x < 1} := by
  -- The proof goes here
  sorry

end complement_intersection_l1012_101286


namespace product_of_slopes_constant_l1012_101214

noncomputable def ellipse (x y : ℝ) := x^2 / 8 + y^2 / 4 = 1

theorem product_of_slopes_constant (a b : ℝ) (h_a_gt_b : a > b) (h_a_b_pos : 0 < a ∧ 0 < b)
  (e : ℝ) (h_eccentricity : e = (Real.sqrt 2) / 2) (P : ℝ × ℝ) (h_point_on_ellipse : (P.1, P.2) = (2, Real.sqrt 2)) :
  (∃ C : ℝ → ℝ → Prop, C = ellipse) ∧ (∃ k : ℝ, -k * 1/2 = -1 / 2) := sorry

end product_of_slopes_constant_l1012_101214


namespace fraction_subtraction_l1012_101211

theorem fraction_subtraction (h : ((8 : ℚ) / 21 - (10 / 63) = (2 / 9))) : 
  8 / 21 - 10 / 63 = 2 / 9 :=
by
  sorry

end fraction_subtraction_l1012_101211


namespace perpendicular_line_through_point_l1012_101293

theorem perpendicular_line_through_point 
 {x y : ℝ}
 (p : (ℝ × ℝ)) 
 (point : p = (-2, 1)) 
 (perpendicular : ∀ x y, 2 * x - y + 4 = 0) : 
 (∀ x y, x + 2 * y = 0) ∧ (p.fst = -2 ∧ p.snd = 1) :=
by
  sorry

end perpendicular_line_through_point_l1012_101293


namespace b_range_l1012_101201

noncomputable def f (a b x : ℝ) := (x - 1) * Real.log x - a * x + a + b

theorem b_range (a b : ℝ)
  (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 = 0 ∧ f a b x2 = 0) :
  b < 0 :=
sorry

end b_range_l1012_101201


namespace sally_investment_l1012_101258

theorem sally_investment (m : ℝ) (hmf : 0 ≤ m) 
  (total_investment : m + 7 * m = 200000) : 
  7 * m = 175000 :=
by
  -- Proof goes here
  sorry

end sally_investment_l1012_101258


namespace line_through_intersection_and_origin_l1012_101207

-- Definitions of the lines
def l1 (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def l2 (x y : ℝ) : Prop := y = 1 - x

-- Prove that the line passing through the intersection of l1 and l2 and the origin has the equation 3x + 2y = 0
theorem line_through_intersection_and_origin (x y : ℝ) 
  (h1 : 2 * x - y + 7 = 0) (h2 : y = 1 - x) : 3 * x + 2 * y = 0 := 
sorry

end line_through_intersection_and_origin_l1012_101207


namespace cone_height_l1012_101275

theorem cone_height (r : ℝ) (θ : ℝ) (h : ℝ)
  (hr : r = 1)
  (hθ : θ = (2 / 3) * Real.pi)
  (h_eq : h = 2 * Real.sqrt 2) :
  ∃ l : ℝ, l = 3 ∧ h = Real.sqrt (l^2 - r^2) :=
by
  sorry

end cone_height_l1012_101275


namespace function_domain_real_l1012_101273

theorem function_domain_real (k : ℝ) : 0 ≤ k ∧ k < 4 ↔ (∀ x : ℝ, k * x^2 + k * x + 1 ≠ 0) :=
by
  sorry

end function_domain_real_l1012_101273


namespace triangle_side_lengths_l1012_101291

-- Define the variables a, b, and c
variables {a b c : ℝ}

-- Assume that a, b, and c are the lengths of the sides of a triangle
-- and the given equation holds
theorem triangle_side_lengths (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) 
    (h_eq : a^2 + 4*a*c + 3*c^2 - 3*a*b - 7*b*c + 2*b^2 = 0) : 
    a + c - 2*b = 0 :=
by
  sorry

end triangle_side_lengths_l1012_101291


namespace polynomial_remainder_l1012_101226

theorem polynomial_remainder (x : ℂ) :
  (x ^ 2030 + 1) % (x ^ 6 - x ^ 4 + x ^ 2 - 1) = x ^ 2 - 1 :=
by
  sorry

end polynomial_remainder_l1012_101226


namespace min_distance_from_P_to_origin_l1012_101205

noncomputable def distance_to_origin : ℝ := 8 / 5

theorem min_distance_from_P_to_origin
  (P : ℝ × ℝ)
  (hA : P.1^2 + P.2^2 = 1)
  (hB : (P.1 - 3)^2 + (P.2 + 4)^2 = 10)
  (h_tangent : PE = PD) :
  dist P (0, 0) = distance_to_origin := 
sorry

end min_distance_from_P_to_origin_l1012_101205


namespace geometric_progression_identity_l1012_101256

theorem geometric_progression_identity 
  (a b c d : ℝ) 
  (h1 : c^2 = b * d) 
  (h2 : b^2 = a * c) 
  (h3 : a * d = b * c) : 
  (a - c)^2 + (b - c)^2 + (b - d)^2 = (a - d)^2 :=
by 
  sorry

end geometric_progression_identity_l1012_101256


namespace distance_between_parallel_lines_l1012_101210

theorem distance_between_parallel_lines (a d : ℝ) :
  (∀ x y : ℝ, 2 * x - y + 3 = 0 ∧ a * x - y + 4 = 0 → (2 = a ∧ d = |(3 - 4)| / Real.sqrt (2 ^ 2 + (-1) ^ 2))) → 
  (a = 2 ∧ d = Real.sqrt 5 / 5) :=
by 
  sorry

end distance_between_parallel_lines_l1012_101210


namespace parallel_lines_condition_l1012_101221

-- We define the conditions as Lean definitions
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + (a^2 - 1) = 0
def parallel_condition (a : ℝ) : Prop := (a ≠ 0) ∧ (a ≠ 1) ∧ (a ≠ -1) ∧ (a * (a^2 - 1) ≠ 6)

-- Mathematically equivalent Lean 4 statement
theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, line1 a x y → line2 a x y → (line1 a x y ↔ line2 a x y)) ↔ (a = -1) :=
by 
  -- The full proof would be written here
  sorry

end parallel_lines_condition_l1012_101221


namespace discount_percentage_l1012_101299

theorem discount_percentage (SP CP SP' discount_gain_percentage: ℝ) 
  (h1 : SP = 30) 
  (h2 : SP = CP + 0.25 * CP) 
  (h3 : SP' = CP + 0.125 * CP) 
  (h4 : discount_gain_percentage = ((SP - SP') / SP) * 100) :
  discount_gain_percentage = 10 :=
by
  -- Skipping the proof
  sorry

end discount_percentage_l1012_101299


namespace krishan_money_l1012_101294

/-- Given that the ratio of money between Ram and Gopal is 7:17, the ratio of money between Gopal and Krishan is 7:17, and Ram has Rs. 588, prove that Krishan has Rs. 12,065. -/
theorem krishan_money (R G K : ℝ) (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : R = 588) : K = 12065 :=
by
  sorry

end krishan_money_l1012_101294


namespace abs_difference_of_opposite_signs_l1012_101246

theorem abs_difference_of_opposite_signs (a b : ℝ) (ha : |a| = 4) (hb : |b| = 2) (hdiff : a * b < 0) : |a - b| = 6 := 
sorry

end abs_difference_of_opposite_signs_l1012_101246


namespace find_b_l1012_101219

theorem find_b (b : ℝ) : (∃ x : ℝ, (x^3 - 3*x^2 = -3*x + b ∧ (3*x^2 - 6*x = -3))) → b = 1 :=
by
  intros h
  sorry

end find_b_l1012_101219


namespace neither_necessary_nor_sufficient_l1012_101298

theorem neither_necessary_nor_sufficient (x : ℝ) :
  ¬ ((-1 < x ∧ x < 2) → (|x - 2| < 1)) ∧ ¬ ((|x - 2| < 1) → (-1 < x ∧ x < 2)) :=
by
  sorry

end neither_necessary_nor_sufficient_l1012_101298


namespace find_ratio_l1012_101243

theorem find_ratio (a b : ℝ) (h : a / b = 3 / 2) : (a + b) / a = 5 / 3 :=
sorry

end find_ratio_l1012_101243


namespace fixed_point_quadratic_l1012_101215

theorem fixed_point_quadratic : 
  (∀ m : ℝ, 3 * a ^ 2 - m * a + 2 * m + 1 = b) → (a = 2 ∧ b = 13) := 
by sorry

end fixed_point_quadratic_l1012_101215


namespace wooden_block_length_is_correct_l1012_101225

noncomputable def length_of_block : ℝ :=
  let initial_length := 31
  let reduction := 30 / 100
  initial_length - reduction

theorem wooden_block_length_is_correct :
  length_of_block = 30.7 :=
by
  sorry

end wooden_block_length_is_correct_l1012_101225


namespace right_angled_triangle_solution_l1012_101280

theorem right_angled_triangle_solution:
  ∃ (a b c : ℕ),
    (a^2 + b^2 = c^2) ∧
    (a + b + c = (a * b) / 2) ∧
    ((a, b, c) = (6, 8, 10) ∨ (a, b, c) = (5, 12, 13)) :=
by
  sorry

end right_angled_triangle_solution_l1012_101280


namespace hikers_rate_l1012_101244

noncomputable def rate_up (rate_down := 15) : ℝ := 5

theorem hikers_rate :
  let R := rate_up
  let distance_down := rate_down
  let time := 2
  let rate_down := 1.5 * R
  distance_down = rate_down * time → R = 5 :=
by
  intro h
  sorry

end hikers_rate_l1012_101244


namespace initial_percentage_filled_l1012_101282

theorem initial_percentage_filled (capacity : ℝ) (added : ℝ) (final_fraction : ℝ) (initial_water : ℝ) :
  capacity = 80 → added = 20 → final_fraction = 3/4 → 
  initial_water = (final_fraction * capacity - added) → 
  100 * (initial_water / capacity) = 50 :=
by
  intros
  sorry

end initial_percentage_filled_l1012_101282


namespace geometric_sequence_sum_l1012_101236

theorem geometric_sequence_sum (a : ℕ → ℝ) (S₄ : ℝ) (S₈ : ℝ) (r : ℝ) 
    (h1 : r = 2) 
    (h2 : S₄ = a 0 + a 0 * r + a 0 * r^2 + a 0 * r^3)
    (h3 : S₄ = 1) 
    (h4 : S₈ = a 0 + a 0 * r + a 0 * r^2 + a 0 * r^3 + a 0 * r^4 + a 0 * r^5 + a 0 * r^6 + a 0 * r^7) :
    S₈ = 17 := by
  sorry

end geometric_sequence_sum_l1012_101236


namespace tan_A_mul_tan_B_lt_one_l1012_101235

theorem tan_A_mul_tan_B_lt_one (A B C : ℝ) (hC: C > 90) (hABC : A + B + C = 180) :
    Real.tan A * Real.tan B < 1 :=
sorry

end tan_A_mul_tan_B_lt_one_l1012_101235


namespace minimum_value_of_function_l1012_101233

theorem minimum_value_of_function :
  ∃ (y : ℝ), y > 0 ∧
  (∀ z : ℝ, z > 0 → y^2 + 10 * y + 100 / y^3 ≤ z^2 + 10 * z + 100 / z^3) ∧ 
  y^2 + 10 * y + 100 / y^3 = 50^(2/3) + 10 * 50^(1/3) + 2 := 
sorry

end minimum_value_of_function_l1012_101233


namespace other_train_speed_l1012_101267

noncomputable def speed_of_other_train (l1 l2 v1 : ℕ) (t : ℝ) : ℝ := 
  let relative_speed := (l1 + l2) / 1000 / (t / 3600)
  relative_speed - v1

theorem other_train_speed :
  speed_of_other_train 210 260 40 16.918646508279338 = 60 := 
by
  sorry

end other_train_speed_l1012_101267


namespace rational_add_positive_square_l1012_101284

theorem rational_add_positive_square (a : ℚ) : a^2 + 1 > 0 := by
  sorry

end rational_add_positive_square_l1012_101284


namespace cube_red_faces_one_third_l1012_101209

theorem cube_red_faces_one_third (n : ℕ) (h : 6 * n^3 ≠ 0) : 
  (2 * n^2) / (6 * n^3) = 1 / 3 → n = 1 :=
by sorry

end cube_red_faces_one_third_l1012_101209


namespace number_that_multiplies_b_l1012_101240

theorem number_that_multiplies_b (a b x : ℝ) (h0 : 4 * a = x * b) (h1 : a * b ≠ 0) (h2 : (a / 5) / (b / 4) = 1) : x = 5 :=
by
  sorry

end number_that_multiplies_b_l1012_101240


namespace thrushes_left_l1012_101248

theorem thrushes_left {init_thrushes : ℕ} (additional_thrushes : ℕ) (killed_ratio : ℚ) (killed : ℕ) (remaining : ℕ) :
  init_thrushes = 20 →
  additional_thrushes = 4 * 2 →
  killed_ratio = 1 / 7 →
  killed = killed_ratio * (init_thrushes + additional_thrushes) →
  remaining = init_thrushes + additional_thrushes - killed →
  remaining = 24 :=
by sorry

end thrushes_left_l1012_101248


namespace eq_abs_distinct_solution_count_l1012_101281

theorem eq_abs_distinct_solution_count :
  ∃! x : ℝ, |x - 10| = |x + 5| + 2 := 
sorry

end eq_abs_distinct_solution_count_l1012_101281


namespace theorem_1_valid_theorem_6_valid_l1012_101251

theorem theorem_1_valid (a b : ℤ) (h1 : a % 7 = 0) (h2 : b % 7 = 0) : (a + b) % 7 = 0 :=
by sorry

theorem theorem_6_valid (a b : ℤ) (h : (a + b) % 7 ≠ 0) : a % 7 ≠ 0 ∨ b % 7 ≠ 0 :=
by sorry

end theorem_1_valid_theorem_6_valid_l1012_101251


namespace average_income_l1012_101253

theorem average_income :
  let income_day1 := 300
  let income_day2 := 150
  let income_day3 := 750
  let income_day4 := 200
  let income_day5 := 600
  (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / 5 = 400 := by
  sorry

end average_income_l1012_101253


namespace quadratic_polynomial_value_bound_l1012_101217

theorem quadratic_polynomial_value_bound (a b : ℝ) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, |(x^2 + a * x + b)| ≥ 1/2 :=
by
  sorry

end quadratic_polynomial_value_bound_l1012_101217


namespace man_speed_with_current_l1012_101290

-- Define the conditions
def current_speed : ℕ := 3
def man_speed_against_current : ℕ := 14

-- Define the man's speed in still water (v) based on the given speed against the current
def man_speed_in_still_water : ℕ := man_speed_against_current + current_speed

-- Prove that the man's speed with the current is 20 kmph
theorem man_speed_with_current : man_speed_in_still_water + current_speed = 20 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end man_speed_with_current_l1012_101290


namespace gabby_l1012_101271

-- Define variables and conditions
variables (watermelons peaches plums total_fruit : ℕ)
variables (h_watermelons : watermelons = 1)
variables (h_peaches : peaches = watermelons + 12)
variables (h_plums : plums = 3 * peaches)
variables (h_total_fruit : total_fruit = watermelons + peaches + plums)

-- The theorem we aim to prove
theorem gabby's_fruit_count (h_watermelons : watermelons = 1)
                           (h_peaches : peaches = watermelons + 12)
                           (h_plums : plums = 3 * peaches)
                           (h_total_fruit : total_fruit = watermelons + peaches + plums) :
  total_fruit = 53 := by
sorry

end gabby_l1012_101271


namespace intersection_A_B_l1012_101259

/-- Definition of set A -/
def A : Set ℕ := {1, 2, 3, 4}

/-- Definition of set B -/
def B : Set ℕ := {x | x > 2}

/-- The theorem to prove the intersection of sets A and B -/
theorem intersection_A_B : A ∩ B = {3, 4} :=
by
  sorry

end intersection_A_B_l1012_101259


namespace transformed_quadratic_l1012_101229

theorem transformed_quadratic (a b c n x : ℝ) (h : a * x^2 + b * x + c = 0) :
  a * x^2 + n * b * x + n^2 * c = 0 :=
sorry

end transformed_quadratic_l1012_101229


namespace total_number_of_items_l1012_101278

theorem total_number_of_items (total_items : ℕ) (selected_items : ℕ) (h1 : total_items = 50) (h2 : selected_items = 10) : total_items = 50 :=
by
  exact h1

end total_number_of_items_l1012_101278


namespace imaginary_condition_l1012_101283

noncomputable def is_imaginary (z : ℂ) : Prop := z.im ≠ 0

theorem imaginary_condition (z1 z2 : ℂ) :
  ( ∃ (z1 : ℂ), is_imaginary z1 ∨ is_imaginary z2 ∨ (is_imaginary (z1 - z2))) ↔
  ∃ (z1 z2 : ℂ), is_imaginary z1 ∨ is_imaginary z2 ∧ ¬ (is_imaginary (z1 - z2)) :=
sorry

end imaginary_condition_l1012_101283


namespace WidgetsPerHour_l1012_101220

theorem WidgetsPerHour 
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (widgets_per_week : ℕ) 
  (H1 : hours_per_day = 8)
  (H2 : days_per_week = 5)
  (H3 : widgets_per_week = 800) : 
  widgets_per_week / (hours_per_day * days_per_week) = 20 := 
sorry

end WidgetsPerHour_l1012_101220


namespace savings_duration_before_investment_l1012_101252

---- Definitions based on conditions ----
def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def stock_price_per_share : ℕ := 50
def shares_bought : ℕ := 25

---- Derived conditions based on definitions ----
def total_spent_on_stocks := shares_bought * stock_price_per_share
def total_savings_before_investment := 2 * total_spent_on_stocks
def monthly_savings_wife := weekly_savings_wife * 4
def total_monthly_savings := monthly_savings_wife + monthly_savings_husband

---- The theorem statement ----
theorem savings_duration_before_investment :
  total_savings_before_investment / total_monthly_savings = 4 :=
sorry

end savings_duration_before_investment_l1012_101252


namespace day_50_of_year_N_minus_1_l1012_101263

-- Definitions for the problem conditions
def day_of_week (n : ℕ) : ℕ := n % 7

-- Given that the 250th day of year N is a Friday
axiom day_250_of_year_N_is_friday : day_of_week 250 = 5

-- Given that the 150th day of year N+1 is a Friday
axiom day_150_of_year_N_plus_1_is_friday : day_of_week 150 = 5

-- Calculate the day of the week for the 50th day of year N-1
theorem day_50_of_year_N_minus_1 :
  day_of_week 50 = 4 :=
  sorry

end day_50_of_year_N_minus_1_l1012_101263


namespace kathryn_more_pints_than_annie_l1012_101295

-- Definitions for conditions
def annie_pints : ℕ := 8
def ben_pints (kathryn_pints : ℕ) : ℕ := kathryn_pints - 3
def total_pints (annie_pints kathryn_pints ben_pints : ℕ) : ℕ := annie_pints + kathryn_pints + ben_pints

-- The problem statement
theorem kathryn_more_pints_than_annie (k : ℕ) (h1 : total_pints annie_pints k (ben_pints k) = 25) : k - annie_pints = 2 :=
sorry

end kathryn_more_pints_than_annie_l1012_101295


namespace quadratic_intersection_l1012_101204

theorem quadratic_intersection
  (a b c d h : ℝ)
  (h_a : a ≠ 0)
  (h_b : b ≠ 0)
  (h_h : h ≠ 0)
  (h_d : d ≠ c) :
  ∃ x y : ℝ, (y = a * x^2 + b * x + c) ∧ (y = a * (x - h)^2 + b * (x - h) + d)
    ∧ x = (d - c) / b
    ∧ y = a * (d - c)^2 / b^2 + d :=
by {
  sorry
}

end quadratic_intersection_l1012_101204


namespace most_stable_city_l1012_101230

def variance_STD : ℝ := 12.5
def variance_A : ℝ := 18.3
def variance_B : ℝ := 17.4
def variance_C : ℝ := 20.1

theorem most_stable_city : variance_STD < variance_A ∧ variance_STD < variance_B ∧ variance_STD < variance_C :=
by {
  -- Proof skipped
  sorry
}

end most_stable_city_l1012_101230


namespace problem_statement_l1012_101288

theorem problem_statement
  (a b A B : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ θ : ℝ, f θ ≥ 0)
  (def_f : ∀ θ : ℝ, f θ = 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ)) :
  a ^ 2 + b ^ 2 ≤ 2 ∧ A ^ 2 + B ^ 2 ≤ 1 := 
by
  sorry

end problem_statement_l1012_101288


namespace intersection_of_sets_l1012_101277

def A (x : ℝ) : Prop := x > -2
def B (x : ℝ) : Prop := 1 - x > 0

theorem intersection_of_sets :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | x > -2 ∧ x < 1} := by
  sorry

end intersection_of_sets_l1012_101277


namespace boys_in_class_l1012_101203

theorem boys_in_class
  (g b : ℕ)
  (h_ratio : g = (3 * b) / 5)
  (h_total : g + b = 32) :
  b = 20 :=
sorry

end boys_in_class_l1012_101203


namespace find_value_of_2_times_x_minus_y_squared_minus_3_l1012_101216

-- Define the conditions as noncomputable variables
variables (x y : ℝ)

-- State the main theorem
theorem find_value_of_2_times_x_minus_y_squared_minus_3 :
  (x^2 - x*y = 12) →
  (y^2 - y*x = 15) →
  2 * (x - y)^2 - 3 = 51 :=
by
  intros h1 h2
  sorry

end find_value_of_2_times_x_minus_y_squared_minus_3_l1012_101216


namespace value_of_a_l1012_101238

theorem value_of_a (a : ℝ) (h : a = -a) : a = 0 :=
by
  sorry

end value_of_a_l1012_101238


namespace hardcover_books_count_l1012_101262

theorem hardcover_books_count (h p : ℕ) (h_condition : h + p = 12) (cost_condition : 30 * h + 15 * p = 270) : h = 6 :=
by
  sorry

end hardcover_books_count_l1012_101262


namespace gcf_of_48_180_120_l1012_101222

theorem gcf_of_48_180_120 : Nat.gcd (Nat.gcd 48 180) 120 = 12 := by
  sorry

end gcf_of_48_180_120_l1012_101222


namespace determine_pairs_l1012_101208

theorem determine_pairs (p q : ℕ) (h : (p + 1)^(p - 1) + (p - 1)^(p + 1) = q^q) : (p = 1 ∧ q = 1) ∨ (p = 2 ∧ q = 2) :=
by
  sorry

end determine_pairs_l1012_101208


namespace original_price_of_house_l1012_101279

theorem original_price_of_house (P : ℝ) 
  (h1 : P * 0.56 = 56000) : P = 100000 :=
sorry

end original_price_of_house_l1012_101279


namespace pete_miles_walked_l1012_101234

noncomputable def steps_from_first_pedometer (flips1 : ℕ) (final_reading1 : ℕ) : ℕ :=
  flips1 * 100000 + final_reading1 

noncomputable def steps_from_second_pedometer (flips2 : ℕ) (final_reading2 : ℕ) : ℕ :=
  flips2 * 400000 + final_reading2 * 4

noncomputable def total_steps (flips1 flips2 final_reading1 final_reading2 : ℕ) : ℕ :=
  steps_from_first_pedometer flips1 final_reading1 + steps_from_second_pedometer flips2 final_reading2

noncomputable def miles_walked (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

theorem pete_miles_walked
  (flips1 flips2 final_reading1 final_reading2 steps_per_mile : ℕ)
  (h_flips1 : flips1 = 50)
  (h_final_reading1 : final_reading1 = 25000)
  (h_flips2 : flips2 = 15)
  (h_final_reading2 : final_reading2 = 30000)
  (h_steps_per_mile : steps_per_mile = 1500) :
  miles_walked (total_steps flips1 flips2 final_reading1 final_reading2) steps_per_mile = 7430 :=
by sorry

end pete_miles_walked_l1012_101234


namespace men_entered_l1012_101257

theorem men_entered (M W x : ℕ) 
  (h1 : 5 * M = 4 * W)
  (h2 : M + x = 14)
  (h3 : 2 * (W - 3) = 24) : 
  x = 2 :=
by
  sorry

end men_entered_l1012_101257


namespace horner_eval_f_at_5_eval_f_at_5_l1012_101212

def f (x: ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem horner_eval_f_at_5 :
  f 5 = ((((5 - 2) * 5 + 1) * 5 + 1) * 5 - 1) * 5 - 5 := by
  sorry

theorem eval_f_at_5 : f 5 = 2015 := by 
  have h : f 5 = ((((5 - 2) * 5 + 1) * 5 + 1) * 5 - 1) * 5 - 5 := by
    apply horner_eval_f_at_5
  rw [h]
  norm_num

end horner_eval_f_at_5_eval_f_at_5_l1012_101212


namespace tan_identity_l1012_101250

theorem tan_identity
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 3 / 7)
  (h2 : Real.tan (β - Real.pi / 4) = -1 / 3)
  : Real.tan (α + Real.pi / 4) = 8 / 9 := by
  sorry

end tan_identity_l1012_101250


namespace line_containing_chord_l1012_101274

variable {x y x₁ y₁ x₂ y₂ : ℝ}

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 4 = 1)

def midpoint_condition (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) : Prop := 
  (x₁ + x₂ = 2) ∧ (y₁ + y₂ = 2)

theorem line_containing_chord (h₁ : ellipse_eq x₁ y₁) 
                               (h₂ : ellipse_eq x₂ y₂) 
                               (hmp : midpoint_condition x₁ x₂ y₁ y₂)
    : 4 * 1 + 9 * 1 - 13 = 0 := 
sorry

end line_containing_chord_l1012_101274


namespace river_length_l1012_101255

theorem river_length (x : ℝ) (h1 : 3 * x + x = 80) : x = 20 :=
sorry

end river_length_l1012_101255


namespace find_length_of_segment_l1012_101245

noncomputable def radius : ℝ := 4
noncomputable def volume_cylinder (L : ℝ) : ℝ := 16 * Real.pi * L
noncomputable def volume_hemispheres : ℝ := 2 * (128 / 3) * Real.pi
noncomputable def total_volume (L : ℝ) : ℝ := volume_cylinder L + volume_hemispheres

theorem find_length_of_segment (L : ℝ) (h : total_volume L = 544 * Real.pi) : 
  L = 86 / 3 :=
by sorry

end find_length_of_segment_l1012_101245


namespace least_positive_int_to_next_multiple_l1012_101269

theorem least_positive_int_to_next_multiple (x : ℕ) (n : ℕ) (h : x = 365 ∧ n > 0) 
  (hm : (x + n) % 5 = 0) : n = 5 :=
by
  sorry

end least_positive_int_to_next_multiple_l1012_101269


namespace range_of_m_l1012_101270

-- Given definitions and conditions
def sequence_a (n : ℕ) : ℕ := if n = 1 then 2 else n * 2^n

def vec_a : ℕ × ℤ := (2, -1)

def vec_b (n : ℕ) : ℕ × ℤ := (sequence_a n + 2^n, sequence_a (n + 1))

def orthogonal (v1 v2 : ℕ × ℤ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Translate the proof problem
theorem range_of_m (n : ℕ) (m : ℝ) (h1 : orthogonal vec_a (vec_b n))
  (h2 : ∀ n : ℕ, n > 0 → (sequence_a n) / (n * (n + 1)^2) > (m^2 - 3 * m) / 9) :
  -1 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l1012_101270


namespace carol_lollipops_l1012_101276

theorem carol_lollipops (total_lollipops : ℝ) (first_day_lollipops : ℝ) (delta_lollipops : ℝ) :
  total_lollipops = 150 → delta_lollipops = 5 →
  (first_day_lollipops + (first_day_lollipops + 5) + (first_day_lollipops + 10) +
  (first_day_lollipops + 15) + (first_day_lollipops + 20) + (first_day_lollipops + 25) = total_lollipops) →
  (first_day_lollipops = 12.5) →
  (first_day_lollipops + 15 = 27.5) :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end carol_lollipops_l1012_101276


namespace range_of_x_l1012_101239

-- Define the function h(a).
def h (a : ℝ) : ℝ := a^2 + 2 * a + 3

-- Define the main theorem
theorem range_of_x (a : ℝ) (x : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) : 
  x^2 + 4 * x - 2 ≤ h a → -5 ≤ x ∧ x ≤ 1 :=
sorry

end range_of_x_l1012_101239


namespace factor_x4_minus_81_l1012_101218

theorem factor_x4_minus_81 : ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intro x
  sorry

end factor_x4_minus_81_l1012_101218


namespace candy_eating_l1012_101237

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l1012_101237


namespace exists_excircle_radius_at_least_three_times_incircle_radius_l1012_101289

variable (a b c s T r ra rb rc : ℝ)
variable (ha : ra = T / (s - a))
variable (hb : rb = T / (s - b))
variable (hc : rc = T / (s - c))
variable (hincircle : r = T / s)

theorem exists_excircle_radius_at_least_three_times_incircle_radius
  (ha : ra = T / (s - a)) (hb : rb = T / (s - b)) (hc : rc = T / (s - c)) (hincircle : r = T / s) :
  ∃ rc, rc ≥ 3 * r :=
by {
  use rc,
  sorry
}

end exists_excircle_radius_at_least_three_times_incircle_radius_l1012_101289


namespace robin_gum_packages_l1012_101265

theorem robin_gum_packages (P : ℕ) (h1 : 7 * P + 6 = 41) : P = 5 :=
by
  sorry

end robin_gum_packages_l1012_101265


namespace John_break_time_l1012_101268

-- Define the constants
def John_dancing_hours : ℕ := 8

-- Define the condition for James's dancing time 
def James_dancing_time (B : ℕ) : ℕ := 
  let total_time := John_dancing_hours + B
  total_time + total_time / 3

-- State the problem as a theorem
theorem John_break_time (B : ℕ) : John_dancing_hours + James_dancing_time B = 20 → B = 1 := 
  by sorry

end John_break_time_l1012_101268


namespace coordinates_of_P_tangent_line_equation_l1012_101266

-- Define point P and center of the circle
def point_P : ℝ × ℝ := (-2, 1)
def center_C : ℝ × ℝ := (-1, 0)

-- Define the circle equation (x + 1)^2 + y^2 = 2
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the tangent line at point P
def tangent_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Prove the coordinates of point P are (-2, 1) given the conditions
theorem coordinates_of_P (n : ℝ) (h1 : n > 0) (h2 : circle_equation (-2) n) :
  point_P = (-2, 1) :=
by
  -- Proof steps would go here
  sorry

-- Prove the equation of the tangent line to the circle C passing through point P is x - y + 3 = 0
theorem tangent_line_equation :
  tangent_line (-2) 1 :=
by
  -- Proof steps would go here
  sorry

end coordinates_of_P_tangent_line_equation_l1012_101266


namespace average_income_correct_l1012_101224

-- Define the incomes for each day
def income_day_1 : ℕ := 300
def income_day_2 : ℕ := 150
def income_day_3 : ℕ := 750
def income_day_4 : ℕ := 400
def income_day_5 : ℕ := 500

-- Define the number of days
def number_of_days : ℕ := 5

-- Define the total income
def total_income : ℕ := income_day_1 + income_day_2 + income_day_3 + income_day_4 + income_day_5

-- Define the average income
def average_income : ℕ := total_income / number_of_days

-- State that the average income is 420
theorem average_income_correct :
  average_income = 420 := by
  sorry

end average_income_correct_l1012_101224


namespace scientific_notation_6500_l1012_101287

theorem scientific_notation_6500 : (6500 : ℝ) = 6.5 * 10^3 := 
by 
  sorry

end scientific_notation_6500_l1012_101287


namespace total_amount_l1012_101231

theorem total_amount (x y z : ℝ) 
  (hx : y = 0.45 * x) 
  (hz : z = 0.50 * x) 
  (hy_share : y = 63) : 
  x + y + z = 273 :=
by 
  sorry

end total_amount_l1012_101231


namespace train_speed_in_kmh_l1012_101228

def length_of_train : ℝ := 156
def length_of_bridge : ℝ := 219.03
def time_to_cross_bridge : ℝ := 30
def speed_of_train_kmh : ℝ := 45.0036

theorem train_speed_in_kmh :
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = speed_of_train_kmh :=
by
  sorry

end train_speed_in_kmh_l1012_101228


namespace shooting_accuracy_l1012_101249

theorem shooting_accuracy (S : ℕ → ℕ) (H1 : ∀ n, S n < 10 * n / 9) (H2 : ∀ n, S n > 10 * n / 9) :
  ∃ n, 10 * (S n) = 9 * n :=
by
  sorry

end shooting_accuracy_l1012_101249


namespace smallest_m_inequality_l1012_101200

theorem smallest_m_inequality (a b c : ℝ) (h1 : a + b + c = 1) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : 
  27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
sorry

end smallest_m_inequality_l1012_101200


namespace marble_ratio_l1012_101242

-- Definitions based on conditions
def dan_marbles : ℕ := 5
def mary_marbles : ℕ := 10

-- Statement of the theorem to prove the ratio is 2:1
theorem marble_ratio : mary_marbles / dan_marbles = 2 := by
  sorry

end marble_ratio_l1012_101242


namespace Lyka_saves_for_8_weeks_l1012_101247

theorem Lyka_saves_for_8_weeks : 
  ∀ (C I W : ℕ), C = 160 → I = 40 → W = 15 → (C - I) / W = 8 := 
by 
  intros C I W hC hI hW
  sorry

end Lyka_saves_for_8_weeks_l1012_101247


namespace number_of_1989_periodic_points_l1012_101260

noncomputable def f (z : ℂ) (m : ℕ) : ℂ := z ^ m

noncomputable def is_periodic_point (z : ℂ) (f : ℂ → ℂ) (n : ℕ) : Prop :=
f^[n] z = z ∧ ∀ k : ℕ, k < n → (f^[k] z) ≠ z

noncomputable def count_periodic_points (m n : ℕ) : ℕ :=
m^n - m^(n / 3) - m^(n / 13) - m^(n / 17) + m^(n / 39) + m^(n / 51) + m^(n / 117) - m^(n / 153)

theorem number_of_1989_periodic_points (m : ℕ) (hm : 1 < m) :
  count_periodic_points m 1989 = m^1989 - m^663 - m^153 - m^117 + m^51 + m^39 + m^9 - m^3 :=
sorry

end number_of_1989_periodic_points_l1012_101260


namespace stephen_hawking_philosophical_implications_l1012_101254

/-- Stephen Hawking's statements -/
def stephen_hawking_statement_1 := "The universe was not created by God"
def stephen_hawking_statement_2 := "Modern science can explain the origin of the universe"

/-- Definitions implied by Hawking's statements -/
def unity_of_world_lies_in_materiality := "The unity of the world lies in its materiality"
def thought_and_existence_identical := "Thought and existence are identical"

/-- Combined implication of Stephen Hawking's statements -/
def correct_philosophical_implications := [unity_of_world_lies_in_materiality, thought_and_existence_identical]

/-- Theorem: The correct philosophical implications of Stephen Hawking's statements are ① and ②. -/
theorem stephen_hawking_philosophical_implications :
  (stephen_hawking_statement_1 = "The universe was not created by God") →
  (stephen_hawking_statement_2 = "Modern science can explain the origin of the universe") →
  correct_philosophical_implications = ["The unity of the world lies in its materiality", "Thought and existence are identical"] :=
by
  sorry

end stephen_hawking_philosophical_implications_l1012_101254


namespace tan_20_plus_4_sin_20_eq_sqrt_3_l1012_101227

theorem tan_20_plus_4_sin_20_eq_sqrt_3 : Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end tan_20_plus_4_sin_20_eq_sqrt_3_l1012_101227


namespace roots_product_of_polynomials_l1012_101213

theorem roots_product_of_polynomials :
  ∃ (b c : ℤ), (∀ r : ℂ, r ^ 2 - 2 * r - 1 = 0 → r ^ 5 - b * r - c = 0) ∧ b * c = 348 :=
by 
  sorry

end roots_product_of_polynomials_l1012_101213


namespace seventeen_number_selection_l1012_101261

theorem seventeen_number_selection : ∃ (n : ℕ), (∀ s : Finset ℕ, (s ⊆ Finset.range 17) → (Finset.card s = n) → ∃ x y : ℕ, (x ∈ s) ∧ (y ∈ s) ∧ (x ≠ y) ∧ (x = 3 * y ∨ y = 3 * x)) ∧ (n = 13) :=
by
  sorry

end seventeen_number_selection_l1012_101261


namespace quadratic_root_sum_product_l1012_101272

theorem quadratic_root_sum_product (m n : ℝ)
  (h1 : m + n = 4)
  (h2 : m * n = -1) :
  m + n - m * n = 5 :=
sorry

end quadratic_root_sum_product_l1012_101272


namespace shaded_area_percentage_l1012_101285

def area_square (side : ℕ) : ℕ := side * side

def shaded_percentage (total_area shaded_area : ℕ) : ℚ :=
  ((shaded_area : ℚ) / total_area) * 100 

theorem shaded_area_percentage (side : ℕ) (total_area : ℕ) (shaded_area : ℕ) 
  (h_side : side = 7) (h_total_area : total_area = area_square side) 
  (h_shaded_area : shaded_area = 4 + 16 + 13) : 
  shaded_percentage total_area shaded_area = 3300 / 49 :=
by
  -- The proof will go here
  sorry

end shaded_area_percentage_l1012_101285


namespace intersection_A_B_l1012_101297

-- Define sets A and B
def A : Set ℤ := {1, 3, 5}
def B : Set ℤ := {-1, 0, 1}

-- Prove that the intersection of A and B is {1}
theorem intersection_A_B : A ∩ B = {1} := by 
  sorry

end intersection_A_B_l1012_101297


namespace product_or_double_is_perfect_square_l1012_101202

variable {a b c : ℤ}

-- Conditions
def sides_of_triangle (a b c : ℤ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def no_common_divisor (a b c : ℤ) : Prop := gcd (gcd a b) c = 1

def all_fractions_are_integers (a b c : ℤ) : Prop :=
  (a + b - c) ≠ 0 ∧ (b + c - a) ≠ 0 ∧ (c + a - b) ≠ 0 ∧
  ((a^2 + b^2 - c^2) % (a + b - c) = 0) ∧ 
  ((b^2 + c^2 - a^2) % (b + c - a) = 0) ∧ 
  ((c^2 + a^2 - b^2) % (c + a - b) = 0)

-- Mathematical proof problem statement in Lean 4
theorem product_or_double_is_perfect_square (a b c : ℤ) 
  (h1 : sides_of_triangle a b c)
  (h2 : no_common_divisor a b c)
  (h3 : all_fractions_are_integers a b c) :
  ∃ k : ℤ, k^2 = (a + b - c) * (b + c - a) * (c + a - b) ∨ 
           k^2 = 2 * (a + b - c) * (b + c - a) * (c + a - b) := sorry

end product_or_double_is_perfect_square_l1012_101202


namespace sum_of_ages_l1012_101264

variable (S T : ℕ)

theorem sum_of_ages (h1 : S = T + 7) (h2 : S + 10 = 3 * (T - 3)) : S + T = 33 := by
  sorry

end sum_of_ages_l1012_101264


namespace current_average_is_35_l1012_101292

noncomputable def cricket_avg (A : ℝ) : Prop :=
  let innings := 10
  let next_runs := 79
  let increase := 4
  (innings * A + next_runs = (A + increase) * (innings + 1))

theorem current_average_is_35 : cricket_avg 35 :=
by
  unfold cricket_avg
  simp only
  sorry

end current_average_is_35_l1012_101292


namespace simplify_exponent_product_l1012_101232

theorem simplify_exponent_product :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_product_l1012_101232


namespace sum_of_three_integers_l1012_101223

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 5^3) : a + b + c = 31 := by
  sorry

end sum_of_three_integers_l1012_101223


namespace smallest_sum_97_l1012_101206

theorem smallest_sum_97 (X Y Z W : ℕ) 
  (h1 : X + Y + Z = 3)
  (h2 : 4 * Z = 7 * Y)
  (h3 : 16 ∣ Y) : 
  X + Y + Z + W = 97 :=
by
  sorry

end smallest_sum_97_l1012_101206


namespace remainder_when_divided_by_198_l1012_101241

-- Define the conditions as Hypotheses
variables (x : ℤ)

-- Hypotheses stating the given conditions
def cond1 : Prop := 2 + x ≡ 9 [ZMOD 8]
def cond2 : Prop := 3 + x ≡ 4 [ZMOD 27]
def cond3 : Prop := 11 + x ≡ 49 [ZMOD 1331]

-- Final statement to prove
theorem remainder_when_divided_by_198 (h1 : cond1 x) (h2 : cond2 x) (h3 : cond3 x) : x ≡ 1 [ZMOD 198] := by
  sorry

end remainder_when_divided_by_198_l1012_101241
