import Mathlib

namespace cost_price_of_one_ball_is_48_l523_52328

-- Define the cost price of one ball
def costPricePerBall (x : ℝ) : Prop :=
  let totalCostPrice20Balls := 20 * x
  let sellingPrice20Balls := 720
  let loss := 5 * x
  totalCostPrice20Balls = sellingPrice20Balls + loss

-- Define the main proof problem
theorem cost_price_of_one_ball_is_48 (x : ℝ) (h : costPricePerBall x) : x = 48 :=
by
  sorry

end cost_price_of_one_ball_is_48_l523_52328


namespace continuous_at_3_l523_52315

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 3 then x^2 + x + 2 else 2 * x + a

theorem continuous_at_3 {a : ℝ} : (∀ x : ℝ, 0 < abs (x - 3) → abs (f x a - f 3 a) < 0.0001) →
a = 8 :=
by
  sorry

end continuous_at_3_l523_52315


namespace BrotherUpperLimit_l523_52351

variable (w : ℝ) -- Arun's weight
variable (b : ℝ) -- Upper limit of Arun's weight according to his brother's opinion

-- Conditions as per the problem
def ArunOpinion (w : ℝ) := 64 < w ∧ w < 72
def BrotherOpinion (w b : ℝ) := 60 < w ∧ w < b
def MotherOpinion (w : ℝ) := w ≤ 67

-- The average of probable weights
def AverageWeight (weights : Set ℝ) (avg : ℝ) := (∀ w ∈ weights, 64 < w ∧ w ≤ 67) ∧ avg = 66

-- The main theorem to be proven
theorem BrotherUpperLimit (hA : ArunOpinion w) (hB : BrotherOpinion w b) (hM : MotherOpinion w) (hAvg : AverageWeight {w | 64 < w ∧ w ≤ 67} 66) : b = 67 := by
  sorry

end BrotherUpperLimit_l523_52351


namespace Cindy_hourly_rate_l523_52307

theorem Cindy_hourly_rate
    (num_courses : ℕ)
    (weekly_hours : ℕ) 
    (monthly_earnings : ℕ) 
    (weeks_in_month : ℕ)
    (monthly_hours_per_course : ℕ)
    (hourly_rate : ℕ) :
    num_courses = 4 →
    weekly_hours = 48 →
    monthly_earnings = 1200 →
    weeks_in_month = 4 →
    monthly_hours_per_course = (weekly_hours / num_courses) * weeks_in_month →
    hourly_rate = monthly_earnings / monthly_hours_per_course →
    hourly_rate = 25 := by
  sorry

end Cindy_hourly_rate_l523_52307


namespace num_ordered_pairs_squares_diff_30_l523_52384

theorem num_ordered_pairs_squares_diff_30 :
  ∃ (n : ℕ), n = 0 ∧
  ∀ (m n: ℕ), 0 < m ∧ 0 < n ∧ m ≥ n ∧ m^2 - n^2 = 30 → false :=
by
  sorry

end num_ordered_pairs_squares_diff_30_l523_52384


namespace percentage_very_satisfactory_l523_52395

-- Definitions based on conditions
def total_parents : ℕ := 120
def needs_improvement_count : ℕ := 6
def excellent_percentage : ℕ := 15
def satisfactory_remaining_percentage : ℕ := 80

-- Theorem statement
theorem percentage_very_satisfactory 
  (total_parents : ℕ) 
  (needs_improvement_count : ℕ) 
  (excellent_percentage : ℕ) 
  (satisfactory_remaining_percentage : ℕ) 
  (result : ℕ) : result = 16 :=
by
  sorry

end percentage_very_satisfactory_l523_52395


namespace sequence_8123_appears_l523_52303

theorem sequence_8123_appears :
  ∃ (a : ℕ → ℕ), (∀ n ≥ 5, a n = (a (n-1) + a (n-2) + a (n-3) + a (n-4)) % 10) ∧
  (a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 3 ∧ a 4 = 4) ∧
  (∃ n, a n = 8 ∧ a (n+1) = 1 ∧ a (n+2) = 2 ∧ a (n+3) = 3) :=
sorry

end sequence_8123_appears_l523_52303


namespace sqrt_expression_equality_l523_52378

theorem sqrt_expression_equality :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 :=
by
  sorry

end sqrt_expression_equality_l523_52378


namespace cost_of_bananas_and_cantaloupe_l523_52314

-- Let a, b, c, and d be real numbers representing the prices of apples, bananas, cantaloupe, and dates respectively.
variables (a b c d : ℝ)

-- Conditions given in the problem
axiom h1 : a + b + c + d = 40
axiom h2 : d = 3 * a
axiom h3 : c = (a + b) / 2

-- Goal is to prove that the sum of the prices of bananas and cantaloupe is 8 dollars.
theorem cost_of_bananas_and_cantaloupe : b + c = 8 :=
by
  sorry

end cost_of_bananas_and_cantaloupe_l523_52314


namespace ratio_a_b_l523_52353

-- Definitions of the arithmetic sequences
open Classical

noncomputable def sequence1 (a y b : ℕ) : ℕ → ℕ
| 0 => a
| 1 => y
| 2 => b
| 3 => 14
| _ => 0 -- only the first four terms are given for sequence1

noncomputable def sequence2 (x y : ℕ) : ℕ → ℕ
| 0 => 2
| 1 => x
| 2 => 6
| 3 => y
| _ => 0 -- only the first four terms are given for sequence2

theorem ratio_a_b (a y b x : ℕ) (h1 : sequence1 a y b 0 = a) (h2 : sequence1 a y b 1 = y) 
  (h3 : sequence1 a y b 2 = b) (h4 : sequence1 a y b 3 = 14)
  (h5 : sequence2 x y 0 = 2) (h6 : sequence2 x y 1 = x) 
  (h7 : sequence2 x y 2 = 6) (h8 : sequence2 x y 3 = y) :
  (a:ℚ) / b = 2 / 3 :=
sorry

end ratio_a_b_l523_52353


namespace fraction_value_l523_52317

theorem fraction_value :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 :=
by
  sorry

end fraction_value_l523_52317


namespace probability_heads_at_least_10_in_12_flips_l523_52399

theorem probability_heads_at_least_10_in_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 79 / 4096 := by
  sorry

end probability_heads_at_least_10_in_12_flips_l523_52399


namespace math_problem_l523_52357

variable {p q r x y : ℝ}

theorem math_problem (h1 : p / q = 6 / 7)
                     (h2 : p / r = 8 / 9)
                     (h3 : q / r = x / y) :
                     x = 28 ∧ y = 27 ∧ 2 * p + q = (19 / 6) * p := 
by 
  sorry

end math_problem_l523_52357


namespace gcd_24_36_l523_52313

theorem gcd_24_36 : Int.gcd 24 36 = 12 := by
  sorry

end gcd_24_36_l523_52313


namespace cost_effective_plan1_l523_52347

/-- 
Plan 1 involves purchasing a 80 yuan card and a subsequent fee of 10 yuan per session.
Plan 2 involves a fee of 20 yuan per session without purchasing the card.
We want to prove that Plan 1 is more cost-effective than Plan 2 for any number of sessions x > 8.
-/
theorem cost_effective_plan1 (x : ℕ) (h : x > 8) : 
  10 * x + 80 < 20 * x :=
sorry

end cost_effective_plan1_l523_52347


namespace fewest_tiles_to_cover_region_l523_52373

namespace TileCoverage

def tile_width : ℕ := 2
def tile_length : ℕ := 6
def region_width_feet : ℕ := 3
def region_length_feet : ℕ := 4

def region_width_inches : ℕ := region_width_feet * 12
def region_length_inches : ℕ := region_length_feet * 12

def region_area : ℕ := region_width_inches * region_length_inches
def tile_area : ℕ := tile_width * tile_length

def fewest_tiles_needed : ℕ := region_area / tile_area

theorem fewest_tiles_to_cover_region :
  fewest_tiles_needed = 144 :=
sorry

end TileCoverage

end fewest_tiles_to_cover_region_l523_52373


namespace negation_of_proposition_l523_52321

-- Definitions from the problem conditions
def proposition (x : ℝ) := ∃ x < 1, x^2 ≤ 1

-- Reformulated proof problem
theorem negation_of_proposition : 
  ¬ (∃ x < 1, x^2 ≤ 1) ↔ ∀ x < 1, x^2 > 1 :=
by
  sorry

end negation_of_proposition_l523_52321


namespace determine_range_of_x_l523_52358

theorem determine_range_of_x (x : ℝ) (h₁ : 1/x < 3) (h₂ : 1/x > -2) : x > 1/3 ∨ x < -1/2 :=
sorry

end determine_range_of_x_l523_52358


namespace net_increase_correct_l523_52311

-- Definitions for the given conditions
def S1 : ℕ := 10
def B1 : ℕ := 15
def S2 : ℕ := 12
def B2 : ℕ := 8
def S3 : ℕ := 9
def B3 : ℕ := 11

def P1 : ℕ := 250
def P2 : ℕ := 275
def P3 : ℕ := 260
def C1 : ℕ := 100
def C2 : ℕ := 110
def C3 : ℕ := 120

def Sale_profit1 : ℕ := S1 * P1
def Sale_profit2 : ℕ := S2 * P2
def Sale_profit3 : ℕ := S3 * P3

def Repair_cost1 : ℕ := B1 * C1
def Repair_cost2 : ℕ := B2 * C2
def Repair_cost3 : ℕ := B3 * C3

def Net_profit1 : ℕ := Sale_profit1 - Repair_cost1
def Net_profit2 : ℕ := Sale_profit2 - Repair_cost2
def Net_profit3 : ℕ := Sale_profit3 - Repair_cost3

def Total_net_profit : ℕ := Net_profit1 + Net_profit2 + Net_profit3

def Net_Increase : ℕ := (B1 - S1) + (B2 - S2) + (B3 - S3)

-- The theorem to be proven
theorem net_increase_correct : Net_Increase = 3 := by
  sorry

end net_increase_correct_l523_52311


namespace problem1_problem2_problem3_l523_52369

-- First problem
theorem problem1 : 24 - |(-2)| + (-16) - 8 = -2 := by
  sorry

-- Second problem
theorem problem2 : (-2) * (3 / 2) / (-3 / 4) * 4 = 4 := by
  sorry

-- Third problem
theorem problem3 : -1^2016 - (1 - 0.5) / 3 * (2 - (-3)^2) = 1 / 6 := by
  sorry

end problem1_problem2_problem3_l523_52369


namespace probability_of_consonant_initials_l523_52370

def number_of_students : Nat := 30
def alphabet_size : Nat := 26
def redefined_vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y'}
def number_of_vowels : Nat := redefined_vowels.card
def number_of_consonants : Nat := alphabet_size - number_of_vowels

theorem probability_of_consonant_initials :
  (number_of_consonants : ℝ) / (number_of_students : ℝ) = 2/3 := 
by
  -- Proof goes here
  sorry

end probability_of_consonant_initials_l523_52370


namespace third_discount_l523_52323

noncomputable def find_discount (P S firstDiscount secondDiscount D3 : ℝ) : Prop :=
  S = P * (1 - firstDiscount / 100) * (1 - secondDiscount / 100) * (1 - D3 / 100)

theorem third_discount (P : ℝ) (S : ℝ) (firstDiscount : ℝ) (secondDiscount : ℝ) (D3 : ℝ) 
  (HP : P = 9649.12) (HS : S = 6600)
  (HfirstDiscount : firstDiscount = 20) (HsecondDiscount : secondDiscount = 10) : 
  find_discount P S firstDiscount secondDiscount 5.01 :=
  by
  rw [HP, HS, HfirstDiscount, HsecondDiscount]
  sorry

end third_discount_l523_52323


namespace prove_identity_l523_52327

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end prove_identity_l523_52327


namespace intersection_points_l523_52388

noncomputable def hyperbola : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 / 9 - y^2 = 1 }

noncomputable def line : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ y = (1 / 3) * (x + 1) }

theorem intersection_points :
  ∃! (p : ℝ × ℝ), p ∈ hyperbola ∧ p ∈ line :=
sorry

end intersection_points_l523_52388


namespace Mary_forgot_pigs_l523_52389

theorem Mary_forgot_pigs (Mary_thinks : ℕ) (actual_animals : ℕ) (double_counted_sheep : ℕ)
  (H_thinks : Mary_thinks = 60) (H_actual : actual_animals = 56)
  (H_double_counted : double_counted_sheep = 7) :
  ∃ pigs_forgot : ℕ, pigs_forgot = 3 :=
by
  let counted_animals := Mary_thinks - double_counted_sheep
  have H_counted_correct : counted_animals = 53 := by sorry -- 60 - 7 = 53
  have pigs_forgot := actual_animals - counted_animals
  have H_pigs_forgot : pigs_forgot = 3 := by sorry -- 56 - 53 = 3
  exact ⟨pigs_forgot, H_pigs_forgot⟩

end Mary_forgot_pigs_l523_52389


namespace equivalent_proof_problem_l523_52367

theorem equivalent_proof_problem (x : ℤ) (h : (x + 2) * (x - 2) = 1221) :
    (x = 35 ∨ x = -35) ∧ ((x + 1) * (x - 1) = 1224) :=
sorry

end equivalent_proof_problem_l523_52367


namespace percent_of_flowers_are_daisies_l523_52329

-- Definitions for the problem
def total_flowers (F : ℕ) := F
def blue_flowers (F : ℕ) := (7/10) * F
def red_flowers (F : ℕ) := (3/10) * F
def blue_tulips (F : ℕ) := (1/2) * (7/10) * F
def blue_daisies (F : ℕ) := (7/10) * F - (1/2) * (7/10) * F
def red_daisies (F : ℕ) := (2/3) * (3/10) * F
def total_daisies (F : ℕ) := blue_daisies F + red_daisies F
def percentage_of_daisies (F : ℕ) := (total_daisies F / F) * 100

-- The statement to prove
theorem percent_of_flowers_are_daisies (F : ℕ) (hF : F > 0) :
  percentage_of_daisies F = 55 := by
  sorry

end percent_of_flowers_are_daisies_l523_52329


namespace checkerboard_probability_l523_52304

def checkerboard_size : ℕ := 10

def total_squares (n : ℕ) : ℕ := n * n

def perimeter_squares (n : ℕ) : ℕ := 4 * n - 4

def inner_squares (n : ℕ) : ℕ := total_squares n - perimeter_squares n

def probability_not_touching_edge (n : ℕ) : ℚ := inner_squares n / total_squares n

theorem checkerboard_probability :
  probability_not_touching_edge checkerboard_size = 16 / 25 := by
  sorry

end checkerboard_probability_l523_52304


namespace solve_eq_l523_52398

theorem solve_eq (x : ℝ) : x^6 - 19*x^3 = 216 ↔ (x = 3 ∨ x = -2) :=
by
  sorry

end solve_eq_l523_52398


namespace quadratic_form_ratio_l523_52380

theorem quadratic_form_ratio (x y u v : ℤ) (h : ∃ k : ℤ, k * (u^2 + 3*v^2) = x^2 + 3*y^2) :
  ∃ a b : ℤ, (x^2 + 3*y^2) / (u^2 + 3*v^2) = a^2 + 3*b^2 := sorry

end quadratic_form_ratio_l523_52380


namespace find_k_l523_52349

-- The expression in terms of x, y, and k
def expression (k x y : ℝ) :=
  4 * x^2 - 6 * k * x * y + (3 * k^2 + 2) * y^2 - 4 * x - 4 * y + 6

-- The mathematical statement to be proved
theorem find_k : ∃ k : ℝ, (∀ x y : ℝ, expression k x y ≥ 0) ∧ (∃ (x y : ℝ), expression k x y = 0) :=
sorry

end find_k_l523_52349


namespace find_x_squared_plus_y_squared_find_xy_l523_52363

variable {x y : ℝ}

theorem find_x_squared_plus_y_squared (h1 : (x - y)^2 = 4) (h2 : (x + y)^2 = 64) : x^2 + y^2 = 34 :=
sorry

theorem find_xy (h1 : (x - y)^2 = 4) (h2 : (x + y)^2 = 64) : x * y = 15 :=
sorry

end find_x_squared_plus_y_squared_find_xy_l523_52363


namespace fraction_of_science_liking_students_l523_52361

open Real

theorem fraction_of_science_liking_students (total_students math_fraction english_fraction no_fav_students math_students english_students fav_students remaining_students science_students fraction_science) :
  total_students = 30 ∧
  math_fraction = 1/5 ∧
  english_fraction = 1/3 ∧
  no_fav_students = 12 ∧
  math_students = total_students * math_fraction ∧
  english_students = total_students * english_fraction ∧
  fav_students = total_students - no_fav_students ∧
  remaining_students = fav_students - (math_students + english_students) ∧
  science_students = remaining_students ∧
  fraction_science = science_students / remaining_students →
  fraction_science = 1 :=
by
  sorry

end fraction_of_science_liking_students_l523_52361


namespace power_of_x_is_one_l523_52334

-- The problem setup, defining the existence of distinct primes and conditions on exponents
theorem power_of_x_is_one (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (a b c : ℕ) (h_divisors : (a + 1) * (b + 1) * (c + 1) = 12) :
  a = 1 :=
sorry

end power_of_x_is_one_l523_52334


namespace change_in_total_berries_l523_52326

-- Define the initial conditions
def blue_box_berries : ℕ := 35
def increase_diff : ℕ := 100

-- Define the number of strawberries in red boxes
def red_box_berries : ℕ := 100

-- Formulate the change in total number of berries
theorem change_in_total_berries :
  (red_box_berries - blue_box_berries) = 65 :=
by
  have h1 : red_box_berries = increase_diff := rfl
  have h2 : blue_box_berries = 35 := rfl
  rw [h1, h2]
  exact rfl

end change_in_total_berries_l523_52326


namespace count_four_digit_integers_with_conditions_l523_52336

def is_four_digit_integer (n : Nat) : Prop := 1000 ≤ n ∧ n < 10000

def thousands_digit_is_seven (n : Nat) : Prop := 
  (n / 1000) % 10 = 7

def hundreds_digit_is_odd (n : Nat) : Prop := 
  let hd := (n / 100) % 10
  hd % 2 = 1

theorem count_four_digit_integers_with_conditions : 
  (Nat.card {n : Nat // is_four_digit_integer n ∧ thousands_digit_is_seven n ∧ hundreds_digit_is_odd n}) = 500 :=
by
  sorry

end count_four_digit_integers_with_conditions_l523_52336


namespace no_solution_l523_52391

theorem no_solution (x : ℝ) : ¬ (3 * x - 2 < (x + 2)^2 ∧ (x + 2)^2 < 9 * x - 5) :=
by
  sorry

end no_solution_l523_52391


namespace lily_lemonade_calories_l523_52316

def total_weight (lemonade_lime_juice lemonade_honey lemonade_water : ℕ) : ℕ :=
  lemonade_lime_juice + lemonade_honey + lemonade_water

def total_calories (weight_lime_juice weight_honey : ℕ) : ℚ :=
  (30 * weight_lime_juice / 100) + (305 * weight_honey / 100)

def calories_in_portion (total_weight total_calories portion_weight : ℚ) : ℚ :=
  (total_calories * portion_weight) / total_weight

theorem lily_lemonade_calories :
  let lemonade_lime_juice := 150
  let lemonade_honey := 150
  let lemonade_water := 450
  let portion_weight := 300
  let total_weight := total_weight lemonade_lime_juice lemonade_honey lemonade_water
  let total_calories := total_calories lemonade_lime_juice lemonade_honey
  calories_in_portion total_weight total_calories portion_weight = 201 := 
by
  sorry

end lily_lemonade_calories_l523_52316


namespace ellipse_properties_l523_52383

theorem ellipse_properties (h k a b : ℝ) (θ : ℝ)
  (h_def : h = -2)
  (k_def : k = 3)
  (a_def : a = 6)
  (b_def : b = 4)
  (θ_def : θ = 45) :
  h + k + a + b = 11 :=
by
  sorry

end ellipse_properties_l523_52383


namespace john_weekly_earnings_l523_52318

theorem john_weekly_earnings :
  (4 * 4 * 10 = 160) :=
by
  -- Proposition: John makes $160 a week from streaming
  -- Condition 1: John streams for 4 days a week
  let days_of_streaming := 4
  -- Condition 2: He streams 4 hours each day.
  let hours_per_day := 4
  -- Condition 3: He makes $10 an hour.
  let earnings_per_hour := 10

  -- Now, calculate the weekly earnings
  -- Weekly earnings = 4 days/week * 4 hours/day * $10/hour
  have weekly_earnings : days_of_streaming * hours_per_day * earnings_per_hour = 160 := sorry
  exact weekly_earnings


end john_weekly_earnings_l523_52318


namespace count_linear_eqs_l523_52359

-- Define each equation as conditions
def eq1 (x y : ℝ) := 3 * x - y = 2
def eq2 (x : ℝ) := x + 1 / x + 2 = 0
def eq3 (x : ℝ) := x^2 - 2 * x - 3 = 0
def eq4 (x : ℝ) := x = 0
def eq5 (x : ℝ) := 3 * x - 1 ≥ 5
def eq6 (x : ℝ) := 1 / 2 * x = 1 / 2
def eq7 (x : ℝ) := (2 * x + 1) / 3 = 1 / 6 * x

-- Proof statement: there are exactly 3 linear equations
theorem count_linear_eqs : 
  (∃ x y, eq1 x y) ∧ eq4 0 ∧ (∃ x, eq6 x) ∧ (∃ x, eq7 x) ∧ 
  ¬ (∃ x, eq2 x) ∧ ¬ (∃ x, eq3 x) ∧ ¬ (∃ x, eq5 x) → 
  3 = 3 :=
sorry

end count_linear_eqs_l523_52359


namespace evaluate_expression_l523_52305

theorem evaluate_expression : 2 + (3 / (4 + (5 / 6))) = 76 / 29 := 
by
  sorry

end evaluate_expression_l523_52305


namespace min_value_arith_seq_l523_52309

noncomputable def S_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem min_value_arith_seq : ∀ n : ℕ, n > 0 → 2 * S_n 2 = (n + 1) * 2 → (n = 4 → (2 * S_n n + 13) / n = 33 / 4) :=
by
  intros n hn hS2 hn_eq_4
  sorry

end min_value_arith_seq_l523_52309


namespace abs_inequality_solution_l523_52392

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ -9 / 2 < x ∧ x < 7 / 2 :=
by
  sorry

end abs_inequality_solution_l523_52392


namespace pages_read_on_fourth_day_l523_52374

-- condition: Hallie reads the whole book in 4 days, read specific pages each day
variable (total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages : ℕ)

-- Given conditions
def conditions : Prop :=
  first_day_pages = 63 ∧
  second_day_pages = 2 * first_day_pages ∧
  third_day_pages = second_day_pages + 10 ∧
  total_pages = 354 ∧
  first_day_pages + second_day_pages + third_day_pages + fourth_day_pages = total_pages

-- Prove Hallie read 29 pages on the fourth day
theorem pages_read_on_fourth_day (h : conditions total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages) :
  fourth_day_pages = 29 := sorry

end pages_read_on_fourth_day_l523_52374


namespace even_m_n_l523_52394

variable {m n : ℕ}

theorem even_m_n
  (h_m : ∃ k : ℕ, m = 2 * k + 1)
  (h_n : ∃ k : ℕ, n = 2 * k + 1) :
  Even ((m - n) ^ 2) ∧ Even ((m - n - 4) ^ 2) ∧ Even (2 * m * n + 4) :=
by
  sorry

end even_m_n_l523_52394


namespace neg_prop_l523_52393

theorem neg_prop : ∃ (a : ℝ), ∀ (x : ℝ), (a * x^2 - 3 * x + 2 = 0) → x ≤ 0 :=
sorry

end neg_prop_l523_52393


namespace subtraction_result_l523_52376

theorem subtraction_result : 3.05 - 5.678 = -2.628 := 
by
  sorry

end subtraction_result_l523_52376


namespace class_B_more_uniform_l523_52360

def x_A : ℝ := 80
def x_B : ℝ := 80
def S2_A : ℝ := 240
def S2_B : ℝ := 180

theorem class_B_more_uniform (h1 : x_A = 80) (h2 : x_B = 80) (h3 : S2_A = 240) (h4 : S2_B = 180) : 
  S2_B < S2_A :=
by {
  exact sorry
}

end class_B_more_uniform_l523_52360


namespace dot_product_v_w_l523_52332

def v : ℝ × ℝ := (-5, 3)
def w : ℝ × ℝ := (7, -9)

theorem dot_product_v_w : v.1 * w.1 + v.2 * w.2 = -62 := 
  by sorry

end dot_product_v_w_l523_52332


namespace scientific_notation_of_9280000000_l523_52310

theorem scientific_notation_of_9280000000 :
  9280000000 = 9.28 * 10^9 :=
by
  sorry

end scientific_notation_of_9280000000_l523_52310


namespace adult_ticket_cost_l523_52386

/--
Tickets at a local theater cost a certain amount for adults and 2 dollars for kids under twelve.
Given that 175 tickets were sold and the profit was 750 dollars, and 75 kid tickets were sold,
prove that an adult ticket costs 6 dollars.
-/
theorem adult_ticket_cost
  (kid_ticket_price : ℕ := 2)
  (kid_tickets_sold : ℕ := 75)
  (total_tickets_sold : ℕ := 175)
  (total_profit : ℕ := 750)
  (adult_tickets_sold : ℕ := total_tickets_sold - kid_tickets_sold)
  (adult_ticket_revenue : ℕ := total_profit - kid_ticket_price * kid_tickets_sold)
  (adult_ticket_cost : ℕ := adult_ticket_revenue / adult_tickets_sold) :
  adult_ticket_cost = 6 :=
by
  sorry

end adult_ticket_cost_l523_52386


namespace slope_of_tangent_line_l523_52354

theorem slope_of_tangent_line (f : ℝ → ℝ) (f_deriv : ∀ x, deriv f x = f x) (h_tangent : ∃ x₀, f x₀ = x₀ * deriv f x₀ ∧ (0 < f x₀)) :
  ∃ k, k = Real.exp 1 :=
by
  sorry

end slope_of_tangent_line_l523_52354


namespace pyramid_side_length_difference_l523_52346

theorem pyramid_side_length_difference (x : ℕ) (h1 : 1 + x^2 + (x + 1)^2 + (x + 2)^2 = 30) : x = 2 :=
by
  sorry

end pyramid_side_length_difference_l523_52346


namespace washer_cost_l523_52339

theorem washer_cost (D : ℝ) (H1 : D + (D + 220) = 1200) : D + 220 = 710 :=
by
  sorry

end washer_cost_l523_52339


namespace number_of_men_is_15_l523_52355

-- Define the conditions
def number_of_people : Prop :=
  ∃ (M W B : ℕ), M = 8 ∧ W = 8 ∧ B = 8 ∧ 8 * M = 120

-- Define the final statement to be proven
theorem number_of_men_is_15 (h: number_of_people) : ∃ M : ℕ, M = 15 :=
by
  obtain ⟨M, W, B, hM, hW, hB, htotal⟩ := h
  use M
  rw [hM] at htotal
  have hM15 : M = 15 := by linarith
  exact hM15

end number_of_men_is_15_l523_52355


namespace no_two_ways_for_z_l523_52379

theorem no_two_ways_for_z (z : ℤ) (x y x' y' : ℕ) 
  (hx : x ≤ y) (hx' : x' ≤ y') : ¬ (z = x! + y! ∧ z = x'! + y'! ∧ (x ≠ x' ∨ y ≠ y')) :=
by
  sorry

end no_two_ways_for_z_l523_52379


namespace total_households_in_apartment_complex_l523_52356

theorem total_households_in_apartment_complex :
  let buildings := 25
  let floors_per_building := 10
  let households_per_floor := 8
  buildings * floors_per_building * households_per_floor = 2000 :=
by
  sorry

end total_households_in_apartment_complex_l523_52356


namespace quadratic_roots_sign_l523_52330

theorem quadratic_roots_sign (p q : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x * y = q ∧ x + y = -p) ↔ q < 0 :=
sorry

end quadratic_roots_sign_l523_52330


namespace remainder_of_3_pow_101_plus_5_mod_11_l523_52385

theorem remainder_of_3_pow_101_plus_5_mod_11 : (3 ^ 101 + 5) % 11 = 8 := by
  -- The theorem statement includes the condition that (3^101 + 5) mod 11 equals 8.
  -- The proof will make use of repetitive behavior and modular arithmetic.
  sorry

end remainder_of_3_pow_101_plus_5_mod_11_l523_52385


namespace zachary_pushups_l523_52342

theorem zachary_pushups (david_pushups : ℕ) (h1 : david_pushups = 44) (h2 : ∀ z : ℕ, z = david_pushups + 7) : z = 51 :=
by
  sorry

end zachary_pushups_l523_52342


namespace tan_pi_over_12_minus_tan_pi_over_6_l523_52375

theorem tan_pi_over_12_minus_tan_pi_over_6 :
  (Real.tan (Real.pi / 12) - Real.tan (Real.pi / 6)) = 7 - 4 * Real.sqrt 3 :=
  sorry

end tan_pi_over_12_minus_tan_pi_over_6_l523_52375


namespace fraction_second_year_students_l523_52382

theorem fraction_second_year_students
  (total_students : ℕ)
  (third_year_students : ℕ)
  (second_year_students : ℕ)
  (h1 : third_year_students = total_students * 30 / 100)
  (h2 : second_year_students = total_students * 10 / 100) :
  (second_year_students : ℚ) / (total_students - third_year_students) = 1 / 7 := by
  sorry

end fraction_second_year_students_l523_52382


namespace binomial_expansion_sum_l523_52302

theorem binomial_expansion_sum (n : ℕ) (h : (2:ℕ)^n = 256) : n = 8 :=
sorry

end binomial_expansion_sum_l523_52302


namespace symmetric_line_eq_l523_52366

-- Definitions for the given line equations
def l1 (x y : ℝ) : Prop := 3 * x - y - 3 = 0
def l2 (x y : ℝ) : Prop := x + y - 1 = 0
def l3 (x y : ℝ) : Prop := x - 3 * y - 1 = 0

-- The theorem to prove
theorem symmetric_line_eq (x y : ℝ) (h1: l1 x y) (h2: l2 x y) : l3 x y :=
sorry

end symmetric_line_eq_l523_52366


namespace relationship_between_x_y_l523_52325

def in_interval (x : ℝ) : Prop := (Real.pi / 4) < x ∧ x < (Real.pi / 2)

noncomputable def x_def (α : ℝ) : ℝ := Real.sin α ^ (Real.log (Real.cos α) / Real.log α)

noncomputable def y_def (α : ℝ) : ℝ := Real.cos α ^ (Real.log (Real.sin α) / Real.log α)

theorem relationship_between_x_y (α : ℝ) (h : in_interval α) : 
  x_def α = y_def α := 
  sorry

end relationship_between_x_y_l523_52325


namespace original_employee_salary_l523_52364

-- Given conditions
def emily_original_salary : ℝ := 1000000
def emily_new_salary : ℝ := 850000
def number_of_employees : ℕ := 10
def employee_new_salary : ℝ := 35000

-- Prove the original salary of each employee
theorem original_employee_salary :
  (emily_original_salary - emily_new_salary) / number_of_employees = employee_new_salary - 20000 := 
by
  sorry

end original_employee_salary_l523_52364


namespace eggs_from_Martha_is_2_l523_52322

def eggs_from_Gertrude : ℕ := 4
def eggs_from_Blanche : ℕ := 3
def eggs_from_Nancy : ℕ := 2
def total_eggs_left : ℕ := 9
def eggs_dropped : ℕ := 2

def total_eggs_before_dropping (eggs_from_Martha : ℕ) :=
  eggs_from_Gertrude + eggs_from_Blanche + eggs_from_Nancy + eggs_from_Martha - eggs_dropped = total_eggs_left

-- The theorem stating the eggs collected from Martha.
theorem eggs_from_Martha_is_2 : ∃ (m : ℕ), total_eggs_before_dropping m ∧ m = 2 :=
by
  use 2
  sorry

end eggs_from_Martha_is_2_l523_52322


namespace no_n_nat_powers_l523_52348

theorem no_n_nat_powers (n : ℕ) : ∀ n : ℕ, ¬∃ m k : ℕ, k ≥ 2 ∧ n * (n + 1) = m ^ k := 
by 
  sorry

end no_n_nat_powers_l523_52348


namespace area_of_f2_equals_7_l523_52337

def f0 (x : ℝ) : ℝ := abs x
def f1 (x : ℝ) : ℝ := abs (f0 x - 1)
def f2 (x : ℝ) : ℝ := abs (f1 x - 2)

theorem area_of_f2_equals_7 : 
  (∫ x in (-3 : ℝ)..3, f2 x) = 7 :=
by
  sorry

end area_of_f2_equals_7_l523_52337


namespace value_of_v_l523_52344

theorem value_of_v (n : ℝ) (v : ℝ) (h1 : 10 * n = v - 2 * n) (h2 : n = -4.5) : v = -9 := by
  sorry

end value_of_v_l523_52344


namespace alice_weekly_walk_distance_l523_52352

theorem alice_weekly_walk_distance :
  let miles_to_school_per_day := 10
  let miles_home_per_day := 12
  let days_per_week := 5
  let weekly_total_miles := (miles_to_school_per_day * days_per_week) + (miles_home_per_day * days_per_week)
  weekly_total_miles = 110 :=
by
  sorry

end alice_weekly_walk_distance_l523_52352


namespace div_by_16_l523_52381

theorem div_by_16 (n : ℕ) : 
  ((2*n - 1)^3 - (2*n)^2 + 2*n + 1) % 16 = 0 :=
sorry

end div_by_16_l523_52381


namespace tan_alpha_eq_2_implies_sin_2alpha_inverse_l523_52371

theorem tan_alpha_eq_2_implies_sin_2alpha_inverse (α : ℝ) (h : Real.tan α = 2) :
  1 / Real.sin (2 * α) = 5 / 4 :=
sorry

end tan_alpha_eq_2_implies_sin_2alpha_inverse_l523_52371


namespace snail_kite_snails_eaten_l523_52341

theorem snail_kite_snails_eaten 
  (a₀ : ℕ) (a₁ : ℕ) (a₂ : ℕ) (a₃ : ℕ) (a₄ : ℕ)
  (h₀ : a₀ = 3)
  (h₁ : a₁ = a₀ + 2)
  (h₂ : a₂ = a₁ + 2)
  (h₃ : a₃ = a₂ + 2)
  (h₄ : a₄ = a₃ + 2)
  : a₀ + a₁ + a₂ + a₃ + a₄ = 35 := 
by 
  sorry

end snail_kite_snails_eaten_l523_52341


namespace relationship_between_abc_l523_52312

theorem relationship_between_abc (a b c k : ℝ) 
  (hA : -3 = - (k^2 + 1) / a)
  (hB : -2 = - (k^2 + 1) / b)
  (hC : 1 = - (k^2 + 1) / c)
  (hk : 0 < k^2 + 1) : c < a ∧ a < b :=
by
  sorry

end relationship_between_abc_l523_52312


namespace product_of_integers_cubes_sum_to_35_l523_52390

-- Define the conditions
def integers_sum_of_cubes (a b : ℤ) : Prop :=
  a^3 + b^3 = 35

-- Define the theorem that the product of integers whose cubes sum to 35 is 6
theorem product_of_integers_cubes_sum_to_35 :
  ∃ a b : ℤ, integers_sum_of_cubes a b ∧ a * b = 6 :=
by
  sorry

end product_of_integers_cubes_sum_to_35_l523_52390


namespace log_relationship_l523_52396

theorem log_relationship (a b c : ℝ) 
  (ha : a = Real.log 3 / Real.log 2) 
  (hb : b = Real.log 4 / Real.log 3) 
  (hc : c = Real.log 5 / Real.log 4) : 
  c < b ∧ b < a :=
by 
  sorry

end log_relationship_l523_52396


namespace count_defective_pens_l523_52365

theorem count_defective_pens
  (total_pens : ℕ) (prob_non_defective : ℚ)
  (h1 : total_pens = 12)
  (h2 : prob_non_defective = 0.5454545454545454) :
  ∃ (D : ℕ), D = 1 := by
  sorry

end count_defective_pens_l523_52365


namespace digits_of_2048_in_base_9_l523_52387

def digits_base9 (n : ℕ) : ℕ :=
if n < 9 then 1 else 1 + digits_base9 (n / 9)

theorem digits_of_2048_in_base_9 : digits_base9 2048 = 4 :=
by sorry

end digits_of_2048_in_base_9_l523_52387


namespace child_current_height_l523_52306

variable (h_last_visit : ℝ) (h_grown : ℝ)

-- Conditions
def last_height (h_last_visit : ℝ) := h_last_visit = 38.5
def height_grown (h_grown : ℝ) := h_grown = 3

-- Theorem statement
theorem child_current_height (h_last_visit h_grown : ℝ) 
    (h_last : last_height h_last_visit) 
    (h_grow : height_grown h_grown) : 
    h_last_visit + h_grown = 41.5 :=
by
  sorry

end child_current_height_l523_52306


namespace meaningful_expression_range_l523_52333

theorem meaningful_expression_range (x : ℝ) : (∃ y, y = 1 / (x - 4)) ↔ x ≠ 4 := 
by
  sorry

end meaningful_expression_range_l523_52333


namespace average_age_before_new_students_l523_52362

theorem average_age_before_new_students
  (N : ℕ) (A : ℚ) 
  (hN : N = 8) 
  (new_avg : (A - 4) = ((A * N) + (32 * 8)) / (N + 8)) 
  : A = 40 := 
by
  sorry

end average_age_before_new_students_l523_52362


namespace michelle_oranges_l523_52308

theorem michelle_oranges (x : ℕ) 
  (h1 : x - x / 3 - 5 = 7) : x = 18 :=
by
  -- We would normally provide the proof here, but it's omitted according to the instructions.
  sorry

end michelle_oranges_l523_52308


namespace point_coordinates_correct_l523_52319

def point_coordinates : (ℕ × ℕ) :=
(11, 9)

theorem point_coordinates_correct :
  point_coordinates = (11, 9) :=
by
  sorry

end point_coordinates_correct_l523_52319


namespace sum_series_l523_52368

theorem sum_series (s : ℕ → ℝ) 
  (h : ∀ n : ℕ, s n = (n+1) / (4 : ℝ)^(n+1)) : 
  tsum s = (4 / 9 : ℝ) :=
sorry

end sum_series_l523_52368


namespace additional_height_last_two_floors_l523_52324

-- Definitions of the problem conditions
def num_floors : ℕ := 20
def height_per_floor : ℕ := 3
def building_total_height : ℤ := 61

-- Condition on the height of first 18 floors
def height_first_18_floors : ℤ := 18 * 3

-- Height of the last two floors
def height_last_two_floors : ℤ := building_total_height - height_first_18_floors
def height_each_last_two_floor : ℤ := height_last_two_floors / 2

-- Height difference between the last two floors and the first 18 floors
def additional_height : ℤ := height_each_last_two_floor - height_per_floor

-- Theorem to prove
theorem additional_height_last_two_floors :
  additional_height = 1 / 2 := 
sorry

end additional_height_last_two_floors_l523_52324


namespace first_chinese_supercomputer_is_milkyway_l523_52377

-- Define the names of the computers
inductive ComputerName
| Universe
| Taihu
| MilkyWay
| Dawn

-- Define a structure to hold the properties of the computer
structure Computer :=
  (name : ComputerName)
  (introduction_year : Nat)
  (calculations_per_second : Nat)

-- Define the properties of the specific computer in the problem
def first_chinese_supercomputer := 
  Computer.mk ComputerName.MilkyWay 1983 100000000

-- The theorem to be proven
theorem first_chinese_supercomputer_is_milkyway :
  first_chinese_supercomputer.name = ComputerName.MilkyWay :=
by
  -- Provide the conditions that lead to the conclusion (proof steps will be added here)
  sorry

end first_chinese_supercomputer_is_milkyway_l523_52377


namespace kim_time_away_from_home_l523_52345

noncomputable def time_away_from_home (distance_to_friend : ℕ) (detour_percent : ℕ) (stay_time : ℕ) (speed_mph : ℕ) : ℕ :=
  let return_distance := distance_to_friend * (1 + detour_percent / 100)
  let total_distance := distance_to_friend + return_distance
  let driving_time := total_distance / speed_mph
  let driving_time_minutes := driving_time * 60
  driving_time_minutes + stay_time

theorem kim_time_away_from_home : 
  time_away_from_home 30 20 30 44 = 120 := 
by
  -- We will handle the proof here
  sorry

end kim_time_away_from_home_l523_52345


namespace mushroom_children_count_l523_52397

variables {n : ℕ} {A V S R : ℕ}

-- Conditions:
def condition1 (n : ℕ) (A : ℕ) (V : ℕ) : Prop :=
  ∀ (k : ℕ), k < n → V + A / 2 = k

def condition2 (S : ℕ) (A : ℕ) (R : ℕ) (V : ℕ) : Prop :=
  S + A = R + V + A

-- Proof statement
theorem mushroom_children_count (n : ℕ) (A : ℕ) (V : ℕ) (S : ℕ) (R : ℕ) :
  condition1 n A V → condition2 S A R V → n = 6 :=
by
  intros hcondition1 hcondition2
  sorry

end mushroom_children_count_l523_52397


namespace transformation_correct_l523_52338

theorem transformation_correct (a x y : ℝ) (h : a * x = a * y) : 3 - a * x = 3 - a * y :=
sorry

end transformation_correct_l523_52338


namespace power_cycle_i_pow_2012_l523_52372

-- Define the imaginary unit i as a complex number
def i : ℂ := Complex.I

-- Define the periodic properties of i
theorem power_cycle (n : ℕ) : Complex := 
  match n % 4 with
  | 0 => 1
  | 1 => i
  | 2 => -1
  | 3 => -i
  | _ => 0 -- this case should never happen

-- Using the periodic properties
theorem i_pow_2012 : (i ^ 2012) = 1 := by
  sorry

end power_cycle_i_pow_2012_l523_52372


namespace no_A_with_integer_roots_l523_52331

theorem no_A_with_integer_roots 
  (A : ℕ) 
  (h1 : A > 0) 
  (h2 : A < 10) 
  : ¬ ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ p + q = 10 + A ∧ p * q = 10 * A + A :=
by sorry

end no_A_with_integer_roots_l523_52331


namespace determine_a_l523_52340

theorem determine_a : 
  (∃ (a : ℝ), ∀ (x y : ℝ), (x, y) = (-1, 2) → 3 * x + y + a = 0) → ∃ (a : ℝ), a = 1 :=
by
  sorry

end determine_a_l523_52340


namespace minimum_value_func_minimum_value_attained_l523_52320

noncomputable def func (x : ℝ) : ℝ := (4 / (x - 1)) + x

theorem minimum_value_func : ∀ (x : ℝ), x > 1 → func x ≥ 5 :=
by
  intros x hx
  -- proof goes here
  sorry

theorem minimum_value_attained : func 3 = 5 :=
by
  -- proof goes here
  sorry

end minimum_value_func_minimum_value_attained_l523_52320


namespace jill_food_spending_l523_52335

theorem jill_food_spending :
  ∀ (T : ℝ) (c f o : ℝ),
    c = 0.5 * T →
    o = 0.3 * T →
    (0.04 * c + 0 + 0.1 * o) = 0.05 * T →
    f = 0.2 * T :=
by
  intros T c f o h_c h_o h_tax
  sorry

end jill_food_spending_l523_52335


namespace solution_is_63_l523_52301

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)
def last_digit (n : ℕ) : ℕ := n % 10
def rhyming_primes_around (r : ℕ) : Prop :=
  r >= 1 ∧ r <= 100 ∧
  ¬ is_prime r ∧
  ∃ ps : List ℕ, (∀ p ∈ ps, is_prime p ∧ last_digit p = last_digit r) ∧
  (∀ q : ℕ, is_prime q ∧ last_digit q = last_digit r → q ∈ ps) ∧
  List.length ps = 4

theorem solution_is_63 : ∃ r : ℕ, rhyming_primes_around r ∧ r = 63 :=
by sorry

end solution_is_63_l523_52301


namespace village_population_l523_52350

theorem village_population (P : ℝ) (h : 0.8 * P = 32000) : P = 40000 := by
  sorry

end village_population_l523_52350


namespace number_of_paintings_per_new_gallery_l523_52343

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

end number_of_paintings_per_new_gallery_l523_52343


namespace toll_for_18_wheel_truck_l523_52300

-- Define the total number of wheels, wheels on the front axle, 
-- and wheels on each of the other axles.
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4

-- Define the formula for calculating the toll.
def toll_formula (x : ℕ) : ℝ := 2.50 + 0.50 * (x - 2)

-- Calculate the number of other axles.
def calc_other_axles (wheels_left : ℕ) (wheels_per_axle : ℕ) : ℕ :=
wheels_left / wheels_per_axle

-- Statement to prove the final toll is $4.00.
theorem toll_for_18_wheel_truck : toll_formula (
  1 + calc_other_axles (total_wheels - front_axle_wheels) other_axle_wheels
) = 4.00 :=
by sorry

end toll_for_18_wheel_truck_l523_52300
