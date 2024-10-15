import Mathlib

namespace NUMINAMATH_GPT_meaningful_sqrt_condition_l2311_231173

theorem meaningful_sqrt_condition (x : ℝ) : (2 * x - 1 ≥ 0) ↔ (x ≥ 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_sqrt_condition_l2311_231173


namespace NUMINAMATH_GPT_original_perimeter_not_necessarily_multiple_of_four_l2311_231135

/-
Define the conditions given in the problem:
1. A rectangle is divided into several smaller rectangles.
2. The perimeter of each of these smaller rectangles is a multiple of 4.
-/
structure Rectangle where
  length : ℕ
  width : ℕ

def perimeter (r : Rectangle) : ℕ :=
  2 * (r.length + r.width)

def is_multiple_of_four (n : ℕ) : Prop :=
  n % 4 = 0

def smaller_rectangles (rs : List Rectangle) : Prop :=
  ∀ r ∈ rs, is_multiple_of_four (perimeter r)

-- Define the main statement to be proved
theorem original_perimeter_not_necessarily_multiple_of_four (original : Rectangle) (rs : List Rectangle)
  (h1 : smaller_rectangles rs) (h2 : ∀ r ∈ rs, r.length * r.width = original.length * original.width) :
  ¬ is_multiple_of_four (perimeter original) :=
by
  sorry

end NUMINAMATH_GPT_original_perimeter_not_necessarily_multiple_of_four_l2311_231135


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2311_231115

open Real

theorem hyperbola_eccentricity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
    (h_right_focus : ∀ x y, x = c ∧ y = 0)
    (h_circle : ∀ x y, (x - c)^2 + y^2 = 4 * a^2)
    (h_tangent : ∀ x y, x = c ∧ y = 0 → (x^2 + y^2 = a^2 + b^2))
    : ∃ e : ℝ, e = sqrt 5 := by sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2311_231115


namespace NUMINAMATH_GPT_equal_intercepts_on_both_axes_l2311_231148

theorem equal_intercepts_on_both_axes (m : ℝ) :
  (5 - 2 * m ≠ 0) ∧
  (- (5 - 2 * m) / (m^2 - 2 * m - 3) = - (5 - 2 * m) / (2 * m^2 + m - 1)) ↔ m = -2 :=
by sorry

end NUMINAMATH_GPT_equal_intercepts_on_both_axes_l2311_231148


namespace NUMINAMATH_GPT_infinite_series_value_l2311_231113

noncomputable def sum_infinite_series : ℝ := ∑' n : ℕ, if n > 0 then 1 / (n * (n + 3)) else 0

theorem infinite_series_value :
  sum_infinite_series = 11 / 18 :=
sorry

end NUMINAMATH_GPT_infinite_series_value_l2311_231113


namespace NUMINAMATH_GPT_sarah_age_is_26_l2311_231100

theorem sarah_age_is_26 (mark_age billy_age ana_age : ℕ) (sarah_age : ℕ) 
  (h1 : sarah_age = 3 * mark_age - 4)
  (h2 : mark_age = billy_age + 4)
  (h3 : billy_age = ana_age / 2)
  (h4 : ana_age = 15 - 3) :
  sarah_age = 26 := 
sorry

end NUMINAMATH_GPT_sarah_age_is_26_l2311_231100


namespace NUMINAMATH_GPT_common_ratio_of_arithmetic_sequence_l2311_231112

theorem common_ratio_of_arithmetic_sequence (S_odd S_even : ℤ) (q : ℤ) 
  (h1 : S_odd + S_even = -240) (h2 : S_odd - S_even = 80) 
  (h3 : q = S_even / S_odd) : q = 2 := 
  sorry

end NUMINAMATH_GPT_common_ratio_of_arithmetic_sequence_l2311_231112


namespace NUMINAMATH_GPT_total_value_of_bills_l2311_231162

theorem total_value_of_bills 
  (total_bills : Nat := 12) 
  (num_5_dollar_bills : Nat := 4) 
  (num_10_dollar_bills : Nat := 8)
  (value_5_dollar_bill : Nat := 5)
  (value_10_dollar_bill : Nat := 10) :
  (num_5_dollar_bills * value_5_dollar_bill + num_10_dollar_bills * value_10_dollar_bill = 100) :=
by
  sorry

end NUMINAMATH_GPT_total_value_of_bills_l2311_231162


namespace NUMINAMATH_GPT_equation_of_line_through_P_l2311_231111

theorem equation_of_line_through_P (P : (ℝ × ℝ)) (A B : (ℝ × ℝ))
  (hP : P = (1, 3))
  (hMidpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hA : A.2 = 0)
  (hB : B.1 = 0) :
  ∃ c : ℝ, 3 * c + 1 = 3 ∧ (3 * A.1 / c + A.2 / 6 = 1) ∧ (3 * B.1 / c + B.2 / 6 = 1) := sorry

end NUMINAMATH_GPT_equation_of_line_through_P_l2311_231111


namespace NUMINAMATH_GPT_inequality_holds_for_all_l2311_231156

theorem inequality_holds_for_all (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) :
  (∀ α β : ℝ, ⌊(m + n) * α⌋ + ⌊(m + n) * β⌋ ≥ ⌊m * α⌋ + ⌊m * β⌋ + ⌊n * (α + β)⌋) → m = n :=
by sorry

end NUMINAMATH_GPT_inequality_holds_for_all_l2311_231156


namespace NUMINAMATH_GPT_unique_two_digit_factors_l2311_231114

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def factors (n : ℕ) (a b : ℕ) : Prop := a * b = n

theorem unique_two_digit_factors : 
  ∃! (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ factors 1950 a b :=
by sorry

end NUMINAMATH_GPT_unique_two_digit_factors_l2311_231114


namespace NUMINAMATH_GPT_first_day_is_sunday_l2311_231170

-- Define the days of the week
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

open Day

-- Function to determine the day of the week for a given day number
def day_of_month (n : ℕ) (start_day : Day) : Day :=
  match n % 7 with
  | 0 => start_day
  | 1 => match start_day with
          | Sunday    => Monday
          | Monday    => Tuesday
          | Tuesday   => Wednesday
          | Wednesday => Thursday
          | Thursday  => Friday
          | Friday    => Saturday
          | Saturday  => Sunday
  | 2 => match start_day with
          | Sunday    => Tuesday
          | Monday    => Wednesday
          | Tuesday   => Thursday
          | Wednesday => Friday
          | Thursday  => Saturday
          | Friday    => Sunday
          | Saturday  => Monday
-- ... and so on for the rest of the days of the week.
  | _ => start_day -- Assuming the pattern continues accordingly.

-- Prove that the first day of the month is a Sunday given that the 18th day of the month is a Wednesday.
theorem first_day_is_sunday (h : day_of_month 18 Wednesday = Wednesday) : day_of_month 1 Wednesday = Sunday :=
  sorry

end NUMINAMATH_GPT_first_day_is_sunday_l2311_231170


namespace NUMINAMATH_GPT_calculate_expression_l2311_231171

theorem calculate_expression : 5^3 + 5^3 + 5^3 + 5^3 = 625 :=
  sorry

end NUMINAMATH_GPT_calculate_expression_l2311_231171


namespace NUMINAMATH_GPT_find_f_2006_l2311_231102

-- Assuming an odd periodic function f with period 3(3x+1), defining the conditions.
def f : ℤ → ℤ := sorry -- Definition of f is not provided.

-- Conditions
axiom odd_function : ∀ x : ℤ, f (-x) = -f x
axiom period_3_function : ∀ x : ℤ, f (3 * x + 1) = f (3 * (x + 1) + 1)
axiom value_at_1 : f 1 = -1

-- Question: What is f(2006)?
theorem find_f_2006 : f 2006 = 1 := sorry

end NUMINAMATH_GPT_find_f_2006_l2311_231102


namespace NUMINAMATH_GPT_find_interval_l2311_231107

theorem find_interval (x : ℝ) : (x > 3/4 ∧ x < 4/5) ↔ (5 * x + 1 > 3 ∧ 5 * x + 1 < 5 ∧ 4 * x > 3 ∧ 4 * x < 5) :=
by
  sorry

end NUMINAMATH_GPT_find_interval_l2311_231107


namespace NUMINAMATH_GPT_tire_circumference_l2311_231185

theorem tire_circumference (rpm : ℕ) (speed_kmh : ℕ) (C : ℝ) 
  (h1 : rpm = 400) 
  (h2 : speed_kmh = 144) 
  (h3 : (speed_kmh * 1000 / 60) = (rpm * C)) : 
  C = 6 :=
by
  sorry

end NUMINAMATH_GPT_tire_circumference_l2311_231185


namespace NUMINAMATH_GPT_unique_triplets_l2311_231155

theorem unique_triplets (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + 
               |c * x + a * y + b * z| = |x| + |y| + |z|) ↔ 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = -1)) :=
sorry

end NUMINAMATH_GPT_unique_triplets_l2311_231155


namespace NUMINAMATH_GPT_kyle_speed_l2311_231139

theorem kyle_speed (S : ℝ) (joseph_speed : ℝ) (joseph_time : ℝ) (kyle_time : ℝ) (H1 : joseph_speed = 50) (H2 : joseph_time = 2.5) (H3 : kyle_time = 2) (H4 : joseph_speed * joseph_time = kyle_time * S + 1) : S = 62 :=
by
  sorry

end NUMINAMATH_GPT_kyle_speed_l2311_231139


namespace NUMINAMATH_GPT_cubic_repeated_root_b_eq_100_l2311_231197

theorem cubic_repeated_root_b_eq_100 (b : ℝ) (h1 : b ≠ 0)
  (h2 : ∃ x : ℝ, (b * x^3 + 15 * x^2 + 9 * x + 2 = 0) ∧ 
                 (3 * b * x^2 + 30 * x + 9 = 0)) :
  b = 100 :=
sorry

end NUMINAMATH_GPT_cubic_repeated_root_b_eq_100_l2311_231197


namespace NUMINAMATH_GPT_fraction_area_above_line_l2311_231165

-- Define the problem conditions
def point1 : ℝ × ℝ := (4, 1)
def point2 : ℝ × ℝ := (9, 5)
def vertex1 : ℝ × ℝ := (4, 0)
def vertex2 : ℝ × ℝ := (9, 0)
def vertex3 : ℝ × ℝ := (9, 5)
def vertex4 : ℝ × ℝ := (4, 5)

-- Define the theorem statement
theorem fraction_area_above_line :
  let area_square := 25
  let area_below_line := 2.5
  let area_above_line := area_square - area_below_line
  area_above_line / area_square = 9 / 10 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_fraction_area_above_line_l2311_231165


namespace NUMINAMATH_GPT_total_population_l2311_231121

theorem total_population (P : ℝ) : 0.96 * P = 23040 → P = 24000 :=
by
  sorry

end NUMINAMATH_GPT_total_population_l2311_231121


namespace NUMINAMATH_GPT_chessboard_accessible_squares_l2311_231124

def is_accessible (board_size : ℕ) (central_exclusion_count : ℕ) (total_squares central_inaccessible : ℕ) : Prop :=
  total_squares = board_size * board_size ∧
  central_inaccessible = central_exclusion_count + 1 + 14 + 14 ∧
  board_size = 15 ∧
  total_squares - central_inaccessible = 196

theorem chessboard_accessible_squares :
  is_accessible 15 29 225 29 :=
by {
  sorry
}

end NUMINAMATH_GPT_chessboard_accessible_squares_l2311_231124


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2311_231158

variables (x y : ℝ)

theorem sufficient_but_not_necessary_condition :
  ((x - 1) ^ 2 + (y - 2) ^ 2 = 0) → ((x - 1) * (y - 2) = 0) ∧ (¬ ((x - 1) * (y-2) = 0 → (x - 1)^2 + (y - 2)^2 = 0)) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2311_231158


namespace NUMINAMATH_GPT_wall_area_in_square_meters_l2311_231119

variable {W H : ℤ} -- We treat W and H as integers referring to centimeters

theorem wall_area_in_square_meters 
  (h₁ : W / 30 = 8) 
  (h₂ : H / 30 = 5) : 
  (W / 100) * (H / 100) = 360 / 100 :=
by 
  sorry

end NUMINAMATH_GPT_wall_area_in_square_meters_l2311_231119


namespace NUMINAMATH_GPT_contradiction_proof_l2311_231145

theorem contradiction_proof (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) : ¬ (¬ (a > 0) ∨ ¬ (b > 0) ∨ ¬ (c > 0)) → false :=
by sorry

end NUMINAMATH_GPT_contradiction_proof_l2311_231145


namespace NUMINAMATH_GPT_train_crossing_time_l2311_231101

/-- Prove the time it takes for a train of length 50 meters running at 60 km/hr to cross a pole is 3 seconds. -/
theorem train_crossing_time
  (speed_kmh : ℝ)
  (length_m : ℝ)
  (conversion_factor : ℝ)
  (time_seconds : ℝ) :
  speed_kmh = 60 →
  length_m = 50 →
  conversion_factor = 1000 / 3600 →
  time_seconds = 3 →
  time_seconds = length_m / (speed_kmh * conversion_factor) := 
by
  intros
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2311_231101


namespace NUMINAMATH_GPT_evaluate_expression_l2311_231151

theorem evaluate_expression (x : Real) (hx : x = -52.7) : 
  ⌈(⌊|x|⌋ + ⌈|x|⌉)⌉ = 105 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2311_231151


namespace NUMINAMATH_GPT_math_proof_problem_l2311_231147

theorem math_proof_problem (a b : ℝ) (h1 : 64 = 8^2) (h2 : 16 = 8^2) :
  8^15 / (64^7) * 16 = 512 :=
by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l2311_231147


namespace NUMINAMATH_GPT_range_of_m_l2311_231143

noncomputable def f (x m : ℝ) := x^2 - 2 * m * x + 4

def P (m : ℝ) : Prop := ∀ x, 2 ≤ x → f x m ≥ f (2 : ℝ) m
def Q (m : ℝ) : Prop := ∀ x, 4 * x^2 + 4 * (m - 2) * x + 1 > 0

theorem range_of_m (m : ℝ) : (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ m ≤ 1 ∨ (2 < m ∧ m < 3) := sorry

end NUMINAMATH_GPT_range_of_m_l2311_231143


namespace NUMINAMATH_GPT_find_k_l2311_231132

theorem find_k (k : ℝ) (x : ℝ) :
  x^2 + k * x + 1 = 0 ∧ x^2 - x - k = 0 → k = 2 := 
sorry

end NUMINAMATH_GPT_find_k_l2311_231132


namespace NUMINAMATH_GPT_find_a_l2311_231180

variable {a : ℝ}

def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, a^2 + 3}

theorem find_a (h : A ∩ (B a) = {2}) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2311_231180


namespace NUMINAMATH_GPT_girls_to_boys_ratio_l2311_231159

theorem girls_to_boys_ratio (g b : ℕ) (h1 : g = b + 5) (h2 : g + b = 35) : g / b = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_girls_to_boys_ratio_l2311_231159


namespace NUMINAMATH_GPT_finitely_many_n_divisors_in_A_l2311_231146

-- Lean 4 statement
theorem finitely_many_n_divisors_in_A (A : Finset ℕ) (a : ℕ) (hA : ∀ p ∈ A, Nat.Prime p) (ha : a ≥ 2) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → ∃ p : ℕ, p ∣ a^n - 1 ∧ p ∉ A := by
  sorry

end NUMINAMATH_GPT_finitely_many_n_divisors_in_A_l2311_231146


namespace NUMINAMATH_GPT_john_blue_pens_l2311_231196

variables (R B Bl : ℕ)

axiom total_pens : R + B + Bl = 31
axiom black_more_red : B = R + 5
axiom blue_twice_black : Bl = 2 * B

theorem john_blue_pens : Bl = 18 :=
by
  apply sorry

end NUMINAMATH_GPT_john_blue_pens_l2311_231196


namespace NUMINAMATH_GPT_find_a_if_even_function_l2311_231163

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 * a^2 - a) * x + 1

theorem find_a_if_even_function (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_find_a_if_even_function_l2311_231163


namespace NUMINAMATH_GPT_willam_farm_tax_l2311_231190

theorem willam_farm_tax
  (T : ℝ)
  (h1 : 0.4 * T * (3840 / (0.4 * T)) = 3840)
  (h2 : 0 < T) :
  0.3125 * T * (3840 / (0.4 * T)) = 3000 := by
  sorry

end NUMINAMATH_GPT_willam_farm_tax_l2311_231190


namespace NUMINAMATH_GPT_rest_of_customers_bought_20_l2311_231186

/-
Let's define the number of melons sold by the stand, number of customers who bought one and three melons, and total number of melons bought by these customers.
-/

def total_melons_sold : ℕ := 46
def customers_bought_one : ℕ := 17
def customers_bought_three : ℕ := 3

def melons_bought_by_those_bought_one := customers_bought_one * 1
def melons_bought_by_those_bought_three := customers_bought_three * 3

def remaining_melons := total_melons_sold - (melons_bought_by_those_bought_one + melons_bought_by_those_bought_three)

-- Now we state the theorem that the number of melons bought by the rest of the customers is 20 
theorem rest_of_customers_bought_20 :
  remaining_melons = 20 :=
by
  -- Skip the proof with 'sorry'
  sorry

end NUMINAMATH_GPT_rest_of_customers_bought_20_l2311_231186


namespace NUMINAMATH_GPT_billy_soda_distribution_l2311_231161

theorem billy_soda_distribution (sisters : ℕ) (brothers : ℕ) (total_sodas : ℕ) (total_siblings : ℕ)
  (h1 : total_sodas = 12)
  (h2 : sisters = 2)
  (h3 : brothers = 2 * sisters)
  (h4 : total_siblings = sisters + brothers) :
  total_sodas / total_siblings = 2 :=
by
  sorry

end NUMINAMATH_GPT_billy_soda_distribution_l2311_231161


namespace NUMINAMATH_GPT_find_n_l2311_231172

theorem find_n : ∃ n : ℕ, n < 2006 ∧ ∀ m : ℕ, 2006 * n = m * (2006 + n) ↔ n = 1475 := by
  sorry

end NUMINAMATH_GPT_find_n_l2311_231172


namespace NUMINAMATH_GPT_chandler_bike_purchase_l2311_231164

theorem chandler_bike_purchase : 
    ∀ (x : ℕ), (200 + 20 * x = 800) → (x = 30) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_chandler_bike_purchase_l2311_231164


namespace NUMINAMATH_GPT_smallest_integer_representation_l2311_231169

theorem smallest_integer_representation :
  ∃ (A B C : ℕ), 0 ≤ A ∧ A < 5 ∧ 0 ≤ B ∧ B < 7 ∧ 0 ≤ C ∧ C < 4 ∧ 6 * A = 8 * B ∧ 6 * A = 5 * C ∧ 8 * B = 5 * C ∧ (6 * A) = 24 :=
  sorry

end NUMINAMATH_GPT_smallest_integer_representation_l2311_231169


namespace NUMINAMATH_GPT_determine_p_l2311_231198

theorem determine_p (m : ℕ) (p : ℕ) (h1: m = 34) 
  (h2: (1 : ℝ)^ (m + 1) / 5^ (m + 1) * 1^18 / 4^18 = 1 / (2 * 10^ p)) : 
  p = 35 := by sorry

end NUMINAMATH_GPT_determine_p_l2311_231198


namespace NUMINAMATH_GPT_total_coins_correct_l2311_231176

-- Define basic parameters
def stacks_pennies : Nat := 3
def coins_per_penny_stack : Nat := 10
def stacks_nickels : Nat := 5
def coins_per_nickel_stack : Nat := 8
def stacks_dimes : Nat := 7
def coins_per_dime_stack : Nat := 4

-- Calculate total coins for each type
def total_pennies : Nat := stacks_pennies * coins_per_penny_stack
def total_nickels : Nat := stacks_nickels * coins_per_nickel_stack
def total_dimes : Nat := stacks_dimes * coins_per_dime_stack

-- Calculate total number of coins
def total_coins : Nat := total_pennies + total_nickels + total_dimes

-- Proof statement
theorem total_coins_correct : total_coins = 98 := by
  -- Proof steps go here (omitted)
  sorry

end NUMINAMATH_GPT_total_coins_correct_l2311_231176


namespace NUMINAMATH_GPT_jay_more_points_than_tobee_l2311_231142

-- Declare variables.
variables (x J S : ℕ)

-- Given conditions
def Tobee_points := 4
def Jay_points := Tobee_points + x -- Jay_score is 4 + x
def Sean_points := (Tobee_points + Jay_points) - 2 -- Sean_score is 4 + Jay - 2

-- The total score condition
def total_score_condition := Tobee_points + Jay_points + Sean_points = 26

-- The main statement to be proven
theorem jay_more_points_than_tobee (h : total_score_condition) : J - Tobee_points = 6 :=
sorry

end NUMINAMATH_GPT_jay_more_points_than_tobee_l2311_231142


namespace NUMINAMATH_GPT_measure_of_angle_C_l2311_231168

theorem measure_of_angle_C (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_l2311_231168


namespace NUMINAMATH_GPT_num_exclusive_multiples_4_6_less_151_l2311_231193

def numMultiplesExclusive (n : ℕ) (a b : ℕ) : ℕ :=
  let lcm_ab := Nat.lcm a b
  (n-1) / a - (n-1) / lcm_ab + (n-1) / b - (n-1) / lcm_ab

theorem num_exclusive_multiples_4_6_less_151 : 
  numMultiplesExclusive 151 4 6 = 38 := 
by 
  sorry

end NUMINAMATH_GPT_num_exclusive_multiples_4_6_less_151_l2311_231193


namespace NUMINAMATH_GPT_additional_time_to_empty_tank_l2311_231136

-- Definitions based on conditions
def tankCapacity : ℕ := 3200  -- litres
def outletTimeAlone : ℕ := 5  -- hours
def inletRate : ℕ := 4  -- litres/min

-- Calculate rates
def outletRate : ℕ := tankCapacity / outletTimeAlone  -- litres/hour
def inletRatePerHour : ℕ := inletRate * 60  -- Convert litres/min to litres/hour

-- Calculate effective_rate when both pipes open
def effectiveRate : ℕ := outletRate - inletRatePerHour  -- litres/hour

-- Calculate times
def timeWithInletOpen : ℕ := tankCapacity / effectiveRate  -- hours
def additionalTime : ℕ := timeWithInletOpen - outletTimeAlone  -- hours

-- Proof statement
theorem additional_time_to_empty_tank : additionalTime = 3 := by
  -- It's clear from calculation above, we just add sorry for now to skip the proof
  sorry

end NUMINAMATH_GPT_additional_time_to_empty_tank_l2311_231136


namespace NUMINAMATH_GPT_Marla_laps_per_hour_l2311_231106

theorem Marla_laps_per_hour (M : ℝ) :
  (0.8 * M = 0.8 * 5 + 4) → M = 10 :=
by
  sorry

end NUMINAMATH_GPT_Marla_laps_per_hour_l2311_231106


namespace NUMINAMATH_GPT_correct_calculation_result_l2311_231157

theorem correct_calculation_result :
  (∃ x : ℤ, 14 * x = 70) → (5 - 6 = -1) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_result_l2311_231157


namespace NUMINAMATH_GPT_book_prices_l2311_231116

theorem book_prices (x : ℝ) (y : ℝ) (h1 : y = 2.5 * x) (h2 : 800 / x - 800 / y = 24) : (x = 20 ∧ y = 50) :=
by
  sorry

end NUMINAMATH_GPT_book_prices_l2311_231116


namespace NUMINAMATH_GPT_davi_minimum_spending_l2311_231154

-- Define the cost of a single bottle
def singleBottleCost : ℝ := 2.80

-- Define the cost of a box of six bottles
def boxCost : ℝ := 15.00

-- Define the number of bottles Davi needs to buy
def totalBottles : ℕ := 22

-- Calculate the minimum amount Davi will spend
def minimumCost : ℝ := 45.00 + 11.20 

-- The theorem to prove
theorem davi_minimum_spending :
  ∃ minCost : ℝ, minCost = 56.20 ∧ minCost = 3 * boxCost + 4 * singleBottleCost := 
by
  use 56.20
  sorry

end NUMINAMATH_GPT_davi_minimum_spending_l2311_231154


namespace NUMINAMATH_GPT_cab_speed_fraction_l2311_231178

theorem cab_speed_fraction (S R : ℝ) (h1 : S * 40 = R * 48) : (R / S) = (5 / 6) :=
sorry

end NUMINAMATH_GPT_cab_speed_fraction_l2311_231178


namespace NUMINAMATH_GPT_minimum_numbers_to_form_triangle_l2311_231144

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem minimum_numbers_to_form_triangle :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 1001) →
    16 ≤ S.card →
    ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ {a, b, c} ⊆ S ∧ is_triangle a b c :=
by
  sorry

end NUMINAMATH_GPT_minimum_numbers_to_form_triangle_l2311_231144


namespace NUMINAMATH_GPT_find_average_speed_l2311_231108

noncomputable def average_speed (distance1 distance2 : ℝ) (time1 time2 : ℝ) : ℝ := 
  (distance1 + distance2) / (time1 + time2)

theorem find_average_speed :
  average_speed 1000 1000 10 4 = 142.86 := by
  sorry

end NUMINAMATH_GPT_find_average_speed_l2311_231108


namespace NUMINAMATH_GPT_repeating_decimal_sum_l2311_231160

-- Definitions based on conditions
def x := 5 / 9  -- We derived this from 0.5 repeating as a fraction
def y := 7 / 99  -- Similarly, derived from 0.07 repeating as a fraction

-- Proposition to prove
theorem repeating_decimal_sum : x + y = 62 / 99 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l2311_231160


namespace NUMINAMATH_GPT_youngest_child_age_l2311_231133

theorem youngest_child_age (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_youngest_child_age_l2311_231133


namespace NUMINAMATH_GPT_machine_produces_480_cans_in_8_hours_l2311_231120

def cans_produced_in_interval : ℕ := 30
def interval_duration_minutes : ℕ := 30
def hours_worked : ℕ := 8
def minutes_in_hour : ℕ := 60

theorem machine_produces_480_cans_in_8_hours :
  (hours_worked * (minutes_in_hour / interval_duration_minutes) * cans_produced_in_interval) = 480 := by
  sorry

end NUMINAMATH_GPT_machine_produces_480_cans_in_8_hours_l2311_231120


namespace NUMINAMATH_GPT_ted_alex_age_ratio_l2311_231134

theorem ted_alex_age_ratio (t a : ℕ) 
  (h1 : t - 3 = 4 * (a - 3))
  (h2 : t - 5 = 5 * (a - 5)) : 
  ∃ x : ℕ, (t + x) / (a + x) = 3 ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_ted_alex_age_ratio_l2311_231134


namespace NUMINAMATH_GPT_sum_of_first_8_terms_l2311_231118

theorem sum_of_first_8_terms (a : ℝ) (h : 15 * a = 1) : 
  (a + 2 * a + 4 * a + 8 * a + 16 * a + 32 * a + 64 * a + 128 * a) = 17 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_8_terms_l2311_231118


namespace NUMINAMATH_GPT_breadth_of_rectangular_plot_l2311_231175

theorem breadth_of_rectangular_plot
  (b l : ℕ)
  (h1 : l = 3 * b)
  (h2 : l * b = 2028) :
  b = 26 :=
sorry

end NUMINAMATH_GPT_breadth_of_rectangular_plot_l2311_231175


namespace NUMINAMATH_GPT_fraction_equation_solution_l2311_231128

theorem fraction_equation_solution (a : ℤ) (hpos : a > 0) (h : (a : ℝ) / (a + 50) = 0.870) : a = 335 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_equation_solution_l2311_231128


namespace NUMINAMATH_GPT_fraction_remains_unchanged_l2311_231174

theorem fraction_remains_unchanged (x y : ℝ) : 
  (3 * (2 * x)) / (2 * (2 * y)) = (3 * x) / (2 * y) :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_remains_unchanged_l2311_231174


namespace NUMINAMATH_GPT_y_coord_range_of_M_l2311_231140

theorem y_coord_range_of_M :
  ∀ (M : ℝ × ℝ), ((M.1 + 1)^2 + M.2^2 = 2) → 
  ((M.1 - 2)^2 + M.2^2 + M.1^2 + M.2^2 ≤ 10) →
  - (Real.sqrt 7) / 2 ≤ M.2 ∧ M.2 ≤ (Real.sqrt 7) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_y_coord_range_of_M_l2311_231140


namespace NUMINAMATH_GPT_scientific_notation_14000000_l2311_231126

theorem scientific_notation_14000000 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 14000000 = a * 10 ^ n ∧ a = 1.4 ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_14000000_l2311_231126


namespace NUMINAMATH_GPT_max_roses_l2311_231129

theorem max_roses (budget : ℝ) (indiv_price : ℝ) (dozen_1_price : ℝ) (dozen_2_price : ℝ) (dozen_5_price : ℝ) (hundred_price : ℝ) 
  (budget_eq : budget = 1000) (indiv_price_eq : indiv_price = 5.30) (dozen_1_price_eq : dozen_1_price = 36) 
  (dozen_2_price_eq : dozen_2_price = 50) (dozen_5_price_eq : dozen_5_price = 110) (hundred_price_eq : hundred_price = 180) : 
  ∃ max_roses : ℕ, max_roses = 548 :=
by
  sorry

end NUMINAMATH_GPT_max_roses_l2311_231129


namespace NUMINAMATH_GPT_max_trains_final_count_l2311_231117

-- Define the conditions
def trains_per_birthdays : Nat := 1
def trains_per_christmas : Nat := 2
def trains_per_easter : Nat := 3
def years : Nat := 7

-- Function to calculate total trains after 7 years
def total_trains_after_years (trains_per_years : Nat) (num_years : Nat) : Nat :=
  trains_per_years * num_years

-- Calculate inputs
def trains_per_year : Nat := trains_per_birthdays + trains_per_christmas + trains_per_easter
def total_initial_trains : Nat := total_trains_after_years trains_per_year years

-- Bonus and final steps
def bonus_trains_from_cousins (initial_trains : Nat) : Nat := initial_trains / 2
def final_total_trains (initial_trains : Nat) (bonus_trains : Nat) : Nat :=
  let after_bonus := initial_trains + bonus_trains
  let additional_from_parents := after_bonus * 3
  after_bonus + additional_from_parents

-- Main theorem
theorem max_trains_final_count : final_total_trains total_initial_trains (bonus_trains_from_cousins total_initial_trains) = 252 := by
  sorry

end NUMINAMATH_GPT_max_trains_final_count_l2311_231117


namespace NUMINAMATH_GPT_frog_climbing_time_l2311_231149

-- Defining the conditions as Lean definitions
def well_depth : ℕ := 12
def climb_distance : ℕ := 3
def slip_distance : ℕ := 1
def climb_time : ℚ := 1 -- time in minutes for the frog to climb 3 meters
def slip_time : ℚ := climb_time / 3
def total_time_per_cycle : ℚ := climb_time + slip_time
def total_climbed_at_817 : ℕ := well_depth - 3 -- 3 meters from the top means it climbed 9 meters

-- The equivalent proof statement in Lean:
theorem frog_climbing_time : 
  ∃ (T : ℚ), T = 22 ∧ 
    (well_depth = 9 + 3) ∧
    (∀ (cycles : ℕ), cycles = 4 → 
         total_time_per_cycle * cycles + 2 = T) :=
by 
  sorry

end NUMINAMATH_GPT_frog_climbing_time_l2311_231149


namespace NUMINAMATH_GPT_tank_capacity_l2311_231182

theorem tank_capacity (x : ℝ) (h : 0.24 * x = 120) : x = 500 := 
sorry

end NUMINAMATH_GPT_tank_capacity_l2311_231182


namespace NUMINAMATH_GPT_alice_stops_in_quarter_D_l2311_231191

-- Definitions and conditions
def indoor_track_circumference : ℕ := 40
def starting_point_S : ℕ := 0
def run_distance : ℕ := 1600

-- Desired theorem statement
theorem alice_stops_in_quarter_D :
  (run_distance % indoor_track_circumference = 0) → 
  (0 ≤ (run_distance % indoor_track_circumference) ∧ 
   (run_distance % indoor_track_circumference) < indoor_track_circumference) → 
  true := by
  sorry

end NUMINAMATH_GPT_alice_stops_in_quarter_D_l2311_231191


namespace NUMINAMATH_GPT_find_k_l2311_231192

variables (m n k : ℤ)  -- Declaring m, n, k as integer variables.

theorem find_k (h1 : m = 2 * n + 5) (h2 : m + 2 = 2 * (n + k) + 5) : k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2311_231192


namespace NUMINAMATH_GPT_geom_seq_ratio_l2311_231141
noncomputable section

theorem geom_seq_ratio (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h₁ : 0 < a_1)
  (h₂ : 0 < a_2)
  (h₃ : 0 < a_3)
  (h₄ : 0 < a_4)
  (h₅ : 0 < a_5)
  (h_seq : a_2 = a_1 * 2)
  (h_seq2 : a_3 = a_1 * 2^2)
  (h_seq3 : a_4 = a_1 * 2^3)
  (h_seq4 : a_5 = a_1 * 2^4)
  (h_ratio : a_4 / a_1 = 8) :
  (a_1 + a_2) * a_4 / ((a_1 + a_3) * a_5) = 3 / 10 := 
by
  sorry

end NUMINAMATH_GPT_geom_seq_ratio_l2311_231141


namespace NUMINAMATH_GPT_sum_of_digits_of_9ab_l2311_231137

noncomputable def a : ℕ := 10^2023 - 1
noncomputable def b : ℕ := 2*(10^2023 - 1) / 3

def digitSum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_of_9ab :
  digitSum (9 * a * b) = 20235 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_9ab_l2311_231137


namespace NUMINAMATH_GPT_solve_system_of_equations_l2311_231167

variable (a x y z : ℝ)

theorem solve_system_of_equations (h1 : x^2 + y^2 - 2 * z^2 = 2 * a^2)
                                  (h2 : x + y + 2 * z = 4 * (a^2 + 1))
                                  (h3 : z^2 - x * y = a^2) :
                                  (x = a^2 + a + 1 ∧ y = a^2 - a + 1 ∧ z = a^2 + 1) ∨
                                  (x = a^2 - a + 1 ∧ y = a^2 + a + 1 ∧ z = a^2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2311_231167


namespace NUMINAMATH_GPT_perimeter_of_polygon_l2311_231125

-- Define the dimensions of the strips and their arrangement
def strip_width : ℕ := 4
def strip_length : ℕ := 16
def num_vertical_strips : ℕ := 2
def num_horizontal_strips : ℕ := 2

-- State the problem condition and the expected perimeter
theorem perimeter_of_polygon : 
  let vertical_perimeter := num_vertical_strips * strip_length
  let horizontal_perimeter := num_horizontal_strips * strip_length
  let corner_segments_perimeter := (num_vertical_strips + num_horizontal_strips) * strip_width
  vertical_perimeter + horizontal_perimeter + corner_segments_perimeter = 80 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_polygon_l2311_231125


namespace NUMINAMATH_GPT_proposition_C_l2311_231194

theorem proposition_C (a b : ℝ) : a^3 > b^3 → a > b :=
sorry

end NUMINAMATH_GPT_proposition_C_l2311_231194


namespace NUMINAMATH_GPT_regular_price_of_shirt_is_50_l2311_231179

-- Define all relevant conditions and given prices.
variables (P : ℝ) (shirt_price_discounted : ℝ) (total_paid : ℝ) (number_of_shirts : ℝ)

-- Define the conditions as hypotheses
def conditions :=
  (shirt_price_discounted = 0.80 * P) ∧
  (total_paid = 240) ∧
  (number_of_shirts = 6) ∧
  (total_paid = number_of_shirts * shirt_price_discounted)

-- State the theorem to prove that the regular price of the shirt is $50.
theorem regular_price_of_shirt_is_50 (h : conditions P shirt_price_discounted total_paid number_of_shirts) :
  P = 50 := 
sorry

end NUMINAMATH_GPT_regular_price_of_shirt_is_50_l2311_231179


namespace NUMINAMATH_GPT_probability_sum_l2311_231152

noncomputable def P : ℕ → ℝ := sorry

theorem probability_sum (n : ℕ) (h : n ≥ 7) :
  P n = (1/6) * (P (n-1) + P (n-2) + P (n-3) + P (n-4) + P (n-5) + P (n-6)) :=
sorry

end NUMINAMATH_GPT_probability_sum_l2311_231152


namespace NUMINAMATH_GPT_unique_a_exists_iff_n_eq_two_l2311_231123

theorem unique_a_exists_iff_n_eq_two (n : ℕ) (h1 : 1 < n) : 
  (∃ a : ℕ, 0 < a ∧ a ≤ n! ∧ n! ∣ a^n + 1 ∧ ∀ b : ℕ, (0 < b ∧ b ≤ n! ∧ n! ∣ b^n + 1) → b = a) ↔ n = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_unique_a_exists_iff_n_eq_two_l2311_231123


namespace NUMINAMATH_GPT_number_of_two_element_subsets_l2311_231131

def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem number_of_two_element_subsets (S : Type*) [Fintype S] 
  (h : binomial_coeff (Fintype.card S) 7 = 36) :
  binomial_coeff (Fintype.card S) 2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_number_of_two_element_subsets_l2311_231131


namespace NUMINAMATH_GPT_melissa_solves_equation_l2311_231110

theorem melissa_solves_equation : 
  ∃ b c : ℤ, (∀ x : ℝ, x^2 - 6 * x + 9 = 0 ↔ (x + b)^2 = c) ∧ b + c = -3 :=
by
  sorry

end NUMINAMATH_GPT_melissa_solves_equation_l2311_231110


namespace NUMINAMATH_GPT_trigonometric_identity_l2311_231138

open Real

-- Lean 4 statement
theorem trigonometric_identity (α β γ x : ℝ) :
  (sin (x - β) * sin (x - γ) / (sin (α - β) * sin (α - γ))) +
  (sin (x - γ) * sin (x - α) / (sin (β - γ) * sin (β - α))) +
  (sin (x - α) * sin (x - β) / (sin (γ - α) * sin (γ - β))) = 1 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l2311_231138


namespace NUMINAMATH_GPT_proportional_function_quadrants_l2311_231153

theorem proportional_function_quadrants (k : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ y = k * x) ∧ (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ y = k * x) → k < 0 :=
by
  sorry

end NUMINAMATH_GPT_proportional_function_quadrants_l2311_231153


namespace NUMINAMATH_GPT_length_SR_l2311_231150

theorem length_SR (cos_S : ℝ) (SP : ℝ) (SR : ℝ) (h1 : cos_S = 0.5) (h2 : SP = 10) (h3 : cos_S = SP / SR) : SR = 20 := by
  sorry

end NUMINAMATH_GPT_length_SR_l2311_231150


namespace NUMINAMATH_GPT_triangle_isosceles_if_equal_bisectors_l2311_231177

theorem triangle_isosceles_if_equal_bisectors
  (A B C : ℝ)
  (a b c l_a l_b : ℝ)
  (ha : l_a = l_b)
  (h1 : l_a = 2 * b * c * Real.cos (A / 2) / (b + c))
  (h2 : l_b = 2 * a * c * Real.cos (B / 2) / (a + c)) :
  a = b :=
by
  sorry

end NUMINAMATH_GPT_triangle_isosceles_if_equal_bisectors_l2311_231177


namespace NUMINAMATH_GPT_find_function_f_l2311_231181

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_function_f (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y →
    f (f y / f x + 1) = f (x + y / x + 1) - f x) →
  ∀ x : ℝ, 0 < x → f x = a * x :=
  by sorry

end NUMINAMATH_GPT_find_function_f_l2311_231181


namespace NUMINAMATH_GPT_probability_three_heads_l2311_231183

noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability (n : ℕ) (k : ℕ) : ℚ :=
  (binom n k) / (2 ^ n)

theorem probability_three_heads : probability 12 3 = 55 / 1024 := 
by
  sorry

end NUMINAMATH_GPT_probability_three_heads_l2311_231183


namespace NUMINAMATH_GPT_Q_over_P_l2311_231122

theorem Q_over_P (P Q : ℚ)
  (h : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -3 → 
    (P / (x + 3) + Q / (x^2 - 3*x) = (x^2 - x + 8) / (x^3 + x^2 - 9*x))) :
  Q / P = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_Q_over_P_l2311_231122


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_l2311_231189

variable (p : ℝ)
variable (x : ℝ)

theorem inequality_holds_for_all_x (h : -3 < p ∧ p < 6) : 
  -9 < (3*x^2 + p*x - 6) / (x^2 - x + 1) ∧ (3*x^2 + p*x - 6) / (x^2 - x + 1) < 6 := by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_l2311_231189


namespace NUMINAMATH_GPT_range_of_a_l2311_231166

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - a * x

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), 0 < x1 ∧ 0 < x2 ∧ x2 > x1 → (f x1 a / x2 - f x2 a / x1 < 0)) ↔ a ≤ Real.exp 1 / 2 := sorry

end NUMINAMATH_GPT_range_of_a_l2311_231166


namespace NUMINAMATH_GPT_lines_intersect_l2311_231187

theorem lines_intersect (a b : ℝ) (h1 : 2 = (1/3) * 1 + a) (h2 : 1 = (1/2) * 2 + b) : a + b = 5 / 3 := 
by {
  -- Skipping the proof itself
  sorry
}

end NUMINAMATH_GPT_lines_intersect_l2311_231187


namespace NUMINAMATH_GPT_root_interval_l2311_231104

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 * x - 3

theorem root_interval (x0 : ℝ) (h : f x0 = 0): x0 ∈ Set.Ioo (1 / 4 : ℝ) (1 / 2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_root_interval_l2311_231104


namespace NUMINAMATH_GPT_helium_balloon_buoyancy_l2311_231199

variable (m m₁ Mₐ M_b : ℝ)
variable (h₁ : m₁ = 10)
variable (h₂ : Mₐ = 4)
variable (h₃ : M_b = 29)

theorem helium_balloon_buoyancy :
  m = (m₁ * Mₐ) / (M_b - Mₐ) :=
by
  sorry

end NUMINAMATH_GPT_helium_balloon_buoyancy_l2311_231199


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l2311_231188

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_cond : (a 0 * (1 + q + q^2)) / (a 0 * q^2) = 3) : q = 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l2311_231188


namespace NUMINAMATH_GPT_triangle_perimeter_l2311_231109

theorem triangle_perimeter (a b : ℝ) (x : ℝ) 
  (h₁ : a = 3) 
  (h₂ : b = 5) 
  (h₃ : x ^ 2 - 5 * x + 6 = 0)
  (h₄ : 2 < x ∧ x < 8) : a + b + x = 11 :=
by sorry

end NUMINAMATH_GPT_triangle_perimeter_l2311_231109


namespace NUMINAMATH_GPT_solution_exists_l2311_231103

theorem solution_exists (x : ℝ) :
  (|x - 10| + |x - 14| = |2 * x - 24|) ↔ (x = 12) :=
by
  sorry

end NUMINAMATH_GPT_solution_exists_l2311_231103


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l2311_231130

theorem geometric_sequence_first_term (a r : ℝ)
    (h1 : a * r^2 = 3)
    (h2 : a * r^4 = 27) :
    a = 1 / 3 := by
    sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l2311_231130


namespace NUMINAMATH_GPT_factorize_polynomial_l2311_231105

theorem factorize_polynomial (a x : ℝ) : 
  (x^3 - 3*x^2 + (a + 2)*x - 2*a) = (x^2 - x + a)*(x - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l2311_231105


namespace NUMINAMATH_GPT_red_fraction_is_three_fifths_l2311_231184

noncomputable def fraction_of_red_marbles (x : ℕ) : ℚ := 
  let blue_marbles := (2 / 3 : ℚ) * x
  let red_marbles := x - blue_marbles
  let new_red_marbles := 3 * red_marbles
  let new_total_marbles := blue_marbles + new_red_marbles
  new_red_marbles / new_total_marbles

theorem red_fraction_is_three_fifths (x : ℕ) (hx : x ≠ 0) : fraction_of_red_marbles x = 3 / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_red_fraction_is_three_fifths_l2311_231184


namespace NUMINAMATH_GPT_ferry_time_increases_l2311_231195

noncomputable def ferryRoundTrip (S V x : ℝ) : ℝ :=
  (S / (V + x)) + (S / (V - x))

theorem ferry_time_increases (S V x : ℝ) (h_V_pos : 0 < V) (h_x_lt_V : x < V) :
  ferryRoundTrip S V (x + 1) > ferryRoundTrip S V x :=
by
  sorry

end NUMINAMATH_GPT_ferry_time_increases_l2311_231195


namespace NUMINAMATH_GPT_ratio_eq_one_l2311_231127

theorem ratio_eq_one (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : (a / 3) / (b / 2) = 1 :=
sorry

end NUMINAMATH_GPT_ratio_eq_one_l2311_231127
