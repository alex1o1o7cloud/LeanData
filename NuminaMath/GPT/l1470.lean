import Mathlib

namespace length_of_PS_l1470_147068

noncomputable def triangle_segments : ℝ := 
  let PR := 15
  let ratio_PS_SR := 3 / 4
  let total_length := 15
  let SR := total_length / (1 + ratio_PS_SR)
  let PS := ratio_PS_SR * SR
  PS

theorem length_of_PS :
  triangle_segments = 45 / 7 :=
by
  sorry

end length_of_PS_l1470_147068


namespace cabinets_and_perimeter_l1470_147001

theorem cabinets_and_perimeter :
  ∀ (original_cabinets : ℕ) (install_factor : ℕ) (num_counters : ℕ) 
    (cabinets_L_1 cabinets_L_2 cabinets_L_3 removed_cabinets cabinet_height total_cabinets perimeter : ℕ),
    original_cabinets = 3 →
    install_factor = 2 →
    num_counters = 4 →
    cabinets_L_1 = 3 →
    cabinets_L_2 = 5 →
    cabinets_L_3 = 7 →
    removed_cabinets = 2 →
    cabinet_height = 2 →
    total_cabinets = (original_cabinets * install_factor * num_counters) + 
                     (cabinets_L_1 + cabinets_L_2 + cabinets_L_3) - removed_cabinets →
    perimeter = (cabinets_L_1 * cabinet_height) +
                (cabinets_L_3 * cabinet_height) +
                2 * (cabinets_L_2 * cabinet_height) →
    total_cabinets = 37 ∧
    perimeter = 40 :=
by
  intros
  sorry

end cabinets_and_perimeter_l1470_147001


namespace repeating_decimal_arithmetic_l1470_147089

def x : ℚ := 0.234 -- repeating decimal 0.234
def y : ℚ := 0.567 -- repeating decimal 0.567
def z : ℚ := 0.891 -- repeating decimal 0.891

theorem repeating_decimal_arithmetic :
  x - y + z = 186 / 333 := 
sorry

end repeating_decimal_arithmetic_l1470_147089


namespace gift_boxes_in_3_days_l1470_147049
-- Conditions:
def inchesPerBox := 18
def dailyWrapper := 90
-- "how many gift boxes will he be able to wrap every 3 days?"
theorem gift_boxes_in_3_days : 3 * (dailyWrapper / inchesPerBox) = 15 :=
by
  sorry

end gift_boxes_in_3_days_l1470_147049


namespace strawberry_cake_cost_proof_l1470_147020

-- Define the constants
def chocolate_cakes : ℕ := 3
def price_per_chocolate_cake : ℕ := 12
def total_bill : ℕ := 168
def number_of_strawberry_cakes : ℕ := 6

-- Define the calculation for the total cost of chocolate cakes
def total_cost_of_chocolate_cakes : ℕ := chocolate_cakes * price_per_chocolate_cake

-- Define the remaining cost for strawberry cakes
def remaining_cost : ℕ := total_bill - total_cost_of_chocolate_cakes

-- Prove the cost per strawberry cake
def cost_per_strawberry_cake : ℕ := remaining_cost / number_of_strawberry_cakes

theorem strawberry_cake_cost_proof : cost_per_strawberry_cake = 22 := by
  -- We skip the proof here. Detailed proof steps would go in the place of sorry
  sorry

end strawberry_cake_cost_proof_l1470_147020


namespace find_a_from_coefficient_l1470_147090

theorem find_a_from_coefficient :
  (∀ x : ℝ, (x + 1)^6 * (a*x - 1)^2 = 20 → a = 0 ∨ a = 5) :=
by
  sorry

end find_a_from_coefficient_l1470_147090


namespace greatest_possible_x_l1470_147056

theorem greatest_possible_x (x : ℕ) (h : x^3 < 15) : x ≤ 2 := by
  sorry

end greatest_possible_x_l1470_147056


namespace greatest_possible_x_l1470_147085

theorem greatest_possible_x : ∃ (x : ℕ), (x^2 + 5 < 30) ∧ ∀ (y : ℕ), (y^2 + 5 < 30) → y ≤ x :=
by
  sorry

end greatest_possible_x_l1470_147085


namespace remainder_when_subtract_div_by_6_l1470_147008

theorem remainder_when_subtract_div_by_6 (m n : ℕ) (h1 : m % 6 = 2) (h2 : n % 6 = 3) (h3 : m > n) : (m - n) % 6 = 5 := 
by
  sorry

end remainder_when_subtract_div_by_6_l1470_147008


namespace find_r_l1470_147075

theorem find_r (r s : ℝ)
  (h1 : ∀ α β : ℝ, (α + β = -r) ∧ (α * β = s) → 
         ∃ t : ℝ, (t^2 - (α^2 + β^2) * t + (α^2 * β^2) = 0) ∧ |α^2 - β^2| = 8)
  (h_sum : ∃ α β : ℝ, α + β = 10) :
  r = -10 := by
  sorry

end find_r_l1470_147075


namespace max_gcd_coprime_l1470_147015

theorem max_gcd_coprime (x y : ℤ) (h : Int.gcd x y = 1) : 
  Int.gcd (x + 2015 * y) (y + 2015 * x) ≤ 4060224 :=
sorry

end max_gcd_coprime_l1470_147015


namespace house_construction_days_l1470_147062

theorem house_construction_days
  (D : ℕ) -- number of planned days to build the house
  (Hwork_done : 1000 + 200 * (D - 10) = 100 * (D + 90)) : 
  D = 110 :=
sorry

end house_construction_days_l1470_147062


namespace smallest_positive_integer_l1470_147009

theorem smallest_positive_integer
    (n : ℕ)
    (h : ∀ (a : Fin n → ℤ), ∃ (i j : Fin n), i ≠ j ∧ (2009 ∣ (a i + a j) ∨ 2009 ∣ (a i - a j))) : n = 1006 := by
  -- Proof is required here
  sorry

end smallest_positive_integer_l1470_147009


namespace cannot_bisect_segment_with_ruler_l1470_147088

noncomputable def projective_transformation (A B M : Point) : Point :=
  -- This definition will use an unspecified projective transformation that leaves A and B invariant
  sorry

theorem cannot_bisect_segment_with_ruler (A B : Point) (method : Point -> Point -> Point) :
  (forall (phi : Point -> Point), phi A = A -> phi B = B -> phi (method A B) ≠ method A B) ->
  ¬ (exists (M : Point), method A B = M) := by
  sorry

end cannot_bisect_segment_with_ruler_l1470_147088


namespace min_value_condition_l1470_147035

variable (a b : ℝ)

theorem min_value_condition (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 2) :
    (1 / a^2) + (1 / b^2) = 9 / 2 :=
sorry

end min_value_condition_l1470_147035


namespace sqrt_identity_l1470_147037

def condition1 (α : ℝ) : Prop := 
  ∃ P : ℝ × ℝ, P = (Real.sin 2, Real.cos 2) ∧ Real.sin α = Real.cos 2

def condition2 (P : ℝ × ℝ) : Prop := 
  P.1 ^ 2 + P.2 ^ 2 = 1

theorem sqrt_identity (α : ℝ) (P : ℝ × ℝ) 
  (h₁ : condition1 α) (h₂ : condition2 P) : 
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := by 
  sorry

end sqrt_identity_l1470_147037


namespace quotient_of_0_009_div_0_3_is_0_03_l1470_147081

-- Statement:
theorem quotient_of_0_009_div_0_3_is_0_03 (x : ℝ) (h : x = 0.3) : 0.009 / x = 0.03 :=
by
  sorry

end quotient_of_0_009_div_0_3_is_0_03_l1470_147081


namespace find_other_root_l1470_147095

variable {m : ℝ} -- m is a real number
variable (x : ℝ)

theorem find_other_root (h : x^2 + m * x - 5 = 0) (hx1 : x = -1) : x = 5 :=
sorry

end find_other_root_l1470_147095


namespace polynomial_form_l1470_147013

noncomputable def polynomial_solution (P : ℝ → ℝ) :=
  ∀ a b c : ℝ, (a * b + b * c + c * a = 0) → (P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c))

theorem polynomial_form :
  ∀ (P : ℝ → ℝ), polynomial_solution P ↔ ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x^2 + b * x^4 :=
by 
  sorry

end polynomial_form_l1470_147013


namespace simplify_tan_cot_60_l1470_147061

theorem simplify_tan_cot_60 :
  let tan60 := Real.sqrt 3
  let cot60 := 1 / Real.sqrt 3
  (tan60^3 + cot60^3) / (tan60 + cot60) = 7 / 3 :=
by
  let tan60 := Real.sqrt 3
  let cot60 := 1 / Real.sqrt 3
  sorry

end simplify_tan_cot_60_l1470_147061


namespace sum_of_ages_l1470_147084

theorem sum_of_ages (M C : ℝ) (h1 : M = C + 12) (h2 : M + 10 = 3 * (C - 6)) : M + C = 52 :=
by
  sorry

end sum_of_ages_l1470_147084


namespace circle_condition_iff_l1470_147042

-- Given a condition a < 2, we need to show it is a necessary and sufficient condition
-- for the equation x^2 + y^2 - 2x + 2y + a = 0 to represent a circle.

theorem circle_condition_iff (a : ℝ) :
  (∃ (x y : ℝ), (x - 1) ^ 2 + (y + 1) ^ 2 = 2 - a) ↔ (a < 2) :=
sorry

end circle_condition_iff_l1470_147042


namespace avg_nested_l1470_147011

def avg (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem avg_nested {x y z : ℕ} :
  avg (avg 2 3 1) (avg 4 1 0) 5 = 26 / 9 :=
by
  sorry

end avg_nested_l1470_147011


namespace infinite_series_value_l1470_147039

theorem infinite_series_value :
  ∑' n : ℕ, (n^3 + 4 * n^2 + 8 * n + 8) / (3^n * (n^3 + 5)) = 1 / 2 :=
by sorry

end infinite_series_value_l1470_147039


namespace derivative_at_one_l1470_147083

theorem derivative_at_one (f : ℝ → ℝ) (df : ℝ → ℝ) 
  (h₁ : ∀ x, f x = x^2) 
  (h₂ : ∀ x, df x = 2 * x) : 
  df 1 = 2 :=
by sorry

end derivative_at_one_l1470_147083


namespace TotalMarks_l1470_147022

def AmayaMarks (Arts Maths Music SocialStudies : ℕ) : Prop :=
  Maths = Arts - 20 ∧
  Maths = (9 * Arts) / 10 ∧
  Music = 70 ∧
  Music + 10 = SocialStudies

theorem TotalMarks (Arts Maths Music SocialStudies : ℕ) : 
  AmayaMarks Arts Maths Music SocialStudies → 
  (Arts + Maths + Music + SocialStudies = 530) :=
by
  sorry

end TotalMarks_l1470_147022


namespace value_of_expression_l1470_147054

theorem value_of_expression : 3 ^ (0 ^ (2 ^ 11)) + ((3 ^ 0) ^ 2) ^ 11 = 2 := by
  sorry

end value_of_expression_l1470_147054


namespace remaining_pictures_l1470_147031

theorem remaining_pictures (k m : ℕ) (d1 := 9 * k + 4) (d2 := 9 * m + 6) :
  (d1 * d2) % 9 = 6 → 9 - (d1 * d2 % 9) = 3 :=
by
  intro h
  sorry

end remaining_pictures_l1470_147031


namespace fraction_of_bikinis_or_trunks_l1470_147074

theorem fraction_of_bikinis_or_trunks (h_bikinis : Real := 0.38) (h_trunks : Real := 0.25) :
  h_bikinis + h_trunks = 0.63 :=
by
  sorry

end fraction_of_bikinis_or_trunks_l1470_147074


namespace rhombus_area_l1470_147098

theorem rhombus_area : 
  ∃ (d1 d2 : ℝ), (∀ (x : ℝ), x^2 - 14 * x + 48 = 0 → x = d1 ∨ x = d2) ∧
  (∀ (A : ℝ), A = d1 * d2 / 2 → A = 24) :=
by 
sorry

end rhombus_area_l1470_147098


namespace unattainable_y_l1470_147023

theorem unattainable_y (x : ℝ) (h : x ≠ -5/4) : ¬∃ y : ℝ, y = (2 - 3 * x) / (4 * x + 5) ∧ y = -3 / 4 :=
by
  sorry

end unattainable_y_l1470_147023


namespace int_999_column_is_C_l1470_147044

def column_of_int (n : ℕ) : String :=
  let m := n - 2
  match (m / 7 % 2, m % 7) with
  | (0, 0) => "A"
  | (0, 1) => "B"
  | (0, 2) => "C"
  | (0, 3) => "D"
  | (0, 4) => "E"
  | (0, 5) => "F"
  | (0, 6) => "G"
  | (1, 0) => "G"
  | (1, 1) => "F"
  | (1, 2) => "E"
  | (1, 3) => "D"
  | (1, 4) => "C"
  | (1, 5) => "B"
  | (1, 6) => "A"
  | _      => "Invalid"

theorem int_999_column_is_C : column_of_int 999 = "C" := by
  sorry

end int_999_column_is_C_l1470_147044


namespace mean_of_added_numbers_l1470_147041

theorem mean_of_added_numbers (mean_seven : ℝ) (mean_ten : ℝ) (x y z : ℝ)
    (h1 : mean_seven = 40)
    (h2 : mean_ten = 55) :
    (mean_seven * 7 + x + y + z) / 10 = mean_ten → (x + y + z) / 3 = 90 :=
by sorry

end mean_of_added_numbers_l1470_147041


namespace largest_n_base_conditions_l1470_147078

theorem largest_n_base_conditions :
  ∃ n: ℕ, n < 10000 ∧ 
  (∃ a: ℕ, 4^a ≤ n ∧ n < 4^(a+1) ∧ 4^a ≤ 3*n ∧ 3*n < 4^(a+1)) ∧
  (∃ b: ℕ, 8^b ≤ n ∧ n < 8^(b+1) ∧ 8^b ≤ 7*n ∧ 7*n < 8^(b+1)) ∧
  (∃ c: ℕ, 16^c ≤ n ∧ n < 16^(c+1) ∧ 16^c ≤ 15*n ∧ 15*n < 16^(c+1)) ∧
  n = 4369 :=
sorry

end largest_n_base_conditions_l1470_147078


namespace platform_length_is_correct_l1470_147055

noncomputable def length_of_platform (T : ℕ) (t_p t_s : ℕ) : ℕ :=
  let speed_of_train := T / t_s
  let distance_when_crossing_platform := speed_of_train * t_p
  distance_when_crossing_platform - T

theorem platform_length_is_correct :
  ∀ (T t_p t_s : ℕ),
  T = 300 → t_p = 33 → t_s = 18 →
  length_of_platform T t_p t_s = 250 :=
by
  intros T t_p t_s hT ht_p ht_s
  simp [length_of_platform, hT, ht_p, ht_s]
  sorry

end platform_length_is_correct_l1470_147055


namespace S_6_equals_12_l1470_147082

noncomputable def S (n : ℕ) : ℝ := sorry -- Definition for the sum of the first n terms

axiom geometric_sequence_with_positive_terms (n : ℕ) : S n > 0

axiom S_3 : S 3 = 3

axiom S_9 : S 9 = 39

theorem S_6_equals_12 : S 6 = 12 := by
  sorry

end S_6_equals_12_l1470_147082


namespace erasers_given_l1470_147021

theorem erasers_given (initial final : ℕ) (h1 : initial = 8) (h2 : final = 11) : (final - initial = 3) :=
by
  sorry

end erasers_given_l1470_147021


namespace total_widgets_sold_after_20_days_l1470_147005

-- Definition of the arithmetic sequence
def widgets_sold_on_day (n : ℕ) : ℕ :=
  2 * n - 1

-- Sum of the first n terms of the sequence
def sum_of_widgets_sold (n : ℕ) : ℕ :=
  n * (widgets_sold_on_day 1 + widgets_sold_on_day n) / 2

-- Prove that the total widgets sold after 20 days is 400
theorem total_widgets_sold_after_20_days : sum_of_widgets_sold 20 = 400 :=
by
  sorry

end total_widgets_sold_after_20_days_l1470_147005


namespace area_of_given_triangle_l1470_147025

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_given_triangle : 
  triangle_area (1, 1) (7, 1) (5, 3) = 6 :=
by
  -- the proof should go here
  sorry

end area_of_given_triangle_l1470_147025


namespace johnny_fishes_l1470_147076

theorem johnny_fishes
  (total_fishes : ℕ)
  (sony_ratio : ℕ)
  (total_is_40 : total_fishes = 40)
  (sony_is_4x_johnny : sony_ratio = 4)
  : ∃ (johnny_fishes : ℕ), johnny_fishes + sony_ratio * johnny_fishes = total_fishes ∧ johnny_fishes = 8 :=
by
  sorry

end johnny_fishes_l1470_147076


namespace burrito_calories_l1470_147047

theorem burrito_calories :
  ∀ (C : ℕ), 
  (10 * C = 6 * (250 - 50)) →
  C = 120 :=
by
  intros C h
  sorry

end burrito_calories_l1470_147047


namespace gcd_117_182_l1470_147014

theorem gcd_117_182 : Int.gcd 117 182 = 13 := 
by 
  sorry

end gcd_117_182_l1470_147014


namespace smallest_m_l1470_147026

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - ⌊x⌋

noncomputable def f (x : ℝ) : ℝ :=
  abs (3 * fractional_part x - 1.5)

theorem smallest_m (m : ℤ) (h1 : ∀ x : ℝ, m^2 * f (x * f x) = x → True) : ∃ m, m = 8 :=
by
  have h2 : ∀ m : ℤ, (∃ (s : ℕ), s ≥ 1008 ∧ (m^2 * abs (3 * fractional_part (s * abs (1.5 - 3 * (fractional_part s) )) - 1.5) = s)) → m = 8
  {
    sorry
  }
  sorry

end smallest_m_l1470_147026


namespace number_halfway_between_l1470_147071

theorem number_halfway_between :
  ∃ x : ℚ, x = (1/12 + 1/14) / 2 ∧ x = 13 / 168 :=
sorry

end number_halfway_between_l1470_147071


namespace pentagon_rectangle_ratio_l1470_147040

theorem pentagon_rectangle_ratio :
  let p : ℝ := 60  -- Perimeter of both the pentagon and the rectangle
  let length_side_pentagon : ℝ := 12
  let w : ℝ := 10
  p / 5 = length_side_pentagon ∧ p/6 = w ∧ length_side_pentagon / w = 6/5 :=
sorry

end pentagon_rectangle_ratio_l1470_147040


namespace top_card_is_11_l1470_147030

-- Define the initial configuration of cards
def initial_array : List (List Nat) := [
  [1, 2, 3, 4, 5, 6],
  [7, 8, 9, 10, 11, 12],
  [13, 14, 15, 16, 17, 18]
]

-- Perform the described sequence of folds
def fold1 (arr : List (List Nat)) : List (List Nat) := [
  [3, 4, 5, 6],
  [9, 10, 11, 12],
  [15, 16, 17, 18],
  [1, 2],
  [7, 8],
  [13, 14]
]

def fold2 (arr : List (List Nat)) : List (List Nat) := [
  [5, 6],
  [11, 12],
  [17, 18],
  [3, 4, 1, 2],
  [9, 10, 7, 8],
  [15, 16, 13, 14]
]

def fold3 (arr : List (List Nat)) : List (List Nat) := [
  [11, 12, 7, 8],
  [17, 18, 13, 14],
  [5, 6, 1, 2],
  [9, 10, 3, 4],
  [15, 16, 9, 10]
]

-- Define the final array after all the folds
def final_array := fold3 (fold2 (fold1 initial_array))

-- Statement to be proven
theorem top_card_is_11 : (final_array.head!.head!) = 11 := 
  by
    sorry -- Proof to be filled in

end top_card_is_11_l1470_147030


namespace unique_pair_exists_l1470_147060

theorem unique_pair_exists :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  (a + b + (Nat.gcd a b)^2 = Nat.lcm a b) ∧
  (Nat.lcm a b = 2 * Nat.lcm (a - 1) b) ∧
  (a, b) = (6, 15) :=
sorry

end unique_pair_exists_l1470_147060


namespace cost_of_popsicle_sticks_l1470_147065

theorem cost_of_popsicle_sticks
  (total_money : ℕ)
  (cost_of_molds : ℕ)
  (cost_per_bottle : ℕ)
  (popsicles_per_bottle : ℕ)
  (sticks_used : ℕ)
  (sticks_left : ℕ)
  (number_of_sticks : ℕ)
  (remaining_money : ℕ) :
  total_money = 10 →
  cost_of_molds = 3 →
  cost_per_bottle = 2 →
  popsicles_per_bottle = 20 →
  sticks_left = 40 →
  number_of_sticks = 100 →
  remaining_money = total_money - cost_of_molds - (sticks_used / popsicles_per_bottle * cost_per_bottle) →
  sticks_used = number_of_sticks - sticks_left →
  remaining_money = 1 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end cost_of_popsicle_sticks_l1470_147065


namespace unique_valid_configuration_l1470_147052

-- Define the conditions: a rectangular array of chairs organized in rows and columns such that
-- each row contains the same number of chairs as every other row, each column contains the
-- same number of chairs as every other column, with at least two chairs in every row and column.
def valid_array_configuration (rows cols : ℕ) : Prop :=
  2 ≤ rows ∧ 2 ≤ cols ∧ rows * cols = 49

-- The theorem statement: determine how many valid arrays are possible given the conditions.
theorem unique_valid_configuration : ∃! (rows cols : ℕ), valid_array_configuration rows cols :=
sorry

end unique_valid_configuration_l1470_147052


namespace range_of_a_in_third_quadrant_l1470_147086

theorem range_of_a_in_third_quadrant (a : ℝ) :
  let Z_re := a^2 - 2*a
  let Z_im := a^2 - a - 2
  (Z_re < 0 ∧ Z_im < 0) → 0 < a ∧ a < 2 :=
by
  sorry

end range_of_a_in_third_quadrant_l1470_147086


namespace find_integer_solutions_l1470_147004

theorem find_integer_solutions :
  {n : ℤ | n + 2 ∣ n^2 + 3} = {-9, -3, -1, 5} :=
  sorry

end find_integer_solutions_l1470_147004


namespace solve_inequality_l1470_147046

theorem solve_inequality (a : ℝ) :
  (a < 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ a < x ∧ x < 1 - a) ∨
  (a > 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ 1 - a < x ∧ x < a) ∨
  (a = 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ false) :=
sorry

end solve_inequality_l1470_147046


namespace min_value_of_expression_min_value_achieved_at_l1470_147077

theorem min_value_of_expression (x : ℝ) (hx : 0 < x) : 
  3 * Real.sqrt x + 4 / (x^2) ≥ 4 * 4^(1/5) :=
sorry

theorem min_value_achieved_at (x : ℝ) (hx : 0 < x) (h : x = 4^(2/5)) :
  3 * Real.sqrt x + 4 / (x^2) = 4 * 4^(1/5) :=
sorry

end min_value_of_expression_min_value_achieved_at_l1470_147077


namespace original_four_digit_number_l1470_147096

theorem original_four_digit_number : 
  ∃ x y z: ℕ, (x = 1 ∧ y = 9 ∧ z = 7 ∧ 1000 * x + 100 * y + 10 * z + y = 1979) ∧ 
  (1000 * y + 100 * z + 10 * y + x - (1000 * x + 100 * y + 10 * z + y) = 7812) ∧ 
  (1000 * y + 100 * z + 10 * y + x < 10000 ∧ 1000 * x + 100 * y + 10 * z + y < 10000) := 
sorry

end original_four_digit_number_l1470_147096


namespace stock_investment_net_increase_l1470_147064

theorem stock_investment_net_increase :
  ∀ (initial_investment : ℝ)
    (increase_first_year : ℝ)
    (decrease_second_year : ℝ)
    (increase_third_year : ℝ),
  initial_investment = 100 → 
  increase_first_year = 0.60 → 
  decrease_second_year = 0.30 → 
  increase_third_year = 0.20 → 
  ((initial_investment * (1 + increase_first_year)) * (1 - decrease_second_year)) * (1 + increase_third_year) - initial_investment = 34.40 :=
by 
  intros initial_investment increase_first_year decrease_second_year increase_third_year 
  intros h_initial_investment h_increase_first_year h_decrease_second_year h_increase_third_year 
  rw [h_initial_investment, h_increase_first_year, h_decrease_second_year, h_increase_third_year]
  sorry

end stock_investment_net_increase_l1470_147064


namespace man_twice_son_age_l1470_147087

theorem man_twice_son_age (S M Y : ℕ) (h1 : S = 27) (h2 : M = S + 29) (h3 : M + Y = 2 * (S + Y)) : Y = 2 := 
by sorry

end man_twice_son_age_l1470_147087


namespace triangle_circle_distance_l1470_147080

open Real

theorem triangle_circle_distance 
  (DE DF EF : ℝ)
  (hDE : DE = 12) (hDF : DF = 16) (hEF : EF = 20) :
  let s := (DE + DF + EF) / 2
  let K := sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let ra := K / (s - EF)
  let DP := s - DF
  let DQ := s
  let DI := sqrt (DP^2 + r^2)
  let DE := sqrt (DQ^2 + ra^2)
  let distance := DE - DI
  distance = 24 * sqrt 2 - 4 * sqrt 10 :=
by
  sorry

end triangle_circle_distance_l1470_147080


namespace standard_equation_line_BC_fixed_point_l1470_147007

section EllipseProof

open Real

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Conditions from the problem
axiom a_gt_b_gt_0 : ∀ (a b : ℝ), a > b → b > 0
axiom passes_through_point : ∀ (a b x y : ℝ), ellipse a b x y → (x = 1 ∧ y = sqrt 2 / 2)
axiom has_eccentricity : ∀ (a b c : ℝ), c / a = sqrt 2 / 2 → c^2 = a^2 - b^2 → b = 1

-- The standard equation of the ellipse
theorem standard_equation (a b : ℝ) (x y : ℝ) :
  a = sqrt 2 → b = 1 → ellipse a b x y → ellipse (sqrt 2) 1 x y :=
sorry

-- Prove that BC always passes through a fixed point
theorem line_BC_fixed_point (a b x1 x2 y1 y2 : ℝ) :
  a = sqrt 2 → b = 1 → 
  ellipse a b x1 y1 → ellipse a b x2 y2 →
  y1 = -y2 → x1 ≠ x2 → (-1, 0) = (-1, 0) →
  ∃ (k : ℝ) (x : ℝ), x = -2 ∧ y = 0 :=
sorry

end EllipseProof

end standard_equation_line_BC_fixed_point_l1470_147007


namespace area_within_fence_l1470_147094

def length_rectangle : ℕ := 15
def width_rectangle : ℕ := 12
def side_cutout_square : ℕ := 3

theorem area_within_fence : (length_rectangle * width_rectangle) - (side_cutout_square * side_cutout_square) = 171 := by
  sorry

end area_within_fence_l1470_147094


namespace total_chairs_all_together_l1470_147000

-- Definitions of given conditions
def rows := 7
def chairs_per_row := 12
def extra_chairs := 11

-- Main statement we want to prove
theorem total_chairs_all_together : 
  (rows * chairs_per_row + extra_chairs = 95) := 
by
  sorry

end total_chairs_all_together_l1470_147000


namespace combined_weight_l1470_147097

theorem combined_weight (x y z : ℕ) (h1 : x + z = 78) (h2 : x + y = 69) (h3 : y + z = 137) : x + y + z = 142 :=
by
  -- Intermediate steps or any additional lemmas could go here
sorry

end combined_weight_l1470_147097


namespace quadratic_inequality_solution_l1470_147072

theorem quadratic_inequality_solution :
  ∀ (x : ℝ), x^2 - 9 * x + 14 ≤ 0 → 2 ≤ x ∧ x ≤ 7 :=
by
  intros x h
  sorry

end quadratic_inequality_solution_l1470_147072


namespace reciprocal_of_neg4_is_neg_one_fourth_l1470_147092

theorem reciprocal_of_neg4_is_neg_one_fourth (x : ℝ) (h : x * -4 = 1) : x = -1/4 := 
by 
  sorry

end reciprocal_of_neg4_is_neg_one_fourth_l1470_147092


namespace number_of_real_solutions_is_one_l1470_147050

noncomputable def num_real_solutions (a b c d : ℝ) : ℕ :=
  let x := Real.sin (a + b + c)
  let y := Real.sin (b + c + d)
  let z := Real.sin (c + d + a)
  let w := Real.sin (d + a + b)
  if (a + b + c + d) % 360 = 0 then 1 else 0

theorem number_of_real_solutions_is_one (a b c d : ℝ) (h : (a + b + c + d) % 360 = 0) :
  num_real_solutions a b c d = 1 :=
by
  sorry

end number_of_real_solutions_is_one_l1470_147050


namespace find_breadth_of_wall_l1470_147033

theorem find_breadth_of_wall
  (b h l V : ℝ)
  (h1 : V = 12.8)
  (h2 : h = 5 * b)
  (h3 : l = 8 * h) :
  b = 0.4 :=
by
  sorry

end find_breadth_of_wall_l1470_147033


namespace probability_neither_red_blue_purple_l1470_147066

def total_balls : ℕ := 240
def white_balls : ℕ := 60
def green_balls : ℕ := 70
def yellow_balls : ℕ := 45
def red_balls : ℕ := 35
def blue_balls : ℕ := 20
def purple_balls : ℕ := 10

theorem probability_neither_red_blue_purple :
  (total_balls - (red_balls + blue_balls + purple_balls)) / total_balls = 35 / 48 := 
by 
  /- Proof details are not necessary -/
  sorry

end probability_neither_red_blue_purple_l1470_147066


namespace binary_equals_octal_l1470_147012

-- Define that 1001101 in binary is a specific integer
def binary_value : ℕ := 0b1001101

-- Define that 115 in octal is a specific integer
def octal_value : ℕ := 0o115

-- State the theorem we need to prove
theorem binary_equals_octal : binary_value = octal_value :=
  by sorry

end binary_equals_octal_l1470_147012


namespace system_solution_unique_l1470_147048

theorem system_solution_unique (w x y z : ℝ) (h1 : w + x + y + z = 12)
  (h2 : w * x * y * z = w * x + w * y + w * z + x * y + x * z + y * z + 27) :
  w = 3 ∧ x = 3 ∧ y = 3 ∧ z = 3 := 
sorry

end system_solution_unique_l1470_147048


namespace coordinates_in_second_quadrant_l1470_147051

section 
variable (x y : ℝ)
variable (hx : x = -7)
variable (hy : y = 4)
variable (quadrant : x < 0 ∧ y > 0)
variable (distance_x : |y| = 4)
variable (distance_y : |x| = 7)

theorem coordinates_in_second_quadrant :
  (x, y) = (-7, 4) := by
  sorry
end

end coordinates_in_second_quadrant_l1470_147051


namespace min_h_for_circle_l1470_147073

theorem min_h_for_circle (h : ℝ) :
  (∀ x y : ℝ, (x - h)^2 + (y - 1)^2 = 1 → x + y + 1 ≥ 0) →
  h = Real.sqrt 2 - 2 :=
sorry

end min_h_for_circle_l1470_147073


namespace car_2_speed_proof_l1470_147018

noncomputable def car_1_speed : ℝ := 30
noncomputable def car_1_start_time : ℝ := 9
noncomputable def car_2_start_delay : ℝ := 10 / 60
noncomputable def catch_up_time : ℝ := 10.5
noncomputable def car_2_start_time : ℝ := car_1_start_time + car_2_start_delay
noncomputable def travel_duration : ℝ := catch_up_time - car_2_start_time
noncomputable def car_1_head_start_distance : ℝ := car_1_speed * car_2_start_delay
noncomputable def car_1_travel_distance : ℝ := car_1_speed * travel_duration
noncomputable def total_distance : ℝ := car_1_head_start_distance + car_1_travel_distance
noncomputable def car_2_speed : ℝ := total_distance / travel_duration

theorem car_2_speed_proof : car_2_speed = 33.75 := 
by 
  sorry

end car_2_speed_proof_l1470_147018


namespace focus_of_parabola_l1470_147024

theorem focus_of_parabola (x y : ℝ) : (y^2 = 4 * x) → (x = 2 ∧ y = 0) :=
by
  sorry

end focus_of_parabola_l1470_147024


namespace last_three_digits_of_16_pow_128_l1470_147002

theorem last_three_digits_of_16_pow_128 : (16 ^ 128) % 1000 = 721 := 
by
  sorry

end last_three_digits_of_16_pow_128_l1470_147002


namespace find_a3_l1470_147070

theorem find_a3 (a : ℕ → ℕ) (h₁ : a 1 = 2)
  (h₂ : ∀ n, (1 + 2 * a (n + 1)) = (1 + 2 * a n) + 1) : a 3 = 3 :=
by
  -- This is where the proof would go, but we'll leave it as sorry for now.
  sorry

end find_a3_l1470_147070


namespace intersection_A_B_l1470_147093

def A (x : ℝ) : Prop := x > 3
def B (x : ℝ) : Prop := x ≤ 4

theorem intersection_A_B : {x | A x} ∩ {x | B x} = {x | 3 < x ∧ x ≤ 4} :=
by
  sorry

end intersection_A_B_l1470_147093


namespace sqrt_square_eq_14_l1470_147059

theorem sqrt_square_eq_14 : Real.sqrt (14 ^ 2) = 14 :=
by
  sorry

end sqrt_square_eq_14_l1470_147059


namespace find_angle_B_l1470_147017

theorem find_angle_B 
  (A B : ℝ)
  (h1 : B + A = 90)
  (h2 : B = 4 * A) : 
  B = 144 :=
by
  sorry

end find_angle_B_l1470_147017


namespace max_consecutive_sum_l1470_147045

theorem max_consecutive_sum (n : ℕ) : 
  (∀ (n : ℕ), (n*(n + 1))/2 ≤ 400 → n ≤ 27) ∧ ((27*(27 + 1))/2 ≤ 400) :=
by
  sorry

end max_consecutive_sum_l1470_147045


namespace find_sale_in_third_month_l1470_147034

def sale_in_first_month := 5700
def sale_in_second_month := 8550
def sale_in_fourth_month := 3850
def sale_in_fifth_month := 14045
def average_sale := 7800
def num_months := 5
def total_sales := average_sale * num_months

theorem find_sale_in_third_month (X : ℕ) 
  (H : total_sales = sale_in_first_month + sale_in_second_month + X + sale_in_fourth_month + sale_in_fifth_month) :
  X = 9455 :=
by
  sorry

end find_sale_in_third_month_l1470_147034


namespace complement_intersection_l1470_147067

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4, 5}) :
  U \ (A ∩ B) = {1, 2, 4, 5} :=
by
  sorry

end complement_intersection_l1470_147067


namespace max_A_excircle_area_ratio_max_A_excircle_area_ratio_eq_l1470_147038

noncomputable def A_excircle_area_ratio (α : Real) (s : Real) : Real :=
  0.5 * Real.sin α

theorem max_A_excircle_area_ratio (α : Real) (s : Real) : (A_excircle_area_ratio α s) ≤ 0.5 :=
by
  sorry

theorem max_A_excircle_area_ratio_eq (s : Real) : 
  (A_excircle_area_ratio (Real.pi / 2) s) = 0.5 :=
by
  sorry

end max_A_excircle_area_ratio_max_A_excircle_area_ratio_eq_l1470_147038


namespace JimAgeInXYears_l1470_147006

-- Definitions based on conditions
def TomCurrentAge := 37
def JimsAge7YearsAgo := 5 + (TomCurrentAge - 7) / 2

-- We introduce a variable X to represent the number of years into the future.
variable (X : ℕ)

-- Lean 4 statement to prove that Jim will be 27 + X years old in X years from now.
theorem JimAgeInXYears : JimsAge7YearsAgo + 7 + X = 27 + X := 
by
  sorry

end JimAgeInXYears_l1470_147006


namespace solve_for_x_l1470_147029

theorem solve_for_x : ∃ x : ℚ, x + 5/6 = 7/18 + 1/2 ∧ x = -7/18 := by
  sorry

end solve_for_x_l1470_147029


namespace smallest_possible_value_m_l1470_147058

theorem smallest_possible_value_m (r y b : ℕ) (h : 16 * r = 18 * y ∧ 18 * y = 20 * b) : 
  ∃ m : ℕ, 30 * m = 16 * r ∧ 30 * m = 720 ∧ m = 24 :=
by {
  sorry
}

end smallest_possible_value_m_l1470_147058


namespace monkey_climb_ladder_l1470_147016

theorem monkey_climb_ladder (n : ℕ) 
  (h1 : ∀ k, (k % 18 = 0 → (k - 18 + 10) % 26 = 8))
  (h2 : ∀ m, (m % 10 = 0 → (m - 10 + 18) % 26 = 18))
  (h3 : ∀ l, (l % 18 = 0 ∧ l % 10 = 0 → l = 0 ∨ l = 26)):
  n = 26 :=
by
  sorry

end monkey_climb_ladder_l1470_147016


namespace min_C2_D2_at_36_l1470_147027

noncomputable def min_value_C2_D2 (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 3) : ℝ :=
  let C := (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))
  let D := (Real.sqrt (x + 1) + Real.sqrt (y + 2) + Real.sqrt (z + 3))
  C^2 - D^2

theorem min_C2_D2_at_36 (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 3) : 
  min_value_C2_D2 x y z hx hy hz = 36 :=
sorry

end min_C2_D2_at_36_l1470_147027


namespace find_a100_l1470_147099

noncomputable def S (k : ℝ) (n : ℤ) : ℝ := k * (n ^ 2) + n
noncomputable def a (k : ℝ) (n : ℤ) : ℝ := S k n - S k (n - 1)

theorem find_a100 (k : ℝ) 
  (h1 : a k 10 = 39) :
  a k 100 = 399 :=
sorry

end find_a100_l1470_147099


namespace minister_can_organize_traffic_l1470_147057

-- Definition of cities and roads
structure City (α : Type) :=
(road : α → α → Prop)

-- Defining the Minister's goal
def organize_traffic {α : Type} (c : City α) (num_days : ℕ) : Prop :=
∀ x y : α, c.road x y → num_days ≤ 214

theorem minister_can_organize_traffic :
  ∃ (c : City ℕ) (num_days : ℕ), (num_days ≤ 214 ∧ organize_traffic c num_days) :=
by {
  sorry
}

end minister_can_organize_traffic_l1470_147057


namespace concert_attendance_l1470_147079

/-
Mrs. Hilt went to a concert. A total of some people attended the concert. 
The next week, she went to a second concert, which had 119 more people in attendance. 
There were 66018 people at the second concert. 
How many people attended the first concert?
-/

variable (first_concert second_concert : ℕ)

theorem concert_attendance (h1 : second_concert = first_concert + 119)
    (h2 : second_concert = 66018) : first_concert = 65899 := 
by
  sorry

end concert_attendance_l1470_147079


namespace number_of_ants_proof_l1470_147019

-- Define the conditions
def width_ft := 500
def length_ft := 600
def ants_per_sq_inch := 4
def inches_per_foot := 12

-- Define the calculation to get the number of ants
def number_of_ants (width_ft : ℕ) (length_ft : ℕ) (ants_per_sq_inch : ℕ) (inches_per_foot : ℕ) :=
  let width_inch := width_ft * inches_per_foot
  let length_inch := length_ft * inches_per_foot
  let area_sq_inch := width_inch * length_inch
  ants_per_sq_inch * area_sq_inch

-- Prove the number of ants is approximately 173 million
theorem number_of_ants_proof :
  number_of_ants width_ft length_ft ants_per_sq_inch inches_per_foot = 172800000 :=
by
  sorry

end number_of_ants_proof_l1470_147019


namespace rational_function_eq_l1470_147036

theorem rational_function_eq (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 :=
by sorry

end rational_function_eq_l1470_147036


namespace tenth_term_is_correct_l1470_147043

-- Define the first term and common difference for the sequence
def a1 : ℚ := 1 / 2
def d : ℚ := 1 / 3

-- The property that defines the n-th term of the arithmetic sequence
def a (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Statement to prove that the tenth term in the arithmetic sequence is 7 / 2
theorem tenth_term_is_correct : a 10 = 7 / 2 := 
by 
  -- To be filled in with the proof later
  sorry

end tenth_term_is_correct_l1470_147043


namespace chord_length_of_tangent_l1470_147003

theorem chord_length_of_tangent (R r : ℝ) (h : R^2 - r^2 = 25) : ∃ c : ℝ, c = 10 :=
by
  sorry

end chord_length_of_tangent_l1470_147003


namespace ratio_d_s_proof_l1470_147032

noncomputable def ratio_d_s (n : ℕ) (s d : ℝ) : ℝ :=
  d / s

theorem ratio_d_s_proof : ∀ (n : ℕ) (s d : ℝ), 
  (n = 30) → 
  ((n ^ 2 * s ^ 2) / (n * s + 2 * n * d) ^ 2 = 0.81) → 
  ratio_d_s n s d = 1 / 18 :=
by
  intros n s d h_n h_area
  sorry

end ratio_d_s_proof_l1470_147032


namespace initial_fish_count_l1470_147028

-- Definitions based on the given conditions
def Fish_given : ℝ := 22.0
def Fish_now : ℝ := 25.0

-- The goal is to prove the initial number of fish Mrs. Sheridan had.
theorem initial_fish_count : (Fish_given + Fish_now) = 47.0 := by
  sorry

end initial_fish_count_l1470_147028


namespace number_of_tiles_per_row_l1470_147091

-- Definitions of conditions
def area : ℝ := 320
def length : ℝ := 16
def tile_size : ℝ := 1

-- Theorem statement
theorem number_of_tiles_per_row : (area / length) / tile_size = 20 := by
  sorry

end number_of_tiles_per_row_l1470_147091


namespace dolly_dresses_shipment_l1470_147053

variable (T : ℕ)

/-- Given that 70% of the total number of Dolly Dresses in the shipment is equal to 140,
    prove that the total number of Dolly Dresses in the shipment is 200. -/
theorem dolly_dresses_shipment (h : (7 * T) / 10 = 140) : T = 200 :=
sorry

end dolly_dresses_shipment_l1470_147053


namespace audi_crossing_intersection_between_17_and_18_l1470_147069

-- Given conditions:
-- Two cars, an Audi and a BMW, are moving along two intersecting roads at equal constant speeds.
-- At both 17:00 and 18:00, the BMW was twice as far from the intersection as the Audi.
-- Let the distance of Audi from the intersection at 17:00 be x and BMW's distance be 2x.
-- Both vehicles travel at a constant speed v.

noncomputable def car_position (initial_distance : ℝ) (velocity : ℝ) (time_elapsed : ℝ) : ℝ :=
  initial_distance + velocity * time_elapsed

theorem audi_crossing_intersection_between_17_and_18 (x v : ℝ) :
  ∃ t : ℝ, (t = 15 ∨ t = 45) ∧
    car_position x (-v) (t/60) = 0 ∧ car_position (2 * x) (-v) (t/60) = 2 * car_position x (-v) (1 - t/60) :=
sorry

end audi_crossing_intersection_between_17_and_18_l1470_147069


namespace percentage_calculation_l1470_147010

/-- If x % of 375 equals 5.4375, then x % equals 1.45 %. -/
theorem percentage_calculation (x : ℝ) (h : x / 100 * 375 = 5.4375) : x = 1.45 := 
sorry

end percentage_calculation_l1470_147010


namespace distance_focus_directrix_l1470_147063

theorem distance_focus_directrix (y x : ℝ) (h : y^2 = 2 * x) : x = 1 := 
by 
  sorry

end distance_focus_directrix_l1470_147063
