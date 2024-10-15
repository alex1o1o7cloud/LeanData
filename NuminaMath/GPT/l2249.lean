import Mathlib

namespace NUMINAMATH_GPT_sum_of_lengths_of_edges_l2249_224972

theorem sum_of_lengths_of_edges (s h : ℝ) 
(volume_eq : s^2 * h = 576) 
(surface_area_eq : 4 * s * h = 384) : 
8 * s + 4 * h = 112 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_lengths_of_edges_l2249_224972


namespace NUMINAMATH_GPT_solve_equation_l2249_224948

theorem solve_equation (x : ℝ) (h : (2 / (x - 3) = 3 / (x - 6))) : x = -3 :=
sorry

end NUMINAMATH_GPT_solve_equation_l2249_224948


namespace NUMINAMATH_GPT_driving_time_constraint_l2249_224992

variable (x y z : ℝ)

theorem driving_time_constraint (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) :
  3 + (60 / x) + (90 / y) + (200 / z) ≤ 10 :=
sorry

end NUMINAMATH_GPT_driving_time_constraint_l2249_224992


namespace NUMINAMATH_GPT_cubic_root_equality_l2249_224968

theorem cubic_root_equality (a b c : ℝ) (h1 : a + b + c = 12) (h2 : a * b + b * c + c * a = 14) (h3 : a * b * c = -3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 268 / 9 := 
by
  sorry

end NUMINAMATH_GPT_cubic_root_equality_l2249_224968


namespace NUMINAMATH_GPT_impossible_to_have_same_number_of_each_color_l2249_224919

-- Define the initial number of coins Laura has
def initial_green : Nat := 1

-- Define the net gain in coins per transaction
def coins_gain_per_transaction : Nat := 4

-- Define a function that calculates the total number of coins after n transactions
def total_coins (n : Nat) : Nat :=
  initial_green + n * coins_gain_per_transaction

-- Define the theorem to prove that it's impossible for Laura to have the same number of red and green coins
theorem impossible_to_have_same_number_of_each_color :
  ¬ ∃ n : Nat, ∃ red green : Nat, red = green ∧ total_coins n = red + green := by
  sorry

end NUMINAMATH_GPT_impossible_to_have_same_number_of_each_color_l2249_224919


namespace NUMINAMATH_GPT_mathematician_correctness_l2249_224922

theorem mathematician_correctness (box1_white1 box1_total1 box2_white1 box2_total1 : ℕ)
                                  (prob1 prob2 : ℚ) :
  (box1_white1 = 4 ∧ box1_total1 = 7 ∧ box2_white1 = 3 ∧ box2_total1 = 5 ∧ prob1 = (4:ℚ) / 7 ∧ prob2 = (3:ℚ) / 5) →
  (box1_white2 = 8 ∧ box1_total2 = 14 ∧ box2_white2 = 3 ∧ box2_total2 = 5 ∧ prob1 = (8:ℚ) / 14 ∧ prob2 = (3:ℚ) / 5) →
  (prob1 < (7:ℚ) / 12 ∧ prob2 > (11:ℚ)/19) →
  ¬((19:ℚ)/35 > (4:ℚ)/7 ∧ (19:ℚ)/35 < (3:ℚ)/5) :=
by
  sorry

end NUMINAMATH_GPT_mathematician_correctness_l2249_224922


namespace NUMINAMATH_GPT_find_g_l2249_224900

def nabla (g h : ℤ) : ℤ := g ^ 2 - h ^ 2

theorem find_g (g : ℤ) (h : ℤ)
  (H1 : 0 < g)
  (H2 : nabla g 6 = 45) :
  g = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_g_l2249_224900


namespace NUMINAMATH_GPT_sequence_ineq_l2249_224916

theorem sequence_ineq (a : ℕ → ℝ) (h1 : a 1 = 15) 
  (h2 : ∀ n, a (n + 1) = a n - 2 / 3) 
  (hk : a k * a (k + 1) < 0) : k = 23 :=
sorry

end NUMINAMATH_GPT_sequence_ineq_l2249_224916


namespace NUMINAMATH_GPT_countColorings_l2249_224931

-- Defining the function that counts the number of valid colorings
def validColorings (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 3 * 2^n - 2

-- Theorem specifying the number of colorings of the grid of length n
theorem countColorings (n : ℕ) : validColorings n = 3 * 2^n - 2 :=
by
  sorry

end NUMINAMATH_GPT_countColorings_l2249_224931


namespace NUMINAMATH_GPT_cot_trig_identity_l2249_224979

noncomputable def cot (x : Real) : Real :=
  Real.cos x / Real.sin x

theorem cot_trig_identity (a b c α β γ : Real) 
  (habc : a^2 + b^2 = 2021 * c^2) 
  (hα : α = Real.arcsin (a / c)) 
  (hβ : β = Real.arcsin (b / c)) 
  (hγ : γ = Real.arccos ((2021 * c^2 - a^2 - b^2) / (2 * 2021 * c^2))) 
  (h_triangle : a^2 = b^2 + c^2 - 2 * b * c * Real.cos α) :
  cot α / (cot β + cot γ) = 1010 :=
by
  sorry

end NUMINAMATH_GPT_cot_trig_identity_l2249_224979


namespace NUMINAMATH_GPT_common_chord_l2249_224944

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + x - 2*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- The common chord is the line where both circle equations are satisfied
theorem common_chord (x y : ℝ) : circle1 x y ∧ circle2 x y → x - 2*y + 5 = 0 :=
sorry

end NUMINAMATH_GPT_common_chord_l2249_224944


namespace NUMINAMATH_GPT_scientific_notation_conversion_l2249_224956

theorem scientific_notation_conversion :
  0.000037 = 3.7 * 10^(-5) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_conversion_l2249_224956


namespace NUMINAMATH_GPT_solve_for_d_l2249_224993

theorem solve_for_d (n k c d : ℝ) (h₁ : n = 2 * k * c * d / (c + d)) (h₂ : 2 * k * c ≠ n) :
  d = n * c / (2 * k * c - n) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_d_l2249_224993


namespace NUMINAMATH_GPT_least_integer_square_condition_l2249_224907

theorem least_integer_square_condition (x : ℤ) (h : x^2 = 3 * x + 36) : x = -6 :=
by sorry

end NUMINAMATH_GPT_least_integer_square_condition_l2249_224907


namespace NUMINAMATH_GPT_find_n_value_l2249_224906

theorem find_n_value (AB AC n m : ℕ) (h1 : AB = 33) (h2 : AC = 21) (h3 : AD = m) (h4 : DE = m) (h5 : EC = m) (h6 : BC = n) : 
  ∃ m : ℕ, m > 7 ∧ m < 21 ∧ n = 30 := 
by sorry

end NUMINAMATH_GPT_find_n_value_l2249_224906


namespace NUMINAMATH_GPT_resulting_shape_is_cone_l2249_224978

-- Assume we have a right triangle
structure right_triangle (α β γ : ℝ) : Prop :=
  (is_right : γ = π / 2)
  (sum_of_angles : α + β + γ = π)
  (acute_angles : α < π / 2 ∧ β < π / 2)

-- Assume we are rotating around one of the legs
def rotate_around_leg (α β : ℝ) : Prop := sorry

theorem resulting_shape_is_cone (α β γ : ℝ) (h : right_triangle α β γ) :
  ∃ (shape : Type), rotate_around_leg α β → shape = cone :=
by
  sorry

end NUMINAMATH_GPT_resulting_shape_is_cone_l2249_224978


namespace NUMINAMATH_GPT_correct_calculation_l2249_224980

theorem correct_calculation (m n : ℤ) :
  (m^2 * m^3 ≠ m^6) ∧
  (- (m - n) = -m + n) ∧
  (m * (m + n) ≠ m^2 + n) ∧
  ((m + n)^2 ≠ m^2 + n^2) :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l2249_224980


namespace NUMINAMATH_GPT_falcon_speed_correct_l2249_224921

-- Definitions based on conditions
def eagle_speed : ℕ := 15
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30
def total_distance : ℕ := 248
def time_hours : ℕ := 2

-- Variables representing the unknown falcon speed
variable {falcon_speed : ℕ}

-- The Lean statement to prove
theorem falcon_speed_correct 
  (h : 2 * falcon_speed + (eagle_speed * time_hours) + (pelican_speed * time_hours) + (hummingbird_speed * time_hours) = total_distance) :
  falcon_speed = 46 :=
sorry

end NUMINAMATH_GPT_falcon_speed_correct_l2249_224921


namespace NUMINAMATH_GPT_sufficient_condition_for_inequality_l2249_224957

open Real

theorem sufficient_condition_for_inequality (a : ℝ) (h : 0 < a ∧ a < 1 / 5) : 1 / a > 3 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_inequality_l2249_224957


namespace NUMINAMATH_GPT_prime_between_40_50_largest_prime_lt_100_l2249_224974

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def between (n m k : ℕ) : Prop := n < k ∧ k < m

theorem prime_between_40_50 :
  {x : ℕ | between 40 50 x ∧ isPrime x} = {41, 43, 47} :=
sorry

theorem largest_prime_lt_100 :
  ∃ p : ℕ, isPrime p ∧ p < 100 ∧ ∀ q : ℕ, isPrime q ∧ q < 100 → q ≤ p :=
sorry

end NUMINAMATH_GPT_prime_between_40_50_largest_prime_lt_100_l2249_224974


namespace NUMINAMATH_GPT_no_solution_l2249_224912

theorem no_solution : ¬∃ x : ℝ, x^3 - 8*x^2 + 16*x - 32 / (x - 2) < 0 := by
  sorry

end NUMINAMATH_GPT_no_solution_l2249_224912


namespace NUMINAMATH_GPT_find_m_from_arithmetic_sequence_l2249_224985

theorem find_m_from_arithmetic_sequence (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -4) (h2 : S m = 0) (h3 : S (m + 1) = 6) : m = 5 := by
  sorry

end NUMINAMATH_GPT_find_m_from_arithmetic_sequence_l2249_224985


namespace NUMINAMATH_GPT_preimage_of_3_1_is_2_1_l2249_224990

-- Definition of the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

-- The Lean theorem statement
theorem preimage_of_3_1_is_2_1 : ∃ (x y : ℝ), f x y = (3, 1) ∧ (x = 2 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_preimage_of_3_1_is_2_1_l2249_224990


namespace NUMINAMATH_GPT_smallest_x_l2249_224981

theorem smallest_x (x y : ℝ) (h1 : 4 < x) (h2 : x < 8) (h3 : 8 < y) (h4 : y < 12) (h5 : y - x = 7) :
  ∃ ε > 0, x = 4 + ε :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_l2249_224981


namespace NUMINAMATH_GPT_largest_natural_S_n_gt_zero_l2249_224925

noncomputable def S_n (n : ℕ) : ℤ :=
  let a1 := 9
  let d := -2
  n * (2 * a1 + (n - 1) * d) / 2

theorem largest_natural_S_n_gt_zero
  (a_2 : ℤ) (a_4 : ℤ)
  (h1 : a_2 = 7) (h2 : a_4 = 3) :
  ∃ n : ℕ, S_n n > 0 ∧ ∀ m : ℕ, m > n → S_n m ≤ 0 := 
sorry

end NUMINAMATH_GPT_largest_natural_S_n_gt_zero_l2249_224925


namespace NUMINAMATH_GPT_petya_time_spent_l2249_224953

theorem petya_time_spent :
  (1 / 3) + (1 / 5) + (1 / 6) + (1 / 70) + (1 / 3) > 1 :=
by
  sorry

end NUMINAMATH_GPT_petya_time_spent_l2249_224953


namespace NUMINAMATH_GPT_christina_age_fraction_l2249_224994

theorem christina_age_fraction {C : ℕ} (h1 : ∃ C : ℕ, (6 + 15) = (3/5 : ℚ) * C)
  (h2 : C + 5 = 40) : (C + 5) / 80 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_christina_age_fraction_l2249_224994


namespace NUMINAMATH_GPT_inradius_circumradius_l2249_224903

variables {T : Type} [MetricSpace T]

theorem inradius_circumradius (K k : ℝ) (d r rho : ℝ) (triangle : T)
  (h1 : (k / K) = (rho / r))
  (h2 : k ≤ K / 2)
  (h3 : 2 * r * rho = r^2 - d^2)
  (h4 : d ≥ 0) :
  r ≥ 2 * rho :=
sorry

end NUMINAMATH_GPT_inradius_circumradius_l2249_224903


namespace NUMINAMATH_GPT_problem_statement_l2249_224961

theorem problem_statement : 3.5 * 2.5 + 6.5 * 2.5 = 25 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2249_224961


namespace NUMINAMATH_GPT_sum_of_coordinates_A_l2249_224963

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_A_l2249_224963


namespace NUMINAMATH_GPT_boots_cost_5_more_than_shoes_l2249_224939

variable (S B : ℝ)

-- Conditions based on the problem statement
axiom h1 : 22 * S + 16 * B = 460
axiom h2 : 8 * S + 32 * B = 560

/-- Theorem to prove that the difference in cost between pairs of boots and pairs of shoes is $5 --/
theorem boots_cost_5_more_than_shoes : B - S = 5 :=
by
  sorry

end NUMINAMATH_GPT_boots_cost_5_more_than_shoes_l2249_224939


namespace NUMINAMATH_GPT_train_speed_in_kmph_l2249_224987

-- Definitions based on the conditions
def train_length : ℝ := 280 -- in meters
def time_to_pass_tree : ℝ := 28 -- in seconds

-- Conversion factor from meters/second to kilometers/hour
def mps_to_kmph : ℝ := 3.6

-- The speed of the train in kilometers per hour
theorem train_speed_in_kmph : (train_length / time_to_pass_tree) * mps_to_kmph = 36 := 
sorry

end NUMINAMATH_GPT_train_speed_in_kmph_l2249_224987


namespace NUMINAMATH_GPT_major_axis_length_proof_l2249_224927

-- Define the conditions
def radius : ℝ := 3
def minor_axis_length : ℝ := 2 * radius
def major_axis_length : ℝ := minor_axis_length + 0.75 * minor_axis_length

-- State the proof problem
theorem major_axis_length_proof : major_axis_length = 10.5 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_major_axis_length_proof_l2249_224927


namespace NUMINAMATH_GPT_annual_profit_growth_rate_l2249_224937

variable (a : ℝ)

theorem annual_profit_growth_rate (ha : a > -1) : 
  (1 + a) ^ 12 - 1 = (1 + a) ^ 12 - 1 := 
by 
  sorry

end NUMINAMATH_GPT_annual_profit_growth_rate_l2249_224937


namespace NUMINAMATH_GPT_tiffany_first_level_treasures_l2249_224936

-- Conditions
def treasure_points : ℕ := 6
def treasures_second_level : ℕ := 5
def total_points : ℕ := 48

-- Definition for the number of treasures on the first level
def points_from_second_level : ℕ := treasures_second_level * treasure_points
def points_from_first_level : ℕ := total_points - points_from_second_level
def treasures_first_level : ℕ := points_from_first_level / treasure_points

-- The theorem to prove
theorem tiffany_first_level_treasures : treasures_first_level = 3 :=
by
  sorry

end NUMINAMATH_GPT_tiffany_first_level_treasures_l2249_224936


namespace NUMINAMATH_GPT_tunnel_length_correct_l2249_224935

noncomputable def tunnel_length (truck_length : ℝ) (time_to_exit : ℝ) (speed_mph : ℝ) (mile_to_feet : ℝ) : ℝ :=
let speed_fps := (speed_mph * mile_to_feet) / 3600
let total_distance := speed_fps * time_to_exit
total_distance - truck_length

theorem tunnel_length_correct :
  tunnel_length 66 6 45 5280 = 330 :=
by
  sorry

end NUMINAMATH_GPT_tunnel_length_correct_l2249_224935


namespace NUMINAMATH_GPT_third_median_length_l2249_224926

variable (a b : ℝ) (A : ℝ)

def two_medians (m₁ m₂ : ℝ) : Prop :=
  m₁ = 4.5 ∧ m₂ = 7.5

def triangle_area (area : ℝ) : Prop :=
  area = 6 * Real.sqrt 20

theorem third_median_length (m₁ m₂ m₃ : ℝ) (area : ℝ) (h₁ : two_medians m₁ m₂)
  (h₂ : triangle_area area) : m₃ = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_third_median_length_l2249_224926


namespace NUMINAMATH_GPT_find_c_l2249_224943

open Real

theorem find_c (c : ℝ) (h : ∀ x, (x ∈ Set.Iio 2 ∨ x ∈ Set.Ioi 7) → -x^2 + c * x - 9 < -4) : 
  c = 9 :=
sorry

end NUMINAMATH_GPT_find_c_l2249_224943


namespace NUMINAMATH_GPT_ratio_of_original_to_reversed_l2249_224995

def original_number : ℕ := 21
def reversed_number : ℕ := 12

theorem ratio_of_original_to_reversed : 
  (original_number : ℚ) / (reversed_number : ℚ) = 7 / 4 := by
  sorry

end NUMINAMATH_GPT_ratio_of_original_to_reversed_l2249_224995


namespace NUMINAMATH_GPT_lori_marbles_l2249_224986

theorem lori_marbles (friends marbles_per_friend : ℕ) (h_friends : friends = 5) (h_marbles_per_friend : marbles_per_friend = 6) : friends * marbles_per_friend = 30 := sorry

end NUMINAMATH_GPT_lori_marbles_l2249_224986


namespace NUMINAMATH_GPT_royalties_amount_l2249_224905

/--
Given the following conditions:
1. No tax for royalties up to 800 yuan.
2. For royalties exceeding 800 yuan but not exceeding 4000 yuan, tax is levied at 14% on the amount exceeding 800 yuan.
3. For royalties exceeding 4000 yuan, tax is levied at 11% of the total royalties.

If someone has paid 420 yuan in taxes for publishing a book, prove that their royalties amount to 3800 yuan.
-/
theorem royalties_amount (r : ℝ) (h₁ : ∀ r, r ≤ 800 → 0 = r * 0 / 100)
  (h₂ : ∀ r, 800 < r ∧ r ≤ 4000 → 0.14 * (r - 800) = r * 0.14 / 100)
  (h₃ : ∀ r, r > 4000 → 0.11 * r = 420) : r = 3800 := sorry

end NUMINAMATH_GPT_royalties_amount_l2249_224905


namespace NUMINAMATH_GPT_parallel_lines_slope_eq_l2249_224982

theorem parallel_lines_slope_eq (k : ℝ) : 
  (∀ x : ℝ, k * x - 1 = 3 * x) → k = 3 :=
by sorry

end NUMINAMATH_GPT_parallel_lines_slope_eq_l2249_224982


namespace NUMINAMATH_GPT_difference_between_local_and_face_value_of_7_in_65793_l2249_224955

theorem difference_between_local_and_face_value_of_7_in_65793 :
  let numeral := 65793
  let digit := 7
  let place := 100
  let local_value := digit * place
  let face_value := digit
  local_value - face_value = 693 := 
by
  sorry

end NUMINAMATH_GPT_difference_between_local_and_face_value_of_7_in_65793_l2249_224955


namespace NUMINAMATH_GPT_calculate_roots_l2249_224962

noncomputable def cube_root (x : ℝ) := x^(1/3 : ℝ)
noncomputable def square_root (x : ℝ) := x^(1/2 : ℝ)

theorem calculate_roots : cube_root (-8) + square_root 9 = 1 :=
by
  sorry

end NUMINAMATH_GPT_calculate_roots_l2249_224962


namespace NUMINAMATH_GPT_peter_age_l2249_224928

variable (x y : ℕ)

theorem peter_age : 
  (x = (3 * y) / 2) ∧ ((4 * y - x) + 2 * y = 54) → x = 18 :=
by
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_peter_age_l2249_224928


namespace NUMINAMATH_GPT_find_number_l2249_224909

theorem find_number (n : ℕ) (h : n / 3 = 10) : n = 30 := by
  sorry

end NUMINAMATH_GPT_find_number_l2249_224909


namespace NUMINAMATH_GPT_total_money_shared_l2249_224920

def A_share (B : ℕ) : ℕ := B / 2
def B_share (C : ℕ) : ℕ := C / 2
def C_share : ℕ := 400

theorem total_money_shared (A B C : ℕ) (h1 : A = A_share B) (h2 : B = B_share C) (h3 : C = C_share) : A + B + C = 700 :=
by
  sorry

end NUMINAMATH_GPT_total_money_shared_l2249_224920


namespace NUMINAMATH_GPT_coach_class_seats_l2249_224910

variable (F C : ℕ)

-- Define the conditions
def totalSeats := F + C = 387
def coachSeats := C = 4 * F + 2

-- State the theorem
theorem coach_class_seats : totalSeats F C → coachSeats F C → C = 310 :=
by sorry

end NUMINAMATH_GPT_coach_class_seats_l2249_224910


namespace NUMINAMATH_GPT_store_paid_price_l2249_224988

-- Definition of the conditions
def selling_price : ℕ := 34
def difference_price : ℕ := 8

-- Statement that needs to be proven.
theorem store_paid_price : (selling_price - difference_price) = 26 :=
by
  sorry

end NUMINAMATH_GPT_store_paid_price_l2249_224988


namespace NUMINAMATH_GPT_car_speed_conversion_l2249_224997

theorem car_speed_conversion (V_kmph : ℕ) (h : V_kmph = 36) : (V_kmph * 1000 / 3600) = 10 := by
  sorry

end NUMINAMATH_GPT_car_speed_conversion_l2249_224997


namespace NUMINAMATH_GPT_focus_of_parabola_l2249_224934

-- Define the given parabola equation
def parabola_eq (x y : ℝ) : Prop := y = (1/4) * x^2

-- Define the conditions about the focus and the parabola direction
def focus_on_y_axis : Prop := True -- Given condition
def opens_upwards : Prop := True -- Given condition

theorem focus_of_parabola (x y : ℝ) 
  (h1 : parabola_eq x y) 
  (h2 : focus_on_y_axis) 
  (h3 : opens_upwards) : 
  (x = 0 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l2249_224934


namespace NUMINAMATH_GPT_probability_of_drawing_orange_marble_second_l2249_224991

noncomputable def probability_second_marble_is_orange (total_A white_A black_A : ℕ) (total_B orange_B green_B blue_B : ℕ) (total_C orange_C green_C blue_C : ℕ) : ℚ := 
  let p_white := (white_A : ℚ) / total_A
  let p_black := (black_A : ℚ) / total_A
  let p_orange_B := (orange_B : ℚ) / total_B
  let p_orange_C := (orange_C : ℚ) / total_C
  (p_white * p_orange_B) + (p_black * p_orange_C)

theorem probability_of_drawing_orange_marble_second :
  probability_second_marble_is_orange 9 4 5 15 7 5 3 10 4 4 2 = 58 / 135 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_orange_marble_second_l2249_224991


namespace NUMINAMATH_GPT_sandy_age_l2249_224901

variable (S M : ℕ)

-- Conditions
def condition1 := M = S + 12
def condition2 := S * 9 = M * 7

theorem sandy_age : condition1 S M → condition2 S M → S = 42 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sandy_age_l2249_224901


namespace NUMINAMATH_GPT_factorization_sum_l2249_224933

variable {a b c : ℤ}

theorem factorization_sum 
  (h1 : ∀ x : ℤ, x^2 + 17 * x + 52 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) : 
  a + b + c = 27 :=
sorry

end NUMINAMATH_GPT_factorization_sum_l2249_224933


namespace NUMINAMATH_GPT_route_comparison_l2249_224915

-- Definitions
def distance (P Z C : Type) : Type := ℝ

variables {P Z C : Type} -- P: Park, Z: Zoo, C: Circus
variables (x y C : ℝ)     -- x: direct distance from Park to Zoo, y: direct distance from Circus to Zoo, C: total circumference

-- Conditions
axiom h1 : x + 3 * x = C -- distance from Park to Zoo via Circus is three times longer than not via Circus
axiom h2 : y = (C - x) / 2 -- distance from Circus to Zoo directly is y
axiom h3 : 2 * y = C - x -- distance from Circus to Zoo via Park is twice as short as not via Park

-- Proof statement
theorem route_comparison (P Z C : Type) (x y C : ℝ) (h1 : x + 3 * x = C) (h2 : y = (C - x) / 2) (h3 : 2 * y = C - x) :
  let direct_route := x
  let via_zoo_route := 3 * x - x
  via_zoo_route = 11 * direct_route := 
sorry

end NUMINAMATH_GPT_route_comparison_l2249_224915


namespace NUMINAMATH_GPT_prime_divisor_of_ones_l2249_224902

theorem prime_divisor_of_ones (p : ℕ) (hp : Nat.Prime p ∧ p ≠ 2 ∧ p ≠ 5) :
  ∃ k : ℕ, p ∣ (10^k - 1) / 9 :=
by
  sorry

end NUMINAMATH_GPT_prime_divisor_of_ones_l2249_224902


namespace NUMINAMATH_GPT_max_cars_quotient_div_10_l2249_224952

theorem max_cars_quotient_div_10 (n : ℕ) (h1 : ∀ v : ℕ, v ≥ 20 * n) (h2 : ∀ d : ℕ, d = 5* (n + 1)) :
  (4000 / 10 = 400) := by
  sorry

end NUMINAMATH_GPT_max_cars_quotient_div_10_l2249_224952


namespace NUMINAMATH_GPT_profit_share_of_B_l2249_224924

-- Defining the initial investments
def a : ℕ := 8000
def b : ℕ := 10000
def c : ℕ := 12000

-- Given difference between profit shares of A and C
def diff_AC : ℕ := 680

-- Define total profit P
noncomputable def P : ℕ := (diff_AC * 15) / 2

-- Calculate B's profit share
noncomputable def B_share : ℕ := (5 * P) / 15

-- The theorem stating B's profit share
theorem profit_share_of_B : B_share = 1700 :=
by sorry

end NUMINAMATH_GPT_profit_share_of_B_l2249_224924


namespace NUMINAMATH_GPT_converse_proposition_true_l2249_224983

theorem converse_proposition_true (x y : ℝ) (h : x > abs y) : x > y := 
by
sorry

end NUMINAMATH_GPT_converse_proposition_true_l2249_224983


namespace NUMINAMATH_GPT_remainder_of_x_pow_150_div_by_x_minus_1_cubed_l2249_224941

theorem remainder_of_x_pow_150_div_by_x_minus_1_cubed :
  (x : ℤ) → (x^150 % (x - 1)^3) = (11175 * x^2 - 22200 * x + 11026) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_remainder_of_x_pow_150_div_by_x_minus_1_cubed_l2249_224941


namespace NUMINAMATH_GPT_find_pairs_l2249_224996

theorem find_pairs (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 / b + b^2 / a = (a + b)^2 / (a + b)) ↔ (a = b) := by
  sorry

end NUMINAMATH_GPT_find_pairs_l2249_224996


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l2249_224989

variable (m n : ℕ)
def positive_integers (m n : ℕ) := m > 0 ∧ n > 0
def at_least_one_is_1 (m n : ℕ) : Prop := m = 1 ∨ n = 1
def sum_gt_product (m n : ℕ) : Prop := m + n > m * n

theorem necessary_and_sufficient_condition (h : positive_integers m n) : 
  sum_gt_product m n ↔ at_least_one_is_1 m n :=
by sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l2249_224989


namespace NUMINAMATH_GPT_ashu_complete_job_in_20_hours_l2249_224966

/--
  Suresh can complete a job in 15 hours.
  Ashutosh alone can complete the same job in some hours.
  Suresh works for 9 hours and then the remaining job is completed by Ashutosh in 8 hours.
  We need to prove that the number of hours it takes for Ashutosh to complete the job alone is 20.
-/
theorem ashu_complete_job_in_20_hours :
  let A : ℝ := 20
  let suresh_work_rate : ℝ := 1 / 15
  let suresh_completed_work_in_9_hours : ℝ := (9 * suresh_work_rate)
  let remaining_work : ℝ := 1 - suresh_completed_work_in_9_hours
  (8 * (1 / A)) = remaining_work → A = 20 :=
by
  sorry

end NUMINAMATH_GPT_ashu_complete_job_in_20_hours_l2249_224966


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l2249_224951

noncomputable def func (x : ℝ) : ℝ :=
  x * Real.log x

noncomputable def derivative (x : ℝ) : ℝ :=
  Real.log x + 1

theorem monotonic_decreasing_interval :
  { x : ℝ | 0 < x ∧ x < Real.exp (-1) } ⊆ { x : ℝ | derivative x < 0 } :=
by
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l2249_224951


namespace NUMINAMATH_GPT_tangent_line_at_origin_tangent_line_passing_through_neg1_neg3_l2249_224984

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x

noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 2

theorem tangent_line_at_origin :
  (∀ x y : ℝ, y = f x → x = 0 → y = 2 * x) := by
  sorry

theorem tangent_line_passing_through_neg1_neg3 :
  (∀ x y : ℝ, y = f x → (x, y) ≠ (-1, -3) → y = 5 * x + 2) := by
  sorry

end NUMINAMATH_GPT_tangent_line_at_origin_tangent_line_passing_through_neg1_neg3_l2249_224984


namespace NUMINAMATH_GPT_quadratic_real_roots_range_l2249_224965

theorem quadratic_real_roots_range (k : ℝ) (h : ∀ x : ℝ, (k - 1) * x^2 - 2 * x + 1 = 0) : k ≤ 2 ∧ k ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_l2249_224965


namespace NUMINAMATH_GPT_average_price_per_pen_l2249_224913

def total_cost_pens_pencils : ℤ := 690
def number_of_pencils : ℕ := 75
def price_per_pencil : ℤ := 2
def number_of_pens : ℕ := 30

theorem average_price_per_pen :
  (total_cost_pens_pencils - number_of_pencils * price_per_pencil) / number_of_pens = 18 :=
by
  sorry

end NUMINAMATH_GPT_average_price_per_pen_l2249_224913


namespace NUMINAMATH_GPT_planted_fraction_correct_l2249_224911

-- Define the vertices of the triangle
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (5, 0)
def C : (ℝ × ℝ) := (0, 12)

-- Define the length of the legs
def leg1 := 5
def leg2 := 12

-- Define the shortest distance from the square to the hypotenuse
def distance_to_hypotenuse := 3

-- Define the area of the triangle
def triangle_area := (1 / 2) * (leg1 * leg2)

-- Assume the side length of the square
def s := 6 / 13

-- Define the area of the square
def square_area := s^2

-- Define the fraction of the field that is unplanted
def unplanted_fraction := square_area / triangle_area

-- Define the fraction of the field that is planted
def planted_fraction := 1 - unplanted_fraction

theorem planted_fraction_correct :
  planted_fraction = 5034 / 5070 :=
sorry

end NUMINAMATH_GPT_planted_fraction_correct_l2249_224911


namespace NUMINAMATH_GPT_minimum_value_of_sum_l2249_224950

open Real

theorem minimum_value_of_sum {a b : ℝ} (h₁ : 0 < a) (h₂ : 0 < b) (h : log a / log 2 + log b / log 2 ≥ 6) :
  a + b ≥ 16 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_sum_l2249_224950


namespace NUMINAMATH_GPT_sum_of_two_even_numbers_is_even_l2249_224904

  theorem sum_of_two_even_numbers_is_even (a b : ℤ) (ha : ∃ k : ℤ, a = 2 * k) (hb : ∃ m : ℤ, b = 2 * m) : ∃ n : ℤ, a + b = 2 * n := by
    sorry
  
end NUMINAMATH_GPT_sum_of_two_even_numbers_is_even_l2249_224904


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l2249_224914

theorem hyperbola_asymptotes (x y : ℝ) (E : x^2 / 4 - y^2 = 1) :
  y = (1 / 2) * x ∨ y = -(1 / 2) * x :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l2249_224914


namespace NUMINAMATH_GPT_area_relationship_l2249_224964

theorem area_relationship (x β : ℝ) (hβ : 0.60 * x^2 = β) : α = (4 / 3) * β :=
by
  -- conditions and goal are stated
  let α := 0.80 * x^2
  sorry

end NUMINAMATH_GPT_area_relationship_l2249_224964


namespace NUMINAMATH_GPT_rainfall_second_week_l2249_224954

theorem rainfall_second_week (r1 r2 : ℝ) (h1 : r1 + r2 = 40) (h2 : r2 = 1.5 * r1) : r2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_rainfall_second_week_l2249_224954


namespace NUMINAMATH_GPT_solve_a_plus_b_l2249_224969

theorem solve_a_plus_b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 7 * a + 2 * b = 54) : a + b = -103 / 31 :=
by
  sorry

end NUMINAMATH_GPT_solve_a_plus_b_l2249_224969


namespace NUMINAMATH_GPT_final_price_correct_l2249_224940

def original_price : Float := 100
def store_discount_rate : Float := 0.20
def promo_discount_rate : Float := 0.10
def sales_tax_rate : Float := 0.05
def handling_fee : Float := 5

def final_price (original_price : Float) 
                (store_discount_rate : Float) 
                (promo_discount_rate : Float) 
                (sales_tax_rate : Float) 
                (handling_fee : Float) 
                : Float :=
  let price_after_store_discount := original_price * (1 - store_discount_rate)
  let price_after_promo := price_after_store_discount * (1 - promo_discount_rate)
  let price_after_tax := price_after_promo * (1 + sales_tax_rate)
  let total_price := price_after_tax + handling_fee
  total_price

theorem final_price_correct : final_price original_price store_discount_rate promo_discount_rate sales_tax_rate handling_fee = 80.60 :=
by
  simp only [
    original_price,
    store_discount_rate,
    promo_discount_rate,
    sales_tax_rate,
    handling_fee
  ]
  norm_num
  sorry

end NUMINAMATH_GPT_final_price_correct_l2249_224940


namespace NUMINAMATH_GPT_simplify_expression_correct_l2249_224967

noncomputable def simplify_expression (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : ℝ :=
  let expr1 := (a^2 - b^2) / (a^2 + 2 * a * b + b^2)
  let expr2 := (2 : ℝ) / (a * b)
  let expr3 := ((1 : ℝ) / a + (1 : ℝ) / b)^2
  let expr4 := (2 : ℝ) / (a^2 - b^2 + 2 * a * b)
  expr1 + expr2 / expr3 * expr4

theorem simplify_expression_correct (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  simplify_expression a b h = 2 / (a + b)^2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_correct_l2249_224967


namespace NUMINAMATH_GPT_number_of_arrangements_l2249_224923

theorem number_of_arrangements (n : ℕ) (h_n : n = 6) : 
  ∃ total : ℕ, total = 90 := 
sorry

end NUMINAMATH_GPT_number_of_arrangements_l2249_224923


namespace NUMINAMATH_GPT_find_weight_of_second_square_l2249_224946

-- Define the initial conditions
def uniform_density_thickness (density : ℝ) (thickness : ℝ) : Prop :=
  ∀ (l₁ l₂ : ℝ), l₁ = l₂ → density = thickness

-- Define the first square properties
def first_square (side_length₁ weight₁ : ℝ) : Prop :=
  side_length₁ = 4 ∧ weight₁ = 16

-- Define the second square properties
def second_square (side_length₂ : ℝ) : Prop :=
  side_length₂ = 6

-- Define the proportional relationship between the area and weight
def proportional_weight (side_length₁ weight₁ side_length₂ weight₂ : ℝ) : Prop :=
  (side_length₁^2 / weight₁) = (side_length₂^2 / weight₂)

-- Lean statement to prove the weight of the second square
theorem find_weight_of_second_square (density thickness side_length₁ weight₁ side_length₂ weight₂ : ℝ)
  (h_density_thickness : uniform_density_thickness density thickness)
  (h_first_square : first_square side_length₁ weight₁)
  (h_second_square : second_square side_length₂)
  (h_proportional_weight : proportional_weight side_length₁ weight₁ side_length₂ weight₂) : 
  weight₂ = 36 :=
by 
  sorry

end NUMINAMATH_GPT_find_weight_of_second_square_l2249_224946


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2249_224959

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h2 : S 5 = 35)
  (h3 : ∀ n, S n = n * a 1 + n * (n - 1) / 2 * d) :
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2249_224959


namespace NUMINAMATH_GPT_fraction_not_on_time_l2249_224998

theorem fraction_not_on_time (n : ℕ) (h1 : ∃ (k : ℕ), 3 * k = 5 * n) 
(h2 : ∃ (k : ℕ), 4 * k = 5 * m) 
(h3 : ∃ (k : ℕ), 5 * k = 6 * f) 
(h4 : m + f = n) 
(h5 : r = rm + rf) 
(h6 : rm = 4/5 * m) 
(h7 : rf = 5/6 * f) :
  (not_on_time : ℚ) = 1/5 := 
by
  sorry

end NUMINAMATH_GPT_fraction_not_on_time_l2249_224998


namespace NUMINAMATH_GPT_parabola_equation_trajectory_midpoint_l2249_224970

-- Given data and conditions
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def point_on_parabola_x3 (p : ℝ) : Prop := ∃ y, parabola p 3 y
def distance_point_to_line (x d : ℝ) : Prop := x + d = 5

-- Prove that given these conditions, the parabola equation is y^2 = 8x
theorem parabola_equation (p : ℝ) (h1 : point_on_parabola_x3 p) (h2 : distance_point_to_line (3 + p / 2) 2) : p = 4 :=
sorry

-- Prove the equation of the trajectory for the midpoint of the line segment FP
def point_on_parabola (p x y : ℝ) : Prop := y^2 = 8 * x
theorem trajectory_midpoint (p x y : ℝ) (h1 : parabola 4 x y) : y^2 = 4 * (x - 1) :=
sorry

end NUMINAMATH_GPT_parabola_equation_trajectory_midpoint_l2249_224970


namespace NUMINAMATH_GPT_number_of_solutions_5x_plus_10y_eq_50_l2249_224945

theorem number_of_solutions_5x_plus_10y_eq_50 : 
  (∃! (n : ℕ), ∃ (xy : ℕ × ℕ), xy.1 + 2 * xy.2 = 10 ∧ n = 6) :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_5x_plus_10y_eq_50_l2249_224945


namespace NUMINAMATH_GPT_percentage_of_discount_l2249_224932

variable (C : ℝ) -- Cost Price of the Book

-- Conditions
axiom profit_with_discount (C : ℝ) : ∃ S_d : ℝ, S_d = C * 1.235
axiom profit_without_discount (C : ℝ) : ∃ S_nd : ℝ, S_nd = C * 2.30

-- Theorem to prove
theorem percentage_of_discount (C : ℝ) : 
  ∃ discount_percentage : ℝ, discount_percentage = 46.304 := by
  sorry

end NUMINAMATH_GPT_percentage_of_discount_l2249_224932


namespace NUMINAMATH_GPT_balance_pots_l2249_224977

theorem balance_pots 
  (w1 : ℕ) (w2 : ℕ) (m : ℕ)
  (h_w1 : w1 = 645)
  (h_w2 : w2 = 237)
  (h_m : m = 1000) :
  ∃ (m1 m2 : ℕ), 
  (w1 + m1 = w2 + m2) ∧ 
  (m1 + m2 = m) ∧ 
  (m1 = 296) ∧ 
  (m2 = 704) := by
  sorry

end NUMINAMATH_GPT_balance_pots_l2249_224977


namespace NUMINAMATH_GPT_product_is_eight_l2249_224947

noncomputable def compute_product (r : ℂ) (hr : r ≠ 1) (hr7 : r^7 = 1) : ℂ :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1)

theorem product_is_eight (r : ℂ) (hr : r ≠ 1) (hr7 : r^7 = 1) : compute_product r hr hr7 = 8 :=
by
  sorry

end NUMINAMATH_GPT_product_is_eight_l2249_224947


namespace NUMINAMATH_GPT_cost_of_individual_roll_is_correct_l2249_224958

-- Definitions given in the problem's conditions
def cost_per_case : ℝ := 9
def number_of_rolls : ℝ := 12
def percent_savings : ℝ := 0.25

-- The cost of one roll sold individually
noncomputable def individual_roll_cost : ℝ := 0.9375

-- The theorem to prove
theorem cost_of_individual_roll_is_correct :
  individual_roll_cost = (cost_per_case * (1 + percent_savings)) / number_of_rolls :=
by
  sorry

end NUMINAMATH_GPT_cost_of_individual_roll_is_correct_l2249_224958


namespace NUMINAMATH_GPT_max_min_f_l2249_224960

noncomputable def f (x : ℝ) : ℝ :=
  if 6 ≤ x ∧ x ≤ 8 then
    (Real.sqrt (8 * x - x^2) - Real.sqrt (114 * x - x^2 - 48))
  else
    0

theorem max_min_f :
  ∀ x, 6 ≤ x ∧ x ≤ 8 → f x ≤ 2 * Real.sqrt 3 ∧ 0 ≤ f x :=
by
  intros
  sorry

end NUMINAMATH_GPT_max_min_f_l2249_224960


namespace NUMINAMATH_GPT_floor_of_neg_five_thirds_l2249_224973

theorem floor_of_neg_five_thirds : Int.floor (-5/3 : ℝ) = -2 := 
by 
  sorry

end NUMINAMATH_GPT_floor_of_neg_five_thirds_l2249_224973


namespace NUMINAMATH_GPT_arithmetic_sequence_150th_term_l2249_224942

theorem arithmetic_sequence_150th_term :
  let a₁ := 3
  let d := 5
  let n := 150
  (a₁ + (n - 1) * d) = 748 :=
by
  let a₁ := 3
  let d := 5
  let n := 150
  show a₁ + (n - 1) * d = 748
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_150th_term_l2249_224942


namespace NUMINAMATH_GPT_scientific_notation_correct_l2249_224938

theorem scientific_notation_correct :
  0.00000164 = 1.64 * 10^(-6) :=
sorry

end NUMINAMATH_GPT_scientific_notation_correct_l2249_224938


namespace NUMINAMATH_GPT_sum_of_longest_altitudes_l2249_224908

-- Define the sides of the triangle
def a : ℕ := 6
def b : ℕ := 8
def c : ℕ := 10

-- Define the sides are the longest altitudes in the right triangle
def longest_altitude1 : ℕ := a
def longest_altitude2 : ℕ := b

-- Define the main theorem to prove
theorem sum_of_longest_altitudes : longest_altitude1 + longest_altitude2 = 14 := 
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_longest_altitudes_l2249_224908


namespace NUMINAMATH_GPT_four_digit_numbers_thousands_digit_5_div_by_5_l2249_224929

theorem four_digit_numbers_thousands_digit_5_div_by_5 :
  ∃ (s : Finset ℕ), (∀ x ∈ s, 5000 ≤ x ∧ x ≤ 5999 ∧ x % 5 = 0) ∧ s.card = 200 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_thousands_digit_5_div_by_5_l2249_224929


namespace NUMINAMATH_GPT_largest_divisor_of_consecutive_odd_product_l2249_224949

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h_even : n % 2 = 0) (h_pos : n > 0) :
  315 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) := 
sorry

end NUMINAMATH_GPT_largest_divisor_of_consecutive_odd_product_l2249_224949


namespace NUMINAMATH_GPT_locus_of_intersection_l2249_224976

theorem locus_of_intersection
  (a b : ℝ) (h_a_nonzero : a ≠ 0) (h_b_nonzero : b ≠ 0) (h_neq : a ≠ b) :
  ∃ (x y : ℝ), 
    (∃ c : ℝ, y = (a/c)*x ∧ (x/b + y/c = 1)) 
    ∧ 
    ( (x - b/2)^2 / (b^2/4) + y^2 / (ab/4) = 1 ) :=
sorry

end NUMINAMATH_GPT_locus_of_intersection_l2249_224976


namespace NUMINAMATH_GPT_remainder_2048_mod_13_l2249_224918

theorem remainder_2048_mod_13 : 2048 % 13 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_2048_mod_13_l2249_224918


namespace NUMINAMATH_GPT_value_of_expression_l2249_224971

theorem value_of_expression : 
  ∀ (x y : ℤ), x = -5 → y = -10 → (y - x) * (y + x) = 75 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_value_of_expression_l2249_224971


namespace NUMINAMATH_GPT_circle_C_equation_l2249_224975

/-- Definitions of circles C1 and C2 -/
def circle_C1 := ∀ (x y : ℝ), (x - 4) ^ 2 + (y - 8) ^ 2 = 1
def circle_C2 := ∀ (x y : ℝ), (x - 6) ^ 2 + (y + 6) ^ 2 = 9

/-- Condition that the center of circle C is on the x-axis -/
def center_on_x_axis (x : ℝ) : Prop := ∃ y : ℝ, y = 0

/-- Bisection condition circle C bisects circumferences of circles C1 and C2 -/
def bisects_circumferences (x : ℝ) : Prop := 
  (∀ (y1 y2 : ℝ), ((x - 4) ^ 2 + (y1 - 8) ^ 2 + 1 = (x - 6) ^ 2 + (y2 + 6) ^ 2 + 9)) ∧ 
  center_on_x_axis x

/-- Statement to prove -/
theorem circle_C_equation : ∃ x y : ℝ, bisects_circumferences x ∧ (x^2 + y^2 = 81) := 
sorry

end NUMINAMATH_GPT_circle_C_equation_l2249_224975


namespace NUMINAMATH_GPT_smallest_positive_perfect_square_divisible_by_2_3_5_l2249_224999

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_positive_perfect_square_divisible_by_2_3_5_l2249_224999


namespace NUMINAMATH_GPT_books_more_than_movies_l2249_224930

theorem books_more_than_movies (books_count movies_count read_books watched_movies : ℕ) 
  (h_books : books_count = 10)
  (h_movies : movies_count = 6)
  (h_read_books : read_books = 10) 
  (h_watched_movies : watched_movies = 6) : 
  read_books - watched_movies = 4 := by
  sorry

end NUMINAMATH_GPT_books_more_than_movies_l2249_224930


namespace NUMINAMATH_GPT_max_profit_is_4sqrt6_add_21_l2249_224917

noncomputable def profit (x : ℝ) : ℝ :=
  let y1 : ℝ := -2 * (3 - x)^2 + 14 * (3 - x)
  let y2 : ℝ := - (1 / 3) * x^3 + 2 * x^2 + 5 * x
  let F : ℝ := y1 + y2 - 3
  F

theorem max_profit_is_4sqrt6_add_21 : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ profit x = 4 * Real.sqrt 6 + 21 :=
sorry

end NUMINAMATH_GPT_max_profit_is_4sqrt6_add_21_l2249_224917
