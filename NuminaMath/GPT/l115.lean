import Mathlib

namespace fraction_ordering_l115_115200

theorem fraction_ordering :
  (12/35 < 10/29) ∧ (10/29 < 6/17) :=
by
  -- Let's provide the comparisons we discovered
  have h1 : 12/35 < 10/29 := sorry,
  have h2 : 10/29 < 6/17 := sorry,
  exact ⟨h1, h2⟩

end fraction_ordering_l115_115200


namespace correct_conclusion_count_l115_115121

open Set

variable {f : ℝ → ℝ}

theorem correct_conclusion_count :
  (∀ x ∈ Ioo 0 1, f x > 0) →
  (∀ x1 x2 ∈ Ioo 0 1, (f x1 / f x2) + (f (1 - x1) / f (1 - x2)) ≤ 2) →
  (∀ x ∈ Ioo 0 1, f x ≠ f (1 - x))
  ∧ (¬ ∀ x ∈ Ioo 0 1, f x > f (1 - x))
  ∧ (∃ k > 0, ∀ x ∈ Ioo 0 1, y = (f x / x) + x ∧ ¬ Monotone (λ x, y))
  ∧ (∀ x ∈ Ioo 0 1, f x = f (1 - x)) :=
begin
  sorry
end

end correct_conclusion_count_l115_115121


namespace importance_of_all_in_locus_definition_l115_115080

def point (α : Type*) [ordered_field α] := (α × α)

def segment (α : Type*) [ordered_field α] := point α × point α

def fixed_angle (α : Type*) [ordered_field α] (θ : α) (A B : segment α) :=
∀ P : point α, (P ≠ A.1 ∧ P ≠ A.2) →
(∀ Q : point α, ∠(A.1, P, A.2) = θ → Q = P)

theorem importance_of_all_in_locus_definition {α : Type*} [ordered_field α] (θ : α) (A B : segment α) :
  (∀ P : point α, fixed_angle α θ (A,B) P ↔ (∃ O₁ O₂ : point α, (circle_eqn O₁ R₁ P ∧ circle_eqn O₂ R₂ P))) :=
sorry

end importance_of_all_in_locus_definition_l115_115080


namespace solve_for_x_l115_115780

theorem solve_for_x (x : ℝ) (h : 3 * x - 8 = 4 * x + 5) : x = -13 :=
by 
  sorry

end solve_for_x_l115_115780


namespace distance_from_start_need_refuel_total_revenue_l115_115487

def distances : List Int := [-3, -15, +19, -1, +5, -12, -6, +12]

def passengers : List Bool := [False, True, True, False, True, True, True, True]

def fuel_consumption_per_km : Float := 0.06

def initial_fuel : Float := 7.0

def refuel_threshold : Float := 2.0

def base_fare : Float := 10.0

def additional_fare_per_km : Float := 1.6

def distance_threshold : Int := 2

-- Part 1: Prove direction and distance from Point A
theorem distance_from_start : 
  ∑ x in distances, x = -1 := sorry

-- Part 2: Prove whether refueling is needed
theorem need_refuel : 
  initial_fuel - (fuel_consumption_per_km * (distances.map (fun x => |x|)).sum) > refuel_threshold := sorry

-- Part 3: Prove total revenue
theorem total_revenue :
  (passengers.zip distances).foldr
    (fun (pd : Bool × Int) acc =>
      acc + if pd.1 = True 
            then base_fare + Float.ofInt (pd.2 - distance_threshold) * additional_fare_per_km 
            else 0.0)
    0 = 151.2 := sorry

end distance_from_start_need_refuel_total_revenue_l115_115487


namespace find_b_area_of_incircle_l115_115087

-- Define triangle and conditions
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (cosA : ℝ) (sinB : ℝ)

-- Given conditions
def triangle_ABC := 
  a = 4 ∧
  cosA = 3 / 4 ∧
  sinB = 5 * sqrt 7 / 16 ∧
  c > 4

-- Prove that b = 5 given the conditions
theorem find_b (h : triangle_ABC A B C a b c cosA sinB) : b = 5 := 
sorry

-- Prove the area of the incircle is 7π/4
theorem area_of_incircle (h : triangle_ABC A B C a b c cosA sinB)
(b_eq : b = 5)
(c_eq : c = 6) : 
  let S := (1 / 2) * a * c * sinB in
  let l := a + b + c in
  let r := 2 * S / l in
  let A := π * r ^ 2 in
  A = 7 * π / 4 :=
sorry

end find_b_area_of_incircle_l115_115087


namespace system_of_equations_solution_l115_115480

theorem system_of_equations_solution :
  ∃ (x1 x2 x3 x4 x5: ℝ), 
    2 * x1 = x5 ^ 2 - 23 ∧ 
    4 * x2 = x1 ^ 2 + 7 ∧ 
    6 * x3 = x2 ^ 2 + 14 ∧ 
    8 * x4 = x3 ^ 2 + 23 ∧ 
    10 * x5 = x4 ^ 2 + 34 ∧ 
    x1 = 1 ∧ 
    x2 = 2 ∧ 
    x3 = 3 ∧ 
    x4 = 4 ∧ 
    x5 = 5 :=
by {
  use 1, 2, 3, 4, 5,
  sorry
}

end system_of_equations_solution_l115_115480


namespace simplify_expression_l115_115476

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) :
  ( (3 / (4 * m)) ^ (-3) ) * (2 * m) ^ 4 = 1024 * m ^ 7 / 27 :=
by
  sorry

end simplify_expression_l115_115476


namespace train_speed_l115_115952

theorem train_speed (train_length : ℕ) (cross_time : ℕ) (speed : ℕ) 
  (h_train_length : train_length = 300)
  (h_cross_time : cross_time = 10)
  (h_speed_eq : speed = train_length / cross_time) : 
  speed = 30 :=
by
  sorry

end train_speed_l115_115952


namespace blocks_for_sphere_l115_115245

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_block (a b c : ℝ) : ℝ :=
  a * b * c

noncomputable def blocks_needed (sphere_volume block_volume : ℝ) : ℕ :=
  ⌈sphere_volume / block_volume⌉.to_nat

theorem blocks_for_sphere : blocks_needed (volume_of_sphere 5) (volume_of_block 4 3 0.5) = 83 :=
  by
  sorry

end blocks_for_sphere_l115_115245


namespace rectangle_perimeter_l115_115634

theorem rectangle_perimeter
  (square_perimeter : ℕ)
  (rectangle_width : ℕ)
  (square_area_eq : ℕ)
  (square_perimeter = 24)
  (rectangle_width = 4)
  (square_area_eq = (square_perimeter / 4) ^ 2) :
  2 * (square_area_eq / rectangle_width + rectangle_width) = 26 :=
by
  sorry

end rectangle_perimeter_l115_115634


namespace sum_of_logarithms_l115_115777

theorem sum_of_logarithms (a b : ℝ) (h1 : 10^a = 5) (h2 : 10^b = 2) : a + b = 1 := 
  sorry

end sum_of_logarithms_l115_115777


namespace find_FC_l115_115414

structure Point where
  x : ℝ
  y : ℝ

structure Rectangle where
  A B C D E F : Point
  m1 : E = midpoint C D
  m2 : right_angle (line A E) (line E F)
  m3 : distance A F = 9
  m4 : distance B F = 6

def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

def right_angle (l1 l2 : Line) : Prop := -- definition omitted for brevity
sorry -- The exact definition would depend on how Line is defined.

def distance (p1 p2 : Point) : ℝ :=
  sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem find_FC (r : Rectangle) : distance r.F r.C = 1.5 :=
sorry

end find_FC_l115_115414


namespace productivity_increase_l115_115959

theorem productivity_increase :
  (∃ d : ℝ, 
   (∀ n : ℕ, 0 < n → n ≤ 30 → 
      (5 + (n - 1) * d ≥ 0) ∧ 
      (30 * 5 + (30 * 29 / 2) * d = 390) ∧ 
      1 / 100 < d ∧ d < 1) ∧
      d = 0.52) :=
sorry

end productivity_increase_l115_115959


namespace sum_of_sequence_l115_115533

noncomputable def sequence_term (n : ℕ) : ℕ :=
(2^n) - 1

noncomputable def sum_sequence (n : ℕ) : ℕ :=
(nat.range n).sum (λ k, sequence_term (k + 1))

theorem sum_of_sequence (n : ℕ) : sum_sequence n = (2^(n+1)) - n - 2 :=
by sorry

end sum_of_sequence_l115_115533


namespace saree_total_cost_l115_115609

/-- Total cost calculation of three sarees after applying discounts and conversion to INR --/
def total_cost_inr (price_usd : ℕ) (price_gbp : ℕ) (price_eur : ℕ)
  (discounts_usd : list ℕ) (discounts_gbp : list ℕ) (discount_eur : ℕ)
  (sales_tax : ℕ)
  (conv_usd_to_inr : ℕ) (conv_gbp_to_inr : ℕ) (conv_eur_to_inr : ℕ)
  : ℕ :=
  let apply_discounts (price : ℕ) (discounts : list ℕ) : ℕ :=
    discounts.foldl (λ p d, p - p * d / 100) price in
  let saree1_inr := (apply_discounts price_usd discounts_usd) * conv_usd_to_inr in
  let saree2_inr := (apply_discounts price_gbp discounts_gbp) * conv_gbp_to_inr in
  let saree3_inr := (apply_discounts price_eur [discount_eur]) * conv_eur_to_inr in
  let add_sales_tax (price_inr : ℕ) := price_inr + price_inr * sales_tax / 100 in
  add_sales_tax saree1_inr + add_sales_tax saree2_inr + add_sales_tax saree3_inr

theorem saree_total_cost :
  total_cost_inr 200 150 180 [20, 15, 5] [10, 7] 12 8 75 100 90 = 39421.08 :=
by sorry

end saree_total_cost_l115_115609


namespace tan_alpha_complex_expr_l115_115748

theorem tan_alpha (m : ℝ) (α : ℝ) (h₁ : m = 1) 
(h : m^2 + (sqrt 2)^2 = 3) : 
tan α = sqrt 2 := by 
  sorry

theorem complex_expr (m : ℝ) (α : ℝ) (h₁ : m = 1) 
(h : m^2 + (sqrt 2)^2 = 3) : 
(2 * (cos(α / 2))^2 - sin α - 1) / (sqrt 2 * sin (π / 4 + α)) = 2 * sqrt 2 - 3 := by 
  sorry

end tan_alpha_complex_expr_l115_115748


namespace average_stamps_collected_per_day_l115_115853

open Nat

-- Define an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + d * (n - 1)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
def a := 10
def d := 10
def n := 7

-- Prove that the average number of stamps collected over 7 days is 40
theorem average_stamps_collected_per_day : 
  sum_arithmetic_sequence a d n / n = 40 := 
by
  sorry

end average_stamps_collected_per_day_l115_115853


namespace completion_time_l115_115233

variables {P E : ℝ}
theorem completion_time (h1 : (20 : ℝ) * P * E / 2 = D * (2.5 * P * E)) : D = 4 :=
by
  -- Given h1 as the condition
  sorry

end completion_time_l115_115233


namespace increasing_intervals_range_l115_115024

theorem increasing_intervals_range {a : ℝ} (h1 : 0 < a ∧ a ≤ π) :
  (∀ x, 0 ≤ x ∧ x ≤ a / 3 → deriv (λ x, sqrt 3 * sin (2 * x) - cos (2 * x)) x > 0) ∧
  (∀ x, 2 * a ≤ x ∧ x ≤ 4 * π / 3 → deriv (λ x, sqrt 3 * sin (2 * x) - cos (2 * x)) x > 0) ↔ 
  (5 * π / 12 ≤ a ∧ a ≤ π) :=
by
  sorry

end increasing_intervals_range_l115_115024


namespace alternating_number_not_prime_l115_115239

-- Define the number n such that it has alternating zeros and ones with 2016 zeros.
def alternating_number_with_zeros (k : Nat) : Nat :=
  (List.range (2 * k + 1)).foldr (λ i acc, acc + if i % 2 = 0 then 10^i else 0) 0

theorem alternating_number_not_prime : 
  let n := alternating_number_with_zeros 2016 in
  ¬ Prime n :=
by
  sorry

end alternating_number_not_prime_l115_115239


namespace closest_fraction_to_team_aus_medals_l115_115265

theorem closest_fraction_to_team_aus_medals 
  (won_medals : ℕ) (total_medals : ℕ) 
  (choices : List ℚ)
  (fraction_won : ℚ)
  (c1 : won_medals = 28)
  (c2 : total_medals = 150)
  (c3 : choices = [1/4, 1/5, 1/6, 1/7, 1/8])
  (c4 : fraction_won = 28 / 150) :
  abs (fraction_won - 1/5) < abs (fraction_won - 1/4) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/6) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/7) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/8) := 
sorry

end closest_fraction_to_team_aus_medals_l115_115265


namespace sum_largest_and_second_smallest_l115_115950

-- Define the list of numbers
def numbers : List ℕ := [10, 11, 12, 13, 14]

-- Define a predicate to get the largest number
def is_largest (n : ℕ) : Prop := ∀ x ∈ numbers, x ≤ n

-- Define a predicate to get the second smallest number
def is_second_smallest (n : ℕ) : Prop :=
  ∃ a b, (a ∈ numbers ∧ b ∈ numbers ∧ a < b ∧ b < n ∧ ∀ x ∈ numbers, (x < a ∨ x > b))

-- The main goal: To prove that the sum of the largest number and the second smallest number is 25
theorem sum_largest_and_second_smallest : 
  ∃ l s, is_largest l ∧ is_second_smallest s ∧ l + s = 25 := 
sorry

end sum_largest_and_second_smallest_l115_115950


namespace sqrt_square_eq_self_iff_ge_one_l115_115203

theorem sqrt_square_eq_self_iff_ge_one (x : ℝ) : (sqrt ((x - 1)^2) = x - 1) → (x ≥ 1) :=
by
  sorry

end sqrt_square_eq_self_iff_ge_one_l115_115203


namespace function_parallel_l115_115875

theorem function_parallel {x y : ℝ} (h : y = -2 * x + 1) : 
    ∀ {a : ℝ}, y = -2 * a + 3 -> y = -2 * x + 1 := by
    sorry

end function_parallel_l115_115875


namespace exam_duration_l115_115412

variable (total_questions : ℕ) (type_a_questions : ℕ) (type_a_time : ℚ) (examination_time : ℚ)

-- Conditions
def condition1 := total_questions = 200
def condition2 := type_a_questions = 50
def condition3 := type_a_time = 72
def condition4 (time_per_a time_per_b : ℚ) := time_per_a = 2 * time_per_b

-- Prove total examination time is 3 hours
theorem exam_duration (time_per_a time_per_b : ℚ) (total_time : ℚ) :
  condition1 →
  condition2 →
  condition3 →
  condition4 time_per_a time_per_b →
  total_time = (type_a_time + (total_questions - type_a_questions) * (time_per_a / 2)) →
  total_time / 60 = 3 := by
  sorry

end exam_duration_l115_115412


namespace triangle_angle_A_l115_115396

theorem triangle_angle_A (C : ℝ) (c : ℝ) (a : ℝ) 
  (hC : C = 45) (hc : c = Real.sqrt 2) (ha : a = Real.sqrt 3) :
  (∃ A : ℝ, A = 60 ∨ A = 120) :=
by
  sorry

end triangle_angle_A_l115_115396


namespace sample_mean_and_variance_significant_improvement_l115_115599

variable (x y : Fin 10 → ℝ)
variable xi_vals : Array ℝ := #[545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
variable yi_vals : Array ℝ := #[536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

noncomputable def z (i : Fin 10) : ℝ := xi_vals[i] - yi_vals[i]

noncomputable def mean_z : ℝ :=
  (1 / 10) * Array.foldr (λ i acc => acc + z i) 0 (Array.indices z)

noncomputable def var_z : ℝ :=
  (1 / 10) * Array.foldr (λ i acc => acc + (z i - mean_z) ^ 2) 0 (Array.indices z)

theorem sample_mean_and_variance :
  mean_z = 11 ∧ var_z = 61 := 
  by
  sorry

theorem significant_improvement : 
  mean_z ≥ 2 * Real.sqrt (var_z / 10) :=
  by
  sorry

end sample_mean_and_variance_significant_improvement_l115_115599


namespace problem_solution_l115_115592

def z (xi yi : ℕ) : ℕ := xi - yi

def z_vals (x y : Fin 10 → ℕ) : Fin 10 → ℕ := fun i => z (x i) (y i)

def mean (z : Fin 10 → ℕ) : ℕ :=
  (∑ i in Finset.univ, z i) / 10

def variance (z : Fin 10 → ℕ) (mean_z : ℕ) : ℕ :=
  (∑ i in Finset.univ, (z i - mean_z)^2) / 10

def significant_improvement (mean_z : ℕ) (var_z : ℕ) : Prop :=
  mean_z >= 2 * Real.sqrt (var_z / 10)

-- Given data
def x : Fin 10 → ℕ
| 0 => 545
| 1 => 533
| 2 => 551
| 3 => 522
| 4 => 575
| 5 => 544
| 6 => 541
| 7 => 568
| 8 => 596
| 9 => 548

def y : Fin 10 → ℕ
| 0 => 536
| 1 => 527
| 2 => 543
| 3 => 530
| 4 => 560
| 5 => 533
| 6 => 522
| 7 => 550
| 8 => 576
| 9 => 536

/-- The final proof statements -/
theorem problem_solution :
  let z_vals := z_vals x y
  let mean_z := mean z_vals
  let var_z := variance z_vals mean_z
  mean_z = 11 ∧ var_z = 61 ∧ significant_improvement mean_z var_z := 
by
  sorry

end problem_solution_l115_115592


namespace train_crossing_time_approx_l115_115639

noncomputable def crossing_time
  (train_length : ℝ) (train_speed_kmph : ℝ) (man_speed_kmph : ℝ) : ℝ :=
let train_speed_mps := train_speed_kmph * 1000 / 3600 in
let man_speed_mps := man_speed_kmph * 1000 / 3600 in
let relative_speed_mps := train_speed_mps + man_speed_mps in
train_length / relative_speed_mps

theorem train_crossing_time_approx
  (train_length : ℝ) (train_speed_kmph : ℝ) (man_speed_kmph : ℝ)
  (h_train_length : train_length = 780)
  (h_train_speed : train_speed_kmph = 60)
  (h_man_speed : man_speed_kmph = 6) :
  abs (crossing_time train_length train_speed_kmph man_speed_kmph - 42.55) < 0.01 :=
by
  sorry

end train_crossing_time_approx_l115_115639


namespace sum_of_squares_of_roots_l115_115673

theorem sum_of_squares_of_roots :
  ∀ (x₁ x₂ : ℝ), (∀ a b c : ℝ, (a ≠ 0) →
  6 * x₁ ^ 2 + 5 * x₁ - 4 = 0 ∧ 6 * x₂ ^ 2 + 5 * x₂ - 4 = 0 →
  x₁ ^ 2 + x₂ ^ 2 = 73 / 36) :=
by
  sorry

end sum_of_squares_of_roots_l115_115673


namespace range_of_eccentricity_l115_115465

variables {a b : ℝ} (h1 : a > b > 0) {e : ℝ}
noncomputable def ellipse_eq : Prop := ∀ (x y : ℝ),
  (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def circle_condition (x y : ℝ) : Prop :=
  ∃ c : ℝ, x = c ∧ (y = b^2 / a ∨ y = -b^2 / a) ∧
    (c^2 + b^2 = a^2) ∧
    (2 * x^2 - y^2 > 0)

theorem range_of_eccentricity:
  (∀ (x y : ℝ), ellipse_eq → circle_condition x y) →
  0 < e ∧ e < (sqrt 6 - sqrt 2) / 2 :=
begin
  sorry
end

end range_of_eccentricity_l115_115465


namespace quadratic_two_distinct_real_roots_l115_115000

theorem quadratic_two_distinct_real_roots 
    (a b c : ℝ)
    (h1 : a > 0)
    (h2 : c < 0) : 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + bx + c = 0 := 
sorry

end quadratic_two_distinct_real_roots_l115_115000


namespace sum_y_coordinates_of_intersection_points_l115_115972

theorem sum_y_coordinates_of_intersection_points (h k r : ℝ) (h₀ : h = 5) (k₀ : k = -3) (r₀ : r = 13) :
  ∑ y in {y : ℝ | (0 - h) ^ 2 + (y - k) ^ 2 = r ^ 2}, y = -6 :=
sorry

end sum_y_coordinates_of_intersection_points_l115_115972


namespace tap_A_rate_l115_115637

theorem tap_A_rate (V : ℕ) (t1 t2 t3 t4 : ℕ) (B_rate A_rate combined_rate : ℕ): 
  V = 36 →
  t1 = 1 →
  t2 = 3 →
  t3 = 20 →
  t4 = 10 →
  B_rate = ((V / t2) / t3) →
  combined_rate = (V / t4) →
  A_rate = combined_rate - B_rate →
  A_rate = 3 :=
by
  intros V_eq t1_eq t2_eq t3_eq t4_eq B_rate_eq combined_rate_eq A_rate_eq
  rw [V_eq, t1_eq, t2_eq, t3_eq, t4_eq, B_rate_eq, combined_rate_eq, A_rate_eq]
  sorry

end tap_A_rate_l115_115637


namespace mean_proportional_approx_l115_115573

noncomputable def a : Real := Real.sqrt 45
noncomputable def b : Real := (7 / 3) * Real.pi
noncomputable def m : Real := Real.sqrt (a * b)

theorem mean_proportional_approx :
  m ≈ 7.014 := by
  sorry

end mean_proportional_approx_l115_115573


namespace train_cross_first_platform_in_15_seconds_l115_115999

noncomputable def length_of_train : ℝ := 100
noncomputable def length_of_second_platform : ℝ := 500
noncomputable def time_to_cross_second_platform : ℝ := 20
noncomputable def length_of_first_platform : ℝ := 350
noncomputable def speed_of_train := (length_of_train + length_of_second_platform) / time_to_cross_second_platform
noncomputable def time_to_cross_first_platform := (length_of_train + length_of_first_platform) / speed_of_train

theorem train_cross_first_platform_in_15_seconds : time_to_cross_first_platform = 15 := by
  sorry

end train_cross_first_platform_in_15_seconds_l115_115999


namespace sin_C_l115_115814

theorem sin_C (A B C : ℝ) (hA1 : sin A = 4/5) (hB1 : cos B = 12/13) (hABC : A + B + C = π) :
  sin C = 63/65 :=
sorry

end sin_C_l115_115814


namespace cuboid_volume_l115_115957

variable (length width height : ℕ)

-- Conditions given in the problem
def cuboid_edges := (length = 2) ∧ (width = 5) ∧ (height = 8)

-- Mathematically equivalent statement to be proved
theorem cuboid_volume : cuboid_edges length width height → length * width * height = 80 := by
  sorry

end cuboid_volume_l115_115957


namespace problem_statement_l115_115807

noncomputable def point_P := (0 : ℝ, real.sqrt 3)
noncomputable def param_line (t : ℝ) := (1/2 * t, real.sqrt 3 + real.sqrt 3 / 2 * t)

def polar_curve (ρ : ℝ) (θ : ℕ) := ρ^2 = 4 / (1 + (real.cos θ)^2)

def cartesian_curve (x y : ℝ) := (x^2 / 2 + y^2 / 4 = 1)

def general_eq_of_line (x y : ℝ) := (real.sqrt 3 * x - y = -real.sqrt 3)

theorem problem_statement :
  (∀x y, general_eq_of_line x y) ∧
  (∀x y, cartesian_curve x y) ∧
  (∃ t1 t2 : ℝ, 
    (param_line t1).1 = (param_line t2).1 ∧ 
    (param_line t1).2 = (param_line t2).2 ∧ 
    |t1 - t2| = 4 * real.sqrt 14 / 5 ∧ 
    (1 / |t1| + 1 / |t2|) = real.sqrt 14) :=
by
  intros,
  sorry

end problem_statement_l115_115807


namespace probability_of_first_card_queen_and_second_card_spade_l115_115190

noncomputable def probability_first_queen_second_spade : ℚ :=
  let p_first_queen := 4 / 52 in
  let p_second_spade_given_first_non_spade_queen := 13 / 51 in
  let p_second_spade_given_first_spade_queen := 12 / 51 in
  (3 / 52 * p_second_spade_given_first_non_spade_queen) + (1 / 52 * 4 / 17)

theorem probability_of_first_card_queen_and_second_card_spade :
  probability_first_queen_second_spade = 289 / 14968 :=
by
  sorry

end probability_of_first_card_queen_and_second_card_spade_l115_115190


namespace square_area_relation_l115_115390

variable {lA lB : ℝ}

theorem square_area_relation (h : lB = 4 * lA) : lB^2 = 16 * lA^2 :=
by sorry

end square_area_relation_l115_115390


namespace correct_set_k_l115_115237

-- Define the probability function
def probability (d : ℕ) : ℝ := real.log10 (d + 1) - real.log10 d

-- Define the problem conditions
def prob_4 := probability 4
def total_prob (k : Finset ℕ) : ℝ := k.sum probability

-- The main statement to prove
theorem correct_set_k (k : Finset ℕ) :
  (3 : ℝ) * prob_4 = total_prob k ↔ k = {6, 7, 8, 9} :=
by
  sorry

end correct_set_k_l115_115237


namespace meat_required_for_hamburgers_l115_115470

theorem meat_required_for_hamburgers :
  (∀ (pounds_per_hamburger : ℝ), pounds_per_hamburger = 4 / 10) →
  (∀ (hamburgers : ℕ), (hamburgers = 30) → 12 = (4 / 10) * 30) :=
by
  intros h1 h2 h3
  rw h3
  norm_num
  sorry

end meat_required_for_hamburgers_l115_115470


namespace average_speed_last_60_minutes_l115_115095

theorem average_speed_last_60_minutes
  (total_distance : ℝ) (total_time : ℝ) (first_segment_time : ℝ)
  (second_segment_time : ℝ) (first_segment_speed : ℝ) (second_segment_speed : ℝ)
  (total_minutes : 120) (total_miles : 120) :
  total_distance = 120 ∧ total_time = 2 ∧ first_segment_time = 0.5 ∧ second_segment_time = 0.5 ∧
  first_segment_speed = 50 ∧ second_segment_speed = 70 →
  (total_distance - ((first_segment_speed * first_segment_time) + 
                     (second_segment_speed * second_segment_time))) / (total_time - first_segment_time - second_segment_time) = 60 := by
  sorry

end average_speed_last_60_minutes_l115_115095


namespace suzie_coin_flips_l115_115888

theorem suzie_coin_flips (m n : ℕ) (hp : Nat.coprime m n) :
    let favorable_outcomes := 12
    let total_outcomes := 2^6
    let probability := favorable_outcomes / total_outcomes
    ∃ m n, probability = m / n ∧ Nat.coprime m n ∧ m + n = 19 := 
  sorry

end suzie_coin_flips_l115_115888


namespace pencils_per_row_l115_115294

theorem pencils_per_row (total_pencils : ℕ) (rows : ℕ) (h_total : total_pencils = 30) (h_rows : rows = 6) :
  total_pencils / rows = 5 :=
by
  rw [h_total, h_rows]
  norm_num

end pencils_per_row_l115_115294


namespace statue_of_liberty_model_ratio_l115_115916

theorem statue_of_liberty_model_ratio : 
  ∀ (height_statue : ℝ) (height_model : ℝ), 
  height_statue = 151 → height_model = 5 → 
  151 / 5 = 30.2 :=
by
  intros height_statue height_model h1 h2
  rw [h1, h2]
  sorry

end statue_of_liberty_model_ratio_l115_115916


namespace pyramid_cross_section_distance_l115_115931

theorem pyramid_cross_section_distance
  (area1 area2 : ℝ) (distance : ℝ)
  (h1 : area1 = 100 * Real.sqrt 3) 
  (h2 : area2 = 225 * Real.sqrt 3) 
  (h3 : distance = 5) : 
  ∃ h : ℝ, h = 15 :=
by
  sorry

end pyramid_cross_section_distance_l115_115931


namespace common_divisors_9240_6300_l115_115372

def num_common_divisors (a b : ℕ) : ℕ :=
  (Nat.gcd a b).divisors.count

theorem common_divisors_9240_6300 : num_common_divisors 9240 6300 = 12 := by
  sorry

end common_divisors_9240_6300_l115_115372


namespace problem_solution_l115_115767

noncomputable def a_k (k : ℕ) : ℝ × ℝ := 
  (Real.cos (k * Real.pi / 6), Real.sin (k * Real.pi / 6) + Real.cos (k * Real.pi / 6))

def dot_product (u v : ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2

def sum_dot_products : ℝ := 
  (Finset.range 12).sum (λ k, dot_product (a_k k) (a_k (k + 1)))

theorem problem_solution : sum_dot_products = 9 * Real.sqrt 3 := 
  sorry

end problem_solution_l115_115767


namespace monotonically_increasing_interval_l115_115507

noncomputable def f (x : ℝ) : ℝ := Math.sin (2 * x) - Math.cos (2 * x)

-- Define the interval to be checked
def interval_D : Set ℝ := {x | -π / 8 ≤ x ∧ x ≤ 3 * π / 8}

-- Define the monotonicity check property
def is_monotonically_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

-- The Lean statement that we need to prove
theorem monotonically_increasing_interval :
  is_monotonically_increasing f interval_D := sorry

end monotonically_increasing_interval_l115_115507


namespace avg_speed_trip_l115_115776

def avg_speed_for_trip (D T : ℝ) (speed_up speed_down : ℝ) : ℝ :=
  (2 * D) / ((D / speed_up) + (D / speed_down))

theorem avg_speed_trip :
  let avg_speed := avg_speed_for_trip 1 1 96 88
  avg_speed ≈ 91.83 := by
  sorry

end avg_speed_trip_l115_115776


namespace no_solution_l115_115694

theorem no_solution (x : ℝ) (h : x ≥ 4) :
  ¬ (sqrt (x + 5 - 6 * sqrt (x - 4)) + sqrt (x + 18 - 8 * sqrt (x - 4)) = 2) :=
sorry

end no_solution_l115_115694


namespace fill_time_difference_correct_l115_115619

-- Define the time to fill one barrel in normal conditions
def normal_fill_time_per_barrel : ℕ := 3

-- Define the time to fill one barrel with a leak
def leak_fill_time_per_barrel : ℕ := 5

-- Define the number of barrels to fill
def barrels_to_fill : ℕ := 12

-- Define the time to fill 12 barrels in normal conditions
def normal_fill_time : ℕ := normal_fill_time_per_barrel * barrels_to_fill

-- Define the time to fill 12 barrels with a leak
def leak_fill_time : ℕ := leak_fill_time_per_barrel * barrels_to_fill

-- Define the time difference
def time_difference : ℕ := leak_fill_time - normal_fill_time

theorem fill_time_difference_correct : time_difference = 24 := by
  sorry

end fill_time_difference_correct_l115_115619


namespace cos_difference_l115_115290

theorem cos_difference (a b : ℝ) : 
  cos (a + b) - cos (a - b) = -2 * sin a * sin b := sorry

end cos_difference_l115_115290


namespace triangles_in_2x2_grid_l115_115670

theorem triangles_in_2x2_grid :
  let num_squares := 2 * 2 in
  let num_smallest_triangles := 2 * num_squares in
  let num_larger_triangles_in_squares := num_squares in
  let num_larger_triangles_across_squares := 4 in
  let num_largest_triangle := 1 in
  (num_smallest_triangles + num_larger_triangles_in_squares + num_larger_triangles_across_squares + num_largest_triangle) = 17 := 
by {
  sorry -- Proof not required per instructions
}

end triangles_in_2x2_grid_l115_115670


namespace walking_distance_difference_l115_115809

theorem walking_distance_difference 
  (total_distance : ℕ)
  (common_ratio : ℚ)
  (days : ℕ)
  (distance_on_day : ℕ → ℚ)
  (h1 : total_distance = 378)
  (h2 : common_ratio = 1/2)
  (h3 : days = 6)
  (h4 : ∀ n, distance_on_day n = distance_on_day 1 * common_ratio ^ (n - 1))
  (h5 : ∑ i in (finset.range days), distance_on_day (i + 1) = total_distance) :
  (distance_on_day 4 - distance_on_day 6) = 18 := by
  sorry

end walking_distance_difference_l115_115809


namespace min_value_is_144_l115_115451

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  x^2 + 4 * x * y + 4 * y^2 + 3 * z^2

theorem min_value_is_144 (x y z : ℝ) (hxyz : x * y * z = 48) : 
  ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ xyz = 48 ∧ min_value_expression x y z = 144 :=
by 
  sorry

end min_value_is_144_l115_115451


namespace correct_volume_of_pyramid_l115_115305

noncomputable def volume_of_pyramid (h Q : ℝ) : ℝ :=
  (1 / 3) * h * sqrt(3) * (sqrt(h^4 + 12 * Q^2) - h^2)

theorem correct_volume_of_pyramid (h Q : ℝ) :
  volume_of_pyramid h Q = (1 / 3) * h * sqrt(3) * (sqrt(h^4 + 12 * Q^2) - h^2) :=
by
  sorry

end correct_volume_of_pyramid_l115_115305


namespace F_8228_smallest_twin_egg_number_l115_115612

def twin_egg_number (m : ℕ) : Prop :=
  (1000 * ((m / 1000) % 10)) + (100 * ((m / 100) % 10)) + (10 * ((m / 10) % 10)) + ((m % 10) = m) ∧
  ((m / 1000) % 10) = (m % 10) ∧ ((m / 100) % 10) = ((m / 10) % 10)

def F (m : ℕ) (m' : ℕ) : ℕ :=
  (m - m') / 11

theorem F_8228 : F 8228 2882 = 486 := by
  sorry

theorem smallest_twin_egg_number : 
  ∃ m : ℕ, twin_egg_number m ∧ (m / 1000) % 10 ≠ (m / 100) % 10 ∧ 
  (3 * (((m / 1000) % 10) - ((m / 100) % 10))) ^ 2 = (F m (1001 * ((m / 100) % 10) + 110 * ((m / 10) % 10) + ((m / 10) % 10) + (10 * (m % 10)))) ∧
  m = 4114 :=
  sorry

end F_8228_smallest_twin_egg_number_l115_115612


namespace line_AB_minimized_PC_AB_l115_115007

noncomputable def circle (x y : ℝ) := x^2 + 2 * x + y^2 = 0
noncomputable def line_l (x y : ℝ) := x + y - 2 = 0

theorem line_AB_minimized_PC_AB :
  ∃ (x y : ℝ) (P : ℝ → ℝ → Prop) (A B : ℝ → ℝ → Prop), 
    (P x y ↔ line_l x y) ∧ 
    (A x y ↔ circle x y) ∧ 
    (B x y ↔ circle x y) ∧ 
    ∀ (AB : ℝ → ℝ → Prop), 
    (∀ x y, AB x y ↔ 3 * x + 3 * y + 1 = 0) :=
sorry

end line_AB_minimized_PC_AB_l115_115007


namespace symmetric_line_x_axis_l115_115173

theorem symmetric_line_x_axis (y : ℝ → ℝ) (x : ℝ) :
  (∀ x, y x = 2 * x + 1) → (∀ x, -y x = 2 * x + 1) → y x = -2 * x -1 :=
by
  intro h1 h2
  sorry

end symmetric_line_x_axis_l115_115173


namespace isosceles_tetrahedron_faces_acute_l115_115615

-- Given conditions
variables (A B C D : Point) (isosceles_tetrahedron : AB = CD ∧ AC = BD ∧ AD = BC)

-- Geometry definitions
open Geometry

theorem isosceles_tetrahedron_faces_acute
  (isosceles_tetrahedron : AB = CD ∧ AC = BD ∧ AD = BC) :
  (∀ (X Y Z : Point), {X, Y, Z} ∈ faces_of_tetrahedron A B C D → acute_triangle X Y Z) :=
sorry

end isosceles_tetrahedron_faces_acute_l115_115615


namespace max_happy_monkeys_l115_115986

-- Definitions for given problem
def pears := 20
def bananas := 30
def peaches := 40
def mandarins := 50
def fruits (x y : Nat) := x + y

-- The theorem to prove
theorem max_happy_monkeys : 
  ∃ (m : Nat), m = (pears + bananas + peaches) / 2 ∧ m ≤ mandarins :=
by
  sorry

end max_happy_monkeys_l115_115986


namespace inequality_solution_l115_115481

-- Conditions and constraints
noncomputable def u (x : ℝ) : ℝ := (5 / (2 * Real.pi)) * Real.arccos x
noncomputable def v (x : ℝ) : ℝ := (10 / (3 * Real.pi)) * Real.arcsin x

theorem inequality_solution (x : ℝ) :
  (0 ≤ u(x) ∧ u(x) ≤ 1) ∧ (0 ≤ v(x) ∧ v(x) ≤ 1) ∧ (u(x)^2 + v(x)^2 > 1) ∧ (4 * u(x) + 3 * v(x) = 5)
  → (Real.arcsin (u(x)) > Real.arccos (v(x)) ↔
     (x ∈ set.Icc (Real.cos (2 * Real.pi / 5)) (Real.cos (8 * Real.pi / 25)) ∪
      set.Icc (Real.cos (8 * Real.pi / 25)) (Real.cos (Real.pi / 5)))) :=
  by sorry

end inequality_solution_l115_115481


namespace part1_part2_l115_115661

-- Part 1
theorem part1 : (9 / 4) ^ (1 / 2) - (-2.5) ^ 0 - (8 / 27) ^ (2 / 3) + (3 / 2) ^ (-2) = 1 / 2 := 
by sorry

-- Part 2
theorem part2 (lg : ℝ → ℝ) -- Assuming a hypothetical lg function for demonstration
  (lg_prop1 : lg 10 = 1)
  (lg_prop2 : ∀ x y, lg (x * y) = lg x + lg y) :
  (lg 5) ^ 2 + lg 2 * lg 50 = 1 := 
by sorry

end part1_part2_l115_115661


namespace part1_part2_part3_l115_115029

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * sin x

-- Define the function g
def g (x : ℝ) : ℝ := Real.exp x

-- Part 1: Define the function F and prove its monotonicity
theorem part1 (a : ℝ) (h : 0 < a ∧ a ≤ 1) :
  ∃ (F : ℝ → ℝ), (∀ x ∈ Ioo 0 1, StrictMonoOn F (Ioo 0 1)) := by
  let F := λ x, f a (1 - x) + Real.log x
  sorry

-- Part 2: Prove the trigonometric inequality
theorem part2 (n : ℕ) (h : n > 0) :
  ∑ k in Finset.range(n + 1).filter (λ k, k > 1), sin (1/(k + 1)^2 : ℝ) < Real.log 2 := by
  sorry

-- Part 3: Prove the inequality for G(x)
theorem part3 (m : ℝ) (h : m < 0):
  ∃ (k : ℤ), k = 2 ∧ ∀ x > 0, g x - m * x^2 - 2 * (x + 1) + k > 0 := by
  let k := 2
  sorry

end part1_part2_part3_l115_115029


namespace percentage_of_remaining_left_l115_115803

noncomputable def initial_population : ℕ := 4543
noncomputable def percentage_died : ℝ := 0.08
noncomputable def reduced_population : ℕ := 3553

noncomputable def number_died : ℕ := (percentage_died * initial_population).to_nat
noncomputable def remaining_after_bombardment : ℕ := initial_population - number_died
noncomputable def number_left_due_to_fear : ℕ := remaining_after_bombardment - reduced_population

noncomputable def percentage_left_due_to_fear : ℝ := (number_left_due_to_fear : ℝ) / (remaining_after_bombardment : ℝ) * 100

theorem percentage_of_remaining_left (h_initial_population : initial_population = 4543)
    (h_percentage_died : percentage_died = 0.08)
    (h_reduced_population : reduced_population = 3553)
    (h_number_died : number_died = 363) -- derived rounding
    (h_remaining_after_bombardment : remaining_after_bombardment = 4180) -- calculated
    (h_number_left_due_to_fear : number_left_due_to_fear = 627) -- calculated
    : percentage_left_due_to_fear = 14.98 :=
by
  sorry

end percentage_of_remaining_left_l115_115803


namespace quadratic_function_series_sum_l115_115832

open Real

noncomputable def P (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 7

theorem quadratic_function_series_sum :
  (∀ (x : ℝ), 0 < x ∧ x < 1 →
    (∑' n, P n * x^n) = (16 * x^2 - 11 * x + 7) / (1 - x)^3) :=
sorry

end quadratic_function_series_sum_l115_115832


namespace medians_intersection_form_square_l115_115949
-- Import the entire Mathlib library

-- Define the square and point inside
variable (A B C D M : Point)

-- Define the assumption that ABCD is a square and M is inside ABCD
variable (is_square : Square ABCD)
variable (M_in_square : InSquare M ABCD)

-- The proof statement: the points of intersection of the medians of triangles 
-- ABM, BCM, CDM, and DAM form a square.
theorem medians_intersection_form_square 
    (A B C D M : Point) 
    (is_square : Square ABCD) 
    (M_in_square : InSquare M ABCD) : 
    is_square (medians_intersections_square ABCD M) :=
    sorry

end medians_intersection_form_square_l115_115949


namespace cost_of_mobile_phone_is_6000_l115_115150

-- Define given conditions
variables (C_R : ℕ) (L_R : ℕ) (P_M : ℕ) (Profit : ℕ)
variable M : ℕ
variable SellingPriceRefrigerator : ℕ
variable SellingPriceMobilePhone : ℕ

-- Define the refrigerator cost price
def costPriceRefrigerator : ℕ := 15000

-- Define the refrigerator loss percentage
def lossPercentageRefrigerator : ℕ := 4

-- Define the profit percentage for the mobile phone
def profitPercentageMobilePhone : ℕ := 10

-- Define the overall profit
def overallProfit : ℕ := 200

-- Selling price computation based on cost price and loss/profit
def sellingPriceRefrigerator : ℕ := costPriceRefrigerator - (costPriceRefrigerator * lossPercentageRefrigerator / 100)
def sellingPriceMobilePhone (M : ℕ) : ℕ := M + (M * profitPercentageMobilePhone / 100)

-- Define the equation representing overall profit
def overallProfitEquation (M : ℕ) : Prop :=
  sellingPriceRefrigerator + sellingPriceMobilePhone M = costPriceRefrigerator + M + overallProfit

-- Statement to prove the cost of the mobile phone
theorem cost_of_mobile_phone_is_6000 : overallProfitEquation 6000 :=
sorry

end cost_of_mobile_phone_is_6000_l115_115150


namespace sufficient_but_not_necessary_condition_for_intersections_l115_115364

theorem sufficient_but_not_necessary_condition_for_intersections
  (k : ℝ) (h : 0 < k ∧ k < 3) :
  ∃ x y : ℝ, (x - y - k = 0) ∧ ((x - 1)^2 + y^2 = 2) :=
sorry

end sufficient_but_not_necessary_condition_for_intersections_l115_115364


namespace last_two_digits_of_sum_l115_115701

theorem last_two_digits_of_sum :
  (7! + 14! + 21! + 28! + 35! + 42! + 49! + 56! + 63! + 70! + 77! + 84! + 91! + 98!) % 100 = 40 :=
by
  sorry

end last_two_digits_of_sum_l115_115701


namespace range_of_a_l115_115391

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a-1) * x^2 + 2 * (a-1) * x - 4 ≥ 0 -> false) ↔ -3 < a ∧ a ≤ 1 := by
  sorry

end range_of_a_l115_115391


namespace volume_of_mixture_l115_115538

theorem volume_of_mixture
    (weight_a : ℝ) (weight_b : ℝ) (ratio_a_b : ℝ) (total_weight : ℝ)
    (h1 : weight_a = 900) (h2 : weight_b = 700)
    (h3 : ratio_a_b = 3/2) (h4 : total_weight = 3280) :
    ∃ Va Vb : ℝ, (Va / Vb = ratio_a_b) ∧ (weight_a * Va + weight_b * Vb = total_weight) ∧ (Va + Vb = 4) := 
by
  sorry

end volume_of_mixture_l115_115538


namespace monotonically_decreasing_interval_l115_115745

noncomputable def f (x : ℝ) : ℝ :=
  (2 * Real.exp 2) * Real.exp (x - 2) - 2 * x + 1/2 * x^2

theorem monotonically_decreasing_interval :
  ∀ x : ℝ, x < 0 → ((2 * Real.exp x - 2 + x) < 0) :=
by
  sorry

end monotonically_decreasing_interval_l115_115745


namespace park_not_packed_l115_115163

variable {T : Type}
variable (temperature sunny packed : T → Prop)
variable (t : T)

theorem park_not_packed (h1 : ∀ t, temperature t ≥ 70 → sunny t → packed t)
    (h2 : ¬ packed t) : temperature t < 70 ∨ ¬ sunny t := 
sorry

end park_not_packed_l115_115163


namespace part1_part2_l115_115812

def a (n : ℕ) : ℕ
| 0       := 2
| 1       := 12
| 2       := 54
| (n+3)   := 3 * a (n+2) + 2 * 3 ^ (n+2)

theorem part1 (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a (2) = 12) (h3 : a (3) = 54)
  (h_geom : ∀ n, a (n + 2) - 3 * a (n + 1) = (6 * 3 ^ n)) :
  ∃ d, ∀ n, (a (n + 1) / 3 ^ n) - (a n / 3 ^ (n - 1)) = d :=
sorry

theorem part2 (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a (2) = 12) (h3 : a (3) = 54)
  (h_geom : ∀ n, a (n + 2) - 3 * a (n + 1) = (6 * 3 ^ n)) :
  ∀ n, ∑ i in finset.range(n), a i = (n - 1 / 2) * 3 ^ n + 1 / 2 :=
sorry

end part1_part2_l115_115812


namespace smallest_possible_s_l115_115913

theorem smallest_possible_s : 
  (∃ s : ℕ, 7.5 + s > 12 ∧ 7.5 + 12 > s ∧ 12 + s > 7.5 ∧ ∀ t : ℕ, (7.5 + t > 12 ∧ 7.5 + 12 > t ∧ 12 + t > 7.5 → t ≥ 5)) :=
by
  use 5
  split
  · linarith
  split
  · linarith
  split
  · linarith
  sorry

end smallest_possible_s_l115_115913


namespace find_1992nd_term_l115_115130

noncomputable def f (n : ℕ) : ℕ :=
  n + (nat.floor (real.sqrt n))

theorem find_1992nd_term :
  f 1992 = 2036 :=
by
sorry

end find_1992nd_term_l115_115130


namespace circle_equation_and_tangent_lines_l115_115606

theorem circle_equation_and_tangent_lines :
  (∃ (a b r : ℝ), (a = 3) ∧ (b = 4) ∧ (r = 5) ∧ 
  (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2) ∧
  (y = - (3/4) * x ∨ (x + y + 5 * real.sqrt 2 - 7 = 0) ∨ (x + y - 5 * real.sqrt 2 - 7 = 0))) :=
begin
  sorry
end

end circle_equation_and_tangent_lines_l115_115606


namespace differences_not_all_different_l115_115279

theorem differences_not_all_different
    (a b : Finset ℕ) 
    (ha : a = {1, 2, 3, 4, 5, 6, 7}) 
    (hb : b = {1, 2, 3, 4, 5, 6, 7})
    (a_list : List ℕ) 
    (b_list : List ℕ) 
    (ha_perm : a_list ~ [1, 2, 3, 4, 5, 6, 7])
    (hb_perm : b_list ~ [1, 2, 3, 4, 5, 6, 7]) 
    : (Finset.image (λ i, |a_list.nthLe i sorry - b_list.nthLe i sorry|) (Finset.range 7) ≠ {0, 1, 2, 3, 4, 5, 6}) :=
by 
  sorry

end differences_not_all_different_l115_115279


namespace probability_correct_l115_115286

def days := {1, 2, 3, 4, 5, 6, 7, 8}

/-- Representation of a random choice of 3 consecutive days -/
def consecutive_days (n : ℕ) :=
  n ∈ days ∧ n + 1 ∈ days ∧ n + 2 ∈ days

/-- Number of valid choices for any 3 consecutive days -/
def total_choices := 6

/-- Number of valid choices within the first to the fourth day -/
def valid_choices :=
  (consecutive_days 1 ∨ consecutive_days 2)

/-- Calculating the desired probability -/
noncomputable def probability :=
  ∃ (prob : ℚ), prob = (2 / total_choices) ∧ prob = 1 / 3

theorem probability_correct : probability :=
by
  sorry

end probability_correct_l115_115286


namespace range_of_a_l115_115009

variables {f : ℝ → ℝ} (a : ℝ)

-- Conditions definitions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f(x) = f(-x)
def is_monotone_increasing_from_zero (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f(x) ≤ f(y)

-- The main statement to prove
theorem range_of_a (even_f : is_even_function f) 
                   (mono_f : is_monotone_increasing_from_zero f)
                   (ineq : f(log 2 a) + f(log (1/2) a) ≤ 2 * f 1) :
  1/2 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_a_l115_115009


namespace parabola_focus_l115_115896

noncomputable def parabola_focus_coordinates (a : ℝ) : ℝ × ℝ :=
  if a ≠ 0 then (0, 1 / (4 * a)) else (0, 0)

theorem parabola_focus {x y : ℝ} (a : ℝ) (h : a = 2) (h_eq : y = a * x^2) :
  parabola_focus_coordinates a = (0, 1 / 8) :=
by sorry

end parabola_focus_l115_115896


namespace exists_differs_by_no_more_than_l115_115879

theorem exists_differs_by_no_more_than (x : ℝ) (n : ℕ) (h_pos: n > 0) :
    ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ |frac (k * x)| ≤ 1 / n :=
by
  sorry

end exists_differs_by_no_more_than_l115_115879


namespace find_angle_l115_115008

theorem find_angle (x : ℝ) (h : 90 - x = 2 * x + 15) : x = 25 :=
by
  sorry

end find_angle_l115_115008


namespace problem_part1_problem_part2_l115_115691

variable {a b c d t : Real}

/-- (1) Prove: If a + d > b + c, then |a - d| > |b - c| -/
theorem problem_part1 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * d = b * c) 
  (h1 : a + d > b + c) : |a - d| > |b - c| :=
by {
  -- proof skipped
  sorry
}

/-- (2) Prove the range of t given the equation -/
theorem problem_part2 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h : t * Real.sqrt(a^2 + b^2) * Real.sqrt(c^2 + d^2) = Real.sqrt(a^4 + c^4) + Real.sqrt(b^4 + d^4)) :
  ∀ t, t ≥ Real.sqrt 2 :=
by {
  -- proof skipped
  sorry
}

end problem_part1_problem_part2_l115_115691


namespace work_completion_time_l115_115644

theorem work_completion_time (x : ℕ) (h1 : ∀ B, ∀ A, A = 2 * B) (h2 : (1/x + 1/(2*x)) * 4 = 1) : x = 12 := 
sorry

end work_completion_time_l115_115644


namespace determine_some_number_l115_115520

theorem determine_some_number (x : ℝ) (n : ℝ) (hx : x = 1.5) (h : (3 + 2 * x)^5 = (1 + n * x)^4) : n = 10 / 3 :=
by {
  sorry
}

end determine_some_number_l115_115520


namespace direction_vector_ell_l115_115278

structure Vector2 where
  x : ℚ
  y : ℚ

def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![[3 / 5, 4 / 5], [4 / 5, -3 / 5]]

def line_direction : Vector2 :=
  ⟨2, 1⟩

theorem direction_vector_ell (a b : ℚ) (h₁ : reflection_matrix.mul_vec ⟨![a, b]⟩ = ⟨![a, b]⟩)
  (h₂ : a > 0) (h₃ : Int.gcd (Int.natAbs (Int.ofRat a)) (Int.natAbs (Int.ofRat b)) = 1) :
  a = 2 ∧ b = 1 := by
  sorry

end direction_vector_ell_l115_115278


namespace bryden_receives_22_50_dollars_l115_115230

-- Define the face value of a regular quarter
def face_value_regular : ℝ := 0.25

-- Define the number of regular quarters Bryden has
def num_regular_quarters : ℕ := 4

-- Define the face value of the special quarter
def face_value_special : ℝ := face_value_regular * 2

-- The collector pays 15 times the face value for regular quarters
def multiplier : ℝ := 15

-- Calculate the total face value of all quarters
def total_face_value : ℝ := (num_regular_quarters * face_value_regular) + face_value_special

-- Calculate the total amount Bryden will receive
def total_amount_received : ℝ := multiplier * total_face_value

-- Prove that the total amount Bryden will receive is $22.50
theorem bryden_receives_22_50_dollars : total_amount_received = 22.50 :=
by
  sorry

end bryden_receives_22_50_dollars_l115_115230


namespace distance_from_P_to_origin_l115_115078

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_from_P_to_origin :
  distance (-1) 2 0 0 = Real.sqrt 5 :=
by
  sorry

end distance_from_P_to_origin_l115_115078


namespace num_subsets_of_set_ab_l115_115375

theorem num_subsets_of_set_ab : 
  ∀ (a b : Type), 
  ∃ s : set (set Type), 
    (s == {{}, {a}, {b}, {a, b}}) ∧ 
    s.card = 4 := 
by
  sorry

end num_subsets_of_set_ab_l115_115375


namespace count_valley_numbers_l115_115197

def is_valley_number (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n / 10) % 10 in
  let d3 := n % 10 in
  100 ≤ n ∧ n < 1000 ∧ d2 < d1 ∧ d2 < d3

theorem count_valley_numbers : 
  { n : ℕ | is_valley_number n }.to_finset.card = 240 := 
  by sorry

end count_valley_numbers_l115_115197


namespace students_earlier_than_hoseok_l115_115256

-- Definitions based on the conditions
def total_students : ℕ := 30
def students_later_than_hoseok : ℕ := 12

-- The proof problem
theorem students_earlier_than_hoseok (total_students = 30) (students_later_than_hoseok = 12) : 
  total_students - students_later_than_hoseok - 1 = 17 :=
sorry

end students_earlier_than_hoseok_l115_115256


namespace table_tennis_ball_cost_l115_115889

theorem table_tennis_ball_cost : 
  ∀ (soccer_ball_cost total_table_tennis_balls cost_of_soccer_balls : ℝ),
    cost_of_soccer_balls = 122.4 → 
    soccer_ball_cost = cost_of_soccer_balls / 36 →
    total_table_tennis_balls = 20 →
    ∀ (table_tennis_ball_cost : ℝ),
      table_tennis_ball_cost = soccer_ball_cost - 0.2 → 
      total_table_tennis_balls * table_tennis_ball_cost = 64 :=
by 
  intros soccer_ball_cost total_table_tennis_balls cost_of_soccer_balls H_cost eq_soccer_ball H_tt_balls table_tennis_ball_cost eq_table_tennis_ball
  have soccer_ball_cost_def : soccer_ball_cost = cost_of_soccer_balls / 36 := eq_soccer_ball
  have table_tennis_ball_cost_def : table_tennis_ball_cost = soccer_ball_cost - 0.2 := eq_table_tennis_ball
  rw [soccer_ball_cost_def, table_tennis_ball_cost_def]
  sorry

end table_tennis_ball_cost_l115_115889


namespace xy_divides_x2_plus_y2_plus_one_l115_115432

theorem xy_divides_x2_plus_y2_plus_one 
    (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : (x * y) ∣ (x^2 + y^2 + 1)) :
  (x^2 + y^2 + 1) / (x * y) = 3 := by
  sorry

end xy_divides_x2_plus_y2_plus_one_l115_115432


namespace gcd_A_is_7_l115_115309

def A (n : ℕ) : ℕ := 2^(3*n) + 3^(6*n+2) + 5^(6*n+2)

theorem gcd_A_is_7 : Nat.gcd_seq (A ∘ Nat.succ) 2000 = 7 := sorry

end gcd_A_is_7_l115_115309


namespace ratio_of_x_y_l115_115109

theorem ratio_of_x_y (x y : ℝ) (h₁ : 3 < (x - y) / (x + y)) (h₂ : (x - y) / (x + y) < 4) (h₃ : ∃ a b : ℤ, x = a * y / b ) (h₄ : x + y = 10) :
  x / y = -2 := sorry

end ratio_of_x_y_l115_115109


namespace sum_square_fraction_zero_l115_115110

theorem sum_square_fraction_zero (y : Fin 50 → ℝ) 
  (h₁ : (∑ i, y i) = 0) 
  (h₂ : (∑ i, y i / (1 + y i)) = 0) : 
  (∑ i, y i ^ 2 / (1 + y i)) = 0 :=
by sorry

end sum_square_fraction_zero_l115_115110


namespace range_of_a_l115_115720

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = a * Real.log x + 1/2 * x^2)
  (h_ineq : ∀ x1 x2 : ℝ, x1 ≠ x2 → 0 < x1 → 0 < x2 → (f x1 - f x2) / (x1 - x2) > 4) : a > 4 :=
sorry

end range_of_a_l115_115720


namespace margie_change_l115_115854

theorem margie_change (n_sold n_cost n_paid : ℕ) (h1 : n_sold = 3) (h2 : n_cost = 50) (h3 : n_paid = 500) : 
  n_paid - (n_sold * n_cost) = 350 := by
  sorry

end margie_change_l115_115854


namespace reduction_in_speed_l115_115859

-- Define the given conditions
def original_speed := 65 -- words per minute
def total_words := 810 -- words
def total_time := 18 -- minutes
def new_speed := total_words / total_time -- Derived new typing speed

-- Prove the reduction in typing speed
theorem reduction_in_speed : original_speed - new_speed = 20 := by
  have h1 : new_speed = 45 := by 
    calc 
      new_speed = total_words / total_time := rfl
      ... = 810 / 18 := rfl
      ... = 45 := by norm_num
  calc
    original_speed - new_speed = 65 - 45 := by congr; exact rfl; exact h1
    ... = 20 := by norm_num

end reduction_in_speed_l115_115859


namespace min_value_proof_l115_115849

noncomputable def min_value (x y : ℝ) : ℝ := x + 3 * y

theorem min_value_proof (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  min_value x y = 4 + 8 * real.sqrt 3 :=
  sorry

end min_value_proof_l115_115849


namespace problem_equivalent_statement_l115_115081

-- Define the operations provided in the problem
inductive Operation
| add
| sub
| mul
| div

open Operation

-- Represents the given equation with the specified operation
def applyOperation (op : Operation) (a b : ℕ) : ℕ :=
  match op with
  | add => a + b
  | sub => a - b
  | mul => a * b
  | div => a / b

theorem problem_equivalent_statement : 
  (∀ (op : Operation), applyOperation op 8 2 - 5 + 7 - (3^2 - 4) ≠ 6) → (¬ ∃ op : Operation, applyOperation op 8 2 = 9) := 
by
  sorry

end problem_equivalent_statement_l115_115081


namespace arithmetic_sequence_solution_l115_115260

theorem arithmetic_sequence_solution :
  ∃ a_1 a_2 d : ℤ, 
    (a_1 + a_2 + (a_1 + d) = 3) ∧
    (a_1 + 4 * d = 10) ∧
    (a_1 = -2 ∧ d = 3) :=
by {
  let a_1 := -2,
  let a_2 := 1,
  let d := 3,
  have h1 : a_1 + a_2 + (a_1 + d) = 3 := by linarith,
  have h2 : a_1 + 4 * d = 10 := by linarith,
  use [a_1, a_2, d],
  tauto,
}

end arithmetic_sequence_solution_l115_115260


namespace countMultiplesOfElevenEndingInFive_l115_115772

-- Definitions based on conditions
def isMultipleOfEleven (n : ℕ) : Prop := ∃ k : ℕ, n = 11 * k
def endsWithDigitFive (n : ℕ) : Prop := n % 10 = 5
def isValidNumber (n : ℕ) : Prop := isMultipleOfEleven n ∧ n < 1500 ∧ endsWithDigitFive n
def numberOfValidNumbers (count : ℕ) : Prop := ∃ k, (0 ≤ k ∧ k < 14 + 1) ∧ validNumbers.count = count

-- Theorem statement
theorem countMultiplesOfElevenEndingInFive : numberOfValidNumbers (14) :=
by sorry

end countMultiplesOfElevenEndingInFive_l115_115772


namespace probability_three_one_color_one_other_l115_115588

theorem probability_three_one_color_one_other (blue red : ℕ) (total_drawn : ℕ) (h_blue : blue = 10) (h_red : red = 8) (h_drawn : total_drawn = 4) :
  (960 + 560 : ℚ) / 3060 = 5 / 9 :=
by
  -- Definitions and numbers based on conditions and question
  have total_ways : ℚ := 3060
  have ways_three_blue_one_red : ℚ := 960
  have ways_one_blue_three_red : ℚ := 560
  have favorable_outcomes : ℚ := ways_three_blue_one_red + ways_one_blue_three_red
  have probability : ℚ := favorable_outcomes / total_ways
  calc
    probability = (960 + 560 : ℚ) / 3060 : by sorry
             ... = 5 / 9 : by sorry

note: 'sorry' is used as placeholder to indicate we skip the proof details here.


end probability_three_one_color_one_other_l115_115588


namespace monotonic_intervals_range_of_a_l115_115349

section Problem

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * x^3 - 3 * x^2

def f' (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

def g (x : ℝ) : ℝ := f a x + f' a x

-- Prove the monotonic intervals of f(x)
theorem monotonic_intervals (a : ℝ) :
  (if a = 0 then (∀ x, x < 0 → 0 < f' a x) ∧ (∀ x, 0 < x → f' a x < 0) else 
  if a > 0 then ((∀ x, x < 0 ∨ x > (2 / a) → 0 < f' a x) ∧ (∀ x, 0 < x ∧ x < (2 / a) → f' a x < 0)) else
  ((∀ x, (2 / a) < x ∧ x < 0 → 0 < f' a x) ∧ ((∀ x, x < (2 / a) ∨ 0 < x → f' a x < 0)))) :=
sorry

-- Prove the range of values for a
theorem range_of_a (h : ∃ x ∈ set.Icc 1 3, g a x ≤ 0) : a ≤ 9 / 4 :=
sorry

end Problem

end monotonic_intervals_range_of_a_l115_115349


namespace sum_of_coefficients_f_l115_115449

theorem sum_of_coefficients_f :
  ∀ (f g : ℝ[X]) (x : ℝ), ((↑(3^(1/2)/2) + (↑x/2) * Complex.I)^2008).re = f.eval 1 →
  ((↑(3^(1/2)/2) + (↑x/2) * Complex.I)^2008).im = g.eval 1 →
  f.eval 1 = -1/2 :=
by
  intros f g x h₁ h₂
  sorry

end sum_of_coefficients_f_l115_115449


namespace shelves_fit_l115_115826

-- Define the total space of the room for the library
def totalSpace : ℕ := 400

-- Define the space each bookshelf takes up
def spacePerBookshelf : ℕ := 80

-- Define the reserved space for desk and walking area
def reservedSpace : ℕ := 160

-- Define the space available for bookshelves
def availableSpace : ℕ := totalSpace - reservedSpace

-- Define the number of bookshelves that can fit in the available space
def numberOfBookshelves : ℕ := availableSpace / spacePerBookshelf

-- The theorem stating the number of bookshelves Jonas can fit in the room
theorem shelves_fit : numberOfBookshelves = 3 := by
  -- We can defer the proof as we only need the statement for now
  sorry

end shelves_fit_l115_115826


namespace overall_gain_or_loss_zero_l115_115897

variable (C_p S_p : ℝ) (C_pc S_pc : ℝ)

def pen_condition : Prop :=
  10 * C_p = 5 * S_p

def pencil_condition : Prop :=
  15 * C_pc = 9 * S_pc

theorem overall_gain_or_loss_zero 
  (hc_pen : pen_condition C_p S_p) 
  (hc_pencil : pencil_condition C_pc S_pc) :
  (TSP_total_pencil_pen C_p S_p C_pc S_pc hc_pen hc_pencil = TCP_total_pencil_pen C_p S_p C_pc S_pc) :=
sorry

def TSP_total_pencil_pen (C_p S_p : ℝ) (C_pc S_pc : ℝ) 
  (hc_pen : pen_condition C_p S_p) 
  (hc_pencil : pencil_condition C_pc S_pc) : ℝ :=
10 * S_p + 15 * S_pc

def TCP_total_pencil_pen (C_p S_p : ℝ) (C_pc S_pc : ℝ) : ℝ :=
10 * C_p + 15 * C_pc

end overall_gain_or_loss_zero_l115_115897


namespace divide_year_into_periods_l115_115610

theorem divide_year_into_periods (n m : ℕ) (h_pos_n : n > 0) (h_pos_m : m > 0) : 
  (n * m = 31536000) → finset.card ((finset.univ.filter (λ p: ℕ × ℕ, p.1 * p.2 = 31536000)).image (λ p, (p : ℕ × ℕ).swap)) = 336 := 
sorry

end divide_year_into_periods_l115_115610


namespace geometric_mean_of_1_and_4_l115_115903

theorem geometric_mean_of_1_and_4 :
  ∃ a : ℝ, a^2 = 4 ∧ (a = 2 ∨ a = -2) :=
by
  sorry

end geometric_mean_of_1_and_4_l115_115903


namespace probability_even_is_5_minus_pi_div_4_l115_115577

noncomputable def probability_even_quotient (x y : ℝ) (hx : x ∈ Set.Ioo 0 1) (hy : y ∈ Set.Ioo 0 1) : ℝ :=
  if ((Int.floor (x / y) % 2) = 0) then 1 else 0

theorem probability_even_is_5_minus_pi_div_4 :
  (∫ x in 0..1, ∫ y in 0..1, probability_even_quotient x y (Set.mem_Ioo.2 ⟨by linarith, by linarith⟩) (Set.mem_Ioo.2 ⟨by linarith, by linarith⟩)) = (5 - Real.pi) / 4 :=
by
  sorry

end probability_even_is_5_minus_pi_div_4_l115_115577


namespace conditional_probability_T1_conditional_probability_T2_approx_l115_115668
open ProbabilityTheory

-- Given conditions for the probability space
variable {Ω : Type*} [ProbabilitySpace Ω]

-- Definitions based on conditions
variables (p Q m : ℝ) (hp : 0 < p) (hq : p < 1)

-- T = 1 case
def Probability_T1_event (m : ℕ) : ℝ :=
  1 - p - (1 - 2 * p^(m-1)) / m

-- T = 2 case
def Probability_T2_approximation (m : ℕ) : ℝ :=
  1 - p - 2 / m + p^(m-2) * (1 - p^2 + 2 / m * (2 * p - 1 + p^2))

-- Theorem to show equality for T = 1 case
theorem conditional_probability_T1 (m : ℕ) : 
  (\mathbb{P}(\bar{A}_2 ∧ \bar{A}_3 ∧ \cdots ∧ \bar{A}_m ∣ A_1) = Probability_T1_event m)
  := sorry

-- Theorem to show approximation for T = 2 case
theorem conditional_probability_T2_approx (m : ℕ) :
  \mathbb{P}(\bar{A}_2 ∧ \bar{A}_3 ∧ \cdots ∧ \bar{A}_m ∣ A_1) = a + b / m + bigO (p^m)
  := sorry

end conditional_probability_T1_conditional_probability_T2_approx_l115_115668


namespace square_of_area_cyclic_quadrilateral_l115_115098

/-
We define the sides of the cyclic quadrilateral and the problem we aim to prove.
-/
variables (a b c d s : ℝ)
variables (AB BC CD DA : ℝ)
variables (cyclic : Prop)
variables (area K : ℝ)

-- The semiperimeter
def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2

-- Brahmagupta's formula for cyclic quadrilateral's area
def brahmagupta (s a b c d : ℝ) : ℝ := (s - a) * (s - b) * (s - c) * (s - d)

-- Define the sides and semiperimeter
def sides : Prop := (AB = 1) ∧ (BC = 2) ∧ (CD = 3) ∧ (DA = 4)

-- Cyclic quadrilateral condition (not used directly but here for completeness)
def cyclic_quadrilateral : Prop := cyclic

-- The main theorem stating the problem
theorem square_of_area_cyclic_quadrilateral 
  (AB BC CD DA : ℝ) (cyclic : Prop) (s : ℝ) (K : ℝ) 
  (h_sides : sides AB BC CD DA) (h_semi : s = semiperimeter AB BC CD DA) 
  (h_brahmagupta : K = real.sqrt (brahmagupta s AB BC CD DA)) :
  (K^2 = 24) :=
sorry

end square_of_area_cyclic_quadrilateral_l115_115098


namespace find_y_interval_l115_115693

open Real

theorem find_y_interval {y : ℝ}
  (hy_nonzero : y ≠ 0)
  (h_denominator_nonzero : 1 + 3 * y - 4 * y^2 ≠ 0) :
  (y^2 + 9 * y - 1 = 0) →
  (∀ y, y ∈ Set.Icc (-(9 + sqrt 85)/2) (-(9 - sqrt 85)/2) \ {y | y = 0 ∨ 1 + 3 * y - 4 * y^2 = 0} ↔
  (y * (3 - 3 * y))/(1 + 3 * y - 4 * y^2) ≤ 1) :=
by
  sorry

end find_y_interval_l115_115693


namespace line_through_B_parallel_to_AC_l115_115640

def point (x y : ℝ) := (x, y)
def line := set (ℝ × ℝ)

def A := point 5 2
def B := point -1 -4
def C := point -5 -3

def parallel {p1 p2 p3 p4 : ℝ × ℝ} : Prop :=
  ∃ k : ℝ, (p3.1 - p1.1) = k * (p4.1 - p2.1) ∧ (p3.2 - p1.2) = k * (p4.2 - p2.2)

def equation_of_line (B : ℝ × ℝ) (d : ℝ × ℝ) : line :=
  {P : ℝ × ℝ | (P.1 - B.1) * d.2 = (P.2 - B.2) * d.1}

theorem line_through_B_parallel_to_AC : 
  let d := (-10, -5) in 
  ∃ l : line, (equation_of_line B d l) ∧ (∀ p : ℝ × ℝ,
    p ∈ l ↔ (p.1 - 2 * p.2 - 7 = 0)) :=
  sorry

end line_through_B_parallel_to_AC_l115_115640


namespace cos_diff_to_product_l115_115293

open Real

theorem cos_diff_to_product (a b : ℝ) : 
  cos (a + b) - cos (a - b) = -2 * sin a * sin b := 
  sorry

end cos_diff_to_product_l115_115293


namespace root_value_l115_115783

theorem root_value (a : ℝ) (h: 3 * a^2 - 4 * a + 1 = 0) : 6 * a^2 - 8 * a + 5 = 3 := 
by 
  sorry

end root_value_l115_115783


namespace rank_from_left_l115_115247

theorem rank_from_left (total_students rank_from_right rank_from_left : ℕ) 
  (h_total : total_students = 31) (h_right : rank_from_right = 21) : 
  rank_from_left = 11 := by
  sorry

end rank_from_left_l115_115247


namespace smallest_possible_median_l115_115158

theorem smallest_possible_median (a b c d e f : ℕ) (h1 : a = 6) (h2 : b = 7) (h3 : c = 2) (h4 : d = 4) (h5 : e = 8) (h6 : f = 5)
    (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) :
    ∃ (l : List ℕ), l = List.sort ([a, b, c, d, e, f, x, y, z]) ∧ l.nth 4 = some 4 :=
by
  sorry

end smallest_possible_median_l115_115158


namespace hair_stylist_cut_off_amount_l115_115258

theorem hair_stylist_cut_off_amount
    (initial_length : ℝ)
    (growth_week1 : ℝ)
    (growth_week2 : ℝ)
    (growth_week3 : ℝ)
    (growth_week4 : ℝ)
    (after_haircut_length : ℝ) :
    initial_length = 11 →
    growth_week1 = 0.5 →
    growth_week2 = 0.75 →
    growth_week3 = 1 →
    growth_week4 = 0.25 →
    after_haircut_length = 7 →
    (initial_length + growth_week1 + growth_week2 + growth_week3 + growth_week4 - after_haircut_length = 6.5) :=
begin
  intros,
  sorry
end

end hair_stylist_cut_off_amount_l115_115258


namespace base7_to_base10_l115_115676

theorem base7_to_base10 : 6 * 7^3 + 4 * 7^2 + 2 * 7^1 + 3 * 7^0 = 2271 := by
  sorry

end base7_to_base10_l115_115676


namespace percentage_gain_at_350_l115_115969

def cost_price : ℝ := 40
def selling_price_350 : ℝ := 350
def selling_price_348 : ℝ := 348

def gain (selling_price : ℝ) : ℝ := selling_price - cost_price

def percentage_gain (selling_price : ℝ) : ℝ := (gain selling_price / cost_price) * 100

theorem percentage_gain_at_350 : percentage_gain selling_price_350 = 775 :=
by
  -- proof omitted
  sorry

end percentage_gain_at_350_l115_115969


namespace incorrect_propositions_count_l115_115021

open Classical

variables (p q : Prop)
variables (a b x : ℝ)
variables (m : ℕ)
def proposition1 : Prop := ¬(p ∧ q) → ¬p ∧ ¬q
def proposition2 : Prop := ¬(a > b → 2^a > 2^b - 1) = (a ≤ b → 2^a ≤ 2^(b:ℝ) - 1)
def proposition3 : Prop := ¬(∀ x : ℝ, x^2 + 1 ≥ 1) = ∃ x : ℝ, x^2 + 1 < 1
def proposition4 : Prop := (am^2 < bm^2) → (a < b)

theorem incorrect_propositions_count : 
  (¬ proposition1) ∧ (proposition2) ∧ (¬ proposition3) ∧ (¬ proposition4) → 
  3 = 3 := 
by 
  intros,
  sorry

end incorrect_propositions_count_l115_115021


namespace capsules_per_bottle_l115_115503

-- Translating conditions into Lean definitions
def days := 180
def daily_serving_size := 2
def total_bottles := 6
def total_capsules_required := days * daily_serving_size

-- The statement to prove
theorem capsules_per_bottle : total_capsules_required / total_bottles = 60 :=
by
  sorry

end capsules_per_bottle_l115_115503


namespace compound_formed_l115_115696

/-- The compound formed in the reaction between BaO and 3 moles of H2O,
given that 459 grams of BaO is required, is Ba(OH)2. -/
theorem compound_formed (mBaO : ℕ) (mH2O : ℕ) (massBaO : ℝ) :
  mBaO = 3 → mH2O = 3 → massBaO = 459 →
  (∀ (BaO H2O BaOH2 : Type), reaction BaO H2O BaOH2 → BaOH2) :=
by
  sorry

end compound_formed_l115_115696


namespace max_area_rect_l115_115909

noncomputable def maximize_area (l w : ℕ) : ℕ :=
  l * w

theorem max_area_rect (l w: ℕ) (hl_even : l % 2 = 0) (h_perim : 2*l + 2*w = 40) :
  maximize_area l w = 100 :=
by
  sorry 

end max_area_rect_l115_115909


namespace max_sum_of_roots_l115_115760

theorem max_sum_of_roots (a b : ℝ) (h_a : a ≠ 0) (m : ℝ) :
  (∀ x : ℝ, (2 * x ^ 2 - 5 * x + m = 0) → 25 - 8 * m ≥ 0) →
  (∃ s, s = -5 / 2) → m = 25 / 8 :=
by
  sorry

end max_sum_of_roots_l115_115760


namespace square_remainder_is_square_l115_115469

theorem square_remainder_is_square (a : ℤ) : ∃ b : ℕ, (a^2 % 16 = b) ∧ (∃ c : ℕ, b = c^2) :=
by
  sorry

end square_remainder_is_square_l115_115469


namespace sum_of_x_coordinates_l115_115133

theorem sum_of_x_coordinates (x y : ℤ) (h1 : y ≡ 7 * x + 3 [ZMOD 20]) (h2 : y ≡ 13 * x + 17 [ZMOD 20]) :
  ∑ (x : ℤ) in { x | 0 ≤ x ∧ x < 20 ∧ (∃ y, y ≡ 7 * x + 3 [ZMOD 20] ∧ y ≡ 13 * x + 17 [ZMOD 20]) }, x = 12 :=
by
  sorry

end sum_of_x_coordinates_l115_115133


namespace room_breadth_l115_115572

theorem room_breadth :
  ∀ (length breadth carpet_width cost_per_meter total_cost : ℝ),
  length = 15 →
  carpet_width = 75 / 100 →
  cost_per_meter = 30 / 100 →
  total_cost = 36 →
  total_cost = cost_per_meter * (total_cost / cost_per_meter) →
  length * breadth = (total_cost / cost_per_meter) * carpet_width →
  breadth = 6 :=
by
  intros length breadth carpet_width cost_per_meter total_cost
  intros h_length h_carpet_width h_cost_per_meter h_total_cost h_total_cost_eq h_area_eq
  sorry

end room_breadth_l115_115572


namespace volume_of_large_cube_l115_115981

noncomputable def num_dices : ℕ := 8
noncomputable def surface_area_of_one_die : ℕ := 96 -- square centimeters
noncomputable def num_faces : ℕ := 6

theorem volume_of_large_cube (h₁ : num_dices = 8) (h₂ : surface_area_of_one_die = 96) (h₃ : num_faces = 6) : 
    let face_area_one_die := surface_area_of_one_die / num_faces,
        side_length_one_die := Int.sqrt face_area_one_die,
        side_length_large_cube := 2 * side_length_one_die,
        volume_large_cube := side_length_large_cube * side_length_large_cube * side_length_large_cube 
    in volume_large_cube = 512 := by
  intros
  sorry

end volume_of_large_cube_l115_115981


namespace sum_of_solutions_f_x_eq_2_l115_115276

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2 * x + 4 else -x / 3 + 3

theorem sum_of_solutions_f_x_eq_2 :
  (∑ x in {x | f x = 2}.to_finset, x) = 2 := by
  sorry

end sum_of_solutions_f_x_eq_2_l115_115276


namespace triangle_least_perimeter_l115_115182

def least_possible_perimeter (a b x : ℕ) (h1 : a = 24) (h2 : b = 37) (h3 : x > 13) (h4 : x < 61) : Prop :=
  a + b + x = 75

theorem triangle_least_perimeter : ∃ x : ℕ, least_possible_perimeter 24 37 x
  :=
sorry

end triangle_least_perimeter_l115_115182


namespace cassette_costs_l115_115584

theorem cassette_costs (A V : ℝ) (hA : 5 * A + 4 * V = 1350) (hV : V = 300) : 7 * A + 3 * V = 1110 :=
begin
  -- Proof omitted
  sorry
end

end cassette_costs_l115_115584


namespace larger_number_l115_115934

theorem larger_number (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : max x y = 16 := by
  sorry

end larger_number_l115_115934


namespace most_likely_number_of_red_balls_l115_115543

-- Define the conditions
def total_balls : ℕ := 20
def red_ball_frequency : ℝ := 0.8

-- Define the statement we want to prove
theorem most_likely_number_of_red_balls : red_ball_frequency * total_balls = 16 :=
by sorry

end most_likely_number_of_red_balls_l115_115543


namespace segment_length_RQ_l115_115075

noncomputable def circle_parametric (ϕ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos ϕ, Real.sin ϕ)

noncomputable def polar_conversion (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

noncomputable def circle_polar (ϕ θ : ℝ) : ℝ :=
  let ρ := 2 * Real.cos θ
  ρ

noncomputable def line_polar (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ + Real.pi / 3) = 3 * Real.sqrt 3

theorem segment_length_RQ :
  let θ := Real.pi / 3
  let R := 1
  let Q := 3
  | R - Q | = 2 := 
by {
  intro,
  sorry
}

end segment_length_RQ_l115_115075


namespace problem1_problem2_l115_115002

variables {A B C a b c : ℝ}

-- Define angle-side relationships
def sides_opposite_angles (a b c A B C : ℝ) : Prop :=
  ∀ (a b c A B C : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ 
  A + B + C = π ∧ sin B + sin C / sin A = 2 - cos B - cos C / cos A

-- Define function properties
def f_property (f : ℝ → ℝ) (ω : ℝ) : Prop :=
  (∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ π / 3 → f x ≤ f y) ∧ 
  (∀ x y, π / 3 ≤ x ∧ x ≤ y ∧ y ≤ 2 * π / 3 → f x ≥ f y)

-- (1) Prove b + c = 2a
theorem problem1 (h1 : sides_opposite_angles a b c A B C) : b + c = 2 * a :=
sorry

-- (2) Prove that triangle ABC is equilateral
theorem problem2 (h1 : sides_opposite_angles a b c A B C)
  (h2 : ∀ (f : ℝ → ℝ) (ω : ℝ), f = (λ x, sin (ω * x)) ∧ ω > 0 ∧ f_property f ω ∧ f (π / 9) = cos A) 
  : A = π / 3 ∧ b = c :=
sorry

end problem1_problem2_l115_115002


namespace function_graph_match_l115_115687

-- Definitions from the problem
def f1 (x : ℝ) : ℝ := x^3 + x^2 + x + 1
def f2 (x : ℝ) : ℝ := x^3 + 2 * x^2 + x + 1
def f3 (x : ℝ) : ℝ := x^3 + x^2 + 3 * x + 1
def f4 (x : ℝ) : ℝ := x^3 + x^2 + x - 1

-- Graph points inferred from image and explanation
def graph_points := [(0, 1), (1, 4), (-1, 0)]

-- Proof statement
theorem function_graph_match : 
  ∀ (x y : ℝ), (x, y) ∈ graph_points → y = f1 x :=
by
  sorry

end function_graph_match_l115_115687


namespace polygon_sides_eq_six_l115_115786

theorem polygon_sides_eq_six (n : ℕ) (h_interior : ∑ i of fin n, (angle_interior i) = (n - 2) * 180)
  (h_exterior : ∑ i of fin n, (angle_exterior i) = 360)
  (h_condition : (n - 2) * 180 = 2 * 360) :
  n = 6 :=
by
  sorry

end polygon_sides_eq_six_l115_115786


namespace minimum_value_of_function_l115_115301

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+1) + 2

theorem minimum_value_of_function :
  ∃ x : ℝ, f x = 0 :=
begin
  use 1,
  unfold f,
  calc
    4^1 - 2^(1 + 1) + 2 = 4 - 4 + 2 : by norm_num
    ... = 0 : by norm_num,
end

end minimum_value_of_function_l115_115301


namespace jed_speed_is_84_l115_115792

noncomputable def speed_limit : ℕ := 50
noncomputable def total_fine : ℕ := 826
noncomputable def red_light_fine : ℕ := 75
noncomputable def running_red_lights : ℕ := 2
noncomputable def cellphone_fine : ℕ := 120
noncomputable def speeding_fine_per_mph : ℕ := 16

def non_speeding_fines : ℕ := running_red_lights * red_light_fine + cellphone_fine
def speeding_fine : ℕ := total_fine - non_speeding_fines
def mph_over_limit : ℕ := speeding_fine / speeding_fine_per_mph
def jed_speed : ℕ := speed_limit + mph_over_limit

theorem jed_speed_is_84 : jed_speed = 84 := by
  sorry

end jed_speed_is_84_l115_115792


namespace greatest_power_of_7_factorial_l115_115905

theorem greatest_power_of_7_factorial (n : ℕ) (hn : n = 50) : 
  ∑ (i : ℕ) in finset.range (nat.log 7 n + 1), n / 7^i = 8 :=
by sorry

end greatest_power_of_7_factorial_l115_115905


namespace no_limits_outside_segment_l115_115221

open Set Filter

-- Defining the segment [a, b] as an interval
def is_segment {α : Type*} [LinearOrder α] (a b : α) (x : α) : Prop :=
  a ≤ x ∧ x ≤ b

-- Defining the attractor property for a sequence in the interval [a, b]
def is_attractor {α : Type*} [LinearOrder α] (a b : α) (x : ℕ → α) : Prop :=
  ∀ N : ℕ, ∃ n ≥ N, is_segment a b (x n)

theorem no_limits_outside_segment {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]
  (a b : α) (x : ℕ → α) :
  is_attractor a b x → ∀ L, (L < a ∨ L > b) → ¬ filter.tendsto x filter.at_top (𝓝 L) :=
begin
  intros attractor L L_outside_segment tendsto_L,
  by_contradiction,
  sorry
end

end no_limits_outside_segment_l115_115221


namespace Carter_card_number_l115_115124

-- Definitions based on conditions
def Marcus_cards : ℕ := 210
def difference : ℕ := 58

-- Definition to infer the number of Carter's baseball cards
def Carter_cards : ℕ := Marcus_cards - difference

-- Theorem statement asserting the number of baseball cards Carter has
theorem Carter_card_number : Carter_cards = 152 := by
  sorry

end Carter_card_number_l115_115124


namespace det_M_pow_five_l115_115378

variable (M : Matrix ℕ ℕ ℝ)

theorem det_M_pow_five (h : det M = 3) : det (M^5) = 243 :=
by sorry

end det_M_pow_five_l115_115378


namespace ball_total_distance_l115_115629

noncomputable def total_distance : ℝ :=
  let h₀ := 10 in
  let r := 4 / 5 in
  lim (λ n, 2 * h₀ * (1 - r ^ n) / (1 - r)) - h₀

theorem ball_total_distance :
  total_distance = 90 :=
by
  sorry

end ball_total_distance_l115_115629


namespace div30k_929260_l115_115781

theorem div30k_929260 (k : ℕ) (h : 30^k ∣ 929260) : 3^k - k^3 = 1 := by
  sorry

end div30k_929260_l115_115781


namespace number_of_bookshelves_l115_115824

def total_space : ℕ := 400
def reserved_space : ℕ := 160
def shelf_space : ℕ := 80

theorem number_of_bookshelves : (total_space - reserved_space) / shelf_space = 3 := by
  sorry

end number_of_bookshelves_l115_115824


namespace pos_numbers_equal_l115_115056

theorem pos_numbers_equal (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_eq : a^3 + b^3 + c^3 - 3 * a * b * c = 0) : a = b ∧ b = c :=
by
  sorry

end pos_numbers_equal_l115_115056


namespace sum_of_fractions_removal_l115_115688

theorem sum_of_fractions_removal :
  (1 / 3 + 1 / 6 + 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18 + 1 / 21) 
  - (1 / 12 + 1 / 21) = 3 / 4 := 
 by sorry

end sum_of_fractions_removal_l115_115688


namespace units_digit_calc_l115_115282

theorem units_digit_calc :
  (2^1201 * 4^1302 * 6^1403) % 10 = 2 := by
suffices (2^1201 % 10) = 2 from
suffices (4^1302 % 10) = 6 from
suffices (6^1403 % 10) = 6 from
  by
    have h1 := (2^1201 % 10 = 2)
    have h2 := (4^1302 % 10 = 6)
    have h3 := (6^1403 % 10 = 6)
    calc
      (2^1201 * 4^1302 * 6^1403) % 10
          = ((2^1201 % 10) * (4^1302 % 10) * (6^1403 % 10)) % 10 : by sorry
      ... = (2 * 6 * 6) % 10 : by rw [h1, h2, h3]
      ... = 2 : by sorry,
sorry, sorry, sorry

end units_digit_calc_l115_115282


namespace unique_very_set_on_line_l115_115939

def very_set (S : Finset (ℝ × ℝ)) : Prop :=
  ∀ X ∈ S, ∃ (r : ℝ), 
  ∀ Y ∈ S, Y ≠ X → ∃ Z ∈ S, Z ≠ X ∧ r * r = dist X Y * dist X Z

theorem unique_very_set_on_line (n : ℕ) (A B : ℝ × ℝ) (S1 S2 : Finset (ℝ × ℝ))
  (h : 2 ≤ n) (hA1 : A ∈ S1) (hB1 : B ∈ S1) (hA2 : A ∈ S2) (hB2 : B ∈ S2)
  (hS1 : S1.card = n) (hS2 : S2.card = n) (hV1 : very_set S1) (hV2 : very_set S2) :
  S1 = S2 := 
sorry

end unique_very_set_on_line_l115_115939


namespace possible_values_of_C_l115_115486

theorem possible_values_of_C {a b C : ℤ} :
  (C = a * (a - 5) ∧ C = b * (b - 8)) ↔ (C = 0 ∨ C = 84) :=
sorry

end possible_values_of_C_l115_115486


namespace avg_speed_l115_115915

variable (d1 d2 t1 t2 : ℕ)

-- Conditions
def distance_first_hour : ℕ := 80
def distance_second_hour : ℕ := 40
def time_first_hour : ℕ := 1
def time_second_hour : ℕ := 1

-- Ensure that total distance and total time are defined correctly from conditions
def total_distance : ℕ := distance_first_hour + distance_second_hour
def total_time : ℕ := time_first_hour + time_second_hour

-- Theorem to prove the average speed
theorem avg_speed : total_distance / total_time = 60 := by
  sorry

end avg_speed_l115_115915


namespace probability_median_is_55_l115_115976

def weights_of_measured := [60, 55, 60, 55, 65, 50, 50]

def unmeasured_weight_range := set.Icc (50 : ℝ) (60 : ℝ)

def is_median_55 (unmeasured_weight : ℝ) : Prop :=
  let all_weights := unmeasured_weight :: weights_of_measured in
  let sorted_weights := all_weights |>.qsort (· < ·) in
  sorted_weights.get 3 = 55 

theorem probability_median_is_55 :
  ∃ (μ : MeasureTheory.Measure (set.Icc 50 60)),
  μ.unmeasured_weight_range = volume.unmeasured_weight_range ∧
  μ {weight | is_median_55 weight } = (1/2 : ℝ) :=
sorry

end probability_median_is_55_l115_115976


namespace total_time_correct_l115_115419

-- Conditions
def minutes_per_story : Nat := 7
def weeks : Nat := 20

-- Total time calculation
def total_minutes : Nat := minutes_per_story * weeks

-- Conversion to hours and minutes
def total_hours : Nat := total_minutes / 60
def remaining_minutes : Nat := total_minutes % 60

-- The proof problem
theorem total_time_correct :
  total_minutes = 140 ∧ total_hours = 2 ∧ remaining_minutes = 20 := by
  sorry

end total_time_correct_l115_115419


namespace find_first_term_and_ratio_l115_115340

variable (b1 q : ℝ)

-- Conditions
def infinite_geometric_series (q : ℝ) : Prop := |q| < 1

def sum_odd_even_difference (b1 q : ℝ) : Prop := 
  b1 / (1 - q^2) = 2 + (b1 * q) / (1 - q^2)

def sum_square_odd_even_difference (b1 q : ℝ) : Prop :=
  b1^2 / (1 - q^4) - (b1^2 * q^2) / (1 - q^4) = 36 / 5

-- Proof problem
theorem find_first_term_and_ratio (b1 q : ℝ) 
  (h1 : infinite_geometric_series q) 
  (h2 : sum_odd_even_difference b1 q)
  (h3 : sum_square_odd_even_difference b1 q) : 
  b1 = 3 ∧ q = 1 / 2 := by
  sorry

end find_first_term_and_ratio_l115_115340


namespace binomial_constant_term_l115_115352

theorem binomial_constant_term (n : ℕ) (x : ℝ) (h1 : (∑ i in finset.range (n + 1), nat.choose n i) = 64) :
  let term := ∑ r in finset.range (n + 1), nat.choose n r * ((-2)^r) * (x^(n - (3 * r))) in
  term = 60 :=
by
  sorry

end binomial_constant_term_l115_115352


namespace perfect_square_polynomial_l115_115779

theorem perfect_square_polynomial (m : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x : ℝ, x^2 + 2*(m-3)*x + 25 = f x * f x) ↔ (m = 8 ∨ m = -2) :=
by
  sorry

end perfect_square_polynomial_l115_115779


namespace coefficient_of_y_l115_115712

theorem coefficient_of_y (x y a : ℝ) (h1 : 7 * x + y = 19) (h2 : x + a * y = 1) (h3 : 2 * x + y = 5) : a = 3 :=
sorry

end coefficient_of_y_l115_115712


namespace probability_sum_odd_l115_115314

def is_odd (n : ℤ) : Prop := n % 2 = 1
def is_even (n : ℤ) : Prop := n % 2 = 0

theorem probability_sum_odd :
  let s := {1, 2, 3, 4, 5} in
  let odd_count := (s.filter is_odd).card in
  let even_count := (s.filter is_even).card in
  let total_ways := s.card.choose 2 in
  let odd_even_ways := odd_count * even_count in
  (odd_even_ways : ℚ) / total_ways = 3 / 5 :=
by
  sorry

end probability_sum_odd_l115_115314


namespace mean_median_mode_sum_correct_l115_115945

noncomputable theory

def numbers : List ℕ := [2, 3, 0, 3, 1, 4, 0, 3, 5]

def mean (l : List ℕ) : ℚ :=
  l.sum / l.length

def median (l : List ℕ) : ℕ :=
  l.sort.get! (l.length / 2)

def mode (l : List ℕ) : ℕ :=
  l.groupBy id
   |>.maxBy (List.length ∘ Prod.snd)
   |>.fst

def mean_median_mode_sum (l : List ℕ) : ℚ :=
  mean l + median l + mode l

theorem mean_median_mode_sum_correct :
  mean_median_mode_sum numbers = 22 / 3 :=
by
  sorry

end mean_median_mode_sum_correct_l115_115945


namespace triangle_ABF_is_isosceles_l115_115799

   noncomputable def is_parallel {V : Type*} [AddCommGroup V] [Module ℝ V] (u v : V) : Prop :=
   ∃ a : ℝ, a ≠ 0 ∧ u = a • v

   noncomputable def is_midpoint {V : Type*} [AddCommGroup V] [Module ℝ V] (m a b : V) : Prop :=
   m = (1/2 : ℝ) • (a + b)

   noncomputable def is_foot_of_perpendicular {V : Type*} [AddCommGroup V] [Module ℝ V] (f b c : V) : Prop :=
   ∃ k : ℝ, f = b + k • (b - c) ∧ ∀ t : ℝ, t • (f - b) = t • (f - c)

   noncomputable def is_isosceles {V : Type*} [AddCommGroup V] [Module ℝ V] (a b c : V) : Prop :=
   dist a b = dist a c ∨ dist b c = dist b a ∨ dist c a = dist c b

   variables {V : Type*} [AddCommGroup V] [Module ℝ V] (A B C D E F : V)

   theorem triangle_ABF_is_isosceles
     (h1 : is_parallelogram A B C D)
     (h2 : is_midpoint E A D)
     (h3 : is_foot_of_perpendicular F B (line_through C E))
     : is_isosceles A B F := 
   sorry
   
end triangle_ABF_is_isosceles_l115_115799


namespace new_number_when_appending_2_l115_115386

theorem new_number_when_appending_2 (h t u : ℕ) (h_range : h < 10) (t_range : t < 10) (u_range : u < 10) :
  let original_number := 100 * h + 10 * t + u in
  let new_number := original_number * 10 + 2 in
  new_number = 1000 * h + 100 * t + 10 * u + 2 :=
by
  intros
  dsimp [original_number, new_number]
  sorry

end new_number_when_appending_2_l115_115386


namespace find_plane_equation_l115_115299

def point := ℝ × ℝ × ℝ

def plane_equation (A B C D : ℝ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

def points := (0, 3, -1) :: (4, 7, 1) :: (2, 5, 0) :: []

def correct_plane_equation : Prop :=
  ∃ A B C D : ℝ, plane_equation A B C D = fun x y z => A * x + B * y + C * z + D = 0 ∧ 
  (A, B, C, D) = (0, 1, -2, -5) ∧ ∀ x y z, (x, y, z) ∈ points → plane_equation A B C D x y z

theorem find_plane_equation : correct_plane_equation :=
sorry

end find_plane_equation_l115_115299


namespace UPOMB_position_l115_115937

-- Define the set of letters B, M, O, P, and U
def letters : List Char := ['B', 'M', 'O', 'P', 'U']

-- Define the word UPOMB
def word := "UPOMB"

-- Define a function that calculates the factorial of a number
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to calculate the position of a word in the alphabetical permutations of a list of characters
def word_position (w : String) (chars : List Char) : Nat :=
  let rec aux (w : List Char) (remaining : List Char) : Nat :=
    match w with
    | [] => 1
    | c :: cs =>
      let before_count := remaining.filter (· < c) |>.length
      let rest_count := factorial (remaining.length - 1)
      before_count * rest_count + aux cs (remaining.erase c)
  aux w.data chars

-- The desired theorem statement
theorem UPOMB_position : word_position word letters = 119 := by
  sorry

end UPOMB_position_l115_115937


namespace bernardo_larger_probability_l115_115658

-- Definitions: Bernardo's and Silvia's selections
def bernardo_set := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def silvia_set := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Definitions: The probability of selecting numbers
noncomputable def total_probability : ℚ := 31 / 48

-- The goal: To prove that the probability Bernardo's number is larger is 31/48
theorem bernardo_larger_probability :
  let B := finset.powersetLen 3 bernardo_set
  let S := finset.powersetLen 3 silvia_set in
  ∑ s in B, ∑ t in S, if (true) then 1 else 0 / (B.card * S.card) = total_probability :=
by 
  sorry

end bernardo_larger_probability_l115_115658


namespace log3_ineq_l115_115272

theorem log3_ineq (x1 x2 : ℝ) (h1 : x1 = 0.4) (h2 : x2 = 0.2) (h3 : 0 < x1) (h4 : 0 < x2) 
  (log_mono : ∀ {a b : ℝ}, 0 < a → 0 < b → a < b → log 3 a < log 3 b) : 
  (log 3 x1) > (log 3 x2) :=
by
  rw [h1, h2]
  have h5 : 0 < 0.4 := by norm_num
  have h6 : 0 < 0.2 := by norm_num
  have h7 : 0.2 < 0.4 := by norm_num
  exact log_mono h6 h5 h7

end log3_ineq_l115_115272


namespace geometric_sequence_extreme_points_l115_115085

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + 4 * x^2 + 9 * x - 1

def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (∃ x', f' x' = 0) ∧ (f' x = 0)

theorem geometric_sequence_extreme_points (a : ℕ → ℝ) :
  is_extreme_point f (a 3) ∧ is_extreme_point f (a 7) ∧ a 3 ≠ a 7 →
  a 5 = -3 :=
by
  sorry

end geometric_sequence_extreme_points_l115_115085


namespace sin_B_of_right_triangle_l115_115415

theorem sin_B_of_right_triangle
  (A B C : Type)
  [right_triangle A B C (angle_eq := 90°) (AB := 5) (AC := 12)] :
  sin_of_angle B = 5 / 13 :=
sorry

end sin_B_of_right_triangle_l115_115415


namespace billiard_triangle_angles_l115_115134

theorem billiard_triangle_angles (A B C : Type) [real_inner_product_space ℝ A] (α β : ℝ) (h_right_angle : (∠ A C B) = 90) 
    (h_bisector_return : ∀ (P : A), ∠ P A B = (1 / 2) * ∠ B A C ∧ reflects_off_third_side_returns_to_start A B C) : 
    ∠ B A C = 36 ∧ ∠ A B C = 54 := 
by
  sorry

end billiard_triangle_angles_l115_115134


namespace cos_angle_between_vectors_l115_115353

variables {α β : ℝ}
variables {e1 e2 a b : ℝ}

-- Conditions from the problem
def unit_vectors_angle (cos_alpha : ℝ) : Prop := cos_alpha = 1 / 3
def vector_a (v1 v2: ℝ) : Prop := a = 3 * v1 - 2 * v2
def vector_b (v1 v2: ℝ) : Prop := b = 3 * v1 - v2

-- The statement we want to prove
theorem cos_angle_between_vectors : 
  ∀ {α β : ℝ} {e1 e2 a b : ℝ}, 
  unit_vectors_angle (cos α) → 
  vector_a e1 e2 → 
  vector_b e1 e2 → 
  (cos β = (2 * sqrt 2) / 3) :=
by
  assume α β e1 e2 a b h1 h2 h3,
  sorry

end cos_angle_between_vectors_l115_115353


namespace fill_time_difference_correct_l115_115620

-- Define the time to fill one barrel in normal conditions
def normal_fill_time_per_barrel : ℕ := 3

-- Define the time to fill one barrel with a leak
def leak_fill_time_per_barrel : ℕ := 5

-- Define the number of barrels to fill
def barrels_to_fill : ℕ := 12

-- Define the time to fill 12 barrels in normal conditions
def normal_fill_time : ℕ := normal_fill_time_per_barrel * barrels_to_fill

-- Define the time to fill 12 barrels with a leak
def leak_fill_time : ℕ := leak_fill_time_per_barrel * barrels_to_fill

-- Define the time difference
def time_difference : ℕ := leak_fill_time - normal_fill_time

theorem fill_time_difference_correct : time_difference = 24 := by
  sorry

end fill_time_difference_correct_l115_115620


namespace y_days_worked_l115_115212

theorem y_days_worked 
  ( W : ℝ )
  ( x_rate : ℝ := W / 21 )
  ( y_rate : ℝ := W / 15 )
  ( d : ℝ )
  ( y_work_done : ℝ := d * y_rate )
  ( x_work_done_after_y_leaves : ℝ := 14 * x_rate )
  ( total_work_done : y_work_done + x_work_done_after_y_leaves = W ) :
  d = 5 := 
sorry

end y_days_worked_l115_115212


namespace third_quadrant_to_first_third_fourth_l115_115377

theorem third_quadrant_to_first_third_fourth (k : ℤ) (α : ℝ) 
  (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2) : 
  ∃ n : ℤ, (2 * k / 3 % 2) * Real.pi + Real.pi / 3 < α / 3 ∧ α / 3 < (2 * k / 3 % 2) * Real.pi + Real.pi / 2 ∨
            (2 * (3 * n + 1) % 2) * Real.pi + Real.pi < α / 3 ∧ α / 3 < (2 * (3 * n + 1) % 2) * Real.pi + 7 * Real.pi / 6 ∨
            (2 * (3 * n + 2) % 2) * Real.pi + 5 * Real.pi / 3 < α / 3 ∧ α / 3 < (2 * (3 * n + 2) % 2) * Real.pi + 11 * Real.pi / 6 :=
sorry

end third_quadrant_to_first_third_fourth_l115_115377


namespace time_spent_making_popcorn_l115_115829

-- Conditions
namespace PopcornProblem

theorem time_spent_making_popcorn (P : ℕ) : 
  let first_movie := 90,
      second_movie := 90 + 30,
      total_movie_time := 90 + 120,
      total_time := 240,
      time_making_fries := 2 * P
  in first_movie + second_movie + time_making_fries + P = total_time → P = 10 := 
by
  intros
  let first_movie := 90
  let second_movie := 90 + 30
  let total_movie_time := first_movie + second_movie
  let total_time := 240
  let time_making_fries := 2 * P
  have : total_movie_time + time_making_fries + P = total_time, from ‹first_movie + second_movie + time_making_fries + P = total_time›
  rw [←add_assoc, ←add_assoc] at this
  have h1 : 210 + time_making_fries + P = total_time :=
    by rw [←add_assoc, add_comm 120 90]
  have h2 : 210 + (2 * P) + P = total_time := by assumption
  have h3 : 210 + 3 * P = total_time := by rw [mul_comm 2 P, add_assoc]
  have h4 : 210 + 3 * P = 240 := h3
  have h5 : 3 * P = 30 := by linarith
  have h6 : P = 10 := by linarith
  exact h6

end PopcornProblem

end time_spent_making_popcorn_l115_115829


namespace unique_fraction_satisfying_condition_l115_115675

theorem unique_fraction_satisfying_condition : ∃! (x y : ℕ), Nat.gcd x y = 1 ∧ y ≠ 0 ∧ (x + 1) * 5 * y = (y + 1) * 6 * x :=
by
  sorry

end unique_fraction_satisfying_condition_l115_115675


namespace theatre_fraction_l115_115401

noncomputable def fraction_theatre_took_elective_last_year (T P Th M : ℕ) : Prop :=
  (P = 1 / 2 * T) ∧
  (Th + M = T - P) ∧
  (1 / 3 * P + M = 2 / 3 * T) ∧
  (Th = 1 / 6 * T)

theorem theatre_fraction (T P Th M : ℕ) :
  fraction_theatre_took_elective_last_year T P Th M →
  Th / T = 1 / 6 :=
by
  intro h
  cases h
  sorry

end theatre_fraction_l115_115401


namespace floor_length_l115_115471

theorem floor_length (tile_length tile_width : ℕ) (floor_width max_tiles : ℕ)
  (h_tile : tile_length = 25 ∧ tile_width = 16)
  (h_floor_width : floor_width = 120)
  (h_max_tiles : max_tiles = 54) :
  ∃ floor_length : ℕ, 
    (∃ num_cols num_rows : ℕ, 
      num_cols * tile_width = floor_width ∧ 
      num_cols * num_rows = max_tiles ∧ 
      num_rows * tile_length = floor_length) ∧
    floor_length = 175 := 
by
  sorry

end floor_length_l115_115471


namespace smallest_class_size_l115_115407

theorem smallest_class_size (n : ℕ) (h : 5 * n + 2 > 40) : 5 * n + 2 ≥ 42 :=
by
  sorry

end smallest_class_size_l115_115407


namespace ben_paints_150_sq_feet_l115_115255

theorem ben_paints_150_sq_feet
    (total_area : ℕ)
    (allen_ratio ben_ratio charlie_ratio : ℕ)
    (ratio_sum_eq : allen_ratio + ben_ratio + charlie_ratio = 10)
    (work_allocation : allen_ratio = 3 ∧ ben_ratio = 5 ∧ charlie_ratio = 2)
    (total_area_eq : total_area = 300) :
    let part_area := total_area / 10,
        ben_area := part_area * ben_ratio
    in
    ben_area = 150 := 
by 
  sorry

end ben_paints_150_sq_feet_l115_115255


namespace find_integers_correct_l115_115695

noncomputable def find_integers (a b c d : ℤ) : Prop :=
  a + b + c = 6 ∧ a + b + d = 7 ∧ a + c + d = 8 ∧ b + c + d = 9

theorem find_integers_correct (a b c d : ℤ) (h : find_integers a b c d) : a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 :=
by
  sorry

end find_integers_correct_l115_115695


namespace a4_is_minus_5_l115_115331

noncomputable def sequence (n : ℕ) : ℤ :=
if h : n = 0 then 0 else nat.rec_on n 1 (λ n' an', an' - (n' : ℤ))

theorem a4_is_minus_5 : sequence 4 = -5 := by
sorry

end a4_is_minus_5_l115_115331


namespace problem_1_omega_theta_problem_2_max_omega_l115_115361

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) + 6 * Real.cos (ω * x / 2) ^ 2 - 3

theorem problem_1_omega_theta (ω θ : ℝ) (h1 : ω > 0) (h2 : 0 < θ) (h3 : θ < (Real.pi / 2)):
  (∀ x : ℝ, f (x + θ) ω = f (-x + θ) ω) → (Real.Tendsto (f (x + Real.pi) ω) x atTop (𝓝 (f x ω))) → ω = 2 ∧ θ = Real.pi / 12 :=
sorry

theorem problem_2_max_omega (ω : ℝ) (h : ω > 0) :
  (∀ x : ℝ, 0 < x ∧ x < (Real.pi / 3) → f (3 * x) ω < f (3 * (x + Real.epsilon x)) ω) → ω ≤ 1/6 :=
sorry

end problem_1_omega_theta_problem_2_max_omega_l115_115361


namespace dogs_prevent_wolf_escape_l115_115576

theorem dogs_prevent_wolf_escape
  (wolf_speed dog_speed : ℝ)
  (at_center: True)
  (dogs_at_vertices: True)
  (wolf_all_over_field: True)
  (dogs_on_perimeter: True)
  (wolf_handles_one_dog: ∀ (d : ℕ), d = 1 → True)
  (wolf_handles_two_dogs: ∀ (d : ℕ), d = 2 → False)
  (dog_faster_than_wolf: dog_speed = 1.5 * wolf_speed) : 
  ∀ (wolf_position : ℝ × ℝ) (boundary_position : ℝ × ℝ), 
  wolf_position != boundary_position → dog_speed > wolf_speed → 
  False := 
by sorry

end dogs_prevent_wolf_escape_l115_115576


namespace work_completion_days_l115_115642

theorem work_completion_days 
  (x : ℕ) 
  (h1 : ∀ t : ℕ, t = x → A_work_rate = 2 * (1 / t))
  (h2 : A_and_B_work_together : ∀ d : ℕ, d = 4 → A_B_combined_rate = 1 / d) :
  x = 12 := 
sorry

end work_completion_days_l115_115642


namespace max_value_a_l115_115014

theorem max_value_a (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + y = 1) : 
  ∃ a, a = 16 ∧ (∀ x y, (x > 0 → y > 0 → x + y = 1 → a ≤ (1/x) + (9/y))) :=
by 
  use 16
  sorry

end max_value_a_l115_115014


namespace find_image_point_l115_115638

noncomputable def lens_equation (t f k : ℝ) : Prop :=
  (1 / k) + (1 / t) = (1 / f)

theorem find_image_point
  (O F T T_star K_star K : ℝ)
  (OT OTw OTw_star FK : ℝ)
  (OT_eq : OT = OTw)
  (OTw_star_eq : OTw_star = OT)
  (similarity_condition : ∀ (CTw_star OF : ℝ), CTw_star / OF = (CTw_star + OK) / OK)
  : lens_equation OTw FK K :=
sorry

end find_image_point_l115_115638


namespace husband_additional_payment_l115_115974

theorem husband_additional_payment (total_medical_cost : ℝ) (total_salary : ℝ) 
                                  (half_medical_cost : ℝ) (deduction_from_salary : ℝ) 
                                  (remaining_salary : ℝ) (total_payment : ℝ)
                                  (each_share : ℝ) (amount_paid_by_husband : ℝ) : 
                                  
                                  total_medical_cost = 128 →
                                  total_salary = 160 →
                                  half_medical_cost = total_medical_cost / 2 →
                                  deduction_from_salary = half_medical_cost →
                                  remaining_salary = total_salary - deduction_from_salary →
                                  total_payment = remaining_salary + half_medical_cost →
                                  each_share = total_payment / 2 →
                                  amount_paid_by_husband = 64 →
                                  (each_share - amount_paid_by_husband) = 16 := by
  sorry

end husband_additional_payment_l115_115974


namespace total_sum_step_l115_115198

-- Defining the conditions
def step_1_sum : ℕ := 2

-- Define the inductive process
def total_sum_labels (n : ℕ) : ℕ :=
  if n = 1 then step_1_sum
  else 2 * 3^(n - 1)

-- The theorem to prove
theorem total_sum_step (n : ℕ) : 
  total_sum_labels n = 2 * 3^(n - 1) :=
by
  sorry

end total_sum_step_l115_115198


namespace find_some_base_l115_115393

noncomputable def base_value (some_base : ℝ) (x y : ℝ) : Prop :=
xy = 1 → 
    (some_base ^ (x + y) ^ 2) / (some_base ^ (x - y) ^ 2) = 256 → 
    some_base = 4

theorem find_some_base :
    ∀ (x y : ℝ), xy = 1 →
        ∀ some_base : ℝ, (some_base ^ (x + y) ^ 2) / (some_base ^ (x - y) ^ 2) = 256 → some_base = 4 :=
by
    intros x y xy some_base h
    sorry

end find_some_base_l115_115393


namespace find_g_neg1_l115_115350

-- Define that f(x) is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Given conditions
variables (f g : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_eq : ∀ x : ℝ, f x = g x + x^2)
variable (h_g1 : g 1 = 1)

-- The statement to prove
theorem find_g_neg1 : g (-1) = -3 :=
sorry

end find_g_neg1_l115_115350


namespace units_digit_S_6789_l115_115047

-- Definitions based on the given problem conditions
def c : ℝ := 5 + 4 * Real.sqrt 2
def d : ℝ := 5 - 4 * Real.sqrt 2
def S (n : ℕ) : ℝ := 1/2 * (c^n + d^n)

-- Main theorem statement
theorem units_digit_S_6789 : S 6789 % 10 = 5 :=
sorry

end units_digit_S_6789_l115_115047


namespace campsite_tents_calculation_l115_115459

noncomputable def calculate_tents (initial: ℕ) (rate: ℝ) (days: ℕ) : ℝ :=
  initial * (1 + rate / 100) ^ days

theorem campsite_tents_calculation :
  let northernmost_tents := calculate_tents 100 10 7
  let eastside_tents := calculate_tents 200 5 7
  let center_tents := calculate_tents 400 15 7
  let southern_tents := calculate_tents 200 7 7
  let total_tents := northernmost_tents + eastside_tents + center_tents + southern_tents
  true := 
  northernmost_tents ≈ 194.87171 ∧
  eastside_tents ≈ 281.4215 ∧
  center_tents ≈ 1103.644 ∧
  southern_tents ≈ 321.1562 ∧
  total_tents ≈ 1901 :=
by
  sorry

end campsite_tents_calculation_l115_115459


namespace general_term_formula_l115_115176

-- Define the given sequence as a function
def seq (n : ℕ) : ℤ :=
  match n with
  | 0 => 3
  | n + 1 => if (n % 2 = 0) then 4 * (n + 1) - 1 else -(4 * (n + 1) - 1)

-- Define the proposed general term formula
def a_n (n : ℕ) : ℤ :=
  (-1)^(n+1) * (4 * n - 1)

-- State the theorem that general term of the sequence equals the proposed formula
theorem general_term_formula : ∀ n : ℕ, seq n = a_n n := 
by
  sorry

end general_term_formula_l115_115176


namespace probability_both_red_probability_different_colors_l115_115963

theorem probability_both_red :
  let balls := {1, 2, 3, 4, 'A', 'B'}
  let red_balls := {1, 2, 3, 4}
  let all_outcomes := {{1,2}, {1,3}, {1,4}, {1,'A'}, {1,'B'}, 
                       {2,3}, {2,4}, {2,'A'}, {2,'B'}, 
                       {3,4}, {3,'A'}, {3,'B'}, 
                       {4,'A'}, {4,'B'}, {'A','B'}}
  let event_M := {{1,2}, {1,3}, {1,4}, {2,3}, {2,4}, {3,4}}
  (event_M.card : ℚ) / (all_outcomes.card) = 2 / 5 := 
by
  sorry

theorem probability_different_colors :
  let balls := {1, 2, 3, 4, 'A', 'B'}
  let red_balls := {1, 2, 3, 4}
  let black_balls := {'A', 'B'}
  let all_outcomes := {{1,2}, {1,3}, {1,4}, {1,'A'}, {1,'B'}, 
                       {2,3}, {2,4}, {2,'A'}, {2,'B'}, 
                       {3,4}, {3,'A'}, {3,'B'}, 
                       {4,'A'}, {4,'B'}, {'A','B'}}
  let event_N := {{1,'A'}, {1,'B'}, {2,'A'}, {2,'B'}, 
                  {3,'A'}, {3,'B'}, {4,'A'}, {4,'B'}}
  (event_N.card : ℚ) / (all_outcomes.card) = 8 / 15 := 
by
  sorry

end probability_both_red_probability_different_colors_l115_115963


namespace sum_first_six_terms_l115_115580

variable {a : ℕ → ℤ}
variable {d : ℤ}
variable {a_1 a_2 a_3 a_4 a_5 a_6 : ℤ}

-- Condition: The sequence {a_n} is arithmetic
axiom H1 : ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions from the problem
axiom H2 : a 1 + a 3 = 4
axiom H3 : a 2 + a 4 = 10

noncomputable def sum_to_n (n : ℕ) : ℤ :=
  (n * (a 0 + a n - 1)) / 2

theorem sum_first_six_terms : sum_to_n 6 = 21 := 
by
  sorry

end sum_first_six_terms_l115_115580


namespace tangent_line_at_1_f_monotonicity_f_two_zeros_l115_115027

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - (a + 2) * x + a * x^2

-- Part (I)
theorem tangent_line_at_1 (a : ℝ) (h : a = 0) :
  let y := f 1 0
  ∃ m b : ℝ, (m = -1) ∧ (b = -2) ∧ (∀ x, y = m * x + b) :=
sorry

-- Part (II)
theorem f_monotonicity (a : ℝ) :
  (∀ x, 0 < x → x < 1/2 → deriv (λ x, f x a) x > 0) ∧ 
  (∀ x, x > 1/2 → deriv (λ x, f x a) x < 0) :=
sorry

-- Part (III)
theorem f_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ↔ 
  (a ∈ Iio (-4 * Real.log 2 - 4)) :=
sorry

end tangent_line_at_1_f_monotonicity_f_two_zeros_l115_115027


namespace max_BQ_squared_l115_115839

-- Given a circle ω with diameter AB, extended to a point C, 
-- and point U on the circle such that CU is tangent to the circle.
-- Given AB = 16 and point Q is the foot of the perpendicular from A to line CU.
-- Prove that the maximum possible length of segment BQ squared is 260.

theorem max_BQ_squared 
  (ω : Type) [circ : ω → Prop]
  (A B C U Q : ω)
  (hABdiameter : ω A = ω B)
  (hCextend : A < C)
  (hUonCircle : ω U)
  (hCUtangent : tangent (line C U) ω)
  (hQfoot : foot A (line C U) Q)
  (hABlength : dist A B = 16):
  ∃ n : ℝ, n^2 = 260 := 
sorry -- proof not required

end max_BQ_squared_l115_115839


namespace cannot_determine_number_of_pens_l115_115984

theorem cannot_determine_number_of_pens 
  (P : ℚ) -- marked price of one pen
  (N : ℕ) -- number of pens = 46
  (discount : ℚ := 0.01) -- 1% discount
  (profit_percent : ℚ := 11.91304347826087) -- given profit percent
  : ¬ ∃ (N : ℕ), 
        profit_percent = ((N * P * (1 - discount) - N * P) / (N * P)) * 100 :=
by
  sorry

end cannot_determine_number_of_pens_l115_115984


namespace a4_is_minus_5_l115_115330

noncomputable def sequence (n : ℕ) : ℤ :=
if h : n = 0 then 0 else nat.rec_on n 1 (λ n' an', an' - (n' : ℤ))

theorem a4_is_minus_5 : sequence 4 = -5 := by
sorry

end a4_is_minus_5_l115_115330


namespace area_of_white_square_l115_115855

theorem area_of_white_square
  (face_area : ℕ)
  (total_surface_area : ℕ)
  (blue_paint_area : ℕ)
  (faces : ℕ)
  (area_of_white_square : ℕ) :
  face_area = 12 * 12 →
  total_surface_area = 6 * face_area →
  blue_paint_area = 432 →
  faces = 6 →
  area_of_white_square = face_area - (blue_paint_area / faces) →
  area_of_white_square = 72 :=
by
  sorry

end area_of_white_square_l115_115855


namespace ratio_bee_eaters_leopards_l115_115659

variables (s f l c a t e r : ℕ)

-- Define the conditions from the problem.
def conditions : Prop :=
  s = 100 ∧
  f = 80 ∧
  l = 20 ∧
  c = s / 2 ∧
  a = 2 * (f + l) ∧
  t = 670 ∧
  e = t - (s + f + l + c + a)

-- The theorem statement proving the ratio.
theorem ratio_bee_eaters_leopards (h : conditions s f l c a t e) : r = (e / l) := by
  sorry

end ratio_bee_eaters_leopards_l115_115659


namespace trapezoid_EF_distance_m_plus_n_l115_115102

noncomputable def isosceles_trapezoid_distance
  (AD BC : ℝ) (angle_at_AD : ℝ) (diagonal_length : ℝ) (distance_from_E_to_A : ℝ) (distance_from_E_to_D : ℝ)
  (EF : ℝ) : ℝ :=
if h : AD = 20 * real.sqrt 7 ∧ distance_from_E_to_A = 10 * real.sqrt 7 ∧ distance_from_E_to_D = 30 * real.sqrt 7 ∧ diagonal_length = 10 * real.sqrt 21 ∧ angle_at_AD = real.pi / 3
then 25 * real.sqrt 7 
else EF

theorem trapezoid_EF_distance :
  isosceles_trapezoid_distance 20 20 (real.pi / 3) (10 * real.sqrt 21) (10 * real.sqrt 7) (30 * real.sqrt 7) (25 * real.sqrt 7) = 25 * real.sqrt 7 :=
by sorry

theorem m_plus_n :
  isosceles_trapezoid_distance 20 20 (real.pi / 3) (10 * real.sqrt 21) (10 * real.sqrt 7) (30 * real.sqrt 7) (25 * real.sqrt 7) = 25 * real.sqrt 7 →
  let m := 25 in
  let n := 7 in
  m + n = 32 :=
by intros h; refl

end trapezoid_EF_distance_m_plus_n_l115_115102


namespace expression_evaluation_l115_115663

theorem expression_evaluation : (∛(-8) + (-1)^2023 - |1 - Real.sqrt 2| + (- Real.sqrt 3)^2 = 1 - Real.sqrt 2) := by
  sorry

end expression_evaluation_l115_115663


namespace arithmetic_sequence_a5_l115_115079

-- Define the concept of an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- The problem's conditions
def a₁ : ℕ := 2
def d : ℕ := 3

-- The proof problem
theorem arithmetic_sequence_a5 : arithmetic_sequence a₁ d 5 = 14 := by
  sorry

end arithmetic_sequence_a5_l115_115079


namespace even_num_squares_with_three_same_colored_vertices_l115_115997

-- Define the vertices and their colors
def color (v : ℕ × ℕ) : Prop :=
  if v = (0, 0) ∨ v = (n, n) then true else
  if v = (0, n) ∨ v = (n, 0) then false else sorry

def is_red (v : ℕ × ℕ) : Prop := color v = true
def is_blue (v : ℕ × ℕ) : Prop := color v = false

-- Define the conditions
def is_term (v : ℕ × ℕ) (n : ℕ) : Prop :=
  (v.1 ≤ n) ∧ (v.2 ≤ n)

-- Small Square with exactly three vertices of the same color
def small_square (vertices : set (ℕ × ℕ)) : Prop :=
  ∃ a b c d, {a, b, c, d} = vertices ∧ 
  ((is_red a ∧ is_red b ∧ is_red c ∧ is_blue d) ∨ (is_blue a ∧ is_blue b ∧ is_blue c ∧ is_red d))

-- Counting small squares with three same-colored vertices
noncomputable def count_squares (n : ℕ) : ℕ := sorry

-- The main theorem to prove
theorem even_num_squares_with_three_same_colored_vertices (n : ℕ) : even (count_squares n) := sorry

end even_num_squares_with_three_same_colored_vertices_l115_115997


namespace angle_EMK_ninety_degrees_l115_115628

-- Defining the geometric setup for the problem
variables (A B C K N M E : Point)
variables (circle : Circle Point)

-- The conditions of the problem
axiom right_triangle (h : ∀ (A B C : Point), right_triangle A B C)
axiom inscribed_circle (h : ∀ (A B C : Point) (circle : Circle Point), inscribed A B C circle)
axiom K_arc_midpoint (h : ∀ (B C : Point) (circle : Circle Point), midpoint_arc_uncontain A B C K)
axiom N_midpoint_AC (h : ∀ (A C : Point), midpoint A C N)
axiom M_intersection_ray_KN (h : ∀ (K N : Point) (circle : Circle Point), intersection_point_ray KN circle M)
axiom E_tangent_points_AC (h : ∀ (A C : Point) (circle : Circle Point), tangent_points_intersection A C E)

-- The theorem statement
theorem angle_EMK_ninety_degrees : ∀ (A B C K N M E : Point) (circle : Circle Point)
  (right_triangle A B C) 
  (inscribed_circle A B C circle)
  (K_arc_midpoint B C circle K)
  (N_midpoint_AC A C N)
  (M_intersection_ray_KN K N circle M)
  (E_tangent_points_AC A C circle E),
  angle E M K = 90 := 
sorry

end angle_EMK_ninety_degrees_l115_115628


namespace Sandy_fingernails_reach_world_record_in_20_years_l115_115474

-- Definitions for the conditions of the problem
def world_record_len : ℝ := 26
def current_len : ℝ := 2
def growth_rate : ℝ := 0.1

-- Proof goal
theorem Sandy_fingernails_reach_world_record_in_20_years :
  (world_record_len - current_len) / growth_rate / 12 = 20 :=
by
  sorry

end Sandy_fingernails_reach_world_record_in_20_years_l115_115474


namespace remainder_of_h_x6_l115_115440

def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

noncomputable def remainder_when_h_x6_divided_by_h (x : ℝ) : ℝ :=
  let hx := h x
  let hx6 := h (x^6)
  hx6 - 6 * hx

theorem remainder_of_h_x6 (x : ℝ) : remainder_when_h_x6_divided_by_h x = 6 :=
  sorry

end remainder_of_h_x6_l115_115440


namespace water_percentage_in_mixture_l115_115464

def type1_water_percentage : ℝ := 20 / 100
def type2_water_percentage : ℝ := 35 / 100
def parts_type1 : ℕ := 10
def parts_type2 : ℕ := 4

def total_water : ℝ := 
  (parts_type1 * type1_water_percentage) + 
  (parts_type2 * type2_water_percentage)

def total_parts : ℕ := parts_type1 + parts_type2

def percentage_water (total_water : ℝ) (total_parts : ℕ) : ℝ :=
  (total_water / total_parts) * 100

theorem water_percentage_in_mixture :
  percentage_water total_water total_parts = 24.29 := by
  sorry

end water_percentage_in_mixture_l115_115464


namespace g_84_value_l115_115508

-- Define the function g with the given conditions
def g (x : ℝ) : ℝ := sorry

-- Conditions given in the problem
axiom g_property1 : ∀ x y : ℝ, g (x * y) = y * g x
axiom g_property2 : g 2 = 48

-- Statement to prove
theorem g_84_value : g 84 = 2016 :=
by
  sorry

end g_84_value_l115_115508


namespace correct_statement_distance_l115_115206

-- Defining the concepts based on the conditions.
def is_distance (a b : Point) : ℝ := classical.some (euclidean_distance a b)

-- The main statement of the proof:
theorem correct_statement_distance :
  "The length of the line segment connecting two points is called the distance between the two points" :=
  sorry

end correct_statement_distance_l115_115206


namespace polygon_quad_area_l115_115475

variable (n : ℕ) (P : Set (∑ K))

-- Definition of a convex polygon
def is_convex_polygon (P : Set (∑ K)) : Prop := sorry

-- Definition of the area of the polygon 
def area (P : Set (∑ K)) : ℝ := sorry

-- Definition of vertices of polygon
def vertices (P : Set (∑ K)) : Finset (Σ K) := sorry

-- Quad lateral area
def quad_area (A B C D : Σ K) : ℝ := sorry

-- The main problem statement
theorem polygon_quad_area (h1: is_convex_polygon P) (h2 : 3 < n) (h3 : area P = 1) :
  ∃ (A B C D : Σ K), A ∈ vertices P ∧ B ∈ vertices P ∧ C ∈ vertices P ∧ D ∈ vertices P ∧
  quad_area A B C D ≥ 1/2 := by
  sorry

end polygon_quad_area_l115_115475


namespace hexagon_angle_Q_l115_115434

-- Definitions of the problem conditions
structure RegularHexagon :=
  (A B C D E F Q : Type) -- Vertices of the hexagon and the intersection point
  (hexagon : list (A × B × C × D × E × F)) -- Definition of a regular hexagon 
  (angle_AB_Q : ℝ) -- Angle measure at Q formed by extended sides AB and DE
  (angle_DE_Q : ℝ) -- Angle measure at Q from extended sides DE and AB

-- Main proof problem statement
theorem hexagon_angle_Q (hex : RegularHexagon) : 
  hex.angle_AB_Q + hex.angle_DE_Q = 120 :=
sorry

end hexagon_angle_Q_l115_115434


namespace king_paths_C5_to_H2_l115_115775

-- We define the conditions as follows:
-- 1. horizontal_distance: calculates the absolute difference between column positions
-- 2. vertical_distance: calculates the absolute difference between row positions
-- 3. diagonal_moves: calculates the minimum of horizontal_distance and vertical_distance

def horizontal_distance : ℕ := |8 - 3|
def vertical_distance : ℕ := |5 - 2|
def diagonal_moves : ℕ := min horizontal_distance vertical_distance
def remaining_horizontal : ℕ := horizontal_distance - diagonal_moves
def remaining_vertical : ℕ := vertical_distance - diagonal_moves
def total_steps : ℕ := diagonal_moves + remaining_horizontal + remaining_vertical

-- The total ways to arrange moves is calculated by combinations
def number_of_paths : ℕ := Nat.choose total_steps remaining_horizontal

-- We can now state the theorem to be proved
theorem king_paths_C5_to_H2 : number_of_paths = 15 := by
  sorry

end king_paths_C5_to_H2_l115_115775


namespace sin_cos_graph_shift_right_by_1_l115_115511

noncomputable def sin_cos_shift (x : ℝ) : ℝ := sin ((π / 3) * x + (π / 6))

noncomputable def cos_func (x : ℝ) : ℝ := cos ((π / 3) * x)

theorem sin_cos_graph_shift_right_by_1 :
  ∃ m > 0, ∀ x : ℝ, sin_cos_shift x = cos_func (x - m) ↔ m = 1 :=
by
  sorry

end sin_cos_graph_shift_right_by_1_l115_115511


namespace alpha_values_l115_115122

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else x^2

theorem alpha_values (α : ℝ) (h : f α = 9) : α = -9 ∨ α = 3 :=
begin
  sorry,
end

end alpha_values_l115_115122


namespace payment_superheroes_l115_115499

-- Define the conditions
def productivity_superman (W : ℝ) : ℝ := 0.1 * W
def productivity_flash (W : ℝ) : ℝ := 2 * productivity_superman W

def time_working_superman_before_flash (W : ℝ) : ℝ := 1
def work_done_superman_before_flash (W : ℝ) : ℝ := 0.1 * W

def remaining_work (W : ℝ) : ℝ := 0.9 * W
def combined_productivity (W : ℝ) : ℝ := productivity_superman W + productivity_flash W

def time_working_together (W : ℝ) : ℝ := remaining_work W / combined_productivity W
def total_time_superman (W : ℝ) : ℝ := time_working_superman_before_flash W + time_working_together W
def total_time_flash (W : ℝ) : ℝ := time_working_together W

def payment (t : ℝ) : ℝ := 90 / t

-- Prove that payments are correct
theorem payment_superheroes (W : ℝ) : 
  payment (total_time_superman W) = 22.5 ∧
  payment (total_time_flash W) = 30 :=
by 
  sorry

end payment_superheroes_l115_115499


namespace simplify_expression_l115_115478

theorem simplify_expression :
  2 * Real.sqrt (1 + Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4 - 4 * Real.cos 4 :=
by
  sorry

end simplify_expression_l115_115478


namespace find_m_value_l115_115765

theorem find_m_value (m : ℝ) (A : Set ℝ) (h₁ : A = {0, m, m^2 - 3 * m + 2}) (h₂ : 2 ∈ A) : m = 3 :=
by
  sorry

end find_m_value_l115_115765


namespace xy_sufficient_but_not_necessary_l115_115052

theorem xy_sufficient_but_not_necessary (x y : ℝ) : (x > 0 ∧ y > 0) → (xy > 0) ∧ ¬(xy > 0 → (x > 0 ∧ y > 0)) :=
by
  intros h
  sorry

end xy_sufficient_but_not_necessary_l115_115052


namespace problem1_l115_115268

theorem problem1 :
  (\left(Real.cbrt 2 * Real.sqrt 3\right)^6 + 
   \left(Real.sqrt \left(2 * Real.sqrt 2\right)\right)^(4 / 3) -
   4 * (16 / 49)^(-1 / 2) -
   Real.sqrt \left(Real.quartrt 2\right) * 8^(0.25) -
   (-2005 : ℝ)^0 = 100) := sorry

end problem1_l115_115268


namespace derangement_formula_l115_115727

def derangement_count (n : ℕ) : ℕ :=
  n! * (∑ k in Finset.range (n + 1), (-1 : ℤ)^k / k!)

theorem derangement_formula (n : ℕ) : derangement_count n = n! * ∑ k in Finset.range (n + 1), (-1 : ℤ)^k / k! := 
sorry

end derangement_formula_l115_115727


namespace perpendicular_PQ_AC_l115_115118

open EuclideanGeometry

variables {A B C D E F P M X Y Q : Point}
variable {ω : Circle}

-- Given conditions in the problem
variables (convex_ABCD : convex_quad ABCD)
variables (angle_ABC_ADC : ∠ ABC = ∠ ADC)
variable (angle_lt_90 : ∠ ABC < 90)
variables (mid_M : midpoint A C M)
variables (circumcircle_BPD : is_circumcircle ω B P D)
variables (intersect_BM_ω : BM ∩ ω = [X])
variables (intersect_DM_ω : DM ∩ ω = [Y])
variables (E_bisector : is_bisector (∠ ABC) AC E)
variables (F_bisector : is_bisector (∠ ADC) AC F)
variables (intersect_angle_bisectors : P = E ∩ F)
variables (Q_intersect : Q = XE ∩ YF)

-- Statement to be proved
theorem perpendicular_PQ_AC :
  PQ ⊥ AC :=
sorry

end perpendicular_PQ_AC_l115_115118


namespace base16_to_binary_bit_count_l115_115562

theorem base16_to_binary_bit_count : 
  let hex_value := 0xA3F52 in
  ∃ n : ℕ, nat.log2 (hex_value) + 1 = 20 :=
by
  let hex_value := 0xA3F52
  use 20
  sorry

end base16_to_binary_bit_count_l115_115562


namespace trapezoid_division_l115_115802

theorem trapezoid_division (a b m p q : ℝ) (h_pos : m > 0) (ratio_pos : p > 0 ∧ q > 0) :
  ∃ x : ℝ, x = real.sqrt ((p * a^2 + q * b^2) / (p + q)) :=
by
  sorry

end trapezoid_division_l115_115802


namespace most_likely_number_of_red_balls_l115_115545

-- Define the conditions
def total_balls : ℕ := 20
def red_ball_frequency : ℝ := 0.8

-- Define the statement we want to prove
theorem most_likely_number_of_red_balls : red_ball_frequency * total_balls = 16 :=
by sorry

end most_likely_number_of_red_balls_l115_115545


namespace exists_a_b_l115_115468

theorem exists_a_b (S : Finset ℕ) (hS : S.card = 43) :
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (a^2 - b^2) % 100 = 0 := 
by
  sorry

end exists_a_b_l115_115468


namespace significant_improvement_l115_115593

section RubberProductElongation

-- Given conditions
def x : Fin 10 → ℕ := ![545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def y : Fin 10 → ℕ := ![536, 527, 543, 530, 560, 533, 522, 550, 576, 536]
def z (i : Fin 10) : ℤ := (x i : ℤ) - (y i : ℤ)

-- Definitions for sample mean and sample variance
def sample_mean (f : Fin 10 → ℤ) : ℤ := (∑ i, f i) / 10
def sample_variance (f : Fin 10 → ℤ) (mean : ℤ) : ℤ := (∑ i, (f i - mean) ^ 2) / 10

-- Correct answers
def z_bar : ℤ := sample_mean z
def s_squared : ℤ := sample_variance z z_bar

-- Proof statement for equivalence
theorem significant_improvement :
  z_bar = 11 ∧
  s_squared = 61 ∧
  (z_bar ≥ 2 * Real.sqrt (s_squared / 10)) :=
by
  sorry

end RubberProductElongation

end significant_improvement_l115_115593


namespace find_cost_price_l115_115953

noncomputable def cost_price (C : ℝ) :=
  let SP1 := 1.10 * C
  let SP2 := SP1 + 100
  SP2 = 1.15 * C

theorem find_cost_price : ∃ C : ℝ, cost_price C ∧ C = 2000 :=
by {
  use 2000,
  unfold cost_price,
  simp,
  sorry
}

end find_cost_price_l115_115953


namespace determine_x_l115_115376

theorem determine_x (x : ℝ) (hx : 1 ∈ ({x, x^2} : set ℝ)) (distinct : x ≠ x^2) : x = -1 := 
by 
  sorry

end determine_x_l115_115376


namespace polar_equation_of_curve_and_max_triangle_area_l115_115421

theorem polar_equation_of_curve_and_max_triangle_area : 
  (∀ r φ, r > 0 → 
    (let x := √3 + r * cos φ in 
     let y := 1 + r * sin φ in 
      (∃ θ ρ, ρ = 4 * sin (θ + π / 3) ∧ 
               ρ * cos (θ + π / 6) + 1 = 0 ))) 
  ∧ 
  (∃ θ, -π / 3 < θ ∧ θ < 2 * π / 3 ∧ 
      let ρ₁ := 4 * sin (θ + π / 3) in 
      let ρ₂ := 4 * sin (θ + π / 2) in 
      1/4 * ρ₁ * ρ₂ = 2 + √3) :=
sorry

end polar_equation_of_curve_and_max_triangle_area_l115_115421


namespace length_AD_l115_115060

noncomputable def find_AD (triangle_ABC : Type*) 
  [euclidean_geometry triangle_ABC] 
  (A B C D : triangle_ABC)
  (AB AC : ℝ) (BC CD : ℝ) 
  (H : midpoint_line_segment A B) 
  (CH_squared : ℝ) : ℝ :=
(AD : ℝ) :=
if AB = 8 ∧ AC = 8 ∧ BC = 3 ∧ CD = 9 ∧ (CH_squared = 61.75) then
  1.5 + real.sqrt 19.25
else 
  0 -- assuming no distance has a physical length of 0.

theorem length_AD (A B C D : Type*) 
  [euclidean_geometry A] 
  (AB AC : ℝ) (BC CD : ℝ) 
  (H : midpoint_line_segment A B) 
  (CH_squared : ℝ) 
  : find_AD A B C D AB AC BC CD H CH_squared = 5.887487845 :=
begin
  sorry
end

end length_AD_l115_115060


namespace problem_statement_l115_115356

variables (e1 e2 : Vector ℝ 2) (λ : ℝ)

-- Conditions
def is_unit_vector (v : Vector ℝ 2) : Prop := ∥v∥ = 1
def angle_between_vectors (v1 v2 : Vector ℝ 2) (α : ℝ) : Prop := 
  acos (v1 ⬝ v2 / (∥v1∥ * ∥v2∥)) = α

def vec_a : Vector ℝ 2 := 3 • e1 + (λ - 1) • e2
def vec_b : Vector ℝ 2 := (2 * λ - 1) • e1 - 2 • e2

-- perpendicularity condition between vec_a and vec_b
def perpendicular (v1 v2 : Vector ℝ 2) : Prop := v1 ⬝ v2 = 0

-- Proof Statement
theorem problem_statement 
  (h1 : is_unit_vector e1)
  (h2 : is_unit_vector e2)
  (h3 : angle_between_vectors e1 e2 (π / 3))
  (h4 : perpendicular (vec_a e1 e2 λ) (vec_b e1 e2 λ))
  : (λ > 0 → vec_a e1 e2 λ = e1) ∧ (λ < 0 → vec_a e1 e2 λ = (2 • e1 - 3 • e2) / Real.sqrt 7) :=
sorry

end problem_statement_l115_115356


namespace range_of_m_l115_115034

def A (m : ℝ) : set (ℝ × ℝ) := {p | ∃ (x y : ℝ), (p = (x, y)) ∧ y = x^2 + m * x + 2}
def B : set (ℝ × ℝ) := {p | ∃ (x y : ℝ), (p = (x, y)) ∧ (x - y + 1 = 0) ∧ (0 ≤ x ∧ x ≤ 2)}

theorem range_of_m (m : ℝ) : (A m ∩ B ≠ ∅) → m ≤ -1 :=
by
  sorry

end range_of_m_l115_115034


namespace platform_length_eq_train_length_l115_115513

-- Define the speed of the train in km/hr
def speed_kmhr : ℝ := 36

-- Convert speed to m/s
def speed_ms : ℝ := speed_kmhr * (1000 / 3600)

-- Define the time to cross the platform in seconds
def time_sec : ℝ := 60

-- Define the length of the train in meters
def train_length : ℝ := 300

-- Define the length of the platform we need to find
def platform_length : ℝ := 600 - train_length

-- The relationship that needs to be proven
theorem platform_length_eq_train_length : 
  platform_length = train_length := 
sorry

end platform_length_eq_train_length_l115_115513


namespace correct_propositions_l115_115843

variables (m n : Line) (alpha beta gamma : Plane)

def Prop1 : Prop :=
  (m ∥ n ∧ n ∥ alpha →  m ∥ alpha ∨ m ⊆ alpha)

def Prop4 : Prop :=
  (alpha ∥ beta ∧ beta ∥ gamma ∧ m ⊥ alpha → m ⊥ gamma)

theorem correct_propositions :
  Prop1 m n alpha ∧ Prop4 m alpha beta gamma :=
by
  sorry

end correct_propositions_l115_115843


namespace polynomial_value_at_five_l115_115195

def f (x : ℤ) : ℤ := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 6 * x + 7

theorem polynomial_value_at_five : f 5 = 2677 := by
  -- The proof goes here.
  sorry

end polynomial_value_at_five_l115_115195


namespace calculate_dividend_l115_115209

def divisor : ℕ := 21
def quotient : ℕ := 14
def remainder : ℕ := 7
def expected_dividend : ℕ := 301

theorem calculate_dividend : (divisor * quotient + remainder = expected_dividend) := 
by
  sorry

end calculate_dividend_l115_115209


namespace total_wheels_in_both_garages_l115_115654

/-- Each cycle type has a different number of wheels. --/
def wheels_per_cycle (cycle_type: String) : ℕ :=
  if cycle_type = "bicycle" then 2
  else if cycle_type = "tricycle" then 3
  else if cycle_type = "unicycle" then 1
  else if cycle_type = "quadracycle" then 4
  else 0

/-- Define the counts of each type of cycle in each garage. --/
def garage1_counts := [("bicycle", 5), ("tricycle", 6), ("unicycle", 9), ("quadracycle", 3)]
def garage2_counts := [("bicycle", 2), ("tricycle", 1), ("unicycle", 3), ("quadracycle", 4)]

/-- Total steps for the calculation --/
def wheels_in_garage (garage_counts: List (String × ℕ)) (missing_wheels_unicycles: ℕ) : ℕ :=
  List.foldl (λ acc (cycle_count: String × ℕ) => 
              acc + (if cycle_count.1 = "unicycle" then (cycle_count.2 * wheels_per_cycle cycle_count.1 - missing_wheels_unicycles) 
                     else (cycle_count.2 * wheels_per_cycle cycle_count.1))) 0 garage_counts

/-- The total number of wheels in both garages. --/
def total_wheels : ℕ := wheels_in_garage garage1_counts 0 + wheels_in_garage garage2_counts 3

/-- Prove that the total number of wheels in both garages is 72. --/
theorem total_wheels_in_both_garages : total_wheels = 72 :=
  by sorry

end total_wheels_in_both_garages_l115_115654


namespace sum_of_first_n_terms_l115_115534

theorem sum_of_first_n_terms (n : ℕ) :
  (∑ k in finset.range n, (2^k)) - n = 2^(n+1) - n - 2 :=
by {
  sorry
}

end sum_of_first_n_terms_l115_115534


namespace mean_is_not_51_l115_115251

def frequencies : List Nat := [5, 8, 7, 13, 7]
def pH_values : List Float := [4.8, 4.9, 5.0, 5.2, 5.3]

def total_observations : Nat := List.sum frequencies

def mean (freqs : List Nat) (values : List Float) : Float :=
  let weighted_sum := List.sum (List.zipWith (· * ·) values (List.map (Float.ofNat) freqs))
  weighted_sum / (Float.ofNat total_observations)

theorem mean_is_not_51 : mean frequencies pH_values ≠ 5.1 := by
  -- Proof skipped
  sorry

end mean_is_not_51_l115_115251


namespace range_of_a_l115_115053

noncomputable def f (a x : ℝ) : ℝ := (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ∧ 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a = -1 := 
by
  sorry

end range_of_a_l115_115053


namespace diamond_identity_l115_115710

def diamond (a b : ℝ) : ℝ := real.sqrt (a ^ 2 + b ^ 2)

theorem diamond_identity : diamond (diamond 7 24) (diamond (-24) 7) = 25 * real.sqrt 2 := by
  sorry

end diamond_identity_l115_115710


namespace trigonometric_identity_l115_115714

theorem trigonometric_identity (α β : ℝ) 
  (h : cos (α + β) * cos β + sin (α + β) * sin β = 1 / 3) : 
  2 * sin (α / 2) ^ 2 - 1 = - (1 / 3) :=
by sorry

end trigonometric_identity_l115_115714


namespace land_for_crop_production_l115_115427

-- Conditions as Lean definitions
def total_land : ℕ := 150
def house_and_machinery : ℕ := 25
def future_expansion : ℕ := 15
def cattle_rearing : ℕ := 40

-- Proof statement defining the goal
theorem land_for_crop_production : 
  total_land - (house_and_machinery + future_expansion + cattle_rearing) = 70 := 
by
  sorry

end land_for_crop_production_l115_115427


namespace reflected_ray_eqn_l115_115742

theorem reflected_ray_eqn (P : ℝ × ℝ)
  (incident_ray : ∀ x : ℝ, P.2 = 2 * P.1 + 1)
  (reflection_line : P.2 = P.1) :
  P.1 - 2 * P.2 - 1 = 0 :=
sorry

end reflected_ray_eqn_l115_115742


namespace inradius_of_isosceles_triangle_l115_115104

theorem inradius_of_isosceles_triangle (A B C I : Point) (r : ℝ)
  (h_isosceles : A.dist B = A.dist C)
  (h_BC : B.dist C = 40)
  (h_I_incenter : I = incenter A B C)
  (h_I_C : I.dist C = 25) :
  inradius A B C = 15 :=
by
  sorry

end inradius_of_isosceles_triangle_l115_115104


namespace remainder_b31_div_33_l115_115438

def b (n : ℕ) : ℕ :=
  (List.range (n + 1)).drop 1 |> List.join

theorem remainder_b31_div_33 :
  let b_n := b 31
  b_n % 33 = 11 := by
sorry

end remainder_b31_div_33_l115_115438


namespace max_segment_length_is_1995_l115_115311

noncomputable def max_segment_length (A B : ℕ → ℕ) : ℕ :=
  let P := 1995 in
  if periodic A P ∧ ∀ n, ¬periodic B n ∨ B n ≠ A (n % P) then P else 0

theorem max_segment_length_is_1995 (A B : ℕ → ℕ) :
  (∀ m, (A (m + 1995) = A m)) →
  (∀ n, (¬(∀ m, B (m + n) = B m) ∨ ∀ k, ∃ j < 1995, B (k + j) = A j)) →
  max_segment_length A B = 1995 :=
begin
  sorry
end

end max_segment_length_is_1995_l115_115311


namespace calculate_c5_l115_115679

def seq (c : ℕ → ℕ) (n : ℕ) : ℕ :=
if n = 1 then 3 else if n = 2 then 4 else c (n - 1) * c (n - 2) + 1

theorem calculate_c5 : seq (seq seq) 5 = 690 :=
by
  sorry

end calculate_c5_l115_115679


namespace sodium_chloride_moles_produced_l115_115703

theorem sodium_chloride_moles_produced (NaOH HCl NaCl : ℕ) : 
    (NaOH = 3) → (HCl = 3) → NaCl = 3 :=
by
  intro hNaOH hHCl
  -- Placeholder for actual proof
  sorry

end sodium_chloride_moles_produced_l115_115703


namespace closed_form_f_l115_115097

noncomputable def f (a d x : ℝ) : ℝ :=
  ∑ n in (set.range (nat.succ)), (x^n) / (a * (set.range (nat.succ n)).prod (λ k, a + k * d))

theorem closed_form_f (a d x : ℝ) (h₁ : 0 < a) (h₂ : 0 < d) : 
  f a d x = 
    let g : ℝ → ℝ := 
      if d ≠ 1 
      then λ x, (exp (x^d / d) * (∫ t in 0..x, exp(-t^d / d) * t^(a-1) dt + C))
      else λ x, (exp(x) * (∫ t in 0..x, exp(-t) * t^(a-1) dt + C)) in
    g (x^(1/d)) / x^(a/d) :=
sorry

end closed_form_f_l115_115097


namespace mary_total_earnings_l115_115856

theorem mary_total_earnings (earn_per_home : ℕ) (num_homes_cleaned : ℕ)
  (h1 : earn_per_home = 46) (h2 : num_homes_cleaned = 6) : earn_per_home * num_homes_cleaned = 276 :=
by
  rw [h1, h2]
  norm_num
  sorry

end mary_total_earnings_l115_115856


namespace possible_integer_roots_l115_115990

theorem possible_integer_roots (b2 b1 : ℤ) :
  {r : ℤ | eval r (X^3 + C b2 * X^2 + C b1 * X - C 13) = 0} ⊆ {-13, -1, 1, 13} :=
by
  sorry

end possible_integer_roots_l115_115990


namespace primes_of_form_P_P_plus_1_below_19_digits_l115_115692

theorem primes_of_form_P_P_plus_1_below_19_digits : 
  ∀ (P : ℕ), prime (P^P + 1) → (P^P + 1).digits ≤ 19 → P = 1 ∨ P = 2 ∨ P = 4 :=
by
  sorry

end primes_of_form_P_P_plus_1_below_19_digits_l115_115692


namespace common_divisors_count_9240_6300_l115_115373

theorem common_divisors_count_9240_6300 : 
  let gcd_val := Nat.gcd 9240 6300 
  in Nat.totient gcd_val = 24 :=
by
  let d := Nat.gcd 9240 6300
  -- gcd(9240, 6300) = 420
  have h1 : d = 420 := by sorry
  have h2 : Nat.totient 420 = 24 := by sorry
  exact h2

end common_divisors_count_9240_6300_l115_115373


namespace find_angle_BCA_l115_115071

-- Definitions and conditions
def AB : ℝ := 10  -- Length of side AB in cm
def AC : ℝ := 5.1 -- Length of side AC in cm
def angle_CAB : ℝ := 58 * Real.pi / 180  -- Convert degrees to radians

-- Goal 
theorem find_angle_BCA : 
  ∃ (BCA : ℝ), |BCA - (58.31 * Real.pi / 180)| < 0.01 * Real.pi / 180 :=
by 
  -- Lean code to formally verify this would be written here.
  sorry

end find_angle_BCA_l115_115071


namespace part1_part2_l115_115721

-- Definition of the function f
def f (a x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- First Proof Statement: Inequality for a = 2
theorem part1 : ∀ x : ℝ, - (1 : ℝ) / 3 ≤ x ∧ x ≤ 5 → f 2 x ≤ 1 :=
by
  sorry

-- Second Proof Statement: Range for a such that -4 ≤ f(x) ≤ 4 for all x ∈ ℝ
theorem part2 : ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4 ↔ a = 1 ∨ a = -1 :=
by
  sorry

end part1_part2_l115_115721


namespace applicants_less_4_years_no_degree_l115_115132

theorem applicants_less_4_years_no_degree
    (total_applicants : ℕ)
    (A : ℕ) 
    (B : ℕ)
    (C : ℕ)
    (D : ℕ)
    (h_total : total_applicants = 30)
    (h_A : A = 10)
    (h_B : B = 18)
    (h_C : C = 9)
    (h_D : total_applicants - (A - C + B - C + C) = D) :
  D = 11 :=
by
  sorry

end applicants_less_4_years_no_degree_l115_115132


namespace find_a_l115_115387

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (a^2 + a = 6)) : a = 2 :=
sorry

end find_a_l115_115387


namespace rectangle_dimensions_l115_115992

theorem rectangle_dimensions (w l : ℚ) (h1 : 2 * l + 2 * w = 2 * l * w) (h2 : l = 3 * w) :
  w = 4 / 3 ∧ l = 4 :=
by
  sorry

end rectangle_dimensions_l115_115992


namespace parallel_planes_of_perpendicular_to_common_line_l115_115004

variables {Line Plane : Type} [IsLine Line] [IsPlane Plane]

/-- Definition of perpendicular relationship between a line and a plane --/
def perpendicular (m : Line) (α : Plane) : Prop := sorry

/-- Definition of parallel relationship between planes --/
def parallel (α β : Plane) : Prop := sorry

variables (m : Line) (α β : Plane)

theorem parallel_planes_of_perpendicular_to_common_line
  (hmα : perpendicular m α) (hmβ : perpendicular m β) :
  parallel α β :=
sorry

end parallel_planes_of_perpendicular_to_common_line_l115_115004


namespace constant_distance_hyperbola_l115_115308

theorem constant_distance_hyperbola:
  ∃ (a b : ℝ), 
    (∀ x : ℝ, (x ∉ set.Icc 0 0) → 
      (sqrt ((x - a)^2 + (1/x - b)^2) = sqrt 2) ∧ 
      (sqrt ((x + a)^2 + (1/x + b)^2) = sqrt 2)) := 
sorry

end constant_distance_hyperbola_l115_115308


namespace fraction_to_percentage_l115_115394

theorem fraction_to_percentage (y : ℝ) (h : y > 0) : ((7 * y) / 20 + (3 * y) / 10) = 0.65 * y :=
by
  -- the proof steps will go here
  sorry

end fraction_to_percentage_l115_115394


namespace not_possible_to_obtain_five_equal_numbers_l115_115557

def initial_numbers : List ℕ := [2, 3, 5, 7, 11]

def arithmetic_mean (a b : ℕ) : ℚ := (a + b) / 2

theorem not_possible_to_obtain_five_equal_numbers (numbers : List ℕ) (h : numbers = initial_numbers) :
  ¬ ∃ m, ∀ x ∈ numbers, x = m :=
by
  have sum := numbers.sum
  -- The sum of the initial numbers is 28
  have sum_initial : numbers.sum = 28 := by
    rw [h]
    simp only [List.sum_cons]
    norm_num
  -- Since the sum must remain constant and 28 is not divisible by 5, we conclude
  have not_divisible_by_5 : 28 % 5 ≠ 0 := by norm_num
  exact λ ⟨m, hm⟩, not_divisible_by_5 (by
    rw [← List.sum_repeat m 5] at sum_initial
    norm_num at sum_initial
    assumption)

end not_possible_to_obtain_five_equal_numbers_l115_115557


namespace tax_rate_calculation_l115_115975

-- Definitions for conditions
def total_paid : ℝ := 30
def sales_tax : ℝ := 1.28
def cost_tax_free : ℝ := 12.72

-- Calculation for taxable items before tax
def cost_taxable_before_tax : ℝ := total_paid - cost_tax_free

-- Definition for tax rate
def tax_rate : ℝ := (sales_tax / cost_taxable_before_tax) * 100

-- The theorem to be proven
theorem tax_rate_calculation : tax_rate ≈ 7.41 :=
by
  sorry

end tax_rate_calculation_l115_115975


namespace sum_of_squares_correct_l115_115548

-- Define the three incorrect entries
def incorrect_entry_1 : Nat := 52
def incorrect_entry_2 : Nat := 81
def incorrect_entry_3 : Nat := 111

-- Define the sum of the squares of these entries
def sum_of_squares : Nat := incorrect_entry_1 ^ 2 + incorrect_entry_2 ^ 2 + incorrect_entry_3 ^ 2

-- State that this sum of squares equals 21586
theorem sum_of_squares_correct : sum_of_squares = 21586 := by
  sorry

end sum_of_squares_correct_l115_115548


namespace monic_polynomials_equal_l115_115833

noncomputable def monic_polynomial (P : Polynomial ℂ) : Prop :=
  P.leadingCoeff = 1

noncomputable def composition_equal (P Q : Polynomial ℂ) : Prop :=
  ∀ x : ℂ, Polynomial.eval (Polynomial.eval P x) = Polynomial.eval (Polynomial.eval Q x)

theorem monic_polynomials_equal (P Q : Polynomial ℂ) 
  (hP : monic_polynomial P) 
  (hQ : monic_polynomial Q) 
  (hPQ : composition_equal P Q) : P = Q := 
sorry

end monic_polynomials_equal_l115_115833


namespace distance_MN_is_one_l115_115433

variables (A B C D A' B' C' D' M N : Type)
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D]
variables [inhabited A'] [inhabited B'] [inhabited C'] [inhabited D']
variables [inhabited M] [inhabited N]

def distance (p1 p2 : (ℝ × ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

noncomputable def find_distance_MN : ℝ :=
  let A' := (0, 0, 12) in
  let B' := (2, 0, 10) in
  let C' := (2, 1, 20) in
  let D' := (0, 1, 24) in
  let M := ((A'.1 + C'.1) / 2, (A'.2 + C'.2) / 2, (A'.3 + C'.3) / 2) in
  let N := ((B'.1 + D'.1) / 2, (B'.2 + D'.2) / 2, (B'.3 + D'.3) / 2) in
  distance M N

theorem distance_MN_is_one : find_distance_MN = 1 :=
sorry

end distance_MN_is_one_l115_115433


namespace sum_decomposition_exists_l115_115838

variable (α : ℝ) (hα : 0 < α ∧ α ≤ 1)
variable (A : Set ℕ)
variable (hA : ∀ n : ℕ, n > 0 → (A ∩ {i : ℕ | 1 ≤ i ∧ i ≤ n}).card ≥ (α * n).to_nat)

theorem sum_decomposition_exists (h : 0 < α ∧ α ≤ 1) :
  ∃ c : ℕ, ∀ n : ℕ, ∃ s : Finset ℕ, (∀ x ∈ s, x ∈ A) ∧ s.card ≤ c ∧ s.sum id = n :=
sorry

end sum_decomposition_exists_l115_115838


namespace multiple_of_91_l115_115217

theorem multiple_of_91 (n : ℕ) (h : n > 0) : 
  ∃ (k : ℕ), k > 0 ∧ k * 91 = nat_of_digits 10 ([2] ++ replicate n 2 ++ [0,2,0,2,0]) :=
by
  sorry

end multiple_of_91_l115_115217


namespace parabola_directrix_l115_115505

theorem parabola_directrix (x y : ℝ) (h : x^2 + 12 * y = 0) : y = 3 :=
sorry

end parabola_directrix_l115_115505


namespace daisy_milk_problem_l115_115677

theorem daisy_milk_problem (total_milk : ℝ) (kids_percentage : ℝ) (remaining_milk : ℝ) (used_milk : ℝ) :
  total_milk = 16 →
  kids_percentage = 0.75 →
  remaining_milk = total_milk * (1 - kids_percentage) →
  used_milk = 2 →
  (used_milk / remaining_milk) * 100 = 50 :=
by
  intros _ _ _ _ 
  sorry

end daisy_milk_problem_l115_115677


namespace turnovers_remaining_after_baking_l115_115862

theorem turnovers_remaining_after_baking
  (turnovers_per_jar : ℕ)
  (cakes_per_jar : ℕ)
  (pans_per_jar : ℕ)
  (jars : ℕ)
  (desired_pans : ℕ)
  (desired_cakes : ℕ)
  (remaining_turnovers : ℕ):
  turnovers_per_jar = 16 → 
  cakes_per_jar = 4 → 
  pans_per_jar = 2*0.5 → 
  jars = 4 → 
  desired_pans = 1 → 
  desired_cakes = 6 → 
  remaining_turnovers = 8 :=
by
  intros h_turnovers_per_jar h_cakes_per_jar h_pans_per_jar h_jars h_desired_pans h_desired_cakes
  sorry

end turnovers_remaining_after_baking_l115_115862


namespace zero_in_interval_l115_115512

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 8

theorem zero_in_interval : 
  (∀ x > 0, ContinuousAt f x) → 
  f 3 < 0 → 
  f 4 > 0 → 
  ∃ c ∈ Ioo 3 4, f c = 0 := by
  sorry

end zero_in_interval_l115_115512


namespace green_ish_count_l115_115410

theorem green_ish_count (total : ℕ) (blue_ish : ℕ) (both : ℕ) (neither : ℕ) (green_ish : ℕ) :
  total = 150 ∧ blue_ish = 90 ∧ both = 40 ∧ neither = 30 → green_ish = 70 :=
by
  sorry

end green_ish_count_l115_115410


namespace find_C_l115_115835

noncomputable def A := {x : ℝ | x^2 - 5 * x + 6 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | a * x - 6 = 0}
def C := {a : ℝ | (A ∪ (B a)) = A}

theorem find_C : C = {0, 2, 3} := by
  sorry

end find_C_l115_115835


namespace find_angle_range_l115_115395

theorem find_angle_range (A B C : Type) [Triangle A B C] 
  (AC : A → B → ℝ) (BC : B → C → ℝ) (angle_A : C → Triangle A B C → ℝ)
  (hAC : AC = 2 * Real.sqrt 2) (hBC : BC = 2) :
  (0 < angle_A ∧ angle_A ≤ π / 4) :=
  sorry

end find_angle_range_l115_115395


namespace line_circle_separate_l115_115868

def point_inside_circle (x0 y0 a : ℝ) : Prop :=
  x0^2 + y0^2 < a^2

def not_center_of_circle (x0 y0 : ℝ) : Prop :=
  x0^2 + y0^2 ≠ 0

theorem line_circle_separate (x0 y0 a : ℝ) (h1 : point_inside_circle x0 y0 a) (h2 : a > 0) (h3 : not_center_of_circle x0 y0) :
  ∀ (x y : ℝ), ¬ (x0 * x + y0 * y = a^2 ∧ x^2 + y^2 = a^2) :=
by
  sorry

end line_circle_separate_l115_115868


namespace well_digging_cost_l115_115660

constant pi : ℝ
namespace WellDigging

def radius : ℝ := 1.5
def height1 : ℝ := 4
def height2 : ℝ := 6
def height3 : ℝ := 4
def cost_per_cubic_meter1 : ℝ := 18
def cost_per_cubic_meter2 : ℝ := 21
def cost_per_cubic_meter3 : ℝ := 24

def volume (r h : ℝ) : ℝ := pi * r^2 * h

def volume1 : ℝ := volume radius height1
def volume2 : ℝ := volume radius height2
def volume3 : ℝ := volume radius height3

def cost1 : ℝ := volume1 * cost_per_cubic_meter1
def cost2 : ℝ := volume2 * cost_per_cubic_meter2
def cost3 : ℝ := volume3 * cost_per_cubic_meter3

def total_expenditure : ℝ := cost1 + cost2 + cost3

theorem well_digging_cost :
  total_expenditure ≈ 2078.48 := sorry

end WellDigging

end well_digging_cost_l115_115660


namespace _l115_115146

noncomputable def Guldin_theorem (n : ℕ) (S : Fin n → ℝ) (y : Fin n → ℝ) : Prop :=
  let total_area := Finset.univ.sum S
  let z := (Finset.univ.sum (λ i, y i * S i)) / total_area
  2 * Real.pi * z * total_area = 2 * Real.pi * Finset.univ.sum (λ i, y i * S i)

#check Guldin_theorem

end _l115_115146


namespace amusing_permutations_formula_l115_115240

-- Definition of amusing permutations and their count
def amusing_permutations_count (n : ℕ) : ℕ :=
  2^(n-1)

-- Theorem statement: The number of amusing permutations of the set {1, 2, ..., n} is 2^(n-1)
theorem amusing_permutations_formula (n : ℕ) : 
  -- The number of amusing permutations should be equal to 2^(n-1)
  amusing_permutations_count n = 2^(n-1) :=
by
  sorry

end amusing_permutations_formula_l115_115240


namespace find_max_min_of_f_l115_115321

noncomputable def f (x : ℝ) : ℝ := 6 - 12*x + x^3

theorem find_max_min_of_f : 
  let interval := Set.Icc (-1 / 3) (1 : ℝ)
  ∃ x_max x_min ∈ interval, 
    f x_max = 27 ∧ 
    f x_min = -5 := 
begin
  let interval := Set.Icc (-1 / 3) 1,
  sorry,
end

end find_max_min_of_f_l115_115321


namespace truck_driver_additional_gas_needed_l115_115250

-- Definitions from the conditions
def miles_per_gallon_flat := 3
def distance_total := 90
def initial_gas := 12
def distance_flat := 60
def distance_hill := 30
def gas_rate_increase_factor := 2

-- The theorem we need to prove
theorem truck_driver_additional_gas_needed : 
  let gas_needed_flat := distance_flat / miles_per_gallon_flat in
  let gas_needed_hill := distance_hill / (miles_per_gallon_flat / gas_rate_increase_factor) in
  let total_gas_needed := gas_needed_flat + gas_needed_hill in
  let additional_gas_needed := total_gas_needed - initial_gas in
  additional_gas_needed = 28 := by
  sorry

end truck_driver_additional_gas_needed_l115_115250


namespace propositions_correct_l115_115174

-- Definitions to match conditions from step a)
def is_hyperbola (A B : Point) (k : ℝ) (P : Point) : Prop :=
  abs (dist P A - dist P B)= k

def roots_eccentricities (a b c : ℝ) (ellipse_e hyperbola_e : ℝ) : Prop :=
  let Δ := b^2 - 4 * a * c
  let root1 := (-b + real.sqrt Δ) / (2 * a)
  let root2 := (-b - real.sqrt Δ) / (2 * a)
  root1 = ellipse_e ∧ root2 = hyperbola_e ∧ Δ ≥ 0

def same_foci (h1 h2 e1 e2 : ConicSection) : Prop :=
  ∃ F1 F2 : Point, is_focus h1 F1 ∧ is_focus h1 F2 ∧ 
                    is_focus e1 F1 ∧ is_focus e1 F2 ∧ 
                    is_focus h2 F1 ∧ is_focus h2 F2 ∧ 
                    is_focus e2 F1 ∧ is_focus e2 F2

def chord_proof (A B F1 F2 : Point) (radius inscribed_length : ℝ) 
  (difference_y : ℝ) : Prop :=
  let triangle_area := (1 / 2) * inscribed_length
  let expected_area := (1 / 2) * radius * |F1 - F2| * difference_y
  triangle_area = expected_area ∧ difference_y = 4 / 3

-- Propositions translation
theorem propositions_correct :
  ∃ (P : Point) (A B : Point) (k : ℝ) (ellipse_e hyperbola_e : ℝ) 
    (h1 h2 e1 e2 : ConicSection) (F1 F2 : Point) (A B : Point)
    (radius inscribed_length difference_y : ℝ),
    ¬ is_hyperbola A B k P ∧
    roots_eccentricities 2 (-5) 2 ellipse_e hyperbola_e ∧
    same_foci h1 h2 e1 e2 ∧
    chord_proof A B F1 F2 radius inscribed_length difference_y :=
  sorry

end propositions_correct_l115_115174


namespace payment_superheroes_l115_115500

-- Define the conditions
def productivity_superman (W : ℝ) : ℝ := 0.1 * W
def productivity_flash (W : ℝ) : ℝ := 2 * productivity_superman W

def time_working_superman_before_flash (W : ℝ) : ℝ := 1
def work_done_superman_before_flash (W : ℝ) : ℝ := 0.1 * W

def remaining_work (W : ℝ) : ℝ := 0.9 * W
def combined_productivity (W : ℝ) : ℝ := productivity_superman W + productivity_flash W

def time_working_together (W : ℝ) : ℝ := remaining_work W / combined_productivity W
def total_time_superman (W : ℝ) : ℝ := time_working_superman_before_flash W + time_working_together W
def total_time_flash (W : ℝ) : ℝ := time_working_together W

def payment (t : ℝ) : ℝ := 90 / t

-- Prove that payments are correct
theorem payment_superheroes (W : ℝ) : 
  payment (total_time_superman W) = 22.5 ∧
  payment (total_time_flash W) = 30 :=
by 
  sorry

end payment_superheroes_l115_115500


namespace solve_for_x_l115_115397

variable (x y z a b w : ℝ)
variable (angle_DEB : ℝ)

def angle_sum_D (x y z angle_DEB : ℝ) : Prop := x + y + z + angle_DEB = 360
def angle_sum_E (a b w angle_DEB : ℝ) : Prop := a + b + w + angle_DEB = 360

theorem solve_for_x 
  (h1 : angle_sum_D x y z angle_DEB) 
  (h2 : angle_sum_E a b w angle_DEB) : 
  x = a + b + w - y - z :=
by
  -- Proof not required
  sorry

end solve_for_x_l115_115397


namespace area_of_ABCD_is_integer_l115_115810

-- Define the conditions as outlined
def AB : ℝ := 16
def CD : ℝ := 4

theorem area_of_ABCD_is_integer (AB CD : ℝ) (perp1 : ∀ AB BC, ⟦ AB ⟧ ⊥ ⟦ BC ⟧)
    (perp2 : ∀ BC CD, ⟦ BC ⟧ ⊥ ⟦ CD ⟧)
    (tangent : ∀ BC circ, BC.is_tangent_to circ)
    : (∃ BF : ℝ, (AB * CD = BF) ∧ (BF ∈ ℤ)) → (∃ area : ℤ, area = (AB + CD) * CD) :=
by
  sorry

end area_of_ABCD_is_integer_l115_115810


namespace curve_two_lines_unique_a_l115_115020

theorem curve_two_lines_unique_a (a : ℝ) : 
  (∀ a, C : ℝ, x y : ℝ, x^2 + y^2 + a * x * y = 1 → 
    (C -> ((a = 2 ∨ a = -2) → (x + y = 1 ∨ x + y = -1 ∨ x - y = 1 ∨ x - y = -1)))) → 
    ¬ (∃! a : ℝ, (∀ x y : ℝ, x^2 + y^2 + a * x * y = 1 → (x + y = 1 ∨ x + y = -1 ∨ x - y = 1 ∨ x - y = -1))) := 
by
  sorry

end curve_two_lines_unique_a_l115_115020


namespace constant_k_value_l115_115188

noncomputable def a_n (n : ℕ) : ℝ := if n = 0 then 0 else -2 * n + 41

def S (k : ℕ) : ℝ := ∑ i in Finset.range (k + 1), a_n i

theorem constant_k_value :
  let k : ℕ := 20 in
  (a_1 + a_4 + a_7 = 99) ∧ 
  (a_2 + a_5 + a_8 = 93) ∧ 
  (∀ n : ℕ, if n = 0 then True else S n ≤ S k) ∧ 
  (k = 20 := k) :=
by sorry

end constant_k_value_l115_115188


namespace range_of_function_find_a_for_max_value_l115_115961

-- Statement for Problem (1)
theorem range_of_function (f : ℝ → ℝ) (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x = (1/2)^(-x^2 + 4*x + 1)) :
  set.range (λ x : ℝ, f x) = set.Icc (1/32 : ℝ) (1/2 : ℝ) :=
sorry

-- Statement for Problem (2)
theorem find_a_for_max_value (f : ℝ → ℝ) (a : ℝ) (h : ∀ x ∈ set.Icc (0 : ℝ) 1, f x = -x^2 + 2*a*x + 1 - a) 
  (h_max : ∃ x ∈ set.Icc (0 : ℝ) 1, f x = 2) :
  a = -1 ∨ a = 2 :=
sorry

end range_of_function_find_a_for_max_value_l115_115961


namespace correct_expression_l115_115948

variable (a b : ℝ)

theorem correct_expression : (∃ x, x = 3 * a + b^2) ∧ 
    (x = (3 * a + b)^2 ∨ x = 3 * (a + b)^2 ∨ x = 3 * a + b^2 ∨ x = (a + 3 * b)^2) → 
    x = 3 * a + b^2 := by sorry

end correct_expression_l115_115948


namespace value_of_x_l115_115716

theorem value_of_x (x y : ℕ) (h1 : x / y = 3) (h2 : y = 25) : x = 75 := by
  sorry

end value_of_x_l115_115716


namespace part_a_exists_infinitely_many_matrices_rational_entries_part_b_exists_finitely_many_matrices_integer_entries_l115_115672

namespace MatrixDeterminant

theorem part_a_exists_infinitely_many_matrices_rational_entries :
  ∃ (M : Matrix (Fin 3) (Fin 3) ℚ), (det M = 1) ∧ (∃^∞ (M : Matrix (Fin 3) (Fin 3) ℚ), det M = 1) := 
sorry

theorem part_b_exists_finitely_many_matrices_integer_entries :
  ∃ (M : Matrix (Fin 3) (Fin 3) ℤ), (det M = 1) ∧ (finite {M : Matrix (Fin 3) (Fin 3) ℤ | det M = 1}) :=
sorry

end MatrixDeterminant

end part_a_exists_infinitely_many_matrices_rational_entries_part_b_exists_finitely_many_matrices_integer_entries_l115_115672


namespace valid_outfit_choices_l115_115043

theorem valid_outfit_choices :
  let shirts := 5 in
  let pants := 3 in
  let hats := 7 in
  let total_outfits := shirts * pants * hats in
  let colors_shared := 3 in  -- colors red, green, blue
  let invalid_combinations := colors_shared * shirts in
  total_outfits - invalid_combinations = 90
:= by
  let shirts := 5
  let pants := 3
  let hats := 7
  let total_outfits := shirts * pants * hats
  let colors_shared := 3  -- colors red, green, blue
  let invalid_combinations := colors_shared * shirts
  show total_outfits - invalid_combinations = 90 from
    sorry

end valid_outfit_choices_l115_115043


namespace average_price_per_book_l115_115149

def books_from_shop1 := 42
def price_from_shop1 := 520
def books_from_shop2 := 22
def price_from_shop2 := 248

def total_books := books_from_shop1 + books_from_shop2
def total_price := price_from_shop1 + price_from_shop2
def average_price := total_price / total_books

theorem average_price_per_book : average_price = 12 := by
  sorry

end average_price_per_book_l115_115149


namespace jane_pens_count_l115_115646

theorem jane_pens_count (initial_pens_alex : Nat)
  (doubling_rate : Nat)
  (weeks : Nat)
  (pens_difference : Nat) :
  initial_pens_alex = 4 →
  doubling_rate = 2 →
  weeks = 4 →
  pens_difference = 16 →
  let final_pens_alex := initial_pens_alex * (doubling_rate ^ weeks) in
  final_pens_alex = 16 + (final_pens_alex - pens_difference) →
  final_pens_alex - pens_difference = 16 :=
by
  intros h0 h1 h2 h3
  simp only [final_pens_alex, h0, h1, h2, h3]
  sorry

end jane_pens_count_l115_115646


namespace cost_of_fencing_around_circular_field_l115_115570

theorem cost_of_fencing_around_circular_field
  (d : ℝ) (rate : ℝ)
  (hd : d = 42) (hr : rate = 5) :
  let C := Real.pi * d in
  let cost := C * rate in
  cost ≈ 660 :=
by
  sorry

end cost_of_fencing_around_circular_field_l115_115570


namespace number_of_special_numbers_l115_115771

-- Define the range of numbers between 100 and 799
def inRange (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 799

-- Function to check if digits of a number are in strictly increasing order
def digitsIncreasing (n : ℕ) : Prop :=
  let digits := List.ofDigits $ toDigits 10 n in
  List.all (List.zip digits (List.drop 1 digits)) (λ p, p.1 < p.2)

-- Function to check if the last two digits are equal
def lastTwoEqual (n : ℕ) : Prop :=
  let digits := List.ofDigits $ toDigits 10 n in
  List.get digits (digits.length - 1) = List.get digits (digits.length - 2)

-- Define the set of numbers that satisfy either condition
def specialNumbers : Finset ℕ := 
  (Finset.range 800).filter (λ n, inRange n ∧ (digitsIncreasing n ∨ lastTwoEqual n))

theorem number_of_special_numbers : specialNumbers.card = 56 :=
  sorry

end number_of_special_numbers_l115_115771


namespace total_rectangles_l115_115706

-- Definitions
def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 4
def exclude_line_pair: ℕ := 1
def total_combinations (n m : ℕ) : ℕ := Nat.choose n m

-- Statement
theorem total_rectangles (h_lines : ℕ) (v_lines : ℕ) 
  (exclude_pair : ℕ) (valid_h_comb : ℕ) (valid_v_comb : ℕ) :
  h_lines = horizontal_lines →
  v_lines = vertical_lines →
  exclude_pair = exclude_line_pair →
  valid_h_comb = total_combinations 5 2 - exclude_pair →
  valid_v_comb = total_combinations 4 2 →
  valid_h_comb * valid_v_comb = 54 :=
by intros; sorry

end total_rectangles_l115_115706


namespace students_taking_statistics_l115_115069

-- Definitions based on conditions
def total_students := 89
def history_students := 36
def history_or_statistics := 59
def history_not_statistics := 27

-- The proof problem
theorem students_taking_statistics : ∃ S : ℕ, S = 32 ∧
  ((history_students - history_not_statistics) + S - (history_students - history_not_statistics)) = history_or_statistics :=
by
  use 32
  sorry

end students_taking_statistics_l115_115069


namespace find_a_l115_115164

noncomputable def f : ℝ → ℝ := sorry

theorem find_a (a : ℝ) :
  (∀ x : ℝ, f(x) + f(1 - x) = 10) →
  (∀ x : ℝ, f(x) + f(-x) = 7) →
  (∀ x : ℝ, f(1 + x) = a + f(x)) →
  a = 3 :=
by
  intros h1 h2 h3
  sorry

end find_a_l115_115164


namespace cosine_angle_AM_AC_l115_115739

-- Conditions: equilateral triangle and vector M condition
variables {A B C M : ℝ}

def equilateral_triangle (A B C : ℝ) : Prop :=
  ∀ X Y Z, dist X Y = dist Y Z ∧ dist Y Z = dist Z X

def vector_condition (A B C M : ℝ) : Prop :=
  (M - A) = (1 / 3) * (B - A) + (1 / 2) * (C - A)

theorem cosine_angle_AM_AC (h1 : equilateral_triangle A B C) (h2 : vector_condition A B C M) :
  cos ∠((M - A), (C - A)) = (4 * sqrt 19) / 19 :=
  sorry

end cosine_angle_AM_AC_l115_115739


namespace prime_iff_binom_mod_l115_115100

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def binom_mod (n k : ℕ) : ℕ := (nat.choose (n - 1) k) % n

theorem prime_iff_binom_mod (n : ℕ) : 
  is_prime n ↔ ∀ k : ℕ, k < n → binom_mod n k = (-1 : ℤ)^k % (n : ℤ) := 
sorry

end prime_iff_binom_mod_l115_115100


namespace train_crossing_time_l115_115962

theorem train_crossing_time 
  (lt : ℕ) -- length of the train
  (ts : ℕ) -- time to cross the signal pole
  (lp : ℕ) -- length of the platform
  (speed := lt / ts) -- the speed of the train
  (total_distance := lt + lp) -- total distance to cross the platform
  (T := total_distance / speed) -- time to cross the platform
  (h1 : lt = 300) 
  (h2 : ts = 26) 
  (h3 : lp = 150) : 
  T ≈ 39 :=
by 
  -- Define constants and calculations based on the conditions
  rw [h1, h2, h3] at *
  unfold speed total_distance T
  -- Now, we need to prove the calculated time is approximately 39 seconds
  have : (300 * 26) = (26 / 26) := sorry
  -- Similar steps to conclude the proof will go here
  sorry

end train_crossing_time_l115_115962


namespace dan_wins_probability_l115_115932
open Classical

-- Define the events and probabilities as given conditions
def probability_six := 1 / 6
def probability_no_six := 5 / 6

-- Define the common ratio for the geometric series
def common_ratio := (probability_no_six) * (probability_no_six)

-- Define the first term of the geometric series
def first_term := probability_six

-- Define the formula for the infinite geometric series sum
def infinite_geometric_sum (a r : ℚ) : ℚ := a / (1 - r)

-- Prove the main result
theorem dan_wins_probability : 
  infinite_geometric_sum first_term common_ratio = 6 / 11 :=
by 
  -- Begin the proof by setting up the problem context
  unfold infinite_geometric_sum first_term common_ratio,
  -- Substitute the known values and simplify
  calc
    (1 / 6) / (1 - (25 / 36)) = (1 / 6) / ((36 - 25) / 36) : by sorry
    ... = (1 / 6) / (11 / 36) : by sorry
    ... = (1 / 6) * (36 / 11)  : by sorry
    ... = 6 / 11              : by sorry

end dan_wins_probability_l115_115932


namespace train_crosses_pole_in_l115_115818

noncomputable def train_crossing_time (length : ℝ) (speed_km_hr : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * (5.0 / 18.0)
  length / speed_m_s

theorem train_crosses_pole_in : train_crossing_time 175 180 = 3.5 :=
by
  -- Proof would be here, but for now, it is omitted.
  sorry

end train_crosses_pole_in_l115_115818


namespace minimize_cost_l115_115608

noncomputable def total_cost (x : ℝ) : ℝ := (16000000 / x) + 40000 * x

theorem minimize_cost : ∃ (x : ℝ), x > 0 ∧ (∀ y > 0, total_cost x ≤ total_cost y) ∧ x = 20 := 
sorry

end minimize_cost_l115_115608


namespace log_product_identity_l115_115698

theorem log_product_identity (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : y ≠ 1) :
  (log (x^2) / log (y^4)) * (log (y^3) / log (x^6)) * (log (x^5) / log (y^2)) * 
  (log (y^2) / log (x^5)) * (log (x^6) / log (y^3)) = (1 / 4) * (log x / log y) :=
by
  sorry

end log_product_identity_l115_115698


namespace angle_QRS_is_right_l115_115423

/-- In a quadrilateral PQRS with PQ = QR = RS = SP and ∠PQR = 90°, show that ∠QRS = 90°. -/
theorem angle_QRS_is_right (P Q R S : Type*)
  [plane_geometry P]
  (h1 : P ∈ triangle P Q R)
  (h2 : PQ_length = QR_length)
  (h3 : QR_length = RS_length)
  (h4 : RS_length = SP_length)
  (h5 : angle P Q R = 90) :
  angle Q R S = 90 :=
sorry

end angle_QRS_is_right_l115_115423


namespace mason_total_bricks_l115_115865

theorem mason_total_bricks :
  (let courses_per_wall := 15,
       bricks_per_course := 25,
       complete_walls := 7,
       incomplete_wall_courses := 14 in
   (complete_walls * (courses_per_wall * bricks_per_course) + incomplete_wall_courses * bricks_per_course) = 2975) :=
by
  let courses_per_wall := 15
  let bricks_per_course := 25
  let complete_walls := 7
  let incomplete_wall_courses := 14
  have h1 : complete_walls * (courses_per_wall * bricks_per_course) =  2625 := by sorry
  have h2 : incomplete_wall_courses * bricks_per_course = 350 := by sorry
  have h3 : 2625 + 350 = 2975 := by sorry
  show 2975 = 2975, by rfl

end mason_total_bricks_l115_115865


namespace height_of_model_tower_l115_115623

noncomputable def volume_ratio (actual_volume model_volume : ℝ) : ℝ :=
  actual_volume / model_volume

noncomputable def scale_factor (volume_ratio : ℝ) : ℝ :=
  real.cbrt volume_ratio

theorem height_of_model_tower (height_actual_tower : ℝ) (volume_actual : ℝ) (volume_model : ℝ)
  (h_actual : height_actual_tower = 60) (v_actual : volume_actual = 200000) (v_model : volume_model = 0.2) : 
  let ratio := volume_ratio volume_actual volume_model,
      scale := scale_factor ratio in
  height_actual_tower / scale = 0.6 :=
by
  sorry

end height_of_model_tower_l115_115623


namespace significant_improvement_l115_115602

noncomputable def z (x y : ℕ → ℕ) (i : ℕ) : ℕ := x i - y i

noncomputable def z_bar (x y : ℕ → ℕ) : ℝ := 
  (z x y 1 + z x y 2 + z x y 3 + z x y 4 + z x y 5 + z x y 6 + 
  z x y 7 + z x y 8 + z x y 9 + z x y 10) / 10

noncomputable def s_squared (x y : ℕ → ℕ) : ℝ := 
  let mean := z_bar x y in 
  ( (z x y 1 - mean) ^ 2 + (z x y 2 - mean) ^ 2 + (z x y 3 - mean) ^ 2 +
    (z x y 4 - mean) ^ 2 + (z x y 5 - mean) ^ 2 + (z x y 6 - mean) ^ 2 +
    (z x y 7 - mean) ^ 2 + (z x y 8 - mean) ^ 2 + (z x y 9 - mean) ^ 2 +
    (z x y 10 - mean) ^ 2) / 10

theorem significant_improvement (x y : ℕ → ℕ)
  (hx : x 1 = 545) (hx2 : x 2 = 533) (hx3 : x 3 = 551) (hx4 : x 4 = 522)
  (hx5 : x 5 = 575) (hx6 : x 6 = 544) (hx7 : x 7 = 541) (hx8 : x 8 = 568)
  (hx9 : x 9 = 596) (hx10 : x 10 = 548)
  (hy : y 1 = 536) (hy2 : y 2 = 527) (hy3 : y 3 = 543) (hy4 : y 4 = 530)
  (hy5 : y 5 = 560) (hy6 : y 6 = 533) (hy7 : y 7 = 522) (hy8 : y 8 = 550)
  (hy9 : y 9 = 576) (hy10 : y 10 = 536) :
  z_bar x y ≥ 2 * real.sqrt(s_squared x y / 10) :=
  sorry

end significant_improvement_l115_115602


namespace angle_between_a_c_is_30deg_l115_115847

open Real

variables (a b c : ℝ^3)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1)
variables (hlin : linear_independent ℝ ![a, b, c])
variables (hvec : a × (b × c) = (2 / sqrt 3) • (b + c))

theorem angle_between_a_c_is_30deg :
  real.angle a c = 30 :=
sorry

end angle_between_a_c_is_30deg_l115_115847


namespace elder_age_is_29_l115_115168

-- Define the problem parameters
variables (y e : ℤ)

-- Given conditions translated to Lean definitions
def condition1 := e = y + 20  -- The ages of two persons differ by 20 years
def condition2 := e - 4 = 5 * (y - 4)  -- 4 years ago, the elder one was 5 times as old as the younger one

-- Conjecture: The present age of the elder person is 29 years.
theorem elder_age_is_29 (h1 : condition1) (h2 : condition2) : e = 29 :=
by sorry

end elder_age_is_29_l115_115168


namespace consecutive_integers_and_sum_of_bases_l115_115836

theorem consecutive_integers_and_sum_of_bases (C D : ℕ) (h1 : C < D) (h2 : 231_C + 56_D = 105_(C+D)) : C + D = 7 :=
sorry

end consecutive_integers_and_sum_of_bases_l115_115836


namespace problem_statement_l115_115323

theorem problem_statement (x y : ℝ) (h₁ : |x| = 3) (h₂ : |y| = 4) (h₃ : x > y) : 2 * x - y = 10 := 
by {
  sorry
}

end problem_statement_l115_115323


namespace M_subset_N_l115_115852

open Set

def M : Set ℝ := {x | ∃ k : ℤ, x = (k : ℝ) / 2 + 1 / 4 }
def N : Set ℝ := {x | ∃ k : ℤ, x = (k : ℝ) / 4 + 1 / 2 }

theorem M_subset_N : M ⊆ N :=
sorry

end M_subset_N_l115_115852


namespace instantaneous_velocity_at_t2_l115_115964

open Real

theorem instantaneous_velocity_at_t2 :
  (∃ (s : ℝ → ℝ), (∀ t, s t = (1 / 8) * t^2) ∧ deriv s 2 = 1 / 2) :=
by {
  let s := λ t : ℝ, (1 / 8) * t^2,
  use s,
  split,
  { intro t,
    refl, },
  { have h_deriv : deriv s 2 = (1 / 4) * 2,
    { dsimp [s],
      calc deriv (λ t, (1 / 8) * t^2) 2
          = (1 / 8) * deriv (λ t, t^2) 2 : by apply deriv_const_mul
      ... = (1 / 8) * (2 * 2) : by simp [deriv_pow],
    },
    rw h_deriv,
    norm_num,
  },
  sorry
}

end instantaneous_velocity_at_t2_l115_115964


namespace complex_subtraction_complex_multiplication_l115_115734

noncomputable def z1 : ℂ := 2 - 3 * complex.i
noncomputable def z2 : ℂ := (15 - 5 * complex.i) / (2 + complex.i^2)

theorem complex_subtraction :
  z1 - z2 = -13 + 2 * complex.i := by
  sorry

theorem complex_multiplication :
  z1 * z2 = 15 - 55 * complex.i := by
  sorry

end complex_subtraction_complex_multiplication_l115_115734


namespace zelda_bound_longer_than_henry_stride_l115_115369

-- Define the problem and its conditions
theorem zelda_bound_longer_than_henry_stride :
  let h_strides := 30
  let h_gaps := 30
  let z_bounds := 8
  let z_gaps := 30
  let distance_feet := 7920
  let h_stride_length := distance_feet / (h_strides * h_gaps)
  let z_bound_length := distance_feet / (z_bounds * z_gaps)
  z_bound_length - h_stride_length = 24.2 := 
by
  let h_strides := 30
  let h_gap_count := 30
  let z_bounds := 8
  let z_gap_count := 30
  let distance := 7920
  let h_stride_length := distance / (h_strides * h_gap_count)
  let z_bound_length := distance / (z_bounds * z_gap_count)
  have stride_length := rfl -- h_stride_length == 7920 / 900 = 8.8 implies a rfl check
  have bound_length := rfl -- z_bound_length == 7920 / 240 = 33 implies another rfl check
  show 33 - 8.8 = 24.2,
  sorry

end zelda_bound_longer_than_henry_stride_l115_115369


namespace rational_sequence_l115_115162

def unit_digit_sum (n : Nat) : Nat := (List.range n).map (λ x => (x + 1) ^ 2 % 10).sum % 10

theorem rational_sequence :
  ∃ P : ℕ, ∀ n : ℕ, unit_digit_sum (n + P) = unit_digit_sum n →
  ∃ r s : ℕ, (0.a₁ a₂ ... aₙ ...) = r / s :=
sorry

end rational_sequence_l115_115162


namespace carterHas152Cards_l115_115126

-- Define the number of baseball cards Marcus has.
def marcusCards : Nat := 210

-- Define the number of baseball cards Carter has.
def carterCards : Nat := marcusCards - 58

-- Theorem to prove Carter's baseball cards total 152 given the conditions.
theorem carterHas152Cards (h1 : marcusCards = 210) (h2 : marcusCards = carterCards + 58) : carterCards = 152 :=
by
  -- Proof omitted for this exercise
  sorry

end carterHas152Cards_l115_115126


namespace domain_of_f_l115_115941

noncomputable def f (x: ℝ) : ℝ := log 2 (log 3 (log 6 (log 7 x)))

theorem domain_of_f : ∀ x, (117649 < x) ↔ real.bounded_above (set_of (λ x, ∃ f x)) :=
by 
  sorry

end domain_of_f_l115_115941


namespace stacy_berries_l115_115160

theorem stacy_berries (skylar_berries : ℕ) (h1 : skylar_berries = 20) 
  (steve_berries : ℕ) (h2 : steve_berries = (1 / 2) * skylar_berries) : 
  ∃ stacy_berries : ℕ, stacy_berries = 3 * steve_berries + 2 ∧ stacy_berries = 32 :=
by
  exists (3 * steve_berries + 2)
  split
  {
    sorry
  }
  {
    sorry
  }

end stacy_berries_l115_115160


namespace probability_diagonals_intersection_l115_115922

theorem probability_diagonals_intersection (n : ℕ) (h_n : n = 10) (d : ℕ) (h_d : d = 3):
  let total_diagonals := (nat.choose n 2) - n in
  let chosen_ways := nat.choose total_diagonals d in
  let intersecting_cases := (nat.choose 10 5) * 5 in
  let probability := (intersecting_cases : ℚ) / (chosen_ways : ℚ) in
  probability = 252 / 1309 :=
by
  sorry

end probability_diagonals_intersection_l115_115922


namespace n_divisible_by_100_l115_115261

theorem n_divisible_by_100 (n : ℤ) (h1 : n > 101) (h2 : 101 ∣ n)
  (h3 : ∀ d : ℤ, 1 < d ∧ d < n → d ∣ n → ∃ k m : ℤ, k ∣ n ∧ m ∣ n ∧ d = k - m) : 100 ∣ n :=
sorry

end n_divisible_by_100_l115_115261


namespace max_diff_units_digit_l115_115919

theorem max_diff_units_digit (n : ℕ) (h1 : n = 850 ∨ n = 855) : ∃ d, d = 5 :=
by 
  sorry

end max_diff_units_digit_l115_115919


namespace conjugate_of_given_z_in_second_quadrant_l115_115016

-- Define the given complex number
noncomputable def given_z : ℂ := 5 / (2 * complex.I - 1)

-- Define the conjugate of the given complex number
def conjugate_z : ℂ := complex.conj given_z

-- Define a function to determine the quadrant of a complex number
def quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
  else "On an axis"

-- Statement to prove that the conjugate of z is in the second quadrant
theorem conjugate_of_given_z_in_second_quadrant : quadrant conjugate_z = "Second quadrant" := by
  sorry

end conjugate_of_given_z_in_second_quadrant_l115_115016


namespace average_excluding_highest_lowest_l115_115893

-- Define the conditions
def batting_average : ℚ := 59
def innings : ℕ := 46
def highest_score : ℕ := 156
def score_difference : ℕ := 150
def lowest_score : ℕ := highest_score - score_difference

-- Prove the average excluding the highest and lowest innings is 58
theorem average_excluding_highest_lowest :
  let total_runs := batting_average * innings
  let runs_excluding := total_runs - highest_score - lowest_score
  let effective_innings := innings - 2
  runs_excluding / effective_innings = 58 := by
  -- Insert proof here
  sorry

end average_excluding_highest_lowest_l115_115893


namespace color_lines_l115_115274

noncomputable def ceil (x : ℝ) : ℤ := if x ≤ ⌊x⌋.toReal then ⌊x⌋ else ⌊x⌋ + 1

theorem color_lines {n : ℕ} (h₁ : n ≥ 3) 
  (h₂ : ∀ (L : finset (ℝ × ℝ)), L.card = n → (∀ (l1 l2 : ℝ × ℝ), l1 ≠ l2 → l1 ≠ l2 ∧ ∀ (x : ℝ × ℝ), (x = l1 ∨ x = l2) → ∃ p : ℝ, p = l1.1 ∨ p = l2.1) → 
    ¬ ∃ (p : ℝ × ℝ), ∀ (l ∈ L), p = l) : 
  ∃ k : ℕ, (k ≥ ceil (real.sqrt (n / 2))) ∧ (∀ (L : finset (ℝ × ℝ)), 
    L.card = n → (∃ B : finset (ℝ × ℝ), B ⊆ L ∧ B.card = k ∧ (∀ S ∈ (F L), ¬ ∀ l ∈ S, l ∈ B))) : 
by
  sorry

end color_lines_l115_115274


namespace total_milks_taken_l115_115709

def total_milks (chocolateMilk strawberryMilk regularMilk : Nat) : Nat :=
  chocolateMilk + strawberryMilk + regularMilk

theorem total_milks_taken :
  total_milks 2 15 3 = 20 :=
by
  sorry

end total_milks_taken_l115_115709


namespace students_enrolled_for_german_l115_115065

theorem students_enrolled_for_german 
  (total_students : ℕ)
  (both_english_german : ℕ)
  (only_english : ℕ)
  (at_least_one_subject : total_students = 32 ∧ both_english_german = 12 ∧ only_english = 10) :
  ∃ G : ℕ, G = 22 :=
by
  -- Lean proof steps will go here.
  sorry

end students_enrolled_for_german_l115_115065


namespace parabola_fixed_point_l115_115759

noncomputable def fixed_point (p : ℝ) (hp : p > 0) : Prop :=
  ∀ (k b : ℝ), k ≠ 0 → b ≠ 0 →
  let l := λ (x : ℝ), k * x + b in
  ∃ (x1 x2 y1 y2 : ℝ),
    (y1 = k * x1 + b) ∧ (y2 = k * x2 + b) ∧ 
    (y1^2 = 2 * p * x1) ∧ (y2^2 = 2 * p * x2) ∧ 
    let k_OA := y1 / x1 in
    let k_OB := y2 / x2 in
    k_OA * k_OB = real.sqrt 3 → 
    l (- 2 * real.sqrt 3 * p / 3) = 0 

theorem parabola_fixed_point :
  ∀ (p : ℝ), p > 0 → fixed_point p (by linarith) :=
sorry

end parabola_fixed_point_l115_115759


namespace f_is_odd_l115_115613

noncomputable def f : ℝ → ℝ := sorry

axiom hx : ∀ x y : ℝ, x ∈ set.Ioo (-1 : ℝ) (1 : ℝ) → y ∈ set.Ioo (-1 : ℝ) (1 : ℝ) → 
  f x - f y = f ((x - y) / (1 - x * y))

axiom hnegative : ∀ x : ℝ, x ∈ set.Ioo (-1 : ℝ) 0 → f x < 0

theorem f_is_odd : ∀ x : ℝ, x ∈ set.Ioo (-1 : ℝ) (1 : ℝ) → f x = - f (-x) :=
begin
  sorry
end

end f_is_odd_l115_115613


namespace cyclist_north_speed_l115_115550

variable {v : ℝ} -- Speed of the cyclist going north.

-- Conditions: 
def speed_south := 15 -- Speed of the cyclist going south (15 kmph).
def time := 2 -- The time after which they are 50 km apart (2 hours).
def distance := 50 -- The distance they are apart after 2 hours (50 km).

-- Theorem statement:
theorem cyclist_north_speed :
    (v + speed_south) * time = distance → v = 10 := by
  intro h
  sorry

end cyclist_north_speed_l115_115550


namespace determine_winner_l115_115971

-- Definition of movements towards each class and the threshold to win
def movement_towards_class_2 := [0.2, 0.8]
def movement_towards_class_1 := [0.5, 1.4, 1.3]
def threshold := 2

-- Function to compute total movement towards class 1
def total_movement : ℝ :=
  - (movement_towards_class_2.sum) + (movement_towards_class_1.sum)

-- Proof statement
theorem determine_winner (movement_towards_class_2 movement_towards_class_1: list ℝ) (threshold : ℝ) 
  (h1 : movement_towards_class_2 = [0.2, 0.8]) 
  (h2 : movement_towards_class_1 = [0.5, 1.4, 1.3]) 
  (h3 : threshold = 2) : 
  total_movement ≥ threshold :=
by {
  sorry
}

end determine_winner_l115_115971


namespace victor_percentage_of_marks_l115_115196

theorem victor_percentage_of_marks (marks_obtained max_marks : ℝ) (percentage : ℝ) 
  (h_marks_obtained : marks_obtained = 368) 
  (h_max_marks : max_marks = 400) 
  (h_percentage : percentage = (marks_obtained / max_marks) * 100) : 
  percentage = 92 := by
sorry

end victor_percentage_of_marks_l115_115196


namespace class_average_is_86_l115_115403

-- Definitions of given conditions
def total_students : Nat := 30
def boys_percentage : Float := 0.40
def girls_percentage : Float := 0.60

def boys_math_avg : Float := 0.80
def boys_science_avg : Float := 0.75
def boys_literature_avg : Float := 0.85

def girls_math_avg : Float := 0.90
def girls_science_avg : Float := 0.92
def girls_literature_avg : Float := 0.88

def math_weight : Float := 0.40
def science_weight : Float := 0.30
def literature_weight : Float := 0.30

open Real in
theorem class_average_is_86 :
  let boys_students := boys_percentage * total_students
  let girls_students := girls_percentage * total_students
  let boys_weighted_avg := boys_math_avg * math_weight +
                           boys_science_avg * science_weight +
                           boys_literature_avg * literature_weight
  let girls_weighted_avg := girls_math_avg * math_weight +
                            girls_science_avg * science_weight +
                            girls_literature_avg * literature_weight
  let overall_class_avg := boys_weighted_avg * (boys_students / total_students) +
                           girls_weighted_avg * (girls_students / total_students)
  overall_class_avg = 0.86 := 
  let boys_students := boys_percentage * total_students in
  let girls_students := girls_percentage * total_students in
  let boys_weighted_avg := boys_math_avg * math_weight + 
                           boys_science_avg * science_weight + 
                           boys_literature_avg * literature_weight in
  let girls_weighted_avg := girls_math_avg * math_weight + 
                            girls_science_avg * science_weight + 
                            girls_literature_avg * literature_weight in
  let overall_class_avg := boys_weighted_avg * (boys_students / total_students) +
                           girls_weighted_avg * (girls_students / total_students) in
  by 
    simp [boys_students, girls_students, boys_weighted_avg, girls_weighted_avg, overall_class_avg]
    sorry

end class_average_is_86_l115_115403


namespace f_monotonic_m_range_l115_115023

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan x - 2 * x

theorem f_monotonic {x : ℝ} (h : x ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2)) :
  Monotone f :=
sorry

theorem m_range {x : ℝ} (h : x ∈ Set.Ioo 0 (Real.pi / 2)) {m : ℝ} (hm : f x ≥ m * x^2) :
  m ≤ 0 :=
sorry

end f_monotonic_m_range_l115_115023


namespace significant_improvement_l115_115595

section RubberProductElongation

-- Given conditions
def x : Fin 10 → ℕ := ![545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def y : Fin 10 → ℕ := ![536, 527, 543, 530, 560, 533, 522, 550, 576, 536]
def z (i : Fin 10) : ℤ := (x i : ℤ) - (y i : ℤ)

-- Definitions for sample mean and sample variance
def sample_mean (f : Fin 10 → ℤ) : ℤ := (∑ i, f i) / 10
def sample_variance (f : Fin 10 → ℤ) (mean : ℤ) : ℤ := (∑ i, (f i - mean) ^ 2) / 10

-- Correct answers
def z_bar : ℤ := sample_mean z
def s_squared : ℤ := sample_variance z z_bar

-- Proof statement for equivalence
theorem significant_improvement :
  z_bar = 11 ∧
  s_squared = 61 ∧
  (z_bar ≥ 2 * Real.sqrt (s_squared / 10)) :=
by
  sorry

end RubberProductElongation

end significant_improvement_l115_115595


namespace line_equation_curve_equation_distance_range_solve_inequality_prove_inequality_l115_115582

-- Part I-(1)
theorem line_equation (t : ℝ) :
  ∃ t, ∀ x y : ℝ, (x = t - 3) ∧ (y = \sqrt{3}t) →
  (\sqrt{3} * x - y + 3 * \sqrt{3} = 0) :=
sorry

theorem curve_equation (ρ θ : ℝ) :
  (ρ^2 - 4 * ρ * cos θ + 3 = 0) →
  ((x = ρ * cos θ ∧ y = ρ * sin θ) →
  ((x - 2)^2 + y^2 = 1)) :=
sorry

-- Part I-(2)
theorem distance_range (θ : ℝ) :
  let d := |2 * cos (θ + π / 6) + 5 * \sqrt{3}| / 2 in
  ∀ P : ℝ, (P = 2 + cos θ) ∧ d ranges within [ 5 * \sqrt{3} / 2 - 1, 5 * \sqrt{3} / 2 + 1] :=
sorry

-- Part II-(1)
theorem solve_inequality (x : ℝ) :
  |x-2| + |x+2| ≥ 6 ↔ (x ∈ (-∞, -3] ∪ [3, +∞)) :=
sorry

-- Part II-(2)
theorem prove_inequality (a b : ℝ) (h₀ : |a| < 1) (h₁ : |b| < 1) (h₂ : a ≠ 0) :
  f(ab) > |a| * f(b / a) :=
sorry

end line_equation_curve_equation_distance_range_solve_inequality_prove_inequality_l115_115582


namespace hyperbola_eccentricity_l115_115983

theorem hyperbola_eccentricity
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (H : ∃ A B C : ℝ × ℝ,
    A = (a, 0) ∧
    ∃ (l : ℝ → ℝ), (∀ x, l x = x + a) ∧
    B = (a^2 / (b - a), (a * b) / (b - a)) ∧
    C = (-a^2 / (b + a), - (a * b) / (b + a)) ∧
    ((fst B - a, snd B) = (fst B - a, snd B) ) / 2 = (fst C - fst B, snd C - snd B) / 2
  ) :
  ∃ e : ℝ, e = sqrt 5 := sorry

end hyperbola_eccentricity_l115_115983


namespace sin_70_eq_1_minus_2k_squared_l115_115717

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by
  sorry

end sin_70_eq_1_minus_2k_squared_l115_115717


namespace modulus_z_solve_a_b_l115_115017

noncomputable def z : ℂ :=
  ((1 + complex.i)^2 + 3*(1 - complex.i)) / (2 + complex.i)

theorem modulus_z : complex.abs z = real.sqrt 2 :=
sorry

theorem solve_a_b : ∃ (a b : ℝ), (a = -3) ∧ (b = 4) ∧ (z^2 + a * z + b = 1 + complex.i) :=
begin
  use [-3, 4],
  split, { refl },
  split, { refl },
  sorry
end

end modulus_z_solve_a_b_l115_115017


namespace range_of_a_l115_115736

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a - 1)*x - 1 < 0
def r (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1 ∨ a > 3

theorem range_of_a (a : ℝ) (h : (p a ∨ q a) ∧ ¬(p a ∧ q a)) : r a := 
by sorry

end range_of_a_l115_115736


namespace solve_for_x_l115_115574

theorem solve_for_x
  (x y : ℝ)
  (h1 : x + 2 * y = 100)
  (h2 : y = 25) :
  x = 50 :=
by
  sorry

end solve_for_x_l115_115574


namespace probability_of_one_defective_l115_115749

theorem probability_of_one_defective :
  (2 : ℕ) ≤ 5 → (0 : ℕ) ≤ 2 → (0 : ℕ) ≤ 3 →
  let total_outcomes := Nat.choose 5 2
  let favorable_outcomes := Nat.choose 3 1 * Nat.choose 2 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = (3 / 5 : ℚ) :=
by
  intros h1 h2 h3
  let total_outcomes := Nat.choose 5 2
  let favorable_outcomes := Nat.choose 3 1 * Nat.choose 2 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  have : total_outcomes = 10 := by sorry
  have : favorable_outcomes = 6 := by sorry
  have : probability = (6 / 10 : ℚ) := by sorry
  have : (6 / 10 : ℚ) = (3 / 5 : ℚ) := by sorry
  exact this

end probability_of_one_defective_l115_115749


namespace ways_to_buy_cup_and_saucer_ways_to_buy_set_total_ways_to_buy_items_l115_115569

-- Define the number of items
def cups : ℕ := 5
def saucers : ℕ := 3
def spoons : ℕ := 4

-- Prove the number of ways to buy a cup and a saucer
theorem ways_to_buy_cup_and_saucer (c s : ℕ) : c = cups → s = saucers → c * s = 15 :=
by intros h1 h2; rw [h1, h2]; exact Nat.mul_eq_mul_left 3 5 3 sorry

-- Prove the number of ways to buy a set consisting of a cup, a saucer, and a spoon
theorem ways_to_buy_set (c s t : ℕ) : c = cups → s = saucers → t = spoons → c * s * t = 60 :=
by intros h1 h2 h3; rw [h1, h2, h3]; exact Nat.mul_eq_mul_right 3 (Nat.mul_eq_mul_left 5 4 12 sorry) sorry

-- Prove the total number of different ways to buy items from the store
theorem total_ways_to_buy_items (c s t : ℕ) : c = cups → s = saucers → t = spoons →
   c + s + t + (c * s) + (c * t) + (s * t) + (c * s * t) = 119 :=
by intros h1 h2 h3; rw [h1, h2, h3]; exact Nat.add_eq_add_right (Nat.add_eq_add_right 
   (Nat.add_eq_add_right (Nat.add_eq_add_right (Nat.add_eq_add_right 
   (Nat.mul_eq_mul_left 5 3 15 sorry) (Nat.mul_eq_mul_left 5 4 20 sorry)) 
   (Nat.mul_eq_mul_left 3 4 12 sorry)) (Nat.mul_eq_mul_right 5 (Nat.mul_eq_mul_left 3 4 60 sorry) sorry)) sorry

end ways_to_buy_cup_and_saucer_ways_to_buy_set_total_ways_to_buy_items_l115_115569


namespace triangle_amb_angles_l115_115466

/-- Let M be a point outside a circle with diameter AB. Let MA and MB intersect the circle at points C and D. The area of the circle inscribed in triangle AMB is four times the area of the circle inscribed in triangle CMD. One of the angles in triangle AMB is twice the size of another. Prove that the angles in triangle AMB are 60 degrees, 40 degrees, and 80 degrees. -/
theorem triangle_amb_angles (M A B C D : Point)
  (h1 : ¬(A = B)) -- Diameter AB is non-degenerate
  (h2 : A ≠ C ∧ B ≠ D) -- Points of intersection are distinct
  (h3 : ∃ (circle : Circle), (circle.diameter = A ∧ B)
    ∧ M ∉ circle 
    ∧ C ∈ segment' A circle ∧ D ∈ segment' B circle) -- (M outside circle with diameter AB, and points C and D on circle)
  (h4 : area (inscribed_circle (triangle A M B)) = 4 * area (inscribed_circle (triangle C M D))) -- Ratio of inscribed circles
  (h5 : ∃ x : angle, x ∈ angles_of_triangle (triangle A M B) ∧ 2 * x = angle_sum (angles_of_triangle (triangle A M B) \ {x})) -- One angle is twice another
  : angles (triangle A M B) = {60, 40, 80} :=
sorry

end triangle_amb_angles_l115_115466


namespace number_of_people_in_group_l115_115798

theorem number_of_people_in_group (n : ℕ) (h : (n-1)! = 144) : n = 6 :=
by
  sorry

end number_of_people_in_group_l115_115798


namespace significant_improvement_l115_115596

section RubberProductElongation

-- Given conditions
def x : Fin 10 → ℕ := ![545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def y : Fin 10 → ℕ := ![536, 527, 543, 530, 560, 533, 522, 550, 576, 536]
def z (i : Fin 10) : ℤ := (x i : ℤ) - (y i : ℤ)

-- Definitions for sample mean and sample variance
def sample_mean (f : Fin 10 → ℤ) : ℤ := (∑ i, f i) / 10
def sample_variance (f : Fin 10 → ℤ) (mean : ℤ) : ℤ := (∑ i, (f i - mean) ^ 2) / 10

-- Correct answers
def z_bar : ℤ := sample_mean z
def s_squared : ℤ := sample_variance z z_bar

-- Proof statement for equivalence
theorem significant_improvement :
  z_bar = 11 ∧
  s_squared = 61 ∧
  (z_bar ≥ 2 * Real.sqrt (s_squared / 10)) :=
by
  sorry

end RubberProductElongation

end significant_improvement_l115_115596


namespace problem_statement_l115_115045

variable (X Y : ℝ)

theorem problem_statement
  (h1 : 0.18 * X = 0.54 * 1200)
  (h2 : X = 4 * Y) :
  X = 3600 ∧ Y = 900 := by
  sorry

end problem_statement_l115_115045


namespace geometric_seq_ratio_l115_115011

theorem geometric_seq_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a (n+1) = q * a n)
  (h2 : 0 < q)                    -- ensuring positivity
  (h3 : 3 * a 0 + 2 * q * a 0 = q^2 * a 0)  -- condition from problem
  : ∀ n, (a (n+3) + a (n+2)) / (a (n+1) + a n) = 9 :=
by
  sorry

end geometric_seq_ratio_l115_115011


namespace BA_equals_given_matrix_l115_115119

variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

theorem BA_equals_given_matrix
  (h1 : A + B = A ⬝ B)
  (h2 : A ⬝ B = !![5, 2; -3, 9]) :
  B ⬝ A = !![5, 2; -3, 9] :=
by
  sorry

end BA_equals_given_matrix_l115_115119


namespace part_a_part_b_l115_115831

-- Problem (a)
theorem part_a (f : ℤ → ℝ) 
  (h1 : ∀ n : ℤ, f(n) ≥ 0)
  (h2 : ∀ m n : ℤ, f(m * n) = f(m) * f(n))
  (h3 : ∀ m n : ℤ, f(m + n) ≤ max (f(m)) (f(n))) :
  ∀ n : ℤ, f(n) ≤ 1 :=
by
  sorry

-- Problem (b)
theorem part_b :
  ∃ f : ℤ → ℝ,
    (∀ n : ℤ, f(n) ≥ 0) ∧
    (∀ m n : ℤ, f(m * n) = f(m) * f(n)) ∧
    (∀ m n : ℤ, f(m + n) ≤ max (f(m)) (f(n))) ∧
    0 < f(2) ∧ f(2) < 1 ∧ f(2007) = 1 :=
by
  sorry

end part_a_part_b_l115_115831


namespace problem_statement_l115_115325

noncomputable def f : ℝ → ℝ :=
λ x, if 0 ≤ x ∧ x < 2 then exp x - 1 else sorry

theorem problem_statement : (∀ x, f (x - 1) = -f (-(x - 1))) ∧
                            (∀ x, x ≥ 0 → f (x - 3/2) = f (x + 1/2)) ∧
                            (∀ x, 0 ≤ x ∧ x < 2 → f x = exp x - 1) →
                            f 2016 + f (-2017) = 1 - exp 1 :=
begin
  sorry
end

end problem_statement_l115_115325


namespace solve_for_x_l115_115380

def h (x : ℝ) : ℝ := Real.sqrt ((x + 5) / 5)

theorem solve_for_x : ∀ x : ℝ, h(x) = Real.sqrt ((x + 5) / 5) → (h(3 * x) = 3 * h(x) → x = -20 / 3) :=
by
  intro x
  intro h_x
  intro h_condition
  sorry

end solve_for_x_l115_115380


namespace Laurent_number_greater_than_Chloe_l115_115270

theorem Laurent_number_greater_than_Chloe :
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 2000) →
    let y := (λ (x : ℝ), (uniform [0, 2 * x]).sample) in
    P(y > x) = 1 / 2 :=
by
  let Chloe_distribution := uniform [0, 2000]
  let Laurent_distribution := λ (x : ℝ), uniform [0, 2 * x]
  sorry

end Laurent_number_greater_than_Chloe_l115_115270


namespace find_some_number_l115_115220

theorem find_some_number :
  ∃ (some_number : ℝ), (0.0077 * 3.6) / (some_number * 0.1 * 0.007) = 990.0000000000001 ∧ some_number = 0.04 :=
  sorry

end find_some_number_l115_115220


namespace probability_nonnegative_slope_l115_115754

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b * sin x

theorem probability_nonnegative_slope:
  let s := ({-2, 0, 1, 2} : Finset ℝ),
      α := ∀ (a b : ℝ), a ∈ s → b ∈ s → a ≠ b → 
      (∃ n, n = 7 ∧ card s * (card (s.erase a)) = 12) →
      (∀ x ∈ Ioo 0 (π / 2), a + b * cos x ≥ 0) :=
  ∃ p, p = 7/12 :=
begin
  sorry
end

end probability_nonnegative_slope_l115_115754


namespace cheese_cut_intervals_l115_115214

/-- 
  For any subinterval (c, c + 0.001) within (0, 1), 
  there exists an a in this interval such that the cheese can be divided into 
  two equal-weight piles using cuts in the ratio a : (1 - a).
-/
theorem cheese_cut_intervals : 
  ∀ (c : ℝ), (0 < c ∧ c + 0.001 < 1) → 
  ∃ (a : ℝ), c < a ∧ a < c + 0.001 ∧ ∃ (n : ℕ), 
  ∀ (k : ℕ), (1 ≤ k ∧ k ≤ n) → 
  -- assume cutting process eventually ends in equal weights
  sorry := 
begin
  sorry
end

end cheese_cut_intervals_l115_115214


namespace ap_bd_ce_concurrent_l115_115112

theorem ap_bd_ce_concurrent
    (A B C P D E : Type)
    [euclidean_geometry P]  -- Assume a standard Euclidean plane geometry.

    (h1 : P ∈ interior_triangle A B C)
    (h2 : ∠ A P B - ∠ A C B = ∠ A P C - ∠ A B C)
    (h3 : D = incenter_triangle A P B)
    (h4 : E = incenter_triangle A P C) :
    concurrent AP BD CE :=
sorry

end ap_bd_ce_concurrent_l115_115112


namespace parabola_focus_distance_l115_115236

theorem parabola_focus_distance (A : ℝ × ℝ) (F : ℝ × ℝ := (1, 0)) 
    (h_parabola : A.2^2 = 4 * A.1) (h_distance : dist A F = 3) :
    A = (2, 2 * Real.sqrt 2) ∨ A = (2, -2 * Real.sqrt 2) :=
by
  sorry

end parabola_focus_distance_l115_115236


namespace prob_grid_completely_black_l115_115399

theorem prob_grid_completely_black :
  let prob_center_black := (1 / 2 : ℝ)
  let prob_pair_half_black := (3 / 4 : ℝ)
  let prob_all_pairs :=
        (prob_pair_half_black * prob_pair_half_black * prob_pair_half_black * prob_pair_half_black)
  in
  (prob_center_black * prob_all_pairs) = (81 / 512 : ℝ) :=
by
  let prob_center_black := (1 / 2 : ℝ)
  let prob_pair_half_black := (3 / 4 : ℝ)
  let prob_all_pairs :=
        (prob_pair_half_black * prob_pair_half_black * prob_pair_half_black * prob_pair_half_black)
  let result := prob_center_black * prob_all_pairs
  have calculation : result = (1 / 2 * (3 / 4) ^ 4) := rfl
  have expected : (81 / 512 : ℝ) = (1 / 2 * (3 / 4) ^ 4) := by norm_num
  exact expected.symm ▸ calculation

end prob_grid_completely_black_l115_115399


namespace ultramen_defeat_monster_l115_115928

theorem ultramen_defeat_monster :
  ∀ (one_attack_rate : ℕ) (other_attack_rate : ℕ) (monster_endurance : ℕ),
  one_attack_rate = 12 →
  other_attack_rate = 8 →
  monster_endurance = 100 →
  (monster_endurance / (one_attack_rate + other_attack_rate)) = 5 :=
by 
  intros one_attack_rate other_attack_rate monster_endurance H₁ H₂ H₃
  rw [H₁, H₂, H₃]
  norm_num

end ultramen_defeat_monster_l115_115928


namespace sin_alpha_sub_3pi_over_2_eq_2sqrt5_over_5_l115_115015

theorem sin_alpha_sub_3pi_over_2_eq_2sqrt5_over_5
    (α : Real)
    (h1 : α ∈ (Real.pi / 2, Real.pi))
    (h2 : Real.sin (-Real.pi - α) = Real.sqrt 5 / 5) :
    Real.sin (α - 3 * Real.pi / 2) = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end sin_alpha_sub_3pi_over_2_eq_2sqrt5_over_5_l115_115015


namespace salary_increase_to_original_l115_115187

-- We define the problem conditions
def original_salary (S : ℝ) := S
def reduced_salary (S : ℝ) := 0.72 * S
def percentage_increase (S : ℝ) := (S / (0.72 * S) - 1) * 100

-- We state the theorem
theorem salary_increase_to_original (S : ℝ) : percentage_increase S = 700 / 18 :=
by sorry

end salary_increase_to_original_l115_115187


namespace exists_real_a_gt_1_with_prime_int_part_l115_115092

theorem exists_real_a_gt_1_with_prime_int_part (P : ℕ → Prop)
  (H : ∀ n : ℕ, ∃ p : ℕ, nat.prime p ∧ n^3 < p ∧ p < (n + 1)^3) :
  ∃ a : ℝ, 1 < a ∧ ∀ k : ℕ, nat.prime (⌊a^(3^k)⌋) :=
by
  sorry

end exists_real_a_gt_1_with_prime_int_part_l115_115092


namespace find_a_having_same_min_value_product_of_zeros_gt_neg_m_squared_minus_m_l115_115722

noncomputable def f (x a : ℝ) := (a * x - 1) * Real.exp x
noncomputable def g (x a : ℝ) := x * (Real.log x - a)
noncomputable def F (x a m : ℝ) := x * f x a - m

theorem find_a_having_same_min_value (a : ℝ) :
  (∀ x, f x a ≥ f (1 - a) / a) ∧ (∀ x, g x a ≥ g (Real.exp (a - 1)) a) → a = 1 := 
sorry

theorem product_of_zeros_gt_neg_m_squared_minus_m (m x1 x2 : ℝ) (a : ℝ) :
  m < 0 ∧ (∀ x, F x a m = 0) ∧ (F x1 a m = 0 ∧ F x2 a m = 0) → x1 * x2 > -m^2 - m :=
sorry

end find_a_having_same_min_value_product_of_zeros_gt_neg_m_squared_minus_m_l115_115722


namespace engineer_progress_l115_115649

theorem engineer_progress (x : ℕ) : 
  ∀ (road_length_in_km : ℝ) 
    (total_days : ℕ) 
    (initial_men : ℕ) 
    (completed_work_in_km : ℝ) 
    (additional_men : ℕ) 
    (new_total_men : ℕ) 
    (remaining_work_in_km : ℝ) 
    (remaining_days : ℕ),
    road_length_in_km = 10 → 
    total_days = 300 → 
    initial_men = 30 → 
    completed_work_in_km = 2 → 
    additional_men = 30 → 
    new_total_men = 60 → 
    remaining_work_in_km = 8 → 
    remaining_days = total_days - x →
  (4 * (total_days - x) = 8 * x) →
  x = 100 :=
by
  intros road_length_in_km total_days initial_men completed_work_in_km additional_men new_total_men remaining_work_in_km remaining_days
  intros h1 h2 h3 h4 h5 h6 h7 h8 h_eqn
  -- Proof
  sorry

end engineer_progress_l115_115649


namespace percentage_of_chocolates_with_nuts_eaten_l115_115951

-- Define the constants and conditions
constant total_chocolates : ℕ := 80
constant remaining_chocolates : ℕ := 28
constant chocolates_with_nuts_initial : ℕ := total_chocolates / 2
constant chocolates_without_nuts_eaten : ℕ := chocolates_with_nuts_initial / 2
constant total_chocolates_eaten : ℕ := total_chocolates - remaining_chocolates
constant chocolates_with_nuts_eaten : ℕ := total_chocolates_eaten - chocolates_without_nuts_eaten

-- Define the percentage function
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Prove the percentage of chocolates with nuts eaten
theorem percentage_of_chocolates_with_nuts_eaten :
  percentage chocolates_with_nuts_eaten chocolates_with_nuts_initial = 80 := 
  by
    sorry

end percentage_of_chocolates_with_nuts_eaten_l115_115951


namespace proof_probability_at_least_one_makes_both_shots_l115_115012

-- Define the shooting percentages for Player A and Player B
def shooting_percentage_A : ℝ := 0.4
def shooting_percentage_B : ℝ := 0.5

-- Define the probability that Player A makes both shots
def prob_A_makes_both_shots : ℝ := shooting_percentage_A * shooting_percentage_A

-- Define the probability that Player B makes both shots
def prob_B_makes_both_shots : ℝ := shooting_percentage_B * shooting_percentage_B

-- Define the probability that neither makes both shots
def prob_neither_makes_both_shots : ℝ := (1 - prob_A_makes_both_shots) * (1 - prob_B_makes_both_shots)

-- Define the probability that at least one of them makes both shots
def prob_at_least_one_makes_both_shots : ℝ := 1 - prob_neither_makes_both_shots

-- Prove that the probability that at least one of them makes both shots is 0.37
theorem proof_probability_at_least_one_makes_both_shots :
  prob_at_least_one_makes_both_shots = 0.37 :=
sorry

end proof_probability_at_least_one_makes_both_shots_l115_115012


namespace parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2_l115_115700

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 2 * x

theorem parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2 :
  parabola_equation 1 (-Real.sqrt 2) :=
by
  sorry

end parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2_l115_115700


namespace probability_odd_numbers_appear_l115_115275

noncomputable def fair_eight_sided_die := finset.range 8 -- Assuming 1 to 8

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem probability_odd_numbers_appear :
  (∀ (n : ℕ), n ∈ {1, 3, 5} → ∃ (k : ℕ), k < n ∧ die_rolls[k] ∈ {1, 3, 5}) →
  (∀ k, k < finite_rolls → die_roll[k] ≠ 7 ∧ die_roll[k] ≠ 8) →
  (finite_rolls > 0) →
  probability(eventually(die_rolls, λ x, x > 6) ∧ (∃ i1 i2 i3, i1 < finite_rolls ∧ die_roll[i1] = 1 ∧ i2 < finite_rolls ∧ die_roll[i2] = 3 ∧ i3 < finite_rolls ∧ die_roll[i3] = 5)) = 1/8 :=
begin
    sorry
end

end probability_odd_numbers_appear_l115_115275


namespace vector_difference_magnitude_l115_115061

noncomputable def vector_length_difference 
  (AB AC : ℝ) (θ : ℝ) : ℝ :=
  real.sqrt (AB^2 + AC^2 - 2 * AB * AC * real.cos θ)

theorem vector_difference_magnitude : 
  vector_length_difference 3 2 (real.pi / 3) = real.sqrt 7 := 
sorry

end vector_difference_magnitude_l115_115061


namespace negate_forall_x_ge_1_negation_problem_final_solution_l115_115908

theorem negate_forall_x_ge_1 (P : ℝ → Prop) :
  (¬ (∀ x : ℝ, x ≥ 1 → P x)) ↔ (∃ x : ℝ, x ≥ 1 ∧ ¬ P x) :=
sorry

def P (x : ℝ) : Prop := (x^2 - 1 < 0)

theorem negation_problem :
  (¬ (∀ x : ℝ, x ≥ 1 → P x)) ↔ (∃ x : ℝ, x ≥ 1 ∧ ¬ P x) :=
begin
  apply negate_forall_x_ge_1,
end

theorem final_solution :
  (¬ (∀ x : ℝ, x ≥ 1 → x^2 - 1 < 0)) ↔ (∃ x : ℝ, x ≥ 1 ∧ x^2 - 1 ≥ 0) :=
by
  rw [negation_problem, ← forall_not]
  congr' 1 with x
  exact not_lt

end negate_forall_x_ge_1_negation_problem_final_solution_l115_115908


namespace train_speed_l115_115552

theorem train_speed (v : ℝ) (d : ℝ) : 
  (v > 0) →
  (d > 0) →
  (d + (d - 55) = 495) →
  (d / v = (d - 55) / 25) →
  v = 31.25 := 
by
  intros hv hd hdist heqn
  -- We can leave the proof part out because we only need the statement
  sorry

end train_speed_l115_115552


namespace least_perimeter_triangle_l115_115179

theorem least_perimeter_triangle :
  ∀ (c : ℕ), (c < 61) ∧ (c > 13) → 24 + 37 + c = 75 → c = 14 :=
by
  intros c hc hperimeter
  cases hc with hc1 hc2
  have c_eq_14 : c = 14 := sorry
  exact c_eq_14

end least_perimeter_triangle_l115_115179


namespace sum_of_first_n_terms_l115_115536

theorem sum_of_first_n_terms (n : ℕ) :
  (∑ k in finset.range n, (2^k)) - n = 2^(n+1) - n - 2 :=
by {
  sorry
}

end sum_of_first_n_terms_l115_115536


namespace magnitude_2a_plus_b_l115_115768

variables (a b : ℝ^3) -- assuming vectors in 3-dimensional real space

-- Conditions
def norm_a : ℝ := real.sqrt (a.dot_product a) -- Definition of the norm of a
def norm_b : ℝ := real.sqrt (b.dot_product b) -- Definition of the norm of b

-- Assertions of the given conditions
axiom norm_a_is_two : norm_a = 2
axiom norm_b_is_one : norm_b = 1
axiom dot_a_b : a.dot_product b = -1

-- The theorem we wish to prove
theorem magnitude_2a_plus_b : real.sqrt ((2 • a + b).dot_product (2 • a + b)) = real.sqrt 13 :=
by {
  sorry
}

end magnitude_2a_plus_b_l115_115768


namespace shaded_fraction_l115_115502

noncomputable def radius_of_larger_semicircle : ℝ := 4 / 2

noncomputable def radius_of_smaller_semicircle :=
  let r := classical.some (exists_sqrt (r^2 = 2 - 4 * r + r^2)) in
  2 - r

noncomputable def area_of_semicircle (r : ℝ) : ℝ :=
  (1 / 2) * real.pi * r^2

theorem shaded_fraction :
  let R := radius_of_larger_semicircle in
  let r := radius_of_smaller_semicircle in
  let large_area := area_of_semicircle R in
  let small_area := area_of_semicircle r in
  (large_area - small_area) / large_area = 1 / 2 :=
sorry

end shaded_fraction_l115_115502


namespace magnitude_of_z_minus_1_l115_115384

def z : ℂ := complex.I * (complex.I - 1)

theorem magnitude_of_z_minus_1 : complex.abs (z - 1) = real.sqrt 5 :=
by
  have h : z = complex.I * (complex.I - 1) := rfl
  sorry

end magnitude_of_z_minus_1_l115_115384


namespace fraction_meaningful_l115_115902

theorem fraction_meaningful {x : ℝ} : x ≠ 1 ↔ (x + 2) / (x - 1) ∈ ℝ :=
begin
  split,
  {
    -- The case for x ≠ 1 to show (x + 2) / (x - 1) is defined
    intro h,
    simp,
    exact div_ne_zero (by simp) (sub_ne_zero_of_ne h),
  },
  {
    -- The case for (x + 2) / (x - 1) being in ℝ to show x ≠ 1
    intro h,
    by_contradiction,
    simp at h,
    exact h,
  }
end

end fraction_meaningful_l115_115902


namespace tan_rewrite_l115_115322

open Real

theorem tan_rewrite (α β : ℝ) 
  (h1 : tan (α + β) = 2 / 5)
  (h2 : tan (β - π / 4) = 1 / 4) : 
  (1 + tan α) / (1 - tan α) = 3 / 22 := 
by
  sorry

end tan_rewrite_l115_115322


namespace average_speed_is_55_l115_115689

theorem average_speed_is_55 
  (initial_reading : ℕ) (final_reading : ℕ) (time_hours : ℕ)
  (H1 : initial_reading = 15951) 
  (H2 : final_reading = 16061)
  (H3 : time_hours = 2) : 
  (final_reading - initial_reading) / time_hours = 55 :=
by
  sorry

end average_speed_is_55_l115_115689


namespace equal_length_routes_l115_115830

-- Definitions based on conditions
variable {City : Type} -- The city type, which is a triangular grid of roads
variable (Point : Type) -- Points on the city's grid representing homes
variable (dist : Point → Point → ℕ) -- Distance function between two points on the grid

-- Kolya and Max's homes are represented by Point K and Point M
variable (K M : Point)

-- The route function modeling a person's travel with a given number of left turns
variable (route : Point → ℕ → list Point)

-- Kolya and Max's routes
variable (KolyaRoute MaxRoute : list Point)

-- Conditions 
axiom Kolya_makes_4_lefts : route K 4 = KolyaRoute
axiom Max_makes_1_left : route M 1 = MaxRoute
axiom routes_equal_length : dist K M = dist M K

-- Proving the routes are equal in length and consist of 8 unit segments
theorem equal_length_routes : dist K M = 8 ∧ dist M K = 8 :=
by {
  sorry,
}

end equal_length_routes_l115_115830


namespace max_non_managers_l115_115400

theorem max_non_managers (N : ℕ) : (8 / N : ℚ) > 7 / 32 → N ≤ 36 :=
by sorry

end max_non_managers_l115_115400


namespace max_successful_teams_l115_115795

theorem max_successful_teams (n : ℕ) (h₀ : n = 16)
    (points_for_win points_for_draw points_for_loss : ℕ)
    (h₁ : points_for_win = 3)
    (h₂ : points_for_draw = 1)
    (h₃ : points_for_loss = 0) :
  let games_each_team := n - 1,
      max_points_per_team := games_each_team * points_for_win,
      min_points_successful := (max_points_per_team + 2) / 2
  in max_successful_teams = 15 :=
by
  sorry

end max_successful_teams_l115_115795


namespace sufficient_but_not_necessary_condition_l115_115960

noncomputable def f (a x : ℝ) : ℝ := |x - a|

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x y : ℝ, -3 ≤ x → x ≤ y → y → f a x ≤ f a y) ↔ (a ≤ -3) :=
by
  sorry

end sufficient_but_not_necessary_condition_l115_115960


namespace slope_of_line_l115_115943

theorem slope_of_line {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (h : 5 / x + 4 / y = 0) :
  ∃ x₁ x₂ y₁ y₂, (5 / x₁ + 4 / y₁ = 0) ∧ (5 / x₂ + 4 / y₂ = 0) ∧ 
  (y₂ - y₁) / (x₂ - x₁) = -4 / 5 :=
sorry

end slope_of_line_l115_115943


namespace irrational_pi_l115_115257

theorem irrational_pi : ∀ (x : Real),
  x ∈ {π, Real.sqrt 4, Real.cbrt (-8), 3.1415926} →
  (π ∈ {x} ↔ ¬ (∃ (p q : ℤ), q ≠ 0 ∧ x = p / q)) :=
by
sorry

end irrational_pi_l115_115257


namespace proof_problem_l115_115103

noncomputable def Q (x : ℝ) : ℝ := x^2 - 5 * x - 4

theorem proof_problem :
  let p := 1 in
  let q := 1 in
  let r := 2 in
  let s := 1 in
  (3 : ℝ) ≤ x → x ≤ (10 : ℝ) →
  ⌊sqrt (Q x)⌋ = sqrt (Q (⌊x⌋ : ℝ)) →
  p + q + r + s = 5 :=
by
  sorry

end proof_problem_l115_115103


namespace chef_michel_total_pies_l115_115666

theorem chef_michel_total_pies
  (shepherd_pie_pieces : ℕ)
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ)
  (chicken_pot_pie_customers : ℕ)
  (H1 : shepherd_pie_pieces = 4)
  (H2 : chicken_pot_pie_pieces = 5)
  (H3 : shepherd_pie_customers = 52)
  (H4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) + (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by
  sorry

end chef_michel_total_pies_l115_115666


namespace part1_part2_l115_115336

def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y
def B (x y : ℝ) : ℝ := x^2 - x * y + x

def difference (x y : ℝ) : ℝ := A x y - 2 * B x y

theorem part1 : difference (-2) 3 = -20 :=
by
  -- Proving that difference (-2) 3 = -20
  sorry

theorem part2 (y : ℝ) : (∀ (x : ℝ), difference x y = 2 * y) → y = 2 / 5 :=
by
  -- Proving that if difference x y is independent of x, then y = 2 / 5
  sorry

end part1_part2_l115_115336


namespace largest_natural_divisible_power_l115_115216

theorem largest_natural_divisible_power (p q : ℤ) (hp : p % 5 = 0) (hq : q % 5 = 0) (hdiscr : p^2 - 4*q > 0) :
  ∀ (α β : ℂ), (α^2 + p*α + q = 0 ∧ β^2 + p*β + q = 0) → (α^100 + β^100) % 5^50 = 0 :=
sorry

end largest_natural_divisible_power_l115_115216


namespace square_root_and_quadratic_solution_l115_115351

theorem square_root_and_quadratic_solution
  (a b : ℤ)
  (h1 : 2 * a + b = 0)
  (h2 : 3 * b + 12 = 0) :
  (2 * a - 3 * b = 16) ∧ (a * x^2 + 4 * b - 2 = 0 → x^2 = 9) :=
by {
  -- Placeholder for proof
  sorry
}

end square_root_and_quadratic_solution_l115_115351


namespace age_difference_l115_115306

theorem age_difference (a1 a2 a3 a4 x y : ℕ) 
  (h1 : (a1 + a2 + a3 + a4 + x) / 5 = 28)
  (h2 : ((a1 + 1) + (a2 + 1) + (a3 + 1) + (a4 + 1) + y) / 5 = 30) : 
  y - (x + 1) = 5 := 
by
  sorry

end age_difference_l115_115306


namespace solve_inequality_l115_115482

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if h : a > -1 then { x : ℝ | -1 < x ∧ x < a }
  else if h : a < -1 then { x : ℝ | a < x ∧ x < -1 }
  else ∅

theorem solve_inequality (x a : ℝ) :
  (x^2 + (1 - a)*x - a < 0) ↔ (
    (a > -1 → x ∈ { x : ℝ | -1 < x ∧ x < a }) ∧
    (a < -1 → x ∈ { x : ℝ | a < x ∧ x < -1 }) ∧
    (a = -1 → False)
  ) :=
sorry

end solve_inequality_l115_115482


namespace n_gon_triangulable_l115_115137

theorem n_gon_triangulable (n : ℕ) (h : n ≥ 3) : ∃ (triangles : list (finset ℕ)),  
  (∀ t ∈ triangles, t.card = 3) ∧ 
  (∀ i j ∈ finset.range(n), i ≠ j → ∀ t₁ t₂ ∈ triangles, t₁ ≠ t₂ → disjoint t₁ t₂) ∧ 
  (∀ x ∈ finset.range(n), ∃ t ∈ triangles, x ∈ t) :=
by
  sorry

end n_gon_triangulable_l115_115137


namespace area_of_intersection_triangle_l115_115993

-- Definitions of points (vertices) and midpoints in a 3D space

structure Point3D :=
  (x y z : ℝ)
  
-- Given vertices
def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨6, 0, 0⟩
def C : Point3D := ⟨3, 3*Real.sqrt 3, 0⟩
def D : Point3D := ⟨3, Real.sqrt 3, 3 * Real.sqrt 2⟩

-- Given midpoints
def P : Point3D := ⟨3, 0, 0⟩
def Q : Point3D := ⟨9/2, 3*Real.sqrt 3/2, 0⟩
def R : Point3D := ⟨9/2, 3*Real.sqrt 3/2, 3*Real.sqrt 2/2⟩

-- The proof statement
theorem area_of_intersection_triangle :
  let PQ := Point3D.mk ((9/2) - 3) ((3*Real.sqrt 3/2) - 0) (0 - 0),
      PR := Point3D.mk ((9/2) - 3) ((3*Real.sqrt 3/2) - 0) ((3*Real.sqrt 2/2) - 0),
      n  := Point3D.mk (-9*Real.sqrt 6/4) 0 (9/4)
  in
  (1/2) * Real.sqrt (n.x^2 + n.y^2 + n.z^2) = 27*Real.sqrt 2/8 :=
by
  sorry

end area_of_intersection_triangle_l115_115993


namespace tangent_lines_to_circle_l115_115285

noncomputable def P := (4 : ℝ, 5 : ℝ)
noncomputable def k := 21 / 20
noncomputable def tangent1 := 21 * (x : ℝ) - 20 * (y : ℝ)
noncomputable def tangent2 := x = 4
noncomputable def circle := (x - 2) ^ 2 + (y : ℝ) ^ 2 = 4

theorem tangent_lines_to_circle :
  (tangent1 + 16 = 0) ∨ (tangent2) :=
by
  -- Proof is omitted
  sorry

end tangent_lines_to_circle_l115_115285


namespace length_of_rectangle_l115_115242

variable (Area Width : ℝ)
variable (Area_eq : Area = 35)
variable (Width_eq : Width = 5)

theorem length_of_rectangle : ∃ (Length : ℝ), Length = 7 ∧ Area = Length * Width := 
by
  use 7
  constructor
  -- Prove the length is 7
  { rfl }
  -- Prove that substituting the length 7 gives the correct area
  { rfl }

end length_of_rectangle_l115_115242


namespace find_g2_l115_115509

-- Given conditions:
variables (g : ℝ → ℝ) 
axiom cond1 : ∀ (x y : ℝ), x * g y = 2 * y * g x
axiom cond2 : g 10 = 5

-- Proof to show g(2) = 2
theorem find_g2 : g 2 = 2 := 
by
  -- Skipping the actual proof
  sorry

end find_g2_l115_115509


namespace cos_double_angle_parabola_l115_115055

theorem cos_double_angle_parabola (a : ℝ) (h_directrix : a = 1) :
  let x := -√3 in let y := a in
  let r := sqrt ((-√3)^2 + (a)^2) in
  let sin_theta := y / r in let cos_theta := x / r in
  cos 2 * atan2 y x = 1 / 2 :=
by
  sorry

end cos_double_angle_parabola_l115_115055


namespace largest_equal_cost_l115_115925

def is_even (n : ℕ) : Prop := n % 2 = 0

def option1_cost (n : ℕ) : ℕ :=
  n.digits 10 |> List.foldl (λ acc d => acc + if is_even d then 2 * d else d) 0

def option2_cost (n : ℕ) : ℕ :=
  n.digits 2 |> List.foldl (λ acc b => acc + if b = 1 then 2 else 1) 0

theorem largest_equal_cost : ∃ n, n < 500 ∧ option1_cost n = option2_cost n ∧ ∀ m, m < 500 ∧ option1_cost m = option2_cost m → m ≤ n :=
begin
  use 237,
  split,
  { exact 237 < 500, },
  split,
  { sorry, },  -- Proof of option1_cost 237 = option2_cost 237
  { intros m h1 h2,
    -- Proof that 237 is largest integer satisfying the conditions
    sorry,
  }
end

end largest_equal_cost_l115_115925


namespace longest_sequence_with_divisors_l115_115558

theorem longest_sequence_with_divisors :
  ∃ (seq : List ℕ), seq.length = 5 
  ∧ seq.head = 1 
  ∧ seq.last = 31 
  ∧ (∀ i j, i < j → seq.get i ∣ seq.get j)
  ∧ (∀ x y ∈ seq, x ≠ y) 
  ∧ (seq = [1, 2, 6, 30, 31]) := by
    sorry

end longest_sequence_with_divisors_l115_115558


namespace periodic_function_of_symmetry_l115_115510

variable {ℝ : Type}
variable (f : ℝ → ℝ)
variable (a b c d : ℝ)

theorem periodic_function_of_symmetry (h1 : ∀ x, f(x) = 2*b - f(2*a - x))
                                     (h2 : ∀ x, f(x) = 2*d - f(2*c - x))
                                     (h3 : a ≠ c) :
  ∃ T > 0, T = 2 * |a - c| ∧ ∀ x, f(x + T) = f(x) :=
by
  sorry

end periodic_function_of_symmetry_l115_115510


namespace quadratic_expression_min_value_l115_115740

noncomputable def min_value_quadratic_expression (x y z : ℝ) : ℝ :=
(x + 5) ^ 2 + (y - 1) ^ 2 + (z + 3) ^ 2

theorem quadratic_expression_min_value :
  ∃ x y z : ℝ, x - 2 * y + 2 * z = 5 ∧ min_value_quadratic_expression x y z = 36 :=
sorry

end quadratic_expression_min_value_l115_115740


namespace university_diploma_percentage_l115_115571

theorem university_diploma_percentage
  (A : ℝ) (B : ℝ) (C : ℝ)
  (hA : A = 0.40)
  (hB : B = 0.10)
  (hC : C = 0.15) :
  A - B + C * (1 - A) = 0.39 := 
sorry

end university_diploma_percentage_l115_115571


namespace hyperbola_eccentricity_l115_115030

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h_hyperbola : ∀ (x y : ℝ), (x, y) ∈ {(x, y) | x^2 / a^2 - y^2 / b^2 = 1})
  (C : ℝ × ℝ) (hC : C = (0, real.sqrt 2 * b))
  (h_bisector : ∀ A B : ℝ × ℝ, is_perpendicular_bisector_A_of_AC_Passing_Through_B A C B (0, real.sqrt 2 * b)) : 
  eccentricity (a b) = real.sqrt 10 / 2 :=
by
  sorry

end hyperbola_eccentricity_l115_115030


namespace green_percentage_in_smaller_pond_l115_115800

def percentageGreenInSmallerPond (x y z w totalGreen: ℕ) (p q r: ℚ) : Prop :=
  x + y = z ∧
  p * y = totalGreen ∧
  q * z = w ∧
  p * 100 = 40 ∧
  q * 100 = 31 →
  (x / 45) * 100 = r

theorem green_percentage_in_smaller_pond :
  ∀ (smallerPond totalDucks largerPond totalGreen ducksInLarger greenInLargerPond greenPercentageTotal percentageGreen: ℕ),
    percentageGreenInSmallerPond smallerPond largerPond totalDucks ducksInLarger greenInLargerPond (40 / 100) (31 / 100) totalGreen →
    percentageGreen = 20 :=
begin
  intros smallerPond totalDucks largerPond totalGreen ducksInLarger greenInLargerPond greenPercentageTotal percentageGreen,
  intro h,
  sorry,
end

end green_percentage_in_smaller_pond_l115_115800


namespace find_side_b_l115_115729

noncomputable theory

-- Define the triangle structure
structure Triangle :=
(A B C : ℝ) -- angles in radians

-- Conditions given in the problem
variables {A B C : ℝ}
variables {a b c : ℝ}

-- Define the conditions
def acute_angle (A : ℝ) := 0 < A ∧ A < π / 2
def condition1 (A : ℝ) := 23 * (Real.cos A)^2 + Real.cos (2 * A) = 0
def side_a := a = 7
def side_c := c = 6

-- The main theorem to be proven
theorem find_side_b (A : ℝ) (a b c : ℝ) 
  (hAcute : acute_angle A) 
  (hCond1 : condition1 A) 
  (hSideA : side_a) 
  (hSideC : side_c) : 
  b = 5 := 
sorry

end find_side_b_l115_115729


namespace serenity_shoes_l115_115878

theorem serenity_shoes (total_shoes shoes_per_pair : ℕ) (h1 : total_shoes = 6) (h2 : shoes_per_pair = 2) :
  (total_shoes / shoes_per_pair = 3) :=
by 
  rw [h1, h2]
  norm_num

end serenity_shoes_l115_115878


namespace combined_area_of_removed_triangles_l115_115998

-- Define side lengths of original and smaller square, and the type of triangles removed
variables (a s x : ℝ)
def original_square_side := 20
def smaller_square_side := 10

-- Define the condition that each triangle is an isosceles right triangle and forms the difference in sides
def isosceles_right_triangle_condition (x : ℝ) : Prop :=
  x^2 + x^2 = (original_square_side - smaller_square_side)^2

-- Define the area of a single triangle and the total area
def single_triangle_area (x : ℝ) :=
  1 / 2 * x^2

def total_removed_area (x : ℝ) :=
  4 * single_triangle_area x

-- The theorem to be proven
theorem combined_area_of_removed_triangles :
  ∃ x, isosceles_right_triangle_condition x ∧ total_removed_area x = 100 := sorry

end combined_area_of_removed_triangles_l115_115998


namespace geometric_sequence_sum_l115_115757

theorem geometric_sequence_sum : 
  ∀ {a : ℕ → ℂ} (q : ℂ),
    a 3 = 4 ∧ (∑ i in range 6, a i) / (∑ i in range 3, a i) = 9 →
    (a 1 + (a 2 / 2) + a 3 + (a 4 / 2) + a 5 + (a 6 / 2) + ⋯ + a 19 + (a 20 / 2)) = 
    (2 * (4^10 - 1)) / 3 :=
by
  sorry

end geometric_sequence_sum_l115_115757


namespace probability_of_two_girls_one_senior_one_junior_l115_115607

-- Defining the problem context
def total_members := 12
def boys := 6
def girls := 6
def seniors (gender : bool) := if gender then 3 else 3  -- True for boys, False for girls
def juniors (gender : bool) := if gender then 3 else 3  -- True for boys, False for girls

-- Calculate total ways to choose 2 members out of 12
def total_ways_to_choose_two : ℕ := Nat.choose total_members 2

-- Calculate ways to choose one senior girl and one junior girl
def ways_to_choose_senior_girl : ℕ := Nat.choose (seniors false) 1
def ways_to_choose_junior_girl : ℕ := Nat.choose (juniors false) 1
def favorable_ways : ℕ := ways_to_choose_senior_girl * ways_to_choose_junior_girl

-- Prove probability
def probability := (favorable_ways : ℚ) / total_ways_to_choose_two

theorem probability_of_two_girls_one_senior_one_junior : 
  probability = 9 / 66 := by
  sorry

end probability_of_two_girls_one_senior_one_junior_l115_115607


namespace number_of_common_points_l115_115517

variable (f : ℝ → ℝ)

theorem number_of_common_points (h : ¬ (Function.has_left_inv f x = 1) ∨ Function.has_left_inv f x = 1) :
  ∃ n, n = 0 ∨ n = 1 :=
by
  sorry

end number_of_common_points_l115_115517


namespace part1_part2_part3_l115_115326

def f (a x : ℝ) : ℝ := log a x + a * x + 1 / (x + 1)

-- Part (1)
theorem part1 (h : 2 = 2) : f 2 (1 / 4) = -7 / 10 := 
  sorry

-- Part (2)
theorem part2 (a : ℝ) (ha : 1 < a) : ∃! x : ℝ, 0 < x ∧ f a x = 0 := 
  sorry

-- Part (3)
theorem part3 (a x0 : ℝ) (ha : 1 < a) (hfx0 : f a x0 = 0) : 
  1 / 2 < f a (sqrt x0) ∧ f a (sqrt x0) < (a + 1) / 2 := 
  sorry

end part1_part2_part3_l115_115326


namespace smallest_number_is_45_l115_115955

theorem smallest_number_is_45 (x : ℝ) (h1 : 2 * x = y) (h2 : 4 * y = z)
  (h3 : (x + y + z) / 3 = 165) : x = 45 :=
by
  have y_def : y = 2 * x := h1
  have z_def : z = 4 * y := h2
  have avg_def : (x + y + z) / 3 = 165 := h3
  sorry

end smallest_number_is_45_l115_115955


namespace third_term_binomial_coefficient_l115_115894

theorem third_term_binomial_coefficient :
  (∃ m : ℕ, m = 4 ∧ ∃ k : ℕ, k = 2 ∧ Nat.choose m k = 6) :=
by
  sorry

end third_term_binomial_coefficient_l115_115894


namespace builder_needs_boards_l115_115966

theorem builder_needs_boards (packages : ℕ) (boards_per_package : ℕ) (total_boards : ℕ)
  (h1 : packages = 52)
  (h2 : boards_per_package = 3)
  (h3 : total_boards = packages * boards_per_package) : 
  total_boards = 156 :=
by
  rw [h1, h2] at h3
  exact h3

end builder_needs_boards_l115_115966


namespace product_of_roots_l115_115303

theorem product_of_roots :
  let a := 18
  let b := 45
  let c := -500
  let prod_roots := c / a
  prod_roots = -250 / 9 := 
by
  -- Define coefficients
  let a := 18
  let c := -500

  -- Calculate product of roots
  let prod_roots := c / a

  -- Statement to prove
  have : prod_roots = -250 / 9 := sorry
  exact this

-- Adding sorry since the proof is not required according to the problem statement.

end product_of_roots_l115_115303


namespace master_liu_problems_l115_115490

-- Define the distances traveled
def distances : List Int := [-3, -15, 19, -1, 5, -12, -6, 12]

-- Define passenger status
def passenger : List Bool := [false, true, true, false, true, true, true, true]

-- Define the total distance calculation
def total_distance (distances : List Int) : Int :=
  distances.sum

-- Define the absolute distance calculation for fuel consumption
def total_distance_abs (distances : List Int) : Int :=
  distances.map (Int.abs).sum

-- Define the fuel consumption rate
def fuel_consumption_rate : Float := 0.06

-- Define the initial fuel in tank
def initial_fuel : Float := 7.0

-- Define the total fuel consumed
def total_fuel_consumed (abs_distance : Int) : Float :=
  fuel_consumption_rate * abs_distance.toFloat

-- Define the remaining fuel
def remaining_fuel (initial : Float) (consumed : Float) : Float :=
  initial - consumed

-- Define fare calculation for a single trip
def fare (distance : Int) : Float :=
  if distance <= 2 then 10.0
  else 10.0 + (distance - 2).toFloat * 1.6

-- Define the total revenue calculation
def total_revenue (distances : List Int) (passenger : List Bool) : Float :=
  (distances.zip passenger).filter (λ p => p.2).map (λ p => fare p.1).sum

-- Main problem statement
theorem master_liu_problems :
  total_distance distances = -1 ∧
  let abs_dist := total_distance_abs distances in
  remaining_fuel initial_fuel (total_fuel_consumed abs_dist) = 2.62 ∧
  remaining_fuel initial_fuel (total_fuel_consumed abs_dist) > 2.0 ∧
  total_revenue distances passenger = 151.2 := 
by
  sorry

end master_liu_problems_l115_115490


namespace root_multiplicity_at_most_two_number_of_values_of_c_with_double_root_eq_1005_l115_115834

noncomputable def f (x : ℝ) (c : ℝ) := (Finset.range 2010).prod (λ k, x + k) - c
noncomputable def f' (x : ℝ) := (Finset.range 2010).prod (λ k, x + k) * (Finset.range 2010).sum (λ k, 1 / (x + k))
noncomputable def f'' (x : ℝ) := f' x * (Finset.range 2010).sum (λ k, 1 / (x + k)) - 
                              (Finset.range 2010).prod (λ k, x + k) * (Finset.range 2010).sum (λ k, 1 / ((x + k) ^ 2))

theorem root_multiplicity_at_most_two (c : ℝ) :
  ∀ r : ℝ, f r c = 0 → f' r = 0 → f'' r ≠ 0 :=
sorry

theorem number_of_values_of_c_with_double_root_eq_1005 :
  (Finset.range 2010).card / 2 = 1005 :=
sorry

end root_multiplicity_at_most_two_number_of_values_of_c_with_double_root_eq_1005_l115_115834


namespace perp_O₁O₂_to_BC_l115_115733

variables (A B C D O₁ O₂ M K : Point)
variables (h_isosceles_trapezoid : is_isosceles_trapezoid ABCD B C D A)
variables (h_base_parallel : is_parallel B C D A)
variables (h_incirc1 : is_incircle O₁ A B C)
variables (h_incirc2 : is_incircle O₂ A B D)
variables (M_contact : contact_point O₁ A B C M B C)
variables (K_contact : contact_point O₂ A B D K D A)

theorem perp_O₁O₂_to_BC :
  perp (line (O₁) O₂) (line B C) :=
sorry

end perp_O₁O₂_to_BC_l115_115733


namespace number_of_vipers_l115_115801

theorem number_of_vipers (crocodiles alligators total_dangerous : ℕ) : 
  crocodiles = 22 → 
  alligators = 23 → 
  total_dangerous = 50 → 
  (total_dangerous - (crocodiles + alligators)) = 5 :=
by
  intros hcroc hallig htotal
  rw [hcroc, hallig, htotal]
  sorry

end number_of_vipers_l115_115801


namespace probability_of_exact_r_successes_is_correct_l115_115746

open Probability

noncomputable def probability_of_exact_r_successes (n r : ℕ) (p : ℝ) (h1 : 0 <= r ∧ r <= n) (h2 : 0 <= p ∧ p <= 1) : ℝ := 
  (nat.choose (n-1) (r-1)) * (p^r) * ((1 - p)^(n - r))

theorem probability_of_exact_r_successes_is_correct (n r : ℕ) (p : ℝ) (h1 : 0 <= r ∧ r <= n) (h2 : 0 <= p ∧ p <= 1) :
  probability_of_exact_r_successes n r p h1 h2 = (nat.choose (n-1) (r-1)) * (p^r) * ((1 - p)^(n - r)) :=
sorry

end probability_of_exact_r_successes_is_correct_l115_115746


namespace distance_between_parallel_lines_l115_115899

theorem distance_between_parallel_lines :
  let line1 := (3, -sqrt(7), 2)
  let line2 := (6, -2 * sqrt(7), 3)
  -- Simplify line2 by dividing by 2 for easier comparison
  let line2_simplified := (3, -sqrt(7), 3 / 2)
  let distance_formula := λ (a b c1 c2 : ℝ), abs (c1 - c2) / sqrt (a^2 + b^2)
  distance_formula line1.1 line1.2 line1.3 line2_simplified.3 = 1 / 8 :=
by {
  sorry
}

end distance_between_parallel_lines_l115_115899


namespace volume_equal_to_base_area_times_height_is_false_l115_115920

-- Definitions for the volumes based on given conditions
def volume_cube (a : ℝ) : ℝ := a * a * a
def volume_prism (l w h : ℝ) : ℝ := l * w * h
def volume_cone (b h : ℝ) : ℝ := (1 / 3) * b * h

-- The main statement to prove
theorem volume_equal_to_base_area_times_height_is_false 
  (a l w h b : ℝ) : 
  ¬ (∀ h, volume_cube a = a * a * h ∧
            volume_prism l w h = l * w * h ∧
            volume_cone b h = b * h) := 
  sorry -- Proof not required

end volume_equal_to_base_area_times_height_is_false_l115_115920


namespace shaded_triangle_ratio_is_correct_l115_115982

noncomputable def ratio_of_shaded_triangle_to_large_square (total_area : ℝ) 
  (midpoint_area_ratio : ℝ := 1 / 24) : ℝ :=
  midpoint_area_ratio * total_area

theorem shaded_triangle_ratio_is_correct 
  (shaded_area total_area : ℝ)
  (n : ℕ)
  (h1 : n = 36)
  (grid_area : ℝ)
  (condition1 : grid_area = total_area / n)
  (condition2 : shaded_area = grid_area / 2 * 3)
  : shaded_area / total_area = 1 / 24 :=
by
  sorry

end shaded_triangle_ratio_is_correct_l115_115982


namespace least_possible_perimeter_minimal_perimeter_triangle_l115_115183

theorem least_possible_perimeter (x : ℕ) 
  (h1 : 13 < x) 
  (h2 : x < 61) : 
  24 + 37 + x ≥ 24 + 37 + 14 := 
sorry

theorem minimal_perimeter_triangle : ∃ x : ℕ, 13 < x ∧ x < 61 ∧ 24 + 37 + x = 75 :=
begin
  existsi 14,
  split,
  { exact dec_trivial, }, -- 13 < 14
  split,
  { exact dec_trivial, }, -- 14 < 61
  { exact dec_trivial, }, -- 24 + 37 + 14 = 75
end

end least_possible_perimeter_minimal_perimeter_triangle_l115_115183


namespace range_of_a_l115_115785

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, a * x^2 + 2 * a * x + 2 < 0) ↔ 0 ≤ a ∧ a ≤ 2 := 
by 
  sorry

end range_of_a_l115_115785


namespace rectangle_remainder_condition_l115_115431

theorem rectangle_remainder_condition
    (n a b : ℕ) (hn : 2 ≤ n)
    (ha : 1 ≤ a) (hb : 1 ≤ b) :
    (n ∣ (a - 1) ∨ n ∣ (b - 1)) ∧ (n ∣ (a + 1) ∨ n ∣ (b + 1)) :=
sorry

end rectangle_remainder_condition_l115_115431


namespace ratio_of_cube_sides_l115_115514

theorem ratio_of_cube_sides 
  (a b : ℝ) 
  (h : (6 * a^2) / (6 * b^2) = 49) :
  a / b = 7 :=
by
  sorry

end ratio_of_cube_sides_l115_115514


namespace multiplication_result_l115_115204

theorem multiplication_result :
  3^2 * 5^2 * 7 * 11^2 = 190575 :=
by sorry

end multiplication_result_l115_115204


namespace smallest_possible_S_l115_115946

theorem smallest_possible_S (n : ℕ) (hn : 6 * n ≥ 2050) :
  ∃ S, (∀ (d : ℕ → ℕ), (∀ i, 1 ≤ d i ∧ d i ≤ 6) → (∑ i in Finset.range n, d i = 2050) → (S = 344 ∧ (∑ i in Finset.range n, (7 - d i)) = S)) :=
sorry

end smallest_possible_S_l115_115946


namespace greatest_product_slopes_l115_115551

theorem greatest_product_slopes (θ : ℝ) (m₁ m₂ : ℝ) (h1 : θ = π / 6) (h2 : m₂ = 4 * m₁)
    (h_angle : |(m₂ - m₁) / (1 + m₁ * m₂)| = tan θ) :
    (m₁ * m₂) ≤ (38 + 6 * Real.sqrt 33) / 16 :=
by
  sorry

end greatest_product_slopes_l115_115551


namespace tangent_line_eq_l115_115358

noncomputable def curve := (x : ℝ) → -x^2 + 4
def pt : ℝ × ℝ := (-1, curve (-1))

def tangent_at (f : ℝ → ℝ) (x₀ : ℝ) : ℝ → ℝ :=
  let k := deriv f x₀ in
  fun x => k * (x - x₀) + f x₀

theorem tangent_line_eq :
  tangent_at curve (-1) = (λ x, 2 * x + 5) :=
by
  sorry

end tangent_line_eq_l115_115358


namespace collinear_E_F_G_l115_115335

variable (A B C D G E F : Type) [Trapezoid A B C D] (AD_parallel_BC : AD ∥ BC)
variable (AB_intersect_DC_G : ∃ G, ∃ AB DC, G ∈ AB ∧ G ∈ DC)
variable (common_tangents_E : ∃ E, Tangent (CircumscribedCircle (Triangle ABC)) (CircumscribedCircle (Triangle ACD)) E)
variable (common_tangents_F : ∃ F, Tangent (CircumscribedCircle (Triangle ABD)) (CircumscribedCircle (Triangle BCD)) F)

theorem collinear_E_F_G : collinear E F G := sorry

end collinear_E_F_G_l115_115335


namespace correct_statements_l115_115968

def distances_A : List ℝ := [94, 96, 99, 99, 105, 107]
def distances_B : List ℝ := [95, 95, 98, 99, 104, 109]

def mode (xs : List ℝ) : ℝ := xs.foldr (λ x acc, if xs.countp (= x) > xs.countp (= acc) then x else acc) xs.head

def range (xs : List ℝ) : ℝ := xs.maximumD - xs.minimumD

def mean (xs : List ℝ) : ℝ := xs.sum / xs.length

def variance (xs : List ℝ) : ℝ := xs.foldr (λ x acc, acc + (x - mean xs)^2) 0 / xs.length

theorem correct_statements : 
  mode distances_A > mode distances_B ∧
  range distances_A < range distances_B ∧
  mean distances_A = mean distances_B ∧
  variance distances_A < variance distances_B :=
by sorry

end correct_statements_l115_115968


namespace connectivity_within_square_l115_115215

theorem connectivity_within_square (side_length : ℝ) (highway1 highway2 : ℝ) 
  (A1 A2 A3 A4 : ℝ → ℝ → Prop) : 
  side_length = 10 → 
  highway1 ≠ highway2 → 
  (∀ x y, (0 ≤ x ∧ x ≤ side_length ∧ 0 ≤ y ∧ y ≤ side_length) → 
    (A1 x y ∨ A2 x y ∨ A3 x y ∨ A4 x y)) →
  ∃ (road_length : ℝ), road_length ≤ 25 := 
sorry

end connectivity_within_square_l115_115215


namespace cosine_alpha_l115_115347

-- Define the necessary conditions.
variables {α β : ℝ}
variables (h1 : α > 0 ∧ α < π / 2) -- acute angle condition for α
variables (h2 : β > 0 ∧ β < π / 2) -- acute angle condition for β
variables (h3 : cos (α + β) = -3/5) -- given condition
variables (h4 : sin β = 12/13)  -- given condition

-- The theorem statement to be proven.
theorem cosine_alpha : cos α = 33/65 :=
by
  sorry

end cosine_alpha_l115_115347


namespace meaning_of_probability_l115_115063

-- Definitions

def probability_of_winning (p : ℚ) : Prop :=
  p = 1 / 4

-- Theorem statement
theorem meaning_of_probability :
  probability_of_winning (1 / 4) →
  ∀ n : ℕ, (n ≠ 0) → (n / 4 * 4) = n :=
by
  -- Placeholder proof
  sorry

end meaning_of_probability_l115_115063


namespace reach_one_in_seven_steps_l115_115884

noncomputable def process (n : ℕ) : ℕ :=
  let x := n / 3 + 2
  if x % 2 = 1 then x - 1 else x

def steps_to_reach_leq_one (start : ℕ) : ℕ :=
  Nat.iterate process start 7

theorem reach_one_in_seven_steps :
  steps_to_reach_leq_one 150 <= 1 :=
by
  sorry

end reach_one_in_seven_steps_l115_115884


namespace roots_cube_reciprocal_eqn_l115_115329

variable (a b c r s : ℝ)

def quadratic_eqn (r s : ℝ) : Prop :=
  3 * a * r ^ 2 + 5 * b * r + 7 * c = 0 ∧ 
  3 * a * s ^ 2 + 5 * b * s + 7 * c = 0

theorem roots_cube_reciprocal_eqn (h : quadratic_eqn a b c r s) :
  (1 / r^3 + 1 / s^3) = (-5 * b * (25 * b ^ 2 - 63 * c) / (343 * c^3)) :=
sorry

end roots_cube_reciprocal_eqn_l115_115329


namespace flea_reach_B_with_7_jumps_flea_reach_C_with_9_jumps_flea_cannot_reach_D_with_2028_jumps_l115_115213

-- Problem (a)
theorem flea_reach_B_with_7_jumps (A B : ℤ) (jumps : ℤ) (distance : ℤ) (ways : ℕ) :
  B = A + 5 → jumps = 7 → distance = 5 → 
  ways = Nat.choose (7) (1) := 
sorry

-- Problem (b)
theorem flea_reach_C_with_9_jumps (A C : ℤ) (jumps : ℤ) (distance : ℤ) (ways : ℕ) :
  C = A + 5 → jumps = 9 → distance = 5 → 
  ways = Nat.choose (9) (2) :=
sorry

-- Problem (c)
theorem flea_cannot_reach_D_with_2028_jumps (A D : ℤ) (jumps : ℤ) (distance : ℤ) :
  D = A + 2013 → jumps = 2028 → distance = 2013 → 
  ∃ x y : ℤ, x + y = 2028 ∧ x - y = 2013 → false :=
sorry

end flea_reach_B_with_7_jumps_flea_reach_C_with_9_jumps_flea_cannot_reach_D_with_2028_jumps_l115_115213


namespace bug_traverse_impossible_l115_115418

-- Define the 3x3x3 Cube and relevant properties
structure Cube3x3x3 :=
  (unit_cubes : Fin 27)

-- Specific bug starting position (center of the cube)
def center_position := 13  -- The index of the central cube in a 3x3x3 grid

-- Define the traversal problem in Lean 4
theorem bug_traverse_impossible (bug_position : Fin 27) (initial_position : bug_position = center_position) :
  ¬ ∃ f : Fin 27 → Fin 27, (∀ i : Fin 27, f i ≠ i) ∧
  (∀ i j : Fin 27, (i ≠ j → f i ≠ f j) ∧ (f i ∈ adjacent i) ∧ (f.center_position = bug_position)) := sorry

-- Adjacency relations in the 3x3x3 cube can be defined separately
def adjacent (i : Fin 27) : Set (Fin 27) := sorry  -- Define adjacency logic for the cubes

end bug_traverse_impossible_l115_115418


namespace solve_quadratics_l115_115882

theorem solve_quadratics (x : ℝ) :
  (x^2 - 7 * x - 18 = 0 → x = 9 ∨ x = -2) ∧
  (4 * x^2 + 1 = 4 * x → x = 1/2) :=
by
  sorry

end solve_quadratics_l115_115882


namespace problem_I_problem_II_problem_III_l115_115346

-- Problem (I) setup and proof requirement.
theorem problem_I (t : ℝ) (h_t : t = 4 * (Real.sqrt 3)) :
  (∃ k : ℝ, (t = 4 * (Real.sqrt 3) ∧ ∀ x y : ℝ, (x = 0 ∨ (x + Real.sqrt 3 * y - 12 = 0)) ∧ (x - 4)^2 + y^2 = 16) → true) := 
sorry

-- Problem (II) setup and proof requirement.
theorem problem_II (t : ℝ) (h_t : t > 0) :
  (∃ x y : ℝ, (x-1)^2 + (y - t / 2)^2 = t^2 / 4 + 1) ∧ 
  (3 * x - 4 * y - 5 = 0 ∧ (x,y) = (1, 2) ∧ ∀ x y : ℝ, (x-1)^2 + (y-2)^2 = 5) := 
sorry

-- Problem (III) setup and proof requirement.
theorem problem_III (t : ℝ) (h_t : t > 0) :
  ∀ x y : ℝ, (x = 2 ∧ y = t) → 
  ∀ (OM : ℝ → ℝ) (A : ℝ → ℝ) (N : ℝ → ℝ) (H : ℝ → ℝ), 
  OM = (λ s, sqrt ((2 : ℝ) + y^2)) ∧ A = (λ _, 1) ∧ N = (λ _, 0) ∧ 
  H = (λ s, 2 / Real.sqrt (4 + y^2) ∧ ON = sqrt 2) := 
sorry

end problem_I_problem_II_problem_III_l115_115346


namespace tan_product_identity_l115_115046

theorem tan_product_identity :
  let A := 30 * (Real.pi / 180)
  let B := 60 * (Real.pi / 180)
  (1 + Real.tan A) * (1 + Real.tan B) = 2 + 4 * Real.sqrt 3 / 3 := by
  let A := (30:ℝ) * (Real.pi / 180)
  let B := (60:ℝ) * (Real.pi / 180)
  have tan_A_val : Real.tan A = 1 / Real.sqrt 3 := by sorry
  have tan_B_val : Real.tan B = Real.sqrt 3 := by sorry
  calc
    (1 + Real.tan A) * (1 + Real.tan B)
      = (1 + 1 / Real.sqrt 3) * (1 + Real.sqrt 3) : by rw [tan_A_val, tan_B_val]
      -- substitute and simplify
      ... = 2 + 4 * Real.sqrt 3 / 3 : by sorry

end tan_product_identity_l115_115046


namespace businessmen_drink_one_type_l115_115655

def total_businessmen : ℕ := 35
def coffee_drinkers : ℕ := 18
def tea_drinkers : ℕ := 15
def juice_drinkers : ℕ := 8
def coffee_and_tea_drinkers : ℕ := 6
def tea_and_juice_drinkers : ℕ := 4
def coffee_and_juice_drinkers : ℕ := 3
def all_three_drinkers : ℕ := 2

theorem businessmen_drink_one_type : 
  coffee_drinkers - coffee_and_tea_drinkers - coffee_and_juice_drinkers + all_three_drinkers +
  tea_drinkers - coffee_and_tea_drinkers - tea_and_juice_drinkers + all_three_drinkers +
  juice_drinkers - tea_and_juice_drinkers - coffee_and_juice_drinkers + all_three_drinkers = 21 := 
sorry

end businessmen_drink_one_type_l115_115655


namespace solve_for_x_l115_115479

theorem solve_for_x : ∃ x : ℝ, 2 * (5 ^ x) = 1250 ∧ x = 4 :=
by
  use 4
  split
  . sorry /* This part of the proof is omitted */
  . refl /* This part of the proof shows the equality x=4 */

end solve_for_x_l115_115479


namespace find_m_l115_115417

noncomputable def slope_at_one (m : ℝ) := 2 + m

noncomputable def tangent_line_eq (m : ℝ) (x : ℝ) := (slope_at_one m) * x - 2 * m

noncomputable def y_intercept (m : ℝ) := tangent_line_eq m 0

noncomputable def x_intercept (m : ℝ) := - (y_intercept m) / (slope_at_one m)

noncomputable def intercept_sum_eq (m : ℝ) := (x_intercept m) + (y_intercept m)

theorem find_m (m : ℝ) (h : m ≠ -2) (h2 : intercept_sum_eq m = 12) : m = -3 ∨ m = -4 := 
sorry

end find_m_l115_115417


namespace original_number_l115_115238

theorem original_number (x : ℝ) (h : x * 1.5 = 105) : x = 70 :=
sorry

end original_number_l115_115238


namespace isosceles_triangle_count_l115_115770

def is_isosceles (a b : ℕ) : Prop :=
  2 * a + b = 25 ∧ a ≥ 2 ∧ b ≥ 2 ∧ (a > b / 2) ∧ (2 * a > b)

theorem isosceles_triangle_count : 
  (finset.filter (λ p : ℕ × ℕ, is_isosceles p.1 p.2) 
    (finset.Icc (2, 2) (24, 21))).card = 3 := 
sorry

end isosceles_triangle_count_l115_115770


namespace factor_x_minus_1_l115_115114

theorem factor_x_minus_1 (P Q R S : Polynomial ℂ) : 
  (P.eval 1 = 0) → 
  (P.eval (x^5) + x * Q.eval (x^5) + x^2 * R.eval (x^5) 
  = (x^4 + x^3 + x^2 + x + 1) * S.eval (x)) :=
sorry

end factor_x_minus_1_l115_115114


namespace papers_left_l115_115665

def total_papers_bought : ℕ := 20
def pictures_drawn_today : ℕ := 6
def pictures_drawn_yesterday_before_work : ℕ := 6
def pictures_drawn_yesterday_after_work : ℕ := 6

theorem papers_left :
  total_papers_bought - (pictures_drawn_today + pictures_drawn_yesterday_before_work + pictures_drawn_yesterday_after_work) = 2 := 
by 
  sorry

end papers_left_l115_115665


namespace ratio_of_areas_l115_115448

theorem ratio_of_areas (x : ℝ) : 
  let AB := x,
      BC := x,
      CA := x,
      BB' := 4 * AB,
      CC' := 4 * BC,
      AA' := 4 * CA,
      AB' := AB + BB',
      BC' := BC + CC',
      CA' := CA + AA',
      area_ABC := (Real.sqrt 3 / 4) * x^2,
      area_A'B'C' := (Real.sqrt 3 / 4) * (5 * x)^2
  in area_A'B'C' / area_ABC = 25 := 
by
  sorry

end ratio_of_areas_l115_115448


namespace verify_geometric_figures_l115_115405

-- Conditions:
-- A cuboid with a square base
variables (cuboid : Type) [is_cuboid cuboid] [has_square_base cuboid]

-- Question: The vertices of the cuboid can form:
def can_form_geometric_figures : Prop :=
  (∃ (v1 v2 v3 v4 : cuboid.vertex), 
     -- ① Rectangle
     is_rectangle v1 v2 v3 v4 ∨
     -- ③ A tetrahedron with three faces being right-angled triangles and one face being an isosceles triangle
     is_tetrahedron v1 v2 v3 v4 ∧
     has_three_right_angled_faces v1 v2 v3 v4 ∧
     has_one_isosceles_face v1 v2 v3 v4 ∨
     -- ④ A tetrahedron with each face being an isosceles triangle
     is_tetrahedron v1 v2 v3 v4 ∧
     has_each_face_isosceles v1 v2 v3 v4 ∨
     -- ⑤ A tetrahedron with each face being a right-angled triangle
     is_tetrahedron v1 v2 v3 v4 ∧
     has_each_face_right_angled v1 v2 v3 v4)

-- Theorem statement
theorem verify_geometric_figures : can_form_geometric_figures cuboid :=
sorry

end verify_geometric_figures_l115_115405


namespace top_right_rectangle_is_R_l115_115690

-- Define the set of corner values for each rectangle.
structure Rectangle :=
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (d : ℕ)

constant P : Rectangle
constant Q : Rectangle
constant R : Rectangle
constant S : Rectangle

-- Given the conditions as properties for each rectangle
axiom P_def : P.a = 5 ∧ P.b = 1 ∧ P.c = 8 ∧ P.d = 2
axiom Q_def : Q.a = 2 ∧ Q.b = 8 ∧ Q.c = 10 ∧ Q.d = 4
axiom R_def : R.a = 4 ∧ R.b = 5 ∧ R.c = 1 ∧ R.d = 7
axiom S_def : S.a = 8 ∧ S.b = 3 ∧ S.c = 7 ∧ S.d = 5

-- The statement to prove
theorem top_right_rectangle_is_R : (P.a = 5 ∧ P.b = 1 ∧ P.c = 8 ∧ P.d = 2) →
  (Q.a = 2 ∧ Q.b = 8 ∧ Q.c = 10 ∧ Q.d = 4) →
  (R.a = 4 ∧ R.b = 5 ∧ R.c = 1 ∧ R.d = 7) →
  (S.a = 8 ∧ S.b = 3 ∧ S.c = 7 ∧ S.d = 5) →
  (top_right_rectangle = R) :=
by
  sorry

end top_right_rectangle_is_R_l115_115690


namespace number_of_equilateral_triangles_in_cube_l115_115518

-- Definition of the cube's structure
structure Cube :=
  (vertices : Fin 8) -- 8 vertices
  (edges : Fin 12) -- 12 edges
  (faces : Fin 6) -- 6 faces

-- Definition of an equilateral triangle in a cube
def is_equilateral_triangle (triangle : List (Fin 8)) : Prop :=
  ∃ (v1 v2 v3 : Fin 8),
    (triangle = [v1, v2, v3]) ∧
    (distance v1 v2 = distance v2 v3 ∧ distance v2 v3 = distance v3 v1)

noncomputable def num_equilateral_triangles (cube : Cube) : ℕ :=
  -- The number of equilateral triangles that can be found in the cube.
  sorry

theorem number_of_equilateral_triangles_in_cube (cube : Cube) : num_equilateral_triangles cube = 8 := by
  sorry

end number_of_equilateral_triangles_in_cube_l115_115518


namespace smallest_common_multiple_l115_115944

theorem smallest_common_multiple (n : ℕ) (h8 : n % 8 = 0) (h15 : n % 15 = 0) : n = 120 :=
sorry

end smallest_common_multiple_l115_115944


namespace find_x_l115_115954

theorem find_x (x y : ℕ) 
  (h1 : 3^x * 4^y = 59049) 
  (h2 : x - y = 10) : 
  x = 10 := 
by 
  sorry

end find_x_l115_115954


namespace prime_ge_7_divisibility_l115_115578

theorem prime_ge_7_divisibility (p : ℕ) (hp : prime p) (h7 : p ≥ 7) : 
  (∃ m, 40 * m = p^2 - 1) ∨ (¬∃ m, 40 * m = p^2 - 1) := 
sorry

end prime_ge_7_divisibility_l115_115578


namespace find_constant_a_l115_115747

theorem find_constant_a (n : ℕ) (a : ℝ)
  (h₁ : ∀ n, (∑ i in finset.range (n + 1), (a - 2) * 3^(i) + 2) = (a - 2) * 3^(n + 1) + 2)
  (h₂ : ∀ n, (∑ i in finset.range n, (a - 2) * 3^(i) + 2) = (a - 2) * 3^(n) + 2)
  (h₃ : (∀ n > 0, (∑ i in finset.range (n + 1), (a - 2) * 3^(i) + 2) - (∑ i in finset.range n, (a - 2) * 3^(i) + 2) = 2 * (a - 2) * 3^n))
  (h₄ : (∀ n > 0, (a - 2) * 3^(2) + 2 = 6 * (a - 2))) :
  a = 4/3 :=
by
  sorry

end find_constant_a_l115_115747


namespace simple_random_sampling_possible_l115_115793

theorem simple_random_sampling_possible
  (total_males : ℕ) (total_females : ℕ)
  (sample_males : ℕ) (sample_females : ℕ)
  (total_students : total_males + total_females = 50)
  (sample_students : sample_males + sample_females = 10) :
  total_males = 20 ∧ total_females = 30 ∧ sample_males = 4 ∧ sample_females = 6 →
  (sample_males * total_females = sample_females * total_males) :=
by
  intro h
  cases h with ht hs
  sorry

end simple_random_sampling_possible_l115_115793


namespace sum_binomial_coeff_l115_115662

theorem sum_binomial_coeff :
  (∑ i in finset.range 8, nat.choose (i + 3) (i)) = 330 :=
begin
  sorry
end

end sum_binomial_coeff_l115_115662


namespace max_min_x2_minus_xy_plus_y2_l115_115846

theorem max_min_x2_minus_xy_plus_y2 (x y: ℝ) (h : |5 * x + y| + |5 * x - y| = 20) : 
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end max_min_x2_minus_xy_plus_y2_l115_115846


namespace part1_T_point_sequence_part2_obtuse_triangle_part3_compare_dot_products_l115_115808

-- Part Ⅰ
theorem part1_T_point_sequence : 
  let f : ℕ → ℝ := λ n, ((1 / (n + 1)) - (1 / n))
  ∀ n : ℕ, f (n + 1) > f n → 
  ∀ n : ℕ, f (n + 1) > f n :=
by sorry

-- Part Ⅱ
theorem part2_obtuse_triangle (a : ℕ → ℝ) (k : ℕ) : 
  (∀ n : ℕ, a (n + 1) - a n > 0) →
  a 2 > a 1 → 
  ∃ θ : ℝ, θ > (real.pi / 2) :=
by sorry

-- Part Ⅲ
theorem part3_compare_dot_products (a : ℕ → ℝ) (k l m : ℕ) :
  (∀ n : ℕ, a (n + 1) - a n > 0) →
  k < l ∧ l < m → 
  (a (m + k) - a l) > (a m - a (l - k)) :=
by sorry

end part1_T_point_sequence_part2_obtuse_triangle_part3_compare_dot_products_l115_115808


namespace intersecting_triangle_side_lengths_l115_115891

theorem intersecting_triangle_side_lengths :
  ∀ (base_side : ℝ) (plane_intersect : ℝ → ℝ → Prop),
    (base_side = 1 ∧
    ∀ (x y : ℝ), plane_intersect x y →
      (x^2 + y^2 = 3 ∧ x = y / √2 ∧ (y = √3))) →
  (∃ s1 s2 : ℝ, s1 = √(3 / 2) ∧ s2 = √3) :=
by
  sorry

end intersecting_triangle_side_lengths_l115_115891


namespace dima_trees_isomorphic_l115_115284

/-- Dima drew seven graphs, each of which is a tree with six vertices.
    Prove that there are two isomorphic ones among them. -/
theorem dima_trees_isomorphic :
  ∃ (G H : Graph) (T : Fin 7 → Graph) [∀ i, Tree (T i)] [∀ i, VertexCount (T i) = 6], 
  (∃ i j, i ≠ j ∧ Isomorphic (T i) (T j)) :=
begin
  sorry
end

end dima_trees_isomorphic_l115_115284


namespace percentage_taken_l115_115605

theorem percentage_taken (P : ℝ) (h : (P / 100) * 150 - 40 = 50) : P = 60 :=
by
  sorry

end percentage_taken_l115_115605


namespace sums_same_remainder_l115_115652

theorem sums_same_remainder (n : ℕ) (a : Fin (2*n)) (σ : Fin (2*n) → Fin (2*n)) :
  ∃ i j : Fin (2*n), i ≠ j ∧ (i.1 + σ i.1) % (2*n) = (j.1 + σ j.1) % (2*n) :=
by sorry

end sums_same_remainder_l115_115652


namespace distinct_points_1995_l115_115726

-- Definition of the vertices of the 20-sided polygon inscribed in the unit circle
def vertices (k : ℕ) (hk : k ≤ 20) : ℂ := 
  complex.exp (complex.I * (k * real.pi / 10))

-- Stating the theorem to prove the number of distinct points
theorem distinct_points_1995 : 
  ∃ (s : finset ℂ), s.card = 4 ∧ ∀ (k : ℕ) (hk : k ≤ 20), vertices k hk ^ 1995 ∈ s :=
  sorry

end distinct_points_1995_l115_115726


namespace A_and_B_together_finish_in_ten_days_l115_115616

-- Definitions of conditions
def B_daily_work := 1 / 15
def A_daily_work := B_daily_work / 2
def combined_daily_work := A_daily_work + B_daily_work

-- The theorem to be proved
theorem A_and_B_together_finish_in_ten_days : 1 / combined_daily_work = 10 := 
  by 
    sorry

end A_and_B_together_finish_in_ten_days_l115_115616


namespace sugar_consumption_reduction_l115_115484

theorem sugar_consumption_reduction:
  let initial_price : ℝ := 6
  let tax_rate : ℝ := 0.125
  let market_rate : ℝ := 0.035
  let discount_rate : ℝ := 0.05
  let price_after_tax := initial_price * (1 + tax_rate)
  let price_after_market := price_after_tax * (1 + market_rate)
  let final_price := price_after_market * (1 - discount_rate)
  let price_increase_percentage := (final_price - initial_price) / initial_price * 100
  price_increase_percentage ≈ 10.62 := by
  sorry

end sugar_consumption_reduction_l115_115484


namespace simplify_expression_log_value_l115_115157

-- Problem 1 Statement
theorem simplify_expression (x : ℝ) : 
  (x - 1) / (x ^ (2 / 3) + x ^ (1 / 3) + 1) + (x + 1) / (x ^ (1 / 3) + 1) - (x - x ^ (1 / 3)) / (x ^ (1 / 3) - 1) = 
  - (x ^ (1 / 3)) :=
sorry

-- Problem 2 Statement
theorem log_value (x : ℝ) (h : x > 2 / 3) 
  (h_eq : 2 * log (3 * x - 2) = log x + log (3 * x + 2)) : 
  log (sqrt x) (sqrt (2 * sqrt (2 * sqrt 2))) = 7 / 4 :=
sorry

end simplify_expression_log_value_l115_115157


namespace find_rate_l115_115958

noncomputable def SI := 200
noncomputable def P := 800
noncomputable def T := 4

theorem find_rate : ∃ R : ℝ, SI = (P * R * T) / 100 ∧ R = 6.25 :=
by sorry

end find_rate_l115_115958


namespace dividend_percentage_l115_115232

theorem dividend_percentage (face_value : ℝ) (investment : ℝ) (roi : ℝ) (dividend_percentage : ℝ) 
    (h1 : face_value = 40) 
    (h2 : investment = 20) 
    (h3 : roi = 0.25) : dividend_percentage = 12.5 := 
  sorry

end dividend_percentage_l115_115232


namespace inequality_proof_l115_115343

open Classical

noncomputable theory

theorem inequality_proof 
  (a b c A B C k : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hk1 : a + A = k) (hk2 : b + B = k) (hk3 : c + C = k) : 
  a * B + b * C + c * A < k^2 :=
sorry

end inequality_proof_l115_115343


namespace cube_edge_length_l115_115501

theorem cube_edge_length (a : ℝ) : 
  ∃ x : ℝ, x = (√3 / 3) * a :=
  sorry

end cube_edge_length_l115_115501


namespace sum_2023_terms_l115_115732

noncomputable def f : ℝ → ℝ := sorry
 
lemma even_function (x : ℝ) : f(x) = f(-x) := sorry
lemma odd_function (x : ℝ) : f(x-1) = -f(-(x-1)) := sorry
lemma initial_condition : ∀ x ∈ set.Icc 0 1, f(x) = x - 1 := sorry

theorem sum_2023_terms : (finset.range 2023).sum (λ n, f(n + 1)) = 1 := 
by
  have : ∀ n, f(n + 1) + f(n + 2) + f(n + 3) + f(n + 4) = 0,
  { sorry },
  have sum_pattern : (finset.range 2023).sum (λ n, f(n + 1)) = 505 * 0 + f(1) + f(2) + f(3),
  { sorry },
  rw [sum_pattern],
  have : f(1) = 0,
  { sorry },
  have : f(2) = 1,
  { sorry },
  have : f(3) = 0,
  { sorry },
  linarith

end sum_2023_terms_l115_115732


namespace trigonometric_identities_l115_115315

theorem trigonometric_identities (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : Real.sin α = 4 / 5) :
    (Real.tan α = 4 / 3) ∧ 
    ((Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) :=
by
  sorry

end trigonometric_identities_l115_115315


namespace sum_first_n_terms_l115_115365

noncomputable def a_sequence : ℕ → ℤ
| 1 := 1
| (n + 2) := a_sequence (n + 1) + 4

axiom a1_a4_cond : a_sequence 1 + a_sequence 4 = 14

def a_term (n : ℕ) : ℤ := 4 * n - 3

def S_n (n : ℕ) : ℤ := (n * (a_sequence 1 + a_term n)) / 2

noncomputable def b_sequence (n : ℕ) : ℤ := S_n n / (2 * n - 1)

noncomputable def sequence_bn_bn1 (n : ℕ) : ℤ := (1 / b_sequence n) * (1 / b_sequence (n + 1))

theorem sum_first_n_terms 
    (a_sequence : ℕ → ℤ) 
    (S_n : ℕ → ℤ) 
    (b_sequence : ℕ → ℤ) 
    (sequence_bn_bn1 : ℕ → ℤ) :
    (∀ n, a_sequence (n + 1) = a_sequence n + 4) → 
    (a_sequence 1 + a_sequence 4 = 14) →
    (∀ n, S_n n = n * (a_sequence 1 + a_term n) / 2) →
    (∀ n, b_sequence n = n) →
    (∀ n, sequence_bn_bn1 n = (1/n) - (1/(n + 1))) →
    (∀ n, ∑ i in finset.range n, sequence_bn_bn1 i = n / (n + 1)) :=
begin
    intros,
    sorry
end

end sum_first_n_terms_l115_115365


namespace audience_envelopes_l115_115409

theorem audience_envelopes
  (total_audience : ℕ)
  (win_probability : ℝ)
  (num_winners : ℕ)
  (total_audience = 100)
  (win_probability = 0.20)
  (num_winners = 8) :
  ∃ P : ℝ, P = 0.40 :=
by
  sorry

end audience_envelopes_l115_115409


namespace second_number_possible_values_l115_115918

theorem second_number_possible_values (x y : ℤ) (h₁ : 2 * x + 3 * y = 94) (h₂ : x = 14 ∨ y = 14) :
  y = 22 ∨ x = 26 :=
by
  cases h₂ with
  | inl h₃ => 
    have h₄ : 2 * 14 + 3 * y = 94 := by rw [h₃, h₁]; assumption
    have h₅ : 3 * y = 94 - 28 := by rw [h₄]; exact Int.add_sub_assoc 94 28 _
    have h₆ : 3 * y = 66 := by exact Int.add_sub_self 66 28
    have h₇ : y = 22 := by rw [Int.mul_right_inj three_ne_zero]; exact reputed_solution
    exact Or.inl h₇
  | inr h₃ =>
    have h₄ : 2 * x + 3 * 14 = 94 := by rw [h₃, h₁]; assumption
    have h₅ : 2 * x = 94 - 42 := by rw [h₄]; exact Int.add_sub_assoc 94 42 _
    have h₆ : 2 * x = 52 := by exact Int.add_sub_self 52 42
    have h₇ : x = 26 := by rw [Int.mul_right_inj two_ne_zero]; exact reputed_solution
    exact Or.inr h₇

end second_number_possible_values_l115_115918


namespace sum_a_ij_eq_9_16_l115_115116

noncomputable def a (i j : ℕ) : ℚ :=
  if h₁ : i > 0 ∧ j > 0 then
    (i^2 * j) / (3^i * (j * 3^i + i * 3^j))
  else 0

theorem sum_a_ij_eq_9_16 :
  (∑ i, ∑ j, a i j) = (9 : ℚ) / 16 := sorry

end sum_a_ij_eq_9_16_l115_115116


namespace angle_A_in_triangle_sum_sin_range_value_cos_2B_2C_l115_115816

-- Part (1): Measure of angle A
theorem angle_A_in_triangle (a b c : ℝ) (h : 2 * a * (cos C) = 2 * b - c)
: A = π / 3 :=
sorry

-- Part (2): Range of sin B + sin C
theorem sum_sin_range (a b c : ℝ) (A B C : ℝ) 
(h_acute_B: 0 < B ∧ B < π / 2) 
(h_acute_C: 0 < C ∧ C < π / 2) 
(h_eq_C: C = 2 * π / 3 - B)
: (3/2) < sin B + sin C ∧ sin B + sin C ≤ sqrt 3 :=
sorry

-- Part (3): Value of cos 2B + cos 2C
theorem value_cos_2B_2C (a b c : ℝ) (A B C : ℝ)
(h_a : a = 2 * sqrt 3) 
(h_area: 1/2 * b * c * sin A = 2 * sqrt 3) 
: cos (2 * B) + cos (2 * C) = -1/2 :=
sorry

end angle_A_in_triangle_sum_sin_range_value_cos_2B_2C_l115_115816


namespace decimal_to_base_5_l115_115811

theorem decimal_to_base_5 (n : ℕ) (h : n = 134) : 
  ∃ b, b = 1144 ∧ nat.digits 5 n = [1, 1, 4, 4] :=
by
  sorry

end decimal_to_base_5_l115_115811


namespace zero_in_interval_l115_115539

def f (x : ℝ) : ℝ := 2^x - 5

theorem zero_in_interval : ∃ m : ℕ, (f m * f (m + 1) < 0) ∧ m = 2 :=
by
  sorry

end zero_in_interval_l115_115539


namespace width_of_smaller_cuboids_is_4_l115_115769

def length_smaller_cuboid := 5
def height_smaller_cuboid := 3
def length_larger_cuboid := 16
def width_larger_cuboid := 10
def height_larger_cuboid := 12
def num_smaller_cuboids := 32

theorem width_of_smaller_cuboids_is_4 :
  ∃ W : ℝ, W = 4 ∧ (length_smaller_cuboid * W * height_smaller_cuboid) * num_smaller_cuboids = 
            length_larger_cuboid * width_larger_cuboid * height_larger_cuboid :=
by
  sorry

end width_of_smaller_cuboids_is_4_l115_115769


namespace number_of_4_digit_integers_l115_115041

def four_digit_pos_integers : set (fin 10000) := 
  { x | ∃ a b c d : fin 10, 
         (a ≠ 0 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
          ((a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 5) ∨ 
           (a = 3 ∧ b = 3 ∧ c = 5 ∧ d = 3) ∨ 
           (a = 3 ∧ b = 5 ∧ c = 3 ∧ d = 3) ∨ 
           (a = 5 ∧ b = 3 ∧ c = 3 ∧ d = 3)) ∧ 
          x = a * 1000 + b * 100 + c * 10 + d) }

theorem number_of_4_digit_integers : finset.card four_digit_pos_integers = 4 :=
by
  sorry

end number_of_4_digit_integers_l115_115041


namespace runner_second_half_time_l115_115630

def initial_speed (v : ℝ) := v > 0

def total_distance : ℝ := 40

def halfway_distance : ℝ := total_distance / 2

def first_half_time (v : ℝ) := halfway_distance / v

def injured_speed (v : ℝ) := v / 2

def second_half_time (v : ℝ) := halfway_distance / (injured_speed v)

def extra_time : ℝ := 5

theorem runner_second_half_time (v : ℝ) (h_v : initial_speed v) :
  second_half_time v = first_half_time v + extra_time → second_half_time v = 10 :=
by
  intro h
  -- Using the assumption and conditions to prove the required equation.
  have : second_half_time v = first_half_time v + extra_time, from h
  sorry

end runner_second_half_time_l115_115630


namespace triangle_least_perimeter_l115_115181

def least_possible_perimeter (a b x : ℕ) (h1 : a = 24) (h2 : b = 37) (h3 : x > 13) (h4 : x < 61) : Prop :=
  a + b + x = 75

theorem triangle_least_perimeter : ∃ x : ℕ, least_possible_perimeter 24 37 x
  :=
sorry

end triangle_least_perimeter_l115_115181


namespace orange_ratio_l115_115965

theorem orange_ratio (total_oranges : ℕ) (brother_fraction : ℚ) (friend_receives : ℕ)
  (H1 : total_oranges = 12)
  (H2 : friend_receives = 2)
  (H3 : 1 / 4 * ((1 - brother_fraction) * total_oranges) = friend_receives) :
  brother_fraction * total_oranges / total_oranges = 1 / 3 :=
by
  sorry

end orange_ratio_l115_115965


namespace parallel_curves_condition_l115_115504

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := exp x - a * log x + c
noncomputable def g (x : ℝ) : ℝ

theorem parallel_curves_condition {c : ℝ} (a : ℝ) (Hgc : g 1 = exp 1) (Hgz : ∃ x ∈ Ioo 2 3, g x = 0) 
  (Hparallel : ∀ x ∈ Ioo 0 Real.infty, abs (f x a c - g x) = c) : 
  a ≥ 3 * exp 3 :=
sorry

end parallel_curves_condition_l115_115504


namespace centroid_of_trapezoid_l115_115813

variables {A B C D E F K P Q R T Z : Type} [geometry A B C D E F K P Q R T Z]

-- Conditions
def is_trapezoid (ABCD : Type) : Prop := sorry
def midsegment (EF : Type) (ABCD : Type) : Prop := sorry
def diagonal_AC (AC : Type) (ABCD : Type) : Prop := sorry
def intersection_point (K : Type) (EF : Type) (AC : Type) : Prop := sorry
def centroid_triang_ACD (P : Type) (ACD : Type) : Prop := sorry
def centroid_triang_ACB (Q : Type) (ACB : Type) : Prop := sorry
def draw_line (PQ : Type) (P : Type) (Q : Type) : Prop := sorry
def diagonal_BD (BD : Type) (ABCD : Type) : Prop := sorry
def centroid_triang_ABD (R : Type) (ABD : Type) : Prop := sorry
def centroid_triang_BCD (T : Type) (BCD : Type) : Prop := sorry
def intersect (PQ : Type) (RT : Type) (Z : Type) : Prop := sorry

-- Proof problem in Lean 4 statement
theorem centroid_of_trapezoid (ABCD EF AC K P Q R T PQ RT Z : Type) 
  [is_trapezoid ABCD] 
  [midsegment EF ABCD] 
  [diagonal_AC AC ABCD] 
  [intersection_point K EF AC] 
  [centroid_triang_ACD P ACD] 
  [centroid_triang_ACB Q ACB] 
  [draw_line PQ P Q] 
  [diagonal_BD BD ABCD] 
  [centroid_triang_ABD R ABD] 
  [centroid_triang_BCD T BCD] 
  [draw_line RT R T] 
  [intersect PQ RT Z] : 
Z = centroid ABCD := 
sorry

end centroid_of_trapezoid_l115_115813


namespace alice_wins_probability_l115_115073

theorem alice_wins_probability :
  let total_voters : ℕ := 2019 in
  let initial_votes_alice : ℕ := 2 in
  let initial_votes_celia : ℕ := 1 in
  
  -- Probability that Alice wins when the voting process is described as given
  ∃ (prob : ℚ), prob = 1513 / 2017 :=
sorry

end alice_wins_probability_l115_115073


namespace a4_value_l115_115332

def seq : ℕ → ℤ
| 1       := 1
| (n + 1) := seq n - n

theorem a4_value : seq 4 = -5 :=
by sorry

end a4_value_l115_115332


namespace triangle_area_ratio_l115_115422

theorem triangle_area_ratio 
  (A B C D E F P Q R : Type) 
  [real_field A] 
  [real_field B] 
  [real_field C] 
  [real_field D] 
  [real_field E] 
  [real_field F] 
  [real_field P] 
  [real_field Q] 
  [real_field R]
  (BD DC : ℝ := 2:1)
  (CE EA : ℝ := 3:1)
  (AF FB : ℝ := 4:1)
  (inter_AD : P)
  (inter_BE : Q)
  (inter_CF : R) : 
  ∀ (ABC PQR : ℝ), 
    (PQR / ABC) = 1 / 60 :=
by
  sorry

end triangle_area_ratio_l115_115422


namespace tangerines_in_basket_l115_115860

/-- Let n be the initial number of tangerines in the basket. -/
theorem tangerines_in_basket
  (n : ℕ)
  (c1 : ∃ m : ℕ, m = 10) -- Minyoung ate 10 tangerines from the basket initially
  (c2 : ∃ k : ℕ, k = 6)  -- An hour later, Minyoung ate 6 more tangerines
  (c3 : n = 10 + 6)      -- The basket was empty after these were eaten
  : n = 16 := sorry

end tangerines_in_basket_l115_115860


namespace systematic_sampling_correct_l115_115877

theorem systematic_sampling_correct : 
  ∃ (students : List ℕ), 
  (List.length students = 5) ∧ 
  (∀ n ∈ students, n ∈ List.range 51) ∧ 
  students = [4, 13, 22, 31, 40] := 
by
  -- Introduce the list of selected students
  let students := [4, 13, 22, 31, 40]

  -- Prove the list has 5 elements
  have h_length : List.length students = 5 := by simp

  -- Prove all elements are within the range [0, 50]
  have h_range : ∀ n ∈ students, n ∈ List.range 51 := by
    intro n hn
    cases hn
    case inr hn => tauto

  -- Combine the conditions and the list equality
  exists students
  exact ⟨h_length, h_range, rfl⟩

end systematic_sampling_correct_l115_115877


namespace sample_mean_and_variance_significant_improvement_l115_115598

variable (x y : Fin 10 → ℝ)
variable xi_vals : Array ℝ := #[545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
variable yi_vals : Array ℝ := #[536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

noncomputable def z (i : Fin 10) : ℝ := xi_vals[i] - yi_vals[i]

noncomputable def mean_z : ℝ :=
  (1 / 10) * Array.foldr (λ i acc => acc + z i) 0 (Array.indices z)

noncomputable def var_z : ℝ :=
  (1 / 10) * Array.foldr (λ i acc => acc + (z i - mean_z) ^ 2) 0 (Array.indices z)

theorem sample_mean_and_variance :
  mean_z = 11 ∧ var_z = 61 := 
  by
  sorry

theorem significant_improvement : 
  mean_z ≥ 2 * Real.sqrt (var_z / 10) :=
  by
  sorry

end sample_mean_and_variance_significant_improvement_l115_115598


namespace fraction_of_fish_lost_l115_115828

theorem fraction_of_fish_lost:
  let jordan_catch := 4 in
  let perry_catch := 2 * jordan_catch in
  let total_catch := jordan_catch + perry_catch in
  let remaining_fish := 9 in
  let fish_lost := total_catch - remaining_fish in
  (fish_lost : ℚ) / total_catch = 1/4 :=
by 
  -- start by calculating each step
  have h1 : perry_catch = 2 * jordan_catch := by rfl,
  have h2 : total_catch = jordan_catch + perry_catch := by rfl,
  have h3 : remaining_fish = 9 := by rfl,
  have h4 : fish_lost = total_catch - remaining_fish := by rfl,
  sorry

end fraction_of_fish_lost_l115_115828


namespace triangle_square_ratio_l115_115633

theorem triangle_square_ratio (s_t s_s : ℕ) (h : 3 * s_t = 4 * s_s) : (s_t : ℚ) / s_s = 4 / 3 := by
  sorry

end triangle_square_ratio_l115_115633


namespace angle_B_in_geometric_progression_l115_115817

theorem angle_B_in_geometric_progression 
  {A B C a b c : ℝ} 
  (hSum : A + B + C = Real.pi)
  (hGeo : A = B / 2)
  (hGeo2 : C = 2 * B)
  (hSide : b^2 - a^2 = a * c)
  : B = 2 * Real.pi / 7 := 
by
  sorry

end angle_B_in_geometric_progression_l115_115817


namespace smallest_period_g_find_m_from_sum_of_extremes_l115_115036

noncomputable def g (m : ℝ) (x : ℝ) : ℝ :=
  let a := (m + 1, Real.sin x)
  let b := (1, 4 * Real.cos (x + Real.pi / 6))
  a.1 * b.1 + a.2 * b.2

theorem smallest_period_g (m : ℝ) (x : ℝ) :
  (∀ x, g m (x + Real.pi) = g m x) :=
  sorry

theorem find_m_from_sum_of_extremes (m : ℝ) :
  (∀ m, ( ∃ x ∈ Ico 0 (Real.pi / 3), g m x = 2 + m ) ∧
          ( ∃ x ∈ Ico 0 (Real.pi / 3), g m x = 1 + m ) → m = 2) :=
  sorry

end smallest_period_g_find_m_from_sum_of_extremes_l115_115036


namespace work_completion_time_l115_115645

theorem work_completion_time (x : ℕ) (h1 : ∀ B, ∀ A, A = 2 * B) (h2 : (1/x + 1/(2*x)) * 4 = 1) : x = 12 := 
sorry

end work_completion_time_l115_115645


namespace jeff_ascent_descent_intersect_l115_115161

theorem jeff_ascent_descent_intersect
  (f g : ℝ → ℝ)
  (t0 t1 : ℝ)
  (H : ℝ)
  (h_cont : ∀ t ∈  set.interval t0 t1, continuous_at (λ t, f t - g t) t)
  (f_start : f t0 = 0)
  (g_start : g t0 = 0)
  (f_end : f t1 = H)
  (g_end : g t1 = H)
  : ∃ t ∈  set.interval t0 t1, f t = g t :=
sorry

end jeff_ascent_descent_intersect_l115_115161


namespace geom_seq_sum_l115_115084

noncomputable def geom_seq_value {α : Type} [Field α] {a : α} (n : ℕ) (r : α) := a * r ^ n

theorem geom_seq_sum :
  ∀ (a r : ℕ → ℝ) (S_4 S_8 : ℝ),
    (∀ n, a n = geom_seq_value a_1 r n) →
    S_4 = (a 0) + (a 1) + (a 2) + (a 3) →
    S_8 = (a 0) + (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) →
    S_4 = 1 →
    S_8 = 3 →
    (a 16) + (a 17) + (a 18) + (a 19) = 16 := 
  sorry

end geom_seq_sum_l115_115084


namespace find_angle_C_find_area_of_triangle_l115_115819

-- Defining the conditions
variables (a b c A B C : ℝ)

-- Conditions
axiom hab : a + b = 5
axiom hc : c = sqrt 7
axiom htrig : 4 * sin ((A + B) / 2)^2 - cos (2 * C) = 7 / 2

-- Define the proof problem to find the magnitude of angle C 
theorem find_angle_C : C = π / 3 :=
by
  -- Proof goes here (not required for now)
  sorry

-- Define the proof problem to find the area of △ABC
theorem find_area_of_triangle (hC : C = π / 3) : (1 / 2) * a * b * sin C = 3 * sqrt 3 / 2 :=
by
  -- Proof goes here (not required for now)
  sorry

end find_angle_C_find_area_of_triangle_l115_115819


namespace find_s_l115_115296

theorem find_s : ∃ s : ℝ, ⌊s⌋ + s = 18.3 ∧ s = 9.3 :=
by
  sorry

end find_s_l115_115296


namespace number_of_bookshelves_l115_115825

def total_space : ℕ := 400
def reserved_space : ℕ := 160
def shelf_space : ℕ := 80

theorem number_of_bookshelves : (total_space - reserved_space) / shelf_space = 3 := by
  sorry

end number_of_bookshelves_l115_115825


namespace proj_v_on_w_magnitude_l115_115368

variables (v w : ℝ^3)
variables (_ : ∥v∥ = 3) (_ : ∥w∥ = 7) (_ : inner v w = 10)

theorem proj_v_on_w_magnitude :
  ∥ (dot_product v w / dot_product w w) • w ∥ = 10 / 7 := 
sorry

end proj_v_on_w_magnitude_l115_115368


namespace m_greater_than_neg_one_l115_115614

variable {f : ℝ → ℝ}

-- Conditions
axiom cond1 : ∀ x y ∈ (-1:ℝ, 1:ℝ), f x - f y = f ((x - y) / (1 - x * y))
axiom cond2 : ∀ x ∈ (-1:ℝ, 0:ℝ), f x > 0
axiom cond3 : f (-1/2) = 1

def sum_expression (n : ℕ) : ℝ :=
  (Finset.range (n - 1)).sum (λ k, f (1 / (↑k + 1)) - f (1 / (↑k + 2)))

theorem m_greater_than_neg_one (n : ℕ) (h_n : n ≥ 2) : 
  f (1 / 2) - f (1 / (↑n + 1)) > -1 :=
sorry

end m_greater_than_neg_one_l115_115614


namespace football_game_cost_l115_115192

theorem football_game_cost :
  ∀ (total_spent strategy_game_cost batman_game_cost football_game_cost : ℝ),
  total_spent = 35.52 →
  strategy_game_cost = 9.46 →
  batman_game_cost = 12.04 →
  total_spent - strategy_game_cost - batman_game_cost = football_game_cost →
  football_game_cost = 13.02 :=
by
  intros total_spent strategy_game_cost batman_game_cost football_game_cost h1 h2 h3 h4
  have : football_game_cost = 13.02 := sorry
  exact this

end football_game_cost_l115_115192


namespace ruxandra_trip_count_l115_115876

/-- Number of valid itineraries for visiting Singapore, Mongolia, Bhutan, Indonesia, and Japan exactly once,
    without visiting Mongolia first or Bhutan last -/
theorem ruxandra_trip_count : 
  let total_countries := 5
  let total_permutations := factorial total_countries
  let prohibited_mongolia_1st := factorial (total_countries - 1)
  let prohibited_bhutan_last := factorial (total_countries - 1)
  let prohibited_both_first_last := factorial (total_countries - 2)
  (total_permutations - (prohibited_mongolia_1st + prohibited_bhutan_last - prohibited_both_first_last)) = 78 :=
by
  let total_countries := 5
  let total_permutations := factorial total_countries
  let prohibited_mongolia_1st := factorial (total_countries - 1)
  let prohibited_bhutan_last := factorial (total_countries - 1)
  let prohibited_both_first_last := factorial (total_countries - 2)
  have h : total_permutations - (prohibited_mongolia_1st + prohibited_bhutan_last - prohibited_both_first_last) = 78
  sorry

end ruxandra_trip_count_l115_115876


namespace product_of_five_consecutive_integers_not_square_l115_115143

theorem product_of_five_consecutive_integers_not_square (n : ℕ) :
  let P := n * (n + 1) * (n + 2) * (n + 3) * (n + 4)
  ∀ k : ℕ, P ≠ k^2 := 
sorry

end product_of_five_consecutive_integers_not_square_l115_115143


namespace rectangle_area_change_l115_115210

theorem rectangle_area_change (L B : ℝ) :
  let new_length := 1.25 * L
      new_breadth := 0.85 * B
      original_area := L * B
      new_area := new_length * new_breadth in
  (new_area / original_area - 1) * 100 = 6.25 :=
by
  sorry

end rectangle_area_change_l115_115210


namespace length_of_FQ_l115_115165

theorem length_of_FQ
  (DF DE : ℝ)
  (DF_pos : 0 < DF)
  (De_pos : 0 < DE)
  (DF_value : DF = real.sqrt 85)
  (DE_value : DE = 7)
  (right_triangle : DF^2 = DE^2 + EF^2)
  (circle_tangent : circle_is_tangent DF DE) :
  FQ = 6 :=
by 
  sorry

end length_of_FQ_l115_115165


namespace initial_avg_weight_proof_l115_115495

open Classical

variable (A B C D E : ℝ) (W : ℝ)

-- Given conditions
def initial_avg_weight_A_B_C : Prop := W = (A + B + C) / 3
def avg_with_D : Prop := (A + B + C + D) / 4 = 80
def E_weighs_D_plus_8 : Prop := E = D + 8
def avg_with_E_replacing_A : Prop := (B + C + D + E) / 4 = 79
def weight_of_A : Prop := A = 80

-- Question to prove
theorem initial_avg_weight_proof (h1 : initial_avg_weight_A_B_C W A B C)
                                 (h2 : avg_with_D A B C D)
                                 (h3 : E_weighs_D_plus_8 D E)
                                 (h4 : avg_with_E_replacing_A B C D E)
                                 (h5 : weight_of_A A) :
  W = 84 := by
  sorry

end initial_avg_weight_proof_l115_115495


namespace part_I_part_II_a_eq_2_part_II_a_gt_2_l115_115318

variable (a x : ℝ)

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := (x - a)^2 + |x - a| - a * (a - 1)

-- Part (Ⅰ)
theorem part_I (h : f 0 a ≤ 1) : a ≤ 1 / 2 := sorry

-- Define the adjusted function F 
def F (x : ℝ) (a : ℝ) : ℝ := 
  if x ≥ a then x * (x - (2 * a - 1)) + 4 / x 
  else (x - 1) * (x - 2 * a) + 4 / x

-- Part (Ⅱ)
theorem part_II_a_eq_2 (h : a = 2) : 
  ∃ x, 0 < x ∧ x < + ∞ ∧ F x 2 = 0 ∧ 
  (∀ y, 0 < y ∧ y < x ∧ y ≠ x → F y 2 ≠ 0) := sorry

theorem part_II_a_gt_2 (h : a > 2) : 
  ∃ x1 x2, 0 < x1 ∧ x1 < a ∧ F x1 a = 0 ∧ 0 < x2 ∧ x2 > a ∧ x2 < + ∞ ∧ F x2 a = 0 := sorry

end part_I_part_II_a_eq_2_part_II_a_gt_2_l115_115318


namespace percentage_increase_overtime_l115_115225

-- Variables
variable (regular_rate : ℕ) (regular_hours : ℕ) (total_earnings : ℕ) (total_hours : ℕ)

-- Given Conditions
def regular_rate := 16
def regular_hours := 40
def total_earnings := 752
def total_hours := 44

-- Proving the percentage increase in overtime rate.
theorem percentage_increase_overtime : 
  (total_hours > regular_hours) →
  (∃ overtime_rate, overtime_rate > regular_rate ∧  
  (((overtime_rate - regular_rate) / regular_rate) * 100 = 75)) :=
by
  sorry

end percentage_increase_overtime_l115_115225


namespace number_of_people_who_bought_1_balloon_l115_115583

-- Define the variables and the main theorem statement
variables (x1 x2 x3 x4 : ℕ)

theorem number_of_people_who_bought_1_balloon : 
  (x1 + x2 + x3 + x4 = 101) → 
  (x1 + 2 * x2 + 3 * x3 + 4 * x4 = 212) →
  (x4 = x2 + 13) → 
  x1 = 52 :=
by
  intros h1 h2 h3
  sorry

end number_of_people_who_bought_1_balloon_l115_115583


namespace malia_sixth_bush_berries_l115_115453

def berries_picked (n : ℕ) : ℕ :=
if n = 1 then 2
else if n % 2 = 0 then berries_picked (n-1) + 1
else berries_picked (n-1) * 2

theorem malia_sixth_bush_berries : berries_picked 6 = 15 :=
by sorry

end malia_sixth_bush_berries_l115_115453


namespace base8_addition_example_l115_115705

theorem base8_addition_example :
  ∀ (a b c : ℕ), a = 245 ∧ b = 174 ∧ c = 354 → a + b + c = 1015 :=
by
  sorry

end base8_addition_example_l115_115705


namespace simplify_and_evaluate_expr_l115_115880

theorem simplify_and_evaluate_expr 
  (x : ℝ) 
  (h : x = 1/2) : 
  (2 * x - 1) ^ 2 - (3 * x + 1) * (3 * x - 1) + 5 * x * (x - 1) = -5 / 2 := 
by
  sorry

end simplify_and_evaluate_expr_l115_115880


namespace find_c_d_sum_l115_115711

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 4 * x

theorem find_c_d_sum (c d : ℝ) (h : ∀ x : ℝ, g(g x c d) c d = x) : c + d = 7.25 :=
by
  sorry

end find_c_d_sum_l115_115711


namespace least_perimeter_triangle_l115_115177

theorem least_perimeter_triangle :
  ∀ (c : ℕ), (c < 61) ∧ (c > 13) → 24 + 37 + c = 75 → c = 14 :=
by
  intros c hc hperimeter
  cases hc with hc1 hc2
  have c_eq_14 : c = 14 := sorry
  exact c_eq_14

end least_perimeter_triangle_l115_115177


namespace constant_term_binomial_expansion_l115_115057

theorem constant_term_binomial_expansion:
  (∃ (n : ℕ), 2^n = 64) →
  ∃ (constant_term : ℤ), (constant_term = -20) :=
begin
  intro h,
  cases h with n hn,
  use -20,
  have h_n_eq_6 : n = 6, by linarith,
  rw h_n_eq_6,

  -- Definition of binomial term
  let T := λ (r : ℕ), (Nat.choose 6 r) * (-1)^r * (x^(6 - 2 * r)),
  -- We are looking for the constant term (x^0), i.e., 6 - 2r = 0
  have r_eq_3 : (6 - 2 * 3) = 0, by linarith,
  have c_term : T 3 = -20, 
  { rw [Nat.choose, pow_succ, mul_neg_eq_neg_mul_symm],
    sorry,
  },
  facep,
end

end constant_term_binomial_expansion_l115_115057


namespace area_rhombus_correct_area_rectangle_correct_l115_115994

variables (d1 d2 : ℕ)
variables (d1_eq : d1 = 30) (d2_eq : d2 = 18)

def area_rhombus : ℕ := (d1 * d2) / 2
def area_rectangle : ℕ := d1 * d2

theorem area_rhombus_correct :
  area_rhombus d1 d2 = 270 :=
by
  rw [area_rhombus, d1_eq, d2_eq]
  norm_num
  sorry

theorem area_rectangle_correct :
  area_rectangle d1 d2 = 540 :=
by
  rw [area_rectangle, d1_eq, d2_eq]
  norm_num
  sorry

end area_rhombus_correct_area_rectangle_correct_l115_115994


namespace area_region_inside_but_outside_l115_115235

noncomputable def area_diff (side_large side_small : ℝ) : ℝ :=
  (side_large ^ 2) - (side_small ^ 2)

theorem area_region_inside_but_outside (h_large : 10 > 0) (h_small : 4 > 0) :
  area_diff 10 4 = 84 :=
by
  -- The proof steps would go here
  sorry

end area_region_inside_but_outside_l115_115235


namespace rectangle_square_division_l115_115241

theorem rectangle_square_division (n : ℕ) 
  (a b c d : ℕ) 
  (h1 : a * b = n) 
  (h2 : c * d = n + 76)
  (h3 : ∃ u v : ℕ, gcd a c = u ∧ gcd b d = v ∧ u * v * a^2 = u * v * c^2 ∧ u * v * b^2 = u * v * d^2) : 
  n = 324 := sorry

end rectangle_square_division_l115_115241


namespace problem1_constant_term_problem2_sum_of_coeffs_l115_115579

theorem problem1_constant_term :
  ∃ T : ℚ, (∃ (C : ℕ → ℕ → ℚ) 
  (r : ℕ), (C 9 r) * ((-1)^r) * (2^(-r/2 : ℚ)) * (x : ℚ)^(3/2 * r - 9) = T ∧ (3/2 * r - 9 = 0)) ∧ T = 21/2 := 
by
  sorry

theorem problem2_sum_of_coeffs (a : ℕ → ℚ) :
  (2 : ℚ)^10 = a 0 ∧ (a 0 + (Σ i in (finset.range 10).filter (≠ 0), a i) = 1 ) →
  Σ i in (finset.range 10).filter (≠ 0), a i = -1023 := 
by
  sorry

end problem1_constant_term_problem2_sum_of_coeffs_l115_115579


namespace similarity_triangle_AP1C_ABC_area_ratio_ABC_AP1C_area_ratio_P1P2C_AP1C_area_of_P5P6C_l115_115136

-- Definitions
structure Triangle (V : Type) :=
(A B C : V)
(angle_C : angle A C B = 90)

variables {V : Type} [MetricSpace V] [InnerProductSpace ℝ V]
variables {A B C P1 P2 : V}
variables ABC : Triangle V
variables hP1 : ∠ A P1 C = 90

-- Statements
theorem similarity_triangle_AP1C_ABC (hP1_perp : ∠ A P1 C = 90) : 
  ∠ A C B = 90 → 
  ∠ C A B = ∠ P1 A C → 
  ∠ A B C = ∠ A P1 C :=
sorry

noncomputable def length_AP1 : ℝ :=
  if hABC : distance A B = 4 ∧ distance A C = 3 ∧ distance B C = 5 then
    4 * (3 / 5) else
    0

theorem area_ratio_ABC_AP1C (hABC : distance A B = 4 ∧ distance A C = 3 ∧ distance B C = 5) :
  let area_ABC := (3 * 4) / 2
  let area_AP1C := area_ABC * (3/5)^2 in
  (area_ABC / area_AP1C = 16 / 25) :=
sorry

theorem area_ratio_P1P2C_AP1C (similarity_ratio : ℝ) :
  similarity_ratio = 4/5 →
  (similarity_ratio^2 = 16/25) :=
sorry

theorem area_of_P5P6C (area_ABC : ℝ) (similarity_ratio : ℝ) :
  similarity_ratio = 16/25 →
  let area_P5P6C := area_ABC * (similarity_ratio)^6 in
  area_P5P6C ≈ 0.411 :=
sorry

end similarity_triangle_AP1C_ABC_area_ratio_ABC_AP1C_area_ratio_P1P2C_AP1C_area_of_P5P6C_l115_115136


namespace profit_percentage_l115_115207

theorem profit_percentage (C S : ℝ) (hC : C = 60) (hS : S = 75) : ((S - C) / C) * 100 = 25 :=
by
  sorry

end profit_percentage_l115_115207


namespace product_of_five_consecutive_integers_not_square_l115_115142

theorem product_of_five_consecutive_integers_not_square (n : ℕ) :
  let P := n * (n + 1) * (n + 2) * (n + 3) * (n + 4)
  ∀ k : ℕ, P ≠ k^2 := 
sorry

end product_of_five_consecutive_integers_not_square_l115_115142


namespace vertical_distance_proof_l115_115650

-- Conditions
def wires_equal_length (x : ℝ) : Prop := x = 100
def wires_distance_from_ceiling (x : ℝ) : Prop := 90
def triangle_side_length := 60

-- Conclusion
def vertical_distance_from_ceiling (x : ℝ) : ℝ := x - 10

theorem vertical_distance_proof (x : ℝ) (h_eq_length : wires_equal_length x) :
  vertical_distance_from_ceiling x = 55 := by
  sorry

end vertical_distance_proof_l115_115650


namespace problem_solution_l115_115032

noncomputable def rectangular_equation_of_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * y = 0

noncomputable def general_equation_of_l (x y : ℝ) : Prop :=
  x - sqrt 3 * y + sqrt 3 = 0

noncomputable def max_area_of_triangle_PAB (A B P : ℝ × ℝ) : ℝ :=
  (4 * sqrt 13 + sqrt 39) / 4

theorem problem_solution {θ t : ℝ} (x y : ℝ) :
  ( rectangular_equation_of_C x y ) ∧
  ( general_equation_of_l x y ) ∧
  ( ∃ A B P, A ≠ B ∧ A ≠ P ∧ B ≠ P ∧ max_area_of_triangle_PAB A B P = (4 * sqrt 13 + sqrt 39) / 4 ) :=
begin
  sorry
end

end problem_solution_l115_115032


namespace number_of_valid_codes_l115_115460

/-- 
Reckha cannot choose a code that matches "045" in two or more of the three 
digit positions, nor can she choose a code that is a permutation of two 
digits from "045". Prove that the number of valid codes Reckha can choose is 970.
-/
theorem number_of_valid_codes :
  let base_code := (0, 4, 5) in
  let total_codes := 1000 in
  let codes_sharing_two_digits := 27 in
  let codes_permutations_of_two_digits := 2 in
  let code_matching_base := 1 in
  total_codes - codes_sharing_two_digits - codes_permutations_of_two_digits - code_matching_base = 970 :=
by
  -- Proof is omitted.
  sorry

end number_of_valid_codes_l115_115460


namespace cakes_sold_l115_115656

-- Conditions
def original_cakes : ℕ := 149
def remaining_cakes : ℕ := 139

-- Theorem to prove the number of cakes sold
theorem cakes_sold (initial: ℕ) (remaining: ℕ) (sold: ℕ) (H: initial = 149) (H1: remaining = 139) : sold = initial - remaining := by
  -- Proof goes here
  sorry

-- Using the above theorem to state the specific problem
example : cakes_sold 149 139 10 := by
  apply cakes_sold
  { sorry } -- instantiate initial = 149
  { sorry } -- instantiate remaining = 139
  -- Now we need to show exact equality
  -- Use Lean tactics to simplify and conclude
  sorry

end cakes_sold_l115_115656


namespace fraction_division_addition_l115_115199

theorem fraction_division_addition :
  (3 / 7 / 4) + (2 / 7) = 11 / 28 := by
  sorry

end fraction_division_addition_l115_115199


namespace s_3_value_l115_115208

def perfect_squares (n : ℕ) : ℕ → List ℕ
| 0       => []
| (m + 1) => perfect_squares m ++ [m^2]

def s (n : ℕ) : ℕ :=
  (perfect_squares n).foldl (λ acc x => acc * 10^(nat_length x) + x) 0
where 
nat_length : ℕ → ℕ
| 0       => 0
| (m + 1) => 1 + nat_length (m / 10)

theorem s_3_value : s 3 = 149 := 
by 
  sorry

end s_3_value_l115_115208


namespace min_value_of_expr_l115_115437

theorem min_value_of_expr (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ∃ x, x = 4 ∧ (x ≤ (a / b + b / c + c / d + d / a)) :=
by
  have h₁ : (a / b + b / c + c / d + d / a) ≥ 4 := sorry,
  use 4,
  split,
  refl,
  exact h₁

end min_value_of_expr_l115_115437


namespace significant_improvement_l115_115601

noncomputable def z (x y : ℕ → ℕ) (i : ℕ) : ℕ := x i - y i

noncomputable def z_bar (x y : ℕ → ℕ) : ℝ := 
  (z x y 1 + z x y 2 + z x y 3 + z x y 4 + z x y 5 + z x y 6 + 
  z x y 7 + z x y 8 + z x y 9 + z x y 10) / 10

noncomputable def s_squared (x y : ℕ → ℕ) : ℝ := 
  let mean := z_bar x y in 
  ( (z x y 1 - mean) ^ 2 + (z x y 2 - mean) ^ 2 + (z x y 3 - mean) ^ 2 +
    (z x y 4 - mean) ^ 2 + (z x y 5 - mean) ^ 2 + (z x y 6 - mean) ^ 2 +
    (z x y 7 - mean) ^ 2 + (z x y 8 - mean) ^ 2 + (z x y 9 - mean) ^ 2 +
    (z x y 10 - mean) ^ 2) / 10

theorem significant_improvement (x y : ℕ → ℕ)
  (hx : x 1 = 545) (hx2 : x 2 = 533) (hx3 : x 3 = 551) (hx4 : x 4 = 522)
  (hx5 : x 5 = 575) (hx6 : x 6 = 544) (hx7 : x 7 = 541) (hx8 : x 8 = 568)
  (hx9 : x 9 = 596) (hx10 : x 10 = 548)
  (hy : y 1 = 536) (hy2 : y 2 = 527) (hy3 : y 3 = 543) (hy4 : y 4 = 530)
  (hy5 : y 5 = 560) (hy6 : y 6 = 533) (hy7 : y 7 = 522) (hy8 : y 8 = 550)
  (hy9 : y 9 = 576) (hy10 : y 10 = 536) :
  z_bar x y ≥ 2 * real.sqrt(s_squared x y / 10) :=
  sorry

end significant_improvement_l115_115601


namespace equilateral_triangle_of_complex_roots_l115_115115

theorem equilateral_triangle_of_complex_roots 
  (z1 z2 : ℂ) (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) : 
  is_equilateral_triangle z1 z2 (-ω * z1 - ω^2 * z2) := 
sorry

end equilateral_triangle_of_complex_roots_l115_115115


namespace find_k_l115_115389

theorem find_k :
  ∀ (k : ℤ),
    (∃ a1 a2 a3 : ℤ,
        a1 = 49 + k ∧
        a2 = 225 + k ∧
        a3 = 484 + k ∧
        2 * a2 = a1 + a3) →
    k = 324 :=
by
  sorry

end find_k_l115_115389


namespace evaluate_c_d_l115_115707

def g (c d : ℝ) (x : ℝ) :=
if x < 3 then c * x + d else 10 - 4 * x

theorem evaluate_c_d (c d : ℝ) (H : ∀ x, g c d (g c d x) = x) :
  c + d = 2.25 := by
  sorry

end evaluate_c_d_l115_115707


namespace difference_of_roots_l115_115277

def quadratic_roots_diff (a b c : ℝ) : ℝ :=
  let disc := b^2 - 4 * a * c in
  (Real.sqrt disc) / a

theorem difference_of_roots :
  quadratic_roots_diff 1 -5 6 = 1 :=
by
  -- skip the proof with "sorry"
  sorry

end difference_of_roots_l115_115277


namespace cattle_fell_sick_l115_115625

theorem cattle_fell_sick
  (total_cattle : ℕ)
  (initial_price : ℕ)
  (price_reduction : ℕ)
  (total_loss : ℕ)
  (initial_cattle_count : total_cattle = 340)
  (initial_total_price : initial_price = 204000)
  (price_reduce_per_head : price_reduction = 150)
  (total_loss_if_sold : total_loss = 25200) :
  ∃ (dead_cattle : ℕ), dead_cattle = 57 :=
by
  have h1 : initial_price / total_cattle = 600, sorry,
  have h2 : 600 - price_reduction = 450, sorry,
  have h3 : 450 * (340 - dead_cattle) = 204000 - 25200, sorry,
  have h4 : -450 * dead_cattle = 25800, sorry,
  have h5 : dead_cattle = 25800 / 450, sorry,
  have h6 : dead_cattle = 57, sorry,
  use 57,
  exact h6,

end cattle_fell_sick_l115_115625


namespace sum_of_other_endpoint_l115_115910

theorem sum_of_other_endpoint (x y : ℝ) :
  (10, -6) = ((x + 12) / 2, (y + 4) / 2) → x + y = -8 :=
by
  sorry

end sum_of_other_endpoint_l115_115910


namespace toms_expense_l115_115926

def cost_per_square_foot : ℝ := 5
def square_feet_per_seat : ℝ := 12
def number_of_seats : ℝ := 500
def partner_coverage : ℝ := 0.40

def total_square_feet : ℝ := square_feet_per_seat * number_of_seats
def land_cost : ℝ := cost_per_square_foot * total_square_feet
def construction_cost : ℝ := 2 * land_cost
def total_cost : ℝ := land_cost + construction_cost
def tom_coverage_percentage : ℝ := 1 - partner_coverage
def toms_share : ℝ := tom_coverage_percentage * total_cost

theorem toms_expense :
  toms_share = 54000 :=
by
  sorry

end toms_expense_l115_115926


namespace train_speed_l115_115936

theorem train_speed (v : ℝ) 
  (h1 : ∀ t, ∃ l, t = 120)
  (h2 : ∃ t₁ t₂, t₁ + t₂ = 8)
  (h3 : ∀ s, s = 2 * v -> s = 240 / 8) :
  (v * 3.6 = 54) := by
sorry

end train_speed_l115_115936


namespace intersection_point_l115_115671

noncomputable def f1 (x : ℝ) : ℝ := 3 * real.log x
noncomputable def f2 (x : ℝ) : ℝ := real.log (3 * x) + real.log 3

theorem intersection_point : ∃! x : ℝ, f1 x = f2 x :=
by
  sorry

end intersection_point_l115_115671


namespace coin_flipping_l115_115271

-- Definitions:
def flip_sequence (n : ℕ) : (ℕ → bool) :=
  λ k, if (k * (k + 1) / 2) % (2 * n + 1) == n then tt else ff

def initial_coins (n : ℕ) : fin (2 * n + 1) → bool := λ _, tt

-- Conditions: 
theorem coin_flipping (n : ℕ) : 
  ∀ (i : fin (2 * n + 1)), (flip_sequence n) i = initial_coins n i ↔ i = ⟨n, sorry⟩ :=
begin
  sorry
end

end coin_flipping_l115_115271


namespace average_score_for_girls_combined_l115_115254

variables (C c D d : ℕ) -- Number of boys and girls at Clinton (C, c) and Dexter (D, d)
variables (avg_boys_clinton avg_boys_dexter avg_boys_combined : ℝ)
variables (avg_girls_clinton avg_girls_dexter : ℝ)
variables (avg_total_clinton avg_total_dexter : ℝ)

-- Given conditions
constants 
  (H1 : avg_boys_clinton = 73)
  (H2 : avg_boys_dexter = 85)
  (H3 : avg_boys_combined = 83)
  (H4 : avg_girls_clinton = 78)
  (H5 : avg_girls_dexter = 92)
  (H6 : avg_total_clinton = 76)
  (H7 : avg_total_dexter = 86)

-- Theorem statement
theorem average_score_for_girls_combined : 
  (∑ n in [c, d], n * avg_girls_clinton) / (c + d) = 83 :=
sorry

end average_score_for_girls_combined_l115_115254


namespace scaled_standard_deviation_l115_115334

variable (x : Fin 5 → ℝ)

def variance (x : Fin 5 → ℝ) : ℝ :=
  1 / 5 * (Finset.univ.sum fun i => (x i - (Finset.univ.sum fun i => x i) / 5) ^ 2)

def standard_deviation (x : Fin 5 → ℝ) : ℝ :=
  Real.sqrt (variance x)

theorem scaled_standard_deviation (h : variance x = 2) : standard_deviation (fun i => 2 * x i) = 2 * Real.sqrt 2 := 
  sorry

end scaled_standard_deviation_l115_115334


namespace misty_is_three_times_smaller_l115_115861

-- Define constants representing the favorite numbers of Misty and Glory
def G : ℕ := 450
def total_sum : ℕ := 600

-- Define Misty's favorite number in terms of the total sum and Glory's favorite number
def M : ℕ := total_sum - G

-- The main theorem stating that Misty's favorite number is 3 times smaller than Glory's favorite number
theorem misty_is_three_times_smaller : G / M = 3 := by
  -- Sorry placeholder indicating the need for further proof
  sorry

end misty_is_three_times_smaller_l115_115861


namespace total_pokemon_cards_l115_115456

-- Definitions based on conditions
def dozen := 12
def amount_per_person := 9 * dozen
def num_people := 4

-- Proposition to prove
theorem total_pokemon_cards :
  num_people * amount_per_person = 432 :=
by sorry

end total_pokemon_cards_l115_115456


namespace tangent_line_at_origin_max_min_on_interval_l115_115025

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_at_origin : 
  let tangent_line := 1
  in tangent_line = 1 :=
by
  sorry

theorem max_min_on_interval :
  let a := 0
  let b := Real.pi / 2
  let max_val := 1
  let min_val := -Real.pi / 2
  in max_val = 1 ∧ min_val = -Real.pi / 2 :=
by
  sorry

end tangent_line_at_origin_max_min_on_interval_l115_115025


namespace problem1_problem2_l115_115359

-- Problem 1 Equivalent Lean Statement
def f1 (x a : ℝ) : ℝ := x^2 - (a + 1) * x - 1

theorem problem1 (a : ℝ) (h₁ : ∃ x ∈ (set.Icc 2 3), f1 x a = 0) :
  1 / 2 ≤ a ∧ a ≤ 5 / 3 :=
sorry

-- Problem 2 Equivalent Lean Statement
def f2 (x a : ℝ) : ℝ := x^2 - (a + 1) * x + a

theorem problem2 (a x : ℝ) (h₁ : a = b) (h₂ : 2 ≤ a ∧ a ≤ 3) (h₃ : ∀ a ∈ set.Icc 2 3, f2 x a < 0) :
  1 < x ∧ x < 2 :=
sorry

end problem1_problem2_l115_115359


namespace exist_pos_integers_x_y_l115_115139

theorem exist_pos_integers_x_y (n : ℤ) (h : n ≥ 3) : 
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 2^n = 7 * x^2 + y^2 := by 
  sorry

end exist_pos_integers_x_y_l115_115139


namespace superhero_payments_l115_115498

-- Define constants and productivity
def productivity_superman (W : ℝ) : ℝ := 0.1 * W
def productivity_flash (W : ℝ) : ℝ := 2 * (productivity_superman W)
def combined_productivity (W : ℝ) : ℝ := (productivity_superman W) + (productivity_flash W)

-- Define the total work and the work completed together
def remaining_work (W : ℝ) : ℝ := 0.9 * W

-- Define payment formula
def payment (t : ℝ) : ℝ := 90 / t

-- Define times
def time_superman_before (W : ℝ) : ℝ := 1
def time_together (W : ℝ) : ℝ := remaining_work W / combined_productivity W
def total_time_superman (W : ℝ) : ℝ := time_superman_before W + time_together W
def total_time_flash (W : ℝ) : ℝ := time_together W

-- Define payments
def payment_superman (W : ℝ) : ℝ := payment (total_time_superman W)
def payment_flash (W : ℝ) : ℝ := payment (total_time_flash W)

theorem superhero_payments (W : ℝ) (h1 : W > 0) :
  payment_superman W = 22.5 ∧ payment_flash W = 30 :=
by
  -- Placeholder for the proof
  sorry

end superhero_payments_l115_115498


namespace common_difference_l115_115797

noncomputable def a : ℕ := 3
noncomputable def an : ℕ := 28
noncomputable def Sn : ℕ := 186

theorem common_difference (d : ℚ) (n : ℕ) (h1 : an = a + (n-1) * d) (h2 : Sn = n * (a + an) / 2) : d = 25 / 11 :=
sorry

end common_difference_l115_115797


namespace find_k_shelf_life_at_11_22_l115_115526

noncomputable def food_shelf_life (k b x : ℝ) : ℝ := Real.exp (k * x + b)

-- Given conditions
def condition1 : food_shelf_life k b 0 = 192 := by sorry
def condition2 : food_shelf_life k b 33 = 24 := by sorry

-- Prove that k = - (Real.log 2) / 11
theorem find_k (k b : ℝ) (h1 : food_shelf_life k b 0 = 192) (h2 : food_shelf_life k b 33 = 24) : 
  k = - (Real.log 2) / 11 :=
by sorry

-- Use the found value of k to determine the shelf life at 11°C and 22°C
theorem shelf_life_at_11_22 (k b : ℝ) (h1 : food_shelf_life k b 0 = 192) (h2 : food_shelf_life k b 33 = 24) 
  (hk : k = - (Real.log 2) / 11) : 
  food_shelf_life k b 11 = 96 ∧ food_shelf_life k b 22 = 48 :=
by sorry

end find_k_shelf_life_at_11_22_l115_115526


namespace min_value_eq_l115_115723

noncomputable def minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : log 2 x + log 8 y = log 2 2) : ℝ :=
  if h_min : x + 3 * y = 1 ∧ x = real.sqrt 3 * y then 2 * real.sqrt 3 + 4 else sorry

theorem min_value_eq :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ log 2 x + log 8 y = log 2 2 ∧ minimum_value x y (by linarith) (by linarith) (by linarith) = 2 * real.sqrt 3 + 4 :=
begin
  sorry,
end

end min_value_eq_l115_115723


namespace max_reallocatable_employees_range_of_values_for_a_l115_115231

variable {x : ℕ}
variable {a : ℝ}

def initial_profit_per_employee : ℝ := 100000
def employees : ℕ := 1000
def reallocated_profit_per_employee (a : ℝ) (x : ℕ) : ℝ := 100000 * (a - (3 * x / 500))
def remaining_profit_increase (x : ℕ) : ℝ := 1 + 0.2 * x / 100

theorem max_reallocatable_employees (hx : x > 0)
  (h_profit : remaining_profit_increase x * 1000 * (1000 - x) ≥ 100000 * 1000) :
  x ≤ 500 := by
  sorry

theorem range_of_values_for_a (hx : x > 0) (hx_le : x ≤ 500)
  (h_profit_remaining : remaining_profit_increase x * 1000 * (1000 - x) ≥ 100000 * 1000)
  (h_profit_tertiary : 100000 * (a - 3 * x / 500) * x ≤ 100000 * (1000 - x) * remaining_profit_increase x) :
  0 < a ∧ a ≤ 5 := by
  sorry

end max_reallocatable_employees_range_of_values_for_a_l115_115231


namespace phase_shift_of_sine_graph_l115_115302

-- Defining the variables and the phase shift calculation
def b : ℝ := 4
def c : ℝ := 3 * Real.pi / 2

theorem phase_shift_of_sine_graph : (c / b) = (3 * Real.pi / 8) :=
sorry

end phase_shift_of_sine_graph_l115_115302


namespace tangent_line_length_l115_115791

theorem tangent_line_length (O A B C D : Point)
  (h_tangents : Tangent AD O ∧ Tangent BC O ∧ Tangent CD O)
  (h_diameter : dist A B = 12)
  (h_AD : dist A D = 4) :
  dist B C = 9 :=
sorry -- Proof steps are omitted

end tangent_line_length_l115_115791


namespace describe_random_event_l115_115563

def idiom_A : Prop := "海枯石烂" = "extremely improbable or far into the future, not random"
def idiom_B : Prop := "守株待兔" = "represents a random event"
def idiom_C : Prop := "画饼充饥" = "unreal hopes, not random"
def idiom_D : Prop := "瓜熟蒂落" = "natural or expected outcome, not random"

theorem describe_random_event : idiom_B := 
by
  -- Proof omitted; conclusion follows from the given definitions
  sorry

end describe_random_event_l115_115563


namespace overall_gain_percentage_l115_115985

theorem overall_gain_percentage (CP1 CP2 CP3 SP1 SP2 SP3 : ℝ)
  (h_CP1 : CP1 = 100) (h_SP1 : SP1 = 115)
  (h_CP2 : CP2 = 150) (h_SP2 : SP2 = 180)
  (h_CP3 : CP3 = 200) (h_SP3 : SP3 = 220) : 
  ((SP1 - CP1 + (SP2 - CP2) + (SP3 - CP3)) / (CP1 + CP2 + CP3) * 100) ≈ 14.44 :=
by 
  sorry

end overall_gain_percentage_l115_115985


namespace fraction_comparison_l115_115667

theorem fraction_comparison (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x^2 - y^2) / (x - y) > (x^2 + y^2) / (x + y) :=
by
  sorry

end fraction_comparison_l115_115667


namespace find_interest_rate_l115_115657

-- Define the given conditions
def initial_investment : ℝ := 2200
def additional_investment : ℝ := 1099.9999999999998
def total_investment : ℝ := initial_investment + additional_investment
def desired_income : ℝ := 0.06 * total_investment
def income_from_additional_investment : ℝ := 0.08 * additional_investment
def income_from_initial_investment (r : ℝ) : ℝ := initial_investment * r

-- State the proof problem
theorem find_interest_rate (r : ℝ) 
    (h : desired_income = income_from_additional_investment + income_from_initial_investment r) :
    r = 0.05 :=
sorry

end find_interest_rate_l115_115657


namespace count_three_digit_numbers_with_4_and_5_l115_115774

theorem count_three_digit_numbers_with_4_and_5 : 
  let numbers := { num : ℕ | 100 ≤ num ∧ num ≤ 999 } in
  ∃ N, N = 50 ∧ finset.card ({ num ∈ numbers | (∃ i j : ℕ, num.digits 10 !i = 4 ∧ num.digits 10 !j = 5)} : finset ℕ) = N :=
by
  sorry

end count_three_digit_numbers_with_4_and_5_l115_115774


namespace gcm_less_than_90_l115_115201

theorem gcm_less_than_90 (a b : ℕ) (h1 : a = 8) (h2 : b = 12) : 
  ∃ x : ℕ, x < 90 ∧ ∀ y : ℕ, y < 90 → (a ∣ y) ∧ (b ∣ y) → y ≤ x → x = 72 :=
sorry

end gcm_less_than_90_l115_115201


namespace father_son_speed_ratio_l115_115611

theorem father_son_speed_ratio
  (F S t : ℝ)
  (distance_hallway : ℝ)
  (distance_meet_from_father : ℝ)
  (H1 : distance_hallway = 16)
  (H2 : distance_meet_from_father = 12)
  (H3 : 12 = F * t)
  (H4 : 4 = S * t)
  : F / S = 3 := by
  sorry

end father_son_speed_ratio_l115_115611


namespace original_painting_width_l115_115987

theorem original_painting_width {W : ℝ} 
  (orig_height : ℝ) (print_height : ℝ) (print_width : ℝ)
  (h1 : orig_height = 10) 
  (h2 : print_height = 25)
  (h3 : print_width = 37.5) :
  W = 15 :=
  sorry

end original_painting_width_l115_115987


namespace common_divisors_9240_6300_l115_115371

def num_common_divisors (a b : ℕ) : ℕ :=
  (Nat.gcd a b).divisors.count

theorem common_divisors_9240_6300 : num_common_divisors 9240 6300 = 12 := by
  sorry

end common_divisors_9240_6300_l115_115371


namespace find_m_value_l115_115761

theorem find_m_value (m : ℝ) (h1 : 2 ∈ ({0, m, m^2 - 3 * m + 2} : set ℝ)) : m = 3 :=
sorry

end find_m_value_l115_115761


namespace side_length_of_hexagonal_bottom_l115_115923

noncomputable def volume_of_hexagonal_prism (a : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2 * a^2) * 3 * Real.sqrt 3 * a

theorem side_length_of_hexagonal_bottom (a : ℝ) (h1 : volume_of_hexagonal_prism a = 108) :
  a = 2 / Real.cbrt 3 :=
by
  sorry

end side_length_of_hexagonal_bottom_l115_115923


namespace rearrange_figure_to_square_l115_115091

def figure : ℕ := 18

theorem rearrange_figure_to_square (n : ℕ) (h : n = 18) : 
  ∃ (a b c : ℕ), (a + b + c = n) ∧  
  (let side := (Real.sqrt (n : ℝ)) in 
    side - side.floor ≤ side.ceil - side) :=
sorry

end rearrange_figure_to_square_l115_115091


namespace inequlity_proof_l115_115068

theorem inequlity_proof (a b : ℝ) : a^2 + a * b + b^2 ≥ 3 * (a + b - 1) := 
  sorry

end inequlity_proof_l115_115068


namespace subset_with_6_and_1_or_2_count_l115_115773

open Finset

theorem subset_with_6_and_1_or_2_count : 
  let s := {1, 2, 3, 4, 5, 6} : Finset ℕ in
  (s.powerset.filter (λ t, 6 ∈ t ∧ (1 ∈ t ∨ 2 ∈ t))).card = 24 :=
by
  sorry

end subset_with_6_and_1_or_2_count_l115_115773


namespace complex_magnitude_of_quadratic_solution_l115_115051

open Complex -- Open the complex number module

-- Define the theorem
theorem complex_magnitude_of_quadratic_solution (z : ℂ) (h : z^2 - z + 1 = 0) : |z| = 1 :=
sorry

end complex_magnitude_of_quadratic_solution_l115_115051


namespace Ryan_problem_l115_115473

-- Define the dubious function δ
noncomputable def δ : ℕ → ℕ
| 1 => 1
| n => ∑ d in (Finset.filter (λ d => d ≠ n) (Finset.divisors n)), δ d

-- Define the infinite series
noncomputable def series_sum (f : ℕ → ℕ) (m : ℕ) :=
∑' k : ℕ, (f m ^ k) / (m ^ k)

-- Define the proof goal
theorem Ryan_problem : ∃ p q : ℕ, Nat.coprime p q ∧ 1000 * p + q = 14013 ∧ series_sum δ 15 = p / q :=
by
  sorry

end Ryan_problem_l115_115473


namespace exist_transversal_parallel_direction_exist_transversal_parallel_axis_l115_115253

-- Definitions of skew lines and transversal line
variables (L₁ L₂ : Set (ℝ × ℝ))
def are_skew_lines (L₁ L₂ : Set (ℝ × ℝ)) : Prop := 
  ¬ ∃ P, P ∈ L₁ ∧ P ∈ L₂ ∧ ¬ (∃ k, L₁ = {P + k * v | k ∈ ℝ} ∧ L₂ = {P + k * w | k ∈ ℝ} ∧ v ≠ w)

def is_transversal (L₁ L₂ L₃ : Set (ℝ × ℝ)) : Prop := 
  ∃ P₁, P₁ ∈ L₁ ∧ P₁ ∈ L₃ ∧ ∃ P₂, P₂ ∈ L₂ ∧ P₂ ∈ L₃

-- Problem: proving the existence of transversal lines with given conditions
theorem exist_transversal_parallel_direction (L₁ L₂ : Set (ℝ × ℝ)) (d : ℝ × ℝ)
  (h_skew : are_skew_lines L₁ L₂) : 
  ∃ L₃, is_transversal L₁ L₂ L₃ ∧ ∀ P ∈ L₃, ∃ k : ℝ, P = (fst (P) + k * fst d, snd (P) + k * snd d) :=
sorry

theorem exist_transversal_parallel_axis (L₁ L₂ : Set (ℝ × ℝ)) (axis : ℝ × ℝ)
  (h_skew : are_skew_lines L₁ L₂) : 
  ∃ L₃, is_transversal L₁ L₂ L₃ ∧ ∀ P ∈ L₃, ∃ k : ℝ, P = (fst P + k * fst axis, snd P + k * snd axis) :=
sorry

end exist_transversal_parallel_direction_exist_transversal_parallel_axis_l115_115253


namespace find_new_person_weight_l115_115956

variables {W : ℕ}
variables {a b c d : ℕ}

-- Conditions
def average_increase (avg_increase : ℤ) : Prop :=
  avg_increase = 3

def replace_weight (replaced_weight : ℤ) : Prop :=
  replaced_weight = 70

def new_weight (new_person_weight : ℤ) (increase : ℤ) (replaced_weight : ℤ) : Prop :=
  new_person_weight = replaced_weight + increase

-- Theorem statement
theorem find_new_person_weight 
  (avg_increase_cond : average_increase 3)
  (replace_weight_cond : replace_weight 70)
  (total_increase : 4 * 3 = 12):
  ∃ W, new_weight W 12 70 :=
begin
  use 82,
  rw [← replace_weight_cond, ← avg_increase_cond],
  transitivity,
  exact total_increase,
  linarith,
end

end find_new_person_weight_l115_115956


namespace median_length_correct_l115_115627

-- Define the total count of names per length
def names_counts : List ℕ := [6, 4, 1, 5, 5]

-- Define the corresponding lengths
def names_lengths : List ℕ := [4, 5, 6, 7, 8]

-- A helper function to calculate the median of a list of counts and lengths
def median_length (counts : List ℕ) (lengths : List ℕ) : ℕ :=
  let position := (List.sum counts + 1) / 2
  let cumulative_sum := List.scanl (+) 0 counts
  lengths[(List.index_where (λ sum, sum ≥ position) cumulative_sum).getD 0]

-- Theorem to prove the median length given the counts and lengths
theorem median_length_correct :
  median_length names_counts names_lengths = 6 :=
by
  -- Proof goes here
  sorry

end median_length_correct_l115_115627


namespace marble_arrangement_l115_115287

theorem marble_arrangement :
  let m := 16 in
  let N := (nat.choose (15, 7)) in
  N % 1000 = 435 :=
by
  -- Let m be the maximum number of red marbles
  let m := 16
  -- The number of arrangements N
  let N := nat.choose 15 7
  -- The expected result after taking modulo 1000
  show N % 1000 = 435
  sorry

end marble_arrangement_l115_115287


namespace distance_A_to_BC_radius_and_side_length_l115_115090

variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variables (α β γ : ℝ)
variables (l : line) (Ω : circle)

-- Given conditions
axiom angle_A_eq_two_angle_C : ∠A = 2 * ∠C
axiom tangent_B : ∀ x, x ∈ l ↔ B ∈ Ω
axiom dist_A_to_tangent : distance(A, l) = 4
axiom dist_C_to_tangent : distance(C, l) = 9

-- Proof problem statements

-- Prove distance from point A to line BC
theorem distance_A_to_BC : distance(A, BC) = 5 :=
sorry

-- Prove radius of the circumcircle Ω and the length of side AB
theorem radius_and_side_length (R AB : ℝ) : 
  R = 32 / 7 ∧ AB = 16 / real.sqrt 7 :=
sorry

end distance_A_to_BC_radius_and_side_length_l115_115090


namespace vectors_collinear_if_magnitude_sum_eq_sum_magnitudes_l115_115647

variables (a b : ℝ^3) -- assuming we are in 3-dimensional real vector space

theorem vectors_collinear_if_magnitude_sum_eq_sum_magnitudes 
  (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ∥a + b∥ = ∥a∥ + ∥b∥) :
  ∃ k : ℝ, a = k • b := 
sorry

end vectors_collinear_if_magnitude_sum_eq_sum_magnitudes_l115_115647


namespace cereal_eating_tournament_l115_115128

def num_players := 5
def num_games := (num_players * (num_players - 1)) / 2
def total_outcomes := 2 ^ num_games
def favorable_outcomes := num_players * (2 ^ (num_games - num_players + 1))
def prob_no_wins_all := (total_outcomes - favorable_outcomes) / total_outcomes
def prob_frac := Rat.mk_nat (total_outcomes - favorable_outcomes) total_outcomes
def m := prob_frac.num.nat_abs
def n := prob_frac.denom

theorem cereal_eating_tournament: 100 * m + n = 1116 :=
by
  -- Proof would go here
  sorry

end cereal_eating_tournament_l115_115128


namespace professors_seating_l115_115288

theorem professors_seating :
  let chairs := 11
  let professors := 3
  let students := 8
  let suitable_positions := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  let effective_positions := 6  -- Since each choice constraints the following choice
  (nat.choose effective_positions professors) * (nat.factorial professors) = 120 :=
by {
    let chairs := 11
    let professors := 3
    let students := 8
    let suitable_positions := {2, 3, 4, 5, 6, 7, 8, 9, 10}
    let effective_positions := 6
    have h1: (nat.choose effective_positions professors) = 20 := sorry, -- Calculation of combinations
    have h2: (nat.factorial professors) = 6 := sorry, -- Calculation of permutations
    have h3: 20 * 6 = 120 := by norm_num,
    exact h3,
  }

end professors_seating_l115_115288


namespace simplify_inv_sum_l115_115561

variables {x y z : ℝ}

theorem simplify_inv_sum (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = xyz / (yz + xz + xy) :=
by
  sorry

end simplify_inv_sum_l115_115561


namespace a4_value_l115_115333

def seq : ℕ → ℤ
| 1       := 1
| (n + 1) := seq n - n

theorem a4_value : seq 4 = -5 :=
by sorry

end a4_value_l115_115333


namespace all_lights_on_l115_115887

def light_on (n : ℕ) : Prop := sorry

axiom light_rule_1 (k : ℕ) (hk: light_on k): light_on (2 * k) ∧ light_on (2 * k + 1)
axiom light_rule_2 (k : ℕ) (hk: ¬ light_on k): ¬ light_on (4 * k + 1) ∧ ¬ light_on (4 * k + 3)
axiom light_2023_on : light_on 2023

theorem all_lights_on (n : ℕ) (hn : n < 2023) : light_on n :=
by sorry

end all_lights_on_l115_115887


namespace sally_picked_3_plums_l115_115127

theorem sally_picked_3_plums (melanie_picked : ℕ) (dan_picked : ℕ) (total_picked : ℕ) 
    (h1 : melanie_picked = 4) (h2 : dan_picked = 9) (h3 : total_picked = 16) : 
    total_picked - (melanie_picked + dan_picked) = 3 := 
by 
  -- proof steps go here
  sorry

end sally_picked_3_plums_l115_115127


namespace club_president_vice_president_l115_115135

theorem club_president_vice_president :
  let total_members := 24
  let boys := 14
  let girls := 10
  let team_A_boys := 8
  let team_A_girls := 6
  let team_B_boys := 6
  let team_B_girls := 4
  total_members = boys + girls ∧
  team_A_boys + team_A_girls + team_B_boys + team_B_girls = total_members →
  (team_A_boys * team_B_girls + team_B_boys * team_A_girls + team_A_girls * team_B_boys + team_B_girls * team_A_boys) = 136 :=
by {
  intros,
  sorry
}

end club_president_vice_president_l115_115135


namespace sequence_integers_l115_115525

theorem sequence_integers (a : ℕ → ℤ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 1) 
  (h₃ : ∀ n, n ≥ 3 → a n = (a (n-1)) ^ 2 + 2 / a (n-2)) : 
  ∀ n, ∃ k : ℤ, a n = k := 
by 
  sorry

end sequence_integers_l115_115525


namespace age_problem_l115_115398

noncomputable def A_age (B_age : ℕ) := B_age + 9

theorem age_problem (B_age : ℕ) (X : ℕ) 
  (h1 : B_age = 39) 
  (h2 : let A := A_age B_age in A + 10 = 2 * (B_age - X)) :
  X = 10 :=
by
  intro h1 h2
  rw [h1] at h2
  have hA : A_age 39 = 48 :=
    by
      rw [A_age, h1]
  rw [hA] at h2
  have h_eq : 48 + 10 = 2 * (39 - X) := h2
  linarith

end age_problem_l115_115398


namespace line_intersects_circle_two_points_l115_115357

theorem line_intersects_circle_two_points (k : ℝ) :
  let C := (x-1)^2 + (y-3)^2 = 25 in
  let l := (1 + 3*k) * x + (3 - 2*k) * y + 4*k - 17 = 0 in
  ∃ p₁ p₂ : ℝ × ℝ, p₁ ≠ p₂ ∧ (C p₁ ∧ l p₁) ∧ (C p₂ ∧ l p₂) :=
sorry

end line_intersects_circle_two_points_l115_115357


namespace range_of_a_l115_115022

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 9 ^ x - 2 * 3 ^ x + a - 3 > 0) → a > 4 :=
by
  sorry

end range_of_a_l115_115022


namespace sequence_positive_integers_sequence_perfect_square_l115_115524

inductive Sequence : Type
| mk : (ℕ → ℤ) → Sequence

def Sequence.a_n (s : Sequence) (n : ℕ) : ℤ :=
  match s with
  | Sequence.mk f => f n

def sequence_def : Sequence :=
  Sequence.mk (λ n,
    match n with
    | 0 => 1
    | _ => (let a_n := Sequence.a_n sequence_def (n - 1) in
            (7 * a_n + int.sqrt(45 * a_n ^ 2 - 36)) / 2)
  )

theorem sequence_positive_integers (s : Sequence) (h : s = sequence_def) :
  ∀ n : ℕ, Sequence.a_n s n > 0 :=
by
  sorry

theorem sequence_perfect_square (s : Sequence) (h : s = sequence_def) :
  ∀ n : ℕ, ∃ k : ℤ, Sequence.a_n s n * Sequence.a_n s (n + 1) - 1 = k ^ 2 :=
by
  sorry

end sequence_positive_integers_sequence_perfect_square_l115_115524


namespace total_students_l115_115980

theorem total_students (boys girls : ℕ) (h_boys : boys = 127) (h_girls : girls = boys + 212) : boys + girls = 466 :=
by
  sorry

end total_students_l115_115980


namespace work_completion_days_l115_115643

theorem work_completion_days 
  (x : ℕ) 
  (h1 : ∀ t : ℕ, t = x → A_work_rate = 2 * (1 / t))
  (h2 : A_and_B_work_together : ∀ d : ℕ, d = 4 → A_B_combined_rate = 1 / d) :
  x = 12 := 
sorry

end work_completion_days_l115_115643


namespace probability_interval_l115_115911

theorem probability_interval (P_A P_B p : ℝ) (hP_A : P_A = 2 / 3) (hP_B : P_B = 3 / 5) :
  4 / 15 ≤ p ∧ p ≤ 3 / 5 := sorry

end probability_interval_l115_115911


namespace negation_universal_proposition_l115_115516

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negation_universal_proposition_l115_115516


namespace find_r_over_s_at_0_l115_115904

noncomputable def r (x : ℝ) : ℝ := -3 * (x + 1) * (x - 2)
noncomputable def s (x : ℝ) : ℝ := (x + 1) * (x - 3)

theorem find_r_over_s_at_0 : (r 0) / (s 0) = 2 := by
  sorry

end find_r_over_s_at_0_l115_115904


namespace total_students_is_17_l115_115794

def total_students_in_class (students_liking_both_baseball_football : ℕ)
                             (students_only_baseball : ℕ)
                             (students_only_football : ℕ)
                             (students_liking_basketball_as_well : ℕ)
                             (students_liking_basketball_and_football_only : ℕ)
                             (students_liking_all_three : ℕ)
                             (students_liking_none : ℕ) : ℕ :=
  students_liking_both_baseball_football -
  students_liking_all_three +
  students_only_baseball +
  students_only_football +
  students_liking_basketball_and_football_only +
  students_liking_all_three +
  students_liking_none +
  (students_liking_basketball_as_well -
   (students_liking_all_three +
    students_liking_basketball_and_football_only))

theorem total_students_is_17 :
    total_students_in_class 7 3 4 2 1 2 5 = 17 :=
by sorry

end total_students_is_17_l115_115794


namespace homework_duration_equation_l115_115194

-- Given conditions
def initial_duration : ℝ := 120
def final_duration : ℝ := 60
variable (x : ℝ)

-- The goal is to prove that the appropriate equation holds
theorem homework_duration_equation : initial_duration * (1 - x)^2 = final_duration := 
sorry

end homework_duration_equation_l115_115194


namespace tan_of_angle_in_fourth_quadrant_l115_115348

theorem tan_of_angle_in_fourth_quadrant (α : ℝ) (h1 : Real.sin α = -5 / 13) (h2 : α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) :
  Real.tan α = -5 / 12 :=
sorry

end tan_of_angle_in_fourth_quadrant_l115_115348


namespace math_problem_l115_115362

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := 1
def f (x : ℝ) : ℝ := a * sin x + 2 * b
def g (x : ℝ) : ℝ := b * sin (a * x)

theorem math_problem :
  (∀ x, f x ≤ 4) ∧ (∀ x, f x ≥ 0) ∧ a > 0 →
  (a = 2 ∧ b = 1 ∧ a + b = 3 ∧ (∃ T > 0, ∀ x, g (x + T) = g x ∧ T = π)) :=
by
  sorry

end math_problem_l115_115362


namespace triangle_least_perimeter_l115_115180

def least_possible_perimeter (a b x : ℕ) (h1 : a = 24) (h2 : b = 37) (h3 : x > 13) (h4 : x < 61) : Prop :=
  a + b + x = 75

theorem triangle_least_perimeter : ∃ x : ℕ, least_possible_perimeter 24 37 x
  :=
sorry

end triangle_least_perimeter_l115_115180


namespace total_candidates_l115_115493

theorem total_candidates (N : ℕ)
  (h_avg_all : (∀ marks : list ℤ, (list.sum marks) / marks.length = 48))
  (h_avg_first10 : (∀ marks : list ℤ, list.length marks = 10 → (list.sum marks) / 10 = 55))
  (h_avg_last11 : (∀ marks : list ℤ, list.length marks = 11 → (list.sum marks) / 11 = 40))
  (h_11th_candidate : 66) :
  N = 21 :=
by
  sorry

end total_candidates_l115_115493


namespace probability_of_correct_number_l115_115857

/--
Max has misplaced Sheila’s phone number. He remembers that the first three digits are 504, 507, or 509.
The remaining five digits are 2, 3, 5, 8, and 9, but he is uncertain of their order.
If Max randomly dials an eight-digit number that meets these conditions, what is the probability that
he dials Sheila’s correct number?
-/
theorem probability_of_correct_number :
  let first_three_digits := {504, 507, 509}
  let last_five_digits := {2, 3, 5, 8, 9}
  let total_possibilities := 3 * fact 5
  let correct_number := 1
  correct_number.to_rat / total_possibilities.to_rat = (1:ℚ) / 360 := 
by sorry

end probability_of_correct_number_l115_115857


namespace cos_difference_l115_115291

theorem cos_difference (a b : ℝ) : 
  cos (a + b) - cos (a - b) = -2 * sin a * sin b := sorry

end cos_difference_l115_115291


namespace adults_eat_one_third_l115_115790

theorem adults_eat_one_third (n c k : ℕ) (hn : n = 120) (hc : c = 4) (hk : k = 20) :
  ((n - c * k) / n : ℚ) = 1 / 3 :=
by
  sorry

end adults_eat_one_third_l115_115790


namespace value_of_f_f_f_2_l115_115439

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem value_of_f_f_f_2 : f (f (f 2)) = 2 :=
by {
  sorry
}

end value_of_f_f_f_2_l115_115439


namespace Derrick_yard_length_l115_115684

variables (Alex_yard Derrick_yard Brianne_yard Carla_yard Derek_yard : ℝ)

-- Given conditions as hypotheses
theorem Derrick_yard_length :
  (Alex_yard = Derrick_yard / 2) →
  (Brianne_yard = 6 * Alex_yard) →
  (Carla_yard = 3 * Brianne_yard + 5) →
  (Derek_yard = Carla_yard / 2 - 10) →
  (Brianne_yard = 30) →
  Derrick_yard = 10 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Derrick_yard_length_l115_115684


namespace problem_solution_count_l115_115218

noncomputable def numSolutions (a : ℝ) : ℝ :=
  if a < 0 then
    4
  else
    0

theorem problem_solution_count (a : ℝ) (h : a < 0) :
  ∃ x : ℝ, x ∈ Ioo (-π) π ∧ (a-1)*(sin(2*x) + cos(x)) + (a-1)*(sin(x) - cos(2*x)) = 0 ->
  numSolutions a = 4 :=
by 
  sorry

end problem_solution_count_l115_115218


namespace rectangle_semicircle_problem_l115_115874

/--
Rectangle ABCD and a semicircle with diameter AB are coplanar and have nonoverlapping interiors.
Let R denote the region enclosed by the semicircle and the rectangle.
Line ℓ meets the semicircle, segment AB, and segment CD at distinct points P, V, and S, respectively.
Line ℓ divides region R into two regions with areas in the ratio 3:1.
Suppose that AV = 120, AP = 180, and VB = 240.
Prove the length of DA = 90 * sqrt(6).
-/
theorem rectangle_semicircle_problem (DA : ℝ) (AV AP VB : ℝ) (h₁ : AV = 120) (h₂ : AP = 180) (h₃ : VB = 240) :
  DA = 90 * Real.sqrt 6 := by
  sorry

end rectangle_semicircle_problem_l115_115874


namespace problem_solution_l115_115590

def z (xi yi : ℕ) : ℕ := xi - yi

def z_vals (x y : Fin 10 → ℕ) : Fin 10 → ℕ := fun i => z (x i) (y i)

def mean (z : Fin 10 → ℕ) : ℕ :=
  (∑ i in Finset.univ, z i) / 10

def variance (z : Fin 10 → ℕ) (mean_z : ℕ) : ℕ :=
  (∑ i in Finset.univ, (z i - mean_z)^2) / 10

def significant_improvement (mean_z : ℕ) (var_z : ℕ) : Prop :=
  mean_z >= 2 * Real.sqrt (var_z / 10)

-- Given data
def x : Fin 10 → ℕ
| 0 => 545
| 1 => 533
| 2 => 551
| 3 => 522
| 4 => 575
| 5 => 544
| 6 => 541
| 7 => 568
| 8 => 596
| 9 => 548

def y : Fin 10 → ℕ
| 0 => 536
| 1 => 527
| 2 => 543
| 3 => 530
| 4 => 560
| 5 => 533
| 6 => 522
| 7 => 550
| 8 => 576
| 9 => 536

/-- The final proof statements -/
theorem problem_solution :
  let z_vals := z_vals x y
  let mean_z := mean z_vals
  let var_z := variance z_vals mean_z
  mean_z = 11 ∧ var_z = 61 ∧ significant_improvement mean_z var_z := 
by
  sorry

end problem_solution_l115_115590


namespace distance_from_integer_l115_115445

theorem distance_from_integer (a : ℝ) (h : a > 0) (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, ∃ m : ℕ, 1 ≤ m ∧ m < n ∧ abs (m * a - k) ≤ (1 / n) :=
by
  sorry

end distance_from_integer_l115_115445


namespace linear_equation_solution_l115_115914

theorem linear_equation_solution (m : ℝ) (x : ℝ) (h : |m| - 2 = 1) (h_ne : m ≠ 3) :
  (2 * m - 6) * x^(|m|-2) = m^2 ↔ x = -(3/4) :=
by
  sorry

end linear_equation_solution_l115_115914


namespace problem_1_calculate_problem_2_simplify_l115_115219

-- Problem 1: Calculate |-5| + (3 - √2)^0 - 2 * tan 45°
theorem problem_1_calculate : abs (-5) + (3 - real.sqrt 2)^0 - 2 * real.tan (real.pi / 4) = 4 :=
by
  sorry

-- Problem 2: Simplify a / (a^2 - 9) ÷ (1 + 3 / (a - 3)) given a ≠ 3 and a ≠ -3
theorem problem_2_simplify (a : ℝ) (h₀ : a ≠ 3) (h₁ : a ≠ -3) :
  a / (a^2 - 9) / (1 + 3 / (a - 3)) = 1 / (a + 3) :=
by
  sorry

end problem_1_calculate_problem_2_simplify_l115_115219


namespace remainder_of_22_divided_by_3_l115_115463

theorem remainder_of_22_divided_by_3 : ∃ (r : ℕ), 22 = 3 * 7 + r ∧ r = 1 := by
  sorry

end remainder_of_22_divided_by_3_l115_115463


namespace min_value_is_1_l115_115515
noncomputable def min_val_function : ℝ :=
  minimalValue (λ x : ℝ, sin x + sqrt 3 * cos x) 0 (π / 2)

theorem min_value_is_1 : min_val_function = 1 := 
  sorry

end min_value_is_1_l115_115515


namespace max_f_value_l115_115678

noncomputable def f (x : ℝ) : ℝ := min (3 * x + 1) (min (- (4 / 3) * x + 3) ((1 / 3) * x + 9))

theorem max_f_value : ∃ x : ℝ, f x = 31 / 13 :=
by 
  sorry

end max_f_value_l115_115678


namespace volume_of_wedge_is_151_l115_115587

-- Define the conditions of the problem
def cylinder_height : ℝ := 9
def cylinder_radius : ℝ := 4

-- Define the problem statement
theorem volume_of_wedge_is_151 :
  let V := π * cylinder_radius^2 * cylinder_height in
  let V_wedge := (1 / 3) * V in
  V_wedge ≈ 151 :=
by
  sorry

end volume_of_wedge_is_151_l115_115587


namespace product_of_five_consecutive_integers_not_perfect_square_l115_115145

theorem product_of_five_consecutive_integers_not_perfect_square (n : ℕ) : 
  ¬ ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = k * k :=
by {
  sorry
}

end product_of_five_consecutive_integers_not_perfect_square_l115_115145


namespace find_prob_real_roots_l115_115991

-- Define the polynomial q(x)
def q (a : ℝ) (x : ℝ) : ℝ := x^4 + 3*a*x^3 + (3*a - 5)*x^2 + (-6*a + 4)*x - 3

-- Define the conditions for a to ensure all roots of the polynomial are real
noncomputable def all_roots_real_condition (a : ℝ) : Prop :=
  a ≤ -1/3 ∨ 1 ≤ a

-- Define the probability that given a in the interval [-12, 32] all q's roots are real
noncomputable def probability_real_roots : ℝ :=
  let total_length := 32 - (-12)
  let excluded_interval_length := 1 - (-1/3)
  let valid_interval_length := total_length - excluded_interval_length
  valid_interval_length / total_length

-- State the theorem
theorem find_prob_real_roots :
  probability_real_roots = 32 / 33 :=
sorry

end find_prob_real_roots_l115_115991


namespace max_groups_l115_115873

def eggs : ℕ := 20
def marbles : ℕ := 6
def eggs_per_group : ℕ := 5
def marbles_per_group : ℕ := 2

def groups_of_eggs := eggs / eggs_per_group
def groups_of_marbles := marbles / marbles_per_group

theorem max_groups (h1 : eggs = 20) (h2 : marbles = 6) 
                    (h3 : eggs_per_group = 5) (h4 : marbles_per_group = 2) : 
                    min (groups_of_eggs) (groups_of_marbles) = 3 :=
by
  sorry

end max_groups_l115_115873


namespace triangle_Y_sum_36_l115_115927

variable {A B C P Q R Y : Type}
variable {triangle_ABC : Triangle}
variable {circumcircle_BPQ : Circle}
variable {circumcircle_CRQ : Circle}

-- Conditions
def is_midpoint (a b m : Point) : Prop := dist a m = dist b m
def circumcircle_intersection (X Y : Circle) (P Q : Point) : Prop := P ≠ Q ∧ (P ∈ X.points ∧ P ∈ Y.points) ∧ (Q ∈ X.points ∧ Q ∈ Y.points)
def dist_eq_36 (a b c : Point) : Prop := dist a b + dist a c + dist b c = 36

-- Triangle ABC with specific side lengths
def triangle_ABC (A B C : Point) (AB BC AC : ℝ) : Prop :=
  dist A B = AB ∧ dist B C = BC ∧ dist A C = AC

-- Points P, Q, R being midpoints
def midpoints (A B C P Q R : Point) : Prop :=
  is_midpoint A B P ∧ is_midpoint B C Q ∧ is_midpoint A C R

-- Intersection of circumcircles at point Y
def circumcircle_intersection_Y (Q Y : Point) (circABC : Circle) (circDEF : Circle) : Prop :=
  circumcircle_intersection circABC circDEF Q Y
  
-- Full problem statement
theorem triangle_Y_sum_36 
  (A B C P Q R Y : Point) 
  (hABC : triangle_ABC A B C 15 16 17) 
  (hMid : midpoints A B C P Q R) 
  (hCircInter : circumcircle_intersection_Y Q Y circumcircle_BPQ circumcircle_CRQ) 
: dist_eq_36 Y A B C := 
sorry

end triangle_Y_sum_36_l115_115927


namespace rectangular_field_area_l115_115867

theorem rectangular_field_area 
  (W : ℝ) (D : ℝ)
  (hW : W = 14) 
  (hD : D = 17)
  (L : ℝ)
  (hPythagorean : D^2 = L^2 + W^2) 
  (hL : L = Real.sqrt (D^2 - W^2)) :
  W * L ≈ 134.96 := sorry

end rectangular_field_area_l115_115867


namespace reward_function_conditions_l115_115641

theorem reward_function_conditions :
  (∀ x : ℝ, (10 ≤ x ∧ x ≤ 1000) → (y = x / 150 + 2 → y ≤ 90 ∧ y ≤ x / 5) → False) ∧
  (∃ a : ℕ, (∀ x : ℝ, (10 ≤ x ∧ x ≤ 1000) → (y = (10 * x - 3 * a) / (x + 2) → y ≤ 9 ∧ y ≤ x / 5)) ∧ (a = 328)) :=
by
  sorry

end reward_function_conditions_l115_115641


namespace younger_brother_height_l115_115172

theorem younger_brother_height
  (O Y : ℕ)
  (h1 : O - Y = 12)
  (h2 : O + Y = 308) :
  Y = 148 :=
by
  sorry

end younger_brother_height_l115_115172


namespace wire_cut_l115_115586

theorem wire_cut (x : ℝ) (h₁ : x + (5/2) * x = 14) : x = 4 :=
by
  sorry

end wire_cut_l115_115586


namespace find_x_l115_115037

noncomputable def vector_a : ℝ × ℝ := (-2, 3)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)
noncomputable def vector_c (x : ℝ) : ℝ × ℝ := (2 * (-2), 3) + (x, -2)

theorem find_x (x : ℝ) (h : (vector_a.1 * vector_c x.1 + vector_a.2 * vector_c x.2) = 0) : x = 10 :=
sorry

end find_x_l115_115037


namespace zero_point_in_interval_l115_115013

noncomputable def f (x : ℝ) : ℝ := Math.log x - (1 / 2)^(x - 2)

theorem zero_point_in_interval :
  ∃ x₀, f x₀ = 0 ∧ 2 < x₀ ∧ x₀ < 3 :=
sorry

end zero_point_in_interval_l115_115013


namespace length_of_la_l115_115554

variables {A b c l_a: ℝ}
variables (S_ABC S_ACA' S_ABA': ℝ)

axiom area_of_ABC: S_ABC = (1 / 2) * b * c * Real.sin A
axiom area_of_ACA: S_ACA' = (1 / 2) * b * l_a * Real.sin (A / 2)
axiom area_of_ABA: S_ABA' = (1 / 2) * c * l_a * Real.sin (A / 2)
axiom sin_double_angle: Real.sin A = 2 * Real.sin (A / 2) * Real.cos (A / 2)

theorem length_of_la :
  l_a = (2 * b * c * Real.cos (A / 2)) / (b + c) :=
sorry

end length_of_la_l115_115554


namespace river_depth_ratio_l115_115804

-- Definitions based on the conditions
def depthMidMay : ℝ := 5
def increaseMidJune : ℝ := 10
def depthMidJune : ℝ := depthMidMay + increaseMidJune
def depthMidJuly : ℝ := 45

-- The theorem based on the question and correct answer
theorem river_depth_ratio : depthMidJuly / depthMidJune = 3 := by 
  -- Proof skipped for illustration purposes
  sorry

end river_depth_ratio_l115_115804


namespace molecular_weight_compound_l115_115202

def molecular_weight (n_atoms N O C H D: ℕ) (w_N w_O w_C w_H w_D: ℝ) : ℝ :=
  n_atoms * w_N + N * w_O + O * w_C + C * w_H + H * w_D + D

theorem molecular_weight_compound :
  molecular_weight 2 3 4 2 1 14.01 16.00 12.01 1.01 2.01 = 128.09 :=
by
  sorry

end molecular_weight_compound_l115_115202


namespace range_g_area_triangle_l115_115752

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 4 * sin x * (sin ((x / 2) + (Real.pi / 4)))^2

noncomputable def g (x : ℝ) : ℝ := 1 + 2 * sin (2 * x - Real.pi / 3)

theorem range_g (x : ℝ) (hx : x ∈ Set.Icc (Real.pi / 12) (Real.pi / 2)) :
  Set.Icc 0 3 = Set.range (g) := 
sorry

noncomputable def f_of_A (A : ℝ) : ℝ := sqrt 2 + 1

structure triangle :=
(A B C a b c : ℝ)
(b_pos : b = 2)
(fA_eq : f_of_A A = sqrt 2 + 1)
(sin_cond : sqrt 3 * a = 2 * b * sin A)
(B_bound : B ∈ Set.Ioo 0 (Real.pi / 2))

noncomputable def area (t : triangle): ℝ :=
1 / 2 * t.a * t.b * sin t.C 

theorem area_triangle (t : triangle) : 
  area t = (3 + sqrt 3) / 3 := 
sorry

end range_g_area_triangle_l115_115752


namespace area_of_triangle_DEF_l115_115229

theorem area_of_triangle_DEF (r1 r2 : ℝ) (DE DF : ℝ) (DEcongDF : DE = DF) (tangent1 tangent2 : ℝ) :
  r1 = 3 → r2 = 4 → tangent1 = tangent2 → r1 + r2 = 7 →
  1 / 2 * (6 * real.sqrt(10)) * 14 = 42 * real.sqrt(10) :=
by
  intro hr1 hr2 htangent hr_sum
  rw [hr1, hr2, htangent, hr_sum]
  have A : r1 = 3 := hr1
  have B : r2 = 4 := hr2
  have C : 6 * real.sqrt(10) = tangent1 := rfl
  have D : tangent1 = tangent2 := htangent
  simp
  sorry

end area_of_triangle_DEF_l115_115229


namespace transform_y_eq_2x_to_y_eq_2x_minus3_minus1_l115_115924

theorem transform_y_eq_2x_to_y_eq_2x_minus3_minus1 :
  ∀ (x : ℝ), (λ x, 2^x) (x - 3) - 1 = 2^(x - 3) - 1 :=
by
  intro x
  sorry

end transform_y_eq_2x_to_y_eq_2x_minus3_minus1_l115_115924


namespace differential_eq_solution_l115_115426

noncomputable def P (x y : ℝ) : ℝ := x * y^2 - y^3
noncomputable def Q (x y : ℝ) : ℝ := 1 - x * y^2
noncomputable def ∂P_∂y (x y : ℝ) : ℝ := 2 * x * y - 3 * y^2
noncomputable def ∂Q_∂x (x y : ℝ) : ℝ := -y^2
noncomputable def μ (y : ℝ) : ℝ := y^(-2)

theorem differential_eq_solution (x y C : ℝ) :
  (x * y^2 - y^3) * (∂P_∂y x y) + (1 - x * y^2) * (∂Q_∂x x y) = 0 :=
by
  sorry

example (x y C : ℝ) : 
  x^2 * y - 2 * x * y^2 - 2 - 2 * C * y = 0 :=
by
  sorry

end differential_eq_solution_l115_115426


namespace hoseok_result_l115_115567

theorem hoseok_result :
  ∃ X : ℤ, (X - 46 = 15) ∧ (X - 29 = 32) :=
by
  sorry

end hoseok_result_l115_115567


namespace original_salary_l115_115522

theorem original_salary (x : ℝ)
  (h1 : x * 1.10 * 0.95 = 3135) : x = 3000 :=
by
  sorry

end original_salary_l115_115522


namespace committee_probability_l115_115167

theorem committee_probability (total_members boys girls committee_size : ℕ) 
  (h1 : total_members = 24) 
  (h2 : boys = 12) 
  (h3 : girls = 12) 
  (h4 : committee_size = 5) 
  : (1 - (2 * nat.choose boys committee_size) / (nat.choose total_members committee_size) : ℚ) = 284 / 295 :=
by
  sorry

end committee_probability_l115_115167


namespace ants_harvesting_time_l115_115066

theorem ants_harvesting_time :
  let initial_sugar : ℕ := 48
  let first_swarm_rate : ℕ := 6
  let second_swarm_rate : ℕ := 3
  let time_until_obstacle : ℕ := 3
  let total_harvest_before_obstacle := time_until_obstacle * (first_swarm_rate + second_swarm_rate)
  let remaining_sugar := initial_sugar - total_harvest_before_obstacle
  let reduced_first_swarm_rate := first_swarm_rate / 2
  let reduced_second_swarm_rate := second_swarm_rate / 2
  let combined_reduced_rate := reduced_first_swarm_rate + reduced_second_swarm_rate
  let additional_hours := (remaining_sugar / combined_reduced_rate).ceil
  in additional_hours = 5 :=
by sorry

end ants_harvesting_time_l115_115066


namespace part1_part2_part3_l115_115327

def f (a x : ℝ) : ℝ := log a x + a * x + 1 / (x + 1)

-- Part (1)
theorem part1 (h : 2 = 2) : f 2 (1 / 4) = -7 / 10 := 
  sorry

-- Part (2)
theorem part2 (a : ℝ) (ha : 1 < a) : ∃! x : ℝ, 0 < x ∧ f a x = 0 := 
  sorry

-- Part (3)
theorem part3 (a x0 : ℝ) (ha : 1 < a) (hfx0 : f a x0 = 0) : 
  1 / 2 < f a (sqrt x0) ∧ f a (sqrt x0) < (a + 1) / 2 := 
  sorry

end part1_part2_part3_l115_115327


namespace quadratic_root_in_interval_range_l115_115912

theorem quadratic_root_in_interval_range (k : ℝ) :
  (∃ x ∈ Ioo 2 5, x^2 - k*x - 2 = 0) ↔ k ∈ Ioo 1 (23 / 5) :=
by
  sorry

end quadratic_root_in_interval_range_l115_115912


namespace find_50th_positive_term_index_l115_115683

def sequence_b (n : ℕ) : ℝ := ∑ k in Finset.range n, Real.cos k

theorem find_50th_positive_term_index : ∃ n, (∀ k < n, sequence_b k ≤ 0) ∧ (sequence_b n > 0) ∧ n = 314 :=
by
  sorry

end find_50th_positive_term_index_l115_115683


namespace most_likely_number_of_red_balls_l115_115541

-- Define the total number of balls and the frequency of picking red balls as given in the conditions
def total_balls : ℕ := 20
def frequency_red : ℝ := 0.8

-- State the equivalent proof problem: Prove that the most likely number of red balls is 16
theorem most_likely_number_of_red_balls : frequency_red * (total_balls : ℝ) = 16 := by
  sorry

end most_likely_number_of_red_balls_l115_115541


namespace total_league_games_l115_115491

theorem total_league_games (divisions : ℕ) (teams_per_division : ℕ) 
  (teams_intra_division_games : ℕ) (teams_inter_division_games : ℕ) 
  (total_teams_plane_themselves_inside_division_twice : teams_intra_division_games = 6 * 2) 
  (total_teams_play_other_division_teams_twice : teams_inter_division_games = 7 * 2) 
  (total_teams : 2 * teams_per_division = 14) : 
  (divisions = 2) ∧ (teams_per_division = 7) → 
  (Sum_of_Total_teams_games = teams_intra_division_games + teams_inter_division_games) → 
  (total_league_scheduled_games : ℕ) -> 
  (total_league_scheduled_games = (total_teams * Sum_of_Total_teams_games) / 2) :=
begin
  -- We expect the total league games to be 182.
  Games_in_the_league == 182
end

end total_league_games_l115_115491


namespace rhombus_perimeter_l115_115898

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 30) (h2 : d2 = 16) (h3 : ∃ ⦃A B C D : ℝ × ℝ⦄, 
  (0 < A.1 ∧ 0 < A.2 ∧ 0 < B.1 ∧ 0 < B.2 ∧ 0 < C.1 ∧ 0 < C.2 ∧ 0 < D.1 ∧ 0 < D.2) ∧ 
  (dist A B = d1 ∧ dist B C = d2 ∧ dist C D = d1 ∧ dist D A = d2) ∧
  (angle A B C = π / 2 ∧ angle B C D = π / 2 ∧ angle C D A = π / 2 ∧ angle D A B = π / 2))
  : perimeter ((sqrt ((d1 / 2)^2 + (d2 / 2)^2)) * 4) = 68 :=
by 
  sorry

end rhombus_perimeter_l115_115898


namespace probability_tail_tail_not_one_third_l115_115566

theorem probability_tail_tail_not_one_third:
  let sample_space := {("Heads", "Heads"), ("Heads", "Tails"), ("Tails", "Heads"), ("Tails", "Tails")}
  let outcome_prob := 1 / 4
  ("Tails", "Tails") ∈ sample_space →
  (1 / 3) ≠ outcome_prob →
  ∀ event ∈ sample_space, event = ("Tails", "Tails") → event → outcome_prob := by
  -- Proof to be completed
  sorry

end probability_tail_tail_not_one_third_l115_115566


namespace f_xh_minus_fx_l115_115382

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 4

theorem f_xh_minus_fx (x h : ℝ) : f(x + h) - f(x) = h * (3 * h + 6 * x + 5) :=
by
  sorry

end f_xh_minus_fx_l115_115382


namespace sausage_cuts_l115_115618

theorem sausage_cuts :
  let red_rings := 5
      yellow_rings := 7
      green_rings := 11 
      red_cuts := red_rings - 1
      yellow_cuts := yellow_rings - 1
      green_cuts := green_cuts - 1
      total_cuts := red_cuts + yellow_cuts + green_cuts
      total_pieces := total_cuts + 1
  in total_pieces = 21 := 
by
  sorry

end sausage_cuts_l115_115618


namespace additional_time_due_to_leak_l115_115621

theorem additional_time_due_to_leak 
  (normal_time_per_barrel : ℕ)
  (leak_time_per_barrel : ℕ)
  (barrels : ℕ)
  (normal_duration : normal_time_per_barrel = 3)
  (leak_duration : leak_time_per_barrel = 5)
  (barrels_needed : barrels = 12) :
  (leak_time_per_barrel * barrels - normal_time_per_barrel * barrels) = 24 := 
by
  sorry

end additional_time_due_to_leak_l115_115621


namespace circle_tangent_ratio_l115_115929
open Real EuclideanGeometry Point Circle Segment

/-- Given two circles ω₁ and ω₂ intersecting at points A and B,
    and segments PQ and RS as common tangents such that P ∈ ω₁, R ∈ ω₁, 
    Q ∈ ω₂, S ∈ ω₂, and RB ∥ PQ, with ray RB intersecting ω₂ at W ≠ B,
    prove that the ratio RB/BW is 1/3. -/
theorem circle_tangent_ratio 
  (ω₁ ω₂ : Circle) 
  (A B P R Q S W : Point)
  (PQ RS₀ : Segment) 
  (hAB : A ∈ (ω₁ ∩ ω₂))
  (hB : B ∈ (ω₁ ∩ ω₂))
  (hP : P ∈ ω₁)
  (hR : R ∈ ω₁)
  (hQ : Q ∈ ω₂)
  (hS : S ∈ ω₂)
  (hPQ : Tangent PQ ω₁ ω₂ P Q)
  (hRS₀ : Tangent RS₀ ω₁ ω₂ R S)
  (hRB_PQ_parallel : RB ∥ PQ)
  (hRW : W ≠ B ∧ W ∈ ω₂ ∧ RB.ray_intersects W) :
  RB.length / BW.length = 1 / 3 := 
sorry

end circle_tangent_ratio_l115_115929


namespace candy_store_spending_l115_115040

-- Define John's total allowance
def allowance : ℝ := 3.375

-- Define the fraction of allowance spent at the arcade
def fraction_arcade : ℝ := 3/5

-- Define the spending at the arcade
def spending_arcade : ℝ := fraction_arcade * allowance

-- Define the remaining allowance after the arcade
def remaining_after_arcade : ℝ := allowance - spending_arcade

-- Define the fraction of the remaining allowance spent at the toy store
def fraction_toy_store : ℝ := 1/3

-- Define the spending at the toy store
def spending_toy_store : ℝ := fraction_toy_store * remaining_after_arcade

-- Define the remaining allowance after the toy store
def remaining_after_toy_store : ℝ := remaining_after_arcade - spending_toy_store

-- The proof problem: Prove that the remaining allowance after the toy store is equal to $0.90
theorem candy_store_spending : remaining_after_toy_store = 0.90 :=
  by
  sorry

end candy_store_spending_l115_115040


namespace find_coordinates_C_l115_115366

-- Definitions based on conditions
def A : ℝ × ℝ := (5, 1)
def line_CM (x y : ℝ) := 2 * x - y - 5 = 0
def line_BH (x y : ℝ) := x - 2 * y - 5 = 0

-- The proof statement
theorem find_coordinates_C :
  ∃ C : ℝ × ℝ,
    let (m, n) := C in
    line_CM m n ∧
    ∀ (x y : ℝ),
      ((x - 5) / (y - 1) = -2 * (5 - x) / (y - 1)) →
      (AC (m + 5) (n + 1) = 0) → 
      C = (4, 3) :=
by
  sorry

end find_coordinates_C_l115_115366


namespace geom_proof_l115_115425

-- Definitions of the geometric objects involved.
variables {A B C G H : Point}
variables {O : Circle}

-- Definition of the geometric relationships and conditions
def is_median (A B C E : Point) : Prop := sorry
def circle_passes_through (O : Circle) (P Q : Point) : Prop := sorry
def circle_tangent_to (O : Circle) (L : Line) (P : Point) : Prop := sorry
def line_intersects_circle (L : Line) (O : Circle) (P Q : Point) : Prop := sorry

-- Main theorem statement
theorem geom_proof (A B C G H : Point) (O : Circle) (BE : Line) (CG : Line) :
  triangle A B C → 
  circle_passes_through O A →
  circle_tangent_to O BE G →
  line_intersects_circle CG O G H →
  CG.length * GH.length = AG.length ^ 2 :=
sorry

end geom_proof_l115_115425


namespace spherical_coordinates_negate_y_l115_115989

theorem spherical_coordinates_negate_y
  (a b c : ℝ)
  (h₁ : a = 3 * sin (Real.pi / 6) * cos (3 * Real.pi / 4))
  (h₂ : b = 3 * sin (Real.pi / 6) * sin (3 * Real.pi / 4))
  (h₃ : c = 3 * cos (Real.pi / 6)) :
  (3, 7 * Real.pi / 4, Real.pi / 6) = spherical_coordinates (a, -b, c) :=
sorry

end spherical_coordinates_negate_y_l115_115989


namespace positive_difference_diagonals_l115_115585

def original_block : Matrix (Fin 5) (Fin 5) ℕ := ![
  ![1, 2, 3, 4, 5],
  ![11, 12, 13, 14, 15],
  ![21, 22, 23, 24, 25],
  ![31, 32, 33, 34, 35],
  ![41, 42, 43, 44, 45]
]

def new_block : Matrix (Fin 5) (Fin 5) ℕ := ![
  ![1, 2, 3, 4, 5],
  ![15, 14, 13, 12, 11],
  ![21, 22, 23, 24, 25],
  ![31, 32, 33, 34, 35],
  ![45, 44, 43, 42, 41]
]

theorem positive_difference_diagonals : 
  let main_diag_sum (mat : Matrix (Fin 5) (Fin 5) ℕ) := ∑ i : Fin 5, mat i i
  let secondary_diag_sum (mat : Matrix (Fin 5) (Fin 5) ℕ) := ∑ i : Fin 5, mat i (4 - i)
  abs (secondary_diag_sum new_block - main_diag_sum new_block) = 4 :=
by
  sorry

end positive_difference_diagonals_l115_115585


namespace divisibility_polynomial_l115_115147

variables {a m x n : ℕ}

theorem divisibility_polynomial (a m x n : ℕ) :
  m ∣ n ↔ (x^m - a^m) ∣ (x^n - a^n) :=
by
  sorry

end divisibility_polynomial_l115_115147


namespace codexia_license_plate_probability_l115_115082

theorem codexia_license_plate_probability : 
  let vowels_or_special_chars := 8
  let non_vowels := 21
  let different_non_vowels := 20
  let even_digits := 5
  let total_plates := vowels_or_special_chars * non_vowels * different_non_vowels * even_digits
  in total_plates = 16800 ∧ (1 / total_plates) = (1 / 16800) :=
by
  let vowels_or_special_chars := 8
  let non_vowels := 21
  let different_non_vowels := 20
  let even_digits := 5
  let total_plates := vowels_or_special_chars * non_vowels * different_non_vowels * even_digits
  have h_total_plates : total_plates = 16800
  sorry
  have h_probability : (1 : ℝ) / total_plates = (1 : ℝ) / 16800
  sorry
  exact ⟨h_total_plates, h_probability⟩

end codexia_license_plate_probability_l115_115082


namespace range_of_t_l115_115815

noncomputable theory

variable {A B C x t : ℝ}

-- Conditions of the problem
axiom triangle_angles (A B C : ℝ) : A + B + C = π
axiom angle_B_lt_pi_div_3 (B : ℝ) : 0 < B ∧ B ≤ π / 3

-- The proof statement
theorem range_of_t (A B C : ℝ) (h_triangle : triangle_angles A B C) (h_B : angle_B_lt_pi_div_3 B) :
  (∀ x : ℝ, (x + 2 + Real.sin (2 * B))^2 + (Real.sqrt 2 * t * Real.sin (B + π / 4))^2 ≥ 1) →
  -1 ≥ t ∨ t ≥ 1 :=
by
  sorry

end range_of_t_l115_115815


namespace inscribed_hexagon_diagonals_sum_l115_115979

theorem inscribed_hexagon_diagonals_sum 
  (A B C D E F : Type) 
  (h : real.inscribed △ A B C ∧ 
       real.inscribed △ A C D ∧
       real.inscribed △ A D E ∧
       real.inscribed △ A E F ∧ 
       real.inscribed △ A F B)
  (hAB : real.segment_length A B = 40)
  (hBC : real.segment_length B C = 100)
  (hCD : real.segment_length C D = 100)
  (hDE : real.segment_length D E = 100)
  (hEF : real.segment_length E F = 100)
  (hFA : real.segment_length F A = 80) :
  real.segment_length A C + real.segment_length A D + real.segment_length A E = 365.4 := 
sorry

end inscribed_hexagon_diagonals_sum_l115_115979


namespace rectangle_same_color_exists_l115_115556

theorem rectangle_same_color_exists (grid : Fin 3 → Fin 7 → Bool) : 
  ∃ (r1 r2 c1 c2 : Fin 3), r1 ≠ r2 ∧ c1 ≠ c2 ∧ grid r1 c1 = grid r1 c2 ∧ grid r1 c1 = grid r2 c1 ∧ grid r1 c1 = grid r2 c2 :=
by
  sorry

end rectangle_same_color_exists_l115_115556


namespace find_number_l115_115252

theorem find_number (x : ℝ) (h : (((x + 1.4) / 3 - 0.7) * 9 = 5.4)) : x = 2.5 :=
by 
  sorry

end find_number_l115_115252


namespace estimated_students_in_sport_A_correct_l115_115227

noncomputable def total_students_surveyed : ℕ := 80
noncomputable def students_in_sport_A_surveyed : ℕ := 30
noncomputable def total_school_population : ℕ := 800
noncomputable def proportion_sport_A : ℚ := students_in_sport_A_surveyed / total_students_surveyed
noncomputable def estimated_students_in_sport_A : ℚ := total_school_population * proportion_sport_A

theorem estimated_students_in_sport_A_correct :
  estimated_students_in_sport_A = 300 :=
by
  sorry

end estimated_students_in_sport_A_correct_l115_115227


namespace weight_of_five_bowling_balls_l115_115131

theorem weight_of_five_bowling_balls (b c : ℕ) (hb : 9 * b = 4 * c) (hc : c = 36) : 5 * b = 80 := by
  sorry

end weight_of_five_bowling_balls_l115_115131


namespace distance_between_points_l115_115298

theorem distance_between_points (x1 y1 x2 y2 : ℝ) (h1 : x1 = 0) (h2 : y1 = -12) (h3 : x2 = -5) (h4 : y2 = 0) :
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 13 :=
by {
  rw [h1, h2, h3, h4],
  norm_num,
  exact real.sqrt_eq_rpow_of_nonneg (show 0 ≤ (25 + 144 : ℝ), by norm_num),
  rw [← real.sqrt_mul_self (show 0 ≤ 13, by norm_num), real.mul_self_sqrt (show 0 ≤ 13^2, by norm_num)],
  norm_num,
  exact real.sqrt_eq_rpow_of_nonneg (show 0 ≤ 169, by norm_num),
  norm_num
}

end distance_between_points_l115_115298


namespace find_m_value_l115_115763

theorem find_m_value (m : ℝ) (h1 : 2 ∈ ({0, m, m^2 - 3 * m + 2} : set ℝ)) : m = 3 :=
sorry

end find_m_value_l115_115763


namespace larger_number_l115_115933

theorem larger_number (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : max x y = 16 := by
  sorry

end larger_number_l115_115933


namespace distance_M1_M12_l115_115026

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 2) * cos (x - π / 2)
def y_value : ℝ := -1 / 2

def M (k : ℕ) : ℝ × ℝ :=
  if k % 2 = 0 then (7 * π / 12 + k * π, 0) else (11 * π / 12 + k * π, 0)

def M1 := M 0
def M12 := M 11

theorem distance_M1_M12 : dist M1 M12 = 16 * π / 3 := by
  sorry

end distance_M1_M12_l115_115026


namespace ram_money_l115_115521

variable (R G K : ℕ)

theorem ram_money (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : K = 2890) : R = 490 :=
by
  sorry

end ram_money_l115_115521


namespace total_legs_in_park_l115_115420

theorem total_legs_in_park :
  let dogs := 109
  let cats := 37
  let birds := 52
  let spiders := 19
  let dog_legs := 4
  let cat_legs := 4
  let bird_legs := 2
  let spider_legs := 8
  dogs * dog_legs + cats * cat_legs + birds * bird_legs + spiders * spider_legs = 840 := by
  sorry

end total_legs_in_park_l115_115420


namespace equilateral_triangle_l115_115411

theorem equilateral_triangle (ABC : Type) [triangle ABC] 
  (A B C A' B' C1 : ABC) 
  (h_acute : is_acute_triangle ABC) 
  (h_angle_C : ∠C = 60) 
  (h_altitudes : is_altitude AA' ∧ is_altitude BB')
  (h_midpoint : is_midpoint C1 A B) :
  is_equilateral_triangle (triangle C1 A' B') :=
sorry

end equilateral_triangle_l115_115411


namespace tourist_group_people_count_l115_115248

theorem tourist_group_people_count 
  (num_large_rooms : ℕ)
  (people_per_large_room : ℕ)
  (total_people : ℕ)
  (num_large_rooms = 8)
  (people_per_large_room = 3)
  (∀ rooms : ℕ, (total_people = rooms * 3 + 0) → num_large_rooms = rooms) :
  total_people = 24 :=
by
  sorry

end tourist_group_people_count_l115_115248


namespace logical_equivalence_l115_115564

theorem logical_equivalence (P Q R : Prop) :
  ((P ∧ ¬R) → ¬Q) ↔ (Q → (¬P ∨ R)) :=
by
  sorry

end logical_equivalence_l115_115564


namespace quadratic_roots_diff_by_2_l115_115674

theorem quadratic_roots_diff_by_2 (q : ℝ) (hq : 0 < q) :
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 - r2 = 2 ∨ r2 - r1 = 2) ∧ r1 ^ 2 + (2 * q - 1) * r1 + q = 0 ∧ r2 ^ 2 + (2 * q - 1) * r2 + q = 0) ↔
  q = 1 + (Real.sqrt 7) / 2 :=
sorry

end quadratic_roots_diff_by_2_l115_115674


namespace sqrt_identity_l115_115010

def condition1 (α : ℝ) : Prop := 
  ∃ P : ℝ × ℝ, P = (Real.sin 2, Real.cos 2) ∧ Real.sin α = Real.cos 2

def condition2 (P : ℝ × ℝ) : Prop := 
  P.1 ^ 2 + P.2 ^ 2 = 1

theorem sqrt_identity (α : ℝ) (P : ℝ × ℝ) 
  (h₁ : condition1 α) (h₂ : condition2 P) : 
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := by 
  sorry

end sqrt_identity_l115_115010


namespace hyperbola_standard_equation_correct_l115_115704

-- Define the initial values given in conditions
def a : ℝ := 12
def b : ℝ := 5
def c : ℝ := 4

-- Define the hyperbola equation form based on conditions and focal properties
noncomputable def hyperbola_standard_equation : Prop :=
  let a2 := (8 / 5)
  let b2 := (72 / 5)
  (∀ x y : ℝ, y^2 / a2 - x^2 / b2 = 1)

-- State the final problem as a theorem
theorem hyperbola_standard_equation_correct :
  ∀ x y : ℝ, y^2 / (8 / 5) - x^2 / (72 / 5) = 1 :=
by
  sorry

end hyperbola_standard_equation_correct_l115_115704


namespace project_completion_equation_l115_115970

variables (x : ℕ)

-- Project completion conditions
def person_A_time : ℕ := 12
def person_B_time : ℕ := 8
def A_initial_work_days : ℕ := 3

-- Work done by Person A when working alone for 3 days
def work_A_initial := (A_initial_work_days:ℚ) / person_A_time

-- Work done by Person A and B after the initial 3 days until completion
def combined_work_remaining := 
  (λ x:ℕ => ((x - A_initial_work_days):ℚ) * (1/person_A_time + 1/person_B_time))

-- The equation representing the total work done equals 1
theorem project_completion_equation (x : ℕ) : 
  (x:ℚ) / person_A_time + (x - A_initial_work_days:ℚ) / person_B_time = 1 :=
sorry

end project_completion_equation_l115_115970


namespace problem_I_problem_II_l115_115267

open Real

theorem problem_I : (sqrt 3 * ((2:ℝ)^(1/3)))^6 - 7 * (sqrt (16 / 49)) - (2018:ℝ)^log10 1 = 103 := by
  sorry

theorem problem_II : 2 * log 3 2 - log 3 (32 / 9) + log 3 8 - log 3 (1 / 81) = 4 := by 
  sorry

end problem_I_problem_II_l115_115267


namespace smallest_integer_m_l115_115947

-- Definition of the problem conditions
def lcm (a b : ℕ) : ℕ := (a * b) / (Nat.gcd a b)

-- Problem statement
theorem smallest_integer_m (m : ℕ) (h1 : lcm 80 m / Nat.gcd 80 m = 40) : m = 50 := by
  sorry

end smallest_integer_m_l115_115947


namespace area_of_sector_AOB_l115_115496

-- Definitions for the conditions
def circumference_sector_AOB : Real := 6 -- Circumference of sector AOB
def central_angle_AOB : Real := 1 -- Central angle of sector AOB

-- Theorem stating the area of the sector is 2 cm²
theorem area_of_sector_AOB (C : Real) (θ : Real) (hC : C = circumference_sector_AOB) (hθ : θ = central_angle_AOB) : 
    ∃ S : Real, S = 2 :=
by
  sorry

end area_of_sector_AOB_l115_115496


namespace count_three_digit_with_f_l115_115444

open Nat

def f : ℕ → ℕ := sorry 

axiom f_add_add (a b : ℕ) : f (a + b) = f (f a + b)
axiom f_add_small (a b : ℕ) (h : a + b < 10) : f (a + b) = f a + f b
axiom f_10 : f 10 = 1

theorem count_three_digit_with_f (hN : ∀ n : ℕ, f 2^(3^(4^5)) = f n):
  ∃ k, k = 100 ∧ ∀ n, 100 ≤ n ∧ n < 1000 → (f n = f 2^(3^(4^5))) :=
sorry

end count_three_digit_with_f_l115_115444


namespace age_ratio_6_years_ago_l115_115537

theorem age_ratio_6_years_ago :
  let henry_age := 20
  let jill_age := 13
  in henry_age + jill_age = 33 →
     (henry_age - 6) / (jill_age - 6) = 2 :=
by
  intros
  sorry

end age_ratio_6_years_ago_l115_115537


namespace gain_percent_is_150_l115_115385

variable (C S : ℝ)
variable (h : 50 * C = 20 * S)

theorem gain_percent_is_150 (h : 50 * C = 20 * S) : ((S - C) / C) * 100 = 150 :=
by
  sorry

end gain_percent_is_150_l115_115385


namespace least_possible_perimeter_minimal_perimeter_triangle_l115_115185

theorem least_possible_perimeter (x : ℕ) 
  (h1 : 13 < x) 
  (h2 : x < 61) : 
  24 + 37 + x ≥ 24 + 37 + 14 := 
sorry

theorem minimal_perimeter_triangle : ∃ x : ℕ, 13 < x ∧ x < 61 ∧ 24 + 37 + x = 75 :=
begin
  existsi 14,
  split,
  { exact dec_trivial, }, -- 13 < 14
  split,
  { exact dec_trivial, }, -- 14 < 61
  { exact dec_trivial, }, -- 24 + 37 + 14 = 75
end

end least_possible_perimeter_minimal_perimeter_triangle_l115_115185


namespace count_values_leq_60_with_g50eq18_correct_l115_115708
noncomputable def g₁ (n : ℕ) : ℕ := 3 * Nat.divisors n |>.length

noncomputable def g (j n : ℕ) : ℕ :=
  if j = 1 then g₁ n else g₁ (g (j-1) n)

def count_values_leq_60_with_g50eq18 : ℕ :=
  (Finset.range 61).filter (λ n => g 50 n = 18) |>.card

theorem count_values_leq_60_with_g50eq18_correct :
  count_values_leq_60_with_g50eq18 = 4 := by
  sorry

end count_values_leq_60_with_g50eq18_correct_l115_115708


namespace inequality_proof_l115_115848

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) : 
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l115_115848


namespace eccentricity_of_ellipse_l115_115338

noncomputable def eccentricity_range (a b c : ℝ) (h1 : a > b > 0) (h2 : a^2 = b^2 + c^2) : Set ℝ :=
  { e | e = c / a ∧ 1 / 2 ≤ e ∧ e < 1 }

theorem eccentricity_of_ellipse (a b c : ℝ) (h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)
  (h_angle : ∀ (M F1 F2 : ℝ), ∠F1 M F2 = π / 3) (h_range : 1 / 2 ≤ c / a ∧ c / a < 1) :
  eccentricity_range a b c (by sorry) (by sorry) := 
begin
  sorry
end

end eccentricity_of_ellipse_l115_115338


namespace vacation_total_cost_l115_115821

def plane_ticket_cost (per_person_cost : ℕ) (num_people : ℕ) : ℕ :=
  num_people * per_person_cost

def hotel_stay_cost (per_person_per_day_cost : ℕ) (num_people : ℕ) (num_days : ℕ) : ℕ :=
  num_people * per_person_per_day_cost * num_days

def total_vacation_cost (plane_ticket_cost : ℕ) (hotel_stay_cost : ℕ) : ℕ :=
  plane_ticket_cost + hotel_stay_cost

theorem vacation_total_cost :
  let per_person_plane_ticket_cost := 24
  let per_person_hotel_cost := 12
  let num_people := 2
  let num_days := 3
  let plane_cost := plane_ticket_cost per_person_plane_ticket_cost num_people
  let hotel_cost := hotel_stay_cost per_person_hotel_cost num_people num_days
  total_vacation_cost plane_cost hotel_cost = 120 := by
  sorry

end vacation_total_cost_l115_115821


namespace additional_time_due_to_leak_l115_115622

theorem additional_time_due_to_leak 
  (normal_time_per_barrel : ℕ)
  (leak_time_per_barrel : ℕ)
  (barrels : ℕ)
  (normal_duration : normal_time_per_barrel = 3)
  (leak_duration : leak_time_per_barrel = 5)
  (barrels_needed : barrels = 12) :
  (leak_time_per_barrel * barrels - normal_time_per_barrel * barrels) = 24 := 
by
  sorry

end additional_time_due_to_leak_l115_115622


namespace sum_of_valid_k_l115_115328

/- Define the base and constraints -/
def base : ℂ := -4 + 1 * complex.i
def digits := {a : ℕ | a ≤ 256}

theorem sum_of_valid_k :
  let k (a_3 a_2 a_1 a_0 : ℕ) := a_3 * (base ^ 3).re + a_2 * (base ^ 2).re + a_1 * base.re + a_0 in
  (44 * a_3 - 8 * a_2 + a_1 = 0) →
  let k_ := (a_3 :: a_2 :: a_1 :: a_0 :: []) in 
  a_3 ∈ digits →
  a_2 ∈ digits →
  a_1 ∈ digits →
  a_0 ∈ digits →
  finset.sum
  (finset.univ.filter (λ (k_ : list ℕ), ∃ a_3 a_2 a_1 a_0 ∈ digits, 44 * a_3 - 8 * a_2 + a_1 = 0 ∧
                                                k_ = [a_3, a_2, a_1, a_0 ∧ a_0 .. a_2 .. a_3 ≤ 256])) 
  k_ =  ?
:= sorry

end sum_of_valid_k_l115_115328


namespace product_of_five_consecutive_integers_not_perfect_square_l115_115144

theorem product_of_five_consecutive_integers_not_perfect_square (n : ℕ) : 
  ¬ ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = k * k :=
by {
  sorry
}

end product_of_five_consecutive_integers_not_perfect_square_l115_115144


namespace unique_integer_solution_l115_115295

-- Define the problem statement and the conditions: integers x, y such that x^4 - 2y^2 = 1
theorem unique_integer_solution (x y: ℤ) (h: x^4 - 2 * y^2 = 1) : (x = 1 ∧ y = 0) :=
sorry

end unique_integer_solution_l115_115295


namespace Mia_Noah_same_game_45_times_l115_115858

/-- Meadowbrook Elementary School has a three-square league consisting of twelve players, including Mia and Noah.
 Each day, the twelve players split into three three-square games, each with four players in no particular order. 
 Over the course of a term, each possible match-up of four players occurs once. Prove that Mia and Noah end up playing in the same game 45 times. -/
theorem Mia_Noah_same_game_45_times :
  let total_players := 12
  let players_in_game := 4
  let Mia_Noah := 2
  (nat.choose (total_players - Mia_Noah) (players_in_game - Mia_Noah) = 45) :=
begin
  -- Conditions setup
  let total_players: ℕ := 12,
  let players_in_game: ℕ := 4,
  let Mia_Noah: ℕ := 2,
  -- Definitions derived from these conditions
  calc nat.choose (total_players - Mia_Noah) (players_in_game - Mia_Noah) = 10.choose 2 : by rfl
  ... = 45 : by sorry -- ⟨expanded steps of combination calculation, if needed⟩,
end

end Mia_Noah_same_game_45_times_l115_115858


namespace range_of_f_l115_115388

def f (x : ℝ) : ℝ := log (2.0) (-x^2 + 5 * x)

example : f 1 = 2 := sorry

theorem range_of_f : set.Iic (log (2.0) (25 / 4)) = {y : ℝ | ∃ x : ℝ, f x = y} :=
sorry

end range_of_f_l115_115388


namespace students_in_class_l115_115458

-- Definitions of the conditions
def current_students (tables : ℕ) (students_per_table : ℕ) : ℕ :=
  tables * students_per_table

def missing_students (bathroom : ℕ) (canteen_ratio : ℕ) : ℕ :=
  bathroom + canteen_ratio * bathroom

def new_groups (groups : ℕ) (students_per_group : ℕ) : ℕ :=
  groups * students_per_group

def foreign_students (germany : ℕ) (france : ℕ) (norway : ℕ) : ℕ :=
  germany + france + norway

def total_students (current : ℕ) (missing : ℕ) (new : ℕ) (foreign : ℕ) : ℕ :=
  current + missing + new + foreign

-- The theorem to prove the number of students
theorem students_in_class :
  current_students 6 3 + missing_students 3 3 + new_groups 2 4 + foreign_students 3 3 3 = 47 :=
by
  unfold current_students missing_students new_groups foreign_students total_students
  simp
  sorry

end students_in_class_l115_115458


namespace functional_eqn_even_function_l115_115447

variable {R : Type*} [AddGroup R] (f : R → ℝ)

theorem functional_eqn_even_function
  (h_not_zero : ∃ x, f x ≠ 0)
  (h_func_eq : ∀ a b, f (a + b) + f (a - b) = 2 * f a + 2 * f b) :
  ∀ x, f (-x) = f x :=
by
  sorry

end functional_eqn_even_function_l115_115447


namespace exists_k_square_congruent_neg_one_iff_l115_115441

theorem exists_k_square_congruent_neg_one_iff (p : ℕ) [Fact p.Prime] :
  (∃ k : ℤ, (k^2 ≡ -1 [ZMOD p])) ↔ (p = 2 ∨ p % 4 = 1) :=
sorry

end exists_k_square_congruent_neg_one_iff_l115_115441


namespace total_items_bought_l115_115787

def total_money : ℝ := 40
def sandwich_cost : ℝ := 5
def chip_cost : ℝ := 2
def soft_drink_cost : ℝ := 1.5

/-- Ike and Mike spend their total money on sandwiches, chips, and soft drinks.
  We want to prove that the total number of items bought (sandwiches, chips, and soft drinks)
  is equal to 8. -/
theorem total_items_bought :
  ∃ (s c d : ℝ), (sandwich_cost * s + chip_cost * c + soft_drink_cost * d ≤ total_money) ∧
  (∀x : ℝ, sandwich_cost * s ≤ total_money) ∧ ((s + c + d) = 8) :=
by {
  sorry
}

end total_items_bought_l115_115787


namespace slab_cost_l115_115996

-- Define the conditions
def cubes_per_stick : ℕ := 4
def cubes_per_slab : ℕ := 80
def total_kabob_cost : ℕ := 50
def kabob_sticks_made : ℕ := 40
def total_cubes_needed := kabob_sticks_made * cubes_per_stick
def slabs_needed := total_cubes_needed / cubes_per_slab

-- Final proof problem statement in Lean 4
theorem slab_cost : (total_kabob_cost / slabs_needed) = 25 := by
  sorry

end slab_cost_l115_115996


namespace find_x_for_y_eq_9_l115_115872

def y (x : ℝ) : ℝ := if x < 0 then x^2 - 2 * x + 6 else (x - 1)^2

theorem find_x_for_y_eq_9 : {x : ℝ | y x = 9} = {-1, 4} :=
by
  sorry

end find_x_for_y_eq_9_l115_115872


namespace distance_from_start_need_refuel_total_revenue_l115_115488

def distances : List Int := [-3, -15, +19, -1, +5, -12, -6, +12]

def passengers : List Bool := [False, True, True, False, True, True, True, True]

def fuel_consumption_per_km : Float := 0.06

def initial_fuel : Float := 7.0

def refuel_threshold : Float := 2.0

def base_fare : Float := 10.0

def additional_fare_per_km : Float := 1.6

def distance_threshold : Int := 2

-- Part 1: Prove direction and distance from Point A
theorem distance_from_start : 
  ∑ x in distances, x = -1 := sorry

-- Part 2: Prove whether refueling is needed
theorem need_refuel : 
  initial_fuel - (fuel_consumption_per_km * (distances.map (fun x => |x|)).sum) > refuel_threshold := sorry

-- Part 3: Prove total revenue
theorem total_revenue :
  (passengers.zip distances).foldr
    (fun (pd : Bool × Int) acc =>
      acc + if pd.1 = True 
            then base_fare + Float.ofInt (pd.2 - distance_threshold) * additional_fare_per_km 
            else 0.0)
    0 = 151.2 := sorry

end distance_from_start_need_refuel_total_revenue_l115_115488


namespace integer_equality_condition_l115_115686

theorem integer_equality_condition
  (x y z : ℤ)
  (h : x * (x - y) + y * (y - z) + z * (z - x) = 0) :
  x = y ∧ y = z :=
sorry

end integer_equality_condition_l115_115686


namespace volume_is_correct_l115_115892

noncomputable def volume_of_pyramid (b : ℝ) : ℝ :=
  if h : b ≠ 0 then
    let AC : ℝ := b
    let BD : ℝ := b
    let theta_diagonal : ℝ := real.angle.pi_div3
    let angle_lateral_edge : ℝ := real.angle.pi_div4
    let S_base : ℝ := (b^2 * real.sqrt 3) / 4
    let height : ℝ := (1/2) * b
    (1 / 3) * S_base * height
  else
    0

theorem volume_is_correct (b : ℝ) : b ≠ 0 → volume_of_pyramid b = (b^3 * real.sqrt 3) / 24 :=
by
  intro h
  rw volume_of_pyramid
  dsimp
  simp [h]
  sorry

end volume_is_correct_l115_115892


namespace prime_divides_a_p_l115_115312

def a : ℕ → ℤ
| 1     := 0
| 2     := 2
| 3     := 3
| (n+3) := a (n+1) + a n

theorem prime_divides_a_p (p : ℕ) [Fact (Nat.Prime p)] : p ∣ a p :=
sorry

end prime_divides_a_p_l115_115312


namespace cos_two_thirds_pi_minus_alpha_l115_115738

variable {α : ℝ}

theorem cos_two_thirds_pi_minus_alpha (h : sin (α - π / 6) = 1 / 3) : 
  cos (2 * π / 3 - α) = 1 / 3 :=
sorry

end cos_two_thirds_pi_minus_alpha_l115_115738


namespace find_50th_positive_term_index_l115_115682

def sequence_b (n : ℕ) : ℝ := ∑ k in Finset.range n, Real.cos k

theorem find_50th_positive_term_index : ∃ n, (∀ k < n, sequence_b k ≤ 0) ∧ (sequence_b n > 0) ∧ n = 314 :=
by
  sorry

end find_50th_positive_term_index_l115_115682


namespace remainder_when_divided_by_6_eq_5_l115_115624

theorem remainder_when_divided_by_6_eq_5 (k : ℕ) (hk1 : k % 5 = 2) (hk2 : k < 41) (hk3 : k % 7 = 3) : k % 6 = 5 :=
sorry

end remainder_when_divided_by_6_eq_5_l115_115624


namespace finishing_order_l115_115408

-- Definitions of conditions
def athletes := ["Grisha", "Sasha", "Lena"]

def overtakes : (String → ℕ) := 
  fun athlete =>
    if athlete = "Grisha" then 10
    else if athlete = "Sasha" then 4
    else if athlete = "Lena" then 6
    else 0

-- All three were never at the same point at the same time
def never_same_point_at_same_time : Prop := True -- Simplified for translation purpose

-- The main theorem stating the finishing order given the provided conditions
theorem finishing_order :
  never_same_point_at_same_time →
  (overtakes "Grisha" = 10) →
  (overtakes "Sasha" = 4) →
  (overtakes "Lena" = 6) →
  athletes = ["Grisha", "Sasha", "Lena"] :=
  by
    intro h1 h2 h3 h4
    exact sorry -- The proof is not required, just ensuring the statement is complete.


end finishing_order_l115_115408


namespace measure_of_angle_B_l115_115086

theorem measure_of_angle_B (a b : ℝ) (A B C : ℝ) (h1 : b ≠ 0) (h2 : a ≠ 0) 
    (h3 : 0 < A) (h4 : 0 < B) (h5 : 0 < C) 
    (h6 : A + B + C = π) (h7 : sqrt 2 * a = 2 * b * sin A):
    B = π / 4 ∨ B = 3 * π / 4 :=
sorry

end measure_of_angle_B_l115_115086


namespace length_of_LM_l115_115169

-- Definitions of the conditions
variable (P Q R L M : Type)
variable (b : Real) (PR_area : Real) (PR_base : Real)
variable (PR_base_eq : PR_base = 15)
variable (crease_parallel : Parallel L M)
variable (projected_area_fraction : Real)
variable (projected_area_fraction_eq : projected_area_fraction = 0.25 * PR_area)

-- Theorem statement to prove the length of LM
theorem length_of_LM : ∀ (LM_length : Real), (LM_length = 7.5) :=
sorry

end length_of_LM_l115_115169


namespace similar_triangles_ratios_l115_115141

-- Define the context
variables {a b c a' b' c' : ℂ}

-- Define the statement of the problem
theorem similar_triangles_ratios (h_sim : ∃ z : ℂ, z ≠ 0 ∧ b - a = z * (b' - a') ∧ c - a = z * (c' - a')) :
  (b - a) / (c - a) = (b' - a') / (c' - a') :=
sorry

end similar_triangles_ratios_l115_115141


namespace helmet_pricing_and_profit_constraint_l115_115940

theorem helmet_pricing_and_profit_constraint :
  ∃ (x y : ℕ) (m : ℕ),
    (10 * x + 15 * y = 1150) ∧
    (6 * x + 12 * y = 810) ∧
    x = 55 ∧ y = 40 ∧
    (40 * m + 30 * (100 - m) ≤ 3400) ∧
    (15 * m + 10 * (100 - m) ≠ 1300) ∧
    m ≤ 40 :=
by {
  let x := 55,
  let y := 40,
  let m := 60,
  have h1: 10 * x + 15 * y = 1150 := by norm_num,
  have h2: 6 * x + 12 * y = 810 := by norm_num,
  have h3: 40 * m + 30 * (100 - m) ≤ 3400 := by norm_num,
  have h4: 15 * m + 10 * (100 - m) ≠ 1300 := by norm_num,
  exact ⟨x, y, m, h1, h2, rfl, rfl, h3, h4, by norm_num⟩,
}

end helmet_pricing_and_profit_constraint_l115_115940


namespace find_m_n_find_max_min_l115_115442

noncomputable def quadratic_eq (x : ℝ) (m n : ℝ) : ℝ := x^2 + m * x + n

theorem find_m_n 
  (x1 x2 : ℝ)
  (h1 : x1 = -1)
  (h2 : x2 = -2)
  (m n : ℝ)
  (h_eq : ∀ x : ℝ, quadratic_eq x m n = 0 ↔ x = x1 ∨ x = x2)
  : m = 3 ∧ n = 2 := 
begin
  sorry
end

theorem find_max_min
  (m n : ℝ)
  (hm : m = 3)
  (hn : n = 2)
  : ∃ min_val max_val : ℝ, 
    min_val = -1 / 4 ∧ max_val = 42 ∧ 
    ∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → 
      (quadratic_eq x m n = max_val ∨ quadratic_eq x m n = min_val) := 
begin
  sorry
end

end find_m_n_find_max_min_l115_115442


namespace husband_additional_payment_l115_115973

theorem husband_additional_payment (total_medical_cost : ℝ) (total_salary : ℝ) 
                                  (half_medical_cost : ℝ) (deduction_from_salary : ℝ) 
                                  (remaining_salary : ℝ) (total_payment : ℝ)
                                  (each_share : ℝ) (amount_paid_by_husband : ℝ) : 
                                  
                                  total_medical_cost = 128 →
                                  total_salary = 160 →
                                  half_medical_cost = total_medical_cost / 2 →
                                  deduction_from_salary = half_medical_cost →
                                  remaining_salary = total_salary - deduction_from_salary →
                                  total_payment = remaining_salary + half_medical_cost →
                                  each_share = total_payment / 2 →
                                  amount_paid_by_husband = 64 →
                                  (each_share - amount_paid_by_husband) = 16 := by
  sorry

end husband_additional_payment_l115_115973


namespace probability_of_both_co_captains_l115_115189

theorem probability_of_both_co_captains :
  let team1_students := 6
  let team2_students := 9
  let team3_students := 10
  let team1_co_captains := 2
  let team2_co_captains := 2
  let team3_co_captains := 3
  let prob_team_selected := 1 / 3
  let combinations (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  let prob_both_co_captains (team_size co_captains : ℕ) : ℚ := (combinations co_captains 2 : ℚ) / (combinations team_size 2 : ℚ)
  in
  prob_team_selected * (prob_both_co_captains team1_students team1_co_captains + 
                        prob_both_co_captains team2_students team2_co_captains + 
                        prob_both_co_captains team3_students team3_co_captains) = 29 / 540 := 
by
  sorry

end probability_of_both_co_captains_l115_115189


namespace range_of_b_l115_115900

theorem range_of_b (b : ℝ) :
  (∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ 9^x1 + |3^x1 + b| = 5 ∧ 9^x2 + |3^x2 + b| = 5) ↔ 
  b ∈ set.Ioo (-21/4) (-5) :=
sorry

end range_of_b_l115_115900


namespace most_likely_number_of_red_balls_l115_115544

-- Define the conditions
def total_balls : ℕ := 20
def red_ball_frequency : ℝ := 0.8

-- Define the statement we want to prove
theorem most_likely_number_of_red_balls : red_ball_frequency * total_balls = 16 :=
by sorry

end most_likely_number_of_red_balls_l115_115544


namespace find_polynomial_l115_115113

-- Define the polynomial conditions
structure CubicPolynomial :=
  (P : ℝ → ℝ)
  (P0 : ℝ)
  (P1 : ℝ)
  (P2 : ℝ)
  (P3 : ℝ)
  (cubic_eq : ∀ x, P x = P0 + P1 * x + P2 * x^2 + P3 * x^3)

theorem find_polynomial (P : CubicPolynomial) (h_neg1 : P.P (-1) = 2) (h0 : P.P 0 = 3) (h1 : P.P 1 = 1) (h2 : P.P 2 = 15) :
  ∀ x, P.P x = 3 + x - 2 * x^2 - x^3 :=
sorry

end find_polynomial_l115_115113


namespace coefficient_x_term_l115_115697

theorem coefficient_x_term :
  Let f := (1 + 2 * x)^3 * (1 - x)^4,
  find_coeff f x = 2 :=
sorry

end coefficient_x_term_l115_115697


namespace number_of_elements_in_intersection_l115_115033

def A (n : ℕ) : ℕ := 3 * n + 2

def A_set : Set ℕ := {x | ∃ n : ℕ, x = A n}

def B_set : Set ℕ := {6, 8, 10, 12, 14}

theorem number_of_elements_in_intersection :
  (A_set ∩ B_set).to_finset.card = 2 := by
  sorry

end number_of_elements_in_intersection_l115_115033


namespace range_of_m_l115_115058

-- Define the problem conditions
def curve (x m : ℝ) : ℝ := log (2^x - m) / log 2

def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop := p.1 = -q.1 ∧ p.2 = -q.2

def on_line_y_eq_x_plus_1 (p : ℝ × ℝ) : Prop := p.2 = p.1 + 1

-- Main statement
theorem range_of_m (m : ℝ) (x : ℝ) (h1 : ∃ (s t : ℝ), 2 < s ∧ curve s m = t ∧ symmetric_wrt_origin (s, t) (neg (s, t)) ∧ on_line_y_eq_x_plus_1 (neg (s, t)) ∧ 2 < x) :
  2 < m ∧ m ≤ 4 :=
by {
  -- Placeholder for proof
  sorry
}

end range_of_m_l115_115058


namespace valerian_equal_l115_115262

section

-- Definitions based on the given conditions
variable (sunny_days : Finset ℕ) (cloudy_days : Finset ℕ)
variable (day : ℕ → ℕ)

-- Assume April consists of 30 days with exactly 15 sunny and 15 cloudy days
axiom april_days_split : sunny_days.card = 15 ∧ cloudy_days.card = 15 ∧
  (∀ d ∈ sunny_days, 1 ≤ d ∧ d ≤ 30) ∧ (∀ d ∈ cloudy_days, 1 ≤ d ∧ d ≤ 30) ∧
  sunny_days ∩ cloudy_days = ∅ ∧ (sunny_days ∪ cloudy_days = Finset.range 30)

-- Define the drinking patterns for both individuals
def andrey_valerian (sunny_days : Finset ℕ) : ℕ :=
  sunny_days.sum (λ i, i)

def ivan_valerian (cloudy_days : Finset ℕ) (day : ℕ → ℕ) : ℕ :=
  cloudy_days.sum day

-- Theorem statement translating the proof problem
theorem valerian_equal (day : ℕ → ℕ) :
  andrey_valerian sunny_days = ivan_valerian cloudy_days day :=
begin
  sorry
end

end

end valerian_equal_l115_115262


namespace area_of_ABCD_l115_115805

open Real

-- Define the vectors BD and AC
def vector_BD : ℝ × ℝ := (-6, 2)
def vector_AC : ℝ × ℝ := (1, 3)

-- The area of the quadrilateral ABCD
theorem area_of_ABCD : 
  let BD_len := ((-6)^2 + 2^2).sqrt
  let AC_len := (1^2 + 3^2).sqrt in
  1/2 * BD_len * AC_len = 10 := 
by
  sorry

end area_of_ABCD_l115_115805


namespace false_proposition_proof_l115_115317

variable {a b c m : ℝ}

-- Condition definitions
def f (x : ℝ) := a * x^2 + b * x + c
def condition1 : Prop := a > 0
def condition2 : Prop := 2 * a * m + b = 0

-- False proposition
def false_proposition : Prop := ∀ x : ℝ, f x ≤ f m

-- Lean statement
theorem false_proposition_proof (h1 : condition1) (h2 : condition2) : ¬ false_proposition := by
  sorry

end false_proposition_proof_l115_115317


namespace surface_area_of_sphere_l115_115728

-- Define the given conditions and the proof statement
theorem surface_area_of_sphere (O : Type) [MetricSpace O] (A B C : O) (R : ℝ) (h1 : dist O A = R) (h2 : dist O B = R) (h3 : dist O C = R) (h4 : Plane.distABC O A B C = R / 2) (h5 : dist A B = 2) (h6 : dist A C = 2) (h7 : angle A B C = real.pi * 2 / 3) : 
  4 * real.pi * R^2 = 64 / 3 * real.pi :=
begin
  sorry
end

end surface_area_of_sphere_l115_115728


namespace childrens_cookbook_cost_l115_115429

def dictionary_cost : ℕ := 11
def dinosaur_book_cost : ℕ := 19
def saved_amount : ℕ := 8
def needed_amount : ℕ := 29

theorem childrens_cookbook_cost :
  let total_cost := saved_amount + needed_amount,
      combined_cost := dictionary_cost + dinosaur_book_cost
  in total_cost - combined_cost = 7 :=
by
  sorry

end childrens_cookbook_cost_l115_115429


namespace range_of_s_l115_115446

variable (a b c : ℝ)

def s (a b c : ℝ) : ℝ :=
  (a + b) / (1 + c) + (b + c) / (1 + a) + (c + a) / (1 + b)

theorem range_of_s (ha : a ∈ set.Icc (1 / 2 : ℝ) 1)
                    (hb : b ∈ set.Icc (1 / 2 : ℝ) 1)
                    (hc : c ∈ set.Icc (1 / 2 : ℝ) 1) : 
  2 ≤ s a b c ∧ s a b c ≤ 3 :=
by
  sorry

end range_of_s_l115_115446


namespace find_number_l115_115881

-- Assume the necessary definitions and conditions
variable (x : ℝ)

-- Sixty-five percent of the number is 21 less than four-fifths of the number
def condition := 0.65 * x = 0.8 * x - 21

-- Final proof goal: We need to prove that the number x is 140
theorem find_number (h : condition x) : x = 140 := by
  sorry

end find_number_l115_115881


namespace lambda_mu_zero_l115_115031

noncomputable def parabola := { P : ℝ × ℝ | P.snd^2 = 4 * P.fst }

def focus := (1, 0)

def directrix_line : set (ℝ × ℝ) :=
  { P | P.fst = -1 }

def intersection_points (p : set (ℝ × ℝ)) (l : set (ℝ × ℝ)) : set (ℝ × ℝ) :=
  { P | P ∈ p ∧ P ∈ l }

noncomputable def A : (ℝ × ℝ) := sorry -- A is the intersection point
noncomputable def B : (ℝ × ℝ) := sorry -- B is the intersection point
noncomputable def P : (ℝ × ℝ) := sorry -- P is the point (-1, some y satisfying the parabola)

def vector (P Q : ℝ × ℝ) := (Q.fst - P.fst, Q.snd - P.snd)

variables (λ μ : ℝ)
axioms
  (h_PA_AF : vector P A = λ • vector A focus)
  (h_PB_BF : vector P B = μ • vector B focus)

theorem lambda_mu_zero (λ μ : ℝ) 
  (h_PA_AF : vector P A = λ • vector A focus)
  (h_PB_BF : vector P B = μ • vector B focus) : 
  λ + μ = 0 :=
sorry

end lambda_mu_zero_l115_115031


namespace percentage_profits_to_revenues_l115_115064

theorem percentage_profits_to_revenues (R P : ℝ) 
  (h1 : R > 0) 
  (h2 : P > 0)
  (h3 : 0.12 * R = 1.2 * P) 
  : P / R = 0.1 :=
by
  sorry

end percentage_profits_to_revenues_l115_115064


namespace mrs_smith_net_gain_l115_115863

-- Define selling price, profit, and loss
def selling_price : ℝ := 2.40
def profit_percentage : ℝ := 0.25
def loss_percentage : ℝ := 0.15

-- Equations for cost price based on profit and loss
def cost_price_first_vase : ℝ := selling_price / (1 + profit_percentage)
def cost_price_second_vase : ℝ := selling_price / (1 - loss_percentage)

-- Total cost and revenue
def total_cost : ℝ := cost_price_first_vase + cost_price_second_vase
def total_revenue : ℝ := selling_price + selling_price

-- Net result calculation
def net_result : ℝ := total_revenue - total_cost

-- Proof statement
theorem mrs_smith_net_gain : 
  net_result = 0.06 :=
by 
  -- Insert the proof here
  sorry

end mrs_smith_net_gain_l115_115863


namespace part_I_part_II_l115_115851

noncomputable def f (x a : ℝ) := exp x - a * x - 2

def intervals_of_monotonicity (a : ℝ) : Prop :=
  if a ≤ 0 then (∀ x y : ℝ, x ≤ y → f x a ≤ f y a) 
  else (∀ x : ℝ, x < real.log a → ∀ y : ℝ, y > x → f y a < f x a) ∧
       (∀ x : ℝ, x > real.log a → ∀ y : ℝ, y > x → f y a > f x a)

theorem part_I {a : ℝ} : intervals_of_monotonicity a :=
  sorry

def inequality_condition (k x : ℝ) (a : ℝ) := x > 0 ∧ (x - k) * (exp x - a) + x + 1 > 0

theorem part_II (k : ℤ) : ∀x > 0, inequality_condition k x 1 → k ≤ 2 :=
  sorry

end part_I_part_II_l115_115851


namespace triangle_medians_and_area_l115_115560

/-- Given a triangle with side lengths 13, 14, and 15,
    prove that the sum of the squares of the lengths of the medians is 385
    and the area of the triangle is 84. -/
theorem triangle_medians_and_area :
  let a := 13
  let b := 14
  let c := 15
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let m_a := Real.sqrt (2 * b^2 + 2 * c^2 - a^2) / 2
  let m_b := Real.sqrt (2 * c^2 + 2 * a^2 - b^2) / 2
  let m_c := Real.sqrt (2 * a^2 + 2 * b^2 - c^2) / 2
  m_a^2 + m_b^2 + m_c^2 = 385 ∧ area = 84 := sorry

end triangle_medians_and_area_l115_115560


namespace other_root_of_quadratic_l115_115019

theorem other_root_of_quadratic (a : ℝ) (h : (Polynomial.C a + Polynomial.C (-2)).eval 1 = 0) :
    ∃ x₂ : ℝ, (Polynomial.C (x₂) - Polynomial.C (-3/2)).eval 0 = 0 :=
by sorry

end other_root_of_quadratic_l115_115019


namespace maximum_n_factor_of_2_in_2017_factorial_l115_115307

theorem maximum_n_factor_of_2_in_2017_factorial :
  ∃ n : ℕ, (∃ (hn : 0 < n), 2^n ∣ nat.factorial 2017) ∧ (∀ n' : ℕ, (∃ (hn' : 0 < n'), 2^n' ∣ nat.factorial 2017) → n' ≤ n) ∧ n = 2010 :=
sorry

end maximum_n_factor_of_2_in_2017_factorial_l115_115307


namespace at_least_two_As_l115_115995

variables {Ω : Type*} [ProbabilityTheory Ω]
variables {A B C : Event Ω} 

noncomputable def prob_A : ℚ := 7 / 8
noncomputable def prob_B : ℚ := 3 / 4
noncomputable def prob_C : ℚ := 5 / 12

axiom independent_events : Indep (A, B) (A ∪ B) ∧ Indep (A, C) (A ∪ C) ∧ Indep (B, C) (B ∪ C)

theorem at_least_two_As : 
  (P(A ∩ B ∩ ¬C) + P(A ∩ ¬B ∩ C) + P(¬A ∩ B ∩ C) + P(A ∩ B ∩ C) = 151 / 192) :=
sorry

end at_least_two_As_l115_115995


namespace calf_probability_l115_115259

theorem calf_probability 
  (P_B1 : ℝ := 0.6)  -- Proportion of calves from the first farm
  (P_B2 : ℝ := 0.3)  -- Proportion of calves from the second farm
  (P_B3 : ℝ := 0.1)  -- Proportion of calves from the third farm
  (P_B1_A : ℝ := 0.15)  -- Conditional probability of a calf weighing more than 300 kg given it is from the first farm
  (P_B2_A : ℝ := 0.25)  -- Conditional probability of a calf weighing more than 300 kg given it is from the second farm
  (P_B3_A : ℝ := 0.35)  -- Conditional probability of a calf weighing more than 300 kg given it is from the third farm)
  (P_A : ℝ := P_B1 * P_B1_A + P_B2 * P_B2_A + P_B3 * P_B3_A) : 
  P_B3 * P_B3_A / P_A = 0.175 := 
by
  sorry

end calf_probability_l115_115259


namespace determine_strictly_increasing_functions_l115_115685

noncomputable def strictly_increasing (f : ℕ → ℕ) :=
  ∀ n m : ℕ, n < m → f(n) < f(m)

theorem determine_strictly_increasing_functions :
  ∀ f : ℕ → ℕ, (strictly_increasing f ∧ (∀ n : ℕ, n > 0 → n * f (f n) = (f n) ^ 2)) ↔ 
  (f = λ x, x) ∨ (∃ c d : ℕ, 1 < c ∧ 
    ∀ x : ℕ, (x < d → f x = x) ∧ (x ≥ d → f x = c * x)) := by
  sorry

end determine_strictly_increasing_functions_l115_115685


namespace cos_diff_to_product_l115_115292

open Real

theorem cos_diff_to_product (a b : ℝ) : 
  cos (a + b) - cos (a - b) = -2 * sin a * sin b := 
  sorry

end cos_diff_to_product_l115_115292


namespace largest_integer_base8_square_l115_115443

theorem largest_integer_base8_square :
  ∃ (N : ℕ), (N^2 >= 8^3) ∧ (N^2 < 8^4) ∧ (N = 63 ∧ N % 8 = 7) := sorry

end largest_integer_base8_square_l115_115443


namespace angle_PSU_correct_l115_115088

noncomputable def anglePSU (P Q R S T U : Type) 
  [inner_product_space ℝ P] [inner_product_space ℝ Q] [inner_product_space ℝ R]
  [inner_product_space ℝ S] [inner_product_space ℝ T] [inner_product_space ℝ U] 
  (angle_PRQ angle_QRP : ℝ) 
  (PQR : triangle P Q R) 
  (S_foot : foot S P QR) 
  (T_circumcenter : circumcenter T PQR) 
  (U_diameter_end : diameter U T P) 
  (h_angle_PRQ : angle_PRQ = 60) 
  (h_angle_QRP : angle_QRP = 50) : ℝ :=
  150

theorem angle_PSU_correct (P Q R S T U : Type) 
  [inner_product_space ℝ P] [inner_product_space ℝ Q] [inner_product_space ℝ R]
  [inner_product_space ℝ S] [inner_product_space ℝ T] [inner_product_space ℝ U] 
  (angle_PRQ angle_QRP : ℝ) 
  (PQR : triangle P Q R) 
  (S_foot : foot S P QR) 
  (T_circumcenter : circumcenter T PQR) 
  (U_diameter_end : diameter U T P) 
  (h_angle_PRQ : angle_PRQ = 60) 
  (h_angle_QRP : angle_QRP = 50) : 
  anglePSU P Q R S T U angle_PRQ angle_QRP PQR S_foot T_circumcenter U_diameter_end h_angle_PRQ h_angle_QRP = 150 :=
by
  sorry

end angle_PSU_correct_l115_115088


namespace arithmetic_sequence_sum_l115_115730

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h0 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2) 
  (h1 : S 10 = 12)
  (h2 : S 20 = 17) :
  S 30 = 15 := by
  sorry

end arithmetic_sequence_sum_l115_115730


namespace students_in_class_l115_115494

theorem students_in_class (n : ℕ) (h1 : (n : ℝ) * 100 = (n * 100 + 60 - 10)) 
  (h2 : (n : ℝ) * 98 = ((n : ℝ) * 100 - 50)) : n = 25 :=
sorry

end students_in_class_l115_115494


namespace min_value_fraction_l115_115089

variable (a b c : ℝ)
variable {A B C : ℝ}
variable [hab : 0 < a] [habc : a < b + c ∧ b < a + c ∧ c < a + b]
variable [hAngleA : 0 < A ∧ A < π] [hAngleB : 0 < B ∧ B < π] [hAngleC : 0 < C ∧ C < π]

-- Given condition c - a = 2a * cos B
axiom cosB_condition : c - a = 2 * a * Real.cos B

-- Prove that the minimum value of (3a + c) / b is 2 * sqrt 2
theorem min_value_fraction : (∃ (A B C : ℝ), 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π ∧ c - a = 2 * a * Real.cos B) → 
  (∃ min_value : ℝ, min_value = 2 * Real.sqrt 2 ∧ ∀ (b c : ℝ), (3 * a + c) / b ≥ min_value) :=
by 
  sorry

end min_value_fraction_l115_115089


namespace sum_of_last_three_digits_l115_115111

theorem sum_of_last_three_digits (C : ℕ) : 
  let D := (2401 * 7^C) % 1000 in
  ((D / 100) % 10 + (D / 10) % 10 + D % 10) = 7 := sorry

end sum_of_last_three_digits_l115_115111


namespace eight_divides_n_is_false_l115_115844

theorem eight_divides_n_is_false (n : ℕ) (hn : 0 < n) (h : (1 / 3 : ℚ) + (1 / 4) + (1 / 8) + (1 / n) ∈ ℤ) : ¬(8 ∣ n) :=
sorry

end eight_divides_n_is_false_l115_115844


namespace geom_seq_min_value_proof_l115_115067

noncomputable def geom_seq_min_value : ℝ := 3 / 2

theorem geom_seq_min_value_proof (a : ℕ → ℝ) (a1 : ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →
  a 2017 = a 2016 + 2 * a 2015 →
  a m * a n = 16 * a1^2 →
  (4 / m + 1 / n) = geom_seq_min_value :=
by {
  sorry
}

end geom_seq_min_value_proof_l115_115067


namespace no_positive_integer_satisfies_equation_l115_115042

theorem no_positive_integer_satisfies_equation :
  ¬ ∃ n : ℕ, n > 0 ∧ (∃ k : ℤ, (n + 500) / 50 = k ∧ k^3 ≤ n ∧ n < (k + 1)^3) :=
begin
  sorry
end

end no_positive_integer_satisfies_equation_l115_115042


namespace general_formula_sum_of_b_l115_115523

-- Define the sequences and their properties
def a (n : ℕ) : ℕ := sorry
def S (n : ℕ) : ℕ := sorry

-- Initial condition and sequence sum condition
axiom a_1 : a 1 = 2
axiom S_n (n : ℕ) (hn : n > 0) : S n = (3 * a n) / 2 - 1

-- b sequence and the sum of first n terms T_n
def b (n : ℕ) : ℕ := n * a n
def T (n : ℕ) : ℕ := ∑ i in finset.range n, b (i + 1)

-- General formula for the n-th term of the sequence
theorem general_formula (n : ℕ) (hn : n > 0) : a n = 2 * 3^(n-1) :=
sorry

-- Sum of the first n terms of b_n
theorem sum_of_b (n : ℕ) (hn : n > 0) : T n = ((2 * n - 1) * 3^n + 1) / 2 :=
sorry

end general_formula_sum_of_b_l115_115523


namespace sum_cot2_in_T_l115_115837

def T : Set ℝ := {x | π/4 < x ∧ x < π/2 ∧ (∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ perm (list.sin_cos_cot x).perm [a, b, c])}

theorem sum_cot2_in_T : ∑ x in T, cot^2 x = 4 - 2 * sqrt 3 := sorry

end sum_cot2_in_T_l115_115837


namespace problem_2_l115_115363

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x ^ 2 + a * Real.log (1 - x)

theorem problem_2 (a : ℝ) (x₁ x₂ : ℝ) (h₀ : 0 < a) (h₁ : a < 1/4) (h₂ : f x₂ a = 0) 
  (h₃ : f x₁ a = 0) (hx₁ : 0 < x₁) (hx₂ : x₁ < 1/2) (h₄ : x₁ < x₂) :
  f x₂ a - x₁ > - (3 + Real.log 4) / 8 := sorry

end problem_2_l115_115363


namespace angle_bisector_l115_115735

open EuclideanGeometry

variables {A B C D E F G : Point} 
variables {l : Line}

-- Definitions of the points and the conditions
def is_parallelogram (A B C D : Point) : Prop := Parallelogram A B C D
def is_cyclic_quadrilateral (B C E D : Point) : Prop := Cyclic B C E D
def on_line (l : Line) (P : Point) : Prop := P ∈ l
def intersect (F L : Point) (DC : Segment) : Prop := F ∈ DC ∧ is_interior_point F DC
def equal_length (EF EG EC : Length) : Prop := EF = EG ∧ EG = EC

-- Main theorem statement
theorem angle_bisector (h1 : is_parallelogram A B C D)
                      (h2 : is_cyclic_quadrilateral B C E D)
                      (h3 : on_line l A)
                      (h4 : intersect F l (segment D C))
                      (h5 : intersect G l (line B C))
                      (h6 : equal_length (distance E F) (distance E G) (distance E C)) :
  is_angle_bisector l (angle D A B) :=
sorry

end angle_bisector_l115_115735


namespace carterHas152Cards_l115_115125

-- Define the number of baseball cards Marcus has.
def marcusCards : Nat := 210

-- Define the number of baseball cards Carter has.
def carterCards : Nat := marcusCards - 58

-- Theorem to prove Carter's baseball cards total 152 given the conditions.
theorem carterHas152Cards (h1 : marcusCards = 210) (h2 : marcusCards = carterCards + 58) : carterCards = 152 :=
by
  -- Proof omitted for this exercise
  sorry

end carterHas152Cards_l115_115125


namespace positional_relationship_l115_115006

-- Definitions for the conditions
variables {α : Type*} [MetricSpace α]
variables {a b : Line α} {π : Plane α}

-- Conditions from the problem
def line_parallel_to_plane (a : Line α) (π : Plane α) : Prop :=
  ∀ p₁ p₂ ∈ a.points, ∃ p₃ ∈ π.points, (p₁ -ᵥ p₃) ⟂ π.normal ∧ (p₂ -ᵥ p₃) ⟂ π.normal

def line_in_plane (b : Line α) (π : Plane α) : Prop :=
  ∀ p ∈ b.points, p ∈ π.points

-- The main theorem we need to state
theorem positional_relationship (h₁ : line_parallel_to_plane a π) (h₂ : line_in_plane b π) :
  Parallel a b ∨ Skew a b :=
by sorry

end positional_relationship_l115_115006


namespace least_perimeter_triangle_l115_115178

theorem least_perimeter_triangle :
  ∀ (c : ℕ), (c < 61) ∧ (c > 13) → 24 + 37 + c = 75 → c = 14 :=
by
  intros c hc hperimeter
  cases hc with hc1 hc2
  have c_eq_14 : c = 14 := sorry
  exact c_eq_14

end least_perimeter_triangle_l115_115178


namespace integer_lengths_from_vertex_E_to_hypotenuse_l115_115472

theorem integer_lengths_from_vertex_E_to_hypotenuse (DE EF : ℝ) (h1 : DE = 24) (h2 : EF = 25) (h3 : ∀ a b : ℝ, a^2 + b^2 > 0) :
  ∃ n : ℕ, n = 14 := by
  -- Calculate the hypotenuse
  let DF := real.sqrt (DE^2 + EF^2)
  -- Calculate the altitude from E to DF
  let Area := (1 / 2) * DE * EF
  let EP := 2 * Area / DF
  -- Calculate the integer lengths of EX
  let possible_lengths := (finset.Icc 18 25) ∪ (finset.Icc 18 24) -- 18 is counted twice
  have h4 : possible_lengths.card = 14 := sorry
  use 14
  exact h4

end integer_lengths_from_vertex_E_to_hypotenuse_l115_115472


namespace significant_improvement_l115_115603

noncomputable def z (x y : ℕ → ℕ) (i : ℕ) : ℕ := x i - y i

noncomputable def z_bar (x y : ℕ → ℕ) : ℝ := 
  (z x y 1 + z x y 2 + z x y 3 + z x y 4 + z x y 5 + z x y 6 + 
  z x y 7 + z x y 8 + z x y 9 + z x y 10) / 10

noncomputable def s_squared (x y : ℕ → ℕ) : ℝ := 
  let mean := z_bar x y in 
  ( (z x y 1 - mean) ^ 2 + (z x y 2 - mean) ^ 2 + (z x y 3 - mean) ^ 2 +
    (z x y 4 - mean) ^ 2 + (z x y 5 - mean) ^ 2 + (z x y 6 - mean) ^ 2 +
    (z x y 7 - mean) ^ 2 + (z x y 8 - mean) ^ 2 + (z x y 9 - mean) ^ 2 +
    (z x y 10 - mean) ^ 2) / 10

theorem significant_improvement (x y : ℕ → ℕ)
  (hx : x 1 = 545) (hx2 : x 2 = 533) (hx3 : x 3 = 551) (hx4 : x 4 = 522)
  (hx5 : x 5 = 575) (hx6 : x 6 = 544) (hx7 : x 7 = 541) (hx8 : x 8 = 568)
  (hx9 : x 9 = 596) (hx10 : x 10 = 548)
  (hy : y 1 = 536) (hy2 : y 2 = 527) (hy3 : y 3 = 543) (hy4 : y 4 = 530)
  (hy5 : y 5 = 560) (hy6 : y 6 = 533) (hy7 : y 7 = 522) (hy8 : y 8 = 550)
  (hy9 : y 9 = 576) (hy10 : y 10 = 536) :
  z_bar x y ≥ 2 * real.sqrt(s_squared x y / 10) :=
  sorry

end significant_improvement_l115_115603


namespace additional_carpet_needed_l115_115823

-- Define the given conditions as part of the hypothesis:
def carpetArea : ℕ := 18
def roomLength : ℕ := 4
def roomWidth : ℕ := 20

-- The theorem we want to prove:
theorem additional_carpet_needed : (roomLength * roomWidth - carpetArea) = 62 := by
  sorry

end additional_carpet_needed_l115_115823


namespace circular_garden_area_l115_115243

theorem circular_garden_area (AD DB DC R : ℝ) 
  (h1 : AD = 10) 
  (h2 : DB = 10) 
  (h3 : DC = 12) 
  (h4 : AD^2 + DC^2 = R^2) : 
  π * R^2 = 244 * π := 
  by 
    sorry

end circular_garden_area_l115_115243


namespace find_k_l115_115930

theorem find_k : 
  (∃ (k : ℝ), let P := (6 : ℝ, 8 : ℝ), S := (0 : ℝ, k), origin := (0 : ℝ, 0 : ℝ), QR := 5 in
    (dist origin P = 10) ∧ (dist origin S = 5) ∧ (QR = 5)) → k = 5 :=
by
  sorry

end find_k_l115_115930


namespace sandy_net_amount_spent_l115_115154

def amount_spent_shorts : ℝ := 13.99
def amount_spent_shirt : ℝ := 12.14
def amount_received_return : ℝ := 7.43

theorem sandy_net_amount_spent :
  amount_spent_shorts + amount_spent_shirt - amount_received_return = 18.70 :=
by
  sorry

end sandy_net_amount_spent_l115_115154


namespace problem_divisibility_l115_115467

theorem problem_divisibility (n : ℕ) : ∃ k : ℕ, 2 ^ (3 ^ n) + 1 = 3 ^ (n + 1) * k :=
sorry

end problem_divisibility_l115_115467


namespace base_angle_of_isosceles_triangle_l115_115782

-- Definitions based on the problem conditions
def is_isosceles_triangle (A B C: ℝ) := (A = B) ∨ (B = C) ∨ (C = A)
def angle_sum_triangle (A B C: ℝ) := A + B + C = 180

-- The main theorem we want to prove
theorem base_angle_of_isosceles_triangle (A B C: ℝ)
(h1: is_isosceles_triangle A B C)
(h2: A = 50 ∨ B = 50 ∨ C = 50):
C = 50 ∨ C = 65 :=
by
  sorry

end base_angle_of_isosceles_triangle_l115_115782


namespace last_two_digits_7_pow_2011_l115_115864

noncomputable def pow_mod_last_two_digits (n : ℕ) : ℕ :=
  (7^n) % 100

theorem last_two_digits_7_pow_2011 : pow_mod_last_two_digits 2011 = 43 :=
by
  sorry

end last_two_digits_7_pow_2011_l115_115864


namespace solve_for_x_l115_115059

theorem solve_for_x (x : ℚ) :
  (35 * x = 49 / 9) → (x ≈ 0.16) :=
sorry

end solve_for_x_l115_115059


namespace total_pamphlets_correct_l115_115129

-- Define the individual printing rates and hours
def Mike_pre_break_rate := 600
def Mike_pre_break_hours := 9
def Mike_post_break_rate := Mike_pre_break_rate / 3
def Mike_post_break_hours := 2

def Leo_pre_break_rate := 2 * Mike_pre_break_rate
def Leo_pre_break_hours := Mike_pre_break_hours / 3
def Leo_post_first_break_rate := Leo_pre_break_rate / 2
def Leo_post_second_break_rate := Leo_post_first_break_rate / 2

def Sally_pre_break_rate := 3 * Mike_pre_break_rate
def Sally_pre_break_hours := Mike_post_break_hours / 2
def Sally_post_break_rate := Leo_post_first_break_rate
def Sally_post_break_hours := 1

-- Calculate the total number of pamphlets printed by each person
def Mike_pamphlets := 
  (Mike_pre_break_rate * Mike_pre_break_hours) + (Mike_post_break_rate * Mike_post_break_hours)

def Leo_pamphlets := 
  (Leo_pre_break_rate * 1) + (Leo_post_first_break_rate * 1) + (Leo_post_second_break_rate * 1)

def Sally_pamphlets := 
  (Sally_pre_break_rate * Sally_pre_break_hours) + (Sally_post_break_rate * Sally_post_break_hours)

-- Calculate the total number of pamphlets printed by all three
def total_pamphlets := Mike_pamphlets + Leo_pamphlets + Sally_pamphlets

theorem total_pamphlets_correct : total_pamphlets = 10700 := by
  sorry

end total_pamphlets_correct_l115_115129


namespace point_outside_circle_l115_115054

theorem point_outside_circle {a b : ℝ} (h : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ a * x + b * y = 1) : a^2 + b^2 > 1 :=
by sorry

end point_outside_circle_l115_115054


namespace find_p_q_r_l115_115281

theorem find_p_q_r : 
  ∃ (p q r : ℕ), 
  p > 0 ∧ q > 0 ∧ r > 0 ∧ 
  4 * (Real.sqrt (Real.sqrt 7) - Real.sqrt (Real.sqrt 6)) 
  = Real.sqrt (Real.sqrt p) + Real.sqrt (Real.sqrt q) - Real.sqrt (Real.sqrt r) 
  ∧ p + q + r = 99 := 
sorry

end find_p_q_r_l115_115281


namespace problem_solution_l115_115589

def z (xi yi : ℕ) : ℕ := xi - yi

def z_vals (x y : Fin 10 → ℕ) : Fin 10 → ℕ := fun i => z (x i) (y i)

def mean (z : Fin 10 → ℕ) : ℕ :=
  (∑ i in Finset.univ, z i) / 10

def variance (z : Fin 10 → ℕ) (mean_z : ℕ) : ℕ :=
  (∑ i in Finset.univ, (z i - mean_z)^2) / 10

def significant_improvement (mean_z : ℕ) (var_z : ℕ) : Prop :=
  mean_z >= 2 * Real.sqrt (var_z / 10)

-- Given data
def x : Fin 10 → ℕ
| 0 => 545
| 1 => 533
| 2 => 551
| 3 => 522
| 4 => 575
| 5 => 544
| 6 => 541
| 7 => 568
| 8 => 596
| 9 => 548

def y : Fin 10 → ℕ
| 0 => 536
| 1 => 527
| 2 => 543
| 3 => 530
| 4 => 560
| 5 => 533
| 6 => 522
| 7 => 550
| 8 => 576
| 9 => 536

/-- The final proof statements -/
theorem problem_solution :
  let z_vals := z_vals x y
  let mean_z := mean z_vals
  let var_z := variance z_vals mean_z
  mean_z = 11 ∧ var_z = 61 ∧ significant_improvement mean_z var_z := 
by
  sorry

end problem_solution_l115_115589


namespace tan_x_iff_l115_115895

theorem tan_x_iff (x : Real) : (∃ k : ℤ, x = k * π + π / 4) ↔ tan x = 1 :=
by
  sorry

end tan_x_iff_l115_115895


namespace T8_valid_string_count_l115_115435

def valid_strings_count : ℕ → ℕ 
| 0 := 1
| 1 := 0
| 2 := 0
| 3 := 0
| n := valid_strings_count (n - 1) + valid_strings_count (n - 2) + valid_strings_count (n - 3) + valid_strings_count (n - 4)

theorem T8_valid_string_count : valid_strings_count 8 = 56 := 
by
  -- Placeholder proof
  sorry

end T8_valid_string_count_l115_115435


namespace ratio_HD_JE_l115_115869

variables (A B C D E F G H J : Type) [point : Type] [line_segment : Type] [metric_space : point]

-- Define points on the line segment AF
variables (AB BC CD DE EF AF : line_segment)
variables (GC GB : line_segment)

-- Lengths of line segments 
variable (len_AB : real)
variable (len_BC : real)
variable (len_CD : real)
variable (len_DE : real)
variable (len_EF : real)
variable (len_AF : real)

-- Non-collinear point
variable (G_not_on_AF : Prop)

-- Points H and J on respective line segments
variable (H_on_GC : H ∈ GC)
variable (J_on_GB : J ∈ GB)

-- Parallel lines condition
variable (HD_JE_parallel_AG : Prop)

-- Distances
variable (len_HD : real)
variable (len_JE : real)

-- The goal to prove
theorem ratio_HD_JE (h1 : len_AB = 1)
                    (h2 : len_BC = 2)
                    (h3 : len_CD = 3)
                    (h4 : len_DE = 4)
                    (h5 : len_EF = 5)
                    (h6 : len_AF = len_AB + len_BC + len_CD + len_DE + len_EF)
                    (h7 : G_not_on_AF)
                    (h8 : H_on_GC)
                    (h9 : J_on_GB)
                    (h10 : HD_JE_parallel_AG)
                    : len_HD / len_JE = 1 / 18 :=
sorry

end ratio_HD_JE_l115_115869


namespace num_digits_base_10_l115_115049

noncomputable def x := 2 ^ (2 ^ (2 ^ 2))

theorem num_digits_base_10 : nat.log10 (2 ^ (2 ^ (2 ^ 2))) + 1 = 5 := by
  sorry

end num_digits_base_10_l115_115049


namespace translate_B_to_origin_l115_115076

structure Point where
  x : ℝ
  y : ℝ

def translate_right (p : Point) (d : ℕ) : Point := 
  { x := p.x + d, y := p.y }

theorem translate_B_to_origin :
  ∀ (A B : Point) (d : ℕ),
  A = { x := -4, y := 0 } →
  B = { x := 0, y := 2 } →
  (translate_right A d).x = 0 →
  translate_right B d = { x := 4, y := 2 } :=
by
  intros A B d hA hB hA'
  sorry

end translate_B_to_origin_l115_115076


namespace angle_MBN_l115_115506

noncomputable def square {α : Type*} [euclidean_space ℝ α] (A B C D M N : α) :=
  let AB := (B - A).norm in
  let BC := (C - B).norm in
  let CD := (D - C).norm in
  let DA := (A - D).norm in
  let AM := (M - A).norm in
  let DM := (M - D).norm in
  let MB := (B - M).norm in
  let MN := (N - M).norm in
  let DMN := (N - D).norm in
  AB = BC ∧ BC = CD ∧ CD = DA ∧
  inner (B - A) (D - A) = 0 ∧  -- orthogonality condition for square
  ∠AMB = 60 ∧ ∠DMN = 60

noncomputable def find_angle_MBN {α : Type*} [euclidean_space ℝ α] (A B C D M N : α) (h : square A B C D M N) : ℝ :=
  45

theorem angle_MBN {α : Type*} [euclidean_space ℝ α] (A B C D M N : α):
  square A B C D M N → find_angle_MBN A B C D M N = 45 :=
sorry

end angle_MBN_l115_115506


namespace solve_inequality_l115_115483

open Set

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - (a + 1) * x + 1 < 0

-- Define the solution sets for different cases of a
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 1}
  else if a < 0 then {x | x < 1 / a ∨ x > 1}
  else if 0 < a ∧ a < 1 then {x | 1 < x ∧ x < 1 / a}
  else if a > 1 then {x | 1 / a < x ∧ x < 1}
  else ∅

-- State the theorem
theorem solve_inequality (a : ℝ) : 
  {x : ℝ | inequality a x} = solution_set a :=
by
  sorry

end solve_inequality_l115_115483


namespace ellipse_properties_l115_115731

-- Define the ellipse and its properties
theorem ellipse_properties {a b : ℝ} (h : a > b ∧ b > 0) (h_pass_through : (0, 1) ∈ λ x y, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h_chord_len : ∃ F : ℝ, F > 0 ∧ ∀ x, (x, F) ∈ λ x y, (x^2 / a^2) + (y^2 / b^2) = 1 → ∃ p : ℝ, p > 0 ∧ ∀ q : ℝ, q > 0 ∧ q ≤ F → (0, q) = (λ x y, (x^2 / a^2) + (y^2 / b^2) = 1)) :
  (a = 2 ∧ b = 1 ∧ 
  (∀ P : ℝ × ℝ, P ≠ (2, 0) ∧ P ≠ (-2, 0) ∧ ∃ ⟨E, F⟩ : ℝ × ℝ, E ∈ λ x y, y = (λ x y, x = 2sqrt2) (axpy id id) 
    ∧ F ∈ λ x y, y = (λ x y, x = 2sqrt2) (axpy id id) → 
    (|DE * DF| = 1)) ∧ 
  (∀ P : ℝ × ℝ, 2 * S_of_triangle (triangle P (-2, 0) (2, 0)) = ∃ ⟨E, F⟩ : ℝ × ℝ, 
     |S_of_triangle (triangle P E F)| = 2 * S_of_triangle (triangle P (-2, 0) (2, 0)) → P = (0, -1))) :=
sorry

end ellipse_properties_l115_115731


namespace coordinates_p_in_ijk_basis_l115_115744

-- Definitions for the vectors a, b, c, and p
def i := ⟨1, 0, 0⟩ -- assuming ⟨1, 0, 0⟩ corresponds to i in ℝ³
def j := ⟨0, 1, 0⟩ -- assuming ⟨0, 1, 0⟩ corresponds to j in ℝ³
def k := ⟨0, 0, 1⟩ -- assuming ⟨0, 0, 1⟩ corresponds to k in ℝ³

def a := i + j
def b := j + k
def c := k + i

def coord_p_acb := (8, 6, 4) -- Coordinates of p in basis {a, b, c}

-- Translate coordinates in the base {a, b, c} to {i, j, k}
def p_in_base_ijk := 8 • a + 6 • b + 4 • c

-- Assert that the coordinates of p in the basis {i, j, k} are (12, 14, 10)
theorem coordinates_p_in_ijk_basis : p_in_base_ijk = 12 • i + 14 • j + 10 • k :=
by {
  -- Compute p in terms of i, j, k using definition of a, b, c
  simp only [a, b, c, i, j, k, add_smul, smul_add, add_assoc, add_comm, add_left_comm],
  -- Simplify to get the vector coordinates
  exact sorry
}

end coordinates_p_in_ijk_basis_l115_115744


namespace AM_GM_proof_equality_condition_l115_115096

variable (a b : ℝ)
variable (ha : 0 < a) (hb : 0 < b)

theorem AM_GM_proof : (a + b)^3 / (a^2 * b) ≥ 27 / 4 :=
sorry

theorem equality_condition : (a + b)^3 / (a^2 * b) = 27 / 4 ↔ a = 2 * b :=
sorry

end AM_GM_proof_equality_condition_l115_115096


namespace max_tubes_C_l115_115547

noncomputable def maxC (a b c : ℕ) :=
  0.1 * a + 0.2 * b + 0.9 * c = 0.2017 * (a + b + c) ∧
  a + b + c = 1000

theorem max_tubes_C : ∃ a b c : ℕ, maxC a b c ∧ c ≤ 73 := by
  sorry

end max_tubes_C_l115_115547


namespace max_ab_ratio_l115_115099

theorem max_ab_ratio (a b : ℕ) (h : a > 0 ∧ b > 0) (h_eq : (a : ℚ)/(a-2) = (b+2021)/(b+2008)) :
  (a:ℚ)/(b:ℚ) ≤ 312/7 :=
begin
  -- Proof omitted
  sorry
end

end max_ab_ratio_l115_115099


namespace geometric_sequence_eccentricities_l115_115354

noncomputable def hyperbola_equation := 
  ∃ (a b : ℝ), a = 2 ∧ b = 3 * real.sqrt 2 ∧ 
  (∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1 ↔ (x,y) ∈ {xy | (xy.1^2 / a^2 - xy.2^2 / b^2 = 1)}) ∧ 
  (∃ focus : ℝ × ℝ, focus = (4, 0)))
  
noncomputable def ellipse_equation := 
  ∃ (a b : ℝ), a = 8 ∧ b^2 = 48 ∧ 
  (∀ x y : ℝ, (x^2 / 64 + y^2 / 48 = 1 ↔ (x,y) ∈ {xy | (xy.1^2 / a^2 + xy.2^2 / b^2 = 1)}) ∧ 
  (∃ focus : ℝ × ℝ, focus = (4, 0)))

noncomputable def parabola_equation := 
  ∀ x y : ℝ, y^2 = 16 * x ↔ (x,y) ∈ {xy | (xy.2^2 = 16 * xy.1)} ∧ 
  (∃ focus : ℝ × ℝ, focus = (4, 0))

theorem geometric_sequence_eccentricities :
  hyperbola_equation ∧ ellipse_equation ∧ parabola_equation := 
sorry

end geometric_sequence_eccentricities_l115_115354


namespace combined_area_of_shapes_l115_115626

theorem combined_area_of_shapes:
  let radius := Real.sqrt 1225
  let side_of_square := 35
  let breadth_of_rectangle := 10
  let length_of_rectangle := 1 / 4 * radius
  let area_of_square := side_of_square ^ 2
  let area_of_circle := Real.pi * radius ^ 2
  let area_of_rectangle := length_of_rectangle * breadth_of_rectangle
  let base_of_parallelogram := 12
  let height_of_parallelogram := 14
  let area_of_parallelogram := base_of_parallelogram * height_of_parallelogram
  in side_of_square = 35 ∧ radius = 35 ∧ area_of_square = 1225 ∧ 
     length_of_rectangle = 8.75 ∧ area_of_rectangle = 87.5 ∧ 
     area_of_parallelogram = 168 ∧ 
     area_of_circle = 1225 * Real.pi ∧
     1225 + 1225 * Real.pi + 87.5 + 168 = 1480.5 + 1225 * Real.pi := 
  by
    intros
    sorry

end combined_area_of_shapes_l115_115626


namespace carson_gold_stars_l115_115664

theorem carson_gold_stars (yesterday_stars today_total_stars earned_today : ℕ) 
  (h1 : yesterday_stars = 6) 
  (h2 : today_total_stars = 15) 
  (h3 : earned_today = today_total_stars - yesterday_stars) 
  : earned_today = 9 :=
sorry

end carson_gold_stars_l115_115664


namespace find_ab_perpendicular_find_ab_parallel_l115_115367

def line1 (a b : ℝ) := λ x y: ℝ, a * x - b * y + 4 = 0
def line2 (a : ℝ) := λ x y: ℝ, (a - 1) * x + y + 2 = 0
def point := (-3 : ℝ, -1 : ℝ)
def y_intercept (b : ℝ) := - 4 / b

theorem find_ab_perpendicular (a b : ℝ) (h1 : line1 a b (-3) (-1))
  (h2 : ∀ x y, line1 a b x y → line2 a x y → false ) :
  a = 2 ∧ b = 2 := 
by
  sorry

theorem find_ab_parallel (a b : ℝ) (h3 : ∀ x y, line1 a b x y → ∃ k, line2 a x (k * y))
  (h4 : y_intercept b = -3) :
  a = 4 ∧ b = -4/3 := 
by
  sorry

end find_ab_perpendicular_find_ab_parallel_l115_115367


namespace equal_area_of_second_square_l115_115413

/-- 
In an isosceles right triangle with legs of length 25√2 cm, if a square is inscribed such that two 
of its vertices lie on one leg and one vertex on each of the hypotenuse and the other leg, 
and the area of the square is 625 cm², prove that the area of another inscribed square 
(with one vertex each on the hypotenuse and one leg, and two vertices on the other leg) is also 625 cm².
-/
theorem equal_area_of_second_square 
  (a b : ℝ) (h1 : a = 25 * Real.sqrt 2)  
  (h2 : b = 625) :
  ∃ c : ℝ, c = 625 :=
by
  sorry

end equal_area_of_second_square_l115_115413


namespace least_perimeter_of_triangle_ABC_is_9_l115_115788

noncomputable def triangle_perimeter_least (A B C : ℝ) (a b c : ℕ) : ℕ :=
  if cond : cos A = 11 / 16 ∧ cos B = 7 / 8 ∧ cos C = -1 / 4 ∧ (a : ℝ) / sin A = (b : ℝ) / sin B ∧ (b : ℝ) / sin B = (c : ℝ) / sin C
  then a + b + c
  else 0

theorem least_perimeter_of_triangle_ABC_is_9 :
  triangle_perimeter_least (real.arccos (11 / 16)) (real.arccos (7 / 8)) (real.arccos (-1 / 4)) 3 2 4 = 9 :=
by
  -- Since the cosine values, ratios, and perimeter calculations are involved, we assume without loss of generality and the conditions hold.
  sorry

end least_perimeter_of_triangle_ABC_is_9_l115_115788


namespace simplify_fractions_l115_115156

theorem simplify_fractions :
  (240 / 18) * (6 / 135) * (9 / 4) = 4 / 3 :=
by
  sorry

end simplify_fractions_l115_115156


namespace speed_on_way_home_l115_115866

theorem speed_on_way_home (d : ℝ) (v_up : ℝ) (v_avg : ℝ) (v_home : ℝ) 
  (h1 : v_up = 110) 
  (h2 : v_avg = 91)
  (h3 : 91 = (2 * d) / (d / 110 + d / v_home)) : 
  v_home = 10010 / 129 := 
sorry

end speed_on_way_home_l115_115866


namespace triangular_prism_volume_ratio_l115_115263

noncomputable def midpoint (p1 p2 : Point) : Point := 
  (1/2) • p1 + (1/2) • p2

theorem triangular_prism_volume_ratio (A B C A1 B1 C1 M N : Point) 
    (prism_volume : ℝ) 
    (cond1 : M = midpoint B B1) 
    (cond2 : N = midpoint B1 C1) 
    (partition_plane_formed : Plane := plane_from_points A M N) 
    (volume_smaller_part : ℝ := volume_partition_triangle_prism smaller partition_plane_formed) :
  (volume_smaller_part / prism_volume = 13 / 36) := sorry

end triangular_prism_volume_ratio_l115_115263


namespace no_m_for_necessary_and_sufficient_condition_range_of_m_for_necessary_condition_l115_115581

noncomputable def P : set ℝ := {x : ℝ | x^2 - 8 * x - 20 ≤ 0}
noncomputable def S (m : ℝ) : set ℝ := {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem no_m_for_necessary_and_sufficient_condition :
  ¬∃ m : ℝ, ∀ x : ℝ, (x ∈ P ↔ x ∈ S m) :=
sorry

theorem range_of_m_for_necessary_condition :
  ∃ m : ℝ, ∀ x : ℝ, x ∈ P → x ∈ S m :=
∃ m ≤ 3, ∀ x : ℝ, x ∈ P → x ∈ S m :=
sorry

end no_m_for_necessary_and_sufficient_condition_range_of_m_for_necessary_condition_l115_115581


namespace G_equiv_floor_l115_115175

-- Define the given constants and function
noncomputable def α : ℝ := (Real.sqrt 5 - 1) / 2

-- Define the function G such that it satisfies the given conditions
def G : ℕ → ℕ 
| 0       := 0
| (n + 1) := n + 1 - G (G n)

-- The equivalence theorem we want to prove
theorem G_equiv_floor (n : ℕ) : 
  G n = ⌊α * (n + 1)⌋ := 
sorry

end G_equiv_floor_l115_115175


namespace twenty_five_point_zero_six_million_in_scientific_notation_l115_115653

theorem twenty_five_point_zero_six_million_in_scientific_notation :
  (25.06e6 : ℝ) = 2.506 * 10^7 :=
by
  -- The proof would go here, but we use sorry to skip the proof.
  sorry

end twenty_five_point_zero_six_million_in_scientific_notation_l115_115653


namespace rides_order_count_l115_115094

theorem rides_order_count : (6.factorial = 720) :=
sorry

end rides_order_count_l115_115094


namespace total_amount_divided_l115_115224

-- Define the amounts of money received by Maya, Annie, and Saiji.
variables (M A S : ℝ)

-- Define the conditions given in the problem.
def condition1 : Prop := M = A / 2
def condition2 : Prop := A = S / 2
def condition3 : Prop := S = 400

-- The statement we need to prove.
theorem total_amount_divided (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  M + A + S = 700 :=
by
  -- We omit the proof and assume it is provided later.
  sorry

end total_amount_divided_l115_115224


namespace find_M_l115_115383

theorem find_M :
  (∃ M: ℕ, (10 + 11 + 12) / 3 = (2022 + 2023 + 2024) / M) → M = 551 :=
by
  sorry

end find_M_l115_115383


namespace fraction_equiv_l115_115048

theorem fraction_equiv (x y : ℝ) (h : x / 2 = y / 5) : x / y = 2 / 5 :=
by
  sorry

end fraction_equiv_l115_115048


namespace lateral_surface_area_base_area_ratio_correct_l115_115244

noncomputable def lateral_surface_area_to_base_area_ratio
  (S P Q R : Type)
  (angle_PSR angle_SQR angle_PSQ : ℝ)
  (h_PSR : angle_PSR = π / 2)
  (h_SQR : angle_SQR = π / 4)
  (h_PSQ : angle_PSQ = 7 * π / 12)
  : ℝ :=
  π * (4 * Real.sqrt 3 - 3) / 13

theorem lateral_surface_area_base_area_ratio_correct
  {S P Q R : Type}
  (angle_PSR angle_SQR angle_PSQ : ℝ)
  (h_PSR : angle_PSR = π / 2)
  (h_SQR : angle_SQR = π / 4)
  (h_PSQ : angle_PSQ = 7 * π / 12) :
  lateral_surface_area_to_base_area_ratio S P Q R angle_PSR angle_SQR angle_PSQ
    h_PSR h_SQR h_PSQ = π * (4 * Real.sqrt 3 - 3) / 13 :=
  by sorry

end lateral_surface_area_base_area_ratio_correct_l115_115244


namespace value_to_subtract_l115_115784

variable (x y : ℝ)

theorem value_to_subtract (h1 : (x - 5) / 7 = 7) (h2 : (x - y) / 8 = 6) : y = 6 := by
  sorry

end value_to_subtract_l115_115784


namespace firing_sequence_hits_submarine_l115_115886

theorem firing_sequence_hits_submarine :
  ∀ (v : ℕ), ∃ (n : ℕ), v * v = n * n :=
by
  intro v
  use v
  sorry

end firing_sequence_hits_submarine_l115_115886


namespace minimum_value_l115_115450

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∃ c : ℝ, c = \sqrt{5} + 1 ∧ 
    ∀ x y : ℝ, 0 < x → 0 < y → 
    c ≤ \frac{\sqrt{(x^2 + y^2) * (5 * x^2 + y^2)}}{x * y}) := 
begin
  sorry
end

end minimum_value_l115_115450


namespace Alice_min_possible_number_l115_115342

def min_possible_M (n r : ℕ) (p : fin r → ℕ) : ℕ :=
  ∏ i, p i

theorem Alice_min_possible_number (n r : ℕ) (p : fin r → ℕ) :
  let M := min_possible_M n r p in
  (∃ (M : ℕ), (M = min_possible_M n r p) ∧
    (∃ (m : ℕ), (m = M ^ (n / 2)) ∧
    ∀ a b : ℕ, ∀ g : gcd a b, ∀ l : lcm a b, g ≤ m ∧ l ≤ m)) := sorry

end Alice_min_possible_number_l115_115342


namespace tan_difference_l115_115316

open Real

noncomputable def tan_difference_intermediate (θ : ℝ) : ℝ :=
  (tan θ - tan (π / 4)) / (1 + tan θ * tan (π / 4))

theorem tan_difference (θ : ℝ) (h1 : cos θ = -12 / 13) (h2 : π < θ ∧ θ < 3 * π / 2) :
  tan (θ - π / 4) = -7 / 17 :=
by
  sorry

end tan_difference_l115_115316


namespace max_value_of_abs_asinx_plus_b_l115_115718

theorem max_value_of_abs_asinx_plus_b 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, |a * (Real.cos x)^2 + b * Real.sin x + c| ≤ 1) : 
  ∃ M, M = 2 ∧ ∀ x : ℝ, |a * Real.sin x + b| ≤ M :=
by
  use 2
  sorry

end max_value_of_abs_asinx_plus_b_l115_115718


namespace four_digit_numbers_neither_5_nor_7_l115_115370

-- Define the range of four-digit numbers
def four_digit_numbers : Set ℕ := {x | 1000 ≤ x ∧ x ≤ 9999}

-- Define the predicates for multiples of 5, 7, and 35
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0
def is_multiple_of_7 (n : ℕ) : Prop := n % 7 = 0
def is_multiple_of_35 (n : ℕ) : Prop := n % 35 = 0

-- Using set notation to define the sets of multiples
def multiples_of_5 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_5 n}
def multiples_of_7 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_7 n}
def multiples_of_35 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_35 n}

-- Total count of 4-digit numbers
def total_four_digit_numbers : ℕ := 9000

-- Count of multiples of 5, 7, and 35 within 4-digit numbers
def count_multiples_of_5 : ℕ := 1800
def count_multiples_of_7 : ℕ := 1286
def count_multiples_of_35 : ℕ := 257

-- Count of multiples of 5 or 7 using the principle of inclusion-exclusion
def count_multiples_of_5_or_7 : ℕ := count_multiples_of_5 + count_multiples_of_7 - count_multiples_of_35

-- Prove that the number of 4-digit numbers which are multiples of neither 5 nor 7 is 6171
theorem four_digit_numbers_neither_5_nor_7 : 
  (total_four_digit_numbers - count_multiples_of_5_or_7) = 6171 := 
by 
  sorry

end four_digit_numbers_neither_5_nor_7_l115_115370


namespace earnings_difference_l115_115159

/-- Let S (Sophia), Sa (Sarah), L (Lisa), J (Jack), and T (Tommy) represent the amounts earned.
Prove that given the following conditions:
1. S + Sa + L + J + T = 120
2. S = 2 * L
3. Sa = S - 10
4. L = (1/2) * (J + T)
5. T = 0.75 * J
Then, the difference between Lisa's earnings (L) and Tommy's earnings (T) is approximately 2.66.
-/
theorem earnings_difference 
    (S Sa L J T : ℝ)
    (h1 : S + Sa + L + J + T = 120)
    (h2 : S = 2 * L)
    (h3 : Sa = S - 10)
    (h4 : L = (1/2) * (J + T))
    (h5 : T = 0.75 * J) : 
    L - T ≈ 2.66 := 
sorry

end earnings_difference_l115_115159


namespace ratio_B_to_A_l115_115967

def work_together_rate : Real := 0.75
def days_for_A : Real := 4

theorem ratio_B_to_A : 
  ∃ (days_for_B : Real), 
    (1/days_for_A + 1/days_for_B = work_together_rate) → 
    (days_for_B / days_for_A = 0.5) :=
by 
  sorry

end ratio_B_to_A_l115_115967


namespace quadratic_two_distinct_real_roots_l115_115001

theorem quadratic_two_distinct_real_roots 
    (a b c : ℝ)
    (h1 : a > 0)
    (h2 : c < 0) : 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + bx + c = 0 := 
sorry

end quadratic_two_distinct_real_roots_l115_115001


namespace midpoints_form_parallelogram_l115_115101

variables {A B C D K L M N : Type}
           [AddGroup A] [VectorSpace ℚ A]
           [AddGroup B] [VectorSpace ℚ B]
           [AddGroup C] [VectorSpace ℚ C]
           [AddGroup D] [VectorSpace ℚ D]
           [AddGroup K] [VectorSpace ℚ K]
           [AddGroup L] [VectorSpace ℚ L]
           [AddGroup M] [VectorSpace ℚ M]
           [AddGroup N] [VectorSpace ℚ N]

-- Midpoints definitions
def midpoint (x y : A) : A := (1/2 : ℚ) • (x + y)

-- Parallelogram property
theorem midpoints_form_parallelogram
  (HA : A) (HB : B) (HC : C) (HD : D)
  (HK : K = midpoint HA HB)
  (HL : L = midpoint HB HC)
  (HM : M = midpoint HC HD)
  (HN : N = midpoint HD HA) :
  ∃ (P Q : Type), (P ∥ Q) ∧ (P = Q) := sorry

end midpoints_form_parallelogram_l115_115101


namespace coefficient_of_term_with_x_degree_1_l115_115392

theorem coefficient_of_term_with_x_degree_1 :
  (∀ (x y : ℝ), (x + y - 1)^3 * (2 * x - y + a)^5) = 32 →
  ∃ (a : ℝ), 
    (a = 1 ∧ 
    (coeff_of_x_degree_1 (x y : ℝ, 
      (x + y - 1)^3 * (2 * x - y + a)^5) 1 = -7)) :=
by
  sorry

end coefficient_of_term_with_x_degree_1_l115_115392


namespace correct_statements_count_l115_115648

theorem correct_statements_count :
  let P1 := false
  let P2 := false
  let P3 := true
  let P4 := true
  ([P1, P2, P3, P4].count (λ p, p = true)) = 2 :=
by
  sorry

end correct_statements_count_l115_115648


namespace sin_theta_plus_45_l115_115005

-- Statement of the problem in Lean 4

theorem sin_theta_plus_45 (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) (sin_θ_eq : Real.sin θ = 3 / 5) :
  Real.sin (θ + π / 4) = 7 * Real.sqrt 2 / 10 :=
sorry

end sin_theta_plus_45_l115_115005


namespace parabola_distance_theorem_l115_115988

noncomputable def parabola_focus_distance (p : ℝ) (M : ℝ × ℝ) := 
  let focus := (p / 2, 0)
  in real.sqrt ((M.1 - (p / 2)) ^ 2 + M.2 ^ 2)

theorem parabola_distance_theorem : 
  parabola_focus_distance 1 (2, 2) = 5 / 2 := 
by 
  -- substitute definitions and simplify expressions 
  sorry

end parabola_distance_theorem_l115_115988


namespace intersection_A_B_l115_115713

def A := {y : ℝ | ∃ x : ℝ, y = 2^x}
def B := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2}
def Intersection := {y : ℝ | 0 < y ∧ y ≤ 2}

theorem intersection_A_B :
  (A ∩ B) = Intersection :=
by
  sorry

end intersection_A_B_l115_115713


namespace sum_of_rel_prime_greater_than_one_l115_115138

theorem sum_of_rel_prime_greater_than_one (a : ℕ) (h : a > 6) : 
  ∃ b c : ℕ, a = b + c ∧ b > 1 ∧ c > 1 ∧ Nat.gcd b c = 1 :=
sorry

end sum_of_rel_prime_greater_than_one_l115_115138


namespace most_likely_number_of_red_balls_l115_115540

-- Define the total number of balls and the frequency of picking red balls as given in the conditions
def total_balls : ℕ := 20
def frequency_red : ℝ := 0.8

-- State the equivalent proof problem: Prove that the most likely number of red balls is 16
theorem most_likely_number_of_red_balls : frequency_red * (total_balls : ℝ) = 16 := by
  sorry

end most_likely_number_of_red_balls_l115_115540


namespace log_eq_7_implies_x_value_l115_115778

theorem log_eq_7_implies_x_value (x : ℝ) : log 4 (x^2) + log (1/4) x = 7 → x = 16384 := by
  sorry

end log_eq_7_implies_x_value_l115_115778


namespace no_solution_for_squares_l115_115870

theorem no_solution_for_squares (x y : ℤ) (hx : x > 0) (hy : y > 0) :
  ¬ ∃ k m : ℤ, x^2 + y + 2 = k^2 ∧ y^2 + 4 * x = m^2 :=
sorry

end no_solution_for_squares_l115_115870


namespace find_y_l115_115105

noncomputable def bowtie (a b : ℝ) : ℝ := 
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + ...)))

theorem find_y (y : ℝ) (h : bowtie 4 y = 10) : y = 30 :=
sorry

end find_y_l115_115105


namespace index_fifty_b_gt_zero_l115_115681

noncomputable def b (n : ℕ) : ℝ :=
  (∑ k in Finset.range n, Real.cos k)

theorem index_fifty_b_gt_zero : ∃ n, b n > 0 ∧ n = 314 :=
by
  sorry

end index_fifty_b_gt_zero_l115_115681


namespace lily_jog_time_l115_115452

theorem lily_jog_time :
  (∃ (max_time : ℕ) (lily_miles_max : ℕ) (max_distance : ℕ) (lily_time_ratio : ℕ) (distance_wanted : ℕ)
      (expected_time : ℕ),
    max_time = 36 ∧
    lily_miles_max = 4 ∧
    max_distance = 6 ∧
    lily_time_ratio = 3 ∧
    distance_wanted = 7 ∧
    expected_time = 21 ∧
    lily_miles_max * lily_time_ratio = max_time ∧
    max_distance * lily_time_ratio = distance_wanted * expected_time) := 
sorry

end lily_jog_time_l115_115452


namespace find_abs_x_minus_y_l115_115228

noncomputable def contestants := [9, 8.7, 9.3, x, y]

theorem find_abs_x_minus_y {x y : ℝ} 
  (h1 : (9 + 8.7 + 9.3 + x + y) / 5 = 9)
  (h2 : ((0)^2 + (-0.3)^2 + (0.3)^2 + (x - 9)^2 + (y - 9)^2) / 5 = 0.1) :
  |x - y| = 0.8 := by
sorry

end find_abs_x_minus_y_l115_115228


namespace RachelRonaAgeRatio_l115_115148

-- Define variables for ages of Rona, Collete, and Rachel
variables (Rona_age Collete_age Rachel_age : ℕ)

-- Rona's age is given as 8
def RonaAgeIs8 : Prop := Rona_age = 8

-- Collete's age is half of Rona's age
def ColleteAge : Prop := Collete_age = Rona_age / 2

-- The difference between the age of Collete and Rachel is 12 years
def AgeDifference : Prop := Rachel_age - Collete_age = 12

-- Rachel's age to Rona's age ratio is 2:1
def RatioRachelToRona : Prop := Rachel_age / Rona_age = 2

-- Final theorem statement combining all conditions and the desired result
theorem RachelRonaAgeRatio
  (hrona : RonaAgeIs8)
  (hcollete : ColleteAge)
  (hdifference : AgeDifference) :
  RatioRachelToRona :=
sorry

end RachelRonaAgeRatio_l115_115148


namespace fraction_ratio_equivalence_l115_115300

theorem fraction_ratio_equivalence (x y : ℚ) :
  (x / y) / (7 / 15) = 4 / 3 → x / y = 28 / 45 :=
by
  intros h
  have h1 : 3 * (x / y) = 4 * (7 / 15) := by sorry
  have h2 : x / y = (4 * (7 / 15)) / 3 := by sorry
  have h3 : x / y = 28 / 45 := by sorry
  exact h3
  sorry

end fraction_ratio_equivalence_l115_115300


namespace sum_of_sequence_l115_115531

noncomputable def sequence_term (n : ℕ) : ℕ :=
(2^n) - 1

noncomputable def sum_sequence (n : ℕ) : ℕ :=
(nat.range n).sum (λ k, sequence_term (k + 1))

theorem sum_of_sequence (n : ℕ) : sum_sequence n = (2^(n+1)) - n - 2 :=
by sorry

end sum_of_sequence_l115_115531


namespace tangents_of_remaining_angles_neq_l115_115404

-- Given a convex quadrilateral where the tangent of one angle is equal to m
-- Prove that the tangents of the remaining three angles cannot all be equal to m
theorem tangents_of_remaining_angles_neq (m : ℝ) :
  ∀ (a b c d : ℝ), (a + b + c + d = 2 * π) ∧ (a < π) ∧ (b < π) ∧ (c < π) ∧ (d < π) ∧ 
  (tan a = m) →
  tan b ≠ m ∨ tan c ≠ m ∨ tan d ≠ m :=
by
  sorry

end tangents_of_remaining_angles_neq_l115_115404


namespace Carter_card_number_l115_115123

-- Definitions based on conditions
def Marcus_cards : ℕ := 210
def difference : ℕ := 58

-- Definition to infer the number of Carter's baseball cards
def Carter_cards : ℕ := Marcus_cards - difference

-- Theorem statement asserting the number of baseball cards Carter has
theorem Carter_card_number : Carter_cards = 152 := by
  sorry

end Carter_card_number_l115_115123


namespace probability_point_above_x_axis_in_trapezoid_l115_115193

noncomputable def PQRS_vertices : list (ℝ × ℝ) := [(-3, 3), (3, 3), (5, -1), (-5, -1)]

theorem probability_point_above_x_axis_in_trapezoid (P Q R S : ℝ × ℝ)
  (hPQ : P = (-3, 3)) (hQ : Q = (3, 3))
  (hR : R = (5, -1)) (hS : S = (-5, -1))
  (h_trapezoid : [P, Q, R, S] = PQRS_vertices) :
  let total_area := (1 / 2) * (abs (fst Q - fst P + fst R - fst S)) * (abs (snd Q - snd R)),
      area_above_x := (fst Q - fst P) * snd P in
  (area_above_x / total_area) = 3 / 4 :=
by intros; sorry

end probability_point_above_x_axis_in_trapezoid_l115_115193


namespace problem_part_I_problem_part_II_l115_115360

noncomputable def f (x : ℝ) : ℝ := (√3 / 2) * Real.sin (2 * x) - (Real.sin (2 * x / 2))^2 + 1 / 2

-- Given the conditions and correct answer
def omega_conditions (f : ℝ → ℝ) (omega : ℝ) : Prop :=
  (f x = (√3 / 2) * Real.sin (omega * x) - (Real.sin (omega * x / 2))^2 + 1 / 2) ∧ omega > 0 ∧
  (∃ (p : ℝ), p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = π)

theorem problem_part_I : ∃ ω : ℝ, omega_conditions f ω → ω = 2 ∧ 
  ∀ k : ℤ, ∀ x, k*π - π/3 ≤ x → x ≤ k*π + π/6 →
  strict_increasing_on f (set.Icc (k*π-π/3) (k*π+π/6)) :=
sorry

theorem problem_part_II : ∀ x ∈ set.Icc (0 : ℝ) (π / 2),
  -1 / 2 ≤ f x ∧ f x ≤ 1 :=
sorry

end problem_part_I_problem_part_II_l115_115360


namespace index_fifty_b_gt_zero_l115_115680

noncomputable def b (n : ℕ) : ℝ :=
  (∑ k in Finset.range n, Real.cos k)

theorem index_fifty_b_gt_zero : ∃ n, b n > 0 ∧ n = 314 :=
by
  sorry

end index_fifty_b_gt_zero_l115_115680


namespace angle_B_in_triangle_l115_115424

theorem angle_B_in_triangle (b C c : ℝ) (hb : b = 2) (hC : C = π / 3) (hc : c = sqrt 3) :
  B = π / 2 :=
sorry

end angle_B_in_triangle_l115_115424


namespace integral_eq_l115_115575

open Real
open IntervalIntegral

theorem integral_eq : 
  ∫ x in 0..1, (4 - 16 * x) * sin (4 * x) = (4 * 1 - 1) * cos (4 * 1) - sin (4 * 1) + (∫ x in 0..1, 0) :=
by
  sorry

end integral_eq_l115_115575


namespace exceptional_performance_net_salary_good_performance_net_salary_average_performance_net_salary_l115_115155

constant S : ℝ -- Sharon's initial weekly salary
constant S_initial : S = 560 -- Given that initial salary is $560.

-- Define salary and net salary calculations for each performance scenario
def exceptional_performance_salary := 1.25 * S
def good_performance_salary := 1.20 * S
def average_performance_salary := 1.15 * S

def exceptional_bonus := 0.05 * exceptional_performance_salary
def good_bonus := 0.03 * good_performance_salary

def exceptional_net_salary := exceptional_performance_salary + exceptional_bonus - 0.10 * exceptional_performance_salary - 0.05 * exceptional_performance_salary - 50
def good_net_salary := good_performance_salary + good_bonus - 0.10 * good_performance_salary - 0.05 * good_performance_salary - 50
def average_net_salary := average_performance_salary - 0.10 * average_performance_salary - 0.05 * average_performance_salary - 50

-- Prove the net salaries match the correct answers
theorem exceptional_performance_net_salary : exceptional_net_salary = 580 := by
  sorry

theorem good_performance_net_salary : good_net_salary = 541.36 := by
  sorry

theorem average_performance_net_salary : average_net_salary = 497.40 := by
  sorry

end exceptional_performance_net_salary_good_performance_net_salary_average_performance_net_salary_l115_115155


namespace dress_total_price_correct_l115_115977

-- Define constants and variables
def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

-- Function to calculate sale price after discount
def sale_price (op : ℝ) (dr : ℝ) : ℝ := op - (op * dr)

-- Function to calculate total price including tax
def total_selling_price (sp : ℝ) (tr : ℝ) : ℝ := sp + (sp * tr)

-- The proof statement to be proven
theorem dress_total_price_correct :
  total_selling_price (sale_price original_price discount_rate) tax_rate = 96.6 :=
  by sorry

end dress_total_price_correct_l115_115977


namespace alex_baskets_correct_l115_115796

def alex_baskets (A : ℕ) : Prop :=
  let S := 3 * A in
  let H := 2 * S in
  A + S + H = 80

theorem alex_baskets_correct {A : ℕ} (h : alex_baskets A) : A = 8 :=
by {
  sorry
}

end alex_baskets_correct_l115_115796


namespace max_f_value_range_of_a_range_of_k_l115_115320

-- Defining the function f
def f (x : ℝ) : ℝ := (1 + 2 * Real.log x) / (x^2)

-- Defining the function g
def g (x : ℝ) (a : ℝ) : ℝ := a * x^2 - 2 * Real.log x

-- Lean statements
theorem max_f_value : ∀ x : ℝ, 0 < x → f x ≤ 1 := sorry

theorem range_of_a (a : ℝ) : a > 0 ∧ a < 1 ↔ ∀ x : ℝ, x > 0 → g x a ≠ 1 := sorry

theorem range_of_k (k : ℝ) : k < (2 / Real.exp 1) ↔ 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1 > 1 ∧ x2 > 1 ∧ |f x1 - f x2| ≥ k * |Real.log x1 - Real.log x2| := sorry

end max_f_value_range_of_a_range_of_k_l115_115320


namespace share_of_A_eq_70_l115_115152

theorem share_of_A_eq_70 (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 595) : A = 70 :=
sorry

end share_of_A_eq_70_l115_115152


namespace emily_age_proof_l115_115289

theorem emily_age_proof (e m : ℕ) (h1 : e = m - 18) (h2 : e + m = 54) : e = 18 :=
by
  sorry

end emily_age_proof_l115_115289


namespace find_m_value_l115_115766

theorem find_m_value (m : ℝ) (A : Set ℝ) (h₁ : A = {0, m, m^2 - 3 * m + 2}) (h₂ : 2 ∈ A) : m = 3 :=
by
  sorry

end find_m_value_l115_115766


namespace problem_conditions_l115_115719

def a_n (n : ℕ) : ℚ := (2 * n) / (3 * n + 2)

theorem problem_conditions :
  let a := a_n in
  a 3 = 6 / 11 ∧ a (n - 1) = (2 * (n - 1)) / (3 * (n - 1) + 2) ∧ ∃ n, a n = 8 / 13 ∧ n = 8 :=
by
  intro a
  have h1: a 3 = 6 / 11 := sorry,
  have h2: a (n - 1) = (2 * (n - 1)) / (3 * (n - 1) + 2) := sorry,
  have h3: ∃ n, a n = 8 / 13 ∧ n = 8 := sorry,
  exact ⟨h1, h2, h3⟩

end problem_conditions_l115_115719


namespace derivative_at_zero_l115_115266

noncomputable def f : ℝ → ℝ :=
λ x, if x = 0 then 0 else x^2 * Real.cos (4 / (3 * x)) + (x^2 / 2)

theorem derivative_at_zero :
  HasDerivAt f 0 0 :=
sorry

end derivative_at_zero_l115_115266


namespace inequality_solution_l115_115758

theorem inequality_solution (a b x : ℝ) (h1 : (2 * a - b) * x + a - 5 * b > 0)
  (h2 : x < 10 / 7) :
  ax + b > 0 = x < -3 / 5 := by
sorry

end inequality_solution_l115_115758


namespace lowest_possible_students_l115_115226

theorem lowest_possible_students 
    (divisible_by_18 : ∀ n, n = 18 → ∃ m, n * m = 72)
    (divisible_by_24 : ∀ n, n = 24 → ∃ m, n * m = 72)
    (size_diff_auth : ∀ n1 n2, (n1 * 24 = n2 * 18) → |n1 - n2| ≤ 2)
    (skill_set : ∀ t, t ≤ 24 → ∃ s, s ≥ 1)
    (preference : ∀ t, t <= 24 → ∃ p, p ≥ 1) :
    72 = 72 :=
by
    sorry

end lowest_possible_students_l115_115226


namespace stratified_sampling_l115_115234

theorem stratified_sampling (total_students total_freshmen total_sophomores total_juniors sample_size : ℕ)
(h_total_students : total_students = 2700)
(h_total_freshmen : total_freshmen = 900)
(h_total_sophomores : total_sophomores = 1200)
(h_total_juniors : total_juniors = 600)
(h_sample_size : sample_size = 135) :
let freshmen_in_sample := (total_freshmen * sample_size) / total_students,
    sophomores_in_sample := (total_sophomores * sample_size) / total_students,
    juniors_in_sample := (total_juniors * sample_size) / total_students in
freshmen_in_sample = 45 ∧ sophomores_in_sample = 60 ∧ juniors_in_sample = 30 := by {
  sorry
}

end stratified_sampling_l115_115234


namespace difference_mean_median_eq_4_l115_115406

-- Define the percentages
def pct_60 := 0.20
def pct_75 := 0.15
def pct_85 := 0.40
def pct_95 := 1.00 - (pct_60 + pct_75 + pct_85)

-- Define the scores
def score_60 := 60
def score_75 := 75
def score_85 := 85
def score_95 := 95

-- Calculate the mean score
def mean_score := (pct_60 * score_60) + (pct_75 * score_75) + (pct_85 * score_85) + (pct_95 * score_95)

-- Define the median score
def median_score := score_85

-- State the theorem
theorem difference_mean_median_eq_4 :
  median_score - mean_score = 4 := by sorry

end difference_mean_median_eq_4_l115_115406


namespace significant_improvement_l115_115594

section RubberProductElongation

-- Given conditions
def x : Fin 10 → ℕ := ![545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def y : Fin 10 → ℕ := ![536, 527, 543, 530, 560, 533, 522, 550, 576, 536]
def z (i : Fin 10) : ℤ := (x i : ℤ) - (y i : ℤ)

-- Definitions for sample mean and sample variance
def sample_mean (f : Fin 10 → ℤ) : ℤ := (∑ i, f i) / 10
def sample_variance (f : Fin 10 → ℤ) (mean : ℤ) : ℤ := (∑ i, (f i - mean) ^ 2) / 10

-- Correct answers
def z_bar : ℤ := sample_mean z
def s_squared : ℤ := sample_variance z z_bar

-- Proof statement for equivalence
theorem significant_improvement :
  z_bar = 11 ∧
  s_squared = 61 ∧
  (z_bar ≥ 2 * Real.sqrt (s_squared / 10)) :=
by
  sorry

end RubberProductElongation

end significant_improvement_l115_115594


namespace tire_price_l115_115636

theorem tire_price {p : ℤ} (h : 4 * p + 1 = 421) : p = 105 :=
sorry

end tire_price_l115_115636


namespace log_inequality_l115_115436

noncomputable def a := Real.log 6 / Real.log 3
noncomputable def b := Real.log 10 / Real.log 5
noncomputable def c := Real.log 14 / Real.log 7

theorem log_inequality :
  a > b ∧ b > c :=
by
  sorry

end log_inequality_l115_115436


namespace midpoint_product_is_neg4_l115_115559

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def product_of_coordinates (p : ℝ × ℝ) : ℝ :=
  p.1 * p.2

theorem midpoint_product_is_neg4 :
  let p1 := (1, -4)
  let p2 := (7, 2)
  product_of_coordinates (midpoint p1 p2) = -4 :=
by
  sorry

end midpoint_product_is_neg4_l115_115559


namespace determine_b_l115_115186

theorem determine_b (b : ℚ) (x y : ℚ) (h1 : x = -3) (h2 : y = 4) (h3 : 2 * b * x + (b + 2) * y = b + 6) :
  b = 2 / 3 := 
sorry

end determine_b_l115_115186


namespace c_value_range_l115_115416

theorem c_value_range 
  (c : ℝ)
  (cond : ∃ P, P ∈ ({p : ℝ × ℝ | p.1^2 + p.2^2 = 4} : set (ℝ × ℝ)) ∧ set.count_points_at_distance ({p : ℝ × ℝ | 12 * p.1 - 5 * p.2 + c = 0} : set (ℝ × ℝ)) P = 1)
  : c ∈ set.Ioo (-13 : ℝ) (13 : ℝ) :=
sorry

end c_value_range_l115_115416


namespace cost_per_pound_correct_l115_115151

noncomputable def cost_per_pound_of_coffee (initial_amount spent_amount pounds_of_coffee : ℕ) : ℚ :=
  (initial_amount - spent_amount) / pounds_of_coffee

theorem cost_per_pound_correct :
  let initial_amount := 70
  let amount_left    := 35.68
  let pounds_of_coffee := 4
  (initial_amount - amount_left) / pounds_of_coffee = 8.58 := 
by
  sorry

end cost_per_pound_correct_l115_115151


namespace midpoint_of_segment_l115_115699

theorem midpoint_of_segment (a b : ℝ) : (a + b) / 2 = (a + b) / 2 :=
sorry

end midpoint_of_segment_l115_115699


namespace weight_of_sugar_is_16_l115_115223

def weight_of_sugar_bag (weight_of_sugar weight_of_salt remaining_weight weight_removed : ℕ) : Prop :=
  weight_of_sugar + weight_of_salt - weight_removed = remaining_weight

theorem weight_of_sugar_is_16 :
  ∃ (S : ℕ), weight_of_sugar_bag S 30 42 4 ∧ S = 16 :=
by
  sorry

end weight_of_sugar_is_16_l115_115223


namespace line_circle_intersect_l115_115519

theorem line_circle_intersect {a : ℝ} :
  ∃ P : ℝ × ℝ, (P.1, P.2) = (-2, 0) ∧ (a * P.1 - P.2 + 2 * a = 0) ∧ (P.1^2 + P.2^2 < 9) :=
by
  use (-2, 0)
  sorry

end line_circle_intersect_l115_115519


namespace f_monotonic_increasing_a_bounds_l115_115850

-- Define the conditions
axiom f : ℝ → ℝ
axiom f_domain : ∀ x, x ∈ ℝ
axiom f_additive : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
axiom f_neg : ∀ x : ℝ, x < 0 → f(x) < 0
axiom f_one: f(1) = 1

-- Define the sequence a_n
axiom a : ℕ+ → ℝ
axiom a_initial : 0 < a 1 ∧ a 1 < 1
axiom a_recurrence : ∀ n : ℕ+, 2 - a (n + 1) = f(2 - a n)

-- Question 1: Prove the monotonicity of f(x)
theorem f_monotonic_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) :=
sorry

-- Question 2: Prove 0 < a_n < 1 for all n in ℕ+
theorem a_bounds : ∀ n : ℕ+, 0 < a n ∧ a n < 1 :=
sorry

end f_monotonic_increasing_a_bounds_l115_115850


namespace range_of_k_l115_115344

theorem range_of_k (k : ℝ) :
  (3 < k ∧ k ≤ 4) ∨ (9 / 2 ≤ k ∧ k < 6) ↔
  let p := (3 < k ∧ k < 9 / 2) in
  let q := (4 < k ∧ k < 6) in
  (p ∨ q) ∧ ¬(p ∧ q) :=
by {
  sorry
}

end range_of_k_l115_115344


namespace range_of_a_l115_115917

theorem range_of_a (a : ℝ) :
  (∑ x in finset.Ico a 4, x) = 6 → -1 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l115_115917


namespace amount_paid_for_peaches_l115_115153

def total_spent := 23.86
def cherries_spent := 11.54
def peaches_spent := 12.32

theorem amount_paid_for_peaches :
  total_spent - cherries_spent = peaches_spent :=
sorry

end amount_paid_for_peaches_l115_115153


namespace find_x_if_perpendicular_l115_115038

-- Definitions based on the conditions provided
structure Vector2 := (x : ℚ) (y : ℚ)

def a : Vector2 := ⟨2, 3⟩
def b (x : ℚ) : Vector2 := ⟨x, 4⟩

def dot_product (v1 v2 : Vector2) : ℚ := v1.x * v2.x + v1.y * v2.y

theorem find_x_if_perpendicular :
  ∀ x : ℚ, dot_product a (Vector2.mk (a.x - (b x).x) (a.y - (b x).y)) = 0 → x = 1/2 :=
by
  intro x
  intro h
  sorry

end find_x_if_perpendicular_l115_115038


namespace circle_eq_tangent_line_trajectory_midpoint_M_area_triangle_ABC_l115_115725

variable (x y a b x1 x2 y1 y2 k : ℝ)

theorem circle_eq_tangent_line : 
  (∃ a > 0, ∀ x y, (x - a)^2 + y^2 = 4 ↔ ∃ C, C ∈ ℝ
  ∧ 3*x - 4*y + 4 = 0 → (Real.abs (3*a + 4) = 10)) → (x - 2)^2 + y^2 = 4 :=
by sorry

theorem trajectory_midpoint_M :
  (∀ a y b, ∀ P : ℝ, P ∈ ℝ → ∀ Q : ℝ, Q = 0 ∧ ∀ M : ℝ, M = Q
  ∧ (P - Q) / 2 → (a = 2 * x ∨ b = 2 * y - 3) ∧ (x, y) ∈ P) → x^2 + (y - 1.5)^2 = 1.25 :=
by sorry

theorem area_triangle_ABC :
  (∀ Q : ℝ, Q = 0 ∧ ∃ l, l = Q → ∀ A B, (A, B) ∈ (x, y, a, b) 
  → (x1 * x2 + y1 * y2 = 3) ∧ (d = Real.abs ((2 - 3) / Real.sqrt 2))) → 
  (|AB| = 2*Real.sqrt 2^2 - (Real.sqrt 2)/2^2 ∧ 
    |O| = 3 / Real.sqrt 2) → (1/2 * |AB| * |O| = 1/2 * Real.sqrt 14 * 3*Real.sqrt 2 / 2 
    = 3*Real.sqrt 7 / 2) :=
by sorry

end circle_eq_tangent_line_trajectory_midpoint_M_area_triangle_ABC_l115_115725


namespace table_tennis_match_outcomes_l115_115935

theorem table_tennis_match_outcomes : 
  (∃ (P1 P2 : ℕ), (∀ n, P1 + P2 = 3 + n) → P1 ≠ P2) ∧ 
  (P1 < 3 ∧ P2 < 3 → P1 = 3 ∨ P2 = 3) → 
  30 := 
sorry

end table_tennis_match_outcomes_l115_115935


namespace min_value_of_odd_function_l115_115842

variable {R : Type*} [LinearOrderedField R]

-- Define the odd function f with given properties
variable (f : R → R)
  (h_odd : ∀ x : R, f (-x) = -f x)
  (h_add : ∀ x y : R, f (x + y) = f x + f y)
  (h_neg : ∀ x : R, x > 0 → f x < 0)

theorem min_value_of_odd_function (a b : R) (h_ab : a ≤ b) : ∃ c ∈ set.Icc a b, ∀ x ∈ set.Icc a b, f c ≤ f x :=
  sorry

end min_value_of_odd_function_l115_115842


namespace time_to_save_for_downpayment_l115_115457

def annual_salary : ℝ := 120000
def savings_percentage : ℝ := 0.15
def house_cost : ℝ := 550000
def downpayment_percentage : ℝ := 0.25

def annual_savings : ℝ := savings_percentage * annual_salary
def downpayment_needed : ℝ := downpayment_percentage * house_cost

theorem time_to_save_for_downpayment :
  (downpayment_needed / annual_savings) = 7.64 :=
by
  -- Proof to be provided
  sorry

end time_to_save_for_downpayment_l115_115457


namespace speed_ratio_hexagon_l115_115264

-- Define the regular hexagon with side length a
def regular_hexagon_exists (a : ℝ) : Prop :=
  ∃ (A B C D E F P M N : ℝ × ℝ),
  -- P is a point on the side AB of the hexagon ABCDEF
  P ∈ segment A B ∧
  -- Line PM is parallel to CD and intersects EF at M
  ((M ∈ (λ t : ℝ, P + t • (C - D))) ∧ (M ∈ segment E F)) ∧
  -- Line PN is parallel to BC and intersects CD at N
  ((N ∈ (λ t : ℝ, P + t • (B - C))) ∧ (N ∈ segment C D)) ∧
  -- Red sprite travels the path NPMED
  let red_path := [N, P, M, E, D] in
  -- Blue sprite travels the path CBAFED
  let blue_path := [C, B, A, F, E, D] in
  -- The distance function given the side length a
  let distance (path : list (ℝ × ℝ)) := list.sum (list.map (λ (p1 p2 : ℝ × ℝ), real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)) (list.zip path (list.tail path))) in
  -- Distances of the paths
  distance red_path = 5 * a ∧
  distance blue_path = 6 * a ∧
  -- Proportion of the speeds given the sprites return to the point N at the same time
  ∀ (speed_red speed_blue : ℝ), speed_blue / speed_red = distance blue_path / distance red_path → speed_blue / speed_red = 1.2

theorem speed_ratio_hexagon (a : ℝ) (h : regular_hexagon_exists a) : 
  ∀ (speed_red speed_blue : ℝ), speed_blue / speed_red = 1.2 :=
by
  cases h with A hA,
  cases hA with B hAB,
  cases hAB with C hABC,
  cases hABC with D hABCD,
  cases hABCD with E hABCDE,
  cases hABCDE with F hABCDEF,
  cases hABCDEF with P hP,
  cases hP with M hPM,
  cases hPM with N hPN_and_paths,
  let red_path := [N, P, M, E, D] in
  let blue_path := [C, B, A, F, E, D] in
  have dist_red := ... -- compute the distance of the red_path
  have dist_blue := ... -- compute the distance of the blue_path
  have ratio := dist_blue / dist_red,
  sorry

end speed_ratio_hexagon_l115_115264


namespace find_y_l115_115737

theorem find_y (y : ℝ) (h : 3^y + 3^y + 3^y + 3^y + 3^y = 729) : y = 6 - real.logb 3 5 :=
by
  sorry

end find_y_l115_115737


namespace values_of_X_l115_115205

theorem values_of_X (X : ℝ) : X^2 = 6 * X → (X = 0 ∨ X = 6) :=
by
  intro h
  by_cases h0 : X = 0
  . left; exact h0
  . right
    have h1 : X ≠ 0 := mt eq.symm h0
    have h2 : X * (X - 6) = 0 := by
      rw [← h, mul_sub, mul_div_cancel' X h1]
    exact eq_zero_or_eq_zero_of_mul_eq_zero h2

end values_of_X_l115_115205


namespace problem_I_problem_II_problem_III_l115_115028

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := ln (a * x + 1) + x^3 - x^2 - a * x

theorem problem_I (a : ℝ) : (∃ x : ℝ, x = 2 / 3 ∧ deriv (λ x, f x a) x = 0) ↔ a = 0 :=
sorry

theorem problem_II (a : ℝ) : (∀ x : ℝ, 1 ≤ x → deriv (λ x, f x a) x ≥ 0) ↔ 0 < a ∧ a ≤ (1 + Real.sqrt 5) / 2 :=
sorry

noncomputable def f_transformed (x : ℝ) : ℝ := ln (x + 1) + x^3 - x^2 + x

theorem problem_III (b : ℝ) : (∃ x : ℝ, 0 < x ∧ f_transformed (1 - x) - (1 - x)^3 = b / x) ↔ b ∈ Iic 0 :=
sorry

end problem_I_problem_II_problem_III_l115_115028


namespace polar_curve_is_circle_l115_115171

theorem polar_curve_is_circle (ρ θ : ℝ) (h : ρ = sin θ + cos θ) :
  ∃ (x y : ℝ), ((x - 1/2)^2 + (y - 1/2)^2 = 1/2) :=
sorry

end polar_curve_is_circle_l115_115171


namespace angle_bisector_quadrilateral_sum_l115_115324

theorem angle_bisector_quadrilateral_sum 
  (α β γ δ : ℝ)
  (Hαβγδ : α + β + γ + δ = 360) : 
  (α / 2 + β / 2) + (γ / 2 + δ / 2) = 180 :=
by 
  calc
    (α / 2 + β / 2) + (γ / 2 + δ / 2) 
        = (α + β + γ + δ) / 2 : by ring
    ... = 360 / 2            : by rw Hαβγδ
    ... = 180                : by norm_num

end angle_bisector_quadrilateral_sum_l115_115324


namespace geometric_sequence_general_formula_l115_115741

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ := a 2 * q^(n-2)

theorem geometric_sequence_general_formula :
  ∀ (a : ℕ → ℝ) (q : ℝ),
  (|a 2 - a 3| = 14) →
  (a 1 * a 2 * a 3 = 343) →
  (∀ n, a n > 0) →
  (q = 3) →
  a 2 = 7 →
  ∀ n, a n = geometric_sequence a q n :=
begin
  intros a q h_condition1 h_condition2 h_positive h_q h_a2,
  sorry,
end

end geometric_sequence_general_formula_l115_115741


namespace sum_sequence_l115_115529

-- Define the sequence term
def sequence_term (k : ℕ) : ℕ :=
  ∑ i in finset.range k, 2^i

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (n : ℕ) : ℕ :=
  ∑ k in finset.range n, sequence_term (k + 1)

theorem sum_sequence (n : ℕ) : sum_first_n_terms n = 2^(n+1) - n - 2 :=
by
sorry

end sum_sequence_l115_115529


namespace master_liu_problems_l115_115489

-- Define the distances traveled
def distances : List Int := [-3, -15, 19, -1, 5, -12, -6, 12]

-- Define passenger status
def passenger : List Bool := [false, true, true, false, true, true, true, true]

-- Define the total distance calculation
def total_distance (distances : List Int) : Int :=
  distances.sum

-- Define the absolute distance calculation for fuel consumption
def total_distance_abs (distances : List Int) : Int :=
  distances.map (Int.abs).sum

-- Define the fuel consumption rate
def fuel_consumption_rate : Float := 0.06

-- Define the initial fuel in tank
def initial_fuel : Float := 7.0

-- Define the total fuel consumed
def total_fuel_consumed (abs_distance : Int) : Float :=
  fuel_consumption_rate * abs_distance.toFloat

-- Define the remaining fuel
def remaining_fuel (initial : Float) (consumed : Float) : Float :=
  initial - consumed

-- Define fare calculation for a single trip
def fare (distance : Int) : Float :=
  if distance <= 2 then 10.0
  else 10.0 + (distance - 2).toFloat * 1.6

-- Define the total revenue calculation
def total_revenue (distances : List Int) (passenger : List Bool) : Float :=
  (distances.zip passenger).filter (λ p => p.2).map (λ p => fare p.1).sum

-- Main problem statement
theorem master_liu_problems :
  total_distance distances = -1 ∧
  let abs_dist := total_distance_abs distances in
  remaining_fuel initial_fuel (total_fuel_consumed abs_dist) = 2.62 ∧
  remaining_fuel initial_fuel (total_fuel_consumed abs_dist) > 2.0 ∧
  total_revenue distances passenger = 151.2 := 
by
  sorry

end master_liu_problems_l115_115489


namespace sum_items_l115_115430

theorem sum_items (A B : ℕ) (h1 : A = 585) (h2 : A = B + 249) : A + B = 921 :=
by
  -- Proof step skipped
  sorry

end sum_items_l115_115430


namespace tangent_line_at_one_minimum_t_inequality_log_n_harmonic_series_inequality_l115_115753

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.log x / (x + 1)

-- Derivative of the function f(x). We do not need to state the derivative explicitly as we skip the proof with sorry.
noncomputable def f' (x : ℝ) : ℝ := (x + 1 - x * Real.log x) / (x * (x + 1)^2)

-- Statement for the tangent line problem at x = 1
theorem tangent_line_at_one : ∀ (x y : ℝ), (f'(1) = 1 / 2) ∧ (f(1) = 0) → x - 2 * y - 1 = 0 := 
by
  sorry

-- Statement for the minimum value of t such that inequality holds
theorem minimum_t_inequality : ∀ (x t : ℝ), x > 0 → (t ≥ (2 * x - x * Real.log x) / (x + 1)) → (t ≥ 1) := 
by 
  sorry

-- Statement for the inequality involving natural logarithm and harmonic series
theorem log_n_harmonic_series_inequality : ∀ (n : ℕ), n ≥ 2 → Real.log n > (1 / 2 + ∑ i in Finset.range (n - 1), 1 / (i + 2)) := 
by
  sorry

end tangent_line_at_one_minimum_t_inequality_log_n_harmonic_series_inequality_l115_115753


namespace human_ages_when_dog_ages_seven_years_l115_115310

theorem human_ages_when_dog_ages_seven_years :
  (∀ (d : ℕ), d = 7 → ∃ (h : ℚ), h = (7 / 6) * d) :=
by
  assume d,
  assume h_dog_ages_seven : d = 7,
  existsi (7/6 : ℚ) * d,
  sorry

end human_ages_when_dog_ages_seven_years_l115_115310


namespace problem_statement_l115_115789

noncomputable def S (A B C : ℝ) : ℝ :=
  sqrt (3 * tan(A / 2) * tan(B / 2) + 1) +
  sqrt (3 * tan(B / 2) * tan(C / 2) + 1) +
  sqrt (3 * tan(C / 2) * tan(A / 2) + 1)

theorem problem_statement (A B C : ℝ) (h : A + B + C = Real.pi) :
  ⌊S A B C⌋ = 4 :=
sorry

end problem_statement_l115_115789


namespace cone_volume_formula_l115_115492

open Real

noncomputable def volume_of_cone (α a : ℝ) : ℝ :=
  (a^3 * π * cos α * (sin (α / 2))^2) / (6 * (cos (α / 2))^4)

theorem cone_volume_formula (α a : ℝ) (hα : 0 < α) (ha : 0 < a) :
  let SO := a - (a / (1 + cos α)) in
  let SA := (a / (1 + cos α)) in
  let H := a * cos α / (1 + cos α) in
  let R := a * sin α / (1 + cos α) in
  volume_of_cone α a = (1 / 3) * π * R^2 * H :=
begin
  sorry
end

end cone_volume_formula_l115_115492


namespace shelves_fit_l115_115827

-- Define the total space of the room for the library
def totalSpace : ℕ := 400

-- Define the space each bookshelf takes up
def spacePerBookshelf : ℕ := 80

-- Define the reserved space for desk and walking area
def reservedSpace : ℕ := 160

-- Define the space available for bookshelves
def availableSpace : ℕ := totalSpace - reservedSpace

-- Define the number of bookshelves that can fit in the available space
def numberOfBookshelves : ℕ := availableSpace / spacePerBookshelf

-- The theorem stating the number of bookshelves Jonas can fit in the room
theorem shelves_fit : numberOfBookshelves = 3 := by
  -- We can defer the proof as we only need the statement for now
  sorry

end shelves_fit_l115_115827


namespace tents_required_l115_115455

-- Define the structure of the family members and their sleeping requirements
inductive FamilyMember
| Matt | Mom | Dad | OlderBrother | OlderBrotherWife | Kid1 | Kid2 | Kid3 | Kid4
| YoungerSister | YoungerSisterHusband | Infant | Kid5
| Grandfather | Grandmother
| Max
| UncleJoe | UncleJoeWife | UncleJoeKid1 | UncleJoeKid2 | UncleJoeKid3
| UncleJoeBrotherInLaw

open FamilyMember

def indoors : Set FamilyMember := {Grandfather, Grandmother, Mom, Dad, YoungerSister, YoungerSisterHusband, Infant}
def refuses_to_sleep_with_siblings : Set FamilyMember := {Kid1, Kid2}
def needs_own_tent : Set FamilyMember := {Max}

def tents_needed : ℕ := 1 + 1 + 1 + 1 + 1 + 1 + 1

-- Theorem that encapsulates the problem statement
theorem tents_required (h_disjoint: ∀ x ∈ indoors, x ∉ ((induct {places: "indoors"})  ++ {FamilyMember}) :
  tents_needed = 7 :=
begin
  sorry
end

end tents_required_l115_115455


namespace most_likely_number_of_red_balls_l115_115542

-- Define the total number of balls and the frequency of picking red balls as given in the conditions
def total_balls : ℕ := 20
def frequency_red : ℝ := 0.8

-- State the equivalent proof problem: Prove that the most likely number of red balls is 16
theorem most_likely_number_of_red_balls : frequency_red * (total_balls : ℝ) = 16 := by
  sorry

end most_likely_number_of_red_balls_l115_115542


namespace largest_possible_b_l115_115106

theorem largest_possible_b (b : ℚ) (h : (3 * b + 7) * (b - 2) = 9 * b) : b ≤ 2 :=
sorry

end largest_possible_b_l115_115106


namespace total_water_hold_l115_115428

variables
  (first : ℕ := 100)
  (second : ℕ := 150)
  (third : ℕ := 75)
  (total : ℕ := 325)

theorem total_water_hold :
  first + second + third = total := by
  sorry

end total_water_hold_l115_115428


namespace sum_sequence_l115_115528

-- Define the sequence term
def sequence_term (k : ℕ) : ℕ :=
  ∑ i in finset.range k, 2^i

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (n : ℕ) : ℕ :=
  ∑ k in finset.range n, sequence_term (k + 1)

theorem sum_sequence (n : ℕ) : sum_first_n_terms n = 2^(n+1) - n - 2 :=
by
sorry

end sum_sequence_l115_115528


namespace complementary_angles_difference_l115_115907

theorem complementary_angles_difference :
  ∃ (θ₁ θ₂ : ℝ), θ₁ + θ₂ = 90 ∧ 5 * θ₁ = 3 * θ₂ ∧ abs (θ₁ - θ₂) = 22.5 :=
by
  sorry

end complementary_angles_difference_l115_115907


namespace ellipse_and_line_equation_l115_115339

theorem ellipse_and_line_equation (a b c : ℝ) (h₀ : a > b) (h₁ : b > 0) (h₂ : 2 * a = 4) (h₃ : c / a = 1 / 2) 
  (h₄ : a^2 = b^2 + c^2) (h₅ : 2 * abs a * c = 3) :
  (a = 2 ∧ c = 1 ∧ b^2 = 3 ∧ ellipse_eq : ∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ∧ 
  (line_eq : ∀ x y : ℝ, (3 * x + 4 * y + 3 = 0) ∨ (3 * x - 4 * y + 3 = 0))) :=
by
  sorry

end ellipse_and_line_equation_l115_115339


namespace necessary_but_not_sufficient_condition_l115_115050

theorem necessary_but_not_sufficient_condition (α β : ℝ) (k : ℤ) :
  (sin α * cos β + cos α * sin β = 1 / 2) ↔ 
  (∃ k : ℤ, α + β = 2 * ↑k * π + π / 6) :=
sorry

end necessary_but_not_sufficient_condition_l115_115050


namespace sum_sequence_l115_115530

-- Define the sequence term
def sequence_term (k : ℕ) : ℕ :=
  ∑ i in finset.range k, 2^i

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (n : ℕ) : ℕ :=
  ∑ k in finset.range n, sequence_term (k + 1)

theorem sum_sequence (n : ℕ) : sum_first_n_terms n = 2^(n+1) - n - 2 :=
by
sorry

end sum_sequence_l115_115530


namespace find_m_value_l115_115764

theorem find_m_value (m : ℝ) (A : Set ℝ) (h₁ : A = {0, m, m^2 - 3 * m + 2}) (h₂ : 2 ∈ A) : m = 3 :=
by
  sorry

end find_m_value_l115_115764


namespace minimum_value_of_f_l115_115381

noncomputable def f (x : ℝ) : ℝ := x + 4 / (x - 1)

theorem minimum_value_of_f (x : ℝ) (hx : x > 1) : (∃ y : ℝ, f x = 5 ∧ ∀ y > 1, f y ≥ 5) :=
sorry

end minimum_value_of_f_l115_115381


namespace smallest_c_for_inverse_l115_115108

noncomputable def g (x : ℝ) : ℝ := (x - 3)^2 + 6

theorem smallest_c_for_inverse :
  ∃ c, (∀ x y : ℝ, c ≤ x → c ≤ y → g x = g y → x = y) ∧ (∀ d, (∀ x y : ℝ, d ≤ x → d ≤ y → g x = g y → x = y) → c ≤ d) :=
by
  use 3
  split
  · intros x y hx hy h
    have : (x - 3)^2 = (y - 3)^2 := by
      linarith
    exact eq_of_sq_eq_sq (sub_nonneg_of_le hx) (sub_nonneg_of_le hy) this
  · intros d hd
    by_contra H
    push_neg at H
    have : d < 3 := H
    have key := hd 3 (3 - (3 - d)) (by linarith) (by linarith) (by linarith)
    linarith
  sorry

end smallest_c_for_inverse_l115_115108


namespace inverse_of_g_at_1_over_32_l115_115379

noncomputable def g (x : ℝ) : ℝ := (x^5 + 2) / 4

theorem inverse_of_g_at_1_over_32 :
  g⁻¹ (1/32) = (-15 / 8)^(1/5) :=
sorry

end inverse_of_g_at_1_over_32_l115_115379


namespace part1_part2_min_part2_max_part3_l115_115755

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * a / x - 3 * Real.log x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a + 2 * a / (x^2) - 3 / x

theorem part1 (a : ℝ) : f' a 1 = 0 -> a = 1 := sorry

noncomputable def f1 (x : ℝ) : ℝ := x - 2 / x - 3 * Real.log x

noncomputable def f1' (x : ℝ) : ℝ := 1 + 2 / (x^2) - 3 / x

theorem part2_min (h_a : 1 = 1) : 
    ∀ (x : ℝ), (1 ≤ x) ∧ (x ≤ Real.exp 1) -> 
    (f1 2 <= f1 x) := sorry

theorem part2_max (h_a : 1 = 1) : 
    ∀ (x : ℝ), (1 ≤ x) ∧ (x ≤ Real.exp 1) ->
    (f1 x <= f1 1) := sorry

theorem part3 (a : ℝ) : 
    (∀ (x : ℝ), x > 0 -> f' a x ≥ 0) -> a ≥ (3 * Real.sqrt 2) / 4 := sorry

end part1_part2_min_part2_max_part3_l115_115755


namespace min_points_on_circle_l115_115461

theorem min_points_on_circle (circumference : ℕ) (min_points : ℕ) :
  circumference = 1956 →
  min_points = 1304 →
  ∃ (S : set ℕ) (f : ℕ → ℕ),
    set.card S = min_points ∧
    (∀ p ∈ S, ∃ q ∈ S, q ≠ p ∧ (abs (f q - f p) = 1 ∨ abs (f q - f p) = 2)) :=
by sorry

end min_points_on_circle_l115_115461


namespace number_of_lines_through_lattice_points_l115_115077

theorem number_of_lines_through_lattice_points :
  let lines_with_lattice_points := 
    { l : ℝ × ℝ → ℝ | ∃ k b, ∀ x y, x^2 + y^2 = 5 → y = k * x + b → x ∈ ℤ ∧ y ∈ ℤ } in
  lines_with_lattice_points.card = 32 :=
by sorry

end number_of_lines_through_lattice_points_l115_115077


namespace max_colors_for_cube_edges_l115_115702

theorem max_colors_for_cube_edges : 
  ∃ (n : ℕ), n = 6 ∧
  ∀ (C : Finset (Fin 12)) (colors : Finset (Fin n)), colors.card ≤ 6 → 
  (∀ (c₁ c₂ : Fin 12), c₁ ≠ c₂ → ∃ (e₁ e₂ : Fin 12), are_adjacent_edges e₁ e₂ ∧ e₁ ∈ C ∧ e₂ ∈ C ∧ colors.contains (edges_color e₁) ∧ colors.contains (edges_color e₂)) :=
sorry

namespace Cube
  def are_adjacent_edges (e₁ e₂ : Fin 12) : Prop :=
    -- Definition of adjacency for cube edges. Each edge is connected to 4 others.
    sorry

  def edges_color (e : Fin 12) : Fin :=
    -- Function to determine the color of edge e.
    sorry
end Cube

end max_colors_for_cube_edges_l115_115702


namespace find_c_l115_115901

theorem find_c (x c : ℝ) (h1 : 3 * x + 8 = 5) (h2 : c * x - 15 = -3) : c = -12 := 
by
  -- Equations and conditions
  have h1 : 3 * x + 8 = 5 := h1
  have h2 : c * x - 15 = -3 := h2
  -- The proof script would go here
  sorry

end find_c_l115_115901


namespace Jenny_original_number_l115_115822

theorem Jenny_original_number (y : ℝ) (h : 10 * (y / 2 - 6) = 70) : y = 26 :=
by
  sorry

end Jenny_original_number_l115_115822


namespace cost_per_book_l115_115191

theorem cost_per_book
  (books_sold_each_time : ℕ)
  (people_bought : ℕ)
  (income_per_book : ℕ)
  (profit : ℕ)
  (total_income : ℕ := books_sold_each_time * people_bought * income_per_book)
  (total_cost : ℕ := total_income - profit)
  (total_books : ℕ := books_sold_each_time * people_bought)
  (cost_per_book : ℕ := total_cost / total_books) :
  books_sold_each_time = 2 ->
  people_bought = 4 ->
  income_per_book = 20 ->
  profit = 120 ->
  cost_per_book = 5 :=
  by intros; sorry

end cost_per_book_l115_115191


namespace find_k_l115_115750

theorem find_k (k r s : ℝ) 
  (h1 : r + s = -k) 
  (h2 : r * s = 12) 
  (h3 : (r + 7) + (s + 7) = k) : 
  k = 7 := by 
  sorry

end find_k_l115_115750


namespace log_c_sin_y_eq_l115_115319

variable {c a y : ℝ}

-- Definitions used in the problem conditions
def c_gt_one : Prop := c > 1
def tan_y_gt_zero : Prop := Real.tan y > 0
def log_c_tan_y_eq_a : Prop := Real.log c (Real.tan y) = a
def cos_y_gt_zero : Prop := Real.cos y > 0

-- Main theorem to be proven
theorem log_c_sin_y_eq : c_gt_one → tan_y_gt_zero → log_c_tan_y_eq_a → cos_y_gt_zero → 
  Real.log c (Real.sin y) = a - (1 / 2) * Real.log c (c^(2 * a) + 1) :=
by intros; sorry

end log_c_sin_y_eq_l115_115319


namespace train_length_is_correct_l115_115249

-- Definitions based on the problem's conditions.
def train_speed_kmh : ℝ := 45  -- Speed of the train in km/hr
def bridge_length_m : ℝ := 235  -- Length of the bridge in meters
def time_seconds : ℝ := 30  -- Time to cross the bridge in seconds

-- Conversion factor from km/hr to m/s.
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

-- Calculate the total distance covered by the train in the given time.
def total_distance (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  (kmh_to_ms speed_kmh) * time_sec

-- The length of the train can be calculated as the total distance minus the length of the bridge.
def train_length (speed_kmh : ℝ) (time_sec : ℝ) (bridge_length : ℝ) : ℝ :=
  total_distance speed_kmh time_sec - bridge_length

-- The theorem to be proven
theorem train_length_is_correct : train_length train_speed_kmh time_seconds bridge_length_m = 140 :=
by sorry

end train_length_is_correct_l115_115249


namespace sum_of_first_nine_terms_l115_115337

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def sum_of_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (n / 2) * (a 0 + a (n - 1))

theorem sum_of_first_nine_terms 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_terms a S)
  (h_sum_terms : a 2 + a 3 + a 4 + a 5 + a 6 = 20) :
  S 9 = 36 :=
sorry

end sum_of_first_nine_terms_l115_115337


namespace cube_coloring_distinct_schemes_l115_115313

def distinct_color_count (n_colors n_faces : ℕ) 
  (different_colors : Prop) 
  (adjacent_diff_colors : Prop) : ℕ :=
if different_colors ∧ adjacent_diff_colors then 230 else 0

theorem cube_coloring_distinct_schemes :
  distinct_color_count 6 6 
    (∀ (f1 f2 : Fin 6), f1 ≠ f2 → ∃ (c1 c2 : Fin 6), c1 ≠ c2) 
    (∀ (f1 f2 : Fin 6), is_adjacent f1 f2 → ∃ (c1 c2 : Fin 6), c1 ≠ c2) = 230 :=
sorry

def is_adjacent (f1 f2 : Fin 6) := sorry -- Definition of adjacency on a cube might be needed.

end cube_coloring_distinct_schemes_l115_115313


namespace determine_x_value_l115_115283

theorem determine_x_value : 
  ∃ x : ℝ, 10^x = 10^(-7) * (sqrt (10^105 / 0.0001)) ∧ x = 47.5 := 
sorry

end determine_x_value_l115_115283


namespace sum_sin_cos_l115_115273

theorem sum_sin_cos :
  ∑ k in Finset.range 181, (Real.sin (k * Real.pi / 180))^4 * (Real.cos (k * Real.pi / 180))^4 = 543 / 128 :=
by
  sorry

end sum_sin_cos_l115_115273


namespace part1_part2_l115_115120

-- Condition Definitions
def parabola (x y : ℝ) (p : ℝ) := x^2 = 2 * p * y
def focus (p : ℝ) := (0, p / 2)
def distance (A B : (ℝ × ℝ)) : ℝ := real.dist (A.1, A.2) (B.1, B.2)
def area_ΔAOB (x1 x2 p : ℝ) : ℝ := (1 / 2) * (p / 2) * (abs (x1 - x2))

-- Theorem Statements
theorem part1 {p : ℝ} (hp : p > 0) :
  let C := λ x y, parabola x y p in
  let F := focus p in
  let line := λ x y, y = sqrt 5 * x + p / 2 in
  let intersect_xs := solution of quadratic equation x^2 - 2 * sqrt 5 * p * x - p^2 = 0 in
  let x1 := first solution of intersect_xs
  let x2 := second solution of intersect_xs
  area_ΔAOB x1 x2 p = 2 * sqrt 6 -> p = 2 -> ∀ x y, C x y = (x^2 = 4 * y) :=
sorry

theorem part2 (k1 k2 : ℝ) :
  let E := (0, 2) in
  let l1 := λ x y, y = k1 * x + 2 in
  let l2 := λ x y, y = k2 * x + 2 in
  let intersect_xs_l1 := solution of quadratic equation x^2 - 4 * k1 * x - 8 = 0 in
  let intersect_xs_l2 := solution of quadratic equation x^2 - 4 * k2 * x - 8 = 0 in
  let d_EP_EQ := 8 * (1 + k1^2) in
  let d_ER_ES := 8 * (1 + k2^2) in
  d_EP_EQ = d_ER_ES -> k1^2 = k2^2 -> k1 ≠ k2 -> ∃ λ, k1 + λ * k2 = 0 ∧ λ = 1 :=
sorry

end part1_part2_l115_115120


namespace Tanya_bought_9_apples_l115_115166

def original_fruit_count : ℕ := 18
def remaining_fruit_count : ℕ := 9
def pears_count : ℕ := 6
def pineapples_count : ℕ := 2
def plums_basket_count : ℕ := 1

theorem Tanya_bought_9_apples : 
  remaining_fruit_count * 2 = original_fruit_count →
  original_fruit_count - (pears_count + pineapples_count + plums_basket_count) = 9 :=
by
  intros h1
  sorry

end Tanya_bought_9_apples_l115_115166


namespace probability_one_tshirt_one_jeans_one_hat_l115_115072

theorem probability_one_tshirt_one_jeans_one_hat :
  let tshirts := 3
  let jeans := 7
  let hats := 4
  let total_clothes := tshirts + jeans + hats
  let total_ways := Nat.choose total_clothes 3
  let favorable_ways := tshirts * jeans * hats
  let probability := (favorable_ways : ℚ) / total_ways
  probability = 21 / 91 :=
by
  -- Conditions
  let tshirts := 3
  let jeans := 7
  let hats := 4
  let total_clothes := tshirts + jeans + hats
  let total_ways := Nat.choose total_clothes 3
  let favorable_ways := tshirts * jeans * hats
  let probability := (favorable_ways : ℚ) / total_ways
  
  -- Show that probability is 21/91
  have h1 : favorable_ways = 84 := by sorry
  have h2 : total_ways = 364 := by sorry
  have h3 : probability = (84 : ℚ) / 364 := by sorry
  rw [h1, h2, h3]
  norm_num
  rfl

end probability_one_tshirt_one_jeans_one_hat_l115_115072


namespace shaded_fraction_is_4_over_15_l115_115635

-- Define the given conditions
def shaded_fraction_series : ℕ → ℚ
| 0     := 1
| (n+1) := (1/16) ^ (n + 1)

-- Recursive definition of shaded fraction at any iteration n
def shaded_fraction (n : ℕ) : ℚ :=
4/16 * (∑ i in finset.range (n + 1), shaded_fraction_series i)

-- Problem statement
theorem shaded_fraction_is_4_over_15 (n : ℕ) :
  ∀ n, shaded_fraction n = 4 / 15 :=
begin
  -- This is where the proof would go, but we skip it.
  sorry
end

end shaded_fraction_is_4_over_15_l115_115635


namespace coefficient_x_squared_expansion_correct_l115_115170

noncomputable def coefficient_x_squared_expansion : ℕ :=
  let f := (fun (x : ℤ) => x + 2 + 1/x)
  in (binomial (10:ℕ) 3)

theorem coefficient_x_squared_expansion_correct :
  coefficient_x_squared_expansion = 120 :=
sorry

end coefficient_x_squared_expansion_correct_l115_115170


namespace rounding_to_hundredth_l115_115555

theorem rounding_to_hundredth (x : ℝ) (h : x = 3.8963) : Real.round (x * 100) / 100 = 3.90 :=
by
  sorry

end rounding_to_hundredth_l115_115555


namespace fruit_platter_count_l115_115462

def totalFruitsInPlatter (green_apples red_apples yellow_apples red_oranges yellow_oranges green_kiwis purple_grapes : Nat) : Nat :=
  green_apples + red_apples + yellow_apples + red_oranges + yellow_oranges + green_kiwis + purple_grapes

theorem fruit_platter_count :
  let initial_counts := (2, 3, 14, 4, 8, 10, 7, 5)
  ∃ (green_apples red_apples yellow_apples red_oranges yellow_oranges green_kiwis purple_grapes: Nat),
    -- The fruit ratios and quantities requirement
    green_apples = 2 ∧ red_apples = 4 ∧ yellow_apples = 3 ∧
    red_oranges = 1 ∧ yellow_oranges = 2 ∧ 
    green_kiwis = 7 ∧ purple_grapes = 7 ∧
    -- Prove the total count
    totalFruitsInPlatter green_apples red_apples yellow_apples red_oranges yellow_oranges green_kiwis purple_grapes = 26 :=
by
  let counts := (2, 3, 14, 4, 8, 10, 7, 5)
  use 2, 4, 3, 1, 2, 7, 7
  constructor
  repeat {
    constructor
  }
  exact rfl
  sorry

end fruit_platter_count_l115_115462


namespace least_possible_perimeter_minimal_perimeter_triangle_l115_115184

theorem least_possible_perimeter (x : ℕ) 
  (h1 : 13 < x) 
  (h2 : x < 61) : 
  24 + 37 + x ≥ 24 + 37 + 14 := 
sorry

theorem minimal_perimeter_triangle : ∃ x : ℕ, 13 < x ∧ x < 61 ∧ 24 + 37 + x = 75 :=
begin
  existsi 14,
  split,
  { exact dec_trivial, }, -- 13 < 14
  split,
  { exact dec_trivial, }, -- 14 < 61
  { exact dec_trivial, }, -- 24 + 37 + 14 = 75
end

end least_possible_perimeter_minimal_perimeter_triangle_l115_115184


namespace algebraic_expression_value_l115_115724

theorem algebraic_expression_value 
  (x1 x2 : ℝ)
  (h1 : x1^2 - x1 - 2022 = 0)
  (h2 : x2^2 - x2 - 2022 = 0) :
  x1^3 - 2022 * x1 + x2^2 = 4045 :=
by 
  sorry

end algebraic_expression_value_l115_115724


namespace customer_receives_more_gold_optimal_lambda_l115_115246

-- Variables and conditions
variables (m n : ℝ) (h : m ≠ n) (λ : ℝ := m / n)

-- Definitions related to the conditions
def x := (5 * m) / n
def y := (5 * n) / m
def z := (10 * m) / n

-- Proof problems
theorem customer_receives_more_gold (h: m ≠ n) :
  x + y > 10 := by
  sorry

theorem optimal_lambda :
  λ = (Real.sqrt 3) / 3 := by
  sorry

end customer_receives_more_gold_optimal_lambda_l115_115246


namespace find_positive_x_l115_115297

noncomputable def positive_solutions :=
  { x : ℝ // (1 / 3 * (4 * x ^ 2 - 1) = (x ^ 2 - 60 * x - 12) * (x ^ 2 + 30 * x + 6)) ∧ x > 0 }

theorem find_positive_x : ∃ x ∈ positive_solutions, x = 30 + Real.sqrt 905 ∨ (x = -15 + 4 * Real.sqrt 14 ∧ 4 * Real.sqrt 14 > 15) :=
sorry

end find_positive_x_l115_115297


namespace common_divisors_count_9240_6300_l115_115374

theorem common_divisors_count_9240_6300 : 
  let gcd_val := Nat.gcd 9240 6300 
  in Nat.totient gcd_val = 24 :=
by
  let d := Nat.gcd 9240 6300
  -- gcd(9240, 6300) = 420
  have h1 : d = 420 := by sorry
  have h2 : Nat.totient 420 = 24 := by sorry
  exact h2

end common_divisors_count_9240_6300_l115_115374


namespace solve_geometric_sequence_product_l115_115070

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r

theorem solve_geometric_sequence_product (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h_a35 : a 3 * a 5 = 4) : 
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 = 128 :=
sorry

end solve_geometric_sequence_product_l115_115070


namespace find_m_value_l115_115762

theorem find_m_value (m : ℝ) (h1 : 2 ∈ ({0, m, m^2 - 3 * m + 2} : set ℝ)) : m = 3 :=
sorry

end find_m_value_l115_115762


namespace max_value_of_f_max_value_achieved_max_value_of_function_f_l115_115806

def op (a b : ℝ) : ℝ := if a >= b then a else b^2

def f (x : ℝ) : ℝ := op 1 x + op 2 x

theorem max_value_of_f : 
  x ∈ set.Icc (-2 : ℝ) 3 →  f x ≤ 18 := sorry

theorem max_value_achieved :
  f 3 = 18 := sorry

theorem max_value_of_function_f : 
  ∃ x ∈ set.Icc (-2 : ℝ) 3, f x = 18 := 
begin
  use 3,
  split,
  { norm_num },
  { exact max_value_achieved },
end

end max_value_of_f_max_value_achieved_max_value_of_function_f_l115_115806


namespace range_of_a_l115_115751

def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then (6 - a) * x - 4 * a else log a x

theorem range_of_a : 
  ∀ a: ℝ, (∀ x y: ℝ, x < y → f a x < f a y) ↔ (6 / 5 ≤ a ∧ a < 6) :=
sorry

end range_of_a_l115_115751


namespace definitely_incorrect_conclusions_l115_115018

theorem definitely_incorrect_conclusions (a b c : ℝ) (x1 x2 : ℝ) 
  (h1 : a * x1^2 + b * x1 + c = 0) 
  (h2 : a * x2^2 + b * x2 + c = 0)
  (h3 : x1 > 0) 
  (h4 : x2 > 0) 
  (h5 : x1 + x2 = -b / a) 
  (h6 : x1 * x2 = c / a) : 
  (a > 0 ∧ b > 0 ∧ c > 0) = false ∧ 
  (a < 0 ∧ b < 0 ∧ c < 0) = false ∧ 
  (a > 0 ∧ b < 0 ∧ c < 0) = true ∧ 
  (a < 0 ∧ b > 0 ∧ c > 0) = true :=
sorry

end definitely_incorrect_conclusions_l115_115018


namespace train_speed_approx_l115_115568

def train_length : ℝ := 200
def bridge_length : ℝ := 300
def crossing_time : ℝ := 45

def total_distance : ℝ := train_length + bridge_length
def speed : ℝ := total_distance / crossing_time

theorem train_speed_approx :
  speed ≈ 11.11 := sorry

end train_speed_approx_l115_115568


namespace proof_problem_l115_115546

theorem proof_problem
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2010)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2009)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2010)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2009)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2010)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2009) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -1 :=
by
  sorry

end proof_problem_l115_115546


namespace find_angle_l115_115211

def complementary (x : ℝ) := 90 - x
def supplementary (x : ℝ) := 180 - x

theorem find_angle (x : ℝ) (h : supplementary x = 3 * complementary x) : x = 45 :=
by 
  sorry

end find_angle_l115_115211


namespace separated_circles_m_range_l115_115743

theorem separated_circles_m_range (m R1 R2 : ℝ) (c1 c2 : ℝ × ℝ)
  (h1 : c1 = (0, 0))
  (h2 : R1 = 1)
  (h3 : c2 = (3, 4))
  (h4 : R2 = real.sqrt(25 - m))
  (h5 : 5 > R1 + R2 ∨ abs (R2 - R1) > 5) :
  m ∈ set.Ioo 9 25 ∨ m < -11 :=
by {
  -- dummy proof to ensure the code builds
  sorry
}

end separated_circles_m_range_l115_115743


namespace inscribed_circles_area_correct_l115_115402

-- Define the main variables and conditions used in the problem
def R : ℝ := 10 -- radius of the larger circle
def d_chords : ℝ := 6 -- distance between the parallel chords
def d_from_diameter : ℝ := d_chords / 2 -- distance of each chord from the centerline
def chord_length : ℝ := 2 * (real.sqrt (R^2 - d_from_diameter^2)) -- length of each chord

-- Define the radius of the smaller inscribed circles
def r_inscribed : ℝ := R - d_from_diameter

-- Define the area of a single inscribed circle
def area_inscribed_circle : ℝ := real.pi * r_inscribed^2

-- Define the total area of both inscribed circles
def total_area_inscribed_circles : ℝ := 2 * area_inscribed_circle

-- State the main theorem we need to prove
theorem inscribed_circles_area_correct :
  total_area_inscribed_circles = 98 * real.pi := by
  sorry

end inscribed_circles_area_correct_l115_115402


namespace worker_times_l115_115553

-- Define the problem
theorem worker_times (x y : ℝ) (h1 : (1 / x + 1 / y = 1 / 8)) (h2 : x = y - 12) :
    x = 24 ∧ y = 12 :=
by
  sorry

end worker_times_l115_115553


namespace find_m_l115_115355

-- Definitions based on conditions
def is_eccentricity (a b c : ℝ) (e : ℝ) : Prop :=
  e = c / a

def ellipse_relation (a b m : ℝ) : Prop :=
  a ^ 2 = 3 ∧ b ^ 2 = m

def eccentricity_square_relation (c a : ℝ) : Prop :=
  (c / a) ^ 2 = 1 / 4

-- Main theorem statement
theorem find_m (m : ℝ) :
  (∀ (a b c : ℝ), ellipse_relation a b m → is_eccentricity a b c (1 / 2) → eccentricity_square_relation c a)
  → (m = 9 / 4 ∨ m = 4) := sorry

end find_m_l115_115355


namespace floor_1000_cos_A_l115_115074

-- Statement of the problem translated to Lean 4
theorem floor_1000_cos_A (A B C D : Point)
                         (h_kite : kite A B C D)
                         (h_angles : ∠A = ∠C)
                         (h_AB_AD : AB = 150)
                         (h_AD_BC : AD = 150)
                         (h_BC_CD : BC = 230)
                         (h_CD_AD : CD = 230)
                         (h_perimeter : AB + BC + CD + DA = 760)
                         (h_BD_AC : BD ≠ AC) :
  ∃ cosA : ℝ, cosA = cos (∠A) ∧ ⌊1000 * cosA⌋ = 777 := by
sorry

end floor_1000_cos_A_l115_115074


namespace solve_equation_l115_115527

theorem solve_equation : ∀ x : ℝ, (2 / (x + 5) = 1 / (3 * x)) → x = 1 :=
by
  intro x
  intro h
  -- The proof would go here
  sorry

end solve_equation_l115_115527


namespace superhero_payments_l115_115497

-- Define constants and productivity
def productivity_superman (W : ℝ) : ℝ := 0.1 * W
def productivity_flash (W : ℝ) : ℝ := 2 * (productivity_superman W)
def combined_productivity (W : ℝ) : ℝ := (productivity_superman W) + (productivity_flash W)

-- Define the total work and the work completed together
def remaining_work (W : ℝ) : ℝ := 0.9 * W

-- Define payment formula
def payment (t : ℝ) : ℝ := 90 / t

-- Define times
def time_superman_before (W : ℝ) : ℝ := 1
def time_together (W : ℝ) : ℝ := remaining_work W / combined_productivity W
def total_time_superman (W : ℝ) : ℝ := time_superman_before W + time_together W
def total_time_flash (W : ℝ) : ℝ := time_together W

-- Define payments
def payment_superman (W : ℝ) : ℝ := payment (total_time_superman W)
def payment_flash (W : ℝ) : ℝ := payment (total_time_flash W)

theorem superhero_payments (W : ℝ) (h1 : W > 0) :
  payment_superman W = 22.5 ∧ payment_flash W = 30 :=
by
  -- Placeholder for the proof
  sorry

end superhero_payments_l115_115497


namespace calculate_F_5_f_6_l115_115107

def f (a : ℤ) : ℤ := a + 3

def F (a b : ℤ) : ℤ := b^3 - 2 * a

theorem calculate_F_5_f_6 : F 5 (f 6) = 719 := by
  sorry

end calculate_F_5_f_6_l115_115107


namespace annie_money_left_l115_115651

-- Definitions according to the conditions
def initial_money : ℕ := 132
def cost_hamburger : ℕ := 4
def cost_milkshake : ℕ := 5
def num_hamburgers : ℕ := 8
def num_milkshakes : ℕ := 6

-- Calculations based on the conditions
def total_cost_hamburgers : ℕ := num_hamburgers * cost_hamburger
def total_cost_milkshakes : ℕ := num_milkshakes * cost_milkshake
def total_spent : ℕ := total_cost_hamburgers + total_cost_milkshakes
def money_left : ℕ := initial_money - total_spent

-- The proof statement
theorem annie_money_left : money_left = 70 := 
by {
  unfold money_left,
  unfold total_spent,
  unfold total_cost_hamburgers,
  unfold total_cost_milkshakes,
  rw [mul_comm num_hamburgers cost_hamburger, mul_comm num_milkshakes cost_milkshake],
  norm_num,
  unfold initial_money,
  norm_num,
  sorry
}

end annie_money_left_l115_115651


namespace cos_double_angle_l115_115715

theorem cos_double_angle : 
  (sin α - cos α) / (sin α + cos α) = 1 / 2 → cos (2 * α) = -4 / 5 := 
by 
  sorry

end cos_double_angle_l115_115715


namespace number_of_false_statements_l115_115841

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

def unit_vector (a : V) : Prop := ∥a∥ = 1

def is_parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b

theorem number_of_false_statements (a a₀ : V) (h₀ : unit_vector a₀) :
  ¬ (∀ a, a = ∥a∥ • a₀) ∧
  ¬ (∀ a, is_parallel a a₀ → a = ∥a∥ • a₀) ∧
  ¬ (∀ a, is_parallel a a₀ → ∥a∥ = 1 → a = a₀) →
  3 := sorry

end number_of_false_statements_l115_115841


namespace pink_highlighters_count_l115_115062

-- Definitions for the problem's conditions
def total_highlighters : Nat := 11
def yellow_highlighters : Nat := 2
def blue_highlighters : Nat := 5
def non_pink_highlighters : Nat := yellow_highlighters + blue_highlighters

-- Statement of the problem as a theorem
theorem pink_highlighters_count : total_highlighters - non_pink_highlighters = 4 :=
by
  sorry

end pink_highlighters_count_l115_115062


namespace alice_savings_l115_115093

variable (B : ℝ)

def savings (B : ℝ) : ℝ :=
  let first_month := 10
  let second_month := first_month + 30 + B
  let third_month := first_month + 30 + 30
  first_month + second_month + third_month

theorem alice_savings (B : ℝ) : savings B = 120 + B :=
by
  sorry

end alice_savings_l115_115093


namespace sum_of_first_n_terms_l115_115535

theorem sum_of_first_n_terms (n : ℕ) :
  (∑ k in finset.range n, (2^k)) - n = 2^(n+1) - n - 2 :=
by {
  sorry
}

end sum_of_first_n_terms_l115_115535


namespace bob_hair_growth_time_l115_115906

theorem bob_hair_growth_time (initial_length final_length growth_rate monthly_to_yearly_conversion : ℝ) 
  (initial_cut : initial_length = 6) 
  (current_length : final_length = 36) 
  (growth_per_month : growth_rate = 0.5) 
  (months_in_year : monthly_to_yearly_conversion = 12) : 
  (final_length - initial_length) / (growth_rate * monthly_to_yearly_conversion) = 5 :=
by
  sorry

end bob_hair_growth_time_l115_115906


namespace complex_magnitude_l115_115044

theorem complex_magnitude (a b : ℝ) (i : ℂ) (h : i = complex.I) (h₁ : (3 + b * i) / (1 - i) = a + b * i) : complex.abs (a + b * i) = 3 :=
by
  sorry

end complex_magnitude_l115_115044


namespace determine_x_l115_115669

-- Define the conditions
def dataset : List ℕ := [20, x, x, 70, 80, 110, 140, 180]

def isMedian (xs : List ℕ) (m : ℕ) := List.sorted xs ∧ m = (xs.get! 3 + xs.get! 4) / 2
def isMode (xs : List ℕ) (m : ℕ) := List.groupBy id xs |>.maximumBy (λ ys => ys.length) = [m]

-- The main theorem
theorem determine_x (x : ℕ) 
  (median_condition : isMedian dataset x) 
  (mode_condition : isMode dataset x) 
  (range_condition : dataset.max! - dataset.min! = 160) : x = 90 :=
by
  sorry

end determine_x_l115_115669


namespace mean_correct_median_correct_mode_correct_l115_115938

noncomputable def numbers : List ℤ := [75, 67, 75, 70, 68, 72, 65, 78, 70, 68]

theorem mean_correct : (numbers.sum / numbers.length) = 70.8 := by
  sorry

theorem median_correct : median numbers = 70 := by
  sorry

theorem mode_correct : mode numbers = {70, 75} := by
  sorry

end mean_correct_median_correct_mode_correct_l115_115938


namespace average_pages_proof_l115_115820

structure Book where
  thickness : ℝ
  width : ℝ
  binding_thickness : ℝ
  pages_per_inch : ℝ

def effective_thickness (b : Book) : ℝ :=
  b.thickness + b.binding_thickness

def total_pages (b : Book) : ℝ :=
  b.thickness * b.pages_per_inch

def average_pages_per_book (books : List Book) : ℝ :=
  let total_pages := books.foldl (λ acc b => acc + total_pages b) 0
  total_pages / books.length

theorem average_pages_proof : 
  let books := [ 
    { thickness := 1.5, width := 6, binding_thickness := 0.1, pages_per_inch := 75 },
    { thickness := 2, width := 5.5, binding_thickness := 0.15, pages_per_inch := 80 },
    { thickness := 3, width := 7, binding_thickness := 0.2, pages_per_inch := 85 },
    { thickness := 1.2, width := 5, binding_thickness := 0.05, pages_per_inch := 90 },
    { thickness := 2.3, width := 6.5, binding_thickness := 0.1, pages_per_inch := 95 },
    { thickness := 2, width := 5, binding_thickness := 0.1, pages_per_inch := 100 }
  ]
  average_pages_per_book books = 175.67 := 
by
  sorry

end average_pages_proof_l115_115820


namespace find_y_l115_115083

-- Given conditions
variables (AP CR BQ AC CB : ℝ)
variable  y : ℝ
variable  x : ℝ

-- Assume the given parallelisms and lengths
axiom parallel1 : AP = 10
axiom parallel2 : BQ = 15
axiom parallel3 : AC = x
axiom cb_length : CB = 12
axiom cr_length : CR = y

-- Proof statement
theorem find_y : CR = 12.5 :=
by
  sorry

end find_y_l115_115083


namespace sum_fraction_rel_prime_l115_115840

noncomputable def sequenceSum : ℚ :=
  let s : ℕ → ℚ := λ n, if n % 2 = 0 then (n+1) / 2^(n+1) else (n+2) / 4^(n+1)
  ∑' n, s n

theorem sum_fraction_rel_prime (a b : ℕ) (h : a.gcd b = 1)
  (hab : (a : ℚ) / b = sequenceSum) : a + b = 106 :=
sorry

end sum_fraction_rel_prime_l115_115840


namespace toys_produced_each_day_l115_115978

theorem toys_produced_each_day (total_weekly_production : ℕ) (days_per_week : ℕ) (H1 : total_weekly_production = 6500) (H2 : days_per_week = 5) : (total_weekly_production / days_per_week = 1300) :=
by {
  sorry
}

end toys_produced_each_day_l115_115978


namespace maria_high_school_students_l115_115454

variable (M D : ℕ)

theorem maria_high_school_students (h1 : M = 4 * D) (h2 : M - D = 1800) : M = 2400 :=
by
  sorry

end maria_high_school_students_l115_115454


namespace expected_value_T_l115_115885

def boys_girls_expected_value (M N : ℕ) : ℚ :=
  2 * ((M / (M + N : ℚ)) * (N / (M + N - 1 : ℚ)))

theorem expected_value_T (M N : ℕ) (hM : M = 10) (hN : N = 10) :
  boys_girls_expected_value M N = 20 / 19 :=
by 
  rw [hM, hN]
  sorry

end expected_value_T_l115_115885


namespace radius_of_circumscribing_sphere_l115_115890

noncomputable def tetrahedron_radius_circumsphere (r : ℝ) (h : r = Real.sqrt 2 - 1) : ℝ :=
  let a := 2 * (Real.sqrt 2 - 1) in
  let R := (a * Real.sqrt 6) / 4 in
  R

theorem radius_of_circumscribing_sphere (h : (Real.sqrt 2 - 1) = tetrahedron_radius_circumsphere (Real.sqrt 2 - 1)) :
  tetrahedron_radius_circumsphere (Real.sqrt 2 - 1) = Real.sqrt 6 + 1 :=
by
  unfold tetrahedron_radius_circumsphere
  rw [←Real.eq_div_iff_mul_eq]
  sorry

end radius_of_circumscribing_sphere_l115_115890


namespace sample_mean_and_variance_significant_improvement_l115_115600

variable (x y : Fin 10 → ℝ)
variable xi_vals : Array ℝ := #[545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
variable yi_vals : Array ℝ := #[536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

noncomputable def z (i : Fin 10) : ℝ := xi_vals[i] - yi_vals[i]

noncomputable def mean_z : ℝ :=
  (1 / 10) * Array.foldr (λ i acc => acc + z i) 0 (Array.indices z)

noncomputable def var_z : ℝ :=
  (1 / 10) * Array.foldr (λ i acc => acc + (z i - mean_z) ^ 2) 0 (Array.indices z)

theorem sample_mean_and_variance :
  mean_z = 11 ∧ var_z = 61 := 
  by
  sorry

theorem significant_improvement : 
  mean_z ≥ 2 * Real.sqrt (var_z / 10) :=
  by
  sorry

end sample_mean_and_variance_significant_improvement_l115_115600


namespace smallestC_l115_115117

def isValidFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) ∧
  f 1 = 1 ∧
  (∀ x y, 0 ≤ x ∧ 0 ≤ y ∧ x + y ≤ 1 → f x + f y ≤ f (x + y))

theorem smallestC (f : ℝ → ℝ) (h : isValidFunction f) : ∃ c, (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ c * x) ∧
  (∀ d, (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ d * x) → 2 ≤ d) :=
sorry

end smallestC_l115_115117


namespace monotonicity_condition_l115_115756

open Real

-- Define the given function f(x)
noncomputable def f (x a : ℝ) := x - (2 / x) + a * (2 - log x)

-- Define the derivative of f(x)
noncomputable def f' (x a : ℝ) := (1:ℝ) + (2 / x^2) - (a / x)

-- Prove that if f(x) is monotonically decreasing in the interval (1, 2), then a ≥ 3
theorem monotonicity_condition (a : ℝ) (h: ∀ x ∈ Ioo (1:ℝ) (2:ℝ), f'(x, a) ≤ 0) : 3 ≤ a :=
by
  have h1 := h 1 (by simp)
  have h2 := h 2 (by simp)
  sorry

end monotonicity_condition_l115_115756


namespace circumradius_of_sector_l115_115632

noncomputable def R_circumradius (θ : ℝ) (r : ℝ) := r / (2 * Real.sin (θ / 2))

theorem circumradius_of_sector (r : ℝ) (θ : ℝ) (hθ : θ = 120) (hr : r = 8) :
  R_circumradius θ r = (8 * Real.sqrt 3) / 3 :=
by
  rw [hθ, hr, R_circumradius]
  sorry

end circumradius_of_sector_l115_115632


namespace gcd_lcm_8951_4267_l115_115942

theorem gcd_lcm_8951_4267 :
  gcd 8951 4267 = 1 ∧ lcm 8951 4267 = 38212917 :=
by
  sorry

end gcd_lcm_8951_4267_l115_115942


namespace max_area_ABC_l115_115845

noncomputable def q (p : ℝ) : ℝ := p^2 - 7*p + 10

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem max_area_ABC : ∃ p : ℝ, 2 ≤ p ∧ p ≤ 5 ∧ 
  triangle_area (2, 0) (5, 4) (p, q p) = 0.536625 := sorry

end max_area_ABC_l115_115845


namespace compare_abc_l115_115003

noncomputable def a : Real := Real.sqrt 2
noncomputable def b : Real := Real.log 3 / Real.log π
noncomputable def c : Real := - Real.log 3 / Real.log 2

theorem compare_abc : c < b ∧ b < a := by
  sorry

end compare_abc_l115_115003


namespace sample_mean_and_variance_significant_improvement_l115_115597

variable (x y : Fin 10 → ℝ)
variable xi_vals : Array ℝ := #[545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
variable yi_vals : Array ℝ := #[536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

noncomputable def z (i : Fin 10) : ℝ := xi_vals[i] - yi_vals[i]

noncomputable def mean_z : ℝ :=
  (1 / 10) * Array.foldr (λ i acc => acc + z i) 0 (Array.indices z)

noncomputable def var_z : ℝ :=
  (1 / 10) * Array.foldr (λ i acc => acc + (z i - mean_z) ^ 2) 0 (Array.indices z)

theorem sample_mean_and_variance :
  mean_z = 11 ∧ var_z = 61 := 
  by
  sorry

theorem significant_improvement : 
  mean_z ≥ 2 * Real.sqrt (var_z / 10) :=
  by
  sorry

end sample_mean_and_variance_significant_improvement_l115_115597


namespace terminating_decimal_expansion_l115_115304

theorem terminating_decimal_expansion : (15 / 625 : ℝ) = 0.024 :=
by
  -- Lean requires a justification for non-trivial facts
  -- Provide math reasoning here if necessary
  sorry

end terminating_decimal_expansion_l115_115304


namespace sum_of_sequence_l115_115532

noncomputable def sequence_term (n : ℕ) : ℕ :=
(2^n) - 1

noncomputable def sum_sequence (n : ℕ) : ℕ :=
(nat.range n).sum (λ k, sequence_term (k + 1))

theorem sum_of_sequence (n : ℕ) : sum_sequence n = (2^(n+1)) - n - 2 :=
by sorry

end sum_of_sequence_l115_115532


namespace feeding_sequences_count_l115_115631

def num_feeding_sequences (num_pairs : ℕ) : ℕ :=
  num_pairs * num_pairs.pred * num_pairs.pred * num_pairs.pred.pred *
  num_pairs.pred.pred * num_pairs.pred.pred.pred * num_pairs.pred.pred.pred *
  1 * 1

theorem feeding_sequences_count (num_pairs : ℕ) (h : num_pairs = 5) :
  num_feeding_sequences num_pairs = 5760 := 
by
  rw [h]
  unfold num_feeding_sequences
  norm_num
  sorry

end feeding_sequences_count_l115_115631


namespace simplified_expression_l115_115477

-- Define the hypothesis for the problem
def sqrt5_times_3 := Real.sqrt (5 * 3)
def sqrt3_4_times_5_2 := Real.sqrt (3^4 * 5^2)

theorem simplified_expression : sqrt5_times_3 * sqrt3_4_times_5_2 = 45 * Real.sqrt 15 := by
  sorry

end simplified_expression_l115_115477


namespace inequality_of_sum_l115_115140

theorem inequality_of_sum 
  (a : ℕ → ℝ)
  (h : ∀ n m, 0 ≤ n → n < m → a n < a m) :
  (0 < a 1 ->
  0 < a 2 ->
  0 < a 3 ->
  0 < a 4 ->
  0 < a 5 ->
  0 < a 6 ->
  0 < a 7 ->
  0 < a 8 ->
  0 < a 9 ->
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) / (a 3 + a 6 + a 9) < 3) :=
by
  intros
  sorry

end inequality_of_sum_l115_115140


namespace number_satisfies_conditions_l115_115565

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, (n / 10^k) % 10 = d

theorem number_satisfies_conditions : 
  ∃ x : ℕ, 
    (x = 165) ∧ 
    (x % 2 = 1) ∧ 
    contains_digit x 5 ∧ 
    (x % 3 = 0) ∧ 
    (144 < x ∧ x < 169) :=
by
  use 165
  then prove each condition is true for 165
  sorry

end number_satisfies_conditions_l115_115565


namespace number_of_possible_a4_values_l115_115617

theorem number_of_possible_a4_values :
  ∀ (a : List ℕ),
    a.length = 5 →
    (a.sum / 5 = 15) →
    ((a.maximumD 0) - (a.minimumD 0) = 20) →
    (let m := (a.sort ℕ).nth_le 2 (by linarith : 2 < 5)
     in m = 10) →
    (let mode := a.mode
     in mode = 10) →
    ({x ∈ (a.sort ℕ).drop 2.slice 0 1| true}.card = 6) := sorry

end number_of_possible_a4_values_l115_115617


namespace number_of_factors_of_2550_with_more_than_3_factors_l115_115280

def num_divisors (n : ℕ) : ℕ := n.divisors.count

def has_more_than_three_factors (n : ℕ) : Bool := 
  num_divisors n > 3

def count_divisors_with_more_than_three_factors (n : ℕ) : ℕ :=
  n.divisors.count (λ k, has_more_than_three_factors k)

theorem number_of_factors_of_2550_with_more_than_3_factors :
  count_divisors_with_more_than_three_factors 2550 = 8 :=
by
  sorry

end number_of_factors_of_2550_with_more_than_3_factors_l115_115280


namespace find_x_l115_115345

variable (x : ℝ)

theorem find_x (h : 2 * x - 12 = -(x + 3)) : x = 3 := 
sorry

end find_x_l115_115345


namespace Haley_shirts_l115_115039

theorem Haley_shirts (bought returned : ℕ) (h1 : bought = 11) (h2 : returned = 6) : (bought - returned) = 5 :=
by
  rw [h1, h2]
  norm_num
  sorry

end Haley_shirts_l115_115039


namespace d_geometric_l115_115485

variable {c : ℕ → ℝ}

-- Condition: c_n > 0 for all n
axiom positive_terms (n : ℕ) : 1 ≤ n → c n > 0

-- Condition: c_n is a geometric sequence
axiom c_geometric (n m : ℕ) : 1 ≤ n → 1 ≤ m → c (n + m - 1) = c n * c m

-- Define d_n
noncomputable def d (n : ℕ) : ℝ := Real.root n (∏ i in Finset.range n, c (i + 1))

theorem d_geometric (h : ∀ n, 1 ≤ n → c n > 0) (h_geo : ∀ n m, 1 ≤ n → 1 ≤ m → c (n + m - 1) = c n * c m) :
  ∀ n m (hn : 1 ≤ n) (hm : 1 ≤ m), d (n + m - 1) = d n * d m :=
by
  sorry

end d_geometric_l115_115485


namespace circle_intersection_solution_count_l115_115222

variables {S1 S2 : Circle} {A B : Point}
variables (O1 O2 : Point) (a : ℝ)

theorem circle_intersection_solution_count :
  let O1O2 := dist O1 O2 in
  O1 ∈ S1 ∧ O2 ∈ S2 ∧ A ∈ S1 ∩ S2 ∧ B ∈ S1 ∩ S2 →
  (if O1O2 > a / 2 then 2
   else if O1O2 = a / 2 then 1
   else 0) = ?n := sorry

end circle_intersection_solution_count_l115_115222


namespace inequality_solution_l115_115883

theorem inequality_solution :
  {x : ℝ | (x - 1) * (x - 4) * (x - 5)^2 / ((x - 3) * (x^2 - 9)) > 0} = { x : ℝ | -3 < x ∧ x < 3 } :=
sorry

end inequality_solution_l115_115883


namespace translate_sin_graph_left_l115_115549

theorem translate_sin_graph_left (x : ℝ) : 
  (∃ y : ℝ, y = 3 * sin (2 * x) → y = 3 * sin (2 * (x + π / 6)) → y = 3 * sin (2 * x + π / 3)) :=
by
  sorry

end translate_sin_graph_left_l115_115549


namespace middle_card_is_6_l115_115921

noncomputable def middle_card_number (a b c : ℕ) : ℕ :=
  if a < b ∧ b < c ∧ a + b + c = 15
    ∧ (∀ x, x = a → (x ≠ 1))
    ∧ (∀ y, y = c → (y ≠ 1 ∧ y ≠ 2 ∧ y ≠ 3 ∧ y ≠ 7 ∧ y ≠ 11 ∧ y ≠ 12))
    ∧ (∀ z, z = b → (∀ w, w ≠ 4 ∧ (w = 5 → z ≠ 2) ∧ (z ≠ 5 → z = 6)))
  then b else 0

theorem middle_card_is_6 {a b c : ℕ} (habc : a < b ∧ b < c) (sum_15 : a + b + c = 15) 
(h_casey : ∀ x, x = a → (x ≠ 1)) (h_tracy : ∀ y, y = c → (y ≠ 1 ∧ y ≠ 2 ∧ y ≠ 3 ∧ y ≠ 7 ∧ y ≠ 11 ∧ y ≠ 12)) 
(h_stacy : ∀ z, z = b → (∀ w, w ≠ 4 ∧ (w = 5 → z ≠ 2) ∧ (z ≠ 5 → z = 6))) : b = 6 :=
begin
  sorry
end

end middle_card_is_6_l115_115921


namespace problem_solution_l115_115591

def z (xi yi : ℕ) : ℕ := xi - yi

def z_vals (x y : Fin 10 → ℕ) : Fin 10 → ℕ := fun i => z (x i) (y i)

def mean (z : Fin 10 → ℕ) : ℕ :=
  (∑ i in Finset.univ, z i) / 10

def variance (z : Fin 10 → ℕ) (mean_z : ℕ) : ℕ :=
  (∑ i in Finset.univ, (z i - mean_z)^2) / 10

def significant_improvement (mean_z : ℕ) (var_z : ℕ) : Prop :=
  mean_z >= 2 * Real.sqrt (var_z / 10)

-- Given data
def x : Fin 10 → ℕ
| 0 => 545
| 1 => 533
| 2 => 551
| 3 => 522
| 4 => 575
| 5 => 544
| 6 => 541
| 7 => 568
| 8 => 596
| 9 => 548

def y : Fin 10 → ℕ
| 0 => 536
| 1 => 527
| 2 => 543
| 3 => 530
| 4 => 560
| 5 => 533
| 6 => 522
| 7 => 550
| 8 => 576
| 9 => 536

/-- The final proof statements -/
theorem problem_solution :
  let z_vals := z_vals x y
  let mean_z := mean z_vals
  let var_z := variance z_vals mean_z
  mean_z = 11 ∧ var_z = 61 ∧ significant_improvement mean_z var_z := 
by
  sorry

end problem_solution_l115_115591


namespace tau_sum_cube_eq_square_sum_l115_115871

-- Define the divisor function
def tau (n : ℕ) : ℕ := 
  n.divisors.card -- Number of divisors of n

-- Define the main theorem
theorem tau_sum_cube_eq_square_sum (n : ℕ) : 
  (∑ d in n.divisors, (tau d) ^ 3) = (∑ d in n.divisors, tau d) ^ 2 :=
by 
  sorry

end tau_sum_cube_eq_square_sum_l115_115871


namespace average_goals_l115_115269

theorem average_goals (c s j : ℕ) (h1 : c = 4) (h2 : s = c / 2) (h3 : j = 2 * s - 3) :
  c + s + j = 7 :=
sorry

end average_goals_l115_115269


namespace find_a_b_l115_115035

theorem find_a_b :
  ∃ a b : ℝ, ∀ (x y : ℝ), 
  ((2*x - 1) + complex.i = y - (3 - y)*complex.i) ∧ 
  ((2*x + a*y) - (4*x - y + b)*complex.i = 9 - 8*complex.i) → 
  (a = 1 ∧ b = 2) :=
by {
  sorry
}

end find_a_b_l115_115035


namespace significant_improvement_l115_115604

noncomputable def z (x y : ℕ → ℕ) (i : ℕ) : ℕ := x i - y i

noncomputable def z_bar (x y : ℕ → ℕ) : ℝ := 
  (z x y 1 + z x y 2 + z x y 3 + z x y 4 + z x y 5 + z x y 6 + 
  z x y 7 + z x y 8 + z x y 9 + z x y 10) / 10

noncomputable def s_squared (x y : ℕ → ℕ) : ℝ := 
  let mean := z_bar x y in 
  ( (z x y 1 - mean) ^ 2 + (z x y 2 - mean) ^ 2 + (z x y 3 - mean) ^ 2 +
    (z x y 4 - mean) ^ 2 + (z x y 5 - mean) ^ 2 + (z x y 6 - mean) ^ 2 +
    (z x y 7 - mean) ^ 2 + (z x y 8 - mean) ^ 2 + (z x y 9 - mean) ^ 2 +
    (z x y 10 - mean) ^ 2) / 10

theorem significant_improvement (x y : ℕ → ℕ)
  (hx : x 1 = 545) (hx2 : x 2 = 533) (hx3 : x 3 = 551) (hx4 : x 4 = 522)
  (hx5 : x 5 = 575) (hx6 : x 6 = 544) (hx7 : x 7 = 541) (hx8 : x 8 = 568)
  (hx9 : x 9 = 596) (hx10 : x 10 = 548)
  (hy : y 1 = 536) (hy2 : y 2 = 527) (hy3 : y 3 = 543) (hy4 : y 4 = 530)
  (hy5 : y 5 = 560) (hy6 : y 6 = 533) (hy7 : y 7 = 522) (hy8 : y 8 = 550)
  (hy9 : y 9 = 576) (hy10 : y 10 = 536) :
  z_bar x y ≥ 2 * real.sqrt(s_squared x y / 10) :=
  sorry

end significant_improvement_l115_115604


namespace c_symmetry_l115_115341

-- Define the function c based on given recursions and base cases
def c : ℕ → ℕ → ℕ
| n, 0     => 1
| n, k     => if h : k ≤ n then
    if k = n then 1
    else 3^k * c n k + c n (k-1)
  else 0 -- case for n < k

-- Define the theorem to prove c(n, k) = c(n, n-k)
theorem c_symmetry (n k : ℕ) (h : n ≥ k) : c n k = c n (n - k) :=
  sorry -- proof is omitted

end c_symmetry_l115_115341
