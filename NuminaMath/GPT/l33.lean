import Mathlib

namespace calculate_shaded_area_l33_33693

noncomputable def square_shaded_area : ℝ := 
  let a := 10 -- side length of the square
  let s := a / 2 -- half side length, used for midpoints
  let total_area := a * a / 2 -- total area of a right triangle with legs a and a
  let triangle_DMA := total_area / 2 -- area of triangle DAM
  let triangle_DNG := triangle_DMA / 5 -- area of triangle DNG
  let triangle_CDM := total_area -- area of triangle CDM
  let shaded_area := triangle_CDM + triangle_DNG - triangle_DMA -- area of shaded region
  shaded_area

theorem calculate_shaded_area : square_shaded_area = 35 := 
by 
sorry

end calculate_shaded_area_l33_33693


namespace circle_equation_correct_l33_33592

def line_through_fixed_point (a : ℝ) :=
  ∀ x y : ℝ, (x + y - 1) - a * (x + 1) = 0 → x = -1 ∧ y = 2

def equation_of_circle (x y: ℝ) :=
  (x + 1)^2 + (y - 2)^2 = 5

theorem circle_equation_correct (a : ℝ) (h : line_through_fixed_point a) :
  ∀ x y : ℝ, equation_of_circle x y ↔ x^2 + y^2 + 2*x - 4*y = 0 :=
sorry

end circle_equation_correct_l33_33592


namespace petya_can_restore_numbers_if_and_only_if_odd_l33_33920

def can_restore_numbers (n : ℕ) : Prop :=
  ∀ (V : Fin n → ℕ) (S : ℕ),
    ∃ f : Fin n → ℕ, 
    (∀ i : Fin n, 
      (V i) = f i ∨ 
      (S = f i)) ↔ (n % 2 = 1)

theorem petya_can_restore_numbers_if_and_only_if_odd (n : ℕ) : can_restore_numbers n ↔ n % 2 = 1 := 
by sorry

end petya_can_restore_numbers_if_and_only_if_odd_l33_33920


namespace min_coins_cover_99_l33_33628

def coin_values : List ℕ := [1, 5, 10, 25, 50]

noncomputable def min_coins_cover (n : ℕ) : ℕ := sorry

theorem min_coins_cover_99 : min_coins_cover 99 = 9 :=
  sorry

end min_coins_cover_99_l33_33628


namespace parabola_chord_constant_l33_33849

noncomputable def calcT (x₁ x₂ c : ℝ) : ℝ :=
  let a := x₁^2 + (2*x₁^2 - c)^2
  let b := x₂^2 + (2*x₂^2 - c)^2
  1 / Real.sqrt a + 1 / Real.sqrt b

theorem parabola_chord_constant (c : ℝ) (m x₁ x₂ : ℝ) 
    (h₁ : 2*x₁^2 - m*x₁ - c = 0) 
    (h₂ : 2*x₂^2 - m*x₂ - c = 0) : 
    calcT x₁ x₂ c = -20 / (7 * c) :=
by
  sorry

end parabola_chord_constant_l33_33849


namespace Yvettes_final_bill_l33_33786

namespace IceCreamShop

def sundae_price_Alicia : Real := 7.50
def sundae_price_Brant : Real := 10.00
def sundae_price_Josh : Real := 8.50
def sundae_price_Yvette : Real := 9.00
def tip_rate : Real := 0.20

theorem Yvettes_final_bill :
  let total_cost := sundae_price_Alicia + sundae_price_Brant + sundae_price_Josh + sundae_price_Yvette
  let tip := tip_rate * total_cost
  let final_bill := total_cost + tip
  final_bill = 42.00 :=
by
  -- calculations are skipped here
  sorry

end IceCreamShop

end Yvettes_final_bill_l33_33786


namespace division_of_15_by_neg_5_l33_33689

theorem division_of_15_by_neg_5 : 15 / (-5) = -3 :=
by
  sorry

end division_of_15_by_neg_5_l33_33689


namespace cricket_initial_average_l33_33361

theorem cricket_initial_average (A : ℕ) (h1 : ∀ A, A * 20 + 137 = 21 * (A + 5)) : A = 32 := by
  -- assumption and proof placeholder
  sorry

end cricket_initial_average_l33_33361


namespace kocourkov_coins_l33_33535

theorem kocourkov_coins :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 
  (∀ n > 53, ∃ x y : ℕ, n = x * a + y * b) ∧ 
  ¬ (∃ x y : ℕ, 53 = x * a + y * b) ∧
  ((a = 2 ∧ b = 55) ∨ (a = 3 ∧ b = 28)) :=
by {
  sorry
}

end kocourkov_coins_l33_33535


namespace problem1_l33_33364

theorem problem1 : 3 * 403 + 5 * 403 + 2 * 403 + 401 = 4431 := by 
  sorry

end problem1_l33_33364


namespace find_w_l33_33037

theorem find_w (p q r u v w : ℝ)
  (h₁ : (x : ℝ) → x^3 - 6 * x^2 + 11 * x + 10 = (x - p) * (x - q) * (x - r))
  (h₂ : (x : ℝ) → x^3 + u * x^2 + v * x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p)))
  (h₃ : p + q + r = 6) :
  w = 80 := sorry

end find_w_l33_33037


namespace shopkeeper_loss_amount_l33_33602

theorem shopkeeper_loss_amount (total_stock_worth : ℝ)
                               (portion_sold_at_profit : ℝ)
                               (portion_sold_at_loss : ℝ)
                               (profit_percentage : ℝ)
                               (loss_percentage : ℝ) :
  total_stock_wworth = 14999.999999999996 →
  portion_sold_at_profit = 0.2 →
  portion_sold_at_loss = 0.8 →
  profit_percentage = 0.10 →
  loss_percentage = 0.05 →
  (total_stock_worth - ((portion_sold_at_profit * total_stock_worth * (1 + profit_percentage)) + 
                        (portion_sold_at_loss * total_stock_worth * (1 - loss_percentage)))) = 300 := 
by 
  sorry

end shopkeeper_loss_amount_l33_33602


namespace min_k_plus_p_is_19199_l33_33877

noncomputable def find_min_k_plus_p : ℕ :=
  let D := 1007
  let domain_len := 1 / D
  let min_k : ℕ := 19  -- Minimum k value for which domain length condition holds, found via problem conditions
  let p_for_k (k : ℕ) : ℕ := (D * (k^2 - 1)) / k
  let k_plus_p (k : ℕ) : ℕ := k + p_for_k k
  k_plus_p min_k

theorem min_k_plus_p_is_19199 : find_min_k_plus_p = 19199 :=
  sorry

end min_k_plus_p_is_19199_l33_33877


namespace weight_loss_percentage_l33_33871

theorem weight_loss_percentage (W : ℝ) (hW : W > 0) : 
  let new_weight := 0.89 * W
  let final_weight_with_clothes := new_weight * 1.02
  (W - final_weight_with_clothes) / W * 100 = 9.22 := by
  sorry

end weight_loss_percentage_l33_33871


namespace union_of_A_and_B_l33_33313

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4} := by
  sorry

end union_of_A_and_B_l33_33313


namespace age_difference_l33_33006

-- Define the hypothesis and statement
theorem age_difference (A B C : ℕ) 
  (h1 : A + B = B + C + 15)
  (h2 : C = A - 15) : 
  (A + B) - (B + C) = 15 :=
by
  sorry

end age_difference_l33_33006


namespace new_car_distance_in_same_time_l33_33445

-- Define the given conditions and the distances
variable (older_car_distance : ℝ := 150)
variable (new_car_speed_factor : ℝ := 1.30)  -- Since the new car is 30% faster, its speed factor is 1.30
variable (time : ℝ)

-- Define the older car's distance as a function of time and speed
def older_car_distance_covered (t : ℝ) (distance : ℝ) : ℝ := distance

-- Define the new car's distance as a function of time and speed factor
def new_car_distance_covered (t : ℝ) (distance : ℝ) (speed_factor : ℝ) : ℝ := speed_factor * distance

theorem new_car_distance_in_same_time
  (older_car_distance : ℝ)
  (new_car_speed_factor : ℝ)
  (time : ℝ)
  (h1 : older_car_distance = 150)
  (h2 : new_car_speed_factor = 1.30) :
  new_car_distance_covered time older_car_distance new_car_speed_factor = 195 := by
  sorry

end new_car_distance_in_same_time_l33_33445


namespace abs_ineq_l33_33036

open Real

noncomputable def absolute_value (x : ℝ) : ℝ := abs x

theorem abs_ineq (a b c : ℝ) (h1 : a + b ≥ 0) (h2 : b + c ≥ 0) (h3 : c + a ≥ 0) :
  a + b + c ≥ (absolute_value a + absolute_value b + absolute_value c) / 3 := by
  sorry

end abs_ineq_l33_33036


namespace cubes_with_no_colored_faces_l33_33625

theorem cubes_with_no_colored_faces (width length height : ℕ) (total_cubes cube_side : ℕ) :
  width = 6 ∧ length = 5 ∧ height = 4 ∧ total_cubes = 120 ∧ cube_side = 1 →
  (width - 2) * (length - 2) * (height - 2) = 24 :=
by
  intros h
  sorry

end cubes_with_no_colored_faces_l33_33625


namespace multiply_millions_l33_33359

theorem multiply_millions :
  (5 * 10^6) * (8 * 10^6) = 40 * 10^12 :=
by 
  sorry

end multiply_millions_l33_33359


namespace Louisa_total_travel_time_l33_33248

theorem Louisa_total_travel_time :
  ∀ (v : ℝ), v > 0 → (200 / v) + 4 = (360 / v) → (200 / v) + (360 / v) = 14 :=
by
  intros v hv eqn
  sorry

end Louisa_total_travel_time_l33_33248


namespace tan_eq_860_l33_33910

theorem tan_eq_860 (n : ℤ) (hn : -180 < n ∧ n < 180) : 
  n = -40 ↔ (Real.tan (n * Real.pi / 180) = Real.tan (860 * Real.pi / 180)) := 
sorry

end tan_eq_860_l33_33910


namespace intersection_with_single_element_union_equals_A_l33_33593

-- Definitions of the sets A and B
def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

-- Statement for question (1)
theorem intersection_with_single_element (a : ℝ) (H : A = {1, 2} ∧ A ∩ B a = {2}) : a = -1 ∨ a = -3 :=
by
  sorry

-- Statement for question (2)
theorem union_equals_A (a : ℝ) (H1 : A = {1, 2}) (H2 : A ∪ B a = A) : (a ≥ -3 ∧ a ≤ -1) :=
by
  sorry

end intersection_with_single_element_union_equals_A_l33_33593


namespace train_cross_pole_time_l33_33526

noncomputable def time_to_cross_pole : ℝ :=
  let speed_km_hr := 60
  let speed_m_s := speed_km_hr * 1000 / 3600
  let length_of_train := 50
  length_of_train / speed_m_s

theorem train_cross_pole_time :
  time_to_cross_pole = 3 := 
by
  sorry

end train_cross_pole_time_l33_33526


namespace triangle_height_l33_33700

theorem triangle_height (base height : ℝ) (area : ℝ) (h1 : base = 2) (h2 : area = 3) (area_formula : area = (base * height) / 2) : height = 3 :=
by
  sorry

end triangle_height_l33_33700


namespace remainder_n_plus_2023_l33_33895

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 7 = 3) : (n + 2023) % 7 = 3 :=
by sorry

end remainder_n_plus_2023_l33_33895


namespace exponentiation_example_l33_33169

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l33_33169


namespace candy_mixture_cost_l33_33912

/-- 
A club mixes 15 pounds of candy worth $8.00 per pound with 30 pounds of candy worth $5.00 per pound.
We need to find the cost per pound of the mixture.
-/
theorem candy_mixture_cost :
    (15 * 8 + 30 * 5) / (15 + 30) = 6 := 
by
  sorry

end candy_mixture_cost_l33_33912


namespace convex_quadrilateral_probability_l33_33516

noncomputable def probability_convex_quadrilateral (n : ℕ) : ℚ :=
  if n = 6 then (Nat.choose 6 4 : ℚ) / (Nat.choose 15 4 : ℚ) else 0

theorem convex_quadrilateral_probability :
  probability_convex_quadrilateral 6 = 1 / 91 :=
by
  sorry

end convex_quadrilateral_probability_l33_33516


namespace part1_part2_l33_33857

-- Definitions and conditions
def prop_p (a : ℝ) : Prop := 
  let Δ := -4 * a^2 + 4 * a + 24 
  Δ ≥ 0

def neg_prop_p (a : ℝ) : Prop := ¬ prop_p a

def prop_q (m a : ℝ) : Prop := 
  (m - 1 ≤ a ∧ a ≤ m + 3)

-- Part 1 theorem statement
theorem part1 (a : ℝ) : neg_prop_p a → (a < -2 ∨ a > 3) :=
by sorry

-- Part 2 theorem statement
theorem part2 (m : ℝ) : 
  (∀ a : ℝ, prop_q m a → prop_p a) ∧ (∃ a : ℝ, prop_p a ∧ ¬ prop_q m a) → (-1 ≤ m ∧ m < 0) :=
by sorry

end part1_part2_l33_33857


namespace range_of_B_l33_33503

theorem range_of_B (a b c : ℝ) (h : a + c = 2 * b) :
  ∃ B : ℝ, 0 < B ∧ B ≤ π / 3 ∧
  ∃ A C : ℝ, ∃ ha : a = c, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π :=
sorry

end range_of_B_l33_33503


namespace faye_age_l33_33916

variables (C D E F G : ℕ)
variables (h1 : D = E - 2)
variables (h2 : E = C + 6)
variables (h3 : F = C + 4)
variables (h4 : G = C - 5)
variables (h5 : D = 16)

theorem faye_age : F = 16 :=
by
  -- Proof will be placed here
  sorry

end faye_age_l33_33916


namespace cookie_problem_l33_33353

theorem cookie_problem : 
  ∃ (B : ℕ), B = 130 ∧ B - 80 = 50 ∧ B/2 + 20 = 85 :=
by
  sorry

end cookie_problem_l33_33353


namespace sequence_from_520_to_523_is_0_to_3_l33_33644

theorem sequence_from_520_to_523_is_0_to_3 
  (repeating_pattern : ℕ → ℕ)
  (h_periodic : ∀ n, repeating_pattern (n + 5) = repeating_pattern n) :
  ((repeating_pattern 520, repeating_pattern 521, repeating_pattern 522, repeating_pattern 523) = (repeating_pattern 0, repeating_pattern 1, repeating_pattern 2, repeating_pattern 3)) :=
by {
  sorry
}

end sequence_from_520_to_523_is_0_to_3_l33_33644


namespace total_oranges_picked_l33_33482

theorem total_oranges_picked :
  let Mary_oranges := 14
  let Jason_oranges := 41
  let Amanda_oranges := 56
  Mary_oranges + Jason_oranges + Amanda_oranges = 111 := by
    sorry

end total_oranges_picked_l33_33482


namespace shapes_values_correct_l33_33648

-- Define variable types and conditions
variables (x y z w : ℕ)
variables (sum1 sum2 sum3 sum4 T : ℕ)

-- Define the conditions for the problem as given in (c)
axiom row_sum1 : x + y + z = sum1
axiom row_sum2 : y + z + w = sum2
axiom row_sum3 : z + w + x = sum3
axiom row_sum4 : w + x + y = sum4
axiom col_sum  : x + y + z + w = T

-- Define the variables with specific values as determined in the solution
def triangle := 2
def square := 0
def a_tilde := 6
def O_value := 1

-- Prove that the assigned values satisfy the conditions
theorem shapes_values_correct :
  x = triangle ∧ y = square ∧ z = a_tilde ∧ w = O_value :=
by { sorry }

end shapes_values_correct_l33_33648


namespace dwarf_heights_l33_33396

-- Define the heights of the dwarfs.
variables (F J M : ℕ)

-- Given conditions
def condition1 : Prop := J + F = M
def condition2 : Prop := M + F = J + 34
def condition3 : Prop := M + J = F + 72

-- Proof statement
theorem dwarf_heights
  (h1 : condition1 F J M)
  (h2 : condition2 F J M)
  (h3 : condition3 F J M) :
  F = 17 ∧ J = 36 ∧ M = 53 :=
by
  sorry

end dwarf_heights_l33_33396


namespace max_annual_profit_l33_33172

noncomputable def R (x : ℝ) : ℝ :=
  if x < 40 then 10 * x^2 + 300 * x
  else (901 * x^2 - 9450 * x + 10000) / x

noncomputable def W (x : ℝ) : ℝ :=
  if x < 40 then -10 * x^2 + 600 * x - 260
  else -x + 9190 - 10000 / x

theorem max_annual_profit : ∃ x : ℝ, W 100 = 8990 :=
by {
  use 100,
  sorry
}

end max_annual_profit_l33_33172


namespace axis_of_symmetry_l33_33716

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (4 - x)) :
  ∀ y : ℝ, (∃ x₁ x₂ : ℝ, y = f x₁ ∧ y = f x₂ ∧ (x₁ + x₂) / 2 = 2) :=
by
  sorry

end axis_of_symmetry_l33_33716


namespace most_frequent_data_is_mode_l33_33380

-- Define the options
inductive Options where
  | Mean
  | Mode
  | Median
  | Frequency

-- Define the problem statement
def mostFrequentDataTerm (freqMost : String) : Options :=
  if freqMost == "Mode" then 
    Options.Mode
  else if freqMost == "Mean" then 
    Options.Mean
  else if freqMost == "Median" then 
    Options.Median
  else 
    Options.Frequency

-- Statement of the problem as a theorem
theorem most_frequent_data_is_mode (freqMost : String) :
  mostFrequentDataTerm freqMost = Options.Mode :=
by
  sorry

end most_frequent_data_is_mode_l33_33380


namespace find_y_l33_33238

variable (a b y : ℝ)
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(3 * b) = a^b * y^b)

theorem find_y : y = 27 * a^2 :=
  by sorry

end find_y_l33_33238


namespace mul_101_eq_10201_l33_33409

theorem mul_101_eq_10201 : 101 * 101 = 10201 := by
  sorry

end mul_101_eq_10201_l33_33409


namespace find_incorrect_statement_l33_33188

variable (q n x y : ℚ)

theorem find_incorrect_statement :
  (∀ q, q < -1 → q < 1/q) ∧
  (∀ n, n ≥ 0 → -n ≥ n) ∧
  (∀ x, x < 0 → x^3 < x) ∧
  (∀ y, y < 0 → y^2 > y) →
  (∃ x, x < 0 ∧ ¬ (x^3 < x)) :=
by
  sorry

end find_incorrect_statement_l33_33188


namespace fraction_spent_l33_33787

theorem fraction_spent (borrowed_from_brother borrowed_from_father borrowed_from_mother gift_from_granny savings remaining amount_spent : ℕ)
  (h_borrowed_from_brother : borrowed_from_brother = 20)
  (h_borrowed_from_father : borrowed_from_father = 40)
  (h_borrowed_from_mother : borrowed_from_mother = 30)
  (h_gift_from_granny : gift_from_granny = 70)
  (h_savings : savings = 100)
  (h_remaining : remaining = 65)
  (h_amount_spent : amount_spent = borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings - remaining) :
  (amount_spent : ℚ) / (borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings) = 3 / 4 :=
by
  sorry

end fraction_spent_l33_33787


namespace inequality_for_five_real_numbers_l33_33208

open Real

theorem inequality_for_five_real_numbers
  (a1 a2 a3 a4 a5 : ℝ)
  (h1 : 1 < a1)
  (h2 : 1 < a2)
  (h3 : 1 < a3)
  (h4 : 1 < a4)
  (h5 : 1 < a5) :
  16 * (a1 * a2 * a3 * a4 * a5 + 1) ≥ (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) * (1 + a5) := 
sorry

end inequality_for_five_real_numbers_l33_33208


namespace zeros_distance_l33_33246

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3*x^2 + a

theorem zeros_distance (a x1 x2 : ℝ) 
  (hx1 : f a x1 = 0) (hx2 : f a x2 = 0) (h_order: x1 < x2) : 
  x2 - x1 = 3 := 
sorry

end zeros_distance_l33_33246


namespace min_unattainable_score_l33_33130

theorem min_unattainable_score : ∀ (score : ℕ), (¬ ∃ (a b c : ℕ), 
  (a = 1 ∨ a = 3 ∨ a = 8 ∨ a = 12 ∨ a = 0) ∧ 
  (b = 1 ∨ b = 3 ∨ b = 8 ∨ b = 12 ∨ b = 0) ∧ 
  (c = 1 ∨ c = 3 ∨ c = 8 ∨ c = 12 ∨ c = 0) ∧ 
  score = a + b + c) ↔ score = 22 := 
by
  sorry

end min_unattainable_score_l33_33130


namespace find_sample_size_l33_33633

theorem find_sample_size :
  ∀ (n : ℕ), 
    (∃ x : ℝ,
      2 * x + 3 * x + 4 * x + 6 * x + 4 * x + x = 1 ∧
      2 * n * x + 3 * n * x + 4 * n * x = 27) →
    n = 60 :=
by
  intro n
  rintro ⟨x, h1, h2⟩
  sorry

end find_sample_size_l33_33633


namespace range_of_a_l33_33319

theorem range_of_a 
    (a : ℝ) 
    (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f x = Real.exp (|x - a|)) 
    (increasing_on_interval : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y) :
    a ≤ 1 :=
sorry

end range_of_a_l33_33319


namespace part_I_part_II_l33_33441

noncomputable section

def f (x a : ℝ) : ℝ := |x + a| + |x - (1 / a)|

theorem part_I (x : ℝ) : f x 1 ≥ 5 ↔ x ≤ -5/2 ∨ x ≥ 5/2 := by
  sorry

theorem part_II (a m : ℝ) (h : ∀ x : ℝ, f x a ≥ |m - 1|) : -1 ≤ m ∧ m ≤ 3 := by
  sorry

end part_I_part_II_l33_33441


namespace smallest_sum_B_d_l33_33454

theorem smallest_sum_B_d :
  ∃ B d : ℕ, (B < 5) ∧ (d > 6) ∧ (125 * B + 25 * B + B = 4 * d + 4) ∧ (B + d = 77) :=
by
  sorry

end smallest_sum_B_d_l33_33454


namespace cos_double_angle_of_parallel_vectors_l33_33365

theorem cos_double_angle_of_parallel_vectors (α : ℝ) 
  (h_parallel : (1 / 3, Real.tan α) = (Real.cos α, 1)) : 
  Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cos_double_angle_of_parallel_vectors_l33_33365


namespace interest_rate_l33_33993

theorem interest_rate (total_investment : ℝ) (investment1 : ℝ) (investment2 : ℝ) (rate2 : ℝ) (interest1 : ℝ → ℝ) (interest2 : ℝ → ℝ) :
  (total_investment = 5400) →
  (investment1 = 3000) →
  (investment2 = total_investment - investment1) →
  (rate2 = 0.10) →
  (interest1 investment1 = investment1 * (interest1 1)) →
  (interest2 investment2 = investment2 * rate2) →
  interest1 investment1 = interest2 investment2 →
  interest1 1 = 0.08 :=
by
  intros
  sorry

end interest_rate_l33_33993


namespace q_evaluation_l33_33018

def q (x y : ℤ) : ℤ :=
if x ≥ 0 ∧ y ≤ 0 then x - y
else if x < 0 ∧ y > 0 then x + 3 * y
else 4 * x - 2 * y

theorem q_evaluation : q (q 2 (-3)) (q (-4) 1) = 6 :=
by
  sorry

end q_evaluation_l33_33018


namespace chosen_number_l33_33277

theorem chosen_number (x : ℝ) (h : 2 * x - 138 = 102) : x = 120 := by
  sorry

end chosen_number_l33_33277


namespace parabola_vertex_l33_33180

theorem parabola_vertex (x y : ℝ) :
  (x^2 - 4 * x + 3 * y + 8 = 0) → (x, y) = (2, -4 / 3) :=
by
  sorry

end parabola_vertex_l33_33180


namespace find_s_l33_33553

theorem find_s (a b r1 r2 : ℝ) (h1 : r1 + r2 = -a) (h2 : r1 * r2 = b) :
    let new_root1 := (r1 + r2) * (r1 + r2)
    let new_root2 := (r1 * r2) * (r1 + r2)
    let s := b * a - a * a
    s = ab - a^2 :=
  by
    -- the proof goes here
    sorry

end find_s_l33_33553


namespace horner_value_at_neg4_l33_33325

noncomputable def f (x : ℝ) : ℝ := 10 + 25 * x - 8 * x^2 + x^4 + 6 * x^5 + 2 * x^6

def horner_rewrite (x : ℝ) : ℝ := (((((2 * x + 6) * x + 1) * x + 0) * x - 8) * x + 25) * x + 10

theorem horner_value_at_neg4 : horner_rewrite (-4) = -36 :=
by sorry

end horner_value_at_neg4_l33_33325


namespace chapter_page_difference_l33_33596

/-- The first chapter of a book has 37 pages -/
def first_chapter_pages : Nat := 37

/-- The second chapter of a book has 80 pages -/
def second_chapter_pages : Nat := 80

/-- Prove the difference in the number of pages between the second and the first chapter is 43 -/
theorem chapter_page_difference : (second_chapter_pages - first_chapter_pages) = 43 := by
  sorry

end chapter_page_difference_l33_33596


namespace farmer_potatoes_initial_l33_33956

theorem farmer_potatoes_initial (P : ℕ) (h1 : 175 + P - 172 = 80) : P = 77 :=
by {
  sorry
}

end farmer_potatoes_initial_l33_33956


namespace original_triangle_area_l33_33563

theorem original_triangle_area (area_of_new_triangle : ℝ) (side_length_ratio : ℝ) (quadrupled : side_length_ratio = 4) (new_area : area_of_new_triangle = 128) : 
  (area_of_new_triangle / side_length_ratio ^ 2) = 8 := by
  sorry

end original_triangle_area_l33_33563


namespace find_c_l33_33309

-- Definitions
def is_root (x c : ℝ) : Prop := x^2 - 3*x + c = 0

-- Main statement
theorem find_c (c : ℝ) (h : is_root 1 c) : c = 2 :=
sorry

end find_c_l33_33309


namespace find_triplets_l33_33824

theorem find_triplets (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a ∣ b + c + 1) (h5 : b ∣ c + a + 1) (h6 : c ∣ a + b + 1) :
  (a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 2, 2) ∨ (a, b, c) = (3, 4, 4) ∨ 
  (a, b, c) = (1, 1, 3) ∨ (a, b, c) = (2, 2, 5) :=
sorry

end find_triplets_l33_33824


namespace find_prime_power_solutions_l33_33219

theorem find_prime_power_solutions (p n m : ℕ) (hp : Nat.Prime p) (hn : n > 0) (hm : m > 0) 
  (h : p^n + 144 = m^2) :
  (p = 2 ∧ n = 9 ∧ m = 36) ∨ (p = 3 ∧ n = 4 ∧ m = 27) :=
by sorry

end find_prime_power_solutions_l33_33219


namespace avery_egg_cartons_filled_l33_33985

-- Definitions (conditions identified in step a)
def total_chickens : ℕ := 20
def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12

-- Theorem statement (equivalent to the problem statement)
theorem avery_egg_cartons_filled : (total_chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  -- Proof omitted; sorry used to denote unfinished proof
  sorry

end avery_egg_cartons_filled_l33_33985


namespace grandparents_to_parents_ratio_l33_33315

-- Definitions corresponding to the conditions
def wallet_cost : ℕ := 100
def betty_half_money : ℕ := wallet_cost / 2
def parents_contribution : ℕ := 15
def betty_needs_more : ℕ := 5
def grandparents_contribution : ℕ := 95 - (betty_half_money + parents_contribution)

-- The mathematical statement for the proof
theorem grandparents_to_parents_ratio :
  grandparents_contribution / parents_contribution = 2 := by
  sorry

end grandparents_to_parents_ratio_l33_33315


namespace find_c_for_circle_radius_5_l33_33915

theorem find_c_for_circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 4 * x + y^2 + 8 * y + c = 0 
    → x^2 + 4 * x + y^2 + 8 * y = 5^2 - 25) 
  → c = -5 :=
by
  sorry

end find_c_for_circle_radius_5_l33_33915


namespace sin_of_angle_l33_33183

theorem sin_of_angle (θ : ℝ) (h : Real.cos (θ + Real.pi) = -1/3) :
  Real.sin (2*θ + Real.pi/2) = -7/9 :=
by
  sorry

end sin_of_angle_l33_33183


namespace factor_difference_of_squares_l33_33306

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l33_33306


namespace rainfall_on_tuesday_l33_33103

noncomputable def R_Tuesday (R_Sunday : ℝ) (D1 : ℝ) : ℝ := 
  R_Sunday + D1

noncomputable def R_Thursday (R_Tuesday : ℝ) (D2 : ℝ) : ℝ :=
  R_Tuesday + D2

noncomputable def total_rainfall (R_Sunday R_Tuesday R_Thursday : ℝ) : ℝ :=
  R_Sunday + R_Tuesday + R_Thursday

theorem rainfall_on_tuesday : R_Tuesday 2 3.75 = 5.75 := 
by 
  sorry -- Proof goes here

end rainfall_on_tuesday_l33_33103


namespace elizabeth_revenue_per_investment_l33_33940

theorem elizabeth_revenue_per_investment :
  ∀ (revenue_per_investment_banks revenue_difference total_investments_banks total_investments_elizabeth : ℕ),
    revenue_per_investment_banks = 500 →
    total_investments_banks = 8 →
    total_investments_elizabeth = 5 →
    revenue_difference = 500 →
    ((revenue_per_investment_banks * total_investments_banks) + revenue_difference) / total_investments_elizabeth = 900 :=
by
  intros revenue_per_investment_banks revenue_difference total_investments_banks total_investments_elizabeth
  intros h_banks_revenue h_banks_investments h_elizabeth_investments h_revenue_difference
  sorry

end elizabeth_revenue_per_investment_l33_33940


namespace decrease_percent_revenue_l33_33398

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) : 
  let original_revenue := T * C
  let new_tax := 0.80 * T
  let new_consumption := 1.10 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 12 := by
  sorry

end decrease_percent_revenue_l33_33398


namespace multiply_exponents_l33_33831

variable (a : ℝ)

theorem multiply_exponents :
  a * a^2 * (-a)^3 = -a^6 := 
sorry

end multiply_exponents_l33_33831


namespace product_of_solutions_l33_33287

theorem product_of_solutions :
  (∀ x : ℝ, |3 * x - 2| + 5 = 23 → x = 20 / 3 ∨ x = -16 / 3) →
  (20 / 3 * -16 / 3 = -320 / 9) :=
by
  intros h
  have h₁ : 20 / 3 * -16 / 3 = -320 / 9 := sorry
  exact h₁

end product_of_solutions_l33_33287


namespace remainder_when_200_divided_by_k_l33_33298

theorem remainder_when_200_divided_by_k (k : ℕ) (hk_pos : 0 < k)
  (h₁ : 125 % (k^3) = 5) : 200 % k = 0 :=
sorry

end remainder_when_200_divided_by_k_l33_33298


namespace ordered_pair_exists_l33_33834

theorem ordered_pair_exists (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) :=
by
  sorry

end ordered_pair_exists_l33_33834


namespace max_points_on_circle_l33_33509

noncomputable def circleMaxPoints (P C : ℝ × ℝ) (r1 r2 d : ℝ) : ℕ :=
  if d = r1 + r2 ∨ d = abs (r1 - r2) then 1 else 
  if d < r1 + r2 ∧ d > abs (r1 - r2) then 2 else 0

theorem max_points_on_circle (P : ℝ × ℝ) (C : ℝ × ℝ) :
  let rC := 5
  let distPC := 9
  let rP := 4
  circleMaxPoints P C rC rP distPC = 1 :=
by sorry

end max_points_on_circle_l33_33509


namespace lcm_8_13_14_is_728_l33_33039

-- Define the numbers and their factorizations
def num1 := 8
def fact1 := 2 ^ 3

def num2 := 13  -- 13 is prime

def num3 := 14
def fact3 := 2 * 7

-- Define the function to calculate the LCM of three integers
def lcm (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- State the theorem to prove that the LCM of 8, 13, and 14 is 728
theorem lcm_8_13_14_is_728 : lcm num1 num2 num3 = 728 :=
by
  -- Prove the equality, skipping proof details with sorry
  sorry

end lcm_8_13_14_is_728_l33_33039


namespace integer_solutions_x2_minus_y2_equals_12_l33_33548

theorem integer_solutions_x2_minus_y2_equals_12 : 
  ∃! (s : Finset (ℤ × ℤ)), (∀ (xy : ℤ × ℤ), xy ∈ s → xy.1^2 - xy.2^2 = 12) ∧ s.card = 4 :=
sorry

end integer_solutions_x2_minus_y2_equals_12_l33_33548


namespace book_original_price_l33_33186

noncomputable def original_price : ℝ := 420 / 1.40

theorem book_original_price (new_price : ℝ) (percentage_increase : ℝ) : 
  new_price = 420 → percentage_increase = 0.40 → original_price = 300 :=
by
  intros h1 h2
  exact sorry

end book_original_price_l33_33186


namespace specific_value_eq_l33_33585

def specific_value (x : ℕ) : ℕ := 25 * x

theorem specific_value_eq : specific_value 27 = 675 := by
  sorry

end specific_value_eq_l33_33585


namespace factorize_difference_of_squares_l33_33437

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4*x^2 = (3 - 2*x) * (3 + 2*x) :=
by
  sorry

end factorize_difference_of_squares_l33_33437


namespace swimming_pool_time_l33_33405

theorem swimming_pool_time
  (A B C : ℝ)
  (h1 : A + B = 1 / 3)
  (h2 : A + C = 1 / 6)
  (h3 : B + C = 1 / 4.5) :
  1 / (A + B + C) = 2.25 :=
by
  sorry

end swimming_pool_time_l33_33405


namespace overlapping_area_fraction_l33_33842

variable (Y X : ℝ)
variable (hY : 0 < Y)
variable (hX : X = (1 / 8) * (2 * Y - X))

theorem overlapping_area_fraction : X = (2 / 9) * Y :=
by
  -- We define the conditions and relationships stated in the problem
  -- Prove the theorem accordingly
  sorry

end overlapping_area_fraction_l33_33842


namespace find_other_number_l33_33093

noncomputable def HCF : ℕ := 14
noncomputable def LCM : ℕ := 396
noncomputable def one_number : ℕ := 154
noncomputable def product_of_numbers : ℕ := HCF * LCM

theorem find_other_number (other_number : ℕ) :
  HCF * LCM = one_number * other_number → other_number = 36 :=
by
  sorry

end find_other_number_l33_33093


namespace final_number_is_correct_l33_33511

-- Define the problem conditions as Lean definitions/statements
def original_number : ℤ := 4
def doubled_number (x : ℤ) : ℤ := 2 * x
def resultant_number (x : ℤ) : ℤ := doubled_number x + 9
def final_number (x : ℤ) : ℤ := 3 * resultant_number x

-- Formulate the theorem using the conditions
theorem final_number_is_correct :
  final_number original_number = 51 :=
by
  sorry

end final_number_is_correct_l33_33511


namespace problem_1_problem_2_l33_33753

open Set Real

-- Definition of the sets A, B, and the complement of B in the real numbers
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Proof problem (1): Prove that A ∩ (complement of B) = [1, 2]
theorem problem_1 : (A ∩ (compl B)) = {x | 1 ≤ x ∧ x ≤ 2} := sorry

-- Proof problem (2): Prove that the set of values for the real number a such that C(a) ∩ A = C(a)
-- is (-∞, 3]
theorem problem_2 : { a : ℝ | C a ⊆ A } = { a : ℝ | a ≤ 3 } := sorry

end problem_1_problem_2_l33_33753


namespace sphere_radius_squared_l33_33927

theorem sphere_radius_squared (R x y z : ℝ)
  (h1 : 2 * Real.sqrt (R^2 - x^2 - y^2) = 5)
  (h2 : 2 * Real.sqrt (R^2 - x^2 - z^2) = 6)
  (h3 : 2 * Real.sqrt (R^2 - y^2 - z^2) = 7) :
  R^2 = 15 :=
sorry

end sphere_radius_squared_l33_33927


namespace total_weight_of_sections_l33_33768

theorem total_weight_of_sections :
  let doll_length := 5
  let doll_weight := 29 / 8
  let tree_length := 4
  let tree_weight := 2.8
  let section_length := 2
  let doll_weight_per_meter := doll_weight / doll_length
  let tree_weight_per_meter := tree_weight / tree_length
  let doll_section_weight := doll_weight_per_meter * section_length
  let tree_section_weight := tree_weight_per_meter * section_length
  doll_section_weight + tree_section_weight = 57 / 20 :=
sorry

end total_weight_of_sections_l33_33768


namespace length_more_than_breadth_by_200_l33_33771

-- Definitions and conditions
def rectangular_floor_length := 23
def painting_cost := 529
def painting_rate := 3
def floor_area := painting_cost / painting_rate
def floor_breadth := floor_area / rectangular_floor_length

-- Prove that the length is more than the breadth by 200%
theorem length_more_than_breadth_by_200 : 
  rectangular_floor_length = floor_breadth * (1 + 200 / 100) :=
sorry

end length_more_than_breadth_by_200_l33_33771


namespace Isaabel_math_pages_l33_33671

theorem Isaabel_math_pages (x : ℕ) (total_problems : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  (reading_pages * problems_per_page = 20) ∧ (total_problems = 30) →
  x * problems_per_page + 20 = total_problems →
  x = 2 := by
  sorry

end Isaabel_math_pages_l33_33671


namespace bounded_g_of_f_l33_33478

theorem bounded_g_of_f
  (f g : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∃ x, f x ≠ 0)
  (h3 : ∀ x, |f x| ≤ 1) :
  ∀ y, |g y| ≤ 1 := 
sorry

end bounded_g_of_f_l33_33478


namespace no_integer_solution_l33_33497

theorem no_integer_solution (a b : ℤ) : ¬(a^2 + b^2 = 10^100 + 3) :=
sorry

end no_integer_solution_l33_33497


namespace solve_A_solve_area_l33_33501

noncomputable def angle_A (A : ℝ) : Prop :=
  2 * (Real.cos (A / 2))^2 + Real.cos A = 0

noncomputable def area_triangle (a b c : ℝ) (A : ℝ) : Prop :=
  a = 2 * Real.sqrt 3 → b + c = 4 → A = 2 * Real.pi / 3 → 
  (1/2) * b * c * Real.sin A = Real.sqrt 3

theorem solve_A (A : ℝ) : angle_A A → A = 2 * Real.pi / 3 :=
sorry

theorem solve_area (a b c A S : ℝ) : 
  a = 2 * Real.sqrt 3 →
  b + c = 4 →
  A = 2 * Real.pi / 3 →
  area_triangle a b c A →
  S = Real.sqrt 3 :=
sorry

end solve_A_solve_area_l33_33501


namespace friends_total_candies_l33_33542

noncomputable def total_candies (T S J C V B : ℕ) : ℕ :=
  T + S + J + C + V + B

theorem friends_total_candies :
  let T := 22
  let S := 16
  let J := T / 2
  let C := 2 * S
  let V := J + S
  let B := (T + C) / 2 + 9
  total_candies T S J C V B = 144 := by
  sorry

end friends_total_candies_l33_33542


namespace harry_total_cost_l33_33688

-- Define the price of each type of seed packet
def pumpkin_price : ℝ := 2.50
def tomato_price : ℝ := 1.50
def chili_pepper_price : ℝ := 0.90
def zucchini_price : ℝ := 1.20
def eggplant_price : ℝ := 1.80

-- Define the quantities Harry wants to buy
def pumpkin_qty : ℕ := 4
def tomato_qty : ℕ := 6
def chili_pepper_qty : ℕ := 7
def zucchini_qty : ℕ := 3
def eggplant_qty : ℕ := 5

-- Calculate the total cost
def total_cost : ℝ :=
  pumpkin_qty * pumpkin_price +
  tomato_qty * tomato_price +
  chili_pepper_qty * chili_pepper_price +
  zucchini_qty * zucchini_price +
  eggplant_qty * eggplant_price

-- The proof problem
theorem harry_total_cost : total_cost = 38.90 := by
  sorry

end harry_total_cost_l33_33688


namespace inverse_f_of_7_l33_33198

def f (x : ℝ) : ℝ := 2 * x^2 + 3

theorem inverse_f_of_7:
  ∀ y : ℝ, f (7) = y ↔ y = 101 :=
by
  sorry

end inverse_f_of_7_l33_33198


namespace trig_identity_l33_33462

open Real

theorem trig_identity : sin (20 * (π / 180)) * cos (10 * (π / 180)) - cos (200 * (π / 180)) * sin (10 * (π / 180)) = 1 / 2 := 
by
  sorry

end trig_identity_l33_33462


namespace nature_of_roots_indeterminate_l33_33523

variable (a b c : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem nature_of_roots_indeterminate (h : b^2 - 4 * a * c = 0) : 
  ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) = 0) ∨ ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) < 0) ∨ ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) > 0) :=
sorry

end nature_of_roots_indeterminate_l33_33523


namespace max_volume_of_hollow_cube_l33_33729

/-- 
We have 1000 solid cubes with edge lengths of 1 unit each. 
The small cubes can be glued together but not cut. 
The cube to be created is hollow with a wall thickness of 1 unit.
Prove that the maximum external volume of the cube we can create is 2197 cubic units.
--/

theorem max_volume_of_hollow_cube :
  ∃ x : ℕ, 6 * x^2 - 12 * x + 8 ≤ 1000 ∧ x^3 = 2197 :=
sorry

end max_volume_of_hollow_cube_l33_33729


namespace base_6_addition_l33_33308

-- Definitions of base conversion and addition
def base_6_to_nat (n : ℕ) : ℕ :=
  n.div 100 * 36 + n.div 10 % 10 * 6 + n % 10

def nat_to_base_6 (n : ℕ) : ℕ :=
  let a := n.div 216
  let b := (n % 216).div 36
  let c := ((n % 216) % 36).div 6
  let d := n % 6
  a * 1000 + b * 100 + c * 10 + d

-- Conversion from base 6 to base 10 for the given numbers
def nat_256 := base_6_to_nat 256
def nat_130 := base_6_to_nat 130

-- The final theorem to prove
theorem base_6_addition : nat_to_base_6 (nat_256 + nat_130) = 1042 :=
by
  -- Proof omitted since it is not required
  sorry

end base_6_addition_l33_33308


namespace garage_sale_total_l33_33837

theorem garage_sale_total (treadmill chest_of_drawers television total_sales : ℝ)
  (h1 : treadmill = 100) 
  (h2 : chest_of_drawers = treadmill / 2) 
  (h3 : television = treadmill * 3) 
  (partial_sales : ℝ) 
  (h4 : partial_sales = treadmill + chest_of_drawers + television) 
  (h5 : partial_sales = total_sales * 0.75) : 
  total_sales = 600 := 
by
  sorry

end garage_sale_total_l33_33837


namespace range_of_function_l33_33595

theorem range_of_function : 
  ∀ y : ℝ, ∃ x : ℝ, y = 4^x + 2^x - 3 ↔ y > -3 :=
by
  sorry

end range_of_function_l33_33595


namespace twelve_position_in_circle_l33_33109

theorem twelve_position_in_circle (a : ℕ → ℕ) (h_cyclic : ∀ i, a (i + 20) = a i)
  (h_sum_six : ∀ i, a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) + a (i + 5) = 24)
  (h_first : a 1 = 1) :
  a 12 = 7 :=
sorry

end twelve_position_in_circle_l33_33109


namespace least_five_digit_congruent_to_6_mod_19_l33_33136

theorem least_five_digit_congruent_to_6_mod_19 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 19 = 6 ∧ n = 10011 :=
by
  sorry

end least_five_digit_congruent_to_6_mod_19_l33_33136


namespace fractional_expression_value_l33_33146

theorem fractional_expression_value (x y z : ℝ) (hz : z ≠ 0) 
  (h1 : 2 * x - 3 * y - z = 0)
  (h2 : x + 3 * y - 14 * z = 0) :
  (x^2 + 3 * x * y) / (y^2 + z^2) = 7 := 
by sorry

end fractional_expression_value_l33_33146


namespace part1_part2_l33_33870

-- Step 1: Define necessary probabilities
def P_A1 : ℚ := 5 / 6
def P_A2 : ℚ := 2 / 3
def P_B1 : ℚ := 3 / 5
def P_B2 : ℚ := 3 / 4

-- Step 2: Winning event probabilities for both participants
def P_A_wins := P_A1 * P_A2
def P_B_wins := P_B1 * P_B2

-- Step 3: Problem statement: Comparing probabilities
theorem part1 (P_A_wins P_A_wins : ℚ) : P_A_wins > P_B_wins := 
  by sorry

-- Step 4: Complement probabilities for not winning the competition
def P_not_A_wins := 1 - P_A_wins
def P_not_B_wins := 1 - P_B_wins

-- Step 5: Probability at least one wins
def P_at_least_one_wins := 1 - (P_not_A_wins * P_not_B_wins)

-- Step 6: Problem statement: At least one wins
theorem part2 : P_at_least_one_wins = 34 / 45 := 
  by sorry

end part1_part2_l33_33870


namespace running_to_weightlifting_ratio_l33_33868

-- Definitions for given conditions in the problem
def total_practice_time : ℕ := 120 -- 120 minutes
def shooting_time : ℕ := total_practice_time / 2
def weightlifting_time : ℕ := 20
def running_time : ℕ := shooting_time - weightlifting_time

-- The goal is to prove that the ratio of running time to weightlifting time is 2:1
theorem running_to_weightlifting_ratio : running_time / weightlifting_time = 2 :=
by
  /- use the given problem conditions directly -/
  exact sorry

end running_to_weightlifting_ratio_l33_33868


namespace emily_mean_seventh_score_l33_33343

theorem emily_mean_seventh_score :
  let a1 := 85
  let a2 := 88
  let a3 := 90
  let a4 := 94
  let a5 := 96
  let a6 := 92
  (a1 + a2 + a3 + a4 + a5 + a6 + a7) / 7 = 91 → a7 = 92 :=
by
  intros
  sorry

end emily_mean_seventh_score_l33_33343


namespace total_earnings_is_correct_l33_33055

def lloyd_normal_hours : ℝ := 7.5
def lloyd_rate : ℝ := 4.5
def lloyd_overtime_rate : ℝ := 2.0
def lloyd_hours_worked : ℝ := 10.5

def casey_normal_hours : ℝ := 8
def casey_rate : ℝ := 5
def casey_overtime_rate : ℝ := 1.5
def casey_hours_worked : ℝ := 9.5

def lloyd_earnings : ℝ := (lloyd_normal_hours * lloyd_rate) + ((lloyd_hours_worked - lloyd_normal_hours) * lloyd_rate * lloyd_overtime_rate)

def casey_earnings : ℝ := (casey_normal_hours * casey_rate) + ((casey_hours_worked - casey_normal_hours) * casey_rate * casey_overtime_rate)

def total_earnings : ℝ := lloyd_earnings + casey_earnings

theorem total_earnings_is_correct : total_earnings = 112 := by
  sorry

end total_earnings_is_correct_l33_33055


namespace simplify_expression_l33_33541

theorem simplify_expression (a : ℚ) (h : a^2 - a - 7/2 = 0) : 
  a^2 - (a - (2 * a) / (a + 1)) / ((a^2 - 2 * a + 1) / (a^2 - 1)) = 7 / 2 := 
by
  sorry

end simplify_expression_l33_33541


namespace solve_pair_l33_33377

theorem solve_pair (x y : ℕ) (h₁ : x = 12785 ∧ y = 12768 ∨ x = 11888 ∧ y = 11893 ∨ x = 12784 ∧ y = 12770 ∨ x = 1947 ∧ y = 1945) :
  1983 = 1982 * 11888 - 1981 * 11893 :=
by {
  sorry
}

end solve_pair_l33_33377


namespace sandy_has_four_times_more_marbles_l33_33414

-- Definitions based on conditions
def jessica_red_marbles : ℕ := 3 * 12
def sandy_red_marbles : ℕ := 144

-- The theorem to prove
theorem sandy_has_four_times_more_marbles : sandy_red_marbles = 4 * jessica_red_marbles :=
by
  sorry

end sandy_has_four_times_more_marbles_l33_33414


namespace time_spent_on_spelling_l33_33375

-- Define the given conditions
def total_time : Nat := 60
def math_time : Nat := 15
def reading_time : Nat := 27

-- Define the question as a Lean theorem statement
theorem time_spent_on_spelling : total_time - math_time - reading_time = 18 := sorry

end time_spent_on_spelling_l33_33375


namespace min_value_y_l33_33605

theorem min_value_y (x : ℝ) (h : x > 1) : 
  ∃ y_min : ℝ, (∀ y, y = (1 / (x - 1) + x) → y ≥ y_min) ∧ y_min = 3 :=
sorry

end min_value_y_l33_33605


namespace arithmetic_sequence_ratio_l33_33706

/-- 
  Given the ratio of the sum of the first n terms of two arithmetic sequences,
  prove the ratio of the 11th terms of these sequences.
-/
theorem arithmetic_sequence_ratio (S T : ℕ → ℚ) 
  (h : ∀ n, S n / T n = (7 * n + 1 : ℚ) / (4 * n + 2)) : 
  S 21 / T 21 = 74 / 43 :=
sorry

end arithmetic_sequence_ratio_l33_33706


namespace percentage_big_bottles_sold_l33_33559

-- Definitions of conditions
def total_small_bottles : ℕ := 6000
def total_big_bottles : ℕ := 14000
def small_bottles_sold_percentage : ℕ := 20
def total_bottles_remaining : ℕ := 15580

-- Theorem statement
theorem percentage_big_bottles_sold : 
  let small_bottles_sold := (small_bottles_sold_percentage * total_small_bottles) / 100
  let small_bottles_remaining := total_small_bottles - small_bottles_sold
  let big_bottles_remaining := total_bottles_remaining - small_bottles_remaining
  let big_bottles_sold := total_big_bottles - big_bottles_remaining
  (100 * big_bottles_sold) / total_big_bottles = 23 := 
by
  sorry

end percentage_big_bottles_sold_l33_33559


namespace race_distance_l33_33207

theorem race_distance 
  (D : ℝ) 
  (A_time : ℝ) (B_time : ℝ) 
  (A_beats_B_by : ℝ) 
  (A_time_eq : A_time = 36)
  (B_time_eq : B_time = 45)
  (A_beats_B_by_eq : A_beats_B_by = 24) :
  ((D / A_time) * B_time = D + A_beats_B_by) -> D = 24 := 
by 
  sorry

end race_distance_l33_33207


namespace prove_ratio_chickens_pigs_horses_sheep_l33_33050

noncomputable def ratio_chickens_pigs_horses_sheep (c p h s : ℕ) : Prop :=
  (∃ k : ℕ, c = 26*k ∧ p = 5*k) ∧
  (∃ l : ℕ, s = 25*l ∧ h = 9*l) ∧
  (∃ m : ℕ, p = 10*m ∧ h = 3*m) ∧
  c = 156 ∧ p = 30 ∧ h = 9 ∧ s = 25

theorem prove_ratio_chickens_pigs_horses_sheep (c p h s : ℕ) :
  ratio_chickens_pigs_horses_sheep c p h s :=
sorry

end prove_ratio_chickens_pigs_horses_sheep_l33_33050


namespace total_sum_of_money_l33_33310

theorem total_sum_of_money (x : ℝ) (A B C D E : ℝ) (hA : A = x) (hB : B = 0.75 * x) 
  (hC : C = 0.60 * x) (hD : D = 0.50 * x) (hE1 : E = 0.40 * x) (hE2 : E = 84) : 
  A + B + C + D + E = 682.50 := 
by sorry

end total_sum_of_money_l33_33310


namespace find_x_y_l33_33767

theorem find_x_y (x y : ℝ) (h1 : (10 + 25 + x + y) / 4 = 20) (h2 : x * y = 156) :
  (x = 12 ∧ y = 33) ∨ (x = 33 ∧ y = 12) :=
by
  sorry

end find_x_y_l33_33767


namespace diagonal_of_rectangular_prism_l33_33711

noncomputable def diagonal_length (a b c : ℕ) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2)

theorem diagonal_of_rectangular_prism :
  diagonal_length 12 18 15 = 3 * Real.sqrt 77 :=
by
  sorry

end diagonal_of_rectangular_prism_l33_33711


namespace relationship_between_abcd_l33_33476

theorem relationship_between_abcd (a b c d : ℝ) (h : d ≠ 0) :
  (∀ x : ℝ, (a * x + c) / (b * x + d) = (a + c * x) / (b + d * x)) ↔ a / b = c / d :=
by
  sorry

end relationship_between_abcd_l33_33476


namespace expression_bounds_l33_33002

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  2 + Real.sqrt 2 ≤ 
  (Real.sqrt (a^2 + (1 - b)^2 + 1) + 
   Real.sqrt (b^2 + (1 - c)^2 + 1) + 
   Real.sqrt (c^2 + (1 - d)^2 + 1) + 
   Real.sqrt (d^2 + (1 - a)^2 + 1)) ∧ 
  (Real.sqrt (a^2 + (1 - b)^2 + 1) + 
   Real.sqrt (b^2 + (1 - c)^2 + 1) + 
   Real.sqrt (c^2 + (1 - d)^2 + 1) + 
   Real.sqrt (d^2 + (1 - a)^2 + 1)) ≤ 4 * Real.sqrt 2 := 
sorry

end expression_bounds_l33_33002


namespace total_weight_30_l33_33570

-- Definitions of initial weights and ratio conditions
variables (a b : ℕ)
def initial_weights (h1 : a = 4 * b) : Prop := True

-- Definitions of transferred weights
def transferred_weights (a' b' : ℕ) (h2 : a' = a - 10) (h3 : b' = b + 10) : Prop := True

-- Definition of the new ratio condition
def new_ratio (a' b' : ℕ) (h4 : 8 * a' = 7 * b') : Prop := True

-- The final proof statement
theorem total_weight_30 (a b a' b' : ℕ)
    (h1 : a = 4 * b) 
    (h2 : a' = a - 10) 
    (h3 : b' = b + 10)
    (h4 : 8 * a' = 7 * b') : a + b = 30 := 
    sorry

end total_weight_30_l33_33570


namespace people_distribution_l33_33537

theorem people_distribution (x : ℕ) (h1 : x > 5):
  100 / (x - 5) = 150 / x :=
sorry

end people_distribution_l33_33537


namespace train_pass_bridge_time_l33_33646

/-- A train is 460 meters long and runs at a speed of 45 km/h. The bridge is 140 meters long. 
Prove that the time it takes for the train to pass the bridge is 48 seconds. -/
theorem train_pass_bridge_time (train_length : ℝ) (bridge_length : ℝ) (speed_kmh : ℝ) 
  (h_train_length : train_length = 460) 
  (h_bridge_length : bridge_length = 140)
  (h_speed_kmh : speed_kmh = 45)
  : (train_length + bridge_length) / (speed_kmh * 1000 / 3600) = 48 := 
by
  sorry

end train_pass_bridge_time_l33_33646


namespace square_area_25_l33_33813

theorem square_area_25 (side_length : ℝ) (h_side_length : side_length = 5) : side_length * side_length = 25 := 
by
  rw [h_side_length]
  norm_num
  done

end square_area_25_l33_33813


namespace divide_cakes_l33_33521

/-- Statement: Eleven cakes can be divided equally among six girls without cutting any cake into 
exactly six equal parts such that each girl receives 1 + 1/2 + 1/4 + 1/12 cakes -/
theorem divide_cakes (cakes girls : ℕ) (h_cakes : cakes = 11) (h_girls : girls = 6) :
  ∃ (parts : ℕ → ℝ), (∀ i, parts i = 1 + 1 / 2 + 1 / 4 + 1 / 12) ∧ (cakes = girls * (1 + 1 / 2 + 1 / 4 + 1 / 12)) :=
by
  sorry

end divide_cakes_l33_33521


namespace bubble_bath_amount_l33_33456

noncomputable def total_bubble_bath_needed 
  (couple_rooms : ℕ) (single_rooms : ℕ) (people_per_couple_room : ℕ) (people_per_single_room : ℕ) (ml_per_bath : ℕ) : ℕ :=
  couple_rooms * people_per_couple_room * ml_per_bath + single_rooms * people_per_single_room * ml_per_bath

theorem bubble_bath_amount :
  total_bubble_bath_needed 13 14 2 1 10 = 400 := by 
  sorry

end bubble_bath_amount_l33_33456


namespace min_value_am_gm_l33_33803

theorem min_value_am_gm (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 := 
sorry

end min_value_am_gm_l33_33803


namespace koala_fiber_intake_l33_33053

theorem koala_fiber_intake (x : ℝ) (h : 0.40 * x = 16) : x = 40 :=
by
  sorry

end koala_fiber_intake_l33_33053


namespace minimum_uninteresting_vertices_correct_maximum_unusual_vertices_correct_l33_33349

-- Definition for the minimum number of uninteresting vertices
def minimum_uninteresting_vertices (n : ℕ) (h : n > 3) : ℕ := 2

-- Theorem for the minimum number of uninteresting vertices
theorem minimum_uninteresting_vertices_correct (n : ℕ) (h : n > 3) :
  minimum_uninteresting_vertices n h = 2 := 
sorry

-- Definition for the maximum number of unusual vertices
def maximum_unusual_vertices (n : ℕ) (h : n > 3) : ℕ := 3

-- Theorem for the maximum number of unusual vertices
theorem maximum_unusual_vertices_correct (n : ℕ) (h : n > 3) :
  maximum_unusual_vertices n h = 3 :=
sorry

end minimum_uninteresting_vertices_correct_maximum_unusual_vertices_correct_l33_33349


namespace time_ratio_krishan_nandan_l33_33060

theorem time_ratio_krishan_nandan 
  (N T k : ℝ) 
  (H1 : N * T = 6000) 
  (H2 : N * T + 6 * N * k * T = 78000) 
  : k = 2 := 
by 
sorry

end time_ratio_krishan_nandan_l33_33060


namespace distance_is_correct_l33_33812

noncomputable def distance_from_center_to_plane
  (O : Point)
  (radius : ℝ)
  (vertices : Point × Point × Point)
  (side_lengths : (ℝ × ℝ × ℝ)) :
  ℝ :=
  8.772

theorem distance_is_correct
  (O : Point)
  (radius : ℝ)
  (A B C : Point)
  (h_radius : radius = 10)
  (h_sides : side_lengths = (17, 17, 16))
  (vertices := (A, B, C)) :
  distance_from_center_to_plane O radius vertices side_lengths = 8.772 := by
  sorry

end distance_is_correct_l33_33812


namespace bill_steps_l33_33458

theorem bill_steps (step_length : ℝ) (total_distance : ℝ) (n_steps : ℕ) 
  (h_step_length : step_length = 1 / 2) 
  (h_total_distance : total_distance = 12) 
  (h_n_steps : n_steps = total_distance / step_length) : 
  n_steps = 24 :=
by sorry

end bill_steps_l33_33458


namespace smallest_student_count_l33_33289

theorem smallest_student_count (x y z w : ℕ) 
  (ratio12to10 : x / y = 3 / 2) 
  (ratio12to11 : x / z = 7 / 4) 
  (ratio12to9 : x / w = 5 / 3) : 
  x + y + z + w = 298 :=
by
  sorry

end smallest_student_count_l33_33289


namespace integer_solutions_to_inequality_l33_33612

noncomputable def count_integer_solutions (n : ℕ) : ℕ :=
1 + 2 * n^2 + 2 * n

theorem integer_solutions_to_inequality (n : ℕ) :
  ∃ (count : ℕ), count = count_integer_solutions n ∧ 
  ∀ (x y : ℤ), |x| + |y| ≤ n → (∃ (k : ℕ), k = count) :=
by
  sorry

end integer_solutions_to_inequality_l33_33612


namespace choose_athlete_B_l33_33363

variable (SA2 : ℝ) (SB2 : ℝ)
variable (num_shots : ℕ) (avg_rings : ℝ)

-- Conditions
def athlete_A_variance := SA2 = 3.5
def athlete_B_variance := SB2 = 2.8
def same_number_of_shots := true -- Implicit condition, doesn't need proof
def same_average_rings := true -- Implicit condition, doesn't need proof

-- Question: prove Athlete B should be chosen
theorem choose_athlete_B 
  (hA_var : athlete_A_variance SA2)
  (hB_var : athlete_B_variance SB2)
  (same_shots : same_number_of_shots)
  (same_avg : same_average_rings) :
  "B" = "B" :=
by 
  sorry

end choose_athlete_B_l33_33363


namespace abraham_initial_budget_l33_33654

-- Definitions based on conditions
def shower_gel_price := 4
def shower_gel_quantity := 4
def toothpaste_price := 3
def laundry_detergent_price := 11
def remaining_budget := 30

-- Calculations based on the conditions
def spent_on_shower_gels := shower_gel_quantity * shower_gel_price
def spent_on_toothpaste := toothpaste_price
def spent_on_laundry_detergent := laundry_detergent_price
def total_spent := spent_on_shower_gels + spent_on_toothpaste + spent_on_laundry_detergent

-- The theorem to prove
theorem abraham_initial_budget :
  (total_spent + remaining_budget) = 60 :=
by
  sorry

end abraham_initial_budget_l33_33654


namespace smallest_n_divisible_by_5_l33_33723

def is_not_divisible_by_5 (x : ℤ) : Prop :=
  ¬ (x % 5 = 0)

def avg_is_integer (xs : List ℤ) : Prop :=
  (List.sum xs) % 5 = 0

theorem smallest_n_divisible_by_5 (n : ℕ) (h1 : n > 1980)
  (h2 : ∀ x ∈ List.range n, is_not_divisible_by_5 x)
  : n = 1985 :=
by
  -- The proof would go here
  sorry

end smallest_n_divisible_by_5_l33_33723


namespace geometric_sequence_fifth_term_l33_33674

theorem geometric_sequence_fifth_term (r : ℕ) (h₁ : 5 * r^3 = 405) : 5 * r^4 = 405 :=
sorry

end geometric_sequence_fifth_term_l33_33674


namespace odd_number_difference_of_squares_not_unique_l33_33123

theorem odd_number_difference_of_squares_not_unique :
  ∀ n : ℤ, Odd n → ∃ X Y X' Y' : ℤ, (n = X^2 - Y^2) ∧ (n = X'^2 - Y'^2) ∧ (X ≠ X' ∨ Y ≠ Y') :=
sorry

end odd_number_difference_of_squares_not_unique_l33_33123


namespace cricket_team_initial_games_l33_33407

theorem cricket_team_initial_games
  (initial_games : ℕ)
  (won_30_percent_initially : ℕ)
  (additional_wins : ℕ)
  (final_win_rate : ℚ) :
  won_30_percent_initially = initial_games * 30 / 100 →
  final_win_rate = (won_30_percent_initially + additional_wins) / (initial_games + additional_wins) →
  additional_wins = 55 →
  final_win_rate = 52 / 100 →
  initial_games = 120 := by sorry

end cricket_team_initial_games_l33_33407


namespace number_of_real_solutions_l33_33440

noncomputable def system_of_equations_solutions_count (x : ℝ) : Prop :=
  3 * x^2 - 45 * (⌊x⌋:ℝ) + 60 = 0 ∧ 2 * x - 3 * (⌊x⌋:ℝ) + 1 = 0

theorem number_of_real_solutions : ∃ (x₁ x₂ x₃ : ℝ), system_of_equations_solutions_count x₁ ∧ system_of_equations_solutions_count x₂ ∧ system_of_equations_solutions_count x₃ ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ :=
sorry

end number_of_real_solutions_l33_33440


namespace determine_n_for_11111_base_n_is_perfect_square_l33_33928

theorem determine_n_for_11111_base_n_is_perfect_square:
  ∃ m : ℤ, m^2 = 3^4 + 3^3 + 3^2 + 3 + 1 :=
by
  sorry

end determine_n_for_11111_base_n_is_perfect_square_l33_33928


namespace complex_number_solution_l33_33829

open Complex

theorem complex_number_solution (z : ℂ) (h : (1 + I) * z = 2 * I) : z = 1 + I :=
sorry

end complex_number_solution_l33_33829


namespace a_minus_b_eq_one_l33_33201

variable (a b : ℕ)

theorem a_minus_b_eq_one
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : Real.sqrt 18 = a * Real.sqrt 2) 
  (h4 : Real.sqrt 8 = 2 * Real.sqrt b) : 
  a - b = 1 := 
sorry

end a_minus_b_eq_one_l33_33201


namespace find_number_of_cows_l33_33766

-- Definitions from the conditions
def number_of_ducks : ℕ := sorry
def number_of_cows : ℕ := sorry

-- Define the number of legs and heads
def legs := 2 * number_of_ducks + 4 * number_of_cows
def heads := number_of_ducks + number_of_cows

-- Given condition from the problem
def condition := legs = 2 * heads + 32

-- Assert the number of cows
theorem find_number_of_cows (h : condition) : number_of_cows = 16 :=
sorry

end find_number_of_cows_l33_33766


namespace value_of_g_3x_minus_5_l33_33099

variable (R : Type) [Field R]
variable (g : R → R)
variable (x y : R)

-- Given condition: g(x) = -3 for all real numbers x
axiom g_is_constant : ∀ x : R, g x = -3

-- Prove that g(3x - 5) = -3
theorem value_of_g_3x_minus_5 : g (3 * x - 5) = -3 :=
by
  sorry

end value_of_g_3x_minus_5_l33_33099


namespace factor_x12_minus_4096_l33_33502

theorem factor_x12_minus_4096 (x : ℝ) : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) :=
by
  sorry

end factor_x12_minus_4096_l33_33502


namespace smallest_n_l33_33781

/-- The smallest value of n > 20 that satisfies
    n ≡ 4 [MOD 6]
    n ≡ 3 [MOD 7]
    n ≡ 5 [MOD 8] is 220. -/
theorem smallest_n (n : ℕ) : 
  (n > 20) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (n % 8 = 5) ↔ (n = 220) :=
by 
  sorry

end smallest_n_l33_33781


namespace three_at_five_l33_33202

def op_at (a b : ℤ) : ℤ := 3 * a - 3 * b

theorem three_at_five : op_at 3 5 = -6 :=
by
  sorry

end three_at_five_l33_33202


namespace find_c_l33_33116

def sum_of_digits (n : ℕ) : ℕ := (n.digits 10).sum

theorem find_c :
  let a := sum_of_digits (4568 ^ 777)
  let b := sum_of_digits a
  let c := sum_of_digits b
  c = 5 :=
by
  let a := sum_of_digits (4568 ^ 777)
  let b := sum_of_digits a
  let c := sum_of_digits b
  sorry

end find_c_l33_33116


namespace volume_of_pond_rect_prism_l33_33752

-- Define the problem as a proposition
theorem volume_of_pond_rect_prism :
  let l := 28
  let w := 10
  let h := 5
  V = l * w * h →
  V = 1400 :=
by
  intros l w h h1
  -- Here, the theorem states the equivalence of the volume given the defined length, width, and height being equal to 1400 cubic meters.
  have : V = 28 * 10 * 5 := by sorry
  exact this

end volume_of_pond_rect_prism_l33_33752


namespace travel_time_on_third_day_l33_33069

-- Definitions based on conditions
def speed_first_day : ℕ := 5
def time_first_day : ℕ := 7
def distance_first_day : ℕ := speed_first_day * time_first_day

def speed_second_day_part1 : ℕ := 6
def time_second_day_part1 : ℕ := 6
def distance_second_day_part1 : ℕ := speed_second_day_part1 * time_second_day_part1

def speed_second_day_part2 : ℕ := 3
def time_second_day_part2 : ℕ := 3
def distance_second_day_part2 : ℕ := speed_second_day_part2 * time_second_day_part2

def distance_second_day : ℕ := distance_second_day_part1 + distance_second_day_part2
def total_distance_first_two_days : ℕ := distance_first_day + distance_second_day

def total_distance : ℕ := 115
def distance_third_day : ℕ := total_distance - total_distance_first_two_days

def speed_third_day : ℕ := 7
def time_third_day : ℕ := distance_third_day / speed_third_day

-- The statement to be proven
theorem travel_time_on_third_day : time_third_day = 5 := by
  sorry

end travel_time_on_third_day_l33_33069


namespace min_value_expression_l33_33872

open Real

theorem min_value_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 27) :
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2 ≥ 60 := 
  sorry

end min_value_expression_l33_33872


namespace min_value_of_function_l33_33402

noncomputable def f (a x : ℝ) : ℝ := (a^x - a)^2 + (a^(-x) - a)^2

theorem min_value_of_function (a : ℝ) (h : a > 0) : ∃ x : ℝ, f a x = 2 :=
by
  sorry

end min_value_of_function_l33_33402


namespace no_positive_integer_satisfies_conditions_l33_33431

theorem no_positive_integer_satisfies_conditions :
  ¬∃ (n : ℕ), n > 1 ∧ (∃ (p1 : ℕ), Prime p1 ∧ n = p1^2) ∧ (∃ (p2 : ℕ), Prime p2 ∧ 3 * n + 16 = p2^2) :=
by
  sorry

end no_positive_integer_satisfies_conditions_l33_33431


namespace apollo_total_cost_l33_33774

def hephaestus_first_half_months : ℕ := 6
def hephaestus_first_half_rate : ℕ := 3
def hephaestus_second_half_rate : ℕ := hephaestus_first_half_rate * 2

def athena_rate : ℕ := 5
def athena_months : ℕ := 12

def ares_first_period_months : ℕ := 9
def ares_first_period_rate : ℕ := 4
def ares_second_period_months : ℕ := 3
def ares_second_period_rate : ℕ := 6

def total_cost := hephaestus_first_half_months * hephaestus_first_half_rate
               + hephaestus_first_half_months * hephaestus_second_half_rate
               + athena_months * athena_rate
               + ares_first_period_months * ares_first_period_rate
               + ares_second_period_months * ares_second_period_rate

theorem apollo_total_cost : total_cost = 168 := by
  -- placeholder for the proof
  sorry

end apollo_total_cost_l33_33774


namespace gcd_1728_1764_l33_33528

theorem gcd_1728_1764 : Int.gcd 1728 1764 = 36 := by
  sorry

end gcd_1728_1764_l33_33528


namespace log_six_two_l33_33579

noncomputable def log_six (x : ℝ) : ℝ := Real.log x / Real.log 6

theorem log_six_two (a : ℝ) (h : log_six 3 = a) : log_six 2 = 1 - a :=
by
  sorry

end log_six_two_l33_33579


namespace larry_wins_game_l33_33113

-- Defining probabilities for Larry and Julius
def larry_throw_prob : ℚ := 2 / 3
def julius_throw_prob : ℚ := 1 / 3

-- Calculating individual probabilities based on the description
def p1 : ℚ := larry_throw_prob
def p3 : ℚ := (julius_throw_prob ^ 2) * larry_throw_prob
def p5 : ℚ := (julius_throw_prob ^ 4) * larry_throw_prob

-- Aggregating the probability that Larry wins the game
def larry_wins_prob : ℚ := p1 + p3 + p5

-- The proof statement
theorem larry_wins_game : larry_wins_prob = 170 / 243 := by
  sorry

end larry_wins_game_l33_33113


namespace gcd_of_three_l33_33876

theorem gcd_of_three (a b c : ℕ) (h₁ : a = 9242) (h₂ : b = 13863) (h₃ : c = 34657) :
  Nat.gcd (Nat.gcd a b) c = 1 :=
by
  sorry

end gcd_of_three_l33_33876


namespace bicycles_sold_saturday_l33_33474

variable (S : ℕ)

theorem bicycles_sold_saturday :
  let net_increase_friday := 15 - 10
  let net_increase_saturday := 8 - S
  let net_increase_sunday := 11 - 9
  (net_increase_friday + net_increase_saturday + net_increase_sunday = 3) → 
  S = 12 :=
by
  intros h
  sorry

end bicycles_sold_saturday_l33_33474


namespace isosceles_triangle_vertex_angle_l33_33860

noncomputable def vertex_angle_of_isosceles (a b : ℝ) : ℝ :=
  if a = b then 40 else 100

theorem isosceles_triangle_vertex_angle (a : ℝ) (interior_angle : ℝ)
  (h_isosceles : a = 40 ∨ a = interior_angle ∧ interior_angle = 40 ∨ interior_angle = 100) :
  vertex_angle_of_isosceles a interior_angle = 40 ∨ vertex_angle_of_isosceles a interior_angle = 100 := 
by
  sorry

end isosceles_triangle_vertex_angle_l33_33860


namespace shaded_area_l33_33709

theorem shaded_area (r : ℝ) (sector_area : ℝ) (h1 : r = 4) (h2 : sector_area = 2 * Real.pi) : 
  sector_area - (1 / 2 * (r * Real.sqrt 2) * (r * Real.sqrt 2)) = 2 * Real.pi - 4 :=
by 
  -- Lean proof follows
  sorry

end shaded_area_l33_33709


namespace rectangle_width_decrease_l33_33255

theorem rectangle_width_decrease (L W : ℝ) (h1 : 0 < L) (h2 : 0 < W) 
(h3 : ∀ W' : ℝ, 0 < W' → (1.3 * L * W' = L * W) → W' = (100 - 23.077) / 100 * W) : 
  ∃ W' : ℝ, 0 < W' ∧ (1.3 * L * W' = L * W) ∧ ((W - W') / W = 23.077 / 100) :=
by
  sorry

end rectangle_width_decrease_l33_33255


namespace ab_le_one_l33_33827

theorem ab_le_one {a b : ℝ} (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : a + b = 2) : ab ≤ 1 :=
by
  sorry

end ab_le_one_l33_33827


namespace max_possible_value_of_gcd_l33_33330

theorem max_possible_value_of_gcd (n : ℕ) : gcd ((8^n - 1) / 7) ((8^(n+1) - 1) / 7) = 1 := by
  sorry

end max_possible_value_of_gcd_l33_33330


namespace geometric_common_ratio_l33_33757

theorem geometric_common_ratio (a₁ q : ℝ) (h₁ : q ≠ 1) (h₂ : (a₁ * (1 - q ^ 3)) / (1 - q) / ((a₁ * (1 - q ^ 2)) / (1 - q)) = 3 / 2) :
  q = 1 ∨ q = -1 / 2 :=
by
  -- Proof omitted
  sorry

end geometric_common_ratio_l33_33757


namespace sum_of_terms_in_sequence_is_215_l33_33791

theorem sum_of_terms_in_sequence_is_215 (a d : ℕ) (h1: Nat.Prime a) (h2: Nat.Prime d)
  (hAP : a + 50 = a + 50)
  (hGP : (a + d) * (a + 50) = (a + 2 * d) ^ 2) :
  (a + (a + d) + (a + 2 * d) + (a + 50)) = 215 := sorry

end sum_of_terms_in_sequence_is_215_l33_33791


namespace sets_relationship_l33_33765

def M : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 3 * k - 2}
def P : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 3 * n + 1}
def S : Set ℤ := {x : ℤ | ∃ m : ℤ, x = 6 * m + 1}

theorem sets_relationship : S ⊆ P ∧ M = P := by
  sorry

end sets_relationship_l33_33765


namespace melanie_trout_catch_l33_33382

def trout_caught_sara : ℕ := 5
def trout_caught_melanie (sara_trout : ℕ) : ℕ := 2 * sara_trout

theorem melanie_trout_catch :
  trout_caught_melanie trout_caught_sara = 10 :=
by
  sorry

end melanie_trout_catch_l33_33382


namespace art_class_students_not_in_science_l33_33619

theorem art_class_students_not_in_science (n S A S_inter_A_only_A : ℕ) 
  (h_n : n = 120) 
  (h_S : S = 85) 
  (h_A : A = 65) 
  (h_union: n = S + A - S_inter_A_only_A) : 
  S_inter_A_only_A = 30 → 
  A - S_inter_A_only_A = 35 :=
by
  intros h
  rw [h]
  sorry

end art_class_students_not_in_science_l33_33619


namespace quadratic_common_root_inverse_other_roots_l33_33997

variables (p q r s : ℝ)
variables (hq : q ≠ -1) (hs : s ≠ -1)

theorem quadratic_common_root_inverse_other_roots :
  (∃ a b : ℝ, (a ≠ b) ∧ (a^2 + p * a + q = 0) ∧ (a * b = 1) ∧ (b^2 + r * b + s = 0)) ↔ 
  (p * r = (q + 1) * (s + 1) ∧ p * (q + 1) * s = r * (s + 1) * q) :=
sorry

end quadratic_common_root_inverse_other_roots_l33_33997


namespace bracelets_count_l33_33368

-- Define the conditions
def stones_total : Nat := 36
def stones_per_bracelet : Nat := 12

-- Define the theorem statement
theorem bracelets_count : stones_total / stones_per_bracelet = 3 := by
  sorry

end bracelets_count_l33_33368


namespace nickel_ate_2_chocolates_l33_33302

def nickels_chocolates (r n : Nat) : Prop :=
r = n + 7

theorem nickel_ate_2_chocolates (r : Nat) (h : r = 9) (h1 : nickels_chocolates r 2) : 2 = 2 :=
by
  sorry

end nickel_ate_2_chocolates_l33_33302


namespace Zixuan_amount_l33_33401

noncomputable def amounts (X Y Z : ℕ) : Prop := 
  (X + Y + Z = 50) ∧
  (X = 3 * (Y + Z) / 2) ∧
  (Y = Z + 4)

theorem Zixuan_amount : ∃ Z : ℕ, ∃ X Y : ℕ, amounts X Y Z ∧ Z = 8 :=
by
  sorry

end Zixuan_amount_l33_33401


namespace anika_sequence_correct_l33_33743

noncomputable def anika_sequence : ℚ :=
  let s0 := 1458
  let s1 := s0 * 3
  let s2 := s1 / 2
  let s3 := s2 * 3
  let s4 := s3 / 2
  let s5 := s4 * 3
  s5

theorem anika_sequence_correct :
  anika_sequence = (3^9 : ℚ) / 2 := by
  sorry

end anika_sequence_correct_l33_33743


namespace f_of_f_inv_e_eq_inv_e_l33_33305

noncomputable def f : ℝ → ℝ := λ x =>
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_of_f_inv_e_eq_inv_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end f_of_f_inv_e_eq_inv_e_l33_33305


namespace cos_7_theta_l33_33242

variable (θ : Real)

namespace CosineProof

theorem cos_7_theta (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -5669 / 16384 := by
  sorry

end CosineProof

end cos_7_theta_l33_33242


namespace find_y_given_area_l33_33822

-- Define the problem parameters and conditions
namespace RectangleArea

variables {y : ℝ} (y_pos : y > 0)

-- Define the vertices, they can be expressed but are not required in the statement
def vertices := [(-2, y), (8, y), (-2, 3), (8, 3)]

-- Define the area condition
def area_condition := 10 * (y - 3) = 90

-- Lean statement proving y = 12 given the conditions
theorem find_y_given_area (y_pos : y > 0) (h : 10 * (y - 3) = 90) : y = 12 :=
by
  sorry

end RectangleArea

end find_y_given_area_l33_33822


namespace parallel_lines_perpendicular_lines_l33_33033

theorem parallel_lines (t s k : ℝ) :
  (∀ t, ∃ s, (1 - 2 * t = s) ∧ (2 + k * t = 1 - 2 * s)) →
  k = 4 :=
by
  sorry

theorem perpendicular_lines (t s k : ℝ) :
  (∀ t, ∃ s, (1 - 2 * t = s) ∧ (2 + k * t = 1 - 2 * s)) →
  k = -1 :=
by
  sorry

end parallel_lines_perpendicular_lines_l33_33033


namespace unique_real_value_for_equal_roots_l33_33480

-- Definitions of conditions
def quadratic_eq (p : ℝ) : Prop := 
  ∀ x : ℝ, x^2 - (p + 1) * x + p = 0

-- Statement of the problem
theorem unique_real_value_for_equal_roots :
  ∃! p : ℝ, ∀ x y : ℝ, (x^2 - (p+1)*x + p = 0) ∧ (y^2 - (p+1)*y + p = 0) → x = y := 
sorry

end unique_real_value_for_equal_roots_l33_33480


namespace bob_grade_is_35_l33_33231

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

theorem bob_grade_is_35 : bob_grade = 35 :=
by
  -- Proof will go here
  sorry

end bob_grade_is_35_l33_33231


namespace flower_pots_count_l33_33324

noncomputable def total_flower_pots (x : ℕ) : ℕ :=
  if h : ((x / 2) + (x / 4) + (x / 7) ≤ x - 1) then x else 0

theorem flower_pots_count : total_flower_pots 28 = 28 :=
by
  sorry

end flower_pots_count_l33_33324


namespace solve_equation1_solve_equation2_l33_33673

theorem solve_equation1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 :=
by sorry

theorem solve_equation2 (x : ℝ) : 2 * x^2 - 6 * x = 3 ↔ x = (3 + Real.sqrt 15) / 2 ∨ x = (3 - Real.sqrt 15) / 2 :=
by sorry

end solve_equation1_solve_equation2_l33_33673


namespace banker_l33_33417

-- Define the given conditions
def present_worth : ℝ := 400
def interest_rate : ℝ := 0.10
def time_period : ℕ := 3

-- Define the amount due in the future
def amount_due (PW : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  PW * (1 + r) ^ n

-- Define the banker's gain
def bankers_gain (A PW : ℝ) : ℝ :=
  A - PW

-- State the theorem we need to prove
theorem banker's_gain_is_correct :
  bankers_gain (amount_due present_worth interest_rate time_period) present_worth = 132.4 :=
by sorry

end banker_l33_33417


namespace tigers_home_games_l33_33701

-- Definitions based on the conditions
def losses : ℕ := 12
def ties : ℕ := losses / 2
def wins : ℕ := 38

-- Statement to prove
theorem tigers_home_games : losses + ties + wins = 56 := by
  sorry

end tigers_home_games_l33_33701


namespace solve_for_x_l33_33811

noncomputable def simplified_end_expr (x : ℝ) := x = 4 - Real.sqrt 7 
noncomputable def expressed_as_2_statement (x : ℝ) := (x ^ 2 - 4 * x + 5) = (4 * (x - 1))
noncomputable def domain_condition (x : ℝ) := (-5 < x) ∧ (x < 3)

theorem solve_for_x (x : ℝ) :
  domain_condition x →
  (expressed_as_2_statement x ↔ simplified_end_expr x) :=
by
  sorry

end solve_for_x_l33_33811


namespace total_nuts_correct_l33_33727

-- Definitions for conditions
def w : ℝ := 0.25
def a : ℝ := 0.25
def p : ℝ := 0.15
def c : ℝ := 0.40

-- The theorem to be proven
theorem total_nuts_correct : w + a + p + c = 1.05 := by
  sorry

end total_nuts_correct_l33_33727


namespace range_of_a_l33_33764

noncomputable def f (x : ℝ) := -Real.exp x - x
noncomputable def g (a x : ℝ) := a * x + Real.cos x

theorem range_of_a :
  (∀ x : ℝ, ∃ y : ℝ, (g a y - g a y) / (y - y) * ((f x - f x) / (x - x)) = -1) →
  (0 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l33_33764


namespace james_total_vegetables_l33_33174

def james_vegetable_count (a b c d e : ℕ) : ℕ :=
  a + b + c + d + e

theorem james_total_vegetables 
    (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :
    a = 22 → b = 18 → c = 15 → d = 10 → e = 12 →
    james_vegetable_count a b c d e = 77 :=
by
  intros ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end james_total_vegetables_l33_33174


namespace smallest_n_in_T_and_largest_N_not_in_T_l33_33518

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (3 * x + 4) / (x + 3)}

theorem smallest_n_in_T_and_largest_N_not_in_T :
  (∀ n, n = 4 / 3 → n ∈ T) ∧ (∀ N, N = 3 → N ∉ T) :=
by
  sorry

end smallest_n_in_T_and_largest_N_not_in_T_l33_33518


namespace integers_in_range_eq_l33_33011

theorem integers_in_range_eq :
  {i : ℤ | i > -2 ∧ i ≤ 3} = {-1, 0, 1, 2, 3} :=
by
  sorry

end integers_in_range_eq_l33_33011


namespace tom_catches_up_in_60_minutes_l33_33220

-- Definitions of the speeds and initial distance
def lucy_speed : ℝ := 4  -- Lucy's speed in miles per hour
def tom_speed : ℝ := 6   -- Tom's speed in miles per hour
def initial_distance : ℝ := 2  -- Initial distance between Tom and Lucy in miles

-- Conclusion that needs to be proved
theorem tom_catches_up_in_60_minutes :
  (initial_distance / (tom_speed - lucy_speed)) * 60 = 60 :=
by
  sorry

end tom_catches_up_in_60_minutes_l33_33220


namespace chris_eats_donuts_l33_33707

def daily_donuts := 10
def days := 12
def donuts_eaten_per_day := 1
def boxes_filled := 10
def donuts_per_box := 10

-- Define the total number of donuts made.
def total_donuts := daily_donuts * days

-- Define the total number of donuts Jeff eats.
def jeff_total_eats := donuts_eaten_per_day * days

-- Define the remaining donuts after Jeff eats his share.
def remaining_donuts := total_donuts - jeff_total_eats

-- Define the total number of donuts in the boxes.
def donuts_in_boxes := boxes_filled * donuts_per_box

-- The proof problem:
theorem chris_eats_donuts : remaining_donuts - donuts_in_boxes = 8 :=
by
  -- Placeholder for proof
  sorry

end chris_eats_donuts_l33_33707


namespace find_product_of_constants_l33_33455

theorem find_product_of_constants
  (M1 M2 : ℝ)
  (h : ∀ x : ℝ, (x - 1) * (x - 2) ≠ 0 → (45 * x - 31) / (x * x - 3 * x + 2) = M1 / (x - 1) + M2 / (x - 2)) :
  M1 * M2 = -826 :=
sorry

end find_product_of_constants_l33_33455


namespace max_min_y_l33_33087

noncomputable def y (x : ℝ) : ℝ :=
  7 - 4 * (Real.sin x) * (Real.cos x) + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4

theorem max_min_y :
  (∃ x : ℝ, y x = 10) ∧ (∃ x : ℝ, y x = 6) := by
  sorry

end max_min_y_l33_33087


namespace ratio_revenue_l33_33604

variable (N D J : ℝ)

theorem ratio_revenue (h1 : J = N / 3) (h2 : D = 2.5 * (N + J) / 2) : N / D = 3 / 5 := by
  sorry

end ratio_revenue_l33_33604


namespace ksyusha_travel_time_l33_33977

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l33_33977


namespace initial_back_squat_weight_l33_33016

-- Define a structure to encapsulate the conditions
structure squat_conditions where
  initial_back_squat : ℝ
  front_squat_ratio : ℝ := 0.8
  back_squat_increase : ℝ := 50
  front_squat_triple_ratio : ℝ := 0.9
  total_weight_moved : ℝ := 540

-- Using the conditions provided to prove John's initial back squat weight
theorem initial_back_squat_weight (c : squat_conditions) :
  (3 * 3 * (c.front_squat_triple_ratio * (c.front_squat_ratio * c.initial_back_squat)) = c.total_weight_moved) →
  c.initial_back_squat = 540 / 6.48 := sorry

end initial_back_squat_weight_l33_33016


namespace total_number_of_squares_l33_33551

theorem total_number_of_squares (n : ℕ) (h : n = 12) : 
  ∃ t, t = 17 :=
by
  -- The proof is omitted here
  sorry

end total_number_of_squares_l33_33551


namespace practice_hours_l33_33263

-- Define the starting and ending hours, and the break duration
def start_hour : ℕ := 8
def end_hour : ℕ := 16
def break_duration : ℕ := 2

-- Compute the total practice hours
def total_practice_time : ℕ := (end_hour - start_hour) - break_duration

-- State that the computed practice time is equal to 6 hours
theorem practice_hours :
  total_practice_time = 6 := 
by
  -- Using the definitions provided to state the proof
  sorry

end practice_hours_l33_33263


namespace sequence_general_term_l33_33129

theorem sequence_general_term (a : ℕ → ℤ) (h₁ : a 1 = 2)
  (h₂ : ∀ n, 1 < n → a n = 2 * (n + a (n - 1))) :
  ∀ n, 1 ≤ n → a n = 2 ^ (n + 2) - 2 * n - 4 :=
by
  sorry

end sequence_general_term_l33_33129


namespace number_of_students_l33_33520

theorem number_of_students (N : ℕ) (h1 : (1/5 : ℚ) * N + (1/4 : ℚ) * N + (1/2 : ℚ) * N + 5 = N) : N = 100 :=
by
  sorry

end number_of_students_l33_33520


namespace popsicle_sticks_sum_l33_33991

-- Define the number of popsicle sticks each person has
def Gino_popsicle_sticks : Nat := 63
def my_popsicle_sticks : Nat := 50

-- Formulate the theorem stating the sum of popsicle sticks
theorem popsicle_sticks_sum : Gino_popsicle_sticks + my_popsicle_sticks = 113 := by
  sorry

end popsicle_sticks_sum_l33_33991


namespace time_for_tom_to_finish_wall_l33_33569

theorem time_for_tom_to_finish_wall (avery_rate tom_rate : ℝ) (combined_duration : ℝ) (remaining_wall : ℝ) :
  avery_rate = 1 / 2 ∧ tom_rate = 1 / 4 ∧ combined_duration = 1 ∧ remaining_wall = 1 / 4 →
  (remaining_wall / tom_rate) = 1 :=
by
  intros h
  -- Definitions from conditions
  let avery_rate := 1 / 2
  let tom_rate := 1 / 4
  let combined_duration := 1
  let remaining_wall := 1 / 4
  -- Question to be proven
  sorry

end time_for_tom_to_finish_wall_l33_33569


namespace total_value_is_correct_l33_33491

-- Define the conditions from the problem
def totalCoins : Nat := 324
def twentyPaiseCoins : Nat := 220
def twentyPaiseValue : Nat := 20
def twentyFivePaiseValue : Nat := 25
def paiseToRupees : Nat := 100

-- Calculate the number of 25 paise coins
def twentyFivePaiseCoins : Nat := totalCoins - twentyPaiseCoins

-- Calculate the total value of 20 paise and 25 paise coins in paise
def totalValueInPaise : Nat :=
  (twentyPaiseCoins * twentyPaiseValue) + 
  (twentyFivePaiseCoins * twentyFivePaiseValue)

-- Convert the total value from paise to rupees
def totalValueInRupees : Nat := totalValueInPaise / paiseToRupees

-- The theorem to be proved
theorem total_value_is_correct : totalValueInRupees = 70 := by
  sorry

end total_value_is_correct_l33_33491


namespace find_integer_n_l33_33660

theorem find_integer_n : 
  ∃ n : ℤ, 50 ≤ n ∧ n ≤ 150 ∧ (n % 7 = 0) ∧ (n % 9 = 3) ∧ (n % 4 = 3) ∧ n = 147 :=
by 
  -- sorry is used here as a placeholder for the actual proof
  sorry

end find_integer_n_l33_33660


namespace probability_even_heads_after_60_flips_l33_33200

noncomputable def P_n (n : ℕ) : ℝ :=
  if n = 0 then 1
  else (3 / 4) - (1 / 2) * P_n (n - 1)

theorem probability_even_heads_after_60_flips :
  P_n 60 = 1 / 2 * (1 + 1 / 2^60) :=
sorry

end probability_even_heads_after_60_flips_l33_33200


namespace find_common_difference_l33_33732

variable {a : ℕ → ℤ}  -- Define a sequence indexed by natural numbers, returning integers
variable (d : ℤ)  -- Define the common difference as an integer

-- The conditions: sequence is arithmetic, a_2 = 14, a_5 = 5
axiom arithmetic_sequence (n : ℕ) : a n = a 0 + n * d
axiom a_2_eq_14 : a 2 = 14
axiom a_5_eq_5 : a 5 = 5

-- The proof statement
theorem find_common_difference : d = -3 :=
by sorry

end find_common_difference_l33_33732


namespace find_number_l33_33734

theorem find_number (x : ℤ) (h : x - 27 = 49) : x = 76 := by
  sorry

end find_number_l33_33734


namespace smallest_prime_square_mod_six_l33_33269

theorem smallest_prime_square_mod_six (p : ℕ) (h_prime : Nat.Prime p) (h_mod : p^2 % 6 = 1) : p = 5 :=
sorry

end smallest_prime_square_mod_six_l33_33269


namespace rectangle_area_at_stage_8_l33_33184

-- Declare constants for the conditions.
def square_side_length : ℕ := 4
def number_of_stages : ℕ := 8
def area_of_single_square : ℕ := square_side_length * square_side_length

-- The statement to prove
theorem rectangle_area_at_stage_8 : number_of_stages * area_of_single_square = 128 := by
  sorry

end rectangle_area_at_stage_8_l33_33184


namespace mrs_lee_earnings_percentage_l33_33358

theorem mrs_lee_earnings_percentage 
  (M F : ℝ)
  (H1 : 1.20 * M = 0.5454545454545454 * (1.20 * M + F)) :
  M = 0.5 * (M + F) :=
by sorry

end mrs_lee_earnings_percentage_l33_33358


namespace jason_total_spent_l33_33819

theorem jason_total_spent (h_shorts : ℝ) (h_jacket : ℝ) (h1 : h_shorts = 14.28) (h2 : h_jacket = 4.74) : h_shorts + h_jacket = 19.02 :=
by
  rw [h1, h2]
  norm_num

end jason_total_spent_l33_33819


namespace rachel_lunch_problems_l33_33790

theorem rachel_lunch_problems (problems_per_minute minutes_before_bed total_problems : ℕ) 
    (h1 : problems_per_minute = 5)
    (h2 : minutes_before_bed = 12)
    (h3 : total_problems = 76) : 
    (total_problems - problems_per_minute * minutes_before_bed) = 16 :=
by
    sorry

end rachel_lunch_problems_l33_33790


namespace katherine_time_20_l33_33810

noncomputable def time_katherine_takes (k : ℝ) :=
  let time_naomi_takes_per_website := (5/4) * k
  let total_websites := 30
  let total_time_naomi := 750
  time_naomi_takes_per_website = 25 ∧ k = 20

theorem katherine_time_20 :
  ∃ k : ℝ, time_katherine_takes k :=
by
  use 20
  sorry

end katherine_time_20_l33_33810


namespace find_k_l33_33981

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of vector c depending on k
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- The theorem to be proven
theorem find_k (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) ↔ (k = -10 / 3) :=
by
  sorry

end find_k_l33_33981


namespace find_x_y_sum_squared_l33_33147

theorem find_x_y_sum_squared (x y : ℝ) (h1 : x * y = 6) (h2 : (1 / x^2) + (1 / y^2) = 7) (h3 : x - y = Real.sqrt 10) :
  (x + y)^2 = 264 := sorry

end find_x_y_sum_squared_l33_33147


namespace plus_signs_count_l33_33589

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l33_33589


namespace father_l33_33283

theorem father's_age :
  ∃ (S F : ℕ), 2 * S + F = 70 ∧ S + 2 * F = 95 ∧ F = 40 :=
by
  sorry

end father_l33_33283


namespace ten_elements_sequence_no_infinite_sequence_l33_33360

def is_valid_seq (a : ℕ → ℕ) : Prop :=
  ∀ n, (a (n + 1))^2 - 4 * (a n) * (a (n + 2)) ≥ 0

theorem ten_elements_sequence : 
  ∃ a : ℕ → ℕ, (a 9 + 1 = 10) ∧ is_valid_seq a :=
sorry

theorem no_infinite_sequence :
  ¬∃ a : ℕ → ℕ, is_valid_seq a ∧ ∀ n, a n ≥ 1 :=
sorry

end ten_elements_sequence_no_infinite_sequence_l33_33360


namespace farmer_price_per_dozen_l33_33620

noncomputable def price_per_dozen 
(farmer_chickens : ℕ) 
(eggs_per_chicken : ℕ) 
(total_money_made : ℕ) 
(total_weeks : ℕ) 
(eggs_per_dozen : ℕ) 
: ℕ :=
total_money_made / (total_weeks * (farmer_chickens * eggs_per_chicken) / eggs_per_dozen)

theorem farmer_price_per_dozen 
  (farmer_chickens : ℕ) 
  (eggs_per_chicken : ℕ) 
  (total_money_made : ℕ) 
  (total_weeks : ℕ) 
  (eggs_per_dozen : ℕ) 
  (h_chickens : farmer_chickens = 46) 
  (h_eggs_per_chicken : eggs_per_chicken = 6) 
  (h_money : total_money_made = 552) 
  (h_weeks : total_weeks = 8) 
  (h_dozen : eggs_per_dozen = 12) 
: price_per_dozen farmer_chickens eggs_per_chicken total_money_made total_weeks eggs_per_dozen = 3 := 
by 
  rw [h_chickens, h_eggs_per_chicken, h_money, h_weeks, h_dozen]
  have : (552 : ℕ) / (8 * (46 * 6) / 12) = 3 := by norm_num
  exact this

end farmer_price_per_dozen_l33_33620


namespace range_of_a_l33_33461

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + (1 / x) + a

theorem range_of_a (a : ℝ) (x : ℝ) (h : ∀ x : ℝ, f a x ≥ f a 0) : 0 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l33_33461


namespace lights_on_after_2011_toggles_l33_33925

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

end lights_on_after_2011_toggles_l33_33925


namespace sabrina_cookies_l33_33189

theorem sabrina_cookies :
  let S0 : ℕ := 28
  let S1 : ℕ := S0 - 10
  let S2 : ℕ := S1 + 3 * 10
  let S3 : ℕ := S2 - S2 / 3
  let S4 : ℕ := S3 + 16 / 4
  let S5 : ℕ := S4 - S4 / 2
  S5 = 18 := 
by
  -- begin proof here
  sorry

end sabrina_cookies_l33_33189


namespace fractional_inequality_solution_l33_33166

theorem fractional_inequality_solution (x : ℝ) :
  (x - 2) / (x + 1) < 0 ↔ -1 < x ∧ x < 2 :=
sorry

end fractional_inequality_solution_l33_33166


namespace balcony_more_than_orchestra_l33_33679

theorem balcony_more_than_orchestra (x y : ℕ) 
  (h1 : x + y = 370) 
  (h2 : 12 * x + 8 * y = 3320) : y - x = 190 :=
sorry

end balcony_more_than_orchestra_l33_33679


namespace rectangle_length_l33_33649

theorem rectangle_length
  (w l : ℝ)
  (h1 : l = 4 * w)
  (h2 : l * w = 100) :
  l = 20 :=
sorry

end rectangle_length_l33_33649


namespace solve_for_x_l33_33290

theorem solve_for_x (x : ℝ) (h : 0.20 * x = 0.15 * 1500 - 15) : x = 1050 := 
by
  sorry

end solve_for_x_l33_33290


namespace min_value_f_l33_33223

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x)^2 / (Real.cos x * Real.sin x - (Real.sin x)^2)

theorem min_value_f :
  ∃ x : ℝ, 0 < x ∧ x < Real.pi / 4 ∧ f x = 4 := 
sorry

end min_value_f_l33_33223


namespace inequality_for_positive_a_b_n_l33_33901

theorem inequality_for_positive_a_b_n (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 
  (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry

end inequality_for_positive_a_b_n_l33_33901


namespace average_weight_of_three_l33_33477

theorem average_weight_of_three
  (rachel_weight jimmy_weight adam_weight : ℝ)
  (h1 : rachel_weight = 75)
  (h2 : jimmy_weight = rachel_weight + 6)
  (h3 : adam_weight = rachel_weight - 15) :
  (rachel_weight + jimmy_weight + adam_weight) / 3 = 72 :=
by
  sorry

end average_weight_of_three_l33_33477


namespace sum_of_arithmetic_sequence_l33_33312

variable {α : Type*} [LinearOrderedField α]

def sum_arithmetic_sequence (a₁ d : α) (n : ℕ) : α :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_arithmetic_sequence {a₁ d : α}
  (h₁ : sum_arithmetic_sequence a₁ d 10 = 12) :
  (a₁ + 4 * d) + (a₁ + 5 * d) = 12 / 5 :=
by
  sorry

end sum_of_arithmetic_sequence_l33_33312


namespace prize_winners_l33_33468

variable (Elaine Frank George Hannah : Prop)

axiom ElaineImpliesFrank : Elaine → Frank
axiom FrankImpliesGeorge : Frank → George
axiom GeorgeImpliesHannah : George → Hannah
axiom OnlyTwoWinners : (Elaine ∧ Frank ∧ ¬George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ Frank ∧ George ∧ ¬Hannah) ∨ (¬Elaine ∧ Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah)

theorem prize_winners : (Elaine ∧ Frank ∧ ¬George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ Frank ∧ George ∧ ¬Hannah) ∨ (¬Elaine ∧ Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah) → (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah) :=
by
  sorry

end prize_winners_l33_33468


namespace stratified_sampling_correct_l33_33861

-- Define the conditions
def num_freshmen : ℕ := 900
def num_sophomores : ℕ := 1200
def num_seniors : ℕ := 600
def total_sample_size : ℕ := 135
def total_students := num_freshmen + num_sophomores + num_seniors

-- Proportions
def proportion_freshmen := (num_freshmen : ℚ) / total_students
def proportion_sophomores := (num_sophomores : ℚ) / total_students
def proportion_seniors := (num_seniors : ℚ) / total_students

-- Expected samples count
def expected_freshmen_samples := (total_sample_size : ℚ) * proportion_freshmen
def expected_sophomores_samples := (total_sample_size : ℚ) * proportion_sophomores
def expected_seniors_samples := (total_sample_size : ℚ) * proportion_seniors

-- Statement to be proven
theorem stratified_sampling_correct :
  expected_freshmen_samples = (45 : ℚ) ∧
  expected_sophomores_samples = (60 : ℚ) ∧
  expected_seniors_samples = (30 : ℚ) := by
  -- Provide the necessary proof or calculation
  sorry

end stratified_sampling_correct_l33_33861


namespace shortest_side_of_similar_triangle_l33_33098

def Triangle (a b c : ℤ) : Prop := a^2 + b^2 = c^2
def SimilarTriangles (a b c a' b' c' : ℤ) : Prop := ∃ k : ℤ, k > 0 ∧ a' = k * a ∧ b' = k * b ∧ c' = k * c 

theorem shortest_side_of_similar_triangle (a b c a' b' c' : ℤ)
  (h₀ : Triangle 15 b 17)
  (h₁ : SimilarTriangles 15 b 17 a' b' c')
  (h₂ : c' = 51) : a' = 24 :=
by
  sorry

end shortest_side_of_similar_triangle_l33_33098


namespace inverse_function_value_l33_33178

-- Defining the function g as a list of pairs
def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 3
  | 2 => 6
  | 3 => 1
  | 4 => 5
  | 5 => 4
  | 6 => 2
  | _ => 0 -- default case which should not be used

-- Defining the inverse function g_inv using the values determined from g
def g_inv (y : ℕ) : ℕ :=
  match y with
  | 3 => 1
  | 6 => 2
  | 1 => 3
  | 5 => 4
  | 4 => 5
  | 2 => 6
  | _ => 0 -- default case which should not be used

theorem inverse_function_value :
  g_inv (g_inv (g_inv 6)) = 2 :=
by
  sorry

end inverse_function_value_l33_33178


namespace hyperbola_eccentricity_proof_l33_33404

noncomputable def hyperbola_eccentricity (a b k1 k2 : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (C_on_hyperbola : ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) 
    (slope_condition : k1 * k2 = b^2 / a^2)
    (minimized_expr : ∀ k1 k2: ℝ , (2 / (k1 * k2)) + Real.log k1 + Real.log k2 ≥ (2 / (b^2 / a^2)) + Real.log ((b^2 / a^2))) : 
    ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_proof (a b k1 k2 : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (C_on_hyperbola : ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) 
    (slope_condition : k1 * k2 = b^2 / a^2)
    (minimized_expr : ∀ k1 k2: ℝ , (2 / (k1 * k2)) + Real.log k1 + Real.log k2 ≥ (2 / (b^2 / a^2)) + Real.log ((b^2 / a^2))) :    
    hyperbola_eccentricity a b k1 k2 ha hb C_on_hyperbola slope_condition minimized_expr = Real.sqrt 3 :=
sorry

end hyperbola_eccentricity_proof_l33_33404


namespace sum_of_digits_8_pow_2003_l33_33943

noncomputable def units_digit (n : ℕ) : ℕ :=
n % 10

noncomputable def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

noncomputable def sum_of_tens_and_units_digits (n : ℕ) : ℕ :=
units_digit n + tens_digit n

theorem sum_of_digits_8_pow_2003 :
  sum_of_tens_and_units_digits (8 ^ 2003) = 2 :=
by
  sorry

end sum_of_digits_8_pow_2003_l33_33943


namespace find_m_value_l33_33756

def magic_box_output (a b : ℝ) : ℝ := a^2 + b - 1

theorem find_m_value :
  ∃ m : ℝ, (magic_box_output m (-2 * m) = 2) ↔ (m = 3 ∨ m = -1) :=
by
  sorry

end find_m_value_l33_33756


namespace range_of_x_plus_one_over_x_l33_33209

theorem range_of_x_plus_one_over_x (x : ℝ) (h : x < 0) : x + 1/x ≤ -2 := by
  sorry

end range_of_x_plus_one_over_x_l33_33209


namespace total_games_l33_33010

-- Defining the conditions.
def games_this_month : ℕ := 9
def games_last_month : ℕ := 8
def games_next_month : ℕ := 7

-- Theorem statement to prove the total number of games.
theorem total_games : games_this_month + games_last_month + games_next_month = 24 := by
  sorry

end total_games_l33_33010


namespace exists_k_for_binary_operation_l33_33825

noncomputable def binary_operation (a b : ℤ) : ℤ := sorry

theorem exists_k_for_binary_operation :
  (∀ (a b c : ℤ), binary_operation a (b + c) = 
      binary_operation b a + binary_operation c a) →
  ∃ (k : ℤ), ∀ (a b : ℤ), binary_operation a b = k * a * b :=
by
  sorry

end exists_k_for_binary_operation_l33_33825


namespace aram_fraction_of_fine_l33_33250

theorem aram_fraction_of_fine
  (F : ℝ)
  (Joe_payment : ℝ := (1 / 4) * F + 3)
  (Peter_payment : ℝ := (1 / 3) * F - 3)
  (Aram_payment : ℝ := (1 / 2) * F - 4)
  (sum_payments_eq_F : Joe_payment + Peter_payment + Aram_payment = F):
  (Aram_payment / F) = (5 / 12) :=
by
  sorry

end aram_fraction_of_fine_l33_33250


namespace study_time_in_minutes_l33_33926

theorem study_time_in_minutes :
  let day1_hours := 2
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  total_hours * 60 = 540 :=
by
  let day1_hours := 2
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  sorry

end study_time_in_minutes_l33_33926


namespace lcm_of_9_12_15_l33_33656

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end lcm_of_9_12_15_l33_33656


namespace geometric_progression_common_ratio_l33_33247

theorem geometric_progression_common_ratio (a r : ℝ) 
(h_pos: a > 0)
(h_condition: ∀ n : ℕ, a * r^(n-1) = (a * r^n + a * r^(n+1))^2):
  r = 0.618 :=
sorry

end geometric_progression_common_ratio_l33_33247


namespace car_trip_problem_l33_33173

theorem car_trip_problem (a b c : ℕ) (x : ℕ) 
(h1 : 1 ≤ a) 
(h2 : a + b + c ≤ 9)
(h3 : 100 * b + 10 * c + a - 100 * a - 10 * b - c = 60 * x) 
: a^2 + b^2 + c^2 = 14 := 
by
  sorry

end car_trip_problem_l33_33173


namespace bamboo_tube_rice_capacity_l33_33798

theorem bamboo_tube_rice_capacity :
  ∃ (a d : ℝ), 3 * a + 3 * d * (1 + 2) = 4.5 ∧ 
               4 * (a + 5 * d) + 4 * d * (6 + 7 + 8) = 3.8 ∧ 
               (a + 3 * d) + (a + 4 * d) = 2.5 :=
by
  sorry

end bamboo_tube_rice_capacity_l33_33798


namespace Carter_card_number_l33_33128

-- Definitions based on conditions
def Marcus_cards : ℕ := 210
def difference : ℕ := 58

-- Definition to infer the number of Carter's baseball cards
def Carter_cards : ℕ := Marcus_cards - difference

-- Theorem statement asserting the number of baseball cards Carter has
theorem Carter_card_number : Carter_cards = 152 := by
  sorry

end Carter_card_number_l33_33128


namespace sarah_bus_time_l33_33088

noncomputable def totalTimeAway : ℝ := (4 + 15/60) + (5 + 15/60)  -- 9.5 hours
noncomputable def totalTimeAwayInMinutes : ℝ := totalTimeAway * 60  -- 570 minutes

noncomputable def timeInClasses : ℝ := 8 * 45  -- 360 minutes
noncomputable def timeInLunch : ℝ := 30  -- 30 minutes
noncomputable def timeInExtracurricular : ℝ := 1.5 * 60  -- 90 minutes
noncomputable def totalTimeInSchoolActivities : ℝ := timeInClasses + timeInLunch + timeInExtracurricular  -- 480 minutes

noncomputable def timeOnBus : ℝ := totalTimeAwayInMinutes - totalTimeInSchoolActivities  -- 90 minutes

theorem sarah_bus_time : timeOnBus = 90 := by
  sorry

end sarah_bus_time_l33_33088


namespace quotient_when_m_divided_by_11_is_2_l33_33481

theorem quotient_when_m_divided_by_11_is_2 :
  let n_values := [1, 2, 3, 4, 5]
  let squares := n_values.map (λ n => n^2)
  let remainders := List.eraseDup (squares.map (λ x => x % 11))
  let m := remainders.sum
  m / 11 = 2 :=
by
  sorry

end quotient_when_m_divided_by_11_is_2_l33_33481


namespace probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l33_33007

theorem probability_winning_on_first_draw : 
  let red := 1 
  let yellow := 3 
  red / (red + yellow) = 1 / 4 :=
by 
  sorry

theorem optimal_ball_to_add_for_fine_gift :
  let red := 1 
  let yellow := 3
  -- After adding a red ball: 2 red, 3 yellow
  let p1 := (2 * 1 + 3 * 2) / (2 + 3) / (1 + 3) = (2/5)
  -- After adding a yellow ball: 1 red, 4 yellow
  let p2 := (1 * 0 + 4 * 3) / (1 + 4) / (1 + 3) = (3/5)
  p1 < p2 :=
by 
  sorry

end probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l33_33007


namespace painting_frame_ratio_l33_33395

theorem painting_frame_ratio (x l : ℝ) (h1 : x > 0) (h2 : l > 0) 
  (h3 : (2 / 3) * x * x = (x + 2 * l) * ((3 / 2) * x + 2 * l) - x * (3 / 2) * x) :
  (x + 2 * l) / ((3 / 2) * x + 2 * l) = 3 / 4 :=
by
  sorry

end painting_frame_ratio_l33_33395


namespace all_possible_values_of_k_l33_33301

def is_partition_possible (k : ℕ) : Prop :=
  ∃ (A B : Finset ℕ), (A ∪ B = Finset.range (k + 1)) ∧ (A ∩ B = ∅) ∧ (A.sum id = 2 * B.sum id)

theorem all_possible_values_of_k (k : ℕ) : 
  is_partition_possible k → ∃ m : ℕ, k = 3 * m ∨ k = 3 * m - 1 :=
by
  intro h
  sorry

end all_possible_values_of_k_l33_33301


namespace greatest_y_value_l33_33884

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
sorry

end greatest_y_value_l33_33884


namespace find_b_l33_33133

theorem find_b
  (b : ℝ)
  (hx : ∃ y : ℝ, 4 * 3 + 2 * y = b ∧ 3 * 3 + 4 * y = 3 * b) :
  b = -15 :=
sorry

end find_b_l33_33133


namespace regular_nine_sided_polygon_has_27_diagonals_l33_33086

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l33_33086


namespace greatest_area_difference_l33_33574

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) 
  (h₁ : 2 * l₁ + 2 * w₁ = 160) 
  (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  1521 = (l₁ * w₁ - l₂ * w₂) → 
  (∃ l w : ℕ, 2 * l + 2 * w = 160 ∧ l * w = 1600 ∧ (l₁ = l ∧ w₁ = w) ∨ (l₂ = l ∧ w₂ = w)) ∧ 
  (∃ l w : ℕ, 2 * l + 2 * w = 160 ∧ l * w = 79 ∧ (l₁ = l ∧ w₁ = w) ∨ (l₂ = l ∧ w₂ = w)) :=
sorry

end greatest_area_difference_l33_33574


namespace intersection_A_B_l33_33779

def A := {x : ℝ | x^2 - ⌊x⌋ = 2}
def B := {x : ℝ | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, Real.sqrt 3} :=
sorry

end intersection_A_B_l33_33779


namespace geometric_sequence_b_value_l33_33124

theorem geometric_sequence_b_value :
  ∀ (a b c : ℝ),
  (a = 5 + 2 * Real.sqrt 6) →
  (c = 5 - 2 * Real.sqrt 6) →
  (b * b = a * c) →
  (b = 1 ∨ b = -1) :=
by
  intros a b c ha hc hgeometric
  sorry

end geometric_sequence_b_value_l33_33124


namespace length_of_room_l33_33948

theorem length_of_room 
  (width : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) 
  (h_width : width = 3.75) 
  (h_total_cost : total_cost = 16500) 
  (h_rate_per_sq_meter : rate_per_sq_meter = 800) : 
  ∃ length : ℝ, length = 5.5 :=
by
  sorry

end length_of_room_l33_33948


namespace simplify_expression_l33_33226

theorem simplify_expression (y : ℝ) : (y - 2) ^ 2 + 2 * (y - 2) * (4 + y) + (4 + y) ^ 2 = 4 * (y + 1) ^ 2 := 
by 
  sorry

end simplify_expression_l33_33226


namespace martin_goldfish_count_l33_33883

-- Define the initial number of goldfish
def initial_goldfish := 18

-- Define the number of goldfish that die each week
def goldfish_die_per_week := 5

-- Define the number of goldfish purchased each week
def goldfish_purchased_per_week := 3

-- Define the number of weeks
def weeks := 7

-- Calculate the expected number of goldfish after 7 weeks
noncomputable def final_goldfish := initial_goldfish - (goldfish_die_per_week * weeks) + (goldfish_purchased_per_week * weeks)

-- State the theorem and the proof target
theorem martin_goldfish_count : final_goldfish = 4 := 
sorry

end martin_goldfish_count_l33_33883


namespace binom_10_3_l33_33692

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l33_33692


namespace f_eq_g_iff_l33_33682

noncomputable def f (m n x : ℝ) := m * x^2 + n * x
noncomputable def g (p q x : ℝ) := p * x + q

theorem f_eq_g_iff (m n p q : ℝ) :
  (∀ x, f m n (g p q x) = g p q (f m n x)) ↔ 2 * m = n := by
  sorry

end f_eq_g_iff_l33_33682


namespace percent_defective_units_l33_33802

-- Definition of the given problem conditions
variable (D : ℝ) -- D represents the percentage of defective units

-- The main statement we want to prove
theorem percent_defective_units (h1 : 0.04 * D = 0.36) : D = 9 := by
  sorry

end percent_defective_units_l33_33802


namespace cars_without_features_l33_33697

theorem cars_without_features (total_cars cars_with_air_bags cars_with_power_windows cars_with_sunroofs 
                               cars_with_air_bags_and_power_windows cars_with_air_bags_and_sunroofs 
                               cars_with_power_windows_and_sunroofs cars_with_all_features: ℕ)
                               (h1 : total_cars = 80)
                               (h2 : cars_with_air_bags = 45)
                               (h3 : cars_with_power_windows = 40)
                               (h4 : cars_with_sunroofs = 25)
                               (h5 : cars_with_air_bags_and_power_windows = 20)
                               (h6 : cars_with_air_bags_and_sunroofs = 15)
                               (h7 : cars_with_power_windows_and_sunroofs = 10)
                               (h8 : cars_with_all_features = 8) : 
    total_cars - (cars_with_air_bags + cars_with_power_windows + cars_with_sunroofs 
                 - cars_with_air_bags_and_power_windows - cars_with_air_bags_and_sunroofs 
                 - cars_with_power_windows_and_sunroofs + cars_with_all_features) = 7 :=
by sorry

end cars_without_features_l33_33697


namespace find_m_l33_33944

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 2 * x - 5 * y + 20 = 0
def l2 (m x y : ℝ) : Prop := m * x + 2 * y - 10 = 0

-- Define the condition of perpendicularity
def lines_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- Proving the value of m given the conditions
theorem find_m (m : ℝ) :
  (∃ x y : ℝ, l1 x y) → (∃ x y : ℝ, l2 m x y) → lines_perpendicular 2 (-5 : ℝ) m 2 → m = 5 :=
sorry

end find_m_l33_33944


namespace transmission_time_estimation_l33_33254

noncomputable def number_of_blocks := 80
noncomputable def chunks_per_block := 640
noncomputable def transmission_rate := 160 -- chunks per second
noncomputable def seconds_per_minute := 60
noncomputable def total_chunks := number_of_blocks * chunks_per_block
noncomputable def total_time_seconds := total_chunks / transmission_rate
noncomputable def total_time_minutes := total_time_seconds / seconds_per_minute

theorem transmission_time_estimation : total_time_minutes = 5 := 
  sorry

end transmission_time_estimation_l33_33254


namespace parabola_translation_l33_33042

theorem parabola_translation :
  ∀ (x y : ℝ), y = 3 * x^2 →
  ∃ (new_x new_y : ℝ), new_y = 3 * (new_x + 3)^2 - 3 :=
by {
  sorry
}

end parabola_translation_l33_33042


namespace inequality_reversal_l33_33115

theorem inequality_reversal (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
by
  sorry

end inequality_reversal_l33_33115


namespace non_organic_chicken_price_l33_33079

theorem non_organic_chicken_price :
  ∀ (x : ℝ), (0.75 * x = 9) → (2 * (0.9 * x) = 21.6) :=
by
  intro x hx
  sorry

end non_organic_chicken_price_l33_33079


namespace cistern_problem_l33_33286

theorem cistern_problem (T : ℝ) (h1 : (1 / 2 - 1 / T) = 1 / 2.571428571428571) : T = 9 :=
by
  sorry

end cistern_problem_l33_33286


namespace problem_solving_ratio_l33_33122

theorem problem_solving_ratio 
  (total_mcqs : ℕ) (total_psqs : ℕ)
  (written_mcqs_fraction : ℚ) (total_remaining_questions : ℕ)
  (h1 : total_mcqs = 35)
  (h2 : total_psqs = 15)
  (h3 : written_mcqs_fraction = 2/5)
  (h4 : total_remaining_questions = 31) :
  (5 : ℚ) / 15 = (1 : ℚ) / 3 := 
by {
  -- given that 5 is the number of problem-solving questions already written,
  -- and 15 is the total number of problem-solving questions
  sorry
}

end problem_solving_ratio_l33_33122


namespace smallest_n_for_factorable_quadratic_l33_33132

open Int

theorem smallest_n_for_factorable_quadratic : ∃ n : ℤ, (∀ A B : ℤ, 3 * A * B = 72 → 3 * B + A = n) ∧ n = 35 :=
by
  sorry

end smallest_n_for_factorable_quadratic_l33_33132


namespace sequence_sum_S5_l33_33957

theorem sequence_sum_S5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : S 2 = 4)
  (h2 : ∀ n, a (n + 1) = 2 * S n + 1)
  (h3 : ∀ n, S (n + 1) - S n = a (n + 1)) :
  S 5 = 121 :=
by
  sorry

end sequence_sum_S5_l33_33957


namespace value_of_a3_a6_a9_l33_33560

variable (a : ℕ → ℤ) -- Define the sequence a as a function from natural numbers to integers
variable (d : ℤ) -- Define the common difference d as an integer

-- Conditions
axiom h1 : a 1 + a 4 + a 7 = 39
axiom h2 : a 2 + a 5 + a 8 = 33
axiom h3 : ∀ n : ℕ, a (n+1) = a n + d -- This condition ensures the sequence is arithmetic

-- Theorem: We need to prove the value of a_3 + a_6 + a_9 is 27
theorem value_of_a3_a6_a9 : a 3 + a 6 + a 9 = 27 :=
by
  sorry

end value_of_a3_a6_a9_l33_33560


namespace total_cost_of_items_l33_33097

variables (E P M : ℝ)

-- Conditions
def condition1 : Prop := E + 3 * P + 2 * M = 240
def condition2 : Prop := 2 * E + 5 * P + 4 * M = 440

-- Question to prove
def question (E P M : ℝ) : ℝ := 3 * E + 4 * P + 6 * M

theorem total_cost_of_items (E P M : ℝ) :
  condition1 E P M →
  condition2 E P M →
  question E P M = 520 := 
by 
  intros h1 h2
  sorry

end total_cost_of_items_l33_33097


namespace division_problem_l33_33995

-- Define the involved constants and operations
def expr1 : ℚ := 5 / 2 * 3
def expr2 : ℚ := 100 / expr1

-- Formulate the final equality
theorem division_problem : expr2 = 40 / 3 :=
  by sorry

end division_problem_l33_33995


namespace sum_first_six_terms_geometric_seq_l33_33751

theorem sum_first_six_terms_geometric_seq (a r : ℝ)
  (h1 : a + a * r = 12)
  (h2 : a + a * r + a * r^2 + a * r^3 = 36) :
  a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 = 84 :=
sorry

end sum_first_six_terms_geometric_seq_l33_33751


namespace complex_division_l33_33622

-- Define complex numbers and imaginary unit
def i : ℂ := Complex.I

theorem complex_division : (3 + 4 * i) / (1 + i) = (7 / 2) + (1 / 2) * i :=
by
  sorry

end complex_division_l33_33622


namespace colored_pictures_count_l33_33815

def initial_pictures_count : ℕ := 44 + 44
def pictures_left : ℕ := 68

theorem colored_pictures_count : initial_pictures_count - pictures_left = 20 := by
  -- Definitions and proof will go here
  sorry

end colored_pictures_count_l33_33815


namespace savings_per_month_l33_33376

noncomputable def annual_salary : ℝ := 48000
noncomputable def monthly_payments : ℝ := 12
noncomputable def savings_percentage : ℝ := 0.10

theorem savings_per_month :
  (annual_salary / monthly_payments) * savings_percentage = 400 :=
by
  sorry

end savings_per_month_l33_33376


namespace ab_value_l33_33035

theorem ab_value (a b : ℝ) (h : 6 * a = 20 ∧ 7 * b = 20) : 84 * (a * b) = 800 :=
by sorry

end ab_value_l33_33035


namespace trajectory_of_M_l33_33975

variables {x y : ℝ}

theorem trajectory_of_M (h : y / (x + 2) + y / (x - 2) = 2) (hx : x ≠ 2) (hx' : x ≠ -2) :
  x * y - x^2 + 4 = 0 :=
by sorry

end trajectory_of_M_l33_33975


namespace Mary_chestnuts_l33_33745

noncomputable def MaryPickedTwicePeter (P M : ℕ) := M = 2 * P
noncomputable def LucyPickedMorePeter (P L : ℕ) := L = P + 2
noncomputable def TotalPicked (P M L : ℕ) := P + M + L = 26

theorem Mary_chestnuts (P M L : ℕ) (h1 : MaryPickedTwicePeter P M) (h2 : LucyPickedMorePeter P L) (h3 : TotalPicked P M L) :
  M = 12 :=
sorry

end Mary_chestnuts_l33_33745


namespace isosceles_triangle_side_length_l33_33071

theorem isosceles_triangle_side_length (P : ℕ := 53) (base : ℕ := 11) (x : ℕ)
  (h1 : x + x + base = P) : x = 21 :=
by {
  -- The proof goes here.
  sorry
}

end isosceles_triangle_side_length_l33_33071


namespace prime_only_one_solution_l33_33742

theorem prime_only_one_solution (p : ℕ) (hp : Nat.Prime p) : 
  (∃ k : ℕ, 2 * p^4 - p^2 + 16 = k^2) → p = 3 := 
by 
  sorry

end prime_only_one_solution_l33_33742


namespace red_apples_sold_l33_33807

-- Define the variables and constants
variables (R G : ℕ)

-- Conditions (Definitions)
def ratio_condition : Prop := R / G = 8 / 3
def combine_condition : Prop := R + G = 44

-- Theorem statement to show number of red apples sold is 32 under given conditions
theorem red_apples_sold : ratio_condition R G → combine_condition R G → R = 32 :=
by
sorry

end red_apples_sold_l33_33807


namespace tournament_rounds_l33_33965

/-- 
Given a tournament where each participant plays several games with every other participant
and a total of 224 games were played, prove that the number of rounds in the competition is 8.
-/
theorem tournament_rounds (x y : ℕ) (hx : x > 1) (hy : y > 0) (h : x * (x - 1) * y = 448) : y = 8 :=
sorry

end tournament_rounds_l33_33965


namespace hyperbola_eq_l33_33490

theorem hyperbola_eq (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (hyp_eq : ∀ x y, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1)
  (asymptote : b / a = Real.sqrt 3)
  (focus_parabola : c = 4) : 
  a^2 = 4 ∧ b^2 = 12 := by
sorry

end hyperbola_eq_l33_33490


namespace pipe_drain_rate_l33_33272

theorem pipe_drain_rate 
(T r_A r_B r_C : ℕ) 
(h₁ : T = 950) 
(h₂ : r_A = 40) 
(h₃ : r_B = 30) 
(h₄ : ∃ m : ℕ, m = 57 ∧ (T = (m / 3) * (r_A + r_B - r_C))) : 
r_C = 20 :=
sorry

end pipe_drain_rate_l33_33272


namespace point_b_not_inside_circle_a_l33_33275

theorem point_b_not_inside_circle_a (a : ℝ) : a < 5 → ¬ (1 < a ∧ a < 5) :=
by
  sorry

end point_b_not_inside_circle_a_l33_33275


namespace trainB_destination_time_l33_33072

def trainA_speed : ℕ := 90
def trainB_speed : ℕ := 135
def trainA_time_after_meeting : ℕ := 9
def trainB_time_after_meeting (x : ℕ) : ℕ := 18 - 3 * x

theorem trainB_destination_time : (trainA_time_after_meeting, trainA_speed) = (9, 90) → 
  (trainB_speed, trainB_time_after_meeting 3) = (135, 3) := by
  sorry

end trainB_destination_time_l33_33072


namespace cylinder_sphere_surface_area_ratio_l33_33762

theorem cylinder_sphere_surface_area_ratio 
  (d : ℝ) -- d represents the diameter of the sphere and the height of the cylinder
  (S1 S2 : ℝ) -- Surface areas of the cylinder and the sphere
  (r := d / 2) -- radius of the sphere
  (S1 := 6 * π * r ^ 2) -- surface area of the cylinder
  (S2 := 4 * π * r ^ 2) -- surface area of the sphere
  : S1 / S2 = 3 / 2 :=
  sorry

end cylinder_sphere_surface_area_ratio_l33_33762


namespace solve_inequality_f_ge_x_no_positive_a_b_satisfy_conditions_l33_33332

noncomputable def f (x : ℝ) : ℝ :=
  |2 * x - 1| - |2 * x - 2|

theorem solve_inequality_f_ge_x :
  {x : ℝ | f x >= x} = {x : ℝ | x <= -1 ∨ x = 1} :=
by sorry

theorem no_positive_a_b_satisfy_conditions :
  ∀ (a b : ℝ), a > 0 → b > 0 → (a + 2 * b = 1) → (2 / a + 1 / b = 4 - 1 / (a * b)) → false :=
by sorry

end solve_inequality_f_ge_x_no_positive_a_b_satisfy_conditions_l33_33332


namespace beef_weight_loss_percentage_l33_33659

theorem beef_weight_loss_percentage (weight_before weight_after weight_lost_percentage : ℝ) 
  (before_process : weight_before = 861.54)
  (after_process : weight_after = 560) 
  (weight_lost : (weight_before - weight_after) = 301.54)
  : weight_lost_percentage = 34.99 :=
by
  sorry

end beef_weight_loss_percentage_l33_33659


namespace prove_fn_value_l33_33303

noncomputable def f (x : ℝ) : ℝ := 2^x / (2^x + 3 * x)

theorem prove_fn_value
  (m n : ℝ)
  (h1 : 2^(m + n) = 3 * m * n)
  (h2 : f m = -1 / 3) :
  f n = 4 :=
by
  sorry

end prove_fn_value_l33_33303


namespace union_A_B_l33_33450

noncomputable def A : Set ℝ := {x | x^2 + x - 6 = 0}
noncomputable def B : Set ℝ := {x | x^2 - 4 = 0}

theorem union_A_B : A ∪ B = {-3, -2, 2} := by
  sorry

end union_A_B_l33_33450


namespace cost_per_box_types_l33_33112

-- Definitions based on conditions
def cost_type_B := 1500
def cost_type_A := cost_type_B + 500

-- Given conditions
def condition1 : cost_type_A = cost_type_B + 500 := by sorry
def condition2 : 6000 / (cost_type_B + 500) = 4500 / cost_type_B := by sorry

-- Theorem to be proved
theorem cost_per_box_types :
  cost_type_A = 2000 ∧ cost_type_B = 1500 ∧
  (∃ (m : ℕ), 20 ≤ m ∧ m ≤ 25 ∧ 2000 * (50 - m) + 1500 * m ≤ 90000) ∧
  (∃ (a b : ℕ), 2500 * a + 3500 * b = 87500 ∧ a + b ≤ 33) :=
sorry

end cost_per_box_types_l33_33112


namespace tank_fill_time_l33_33447

noncomputable def fill_time (T rA rB rC : ℝ) : ℝ :=
  let cycle_fill := rA + rB + rC
  let cycles := T / cycle_fill
  let cycle_time := 3
  cycles * cycle_time

theorem tank_fill_time
  (T : ℝ) (rA rB rC : ℝ) (hT : T = 800) (hrA : rA = 40) (hrB : rB = 30) (hrC : rC = -20) :
  fill_time T rA rB rC = 48 :=
by
  sorry

end tank_fill_time_l33_33447


namespace average_temperature_week_l33_33264

theorem average_temperature_week 
  (T_sun : ℝ := 40)
  (T_mon : ℝ := 50)
  (T_tue : ℝ := 65)
  (T_wed : ℝ := 36)
  (T_thu : ℝ := 82)
  (T_fri : ℝ := 72)
  (T_sat : ℝ := 26) :
  (T_sun + T_mon + T_tue + T_wed + T_thu + T_fri + T_sat) / 7 = 53 :=
by
  sorry

end average_temperature_week_l33_33264


namespace final_price_is_correct_l33_33092

def original_price : ℝ := 450
def discounts : List ℝ := [0.10, 0.20, 0.05]

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

noncomputable def final_sale_price (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem final_price_is_correct:
  final_sale_price original_price discounts = 307.8 :=
by
  sorry

end final_price_is_correct_l33_33092


namespace correct_operation_l33_33222

variable (a b : ℝ)

theorem correct_operation :
  -a^6 / a^3 = -a^3 := by
  sorry

end correct_operation_l33_33222


namespace gain_amount_is_ten_l33_33527

theorem gain_amount_is_ten (S : ℝ) (C : ℝ) (g : ℝ) (G : ℝ) 
  (h1 : S = 110) (h2 : g = 0.10) (h3 : S = C + g * C) : G = 10 :=
by 
  sorry

end gain_amount_is_ten_l33_33527


namespace demand_change_for_revenue_l33_33941

theorem demand_change_for_revenue (P D D' : ℝ)
  (h1 : D' = (1.10 * D) / 1.20)
  (h2 : P' = 1.20 * P)
  (h3 : P * D = P' * D') :
  (D' - D) / D * 100 = -8.33 := by
sorry

end demand_change_for_revenue_l33_33941


namespace sum_greater_than_product_l33_33160

theorem sum_greater_than_product (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b > a * b) ↔ (a = 1 ∨ b = 1) := 
by { sorry }

end sum_greater_than_product_l33_33160


namespace volume_of_prism_l33_33508

variables (a b : ℝ) (α β : ℝ)
  (h1 : a > b)
  (h2 : 0 < α ∧ α < π / 2)
  (h3 : 0 < β ∧ β < π / 2)

noncomputable def volume_prism : ℝ :=
  (a^2 - b^2) * (a - b) / 8 * (Real.tan α)^2 * Real.tan β

theorem volume_of_prism (a b α β : ℝ) (h1 : a > b) (h2 : 0 < α ∧ α < π / 2) (h3 : 0 < β ∧ β < π / 2) :
  volume_prism a b α β = (a^2 - b^2) * (a - b) / 8 * (Real.tan α)^2 * Real.tan β := by
  sorry

end volume_of_prism_l33_33508


namespace cost_of_fencing_l33_33816

open Real

theorem cost_of_fencing
  (ratio_length_width : ∃ x : ℝ, 3 * x * 2 * x = 3750)
  (cost_per_meter : ℝ := 0.50) :
  ∃ cost : ℝ, cost = 125 := by
  sorry

end cost_of_fencing_l33_33816


namespace min_ab_value_l33_33040

theorem min_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = a + 9 * b + 7) : a * b ≥ 49 :=
sorry

end min_ab_value_l33_33040


namespace angle_relationship_l33_33241

open Real

variables (A B C D : Point)
variables (AB AC AD : ℝ)
variables (CAB DAC BDC DBC : ℝ)
variables (k : ℝ)

-- Given conditions
axiom h1 : AB = AC
axiom h2 : AC = AD
axiom h3 : DAC = k * CAB

-- Proof to be shown
theorem angle_relationship : DBC = k * BDC :=
  sorry

end angle_relationship_l33_33241


namespace five_pow_10000_mod_1000_l33_33739

theorem five_pow_10000_mod_1000 (h : 5^500 ≡ 1 [MOD 1000]) : 5^10000 ≡ 1 [MOD 1000] := sorry

end five_pow_10000_mod_1000_l33_33739


namespace ratio_of_radii_l33_33552

open Real

theorem ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 5 * π * a^2) : a / b = 1 / sqrt 6 :=
by
  sorry

end ratio_of_radii_l33_33552


namespace number_to_remove_l33_33525

def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

theorem number_to_remove (s : List ℕ) (x : ℕ) 
  (h₀ : s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
  (h₁ : x ∈ s)
  (h₂ : mean (List.erase s x) = 6.1) : x = 5 := sorry

end number_to_remove_l33_33525


namespace num_of_nickels_l33_33210

theorem num_of_nickels (n : ℕ) (h1 : n = 17) (h2 : (17 * n) - 1 = 18 * (n - 1)) : n = 17 → 17 * n = 289 → ∃ k, k = 2 :=
by 
  intros hn hv
  sorry

end num_of_nickels_l33_33210


namespace no_such_function_exists_l33_33835

theorem no_such_function_exists (f : ℕ → ℕ) (h : ∀ n, f (f n) = n + 2019) : false :=
sorry

end no_such_function_exists_l33_33835


namespace total_necklaces_made_l33_33393

-- Definitions based on conditions
def first_machine_necklaces : ℝ := 45
def second_machine_necklaces : ℝ := 2.4 * first_machine_necklaces

-- Proof statement
theorem total_necklaces_made : (first_machine_necklaces + second_machine_necklaces) = 153 := by
  sorry

end total_necklaces_made_l33_33393


namespace circles_are_separate_l33_33976

def circle_center (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

theorem circles_are_separate :
  circle_center 0 0 1 x y → 
  circle_center 3 (-4) 3 x' y' →
  dist (0, 0) (3, -4) > 1 + 3 :=
by
  intro h₁ h₂
  sorry

end circles_are_separate_l33_33976


namespace bryan_books_l33_33529

theorem bryan_books (books_per_continent : ℕ) (total_books : ℕ) 
  (h1 : books_per_continent = 122) 
  (h2 : total_books = 488) : 
  total_books / books_per_continent = 4 := 
by 
  sorry

end bryan_books_l33_33529


namespace at_least_4_stayed_l33_33718

-- We define the number of people and their respective probabilities of staying.
def numPeople : ℕ := 8
def numCertain : ℕ := 5
def numUncertain : ℕ := 3
def probUncertainStay : ℚ := 1 / 3

-- We state the problem formally:
theorem at_least_4_stayed :
  (probUncertainStay ^ 3 * 3 + (probUncertainStay ^ 2 * (2 / 3) * 3) + (probUncertainStay * (2 / 3)^2 * 3)) = 19 / 27 :=
by
  sorry

end at_least_4_stayed_l33_33718


namespace probability_different_numbers_probability_sum_six_probability_three_odds_in_five_throws_l33_33483

-- Probability for different numbers facing up when die is thrown twice
theorem probability_different_numbers :
  let n_faces := 6
  let total_outcomes := n_faces * n_faces
  let favorable_outcomes := n_faces * (n_faces - 1)
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry -- Proof to be filled

-- Probability for sum of numbers being 6 when die is thrown twice
theorem probability_sum_six :
  let n_faces := 6
  let total_outcomes := n_faces * n_faces
  let favorable_outcomes := 5
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 5 / 36 :=
by
  sorry -- Proof to be filled

-- Probability for exactly three outcomes being odd when die is thrown five times
theorem probability_three_odds_in_five_throws :
  let n_faces := 6
  let n_throws := 5
  let p_odd := 3 / n_faces
  let p_even := 1 - p_odd
  let binomial_coeff := Nat.choose n_throws 3
  let p_three_odds := (binomial_coeff : ℚ) * (p_odd ^ 3) * (p_even ^ 2)
  p_three_odds = 5 / 16 :=
by
  sorry -- Proof to be filled

end probability_different_numbers_probability_sum_six_probability_three_odds_in_five_throws_l33_33483


namespace factorize_polynomial_l33_33539

theorem factorize_polynomial {x : ℝ} : x^3 + 2 * x^2 - 3 * x = x * (x + 3) * (x - 1) :=
by sorry

end factorize_polynomial_l33_33539


namespace find_integer_mul_a_l33_33902

noncomputable def integer_mul_a (a b : ℤ) (n : ℤ) : Prop :=
  n * a * (-8 * b) + a * b = 89 ∧ n < 0 ∧ n * a < 0 ∧ -8 * b < 0

theorem find_integer_mul_a (a b : ℤ) (n : ℤ) (h : integer_mul_a a b n) : n = -11 :=
  sorry

end find_integer_mul_a_l33_33902


namespace intersection_complement_l33_33181

open Set

noncomputable def N := {x : ℕ | true}

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}
def C_N (B : Set ℕ) : Set ℕ := {n ∈ N | n ∉ B}

theorem intersection_complement :
  A ∩ (C_N B) = {1} :=
by
  sorry

end intersection_complement_l33_33181


namespace problem_statement_l33_33351

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -6 < x ∧ x < 1}

theorem problem_statement : M ∩ N = N := by
  ext x
  constructor
  · intro h
    exact h.2
  · intro h
    exact ⟨h.2, h⟩

end problem_statement_l33_33351


namespace find_marks_of_a_l33_33436

theorem find_marks_of_a (A B C D E : ℕ)
  (h1 : (A + B + C) / 3 = 48)
  (h2 : (A + B + C + D) / 4 = 47)
  (h3 : E = D + 3)
  (h4 : (B + C + D + E) / 4 = 48) : 
  A = 43 :=
by
  sorry

end find_marks_of_a_l33_33436


namespace find_a_l33_33820
open Real

noncomputable def f (a x : ℝ) := x * sin x + a * x

theorem find_a (a : ℝ) : (deriv (f a) (π / 2) = 1) → a = 0 := by
  sorry

end find_a_l33_33820


namespace value_division_l33_33486

theorem value_division (x : ℝ) (h1 : 54 / x = 54 - 36) : x = 3 := by
  sorry

end value_division_l33_33486


namespace inequality_solution_set_l33_33665

theorem inequality_solution_set (m n : ℝ) 
    (h₁ : ∀ x : ℝ, mx - n > 0 ↔ x < 1 / 3) 
    (h₂ : m + n < 0) 
    (h₃ : m = 3 * n) 
    (h₄ : n < 0) : 
    ∀ x : ℝ, (m + n) * x < n - m ↔ x > -1 / 2 :=
by
  sorry

end inequality_solution_set_l33_33665


namespace max_sum_of_abc_l33_33950

theorem max_sum_of_abc (A B C : ℕ) (h1 : A * B * C = 1386) (h2 : A ≠ B) (h3 : A ≠ C) (h4 : B ≠ C) : 
  A + B + C ≤ 88 :=
sorry

end max_sum_of_abc_l33_33950


namespace find_a_l33_33095

-- Definitions from conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2
def directrix : ℝ := 1

-- Statement to prove
theorem find_a (a : ℝ) (h : directrix = 1) : a = -1/4 :=
sorry

end find_a_l33_33095


namespace solve_system_l33_33631

theorem solve_system (x y : ℚ) 
  (h1 : x + 2 * y = -1) 
  (h2 : 2 * x + y = 3) : 
  x + y = 2 / 3 := 
sorry

end solve_system_l33_33631


namespace upper_limit_arun_weight_l33_33545

theorem upper_limit_arun_weight (x w : ℝ) :
  (65 < w ∧ w < x) ∧
  (60 < w ∧ w < 70) ∧
  (w ≤ 68) ∧
  (w = 67) →
  x = 68 :=
by
  sorry

end upper_limit_arun_weight_l33_33545


namespace Y_tagged_value_l33_33908

variables (W X Y Z : ℕ)
variables (tag_W : W = 200)
variables (tag_X : X = W / 2)
variables (tag_Z : Z = 400)
variables (total : W + X + Y + Z = 1000)

theorem Y_tagged_value : Y = 300 :=
by sorry

end Y_tagged_value_l33_33908


namespace square_area_on_parabola_l33_33971

theorem square_area_on_parabola (s : ℝ) (h : 0 < s) (hG : (3 + s)^2 - 6 * (3 + s) + 5 = -2 * s) : 
  (2 * s) * (2 * s) = 24 - 8 * Real.sqrt 5 := 
by 
  sorry

end square_area_on_parabola_l33_33971


namespace find_nonzero_c_l33_33984

def quadratic_has_unique_solution (c b : ℝ) : Prop :=
  (b^4 + (1 - 4 * c) * b^2 + 1 = 0) ∧ (1 - 4 * c)^2 - 4 = 0

theorem find_nonzero_c (c : ℝ) (b : ℝ) (h_nonzero : c ≠ 0) (h_unique_sol : quadratic_has_unique_solution c b) : 
  c = 3 / 4 := 
sorry

end find_nonzero_c_l33_33984


namespace cricket_current_average_l33_33661

theorem cricket_current_average (A : ℕ) (h1: 10 * A + 77 = 11 * (A + 4)) : 
  A = 33 := 
by 
  sorry

end cricket_current_average_l33_33661


namespace only_natural_number_solution_l33_33137

theorem only_natural_number_solution (n : ℕ) :
  (∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = n * x * y * z) ↔ (n = 3) := 
sorry

end only_natural_number_solution_l33_33137


namespace sonya_falls_6_l33_33934

def number_of_falls_steven : ℕ := 3
def number_of_falls_stephanie : ℕ := number_of_falls_steven + 13
def number_of_falls_sonya : ℕ := (number_of_falls_stephanie / 2) - 2

theorem sonya_falls_6 : number_of_falls_sonya = 6 := 
by
  -- The actual proof is to be filled in here
  sorry

end sonya_falls_6_l33_33934


namespace Kyler_wins_l33_33749

variable (K : ℕ) -- Kyler's wins

/- Constants based on the problem statement -/
def Peter_wins := 5
def Peter_losses := 3
def Emma_wins := 2
def Emma_losses := 4
def Total_games := 15
def Kyler_losses := 4

/- Definition that calculates total games played -/
def total_games_played := 2 * Total_games

/- Game equation based on the total count of played games -/
def game_equation := Peter_wins + Peter_losses + Emma_wins + Emma_losses + K + Kyler_losses = total_games_played

/- Question: Calculate Kyler's wins assuming the given conditions -/
theorem Kyler_wins : K = 1 :=
by
  sorry

end Kyler_wins_l33_33749


namespace girls_in_club_l33_33738

/-
A soccer club has 30 members. For a recent team meeting, only 18 members could attend:
one-third of the girls attended but all of the boys attended. Prove that the number of 
girls in the soccer club is 18.
-/

variables (B G : ℕ)

-- Conditions
def total_members (B G : ℕ) := B + G = 30
def meeting_attendance (B G : ℕ) := (1/3 : ℚ) * G + B = 18

theorem girls_in_club (B G : ℕ) (h1 : total_members B G) (h2 : meeting_attendance B G) : G = 18 :=
  sorry

end girls_in_club_l33_33738


namespace find_weight_B_l33_33675

-- Define the weights of A, B, and C
variables (A B C : ℝ)

-- Conditions
def avg_weight_ABC := A + B + C = 135
def avg_weight_AB := A + B = 80
def avg_weight_BC := B + C = 86

-- The statement to be proved
theorem find_weight_B (h1: avg_weight_ABC A B C) (h2: avg_weight_AB A B) (h3: avg_weight_BC B C) : B = 31 :=
sorry

end find_weight_B_l33_33675


namespace mary_final_books_l33_33105

-- Initial number of books
def initial_books : ℕ := 72

-- Books received each month from book club for 12 months
def books_from_club : ℕ := 12 * 1

-- Books bought from different sources
def books_from_bookstore : ℕ := 5
def books_from_yard_sales : ℕ := 2

-- Books received as gifts
def books_from_daughter : ℕ := 1
def books_from_mother : ℕ := 4

-- Books gotten rid of
def books_donated : ℕ := 12
def books_sold : ℕ := 3

-- Final calculation
theorem mary_final_books : 
  initial_books + books_from_club + books_from_bookstore + books_from_yard_sales + books_from_daughter + books_from_mother - (books_donated + books_sold) = 81 :=
  by sorry

end mary_final_books_l33_33105


namespace minimum_surface_area_l33_33292

def small_cuboid_1_length := 3 -- Edge length of small cuboid
def small_cuboid_2_length := 4 -- Edge length of small cuboid
def small_cuboid_3_length := 5 -- Edge length of small cuboid

def num_small_cuboids := 24 -- Number of small cuboids used to build the large cuboid

def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

def large_cuboid_length := 15 -- Corrected length dimension
def large_cuboid_width := 10  -- Corrected width dimension
def large_cuboid_height := 16 -- Corrected height dimension

theorem minimum_surface_area : surface_area large_cuboid_length large_cuboid_width large_cuboid_height = 788 := by
  sorry -- Proof to be completed

end minimum_surface_area_l33_33292


namespace total_legs_among_tables_l33_33326

noncomputable def total_legs (total_tables four_legged_tables: ℕ) : ℕ :=
  let three_legged_tables := total_tables - four_legged_tables
  4 * four_legged_tables + 3 * three_legged_tables

theorem total_legs_among_tables : total_legs 36 16 = 124 := by
  sorry

end total_legs_among_tables_l33_33326


namespace toys_sold_in_first_week_l33_33783

/-
  Problem statement:
  An online toy store stocked some toys. It sold some toys at the first week and 26 toys at the second week.
  If it had 19 toys left and there were 83 toys in stock at the beginning, how many toys were sold in the first week?
-/

theorem toys_sold_in_first_week (initial_stock toys_left toys_sold_second_week : ℕ) 
  (h_initial_stock : initial_stock = 83) 
  (h_toys_left : toys_left = 19) 
  (h_toys_sold_second_week : toys_sold_second_week = 26) : 
  (initial_stock - toys_left - toys_sold_second_week) = 38 :=
by
  -- Proof goes here
  sorry

end toys_sold_in_first_week_l33_33783


namespace xiaobin_duration_l33_33856

def t1 : ℕ := 9
def t2 : ℕ := 15

theorem xiaobin_duration : t2 - t1 = 6 := by
  sorry

end xiaobin_duration_l33_33856


namespace prob_three_red_cards_l33_33168

noncomputable def probability_of_three_red_cards : ℚ :=
  let total_ways := 52 * 51 * 50
  let ways_to_choose_red_cards := 26 * 25 * 24
  ways_to_choose_red_cards / total_ways

theorem prob_three_red_cards : probability_of_three_red_cards = 4 / 17 := sorry

end prob_three_red_cards_l33_33168


namespace segments_count_bound_l33_33403

-- Define the overall setup of the problem
variable (n : ℕ) (points : Finset ℕ)

-- The main hypothesis and goal
theorem segments_count_bound (hn : n ≥ 2) (hpoints : points.card = 3 * n) :
  ∃ A B : Finset (ℕ × ℕ), (∀ (i j : ℕ), i ∈ points → j ∈ points → i ≠ j → ((i, j) ∈ A ↔ (i, j) ∉ B)) ∧
  ∀ (X : Finset ℕ) (hX : X.card = n), ∃ C : Finset (ℕ × ℕ), (C ⊆ A) ∧ (X ⊆ points) ∧
  (∃ count : ℕ, count ≥ (n - 1) / 6 ∧ count = C.card ∧ ∀ (a b : ℕ), (a, b) ∈ C → a ∈ X ∧ b ∈ points \ X) := sorry

end segments_count_bound_l33_33403


namespace Andre_final_price_l33_33463

theorem Andre_final_price :
  let treadmill_price := 1350
  let treadmill_discount_rate := 0.30
  let plate_price := 60
  let num_of_plates := 2
  let plate_discount_rate := 0.15
  let sales_tax_rate := 0.07
  let treadmill_discount := treadmill_price * treadmill_discount_rate
  let treadmill_discounted_price := treadmill_price - treadmill_discount
  let total_plate_price := plate_price * num_of_plates
  let plate_discount := total_plate_price * plate_discount_rate
  let plate_discounted_price := total_plate_price - plate_discount
  let total_price_before_tax := treadmill_discounted_price + plate_discounted_price
  let sales_tax := total_price_before_tax * sales_tax_rate
  let final_price := total_price_before_tax + sales_tax
  final_price = 1120.29 := 
by
  repeat { 
    sorry 
  }

end Andre_final_price_l33_33463


namespace divisibility_by_7_l33_33230

theorem divisibility_by_7 (n : ℕ) (h : 0 < n) : 7 ∣ (3 ^ (2 * n + 2) - 2 ^ (n + 1)) :=
sorry

end divisibility_by_7_l33_33230


namespace area_of_triangle_ABC_l33_33892

-- Define the sides of the triangle
def AB : ℝ := 12
def BC : ℝ := 9

-- Define the expected area of the triangle
def expectedArea : ℝ := 54

-- Prove the area of the triangle using the given conditions
theorem area_of_triangle_ABC : (1/2) * AB * BC = expectedArea := 
by
  sorry

end area_of_triangle_ABC_l33_33892


namespace vector_parallel_x_is_neg1_l33_33162

variables (a b : ℝ × ℝ)
variable (x : ℝ)

def vectors_parallel : Prop := 
  (a = (1, -1)) ∧ (b = (x, 1)) ∧ (a.1 * b.2 - a.2 * b.1 = 0)

theorem vector_parallel_x_is_neg1 (h : vectors_parallel a b x) : x = -1 :=
sorry

end vector_parallel_x_is_neg1_l33_33162


namespace find_omega2019_value_l33_33852

noncomputable def omega_n (n : ℕ) : ℝ := (2 * n - 1) * Real.pi / 2

theorem find_omega2019_value :
  omega_n 2019 = 4037 * Real.pi / 2 :=
by
  sorry

end find_omega2019_value_l33_33852


namespace function_always_negative_iff_l33_33199

theorem function_always_negative_iff (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ -4 < k ∧ k ≤ 0 :=
by
  -- Proof skipped
  sorry

end function_always_negative_iff_l33_33199


namespace ratio_of_weights_l33_33615

variable (x : ℝ)

-- Conditions as definitions in Lean 4
def seth_loss : ℝ := 17.5
def jerome_loss : ℝ := 17.5 * x
def veronica_loss : ℝ := 17.5 + 1.5 -- 19 pounds
def total_loss : ℝ := seth_loss + jerome_loss x + veronica_loss

-- Statement to prove
theorem ratio_of_weights (h : total_loss x = 89) : jerome_loss x / seth_loss = 3 :=
by sorry

end ratio_of_weights_l33_33615


namespace max_value_expression_l33_33046

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l33_33046


namespace jessica_money_left_l33_33451

theorem jessica_money_left : 
  let initial_amount := 11.73
  let amount_spent := 10.22
  initial_amount - amount_spent = 1.51 :=
by
  sorry

end jessica_money_left_l33_33451


namespace sales_tax_calculation_l33_33142

theorem sales_tax_calculation 
  (total_amount_paid : ℝ)
  (tax_rate : ℝ)
  (cost_tax_free : ℝ) :
  total_amount_paid = 30 → tax_rate = 0.08 → cost_tax_free = 12.72 → 
  (∃ sales_tax : ℝ, sales_tax = 1.28) :=
by
  intros H1 H2 H3
  sorry

end sales_tax_calculation_l33_33142


namespace final_total_cost_l33_33969

def initial_spiral_cost : ℝ := 15
def initial_planner_cost : ℝ := 10
def spiral_discount_rate : ℝ := 0.20
def planner_discount_rate : ℝ := 0.15
def num_spirals : ℝ := 4
def num_planners : ℝ := 8
def sales_tax_rate : ℝ := 0.07

theorem final_total_cost :
  let discounted_spiral_cost := initial_spiral_cost * (1 - spiral_discount_rate)
  let discounted_planner_cost := initial_planner_cost * (1 - planner_discount_rate)
  let total_before_tax := num_spirals * discounted_spiral_cost + num_planners * discounted_planner_cost
  let total_tax := total_before_tax * sales_tax_rate
  let total_cost := total_before_tax + total_tax
  total_cost = 124.12 :=
by
  sorry

end final_total_cost_l33_33969


namespace leopards_arrangement_l33_33655

theorem leopards_arrangement :
  let leopards := 10
  let shortest := 3
  let remaining := leopards - shortest
  (shortest! * remaining! = 30240) :=
by
  let leopards := 10
  let shortest := 3
  let remaining := leopards - shortest
  have factorials_eq: shortest! * remaining! = 30240 := sorry
  exact factorials_eq

end leopards_arrangement_l33_33655


namespace spencer_total_distance_l33_33233

-- Definitions for the given conditions
def distance_house_to_library : ℝ := 0.3
def distance_library_to_post_office : ℝ := 0.1
def distance_post_office_to_home : ℝ := 0.4

-- Define the total distance based on the given conditions
def total_distance : ℝ := distance_house_to_library + distance_library_to_post_office + distance_post_office_to_home

-- Statement to prove
theorem spencer_total_distance : total_distance = 0.8 := by
  sorry

end spencer_total_distance_l33_33233


namespace problem_statement_l33_33676

theorem problem_statement : (29.7 + 83.45) - 0.3 = 112.85 := sorry

end problem_statement_l33_33676


namespace compute_alpha_l33_33205

-- Define the main hypothesis with complex numbers
variable (α γ : ℂ)
variable (h1 : γ = 4 + 3 * Complex.I)
variable (h2 : ∃r1 r2: ℝ, r1 > 0 ∧ r2 > 0 ∧ (α + γ = r1) ∧ (Complex.I * (α - 3 * γ) = r2))

-- The main theorem
theorem compute_alpha : α = 12 + 3 * Complex.I :=
by
  sorry

end compute_alpha_l33_33205


namespace isosceles_triangle_vertex_angle_l33_33629

theorem isosceles_triangle_vertex_angle (a b : ℕ) (h : a = 2 * b) 
  (h1 : a + b + b = 180): a = 90 ∨ a = 36 :=
by
  sorry

end isosceles_triangle_vertex_angle_l33_33629


namespace find_a_and_root_l33_33639

def equation_has_double_root (a x : ℝ) : Prop :=
  a * x^2 + 4 * x - 1 = 0

theorem find_a_and_root (a x : ℝ)
  (h_eqn : equation_has_double_root a x)
  (h_discriminant : 16 + 4 * a = 0) :
  a = -4 ∧ x = 1 / 2 :=
sorry

end find_a_and_root_l33_33639


namespace total_students_accommodated_l33_33887

def num_columns : ℕ := 4
def num_rows : ℕ := 10
def num_buses : ℕ := 6

theorem total_students_accommodated : num_columns * num_rows * num_buses = 240 := by
  sorry

end total_students_accommodated_l33_33887


namespace calculate_max_income_l33_33493

variables 
  (total_lunch_pasta : ℕ) (total_lunch_chicken : ℕ) (total_lunch_fish : ℕ)
  (sold_lunch_pasta : ℕ) (sold_lunch_chicken : ℕ) (sold_lunch_fish : ℕ)
  (dinner_pasta : ℕ) (dinner_chicken : ℕ) (dinner_fish : ℕ)
  (price_pasta : ℝ) (price_chicken : ℝ) (price_fish : ℝ)
  (discount : ℝ)
  (max_income : ℝ)

def unsold_lunch_pasta := total_lunch_pasta - sold_lunch_pasta
def unsold_lunch_chicken := total_lunch_chicken - sold_lunch_chicken
def unsold_lunch_fish := total_lunch_fish - sold_lunch_fish

def discounted_price (price : ℝ) := price * (1 - discount)

def income_lunch (sold : ℕ) (price : ℝ) := sold * price
def income_dinner (fresh : ℕ) (price : ℝ) := fresh * price
def income_unsold (unsold : ℕ) (price : ℝ) := unsold * discounted_price price

theorem calculate_max_income 
  (h_pasta_total : total_lunch_pasta = 8) (h_chicken_total : total_lunch_chicken = 5) (h_fish_total : total_lunch_fish = 4)
  (h_pasta_sold : sold_lunch_pasta = 6) (h_chicken_sold : sold_lunch_chicken = 3) (h_fish_sold : sold_lunch_fish = 3)
  (h_dinner_pasta : dinner_pasta = 2) (h_dinner_chicken : dinner_chicken = 2) (h_dinner_fish : dinner_fish = 1)
  (h_price_pasta: price_pasta = 12) (h_price_chicken: price_chicken = 15) (h_price_fish: price_fish = 18)
  (h_discount: discount = 0.10) 
  : max_income = 136.80 :=
  sorry

end calculate_max_income_l33_33493


namespace tan_ratio_of_triangle_l33_33047

theorem tan_ratio_of_triangle (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = (3 / 5) * c) : 
  Real.tan A / Real.tan B = 4 :=
sorry

end tan_ratio_of_triangle_l33_33047


namespace find_z_l33_33061

theorem find_z : 
    ∃ z : ℝ, ( ( 2 ^ 5 ) * ( 9 ^ 2 ) ) / ( z * ( 3 ^ 5 ) ) = 0.16666666666666666 ↔ z = 64 :=
by
    sorry

end find_z_l33_33061


namespace contribution_per_person_l33_33354

-- Define constants for the given conditions
def total_price : ℕ := 67
def coupon : ℕ := 4
def number_of_people : ℕ := 3

-- The theorem to prove
theorem contribution_per_person : (total_price - coupon) / number_of_people = 21 :=
by 
  -- The proof is omitted for this exercise
  sorry

end contribution_per_person_l33_33354


namespace students_per_row_l33_33228

theorem students_per_row (x : ℕ) : 45 = 11 * x + 1 → x = 4 :=
by
  intro h
  sorry

end students_per_row_l33_33228


namespace range_of_c_l33_33496

variable {a c : ℝ}

theorem range_of_c (h : a ≥ 1 / 8) (sufficient_but_not_necessary : ∀ x > 0, 2 * x + a / x ≥ c) : c ≤ 1 := 
sorry

end range_of_c_l33_33496


namespace minimum_value_inequality_l33_33296

theorem minimum_value_inequality (m n : ℝ) (h₁ : m > n) (h₂ : n > 0) : m + (n^2 - mn + 4)/(m - n) ≥ 4 :=
  sorry

end minimum_value_inequality_l33_33296


namespace problem_statement_l33_33989

theorem problem_statement (x Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) :
    10 * (6 * x + 14 * Real.pi) = 4 * Q := 
sorry

end problem_statement_l33_33989


namespace solve_equation_l33_33935

theorem solve_equation :
  ∀ x : ℝ,
  (1 / (x^2 + 12 * x - 9) + 
   1 / (x^2 + 3 * x - 9) + 
   1 / (x^2 - 12 * x - 9) = 0) ↔ 
  (x = 1 ∨ x = -9 ∨ x = 3 ∨ x = -3) := 
by
  sorry

end solve_equation_l33_33935


namespace calculate_rent_l33_33145

def monthly_income : ℝ := 3200
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous : ℝ := 200
def car_payment : ℝ := 350
def gas_maintenance : ℝ := 350

def total_expenses : ℝ := utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment + gas_maintenance
def rent : ℝ := monthly_income - total_expenses

theorem calculate_rent : rent = 1250 := by
  -- condition proof here
  sorry

end calculate_rent_l33_33145


namespace problem1_problem2_l33_33221

variables (a b : ℝ)

theorem problem1 : ((a^2)^3 / (-a)^2) = a^4 :=
sorry

theorem problem2 : ((a + 2 * b) * (a + b) - 3 * a * (a + b)) = -2 * a^2 + 2 * b^2 :=
sorry

end problem1_problem2_l33_33221


namespace expected_babies_is_1008_l33_33206

noncomputable def babies_expected_after_loss
  (num_kettles : ℕ)
  (pregnancies_per_kettle : ℕ)
  (babies_per_pregnancy : ℕ)
  (loss_percentage : ℤ) : ℤ :=
  let total_babies := num_kettles * pregnancies_per_kettle * babies_per_pregnancy
  let survival_rate := (100 - loss_percentage) / 100
  total_babies * survival_rate

theorem expected_babies_is_1008 :
  babies_expected_after_loss 12 20 6 30 = 1008 :=
by
  sorry

end expected_babies_is_1008_l33_33206


namespace find_m_n_l33_33532

-- Define the set A
def set_A : Set ℝ := {x | |x + 2| < 3}

-- Define the set B in terms of a variable m
def set_B (m : ℝ) : Set ℝ := {x | (x - m) * (x - 2) < 0}

-- State the theorem
theorem find_m_n (m n : ℝ) (hA : set_A = {x | -5 < x ∧ x < 1}) (h_inter : set_A ∩ set_B m = {x | -1 < x ∧ x < n}) : 
  m = -1 ∧ n = 1 :=
by
  -- Proof is omitted
  sorry

end find_m_n_l33_33532


namespace pints_in_5_liters_l33_33587

-- Define the condition based on the given conversion factor from liters to pints
def conversion_factor : ℝ := 2.1

-- The statement we need to prove
theorem pints_in_5_liters : 5 * conversion_factor = 10.5 :=
by sorry

end pints_in_5_liters_l33_33587


namespace average_class_score_l33_33853

theorem average_class_score (total_students assigned_day_students make_up_date_students : ℕ)
  (assigned_day_percentage make_up_date_percentage assigned_day_avg_score make_up_date_avg_score : ℝ)
  (h1 : total_students = 100)
  (h2 : assigned_day_percentage = 0.70)
  (h3 : make_up_date_percentage = 0.30)
  (h4 : assigned_day_students = 70)
  (h5 : make_up_date_students = 30)
  (h6 : assigned_day_avg_score = 55)
  (h7 : make_up_date_avg_score = 95) :
  (assigned_day_avg_score * assigned_day_students + make_up_date_avg_score * make_up_date_students) / total_students = 67 :=
by
  sorry

end average_class_score_l33_33853


namespace shopkeeper_gain_percent_l33_33085

theorem shopkeeper_gain_percent
    (SP₁ SP₂ CP : ℝ)
    (h₁ : SP₁ = 187)
    (h₂ : SP₂ = 264)
    (h₃ : SP₁ = 0.85 * CP) :
    ((SP₂ - CP) / CP) * 100 = 20 := by 
  sorry

end shopkeeper_gain_percent_l33_33085


namespace rosy_fish_count_l33_33170

theorem rosy_fish_count (L R T : ℕ) (hL : L = 10) (hT : T = 19) : R = T - L := by
  sorry

end rosy_fish_count_l33_33170


namespace value_of_expression_l33_33886

theorem value_of_expression : (1 / (3 + 1 / (3 + 1 / (3 - 1 / 3)))) = (27 / 89) :=
by
  sorry

end value_of_expression_l33_33886


namespace train_speed_l33_33832

theorem train_speed
  (length_train : ℝ)
  (length_bridge : ℝ)
  (time_seconds : ℝ) :
  length_train = 140 →
  length_bridge = 235.03 →
  time_seconds = 30 →
  (length_train + length_bridge) / time_seconds * 3.6 = 45.0036 :=
by
  intros h1 h2 h3
  sorry

end train_speed_l33_33832


namespace original_rectangle_area_l33_33271

theorem original_rectangle_area
  (A : ℝ)
  (h1 : ∀ (a : ℝ), a = 2 * A)
  (h2 : 4 * A = 32) : 
  A = 8 := 
by
  sorry

end original_rectangle_area_l33_33271


namespace intersection_of_sets_l33_33253

-- Define sets A and B as given in the conditions
def A : Set ℝ := { x | -2 < x ∧ x < 2 }

def B : Set ℝ := {0, 1, 2}

-- Define the proposition to be proved
theorem intersection_of_sets : A ∩ B = {0, 1} :=
by
  sorry

end intersection_of_sets_l33_33253


namespace find_number_l33_33638

theorem find_number (A : ℕ) (B : ℕ) (H1 : B = 300) (H2 : Nat.lcm A B = 2310) (H3 : Nat.gcd A B = 30) : A = 231 := 
by 
  sorry

end find_number_l33_33638


namespace area_of_right_triangle_l33_33990

theorem area_of_right_triangle
  (BC AC : ℝ)
  (h1 : BC * AC = 16) : 
  0.5 * BC * AC = 8 := by 
  sorry

end area_of_right_triangle_l33_33990


namespace starting_number_l33_33599

theorem starting_number (x : ℝ) (h : (x + 26) / 2 = 19) : x = 12 :=
by
  sorry

end starting_number_l33_33599


namespace sum_of_coeffs_l33_33911

theorem sum_of_coeffs (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5, (2 - x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5)
  → (a_0 = 32 ∧ 1 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5)
  → a_1 + a_2 + a_3 + a_4 + a_5 = -31 :=
by
  sorry

end sum_of_coeffs_l33_33911


namespace unique_line_through_A_parallel_to_a_l33_33951

variables {Point Line Plane : Type}
variables {α β : Plane}
variables {a l : Line}
variables {A : Point}

-- Definitions are necessary from conditions in step a)
def parallel_to (a b : Line) : Prop := sorry -- Definition that two lines are parallel
def contains (p : Plane) (x : Point) : Prop := sorry -- Definition that a plane contains a point
def line_parallel_to_plane (a : Line) (p : Plane) : Prop := sorry -- Definition that a line is parallel to a plane

-- Given conditions in the proof problem
variable (a_parallel_α : line_parallel_to_plane a α)
variable (A_in_α : contains α A)

-- Statement to be proven: There is only one line that passes through point A and is parallel to line a, and that line is within plane α.
theorem unique_line_through_A_parallel_to_a : 
  ∃! l : Line, contains α A ∧ parallel_to l a := sorry

end unique_line_through_A_parallel_to_a_l33_33951


namespace least_xy_value_l33_33946

theorem least_xy_value (x y : ℕ) (hposx : x > 0) (hposy : y > 0) (h : 1/x + 1/(3*y) = 1/8) :
  xy = 96 :=
by
  sorry

end least_xy_value_l33_33946


namespace roots_of_polynomial_l33_33937

theorem roots_of_polynomial : ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (x + 2) = 0 ↔ x = 2 ∨ x = 3 ∨ x = -2 :=
by sorry

end roots_of_polynomial_l33_33937


namespace find_x_l33_33420

/-!
# Problem Statement
Given that the segment with endpoints (-8, 0) and (32, 0) is the diameter of a circle,
and the point (x, 20) lies on the circle, prove that x = 12.
-/

def point_on_circle (x y : ℝ) (center_x center_y radius : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = radius^2

theorem find_x : 
  let center_x := (32 + (-8)) / 2
  let center_y := (0 + 0) / 2
  let radius := (32 - (-8)) / 2
  ∃ x : ℝ, point_on_circle x 20 center_x center_y radius → x = 12 :=
by
  sorry

end find_x_l33_33420


namespace find_m_n_sum_l33_33211

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem find_m_n_sum (x₀ : ℝ) (m n : ℤ) 
  (hmn_adj : n = m + 1) 
  (hx₀_zero : f x₀ = 0) 
  (hx₀_interval : (m : ℝ) < x₀ ∧ x₀ < (n : ℝ)) :
  m + n = 1 :=
sorry

end find_m_n_sum_l33_33211


namespace find_t_find_s_find_a_find_c_l33_33736

-- Proof Problem I4.1
theorem find_t (p q r t : ℝ) (h1 : (p + q + r) / 3 = 12) (h2 : (p + q + r + t + 2 * t) / 5 = 15) : t = 13 :=
sorry

-- Proof Problem I4.2
theorem find_s (k t s : ℝ) (hk : k ≠ 0) (h1 : k^4 + (1 / k^4) = t + 1) (h2 : t = 13) (h_s : s = k^2 + (1 / k^2)) : s = 4 :=
sorry

-- Proof Problem I4.3
theorem find_a (s a b : ℝ) (hxₘ : 1 ≠ 11) (hyₘ : 2 ≠ 7) (h1 : (a, b) = ((1 * 11 + s * 1) / (1 + s), (1 * 7 + s * 2) / (1 + s))) (h_s : s = 4) : a = 3 :=
sorry

-- Proof Problem I4.4
theorem find_c (a c : ℝ) (h1 : ∀ x, a * x^2 + 12 * x + c = 0 → (a*x^2 + 12 * x + c = 0)) (h2 : ∃ x, a * x^2 + 12 * x + c = 0) : c = 36 / a :=
sorry

end find_t_find_s_find_a_find_c_l33_33736


namespace symmetric_function_expression_l33_33081

variable (f : ℝ → ℝ)
variable (h_sym : ∀ x y, f (-2 - x) = - f x)
variable (h_def : ∀ x, 0 < x → f x = 1 / x)

theorem symmetric_function_expression : ∀ x, x < -2 → f x = 1 / (2 + x) :=
by
  intro x
  intro hx
  sorry

end symmetric_function_expression_l33_33081


namespace percentage_equivalence_l33_33066

theorem percentage_equivalence (x : ℝ) :
  (70 / 100) * 600 = (x / 100) * 1050 → x = 40 :=
by
  sorry

end percentage_equivalence_l33_33066


namespace find_value_of_x_l33_33500

theorem find_value_of_x (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 36) : x = 28 := 
sorry

end find_value_of_x_l33_33500


namespace stratified_leader_selection_probability_of_mixed_leaders_l33_33714

theorem stratified_leader_selection :
  let num_first_grade := 150
  let num_second_grade := 100
  let total_leaders := 5
  let leaders_first_grade := (total_leaders * num_first_grade) / (num_first_grade + num_second_grade)
  let leaders_second_grade := (total_leaders * num_second_grade) / (num_first_grade + num_second_grade)
  leaders_first_grade = 3 ∧ leaders_second_grade = 2 :=
by
  sorry

theorem probability_of_mixed_leaders :
  let num_first_grade_leaders := 3
  let num_second_grade_leaders := 2
  let total_leaders := 5
  let total_ways := 10
  let favorable_ways := 6
  (favorable_ways / total_ways) = (3 / 5) :=
by
  sorry

end stratified_leader_selection_probability_of_mixed_leaders_l33_33714


namespace correct_statement_l33_33978

variables {α β γ : ℝ → ℝ → ℝ → Prop} -- planes
variables {a b c : ℝ → ℝ → ℝ → Prop} -- lines

def is_parallel (P Q : ℝ → ℝ → ℝ → Prop) : Prop :=
∀ x : ℝ, ∀ y : ℝ, ∀ z : ℝ, (P x y z → Q x y z) ∧ (Q x y z → P x y z)

def is_perpendicular (L : ℝ → ℝ → ℝ → Prop) (P : ℝ → ℝ → ℝ → Prop) : Prop :=
∀ x : ℝ, ∀ y : ℝ, ∀ z : ℝ, L x y z ↔ ¬ P x y z 

theorem correct_statement : 
  (is_perpendicular a α) → 
  (is_parallel b β) → 
  (is_parallel α β) → 
  (is_perpendicular a b) :=
by
  sorry

end correct_statement_l33_33978


namespace sample_capacity_l33_33096

theorem sample_capacity (n : ℕ) (A B C : ℕ) (h_ratio : A / (A + B + C) = 3 / 14) (h_A : A = 15) : n = 70 :=
by
  sorry

end sample_capacity_l33_33096


namespace min_value_of_expr_l33_33699

theorem min_value_of_expr (a b c : ℝ) (h1 : 0 < a ∧ a ≤ b ∧ b ≤ c) (h2 : a * b * c = 1) :
    (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (a + c)) + 1 / (c ^ 3 * (a + b))) ≥ 3 / 2 := 
by
  sorry

end min_value_of_expr_l33_33699


namespace cylinder_surface_area_l33_33759

namespace SurfaceAreaProof

variables (a b : ℝ)

theorem cylinder_surface_area (a b : ℝ) :
  (2 * Real.pi * a * b) = (2 * Real.pi * a * b) :=
by sorry

end SurfaceAreaProof

end cylinder_surface_area_l33_33759


namespace percentage_exceeds_l33_33022

-- Defining the constants and conditions
variables {y z x : ℝ}

-- Conditions
def condition1 (y x : ℝ) : Prop := x = 0.6 * y
def condition2 (x z : ℝ) : Prop := z = 1.25 * x

-- Proposition to prove
theorem percentage_exceeds (hyx : condition1 y x) (hxz : condition2 x z) : y = 4/3 * z :=
by 
  -- We skip the proof as requested
  sorry

end percentage_exceeds_l33_33022


namespace math_problem_l33_33367

theorem math_problem :
  (∃ n : ℕ, 28 = 4 * n) ∧
  ((∃ n1 : ℕ, 361 = 19 * n1) ∧ ¬(∃ n2 : ℕ, 63 = 19 * n2)) ∧
  (¬((∃ n3 : ℕ, 90 = 30 * n3) ∧ ¬(∃ n4 : ℕ, 65 = 30 * n4))) ∧
  ((∃ n5 : ℕ, 45 = 15 * n5) ∧ (∃ n6 : ℕ, 30 = 15 * n6)) ∧
  (∃ n7 : ℕ, 144 = 12 * n7) :=
by {
  -- We need to prove each condition to be true and then prove the statements A, B, D, E are true.
  sorry
}

end math_problem_l33_33367


namespace max_a_value_l33_33118

theorem max_a_value (a : ℝ) :
  (∀ x : ℝ, x < a → x^2 - 2 * x - 3 > 0) →
  (¬ (∀ x : ℝ, x^2 - 2 * x - 3 > 0 → x < a)) →
  a = -1 :=
by
  sorry

end max_a_value_l33_33118


namespace marilyn_bananas_l33_33425

-- Defining the conditions
def boxes : ℕ := 8
def bananas_per_box : ℕ := 5

-- The statement that Marilyn has 40 bananas
theorem marilyn_bananas : boxes * bananas_per_box = 40 :=
by
  sorry

end marilyn_bananas_l33_33425


namespace certain_number_is_correct_l33_33505

theorem certain_number_is_correct (x : ℝ) (h : x / 1.45 = 17.5) : x = 25.375 :=
sorry

end certain_number_is_correct_l33_33505


namespace system_of_equations_solution_l33_33554

theorem system_of_equations_solution (x y : ℝ) :
  (2 * x + y + 2 * x * y = 11 ∧ 2 * x^2 * y + x * y^2 = 15) ↔
  ((x = 1/2 ∧ y = 5) ∨ (x = 1 ∧ y = 3) ∨ (x = 3/2 ∧ y = 2) ∨ (x = 5/2 ∧ y = 1)) :=
by 
  sorry

end system_of_equations_solution_l33_33554


namespace range_of_m_l33_33027

-- Definitions and the main problem statement
def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f m x < 0) ↔ (-4 < m ∧ m ≤ 0) :=
by
  sorry

end range_of_m_l33_33027


namespace simple_interest_sum_l33_33156

theorem simple_interest_sum (SI R T : ℝ) (hSI : SI = 4016.25) (hR : R = 0.01) (hT : T = 3) :
  SI / (R * T) = 133875 := by
  sorry

end simple_interest_sum_l33_33156


namespace smallest_sum_zero_l33_33614

theorem smallest_sum_zero : ∃ x ∈ ({-1, -2, 1, 2} : Set ℤ), ∀ y ∈ ({-1, -2, 1, 2} : Set ℤ), x + 0 ≤ y + 0 :=
sorry

end smallest_sum_zero_l33_33614


namespace triangle_is_right_angled_l33_33668

theorem triangle_is_right_angled
  (a b c : ℝ)
  (h1 : a ≠ c)
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : c > 0)
  (h5 : ∃ x : ℝ, x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0 ∧ x ≠ 0) :
  c^2 + b^2 = a^2 :=
by sorry

end triangle_is_right_angled_l33_33668


namespace fraction_solution_l33_33996

theorem fraction_solution (x : ℝ) (h1 : (x - 4) / (x^2) = 0) (h2 : x ≠ 0) : x = 4 :=
sorry

end fraction_solution_l33_33996


namespace widget_cost_reduction_l33_33556

theorem widget_cost_reduction:
  ∀ (C C_reduced : ℝ), 
  6 * C = 27.60 → 
  8 * C_reduced = 27.60 → 
  C - C_reduced = 1.15 := 
by
  intros C C_reduced h1 h2
  sorry

end widget_cost_reduction_l33_33556


namespace total_length_of_wire_l33_33894

-- Definitions based on conditions
def num_squares : ℕ := 15
def length_of_grid : ℕ := 10
def width_of_grid : ℕ := 5
def height_of_grid : ℕ := 3
def side_length : ℕ := length_of_grid / width_of_grid -- 2 units
def num_horizontal_wires : ℕ := height_of_grid + 1    -- 4 wires
def num_vertical_wires : ℕ := width_of_grid + 1      -- 6 wires
def total_length_horizontal_wires : ℕ := num_horizontal_wires * length_of_grid -- 40 units
def total_length_vertical_wires : ℕ := num_vertical_wires * (height_of_grid * side_length) -- 36 units

-- The theorem to prove the total length of wire needed
theorem total_length_of_wire : total_length_horizontal_wires + total_length_vertical_wires = 76 :=
by
  sorry

end total_length_of_wire_l33_33894


namespace smallest_of_seven_even_numbers_sum_448_l33_33776

theorem smallest_of_seven_even_numbers_sum_448 :
  ∃ n : ℤ, n + (n+2) + (n+4) + (n+6) + (n+8) + (n+10) + (n+12) = 448 ∧ n = 58 := 
by
  sorry

end smallest_of_seven_even_numbers_sum_448_l33_33776


namespace remainder_when_divided_by_7_l33_33695

-- Definitions based on conditions
def k_condition (k : ℕ) : Prop :=
(k % 5 = 2) ∧ (k % 6 = 5) ∧ (k < 38)

-- Theorem based on the question and correct answer
theorem remainder_when_divided_by_7 {k : ℕ} (h : k_condition k) : k % 7 = 3 :=
sorry

end remainder_when_divided_by_7_l33_33695


namespace greatest_value_of_a_greatest_value_of_a_achieved_l33_33854

theorem greatest_value_of_a (a b : ℕ) (h1 : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) : a ≤ 20 :=
sorry

theorem greatest_value_of_a_achieved (a b : ℕ) (h1 : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120)
  (h2 : Nat.gcd a b = 10) (h3 : 10 ∣ a ∧ 10 ∣ b) (h4 : Nat.lcm a b = 20) : a = 20 :=
sorry

end greatest_value_of_a_greatest_value_of_a_achieved_l33_33854


namespace profit_per_metre_l33_33879

/-- 
Given:
1. A trader sells 85 meters of cloth for Rs. 8925.
2. The cost price of one metre of cloth is Rs. 95.

Prove:
The profit per metre of cloth is Rs. 10.
-/
theorem profit_per_metre 
  (SP : ℕ) (CP : ℕ)
  (total_SP : SP = 8925)
  (total_meters : ℕ := 85)
  (cost_per_meter : CP = 95) :
  (SP - total_meters * CP) / total_meters = 10 :=
by
  sorry

end profit_per_metre_l33_33879


namespace sharon_trip_distance_l33_33400

theorem sharon_trip_distance
  (h1 : ∀ (d : ℝ), (180 * d) = 1 ∨ (d = 0))  -- Any distance traveled in 180 minutes follows 180d=1 (usual speed)
  (h2 : ∀ (d : ℝ), (276 * (d - 20 / 60)) = 1 ∨ (d = 0))  -- With reduction in speed due to snowstorm too follows a similar relation
  (h3: ∀ (total_time : ℝ), total_time = 276 ∨ total_time = 0)  -- Total time is 276 minutes
  : ∃ (x : ℝ), x = 135 := sorry

end sharon_trip_distance_l33_33400


namespace prob_A_fee_exactly_6_yuan_prob_sum_fees_A_B_36_yuan_l33_33032

section ParkingProblem

variable (P_A_more_1_no_more_2 : ℚ) (P_A_more_than_14 : ℚ)

theorem prob_A_fee_exactly_6_yuan :
  (P_A_more_1_no_more_2 = 1/3) →
  (P_A_more_than_14 = 5/12) →
  (1 - (P_A_more_1_no_more_2 + P_A_more_than_14)) = 1/4 :=
by
  -- Skipping the proof
  intros _ _
  sorry

theorem prob_sum_fees_A_B_36_yuan :
  (1/4 : ℚ) = 1/4 :=
by
  -- Skipping the proof
  exact rfl

end ParkingProblem

end prob_A_fee_exactly_6_yuan_prob_sum_fees_A_B_36_yuan_l33_33032


namespace base_length_of_isosceles_triangle_triangle_l33_33307

section Geometry

variable {b m x : ℝ}

-- Define the conditions
def isosceles_triangle (b : ℝ) : Prop :=
∀ {A B C : ℝ}, A = b ∧ B = b -- representing an isosceles triangle with two equal sides

def segment_length (m : ℝ) : Prop :=
∀ {D E : ℝ}, D - E = m -- the segment length between points where bisectors intersect sides is m

-- The theorem we want to prove
theorem base_length_of_isosceles_triangle_triangle (h1 : isosceles_triangle b) (h2 : segment_length m) : x = b * m / (b - m) :=
sorry

end Geometry

end base_length_of_isosceles_triangle_triangle_l33_33307


namespace comedies_in_terms_of_a_l33_33653

variable (T a : ℝ)
variables (Comedies Dramas Action : ℝ)
axiom Condition1 : Comedies = 0.64 * T
axiom Condition2 : Dramas = 5 * a
axiom Condition3 : Action = a
axiom Condition4 : Comedies + Dramas + Action = T

theorem comedies_in_terms_of_a : Comedies = 10.67 * a :=
by sorry

end comedies_in_terms_of_a_l33_33653


namespace businessmen_drink_one_type_l33_33318

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

end businessmen_drink_one_type_l33_33318


namespace complex_number_solution_l33_33524

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i * i = -1) (hz : z * (i - 1) = 2 * i) : 
z = 1 - i :=
by 
  sorry

end complex_number_solution_l33_33524


namespace composite_sum_of_squares_l33_33245

theorem composite_sum_of_squares (a b : ℤ) (h_roots : ∃ x1 x2 : ℕ, (x1 + x2 : ℤ) = -a ∧ (x1 * x2 : ℤ) = b + 1) :
  ∃ m n : ℕ, a^2 + b^2 = m * n ∧ 1 < m ∧ 1 < n :=
sorry

end composite_sum_of_squares_l33_33245


namespace probability_exactly_three_primes_l33_33750

noncomputable def prime_faces : Finset ℕ := {2, 3, 5, 7, 11}

def num_faces : ℕ := 12
def num_dice : ℕ := 7
def target_primes : ℕ := 3

def probability_three_primes : ℚ :=
  35 * ((5 / 12)^3 * (7 / 12)^4)

theorem probability_exactly_three_primes :
  probability_three_primes = (4375 / 51821766) :=
by
  sorry

end probability_exactly_three_primes_l33_33750


namespace ceil_add_eq_double_of_int_l33_33677

theorem ceil_add_eq_double_of_int {x : ℤ} (h : ⌈(x : ℝ)⌉ + ⌊(x : ℝ)⌋ = 2 * (x : ℝ)) : ⌈(x : ℝ)⌉ + x = 2 * x :=
by
  sorry

end ceil_add_eq_double_of_int_l33_33677


namespace find_some_number_l33_33681

def some_number (x : Int) (some_num : Int) : Prop :=
  (3 < x ∧ x < 10) ∧
  (5 < x ∧ x < 18) ∧
  (9 > x ∧ x > -2) ∧
  (8 > x ∧ x > 0) ∧
  (x + some_num < 9)

theorem find_some_number :
  ∀ (some_num : Int), some_number 7 some_num → some_num < 2 :=
by
  intros some_num H
  sorry

end find_some_number_l33_33681


namespace z_in_third_quadrant_l33_33224

def i := Complex.I

def z := i + 2 * (i^2) + 3 * (i^3)

theorem z_in_third_quadrant : 
    let z_real := Complex.re z
    let z_imag := Complex.im z
    z_real < 0 ∧ z_imag < 0 :=
by
  sorry

end z_in_third_quadrant_l33_33224


namespace pears_left_l33_33607

theorem pears_left (jason_pears : ℕ) (keith_pears : ℕ) (mike_ate : ℕ) 
  (h1 : jason_pears = 46) 
  (h2 : keith_pears = 47) 
  (h3 : mike_ate = 12) : 
  jason_pears + keith_pears - mike_ate = 81 := 
by 
  sorry

end pears_left_l33_33607


namespace smallest_a1_l33_33999

noncomputable def a_seq (a : ℕ → ℝ) : Prop :=
∀ n > 1, a n = 13 * a (n - 1) - 2 * n

noncomputable def positive_sequence (a : ℕ → ℝ) : Prop :=
∀ i, a i > 0

theorem smallest_a1 : ∃ a : ℕ → ℝ, a_seq a ∧ positive_sequence a ∧ a 1 = 13 / 36 :=
by
  sorry

end smallest_a1_l33_33999


namespace exponential_inequality_l33_33881

variable (a b : ℝ)

theorem exponential_inequality (h : -1 < a ∧ a < b ∧ b < 1) : Real.exp a < Real.exp b :=
by
  sorry

end exponential_inequality_l33_33881


namespace even_function_inequality_l33_33281

variable {α : Type*} [LinearOrderedField α]

def is_even_function (f : α → α) : Prop := ∀ x, f x = f (-x)

-- The hypothesis and the assertion in Lean
theorem even_function_inequality
  (f : α → α)
  (h_even : is_even_function f)
  (h3_gt_1 : f 3 > f 1)
  : f (-1) < f 3 :=
sorry

end even_function_inequality_l33_33281


namespace part_I_part_II_l33_33185

noncomputable def f (x : ℝ) := Real.cos (x + Real.pi / 4)

-- Part I
theorem part_I : f (Real.pi / 6) + f (-Real.pi / 6) = Real.sqrt 6 / 2 :=
by
  sorry

-- Part II
theorem part_II (x : ℝ) (h : f x = Real.sqrt 2 / 3) : Real.sin (2 * x) = 5 / 9 :=
by
  sorry

end part_I_part_II_l33_33185


namespace find_numbers_l33_33311

variables {x y : ℤ}

theorem find_numbers (x y : ℤ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 185) (h3 : (x - y)^2 = 121) :
  (x = 13 ∧ y = 2) ∨ (x = -5 ∧ y = -16) :=
sorry

end find_numbers_l33_33311


namespace powers_of_2_not_powers_of_4_below_1000000_equals_10_l33_33038

def num_powers_of_2_not_4 (n : ℕ) : ℕ :=
  let powers_of_2 := (List.range n).filter (fun k => (2^k < 1000000));
  let powers_of_4 := (List.range n).filter (fun k => (4^k < 1000000));
  powers_of_2.length - powers_of_4.length

theorem powers_of_2_not_powers_of_4_below_1000000_equals_10 : 
  num_powers_of_2_not_4 20 = 10 :=
by
  sorry

end powers_of_2_not_powers_of_4_below_1000000_equals_10_l33_33038


namespace average_monthly_balance_l33_33664

theorem average_monthly_balance
  (jan feb mar apr may : ℕ) 
  (Hjan : jan = 200)
  (Hfeb : feb = 300)
  (Hmar : mar = 100)
  (Hapr : apr = 250)
  (Hmay : may = 150) :
  (jan + feb + mar + apr + may) / 5 = 200 := 
  by
  sorry

end average_monthly_balance_l33_33664


namespace quadratic_has_two_zeros_l33_33506

theorem quadratic_has_two_zeros {a b c : ℝ} (h : a * c < 0) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
by
  sorry

end quadratic_has_two_zeros_l33_33506


namespace smallest_number_remainder_problem_l33_33945

theorem smallest_number_remainder_problem :
  ∃ N : ℕ, (N % 13 = 2) ∧ (N % 15 = 4) ∧ (∀ n : ℕ, (n % 13 = 2 ∧ n % 15 = 4) → n ≥ N) :=
sorry

end smallest_number_remainder_problem_l33_33945


namespace ganesh_speed_x_to_y_l33_33424

-- Define the conditions
variables (D : ℝ) (V : ℝ)

-- Theorem statement: Prove that Ganesh's average speed from x to y is 44 km/hr
theorem ganesh_speed_x_to_y
  (H1 : 39.6 = 2 * D / (D / V + D / 36))
  (H2 : V = 44) :
  true :=
sorry

end ganesh_speed_x_to_y_l33_33424


namespace not_algorithm_is_C_l33_33809

-- Definitions based on the conditions recognized in a)
def option_A := "To go from Zhongshan to Beijing, first take a bus, then take a train."
def option_B := "The steps to solve a linear equation are to eliminate the denominator, remove the brackets, transpose terms, combine like terms, and make the coefficient 1."
def option_C := "The equation x^2 - 4x + 3 = 0 has two distinct real roots."
def option_D := "When solving the inequality ax + 3 > 0, the first step is to transpose terms, and the second step is to discuss the sign of a."

-- Problem statement
theorem not_algorithm_is_C : 
  (option_C ≠ "algorithm for solving a problem") ∧ 
  (option_A = "algorithm for solving a problem") ∧ 
  (option_B = "algorithm for solving a problem") ∧ 
  (option_D = "algorithm for solving a problem") :=
  by 
  sorry

end not_algorithm_is_C_l33_33809


namespace find_other_endpoint_l33_33931

theorem find_other_endpoint (x_m y_m x_1 y_1 x_2 y_2 : ℝ) 
  (h_mid_x : x_m = (x_1 + x_2) / 2)
  (h_mid_y : y_m = (y_1 + y_2) / 2)
  (h_x_m : x_m = 3)
  (h_y_m : y_m = 4)
  (h_x_1 : x_1 = 0)
  (h_y_1 : y_1 = -1) :
  (x_2, y_2) = (6, 9) :=
sorry

end find_other_endpoint_l33_33931


namespace acetone_C_mass_percentage_l33_33844

noncomputable def mass_percentage_C_in_acetone : ℝ :=
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_acetone := (3 * atomic_mass_C) + (6 * atomic_mass_H) + (1 * atomic_mass_O)
  let total_mass_C := 3 * atomic_mass_C
  (total_mass_C / molar_mass_acetone) * 100

theorem acetone_C_mass_percentage :
  abs (mass_percentage_C_in_acetone - 62.01) < 0.01 := by
  sorry

end acetone_C_mass_percentage_l33_33844


namespace evaluate_expression_l33_33558

open Complex

theorem evaluate_expression (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + a * b + b^2 = 0) :
  (a^6 + b^6) / (a + b)^6 = 18 :=
by
  sorry

end evaluate_expression_l33_33558


namespace solve_for_n_l33_33624

theorem solve_for_n (n : ℕ) (h : (8 ^ n) * (8 ^ n) * (8 ^ n) = 64 ^ 3) : n = 2 :=
by sorry

end solve_for_n_l33_33624


namespace count_even_three_digit_numbers_less_than_800_l33_33448

def even_three_digit_numbers_less_than_800 : Nat :=
  let hundreds_choices := 7
  let tens_choices := 8
  let units_choices := 4
  hundreds_choices * tens_choices * units_choices

theorem count_even_three_digit_numbers_less_than_800 :
  even_three_digit_numbers_less_than_800 = 224 := 
by 
  unfold even_three_digit_numbers_less_than_800
  rfl

end count_even_three_digit_numbers_less_than_800_l33_33448


namespace volleyball_not_basketball_l33_33534

def class_size : ℕ := 40
def basketball_enjoyers : ℕ := 15
def volleyball_enjoyers : ℕ := 20
def neither_sport : ℕ := 10

theorem volleyball_not_basketball :
  (volleyball_enjoyers - (basketball_enjoyers + volleyball_enjoyers - (class_size - neither_sport))) = 15 :=
by
  sorry

end volleyball_not_basketball_l33_33534


namespace line_l_prime_eq_2x_minus_3y_plus_5_l33_33617

theorem line_l_prime_eq_2x_minus_3y_plus_5 (m : ℝ) (x y : ℝ) : 
  (2 * m + 1) * x + (m + 1) * y + m = 0 →
  (2 * -1 + 1) * (-1) + (1 + 1) * 1 + m = 0 →
  ∀ a b : ℝ, (3 * b, 2 * b) = (3 * 1, 2 * 1) → (a, b) = (-1, 1) → 
  2 * x - 3 * y + 5 = 0 :=
by
  intro h1 h2 a b h3 h4
  sorry

end line_l_prime_eq_2x_minus_3y_plus_5_l33_33617


namespace real_solutions_x_inequality_l33_33054

theorem real_solutions_x_inequality (x : ℝ) :
  (∃ y : ℝ, y^2 + 6 * x * y + x + 8 = 0) ↔ (x ≤ -8 / 9 ∨ x ≥ 1) := 
sorry

end real_solutions_x_inequality_l33_33054


namespace polynomial_R_result_l33_33678

noncomputable def polynomial_Q_R (z : ℤ) : Prop :=
  ∃ Q R : Polynomial ℂ, 
  z ^ 2020 + 1 = (z ^ 2 - z + 1) * Q + R ∧ R.degree < 2 ∧ R = 2

theorem polynomial_R_result :
  polynomial_Q_R z :=
by 
  sorry

end polynomial_R_result_l33_33678


namespace compare_b_d_l33_33251

noncomputable def percentage_increase (x : ℝ) (p : ℝ) := x * (1 + p)
noncomputable def percentage_decrease (x : ℝ) (p : ℝ) := x * (1 - p)

theorem compare_b_d (a b c d : ℝ)
  (h1 : 0 < b)
  (h2 : a = percentage_increase b 0.02)
  (h3 : c = percentage_decrease a 0.01)
  (h4 : d = percentage_decrease c 0.01) :
  b > d :=
sorry

end compare_b_d_l33_33251


namespace initial_pigs_l33_33338

theorem initial_pigs (x : ℕ) (h : x + 86 = 150) : x = 64 :=
by
  sorry

end initial_pigs_l33_33338


namespace pears_more_than_apples_l33_33741

theorem pears_more_than_apples (red_apples green_apples pears : ℕ) (h1 : red_apples = 15) (h2 : green_apples = 8) (h3 : pears = 32) : (pears - (red_apples + green_apples) = 9) :=
by
  sorry

end pears_more_than_apples_l33_33741


namespace cost_of_each_pant_l33_33117

theorem cost_of_each_pant (shirts pants : ℕ) (cost_shirt cost_total : ℕ) (cost_pant : ℕ) :
  shirts = 10 ∧ pants = (shirts / 2) ∧ cost_shirt = 6 ∧ cost_total = 100 →
  (shirts * cost_shirt + pants * cost_pant = cost_total) →
  cost_pant = 8 :=
by
  sorry

end cost_of_each_pant_l33_33117


namespace sum_prime_numbers_l33_33410

theorem sum_prime_numbers (a b c : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Prime b) (h3 : Nat.Prime c) (hEqn : a * b * c + a = 851) : 
  a + b + c = 50 :=
sorry

end sum_prime_numbers_l33_33410


namespace distance_traveled_l33_33632

-- Given conditions
def speed : ℕ := 100 -- Speed in km/hr
def time : ℕ := 5    -- Time in hours

-- The goal is to prove the distance traveled is 500 km
theorem distance_traveled : speed * time = 500 := by
  -- we state the proof goal
  sorry

end distance_traveled_l33_33632


namespace initial_games_l33_33797

theorem initial_games (X : ℕ) (h1 : X - 68 + 47 = 74) : X = 95 :=
by
  sorry

end initial_games_l33_33797


namespace pete_ran_least_distance_l33_33144

theorem pete_ran_least_distance
  (phil_distance : ℕ := 4)
  (tom_distance : ℕ := 6)
  (pete_distance : ℕ := 2)
  (amal_distance : ℕ := 8)
  (sanjay_distance : ℕ := 7) :
  pete_distance ≤ phil_distance ∧
  pete_distance ≤ tom_distance ∧
  pete_distance ≤ amal_distance ∧
  pete_distance ≤ sanjay_distance :=
by {
  sorry
}

end pete_ran_least_distance_l33_33144


namespace tan_lt_neg_one_implies_range_l33_33530

theorem tan_lt_neg_one_implies_range {x : ℝ} (h1 : 0 < x) (h2 : x < π) (h3 : Real.tan x < -1) :
  (π / 2 < x) ∧ (x < 3 * π / 4) :=
sorry

end tan_lt_neg_one_implies_range_l33_33530


namespace div_identity_l33_33356

theorem div_identity :
  let a := 6 / 2
  let b := a * 3
  120 / b = 120 / 9 :=
by
  sorry

end div_identity_l33_33356


namespace problem_statement_l33_33131

/-
If x is equal to the sum of the even integers from 40 to 60 inclusive,
y is the number of even integers from 40 to 60 inclusive,
and z is the sum of the odd integers from 41 to 59 inclusive,
prove that x + y + z = 1061.
-/
theorem problem_statement :
  let x := (11 / 2) * (40 + 60)
  let y := 11
  let z := (10 / 2) * (41 + 59)
  x + y + z = 1061 :=
by
  sorry

end problem_statement_l33_33131


namespace num_ordered_pairs_solutions_l33_33896

theorem num_ordered_pairs_solutions :
  ∃ (n : ℕ), n = 18 ∧
    (∀ (a b : ℝ), (∃ x y : ℤ , a * (x : ℝ) + b * (y : ℝ) = 1 ∧ (x * x + y * y = 50))) :=
sorry

end num_ordered_pairs_solutions_l33_33896


namespace basketball_price_l33_33914

variable (P : ℝ)

def coachA_cost : ℝ := 10 * P
def coachB_baseball_cost : ℝ := 14 * 2.5
def coachB_bat_cost : ℝ := 18
def coachB_total_cost : ℝ := coachB_baseball_cost + coachB_bat_cost
def coachA_excess_cost : ℝ := 237

theorem basketball_price (h : coachA_cost P = coachB_total_cost + coachA_excess_cost) : P = 29 :=
by
  sorry

end basketball_price_l33_33914


namespace bella_eats_six_apples_a_day_l33_33257

variable (A : ℕ) -- Number of apples Bella eats per day
variable (G : ℕ) -- Total number of apples Grace picks in 6 weeks
variable (B : ℕ) -- Total number of apples Bella eats in 6 weeks

-- Definitions for the conditions 
def condition1 := B = 42 * A
def condition2 := B = (1 / 3) * G
def condition3 := (2 / 3) * G = 504

-- Final statement that needs to be proved
theorem bella_eats_six_apples_a_day (A G B : ℕ) 
  (h1 : condition1 A B) 
  (h2 : condition2 G B) 
  (h3 : condition3 G) 
  : A = 6 := by sorry

end bella_eats_six_apples_a_day_l33_33257


namespace minimum_races_to_find_top3_l33_33452

-- Define a constant to represent the number of horses and maximum horses per race
def total_horses : ℕ := 25
def max_horses_per_race : ℕ := 5

-- Define the problem statement as a theorem
theorem minimum_races_to_find_top3 (total_horses : ℕ) (max_horses_per_race : ℕ) : ℕ :=
  if total_horses = 25 ∧ max_horses_per_race = 5 then 7 else sorry

end minimum_races_to_find_top3_l33_33452


namespace blue_pens_removed_l33_33419

def initial_blue_pens := 9
def initial_black_pens := 21
def initial_red_pens := 6
def removed_black_pens := 7
def pens_left := 25

theorem blue_pens_removed (x : ℕ) :
  initial_blue_pens - x + (initial_black_pens - removed_black_pens) + initial_red_pens = pens_left ↔ x = 4 := 
by 
  sorry

end blue_pens_removed_l33_33419


namespace king_total_payment_l33_33267

theorem king_total_payment
  (crown_cost : ℕ)
  (architect_cost : ℕ)
  (chef_cost : ℕ)
  (crown_tip_percent : ℕ)
  (architect_tip_percent : ℕ)
  (chef_tip_percent : ℕ)
  (crown_tip : ℕ)
  (architect_tip : ℕ)
  (chef_tip : ℕ)
  (total_crown_cost : ℕ)
  (total_architect_cost : ℕ)
  (total_chef_cost : ℕ)
  (total_paid : ℕ) :
  crown_cost = 20000 →
  architect_cost = 50000 →
  chef_cost = 10000 →
  crown_tip_percent = 10 →
  architect_tip_percent = 5 →
  chef_tip_percent = 15 →
  crown_tip = crown_cost * crown_tip_percent / 100 →
  architect_tip = architect_cost * architect_tip_percent / 100 →
  chef_tip = chef_cost * chef_tip_percent / 100 →
  total_crown_cost = crown_cost + crown_tip →
  total_architect_cost = architect_cost + architect_tip →
  total_chef_cost = chef_cost + chef_tip →
  total_paid = total_crown_cost + total_architect_cost + total_chef_cost →
  total_paid = 86000 := by
  sorry

end king_total_payment_l33_33267


namespace degree_to_radian_conversion_l33_33379

theorem degree_to_radian_conversion : (1440 * (Real.pi / 180) = 8 * Real.pi) := 
by
  sorry

end degree_to_radian_conversion_l33_33379


namespace Malcom_cards_after_giving_away_half_l33_33955

def Brandon_cards : ℕ := 20
def Malcom_initial_cards : ℕ := Brandon_cards + 8
def Malcom_remaining_cards : ℕ := Malcom_initial_cards - (Malcom_initial_cards / 2)

theorem Malcom_cards_after_giving_away_half :
  Malcom_remaining_cards = 14 :=
by
  sorry

end Malcom_cards_after_giving_away_half_l33_33955


namespace lake_circumference_ratio_l33_33459

theorem lake_circumference_ratio 
    (D C : ℝ) 
    (hD : D = 100) 
    (hC : C = 314) : 
    C / D = 3.14 := 
sorry

end lake_circumference_ratio_l33_33459


namespace percentage_increase_in_price_l33_33385

theorem percentage_increase_in_price (initial_price : ℝ) (total_cost : ℝ) (num_family_members : ℕ) 
  (pounds_per_person : ℝ) (new_price : ℝ) (percentage_increase : ℝ) :
  initial_price = 1.6 → 
  total_cost = 16 → 
  num_family_members = 4 → 
  pounds_per_person = 2 → 
  (total_cost / (num_family_members * pounds_per_person)) = new_price → 
  percentage_increase = ((new_price - initial_price) / initial_price) * 100 → 
  percentage_increase = 25 :=
by
  intros h_initial h_total h_members h_pounds h_new_price h_percentage
  sorry

end percentage_increase_in_price_l33_33385


namespace cannot_be_expressed_as_difference_of_squares_can_be_expressed_as_difference_of_squares_2004_can_be_expressed_as_difference_of_squares_2005_can_be_expressed_as_difference_of_squares_2007_l33_33580

theorem cannot_be_expressed_as_difference_of_squares (a b : ℤ) (h : 2006 = a^2 - b^2) : False := sorry

theorem can_be_expressed_as_difference_of_squares_2004 : ∃ (a b : ℤ), 2004 = a^2 - b^2 := by
  use 502, 500
  norm_num

theorem can_be_expressed_as_difference_of_squares_2005 : ∃ (a b : ℤ), 2005 = a^2 - b^2 := by
  use 1003, 1002
  norm_num

theorem can_be_expressed_as_difference_of_squares_2007 : ∃ (a b : ℤ), 2007 = a^2 - b^2 := by
  use 1004, 1003
  norm_num

end cannot_be_expressed_as_difference_of_squares_can_be_expressed_as_difference_of_squares_2004_can_be_expressed_as_difference_of_squares_2005_can_be_expressed_as_difference_of_squares_2007_l33_33580


namespace rowing_speed_in_still_water_l33_33652

variable (v c t : ℝ)
variable (h1 : c = 1.3)
variable (h2 : 2 * ((v - c) * t) = ((v + c) * t))

theorem rowing_speed_in_still_water : v = 3.9 := by
  sorry

end rowing_speed_in_still_water_l33_33652


namespace box_volume_increase_l33_33903

theorem box_volume_increase (l w h : ℝ)
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
  sorry

end box_volume_increase_l33_33903


namespace vertical_asymptote_l33_33432

theorem vertical_asymptote (x : ℝ) : (4 * x + 6 = 0) -> x = -3 / 2 :=
by
  sorry

end vertical_asymptote_l33_33432


namespace correct_flowchart_requirement_l33_33232

def flowchart_requirement (option : String) : Prop := 
  option = "From left to right, from top to bottom" ∨
  option = "From right to left, from top to bottom" ∨
  option = "From left to right, from bottom to top" ∨
  option = "From right to left, from bottom to top"

theorem correct_flowchart_requirement : 
  (∀ option, flowchart_requirement option → option = "From left to right, from top to bottom") :=
by
  sorry

end correct_flowchart_requirement_l33_33232


namespace inscribed_circle_diameter_of_right_triangle_l33_33623

theorem inscribed_circle_diameter_of_right_triangle (a b : ℕ) (hc : a = 8) (hb : b = 15) :
  2 * (60 / (a + b + Int.sqrt (a ^ 2 + b ^ 2))) = 6 :=
by
  sorry

end inscribed_circle_diameter_of_right_triangle_l33_33623


namespace teacherZhangAge_in_5_years_correct_l33_33794

variable (a : ℕ)

def teacherZhangAgeCurrent := 3 * a - 2

def teacherZhangAgeIn5Years := teacherZhangAgeCurrent a + 5

theorem teacherZhangAge_in_5_years_correct :
  teacherZhangAgeIn5Years a = 3 * a + 3 := by
  sorry

end teacherZhangAge_in_5_years_correct_l33_33794


namespace chromosomes_mitosis_late_stage_l33_33968

/-- A biological cell with 24 chromosomes at the late stage of the second meiotic division. -/
def cell_chromosomes_meiosis_late_stage : ℕ := 24

/-- The number of chromosomes in this organism at the late stage of mitosis is double that at the late stage of the second meiotic division. -/
theorem chromosomes_mitosis_late_stage : cell_chromosomes_meiosis_late_stage * 2 = 48 :=
by
  -- We will add the necessary proof here.
  sorry

end chromosomes_mitosis_late_stage_l33_33968


namespace books_fraction_sold_l33_33538

theorem books_fraction_sold (B : ℕ) (h1 : B - 36 * 2 = 144) :
  (B - 36) / B = 2 / 3 := by
  sorry

end books_fraction_sold_l33_33538


namespace mary_keep_warm_hours_l33_33865

-- Definitions based on the conditions
def sticks_from_chairs (chairs : ℕ) : ℕ := chairs * 6
def sticks_from_tables (tables : ℕ) : ℕ := tables * 9
def sticks_from_stools (stools : ℕ) : ℕ := stools * 2
def sticks_needed_per_hour : ℕ := 5

-- Given counts of furniture
def chairs : ℕ := 18
def tables : ℕ := 6
def stools : ℕ := 4

-- Total number of sticks
def total_sticks : ℕ := (sticks_from_chairs chairs) + (sticks_from_tables tables) + (sticks_from_stools stools)

-- Proving the number of hours Mary can keep warm
theorem mary_keep_warm_hours : total_sticks / sticks_needed_per_hour = 34 := by
  sorry

end mary_keep_warm_hours_l33_33865


namespace part_I_part_II_l33_33341

-- Conditions
def p (x m : ℝ) : Prop := x > m → 2 * x - 5 > 0
def q (m : ℝ) : Prop := ∃ x y : ℝ, (x^2 / (m - 1)) + (y^2 / (2 - m)) = 1

-- Statements for proof
theorem part_I (m x : ℝ) (hq: q m) (hp: p x m) : 
  m < 1 ∨ (2 < m ∧ m ≤ 5 / 2) :=
sorry

theorem part_II (m x : ℝ) (hq: ¬ q m ∧ ¬(p x m ∧ q m) ∧ (p x m ∨ q m)) : 
  (1 ≤ m ∧ m ≤ 2) ∨ (m > 5 / 2) :=
sorry

end part_I_part_II_l33_33341


namespace woman_speed_in_still_water_l33_33299

noncomputable def speed_in_still_water (V_c : ℝ) (t : ℝ) (d : ℝ) : ℝ :=
  let V_downstream := d / (t / 3600)
  V_downstream - V_c

theorem woman_speed_in_still_water :
  let V_c := 60
  let t := 9.99920006399488
  let d := 0.5 -- 500 meters converted to kilometers
  speed_in_still_water V_c t d = 120.01800180018 :=
by
  unfold speed_in_still_water
  sorry

end woman_speed_in_still_water_l33_33299


namespace sum_13_gt_0_l33_33784

noncomputable def a_n : ℕ → ℝ := sorry
noncomputable def S_n : ℕ → ℝ := sorry

axiom a7_gt_0 : 0 < a_n 7
axiom a8_lt_0 : a_n 8 < 0

theorem sum_13_gt_0 : S_n 13 > 0 :=
sorry

end sum_13_gt_0_l33_33784


namespace determine_a_l33_33280

theorem determine_a 
(h : ∃x, x = -1 ∧ 2 * x ^ 2 + a * x - a ^ 2 = 0) : a = -2 ∨ a = 1 :=
by
  -- Proof omitted
  sorry

end determine_a_l33_33280


namespace train_cross_bridge_time_l33_33650

theorem train_cross_bridge_time
  (length_train : ℕ) (speed_train_kmph : ℕ) (length_bridge : ℕ) 
  (km_to_m : ℕ) (hour_to_s : ℕ)
  (h1 : length_train = 165) 
  (h2 : speed_train_kmph = 54) 
  (h3 : length_bridge = 720) 
  (h4 : km_to_m = 1000) 
  (h5 : hour_to_s = 3600) 
  : (length_train + length_bridge) / ((speed_train_kmph * km_to_m) / hour_to_s) = 59 := 
sorry

end train_cross_bridge_time_l33_33650


namespace three_digit_non_multiples_of_3_or_11_l33_33746

theorem three_digit_non_multiples_of_3_or_11 : 
  ∃ (n : ℕ), n = 546 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → 
    ¬ (x % 3 = 0 ∨ x % 11 = 0) → 
    n = (900 - (300 + 81 - 27))) := 
by 
  sorry

end three_digit_non_multiples_of_3_or_11_l33_33746


namespace simplify_expression_l33_33190

theorem simplify_expression (a b : ℝ) : -3 * (a - b) + (2 * a - 3 * b) = -a :=
by
  sorry

end simplify_expression_l33_33190


namespace value_of_b_l33_33234

theorem value_of_b (a b : ℕ) (r : ℝ) (h₁ : a = 2020) (h₂ : r = a / b) (h₃ : r = 0.5) : b = 4040 := 
by
  -- Hint: The proof takes steps to transform the conditions using basic algebraic manipulations.
  sorry

end value_of_b_l33_33234


namespace lunks_needed_for_bananas_l33_33284

theorem lunks_needed_for_bananas :
  (7 : ℚ) / 4 * (20 * 3 / 5) = 21 :=
by
  sorry

end lunks_needed_for_bananas_l33_33284


namespace hyperbola_range_of_k_l33_33598

theorem hyperbola_range_of_k (x y k : ℝ) :
  (∃ x y : ℝ, (x^2 / (1 - 2 * k) - y^2 / (k - 2) = 1) ∧ (1 - 2 * k < 0) ∧ (k - 2 < 0)) →
  (1 / 2 < k ∧ k < 2) :=
by 
  sorry

end hyperbola_range_of_k_l33_33598


namespace inverse_proportional_l33_33577

/-- Given that α is inversely proportional to β and α = -3 when β = -6,
    prove that α = 9/4 when β = 8. --/
theorem inverse_proportional (α β : ℚ) 
  (h1 : α * β = 18)
  (h2 : β = 8) : 
  α = 9 / 4 :=
by
  sorry

end inverse_proportional_l33_33577


namespace power_set_card_greater_l33_33917

open Set

variables {A : Type*} (α : ℕ) [Fintype A] (hA : Fintype.card A = α)

theorem power_set_card_greater (h : Fintype.card A = α) :
  2 ^ α > α :=
sorry

end power_set_card_greater_l33_33917


namespace real_solutions_quadratic_l33_33584

theorem real_solutions_quadratic (d : ℝ) (h : 0 < d) :
  ∃ x : ℝ, x^2 - 8 * x + d < 0 ↔ 0 < d ∧ d < 16 :=
by
  sorry

end real_solutions_quadratic_l33_33584


namespace dihedral_angle_is_60_degrees_l33_33635

def point (x y z : ℝ) := (x, y, z)

noncomputable def dihedral_angle (P Q R S T : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem dihedral_angle_is_60_degrees :
  dihedral_angle 
    (point 1 0 0)  -- A
    (point 1 1 0)  -- B
    (point 0 0 0)  -- D
    (point 1 0 1)  -- A₁
    (point 0 0 1)  -- D₁
 = 60 :=
sorry

end dihedral_angle_is_60_degrees_l33_33635


namespace simplify_t_l33_33780

theorem simplify_t (t : ℝ) (cbrt3 : ℝ) (h : cbrt3 ^ 3 = 3) 
  (ht : t = 1 / (1 - cbrt3)) : 
  t = - (1 + cbrt3 + cbrt3 ^ 2) / 2 := 
sorry

end simplify_t_l33_33780


namespace remainder_of_sum_l33_33203

theorem remainder_of_sum (a b c : ℕ) (h₁ : a % 15 = 11) (h₂ : b % 15 = 12) (h₃ : c % 15 = 13) : 
  (a + b + c) % 15 = 6 :=
by 
  sorry

end remainder_of_sum_l33_33203


namespace maria_baggies_count_l33_33323

def total_cookies (chocolate_chip : ℕ) (oatmeal : ℕ) : ℕ :=
  chocolate_chip + oatmeal

def baggies_count (total_cookies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  total_cookies / cookies_per_bag

theorem maria_baggies_count :
  let choco_chip := 2
  let oatmeal := 16
  let cookies_per_bag := 3
  baggies_count (total_cookies choco_chip oatmeal) cookies_per_bag = 6 :=
by
  sorry

end maria_baggies_count_l33_33323


namespace initial_number_2008_l33_33418

theorem initial_number_2008 (x : ℕ) (h : x = 2008 ∨ (∃ y: ℕ, (x = 2*y + 1 ∨ (x = y / (y + 2))))): x = 2008 :=
by
  cases h with
  | inl h2008 => exact h2008
  | inr hexists => cases hexists with
    | intro y hy =>
        cases hy
        case inl h2y => sorry
        case inr hdiv => sorry

end initial_number_2008_l33_33418


namespace marbles_per_box_l33_33126

-- Define the total number of marbles
def total_marbles : Nat := 18

-- Define the number of boxes
def number_of_boxes : Nat := 3

-- Prove there are 6 marbles in each box
theorem marbles_per_box : total_marbles / number_of_boxes = 6 := by
  sorry

end marbles_per_box_l33_33126


namespace A_is_11_years_older_than_B_l33_33161

-- Define the constant B as given in the problem
def B : ℕ := 41

-- Define the condition based on the problem statement
def condition (A : ℕ) := A + 10 = 2 * (B - 10)

-- Prove the main statement that A is 11 years older than B
theorem A_is_11_years_older_than_B (A : ℕ) (h : condition A) : A - B = 11 :=
by
  sorry

end A_is_11_years_older_than_B_l33_33161


namespace latte_cost_l33_33613

theorem latte_cost (L : ℝ) 
  (latte_days : ℝ := 5)
  (iced_coffee_cost : ℝ := 2)
  (iced_coffee_days : ℝ := 3)
  (weeks_in_year : ℝ := 52)
  (spending_reduction : ℝ := 0.25)
  (savings : ℝ := 338) 
  (current_annual_spending : ℝ := 4 * savings)
  (weekly_spending : ℝ := latte_days * L + iced_coffee_days * iced_coffee_cost)
  (annual_spending_eq : weeks_in_year * weekly_spending = current_annual_spending) :
  L = 4 := 
sorry

end latte_cost_l33_33613


namespace path_area_and_cost_l33_33025

-- Define the initial conditions
def field_length : ℝ := 65
def field_width : ℝ := 55
def path_width : ℝ := 2.5
def cost_per_sq_m : ℝ := 2

-- Define the extended dimensions including the path
def extended_length := field_length + 2 * path_width
def extended_width := field_width + 2 * path_width

-- Define the areas
def area_with_path := extended_length * extended_width
def area_of_field := field_length * field_width
def area_of_path := area_with_path - area_of_field

-- Define the cost
def cost_of_constructing_path := area_of_path * cost_per_sq_m

theorem path_area_and_cost :
  area_of_path = 625 ∧ cost_of_constructing_path = 1250 :=
by
  sorry

end path_area_and_cost_l33_33025


namespace expenditure_ratio_l33_33236

theorem expenditure_ratio (I_A I_B E_A E_B : ℝ) (h1 : I_A / I_B = 5 / 6)
  (h2 : I_B = 7200) (h3 : 1800 = I_A - E_A) (h4 : 1600 = I_B - E_B) :
  E_A / E_B = 3 / 4 :=
sorry

end expenditure_ratio_l33_33236


namespace find_first_parrot_weight_l33_33154

def cats_weights := [7, 10, 13, 15]
def cats_sum := List.sum cats_weights
def dog1 := cats_sum - 2
def dog2 := cats_sum + 7
def dog3 := (dog1 + dog2) / 2
def dogs_sum := dog1 + dog2 + dog3
def total_parrots_weight := 2 / 3 * dogs_sum

noncomputable def parrot1 := 2 / 5 * total_parrots_weight
noncomputable def parrot2 := 3 / 5 * total_parrots_weight

theorem find_first_parrot_weight : parrot1 = 38 :=
by
  sorry

end find_first_parrot_weight_l33_33154


namespace han_xin_troop_min_soldiers_l33_33643

theorem han_xin_troop_min_soldiers (n : ℕ) : 
  (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 7 = 4) → n = 53 :=
  sorry

end han_xin_troop_min_soldiers_l33_33643


namespace fred_weekend_earnings_l33_33350

noncomputable def fred_initial_dollars : ℕ := 19
noncomputable def fred_final_dollars : ℕ := 40

theorem fred_weekend_earnings :
  fred_final_dollars - fred_initial_dollars = 21 :=
by
  sorry

end fred_weekend_earnings_l33_33350


namespace largest_digit_change_l33_33694

-- Definitions
def initial_number : ℝ := 0.12345

def change_digit (k : Fin 5) : ℝ :=
  match k with
  | 0 => 0.92345
  | 1 => 0.19345
  | 2 => 0.12945
  | 3 => 0.12395
  | 4 => 0.12349

theorem largest_digit_change :
  ∀ k : Fin 5, k ≠ 0 → change_digit 0 > change_digit k :=
by
  intros k hk
  sorry

end largest_digit_change_l33_33694


namespace tan_sum_pi_div_12_l33_33710

theorem tan_sum_pi_div_12 (h1 : Real.tan (Real.pi / 12) ≠ 0) (h2 : Real.tan (5 * Real.pi / 12) ≠ 0) :
  Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 := 
by
  sorry

end tan_sum_pi_div_12_l33_33710


namespace julie_bought_boxes_l33_33531

-- Definitions for the conditions
def packages_per_box := 5
def sheets_per_package := 250
def sheets_per_newspaper := 25
def newspapers := 100

-- Calculations based on conditions
def total_sheets_needed := newspapers * sheets_per_newspaper
def sheets_per_box := packages_per_box * sheets_per_package

-- The goal: to prove that the number of boxes of paper Julie bought is 2
theorem julie_bought_boxes : total_sheets_needed / sheets_per_box = 2 :=
  by
    sorry

end julie_bought_boxes_l33_33531


namespace no_solution_equation_l33_33108

theorem no_solution_equation (m : ℝ) : 
  ¬∃ x : ℝ, x ≠ 2 ∧ (x - 3) / (x - 2) = m / (2 - x) → m = 1 := 
by 
  sorry

end no_solution_equation_l33_33108


namespace line_intersects_hyperbola_left_branch_l33_33691

noncomputable def problem_statement (k : ℝ) : Prop :=
  ∀ (x y : ℝ), y = k * x - 1 ∧ x^2 - y^2 = 1 ∧ y < 0 → 
  k ∈ Set.Ioo (-Real.sqrt 2) (-1)

theorem line_intersects_hyperbola_left_branch (k : ℝ) :
  problem_statement k :=
by
  sorry

end line_intersects_hyperbola_left_branch_l33_33691


namespace range_of_a_l33_33498

theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, a * x^2 + a * x + 1 > 0) : a ∈ Set.Icc 0 4 :=
sorry

end range_of_a_l33_33498


namespace factorization_correct_l33_33121

theorem factorization_correct (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end factorization_correct_l33_33121


namespace transportation_trucks_l33_33918

theorem transportation_trucks (boxes : ℕ) (total_weight : ℕ) (box_weight : ℕ) (truck_capacity : ℕ) :
  (total_weight = 10) → (∀ (b : ℕ), b ≤ boxes → box_weight ≤ 1) → (truck_capacity = 3) → 
  ∃ (trucks : ℕ), trucks = 5 :=
by
  sorry

end transportation_trucks_l33_33918


namespace height_of_parallelogram_l33_33084

theorem height_of_parallelogram (A B H : ℕ) (hA : A = 308) (hB : B = 22) (h_eq : H = A / B) : H = 14 := 
by sorry

end height_of_parallelogram_l33_33084


namespace angle_ACD_l33_33080

theorem angle_ACD (E : ℝ) (arc_eq : ∀ (AB BC CD : ℝ), AB = BC ∧ BC = CD) (angle_eq : E = 40) : ∃ (ACD : ℝ), ACD = 15 :=
by
  sorry

end angle_ACD_l33_33080


namespace remainder_24_l33_33540

-- Statement of the problem in Lean 4
theorem remainder_24 (y : ℤ) (h : y % 288 = 45) : y % 24 = 21 :=
by
  sorry

end remainder_24_l33_33540


namespace largest_possible_package_size_l33_33959

theorem largest_possible_package_size :
  ∃ (p : ℕ), Nat.gcd 60 36 = p ∧ p = 12 :=
by
  use 12
  sorry -- The proof is skipped as per instructions

end largest_possible_package_size_l33_33959


namespace max_load_per_truck_l33_33846

-- Definitions based on given conditions
def num_trucks : ℕ := 3
def total_boxes : ℕ := 240
def lighter_box_weight : ℕ := 10
def heavier_box_weight : ℕ := 40

-- Proof problem statement
theorem max_load_per_truck :
  (total_boxes / 2) * lighter_box_weight + (total_boxes / 2) * heavier_box_weight = 6000 →
  6000 / num_trucks = 2000 :=
by sorry

end max_load_per_truck_l33_33846


namespace range_of_k_l33_33799

theorem range_of_k (k : ℝ) : (2 > 0) ∧ (k > 0) ∧ (k < 2) ↔ (0 < k ∧ k < 2) :=
by
  sorry

end range_of_k_l33_33799


namespace facemasks_per_box_l33_33218

theorem facemasks_per_box (x : ℝ) :
  (3 * x * 0.50) - 15 = 15 → x = 20 :=
by
  intros h
  sorry

end facemasks_per_box_l33_33218


namespace solve_for_n_l33_33680

theorem solve_for_n (n : ℕ) : 
  9^n * 9^n * 9^(2*n) = 81^4 → n = 2 :=
by
  sorry

end solve_for_n_l33_33680


namespace total_volume_of_cubes_l33_33291

theorem total_volume_of_cubes 
  (Carl_cubes : ℕ)
  (Carl_side_length : ℕ)
  (Kate_cubes : ℕ)
  (Kate_side_length : ℕ)
  (h1 : Carl_cubes = 8)
  (h2 : Carl_side_length = 2)
  (h3 : Kate_cubes = 3)
  (h4 : Kate_side_length = 3) :
  Carl_cubes * Carl_side_length ^ 3 + Kate_cubes * Kate_side_length ^ 3 = 145 :=
by
  sorry

end total_volume_of_cubes_l33_33291


namespace integer_solutions_count_l33_33909

theorem integer_solutions_count :
  (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 15 ∧
    ∀ (pair : ℕ × ℕ), pair ∈ pairs ↔ (∃ x y, pair = (x, y) ∧ (Nat.sqrt x + Nat.sqrt y = 14))) :=
by
  sorry

end integer_solutions_count_l33_33909


namespace edward_books_bought_l33_33171

def money_spent : ℕ := 6
def cost_per_book : ℕ := 3

theorem edward_books_bought : money_spent / cost_per_book = 2 :=
by
  sorry

end edward_books_bought_l33_33171


namespace sum_of_fractions_l33_33344

theorem sum_of_fractions :
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (5 / 10) + (6 / 10) + (7 / 10) + (8 / 10) + (10 / 10) + (60 / 10) = 10.6 := by
  sorry

end sum_of_fractions_l33_33344


namespace probability_heads_and_3_l33_33684

noncomputable def biased_coin_heads_prob : ℝ := 0.4
def die_sides : ℕ := 8

theorem probability_heads_and_3 : biased_coin_heads_prob * (1 / die_sides) = 0.05 := sorry

end probability_heads_and_3_l33_33684


namespace sum_of_digits_succ_2080_l33_33470

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_succ_2080 (m : ℕ) (h : sum_of_digits m = 2080) :
  sum_of_digits (m + 1) = 2081 ∨ sum_of_digits (m + 1) = 2090 :=
sorry

end sum_of_digits_succ_2080_l33_33470


namespace real_solutions_eq_l33_33148

def satisfies_equations (x y : ℝ) : Prop :=
  (4 * x + 5 * y = 13) ∧ (2 * x - 3 * y = 1)

theorem real_solutions_eq {x y : ℝ} : satisfies_equations x y ↔ (x = 2 ∧ y = 1) :=
by sorry

end real_solutions_eq_l33_33148


namespace perpendicular_lines_a_eq_3_l33_33412

theorem perpendicular_lines_a_eq_3 (a : ℝ) :
  let l₁ := (a + 1) * x + 2 * y + 6
  let l₂ := x + (a - 5) * y + a^2 - 1
  (a ≠ 5 → -((a + 1) / 2) * (1 / (5 - a)) = -1) → a = 3 := by
  intro l₁ l₂ h
  sorry

end perpendicular_lines_a_eq_3_l33_33412


namespace divisor_of_12401_76_13_l33_33024

theorem divisor_of_12401_76_13 (D : ℕ) (h1: 12401 = (D * 76) + 13) : D = 163 :=
sorry

end divisor_of_12401_76_13_l33_33024


namespace parallel_line_equation_l33_33026

theorem parallel_line_equation :
  ∃ (c : ℝ), 
    (∀ x : ℝ, y = (3 / 4) * x + 6 → (y = (3 / 4) * x + c → abs (c - 6) = 4 * (5 / 4))) → c = 1 :=
by
  sorry

end parallel_line_equation_l33_33026


namespace radius_of_semicircle_l33_33905

theorem radius_of_semicircle (P : ℝ) (π_val : ℝ) (h1 : P = 162) (h2 : π_val = Real.pi) : 
  ∃ r : ℝ, r = 162 / (π + 2) :=
by
  use 162 / (Real.pi + 2)
  sorry

end radius_of_semicircle_l33_33905


namespace cargo_loaded_in_bahamas_l33_33818

def initial : ℕ := 5973
def final : ℕ := 14696
def loaded : ℕ := final - initial

theorem cargo_loaded_in_bahamas : loaded = 8723 := by
  sorry

end cargo_loaded_in_bahamas_l33_33818


namespace instrument_costs_purchasing_plans_l33_33800

variable (x y : ℕ)
variable (a b : ℕ)

theorem instrument_costs : 
  (2 * x + 3 * y = 1700 ∧ 3 * x + y = 1500) →
  x = 400 ∧ y = 300 := 
by 
  intros h
  sorry

theorem purchasing_plans :
  (x = 400) → (y = 300) → (3 * a + 10 = b) →
  (400 * a + 300 * b ≤ 30000) →
  ((760 - 400) * a + (540 - 300) * b ≥ 21600) →
  (a = 18 ∧ b = 64 ∨ a = 19 ∧ b = 67 ∨ a = 20 ∧ b = 70) :=
by
  intros hx hy hab hcost hprofit
  sorry

end instrument_costs_purchasing_plans_l33_33800


namespace minimum_cuts_l33_33260

theorem minimum_cuts (n : Nat) : n >= 50 :=
by
  sorry

end minimum_cuts_l33_33260


namespace intersection_point_parabola_l33_33878

theorem intersection_point_parabola :
  ∃ k : ℝ, (∀ x : ℝ, (3 * (x - 4)^2 + k = 0 ↔ x = 2 ∨ x = 6)) :=
by
  sorry

end intersection_point_parabola_l33_33878


namespace minimum_value_y_l33_33357

theorem minimum_value_y (x : ℝ) (hx : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ ∀ z, (z = x + 4 / (x - 2) → z ≥ 6) :=
by
  sorry

end minimum_value_y_l33_33357


namespace estimate_flight_time_around_earth_l33_33194

theorem estimate_flight_time_around_earth 
  (radius : ℝ) 
  (speed : ℝ)
  (h_radius : radius = 6000) 
  (h_speed : speed = 600) 
  : abs (20 * Real.pi - 63) < 1 :=
by
  sorry

end estimate_flight_time_around_earth_l33_33194


namespace point_on_line_and_equidistant_l33_33875

theorem point_on_line_and_equidistant {x y : ℝ} :
  (4 * x + 3 * y = 12) ∧ (x = y) ∧ (x ≥ 0) ∧ (y ≥ 0) ↔ x = 12 / 7 ∧ y = 12 / 7 :=
by
  sorry

end point_on_line_and_equidistant_l33_33875


namespace find_integer_values_l33_33058

theorem find_integer_values (a : ℤ) (h : ∃ (n : ℤ), (a + 9) = n * (a + 6)) :
  a = -5 ∨ a = -7 ∨ a = -3 ∨ a = -9 :=
by
  sorry

end find_integer_values_l33_33058


namespace probability_heads_3_ace_l33_33683

def fair_coin_flip : ℕ := 2
def six_sided_die : ℕ := 6
def standard_deck_cards : ℕ := 52

def successful_outcomes : ℕ := 1 * 1 * 4
def total_possible_outcomes : ℕ := fair_coin_flip * six_sided_die * standard_deck_cards

theorem probability_heads_3_ace :
  (successful_outcomes : ℚ) / (total_possible_outcomes : ℚ) = 1 / 156 := 
sorry

end probability_heads_3_ace_l33_33683


namespace alpha_range_midpoint_trajectory_l33_33078

noncomputable def circle_parametric_eqn (θ : ℝ) : ℝ × ℝ :=
  ⟨Real.cos θ, Real.sin θ⟩

theorem alpha_range (α : ℝ) (h1 : 0 < α ∧ α < 2 * Real.pi) :
  (Real.tan α) > 1 ∨ (Real.tan α) < -1 ↔ (Real.pi / 4 < α ∧ α < 3 * Real.pi / 4) ∨ 
                                          (5 * Real.pi / 4 < α ∧ α < 7 * Real.pi / 4) := 
  sorry

theorem midpoint_trajectory (m : ℝ) (h2 : -1 < m ∧ m < 1) :
  ∃ x y : ℝ, x = (Real.sqrt 2 * m) / (m^2 + 1) ∧ 
             y = -(Real.sqrt 2 * m^2) / (m^2 + 1) :=
  sorry

end alpha_range_midpoint_trajectory_l33_33078


namespace A_intersection_B_eq_C_l33_33735

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B := {x : ℝ | 0 < x ∧ x < 3}
def C := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem A_intersection_B_eq_C : A ∩ B = C := 
by sorry

end A_intersection_B_eq_C_l33_33735


namespace number_of_chords_with_integer_length_l33_33155

theorem number_of_chords_with_integer_length 
(centerP_dist radius : ℝ) 
(h1 : centerP_dist = 12) 
(h2 : radius = 20) : 
  ∃ n : ℕ, n = 9 := 
by 
  sorry

end number_of_chords_with_integer_length_l33_33155


namespace three_digit_number_increase_l33_33111

theorem three_digit_number_increase (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (n * 1001 / n) = 1001 :=
by
  sorry

end three_digit_number_increase_l33_33111


namespace sum_of_remainders_mod_11_l33_33994

theorem sum_of_remainders_mod_11
    (a b c d : ℤ)
    (h₁ : a % 11 = 2)
    (h₂ : b % 11 = 4)
    (h₃ : c % 11 = 6)
    (h₄ : d % 11 = 8) :
    (a + b + c + d) % 11 = 9 :=
by
  sorry

end sum_of_remainders_mod_11_l33_33994


namespace total_water_in_heaters_l33_33992

theorem total_water_in_heaters (wallace_capacity : ℕ) (catherine_capacity : ℕ) 
(wallace_water : ℕ) (catherine_water : ℕ) :
  wallace_capacity = 40 →
  (wallace_water = (3 * wallace_capacity) / 4) →
  wallace_capacity = 2 * catherine_capacity →
  (catherine_water = (3 * catherine_capacity) / 4) →
  wallace_water + catherine_water = 45 :=
by
  sorry

end total_water_in_heaters_l33_33992


namespace greatest_number_of_police_officers_needed_l33_33555

-- Define the conditions within Math City
def number_of_streets : ℕ := 10
def number_of_tunnels : ℕ := 2
def intersections_without_tunnels : ℕ := (number_of_streets * (number_of_streets - 1)) / 2
def intersections_bypassed_by_tunnels : ℕ := number_of_tunnels

-- Define the number of police officers required (which is the same as the number of intersections not bypassed)
def police_officers_needed : ℕ := intersections_without_tunnels - intersections_bypassed_by_tunnels

-- The main theorem: Given the conditions, the greatest number of police officers needed is 43.
theorem greatest_number_of_police_officers_needed : police_officers_needed = 43 := 
by {
  -- Proof would go here, but we'll use sorry to indicate it's not provided.
  sorry
}

end greatest_number_of_police_officers_needed_l33_33555


namespace marcie_and_martin_in_picture_l33_33068

noncomputable def marcie_prob_in_picture : ℚ :=
  let marcie_lap_time := 100
  let martin_lap_time := 75
  let start_time := 720
  let end_time := 780
  let picture_duration := 60
  let marcie_position_720 := (720 % marcie_lap_time) / marcie_lap_time
  let marcie_in_pic_start := 0
  let marcie_in_pic_end := 20 + 33 + 1/3
  let martin_position_720 := (720 % martin_lap_time) / martin_lap_time
  let martin_in_pic_start := 20
  let martin_in_pic_end := 45 + 25
  let overlap_start := max marcie_in_pic_start martin_in_pic_start
  let overlap_end := min marcie_in_pic_end martin_in_pic_end
  let overlap_duration := overlap_end - overlap_start
  overlap_duration / picture_duration

theorem marcie_and_martin_in_picture :
  marcie_prob_in_picture = 111 / 200 :=
by
  sorry

end marcie_and_martin_in_picture_l33_33068


namespace calculate_total_revenue_l33_33261

-- Definitions based on conditions
def apple_pie_slices := 8
def peach_pie_slices := 6
def cherry_pie_slices := 10

def apple_pie_price := 3
def peach_pie_price := 4
def cherry_pie_price := 5

def apple_pie_customers := 88
def peach_pie_customers := 78
def cherry_pie_customers := 45

-- Definition of total revenue
def total_revenue := 
  (apple_pie_customers * apple_pie_price) + 
  (peach_pie_customers * peach_pie_price) + 
  (cherry_pie_customers * cherry_pie_price)

-- Target theorem to prove: total revenue equals 801
theorem calculate_total_revenue : total_revenue = 801 := by
  sorry

end calculate_total_revenue_l33_33261


namespace equally_spaced_markings_number_line_l33_33449

theorem equally_spaced_markings_number_line 
  (steps : ℕ) (distance : ℝ) (z_steps : ℕ) (z : ℝ)
  (h1 : steps = 4)
  (h2 : distance = 16)
  (h3 : z_steps = 2) :
  z = (distance / steps) * z_steps :=
by
  sorry

end equally_spaced_markings_number_line_l33_33449


namespace cos_beta_value_l33_33704

theorem cos_beta_value (α β : ℝ) (hα1 : 0 < α ∧ α < π/2) (hβ1 : 0 < β ∧ β < π/2) 
  (h1 : Real.sin α = 4/5) (h2 : Real.cos (α + β) = -12/13) : 
  Real.cos β = -16/65 := 
by 
  sorry

end cos_beta_value_l33_33704


namespace decipher_numbers_l33_33196

variable (K I S : Nat)

theorem decipher_numbers
  (h1: 1 ≤ K ∧ K < 5)
  (h2: I ≠ 0)
  (h3: I ≠ K)
  (h_eq: K * 100 + I * 10 + S + K * 10 + S * 10 + I = I * 100 + S * 10 + K):
  (K, I, S) = (4, 9, 5) :=
by sorry

end decipher_numbers_l33_33196


namespace five_digit_numbers_with_alternating_parity_l33_33014

theorem five_digit_numbers_with_alternating_parity : 
  ∃ n : ℕ, n = 5625 ∧ ∀ (x : ℕ), (10000 ≤ x ∧ x < 100000) → 
    (∀ i, i < 4 → (((x / 10^i) % 10) % 2 ≠ ((x / 10^(i+1)) % 10) % 2)) ↔ 
    (x = 5625) := 
sorry

end five_digit_numbers_with_alternating_parity_l33_33014


namespace intersection_of_M_and_N_l33_33891

def M : Set ℤ := {x : ℤ | -4 < x ∧ x < 2}
def N : Set ℤ := {x : ℤ | x^2 < 4}

theorem intersection_of_M_and_N : (M ∩ N) = { -1, 0, 1 } :=
by
  sorry

end intersection_of_M_and_N_l33_33891


namespace average_speed_correct_l33_33547

noncomputable def average_speed (initial_odometer : ℝ) (lunch_odometer : ℝ) (final_odometer : ℝ) (total_time : ℝ) : ℝ :=
  (final_odometer - initial_odometer) / total_time

theorem average_speed_correct :
  average_speed 212.3 372 467.2 6.25 = 40.784 :=
by
  unfold average_speed
  sorry

end average_speed_correct_l33_33547


namespace reflection_line_sum_l33_33686

-- Prove that the sum of m and b is 10 given the reflection conditions

theorem reflection_line_sum
    (m b : ℚ)
    (H : ∀ (x y : ℚ), (2, 2) = (x, y) → (8, 6) = (2 * (5 - (3 / 2) * (2 - x)), 2 + m * (y - 2)) ∧ y = m * x + b) :
  m + b = 10 :=
sorry

end reflection_line_sum_l33_33686


namespace diana_erasers_l33_33836

theorem diana_erasers (number_of_friends : ℕ) (erasers_per_friend : ℕ) (total_erasers : ℕ) :
  number_of_friends = 48 →
  erasers_per_friend = 80 →
  total_erasers = number_of_friends * erasers_per_friend →
  total_erasers = 3840 :=
by
  intros h_friends h_erasers h_total
  sorry

end diana_erasers_l33_33836


namespace area_of_transformed_region_l33_33048

-- Given conditions
def matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 0], ![5, 3]]
def area_T : ℝ := 9

-- Theorem statement
theorem area_of_transformed_region : 
  let det_matrix := matrix.det
  (det_matrix = 9) → (area_T = 9) → (area_T * det_matrix = 81) :=
by
  intros h₁ h₂
  sorry

end area_of_transformed_region_l33_33048


namespace trig_identity_l33_33814

theorem trig_identity (α : ℝ) (h : Real.tan (π - α) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 3 :=
by
  sorry

end trig_identity_l33_33814


namespace contrapositive_example_l33_33215

theorem contrapositive_example (a b : ℝ) :
  (a > b → a - 1 > b - 2) ↔ (a - 1 ≤ b - 2 → a ≤ b) := 
by
  sorry

end contrapositive_example_l33_33215


namespace part1_part2_l33_33192

-- Prove Part (1)
theorem part1 (M : ℕ) (N : ℕ) (h : M = 9) (h2 : N - 4 + 6 = M) : N = 7 :=
sorry

-- Prove Part (2)
theorem part2 (M : ℕ) (h : M = 9) : M - 4 = 5 ∨ M + 4 = 13 :=
sorry

end part1_part2_l33_33192


namespace sum_of_side_lengths_l33_33177

theorem sum_of_side_lengths (A B C : ℕ) (hA : A = 10) (h_nat_B : B > 0) (h_nat_C : C > 0)
(h_eq_area : B^2 + C^2 = A^2) : B + C = 14 :=
sorry

end sum_of_side_lengths_l33_33177


namespace length_after_y_months_isabella_hair_length_l33_33601

-- Define the initial length of the hair
def initial_length : ℝ := 18

-- Define the growth rate of the hair per month
def growth_rate (x : ℝ) : ℝ := x

-- Define the number of months passed
def months_passed (y : ℕ) : ℕ := y

-- Prove the length of the hair after 'y' months
theorem length_after_y_months (x : ℝ) (y : ℕ) : ℝ :=
  initial_length + growth_rate x * y

-- Theorem statement to prove that the length of Isabella's hair after y months is 18 + xy
theorem isabella_hair_length (x : ℝ) (y : ℕ) : length_after_y_months x y = 18 + x * y :=
by sorry

end length_after_y_months_isabella_hair_length_l33_33601


namespace total_distance_l33_33416

theorem total_distance (D : ℝ) (h_walk : ∀ d t, d = 4 * t) 
                       (h_run : ∀ d t, d = 8 * t) 
                       (h_time : ∀ t_walk t_run, t_walk + t_run = 0.75) 
                       (h_half : D / 2 = d_walk ∧ D / 2 = d_run) :
                       D = 8 := 
by
  sorry

end total_distance_l33_33416


namespace dried_mushrooms_weight_l33_33806

theorem dried_mushrooms_weight (fresh_weight : ℝ) (water_content_fresh : ℝ) (water_content_dried : ℝ) :
  fresh_weight = 22 →
  water_content_fresh = 0.90 →
  water_content_dried = 0.12 →
  ∃ x : ℝ, x = 2.5 :=
by
  intros h1 h2 h3
  have hw_fresh : ℝ := fresh_weight * water_content_fresh
  have dry_material_fresh : ℝ := fresh_weight - hw_fresh
  have dry_material_dried : ℝ := 1.0 - water_content_dried
  have hw_dried := dry_material_fresh / dry_material_dried
  use hw_dried
  sorry

end dried_mushrooms_weight_l33_33806


namespace complement_U_A_eq_l33_33265
noncomputable def U := {x : ℝ | x ≥ -2}
noncomputable def A := {x : ℝ | x > -1}
noncomputable def comp_U_A := {x ∈ U | x ∉ A}

theorem complement_U_A_eq : comp_U_A = {x : ℝ | -2 ≤ x ∧ x < -1} :=
by sorry

end complement_U_A_eq_l33_33265


namespace expression_simplification_l33_33863

theorem expression_simplification (x y : ℝ) :
  20 * (x + y) - 19 * (x + y) = x + y :=
by
  sorry

end expression_simplification_l33_33863


namespace percent_decrease_call_cost_l33_33777

theorem percent_decrease_call_cost (c1990 c2010 : ℝ) (h1990 : c1990 = 50) (h2010 : c2010 = 10) :
  ((c1990 - c2010) / c1990) * 100 = 80 :=
by
  sorry

end percent_decrease_call_cost_l33_33777


namespace probability_four_friends_same_group_l33_33626

-- Define the conditions of the problem
def total_students : ℕ := 900
def groups : ℕ := 5
def friends : ℕ := 4
def probability_per_group : ℚ := 1 / groups

-- Define the statement we need to prove
theorem probability_four_friends_same_group :
  (probability_per_group * probability_per_group * probability_per_group) = 1 / 125 :=
sorry

end probability_four_friends_same_group_l33_33626


namespace prove_dollar_op_l33_33492

variable {a b x y : ℝ}

def dollar_op (a b : ℝ) : ℝ := (a - b) ^ 2

theorem prove_dollar_op :
  dollar_op (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2 * x^2 * y^2 + y^4) := by
  sorry

end prove_dollar_op_l33_33492


namespace smallest_integer_n_l33_33373

theorem smallest_integer_n (n : ℕ) (h : Nat.lcm 60 n / Nat.gcd 60 n = 75) : n = 500 :=
sorry

end smallest_integer_n_l33_33373


namespace boat_speed_in_still_water_l33_33575

theorem boat_speed_in_still_water (b s : ℝ) 
  (h1 : b + s = 11) 
  (h2 : b - s = 3) : b = 7 :=
by
  sorry

end boat_speed_in_still_water_l33_33575


namespace power_subtraction_l33_33466

variable {a m n : ℝ}

theorem power_subtraction (hm : a^m = 8) (hn : a^n = 2) : a^(m - 3 * n) = 1 := by
  sorry

end power_subtraction_l33_33466


namespace triangle_angle_bisector_proportion_l33_33464

theorem triangle_angle_bisector_proportion
  (a b c x y : ℝ)
  (h : x / c = y / a)
  (h2 : x + y = b) :
  x / c = b / (a + c) :=
sorry

end triangle_angle_bisector_proportion_l33_33464


namespace min_width_l33_33044

theorem min_width (w : ℝ) (h : w * (w + 20) ≥ 150) : w ≥ 10 := by
  sorry

end min_width_l33_33044


namespace complement_of_N_is_135_l33_33634

-- Define the universal set M and subset N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 4}

-- Prove that the complement of N in M is {1, 3, 5}
theorem complement_of_N_is_135 : M \ N = {1, 3, 5} := 
by
  sorry

end complement_of_N_is_135_l33_33634


namespace students_shorter_than_yoongi_l33_33073

variable (total_students taller_than_yoongi : Nat)

theorem students_shorter_than_yoongi (h₁ : total_students = 20) (h₂ : taller_than_yoongi = 11) : 
    total_students - (taller_than_yoongi + 1) = 8 :=
by
  -- Here would be the proof
  sorry

end students_shorter_than_yoongi_l33_33073


namespace paths_from_A_to_B_no_revisits_l33_33090

noncomputable def numPaths : ℕ :=
  16

theorem paths_from_A_to_B_no_revisits : numPaths = 16 :=
by
  sorry

end paths_from_A_to_B_no_revisits_l33_33090


namespace arithmetic_geometric_progressions_l33_33149

theorem arithmetic_geometric_progressions (a b : ℕ → ℕ) (d r : ℕ) 
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = r * b n)
  (h_comm_ratio : r = 2)
  (h_eq1 : a 1 + d - 2 * (b 1) = a 1 + 2 * d - 4 * (b 1))
  (h_eq2 : a 1 + d - 2 * (b 1) = 8 * (b 1) - (a 1 + 3 * d)) :
  (a 1 = b 1) ∧ (∃ n, ∀ k, 1 ≤ k ∧ k ≤ 10 → (b (k + 1) = a (1 + n * d) + a 1)) := by
  sorry

end arithmetic_geometric_progressions_l33_33149


namespace third_box_weight_l33_33268

def b1 : ℕ := 2
def difference := 11

def weight_third_box (b1 b3 difference : ℕ) : Prop :=
  b3 - b1 = difference

theorem third_box_weight : weight_third_box b1 13 difference :=
by
  simp [b1, difference]
  sorry

end third_box_weight_l33_33268


namespace functional_equation_solution_l33_33029

theorem functional_equation_solution (f : ℤ → ℝ) (hf : ∀ x y : ℤ, f (↑((x + y) / 3)) = (f x + f y) / 2) :
    ∃ c : ℝ, ∀ x : ℤ, x ≠ 0 → f x = c :=
sorry

end functional_equation_solution_l33_33029


namespace additional_plates_correct_l33_33840

-- Define the conditions
def original_set_1 : Finset Char := {'B', 'F', 'J', 'N', 'T'}
def original_set_2 : Finset Char := {'E', 'U'}
def original_set_3 : Finset Char := {'G', 'K', 'R', 'Z'}

-- Define the sizes of the original sets
def size_set_1 := (original_set_1.card : Nat) -- 5
def size_set_2 := (original_set_2.card : Nat) -- 2
def size_set_3 := (original_set_3.card : Nat) -- 4

-- Sizes after adding new letters
def new_size_set_1 := size_set_1 + 1 -- 6
def new_size_set_2 := size_set_2 + 1 -- 3
def new_size_set_3 := size_set_3 + 1 -- 5

-- Calculate the original and new number of plates
def original_plates : Nat := size_set_1 * size_set_2 * size_set_3 -- 5 * 2 * 4 = 40
def new_plates : Nat := new_size_set_1 * new_size_set_2 * new_size_set_3 -- 6 * 3 * 5 = 90

-- Calculate the additional plates
def additional_plates : Nat := new_plates - original_plates -- 90 - 40 = 50

-- The proof statement
theorem additional_plates_correct : additional_plates = 50 :=
by
  -- Proof can be filled in here
  sorry

end additional_plates_correct_l33_33840


namespace length_of_BC_l33_33237

-- Definitions of given conditions
def AB : ℝ := 4
def AC : ℝ := 3
def dot_product_AC_BC : ℝ := 1

-- Hypothesis used in the problem
axiom nonneg_AC (AC : ℝ) : AC ≥ 0
axiom nonneg_AB (AB : ℝ) : AB ≥ 0

-- Statement to be proved
theorem length_of_BC (AB AC dot_product_AC_BC : ℝ)
  (h1 : AB = 4) (h2 : AC = 3) (h3 : dot_product_AC_BC = 1) : exists (BC : ℝ), BC = 3 := by
  sorry

end length_of_BC_l33_33237


namespace sum_of_squares_of_geometric_progression_l33_33057

theorem sum_of_squares_of_geometric_progression 
  {b_1 q S_1 S_2 : ℝ} 
  (h1 : |q| < 1) 
  (h2 : S_1 = b_1 / (1 - q))
  (h3 : S_2 = b_1 / (1 + q)) : 
  (b_1^2 / (1 - q^2)) = S_1 * S_2 := 
by
  sorry

end sum_of_squares_of_geometric_progression_l33_33057


namespace find_f_2_l33_33135

def f (a b x : ℝ) := a * x^3 - b * x + 1

theorem find_f_2 (a b : ℝ) (h : f a b (-2) = -1) : f a b 2 = 3 :=
by
  sorry

end find_f_2_l33_33135


namespace log_pi_inequality_l33_33134

theorem log_pi_inequality (a b : ℝ) (π : ℝ) (h1 : 2^a = π) (h2 : 5^b = π) (h3 : a = Real.log π / Real.log 2) (h4 : b = Real.log π / Real.log 5) :
  (1 / a) + (1 / b) > 2 :=
by
  sorry

end log_pi_inequality_l33_33134


namespace smallest_positive_integer_l33_33973

theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, 3003 * m + 66666 * n = 3 :=
by
  sorry

end smallest_positive_integer_l33_33973


namespace problem_statement_l33_33788

noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 30
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by 
  sorry

end problem_statement_l33_33788


namespace tom_initial_money_l33_33712

theorem tom_initial_money (spent_on_game : ℕ) (toy_cost : ℕ) (number_of_toys : ℕ)
    (total_spent : ℕ) (h1 : spent_on_game = 49) (h2 : toy_cost = 4)
    (h3 : number_of_toys = 2) (h4 : total_spent = spent_on_game + number_of_toys * toy_cost) :
  total_spent = 57 := by
  sorry

end tom_initial_money_l33_33712


namespace arrange_polynomial_ascending_order_l33_33817

variable {R : Type} [Ring R] (x : R)

def p : R := 3 * x ^ 2 - x + x ^ 3 - 1

theorem arrange_polynomial_ascending_order : 
  p x = -1 - x + 3 * x ^ 2 + x ^ 3 :=
by
  sorry

end arrange_polynomial_ascending_order_l33_33817


namespace initial_mixture_two_l33_33314

theorem initial_mixture_two (x : ℝ) (h : 0.25 * (x + 0.4) = 0.10 * x + 0.4) : x = 2 :=
by
  sorry

end initial_mixture_two_l33_33314


namespace fewer_twos_to_hundred_l33_33064

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l33_33064


namespace triangle_side_lengths_l33_33778

theorem triangle_side_lengths 
  (r : ℝ) (CD : ℝ) (DB : ℝ) 
  (h_r : r = 4) 
  (h_CD : CD = 8) 
  (h_DB : DB = 10) :
  ∃ (AB AC : ℝ), AB = 14.5 ∧ AC = 12.5 :=
by
  sorry

end triangle_side_lengths_l33_33778


namespace sum_of_digits_of_N_l33_33282

-- Define N
def N := 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8 + 10^9

-- Define function to calculate sum of digits
def sum_of_digits(n: Nat) : Nat :=
  n.digits 10 |>.sum

-- Theorem statement
theorem sum_of_digits_of_N : sum_of_digits N = 7 :=
  sorry

end sum_of_digits_of_N_l33_33282


namespace vector_subtraction_parallel_l33_33321

theorem vector_subtraction_parallel (t : ℝ) 
  (h_parallel : -1 / 2 = -3 / t) : 
  ( (-1 : ℝ), -3 ) - ( 2, t ) = (-3, -9) :=
by
  -- proof goes here
  sorry

end vector_subtraction_parallel_l33_33321


namespace roots_polynomial_l33_33008

theorem roots_polynomial (a b c : ℝ) (h1 : a + b + c = 18) (h2 : a * b + b * c + c * a = 19) (h3 : a * b * c = 8) : 
  (1 + a) * (1 + b) * (1 + c) = 46 :=
by
  sorry

end roots_polynomial_l33_33008


namespace problem_l33_33114

theorem problem (C D : ℝ) (h : ∀ x : ℝ, x ≠ 4 → 
  (C / (x - 4)) + D * (x + 2) = (-2 * x^3 + 8 * x^2 + 35 * x + 48) / (x - 4)) : 
  C + D = 174 :=
sorry

end problem_l33_33114


namespace area_of_rectangle_l33_33698

noncomputable def rectangle_area : ℚ :=
  let side1 : ℚ := 73 / 10
  let side2 : ℚ := 94 / 10
  let side3 : ℚ := 113 / 10
  let perimeter_triangle : ℚ := side1 + side2 + side3
  let width : ℚ := perimeter_triangle / 6
  let length : ℚ := 2 * width
  length * width

theorem area_of_rectangle : rectangle_area = 392 / 9 :=
  by 
  let side1 : ℚ := 73 / 10
  let side2 : ℚ := 94 / 10
  let side3 : ℚ := 113 / 10
  let perimeter_triangle : ℚ := side1 + side2 + side3
  let width : ℚ := perimeter_triangle / 6
  let length : ℚ := 2 * width
  have : length * width = 392 / 9 := sorry
  exact this

end area_of_rectangle_l33_33698


namespace sin_sq_sub_sin_double_l33_33796

open Real

theorem sin_sq_sub_sin_double (alpha : ℝ) (h : tan alpha = 1 / 2) : sin alpha ^ 2 - sin (2 * alpha) = -3 / 5 := 
by 
  sorry

end sin_sq_sub_sin_double_l33_33796


namespace certain_amount_is_19_l33_33566

theorem certain_amount_is_19 (x y certain_amount : ℤ) 
  (h1 : x + y = 15)
  (h2 : 3 * x = 5 * y - certain_amount)
  (h3 : x = 7) : 
  certain_amount = 19 :=
by
  sorry

end certain_amount_is_19_l33_33566


namespace julia_investment_l33_33696

-- Define the total investment and the relationship between the investments
theorem julia_investment:
  ∀ (m : ℕ), 
  m + 6 * m = 200000 → 6 * m = 171428 := 
by
  sorry

end julia_investment_l33_33696


namespace count_two_digit_or_less_numbers_l33_33733

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l33_33733


namespace complex_number_evaluation_l33_33982

noncomputable def i := Complex.I

theorem complex_number_evaluation :
  (1 - i) * (i * i) / (1 + 2 * i) = (1/5 : ℂ) + (3/5 : ℂ) * i :=
by
  sorry

end complex_number_evaluation_l33_33982


namespace fraction_representation_of_3_36_l33_33387

theorem fraction_representation_of_3_36 : (336 : ℚ) / 100 = 84 / 25 := 
by sorry

end fraction_representation_of_3_36_l33_33387


namespace number_of_factors_n_l33_33578

-- Defining the value of n with its prime factorization
def n : ℕ := 2^5 * 3^9 * 5^5

-- Theorem stating the number of natural-number factors of n
theorem number_of_factors_n : 
  (Nat.divisors n).card = 360 := by
  -- Proof is omitted
  sorry

end number_of_factors_n_l33_33578


namespace slope_angle_of_line_l33_33378

theorem slope_angle_of_line (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : (m^2 + n^2) / m^2 = 4) :
  ∃ θ : ℝ, θ = π / 6 ∨ θ = 5 * π / 6 :=
by
  sorry

end slope_angle_of_line_l33_33378


namespace total_marks_l33_33009

-- Define the conditions
def average_marks : ℝ := 35
def number_of_candidates : ℕ := 120

-- Define the total marks as a goal to prove
theorem total_marks : number_of_candidates * average_marks = 4200 :=
by
  sorry

end total_marks_l33_33009


namespace rational_numbers_satisfying_conditions_l33_33239

theorem rational_numbers_satisfying_conditions :
  (∃ n : ℕ, n = 166 ∧ ∀ (m : ℚ),
  abs m < 500 → (∃ x : ℤ, 3 * x^2 + m * x + 25 = 0) ↔ n = 166)
:=
sorry

end rational_numbers_satisfying_conditions_l33_33239


namespace ratio_of_areas_l33_33003

theorem ratio_of_areas (n : ℕ) (r s : ℕ) (square_area : ℕ) (triangle_adf_area : ℕ)
  (h_square_area : square_area = 4)
  (h_triangle_adf_area : triangle_adf_area = n * square_area)
  (h_triangle_sim : s = 8 / r)
  (h_r_eq_n : r = n):
  (s / square_area) = 2 / n :=
by
  sorry

end ratio_of_areas_l33_33003


namespace number_of_men_in_first_group_l33_33907

-- Condition: Let M be the number of men in the first group
variable (M : ℕ)

-- Condition: M men can complete the work in 20 hours
-- Condition: 15 men can complete the same work in 48 hours
-- We want to prove that if M * 20 = 15 * 48, then M = 36
theorem number_of_men_in_first_group (h : M * 20 = 15 * 48) : M = 36 := by
  sorry

end number_of_men_in_first_group_l33_33907


namespace largest_of_three_consecutive_integers_sum_18_l33_33567

theorem largest_of_three_consecutive_integers_sum_18 (n : ℤ) (h : n + (n + 1) + (n + 2) = 18) : n + 2 = 7 :=
by
  sorry

end largest_of_three_consecutive_integers_sum_18_l33_33567


namespace evaluate_f_difference_l33_33610

def f (x : ℤ) : ℤ := x^6 + 3 * x^4 - 4 * x^3 + x^2 + 2 * x

theorem evaluate_f_difference : f 3 - f (-3) = -204 := by
  sorry

end evaluate_f_difference_l33_33610


namespace function_identity_l33_33513

theorem function_identity {f : ℕ → ℕ} (h₀ : f 1 > 0) 
  (h₁ : ∀ m n : ℕ, f (m^2 + n^2) = f m^2 + f n^2) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_identity_l33_33513


namespace problem_statement_l33_33444

open Real

theorem problem_statement (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : A ≠ 0)
    (h3 : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) : 
    |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| := sorry

end problem_statement_l33_33444


namespace quadrilateral_count_l33_33214

-- Define the number of points
def num_points := 9

-- Define the number of vertices in a quadrilateral
def vertices_in_quadrilateral := 4

-- Use a combination function to find the number of ways to choose 4 points out of 9
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem that asserts the number of quadrilaterals that can be formed
theorem quadrilateral_count : combination num_points vertices_in_quadrilateral = 126 :=
by
  -- The proof would go here
  sorry

end quadrilateral_count_l33_33214


namespace clock_hands_coincide_22_times_clock_hands_straight_angle_24_times_clock_hands_right_angle_48_times_l33_33669

theorem clock_hands_coincide_22_times
  (minute_hand_cycles_24_hours : ℕ := 24)
  (hour_hand_cycles_24_hours : ℕ := 2)
  (minute_hand_overtakes_hour_hand_per_12_hours : ℕ := 11) :
  2 * minute_hand_overtakes_hour_hand_per_12_hours = 22 :=
by
  -- Proof should be filled here
  sorry

theorem clock_hands_straight_angle_24_times
  (hours_in_day : ℕ := 24)
  (straight_angle_per_hour : ℕ := 1) :
  hours_in_day * straight_angle_per_hour = 24 :=
by
  -- Proof should be filled here
  sorry

theorem clock_hands_right_angle_48_times
  (hours_in_day : ℕ := 24)
  (right_angles_per_hour : ℕ := 2) :
  hours_in_day * right_angles_per_hour = 48 :=
by
  -- Proof should be filled here
  sorry

end clock_hands_coincide_22_times_clock_hands_straight_angle_24_times_clock_hands_right_angle_48_times_l33_33669


namespace each_friend_gave_bella_2_roses_l33_33847

-- Define the given conditions
def total_roses_from_parents : ℕ := 2 * 12
def total_roses_bella_received : ℕ := 44
def number_of_dancer_friends : ℕ := 10

-- Define the mathematical goal
def roses_from_each_friend (total_roses_from_parents total_roses_bella_received number_of_dancer_friends : ℕ) : ℕ :=
  (total_roses_bella_received - total_roses_from_parents) / number_of_dancer_friends

-- Prove that each dancer friend gave Bella 2 roses
theorem each_friend_gave_bella_2_roses :
  roses_from_each_friend total_roses_from_parents total_roses_bella_received number_of_dancer_friends = 2 :=
by
  sorry

end each_friend_gave_bella_2_roses_l33_33847


namespace shaded_area_correct_l33_33603

def first_rectangle_area (w l : ℕ) : ℕ := w * l
def second_rectangle_area (w l : ℕ) : ℕ := w * l
def overlap_triangle_area (b h : ℕ) : ℕ := (b * h) / 2
def total_shaded_area (area1 area2 overlap : ℕ) : ℕ := area1 + area2 - overlap

theorem shaded_area_correct :
  let w1 := 4
  let l1 := 12
  let w2 := 5
  let l2 := 10
  let b := 4
  let h := 5
  let area1 := first_rectangle_area w1 l1
  let area2 := second_rectangle_area w2 l2
  let overlap := overlap_triangle_area b h
  total_shaded_area area1 area2 overlap = 88 := 
by
  sorry

end shaded_area_correct_l33_33603


namespace technicians_count_l33_33737

noncomputable def total_salary := 8000 * 21
noncomputable def average_salary_all := 8000
noncomputable def average_salary_technicians := 12000
noncomputable def average_salary_rest := 6000
noncomputable def total_workers := 21

theorem technicians_count :
  ∃ (T R : ℕ),
  T + R = total_workers ∧
  average_salary_technicians * T + average_salary_rest * R = total_salary ∧
  T = 7 :=
by
  sorry

end technicians_count_l33_33737


namespace intersection_A_B_l33_33274

open Set

variable (l : ℝ)

def A := {x : ℝ | x > l}
def B := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_A_B (h₁ : l = 1) :
  A l ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l33_33274


namespace find_duplicate_page_l33_33293

theorem find_duplicate_page (n p : ℕ) (h : (n * (n + 1) / 2) + p = 3005) : p = 2 := 
sorry

end find_duplicate_page_l33_33293


namespace find_number_of_persons_l33_33435

-- Definitions of the given conditions
def total_amount : ℕ := 42900
def amount_per_person : ℕ := 1950

-- The statement to prove
theorem find_number_of_persons (n : ℕ) (h : total_amount = n * amount_per_person) : n = 22 :=
sorry

end find_number_of_persons_l33_33435


namespace painting_rate_l33_33522

/-- Define various dimensions and costs for the room -/
def room_length : ℝ := 10
def room_width  : ℝ := 7
def room_height : ℝ := 5

def door_width  : ℝ := 1
def door_height : ℝ := 3
def num_doors   : ℕ := 2

def large_window_width  : ℝ := 2
def large_window_height : ℝ := 1.5
def num_large_windows   : ℕ := 1

def small_window_width  : ℝ := 1
def small_window_height : ℝ := 1.5
def num_small_windows   : ℕ := 2

def painting_cost : ℝ := 474

/-- The rate for painting the walls is Rs. 3 per sq m -/
theorem painting_rate : (painting_cost / 
  ((2 * (room_length * room_height) + 2 * (room_width * room_height)) -
   (num_doors * (door_width * door_height) +
    num_large_windows * (large_window_width * large_window_height) +
    num_small_windows * (small_window_width * small_window_height)))) = 3 := 
by 
  -- Proof is omitted
  sorry

end painting_rate_l33_33522


namespace total_drums_hit_l33_33163

/-- 
Given the conditions of the problem, Juanita hits 4500 drums in total. 
-/
theorem total_drums_hit (entry_fee cost_per_drum_hit earnings_per_drum_hit_beyond_200_double
                         net_loss: ℝ) 
                         (first_200_drums hits_after_200: ℕ) :
  entry_fee = 10 → 
  cost_per_drum_hit = 0.02 →
  earnings_per_drum_hit_beyond_200_double = 0.025 →
  net_loss = -7.5 →
  hits_after_200 = 4300 →
  first_200_drums = 200 →
  (-net_loss = entry_fee + (first_200_drums * cost_per_drum_hit) +
   (hits_after_200 * (earnings_per_drum_hit_beyond_200_double - cost_per_drum_hit))) →
  first_200_drums + hits_after_200 = 4500 :=
by
  intro h_entry_fee h_cost_per_drum_hit h_earnings_per_drum_hit_beyond_200_double h_net_loss h_hits_after_200
       h_first_200_drums h_loss_equation
  sorry

end total_drums_hit_l33_33163


namespace mark_boxes_sold_l33_33333

theorem mark_boxes_sold (n : ℕ) (M A : ℕ) (h1 : A = n - 2) (h2 : M + A < n) (h3 :  1 ≤ M) (h4 : 1 ≤ A) (hn : n = 12) : M = 1 :=
by
  sorry

end mark_boxes_sold_l33_33333


namespace present_population_l33_33383

theorem present_population (P : ℝ) (h1 : (P : ℝ) * (1 + 0.1) ^ 2 = 14520) : P = 12000 :=
sorry

end present_population_l33_33383


namespace anicka_savings_l33_33494

theorem anicka_savings (x y : ℕ) (h1 : x + y = 290) (h2 : (1/4 : ℚ) * (2 * y) = (1/3 : ℚ) * x) : 2 * y + x = 406 :=
by
  sorry

end anicka_savings_l33_33494


namespace compute_division_l33_33127

variable (a b c : ℕ)
variable (ha : a = 3)
variable (hb : b = 2)
variable (hc : c = 2)

theorem compute_division : (c * a^3 + c * b^3) / (a^2 - a * b + b^2) = 10 := by
  sorry

end compute_division_l33_33127


namespace S_5_is_121_l33_33590

-- Definitions of the sequence and its terms
def S : ℕ → ℕ := sorry  -- Define S_n
def a : ℕ → ℕ := sorry  -- Define a_n

-- Conditions
axiom S_2 : S 2 = 4
axiom recurrence_relation : ∀ n : ℕ, S (n + 1) = 1 + 2 * S n

-- Proof that S_5 = 121 given the conditions
theorem S_5_is_121 : S 5 = 121 := by
  sorry

end S_5_is_121_l33_33590


namespace towel_bleach_percentage_decrease_l33_33140

theorem towel_bleach_percentage_decrease :
  ∀ (L B : ℝ), (L > 0) → (B > 0) → 
  let L' := 0.70 * L 
  let B' := 0.75 * B 
  let A := L * B 
  let A' := L' * B' 
  (A - A') / A * 100 = 47.5 :=
by sorry

end towel_bleach_percentage_decrease_l33_33140


namespace p_is_sufficient_but_not_necessary_for_q_l33_33721

-- Definitions and conditions
def p (x : ℝ) : Prop := (x = 1)
def q (x : ℝ) : Prop := (x^2 - 3 * x + 2 = 0)

-- Theorem statement
theorem p_is_sufficient_but_not_necessary_for_q : ∀ x : ℝ, (p x → q x) ∧ (¬ (q x → p x)) :=
by
  sorry

end p_is_sufficient_but_not_necessary_for_q_l33_33721


namespace largest_n_for_factoring_polynomial_l33_33438

theorem largest_n_for_factoring_polynomial :
  ∃ A B : ℤ, A * B = 120 ∧ (∀ n, (5 * 120 + 1 ≤ n → n ≤ 601)) := sorry

end largest_n_for_factoring_polynomial_l33_33438


namespace length_of_field_l33_33900

variable (w : ℕ) (l : ℕ)

def length_field_is_double_width (w l : ℕ) : Prop :=
  l = 2 * w

def pond_area_equals_one_eighth_field_area (w l : ℕ) : Prop :=
  36 = 1 / 8 * (l * w)

theorem length_of_field (w l : ℕ) (h1 : length_field_is_double_width w l) (h2 : pond_area_equals_one_eighth_field_area w l) : l = 24 := 
by
  sorry

end length_of_field_l33_33900


namespace min_value_theorem_l33_33240

noncomputable def min_value (x y : ℝ) : ℝ :=
  (x + 2) * (2 * y + 1) / (x * y)

theorem min_value_theorem {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  min_value x y = 19 + 4 * Real.sqrt 15 :=
sorry

end min_value_theorem_l33_33240


namespace part1_solution_part2_solution_l33_33020

-- Part (1) Statement
theorem part1_solution (x : ℝ) (m : ℝ) (h_m : m = -1) :
  (3 * x - m) / 2 - (x + m) / 3 = 5 / 6 → x = 0 :=
by
  intros h_eq
  rw [h_m] at h_eq
  sorry  -- Proof to be filled in

-- Part (2) Statement
theorem part2_solution (x m : ℝ) (h_x : x = 5)
  (h_eq : (3 * x - m) / 2 - (x + m) / 3 = 5 / 6) :
  (1 / 2) * m^2 + 2 * m = 30 :=
by
  rw [h_x] at h_eq
  sorry  -- Proof to be filled in

end part1_solution_part2_solution_l33_33020


namespace compute_f_at_919_l33_33504

-- Given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) : Prop :=
∀ x, f (x + 4) = f (x - 2)

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ [-3, 0] then 6^(-x) else sorry

-- Lean statement for the proof problem
theorem compute_f_at_919 (f : ℝ → ℝ)
    (h_even : is_even_function f)
    (h_periodic : periodic_function f)
    (h_defined : ∀ x ∈ [-3, 0], f x = 6^(-x)) :
    f 919 = 6 := sorry

end compute_f_at_919_l33_33504


namespace car_travel_first_hour_l33_33929

-- Define the conditions as variables and the ultimate equality to be proved
theorem car_travel_first_hour (x : ℕ) (h : 12 * x + 132 = 612) : x = 40 :=
by
  -- Proof will be completed here
  sorry

end car_travel_first_hour_l33_33929


namespace divisor_of_51234_plus_3_l33_33075

theorem divisor_of_51234_plus_3 : ∃ d : ℕ, d > 1 ∧ (51234 + 3) % d = 0 ∧ d = 3 :=
by {
  sorry
}

end divisor_of_51234_plus_3_l33_33075


namespace solution_set_of_inequality_l33_33730

theorem solution_set_of_inequality (a x : ℝ) (h : a > 0) : 
  (x^2 - (a + 1/a + 1) * x + a + 1/a < 0) ↔ (1 < x ∧ x < a + 1/a) :=
by sorry

end solution_set_of_inequality_l33_33730


namespace total_parallelograms_in_grid_l33_33030

theorem total_parallelograms_in_grid (n : ℕ) : 
  ∃ p : ℕ, p = 3 * Nat.choose (n + 2) 4 :=
sorry

end total_parallelograms_in_grid_l33_33030


namespace actual_length_correct_l33_33316

-- Definitions based on the conditions
def blueprint_scale : ℝ := 20
def measured_length_cm : ℝ := 16

-- Statement of the proof problem
theorem actual_length_correct :
  measured_length_cm * blueprint_scale = 320 := 
sorry

end actual_length_correct_l33_33316


namespace parrots_per_cage_l33_33744

theorem parrots_per_cage (P : ℕ) (total_birds total_cages parakeets_per_cage : ℕ)
  (h1 : total_cages = 4)
  (h2 : parakeets_per_cage = 2)
  (h3 : total_birds = 40)
  (h4 : total_birds = total_cages * (P + parakeets_per_cage)) :
  P = 8 :=
by
  sorry

end parrots_per_cage_l33_33744


namespace polynomial_even_or_odd_polynomial_divisible_by_3_l33_33191

theorem polynomial_even_or_odd (p q : ℤ) :
  (∀ x : ℤ, (x^2 + p * x + q) % 2 = 0 ↔ (q % 2 = 0) ∧ (p % 2 = 1)) ∧
  (∀ x : ℤ, (x^2 + p * x + q) % 2 = 1 ↔ (q % 2 = 1) ∧ (p % 2 = 1)) := 
sorry

theorem polynomial_divisible_by_3 (p q : ℤ) :
  (∀ x : ℤ, (x^3 + p * x + q) % 3 = 0) ↔ (q % 3 = 0) ∧ (p % 3 = 2) := 
sorry

end polynomial_even_or_odd_polynomial_divisible_by_3_l33_33191


namespace find_a11_l33_33426

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a11 (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 1 * a 4 = 20)
  (h3 : a 0 + a 5 = 9) :
  a 10 = 25 / 4 :=
sorry

end find_a11_l33_33426


namespace horse_food_calculation_l33_33862

theorem horse_food_calculation
  (num_sheep : ℕ)
  (ratio_sheep_horses : ℕ)
  (total_horse_food : ℕ)
  (H : ℕ)
  (num_sheep_eq : num_sheep = 56)
  (ratio_eq : ratio_sheep_horses = 7)
  (total_food_eq : total_horse_food = 12880)
  (num_horses : H = num_sheep * 1 / ratio_sheep_horses)
  : num_sheep = ratio_sheep_horses → total_horse_food / H = 230 :=
by
  sorry

end horse_food_calculation_l33_33862


namespace nat_numbers_l33_33067

theorem nat_numbers (n : ℕ) (h1 : n ≥ 2) (h2 : ∃a b : ℕ, a * b = n ∧ ∀ c : ℕ, 1 < c ∧ c ∣ n → a ≤ c ∧ n = a^2 + b^2) : 
  n = 5 ∨ n = 8 ∨ n = 20 :=
by
  sorry

end nat_numbers_l33_33067


namespace alex_friends_invite_l33_33031

theorem alex_friends_invite (burger_buns_per_pack : ℕ)
                            (packs_of_buns : ℕ)
                            (buns_needed_by_each_guest : ℕ)
                            (total_buns : ℕ)
                            (friends_who_dont_eat_buns : ℕ)
                            (friends_who_dont_eat_meat : ℕ)
                            (total_friends_invited : ℕ) 
                            (h1 : burger_buns_per_pack = 8)
                            (h2 : packs_of_buns = 3)
                            (h3 : buns_needed_by_each_guest = 3)
                            (h4 : total_buns = packs_of_buns * burger_buns_per_pack)
                            (h5 : friends_who_dont_eat_buns = 1)
                            (h6 : friends_who_dont_eat_meat = 1)
                            (h7 : total_friends_invited = (total_buns / buns_needed_by_each_guest) + friends_who_dont_eat_buns) :
  total_friends_invited = 9 :=
by sorry

end alex_friends_invite_l33_33031


namespace simplify_expression_l33_33429

def expression1 (x : ℝ) : ℝ :=
  3 * x^3 + 4 * x^2 + 2 * x + 5 - (2 * x^3 - 5 * x^2 + x - 3) + (x^3 - 2 * x^2 - 4 * x + 6)

def expression2 (x : ℝ) : ℝ :=
  2 * x^3 + 7 * x^2 - 3 * x + 14

theorem simplify_expression (x : ℝ) : expression1 x = expression2 x :=
by 
  sorry

end simplify_expression_l33_33429


namespace distance_from_pole_to_line_l33_33758

/-- Definition of the line in polar coordinates -/
def line_polar (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- Definition of the pole in Cartesian coordinates -/
def pole_cartesian : ℝ × ℝ := (0, 0)

/-- Convert the line from polar to Cartesian -/
def line_cartesian (x y : ℝ) : Prop := x = 2

/-- The distance function between a point and a line in Cartesian coordinates -/
def distance_to_line (p : ℝ × ℝ) : ℝ := abs (p.1 - 2)

/-- Prove that the distance from the pole to the line is 2 -/
theorem distance_from_pole_to_line : distance_to_line pole_cartesian = 2 := by
  sorry

end distance_from_pole_to_line_l33_33758


namespace abs_value_expression_l33_33216

theorem abs_value_expression : abs (3 * Real.pi - abs (3 * Real.pi - 10)) = 6 * Real.pi - 10 :=
by sorry

end abs_value_expression_l33_33216


namespace area_of_fourth_rectangle_l33_33519

theorem area_of_fourth_rectangle
  (A1 A2 A3 A_total : ℕ)
  (h1 : A1 = 24)
  (h2 : A2 = 30)
  (h3 : A3 = 18)
  (h_total : A_total = 100) :
  ∃ A4 : ℕ, A1 + A2 + A3 + A4 = A_total ∧ A4 = 28 :=
by
  sorry

end area_of_fourth_rectangle_l33_33519


namespace shadow_length_l33_33258

variable (H h d : ℝ) (h_pos : h > 0) (H_pos : H > 0) (H_neq_h : H ≠ h)

theorem shadow_length (x : ℝ) (hx : x = d * h / (H - h)) :
  x = d * h / (H - h) :=
sorry

end shadow_length_l33_33258


namespace problem_I_problem_II_l33_33833

-- Definition of the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Problem (I): Prove solution set
theorem problem_I (x : ℝ) : f (x - 1) + f (x + 3) ≥ 6 ↔ (x ≤ -3 ∨ x ≥ 3) := by
  sorry

-- Problem (II): Prove inequality given conditions
theorem problem_II (a b : ℝ) (ha: |a| < 1) (hb: |b| < 1) (hano: a ≠ 0) : 
  f (a * b) > |a| * f (b / a) := by
  sorry

end problem_I_problem_II_l33_33833


namespace thirtieth_term_value_l33_33317

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l33_33317


namespace oplus_self_twice_l33_33487

def my_oplus (x y : ℕ) := 3^x - y

theorem oplus_self_twice (a : ℕ) : my_oplus a (my_oplus a a) = a := by
  sorry

end oplus_self_twice_l33_33487


namespace distance_between_planes_l33_33446

def plane1 (x y z : ℝ) := 3 * x - y + z - 3 = 0
def plane2 (x y z : ℝ) := 6 * x - 2 * y + 2 * z + 4 = 0

theorem distance_between_planes :
  ∃ d : ℝ, d = (5 * Real.sqrt 11) / 11 ∧ 
            ∀ x y z : ℝ, plane1 x y z → plane2 x y z → d = (5 * Real.sqrt 11) / 11 :=
sorry

end distance_between_planes_l33_33446


namespace determine_value_of_c_l33_33536

theorem determine_value_of_c (b : ℝ) (h₁ : ∀ x : ℝ, 0 ≤ x^2 + x + b) (h₂ : ∃ m : ℝ, ∀ x : ℝ, x^2 + x + b < c ↔ x = m + 8) : 
    c = 16 :=
sorry

end determine_value_of_c_l33_33536


namespace field_length_l33_33252

theorem field_length (w l : ℝ) (A_f A_p : ℝ) 
  (h1 : l = 3 * w)
  (h2 : A_p = 150) 
  (h3 : A_p = 0.4 * A_f)
  (h4 : A_f = l * w) : 
  l = 15 * Real.sqrt 5 :=
by
  sorry

end field_length_l33_33252


namespace mrs_wilsborough_tickets_l33_33110

theorem mrs_wilsborough_tickets :
  ∀ (saved vip_ticket_cost regular_ticket_cost vip_tickets left : ℕ),
    saved = 500 →
    vip_ticket_cost = 100 →
    regular_ticket_cost = 50 →
    vip_tickets = 2 →
    left = 150 →
    (saved - left - (vip_tickets * vip_ticket_cost)) / regular_ticket_cost = 3 :=
by
  intros saved vip_ticket_cost regular_ticket_cost vip_tickets left
  sorry

end mrs_wilsborough_tickets_l33_33110


namespace problem_equivalent_l33_33720

def modified_op (a b : ℝ) : ℝ := (a + b) ^ 2

theorem problem_equivalent (x y : ℝ) : 
  modified_op ((x + y) ^ 2) ((y + x) ^ 2) = 4 * (x + y) ^ 4 := 
by 
  sorry

end problem_equivalent_l33_33720


namespace gambler_largest_amount_proof_l33_33609

noncomputable def largest_amount_received_back (initial_amount : ℝ) (value_25 : ℝ) (value_75 : ℝ) (value_250 : ℝ) 
                                               (total_lost_chips : ℝ) (coef_25_75_lost : ℝ) (coef_75_250_lost : ℝ) : ℝ :=
    initial_amount - (
    coef_25_75_lost * (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_25 +
    (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_75 +
    coef_75_250_lost * (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_250)

theorem gambler_largest_amount_proof :
    let initial_amount := 15000
    let value_25 := 25
    let value_75 := 75
    let value_250 := 250
    let total_lost_chips := 40
    let coef_25_75_lost := 2 -- number of lost $25 chips is twice the number of lost $75 chips
    let coef_75_250_lost := 2 -- number of lost $250 chips is twice the number of lost $75 chips
    largest_amount_received_back initial_amount value_25 value_75 value_250 total_lost_chips coef_25_75_lost coef_75_250_lost = 10000 :=
by {
    sorry
}

end gambler_largest_amount_proof_l33_33609


namespace athena_total_spent_l33_33922

-- Define the conditions
def sandwiches_quantity : ℕ := 3
def sandwich_price : ℝ := 3.0
def drinks_quantity : ℕ := 2
def drink_price : ℝ := 2.5

-- Define the calculations
def total_sandwich_cost : ℝ := sandwiches_quantity * sandwich_price
def total_drink_cost : ℝ := drinks_quantity * drink_price

-- Define the total cost
def total_amount_spent : ℝ := total_sandwich_cost + total_drink_cost

-- Prove the total amount spent
theorem athena_total_spent : total_amount_spent = 14 := by
  sorry

end athena_total_spent_l33_33922


namespace min_a2_k2b2_l33_33062

variable (a b t k : ℝ)
variable (hk : 0 < k)
variable (h : a + k * b = t)

theorem min_a2_k2b2 (a b t k : ℝ) (hk : 0 < k) (h : a + k * b = t) :
  a^2 + (k * b)^2 ≥ (1 + k^2) * (t^2) / ((1 + k)^2) :=
sorry

end min_a2_k2b2_l33_33062


namespace mary_sailboat_canvas_l33_33755

def rectangular_sail_area (length width : ℕ) : ℕ :=
  length * width

def triangular_sail_area (base height : ℕ) : ℕ :=
  (base * height) / 2

def total_canvas_area (length₁ width₁ base₁ height₁ base₂ height₂ : ℕ) : ℕ :=
  rectangular_sail_area length₁ width₁ +
  triangular_sail_area base₁ height₁ +
  triangular_sail_area base₂ height₂

theorem mary_sailboat_canvas :
  total_canvas_area 5 8 3 4 4 6 = 58 :=
by
  -- Begin proof (proof steps omitted, we just need the structure here)
  sorry -- end proof

end mary_sailboat_canvas_l33_33755


namespace solution_ratio_l33_33013

-- Describe the problem conditions
variable (a b : ℝ) -- amounts of solutions A and B

-- conditions
def proportion_A : ℝ := 0.20 -- Alcohol concentration in solution A
def proportion_B : ℝ := 0.60 -- Alcohol concentration in solution B
def final_proportion : ℝ := 0.40 -- Final alcohol concentration

-- Lean statement
theorem solution_ratio (h : 0.20 * a + 0.60 * b = 0.40 * (a + b)) : a = b := by
  sorry

end solution_ratio_l33_33013


namespace range_of_x_sq_add_y_sq_l33_33179

theorem range_of_x_sq_add_y_sq (x y : ℝ) (h : x^2 + y^2 = 4 * x) : 
  ∃ (a b : ℝ), a ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ b ∧ a = 0 ∧ b = 16 :=
by
  sorry

end range_of_x_sq_add_y_sq_l33_33179


namespace sector_angle_l33_33125

noncomputable def central_angle_of_sector (r l : ℝ) : ℝ := l / r

theorem sector_angle (r l : ℝ) (h1 : 2 * r + l = 6) (h2 : (1 / 2) * l * r = 2) :
  central_angle_of_sector r l = 1 ∨ central_angle_of_sector r l = 4 :=
by
  sorry

end sector_angle_l33_33125


namespace min_1x1_tiles_l33_33845

/-- To cover a 23x23 grid using 1x1, 2x2, and 3x3 tiles (without gaps or overlaps),
the minimum number of 1x1 tiles required is 1. -/
theorem min_1x1_tiles (a b c : ℕ) (h : a + 2 * b + 3 * c = 23 * 23) : 
  a ≥ 1 :=
sorry

end min_1x1_tiles_l33_33845


namespace subcommittee_combinations_l33_33495

open Nat

theorem subcommittee_combinations :
  (choose 8 3) * (choose 6 2) = 840 := by
  sorry

end subcommittee_combinations_l33_33495


namespace sequence_next_term_l33_33561

theorem sequence_next_term (a b c d e : ℕ) (h1 : a = 34) (h2 : b = 45) (h3 : c = 56) (h4 : d = 67) (h5 : e = 78) (h6 : b = a + 11) (h7 : c = b + 11) (h8 : d = c + 11) (h9 : e = d + 11) : e + 11 = 89 :=
by
  sorry

end sequence_next_term_l33_33561


namespace first_player_wins_game_l33_33517

theorem first_player_wins_game :
  ∀ (coins : ℕ), coins = 2019 →
  (∀ (n : ℕ), n % 2 = 1 ∧ 1 ≤ n ∧ n ≤ 99) →
  (∀ (m : ℕ), m % 2 = 0 ∧ 2 ≤ m ∧ m ≤ 100) →
  ∃ (f : ℕ → ℕ → ℕ), (∀ (c : ℕ), c <= coins → c = 0) :=
by
  sorry

end first_player_wins_game_l33_33517


namespace coloring_ways_l33_33970

-- Definitions of the problem:
def column1 := 1
def column2 := 2
def column3 := 3
def column4 := 4
def column5 := 3
def column6 := 2
def column7 := 1
def total_colors := 3 -- Blue, Yellow, Green

-- Adjacent coloring constraints:
def adjacent_constraints (c1 c2 : ℕ) : Prop := c1 ≠ c2

-- Number of ways to color figure:
theorem coloring_ways : 
  (∃ (n : ℕ), n = 2^5) ∧ 
  n = 32 :=
by 
  sorry

end coloring_ways_l33_33970


namespace min_initial_seeds_l33_33370

/-- Given conditions:
  - The farmer needs to sell at least 10,000 watermelons each year.
  - Each watermelon produces 250 seeds when used for seeds but cannot be sold if used for seeds.
  - We need to find the minimum number of initial seeds S the farmer must buy to never buy seeds again.
-/
theorem min_initial_seeds : ∃ (S : ℕ), S = 10041 ∧ ∀ (yearly_sales : ℕ), yearly_sales = 10000 →
  ∀ (seed_yield : ℕ), seed_yield = 250 →
  ∃ (x : ℕ), S = yearly_sales + x ∧ x * seed_yield ≥ S :=
sorry

end min_initial_seeds_l33_33370


namespace value_of_a2018_l33_33499

noncomputable def a : ℕ → ℝ
| 0       => 2
| (n + 1) => (1 + a n) / (1 - a n)

theorem value_of_a2018 : a 2017 = -3 := sorry

end value_of_a2018_l33_33499


namespace find_a_l33_33899

theorem find_a (f g : ℝ → ℝ) (a : ℝ) 
  (h₁ : ∀ x, f x = 2 * x / 3 + 4) 
  (h₂ : ∀ x, g x = 5 - 2 * x) 
  (h₃ : f (g a) = 7) : 
  a = 1 / 4 := 
sorry

end find_a_l33_33899


namespace number_of_special_permutations_l33_33076

noncomputable def count_special_permutations : ℕ :=
  (Nat.choose 12 6)

theorem number_of_special_permutations : count_special_permutations = 924 :=
  by
    sorry

end number_of_special_permutations_l33_33076


namespace solution_set_of_inequality_l33_33804

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
sorry

end solution_set_of_inequality_l33_33804


namespace elena_subtracts_99_to_compute_49_squared_l33_33608

noncomputable def difference_between_squares_50_49 : ℕ := 99

theorem elena_subtracts_99_to_compute_49_squared :
  ∀ (n : ℕ), n = 50 → (n - 1)^2 = n^2 - difference_between_squares_50_49 :=
by
  intro n
  sorry

end elena_subtracts_99_to_compute_49_squared_l33_33608


namespace part_I_part_II_l33_33049

noncomputable def f (x a : ℝ) := |x - 4| + |x - a|

theorem part_I (x : ℝ) : (f x 2 > 10) ↔ (x > 8 ∨ x < -2) :=
by sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≥ 1) ↔ (a ≥ 5 ∨ a ≤ 3) :=
by sorry

end part_I_part_II_l33_33049


namespace correct_number_of_arrangements_l33_33703

def arrangements_with_conditions (n : ℕ) : ℕ := 
  if n = 6 then
    let case1 := 120  -- when B is at the far right
    let case2 := 96   -- when A is at the far right
    case1 + case2
  else 0

theorem correct_number_of_arrangements : arrangements_with_conditions 6 = 216 :=
by {
  -- The detailed proof is omitted here
  sorry
}

end correct_number_of_arrangements_l33_33703


namespace ratio_initial_to_doubled_l33_33583

theorem ratio_initial_to_doubled (x : ℕ) (h : 3 * (2 * x + 5) = 105) : x / (2 * x) = 1 / 2 :=
by
  sorry

end ratio_initial_to_doubled_l33_33583


namespace rectangle_perimeter_l33_33157

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem rectangle_perimeter (x y : ℝ) (A : ℝ) (E : ℝ) (fA fB : Real) (p : ℝ) 
  (h1 : y = 2 * x)
  (h2 : x * y = 2015)
  (h3 : E = 2006 * π)
  (h4 : fA = x + y)
  (h5 : fB ^ 2 = (3 / 2)^2 * 1007.5 - (p / 2)^2)
  (h6 : 2 * (3 / 2 * sqrt 1007.5 * sqrt 1009.375) = 2006 / π) :
  2 * (x + y) = 6 * sqrt 1007.5 := 
by
  sorry

end rectangle_perimeter_l33_33157


namespace wire_length_unique_l33_33362

noncomputable def distance_increment := (5 / 3)

theorem wire_length_unique (d L : ℝ) 
  (h1 : L = 25 * d) 
  (h2 : L = 24 * (d + distance_increment)) :
  L = 1000 := by
  sorry

end wire_length_unique_l33_33362


namespace age_of_replaced_person_l33_33421

theorem age_of_replaced_person
    (T : ℕ) -- total age of the original group of 10 persons
    (age_person_replaced : ℕ) -- age of the person who was replaced
    (age_new_person : ℕ) -- age of the new person
    (h1 : age_new_person = 15)
    (h2 : (T / 10) - 3 = (T - age_person_replaced + age_new_person) / 10) :
    age_person_replaced = 45 :=
by
  sorry

end age_of_replaced_person_l33_33421


namespace willie_final_stickers_l33_33702

-- Definitions of initial stickers and given stickers
def willie_initial_stickers : ℝ := 36.0
def emily_gives : ℝ := 7.0

-- The statement to prove
theorem willie_final_stickers : willie_initial_stickers + emily_gives = 43.0 := by
  sorry

end willie_final_stickers_l33_33702


namespace percentage_change_area_l33_33021

theorem percentage_change_area
    (L B : ℝ)
    (Area_original : ℝ) (Area_new : ℝ)
    (Length_new : ℝ) (Breadth_new : ℝ) :
    Area_original = L * B →
    Length_new = L / 2 →
    Breadth_new = 3 * B →
    Area_new = Length_new * Breadth_new →
    (Area_new - Area_original) / Area_original * 100 = 50 :=
  by
  intro h_orig_area hl_new hb_new ha_new
  sorry

end percentage_change_area_l33_33021


namespace certain_number_is_1862_l33_33616

theorem certain_number_is_1862 (G N : ℕ) (hG: G = 4) (hN: ∃ k : ℕ, N = G * k + 6) (h1856: ∃ m : ℕ, 1856 = G * m + 4) : N = 1862 :=
by
  sorry

end certain_number_is_1862_l33_33616


namespace planes_parallel_or_coincide_l33_33443

-- Define normal vectors
def normal_vector_u : ℝ × ℝ × ℝ := (1, 2, -2)
def normal_vector_v : ℝ × ℝ × ℝ := (-3, -6, 6)

-- The theorem states that planes defined by these normal vectors are either 
-- parallel or coincide if their normal vectors are collinear.
theorem planes_parallel_or_coincide (u v : ℝ × ℝ × ℝ) 
  (h_u : u = normal_vector_u) 
  (h_v : v = normal_vector_v) 
  (h_collinear : v = (-3) • u) : 
    ∃ k : ℝ, v = k • u := 
by
  sorry

end planes_parallel_or_coincide_l33_33443


namespace odd_number_diff_of_squares_l33_33621

theorem odd_number_diff_of_squares (k : ℕ) : ∃ n : ℕ, k = (n+1)^2 - n^2 ↔ ∃ m : ℕ, k = 2 * m + 1 := 
by 
  sorry

end odd_number_diff_of_squares_l33_33621


namespace classify_triangles_by_angles_l33_33705

-- Define the basic types and properties for triangles and their angle classifications
def acute_triangle (α β γ : ℝ) : Prop :=
  α < 90 ∧ β < 90 ∧ γ < 90

def right_triangle (α β γ : ℝ) : Prop :=
  α = 90 ∨ β = 90 ∨ γ = 90

def obtuse_triangle (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

-- Problem: Classify triangles by angles and prove that the correct classification is as per option A
theorem classify_triangles_by_angles :
  (∀ (α β γ : ℝ), acute_triangle α β γ ∨ right_triangle α β γ ∨ obtuse_triangle α β γ) :=
sorry

end classify_triangles_by_angles_l33_33705


namespace symmetry_axis_of_function_l33_33843

noncomputable def f (varphi : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * x + varphi)

theorem symmetry_axis_of_function
  (varphi : ℝ) (h1 : |varphi| < Real.pi / 2)
  (h2 : f varphi (Real.pi / 6) = 1) :
  ∃ k : ℤ, (k * Real.pi / 2 + Real.pi / 3 = Real.pi / 3) :=
sorry

end symmetry_axis_of_function_l33_33843


namespace jimmy_max_loss_l33_33371

-- Definition of the conditions
def exam_points : ℕ := 20
def number_of_exams : ℕ := 3
def points_lost_for_behavior : ℕ := 5
def passing_score : ℕ := 50

-- Total points Jimmy has earned and lost
def total_points : ℕ := (number_of_exams * exam_points) - points_lost_for_behavior

-- The maximum points Jimmy can lose and still pass
def max_points_jimmy_can_lose : ℕ := total_points - passing_score

-- Statement to prove
theorem jimmy_max_loss : max_points_jimmy_can_lose = 5 := 
by
  sorry

end jimmy_max_loss_l33_33371


namespace two_vertical_asymptotes_l33_33775

theorem two_vertical_asymptotes (k : ℝ) : 
  (∀ x : ℝ, (x ≠ 3 ∧ x ≠ -2) → 
           (∃ δ > 0, ∀ ε > 0, ∃ x' : ℝ, x + δ > x' ∧ x' > x - δ ∧ 
                             (x' ≠ 3 ∧ x' ≠ -2) → 
                             |(x'^2 + 2 * x' + k) / (x'^2 - x' - 6)| > 1/ε)) ↔ 
  (k ≠ -15 ∧ k ≠ 0) :=
sorry

end two_vertical_asymptotes_l33_33775


namespace sarah_meals_count_l33_33270

theorem sarah_meals_count :
  let main_courses := 4
  let sides := 3
  let drinks := 2
  let desserts := 2
  main_courses * sides * drinks * desserts = 48 := 
by
  let main_courses := 4
  let sides := 3
  let drinks := 2
  let desserts := 2
  calc
    4 * 3 * 2 * 2 = 48 := sorry

end sarah_meals_count_l33_33270


namespace distance_to_convenience_store_l33_33150

def distance_work := 6
def days_work := 5
def distance_dog_walk := 2
def times_dog_walk := 2
def days_week := 7
def distance_friend_house := 1
def times_friend_visit := 1
def total_miles := 95
def trips_convenience_store := 2

theorem distance_to_convenience_store :
  ∃ x : ℝ,
    (distance_work * 2 * days_work) +
    (distance_dog_walk * times_dog_walk * days_week) +
    (distance_friend_house * 2 * times_friend_visit) +
    (x * trips_convenience_store) = total_miles
    → x = 2.5 :=
by
  sorry

end distance_to_convenience_store_l33_33150


namespace volumes_comparison_l33_33345

variable (a : ℝ) (h_a : a ≠ 3)

def volume_A := 3 * 3 * 3
def volume_B := 3 * 3 * a
def volume_C := a * a * 3
def volume_D := a * a * a

theorem volumes_comparison (h_a : a ≠ 3) :
  (volume_A + volume_D) > (volume_B + volume_C) :=
by
  have volume_A : ℝ := 27
  have volume_B := 9 * a
  have volume_C := 3 * a * a
  have volume_D := a * a * a
  sorry

end volumes_comparison_l33_33345


namespace unique_four_digit_perfect_cube_divisible_by_16_and_9_l33_33070

theorem unique_four_digit_perfect_cube_divisible_by_16_and_9 :
  ∃! n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℕ, n = k^3) ∧ n % 16 = 0 ∧ n % 9 = 0 ∧ n = 1728 :=
by sorry

end unique_four_digit_perfect_cube_divisible_by_16_and_9_l33_33070


namespace min_employees_needed_l33_33077

-- Definitions for the problem conditions
def hardware_employees : ℕ := 150
def software_employees : ℕ := 130
def both_employees : ℕ := 50

-- Statement of the proof problem
theorem min_employees_needed : hardware_employees + software_employees - both_employees = 230 := 
by 
  -- Calculation skipped with sorry
  sorry

end min_employees_needed_l33_33077


namespace find_shares_l33_33772

def shareA (B : ℝ) : ℝ := 3 * B
def shareC (B : ℝ) : ℝ := B - 25
def shareD (A B : ℝ) : ℝ := A + B - 10
def total_share (A B C D : ℝ) : ℝ := A + B + C + D

theorem find_shares :
  ∃ (A B C D : ℝ),
  A = 744.99 ∧
  B = 248.33 ∧
  C = 223.33 ∧
  D = 983.32 ∧
  A = shareA B ∧
  C = shareC B ∧
  D = shareD A B ∧
  total_share A B C D = 2200 := 
sorry

end find_shares_l33_33772


namespace find_F_58_59_60_l33_33288

def F : ℤ → ℤ → ℤ → ℝ := sorry

axiom F_scaling (a b c n : ℤ) : F (n * a) (n * b) (n * c) = n * F a b c
axiom F_shift (a b c n : ℤ) : F (a + n) (b + n) (c + n) = F a b c + n
axiom F_symmetry (a b c : ℤ) : F a b c = F c b a

theorem find_F_58_59_60 : F 58 59 60 = 59 :=
sorry

end find_F_58_59_60_l33_33288


namespace expression_value_l33_33028

theorem expression_value (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x :=
by
  -- Insert proof here
  sorry

end expression_value_l33_33028


namespace initial_points_l33_33919

theorem initial_points (n : ℕ) (h : 16 * n - 15 = 225) : n = 15 :=
sorry

end initial_points_l33_33919


namespace theresa_more_than_thrice_julia_l33_33830

-- Define the problem parameters
variable (tory julia theresa : ℕ)

def tory_videogames : ℕ := 6
def theresa_videogames : ℕ := 11

-- Define the relationships between the numbers of video games
def julia_relationship := julia = tory / 3
def theresa_compared_to_julia := theresa = theresa_videogames
def tory_value := tory = tory_videogames

theorem theresa_more_than_thrice_julia (h1 : julia_relationship tory julia) 
                                       (h2 : tory_value tory)
                                       (h3 : theresa_compared_to_julia theresa) :
  theresa - 3 * julia = 5 :=
by 
  -- Here comes the proof (not required for the task)
  sorry

end theresa_more_than_thrice_julia_l33_33830


namespace trig_identity_l33_33921

open Real

theorem trig_identity 
  (θ : ℝ)
  (h : tan (π / 4 + θ) = 3) : 
  sin (2 * θ) - 2 * cos θ ^ 2 = -3 / 4 :=
by
  sorry

end trig_identity_l33_33921


namespace pow_mod_eq_l33_33225

theorem pow_mod_eq : (6 ^ 2040) % 50 = 26 := by
  sorry

end pow_mod_eq_l33_33225


namespace arithmetic_sequence_sum_l33_33381

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {n : ℕ}

-- Conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def S9_is_90 (S : ℕ → ℝ) := S 9 = 90

-- The proof goal
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : is_arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : S9_is_90 S) :
  a 3 + a 5 + a 7 = 30 :=
by
  sorry

end arithmetic_sequence_sum_l33_33381


namespace treasure_value_l33_33352

theorem treasure_value
    (fonzie_paid : ℕ) (auntbee_paid : ℕ) (lapis_paid : ℕ)
    (lapis_share : ℚ) (lapis_received : ℕ) (total_value : ℚ)
    (h1 : fonzie_paid = 7000) 
    (h2 : auntbee_paid = 8000) 
    (h3 : lapis_paid = 9000) 
    (h4 : fonzie_paid + auntbee_paid + lapis_paid = 24000) 
    (h5 : lapis_share = lapis_paid / (fonzie_paid + auntbee_paid + lapis_paid)) 
    (h6 : lapis_received = 337500) 
    (h7 : lapis_share * total_value = lapis_received) :
  total_value = 1125000 := by
  sorry

end treasure_value_l33_33352


namespace sin_double_angle_l33_33017

theorem sin_double_angle (x : ℝ) (h : Real.sin (π / 4 - x) = 4 / 5) : Real.sin (2 * x) = -7 / 25 := 
by 
  sorry

end sin_double_angle_l33_33017


namespace find_x_l33_33851

theorem find_x (x : ℝ) (h : 0 < x) (hx : 0.01 * x * x^2 = 16) : x = 12 :=
sorry

end find_x_l33_33851


namespace possible_values_of_b_l33_33708

-- Set up the basic definitions and conditions
variable (a b c : ℝ)
variable (A B C : ℝ)

-- Assuming the conditions provided in the problem
axiom cond1 : a * (1 - Real.cos B) = b * Real.cos A
axiom cond2 : c = 3
axiom cond3 : 1 / 2 * a * c * Real.sin B = 2 * Real.sqrt 2

-- The theorem expressing the question and the correct answer
theorem possible_values_of_b : b = 2 ∨ b = 4 * Real.sqrt 2 := sorry

end possible_values_of_b_l33_33708


namespace value_standard_deviations_less_than_mean_l33_33101

-- Definitions of the given conditions
def mean : ℝ := 15
def std_dev : ℝ := 1.5
def value : ℝ := 12

-- Lean 4 statement to prove the question
theorem value_standard_deviations_less_than_mean :
  (mean - value) / std_dev = 2 := by
  sorry

end value_standard_deviations_less_than_mean_l33_33101


namespace total_cans_collected_l33_33004

def bags_on_saturday : ℕ := 6
def bags_on_sunday : ℕ := 3
def cans_per_bag : ℕ := 8
def total_cans : ℕ := 72

theorem total_cans_collected :
  (bags_on_saturday + bags_on_sunday) * cans_per_bag = total_cans :=
by
  sorry

end total_cans_collected_l33_33004


namespace range_of_a_l33_33104

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x + 1| - 2 * a + 2 < 0) → (a > 2) :=
by
  sorry

end range_of_a_l33_33104


namespace halloween_candy_l33_33838

theorem halloween_candy : 23 - 7 + 21 = 37 :=
by
  sorry

end halloween_candy_l33_33838


namespace smallest_angle_in_triangle_l33_33893

theorem smallest_angle_in_triangle (x : ℝ) 
  (h_ratio : 4 * x < 5 * x ∧ 5 * x < 9 * x) 
  (h_sum : 4 * x + 5 * x + 9 * x = 180) : 
  4 * x = 40 :=
by
  sorry

end smallest_angle_in_triangle_l33_33893


namespace final_price_correct_l33_33204

-- Definitions that follow the given conditions
def initial_price : ℝ := 150
def increase_percentage_year1 : ℝ := 1.5
def decrease_percentage_year2 : ℝ := 0.3

-- Compute intermediate values
noncomputable def price_end_year1 : ℝ := initial_price + (increase_percentage_year1 * initial_price)
noncomputable def price_end_year2 : ℝ := price_end_year1 - (decrease_percentage_year2 * price_end_year1)

-- The final theorem stating the price at the end of the second year
theorem final_price_correct : price_end_year2 = 262.5 := by
  sorry

end final_price_correct_l33_33204


namespace total_capacity_of_schools_l33_33904

theorem total_capacity_of_schools (a b c d t : ℕ) (h_a : a = 2) (h_b : b = 2) (h_c : c = 400) (h_d : d = 340) :
  t = a * c + b * d → t = 1480 := by
  intro h
  rw [h_a, h_b, h_c, h_d] at h
  simp at h
  exact h

end total_capacity_of_schools_l33_33904


namespace problem1_correctness_problem2_correctness_l33_33961

noncomputable def problem1_solution_1 (x : ℝ) : Prop := x = Real.sqrt 5 - 1
noncomputable def problem1_solution_2 (x : ℝ) : Prop := x = -Real.sqrt 5 - 1
noncomputable def problem2_solution_1 (x : ℝ) : Prop := x = 5
noncomputable def problem2_solution_2 (x : ℝ) : Prop := x = -1 / 3

theorem problem1_correctness (x : ℝ) :
  (x^2 + 2*x - 4 = 0) → (problem1_solution_1 x ∨ problem1_solution_2 x) :=
by sorry

theorem problem2_correctness (x : ℝ) :
  (3 * x * (x - 5) = 5 - x) → (problem2_solution_1 x ∨ problem2_solution_2 x) :=
by sorry

end problem1_correctness_problem2_correctness_l33_33961


namespace arithmetic_sequence_15th_term_l33_33453

theorem arithmetic_sequence_15th_term :
  let first_term := 3
  let second_term := 8
  let third_term := 13
  let common_difference := second_term - first_term
  (first_term + (15 - 1) * common_difference) = 73 :=
by
  sorry

end arithmetic_sequence_15th_term_l33_33453


namespace square_segment_ratio_l33_33533

theorem square_segment_ratio
  (A B C D E M P Q : ℝ × ℝ)
  (h_square: A = (0, 16) ∧ B = (16, 16) ∧ C = (16, 0) ∧ D = (0, 0))
  (h_E: E = (7, 0))
  (h_midpoint: M = ((0 + 7) / 2, (16 + 0) / 2))
  (h_bisector_P: P = (P.1, 16) ∧ (16 - 8 = (7 / 16) * (P.1 - 3.5)))
  (h_bisector_Q: Q = (Q.1, 0) ∧ (0 - 8 = (7 / 16) * (Q.1 - 3.5)))
  (h_PM: abs (16 - 8) = abs (P.2 - M.2))
  (h_MQ: abs (8 - 0) = abs (M.2 - Q.2)) :
  abs (P.2 - M.2) = abs (M.2 - Q.2) :=
sorry

end square_segment_ratio_l33_33533


namespace card_average_2023_l33_33339

theorem card_average_2023 (n : ℕ) (h_pos : 0 < n) (h_avg : (2 * n + 1) / 3 = 2023) : n = 3034 := by
  sorry

end card_average_2023_l33_33339


namespace time_for_one_paragraph_l33_33430

-- Definitions for the given conditions
def short_answer_time := 3 -- minutes
def essay_time := 60 -- minutes
def total_homework_time := 240 -- minutes
def essays_assigned := 2
def paragraphs_assigned := 5
def short_answers_assigned := 15

-- Function to calculate total time from given conditions
def total_time_for_essays (essays : ℕ) : ℕ :=
  essays * essay_time

def total_time_for_short_answers (short_answers : ℕ) : ℕ :=
  short_answers * short_answer_time

def total_time_for_paragraphs (paragraphs : ℕ) : ℕ :=
  total_homework_time - (total_time_for_essays essays_assigned + total_time_for_short_answers short_answers_assigned)

def time_per_paragraph (paragraphs : ℕ) : ℕ :=
  total_time_for_paragraphs paragraphs / paragraphs_assigned

-- Proving the question part
theorem time_for_one_paragraph : 
  time_per_paragraph paragraphs_assigned = 15 := by
  sorry

end time_for_one_paragraph_l33_33430


namespace smallest_rel_prime_to_180_l33_33666

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l33_33666


namespace least_number_to_subtract_l33_33663

theorem least_number_to_subtract (n d : ℕ) (n_val : n = 13602) (d_val : d = 87) : 
  ∃ r, (n - r) % d = 0 ∧ r = 30 := by
  sorry

end least_number_to_subtract_l33_33663


namespace find_divisor_l33_33336

theorem find_divisor (d : ℕ) (N : ℕ) (a b : ℕ)
  (h1 : a = 9) (h2 : b = 79) (h3 : N = 7) :
  (∃ d, (∀ k : ℕ, a ≤ k*d ∧ k*d ≤ b → (k*d) % d = 0) ∧
   ∀ count : ℕ, count = (b / d) - ((a - 1) / d) → count = N) →
  d = 11 :=
by
  sorry

end find_divisor_l33_33336


namespace total_number_of_numbers_l33_33175

theorem total_number_of_numbers (avg : ℝ) (sum1 sum2 sum3 : ℝ) (N : ℝ) :
  avg = 3.95 →
  sum1 = 2 * 3.8 →
  sum2 = 2 * 3.85 →
  sum3 = 2 * 4.200000000000001 →
  avg = (sum1 + sum2 + sum3) / N →
  N = 6 :=
by
  intros h_avg h_sum1 h_sum2 h_sum3 h_total
  sorry

end total_number_of_numbers_l33_33175


namespace part_I_min_value_part_II_nonexistence_l33_33197

theorem part_I_min_value (a b : ℝ) (hab : a > 0 ∧ b > 0 ∧ a + 4 * b = (a * b)^(3/2)) : a^2 + 16 * b^2 ≥ 32 :=
by
  sorry

theorem part_II_nonexistence (a b : ℝ) (hab : a > 0 ∧ b > 0 ∧ a + 4 * b = (a * b)^(3/2)) : ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 6 :=
by
  sorry

end part_I_min_value_part_II_nonexistence_l33_33197


namespace stream_speed_l33_33588

theorem stream_speed (c v : ℝ) (h1 : c - v = 9) (h2 : c + v = 12) : v = 1.5 :=
by
  sorry

end stream_speed_l33_33588


namespace steps_A_l33_33939

theorem steps_A (t_A t_B : ℝ) (a e t : ℝ) :
  t_A = 3 * t_B →
  t_B = t / 75 →
  a + e * t = 100 →
  75 + e * t = 100 →
  a = 75 :=
by sorry

end steps_A_l33_33939


namespace value_of_a_minus_b_l33_33413

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a - b| = a - b) : a - b = 7 ∨ a - b = 3 :=
sorry

end value_of_a_minus_b_l33_33413


namespace remainder_of_multiple_l33_33942

theorem remainder_of_multiple (m k : ℤ) (h1 : m % 5 = 2) (h2 : (2 * k) % 5 = 1) : 
  (k * m) % 5 = 1 := 
sorry

end remainder_of_multiple_l33_33942


namespace set_difference_correct_l33_33001

-- Define the sets A and B
def A : Set ℤ := {-1, 1, 3, 5, 7, 9}
def B : Set ℤ := {-1, 5, 7}

-- Define the set difference A - B
def A_minus_B : Set ℤ := {x | x ∈ A ∧ x ∉ B} -- This is the operation A - B

-- The theorem stating the required proof
theorem set_difference_correct : A_minus_B = {1, 3, 9} :=
by {
  -- Proof goes here; however, we have requested no proof, so we put sorry.
  sorry
}

end set_difference_correct_l33_33001


namespace class_mean_correct_l33_33005

noncomputable def new_class_mean (number_students_midterm : ℕ) (avg_score_midterm : ℚ)
                                 (number_students_next_day : ℕ) (avg_score_next_day : ℚ)
                                 (number_students_final_day : ℕ) (avg_score_final_day : ℚ)
                                 (total_students : ℕ) : ℚ :=
  let total_score_midterm := number_students_midterm * avg_score_midterm
  let total_score_next_day := number_students_next_day * avg_score_next_day
  let total_score_final_day := number_students_final_day * avg_score_final_day
  let total_score := total_score_midterm + total_score_next_day + total_score_final_day
  total_score / total_students

theorem class_mean_correct :
  new_class_mean 50 65 8 85 2 55 60 = 67 :=
by
  sorry

end class_mean_correct_l33_33005


namespace triangle_angle_sixty_degrees_l33_33139

theorem triangle_angle_sixty_degrees (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) : 
  ∃ (θ : ℝ), θ = 60 ∧ ∃ (a b c : ℝ), a * b * c ≠ 0 ∧ ∀ {α β γ : ℝ}, (a + b + c = α + β + γ + θ) := 
sorry

end triangle_angle_sixty_degrees_l33_33139


namespace combined_distance_correct_l33_33212

-- Define the conditions
def wheelA_rotations_per_minute := 20
def wheelA_distance_per_rotation_cm := 35
def wheelB_rotations_per_minute := 30
def wheelB_distance_per_rotation_cm := 50

-- Calculate distances in meters
def wheelA_distance_per_minute_m :=
  (wheelA_rotations_per_minute * wheelA_distance_per_rotation_cm) / 100

def wheelB_distance_per_minute_m :=
  (wheelB_rotations_per_minute * wheelB_distance_per_rotation_cm) / 100

def wheelA_distance_per_hour_m :=
  wheelA_distance_per_minute_m * 60

def wheelB_distance_per_hour_m :=
  wheelB_distance_per_minute_m * 60

def combined_distance_per_hour_m :=
  wheelA_distance_per_hour_m + wheelB_distance_per_hour_m

theorem combined_distance_correct : combined_distance_per_hour_m = 1320 := by
  -- skip the proof here with sorry
  sorry

end combined_distance_correct_l33_33212


namespace cannot_fill_box_exactly_l33_33869

def box_length : ℝ := 70
def box_width : ℝ := 40
def box_height : ℝ := 25
def cube_side : ℝ := 4.5

theorem cannot_fill_box_exactly : 
  ¬ (∃ n : ℕ, n * cube_side^3 = box_length * box_width * box_height ∧
               (∃ x y z : ℕ, x * cube_side = box_length ∧ 
                             y * cube_side = box_width ∧ 
                             z * cube_side = box_height)) :=
by sorry

end cannot_fill_box_exactly_l33_33869


namespace quadrilateral_has_four_sides_and_angles_l33_33850

-- Define the conditions based on the characteristics of a quadrilateral
def quadrilateral (sides angles : Nat) : Prop :=
  sides = 4 ∧ angles = 4

-- Statement: Verify the property of a quadrilateral
theorem quadrilateral_has_four_sides_and_angles (sides angles : Nat) (h : quadrilateral sides angles) : sides = 4 ∧ angles = 4 :=
by
  -- We provide a proof by the characteristics of a quadrilateral
  sorry

end quadrilateral_has_four_sides_and_angles_l33_33850


namespace solve_rational_eq_l33_33763

theorem solve_rational_eq {x : ℝ} (h : 1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 4 * x - 5) + 1 / (x^2 - 15 * x - 12) = 0) :
  x = 3 ∨ x = -4 ∨ x = 1 ∨ x = -5 :=
by {
  sorry
}

end solve_rational_eq_l33_33763


namespace work_rate_solution_l33_33618

theorem work_rate_solution (x : ℝ) (hA : 60 > 0) (hB : x > 0) (hTogether : 15 > 0) :
  (1 / 60 + 1 / x = 1 / 15) → (x = 20) :=
by 
  sorry -- Proof Placeholder

end work_rate_solution_l33_33618


namespace floor_sub_le_l33_33640

theorem floor_sub_le : ∀ (x y : ℝ), ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ :=
by sorry

end floor_sub_le_l33_33640


namespace total_lives_l33_33176

theorem total_lives (initial_players new_players lives_per_person : ℕ)
  (h_initial : initial_players = 8)
  (h_new : new_players = 2)
  (h_lives : lives_per_person = 6)
  : (initial_players + new_players) * lives_per_person = 60 := 
by
  sorry

end total_lives_l33_33176


namespace common_real_solution_unique_y_l33_33485

theorem common_real_solution_unique_y (x y : ℝ) 
  (h1 : x^2 + y^2 = 16) 
  (h2 : x^2 - 3 * y + 12 = 0) : 
  y = 4 :=
by
  sorry

end common_real_solution_unique_y_l33_33485


namespace Wendy_did_not_recycle_2_bags_l33_33340

theorem Wendy_did_not_recycle_2_bags (points_per_bag : ℕ) (total_bags : ℕ) (points_earned : ℕ) (did_not_recycle : ℕ) : 
  points_per_bag = 5 → 
  total_bags = 11 → 
  points_earned = 45 → 
  5 * (11 - did_not_recycle) = 45 → 
  did_not_recycle = 2 :=
by
  intros h_points_per_bag h_total_bags h_points_earned h_equation
  sorry

end Wendy_did_not_recycle_2_bags_l33_33340


namespace cost_of_scooter_l33_33550

-- Given conditions
variables (M T : ℕ)
axiom h1 : T = M + 4
axiom h2 : T = 15

-- Proof goal: The cost of the scooter is $26
theorem cost_of_scooter : M + T = 26 :=
by sorry

end cost_of_scooter_l33_33550


namespace jorge_goals_this_season_l33_33571

def jorge_goals_last_season : Nat := 156
def jorge_goals_total : Nat := 343

theorem jorge_goals_this_season :
  ∃ g_s : Nat, g_s = jorge_goals_total - jorge_goals_last_season ∧ g_s = 187 :=
by
  -- proof goes here, we use 'sorry' for now
  sorry

end jorge_goals_this_season_l33_33571


namespace min_value_of_expression_l33_33848

noncomputable def given_expression (x : ℝ) : ℝ := 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) + Real.sqrt ((2 + x)^2 + x^2)

theorem min_value_of_expression : ∃ x : ℝ, given_expression x = 6 * Real.sqrt 2 := 
by 
  use 0
  sorry

end min_value_of_expression_l33_33848


namespace revenue_ratio_l33_33335

variable (R_d : ℝ) (R_n : ℝ) (R_j : ℝ)

theorem revenue_ratio
  (nov_cond : R_n = 2 / 5 * R_d)
  (jan_cond : R_j = 1 / 2 * R_n) :
  R_d = 10 / 3 * ((R_n + R_j) / 2) := by
  -- Proof steps go here
  sorry

end revenue_ratio_l33_33335


namespace abs_gt_one_iff_square_inequality_l33_33327

theorem abs_gt_one_iff_square_inequality (x : ℝ) : |x| > 1 ↔ x^2 - 1 > 0 := 
sorry

end abs_gt_one_iff_square_inequality_l33_33327


namespace difference_of_squares_l33_33083

noncomputable def product_of_consecutive_integers (n : ℕ) := n * (n + 1)

theorem difference_of_squares (h : ∃ n : ℕ, product_of_consecutive_integers n = 2720) :
  ∃ a b : ℕ, product_of_consecutive_integers a = 2720 ∧ (b = a + 1) ∧ (b * b - a * a = 103) :=
by
  sorry

end difference_of_squares_l33_33083


namespace trader_profit_loss_l33_33647

noncomputable def profit_loss_percentage (sp1 sp2: ℝ) (gain_loss_rate1 gain_loss_rate2: ℝ) : ℝ :=
  let cp1 := sp1 / (1 + gain_loss_rate1)
  let cp2 := sp2 / (1 - gain_loss_rate2)
  let tcp := cp1 + cp2
  let tsp := sp1 + sp2
  let profit_or_loss := tsp - tcp
  profit_or_loss / tcp * 100

theorem trader_profit_loss : 
  profit_loss_percentage 325475 325475 0.15 0.15 = -2.33 := 
by 
  sorry

end trader_profit_loss_l33_33647


namespace race_probability_l33_33465

theorem race_probability (Px : ℝ) (Py : ℝ) (Pz : ℝ) 
  (h1 : Px = 1 / 6) 
  (h2 : Pz = 1 / 8) 
  (h3 : Px + Py + Pz = 0.39166666666666666) : Py = 0.1 := 
sorry

end race_probability_l33_33465


namespace sum_is_3600_l33_33143

variables (P R T : ℝ)
variables (CI SI : ℝ)

theorem sum_is_3600
  (hR : R = 10)
  (hT : T = 2)
  (hCI : CI = P * (1 + R / 100) ^ T - P)
  (hSI : SI = P * R * T / 100)
  (h_diff : CI - SI = 36) :
  P = 3600 :=
sorry

end sum_is_3600_l33_33143


namespace sin_double_angle_neg_one_l33_33726

theorem sin_double_angle_neg_one (α : ℝ) (a b : ℝ × ℝ) (h₁ : a = (1, Real.cos α)) (h₂ : b = (Real.sin α, 1)) (h₃ : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.sin (2 * α) = -1 :=
sorry

end sin_double_angle_neg_one_l33_33726


namespace remainder_when_sum_divided_by_5_l33_33826

theorem remainder_when_sum_divided_by_5 (f y : ℤ) (k m : ℤ) 
  (hf : f = 5 * k + 3) (hy : y = 5 * m + 4) : 
  (f + y) % 5 = 2 := 
by {
  sorry
}

end remainder_when_sum_divided_by_5_l33_33826


namespace find_circle_center_l33_33898

theorem find_circle_center :
  ∃ (a b : ℝ), a = 1 / 2 ∧ b = 7 / 6 ∧
  (0 - a)^2 + (1 - b)^2 = (1 - a)^2 + (1 - b)^2 ∧
  (1 - a) * 3 = b - 1 :=
by {
  sorry
}

end find_circle_center_l33_33898


namespace select_two_people_l33_33841

theorem select_two_people {n : ℕ} (h1 : n ≠ 0) (h2 : n ≥ 2) (h3 : (n - 1) ^ 2 = 25) : n = 6 :=
by
  sorry

end select_two_people_l33_33841


namespace paul_total_vertical_distance_l33_33366

def total_vertical_distance
  (n_stories : ℕ)
  (trips_per_day : ℕ)
  (days_in_week : ℕ)
  (height_per_story : ℕ)
  : ℕ :=
  let trips_per_week := trips_per_day * days_in_week
  let distance_per_trip := n_stories * height_per_story
  trips_per_week * distance_per_trip

theorem paul_total_vertical_distance :
  total_vertical_distance 5 6 7 10 = 2100 :=
by
  -- Proof is omitted.
  sorry

end paul_total_vertical_distance_l33_33366


namespace at_least_100_arcs_of_21_points_l33_33507

noncomputable def count_arcs (n : ℕ) (θ : ℝ) : ℕ :=
-- Please note this function needs to be defined appropriately, here we assume it computes the number of arcs of θ degrees or fewer between n points on a circle
sorry

theorem at_least_100_arcs_of_21_points :
  ∃ (n : ℕ), n = 21 ∧ count_arcs n (120 : ℝ) ≥ 100 :=
sorry

end at_least_100_arcs_of_21_points_l33_33507


namespace find_Q_l33_33089

variable (Q U P k : ℝ)

noncomputable def varies_directly_and_inversely : Prop :=
  P = k * (Q / U)

theorem find_Q (h : varies_directly_and_inversely Q U P k)
  (h1 : P = 6) (h2 : Q = 8) (h3 : U = 4)
  (h4 : P = 18) (h5 : U = 9) :
  Q = 54 :=
sorry

end find_Q_l33_33089


namespace absolute_diff_half_l33_33685

theorem absolute_diff_half (x y : ℝ) 
  (h : ((x + y = x - y ∧ x - y = x * y) ∨ 
       (x + y = x * y ∧ x * y = x / y) ∨ 
       (x - y = x * y ∧ x * y = x / y))
       ∧ x ≠ 0 ∧ y ≠ 0) : 
     |y| - |x| = 1 / 2 := 
sorry

end absolute_diff_half_l33_33685


namespace time_for_completion_l33_33974

noncomputable def efficiency_b : ℕ := 100

noncomputable def efficiency_a := 130

noncomputable def total_work := efficiency_a * 23

noncomputable def combined_efficiency := efficiency_a + efficiency_b

noncomputable def time_taken := total_work / combined_efficiency

theorem time_for_completion (h1 : efficiency_a = 130)
                           (h2 : efficiency_b = 100)
                           (h3 : total_work = 2990)
                           (h4 : combined_efficiency = 230) :
  time_taken = 13 := by
  sorry

end time_for_completion_l33_33974


namespace avery_shirts_count_l33_33637

theorem avery_shirts_count {S : ℕ} (h_total : S + 2 * S + S = 16) : S = 4 :=
by
  sorry

end avery_shirts_count_l33_33637


namespace whose_number_is_larger_l33_33041

theorem whose_number_is_larger
    (vasya_prod : ℕ := 4^12)
    (petya_prod : ℕ := 2^25) :
    petya_prod > vasya_prod :=
    by
    sorry

end whose_number_is_larger_l33_33041


namespace x_equals_l33_33074

variable (x y: ℝ)

theorem x_equals:
  (x / (x - 2) = (y^2 + 3 * y + 1) / (y^2 + 3 * y - 1)) → x = 2 * y^2 + 6 * y + 2 := by
  sorry

end x_equals_l33_33074


namespace percentage_of_alcohol_in_original_solution_l33_33630

noncomputable def alcohol_percentage_in_original_solution (P: ℝ) (V_original: ℝ) (V_water: ℝ) (percentage_new: ℝ): ℝ :=
  (P * V_original) / (V_original + V_water) * 100

theorem percentage_of_alcohol_in_original_solution : 
  ∀ (P: ℝ) (V_original : ℝ) (V_water : ℝ) (percentage_new : ℝ), 
  V_original = 3 → 
  V_water = 1 → 
  percentage_new = 24.75 →
  alcohol_percentage_in_original_solution P V_original V_water percentage_new = 33 := 
by
  sorry

end percentage_of_alcohol_in_original_solution_l33_33630


namespace gcd_12345_6789_eq_3_l33_33278

theorem gcd_12345_6789_eq_3 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_eq_3_l33_33278


namespace selectedParticipants_correct_l33_33859

-- Define the random number table portion used in the problem
def randomNumTable := [
  [12, 56, 85, 99, 26, 96, 96, 68, 27, 31, 05, 03, 72, 93, 15, 57, 12, 10, 14, 21, 88, 26, 49, 81, 76]
]

-- Define the conditions
def totalStudents := 247
def selectedStudentsCount := 4
def startingIndexRow := 4
def startingIndexCol := 9
def startingNumber := randomNumTable[0][8]

-- Define the expected selected participants' numbers
def expectedParticipants := [050, 121, 014, 218]

-- The Lean statement that needs to be proved
theorem selectedParticipants_correct : expectedParticipants = [050, 121, 014, 218] := by
  sorry

end selectedParticipants_correct_l33_33859


namespace ratio_of_p_to_q_l33_33544

theorem ratio_of_p_to_q (p q : ℝ) (h₁ : (p + q) / (p - q) = 4 / 3) (h₂ : p / q = r) : r = 7 :=
sorry

end ratio_of_p_to_q_l33_33544


namespace no_food_dogs_l33_33986

theorem no_food_dogs (total_dogs watermelon_liking salmon_liking chicken_liking ws_liking sc_liking wc_liking wsp_liking : ℕ) 
    (h_total : total_dogs = 100)
    (h_watermelon : watermelon_liking = 20) 
    (h_salmon : salmon_liking = 70) 
    (h_chicken : chicken_liking = 10) 
    (h_ws : ws_liking = 10) 
    (h_sc : sc_liking = 5) 
    (h_wc : wc_liking = 3) 
    (h_wsp : wsp_liking = 2) :
    (total_dogs - ((watermelon_liking - ws_liking - wc_liking + wsp_liking) + 
    (salmon_liking - ws_liking - sc_liking + wsp_liking) + 
    (chicken_liking - sc_liking - wc_liking + wsp_liking) + 
    (ws_liking - wsp_liking) + 
    (sc_liking - wsp_liking) + 
    (wc_liking - wsp_liking) + wsp_liking)) = 28 :=
  by sorry

end no_food_dogs_l33_33986


namespace calculation_result_l33_33713

theorem calculation_result :
  1500 * 451 * 0.0451 * 25 = 7627537500 :=
by
  -- Simply state without proof as instructed
  sorry

end calculation_result_l33_33713


namespace solve_quadratic_sum_l33_33936

theorem solve_quadratic_sum (a b : ℕ) (x : ℝ) (h₁ : x^2 + 10 * x = 93)
  (h₂ : x = Real.sqrt a - b) (ha_pos : 0 < a) (hb_pos : 0 < b) : a + b = 123 := by
  sorry

end solve_quadratic_sum_l33_33936


namespace Cherry_weekly_earnings_l33_33102

theorem Cherry_weekly_earnings :
  let charge_small_cargo := 2.50
  let charge_large_cargo := 4.00
  let daily_small_cargo := 4
  let daily_large_cargo := 2
  let days_in_week := 7
  let daily_earnings := (charge_small_cargo * daily_small_cargo) + (charge_large_cargo * daily_large_cargo)
  let weekly_earnings := daily_earnings * days_in_week
  weekly_earnings = 126 := sorry

end Cherry_weekly_earnings_l33_33102


namespace total_scarves_l33_33966

def total_yarns_red : ℕ := 2
def total_yarns_blue : ℕ := 6
def total_yarns_yellow : ℕ := 4
def scarves_per_yarn : ℕ := 3

theorem total_scarves : 
  (total_yarns_red * scarves_per_yarn) + 
  (total_yarns_blue * scarves_per_yarn) + 
  (total_yarns_yellow * scarves_per_yarn) = 36 := 
by
  sorry

end total_scarves_l33_33966


namespace prime_solution_l33_33795

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem prime_solution : ∀ (p q : ℕ), 
  is_prime p → is_prime q → 7 * p * q^2 + p = q^3 + 43 * p^3 + 1 → (p = 2 ∧ q = 7) :=
by
  intros p q hp hq h
  sorry

end prime_solution_l33_33795


namespace members_who_didnt_show_up_l33_33873

theorem members_who_didnt_show_up (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) 
  (h1 : total_members = 5) (h2 : points_per_member = 6) (h3 : total_points = 18) : 
  total_members - total_points / points_per_member = 2 :=
by
  sorry

end members_who_didnt_show_up_l33_33873


namespace fixed_point_through_1_neg2_l33_33923

noncomputable def fixed_point (a : ℝ) (x : ℝ) : ℝ :=
a^(x - 1) - 3

-- The statement to prove
theorem fixed_point_through_1_neg2 (a : ℝ) (h : a > 0) (h' : a ≠ 1) :
  fixed_point a 1 = -2 :=
by
  unfold fixed_point
  sorry

end fixed_point_through_1_neg2_l33_33923


namespace ratio_is_five_over_twelve_l33_33386

theorem ratio_is_five_over_twelve (a b c d : ℚ) (h1 : b = 4 * a) (h2 : d = 2 * c) :
    (a + b) / (c + d) = 5 / 12 :=
sorry

end ratio_is_five_over_twelve_l33_33386


namespace grocer_pounds_of_bananas_purchased_l33_33512

/-- 
Given:
1. The grocer purchased bananas at a rate of 3 pounds for $0.50.
2. The grocer sold the entire quantity at a rate of 4 pounds for $1.00.
3. The profit from selling the bananas was $11.00.

Prove that the number of pounds of bananas the grocer purchased is 132. 
-/
theorem grocer_pounds_of_bananas_purchased (P : ℕ) 
    (h1 : ∃ P, (3 * P / 0.5) - (4 * P / 1.0) = 11) : 
    P = 132 := 
sorry

end grocer_pounds_of_bananas_purchased_l33_33512


namespace cos_segments_ratio_proof_l33_33858

open Real

noncomputable def cos_segments_ratio := 
  let p := 5
  let q := 26
  ∀ x : ℝ, (cos x = cos 50) → (p, q) = (5, 26)

theorem cos_segments_ratio_proof : cos_segments_ratio :=
by 
  sorry

end cos_segments_ratio_proof_l33_33858


namespace new_ratio_of_partners_to_associates_l33_33960

theorem new_ratio_of_partners_to_associates
  (partners associates : ℕ)
  (rat_partners_associates : 2 * associates = 63 * partners)
  (partners_count : partners = 18)
  (add_assoc : associates + 45 = 612) :
  (partners:ℚ) / (associates + 45) = 1 / 34 :=
by
  -- Actual proof goes here
  sorry

end new_ratio_of_partners_to_associates_l33_33960


namespace total_juice_boxes_needed_l33_33855

-- Definitions for the conditions
def john_juice_per_week : Nat := 2 * 5
def john_school_weeks : Nat := 18 - 2 -- taking into account the holiday break

def samantha_juice_per_week : Nat := 1 * 5
def samantha_school_weeks : Nat := 16 - 2 -- taking into account after-school and holiday break

def heather_mon_wed_juice : Nat := 3 * 2
def heather_tue_thu_juice : Nat := 2 * 2
def heather_fri_juice : Nat := 1
def heather_juice_per_week : Nat := heather_mon_wed_juice + heather_tue_thu_juice + heather_fri_juice
def heather_school_weeks : Nat := 17 - 2 -- taking into account personal break and holiday break

-- Question and Answer in lean
theorem total_juice_boxes_needed : 
  (john_juice_per_week * john_school_weeks) + 
  (samantha_juice_per_week * samantha_school_weeks) + 
  (heather_juice_per_week * heather_school_weeks) = 395 := 
by
  sorry

end total_juice_boxes_needed_l33_33855


namespace registration_methods_l33_33244

theorem registration_methods :
  ∀ (interns : ℕ) (companies : ℕ), companies = 4 → interns = 5 → companies^interns = 1024 :=
by intros interns companies h1 h2; rw [h1, h2]; exact rfl

end registration_methods_l33_33244


namespace sum_series_eq_two_l33_33740

theorem sum_series_eq_two :
  ∑' k : Nat, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end sum_series_eq_two_l33_33740


namespace fuel_efficiency_l33_33094

noncomputable def gas_cost_per_gallon : ℝ := 4
noncomputable def money_spent_on_gas : ℝ := 42
noncomputable def miles_traveled : ℝ := 336

theorem fuel_efficiency : (miles_traveled / (money_spent_on_gas / gas_cost_per_gallon)) = 32 := by
  sorry

end fuel_efficiency_l33_33094


namespace find_value_l33_33423

variable (x y : ℝ)

def conditions (x y : ℝ) :=
  y > 2 * x ∧ 2 * x > 0 ∧ (x / y + y / x = 8)

theorem find_value (h : conditions x y) : (x + y) / (x - y) = -Real.sqrt (5 / 3) :=
sorry

end find_value_l33_33423


namespace number_of_subsets_of_set_A_l33_33256

theorem number_of_subsets_of_set_A : 
  (setOfSubsets : Finset (Finset ℕ)) = Finset.powerset {2, 4, 5} → 
  setOfSubsets.card = 8 :=
by
  sorry

end number_of_subsets_of_set_A_l33_33256


namespace evaluate_expression_l33_33390

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := 
by 
  sorry

end evaluate_expression_l33_33390


namespace sufficient_but_not_necessary_condition_l33_33428

-- Step d: Lean 4 statement
theorem sufficient_but_not_necessary_condition 
  (m n : ℕ) (e : ℚ) (h₁ : m = 5) (h₂ : n = 4) (h₃ : e = 3 / 5)
  (ellipse_eq : ∀ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1) :
  (m = 5 ∧ n = 4) → (e = 3 / 5) ∧ (¬(e = 3 / 5 → m = 5 ∧ n = 4)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l33_33428


namespace evaluate_product_l33_33964

theorem evaluate_product (m : ℕ) (h : m = 3) : (m - 2) * (m - 1) * m * (m + 1) * (m + 2) * (m + 3) = 720 :=
by {
  sorry
}

end evaluate_product_l33_33964


namespace depth_of_melted_sauce_l33_33262

theorem depth_of_melted_sauce
  (r_sphere : ℝ) (r_cylinder : ℝ) (h_cylinder : ℝ) (volume_conserved : Bool) :
  r_sphere = 3 ∧ r_cylinder = 10 ∧ volume_conserved → h_cylinder = 9/25 :=
by
  -- Explanation of the condition: 
  -- r_sphere is the radius of the original spherical globe (3 inches)
  -- r_cylinder is the radius of the cylindrical puddle (10 inches)
  -- h_cylinder is the depth we need to prove is 9/25 inches
  -- volume_conserved indicates that the volume is conserved
  sorry

end depth_of_melted_sauce_l33_33262


namespace min_paint_steps_l33_33938

-- Checkered square of size 2021x2021 where all cells initially white.
-- Ivan selects two cells and paints them black.
-- Cells with at least one black neighbor by side are painted black simultaneously each step.

-- Define a function to represent the steps required to paint the square black
noncomputable def min_steps_to_paint_black (n : ℕ) (a b : ℕ × ℕ) : ℕ :=
  sorry -- Placeholder for the actual function definition, as we're focusing on the statement.

-- Define the specific instance of the problem
def square_size := 2021
def initial_cells := ((505, 1010), (1515, 1010))

-- Theorem statement: Proving the minimal number of steps required is 1515
theorem min_paint_steps : min_steps_to_paint_black square_size initial_cells.1 initial_cells.2 = 1515 :=
sorry

end min_paint_steps_l33_33938


namespace reduced_flow_rate_is_correct_l33_33243

-- Define the original flow rate
def original_flow_rate : ℝ := 5.0

-- Define the function for the reduced flow rate
def reduced_flow_rate (x : ℝ) : ℝ := 0.6 * x - 1

-- Prove that the reduced flow rate is 2.0 gallons per minute
theorem reduced_flow_rate_is_correct : reduced_flow_rate original_flow_rate = 2.0 := by
  sorry

end reduced_flow_rate_is_correct_l33_33243


namespace rahul_savings_l33_33348

variable (NSC PPF total_savings : ℕ)

theorem rahul_savings (h1 : NSC / 3 = PPF / 2) (h2 : PPF = 72000) : total_savings = 180000 :=
by
  sorry

end rahul_savings_l33_33348


namespace simplify_and_evaluate_expression_l33_33457

-- Define the parameters for m and n.
def m : ℚ := -1 / 3
def n : ℚ := 1 / 2

-- Define the expression to simplify and evaluate.
def complex_expr (m n : ℚ) : ℚ :=
  -2 * (m * n - 3 * m^2) + 3 * (2 * m * n - 5 * m^2)

-- State the theorem that proves the expression equals -5/3.
theorem simplify_and_evaluate_expression :
  complex_expr m n = -5 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l33_33457


namespace no_integer_solution_system_l33_33304

theorem no_integer_solution_system (
  x y z : ℤ
) : x^6 + x^3 + x^3 * y + y ≠ 147 ^ 137 ∨ x^3 + x^3 * y + y^2 + y + z^9 ≠ 157 ^ 117 :=
by
  sorry

end no_integer_solution_system_l33_33304


namespace reduced_price_l33_33782

theorem reduced_price (P R : ℝ) (Q : ℝ) (h₁ : R = 0.80 * P) 
                      (h₂ : 800 = Q * P) 
                      (h₃ : 800 = (Q + 5) * R) 
                      : R = 32 :=
by
  -- Code that proves the theorem goes here.
  sorry

end reduced_price_l33_33782


namespace hyperbola_intersection_l33_33562

theorem hyperbola_intersection (b : ℝ) (h₁ : b > 0) :
  (b > 1) → (∀ x y : ℝ, ((x + 3 * y - 1 = 0) → ( ∃ x y : ℝ, (x^2 / 4 - y^2 / b^2 = 1) ∧ (x + 3 * y - 1 = 0))))
  :=
  sorry

end hyperbola_intersection_l33_33562


namespace price_range_of_book_l33_33906

variable (x : ℝ)

theorem price_range_of_book (h₁ : ¬(x ≥ 15)) (h₂ : ¬(x ≤ 12)) (h₃ : ¬(x ≤ 10)) : 12 < x ∧ x < 15 := 
by
  sorry

end price_range_of_book_l33_33906


namespace percentage_students_camping_trip_l33_33897

theorem percentage_students_camping_trip 
  (total_students : ℝ)
  (camping_trip_with_more_than_100 : ℝ) 
  (camping_trip_without_more_than_100_ratio : ℝ) :
  camping_trip_with_more_than_100 / (camping_trip_with_more_than_100 / 0.25) = 0.8 :=
by
  sorry

end percentage_students_camping_trip_l33_33897


namespace sum_of_angles_is_290_l33_33012

-- Given conditions
def angle_A : ℝ := 40
def angle_C : ℝ := 70
def angle_D : ℝ := 50
def angle_F : ℝ := 60

-- Calculate angle B (which is same as angle E)
def angle_B : ℝ := 180 - angle_A - angle_C
def angle_E := angle_B  -- by the condition that B and E are identical

-- Total sum of angles
def total_angle_sum : ℝ := angle_A + angle_B + angle_C + angle_D + angle_F

-- Theorem statement
theorem sum_of_angles_is_290 : total_angle_sum = 290 := by
  sorry

end sum_of_angles_is_290_l33_33012


namespace number_of_possible_routes_l33_33389

def f (x y : ℕ) : ℕ :=
  if y = 2 then sorry else sorry -- Here you need the exact definition of f(x, y)

theorem number_of_possible_routes (n : ℕ) (h : n > 0) : 
  f n 2 = (1 / 2 : ℚ) * (n^2 + 3 * n + 2) := 
by 
  sorry

end number_of_possible_routes_l33_33389


namespace solve_inequality_l33_33600

theorem solve_inequality (x : ℝ) (h : x ≠ -2 / 3) :
  3 - (1 / (3 * x + 2)) < 5 ↔ (x < -7 / 6 ∨ x > -2 / 3) := by
  sorry

end solve_inequality_l33_33600


namespace jake_later_than_austin_by_20_seconds_l33_33388

theorem jake_later_than_austin_by_20_seconds :
  (9 * 30) / 3 - 60 = 20 :=
by
  sorry

end jake_later_than_austin_by_20_seconds_l33_33388


namespace compare_abc_l33_33394

noncomputable def a : ℝ := Real.exp 0.25
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := -4 * Real.log 0.75

theorem compare_abc : b < c ∧ c < a := by
  -- Additional proof steps would follow here
  sorry

end compare_abc_l33_33394


namespace crit_value_expr_l33_33880

theorem crit_value_expr : 
  ∃ x : ℝ, -4 < x ∧ x < 1 ∧ (x^2 - 2*x + 2) / (2*x - 2) = -1 :=
sorry

end crit_value_expr_l33_33880


namespace pipe_fill_time_l33_33328

variable (t : ℝ)

theorem pipe_fill_time (h1 : 0 < t) (h2 : 0 < t / 5) (h3 : (1 / t) + (5 / t) = 1 / 5) : t = 30 :=
by
  sorry

end pipe_fill_time_l33_33328


namespace solution_set_inequality_l33_33460

theorem solution_set_inequality (m : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.exp x + Real.exp (-x))
  (h2 : ∀ x, f (-x) = f x) (h3 : ∀ x, 0 ≤ x → ∀ y, 0 ≤ y → x ≤ y → f x ≤ f y) :
  (f (2 * m) > f (m - 2)) ↔ (m > (2 / 3) ∨ m < -2) :=
  sorry

end solution_set_inequality_l33_33460


namespace find_m_eq_2_l33_33572

theorem find_m_eq_2 (a m : ℝ) (h1 : a > 0) (h2 : -a * m^2 + 2 * a * m + 3 = 3) (h3 : m ≠ 0) : m = 2 :=
by
  sorry

end find_m_eq_2_l33_33572


namespace problem_statement_l33_33642

variable (a b c : ℝ)

-- Conditions given in the problem
axiom h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -24
axiom h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8

-- The Lean statement for the proof problem
theorem problem_statement (a b c : ℝ) (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -24)
    (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8) :
    (b / (a + b) + c / (b + c) + a / (c + a)) = 19 / 2 :=
sorry

end problem_statement_l33_33642


namespace divisibility_by_7_l33_33972

theorem divisibility_by_7 (A X : Nat) (h1 : A < 10) (h2 : X < 10) : (100001 * A + 100010 * X) % 7 = 0 := 
by
  sorry

end divisibility_by_7_l33_33972


namespace find_m_l33_33867

theorem find_m (m : ℝ) (A : Set ℝ) (hA : A = {0, m, m^2 - 3 * m + 2}) (h2 : 2 ∈ A) : m = 3 :=
  sorry

end find_m_l33_33867


namespace value_of_a_l33_33761

theorem value_of_a (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {a, a^2}) (hB : B = {1, b}) (hAB : A = B) : a = -1 := 
by 
  sorry

end value_of_a_l33_33761


namespace simplify_expression_correct_l33_33019

def simplify_expression (x : ℝ) : Prop :=
  (5 - 2 * x) - (7 + 3 * x) = -2 - 5 * x

theorem simplify_expression_correct (x : ℝ) : simplify_expression x :=
  by
    sorry

end simplify_expression_correct_l33_33019


namespace xiao_ying_performance_l33_33954

def regular_weight : ℝ := 0.20
def midterm_weight : ℝ := 0.30
def final_weight : ℝ := 0.50

def regular_score : ℝ := 85
def midterm_score : ℝ := 90
def final_score : ℝ := 92

-- Define the function that calculates the weighted average
def semester_performance (rw mw fw rs ms fs : ℝ) : ℝ :=
  rw * rs + mw * ms + fw * fs

-- The theorem that the weighted average of the scores is 90
theorem xiao_ying_performance : semester_performance regular_weight midterm_weight final_weight regular_score midterm_score final_score = 90 := by
  sorry

end xiao_ying_performance_l33_33954


namespace symmetric_points_power_l33_33259

variables (m n : ℝ)

def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetric_points_power 
  (h : symmetric_y_axis (m, 3) (4, n)) : 
  (m + n) ^ 2023 = -1 :=
by 
  sorry

end symmetric_points_power_l33_33259


namespace car_kilometers_per_gallon_l33_33355

-- Define the given conditions as assumptions
variable (total_distance : ℝ) (total_gallons : ℝ)
-- Assume the given conditions
axiom h1 : total_distance = 180
axiom h2 : total_gallons = 4.5

-- The statement to be proven
theorem car_kilometers_per_gallon : (total_distance / total_gallons) = 40 :=
by
  -- Sorry is used to skip the proof
  sorry

end car_kilometers_per_gallon_l33_33355


namespace range_of_a_l33_33329

variable (a : ℝ)
def is_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0
def z : ℂ := 4 - 2 * Complex.I

theorem range_of_a (ha : is_second_quadrant ((z + a * Complex.I) ^ 2)) : a > 6 := by
  sorry

end range_of_a_l33_33329


namespace monotonicity_and_extremes_l33_33467

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 + x^2 - 3 * x + 1

theorem monotonicity_and_extremes :
  (∀ x, f x > f (-3) ∨ f x < f (-3)) ∧
  (∀ x, f x > f 1 ∨ f x < f 1) ∧
  (∀ x, (x < -3 → (∀ y, y < x → f y < f x)) ∧ (x > 1 → (∀ y, y > x → f y < f x))) ∧
  f (-3) = 10 ∧ f 1 = -(2 / 3) :=
sorry

end monotonicity_and_extremes_l33_33467


namespace decreasing_function_l33_33406

def f (a x : ℝ) : ℝ := a * x^3 - x

theorem decreasing_function (a : ℝ) 
  (h : ∀ x y : ℝ, x < y → f a y ≤ f a x) : a ≤ 0 :=
by
  sorry

end decreasing_function_l33_33406


namespace cookies_per_batch_l33_33717

def family_size := 4
def chips_per_person := 18
def chips_per_cookie := 2
def batches := 3

theorem cookies_per_batch : (family_size * chips_per_person) / chips_per_cookie / batches = 12 := 
by
  -- Proof will go here
  sorry

end cookies_per_batch_l33_33717


namespace find_f_inv_difference_l33_33636

axiom f : ℤ → ℤ
axiom f_inv : ℤ → ℤ
axiom f_has_inverse : ∀ x : ℤ, f_inv (f x) = x ∧ f (f_inv x) = x
axiom f_inverse_conditions : ∀ x : ℤ, f (x + 2) = f_inv (x - 1)

theorem find_f_inv_difference :
  f_inv 2004 - f_inv 1 = 4006 :=
sorry

end find_f_inv_difference_l33_33636


namespace problem1_problem2_l33_33056

def f (x a : ℝ) := |x - 1| + |x - a|

/-
  Problem 1:
  Prove that if a = 3, the solution set to the inequality f(x) ≥ 4 is 
  {x | x ≤ 0 ∨ x ≥ 4}.
-/
theorem problem1 (f : ℝ → ℝ → ℝ) (a : ℝ) (h : a = 3) : 
  {x : ℝ | f x a ≥ 4} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := 
sorry

/-
  Problem 2:
  Prove that for any x₁ ∈ ℝ, if f(x₁) ≥ 2 holds true, the range of values for
  a is {a | a ≥ 3 ∨ a ≤ -1}.
-/
theorem problem2 (f : ℝ → ℝ → ℝ) (x₁ : ℝ) :
  (∀ x₁ : ℝ, f x₁ a ≥ 2) ↔ (a ≥ 3 ∨ a ≤ -1) :=
sorry

end problem1_problem2_l33_33056


namespace total_pets_count_l33_33235

/-- Taylor and his six friends have a total of 45 pets, given the specified conditions about the number of each type of pet they have. -/
theorem total_pets_count
  (Taylor_cats : ℕ := 4)
  (Friend1_pets : ℕ := 8 * 3)
  (Friend2_dogs : ℕ := 3)
  (Friend2_birds : ℕ := 1)
  (Friend3_dogs : ℕ := 5)
  (Friend3_cats : ℕ := 2)
  (Friend4_reptiles : ℕ := 2)
  (Friend4_birds : ℕ := 3)
  (Friend4_cats : ℕ := 1) :
  Taylor_cats + Friend1_pets + Friend2_dogs + Friend2_birds + Friend3_dogs + Friend3_cats + Friend4_reptiles + Friend4_birds + Friend4_cats = 45 :=
sorry

end total_pets_count_l33_33235


namespace intersection_complement_M_N_l33_33773

def M := { x : ℝ | x ≤ 1 / 2 }
def N := { x : ℝ | x^2 ≤ 1 }
def complement_M := { x : ℝ | x > 1 / 2 }

theorem intersection_complement_M_N :
  (complement_M ∩ N = { x : ℝ | 1 / 2 < x ∧ x ≤ 1 }) :=
by
  sorry

end intersection_complement_M_N_l33_33773


namespace n_is_perfect_square_l33_33953

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k ^ 2

theorem n_is_perfect_square (a b c d : ℤ) (h : a + b + c + d = 0) : 
  is_perfect_square ((ab - cd) * (bc - ad) * (ca - bd)) := 
  sorry

end n_is_perfect_square_l33_33953


namespace johns_original_earnings_l33_33331

def JohnsEarningsBeforeRaise (currentEarnings: ℝ) (percentageIncrease: ℝ) := 
  ∀ x, currentEarnings = x + x * percentageIncrease → x = 50

theorem johns_original_earnings : 
  JohnsEarningsBeforeRaise 80 0.60 :=
by
  intro x
  intro h
  sorry

end johns_original_earnings_l33_33331


namespace variation_of_powers_l33_33415

theorem variation_of_powers (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
by
  sorry

end variation_of_powers_l33_33415


namespace total_numbers_is_eight_l33_33337

theorem total_numbers_is_eight
  (avg_all : ∀ n : ℕ, (total_sum : ℝ) / n = 25)
  (avg_first_two : ∀ a₁ a₂ : ℝ, (a₁ + a₂) / 2 = 20)
  (avg_next_three : ∀ a₃ a₄ a₅ : ℝ, (a₃ + a₄ + a₅) / 3 = 26)
  (h_sixth : ∀ a₆ a₇ a₈ : ℝ, a₆ + 4 = a₇ ∧ a₆ + 6 = a₈)
  (last_num : ∀ a₈ : ℝ, a₈ = 30) :
  ∃ n : ℕ, n = 8 :=
by
  sorry

end total_numbers_is_eight_l33_33337


namespace alice_lawn_area_l33_33748

theorem alice_lawn_area (posts : ℕ) (distance : ℕ) (ratio : ℕ) : 
    posts = 24 → distance = 5 → ratio = 3 → 
    ∃ (short_side long_side : ℕ), 
        (2 * (short_side + long_side - 2) = posts) ∧
        (long_side = ratio * short_side) ∧
        (distance * (short_side - 1) * distance * (long_side - 1) = 825) :=
by
  intros h_posts h_distance h_ratio
  sorry

end alice_lawn_area_l33_33748


namespace max_n_is_2_l33_33063

def is_prime_seq (q : ℕ → ℕ) : Prop :=
  ∀ i, Nat.Prime (q i)

def gen_seq (q0 : ℕ) : ℕ → ℕ
  | 0 => q0
  | (i + 1) => (gen_seq q0 i - 1)^3 + 3

theorem max_n_is_2 (q0 : ℕ) (hq0 : q0 > 0) :
  ∀ (q1 q2 : ℕ), q1 = gen_seq q0 1 → q2 = gen_seq q0 2 → 
  is_prime_seq (gen_seq q0) → q2 = (q1 - 1)^3 + 3 := 
  sorry

end max_n_is_2_l33_33063


namespace find_a_l33_33120

theorem find_a (a : ℝ) (h1 : a + 3 > 0) (h2 : abs (a + 3) = 5) : a = 2 := 
by
  sorry

end find_a_l33_33120


namespace total_animals_correct_l33_33119

def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

def added_cows : ℕ := 3
def added_pigs : ℕ := 5
def added_goats : ℕ := 2

def total_cows : ℕ := initial_cows + added_cows
def total_pigs : ℕ := initial_pigs + added_pigs
def total_goats : ℕ := initial_goats + added_goats

def total_animals : ℕ := total_cows + total_pigs + total_goats

theorem total_animals_correct : total_animals = 21 := by
  sorry

end total_animals_correct_l33_33119


namespace ratio_of_area_to_breadth_is_15_l33_33229

-- Definitions for our problem
def breadth := 5
def length := 15 -- since l - b = 10 and b = 5

-- Given conditions
axiom area_is_ktimes_breadth (k : ℝ) : length * breadth = k * breadth
axiom length_breadth_difference : length - breadth = 10

-- The proof statement
theorem ratio_of_area_to_breadth_is_15 : (length * breadth) / breadth = 15 := by
  sorry

end ratio_of_area_to_breadth_is_15_l33_33229


namespace box_base_length_max_l33_33434

noncomputable def V (x : ℝ) := x^2 * ((60 - x) / 2)

theorem box_base_length_max 
  (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 60)
  (h3 : ∀ y : ℝ, 0 < y ∧ y < 60 → V x ≥ V y)
  : x = 40 :=
sorry

end box_base_length_max_l33_33434


namespace complement_of_intersection_l33_33000

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 4, 5} := by
  sorry

end complement_of_intersection_l33_33000


namespace transformation_correct_l33_33864

variables {x y : ℝ}

theorem transformation_correct (h : x = y) : x - 2 = y - 2 := by
  sorry

end transformation_correct_l33_33864


namespace parallelogram_area_l33_33052

theorem parallelogram_area (base height : ℕ) (h_base : base = 36) (h_height : height = 24) : base * height = 864 := by
  sorry

end parallelogram_area_l33_33052


namespace cubic_roots_l33_33249

theorem cubic_roots (a x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ = 1) (h₂ : x₂ = 1) (h₃ : x₃ = a)
  (cond : (2 / x₁) + (2 / x₂) = (3 / x₃)) :
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = a ∧ (a = 2 ∨ a = 3 / 4)) :=
by
  sorry

end cubic_roots_l33_33249


namespace sheila_will_attend_picnic_l33_33667

noncomputable def prob_sheila_attends_picnic (P_Rain P_Attend_if_Rain P_Attend_if_Sunny P_Special : ℝ) : ℝ :=
  let P_Sunny := 1 - P_Rain
  let P_Rain_and_Attend := P_Rain * P_Attend_if_Rain
  let P_Sunny_and_Attend := P_Sunny * P_Attend_if_Sunny
  let P_Attends := P_Rain_and_Attend + P_Sunny_and_Attend + P_Special - P_Rain_and_Attend * P_Special - P_Sunny_and_Attend * P_Special
  P_Attends

theorem sheila_will_attend_picnic :
  prob_sheila_attends_picnic 0.3 0.25 0.7 0.15 = 0.63025 :=
by
  sorry

end sheila_will_attend_picnic_l33_33667


namespace A_finishes_remaining_work_in_2_days_l33_33276

/-- 
Given that A's daily work rate is 1/6 of the work and B's daily work rate is 1/15 of the work,
and B has already completed 2/3 of the work, 
prove that A can finish the remaining work in 2 days.
-/
theorem A_finishes_remaining_work_in_2_days :
  let A_work_rate := (1 : ℝ) / 6
  let B_work_rate := (1 : ℝ) / 15
  let B_work_in_10_days := (10 : ℝ) * B_work_rate
  let remaining_work := (1 : ℝ) - B_work_in_10_days
  let days_for_A := remaining_work / A_work_rate
  B_work_in_10_days = 2 / 3 → 
  remaining_work = 1 / 3 → 
  days_for_A = 2 :=
by
  sorry

end A_finishes_remaining_work_in_2_days_l33_33276


namespace sugar_in_first_combination_l33_33399

def cost_per_pound : ℝ := 0.45
def cost_combination_1 (S : ℝ) : ℝ := cost_per_pound * S + cost_per_pound * 16
def cost_combination_2 : ℝ := cost_per_pound * 30 + cost_per_pound * 25
def total_weight_combination_2 : ℕ := 30 + 25
def total_weight_combination_1 (S : ℕ) : ℕ := S + 16

theorem sugar_in_first_combination :
  ∀ (S : ℕ), cost_combination_1 S = 26 ∧ cost_combination_2 = 26 → total_weight_combination_1 S = total_weight_combination_2 → S = 39 :=
by sorry

end sugar_in_first_combination_l33_33399


namespace man_son_age_ratio_is_two_to_one_l33_33411

-- Define the present age of the son
def son_present_age := 33

-- Define the present age of the man
def man_present_age := son_present_age + 35

-- Define the son's age in two years
def son_age_in_two_years := son_present_age + 2

-- Define the man's age in two years
def man_age_in_two_years := man_present_age + 2

-- Define the expected ratio of the man's age to son's age in two years
def ratio := man_age_in_two_years / son_age_in_two_years

-- Theorem statement verifying the ratio
theorem man_son_age_ratio_is_two_to_one : ratio = 2 := by
  -- Note: Proof not required, so we use sorry to denote the missing proof
  sorry

end man_son_age_ratio_is_two_to_one_l33_33411


namespace red_balls_approximation_l33_33594

def total_balls : ℕ := 50
def red_ball_probability : ℚ := 7 / 10

theorem red_balls_approximation (r : ℕ)
  (h1 : total_balls = 50)
  (h2 : red_ball_probability = 0.7) :
  r = 35 := by
  sorry

end red_balls_approximation_l33_33594


namespace largest_square_factor_of_1800_l33_33565

theorem largest_square_factor_of_1800 : 
  ∃ n, n^2 ∣ 1800 ∧ ∀ m, m^2 ∣ 1800 → m^2 ≤ n^2 :=
sorry

end largest_square_factor_of_1800_l33_33565


namespace slope_of_line_l33_33408

theorem slope_of_line (x1 y1 x2 y2 : ℝ)
  (h1 : 4 * y1 + 6 * x1 = 0)
  (h2 : 4 * y2 + 6 * x2 = 0)
  (h1x2 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by sorry

end slope_of_line_l33_33408


namespace simplify_expression_l33_33469

theorem simplify_expression : 
  let a := (3 + 2 : ℚ)
  let b := a⁻¹ + 2
  let c := b⁻¹ + 2
  let d := c⁻¹ + 2
  d = 65 / 27 := by
  sorry

end simplify_expression_l33_33469


namespace sahil_selling_price_correct_l33_33107

-- Define the conditions as constants
def cost_of_machine : ℕ := 13000
def cost_of_repair : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

-- Define the total cost calculation
def total_cost : ℕ := cost_of_machine + cost_of_repair + transportation_charges

-- Define the profit calculation
def profit : ℕ := total_cost * profit_percentage / 100

-- Define the selling price calculation
def selling_price : ℕ := total_cost + profit

-- Now we express our proof problem
theorem sahil_selling_price_correct :
  selling_price = 28500 := by
  -- sorries to skip the proof.
  sorry

end sahil_selling_price_correct_l33_33107


namespace irreducible_fraction_iff_not_congruent_mod_5_l33_33885

theorem irreducible_fraction_iff_not_congruent_mod_5 (n : ℕ) : 
  (Nat.gcd (21 * n + 4) (14 * n + 1) = 1) ↔ (n % 5 ≠ 1) := 
by 
  sorry

end irreducible_fraction_iff_not_congruent_mod_5_l33_33885


namespace seq_a6_l33_33473

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 3 * a n - 2

theorem seq_a6 (a : ℕ → ℕ) (h : seq a) : a 6 = 1 :=
by
  sorry

end seq_a6_l33_33473


namespace train_speed_l33_33793

theorem train_speed (L : ℝ) (T : ℝ) (L_pos : 0 < L) (T_pos : 0 < T) (L_eq : L = 150) (T_eq : T = 3) : L / T = 50 := by
  sorry

end train_speed_l33_33793


namespace smallest_x_correct_l33_33731

noncomputable def smallest_x (K : ℤ) : ℤ := 135000

theorem smallest_x_correct (K : ℤ) :
  (∃ x : ℤ, 180 * x = K ^ 5 ∧ x > 0) → smallest_x K = 135000 :=
by
  sorry

end smallest_x_correct_l33_33731


namespace intersection_M_N_l33_33785

def set_M : Set ℝ := { x : ℝ | -3 ≤ x ∧ x < 4 }
def set_N : Set ℝ := { x : ℝ | x^2 - 2 * x - 8 ≤ 0 }

theorem intersection_M_N : (set_M ∩ set_N) = { x : ℝ | -2 ≤ x ∧ x < 4 } :=
sorry

end intersection_M_N_l33_33785


namespace intervals_of_increase_of_f_l33_33294

theorem intervals_of_increase_of_f :
  ∀ k : ℤ,
  ∀ x y : ℝ,
  k * π - (5 / 8) * π ≤ x ∧ x ≤ y ∧ y ≤ k * π - (1 / 8) * π →
  3 * Real.sin ((π / 4) - 2 * x) - 2 ≤ 3 * Real.sin ((π / 4) - 2 * y) - 2 :=
by
  sorry

end intervals_of_increase_of_f_l33_33294


namespace find_positive_integers_l33_33034

theorem find_positive_integers 
    (a b : ℕ) 
    (ha : a > 0) 
    (hb : b > 0) 
    (h1 : ∃ k1 : ℤ, (a^3 * b - 1) = k1 * (a + 1))
    (h2 : ∃ k2 : ℤ, (b^3 * a + 1) = k2 * (b - 1)) : 
    (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
sorry

end find_positive_integers_l33_33034


namespace door_solution_l33_33159

def door_problem (x : ℝ) : Prop :=
  let w := x - 4
  let h := x - 2
  let diagonal := x
  (diagonal ^ 2 - (h) ^ 2 = (w) ^ 2)

theorem door_solution (x : ℝ) : door_problem x :=
  sorry

end door_solution_l33_33159


namespace Luke_trips_l33_33475

variable (carries : Nat) (table1 : Nat) (table2 : Nat)

theorem Luke_trips (h1 : carries = 4) (h2 : table1 = 20) (h3 : table2 = 16) : 
  (table1 / carries + table2 / carries) = 9 :=
by
  sorry

end Luke_trips_l33_33475


namespace stratified_sampling_correct_l33_33627

-- Defining the conditions
def first_grade_students : ℕ := 600
def second_grade_students : ℕ := 680
def third_grade_students : ℕ := 720
def total_sample_size : ℕ := 50
def total_students := first_grade_students + second_grade_students + third_grade_students

-- Expected number of students to be sampled from first, second, and third grades
def expected_first_grade_sample := total_sample_size * first_grade_students / total_students
def expected_second_grade_sample := total_sample_size * second_grade_students / total_students
def expected_third_grade_sample := total_sample_size * third_grade_students / total_students

-- Main theorem statement
theorem stratified_sampling_correct :
  expected_first_grade_sample = 15 ∧
  expected_second_grade_sample = 17 ∧
  expected_third_grade_sample = 18 := by
  sorry

end stratified_sampling_correct_l33_33627


namespace ratio_of_screws_l33_33952

def initial_screws : Nat := 8
def total_required_screws : Nat := 4 * 6
def screws_to_buy : Nat := total_required_screws - initial_screws

theorem ratio_of_screws :
  (screws_to_buy : ℚ) / initial_screws = 2 :=
by
  simp [initial_screws, total_required_screws, screws_to_buy]
  sorry

end ratio_of_screws_l33_33952


namespace women_to_total_population_ratio_l33_33557

/-- original population of Salem -/
def original_population (pop_leesburg : ℕ) : ℕ := 15 * pop_leesburg

/-- new population after people moved out -/
def new_population (orig_pop : ℕ) (moved_out : ℕ) : ℕ := orig_pop - moved_out

/-- ratio of two numbers -/
def ratio (num : ℕ) (denom : ℕ) : ℚ := num / denom

/-- population data -/
structure PopulationData :=
  (pop_leesburg : ℕ)
  (moved_out : ℕ)
  (women : ℕ)

/-- prove ratio of women to the total population in Salem -/
theorem women_to_total_population_ratio (data : PopulationData)
  (pop_leesburg_eq : data.pop_leesburg = 58940)
  (moved_out_eq : data.moved_out = 130000)
  (women_eq : data.women = 377050) : 
  ratio data.women (new_population (original_population data.pop_leesburg) data.moved_out) = 377050 / 754100 :=
by
  sorry

end women_to_total_population_ratio_l33_33557


namespace total_tickets_sold_l33_33489

theorem total_tickets_sold 
  (ticket_price : ℕ) 
  (discount_40_percent : ℕ → ℕ) 
  (discount_15_percent : ℕ → ℕ) 
  (revenue : ℕ) 
  (people_10_discount_40 : ℕ) 
  (people_20_discount_15 : ℕ) 
  (people_full_price : ℕ)
  (h_ticket_price : ticket_price = 20)
  (h_discount_40 : ∀ n, discount_40_percent n = n * 12)
  (h_discount_15 : ∀ n, discount_15_percent n = n * 17)
  (h_revenue : revenue = 760)
  (h_people_10_discount_40 : people_10_discount_40 = 10)
  (h_people_20_discount_15 : people_20_discount_15 = 20)
  (h_people_full_price : people_full_price * ticket_price = 300) :
  (people_10_discount_40 + people_20_discount_15 + people_full_price = 45) :=
by
  sorry

end total_tickets_sold_l33_33489


namespace ways_to_write_1800_as_sum_of_4s_and_5s_l33_33728

theorem ways_to_write_1800_as_sum_of_4s_and_5s : 
  ∃ S : Finset (ℕ × ℕ), S.card = 91 ∧ ∀ (nm : ℕ × ℕ), nm ∈ S ↔ 4 * nm.1 + 5 * nm.2 = 1800 ∧ nm.1 ≥ 0 ∧ nm.2 ≥ 0 :=
by
  sorry

end ways_to_write_1800_as_sum_of_4s_and_5s_l33_33728


namespace green_hat_cost_l33_33724

theorem green_hat_cost (G : ℝ) (total_hats : ℕ) (blue_hats : ℕ) (green_hats : ℕ) (blue_cost : ℝ) (total_cost : ℝ) 
    (h₁ : blue_hats = 85) (h₂ : blue_cost = 6) (h₃ : green_hats = 90) (h₄ : total_cost = 600) 
    (h₅ : total_hats = blue_hats + green_hats) 
    (h₆ : total_cost = blue_hats * blue_cost + green_hats * G) : 
    G = 1 := by
  sorry

end green_hat_cost_l33_33724


namespace intersection_complement_l33_33662

open Set

variable (x : ℝ)

def M : Set ℝ := { x | -1 < x ∧ x < 2 }
def N : Set ℝ := { x | 1 ≤ x }

theorem intersection_complement :
  M ∩ (univ \ N) = { x | -1 < x ∧ x < 1 } := by
  sorry

end intersection_complement_l33_33662


namespace find_x_l33_33065

-- Define x as a function of n, where n is an odd natural number
def x (n : ℕ) (h_odd : n % 2 = 1) : ℕ :=
  6^n + 1

-- Define the main theorem
theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_prime_div : ∀ p, p.Prime → p ∣ x n h_odd → (p = 11 ∨ p = 7 ∨ p = 101)) : x 1 (by norm_num) = 7777 :=
  sorry

end find_x_l33_33065


namespace claire_needs_80_tiles_l33_33392

def room_length : ℕ := 14
def room_width : ℕ := 18
def border_width : ℕ := 2
def small_tile_side : ℕ := 1
def large_tile_side : ℕ := 3

def num_small_tiles : ℕ :=
  let perimeter_length := (2 * (room_width - 2 * border_width))
  let perimeter_width := (2 * (room_length - 2 * border_width))
  let corner_tiles := (2 * border_width) * 4
  perimeter_length + perimeter_width + corner_tiles

def num_large_tiles : ℕ :=
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let inner_area := inner_length * inner_width
  Nat.ceil (inner_area / (large_tile_side * large_tile_side))

theorem claire_needs_80_tiles : num_small_tiles + num_large_tiles = 80 :=
by sorry

end claire_needs_80_tiles_l33_33392


namespace edge_length_increase_l33_33719

theorem edge_length_increase (e e' : ℝ) (A : ℝ) (hA : ∀ e, A = 6 * e^2)
  (hA' : 2.25 * A = 6 * e'^2) :
  (e' - e) / e * 100 = 50 :=
by
  sorry

end edge_length_increase_l33_33719


namespace expression_evaluation_l33_33167

variable {x y : ℝ}

theorem expression_evaluation (h : (x-2)^2 + |y-3| = 0) :
  ( (x - 2 * y) * (x + 2 * y) - (x - y) ^ 2 + y * (y + 2 * x) ) / (-2 * y) = 2 :=
by
  sorry

end expression_evaluation_l33_33167


namespace range_of_inverse_proportion_function_l33_33670

noncomputable def f (x : ℝ) : ℝ := 6 / x

theorem range_of_inverse_proportion_function (x : ℝ) (hx : x > 2) : 
  0 < f x ∧ f x < 3 :=
sorry

end range_of_inverse_proportion_function_l33_33670


namespace fraction_multiplication_validity_l33_33043

theorem fraction_multiplication_validity (a b m x : ℝ) (hb : b ≠ 0) : 
  (x ≠ m) ↔ (b * (x - m) ≠ 0) :=
by
  sorry

end fraction_multiplication_validity_l33_33043


namespace min_value_of_F_on_negative_half_l33_33187

variable (f g : ℝ → ℝ)
variable (a b : ℝ)

def F (x : ℝ) := a * f x + b * g x + 2

def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem min_value_of_F_on_negative_half
  (h_f : is_odd f) (h_g : is_odd g)
  (max_F_positive_half : ∃ x, x > 0 ∧ F f g a b x = 5) :
  ∃ x, x < 0 ∧ F f g a b x = -3 :=
by {
  sorry
}

end min_value_of_F_on_negative_half_l33_33187


namespace ratio_of_dinner_to_lunch_l33_33641

theorem ratio_of_dinner_to_lunch
  (dinner: ℕ) (lunch: ℕ) (breakfast: ℕ) (k: ℕ)
  (h1: dinner = 240)
  (h2: dinner = k * lunch)
  (h3: dinner = 6 * breakfast)
  (h4: breakfast + lunch + dinner = 310) :
  dinner / lunch = 8 :=
by
  -- Proof to be completed
  sorry

end ratio_of_dinner_to_lunch_l33_33641


namespace total_corn_yield_l33_33391

/-- 
The total corn yield in centners, harvested from a certain field area, is expressed 
as a four-digit number composed of the digits 0, 2, 3, and 5. When the average 
yield per hectare was calculated, it was found to be the same number of centners 
as the number of hectares of the field area. 
This statement proves that the total corn yield is 3025. 
-/
theorem total_corn_yield : ∃ (Y A : ℕ), (Y = A^2) ∧ (A >= 10 ∧ A < 100) ∧ 
  (Y / 1000 != 0) ∧ (Y / 1000 != 1) ∧ (Y / 10 % 10 != 4) ∧ 
  (Y % 10 != 1) ∧ (Y % 10 = 0 ∨ Y % 10 = 5) ∧ 
  (Y / 100 % 10 == 0 ∨ Y / 100 % 10 == 2 ∨ Y / 100 % 10 == 3 ∨ Y / 100 % 10 == 5) ∧ 
  Y = 3025 := 
by 
  sorry

end total_corn_yield_l33_33391


namespace probability_XOX_OXO_l33_33427

open Nat

/-- Setting up the math problem to be proved -/
def X : Finset ℕ := {1, 2, 3, 4}
def O : Finset ℕ := {5, 6, 7}

def totalArrangements : ℕ := choose 7 4

def favorableArrangements : ℕ := 1

theorem probability_XOX_OXO : (favorableArrangements : ℚ) / (totalArrangements : ℚ) = 1 / 35 := by
  have h_total : totalArrangements = 35 := by sorry
  have h_favorable : favorableArrangements = 1 := by sorry
  rw [h_total, h_favorable]
  norm_num

end probability_XOX_OXO_l33_33427


namespace minor_premise_is_wrong_l33_33890

theorem minor_premise_is_wrong (a : ℝ) : ¬ (0 < a^2) := by
  sorry

end minor_premise_is_wrong_l33_33890


namespace sin_identity_l33_33564

theorem sin_identity (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α + π / 4) ^ 2 = 5 / 6 := 
sorry

end sin_identity_l33_33564


namespace price_cashews_l33_33158

noncomputable def price_per_pound_cashews 
  (price_mixed_nuts_per_pound : ℝ) 
  (weight_mixed_nuts : ℕ) 
  (weight_peanuts : ℕ) 
  (price_peanuts_per_pound : ℝ) 
  (weight_cashews : ℕ) : ℝ := 
  (price_mixed_nuts_per_pound * weight_mixed_nuts - price_peanuts_per_pound * weight_peanuts) / weight_cashews

open Real

theorem price_cashews 
  (price_mixed_nuts_per_pound : ℝ) 
  (weight_mixed_nuts : ℕ) 
  (weight_peanuts : ℕ) 
  (price_peanuts_per_pound : ℝ) 
  (weight_cashews : ℕ)
  (h1 : price_mixed_nuts_per_pound = 2.50) 
  (h2 : weight_mixed_nuts = 100) 
  (h3 : weight_peanuts = 40) 
  (h4 : price_peanuts_per_pound = 3.50) 
  (h5 : weight_cashews = 60) : 
  price_per_pound_cashews price_mixed_nuts_per_pound weight_mixed_nuts weight_peanuts price_peanuts_per_pound weight_cashews = 11 / 6 := by 
  sorry

end price_cashews_l33_33158


namespace Roger_needs_to_delete_20_apps_l33_33823

def max_apps := 50
def recommended_apps := 35
def current_apps := 2 * recommended_apps
def apps_to_delete := current_apps - max_apps

theorem Roger_needs_to_delete_20_apps : apps_to_delete = 20 := by
  sorry

end Roger_needs_to_delete_20_apps_l33_33823


namespace find_gamma_delta_l33_33141

theorem find_gamma_delta (γ δ : ℝ) (h : ∀ x : ℝ, (x - γ) / (x + δ) = (x^2 - 90 * x + 1980) / (x^2 + 60 * x - 3240)) : 
  γ + δ = 140 :=
sorry

end find_gamma_delta_l33_33141


namespace mady_balls_2010th_step_l33_33586

theorem mady_balls_2010th_step :
  let base_5_digits (n : Nat) : List Nat := (Nat.digits 5 n)
  (base_5_digits 2010).sum = 6 := by
  sorry

end mady_balls_2010th_step_l33_33586


namespace robin_gum_pieces_l33_33384

-- Defining the conditions
def packages : ℕ := 9
def pieces_per_package : ℕ := 15
def total_pieces : ℕ := 135

-- Theorem statement
theorem robin_gum_pieces (h1 : packages = 9) (h2 : pieces_per_package = 15) : packages * pieces_per_package = total_pieces := by
  -- According to the problem, the correct answer is 135 pieces
  have h: 9 * 15 = 135 := by norm_num
  rw [h1, h2]
  exact h

end robin_gum_pieces_l33_33384


namespace ratio_PA_AB_l33_33979

theorem ratio_PA_AB (A B C P : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup P]
  (h1 : ∃ AC CB : ℕ, AC = 2 * CB)
  (h2 : ∃ PA AB : ℕ, PA = 4 * (AB / 5)) :
  PA / AB = 4 / 5 := sorry

end ratio_PA_AB_l33_33979


namespace f_2020_eq_neg_1_l33_33484

noncomputable def f: ℝ → ℝ :=
sorry

axiom f_2_x_eq_neg_f_x : ∀ x: ℝ, f (2 - x) = -f x
axiom f_x_minus_2_eq_f_neg_x : ∀ x: ℝ, f (x - 2) = f (-x)
axiom f_specific : ∀ x : ℝ, -1 < x ∧ x < 1 -> f x = x^2 + 1

theorem f_2020_eq_neg_1 : f 2020 = -1 :=
sorry

end f_2020_eq_neg_1_l33_33484


namespace find_f_l33_33725

theorem find_f
  (d e f : ℝ)
  (vertex_x vertex_y : ℝ)
  (p_x p_y : ℝ)
  (vertex_cond : vertex_x = 3 ∧ vertex_y = -1)
  (point_cond : p_x = 5 ∧ p_y = 1)
  (equation : ∀ y : ℝ, ∃ x : ℝ, x = d * y^2 + e * y + f) :
  f = 7 / 2 :=
by
  sorry

end find_f_l33_33725


namespace percent_of_x_is_y_l33_33998

theorem percent_of_x_is_y 
    (x y : ℝ) 
    (h : 0.30 * (x - y) = 0.20 * (x + y)) : 
    y / x = 0.2 :=
  sorry

end percent_of_x_is_y_l33_33998


namespace sufficient_but_not_necessary_l33_33439

variable (m : ℝ)

def P : Prop := ∀ x : ℝ, x^2 - 4*x + 3*m > 0
def Q : Prop := ∀ x : ℝ, 3*x^2 + 4*x + m ≥ 0

theorem sufficient_but_not_necessary : (P m → Q m) ∧ ¬(Q m → P m) :=
by
  sorry

end sufficient_but_not_necessary_l33_33439


namespace length_AC_l33_33789

theorem length_AC {AB BC : ℝ} (h1: AB = 6) (h2: BC = 4) : (AC = 2 ∨ AC = 10) :=
sorry

end length_AC_l33_33789


namespace brianne_savings_in_may_l33_33889

-- Definitions based on conditions from a)
def initial_savings_jan : ℕ := 20
def multiplier : ℕ := 3
def additional_income : ℕ := 50

-- Savings in successive months
def savings_feb : ℕ := multiplier * initial_savings_jan
def savings_mar : ℕ := multiplier * savings_feb + additional_income
def savings_apr : ℕ := multiplier * savings_mar + additional_income
def savings_may : ℕ := multiplier * savings_apr + additional_income

-- The main theorem to verify
theorem brianne_savings_in_may : savings_may = 2270 :=
sorry

end brianne_savings_in_may_l33_33889


namespace compute_expression_l33_33924

theorem compute_expression : 12 + 5 * (4 - 9)^2 - 3 = 134 := by
  sorry

end compute_expression_l33_33924


namespace intersection_setA_setB_l33_33059

-- Define set A
def setA : Set ℝ := {x | 2 * x ≤ 4}

-- Define set B as the domain of the function y = log(x - 1)
def setB : Set ℝ := {x | x > 1}

-- Theorem to prove
theorem intersection_setA_setB : setA ∩ setB = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_setA_setB_l33_33059


namespace solve_for_x_l33_33195

theorem solve_for_x :
  ∃ x : ℝ, ((17.28 / x) / (3.6 * 0.2)) = 2 ∧ x = 12 :=
by
  sorry

end solve_for_x_l33_33195


namespace solution_set_inequality1_solution_set_inequality2_l33_33152

def inequality1 (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def inequality2 (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0

theorem solution_set_inequality1 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality1 x} :=
sorry

theorem solution_set_inequality2 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality2 x} :=
sorry

end solution_set_inequality1_solution_set_inequality2_l33_33152


namespace find_m_l33_33322

-- Given definitions and conditions
def is_ellipse (x y m : ℝ) := (x^2 / m) + (y^2 / 4) = 1
def eccentricity (m : ℝ) := Real.sqrt ((m - 4) / m) = 1 / 2

-- Prove that m = 16 / 3 given the conditions
theorem find_m (m : ℝ) (cond1 : is_ellipse 1 1 m) (cond2 : eccentricity m) (cond3 : m > 4) : m = 16 / 3 :=
by
  sorry

end find_m_l33_33322


namespace green_notebook_cost_l33_33514

-- Define the conditions
def num_notebooks : Nat := 4
def num_green_notebooks : Nat := 2
def num_black_notebooks : Nat := 1
def num_pink_notebooks : Nat := 1
def total_cost : ℕ := 45
def black_notebook_cost : ℕ := 15
def pink_notebook_cost : ℕ := 10

-- Define what we need to prove: The cost of each green notebook
def green_notebook_cost_each : ℕ := 10

-- The statement that combines the conditions with the goal to prove
theorem green_notebook_cost : 
  num_notebooks = 4 ∧ 
  num_green_notebooks = 2 ∧ 
  num_black_notebooks = 1 ∧ 
  num_pink_notebooks = 1 ∧ 
  total_cost = 45 ∧ 
  black_notebook_cost = 15 ∧ 
  pink_notebook_cost = 10 →
  2 * green_notebook_cost_each = total_cost - (black_notebook_cost + pink_notebook_cost) :=
by
  sorry

end green_notebook_cost_l33_33514


namespace angle_condition_l33_33645

theorem angle_condition
  {θ : ℝ}
  (h₀ : 0 ≤ θ)
  (h₁ : θ < π)
  (h₂ : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + 2 * (1 - x)^2 * Real.sin θ > 0) :
  0 < θ ∧ θ < π / 2 :=
by
  sorry

end angle_condition_l33_33645


namespace distance_p_runs_l33_33347

-- Given conditions
def runs_faster (speed_q : ℝ) : ℝ := 1.20 * speed_q
def head_start : ℝ := 50

-- Proof statement
theorem distance_p_runs (speed_q distance_q : ℝ) (h1 : runs_faster speed_q = 1.20 * speed_q)
                         (h2 : head_start = 50)
                         (h3 : (distance_q / speed_q) = ((distance_q + head_start) / (runs_faster speed_q))) :
                         (distance_q + head_start = 300) :=
by
  sorry

end distance_p_runs_l33_33347


namespace range_of_m_l33_33962

def A (x : ℝ) : Prop := 1/2 < x ∧ x < 1

def B (x : ℝ) (m : ℝ) : Prop := x^2 + 2 * x + 1 - m ≤ 0

theorem range_of_m (m : ℝ) : (∀ x : ℝ, A x → B x m) → 4 ≤ m := by
  sorry

end range_of_m_l33_33962


namespace factor_expression_l33_33821

theorem factor_expression (x : ℝ) : 
  75 * x^11 + 225 * x^22 = 75 * x^11 * (1 + 3 * x^11) :=
by sorry

end factor_expression_l33_33821


namespace proof_for_y_l33_33546

theorem proof_for_y (x y : ℝ) (h1 : 3 * x^2 + 5 * x + 4 * y + 2 = 0) (h2 : 3 * x + y + 4 = 0) : 
  y^2 + 15 * y + 2 = 0 :=
sorry

end proof_for_y_l33_33546


namespace correct_statements_l33_33091
noncomputable def is_pythagorean_triplet (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem correct_statements {a b c : ℕ} (h1 : is_pythagorean_triplet a b c) (h2 : a^2 + b^2 = c^2) :
  (∀ (a b c : ℕ), (is_pythagorean_triplet a b c → a^2 + b^2 = c^2)) ∧
  (∀ (a b c : ℕ), (is_pythagorean_triplet a b c → is_pythagorean_triplet (2 * a) (2 * b) (2 * c))) :=
by sorry

end correct_statements_l33_33091


namespace smallest_fraction_numerator_l33_33932

theorem smallest_fraction_numerator (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : (a * 4) > (b * 3)) : a = 73 :=
  sorry

end smallest_fraction_numerator_l33_33932


namespace PlanY_more_cost_effective_l33_33754

-- Define the gigabytes Tim uses
variable (y : ℕ)

-- Define the cost functions for Plan X and Plan Y in cents
def cost_PlanX (y : ℕ) := 25 * y
def cost_PlanY (y : ℕ) := 1500 + 15 * y

-- Prove that Plan Y is cheaper than Plan X when y >= 150
theorem PlanY_more_cost_effective (y : ℕ) : y ≥ 150 → cost_PlanY y < cost_PlanX y := by
  sorry

end PlanY_more_cost_effective_l33_33754


namespace number_of_divisors_36_l33_33151

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l33_33151


namespace set_intersection_complement_l33_33576

open Set

noncomputable def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem set_intersection_complement :
  (compl A ∩ B) = {x | 0 < x ∧ x ≤ 3} :=
by
  sorry

end set_intersection_complement_l33_33576


namespace max_value_of_function_l33_33164

noncomputable def function_to_maximize (x : ℝ) : ℝ :=
  (Real.sin x)^4 + (Real.cos x)^4 + 1 / ((Real.sin x)^2 + (Real.cos x)^2 + 1)

theorem max_value_of_function :
  ∃ x : ℝ, function_to_maximize x = 7 / 4 :=
sorry

end max_value_of_function_l33_33164


namespace division_result_l33_33015

def expr := 180 / (12 + 13 * 2)

theorem division_result : expr = 90 / 19 := by
  sorry

end division_result_l33_33015


namespace find_x_l33_33369

-- Definitions of the conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Inner product definition
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Perpendicular condition
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_x (x : ℝ) (h : is_perpendicular vector_a (vector_b x)) : x = 4 :=
  sorry

end find_x_l33_33369


namespace color_block_prob_l33_33760

-- Definitions of the problem's conditions
def colors : List (List String) := [
    ["red", "blue", "yellow", "green"],
    ["red", "blue", "yellow", "white"]
]

-- The events in which at least one box receives 3 blocks of the same color
def event_prob : ℚ := 3 / 64

-- Tuple as a statement to prove in Lean
theorem color_block_prob (m n : ℕ) (h : m + n = 67) : 
  ∃ (m n : ℕ), (m / n : ℚ) = event_prob := 
by
  use 3
  use 64
  simp
  sorry

end color_block_prob_l33_33760


namespace find_k_l33_33279

-- Define the conditions and the question
theorem find_k (t k : ℝ) (h1 : t = 50) (h2 : t = (5 / 9) * (k - 32)) : k = 122 := by
  -- Proof will go here
  sorry

end find_k_l33_33279


namespace binom_25_5_l33_33510

theorem binom_25_5 :
  (Nat.choose 23 3 = 1771) ∧
  (Nat.choose 23 4 = 8855) ∧
  (Nat.choose 23 5 = 33649) → 
  Nat.choose 25 5 = 53130 := by
sorry

end binom_25_5_l33_33510


namespace correct_calculation_l33_33882

theorem correct_calculation (x y : ℝ) : -x^2 * y + 3 * x^2 * y = 2 * x^2 * y :=
by
  sorry

end correct_calculation_l33_33882


namespace original_fund_was_830_l33_33770

/- Define the number of employees as a variable -/
variables (n : ℕ)

/- Define the conditions given in the problem -/
def initial_fund := 60 * n - 10
def new_fund_after_distributing_50 := initial_fund - 50 * n
def remaining_fund := 130

/- State the proof goal -/
theorem original_fund_was_830 :
  initial_fund = 830 :=
by sorry

end original_fund_was_830_l33_33770


namespace johns_contribution_l33_33442

-- Definitions
variables (A J : ℝ)
axiom h1 : 1.5 * A = 75
axiom h2 : (2 * A + J) / 3 = 75

-- Statement of the proof problem
theorem johns_contribution : J = 125 :=
by
  sorry

end johns_contribution_l33_33442


namespace rectangle_area_l33_33839

variable (a b c : ℝ)

theorem rectangle_area (h : a^2 + b^2 = c^2) : a * b = area :=
by sorry

end rectangle_area_l33_33839


namespace wall_area_l33_33568

theorem wall_area (width : ℝ) (height : ℝ) (h1 : width = 2) (h2 : height = 4) : width * height = 8 := by
  sorry

end wall_area_l33_33568


namespace problem1_problem2_l33_33023

-- Problem 1: y(x + y) + (x + y)(x - y) = x^2
theorem problem1 (x y : ℝ) : y * (x + y) + (x + y) * (x - y) = x^2 := 
by sorry

-- Problem 2: ( (2m + 1) / (m + 1) + m - 1 ) ÷ ( (m + 2) / (m^2 + 2m + 1) ) = m^2 + m
theorem problem2 (m : ℝ) (h1 : m ≠ -1) : 
  ( (2 * m + 1) / (m + 1) + m - 1 ) / ( (m + 2) / ((m + 1)^2) ) = m^2 + m := 
by sorry

end problem1_problem2_l33_33023


namespace original_price_l33_33963

theorem original_price (P : ℝ) (h_discount : 0.75 * P = 560): P = 746.68 :=
sorry

end original_price_l33_33963


namespace percentage_of_apples_after_removal_l33_33334

-- Declare the initial conditions as Lean definitions
def initial_apples : Nat := 12
def initial_oranges : Nat := 23
def removed_oranges : Nat := 15

-- Calculate the new totals
def new_oranges : Nat := initial_oranges - removed_oranges
def new_total_fruit : Nat := initial_apples + new_oranges

-- Define the expected percentage of apples as a real number
def expected_percentage_apples : Nat := 60

-- Prove that the percentage of apples after removing the specified number of oranges is 60%
theorem percentage_of_apples_after_removal :
  (initial_apples * 100 / new_total_fruit) = expected_percentage_apples := by
  sorry

end percentage_of_apples_after_removal_l33_33334


namespace base_b_for_three_digits_l33_33687

theorem base_b_for_three_digits (b : ℕ) : b = 7 ↔ b^2 ≤ 256 ∧ 256 < b^3 := by
  sorry

end base_b_for_three_digits_l33_33687


namespace central_angle_of_section_l33_33051

theorem central_angle_of_section (A : ℝ) (x: ℝ) (H : (1 / 8 : ℝ) = (x / 360)) : x = 45 :=
by
  sorry

end central_angle_of_section_l33_33051


namespace people_per_table_l33_33658

theorem people_per_table (kids adults tables : ℕ) (h_kids : kids = 45) (h_adults : adults = 123) (h_tables : tables = 14) :
  ((kids + adults) / tables) = 12 :=
by
  -- Placeholder for proof
  sorry

end people_per_table_l33_33658


namespace gcd_fact_8_10_l33_33213

theorem gcd_fact_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = 40320 := by
  -- No proof needed
  sorry

end gcd_fact_8_10_l33_33213


namespace max_consecutive_sum_terms_l33_33672

theorem max_consecutive_sum_terms (S : ℤ) (n : ℕ) (H1 : S = 2015) (H2 : 0 < n) :
  (∃ a : ℤ, S = (a * n + (n * (n - 1)) / 2)) → n = 4030 :=
sorry

end max_consecutive_sum_terms_l33_33672


namespace min_eq_floor_sqrt_l33_33747

theorem min_eq_floor_sqrt (n : ℕ) (h : n > 0) : 
  (∀ k : ℕ, k > 0 → (k + n / k) ≥ ⌊(Real.sqrt (4 * n + 1))⌋) := 
sorry

end min_eq_floor_sqrt_l33_33747


namespace valid_outfit_choices_l33_33651

def shirts := 6
def pants := 6
def hats := 12
def patterned_hats := 6

theorem valid_outfit_choices : 
  (shirts * pants * hats) - shirts - (patterned_hats * shirts * (pants - 1)) = 246 := by
  sorry

end valid_outfit_choices_l33_33651


namespace correct_answer_statement_l33_33374

theorem correct_answer_statement
  (A := "In order to understand the situation of extracurricular reading among middle school students in China, a comprehensive survey should be conducted.")
  (B := "The median and mode of a set of data 1, 2, 5, 5, 5, 3, 3 are both 5.")
  (C := "When flipping a coin 200 times, there will definitely be 100 times when it lands 'heads up.'")
  (D := "If the variance of data set A is 0.03 and the variance of data set B is 0.1, then data set A is more stable than data set B.")
  (correct_answer := "D") : 
  correct_answer = "D" :=
  by sorry

end correct_answer_statement_l33_33374


namespace line_passes_through_fixed_point_l33_33479

theorem line_passes_through_fixed_point (k : ℝ) : ∃ (x y : ℝ), x = -2 ∧ y = 1 ∧ y = k * x + 2 * k + 1 :=
by
  sorry

end line_passes_through_fixed_point_l33_33479


namespace range_x_minus_y_l33_33549

-- Definition of the curve in polar coordinates
def curve_polar (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta + 2 * Real.sin theta

-- Conversion to rectangular coordinates
noncomputable def curve_rectangular (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * x + 2 * y

-- The final Lean 4 statement
theorem range_x_minus_y (x y : ℝ) (h : curve_rectangular x y) :
  1 - Real.sqrt 10 ≤ x - y ∧ x - y ≤ 1 + Real.sqrt 10 :=
sorry

end range_x_minus_y_l33_33549


namespace new_person_weight_l33_33300

theorem new_person_weight
  (initial_weight : ℝ)
  (average_increase : ℝ)
  (num_people : ℕ)
  (weight_replace : ℝ)
  (total_increase : ℝ)
  (W : ℝ)
  (h1 : num_people = 10)
  (h2 : average_increase = 3.5)
  (h3 : weight_replace = 65)
  (h4 : total_increase = num_people * average_increase)
  (h5 : total_increase = 35)
  (h6 : W = weight_replace + total_increase) :
  W = 100 := sorry

end new_person_weight_l33_33300


namespace investment_time_period_l33_33472

variable (P : ℝ) (r15 r12 : ℝ) (T : ℝ)
variable (hP : P = 15000)
variable (hr15 : r15 = 0.15)
variable (hr12 : r12 = 0.12)
variable (diff : 2250 * T - 1800 * T = 900)

theorem investment_time_period :
  T = 2 := by
  sorry

end investment_time_period_l33_33472


namespace evaluate_series_l33_33217

theorem evaluate_series : 1 + (1 / 2) + (1 / 4) + (1 / 8) = 15 / 8 := by
  sorry

end evaluate_series_l33_33217


namespace max_quotient_l33_33657

theorem max_quotient (x y : ℝ) (h1 : -5 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 6) : 
  ∃ z, z = (x + y) / x ∧ ∀ w, w = (x + y) / x → w ≤ 0 :=
by
  sorry

end max_quotient_l33_33657


namespace baseball_card_decrease_l33_33792

theorem baseball_card_decrease (V₀ : ℝ) (V₁ V₂ : ℝ)
  (h₁: V₁ = V₀ * (1 - 0.20))
  (h₂: V₂ = V₁ * (1 - 0.20)) :
  ((V₀ - V₂) / V₀) * 100 = 36 :=
by
  sorry

end baseball_card_decrease_l33_33792


namespace winner_won_by_l33_33801

theorem winner_won_by (V : ℝ) (h₁ : 0.62 * V = 806) : 806 - 0.38 * V = 312 :=
by
  sorry

end winner_won_by_l33_33801


namespace find_f_5_l33_33958

-- Definitions from conditions
def f (x : ℝ) (a b : ℝ) : ℝ := a * x ^ 3 - b * x + 2

-- Stating the theorem
theorem find_f_5 (a b : ℝ) (h : f (-5) a b = 17) : f 5 a b = -13 :=
by
  sorry

end find_f_5_l33_33958


namespace find_bigger_number_l33_33266

noncomputable def common_factor (x : ℕ) : Prop :=
  8 * x + 3 * x = 143

theorem find_bigger_number (x : ℕ) (h : common_factor x) : 8 * x = 104 :=
by
  sorry

end find_bigger_number_l33_33266


namespace train_speed_l33_33828

/-- Proof problem: Speed calculation of a train -/
theorem train_speed :
  ∀ (length : ℝ) (time_seconds : ℝ) (speed_kmph : ℝ),
    length = 40 →
    time_seconds = 0.9999200063994881 →
    speed_kmph = (length / 1000) / (time_seconds / 3600) →
    speed_kmph = 144 :=
by
  intros length time_seconds speed_kmph h_length h_time_seconds h_speed_kmph
  rw [h_length, h_time_seconds] at h_speed_kmph
  -- sorry is used to skip the proof steps
  sorry 

end train_speed_l33_33828


namespace teacher_age_is_45_l33_33100

def avg_age_of_students := 14
def num_students := 30
def avg_age_with_teacher := 15
def num_people_with_teacher := 31

def total_age_of_students := avg_age_of_students * num_students
def total_age_with_teacher := avg_age_with_teacher * num_people_with_teacher

theorem teacher_age_is_45 : (total_age_with_teacher - total_age_of_students = 45) :=
by
  sorry

end teacher_age_is_45_l33_33100


namespace smallest_possible_value_other_integer_l33_33397

theorem smallest_possible_value_other_integer (x : ℕ) (n : ℕ) (h_pos : x > 0)
  (h_gcd : ∃ m, Nat.gcd m n = x + 3 ∧ m = 30) 
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) :
  n = 162 := 
by sorry

end smallest_possible_value_other_integer_l33_33397


namespace cars_sold_l33_33346

theorem cars_sold (sales_Mon sales_Tue sales_Wed cars_Thu_Fri_Sat : ℕ) 
  (mean : ℝ) (h1 : sales_Mon = 8) 
  (h2 : sales_Tue = 3) 
  (h3 : sales_Wed = 10) 
  (h4 : mean = 5.5) 
  (h5 : mean * 6 = sales_Mon + sales_Tue + sales_Wed + cars_Thu_Fri_Sat):
  cars_Thu_Fri_Sat = 12 :=
sorry

end cars_sold_l33_33346


namespace min_value_SN64_by_aN_is_17_over_2_l33_33874

noncomputable def a_n (n : ℕ) : ℕ := 2 * n
noncomputable def S_n (n : ℕ) : ℕ := n^2 + n

theorem min_value_SN64_by_aN_is_17_over_2 :
  ∃ (n : ℕ), 2 ≤ n ∧ (a_2 = 4 ∧ S_10 = 110) →
  ((S_n n + 64) / a_n n) = 17 / 2 :=
by
  sorry

end min_value_SN64_by_aN_is_17_over_2_l33_33874


namespace number_of_bottles_poured_l33_33808

/-- Definition of full cylinder capacity (fixed as 80 bottles) --/
def full_capacity : ℕ := 80

/-- Initial fraction of full capacity --/
def initial_fraction : ℚ := 3 / 4

/-- Final fraction of full capacity --/
def final_fraction : ℚ := 4 / 5

/-- Proof problem: Prove the number of bottles of oil poured into the cylinder --/
theorem number_of_bottles_poured :
  (final_fraction * full_capacity) - (initial_fraction * full_capacity) = 4 := by
  sorry

end number_of_bottles_poured_l33_33808


namespace baseball_glove_price_l33_33320

noncomputable def original_price_glove : ℝ := 42.50

theorem baseball_glove_price (cards bat glove_discounted cleats total : ℝ) 
  (h1 : cards = 25) 
  (h2 : bat = 10) 
  (h3 : cleats = 2 * 10)
  (h4 : total = 79) 
  (h5 : glove_discounted = total - (cards + bat + cleats)) 
  (h6 : glove_discounted = 0.80 * original_price_glove) : 
  original_price_glove = 42.50 := by 
  sorry

end baseball_glove_price_l33_33320


namespace find_d_l33_33433

theorem find_d (a b c d x : ℝ)
  (h1 : ∀ x, 2 ≤ a * (Real.cos (b * x + c)) + d ∧ a * (Real.cos (b * x + c)) + d ≤ 4)
  (h2 : Real.cos (b * 0 + c) = 1) :
  d = 3 :=
sorry

end find_d_l33_33433


namespace compute_expression_l33_33597

theorem compute_expression (x : ℝ) (hx : x + 1 / x = 7) : 
  (x - 3)^2 + 36 / (x - 3)^2 = 12.375 := 
  sorry

end compute_expression_l33_33597


namespace series_sum_eq_50_l33_33949

noncomputable def series_sum (x : ℝ) : ℝ :=
  2 + 6 * x + 10 * x^2 + 14 * x^3 -- This represents the series

theorem series_sum_eq_50 : 
  ∃ x : ℝ, series_sum x = 50 ∧ x = 0.59 :=
by
  sorry

end series_sum_eq_50_l33_33949


namespace covered_ratio_battonya_covered_ratio_sopron_l33_33690

noncomputable def angular_diameter_sun : ℝ := 1899 / 2
noncomputable def angular_diameter_moon : ℝ := 1866 / 2

def max_phase_battonya : ℝ := 0.766
def max_phase_sopron : ℝ := 0.678

def center_distance (R_M R_S f : ℝ) : ℝ :=
  R_M - (2 * f - 1) * R_S

-- Defining the hypothetical calculation (details omitted for brevity)
def covered_ratio (R_S R_M d : ℝ) : ℝ := 
  -- Placeholder for the actual calculation logic
  sorry

theorem covered_ratio_battonya :
  covered_ratio angular_diameter_sun angular_diameter_moon (center_distance angular_diameter_moon angular_diameter_sun max_phase_battonya) = 0.70 :=
  sorry

theorem covered_ratio_sopron :
  covered_ratio angular_diameter_sun angular_diameter_moon (center_distance angular_diameter_moon angular_diameter_sun max_phase_sopron) = 0.59 :=
  sorry

end covered_ratio_battonya_covered_ratio_sopron_l33_33690


namespace total_cars_produced_l33_33045

def cars_produced_north_america : ℕ := 3884
def cars_produced_europe : ℕ := 2871
def cars_produced_asia : ℕ := 5273
def cars_produced_south_america : ℕ := 1945

theorem total_cars_produced : cars_produced_north_america + cars_produced_europe + cars_produced_asia + cars_produced_south_america = 13973 := by
  sorry

end total_cars_produced_l33_33045


namespace combined_cost_price_l33_33471

theorem combined_cost_price :
  let face_value_A : ℝ := 100
  let discount_A : ℝ := 2
  let purchase_price_A := face_value_A - (discount_A / 100 * face_value_A)
  let brokerage_A := 0.2 / 100 * purchase_price_A
  let total_cost_price_A := purchase_price_A + brokerage_A

  let face_value_B : ℝ := 100
  let premium_B : ℝ := 1.5
  let purchase_price_B := face_value_B + (premium_B / 100 * face_value_B)
  let brokerage_B := 0.2 / 100 * purchase_price_B
  let total_cost_price_B := purchase_price_B + brokerage_B

  let combined_cost_price := total_cost_price_A + total_cost_price_B

  combined_cost_price = 199.899 := by
  sorry

end combined_cost_price_l33_33471


namespace prob_one_side_of_tri_in_decagon_is_half_l33_33930

noncomputable def probability_one_side_of_tri_in_decagon : ℚ :=
  let num_vertices := 10
  let total_triangles := Nat.choose num_vertices 3
  let favorable_triangles := 10 * 6
  favorable_triangles / total_triangles

theorem prob_one_side_of_tri_in_decagon_is_half :
  probability_one_side_of_tri_in_decagon = 1 / 2 := by
  sorry

end prob_one_side_of_tri_in_decagon_is_half_l33_33930


namespace total_age_in_3_years_l33_33606

theorem total_age_in_3_years (Sam Sue Kendra : ℕ)
  (h1 : Kendra = 18)
  (h2 : Kendra = 3 * Sam)
  (h3 : Sam = 2 * Sue) :
  Sam + Sue + Kendra + 3 * 3 = 36 :=
by
  sorry

end total_age_in_3_years_l33_33606


namespace simplify_expression_d_l33_33715

variable (a b c : ℝ)

theorem simplify_expression_d : a - (b - c) = a - b + c :=
  sorry

end simplify_expression_d_l33_33715


namespace intersection_M_N_l33_33153

def M : Set ℝ := { x | (x - 2) / (x - 3) < 0 }
def N : Set ℝ := { x | Real.log (x - 2) / Real.log (1 / 2) ≥ 1 }

theorem intersection_M_N : M ∩ N = { x | 2 < x ∧ x ≤ 5 / 2 } :=
by
  sorry

end intersection_M_N_l33_33153


namespace triangle_angles_in_given_ratio_l33_33515

theorem triangle_angles_in_given_ratio (x : ℝ) (y : ℝ) (z : ℝ) (h : x + y + z = 180) (r : x / 1 = y / 4 ∧ y / 4 = z / 7) : 
  x = 15 ∧ y = 60 ∧ z = 105 :=
by
  sorry

end triangle_angles_in_given_ratio_l33_33515


namespace determine_a_value_l33_33591

theorem determine_a_value (a : ℝ) :
  (∀ y₁ y₂ : ℝ, ∃ m₁ m₂ : ℝ, (m₁, y₁) = (a, -2) ∧ (m₂, y₂) = (3, -4) ∧ (m₁ = m₂)) → a = 3 :=
by
  sorry

end determine_a_value_l33_33591


namespace probability_of_Y_l33_33193

variable (P_X : ℝ) (P_X_and_Y : ℝ) (P_Y : ℝ)

theorem probability_of_Y (h1 : P_X = 1 / 7)
                         (h2 : P_X_and_Y = 0.031746031746031744) :
  P_Y = 0.2222222222222222 :=
sorry

end probability_of_Y_l33_33193


namespace find_x_l33_33182

noncomputable def isCorrectValue (x : ℝ) : Prop :=
  ⌊x⌋ + x = 13.4

theorem find_x (x : ℝ) (h : isCorrectValue x) : x = 6.4 :=
  sorry

end find_x_l33_33182


namespace fraction_zero_x_value_l33_33947

theorem fraction_zero_x_value (x : ℝ) (h1 : 2 * x = 0) (h2 : x + 3 ≠ 0) : x = 0 :=
by
  sorry

end fraction_zero_x_value_l33_33947


namespace intersection_eq_T_l33_33980

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l33_33980


namespace Markus_bags_count_l33_33422

-- Definitions of the conditions
def Mara_bags : ℕ := 12
def Mara_marbles_per_bag : ℕ := 2
def Markus_marbles_per_bag : ℕ := 13
def marbles_difference : ℕ := 2

-- Derived conditions
def Mara_total_marbles : ℕ := Mara_bags * Mara_marbles_per_bag
def Markus_total_marbles : ℕ := Mara_total_marbles + marbles_difference

-- Statement to prove
theorem Markus_bags_count : Markus_total_marbles / Markus_marbles_per_bag = 2 :=
by
  -- Skip the proof, leaving it as a task for the prover
  sorry

end Markus_bags_count_l33_33422


namespace intersection_eq_l33_33138

-- Given conditions
def M : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def N : Set ℝ := { x | x > 1 }

-- Statement of the problem to be proved
theorem intersection_eq : M ∩ N = { x | 1 < x ∧ x < 3 } :=
sorry

end intersection_eq_l33_33138


namespace area_ratio_of_shapes_l33_33106

theorem area_ratio_of_shapes (l w r : ℝ) (h1 : 2 * l + 2 * w = 2 * π * r) (h2 : l = 3 * w) :
  (l * w) / (π * r^2) = (3 * π) / 16 :=
by sorry

end area_ratio_of_shapes_l33_33106


namespace find_a_find_b_find_T_l33_33967

open Real

def S (n : ℕ) : ℝ := 2 * n^2 + n

def a (n : ℕ) : ℝ := if n = 1 then 3 else S n - S (n - 1)

def b (n : ℕ) : ℝ := 2^(n - 1)

def T (n : ℕ) : ℝ := (4 * n - 5) * 2^n + 5

theorem find_a (n : ℕ) (hn : n > 0) : a n = 4 * n - 1 :=
by sorry

theorem find_b (n : ℕ) (hn : n > 0) : b n = 2^(n-1) :=
by sorry

theorem find_T (n : ℕ) (hn : n > 0) (a_def : ∀ n, a n = 4 * n - 1) (b_def : ∀ n, b n = 2^(n-1)) : T n = (4 * n - 5) * 2^n + 5 :=
by sorry

end find_a_find_b_find_T_l33_33967


namespace people_per_van_is_six_l33_33165

noncomputable def n_vans : ℝ := 6.0
noncomputable def n_buses : ℝ := 8.0
noncomputable def p_bus : ℝ := 18.0
noncomputable def people_difference : ℝ := 108

theorem people_per_van_is_six (x : ℝ) (h : n_buses * p_bus = n_vans * x + people_difference) : x = 6.0 := 
by
  sorry

end people_per_van_is_six_l33_33165


namespace units_digit_of_k3_plus_5k_l33_33372

def k : ℕ := 2024^2 + 3^2024

theorem units_digit_of_k3_plus_5k (k := 2024^2 + 3^2024) : 
  ((k^3 + 5^k) % 10) = 8 := 
by 
  sorry

end units_digit_of_k3_plus_5k_l33_33372


namespace min_sides_of_polygon_that_overlaps_after_rotation_l33_33581

theorem min_sides_of_polygon_that_overlaps_after_rotation (θ : ℝ) (n : ℕ) 
  (hθ: θ = 36) (hdiv: 360 % θ = 0) :
    n = 10 :=
by
  sorry

end min_sides_of_polygon_that_overlaps_after_rotation_l33_33581


namespace car_distance_l33_33297

/-- A car takes 4 hours to cover a certain distance. We are given that the car should maintain a speed of 90 kmph to cover the same distance in (3/2) of the previous time (which is 6 hours). We need to prove that the distance the car needs to cover is 540 km. -/
theorem car_distance (time_initial : ℝ) (speed : ℝ) (time_new : ℝ) (distance : ℝ) 
  (h1 : time_initial = 4) 
  (h2 : speed = 90)
  (h3 : time_new = (3/2) * time_initial)
  (h4 : distance = speed * time_new) : 
  distance = 540 := 
sorry

end car_distance_l33_33297


namespace giftWrapperPerDay_l33_33866

variable (giftWrapperPerBox : ℕ)
variable (boxesPer3Days : ℕ)

def giftWrapperUsedIn3Days := giftWrapperPerBox * boxesPer3Days

theorem giftWrapperPerDay (h_giftWrapperPerBox : giftWrapperPerBox = 18)
  (h_boxesPer3Days : boxesPer3Days = 15) : giftWrapperUsedIn3Days giftWrapperPerBox boxesPer3Days / 3 = 90 :=
by
  sorry

end giftWrapperPerDay_l33_33866


namespace problem_l33_33722

variable (x y z w : ℚ)

theorem problem
  (h1 : x / y = 7)
  (h2 : z / y = 5)
  (h3 : z / w = 3 / 4) :
  w / x = 20 / 21 :=
by sorry

end problem_l33_33722


namespace sum_of_center_coords_l33_33295

theorem sum_of_center_coords (x y : ℝ) (h : x^2 + y^2 = 4 * x - 6 * y + 9) : 2 + (-3) = -1 :=
by
  sorry

end sum_of_center_coords_l33_33295


namespace geometric_seq_sum_l33_33933

theorem geometric_seq_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a1_pos : a 1 > 0)
  (h_a4_7 : a 4 + a 7 = 2)
  (h_a5_6 : a 5 * a 6 = -8) :
  a 1 + a 4 + a 7 + a 10 = -5 := 
sorry

end geometric_seq_sum_l33_33933


namespace problems_solved_by_trainees_l33_33342

theorem problems_solved_by_trainees (n m : ℕ) (h : ∀ t, t < m → (∃ p, p < n → p ≥ n / 2)) :
  ∃ p < n, (∃ t, t < m → t ≥ m / 2) :=
by
  sorry

end problems_solved_by_trainees_l33_33342


namespace angle_CBD_is_4_l33_33988

theorem angle_CBD_is_4 (angle_ABC : ℝ) (angle_ABD : ℝ) (h₁ : angle_ABC = 24) (h₂ : angle_ABD = 20) : angle_ABC - angle_ABD = 4 :=
by 
  sorry

end angle_CBD_is_4_l33_33988


namespace children_eating_porridge_today_l33_33227

theorem children_eating_porridge_today
  (eat_every_day : ℕ)
  (eat_every_other_day : ℕ)
  (ate_yesterday : ℕ) :
  eat_every_day = 5 →
  eat_every_other_day = 7 →
  ate_yesterday = 9 →
  (eat_every_day + (eat_every_other_day - (ate_yesterday - eat_every_day)) = 8) :=
by
  intros h1 h2 h3
  sorry

end children_eating_porridge_today_l33_33227


namespace quadratic_intersects_x_axis_at_two_points_l33_33983

theorem quadratic_intersects_x_axis_at_two_points (k : ℝ) :
  (k < 1 ∧ k ≠ 0) ↔ ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (kx1^2 + 2 * x1 + 1 = 0) ∧ (kx2^2 + 2 * x2 + 1 = 0) := 
by
  sorry

end quadratic_intersects_x_axis_at_two_points_l33_33983


namespace intersection_correct_l33_33543

-- Conditions
def M : Set ℤ := { -1, 0, 1, 3, 5 }
def N : Set ℤ := { -2, 1, 2, 3, 5 }

-- Statement to prove
theorem intersection_correct : M ∩ N = { 1, 3, 5 } :=
by
  sorry

end intersection_correct_l33_33543


namespace surface_is_plane_l33_33987

-- Define cylindrical coordinates
structure CylindricalCoordinate where
  r : ℝ
  θ : ℝ
  z : ℝ

-- Define the property for a constant θ
def isConstantTheta (c : ℝ) (coord : CylindricalCoordinate) : Prop :=
  coord.θ = c

-- Define the plane in cylindrical coordinates
def isPlane (c : ℝ) (coord : CylindricalCoordinate) : Prop :=
  isConstantTheta c coord

-- Theorem: The surface described by θ = c in cylindrical coordinates is a plane.
theorem surface_is_plane (c : ℝ) (coord : CylindricalCoordinate) :
    isPlane c coord ↔ isConstantTheta c coord := sorry

end surface_is_plane_l33_33987


namespace steve_average_speed_l33_33488

theorem steve_average_speed 
  (Speed1 Time1 Speed2 Time2 : ℝ) 
  (cond1 : Speed1 = 40) 
  (cond2 : Time1 = 5)
  (cond3 : Speed2 = 80) 
  (cond4 : Time2 = 3) 
: 
(Speed1 * Time1 + Speed2 * Time2) / (Time1 + Time2) = 55 := 
sorry

end steve_average_speed_l33_33488


namespace intersection_eq_expected_l33_33285

def setA := { x : ℝ | 0 ≤ x ∧ x ≤ 3 }
def setB := { x : ℝ | 1 ≤ x ∧ x < 4 }
def expectedSet := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_eq_expected :
  {x : ℝ | x ∈ setA ∧ x ∈ setB} = expectedSet :=
by
  sorry

end intersection_eq_expected_l33_33285


namespace minimum_value_of_l33_33888

noncomputable def minimum_value (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem minimum_value_of (x y z : ℝ) (hxyz : x > 0 ∧ y > 0 ∧ z > 0) (h : 1/x + 1/y + 1/z = 9) :
  minimum_value x y z = 1 / 3456 := 
sorry

end minimum_value_of_l33_33888


namespace who_is_first_l33_33913

def positions (A B C D : ℕ) : Prop :=
  A + B + D = 6 ∧ B + C = 6 ∧ B < A ∧ A + B + C + D = 10

theorem who_is_first (A B C D : ℕ) (h : positions A B C D) : D = 1 :=
sorry

end who_is_first_l33_33913


namespace volume_of_rect_box_l33_33573

open Real

/-- Proof of the volume of a rectangular box given its face areas -/
theorem volume_of_rect_box (l w h : ℝ) 
  (A1 : l * w = 40) 
  (A2 : w * h = 10) 
  (A3 : l * h = 8) : 
  l * w * h = 40 * sqrt 2 :=
by
  sorry

end volume_of_rect_box_l33_33573


namespace increasing_sequence_range_of_a_l33_33582

theorem increasing_sequence_range_of_a (a : ℝ) (a_n : ℕ → ℝ) (h : ∀ n : ℕ, a_n n = a * n ^ 2 + n) (increasing : ∀ n : ℕ, a_n (n + 1) > a_n n) : 0 ≤ a :=
by
  sorry

end increasing_sequence_range_of_a_l33_33582


namespace smallest_value_of_y_l33_33769

open Real

theorem smallest_value_of_y : 
  ∃ (y : ℝ), 6 * y^2 - 29 * y + 24 = 0 ∧ (∀ z : ℝ, 6 * z^2 - 29 * z + 24 = 0 → y ≤ z) ∧ y = 4 / 3 := 
sorry

end smallest_value_of_y_l33_33769


namespace mgp_inequality_l33_33082

theorem mgp_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (1 / Real.sqrt (1 / 2 + a + a * b + a * b * c) + 
   1 / Real.sqrt (1 / 2 + b + b * c + b * c * d) + 
   1 / Real.sqrt (1 / 2 + c + c * d + c * d * a) + 
   1 / Real.sqrt (1 / 2 + d + d * a + d * a * b)) 
  ≥ Real.sqrt 2 := 
sorry

end mgp_inequality_l33_33082


namespace total_volume_collection_l33_33273

-- Define the conditions
def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15
def cost_per_box : ℚ := 0.5
def minimum_total_cost : ℚ := 255

-- Define the volume of one box
def volume_of_one_box : ℕ := box_length * box_width * box_height

-- Define the number of boxes needed
def number_of_boxes : ℚ := minimum_total_cost / cost_per_box

-- Define the total volume of the collection
def total_volume : ℚ := volume_of_one_box * number_of_boxes

-- The goal is to prove that the total volume of the collection is as calculated
theorem total_volume_collection :
  total_volume = 3060000 := by
  sorry

end total_volume_collection_l33_33273


namespace students_remaining_after_fifth_stop_l33_33805

theorem students_remaining_after_fifth_stop (initial_students : ℕ) (stops : ℕ) :
  initial_students = 60 →
  stops = 5 →
  (∀ n, (n < stops → ∃ k, n = 3 * k + 1) → ∀ x, x = initial_students * ((2 : ℚ) / 3)^stops) →
  initial_students * ((2 : ℚ) / 3)^stops = (640 / 81 : ℚ) :=
by
  intros h_initial h_stops h_formula
  sorry

end students_remaining_after_fifth_stop_l33_33805


namespace counterexample_to_proposition_l33_33611

theorem counterexample_to_proposition (a b : ℝ) (ha : a = 1) (hb : b = -1) :
  a > b ∧ ¬ (1 / a < 1 / b) :=
by
  sorry

end counterexample_to_proposition_l33_33611
