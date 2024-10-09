import Mathlib

namespace tiger_speed_l1448_144867

variable (v_t : ℝ) (hours_head_start : ℝ := 5) (hours_zebra_to_catch : ℝ := 6) (speed_zebra : ℝ := 55)

-- Define the distance covered by the tiger and the zebra
def distance_tiger (v_t : ℝ) (hours : ℝ) : ℝ := v_t * hours
def distance_zebra (hours : ℝ) (speed_zebra : ℝ) : ℝ := speed_zebra * hours

theorem tiger_speed :
  v_t * hours_head_start + v_t * hours_zebra_to_catch = distance_zebra hours_zebra_to_catch speed_zebra →
  v_t = 30 :=
by
  sorry

end tiger_speed_l1448_144867


namespace ab_divisibility_l1448_144885

theorem ab_divisibility (a b : ℕ) (h_a : a ≥ 2) (h_b : b ≥ 2) : 
  (ab - 1) % ((a - 1) * (b - 1)) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) :=
sorry

end ab_divisibility_l1448_144885


namespace selling_price_when_profit_equals_loss_l1448_144824

theorem selling_price_when_profit_equals_loss (CP SP Rs_57 : ℕ) (h1: CP = 50) (h2: Rs_57 = 57) (h3: Rs_57 - CP = CP - SP) : 
  SP = 43 := by
  sorry

end selling_price_when_profit_equals_loss_l1448_144824


namespace remove_green_balls_l1448_144889

theorem remove_green_balls (total_balls green_balls yellow_balls x : ℕ) 
  (h1 : total_balls = 600)
  (h2 : green_balls = 420)
  (h3 : yellow_balls = 180)
  (h4 : green_balls = 70 * total_balls / 100)
  (h5 : yellow_balls = total_balls - green_balls)
  (h6 : (green_balls - x) = 60 * (total_balls - x) / 100) :
  x = 150 := 
by {
  -- sorry placeholder for proof.
  sorry
}

end remove_green_balls_l1448_144889


namespace profit_percent_is_25_l1448_144844

noncomputable def SP : ℝ := sorry
noncomputable def CP : ℝ := 0.80 * SP
noncomputable def Profit : ℝ := SP - CP
noncomputable def ProfitPercent : ℝ := (Profit / CP) * 100

theorem profit_percent_is_25 :
  ProfitPercent = 25 :=
by
  sorry

end profit_percent_is_25_l1448_144844


namespace ratio_of_time_charged_l1448_144871

theorem ratio_of_time_charged (P K M : ℕ) (r : ℚ) 
  (h1 : P + K + M = 144) 
  (h2 : P = r * K)
  (h3 : P = 1/3 * M)
  (h4 : M = K + 80) : 
  r = 2 := 
  sorry

end ratio_of_time_charged_l1448_144871


namespace count_three_digit_values_with_double_sum_eq_six_l1448_144832

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_three_digit (x : ℕ) : Prop := 
  100 ≤ x ∧ x < 1000

theorem count_three_digit_values_with_double_sum_eq_six :
  ∃ count : ℕ, is_three_digit count ∧ (
    (∀ x, is_three_digit x → sum_of_digits (sum_of_digits x) = 6) ↔ count = 30
  ) :=
sorry

end count_three_digit_values_with_double_sum_eq_six_l1448_144832


namespace translation_coordinates_l1448_144845

theorem translation_coordinates
  (a b : ℝ)
  (h₁ : 4 = a + 2)
  (h₂ : -3 = b - 6) :
  (a, b) = (2, 3) :=
by
  sorry

end translation_coordinates_l1448_144845


namespace avg_math_chem_l1448_144886

variables (M P C : ℕ)

def total_marks (M P : ℕ) := M + P = 50
def chemistry_marks (P C : ℕ) := C = P + 20

theorem avg_math_chem (M P C : ℕ) (h1 : total_marks M P) (h2 : chemistry_marks P C) :
  (M + C) / 2 = 35 :=
by
  sorry

end avg_math_chem_l1448_144886


namespace ab_value_l1448_144804

theorem ab_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 30) (h4 : 3 * a * b + 5 * a = 4 * b + 180) : a * b = 29 :=
sorry

end ab_value_l1448_144804


namespace quadratic_roots_ratio_l1448_144820

theorem quadratic_roots_ratio (a b c : ℝ) (h1 : ∀ (s1 s2 : ℝ), s1 * s2 = a → s1 + s2 = -c → 3 * s1 + 3 * s2 = -a → 9 * s1 * s2 = b) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  b / c = 27 := sorry

end quadratic_roots_ratio_l1448_144820


namespace binomial_coeff_sum_l1448_144846

-- Define the problem: compute the numerical sum of the binomial coefficients
theorem binomial_coeff_sum (a b : ℕ) (h_a1 : a = 1) (h_b1 : b = 1) : 
  (a + b) ^ 8 = 256 :=
by
  -- Therefore, the sum must be 256
  sorry

end binomial_coeff_sum_l1448_144846


namespace angle_in_fourth_quadrant_l1448_144854

theorem angle_in_fourth_quadrant (α : ℝ) (h : 0 < α ∧ α < 90) : 270 < 360 - α ∧ 360 - α < 360 :=
by
  sorry

end angle_in_fourth_quadrant_l1448_144854


namespace problem_1_problem_2_l1448_144808

def f (x a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x + 3)
def g (x : ℝ) : ℝ := abs (2 * x - 3) + 2

theorem problem_1 (x : ℝ) :
  abs (g x) < 5 → 0 < x ∧ x < 3 :=
sorry

theorem problem_2 (a : ℝ) :
  (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) →
  (a ≥ -1 ∨ a ≤ -5) :=
sorry

end problem_1_problem_2_l1448_144808


namespace solve_equation_l1448_144837

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l1448_144837


namespace average_remaining_two_numbers_l1448_144893

theorem average_remaining_two_numbers 
  (a1 a2 a3 a4 a5 a6 : ℝ)
  (h_avg_6 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 2.80)
  (h_avg_2_1 : (a1 + a2) / 2 = 2.4)
  (h_avg_2_2 : (a3 + a4) / 2 = 2.3) :
  (a5 + a6) / 2 = 3.7 :=
by
  sorry

end average_remaining_two_numbers_l1448_144893


namespace find_divisor_l1448_144894

variable (r q d v : ℕ)
variable (h1 : r = 8)
variable (h2 : q = 43)
variable (h3 : d = 997)

theorem find_divisor : d = v * q + r → v = 23 :=
by
  sorry

end find_divisor_l1448_144894


namespace average_of_w_x_z_eq_one_sixth_l1448_144827

open Real

variable {w x y z t : ℝ}

theorem average_of_w_x_z_eq_one_sixth
  (h1 : 3 / w + 3 / x + 3 / z = 3 / (y + t))
  (h2 : w * x * z = y + t)
  (h3 : w * z + x * t + y * z = 3 * w + 3 * x + 3 * z) :
  (w + x + z) / 3 = 1 / 6 :=
by 
  sorry

end average_of_w_x_z_eq_one_sixth_l1448_144827


namespace more_birds_than_storks_l1448_144813

-- Defining the initial number of birds
def initial_birds : ℕ := 2

-- Defining the number of birds that joined
def additional_birds : ℕ := 5

-- Defining the number of storks that joined
def storks : ℕ := 4

-- Defining the total number of birds
def total_birds : ℕ := initial_birds + additional_birds

-- Defining the problem statement in Lean 4
theorem more_birds_than_storks : (total_birds - storks) = 3 := by
  sorry

end more_birds_than_storks_l1448_144813


namespace find_x_l1448_144887

-- Let \( x \) be a real number.
variable (x : ℝ)

-- Condition given in the problem.
def condition : Prop := x = (3 / 7) * x + 200

-- The main statement to be proved.
theorem find_x (h : condition x) : x = 350 :=
  sorry

end find_x_l1448_144887


namespace triangle_height_l1448_144870

theorem triangle_height (x y : ℝ) :
  let area := (x^3 * y)^2
  let base := (2 * x * y)^2
  base ≠ 0 →
  (2 * area) / base = x^4 / 2 :=
by
  sorry

end triangle_height_l1448_144870


namespace number_x_is_divided_by_l1448_144812

-- Define the conditions
variable (x y n : ℕ)
variable (cond1 : x = n * y + 4)
variable (cond2 : 2 * x = 8 * 3 * y + 3)
variable (cond3 : 13 * y - x = 1)

-- Define the statement to be proven
theorem number_x_is_divided_by : n = 11 :=
by
  sorry

end number_x_is_divided_by_l1448_144812


namespace sum_a1_to_a14_equals_zero_l1448_144842

theorem sum_a1_to_a14_equals_zero 
  (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 : ℝ) 
  (h1 : (1 + x - x^2)^3 * (1 - 2 * x^2)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 + a9 * x^9 + a10 * x^10 + a11 * x^11 + a12 * x^12 + a13 * x^13 + a14 * x^14) 
  (h2 : a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 = 1) 
  (h3 : a = 1) : 
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 = 0 := by
  sorry

end sum_a1_to_a14_equals_zero_l1448_144842


namespace odd_function_decreasing_function_max_min_values_on_interval_l1448_144834

variable (f : ℝ → ℝ)

axiom func_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom func_negative_for_positive : ∀ x : ℝ, (0 < x) → f x < 0
axiom func_value_at_one : f 1 = -2

theorem odd_function : ∀ x : ℝ, f (-x) = -f x := by
  have f_zero : f 0 = 0 := by sorry
  sorry

theorem decreasing_function : ∀ x₁ x₂ : ℝ, (x₁ < x₂) → f x₁ > f x₂ := by sorry

theorem max_min_values_on_interval :
  (f (-3) = 6) ∧ (f 3 = -6) := by sorry

end odd_function_decreasing_function_max_min_values_on_interval_l1448_144834


namespace probability_same_color_pair_l1448_144811

theorem probability_same_color_pair : 
  let total_shoes := 28
  let black_pairs := 8
  let brown_pairs := 4
  let gray_pairs := 2
  total_shoes = 2 * (black_pairs + brown_pairs + gray_pairs) → 
  ∃ (prob : ℚ), prob = 7 / 32 := by
  sorry

end probability_same_color_pair_l1448_144811


namespace problem_statement_l1448_144839

noncomputable def M (x y : ℝ) : ℝ := max x y
noncomputable def m (x y : ℝ) : ℝ := min x y

theorem problem_statement {p q r s t : ℝ} (h1 : p < q) (h2 : q < r) (h3 : r < s) (h4 : s < t) :
  M (M p (m q r)) (m s (m p t)) = q :=
by
  sorry

end problem_statement_l1448_144839


namespace functional_equation_odd_l1448_144815

   variable {R : Type*} [AddCommGroup R] [Module ℝ R]

   def isOdd (f : ℝ → ℝ) : Prop :=
     ∀ x : ℝ, f (-x) = -f x

   theorem functional_equation_odd (f : ℝ → ℝ)
       (h_fun : ∀ x y : ℝ, f (x + y) = f x + f y) : isOdd f :=
   by
     sorry
   
end functional_equation_odd_l1448_144815


namespace tan_of_cos_l1448_144817

theorem tan_of_cos (α : ℝ) (h_cos : Real.cos α = -4 / 5) (h_alpha : 0 < α ∧ α < Real.pi) : 
  Real.tan α = -3 / 4 :=
sorry

end tan_of_cos_l1448_144817


namespace cylinder_h_over_r_equals_one_l1448_144800

theorem cylinder_h_over_r_equals_one
  (A : ℝ) (r h : ℝ)
  (h_surface_area : A = 2 * π * r^2 + 2 * π * r * h)
  (V : ℝ := π * r^2 * h)
  (max_V : ∀ r' h', (A = 2 * π * r'^2 + 2 * π * r' * h') → (π * r'^2 * h' ≤ V) → (r' = r ∧ h' = h)) :
  h / r = 1 := by
sorry

end cylinder_h_over_r_equals_one_l1448_144800


namespace positive_number_property_l1448_144841

theorem positive_number_property (x : ℝ) (h : x > 0) (hx : (x / 100) * x = 9) : x = 30 := by
  sorry

end positive_number_property_l1448_144841


namespace area_of_rectangle_is_108_l1448_144858

-- Define the conditions and parameters
variables (P Q R S : Type) (diameter : ℝ) (height : ℝ) (width : ℝ) (area : ℝ)
variable (isTangentToSides : Prop)
variable (centersFormLineParallelToLongerSide : Prop)

-- Assume the given conditions
axiom h1 : diameter = 6
axiom h2 : isTangentToSides
axiom h3 : centersFormLineParallelToLongerSide

-- Define the goal to prove
theorem area_of_rectangle_is_108 (P Q R S : Type) (diameter : ℝ) (height : ℝ) (width : ℝ) (area : ℝ)
    (isTangentToSides : Prop) (centersFormLineParallelToLongerSide : Prop)
    (h1 : diameter = 6)
    (h2 : isTangentToSides)
    (h3 : centersFormLineParallelToLongerSide) :
    area = 108 :=
by
  -- Lean code requires an actual proof here, but for now, we'll use sorry.
  sorry

end area_of_rectangle_is_108_l1448_144858


namespace jill_and_bob_payment_l1448_144880

-- Definitions of the conditions
def price_of_first_house (X : ℝ) := X
def price_of_second_house (Y X : ℝ) := 2 * X

theorem jill_and_bob_payment :
  ∃ X, ∃ Y, Y = 2 * X ∧ X + Y = 600000 ∧ X = 200000 :=
by
  sorry

end jill_and_bob_payment_l1448_144880


namespace imaginary_part_of_z_l1448_144816

open Complex

theorem imaginary_part_of_z :
  ∃ z: ℂ, (3 - 4 * I) * z = abs (4 + 3 * I) ∧ z.im = 4 / 5 :=
by
  sorry

end imaginary_part_of_z_l1448_144816


namespace minimize_f_l1448_144829

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l1448_144829


namespace find_f_neg2007_l1448_144890

variable (f : ℝ → ℝ)

-- Conditions
axiom cond1 (x y w : ℝ) (hx : x > y) (hw : f x + x ≥ w ∧ w ≥ f y + y) : 
  ∃ z ∈ Set.Icc y x, f z = w - z

axiom cond2 : ∃ u, f u = 0 ∧ ∀ v, f v = 0 → u ≤ v

axiom cond3 : f 0 = 1

axiom cond4 : f (-2007) ≤ 2008

axiom cond5 (x y : ℝ) : f x * f y = f (x * f y + y * f x + x * y)

theorem find_f_neg2007 : f (-2007) = 2008 := 
sorry

end find_f_neg2007_l1448_144890


namespace sum_of_numbers_l1448_144873

theorem sum_of_numbers : 217 + 2.017 + 0.217 + 2.0017 = 221.2357 :=
by
  sorry

end sum_of_numbers_l1448_144873


namespace probability_point_in_circle_l1448_144851

theorem probability_point_in_circle (r : ℝ) (h: r = 2) :
  let side_length := 2 * r
  let area_square := side_length ^ 2
  let area_circle := Real.pi * r ^ 2
  (area_circle / area_square) = Real.pi / 4 :=
by
  sorry

end probability_point_in_circle_l1448_144851


namespace sum_first_8_terms_l1448_144826

variable {α : Type*} [LinearOrderedField α]

-- Define the arithmetic sequence
def arithmetic_sequence (a_1 d : α) (n : ℕ) : α := a_1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a_1 d : α) (n : ℕ) : α :=
  (n * (2 * a_1 + (n - 1) * d)) / 2

-- Define the given condition
variable (a_1 d : α)
variable (h : arithmetic_sequence a_1 d 3 = 20 - arithmetic_sequence a_1 d 6)

-- Statement of the problem
theorem sum_first_8_terms : sum_arithmetic_sequence a_1 d 8 = 80 :=
by
  sorry

end sum_first_8_terms_l1448_144826


namespace compare_neg_fractions_l1448_144897

theorem compare_neg_fractions : - (3 / 5 : ℚ) < - (1 / 5 : ℚ) :=
by
  sorry

end compare_neg_fractions_l1448_144897


namespace complement_intersection_l1448_144840

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {2, 4}
def N : Set ℕ := {3, 5}

theorem complement_intersection (hU: U = {1, 2, 3, 4, 5}) (hM: M = {2, 4}) (hN: N = {3, 5}) : 
  (U \ M) ∩ N = {3, 5} := 
by 
  sorry

end complement_intersection_l1448_144840


namespace ratio_male_whales_l1448_144862

def num_whales_first_trip_males : ℕ := 28
def num_whales_first_trip_females : ℕ := 56
def num_whales_second_trip_babies : ℕ := 8
def num_whales_second_trip_parents_males : ℕ := 8
def num_whales_second_trip_parents_females : ℕ := 8
def num_whales_third_trip_females : ℕ := 56
def total_whales : ℕ := 178

theorem ratio_male_whales (M : ℕ) (ratio : ℕ × ℕ) 
  (h_total_whales : num_whales_first_trip_males + num_whales_first_trip_females 
    + num_whales_second_trip_babies + num_whales_second_trip_parents_males 
    + num_whales_second_trip_parents_females + M + num_whales_third_trip_females = total_whales) 
  (h_ratio : ratio = ((M : ℕ) / Nat.gcd M num_whales_first_trip_males, 
                       num_whales_first_trip_males / Nat.gcd M num_whales_first_trip_males)) 
  : ratio = (1, 2) :=
by
  sorry

end ratio_male_whales_l1448_144862


namespace systematic_sampling_eighth_group_l1448_144888

theorem systematic_sampling_eighth_group
  (total_employees : ℕ)
  (target_sample : ℕ)
  (third_group_value : ℕ)
  (group_count : ℕ)
  (common_difference : ℕ)
  (eighth_group_value : ℕ) :
  total_employees = 840 →
  target_sample = 42 →
  third_group_value = 44 →
  group_count = total_employees / target_sample →
  common_difference = group_count →
  eighth_group_value = third_group_value + (8 - 3) * common_difference →
  eighth_group_value = 144 :=
sorry

end systematic_sampling_eighth_group_l1448_144888


namespace sqrt_product_l1448_144847

open Real

theorem sqrt_product :
  sqrt 54 * sqrt 48 * sqrt 6 = 72 * sqrt 3 := by
  sorry

end sqrt_product_l1448_144847


namespace gravity_anomaly_l1448_144830

noncomputable def gravity_anomaly_acceleration
  (α : ℝ) (v₀ : ℝ) (g : ℝ) (S : ℝ) (g_a : ℝ) : Prop :=
  α = 30 ∧ v₀ = 10 ∧ g = 10 ∧ S = 3 * Real.sqrt 3 → g_a = 250

theorem gravity_anomaly (α v₀ g S g_a : ℝ) : gravity_anomaly_acceleration α v₀ g S g_a :=
by
  intro h
  sorry

end gravity_anomaly_l1448_144830


namespace capacity_of_new_bucket_l1448_144803

def number_of_old_buckets : ℕ := 26
def capacity_of_old_bucket : ℝ := 13.5
def total_volume : ℝ := number_of_old_buckets * capacity_of_old_bucket
def number_of_new_buckets : ℕ := 39

theorem capacity_of_new_bucket :
  total_volume / number_of_new_buckets = 9 :=
sorry

end capacity_of_new_bucket_l1448_144803


namespace frac_equiv_l1448_144853

-- Define the given values of x and y.
def x : ℚ := 2 / 7
def y : ℚ := 8 / 11

-- Define the statement to prove.
theorem frac_equiv : (7 * x + 11 * y) / (77 * x * y) = 5 / 8 :=
by
  -- The proof will go here (use 'sorry' for now)
  sorry

end frac_equiv_l1448_144853


namespace totalCroissants_is_18_l1448_144836

def jorgeCroissants : ℕ := 7
def giulianaCroissants : ℕ := 5
def matteoCroissants : ℕ := 6

def totalCroissants : ℕ := jorgeCroissants + giulianaCroissants + matteoCroissants

theorem totalCroissants_is_18 : totalCroissants = 18 := by
  -- Proof will be provided here
  sorry

end totalCroissants_is_18_l1448_144836


namespace interval_length_l1448_144872

theorem interval_length (c : ℝ) (h : ∀ x : ℝ, 3 ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ c → 
                             (3 * (x) + 4 ≤ c ∧ 3 ≤ 3 * x + 4)) :
  (∃ c : ℝ, ((c - 4) / 3) - ((-1) / 3) = 15) → (c - 3 = 45) :=
sorry

end interval_length_l1448_144872


namespace sequence_sum_square_l1448_144865

-- Definition of the sum of the symmetric sequence.
def sequence_sum (n : ℕ) : ℕ :=
  (List.range' 1 (n+1)).sum + (List.range' 1 n).sum

-- The conjecture that the sum of the sequence equals n^2.
theorem sequence_sum_square (n : ℕ) : sequence_sum n = n^2 := by
  sorry

end sequence_sum_square_l1448_144865


namespace power_equation_value_l1448_144802

theorem power_equation_value (n : ℕ) (h : n = 20) : n ^ (n / 2) = 102400000000000000000 := by
  sorry

end power_equation_value_l1448_144802


namespace sqrt_infinite_series_eq_two_l1448_144883

theorem sqrt_infinite_series_eq_two (m : ℝ) (hm : 0 < m) :
  (m ^ 2 = 2 + m) → m = 2 :=
by {
  sorry
}

end sqrt_infinite_series_eq_two_l1448_144883


namespace odometer_problem_l1448_144899

theorem odometer_problem
    (x a b c : ℕ)
    (h_dist : 60 * x = (100 * b + 10 * c + a) - (100 * a + 10 * b + c))
    (h_b_ge_1 : b ≥ 1)
    (h_sum_le_9 : a + b + c ≤ 9) :
    a^2 + b^2 + c^2 = 29 :=
sorry

end odometer_problem_l1448_144899


namespace ratio_of_a_to_b_l1448_144882

variable (a b x m : ℝ)
variable (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_x : x = 1.25 * a) (h_m : m = 0.6 * b)
variable (h_ratio : m / x = 0.6)

theorem ratio_of_a_to_b (h_x : x = 1.25 * a) (h_m : m = 0.6 * b) (h_ratio : m / x = 0.6) : a / b = 0.8 :=
by
  sorry

end ratio_of_a_to_b_l1448_144882


namespace bananas_count_l1448_144866

theorem bananas_count 
  (total_oranges : ℕ)
  (total_percentage_good : ℝ)
  (percentage_rotten_oranges : ℝ)
  (percentage_rotten_bananas : ℝ)
  (total_good_fruits_percentage : ℝ)
  (B : ℝ) :
  total_oranges = 600 →
  total_percentage_good = 0.85 →
  percentage_rotten_oranges = 0.15 →
  percentage_rotten_bananas = 0.03 →
  total_good_fruits_percentage = 0.898 →
  B = 400  :=
by
  intros h_oranges h_good_percentage h_rotten_oranges h_rotten_bananas h_good_fruits_percentage
  sorry

end bananas_count_l1448_144866


namespace total_asphalt_used_1520_tons_l1448_144860

noncomputable def asphalt_used (L W : ℕ) (asphalt_per_100m2 : ℕ) : ℕ :=
  (L * W / 100) * asphalt_per_100m2

theorem total_asphalt_used_1520_tons :
  asphalt_used 800 50 3800 = 1520000 := by
  sorry

end total_asphalt_used_1520_tons_l1448_144860


namespace exists_large_cube_construction_l1448_144828

theorem exists_large_cube_construction (n : ℕ) :
  ∃ N : ℕ, ∀ n > N, ∃ k : ℕ, k^3 = n :=
sorry

end exists_large_cube_construction_l1448_144828


namespace num_points_on_ellipse_with_area_l1448_144831

-- Define the line equation
def line_eq (x y : ℝ) : Prop := (x / 4) + (y / 3) = 1

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1

-- Define the area condition for the triangle
def area_condition (xA yA xB yB xP yP : ℝ) : Prop :=
  abs (xA * (yB - yP) + xB * (yP - yA) + xP * (yA - yB)) = 6

-- Define the main theorem statement
theorem num_points_on_ellipse_with_area (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  ∃ P1 P2 : ℝ × ℝ, 
    (ellipse_eq P1.1 P1.2) ∧ 
    (ellipse_eq P2.1 P2.2) ∧ 
    (area_condition A.1 A.2 B.1 B.2 P1.1 P1.2) ∧ 
    (area_condition A.1 A.2 B.1 B.2 P2.1 P2.2) ∧ 
    P1 ≠ P2 := sorry

end num_points_on_ellipse_with_area_l1448_144831


namespace dima_story_telling_l1448_144891

theorem dima_story_telling (initial_spoons final_spoons : ℕ) 
  (h1 : initial_spoons = 26) (h2 : final_spoons = 33696)
  (h3 : ∃ (n : ℕ), final_spoons = initial_spoons * (2^5 * 3^4) * 13) : 
  ∃ n : ℕ, n = 9 := 
sorry

end dima_story_telling_l1448_144891


namespace largest_square_side_l1448_144818

theorem largest_square_side (width length : ℕ) (h_width : width = 63) (h_length : length = 42) : 
  Nat.gcd width length = 21 :=
by
  rw [h_width, h_length]
  sorry

end largest_square_side_l1448_144818


namespace find_a7_l1448_144874

variable {a : ℕ → ℝ}

-- Conditions
def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 1 < q ∧ ∀ n : ℕ, a (n + 1) = a n * q

axiom a3_eq_4 : a 3 = 4
axiom harmonic_condition : (1 / a 1 + 1 / a 5 = 5 / 8)
axiom increasing_geometric : is_increasing_geometric_sequence a

-- The problem is to prove that a 7 = 16 given the above conditions.
theorem find_a7 : a 7 = 16 :=
by
  -- Proof goes here
  sorry

end find_a7_l1448_144874


namespace ineq_x4_y4_l1448_144868

theorem ineq_x4_y4 (x y : ℝ) (h1 : x > Real.sqrt 2) (h2 : y > Real.sqrt 2) :
    x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by
  sorry

end ineq_x4_y4_l1448_144868


namespace agency_comparison_l1448_144895

variable (days m : ℝ)

theorem agency_comparison (h : 20.25 * days + 0.14 * m < 18.25 * days + 0.22 * m) : m > 25 * days :=
by
  sorry

end agency_comparison_l1448_144895


namespace x_intercept_is_one_l1448_144833

theorem x_intercept_is_one (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, -1)) (h2 : (x2, y2) = (-2, 3)) :
    ∃ x : ℝ, (0 = ((y2 - y1) / (x2 - x1)) * (x - x1) + y1) ∧ x = 1 :=
by
  sorry

end x_intercept_is_one_l1448_144833


namespace fraction_value_l1448_144869

theorem fraction_value : (5 * 7 : ℝ) / 10 = 3.5 := by
  sorry

end fraction_value_l1448_144869


namespace complement_of_A_in_U_l1448_144850

-- Conditions definitions
def U : Set ℕ := {x | x ≤ 5}
def A : Set ℕ := {x | 2 * x - 5 < 0}

-- Theorem stating the question and the correct answer
theorem complement_of_A_in_U :
  U \ A = {x | 3 ≤ x ∧ x ≤ 5} :=
by
  -- The proof will go here
  sorry

end complement_of_A_in_U_l1448_144850


namespace bologna_sandwiches_l1448_144806

variable (C B P : ℕ)

theorem bologna_sandwiches (h1 : C = 1) (h2 : B = 7) (h3 : P = 8)
                          (h4 : C + B + P = 16) (h5 : 80 / 16 = 5) :
                          B * 5 = 35 :=
by
  -- omit the proof part
  sorry

end bologna_sandwiches_l1448_144806


namespace peter_takes_last_stone_l1448_144859

theorem peter_takes_last_stone (n : ℕ) (h : ∀ p, Nat.Prime p → p < n) :
  ∃ P, ∀ stones: ℕ, stones > n^2 → (∃ k : ℕ, 
  ((k = 1 ∨ (∃ p : ℕ, Nat.Prime p ∧ p < n ∧ k = p) ∨ (∃ m : ℕ, k = m * n)) ∧
  stones ≥ k ∧ stones - k > n^2) →
  P = stones - k) := 
sorry

end peter_takes_last_stone_l1448_144859


namespace minimum_value_f_maximum_value_f_l1448_144864

-- Problem 1: Minimum value of f(x) = 12/x + 3x for x > 0
theorem minimum_value_f (x : ℝ) (h : x > 0) : 
  (12 / x + 3 * x) ≥ 12 :=
sorry

-- Problem 2: Maximum value of f(x) = x(1 - 3x) for 0 < x < 1/3
theorem maximum_value_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1 / 3) :
  x * (1 - 3 * x) ≤ 1 / 12 :=
sorry

end minimum_value_f_maximum_value_f_l1448_144864


namespace greatest_integer_inequality_l1448_144876

theorem greatest_integer_inequality : 
  ⌊ (3 ^ 100 + 2 ^ 100 : ℝ) / (3 ^ 96 + 2 ^ 96) ⌋ = 80 :=
by
  sorry

end greatest_integer_inequality_l1448_144876


namespace max_area_garden_l1448_144825

/-- Given a rectangular garden with a total perimeter of 480 feet and one side twice as long as another,
    prove that the maximum area of the garden is 12800 square feet. -/
theorem max_area_garden (l w : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 480) : l * w = 12800 := 
sorry

end max_area_garden_l1448_144825


namespace degree_measure_OC1D_l1448_144884

/-- Define points on the sphere -/
structure Point (latitude longitude : ℝ) :=
(lat : ℝ := latitude)
(long : ℝ := longitude)

noncomputable def cos_deg (deg : ℝ) : ℝ := Real.cos (deg * Real.pi / 180)

noncomputable def angle_OC1D : ℝ :=
  Real.arccos ((cos_deg 44) * (cos_deg (-123)))

/-- The main theorem: the degree measure of ∠OC₁D is 113 -/
theorem degree_measure_OC1D :
  angle_OC1D = 113 := sorry

end degree_measure_OC1D_l1448_144884


namespace sum_from_one_to_twelve_l1448_144879

-- Define the sum of an arithmetic series
def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  (n * (a + l)) / 2

-- Theorem stating the sum of numbers from 1 to 12
theorem sum_from_one_to_twelve : sum_arithmetic_series 12 1 12 = 78 := by
  sorry

end sum_from_one_to_twelve_l1448_144879


namespace range_of_a_l1448_144849

theorem range_of_a (x y a : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∀ (x y : ℝ), 0 < x → 0 < y → (y / 4 - (Real.cos x)^2) ≥ a * (Real.sin x) - 9 / y) ↔ (-3 ≤ a ∧ a ≤ 3) :=
sorry

end range_of_a_l1448_144849


namespace inequality_chain_l1448_144843

theorem inequality_chain (a : ℝ) (h : a - 1 > 0) : -a < -1 ∧ -1 < 1 ∧ 1 < a := by
  sorry

end inequality_chain_l1448_144843


namespace percentage_increase_of_x_compared_to_y_l1448_144857

-- We are given that y = 0.5 * z and x = 0.6 * z
-- We need to prove that the percentage increase of x compared to y is 20%

theorem percentage_increase_of_x_compared_to_y (x y z : ℝ) 
  (h1 : y = 0.5 * z) 
  (h2 : x = 0.6 * z) : 
  (x / y - 1) * 100 = 20 :=
by 
  -- Placeholder for actual proof
  sorry

end percentage_increase_of_x_compared_to_y_l1448_144857


namespace final_image_of_F_is_correct_l1448_144823

-- Define the initial F position as a struct
structure Position where
  base : (ℝ × ℝ)
  stem : (ℝ × ℝ)

-- Function to rotate a point 90 degrees counterclockwise around the origin
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Function to reflect a point in the x-axis
def reflectX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Function to rotate a point by 180 degrees around the origin (half turn)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- Define the initial state of F
def initialFPosition : Position := {
  base := (-1, 0),  -- Base along the negative x-axis
  stem := (0, -1)   -- Stem along the negative y-axis
}

-- Perform all transformations on the Position of F
def transformFPosition (pos : Position) : Position :=
  let afterRotation90 := Position.mk (rotate90 pos.base) (rotate90 pos.stem)
  let afterReflectionX := Position.mk (reflectX afterRotation90.base) (reflectX afterRotation90.stem)
  let finalPosition := Position.mk (rotate180 afterReflectionX.base) (rotate180 afterReflectionX.stem)
  finalPosition

-- Define the target final position we expect
def finalFPosition : Position := {
  base := (0, 1),   -- Base along the positive y-axis
  stem := (1, 0)    -- Stem along the positive x-axis
}

-- The theorem statement: After the transformations, the position of F
-- should match the final expected position
theorem final_image_of_F_is_correct :
  transformFPosition initialFPosition = finalFPosition := by
  sorry

end final_image_of_F_is_correct_l1448_144823


namespace sum_c_2017_l1448_144848

def a (n : ℕ) : ℕ := 3 * n + 1

def b (n : ℕ) : ℕ := 4^(n-1)

def c (n : ℕ) : ℕ := if n = 1 then 7 else 3 * 4^(n-1)

theorem sum_c_2017 : (Finset.range 2017).sum c = 4^2017 + 3 :=
by
  -- definitions and required assumptions
  sorry

end sum_c_2017_l1448_144848


namespace number_of_exchanges_l1448_144819

theorem number_of_exchanges (n : ℕ) (hz_initial : ℕ) (hl_initial : ℕ) 
  (hz_decrease : ℕ) (hl_decrease : ℕ) (k : ℕ) :
  hz_initial = 200 →
  hl_initial = 20 →
  hz_decrease = 6 →
  hl_decrease = 1 →
  k = 11 →
  (hz_initial - n * hz_decrease) = k * (hl_initial - n * hl_decrease) →
  n = 4 := 
sorry

end number_of_exchanges_l1448_144819


namespace math_problem_l1448_144861

noncomputable def f (x : ℝ) := (x^9 - 27*x^6 + 216*x^3 - 512) / (x^3 - 8)

theorem math_problem : f 6 = 43264 := by
  sorry

end math_problem_l1448_144861


namespace sin_law_ratio_l1448_144805

theorem sin_law_ratio {A B C : ℝ} {a b c : ℝ} (hA : a = 1) (hSinA : Real.sin A = 1 / 3) :
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 3 := 
  sorry

end sin_law_ratio_l1448_144805


namespace proj_a_b_l1448_144892

open Real

def vector (α : Type*) := (α × α)

noncomputable def dot_product (a b: vector ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (v: vector ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

noncomputable def projection (a b: vector ℝ) : ℝ := (dot_product a b) / (magnitude b)

-- Define the vectors a and b
def a : vector ℝ := (-1, 3)
def b : vector ℝ := (3, 4)

-- The projection of a in the direction of b
theorem proj_a_b : projection a b = 9 / 5 := 
  by sorry

end proj_a_b_l1448_144892


namespace sin_225_cos_225_l1448_144810

noncomputable def sin_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2

noncomputable def cos_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem sin_225 : sin_225_eq_neg_sqrt2_div_2 := by
  sorry

theorem cos_225 : cos_225_eq_neg_sqrt2_div_2 := by
  sorry

end sin_225_cos_225_l1448_144810


namespace quadratic_b_value_l1448_144814
open Real

theorem quadratic_b_value (b n : ℝ) 
  (h1: b < 0) 
  (h2: ∀ x, x^2 + b * x + (1 / 4) = (x + n)^2 + (1 / 16)) :
  b = - (sqrt 3 / 2) :=
by
  -- sorry is used to skip the proof
  sorry

end quadratic_b_value_l1448_144814


namespace simplify_expression_l1448_144881

theorem simplify_expression : (2^3002 * 3^3004) / 6^3003 = 3 / 4 := by
  sorry

end simplify_expression_l1448_144881


namespace find_a3_l1448_144821

noncomputable def S (n : ℕ) : ℤ := 2 * n^2 - 1
noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem find_a3 : a 3 = 10 := by
  sorry

end find_a3_l1448_144821


namespace range_of_a_l1448_144852

def p (a m : ℝ) : Prop := m^2 + 12 * a^2 < 7 * a * m ∧ a > 0
def q (m : ℝ) : Prop := 1 < m ∧ m < 3 / 2

theorem range_of_a (a : ℝ) :
  (∀ m : ℝ, p a m → q m) → 
  (∃ (a_lower a_upper : ℝ), a_lower ≤ a ∧ a ≤ a_upper ∧ a_lower = 1 / 3 ∧ a_upper = 3 / 8) :=
sorry

end range_of_a_l1448_144852


namespace proportion_x_l1448_144838

theorem proportion_x (x : ℝ) (h : 3 / 12 = x / 16) : x = 4 :=
sorry

end proportion_x_l1448_144838


namespace part1_solution_set_part2_range_a_l1448_144877

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 1)

theorem part1_solution_set :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3 / 2} ∪ {x : ℝ | x ≥ 3 / 2} := 
sorry

theorem part2_range_a (a : ℝ) : 
  (∀ x : ℝ, f x ≥ a^2 - a) ↔ (-1 ≤ a ∧ a ≤ 2) := 
sorry

end part1_solution_set_part2_range_a_l1448_144877


namespace aquarium_height_l1448_144809

theorem aquarium_height (h : ℝ) (V : ℝ) (final_volume : ℝ) :
  let length := 4
  let width := 6
  let halfway_volume := (length * width * h) / 2
  let spilled_volume := halfway_volume / 2
  let tripled_volume := 3 * spilled_volume
  tripled_volume = final_volume →
  final_volume = 54 →
  h = 3 := by
  intros
  sorry

end aquarium_height_l1448_144809


namespace train_scheduled_speed_l1448_144801

theorem train_scheduled_speed (a v : ℝ) (hv : 0 < v)
  (h1 : a / v - a / (v + 5) = 1 / 3)
  (h2 : a / (v - 5) - a / v = 5 / 12) : v = 45 :=
by
  sorry

end train_scheduled_speed_l1448_144801


namespace cab_driver_income_l1448_144878

theorem cab_driver_income (x : ℕ)
  (h1 : 50 + 60 + 65 + 70 + x = 5 * 58) :
  x = 45 :=
by
  sorry

end cab_driver_income_l1448_144878


namespace number_of_days_worked_l1448_144835

theorem number_of_days_worked (total_toys_per_week : ℕ) (toys_per_day : ℕ) (h₁ : total_toys_per_week = 6000) (h₂ : toys_per_day = 1500) : (total_toys_per_week / toys_per_day) = 4 :=
by
  sorry

end number_of_days_worked_l1448_144835


namespace kevin_total_miles_l1448_144896

theorem kevin_total_miles : 
  ∃ (d1 d2 d3 d4 d5 : ℕ), 
  d1 = 60 / 6 ∧ 
  d2 = 60 / (6 + 6 * 1) ∧ 
  d3 = 60 / (6 + 6 * 2) ∧ 
  d4 = 60 / (6 + 6 * 3) ∧ 
  d5 = 60 / (6 + 6 * 4) ∧ 
  (d1 + d2 + d3 + d4 + d5) = 13 := 
by
  sorry

end kevin_total_miles_l1448_144896


namespace ratio_unit_price_l1448_144855

theorem ratio_unit_price
  (v : ℝ) (p : ℝ) (h_v : v > 0) (h_p : p > 0)
  (vol_A : ℝ := 1.25 * v)
  (price_A : ℝ := 0.85 * p) :
  (price_A / vol_A) / (p / v) = 17 / 25 :=
by
  sorry

end ratio_unit_price_l1448_144855


namespace smallest_b_l1448_144822

theorem smallest_b (b : ℕ) : 
  (b % 3 = 2) ∧ (b % 4 = 3) ∧ (b % 5 = 4) ∧ (b % 7 = 6) ↔ b = 419 :=
by sorry

end smallest_b_l1448_144822


namespace bureaucrats_total_l1448_144898

-- Define the parameters and conditions as stated in the problem
variables (a b c : ℕ)

-- Conditions stated in the problem
def condition_1 : Prop :=
  ∀ (i j : ℕ) (h1 : i ≠ j), 
    (10 * a * b = 10 * a * c ∧ 10 * b * c = 10 * a * b)

-- The main goal: proving the total number of bureaucrats
theorem bureaucrats_total (h1 : a = b) (h2 : b = c) (h3 : condition_1 a b c) : 
  3 * a = 120 :=
by sorry

end bureaucrats_total_l1448_144898


namespace karen_group_size_l1448_144875

theorem karen_group_size (total_students : ℕ) (zack_group_size number_of_groups : ℕ) (karen_group_size : ℕ) (h1 : total_students = 70) (h2 : zack_group_size = 14) (h3 : number_of_groups = total_students / zack_group_size) (h4 : number_of_groups = total_students / karen_group_size) : karen_group_size = 14 :=
by
  sorry

end karen_group_size_l1448_144875


namespace problem_statement_l1448_144856

theorem problem_statement (m n : ℝ) 
  (h₁ : m^2 - 1840 * m + 2009 = 0)
  (h₂ : n^2 - 1840 * n + 2009 = 0) : 
  (m^2 - 1841 * m + 2009) * (n^2 - 1841 * n + 2009) = 2009 :=
sorry

end problem_statement_l1448_144856


namespace additional_days_needed_is_15_l1448_144863

-- Definitions and conditions from the problem statement
def good_days_2013 : ℕ := 365 * 479 / 100  -- Number of good air quality days in 2013
def target_increase : ℕ := 20              -- Target increase in percentage for 2014
def additional_days_first_half_2014 : ℕ := 20 -- Additional good air quality days in first half of 2014 compared to 2013
def half_good_days_2013 : ℕ := good_days_2013 / 2 -- Good air quality days in first half of 2013

-- Target number of good air quality days for 2014
def target_days_2014 : ℕ := good_days_2013 * (100 + target_increase) / 100

-- Good air quality days in the first half of 2014
def good_days_first_half_2014 : ℕ := half_good_days_2013 + additional_days_first_half_2014

-- Additional good air quality days needed in the second half of 2014
def additional_days_2014_second_half (target_days good_days_first_half_2014 : ℕ) : ℕ := 
  target_days - good_days_first_half_2014 - half_good_days_2013

-- Final theorem verifying the number of additional days needed in the second half of 2014 is 15
theorem additional_days_needed_is_15 : 
  additional_days_2014_second_half target_days_2014 good_days_first_half_2014 = 15 :=
sorry

end additional_days_needed_is_15_l1448_144863


namespace second_tray_holds_l1448_144807

-- The conditions and the given constants
variables (x : ℕ) (h1 : 2 * x - 20 = 500)

-- The theorem proving the number of cups the second tray holds is 240 
theorem second_tray_holds (h2 : x = 260) : x - 20 = 240 := by
  sorry

end second_tray_holds_l1448_144807
