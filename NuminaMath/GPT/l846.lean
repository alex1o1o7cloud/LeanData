import Mathlib

namespace NUMINAMATH_GPT_ratio_addition_l846_84616

theorem ratio_addition (a b : ℕ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := 
by sorry

end NUMINAMATH_GPT_ratio_addition_l846_84616


namespace NUMINAMATH_GPT_simplify_expression_and_evaluate_at_zero_l846_84686

theorem simplify_expression_and_evaluate_at_zero :
  ((2 * (0 : ℝ) - 1) / (0 + 1) - 0 + 1) / ((0 - 2) / ((0 ^ 2) + 2 * 0 + 1)) = 0 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_simplify_expression_and_evaluate_at_zero_l846_84686


namespace NUMINAMATH_GPT_overall_average_of_25_results_l846_84631

theorem overall_average_of_25_results (first_12_avg last_12_avg thirteenth_result : ℝ) 
  (h1 : first_12_avg = 14) (h2 : last_12_avg = 17) (h3 : thirteenth_result = 78) :
  (12 * first_12_avg + thirteenth_result + 12 * last_12_avg) / 25 = 18 :=
by
  sorry

end NUMINAMATH_GPT_overall_average_of_25_results_l846_84631


namespace NUMINAMATH_GPT_digits_subtraction_eq_zero_l846_84632

theorem digits_subtraction_eq_zero (d A B : ℕ) (h1 : d > 8)
  (h2 : A < d) (h3 : B < d)
  (h4 : A * d + B + A * d + A = 2 * d + 3 * d + 4) :
  A - B = 0 :=
by sorry

end NUMINAMATH_GPT_digits_subtraction_eq_zero_l846_84632


namespace NUMINAMATH_GPT_tire_circumference_constant_l846_84638

/--
Given the following conditions:
1. Car speed v = 120 km/h
2. Tire rotation rate n = 400 rpm
3. Tire pressure P = 32 psi
4. Tire radius changes according to the formula R = R_0(1 + kP)
5. R_0 is the initial tire radius
6. k is a constant relating to the tire's elasticity
7. Change in tire pressure due to the incline is negligible

Prove that the circumference C of the tire is 5 meters.
-/
theorem tire_circumference_constant (v : ℝ) (n : ℝ) (P : ℝ) (R_0 : ℝ) (k : ℝ) 
  (h1 : v = 120 * 1000 / 3600) -- Car speed in m/s
  (h2 : n = 400 / 60)           -- Tire rotation rate in rps
  (h3 : P = 32)                 -- Tire pressure in psi
  (h4 : ∀ R P, R = R_0 * (1 + k * P)) -- Tire radius formula
  (h5 : ∀ P, P = 0)             -- Negligible change in tire pressure
  : C = 5 :=
  sorry

end NUMINAMATH_GPT_tire_circumference_constant_l846_84638


namespace NUMINAMATH_GPT_unique_B_squared_l846_84624

theorem unique_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B^4 = 0) : 
  ∃! B2 : Matrix (Fin 2) (Fin 2) ℝ, B2 = B * B :=
sorry

end NUMINAMATH_GPT_unique_B_squared_l846_84624


namespace NUMINAMATH_GPT_picture_area_l846_84674

theorem picture_area (x y : ℕ) (hx : 1 < x) (hy : 1 < y) 
  (h_area : (3 * x + 4) * (y + 3) = 60) : x * y = 15 := 
by 
  sorry

end NUMINAMATH_GPT_picture_area_l846_84674


namespace NUMINAMATH_GPT_solve_inequality_l846_84681

open Set

theorem solve_inequality :
  { x : ℝ | (2 * x - 2) / (x^2 - 5*x + 6) ≤ 3 } = Ioo (5/3) 2 ∪ Icc 3 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l846_84681


namespace NUMINAMATH_GPT_condition_on_p_l846_84646

theorem condition_on_p (p q r M : ℝ) (hq : 0 < q ∧ q < 100) (hr : 0 < r ∧ r < 100) (hM : 0 < M) :
  p > (100 * (q + r)) / (100 - q - r) → 
  M * (1 + p / 100) * (1 - q / 100) * (1 - r / 100) > M :=
by
  intro h
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_condition_on_p_l846_84646


namespace NUMINAMATH_GPT_vector_expression_l846_84640

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (i j k a b : V)
variables (h_i_j_k_non_coplanar : ∃ (l m n : ℝ), l • i + m • j + n • k = 0 → l = 0 ∧ m = 0 ∧ n = 0)
variables (h_a : a = (1 / 2 : ℝ) • i - j + k)
variables (h_b : b = 5 • i - 2 • j - k)

theorem vector_expression :
  4 • a - 3 • b = -13 • i + 2 • j + 7 • k :=
by
  sorry

end NUMINAMATH_GPT_vector_expression_l846_84640


namespace NUMINAMATH_GPT_min_value_of_f_value_of_a_l846_84666

-- Definition of the function f
def f (x : ℝ) : ℝ := abs (x + 2) + 2 * abs (x - 1)

-- Problem: Prove that the minimum value of f(x) is 3
theorem min_value_of_f : ∃ x : ℝ, f x = 3 := sorry

-- Additional definitions for the second part of the problem
def g (x a : ℝ) : ℝ := f x + x - a

-- Problem: Given that the solution set of g(x,a) < 0 is (m, n) and n - m = 6, prove that a = 8
theorem value_of_a (a : ℝ) (m n : ℝ) (h : ∀ x : ℝ, g x a < 0 ↔ m < x ∧ x < n) (h_interval : n - m = 6) : a = 8 := sorry

end NUMINAMATH_GPT_min_value_of_f_value_of_a_l846_84666


namespace NUMINAMATH_GPT_plane_split_into_regions_l846_84610

theorem plane_split_into_regions : 
  let line1 (x : ℝ) := 3 * x
  let line2 (x : ℝ) := (1 / 3) * x
  let line3 (x : ℝ) := 4 * x
  ∃ regions : ℕ, regions = 7 :=
by
  let line1 (x : ℝ) := 3 * x
  let line2 (x : ℝ) := (1 / 3) * x
  let line3 (x : ℝ) := 4 * x
  existsi 7
  sorry

end NUMINAMATH_GPT_plane_split_into_regions_l846_84610


namespace NUMINAMATH_GPT_intersection_complement_l846_84639

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3, 5}

theorem intersection_complement : A ∩ (U \ B) = {0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l846_84639


namespace NUMINAMATH_GPT_find_prime_squares_l846_84602

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

theorem find_prime_squares :
  ∀ (p q : ℕ), is_prime p → is_prime q → is_square (p^(q+1) + q^(p+1)) → (p = 2 ∧ q = 2) :=
by 
  intros p q h_prime_p h_prime_q h_square
  sorry

end NUMINAMATH_GPT_find_prime_squares_l846_84602


namespace NUMINAMATH_GPT_maximize_sales_volume_l846_84650

open Real

def profit (x : ℝ) : ℝ := (x - 20) * (400 - 20 * (x - 30))

theorem maximize_sales_volume : 
  ∃ x : ℝ, (∀ x' : ℝ, profit x' ≤ profit x) ∧ x = 35 := 
by
  sorry

end NUMINAMATH_GPT_maximize_sales_volume_l846_84650


namespace NUMINAMATH_GPT_min_n_for_constant_term_l846_84678

theorem min_n_for_constant_term (n : ℕ) (h : 0 < n) : 
  (∃ (r : ℕ), 0 = n - 4 * r / 3) → n = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_n_for_constant_term_l846_84678


namespace NUMINAMATH_GPT_c_work_rate_l846_84695

/--
A can do a piece of work in 4 days.
B can do it in 8 days.
With the assistance of C, A and B completed the work in 2 days.
Prove that C alone can do the work in 8 days.
-/
theorem c_work_rate :
  (1 / 4 + 1 / 8 + 1 / c = 1 / 2) → c = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_c_work_rate_l846_84695


namespace NUMINAMATH_GPT_digital_earth_sustainable_development_l846_84656

theorem digital_earth_sustainable_development :
  (after_realization_digital_earth : Prop) → (scientists_can : Prop) :=
sorry

end NUMINAMATH_GPT_digital_earth_sustainable_development_l846_84656


namespace NUMINAMATH_GPT_no_unique_symbols_for_all_trains_l846_84658

def proposition (a b c d : Prop) : Prop :=
  (¬a ∧  b ∧ ¬c ∧  d)
∨ ( a ∧ ¬b ∧ ¬c ∧ ¬d)

theorem no_unique_symbols_for_all_trains 
    (a b c d : Prop)
    (p : proposition a b c d)
    (s1 : ¬a ∧  b ∧ ¬c ∧  d)
    (s2 :  a ∧ ¬b ∧ ¬c ∧ ¬d) : 
    False :=
by {cases s1; cases s2; contradiction}

end NUMINAMATH_GPT_no_unique_symbols_for_all_trains_l846_84658


namespace NUMINAMATH_GPT_probability_of_specific_combination_l846_84670

theorem probability_of_specific_combination :
  let shirts := 6
  let shorts := 8
  let socks := 7
  let total_clothes := shirts + shorts + socks
  let ways_total := Nat.choose total_clothes 4
  let ways_shirts := Nat.choose shirts 2
  let ways_shorts := Nat.choose shorts 1
  let ways_socks := Nat.choose socks 1
  let ways_favorable := ways_shirts * ways_shorts * ways_socks
  let probability := (ways_favorable: ℚ) / ways_total
  probability = 56 / 399 :=
by
  simp
  sorry

end NUMINAMATH_GPT_probability_of_specific_combination_l846_84670


namespace NUMINAMATH_GPT_woman_first_half_speed_l846_84618

noncomputable def first_half_speed (total_time : ℕ) (second_half_speed : ℕ) (total_distance : ℕ) : ℕ :=
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let second_half_time := second_half_distance / second_half_speed
  let first_half_time := total_time - second_half_time
  first_half_distance / first_half_time

theorem woman_first_half_speed : first_half_speed 20 24 448 = 21 := by
  sorry

end NUMINAMATH_GPT_woman_first_half_speed_l846_84618


namespace NUMINAMATH_GPT_car_speed_without_red_light_l846_84645

theorem car_speed_without_red_light (v : ℝ) :
  (∃ k : ℕ+, v = 10 / k) ↔ 
  ∀ (dist : ℝ) (green_duration red_duration total_cycle : ℝ),
    dist = 1500 ∧ green_duration = 90 ∧ red_duration = 60 ∧ total_cycle = 150 →
    v * total_cycle = dist / (green_duration + red_duration) := 
by
  sorry

end NUMINAMATH_GPT_car_speed_without_red_light_l846_84645


namespace NUMINAMATH_GPT_correctly_calculated_value_l846_84684

theorem correctly_calculated_value : 
  ∃ x : ℝ, (x + 4 = 40) ∧ (x / 4 = 9) :=
sorry

end NUMINAMATH_GPT_correctly_calculated_value_l846_84684


namespace NUMINAMATH_GPT_mean_height_calc_l846_84660

/-- Heights of players on the soccer team -/
def heights : List ℕ := [47, 48, 50, 50, 54, 55, 57, 59, 63, 63, 64, 65]

/-- Total number of players -/
def total_players : ℕ := heights.length

/-- Sum of heights of players -/
def sum_heights : ℕ := heights.sum

/-- Mean height of players on the soccer team -/
def mean_height : ℚ := sum_heights / total_players

/-- Proof that the mean height is correct -/
theorem mean_height_calc : mean_height = 56.25 := by
  sorry

end NUMINAMATH_GPT_mean_height_calc_l846_84660


namespace NUMINAMATH_GPT_range_of_a_l846_84665

variable (a x : ℝ)
def A (a : ℝ) := {x : ℝ | 2 * a ≤ x ∧ x ≤ a ^ 2 + 1}
def B (a : ℝ) := {x : ℝ | (x - 2) * (x - (3 * a + 1)) ≤ 0}

theorem range_of_a (a : ℝ) : (∀ x, x ∈ A a → x ∈ B a) ↔ (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) := by sorry

end NUMINAMATH_GPT_range_of_a_l846_84665


namespace NUMINAMATH_GPT_sally_cost_is_42000_l846_84634

-- Definitions for conditions
def lightningCost : ℕ := 140000
def materCost : ℕ := (10 * lightningCost) / 100
def sallyCost : ℕ := 3 * materCost

-- Theorem statement
theorem sally_cost_is_42000 : sallyCost = 42000 := by
  sorry

end NUMINAMATH_GPT_sally_cost_is_42000_l846_84634


namespace NUMINAMATH_GPT_four_digit_integer_product_l846_84698

theorem four_digit_integer_product :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧ 
  a^2 + b^2 + c^2 + d^2 = 65 ∧ a * b * c * d = 140 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_integer_product_l846_84698


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l846_84647

theorem no_real_roots_of_quadratic :
  ∀ (a b c : ℝ), a = 1 → b = -Real.sqrt 5 → c = Real.sqrt 2 →
  (b^2 - 4 * a * c < 0) → ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  intros a b c ha hb hc hD
  rw [ha, hb, hc] at hD
  sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l846_84647


namespace NUMINAMATH_GPT_cars_minus_trucks_l846_84636

theorem cars_minus_trucks (total : ℕ) (trucks : ℕ) (h_total : total = 69) (h_trucks : trucks = 21) :
  (total - trucks) - trucks = 27 :=
by
  sorry

end NUMINAMATH_GPT_cars_minus_trucks_l846_84636


namespace NUMINAMATH_GPT_simplify_expression_l846_84693

theorem simplify_expression (x : ℝ) :
  4*x^3 + 5*x + 6*x^2 + 10 - (3 - 6*x^2 - 4*x^3 + 2*x) = 8*x^3 + 12*x^2 + 3*x + 7 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l846_84693


namespace NUMINAMATH_GPT_inequality_Cauchy_Schwarz_l846_84682

theorem inequality_Cauchy_Schwarz (a b : ℝ) : 
  (a^4 + b^4) * (a^2 + b^2) ≥ (a^3 + b^3)^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_Cauchy_Schwarz_l846_84682


namespace NUMINAMATH_GPT_construction_company_doors_needed_l846_84652

-- Definitions based on conditions
def num_floors_per_building : ℕ := 20
def num_apartments_per_floor : ℕ := 8
def num_buildings : ℕ := 3
def num_doors_per_apartment : ℕ := 10

-- Total number of apartments
def total_apartments : ℕ :=
  num_floors_per_building * num_apartments_per_floor * num_buildings

-- Total number of doors
def total_doors_needed : ℕ :=
  num_doors_per_apartment * total_apartments

-- Theorem statement to prove the number of doors needed
theorem construction_company_doors_needed :
  total_doors_needed = 4800 :=
sorry

end NUMINAMATH_GPT_construction_company_doors_needed_l846_84652


namespace NUMINAMATH_GPT_painting_methods_correct_l846_84620

noncomputable def num_painting_methods : ℕ :=
  sorry 

theorem painting_methods_correct :
  num_painting_methods = 24 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_painting_methods_correct_l846_84620


namespace NUMINAMATH_GPT_find_n_l846_84609

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 101) (h3 : 100 * n % 101 = 72) : n = 29 := 
by
  sorry

end NUMINAMATH_GPT_find_n_l846_84609


namespace NUMINAMATH_GPT_product_of_g_on_roots_l846_84637

-- Define the given polynomials f and g
def f (x : ℝ) : ℝ := x^5 + 3 * x^2 + 1
def g (x : ℝ) : ℝ := x^2 - 5

-- Define the roots of the polynomial f
axiom roots : ∃ (x1 x2 x3 x4 x5 : ℝ), 
  f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0 ∧ f x5 = 0

theorem product_of_g_on_roots : 
  (∃ x1 x2 x3 x4 x5: ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0 ∧ f x5 = 0) 
  → g x1 * g x2 * g x3 * g x4 * g x5 = 131 := 
by
  sorry

end NUMINAMATH_GPT_product_of_g_on_roots_l846_84637


namespace NUMINAMATH_GPT_first_recipe_cups_l846_84659

-- Definitions based on the given conditions
def ounces_per_bottle : ℕ := 16
def ounces_per_cup : ℕ := 8
def cups_second_recipe : ℕ := 1
def cups_third_recipe : ℕ := 3
def total_bottles : ℕ := 3
def total_ounces : ℕ := total_bottles * ounces_per_bottle
def total_cups_needed : ℕ := total_ounces / ounces_per_cup

-- Proving the amount of cups of soy sauce needed for the first recipe
theorem first_recipe_cups : 
  total_cups_needed - (cups_second_recipe + cups_third_recipe) = 2 
:= by 
-- Proof omitted
  sorry

end NUMINAMATH_GPT_first_recipe_cups_l846_84659


namespace NUMINAMATH_GPT_knights_round_table_l846_84657

theorem knights_round_table (n : ℕ) (h : ∃ (f e : ℕ), f = e ∧ f + e = n) : n % 4 = 0 :=
sorry

end NUMINAMATH_GPT_knights_round_table_l846_84657


namespace NUMINAMATH_GPT_least_possible_number_l846_84627

theorem least_possible_number {x : ℕ} (h1 : x % 6 = 2) (h2 : x % 4 = 3) : x = 50 :=
sorry

end NUMINAMATH_GPT_least_possible_number_l846_84627


namespace NUMINAMATH_GPT_exists_three_digit_number_l846_84611

theorem exists_three_digit_number : ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ (100 * a + 10 * b + c = a^3 + b^3 + c^3) ∧ (100 * a + 10 * b + c ≥ 100 ∧ 100 * a + 10 * b + c < 1000) := 
sorry

end NUMINAMATH_GPT_exists_three_digit_number_l846_84611


namespace NUMINAMATH_GPT_NicoleEndsUpWith36Pieces_l846_84633

namespace ClothingProblem

noncomputable def NicoleClothesStart := 10
noncomputable def FirstOlderSisterClothes := NicoleClothesStart / 2
noncomputable def NextOldestSisterClothes := NicoleClothesStart + 2
noncomputable def OldestSisterClothes := (NicoleClothesStart + FirstOlderSisterClothes + NextOldestSisterClothes) / 3

theorem NicoleEndsUpWith36Pieces : 
  NicoleClothesStart + FirstOlderSisterClothes + NextOldestSisterClothes + OldestSisterClothes = 36 :=
  by
    sorry

end ClothingProblem

end NUMINAMATH_GPT_NicoleEndsUpWith36Pieces_l846_84633


namespace NUMINAMATH_GPT_geom_seq_a7_a10_sum_l846_84626

theorem geom_seq_a7_a10_sum (a_n : ℕ → ℝ) (q a1 : ℝ)
  (h_seq : ∀ n, a_n (n + 1) = a1 * (q ^ n))
  (h1 : a1 + a1 * q = 2)
  (h2 : a1 * (q ^ 2) + a1 * (q ^ 3) = 4) :
  a_n 7 + a_n 8 + a_n 9 + a_n 10 = 48 := 
sorry

end NUMINAMATH_GPT_geom_seq_a7_a10_sum_l846_84626


namespace NUMINAMATH_GPT_difference_between_sums_l846_84680

open Nat

-- Sum of the first 'n' positive odd integers formula: n^2
def sum_of_first_odd (n : ℕ) : ℕ := n * n

-- Sum of the first 'n' positive even integers formula: n(n+1)
def sum_of_first_even (n : ℕ) : ℕ := n * (n + 1)

-- The main theorem stating the difference between the sums
theorem difference_between_sums (n : ℕ) (h : n = 3005) :
  sum_of_first_even n - sum_of_first_odd n = 3005 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_sums_l846_84680


namespace NUMINAMATH_GPT_distance_to_grandmas_house_is_78_l846_84690

-- Define the conditions
def miles_to_pie_shop : ℕ := 35
def miles_to_gas_station : ℕ := 18
def miles_remaining : ℕ := 25

-- Define the mathematical claim
def total_distance_to_grandmas_house : ℕ :=
  miles_to_pie_shop + miles_to_gas_station + miles_remaining

-- Prove the claim
theorem distance_to_grandmas_house_is_78 :
  total_distance_to_grandmas_house = 78 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_grandmas_house_is_78_l846_84690


namespace NUMINAMATH_GPT_cost_of_bag_l846_84651

variable (cost_per_bag : ℝ)
variable (chips_per_bag : ℕ := 24)
variable (calories_per_chip : ℕ := 10)
variable (total_calories : ℕ := 480)
variable (total_cost : ℝ := 4)

theorem cost_of_bag :
  (chips_per_bag * (total_calories / calories_per_chip / chips_per_bag) = (total_calories / calories_per_chip)) →
  (total_cost / (total_calories / (calories_per_chip * chips_per_bag))) = 2 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_bag_l846_84651


namespace NUMINAMATH_GPT_must_true_l846_84683

axiom p : Prop
axiom q : Prop
axiom h1 : ¬ (p ∧ q)
axiom h2 : p ∨ q

theorem must_true : (¬ p) ∨ (¬ q) := by
  sorry

end NUMINAMATH_GPT_must_true_l846_84683


namespace NUMINAMATH_GPT_solve_abs_quadratic_eq_l846_84699

theorem solve_abs_quadratic_eq (x : ℝ) (h : |2 * x + 4| = 1 - 3 * x + x ^ 2) :
    x = (5 + Real.sqrt 37) / 2 ∨ x = (5 - Real.sqrt 37) / 2 := by
  sorry

end NUMINAMATH_GPT_solve_abs_quadratic_eq_l846_84699


namespace NUMINAMATH_GPT_employees_bonus_l846_84653

theorem employees_bonus (x y z : ℝ) 
  (h1 : x + y + z = 2970) 
  (h2 : y = (1 / 3) * x + 180) 
  (h3 : z = (1 / 3) * y + 130) :
  x = 1800 ∧ y = 780 ∧ z = 390 :=
by
  sorry

end NUMINAMATH_GPT_employees_bonus_l846_84653


namespace NUMINAMATH_GPT_initial_guppies_l846_84628

theorem initial_guppies (total_gups : ℕ) (dozen_gups : ℕ) (extra_gups : ℕ) (baby_gups_initial : ℕ) (baby_gups_later : ℕ) :
  total_gups = 52 → dozen_gups = 12 → extra_gups = 3 → baby_gups_initial = 3 * 12 → baby_gups_later = 9 → 
  total_gups - (baby_gups_initial + baby_gups_later) = 7 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_initial_guppies_l846_84628


namespace NUMINAMATH_GPT_constant_term_correct_l846_84673

theorem constant_term_correct:
    ∀ (a k n : ℤ), 
      (∀ x : ℤ, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) 
      → a - n + k = 7 
      → n = -6 := 
by
    intros a k n h h2
    have h1 := h 0
    sorry

end NUMINAMATH_GPT_constant_term_correct_l846_84673


namespace NUMINAMATH_GPT_mike_spend_on_plants_l846_84654

def Mike_buys : Prop :=
  let rose_bushes_total := 6
  let rose_bush_cost := 75
  let friend_rose_bushes := 2
  let self_rose_bushes := rose_bushes_total - friend_rose_bushes
  let self_rose_bush_cost := self_rose_bushes * rose_bush_cost
  let tiger_tooth_aloe_total := 2
  let aloe_cost := 100
  let self_aloe_cost := tiger_tooth_aloe_total * aloe_cost
  self_rose_bush_cost + self_aloe_cost = 500

theorem mike_spend_on_plants :
  Mike_buys := by
  sorry

end NUMINAMATH_GPT_mike_spend_on_plants_l846_84654


namespace NUMINAMATH_GPT_decreasing_interval_of_even_function_l846_84644

-- Defining the function f(x)
def f (x : ℝ) (k : ℝ) : ℝ := (k-2) * x^2 + (k-1) * x + 3

-- Defining the condition that f is an even function
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem decreasing_interval_of_even_function (k : ℝ) :
  isEvenFunction (f · k) → k = 1 ∧ ∀ x ≥ 0, f x k ≤ f 0 k :=
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_of_even_function_l846_84644


namespace NUMINAMATH_GPT_conversion_base8_to_base10_l846_84687

theorem conversion_base8_to_base10 : 5 * 8^3 + 2 * 8^2 + 1 * 8^1 + 4 * 8^0 = 2700 :=
by 
  sorry

end NUMINAMATH_GPT_conversion_base8_to_base10_l846_84687


namespace NUMINAMATH_GPT_exists_unique_xy_l846_84685

theorem exists_unique_xy (n : ℕ) : ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
sorry

end NUMINAMATH_GPT_exists_unique_xy_l846_84685


namespace NUMINAMATH_GPT_solve_quadratic_eqn_l846_84677

theorem solve_quadratic_eqn :
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ (x = 2 ∨ x = -3) :=
by
  intros
  simp
  sorry

end NUMINAMATH_GPT_solve_quadratic_eqn_l846_84677


namespace NUMINAMATH_GPT_not_basic_logical_structure_l846_84603

def basic_structures : Set String := {"Sequential structure", "Conditional structure", "Loop structure"}

theorem not_basic_logical_structure : "Operational structure" ∉ basic_structures := by
  sorry

end NUMINAMATH_GPT_not_basic_logical_structure_l846_84603


namespace NUMINAMATH_GPT_fraction_conversion_integer_l846_84604

theorem fraction_conversion_integer (x : ℝ) :
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1 →
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1 :=
by sorry

end NUMINAMATH_GPT_fraction_conversion_integer_l846_84604


namespace NUMINAMATH_GPT_postal_service_revenue_l846_84691

theorem postal_service_revenue 
  (price_colored : ℝ := 0.50)
  (price_bw : ℝ := 0.35)
  (price_golden : ℝ := 2.00)
  (sold_colored : ℕ := 578833)
  (sold_bw : ℕ := 523776)
  (sold_golden : ℕ := 120456) : 
  (price_colored * (sold_colored : ℝ) + 
  price_bw * (sold_bw : ℝ) + 
  price_golden * (sold_golden : ℝ) = 713650.10) :=
by
  sorry

end NUMINAMATH_GPT_postal_service_revenue_l846_84691


namespace NUMINAMATH_GPT_A_completion_time_l846_84642

theorem A_completion_time :
  ∃ A : ℝ, (A > 0) ∧ (
    (2 * (1 / A + 1 / 10) + 3.0000000000000004 * (1 / 10) = 1) ↔ A = 4
  ) :=
by
  have B_workday := 10
  sorry -- proof would go here

end NUMINAMATH_GPT_A_completion_time_l846_84642


namespace NUMINAMATH_GPT_tyrone_gave_marbles_l846_84692

theorem tyrone_gave_marbles :
  ∃ x : ℝ, (120 - x = 3 * (30 + x)) ∧ x = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_tyrone_gave_marbles_l846_84692


namespace NUMINAMATH_GPT_given_condition_required_solution_l846_84615

-- Define the polynomial f.
noncomputable def f (x : ℝ) : ℝ := x^2 + x - 6

-- Given condition
theorem given_condition (x : ℝ) : f (x^2 + 2) = x^4 + 5 * x^2 := by sorry

-- Proving the required equivalence
theorem required_solution (x : ℝ) : f (x^2 - 2) = x^4 - 3 * x^2 - 4 := by sorry

end NUMINAMATH_GPT_given_condition_required_solution_l846_84615


namespace NUMINAMATH_GPT_person_speed_l846_84623

theorem person_speed (distance_m : ℝ) (time_min : ℝ) (h₁ : distance_m = 800) (h₂ : time_min = 5) : 
  let distance_km := distance_m / 1000
  let time_hr := time_min / 60
  distance_km / time_hr = 9.6 := 
by
  sorry

end NUMINAMATH_GPT_person_speed_l846_84623


namespace NUMINAMATH_GPT_area_of_triangle_l846_84606

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 8 = 1

def foci_distance (F1 F2 : ℝ × ℝ) : Prop := (F1.1, F1.2) = (-3, 0) ∧ (F2.1, F2.2) = (3, 0)

def point_on_hyperbola (x y : ℝ) : Prop := hyperbola x y

def distance_ratios (P F1 F2 : ℝ × ℝ) : Prop := 
  let PF1 := (P.1 - F1.1)^2 + (P.2 - F1.2)^2
  let PF2 := (P.1 - F2.1)^2 + (P.2 - F2.2)^2
  PF1 / PF2 = 3 / 4

theorem area_of_triangle {P F1 F2 : ℝ × ℝ} 
  (H1 : foci_distance F1 F2)
  (H2 : point_on_hyperbola P.1 P.2)
  (H3 : distance_ratios P F1 F2) :
  let area := 1 / 2 * (6:ℝ) * (8:ℝ) * Real.sqrt 5
  area = 8 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_area_of_triangle_l846_84606


namespace NUMINAMATH_GPT_tan_arccos_eq_2y_l846_84613

noncomputable def y_squared : ℝ :=
  (-1 + Real.sqrt 17) / 8

theorem tan_arccos_eq_2y (y : ℝ) (hy : 0 < y) (htan : Real.tan (Real.arccos y) = 2 * y) :
  y^2 = y_squared := sorry

end NUMINAMATH_GPT_tan_arccos_eq_2y_l846_84613


namespace NUMINAMATH_GPT_rectangle_area_l846_84622

theorem rectangle_area (sqr_area : ℕ) (rect_width rect_length : ℕ) (h1 : sqr_area = 25)
    (h2 : rect_width = Int.sqrt sqr_area) (h3 : rect_length = 2 * rect_width) :
    rect_width * rect_length = 50 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l846_84622


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_has_remainder_2_l846_84635

def arithmetic_sequence_remainder : ℕ := 
  let first_term := 1
  let common_difference := 6
  let last_term := 259
  -- Calculate number of terms
  let n := (last_term + 5) / common_difference
  -- Sum of remainders of each term when divided by 6
  let sum_of_remainders := n * 1
  -- The remainder when this sum is divided by 6
  sum_of_remainders % 6 
theorem sum_of_arithmetic_sequence_has_remainder_2 : 
  arithmetic_sequence_remainder = 2 := by 
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_has_remainder_2_l846_84635


namespace NUMINAMATH_GPT_pagoda_top_story_lanterns_l846_84662

/--
Given a 7-story pagoda where each story has twice as many lanterns as the one above it, 
and a total of 381 lanterns across all stories, prove the number of lanterns on the top (7th) story is 3.
-/
theorem pagoda_top_story_lanterns (a : ℕ) (n : ℕ) (r : ℚ) (sum_lanterns : ℕ) :
  n = 7 → r = 1 / 2 → sum_lanterns = 381 →
  (a * (1 - r^n) / (1 - r) = sum_lanterns) → (a * r^(n - 1) = 3) :=
by
  intros h_n h_r h_sum h_geo_sum
  let a_val := 192 -- from the solution steps
  rw [h_n, h_r, h_sum] at h_geo_sum
  have h_a : a = a_val := by sorry
  rw [h_a, h_n, h_r]
  exact sorry

end NUMINAMATH_GPT_pagoda_top_story_lanterns_l846_84662


namespace NUMINAMATH_GPT_least_clock_equivalent_to_square_greater_than_4_l846_84671

theorem least_clock_equivalent_to_square_greater_than_4 : 
  ∃ (x : ℕ), x > 4 ∧ (x^2 - x) % 12 = 0 ∧ ∀ (y : ℕ), y > 4 → (y^2 - y) % 12 = 0 → x ≤ y :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_least_clock_equivalent_to_square_greater_than_4_l846_84671


namespace NUMINAMATH_GPT_range_of_a_l846_84672

variable {x a : ℝ}

def p (a : ℝ) (x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

theorem range_of_a (ha : a < 0) 
  (H : (∀ x, ¬ p a x → q x) ∧ ∃ x, q x ∧ ¬ p a x ∧ ¬ q x) : a ≤ -4 := 
sorry

end NUMINAMATH_GPT_range_of_a_l846_84672


namespace NUMINAMATH_GPT_train_length_l846_84608

theorem train_length
  (speed_km_hr : ℕ)
  (time_sec : ℕ)
  (length_train : ℕ)
  (length_platform : ℕ)
  (h_eq_len : length_train = length_platform)
  (h_speed : speed_km_hr = 108)
  (h_time : time_sec = 60) :
  length_train = 900 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l846_84608


namespace NUMINAMATH_GPT_jared_annual_earnings_l846_84696

open Nat

noncomputable def diploma_monthly_pay : ℕ := 4000
noncomputable def months_in_year : ℕ := 12
noncomputable def multiplier : ℕ := 3

theorem jared_annual_earnings :
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  jared_annual_earnings = 144000 :=
by
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  exact sorry

end NUMINAMATH_GPT_jared_annual_earnings_l846_84696


namespace NUMINAMATH_GPT_fraction_addition_l846_84643

variable (a : ℝ)

theorem fraction_addition (ha : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by
  sorry

end NUMINAMATH_GPT_fraction_addition_l846_84643


namespace NUMINAMATH_GPT_sandwiches_per_person_l846_84675

-- Definitions derived from conditions
def cost_of_12_croissants := 8.0
def number_of_people := 24
def total_spending := 32.0
def croissants_per_set := 12

-- Statement to be proved
theorem sandwiches_per_person :
  ∀ (cost_of_12_croissants total_spending croissants_per_set number_of_people : ℕ),
  total_spending / cost_of_12_croissants * croissants_per_set / number_of_people = 2 :=
by
  sorry

end NUMINAMATH_GPT_sandwiches_per_person_l846_84675


namespace NUMINAMATH_GPT_g_of_neg_2_l846_84688

def f (x : ℚ) : ℚ := 4 * x - 9

def g (y : ℚ) : ℚ :=
  3 * ((y + 9) / 4)^2 - 4 * ((y + 9) / 4) + 2

theorem g_of_neg_2 : g (-2) = 67 / 16 :=
by
  sorry

end NUMINAMATH_GPT_g_of_neg_2_l846_84688


namespace NUMINAMATH_GPT_binomial_expansion_product_l846_84607

theorem binomial_expansion_product (a a1 a2 a3 a4 a5 : ℤ)
  (h1 : (1 - 1)^5 = a + a1 + a2 + a3 + a4 + a5)
  (h2 : (1 - (-1))^5 = a - a1 + a2 - a3 + a4 - a5) :
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := by
  sorry

end NUMINAMATH_GPT_binomial_expansion_product_l846_84607


namespace NUMINAMATH_GPT_find_integer_n_l846_84630

theorem find_integer_n : ∃ n, 5 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [MOD 6] ∧ n = 9 :=   
by 
  -- The proof will be written here.
  sorry

end NUMINAMATH_GPT_find_integer_n_l846_84630


namespace NUMINAMATH_GPT_least_tiles_required_l846_84668

def room_length : ℕ := 7550
def room_breadth : ℕ := 2085
def tile_size : ℕ := 5
def total_area : ℕ := room_length * room_breadth
def tile_area : ℕ := tile_size * tile_size
def number_of_tiles : ℕ := total_area / tile_area

theorem least_tiles_required : number_of_tiles = 630270 := by
  sorry

end NUMINAMATH_GPT_least_tiles_required_l846_84668


namespace NUMINAMATH_GPT_cost_of_one_package_of_berries_l846_84655

noncomputable def martin_daily_consumption : ℚ := 1 / 2

noncomputable def package_content : ℚ := 1

noncomputable def total_period_days : ℚ := 30

noncomputable def total_spent : ℚ := 30

theorem cost_of_one_package_of_berries :
  (total_spent / (total_period_days * martin_daily_consumption / package_content)) = 2 :=
sorry

end NUMINAMATH_GPT_cost_of_one_package_of_berries_l846_84655


namespace NUMINAMATH_GPT_sum_zero_implies_inequality_l846_84663

variable {a b c d : ℝ}

theorem sum_zero_implies_inequality
  (h : a + b + c + d = 0) :
  5 * (a * b + b * c + c * d) + 8 * (a * c + a * d + b * d) ≤ 0 := 
sorry

end NUMINAMATH_GPT_sum_zero_implies_inequality_l846_84663


namespace NUMINAMATH_GPT_another_seat_in_sample_l846_84676

-- Definition of the problem
def total_students := 56
def sample_size := 4
def sample_set : Finset ℕ := {3, 17, 45}

-- Lean 4 statement for the proof problem
theorem another_seat_in_sample :
  (sample_set = sample_set ∪ {31}) ∧
  (31 ∉ sample_set) ∧
  (∀ x ∈ sample_set ∪ {31}, x ≤ total_students) :=
by
  sorry

end NUMINAMATH_GPT_another_seat_in_sample_l846_84676


namespace NUMINAMATH_GPT_calculate_division_l846_84605

theorem calculate_division :
  (- (3 / 4) - 5 / 9 + 7 / 12) / (- 1 / 36) = 26 := by
  sorry

end NUMINAMATH_GPT_calculate_division_l846_84605


namespace NUMINAMATH_GPT_intersection_in_second_quadrant_l846_84619

theorem intersection_in_second_quadrant (k : ℝ) (x y : ℝ) 
  (hk : 0 < k) (hk2 : k < 1/2) 
  (h1 : k * x - y = k - 1) 
  (h2 : k * y - x = 2 * k) : 
  x < 0 ∧ y > 0 := 
sorry

end NUMINAMATH_GPT_intersection_in_second_quadrant_l846_84619


namespace NUMINAMATH_GPT_number_of_ideal_subsets_l846_84694

def is_ideal_subset (p q : ℕ) (S : Set ℕ) : Prop :=
  0 ∈ S ∧ ∀ n ∈ S, n + p ∈ S ∧ n + q ∈ S

theorem number_of_ideal_subsets (p q : ℕ) (hpq : Nat.Coprime p q) :
  ∃ n, n = Nat.choose (p + q) p / (p + q) :=
sorry

end NUMINAMATH_GPT_number_of_ideal_subsets_l846_84694


namespace NUMINAMATH_GPT_sector_area_l846_84689

theorem sector_area (r α S : ℝ) (h1 : α = 2) (h2 : 2 * r + α * r = 8) : S = 4 :=
sorry

end NUMINAMATH_GPT_sector_area_l846_84689


namespace NUMINAMATH_GPT_number_of_divisors_of_2744_l846_84617

-- Definition of the integer and its prime factorization
def two := 2
def seven := 7
def n := two^3 * seven^3

-- Define the property for the number of divisors
def num_divisors (n : ℕ) : ℕ := (3 + 1) * (3 + 1)

-- Main proof statement
theorem number_of_divisors_of_2744 : num_divisors n = 16 := by
  sorry

end NUMINAMATH_GPT_number_of_divisors_of_2744_l846_84617


namespace NUMINAMATH_GPT_more_flour_than_sugar_l846_84601

def cups_of_flour : Nat := 9
def cups_of_sugar : Nat := 6
def flour_added : Nat := 2
def flour_needed : Nat := cups_of_flour - flour_added -- 9 - 2 = 7

theorem more_flour_than_sugar : flour_needed - cups_of_sugar = 1 :=
by
  sorry

end NUMINAMATH_GPT_more_flour_than_sugar_l846_84601


namespace NUMINAMATH_GPT_arithmetic_expression_l846_84629

theorem arithmetic_expression : 125 - 25 * 4 = 25 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l846_84629


namespace NUMINAMATH_GPT_problem_I_II_l846_84661

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

def seq_a (a : ℕ → ℝ) (a1 : ℝ) : Prop :=
  a 0 = a1 ∧ (∀ n, a (n + 1) = f (a n))

theorem problem_I_II (a : ℕ → ℝ) (a1 : ℝ) (h_a1 : 0 < a1 ∧ a1 < 1) (h_seq : seq_a a a1) :
  (∀ n, 0 < a (n + 1) ∧ a (n + 1) < a n ∧ a n < 1) ∧
  (∀ n, a (n + 1) < (1 / 6) * (a n) ^ 3) :=
  sorry

end NUMINAMATH_GPT_problem_I_II_l846_84661


namespace NUMINAMATH_GPT_flashlight_distance_difference_l846_84625

/--
Veronica's flashlight can be seen from 1000 feet. Freddie's flashlight can be seen from a distance
three times that of Veronica's flashlight. Velma's flashlight can be seen from a distance 2000 feet
less than 5 times Freddie's flashlight distance. We want to prove that Velma's flashlight can be seen 
12000 feet farther than Veronica's flashlight.
-/
theorem flashlight_distance_difference :
  let v_d := 1000
  let f_d := 3 * v_d
  let V_d := 5 * f_d - 2000
  V_d - v_d = 12000 := by
    sorry

end NUMINAMATH_GPT_flashlight_distance_difference_l846_84625


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l846_84649

def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem necessary_but_not_sufficient : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) :=
by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l846_84649


namespace NUMINAMATH_GPT_total_sacks_after_6_days_l846_84669

-- Define the conditions
def sacks_per_day : ℕ := 83
def days : ℕ := 6

-- Prove the total number of sacks after 6 days is 498
theorem total_sacks_after_6_days : sacks_per_day * days = 498 := by
  -- Proof Content Placeholder
  sorry

end NUMINAMATH_GPT_total_sacks_after_6_days_l846_84669


namespace NUMINAMATH_GPT_exist_odd_a_b_k_l846_84667

theorem exist_odd_a_b_k (m : ℤ) : 
  ∃ (a b k : ℤ), (a % 2 = 1) ∧ (b % 2 = 1) ∧ (k ≥ 0) ∧ (2 * m = a^19 + b^99 + k * 2^1999) :=
by {
  sorry
}

end NUMINAMATH_GPT_exist_odd_a_b_k_l846_84667


namespace NUMINAMATH_GPT_vector_magnitude_proof_l846_84614

theorem vector_magnitude_proof (a b : ℝ × ℝ) 
  (h₁ : ‖a‖ = 1) 
  (h₂ : ‖b‖ = 2)
  (h₃ : a - b = (Real.sqrt 3, Real.sqrt 2)) : 
‖a + (2:ℝ) • b‖ = Real.sqrt 17 := 
sorry

end NUMINAMATH_GPT_vector_magnitude_proof_l846_84614


namespace NUMINAMATH_GPT_functional_equation_solution_l846_84679

-- The mathematical problem statement in Lean 4

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_monotonic : ∀ x y : ℝ, (f x) * (f y) = f (x + y))
  (h_mono : ∀ x y : ℝ, x < y → f x < f y ∨ f x > f y) :
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, f x = a^x :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l846_84679


namespace NUMINAMATH_GPT_triangle_area_of_tangent_line_l846_84648

theorem triangle_area_of_tangent_line (a : ℝ) 
  (h : a > 0) 
  (ha : (1/2) * 3 * a * (3 / (2 * a ^ (1/2))) = 18)
  : a = 64 := 
sorry

end NUMINAMATH_GPT_triangle_area_of_tangent_line_l846_84648


namespace NUMINAMATH_GPT_sequence_property_l846_84641

theorem sequence_property : 
  ∀ (a : ℕ → ℝ), 
    a 1 = 1 →
    a 2 = 1 → 
    (∀ n, a (n + 2) = a (n + 1) + 1 / a n) →
    a 180 > 19 :=
by
  intros a h1 h2 h3
  sorry

end NUMINAMATH_GPT_sequence_property_l846_84641


namespace NUMINAMATH_GPT_john_weekly_loss_is_525000_l846_84664

-- Define the constants given in the problem
def daily_production : ℕ := 1000
def production_cost_per_tire : ℝ := 250
def selling_price_factor : ℝ := 1.5
def potential_daily_sales : ℕ := 1200
def days_in_week : ℕ := 7

-- Define the selling price per tire
def selling_price_per_tire : ℝ := production_cost_per_tire * selling_price_factor

-- Define John's current daily earnings from selling 1000 tires
def current_daily_earnings : ℝ := daily_production * selling_price_per_tire

-- Define John's potential daily earnings from selling 1200 tires
def potential_daily_earnings : ℝ := potential_daily_sales * selling_price_per_tire

-- Define the daily loss by not being able to produce all the tires
def daily_loss : ℝ := potential_daily_earnings - current_daily_earnings

-- Define the weekly loss
def weekly_loss : ℝ := daily_loss * days_in_week

-- Statement: Prove that John's weekly financial loss is $525,000
theorem john_weekly_loss_is_525000 : weekly_loss = 525000 :=
by
  sorry

end NUMINAMATH_GPT_john_weekly_loss_is_525000_l846_84664


namespace NUMINAMATH_GPT_chord_square_length_eq_512_l846_84697

open Real

/-
The conditions are:
1. The radii of two smaller circles are 4 and 8.
2. These circles are externally tangent to each other.
3. Both smaller circles are internally tangent to a larger circle with radius 12.
4. A common external tangent to the two smaller circles serves as a chord of the larger circle.
-/

noncomputable def radius_small1 : ℝ := 4
noncomputable def radius_small2 : ℝ := 8
noncomputable def radius_large : ℝ := 12

/-- Show that the square of the length of the chord formed by the common external tangent of two smaller circles 
which are externally tangent to each other and internally tangent to a larger circle is 512. -/
theorem chord_square_length_eq_512 : ∃ (PQ : ℝ), PQ^2 = 512 := by
  sorry

end NUMINAMATH_GPT_chord_square_length_eq_512_l846_84697


namespace NUMINAMATH_GPT_find_y_l846_84621

open Real

structure Vec3 where
  x : ℝ
  y : ℝ
  z : ℝ

def parallel (v₁ v₂ : Vec3) : Prop := ∃ s : ℝ, v₁ = ⟨s * v₂.x, s * v₂.y, s * v₂.z⟩

def orthogonal (v₁ v₂ : Vec3) : Prop := (v₁.x * v₂.x + v₁.y * v₂.y + v₁.z * v₂.z) = 0

noncomputable def correct_y (x y : Vec3) : Vec3 :=
  ⟨(8 : ℝ) - 2 * (2 : ℝ), (-4 : ℝ) - 2 * (2 : ℝ), (2 : ℝ) - 2 * (2 : ℝ)⟩

theorem find_y :
  ∀ (x y : Vec3),
    (x.x + y.x = 8) ∧ (x.y + y.y = -4) ∧ (x.z + y.z = 2) →
    (parallel x ⟨2, 2, 2⟩) →
    (orthogonal y ⟨1, -1, 0⟩) →
    y = ⟨4, -8, -2⟩ :=
by
  intros x y Hxy Hparallel Horthogonal
  sorry

end NUMINAMATH_GPT_find_y_l846_84621


namespace NUMINAMATH_GPT_third_highest_score_l846_84600

theorem third_highest_score
  (mean15 : ℕ → ℚ) (mean12 : ℕ → ℚ) 
  (sum15 : ℕ) (sum12 : ℕ) (highest : ℕ) (third_highest : ℕ) (third_is_100: third_highest = 100) :
  (mean15 15 = 90) →
  (mean12 12 = 85) →
  (highest = 120) →
  (sum15 = 15 * 90) →
  (sum12 = 12 * 85) →
  (sum15 - sum12 = highest + 210) →
  third_highest = 100 := 
by
  intros hm15 hm12 hhigh hsum15 hsum12 hdiff
  sorry

end NUMINAMATH_GPT_third_highest_score_l846_84600


namespace NUMINAMATH_GPT_points_on_same_line_l846_84612

theorem points_on_same_line (p : ℝ) :
  (∃ m : ℝ, m = ( -3.5 - 0.5 ) / ( 3 - (-1)) ∧ ∀ x y : ℝ, 
    (x = -1 ∧ y = 0.5) ∨ (x = 3 ∧ y = -3.5) ∨ (x = 7 ∧ y = p) → y = m * x + (0.5 - m * (-1))) →
    p = -7.5 :=
by
  sorry

end NUMINAMATH_GPT_points_on_same_line_l846_84612
