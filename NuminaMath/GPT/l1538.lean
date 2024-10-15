import Mathlib

namespace NUMINAMATH_GPT_no_integer_solutions_l1538_153898

theorem no_integer_solutions (x y : ℤ) : 15 * x^2 - 7 * y^2 ≠ 9 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1538_153898


namespace NUMINAMATH_GPT_find_n_l1538_153848

theorem find_n (x n : ℝ) (h₁ : x = 1) (h₂ : 5 / (n + 1 / x) = 1) : n = 4 :=
sorry

end NUMINAMATH_GPT_find_n_l1538_153848


namespace NUMINAMATH_GPT_question_1_question_2_l1538_153892

open Real

noncomputable def f (x a : ℝ) := abs (x - a) + 3 * x

theorem question_1 :
  {x : ℝ | f x 1 > 3 * x + 2} = {x : ℝ | x > 3 ∨ x < -1} :=
by 
  sorry
  
theorem question_2 (h : {x : ℝ | f x a ≤ 0} = {x : ℝ | x ≤ -1}) :
  a = 2 :=
by 
  sorry

end NUMINAMATH_GPT_question_1_question_2_l1538_153892


namespace NUMINAMATH_GPT_find_value_perpendicular_distances_l1538_153846

variable {R a b c D E F : ℝ}
variable {ABC : Triangle}

-- Assume the distances from point P on the circumcircle of triangle ABC
-- to the sides BC, CA, and AB respectively.
axiom D_def : D = R * a / (2 * R)
axiom E_def : E = R * b / (2 * R)
axiom F_def : F = R * c / (2 * R)

theorem find_value_perpendicular_distances
    (a b c R : ℝ) (D E F : ℝ) 
    (hD : D = R * a / (2 * R)) 
    (hE : E = R * b / (2 * R)) 
    (hF : F = R * c / (2 * R)) : 
    a^2 * D^2 + b^2 * E^2 + c^2 * F^2 = (a^4 + b^4 + c^4) / (4 * R^2) :=
by
  sorry

end NUMINAMATH_GPT_find_value_perpendicular_distances_l1538_153846


namespace NUMINAMATH_GPT_angle_sum_property_l1538_153835

theorem angle_sum_property 
  (P Q R S : Type) 
  (alpha beta : ℝ)
  (h1 : alpha = 3 * x)
  (h2 : beta = 2 * x)
  (h3 : alpha + beta = 90) :
  x = 18 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_property_l1538_153835


namespace NUMINAMATH_GPT_arithmetic_sequence_30th_term_value_l1538_153806

def arithmetic_sequence (a_1 d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

-- Given conditions
def a1 : ℤ := 3
def a2 : ℤ := 15
def a3 : ℤ := 27

-- Calculate the common difference d
def d : ℤ := a2 - a1

-- Define the 30th term
def a30 := arithmetic_sequence a1 d 30

theorem arithmetic_sequence_30th_term_value :
  a30 = 351 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_30th_term_value_l1538_153806


namespace NUMINAMATH_GPT_product_of_ratios_l1538_153827

theorem product_of_ratios 
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (hx1 : x1^3 - 3 * x1 * y1^2 = 2005)
  (hy1 : y1^3 - 3 * x1^2 * y1 = 2004)
  (hx2 : x2^3 - 3 * x2 * y2^2 = 2005)
  (hy2 : y2^3 - 3 * x2^2 * y2 = 2004)
  (hx3 : x3^3 - 3 * x3 * y3^2 = 2005)
  (hy3 : y3^3 - 3 * x3^2 * y3 = 2004) :
  (1 - x1/y1) * (1 - x2/y2) * (1 - x3/y3) = 1/1002 := 
sorry

end NUMINAMATH_GPT_product_of_ratios_l1538_153827


namespace NUMINAMATH_GPT_prove_zero_function_l1538_153813

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq : ∀ x y : ℝ, f (x ^ 333 + y) = f (x ^ 2018 + 2 * y) + f (x ^ 42)

theorem prove_zero_function : ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_prove_zero_function_l1538_153813


namespace NUMINAMATH_GPT_garden_area_difference_l1538_153877

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end NUMINAMATH_GPT_garden_area_difference_l1538_153877


namespace NUMINAMATH_GPT_solve_for_x_l1538_153834

theorem solve_for_x (x : ℝ) (h : (x * (x ^ (5 / 2))) ^ (1 / 4) = 4) : 
  x = 4 ^ (8 / 7) :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1538_153834


namespace NUMINAMATH_GPT_construct_right_triangle_l1538_153896

noncomputable def quadrilateral (A B C D : Type) : Prop :=
∃ (AB BC CA : ℝ), 
AB = BC ∧ BC = CA ∧ 
∃ (angle_D : ℝ), 
angle_D = 30

theorem construct_right_triangle (A B C D : Type) (angle_D: ℝ) (AB BC CA : ℝ) 
    (h1 : AB = BC) (h2 : BC = CA) (h3 : angle_D = 30) : 
    exists DA DB DC : ℝ, (DA * DA) + (DC * DC) = (AD * AD) :=
by sorry

end NUMINAMATH_GPT_construct_right_triangle_l1538_153896


namespace NUMINAMATH_GPT_has_zero_in_intervals_l1538_153883

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x
noncomputable def f' (x : ℝ) : ℝ := (1 / 3) - (1 / x)

theorem has_zero_in_intervals : 
  (∃ x : ℝ, 0 < x ∧ x < 3 ∧ f x = 0) ∧ (∃ x : ℝ, 3 < x ∧ f x = 0) :=
sorry

end NUMINAMATH_GPT_has_zero_in_intervals_l1538_153883


namespace NUMINAMATH_GPT_find_u_l1538_153895

variable (α β γ : ℝ)
variables (q s u : ℝ)

-- The first polynomial has roots α, β, γ
axiom roots_first_poly : ∀ x : ℝ, x^3 + 4 * x^2 + 6 * x - 8 = (x - α) * (x - β) * (x - γ)

-- Sum of the roots α + β + γ = -4
axiom sum_roots_first_poly : α + β + γ = -4

-- Product of the roots αβγ = 8
axiom product_roots_first_poly : α * β * γ = 8

-- The second polynomial has roots α + β, β + γ, γ + α
axiom roots_second_poly : ∀ x : ℝ, x^3 + q * x^2 + s * x + u = (x - (α + β)) * (x - (β + γ)) * (x - (γ + α))

theorem find_u : u = 32 :=
sorry

end NUMINAMATH_GPT_find_u_l1538_153895


namespace NUMINAMATH_GPT_piravena_trip_total_cost_l1538_153819

-- Define the distances
def d_A_to_B : ℕ := 4000
def d_B_to_C : ℕ := 3000

-- Define the costs per kilometer
def bus_cost_per_km : ℝ := 0.15
def airplane_cost_per_km : ℝ := 0.12
def airplane_booking_fee : ℝ := 120

-- Define the individual costs and the total cost
def cost_A_to_B : ℝ := d_A_to_B * airplane_cost_per_km + airplane_booking_fee
def cost_B_to_C : ℝ := d_B_to_C * bus_cost_per_km
def total_cost : ℝ := cost_A_to_B + cost_B_to_C

-- Define the theorem we want to prove
theorem piravena_trip_total_cost :
  total_cost = 1050 := sorry

end NUMINAMATH_GPT_piravena_trip_total_cost_l1538_153819


namespace NUMINAMATH_GPT_hotel_charge_percentage_l1538_153823

theorem hotel_charge_percentage (G R P : ℝ) 
  (hR : R = 1.60 * G) 
  (hP : P = 0.80 * G) : 
  ((R - P) / R) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_hotel_charge_percentage_l1538_153823


namespace NUMINAMATH_GPT_hcf_of_abc_l1538_153894

-- Given conditions
variables (a b c : ℕ)
def lcm_abc := Nat.lcm (Nat.lcm a b) c
def product_abc := a * b * c

-- Statement to prove
theorem hcf_of_abc (H1 : lcm_abc a b c = 1200) (H2 : product_abc a b c = 108000) : 
  Nat.gcd (Nat.gcd a b) c = 90 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_abc_l1538_153894


namespace NUMINAMATH_GPT_find_number_l1538_153800

theorem find_number (x : ℝ) : 0.40 * x = 0.80 * 5 + 2 → x = 15 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_number_l1538_153800


namespace NUMINAMATH_GPT_tangent_function_intersection_l1538_153803

theorem tangent_function_intersection (ω : ℝ) (hω : ω > 0) (h_period : (π / ω) = 3 * π) :
  let f (x : ℝ) := Real.tan (ω * x + π / 3)
  f π = -Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tangent_function_intersection_l1538_153803


namespace NUMINAMATH_GPT_trapezium_parallel_side_length_l1538_153849

theorem trapezium_parallel_side_length (a h area x : ℝ) (h1 : a = 20) (h2 : h = 15) (h3 : area = 285) :
  area = 1/2 * (a + x) * h → x = 18 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_trapezium_parallel_side_length_l1538_153849


namespace NUMINAMATH_GPT_no_solution_in_A_l1538_153893

def A : Set ℕ := 
  {n | ∃ k : ℤ, abs (n * Real.sqrt 2022 - 1 / 3 - k) ≤ 1 / 2022}

theorem no_solution_in_A (x y z : ℕ) (hx : x ∈ A) (hy : y ∈ A) (hz : z ∈ A) : 
  20 * x + 21 * y ≠ 22 * z := 
sorry

end NUMINAMATH_GPT_no_solution_in_A_l1538_153893


namespace NUMINAMATH_GPT_least_possible_k_l1538_153860

-- Define the conditions
def prime_factor_form (k : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ k = 2^a * 3^b * 5^c

def divisible_by_1680 (k : ℕ) : Prop :=
  (k ^ 4) % 1680 = 0

-- Define the proof problem
theorem least_possible_k (k : ℕ) (h_div : divisible_by_1680 k) (h_prime : prime_factor_form k) : k = 210 :=
by
  -- Statement of the problem, proof to be filled
  sorry

end NUMINAMATH_GPT_least_possible_k_l1538_153860


namespace NUMINAMATH_GPT_tile_difference_correct_l1538_153889

def initial_blue_tiles := 23
def initial_green_tiles := 16
def first_border_green_tiles := 6 * 1
def second_border_green_tiles := 6 * 2
def total_green_tiles := initial_green_tiles + first_border_green_tiles + second_border_green_tiles
def difference_tiling := total_green_tiles - initial_blue_tiles

theorem tile_difference_correct : difference_tiling = 11 := by
  sorry

end NUMINAMATH_GPT_tile_difference_correct_l1538_153889


namespace NUMINAMATH_GPT_remainder_when_four_times_number_minus_nine_divided_by_eight_l1538_153865

theorem remainder_when_four_times_number_minus_nine_divided_by_eight
  (n : ℤ) (h : n % 8 = 3) : (4 * n - 9) % 8 = 3 := by
  sorry

end NUMINAMATH_GPT_remainder_when_four_times_number_minus_nine_divided_by_eight_l1538_153865


namespace NUMINAMATH_GPT_solve_for_q_l1538_153872

-- Define the conditions
variables (p q : ℝ)
axiom condition1 : 3 * p + 4 * q = 8
axiom condition2 : 4 * p + 3 * q = 13

-- State the goal to prove q = -1
theorem solve_for_q : q = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_q_l1538_153872


namespace NUMINAMATH_GPT_find_c_l1538_153821

-- Let a, b, c, d, and e be positive consecutive integers.
variables {a b c d e : ℕ}

-- Conditions: 
def conditions (a b c d e : ℕ) : Prop :=
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a + b = e - 1 ∧
  a * b = d + 1

-- Proof statement
theorem find_c (h : conditions a b c d e) : c = 4 :=
by sorry

end NUMINAMATH_GPT_find_c_l1538_153821


namespace NUMINAMATH_GPT_minimum_sum_of_dimensions_l1538_153885

   theorem minimum_sum_of_dimensions (a b c : ℕ) (habc : a * b * c = 3003) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
     a + b + c = 45 :=
   sorry
   
end NUMINAMATH_GPT_minimum_sum_of_dimensions_l1538_153885


namespace NUMINAMATH_GPT_no_p_safe_numbers_l1538_153888

/-- A number n is p-safe if it differs in absolute value by more than 2 from all multiples of p. -/
def p_safe (n p : ℕ) : Prop := ∀ k : ℤ, abs (n - k * p) > 2 

/-- The main theorem stating that there are no numbers that are simultaneously 5-safe, 
    7-safe, and 9-safe from 1 to 15000. -/
theorem no_p_safe_numbers (n : ℕ) (hp : 1 ≤ n ∧ n ≤ 15000) : 
  ¬ (p_safe n 5 ∧ p_safe n 7 ∧ p_safe n 9) :=
sorry

end NUMINAMATH_GPT_no_p_safe_numbers_l1538_153888


namespace NUMINAMATH_GPT_find_fraction_l1538_153854

theorem find_fraction (F : ℝ) (N : ℝ) (X : ℝ)
  (h1 : 0.85 * F = 36)
  (h2 : N = 70.58823529411765)
  (h3 : F = 42.35294117647059) :
  X * N = 42.35294117647059 → X = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l1538_153854


namespace NUMINAMATH_GPT_households_with_only_bike_l1538_153868

theorem households_with_only_bike
  (N : ℕ) (H_no_car_or_bike : ℕ) (H_car_bike : ℕ) (H_car : ℕ)
  (hN : N = 90)
  (h_no_car_or_bike : H_no_car_or_bike = 11)
  (h_car_bike : H_car_bike = 16)
  (h_car : H_car = 44) :
  ∃ (H_bike_only : ℕ), H_bike_only = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_households_with_only_bike_l1538_153868


namespace NUMINAMATH_GPT_complex_root_condition_l1538_153884

open Complex

theorem complex_root_condition (u v : ℂ) 
    (h1 : 3 * abs (u + 1) * abs (v + 1) ≥ abs (u * v + 5 * u + 5 * v + 1))
    (h2 : abs (u + v) = abs (u * v + 1)) :
    u = 1 ∨ v = 1 :=
sorry

end NUMINAMATH_GPT_complex_root_condition_l1538_153884


namespace NUMINAMATH_GPT_photographer_choice_l1538_153852

theorem photographer_choice : 
  (Nat.choose 7 4) + (Nat.choose 7 5) = 56 := 
by 
  sorry

end NUMINAMATH_GPT_photographer_choice_l1538_153852


namespace NUMINAMATH_GPT_y1_gt_y2_l1538_153839

theorem y1_gt_y2 (y : ℤ → ℤ) (h_eq : ∀ x, y x = 8 * x - 1)
  (y1 y2 : ℤ) (h_y1 : y 3 = y1) (h_y2 : y 2 = y2) : y1 > y2 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_y1_gt_y2_l1538_153839


namespace NUMINAMATH_GPT_youngest_brother_age_l1538_153882

theorem youngest_brother_age 
  (x : ℤ) 
  (h1 : ∃ (a b c : ℤ), a = x ∧ b = x + 1 ∧ c = x + 2 ∧ a + b + c = 96) : 
  x = 31 :=
by sorry

end NUMINAMATH_GPT_youngest_brother_age_l1538_153882


namespace NUMINAMATH_GPT_negation_of_p_l1538_153826

def proposition_p := ∃ x : ℝ, x ≥ 1 ∧ x^2 - x < 0

theorem negation_of_p : (∀ x : ℝ, x ≥ 1 → x^2 - x ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l1538_153826


namespace NUMINAMATH_GPT_function_decreasing_range_k_l1538_153822

theorem function_decreasing_range_k : 
  ∀ k : ℝ, (∀ x : ℝ, 1 ≤ x → ∀ y : ℝ, 1 ≤ y → x ≤ y → (k * x ^ 2 + (3 * k - 2) * x - 5) ≥ (k * y ^ 2 + (3 * k - 2) * y - 5)) ↔ (k ∈ Set.Iic 0) :=
by sorry

end NUMINAMATH_GPT_function_decreasing_range_k_l1538_153822


namespace NUMINAMATH_GPT_carla_marbles_start_l1538_153850

-- Conditions defined as constants
def marblesBought : ℝ := 489.0
def marblesTotalNow : ℝ := 2778.0

-- Theorem statement
theorem carla_marbles_start (marblesBought marblesTotalNow: ℝ) :
  marblesTotalNow - marblesBought = 2289.0 := by
  sorry

end NUMINAMATH_GPT_carla_marbles_start_l1538_153850


namespace NUMINAMATH_GPT_percent_of_x_is_y_l1538_153861

variable {x y : ℝ}

theorem percent_of_x_is_y
  (h : 0.5 * (x - y) = 0.4 * (x + y)) :
  y = (1 / 9) * x :=
sorry

end NUMINAMATH_GPT_percent_of_x_is_y_l1538_153861


namespace NUMINAMATH_GPT_translation_graph_pass_through_point_l1538_153863

theorem translation_graph_pass_through_point :
  (∃ a : ℝ, (∀ x y : ℝ, y = -2 * x + 1 - 3 → y = 3 → x = a) → a = -5/2) :=
sorry

end NUMINAMATH_GPT_translation_graph_pass_through_point_l1538_153863


namespace NUMINAMATH_GPT_sum_of_two_numbers_eq_l1538_153873

theorem sum_of_two_numbers_eq (x y : ℝ) (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) : x + y = (16 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_GPT_sum_of_two_numbers_eq_l1538_153873


namespace NUMINAMATH_GPT_probability_X1_lt_X2_lt_X3_is_1_6_l1538_153829

noncomputable def probability_X1_lt_X2_lt_X3 (n : ℕ) (h : n ≥ 3) : ℚ :=
if h : n ≥ 3 then
  1/6
else
  0

theorem probability_X1_lt_X2_lt_X3_is_1_6 (n : ℕ) (h : n ≥ 3) :
  probability_X1_lt_X2_lt_X3 n h = 1/6 :=
sorry

end NUMINAMATH_GPT_probability_X1_lt_X2_lt_X3_is_1_6_l1538_153829


namespace NUMINAMATH_GPT_smallest_M_bound_l1538_153874

theorem smallest_M_bound {f : ℕ → ℝ} (hf1 : f 1 = 2) 
  (hf2 : ∀ n : ℕ, f (n + 1) ≥ f n ∧ f n ≥ (n / (n + 1)) * f (2 * n)) : 
  ∃ M : ℕ, (∀ n : ℕ, f n < M) ∧ M = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_M_bound_l1538_153874


namespace NUMINAMATH_GPT_ratio_cookies_to_pie_l1538_153864

def num_surveyed_students : ℕ := 800
def num_students_preferred_cookies : ℕ := 280
def num_students_preferred_pie : ℕ := 160

theorem ratio_cookies_to_pie : num_students_preferred_cookies / num_students_preferred_pie = 7 / 4 := by
  sorry

end NUMINAMATH_GPT_ratio_cookies_to_pie_l1538_153864


namespace NUMINAMATH_GPT_colored_ints_square_diff_l1538_153870

-- Define a coloring function c as a total function from ℤ to a finite set {0, 1, 2}
def c : ℤ → Fin 3 := sorry

-- Lean 4 statement for the problem
theorem colored_ints_square_diff : 
  ∃ a b : ℤ, a ≠ b ∧ c a = c b ∧ ∃ k : ℤ, a - b = k ^ 2 :=
sorry

end NUMINAMATH_GPT_colored_ints_square_diff_l1538_153870


namespace NUMINAMATH_GPT_part_a_part_b_l1538_153880

variable {A : Type*} [Ring A]

def B (A : Type*) [Ring A] : Set A :=
  {a | a^2 = 1}

variable (a : A) (b : B A)

theorem part_a (a : A) (b : A) (h : b ∈ B A) : a * b - b * a = b * a * b - a := by
  sorry

theorem part_b (A : Type*) [Ring A] (h : ∀ x : A, x^2 = 0 -> x = 0) : Group (B A) := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1538_153880


namespace NUMINAMATH_GPT_complete_residue_system_infinitely_many_positive_integers_l1538_153832

def is_complete_residue_system (n m : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → i ≠ j → (i^n % m ≠ j^n % m)

theorem complete_residue_system_infinitely_many_positive_integers (m : ℕ) (h_pos : 0 < m) :
  ∃ᶠ n in at_top, is_complete_residue_system n m :=
sorry

end NUMINAMATH_GPT_complete_residue_system_infinitely_many_positive_integers_l1538_153832


namespace NUMINAMATH_GPT_total_oranges_l1538_153843

def oranges_from_first_tree : Nat := 80
def oranges_from_second_tree : Nat := 60
def oranges_from_third_tree : Nat := 120

theorem total_oranges : oranges_from_first_tree + oranges_from_second_tree + oranges_from_third_tree = 260 :=
by
  sorry

end NUMINAMATH_GPT_total_oranges_l1538_153843


namespace NUMINAMATH_GPT_f_inequality_l1538_153876

-- Define the function f.
def f (x : ℝ) : ℝ := x^2 - x + 13

-- The main theorem to prove the given inequality.
theorem f_inequality (x m : ℝ) (h : |x - m| < 1) : |f x - f m| < 2*(|m| + 1) :=
by
  sorry

end NUMINAMATH_GPT_f_inequality_l1538_153876


namespace NUMINAMATH_GPT_fraction_subtraction_identity_l1538_153847

theorem fraction_subtraction_identity (x y : ℕ) (hx : x = 3) (hy : y = 4) : (1 / (x : ℚ) - 1 / (y : ℚ) = 1 / 12) :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_identity_l1538_153847


namespace NUMINAMATH_GPT_max_height_reached_by_rocket_l1538_153875

def h (t : ℝ) : ℝ := -12 * t^2 + 72 * t + 36

theorem max_height_reached_by_rocket : ∃ t : ℝ, h t = 144 ∧ ∀ t' : ℝ, h t' ≤ 144 := sorry

end NUMINAMATH_GPT_max_height_reached_by_rocket_l1538_153875


namespace NUMINAMATH_GPT_number_divisibility_l1538_153837

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem number_divisibility (n : ℕ) :
  (3^n ∣ A_n n) ∧ ¬ (3^(n + 1) ∣ A_n n) := by
  sorry

end NUMINAMATH_GPT_number_divisibility_l1538_153837


namespace NUMINAMATH_GPT_escher_consecutive_probability_l1538_153871

open Classical

noncomputable def probability_Escher_consecutive (total_pieces escher_pieces: ℕ): ℚ :=
  if total_pieces < escher_pieces then 0 else (Nat.factorial (total_pieces - escher_pieces) * Nat.factorial escher_pieces) / Nat.factorial (total_pieces - 1)

theorem escher_consecutive_probability :
  probability_Escher_consecutive 12 4 = 1 / 41 :=
by
  sorry

end NUMINAMATH_GPT_escher_consecutive_probability_l1538_153871


namespace NUMINAMATH_GPT_problem_l1538_153844

-- Helper definition for point on a line
def point_on_line (x y : ℝ) (a b : ℝ) : Prop := y = a * x + b

-- Given condition: Point P(1, 3) lies on the line y = 2x + b
def P_on_l (b : ℝ) : Prop := point_on_line 1 3 2 b

-- The proof problem: Proving (2, 5) also lies on the line y = 2x + b where b is the constant found using P
theorem problem (b : ℝ) (h: P_on_l b) : point_on_line 2 5 2 b :=
by
  sorry

end NUMINAMATH_GPT_problem_l1538_153844


namespace NUMINAMATH_GPT_factor_quadratic_l1538_153824

theorem factor_quadratic (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := 
by 
  sorry

end NUMINAMATH_GPT_factor_quadratic_l1538_153824


namespace NUMINAMATH_GPT_find_f_1000_l1538_153810

theorem find_f_1000 (f : ℕ → ℕ) 
    (h1 : ∀ n : ℕ, 0 < n → f (f n) = 2 * n) 
    (h2 : ∀ n : ℕ, 0 < n → f (3 * n + 1) = 3 * n + 2) : 
    f 1000 = 1008 :=
by
  sorry

end NUMINAMATH_GPT_find_f_1000_l1538_153810


namespace NUMINAMATH_GPT_smallest_prime_dividing_sum_l1538_153802

theorem smallest_prime_dividing_sum :
  ∃ p : ℕ, Prime p ∧ p ∣ (7^14 + 11^15) ∧ ∀ q : ℕ, Prime q ∧ q ∣ (7^14 + 11^15) → p ≤ q := by
  sorry

end NUMINAMATH_GPT_smallest_prime_dividing_sum_l1538_153802


namespace NUMINAMATH_GPT_number_of_triplets_l1538_153867

theorem number_of_triplets (N : ℕ) (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : 2017 ≥ 10 * a) (h5 : 10 * a ≥ 100 * b) (h6 : 100 * b ≥ 1000 * c) : 
  N = 574 := 
sorry

end NUMINAMATH_GPT_number_of_triplets_l1538_153867


namespace NUMINAMATH_GPT_abs_inequality_solution_l1538_153825

theorem abs_inequality_solution (x : ℝ) : 
  (|5 - 2*x| >= 3) ↔ (x ≤ 1 ∨ x ≥ 4) := sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1538_153825


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l1538_153891

theorem necessary_not_sufficient_condition {a : ℝ} :
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) →
  (¬ (∀ x : ℝ, x ≥ a → |x - 1| < 1)) →
  a ≤ 0 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l1538_153891


namespace NUMINAMATH_GPT_greatest_value_x_l1538_153836

theorem greatest_value_x (x : ℝ) : 
  (x ≠ 9) → 
  (x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 6) →
  x ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_x_l1538_153836


namespace NUMINAMATH_GPT_range_of_m_l1538_153817

open Set Real

noncomputable def A := {x : ℝ | x^2 - 2 * x - 3 < 0}
noncomputable def B (m : ℝ) := {x : ℝ | -1 < x ∧ x < m}

theorem range_of_m (m : ℝ) : 
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) → 3 < m :=
by sorry

end NUMINAMATH_GPT_range_of_m_l1538_153817


namespace NUMINAMATH_GPT_A_salary_less_than_B_by_20_percent_l1538_153801

theorem A_salary_less_than_B_by_20_percent (A B : ℝ) (h1 : B = 1.25 * A) : 
  (B - A) / B * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_A_salary_less_than_B_by_20_percent_l1538_153801


namespace NUMINAMATH_GPT_circle_area_l1538_153811

theorem circle_area (x y : ℝ) :
  (3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
  (π * ((1 / 2) * (1 / 2)) = (π / 4)) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_circle_area_l1538_153811


namespace NUMINAMATH_GPT_percentage_seeds_from_dandelions_l1538_153820

def Carla_sunflowers := 6
def Carla_dandelions := 8
def seeds_per_sunflower := 9
def seeds_per_dandelion := 12

theorem percentage_seeds_from_dandelions :
  96 / 150 * 100 = 64 := by
  sorry

end NUMINAMATH_GPT_percentage_seeds_from_dandelions_l1538_153820


namespace NUMINAMATH_GPT_solve_for_q_l1538_153845

variable (k h q : ℝ)

-- Conditions given in the problem
axiom cond1 : (3 / 4) = (k / 48)
axiom cond2 : (3 / 4) = ((h + 36) / 60)
axiom cond3 : (3 / 4) = ((q - 9) / 80)

-- Our goal is to state that q = 69
theorem solve_for_q : q = 69 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_solve_for_q_l1538_153845


namespace NUMINAMATH_GPT_consecutive_integers_sum_l1538_153866

theorem consecutive_integers_sum (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end NUMINAMATH_GPT_consecutive_integers_sum_l1538_153866


namespace NUMINAMATH_GPT_f_45_g_10_l1538_153830

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_condition1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom g_condition2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x + y) = g x + g y
axiom f_15 : f 15 = 10
axiom g_5 : g 5 = 3

theorem f_45 : f 45 = 10 / 3 := sorry
theorem g_10 : g 10 = 6 := sorry

end NUMINAMATH_GPT_f_45_g_10_l1538_153830


namespace NUMINAMATH_GPT_tutors_meeting_schedule_l1538_153869

/-- In a school, five tutors, Jaclyn, Marcelle, Susanna, Wanda, and Thomas, 
are scheduled to work in the library. Their schedules are as follows: 
Jaclyn works every fifth school day, Marcelle works every sixth school day, 
Susanna works every seventh school day, Wanda works every eighth school day, 
and Thomas works every ninth school day. Today, all five tutors are working 
in the library. Prove that the least common multiple of 5, 6, 7, 8, and 9 is 2520 days. 
-/
theorem tutors_meeting_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))) = 2520 := 
by
  sorry

end NUMINAMATH_GPT_tutors_meeting_schedule_l1538_153869


namespace NUMINAMATH_GPT_price_per_hotdog_l1538_153897

-- The conditions
def hot_dogs_per_hour := 10
def hours := 10
def total_sales := 200

-- Conclusion we need to prove
theorem price_per_hotdog : total_sales / (hot_dogs_per_hour * hours) = 2 := by
  sorry

end NUMINAMATH_GPT_price_per_hotdog_l1538_153897


namespace NUMINAMATH_GPT_find_x_values_l1538_153807

theorem find_x_values (x : ℝ) :
  x^3 - 9 * x^2 + 27 * x > 0 ↔ (0 < x ∧ x < 3) ∨ (6 < x) :=
by
  sorry

end NUMINAMATH_GPT_find_x_values_l1538_153807


namespace NUMINAMATH_GPT_lock_combination_l1538_153878

-- Define the digits as distinct
def distinct_digits (V E N U S I A R : ℕ) : Prop :=
  V ≠ E ∧ V ≠ N ∧ V ≠ U ∧ V ≠ S ∧ V ≠ I ∧ V ≠ A ∧ V ≠ R ∧
  E ≠ N ∧ E ≠ U ∧ E ≠ S ∧ E ≠ I ∧ E ≠ A ∧ E ≠ R ∧
  N ≠ U ∧ N ≠ S ∧ N ≠ I ∧ N ≠ A ∧ N ≠ R ∧
  U ≠ S ∧ U ≠ I ∧ U ≠ A ∧ U ≠ R ∧
  S ≠ I ∧ S ≠ A ∧ S ≠ R ∧
  I ≠ A ∧ I ≠ R ∧
  A ≠ R

-- Define the base 12 addition for the equation
def base12_addition (V E N U S I A R : ℕ) : Prop :=
  let VENUS := V * 12^4 + E * 12^3 + N * 12^2 + U * 12^1 + S
  let IS := I * 12^1 + S
  let NEAR := N * 12^3 + E * 12^2 + A * 12^1 + R
  let SUN := S * 12^2 + U * 12^1 + N
  VENUS + IS + NEAR = SUN

-- The theorem statement
theorem lock_combination :
  ∃ (V E N U S I A R : ℕ),
    distinct_digits V E N U S I A R ∧
    base12_addition V E N U S I A R ∧
    (S * 12^2 + U * 12^1 + N) = 655 := 
sorry

end NUMINAMATH_GPT_lock_combination_l1538_153878


namespace NUMINAMATH_GPT_valid_license_plates_count_l1538_153842

-- Defining the total number of choices for letters and digits
def num_letter_choices := 26
def num_digit_choices := 10

-- Function to calculate the total number of valid license plates
def total_license_plates := num_letter_choices ^ 3 * num_digit_choices ^ 4

-- The proof statement
theorem valid_license_plates_count : total_license_plates = 175760000 := 
by 
  -- The placeholder for the proof
  sorry

end NUMINAMATH_GPT_valid_license_plates_count_l1538_153842


namespace NUMINAMATH_GPT_pencil_price_in_units_l1538_153818

noncomputable def price_of_pencil_in_units (base_price additional_price unit_size : ℕ) : ℝ :=
  (base_price + additional_price) / unit_size

theorem pencil_price_in_units :
  price_of_pencil_in_units 5000 200 10000 = 0.52 := 
  by 
  sorry

end NUMINAMATH_GPT_pencil_price_in_units_l1538_153818


namespace NUMINAMATH_GPT_remainder_eq_52_l1538_153805

noncomputable def polynomial : Polynomial ℤ := Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C (-4) * Polynomial.X ^ 2 + Polynomial.C 7

theorem remainder_eq_52 : Polynomial.eval (-3) polynomial = 52 :=
by
    sorry

end NUMINAMATH_GPT_remainder_eq_52_l1538_153805


namespace NUMINAMATH_GPT_money_left_after_purchase_l1538_153853

-- The costs and amounts for each item
def bread_cost : ℝ := 2.35
def num_bread : ℝ := 4
def peanut_butter_cost : ℝ := 3.10
def num_peanut_butter : ℝ := 2
def honey_cost : ℝ := 4.50
def num_honey : ℝ := 1

-- The coupon discount and budget
def coupon_discount : ℝ := 2
def budget : ℝ := 20

-- Calculate the total cost before applying the coupon
def total_before_coupon : ℝ := num_bread * bread_cost + num_peanut_butter * peanut_butter_cost + num_honey * honey_cost

-- Calculate the total cost after applying the coupon
def total_after_coupon : ℝ := total_before_coupon - coupon_discount

-- Calculate the money left over after the purchase
def money_left_over : ℝ := budget - total_after_coupon

-- The theorem to be proven
theorem money_left_after_purchase : money_left_over = 1.90 :=
by
  -- The proof of this theorem will involve the specific calculations and will be filled in later
  sorry

end NUMINAMATH_GPT_money_left_after_purchase_l1538_153853


namespace NUMINAMATH_GPT_daughter_age_in_3_years_l1538_153862

theorem daughter_age_in_3_years (mother_age_now : ℕ) (h1 : mother_age_now = 41)
  (h2 : ∃ daughter_age_5_years_ago : ℕ, mother_age_now - 5 = 2 * daughter_age_5_years_ago) :
  ∃ daughter_age_in_3_years : ℕ, daughter_age_in_3_years = 26 :=
by {
  sorry
}

end NUMINAMATH_GPT_daughter_age_in_3_years_l1538_153862


namespace NUMINAMATH_GPT_total_cost_is_135_25_l1538_153804

-- defining costs and quantities
def cost_A : ℕ := 9
def num_A : ℕ := 4
def cost_B := cost_A + 5
def num_B : ℕ := 2
def cost_clay_pot := cost_A + 20
def cost_bag_soil := cost_A - 2
def cost_fertilizer := cost_A + (cost_A / 2)
def cost_gardening_tools := cost_clay_pot - (cost_clay_pot / 4)

-- total cost calculation
def total_cost : ℚ :=
  (num_A * cost_A) + 
  (num_B * cost_B) + 
  cost_clay_pot + 
  cost_bag_soil + 
  cost_fertilizer + 
  cost_gardening_tools

theorem total_cost_is_135_25 : total_cost = 135.25 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_135_25_l1538_153804


namespace NUMINAMATH_GPT_interior_diagonal_length_l1538_153859

variables (a b c : ℝ)

-- Conditions
def surface_area_eq : Prop := 2 * (a * b + b * c + c * a) = 22
def edge_length_eq : Prop := 4 * (a + b + c) = 24

-- Question to be proved
theorem interior_diagonal_length :
  surface_area_eq a b c → edge_length_eq a b c → (Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 14) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_interior_diagonal_length_l1538_153859


namespace NUMINAMATH_GPT_projection_of_a_in_direction_of_b_l1538_153814

noncomputable def vector_projection_in_direction (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / magnitude_b

theorem projection_of_a_in_direction_of_b :
  vector_projection_in_direction (3, 2) (-2, 1) = -4 * Real.sqrt 5 / 5 := 
by
  sorry

end NUMINAMATH_GPT_projection_of_a_in_direction_of_b_l1538_153814


namespace NUMINAMATH_GPT_impossible_to_have_only_stacks_of_three_l1538_153815

theorem impossible_to_have_only_stacks_of_three (n J : ℕ) (h_initial_n : n = 1) (h_initial_J : J = 1001) :
  (∀ n J, (n + J = 1002) → (∀ k : ℕ, 3 * k ≤ J → k + 3 * k ≠ 1002)) 
  :=
sorry

end NUMINAMATH_GPT_impossible_to_have_only_stacks_of_three_l1538_153815


namespace NUMINAMATH_GPT_john_safe_weight_l1538_153886

-- Assuming the conditions provided that form the basis of our problem.
def max_capacity : ℝ := 1000
def safety_margin : ℝ := 0.20
def john_weight : ℝ := 250
def safe_weight (max_capacity safety_margin john_weight : ℝ) : ℝ := 
  (max_capacity * (1 - safety_margin)) - john_weight

-- The main theorem to prove based on the provided problem statement.
theorem john_safe_weight : safe_weight max_capacity safety_margin john_weight = 550 := by
  -- skipping the proof details as instructed
  sorry

end NUMINAMATH_GPT_john_safe_weight_l1538_153886


namespace NUMINAMATH_GPT_probability_of_same_type_l1538_153838

-- Definitions for the given conditions
def total_books : ℕ := 12 + 9
def novels : ℕ := 12
def biographies : ℕ := 9

-- Define the number of ways to pick any two books
def total_ways_to_pick_two_books : ℕ := Nat.choose total_books 2

-- Define the number of ways to pick two novels
def ways_to_pick_two_novels : ℕ := Nat.choose novels 2

-- Define the number of ways to pick two biographies
def ways_to_pick_two_biographies : ℕ := Nat.choose biographies 2

-- Define the number of ways to pick two books of the same type
def ways_to_pick_two_books_of_same_type : ℕ := ways_to_pick_two_novels + ways_to_pick_two_biographies

-- Calculate the probability
noncomputable def probability_same_type (total_ways ways_same_type : ℕ) : ℚ :=
  ways_same_type / total_ways

theorem probability_of_same_type :
  probability_same_type total_ways_to_pick_two_books ways_to_pick_two_books_of_same_type = 17 / 35 := by
  sorry

end NUMINAMATH_GPT_probability_of_same_type_l1538_153838


namespace NUMINAMATH_GPT_chord_length_perpendicular_bisector_of_radius_l1538_153857

theorem chord_length_perpendicular_bisector_of_radius (r : ℝ) (h : r = 15) :
  ∃ (CD : ℝ), CD = 15 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_perpendicular_bisector_of_radius_l1538_153857


namespace NUMINAMATH_GPT_largest_inscribed_equilateral_triangle_area_l1538_153831

noncomputable def inscribed_triangle_area (r : ℝ) : ℝ :=
  let s := r * (3 / Real.sqrt 3)
  let h := (Real.sqrt 3 / 2) * s
  (1 / 2) * s * h

theorem largest_inscribed_equilateral_triangle_area :
  inscribed_triangle_area 10 = 75 * Real.sqrt 3 :=
by
  simp [inscribed_triangle_area]
  sorry

end NUMINAMATH_GPT_largest_inscribed_equilateral_triangle_area_l1538_153831


namespace NUMINAMATH_GPT_password_probability_l1538_153809

def isNonNegativeSingleDigit (n : ℕ) : Prop := n ≤ 9

def isOddSingleDigit (n : ℕ) : Prop := isNonNegativeSingleDigit n ∧ n % 2 = 1

def isPositiveSingleDigit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def isVowel (c : Char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

-- Probability that an odd single-digit number followed by a vowel and a positive single-digit number
def prob_odd_vowel_positive_digits : ℚ :=
  let prob_first := 5 / 10 -- Probability of odd single-digit number
  let prob_vowel := 5 / 26 -- Probability of vowel
  let prob_last := 9 / 10 -- Probability of positive single-digit number
  prob_first * prob_vowel * prob_last

theorem password_probability :
  prob_odd_vowel_positive_digits = 9 / 104 :=
by
  sorry

end NUMINAMATH_GPT_password_probability_l1538_153809


namespace NUMINAMATH_GPT_remainder_of_7_pow_51_mod_8_l1538_153851

theorem remainder_of_7_pow_51_mod_8 : (7^51 % 8) = 7 := sorry

end NUMINAMATH_GPT_remainder_of_7_pow_51_mod_8_l1538_153851


namespace NUMINAMATH_GPT_rational_square_l1538_153840

theorem rational_square (a b c : ℚ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) : ∃ r : ℚ, (1 / (a - b)^2) + (1 / (b - c)^2) + (1 / (c - a)^2) = r^2 := 
by 
  sorry

end NUMINAMATH_GPT_rational_square_l1538_153840


namespace NUMINAMATH_GPT_moving_circle_passes_through_fixed_point_l1538_153816

-- Define the parabola x^2 = 12y
def parabola (x y : ℝ) : Prop := x^2 = 12 * y

-- Define the directrix line y = -3
def directrix (y : ℝ) : Prop := y = -3

-- The fixed point we need to show the circle always passes through
def fixed_point : ℝ × ℝ := (0, 3)

-- Define the condition that the moving circle is centered on the parabola and tangent to the directrix
def circle_centered_on_parabola_and_tangent_to_directrix (x y : ℝ) (r : ℝ) : Prop :=
  parabola x y ∧ r = abs (y + 3)

-- Main theorem statement
theorem moving_circle_passes_through_fixed_point :
  (∀ (x y r : ℝ), circle_centered_on_parabola_and_tangent_to_directrix x y r → 
    (∃ (px py : ℝ), (px, py) = fixed_point ∧ (px - x)^2 + (py - y)^2 = r^2)) :=
sorry

end NUMINAMATH_GPT_moving_circle_passes_through_fixed_point_l1538_153816


namespace NUMINAMATH_GPT_profit_function_is_correct_marginal_profit_function_is_correct_profit_function_max_value_marginal_profit_function_max_value_profit_and_marginal_profit_max_not_equal_l1538_153881

noncomputable def R (x : ℕ) : ℝ := 3000 * x - 20 * x^2
noncomputable def C (x : ℕ) : ℝ := 500 * x + 4000
noncomputable def p (x : ℕ) : ℝ := R x - C x
noncomputable def Mp (x : ℕ) : ℝ := p (x + 1) - p x

theorem profit_function_is_correct : ∀ x, p x = -20 * x^2 + 2500 * x - 4000 := 
by 
  intro x
  sorry

theorem marginal_profit_function_is_correct : ∀ x, 0 < x ∧ x ≤ 100 → Mp x = -40 * x + 2480 := 
by 
  intro x
  sorry

theorem profit_function_max_value : ∃ x, (x = 62 ∨ x = 63) ∧ p x = 74120 :=
by 
  sorry

theorem marginal_profit_function_max_value : ∃ x, x = 1 ∧ Mp x = 2440 :=
by 
  sorry

theorem profit_and_marginal_profit_max_not_equal : ¬ (∃ x y, (x = 62 ∨ x = 63) ∧ y = 1 ∧ p x = Mp y) :=
by 
  sorry

end NUMINAMATH_GPT_profit_function_is_correct_marginal_profit_function_is_correct_profit_function_max_value_marginal_profit_function_max_value_profit_and_marginal_profit_max_not_equal_l1538_153881


namespace NUMINAMATH_GPT_sum_of_ages_five_years_from_now_l1538_153841

noncomputable def viggo_age_when_brother_was_2 (brother_age: ℕ) : ℕ :=
  10 + 2 * brother_age

noncomputable def current_viggo_age (viggo_age_at_2: ℕ) (current_brother_age: ℕ) : ℕ :=
  viggo_age_at_2 + (current_brother_age - 2)

def sister_age (viggo_age: ℕ) : ℕ :=
  viggo_age + 5

noncomputable def cousin_age (viggo_age: ℕ) (brother_age: ℕ) (sister_age: ℕ) : ℕ :=
  ((viggo_age + brother_age + sister_age) / 3)

noncomputable def future_ages_sum (viggo_age: ℕ) (brother_age: ℕ) (sister_age: ℕ) (cousin_age: ℕ) : ℕ :=
  viggo_age + 5 + brother_age + 5 + sister_age + 5 + cousin_age + 5

theorem sum_of_ages_five_years_from_now :
  let current_brother_age := 10
  let viggo_age_at_2 := viggo_age_when_brother_was_2 2
  let current_viggo_age := current_viggo_age viggo_age_at_2 current_brother_age
  let current_sister_age := sister_age current_viggo_age
  let current_cousin_age := cousin_age current_viggo_age current_brother_age current_sister_age
  future_ages_sum current_viggo_age current_brother_age current_sister_age current_cousin_age = 99 := sorry

end NUMINAMATH_GPT_sum_of_ages_five_years_from_now_l1538_153841


namespace NUMINAMATH_GPT_find_c_l1538_153812

theorem find_c (x c : ℝ) (h : ((5 * x + 38 + c) / 5) = (x + 4) + 5) : c = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1538_153812


namespace NUMINAMATH_GPT_frac_mul_square_l1538_153833

theorem frac_mul_square 
  : (8/9)^2 * (1/3)^2 = 64/729 := 
by 
  sorry

end NUMINAMATH_GPT_frac_mul_square_l1538_153833


namespace NUMINAMATH_GPT_Annika_hiking_rate_is_correct_l1538_153855

def AnnikaHikingRate
  (distance_partial_east distance_total_east : ℕ)
  (time_back_to_start : ℕ)
  (equality_rate : Nat) : Prop :=
  distance_partial_east = 2750 / 1000 ∧
  distance_total_east = 3500 / 1000 ∧
  time_back_to_start = 51 ∧
  equality_rate = 34

theorem Annika_hiking_rate_is_correct :
  ∃ R : ℕ, ∀ d1 d2 t,
  AnnikaHikingRate d1 d2 t R → R = 34 :=
by
  sorry

end NUMINAMATH_GPT_Annika_hiking_rate_is_correct_l1538_153855


namespace NUMINAMATH_GPT_price_of_fifth_basket_l1538_153887

-- Define the initial conditions
def avg_cost_of_4_baskets (total_cost_4 : ℝ) : Prop :=
  total_cost_4 / 4 = 4

def avg_cost_of_5_baskets (total_cost_5 : ℝ) : Prop :=
  total_cost_5 / 5 = 4.8

-- Theorem statement to be proved
theorem price_of_fifth_basket
  (total_cost_4 : ℝ)
  (h1 : avg_cost_of_4_baskets total_cost_4)
  (total_cost_5 : ℝ)
  (h2 : avg_cost_of_5_baskets total_cost_5) :
  total_cost_5 - total_cost_4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_price_of_fifth_basket_l1538_153887


namespace NUMINAMATH_GPT_pen_and_notebook_cost_l1538_153899

theorem pen_and_notebook_cost :
  ∃ (p n : ℕ), 17 * p + 5 * n = 200 ∧ p > n ∧ p + n = 16 := 
by
  sorry

end NUMINAMATH_GPT_pen_and_notebook_cost_l1538_153899


namespace NUMINAMATH_GPT_peter_has_142_nickels_l1538_153879

-- Define the conditions
def nickels (n : ℕ) : Prop :=
  40 < n ∧ n < 400 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 7 = 2

-- The theorem to prove the number of nickels
theorem peter_has_142_nickels : ∃ (n : ℕ), nickels n ∧ n = 142 :=
by {
  sorry
}

end NUMINAMATH_GPT_peter_has_142_nickels_l1538_153879


namespace NUMINAMATH_GPT_white_red_balls_l1538_153808

theorem white_red_balls (w r : ℕ) 
  (h1 : 3 * w = 5 * r)
  (h2 : w + 15 + r = 50) : 
  r = 12 :=
by
  sorry

end NUMINAMATH_GPT_white_red_balls_l1538_153808


namespace NUMINAMATH_GPT_circle_area_irrational_if_rational_diameter_l1538_153858

noncomputable def pi : ℝ := Real.pi

theorem circle_area_irrational_if_rational_diameter (d : ℚ) :
  ¬ ∃ (A : ℝ), A = pi * (d / 2)^2 ∧ (∃ (q : ℚ), A = q) :=
by
  sorry

end NUMINAMATH_GPT_circle_area_irrational_if_rational_diameter_l1538_153858


namespace NUMINAMATH_GPT_triangle_incircle_ratio_l1538_153890

theorem triangle_incircle_ratio
  (a b c : ℝ) (ha : a = 15) (hb : b = 12) (hc : c = 9)
  (r s : ℝ) (hr : r + s = c) (r_lt_s : r < s) :
  r / s = 1 / 2 :=
sorry

end NUMINAMATH_GPT_triangle_incircle_ratio_l1538_153890


namespace NUMINAMATH_GPT_price_difference_l1538_153856

theorem price_difference (total_cost shirt_price : ℝ) (h1 : total_cost = 80.34) (h2 : shirt_price = 36.46) :
  (total_cost - shirt_price) - shirt_price = 7.42 :=
by
  sorry

end NUMINAMATH_GPT_price_difference_l1538_153856


namespace NUMINAMATH_GPT_odd_function_def_l1538_153828

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x * (x - 1)
else -x * (x + 1)

theorem odd_function_def {x : ℝ} (h : x > 0) :
  f x = -x * (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_odd_function_def_l1538_153828
