import Mathlib

namespace measure_angle_F_correct_l584_584002

noncomputable def measure_angle_D : ℝ := 80
noncomputable def measure_angle_F : ℝ := 70 / 3
noncomputable def measure_angle_E (angle_F : ℝ) : ℝ := 2 * angle_F + 30
noncomputable def angle_sum_property (angle_D angle_E angle_F : ℝ) : Prop :=
  angle_D + angle_E + angle_F = 180

theorem measure_angle_F_correct : measure_angle_F = 70 / 3 :=
by
  let angle_D := measure_angle_D
  let angle_F := measure_angle_F
  have h1 : measure_angle_E angle_F = 2 * angle_F + 30 := rfl
  have h2 : angle_sum_property angle_D (measure_angle_E angle_F) angle_F := sorry
  sorry

end measure_angle_F_correct_l584_584002


namespace mr_blue_carrots_l584_584572

theorem mr_blue_carrots :
  let steps_length := 3 -- length of each step in feet
  let garden_length_steps := 25 -- length of garden in steps
  let garden_width_steps := 35 -- width of garden in steps
  let length_feet := garden_length_steps * steps_length -- length of garden in feet
  let width_feet := garden_width_steps * steps_length -- width of garden in feet
  let area_feet2 := length_feet * width_feet -- area of garden in square feet
  let yield_rate := 3 / 4 -- yield rate of carrots in pounds per square foot
  let expected_yield := area_feet2 * yield_rate -- expected yield in pounds
  expected_yield = 5906.25
:= by
  sorry

end mr_blue_carrots_l584_584572


namespace inequality_solution_l584_584607

theorem inequality_solution (x : ℝ) : 
  (x + 10) / (x^2 + 2 * x + 5) ≥ 0 ↔ x ∈ Set.Ici (-10) :=
sorry

end inequality_solution_l584_584607


namespace part1_part2_l584_584927

variables (a b c : ℝ)
-- Assuming a, b, c are positive and satisfy the given equation
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : 0 < c)
variable (h_eq : 4 * a ^ 2 + b ^ 2 + 16 * c ^ 2 = 1)

-- Statement for the first part: 0 < ab < 1/4
theorem part1 : 0 < a * b ∧ a * b < 1 / 4 :=
  sorry

-- Statement for the second part: 1/a² + 1/b² + 1/(4abc²) > 49
theorem part2 : 1 / (a ^ 2) + 1 / (b ^ 2) + 1 / (4 * a * b * c ^ 2) > 49 :=
  sorry

end part1_part2_l584_584927


namespace sin_60_proof_l584_584374

noncomputable def sin_60_eq_sqrt3_div_2 : Prop :=
  Real.sin (π / 3) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq_sqrt3_div_2 :=
sorry

end sin_60_proof_l584_584374


namespace math_problem_l584_584152

noncomputable theory

def coefficient_of_term_in_expansion (f : ℕ → ℕ → ℕ) : Prop :=
  f 5 2 = -90

theorem math_problem :
  coefficient_of_term_in_expansion (λ a b => 
    (∑ r in Finset.range (a + 1), (Nat.choose a r) * (3:ℤ)^(a-r) * (x * x)^r * y^b)) :=
sorry

end math_problem_l584_584152


namespace constant_term_is_8_l584_584811

open BigOperators

noncomputable def constant_term_in_expansion (a : ℝ) : ℝ :=
  let f := (X^2 + a) * (1 + (1 / X^2)) ^ 6 in
  harmonic_term_eval f 0

theorem constant_term_is_8 (a : ℝ) (h : ∑_(i : ℕ) coefficient (expand (X^2 + a) * (1 + (1 / X^2)) ^ 6) i = 192) :
  constant_term_in_expansion a = 8 :=
sorry

end constant_term_is_8_l584_584811


namespace range_of_a_l584_584045

def f (a x : ℝ) : ℝ := x^2 - a*x + a + 3
def g (a x : ℝ) : ℝ := x - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(f a x < 0 ∧ g a x < 0)) ↔ a ∈ Set.Icc (-3 : ℝ) 6 :=
sorry

end range_of_a_l584_584045


namespace income_growth_l584_584511

theorem income_growth (x : ℝ) : 12000 * (1 + x)^2 = 14520 :=
sorry

end income_growth_l584_584511


namespace perfect_square_trinomial_l584_584452

noncomputable def p (k : ℝ) (x : ℝ) : ℝ :=
  4 * x^2 + 2 * k * x + 9

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, p k x = (2 * x + b)^2) → (k = 6 ∨ k = -6) :=
by 
  intro h
  sorry

end perfect_square_trinomial_l584_584452


namespace problem1_problem2_l584_584349

theorem problem1 : -7 + 13 - 6 + 20 = 20 := 
by
  sorry

theorem problem2 : -2^3 + (2 - 3) - 2 * (-1)^2023 = -7 := 
by
  sorry

end problem1_problem2_l584_584349


namespace convert_536_oct_to_base7_l584_584386

def octal_to_decimal (n : ℕ) : ℕ :=
  n % 10 + (n / 10 % 10) * 8 + (n / 100 % 10) * 64

def decimal_to_base7 (n : ℕ) : ℕ :=
  n % 7 + (n / 7 % 7) * 10 + (n / 49 % 7) * 100 + (n / 343 % 7) * 1000

theorem convert_536_oct_to_base7 : 
  decimal_to_base7 (octal_to_decimal 536) = 1010 :=
by
  sorry

end convert_536_oct_to_base7_l584_584386


namespace smallest_perimeter_triangle_l584_584240

theorem smallest_perimeter_triangle (PQ PR QR : ℕ) (J : Point) :
  PQ = PR →
  QJ = 10 →
  QR = 2 * 10 →
  PQ + PR + QR = 40 :=
by
  sorry

structure Point : Type :=
mk :: (QJ : ℕ)

noncomputable def smallest_perimeter_triangle : Prop :=
  ∃ (PQ PR QR : ℕ) (J : Point), PQ = PR ∧ J.QJ = 10 ∧ QR = 2 * 10 ∧ PQ + PR + QR = 40

end smallest_perimeter_triangle_l584_584240


namespace smallest_angle_of_convex_15_gon_arithmetic_sequence_l584_584993

theorem smallest_angle_of_convex_15_gon_arithmetic_sequence :
  ∃ (a d : ℕ), (∀ k : ℕ, k < 15 → (let angle := a + k * d in angle < 180)) ∧
  (∀ i j : ℕ, i < j → i < 15 → j < 15 → (a + i * d) < (a + j * d)) ∧
  (let sequence_sum := 15 * a + d * 7 * 14 in sequence_sum = 2340) ∧
  (d = 3) ∧
  (a = 135) :=
by
  sorry

end smallest_angle_of_convex_15_gon_arithmetic_sequence_l584_584993


namespace part_a_part_b_l584_584719

-- Definitions based on the conditions:
def probability_of_hit (p : ℝ) := p
def probability_of_miss (p : ℝ) := 1 - p

-- Condition: exactly three unused rockets after firing at five targets
def exactly_three_unused_rockets (p : ℝ) : ℝ := 10 * (probability_of_hit p) ^ 3 * (probability_of_miss p) ^ 2

-- Condition: expected number of targets hit when there are nine targets
def expected_targets_hit (p : ℝ) : ℝ := 10 * p - p ^ 10

-- Lean 4 statements representing the proof problems:
theorem part_a (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) : 
  exactly_three_unused_rockets p = 10 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

theorem part_b (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) :
  expected_targets_hit p = 10 * p - p ^ 10 :=
by sorry

end part_a_part_b_l584_584719


namespace cars_per_day_l584_584688

noncomputable def paul_rate : ℝ := 2
noncomputable def jack_rate : ℝ := 3
noncomputable def paul_jack_rate : ℝ := paul_rate + jack_rate
noncomputable def hours_per_day : ℝ := 8
noncomputable def total_cars : ℝ := paul_jack_rate * hours_per_day

theorem cars_per_day : total_cars = 40 := by
  sorry

end cars_per_day_l584_584688


namespace simplify_sqrt_sum_l584_584093

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l584_584093


namespace simplify_sqrt_sum_l584_584139

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l584_584139


namespace average_of_first_two_numbers_l584_584980

theorem average_of_first_two_numbers (s1 s2 s3 s4 s5 s6 a b c : ℝ) 
  (h_average_six : (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 4.6)
  (h_average_set2 : (s3 + s4) / 2 = 3.8)
  (h_average_set3 : (s5 + s6) / 2 = 6.6)
  (h_total_sum : s1 + s2 + s3 + s4 + s5 + s6 = 27.6) : 
  (s1 + s2) / 2 = 3.4 :=
sorry

end average_of_first_two_numbers_l584_584980


namespace unique_triplet_l584_584759

theorem unique_triplet (a b c : ℝ) : 
  2 * a + 2 * b + 6 = 5 * c ∧ 
  ({a^2 - 4 * c, b^2 - 2 * a, c^2 - 2 * b} = {a - c, b - 4 * c, a + b}) ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  (a, b, c) = (1, 1, 2) := 
by
  sorry

end unique_triplet_l584_584759


namespace unique_three_digit_numbers_count_l584_584660

theorem unique_three_digit_numbers_count :
  ∃ l : List Nat, (∀ n ∈ l, 100 ≤ n ∧ n < 1000) ∧ 
    l = [230, 203, 302, 320] ∧ l.length = 4 := 
by
  sorry

end unique_three_digit_numbers_count_l584_584660


namespace shaded_area_proof_l584_584892

-- Given Definitions
def rectangle_area (length : ℕ) (width : ℕ) : ℕ := length * width
def triangle_area (base : ℕ) (height : ℕ) : ℕ := (base * height) / 2

-- Conditions
def grid_area : ℕ :=
  rectangle_area 2 3 + rectangle_area 3 4 + rectangle_area 4 5

def unshaded_triangle_area : ℕ := triangle_area 12 4

-- Question
def shaded_area : ℕ := grid_area - unshaded_triangle_area

-- Proof statement
theorem shaded_area_proof : shaded_area = 14 := by
  sorry

end shaded_area_proof_l584_584892


namespace scientific_notation_of_soap_bubble_l584_584316

theorem scientific_notation_of_soap_bubble :
  (0.0000007 : ℝ) = 7 * 10 ^ (-7) :=
by
  sorry

end scientific_notation_of_soap_bubble_l584_584316


namespace difference_two_numbers_is_five_l584_584638

def nat_number := ℕ

theorem difference_two_numbers_is_five
  (a b : nat_number)
  (h_sum : a + b = 18350)
  (h_divisible_by_5 : ∃ k : nat_number, b = 10 * k + 5)
  (h_units_digit_change : ∃ k : nat_number, a = 10 * k + 5) :
  abs (a - b) = 5 :=
sorry

end difference_two_numbers_is_five_l584_584638


namespace distance_from_P_to_l_l584_584209

open Real

def curve (x : ℝ) : ℝ :=
  2 * x - x^3

def line_l (x y : ℝ) : Prop :=
  x + y + 2 = 0

def point_on_curve (x_val : ℝ) (curve : ℝ → ℝ) : ℝ × ℝ :=
  (x_val, curve x_val)

def tangent_at_point (x_val : ℝ) (curve : ℝ → ℝ) : ℝ → ℝ × ℝ :=
  let slope := deriv curve x_val in
  let y_val := curve x_val in
  (fun x => (x, slope * (x - x_val) + y_val))

def distance_point_to_line (P : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  abs (3 + 2 + 2) / sqrt (1 + 1)

theorem distance_from_P_to_l : distance_point_to_line (3, 2) line_l = 7 * sqrt 2 / 2 := 
by
  sorry

end distance_from_P_to_l_l584_584209


namespace eccentricity_squared_l584_584444

variable (a b c : ℝ) (e : ℝ)
variable (F1 F2 P A : ℝ × ℝ)

-- Conditions
axiom ellipse_eq : a > b ∧ b > 0 ∧ (P.1^2 / a^2) + (P.2^2 / b^2) = 1
axiom foci_def : F1 = (c, 0) ∧ F2 = (-c, 0)
axiom P_on_ellipse : (P.1, P.2) satisfies ellipse_eq
axiom perpendicular_PF2_F1F2 : (P.1 - (-c)) * (F1.2 - F2.2) = - (P.2 - F2.2) * (F1.1 - F2.1)
axiom perpendicular_to_x_axis : (A.2 = 0) ∧ (P.1, P.2) and (A.1, 0) are projections of line from P to F1P
axiom AF2_eq_half_c : dist A F2 = c / 2
axiom eccentricity_def : e = c / a

-- Proof Statement
theorem eccentricity_squared : e^2 = (3 - Real.sqrt 5) / 2 :=
sorry

end eccentricity_squared_l584_584444


namespace set_complement_l584_584977

variable {U : Set ℝ} (A : Set ℝ)

theorem set_complement :
  (U = {x : ℝ | x > 1}) →
  (A ⊆ U) →
  (U \ A = {x : ℝ | x > 9}) →
  (A = {x : ℝ | 1 < x ∧ x ≤ 9}) :=
by
  intros hU hA hC
  sorry

end set_complement_l584_584977


namespace sum_a3_a7_l584_584888

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_a3_a7 (a : ℕ → ℝ)
  (h₁ : arithmetic_sequence a)
  (h₂ : a 1 + a 9 + a 2 + a 8 = 20) :
  a 3 + a 7 = 10 :=
sorry

end sum_a3_a7_l584_584888


namespace tangent_line_value_intersecting_line_values_line_equations_l584_584446

noncomputable def circleC (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*y + 12 = 0

def lineL (a x y : ℝ) : Prop :=
  a * x + y + 2 * a = 0

theorem tangent_line_value (a : ℝ) :
  ( ∃ x y, circleC x y ∧ lineL a x y ∧ (∀ x' y', circleC x' y' ∧ lineL a x' y' → (x = x' ∧ y = y'))) ↔ (a = -3/4) :=
sorry

theorem intersecting_line_values (a : ℝ) :
  ( ∃ x1 y1 x2 y2, circleC x1 y1 ∧ circleC x2 y2 ∧ lineL a x1 y1 ∧ lineL a x2 y2 ∧ 
    ((x1 - x2)^2 + (y1 - y2)^2 = (2 * real.sqrt 2)^2)) ↔ 
  (a = -7 ∨ a = -1) :=
sorry

theorem line_equations : 
  ( ∃ a, (a = -7 ∨ a = -1) → ( ∃ x1 y1 x2 y2, 
    circleC x1 y1 ∧ circleC x2 y2 ∧ 
    lineL a x1 y1 ∧ lineL a x2 y2 ∧ 
    ((x1 - x2)^2 + (y1 - y2)^2 = (2 * real.sqrt 2)^2) ) ) → 
  (( ∃ x y, lineL (-7) x y ) ∧ ( ∃ x y, lineL (-1) x y)) :=
sorry

end tangent_line_value_intersecting_line_values_line_equations_l584_584446


namespace triangle_KBC_area_l584_584894

noncomputable def hexagon_conditions : Prop :=
  let equiangular : Prop := true -- Since we don't have further information implicating this condition
  let area_ABJI : Prop := (4 * sqrt 2) ^ 2 = 32
  let area_FEHG : Prop := (3 * sqrt 2) ^ 2 = 18
  let right_triangle_JBK : Prop := ∀ JB BK, JB = 4 * sqrt 2 → BK = sqrt 14 → right_triangle JB BK 
  let FE_eq_BC : Prop := 3 * sqrt 2 = 3 * sqrt 2
  equiangular ∧ area_ABJI ∧ area_FEHG ∧ right_triangle_JBK ∧ FE_eq_BC

theorem triangle_KBC_area : hexagon_conditions → 
  let BC := 3 * sqrt 2
  let BK := sqrt 14
  let area_KBC := (BC * BK) / 2
  area_KBC = 3 * sqrt 28 / 2 :=
begin
  intros,
  rw [BC, BK, area_KBC],
  norm_num,
  exact (by linarith),
end

end triangle_KBC_area_l584_584894


namespace simplify_sqrt_sum_l584_584095

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l584_584095


namespace custom_star_calc_l584_584396

-- defining the custom operation "*"
def custom_star (a b : ℤ) : ℤ :=
  a * b - (b-1) * b

-- providing the theorem statement
theorem custom_star_calc : custom_star 2 (-3) = -18 :=
  sorry

end custom_star_calc_l584_584396


namespace collinear_vectors_sum_eq_five_l584_584483

variables {x y : ℝ}

def vec_a := (x, 2, 2)
def vec_b := (2, y, 4)

theorem collinear_vectors_sum_eq_five (h : ∃ m : ℝ, vec_b = (m * x, m * 2, m * 2)) : x + y = 5 :=
sorry

end collinear_vectors_sum_eq_five_l584_584483


namespace simplify_sqrt72_add_sqrt32_l584_584127

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l584_584127


namespace inequality_proof_l584_584077

-- Assuming C function is defined for combination as follows:
def C (n k : ℕ) : ℕ := if k > n then 0 else Nat.choose n k

def P_k_X (X : Set) (k : ℕ) : Prop := sorry -- placeholder for the property P_k(X)

def M (n k h : ℕ) : ℕ := sorry -- assuming it's defined according to the context

theorem inequality_proof (n k h : ℕ) : 
  (C n h / C k h) ≤ M n k h ∧ M n k h ≤ C (n-k+h) h := 
by
  sorry

end inequality_proof_l584_584077


namespace original_number_of_boys_l584_584645

/-- 
There are some boys and 13 girls in a class. If 1 boy is added
to the class, the percentage of the class that are girls is 52%. 
How many boys were originally in the class? 
-/
theorem original_number_of_boys (b : ℕ) (h1 : if (1 + 13) / (b + 1 + 13) = 0.52 then 13 / (b + 14) = 0.52) :
  b = 11 :=
sorry

end original_number_of_boys_l584_584645


namespace all_metals_conduct_electricity_l584_584695

def Gold_conducts : Prop := sorry
def Silver_conducts : Prop := sorry
def Copper_conducts : Prop := sorry
def Iron_conducts : Prop := sorry
def inductive_reasoning : Prop := sorry

theorem all_metals_conduct_electricity (g: Gold_conducts) (s: Silver_conducts) (c: Copper_conducts) (i: Iron_conducts) : inductive_reasoning := 
sorry

end all_metals_conduct_electricity_l584_584695


namespace nested_parens_eq_one_for_1999_l584_584605

theorem nested_parens_eq_one_for_1999 : ∀ x : ℝ, x - (iterate (λ y, x - y) 1999 (x - 1)) = 1 :=
by 
  intro x
  sorry

end nested_parens_eq_one_for_1999_l584_584605


namespace sin_60_eq_sqrt_three_div_two_l584_584359

theorem sin_60_eq_sqrt_three_div_two :
  Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_60_eq_sqrt_three_div_two_l584_584359


namespace net_rate_of_pay_is_25_dollars_per_hour_l584_584711

variables (hours : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (compensation_rate : ℝ) (gas_cost_per_gallon : ℝ)
variables (total_distance : ℝ) (gallons_used : ℝ) (earnings : ℝ) (gas_cost : ℝ) (net_earnings : ℝ) (net_rate_per_hour : ℝ)

def problem_conditions : Prop :=
  hours = 3 ∧
  speed = 50 ∧
  fuel_efficiency = 25 ∧
  compensation_rate = 0.60 ∧
  gas_cost_per_gallon = 2.50 ∧
  total_distance = hours * speed ∧
  gallons_used = total_distance / fuel_efficiency ∧
  earnings = total_distance * compensation_rate ∧
  gas_cost = gallons_used * gas_cost_per_gallon ∧
  net_earnings = earnings - gas_cost ∧
  net_rate_per_hour = net_earnings / hours

theorem net_rate_of_pay_is_25_dollars_per_hour (h : problem_conditions) : net_rate_per_hour = 25 :=
sorry

end net_rate_of_pay_is_25_dollars_per_hour_l584_584711


namespace ratio_black_to_blue_socks_l584_584905

theorem ratio_black_to_blue_socks {b : ℕ} {x : ℕ} 
  (original_cost : 12 * x + b * x)
  (interchanged_cost : 2 * b * x + 6 * x = 1.6 * (12 * x + b * x)) :
  6 / b = 2 / 11 :=
by
  sorry

end ratio_black_to_blue_socks_l584_584905


namespace number_of_solutions_l584_584456

theorem number_of_solutions : 
  ∃ n : ℕ, (∀ x : ℕ, x < 150 ∧ x > 0 → (x + 25) % 47 = 80 % 47 → x ∈ {8 + 47*k | k : ℕ ∧ k < 4}) ∧ n = 4 :=
by
  sorry

end number_of_solutions_l584_584456


namespace math_problem_l584_584748

theorem math_problem :
  (-1)^2024 + (-10) / (1/2) * 2 + (2 - (-3)^3) = -10 := by
  sorry

end math_problem_l584_584748


namespace infinite_pairs_exist_l584_584076

noncomputable def a : ℕ → ℕ
| 0     := 4
| 1     := 11
| (n+2) := 3 * a (n+1) - a n

theorem infinite_pairs_exist :
  ∀ n : ℕ,
  let a_n := a n,
  let a_n1 := a (n+1)
  in gcd a_n a_n1 = 1 ∧ a_n ∣ (a_n1^2 - 5) ∧ a_n1 ∣ (a_n^2 - 5) :=
by
  sorry

end infinite_pairs_exist_l584_584076


namespace distance_after_12_sec_time_to_travel_380_meters_l584_584623

-- Define the function expressing the distance s in terms of the travel time t
def distance (t : ℝ) : ℝ := 9 * t + (1 / 2) * t^2

-- Proof problem 1: Distance traveled after 12 seconds
theorem distance_after_12_sec : distance 12 = 180 := 
sorry

-- Proof problem 2: Time to travel 380 meters
theorem time_to_travel_380_meters (t : ℝ) (h : distance t = 380) : t = 20 := 
sorry

end distance_after_12_sec_time_to_travel_380_meters_l584_584623


namespace no_rectangle_fully_covered_l584_584899

noncomputable theory
open Classical

structure Rectangle where
  width  : ℝ
  height : ℝ

def T (n : ℕ) : Rectangle :=
  { width := 8^n, height := 1 / 8^n }

def area (r : Rectangle) : ℝ := r.width * r.height

theorem no_rectangle_fully_covered (n : ℕ) :
  ∀ {m : ℕ}, m ≠ n → ¬ (∃ (covers : List Rectangle), (∀ r ∈ covers, r ≠ T n) ∧ (covers.map area).sum ≥ area (T n)) :=
by
  intro m h
  exact sorry

end no_rectangle_fully_covered_l584_584899


namespace circle_equations_centered_on_line_y_equals_2x_l584_584281

noncomputable def circle_equation : set (set (ℝ × ℝ)) :=
  {circle | ∃ a R, 
    circle = {p | (p.1 - a)^2 + (p.2 - 2 * a)^2 = R^2} ∧
    (a - 2)^2 + (2 * a)^2 = R^2 ∧
    a^2 + (2 * a - 4)^2 = R^2}

theorem circle_equations_centered_on_line_y_equals_2x :
  circle_equation =
    {{p | (p.1 - 1)^2 + (p.2 - 2)^2 = 5} ∨ {p | (p.1 + 1)^2 + (p.2 + 2)^2 = 5}} :=
  sorry

end circle_equations_centered_on_line_y_equals_2x_l584_584281


namespace distance_corresponds_to_additional_charge_l584_584012

-- Define the initial fee
def initial_fee : ℝ := 2.5

-- Define the charge per part of a mile
def charge_per_part_of_mile : ℝ := 0.35

-- Define the total charge for a 3.6 miles trip
def total_charge : ℝ := 5.65

-- Define the correct distance corresponding to the additional charge
def correct_distance : ℝ := 0.9

-- The theorem to prove
theorem distance_corresponds_to_additional_charge :
  (total_charge - initial_fee) / charge_per_part_of_mile * (0.1) = correct_distance :=
by
  sorry

end distance_corresponds_to_additional_charge_l584_584012


namespace students_who_wanted_fruit_l584_584168

theorem students_who_wanted_fruit (red_apples green_apples extra_apples ordered_apples served_apples students_wanted_fruit : ℕ)
    (h1 : red_apples = 43)
    (h2 : green_apples = 32)
    (h3 : extra_apples = 73)
    (h4 : ordered_apples = red_apples + green_apples)
    (h5 : served_apples = ordered_apples + extra_apples)
    (h6 : students_wanted_fruit = served_apples - ordered_apples) :
    students_wanted_fruit = 73 := 
by
    sorry

end students_who_wanted_fruit_l584_584168


namespace remainder_polynomial_division_l584_584423

noncomputable def polynomial := Polynomial ℚ

theorem remainder_polynomial_division :
  let f : polynomial := X^4 + 4 * X^2 + 2
  let g : polynomial := (X - 2)^2
  let r : polynomial := 4 * X^2 - 16 * X - 30
  (EuclideanDomain.modBy f g) = r :=
by
  sorry

end remainder_polynomial_division_l584_584423


namespace fraction_female_robins_l584_584879

-- Define the given conditions
def fraction_robins : ℚ := 2/5
def fraction_bluejays : ℚ := 3/5
def fraction_female_bluejays : ℚ := 2/3
def fraction_male_birds : ℚ := 7/15

-- Define the fraction of robins that are female such that the solution can be derived
theorem fraction_female_robins (
    h1: fraction_robins = 2/5
    h2: fraction_bluejays = 3/5
    h3: fraction_female_bluejays = 2/3
    h4: fraction_male_birds = 7/15
) : fraction_female_robins = 1/3 := by
  sorry

end fraction_female_robins_l584_584879


namespace sum_fractions_eq_1001_l584_584931

def f (x : ℝ) : ℝ := 3 / (4^x + 3)

theorem sum_fractions_eq_1001 :
  ∑ k in Finset.range 2002 \ {0}, f (k / 2002) = 1001 := by
sorry

end sum_fractions_eq_1001_l584_584931


namespace no_integer_solution_l584_584765

theorem no_integer_solution (a b : ℤ) : ¬ (4 ∣ a^2 + b^2 + 1) :=
by
  -- Prevent use of the solution steps and add proof obligations
  sorry

end no_integer_solution_l584_584765


namespace part_a_part_b_l584_584721

-- Definitions based on the conditions:
def probability_of_hit (p : ℝ) := p
def probability_of_miss (p : ℝ) := 1 - p

-- Condition: exactly three unused rockets after firing at five targets
def exactly_three_unused_rockets (p : ℝ) : ℝ := 10 * (probability_of_hit p) ^ 3 * (probability_of_miss p) ^ 2

-- Condition: expected number of targets hit when there are nine targets
def expected_targets_hit (p : ℝ) : ℝ := 10 * p - p ^ 10

-- Lean 4 statements representing the proof problems:
theorem part_a (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) : 
  exactly_three_unused_rockets p = 10 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

theorem part_b (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) :
  expected_targets_hit p = 10 * p - p ^ 10 :=
by sorry

end part_a_part_b_l584_584721


namespace omar_drinks_2_5_ounces_l584_584953

-- Defining the conditions as hypotheses
def initial_coffee : ℕ := 12

def coffee_after_commute (initial: ℕ) : ℕ := initial - (initial / 4)

def coffee_after_office (after_commute: ℕ) : ℕ := after_commute - (after_commute / 2)

def coffee_left (after_office: ℕ) (remaining: ℕ) : ℕ := after_office - remaining

-- The proposition we need to prove
theorem omar_drinks_2_5_ounces :
  coffee_left (coffee_after_office (coffee_after_commute initial_coffee)) 2 = 2.5
:=
  sorry

end omar_drinks_2_5_ounces_l584_584953


namespace convex_15gon_smallest_angle_arith_seq_l584_584989

noncomputable def smallest_angle (n : ℕ) (avg_angle d : ℕ) : ℕ :=
156 - 7 * d

theorem convex_15gon_smallest_angle_arith_seq :
  let n := 15 in
  ∀ (a d : ℕ), 
  (a = 156 - 7 * d) ∧
  (avg_angle = (13 * 180) / n) ∧
  (forall i : ℕ, 1 ≤ i ∧ i < n → d < 24 / 7) →
  a = 135 :=
sorry

end convex_15gon_smallest_angle_arith_seq_l584_584989


namespace convex_15gon_smallest_angle_arith_seq_l584_584991

noncomputable def smallest_angle (n : ℕ) (avg_angle d : ℕ) : ℕ :=
156 - 7 * d

theorem convex_15gon_smallest_angle_arith_seq :
  let n := 15 in
  ∀ (a d : ℕ), 
  (a = 156 - 7 * d) ∧
  (avg_angle = (13 * 180) / n) ∧
  (forall i : ℕ, 1 ≤ i ∧ i < n → d < 24 / 7) →
  a = 135 :=
sorry

end convex_15gon_smallest_angle_arith_seq_l584_584991


namespace cosine_problem_l584_584683

theorem cosine_problem :
  (cos (70 * π / 180) + cos (50 * π / 180)) * (cos (310 * π / 180) + cos (290 * π / 180)) +
  (cos (40 * π / 180) + cos (160 * π / 180)) * (cos (320 * π / 180) - cos (380 * π / 180)) = 1 :=
by
  sorry

end cosine_problem_l584_584683


namespace total_tickets_sold_l584_584216

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

end total_tickets_sold_l584_584216


namespace extremum_neither_necessary_nor_sufficient_l584_584506

theorem extremum_neither_necessary_nor_sufficient {α : Type*} [TopologicalSpace α] 
  [NormedAddCommGroup α] [NormedSpace ℝ α] {f : ℝ → α} (a : ℝ) :
  ContinuousAt f a →
  ¬ (∀ x ∈ set.univ, deriv f x = 0 → f x ≤ f a ∧ f a ≤ f x) ∧
  ¬ (∀ x ∈ set.univ, (f x ≤ f a ∧ f a ≤ f x) → deriv f x = 0) :=
begin
  sorry
end

end extremum_neither_necessary_nor_sufficient_l584_584506


namespace train_length_l584_584322

variable (L : ℝ) -- The length of the train

def length_of_platform : ℝ := 250 -- The length of the platform

def time_to_cross_platform : ℝ := 33 -- Time to cross the platform in seconds

def time_to_cross_pole : ℝ := 18 -- Time to cross the signal pole in seconds

-- The speed of the train is constant whether it crosses the platform or the signal pole.
-- Therefore, we equate the expressions for speed and solve for L.
theorem train_length (h1 : time_to_cross_platform * L = time_to_cross_pole * (L + length_of_platform)) :
  L = 300 :=
by
  -- Proof will be here
  sorry

end train_length_l584_584322


namespace pages_per_side_is_4_l584_584533

-- Define the conditions
def num_books := 2
def pages_per_book := 600
def sheets_used := 150
def sides_per_sheet := 2

-- Define the total number of pages and sides
def total_pages := num_books * pages_per_book
def total_sides := sheets_used * sides_per_sheet

-- Prove the number of pages per side is 4
theorem pages_per_side_is_4 : total_pages / total_sides = 4 := by
  sorry

end pages_per_side_is_4_l584_584533


namespace part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l584_584733

-- Proof Problem for Part (a):
theorem part_a_probability_three_unused_rockets (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ q : ℝ, q = 10 * p^3 * (1-p)^2) := sorry

-- Proof Problem for Part (b):
theorem part_b_expected_targets_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ e : ℝ, e = 10 * p - p^10) := sorry

end part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l584_584733


namespace middle_part_is_28_4_over_11_l584_584852

theorem middle_part_is_28_4_over_11 (x : ℚ) :
  let part1 := x
  let part2 := (1/2) * x
  let part3 := (1/3) * x
  part1 + part2 + part3 = 104
  ∧ part2 = 28 + 4/11 := by
  sorry

end middle_part_is_28_4_over_11_l584_584852


namespace parallelogram_rhombus_iff_perpendicular_diagonals_l584_584525

-- Definitions for the given conditions
variables {A B C D O : Type} [Field O]
variables (AB CD AC BD : O) -- The sides and diagonals of the parallelogram
def isParallelogram (ABCD : Type) := true -- A placeholder definition for a parallelogram

-- The main statement
theorem parallelogram_rhombus_iff_perpendicular_diagonals  {A B C D : Type} [Field O] 
  (h1: isParallelogram (A, B, C, D))
  (h2: AC ⊥ BD) : 
  isRhombus (A, B, C, D) := 
sorry

end parallelogram_rhombus_iff_perpendicular_diagonals_l584_584525


namespace median_of_81_consecutive_integers_l584_584179

theorem median_of_81_consecutive_integers (ints : ℕ → ℤ) 
  (h_consecutive: ∀ n : ℕ, ints (n+1) = ints n + 1) 
  (h_sum: (∑ i in finset.range 81, ints i) = 9^5) : 
  (ints 40) = 9^3 := 
sorry

end median_of_81_consecutive_integers_l584_584179


namespace extremum_at_x_zero_maximum_value_on_interval_no_zeros_range_l584_584822

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x - a

theorem extremum_at_x_zero (a : ℝ) (h : ∃ x, f x a = f 0 a) : a = -1 := sorry

theorem maximum_value_on_interval (a : ℝ) (h : a = -1) : ∃ x ∈ (Set.interval (-2 : ℝ) (1 : ℝ)), f x a = Real.exp (-2) + 3 := sorry

theorem no_zeros_range (a : ℝ) (h : ∀ x, f x a ≠ 0) : -Real.exp 2 < a ∧ a < 0 := sorry

end extremum_at_x_zero_maximum_value_on_interval_no_zeros_range_l584_584822


namespace find_PQ_in_triangle_l584_584006

theorem find_PQ_in_triangle
  (X Y Z D E P Q R S : Type) -- Declaring all points as types for generality
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z] -- Assuming metric spaces for the distances
  (XY : ℝ) (XZ : ℝ) (YZ : ℝ)
  (hXY : XY = 130) -- Given conditions
  (hXZ : XZ = 110)
  (hYZ : YZ = 100)
  (angle_bisector_X_D : Line X D ∧ IsAngleBisector ∠X (Segment YZ) D) -- Angle bisector of X
  (angle_bisector_Y_E : Line Y E ∧ IsAngleBisector ∠Y (Segment XZ) E) -- Angle bisector of Y
  (feet_perpendicular_Z_YE_P : PerpendicularFoot Z (Line Y E) P) -- Perpendicular foot from Z to YE
  (feet_perpendicular_Z_XD_Q : PerpendicularFoot Z (Line X D) Q) -- Perpendicular foot from Z to XD
  (extended_ZP_R : ExtendsTo R (Line Z P) (Line X Y)) -- Extending ZP to intersect XY
  (extended_ZQ_S : ExtendsTo S (Line Z Q) (Line X Y)) -- Extending ZQ to intersect XY
  (YR_eq_YZ : Distance Y R = YZ) -- YR equals YZ
  (XS_eq_XZ : Distance X S = XZ) -- XS equals XZ
  (midpoint_RP_P : Midpoint R P P) -- P as midpoint of RP
  (midpoint_SQ_Q : Midpoint S Q Q) -- Q as midpoint of SQ
  : Distance P Q = 40 := by
  sorry

end find_PQ_in_triangle_l584_584006


namespace circumcircles_tangent_at_F_l584_584694

-- Definitions of points and circle
variables {A B C D E F O : Type}
variables [points : line_segment_space A B C D E F O]
variables Γ : circle O
variables (ABC_perp: A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ ∧ angle (B - A) (C - A) > pi / 2)
variables (D : Type) (D_def: collinear A B D ∧ perp (C - A) (D - C))
variables (l : line_segment) (l_def: passes_through D l ∧ perp (O - A) l)
variables (E : Type) (E_def: meets_at E l ∧ collinear A C E)
variables (F : Type) (F_def: meets_at F Γ ∧ lies_between D E F)

-- Statement to prove
theorem circumcircles_tangent_at_F :
  tangent_at (circumcircle B F E) (circumcircle C F D) F :=
sorry

end circumcircles_tangent_at_F_l584_584694


namespace tshirt_more_expensive_l584_584846

-- Definitions based on given conditions
def jeans_price : ℕ := 30
def socks_price : ℕ := 5
def tshirt_price : ℕ := jeans_price / 2

-- Statement to prove (The t-shirt is $10 more expensive than the socks)
theorem tshirt_more_expensive : (tshirt_price - socks_price) = 10 :=
by
  rw [tshirt_price, socks_price]
  sorry  -- proof steps are omitted

end tshirt_more_expensive_l584_584846


namespace circumradius_right_triangle_l584_584282

theorem circumradius_right_triangle {a b c : ℚ} (h : a^2 + b^2 = c^2) : 
  (a = 7.5) → 
  (b = 10) → 
  (c = 12.5) → 
  ∃ R, R = c / 2 ∧ R = 25 / 4 :=
by
  intros ha hb hc
  use c / 2
  split
  exact by rw [hc, (show 12.5 = 25 / 2 by norm_cast)]
  exact by norm_cast
  sorry  -- proof remains to be filled in

end circumradius_right_triangle_l584_584282


namespace maximum_value_of_f_l584_584626

-- Let's define the function f(x)
def f (x : ℝ) : ℝ := 2 * Real.cos x + Real.sin x 

-- Now we state the theorem that the maximum value of the function f is √5.
theorem maximum_value_of_f :
  ∃ x : ℝ, ∀ y : ℝ, f(y) ≤ f(x) ∧ f(x) = Real.sqrt 5 :=
sorry

end maximum_value_of_f_l584_584626


namespace simplify_sqrt_sum_l584_584100

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l584_584100


namespace probability_of_ascension_l584_584381

noncomputable def P : ℤ × ℤ → ℚ
| (6 * m, 6 * n) := 1
| (6 * m + 3, 6 * n + 3) := 0
| (m, n) := (P (m - 1, n) + P (m + 1, n) + P (m, n - 1) + P (m, n + 1)) / 4

theorem probability_of_ascension : P (1, 1) = 13 / 22 := sorry

end probability_of_ascension_l584_584381


namespace ordinate_of_P_l584_584453

noncomputable def hyperbola := { p : ℝ × ℝ | p.1^2 - p.2^2 / 8 = 1 }

def A : ℝ × ℝ := (0, 6 * Real.sqrt 6)

def right_focus : ℝ × ℝ := (3, 0)
def left_focus : ℝ × ℝ := (-3, 0)

def is_point_on_left_branch (P : ℝ × ℝ) : Prop :=
  P ∈ hyperbola ∧ P.1 < 0

def minimized_perimeter (P : ℝ × ℝ) : Prop :=
  is_point_on_left_branch P ∧
  (let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2),
       PF := Real.sqrt ((P.1 - right_focus.1)^2 + (P.2 - right_focus.2)^2),
       perimeter := PA + PF + 15 in
   (∀ Q, is_point_on_left_branch Q → (let QA := Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2),
                                          QF := Real.sqrt ((Q.1 - right_focus.1)^2 + (Q.2 - right_focus.2)^2),
                                          Q_perimeter := QA + QF + 15 in
                                      perimeter ≤ Q_perimeter)))

theorem ordinate_of_P (P : ℝ × ℝ) (h : minimized_perimeter P) : P.2 = 2 * Real.sqrt 6 :=
sorry

end ordinate_of_P_l584_584453


namespace necessary_not_sufficient_l584_584696

theorem necessary_not_sufficient (a b c d : ℝ) : 
  (a + c > b + d) → (a > b ∧ c > d) :=
sorry

end necessary_not_sufficient_l584_584696


namespace diagonals_of_prism_not_possible_l584_584680

theorem diagonals_of_prism_not_possible (d1 d2 d3 : ℕ) (h : {d1, d2, d3} = {6, 8, 11}) : 
¬(∃ (a b c : ℕ), 
  {d1, d2, d3} = {Int.sqrt (a^2 + b^2), Int.sqrt (b^2 + c^2), Int.sqrt (a^2 + c^2)}) := 
by 
  sorry

end diagonals_of_prism_not_possible_l584_584680


namespace min_dot_product_l584_584801

theorem min_dot_product (m n : ℝ) (x1 x2 : ℝ)
    (h1 : m ≠ 0) 
    (h2 : n ≠ 0)
    (h3 : (x1 + 2) * (x2 - x1) + m * x1 * (n - m * x1) = 0) :
    ∃ (x1 : ℝ), (x1 = -2 / (m^2 + 1)) → 
    (x1 + 2) * (x2 + 2) + m * n * x1 = 4 * m^2 / (m^2 + 1) := 
sorry

end min_dot_product_l584_584801


namespace expected_sum_of_rolls_l584_584291

def fair_die : ℙ (Sum int → int) := 
sorry

noncomputable def expected_rolls_to_2010 (die: ℙ (Sum int → int)) : ℝ :=
sorry

theorem expected_sum_of_rolls :
  expected_rolls_to_2010(fair_die) = 574.5238095 :=
sorry

end expected_sum_of_rolls_l584_584291


namespace difference_in_widgets_l584_584954

-- Define the conditions as hypotheses
variables {t w : ℝ}
hypothesis (h1 : w = 3 * t - 1)

-- Define the difference in widgets produced between Monday and Tuesday
def widget_diff : ℝ := (w * t) - ((w + 6) * (t - 3))

-- State the theorem that needs to be proven
theorem difference_in_widgets (ht : t : ℝ) (hw : w : ℝ) (h1 : w = 3 * t - 1) : widget_diff = 3 * t + 15 :=
by
  sorry

end difference_in_widgets_l584_584954


namespace cows_in_herd_l584_584293

theorem cows_in_herd (n : ℕ) (h1 : n / 3 + n / 6 + n / 7 < n) (h2 : 15 = n * 5 / 14) : n = 42 :=
sorry

end cows_in_herd_l584_584293


namespace jill_water_stored_l584_584011

theorem jill_water_stored (n : ℕ) (h : n = 24) : 
  8 * (1 / 4 : ℝ) + 8 * (1 / 2 : ℝ) + 8 * 1 = 14 :=
by
  sorry

end jill_water_stored_l584_584011


namespace set_operation_proof_l584_584397

def SetOp (M N : Set ℤ) : Set ℤ := { x | x ∈ M ∨ x ∈ N, x ∉ M ∩ N }

def M : Set ℤ := {0, 2, 4, 6, 8, 10}
def N : Set ℤ := {0, 3, 6, 9, 12, 15}

theorem set_operation_proof : SetOp (SetOp M N) M = N := 
by
  sorry

end set_operation_proof_l584_584397


namespace quarterback_steps_back_l584_584310

variable (T : ℕ)

/-- A quarterback steps back to throw some times in a game.
30 percent of the time he does not get a pass thrown.
Half of the times that he does not throw the ball he is sacked for a loss.
The quarterback is sacked for a loss 12 times in the game.
Prove that the quarterback stepped back to throw 80 times in the game.
-/
theorem quarterback_steps_back (hT_non_throws : 0.30 * T ∈ ℤ)
  (h_sacked_loss : (1/2) * (0.30 * T) = 12) : T = 80 :=
sorry

end quarterback_steps_back_l584_584310


namespace fraction_value_l584_584675

theorem fraction_value : (2020 / (20 * 20 : ℝ)) = 5.05 := by
  sorry

end fraction_value_l584_584675


namespace unique_prime_pair_l584_584246

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_pair :
  ∀ p : ℕ, is_prime p ∧ is_prime (p + 1) → p = 2 := by
  sorry

end unique_prime_pair_l584_584246


namespace smallest_angle_convex_15_polygon_l584_584984

theorem smallest_angle_convex_15_polygon :
  ∃ (a : ℕ) (d : ℕ), (∀ n : ℕ, n ∈ Finset.range 15 → (a + n * d < 180)) ∧
  15 * (a + 7 * d) = 2340 ∧ 15 * d <= 24 -> a = 135 :=
by
  -- Proof omitted
  sorry

end smallest_angle_convex_15_polygon_l584_584984


namespace shorter_side_of_rectangle_is_twelve_cm_l584_584706

theorem shorter_side_of_rectangle_is_twelve_cm 
  (radius : ℝ) (h_radius : radius = 6)
  (area_rectangle triple_circle_area : ℝ)
  (h_circle_area : area_rectangle = 3 * (Real.pi * radius ^ 2))
  (h_tangent_conditions : ∀ (l w : ℝ), (l < w) → (area_rectangle = l * w) → (w = 2 * radius)) :
  let s := 2 * radius in
  s = 12 :=
by
  sorry

end shorter_side_of_rectangle_is_twelve_cm_l584_584706


namespace inequality_solution_set_l584_584830

noncomputable def solution_set (a b : ℝ) := {x : ℝ | 2 < x ∧ x < 3}

theorem inequality_solution_set (a b : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → (ax^2 + 5 * x + b > 0)) →
  (∀ x : ℝ, (-6) * x^2 - 5 * x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by
  sorry

end inequality_solution_set_l584_584830


namespace coefficient_x3y3_in_expansion_l584_584768

theorem coefficient_x3y3_in_expansion :
  coefficient (x^3 * y^3) (expand (x + 2 * y) ((2 * x - y)^5)) = 120 :=
by sorry

end coefficient_x3y3_in_expansion_l584_584768


namespace seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_l584_584692

-- Problem 1
theorem seven_divides_n_iff_seven_divides_q_minus_2r (n q r : ℕ) (h : n = 10 * q + r) :
  (7 ∣ n) ↔ (7 ∣ (q - 2 * r)) := sorry

-- Problem 2
theorem seven_divides_2023 : 7 ∣ 2023 :=
  let q := 202
  let r := 3
  have h : 2023 = 10 * q + r := by norm_num
  have h1 : (7 ∣ 2023) ↔ (7 ∣ (q - 2 * r)) :=
    seven_divides_n_iff_seven_divides_q_minus_2r 2023 q r h
  sorry -- Here you would use h1 and prove the statement using it

-- Problem 3
theorem thirteen_divides_n_iff_thirteen_divides_q_plus_4r (n q r : ℕ) (h : n = 10 * q + r) :
  (13 ∣ n) ↔ (13 ∣ (q + 4 * r)) := sorry

end seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_l584_584692


namespace greatest_value_of_sum_l584_584247

theorem greatest_value_of_sum (x y : ℝ) (h₁ : x^2 + y^2 = 100) (h₂ : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
by
  sorry

end greatest_value_of_sum_l584_584247


namespace maze_exit_probabilities_l584_584297

def maze_probabilities : Prop :=
let prob_exit_1_hour := 1 / 3 in
let prob_exit_exceeds_3_hours := 1 / 2 in
(prob_exit_1_hour = 1 / 3) ∧ (prob_exit_exceeds_3_hours = 1 / 2)

theorem maze_exit_probabilities :
  maze_probabilities :=
by
  unfold maze_probabilities
  constructor
  · simp
  · simp
  sorry

end maze_exit_probabilities_l584_584297


namespace simplify_sqrt_72_plus_sqrt_32_l584_584091

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l584_584091


namespace circle_area_l584_584343

-- Define the conditions of the problem
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 9 = 0

-- State the proof problem
theorem circle_area : (∀ (x y : ℝ), circle_equation x y) → (∀ r : ℝ, r = 2 → π * r^2 = 4 * π) :=
by
  sorry

end circle_area_l584_584343


namespace hyperbola_equation_l584_584828

-- Definition of hyperbola C with parameters a and b
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

-- Definitions of given conditions
def e : ℝ := sqrt 10 / 2
def point_on_hyperbola : ℝ × ℝ := (2, sqrt 3)

-- The main theorem to prove
theorem hyperbola_equation (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
    (h_eccentricity : (sqrt 10 / 2) = sqrt ((a^2 + b^2) / a^2))
    (h_point : hyperbola a b (2) (sqrt 3)) :
    hyperbola (sqrt 2) (sqrt 3) :=
by {
  sorry
}

end hyperbola_equation_l584_584828


namespace pencil_distribution_l584_584780

open_locale big_operators

theorem pencil_distribution : 
  ∃ (ways : ℕ), ways = 58 ∧ ∃ (a b c d : ℕ), a + b + c + d = 10 ∧ a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ d ≥ 1 :=
by 
  sorry

end pencil_distribution_l584_584780


namespace tim_income_percent_less_than_juan_l584_584570

theorem tim_income_percent_less_than_juan (T M J : ℝ) (h1 : M = 1.5 * T) (h2 : M = 0.9 * J) :
  (J - T) / J = 0.4 :=
by
  sorry

end tim_income_percent_less_than_juan_l584_584570


namespace sin_60_proof_l584_584370

noncomputable def sin_60_eq : Prop :=
  sin (60 * real.pi / 180) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq :=
sorry

end sin_60_proof_l584_584370


namespace distinct_real_solutions_l584_584141

theorem distinct_real_solutions : ∃! (x : Set ℝ), abs (x - abs (3 * x - 2)) = 5 := by
  sorry

end distinct_real_solutions_l584_584141


namespace sin_60_proof_l584_584376

noncomputable def sin_60_eq_sqrt3_div_2 : Prop :=
  Real.sin (π / 3) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq_sqrt3_div_2 :=
sorry

end sin_60_proof_l584_584376


namespace ratio_of_AB_to_AD_l584_584969

theorem ratio_of_AB_to_AD :
  ∀ (ABCD EFGH : Type) (s w h : ℝ),
  let area_square := s^2,
      area_overlap_square := 0.25 * s^2,
      area_rectangle := w * h,
      area_overlap_rectangle := 0.4 * w * h,
      AB := w, AD := h in
  area_overlap_square = area_overlap_rectangle →
  h = s / 4 →
  AB / AD = 10 := by
  intros ABCD EFGH s w h area_square area_overlap_square area_rectangle area_overlap_rectangle AB AD
  intro h_eq
  intro h_def
  sorry

end ratio_of_AB_to_AD_l584_584969


namespace triangle_incircle_concurrency_l584_584520

theorem triangle_incircle_concurrency {ABC : Triangle} {I : Point}
  (incircle_tangent : incircle_centered_at_I I ABC)
  (tangent_D : tangent_to_side I ABC BC at D)
  (tangent_E : tangent_to_side I ABC AC at E)
  (tangent_F : tangent_to_side I ABC AB at F)
  (feet_X : perpendicular_from A to_line l = X)
  (feet_Y : perpendicular_from B to_line l = Y)
  (feet_Z : perpendicular_from C to_line l = Z)
  (line_through_I : passes_through_line l I) :
  concurrent (lines_through [D, X, EY]) (lines_through [E, Y, EY]) (lines_through [F, Z, FZ]) := 
sorry

end triangle_incircle_concurrency_l584_584520


namespace probability_three_unused_rockets_expected_targets_hit_l584_584729

section RocketArtillery

variables (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1)

-- Probability that exactly three unused rockets remain after firing at five targets
theorem probability_three_unused_rockets (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let prob := 10 * (p ^ 3) * ((1 - p) ^ 2) in
  prob = 10 * (p ^ 3) * ((1 - p) ^ 2) := 
by
  sorry

-- Expected number of targets hit if there are nine targets
theorem expected_targets_hit (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let expected_hits := 10 * p - (p ^ 10) in
  expected_hits = 10 * p - (p ^ 10) := 
by
  sorry

end RocketArtillery

end probability_three_unused_rockets_expected_targets_hit_l584_584729


namespace find_DG_l584_584523

-- Define the points O, A, B, C
variable (O A B C : Type)
variable [AddGroup O] [AddGroup A] [AddGroup B] [AddGroup C]
variable [Module ℝ O] [Module ℝ A] [Module ℝ B] [Module ℝ C]

-- Midpoint definition
def midpoint (p q : O) : O := (p + q) / 2

-- Centroid definition for triangle OBC
def centroid (O B C : O) : O := (O + B + C) / 3

-- Given conditions and proof goal
theorem find_DG
  (D : O) (H1 : D = midpoint A B)
  (G : O) (H2 : G = centroid O B C) :
  (G - D) = - (1/2 : ℝ) • A - (1/6 : ℝ) • B + (1/3 : ℝ) • C :=
sorry

end find_DG_l584_584523


namespace sqrt_sum_simplify_l584_584122

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l584_584122


namespace max_kings_on_chessboard_no_attack_l584_584249

def king_moves : ℤ × ℤ → set (ℤ × ℤ)
| (x, y) := { (x', y') | abs (x - x') ≤ 1 ∧ abs (y - y') ≤ 1 ∧ (x, y) ≠ (x', y') }

theorem max_kings_on_chessboard_no_attack (kings : set (ℤ × ℤ)) :
  (∀ (x, y) ∈ kings, ∀ (x', y') ∈ kings, (x, y) ≠ (x', y') → (x', y') ∉ king_moves (x, y)) →
  kings.card ≤ 16 := sorry

end max_kings_on_chessboard_no_attack_l584_584249


namespace part_a_part_b_l584_584939

theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z ≥ 3) :
  ¬ (1/x + 1/y + 1/z ≤ 3) :=
sorry

theorem part_b (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z ≤ 3) :
  1/x + 1/y + 1/z ≥ 3 :=
sorry

end part_a_part_b_l584_584939


namespace arithmetic_geometric_seq_proof_l584_584804

theorem arithmetic_geometric_seq_proof
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : a1 - a2 = -1)
  (h2 : 1 * (b2 * b2) = 4)
  (h3 : b2 > 0) :
  (a1 - a2) / b2 = -1 / 2 :=
by
  sorry

end arithmetic_geometric_seq_proof_l584_584804


namespace distance_after_12_seconds_time_to_travel_380_meters_l584_584621

def distance_travelled (t : ℝ) : ℝ := 9 * t + 0.5 * t^2

theorem distance_after_12_seconds : distance_travelled 12 = 180 :=
by 
  sorry

theorem time_to_travel_380_meters : ∃ t : ℝ, distance_travelled t = 380 ∧ t = 20 :=
by 
  sorry

end distance_after_12_seconds_time_to_travel_380_meters_l584_584621


namespace square_division_l584_584592

theorem square_division (n : ℕ) (h : n ≥ 6) :
  ∃ (sq_div : ℕ → Prop), sq_div 6 ∧ (∀ n, sq_div n → sq_div (n + 3)) :=
by
  sorry

end square_division_l584_584592


namespace mass_percentage_C_in_Butanoic_acid_approx_54_51_l584_584248

structure ChemicalElement :=
  (symbol : String)
  (atomic_mass : Float)

def C : ChemicalElement := { symbol := "C", atomic_mass := 12.01 }
def H : ChemicalElement := { symbol := "H", atomic_mass := 1.008 }
def O : ChemicalElement := { symbol := "O", atomic_mass := 16.00 }

structure Molecule :=
  (formula : List (ChemicalElement × Nat))

def Butanoic_acid : Molecule := { formula := [(C, 4), (H, 8), (O, 2)] }

noncomputable def molar_mass (m : Molecule) : Float :=
  m.formula.foldr (λ (elem_count : ChemicalElement × Nat) acc => acc + elem_count.fst.atomic_mass * elem_count.snd) 0

noncomputable def mass_percentage_of_element (elem : ChemicalElement) (m : Molecule) : Float :=
  let total_mass := molar_mass m
  let elem_mass := (m.formula.filter (λ p => p.fst = elem)).foldr (λ (e_c : ChemicalElement × Nat) acc => acc + e_c.fst.atomic_mass * e_c.snd) 0
  (elem_mass / total_mass) * 100

-- Prove that the mass percentage of Carbon in Butanoic acid is approximately 54.51%
theorem mass_percentage_C_in_Butanoic_acid_approx_54_51 :
  mass_percentage_of_element C Butanoic_acid ≈ 54.51 := 
sorry

end mass_percentage_C_in_Butanoic_acid_approx_54_51_l584_584248


namespace smallest_perimeter_triangle_l584_584238

theorem smallest_perimeter_triangle (PQ PR QR : ℕ) (J : Point) :
  PQ = PR →
  QJ = 10 →
  QR = 2 * 10 →
  PQ + PR + QR = 40 :=
by
  sorry

structure Point : Type :=
mk :: (QJ : ℕ)

noncomputable def smallest_perimeter_triangle : Prop :=
  ∃ (PQ PR QR : ℕ) (J : Point), PQ = PR ∧ J.QJ = 10 ∧ QR = 2 * 10 ∧ PQ + PR + QR = 40

end smallest_perimeter_triangle_l584_584238


namespace smallest_perimeter_of_triangle_PQR_l584_584225

noncomputable def triangle_PQR_perimeter (PQ PR QR : ℕ) (QJ : ℝ) 
  (h1 : PQ = PR) (h2 : QJ = 10) : ℕ :=
2 * (PQ + QR)

theorem smallest_perimeter_of_triangle_PQR (PQ PR QR : ℕ) (QJ : ℝ) :
  PQ = PR → QJ = 10 → 
  ∃ p, p = triangle_PQR_perimeter PQ PR QR QJ (by assumption) (by assumption) ∧ p = 78 :=
sorry

end smallest_perimeter_of_triangle_PQR_l584_584225


namespace sqrt_sum_l584_584115

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l584_584115


namespace x_squared_minus_y_squared_l584_584861

theorem x_squared_minus_y_squared :
  ∀ (x y : ℚ), x + y = 9 / 17 ∧ x - y = 1 / 119 → x^2 - y^2 = 9 / 2003 :=
by
  intros x y h
  cases h with h1 h2
  sorry

end x_squared_minus_y_squared_l584_584861


namespace cafe_open_days_from_April_1_to_April_27_l584_584613

theorem cafe_open_days_from_April_1_to_April_27 :
  (∀ n : ℕ, (1 ≤ n ∧ n ≤ 27) → ¬ (n % 7 = 1)) → 
  (∑ n in finset.range 20, if 1 ≤ n + 1 ∧ (n + 1) % 7 ≠ 1 then 1 else 0 = 18 →
  ∑ n in finset.range 21, if 10 ≤ n + 10 ∧ (n + 10) % 7 ≠ 1 then 1 else 0 = 18 →
  ∑ n in finset.range 27, if 1 ≤ n + 1 ∧ (n + 1) % 7 ≠ 1 then 1 else 0 = 23) :=
begin
  sorry
end

end cafe_open_days_from_April_1_to_April_27_l584_584613


namespace tan_600_eq_sqrt_3_l584_584641

theorem tan_600_eq_sqrt_3 : tan (600 * pi / 180) = sqrt 3 :=
by
  have h1 : 600 * pi / 180 = 3 * pi + pi / 3 := by norm_num
  rw [h1, tan_add_pi]
  exact tan_pi_div_three_eq (_ : pi/3 = 60 * pi / 180)

end tan_600_eq_sqrt_3_l584_584641


namespace jamal_green_marbles_l584_584532

theorem jamal_green_marbles
  (Y B K T : ℕ)
  (hY : Y = 12)
  (hB : B = 10)
  (hK : K = 1)
  (h_total : 1 / T = 1 / 28) :
  T - (Y + B + K) = 5 :=
by
  -- sorry, proof goes here
  sorry

end jamal_green_marbles_l584_584532


namespace min_a_l584_584829

theorem min_a (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → (x + y) * (1/x + a/y) ≥ 25) : a ≥ 16 :=
sorry  -- Proof is omitted

end min_a_l584_584829


namespace determine_number_of_20_pound_boxes_l584_584519

variable (numBoxes : ℕ) (avgWeight : ℕ) (x : ℕ) (y : ℕ)

theorem determine_number_of_20_pound_boxes 
  (h1 : numBoxes = 30) 
  (h2 : avgWeight = 18) 
  (h3 : x + y = 30) 
  (h4 : 10 * x + 20 * y = 540) : 
  y = 24 :=
  by
  sorry

end determine_number_of_20_pound_boxes_l584_584519


namespace find_m_l584_584620

theorem find_m : ∃ m : ℝ, 
  (∀ x : ℝ, 0 < x → (m^2 - m - 5) * x^(m - 1) = (m^2 - m - 5) * x^(m - 1) ∧ 
  (m^2 - m - 5) * (m - 1) * x^(m - 2) > 0) → m = 3 :=
by
  sorry

end find_m_l584_584620


namespace periodic_sequence_from_rational_l584_584945

theorem periodic_sequence_from_rational
  (a : ℕ → ℚ)
  (h₁ : ∀ n, abs (a (n + 1) - 2 * a n) = 2)
  (h₂ : ∀ n, abs (a n) ≤ 2)
  (h₀ : ∀ n, a 1 ∈ ℚ) :
  ∃ N, ∀ k, a (N + k) = a (N + k % period) :=
sorry

end periodic_sequence_from_rational_l584_584945


namespace moles_C2H6_required_l584_584847

theorem moles_C2H6_required
  (n_Cl2 : ℕ)
  (n_C2H4Cl2 : ℕ)
  (n_HCl : ℕ)
  (h_balance : ∀ (x : ℕ), (x : ℕ) ∗ 1 (ethane) + x * 2 (chlorine) = x * 1 (chlorinated_ethane) + x * 2 (HCl))
  (h_given_Cl2 : n_Cl2 = 6)
  (h_ans_Cl2 : (n_C2H4Cl2 ∗ 1) / 2 = 3)
  (h_required_Cl2 : n_HCl = 6) :
  (n_C2H6 : ℕ) = 3 :=
by
  apply sorry

end moles_C2H6_required_l584_584847


namespace good_numbers_not_good_number_l584_584501

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n),
    ∀ k : Fin n, is_perfect_square (k + 1 + a k + 1)

theorem good_numbers :
  ∀ n : ℕ, 
    n ∈ {13, 15, 17, 19} ↔ is_good_number n :=
begin
  intro n,
  cases n,
  apply Iff.intro,
  { intro h,
    repeat
      { cases h } <|>
      { cases h } <|>
      { cases h with _ h },
    all_goals {exact true_intro} },
  
  { intro h,
    cases h with a ha,
    sorry }
end

theorem not_good_number :
  ¬ is_good_number 11 :=
begin
  intro h,
  cases h with a ha,
  sorry
end

end good_numbers_not_good_number_l584_584501


namespace angle_sum_l584_584000

-- Define the angles P, Q, R
def angleP : ℝ := 34
def angleQ : ℝ := 80
def angleR : ℝ := 30

-- Define the angles a and b
def a : ℝ
def b : ℝ

-- The theorem we want to prove
theorem angle_sum (a b : ℝ) (angleP angleQ angleR : ℝ):
  angleP = 34 → angleQ = 80 → angleR = 30 → a + b = 144 := 
by
  intros hP hQ hR
  sorry

end angle_sum_l584_584000


namespace sin_60_proof_l584_584375

noncomputable def sin_60_eq_sqrt3_div_2 : Prop :=
  Real.sin (π / 3) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq_sqrt3_div_2 :=
sorry

end sin_60_proof_l584_584375


namespace count_valid_n_l584_584760

theorem count_valid_n :
  ∃ (count : ℕ), count = 
    (finset.filter (λ (n : ℕ), (n ≠ 0 ∧ n ≤ 1200 ∧
                   ( (floor (1197 / n) + floor (1198 / n) + floor (1199 / n) + floor (1200 / n)) % 4 ≠ 0)
              ))
  (finset.range 1201)).card
  ∧ count = 18 :=
begin
  sorry
end

end count_valid_n_l584_584760


namespace sin_60_eq_sqrt3_div_2_l584_584355

theorem sin_60_eq_sqrt3_div_2 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  -- proof skipped
  sorry

end sin_60_eq_sqrt3_div_2_l584_584355


namespace polygon_perimeter_eq_21_l584_584306

-- Definitions and conditions from the given problem
def rectangle_side_a := 6
def rectangle_side_b := 4
def triangle_hypotenuse := 5

-- The combined polygon perimeter proof statement
theorem polygon_perimeter_eq_21 :
  let rectangle_perimeter := 2 * (rectangle_side_a + rectangle_side_b)
  let adjusted_perimeter := rectangle_perimeter - rectangle_side_b + triangle_hypotenuse
  adjusted_perimeter = 21 :=
by 
  -- Skip the proof part by adding sorry
  sorry

end polygon_perimeter_eq_21_l584_584306


namespace smallest_perimeter_of_triangle_PQR_l584_584224

noncomputable def triangle_PQR_perimeter (PQ PR QR : ℕ) (QJ : ℝ) 
  (h1 : PQ = PR) (h2 : QJ = 10) : ℕ :=
2 * (PQ + QR)

theorem smallest_perimeter_of_triangle_PQR (PQ PR QR : ℕ) (QJ : ℝ) :
  PQ = PR → QJ = 10 → 
  ∃ p, p = triangle_PQR_perimeter PQ PR QR QJ (by assumption) (by assumption) ∧ p = 78 :=
sorry

end smallest_perimeter_of_triangle_PQR_l584_584224


namespace inversion_circumcircle_is_circle_through_midpoints_equation_satisfied_l584_584751

-- Definitions and given conditions
structure BicentricQuadrilateral (R r d : ℝ) :=
  (A B C D : ℝ) -- Vertices of quadrilateral
  (I O : ℝ) -- Centers of incircle and circumcircle
  (R_pos : R > 0) -- Radius of circumcircle
  (r_pos : r > 0) -- Radius of incircle
  (d_pos : d > 0) -- Distance between the centers I and O
  -- more conditions can be added here as required
  (incenter_properties : True) -- This could be expanded into specific properties about the intersection points with sides AB, BC, CD, DA at E, F, G, H respectively.
  (diagonals_intersect_at_K : True) -- Additional intersection properties
  
theorem inversion_circumcircle_is_circle_through_midpoints (R r d : ℝ) (BQ : BicentricQuadrilateral R r d) :
  True :=
sorry

theorem equation_satisfied (R r d : ℝ) (BQ : BicentricQuadrilateral R r d) :
  \(\frac{1}{(R+d)^2} + \frac{1}{(R-d)^2} = \frac{1}{r^2}\) :=
sorry

end inversion_circumcircle_is_circle_through_midpoints_equation_satisfied_l584_584751


namespace simplify_sqrt72_add_sqrt32_l584_584124

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l584_584124


namespace find_y_l584_584490

theorem find_y (y : ℕ) (h1 : 27 = 3^3) (h2 : 3^9 = 27^y) : y = 3 := 
by 
  sorry

end find_y_l584_584490


namespace num_marbles_removed_l584_584279

theorem num_marbles_removed (total_marbles red_marbles : ℕ) (prob_neither_red : ℚ) 
  (h₁ : total_marbles = 84) (h₂ : red_marbles = 12) (h₃ : prob_neither_red = 36 / 49) : 
  total_marbles - red_marbles = 2 :=
by
  sorry

end num_marbles_removed_l584_584279


namespace max_marks_correct_l584_584320

-- Define the conditions
def pass_percentage : ℝ := 0.65
def student_marks : ℝ := 250
def failure_margin : ℝ := 45

-- Define the maximum_marks to be proved
def maximum_marks : ℝ := 454

-- Lean statement to prove the maximum marks
theorem max_marks_correct :
  let required_marks := student_marks + failure_margin in
  required_marks = pass_percentage * maximum_marks → maximum_marks = 454 :=
by
  sorry

end max_marks_correct_l584_584320


namespace sqrt_c_is_202_l584_584543

theorem sqrt_c_is_202 (a b c : ℝ) (h1 : a + b = -2020) (h2 : a * b = c) (h3 : a / b + b / a = 98) : 
  Real.sqrt c = 202 :=
by
  sorry

end sqrt_c_is_202_l584_584543


namespace pipes_empty_cistern_l584_584959

theorem pipes_empty_cistern :
  (∀ C : ℝ, C > 0) → 
  ((3 / 4) * C / 12 = 1 / (16 : ℝ)) →
  ((1 / 2) * C / 15 = 1 / (30 : ℝ)) →
  ((1 / 3) * C / 10 = 1 / (30 : ℝ)) →
  ∀ t : ℝ, t = 8 →
  (C / 16 + C / 30 + C / 30) * t >= C :=
begin
  intros hC hA hB hC' ht,
  have H : C / 16 + C / 30 + C / 30 = 31 * C / 240,
  { sorry },
  rw ht at *,
  change C * (31 / 240) * 8 with (31 / 30) * C,
  have : (31 / 30) * C >= C,
  { sorry },
  exact this,
end

end pipes_empty_cistern_l584_584959


namespace max_value_reached_max_value_exists_l584_584034

theorem max_value_reached (x y z : ℝ) (h : x + 3 * y + z = 6) :
  (2 * x * y + x * z + y * z) ≤ 4 :=
sorry

-- proof step
theorem max_value_exists (x y z : ℝ) (h : x + 3 * y + z = 6) :
  ∃ x y z : ℝ, x + 3 * y + z = 6 ∧ (2 * x * y + x * z + y * z) = 4 :=
begin
  use [3, 0, 3],
  split,
  { norm_num },
  { norm_num },
end

end max_value_reached_max_value_exists_l584_584034


namespace median_of_81_consecutive_integers_l584_584201

theorem median_of_81_consecutive_integers (s : ℕ) (h_sum : s = 9^5) : 
  let n := 81 in
  let median := s / n in
  median = 729 :=
by
  have h₁ : 9^5 = 59049 := by norm_num
  have h₂ : 81 = 81 := rfl
  have h₃ : 59049 / 81 = 729 := by norm_num
  
  -- Apply the conditions
  rw [h_sum, <-h₁] at h_sum
  rw h₂
  
  -- Conclude the median
  exact h₃

end median_of_81_consecutive_integers_l584_584201


namespace max_points_guaranteed_after_2014_steps_l584_584513

theorem max_points_guaranteed_after_2014_steps :
  ∀ (k: ℕ), 2 * k = 2014 → (1007 * 1007 = 1014049) :=
by
  intros k h
  have h1 : k = 1007 := by linarith
  rw h1
  norm_num
  sorry

end max_points_guaranteed_after_2014_steps_l584_584513


namespace smallest_integer_l584_584652

theorem smallest_integer {x y z : ℕ} (h1 : 2*y = x) (h2 : 3*y = z) (h3 : x + y + z = 60) : y = 6 :=
by
  sorry

end smallest_integer_l584_584652


namespace inverse_of_matrix_A_l584_584771

open Matrix

def matrix_A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 7], ![2, 3]]

def matrix_A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![(-3 : ℚ) / 2, (7 : ℚ) / 2], ![1, -2]]

theorem inverse_of_matrix_A : inverse matrix_A = matrix_A_inv :=
by
  -- Proof goes here
  sorry

end inverse_of_matrix_A_l584_584771


namespace solution_a_solution_b_solution_c_l584_584606

/- 
  Problem Definition:
  Given x is a real number and √(x + √(2x - 1)) + √(x - √(2x - 1)) = y, find the set of x that satisfies the following:
  (a) y = √2
  (b) y = 1
  (c) y = 2
-/

noncomputable def problem_a (x : ℝ) : Prop :=
  x ≥ 1 / 2 ∧ (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = sqrt 2)

noncomputable def problem_b (x : ℝ) : Prop :=
  x ≥ 1 / 2 ∧ (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 1)

noncomputable def problem_c (x : ℝ) : Prop :=
  x ≥ 1 / 2 ∧ (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 2)

theorem solution_a (x : ℝ) : problem_a x ↔ (1/2 ≤ x ∧ x ≤ 1) := 
sorry

theorem solution_b (x : ℝ) : ¬ problem_b x := 
sorry

theorem solution_c (x : ℝ) : problem_c x ↔ x = 3 / 2 := 
sorry

end solution_a_solution_b_solution_c_l584_584606


namespace complex_modulus_l584_584810

theorem complex_modulus (a b : ℝ) (h : (complex.of_real a + complex.I) * (1 - complex.I) = 3 + b * complex.I) :
  complex.abs (complex.of_real a + b * complex.I) = real.sqrt 5 :=
sorry

end complex_modulus_l584_584810


namespace minimum_omega_l584_584821

theorem minimum_omega (ω : ℝ) (k : ℤ) (hω : ω > 0) 
  (h_symmetry : ∃ k : ℤ, ω * (π / 12) + π / 6 = k * π + π / 2) : ω = 4 :=
sorry

end minimum_omega_l584_584821


namespace sqrt_sum_l584_584110

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l584_584110


namespace no_finite_subset_y_finite_subset_y_same_size_l584_584145

variable {α : Type*} [fintype α] [decidable_eq α]

-- Definitions for condition part (a)
def family_of_fin_sets_1 (F : set (set ℕ)) : Prop :=
  ∀ A B ∈ F, (A ∩ B ≠ ∅)

-- Definitions for condition part (a)
def family_of_fin_sets_2 (F : set (set ℕ)) : Prop :=
  ∀ A B ∈ F, A.card = B.card ∧ (A ∩ B ≠ ∅)

-- Theorem corresponding to (a)
theorem no_finite_subset_y (F : set (set ℕ)) (h₁ : family_of_fin_sets_1 F) :
  ¬ (∃ Y : set ℕ, Y.finite ∧ (∀ A B ∈ F, (A ∩ B ∩ Y ≠ ∅))) :=
sorry

-- Theorem corresponding to (b)
theorem finite_subset_y_same_size (F : set (set ℕ)) (h₂ : family_of_fin_sets_2 F) :
  ∃ Y : set ℕ, Y.finite ∧ (∀ A B ∈ F, (A ∩ B ∩ Y ≠ ∅)) :=
sorry

end no_finite_subset_y_finite_subset_y_same_size_l584_584145


namespace intersection_with_y_axis_l584_584329

-- Define the original linear function
def original_function (x : ℝ) : ℝ := -2 * x + 3

-- Define the function after moving it up by 2 units
def moved_up_function (x : ℝ) : ℝ := original_function x + 2

-- State the theorem to prove the intersection with the y-axis
theorem intersection_with_y_axis : moved_up_function 0 = 5 :=
by
  sorry

end intersection_with_y_axis_l584_584329


namespace sum_divisible_by_2_l584_584775

open Nat

theorem sum_divisible_by_2 (n : ℕ) :
  let s_n := ∑ k in Finset.range (n / 2 + 1), 1973^k * Nat.choose n (2*k + 1)
  ∃ m : ℕ, s_n = 2^(n-1) * m := by
  sorry

end sum_divisible_by_2_l584_584775


namespace final_result_l584_584778

noncomputable def f : ℝ → ℝ := sorry

axiom condition_1 : ∀ (x : ℝ), f(x + 2) - f(x) = 2 * f(1)
axiom condition_2 : ∀ (x : ℝ), f(x - 1) = (f(2 - x) : ℝ)  -- Symmetric about x = 1
axiom condition_3 : f(0) = 2

theorem final_result : f(2015) + f(2016) = 2 :=
by
  sorry

end final_result_l584_584778


namespace billy_points_l584_584340

theorem billy_points (B : ℤ) (h : B - 9 = 2) : B = 11 := 
by 
  sorry

end billy_points_l584_584340


namespace seashells_ratio_l584_584050

theorem seashells_ratio (s_1 s_2 S t s3 : ℕ) (hs1 : s_1 = 5) (hs2 : s_2 = 7) (hS : S = 36)
  (ht : t = s_1 + s_2) (hs3 : s3 = S - t) :
  s3 / t = 2 :=
by
  rw [hs1, hs2] at ht
  simp at ht
  rw [hS, ht] at hs3
  simp at hs3
  sorry

end seashells_ratio_l584_584050


namespace simplify_sqrt72_add_sqrt32_l584_584125

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l584_584125


namespace graph_intersects_x_axis_l584_584753

noncomputable def f (x : ℝ) : ℝ := log 3 (x - 2) + 1

theorem graph_intersects_x_axis :
  ∃ x : ℝ, f x = 0 :=
by
  use 7/3
  simp [f]
  rw [log_inj (by linarith) (by norm_num : 0 < 3)]
  norm_num
  sorry

end graph_intersects_x_axis_l584_584753


namespace owen_wins_with_n_bullseyes_l584_584337

-- Define the parameters and conditions
def initial_score_lead : ℕ := 60
def total_shots : ℕ := 120
def bullseye_points : ℕ := 9
def minimum_points_per_shot : ℕ := 3
def max_points_per_shot : ℕ := 9
def n : ℕ := 111

-- Define the condition for Owen's winning requirement
theorem owen_wins_with_n_bullseyes :
  6 * 111 + 360 > 1020 :=
by
  sorry

end owen_wins_with_n_bullseyes_l584_584337


namespace second_coloring_book_pictures_l584_584597

-- Let P1 be the number of pictures in the first coloring book.
def P1 := 23

-- Let P2 be the number of pictures in the second coloring book.
variable (P2 : Nat)

-- Let colored_pics be the number of pictures Rachel colored.
def colored_pics := 44

-- Let remaining_pics be the number of pictures Rachel still has to color.
def remaining_pics := 11

-- Total number of pictures in both coloring books.
def total_pics := colored_pics + remaining_pics

theorem second_coloring_book_pictures :
  P2 = total_pics - P1 :=
by
  -- We need to prove that P2 = 32.
  sorry

end second_coloring_book_pictures_l584_584597


namespace parabola_hyperbola_tangent_l584_584164

theorem parabola_hyperbola_tangent (n : ℝ) :
  (∀ (x y : ℝ), y = x^2 + 9 → y^2 - n * x^2 = 1) ↔ n = 18 + 20 * real.sqrt 2 := by
  sorry

end parabola_hyperbola_tangent_l584_584164


namespace time_difference_for_x_miles_l584_584956

def time_old_shoes (n : Nat) : Int := 10 * n
def time_new_shoes (n : Nat) : Int := 13 * n
def time_difference_for_5_miles : Int := time_new_shoes 5 - time_old_shoes 5

theorem time_difference_for_x_miles (x : Nat) (h : time_difference_for_5_miles = 15) : 
  time_new_shoes x - time_old_shoes x = 3 * x := 
by
  sorry

end time_difference_for_x_miles_l584_584956


namespace fewer_people_with_life_jackets_l584_584214

-- Conditions
variables (raft_capacity_no_life_jackets total_capacity_with_some_life_jackets people_with_life_jackets : ℕ)
variables (total_people_with_all_life_jackets : ℕ)

-- Given Conditions from Problem a)
axiom h1 : raft_capacity_no_life_jackets = 21
axiom h2 : people_with_life_jackets = 8
axiom h3 : total_capacity_with_some_life_jackets = 17

-- To Prove
theorem fewer_people_with_life_jackets :
  total_people_with_all_life_jackets = 11 → 
  raft_capacity_no_life_jackets - total_people_with_all_life_jackets = 10 :=
by
  intros h
  simp [h1, h]
  sorry

end fewer_people_with_life_jackets_l584_584214


namespace values_only_solution_l584_584404

variables (m n : ℝ) (x a b c : ℝ)

noncomputable def equation := (x + m)^3 - (x + n)^3 = (m + n)^3

theorem values_only_solution (hm : m ≠ 0) (hn : n ≠ 0) (hne : m ≠ n)
  (hx : x = a * m + b * n + c) : a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end values_only_solution_l584_584404


namespace michael_probability_l584_584949

noncomputable def probability_of_covering_three_languages (N F S G FS SG FG FSG : ℕ) :=
  let total_pairs := (N.choose 2)
  let purely_French := F - (FS + FG - FSG)
  let purely_Spanish := S - (FS + SG - FSG)
  let purely_German := G - (SG + FG - FSG)
  let only_FS := FS - FSG
  let only_SG := SG - FSG
  let only_FG := FG - FSG
  let irrelevant_pairs := (purely_French.choose 2) + (purely_Spanish.choose 2) + (purely_German.choose 2)
                     + only_FS * (purely_French + purely_Spanish)
                     + only_SG * (purely_Spanish + purely_German)
                     + only_FG * (purely_French + purely_German)
  let coverage_pairs := total_pairs - irrelevant_pairs
  let probability := coverage_pairs.toRat / total_pairs.toRat
  probability

theorem michael_probability :
  probability_of_covering_three_languages 30 20 18 10 12 5 4 3 = 101 / 145 := 
by
  sorry

end michael_probability_l584_584949


namespace find_sum_log2_500_l584_584024

def greatest_integer (x : ℝ) : ℤ :=
  int.floor x

noncomputable def log2 (x : ℕ) : ℝ :=
  real.log x / real.log 2

noncomputable def sum_greatest_integer_log2 (n : ℕ) : ℤ :=
  ∑ k in finset.range (n + 1), greatest_integer (log2 k)

theorem find_sum_log2_500 :
  sum_greatest_integer_log2 500 = 3498 := 
by
  sorry

end find_sum_log2_500_l584_584024


namespace value_of_f_at_pi_over_4_range_and_increasing_interval_l584_584471

-- Define the function f(x)
def f (x : ℝ) : ℝ := Math.sin (2 * x + Real.pi / 3) + Math.cos (2 * x - Real.pi / 6)

-- Question 1: Prove f(π/4) = 1
theorem value_of_f_at_pi_over_4 : f (Real.pi / 4) = 1 := sorry

-- Question 2: Prove range and interval where f(x) is strictly increasing
theorem range_and_increasing_interval :
  (∀ x : ℝ, f(x) ∈ Set.Icc (-2 : ℝ) (2 : ℝ)) ∧
  (∀ k : ℤ, StrictlyIncreasing (Set.Icc (-5 * Real.pi / 12 + k * Real.pi) (k * Real.pi + Real.pi / 12)) (f)) := sorry

end value_of_f_at_pi_over_4_range_and_increasing_interval_l584_584471


namespace not_monotonic_interval_l584_584507

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - log x
noncomputable def f' (x : ℝ) : ℝ := 4 * x - (1/x)

theorem not_monotonic_interval (k : ℝ) (h1 : k - 1 < 1/2) (h2 : k - 1 >= 0) : 
  1 <= k ∧ k < 3/2 :=
sorry

end not_monotonic_interval_l584_584507


namespace tan_of_internal_angle_l584_584457

theorem tan_of_internal_angle (α : ℝ) (h1 : α > 0 ∧ α < π/2) (h2 : cos α = 4/5) : tan α = 3/4 :=
sorry

end tan_of_internal_angle_l584_584457


namespace solve_cubic_eq_l584_584674

theorem solve_cubic_eq (x : ℝ) : (8 - x)^3 = x^3 → x = 8 :=
by
  sorry

end solve_cubic_eq_l584_584674


namespace _l584_584891

variables {A B C D X : Type} [real_point A] [real_point B] [real_point C] [real_point D] [real_point X]

def is_convex_quadrilateral (A B C D : Type) := 
  ∃ (condition_1 : A ≠ B) (condition_2 : B ≠ C) (condition_3 : C ≠ D) (condition_4 : D ≠ A),
    True

-- We assume someone has defined angle, mul_points and ins by proving mathematical equivalents in Lean 4.
variables (AB CD BC DA : Real)
variables (angle_XAB angle_XCD angle_XBC angle_XDA angle_BXA angle_DXC : ℝ)

def angle_equality_1 := angle_XAB = angle_XCD 
def angle_equality_2 := angle_XBC = angle_XDA

-- Using real number points and tri_angle_axioms translations in Lean, we can usually skip this through noncomputation functions.
@[instance] noncomputable def angle_proof_theorem (A B C D X : Type) [is_convex_quadrilateral A B C D]
  (mul_equality : AB * CD = BC * DA ) 
  (angle_equality_1 : angle_XAB = angle_XCD) 
  (angle_equality_2 : angle_XBC = angle_XDA) : (angle_BXA + angle_DXC = 180) :=
sorry

end _l584_584891


namespace sin_60_eq_sqrt3_div_2_l584_584356

theorem sin_60_eq_sqrt3_div_2 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  -- proof skipped
  sorry

end sin_60_eq_sqrt3_div_2_l584_584356


namespace convert_base8_to_base7_l584_584388

theorem convert_base8_to_base7 (n : ℕ) : n = 536 → (num_to_base7 536) = 1054 :=
by
  sorry

def num_to_base10 (n : ℕ) : ℕ :=
  let d2 := (n / 100) % 10 * 8^2
  let d1 := (n / 10) % 10 * 8^1
  let d0 := (n / 1) % 10 * 8^0
  d2 + d1 + d0

def num_to_base7_aux (n : ℕ) (acc : ℕ) (pos : ℕ) : ℕ :=
  if n = 0 then acc
  else
    let q := n / 7
    let r := n % 7
    num_to_base7_aux q ((r * 10^pos) + acc) (pos + 1)

def num_to_base7 (n : ℕ) : ℕ :=
  num_to_base7_aux (num_to_base10 n) 0 0

end convert_base8_to_base7_l584_584388


namespace convex_15gon_smallest_angle_arith_seq_l584_584992

noncomputable def smallest_angle (n : ℕ) (avg_angle d : ℕ) : ℕ :=
156 - 7 * d

theorem convex_15gon_smallest_angle_arith_seq :
  let n := 15 in
  ∀ (a d : ℕ), 
  (a = 156 - 7 * d) ∧
  (avg_angle = (13 * 180) / n) ∧
  (forall i : ℕ, 1 ≤ i ∧ i < n → d < 24 / 7) →
  a = 135 :=
sorry

end convex_15gon_smallest_angle_arith_seq_l584_584992


namespace inverse_of_matrix_A_l584_584770

open Matrix

def matrix_A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 7], ![2, 3]]

def matrix_A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![(-3 : ℚ) / 2, (7 : ℚ) / 2], ![1, -2]]

theorem inverse_of_matrix_A : inverse matrix_A = matrix_A_inv :=
by
  -- Proof goes here
  sorry

end inverse_of_matrix_A_l584_584770


namespace sqrt_sum_simplify_l584_584118

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l584_584118


namespace sum_max_min_fA_l584_584036

/-- Define the set A as a finite set of 100 distinct positive integers. -/
variable (A : Finset ℕ)
variable hA_size : A.card = 100

/-- Define the set B constructed from A as given in the problem. -/
def B (A : Finset ℕ) : Finset ℚ := 
  { x | ∃ a b : ℕ, a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ x = (a : ℚ) / (b : ℚ) }

/-- Define the function f(A) which returns the number of distinct elements in B. -/
def f (A : Finset ℕ) : ℕ := (B A).card

theorem sum_max_min_fA : 
  (f A).max (λ A, f A) + (f A).min (λ A, f A) = 10098 :=
sorry

end sum_max_min_fA_l584_584036


namespace intersection_eq_l584_584478

open Set

-- Define the sets A and B according to the given conditions
def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | (x - 1) * (x + 2) < 0}

-- Define the intended intersection result
def C : Set ℤ := {-1, 0}

-- The theorem to prove
theorem intersection_eq : A ∩ {x | (x - 1) * (x + 2) < 0} = C := by
  sorry

end intersection_eq_l584_584478


namespace probability_between_C_and_D_l584_584062

-- Conditions
variables (A B C D : Type)
variables {length : A → B → ℝ}
variables {AB AD BC CD : ℝ}
variables {segment : A → A → set ℝ}

-- Condition 1: AB = 4 * AD
def condition1 : Prop := AB = 4 * AD

-- Condition 2: AB = 8 * BC
def condition2 : Prop := AB = 8 * BC

-- Correct answer: probability that a point is between C and D is 5/8
def solution : Prop :=
  let CD := AB - AD - BC in
  CD / AB = (5 : ℚ) / 8

-- Lean statement
theorem probability_between_C_and_D :
  condition1 ∧ condition2 → solution :=
by
  intros h,
  sorry

end probability_between_C_and_D_l584_584062


namespace toby_money_share_l584_584220

theorem toby_money_share (initial_money : ℕ) (fraction : ℚ) (brothers : ℕ) (money_per_brother : ℚ)
  (total_shared : ℕ) (remaining_money : ℕ) :
  initial_money = 343 →
  fraction = 1/7 →
  brothers = 2 →
  money_per_brother = fraction * initial_money →
  total_shared = brothers * money_per_brother →
  remaining_money = initial_money - total_shared →
  remaining_money = 245 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end toby_money_share_l584_584220


namespace jonathan_weekly_caloric_deficit_l584_584907

def jonathan_caloric_deficit 
  (daily_calories : ℕ) (extra_calories_saturday : ℕ) (daily_burn : ℕ) 
  (days : ℕ) (saturday : ℕ) : ℕ :=
  let total_consumed := daily_calories * days + (daily_calories + extra_calories_saturday) * saturday in
  let total_burned := daily_burn * (days + saturday) in
  total_burned - total_consumed

theorem jonathan_weekly_caloric_deficit :
  jonathan_caloric_deficit 2500 1000 3000 6 1 = 2500 :=
by
  sorry

end jonathan_weekly_caloric_deficit_l584_584907


namespace right_triangle_hypotenuse_l584_584314

theorem right_triangle_hypotenuse (x y : ℝ)
    (h1 : sqrt (y^2 + (2*x)^2 / 4) = sqrt 52)
    (h2 : sqrt (x^2 + (2*y)^2 / 4) = 3) :
    sqrt ((2*x)^2 + (2*y)^2) ≈ 13.97 := 
by
  sorry

end right_triangle_hypotenuse_l584_584314


namespace radius_triple_area_l584_584149

variable (r n : ℝ)

theorem radius_triple_area (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) : r = (n / 2) * (Real.sqrt 3 - 1) :=
sorry

end radius_triple_area_l584_584149


namespace simplify_sqrt72_add_sqrt32_l584_584128

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l584_584128


namespace Yankees_Mets_Ratio_l584_584874

theorem Yankees_Mets_Ratio :
  (∀ (M B : ℕ), M = 96 → M + B = 216 → (M : B) = 4 / 5) →
  (∀ (Y M B : ℕ), Y + M + B = 360 → M = 96 → B = 120 → (Y : M) = 3 / 2) :=
by
  intros h1 h2
  sorry

end Yankees_Mets_Ratio_l584_584874


namespace perimeter_triangle_pqr_l584_584232

theorem perimeter_triangle_pqr (PQ PR QR QJ : ℕ) (h1 : PQ = PR) (h2 : QJ = 10) :
  ∃ PQR', PQR' = 198 ∧ triangle PQR PQ PR QR := sorry

end perimeter_triangle_pqr_l584_584232


namespace units_selected_from_model_C_l584_584290

def total_production_volume : ℕ := 200 + 400 + 300 + 100

def sampling_ratio : ℚ := 60 / total_production_volume

def production_volumes_C : ℕ := 300

def units_selected_from_C : ℕ := production_volumes_C * (sampling_ratio : ℚ).toRat

theorem units_selected_from_model_C : units_selected_from_C = 18 := by
  have h1 : total_production_volume = 1000 := rfl
  have h2 : sampling_ratio = 60 / 1000 := by
    rw [h1]
    norm_num
  have h3 : (60 / 1000 : ℚ) = 6 / 100 := by norm_num
  have h4 : production_volumes_C = 300 := rfl
  have h5 : units_selected_from_C = production_volumes_C * 6 / 100 :=
    by rw [h4, h3]; norm_num
  rw [h5]
  norm_num
  sorry

end units_selected_from_model_C_l584_584290


namespace west_move_7m_l584_584580

-- Definitions and conditions
def east_move (distance : Int) : Int := distance -- Moving east
def west_move (distance : Int) : Int := -distance -- Moving west is represented as negative

-- Problem: Prove that moving west by 7m is denoted by -7m given the conditions.
theorem west_move_7m : west_move 7 = -7 :=
by
  -- Proof will be handled here normally, but it's omitted as per instruction
  sorry

end west_move_7m_l584_584580


namespace vertex_shifted_coordinates_l584_584614

theorem vertex_shifted_coordinates :
  ∀ (x : ℝ), let original := x^2 + 2*x in
  let shifted := (x - 1)^2 + 2 in
  (shifted = (x + 1)^2 - 1 + 2) ∧ ∃ (vx vy : ℝ), (vx, vy) = (0, 1) :=
by
  sorry

end vertex_shifted_coordinates_l584_584614


namespace perimeter_triangle_pqr_l584_584230

theorem perimeter_triangle_pqr (PQ PR QR QJ : ℕ) (h1 : PQ = PR) (h2 : QJ = 10) :
  ∃ PQR', PQR' = 198 ∧ triangle PQR PQ PR QR := sorry

end perimeter_triangle_pqr_l584_584230


namespace ellipse_with_given_foci_and_point_l584_584157

noncomputable def areFociEqual (a b c₁ c₂ : ℝ) : Prop :=
  c₁ = Real.sqrt (a^2 - b^2) ∧ c₂ = Real.sqrt (a^2 - b^2)

noncomputable def isPointOnEllipse (x₀ y₀ a₂ b₂ : ℝ) : Prop :=
  (x₀^2 / a₂) + (y₀^2 / b₂) = 1

theorem ellipse_with_given_foci_and_point :
  ∃a b : ℝ, 
    areFociEqual 8 3 a b ∧
    a = Real.sqrt 5 ∧ b = Real.sqrt 5 ∧
    isPointOnEllipse 3 (-2) 15 10  :=
sorry

end ellipse_with_given_foci_and_point_l584_584157


namespace probability_between_C_and_D_l584_584063

-- Conditions
variables (A B C D : Type)
variables {length : A → B → ℝ}
variables {AB AD BC CD : ℝ}
variables {segment : A → A → set ℝ}

-- Condition 1: AB = 4 * AD
def condition1 : Prop := AB = 4 * AD

-- Condition 2: AB = 8 * BC
def condition2 : Prop := AB = 8 * BC

-- Correct answer: probability that a point is between C and D is 5/8
def solution : Prop :=
  let CD := AB - AD - BC in
  CD / AB = (5 : ℚ) / 8

-- Lean statement
theorem probability_between_C_and_D :
  condition1 ∧ condition2 → solution :=
by
  intros h,
  sorry

end probability_between_C_and_D_l584_584063


namespace simplify_sqrt_72_plus_sqrt_32_l584_584089

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l584_584089


namespace common_number_condition_l584_584647

theorem common_number_condition
  (first_five_avg last_five_avg total_nine_avg : ℚ)
  (h1 : first_five_avg = 7)
  (h2 : last_five_avg = 9)
  (h3 : total_nine_avg = 73/9) :
  ∃ common_number, common_number = 7 :=
by
  let total_first_five := 5 * first_five_avg
  let total_last_five := 5 * last_five_avg
  let total_sum := 9 * total_nine_avg
  let total_with_overlap := total_first_five + total_last_five
  let common_number := total_with_overlap - total_sum
  use common_number
  sorry

end common_number_condition_l584_584647


namespace percent_increase_correct_l584_584539

def gadget_cost (price: ℝ) (tax_rate: ℝ) : ℝ := price + price * tax_rate

def percent_increase (initial: ℝ) (final: ℝ) : ℝ := (final - initial) / initial * 100

theorem percent_increase_correct:
  ∀ (price : ℝ) (tax_rate1 tax_rate2 : ℝ),
    price = 120 →
    tax_rate1 = 0.10 →
    tax_rate2 = 0.05 →
    percent_increase (gadget_cost price tax_rate2) (gadget_cost price tax_rate1) = 4.76 :=
by
  intros price tax_rate1 tax_rate2
  intro H_price
  intro H_tax_rate1
  intro H_tax_rate2
  sorry

end percent_increase_correct_l584_584539


namespace unit_length_constructible_l584_584317

def is_constructible (x: ℝ) := sorry
-- This is a placeholder for the definition of constructibility using compass and straightedge.

theorem unit_length_constructible (α : ℝ) (hα: α = real.sqrt 2 + real.sqrt 3 + real.sqrt 5) 
  (h_constructible: is_constructible α) : is_constructible 1 :=
sorry

end unit_length_constructible_l584_584317


namespace max_value_problem_l584_584028

theorem max_value_problem
  (a b c : ℝ)
  (h : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_sum : a + b + c = 2) :
  (∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 2 ∧ 
    ∀ x y z, 
      (x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) ∧ (x + y + z = 2) → 
      (frac (x*y) (2*(x + y)) + frac (x*z) (2*(x + z)) + frac (y*z) (2*(y + z)) ≤ 1/2)) :=
begin
  sorry
end

end max_value_problem_l584_584028


namespace average_of_first_21_multiples_of_7_l584_584268

theorem average_of_first_21_multiples_of_7 :
  let a1 := 7
  let d := 7
  let n := 21
  let an := a1 + (n - 1) * d
  let Sn := n / 2 * (a1 + an)
  Sn / n = 77 :=
by
  let a1 := 7
  let d := 7
  let n := 21
  let an := a1 + (n - 1) * d
  let Sn := n / 2 * (a1 + an)
  have h1 : an = 147 := by
    sorry
  have h2 : Sn = 1617 := by
    sorry
  have h3 : Sn / n = 77 := by
    sorry
  exact h3

end average_of_first_21_multiples_of_7_l584_584268


namespace circle_intersection_MN_length_l584_584283

noncomputable def circleEquation (D E F : ℝ) (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_intersection_MN_length :
  ∃ (D E F : ℝ), 
    (circleEquation D E F 1 3) ∧ 
    (circleEquation D E F 4 2) ∧ 
    (circleEquation D E F 1 (-7)) ∧ 
    (let y := -2 + 2 * Real.sqrt 6 in 
    let y' := -2 - 2 * Real.sqrt 6 in
    abs (y - y') = 4 * Real.sqrt 6) :=
by
  sorry

end circle_intersection_MN_length_l584_584283


namespace fraction_sum_l584_584427

theorem fraction_sum : (3 / 8) + (9 / 14) = (57 / 56) := by
  sorry

end fraction_sum_l584_584427


namespace total_distance_l584_584500

-- Definitions from conditions
def KD : ℝ := 4
def DM : ℝ := KD / 2  -- Derived from the condition KD = 2 * DM

/-- The total distance Ken covers is 4 miles (to Dawn's) + 2 miles (to Mary's) + 2 miles (back to Dawn's) + 4 miles (back to Ken's). -/
theorem total_distance : KD + DM + DM + KD = 12 :=
by
  sorry

end total_distance_l584_584500


namespace sum_of_three_integers_l584_584165

theorem sum_of_three_integers :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = 125 ∧ a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_l584_584165


namespace pool_total_capacity_l584_584685

-- Given conditions
variables (C : ℕ)  -- total capacity of the pool
variables (initial_water additional_water : ℕ)
variables (initial_percentage additional_percentage total_percentage : ℝ)

-- Initial condition:
-- - additional 300 gallons to reach 80% capacity
-- - 300 gallons increase water by 30%
def pool_capacity := initial_percentage * C = initial_water ∧
                     additional_water = 300 ∧
                     additional_percentage * C = additional_water ∧
                     total_percentage * C = initial_water + additional_water ∧
                     total_percentage = 0.8 ∧
                     additional_percentage = 0.3

-- Proving that the total capacity of the pool is 1000 gallons.
theorem pool_total_capacity (h : pool_capacity C 0.5 0.3 0.8) : C = 1000 :=
by sorry

end pool_total_capacity_l584_584685


namespace range_of_t_l584_584944

theorem range_of_t (f : ℝ → ℝ) (h1 : ∀ x, 0 < x ∧ x < 1 → f x = x + Real.cos x) :
  {t : ℝ | f (t^2) > f (2*t - 1)} = {t : ℝ | t ∈ (1/2, 1)} :=
by
  sorry

end range_of_t_l584_584944


namespace problem_equivalent_proof_l584_584598

noncomputable def probability_no_distinct_positive_real_roots : ℚ :=
  let valid_pairs := { (b, c) : ℤ × ℤ | -5 ≤ b ∧ b ≤ 5 ∧ -5 ≤ c ∧ c ≤ 5 }.to_finset in
  let total_pairs := valid_pairs.card in
  let invalid_pairs := { (b, c) ∈ valid_pairs | b^2 < 4*c ∨ b ≥ 0 ∨ c ≤ 0 ∨ (b < 0 ∧ c > 0 ∧ b^2 = 4*c) }.to_finset.card in
  1 - (invalid_pairs / total_pairs : ℚ)

theorem problem_equivalent_proof : probability_no_distinct_positive_real_roots = 111 / 121 := sorry

end problem_equivalent_proof_l584_584598


namespace sqrt_sum_simplify_l584_584119

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l584_584119


namespace cos_C_in_triangle_l584_584001

theorem cos_C_in_triangle (A B C : ℝ)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_cos_A : Real.cos A = 3/5)
  (h_sin_B : Real.sin B = 12/13) :
  Real.cos C = 63/65 ∨ Real.cos C = 33/65 :=
sorry

end cos_C_in_triangle_l584_584001


namespace power_function_monotonicity_l584_584867

def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

theorem power_function_monotonicity (m : ℝ) (f : ℝ → ℝ) :
  (f = λ x, (2 * m^2 - 17) * x^(m - 2)) →
  (2 * m^2 - 17 > 0) →
  (m - 2 > 0) →
  monotonically_increasing f →
  m = 3 :=
by sorry

end power_function_monotonicity_l584_584867


namespace cubic_root_sum_l584_584029

-- Assume we have three roots a, b, and c of the polynomial x^3 - 3x - 2 = 0
variables {a b c : ℝ}

-- Using Vieta's formulas for the polynomial x^3 - 3x - 2 = 0
axiom Vieta1 : a + b + c = 0
axiom Vieta2 : a * b + a * c + b * c = -3
axiom Vieta3 : a * b * c = -2

-- The proof that the given expression evaluates to 9
theorem cubic_root_sum:
  a^2 * (b - c)^2 + b^2 * (c - a)^2 + c^2 * (a - b)^2 = 9 :=
by
  sorry

end cubic_root_sum_l584_584029


namespace inequality_holds_l584_584779

theorem inequality_holds (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧ (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
sorry

end inequality_holds_l584_584779


namespace sequence_sum_l584_584035

variables {f : ℝ → ℝ} {a : ℕ → ℝ}

-- Define the function f(x) and the sequence a_n
def g (x : ℝ) : ℝ := (x-2)^5 + 2 * x

-- Define properties of sequence a_n
axiom hz (n : ℕ) : a n ≠ a (n + 1)

axiom h_par : ∀ x y, (1, (x-2)^5) = (1, y - 2*x) → f(x) = (x-2)^5 + 2*x
axiom h_seq : f(a 1) + f(a 2) + ... + f(a 9) = 36
axiom h_seq_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ d ≠ 0

-- Define the sequence sum problem
theorem sequence_sum :
  a 1 + a 2 + ... + a 9 = 18 :=
by
  sorry

end sequence_sum_l584_584035


namespace ratio_of_larger_to_smaller_l584_584053

theorem ratio_of_larger_to_smaller (S L k : ℕ) 
  (hS : S = 32)
  (h_sum : S + L = 96)
  (h_multiple : L = k * S) : L / S = 2 :=
by
  sorry

end ratio_of_larger_to_smaller_l584_584053


namespace ken_total_distance_is_12_l584_584497

-- Definitions:
def dist_ken_dawn : ℝ := 4
def dist_ken_mary : ℝ := dist_ken_dawn / 2
def dist_dawn_mary : ℝ := dist_ken_mary

-- Total distance Ken travels
def total_distance : ℝ := dist_ken_dawn + dist_dawn_mary + dist_dawn_mary + dist_ken_dawn

-- Proof statement
theorem ken_total_distance_is_12 : total_distance = 12 := 
by
  -- Proof omitted
  sorry

end ken_total_distance_is_12_l584_584497


namespace toby_remaining_money_l584_584219

theorem toby_remaining_money : 
  let initial_amount : ℕ := 343
  let fraction_given : ℚ := 1/7
  let total_given := 2 * (initial_amount * fraction_given).to_nat
  initial_amount - total_given = 245 :=
by
  let initial_amount := 343
  let fraction_given := (1/7 : ℚ)
  let given_each := (fraction_given * initial_amount).to_nat
  let total_given := 2 * given_each
  have h : initial_amount - total_given = 245 := by
    calc
      initial_amount - total_given
          = 343 - 2 * 49 : by rw [given_each, (343 : ℕ).mul_div_cancel' (7 : ℕ).nat_abs_pos]
      ... = 343 - 98 : by norm_num
      ... = 245 : by norm_num
  exact h

end toby_remaining_money_l584_584219


namespace students_standing_count_l584_584407

def students_seated : ℕ := 300
def teachers_seated : ℕ := 30
def total_attendees : ℕ := 355

theorem students_standing_count : total_attendees - (students_seated + teachers_seated) = 25 :=
by
  sorry

end students_standing_count_l584_584407


namespace proof_problem_l584_584872

variables (A B C a b c S : ℝ)
variables (sin cos : ℝ → ℝ)
variables (triangle_angle_sum : ∀ A B C, A + B + C = π)
variables (sine_law : a / sin A = b / sin B = c / sin C)
variables (cosine_law : c^2 = a^2 + b^2 - 2 * a * b * cos C)

-- Conditions
def condition1 : Prop := 
  (a * cos B + b * cos A) * cos (2 * C) = c * cos C
def condition2 : Prop := 
  S = (sqrt 3 / 2) * sin A * sin B

-- Questions to be proved
def question1 : Prop := 
  C = 2 * π / 3
def question2 : Prop := 
  b = 2 * a → S = (sqrt 3 / 2) * sin A * sin B → 
  ∃ (c sin_A : ℝ), c = sqrt 2 * sin (2 * π / 3) ∧ sin A = sqrt 21 / 14

theorem proof_problem : condition1 → question1 ∧ 
  (condition2 → question2) :=
sorry

end proof_problem_l584_584872


namespace range_of_b_l584_584472

section ProofProblem

-- Conditions
variable {x : ℝ} {a b : ℝ}
def f (x : ℝ) (a : ℝ) : ℝ := a * x - log x

theorem range_of_b (h_a : 0 < a)
  (h_f : ∀ x > 0, f x a ≥ a^2 / 2 + b) :
  b ≤ 1 / 2 := sorry

end ProofProblem

end range_of_b_l584_584472


namespace hyperbola_center_l584_584295

theorem hyperbola_center (a b c d : ℝ) 
  (h₁ : a = 2) (h₂ : b = 3) (h₃ : c = 10) (h₄ : d = 7) :
  (a + c) / 2 = 6 ∧ (b + d) / 2 = 5 :=
by 
  simp [h₁, h₂, h₃, h₄]
  split
  { norm_num }
  { norm_num }

end hyperbola_center_l584_584295


namespace abc_equilateral_midlines_l584_584583

theorem abc_equilateral_midlines
  (A B C A1 B1 C1 M1 M2 M3 : Point)
  (hABC : is_equilateral_triangle A B C)
  (h_mid : is_midpoint M1 B C ∧ is_midpoint M2 C A ∧ is_midpoint M3 A B)
  (hA1 : on_extension A1 M2 M3 ∧ hB1 : on_extension B1 M3 M1 ∧ hC1 : on_extension C1 M1 M2):
  (passes_through C1 A1 A) ∧ 
  (passes_through A1 B1 B) ∧ 
  (passes_through B1 C1 C) ∧
  is_equilateral_triangle A1 B1 C1 ∧
  divides_in_ratio C1 M1 M2 (1+√5) 2 ∧
  sorry

end abc_equilateral_midlines_l584_584583


namespace probability_of_infinite_events_l584_584568

open MeasureTheory ProbabilityTheory

variables {Ω : Type*} [ProbabilitySpace Ω]
variables (A : ℕ → Set Ω)

theorem probability_of_infinite_events (h : ProbMeasure.prob_eventually_inf A > 0) :
  ∃ (n_k : ℕ → ℕ), ∀ K : ℕ, ProbMeasure.prob (⋂ i in (finset.range K), A (n_k i)) > 0 :=
sorry

end probability_of_infinite_events_l584_584568


namespace median_of_81_consecutive_integers_l584_584176

theorem median_of_81_consecutive_integers (n : ℕ) (S : ℕ) (h1 : n = 81) (h2 : S = 9^5) : 
  let M := S / n in M = 729 :=
by
  sorry

end median_of_81_consecutive_integers_l584_584176


namespace f_neg2_eq_neg1_l584_584856

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem f_neg2_eq_neg1 : f (-2) = -1 := by
  sorry

end f_neg2_eq_neg1_l584_584856


namespace fruit_shop_problem_l584_584518

variable (x y z : ℝ)

theorem fruit_shop_problem
  (h1 : x + 4 * y + 2 * z = 27.2)
  (h2 : 2 * x + 6 * y + 2 * z = 32.4) :
  x + 2 * y = 5.2 :=
by
  sorry

end fruit_shop_problem_l584_584518


namespace cosine_of_largest_angle_l584_584459

-- Given Definitions as conditions
variables {A B C : Type*} [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
variables (AB BC CA : A)

-- Conditions from the problem.
axiom dot_product_condition1 : ⟪AB, BC⟫ = 2 * ⟪BC, CA⟫
axiom dot_product_condition2 : 2 * ⟪BC, CA⟫ = 4 * ⟪CA, AB⟫

-- Statement to prove
theorem cosine_of_largest_angle (cos_A : ℝ) :
  cos_A = (real.sqrt 15) / 15 :=
sorry

end cosine_of_largest_angle_l584_584459


namespace smallest_palindrome_divisible_by_6_l584_584669

def is_palindrome (x : Nat) : Prop :=
  let d1 := x / 1000
  let d2 := (x / 100) % 10
  let d3 := (x / 10) % 10
  let d4 := x % 10
  d1 = d4 ∧ d2 = d3

def is_divisible_by (x n : Nat) : Prop :=
  x % n = 0

theorem smallest_palindrome_divisible_by_6 : ∃ n : Nat, is_palindrome n ∧ is_divisible_by n 6 ∧ 1000 ≤ n ∧ n < 10000 ∧ ∀ m : Nat, (is_palindrome m ∧ is_divisible_by m 6 ∧ 1000 ≤ m ∧ m < 10000) → n ≤ m := 
  by
    exists 2112
    sorry

end smallest_palindrome_divisible_by_6_l584_584669


namespace segment_length_after_reflection_l584_584654

structure Point :=
(x : ℝ)
(y : ℝ)

def reflect_over_x_axis (p : Point) : Point :=
{ x := p.x, y := -p.y }

def distance (p1 p2 : Point) : ℝ :=
abs (p1.y - p2.y)

theorem segment_length_after_reflection :
  let C : Point := {x := -3, y := 2}
  let C' : Point := reflect_over_x_axis C
  distance C C' = 4 :=
by
  sorry

end segment_length_after_reflection_l584_584654


namespace find_constant_N_l584_584394

variables (r h V_A V_B : ℝ)

theorem find_constant_N 
  (h_eq_r : h = r) 
  (r_eq_h : r = h) 
  (vol_relation : V_A = 3 * V_B) 
  (vol_A : V_A = π * r^2 * h) 
  (vol_B : V_B = π * h^2 * r) : 
 ∃ N : ℝ, V_A = N * π * h^3 ∧ N = 9 := 
by 
  use 9
  split
  sorry  -- Proof that V_A = 9 * π * h^3 goes here
  exact eq.refl 9  -- This confirms N = 9 without further proof.


end find_constant_N_l584_584394


namespace john_earns_72_dollars_per_week_l584_584906

noncomputable def johns_weekly_earnings (baskets_per_time: ℕ) (times_per_week: ℕ) (crabs_per_basket: ℕ) (price_per_crab: ℕ) : ℕ :=
  baskets_per_time * times_per_week * crabs_per_basket * price_per_crab

theorem john_earns_72_dollars_per_week :
  ∀ (baskets_per_time times_per_week crabs_per_basket price_per_crab: ℕ),
    baskets_per_time = 3 → times_per_week = 2 → crabs_per_basket = 4 → price_per_crab = 3 →
    johns_weekly_earnings baskets_per_time times_per_week crabs_per_basket price_per_crab = 72 :=
by
  intros baskets_per_time times_per_week crabs_per_basket price_per_crab
  intros h1 h2 h3 h4
  simp [johns_weekly_earnings, h1, h2, h3, h4]
  sorry

end john_earns_72_dollars_per_week_l584_584906


namespace probability_ascending_order_l584_584709

theorem probability_ascending_order :
  let S := {1, 2, 3, 4, 5, 6, 7} in
  let all_selections := finset.univ.powerset.filter (λ s, s.card = 5) in
  let ascending_orderings := all_selections.filter (λ s, s.to_list = s.to_list.qsort (≤)) in
  (ascending_orderings.card : ℝ) / (all_selections.card : ℝ) = 1 / 120 := by
    sorry

end probability_ascending_order_l584_584709


namespace smallest_angle_of_convex_15_gon_arithmetic_sequence_l584_584996

theorem smallest_angle_of_convex_15_gon_arithmetic_sequence :
  ∃ (a d : ℕ), (∀ k : ℕ, k < 15 → (let angle := a + k * d in angle < 180)) ∧
  (∀ i j : ℕ, i < j → i < 15 → j < 15 → (a + i * d) < (a + j * d)) ∧
  (let sequence_sum := 15 * a + d * 7 * 14 in sequence_sum = 2340) ∧
  (d = 3) ∧
  (a = 135) :=
by
  sorry

end smallest_angle_of_convex_15_gon_arithmetic_sequence_l584_584996


namespace median_of_81_consecutive_integers_l584_584173

theorem median_of_81_consecutive_integers (n : ℕ) (S : ℕ) (h1 : n = 81) (h2 : S = 9^5) : 
  let M := S / n in M = 729 :=
by
  sorry

end median_of_81_consecutive_integers_l584_584173


namespace find_a14_l584_584812

-- Define the arithmetic sequence properties
def sum_of_first_n_terms (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

def nth_term (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

theorem find_a14 (a1 d : ℤ) (S11 : sum_of_first_n_terms a1 d 11 = 55)
  (a10 : nth_term a1 d 10 = 9) : nth_term a1 d 14 = 13 :=
sorry

end find_a14_l584_584812


namespace angle_inclination_obtuse_implies_m_lt_2_l584_584447

theorem angle_inclination_obtuse_implies_m_lt_2 (m : ℝ) :
  let A := (-1 : ℝ, 2 : ℝ),
      B := (2 : ℝ, m : ℝ) in
  let slope := (B.2 - A.2) / (B.1 - A.1) in
  slope < 0 → m < 2 :=
by
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (2, m)
  let slope : ℝ := (B.snd - A.snd) / (B.fst - A.fst)
  have slope_is_neg : slope < 0
  intros
  sorry

end angle_inclination_obtuse_implies_m_lt_2_l584_584447


namespace problem_statement_l584_584859

def g (x : ℝ) : ℝ := 3 * x + 2

theorem problem_statement : g (g (g 3)) = 107 := by
  sorry

end problem_statement_l584_584859


namespace machines_working_together_l584_584650

theorem machines_working_together (x : ℝ) :
  (∀ P Q R : ℝ, P = x + 4 ∧ Q = x + 2 ∧ R = 2 * x + 2 ∧ (1 / P + 1 / Q + 1 / R = 1 / x)) ↔ (x = 2 / 3) :=
by
  sorry

end machines_working_together_l584_584650


namespace triple_composition_l584_584858

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition :
  g (g (g 3)) = 107 :=
by
  sorry

end triple_composition_l584_584858


namespace toby_money_share_l584_584221

theorem toby_money_share (initial_money : ℕ) (fraction : ℚ) (brothers : ℕ) (money_per_brother : ℚ)
  (total_shared : ℕ) (remaining_money : ℕ) :
  initial_money = 343 →
  fraction = 1/7 →
  brothers = 2 →
  money_per_brother = fraction * initial_money →
  total_shared = brothers * money_per_brother →
  remaining_money = initial_money - total_shared →
  remaining_money = 245 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end toby_money_share_l584_584221


namespace median_of_81_consecutive_integers_l584_584182

theorem median_of_81_consecutive_integers (ints : ℕ → ℤ) 
  (h_consecutive: ∀ n : ℕ, ints (n+1) = ints n + 1) 
  (h_sum: (∑ i in finset.range 81, ints i) = 9^5) : 
  (ints 40) = 9^3 := 
sorry

end median_of_81_consecutive_integers_l584_584182


namespace system_of_equations_solution_l584_584143

theorem system_of_equations_solution 
  (a : ℝ) 
  (x y z : ℝ) 
  (h1 : a^3 * x + a * y + z = a^2)
  (h2 : x + y + z = 1)
  (h3 : 8 * x + 2 * y + z = 4) : 
  (a = 1 → ∃ x y z : ℝ, a^3 * x + a * y + z = a^2 ∧ x + y + z = 1 ∧ 8 * x + 2 * y + z = 4 ∧ 
    (∃ d : ℝ, ∀ x' y' z' : ℝ, a^3 * x' + a * y' + z' = a^2 ∧ x' + y' + z' = 1 ∧ 8 * x' + 2 * y' + z' = 4 → 
    ∃ k : ℝ, x' = x + k * d ∧ y' = y + k * d ∧ z' = z + k * d))) ∧ 
  (a = 2 → (x = 1/5 ∧ y = 8/5 ∧ z = -2/5)) ∧
  (a = -3 → ¬∃ x y z : ℝ, a^3 * x + a * y + z = a^2 ∧ x + y + z = 1 ∧ 8 * x + 2 * y + z = 4) := 
sorry

end system_of_equations_solution_l584_584143


namespace median_of_consecutive_integers_l584_584185

theorem median_of_consecutive_integers (m : ℤ) (h : (∑ k in (finset.range 81).image (λ x, x - 40 + m), k) = 9^5) :
  m = 729 :=
by
  sorry

end median_of_consecutive_integers_l584_584185


namespace sum_of_first_five_terms_l584_584443

-- Definitions of the conditions in the Lean 4 code
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

def a₁ := 2

def s6_over_s2 (a : ℕ → ℝ) : Prop := sn a 6 / sn a 2 = 21

-- The sequence we're interested in
def b (a : ℕ → ℝ) (n : ℕ) := 1 / a n

-- Statement of the problem
theorem sum_of_first_five_terms (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : a 1 = a₁) (h3 : s6_over_s2 a) :
  sn (b a) 5 = 31 / 32 ∨ sn (b a) 5 = 11 / 32 := 
sorry

end sum_of_first_five_terms_l584_584443


namespace brocard_point_inequality_l584_584073

-- Define the Brocard point condition
def isBrocardPoint (A B C P : Type) [MetricSpace P] [HasAngle P] :
  Prop := ∠PAB = ∠PBC ∧ ∠PBC = ∠PCA

theorem brocard_point_inequality
  {A B C P : Type} [MetricSpace P] [HasAngle P]
  (h : isBrocardPoint A B C P) :
  a^2 + b^2 + c^2 ≤ 3 * (PA^2 + PB^2 + PC^2) :=
sorry

end brocard_point_inequality_l584_584073


namespace area_of_path_cost_of_constructing_path_l584_584715

-- Definitions for the problem
def original_length : ℕ := 75
def original_width : ℕ := 40
def path_width : ℕ := 25 / 10  -- 2.5 converted to a Lean-readable form

-- Conditions
def new_length := original_length + 2 * path_width
def new_width := original_width + 2 * path_width

def area_with_path := new_length * new_width
def area_without_path := original_length * original_width

-- Statements to prove
theorem area_of_path : area_with_path - area_without_path = 600 := sorry

def cost_per_sq_m : ℕ := 2
def total_cost := (area_with_path - area_without_path) * cost_per_sq_m

theorem cost_of_constructing_path : total_cost = 1200 := sorry

end area_of_path_cost_of_constructing_path_l584_584715


namespace simplify_sqrt_sum_l584_584137

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l584_584137


namespace find_f_of_e_l584_584791

noncomputable def f : ℝ → ℝ := λ x, (1/3) * (Real.log x) - 1

theorem find_f_of_e (e : ℝ) (h_exp : e > 0) (h_e : Real.exp 1 = e) : f(e) = - (2/3) :=
by {
  -- this is where the proof would go, but we're skipping it as per instructions
  sorry
}

end find_f_of_e_l584_584791


namespace matrix_transformation_l584_584414

noncomputable def P : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![0, 0, 1], ![0, 3, 0], ![1, 0, 0]]

def Q : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![a, b, c], ![d, e, f], ![g, h, i]]

def targetQ : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![g, h, i], ![3*d, 3*e, 3*f], ![a, b, c]]

theorem matrix_transformation (a b c d e f g h i : ℝ) :
  (P ⬝ Q) = targetQ :=
by
  sorry

end matrix_transformation_l584_584414


namespace tetrahedron_volume_constant_l584_584885

variables {A B C D : Type*}
variable [metric_space A] [add_comm_group A] [module ℝ A]

-- Definitions for edge lengths and points in space
def edge_length (x y: A) : ℝ := dist x y

-- Definitions for edges in the tetrahedron
variables (a b d φ : ℝ)

theorem tetrahedron_volume_constant {A B C D : A}
  (hAB : edge_length A B = a)
  (hCD : edge_length C D = b)
  (hDistanceBetweenLines : dist_between_skew_lines A C = d)
  (hAngle : angle_between_directions A C = φ) :
  ∃ V : ℝ, V = (1 / 6) * a * b * d * real.sin φ :=
by
  sorry

end tetrahedron_volume_constant_l584_584885


namespace medians_intersect_at_centroid_l584_584659

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Definitions of points and vectors in a triangle
variables (A B C M N P : α)
variables (b p n : α)
variables (a b c m n p : α)
variables (x : ℝ)

-- Midpoints condition
def is_midpoint (M : α) (A B : α) : Prop := M = (A + B) / 2

-- Medians definition based on vectors
def median (A M : α) : α := (A + M) / 2

-- Collinearity condition
def collinear (A B C : α) : Prop :=
x • (C - B) + A = C - B

-- Prove that three medians intersect at a single point
theorem medians_intersect_at_centroid (h₁ : is_midpoint M B C)
                                      (h₂ : is_midpoint N A C)
                                      (h₃ : median A M = median B N)
                                      (h₄ : collinear B P N) 
                                      :
                                      ∃ G : α, ∀ M N P, G = (M + N + P) / 3 :=
sorry

end medians_intersect_at_centroid_l584_584659


namespace smallest_congruent_difference_l584_584556

theorem smallest_congruent_difference :
  let m := Nat.find (λ m, m ≥ 100 ∧ m % 13 = 7) in
  let n := Nat.find (λ n, n ≥ 1000 ∧ n % 13 = 7) in
  n - m = 897 :=
by
  let m := Nat.find (λ m, m ≥ 100 ∧ m % 13 = 7)
  let n := Nat.find (λ n, n ≥ 1000 ∧ n % 13 = 7)
  have m_def : m = 111 := by sorry -- Finding m, based on the problem statement
  have n_def : n = 1008 := by sorry -- Finding n, based on the problem statement
  rw [m_def, n_def]
  norm_num
-- Sorry skips the computational proof steps

end smallest_congruent_difference_l584_584556


namespace sequence_b_is_4_consecutively_representable_l584_584439

-- Define what it means for a sequence to be 4-consecutively representable
def four_consecutively_representable (s : List ℤ) : Prop :=
  ∀ n ∈ {1, 2, 3, 4}, ∃ (i j : ℕ), i ≤ j ∧ j < s.length ∧ List.sum (s.slice i (j + 1)) = n

-- Define the sequence in question
def sequence_b : List ℤ := [1, 1, 2]

-- The proof statement
theorem sequence_b_is_4_consecutively_representable : four_consecutively_representable sequence_b :=
  sorry

end sequence_b_is_4_consecutively_representable_l584_584439


namespace mark_more_than_kate_l584_584586

variables {K P M : ℕ}

-- Conditions
def total_hours (P K M : ℕ) : Prop := P + K + M = 189
def pat_as_kate (P K : ℕ) : Prop := P = 2 * K
def pat_as_mark (P M : ℕ) : Prop := P = M / 3

-- Statement
theorem mark_more_than_kate (K P M : ℕ) (h1 : total_hours P K M)
  (h2 : pat_as_kate P K) (h3 : pat_as_mark P M) : M - K = 105 :=
by {
  sorry
}

end mark_more_than_kate_l584_584586


namespace greatest_value_x_plus_inv_x_l584_584502

theorem greatest_value_x_plus_inv_x (x : ℝ) (h : 13 = x^2 + 1 / x^2) : x + 1 / x ≤ sqrt 15 :=
by
  -- Proof goes here
  sorry

end greatest_value_x_plus_inv_x_l584_584502


namespace time_morning_is_one_l584_584678

variable (D : ℝ)  -- Define D as the distance between the two points.

def morning_speed := 20 -- Morning speed (km/h)
def afternoon_speed := 10 -- Afternoon speed (km/h)
def time_difference := 1 -- Time difference (hour)

-- Proving that the morning time t_m is equal to 1 hour
theorem time_morning_is_one (t_m t_a : ℝ) 
  (h1 : t_m - t_a = time_difference) 
  (h2 : D = morning_speed * t_m) 
  (h3 : D = afternoon_speed * t_a) : 
  t_m = 1 := 
by
  sorry

end time_morning_is_one_l584_584678


namespace ratio_of_democrats_l584_584644

theorem ratio_of_democrats (F M : ℕ) (h1 : F + M = 750) (h2 : (1/2 : ℚ) * F = 125) (h3 : (1/4 : ℚ) * M = 125) :
  (125 + 125 : ℚ) / 750 = 1 / 3 := by
  sorry

end ratio_of_democrats_l584_584644


namespace remainder_of_division_l584_584666

-- Define the polynomial and divisor
def polynomial : polynomial ℤ := 3 * X^2 - 11 * X + 18
def divisor : polynomial ℤ := X - 3

-- State the theorem
theorem remainder_of_division (p : polynomial ℤ) (d : polynomial ℤ) : p = polynomial → d = divisor → polynomial % divisor = 12 := 
by
  intros h1 h2
  rw [h1, h2]
  -- prove the division and remainder
  sorry

end remainder_of_division_l584_584666


namespace de_morgan_birth_year_jenkins_birth_year_l584_584948

open Nat

theorem de_morgan_birth_year
  (x : ℕ) (hx : x = 43) (hx_square : x * x = 1849) :
  1849 - 43 = 1806 :=
by
  sorry

theorem jenkins_birth_year
  (a b : ℕ) (ha : a = 5) (hb : b = 6) (m : ℕ) (hm : m = 31) (n : ℕ) (hn : n = 5)
  (ha_sq : a * a = 25) (hb_sq : b * b = 36) (ha4 : a * a * a * a = 625)
  (hb4 : b * b * b * b = 1296) (hm2 : m * m = 961) (hn4 : n * n * n * n = 625) :
  1921 - 61 = 1860 ∧
  1922 - 62 = 1860 ∧
  1875 - 15 = 1860 :=
by
  sorry

end de_morgan_birth_year_jenkins_birth_year_l584_584948


namespace problem_statement_l584_584441

noncomputable def m (α : ℝ) : ℝ := - (Real.sqrt 2) / 4

noncomputable def tan_alpha (α : ℝ) : ℝ := 2 * Real.sqrt 2

theorem problem_statement (α : ℝ) (P : (ℝ × ℝ)) (h1 : P = (m α, 1)) (h2 : Real.cos α = - 1 / 3) :
  (P.1 = - (Real.sqrt 2) / 4) ∧ (Real.tan α = 2 * Real.sqrt 2) :=
by
  sorry

end problem_statement_l584_584441


namespace sally_nickels_l584_584970

theorem sally_nickels :
  ∀ (initial_nickels dad_nickels mom_nickels : ℕ),
    initial_nickels = 7 →
    dad_nickels = 9 →
    mom_nickels = 2 →
    initial_nickels + dad_nickels + mom_nickels = 18 :=
by
  intros initial_nickels dad_nickels mom_nickels h_initial h_dad h_mom
  rw [h_initial, h_dad, h_mom]
  norm_num

end sally_nickels_l584_584970


namespace math_problem_l584_584348

theorem math_problem : (sqrt 18 / 3 + |sqrt 2 - 2| + 2023^0 - (-1)^1) = 2 :=
by sorry

end math_problem_l584_584348


namespace sum_fractions_abs_leq_half_l584_584792

theorem sum_fractions_abs_leq_half (n : ℕ) (x : ℕ → ℝ)
  (h1 : ∑ i in finset.range n, |x i| = 1)
  (h2 : ∑ i in finset.range n, x i = 0) :
  |∑ i in finset.range n, (x i) / (i + 1)| ≤ 1 / 2 - 1 / (2 * n) := 
sorry

end sum_fractions_abs_leq_half_l584_584792


namespace problem_statement_l584_584860

def g (x : ℝ) : ℝ := 3 * x + 2

theorem problem_statement : g (g (g 3)) = 107 := by
  sorry

end problem_statement_l584_584860


namespace solution_set_of_inequality_l584_584758

variable (f : ℝ → ℝ)

theorem solution_set_of_inequality (h1 : ∀ x, f'' x > 1 - f x) 
    (h2 : f 0 = 6) 
    (h3 : ∀ x, deriv f'' x = deriv (λ y, deriv f y)) :
    { x : ℝ | e^x * f x > e^x + 5 } = set.Ioi 0 :=
by
  -- skip the proof
  sorry

end solution_set_of_inequality_l584_584758


namespace convex_15gon_smallest_angle_arith_seq_l584_584990

noncomputable def smallest_angle (n : ℕ) (avg_angle d : ℕ) : ℕ :=
156 - 7 * d

theorem convex_15gon_smallest_angle_arith_seq :
  let n := 15 in
  ∀ (a d : ℕ), 
  (a = 156 - 7 * d) ∧
  (avg_angle = (13 * 180) / n) ∧
  (forall i : ℕ, 1 ≤ i ∧ i < n → d < 24 / 7) →
  a = 135 :=
sorry

end convex_15gon_smallest_angle_arith_seq_l584_584990


namespace median_of_81_consecutive_integers_l584_584194

theorem median_of_81_consecutive_integers (sum : ℕ) (h₁ : sum = 9^5) : 
  let mean := sum / 81 in mean = 729 :=
by
  have h₂ : sum = 59049 := by
    rw h₁
    norm_num
  have h₃ : mean = 59049 / 81 := by
    rw h₂
    rfl
  have result : mean = 729 := by
    rw h₃
    norm_num
  exact result

end median_of_81_consecutive_integers_l584_584194


namespace fair_coin_expected_value_1000_l584_584509

open ProbabilityTheory

noncomputable def coin_flip_expected_value (n : ℕ) : ℝ :=
  ∑ k in finset.range (n+1), (n*(n-1))/4

theorem fair_coin_expected_value_1000 :
  coin_flip_expected_value 1000 = 249750 := by
  sorry

end fair_coin_expected_value_1000_l584_584509


namespace tetrahedron_properties_l584_584526

theorem tetrahedron_properties 
  (S A B C : Point) 
  (tetrahedron : Tetrahedron S A B C)
  (angle_SBA : ∠ SBA = 90)
  (angle_SCA : ∠ SCA = 90)
  (isosceles_right_triangle : IsIsoscelesRightTriangle A B C)
  (hypotenuse_length : length AB = a) :
  (angle_between_skew_lines SB AC = 90) ∧ 
  (perpendicular_to_plane SB (plane ABC)) ∧ 
  (perpendicular_planes (plane SBC) (plane SAC)) ∧ 
  (distance_from_point_to_plane C (plane SAB) = a / 2) :=
by
  sorry

end tetrahedron_properties_l584_584526


namespace simplify_sqrt_sum_l584_584135

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l584_584135


namespace exists_geometric_arithmetic_progressions_l584_584544

theorem exists_geometric_arithmetic_progressions (n : ℕ) (hn : n > 3) :
  ∃ (x y : ℕ → ℕ),
  (∀ m < n, x (m + 1) = (1 + ε)^m ∧ y (m + 1) = (1 + (m + 1) * ε - δ)) ∧
  ∀ m < n, x m < y m ∧ y m < x (m + 1) :=
by
  sorry

end exists_geometric_arithmetic_progressions_l584_584544


namespace measure_angle_F_correct_l584_584003

noncomputable def measure_angle_D : ℝ := 80
noncomputable def measure_angle_F : ℝ := 70 / 3
noncomputable def measure_angle_E (angle_F : ℝ) : ℝ := 2 * angle_F + 30
noncomputable def angle_sum_property (angle_D angle_E angle_F : ℝ) : Prop :=
  angle_D + angle_E + angle_F = 180

theorem measure_angle_F_correct : measure_angle_F = 70 / 3 :=
by
  let angle_D := measure_angle_D
  let angle_F := measure_angle_F
  have h1 : measure_angle_E angle_F = 2 * angle_F + 30 := rfl
  have h2 : angle_sum_property angle_D (measure_angle_E angle_F) angle_F := sorry
  sorry

end measure_angle_F_correct_l584_584003


namespace large_beaker_multiple_small_beaker_l584_584318

variables (S L : ℝ) (k : ℝ)

theorem large_beaker_multiple_small_beaker 
  (h1 : Small_beaker = S)
  (h2 : Large_beaker = k * S)
  (h3 : Salt_water_in_small = S/2)
  (h4 : Fresh_water_in_large = (Large_beaker) / 5)
  (h5 : (Salt_water_in_small + Fresh_water_in_large = 0.3 * (Large_beaker))) :
  k = 5 :=
sorry

end large_beaker_multiple_small_beaker_l584_584318


namespace correct_propositions_count_l584_584742

theorem correct_propositions_count : (∃ a b c d : Prop, 
  (a = (∀ (q : Quadrilateral), (q.sides_equal -> q.is_rhombus)) ∧
  b = (∀ (q : Quadrilateral), (q.has_two_right_angles -> q.is_cyclic)) ∧
  c = (∀ (p : Plane) (l : Line), ((¬ p.contains l) ↔ (at_most_one_point_in_common p l))) ∧
  d = (∀ (p1 p2 : Plane) (l : Line), (p1.common_line p2 l -> (∀ x, (p1.contains x ∧ p2.contains x) -> l.contains x))) ∧ 
  (c ∧ d)) ∧ (¬a) ∧ (¬b)) :=
by
  sorry

end correct_propositions_count_l584_584742


namespace prove_zero_function_l584_584411

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq : ∀ x y : ℝ, f (x ^ 333 + y) = f (x ^ 2018 + 2 * y) + f (x ^ 42)

theorem prove_zero_function : ∀ x : ℝ, f x = 0 :=
by
  sorry

end prove_zero_function_l584_584411


namespace part_a_part_b_l584_584725

variables (p : ℝ) (h : p ≥ 0 ∧ p ≤ 1) -- Probability p is between 0 and 1 inclusive

-- Part (a)
theorem part_a (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (5.choose 3) * p^3 * (1 - p)^2 = 10 * p^3 * (1 - p)^2 := by 
sorry

-- Part (b)
theorem part_b (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  10 * p - p^10 = (10 * p - p^10) := by 
sorry

end part_a_part_b_l584_584725


namespace range_of_a_l584_584957

noncomputable def f (x : ℝ) : ℝ := -exp(x) - x
noncomputable def g (a x : ℝ) : ℝ := a * x + 2 * Real.cos x

theorem range_of_a : {a : ℝ | ∀ x₀ : ℝ, ∃ t : ℝ, (-exp(x₀) - 1) * (a - 2 * Real.sin(t)) = -1 } = Set.Icc (-1 : ℝ) 2 :=
by
  sorry

end range_of_a_l584_584957


namespace probability_green_l584_584213

def total_marbles : ℕ := 100

def P_white : ℚ := 1 / 4

def P_red_or_blue : ℚ := 0.55

def P_sum : ℚ := 1

theorem probability_green :
  P_sum = P_white + P_red_or_blue + P_green →
  P_green = 0.2 :=
sorry

end probability_green_l584_584213


namespace perimeter_triangle_pqr_l584_584228

theorem perimeter_triangle_pqr (PQ PR QR QJ : ℕ) (h1 : PQ = PR) (h2 : QJ = 10) :
  ∃ PQR', PQR' = 198 ∧ triangle PQR PQ PR QR := sorry

end perimeter_triangle_pqr_l584_584228


namespace probability_between_CD_l584_584069

-- Define the points A, B, C, and D on a line segment
variables (A B C D : ℝ)

-- Provide the conditions
axiom h_ab_ad : B - A = 4 * (D - A)
axiom h_ab_bc : B - A = 8 * (B - C)

-- Define the probability statement 
theorem probability_between_CD (AB length: ℝ) : 
  (0 ≤ A ∧ A < D ∧ D < C ∧ C < B) → (B - A = 1) → 
  (B - A = 4 * (D - A)) → (B - A = 8 * (B - C)) → 
  probability (A B C D) = 5 / 8 :=
by
  sorry

end probability_between_CD_l584_584069


namespace tan_7pi_over_6_l584_584276

noncomputable def tan_periodic (θ : ℝ) : Prop :=
  ∀ k : ℤ, Real.tan (θ + k * Real.pi) = Real.tan θ

theorem tan_7pi_over_6 : Real.tan (7 * Real.pi / 6) = Real.sqrt 3 / 3 :=
by
  sorry

end tan_7pi_over_6_l584_584276


namespace median_of_consecutive_integers_l584_584189

theorem median_of_consecutive_integers (m : ℤ) (h : (∑ k in (finset.range 81).image (λ x, x - 40 + m), k) = 9^5) :
  m = 729 :=
by
  sorry

end median_of_consecutive_integers_l584_584189


namespace percent_increase_l584_584684

theorem percent_increase (o n : ℕ) (ho : o = 30) (hn : n = 60) :
  (n - o) * 100 / o = 100 :=
by
  have h1 : n - o = 30 := by rw [ho, hn]; norm_num
  have h2 : (n - o) * 100 = 3000 := by rw [h1]; norm_num
  have h3 : 3000 / o = 100 := by rw [ho]; norm_num
  rw [← h2, h3]
  exact rfl

end percent_increase_l584_584684


namespace correct_answer_l584_584653

def total_contestants : Nat := 56
def selected_contestants : Nat := 14

theorem correct_answer :
  (total_contestants = 56) →
  (selected_contestants = 14) →
  (selected_contestants = 14) :=
by
  intro h_total h_selected
  exact h_selected

end correct_answer_l584_584653


namespace range_of_k_for_sine_in_M_l584_584046

open Real

-- Define the set of functions satisfying the given property
def M (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f(x + T) = T * f(x)

-- Define the function in question
def f (k x : ℝ) : ℝ := sin (k * x)

-- State the main theorem to be proved
theorem range_of_k_for_sine_in_M (k : ℝ) : M (f k) ↔ ∃ m : ℤ, k = m * π :=
sorry

end range_of_k_for_sine_in_M_l584_584046


namespace length_of_AC_l584_584434

noncomputable def cos_30 : ℝ := (√(3))/2

theorem length_of_AC (b : ℝ) :
  let AB := √3
  let BC := 1
  let A := 30 * Math.pi / 180
  (BC^2 = b^2 + AB^2 - 2 * AB * b * cos_30) →
  b = 1 ∨ b = 2 :=
begin
  sorry
end

end length_of_AC_l584_584434


namespace smallest_perimeter_iso_triangle_l584_584234

theorem smallest_perimeter_iso_triangle :
  ∃ (x y : ℕ), (PQ = PR ∧ PQ = x ∧ PR = x ∧ QR = y ∧ QJ = 10 ∧ PQ + PR + QR = 416 ∧ 
  PQ = PR ∧ y = 8 ∧ 2 * (x + y) = 416 ∧ y^2 - 50 > 0 ∧ y < 10) :=
sorry

end smallest_perimeter_iso_triangle_l584_584234


namespace largest_constructible_n_gon_under_limit_l584_584008

def is_Fermat_prime (p : ℕ) : Prop :=
  ∃ k : ℕ, p = 2^(2^k) + 1 ∧ Nat.prime p

def constructible_n_gon (n : ℕ) : Prop :=
  ∃ (a : ℕ) (primes : List ℕ), n = 2^a * (primes.prod) ∧ (∀ p ∈ primes, is_Fermat_prime p)

theorem largest_constructible_n_gon_under_limit (limit : ℕ) : ∃ n : ℕ, n < limit ∧ constructible_n_gon n ∧ 
  ∀ m : ℕ, m < limit ∧ constructible_n_gon m → m ≤ n :=
  let limit := 4300000000 in
  ∃ n : ℕ, n < limit ∧ constructible_n_gon n ∧ (n = 2^32) :=
sorry

end largest_constructible_n_gon_under_limit_l584_584008


namespace coeff_x4_is_3_l584_584417

-- Define the polynomial expression
def polynomial_expr : ℤ[X] := 4*X^4 - 8*X^3 + 3*X - 3*X^4 + 6*X^5 - 5*X^5 + 2*X^4

-- Prove the coefficient of x^4 is 3
theorem coeff_x4_is_3 : polynomial_expr.coeff 4 = 3 :=
by
  -- The proof is omitted.
  sorry

end coeff_x4_is_3_l584_584417


namespace problem_sol_l584_584805

noncomputable theory

def O : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (-1, 1)
def B : (ℝ × ℝ) := (1, 3)
def C : (ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Statement A: Check that AB vector is not (-2, -2)
def AB : (ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)

-- Statement B: Check coordinates of C
def is_midpoint (P Q M : ℝ × ℝ) : Prop := 
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Statement C: Perpendicular check
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def perpendicular (u v : ℝ × ℝ) : Prop := dot_product u v = 0

-- Statement D: Angle check between vectors OA and OC
def magnitude (u : ℝ × ℝ) : ℝ := sqrt (u.1^2 + u.2^2)
def cos_angle (u v : ℝ × ℝ) : ℝ := dot_product u v / (magnitude u * magnitude v)
def angle π (u v : ℝ × ℝ) : Prop := cos_angle u v = (√2 / 2)

theorem problem_sol (h : is_midpoint A B C) : 
  AB ≠ (-2, -2) ∧
  C = (0, 2) ∧
  perpendicular A AB ∧
  angle π (A) (C) :=
by 
  split;
  sorry

end problem_sol_l584_584805


namespace cheetah_catch_up_time_l584_584289

-- Define the conditions
def deer_speed : ℝ := 50 -- miles per hour
def cheetah_speed : ℝ := 60 -- miles per hour
def cheetah_delay : ℝ := 2 / 60 -- time in hours (2 minutes)

-- Define the distance covered by deer in the initial 2 minutes
noncomputable def distance_deer : ℝ := deer_speed * cheetah_delay -- miles

-- Define the relative speed
def relative_speed : ℝ := cheetah_speed - deer_speed -- miles per hour

-- Define the time for cheetah to catch up in hours
noncomputable def time_to_catch_up : ℝ := distance_deer / relative_speed -- hours

-- Define the time for cheetah to catch up in minutes
noncomputable def time_to_catch_up_minutes : ℝ := time_to_catch_up * 60 -- minutes

-- The proof problem statement
theorem cheetah_catch_up_time :
  time_to_catch_up_minutes = 10 :=
sorry

end cheetah_catch_up_time_l584_584289


namespace tripod_height_l584_584326

-- Define the conditions of the problem
structure Tripod where
  leg_length : ℝ
  angle_equal : Bool
  top_height : ℝ
  broken_length : ℝ

def m : ℕ := 27
def n : ℕ := 10

noncomputable def h : ℝ := m / Real.sqrt n

theorem tripod_height :
  ∀ (t : Tripod),
  t.leg_length = 6 →
  t.angle_equal = true →
  t.top_height = 3 →
  t.broken_length = 2 →
  (h = m / Real.sqrt n) →
  (⌊m + Real.sqrt n⌋ = 30) :=
by
  intros
  sorry

end tripod_height_l584_584326


namespace probability_between_CD_l584_584071

-- Define the points A, B, C, and D on a line segment
variables (A B C D : ℝ)

-- Provide the conditions
axiom h_ab_ad : B - A = 4 * (D - A)
axiom h_ab_bc : B - A = 8 * (B - C)

-- Define the probability statement 
theorem probability_between_CD (AB length: ℝ) : 
  (0 ≤ A ∧ A < D ∧ D < C ∧ C < B) → (B - A = 1) → 
  (B - A = 4 * (D - A)) → (B - A = 8 * (B - C)) → 
  probability (A B C D) = 5 / 8 :=
by
  sorry

end probability_between_CD_l584_584071


namespace domain_of_h_l584_584662

def h (x : ℝ) : ℝ := (2 * x - 5) / (x - 3)

theorem domain_of_h : { x : ℝ | x ≠ 3 } = set.Ioo (-∞ : ℝ) 3 ∪ set.Ioo 3 ∞ :=
by
  sorry

end domain_of_h_l584_584662


namespace general_term_l584_584836

def a : ℕ → ℚ
| 0     := 3
| (n+1) := a n + n + 1

theorem general_term (n : ℕ) : a n = (n^2 + n + 4) / 2 := sorry

end general_term_l584_584836


namespace repeating_period_of_fraction_l584_584642

theorem repeating_period_of_fraction (n : ℕ) (h1 : 221 = 13 * 17)
  (h2 : ∀ x : ℕ, (10^x : ℤ) % 13 = 1 ↔ x % 6 = 0)
  (h3 : ∀ y : ℕ, (10^y : ℤ) % 17 = 1 ↔ y % 16 = 0) :
  ∃ m : ℕ, m = 48 ∧ (10^m : ℤ) % 221 = 1 :=
begin
  use 48,
  split,
  { refl, },
  { sorry, }
end

end repeating_period_of_fraction_l584_584642


namespace sum_of_coefficients_l584_584492

theorem sum_of_coefficients :
  let a : Fin 2025 → ℝ := λ n, if n = 0 then 1 else 2 * (-1)^2023
  in ∑ i in Finset.range 2025.erase 0, (a i) = -3 := by
    sorry

end sum_of_coefficients_l584_584492


namespace median_of_81_consecutive_integers_l584_584204

theorem median_of_81_consecutive_integers (S : ℤ) 
  (h1 : ∃ l : ℤ, ∀ k, (0 ≤ k ∧ k < 81) → (l + k) ∈ S) 
  (h2 : S = 9^5) : 
  (81 * 729 = 9^5) :=
by
  have h_sum : (S : ℤ) = ∑ i in ((finset.range 81).map (λ n, l + n)), id i, from sorry
  have h_eq : (81 * 729 = 9^5) := calc
    81 * 729 = 9^2 * 9^3 : by norm_num
    ... = 9^(2+3) : by ring
    ... = 9^5 : by norm_num
  exact h_eq

end median_of_81_consecutive_integers_l584_584204


namespace student_incorrect_answer_l584_584876

theorem student_incorrect_answer (D : ℕ) (h1 : D / 36 = 48) : D / 72 = 24 :=
by {
  have h2 := h1,
  -- We can use this intermediate step to clarify our thinking but it's not necessary in actual Lean code.
  -- have h2 : D = 36 * 48, 
  sorry
}

end student_incorrect_answer_l584_584876


namespace puppy_food_per_day_after_first_60_days_approx_l584_584292

-- Definitions based on conditions
def first_60_days_food_per_day : ℕ := 2 -- ounces
def total_days_in_year : ℕ := 365
def first_60_days : ℕ := 60
def bags_purchased : ℕ := 17
def ounces_per_pound : ℕ := 16
def pounds_per_bag : ℕ := 5

-- The total amount of food in ounces in one bag
def ounces_per_bag : ℕ := pounds_per_bag * ounces_per_pound

-- The total amount of food purchased in ounces
def total_food_ounces : ℕ := bags_purchased * ounces_per_bag

-- The amount of food consumed in the first 60 days
def food_consumed_first_60_days : ℕ := first_60_days * first_60_days_food_per_day

-- Remaining food after the first 60 days
def remaining_food_ounces : ℕ := total_food_ounces - food_consumed_first_60_days

-- Number of days after the first 60 days
def remaining_days : ℕ := total_days_in_year - first_60_days

-- The amount of food the puppy needs to eat per day after the first 60 days (non-rounded)
def food_per_day_after_first_60_days : ℝ :=
  remaining_food_ounces / remaining_days.to_real 

theorem puppy_food_per_day_after_first_60_days_approx :
  food_per_day_after_first_60_days ≈ 4.07 :=
  by
    sorry

end puppy_food_per_day_after_first_60_days_approx_l584_584292


namespace profit_percentage_is_50_l584_584600

noncomputable def cost_of_machine := 11000
noncomputable def repair_cost := 5000
noncomputable def transportation_charges := 1000
noncomputable def selling_price := 25500

noncomputable def total_cost := cost_of_machine + repair_cost + transportation_charges
noncomputable def profit := selling_price - total_cost
noncomputable def profit_percentage := (profit / total_cost) * 100

theorem profit_percentage_is_50 : profit_percentage = 50 := by
  sorry

end profit_percentage_is_50_l584_584600


namespace find_n_l584_584558

theorem find_n :
  (∃ n : ℕ, let x := (1 + 2) * (1 + 2^2) * (1 + 2^4) * (1 + 2^8) * ... * (1 + 2^n) in x + 1 = 2^128) →
  ∃ n : ℕ, n = 64 := by
  sorry

end find_n_l584_584558


namespace third_car_year_l584_584651

theorem third_car_year (y1 y2 y3 : ℕ) (h1 : y1 = 1970) (h2 : y2 = y1 + 10) (h3 : y3 = y2 + 20) : y3 = 2000 :=
by
  sorry

end third_car_year_l584_584651


namespace complex_conjugate_quadrant_l584_584460

theorem complex_conjugate_quadrant (z : ℂ) (h : z / (1 + complex.I) = 2 * complex.I) :
    (complex.conj z).re < 0 ∧ (complex.conj z).im < 0 :=
by
  sorry

end complex_conjugate_quadrant_l584_584460


namespace kevin_leaves_l584_584538

theorem kevin_leaves (n : ℕ) (h : n > 1) : ∃ k : ℕ, n = k^3 ∧ n^2 = k^6 ∧ n = 8 := by
  sorry

end kevin_leaves_l584_584538


namespace original_price_of_trouser_l584_584535

theorem original_price_of_trouser (sale_price : ℝ) (discount : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 30) (h2 : discount = 0.70) : 
  original_price = 100 :=
by
  sorry

end original_price_of_trouser_l584_584535


namespace part1_part2_part3_l584_584827

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

theorem part1 :
  ∀ x : ℝ, |f x| = |x - 1| → x = -2 ∨ x = 0 ∨ x = 1 :=
sorry

theorem part2 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |f x1| = g a x1 ∧ |f x2| = g a x2) ↔ (a = 0 ∨ a = 2) :=
sorry

theorem part3 (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) ↔ (a ≤ -2) :=
sorry

end part1_part2_part3_l584_584827


namespace prime_m_l584_584400

theorem prime_m (m : ℕ) (hm : m ≥ 2) :
  (∀ n : ℕ, (m / 3 ≤ n) → (n ≤ m / 2) → (n ∣ Nat.choose n (m - 2 * n))) → Nat.Prime m :=
by
  intro h
  sorry

end prime_m_l584_584400


namespace simplify_sqrt_sum_l584_584106

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l584_584106


namespace arithmetic_sequence_sum_l584_584633

theorem arithmetic_sequence_sum :
  ∃ (a b c : ℕ), 
    (∀ k, 3 + 6 * k ∈ {3, 9, 15, a, b, c, 33}) ∧ 
    a + b + c = 81 :=
begin
  let d := 6,
  use [21, 27, 33],
  split,
  { intro k,
    cases k,
    { left, refl },
    { cases k,
      { right, left, refl },
      { cases k,
        { right, right, left, refl },
        { cases k,
          { right, right, right, left, refl },
          { cases k,
            { right, right, right, right, left, refl },
            { cases k,
              { right, right, right, right, right, left, refl },
              { right, right, right, right, right, right, right, refl } } } } } } } },
  { simp }
end

end arithmetic_sequence_sum_l584_584633


namespace beak_theorem_l584_584687

theorem beak_theorem (A O P Q : Point) (r : ℝ)
    (hTangent : Tangent A P) (hTangent : Tangent A Q)
    (hCenter : Center O)
    (hOrthogonal1 : Ortho P O A)
    (hOrthogonal2 : Ortho Q O A) : 
    dist A P = dist A Q := 
sorry

end beak_theorem_l584_584687


namespace intersection_of_sets_values_of_a_and_b_l584_584044

noncomputable section

def setA := { x : ℝ | 2^x - 1 > 0 }
def setB := { x : ℝ | (x^2 - x - 6 > 0) }

theorem intersection_of_sets :
  (setA ∩ setB) = (set.Ioi 3) := sorry

theorem values_of_a_and_b (a b : ℝ) :
  (∀ x : ℝ, ax^2 + 2x + b > 0 ↔ x ∈ (setA ∪ setB)) → a = 1 ∧ b = 0 := sorry

end intersection_of_sets_values_of_a_and_b_l584_584044


namespace cats_on_edges_l584_584686

variables {W1 W2 B1 B2 : ℕ}  -- representing positions of cats on a line

def distance_from_white_to_black_sum_1 (a1 a2 : ℕ) : Prop := a1 + a2 = 4
def distance_from_white_to_black_sum_2 (b1 b2 : ℕ) : Prop := b1 + b2 = 8
def distance_from_black_to_white_sum_1 (b1 a1 : ℕ) : Prop := b1 + a1 = 9
def distance_from_black_to_white_sum_2 (b2 a2 : ℕ) : Prop := b2 + a2 = 3

theorem cats_on_edges
  (a1 a2 b1 b2 : ℕ)
  (h1 : distance_from_white_to_black_sum_1 a1 a2)
  (h2 : distance_from_white_to_black_sum_2 b1 b2)
  (h3 : distance_from_black_to_white_sum_1 b1 a1)
  (h4 : distance_from_black_to_white_sum_2 b2 a2) :
  (a1 = 2) ∧ (a2 = 2) ∧ (b1 = 7) ∧ (b2 = 1) ∧ (W1 = min W1 W2) ∧ (B2 = max B1 B2) :=
sorry

end cats_on_edges_l584_584686


namespace angle_F_calculation_l584_584004

theorem angle_F_calculation (D E F : ℝ) :
  D = 80 ∧ E = 2 * F + 30 ∧ D + E + F = 180 → F = 70 / 3 :=
by
  intro h
  cases' h with hD h_remaining
  cases' h_remaining with hE h_sum
  sorry

end angle_F_calculation_l584_584004


namespace wicket_keeper_age_difference_l584_584516

theorem wicket_keeper_age_difference 
  (team_size : ℕ)
  (captain_age : ℕ)
  (average_team_age : ℕ)
  (average_remaining_age : ℕ)
  (remaining_player_count : ℕ)
  (total_team_age : ℕ)
  (total_remaining_age : ℕ)
  (W : ℕ)
  (x : ℕ) :
  team_size = 11 →
  captain_age = 27 →
  average_team_age = 23 →
  average_remaining_age = 22 →
  remaining_player_count = 9 →
  total_team_age = average_team_age * team_size →
  total_remaining_age = average_remaining_age * remaining_player_count →
  253 = captain_age + (captain_age + x) + total_remaining_age →
  x = 1 :=
by
  intros team_size captain_age average_team_age average_remaining_age remaining_player_count total_team_age total_remaining_age W x
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end wicket_keeper_age_difference_l584_584516


namespace number_of_positive_integers_in_T_l584_584924

-- Define the conditions used in the problem statement
def repeating_decimal_condition (n : ℕ) : Prop :=
  ∃ l : ℕ, n > 1 ∧ (∃ d : Fin 18 → ℕ, ∀ i j : ℕ, i % 18 = j % 18 → (i < 18) → d ⟨i % 18, by linarith [show i % 18 < 18 from Nat.mod_ltₓ i (by norm_num)]⟩ = d ⟨j % 18, by linarith [show j % 18 < 18 from Nat.mod_ltₓ j (by norm_num)]⟩)

-- Define the mathematical statement
theorem number_of_positive_integers_in_T (h1 : Prime 9901) (h2 : Prime 10001) :
  ∃ T : Finset ℕ, (∀ n ∈ T, repeating_decimal_condition n) ∧ T.card = 15 := sorry

end number_of_positive_integers_in_T_l584_584924


namespace valid_rearrangements_count_l584_584764

theorem valid_rearrangements_count : 
  let chairs := {1, 2, 3, 4, 5, 6, 7, 8}
  ∃ (f : chairs → chairs), 
    (∀ x, x ≠ f x) ∧
    (∀ x, ¬((x - f x) % 8 = 1 ∨ (f x - x) % 8 = 1)) ∧
    (∀ x, ¬((x - f x) % 8 = 4 ∨ (f x - x) % 8 = 4)) :=
  sorry

end valid_rearrangements_count_l584_584764


namespace intersection_is_2_l584_584451

-- Define the sets A and B
def A : Set ℝ := { x | x < 1 }
def B : Set ℝ := { -1, 0, 2 }

-- Define the complement of A
def A_complement : Set ℝ := { x | x ≥ 1 }

-- Define the intersection of the complement of A and B
def intersection : Set ℝ := A_complement ∩ B

-- Prove that the intersection is {2}
theorem intersection_is_2 : intersection = {2} := by
  sorry

end intersection_is_2_l584_584451


namespace simplify_sqrt_sum_l584_584136

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l584_584136


namespace height_of_pyramid_l584_584717

noncomputable def square_side (perimeter : ℝ) : ℝ := perimeter / 4
noncomputable def half_diagonal (side : ℝ) : ℝ := (side * Real.sqrt 2) / 2
noncomputable def height (distance_to_vertex : ℝ) (half_diagonal : ℝ) : ℝ :=
  Real.sqrt (distance_to_vertex^2 - half_diagonal^2)

theorem height_of_pyramid (perimeter : ℝ) (distance_to_vertex : ℝ) :
  perimeter = 32 ∧ distance_to_vertex = 12 →
  height distance_to_vertex (half_diagonal (square_side perimeter)) = 4 * Real.sqrt 7 :=
by
  -- Mathematical steps are omitted, only the statement and needed conditions are provided
  sorry

end height_of_pyramid_l584_584717


namespace number_of_unique_rectangular_boxes_l584_584749

-- Define the coordinates for the grid points in a 3x3x2 rectangular prism
def grid_points := { (i, j, k) // 0 ≤ i ∧ i ≤ 2 ∧ 0 ≤ j ∧ j ≤ 2 ∧ 0 ≤ k ∧ k ≤ 1 }

-- Total number of unique combinations of forming rectangular boxes
-- from the given grid points
def num_rectangular_boxes : ℕ :=
  (nat.choose 3 2) * (nat.choose 3 2) * (nat.choose 2 2)

-- Theorem stating the number of unique rectangular boxes that can be formed
theorem number_of_unique_rectangular_boxes : num_rectangular_boxes = 9 := by
  sorry

end number_of_unique_rectangular_boxes_l584_584749


namespace solve_quadratic_l584_584976

theorem solve_quadratic :
  ∀ x, (x^2 - x - 12 = 0) → (x = -3 ∨ x = 4) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_l584_584976


namespace probability_between_C_and_D_is_five_eighths_l584_584059

noncomputable def AB : ℝ := 1
def AD : ℝ := AB / 4
def BC : ℝ := AB / 8
def pos_C : ℝ := AB - BC
def pos_D : ℝ := AD
def CD : ℝ := pos_C - pos_D

theorem probability_between_C_and_D_is_five_eighths : CD / AB = 5 / 8 :=
by
  simp [AB, AD, BC, pos_C, pos_D, CD]
  sorry

end probability_between_C_and_D_is_five_eighths_l584_584059


namespace sum_of_coefficients_l584_584921

theorem sum_of_coefficients (p : ℕ) (hp : Nat.Prime p) (a b c : ℕ)
    (f : ℕ → ℕ) (hf : ∀ x, f x = a * x ^ 2 + b * x + c)
    (h_range : 0 < a ∧ a ≤ p ∧ 0 < b ∧ b ≤ p ∧ 0 < c ∧ c ≤ p)
    (h_div : ∀ x, x > 0 → p ∣ (f x)) : 
    a + b + c = 3 * p := 
sorry

end sum_of_coefficients_l584_584921


namespace sum_s_t_k_l584_584550

variables {ℝ : Type} [normed_field ℝ] [normed_space ℝ (euclidean_space ℝ (fin 3))]

-- Problem Definitions and Conditions
variables (u v w : euclidean_space ℝ (fin 3))
variables (s t k : ℝ)

noncomputable def mutually_orthogonal : Prop := 
  (u ⬝ v = 0) ∧ (v ⬝ w = 0) ∧ (w ⬝ u = 0)

noncomputable def unit_vectors : Prop := 
  (∥u∥ = 1) ∧ (∥v∥ = 1) ∧ (∥w∥ = 1)

noncomputable def linear_combination : Prop := 
  u = s • (u ⬝ₐ v) + t • (v ⬝ₐ w) + k • (w ⬝ₐ u)

noncomputable def scalar_triple_product : Prop := 
  u ⬝ (v ⬝ₐ w) = -1

theorem sum_s_t_k (h₁ : mutually_orthogonal u v w) 
                  (h₂ : unit_vectors u v w) 
                  (h₃ : linear_combination u v w s t k) 
                  (h₄ : scalar_triple_product u v w) : 
                  s + t + k = -1 := sorry

end sum_s_t_k_l584_584550


namespace smallest_four_digit_number_l584_584762

theorem smallest_four_digit_number (x : ℕ)
    (h1 : 5 * x ≡ 25 [MOD 20])
    (h2 : 3 * x + 4 ≡ 10 [MOD 7])
    (h3 : -x + 3 ≡ 2 * x [MOD 15]) :
    x = 1021 :=
by
  -- This will be proven here.
  sorry

end smallest_four_digit_number_l584_584762


namespace no_valid_x_for_given_circle_conditions_l584_584886

theorem no_valid_x_for_given_circle_conditions :
  ∀ x : ℝ,
    ¬ ((x - 15)^2 + 18^2 = 225 ∧ (x - 15)^2 + (-18)^2 = 225) :=
by
  sorry

end no_valid_x_for_given_circle_conditions_l584_584886


namespace quadratic_to_square_form_l584_584755

theorem quadratic_to_square_form (x m n : ℝ) (h : x^2 + 6 * x - 1 = 0) 
  (hm : m = 3) (hn : n = 10) : m - n = -7 :=
by 
  -- Proof steps (skipped, as per instructions)
  sorry

end quadratic_to_square_form_l584_584755


namespace karen_paddling_speed_on_still_pond_l584_584016

-- Define the constants and variables
constant river_current_speed : ℝ := 4
constant river_length : ℝ := 12
constant time_to_paddle_up : ℝ := 2

-- Define the effective speed and distances
def effective_speed (v : ℝ) : ℝ := v - river_current_speed
def distance_travelled (v : ℝ) : ℝ := effective_speed(v) * time_to_paddle_up

-- The theorem statement
theorem karen_paddling_speed_on_still_pond (v : ℝ) (hv : distance_travelled(v) = river_length) : v = 10 :=
by
  sorry

end karen_paddling_speed_on_still_pond_l584_584016


namespace sin_60_eq_sqrt3_div_2_l584_584352

theorem sin_60_eq_sqrt3_div_2 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  -- proof skipped
  sorry

end sin_60_eq_sqrt3_div_2_l584_584352


namespace midpoint_x_values_l584_584887

noncomputable def midpoint_x (M A B : ℝ × ℝ) : ℝ :=
  (M.1)

theorem midpoint_x_values :
  ∃ (x_0 : ℝ), (∃ (y_0 : ℝ), 
  (∀ A B : ℝ × ℝ, (A.1 ^ 2 + A.2 ^ 2 = 4) → (B.1 ^ 2 + B.2 ^ 2 = 4) → 
  (dist A B = 2 * real.sqrt 2) →
  (let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in 
  (∃ (P : ℝ × ℝ), P = (3, -1) ∧ 
  (P.1 * (M.1 - P.1) + P.2 * (M.2 - P.2) = 8) → 
  x_0 = midpoint_x M A B))) ∧ (x_0 = 1 ∨ x_0 = 1 / 5)) :=
sorry

end midpoint_x_values_l584_584887


namespace sin_60_eq_sqrt3_div_2_l584_584362

theorem sin_60_eq_sqrt3_div_2 :
  ∃ (Q : ℝ × ℝ), dist Q (1, 0) = 1 ∧ angle (1, 0) Q = real.pi / 3 ∧ Q.2 = real.sqrt 3 / 2 := sorry

end sin_60_eq_sqrt3_div_2_l584_584362


namespace simplify_sqrt_sum_l584_584092

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l584_584092


namespace length_of_AD_l584_584522

theorem length_of_AD {A B C D : Type}
  (dist_AB : ℝ) (dist_BC : ℝ) (dist_CD : ℝ)
  (angle_B : ∠B = 90) (angle_C : ∠C = 90)
  (h_AB : dist_AB = 7) (h_BC : dist_BC = 10) (h_CD : dist_CD = 26) :
  ∃ (AD : ℝ), AD = Real.sqrt 461 :=
by
  sorry

end length_of_AD_l584_584522


namespace smallest_perimeter_of_triangle_PQR_l584_584227

noncomputable def triangle_PQR_perimeter (PQ PR QR : ℕ) (QJ : ℝ) 
  (h1 : PQ = PR) (h2 : QJ = 10) : ℕ :=
2 * (PQ + QR)

theorem smallest_perimeter_of_triangle_PQR (PQ PR QR : ℕ) (QJ : ℝ) :
  PQ = PR → QJ = 10 → 
  ∃ p, p = triangle_PQR_perimeter PQ PR QR QJ (by assumption) (by assumption) ∧ p = 78 :=
sorry

end smallest_perimeter_of_triangle_PQR_l584_584227


namespace range_of_b_l584_584473

section ProofProblem

-- Conditions
variable {x : ℝ} {a b : ℝ}
def f (x : ℝ) (a : ℝ) : ℝ := a * x - log x

theorem range_of_b (h_a : 0 < a)
  (h_f : ∀ x > 0, f x a ≥ a^2 / 2 + b) :
  b ≤ 1 / 2 := sorry

end ProofProblem

end range_of_b_l584_584473


namespace sqrt_sum_l584_584108

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l584_584108


namespace solve_arcsin_eq_l584_584142

noncomputable def problem_statement (x : ℝ) : Prop :=
  arcsin (2 * x / sqrt 15) + arcsin (3 * x / sqrt 15) = arcsin (4 * x / sqrt 15) 

theorem solve_arcsin_eq (x : ℝ) (h1 : |x| ≤ sqrt 15 / 4) 
  (h2 : problem_statement x) : 
  x = 0 ∨ x = 15 / 16 ∨ x = -15 / 16 :=
sorry

end solve_arcsin_eq_l584_584142


namespace common_limit_geometric_mean_sequences_coincide_l584_584040

-- Definitions based on conditions
def sequence_a : ℕ → ℝ → ℝ → ℝ
| 0, a, _ => a
| n + 1, a, b => (2 * sequence_a n a b * (sequence_b n a b)) / (sequence_a n a b + sequence_b n a b)

def sequence_b : ℕ → ℝ → ℝ → ℝ
| 0, _, b => b
| n + 1, a, b => (sequence_a n a b + sequence_b n a b) / 2

-- Part (a): Prove both sequences have a common limit
theorem common_limit {a b : ℝ} (h0 : 0 < a) (h1 : a < b) :
  ∃ L : ℝ, tendsto (λ n, sequence_a n a b) at_top (𝓝 L) ∧ tendsto (λ n, sequence_b n a b) at_top (𝓝 L) := 
sorry

-- Part (b): Prove the common limit is the geometric mean of a and b
theorem geometric_mean {a b : ℝ} (h0 : 0 < a) (h1 : a < b) :
  ∃ L: ℝ, L = real.sqrt (a * b) ∧ tendsto (λ n, sequence_a n a b) at_top (𝓝 L) ∧ tendsto (λ n, sequence_b n a b) at_top (𝓝 L) :=
sorry

-- Part (c): Relate sequence {bn} with {xn} from another problem for specific a and b values
noncomputable def sequence_x (n : ℕ) (k : ℝ) : ℝ := 
  (sequence_b n 1 k)

theorem sequences_coincide (k : ℝ) :
  ∀ n, sequence_x n k = sequence_b n 1 k :=
sorry

end common_limit_geometric_mean_sequences_coincide_l584_584040


namespace sin_60_proof_l584_584371

noncomputable def sin_60_eq : Prop :=
  sin (60 * real.pi / 180) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq :=
sorry

end sin_60_proof_l584_584371


namespace standard_equation_ellipse_exists_line_l_l584_584816

-- Define the ellipse and its properties
def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | p.1^2 / a^2 + p.2^2 / b^2 = 1}

def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Define the given conditions
variables {a b c : ℝ}
variables {A B : ℝ × ℝ}
variables (O : ℝ × ℝ := (0, 0))
variables (e : ℝ := 1/2)

-- Given conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom a_gt_b : a > b
axiom A_def : A = (2, 3)
axiom B_def : B = (0, -4)
axiom ecc_def : eccentricity a c = 1/2
axiom ellipse_contains_A : (2, 3) ∈ ellipse a b

-- Prove the standard equation of the ellipse
theorem standard_equation_ellipse (a b c : ℝ) (A : ℝ × ℝ)
  (ecc_def : eccentricity a c = 1/2)
  (A_def : A = (2, 3))
  (ellipse_contains_A : A ∈ ellipse a b)
  (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) :
  ellipse a b = {p : ℝ × ℝ | p.1^2 / 16 + p.2^2 / 12 = 1} :=
sorry

-- Prove the existence of line l and find its equation
theorem exists_line_l
  (a b c : ℝ) (B : ℝ × ℝ)
  (O : ℝ × ℝ := (0, 0))
  (A_def : A = (2, 3))
  (B_def : B = (0, -4))
  (ecc_def : eccentricity a c = 1/2)
  (ellipse_contains_A : A ∈ ellipse a b)
  (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
  (dot_product_condition : ∀ M N : ℝ × ℝ, M ∈ ellipse a b → N ∈ ellipse a b →
      (M.1 * N.1 + M.2 * N.2 = 16 / 7) → true ) :
  ∃ (k : ℝ), (B.2 = k * B.1 + 4) ∧
  (∀ M N : ℝ × ℝ, M ∈ ellipse a b → N ∈ ellipse a b →
    (M.1 * N.1 + M.2 * N.2 = 16 / 7) →
    ∃ l : ℝ, M.2 = l * M.1 + 4) ∧
  (∀ M N : ℝ × ℝ, M ∈ ellipse a b → N ∈ ellipse a b →
    (M.1 * N.1 + M.2 * N.2 = 16 / 7) →
    ∃ l : ℝ, N.2 = l * N.1 + 4) :=
sorry

end standard_equation_ellipse_exists_line_l_l584_584816


namespace combined_average_rainfall_is_correct_l584_584637

noncomputable def virginia_rainfall := [3.79, 4.5, 3.95, 3.09, 4.67]
noncomputable def maryland_rainfall := [3.99, 4.0, 4.25, 3.5, 4.9]
noncomputable def nc_rainfall := [4.1, 4.4, 4.2, 4.0, 5.0]

noncomputable def total_rainfall (rainfalls : List ℝ) : ℝ :=
  rainfalls.foldr (.+.) 0

noncomputable def average_rainfall (rainfalls : List ℝ) : ℝ :=
  total_rainfall rainfalls / rainfalls.length

noncomputable def combined_average_rainfall (lists : List (List ℝ)) : ℝ :=
  let averages := lists.map average_rainfall
  averages.foldr (.+.) 0 / averages.length

theorem combined_average_rainfall_is_correct :
  combined_average_rainfall [virginia_rainfall, maryland_rainfall, nc_rainfall] = 4.156 :=
by
  sorry

end combined_average_rainfall_is_correct_l584_584637


namespace collinear_points_l584_584639

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem collinear_points {a b : ℝ} (h : slope (5, -3) (-a + 4, b) = slope (5, -3) (3a + 4, b - 1)) :
  a = -1/14 ∧ b = 1 :=
by
  sorry

end collinear_points_l584_584639


namespace range_of_a_l584_584468

def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then 2*x^3 + 3*x^2 + 1 else Real.exp (a * x)

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 2) → a ∈ Set.Iic (1/2 * Real.log 2) :=
by
  sorry

end range_of_a_l584_584468


namespace charity_tickets_solution_l584_584704

theorem charity_tickets_solution (f h d p : ℕ) (ticket_count : f + h + d = 200)
  (revenue : f * p + h * (p / 2) + d * (2 * p) = 3600) : f = 80 := by
  sorry

end charity_tickets_solution_l584_584704


namespace sum_of_squares_of_projections_l584_584966

theorem sum_of_squares_of_projections (n : ℕ) (a : ℝ) (e : Finₓ n → ℝ × ℝ) :
  let projections := fun x : ℝ × ℝ => dot e x in
  ∑ i in Finₓ.range n, (projections a i)^2 = (n : ℝ) * a^2 / 2 :=
by
  sorry

end sum_of_squares_of_projections_l584_584966


namespace median_of_consecutive_integers_l584_584184

theorem median_of_consecutive_integers (m : ℤ) (h : (∑ k in (finset.range 81).image (λ x, x - 40 + m), k) = 9^5) :
  m = 729 :=
by
  sorry

end median_of_consecutive_integers_l584_584184


namespace unique_determination_l584_584562

noncomputable def is_lipschitz {α : Type*} [PseudoMetricSpace α] (f : α → ℝ) (K : ℝ) : Prop :=
∀ x y, abs (f x - f y) ≤ K * dist x y

noncomputable def varphi (X : ℝ → ℝ) (x : ℝ) : ℝ := ∫ (t : ℝ), abs (X t - x)

axiom existence_expectation (X : ℝ → ℝ) (hX : ∫ (t : ℝ), abs (X t) < ∞) : ∃ μ : ℝ, ∀ x : ℝ, ∫ (t : ℝ), abs (X t - x) = varphi X x

theorem unique_determination (X : ℝ → ℝ) (hX : ∫ (t : ℝ), abs (X t) < ∞) :
  ∃! (F : ℝ → ℝ), ∀ x : ℝ, F x = (1 / 2) * (varphi X x)' + 1 :=
sorry

end unique_determination_l584_584562


namespace arrangement_problem_l584_584881

-- Definitions based on conditions
def total_students : ℕ := 5
def total_arrangements : ℕ := Nat.factorial total_students

-- Definitions for restricted cases
def blocks : ℕ := 4
def restricted_arrangements : ℕ := (Nat.factorial blocks) * 2

-- Correct answer calculation: Total arrangements minus restricted cases
def allowed_arrangements : ℕ := total_arrangements - restricted_arrangements

-- Lean statement to prove the problem
theorem arrangement_problem (total_students = 5) (blocks = 4) :
  allowed_arrangements = 72 := by
  sorry -- Proof is skipped

end arrangement_problem_l584_584881


namespace median_of_81_consecutive_integers_l584_584181

theorem median_of_81_consecutive_integers (ints : ℕ → ℤ) 
  (h_consecutive: ∀ n : ℕ, ints (n+1) = ints n + 1) 
  (h_sum: (∑ i in finset.range 81, ints i) = 9^5) : 
  (ints 40) = 9^3 := 
sorry

end median_of_81_consecutive_integers_l584_584181


namespace minimum_pigs_l584_584052

theorem minimum_pigs (P T : ℕ) (h_cond1 : 0.54 * T ≤ P) (h_cond2 : P ≤ 0.57 * T) : P ≥ 11 :=
by
  -- placeholder for the proof
  sorry

end minimum_pigs_l584_584052


namespace binomial_sum_mod_3_l584_584962

open BigOperators

theorem binomial_sum_mod_3 :
  let n := 6002
  ∑ k in finset.Ico (0 : ℕ) (n / 6), nat.choose n (k * 6 + 1) % 3 = 1 :=
by
  sorry

end binomial_sum_mod_3_l584_584962


namespace aunt_masha_butter_usage_l584_584745

theorem aunt_masha_butter_usage
  (x y : ℝ)
  (h1 : x + 10 * y = 600)
  (h2 : x = 5 * y) :
  (2 * x + 2 * y = 480) := 
by
  sorry

end aunt_masha_butter_usage_l584_584745


namespace binom_15_12_l584_584342

theorem binom_15_12 : nat.choose 15 12 = 455 :=
by
  sorry

end binom_15_12_l584_584342


namespace smallest_side_of_triangle_min_sum_B_l584_584870

variables {A B C D E : Type} [IsTriangle ABC] (A : Point) (B : Point) (C : Point) (D : Point) (E : Point)
variables (b c : ℝ) [has_sum2 (div3 7 c) (mul3 div7 c c) 4 e ac ] (x : ℝ)

-- Given conditions
def angle_bisector (AD : A bisect B) : find_length := sorry
def divides_angle (AE : A divides BAD B) (DE EC : property) : find_length := sorry
def update_AD : AD := sorry
def update_AE : AE := sorry
def Geometry (c : ℝ) [has_sum (my_new_function x) = c := sorry
def Update : Update := sorry
def find_length (x : Type) := sorry
def smallest_side {AB b} : min_sum b := sorry

theorem smallest_side_of_triangle (ad : AD bisect B) (e : AE divides BAD) (d : divide_angle) (de de) :
  side_length ∈ [3,4,3,7]  b (get_d AC) min_sum (match a with h := x )

theorem min_sum_B (A : Point) (B : Angle_bisector in geometry_perimeters) ( smallest_a := my_new_function) 
match (e := min_sum_B  of   find_length  : begin
intro,
end

end smallest_side_of_triangle_min_sum_B_l584_584870


namespace mall_discount_l584_584712

theorem mall_discount (total_price : ℝ) (p1 p2 : ℝ) : 
  p1 = 148 -> p2 = 423 -> total_price = p1 + p2 :=
begin
  intro h1,
  intro h2,
  rw [h1, h2],
  let total = 148 + (423 / 0.9),
  have htot : total = 618,
  {
    -- Calculating 423 / 0.9
    sorry
  },
  have final_price : total - 118 = 500,
  {
    rw total,
    -- 618 - 118 = 500
    sorry
  },
  have discounted_total : 500 * 0.9 + 118 * 0.8 = 544.4,
  {
    -- Calculate both discounted amounts and sum them
    sorry
  },
  rw discounted_total,
  exact 544.4,
  sorry
end

end mall_discount_l584_584712


namespace circle_radius_correct_l584_584466

noncomputable def circle_radius (x y : ℝ) : ℝ := sqrt 9

theorem circle_radius_correct :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y - 4 = 0) → (circle_radius x y = 3) :=
by
  sorry

end circle_radius_correct_l584_584466


namespace triple_composition_l584_584857

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition :
  g (g (g 3)) = 107 :=
by
  sorry

end triple_composition_l584_584857


namespace sum_first_five_terms_l584_584877

-- Definitions related to the problem
def a : ℕ → ℝ := sorry  -- the sequence a_n
def S (n : ℕ) : ℝ := sorry  -- the sum of the first n terms

-- Conditions of the problem
axiom h1 : a 1 = 1
axiom h2 (n : ℕ) : S n = ∑ i in range n, a (i+1)
axiom h3 : (1/a 1) - (1/a 2) = (2/a 3)

-- Theorem to prove
theorem sum_first_five_terms : S 5 = 31 :=
by
  sorry  -- proof is not required

end sum_first_five_terms_l584_584877


namespace sqrt_sum_l584_584111

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l584_584111


namespace perpendicular_dot_product_zero_max_length_l584_584843

variables (α β : ℝ) 
noncomputable def a : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def b : ℝ × ℝ := (Real.cos β, Real.sin β)
noncomputable def c : ℝ × ℝ := (-1, 0)
noncomputable def bs : ℝ × ℝ := (Real.cos β - 1, Real.sin β)
noncomputable def maximum_length : ℝ := 2

theorem perpendicular_dot_product_zero (hα : α = Real.pi / 4) 
  (h_perp : (Real.cos α * (Real.cos β - 1) + Real.sin α * Real.sin β) = 0) :
  Real.cos β = 0 ∨ Real.cos β = 1 := sorry

theorem max_length (hβ : -1 ≤ Real.cos β ∧ Real.cos β ≤ 1) :
  ∥bs β∥ = maximum_length := sorry

end perpendicular_dot_product_zero_max_length_l584_584843


namespace distance_from_foci_to_asymptotes_of_hyperbola_l584_584616

theorem distance_from_foci_to_asymptotes_of_hyperbola :
  let c := real.sqrt 2
  let foci := [(-c, 0), (c, 0)]
  let asymptotes := [λ x y : ℝ, x + y = 0, λ x y : ℝ, x - y = 0]
  in ∀ p ∈ foci, ∀ l ∈ asymptotes, distance p l = 1
:= 
begin
  sorry
end

end distance_from_foci_to_asymptotes_of_hyperbola_l584_584616


namespace smallest_angle_convex_15_polygon_l584_584983

theorem smallest_angle_convex_15_polygon :
  ∃ (a : ℕ) (d : ℕ), (∀ n : ℕ, n ∈ Finset.range 15 → (a + n * d < 180)) ∧
  15 * (a + 7 * d) = 2340 ∧ 15 * d <= 24 -> a = 135 :=
by
  -- Proof omitted
  sorry

end smallest_angle_convex_15_polygon_l584_584983


namespace expression_for_fx_pos_l584_584454

variable {R : Type} [Real R]

-- Define the function f with conditions given
def f (x : R) : R := if x ≤ 0 then log (1 / 2) (-x + 1) else sorry

-- State the condition that f is an odd function
def odd_function (f : R → R) := ∀ x, f (-x) = -f x

-- Given conditions
axiom f_odd : odd_function f
axiom f_def_neg : ∀ x, x ≤ 0 → f x = log (1 / 2) (-x + 1)

-- The main question to be proved
theorem expression_for_fx_pos (x : R) (h : x > 0) : f x = - log (1 / 2) (x + 1) :=
sorry

end expression_for_fx_pos_l584_584454


namespace mod_residue_l584_584345

theorem mod_residue : (250 * 15 - 337 * 5 + 22) % 13 = 7 := by
  sorry

end mod_residue_l584_584345


namespace min_distinct_midpoints_l584_584643

theorem min_distinct_midpoints :
  ∀ (P : Fin 2017 → ℝ × ℝ), ∃! (MP : Finset (ℝ × ℝ)), 
  (∀ i j, i ≠ j → ((P i, P j) ∈ MP)) ∧ MP.card = 2016 :=
by
  -- let P be any 2017 distinct points in the plane
  intros P,
  -- find the unique set MP of distinct midpoints such that
  -- for all pairs (P_i, P_j) with i ≠ j, the midpoint (P_i + P_j) / 2 is an element of MP
  -- and the cardinality of MP is 2016
  -- sorry is used to represent the proof placeholder.
  sorry

end min_distinct_midpoints_l584_584643


namespace median_of_81_consecutive_integers_l584_584191

theorem median_of_81_consecutive_integers (sum : ℕ) (h₁ : sum = 9^5) : 
  let mean := sum / 81 in mean = 729 :=
by
  have h₂ : sum = 59049 := by
    rw h₁
    norm_num
  have h₃ : mean = 59049 / 81 := by
    rw h₂
    rfl
  have result : mean = 729 := by
    rw h₃
    norm_num
  exact result

end median_of_81_consecutive_integers_l584_584191


namespace sequence_general_term_and_sum_l584_584463

-- Given: The sum of the first n terms of the sequence {a_n} is S_n = n² - 2n
-- Prove: The general term formula a_n = 2n - 3
-- Prove: If b_n = a_n / 3^n, then the sum T_n of the first n terms of the sequence {b_n} is T_n = -n / 3^n.

theorem sequence_general_term_and_sum (S : ℕ → ℤ) (a b : ℕ → ℤ) (T : ℕ → ℚ) :
  (∀ n, S n = n^2 - 2 * n) →
  (∀ n, a 1 = -1 ∧ (n ≥ 2 → a n = S n - S (n - 1))) →
  (∀ n, b n = a n / 3^n) →
  (∀ n, T 1 = -1 / 3 ∧ ∀ n ≥ 2, T n = ∑ i in finset.range (n+1), b i) →
  (∀ n, a n = 2 * n - 3) ∧
  (∀ n, T n = -n / 3^n) :=
begin
  intros,
  sorry
end

end sequence_general_term_and_sum_l584_584463


namespace sampling_is_systematic_l584_584705

-- Define the total seats in each row and the total number of rows
def total_seats_per_row : ℕ := 25
def total_rows : ℕ := 30

-- Define a function to identify if the sampling is systematic
def is_systematic_sampling (sample_count : ℕ) (n : ℕ) (interval : ℕ) : Prop :=
  interval = total_seats_per_row ∧ sample_count = total_rows

-- Define the count and interval for the problem
def sample_count : ℕ := 30
def sampling_interval : ℕ := 25

-- Theorem statement: Given the conditions, it is systematic sampling
theorem sampling_is_systematic :
  is_systematic_sampling sample_count total_rows sampling_interval = true :=
sorry

end sampling_is_systematic_l584_584705


namespace convert_base8_to_base7_l584_584390

theorem convert_base8_to_base7 (n : ℕ) : n = 536 → (num_to_base7 536) = 1054 :=
by
  sorry

def num_to_base10 (n : ℕ) : ℕ :=
  let d2 := (n / 100) % 10 * 8^2
  let d1 := (n / 10) % 10 * 8^1
  let d0 := (n / 1) % 10 * 8^0
  d2 + d1 + d0

def num_to_base7_aux (n : ℕ) (acc : ℕ) (pos : ℕ) : ℕ :=
  if n = 0 then acc
  else
    let q := n / 7
    let r := n % 7
    num_to_base7_aux q ((r * 10^pos) + acc) (pos + 1)

def num_to_base7 (n : ℕ) : ℕ :=
  num_to_base7_aux (num_to_base10 n) 0 0

end convert_base8_to_base7_l584_584390


namespace k_value_for_inequality_l584_584413

theorem k_value_for_inequality :
    (∀ a b c d : ℝ, a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 → a^3 + b^3 + c^3 + d^3 + 1 ≥ (3/4) * (a + b + c + d)) ∧
    (∀ k : ℝ, (∀ a b c d : ℝ, a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 → a^3 + b^3 + c^3 + d^3 + 1 ≥ k * (a + b + c + d)) → k = 3/4) :=
sorry

end k_value_for_inequality_l584_584413


namespace magazines_sold_l584_584916

theorem magazines_sold (t n m : Real) (h1 : t = 425.0) (h2 : n = 275.0) (h3 : m = t - n) : m = 150.0 := 
by 
  rw [h1, h2] at h3 
  norm_num at h3
  exact h3

end magazines_sold_l584_584916


namespace daily_profit_1200_impossible_daily_profit_1600_l584_584702

-- Definitions of given conditions
def avg_shirts_sold_per_day : ℕ := 30
def profit_per_shirt : ℕ := 40

-- Function for the number of shirts sold given a price reduction
def shirts_sold (x : ℕ) : ℕ := avg_shirts_sold_per_day + 2 * x

-- Function for the profit per shirt given a price reduction
def new_profit_per_shirt (x : ℕ) : ℕ := profit_per_shirt - x

-- Function for the daily profit given a price reduction
def daily_profit (x : ℕ) : ℕ := (new_profit_per_shirt x) * (shirts_sold x)

-- Proving the desired conditions in Lean

-- Part 1: Prove that reducing the price by 25 yuan results in a daily profit of 1200 yuan
theorem daily_profit_1200 (x : ℕ) : daily_profit x = 1200 ↔ x = 25 :=
by
  { sorry }

-- Part 2: Prove that a daily profit of 1600 yuan is not achievable
theorem impossible_daily_profit_1600 (x : ℕ) : daily_profit x ≠ 1600 :=
by
  { sorry }

end daily_profit_1200_impossible_daily_profit_1600_l584_584702


namespace no_primes_in_list_count_primes_in_list_l584_584487

theorem no_primes_in_list {n : ℕ} (hn : n ≥ 1) : 
  ∀ k, k ∈ [1, 2, ..., n] → 
  ¬ prime (57 * (10^(2*(k-1)) + 10^(2*(k-2)) + ... + 10^2 + 1)) := 
by sorry

theorem count_primes_in_list (n : ℕ) (h : n ≥ 1) : 
  ∑ k in (finset.range n), 0 = 0 := 
by sorry

end no_primes_in_list_count_primes_in_list_l584_584487


namespace product_of_solutions_l584_584421

theorem product_of_solutions (x : ℝ) (h : x^2 = 81) :
  ∃ a b : ℝ, (a^2 = 81 ∧ b^2 = 81) ∧ a * b = -81 :=
by
  use [9, -9]
  split
  { split
    { exact rfl }
    { exact rfl } }
  calc 9 * -9 = -81 : by norm_num

end product_of_solutions_l584_584421


namespace calculate_ab_l584_584210

theorem calculate_ab {a b c : ℝ} (hc : c ≠ 0) (h1 : (a * b) / c = 4) (h2 : a * (b / c) = 12) : a * b = 12 :=
by
  sorry

end calculate_ab_l584_584210


namespace center_of_gravity_ellipse_l584_584416

noncomputable def center_of_gravity (a b : ℝ) : ℝ × ℝ :=
  let p : ℝ → ℝ → ℝ := λ x y, x * y
  let Ω := {p : ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ (p.1^2 / a^2 + p.2^2 / b^2) ≤ 1}
  let M := ∫ (x : ℝ), ∫ (y : ℝ) in 0..b * Real.sqrt (1 - x^2 / a^2), p x y
  let m_y := ∫ (x : ℝ), ∫ (y : ℝ) in 0..b * Real.sqrt (1 - x^2 / a^2), x * p x y
  let m_x := ∫ (x : ℝ), ∫ (y : ℝ) in 0..b * Real.sqrt (1 - x^2 / a^2), y * p x y
  (m_y / M, m_x / M)

theorem center_of_gravity_ellipse (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  center_of_gravity a b = (8*a/15, 8*b/15) :=
sorry

end center_of_gravity_ellipse_l584_584416


namespace reeya_weighted_average_l584_584599

noncomputable def weighted_average (scores : List ℚ) (weights : List ℚ) : ℚ :=
  let weighted_sum := (List.zip scores weights).foldl (fun acc p => acc + (p.fst * p.snd)) 0
  let sum_weights := weights.foldl ( + ) 0
  weighted_sum / sum_weights

theorem reeya_weighted_average :
  weighted_average [65, 67, 76, 82, 85] [2, 3, 1, 4, 1.5] = 75 :=
by
  sorry

end reeya_weighted_average_l584_584599


namespace sum_T_lt_half_l584_584442

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))
noncomputable def T (n : ℕ) : ℚ := ∑ k in finset.range n, b (k + 1)

theorem sum_T_lt_half (n : ℕ) : T n < 1 / 2 := 
by sorry

end sum_T_lt_half_l584_584442


namespace smallest_possible_X_l584_584022

noncomputable def T := 11100
noncomputable def X := T / 12

theorem smallest_possible_X : 
  (T % 12 = 0) ∧ (T % 10 / 4 = 0) ∧ (∀ digit ∈ T.digits, digit = 0 ∨ digit = 1) ∧ T.digits.sum % 3 = 0 → X = 925 :=
by
  sorry

end smallest_possible_X_l584_584022


namespace lines_intersect_at_point_l584_584403

theorem lines_intersect_at_point :
  ∃ (x y : ℝ), (3 * x + 4 * y + 7 = 0) ∧ (x - 2 * y - 1 = 0) ∧ (x = -1) ∧ (y = -1) :=
by
  sorry

end lines_intersect_at_point_l584_584403


namespace remainder_of_q1_div_x_minus_1_l584_584938

-- Define the polynomial x^5
def f (x : ℝ) : ℝ := x^5

-- Define q1(x) which is the quotient obtained when f(x) is divided by (x - 1/2)
noncomputable def q1 (x : ℝ) : ℝ := x^4 + (1/2)*x^3 + (1/4)*x^2 + (1/8)*x + (1/16)

-- Define remainder r1 when f(x) is divided by (x - 1/2)
def r1 : ℝ := (1/2)^5

-- Prove that the remainder when q1(x) is divided by x - 1 is 2.9375
theorem remainder_of_q1_div_x_minus_1 : 
  q1 1 = 2.9375 := by
    sorry

end remainder_of_q1_div_x_minus_1_l584_584938


namespace distance_from_origin_to_M_l584_584835

-- Declare the points A and B
def A : ℝ × ℝ := (-1, 5)
def B : ℝ × ℝ := (3, -7)

-- Definition of the midpoint of two points
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Calculate the midpoint of A and B
def M : ℝ × ℝ := midpoint A B

-- Distance formula between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Prove the distance from origin to the midpoint of A and B is sqrt(2)
theorem distance_from_origin_to_M :
  distance (0, 0) M = Real.sqrt 2 := by
  sorry

end distance_from_origin_to_M_l584_584835


namespace servings_left_proof_l584_584763

-- Define the number of servings prepared
def total_servings : ℕ := 61

-- Define the number of guests
def total_guests : ℕ := 8

-- Define the fraction of servings the first 3 guests shared
def first_three_fraction : ℚ := 2 / 5

-- Define the fraction of servings the next 4 guests shared
def next_four_fraction : ℚ := 1 / 4

-- Define the number of servings consumed by the 8th guest
def eighth_guest_servings : ℕ := 5

-- Total consumed servings by the first three guests (rounded down)
def first_three_consumed := (first_three_fraction * total_servings).floor

-- Total consumed servings by the next four guests (rounded down)
def next_four_consumed := (next_four_fraction * total_servings).floor

-- Total consumed servings in total
def total_consumed := first_three_consumed + next_four_consumed + eighth_guest_servings

-- The number of servings left unconsumed
def servings_left_unconsumed := total_servings - total_consumed

-- The theorem stating there are 17 servings left unconsumed
theorem servings_left_proof : servings_left_unconsumed = 17 := by
  sorry

end servings_left_proof_l584_584763


namespace perimeter_PXY_equals_35_l584_584527

-- Define the structure of the triangle and the points
structure Triangle :=
  (P Q R : Point)
  (PQ : line_segment P Q)
  (PR : line_segment P R)
  (QR : line_segment Q R)

-- Define the constraints on the side lengths
def side_lengths (T : Triangle) : Prop :=
  T.PQ.length = 15 ∧ T.PR.length = 20 ∧ T.QR.length = 25

-- Define the incenter and parallel line intersection points
def incenter_parallel (T : Triangle) : Point × Point :=
  let I := incenter T.P T.Q T.R in
  let (X, Y) := (intersection (parallel QR I) PQ, intersection (parallel QR I) PR) in
  (X, Y)

-- Define the triangle PXY formed by the intersections
def Triangle_PXY (T : Triangle) : Triangle :=
  let (X, Y) := incenter_parallel T in
  Triangle.mk T.P X Y (line_segment.from_points T.P X) (line_segment.from_points T.P Y) (line_segment.from_points X Y)

-- Lean 4 theorem statement
theorem perimeter_PXY_equals_35 (T : Triangle) 
  (h1 : side_lengths T) :
  let PXY := Triangle_PXY T in
  PXY.PQ.length + PXY.PR.length + PXY.QR.length = 35 :=
sorry

end perimeter_PXY_equals_35_l584_584527


namespace sin_60_proof_l584_584373

noncomputable def sin_60_eq_sqrt3_div_2 : Prop :=
  Real.sin (π / 3) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq_sqrt3_div_2 :=
sorry

end sin_60_proof_l584_584373


namespace trains_cross_time_opposite_directions_l584_584655

noncomputable def length_first_train : ℝ := 150
noncomputable def length_second_train : ℝ := 200
noncomputable def time_first_cross_post : ℝ := 10
noncomputable def time_second_cross_post : ℝ := 12
noncomputable def speed_first_train : ℝ := 100
noncomputable def speed_second_train : ℝ := 120

theorem trains_cross_time_opposite_directions :
  let relative_speed := speed_first_train + speed_second_train,
      combined_length := length_first_train + length_second_train in
  (combined_length / relative_speed) = 1.5909 :=
by
  -- all values used are from the conditions given
  let relative_speed := 100 + 120
  let combined_length := 150 + 200
  have h: ((150 + 200) / (100 + 120)) = 1.5909 := by sorry
  exact h

end trains_cross_time_opposite_directions_l584_584655


namespace appetizer_cost_per_person_l584_584429

theorem appetizer_cost_per_person
    (cost_per_bag: ℕ)
    (num_bags: ℕ)
    (cost_creme_fraiche: ℕ)
    (cost_caviar: ℕ)
    (num_people: ℕ)
    (h1: cost_per_bag = 1)
    (h2: num_bags = 3)
    (h3: cost_creme_fraiche = 5)
    (h4: cost_caviar = 73)
    (h5: num_people = 3):
    (cost_per_bag * num_bags + cost_creme_fraiche + cost_caviar) / num_people = 27 := 
  by
    sorry

end appetizer_cost_per_person_l584_584429


namespace intersection_A_B_l584_584450

def A : Set ℝ := { x | |x - 1| < 2 }
def B : Set ℝ := { x | Real.log x / Real.log 2 ≤ 1 }

theorem intersection_A_B :
  A ∩ B = {x | 0 < x ∧ x ≤ 2} := 
sorry

end intersection_A_B_l584_584450


namespace range_of_a_l584_584865

-- Define the function f(x)
def f (a x : ℝ) : ℝ := log a (a * x^2 - 4 * x + 9)

-- Define the interval [1, 3]
def interval := Icc 1 3

-- Define the condition for function g(x) = ax^2 - 4x + 9 to be positive in [1, 3]
def g_pos (a x : ℝ) : Prop := a * x^2 - 4 * x + 9 > 0

-- State the theorem
theorem range_of_a (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) :
  (∀ x ∈ interval, g_pos a x) →
  (strict_mono_on (f a) interval ↔ a ∈ (Ioc (1/3 : ℝ) (2/3)) ∪ Ici (2 : ℝ)) :=
by sorry

end range_of_a_l584_584865


namespace quadratic_root_implies_q_value_l584_584528

theorem quadratic_root_implies_q_value :
  (∃ (p q : ℝ), ∀ (x : ℂ), 3 * x^2 + p * x + q = 0 → (x = 4 + complex.i ∨ x = 4 - complex.i)) → q = -51 :=
by
  sorry

end quadratic_root_implies_q_value_l584_584528


namespace count_two_digit_markers_l584_584615

def uses_two_digits (n : Nat) : Prop :=
  let ds := (n.toString.toList.eraseDuplicates.length)
  ds ≤ 2

theorem count_two_digit_markers :
  (Finset.filter (λ k => uses_two_digits k ∧ uses_two_digits (999 - k)) (Finset.range 1000)).card = 40 :=
by 
  sorry

end count_two_digit_markers_l584_584615


namespace simplify_sqrt_sum_l584_584133

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l584_584133


namespace percentage_problem_l584_584496

theorem percentage_problem (x : ℝ) (h : 0.25 * x = 0.12 * 1500 - 15) : x = 660 :=
by 
  sorry

end percentage_problem_l584_584496


namespace alien_run_time_l584_584648

variable (v_r v_f : ℝ) -- velocities in km/h
variable (T_r T_f : ℝ) -- time in hours
variable (D_r D_f : ℝ) -- distances in kilometers

theorem alien_run_time :
  v_r = 15 ∧ v_f = 10 ∧ (T_f = T_r + 0.5) ∧ (D_r = D_f) ∧ (D_r = v_r * T_r) ∧ (D_f = v_f * T_f) → T_f = 1.5 :=
by
  intros h
  rcases h with ⟨_, _, _, _, _, _⟩
  -- proof goes here
  sorry

end alien_run_time_l584_584648


namespace solve_inequality_l584_584144

theorem solve_inequality (a x : ℝ) (ha : a ≠ 0) :
  (a > 0 → (x^2 - 5 * a * x + 6 * a^2 > 0 ↔ (x < 2 * a ∨ x > 3 * a))) ∧
  (a < 0 → (x^2 - 5 * a * x + 6 * a^2 > 0 ↔ (x < 3 * a ∨ x > 2 * a))) :=
by
  sorry

end solve_inequality_l584_584144


namespace coefficient_x_squared_l584_584418

def p (a : ℝ) : Polynomial ℝ :=
  C (2 * a) * X ^ 3 + C 5 * X ^ 2 - C 3 * X

def q (b : ℝ) : Polynomial ℝ :=
  C (3 * b) * X ^ 2 - C 8 * X - C 5

theorem coefficient_x_squared (a b : ℝ) :
  (p a * q b).coeff 2 = -1 :=
by
  sorry

end coefficient_x_squared_l584_584418


namespace median_of_81_consecutive_integers_l584_584175

theorem median_of_81_consecutive_integers (n : ℕ) (S : ℕ) (h1 : n = 81) (h2 : S = 9^5) : 
  let M := S / n in M = 729 :=
by
  sorry

end median_of_81_consecutive_integers_l584_584175


namespace problem_correct_statements_l584_584161

/-- Conditions given:
    length = 3,
    3 > b > c,
    surface area of circumscribed sphere = 14π,
    volume = 6
    Prove correct statements: A: (3, 2, 1), D: m^3 = 36√2 -/
theorem problem_correct_statements :
  ∃ (b c : ℝ),
    3 > b ∧ b > c ∧
    3 * b * c = 6 ∧
    3 ^ 2 + b ^ 2 + c ^ 2 = 14 ∧
    (∀ m, (m^3 = 36 * real.sqrt 2) → b = 2 ∧ c = 1 ∧
     ¬ (sqrt ((2 + 3) ^ 2 + 1 ^ 2) = 2 * sqrt 5 ) ∧
     ¬ ( sqrt (( 1 + 3 ) ^ 2 + 2 ^ 2 ) = 2 ) ) :=
sorry

end problem_correct_statements_l584_584161


namespace storage_device_data_transfer_l584_584530

theorem storage_device_data_transfer:
  ∀ (N : ℕ), N ≥ 2 → 
    let A₀ := 0
        B₀ := 0
        C₀ := 0
        A₁ := N
        B₁ := N
        C₁ := N
        A₂ := A₁ - 2
        B₂ := B₁ + 2
        C₂ := C₁
        A₃ := A₂
        B₃ := B₂ + 1
        C₃ := C₂ - 1
        A₄ := 2 * A₃
        B₄ := B₃ - A₃
        C₄ := C₃ 
    in B₄ = 5 :=
by intro N hN;
    let A₀ := 0;
    let B₀ := 0;
    let C₀ := 0;
    let A₁ := N;
    let B₁ := N;
    let C₁ := N;
    let A₂ := A₁ - 2;
    let B₂ := B₁ + 2;
    let C₂ := C₁;
    let A₃ := A₂;
    let B₃ := B₂ + 1;
    let C₃ := C₂ - 1;
    let A₄ := 2 * A₃;
    let B₄ := B₃ - A₃;
    let C₄ := C₃;
    sorry

end storage_device_data_transfer_l584_584530


namespace sum_of_digits_of_B_is_7_l584_584560

theorem sum_of_digits_of_B_is_7 : 
  let A := 16 ^ 16
  let sum_digits (n : ℕ) : ℕ := n.digits 10 |>.sum
  let S := sum_digits
  let B := S (S A)
  sum_digits B = 7 :=
sorry

end sum_of_digits_of_B_is_7_l584_584560


namespace sqrt_sum_l584_584114

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l584_584114


namespace correct_answer_none_of_these_l584_584566

noncomputable def right_triangle_area_ratio (h a r : ℝ) : Prop :=
  let b := real.sqrt (h^2 - a^2) in
  let A := 1/2 * a * b in
  let s := (a + b + h) / 2 in
  let r_calc := a * b / (a + b + h) in
  let ratio := (π * r^2) / A in
  r = r_calc ∧
  ratio ≠ (π * r^2) / (2 * h * r + a * r) ∧
  ratio ≠ (π * r) / (2 * a + h) ∧
  ratio ≠ (π * r^2) / (h^2 + a^2) ∧
  ratio ≠ (π * r^2) / (2 * h * r + r^2)

theorem correct_answer_none_of_these (h a r : ℝ) (h_pos: h > 0) (a_pos: a > 0) (r_pos: r > 0) :
  right_triangle_area_ratio h a r :=
by
  -- Proof omitted
  sorry

end correct_answer_none_of_these_l584_584566


namespace cube_root_scale_l584_584789

theorem cube_root_scale (x : ℝ) (h₁ : (326:ℝ)^(1/3) ≈ 6.882) (h₂ : x^(1/3) ≈ 68.82) : x ≈ 326000 := 
sorry

end cube_root_scale_l584_584789


namespace median_of_81_consecutive_integers_l584_584198

theorem median_of_81_consecutive_integers (s : ℕ) (h_sum : s = 9^5) : 
  let n := 81 in
  let median := s / n in
  median = 729 :=
by
  have h₁ : 9^5 = 59049 := by norm_num
  have h₂ : 81 = 81 := rfl
  have h₃ : 59049 / 81 = 729 := by norm_num
  
  -- Apply the conditions
  rw [h_sum, <-h₁] at h_sum
  rw h₂
  
  -- Conclude the median
  exact h₃

end median_of_81_consecutive_integers_l584_584198


namespace range_of_g_l584_584422

noncomputable def g (x : ℝ) : ℝ :=
  sin x ^ 6 - sin x * cos x + cos x ^ 6

theorem range_of_g :
  ∀ y, ∃ x, g x = y ↔ (0 ≤ y ∧ y ≤ 1) :=
begin
  sorry
end

end range_of_g_l584_584422


namespace average_donation_l584_584716

theorem average_donation (d : ℕ) (n : ℕ) (r : ℕ) (average_donation : ℕ) 
  (h1 : d = 10)   -- $10 donated by customers
  (h2 : r = 2)    -- $2 donated by restaurant
  (h3 : n = 40)   -- number of customers
  (h4 : (r : ℕ) * n / d = 24) -- total donation by restaurant is $24
  : average_donation = 3 := 
by
  sorry

end average_donation_l584_584716


namespace max_area_isosceles_triangle_l584_584882

theorem max_area_isosceles_triangle (b : ℝ) (h : ℝ) (area : ℝ) 
  (h_cond : h^2 = 1 - b^2 / 4)
  (area_def : area = 1 / 2 * b * h) : 
  area ≤ 2 * Real.sqrt 2 / 3 := 
sorry

end max_area_isosceles_triangle_l584_584882


namespace part_a_part_b_l584_584723

variables (p : ℝ) (h : p ≥ 0 ∧ p ≤ 1) -- Probability p is between 0 and 1 inclusive

-- Part (a)
theorem part_a (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (5.choose 3) * p^3 * (1 - p)^2 = 10 * p^3 * (1 - p)^2 := by 
sorry

-- Part (b)
theorem part_b (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  10 * p - p^10 = (10 * p - p^10) := by 
sorry

end part_a_part_b_l584_584723


namespace odd_function_value_l584_584436

def f (a x : ℝ) : ℝ := -x^3 + (a-2)*x^2 + x

-- Test that f(x) is an odd function:
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem odd_function_value (a : ℝ) (h : is_odd_function (f a)) : f a a = -6 :=
by
  sorry

end odd_function_value_l584_584436


namespace width_of_rectangle_l584_584610

-- Define the problem constants and parameters
variable (L W : ℝ)

-- State the main theorem about the width
theorem width_of_rectangle (h₁ : L * W = 50) (h₂ : L + W = 15) : W = 5 :=
sorry

end width_of_rectangle_l584_584610


namespace roots_quadratic_relation_l584_584259

theorem roots_quadratic_relation (a b c d A B : ℝ)
  (h1 : a^2 + A * a + 1 = 0)
  (h2 : b^2 + A * b + 1 = 0)
  (h3 : c^2 + B * c + 1 = 0)
  (h4 : d^2 + B * d + 1 = 0) :
  (a - c) * (b - c) * (a + d) * (b + d) = B^2 - A^2 :=
sorry

end roots_quadratic_relation_l584_584259


namespace no_such_integers_l584_584919

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem no_such_integers (a b c d k : ℤ) (h : k > 1) :
  (a + b * omega + c * omega^2 + d * omega^3)^k ≠ 1 + omega :=
sorry

end no_such_integers_l584_584919


namespace sin_60_eq_sqrt_three_div_two_l584_584360

theorem sin_60_eq_sqrt_three_div_two :
  Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_60_eq_sqrt_three_div_two_l584_584360


namespace proof_problem_l584_584930

variables {a b c : ℝ}

theorem proof_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 4 * a^2 + b^2 + 16 * c^2 = 1) :
  (0 < a * b ∧ a * b < 1 / 4) ∧ (1 / a^2 + 1 / b^2 + 1 / (4 * a * b * c^2) > 49) :=
by
  sorry

end proof_problem_l584_584930


namespace sin_60_eq_sqrt3_div_2_l584_584365

theorem sin_60_eq_sqrt3_div_2 :
  ∃ (Q : ℝ × ℝ), dist Q (1, 0) = 1 ∧ angle (1, 0) Q = real.pi / 3 ∧ Q.2 = real.sqrt 3 / 2 := sorry

end sin_60_eq_sqrt3_div_2_l584_584365


namespace CarriesJellybeanCount_l584_584339

-- Definitions based on conditions in part a)
def BertBoxJellybeans : ℕ := 150
def BertBoxVolume : ℕ := 6
def CarriesBoxVolume : ℕ := 3 * 2 * 4 * BertBoxVolume -- (3 * height, 2 * width, 4 * length)

-- Theorem statement in Lean based on part c)
theorem CarriesJellybeanCount : (CarriesBoxVolume / BertBoxVolume) * BertBoxJellybeans = 3600 := by 
  sorry

end CarriesJellybeanCount_l584_584339


namespace pencils_distribution_count_l584_584783

def count_pencils_distribution : ℕ :=
  let total_pencils := 10
  let friends := 4
  let adjusted_pencils := total_pencils - friends
  Nat.choose (adjusted_pencils + friends - 1) (friends - 1)

theorem pencils_distribution_count :
  count_pencils_distribution = 84 := 
  by sorry

end pencils_distribution_count_l584_584783


namespace clara_age_in_5_years_l584_584155

variable (A_pens : ℕ) (C_pens_frac : ℚ) (A_age : ℕ) (C_older : Prop) (pen_age_diff : ℕ)

-- Conditions given in the problem
def alice_pens := 60
def clara_pens := (2/5 : ℚ) * alice_pens
def alice_age := 20
def clara_age := alice_age + pen_age_diff

-- Clara is older than Alice
axiom clara_older : clara_age > alice_age

-- The difference in number of pens matches their age difference
axiom pen_age_difference : alice_pens - clara_pens.toNat = pen_age_diff

-- Question to be proved: Calculate Clara's age in 5 years
theorem clara_age_in_5_years : clara_age + 5 = 61 :=
by
  let A_pens := 60
  let C_pens_frac := 2 / 5
  let A_age := 20
  let pens_diff := A_pens - (C_pens_frac * A_pens).toInt
  have h_clara_age : clara_age = A_age + pens_diff := by
    sorry
  have h_clara_age_in_5_years : clara_age + 5 = (A_age + pens_diff) + 5 := by
    rw [h_clara_age]
  show clara_age + 5 = 61 from by
    sorry

end clara_age_in_5_years_l584_584155


namespace distance_Picklminster_Quickville_l584_584785

theorem distance_Picklminster_Quickville
  (v_A v_B v_C v_D t_1 t_2 t_3 t_4 d : ℝ)
  (h1: v_A * t_1 = 120)
  (h2: v_C * t_1 = d - 120)
  (h3: v_A * t_2 = 140)
  (h4: v_D * t_2 = d - 140)
  (h5: v_B * t_3 = d - 126)
  (h6: v_C * t_3 = 126)
  (h7: v_B = v_D)
  (h8: v_B * t_4 = d / 2)
  (h9: v_D * t_4 = d / 2) :
  d = 210 := by
  have eq1 : 120 * v_C = v_A * (d - 120) := by
    rw [←h1, ←h2]
  have eq2 : 140 * v_D = v_A * (d - 140) := by
    rw [←h3, ←h4]
  have eq3 : 126 * v_B = v_C * (d - 126) := by
    rw [←h5, ←h6]
  have eq4 : v_B = v_D := h7
  have eq5 : v_B * t_4 = d / 2 := h8
  have eq6 : v_D * t_4 = d / 2 := h9
  have v_eq : v_B = v_D := eq4
  sorry

end distance_Picklminster_Quickville_l584_584785


namespace convert_536_oct_to_base7_l584_584387

def octal_to_decimal (n : ℕ) : ℕ :=
  n % 10 + (n / 10 % 10) * 8 + (n / 100 % 10) * 64

def decimal_to_base7 (n : ℕ) : ℕ :=
  n % 7 + (n / 7 % 7) * 10 + (n / 49 % 7) * 100 + (n / 343 % 7) * 1000

theorem convert_536_oct_to_base7 : 
  decimal_to_base7 (octal_to_decimal 536) = 1010 :=
by
  sorry

end convert_536_oct_to_base7_l584_584387


namespace median_of_81_consecutive_integers_l584_584200

theorem median_of_81_consecutive_integers (s : ℕ) (h_sum : s = 9^5) : 
  let n := 81 in
  let median := s / n in
  median = 729 :=
by
  have h₁ : 9^5 = 59049 := by norm_num
  have h₂ : 81 = 81 := rfl
  have h₃ : 59049 / 81 = 729 := by norm_num
  
  -- Apply the conditions
  rw [h_sum, <-h₁] at h_sum
  rw h₂
  
  -- Conclude the median
  exact h₃

end median_of_81_consecutive_integers_l584_584200


namespace kim_hours_of_classes_per_day_l584_584914

-- Definitions based on conditions
def original_classes : Nat := 4
def hours_per_class : Nat := 2
def dropped_classes : Nat := 1

-- Prove that Kim now has 6 hours of classes per day
theorem kim_hours_of_classes_per_day : (original_classes - dropped_classes) * hours_per_class = 6 := by
  sorry

end kim_hours_of_classes_per_day_l584_584914


namespace sin_60_eq_sqrt_three_div_two_l584_584361

theorem sin_60_eq_sqrt_three_div_two :
  Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_60_eq_sqrt_three_div_two_l584_584361


namespace trader_sold_80_meters_l584_584737

variable (x : ℕ)
variable (selling_price_per_meter profit_per_meter cost_price_per_meter total_selling_price : ℕ)

theorem trader_sold_80_meters
  (h_cost_price : cost_price_per_meter = 118)
  (h_profit : profit_per_meter = 7)
  (h_selling_price : selling_price_per_meter = cost_price_per_meter + profit_per_meter)
  (h_total_selling_price : total_selling_price = 10000)
  (h_eq : selling_price_per_meter * x = total_selling_price) :
  x = 80 := by
    sorry

end trader_sold_80_meters_l584_584737


namespace perimeter_triangle_pqr_l584_584229

theorem perimeter_triangle_pqr (PQ PR QR QJ : ℕ) (h1 : PQ = PR) (h2 : QJ = 10) :
  ∃ PQR', PQR' = 198 ∧ triangle PQR PQ PR QR := sorry

end perimeter_triangle_pqr_l584_584229


namespace max_term_in_sequence_l584_584838

noncomputable def a_n (n : ℕ) := -2 * (n : ℤ)^2 + 9 * n + 3

theorem max_term_in_sequence : ∃ n ∈ {0, 1, 2, 3, 4, 5, 6}, a_n n = 13 :=
by
  have h_a2 : a_n 2 = 13 := by
    norm_num [a_n]
  have h_a3 : a_n 3 = 12 := by
    norm_num [a_n]
  existsi 2
  split
  norm_num
  exact h_a2

end max_term_in_sequence_l584_584838


namespace cookies_total_l584_584048

theorem cookies_total :
  (let Mona := 20 in
   let Jasmine := Mona - 5 in
   let Rachel := Jasmine + 10 in
   let Carlos := Rachel * 2 in
   Mona + Jasmine + Rachel + Carlos = 110) :=
by
  sorry

end cookies_total_l584_584048


namespace polynomial_coefficients_sum_l584_584787

theorem polynomial_coefficients_sum :
  ∀ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} : ℝ),
  ∀ (x : ℝ),
  (∀ x, (x ^ 2 + 1) * (x - 2) ^ 9 =
    a + a_1 * (x - 1) + a_2 * (x - 1) ^ 2 + a_3 * (x - 1) ^ 3 +
    a_4 * (x - 1) ^ 4 + a_5 * (x - 1) ^ 5 + a_6 * (x - 1) ^ 6 +
    a_7 * (x - 1) ^ 7 + a_8 * (x - 1) ^ 8 +
    a_9 * (x - 1) ^ 9 + a_{10} * (x - 1) ^ 10 + a_{11} * (x - 1) ^ 11) →
  (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} + a_{11} = 2) :=
by 
  intros a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} x h,
  sorry

end polynomial_coefficients_sum_l584_584787


namespace Kim_total_hours_l584_584912

-- Define the initial conditions
def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_class : ℕ := 1

-- The proof problem: Given the initial conditions, prove the total hours of classes per day is 6
theorem Kim_total_hours : (initial_classes - dropped_class) * hours_per_class = 6 := by
  sorry

end Kim_total_hours_l584_584912


namespace diagonals_form_triangle_in_convex_pentagon_l584_584074

theorem diagonals_form_triangle_in_convex_pentagon 
  (A B C D E : Type) [has_le A] [has_lt A] [has_add A] [linear_order A] 
  {ABCDE : convex_pentagon A B C D E} : 
  ∃ (BE BD EC : A), BE < BD + EC := 
sorry

end diagonals_form_triangle_in_convex_pentagon_l584_584074


namespace arithmetic_sequence_S17_over_a3_l584_584799

variable (a₁ d : ℝ) (n : ℕ)

def a_n (n : ℕ) := a₁ + (n - 1) * d

def S_n (n : ℕ) := (n / 2 : ℝ) * (2 * a₁ + (n - 1) * d)

theorem arithmetic_sequence_S17_over_a3 :
  d ≠ 0 → a₁ + 5 * d = 2 * (a₁ + 2 * d) → (S_n a₁ d 17) / (a_n a₁ d 3) = 51 :=
by
  intros h₀ h₁
  sorry

end arithmetic_sequence_S17_over_a3_l584_584799


namespace ratio_of_areas_l584_584055

-- Definitions and assumptions directly from conditions
def Circle (center : ℝ × ℝ) (radius : ℝ) : Prop := true
def Angle (A B C : ℝ × ℝ) (θ : ℝ) : Prop := true
def InternallyTangent (C1 C2 : Prop) : Prop := true
def TangentToLine (C : Prop) (P Q : ℝ × ℝ) : Prop := true

-- Given conditions
variables (O A B : ℝ × ℝ)
variables (R r : ℝ)
variables (circle1 : Circle O R)
variables (circle2 : Circle O r)
variables (angleAOB : Angle O A B 60)
variables (tangent1 : InternallyTangent circle1 circle2)
variables (tangent2 : TangentToLine circle2 O A)
variables (tangent3 : TangentToLine circle2 O B)

-- To prove
theorem ratio_of_areas (O A B : ℝ × ℝ) (R r : ℝ)
    (circle1 : Circle O R) (circle2 : Circle O r)
    (angleAOB : Angle O A B 60) (tangent1 : InternallyTangent circle1 circle2)
    (tangent2 : TangentToLine circle2 O A) (tangent3 : TangentToLine circle2 O B) :
    (r / R) ^ 2 = 1 / 9 :=
by
  sorry

end ratio_of_areas_l584_584055


namespace max_kings_no_attack_l584_584251

-- Define the chessboard as an 8x8 grid
def Chessboard : Type := ℕ × ℕ

-- Define the conditions of the king's movement
def attacks (k1 k2 : Chessboard) : Prop :=
  (abs (k1.1 - k2.1) ≤ 1) ∧
  (abs (k1.2 - k2.2) ≤ 1) ∧
  (k1 ≠ k2)

-- Define a valid placement of kings where no two kings attack each other
def valid_placement (kings : Finset Chessboard) : Prop :=
  ∀ (k1 k2 : Chessboard), k1 ∈ kings → k2 ∈ kings → k1 ≠ k2 → ¬ attacks k1 k2

-- Prove that the maximum number of kings under the valid placement condition is 16
theorem max_kings_no_attack : ∃ kings : Finset Chessboard, valid_placement kings ∧ kings.card = 16 :=
  sorry

end max_kings_no_attack_l584_584251


namespace max_storage_weeks_l584_584701

theorem max_storage_weeks 
  (cost : ℝ) (total_weight : ℝ) (price_per_ton : ℝ) (loss_rate : ℝ) (minimum_profit : ℝ) 
  (h1 : cost = 64000) 
  (h2 : total_weight = 80) 
  (h3 : price_per_ton = 1200) 
  (h4 : loss_rate = 2) 
  (h5 : minimum_profit = 20000) : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 5 ∧ price_per_ton * (total_weight - loss_rate * x) - cost ≥ minimum_profit :=
begin
  sorry
end

end max_storage_weeks_l584_584701


namespace simplify_sqrt_sum_l584_584132

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l584_584132


namespace part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l584_584731

-- Proof Problem for Part (a):
theorem part_a_probability_three_unused_rockets (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ q : ℝ, q = 10 * p^3 * (1-p)^2) := sorry

-- Proof Problem for Part (b):
theorem part_b_expected_targets_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ e : ℝ, e = 10 * p - p^10) := sorry

end part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l584_584731


namespace proof_c_plus_d_l584_584553

def a (n : ℕ) : ℕ :=
  n

def b (n : ℕ) : ℕ :=
  if even n then 2^(2 + 2 * (n - 1)) else 3^(2 + 2 * (n - 1))

noncomputable def U : ℕ := 
    ∑' n, if even n then (a n) * (2 ^ (-2 * (2 - 2 * (n - 1)))) else 0

noncomputable def V : ℕ := 
    ∑' n, if ¬ (even n) then (a n) * (3 ^ (-2 * (2 - 2 * (n - 1)))) else 0

noncomputable def fraction_sum : ℚ :=
  U + V

theorem proof_c_plus_d : 
    ∃ (c d : ℕ), c + d = 7 ∧ fraction_sum = (c : ℚ) / d ∧ nat.coprime c d := 
sorry

end proof_c_plus_d_l584_584553


namespace intersection_A_B_l584_584839

def setA : Set ℝ := { x | |x| < 2 }
def setB : Set ℝ := { x | x^2 - 4 * x + 3 < 0 }
def setC : Set ℝ := { x | 1 < x ∧ x < 2 }

theorem intersection_A_B : setA ∩ setB = setC := by
  sorry

end intersection_A_B_l584_584839


namespace probability_of_point_between_C_and_D_l584_584066

open_locale classical

noncomputable theory

def probability_C_to_D (A B C D : ℝ) (AB AD BC : ℝ) (h1 : AB = 4 * AD) (h2 : AB = 8 * BC) : ℝ :=
  let CD := BC - AD in
  CD / AB

theorem probability_of_point_between_C_and_D 
  (A B C D AB AD BC : ℝ) 
  (h1 : AB = 4 * AD) 
  (h2 : AB = 8 * BC) : 
  probability_C_to_D A B C D AB AD BC h1 h2 = 5 / 8 :=
by
  sorry

end probability_of_point_between_C_and_D_l584_584066


namespace log_simplification_l584_584275

theorem log_simplification :
  log (5 / 2) + 2 * log 2 - (1 / 2)⁻¹ = -1 :=
by
  sorry

end log_simplification_l584_584275


namespace find_m_l584_584794

-- We define the function f and the condition that it is increasing on (0, ∞)
definition power_function (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 - 2m - 1)

theorem find_m (m : ℝ) : (∀ x : ℝ, 0 < x → deriv (power_function m) x > 0) → m = -1 := by
  sorry

end find_m_l584_584794


namespace SoccerBallPrices_SoccerBallPurchasingPlans_l584_584703

theorem SoccerBallPrices :
  ∃ (priceA priceB : ℕ), priceA = 100 ∧ priceB = 80 ∧ (900 / priceA) = (720 / (priceB - 20)) :=
sorry

theorem SoccerBallPurchasingPlans :
  ∃ (m n : ℕ), (m + n = 90) ∧ (m ≥ 2 * n) ∧ (100 * m + 80 * n ≤ 8500) ∧
  (m ∈ Finset.range 66 \ Finset.range 60) ∧ 
  (∀ k ∈ Finset.range 66 \ Finset.range 60, 100 * k + 80 * (90 - k) ≥ 8400) :=
sorry

end SoccerBallPrices_SoccerBallPurchasingPlans_l584_584703


namespace infinite_primes_in_sequence_l584_584039

noncomputable def infinite_primes_in_sequence (a : ℕ) (h : a > 1) : Prop :=
  ∀ n : ℕ, 1 < a → (∃ p : ℕ, nat.prime p ∧ ∃ m : ℕ, m ≥ n ∧ p ∣ (seq a m))

/--
Let \( a > 1 \) be a given natural number. The sequence \{aₙ\} satisfies: 
\[ a₁ = 1, a₂ = a, aₙ₊₂ = a aₙ₊₁ − aₙ \]
for \( n ≥ 1 \). Prove that there exist infinitely many primes such that each of them is a factor of some term in the sequence \( \{aₙ\}_{n ≥ 1} \).
-/
theorem infinite_primes_in_sequence (a : ℕ) (ha : a > 1) : infinite_primes_in_sequence a ha :=
  sorry

end infinite_primes_in_sequence_l584_584039


namespace max_kings_no_attack_l584_584252

-- Define the chessboard as an 8x8 grid
def Chessboard : Type := ℕ × ℕ

-- Define the conditions of the king's movement
def attacks (k1 k2 : Chessboard) : Prop :=
  (abs (k1.1 - k2.1) ≤ 1) ∧
  (abs (k1.2 - k2.2) ≤ 1) ∧
  (k1 ≠ k2)

-- Define a valid placement of kings where no two kings attack each other
def valid_placement (kings : Finset Chessboard) : Prop :=
  ∀ (k1 k2 : Chessboard), k1 ∈ kings → k2 ∈ kings → k1 ≠ k2 → ¬ attacks k1 k2

-- Prove that the maximum number of kings under the valid placement condition is 16
theorem max_kings_no_attack : ∃ kings : Finset Chessboard, valid_placement kings ∧ kings.card = 16 :=
  sorry

end max_kings_no_attack_l584_584252


namespace number_has_divisors_l584_584627

-- Define the problem
theorem number_has_divisors (A : ℕ) (n : ℕ) 
  (hA : ∃ s : Finset ℕ, s.card = n ∧ (∀ x ∈ s, x > 0) ∧ A = s.prod id) : 
  ∃ d : Finset ℕ, d.card ≥ (n * (n - 1)) / 2 + 1 :=
begin
  sorry
end

end number_has_divisors_l584_584627


namespace median_of_81_consecutive_integers_l584_584196

theorem median_of_81_consecutive_integers (s : ℕ) (h_sum : s = 9^5) : 
  let n := 81 in
  let median := s / n in
  median = 729 :=
by
  have h₁ : 9^5 = 59049 := by norm_num
  have h₂ : 81 = 81 := rfl
  have h₃ : 59049 / 81 = 729 := by norm_num
  
  -- Apply the conditions
  rw [h_sum, <-h₁] at h_sum
  rw h₂
  
  -- Conclude the median
  exact h₃

end median_of_81_consecutive_integers_l584_584196


namespace distance_after_12_seconds_time_to_travel_380_meters_l584_584622

def distance_travelled (t : ℝ) : ℝ := 9 * t + 0.5 * t^2

theorem distance_after_12_seconds : distance_travelled 12 = 180 :=
by 
  sorry

theorem time_to_travel_380_meters : ∃ t : ℝ, distance_travelled t = 380 ∧ t = 20 :=
by 
  sorry

end distance_after_12_seconds_time_to_travel_380_meters_l584_584622


namespace channel_depth_l584_584153

theorem channel_depth
  (top_width bottom_width area : ℝ)
  (h : ℝ)
  (trapezium_area_formula : area = (1 / 2) * (top_width + bottom_width) * h)
  (top_width_val : top_width = 14)
  (bottom_width_val : bottom_width = 8)
  (area_val : area = 770) :
  h = 70 := 
by
  sorry

end channel_depth_l584_584153


namespace length_of_bridge_is_200_2_l584_584160

-- Definitions for the conditions
def train_length : ℝ := 500
def train_speed_kmph : ℝ := 42
def crossing_time_seconds : ℝ := 60

-- Conversion from km/hr to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

-- Calculating the speed in m/s
def train_speed_mps : ℝ :=
  kmph_to_mps train_speed_kmph

-- Calculating the total distance traveled in crossing time
def total_distance : ℝ :=
  train_speed_mps * crossing_time_seconds

-- Calculate the length of the bridge
def bridge_length : ℝ :=
  total_distance - train_length

-- Statement to prove the length of the bridge equals 200.2 meters
theorem length_of_bridge_is_200_2 : bridge_length = 200.2 := by
  sorry

end length_of_bridge_is_200_2_l584_584160


namespace number_of_books_l584_584319

theorem number_of_books (B : ℕ) (h1 : 0.3 * B + 45 = B) : B = 64 := sorry

end number_of_books_l584_584319


namespace avg_properties_l584_584467

-- Definitions of averaging (arithmetic mean) and its properties
def avg (x y : ℝ) : ℝ := (x + y) / 2

def assoc (f : ℝ → ℝ → ℝ) : Prop :=
∀ x y z : ℝ, f (f x y) z = f x (f y z)

def comm (f : ℝ → ℝ → ℝ) : Prop :=
∀ x y : ℝ, f x y = f y x

def distributes_over_sub (f : ℝ → ℝ → ℝ) : Prop :=
∀ x y z : ℝ, f x (y - z) = (f x y) - (f x z)

def sub_distributes_over (f : ℝ → ℝ → ℝ) : Prop :=
∀ v w z : ℝ, v - f w z = f (v - w) (v - z)

def has_identity (f : ℝ → ℝ → ℝ) (e : ℝ) : Prop :=
∀ x : ℝ, f x e = x

-- Formulating the proof problem
theorem avg_properties :
  ¬ assoc avg ∧
  comm avg ∧
  ¬ distributes_over_sub avg ∧
  sub_distributes_over avg ∧
  ¬ ∃ e : ℝ, has_identity avg e :=
by sorry

end avg_properties_l584_584467


namespace g_sqrt_45_l584_584565

noncomputable def g (x : ℝ) : ℝ :=
if x % 1 = 0 then 7 * x + 6 else ⌊x⌋ + 7

theorem g_sqrt_45 : g (Real.sqrt 45) = 13 := by
  sorry

end g_sqrt_45_l584_584565


namespace math_proof_problem_l584_584438

-- Define the sequence satisfying the given recurrence relation and initial conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

-- Define the conditions
def condition1 (a : ℕ → ℤ) : Prop := a 3 + a 7 = 20
def condition2 (a : ℕ → ℤ) : Prop := a 2 + a 5 = 14

-- Define the conclusion for the first question: the general term formula
def general_term (a : ℕ → ℤ) : Prop := ∀ n : ℕ, a n = 2 * ↑n

-- Define the sequence b_n based on a_n and T_n
def b_n (a : ℕ → ℤ) (n : ℕ) : ℚ := 1 / (((a n : ℚ) - 1) * ((a n : ℚ) + 1))
def T_n (a : ℕ → ℤ) (n : ℕ) : ℚ := (Finset.range n).sum (b_n a)

-- Define the conclusion for the second question: T_n < 1/2
def Tn_lt_half (a : ℕ → ℤ) : Prop := ∀ n : ℕ, T_n a n < 1 / 2

-- The final proof problem combining all conditions and assertions
theorem math_proof_problem (a : ℕ → ℤ) :
  arithmetic_sequence a →
  condition1 a →
  condition2 a →
  general_term a ∧ Tn_lt_half a :=
by
  intros h_seq h_cond1 h_cond2
  have : (∀ n, a n = 2 * n), from sorry, -- general term proof
  have : ∀ n, T_n a n < 1 / 2, from sorry -- Tn < 1 / 2 proof
  exact ⟨this.1, this.2⟩

end math_proof_problem_l584_584438


namespace square_semicircle_ratio_l584_584260

theorem square_semicircle_ratio (A B C D X: ℝ) (h_square: (B - A = D - A) ∧ (C - B = D - C) ∧ (B - A) + (D - A) = (C - B)) 
  (h_point: A < X ∧ X < D) 
  (h_touch: ∃ O Y : ℝ, (O = (C + X) / 2) ∧ (Y ≠ A ∧ Y ≠ B ∧ Y ≠ C ∧ Y ≠ D) ∧ (Y ≠ X) ∧ (abs(Y - O) = (C - O) ∧ abs(X - Y) = O - X): ℝ) :
  (AX / XD = ⅓) :=
by
  sorry

end square_semicircle_ratio_l584_584260


namespace smallest_perimeter_iso_triangle_l584_584236

theorem smallest_perimeter_iso_triangle :
  ∃ (x y : ℕ), (PQ = PR ∧ PQ = x ∧ PR = x ∧ QR = y ∧ QJ = 10 ∧ PQ + PR + QR = 416 ∧ 
  PQ = PR ∧ y = 8 ∧ 2 * (x + y) = 416 ∧ y^2 - 50 > 0 ∧ y < 10) :=
sorry

end smallest_perimeter_iso_triangle_l584_584236


namespace elliptical_difference_l584_584831

def point := (ℝ × ℝ)
def line (t : ℝ) := (1 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)
def curve (θ : ℝ) := (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)
def F : point := (1, 0)
def F₁ : point := (-1, 0)
def intersects (t₁ t₂ : ℝ) : Prop := 
  let A := line t₁ in 
  let B := line t₂ in 
  curve (2 * Real.acos ((line t₁).fst / 2)) = A ∧
  curve (2 * Real.acos ((line t₂).fst / 2)) = B ∧
  A.snd > B.snd

theorem elliptical_difference (t₁ t₂ : ℝ) 
  (h_intersect : intersects t₁ t₂) : 
  (abs (dist F₁ (line t₁)) - abs (dist F₁ (line t₂))) = 6 * Real.sqrt 2 / 7 
:= by sorry

end elliptical_difference_l584_584831


namespace convert_angle_form_l584_584391

theorem convert_angle_form (theta : ℝ) (k : ℤ) (alpha : ℝ) :
  theta = -1485 * (real.pi / 180) →
  theta = 2 * k * real.pi + alpha →
  0 < alpha ∧ alpha < 2 * real.pi →
  theta = -10 * real.pi + (7 * real.pi) / 4 :=
begin
  intros h1 h2 h3,
  sorry
end

end convert_angle_form_l584_584391


namespace vectors_parallel_l584_584482

   theorem vectors_parallel (m : ℝ) 
     (h_parallel : (-1 : ℝ) * m = 4) : 
     3 • (-1, 2) + 2 • (2, m) = (1, -2) :=
   by
     have h_m : m = -4 := sorry
     rw [h_m]
     -- calculations here
     sorry
   
end vectors_parallel_l584_584482


namespace simplify_sqrt_sum_l584_584101

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l584_584101


namespace pencils_distribution_count_l584_584782

def count_pencils_distribution : ℕ :=
  let total_pencils := 10
  let friends := 4
  let adjusted_pencils := total_pencils - friends
  Nat.choose (adjusted_pencils + friends - 1) (friends - 1)

theorem pencils_distribution_count :
  count_pencils_distribution = 84 := 
  by sorry

end pencils_distribution_count_l584_584782


namespace median_of_81_consecutive_integers_l584_584205

theorem median_of_81_consecutive_integers (S : ℤ) 
  (h1 : ∃ l : ℤ, ∀ k, (0 ≤ k ∧ k < 81) → (l + k) ∈ S) 
  (h2 : S = 9^5) : 
  (81 * 729 = 9^5) :=
by
  have h_sum : (S : ℤ) = ∑ i in ((finset.range 81).map (λ n, l + n)), id i, from sorry
  have h_eq : (81 * 729 = 9^5) := calc
    81 * 729 = 9^2 * 9^3 : by norm_num
    ... = 9^(2+3) : by ring
    ... = 9^5 : by norm_num
  exact h_eq

end median_of_81_consecutive_integers_l584_584205


namespace correct_option_D_l584_584258

-- Definitions of reasoning types and analysis method
def inductive_reasoning (p q : Prop) : Prop := p → q
def deductive_reasoning (p q : Prop) : Prop := q → p
def analogical_reasoning (p q : Prop) : Prop := p → q

-- Definition for the complex plane condition
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Condition for z being a complex number satisfying iz = 2 + 4i
def complex_condition (z : ℂ) : Prop :=
  complex.I * z = 2 + 4 * complex.I

-- Main theorem statement
theorem correct_option_D :
  inductive_reasoning (specific) (to_general) ∧
  deductive_reasoning (general) (to_specific) ∧
  ¬analogical_reasoning (specific) (to_specific) ∧
  ¬analysis_method_is (indirect_proof) ∧
  ∀ z : ℂ, complex_condition z → in_fourth_quadrant z →
  option_D_true :=
sorry

end correct_option_D_l584_584258


namespace percentage_decrease_in_ratio_l584_584263

theorem percentage_decrease_in_ratio :
  let ratio1 := 6 / 20 : ℝ
  let ratio2 := 9 / 108 : ℝ
  let percentage_decrease := ((ratio1 - ratio2) / ratio1) * 100
  percentage_decrease ≈ 72.23 :=
by
  unfold ratio1 ratio2 percentage_decrease
  sorry

end percentage_decrease_in_ratio_l584_584263


namespace probability_even_l584_584649

open Finset

-- Define the problem conditions
def is_even (n : ℤ) : Prop := ∃ k, n = 2 * k

def chosen_numbers (s : Finset ℤ) : Prop :=
  s.card = 3 ∧ ∀ x ∈ s, x ∈ (range 12).map Succ.succ

def valid_combination (x y z : ℤ) : Prop := xyz - x - y - z ∈ {0, 2, 4, ...}

-- The main problem statement
theorem probability_even (s : Finset ℤ) (h : chosen_numbers s) :
  ∃ x y z ∈ s, valid_combination x y z -> (nat.choose 6 3 : ℚ) / (nat.choose 12 3 : ℚ) = 1 / 11 :=
sorry


end probability_even_l584_584649


namespace probability_of_perfect_square_l584_584307

theorem probability_of_perfect_square : 
  ∀ (p : ℝ) (n : ℕ), 
  (n <= 120) → 
  (∀ n, n ≤ 60 → probability_of n = p) → 
  (∀ n, n > 60 → probability_of n = 3 * p) → 
  60 * p + 60 * 3 * p = 1 → 
  p = 1 / 240 →
  (7 * (1 / 240) + 3 * (3 * (1 / 240))) = 0.0667 := 
  by sorry

end probability_of_perfect_square_l584_584307


namespace part1_part2_l584_584928

variables (a b c : ℝ)
-- Assuming a, b, c are positive and satisfy the given equation
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : 0 < c)
variable (h_eq : 4 * a ^ 2 + b ^ 2 + 16 * c ^ 2 = 1)

-- Statement for the first part: 0 < ab < 1/4
theorem part1 : 0 < a * b ∧ a * b < 1 / 4 :=
  sorry

-- Statement for the second part: 1/a² + 1/b² + 1/(4abc²) > 49
theorem part2 : 1 / (a ^ 2) + 1 / (b ^ 2) + 1 / (4 * a * b * c ^ 2) > 49 :=
  sorry

end part1_part2_l584_584928


namespace least_fraction_to_unity_l584_584269

theorem least_fraction_to_unity :
  let series_sum := (∑ n in finset.range 20, (1 / (↑(n + 2) * (↑(n + 2) + 1)))) in
  1 - series_sum = 15 / 22 :=
by
  -- To be proven
  sorry

end least_fraction_to_unity_l584_584269


namespace no_solution_for_99_l584_584033

theorem no_solution_for_99 :
  ∃ n : ℕ, (¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 9 * x + 11 * y = n) ∧
  (∀ m : ℕ, n < m → ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 9 * x + 11 * y = m) ∧
  n = 99 :=
by
  sorry

end no_solution_for_99_l584_584033


namespace difference_of_squares_144_l584_584351

theorem difference_of_squares_144 (n : ℕ) (h : 3 * n + 3 < 150) : (n + 2)^2 - n^2 = 144 :=
by
  -- Given the conditions, we need to show this holds.
  sorry

end difference_of_squares_144_l584_584351


namespace square_division_l584_584593

theorem square_division (n : ℕ) (h : n ≥ 6) :
  ∃ (sq_div : ℕ → Prop), sq_div 6 ∧ (∀ n, sq_div n → sq_div (n + 3)) :=
by
  sorry

end square_division_l584_584593


namespace number_line_steps_l584_584577

theorem number_line_steps (total_steps : ℕ) (total_distance : ℕ) (steps_taken : ℕ) (result_distance : ℕ) 
  (h1 : total_distance = 36) (h2 : total_steps = 9) (h3 : steps_taken = 6) : 
  result_distance = (steps_taken * (total_distance / total_steps)) → result_distance = 24 :=
by
  intros H
  sorry

end number_line_steps_l584_584577


namespace digits_divisible_by_power_of_two_l584_584776

theorem digits_divisible_by_power_of_two (n : ℕ) : ∃ (N : ℕ), (∀ i, (i < n) → (N / 10^i) % 10 ∈ {1, 2}) ∧ (N % 2^n = 0) := 
sorry

end digits_divisible_by_power_of_two_l584_584776


namespace measure_of_angle_C_l584_584871

theorem measure_of_angle_C (A B C : ℝ) (h : sin A * sin A - sin C * sin C = (sin A - sin B) * sin B) :
  C = π / 3 :=
sorry

end measure_of_angle_C_l584_584871


namespace find_a_l584_584793

variable (a : ℝ)
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem find_a (h : is_pure_imaginary ((1 - complex.I) * (a + complex.I))) : a = -1 :=
sorry

end find_a_l584_584793


namespace valid_subcommittees_example_l584_584286

def num_valid_subcommittees (people : Finset String) (excluded_pair : Finset (String × String)) : Nat :=
  (people.card.choose 2) - excluded_pair.card

theorem valid_subcommittees_example :
  let people := {"Alice", "Bob", "Charlie", "David", "Ellen", "Frank", "Gina", "Harry"}.to_finset
  let excluded_pair := {("Alice", "Bob")}.to_finset
  num_valid_subcommittees people excluded_pair = 27 :=
by
  let people := {"Alice", "Bob", "Charlie", "David", "Ellen", "Frank", "Gina", "Harry"}.to_finset
  let excluded_pair := {("Alice", "Bob")}.to_finset
  sorry

end valid_subcommittees_example_l584_584286


namespace position_of_16856_in_sequence_l584_584754

theorem position_of_16856_in_sequence :
  ∃ (n : ℕ), 16856 = ∑ (k : ℕ) in { x : ℕ | x < n ∧ x.count_ones = 1 }, 7^k ∧ n = 36 :=
begin
  sorry
end

end position_of_16856_in_sequence_l584_584754


namespace coefficient_of_x2_y2_in_expansion_l584_584151

theorem coefficient_of_x2_y2_in_expansion :
  binomial 3 2 * binomial 4 2 = 18 :=
by sorry

end coefficient_of_x2_y2_in_expansion_l584_584151


namespace du_chin_remaining_money_l584_584958

noncomputable def du_chin_revenue_over_week : ℝ := 
  let day0_revenue := 200 * 20
  let day0_cost := 3 / 5 * day0_revenue
  let day0_remaining := day0_revenue - day0_cost

  let day1_revenue := day0_remaining * 1.10
  let day1_cost := day0_cost * 1.10
  let day1_remaining := day1_revenue - day1_cost

  let day2_revenue := day1_remaining * 0.95
  let day2_cost := day1_cost * 0.90
  let day2_remaining := day2_revenue - day2_cost

  let day3_revenue := day2_remaining
  let day3_cost := day2_cost
  let day3_remaining := day3_revenue - day3_cost

  let day4_revenue := day3_remaining * 1.15
  let day4_cost := day3_cost * 1.05
  let day4_remaining := day4_revenue - day4_cost

  let day5_revenue := day4_remaining * 0.92
  let day5_cost := day4_cost * 0.95
  let day5_remaining := day5_revenue - day5_cost

  let day6_revenue := day5_remaining * 1.05
  let day6_cost := day5_cost
  let day6_remaining := day6_revenue - day6_cost

  day0_remaining + day1_remaining + day2_remaining + day3_remaining + day4_remaining + day5_remaining + day6_remaining

theorem du_chin_remaining_money : du_chin_revenue_over_week = 13589.08 := 
  sorry

end du_chin_remaining_money_l584_584958


namespace median_of_consecutive_integers_l584_584186

theorem median_of_consecutive_integers (m : ℤ) (h : (∑ k in (finset.range 81).image (λ x, x - 40 + m), k) = 9^5) :
  m = 729 :=
by
  sorry

end median_of_consecutive_integers_l584_584186


namespace sum_of_x_coordinates_of_intersections_l584_584982

-- Definition of g(x) as a piecewise linear function with five segments
def g (x : ℝ) : ℝ := sorry  -- The exact definition is omitted in the general statement

-- To assume that g(x) intersects with y = x + 2
noncomputable def intersection_points : list ℝ := [-1, 0, 3]  -- Assume given points

-- Summing up the x-coordinates of the intersection points
theorem sum_of_x_coordinates_of_intersections :
  (intersection_points.sum = 2) :=
by
  -- Verification is skipped
  sorry

end sum_of_x_coordinates_of_intersections_l584_584982


namespace sin_60_eq_sqrt_three_div_two_l584_584357

theorem sin_60_eq_sqrt_three_div_two :
  Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_60_eq_sqrt_three_div_two_l584_584357


namespace median_of_81_consecutive_integers_l584_584199

theorem median_of_81_consecutive_integers (s : ℕ) (h_sum : s = 9^5) : 
  let n := 81 in
  let median := s / n in
  median = 729 :=
by
  have h₁ : 9^5 = 59049 := by norm_num
  have h₂ : 81 = 81 := rfl
  have h₃ : 59049 / 81 = 729 := by norm_num
  
  -- Apply the conditions
  rw [h_sum, <-h₁] at h_sum
  rw h₂
  
  -- Conclude the median
  exact h₃

end median_of_81_consecutive_integers_l584_584199


namespace conjugate_in_first_quadrant_l584_584890

-- Definition for the modulus of a complex number
def modulus (z : ℂ) : ℝ := complex.norm z

-- Definition for the point corresponding to the conjugate of a complex number
def point_of_conjugate (z : ℂ) : ℝ × ℝ :=
  let conj_z := complex.conj z in
  (conj_z.re, conj_z.im)

-- The main statement
theorem conjugate_in_first_quadrant (z : ℂ) (cond : z * (1 + complex.I) = modulus (1 + real.sqrt 3 * complex.I)) :
  let p := point_of_conjugate z in p.1 > 0 ∧ p.2 > 0 :=
sorry

end conjugate_in_first_quadrant_l584_584890


namespace show_adult_tickets_l584_584740

variable (A C : ℕ) -- variables representing the number of adult and children's tickets
variable (h1 : A = 2 * C) -- twice as many adults as children
variable (h2 : 5.50 * A + 2.50 * C = 1026) -- total receipts equation

theorem show_adult_tickets : A = 152 := 
by
  sorry

end show_adult_tickets_l584_584740


namespace problem_2005th_element_of_M_is_correct_l584_584480

def T : Set ℤ := {0, 1, 2, 3, 4, 5, 6}

def M : Set ℚ := {
  (1 / 7) * a1 + (1 / 7^2) * a2 + (1 / 7^3) * a3 + (1 / 7^4) * a4 |
  a1 ∈ T, a2 ∈ T, a3 ∈ T, a4 ∈ T
}

noncomputable def nth_largest (s : Set ℚ) (n : ℕ) [LinearOrder ℚ] : ℚ :=
  (s.toFinset.sort (· ≥ ·)).get (n - 1)

theorem problem_2005th_element_of_M_is_correct :
  nth_largest M 2005 = 1/7 + 1/7^2 + 0/7^3 + 4/7^4 :=
sorry

end problem_2005th_element_of_M_is_correct_l584_584480


namespace part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l584_584732

-- Proof Problem for Part (a):
theorem part_a_probability_three_unused_rockets (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ q : ℝ, q = 10 * p^3 * (1-p)^2) := sorry

-- Proof Problem for Part (b):
theorem part_b_expected_targets_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ e : ℝ, e = 10 * p - p^10) := sorry

end part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l584_584732


namespace points_M_N_A_C_concyclic_l584_584564

open EuclideanGeometry

variables {A B C D Q R P S M N : Point}

-- Conditions from the problem
variables (convex_quadrilateral : ConvexQuadrilateral A B C D)
variables (angle_DAB_eq_90 : ∠ D A B = 90)
variables (angle_BCD_eq_90 : ∠ B C D = 90)
variables (angle_ABC_gt_angle_CDA : ∠ A B C > ∠ C D A)
variables (Q_on_BC : Collinear Q B C)
variables (R_on_CD : Collinear R C D)
variables (QR_intersects_AB_at_P : LineThrough Q R ⊆ LineThrough A B)
variables (QR_intersects_AD_at_S : LineThrough Q R ⊆ LineThrough A D)
variables (PQ_eq_RS : distance P Q = distance R S)
variables (M_midpoint_BD : Midpoint M B D)
variables (N_midpoint_QR : Midpoint N Q R)

theorem points_M_N_A_C_concyclic :
  Concyclic M N A C :=
  sorry

end points_M_N_A_C_concyclic_l584_584564


namespace loss_percentage_is_correct_l584_584301

noncomputable def watch_loss_percentage (SP_loss SP_profit : ℕ) (profit_percentage : ℕ) : ℕ :=
  let CP := SP_profit / (1 + profit_percentage / 100) in
  let loss := CP - SP_loss in
  loss * 100 / CP

theorem loss_percentage_is_correct :
  watch_loss_percentage 1140 1260 5 = 5 :=
by
  sorry

end loss_percentage_is_correct_l584_584301


namespace general_formula_a_n_sum_first_n_b_l584_584549

noncomputable def a : ℕ → ℕ
| 0       := 3
| (n + 1) := a n + 2

theorem general_formula_a_n (n : ℕ) : a n = 2 * n + 1 :=
sorry

noncomputable def b (n : ℕ) : ℚ :=
1 / (a n * a (n + 1))

def sum_b (n : ℕ) : ℚ :=
∑ i in Finset.range n, b i

theorem sum_first_n_b (n : ℕ) : sum_b n = n / (6 * n + 9) :=
sorry

end general_formula_a_n_sum_first_n_b_l584_584549


namespace solution_set_inequality_l584_584635

theorem solution_set_inequality (x : ℝ) : (x^2 + x - 2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) := 
sorry

end solution_set_inequality_l584_584635


namespace eccentricity_squared_l584_584445

variable (a b c : ℝ) (e : ℝ)
variable (F1 F2 P A : ℝ × ℝ)

-- Conditions
axiom ellipse_eq : a > b ∧ b > 0 ∧ (P.1^2 / a^2) + (P.2^2 / b^2) = 1
axiom foci_def : F1 = (c, 0) ∧ F2 = (-c, 0)
axiom P_on_ellipse : (P.1, P.2) satisfies ellipse_eq
axiom perpendicular_PF2_F1F2 : (P.1 - (-c)) * (F1.2 - F2.2) = - (P.2 - F2.2) * (F1.1 - F2.1)
axiom perpendicular_to_x_axis : (A.2 = 0) ∧ (P.1, P.2) and (A.1, 0) are projections of line from P to F1P
axiom AF2_eq_half_c : dist A F2 = c / 2
axiom eccentricity_def : e = c / a

-- Proof Statement
theorem eccentricity_squared : e^2 = (3 - Real.sqrt 5) / 2 :=
sorry

end eccentricity_squared_l584_584445


namespace line_AB_passes_through_fixed_point_l584_584465

-- Define the initial conditions and setup
variables {a b : ℝ} (P Q : ℝ × ℝ) (F1 F2 O : ℝ × ℝ) (k1 k2: ℝ)

-- Define the properties of the ellipse
def ellipse (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- The initial conditions
axiom a_gt_b_gt_0 : a > b ∧ b > 0
axiom passes_through_P : ellipse 1 (real.sqrt 2 / 2) a b

-- Vectors and intersection condition
axiom F1F2_perpendicular : F2 = (1, 0) ∧ F1 = (-1, 0)
axiom vec_PF2_eq_2_vec_QO : (P, F2 - (Q, (0, 0))) = 2 * (Q, (0, 0))

-- The slope condition
axiom slopes_k1_k2 : k1 + k2 = 2

-- Defining the proof
theorem line_AB_passes_through_fixed_point :
  ∃ A B : ℝ × ℝ, ellipse A.1 A.2 a b ∧ ellipse B.1 B.2 a b ∧
  (∃ m : ℝ, m ≠ 1 ∧ line_through M A B k1 k2) →
  ∃ fixed_point : ℝ × ℝ, fixed_point = (-1, -1) :=
sorry

end line_AB_passes_through_fixed_point_l584_584465


namespace sum_XY_Intersection_l584_584840

def A : Set ℕ := {1, 2, 3}
def X : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ A ∧ z = 4 * x + y}
def Y : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ A ∧ z = 4 * x - y}
def XY_Intersection_Sum : ℕ := ∑ z in (X ∩ Y), z

theorem sum_XY_Intersection : XY_Intersection_Sum = 48 := 
by 
  sorry

end sum_XY_Intersection_l584_584840


namespace part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l584_584730

-- Proof Problem for Part (a):
theorem part_a_probability_three_unused_rockets (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ q : ℝ, q = 10 * p^3 * (1-p)^2) := sorry

-- Proof Problem for Part (b):
theorem part_b_expected_targets_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ e : ℝ, e = 10 * p - p^10) := sorry

end part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l584_584730


namespace simplify_sqrt_72_plus_sqrt_32_l584_584085

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l584_584085


namespace problem_statement_l584_584476

/-- Definition of the function under consideration -/
def f (x a : ℝ) : ℝ := x * abs (x - a)

theorem problem_statement (a : ℝ) :
  (f 0 a = 0) ∧
  (f(-x) a = -f x a → a = 0) ∧
  ((a > 2) → ∀ x, x ≤ 2 → f x a = -x^2 + a * x) ∧
  ¬ ((a = 1) → ∃ x, f x a = 1/4) ∧
  ((a = 2) → ∃ m, ∃ x1 x2 x3 : ℝ, f x1 a - m = 0 ∧ f x2 a - m = 0 ∧ f x3 a - m = 0 ∧ 0 < m ∧ m < 1)
:= sorry

end problem_statement_l584_584476


namespace marta_sold_on_saturday_l584_584947

-- Definitions of conditions
def initial_shipment : ℕ := 1000
def rotten_tomatoes : ℕ := 200
def second_shipment : ℕ := 2000
def tomatoes_on_tuesday : ℕ := 2500
def x := 300

-- Total tomatoes on Monday after the second shipment
def tomatoes_on_monday (sold_tomatoes : ℕ) : ℕ :=
  initial_shipment - sold_tomatoes - rotten_tomatoes + second_shipment

-- Theorem statement to prove
theorem marta_sold_on_saturday : (tomatoes_on_monday x = tomatoes_on_tuesday) -> (x = 300) :=
by 
  intro h
  sorry

end marta_sold_on_saturday_l584_584947


namespace ellipse_major_axis_length_l584_584743

theorem ellipse_major_axis_length :
  let F1 : ℝ × ℝ := (11, 30)
  let F2 : ℝ × ℝ := (51, 65)
  let line_y_tangent : ℝ := 10
  in ∃ (a : ℝ), a = 85 ∧ (ellipse_with_foci_tangent F1 F2 line_y_tangent a) := 
begin
  sorry
end

end ellipse_major_axis_length_l584_584743


namespace max_intersection_points_l584_584588

-- Define the conditions
structure Circle (center : ℝ × ℝ) (radius : ℝ)

-- Define circle C
def circle_C : Circle (0, 0) 5 := ⟨⟨0, 0⟩, 5⟩

-- Define point P and its distance from the center of C
def P : ℝ × ℝ := (6, 0)

-- Define the circle centered at P with radius 4
def circle_P : Circle P 4 := ⟨P, 4⟩

-- Define theorem
theorem max_intersection_points (C : Circle (0, 0) 5) (P : ℝ × ℝ) (P_dist : P = (6, 0)) (r2 : ℝ) (h : r2 = 4) :
  ∃ n : ℕ, n = 2 := 
sorry

end max_intersection_points_l584_584588


namespace sum_of_satisfying_integers_l584_584424

noncomputable def cot (x : ℝ) := 1 / tan x
noncomputable def inequality (x : ℤ) : Prop :=
  (1 - (cot (Real.pi * x / 12)) ^ 2) *
  (1 - 3 * (cot (Real.pi * x / 12)) ^ 2) *
  (1 - tan (Real.pi * x / 6) * (cot (Real.pi * x / 4))) ≤ 16

theorem sum_of_satisfying_integers :
  (Finset.filter (λx, inequality x) (Finset.Icc (-3 : ℤ) 13)).sum id = 28 :=
  sorry

end sum_of_satisfying_integers_l584_584424


namespace average_marks_l584_584150

theorem average_marks 
  (n1 : ℕ) (n2 : ℕ) (avg1 : ℕ) (avg2 : ℕ) 
  (h1 : n1 = 30) (h2 : n2 = 50) (h3 : avg1 = 40) (h4 : avg2 = 70) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 58.75 := 
by
  sorry

end average_marks_l584_584150


namespace point_distance_proof_l584_584714

noncomputable def find_distance (x y : ℝ) : ℝ := 
  real.sqrt (x^2 + y^2)

theorem point_distance_proof : 
  ∀ (x y : ℝ), y = 12 ∧ real.sqrt ((x - 1)^2 + (y - 6)^2) = 10 ∧ x > 1 → find_distance x y = 15 :=
by 
  intros x y h
  cases h with hy1 hxy
  cases hxy with hxy1 hx
  rw [hy1, find_distance]
  have h_square_dist : (x - 1)^2 + 36 = 100 := by
    rw [hy1] at hxy1
    exact hxy1
  have h_eq : (x - 1)^2 = 64 := 
    by linarith
  have h_x_values : x = 9 ∨ x = -7 := 
    by apply (real.eq_or_ne (x - 1 = 8) (x - 1 = -8)).mp
  cases h_x_values with hx_pos hx_neg
  { simp at hx_pos
    rw hx_pos
    rw [hy1] --
    calc real.sqrt (9^2 + 12^2) 
         = real.sqrt 225 : by ring
         = 15 : by norm_num }
  {  exfalso
     have : -7 ≠ 9 := by norm_num
     contradiction }
  { sorry }

end point_distance_proof_l584_584714


namespace part_I_part_II_l584_584790

variables {A B C a b c : ℝ} 

-- Condition: Given a triangle ABC with specific sides and the trigonometric relation
axiom h1 : ∀ {A B C : ℝ}, sin (2 * A + B) / sin A = 2 + 2 * cos (A + B)

-- Proof of part I: Find the value of b/a.
theorem part_I (h1 : ∀ {A B : ℝ}, sin (2 * A + B) / sin A = 2 + 2 * cos (A + B)) :
  b / a = 2 := sorry

-- Part II: Given specific values for a and c, find the area of triangle ABC.
noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let C := real.acos ((a^2 + b^2 - c^2) / (2 * a * b)) in
  1 / 2 * a * b * real.sin C

-- Theorem for part II: Given a = 1, b = 2 and c = sqrt(7), find the area of the triangle.
theorem part_II : ∀ {a b c : ℝ}, a = 1 → c = real.sqrt 7 → b / a = 2 → 
  area_of_triangle 1 2 (real.sqrt 7) = real.sqrt 3 / 2 := sorry

end part_I_part_II_l584_584790


namespace total_distance_l584_584499

-- Definitions from conditions
def KD : ℝ := 4
def DM : ℝ := KD / 2  -- Derived from the condition KD = 2 * DM

/-- The total distance Ken covers is 4 miles (to Dawn's) + 2 miles (to Mary's) + 2 miles (back to Dawn's) + 4 miles (back to Ken's). -/
theorem total_distance : KD + DM + DM + KD = 12 :=
by
  sorry

end total_distance_l584_584499


namespace median_of_81_consecutive_integers_l584_584174

theorem median_of_81_consecutive_integers (n : ℕ) (S : ℕ) (h1 : n = 81) (h2 : S = 9^5) : 
  let M := S / n in M = 729 :=
by
  sorry

end median_of_81_consecutive_integers_l584_584174


namespace distance_between_given_parallel_lines_l584_584842

noncomputable def distance_between_parallel_lines (L1 L2 : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2))) : ℝ :=
  -- Define the distance formula for two parallel lines
sorry

theorem distance_between_given_parallel_lines :
  let L1 := (3:ℝ) • EuclideanSpace.mk 2 A • (EuclideanSpace.mk 1 B • (6:ℝ) • EuclideanSpace.mk 2 C) = 0
  let L2 := (6:ℝ) • EuclideanSpace.mk 1 A • (4:ℝ) • (EuclideanSpace.mk 2 B • (3:ℝ)) = 0
  (distance_between_parallel_lines L1 L2 = 3) := 
by
sorry

end distance_between_given_parallel_lines_l584_584842


namespace simplify_sqrt72_add_sqrt32_l584_584131

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l584_584131


namespace original_price_of_ipod_l584_584573

theorem original_price_of_ipod
  (P : ℝ)
  (discounted_price : ℝ)
  (discount_rate : ℝ)
  (h : discounted_price = P * (1 - discount_rate)) :
  discounted_price = 83.2 → P = 128 :=
by
  assume h1 : discounted_price = 83.2,
  rw h1 at h,
  exact sorry

end original_price_of_ipod_l584_584573


namespace find_angle_B_max_area_triangle_l584_584873

variables {A B C : ℝ} {a b c : ℝ}

theorem find_angle_B (h : (sin A) / a = (sqrt 3) * (cos B) / b) : B = π / 3 :=
sorry

theorem max_area_triangle (h : (sin A) / a = (sqrt 3) * (cos B) / b) (hb : b = 2) (hB : B = π / 3) :
  ∃ (S_max : ℝ), S_max = sqrt 3 :=
sorry

end find_angle_B_max_area_triangle_l584_584873


namespace area_transformed_region_l584_584023

noncomputable def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 2], ![5, -4]]
def original_area : ℝ := 15

theorem area_transformed_region : 
  let det := Matrix.det transformation_matrix in
  let scale_factor := |det| in
  let transformed_area := original_area * scale_factor in
  transformed_area = 330 :=
by
  sorry

end area_transformed_region_l584_584023


namespace simplify_expression_l584_584083

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b ^ 2 + 2 * b) - 4 * b ^ 2 = 9 * b ^ 3 + 2 * b ^ 2 :=
by
  sorry

end simplify_expression_l584_584083


namespace min_socks_for_15_pairs_l584_584288

theorem min_socks_for_15_pairs :
  let num_socks := λ (r g b bl w y : ℕ), r + g + b + bl + w + y
  in r = 120 ∧ g = 90 ∧ b = 70 ∧ bl = 50 ∧ w = 30 ∧ y = 10 →
     ∀ r g b bl w y, num_socks r g b bl w y = 370 →
     ∃ n, n = 30 ∧ (∀ selected : C, selected.total ≥ n →
       (selected.pairs ≥ 15)) :=
begin
  sorry
end

structure C :=
(total : ℕ)
(pairs : ℕ)

end min_socks_for_15_pairs_l584_584288


namespace calculate_loss_percentage_l584_584302

theorem calculate_loss_percentage
  (CP SP₁ SP₂ : ℝ)
  (h₁ : SP₁ = CP * 1.05)
  (h₂ : SP₂ = 1140) :
  (CP = 1200) → (SP₁ = 1260) → ((CP - SP₂) / CP * 100 = 5) :=
by
  intros h1 h2
  -- Here, we will eventually provide the actual proof steps.
  sorry

end calculate_loss_percentage_l584_584302


namespace find_W_l584_584313

noncomputable def volume_of_space (r_sphere r_cylinder h_cylinder : ℝ) : ℝ :=
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * h_cylinder
  let V_cone := (1 / 3) * Real.pi * r_cylinder^2 * h_cylinder
  V_sphere - V_cylinder - V_cone

theorem find_W : volume_of_space 6 4 10 = (224 / 3) * Real.pi := by
  sorry

end find_W_l584_584313


namespace stuffed_animal_ratio_l584_584047

theorem stuffed_animal_ratio
  (K : ℕ)
  (h1 : 34 + K + (K + 5) = 175) :
  K / 34 = 2 :=
by sorry

end stuffed_animal_ratio_l584_584047


namespace median_of_81_consecutive_integers_l584_584207

theorem median_of_81_consecutive_integers (S : ℤ) 
  (h1 : ∃ l : ℤ, ∀ k, (0 ≤ k ∧ k < 81) → (l + k) ∈ S) 
  (h2 : S = 9^5) : 
  (81 * 729 = 9^5) :=
by
  have h_sum : (S : ℤ) = ∑ i in ((finset.range 81).map (λ n, l + n)), id i, from sorry
  have h_eq : (81 * 729 = 9^5) := calc
    81 * 729 = 9^2 * 9^3 : by norm_num
    ... = 9^(2+3) : by ring
    ... = 9^5 : by norm_num
  exact h_eq

end median_of_81_consecutive_integers_l584_584207


namespace median_of_consecutive_integers_l584_584188

theorem median_of_consecutive_integers (m : ℤ) (h : (∑ k in (finset.range 81).image (λ x, x - 40 + m), k) = 9^5) :
  m = 729 :=
by
  sorry

end median_of_consecutive_integers_l584_584188


namespace nancy_deleted_files_correct_l584_584575

-- Variables and conditions
def nancy_original_files : Nat := 43
def files_per_folder : Nat := 6
def number_of_folders : Nat := 2

-- Definition of the number of files that were deleted
def nancy_files_deleted : Nat :=
  nancy_original_files - (files_per_folder * number_of_folders)

-- Theorem to prove
theorem nancy_deleted_files_correct :
  nancy_files_deleted = 31 :=
by
  sorry

end nancy_deleted_files_correct_l584_584575


namespace median_of_consecutive_integers_l584_584187

theorem median_of_consecutive_integers (m : ℤ) (h : (∑ k in (finset.range 81).image (λ x, x - 40 + m), k) = 9^5) :
  m = 729 :=
by
  sorry

end median_of_consecutive_integers_l584_584187


namespace find_x_satisfying_inequality_l584_584809

variable {f : ℝ → ℝ}

-- Define an even function
def even_function (f : ℝ → ℝ) :=
  ∀ x, f(x) = f(-x)

-- Define a monotonically decreasing function on an interval
def monotone_decreasing_on_interval (f : ℝ → ℝ) (I : set ℝ) :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

theorem find_x_satisfying_inequality (h₁ : even_function f)
  (h₂ : monotone_decreasing_on_interval f {x : ℝ | x < 0}) :
  {x : ℝ | f (x^2 + 2*x + 3) > f (-x^2 - 4*x - 5)} = {x | x < -1} :=
by {
  sorry
}

end find_x_satisfying_inequality_l584_584809


namespace volume_of_truncated_pyramid_l584_584612

-- Definitions based on provided conditions
variable (a b : ℝ)
variable (h₁ : a > b)
variable (h₂ : ∀ θ, θ = 45 → real.tan θ = 1)

-- Main theorem statement
theorem volume_of_truncated_pyramid (a b : ℝ) (h₁ : a > b) (h₂ : ∀ θ, θ = 45 → real.tan θ = 1) :
  let H := (real.sqrt 2 / 2) * (a - b) in
  let V := (real.sqrt 2 / 6) * (a^3 - b^3) in
  V = (1 / 3) * H * (a^2 + a * b + b^2) := 
by
  sorry

end volume_of_truncated_pyramid_l584_584612


namespace one_of_them_is_mistaken_l584_584646

theorem one_of_them_is_mistaken
  (k n x y : ℕ) 
  (hYakov: k * (x + 5) = n * y)
  (hYuri: k * x = n * (y - 3)) :
  False :=
by
  sorry

end one_of_them_is_mistaken_l584_584646


namespace right_triangle_area_4_l584_584896

-- Define points P and Q
structure Point :=
(x : ℝ)
(y : ℝ)

def P : Point := { x := 2, y := 0 }
def Q : Point := { x := -2, y := 0 }

-- Define the conditions
def is_right_triangle (P Q R : Point) : Prop :=
((P.y - Q.y)*(Q.x - R.x) = (Q.y - R.y)*(P.x - Q.x))

def area_triangle (P Q R : Point) : ℝ :=
0.5 * abs ((P.x - R.x) * (Q.y - R.y) - (P.y - R.y) * (Q.x - R.x))

def R_points : set Point :=
{ R : Point | is_right_triangle P Q R ∧ area_triangle P Q R = 4 }

def horizontal_lines (y1 y2 : ℝ) : set Point :=
{ R : Point | R.y = y1 ∨ R.y = y2 }

-- The proof statement
theorem right_triangle_area_4 :
  R_points = horizontal_lines 2 (-2) :=
sorry

end right_triangle_area_4_l584_584896


namespace contestant_origin_and_prize_l584_584338

def contestant := {name : String, city : String, prize : Nat}

variable (Beibei Jingjing Huanhuan : contestant)

namespace WCMC

theorem contestant_origin_and_prize :
  (Beibei.city ≠ "Lucheng") →
  (Jingjing.city ≠ "Yongjia") →
  (∀ c, c.city = "Lucheng" → c.prize ≠ 1) →
  (∀ c, c.city = "Yongjia" → c.prize = 2) →
  (Jingjing.prize ≠ 3) →
  (Huanhuan.city = "Lucheng" ∧ Huanhuan.prize = 3) :=
by
  intros h1 h2 h3 h4 h5
  -- Proof omitted
  sorry

end WCMC

end contestant_origin_and_prize_l584_584338


namespace stacy_height_now_l584_584917

-- Definitions based on the given conditions
def S_initial : ℕ := 50
def J_initial : ℕ := 45
def J_growth : ℕ := 1
def S_growth : ℕ := J_growth + 6

-- Prove statement about Stacy's current height
theorem stacy_height_now : S_initial + S_growth = 57 := by
  sorry

end stacy_height_now_l584_584917


namespace find_length_of_BD_l584_584884

variable (A B C D E F : Type)
variables {r1 r2 r3 r4 r5 : Real}
variable [Core.Axioms.num {r1 r2 r3 r4 r5}]
variables (ABCD : Rectangle r1 r2 r3 r4)
variable (E_on_AB : PointOn E AB)
variable (F_on_CD : PointOn F CD)
variables (AF_perp_BD : Perpendicular AF BD)
variables (CE_perp_BD : Perpendicular CE BD)
variable (areas_equal : AreasEqual (BF, DE, ABCD))
variable (EF_eq_one : EF = 1)

theorem find_length_of_BD: 
  BD(ABCD) = sqrt 3 :=
by
  sorry

end find_length_of_BD_l584_584884


namespace compute_P_2_4_8_l584_584146

noncomputable def P : ℝ → ℝ → ℝ → ℝ := sorry

axiom homogeneity (x y z k : ℝ) : P (k * x) (k * y) (k * z) = (k ^ 4) * P x y z

axiom symmetry (a b c : ℝ) : P a b c = P b c a

axiom zero_cond (a b : ℝ) : P a a b = 0

axiom initial_cond : P 1 2 3 = 1

theorem compute_P_2_4_8 : P 2 4 8 = 56 := sorry

end compute_P_2_4_8_l584_584146


namespace lizzie_wins_if_and_only_if_composite_l584_584920

theorem lizzie_wins_if_and_only_if_composite (n : ℕ) (h : n ≥ 3) :
  (∀ a : vector ℕ n, ∃ b : vector ℚ n, ∀ m : ℕ, Lizzie_wins (a, b, m) ↔ is_composite n) :=
sorry

end lizzie_wins_if_and_only_if_composite_l584_584920


namespace sum_largest_smallest_5_6_7_l584_584426

/--
Given the digits 5, 6, and 7, if we form all possible three-digit numbers using each digit exactly once, 
then the sum of the largest and smallest of these numbers is 1332.
-/
theorem sum_largest_smallest_5_6_7 : 
  let d1 := 5
  let d2 := 6
  let d3 := 7
  let smallest := 100 * d1 + 10 * d2 + d3
  let largest := 100 * d3 + 10 * d2 + d1
  smallest + largest = 1332 := 
by
  sorry

end sum_largest_smallest_5_6_7_l584_584426


namespace impossible_tiling_8x8_l584_584964

/--
It is impossible to cover an 8 × 8 rectangular floor perfectly using 15 tiles
of size 1 × 4 and 1 tile of size 2 × 2.
-/
theorem impossible_tiling_8x8 : ¬ (∃ (tiling : Array (Tile)), is_tiling tiling 64 ∧ 
  (count_tiles tiling (1, 4) = 15) ∧ 
  (count_tiles tiling (2, 2) = 1)) :=
sorry

-- Definitions for tiles and auxiliary functions
def Tile := (Nat × Nat)

def is_tiling (tiles : Array Tile) (total_squares : Nat) : Prop :=
  total_squares = tiles.foldl (λ acc tile, acc + tile.1 * tile.2) 0

def count_tiles (tiles : Array Tile) (size : Tile) : Nat :=
  tiles.foldl (λ acc tile, if tile = size then acc + 1 else acc) 0

end impossible_tiling_8x8_l584_584964


namespace cylinder_volume_relation_l584_584393

theorem cylinder_volume_relation (r h : ℝ) (π_pos : 0 < π) :
  (∀ B_h B_r A_h A_r : ℝ, B_h = r ∧ B_r = h ∧ A_h = h ∧ A_r = r 
   → 3 * (π * h^2 * r) = π * r^2 * h) → 
  ∃ N : ℝ, (π * (3 * h)^2 * h) = N * π * h^3 ∧ N = 9 :=
by 
  sorry

end cylinder_volume_relation_l584_584393


namespace smallest_n_satisfying_condition_l584_584032

noncomputable def f (x : ℝ) : ℝ := |real.cos (π * (x - real.floor x))|

theorem smallest_n_satisfying_condition :
  ∃ n : ℕ, n = 500 ∧ (∃ t : ℕ → ℝ, (∀ i < 1000, nf (t i f (t i f (t i)) = t i)) :=
begin
  sorry,
end

end smallest_n_satisfying_condition_l584_584032


namespace ratio_of_rectangle_sides_l584_584898

theorem ratio_of_rectangle_sides 
  (x y : ℝ) 
  (h : (x + y) - real.sqrt (x^2 + y^2) = (1 / 3) * y) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hxy : x < y) 
  : x / y = 5 / 12 := 
sorry

end ratio_of_rectangle_sides_l584_584898


namespace Jean_calls_thursday_l584_584010

theorem Jean_calls_thursday :
  ∃ (thursday_calls : ℕ), thursday_calls = 61 ∧ 
  (∃ (mon tue wed fri : ℕ),
    mon = 35 ∧ 
    tue = 46 ∧ 
    wed = 27 ∧ 
    fri = 31 ∧ 
    (mon + tue + wed + thursday_calls + fri = 40 * 5)) :=
sorry

end Jean_calls_thursday_l584_584010


namespace min_num_of_edge_values_l584_584897

def cube_edges_are_diffs (v : ℕ → ℕ) (i j : ℕ):=
  abs (v i - v j)

def min_diff_num_on_edges (v : ℕ → ℕ) :=
  ∀ (f v),
    (∃ (e : finset ℕ), 
      (∀ i j, i < j -> cube_edges_are_diffs v i j ∈ e) ∧ 
      (e.card = 3))

theorem min_num_of_edge_values {v : ℕ → ℕ} :
  (∀ i, i < 8 → v i ∈ finset.range 1 9) →
  min_diff_num_on_edges v :=
begin
  sorry, -- the proof will go here
end

end min_num_of_edge_values_l584_584897


namespace max_kings_on_chessboard_no_attack_l584_584250

def king_moves : ℤ × ℤ → set (ℤ × ℤ)
| (x, y) := { (x', y') | abs (x - x') ≤ 1 ∧ abs (y - y') ≤ 1 ∧ (x, y) ≠ (x', y') }

theorem max_kings_on_chessboard_no_attack (kings : set (ℤ × ℤ)) :
  (∀ (x, y) ∈ kings, ∀ (x', y') ∈ kings, (x, y) ≠ (x', y') → (x', y') ∉ king_moves (x, y)) →
  kings.card ≤ 16 := sorry

end max_kings_on_chessboard_no_attack_l584_584250


namespace solve_equation_l584_584975

theorem solve_equation (x : ℝ) : 3 * x * (x + 3) = 2 * (x + 3) ↔ (x = -3 ∨ x = 2/3) :=
by
  sorry

end solve_equation_l584_584975


namespace probability_between_C_and_D_is_five_eighths_l584_584058

noncomputable def AB : ℝ := 1
def AD : ℝ := AB / 4
def BC : ℝ := AB / 8
def pos_C : ℝ := AB - BC
def pos_D : ℝ := AD
def CD : ℝ := pos_C - pos_D

theorem probability_between_C_and_D_is_five_eighths : CD / AB = 5 / 8 :=
by
  simp [AB, AD, BC, pos_C, pos_D, CD]
  sorry

end probability_between_C_and_D_is_five_eighths_l584_584058


namespace problem1_problem2_l584_584448

variable (x a : ℝ)

def P := x^2 - 5*a*x + 4*a^2 < 0
def Q := (x^2 - 2*x - 8 <= 0) ∧ (x^2 + 3*x - 10 > 0)

theorem problem1 (h : 1 = a) (hP : P x a) (hQ : Q x) : 2 < x ∧ x ≤ 4 :=
sorry

theorem problem2 (h1 : ∀ x, ¬P x a → ¬Q x) (h2 : ∃ x, P x a ∧ ¬Q x) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l584_584448


namespace tetrahedron_plane_intersection_l584_584693

theorem tetrahedron_plane_intersection (T : Tetrahedron) :
  ¬∃ (P₁ P₂ : Plane), (P₁ ∩ T).is_square ∧ (P₂ ∩ T).is_square ∧
  ((P₁ ∩ T).side_length ≤ 1) ∧ ((P₂ ∩ T).side_length ≥ 100) :=
by
  sorry

end tetrahedron_plane_intersection_l584_584693


namespace height_of_triangle_l584_584979

theorem height_of_triangle
    (A : ℝ) (b : ℝ) (h : ℝ)
    (h1 : A = 30)
    (h2 : b = 12)
    (h3 : A = (b * h) / 2) :
    h = 5 :=
by
  sorry

end height_of_triangle_l584_584979


namespace maximum_garden_area_l584_584571

theorem maximum_garden_area :
  ∃ (l w : ℝ), (2 * l + 2 * w = 400) ∧
               (l ≥ 100) ∧
               (w ≥ 50) ∧
               (∀ l' w' : ℝ, (2 * l' + 2 * w' = 400) ∧
                             (l' ≥ 100) ∧
                             (w' ≥ 50) → (l' * w' ≤ l * w)) ∧
               (l * w = 10000) :=
begin
  sorry
end

end maximum_garden_area_l584_584571


namespace exists_infinite_B_with_1980_solutions_l584_584075

noncomputable def floor (z : ℝ) : ℤ := Real.floor z

theorem exists_infinite_B_with_1980_solutions :
  ∃ᶠ B : ℕ in at_top, ∃ (s : Finset (ℕ × ℕ)), s.card ≥ 1980 ∧
    ∀ (x y : ℕ), (x, y) ∈ s → floor (x ^ (3 / 2)) + floor (y ^ (3 / 2)) = B :=
begin
  sorry
end

end exists_infinite_B_with_1980_solutions_l584_584075


namespace does_not_determine_equilateral_triangle_l584_584682

theorem does_not_determine_equilateral_triangle :
  ¬ ∃ (altitude semi_perimeter : ℝ), ∀ (a b c : ℝ), 
  (altitude = a ∧ semi_perimeter = (a + b + c) / 2) →
  (equilateral a b c → (a = b ∧ b = c)) := 
sorry

end does_not_determine_equilateral_triangle_l584_584682


namespace shortest_distance_parabola_to_line_l584_584834

-- Define the parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Define the line x - y - 2 = 0
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- Distance formula from a point to a line
def distance_from_point_to_line (m : ℝ) : ℝ := abs (m - m^2 - 2) / sqrt 2

-- The statement to prove the shortest distance
theorem shortest_distance_parabola_to_line : 
  (∀ m : ℝ, distance_from_point_to_line m) ≥ 0 ∧
  (∃ m : ℝ, distance_from_point_to_line m = 7 * sqrt 2 / 8) := 
  sorry

end shortest_distance_parabola_to_line_l584_584834


namespace sin_60_eq_sqrt_three_div_two_l584_584358

theorem sin_60_eq_sqrt_three_div_two :
  Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_60_eq_sqrt_three_div_two_l584_584358


namespace molly_age_l584_584689

theorem molly_age
  (S M : ℕ)
  (h1 : S / M = 4 / 3)
  (h2 : S + 6 = 30) :
  M = 18 :=
sorry

end molly_age_l584_584689


namespace solve_equation_l584_584974

theorem solve_equation (x : ℝ) : 125 = 5 * 25^(x - 1) → x = 2 :=
by {
  sorry
}

end solve_equation_l584_584974


namespace limit_solution_l584_584344

noncomputable def limit_problem : Prop :=
  ∀ (x : ℝ), tendsto (λ x, (real.cot (x / 4)) ^ (1 / (real.cos (x / 2)))) (nhds x) (nhds (real.exp 1))

theorem limit_solution : limit_problem :=
sorry

end limit_solution_l584_584344


namespace units_digit_sum_l584_584673

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum :
  units_digit (24^3 + 17^3) = 7 :=
by
  sorry

end units_digit_sum_l584_584673


namespace sin_60_eq_sqrt3_div_2_l584_584363

theorem sin_60_eq_sqrt3_div_2 :
  ∃ (Q : ℝ × ℝ), dist Q (1, 0) = 1 ∧ angle (1, 0) Q = real.pi / 3 ∧ Q.2 = real.sqrt 3 / 2 := sorry

end sin_60_eq_sqrt3_div_2_l584_584363


namespace max_value_of_expression_l584_584773

theorem max_value_of_expression (θ : ℝ) (h : 0 < θ ∧ θ < π) :
  (∃ (x : ℝ), x = sin (θ / 2) * (1 + cos θ) ∧ ∀ y, (y = sin (θ / 2) * (1 + cos θ)) → y ≤ (4 * real.sqrt 3) / 9) :=
sorry

end max_value_of_expression_l584_584773


namespace largest_prime_divisor_of_16_squared_plus_63_squared_l584_584772

theorem largest_prime_divisor_of_16_squared_plus_63_squared :
  ∃ p, prime p ∧ p ∈ (nat.factors (16^2 + 63^2)) ∧ (∀ q, prime q ∧ q ∈ (nat.factors (16^2 + 63^2)) → q ≤ p) :=
begin
  have h : 16^2 + 63^2 = 4225 := by norm_num,
  rw h,
  use 5,
  split,
  { norm_num },

  split,
  { 
    rw nat.factors_4225, -- assumes we have proved factors of 4225 is [5]
    simp,
  },

  intros q hq1 hq2,
  rw nat.factors_4225 at hq2,
  simp at hq2,
  exact list.le_max_elem_of_mem_nat hq2,
end

end largest_prime_divisor_of_16_squared_plus_63_squared_l584_584772


namespace proof_problem_l584_584929

variables {a b c : ℝ}

theorem proof_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 4 * a^2 + b^2 + 16 * c^2 = 1) :
  (0 < a * b ∧ a * b < 1 / 4) ∧ (1 / a^2 + 1 / b^2 + 1 / (4 * a * b * c^2) > 49) :=
by
  sorry

end proof_problem_l584_584929


namespace loss_percentage_is_correct_l584_584300

noncomputable def watch_loss_percentage (SP_loss SP_profit : ℕ) (profit_percentage : ℕ) : ℕ :=
  let CP := SP_profit / (1 + profit_percentage / 100) in
  let loss := CP - SP_loss in
  loss * 100 / CP

theorem loss_percentage_is_correct :
  watch_loss_percentage 1140 1260 5 = 5 :=
by
  sorry

end loss_percentage_is_correct_l584_584300


namespace parallel_AA1_BB1_l584_584803

open EuclideanGeometry

-- Let A, B, M, and N be points on a circle.
variable {A B M N A₁ B₁ : Point}

-- Chord MA₁ is drawn from point M and is perpendicular to line NB.
axiom MA1_perpendicular_NB : Perpendicular (line_through M A₁) (line_through N B)

-- Chord MB₁ is drawn from point M and is perpendicular to line NA.
axiom MB1_perpendicular_NA : Perpendicular (line_through M B₁) (line_through N A)

-- We need to prove that AA₁ is parallel to BB₁.
theorem parallel_AA1_BB1 : Parallel (line_through A A₁) (line_through B B₁) := sorry

end parallel_AA1_BB1_l584_584803


namespace simplify_sqrt_sum_l584_584097

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l584_584097


namespace distance_after_12_sec_time_to_travel_380_meters_l584_584624

-- Define the function expressing the distance s in terms of the travel time t
def distance (t : ℝ) : ℝ := 9 * t + (1 / 2) * t^2

-- Proof problem 1: Distance traveled after 12 seconds
theorem distance_after_12_sec : distance 12 = 180 := 
sorry

-- Proof problem 2: Time to travel 380 meters
theorem time_to_travel_380_meters (t : ℝ) (h : distance t = 380) : t = 20 := 
sorry

end distance_after_12_sec_time_to_travel_380_meters_l584_584624


namespace angle_DCE_const_l584_584272

noncomputable theory

variables {A B P C D E : Point}
variables (h : Circle) (U V : Point)

-- Define the various points and their relationships as per the conditions
def on_semicircle (h : Circle) (P : Point) : Prop := -- P is on AB diameter, with corresponding perpendicular creating C on h...

def perpendicular (P C : Point) (AB : Line) : Prop := -- PC is perpendicular to AB...

def inscribed_circles (AB PC : Line) (h : Circle) (D E : Point) : Prop := -- Define the inscribed circles touching AB, PC...

-- Proposition to prove
theorem angle_DCE_const (h : Circle) (A B P C D E : Point) 
  (P_on_AB : on_AB A B P) 
  (PC_perpendicular_AB : perpendicular P C AB)
  (PC_intersects_h_at_C : intersects h P C)
  (inscribed_circles_AB_PC_h : inscribed_circles AB PC h D E) : 
  ∀ P on AB, ∠ (Line.mk D C) (Line.mk C E) = 45 :=
by
  sorry

end angle_DCE_const_l584_584272


namespace number_of_extreme_points_range_of_a_nonnegative_l584_584825

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ :=
  x + a * (exp (2 * x) - 3 * exp x + 2)

-- Define the derivative f'(x)
def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  2 * a * exp (2 * x) - 3 * a * exp x + 1

-- Result 1: Number of extreme points based on a
theorem number_of_extreme_points (a : ℝ) :
  (a = 0 → ∀ x : ℝ, f_prime x a > 0) ∧ 
  (a < 0 → ∃ x : ℝ, f_prime x a = 0) ∧ 
  (a > 8/9 → ∃ x1 x2 : ℝ, x1 < x2 ∧ f_prime x1 a = 0 ∧ f_prime x2 a = 0) := sorry

-- Result 2: Range of a such that ∀ x > 0, f(x) ≥ 0
theorem range_of_a_nonnegative :
  (∀ x > 0, f x a ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 := sorry

end number_of_extreme_points_range_of_a_nonnegative_l584_584825


namespace sin_sides_of_triangle_l584_584041

theorem sin_sides_of_triangle {a b c : ℝ} 
  (habc: a + b > c) (hbac: a + c > b) (hcbc: b + c > a) (h_sum: a + b + c ≤ 2 * Real.pi) :
  a > 0 ∧ a < Real.pi ∧ b > 0 ∧ b < Real.pi ∧ c > 0 ∧ c < Real.pi ∧ 
  (Real.sin a + Real.sin b > Real.sin c) ∧ 
  (Real.sin a + Real.sin c > Real.sin b) ∧ 
  (Real.sin b + Real.sin c > Real.sin a) :=
by
  sorry

end sin_sides_of_triangle_l584_584041


namespace ms_walker_drives_24_miles_each_way_l584_584574

theorem ms_walker_drives_24_miles_each_way
  (D : ℝ)
  (H1 : 1 / 60 * D + 1 / 40 * D = 1) :
  D = 24 := 
sorry

end ms_walker_drives_24_miles_each_way_l584_584574


namespace no_two_digit_N_satisfies_condition_l584_584489

theorem no_two_digit_N_satisfies_condition :
  ∀ (N : ℕ), 10 ≤ N ∧ N ≤ 99 →
  let t := N / 10 in
  let u := N % 10 in
  (10 * t + u) + (10 * u + t) ≠ 21 :=
by
  sorry

end no_two_digit_N_satisfies_condition_l584_584489


namespace smallest_perimeter_of_triangle_PQR_l584_584226

noncomputable def triangle_PQR_perimeter (PQ PR QR : ℕ) (QJ : ℝ) 
  (h1 : PQ = PR) (h2 : QJ = 10) : ℕ :=
2 * (PQ + QR)

theorem smallest_perimeter_of_triangle_PQR (PQ PR QR : ℕ) (QJ : ℝ) :
  PQ = PR → QJ = 10 → 
  ∃ p, p = triangle_PQR_perimeter PQ PR QR QJ (by assumption) (by assumption) ∧ p = 78 :=
sorry

end smallest_perimeter_of_triangle_PQR_l584_584226


namespace find_constant_N_l584_584395

variables (r h V_A V_B : ℝ)

theorem find_constant_N 
  (h_eq_r : h = r) 
  (r_eq_h : r = h) 
  (vol_relation : V_A = 3 * V_B) 
  (vol_A : V_A = π * r^2 * h) 
  (vol_B : V_B = π * h^2 * r) : 
 ∃ N : ℝ, V_A = N * π * h^3 ∧ N = 9 := 
by 
  use 9
  split
  sorry  -- Proof that V_A = 9 * π * h^3 goes here
  exact eq.refl 9  -- This confirms N = 9 without further proof.


end find_constant_N_l584_584395


namespace square_division_l584_584590

theorem square_division (n : ℕ) (h : n ≥ 6) : ∃ squares : ℕ, squares = n ∧ can_divide_into_squares(squares) := 
sorry

end square_division_l584_584590


namespace variance_of_ξ_l584_584167

-- Define the random variable ξ and its probabilities
variables (ξ : ℕ → ℝ) (p0 p1 p2 : ℝ)

-- Hypotheses
def H1 : Prop := ξ 0 = 0
def H2 : Prop := ξ 1 = 1
def H3 : Prop := ξ 2 = 2
def H4 : Prop := ∑ x in {0,1,2}, if x = 0 then p0 else if x = 1 then p1 else p2 = 1
def H5 : Prop := p0 = 1 / 5
def H6 : Prop := (0:ℝ) * p0 + 1 * p1 + 2 * p2 = 1

-- Statement of the problem
theorem variance_of_ξ (H1 : H1) (H2 : H2) (H3 : H3) (H4 : H4) (H5 : H5) (H6 : H6) :
  (0 - 1)^2 * p0 + (1 - 1)^2 * p1 + (2 - 1)^2 * p2 = 2 / 5 :=
sorry

end variance_of_ξ_l584_584167


namespace find_actual_balance_l584_584265

-- Define the given conditions
def current_balance : ℝ := 90000
def rate : ℝ := 0.10

-- Define the target
def actual_balance_before_deduction (X : ℝ) : Prop :=
  (X * (1 - rate) = current_balance)

-- Statement of the theorem
theorem find_actual_balance : ∃ X : ℝ, actual_balance_before_deduction X :=
  sorry

end find_actual_balance_l584_584265


namespace convert_base8_to_base7_l584_584383

theorem convert_base8_to_base7 : (536%8).toBase 7 = 1010%7 :=
by
  sorry

end convert_base8_to_base7_l584_584383


namespace probability_colors_match_l584_584739

noncomputable def prob_abe_shows_blue : ℚ := 2 / 4
noncomputable def prob_bob_shows_blue : ℚ := 3 / 6
noncomputable def prob_abe_shows_green : ℚ := 2 / 4
noncomputable def prob_bob_shows_green : ℚ := 1 / 6

noncomputable def prob_same_color : ℚ :=
  (prob_abe_shows_blue * prob_bob_shows_blue) + (prob_abe_shows_green * prob_bob_shows_green)

theorem probability_colors_match : prob_same_color = 1 / 3 :=
by
  sorry

end probability_colors_match_l584_584739


namespace iterate_F_l584_584037

def F (x : ℝ) : ℝ := x^3 + 3*x^2 + 3*x

theorem iterate_F (x : ℝ) : (Nat.iterate F 2017 x) = (x + 1)^(3^2017) - 1 :=
by
  sorry

end iterate_F_l584_584037


namespace part_a_part_b_l584_584718

-- Definitions based on the conditions:
def probability_of_hit (p : ℝ) := p
def probability_of_miss (p : ℝ) := 1 - p

-- Condition: exactly three unused rockets after firing at five targets
def exactly_three_unused_rockets (p : ℝ) : ℝ := 10 * (probability_of_hit p) ^ 3 * (probability_of_miss p) ^ 2

-- Condition: expected number of targets hit when there are nine targets
def expected_targets_hit (p : ℝ) : ℝ := 10 * p - p ^ 10

-- Lean 4 statements representing the proof problems:
theorem part_a (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) : 
  exactly_three_unused_rockets p = 10 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

theorem part_b (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) :
  expected_targets_hit p = 10 * p - p ^ 10 :=
by sorry

end part_a_part_b_l584_584718


namespace joel_current_age_l584_584534

-- Define the age of Joel and his dad
variables (J D : ℕ)

-- The conditions given in the problem
axiom dad_age : D = 32
axiom future_age_condition : 27 = J + (54 - D)

-- The theorem to prove Joel's current age
theorem joel_current_age : J = 5 :=
by
  -- Initial setup and axioms
  have h1 : D = 32 := dad_age,
  have h2 : 27 = J + (54 - D) := future_age_condition,
  
  -- Start the proof, but "sorry" will be used as a placeholder to skip the proof steps
  sorry

end joel_current_age_l584_584534


namespace min_possible_area_BI1I2_l584_584551

noncomputable def min_area_BI1I2 (AB BC AC : ℝ) (hAB : AB = 40) (hBC : BC = 24) (hAC : AC = 32) : ℝ :=
  let A := (0 : ℝ, 0 : ℝ)
  let B := (40 : ℝ, 0 : ℝ)
  let C := (32 * (BC / sqrt (AB ^ 2 + AC ^ 2 - 2 * AB * AC * cos (acos ((AB ^ 2 + AC ^ 2 - BC ^ 2) / (2 * AB * AC))))), 0 : ℝ)
  let Y := λ k : ℝ, (k * AC + 0 * (1 - k), 0 * k + 0 * (1 - k))
  let I1 := incent_center (A, B, Y k)
  let I2 := incent_center (B, C, Y k)
  let area_BI1I2 := area (B, I1, I2)
  in min (area_BI1I2)

theorem min_possible_area_BI1I2 : min_area_BI1I2 40 24 32 = 96 :=
sorry

end min_possible_area_BI1I2_l584_584551


namespace median_of_81_consecutive_integers_l584_584197

theorem median_of_81_consecutive_integers (s : ℕ) (h_sum : s = 9^5) : 
  let n := 81 in
  let median := s / n in
  median = 729 :=
by
  have h₁ : 9^5 = 59049 := by norm_num
  have h₂ : 81 = 81 := rfl
  have h₃ : 59049 / 81 = 729 := by norm_num
  
  -- Apply the conditions
  rw [h_sum, <-h₁] at h_sum
  rw h₂
  
  -- Conclude the median
  exact h₃

end median_of_81_consecutive_integers_l584_584197


namespace no_solution_system_l584_584868

theorem no_solution_system (a : ℝ) : 
  (∀ x : ℝ, (x - 2 * a > 0) → (3 - 2 * x > x - 6) → false) ↔ a ≥ 3 / 2 :=
by
  sorry

end no_solution_system_l584_584868


namespace f_f_f_f_f_problem_solution_l584_584855

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem f_f_f_f_f (x : ℝ) [hx : x ≠ 0] : f (f (f (f (f x)))) = x :=
by sorry

theorem problem_solution : f (f (f (f (f 2)))) = 1 / 2 :=
by
  have hx : 2 ≠ 0 := by norm_num
  exact f_f_f_f_f 2 hx

end f_f_f_f_f_problem_solution_l584_584855


namespace same_numbers_on_both_boards_l584_584244

-- Defining the conditions as hypotheses in Lean

structure LibraryActivity where
  n_in : List Nat    -- List of numbers of readers present on entry
  n_out : List Nat   -- List of numbers of readers remaining on exit

-- Theorem stating that the two lists n_in and n_out are permutations of each other
theorem same_numbers_on_both_boards (activity : LibraryActivity) (h1 : activity.n_in.head = 0)
  (h2 : activity.n_in.last = activity.n_out.head) (h3 : activity.n_out.last = 0) :
  activity.n_in ~ activity.n_out :=
sorry

end same_numbers_on_both_boards_l584_584244


namespace middle_number_consecutive_even_l584_584208

theorem middle_number_consecutive_even (a b c : ℤ) 
  (h1 : a = b - 2) 
  (h2 : c = b + 2) 
  (h3 : a + b = 18) 
  (h4 : a + c = 22) 
  (h5 : b + c = 28) : 
  b = 11 :=
by sorry

end middle_number_consecutive_even_l584_584208


namespace smallest_angle_convex_15_polygon_l584_584986

theorem smallest_angle_convex_15_polygon :
  ∃ (a : ℕ) (d : ℕ), (∀ n : ℕ, n ∈ Finset.range 15 → (a + n * d < 180)) ∧
  15 * (a + 7 * d) = 2340 ∧ 15 * d <= 24 -> a = 135 :=
by
  -- Proof omitted
  sorry

end smallest_angle_convex_15_polygon_l584_584986


namespace Kim_total_hours_l584_584913

-- Define the initial conditions
def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_class : ℕ := 1

-- The proof problem: Given the initial conditions, prove the total hours of classes per day is 6
theorem Kim_total_hours : (initial_classes - dropped_class) * hours_per_class = 6 := by
  sorry

end Kim_total_hours_l584_584913


namespace sergey_max_rows_l584_584081

-- Definitions based on conditions
def sergey_numbers : list ℕ := list.range' 500 1000
def next_row (l : list ℕ) : list ℕ := 
  list.zip_with nat.gcd l (list.tail l)

-- Question and answer tuple
theorem sergey_max_rows :
  ∀ l : list ℕ, l.length = 1000 → 
  (∀ i : ℕ, i < 1000 → 500 ≤ l.nth_le i sorry ∧ l.nth_le i sorry ≤ 1499) →
  ∃ n : ℕ, ∀ r : list ℕ, r.length = n →
  (∀ i : ℕ, i < r.length → r.nth_le i sorry = 1) → 
  n ≤ 501 :=
sorry

end sergey_max_rows_l584_584081


namespace proportion_third_number_l584_584495

theorem proportion_third_number
  (x : ℝ) (y : ℝ)
  (h1 : 0.60 * 4 = x * y)
  (h2 : x = 0.39999999999999997) :
  y = 6 :=
by
  sorry

end proportion_third_number_l584_584495


namespace max_label_number_l584_584757

/-- 
Daniel Decimal has numerous 0's, 1's, 3's, 4's, 5's, 6's, 7's, 8's, and 9's, but only twenty-five 2's. 
We want to prove that the farthest number Daniel can label on his number wall is 152.
-/
theorem max_label_number (numerous_digits : list ℕ) 
  (num_2s : ℕ) 
  (valid_digits : numerous_digits = [0, 1, 3, 4, 5, 6, 7, 8, 9])
  (limited_twos : num_2s = 25) : 
  ∃ max_n : ℕ, max_n = 152 :=
  sorry

end max_label_number_l584_584757


namespace median_of_81_consecutive_integers_l584_584180

theorem median_of_81_consecutive_integers (ints : ℕ → ℤ) 
  (h_consecutive: ∀ n : ℕ, ints (n+1) = ints n + 1) 
  (h_sum: (∑ i in finset.range 81, ints i) = 9^5) : 
  (ints 40) = 9^3 := 
sorry

end median_of_81_consecutive_integers_l584_584180


namespace triangle_circumcenter_similarity_l584_584440

theorem triangle_circumcenter_similarity
  (A B C P O1 O2 : Type*)
  [triangle_ABC : triangle A B C]
  (H1 : P ∈ segment B C)
  (H2 : is_circumcenter O1 A P B)
  (H3 : is_circumcenter O2 A P C) :
  similar (triangle A O1 O2) (triangle A B C) :=
sorry -- Proof left as an exercise

end triangle_circumcenter_similarity_l584_584440


namespace palindrome_sum_digits_l584_584559

def is_palindrome (n : ℕ) : Prop :=
  let str := n.toString
  str = str.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  n.toString.toList.foldr (λ c acc => acc + c.toNat - '0'.toNat) 0

theorem palindrome_sum_digits (x : ℕ) (h1 : is_palindrome x) (h2 : is_palindrome (x + 50)) 
  (h3 : 1000 ≤ x ∧ x ≤ 9999) : sum_of_digits x = 30 :=
  sorry

end palindrome_sum_digits_l584_584559


namespace median_of_81_consecutive_integers_l584_584195

theorem median_of_81_consecutive_integers (sum : ℕ) (h₁ : sum = 9^5) : 
  let mean := sum / 81 in mean = 729 :=
by
  have h₂ : sum = 59049 := by
    rw h₁
    norm_num
  have h₃ : mean = 59049 / 81 := by
    rw h₂
    rfl
  have result : mean = 729 := by
    rw h₃
    norm_num
  exact result

end median_of_81_consecutive_integers_l584_584195


namespace orthocenter_lies_on_OI_l584_584019

noncomputable def problem_statement (ABC : Triangle) (O I T1 T2 T3 : Point) : Prop :=
  let T := Triangle.mk T1 T2 T3 in
  let inc ABC = I incenter ABC in
  let circumcenter ABC = O in
  ∃ H : Point, H = orthocenter T ∧ collinear [O, I, H]

axiom triangle_not_equilateral (ABC : Triangle) : ¬equilateral_t ABC

theorem orthocenter_lies_on_OI 
  (ABC : Triangle)
  (O I T1 T2 T3 : Point)
  (h1 : incenter ABC = I)
  (h2 : circumcenter ABC = O)
  (h3 : tangency_point ABC.incircle BC = T1)
  (h4 : tangency_point ABC.incircle CA = T2)
  (h5 : tangency_point ABC.incircle AB = T3)
  (h6 : triangle_not_equilateral ABC) :
  ∃ H : Point, H = orthocenter (Triangle.mk T1 T2 T3) ∧ collinear [O, I, H] :=
sorry

end orthocenter_lies_on_OI_l584_584019


namespace factor_formulas_l584_584618

theorem factor_formulas (x y a b m n : ℝ) :
  (¬ ∃ p q : ℝ, -x^2 - y^2 = p * q) ∧
  (∃ p q : ℝ, - (1/4) * a^2 * b^2 + 1 = p * q) ∧
  (¬ ∃ p q : ℝ, a^2 + a * b + b^2 = p * q) ∧
  (∃ r s : ℝ, (1/4) - m * n + m^2 * n^2 = r * s) :=
by
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

end factor_formulas_l584_584618


namespace simplify_sqrt_sum_l584_584134

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l584_584134


namespace sally_balloons_l584_584971

theorem sally_balloons (F S : ℕ) (h1 : F = 3 * S) (h2 : F = 18) : S = 6 :=
by sorry

end sally_balloons_l584_584971


namespace intersection_eq_l584_584567

open Set

def setA : Set ℤ := {x | x ≥ -4}
def setB : Set ℤ := {x | x ≤ 3}

theorem intersection_eq : (setA ∩ setB) = {x | -4 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_eq_l584_584567


namespace meeting_time_eqn_l584_584656

-- Mathematical definitions derived from conditions:
def distance := 270 -- Cities A and B are 270 kilometers apart.
def speed_fast_train := 120 -- Speed of the fast train is 120 km/h.
def speed_slow_train := 75 -- Speed of the slow train is 75 km/h.
def time_head_start := 1 -- Slow train departs 1 hour before the fast train.

-- Let x be the number of hours it takes for the two trains to meet after the fast train departs
def x : Real := sorry

-- Proving the equation representing the situation:
theorem meeting_time_eqn : 75 * 1 + (120 + 75) * x = 270 :=
by
  sorry

end meeting_time_eqn_l584_584656


namespace matrix_eq_l584_584026

open Matrix

def matA : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 3], ![4, 2]]
def matI : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem matrix_eq (A : Matrix (Fin 2) (Fin 2) ℤ)
  (hA : A = ![![1, 3], ![4, 2]]) :
  A ^ 7 = 9936 * A ^ 2 + 12400 * 1 :=
  by
    sorry

end matrix_eq_l584_584026


namespace find_k_l584_584866

theorem find_k (k : ℝ) (h1 : k > 0) (h2 : ∀ x ∈ Set.Icc (2 : ℝ) 4, y = k / x → y ≥ 5) : k = 20 :=
sorry

end find_k_l584_584866


namespace solve_system_of_equations_l584_584608

theorem solve_system_of_equations :
  ∃ x y: ℤ, (2 * x - 3 * y = 5) ∧ (3 * x + y = 2) ∧ (x = 1) ∧ (y = -1) :=
by {
  existsi 1,
  existsi (-1),
  split; norm_num; split; norm_num; split; norm_num; norm_num
}

end solve_system_of_equations_l584_584608


namespace sequence_a_general_formula_sum_of_sequence_b_l584_584798

noncomputable def a (n : ℕ) : ℕ := 3 * n + 2

theorem sequence_a_general_formula 
  (a_2_eq : a 2 = 8) 
  (S₁₀_eq : (∑ i in Finset.range 10, a (i+1)) = 185) :
  ∀ n : ℕ, a n = 3 * n + 2 := 
by 
  sorry

noncomputable def b (n : ℕ) : ℕ := a (3^n)

theorem sum_of_sequence_b (n : ℕ) :
  (∑ i in Finset.range n, b (i+1)) = (1 / 2 : ℚ) * 3^(n+2) + 2 * n - 9 / 2 := 
by 
  sorry

end sequence_a_general_formula_sum_of_sequence_b_l584_584798


namespace part_a_part_b_l584_584720

-- Definitions based on the conditions:
def probability_of_hit (p : ℝ) := p
def probability_of_miss (p : ℝ) := 1 - p

-- Condition: exactly three unused rockets after firing at five targets
def exactly_three_unused_rockets (p : ℝ) : ℝ := 10 * (probability_of_hit p) ^ 3 * (probability_of_miss p) ^ 2

-- Condition: expected number of targets hit when there are nine targets
def expected_targets_hit (p : ℝ) : ℝ := 10 * p - p ^ 10

-- Lean 4 statements representing the proof problems:
theorem part_a (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) : 
  exactly_three_unused_rockets p = 10 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

theorem part_b (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) :
  expected_targets_hit p = 10 * p - p ^ 10 :=
by sorry

end part_a_part_b_l584_584720


namespace median_of_81_consecutive_integers_l584_584206

theorem median_of_81_consecutive_integers (S : ℤ) 
  (h1 : ∃ l : ℤ, ∀ k, (0 ≤ k ∧ k < 81) → (l + k) ∈ S) 
  (h2 : S = 9^5) : 
  (81 * 729 = 9^5) :=
by
  have h_sum : (S : ℤ) = ∑ i in ((finset.range 81).map (λ n, l + n)), id i, from sorry
  have h_eq : (81 * 729 = 9^5) := calc
    81 * 729 = 9^2 * 9^3 : by norm_num
    ... = 9^(2+3) : by ring
    ... = 9^5 : by norm_num
  exact h_eq

end median_of_81_consecutive_integers_l584_584206


namespace attempts_required_to_open_safe_l584_584951

/-- The set of digits available on the lock buttons -/
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- Define a valid password as a tuple of three digits -/
def password := Fin (10 × 10 × 10)

/-- Check if two passwords have at least two matching digits in corresponding positions -/
def matches_at_least_two (p1 p2 : password) : Bool :=
  let ⟨a1, a2, a3⟩ := p1
  let ⟨b1, b2, b3⟩ := p2
  (a1 = b1 ∧ a2 = b2) ∨ (a1 = b1 ∧ a3 = b3) ∨ (a2 = b2 ∧ a3 = b3)

/-- Prove that 50 attempts are necessary to guarantee that the safe opens-/
theorem attempts_required_to_open_safe : ∃ (attempts : List password), attempts.length = 50 ∧ ∀ pw : password, ∃ t ∈ attempts, matches_at_least_two pw t = true :=
by
  sorry

end attempts_required_to_open_safe_l584_584951


namespace milkman_total_profit_l584_584298

-- Declare the conditions
def initialMilk : ℕ := 50
def initialWater : ℕ := 15
def firstMixtureMilk : ℕ := 30
def firstMixtureWater : ℕ := 8
def remainingMilk : ℕ := initialMilk - firstMixtureMilk
def secondMixtureMilk : ℕ := remainingMilk
def secondMixtureWater : ℕ := 7
def costOfMilkPerLiter : ℕ := 20
def sellingPriceFirstMixturePerLiter : ℕ := 17
def sellingPriceSecondMixturePerLiter : ℕ := 15
def totalCostOfMilk := (firstMixtureMilk + secondMixtureMilk) * costOfMilkPerLiter
def totalRevenueFirstMixture := (firstMixtureMilk + firstMixtureWater) * sellingPriceFirstMixturePerLiter
def totalRevenueSecondMixture := (secondMixtureMilk + secondMixtureWater) * sellingPriceSecondMixturePerLiter
def totalRevenue := totalRevenueFirstMixture + totalRevenueSecondMixture
def totalProfit := totalRevenue - totalCostOfMilk

-- Proof statement
theorem milkman_total_profit : totalProfit = 51 := by
  sorry

end milkman_total_profit_l584_584298


namespace shift_graph_l584_584217

theorem shift_graph :
  ∀ x, (∃ c : ℝ, c = 1 ∧ y = 3^(x + c)) ↔ y = 3^(x + 1) :=
by sorry

end shift_graph_l584_584217


namespace ellipse_standard_equation_triangle_area_range_l584_584800

-- Define the given properties of the ellipse
variables (a b c : ℝ) (F₁ F₂ : ℝ × ℝ) (eccentricity : ℝ)
hypothesis h1 : F₁ = (0, -1)
hypothesis h2 : eccentricity = real.sqrt 3 / 3
hypothesis h3 : c = 1
define ellipse_equation := ∀ x y : ℝ, (y^2)/(a^2) + (x^2)/(b^2) = 1

theorem ellipse_standard_equation : 
  (a = sqrt 3) → (b = sqrt (a^2 - c^2)) → ellipse_equation x y := 
by { sorry }

theorem triangle_area_range : 
  (∀ x1 x2 : ℝ, ∃ (F₂ : ℝ × ℝ), (abs (x1 - x2) ∈ set.Icc 0 (4 * sqrt 3 / 3))) :=
by { sorry }

end ellipse_standard_equation_triangle_area_range_l584_584800


namespace smallest_perimeter_iso_triangle_l584_584237

theorem smallest_perimeter_iso_triangle :
  ∃ (x y : ℕ), (PQ = PR ∧ PQ = x ∧ PR = x ∧ QR = y ∧ QJ = 10 ∧ PQ + PR + QR = 416 ∧ 
  PQ = PR ∧ y = 8 ∧ 2 * (x + y) = 416 ∧ y^2 - 50 > 0 ∧ y < 10) :=
sorry

end smallest_perimeter_iso_triangle_l584_584237


namespace median_of_81_consecutive_integers_l584_584183

theorem median_of_81_consecutive_integers (ints : ℕ → ℤ) 
  (h_consecutive: ∀ n : ℕ, ints (n+1) = ints n + 1) 
  (h_sum: (∑ i in finset.range 81, ints i) = 9^5) : 
  (ints 40) = 9^3 := 
sorry

end median_of_81_consecutive_integers_l584_584183


namespace roots_complex_nonreal_for_pure_imaginary_k_l584_584379

theorem roots_complex_nonreal_for_pure_imaginary_k (k : ℂ) (hk : k.im ≠ 0 ∧ k.re = 0) :
  ∀ (z : ℂ), (2 * z^2 + 5 * complex.I * z = k) → z.re ≠ 0 :=
by
  sorry

end roots_complex_nonreal_for_pure_imaginary_k_l584_584379


namespace find_two_adjacent_triangles_diff_more_than_3_l584_584332

theorem find_two_adjacent_triangles_diff_more_than_3 :
  ∃ (u v : ℕ), u ∈ finset.range 26 ∧ v ∈ finset.range 26 ∧ (abs (u - v) > 3) ∧ adjacent u v :=
sorry

end find_two_adjacent_triangles_diff_more_than_3_l584_584332


namespace determinant_squared_l584_584940

variables (u v w : ℝ^3)

def matrix1 : Matrix 3 3 ℝ := ![
  u + 2 • v,
  v + 2 • w,
  w + 2 • u
]

def E : ℝ := Matrix.det matrix1

def matrix2 : Matrix 3 3 ℝ := ![
  u × (v + 2 • w),
  (v + 2 • w) × (w + 2 • u),
  (w + 2 • u) × (u + 2 • v)
]

theorem determinant_squared (E : ℝ) : Matrix.det matrix2 = E ^ 2 :=
by
  sorry

end determinant_squared_l584_584940


namespace driving_distance_l584_584336

theorem driving_distance:
  ∀ a b: ℕ, (a + b = 500 ∧ a ≥ 150 ∧ b ≥ 150) → 
  (⌊Real.sqrt (a^2 + b^2)⌋ = 380) :=
by
  intro a b
  intro h
  sorry

end driving_distance_l584_584336


namespace base5_2004_to_decimal_is_254_l584_584808

def base5_to_decimal (n : Nat) : Nat :=
  match n with
  | 2004 => 2 * 5^3 + 0 * 5^2 + 0 * 5^1 + 4 * 5^0
  | _ => 0

theorem base5_2004_to_decimal_is_254 :
  base5_to_decimal 2004 = 254 :=
by
  -- Proof goes here
  sorry

end base5_2004_to_decimal_is_254_l584_584808


namespace smallest_n_partition_l584_584934

def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 40 }

theorem smallest_n_partition (n : ℕ) : 
  (∃ (part : Finset (Finset ℕ)), part.card = n ∧ 
    (∀ s ∈ part, s ⊆ M ∧ ∀ a b c ∈ s, a ≠ b + c)) ↔ n = 4 := sorry

end smallest_n_partition_l584_584934


namespace smallest_perimeter_triangle_l584_584241

theorem smallest_perimeter_triangle (PQ PR QR : ℕ) (J : Point) :
  PQ = PR →
  QJ = 10 →
  QR = 2 * 10 →
  PQ + PR + QR = 40 :=
by
  sorry

structure Point : Type :=
mk :: (QJ : ℕ)

noncomputable def smallest_perimeter_triangle : Prop :=
  ∃ (PQ PR QR : ℕ) (J : Point), PQ = PR ∧ J.QJ = 10 ∧ QR = 2 * 10 ∧ PQ + PR + QR = 40

end smallest_perimeter_triangle_l584_584241


namespace probability_between_CD_l584_584070

-- Define the points A, B, C, and D on a line segment
variables (A B C D : ℝ)

-- Provide the conditions
axiom h_ab_ad : B - A = 4 * (D - A)
axiom h_ab_bc : B - A = 8 * (B - C)

-- Define the probability statement 
theorem probability_between_CD (AB length: ℝ) : 
  (0 ≤ A ∧ A < D ∧ D < C ∧ C < B) → (B - A = 1) → 
  (B - A = 4 * (D - A)) → (B - A = 8 * (B - C)) → 
  probability (A B C D) = 5 / 8 :=
by
  sorry

end probability_between_CD_l584_584070


namespace no_consecutive_red_lights_l584_584576

-- Definitions for initial conditions
def n := 8 -- number of traffic lights
def p_red := 0.4 -- probability that a light is red
def p_green := 0.6 -- probability that a light is green

-- Theorem statement
theorem no_consecutive_red_lights :
  let P := 0.6^8 * nat.choose 9 0 
           + 0.6^7 * 0.4 * nat.choose 8 1
           + 0.6^6 * 0.4^2 * nat.choose 7 2
           + 0.6^5 * 0.4^3 * nat.choose 6 3
           + 0.6^4 * 0.4^4 * nat.choose 5 4
  in  P ≈ 0.35 :=
by
  sorry

end no_consecutive_red_lights_l584_584576


namespace tiao_ri_method_third_approximation_l584_584148

theorem tiao_ri_method_third_approximation :
  ∀ (a b c d : ℕ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) (h_d : d > 0),
  let x : ℝ := 3.14159 in
  let first_approx := (b + d) / (a + c) in
  let second_approx := (b + 2*d) / (a + 2*c) in
  let third_approx := (2*b + 3*d) / (2*a + 3*c) in
  a = 10 ∧ b = 31 ∧ c = 15 ∧ d = 49 →
  first_approx = 16 / 5 ∧ second_approx = 47 / 15 ∧ third_approx = 63 / 20 :=
begin
  intros a b c d h_a h_b h_c h_d x first_approx second_approx third_approx h,
  -- Proof omitted
  sorry,
end

end tiao_ri_method_third_approximation_l584_584148


namespace first_player_can_always_make_A_eq_6_l584_584051

def maxSum3x3In5x5Board (board : Fin 5 → Fin 5 → ℕ) (i j : Fin 3) : ℕ :=
  (i + 3 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 3 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 3 : Fin 5) * (j + 5 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 5 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 5 : Fin 5)

theorem first_player_can_always_make_A_eq_6 :
  ∀ (board : Fin 5 → Fin 5 → ℕ), 
  (∀ (i j : Fin 3), maxSum3x3In5x5Board board i j = 6)
  :=
by
  intros board i j
  sorry

end first_player_can_always_make_A_eq_6_l584_584051


namespace intersection_points_eval_l584_584832

noncomputable theory

open Real

theorem intersection_points_eval
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 + y1 = 5)
  (h2 : x2 + y2 = 5)
  (h3 : x1^2 + y1^2 = 16)
  (h4 : x2^2 + y2^2 = 16)
  (h5 : x1 ≠ x2 ∨ y1 ≠ y2) :
  x1 * y2 + x2 * y1 = 16 :=
sorry

end intersection_points_eval_l584_584832


namespace complementary_event_l584_584078

-- Definitions based on the conditions
def EventA (products : List Bool) : Prop := 
  (products.filter (λ x => x = true)).length ≥ 2

def complementEventA (products : List Bool) : Prop := 
  (products.filter (λ x => x = true)).length ≤ 1

-- Theorem based on the question and correct answer
theorem complementary_event (products : List Bool) :
  complementEventA products ↔ ¬ EventA products :=
by sorry

end complementary_event_l584_584078


namespace same_suit_same_row_l584_584147

theorem same_suit_same_row
  (deck : ℕ → ℕ → (ℕ × ℕ))
  (h_deck : ∀ i j, i < 13 → j < 4 → (deck i j).fst ∈ {1, 2, 3, 4} ∧ (deck i j).snd ∈ finset.range 13)
  (h_adj_suit_val : ∀ i j, (i < 12 ∧ deck i j).fst = (deck (i + 1) j).fst ∨ (i < 12 ∧ deck i j).snd = (deck (i + 1) j).snd)
    ∧ (j < 3 ∧ deck i j).fst = (deck i (j + 1)).fst ∨ (j < 3 ∧ deck i j).snd = (deck i (j + 1)).snd) :
  ∀ s ∈ {1, 2, 3, 4}, ∃ r, ∀ i j, (deck i j).fst = s → i = r :=
sorry

end same_suit_same_row_l584_584147


namespace problem1_problem2_l584_584817

-- Condition definitions for Problem 1
def f (x : Real) : Real := 4 * sin x * (sin (π / 4 + x / 2))^2 + cos (2 * x) - 1
def interval : Set Real := set.Icc (-π / 2) (2 * π / 3)
def fy (ω x : Real) : Real := f (ω * x)

-- Proof for Problem 1
theorem problem1 (ω : Real) (hω : ω > 0) (h_inc : ∀ x ∈ interval, monotone (fy ω)) : ω ∈ Set.Icc 0 (3 / 4) :=
sorry

-- Condition definitions for Problem 2
def A : Set Real := { x | π / 6 ≤ x ∧ x ≤ 2 * π / 3 }
def B (m : Real) (x : Real) : Bool := (1/2 * f x)^2 - m * f x + m^2 + m - 1 > 0

-- Proof for Problem 2
theorem problem2 (m : Real) (h : ∀ x ∈ A, B m x) : m < -((√3)/2) ∨ m > 1 :=
sorry

end problem1_problem2_l584_584817


namespace john_needs_9_reams_l584_584536

noncomputable def john_needs_reams 
  (stories_per_week : ℕ)
  (pages_per_story : ℕ)
  (novel_pages_per_year : ℕ)
  (pages_per_sheet : ℕ)
  (sheets_per_ream : ℕ)
  (weeks : ℕ) : ℕ :=
  let sheets_needed := (weeks * stories_per_week * pages_per_story + weeks * novel_pages_per_year / 52) / pages_per_sheet in
  sheets_needed / sheets_per_ream

theorem john_needs_9_reams 
  (h1 : 3 = stories_per_week)
  (h2 : 50 = pages_per_story)
  (h3 : 1200 = novel_pages_per_year)
  (h4 : 2 = pages_per_sheet)
  (h5 : 500 = sheets_per_ream)
  (h6 : 12 = weeks) :
  john_needs_reams 3 50 1200 2 500 12 = 9 :=
by
  sorry

end john_needs_9_reams_l584_584536


namespace original_number_contains_digit_at_least_five_l584_584215

theorem original_number_contains_digit_at_least_five
    (digits : List ℕ)
    (h_digits_no_zero : ∀ d ∈ digits, d ≠ 0)
    (h_sum_is_all_ones : ∃ k, (nat_to_num (digits_to_value digits) + nat_to_num (digits_to_value (permute_1 digits)) + nat_to_num (digits_to_value (permute_2 digits)) + nat_to_num (digits_to_value (permute_3 digits))) = 10^k - 1)
    : ∃ d ∈ digits, d ≥ 5 := 
sorry

end original_number_contains_digit_at_least_five_l584_584215


namespace smallest_four_digit_palindrome_divisible_by_6_l584_584670

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_divisible_by_6 (n : ℕ) : Prop :=
  n % 6 = 0

def is_four_digit (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

theorem smallest_four_digit_palindrome_divisible_by_6 : 
  ∃ (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ is_divisible_by_6 n ∧ 
  ∀ (m : ℕ), is_four_digit m ∧ is_palindrome m ∧ is_divisible_by_6 m → n ≤ m :=
  sorry

end smallest_four_digit_palindrome_divisible_by_6_l584_584670


namespace simplify_sqrt_sum_l584_584099

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l584_584099


namespace modulus_of_Z_l584_584437

-- The statement that we need to prove
theorem modulus_of_Z (Z : ℂ) (h : Z * (2 - 3 * complex.i) = 6 + 4 * complex.i) : complex.abs Z = 2 :=
by
  sorry

end modulus_of_Z_l584_584437


namespace vacation_cost_correct_l584_584537

namespace VacationCost

-- Define constants based on conditions
def starting_charge_per_dog : ℝ := 2
def charge_per_block : ℝ := 1.25
def number_of_dogs : ℕ := 20
def total_blocks : ℕ := 128
def family_members : ℕ := 5

-- Define total earnings from walking dogs
def total_earnings : ℝ :=
  (number_of_dogs * starting_charge_per_dog) + (total_blocks * charge_per_block)

-- Define the total cost of the vacation
noncomputable def total_cost_of_vacation : ℝ :=
  total_earnings / family_members * family_members

-- Proof statement: The total cost of the vacation is $200
theorem vacation_cost_correct : total_cost_of_vacation = 200 := by
  sorry

end VacationCost

end vacation_cost_correct_l584_584537


namespace find_x_l584_584255

theorem find_x :
  ∃ x : ℝ, (2020 + x)^2 = x^2 ∧ x = -1010 :=
sorry

end find_x_l584_584255


namespace AP_eq_AQ_l584_584582

variables {A B C D E P Q : Type} [EuclideanGeometry A B C D E P Q]

-- Given conditions
axiom BD_eq_CE : BD = CE
axiom DE_arc_condition : (P ∈ arc DE) ∧ (Q ∈ arc DE) ∧ ¬(A ∈ arc DE)
axiom AB_eq_PC : AB = PC
axiom AC_eq_BQ : AC = BQ

theorem AP_eq_AQ : AP = AQ :=
by sorry

end AP_eq_AQ_l584_584582


namespace collinear_XAD_l584_584744

-- Definitions for points A, B, C inscribed in circle Γ
variables {Γ : Type*} [metric_space Γ] {A B C D X : Γ}
-- Assume the circle γ is the circumcircle of triangle ABC
variable (abc_inscribed : ∃ circle (Γ), circumscribed (A B C))

-- Assume tangents to circle γ at B and C intersect at D
variables (tangent_B : tangent (B Γ)) (tangent_C : tangent (C Γ))
variable (intersection_tangents : intersection (tangent_B) (tangent_C) = D)

-- Assume construction of squares BAGH and ACEF outward on sides AB and AC respectively
variables (square_bagh : square_construction_outward (B A G H))
variables (square_acef : square_construction_outward (A C E F))

-- Assume lines EF and GH intersection at point X
variables (intersection_ef_gh : intersection_line (line (E F)) (line (G H)) = X)

-- Theorem stating that points X, A, and D are collinear
theorem collinear_XAD (abc_inscribed : ∃ circle (Γ), circumscribed (A B C))
(tangent_B : tangent (B Γ)) (tangent_C : tangent (C Γ))
(intersection_tangents : intersection (tangent_B) (tangent_C) = D)
(square_bagh : square_construction_outward (B A G H))
(square_acef : square_construction_outward (A C E F))
(intersection_ef_gh : intersection_line (line (E F)) (line (G H)) = X) :
collinear X A D := sorry

end collinear_XAD_l584_584744


namespace ken_total_distance_is_12_l584_584498

-- Definitions:
def dist_ken_dawn : ℝ := 4
def dist_ken_mary : ℝ := dist_ken_dawn / 2
def dist_dawn_mary : ℝ := dist_ken_mary

-- Total distance Ken travels
def total_distance : ℝ := dist_ken_dawn + dist_dawn_mary + dist_dawn_mary + dist_ken_dawn

-- Proof statement
theorem ken_total_distance_is_12 : total_distance = 12 := 
by
  -- Proof omitted
  sorry

end ken_total_distance_is_12_l584_584498


namespace TotalMilesCycled_l584_584531

-- Define the variables in the problem
variables (D JackHomeToStoreSpeed JackStoreToPeterSpeed StoreToPeterDistance 
  ReturnSpeedWithDetour DetourDistance T)

-- Mathematical definitions corresponding to the conditions
def JackHomeToStoreSpeed := 15
def JackStoreToPeterSpeed := 20
def StoreToPeterDistance := 50
def ReturnSpeedWithDetour := 18
def DetourDistance := 10
def TimeStoreToPeter := StoreToPeterDistance / JackStoreToPeterSpeed
def TimeHomeToStore := 2 * TimeStoreToPeter
def DistanceHomeToStore := JackHomeToStoreSpeed * TimeHomeToStore

-- The distances
def TotalJackDistance := DistanceHomeToStore + StoreToPeterDistance + (StoreToPeterDistance + DetourDistance)
def TotalPeterDistance := StoreToPeterDistance + (StoreToPeterDistance + DetourDistance)

-- Assert the total miles cycled by both Jack and Peter
theorem TotalMilesCycled : TotalJackDistance + TotalPeterDistance = 295 := by
  sorry

end TotalMilesCycled_l584_584531


namespace object_is_clock_l584_584517

def is_object_opposite_directions_22_times_in_day (obj : Type) : Prop :=
  -- The predicate to indicate that the object's hands show opposite directions 22 times in a day.

axiom clock : Type
axiom object_has_opposite_positions_22_times : is_object_opposite_directions_22_times_in_day clock

theorem object_is_clock (obj : Type) :
  is_object_opposite_directions_22_times_in_day obj → obj = clock :=
by
  intro h
  sorry

end object_is_clock_l584_584517


namespace trajectory_of_moving_circle_eq_l584_584462

theorem trajectory_of_moving_circle_eq :
  let C := {p : ℝ × ℝ | p.1^2 + (p.2 + 3)^2 = 1},
      tangent_line := {p : ℝ × ℝ | p.2 = 2},
      M := λ (p : ℝ × ℝ), true in
  (∀ (p : ℝ × ℝ), M p → p ∈ tangent_line ∧ ∀ q ∈ C, dist p q = dist p (0, 2)) →
  ∀ (p : ℝ × ℝ), M p → p.1^2 = -12 * p.2 :=
sorry

end trajectory_of_moving_circle_eq_l584_584462


namespace part_a_part_b_l584_584722

variables (p : ℝ) (h : p ≥ 0 ∧ p ≤ 1) -- Probability p is between 0 and 1 inclusive

-- Part (a)
theorem part_a (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (5.choose 3) * p^3 * (1 - p)^2 = 10 * p^3 * (1 - p)^2 := by 
sorry

-- Part (b)
theorem part_b (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  10 * p - p^10 = (10 * p - p^10) := by 
sorry

end part_a_part_b_l584_584722


namespace sin_60_eq_sqrt3_div_2_l584_584364

theorem sin_60_eq_sqrt3_div_2 :
  ∃ (Q : ℝ × ℝ), dist Q (1, 0) = 1 ∧ angle (1, 0) Q = real.pi / 3 ∧ Q.2 = real.sqrt 3 / 2 := sorry

end sin_60_eq_sqrt3_div_2_l584_584364


namespace pencil_eraser_cost_l584_584054

/-- Oscar buys 13 pencils and 3 erasers for 100 cents. A pencil costs more than an eraser, 
    and both items cost a whole number of cents. 
    We need to prove that the total cost of one pencil and one eraser is 10 cents. -/
theorem pencil_eraser_cost (p e : ℕ) (h1 : 13 * p + 3 * e = 100) (h2 : p > e) : p + e = 10 :=
sorry

end pencil_eraser_cost_l584_584054


namespace problem1_problem2_l584_584350

-- Definitions for question 1
def expr1 := -9 / 3 + (1 / 2 - 2 / 3) * 12 - |-(4:ℤ)^3|

-- Definitions for question 2
variables (a b : ℝ) 
def expr2 := 2 * (a^2 + 2 * b^2) - 3 * (2 * a^2 - b^2)

-- The theorems to prove the answers
theorem problem1 : expr1 = -69 :=
by
  -- The proof goes here
  sorry

theorem problem2 : expr2 = -4 * a^2 + 7 * b^2 :=
by
  -- The proof goes here
  sorry

end problem1_problem2_l584_584350


namespace calculate_candy_bars_l584_584902

theorem calculate_candy_bars
  (soft_drink_calories : ℕ)
  (percent_added_sugar : ℕ)
  (recommended_intake : ℕ)
  (exceeded_percentage : ℕ)
  (candy_bar_calories : ℕ)
  (soft_drink_calories = 2500)
  (percent_added_sugar = 5)
  (recommended_intake = 150)
  (exceeded_percentage = 100)
  (candy_bar_calories = 25) :
  let added_sugar_from_drink := soft_drink_calories * percent_added_sugar / 100,
      exceeded_amount := recommended_intake * exceeded_percentage / 100,
      total_added_sugar := recommended_intake + exceeded_amount,
      added_sugar_from_candy_bars := total_added_sugar - added_sugar_from_drink in
  added_sugar_from_candy_bars / candy_bar_calories = 7 :=
by 
  -- proof
  sorry

end calculate_candy_bars_l584_584902


namespace inradius_of_right_triangle_l584_584325

theorem inradius_of_right_triangle (a b c : ℕ) (h : a = 7 ∧ b = 24 ∧ c = 25 ∧ a * a + b * b = c * c) : 
  (2 * 84 / (7 + 24 + 25) = 3) :=
by
  have h1 : a * a = 49 := by sorry
  have h2 : b * b = 576 := by sorry
  have h3 : c * c = 625 := by sorry
  have h4 : 49 + 576 = 625 := by sorry
  have h5 : 625 = 625 := by sorry
  have right_triangle : a * a + b * b = c * c := by rw [h1, h2, h3]; sorry
  have s : (7 + 24 + 25) / 2 = 28 := by sorry
  have A : 1 / 2 * 7 * 24 = 84 := by sorry
  have inradius_formula : A = 28 * (2 * 84 / (7 + 24 + 25)) := by sorry
  have inradius_value : 2 * 84 / (7 + 24 + 25) = 3 := by sorry
  exact inradius_value


end inradius_of_right_triangle_l584_584325


namespace probability_of_point_between_C_and_D_l584_584065

open_locale classical

noncomputable theory

def probability_C_to_D (A B C D : ℝ) (AB AD BC : ℝ) (h1 : AB = 4 * AD) (h2 : AB = 8 * BC) : ℝ :=
  let CD := BC - AD in
  CD / AB

theorem probability_of_point_between_C_and_D 
  (A B C D AB AD BC : ℝ) 
  (h1 : AB = 4 * AD) 
  (h2 : AB = 8 * BC) : 
  probability_C_to_D A B C D AB AD BC h1 h2 = 5 / 8 :=
by
  sorry

end probability_of_point_between_C_and_D_l584_584065


namespace median_of_81_consecutive_integers_l584_584177

theorem median_of_81_consecutive_integers (n : ℕ) (S : ℕ) (h1 : n = 81) (h2 : S = 9^5) : 
  let M := S / n in M = 729 :=
by
  sorry

end median_of_81_consecutive_integers_l584_584177


namespace probability_between_C_and_D_l584_584060

-- Conditions
variables (A B C D : Type)
variables {length : A → B → ℝ}
variables {AB AD BC CD : ℝ}
variables {segment : A → A → set ℝ}

-- Condition 1: AB = 4 * AD
def condition1 : Prop := AB = 4 * AD

-- Condition 2: AB = 8 * BC
def condition2 : Prop := AB = 8 * BC

-- Correct answer: probability that a point is between C and D is 5/8
def solution : Prop :=
  let CD := AB - AD - BC in
  CD / AB = (5 : ℚ) / 8

-- Lean statement
theorem probability_between_C_and_D :
  condition1 ∧ condition2 → solution :=
by
  intros h,
  sorry

end probability_between_C_and_D_l584_584060


namespace no_tangent_circle_of_radius_2_l584_584548

def circle (r : ℝ) := {c : ℝ × ℝ // c.1 ^ 2 + c.2 ^ 2 = r ^ 2}

variable (C1 C2 C3 : circle 1)

axiom circles_in_configuration :
  ∀ (x1 y1 x2 y2 x3 y3 : ℝ), 
    C1.val = (x1, y1) → 
    C2.val = (x2, y2) → 
    C3.val = (x3, y3) →
    (x2 - x1)^2 + (y2 - y1)^2 = 4 ∧
    (x3 - x2)^2 + (y3 - y2)^2 = 4 ∧
    (x3 - x1)^2 + (y3 - y1)^2 = 4

theorem no_tangent_circle_of_radius_2 :
  ¬ ∃ (C : circle 2), 
    (dist C1.val C.val = 1 + 2) ∧ 
    (dist C2.val C.val = 1 + 2) ∧ 
    (dist C3.val C.val = 1 + 2) :=
  sorry

end no_tangent_circle_of_radius_2_l584_584548


namespace simplify_sqrt_sum_l584_584105

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l584_584105


namespace find_b_l584_584864

theorem find_b (b : ℝ) : 
  (∀ x, 2 ≤ x ∧ x ≤ 2 * b → 2 ≤ f x ∧ f x ≤ 2 * b) 
  ∧ f 2 = 2 
  ∧ f (2 * b) = 2 * b 
  ∧ (∀ x, 2 ≤ x → f x ≤ f (2 * b)) 
  → b = 2 :=
begin
  sorry
end

end find_b_l584_584864


namespace necessary_but_not_sufficient_condition_for_even_function_l584_584402

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), f(-x) = f(x)

def given_function (a : ℝ) : ℝ → ℝ :=
  λ x, a * Real.exp(x) + Real.log(x ^ 2)

theorem necessary_but_not_sufficient_condition_for_even_function (a : ℝ) :
  is_even_function (given_function a) ↔ a = 0 :=
by
  sorry

end necessary_but_not_sufficient_condition_for_even_function_l584_584402


namespace cot_alpha_div_cot_beta_plus_cot_gamma_l584_584926

theorem cot_alpha_div_cot_beta_plus_cot_gamma (a b c : ℝ) (α β γ : ℝ) 
    (h₁ : a^2 + b^2 = 2021 * c^2) 
    (h₂ : a = c * sin α / sin γ) (h₃ : b = c * sin β / sin γ) 
    (h₄ : γ = π - α - β)
    (h₅ : a^2 = 2021 * c^2): 
    ∂ \(\cot \alpha ∕ \((\cot \sign β ) + (\cot \sign γ )) \= 505.\) :=
begin
  sorry
end

end cot_alpha_div_cot_beta_plus_cot_gamma_l584_584926


namespace sqrt_sum_l584_584109

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l584_584109


namespace distinct_positive_integers_solution_l584_584399

theorem distinct_positive_integers_solution (x y : ℕ) (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h : 1 / x + 1 / y = 2 / 7) : (x = 4 ∧ y = 28) ∨ (x = 28 ∧ y = 4) :=
by
  sorry -- proof to be filled in.

end distinct_positive_integers_solution_l584_584399


namespace trapezoid_common_side_length_l584_584324

variables (a b k p : ℝ)
-- Assuming a > b for the bases of the trapezoid
-- and k, p for the ratio of the areas of the resulting trapezoids

noncomputable def common_side_length (a b k p : ℝ) (h : a > b) : ℝ :=
  sqrt((k * a^2 + p * b^2) / (p + k))

theorem trapezoid_common_side_length (a b k p : ℝ) (h : a > b) :
  common_side_length a b k p h = sqrt((k * a^2 + p * b^2) / (p + k)) :=
sorry

end trapezoid_common_side_length_l584_584324


namespace tens_digit_3_pow_2016_eq_2_l584_584672

def tens_digit (n : Nat) : Nat := (n / 10) % 10

theorem tens_digit_3_pow_2016_eq_2 : tens_digit (3 ^ 2016) = 2 := by
  sorry

end tens_digit_3_pow_2016_eq_2_l584_584672


namespace locus_intersection_AC_BD_l584_584328

theorem locus_intersection_AC_BD (AB : chord Γ) (CD : diameter Γ) (h: AB.is_not_diameter) :
  ∃ O', is_circle O' ∧ ∀ X, X = intersection (line_through A C) (line_through B D) → 
  X ∈ O' →
  ∀ A B, (A B : point Γ),
  A ≠ B → 
  AB.is_chord →
  ∃ C D, 
  (C D : point Γ) →
  CD.is_diameter → 
  X ∈ circle_passing_through A (B) :=
sorry

end locus_intersection_AC_BD_l584_584328


namespace distance_from_sphere_center_to_triangle_plane_l584_584735

theorem distance_from_sphere_center_to_triangle_plane :
  ∀ (O : ℝ × ℝ × ℝ) (r : ℝ) (a b c : ℝ), 
  r = 9 →
  a = 13 →
  b = 13 →
  c = 10 →
  (∀ (d : ℝ), d = distance_from_O_to_plane) →
  d = 8.36 :=
by
  intro O r a b c hr ha hb hc hd
  sorry

end distance_from_sphere_center_to_triangle_plane_l584_584735


namespace simplify_sqrt_sum_l584_584098

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l584_584098


namespace distance_between_foci_of_hyperbola_l584_584611

-- Define the asymptotes as lines
def asymptote1 (x : ℝ) : ℝ := 2 * x + 3
def asymptote2 (x : ℝ) : ℝ := -2 * x + 7

-- Define the condition that the hyperbola passes through the point (4, 5)
def passes_through (x y : ℝ) : Prop := (x, y) = (4, 5)

-- Statement to prove
theorem distance_between_foci_of_hyperbola : 
  (asymptote1 4 = 5) ∧ (asymptote2 4 = 5) ∧ passes_through 4 5 → 
  (∀ a b c : ℝ, a^2 = 9 ∧ b^2 = 9/4 ∧ c^2 = a^2 + b^2 → 2 * c = 3 * Real.sqrt 5) := 
by
  intro h
  sorry

end distance_between_foci_of_hyperbola_l584_584611


namespace sequence_contains_2001_iff_l584_584378

noncomputable def sequence (a₁ a₂ : ℝ) : ℕ → ℝ 
| 0 := a₁
| 1 := a₂
| (n + 2) := (sequence n + 1) / (sequence (n + 1))

theorem sequence_contains_2001_iff :
  ∀ a₁, a₂ = 2000 →
  ∃ n, sequence a₁ a₂ n = 2001 ↔
    a₁ = 2001 ∨
    a₁ = 1 ∨
    a₁ = 2001 / 4001999 ∨
    a₁ = 4001999 :=
by
  intros a₁ a₂ h
  sorry

end sequence_contains_2001_iff_l584_584378


namespace smallest_perimeter_iso_triangle_l584_584235

theorem smallest_perimeter_iso_triangle :
  ∃ (x y : ℕ), (PQ = PR ∧ PQ = x ∧ PR = x ∧ QR = y ∧ QJ = 10 ∧ PQ + PR + QR = 416 ∧ 
  PQ = PR ∧ y = 8 ∧ 2 * (x + y) = 416 ∧ y^2 - 50 > 0 ∧ y < 10) :=
sorry

end smallest_perimeter_iso_triangle_l584_584235


namespace part1_condition1_implies_a_le_1_condition2_implies_a_le_2_condition3_implies_a_le_1_l584_584813

section Problem

-- Universal set is ℝ
def universal_set : Set ℝ := Set.univ

-- Set A
def set_A : Set ℝ := { x | x^2 - x - 6 ≤ 0 }

-- Set A complement in ℝ
def complement_A : Set ℝ := universal_set \ set_A

-- Set B
def set_B : Set ℝ := { x | (x - 4)/(x + 1) < 0 }

-- Set C
def set_C (a : ℝ) : Set ℝ := { x | 2 - a < x ∧ x < 2 + a }

-- Prove (complement_A ∩ set_B = (3, 4))
theorem part1 : (complement_A ∩ set_B) = { x | 3 < x ∧ x < 4 } :=
  sorry

-- Assume a definition for real number a (non-negative)
variable (a : ℝ)

-- Prove range of a given the conditions
-- Condition 1: A ∩ C = C implies a ≤ 1
theorem condition1_implies_a_le_1 (h : set_A ∩ set_C a = set_C a) : a ≤ 1 :=
  sorry

-- Condition 2: B ∪ C = B implies a ≤ 2
theorem condition2_implies_a_le_2 (h : set_B ∪ set_C a = set_B) : a ≤ 2 :=
  sorry

-- Condition 3: C ⊆ (A ∩ B) implies a ≤ 1
theorem condition3_implies_a_le_1 (h : set_C a ⊆ set_A ∩ set_B) : a ≤ 1 :=
  sorry

end Problem

end part1_condition1_implies_a_le_1_condition2_implies_a_le_2_condition3_implies_a_le_1_l584_584813


namespace median_of_81_consecutive_integers_l584_584193

theorem median_of_81_consecutive_integers (sum : ℕ) (h₁ : sum = 9^5) : 
  let mean := sum / 81 in mean = 729 :=
by
  have h₂ : sum = 59049 := by
    rw h₁
    norm_num
  have h₃ : mean = 59049 / 81 := by
    rw h₂
    rfl
  have result : mean = 729 := by
    rw h₃
    norm_num
  exact result

end median_of_81_consecutive_integers_l584_584193


namespace simplify_sqrt_72_plus_sqrt_32_l584_584090

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l584_584090


namespace count_sweet_numbers_in_1_to_60_l584_584080

noncomputable def transform (n : ℕ) : ℕ :=
if n <= 30 then min (2 * n) 40 else n - 15

def isSweetNumber (n : ℕ) : Prop :=
¬(∃ m : ℕ, transform^[m] n = 20)

def countSweetNumbers : ℕ :=
Nat.card (setOf (λ n, 1 <= n ∧ n <= 60 ∧ isSweetNumber n))

theorem count_sweet_numbers_in_1_to_60 : countSweetNumbers = 13 := by
  sorry

end count_sweet_numbers_in_1_to_60_l584_584080


namespace simplify_sqrt_sum_l584_584107

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l584_584107


namespace complement_M_in_U_l584_584481

-- Define the universal set U and set M
def U : Finset ℕ := {4, 5, 6, 8, 9}
def M : Finset ℕ := {5, 6, 8}

-- Define the complement of M in U
def complement (U M : Finset ℕ) : Finset ℕ := U \ M

-- Prove that the complement of M in U is {4, 9}
theorem complement_M_in_U : complement U M = {4, 9} := by
  sorry

end complement_M_in_U_l584_584481


namespace longer_side_length_l584_584285

-- Define the relevant entities: radius, area of the circle, and rectangle conditions.
noncomputable def radius : ℝ := 6
noncomputable def area_circle : ℝ := Real.pi * radius^2
noncomputable def area_rectangle : ℝ := 3 * area_circle
noncomputable def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm.
theorem longer_side_length : ∃ (l : ℝ), (area_rectangle = l * shorter_side) → (l = 9 * Real.pi) :=
by
  sorry

end longer_side_length_l584_584285


namespace monotonic_intervals_k_range_l584_584818

noncomputable def f (x k : ℝ) : ℝ := (x - k) ^ 2 * Real.exp (x / k)

-- (I) Prove monotonic intervals
theorem monotonic_intervals {k : ℝ} :
  (k > 0 → 
    ((∀ x ∈ Iio (-k), deriv (f x k) x > 0) 
     ∧ (∀ x ∈ Ioo (-k) k, deriv (f x k) x < 0) 
     ∧ (∀ x ∈ Ioi k, deriv (f x k) x > 0))) 
  ∧ (k < 0 → 
    ((∀ x ∈ Iio k, deriv (f x k) x < 0) 
     ∧ (∀ x ∈ Ioo k (-k), deriv (f x k) x > 0) 
     ∧ (∀ x ∈ Ioi (-k), deriv (f x k) x < 0))) :=
sorry

-- (II) Prove k range given the condition on f(x)
theorem k_range {k : ℝ} : 
  (∀ x ∈ set.Ioi (0 : ℝ), f x k ≤ (1 / Real.exp 1)) → (-1 / 2 ≤ k ∧ k < 0) :=
sorry

end monotonic_intervals_k_range_l584_584818


namespace complex_sum_modulus_ge_one_sixth_l584_584042

open Complex

theorem complex_sum_modulus_ge_one_sixth (n : ℕ) (z : Fin n → ℂ) 
  (h : ∑ i, ∥z i∥ = 1) : 
  ∃ S : Finset (Fin n), ∥∑ i in S, z i∥ ≥ 1/6 := sorry

end complex_sum_modulus_ge_one_sixth_l584_584042


namespace angle_F_calculation_l584_584005

theorem angle_F_calculation (D E F : ℝ) :
  D = 80 ∧ E = 2 * F + 30 ∧ D + E + F = 180 → F = 70 / 3 :=
by
  intro h
  cases' h with hD h_remaining
  cases' h_remaining with hE h_sum
  sorry

end angle_F_calculation_l584_584005


namespace problem1_problem2_l584_584844

noncomputable def a (λ x : Real) : Real × Real := (2 * λ * Real.sin x, Real.sin x + Real.cos x)
noncomputable def b (λ x : Real) : Real × Real := (Real.sqrt 3 * Real.cos x, λ * (Real.sin x - Real.cos x))

noncomputable def f (λ : Real) (x : Real) : Real :=
  let a_val := a λ x
  let b_val := b λ x
  (a_val.1 * b_val.1) + (a_val.2 * b_val.2)

theorem problem1 (λ : Real) (x : Real) (k : ℤ) (hλ : 0 < λ) (hfmax : f λ x ≤ 2) :
    k * Real.pi + Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 6 :=
  sorry

theorem problem2 (A a b c m : Real) (hA : 0 < A ∧ A < 2 * Real.pi / 3)
    (hcosA : Real.cos A = (2 * b - a) / (2 * c))
    (hineq : 2 * Real.sin(2 * A - Real.pi / 6) - m > 0) :
    m ≤ -1 :=
  sorry

end problem1_problem2_l584_584844


namespace correct_interval_l584_584569

def f (x : ℝ) : ℝ := Real.exp (14 * |x|) - 1 / (1 + x^4)

theorem correct_interval (x : ℝ) (h : f (2 * x) < f (1 - x)) : 
  x ∈ Set.Ioo (-1 : ℝ) (1 / 3 : ℝ) :=
by
  sorry

end correct_interval_l584_584569


namespace extreme_values_a_1_turning_point_a_8_l584_584823

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - (a + 2) * x + a * Real.log x

def turning_point (g : ℝ → ℝ) (P : ℝ × ℝ) (h : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), x ≠ P.1 → (g x - h x) / (x - P.1) > 0

theorem extreme_values_a_1 :
  (∀ (x : ℝ), f x 1 ≤ f (1/2) 1 → f x 1 = f (1/2) 1) ∧ (∀ (x : ℝ), f x 1 ≥ f 1 1 → f x 1 = f 1 1) :=
sorry

theorem turning_point_a_8 :
  ∀ (x₀ : ℝ), x₀ = 2 → turning_point (f · 8) (x₀, f x₀ 8) (λ x => (2 * x₀ + 8 / x₀ - 10) * (x - x₀) + x₀^2 - 10 * x₀ + 8 * Real.log x₀) :=
sorry

end extreme_values_a_1_turning_point_a_8_l584_584823


namespace limit_example_l584_584589

theorem limit_example (f : ℝ → ℝ) (c A : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - c| ∧ |x - c| < δ → |f x - A| < ε) →
  (f = λ x, (2 * x^2 + 5 * x - 3) / (x + 3)) →
  (c = -3) →
  (A = -7) →
  ∀ ε > 0, ∃ δ > 0, (δ = ε / 2) :=
by
  sorry

end limit_example_l584_584589


namespace unique_position_all_sequences_one_l584_584274

-- Define the main theorem
theorem unique_position_all_sequences_one (n : ℕ) (sequences : Fin (2^(n-1)) → Fin n → Bool) :
  (∀ a b c : Fin (2^(n-1)), ∃ p : Fin n, sequences a p = true ∧ sequences b p = true ∧ sequences c p = true) →
  ∃! p : Fin n, ∀ i : Fin (2^(n-1)), sequences i p = true :=
by
  sorry

end unique_position_all_sequences_one_l584_584274


namespace wishing_well_probability_l584_584327

theorem wishing_well_probability :
  ∃ m n : ℕ, Nat.Coprime m n ∧ 
    (∀ y ∈ Finset.range 11, 
      ∃ a b ∈ Finset.range 10, 11 * (a - y) = a * (b - y)) ∧ 
    m + n = 111 :=
by
  sorry

end wishing_well_probability_l584_584327


namespace ferry_captives_successfully_l584_584009

-- Definition of conditions
def valid_trip_conditions (trips: ℕ) (captives: ℕ) : Prop :=
  captives = 43 ∧
  (∀ k < trips, k % 2 = 0 ∨ k % 2 = 1) ∧     -- Trips done in pairs or singles
  (∀ k < captives, k > 40)                    -- At least 40 other captives known as werewolves

-- Theorem statement to be proved
theorem ferry_captives_successfully (trips : ℕ) (captives : ℕ) (result : Prop) : 
  valid_trip_conditions trips captives → result = true := by sorry

end ferry_captives_successfully_l584_584009


namespace minimum_value_l584_584485

-- Define the conditions and problem statement
variables {x y : ℝ}

-- Declare the conditions as definitions
def vec_a := (x - 1, y)
def vec_b := (1, 2)
def perp : Prop := (x - 1) * 1 + y * 2 = 0
def positive_x : Prop := x > 0
def positive_y : Prop := y > 0

-- The theorem to be proven
theorem minimum_value (h1 : perp) (h2 : positive_x) (h3 : positive_y) : 
  ∃ x y : ℝ, (x + 2 * y = 1 ∧ x > 0 ∧ y > 0) → (1 / x + 1 / y) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end minimum_value_l584_584485


namespace grid_sum_equality_l584_584273

-- Define the type of the grid
def grid : Type := Fin 2009 → ℕ

-- Define the conditions
def condition (g : grid) : Prop :=
  ∀ i : Fin (2009 - 89), (Finset.range 90).sum (λ j, g ⟨i + j, by linarith⟩) = 65

-- Define the sum of grid
def total_sum (g : grid) : ℕ :=
  (Finset.range 2009).sum g

-- The proof problem statement
theorem grid_sum_equality (g : grid) (h : condition g) : total_sum g = 1450 ∨ total_sum g = 1451 := 
sorry


end grid_sum_equality_l584_584273


namespace median_of_81_consecutive_integers_l584_584203

theorem median_of_81_consecutive_integers (S : ℤ) 
  (h1 : ∃ l : ℤ, ∀ k, (0 ≤ k ∧ k < 81) → (l + k) ∈ S) 
  (h2 : S = 9^5) : 
  (81 * 729 = 9^5) :=
by
  have h_sum : (S : ℤ) = ∑ i in ((finset.range 81).map (λ n, l + n)), id i, from sorry
  have h_eq : (81 * 729 = 9^5) := calc
    81 * 729 = 9^2 * 9^3 : by norm_num
    ... = 9^(2+3) : by ring
    ... = 9^5 : by norm_num
  exact h_eq

end median_of_81_consecutive_integers_l584_584203


namespace graph_symmetric_about_line_special_l584_584820

noncomputable def f (x : ℝ) : ℝ := 2 * cos x * (sin x + cos x)

theorem graph_symmetric_about_line_special :
  ∀ x, f (x) = f (π/4 - x) :=
by
  sorry

end graph_symmetric_about_line_special_l584_584820


namespace remainder_rounded_eq_l584_584256

noncomputable def x : ℝ := 458.64 / 14

theorem remainder_rounded_eq :
  (x % 17).round_nd 2 = 15.76 :=
by
  -- Prove that x := 458.64 / 14
  have hx : x = 32.76 := by sorry
  -- Divide x by 17 and find the remainder
  have hrem : x % 17 = 15.76 := by sorry
  -- Round the remainder to two decimal places
  have hround : (15.76).round_nd 2 = 15.76 := by
    sorry
  exact hround

end remainder_rounded_eq_l584_584256


namespace relay_team_order_count_l584_584014

/-- Jordan and his three best friends are on a relay team. His relay team will run a race, where the first runner runs a lap, then the second, then the third, then the fourth. Given that Jordan is fixed to run the fourth lap, prove that the number of different orders the remaining three team members can run is 6. -/
theorem relay_team_order_count (friends : Fin 3 → String) :
  (∃ jordan : String, (∃ i : Fin 3, jordan ≠ friends i)) →
  (∃ permutations : List (Fin 3), permutations.length = 3 ∧ permutations.nodup) →
  (∃ orders : Finset (Fin 3 → Fin 3), orders.card = 6) :=
by
  sorry

end relay_team_order_count_l584_584014


namespace smallest_perimeter_triangle_l584_584239

theorem smallest_perimeter_triangle (PQ PR QR : ℕ) (J : Point) :
  PQ = PR →
  QJ = 10 →
  QR = 2 * 10 →
  PQ + PR + QR = 40 :=
by
  sorry

structure Point : Type :=
mk :: (QJ : ℕ)

noncomputable def smallest_perimeter_triangle : Prop :=
  ∃ (PQ PR QR : ℕ) (J : Point), PQ = PR ∧ J.QJ = 10 ∧ QR = 2 * 10 ∧ PQ + PR + QR = 40

end smallest_perimeter_triangle_l584_584239


namespace number_of_triples_l584_584937

open Nat

def triples_count (n : ℕ) : ℕ :=
  n * ∏ p in n.factors.toFinset, (1 + 1/p)

theorem number_of_triples (n : ℕ) (h : 0 < n) :
  ∃ f : ℕ → ℕ,
    (∀ a b c : ℕ, (a * b = n ∧ 1 ≤ c ∧ c ≤ b ∧ gcd (gcd a b) c = 1) ↔ f (a, b, c) = 1) ∧
    f n = n * ∏ p in n.factors.toFinset, (1 + 1/p) := 
by
  sorry

end number_of_triples_l584_584937


namespace smallest_angle_convex_15_polygon_l584_584985

theorem smallest_angle_convex_15_polygon :
  ∃ (a : ℕ) (d : ℕ), (∀ n : ℕ, n ∈ Finset.range 15 → (a + n * d < 180)) ∧
  15 * (a + 7 * d) = 2340 ∧ 15 * d <= 24 -> a = 135 :=
by
  -- Proof omitted
  sorry

end smallest_angle_convex_15_polygon_l584_584985


namespace tangent_perpendicular_max_point_inequality_l584_584470

-- Question I
def f (x : ℝ) (a : ℝ) := log x - 2 * a * x

theorem tangent_perpendicular (a : ℝ) :
  (∃ x > 0, deriv (λ x, f x a) x = -1 / 2) → a > 1 / 4 :=
by sorry

-- Question II
def g (x : ℝ) (a : ℝ) := f x a + 1 / 2 * x^2

theorem max_point_inequality (x₁ a : ℝ) :
  (0 < x₁ ∧ x₁ < 1) →
  (deriv (λ x, g x a) x₁ = 0) →
  (∀ x < x₁, concave_on (λ x, g x a) (set.Icc 0 x₁)) →
  log x₁ / x₁ + 1 / x₁^2 > a :=
by sorry

end tangent_perpendicular_max_point_inequality_l584_584470


namespace area_of_rectangle_is_117_l584_584547

-- Definitions based on the conditions
def Rectangle (A B C D : Point) : Prop :=
  True -- Details of rectangular properties like parallel sides, equal opposite sides, all right angles

def Diagonal (A C : Point) (X Y : Point) :=
  X ∈ Segment A C ∧ Y ∈ Segment A C ∧ X ≠ Y ∧ ∃ D B : Point, D ∈ Segment A X ∧ B ∈ Segment Y C

structure PointsOnDiagonal (A C : Point) (X Y : Point) where
  AX : ℝ := 4
  XY : ℝ := 3
  YC : ℝ := 6
  X_on_diag : X ∈ Segment A C
  Y_on_diag : Y ∈ Segment A C
  X_between_A_Y : X ≠ A ∧ X ≠ Y
  angle_AXD_90 : ∀ D, D ∈ Segment A X → ∠ AXD = 90
  angle_BYC_90 : ∀ B, B ∈ Segment Y C → ∠ BYC = 90

-- Problem statement
theorem area_of_rectangle_is_117 {A B C D X Y : Point}
  (rect : Rectangle A B C D)
  (diag : Diagonal A C X Y)
  (points_conditions : PointsOnDiagonal A C X Y) :
  area A B C D = 117 :=
sorry

end area_of_rectangle_is_117_l584_584547


namespace train_speed_is_72_km_per_hr_l584_584321

def length_of_train : ℝ := 110
def length_of_bridge : ℝ := 136
def time_to_cross : ℝ := 12.299016078713702
def speed_conversion_factor : ℝ := 3.6

-- Define the total distance covered by the train
def total_distance : ℝ := length_of_train + length_of_bridge

-- Define the speed of the train in m/s
def speed_m_per_s : ℝ := total_distance / time_to_cross

-- Define the speed of the train in km/hr
def speed_km_per_hr : ℝ := speed_m_per_s * speed_conversion_factor

theorem train_speed_is_72_km_per_hr : speed_km_per_hr = 72 := by sorry

end train_speed_is_72_km_per_hr_l584_584321


namespace coefficient_x2_l584_584661

noncomputable def polynomial1 := 3 * X^3 - 4 * X^2 + 5 * X - 2
noncomputable def polynomial2 := 2 * X^2 + 3 * X + 4

theorem coefficient_x2 :
  (coeff (polynomial1 * polynomial2) 2) = -5 :=
by
  sorry

end coefficient_x2_l584_584661


namespace major_axis_length_of_given_ellipse_l584_584331

def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_tangent_to_y_axis (ellipse : (ℝ × ℝ) × (ℝ × ℝ) → Prop) :=
  ∃ y : ℝ, ellipse ((0, y), (0, y))

def is_foci (p1 p2 focus1 focus2 : (ℝ × ℝ)) : Prop :=
  p1 = focus1 ∧ p2 = focus2

noncomputable def length_major_axis_of_ellipse (p1 p2 : (ℝ × ℝ)) :=
  distance p1 p2

theorem major_axis_length_of_given_ellipse :
  ∀ (focus1 focus2 : (ℝ × ℝ)),
  is_foci focus1 focus2 (-15, 10) (15, 30) →
  is_tangent_to_y_axis (fun q => length_major_axis_of_ellipse q.1 q.2) →
  length_major_axis_of_ellipse (15, 10) (15, 30) = 20 :=
by
  intros focus1 focus2 foci_cond tangent_cond
  sorry

end major_axis_length_of_given_ellipse_l584_584331


namespace find_y_l584_584491

theorem find_y (y : ℕ) (h1 : 27 = 3^3) (h2 : 3^9 = 27^y) : y = 3 := 
by 
  sorry

end find_y_l584_584491


namespace find_polynomials_l584_584412

theorem find_polynomials (P : ℤ[X]) :
  (∀ n ≥ 2001, P.eval n ∣ n^(n-1) - 1) ↔
  (P = 1 ∨ P = -1 ∨ P = X - 1 ∨ P = -(X - 1) ∨ P = (X - 1)^2 ∨ P = -(X - 1)^2) :=
by 
  sorry

end find_polynomials_l584_584412


namespace log2_plus_log5_eq_one_eight_pow_two_thirds_l584_584346

-- Define logarithm properties
axiom log_prod (a b : ℝ) : (log (10 : ℝ)) = 1 → log (a * b) = log a + log b
-- Define exponential properties
axiom exp_power (a b c : ℝ) : a > 0 -> (a ^ b) ^ c = a ^ (b * c)

theorem log2_plus_log5_eq_one :
  log (2 : ℝ) + log (5 : ℝ) = 1 :=
by
  have h1 : log 10 = 1, from sorry,
  have h2 : log 10 = log (2 * 5), from sorry,
  rw ←log_prod 2 5,
  rw h1,
  sorry

theorem eight_pow_two_thirds :
  (8 : ℝ) ^ (2 / 3) = 4 :=
by
  have h1 : 8 = (2 : ℝ) ^ 3, from sorry,
  calc
  (2 ^ 3) ^ (2 / 3) = 2 ^ (3 * (2 / 3)) : by rw exp_power 2 3 (2 / 3) (by norm_num)
  ... = 2 ^ 2 : by norm_num
  ... = 4 : by norm_num

end log2_plus_log5_eq_one_eight_pow_two_thirds_l584_584346


namespace average_after_12th_inning_revised_average_not_out_l584_584280

theorem average_after_12th_inning (A : ℝ) (H_innings : 11 * A + 92 = 12 * (A + 2)) : (A + 2) = 70 :=
by
  -- Calculation steps are skipped
  sorry

theorem revised_average_not_out (A : ℝ) (H_innings : 11 * A + 92 = 12 * (A + 2)) (H_not_out : 11 * A + 92 = 840) :
  (11 * A + 92) / 9 = 93.33 :=
by
  -- Calculation steps are skipped
  sorry

end average_after_12th_inning_revised_average_not_out_l584_584280


namespace angle_bisector_segment_length_l584_584634

theorem angle_bisector_segment_length 
  (triangle_sides_ratio : ∃ (PQ QR PR : ℝ), PQ / QR = 3 / 4 ∧ QR / PR = 4 / 5 ∧ PR = 15)
  (angle_bisector : ∃ (PS SR QS : ℝ), PS + SR = PR ∧ SR / PS = 4 / 3) : 
  SR = 60 / 7 :=
begin
  obtain ⟨PQ, QR, PR, ratio_PQ_QR, ratio_QR_PR, PR_length⟩ := triangle_sides_ratio,
  obtain ⟨PS, SR, QS, seg_sum, seg_ratio⟩ := angle_bisector,
  have PS_eq_SR_mul : PS = (3 / 4) * SR, from by rw seg_ratio; ring_nf,
  rw [PS_eq_SR_mul, ←add_mul, seg_sum, PR_length] at seg_sum,
  linarith,
  sorry
end

end angle_bisector_segment_length_l584_584634


namespace popton_school_bus_toes_l584_584584

theorem popton_school_bus_toes :
  let hoopit_toes : ℕ := 3 * 4,
      neglart_toes : ℕ := 2 * 5,
      num_hoopits : ℕ := 7,
      num_neglarts : ℕ := 8 in
  hoopit_toes * num_hoopits + neglart_toes * num_neglarts = 164 := by
  let hoopit_toes := 3 * 4
  let neglart_toes := 2 * 5
  let num_hoopits := 7
  let num_neglarts := 8
  calc
    hoopit_toes * num_hoopits + neglart_toes * num_neglarts
      = 12 * 7 + 10 * 8 : by rfl
  ... = 84 + 80 : by rfl
  ... = 164 : by rfl

end popton_school_bus_toes_l584_584584


namespace simplify_sqrt_sum_l584_584138

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l584_584138


namespace online_store_commission_l584_584710

theorem online_store_commission (cost : ℝ) (desired_profit_pct : ℝ) (online_price : ℝ) (commission_pct : ℝ) :
  cost = 19 →
  desired_profit_pct = 0.20 →
  online_price = 28.5 →
  commission_pct = 25 :=
by
  sorry

end online_store_commission_l584_584710


namespace blocks_per_tree_l584_584968

def trees_per_day : ℕ := 2
def blocks_after_5_days : ℕ := 30
def days : ℕ := 5

theorem blocks_per_tree : (blocks_after_5_days / (trees_per_day * days)) = 3 :=
by
  sorry

end blocks_per_tree_l584_584968


namespace problem_p_value_l584_584862

theorem problem_p_value (a p : ℝ)
  (h : (a^2 * (2a - 1) - p * a * (2a - 1) + 6 * (2a - 1)).coeff 2 = 0) :
  p = -1/2 :=
sorry

end problem_p_value_l584_584862


namespace total_weekly_salary_l584_584657

theorem total_weekly_salary (n_salary : ℝ) (h₁ : n_salary = 265) (h₂ : ∀ m_salary : ℝ, m_salary = 1.2 * n_salary) : 
  let m_salary := 1.2 * n_salary in
  m_salary + n_salary = 583 := by
  sorry

end total_weekly_salary_l584_584657


namespace sam_pennies_left_l584_584972

theorem sam_pennies_left (p_initial p_spent p_left : ℕ) (h_initial : p_initial = 98) (h_spent : p_spent = 93) :
  p_initial - p_spent = 5 := 
by
  rw [h_initial, h_spent]
  exact Nat.sub_self 93
  sorry

end sam_pennies_left_l584_584972


namespace system_of_equations_l584_584524

variable {x y : ℕ}

theorem system_of_equations (h1 : 3 * (y - 2) = x) (h2 : 2 * y + 9 = x) :
  (3 * (y - 2) = x ∧ 2 * y + 9 = x) :=
by
  exact ⟨h1, h2⟩
  -- The proof is just restating the given hypotheses in conjunction.
  sorry

end system_of_equations_l584_584524


namespace west_movement_is_negative_seven_l584_584578

-- Define a function to represent the movement notation
def movement_notation (direction: String) (distance: Int) : Int :=
  if direction = "east" then distance else -distance

-- Define the movement in the east direction
def east_movement := movement_notation "east" 3

-- Define the movement in the west direction
def west_movement := movement_notation "west" 7

-- Theorem statement
theorem west_movement_is_negative_seven : west_movement = -7 := by
  sorry

end west_movement_is_negative_seven_l584_584578


namespace appetizer_cost_per_person_l584_584428

theorem appetizer_cost_per_person
    (cost_per_bag: ℕ)
    (num_bags: ℕ)
    (cost_creme_fraiche: ℕ)
    (cost_caviar: ℕ)
    (num_people: ℕ)
    (h1: cost_per_bag = 1)
    (h2: num_bags = 3)
    (h3: cost_creme_fraiche = 5)
    (h4: cost_caviar = 73)
    (h5: num_people = 3):
    (cost_per_bag * num_bags + cost_creme_fraiche + cost_caviar) / num_people = 27 := 
  by
    sorry

end appetizer_cost_per_person_l584_584428


namespace probability_between_C_and_D_l584_584061

-- Conditions
variables (A B C D : Type)
variables {length : A → B → ℝ}
variables {AB AD BC CD : ℝ}
variables {segment : A → A → set ℝ}

-- Condition 1: AB = 4 * AD
def condition1 : Prop := AB = 4 * AD

-- Condition 2: AB = 8 * BC
def condition2 : Prop := AB = 8 * BC

-- Correct answer: probability that a point is between C and D is 5/8
def solution : Prop :=
  let CD := AB - AD - BC in
  CD / AB = (5 : ℚ) / 8

-- Lean statement
theorem probability_between_C_and_D :
  condition1 ∧ condition2 → solution :=
by
  intros h,
  sorry

end probability_between_C_and_D_l584_584061


namespace simplify_sqrt_72_plus_sqrt_32_l584_584087

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l584_584087


namespace smallest_palindrome_divisible_by_6_l584_584668

def is_palindrome (x : Nat) : Prop :=
  let d1 := x / 1000
  let d2 := (x / 100) % 10
  let d3 := (x / 10) % 10
  let d4 := x % 10
  d1 = d4 ∧ d2 = d3

def is_divisible_by (x n : Nat) : Prop :=
  x % n = 0

theorem smallest_palindrome_divisible_by_6 : ∃ n : Nat, is_palindrome n ∧ is_divisible_by n 6 ∧ 1000 ≤ n ∧ n < 10000 ∧ ∀ m : Nat, (is_palindrome m ∧ is_divisible_by m 6 ∧ 1000 ≤ m ∧ m < 10000) → n ≤ m := 
  by
    exists 2112
    sorry

end smallest_palindrome_divisible_by_6_l584_584668


namespace toby_remaining_money_l584_584218

theorem toby_remaining_money : 
  let initial_amount : ℕ := 343
  let fraction_given : ℚ := 1/7
  let total_given := 2 * (initial_amount * fraction_given).to_nat
  initial_amount - total_given = 245 :=
by
  let initial_amount := 343
  let fraction_given := (1/7 : ℚ)
  let given_each := (fraction_given * initial_amount).to_nat
  let total_given := 2 * given_each
  have h : initial_amount - total_given = 245 := by
    calc
      initial_amount - total_given
          = 343 - 2 * 49 : by rw [given_each, (343 : ℕ).mul_div_cancel' (7 : ℕ).nat_abs_pos]
      ... = 343 - 98 : by norm_num
      ... = 245 : by norm_num
  exact h

end toby_remaining_money_l584_584218


namespace determine_a_l584_584027

theorem determine_a (a : ℝ) (h : ∃ r : ℝ, (a / (1 + 2 * complex.i) + (1 + 2 * complex.i) / 5) = r) : a = 1 :=
sorry

end determine_a_l584_584027


namespace solution_set_l584_584156

noncomputable def f : ℝ → ℝ := sorry
axiom h1 : ∀ x : ℝ, differentiable ℝ f
axiom h2 : f (-1) = 2
axiom h3 : ∀ x : ℝ, deriv f x > 2

theorem solution_set : {x : ℝ | f x > 2 * x + 4} = Ioi (-1) := sorry

end solution_set_l584_584156


namespace smallest_angle_of_convex_15_gon_arithmetic_sequence_l584_584994

theorem smallest_angle_of_convex_15_gon_arithmetic_sequence :
  ∃ (a d : ℕ), (∀ k : ℕ, k < 15 → (let angle := a + k * d in angle < 180)) ∧
  (∀ i j : ℕ, i < j → i < 15 → j < 15 → (a + i * d) < (a + j * d)) ∧
  (let sequence_sum := 15 * a + d * 7 * 14 in sequence_sum = 2340) ∧
  (d = 3) ∧
  (a = 135) :=
by
  sorry

end smallest_angle_of_convex_15_gon_arithmetic_sequence_l584_584994


namespace terminal_side_is_fourth_quadrant_l584_584853

theorem terminal_side_is_fourth_quadrant 
  (θ : ℝ) 
  (h1 : Real.sin (π + θ) = 4 / 5) 
  (h2 : Real.sin (π / 2 + θ) = 3 / 5) : 
  θ ∈ Set.range (fun k : ℕ => 4 * π * k + 4 * π - π / 2 .. 4 * π * k + 4 * π + π / 2) :=
by
  sorry

end terminal_side_is_fourth_quadrant_l584_584853


namespace square_board_tiling_l584_584961

theorem square_board_tiling {n : ℕ} (h : n % 3 ≠ 0) (board : ℕ → ℕ → Prop) (removed_cell : ℕ × ℕ) :
  (∃ tiles : list (set (ℕ × ℕ → Prop)),  ∀ tile ∈ tiles, 
  (tile = {p | board p.fst p.snd} ∧ ∃ removed_tile_cell : (ℕ × ℕ), (tile removed_tile_cell = false)) ∧
  (∀ (x y : ℕ), board x y → (x, y) ≠ removed_cell) ∧ 
  (board 2*n 2*n) = true) :=
begin
  sorry
end

end square_board_tiling_l584_584961


namespace value_of_a_l584_584943

noncomputable def f (a : ℝ) (x : ℝ) := a * x + 3

theorem value_of_a (a : ℝ) (h : (fun x => (f a x))' 1 = 3) : a = 3 :=
by
  sorry

end value_of_a_l584_584943


namespace probability_between_C_and_D_is_five_eighths_l584_584056

noncomputable def AB : ℝ := 1
def AD : ℝ := AB / 4
def BC : ℝ := AB / 8
def pos_C : ℝ := AB - BC
def pos_D : ℝ := AD
def CD : ℝ := pos_C - pos_D

theorem probability_between_C_and_D_is_five_eighths : CD / AB = 5 / 8 :=
by
  simp [AB, AD, BC, pos_C, pos_D, CD]
  sorry

end probability_between_C_and_D_is_five_eighths_l584_584056


namespace amrita_bakes_cake_next_thursday_l584_584330

theorem amrita_bakes_cake_next_thursday (n m : ℕ) (h1 : n = 5) (h2 : m = 7) : Nat.lcm n m = 35 :=
by
  -- Proof goes here
  sorry

end amrita_bakes_cake_next_thursday_l584_584330


namespace compute_b_minus_c_l584_584432

noncomputable def a (n : ℕ) (h : n > 1) : ℝ := (Real.log n) / (Real.log 3003)

def b : ℝ := (a 3 (by decide)) + (a 4 (by decide)) + (a 5 (by decide)) + (a 6 (by decide))
def c : ℝ := (a 12 (by decide)) + (a 13 (by decide)) + (a 14 (by decide)) + (a 15 (by decide)) + (a 16 (by decide))

theorem compute_b_minus_c : b - c = -Real.log 3003 1456 := sorry

end compute_b_minus_c_l584_584432


namespace part_I_part_II_l584_584477

noncomputable def f (a : ℝ) (x : ℝ) := a * x - Real.log x
noncomputable def F (a : ℝ) (x : ℝ) := Real.exp x + a * x
noncomputable def g (a : ℝ) (x : ℝ) := x * Real.exp (a * x - 1) - 2 * a * x + f a x

def monotonicity_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < b ∧ a < y ∧ y < b ∧ x < y → f x < f y

theorem part_I (a : ℝ) : 
  monotonicity_in_interval (f a) 0 (Real.log 3) = monotonicity_in_interval (F a) 0 (Real.log 3) ↔ a ≤ -3 :=
sorry

theorem part_II (a : ℝ) (ha : a ∈ Set.Iic (-1 / Real.exp 2)) : 
  (∃ x, x > 0 ∧ g a x = M) → M ≥ 0 :=
sorry

end part_I_part_II_l584_584477


namespace intersection_of_M_and_N_l584_584479

namespace IntersectionProblem

def M : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }

def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 ∧ x ∈ Set.Univ.filter (λ x : ℝ, x = Int.cast x) }

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end IntersectionProblem

end intersection_of_M_and_N_l584_584479


namespace correct_number_of_eggs_to_buy_l584_584845

/-- Define the total number of eggs needed and the number of eggs given by Andrew -/
def total_eggs_needed : ℕ := 222
def eggs_given_by_andrew : ℕ := 155

/-- Define a statement asserting the correct number of eggs to buy -/
def remaining_eggs_to_buy : ℕ := total_eggs_needed - eggs_given_by_andrew

/-- The statement of the proof problem -/
theorem correct_number_of_eggs_to_buy : remaining_eggs_to_buy = 67 :=
by sorry

end correct_number_of_eggs_to_buy_l584_584845


namespace profit_difference_l584_584287

theorem profit_difference
  (p1 p2 : ℝ)
  (h1 : p1 > p2)
  (h2 : p1 + p2 = 3635000)
  (h3 : p2 = 442500) :
  p1 - p2 = 2750000 :=
by 
  sorry

end profit_difference_l584_584287


namespace min_value_proof_l584_584806

noncomputable def min_value_abs :: real := 
  (|10 * real.pi - \alpha_1 - \alpha_2| >= 1 / 4 * real.pi)

theorem min_value_proof (a_1 a_2 : real) (h : (1 / (2 + real.sin a_1) + 1 / (2 + real.sin (2 * a_2))) = 2) 
  : min_value_abs :=
sorry

end min_value_proof_l584_584806


namespace not_eight_divides_n_l584_584557

theorem not_eight_divides_n (n : ℕ) (h : (1/3 : ℚ) + (1/4) + (1/8) + 1/n ∈ ℤ) : ¬ 8 ∣ n :=
sorry

end not_eight_divides_n_l584_584557


namespace square_side_length_l584_584154

theorem square_side_length (d : ℝ) (s : ℝ) (h : d = Real.sqrt 2) (h2 : d = Real.sqrt 2 * s) : s = 1 :=
by
  sorry

end square_side_length_l584_584154


namespace appetizer_cost_per_person_l584_584431

def chip_cost : ℝ := 3 * 1.00
def creme_fraiche_cost : ℝ := 5.00
def caviar_cost : ℝ := 73.00
def total_cost : ℝ := chip_cost + creme_fraiche_cost + caviar_cost
def number_people : ℝ := 3
def cost_per_person : ℝ := total_cost / number_people

theorem appetizer_cost_per_person : cost_per_person = 27.00 := 
by
  -- proof would go here
  sorry

end appetizer_cost_per_person_l584_584431


namespace mark_candy_bars_consumption_l584_584900

theorem mark_candy_bars_consumption 
  (recommended_intake : ℕ := 150)
  (soft_drink_calories : ℕ := 2500)
  (soft_drink_added_sugar_percent : ℕ := 5)
  (candy_bar_added_sugar_calories : ℕ := 25)
  (exceeded_percentage : ℕ := 100)
  (actual_intake := recommended_intake + (recommended_intake * exceeded_percentage / 100))
  (soft_drink_added_sugar := soft_drink_calories * soft_drink_added_sugar_percent / 100)
  (candy_bars_added_sugar := actual_intake - soft_drink_added_sugar)
  (number_of_bars := candy_bars_added_sugar / candy_bar_added_sugar_calories) : 
  number_of_bars = 7 := 
by
  sorry

end mark_candy_bars_consumption_l584_584900


namespace side_length_square_field_l584_584699

theorem side_length_square_field
  (time: ℝ) (speed_kmph: ℝ)
  (h1 : time = 72)
  (h2 : speed_kmph = 9) :
  ∃ s : ℝ, s = 45 :=
begin
  sorry
end

end side_length_square_field_l584_584699


namespace books_loaned_out_l584_584262

theorem books_loaned_out (x : ℕ) (h1 : 75 - 64 = 11) (h2 : 0.2 * x = 11): 
  x = 55 :=
sorry

end books_loaned_out_l584_584262


namespace polynomial_real_root_l584_584766

-- Definition of the polynomial at value of x
def polynomial (x b : ℝ) : ℝ := x^6 + b * x^4 - x^3 + b * x^2 + 1

-- Statement: There exists an x where the polynomial evaluates to 0, if and only if b in (-∞, -3/2]
theorem polynomial_real_root (b : ℝ) : (∃ x : ℝ, polynomial x b = 0) ↔ b ∈ (-∞ : ℝ, -3/2] :=
by
  sorry

end polynomial_real_root_l584_584766


namespace sin_60_eq_sqrt3_div_2_l584_584354

theorem sin_60_eq_sqrt3_div_2 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  -- proof skipped
  sorry

end sin_60_eq_sqrt3_div_2_l584_584354


namespace chessboard_line_max_squares_l584_584736

theorem chessboard_line_max_squares (n : ℕ) (h : n = 8) : 
  ∃ m, (max_unit_squares_with_interior_points n = m) ∧ m = 15 :=
by
  unfold max_unit_squares_with_interior_points
  simp [h]
  sorry

--- Define the auxiliary function max_unit_squares_with_interior_points
def max_unit_squares_with_interior_points (n : ℕ) : ℕ := sorry

end chessboard_line_max_squares_l584_584736


namespace range_of_k_l584_584162

theorem range_of_k 
  (k : ℝ) 
  (line_intersects_hyperbola : ∃ x y : ℝ, y = k * x + 2 ∧ x^2 - y^2 = 6) : 
  -Real.sqrt (15) / 3 < k ∧ k < Real.sqrt (15) / 3 := 
by
  sorry

end range_of_k_l584_584162


namespace simplify_sqrt_72_plus_sqrt_32_l584_584084

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l584_584084


namespace median_of_81_consecutive_integers_l584_584202

theorem median_of_81_consecutive_integers (S : ℤ) 
  (h1 : ∃ l : ℤ, ∀ k, (0 ≤ k ∧ k < 81) → (l + k) ∈ S) 
  (h2 : S = 9^5) : 
  (81 * 729 = 9^5) :=
by
  have h_sum : (S : ℤ) = ∑ i in ((finset.range 81).map (λ n, l + n)), id i, from sorry
  have h_eq : (81 * 729 = 9^5) := calc
    81 * 729 = 9^2 * 9^3 : by norm_num
    ... = 9^(2+3) : by ring
    ... = 9^5 : by norm_num
  exact h_eq

end median_of_81_consecutive_integers_l584_584202


namespace AB_BP_eq_2_BM2_l584_584561

universe u 
variable {α : Type u}

-- Definitions of points and triangle
structure Point :=
  (x y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Definition of the midpoint function
def midpoint (P Q : Point) : Point := 
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

-- Basic distance function for points (assuming ℝ is the type for coordinates)
def dist (P Q : Point) : ℝ :=
  ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2) ^ (1 / 2)

-- Proof statement begins here
theorem AB_BP_eq_2_BM2 (ABC : Triangle) (M P : Point)
  (hM : M = midpoint ABC.A ABC.C)
  (hTangentCircle : ∀ (C : Circle), C.tangent_to_segment ABC.B ABC.C ∧ C.passes_through M)
  (hP : P ∈ segment ABC.A ABC.B)
  : dist ABC.A ABC.B * dist ABC.B P = 2 * (dist ABC.B M)^2 := 
  sorry

end AB_BP_eq_2_BM2_l584_584561


namespace largest_number_l584_584294

theorem largest_number (a b c d : ℤ)
  (h1 : (a + b + c) / 3 + d = 17)
  (h2 : (a + b + d) / 3 + c = 21)
  (h3 : (a + c + d) / 3 + b = 23)
  (h4 : (b + c + d) / 3 + a = 29) :
  d = 21 := 
sorry

end largest_number_l584_584294


namespace part1_part2_part3_part4_l584_584946

open Set

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α] [Archimedean α]

noncomputable def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}
noncomputable def B : Set ℝ := {x | x > 4}

theorem part1 : A ∩ B = {x | 4 < x ∧ x < 5} :=
by sorry

theorem part2 : A ∪ B = {x | 2 ≤ x} :=
by sorry

theorem part3 : A ∩ (univ \ B) = {x | 2 ≤ x ∧ x ≤ 4} :=
by sorry
  
theorem part4 : (univ \ A) ∩ (univ \ B) = {x | x < 2} :=
by sorry

end part1_part2_part3_part4_l584_584946


namespace determine_A_l584_584609

open Real

theorem determine_A (A B C : ℝ)
  (h_decomposition : ∀ x, x ≠ 4 ∧ x ≠ -2 -> (x + 2) / (x^3 - 9 * x^2 + 14 * x + 24) = A / (x - 4) + B / (x - 3) + C / (x + 2)^2)
  (h_factorization : ∀ x, (x^3 - 9 * x^2 + 14 * x + 24) = (x - 4) * (x - 3) * (x + 2)^2) :
  A = 1 / 6 := 
sorry

end determine_A_l584_584609


namespace fib_binet_l584_584025

def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fib (n + 1) + fib n

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def ψ : ℝ := (1 - Real.sqrt 5) / 2

lemma φ_property : φ^2 = φ + 1 := by
  unfold φ
  ring
  rw [Real.sqr_sqrt]
  ring
  apply le_add_of_nonneg_left
  norm_num

lemma ψ_property : ψ^2 = ψ + 1 := by
  unfold ψ
  ring
  rw [Real.sqr_sqrt]
  ring
  apply le_add_of_nonneg_left
  norm_num

theorem fib_binet (n : ℕ) : 
  (fib n : ℝ) = (φ ^ n - ψ ^ n) / Real.sqrt 5 := by
  sorry

end fib_binet_l584_584025


namespace triangle_angle_l584_584880

variable (a b c : ℝ)
variable (C : ℝ)

theorem triangle_angle (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : (a^2 + b^2) * (a^2 + b^2 - c^2) = 3 * a^2 * b^2) :
  C = Real.arccos ((a^4 + b^4 - a^2 * b^2) / (2 * a * b * (a^2 + b^2))) :=
sorry

end triangle_angle_l584_584880


namespace candy_total_l584_584784

theorem candy_total (n m : ℕ) (h1 : n = 2) (h2 : m = 8) : n * m = 16 :=
by
  -- This will contain the proof
  sorry

end candy_total_l584_584784


namespace central_angle_of_sector_l584_584981

theorem central_angle_of_sector (R r n : ℝ) (h_lateral_area : 2 * π * r^2 = π * r * R) 
  (h_arc_length : (n * π * R) / 180 = 2 * π * r) : n = 180 :=
by 
  sorry

end central_angle_of_sector_l584_584981


namespace equal_distances_to_line_l584_584802

open Real

theorem equal_distances_to_line (a : ℝ) :
  let A_x : ℝ := a
  let A_y : ℝ := 1
  let B_x : ℝ := 4
  let B_y : ℝ := 8
  let line_eqn := (λ (x y : ℝ), x + y + 1)
  dist_to_line : ℝ := λ x y, abs (line_eqn x y) / sqrt (1 * 1 + 1 * 1)
  dist_to_line A_x A_y = dist_to_line B_x B_y →
  a = 11 ∨ a = -15 := by
  sorry

end equal_distances_to_line_l584_584802


namespace cylinder_volume_relation_l584_584392

theorem cylinder_volume_relation (r h : ℝ) (π_pos : 0 < π) :
  (∀ B_h B_r A_h A_r : ℝ, B_h = r ∧ B_r = h ∧ A_h = h ∧ A_r = r 
   → 3 * (π * h^2 * r) = π * r^2 * h) → 
  ∃ N : ℝ, (π * (3 * h)^2 * h) = N * π * h^3 ∧ N = 9 :=
by 
  sorry

end cylinder_volume_relation_l584_584392


namespace trajectory_of_center_line_passes_fixed_point_l584_584795

-- Define the conditions
def pointA : ℝ × ℝ := (4, 0)
def chord_length : ℝ := 8
def pointB : ℝ × ℝ := (-3, 0)
def not_perpendicular_to_x_axis (t : ℝ) : Prop := t ≠ 0
def trajectory_eq (x y : ℝ) : Prop := y^2 = 8 * x
def line_eq (t m y x : ℝ) : Prop := x = t * y + m
def x_axis_angle_bisector (y1 x1 y2 x2 : ℝ) : Prop := (y1 / (x1 + 3)) + (y2 / (x2 + 3)) = 0

-- Prove the trajectory of the center of the moving circle is \( y^2 = 8x \)
theorem trajectory_of_center (x y : ℝ) 
  (H1: (x-4)^2 + y^2 = 4^2 + x^2) 
  (H2: trajectory_eq x y) : 
  trajectory_eq x y := sorry

-- Prove the line passes through the fixed point (3, 0)
theorem line_passes_fixed_point (t m y1 x1 y2 x2 : ℝ) 
  (Ht: not_perpendicular_to_x_axis t)
  (Hsys: ∀ y x, line_eq t m y x → trajectory_eq x y)
  (Hangle: x_axis_angle_bisector y1 x1 y2 x2) : 
  (m = 3) ∧ ∃ y, line_eq t 3 y 3 := sorry

end trajectory_of_center_line_passes_fixed_point_l584_584795


namespace Triangle_area_is_18_l584_584038

-- Define the properties and the problem conditions
structure Triangle :=
  (A B C : Point)
  (centroid : Point)
  (medians_divide_ratios : ∀ P Q : Point, centroid ∈ median P Q → P.segment(centroid).len / centroid.segment(Q).len = 2)

def centroid (ABC : Triangle) : Point := 
  sorry  -- Definition of centroid depending on triangle coordinates

axiom AM_eq_3 : ∀ (ABC : Triangle), 
(centroid(ABC)).distance_to(ABC.A) = 3

axiom BM_eq_4 : ∀ (ABC : Triangle), 
(centroid(ABC)).distance_to(ABC.B) = 4

axiom CM_eq_5 : ∀ (ABC : Triangle), 
(centroid(ABC)).distance_to(ABC.C) = 5

-- The main statement, that the area of ΔABC is 18 given the conditions
theorem Triangle_area_is_18 : ∀ (ABC : Triangle), 
area(ABC) = 18 :=
by
  sorry

end Triangle_area_is_18_l584_584038


namespace chime_date_is_march_22_2003_l584_584707

-- Definitions
def clock_chime (n : ℕ) : ℕ := n % 12

def half_hour_chimes (half_hours : ℕ) : ℕ := half_hours
def hourly_chimes (hours : List ℕ) : ℕ := hours.map clock_chime |>.sum

-- Problem conditions and result
def initial_chimes_and_half_hours : ℕ := half_hour_chimes 9
def initial_hourly_chimes : ℕ := hourly_chimes [4, 5, 6, 7, 8, 9, 10, 11, 0]
def chimes_on_february_28_2003 : ℕ := initial_chimes_and_half_hours + initial_hourly_chimes

def half_hour_chimes_per_day : ℕ := half_hour_chimes 24
def hourly_chimes_per_day : ℕ := hourly_chimes (List.range 12 ++ List.range 12)
def total_chimes_per_day : ℕ := half_hour_chimes_per_day + hourly_chimes_per_day

def remaining_chimes_needed : ℕ := 2003 - chimes_on_february_28_2003
def full_days_needed : ℕ := remaining_chimes_needed / total_chimes_per_day
def additional_chimes_needed : ℕ := remaining_chimes_needed % total_chimes_per_day

-- Lean theorem statement
theorem chime_date_is_march_22_2003 :
    (full_days_needed = 21) → (additional_chimes_needed < total_chimes_per_day) → 
    true :=
by
  sorry

end chime_date_is_march_22_2003_l584_584707


namespace find_a_is_minus_1_l584_584863

-- The conditions from a)
noncomputable def complex_number_is_pure_imaginary (a : ℝ) : Prop :=
  ∃ (z : ℂ), z = (a - complex.I) / (1 - complex.I) ∧ z.re = 0

-- The statement proving the question is equivalent to the answer from b)
theorem find_a_is_minus_1 (a : ℝ) : complex_number_is_pure_imaginary a → a = -1 := 
by
  sorry

end find_a_is_minus_1_l584_584863


namespace convert_536_oct_to_base7_l584_584385

def octal_to_decimal (n : ℕ) : ℕ :=
  n % 10 + (n / 10 % 10) * 8 + (n / 100 % 10) * 64

def decimal_to_base7 (n : ℕ) : ℕ :=
  n % 7 + (n / 7 % 7) * 10 + (n / 49 % 7) * 100 + (n / 343 % 7) * 1000

theorem convert_536_oct_to_base7 : 
  decimal_to_base7 (octal_to_decimal 536) = 1010 :=
by
  sorry

end convert_536_oct_to_base7_l584_584385


namespace distinct_values_of_a_plus_b_l584_584504

theorem distinct_values_of_a_plus_b (a b : ℝ) (f : ℝ → ℝ) (D : set ℝ) (R : set ℝ) :
    f = (λ x, x^3) ∧ D = set.Icc a b ∧ R = set.Icc a b ∧ D = f '' D ∧ R = f '' D →
    set.card {a + b | ∃ (a b : ℝ), f = (λ x, x^3) ∧ D = set.Icc a b ∧ R = set.Icc a b ∧ D = f '' D ∧ R = f '' D} = 3 := sorry

end distinct_values_of_a_plus_b_l584_584504


namespace trajectory_of_Q_is_circle_l584_584960

-- Definitions of points on hyperbola and foci
variables {P Q F1 F2 : Point}
variables {C : Hyperbola}

-- Conditions of the problem
def point_on_hyperbola (P: Point) (C: Hyperbola) : Prop := P ∈ C
def foci_of_hyperbola (F1 F2: Point) (C: Hyperbola) : Prop := foci C = (F1, F2)
def perpendicular_bisector (F1 F2 P Q: Point) : Prop := 
  is_perpendicular (line_through F1 P) (line_through P Q) ∧ 
  angle_bisector (line_through F1 P) (line_through P F2) = line_through P Q ∧ 
  foot_of_perpendicular (F1, F2, P, Q)

-- Statement to prove: The trajectory of point Q is a circle
theorem trajectory_of_Q_is_circle (P Q F1 F2 : Point) (C : Hyperbola)
  (h1 : point_on_hyperbola P C)
  (h2 : foci_of_hyperbola F1 F2 C)
  (h3 : perpendicular_bisector F1 F2 P Q) :
  trajectory Q = circle :=
sorry

end trajectory_of_Q_is_circle_l584_584960


namespace weekly_caloric_deficit_l584_584910

-- Define the conditions
def daily_calories (day : String) : Nat :=
  if day = "Saturday" then 3500 else 2500

def daily_burn : Nat := 3000

-- Define the total calories consumed in a week
def total_weekly_consumed : Nat :=
  (2500 * 6) + 3500

-- Define the total calories burned in a week
def total_weekly_burned : Nat :=
  daily_burn * 7

-- Define the weekly deficit
def weekly_deficit : Nat :=
  total_weekly_burned - total_weekly_consumed

-- The proof goal
theorem weekly_caloric_deficit : weekly_deficit = 2500 :=
by
  -- Proof steps would go here; however, per instructions, we use sorry
  sorry

end weekly_caloric_deficit_l584_584910


namespace brocard_angle_theorem_l584_584264

-- Define the required geometrical elements for triangle and Brocard point
variables {A B C P : Type}

-- Define the Brocard point conditions
def is_brocard_point (triangle : Triangle A B C) (P : Point) : Prop :=
  (angle A B P = angle B C P) ∧ (angle B C P = angle C A P)

-- Define the Brocard angle condition
def brocard_angle (triangle : Triangle A B C) (P : Point) : Angle :=
  if is_brocard_point triangle P then angle A B P else 0

-- The main theorem statement in Lean
theorem brocard_angle_theorem (triangle : Triangle A B C) (P : Point) :
  is_brocard_point triangle P →
  let varphi := brocard_angle triangle P in
  cot varphi = cot (angle B C A) + cot (angle A B C) + cot (angle C A B) :=
by
  intros,
  sorry -- Complete proof is omitted

end brocard_angle_theorem_l584_584264


namespace simplify_sqrt72_add_sqrt32_l584_584126

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l584_584126


namespace disease_cases_1975_l584_584510

theorem disease_cases_1975 (cases_1950 cases_2000 : ℕ) (cases_1950_eq : cases_1950 = 500000)
  (cases_2000_eq : cases_2000 = 1000) (linear_decrease : ∀ t : ℕ, 1950 ≤ t ∧ t ≤ 2000 →
  ∃ k : ℕ, cases_1950 - (k * (t - 1950)) = cases_2000) : 
  ∃ cases_1975 : ℕ, cases_1975 = 250500 := 
by
  -- Setting up known values
  let decrease_duration := 2000 - 1950
  let total_decrease := cases_1950 - cases_2000
  let annual_decrease := total_decrease / decrease_duration
  let years_from_1950_to_1975 := 1975 - 1950
  let decline_by_1975 := annual_decrease * years_from_1950_to_1975
  let cases_1975 := cases_1950 - decline_by_1975
  -- Returning the desired value
  use cases_1975
  sorry

end disease_cases_1975_l584_584510


namespace solve_equation_l584_584401

theorem solve_equation (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) (h : a^2 = b * (b + 7)) : 
  (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
by sorry

end solve_equation_l584_584401


namespace probability_is_correct_l584_584698

-- Given definitions
def total_marbles : ℕ := 100
def red_marbles : ℕ := 35
def white_marbles : ℕ := 30
def green_marbles : ℕ := 10

-- Probe the probability
noncomputable def probability_red_white_green : ℚ :=
  (red_marbles + white_marbles + green_marbles : ℚ) / total_marbles

-- The theorem we need to prove
theorem probability_is_correct :
  probability_red_white_green = 0.75 := by
  sorry

end probability_is_correct_l584_584698


namespace weekly_caloric_deficit_l584_584909

-- Define the conditions
def daily_calories (day : String) : Nat :=
  if day = "Saturday" then 3500 else 2500

def daily_burn : Nat := 3000

-- Define the total calories consumed in a week
def total_weekly_consumed : Nat :=
  (2500 * 6) + 3500

-- Define the total calories burned in a week
def total_weekly_burned : Nat :=
  daily_burn * 7

-- Define the weekly deficit
def weekly_deficit : Nat :=
  total_weekly_burned - total_weekly_consumed

-- The proof goal
theorem weekly_caloric_deficit : weekly_deficit = 2500 :=
by
  -- Proof steps would go here; however, per instructions, we use sorry
  sorry

end weekly_caloric_deficit_l584_584909


namespace rectangle_perimeter_l584_584079

-- Defining the given conditions
def rectangleArea := 4032
noncomputable def ellipseArea := 4032 * Real.pi
noncomputable def b := Real.sqrt 2016
noncomputable def a := 2 * Real.sqrt 2016

-- Problem statement: the perimeter of the rectangle
theorem rectangle_perimeter (x y : ℝ) (h1 : x * y = rectangleArea)
  (h2 : x + y = 2 * a) : 2 * (x + y) = 8 * Real.sqrt 2016 :=
by
  sorry

end rectangle_perimeter_l584_584079


namespace mathematically_equivalent_proof_problem_l584_584277

noncomputable def find_ab (a b : ℝ) : Prop :=
  ∀ (x : ℝ), (ax^2 - 3x + 6 > 4) ↔ (x < 1 ∨ x > b)

noncomputable def solve_inequality (a b c : ℝ) : Set ℝ :=
  if c > 2 then
    {x : ℝ | 2 < x ∧ x < c}
  else if c < 2 then
    {x : ℝ | c < x ∧ x < 2}
  else
    ∅

theorem mathematically_equivalent_proof_problem (a b c : ℝ) (h : find_ab a b):
  ∀ (x : ℝ), (ax^2 - (ac + b)x + bc < 0) ↔ (x ∈ solve_inequality a b c) :=
by
  sorry

end mathematically_equivalent_proof_problem_l584_584277


namespace max_cos_a_correct_l584_584552

noncomputable def max_cos_a (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) : ℝ :=
  Real.sqrt 3 - 1

theorem max_cos_a_correct (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) :
  max_cos_a a b h = Real.sqrt 3 - 1 :=
sorry

end max_cos_a_correct_l584_584552


namespace number_of_distinct_intersecting_subsets_l584_584941

theorem number_of_distinct_intersecting_subsets (S : Type) (n : ℕ) (hs : S.finite) (h_s_card : hs.to_finset.card = n)
  (A : ℕ → set S) (k : ℕ) 
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j) 
  (h_nonempty : ∀ i, (A i).nonempty)
  (h_pairwise_intersect : ∀ i j, i ≠ j → (A i ∩ A j).nonempty)
  (h_no_other_intersect : ∀ B, (∀ i, B ∩ A i ≠ ∅) → B ∈ set.range A) :
  k = 2^(n-1) :=
sorry

end number_of_distinct_intersecting_subsets_l584_584941


namespace expansion_l584_584409

variable (x : ℝ)

noncomputable def expr : ℝ := (3 / 4) * (8 / (x^2) + 5 * x - 6)

theorem expansion :
  expr x = (6 / (x^2)) + (15 * x / 4) - 4.5 :=
by
  sorry

end expansion_l584_584409


namespace exist_non_negative_product_l584_584072

theorem exist_non_negative_product (a1 a2 a3 a4 a5 a6 a7 a8 : ℝ) :
  0 ≤ a1 * a3 + a2 * a4 ∨
  0 ≤ a1 * a5 + a2 * a6 ∨
  0 ≤ a1 * a7 + a2 * a8 ∨
  0 ≤ a3 * a5 + a4 * a6 ∨
  0 ≤ a3 * a7 + a4 * a8 ∨
  0 ≤ a5 * a7 + a6 * a8 :=
sorry

end exist_non_negative_product_l584_584072


namespace y_is_multiple_of_3_and_6_l584_584545

-- Define y as a sum of given numbers
def y : ℕ := 48 + 72 + 144 + 216 + 432 + 648 + 2592

theorem y_is_multiple_of_3_and_6 :
  (y % 3 = 0) ∧ (y % 6 = 0) :=
by
  -- Proof would go here, but we will end with sorry
  sorry

end y_is_multiple_of_3_and_6_l584_584545


namespace card_length_l584_584746

noncomputable def width_card : ℕ := 2
noncomputable def side_poster_board : ℕ := 12
noncomputable def total_cards : ℕ := 24

theorem card_length :
  ∃ (card_length : ℕ),
    (side_poster_board / width_card) * (side_poster_board / card_length) = total_cards ∧ 
    card_length = 3 := by
  sorry

end card_length_l584_584746


namespace constants_c_d_l584_584925

open Matrix
open FiniteDimensional

noncomputable def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, -1], ![2, -4]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℚ := 1

theorem constants_c_d :
  let c : ℚ := 1 / 10
  let d : ℚ := 1 / 5
  (N⁻¹ = c • N + d • I) :=
by
  sorry

end constants_c_d_l584_584925


namespace remainder_of_899830_divided_by_16_is_6_l584_584266

theorem remainder_of_899830_divided_by_16_is_6 :
  ∃ k : ℕ, 899830 = 16 * k + 6 :=
by
  sorry

end remainder_of_899830_divided_by_16_is_6_l584_584266


namespace area_of_region_d_l584_584415

theorem area_of_region_d :
  let D := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 12 ∧ p.1 ≥ p.2^2 / Real.sqrt 6}
  (area_D : ℝ) :=
  (∫ x in -Real.sqrt 6 .. Real.sqrt 6, ∫ y in (x^2 / Real.sqrt 6) .. Real.sqrt (12 - x^2), 1) = 3 * Real.pi + 2 := sorry

end area_of_region_d_l584_584415


namespace boat_speed_in_still_water_l584_584171

variable (x : ℝ)

theorem boat_speed_in_still_water :
  (∀ (current_speed downstream_distance : ℝ),
    current_speed = 8 ∧
    downstream_distance = 6.283333333333333
    → (x + current_speed) * (13 / 60) = downstream_distance →
    x = 21) :=
begin
  intros current_speed downstream_distance h_condition h_equation,
  have h1 : current_speed = 8 := h_condition.1,
  have h2 : downstream_distance = 6.283333333333333 := h_condition.2,
  rw [h1, h2] at h_equation,
  sorry
end

end boat_speed_in_still_water_l584_584171


namespace shortest_distance_parabola_line_l584_584922

noncomputable def distance (C D : ℝ × ℝ) : ℝ :=
  real.sqrt ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2)

def parabola (x : ℝ) : ℝ := x^2 - 4 * x + 4
def line (x : ℝ) : ℝ := 2 * x - 6

theorem shortest_distance_parabola_line :
  let C := λ c : ℝ, (c, parabola c) in
  let D := λ d : ℝ, (d, line d) in
  ∀ c : ℝ, ∀ d : ℝ, d = c → 
  distance (C 3) (D 3) = 1 / real.sqrt 5 :=
by
  sorry

end shortest_distance_parabola_line_l584_584922


namespace center_of_symmetry_on_line_l584_584767

theorem center_of_symmetry_on_line :
  (∃ (θ : ℝ), ∀ (x y : ℝ), x = -1 + cos θ ∧ y = 2 + sin θ → y = -2 * x) :=
sorry

end center_of_symmetry_on_line_l584_584767


namespace sqrt_sum_simplify_l584_584123

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l584_584123


namespace find_non_integer_root_l584_584540

noncomputable def p (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) (x y : ℝ) : ℝ :=
  a₀ + a₁ * x + a₂ * y + a₃ * x^2 + a₄ * x * y + a₅ * y^2 + a₆ * x^3 + a₇ * x^2 * y + a₈ * x * y^2 + a₉ * y^3

theorem find_non_integer_root :
  (∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ
    (p a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ 0 0 = 0) ∧
    (p a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ 1 0 = 0) ∧
    (p a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ (-1) 0 = 0) ∧
    (p a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ 0 1 = 0) ∧
    (p a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ 0 (-1) = 0) ∧
    (p a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ 1 1 = 0) ∧
    (p a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ 1 (-1) = 0) ∧
    (p a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ 2 2 = 0),
    p a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ (5 / 19) (16 / 19) = 0) :=
sorry

end find_non_integer_root_l584_584540


namespace irreducible_fractions_choice_l584_584587

-- Given that p, q be integers
-- p and q are not divisible by 2 or 5
-- r = 500
-- p + q = 1000

theorem irreducible_fractions_choice :
  {p : ℕ // gcd p 500 = 1} × {q : ℕ // gcd q 500 = 1} × (p + q = 1000) → 
  200 := 
sorry

end irreducible_fractions_choice_l584_584587


namespace solution_set_of_inequality_l584_584170

theorem solution_set_of_inequality (x : ℝ) : 
  (3*x^2 - 4*x + 7 > 0) → (1 - 2*x) / (3*x^2 - 4*x + 7) ≥ 0 ↔ x ≤ 1 / 2 :=
by
  intros
  sorry

end solution_set_of_inequality_l584_584170


namespace sum_of_solutions_is_156_l584_584628

def relatively_prime (p q : ℕ) : Prop := Nat.gcd p q = 1

theorem sum_of_solutions_is_156 (p q : ℕ) (a : ℚ) 
  (h : a = p / q) 
  (rel_prime : relatively_prime p q) 
  (summation_is_156 : ∑ x in { x : ℝ | (∃ (w : ℤ) (f : ℝ), 0 ≤ f ∧ f < 1 ∧ x = w + f ∧ w * f = a * x^3)}, x = 156) : 
  (p + q = 257) := 
sorry

end sum_of_solutions_is_156_l584_584628


namespace shorts_diff_from_jersey_prob_l584_584911

theorem shorts_diff_from_jersey_prob
  (shorts_colors : Finset ℕ)
  (jersey_colors : Finset ℕ)
  (h_shorts_len : shorts_colors.card = 3)
  (h_jerseys_len : jersey_colors.card = 4) :
  let mismatch_count := 
      shorts_colors.sum (λ s, (jersey_colors.filter (≠ s)).card) 
  in  mismatch_count / (shorts_colors.card * jersey_colors.card) = 5 / 6 :=
by 
  sorry

end shorts_diff_from_jersey_prob_l584_584911


namespace minimum_points_form_square_l584_584512

-- Problem definition for lattice points in a 4x4 grid and the property that 11 points always form a square
theorem minimum_points_form_square :
  ∀ (points : Finset (ℕ × ℕ)), (∀ i j, (i ≤ 3) → (j ≤ 3) → (i, j) ∈ points) →
  ((points.card = 11) → 
  ∃ (a b c d : ℕ × ℕ), a ∈ points ∧ b ∈ points ∧ c ∈ points ∧ d ∈ points ∧
  ((a.1 = b.1) ∧ (c.1 = d.1) ∧ (a.2 = c.2) ∧ (b.2 = d.2) ∨ 
   (a.1 = c.1) ∧ (b.1 = d.1) ∧ (a.2 = b.2) ∧ (c.2 = d.2))) :=
begin
  sorry
end

end minimum_points_form_square_l584_584512


namespace complex_purely_imaginary_m_value_l584_584503

theorem complex_purely_imaginary_m_value (m : ℝ) :
  (m^2 - 1 = 0) ∧ (m + 1 ≠ 0) → m = 1 :=
by
  sorry

end complex_purely_imaginary_m_value_l584_584503


namespace max_students_distribute_eq_pens_pencils_l584_584690

theorem max_students_distribute_eq_pens_pencils (n_pens n_pencils n : ℕ) (h_pens : n_pens = 890) (h_pencils : n_pencils = 630) :
  (∀ k : ℕ, k > n → (n_pens % k ≠ 0 ∨ n_pencils % k ≠ 0)) → (n = Nat.gcd n_pens n_pencils) := by
  sorry

end max_students_distribute_eq_pens_pencils_l584_584690


namespace sin_60_proof_l584_584367

noncomputable def sin_60_eq : Prop :=
  sin (60 * real.pi / 180) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq :=
sorry

end sin_60_proof_l584_584367


namespace ninety_percent_can_play_and_travel_free_l584_584875

theorem ninety_percent_can_play_and_travel_free (n : ℕ) (h : 0 < n) :
  ∃(play_ball_transport_free : ℕ → Prop), 
    (∀ i : ℕ, i < n → 
      (play_ball_transport_free i ↔ 
        (∃ (r_ball r_trans : ℕ), (∀ j : ℕ, j ≠ i → (distances (i, j)) ≤ r_ball → heights j < heights i) 
                                  ∧ 
                                  (∀ j : ℕ, j ≠ i → (distances (i, j)) ≤ r_trans → heights j > heights i))) 
      ∧ 
      (finset.card (play_ball_transport_free) ≥ (9 * n / 10))) :=
sorry

end ninety_percent_can_play_and_travel_free_l584_584875


namespace find_f_minus_2_l584_584461

-- Define the conditions and the function definitions
def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f(x) + x^2

-- Theorem statement proving f(-2) = -9 given the conditions
theorem find_f_minus_2 (f : ℝ → ℝ)
  (h_odd : ∀ x, F f (-x) = -F f x) 
  (h_f2 : f 2 = 1) 
  : f (-2) = -9 :=
by
  sorry

end find_f_minus_2_l584_584461


namespace ara_always_wins_if_first_bea_always_wins_if_first_l584_584334

-- Definitions for Ara can always win if he goes first
theorem ara_always_wins_if_first :
  ∀ (board : Array (Option Nat)), 
  (∀ i, i ∈ board → i ∈ {1, 2, 3, 4, 5}) → 
  (board.contains none) → 
  (ara_goal : (diagonal1_sum board = diagonal2_sum board)) → 
  (bea_goal : (diagonal1_sum board ≠ diagonal2_sum board)) → 
  (ara_first : true)
  → ∃ board', 
    (∀ i, i ∈ board' → i ∈ {1, 2, 3, 4, 5}) 
    ∧ (¬board'.contains none)
    ∧ diagonal1_sum board' = diagonal2_sum board'
:= sorry

-- Definitions for Bea can always win if she goes first
theorem bea_always_wins_if_first :
  ∀ (board : Array (Option Nat)), 
  (∀ i, i ∈ board → i ∈ {1, 2, 3, 4, 5}) → 
  (board.contains none) → 
  (ara_goal : (diagonal1_sum board = diagonal2_sum board)) →  
  (bea_goal : (diagonal1_sum board ≠ diagonal2_sum board)) → 
  (bea_first : true) 
  → ∃ board', 
    (∀ i, i ∈ board' → i ∈ {1, 2, 3, 4, 5})
    ∧ (¬board'.contains none)
    ∧ diagonal1_sum board' ≠ diagonal2_sum board'
:= sorry

end ara_always_wins_if_first_bea_always_wins_if_first_l584_584334


namespace train_length_l584_584323

theorem train_length (speed_first_train speed_second_train : ℝ) (length_second_train : ℝ) (cross_time : ℝ) (L1 : ℝ) : 
  speed_first_train = 100 ∧ 
  speed_second_train = 60 ∧ 
  length_second_train = 300 ∧ 
  cross_time = 18 → 
  L1 = 420 :=
by
  sorry

end train_length_l584_584323


namespace completing_the_square_result_l584_584658

theorem completing_the_square_result : ∀ (x : ℝ), x^2 - 4 * x + 1 = 0 → (x - 2)^2 = 3 :=
by intro x h
-- proof goes here
sorry

end completing_the_square_result_l584_584658


namespace monotonic_intervals_range_of_b_l584_584475

def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem monotonic_intervals (a : ℝ) (x : ℝ) (h : x > 0) :
  if a > 0 then (f a x is decreasing on (0, 1 / a) ∧ f a x is increasing on (1 / a, +∞)) 
  else f a x is decreasing on (0, +∞) :=
sorry

theorem range_of_b (a b : ℝ) (hp : a > 0) (hx : ∀ x > 0, f a x ≥ (a ^ 2) / 2 + b) :
  b ≤ 1 / 2 :=
sorry

end monotonic_intervals_range_of_b_l584_584475


namespace six_digits_sum_l584_584140

theorem six_digits_sum 
  (a b c d e f g : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e) (h5 : a ≠ f) (h6 : a ≠ g)
  (h7 : b ≠ c) (h8 : b ≠ d) (h9 : b ≠ e) (h10 : b ≠ f) (h11 : b ≠ g)
  (h12 : c ≠ d) (h13 : c ≠ e) (h14 : c ≠ f) (h15 : c ≠ g)
  (h16 : d ≠ e) (h17 : d ≠ f) (h18 : d ≠ g)
  (h19 : e ≠ f) (h20 : e ≠ g)
  (h21 : f ≠ g)
  (h22 : 2 ≤ a) (h23 : a ≤ 9) 
  (h24 : 2 ≤ b) (h25 : b ≤ 9) 
  (h26 : 2 ≤ c) (h27 : c ≤ 9)
  (h28 : 2 ≤ d) (h29 : d ≤ 9)
  (h30 : 2 ≤ e) (h31 : e ≤ 9)
  (h32 : 2 ≤ f) (h33 : f ≤ 9)
  (h34 : 2 ≤ g) (h35 : g ≤ 9)
  (h36 : a + b + c = 25)
  (h37 : d + e + f + g = 15)
  (h38 : b = e) :
  a + b + c + d + f + g = 31 := 
sorry

end six_digits_sum_l584_584140


namespace correct_answer_l584_584630

-- Definitions extracted from the conditions
def optionA : Prop := "The trace of a door moving in the air when it rotates"
def optionB : Prop := "The trajectory of a small stone flying in the air when thrown"
def optionC : Prop := "A meteor streaking across the sky"
def optionD : Prop := "The trace drawn on the windshield by a car wiper"

-- The key phenomenon description
def phenomenon : Prop := "line moving into surface"

-- Theorem stating that optionD is the correct answer
theorem correct_answer : (optionD = phenomenon) := 
sorry

end correct_answer_l584_584630


namespace sum_of_turning_angles_l584_584514

variable (radius distance : ℝ) (C : ℝ)

theorem sum_of_turning_angles (H1 : radius = 10) (H2 : distance = 30000) (H3 : C = 2 * radius * Real.pi) :
  (distance / C) * 2 * Real.pi ≥ 2998 :=
by
  sorry

end sum_of_turning_angles_l584_584514


namespace sin_60_proof_l584_584372

noncomputable def sin_60_eq_sqrt3_div_2 : Prop :=
  Real.sin (π / 3) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq_sqrt3_div_2 :=
sorry

end sin_60_proof_l584_584372


namespace aspirin_mass_percentages_l584_584747

noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def molar_mass_aspirin : ℝ := (9 * atomic_mass_C) + (8 * atomic_mass_H) + (4 * atomic_mass_O)

theorem aspirin_mass_percentages :
  let mass_percent_H := ((8 * atomic_mass_H) / molar_mass_aspirin) * 100
  let mass_percent_C := ((9 * atomic_mass_C) / molar_mass_aspirin) * 100
  let mass_percent_O := ((4 * atomic_mass_O) / molar_mass_aspirin) * 100
  mass_percent_H = 4.48 ∧ mass_percent_C = 60.00 ∧ mass_percent_O = 35.52 :=
by
  -- Placeholder for the proof
  sorry

end aspirin_mass_percentages_l584_584747


namespace negation_of_some_triangles_is_isosceles_l584_584163

-- Definition of the conditions
def some_triangles_are_isosceles : Prop :=
  ∃ t : Triangle, is_isosceles t

def negation_of_existential_is_universal (P : Prop) : Prop :=
  ∀ x, ¬ P

-- Claim to be proven
theorem negation_of_some_triangles_is_isosceles :
  negation_of_existential_is_universal some_triangles_are_isosceles =
  ∀ t : Triangle, ¬ is_isosceles t :=
sorry

end negation_of_some_triangles_is_isosceles_l584_584163


namespace ara_height_l584_584335

theorem ara_height (shea_height_now : ℝ) (shea_growth_percent : ℝ) (ara_growth_fraction : ℝ)
    (height_now : shea_height_now = 75) (growth_percent : shea_growth_percent = 0.25) 
    (growth_fraction : ara_growth_fraction = (2/3)) : 
    ∃ ara_height_now : ℝ, ara_height_now = 70 := by
  sorry

end ara_height_l584_584335


namespace probability_between_CD_l584_584068

-- Define the points A, B, C, and D on a line segment
variables (A B C D : ℝ)

-- Provide the conditions
axiom h_ab_ad : B - A = 4 * (D - A)
axiom h_ab_bc : B - A = 8 * (B - C)

-- Define the probability statement 
theorem probability_between_CD (AB length: ℝ) : 
  (0 ≤ A ∧ A < D ∧ D < C ∧ C < B) → (B - A = 1) → 
  (B - A = 4 * (D - A)) → (B - A = 8 * (B - C)) → 
  probability (A B C D) = 5 / 8 :=
by
  sorry

end probability_between_CD_l584_584068


namespace correct_statements_l584_584455

-- Define the conditions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := 2 < a ∧ a < 3
def r (a : ℝ) : Prop := a < 3
def s (a : ℝ) : Prop := a > 1

-- Prove the statements
theorem correct_statements (a : ℝ) : (p a → q a) ∧ (r a → q a) :=
by {
    sorry
}

end correct_statements_l584_584455


namespace part1_part2_l584_584833

-- Define the parabola C as y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line l with slope k passing through point P(-2, 1)
def line (x y k : ℝ) : Prop := y - 1 = k * (x + 2)

-- Part 1: Prove the range of k for which line l intersects parabola C at two points is -1 < k < -1/2
theorem part1 (k : ℝ) : 
  (∃ x y, parabola x y ∧ line x y k) ∧ (∃ u v, parabola u v ∧ u ≠ x ∧ line u v k) ↔ -1 < k ∧ k < -1/2 := sorry

-- Part 2: Prove the equations of line l when it intersects parabola C at only one point are y = 1, y = -x - 1, and y = -1/2 x
theorem part2 (k : ℝ) : 
  (∃! x y, parabola x y ∧ line x y k) ↔ (k = 0 ∨ k = -1 ∨ k = -1/2) := sorry

end part1_part2_l584_584833


namespace sgn_g_eq_neg_sgn_x_l584_584841

def sgn (x : ℝ) : ℝ :=
if x > 0 then 1 else if x = 0 then 0 else -1

variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (inc_f : ∀ x y, x ≤ y → f(x) ≤ f(y))
variable (a_gt_1 : a > 1)

def g (x : ℝ) : ℝ := f(x) - f(a * x)

theorem sgn_g_eq_neg_sgn_x : ∀ x, sgn(g f a x) = -sgn(x) :=
sorry

end sgn_g_eq_neg_sgn_x_l584_584841


namespace sum_of_other_y_coords_l584_584850

theorem sum_of_other_y_coords (y1 y2 y3 y4 : ℤ) (x1 x2 x3 x4 : ℤ) 
  (h1 : (x1, y1) = (3, 17)) (h2 : (x2, y2) = (9, -4)) 
  (h3 : (x3, y3)) (h4 : (x4, y4)) 
  (h_rect : ((x1, y1), (x2, y2)) = ((x3, y3), (x4, y4))) :
  y3 + y4 = 13 :=
sorry

end sum_of_other_y_coords_l584_584850


namespace median_of_81_consecutive_integers_l584_584172

theorem median_of_81_consecutive_integers (n : ℕ) (S : ℕ) (h1 : n = 81) (h2 : S = 9^5) : 
  let M := S / n in M = 729 :=
by
  sorry

end median_of_81_consecutive_integers_l584_584172


namespace total_profit_l584_584691

theorem total_profit (xInvestment yInvestment zInvestment zProfit : ℝ) (xMonths yMonths zMonths : ℝ) : 
  xInvestment = 36000 → 
  yInvestment = 42000 → 
  zInvestment = 48000 → 
  xMonths = 12 → 
  yMonths = 12 → 
  zMonths = 8 → 
  zProfit = 4096 → 
  let xTotal := xInvestment * xMonths in
  let yTotal := yInvestment * yMonths in
  let zTotal := zInvestment * zMonths in
  let totalInvestment := xTotal + yTotal + zTotal in
  let profitShare := zTotal / totalInvestment in
  let P := zProfit / profitShare in
  P = 14080 :=
by
  intros _
  intros _
  intros _
  intros _
  intros _
  intros _
  intros _
  let xTotal := xInvestment * xMonths
  let yTotal := yInvestment * yMonths
  let zTotal := zInvestment * zMonths
  let totalInvestment := xTotal + yTotal + zTotal
  let profitShare := zTotal / totalInvestment
  let P := zProfit / profitShare
  sorry

end total_profit_l584_584691


namespace area_ratio_ABMO_EDCMO_l584_584878

variable {A B C D E F G H I J M N : Point}
variable (decagon : RegularDecagon A B C D E F G H I J)
variable (M_midpoint : Midpoint M B C)
variable (N_midpoint : Midpoint N F G)

theorem area_ratio_ABMO_EDCMO :
  (Area (Polygon.mk [A, B, M, O])) / (Area (Polygon.mk [E, D, C, M, O])) = 3 / 5 := 
sorry

end area_ratio_ABMO_EDCMO_l584_584878


namespace series_accuracy_l584_584848

theorem series_accuracy (ε : ℝ) (hε : 0 < ε ∧ ε = 0.01) :
  let a := λ (n : ℕ), (↑n / ((2 * ↑n + 1) * 5^n)) in
  abs (a 3) < ε :=
by
  let a := λ (n : ℕ), (↑n / ((2 * ↑n + 1) * 5^n))
  sorry

end series_accuracy_l584_584848


namespace leo_keeps_no_balloons_l584_584017

theorem leo_keeps_no_balloons :
  let blue_balloons := 23
      orange_balloons := 19
      violet_balloons := 47
      aqua_balloons := 55
      friends := 9
      total_balloons := blue_balloons + orange_balloons + violet_balloons + aqua_balloons
  in
  total_balloons % friends = 0 :=
by
  let blue_balloons := 23
  let orange_balloons := 19
  let violet_balloons := 47
  let aqua_balloons := 55
  let friends := 9
  let total_balloons := blue_balloons + orange_balloons + violet_balloons + aqua_balloons
  show total_balloons % friends = 0
  sorry

end leo_keeps_no_balloons_l584_584017


namespace calculate_loss_percentage_l584_584303

theorem calculate_loss_percentage
  (CP SP₁ SP₂ : ℝ)
  (h₁ : SP₁ = CP * 1.05)
  (h₂ : SP₂ = 1140) :
  (CP = 1200) → (SP₁ = 1260) → ((CP - SP₂) / CP * 100 = 5) :=
by
  intros h1 h2
  -- Here, we will eventually provide the actual proof steps.
  sorry

end calculate_loss_percentage_l584_584303


namespace west_movement_is_negative_seven_l584_584579

-- Define a function to represent the movement notation
def movement_notation (direction: String) (distance: Int) : Int :=
  if direction = "east" then distance else -distance

-- Define the movement in the east direction
def east_movement := movement_notation "east" 3

-- Define the movement in the west direction
def west_movement := movement_notation "west" 7

-- Theorem statement
theorem west_movement_is_negative_seven : west_movement = -7 := by
  sorry

end west_movement_is_negative_seven_l584_584579


namespace smallest_perimeter_iso_triangle_l584_584233

theorem smallest_perimeter_iso_triangle :
  ∃ (x y : ℕ), (PQ = PR ∧ PQ = x ∧ PR = x ∧ QR = y ∧ QJ = 10 ∧ PQ + PR + QR = 416 ∧ 
  PQ = PR ∧ y = 8 ∧ 2 * (x + y) = 416 ∧ y^2 - 50 > 0 ∧ y < 10) :=
sorry

end smallest_perimeter_iso_triangle_l584_584233


namespace probability_of_point_between_C_and_D_l584_584064

open_locale classical

noncomputable theory

def probability_C_to_D (A B C D : ℝ) (AB AD BC : ℝ) (h1 : AB = 4 * AD) (h2 : AB = 8 * BC) : ℝ :=
  let CD := BC - AD in
  CD / AB

theorem probability_of_point_between_C_and_D 
  (A B C D AB AD BC : ℝ) 
  (h1 : AB = 4 * AD) 
  (h2 : AB = 8 * BC) : 
  probability_C_to_D A B C D AB AD BC h1 h2 = 5 / 8 :=
by
  sorry

end probability_of_point_between_C_and_D_l584_584064


namespace probability_three_unused_rockets_expected_targets_hit_l584_584727

section RocketArtillery

variables (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1)

-- Probability that exactly three unused rockets remain after firing at five targets
theorem probability_three_unused_rockets (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let prob := 10 * (p ^ 3) * ((1 - p) ^ 2) in
  prob = 10 * (p ^ 3) * ((1 - p) ^ 2) := 
by
  sorry

-- Expected number of targets hit if there are nine targets
theorem expected_targets_hit (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let expected_hits := 10 * p - (p ^ 10) in
  expected_hits = 10 * p - (p ^ 10) := 
by
  sorry

end RocketArtillery

end probability_three_unused_rockets_expected_targets_hit_l584_584727


namespace pencil_distribution_l584_584781

open_locale big_operators

theorem pencil_distribution : 
  ∃ (ways : ℕ), ways = 58 ∧ ∃ (a b c d : ℕ), a + b + c + d = 10 ∧ a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ d ≥ 1 :=
by 
  sorry

end pencil_distribution_l584_584781


namespace smallest_four_digit_palindrome_divisible_by_6_l584_584671

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_divisible_by_6 (n : ℕ) : Prop :=
  n % 6 = 0

def is_four_digit (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

theorem smallest_four_digit_palindrome_divisible_by_6 : 
  ∃ (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ is_divisible_by_6 n ∧ 
  ∀ (m : ℕ), is_four_digit m ∧ is_palindrome m ∧ is_divisible_by_6 m → n ≤ m :=
  sorry

end smallest_four_digit_palindrome_divisible_by_6_l584_584671


namespace linear_function_has_form_l584_584435

theorem linear_function_has_form (f : ℝ → ℝ)
  (h_linear : ∃ k b, ∀ x, f(x) = k * x + b)
  (h1 : f(2) = 1)
  (h2 : f(-1) = -5) :
  ∃ k b, k = 2 ∧ b = -3 ∧ ∀ x, f(x) = k * x + b :=
by
  sorry

end linear_function_has_form_l584_584435


namespace speed_in_still_water_l584_584261

variable (upstream downstream : ℝ)

-- Conditions
def upstream_speed : Prop := upstream = 26
def downstream_speed : Prop := downstream = 40

-- Question and correct answer
theorem speed_in_still_water (h1 : upstream_speed upstream) (h2 : downstream_speed downstream) :
  (upstream + downstream) / 2 = 33 := by
  sorry

end speed_in_still_water_l584_584261


namespace range_of_a_l584_584469

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x : ℝ, if x < 1 then x^2 - 4 * x + a else log x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≥ 4 :=
begin
  sorry
end

end range_of_a_l584_584469


namespace area_of_AEB_l584_584883

-- Definition of conditions
variables (AB BC CD DF GC : ℝ)
variables (rectangle : CD = AB)
variables (FG: FG = CD - DF - GC)
variables (EH: 2 * EH = EH + 6)
variables (area_AEB: ℝ)
variables (E: ℝ)

-- The main proof statement in Lean 4
theorem area_of_AEB :
  ∃ (AB BC DF GC CD : ℝ),
    AB = 8 ∧ BC = 6 ∧ CD = 8 ∧ DF = 3 ∧ GC = 1 ∧
    CD = AB ∧
    FG = 4 ∧
    (2 * EH = EH + 6) ∧
    (area_AEB = 24) :=
  sorry

end area_of_AEB_l584_584883


namespace triangle_angle_60_or_120_l584_584529

theorem triangle_angle_60_or_120
  (ABC : Type)
  [triangle ABC]
  {A B C E D : ABC}
  (is_bisector_AE : is_bisector A E)
  (is_bisector_CD : is_bisector C D)
  (angle_CDE_eq_30 : angle C D E = 30) : 
  (angle A B C = 60) ∨ (angle A B C = 120) :=
sorry

end triangle_angle_60_or_120_l584_584529


namespace simplify_sqrt72_add_sqrt32_l584_584130

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l584_584130


namespace sum_lent_correct_l584_584304

-- Define the terms and conditions
variables {P I : ℝ}
def R : ℝ := 4
def T : ℝ := 8
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := (P * R * T) / 100

-- The conditions provided
def condition1 : I = P - 238 := sorry
def condition2 : simple_interest P R T = I := sorry

-- The proof statement: Prove that P equals 350
theorem sum_lent_correct : P = 350 :=
  begin
    sorry  -- Proof will be written here
  end

end sum_lent_correct_l584_584304


namespace west_move_7m_l584_584581

-- Definitions and conditions
def east_move (distance : Int) : Int := distance -- Moving east
def west_move (distance : Int) : Int := -distance -- Moving west is represented as negative

-- Problem: Prove that moving west by 7m is denoted by -7m given the conditions.
theorem west_move_7m : west_move 7 = -7 :=
by
  -- Proof will be handled here normally, but it's omitted as per instruction
  sorry

end west_move_7m_l584_584581


namespace square_plot_area_is_289_l584_584677

theorem square_plot_area_is_289:
  (price_per_foot fence: ℝ = 57) →
  (total_cost: ℝ = 3876) →
  (perimeter s: ℝ = 4 * s) →
  (cost_eq: 4 * s * 57 = 3876) →
  (s = 17) →
  (area: ℝ = s^2) →
  area = 289 :=
begin
  intros,
  sorry
end

end square_plot_area_is_289_l584_584677


namespace sqrt_sum_l584_584113

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l584_584113


namespace sqrt_sum_simplify_l584_584116

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l584_584116


namespace possible_missile_numbers_l584_584786

noncomputable def systematic_sampling (total_missiles : ℕ) (num_selected : ℕ) (interval : ℕ) (start : ℕ) : list ℕ :=
(list.range num_selected).map (λ i, start + i * interval)

theorem possible_missile_numbers :
  let total_missiles := 50
  let num_selected := 5
  let possible_set := {3, 13, 23, 33, 43}
  ∃ (start interval : ℕ), systematic_sampling total_missiles num_selected interval start = [3, 13, 23, 33, 43] :=
sorry

end possible_missile_numbers_l584_584786


namespace joe_used_225_gallons_l584_584267

def initial_paint : ℕ := 360

def paint_first_week (initial : ℕ) : ℕ := initial / 4

def remaining_paint_after_first_week (initial : ℕ) : ℕ :=
  initial - paint_first_week initial

def paint_second_week (remaining : ℕ) : ℕ := remaining / 2

def total_paint_used (initial : ℕ) : ℕ :=
  paint_first_week initial + paint_second_week (remaining_paint_after_first_week initial)

theorem joe_used_225_gallons :
  total_paint_used initial_paint = 225 :=
by
  sorry

end joe_used_225_gallons_l584_584267


namespace students_play_neither_l584_584515

theorem students_play_neither (S F T B : ℕ) (h₁ : S = 50) (h₂ : F = 32) (h₃ : T = 28) (h₄ : B = 24) :
  S - (F + T - B) = 14 := by
  -- Students that play either football or tennis (or both)
  have h_total_play : F + T - B = 36 := by
    rw [h₂, h₃, h₄]
    norm_num

  -- Number of students who play neither sport
  have h_neither : S - 36 = 14 := by
    rw [h₁, h_total_play]
    norm_num

  exact h_neither

-- Use sorry to skip the proof if needed

end students_play_neither_l584_584515


namespace needs_debugging_defective_parts_count_parts_exceeding_200_12_l584_584895

noncomputable def normal_distribution (μ σ : ℝ) : Type := sorry

def part_inner_diameter := normal_distribution 200 0.06

def valid_inner_diameter_range := (199.82, 200.18)

def measured_diameters := [199.87, 199.91, 199.99, 200.13, 200.19]

theorem needs_debugging : ¬ (∀ x ∈ measured_diameters, 199.82 < x ∧ x < 200.18) :=
sorry

def total_parts := 10000

def defective p : ℕ → ℝ := (1 - p)

theorem defective_parts_count (p : ℝ) (n : ℕ) (h1 : p = 0.003) (h2 : n = total_parts) :
  30 ≤ n * p ∧ n * p < 31 :=
sorry

theorem parts_exceeding_200_12 (p : ℝ) (h : p = 0.0225) :
  225 ≤ total_parts * p :=
sorry

end needs_debugging_defective_parts_count_parts_exceeding_200_12_l584_584895


namespace sqrt_sum_simplify_l584_584117

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l584_584117


namespace square_division_l584_584591

theorem square_division (n : ℕ) (h : n ≥ 6) : ∃ squares : ℕ, squares = n ∧ can_divide_into_squares(squares) := 
sorry

end square_division_l584_584591


namespace roots_of_f_min_value_of_f_l584_584819

noncomputable def f (a x : ℝ) : ℝ :=
  abs (x^2 - x) - a * x

theorem roots_of_f (a : ℝ) (ha : a = 1 / 3) :
  {x : ℝ | f a x = 0} = {0, 2 / 3, 4 / 3} :=
by
  sorry

theorem min_value_of_f (a : ℝ) (ha : a ≤ -1) :
  ∃ x : ℝ, x ∈ Icc (-2 : ℝ) 3 ∧ ∀ y ∈ Icc (-2 : ℝ) 3, f a x ≤ f a y ∧
  f a x = if a ≤ -5 then 2 * a + 6 else - (a + 1)^2 / 4 :=
by
  sorry

end roots_of_f_min_value_of_f_l584_584819


namespace not_quadratic_D_l584_584257

-- Define the given functions A, B, C, D
def funcA (x : ℝ) := (x - 1)^2
def funcB (x : ℝ) := sqrt 2 * x^2 - 1
def funcC (x : ℝ) := 3 * x^2 + 2 * x - 1
def funcD (x : ℝ) := (x + 1)^2 - x^2

-- Prove that funcD is not a quadratic function while funcA, funcB, and funcC are quadratic functions.
theorem not_quadratic_D : ¬ (∃ a b c : ℝ, (∀ x : ℝ, funcD x = a * x^2 + b * x + c)) ∧ 
  (∃ a b c : ℝ, (∀ x : ℝ, funcA x = a * x^2 + b * x + c)) ∧
  (∃ a b c : ℝ, (∀ x : ℝ, funcB x = a * x^2 + b * x + c)) ∧
  (∃ a b c : ℝ, (∀ x : ℝ, funcC x = a * x^2 + b * x + c)) :=
by
  sorry

end not_quadratic_D_l584_584257


namespace product_of_base8_digits_of_8654_l584_584665

theorem product_of_base8_digits_of_8654 : 
  let base10 := 8654
  let base8_rep := [2, 0, 7, 1, 6] -- Representing 8654(10) to 20716(8)
  (base8_rep.prod = 0) :=
  sorry

end product_of_base8_digits_of_8654_l584_584665


namespace number_of_tiles_l584_584315

open Real

noncomputable def room_length : ℝ := 10
noncomputable def room_width : ℝ := 15
noncomputable def tile_length : ℝ := 5 / 12
noncomputable def tile_width : ℝ := 2 / 3

theorem number_of_tiles :
  (room_length * room_width) / (tile_length * tile_width) = 540 := by
  sorry

end number_of_tiles_l584_584315


namespace area_triangle_ineq_l584_584541

-- Define the input conditions
variables {A B C M N P A1 B1 C1 : Type}
variable [triangle : Triangle A B C]

-- Assume M, N, P are midpoints
variable [midpoint_M : Midpoint M B C]
variable [midpoint_N : Midpoint N C A]
variable [midpoint_P : Midpoint P A B]

-- Assume A1, B1, C1 lie on the circumcircle
variables [circumcircle : Circumcircle A B C A1 B1 C1]
variables [intersection_A1 : Intersection A1 A M]
variables [intersection_B1 : Intersection B1 B N]
variables [intersection_C1 : Intersection C1 C P]

-- The statement to be proven
theorem area_triangle_ineq :
  Area (triangle ABC) ≤ Area (triangle BCA1) + Area (triangle CAB1) + Area (triangle ABC1) :=
sorry

end area_triangle_ineq_l584_584541


namespace probability_of_at_least_one_white_ball_l584_584521

-- Define the conditions
def bagA := { red := 3, white := 2 }
def bagB := { red := 2, white := 1 }

-- Define the problem as a theorem statement
theorem probability_of_at_least_one_white_ball :
  ((bagA.red + bagA.white) = 5) ∧ ((bagB.red + bagB.white) = 3) →
  (let total_draws := 15 in  
    let white_ball_draws := 7 + 2 in 
    (white_ball_draws / total_draws : ℚ) = 3 / 5) :=
by
  intros h_sum
  sorry

end probability_of_at_least_one_white_ball_l584_584521


namespace median_of_81_consecutive_integers_l584_584192

theorem median_of_81_consecutive_integers (sum : ℕ) (h₁ : sum = 9^5) : 
  let mean := sum / 81 in mean = 729 :=
by
  have h₂ : sum = 59049 := by
    rw h₁
    norm_num
  have h₃ : mean = 59049 / 81 := by
    rw h₂
    rfl
  have result : mean = 729 := by
    rw h₃
    norm_num
  exact result

end median_of_81_consecutive_integers_l584_584192


namespace odd_square_divisors_l584_584406

theorem odd_square_divisors (n : ℕ) (h_odd : n % 2 = 1) : 
  ∃ (f g : ℕ), (f > g) ∧ (∀ d, d ∣ (n * n) → d % 4 = 1 ↔ (0 < f)) ∧ (∀ d, d ∣ (n * n) → d % 4 = 3 ↔ (0 < g)) :=
by
  sorry

end odd_square_divisors_l584_584406


namespace find_f_neg3_l584_584031

-- Define the even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f(x) = f(-x)

-- Given conditions
variables (f : ℝ → ℝ)
variable h1 : is_even f
variable h2 : ∀ x > 0, f(2 + x) = -2 * f(2 - x)
variable h3 : f(-1) = 4

-- Statement to prove
theorem find_f_neg3 : f(-3) = -8 :=
by {
    sorry
}

end find_f_neg3_l584_584031


namespace number_of_real_roots_l584_584604

theorem number_of_real_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a * b^2 + 1 = 0) :
  (c > 0 → ∃ x1 x2 x3 : ℝ, 
    (x1 = b * Real.sqrt c ∨ x1 = -b * Real.sqrt c ∨ x1 = -c / b) ∧
    (x2 = b * Real.sqrt c ∨ x2 = -b * Real.sqrt c ∨ x2 = -c / b) ∧
    (x3 = b * Real.sqrt c ∨ x3 = -b * Real.sqrt c ∨ x3 = -c / b)) ∧
  (c < 0 → ∃ x1 : ℝ, x1 = -c / b) :=
by
  sorry

end number_of_real_roots_l584_584604


namespace set_pairings_l584_584788

open Set

theorem set_pairings (A B : Set ℕ) (C : Set (ℕ × ℕ)) :
  A = {1, 3, 5, 7} →
  B = {2, 4, 6} →
  C = {(x, y) | x ∈ A ∧ y ∈ B} →
  C = {(1, 2), (1, 4), (1, 6), (3, 2), (3, 4), (3, 6), (5, 2), (5, 4), (5, 6), (7, 2), (7, 4), (7, 6)} :=
sorry

end set_pairings_l584_584788


namespace probability_three_unused_rockets_expected_targets_hit_l584_584728

section RocketArtillery

variables (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1)

-- Probability that exactly three unused rockets remain after firing at five targets
theorem probability_three_unused_rockets (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let prob := 10 * (p ^ 3) * ((1 - p) ^ 2) in
  prob = 10 * (p ^ 3) * ((1 - p) ^ 2) := 
by
  sorry

-- Expected number of targets hit if there are nine targets
theorem expected_targets_hit (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let expected_hits := 10 * p - (p ^ 10) in
  expected_hits = 10 * p - (p ^ 10) := 
by
  sorry

end RocketArtillery

end probability_three_unused_rockets_expected_targets_hit_l584_584728


namespace simplify_sqrt72_add_sqrt32_l584_584129

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l584_584129


namespace convert_base8_to_base7_l584_584389

theorem convert_base8_to_base7 (n : ℕ) : n = 536 → (num_to_base7 536) = 1054 :=
by
  sorry

def num_to_base10 (n : ℕ) : ℕ :=
  let d2 := (n / 100) % 10 * 8^2
  let d1 := (n / 10) % 10 * 8^1
  let d0 := (n / 1) % 10 * 8^0
  d2 + d1 + d0

def num_to_base7_aux (n : ℕ) (acc : ℕ) (pos : ℕ) : ℕ :=
  if n = 0 then acc
  else
    let q := n / 7
    let r := n % 7
    num_to_base7_aux q ((r * 10^pos) + acc) (pos + 1)

def num_to_base7 (n : ℕ) : ℕ :=
  num_to_base7_aux (num_to_base10 n) 0 0

end convert_base8_to_base7_l584_584389


namespace smallest_angle_convex_15_polygon_l584_584987

theorem smallest_angle_convex_15_polygon :
  ∃ (a : ℕ) (d : ℕ), (∀ n : ℕ, n ∈ Finset.range 15 → (a + n * d < 180)) ∧
  15 * (a + 7 * d) = 2340 ∧ 15 * d <= 24 -> a = 135 :=
by
  -- Proof omitted
  sorry

end smallest_angle_convex_15_polygon_l584_584987


namespace variance_of_data_set_l584_584815

theorem variance_of_data_set (data : List ℝ) (h : data = [4.7, 4.8, 5.1, 5.4, 5.5]) : 
  let n := data.length
  let mean := (data.sum / n)
  S^2 = (1 / n) * (data.map (λ x => (x - mean) ^ 2)).sum
  S^2 = 0.1 := 
by 
  sorry

end variance_of_data_set_l584_584815


namespace triangle_right_triangle_l584_584211

-- Defining the sides of the triangle
variables (a b c : ℝ)

-- Theorem statement
theorem triangle_right_triangle (h : (a + b)^2 = c^2 + 2 * a * b) : a^2 + b^2 = c^2 :=
by {
  sorry
}

end triangle_right_triangle_l584_584211


namespace smallest_perimeter_triangle_l584_584242

theorem smallest_perimeter_triangle (PQ PR QR : ℕ) (J : Point) :
  PQ = PR →
  QJ = 10 →
  QR = 2 * 10 →
  PQ + PR + QR = 40 :=
by
  sorry

structure Point : Type :=
mk :: (QJ : ℕ)

noncomputable def smallest_perimeter_triangle : Prop :=
  ∃ (PQ PR QR : ℕ) (J : Point), PQ = PR ∧ J.QJ = 10 ∧ QR = 2 * 10 ∧ PQ + PR + QR = 40

end smallest_perimeter_triangle_l584_584242


namespace solve_eq_l584_584603

theorem solve_eq (x : ℝ) (h : sqrt(9 * x - 2) + 18 / sqrt(9 * x - 2) = 11) : 
  x = 83 / 9 ∨ x = 2 / 3 :=
sorry

end solve_eq_l584_584603


namespace odd_integers_between_fractions_l584_584488

theorem odd_integers_between_fractions :
  let a := 21 / 5
  let b := 47 / 3
  ∃ n, n = 6 ∧ ∀ k, (k > a ∧ k < b ∧ k % 2 = 1) ↔ k ∈ {5, 7, 9, 11, 13, 15} :=
by
  let a := 21 / 5
  let b := 47 / 3
  -- proof here
  sorry

end odd_integers_between_fractions_l584_584488


namespace non_zero_coefficient_r_l584_584159

noncomputable def distinct_roots 
(Q : Polynomial ℝ) 
(p q r s t : ℝ) : bool :=
(Q.roots = [0, α, -α, β, -β]) ∧
(p = 0) ∧ (q = 0) ∧ (s = 0) ∧ (t = 0) ∧ (Q.coeff 5 = 1) 

theorem non_zero_coefficient_r
  (p q r s t : ℝ)
  (Q : Polynomial ℝ)
  (hQ : Q = Polynomial.C t + Polynomial.monomial 1 s + 
              Polynomial.monomial 2 r + Polynomial.monomial 3 q +
              Polynomial.monomial 4 p + Polynomial.monomial 5 1)
  (h_roots : distinct_roots Q p q r s t):
  r ≠ 0 :=
by sorry

end non_zero_coefficient_r_l584_584159


namespace most_likely_hits_l584_584807

theorem most_likely_hits (p : ℝ) (n : ℕ) (h : p = 0.8) (hn : n = 6) :
  (p * n).round = 5 :=
by
  rw [h, hn]
  -- necessary computational steps and rounding logic go here
  sorry

end most_likely_hits_l584_584807


namespace find_x_l584_584869

-- Defining the sum of integers from 30 to 40 inclusive
def sum_30_to_40 : ℕ := (30 + 31 + 32 + 33 + 34 + 35 + 36 + 37 + 38 + 39 + 40)

-- Defining the number of even integers from 30 to 40 inclusive
def count_even_30_to_40 : ℕ := 6

-- Given that x + y = 391, and y = count_even_30_to_40
-- Prove that x is equal to 385
theorem find_x (h : sum_30_to_40 + count_even_30_to_40 = 391) : sum_30_to_40 = 385 :=
by
  simp [sum_30_to_40, count_even_30_to_40] at h
  sorry

end find_x_l584_584869


namespace min_distance_curve_to_point_l584_584889

open Real

noncomputable def min_distance_to_fixed_point : ℝ :=
  let A := (4, 4)
  let curve := λ x : ℝ, 1 / x
  let dist_sq := λ x : ℝ, (x - 4)^2 + (curve x - 4)^2
  let min_t := 4
  (dist_sq min_t)^0.5

theorem min_distance_curve_to_point :
  min_distance_to_fixed_point = sqrt 14 := 
sorry

end min_distance_curve_to_point_l584_584889


namespace mark_candy_bars_consumption_l584_584901

theorem mark_candy_bars_consumption 
  (recommended_intake : ℕ := 150)
  (soft_drink_calories : ℕ := 2500)
  (soft_drink_added_sugar_percent : ℕ := 5)
  (candy_bar_added_sugar_calories : ℕ := 25)
  (exceeded_percentage : ℕ := 100)
  (actual_intake := recommended_intake + (recommended_intake * exceeded_percentage / 100))
  (soft_drink_added_sugar := soft_drink_calories * soft_drink_added_sugar_percent / 100)
  (candy_bars_added_sugar := actual_intake - soft_drink_added_sugar)
  (number_of_bars := candy_bars_added_sugar / candy_bar_added_sugar_calories) : 
  number_of_bars = 7 := 
by
  sorry

end mark_candy_bars_consumption_l584_584901


namespace star_impossible_l584_584007

variables (A_1 A_2 A_3 A_4 A_5 : Point)

def above_plane (A B C P : Point) : Prop := -- omitted: definition of "above the plane formed by A, B, C"

def below_plane (A B C P : Point) : Prop := -- omitted: definition of "below the plane formed by A, B, C"

theorem star_impossible : 
  above_plane A_1 A_3 A_5 A_2 →
  below_plane A_1 A_3 A_5 A_4 →
  (segment A_2 A_4 → above_plane A_1 A_3 A_5 A_4 ∧ below_plane A_1 A_3 A_5 A_2) →
  false :=
by 
s sorry

end star_impossible_l584_584007


namespace tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth_l584_584854

theorem tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth (a : ℝ) (h : Real.tan a = 2) :
  Real.cos (2 * a) + Real.sin (2 * a) = 1 / 5 :=
by
  sorry

end tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth_l584_584854


namespace xiao_wang_mode_median_l584_584619

noncomputable def problem_points : list ℕ := [65, 57, 56, 58, 56, 58, 56]

def mode (l : list ℕ) : ℕ :=
  l.group_by id 
    |> list.map (λ g, (g.head, g.length))
    |> list.max_by (λ x, x.snd)
    |> prod.fst

def median (l : list ℕ) : ℕ :=
  let sorted := l.qsort (≤)
  sorted.get (sorted.length / 2)

theorem xiao_wang_mode_median :
  mode problem_points = 56 ∧ median problem_points = 57 :=
by
  sorry

end xiao_wang_mode_median_l584_584619


namespace smallest_angle_of_convex_15_gon_arithmetic_sequence_l584_584995

theorem smallest_angle_of_convex_15_gon_arithmetic_sequence :
  ∃ (a d : ℕ), (∀ k : ℕ, k < 15 → (let angle := a + k * d in angle < 180)) ∧
  (∀ i j : ℕ, i < j → i < 15 → j < 15 → (a + i * d) < (a + j * d)) ∧
  (let sequence_sum := 15 * a + d * 7 * 14 in sequence_sum = 2340) ∧
  (d = 3) ∧
  (a = 135) :=
by
  sorry

end smallest_angle_of_convex_15_gon_arithmetic_sequence_l584_584995


namespace perimeter_bounds_l584_584245

theorem perimeter_bounds (r p : ℝ) (h : 0 < r) :
    r * (2 - real.sqrt 2) ≤ p ∧ p ≤ r * (2 + real.sqrt 2) :=
begin
  sorry
end

end perimeter_bounds_l584_584245


namespace find_range_of_a_l584_584493

theorem find_range_of_a (a : ℝ) :
    (∀ ε > 0, ∃ N, ∀ n ≥ N, |(3^n / (3^(n+1) + (a+1)^n)) - 1/3| < ε) →
    a ∈ Ioo (-4 : ℝ) 2 :=
begin
  sorry
end

end find_range_of_a_l584_584493


namespace right_triangle_perimeter_eq_sum_radii_l584_584965

variables {a b c : ℝ}

def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

def radius_inscribed (a b c s : ℝ) := s - c

def radius_excircle_a (s a : ℝ) := s - a
def radius_excircle_b (s b : ℝ) := s - b
def radius_excircle_c (s : ℝ) := s

theorem right_triangle_perimeter_eq_sum_radii
  (h_right : c^2 = a^2 + b^2)
  (s := semi_perimeter a b c)
  (ρ := radius_inscribed a b c s)
  (ρ_a := radius_excircle_a s a)
  (ρ_b := radius_excircle_b s b)
  (ρ_c := radius_excircle_c s) :
  a + b + c = ρ + ρ_a + ρ_b + ρ_c :=
sorry

end right_triangle_perimeter_eq_sum_radii_l584_584965


namespace expression_for_T_l584_584021

theorem expression_for_T (y : ℝ) :
  let T := (y - 1)^5 + 5 * (y - 1)^4 + 10 * (y - 1)^3 + 10 * (y - 1)^2 + 5 * (y - 1) + 1
  in T = y^5 :=
by
  let T := (y - 1)^5 + 5 * (y - 1)^4 + 10 * (y - 1)^3 + 10 * (y - 1)^2 + 5 * (y - 1) + 1
  show T = y^5
  sorry

end expression_for_T_l584_584021


namespace simplify_sqrt_sum_l584_584103

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l584_584103


namespace increasing_cos_ax_implies_a_geq_1_l584_584508

theorem increasing_cos_ax_implies_a_geq_1 (a : ℝ) :
  (∀ x : ℝ, -π/2 ≤ x ∧ x ≤ π/2 → -sin x + a ≥ 0) → a ≥ 1 :=
by
  sorry

end increasing_cos_ax_implies_a_geq_1_l584_584508


namespace probability_of_point_between_C_and_D_l584_584067

open_locale classical

noncomputable theory

def probability_C_to_D (A B C D : ℝ) (AB AD BC : ℝ) (h1 : AB = 4 * AD) (h2 : AB = 8 * BC) : ℝ :=
  let CD := BC - AD in
  CD / AB

theorem probability_of_point_between_C_and_D 
  (A B C D AB AD BC : ℝ) 
  (h1 : AB = 4 * AD) 
  (h2 : AB = 8 * BC) : 
  probability_C_to_D A B C D AB AD BC h1 h2 = 5 / 8 :=
by
  sorry

end probability_of_point_between_C_and_D_l584_584067


namespace age_ratio_l584_584697

variable (p q : ℕ)

-- Conditions
def condition1 := p - 6 = (q - 6) / 2
def condition2 := p + q = 21

-- Theorem stating the desired ratio
theorem age_ratio (h1 : condition1 p q) (h2 : condition2 p q) : p / Nat.gcd p q = 3 ∧ q / Nat.gcd p q = 4 :=
by
  sorry

end age_ratio_l584_584697


namespace room_breadth_l584_584420

theorem room_breadth (length height diagonal : ℕ) (h_length : length = 12) (h_height : height = 9) (h_diagonal : diagonal = 17) : 
  ∃ breadth : ℕ, breadth = 8 :=
by
  -- Using the three-dimensional Pythagorean theorem:
  -- d² = length² + breadth² + height²
  -- 17² = 12² + b² + 9²
  -- 289 = 144 + b² + 81
  -- 289 = 225 + b²
  -- b² = 289 - 225
  -- b² = 64
  -- Taking the square root of both sides, we find:
  -- b = √64
  -- b = 8
  let b := 8
  existsi b
  -- This is a skip step, where we assert the breadth equals 8
  sorry

end room_breadth_l584_584420


namespace sum_of_satisfying_integers_l584_584425

noncomputable def cot (x : ℝ) := 1 / tan x
noncomputable def inequality (x : ℤ) : Prop :=
  (1 - (cot (Real.pi * x / 12)) ^ 2) *
  (1 - 3 * (cot (Real.pi * x / 12)) ^ 2) *
  (1 - tan (Real.pi * x / 6) * (cot (Real.pi * x / 4))) ≤ 16

theorem sum_of_satisfying_integers :
  (Finset.filter (λx, inequality x) (Finset.Icc (-3 : ℤ) 13)).sum id = 28 :=
  sorry

end sum_of_satisfying_integers_l584_584425


namespace problem_statement_l584_584595

-- Define S(n) as the given series
def S (n : ℕ) : ℤ := Finset.sum (Finset.range (n + 1)) (λ m, (-1 : ℤ)^m * Nat.choose n m)

-- Statement of the problem in Lean 4
theorem problem_statement : 1990 * (∑ m in Finset.range 1991, (-1 : ℤ)^m / (1990 - m) * Nat.choose 1990 m) + 1 = 0 := by
  sorry

end problem_statement_l584_584595


namespace csc_of_7pi_over_4_l584_584410

theorem csc_of_7pi_over_4 :
  csc (7 * Real.pi / 4) = -Real.sqrt 2 :=
by
  sorry

end csc_of_7pi_over_4_l584_584410


namespace sqrt_sum_simplify_l584_584120

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l584_584120


namespace monotonic_intervals_range_of_b_l584_584474

def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem monotonic_intervals (a : ℝ) (x : ℝ) (h : x > 0) :
  if a > 0 then (f a x is decreasing on (0, 1 / a) ∧ f a x is increasing on (1 / a, +∞)) 
  else f a x is decreasing on (0, +∞) :=
sorry

theorem range_of_b (a b : ℝ) (hp : a > 0) (hx : ∀ x > 0, f a x ≥ (a ^ 2) / 2 + b) :
  b ≤ 1 / 2 :=
sorry

end monotonic_intervals_range_of_b_l584_584474


namespace sufficient_but_not_necessary_l584_584270

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1 → x^2 > 1) ∧ ¬(x^2 > 1 → x < -1) :=
by
  sorry

end sufficient_but_not_necessary_l584_584270


namespace H_lies_on_line_EF_l584_584018

variables {A B C D E F H X Y : Type}
variables [is_acute_angled_triangle : acute_angled_triangle A B C]
variables [is_altitude : altitude A D B E C F]
variables [is_cyclic_quadrilateral : cyclic_quadrilateral A D X Y]
variables [is_orthocenter : orthocenter H A X Y]

theorem H_lies_on_line_EF (h₁ : acute_angled_triangle A B C)
                         (h₂ : altitude A D B E C F)
                         (h₃ : \not= A B X)
                         (h₄ : \not= A C Y)
                         (h₅ : cyclic_quadrilateral A D X Y)
                         (h₆ : orthocenter H A X Y) : lies_on_line H E F :=
sorry

end H_lies_on_line_EF_l584_584018


namespace john_hits_free_throws_l584_584013

theorem john_hits_free_throws :
  ∀ (F: ℕ) (S: ℕ)
  (G: ℕ)
  (P : ℕ)
  (FT_hit : ℕ),
  (F = 5) ∧ (S = 2) ∧ (G = 20) ∧ (P = 80) ∧ (FT_hit = 112) → 
  (FT_hit.toRat / (Nat.ceil (P.toRat / 100 * G.toRat) * F * S)).toReal * 100 = 70 :=
by intros F S G P FT_hit h,
   cases h with h5 hrest,
   cases hrest with h2 hrest,
   cases hrest with h20 hrest,
   cases hrest with h80 h112,
   rw [h5, h2, h20, h80, h112],    
   sorry

end john_hits_free_throws_l584_584013


namespace new_milk_water_ratio_l584_584299

theorem new_milk_water_ratio
  (original_milk : ℚ)
  (original_water : ℚ)
  (added_water : ℚ)
  (h_ratio : original_milk / original_water = 2 / 1)
  (h_milk_qty : original_milk = 45)
  (h_added_water : added_water = 10) :
  original_milk / (original_water + added_water) = 18 / 13 :=
by
  sorry

end new_milk_water_ratio_l584_584299


namespace problem_lean_statement_l584_584936

variable {N : ℕ}
variable (n p : ℕ)

-- Composite natural number n and proper divisor p
def composite_n (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k
def proper_divisor (n p : ℕ) : Prop := p > 1 ∧ p < n ∧ n % p = 0

-- The binary representation of N
def binary_representation (n p N : ℕ) : Prop :=
  ((¬ (2 ∣ n / p) → ∃ k: ℕ, N = 2^k + 2^(k + p) + 2^(k + 2*p) + ... + 2 * 2^k) ∨
   (2 ∣ n / p → ∃ k: ℕ, N = 2^(k-1) + 2^(k - 1 + p + 1) + ... + 2 * 2^k))

-- The main statement to be proved
theorem problem_lean_statement (h1 : composite_n n) (h2 : proper_divisor n p)
  (h3 : ∃! N, ((1 + 2^n + 2^p - n) * N - 1) % (2^p) = 0) :
  binary_representation n p N :=
sorry

end problem_lean_statement_l584_584936


namespace twice_gcd_of_180_270_450_eq_180_l584_584676

theorem twice_gcd_of_180_270_450_eq_180 :
  let gcf (a b c : Nat) := (Nat.gcd (Nat.gcd a b) c)
  2 * gcf 180 270 450 = 180 := by
  have h1 : Nat.gcd 180 270 = 90 := by sorry
  have h2 : Nat.gcd 90 450 = 90 := by sorry
  have h3 : gcf 180 270 450 = 90 := by
    unfold gcf
    rw [h1, h2]
  rw [h3]
  norm_num

end twice_gcd_of_180_270_450_eq_180_l584_584676


namespace find_plane_equation_l584_584419

-- Definitions for points and planes
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def plane_eq (A B C D : ℝ) (p : Point3D) : Prop :=
  A * p.x + B * p.y + C * p.z + D = 0

noncomputable def perpendicular_planes (A1 B1 C1 d1 A2 B2 C2 d2 : ℝ) : Prop :=
  A1 * A2 + B1 * B2 + C1 * C2 = 0

def same_plane_eq_form (A B C D : ℝ) : Prop :=
  A > 0 ∧ Int.gcd (Int.natAbs A.natCeil) (Int.gcd (Int.natAbs B.natCeil) (Int.gcd (Int.natAbs C.natCeil) (Int.natAbs D.natCeil))) = 1

-- Given conditions as Lean definitions
def point1 := { x := 0, y := 2, z := 1 }
def point2 := { x := 2, y := 0, z := 1 }
def perp_plane := { x := 2, y := 1, z := 4 }

-- Problem statement in Lean
theorem find_plane_equation :
  ∃ (A B C D : ℝ), same_plane_eq_form A B C D ∧
  plane_eq A B C D point1 ∧
  plane_eq A B C D point2 ∧
  perpendicular_planes A B C D perp_plane.x perp_plane.y perp_plane.z -7 ∧
  plane_eq A B C D { x := 4, y := 4, z := -3 } :=
begin
  use [4, 4, -3, -5],
  split,
  {
    split, -- A > 0
    exact dec_trivial,
    reflexivity -- gcd(|A|, |B|, |C|, |D|) = 1
  },
  split,
  { unfold plane_eq, simp }, -- plane passes through point1
  split,
  { unfold plane_eq, simp }, -- plane passes through point2
  split,
  { unfold perpendicular_planes, simp }, -- plane is perpendicular
  { unfold plane_eq, simp } -- correct plane equation
end

end find_plane_equation_l584_584419


namespace smallest_angle_in_convex_15sided_polygon_l584_584999

def isConvexPolygon (n : ℕ) (angles : Fin n → ℚ) : Prop :=
  ∑ i, angles i = (n - 2) * 180 ∧ ∀ i,  angles i < 180

def arithmeticSequence (angles : Fin 15 → ℚ) : Prop :=
  ∃ a d : ℚ, ∀ i : Fin 15, angles i = a + i * d

def increasingSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i j : Fin 15, i < j → angles i < angles j

def integerSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i : Fin 15, (angles i : ℚ) = angles i

theorem smallest_angle_in_convex_15sided_polygon :
  ∃ (angles : Fin 15 → ℚ),
    isConvexPolygon 15 angles ∧
    arithmeticSequence angles ∧
    increasingSequence angles ∧
    integerSequence angles ∧
    angles 0 = 135 :=
by
  sorry

end smallest_angle_in_convex_15sided_polygon_l584_584999


namespace sin_60_eq_sqrt3_div_2_l584_584366

theorem sin_60_eq_sqrt3_div_2 :
  ∃ (Q : ℝ × ℝ), dist Q (1, 0) = 1 ∧ angle (1, 0) Q = real.pi / 3 ∧ Q.2 = real.sqrt 3 / 2 := sorry

end sin_60_eq_sqrt3_div_2_l584_584366


namespace simplify_sqrt_sum_l584_584102

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l584_584102


namespace cardinality_of_sigma_algebra_l584_584043

section CardinalityOfSigmaAlgebra

variables {Ω : Type} {D : Set (Set Ω)}

-- Assume D is a countable partition of Ω with nonempty sets
def is_countable_partition (D : Set (Set Ω)) (Ω : Type) :=
  (∀ d ∈ D, d ≠ ∅) ∧
  (∀ d1 d2 ∈ D, d1 ≠ d2 → d1 ∩ d2 = ∅) ∧
  (⋃₀ D = Ω)

theorem cardinality_of_sigma_algebra
  (Ω : Type)
  (D : Set (Set Ω))
  (hD : is_countable_partition D Ω) :
  ∃ c : cardinal, c = cardinal.continuum ∧ cardinal.mk (MeasurableSpace σ(D)) = c :=
sorry

end CardinalityOfSigmaAlgebra

end cardinality_of_sigma_algebra_l584_584043


namespace calculate_average_speed_l584_584738

-- Definitions for the distance and speed of each segment
def d1 (x : ℝ) := x
def d2 (x : ℝ) := 1.5 * x
def d3 (x : ℝ) := 2 * x
def d4 (x : ℝ) := 2.5 * x
def d5 (x : ℝ) := 0.5 * x

def s1 : ℝ := 40
def s2 : ℝ := 30
def s3 : ℝ := 20
def s4 : ℝ := 25
def s5 : ℝ := 50

-- Sum of distances
def total_distance (x : ℝ) := d1 x + d2 x + d3 x + d4 x + d5 x

-- Time taken for each segment
def t1 (x : ℝ) := (d1 x) / s1
def t2 (x : ℝ) := (d2 x) / s2
def t3 (x : ℝ) := (d3 x) / s3
def t4 (x : ℝ) := (d4 x) / s4
def t5 (x : ℝ) := (d5 x) / s5

-- Sum of times
def total_time (x : ℝ) := t1 x + t2 x + t3 x + t4 x + t5 x

-- Average speed
def average_speed (x : ℝ) := total_distance x / total_time x

-- The goal is to prove that with the given conditions, the average speed equals approximately 27.66 kmph
theorem calculate_average_speed (x : ℝ) : abs (average_speed x - 27.66) < 0.01 :=
  by 
    -- Here, we would provide the steps to validate the calculated average speed
    sorry  -- Proof placeholder

end calculate_average_speed_l584_584738


namespace simplify_sqrt_72_plus_sqrt_32_l584_584088

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l584_584088


namespace original_repayment_plan_l584_584433

def gary_loan : ℕ := 6000
def repayment_period_2_years : ℕ := 2
def extra_payment_monthly : ℕ := 150
def monthly_payment_2_years : ℕ := gary_loan / (repayment_period_2_years * 12)

theorem original_repayment_plan : 
  let original_monthly_payment := monthly_payment_2_years - extra_payment_monthly in
  let total_months := gary_loan / original_monthly_payment in
  total_months / 12 = 5 :=
by
  -- Proof to be filled in
  sorry

end original_repayment_plan_l584_584433


namespace children_in_family_l584_584296

theorem children_in_family
  (adult_tickets : ℝ)
  (elderly_discount : ℝ)
  (adult_cost : ℝ)
  (children_discount : ℝ)
  (total_money : ℝ)
  (change : ℝ)
  (num_adults : ℕ)
  (num_elderly : ℕ)
  (num_remaining_money : ℝ) :
  (adult_tickets = 18) →
  (elderly_discount = 8) →
  (adult_cost = 15) →
  (children_discount = 6) →
  (total_money = 270) →
  (change = 10) →
  (num_adults = 5) →
  (num_elderly = 3) →
  (num_remaining_money = 260 - 15 * 5 - (adult_tickets - elderly_discount) * 3) →
  nat.floor (num_remaining_money / (adult_cost - children_discount)) = 17 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end children_in_family_l584_584296


namespace log2_at_one_is_zero_l584_584741

theorem log2_at_one_is_zero : (Real.log 2 2) = 0 := by
  sorry

end log2_at_one_is_zero_l584_584741


namespace plumber_total_spent_l584_584305

theorem plumber_total_spent : 
  ∀ (copper_pipe_length plastic_pipe_extra_length cost_per_meter : ℕ), 
    copper_pipe_length = 10 →
    plastic_pipe_extra_length = 5 →
    cost_per_meter = 4 →
    let plastic_pipe_length := copper_pipe_length + plastic_pipe_extra_length in
    let cost_copper := copper_pipe_length * cost_per_meter in
    let cost_plastic := plastic_pipe_length * cost_per_meter in
    cost_copper + cost_plastic = 100 :=
by
  intros copper_pipe_length plastic_pipe_extra_length cost_per_meter 
         h_copper h_plastic_extra h_cost
  let plastic_pipe_length := copper_pipe_length + plastic_pipe_extra_length
  let cost_copper := copper_pipe_length * cost_per_meter
  let cost_plastic := plastic_pipe_length * cost_per_meter
  have h1 : cost_copper = 10 * 4 := by rw [h_copper, h_cost]
  have h2 : plastic_pipe_length = 10 + 5 := by rw [h_copper, h_plastic_extra]
  have h3 : cost_plastic = plastic_pipe_length * 4 := by rw [h_cost]
  have h4 : cost_plastic = (10 + 5) * 4 := by rw [h2]
  have h5 : cost_plastic = 60 := by simp [h4]
  have h6 : cost_copper + cost_plastic = 40 + 60 := by rw [h1, h5]
  have : 40 + 60 = 100 := by norm_num
  rw this at h6
  rw h6
  refl

end plumber_total_spent_l584_584305


namespace convex_15gon_smallest_angle_arith_seq_l584_584988

noncomputable def smallest_angle (n : ℕ) (avg_angle d : ℕ) : ℕ :=
156 - 7 * d

theorem convex_15gon_smallest_angle_arith_seq :
  let n := 15 in
  ∀ (a d : ℕ), 
  (a = 156 - 7 * d) ∧
  (avg_angle = (13 * 180) / n) ∧
  (forall i : ℕ, 1 ≤ i ∧ i < n → d < 24 / 7) →
  a = 135 :=
sorry

end convex_15gon_smallest_angle_arith_seq_l584_584988


namespace simplify_sum_l584_584667

theorem simplify_sum : 
  (-1: ℤ)^(2010) + (-1: ℤ)^(2011) + (1: ℤ)^(2012) + (-1: ℤ)^(2013) = -2 := by
  sorry

end simplify_sum_l584_584667


namespace max_quotient_l584_584851

theorem max_quotient (x y : ℝ) (hx : 100 ≤ x ∧ x ≤ 300) (hy : 900 ≤ y ∧ y ≤ 1800) : 
  (∀ x y, (100 ≤ x ∧ x ≤ 300) ∧ (900 ≤ y ∧ y ≤ 1800) → y / x ≤ 18) ∧ 
  (∃ x y, (100 ≤ x ∧ x ≤ 300) ∧ (900 ≤ y ∧ y ≤ 1800) ∧ y / x = 18) :=
by
  sorry

end max_quotient_l584_584851


namespace scooter_gain_percent_l584_584601

def initial_cost : ℝ := 900
def first_repair_cost : ℝ := 150
def second_repair_cost : ℝ := 75
def third_repair_cost : ℝ := 225
def selling_price : ℝ := 1800

theorem scooter_gain_percent :
  let total_cost := initial_cost + first_repair_cost + second_repair_cost + third_repair_cost
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 33.33 :=
by
  sorry

end scooter_gain_percent_l584_584601


namespace proof_problem_l584_584494

noncomputable def c : ℝ := Real.log 16 / Real.log 3
noncomputable def d : ℝ := Real.log 81 / Real.log 4

theorem proof_problem : 9^(c / d) + 4^(d / c) = 528 := by
  sorry

end proof_problem_l584_584494


namespace kim_hours_of_classes_per_day_l584_584915

-- Definitions based on conditions
def original_classes : Nat := 4
def hours_per_class : Nat := 2
def dropped_classes : Nat := 1

-- Prove that Kim now has 6 hours of classes per day
theorem kim_hours_of_classes_per_day : (original_classes - dropped_classes) * hours_per_class = 6 := by
  sorry

end kim_hours_of_classes_per_day_l584_584915


namespace smallest_angle_of_convex_15_gon_arithmetic_sequence_l584_584997

theorem smallest_angle_of_convex_15_gon_arithmetic_sequence :
  ∃ (a d : ℕ), (∀ k : ℕ, k < 15 → (let angle := a + k * d in angle < 180)) ∧
  (∀ i j : ℕ, i < j → i < 15 → j < 15 → (a + i * d) < (a + j * d)) ∧
  (let sequence_sum := 15 * a + d * 7 * 14 in sequence_sum = 2340) ∧
  (d = 3) ∧
  (a = 135) :=
by
  sorry

end smallest_angle_of_convex_15_gon_arithmetic_sequence_l584_584997


namespace complex_number_solution_l584_584464

theorem complex_number_solution (z : ℂ) (h : (z - complex.I) * complex.I = 2 + complex.I) : 
  z = 1 - complex.I :=
sorry

end complex_number_solution_l584_584464


namespace sqrt_sum_l584_584112

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l584_584112


namespace divides_iff_l584_584271

-- Definitions of the sequences a_k and a_l
def a_k (k : ℕ) : ℕ := (List.range k).foldr (λ _ acc, acc * 10 + 1) 0

theorem divides_iff (k l : ℕ) (hk : k ≥ 1) : (a_k k ∣ a_k l) ↔ (k ∣ l) :=
begin
  sorry
end

end divides_iff_l584_584271


namespace sin_range_l584_584761

theorem sin_range :
  ∀ x, (-Real.pi / 4 ≤ x ∧ x ≤ 3 * Real.pi / 4) → (∃ y, y = Real.sin x ∧ -Real.sqrt 2 / 2 ≤ y ∧ y ≤ 1) := by
  sorry

end sin_range_l584_584761


namespace largest_prime_factor_of_sum_cyclic_sequence_divisible_by_101_l584_584377

-- Define a four-digit integer sequence with the given cyclic property
def is_cyclic_sequence (s : List ℕ) : Prop :=
  ∀ i, i < s.length →
    (s[i] % 10 == (s[(i+1) % s.length] / 1000)) ∧
    ((s[i] / 10) % 10 == ((s[(i+1) % s.length] / 100) % 10)) ∧
    ((s[i] / 100) % 10 == ((s[(i+1) % s.length] / 10) % 10))

-- Define that the sum of such sequence is divisible by 101 and has 101 as its largest prime factor
theorem largest_prime_factor_of_sum_cyclic_sequence_divisible_by_101 (s : List ℕ) (h : is_cyclic_sequence s) :
  ∃ p, p.prime ∧ p ∣ (s.sum) ∧ p = 101 :=
sorry

end largest_prime_factor_of_sum_cyclic_sequence_divisible_by_101_l584_584377


namespace max_halls_visitable_max_triangles_in_chain_l584_584700

-- Definition of the problem conditions
def castle_side_length : ℝ := 100
def num_halls : ℕ := 100
def hall_side_length : ℝ := 10
def max_visitable_halls : ℕ := 91

-- Theorem statements
theorem max_halls_visitable (S : ℝ) (n : ℕ) (H : ℝ) :
  S = 100 ∧ n = 100 ∧ H = 10 → max_visitable_halls = 91 :=
by sorry

-- Definitions for subdividing an equilateral triangle and the chain of triangles
def side_divisions (k : ℕ) : ℕ := k
def total_smaller_triangles (k : ℕ) : ℕ := k^2
def max_chain_length (k : ℕ) : ℕ := k^2 - k + 1

-- Theorem statements
theorem max_triangles_in_chain (k : ℕ) :
  max_chain_length k = k^2 - k + 1 :=
by sorry

end max_halls_visitable_max_triangles_in_chain_l584_584700


namespace find_f_prime_1_l584_584942

noncomputable def f (x : ℝ) (f_prime_1 : ℝ) : ℝ := x^2 + 2*x*f_prime_1 + 3

theorem find_f_prime_1 : ∃ f_prime_1 : ℝ, ∀ x : ℝ, f_prime_1 = -2 ∧ (derivative (λ x, f x f_prime_1) 1 = 2*x + 2*f_prime_1) := 
  sorry

end find_f_prime_1_l584_584942


namespace range_q_l584_584933

noncomputable def q (x : ℝ) : ℝ :=
  if (⌊x⌋ : ℝ) % 2 = 0 then x^2 + 1
  else 
    let y := if h : 2 ≤ x then (Nat.minFac (⌊x⌋.toNat)) else 2 in
    q y + (x + 1 - ⌊x⌋)

theorem range_q : set.range (λ x, q x) = set.Icc 5 197 := 
sorry

end range_q_l584_584933


namespace arc_length_PQ_radius_24_l584_584923

noncomputable def circumference_circle (r : ℝ) : ℝ := 2 * Real.pi * r

def minor_arc_length (C : ℝ) (angle : ℝ) : ℝ := C * (angle / 360)

theorem arc_length_PQ_radius_24 (P Q R : ℝ) (r : ℝ) (angle : ℝ) 
  (h1 : r = 24)
  (h2 : angle = 120) :
  minor_arc_length (circumference_circle r) (360 - angle) = 32 * Real.pi :=
by
  sorry

end arc_length_PQ_radius_24_l584_584923


namespace sum_of_powers_equiv_l584_584963

theorem sum_of_powers_equiv (k : ℤ) 
  (b : ℕ → ℕ) 
  (hk : 2 ≤ k ∧ k ≤ 100) 
  (hpos : ∀ i, 2 ≤ i → i ≤ 101 → 0 < b i) :
  (∑ i in finset.Icc 2 k, b i ^ i) = (∑ i in finset.Icc k 101, b i ^ i) := 
sorry

end sum_of_powers_equiv_l584_584963


namespace jonathan_weekly_caloric_deficit_l584_584908

def jonathan_caloric_deficit 
  (daily_calories : ℕ) (extra_calories_saturday : ℕ) (daily_burn : ℕ) 
  (days : ℕ) (saturday : ℕ) : ℕ :=
  let total_consumed := daily_calories * days + (daily_calories + extra_calories_saturday) * saturday in
  let total_burned := daily_burn * (days + saturday) in
  total_burned - total_consumed

theorem jonathan_weekly_caloric_deficit :
  jonathan_caloric_deficit 2500 1000 3000 6 1 = 2500 :=
by
  sorry

end jonathan_weekly_caloric_deficit_l584_584908


namespace find_x_to_print_800_leaflets_in_3_minutes_l584_584309

theorem find_x_to_print_800_leaflets_in_3_minutes (x : ℝ) :
  (800 / 12 + 800 / x = 800 / 3) → (1 / 12 + 1 / x = 1 / 3) :=
by
  intro h
  have h1 : 800 / 12 = 200 / 3 := by norm_num
  have h2 : 800 / 3 = 800 / 3 := by norm_num
  sorry

end find_x_to_print_800_leaflets_in_3_minutes_l584_584309


namespace monotonicity_and_min_bound_l584_584824

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (a / 2) * x^2 - (a^2 + 1) * x

noncomputable def g (a : ℝ) : ℝ :=
  f a a

theorem monotonicity_and_min_bound (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, 0 < x → a ≤ 1 → f' a x ≤ 0) ∧
  (∀ x : ℝ, 0 < x → 0 < a ∧ a < 1 → (x < 1/a ∧ f' a x < 0) ∨ (a < x ∧ f' a x > 0)) ∧
  (∀ x : ℝ, 0 < x → a = 1 → f' a x ≥ 0) ∧
  (∀ x : ℝ, 0 < x → a > 1 → 
    ((1/a < x ∧ x < a ∧ f' a x < 0) ∨ (0 < x ∧ x < 1/a ∧ f' a x > 0) ∨ (a < x ∧ f' a x > 0))) →
  (a > 1 → ∃ b : ℝ, Integer.floor b = 0 ∧ ∀ a : ℝ, a > 1 → g a < b - 1/4 * (2 * a^3 - 2 * a^2 + 5 * a)) :=
sorry

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ :=
  a / x + a * x - (a^2 + 1)

end monotonicity_and_min_bound_l584_584824


namespace calculate_candy_bars_l584_584903

theorem calculate_candy_bars
  (soft_drink_calories : ℕ)
  (percent_added_sugar : ℕ)
  (recommended_intake : ℕ)
  (exceeded_percentage : ℕ)
  (candy_bar_calories : ℕ)
  (soft_drink_calories = 2500)
  (percent_added_sugar = 5)
  (recommended_intake = 150)
  (exceeded_percentage = 100)
  (candy_bar_calories = 25) :
  let added_sugar_from_drink := soft_drink_calories * percent_added_sugar / 100,
      exceeded_amount := recommended_intake * exceeded_percentage / 100,
      total_added_sugar := recommended_intake + exceeded_amount,
      added_sugar_from_candy_bars := total_added_sugar - added_sugar_from_drink in
  added_sugar_from_candy_bars / candy_bar_calories = 7 :=
by 
  -- proof
  sorry

end calculate_candy_bars_l584_584903


namespace sqrt_36_eq_6_l584_584640

theorem sqrt_36_eq_6 : Real.sqrt 36 = 6 := by
  sorry

end sqrt_36_eq_6_l584_584640


namespace probability_heads_twice_and_die_three_l584_584849

-- Define a fair coin flip and regular six-sided die roll
def fair_coin : finset (bool) := {tt, ff}
def six_sided_die : finset (ℕ) := {1, 2, 3, 4, 5, 6}

-- Define the probability calculation function
noncomputable theory
open_locale classical

def probability_of_heads_times_two_and_die_shows_three : ℚ :=
  (1 / (fair_coin.card * fair_coin.card)) * (1 / six_sided_die.card)

-- Proof statement for the problem
theorem probability_heads_twice_and_die_three :
  probability_of_heads_times_two_and_die_shows_three = 1 / 24 := 
by {
  -- Calculation that follows directly from conditions
  have h_coin_card : fair_coin.card = 2 := by simp [fair_coin],
  have h_die_card : six_sided_die.card = 6 := by simp [six_sided_die],
  simp [probability_of_heads_times_two_and_die_shows_three, h_coin_card, h_die_card],
  norm_num,
}

end probability_heads_twice_and_die_three_l584_584849


namespace sequence_formula_M_100_sum_l584_584797

open Nat

def sequence (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 2
  | n + 3 => n + 1

def sum_first_n (f : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range (n + 1)).map f |>.sum

noncomputable def s_n (n : ℕ) : ℕ := 
  sum_first_n sequence n

axiom seq_recursive_rel (n : ℕ) (h : n > 0) :
  (2 * n + 3) * s_n (n + 1) = (n + 2) * s_n n + (n + 1) * s_n (n + 2)

theorem sequence_formula (n : ℕ) :
  sequence n = n := 
sorry

def T (n : ℕ) : ℕ := 
  Nat.logBase 2 (sequence (n+1) / sequence n)

def c (n : ℕ) : ℕ := 
  T n

theorem M_100_sum :
  (List.range 100).map c |>.sum = 486 :=
sorry

end sequence_formula_M_100_sum_l584_584797


namespace angle_between_a_c_at_pi_over_six_min_value_of_f_in_interval_l584_584484

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.cos x)
def vec_c : ℝ × ℝ := (-1, 0)
noncomputable def f (x : ℝ) : ℝ := 2 * (vec_a x).1 * (vec_b x).1 + 2 * (vec_a x).2 * (vec_b x).2 + 1

theorem angle_between_a_c_at_pi_over_six : 
  let x := Real.pi / 6
  (Real.acos (((vec_a x).1 * vec_c.1 + (vec_a x).2 * vec_c.2) / 
             (Real.sqrt ((vec_a x).1 ^ 2 + (vec_a x).2 ^ 2) * Real.sqrt (vec_c.1 ^ 2 + vec_c.2 ^ 2)))) = 5 * Real.pi / 6 := 
by 
  sorry

theorem min_value_of_f_in_interval :
  (∀ x, x ∈ Set.Icc (Real.pi / 2) (9 * Real.pi / 8) → f x ≥ -Real.sqrt 2) ∧ 
  (∃ x, x ∈ Set.Icc (Real.pi / 2) (9 * Real.pi / 8) ∧ f x = -Real.sqrt 2) :=
by 
  sorry

end angle_between_a_c_at_pi_over_six_min_value_of_f_in_interval_l584_584484


namespace solve_for_x_l584_584408

theorem solve_for_x : ∀ x : ℝ, (x^2 + 85 = (x - 17)^2) → x = 6 :=
by
  intro x
  assume h : x^2 + 85 = (x - 17)^2
  sorry

end solve_for_x_l584_584408


namespace median_of_81_consecutive_integers_l584_584178

theorem median_of_81_consecutive_integers (ints : ℕ → ℤ) 
  (h_consecutive: ∀ n : ℕ, ints (n+1) = ints n + 1) 
  (h_sum: (∑ i in finset.range 81, ints i) = 9^5) : 
  (ints 40) = 9^3 := 
sorry

end median_of_81_consecutive_integers_l584_584178


namespace probability_between_C_and_D_is_five_eighths_l584_584057

noncomputable def AB : ℝ := 1
def AD : ℝ := AB / 4
def BC : ℝ := AB / 8
def pos_C : ℝ := AB - BC
def pos_D : ℝ := AD
def CD : ℝ := pos_C - pos_D

theorem probability_between_C_and_D_is_five_eighths : CD / AB = 5 / 8 :=
by
  simp [AB, AD, BC, pos_C, pos_D, CD]
  sorry

end probability_between_C_and_D_is_five_eighths_l584_584057


namespace perimeter_triangle_pqr_l584_584231

theorem perimeter_triangle_pqr (PQ PR QR QJ : ℕ) (h1 : PQ = PR) (h2 : QJ = 10) :
  ∃ PQR', PQR' = 198 ∧ triangle PQR PQ PR QR := sorry

end perimeter_triangle_pqr_l584_584231


namespace Sharmila_hourly_wage_l584_584082

def Sharmila_hours_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 10
  else if day = "Tuesday" ∨ day = "Thursday" then 8
  else 0

def weekly_total_hours : ℕ :=
  Sharmila_hours_per_day "Monday" + Sharmila_hours_per_day "Tuesday" +
  Sharmila_hours_per_day "Wednesday" + Sharmila_hours_per_day "Thursday" +
  Sharmila_hours_per_day "Friday"

def weekly_earnings : ℤ := 460

def hourly_wage : ℚ :=
  weekly_earnings / weekly_total_hours

theorem Sharmila_hourly_wage :
  hourly_wage = (10 : ℚ) :=
by
  -- proof skipped
  sorry

end Sharmila_hourly_wage_l584_584082


namespace convert_base8_to_base7_l584_584384

theorem convert_base8_to_base7 : (536%8).toBase 7 = 1010%7 :=
by
  sorry

end convert_base8_to_base7_l584_584384


namespace cyclic_quad_eq_l584_584893

open BigOperators

variable {R : Type*} [Field R]

variables (A B C D I E : R^2)
variable (circle : set (R^2))
variable [InscribedInCircle ABCD : InscribedInCircle circle]

-- Conditions
def cyclic_quad (ABCD : R^2) : Prop :=
  ∀ (P Q : R^2), P ∈ ABCD → Q ∈ ABCD → (P - Q).dot (P - Q) = 0

def perp (u v : R^2) : Prop := (u.dot v = 0)

def incenter (I : R^2) (A B D : R^2) : Prop :=
  let AI := dist A I
  let BI := dist B I
  let DI := dist D I
  AI = BI ∧ AI = DI 

def ie_perp_bd (I E B D : R^2) : Prop :=
  perp (I - E) (B - D)

def ia_eq_ic (I A C : R^2) : Prop :=
  dist I A = dist I C 

-- Proof problem
theorem cyclic_quad_eq (ABCD : R^2) (I E C : R^2) (inscribed : InscribedInCircle circle)
  (h1 : cyclic_quad ABCD)
  (h2 : perp (A - C) (B - D))
  (h3 : incenter I A B D)
  (h4 : ie_perp_bd I E B D)
  (h5 : ia_eq_ic I A C) :
  dist E C = dist E I := sorry

end cyclic_quad_eq_l584_584893


namespace dice_probability_not_all_same_no_sequence_l584_584664

theorem dice_probability_not_all_same_no_sequence :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let sequence_outcomes := 2 * nat.factorial 5
  let valid_outcomes := total_outcomes - same_number_outcomes - sequence_outcomes
  let probability := (valid_outcomes : ℚ) / total_outcomes
  probability = 7530 / 7776 :=
by
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let sequence_outcomes := 2 * nat.factorial 5
  let valid_outcomes := total_outcomes - same_number_outcomes - sequence_outcomes
  let probability := (valid_outcomes : ℚ) / total_outcomes
  sorry

end dice_probability_not_all_same_no_sequence_l584_584664


namespace sum_coefficients_l584_584546

theorem sum_coefficients (n : ℕ) (a : ℕ → ℕ)
  (h : (∑ i in Finset.range (n + 1), (1 + (1:ℕ)^(i + 1 - 1))) = ∑ i in Finset.range (n + 1), a i)
  (a_n_minus_one : a (n - 1) = 2009) :
  ∑ i in Finset.range (n + 1), a i = 2^2009 - 2 := 
sorry

end sum_coefficients_l584_584546


namespace range_of_h_in_interval_intervals_of_monotonicity_and_extremum_range_of_k_if_H_has_two_real_roots_l584_584826

def f (x : ℝ) : ℝ := 1 + 4 / x

def g (x : ℝ) : ℝ := Real.log x / Real.log 2

def h (x : ℝ) : ℝ := g x - f x

def H (x : ℝ) : ℝ := min (f x) (g x)

theorem range_of_h_in_interval :
  ∀ x, 2 ≤ x ∧ x ≤ 4 → -2 ≤ h x ∧ h x ≤ 0 := by
  sorry

theorem intervals_of_monotonicity_and_extremum :
  (∀ x, 0 < x ∧ x ≤ 4 → H x = g x) ∧
  (∀ x, 4 < x → H x = f x) ∧
  (∀ x, 0 < x ∧ x ≠ 4 → (H x ≤ 2)) := by
  sorry

theorem range_of_k_if_H_has_two_real_roots :
  ∀ k, (∃ x1 x2, x1 ≠ x2 ∧ H x1 = k ∧ H x2 = k) → 1 < k ∧ k < 2 := by
  sorry

end range_of_h_in_interval_intervals_of_monotonicity_and_extremum_range_of_k_if_H_has_two_real_roots_l584_584826


namespace intersection_midpoint_l584_584542

noncomputable theory

variables {A B C L F M : Point}
variables {l1 l2 : Line}

def midpoint (A B M : Point) : Prop :=
  dist A M = dist M B

theorem intersection_midpoint 
  (hAC_2AB : dist A C = 2 * dist A B)
  (hAL_bisector : inside_angle_bisector (angle A B C) L)
  (hl1_parallel : is_parallel l1 (line_through A B))
  (hl2_perpendicular : is_perpendicular l2 (line_through A L))
  (hF_intersection : F ∈ l1 ∧ F ∈ l2)
  (hL_on_AC : L ∈ line_through A C) :
  M = intersection (line_through F L) (line_through A C) →
  midpoint A C M := sorry

end intersection_midpoint_l584_584542


namespace line_sqrt10_away_from_l1_reflected_ray_eq_l584_584458

open Real

section
variables {x y: ℝ}
variables (l1 := 3 * x - y + 7 = 0) (l2 := 2 * x + y + 3 = 0)
variables (M := (-2, 1): ℝ × ℝ) (N := (1, 0): ℝ × ℝ)

-- Define the function to check if two lines are parallel
def are_parallel (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 / b1 = a2 / b2

-- Problem part (1): Line equations that are sqrt(10) units away from l1
theorem line_sqrt10_away_from_l1 : 
  are_parallel 3 (-1) 3 (-1) ∧ (∀ (C: ℝ), (|C-7| / sqrt(10) = sqrt(10) ↔ C = -3 ∨ C = 17)) :=
sorry

-- Problem part (2): Equation of the reflected ray
theorem reflected_ray_eq : 
  ∃ (k: ℝ), k = -1/3 ∧ ∀ (m: ℝ), m = 1/3 ∧ eqn = (x - 3 * y - 1) :=
sorry

end line_sqrt10_away_from_l1_reflected_ray_eq_l584_584458


namespace correct_conclusion_l584_584679

-- Define the terms and conditions
def is_quadratic_trinomial (expr : ℕ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ expr 0 = c ∧ expr 1 = b ∧ expr 2 = a

def like_terms (t1 t2 : ℕ → ℕ → ℝ) : Prop :=
  ∀ n m, (t1 n m ≠ 0 ∧ t2 n m ≠ 0) → (n = m)

def constant_term (expr : ℕ → ℝ) : ℝ :=
  expr 0

def coefficient (monomial : ℕ → ℕ → ℝ) : ℝ :=
  monomial 2 1 / monomial 0 1

def degree (monomial : ℕ → ℕ → ℝ) : ℕ :=
  2 + 1

-- Define the given expressions
def expr1 (n : ℕ) : ℝ :=
  if n = 0 then -3 else if n = 1 then 4 else if n = 2 then Real.pi else 0

def term1 (n m : ℕ) : ℝ :=
  if n = 2 ∧ m = 1 then 3 else 0

def term2 (n m : ℕ) : ℝ :=
  if n = 1 ∧ m = 2 then -2 else 0

def monomial (n m : ℕ) : ℝ :=
  if n = 2 ∧ m = 1 then -3/5 else 0

-- The main proof statement
theorem correct_conclusion : 
  is_quadratic_trinomial expr1 ∧
  ¬ like_terms term1 term2 ∧
  constant_term expr1 ≠ 3 ∧
  coefficient monomial = -3/5 ∧ 
  degree monomial ≠ 2 
  → true :=
by
  sorry

end correct_conclusion_l584_584679


namespace volume_of_sphere_is_correct_l584_584708

noncomputable def volume_of_sphere_circumscribing_cube (V_cube : ℝ) 
  (hV_cube : V_cube = 8) : ℝ :=
let a := real.cbrt V_cube in
let R := real.sqrt 3 in
(4 * real.sqrt 3 * real.pi)

theorem volume_of_sphere_is_correct
  (V_cube : ℝ)
  (hV_cube : V_cube = 8)
  : volume_of_sphere_circumscribing_cube V_cube hV_cube = 4 * real.sqrt 3 * real.pi :=
by {
  rw [volume_of_sphere_circumscribing_cube],
  sorry
}

end volume_of_sphere_is_correct_l584_584708


namespace cupcakes_frosted_in_10_minutes_l584_584341

theorem cupcakes_frosted_in_10_minutes (r1 r2 time : ℝ) (cagney_rate lacey_rate : r1 = 1 / 15 ∧ r2 = 1 / 25)
  (time_in_seconds : time = 600) :
  (1 / ((1 / r1) + (1 / r2)) * time) = 64 := by
  sorry

end cupcakes_frosted_in_10_minutes_l584_584341


namespace min_value_2_l584_584449

noncomputable def min_value (a b : ℝ) : ℝ :=
  1 / a + 1 / (b + 1)

theorem min_value_2 {a b : ℝ} (h1 : a > 0) (h2 : b > -1) (h3 : a + b = 1) : min_value a b = 2 :=
by
  sorry

end min_value_2_l584_584449


namespace median_of_81_consecutive_integers_l584_584190

theorem median_of_81_consecutive_integers (sum : ℕ) (h₁ : sum = 9^5) : 
  let mean := sum / 81 in mean = 729 :=
by
  have h₂ : sum = 59049 := by
    rw h₁
    norm_num
  have h₃ : mean = 59049 / 81 := by
    rw h₂
    rfl
  have result : mean = 729 := by
    rw h₃
    norm_num
  exact result

end median_of_81_consecutive_integers_l584_584190


namespace purely_imaginary_roots_iff_l584_584752

theorem purely_imaginary_roots_iff (z : ℂ) (k : ℝ) (i : ℂ) (h_i2 : i^2 = -1) :
  (∀ r : ℂ, (20 * r^2 + 6 * i * r - ↑k = 0) → (∃ b : ℝ, r = b * i)) ↔ (k = 9 / 5) :=
sorry

end purely_imaginary_roots_iff_l584_584752


namespace range_of_a_l584_584814

theorem range_of_a (a : ℝ) (h : a > 0) :
  let C : set (ℝ × ℝ) := {p | (p.1 + 1) ^ 2 + (p.2 - 2) ^ 2 = 2}
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (0, a)
  (∃ M ∈ C, dist M A = real.sqrt 2 * dist M O) ↔ sqrt 3 ≤ a ∧ a ≤ 4 + real.sqrt 19 := 
sorry

end range_of_a_l584_584814


namespace remaining_pencils_l584_584212

/-
Given the initial number of pencils in the drawer and the number of pencils Sally took out,
prove that the number of pencils remaining in the drawer is 5.
-/
def pencils_in_drawer (initial_pencils : ℕ) (pencils_taken : ℕ) : ℕ :=
  initial_pencils - pencils_taken

theorem remaining_pencils : pencils_in_drawer 9 4 = 5 := by
  sorry

end remaining_pencils_l584_584212


namespace no_such_complex_numbers_l584_584405

theorem no_such_complex_numbers :
  ¬ ∃ (a b c : ℂ) (h : ℕ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    ∀ (k l m : ℤ),
      |k| + |l| + |m| ≥ 1996 →
      |1 + k * a + l * b + m * c| > 1 / h :=
sorry

end no_such_complex_numbers_l584_584405


namespace calculate_expression_l584_584347

theorem calculate_expression : 
  -1^4 - (1 - 0.5) * (2 - (-3)^2) = 5 / 2 :=
by
  sorry

end calculate_expression_l584_584347


namespace sqrt_sum_simplify_l584_584121

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l584_584121


namespace part_a_part_b_l584_584724

variables (p : ℝ) (h : p ≥ 0 ∧ p ≤ 1) -- Probability p is between 0 and 1 inclusive

-- Part (a)
theorem part_a (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (5.choose 3) * p^3 * (1 - p)^2 = 10 * p^3 * (1 - p)^2 := by 
sorry

-- Part (b)
theorem part_b (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  10 * p - p^10 = (10 * p - p^10) := by 
sorry

end part_a_part_b_l584_584724


namespace cos_inequality_l584_584278

theorem cos_inequality (A B : ℝ) (n : ℕ) (hn : n > 1) (hneq : cos A ≠ cos B) :
  |cos (n * A) * cos B - cos A * cos (n * B)| < (n^2 - 1) * |cos A - cos B| :=
sorry

end cos_inequality_l584_584278


namespace decreasing_function_iff_a_range_l584_584505

noncomputable def f (a x : ℝ) : ℝ := (1 - 2 * a) ^ x

theorem decreasing_function_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ 0 < a ∧ a < 1/2 :=
by
  sorry

end decreasing_function_iff_a_range_l584_584505


namespace sum_odd_coefficients_l584_584632

noncomputable def sum_odd_coeffs (p : Polynomial ℝ) : ℝ :=
  p.sum (λ n a, if odd n then a else 0)

theorem sum_odd_coefficients {p : Polynomial ℝ} : 
  (p.eval (-1) = 7) → 
  (p.eval 1 = 5) → 
  sum_odd_coeffs p = -1 := 
  by 
    sorry

end sum_odd_coefficients_l584_584632


namespace tony_spending_per_sq_ft_l584_584222

noncomputable def total_sq_ft_master_bath_living : ℕ := 500 + 400
noncomputable def total_monthly_costs : ℕ := 3000 + 250 + 100
noncomputable def cost_per_sq_ft (costs sq_ft : ℕ) := (costs : ℚ) / sq_ft

theorem tony_spending_per_sq_ft :
  cost_per_sq_ft total_monthly_costs total_sq_ft_master_bath_living ≈ 3.72 :=
sorry

end tony_spending_per_sq_ft_l584_584222


namespace pq_condition_l584_584932

theorem pq_condition (p q : ℝ) (h1 : p * q = 16) (h2 : p + q = 10) : (p - q)^2 = 36 :=
by
  sorry

end pq_condition_l584_584932


namespace point_M_polar_coordinates_l584_584978

noncomputable def cartesianToPolar (x y : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan2 y x
  (ρ, θ)

theorem point_M_polar_coordinates :
  cartesianToPolar (Real.sqrt 3) (-1) = (2, 11 * Real.pi / 6) := sorry

end point_M_polar_coordinates_l584_584978


namespace appetizer_cost_per_person_l584_584430

def chip_cost : ℝ := 3 * 1.00
def creme_fraiche_cost : ℝ := 5.00
def caviar_cost : ℝ := 73.00
def total_cost : ℝ := chip_cost + creme_fraiche_cost + caviar_cost
def number_people : ℝ := 3
def cost_per_person : ℝ := total_cost / number_people

theorem appetizer_cost_per_person : cost_per_person = 27.00 := 
by
  -- proof would go here
  sorry

end appetizer_cost_per_person_l584_584430


namespace probability_three_unused_rockets_expected_targets_hit_l584_584726

section RocketArtillery

variables (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1)

-- Probability that exactly three unused rockets remain after firing at five targets
theorem probability_three_unused_rockets (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let prob := 10 * (p ^ 3) * ((1 - p) ^ 2) in
  prob = 10 * (p ^ 3) * ((1 - p) ^ 2) := 
by
  sorry

-- Expected number of targets hit if there are nine targets
theorem expected_targets_hit (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let expected_hits := 10 * p - (p ^ 10) in
  expected_hits = 10 * p - (p ^ 10) := 
by
  sorry

end RocketArtillery

end probability_three_unused_rockets_expected_targets_hit_l584_584726


namespace sin_max_value_l584_584166

noncomputable def sin_max_in_triangle (A B C : ℝ) (hA : 0 < A) (hA_lt_pi : A < π) (hB : 0 < B) (hB_lt_pi : B < π) (hC : 0 < C) (hC_lt_pi : C < π) (h_sum_ABC : A + B + C = π) : ℝ :=
begin
  sorry,
end

theorem sin_max_value (A B C : ℝ) (hA : 0 < A) (hA_lt_pi : A < π) (hB : 0 < B) (hB_lt_pi : B < π) (hC : 0 < C) (hC_lt_pi : C < π) (h_sum_ABC : A + B + C = π) :
  sin A + sin B + sin C ≤ 3 * (Real.sin (π / 3)) :=
begin
  sorry,
end

end sin_max_value_l584_584166


namespace binomial_sum_identity_l584_584777

open Finset

theorem binomial_sum_identity (n : ℕ) (h : 0 < n) :
  ∑ k in range (n + 1), (nat.choose n k) * (2^k) * (nat.choose (n - (π - k) / k) k) = nat.choose (2 * n - 1) n :=
sorry

end binomial_sum_identity_l584_584777


namespace correct_options_l584_584681

theorem correct_options :
  tan (3 * Real.pi / 5) < tan (Real.pi / 5) ∧
  cos (-17 * Real.pi / 4) > cos (-23 * Real.pi / 5) :=
by {
  sorry
}

end correct_options_l584_584681


namespace total_hours_worked_l584_584333

def hours_day1 : ℝ := 2.5
def increment_day2 : ℝ := 0.5
def hours_day2 : ℝ := hours_day1 + increment_day2
def hours_day3 : ℝ := 3.75

theorem total_hours_worked :
  hours_day1 + hours_day2 + hours_day3 = 9.25 :=
sorry

end total_hours_worked_l584_584333


namespace retail_price_increase_l584_584312

theorem retail_price_increase (R W : ℝ) (h1 : 0.80 * R = 1.44000000000000014 * W)
  : ((R - W) / W) * 100 = 80 :=
by 
  sorry

end retail_price_increase_l584_584312


namespace simplify_expression_l584_584973

theorem simplify_expression :
  (1 / (Real.sqrt 8 + Real.sqrt 11) +
   1 / (Real.sqrt 11 + Real.sqrt 14) +
   1 / (Real.sqrt 14 + Real.sqrt 17) +
   1 / (Real.sqrt 17 + Real.sqrt 20) +
   1 / (Real.sqrt 20 + Real.sqrt 23) +
   1 / (Real.sqrt 23 + Real.sqrt 26) +
   1 / (Real.sqrt 26 + Real.sqrt 29) +
   1 / (Real.sqrt 29 + Real.sqrt 32)) = 
  (2 * Real.sqrt 2 / 3) :=
by sorry

end simplify_expression_l584_584973


namespace calculate_b6_l584_584030

noncomputable def a : ℕ → ℚ
| 0       := 3
| (n + 1) := a n ^ 3 / b n

noncomputable def b : ℕ → ℚ
| 0       := 5
| (n + 1) := b n ^ 3 / a n

theorem calculate_b6 : b 6 = (5 : ℚ) ^ 377 / (3 : ℚ) ^ 376 := 
sorry

end calculate_b6_l584_584030


namespace intersection_point_l584_584585

theorem intersection_point (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) :
  ∃ x y, (y = a * x^2 + b * x + c) ∧ (y = a * x^2 - b * x + c + d) ∧ x = d / (2 * b) ∧ y = a * (d / (2 * b))^2 + (d / 2) + c :=
by
  sorry

end intersection_point_l584_584585


namespace friend_reading_time_l584_584049

def my_reading_time : ℕ := 120  -- It takes me 120 minutes to read the novella

def speed_ratio : ℕ := 3  -- My friend reads three times as fast as I do

theorem friend_reading_time : my_reading_time / speed_ratio = 40 := by
  -- Proof
  sorry

end friend_reading_time_l584_584049


namespace sum_first_10_terms_b_n_l584_584837

theorem sum_first_10_terms_b_n 
  (a : ℕ → ℚ)
  (b : ℕ → ℚ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (n + 1) = 2 * a n) 
  (h3 : ∀ n : ℕ, b n = real.log (a n) / real.log 2) :
  (∑ i in finset.range 10, b (i + 1)) = 55 :=
by
  sorry

end sum_first_10_terms_b_n_l584_584837


namespace inverse_function_l584_584769

noncomputable def f (x : ℝ) (hx : x > 0) : ℝ := log (1 + x) / log 2

noncomputable def f_inv (x : ℝ) (hx : x > 0) : ℝ := 2^x - 1

theorem inverse_function 
  (x : ℝ) (hx : x > 0) :
  f_inv (f x hx) (by have : 0 < log (1 + x) / log 2 := sorry; exact this) = x ∧ 
  (∀ y > 0, f (f_inv y (by have : 0 < y := sorry; exact this)) (by have : 0 < 2^y - 1 := sorry; exact this) = y) :=
sorry

end inverse_function_l584_584769


namespace common_difference_l584_584020

theorem common_difference (a : ℕ → ℝ) (d : ℝ) 
  (ha3 : a 3 = 3) 
  (S7 : ∑ i in Finset.range 7, a i = 14) 
  (h1 : ∀ n, a n = a 0 + n * d)
  (h2 : ∀ n, ∑ i in Finset.range n, a i = n * ((a 0) + d * (n - 1) / 2)) :
  d = -1 := by 
  sorry

end common_difference_l584_584020


namespace number_of_polynomials_l584_584617

-- Define conditions
def is_positive_integer (n : ℤ) : Prop :=
  5 * 151 * n > 0

-- Define the main theorem
theorem number_of_polynomials (n : ℤ) (h : is_positive_integer n) : 
  ∃ k : ℤ, k = ⌊n / 2⌋ + 1 :=
by
  sorry

end number_of_polynomials_l584_584617


namespace problem_equivalence_l584_584594

theorem problem_equivalence (x : ℝ) :
  (∑ n in Finset.range (10 + 1), (2 * (n + 1)) / (x^2 - (n + 1)^2)) = 
  (∑ n in Finset.range (10 + 1), 11 / ((x - (n + 1)) * (x + 10 - (n + 1)))) :=
by
  sorry

end problem_equivalence_l584_584594


namespace simplify_sqrt_sum_l584_584104

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l584_584104


namespace sum_of_angles_x_y_l584_584284

theorem sum_of_angles_x_y :
  let num_arcs := 15
  let angle_per_arc := 360 / num_arcs
  let central_angle_x := 3 * angle_per_arc
  let central_angle_y := 5 * angle_per_arc
  let inscribed_angle (central_angle : ℝ) := central_angle / 2
  let angle_x := inscribed_angle central_angle_x
  let angle_y := inscribed_angle central_angle_y
  angle_x + angle_y = 96 := 
  sorry

end sum_of_angles_x_y_l584_584284


namespace determine_xy_l584_584629

noncomputable section

open Real

def op_defined (ab xy : ℝ × ℝ) : ℝ × ℝ :=
  (ab.1 * xy.1 + ab.2 * xy.2, ab.1 * xy.2 + ab.2 * xy.1)

theorem determine_xy (x y : ℝ) :
  (∀ (a b : ℝ), op_defined (a, b) (x, y) = (a, b)) → (x = 1 ∧ y = 0) :=
by
  sorry

end determine_xy_l584_584629


namespace perimeter_of_8_sided_figure_l584_584663

theorem perimeter_of_8_sided_figure (n : ℕ) (len : ℕ) (h1 : n = 8) (h2 : len = 2) :
  n * len = 16 := by
  sorry

end perimeter_of_8_sided_figure_l584_584663


namespace collinearity_and_direction_vector_l584_584380

noncomputable def point1 : ℝ × ℝ := (-3, 2)
noncomputable def point2 : ℝ × ℝ := (2, -3)
noncomputable def point3 : ℝ × ℝ := (4, -5)

def direction_vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)

def line_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let v1 := direction_vector p1 p2
  let v2 := direction_vector p1 p3
  v1.1 * v2.2 = v1.2 * v2.1

theorem collinearity_and_direction_vector :
  line_collinear point1 point2 point3 ∧ 
  ∃ b : ℝ, (∃ k : ℝ, (k * direction_vector point1 point2 = (b, -1))) ∧ b = 1 := 
by
  sorry

end collinearity_and_direction_vector_l584_584380


namespace rational_operations_l584_584967

theorem rational_operations (a b n p : ℤ) (hb : b ≠ 0) (hp : p ≠ 0) (hn : n ≠ 0) :
  (∃ q : ℚ, q = (a : ℚ) / b + (n : ℚ) / p) ∧
  (∃ q : ℚ, q = (a : ℚ) / b - (n : ℚ) / p) ∧
  (∃ q : ℚ, q = (a : ℚ) / b * (n : ℚ) / p) ∧
  (∃ q : ℚ, q = (a : ℚ) / b / ((n : ℚ) / p)) :=
by 
  sorry

end rational_operations_l584_584967


namespace dollar_eval_l584_584398

def dollar (a b : ℝ) : ℝ := (a^2 - b^2)^2

theorem dollar_eval (x : ℝ) : dollar (x^3 + x) (x - x^3) = 16 * x^8 :=
by
  sorry

end dollar_eval_l584_584398


namespace squared_difference_of_roots_l584_584554

theorem squared_difference_of_roots :
  let (f g : ℚ) := (2, -7/3) in
  (f - g) ^ 2 = 169 / 9 :=
by {
  let f := (2 : ℚ),
  let g := (-7 / 3 : ℚ),
  sorry
}

end squared_difference_of_roots_l584_584554


namespace sum_first_2002_terms_periodic_sequence_l584_584796

def sequence (x : ℕ → ℕ) (a : ℕ) : Prop :=
  (x 1 = 1) ∧ (x 2 = a) ∧ (∀ n ≥ 3, x n = abs (x (n - 1) - x (n - 2)))

theorem sum_first_2002_terms_periodic_sequence
  (a : ℕ) (h1 : a ≥ 0)
  (x : ℕ → ℕ)
  (hs : sequence x a)
  (hx_periodic : ∃ T, ∀ n ≥ T, x (n + T) = x n)
  (hT_min : ∀ T' < T, ∃ n, x (n + T') ≠ x n)
  : (finset.range 2002).sum x = 1335 :=
sorry

end sum_first_2002_terms_periodic_sequence_l584_584796


namespace systematic_sampling_first_group_l584_584734

theorem systematic_sampling_first_group (x : ℕ) (n : ℕ) (k : ℕ) (total_students : ℕ) (sampled_students : ℕ) 
  (interval : ℕ) (group_num : ℕ) (group_val : ℕ) 
  (h1 : total_students = 1000) (h2 : sampled_students = 40) (h3 : interval = total_students / sampled_students)
  (h4 : interval = 25) (h5 : group_num = 18) 
  (h6 : group_val = 443) (h7 : group_val = x + (group_num - 1) * interval) : 
  x = 18 := 
by 
  sorry

end systematic_sampling_first_group_l584_584734


namespace smallest_angle_in_convex_15sided_polygon_l584_584998

def isConvexPolygon (n : ℕ) (angles : Fin n → ℚ) : Prop :=
  ∑ i, angles i = (n - 2) * 180 ∧ ∀ i,  angles i < 180

def arithmeticSequence (angles : Fin 15 → ℚ) : Prop :=
  ∃ a d : ℚ, ∀ i : Fin 15, angles i = a + i * d

def increasingSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i j : Fin 15, i < j → angles i < angles j

def integerSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i : Fin 15, (angles i : ℚ) = angles i

theorem smallest_angle_in_convex_15sided_polygon :
  ∃ (angles : Fin 15 → ℚ),
    isConvexPolygon 15 angles ∧
    arithmeticSequence angles ∧
    increasingSequence angles ∧
    integerSequence angles ∧
    angles 0 = 135 :=
by
  sorry

end smallest_angle_in_convex_15sided_polygon_l584_584998


namespace total_weight_of_peppers_l584_584486

theorem total_weight_of_peppers
  (green_peppers : ℝ) 
  (red_peppers : ℝ)
  (h_green : green_peppers = 0.33)
  (h_red : red_peppers = 0.33) :
  green_peppers + red_peppers = 0.66 := 
by
  sorry

end total_weight_of_peppers_l584_584486


namespace rectangle_ratio_at_least_two_l584_584311

theorem rectangle_ratio_at_least_two (R : Type) [rectangle R]
    (divided_into_right_triangles : ∀ (t1 t2 : right_triangle), 
        adjacent_sides (R t1 t2) → (common_side_leg_or_hypotenuse t1 t2)) :
    (longer_side R) / (shorter_side R) ≥ 2 :=
sorry

end rectangle_ratio_at_least_two_l584_584311


namespace simplify_sqrt_sum_l584_584094

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l584_584094


namespace convert_base8_to_base7_l584_584382

theorem convert_base8_to_base7 : (536%8).toBase 7 = 1010%7 :=
by
  sorry

end convert_base8_to_base7_l584_584382


namespace simplify_sqrt_sum_l584_584096

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l584_584096


namespace farm_own_more_horses_than_cows_after_transaction_l584_584955

theorem farm_own_more_horses_than_cows_after_transaction :
  ∀ (x : Nat), 
    3 * (3 * x - 15) = 5 * (x + 15) →
    75 - 45 = 30 :=
by
  intro x h
  -- This is a placeholder for the proof steps which we skip.
  sorry

end farm_own_more_horses_than_cows_after_transaction_l584_584955


namespace binom_coprime_l584_584563

-- Definitions used in conditions
variable {p a b : ℕ}
variable [fact p.prime]
variable (a_i b_i : ℕ → ℕ)
variable (h : ∀ i, a_i i ≥ b_i i)
variable (hp_ab : ∑ i, a_i i * p^i = a ∧ ∑ i, b_i i * p^i = b)

-- Statement of the problem
theorem binom_coprime {p a b : ℕ} [fact p.prime]
  (a_i b_i : ℕ → ℕ)
  (h : ∀ i, a_i i ≥ b_i i)
  (hp_ab : ∑ i, a_i i * p^i = a ∧ ∑ i, b_i i * p^i = b) :
  Nat.coprime (Nat.choose a b) p :=
sorry

end binom_coprime_l584_584563


namespace banks_investments_count_l584_584950

-- Conditions
def revenue_per_investment_banks := 500
def revenue_per_investment_elizabeth := 900
def number_of_investments_elizabeth := 5
def extra_revenue_elizabeth := 500

-- Total revenue calculations
def total_revenue_elizabeth := number_of_investments_elizabeth * revenue_per_investment_elizabeth
def total_revenue_banks := total_revenue_elizabeth - extra_revenue_elizabeth

-- Number of investments for Mr. Banks
def number_of_investments_banks := total_revenue_banks / revenue_per_investment_banks

theorem banks_investments_count : number_of_investments_banks = 8 := by
  sorry

end banks_investments_count_l584_584950


namespace sin_60_eq_sqrt3_div_2_l584_584353

theorem sin_60_eq_sqrt3_div_2 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  -- proof skipped
  sorry

end sin_60_eq_sqrt3_div_2_l584_584353


namespace jungkook_has_most_apples_l584_584015

theorem jungkook_has_most_apples :
  let jungkook_initial := 6
  let jungkook_more := 3
  let jungkook_total := jungkook_initial + jungkook_more
  let yoongi := 4
  let yuna := 5
  max jungkook_total (max yoongi yuna) = jungkook_total :=
by
  -- Definitions from conditions
  let jungkook_initial := 6
  let jungkook_more := 3
  let jungkook_total := jungkook_initial + jungkook_more
  let yoongi := 4
  let yuna := 5

  -- Statement to be proved
  have h1 : jungkook_total = 9 := by rfl
  have h2 : max yoongi yuna = 5 := by rfl
  have h3 : max jungkook_total 5 = jungkook_total := by rfl
  
  rw [h1, h2, h3]
  exact h1

end jungkook_has_most_apples_l584_584015


namespace period_tan_2x_3_l584_584253

noncomputable def period_of_tan_transformed : Real :=
  let period_tan := Real.pi
  let coeff := 2/3
  (period_tan / coeff : Real)

theorem period_tan_2x_3 : period_of_tan_transformed = 3 * Real.pi / 2 :=
  sorry

end period_tan_2x_3_l584_584253


namespace smallest_sum_of_five_primes_l584_584756

-- Define the conditions for the problem
def is_prime (n : ℕ) : Prop := Prime n
def digits_used (n : ℕ) : List ℕ := (n.to_digits : List ℕ).filter (λ d, d ≠ 0)
def nonzero_digits_1_to_9 : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Definition for the set of five primes
def five_primes (ps : List ℕ) : Prop := 
  ps.length = 5 ∧ 
  (∀ p ∈ ps, is_prime p) ∧ 
  (digits_used ps.concat == nonzero_digits_1_to_9)

-- Problem statement in Lean 4
theorem smallest_sum_of_five_primes :
  ∃ (ps : List ℕ), five_primes ps ∧ ps.sum = 106 :=
by
  sorry

end smallest_sum_of_five_primes_l584_584756


namespace trigonometric_identity_l584_584774

theorem trigonometric_identity :
  (3 / (Real.sin (20 * Real.pi / 180))^2) - (1 / (Real.cos (20 * Real.pi / 180))^2) + 64 * (Real.sin (20 * Real.pi / 180))^2 = 32 :=
by
  sorry

end trigonometric_identity_l584_584774


namespace sin_60_proof_l584_584369

noncomputable def sin_60_eq : Prop :=
  sin (60 * real.pi / 180) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq :=
sorry

end sin_60_proof_l584_584369


namespace ratio_noah_to_joe_l584_584952

def noah_age_after_10_years : ℕ := 22
def years_elapsed : ℕ := 10
def joe_age : ℕ := 6
def noah_age : ℕ := noah_age_after_10_years - years_elapsed

theorem ratio_noah_to_joe : noah_age / joe_age = 2 := by
  -- calculation omitted for brevity
  sorry

end ratio_noah_to_joe_l584_584952


namespace simplify_sqrt_72_plus_sqrt_32_l584_584086

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l584_584086


namespace solution_set_inequality_l584_584636

theorem solution_set_inequality (x : ℝ) (h1 : x < -3) (h2 : x < 2) : x < -3 :=
by
  exact h1

end solution_set_inequality_l584_584636


namespace probability_perfect_square_divisor_l584_584308

theorem probability_perfect_square_divisor (m n : ℕ) (h_rel_prime : Nat.coprime m n) (h_prob : (m : ℚ) / n = 1 / 36) : m + n = 37 := by
  sorry

end probability_perfect_square_divisor_l584_584308


namespace number_of_possible_values_l584_584750

theorem number_of_possible_values (a b c : ℕ) (h : a + 11 * b + 111 * c = 1050) :
  ∃ (n : ℕ), 6 ≤ n ∧ n ≤ 1050 ∧ (n % 9 = 6) ∧ (n = a + 2 * b + 3 * c) :=
sorry

end number_of_possible_values_l584_584750


namespace Doug_age_l584_584243

theorem Doug_age
  (B : ℕ) (D : ℕ) (N : ℕ)
  (h1 : 2 * B = N)
  (h2 : B + D = 90)
  (h3 : 20 * N = 2000) : 
  D = 40 := sorry

end Doug_age_l584_584243


namespace range_of_function_l584_584631

theorem range_of_function : 
  ∀ x : ℝ, y = 1 - 2^x → y ∈ Iio (1) := 
by
  sorry

end range_of_function_l584_584631


namespace det_A_sq_plus_B_sq_nonneg_l584_584918

noncomputable def det_poly_deg_le_two {A B : Matrix (Fin 3) (Fin 3) ℝ} (h : A.mul B = 0) :
    (∃ f : ℂ → ℂ, ∀ x : ℂ, f x = Matrix.det (A.mul A + B.mul B + x • A.mul B) ∧ polynomial.degree (@polynomial.of_fn ℂ _ _ f) ≤ 2) :=
sorry

theorem det_A_sq_plus_B_sq_nonneg {A B : Matrix (Fin 3) (Fin 3) ℝ} (h : A.mul B = 0) :
    Matrix.det (A.mul A + B.mul B) ≥ 0 :=
sorry

end det_A_sq_plus_B_sq_nonneg_l584_584918


namespace simplify_trig_expression_l584_584602

theorem simplify_trig_expression : 
  ∀ (deg : ℝ), deg = 1 :=
by
  let sin := Real.sin
  let cos := Real.cos
  have h : ∀ (x : ℝ), cos (-x) = cos x := by sorry
  have h10_20_eq := sin 10 + sin 20
  have h_cos10_20_eq := cos 10 + cos 20
  have h1 := h10_20_eq
  have h2 := h_cos10_20_eq
  have main_eq := (sin 10 + sin 20) / (cos 10 + cos 20)
  have main_term := (2 * sin (15) * cos (-5)) / (2 * cos (15) * cos (-5))
  have simpl := main_term = (sin 15 / cos 15)
  have final := simpl = Real.tan (15)
  show final = Real.tan (15)

end simplify_trig_expression_l584_584602


namespace slope_angle_inclination_l584_584169

theorem slope_angle_inclination (x1 y1 x2 y2 : ℝ) (hA : x1 = -1) (hB : y1 = 3)
  (hC : x2 = real.sqrt 3) (hD : y2 = -real.sqrt 3) :
  ∃ θ : ℝ, θ = 120 ∧ tan θ = (y2 - y1) / (x2 - x1) :=
sorry

end slope_angle_inclination_l584_584169


namespace solution_to_exponential_equation_l584_584254

theorem solution_to_exponential_equation :
  ∃ x : ℕ, (8^12 + 8^12 + 8^12 = 2^x) ∧ x = 38 :=
by
  sorry

end solution_to_exponential_equation_l584_584254


namespace problem_sum_zero_l584_584935

def f (x : ℚ) : ℚ := x^2 * (1 - x)^2

theorem problem_sum_zero :
  (finset.sum (finset.range 2020) (λ k, (-1)^k * f (k / 2021))) = 0 :=
by sorry

end problem_sum_zero_l584_584935


namespace John_l584_584904

theorem John's_net_profit 
  (gross_income : ℕ)
  (car_purchase_cost : ℕ)
  (car_maintenance : ℕ → ℕ → ℕ)
  (car_insurance : ℕ)
  (car_tire_replacement : ℕ)
  (trade_in_value : ℕ)
  (tax_rate : ℚ)
  (total_taxes : ℕ)
  (monthly_maintenance_cost : ℕ)
  (months : ℕ)
  (net_profit : ℕ) :
  gross_income = 30000 →
  car_purchase_cost = 20000 →
  car_maintenance monthly_maintenance_cost months = 3600 →
  car_insurance = 1200 →
  car_tire_replacement = 400 →
  trade_in_value = 6000 →
  tax_rate = 15/100 →
  total_taxes = 4500 →
  monthly_maintenance_cost = 300 →
  months = 12 →
  net_profit = gross_income - (car_purchase_cost + car_maintenance monthly_maintenance_cost months + car_insurance + car_tire_replacement + total_taxes) + trade_in_value →
  net_profit = 6300 := 
by 
  sorry -- Proof to be provided

end John_l584_584904


namespace smallest_perimeter_of_triangle_PQR_l584_584223

noncomputable def triangle_PQR_perimeter (PQ PR QR : ℕ) (QJ : ℝ) 
  (h1 : PQ = PR) (h2 : QJ = 10) : ℕ :=
2 * (PQ + QR)

theorem smallest_perimeter_of_triangle_PQR (PQ PR QR : ℕ) (QJ : ℝ) :
  PQ = PR → QJ = 10 → 
  ∃ p, p = triangle_PQR_perimeter PQ PR QR QJ (by assumption) (by assumption) ∧ p = 78 :=
sorry

end smallest_perimeter_of_triangle_PQR_l584_584223


namespace find_m_times_t_l584_584555

-- Define the function g with given properties
variable (g : ℝ → ℝ)

-- Assume the conditions given in the problem
axiom g2 : g 2 = 2
axiom functional_eq : ∀ (x y : ℝ), g (x * y + g x) = x * g y + g x

-- Define the proof goal in terms of m and t
theorem find_m_times_t : 
  let m := 1 in -- Because g(1/3) has only one possible value, 2/3
  let t := g (1 / 3) in
  m * t = 2/3 :=
by
  sorry

end find_m_times_t_l584_584555


namespace dihedral_angle_problem_l584_584596

noncomputable def dihedral_angle_cos_eq (a s : ℝ) (h1 : s = 2 * a)
    (h2 : ∠ (2*a) (2*a) = 60) : ℝ  :=
    let cos_theta := -1 in
    let p := -1 in
    let q := 1 in
    p + q

-- The main theorem statement
theorem dihedral_angle_problem (a s : ℝ) (h1 : s = 2 * a)
    (h2 : ∠ (2*a) (2*a) = 60) : dihedral_angle_cos_eq a s h1 h2 = 0 :=
by sorry

end dihedral_angle_problem_l584_584596


namespace minimum_value_of_f_l584_584158

def f (x : ℝ) : ℝ := 4 * x^2 - 12 * x - 1

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f(y) ≥ f(x)) ∧ f(x) = -10 :=
by
  sorry

end minimum_value_of_f_l584_584158


namespace greatest_savings_option2_l584_584713

-- Define the initial price
def initial_price : ℝ := 15000

-- Define the discounts for each option
def discounts_option1 : List ℝ := [0.75, 0.85, 0.95]
def discounts_option2 : List ℝ := [0.65, 0.90, 0.95]
def discounts_option3 : List ℝ := [0.70, 0.90, 0.90]

-- Define a function to compute the final price after successive discounts
def final_price (initial : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ acc d => acc * d) initial

-- Define the savings for each option
def savings_option1 : ℝ := initial_price - (final_price initial_price discounts_option1)
def savings_option2 : ℝ := initial_price - (final_price initial_price discounts_option2)
def savings_option3 : ℝ := initial_price - (final_price initial_price discounts_option3)

-- Formulate the proof
theorem greatest_savings_option2 :
  max (max savings_option1 savings_option2) savings_option3 = savings_option2 :=
by
  sorry

end greatest_savings_option2_l584_584713


namespace sin_60_proof_l584_584368

noncomputable def sin_60_eq : Prop :=
  sin (60 * real.pi / 180) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq :=
sorry

end sin_60_proof_l584_584368


namespace find_b_for_perpendicular_lines_l584_584625

theorem find_b_for_perpendicular_lines:
  (∃ b : ℝ, ∀ (x y : ℝ), (3 * x + y - 5 = 0) ∧ (b * x + y + 2 = 0) → b = -1/3) :=
by
  sorry

end find_b_for_perpendicular_lines_l584_584625
