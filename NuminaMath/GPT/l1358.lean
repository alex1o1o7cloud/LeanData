import Mathlib

namespace number_of_boys_l1358_135864

theorem number_of_boys (M W B : ℕ) (X : ℕ) 
  (h1 : 5 * M = W) 
  (h2 : W = B) 
  (h3 : 5 * M * 12 + W * X + B * X = 180) 
  : B = 15 := 
by sorry

end number_of_boys_l1358_135864


namespace solve_inequalities_l1358_135820

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solve_inequalities_l1358_135820


namespace correct_operation_B_incorrect_operation_A_incorrect_operation_C_incorrect_operation_D_l1358_135842

theorem correct_operation_B (a : ℝ) : a^3 / a = a^2 := 
by sorry

theorem incorrect_operation_A (a : ℝ) : a^2 + a^5 ≠ a^7 := 
by sorry

theorem incorrect_operation_C (a : ℝ) : (3 * a^2)^2 ≠ 6 * a^4 := 
by sorry

theorem incorrect_operation_D (a b : ℝ) : (a - b)^2 ≠ a^2 - b^2 := 
by sorry

end correct_operation_B_incorrect_operation_A_incorrect_operation_C_incorrect_operation_D_l1358_135842


namespace problem_statement_l1358_135855

theorem problem_statement (x y : ℕ) (hx : x = 7) (hy : y = 3) : (x - y)^2 * (x + y)^2 = 1600 :=
by
  rw [hx, hy]
  sorry

end problem_statement_l1358_135855


namespace meet_time_approx_l1358_135881

noncomputable def length_of_track : ℝ := 1800 -- in meters
noncomputable def speed_first_woman : ℝ := 10 * 1000 / 3600 -- in meters per second
noncomputable def speed_second_woman : ℝ := 20 * 1000 / 3600 -- in meters per second
noncomputable def relative_speed : ℝ := speed_first_woman + speed_second_woman

theorem meet_time_approx (ε : ℝ) (hε : ε = 216.048) :
  ∃ t : ℝ, t = length_of_track / relative_speed ∧ abs (t - ε) < 0.001 :=
by
  sorry

end meet_time_approx_l1358_135881


namespace repeating_decimal_product_l1358_135858

theorem repeating_decimal_product :
  (8 / 99) * (36 / 99) = 288 / 9801 :=
by
  sorry

end repeating_decimal_product_l1358_135858


namespace new_person_weight_l1358_135856

noncomputable def weight_of_new_person (weight_of_replaced : ℕ) (number_of_persons : ℕ) (increase_in_average : ℕ) := 
  weight_of_replaced + number_of_persons * increase_in_average

theorem new_person_weight:
  weight_of_new_person 70 8 3 = 94 :=
  by
  -- Proof omitted
  sorry

end new_person_weight_l1358_135856


namespace g_600_l1358_135879

def g : ℕ → ℕ := sorry

axiom g_mul (x y : ℕ) (hx : x > 0) (hy : y > 0) : g (x * y) = g x + g y
axiom g_12 : g 12 = 18
axiom g_48 : g 48 = 26

theorem g_600 : g 600 = 36 :=
by 
  sorry

end g_600_l1358_135879


namespace range_of_f_l1358_135884

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (2 * x - Real.pi / 3) + 2 * Real.sin (x - Real.pi / 4) * Real.sin (x + Real.pi / 4)

theorem range_of_f : ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2), 
  -Real.sqrt 3 / 2 ≤ f x ∧ f x ≤ 1 := by
  sorry

end range_of_f_l1358_135884


namespace least_palindrome_divisible_by_25_l1358_135823

theorem least_palindrome_divisible_by_25 : ∃ (n : ℕ), 
  (10^4 ≤ n ∧ n < 10^5) ∧
  (∀ (a b c : ℕ), n = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a) ∧
  n % 25 = 0 ∧
  n = 10201 :=
by
  sorry

end least_palindrome_divisible_by_25_l1358_135823


namespace cos_540_eq_neg_one_l1358_135849

theorem cos_540_eq_neg_one : Real.cos (540 : ℝ) = -1 := by
  sorry

end cos_540_eq_neg_one_l1358_135849


namespace curve_of_polar_equation_is_line_l1358_135868

theorem curve_of_polar_equation_is_line (r θ : ℝ) :
  (r = 1 / (Real.sin θ - Real.cos θ)) →
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ (x y : ℝ), r * (Real.sin θ) = y ∧ r * (Real.cos θ) = x → a * x + b * y = c :=
by
  sorry

end curve_of_polar_equation_is_line_l1358_135868


namespace circle_division_parts_l1358_135891

-- Define the number of parts a circle is divided into by the chords.
noncomputable def numberOfParts (n : ℕ) : ℚ :=
  (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24

-- Prove that the number of parts is given by the defined function.
theorem circle_division_parts (n : ℕ) : numberOfParts n = (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24 := by
  sorry

end circle_division_parts_l1358_135891


namespace binary_operations_unique_l1358_135832

def binary_operation (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → (f a (f b c) = (f a b) * c)
  ∧ ∀ a : ℝ, a > 0 → a ≥ 1 → f a a ≥ 1

theorem binary_operations_unique (f : ℝ → ℝ → ℝ) (h : binary_operation f) :
  (∀ a b, f a b = a * b) ∨ (∀ a b, f a b = a / b) :=
sorry

end binary_operations_unique_l1358_135832


namespace tan_x_neg7_l1358_135840

theorem tan_x_neg7 (x : ℝ) (h1 : Real.sin (x + π / 4) = 3 / 5) (h2 : Real.sin (x - π / 4) = 4 / 5) : 
  Real.tan x = -7 :=
sorry

end tan_x_neg7_l1358_135840


namespace smallest_E_of_positive_reals_l1358_135893

noncomputable def E (a b c : ℝ) : ℝ :=
  (a^3) / (1 - a^2) + (b^3) / (1 - b^2) + (c^3) / (1 - c^2)

theorem smallest_E_of_positive_reals (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) : 
  E a b c = 1 / 8 := 
sorry

end smallest_E_of_positive_reals_l1358_135893


namespace max_students_late_all_three_days_l1358_135882

theorem max_students_late_all_three_days (A B C total l: ℕ) 
  (hA: A = 20) 
  (hB: B = 13) 
  (hC: C = 7) 
  (htotal: total = 30) 
  (hposA: 0 ≤ A) (hposB: 0 ≤ B) (hposC: 0 ≤ C) 
  (hpostotal: 0 ≤ total) 
  : l = 5 := by
  sorry

end max_students_late_all_three_days_l1358_135882


namespace ratio_of_x_and_y_l1358_135833

theorem ratio_of_x_and_y (x y : ℝ) (h : (x - y) / (x + y) = 4) : x / y = -5 / 3 :=
by sorry

end ratio_of_x_and_y_l1358_135833


namespace number_of_packets_l1358_135860

def ounces_in_packet : ℕ := 16 * 16 + 4
def ounces_in_ton : ℕ := 2500 * 16
def gunny_bag_capacity_in_ounces : ℕ := 13 * ounces_in_ton

theorem number_of_packets : gunny_bag_capacity_in_ounces / ounces_in_packet = 2000 :=
by
  sorry

end number_of_packets_l1358_135860


namespace hyperbola_condition_l1358_135889

theorem hyperbola_condition (m : ℝ) : (m > 0) ↔ (2 + m > 0 ∧ 1 + m > 0) :=
by sorry

end hyperbola_condition_l1358_135889


namespace find_sixth_term_l1358_135853

noncomputable def arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def sum_first_n_terms (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem find_sixth_term :
  ∀ (a1 S3 : ℕ),
  a1 = 2 →
  S3 = 12 →
  ∃ d : ℕ, sum_first_n_terms a1 d 3 = S3 ∧ arithmetic_sequence a1 d 6 = 12 :=
by
  sorry

end find_sixth_term_l1358_135853


namespace mean_weight_of_cats_l1358_135869

def weight_list : List ℝ :=
  [87, 90, 93, 95, 95, 98, 104, 106, 106, 107, 109, 110, 111, 112]

noncomputable def total_weight : ℝ := weight_list.sum

noncomputable def mean_weight : ℝ := total_weight / weight_list.length

theorem mean_weight_of_cats : mean_weight = 101.64 := by
  sorry

end mean_weight_of_cats_l1358_135869


namespace find_f_ln6_l1358_135865

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x = x - Real.exp (-x)

noncomputable def given_function_value : ℝ := Real.log 6

theorem find_f_ln6 (f : ℝ → ℝ)
  (h1 : odd_function f)
  (h2 : condition1 f) :
  f given_function_value = given_function_value + 6 :=
by
  sorry

end find_f_ln6_l1358_135865


namespace divisibility_by_11_l1358_135824

theorem divisibility_by_11
  (n : ℕ) (hn : n ≥ 2)
  (h : (n^2 + (4^n) + (7^n)) % n = 0) :
  (n^2 + 4^n + 7^n) % 11 = 0 := 
by
  sorry

end divisibility_by_11_l1358_135824


namespace find_m_l1358_135845

theorem find_m (m : ℤ) (h1 : m + 1 ≠ 0) (h2 : m^2 + 3 * m + 1 = -1) : m = -2 := 
by 
  sorry

end find_m_l1358_135845


namespace bouquet_count_l1358_135861

theorem bouquet_count : ∃ n : ℕ, n = 9 ∧ ∀ (r c : ℕ), 3 * r + 2 * c = 50 → n = 9 :=
by
  sorry

end bouquet_count_l1358_135861


namespace quadratic_equation_unique_l1358_135872

/-- Prove that among the given options, the only quadratic equation in \( x \) is \( x^2 - 3x = 0 \). -/
theorem quadratic_equation_unique (A B C D : ℝ → ℝ) :
  A = (3 * x + 2) →
  B = (x^2 - 3 * x) →
  C = (x + 3 * x * y - 1) →
  D = (1 / x - 4) →
  ∃! (eq : ℝ → ℝ), eq = B := by
  sorry

end quadratic_equation_unique_l1358_135872


namespace find_a_from_roots_l1358_135843

theorem find_a_from_roots (a : ℝ) :
  let A := {x | (x = a) ∨ (x = a - 1)}
  2 ∈ A → a = 2 ∨ a = 3 :=
by
  intros A h
  sorry

end find_a_from_roots_l1358_135843


namespace num_people_at_gathering_l1358_135878

noncomputable def total_people_at_gathering : ℕ :=
  let wine_soda := 12
  let wine_juice := 10
  let wine_coffee := 6
  let wine_tea := 4
  let soda_juice := 8
  let soda_coffee := 5
  let soda_tea := 3
  let juice_coffee := 7
  let juice_tea := 2
  let coffee_tea := 4
  let wine_soda_juice := 3
  let wine_soda_coffee := 1
  let wine_soda_tea := 2
  let wine_juice_coffee := 3
  let wine_juice_tea := 1
  let wine_coffee_tea := 2
  let soda_juice_coffee := 3
  let soda_juice_tea := 1
  let soda_coffee_tea := 2
  let juice_coffee_tea := 3
  let all_five := 1
  wine_soda + wine_juice + wine_coffee + wine_tea +
  soda_juice + soda_coffee + soda_tea + juice_coffee +
  juice_tea + coffee_tea + wine_soda_juice + wine_soda_coffee +
  wine_soda_tea + wine_juice_coffee + wine_juice_tea +
  wine_coffee_tea + soda_juice_coffee + soda_juice_tea +
  soda_coffee_tea + juice_coffee_tea + all_five

theorem num_people_at_gathering : total_people_at_gathering = 89 := by
  sorry

end num_people_at_gathering_l1358_135878


namespace initial_number_of_girls_is_31_l1358_135866

-- Define initial number of boys and girls
variables (b g : ℕ)

-- Conditions
def first_condition (g b : ℕ) : Prop := b = 3 * (g - 18)
def second_condition (g b : ℕ) : Prop := 4 * (b - 36) = g - 18

-- Theorem statement
theorem initial_number_of_girls_is_31 (b g : ℕ) (h1 : first_condition g b) (h2 : second_condition g b) : g = 31 :=
by
  sorry

end initial_number_of_girls_is_31_l1358_135866


namespace red_mushrooms_bill_l1358_135870

theorem red_mushrooms_bill (R : ℝ) : 
  (2/3) * R + 6 + 3 = 17 → R = 12 :=
by
  intro h
  sorry

end red_mushrooms_bill_l1358_135870


namespace functions_unique_l1358_135888

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem functions_unique (f g: ℝ → ℝ) :
  (∀ x : ℝ, x < 0 → (f (g x) = x / (x * f x - 2)) ∧ (g (f x) = x / (x * g x - 2))) →
  (∀ x : ℝ, 0 < x → (f x = 3 / x ∧ g x = 3 / x)) :=
by
  sorry

end functions_unique_l1358_135888


namespace number_of_spotted_blue_fish_l1358_135829

def total_fish := 60
def blue_fish := total_fish / 3
def spotted_blue_fish := blue_fish / 2

theorem number_of_spotted_blue_fish : spotted_blue_fish = 10 :=
by
  -- Proof is omitted
  sorry

end number_of_spotted_blue_fish_l1358_135829


namespace count_even_three_digit_numbers_less_than_600_l1358_135862

-- Define the digits
def digits : List ℕ := [1, 2, 3, 4, 5, 6]

-- Condition: the number must be less than 600, i.e., hundreds digit in {1, 2, 3, 4, 5}
def valid_hundreds (d : ℕ) : Prop := d ∈ [1, 2, 3, 4, 5]

-- Condition: the units (ones) digit must be even
def valid_units (d : ℕ) : Prop := d ∈ [2, 4, 6]

-- Problem: total number of valid three-digit numbers
def total_valid_numbers : ℕ :=
  List.product (List.product [1, 2, 3, 4, 5] digits) [2, 4, 6] |>.length

-- Proof statement
theorem count_even_three_digit_numbers_less_than_600 :
  total_valid_numbers = 90 := by
  sorry

end count_even_three_digit_numbers_less_than_600_l1358_135862


namespace polyhedron_volume_l1358_135892

-- Define the properties of the polygons
def isosceles_right_triangle (a : ℝ) := a ≠ 0 ∧ ∀ (x y : ℝ), x = y

def square (side : ℝ) := side = 2

def equilateral_triangle (side : ℝ) := side = 2 * Real.sqrt 2

-- Define the conditions
def condition_AE : Prop := isosceles_right_triangle 2
def condition_B : Prop := square 2
def condition_C : Prop := square 2
def condition_D : Prop := square 2
def condition_G : Prop := equilateral_triangle (2 * Real.sqrt 2)

-- Define the polyhedron volume calculation problem
theorem polyhedron_volume (hA : condition_AE) (hE : condition_AE) (hF : condition_AE) (hB : condition_B) (hC : condition_C) (hD : condition_D) (hG : condition_G) : 
  ∃ V : ℝ, V = 16 := 
sorry

end polyhedron_volume_l1358_135892


namespace km_per_gallon_proof_l1358_135875

-- Define the given conditions
def distance := 100
def gallons := 10

-- Define what we need to prove the correct answer
def kilometers_per_gallon := distance / gallons

-- Prove that the calculated kilometers per gallon is equal to 10
theorem km_per_gallon_proof : kilometers_per_gallon = 10 := by
  sorry

end km_per_gallon_proof_l1358_135875


namespace factorize_x4_plus_81_l1358_135801

theorem factorize_x4_plus_81 : 
  ∀ x : ℝ, 
    (x^4 + 81 = (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9)) :=
by
  intro x
  sorry

end factorize_x4_plus_81_l1358_135801


namespace average_steps_per_day_l1358_135876

theorem average_steps_per_day (total_steps : ℕ) (h : total_steps = 56392) : 
  (total_steps / 7 : ℚ) = 8056.00 :=
by
  sorry

end average_steps_per_day_l1358_135876


namespace hexagonal_prism_cross_section_l1358_135841

theorem hexagonal_prism_cross_section (n : ℕ) (h₁: n ≥ 3) (h₂: n ≤ 8) : ¬ (n = 9):=
sorry

end hexagonal_prism_cross_section_l1358_135841


namespace initial_alloy_weight_l1358_135886

theorem initial_alloy_weight
  (x : ℝ)  -- Weight of the initial alloy in ounces
  (h1 : 0.80 * (x + 24) = 0.50 * x + 24)  -- Equation derived from conditions
: x = 16 := 
sorry

end initial_alloy_weight_l1358_135886


namespace nap_time_l1358_135899

-- Definitions of given conditions
def flight_duration : ℕ := 680
def reading_time : ℕ := 120
def movie_time : ℕ := 240
def dinner_time : ℕ := 30
def radio_time : ℕ := 40
def game_time : ℕ := 70

def total_activity_time : ℕ := reading_time + movie_time + dinner_time + radio_time + game_time

-- Theorem statement
theorem nap_time : (flight_duration - total_activity_time) / 60 = 3 := by
  -- Here would go the proof steps verifying the equality
  sorry

end nap_time_l1358_135899


namespace find_f_two_l1358_135805

-- Define the function f with the given properties
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 1

-- Given conditions
variable (a b : ℝ)
axiom f_neg_two_zero : f (-2) a b = 0

-- Statement to be proven
theorem find_f_two : f 2 a b = 2 := 
by {
  sorry
}

end find_f_two_l1358_135805


namespace find_a3_l1358_135837

noncomputable def geometric_term (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q^(n-1)

noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (q^n - 1) / (q - 1)

theorem find_a3 (a : ℝ) (q : ℝ) (h_q : q = 3)
  (h_sum : geometric_sum a q 3 + geometric_sum a q 4 = 53 / 3) :
  geometric_term a q 3 = 3 :=
by
  sorry

end find_a3_l1358_135837


namespace cube_volume_from_surface_area_l1358_135880

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 150) : (S / 6) ^ (3 / 2) = 125 := by
  sorry

end cube_volume_from_surface_area_l1358_135880


namespace marquita_garden_width_l1358_135890

theorem marquita_garden_width
  (mancino_gardens : ℕ) (marquita_gardens : ℕ)
  (mancino_length mancnio_width marquita_length total_area : ℕ)
  (h1 : mancino_gardens = 3)
  (h2 : mancino_length = 16)
  (h3 : mancnio_width = 5)
  (h4 : marquita_gardens = 2)
  (h5 : marquita_length = 8)
  (h6 : total_area = 304) :
  ∃ (marquita_width : ℕ), marquita_width = 4 :=
by
  sorry

end marquita_garden_width_l1358_135890


namespace value_of_fraction_l1358_135852

variable {x y : ℝ}

theorem value_of_fraction (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 * x + y) / (x - 3 * y) = -2) :
  (x + 3 * y) / (3 * x - y) = 2 :=
sorry

end value_of_fraction_l1358_135852


namespace find_blue_highlighters_l1358_135806

theorem find_blue_highlighters
(h_pink : P = 9)
(h_yellow : Y = 8)
(h_total : T = 22)
(h_sum : P + Y + B = T) :
  B = 5 :=
by
  -- Proof would go here
  sorry

end find_blue_highlighters_l1358_135806


namespace M_eq_N_l1358_135867

def M : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}
def N : Set ℝ := {y | ∃ n : ℤ, y = Real.cos (n * Real.pi / 3)}

theorem M_eq_N : M = N := 
by 
  sorry

end M_eq_N_l1358_135867


namespace inequality_inequality_hold_l1358_135803

theorem inequality_inequality_hold (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab_sum : a^2 + b^2 = 1/2) :
  (1 / (1 - a)) + (1 / (1 - b)) ≥ 4 :=
by
  sorry

end inequality_inequality_hold_l1358_135803


namespace find_circumcenter_l1358_135816

-- Define a quadrilateral with vertices A, B, C, and D
structure Quadrilateral :=
  (A B C D : (ℝ × ℝ))

-- Define the coordinates of the circumcenter
def circumcenter (q : Quadrilateral) : ℝ × ℝ := (6, 1)

-- Given condition that A, B, C, and D are vertices of a quadrilateral
-- Prove that the circumcenter of the circumscribed circle is (6, 1)
theorem find_circumcenter (q : Quadrilateral) : 
  circumcenter q = (6, 1) :=
by sorry

end find_circumcenter_l1358_135816


namespace radius_of_larger_circle_l1358_135896

theorem radius_of_larger_circle (R1 R2 : ℝ) (α : ℝ) (h1 : α = 60) (h2 : R1 = 24) (h3 : R2 = 3 * R1) : 
  R2 = 72 := 
by
  sorry

end radius_of_larger_circle_l1358_135896


namespace non_congruent_right_triangles_unique_l1358_135819

theorem non_congruent_right_triangles_unique :
  ∃! (a: ℝ) (b: ℝ) (c: ℝ), a > 0 ∧ b = 2 * a ∧ c = a * Real.sqrt 5 ∧
  (3 * a + a * Real.sqrt 5 - a^2 = a * Real.sqrt 5) :=
by
  sorry

end non_congruent_right_triangles_unique_l1358_135819


namespace parallel_lines_perpendicular_lines_l1358_135809

section LineEquation

variables (a : ℝ) (x y : ℝ)

def l1 := (a-2) * x + 3 * y + a = 0
def l2 := a * x + (a-2) * y - 1 = 0

theorem parallel_lines (a : ℝ) :
  ((a-2)/a = 3/(a-2)) ↔ (a = (7 + Real.sqrt 33) / 2 ∨ a = (7 - Real.sqrt 33) / 2) := sorry

theorem perpendicular_lines (a : ℝ) :
  (a = 2 ∨ ((2-a)/3 * (a/(2-a)) = -1)) ↔ (a = 2 ∨ a = -3) := sorry

end LineEquation

end parallel_lines_perpendicular_lines_l1358_135809


namespace incorrect_number_read_as_l1358_135810

theorem incorrect_number_read_as (n a_incorrect a_correct correct_number incorrect_number : ℕ) 
(hn : n = 10) (h_inc_avg : a_incorrect = 18) (h_cor_avg : a_correct = 22) (h_cor_num : correct_number = 66) :
incorrect_number = 26 := by
  sorry

end incorrect_number_read_as_l1358_135810


namespace expected_dietary_restriction_l1358_135863

theorem expected_dietary_restriction (n : ℕ) (p : ℚ) (sample_size : ℕ) (expected : ℕ) :
  p = 1 / 4 ∧ sample_size = 300 ∧ expected = sample_size * p → expected = 75 := by
  sorry

end expected_dietary_restriction_l1358_135863


namespace net_increase_in_wealth_l1358_135847

-- Definitions for yearly changes and fees
def firstYearChange (initialAmt : ℝ) : ℝ := initialAmt * 1.75 - 0.02 * initialAmt * 1.75
def secondYearChange (amt : ℝ) : ℝ := amt * 0.7 - 0.02 * amt * 0.7
def thirdYearChange (amt : ℝ) : ℝ := amt * 1.45 - 0.02 * amt * 1.45
def fourthYearChange (amt : ℝ) : ℝ := amt * 0.85 - 0.02 * amt * 0.85

-- Total Value after 4th year accounting all changes and fees
def totalAfterFourYears (initialAmt : ℝ) : ℝ :=
  let afterFirstYear := firstYearChange initialAmt
  let afterSecondYear := secondYearChange afterFirstYear
  let afterThirdYear := thirdYearChange afterSecondYear
  fourthYearChange afterThirdYear

-- Capital gains tax calculation
def capitalGainsTax (initialAmt finalAmt : ℝ) : ℝ :=
  0.20 * (finalAmt - initialAmt)

-- Net value after taxes
def netValueAfterTaxes (initialAmt : ℝ) : ℝ :=
  let total := totalAfterFourYears initialAmt
  total - capitalGainsTax initialAmt total

-- Main theorem statement
theorem net_increase_in_wealth :
  ∀ (initialAmt : ℝ), netValueAfterTaxes initialAmt = initialAmt * 1.31408238206 := sorry

end net_increase_in_wealth_l1358_135847


namespace simple_interest_years_l1358_135877

theorem simple_interest_years (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : P = 300) 
  (h2 : (P * ((R + 6) / 100) * T) = (P * (R / 100) * T + 90)) : 
  T = 5 := 
by 
  -- Necessary proof steps go here
  sorry

end simple_interest_years_l1358_135877


namespace total_area_correct_l1358_135828

noncomputable def total_area (r p q : ℝ) : ℝ :=
  r^2 + 4*p^2 + 12*q

theorem total_area_correct
  (r p q : ℝ)
  (h : 12 * q = r^2 + 4 * p^2 + 45)
  (r_val : r = 6)
  (p_val : p = 1.5)
  (q_val : q = 7.5) :
  total_area r p q = 135 := by
  sorry

end total_area_correct_l1358_135828


namespace area_of_estate_l1358_135871

theorem area_of_estate (side_length_in_inches : ℝ) (scale : ℝ) (real_side_length : ℝ) (area : ℝ) :
  side_length_in_inches = 12 →
  scale = 100 →
  real_side_length = side_length_in_inches * scale →
  area = real_side_length ^ 2 →
  area = 1440000 :=
by
  sorry

end area_of_estate_l1358_135871


namespace max_prime_p_l1358_135811

-- Define the variables and conditions
variable (a b : ℕ)
variable (p : ℝ)

-- Define the prime condition
def is_prime (n : ℝ) : Prop := sorry -- Placeholder for the prime definition

-- Define the equation condition
def p_eq (p : ℝ) (a b : ℕ) : Prop := 
  p = (b / 4) * Real.sqrt ((2 * a - b) / (2 * a + b))

-- The theorem to prove
theorem max_prime_p (a b : ℕ) (p_max : ℝ) :
  (∃ p, is_prime p ∧ p_eq p a b) → p_max = 5 := 
sorry

end max_prime_p_l1358_135811


namespace greg_books_difference_l1358_135895

theorem greg_books_difference (M K G X : ℕ)
  (hM : M = 32)
  (hK : K = M / 4)
  (hG : G = 2 * K + X)
  (htotal : M + K + G = 65) :
  X = 9 :=
by
  sorry

end greg_books_difference_l1358_135895


namespace ball_drawing_ways_l1358_135839

theorem ball_drawing_ways :
    ∃ (r w y : ℕ), 
      0 ≤ r ∧ r ≤ 2 ∧
      0 ≤ w ∧ w ≤ 3 ∧
      0 ≤ y ∧ y ≤ 5 ∧
      r + w + y = 5 ∧
      10 ≤ 5 * r + 2 * w + y ∧ 
      5 * r + 2 * w + y ≤ 15 := 
sorry

end ball_drawing_ways_l1358_135839


namespace radius_of_tangent_circle_l1358_135802

-- Define the conditions
def is_45_45_90_triangle (A B C : ℝ × ℝ) (AB BC AC : ℝ) : Prop :=
  (AB = 2 ∧ BC = 2 ∧ AC = 2 * Real.sqrt 2) ∧
  (A = (0, 0) ∧ B = (2, 0) ∧ C = (2, 2))

def is_tangent_to_axes (O : ℝ × ℝ) (r : ℝ) : Prop :=
  O = (r, r)

def is_tangent_to_hypotenuse (O : ℝ × ℝ) (r : ℝ) (C : ℝ × ℝ) : Prop :=
  (C.1 - O.1) = Real.sqrt 2 * r ∧ (C.2 - O.2) = Real.sqrt 2 * r

-- Main theorem
theorem radius_of_tangent_circle :
  ∃ r : ℝ, ∀ (A B C O : ℝ × ℝ),
    is_45_45_90_triangle A B C (2) (2) (2 * Real.sqrt 2) →
    is_tangent_to_axes O r →
    is_tangent_to_hypotenuse O r C →
    r = Real.sqrt 2 :=
by
  sorry

end radius_of_tangent_circle_l1358_135802


namespace bus_stop_time_per_hour_l1358_135851

theorem bus_stop_time_per_hour
  (speed_no_stops : ℝ)
  (speed_with_stops : ℝ)
  (h1 : speed_no_stops = 50)
  (h2 : speed_with_stops = 35) : 
  18 = (60 * (1 - speed_with_stops / speed_no_stops)) :=
by
  sorry

end bus_stop_time_per_hour_l1358_135851


namespace neither_necessary_nor_sufficient_l1358_135800

-- defining polynomial inequalities
def inequality_1 (a1 b1 c1 x : ℝ) : Prop := a1 * x^2 + b1 * x + c1 > 0
def inequality_2 (a2 b2 c2 x : ℝ) : Prop := a2 * x^2 + b2 * x + c2 > 0

-- defining proposition P and proposition Q
def P (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := ∀ x : ℝ, inequality_1 a1 b1 c1 x ↔ inequality_2 a2 b2 c2 x
def Q (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2

-- prove that Q is neither a necessary nor sufficient condition for P
theorem neither_necessary_nor_sufficient {a1 b1 c1 a2 b2 c2 : ℝ} : ¬(Q a1 b1 c1 a2 b2 c2 ↔ P a1 b1 c1 a2 b2 c2) := 
sorry

end neither_necessary_nor_sufficient_l1358_135800


namespace derivative_at_2_l1358_135838

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem derivative_at_2 : (deriv f 2) = 1 / 4 :=
by 
  sorry

end derivative_at_2_l1358_135838


namespace jane_chickens_l1358_135885

-- Conditions
def eggs_per_chicken_per_week : ℕ := 6
def egg_price_per_dozen : ℕ := 2
def total_income_in_2_weeks : ℕ := 20

-- Mathematical problem
theorem jane_chickens : (total_income_in_2_weeks / egg_price_per_dozen) * 12 / (eggs_per_chicken_per_week * 2) = 10 :=
by
  sorry

end jane_chickens_l1358_135885


namespace solve_equation_l1358_135821

theorem solve_equation : ∀ x : ℝ, x ≠ -2 → x ≠ 0 → (3 / (x + 2) - 1 / x = 0 ↔ x = 1) :=
by
  intro x h1 h2
  sorry

end solve_equation_l1358_135821


namespace willam_tax_paid_l1358_135836

-- Define our conditions
variables (T : ℝ) (tax_collected : ℝ) (willam_percent : ℝ)

-- Initialize the conditions according to the problem statement
def is_tax_collected (tax_collected : ℝ) : Prop := tax_collected = 3840
def is_farm_tax_levied_on_cultivated_land : Prop := true -- Essentially means we acknowledge it is 50%
def is_willam_taxable_land_percentage (willam_percent : ℝ) : Prop := willam_percent = 0.25

-- The final theorem that states Mr. Willam's tax payment is $960 given the conditions
theorem willam_tax_paid  : 
  ∀ (T : ℝ),
  is_tax_collected 3840 → 
  is_farm_tax_levied_on_cultivated_land →
  is_willam_taxable_land_percentage 0.25 →
  0.25 * 3840 = 960 :=
sorry

end willam_tax_paid_l1358_135836


namespace cost_price_of_article_l1358_135835

-- Define the conditions
variable (C : ℝ) -- Cost price of the article
variable (SP : ℝ) -- Selling price of the article

-- Conditions according to the problem
def condition1 : Prop := SP = 0.75 * C
def condition2 : Prop := SP + 500 = 1.15 * C

-- The theorem to prove the cost price
theorem cost_price_of_article (h₁ : condition1 C SP) (h₂ : condition2 C SP) : C = 1250 :=
by
  sorry

end cost_price_of_article_l1358_135835


namespace greatest_three_digit_multiple_of_17_l1358_135854

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l1358_135854


namespace smallest_num_rectangles_to_cover_square_l1358_135859

-- Define essential conditions
def area_3by4_rectangle : ℕ := 3 * 4
def area_square (side_length : ℕ) : ℕ := side_length * side_length
def can_be_tiled_with_3by4 (side_length : ℕ) : Prop := (area_square side_length) % area_3by4_rectangle = 0

-- Define the main theorem
theorem smallest_num_rectangles_to_cover_square :
  can_be_tiled_with_3by4 12 → ∃ n : ℕ, n = (area_square 12) / area_3by4_rectangle ∧ n = 12 :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l1358_135859


namespace prob_yellow_straight_l1358_135818

variable {P : ℕ → ℕ → ℚ}
-- Defining the probabilities of the given events
def prob_green : ℚ := 2 / 3
def prob_straight : ℚ := 1 / 2
def prob_rose : ℚ := 1 / 4
def prob_daffodil : ℚ := 1 / 2
def prob_tulip : ℚ := 1 / 4
def prob_rose_straight : ℚ := 1 / 6
def prob_daffodil_curved : ℚ := 1 / 3
def prob_tulip_straight : ℚ := 1 / 8

/-- The probability of picking a yellow and straight-petaled flower is 1/6 -/
theorem prob_yellow_straight : P 1 1 = 1 / 6 := sorry

end prob_yellow_straight_l1358_135818


namespace triangle_vertices_l1358_135850

theorem triangle_vertices : 
  (∃ (x y : ℚ), 2 * x + y = 6 ∧ x - y = -4 ∧ x = 2 / 3 ∧ y = 14 / 3) ∧ 
  (∃ (x y : ℚ), x - y = -4 ∧ y = -1 ∧ x = -5) ∧
  (∃ (x y : ℚ), 2 * x + y = 6 ∧ y = -1 ∧ x = 7 / 2) :=
by
  sorry

end triangle_vertices_l1358_135850


namespace proof_2_abs_a_plus_b_less_abs_4_plus_ab_l1358_135857

theorem proof_2_abs_a_plus_b_less_abs_4_plus_ab (a b : ℝ) (h1 : abs a < 2) (h2 : abs b < 2) :
    2 * abs (a + b) < abs (4 + a * b) := 
by
  sorry

end proof_2_abs_a_plus_b_less_abs_4_plus_ab_l1358_135857


namespace solve_system_eq_l1358_135814

theorem solve_system_eq (x y : ℝ) (h1 : x - y = 1) (h2 : 2 * x + 3 * y = 7) :
  x = 2 ∧ y = 1 := by
  sorry

end solve_system_eq_l1358_135814


namespace find_cost_price_l1358_135808

/-- Define the given conditions -/
def selling_price : ℝ := 100
def profit_percentage : ℝ := 0.15
def cost_price : ℝ := 86.96

/-- Define the relationship between selling price and cost price -/
def relation (CP SP : ℝ) : Prop := SP = CP * (1 + profit_percentage)

/-- State the theorem based on the conditions and required proof -/
theorem find_cost_price 
  (SP : ℝ) (CP : ℝ) 
  (h1 : SP = selling_price) 
  (h2 : relation CP SP) : 
  CP = cost_price := 
by
  sorry

end find_cost_price_l1358_135808


namespace mary_remaining_cards_l1358_135830

variable (initial_cards : ℝ) (bought_cards : ℝ) (promised_cards : ℝ)

def remaining_cards (initial : ℝ) (bought : ℝ) (promised : ℝ) : ℝ :=
  initial + bought - promised

theorem mary_remaining_cards :
  initial_cards = 18.0 →
  bought_cards = 40.0 →
  promised_cards = 26.0 →
  remaining_cards initial_cards bought_cards promised_cards = 32.0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end mary_remaining_cards_l1358_135830


namespace unique_arrangements_of_BANANA_l1358_135831

-- Define the conditions as separate definitions in Lean 4
def word := "BANANA"
def total_letters := 6
def count_A := 3
def count_N := 2
def count_B := 1

-- State the theorem to be proven
theorem unique_arrangements_of_BANANA : 
  (total_letters.factorial) / (count_A.factorial * count_N.factorial * count_B.factorial) = 60 := 
by
  sorry

end unique_arrangements_of_BANANA_l1358_135831


namespace tan_30_eq_sqrt3_div3_l1358_135844

theorem tan_30_eq_sqrt3_div3 (sin_30_cos_30 : ℝ → ℝ → Prop)
  (h1 : sin_30_cos_30 (1 / 2) (Real.sqrt 3 / 2)) :
  ∃ t, t = Real.tan (Real.pi / 6) ∧ t = Real.sqrt 3 / 3 :=
by
  existsi Real.tan (Real.pi / 6)
  sorry

end tan_30_eq_sqrt3_div3_l1358_135844


namespace average_marks_of_all_students_l1358_135826

/-
Consider two classes:
- The first class has 12 students with an average mark of 40.
- The second class has 28 students with an average mark of 60.

We are to prove that the average marks of all students from both classes combined is 54.
-/

theorem average_marks_of_all_students (s1 s2 : ℕ) (m1 m2 : ℤ)
  (h1 : s1 = 12) (h2 : m1 = 40) (h3 : s2 = 28) (h4 : m2 = 60) :
  (s1 * m1 + s2 * m2) / (s1 + s2) = 54 :=
by
  rw [h1, h2, h3, h4]
  sorry

end average_marks_of_all_students_l1358_135826


namespace total_solutions_l1358_135815

-- Definitions and conditions
def tetrahedron_solutions := 1
def cube_solutions := 1
def octahedron_solutions := 3
def dodecahedron_solutions := 2
def icosahedron_solutions := 3

-- Main theorem statement
theorem total_solutions : 
  tetrahedron_solutions + cube_solutions + octahedron_solutions + dodecahedron_solutions + icosahedron_solutions = 10 := by
  sorry

end total_solutions_l1358_135815


namespace count_positive_integers_l1358_135887

theorem count_positive_integers (x : ℤ) : 
  (25 < x^2 + 6 * x + 8) → (x^2 + 6 * x + 8 < 50) → (x > 0) → (x = 3 ∨ x = 4) :=
by sorry

end count_positive_integers_l1358_135887


namespace range_of_m_if_solution_set_empty_solve_inequality_y_geq_m_l1358_135874

noncomputable def quadratic_function (m x : ℝ) : ℝ :=
  (m + 1) * x^2 - m * x + m - 1

-- Part 1
theorem range_of_m_if_solution_set_empty (m : ℝ) :
  (∀ x : ℝ, quadratic_function m x < 0 → false) ↔ m ≥ 2 * Real.sqrt 3 / 3 := sorry

-- Part 2
theorem solve_inequality_y_geq_m (m x : ℝ) (h : m > -2) :
  (quadratic_function m x ≥ m) ↔ 
  (m = -1 → x ≥ 1) ∧
  (m > -1 → x ≤ -1/(m+1) ∨ x ≥ 1) ∧
  (m > -2 ∧ m < -1 → 1 ≤ x ∧ x ≤ -1/(m+1)) := sorry

end range_of_m_if_solution_set_empty_solve_inequality_y_geq_m_l1358_135874


namespace emilia_donut_holes_count_l1358_135807

noncomputable def surface_area (r : ℕ) : ℕ := 4 * r^2

def lcm (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

def donut_holes := 5103

theorem emilia_donut_holes_count :
  ∀ (S1 S2 S3 : ℕ), 
  S1 = surface_area 5 → 
  S2 = surface_area 7 → 
  S3 = surface_area 9 → 
  donut_holes = lcm S1 S2 S3 / S1 :=
by
  intros S1 S2 S3 hS1 hS2 hS3
  sorry

end emilia_donut_holes_count_l1358_135807


namespace solve_ineq_for_a_eq_0_values_of_a_l1358_135817

theorem solve_ineq_for_a_eq_0 :
  ∀ x : ℝ, (|x + 2| - 3 * |x|) ≥ 0 ↔ (-1/2 <= x ∧ x <= 1) := 
by
  sorry

theorem values_of_a :
  ∀ x a : ℝ, (|x + 2| - 3 * |x|) ≥ a → (a ≤ 2) := 
by
  sorry

end solve_ineq_for_a_eq_0_values_of_a_l1358_135817


namespace iris_to_tulip_ratio_l1358_135822

theorem iris_to_tulip_ratio (earnings_per_bulb : ℚ)
  (tulip_bulbs daffodil_bulbs crocus_ratio total_earnings : ℕ)
  (iris_bulbs : ℕ) (h0 : earnings_per_bulb = 0.50)
  (h1 : tulip_bulbs = 20) (h2 : daffodil_bulbs = 30)
  (h3 : crocus_ratio = 3) (h4 : total_earnings = 75)
  (h5 : total_earnings = earnings_per_bulb * (tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_ratio * daffodil_bulbs))
  : iris_bulbs = 10 → tulip_bulbs = 20 → (iris_bulbs : ℚ) / (tulip_bulbs : ℚ) = 1 / 2 :=
by {
  intros; sorry
}

end iris_to_tulip_ratio_l1358_135822


namespace Ram_money_l1358_135873

theorem Ram_money (R G K : ℕ) (h1 : R = 7 * G / 17) (h2 : G = 7 * K / 17) (h3 : K = 4046) : R = 686 := by
  sorry

end Ram_money_l1358_135873


namespace greater_solution_of_quadratic_l1358_135848

theorem greater_solution_of_quadratic :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 5 * x₁ - 84 = 0) ∧ (x₂^2 - 5 * x₂ - 84 = 0) ∧ (max x₁ x₂ = 12) :=
by
  sorry

end greater_solution_of_quadratic_l1358_135848


namespace like_monomials_are_same_l1358_135813

theorem like_monomials_are_same (m n : ℤ) (h1 : 2 * m + 4 = 8) (h2 : 2 * n - 3 = 5) : m = 2 ∧ n = 4 :=
by
  sorry

end like_monomials_are_same_l1358_135813


namespace sales_tax_difference_l1358_135834

theorem sales_tax_difference :
  let price_before_tax := 40
  let tax_rate_8_percent := 0.08
  let tax_rate_7_percent := 0.07
  let sales_tax_8_percent := price_before_tax * tax_rate_8_percent
  let sales_tax_7_percent := price_before_tax * tax_rate_7_percent
  sales_tax_8_percent - sales_tax_7_percent = 0.4 := 
by
  sorry

end sales_tax_difference_l1358_135834


namespace chemistry_problem_l1358_135825

theorem chemistry_problem 
(C : ℝ)  -- concentration of the original salt solution
(h_mix : 1 * C / 100 = 15 * 2 / 100) : 
  C = 30 := 
sorry

end chemistry_problem_l1358_135825


namespace tyre_flattening_time_l1358_135883

theorem tyre_flattening_time (R1 R2 : ℝ) (hR1 : R1 = 1 / 9) (hR2 : R2 = 1 / 6) : 
  1 / (R1 + R2) = 3.6 :=
by 
  sorry

end tyre_flattening_time_l1358_135883


namespace units_digit_product_l1358_135804

theorem units_digit_product (a b : ℕ) (h1 : (a % 10 ≠ 0) ∧ (b % 10 ≠ 0)) : (a * b % 10 = 0) ∨ (a * b % 10 ≠ 0) :=
by
  sorry

end units_digit_product_l1358_135804


namespace happy_numbers_l1358_135827

theorem happy_numbers (n : ℕ) (h1 : n < 1000) 
(h2 : 7 ∣ n^2) (h3 : 8 ∣ n^2) (h4 : 9 ∣ n^2) (h5 : 10 ∣ n^2) : 
n = 420 ∨ n = 840 :=
sorry

end happy_numbers_l1358_135827


namespace bisection_method_example_l1358_135894

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 4

theorem bisection_method_example :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0) →
  (∃ x : ℝ, (1 / 2) < x ∧ x < 1 ∧ f x = 0) :=
by
  sorry

end bisection_method_example_l1358_135894


namespace sum_of_numbers_l1358_135812

theorem sum_of_numbers (a b c d : ℕ) (h1 : a > d) (h2 : a * b = c * d) (h3 : a + b + c + d = a * c) (h4 : ∀ x y z w: ℕ, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ) : a + b + c + d = 12 :=
sorry

end sum_of_numbers_l1358_135812


namespace relationship_xyz_l1358_135897

theorem relationship_xyz (x y z : ℝ) (h1 : x = Real.log x) (h2 : y = Real.logb 5 2) (h3 : z = Real.exp (-0.5)) : x > z ∧ z > y :=
by
  sorry

end relationship_xyz_l1358_135897


namespace consecutive_days_sum_l1358_135898

theorem consecutive_days_sum (x : ℕ) (h : 3 * x + 3 = 33) : x = 10 ∧ x + 1 = 11 ∧ x + 2 = 12 :=
by {
  sorry
}

end consecutive_days_sum_l1358_135898


namespace sum_of_perimeters_l1358_135846

theorem sum_of_perimeters (a : ℕ → ℝ) (h₁ : a 0 = 180) (h₂ : ∀ n, a (n + 1) = 1 / 2 * a n) :
  (∑' n, a n) = 360 :=
by
  sorry

end sum_of_perimeters_l1358_135846
