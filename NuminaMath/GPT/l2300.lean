import Mathlib

namespace NUMINAMATH_GPT_value_of_p_l2300_230047

theorem value_of_p (p q r : ℕ) (h1 : p + q + r = 70) (h2 : p = 2*q) (h3 : q = 3*r) : p = 42 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_p_l2300_230047


namespace NUMINAMATH_GPT_power_summation_l2300_230096

theorem power_summation :
  (-1:ℤ)^(49) + (2:ℝ)^(3^3 + 5^2 - 48^2) = -1 + 1 / 2 ^ (2252 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_power_summation_l2300_230096


namespace NUMINAMATH_GPT_no_value_of_b_l2300_230013

theorem no_value_of_b (b : ℤ) : ¬ ∃ (n : ℤ), 2 * b^2 + 3 * b + 2 = n^2 := 
sorry

end NUMINAMATH_GPT_no_value_of_b_l2300_230013


namespace NUMINAMATH_GPT_used_car_percentage_l2300_230065

-- Define the variables and conditions
variables (used_car_price original_car_price : ℕ) (h_used_car_price : used_car_price = 15000) (h_original_price : original_car_price = 37500)

-- Define the statement to prove the percentage
theorem used_car_percentage (h : used_car_price / original_car_price * 100 = 40) : true :=
sorry

end NUMINAMATH_GPT_used_car_percentage_l2300_230065


namespace NUMINAMATH_GPT_assignment_ways_l2300_230004

-- Definitions
def graduates := 5
def companies := 3

-- Statement to be proven
theorem assignment_ways :
  ∃ (ways : ℕ), ways = 150 :=
sorry

end NUMINAMATH_GPT_assignment_ways_l2300_230004


namespace NUMINAMATH_GPT_collinear_values_k_l2300_230081

/-- Define the vectors OA, OB, and OC using the given conditions. -/
def vectorOA (k : ℝ) : ℝ × ℝ := (k, 12)
def vectorOB : ℝ × ℝ := (4, 5)
def vectorOC (k : ℝ) : ℝ × ℝ := (10, k)

/-- Define vectors AB and BC using vector subtraction. -/
def vectorAB (k : ℝ) : ℝ × ℝ := (4 - k, -7)
def vectorBC (k : ℝ) : ℝ × ℝ := (6, k - 5)

/-- Collinearity condition for vectors AB and BC. -/
def collinear (k : ℝ) : Prop :=
  (4 - k) * (k - 5) + 42 = 0

/-- Prove that the value of k is 11 or -2 given the collinearity condition. -/
theorem collinear_values_k : ∀ k : ℝ, collinear k → (k = 11 ∨ k = -2) :=
by
  intros k h
  sorry

end NUMINAMATH_GPT_collinear_values_k_l2300_230081


namespace NUMINAMATH_GPT_f_one_equals_half_f_increasing_l2300_230084

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_half (x y : ℝ) : f (x + y) = f x + f y + 1/2

axiom f_half     : f (1/2) = 0

axiom f_positive (x : ℝ) (hx : x > 1/2) : f x > 0

theorem f_one_equals_half : f 1 = 1/2 := 
by 
  sorry

theorem f_increasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 > f x2 := 
by 
  sorry

end NUMINAMATH_GPT_f_one_equals_half_f_increasing_l2300_230084


namespace NUMINAMATH_GPT_rational_coordinates_l2300_230050

theorem rational_coordinates (x : ℚ) : ∃ y : ℚ, 2 * x^3 + 2 * y^3 - 3 * x^2 - 3 * y^2 + 1 = 0 :=
by
  use (1 - x)
  sorry

end NUMINAMATH_GPT_rational_coordinates_l2300_230050


namespace NUMINAMATH_GPT_smallest_n_l2300_230080

theorem smallest_n :
∃ (n : ℕ), (0 < n) ∧ (∃ k1 : ℕ, 5 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 7 * n = k2 ^ 3) ∧ n = 1225 :=
sorry

end NUMINAMATH_GPT_smallest_n_l2300_230080


namespace NUMINAMATH_GPT_sum_of_remainders_is_six_l2300_230070

def sum_of_remainders (n : ℕ) : ℕ :=
  n % 4 + (n + 1) % 4 + (n + 2) % 4 + (n + 3) % 4

theorem sum_of_remainders_is_six : ∀ n : ℕ, sum_of_remainders n = 6 :=
by
  intro n
  sorry

end NUMINAMATH_GPT_sum_of_remainders_is_six_l2300_230070


namespace NUMINAMATH_GPT_polynomial_remainder_l2300_230056

theorem polynomial_remainder (a : ℝ) (h : ∀ x : ℝ, x^3 + a * x^2 + 1 = (x^2 - 1) * (x + 2) + (x + 3)) : a = 2 :=
sorry

end NUMINAMATH_GPT_polynomial_remainder_l2300_230056


namespace NUMINAMATH_GPT_find_range_of_a_l2300_230090

noncomputable def value_range_for_a : Set ℝ := {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4}

theorem find_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0)  ∧
  (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0) ∧
  (¬ (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0) → ¬ (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0))
  → a ∈ value_range_for_a :=
sorry

end NUMINAMATH_GPT_find_range_of_a_l2300_230090


namespace NUMINAMATH_GPT_solution_set_inequalities_l2300_230099

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequalities_l2300_230099


namespace NUMINAMATH_GPT_jessica_borrowed_amount_l2300_230094

def payment_pattern (hour : ℕ) : ℕ :=
  match (hour % 6) with
  | 1 => 2
  | 2 => 4
  | 3 => 6
  | 4 => 8
  | 5 => 10
  | _ => 12

def total_payment (hours_worked : ℕ) : ℕ :=
  (hours_worked / 6) * 42 + (List.sum (List.map payment_pattern (List.range (hours_worked % 6))))

theorem jessica_borrowed_amount :
  total_payment 45 = 306 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_jessica_borrowed_amount_l2300_230094


namespace NUMINAMATH_GPT_initial_investment_C_l2300_230029

def total_investment : ℝ := 425
def increase_A (a : ℝ) : ℝ := 0.05 * a
def increase_B (b : ℝ) : ℝ := 0.08 * b
def increase_C (c : ℝ) : ℝ := 0.10 * c

theorem initial_investment_C (a b c : ℝ) (h1 : a + b + c = total_investment)
  (h2 : increase_A a = increase_B b) (h3 : increase_B b = increase_C c) : c = 100 := by
  sorry

end NUMINAMATH_GPT_initial_investment_C_l2300_230029


namespace NUMINAMATH_GPT_trigonometric_identity_solution_l2300_230091

theorem trigonometric_identity_solution (x : ℝ) (k : ℤ) :
  8.469 * (Real.sin x)^4 + 2 * (Real.cos x)^3 + 2 * (Real.sin x)^2 - Real.cos x + 1 = 0 ↔
  ∃ (k : ℤ), x = Real.pi + 2 * Real.pi * k := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_solution_l2300_230091


namespace NUMINAMATH_GPT_find_a_l2300_230036

theorem find_a (x a : ℝ) (h₁ : x^2 + x - 6 = 0) :
  (ax + 1 = 0 → (a = -1/2 ∨ a = -1/3) ∧ ax + 1 ≠ 0 ↔ false) := 
by
  sorry

end NUMINAMATH_GPT_find_a_l2300_230036


namespace NUMINAMATH_GPT_complete_square_eq_l2300_230001

theorem complete_square_eq (x : ℝ) : (x^2 - 6 * x - 5 = 0) -> (x - 3)^2 = 14 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_complete_square_eq_l2300_230001


namespace NUMINAMATH_GPT_lilly_fish_count_l2300_230071

-- Define the number of fish Rosy has
def rosy_fish : ℕ := 9

-- Define the total number of fish
def total_fish : ℕ := 19

-- Define the statement that Lilly has 10 fish given the conditions
theorem lilly_fish_count : rosy_fish + lilly_fish = total_fish → lilly_fish = 10 := by
  intro h
  sorry

end NUMINAMATH_GPT_lilly_fish_count_l2300_230071


namespace NUMINAMATH_GPT_race_result_l2300_230033

-- Definitions based on conditions
variable (hare_won : Bool)
variable (fox_second : Bool)
variable (hare_second : Bool)
variable (moose_first : Bool)

-- Condition that each squirrel had one error.
axiom owl_statement : xor hare_won fox_second ∧ xor hare_second moose_first

-- The final proof problem
theorem race_result : moose_first = true ∧ fox_second = true :=
by {
  -- Proving based on the owl's statement that each squirrel had one error
  sorry
}

end NUMINAMATH_GPT_race_result_l2300_230033


namespace NUMINAMATH_GPT_book_arrangement_count_l2300_230044

theorem book_arrangement_count :
  let total_books := 7
  let identical_math_books := 3
  let identical_physics_books := 2
  (Nat.factorial 7) / ((Nat.factorial 3) * (Nat.factorial 2)) = 420 := 
by
  sorry

end NUMINAMATH_GPT_book_arrangement_count_l2300_230044


namespace NUMINAMATH_GPT_multiply_preserve_equiv_l2300_230027

noncomputable def conditions_equiv_eqn (N D F : Polynomial ℝ) : Prop :=
  (D = F * (D / F)) ∧ (N.degree ≥ F.degree) ∧ (D ≠ 0)

theorem multiply_preserve_equiv (N D F : Polynomial ℝ) :
  conditions_equiv_eqn N D F →
  (N / D = 0 ↔ (N * F) / (D * F) = 0) :=
by
  sorry

end NUMINAMATH_GPT_multiply_preserve_equiv_l2300_230027


namespace NUMINAMATH_GPT_amount_spent_on_raw_materials_l2300_230092

-- Given conditions
def spending_on_machinery : ℝ := 125
def spending_as_cash (total_amount : ℝ) : ℝ := 0.10 * total_amount
def total_amount : ℝ := 250

-- Mathematically equivalent problem
theorem amount_spent_on_raw_materials :
  (X : ℝ) → X + spending_on_machinery + spending_as_cash total_amount = total_amount →
    X = 100 :=
by
  (intro X h)
  sorry

end NUMINAMATH_GPT_amount_spent_on_raw_materials_l2300_230092


namespace NUMINAMATH_GPT_annual_percentage_increase_l2300_230012

theorem annual_percentage_increase (present_value future_value : ℝ) (years: ℝ) (r : ℝ) 
  (h1 : present_value = 20000)
  (h2 : future_value = 24200)
  (h3 : years = 2) : 
  future_value = present_value * (1 + r)^years → r = 0.1 :=
sorry

end NUMINAMATH_GPT_annual_percentage_increase_l2300_230012


namespace NUMINAMATH_GPT_find_other_number_l2300_230010

theorem find_other_number (a b : ℕ) (gcd_ab : Nat.gcd a b = 45) (lcm_ab : Nat.lcm a b = 1260) (a_eq : a = 180) : b = 315 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_other_number_l2300_230010


namespace NUMINAMATH_GPT_problem1_l2300_230017

theorem problem1 (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) := by 
  sorry

end NUMINAMATH_GPT_problem1_l2300_230017


namespace NUMINAMATH_GPT_simplify_expression_l2300_230043

theorem simplify_expression (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a ≠ b) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2300_230043


namespace NUMINAMATH_GPT_sum_ratios_eq_l2300_230003

-- Define points A, B, C, D, E, and G as well as their relationships
variables {A B C D E G : Type}

-- Given conditions
axiom BD_2DC : ∀ {BD DC : ℝ}, BD = 2 * DC
axiom AE_3EB : ∀ {AE EB : ℝ}, AE = 3 * EB
axiom AG_2GD : ∀ {AG GD : ℝ}, AG = 2 * GD

-- Mass assumptions for the given problem
noncomputable def mC := 1
noncomputable def mB := 2
noncomputable def mD := mB + 2 * mC  -- mD = B's mass + 2*C's mass
noncomputable def mA := 1
noncomputable def mE := 3 * mA + mB  -- mE = 3A's mass + B's mass
noncomputable def mG := 2 * mA + mD  -- mG = 2A's mass + D's mass

-- Ratios defined according to the problem statement
noncomputable def ratio1 := (1 : ℝ) / mE
noncomputable def ratio2 := mD / mA
noncomputable def ratio3 := mD / mG

-- The Lean theorem to state the problem and correct answer
theorem sum_ratios_eq : ratio1 + ratio2 + ratio3 = (73 / 15 : ℝ) :=
by
  unfold ratio1 ratio2 ratio3
  sorry

end NUMINAMATH_GPT_sum_ratios_eq_l2300_230003


namespace NUMINAMATH_GPT_max_value_of_x2_plus_y2_l2300_230008

theorem max_value_of_x2_plus_y2 {x y : ℝ} 
  (h1 : x ≥ 1)
  (h2 : y ≥ x)
  (h3 : x - 2 * y + 3 ≥ 0) : 
  x^2 + y^2 ≤ 18 :=
sorry

end NUMINAMATH_GPT_max_value_of_x2_plus_y2_l2300_230008


namespace NUMINAMATH_GPT_john_eggs_per_week_l2300_230064

theorem john_eggs_per_week
  (pens : ℕ)
  (emus_per_pen : ℕ)
  (female_ratio : ℚ)
  (eggs_per_female_per_day : ℕ)
  (days_in_week : ℕ) :
  pens = 4 →
  emus_per_pen = 6 →
  female_ratio = 1/2 →
  eggs_per_female_per_day = 1 →
  days_in_week = 7 →
  (pens * emus_per_pen * female_ratio * eggs_per_female_per_day * days_in_week = 84) :=
by
  intros h_pens h_emus h_ratio h_eggs h_days
  rw [h_pens, h_emus, h_ratio, h_eggs, h_days]
  norm_num

end NUMINAMATH_GPT_john_eggs_per_week_l2300_230064


namespace NUMINAMATH_GPT_A_inter_B_eq_l2300_230022

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 > 1}

theorem A_inter_B_eq : A ∩ B = {-2, 2} := 
by
  sorry

end NUMINAMATH_GPT_A_inter_B_eq_l2300_230022


namespace NUMINAMATH_GPT_total_seats_in_theater_l2300_230098

def theater_charges_adults : ℝ := 3.0
def theater_charges_children : ℝ := 1.5
def total_income : ℝ := 510
def number_of_children : ℕ := 60

theorem total_seats_in_theater :
  ∃ (A C : ℕ), C = number_of_children ∧ theater_charges_adults * A + theater_charges_children * C = total_income ∧ A + C = 200 :=
by
  sorry

end NUMINAMATH_GPT_total_seats_in_theater_l2300_230098


namespace NUMINAMATH_GPT_prove_m_add_n_l2300_230052

-- Definitions from conditions
variables (m n : ℕ)

def condition1 : Prop := m + 1 = 3
def condition2 : Prop := m = n - 1

-- Statement to prove
theorem prove_m_add_n (h1 : condition1 m) (h2 : condition2 m n) : m + n = 5 := 
sorry

end NUMINAMATH_GPT_prove_m_add_n_l2300_230052


namespace NUMINAMATH_GPT_arithmetic_sequence_length_l2300_230042

theorem arithmetic_sequence_length :
  ∀ (a d a_n : ℕ), a = 6 → d = 4 → a_n = 154 → ∃ n: ℕ, a_n = a + (n-1) * d ∧ n = 38 :=
by
  intro a d a_n ha hd ha_n
  use 38
  rw [ha, hd, ha_n]
  -- Leaving the proof as an exercise
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_length_l2300_230042


namespace NUMINAMATH_GPT_polynomial_sat_condition_l2300_230046

theorem polynomial_sat_condition (P : Polynomial ℝ) (k : ℕ) (hk : 0 < k) :
  (P.comp P = P ^ k) →
  (P = 0 ∨ P = 1 ∨ (k % 2 = 1 ∧ P = -1) ∨ P = Polynomial.X ^ k) :=
sorry

end NUMINAMATH_GPT_polynomial_sat_condition_l2300_230046


namespace NUMINAMATH_GPT_binary_multiplication_l2300_230058

theorem binary_multiplication :
  let a := 0b1101101
  let b := 0b1011
  let product := 0b10001001111
  a * b = product :=
sorry

end NUMINAMATH_GPT_binary_multiplication_l2300_230058


namespace NUMINAMATH_GPT_one_greater_than_one_l2300_230007

theorem one_greater_than_one (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) : a > 1 ∨ b > 1 ∨ c > 1 :=
by
  sorry

end NUMINAMATH_GPT_one_greater_than_one_l2300_230007


namespace NUMINAMATH_GPT_union_sets_l2300_230055

def set_A : Set ℝ := {x | x^3 - 3 * x^2 - x + 3 < 0}
def set_B : Set ℝ := {x | |x + 1 / 2| ≥ 1}

theorem union_sets :
  set_A ∪ set_B = ( {x : ℝ | x < -1} ∪ {x : ℝ | x ≥ 1 / 2} ) :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l2300_230055


namespace NUMINAMATH_GPT_equation_solution_unique_l2300_230077

theorem equation_solution_unique (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) :
    (2 / (x - 3) = 3 / x ↔ x = 9) :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_unique_l2300_230077


namespace NUMINAMATH_GPT_evaluate_expression_l2300_230095

theorem evaluate_expression :
  3 + 2*Real.sqrt 3 + 1/(3 + 2*Real.sqrt 3) + 1/(2*Real.sqrt 3 - 3) = 3 + (16 * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2300_230095


namespace NUMINAMATH_GPT_tangent_line_at_1_l2300_230075

-- Assume the curve and the point of tangency
noncomputable def curve (x : ℝ) : ℝ := x^3 - 2*x^2 + 4*x + 5

-- Define the point of tangency
def point_of_tangency : ℝ := 1

-- Define the expected tangent line equation in standard form Ax + By + C = 0
def tangent_line (x y : ℝ) : Prop := 3 * x - y + 5 = 0

theorem tangent_line_at_1 :
  tangent_line point_of_tangency (curve point_of_tangency) := 
sorry

end NUMINAMATH_GPT_tangent_line_at_1_l2300_230075


namespace NUMINAMATH_GPT_no_positive_integer_solution_l2300_230054

theorem no_positive_integer_solution (m n : ℕ) (h : 0 < m) (h1 : 0 < n) : ¬ (5 * m^2 - 6 * m * n + 7 * n^2 = 2006) :=
sorry

end NUMINAMATH_GPT_no_positive_integer_solution_l2300_230054


namespace NUMINAMATH_GPT_exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l2300_230030

-- Lean 4 statement for part (a)
theorem exists_palindromic_number_divisible_by_5 : 
  ∃ (n : ℕ), (n = 51715) ∧ (n % 5 = 0) := sorry

-- Lean 4 statement for part (b)
theorem count_palindromic_numbers_divisible_by_5 : 
  (∃ (count : ℕ), count = 100) := sorry

end NUMINAMATH_GPT_exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l2300_230030


namespace NUMINAMATH_GPT_terminating_decimal_l2300_230067

theorem terminating_decimal : (45 / (2^2 * 5^3) : ℚ) = 0.090 :=
by
  sorry

end NUMINAMATH_GPT_terminating_decimal_l2300_230067


namespace NUMINAMATH_GPT_intersecting_circles_l2300_230074

theorem intersecting_circles (m n : ℝ) (h_intersect : ∃ c1 c2 : ℝ × ℝ, 
  (c1.1 - c1.2 - 2 = 0) ∧ (c2.1 - c2.2 - 2 = 0) ∧
  ∃ r1 r2 : ℝ, (c1.1 - 1)^2 + (c1.2 - 3)^2 = r1^2 ∧ (c2.1 - 1)^2 + (c2.2 - 3)^2 = r2^2 ∧
  (c1.1 - m)^2 + (c1.2 - n)^2 = r1^2 ∧ (c2.1 - m)^2 + (c2.2 - n)^2 = r2^2) :
  m + n = 4 :=
sorry

end NUMINAMATH_GPT_intersecting_circles_l2300_230074


namespace NUMINAMATH_GPT_hyperbola_condition_l2300_230086

theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (m + 2)) + (y^2 / (m + 1)) = 1) ↔ (-2 < m ∧ m < -1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l2300_230086


namespace NUMINAMATH_GPT_shortest_fence_length_l2300_230061

open Real

noncomputable def area_of_garden (length width : ℝ) : ℝ := length * width

theorem shortest_fence_length (length width : ℝ) (h : area_of_garden length width = 64) :
  4 * sqrt 64 = 32 :=
by
  -- The statement sets up the condition that the area is 64 and asks to prove minimum perimeter (fence length = perimeter).
  sorry

end NUMINAMATH_GPT_shortest_fence_length_l2300_230061


namespace NUMINAMATH_GPT_correct_multiplier_l2300_230000

theorem correct_multiplier
  (x : ℕ)
  (incorrect_multiplier : ℕ := 34)
  (difference : ℕ := 1215)
  (number_to_be_multiplied : ℕ := 135) :
  number_to_be_multiplied * x - number_to_be_multiplied * incorrect_multiplier = difference →
  x = 43 :=
  sorry

end NUMINAMATH_GPT_correct_multiplier_l2300_230000


namespace NUMINAMATH_GPT_problem_solution_l2300_230035

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10

theorem problem_solution : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2300_230035


namespace NUMINAMATH_GPT_distinct_values_for_T_l2300_230078

-- Define the conditions given in the problem:
def distinct_digits (n : ℕ) : Prop :=
  n / 1000 ≠ (n / 100 % 10) ∧ n / 1000 ≠ (n / 10 % 10) ∧ n / 1000 ≠ (n % 10) ∧
  (n / 100 % 10) ≠ (n / 10 % 10) ∧ (n / 100 % 10) ≠ (n % 10) ∧
  (n / 10 % 10) ≠ (n % 10)

def Psum (P S T : ℕ) : Prop := P + S = T

-- Main theorem statement:
theorem distinct_values_for_T : ∀ (P S T : ℕ),
  distinct_digits P ∧ distinct_digits S ∧ distinct_digits T ∧
  Psum P S T → 
  (∃ (values : Finset ℕ), values.card = 7 ∧ ∀ val ∈ values, val = T) :=
by
  sorry

end NUMINAMATH_GPT_distinct_values_for_T_l2300_230078


namespace NUMINAMATH_GPT_given_roots_find_coefficients_l2300_230041

theorem given_roots_find_coefficients {a b c : ℝ} :
  (1:ℝ)^5 + 2*(1)^4 + a * (1:ℝ)^2 + b * (1:ℝ) = c →
  (-1:ℝ)^5 + 2*(-1:ℝ)^4 + a * (-1:ℝ)^2 + b * (-1:ℝ) = c →
  a = -6 ∧ b = -1 ∧ c = -4 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_given_roots_find_coefficients_l2300_230041


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2300_230038

theorem arithmetic_sequence_common_difference (a : Nat → Int)
  (h1 : a 1 = 2) 
  (h3 : a 3 = 8)
  (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1))  -- General form for an arithmetic sequence given two terms
  : a 2 - a 1 = 3 :=
by
  -- The main steps of the proof will follow from the arithmetic progression properties
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2300_230038


namespace NUMINAMATH_GPT_painting_time_l2300_230069

variable (a d e : ℕ)

theorem painting_time (h : a * e * d = a * d * e) : (d * x = a^2 * e) := 
by
   sorry

end NUMINAMATH_GPT_painting_time_l2300_230069


namespace NUMINAMATH_GPT_find_k_for_xy_solution_l2300_230021

theorem find_k_for_xy_solution :
  ∀ (k : ℕ), (∃ (x y : ℕ), x * (x + k) = y * (y + 1))
  → k = 1 ∨ k ≥ 4 :=
by
  intros k h
  sorry -- proof goes here

end NUMINAMATH_GPT_find_k_for_xy_solution_l2300_230021


namespace NUMINAMATH_GPT_symmetric_line_eq_x_axis_l2300_230020

theorem symmetric_line_eq_x_axis (x y : ℝ) :
  (3 * x - 4 * y + 5 = 0) → (3 * x + 4 * y + 5 = 0) :=
sorry

end NUMINAMATH_GPT_symmetric_line_eq_x_axis_l2300_230020


namespace NUMINAMATH_GPT_Kyle_age_l2300_230083

-- Let's define the variables for each person's age.
variables (Shelley Kyle Julian Frederick Tyson Casey Sandra David Fiona : ℕ) 

-- Defining conditions based on given problem.
axiom condition1 : Shelley = Kyle - 3
axiom condition2 : Shelley = Julian + 4
axiom condition3 : Julian = Frederick - 20
axiom condition4 : Julian = Fiona + 5
axiom condition5 : Frederick = 2 * Tyson
axiom condition6 : Tyson = 2 * Casey
axiom condition7 : Casey = Fiona - 2
axiom condition8 : Casey = Sandra / 2
axiom condition9 : Sandra = David + 4
axiom condition10 : David = 16

-- The goal is to prove Kyle's age is 23 years old.
theorem Kyle_age : Kyle = 23 :=
by sorry

end NUMINAMATH_GPT_Kyle_age_l2300_230083


namespace NUMINAMATH_GPT_triangle_angles_median_bisector_altitude_l2300_230002

theorem triangle_angles_median_bisector_altitude {α β γ : ℝ} 
  (h : α + β + γ = 180) 
  (median_angle_condition : α / 4 + β / 4 + γ / 4 = 45) -- Derived from 90/4 = 22.5
  (median_from_C : 4 * α = γ) -- Given condition that angle is divided into 4 equal parts
  (median_angle_C : γ = 90) -- Derived that angle @ C must be right angle (90°)
  (sum_angles_C : α + β = 90) : 
  α = 22.5 ∧ β = 67.5 ∧ γ = 90 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angles_median_bisector_altitude_l2300_230002


namespace NUMINAMATH_GPT_flatville_additional_plates_max_count_l2300_230014

noncomputable def flatville_initial_plate_count : Nat :=
  6 * 4 * 5

noncomputable def flatville_max_plate_count : Nat :=
  6 * 6 * 6

theorem flatville_additional_plates_max_count : flatville_max_plate_count - flatville_initial_plate_count = 96 :=
by
  sorry

end NUMINAMATH_GPT_flatville_additional_plates_max_count_l2300_230014


namespace NUMINAMATH_GPT_oranges_to_put_back_l2300_230028

theorem oranges_to_put_back
  (p_A p_O : ℕ)
  (A O : ℕ)
  (total_fruits : ℕ)
  (initial_avg_price new_avg_price : ℕ)
  (x : ℕ)
  (h1 : p_A = 40)
  (h2 : p_O = 60)
  (h3 : total_fruits = 15)
  (h4 : initial_avg_price = 48)
  (h5 : new_avg_price = 45)
  (h6 : A + O = total_fruits)
  (h7 : (p_A * A + p_O * O) / total_fruits = initial_avg_price)
  (h8 : (720 - 60 * x) / (15 - x) = 45) :
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_oranges_to_put_back_l2300_230028


namespace NUMINAMATH_GPT_remainder_6_pow_23_mod_5_l2300_230051

theorem remainder_6_pow_23_mod_5 : (6 ^ 23) % 5 = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_6_pow_23_mod_5_l2300_230051


namespace NUMINAMATH_GPT_rotation_problem_l2300_230016

theorem rotation_problem (y : ℝ) (hy : y < 360) :
  (450 % 360 == 90) ∧ (y == 360 - 90) ∧ (90 + (360 - y) % 360 == 0) → y == 270 :=
by {
  -- Proof steps go here
  sorry
}

end NUMINAMATH_GPT_rotation_problem_l2300_230016


namespace NUMINAMATH_GPT_find_num_biology_books_l2300_230025

-- Given conditions
def num_chemistry_books : ℕ := 8
def total_ways_to_pick : ℕ := 2548

-- Function to calculate combinations
def combination (n k : ℕ) := n.choose k

-- Statement to be proved
theorem find_num_biology_books (B : ℕ) (h1 : combination num_chemistry_books 2 = 28) 
  (h2 : combination B 2 * 28 = total_ways_to_pick) : B = 14 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_num_biology_books_l2300_230025


namespace NUMINAMATH_GPT_joggers_meetings_l2300_230072

theorem joggers_meetings (road_length : ℝ)
  (speed_A speed_B : ℝ)
  (start_time : ℝ)
  (meeting_time : ℝ) :
  road_length = 400 → 
  speed_A = 3 → 
  speed_B = 2.5 →
  start_time = 0 → 
  meeting_time = 1200 → 
  ∃ y : ℕ, y = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_joggers_meetings_l2300_230072


namespace NUMINAMATH_GPT_karsyn_total_payment_l2300_230034

-- Define the initial price of the phone
def initial_price : ℝ := 600

-- Define the discounted rate for the phone
def discount_rate_phone : ℝ := 0.20

-- Define the prices for additional items
def phone_case_price : ℝ := 25
def screen_protector_price : ℝ := 15

-- Define the discount rates
def discount_rate_125 : ℝ := 0.05
def discount_rate_150 : ℝ := 0.10
def final_discount_rate : ℝ := 0.03

-- Define the tax rate and fee
def exchange_rate_fee : ℝ := 0.02

noncomputable def total_payment (initial_price : ℝ) (discount_rate_phone : ℝ) 
  (phone_case_price : ℝ) (screen_protector_price : ℝ) (discount_rate_125 : ℝ) 
  (discount_rate_150 : ℝ) (final_discount_rate : ℝ) (exchange_rate_fee : ℝ) : ℝ :=
  let discounted_phone_price := initial_price * discount_rate_phone
  let additional_items_price := phone_case_price + screen_protector_price
  let total_before_discounts := discounted_phone_price + additional_items_price
  let total_after_first_discount := total_before_discounts * (1 - discount_rate_125)
  let total_after_second_discount := total_after_first_discount * (1 - discount_rate_150)
  let total_after_all_discounts := total_after_second_discount * (1 - final_discount_rate)
  let total_with_exchange_fee := total_after_all_discounts * (1 + exchange_rate_fee)
  total_with_exchange_fee

theorem karsyn_total_payment :
  total_payment initial_price discount_rate_phone phone_case_price screen_protector_price 
    discount_rate_125 discount_rate_150 final_discount_rate exchange_rate_fee = 135.35 := 
  by 
  -- Specify proof steps here
  sorry

end NUMINAMATH_GPT_karsyn_total_payment_l2300_230034


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l2300_230026

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q) 
  (h_inc : ∀ n, a (n + 1) > a n) (h2 : a 2 = 2) (h3 : a 4 - a 3 = 4) : q = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l2300_230026


namespace NUMINAMATH_GPT_pentagon_number_arrangement_l2300_230082

def no_common_divisor_other_than_one (a b : ℕ) : Prop :=
  ∀ d : ℕ, d > 1 → (d ∣ a ∧ d ∣ b) → false

def has_common_divisor_greater_than_one (a b : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d ∣ a ∧ d ∣ b

theorem pentagon_number_arrangement :
  ∃ (A B C D E : ℕ),
    no_common_divisor_other_than_one A B ∧
    no_common_divisor_other_than_one B C ∧
    no_common_divisor_other_than_one C D ∧
    no_common_divisor_other_than_one D E ∧
    no_common_divisor_other_than_one E A ∧
    has_common_divisor_greater_than_one A C ∧
    has_common_divisor_greater_than_one A D ∧
    has_common_divisor_greater_than_one B D ∧
    has_common_divisor_greater_than_one B E ∧
    has_common_divisor_greater_than_one C E :=
sorry

end NUMINAMATH_GPT_pentagon_number_arrangement_l2300_230082


namespace NUMINAMATH_GPT_number_of_outfits_l2300_230060

-- Definitions based on conditions
def trousers : ℕ := 4
def shirts : ℕ := 8
def jackets : ℕ := 3
def belts : ℕ := 2

-- The statement to prove
theorem number_of_outfits : trousers * shirts * jackets * belts = 192 := by
  sorry

end NUMINAMATH_GPT_number_of_outfits_l2300_230060


namespace NUMINAMATH_GPT_find_hourly_charge_computer_B_l2300_230062

noncomputable def hourly_charge_computer_B (B : ℝ) :=
  ∃ (A h : ℝ),
    A = 1.4 * B ∧
    B * (h + 20) = 550 ∧
    A * h = 550 ∧
    B = 7.86

theorem find_hourly_charge_computer_B : ∃ B : ℝ, hourly_charge_computer_B B :=
  sorry

end NUMINAMATH_GPT_find_hourly_charge_computer_B_l2300_230062


namespace NUMINAMATH_GPT_sophie_aunt_money_l2300_230085

noncomputable def totalMoneyGiven (shirts: ℕ) (shirtCost: ℝ) (trousers: ℕ) (trouserCost: ℝ) (additionalItems: ℕ) (additionalItemCost: ℝ) : ℝ :=
  shirts * shirtCost + trousers * trouserCost + additionalItems * additionalItemCost

theorem sophie_aunt_money : totalMoneyGiven 2 18.50 1 63 4 40 = 260 := 
by
  sorry

end NUMINAMATH_GPT_sophie_aunt_money_l2300_230085


namespace NUMINAMATH_GPT_proof_problem_l2300_230009

noncomputable def M : ℕ := 50
noncomputable def T : ℕ := M + Nat.div M 10
noncomputable def W : ℕ := 2 * (M + T)
noncomputable def Th : ℕ := W / 2
noncomputable def total_T_T_W_Th : ℕ := T + W + Th
noncomputable def total_M_T_W_Th : ℕ := M + total_T_T_W_Th
noncomputable def F_S_sun : ℕ := Nat.div (450 - total_M_T_W_Th) 3
noncomputable def car_tolls : ℕ := 150 * 2
noncomputable def bus_tolls : ℕ := 150 * 5
noncomputable def truck_tolls : ℕ := 150 * 10
noncomputable def total_toll : ℕ := car_tolls + bus_tolls + truck_tolls

theorem proof_problem :
  (total_T_T_W_Th = 370) ∧
  (F_S_sun = 10) ∧
  (total_toll = 2550) := by
  sorry

end NUMINAMATH_GPT_proof_problem_l2300_230009


namespace NUMINAMATH_GPT_perp_lines_a_value_l2300_230093

theorem perp_lines_a_value :
  ∀ a : ℝ, ((a + 1) * 1 - 2 * (-a) = 0) → a = 1 :=
by
  intro a
  intro h
  -- We now state that a must satisfy the given condition and show that this leads to a = 1
  -- The proof is left as sorry
  sorry

end NUMINAMATH_GPT_perp_lines_a_value_l2300_230093


namespace NUMINAMATH_GPT_probability_of_head_equal_half_l2300_230089

def fair_coin_probability : Prop :=
  ∀ (H T : ℕ), (H = 1 ∧ T = 1 ∧ (H + T = 2)) → ((H / (H + T)) = 1 / 2)

theorem probability_of_head_equal_half : fair_coin_probability :=
sorry

end NUMINAMATH_GPT_probability_of_head_equal_half_l2300_230089


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees_l2300_230066

theorem sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees :
  ∃ (n : ℕ), (n * (n - 3) / 2 = 14) → ((n - 2) * 180 = 900) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees_l2300_230066


namespace NUMINAMATH_GPT_inequality_solution_l2300_230045

theorem inequality_solution :
  {x : ℝ | |2 * x - 3| + |x + 1| < 7 ∧ x ≤ 4} = {x : ℝ | -5 / 3 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2300_230045


namespace NUMINAMATH_GPT_nina_jerome_age_ratio_l2300_230049

variable (N J L : ℕ)

theorem nina_jerome_age_ratio (h1 : L = N - 4) (h2 : L + N + J = 36) (h3 : L = 6) : N / J = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_nina_jerome_age_ratio_l2300_230049


namespace NUMINAMATH_GPT_sum_of_tangencies_l2300_230087

noncomputable def f (x : ℝ) : ℝ := max (-7 * x - 23) (max (2 * x + 5) (5 * x + 17))

noncomputable def q (x : ℝ) : ℝ := sorry  -- since the exact form of q is not specified, we use sorry here

-- Define the tangency condition
def is_tangent (q f : ℝ → ℝ) (x : ℝ) : Prop := (q x = f x) ∧ (deriv q x = deriv f x)

-- Define the three points of tangency
variable {x₄ x₅ x₆ : ℝ}

-- q(x) is tangent to f(x) at points x₄, x₅, x₆
axiom tangent_x₄ : is_tangent q f x₄
axiom tangent_x₅ : is_tangent q f x₅
axiom tangent_x₆ : is_tangent q f x₆

-- Now state the theorem
theorem sum_of_tangencies : x₄ + x₅ + x₆ = -70 / 9 :=
sorry

end NUMINAMATH_GPT_sum_of_tangencies_l2300_230087


namespace NUMINAMATH_GPT_maximum_value_of_f_l2300_230037

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x - 2 * Real.sin (3 * x))

theorem maximum_value_of_f :
  ∃ x : ℝ, f x = (16 * Real.sqrt 3) / 9 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_f_l2300_230037


namespace NUMINAMATH_GPT_triangle_angle_ratio_l2300_230040

theorem triangle_angle_ratio (A B C D : Type*) 
  (α β γ δ : ℝ) -- α = ∠BAC, β = ∠ABC, γ = ∠BCA, δ = external angles
  (h1 : α + β + γ = 180)
  (h2 : δ = α + γ)
  (h3 : δ = β + γ) : (2 * 180 - (α + β)) / (α + β) = 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_ratio_l2300_230040


namespace NUMINAMATH_GPT_negation_of_existential_proposition_l2300_230063

theorem negation_of_existential_proposition :
  (¬ (∃ x : ℝ, x > Real.sin x)) ↔ (∀ x : ℝ, x ≤ Real.sin x) :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_existential_proposition_l2300_230063


namespace NUMINAMATH_GPT_meet_floor_l2300_230076

noncomputable def xiaoming_meets_xiaoying (x y meet_floor: ℕ) : Prop :=
  x = 4 → y = 3 → (meet_floor = 22)

theorem meet_floor (x y meet_floor: ℕ) (h1: x = 4) (h2: y = 3) :
  xiaoming_meets_xiaoying x y meet_floor :=
by
  sorry

end NUMINAMATH_GPT_meet_floor_l2300_230076


namespace NUMINAMATH_GPT_algebraic_expression_value_l2300_230024

theorem algebraic_expression_value (a x : ℝ) (h : 3 * a - x = x + 2) (hx : x = 2) : a^2 - 2 * a + 1 = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_algebraic_expression_value_l2300_230024


namespace NUMINAMATH_GPT_avg_weights_N_square_of_integer_l2300_230019

theorem avg_weights_N_square_of_integer (N : ℕ) :
  (∃ S : ℕ, S > 0 ∧ ∃ k : ℕ, k * k = N + 1 ∧ S = (N * (N + 1)) / 2 / (N - k + 1) ∧ (N * (N + 1)) / 2 - S = (N - k) * S) ↔ (∃ k : ℕ, k * k = N + 1) := by
  sorry

end NUMINAMATH_GPT_avg_weights_N_square_of_integer_l2300_230019


namespace NUMINAMATH_GPT_distance_to_parabola_focus_l2300_230088

theorem distance_to_parabola_focus :
  ∀ (x : ℝ), ((4 : ℝ) = (1 / 4) * x^2) → dist (0, 4) (0, 5) = 5 := 
by
  intro x
  intro hyp
  -- initial conditions indicate the distance is 5 and can be directly given
  sorry

end NUMINAMATH_GPT_distance_to_parabola_focus_l2300_230088


namespace NUMINAMATH_GPT_functional_relationship_l2300_230097

-- Define the conditions
def directlyProportional (y x k : ℝ) : Prop :=
  y + 6 = k * (x + 1)

def specificCondition1 (x y : ℝ) : Prop :=
  x = 3 ∧ y = 2

-- State the theorem
theorem functional_relationship (k : ℝ) :
  (∀ x y, directlyProportional y x k) →
  specificCondition1 3 2 →
  ∀ x, ∃ y, y = 2 * x - 4 :=
by
  intro directProp
  intro specCond
  sorry

end NUMINAMATH_GPT_functional_relationship_l2300_230097


namespace NUMINAMATH_GPT_nitin_rank_last_l2300_230048

theorem nitin_rank_last (total_students : ℕ) (rank_start : ℕ) (rank_last : ℕ) 
  (h1 : total_students = 58) 
  (h2 : rank_start = 24) 
  (h3 : rank_last = total_students - rank_start + 1) : 
  rank_last = 35 := 
by 
  -- proof can be filled in here
  sorry

end NUMINAMATH_GPT_nitin_rank_last_l2300_230048


namespace NUMINAMATH_GPT_sam_dimes_l2300_230032

theorem sam_dimes (dimes_original dimes_given : ℕ) :
  dimes_original = 9 → dimes_given = 7 → dimes_original + dimes_given = 16 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sam_dimes_l2300_230032


namespace NUMINAMATH_GPT_stone_solution_l2300_230005

noncomputable def stone_problem : Prop :=
  ∃ y : ℕ, (∃ x z : ℕ, x + y + z = 100 ∧ x + 10 * y + 50 * z = 500) ∧
    ∀ y1 y2 : ℕ, (∃ x1 z1 : ℕ, x1 + y1 + z1 = 100 ∧ x1 + 10 * y1 + 50 * z1 = 500) ∧
                (∃ x2 z2 : ℕ, x2 + y2 + z2 = 100 ∧ x2 + 10 * y2 + 50 * z2 = 500) →
                y1 = y2

theorem stone_solution : stone_problem :=
sorry

end NUMINAMATH_GPT_stone_solution_l2300_230005


namespace NUMINAMATH_GPT_mixture_weight_l2300_230011

theorem mixture_weight (a b : ℝ) (h1 : a = 26.1) (h2 : a / (a + b) = 9 / 20) : a + b = 58 :=
sorry

end NUMINAMATH_GPT_mixture_weight_l2300_230011


namespace NUMINAMATH_GPT_parabolas_pass_through_origin_l2300_230031

-- Definition of a family of parabolas
def parabola_family (p q : ℝ) (x : ℝ) : ℝ := -x^2 + p * x + q

-- Definition of vertices lying on y = x^2
def vertex_condition (p q : ℝ) : Prop :=
  ∃ a : ℝ, (a^2 = -a^2 + p * a + q)

-- Proving that all such parabolas pass through the point (0, 0)
theorem parabolas_pass_through_origin :
  ∀ (p q : ℝ), vertex_condition p q → parabola_family p q 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_parabolas_pass_through_origin_l2300_230031


namespace NUMINAMATH_GPT_probability_two_red_balls_l2300_230018

def total_balls : ℕ := 15
def red_balls_initial : ℕ := 7
def blue_balls_initial : ℕ := 8
def red_balls_after_first_draw : ℕ := 6
def remaining_balls_after_first_draw : ℕ := 14

theorem probability_two_red_balls :
  (red_balls_initial / total_balls) *
  (red_balls_after_first_draw / remaining_balls_after_first_draw) = 1 / 5 :=
by sorry

end NUMINAMATH_GPT_probability_two_red_balls_l2300_230018


namespace NUMINAMATH_GPT_min_value_of_2a_plus_b_l2300_230079

variable (a b : ℝ)

def condition := a > 0 ∧ b > 0 ∧ a - 2 * a * b + b = 0

-- Define what needs to be proved
theorem min_value_of_2a_plus_b (h : condition a b) : ∃ a b : ℝ, 2 * a + b = (3 / 2) + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_2a_plus_b_l2300_230079


namespace NUMINAMATH_GPT_consecutive_even_numbers_divisible_by_384_l2300_230057

theorem consecutive_even_numbers_divisible_by_384 (n : Nat) (h1 : n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) = 384) : n = 6 :=
sorry

end NUMINAMATH_GPT_consecutive_even_numbers_divisible_by_384_l2300_230057


namespace NUMINAMATH_GPT_henry_twice_jill_years_ago_l2300_230073

def henry_age : ℕ := 23
def jill_age : ℕ := 17
def sum_of_ages (H J : ℕ) : Prop := H + J = 40

theorem henry_twice_jill_years_ago (H J : ℕ) (H1 : sum_of_ages H J) (H2 : H = 23) (H3 : J = 17) : ∃ x : ℕ, H - x = 2 * (J - x) ∧ x = 11 := 
by
  sorry

end NUMINAMATH_GPT_henry_twice_jill_years_ago_l2300_230073


namespace NUMINAMATH_GPT_discount_percentage_l2300_230068

theorem discount_percentage (original_price sale_price : ℝ) (h1 : original_price = 150) (h2 : sale_price = 135) : 
  (original_price - sale_price) / original_price * 100 = 10 :=
by 
  sorry

end NUMINAMATH_GPT_discount_percentage_l2300_230068


namespace NUMINAMATH_GPT_average_weight_of_whole_class_l2300_230015

/-- Section A has 30 students -/
def num_students_A : ℕ := 30

/-- Section B has 20 students -/
def num_students_B : ℕ := 20

/-- The average weight of Section A is 40 kg -/
def avg_weight_A : ℕ := 40

/-- The average weight of Section B is 35 kg -/
def avg_weight_B : ℕ := 35

/-- The average weight of the whole class is 38 kg -/
def avg_weight_whole_class : ℕ := 38

-- Proof that the average weight of the whole class is equal to 38 kg

theorem average_weight_of_whole_class : 
  ((num_students_A * avg_weight_A) + (num_students_B * avg_weight_B)) / (num_students_A + num_students_B) = avg_weight_whole_class :=
by
  -- Sorry indicates that the proof is omitted.
  sorry

end NUMINAMATH_GPT_average_weight_of_whole_class_l2300_230015


namespace NUMINAMATH_GPT_proof_problem_l2300_230053

open Real

noncomputable def problem (c d : ℝ) : ℝ :=
  5^(c / d) + 2^(d / c)

theorem proof_problem :
  let c := log 8
  let d := log 25
  problem c d = 2 * sqrt 2 + 5^(2 / 3) :=
by
  intro c d
  have c_def : c = log 8 := rfl
  have d_def : d = log 25 := rfl
  rw [c_def, d_def]
  sorry

end NUMINAMATH_GPT_proof_problem_l2300_230053


namespace NUMINAMATH_GPT_volume_tetrahedron_ABCD_l2300_230006

noncomputable def volume_of_tetrahedron (AB CD distance angle : ℝ) : ℝ :=
  (1 / 3) * ((1 / 2) * AB * CD * Real.sin angle) * distance

theorem volume_tetrahedron_ABCD :
  volume_of_tetrahedron 1 (Real.sqrt 3) 2 (Real.pi / 3) = 1 / 2 :=
by
  unfold volume_of_tetrahedron
  sorry

end NUMINAMATH_GPT_volume_tetrahedron_ABCD_l2300_230006


namespace NUMINAMATH_GPT_horner_method_value_at_neg1_l2300_230059

theorem horner_method_value_at_neg1 : 
  let f (x : ℤ) := 4 * x ^ 4 + 3 * x ^ 3 - 6 * x ^ 2 + x - 1
  let x := -1
  let v0 := 4
  let v1 := v0 * x + 3
  let v2 := v1 * x - 6
  v2 = -5 := by
  sorry

end NUMINAMATH_GPT_horner_method_value_at_neg1_l2300_230059


namespace NUMINAMATH_GPT_repeating_decimal_fraction_l2300_230039

noncomputable def repeating_decimal := 4.66666 -- Assuming repeating forever

theorem repeating_decimal_fraction : repeating_decimal = 14 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_l2300_230039


namespace NUMINAMATH_GPT_bounded_roots_l2300_230023

open Polynomial

noncomputable def P : ℤ[X] := sorry -- Replace with actual polynomial if necessary

theorem bounded_roots (P : ℤ[X]) (n : ℕ) (hPdeg : P.degree = n) (hdec : 1 ≤ n) :
  ∀ k : ℤ, (P.eval k) ^ 2 = 1 → ∃ (r s : ℕ), r + s ≤ n + 2 := 
by 
  sorry

end NUMINAMATH_GPT_bounded_roots_l2300_230023
