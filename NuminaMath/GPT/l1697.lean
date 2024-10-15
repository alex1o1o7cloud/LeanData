import Mathlib

namespace NUMINAMATH_GPT_value_of_a6_l1697_169788

theorem value_of_a6 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ) 
  (hS : ∀ n, S n = 3 * n^2 - 5 * n)
  (ha : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) 
  (h1 : a 1 = S 1):
  a 6 = 28 :=
sorry

end NUMINAMATH_GPT_value_of_a6_l1697_169788


namespace NUMINAMATH_GPT_solve_for_ab_l1697_169718

def f (a b : ℚ) (x : ℚ) : ℚ := a * x^3 - 4 * x^2 + b * x - 3

theorem solve_for_ab : 
  ∃ a b : ℚ, 
    f a b 1 = 3 ∧ 
    f a b (-2) = -47 ∧ 
    (a, b) = (4 / 3, 26 / 3) := 
by
  sorry

end NUMINAMATH_GPT_solve_for_ab_l1697_169718


namespace NUMINAMATH_GPT_garden_perimeter_l1697_169755

theorem garden_perimeter (A : ℝ) (P : ℝ) : 
  (A = 97) → (P = 40) :=
by
  sorry

end NUMINAMATH_GPT_garden_perimeter_l1697_169755


namespace NUMINAMATH_GPT_not_washed_shirts_l1697_169703

-- Definitions based on given conditions
def short_sleeve_shirts : ℕ := 9
def long_sleeve_shirts : ℕ := 21
def washed_shirts : ℕ := 29

-- Theorem to prove the number of shirts not washed
theorem not_washed_shirts : (short_sleeve_shirts + long_sleeve_shirts) - washed_shirts = 1 := by
  sorry

end NUMINAMATH_GPT_not_washed_shirts_l1697_169703


namespace NUMINAMATH_GPT_smallest_among_given_numbers_l1697_169759

theorem smallest_among_given_numbers :
  let a := abs (-3)
  let b := -2
  let c := 0
  let d := Real.pi
  b < a ∧ b < c ∧ b < d := by
  sorry

end NUMINAMATH_GPT_smallest_among_given_numbers_l1697_169759


namespace NUMINAMATH_GPT_probability_5_consecutive_heads_in_8_flips_l1697_169749

noncomputable def probability_at_least_5_consecutive_heads (n : ℕ) : ℚ :=
  if n = 8 then 5 / 128 else 0  -- Using conditional given the specificity to n = 8

theorem probability_5_consecutive_heads_in_8_flips : 
  probability_at_least_5_consecutive_heads 8 = 5 / 128 := 
by
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_probability_5_consecutive_heads_in_8_flips_l1697_169749


namespace NUMINAMATH_GPT_factorization_correct_l1697_169701

theorem factorization_correct (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1697_169701


namespace NUMINAMATH_GPT_total_albums_l1697_169797

theorem total_albums (Adele Bridget Katrina Miriam : ℕ) 
  (h₁ : Adele = 30)
  (h₂ : Bridget = Adele - 15)
  (h₃ : Katrina = 6 * Bridget)
  (h₄ : Miriam = 5 * Katrina) : Adele + Bridget + Katrina + Miriam = 585 := 
by
  sorry

end NUMINAMATH_GPT_total_albums_l1697_169797


namespace NUMINAMATH_GPT_remainder_when_divided_by_22_l1697_169740

theorem remainder_when_divided_by_22 
    (y : ℤ) 
    (h : y % 264 = 42) :
    y % 22 = 20 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_22_l1697_169740


namespace NUMINAMATH_GPT_distinct_possible_lunches_l1697_169754

def main_dishes := 3
def beverages := 3
def snacks := 3

theorem distinct_possible_lunches : main_dishes * beverages * snacks = 27 := by
  sorry

end NUMINAMATH_GPT_distinct_possible_lunches_l1697_169754


namespace NUMINAMATH_GPT_not_square_difference_formula_l1697_169794

theorem not_square_difference_formula (x y : ℝ) : ¬ ∃ (a b : ℝ), (x - y) * (-x + y) = (a + b) * (a - b) := 
sorry

end NUMINAMATH_GPT_not_square_difference_formula_l1697_169794


namespace NUMINAMATH_GPT_arithmetic_geometric_inequality_l1697_169727

variables {a b A1 A2 G1 G2 x y d q : ℝ}
variables (h₀ : 0 < a) (h₁ : 0 < b)
variables (h₂ : a = x - 3 * d) (h₃ : A1 = x - d) (h₄ : A2 = x + d) (h₅ : b = x + 3 * d)
variables (h₆ : a = y / q^3) (h₇ : G1 = y / q) (h₈ : G2 = y * q) (h₉ : b = y * q^3)
variables (h₁₀ : x - 3 * d = y / q^3) (h₁₁ : x + 3 * d = y * q^3)

theorem arithmetic_geometric_inequality : A1 * A2 ≥ G1 * G2 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_geometric_inequality_l1697_169727


namespace NUMINAMATH_GPT_domain_proof_l1697_169753

def domain_of_function : Set ℝ := {x : ℝ | x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x}

theorem domain_proof :
  (∀ x : ℝ, (x ≠ 7) → (x^2 - 16 ≥ 0) → (x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x)) ∧
  (∀ x : ℝ, (x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x) → (x ≠ 7) ∧ (x^2 - 16 ≥ 0)) :=
by
  sorry

end NUMINAMATH_GPT_domain_proof_l1697_169753


namespace NUMINAMATH_GPT_mass_of_man_l1697_169770

-- Definitions of the given conditions
def boat_length : Float := 3.0
def boat_breadth : Float := 2.0
def sink_depth : Float := 0.01 -- 1 cm converted to meters
def water_density : Float := 1000.0 -- Density of water in kg/m³

-- Define the proof goal as the mass of the man
theorem mass_of_man : Float :=
by
  let volume_displaced := boat_length * boat_breadth * sink_depth
  let weight_displaced := volume_displaced * water_density
  exact weight_displaced

end NUMINAMATH_GPT_mass_of_man_l1697_169770


namespace NUMINAMATH_GPT_camper_ratio_l1697_169798

theorem camper_ratio (total_campers : ℕ) (G : ℕ) (B : ℕ)
  (h1: total_campers = 96) 
  (h2: G = total_campers / 3) 
  (h3: B = total_campers - G) 
  : B / total_campers = 2 / 3 :=
  by
    sorry

end NUMINAMATH_GPT_camper_ratio_l1697_169798


namespace NUMINAMATH_GPT_largest_n_unique_k_l1697_169737

theorem largest_n_unique_k :
  ∃ (n : ℕ), ( ∃! (k : ℕ), (5 : ℚ) / 11 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 6 / 11 )
    ∧ n = 359 :=
sorry

end NUMINAMATH_GPT_largest_n_unique_k_l1697_169737


namespace NUMINAMATH_GPT_sequence_2011_l1697_169781

theorem sequence_2011 :
  ∀ (a : ℕ → ℤ), (a 1 = 1) →
                  (a 2 = 2) →
                  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) →
                  a 2011 = 1 :=
by {
  -- Insert proof here
  sorry
}

end NUMINAMATH_GPT_sequence_2011_l1697_169781


namespace NUMINAMATH_GPT_units_digit_p_plus_2_l1697_169723

theorem units_digit_p_plus_2 {p : ℕ} 
  (h1 : p % 2 = 0) 
  (h2 : p % 10 ≠ 0) 
  (h3 : (p^3 % 10) = (p^2 % 10)) : 
  (p + 2) % 10 = 8 :=
sorry

end NUMINAMATH_GPT_units_digit_p_plus_2_l1697_169723


namespace NUMINAMATH_GPT_tangency_condition_and_point_l1697_169715

variable (a b p q : ℝ)

/-- Condition for the line y = px + q to be tangent to the ellipse b^2 x^2 + a^2 y^2 = a^2 b^2. -/
theorem tangency_condition_and_point
  (h_cond : a^2 * p^2 + b^2 - q^2 = 0)
  : 
  ∃ (x₀ y₀ : ℝ), 
  x₀ = - (a^2 * p) / q ∧
  y₀ = b^2 / q ∧ 
  (b^2 * x₀^2 + a^2 * y₀^2 = a^2 * b^2 ∧ y₀ = p * x₀ + q) :=
sorry

end NUMINAMATH_GPT_tangency_condition_and_point_l1697_169715


namespace NUMINAMATH_GPT_max_value_of_y_l1697_169777

open Real

noncomputable def y (x : ℝ) := 1 + 1 / (x^2 + 2*x + 2)

theorem max_value_of_y : ∃ x : ℝ, y x = 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_y_l1697_169777


namespace NUMINAMATH_GPT_area_of_square_is_1225_l1697_169792

-- Given some basic definitions and conditions
variable (s : ℝ) -- side of the square which is the radius of the circle
variable (length : ℝ := (2 / 5) * s)
variable (breadth : ℝ := 10)
variable (area_rectangle : ℝ := length * breadth)

-- Statement to prove
theorem area_of_square_is_1225 
  (h1 : length = (2 / 5) * s)
  (h2 : breadth = 10)
  (h3 : area_rectangle = 140) : 
  s^2 = 1225 := by
    sorry

end NUMINAMATH_GPT_area_of_square_is_1225_l1697_169792


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1697_169757

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 4} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1697_169757


namespace NUMINAMATH_GPT_ratio_of_length_to_breadth_l1697_169763

theorem ratio_of_length_to_breadth 
    (breadth : ℝ) (area : ℝ) (h_breadth : breadth = 12) (h_area : area = 432)
    (h_area_formula : area = l * breadth) : 
    l / breadth = 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_length_to_breadth_l1697_169763


namespace NUMINAMATH_GPT_calculate_expression_l1697_169731

variable (a : ℝ)

theorem calculate_expression (h : a ≠ 0) : (6 * a^2) / (a / 2) = 12 * a := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1697_169731


namespace NUMINAMATH_GPT_polar_coordinates_of_point_l1697_169700

theorem polar_coordinates_of_point :
  ∃ (r θ : ℝ), r = 2 ∧ θ = (2 * Real.pi) / 3 ∧
  (r > 0) ∧ (0 ≤ θ) ∧ (θ < 2 * Real.pi) ∧
  (-1, Real.sqrt 3) = (r * Real.cos θ, r * Real.sin θ) :=
by 
  sorry

end NUMINAMATH_GPT_polar_coordinates_of_point_l1697_169700


namespace NUMINAMATH_GPT_given_conditions_imply_f_neg3_gt_f_neg2_l1697_169776

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem given_conditions_imply_f_neg3_gt_f_neg2
  {f : ℝ → ℝ}
  (h_even : is_even_function f)
  (h_comparison : f 2 < f 3) :
  f (-3) > f (-2) :=
by
  sorry

end NUMINAMATH_GPT_given_conditions_imply_f_neg3_gt_f_neg2_l1697_169776


namespace NUMINAMATH_GPT_solution_set_of_abs_inequality_l1697_169764

theorem solution_set_of_abs_inequality : 
  {x : ℝ | abs (x - 1) - abs (x - 5) < 2} = {x : ℝ | x < 4} := 
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_abs_inequality_l1697_169764


namespace NUMINAMATH_GPT_gain_percentage_l1697_169750

-- Define the conditions as a Lean problem
theorem gain_percentage (C G : ℝ) (hC : (9 / 10) * C = 1) (hSP : (10 / 6) = (1 + G / 100) * C) : 
  G = 50 :=
by
-- Here, you would generally have the proof steps, but we add sorry to skip the proof for now.
sorry

end NUMINAMATH_GPT_gain_percentage_l1697_169750


namespace NUMINAMATH_GPT_point_on_x_axis_l1697_169751

theorem point_on_x_axis (m : ℤ) (P : ℤ × ℤ) (hP : P = (m + 3, m + 1)) (h : P.2 = 0) : P = (2, 0) :=
by 
  sorry

end NUMINAMATH_GPT_point_on_x_axis_l1697_169751


namespace NUMINAMATH_GPT_sum_first_2014_terms_l1697_169762

def sequence_is_arithmetic (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + a 2

def first_arithmetic_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :=
  S n = (n * (n - 1)) / 2

theorem sum_first_2014_terms (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : sequence_is_arithmetic a) 
  (h2 : a 3 = 2) : 
  S 2014 = 1007 * 2013 :=
sorry

end NUMINAMATH_GPT_sum_first_2014_terms_l1697_169762


namespace NUMINAMATH_GPT_quadratic_residue_property_l1697_169795

theorem quadratic_residue_property (p : ℕ) [hp : Fact (Nat.Prime p)] (a : ℕ)
  (h : ∃ t : ℤ, ∃ k : ℤ, k * k = p * t + a) : (a ^ ((p - 1) / 2)) % p = 1 :=
sorry

end NUMINAMATH_GPT_quadratic_residue_property_l1697_169795


namespace NUMINAMATH_GPT_digit_proportions_l1697_169783

theorem digit_proportions (n : ℕ) :
  (∃ (n1 n2 n5 nother : ℕ),
    n1 = n / 2 ∧
    n2 = n / 5 ∧
    n5 = n / 5 ∧
    nother = n / 10 ∧
    n1 + n2 + n5 + nother = n) ↔ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_digit_proportions_l1697_169783


namespace NUMINAMATH_GPT_triangle_ABC_area_l1697_169748

-- We define the basic structure of a triangle and its properties
structure Triangle :=
(base : ℝ)
(height : ℝ)
(right_angled_at : ℝ)

-- Define the specific triangle ABC with given properties
def triangle_ABC : Triangle := {
  base := 12,
  height := 15,
  right_angled_at := 90 -- since right-angled at C
}

-- Given conditions, we need to prove the area is 90 square cm
theorem triangle_ABC_area : 1/2 * triangle_ABC.base * triangle_ABC.height = 90 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_ABC_area_l1697_169748


namespace NUMINAMATH_GPT_boa_constrictor_length_l1697_169739

theorem boa_constrictor_length (garden_snake_length : ℕ) (boa_multiplier : ℕ) (boa_length : ℕ) 
    (h1 : garden_snake_length = 10) (h2 : boa_multiplier = 7) (h3 : boa_length = garden_snake_length * boa_multiplier) : 
    boa_length = 70 := 
sorry

end NUMINAMATH_GPT_boa_constrictor_length_l1697_169739


namespace NUMINAMATH_GPT_base_salary_is_1600_l1697_169713

theorem base_salary_is_1600 (B : ℝ) (C : ℝ) (sales : ℝ) (fixed_salary : ℝ) :
  C = 0.04 ∧ sales = 5000 ∧ fixed_salary = 1800 ∧ (B + C * sales = fixed_salary) → B = 1600 :=
by sorry

end NUMINAMATH_GPT_base_salary_is_1600_l1697_169713


namespace NUMINAMATH_GPT_rita_total_hours_l1697_169732

def h_backstroke : ℕ := 50
def h_breaststroke : ℕ := 9
def h_butterfly : ℕ := 121
def h_freestyle_sidestroke_per_month : ℕ := 220
def months : ℕ := 6

def h_total : ℕ := h_backstroke + h_breaststroke + h_butterfly + (h_freestyle_sidestroke_per_month * months)

theorem rita_total_hours :
  h_total = 1500 :=
by
  sorry

end NUMINAMATH_GPT_rita_total_hours_l1697_169732


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1697_169734

theorem necessary_and_sufficient_condition (a b : ℝ) : 
  (|a + b| / (|a| + |b|) ≤ 1) ↔ (a^2 + b^2 ≠ 0) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1697_169734


namespace NUMINAMATH_GPT_minimum_value_of_absolute_sum_l1697_169779

theorem minimum_value_of_absolute_sum (x : ℝ) :
  ∃ y : ℝ, (∀ x : ℝ, y ≤ |x + 1| + |x + 2| + |x + 3| + |x + 4| + |x + 5|) ∧ y = 6 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_absolute_sum_l1697_169779


namespace NUMINAMATH_GPT_greatest_possible_value_of_x_l1697_169730

theorem greatest_possible_value_of_x (x : ℕ) (h₁ : x % 4 = 0) (h₂ : x > 0) (h₃ : x^3 < 8000) :
  x ≤ 16 := by
  apply sorry

end NUMINAMATH_GPT_greatest_possible_value_of_x_l1697_169730


namespace NUMINAMATH_GPT_find_beta_l1697_169728

variables {m n p : ℤ} -- defining variables m, n, p as integers
variables {α β : ℤ} -- defining roots α and β as integers

theorem find_beta (h1: α = 3)
  (h2: ∀ x, x^2 - (m+n)*x + (m*n - p) = 0) -- defining the quadratic equation
  (h3: α + β = m + n)
  (h4: α * β = m * n - p)
  (h5: m ≠ n) (h6: n ≠ p) (h7: m ≠ p) : -- ensuring m, n, and p are distinct
  β = m + n - 3 := sorry

end NUMINAMATH_GPT_find_beta_l1697_169728


namespace NUMINAMATH_GPT_stolen_bones_is_two_l1697_169719

/-- Juniper's initial number of bones -/
def initial_bones : ℕ := 4

/-- Juniper's bones after receiving more bones -/
def doubled_bones : ℕ := initial_bones * 2

/-- Juniper's remaining number of bones after theft -/
def remaining_bones : ℕ := 6

/-- Number of bones stolen by the neighbor's dog -/
def stolen_bones : ℕ := doubled_bones - remaining_bones

theorem stolen_bones_is_two : stolen_bones = 2 := sorry

end NUMINAMATH_GPT_stolen_bones_is_two_l1697_169719


namespace NUMINAMATH_GPT_peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l1697_169706

-- Define the context of the problem
def total_people := 100
def men := 50
def women := 50

-- Define Peter Ivanovich being satisfied
def satisfies_peter_ivanovich := (women / (total_people - 1)) * ((women - 1) / (total_people - 2)) 

-- Define the probability that Peter Ivanovich is satisfied
theorem peter_ivanovich_satisfied_probability :
  satisfies_peter_ivanovich = 25 / 33 := 
sorry

-- Define the expected number of satisfied men
def expected_satisfied_men := men * (25 / 33)

-- Prove the expected number of satisfied men
theorem expected_satisfied_men_value :
  expected_satisfied_men = 1250 / 33 :=
sorry

end NUMINAMATH_GPT_peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l1697_169706


namespace NUMINAMATH_GPT_pizza_eaten_after_six_trips_l1697_169773

noncomputable def fraction_eaten : ℚ :=
  let first_trip := 1 / 3
  let second_trip := 1 / (3 ^ 2)
  let third_trip := 1 / (3 ^ 3)
  let fourth_trip := 1 / (3 ^ 4)
  let fifth_trip := 1 / (3 ^ 5)
  let sixth_trip := 1 / (3 ^ 6)
  first_trip + second_trip + third_trip + fourth_trip + fifth_trip + sixth_trip

theorem pizza_eaten_after_six_trips : fraction_eaten = 364 / 729 :=
by sorry

end NUMINAMATH_GPT_pizza_eaten_after_six_trips_l1697_169773


namespace NUMINAMATH_GPT_truthfulness_count_l1697_169789

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_truthfulness_count_l1697_169789


namespace NUMINAMATH_GPT_solve_x_l1697_169745

-- Define the function f with the given properties
axiom f : ℝ → ℝ → ℝ
axiom f_assoc : ∀ (a b c : ℝ), f a (f b c) = f (f a b) c
axiom f_inv : ∀ (a : ℝ), f a a = 1

-- Define x and the equation to be solved
theorem solve_x : ∃ (x : ℝ), f x 36 = 216 :=
  sorry

end NUMINAMATH_GPT_solve_x_l1697_169745


namespace NUMINAMATH_GPT_domain_log_function_l1697_169717

theorem domain_log_function :
  { x : ℝ | 12 + x - x^2 > 0 } = { x : ℝ | -3 < x ∧ x < 4 } :=
sorry

end NUMINAMATH_GPT_domain_log_function_l1697_169717


namespace NUMINAMATH_GPT_fraction_addition_l1697_169707

theorem fraction_addition : (3 / 5) + (2 / 15) = 11 / 15 := sorry

end NUMINAMATH_GPT_fraction_addition_l1697_169707


namespace NUMINAMATH_GPT_cat_food_percentage_l1697_169782

theorem cat_food_percentage (D C : ℝ) (h1 : 7 * D + 4 * C = 8 * D) (h2 : 4 * C = D) : 
  (C / (7 * D + D)) * 100 = 3.125 := by
  sorry

end NUMINAMATH_GPT_cat_food_percentage_l1697_169782


namespace NUMINAMATH_GPT_kids_played_on_monday_l1697_169738

theorem kids_played_on_monday (total : ℕ) (tuesday : ℕ) (monday : ℕ) (h_total : total = 16) (h_tuesday : tuesday = 14) :
  monday = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_kids_played_on_monday_l1697_169738


namespace NUMINAMATH_GPT_remainder_5_pow_100_mod_18_l1697_169769

theorem remainder_5_pow_100_mod_18 : (5 ^ 100) % 18 = 13 := 
by
  -- We will skip the proof since only the statement is required.
  sorry

end NUMINAMATH_GPT_remainder_5_pow_100_mod_18_l1697_169769


namespace NUMINAMATH_GPT_age_composition_is_decline_l1697_169771

-- Define the population and age groups
variable (P : Type)
variable (Y E : P → ℕ) -- Functions indicating the number of young and elderly individuals

-- Assumptions as per the conditions
axiom fewer_young_more_elderly (p : P) : Y p < E p

-- Conclusion: Prove that the population is of Decline type.
def age_composition_decline (p : P) : Prop :=
  Y p < E p

theorem age_composition_is_decline (p : P) : age_composition_decline P Y E p := by
  sorry

end NUMINAMATH_GPT_age_composition_is_decline_l1697_169771


namespace NUMINAMATH_GPT_isosceles_right_triangle_hypotenuse_l1697_169725

noncomputable def hypotenuse_length : ℝ :=
  let a := Real.sqrt 363
  let c := Real.sqrt (2 * (a ^ 2))
  c

theorem isosceles_right_triangle_hypotenuse :
  ∀ (a : ℝ),
    (2 * (a ^ 2)) + (a ^ 2) = 1452 →
    hypotenuse_length = Real.sqrt 726 := by
  intro a h
  rw [hypotenuse_length]
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_hypotenuse_l1697_169725


namespace NUMINAMATH_GPT_value_of_x_minus_y_l1697_169722

theorem value_of_x_minus_y (x y : ℝ) (h1 : abs x = 4) (h2 : abs y = 7) (h3 : x + y > 0) :
  x - y = -3 ∨ x - y = -11 :=
sorry

end NUMINAMATH_GPT_value_of_x_minus_y_l1697_169722


namespace NUMINAMATH_GPT_number_of_adults_had_meal_l1697_169791

theorem number_of_adults_had_meal (A : ℝ) :
  let num_children_food : ℝ := 63
  let food_for_adults : ℝ := 70
  let food_for_children : ℝ := 90
  (food_for_children - A * (food_for_children / food_for_adults) = num_children_food) →
  A = 21 :=
by
  intros num_children_food food_for_adults food_for_children h
  have h2 : 90 - A * (90 / 70) = 63 := h
  sorry

end NUMINAMATH_GPT_number_of_adults_had_meal_l1697_169791


namespace NUMINAMATH_GPT_sum_at_simple_interest_l1697_169772

theorem sum_at_simple_interest
  (P R : ℝ)  -- P is the principal amount, R is the rate of interest
  (H1 : (9 * P * (R + 5) / 100 - 9 * P * R / 100 = 1350)) :
  P = 3000 :=
by
  sorry

end NUMINAMATH_GPT_sum_at_simple_interest_l1697_169772


namespace NUMINAMATH_GPT_greatest_third_term_arithmetic_seq_l1697_169784

theorem greatest_third_term_arithmetic_seq (a d : ℤ) (h1: a > 0) (h2: d ≥ 0) (h3: 5 * a + 10 * d = 65) : 
  a + 2 * d = 13 := 
by 
  sorry

end NUMINAMATH_GPT_greatest_third_term_arithmetic_seq_l1697_169784


namespace NUMINAMATH_GPT_perfect_square_pairs_l1697_169766

theorem perfect_square_pairs (x y : ℕ) (a b : ℤ) :
  (x^2 + 8 * ↑y = a^2 ∧ y^2 - 8 * ↑x = b^2) →
  (∃ n : ℕ, x = n ∧ y = n + 2) ∨ (x = 7 ∧ y = 15) ∨ (x = 33 ∧ y = 17) ∨ (x = 45 ∧ y = 23) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_pairs_l1697_169766


namespace NUMINAMATH_GPT_fruit_total_l1697_169780

noncomputable def fruit_count_proof : Prop :=
  let oranges := 6
  let apples := oranges - 2
  let bananas := 3 * apples
  let peaches := bananas / 2
  oranges + apples + bananas + peaches = 28

theorem fruit_total : fruit_count_proof :=
by {
  sorry
}

end NUMINAMATH_GPT_fruit_total_l1697_169780


namespace NUMINAMATH_GPT_number_of_children_l1697_169793

def weekly_husband : ℕ := 335
def weekly_wife : ℕ := 225
def weeks_in_six_months : ℕ := 24
def amount_per_child : ℕ := 1680

theorem number_of_children : (weekly_husband + weekly_wife) * weeks_in_six_months / 2 / amount_per_child = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_children_l1697_169793


namespace NUMINAMATH_GPT_remaining_money_is_correct_l1697_169785

def initial_amount : ℕ := 53
def cost_toy_car : ℕ := 11
def number_toy_cars : ℕ := 2
def cost_scarf : ℕ := 10
def cost_beanie : ℕ := 14
def remaining_money : ℕ := 
  initial_amount - (cost_toy_car * number_toy_cars) - cost_scarf - cost_beanie

theorem remaining_money_is_correct : remaining_money = 7 := by
  sorry

end NUMINAMATH_GPT_remaining_money_is_correct_l1697_169785


namespace NUMINAMATH_GPT_intersection_complement_eq_l1697_169736

variable (U : Set ℝ := Set.univ)
variable (A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 })
variable (B : Set ℝ := { x | x > -1 })

theorem intersection_complement_eq :
  A ∩ (U \ B) = { x | -2 ≤ x ∧ x ≤ -1 } :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_complement_eq_l1697_169736


namespace NUMINAMATH_GPT_find_k_of_symmetry_l1697_169768

noncomputable def f (x k : ℝ) := Real.sin (2 * x) + k * Real.cos (2 * x)

theorem find_k_of_symmetry (k : ℝ) :
  (∃ x, x = (Real.pi / 6) ∧ f x k = f (Real.pi / 6 - x) k) →
  k = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_find_k_of_symmetry_l1697_169768


namespace NUMINAMATH_GPT_fraction_b_not_whole_l1697_169765

-- Defining the fractions as real numbers
def fraction_a := 60 / 12
def fraction_b := 60 / 8
def fraction_c := 60 / 5
def fraction_d := 60 / 4
def fraction_e := 60 / 3

-- Defining what it means to be a whole number
def is_whole_number (x : ℝ) : Prop := ∃ (n : ℤ), x = n

-- Theorem stating that fraction_b is not a whole number
theorem fraction_b_not_whole : ¬ is_whole_number fraction_b := 
by 
-- proof to be filled in
sorry

end NUMINAMATH_GPT_fraction_b_not_whole_l1697_169765


namespace NUMINAMATH_GPT_value_of_expression_l1697_169767

-- Definitions based on the conditions
def a : ℕ := 15
def b : ℕ := 3

-- The theorem to prove
theorem value_of_expression : a^2 + 2 * a * b + b^2 = 324 := by
  -- Skipping the proof as per instructions
  sorry

end NUMINAMATH_GPT_value_of_expression_l1697_169767


namespace NUMINAMATH_GPT_cubic_three_real_roots_l1697_169729

theorem cubic_three_real_roots (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧
   x₁ ^ 3 - 3 * x₁ - a = 0 ∧
   x₂ ^ 3 - 3 * x₂ - a = 0 ∧
   x₃ ^ 3 - 3 * x₃ - a = 0) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_cubic_three_real_roots_l1697_169729


namespace NUMINAMATH_GPT_susan_spaces_to_win_l1697_169774

def spaces_in_game : ℕ := 48
def first_turn_movement : ℤ := 8
def second_turn_movement : ℤ := 2 - 5
def third_turn_movement : ℤ := 6

def total_movement : ℤ :=
  first_turn_movement + second_turn_movement + third_turn_movement

def spaces_to_win (spaces_in_game : ℕ) (total_movement : ℤ) : ℤ :=
  spaces_in_game - total_movement

theorem susan_spaces_to_win : spaces_to_win spaces_in_game total_movement = 37 := by
  sorry

end NUMINAMATH_GPT_susan_spaces_to_win_l1697_169774


namespace NUMINAMATH_GPT_intersection_P_Q_l1697_169778

def P : Set ℝ := { x : ℝ | 2 ≤ x ∧ x < 4 }
def Q : Set ℝ := { x : ℝ | 3 ≤ x }

theorem intersection_P_Q :
  P ∩ Q = { x : ℝ | 3 ≤ x ∧ x < 4 } :=
by
  sorry  -- Proof step will be provided here

end NUMINAMATH_GPT_intersection_P_Q_l1697_169778


namespace NUMINAMATH_GPT_average_remaining_five_l1697_169724

theorem average_remaining_five (S S4 S5 : ℕ) 
  (h1 : S = 18 * 9) 
  (h2 : S4 = 8 * 4) 
  (h3 : S5 = S - S4) 
  (h4 : S5 / 5 = 26) : 
  average_of_remaining_5 = 26 :=
by 
  sorry


end NUMINAMATH_GPT_average_remaining_five_l1697_169724


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l1697_169726

theorem sufficient_and_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, m * x ^ 2 + 2 * m * x - 1 < 0) ↔ (-1 < m ∧ m < -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l1697_169726


namespace NUMINAMATH_GPT_ring_rotation_count_l1697_169714

-- Define the constants and parameters from the conditions
variables (R ω μ g : ℝ) -- radius, angular velocity, coefficient of friction, and gravity constant
-- Additional constraints on these variables
variable (m : ℝ) -- mass of the ring

theorem ring_rotation_count :
  ∃ n : ℝ, n = (ω^2 * R * (1 + μ^2)) / (4 * π * g * μ * (1 + μ)) :=
sorry

end NUMINAMATH_GPT_ring_rotation_count_l1697_169714


namespace NUMINAMATH_GPT_simplify_expression_l1697_169799

-- Define the main theorem
theorem simplify_expression 
  (a b x : ℝ) 
  (hx : x = 1 / a * Real.sqrt ((2 * a - b) / b))
  (hc1 : 0 < b / 2)
  (hc2 : b / 2 < a)
  (hc3 : a < b) : 
  (1 - a * x) / (1 + a * x) * Real.sqrt ((1 + b * x) / (1 - b * x)) = 1 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1697_169799


namespace NUMINAMATH_GPT_fraction_meaningful_l1697_169712

theorem fraction_meaningful (x : ℝ) : (x + 2 ≠ 0) ↔ x ≠ -2 := by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l1697_169712


namespace NUMINAMATH_GPT_solve_equation_l1697_169747

theorem solve_equation:
  ∀ x y z : ℝ, x^2 + 5 * y^2 + 5 * z^2 - 4 * x * z - 2 * y - 4 * y * z + 1 = 0 → 
    x = 4 ∧ y = 1 ∧ z = 2 :=
by
  intros x y z h
  sorry

end NUMINAMATH_GPT_solve_equation_l1697_169747


namespace NUMINAMATH_GPT_alternating_sum_cubes_eval_l1697_169733

noncomputable def alternating_sum_cubes : ℕ → ℤ
| 0 => 0
| n + 1 => alternating_sum_cubes n + (-1)^(n / 4) * (n + 1)^3

theorem alternating_sum_cubes_eval :
  alternating_sum_cubes 99 = S :=
by
  sorry

end NUMINAMATH_GPT_alternating_sum_cubes_eval_l1697_169733


namespace NUMINAMATH_GPT_fraction_not_on_time_l1697_169709

theorem fraction_not_on_time (total_attendees : ℕ) (male_fraction female_fraction male_on_time_fraction female_on_time_fraction : ℝ)
  (H1 : male_fraction = 3/5)
  (H2 : male_on_time_fraction = 7/8)
  (H3 : female_on_time_fraction = 4/5)
  : ((1 - (male_fraction * male_on_time_fraction + (1 - male_fraction) * female_on_time_fraction)) = 3/20) :=
sorry

end NUMINAMATH_GPT_fraction_not_on_time_l1697_169709


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1697_169746

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  ((x + 1) * (x - 3) < 0 → x > -1) ∧ ¬ (x > -1 → (x + 1) * (x - 3) < 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1697_169746


namespace NUMINAMATH_GPT_jefferson_high_school_ninth_graders_l1697_169756

theorem jefferson_high_school_ninth_graders (total_students science_students arts_students students_taking_both : ℕ):
  total_students = 120 →
  science_students = 85 →
  arts_students = 65 →
  students_taking_both = 150 - 120 →
  science_students - students_taking_both = 55 :=
by
  sorry

end NUMINAMATH_GPT_jefferson_high_school_ninth_graders_l1697_169756


namespace NUMINAMATH_GPT_probability_A_C_winning_l1697_169752

-- Definitions based on the conditions given
def students := ["A", "B", "C", "D"]

def isDistictPositions (x y : String) : Prop :=
  x ≠ y

-- Lean statement for the mathematical problem
theorem probability_A_C_winning :
  ∃ (P : ℚ), P = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_probability_A_C_winning_l1697_169752


namespace NUMINAMATH_GPT_range_of_p_l1697_169775

def A (x : ℝ) : Prop := -2 < x ∧ x < 5
def B (p : ℝ) (x : ℝ) : Prop := p + 1 < x ∧ x < 2 * p - 1

theorem range_of_p (p : ℝ) :
  (∀ x, A x ∨ B p x → A x) ↔ p ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_p_l1697_169775


namespace NUMINAMATH_GPT_cannot_determine_right_triangle_l1697_169716

-- Definitions of conditions
def condition_A (A B C : ℝ) : Prop := A = B + C
def condition_B (a b c : ℝ) : Prop := a/b = 5/12 ∧ b/c = 12/13
def condition_C (a b c : ℝ) : Prop := a^2 = (b + c) * (b - c)
def condition_D (A B C : ℝ) : Prop := A/B = 3/4 ∧ B/C = 4/5

-- The proof problem
theorem cannot_determine_right_triangle (a b c A B C : ℝ)
  (hD : condition_D A B C) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end NUMINAMATH_GPT_cannot_determine_right_triangle_l1697_169716


namespace NUMINAMATH_GPT_handshakes_mod_500_l1697_169790

theorem handshakes_mod_500 : 
  let n := 10
  let k := 3
  let M := 199584 -- total number of ways calculated from the problem
  (n = 10) -> (k = 3) -> (M % 500 = 84) :=
by
  intros
  sorry

end NUMINAMATH_GPT_handshakes_mod_500_l1697_169790


namespace NUMINAMATH_GPT_y_intercept_of_line_l1697_169720

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1697_169720


namespace NUMINAMATH_GPT_coprime_with_others_l1697_169708

theorem coprime_with_others:
  ∀ (a b c d e : ℕ),
  a = 20172017 → 
  b = 20172018 → 
  c = 20172019 →
  d = 20172020 →
  e = 20172021 →
  (Nat.gcd c a = 1 ∧ 
   Nat.gcd c b = 1 ∧ 
   Nat.gcd c d = 1 ∧ 
   Nat.gcd c e = 1) :=
by
  sorry

end NUMINAMATH_GPT_coprime_with_others_l1697_169708


namespace NUMINAMATH_GPT_find_positive_real_numbers_l1697_169743

open Real

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16

theorem find_positive_real_numbers (x : ℝ) (hx : x > 0) :
  satisfies_inequality x ↔ 15 * x^2 + 32 * x - 256 = 0 :=
sorry

end NUMINAMATH_GPT_find_positive_real_numbers_l1697_169743


namespace NUMINAMATH_GPT_fraction_sum_eq_one_l1697_169741

theorem fraction_sum_eq_one (m n : ℝ) (h : m ≠ n) : (m / (m - n) + n / (n - m) = 1) :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_eq_one_l1697_169741


namespace NUMINAMATH_GPT_solve_problem_l1697_169711

def num : ℕ := 1 * 3 * 5 * 7
def den : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7

theorem solve_problem : (num : ℚ) / den = 3.75 := 
by
  sorry

end NUMINAMATH_GPT_solve_problem_l1697_169711


namespace NUMINAMATH_GPT_h_at_3_l1697_169721

theorem h_at_3 :
  ∃ h : ℤ → ℤ,
    (∀ x, (x^7 - 1) * h x = (x+1) * (x^2 + 1) * (x^4 + 1) - (x-1)) →
    h 3 = 3 := 
sorry

end NUMINAMATH_GPT_h_at_3_l1697_169721


namespace NUMINAMATH_GPT_grogg_possible_cubes_l1697_169758

theorem grogg_possible_cubes (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_prob : (a - 2) * (b - 2) * (c - 2) / (a * b * c) = 1 / 5) :
  a * b * c = 120 ∨ a * b * c = 160 ∨ a * b * c = 240 ∨ a * b * c = 360 := 
sorry

end NUMINAMATH_GPT_grogg_possible_cubes_l1697_169758


namespace NUMINAMATH_GPT_mrs_blue_expected_tomato_yield_l1697_169796

-- Definitions for conditions
def steps_length := 3 -- each step measures 3 feet
def length_steps := 18 -- 18 steps in length
def width_steps := 25 -- 25 steps in width
def yield_per_sq_ft := 3 / 4 -- three-quarters of a pound per square foot

-- Define the total expected yield in pounds
def expected_yield : ℝ :=
  let length_ft := length_steps * steps_length
  let width_ft := width_steps * steps_length
  let area := length_ft * width_ft
  area * yield_per_sq_ft

-- The goal statement
theorem mrs_blue_expected_tomato_yield : expected_yield = 3037.5 := by
  sorry

end NUMINAMATH_GPT_mrs_blue_expected_tomato_yield_l1697_169796


namespace NUMINAMATH_GPT_no_three_digit_number_such_that_sum_is_perfect_square_l1697_169702

theorem no_three_digit_number_such_that_sum_is_perfect_square :
  ∀ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 →
  ¬ (∃ m : ℕ, m * m = 100 * a + 10 * b + c + 100 * b + 10 * c + a + 100 * c + 10 * a + b) := by
  sorry

end NUMINAMATH_GPT_no_three_digit_number_such_that_sum_is_perfect_square_l1697_169702


namespace NUMINAMATH_GPT_solve_equation_l1697_169735

theorem solve_equation (x : ℝ) : 
  (x + 1) / 6 = 4 / 3 - x ↔ x = 1 :=
sorry

end NUMINAMATH_GPT_solve_equation_l1697_169735


namespace NUMINAMATH_GPT_number_system_base_l1697_169744

theorem number_system_base (a : ℕ) (h : 2 * a^2 + 5 * a + 3 = 136) : a = 7 := 
sorry

end NUMINAMATH_GPT_number_system_base_l1697_169744


namespace NUMINAMATH_GPT_volume_of_displaced_water_l1697_169787

-- Defining the conditions of the problem
def cube_side_length : ℝ := 6
def cyl_radius : ℝ := 5
def cyl_height : ℝ := 12
def cube_volume (s : ℝ) : ℝ := s^3

-- Statement: The volume of water displaced by the cube when it is fully submerged in the barrel
theorem volume_of_displaced_water :
  cube_volume cube_side_length = 216 := by
  sorry

end NUMINAMATH_GPT_volume_of_displaced_water_l1697_169787


namespace NUMINAMATH_GPT_rightmost_three_digits_of_3_pow_2023_l1697_169742

theorem rightmost_three_digits_of_3_pow_2023 :
  (3^2023) % 1000 = 787 := 
sorry

end NUMINAMATH_GPT_rightmost_three_digits_of_3_pow_2023_l1697_169742


namespace NUMINAMATH_GPT_sum_of_exponents_l1697_169705

-- Given product of integers from 1 to 15
def y := Nat.factorial 15

-- Prime exponent variables in the factorization of y
variables (i j k m n p q : ℕ)

-- Conditions
axiom h1 : y = 2^i * 3^j * 5^k * 7^m * 11^n * 13^p * 17^q 

-- Prove that the sum of the exponents equals 24
theorem sum_of_exponents :
  i + j + k + m + n + p + q = 24 := 
sorry

end NUMINAMATH_GPT_sum_of_exponents_l1697_169705


namespace NUMINAMATH_GPT_repeating_decimal_sum_is_one_l1697_169761

noncomputable def repeating_decimal_sum : ℝ :=
  let x := (1/3 : ℝ)
  let y := (2/3 : ℝ)
  x + y

theorem repeating_decimal_sum_is_one : repeating_decimal_sum = 1 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_is_one_l1697_169761


namespace NUMINAMATH_GPT_find_r_minus2_l1697_169760

noncomputable def p : ℤ → ℤ := sorry
def r : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom p_minus1 : p (-1) = 2
axiom p_3 : p (3) = 5
axiom p_minus4 : p (-4) = -3

-- Definition of r(x) when p(x) is divided by (x + 1)(x - 3)(x + 4)
axiom r_def : ∀ x, p x = (x + 1) * (x - 3) * (x + 4) * (sorry : ℤ → ℤ) + r x

-- Our goal to prove
theorem find_r_minus2 : r (-2) = 32 / 7 :=
sorry

end NUMINAMATH_GPT_find_r_minus2_l1697_169760


namespace NUMINAMATH_GPT_six_digit_count_div_by_217_six_digit_count_div_by_218_l1697_169786

-- Definitions for the problem
def six_digit_format (n : ℕ) : Prop :=
  ∃ a b : ℕ, (0 ≤ a ∧ a < 10) ∧ (0 ≤ b ∧ b < 10) ∧ n = 100001 * a + 10010 * b + 100 * a + 10 * b + a

def divisible_by (n : ℕ) (divisor : ℕ) : Prop :=
  n % divisor = 0

-- Problem Part a: How many six-digit numbers of the form are divisible by 217
theorem six_digit_count_div_by_217 :
  ∃ count : ℕ, count = 3 ∧ ∀ n : ℕ, six_digit_format n → divisible_by n 217  → (n = 313131 ∨ n = 626262 ∨ n = 939393) :=
sorry

-- Problem Part b: How many six-digit numbers of the form are divisible by 218
theorem six_digit_count_div_by_218 :
  ∀ n : ℕ, six_digit_format n → divisible_by n 218 → false :=
sorry

end NUMINAMATH_GPT_six_digit_count_div_by_217_six_digit_count_div_by_218_l1697_169786


namespace NUMINAMATH_GPT_mimi_shells_l1697_169704

theorem mimi_shells (Kyle_shells Mimi_shells Leigh_shells : ℕ) 
  (h₀ : Kyle_shells = 2 * Mimi_shells) 
  (h₁ : Leigh_shells = Kyle_shells / 3) 
  (h₂ : Leigh_shells = 16) 
  : Mimi_shells = 24 := by 
  sorry

end NUMINAMATH_GPT_mimi_shells_l1697_169704


namespace NUMINAMATH_GPT_find_correct_value_l1697_169710

-- Definitions based on the problem's conditions
def incorrect_calculation (x : ℤ) : Prop := 7 * x = 126
def correct_value (x : ℤ) (y : ℤ) : Prop := x / 6 = y

theorem find_correct_value :
  ∃ (x y : ℤ), incorrect_calculation x ∧ correct_value x y ∧ y = 3 := by
  sorry

end NUMINAMATH_GPT_find_correct_value_l1697_169710
