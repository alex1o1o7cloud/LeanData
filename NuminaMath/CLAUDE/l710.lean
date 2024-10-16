import Mathlib

namespace NUMINAMATH_CALUDE_corporation_employee_count_l710_71047

/-- The number of employees at a corporation. -/
structure Corporation where
  female_employees : ℕ
  total_managers : ℕ
  male_associates : ℕ
  female_managers : ℕ

/-- The total number of employees in the corporation. -/
def Corporation.total_employees (c : Corporation) : ℕ :=
  c.female_employees + c.male_associates + (c.total_managers - c.female_managers)

/-- Theorem stating that the total number of employees is 250 given the specific conditions. -/
theorem corporation_employee_count (c : Corporation)
  (h1 : c.female_employees = 90)
  (h2 : c.total_managers = 40)
  (h3 : c.male_associates = 160)
  (h4 : c.female_managers = 40) :
  c.total_employees = 250 := by
  sorry

#check corporation_employee_count

end NUMINAMATH_CALUDE_corporation_employee_count_l710_71047


namespace NUMINAMATH_CALUDE_sum_of_xy_l710_71005

theorem sum_of_xy (x y : ℕ) (hx : 0 < x ∧ x < 20) (hy : 0 < y ∧ y < 20) 
  (h_eq : x + y + x * y = 76) : x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l710_71005


namespace NUMINAMATH_CALUDE_range_of_x_l710_71006

-- Define the set of real numbers that satisfy the given condition
def S : Set ℝ := {x | ¬(x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)}

-- Theorem stating that S is equal to the interval [1,2)
theorem range_of_x : S = Set.Ico 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l710_71006


namespace NUMINAMATH_CALUDE_hamburger_problem_l710_71060

theorem hamburger_problem (total_spent : ℚ) (total_burgers : ℕ) 
  (single_cost : ℚ) (double_cost : ℚ) (h1 : total_spent = 68.5) 
  (h2 : total_burgers = 50) (h3 : single_cost = 1) (h4 : double_cost = 1.5) :
  ∃ (single_count double_count : ℕ),
    single_count + double_count = total_burgers ∧
    single_count * single_cost + double_count * double_cost = total_spent ∧
    double_count = 37 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_problem_l710_71060


namespace NUMINAMATH_CALUDE_polynomial_determination_l710_71094

theorem polynomial_determination (p : ℝ → ℝ) : 
  (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) →  -- p is quadratic
  p 3 = 0 →                                        -- p(3) = 0
  p (-1) = 0 →                                     -- p(-1) = 0
  p 2 = 10 →                                       -- p(2) = 10
  ∀ x, p x = -10/3 * x^2 + 20/3 * x + 10 :=        -- conclusion
by sorry

end NUMINAMATH_CALUDE_polynomial_determination_l710_71094


namespace NUMINAMATH_CALUDE_hcd_7560_270_minus_4_l710_71001

theorem hcd_7560_270_minus_4 : Nat.gcd 7560 270 - 4 = 266 := by
  sorry

end NUMINAMATH_CALUDE_hcd_7560_270_minus_4_l710_71001


namespace NUMINAMATH_CALUDE_max_n_satisfying_condition_l710_71044

def sequence_a (n : ℕ) : ℕ := 2^n - 1

def sum_S (n : ℕ) : ℕ := 2 * sequence_a n - n

theorem max_n_satisfying_condition :
  (∀ n : ℕ, sum_S n = 2 * sequence_a n - n) →
  (∃ max_n : ℕ, (∀ n : ℕ, n ≤ max_n ↔ sequence_a n ≤ 10 * n) ∧ max_n = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_n_satisfying_condition_l710_71044


namespace NUMINAMATH_CALUDE_triangle_side_length_l710_71015

theorem triangle_side_length (A B C : Real) (R : Real) (a b c : Real) :
  R = 5/6 →
  Real.cos B = 3/5 →
  Real.cos A = 12/13 →
  c = 21/13 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l710_71015


namespace NUMINAMATH_CALUDE_consecutive_six_product_not_776965920_l710_71010

theorem consecutive_six_product_not_776965920 (n : ℕ) : 
  n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ≠ 776965920 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_six_product_not_776965920_l710_71010


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l710_71040

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x - m ≥ 0) → -4 ≤ m ∧ m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l710_71040


namespace NUMINAMATH_CALUDE_lilys_lottery_prize_l710_71032

/-- The amount of money the lottery winner will receive -/
def lottery_prize (num_tickets : ℕ) (initial_price : ℕ) (price_increment : ℕ) (profit : ℕ) : ℕ :=
  let total_sales := (num_tickets * (2 * initial_price + (num_tickets - 1) * price_increment)) / 2
  total_sales - profit

/-- Theorem stating the lottery prize for Lily's specific scenario -/
theorem lilys_lottery_prize :
  lottery_prize 5 1 1 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_lilys_lottery_prize_l710_71032


namespace NUMINAMATH_CALUDE_exponent_division_l710_71096

theorem exponent_division (x : ℝ) (hx : x ≠ 0) : 2 * x^4 / x^3 = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l710_71096


namespace NUMINAMATH_CALUDE_intersection_M_N_l710_71031

def M : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l710_71031


namespace NUMINAMATH_CALUDE_alpha_cheaper_at_min_shirts_l710_71012

/-- Alpha T-Shirt Company's pricing model -/
def alpha_cost (n : ℕ) : ℚ := 80 + 12 * n

/-- Omega T-Shirt Company's pricing model -/
def omega_cost (n : ℕ) : ℚ := 10 + 18 * n

/-- The minimum number of shirts for which Alpha becomes cheaper -/
def min_shirts_for_alpha : ℕ := 12

theorem alpha_cheaper_at_min_shirts :
  alpha_cost min_shirts_for_alpha < omega_cost min_shirts_for_alpha ∧
  ∀ m : ℕ, m < min_shirts_for_alpha → alpha_cost m ≥ omega_cost m :=
by sorry

end NUMINAMATH_CALUDE_alpha_cheaper_at_min_shirts_l710_71012


namespace NUMINAMATH_CALUDE_right_triangle_area_l710_71025

theorem right_triangle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a = (4/3) * b) (h5 : a = (2/3) * c) (h6 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 2/3 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l710_71025


namespace NUMINAMATH_CALUDE_function_minimum_l710_71000

open Real

theorem function_minimum (x : ℝ) (hx : 0 < x ∧ x < π/2) :
  let y := sin x - 2 * cos x + 32 / (125 * sin x * (1 - cos x))
  y ≥ 2/5 :=
sorry

end NUMINAMATH_CALUDE_function_minimum_l710_71000


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l710_71099

/-- Given three square regions A, B, and C, where the perimeter of A is 16 units
    and the perimeter of B is 32 units, prove that the ratio of the area of
    region A to the area of region C is 1/9. -/
theorem area_ratio_of_squares (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) →
  (4 * a = 16) → (4 * b = 32) → (c = 3 * a) →
  (a^2) / (c^2) = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l710_71099


namespace NUMINAMATH_CALUDE_fraction_addition_l710_71076

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l710_71076


namespace NUMINAMATH_CALUDE_fraction_comparison_l710_71055

theorem fraction_comparison (x y : ℕ+) (h : y > x) : (x + 1 : ℚ) / (y + 1) > (x : ℚ) / y := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l710_71055


namespace NUMINAMATH_CALUDE_cos_to_sin_shift_l710_71082

theorem cos_to_sin_shift (x : ℝ) : 
  3 * Real.cos (2 * x - π / 4) = 3 * Real.sin (2 * (x + π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_cos_to_sin_shift_l710_71082


namespace NUMINAMATH_CALUDE_min_selection_for_tenfold_l710_71050

theorem min_selection_for_tenfold (n : ℕ) (h : n = 2020) :
  ∃ k : ℕ, k = 203 ∧
  (∀ S : Finset ℕ, S.card < k → S ⊆ Finset.range (n + 1) →
    ¬∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a = 10 * b) ∧
  (∃ S : Finset ℕ, S.card = k ∧ S ⊆ Finset.range (n + 1) ∧
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a = 10 * b) :=
by sorry

end NUMINAMATH_CALUDE_min_selection_for_tenfold_l710_71050


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l710_71075

/-- The lateral surface area of a cone with base radius 1 and height 2√2 is 3π. -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 1  -- base radius
  let h : ℝ := 2 * Real.sqrt 2  -- height
  let l : ℝ := Real.sqrt (r^2 + h^2)  -- slant height
  π * r * l = 3 * π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l710_71075


namespace NUMINAMATH_CALUDE_pencils_given_to_joyce_l710_71073

theorem pencils_given_to_joyce (initial_pencils : ℝ) (remaining_pencils : ℕ) 
  (h1 : initial_pencils = 51.0)
  (h2 : remaining_pencils = 45) :
  initial_pencils - remaining_pencils = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencils_given_to_joyce_l710_71073


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l710_71036

/-- The maximum number of students among whom 781 pens and 710 pencils can be distributed equally -/
theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h_pens : pens = 781) (h_pencils : pencils = 710) :
  (∃ (students pen_per_student pencil_per_student : ℕ), 
    students * pen_per_student = pens ∧ 
    students * pencil_per_student = pencils ∧ 
    ∀ s : ℕ, s * pen_per_student = pens → s * pencil_per_student = pencils → s ≤ students) →
  Nat.gcd pens pencils = 71 :=
by sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l710_71036


namespace NUMINAMATH_CALUDE_number_problem_l710_71064

theorem number_problem (x : ℝ) : 
  3 - (1/4 * 2) - (1/3 * 3) - (1/7 * x) = 27 → 
  (10/100) * x = 17.85 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l710_71064


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l710_71020

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (y - 5) = 9 → y = 86 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l710_71020


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_33_l710_71059

theorem consecutive_integers_sqrt_33 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 33) → (Real.sqrt 33 < b) → (a + b = 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_33_l710_71059


namespace NUMINAMATH_CALUDE_election_invalid_votes_percentage_l710_71037

theorem election_invalid_votes_percentage 
  (total_votes : ℕ) 
  (b_votes : ℕ) 
  (h1 : total_votes = 8720)
  (h2 : b_votes = 2834)
  (h3 : ∃ (a_votes : ℕ), a_votes = b_votes + (15 * total_votes) / 100) :
  (total_votes - (b_votes + (b_votes + (15 * total_votes) / 100))) * 100 / total_votes = 20 := by
  sorry

end NUMINAMATH_CALUDE_election_invalid_votes_percentage_l710_71037


namespace NUMINAMATH_CALUDE_modular_inverse_of_7_mod_31_l710_71087

theorem modular_inverse_of_7_mod_31 : ∃ x : ℕ, x ≤ 30 ∧ (7 * x) % 31 = 1 :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_7_mod_31_l710_71087


namespace NUMINAMATH_CALUDE_polynomial_coefficient_property_l710_71017

theorem polynomial_coefficient_property (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_property_l710_71017


namespace NUMINAMATH_CALUDE_geometric_progression_condition_l710_71016

/-- 
Given a, b, c are real numbers and k, n, p are integers,
if a, b, c are the k-th, n-th, and p-th terms respectively of a geometric progression,
then (a/b)^(k-p) = (a/c)^(k-n)
-/
theorem geometric_progression_condition 
  (a b c : ℝ) (k n p : ℤ) 
  (hk : k ≠ n) (hn : n ≠ p) (hp : p ≠ k)
  (hgp : ∃ (r : ℝ), r ≠ 0 ∧ b = a * r^(n-k) ∧ c = a * r^(p-k)) :
  (a/b)^(k-p) = (a/c)^(k-n) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_condition_l710_71016


namespace NUMINAMATH_CALUDE_ice_cream_arrangements_l710_71091

theorem ice_cream_arrangements : (Nat.factorial 5) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangements_l710_71091


namespace NUMINAMATH_CALUDE_probability_of_three_common_books_l710_71080

theorem probability_of_three_common_books :
  let total_books : ℕ := 12
  let books_per_student : ℕ := 6
  let common_books : ℕ := 3
  
  let total_outcomes : ℕ := (Nat.choose total_books books_per_student) ^ 2
  let successful_outcomes : ℕ := 
    (Nat.choose total_books common_books) * 
    (Nat.choose (total_books - common_books) (books_per_student - common_books)) * 
    (Nat.choose (total_books - books_per_student) (books_per_student - common_books))
  
  (successful_outcomes : ℚ) / total_outcomes = 5 / 23
  := by sorry

end NUMINAMATH_CALUDE_probability_of_three_common_books_l710_71080


namespace NUMINAMATH_CALUDE_optimal_box_volume_l710_71072

/-- The volume of an open box made from a 48m x 36m sheet by cutting squares from corners -/
def box_volume (x : ℝ) : ℝ := (48 - 2*x) * (36 - 2*x) * x

/-- The derivative of the box volume function -/
def box_volume_derivative (x : ℝ) : ℝ := 1728 - 336*x + 12*x^2

theorem optimal_box_volume :
  ∃ (x : ℝ),
    x = 12 ∧
    (∀ y : ℝ, 0 < y ∧ y < 24 → box_volume y ≤ box_volume x) ∧
    box_volume x = 3456 :=
by sorry

end NUMINAMATH_CALUDE_optimal_box_volume_l710_71072


namespace NUMINAMATH_CALUDE_sqrt_pattern_main_problem_l710_71078

theorem sqrt_pattern (n : ℕ) (h : n > 0) :
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = 1 + 1 / n - 1 / (n + 1) :=
sorry

theorem main_problem :
  Real.sqrt (50 / 49 + 1 / 64) = 1 + 1 / 56 :=
sorry

end NUMINAMATH_CALUDE_sqrt_pattern_main_problem_l710_71078


namespace NUMINAMATH_CALUDE_unique_base_solution_l710_71095

/-- Converts a base-10 number to base-a representation --/
def toBaseA (n : ℕ) (a : ℕ) : List ℕ :=
  sorry

/-- Converts a base-a number (represented as a list of digits) to base-10 --/
def fromBaseA (digits : List ℕ) (a : ℕ) : ℕ :=
  sorry

/-- Checks if the equation 452_a + 127_a = 5B0_a holds for a given base a --/
def checkEquation (a : ℕ) : Prop :=
  fromBaseA (toBaseA 452 a) a + fromBaseA (toBaseA 127 a) a = 
  fromBaseA ([5, 11, 0]) a

theorem unique_base_solution :
  ∃! a : ℕ, a > 11 ∧ checkEquation a ∧ fromBaseA ([11]) a = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_base_solution_l710_71095


namespace NUMINAMATH_CALUDE_range_of_a_l710_71053

/-- Proposition p: The solution set of the inequality x^2 + (a-1)x + a^2 < 0 regarding x is an empty set. -/
def proposition_p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a-1)*x + a^2 ≥ 0

/-- Quadratic function f(x) = x^2 - mx + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2

/-- Proposition q: Given the quadratic function f(x) = x^2 - mx + 2 satisfies f(3/2 + x) = f(3/2 - x),
    and its maximum value is 2 when x ∈ [0,a]. -/
def proposition_q (a : ℝ) : Prop :=
  ∃ m, (∀ x, f m (3/2 + x) = f m (3/2 - x)) ∧
       (∀ x ∈ Set.Icc 0 a, f m x ≤ 2) ∧
       (∃ x ∈ Set.Icc 0 a, f m x = 2)

/-- The range of a given the logical conditions on p and q -/
theorem range_of_a :
  ∀ a : ℝ, (¬(proposition_p a ∧ proposition_q a) ∧ (proposition_p a ∨ proposition_q a)) ↔
            a ∈ Set.Iic (-1) ∪ Set.Ioo 0 (1/3) ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l710_71053


namespace NUMINAMATH_CALUDE_solution_count_l710_71090

theorem solution_count (S : Finset ℝ) (p : ℝ) : 
  S.card = 12 → p = 1/6 → ∃ n : ℕ, n = 2 ∧ n = (S.card : ℝ) * p := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l710_71090


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l710_71077

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l710_71077


namespace NUMINAMATH_CALUDE_usable_field_area_l710_71085

/-- Calculates the area of a usable rectangular field with an L-shaped obstacle -/
theorem usable_field_area
  (breadth : ℕ)
  (h1 : breadth + 30 = 150)  -- Length is 30 meters more than breadth
  (h2 : 2 * (breadth + (breadth + 30)) = 540)  -- Perimeter is 540 meters
  : (breadth - 5) * (breadth + 30 - 10) = 16100 :=
by sorry

end NUMINAMATH_CALUDE_usable_field_area_l710_71085


namespace NUMINAMATH_CALUDE_constant_in_exponent_l710_71024

theorem constant_in_exponent (w : ℕ) (h1 : 2^(2*w) = 8^(w-4)) (h2 : w = 12) : 
  ∃ k : ℕ, 2^(2*w) = 8^(w-k) ∧ k = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_in_exponent_l710_71024


namespace NUMINAMATH_CALUDE_unique_non_six_order_l710_71063

theorem unique_non_six_order (a : ℤ) : 
  (a > 1 ∧ ∀ p : ℕ, Nat.Prime p → ∀ n : ℕ, n > 0 ∧ a^n ≡ 1 [ZMOD p] → n ≠ 6) ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_non_six_order_l710_71063


namespace NUMINAMATH_CALUDE_fourth_root_of_x_sqrt_x_squared_l710_71028

theorem fourth_root_of_x_sqrt_x_squared (x : ℝ) (hx : x > 0) : 
  (((x * Real.sqrt x) ^ 2) ^ (1/4 : ℝ)) = x ^ (3/4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_fourth_root_of_x_sqrt_x_squared_l710_71028


namespace NUMINAMATH_CALUDE_household_expenses_equal_savings_l710_71098

/-- The number of years it takes to buy a house with all earnings -/
def years_to_buy : ℕ := 4

/-- The total number of years to buy the house -/
def total_years : ℕ := 24

/-- The number of years spent saving -/
def years_saving : ℕ := 12

/-- The number of years spent on household expenses -/
def years_household : ℕ := total_years - years_saving

theorem household_expenses_equal_savings : years_household = years_saving := by
  sorry

end NUMINAMATH_CALUDE_household_expenses_equal_savings_l710_71098


namespace NUMINAMATH_CALUDE_combined_weight_of_acids_l710_71057

/-- The atomic mass of carbon in g/mol -/
def carbon_mass : ℝ := 12.01

/-- The atomic mass of hydrogen in g/mol -/
def hydrogen_mass : ℝ := 1.01

/-- The atomic mass of oxygen in g/mol -/
def oxygen_mass : ℝ := 16.00

/-- The atomic mass of sulfur in g/mol -/
def sulfur_mass : ℝ := 32.07

/-- The molar mass of C6H8O7 in g/mol -/
def citric_acid_mass : ℝ := 6 * carbon_mass + 8 * hydrogen_mass + 7 * oxygen_mass

/-- The molar mass of H2SO4 in g/mol -/
def sulfuric_acid_mass : ℝ := 2 * hydrogen_mass + sulfur_mass + 4 * oxygen_mass

/-- The number of moles of C6H8O7 -/
def citric_acid_moles : ℝ := 8

/-- The number of moles of H2SO4 -/
def sulfuric_acid_moles : ℝ := 4

/-- The combined weight of C6H8O7 and H2SO4 in grams -/
def combined_weight : ℝ := citric_acid_moles * citric_acid_mass + sulfuric_acid_moles * sulfuric_acid_mass

theorem combined_weight_of_acids : combined_weight = 1929.48 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_of_acids_l710_71057


namespace NUMINAMATH_CALUDE_brad_read_more_books_l710_71066

def william_last_month : ℕ := 6
def brad_this_month : ℕ := 8

def brad_last_month : ℕ := 3 * william_last_month
def william_this_month : ℕ := 2 * brad_this_month

def william_total : ℕ := william_last_month + william_this_month
def brad_total : ℕ := brad_last_month + brad_this_month

theorem brad_read_more_books : brad_total = william_total + 4 := by
  sorry

end NUMINAMATH_CALUDE_brad_read_more_books_l710_71066


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l710_71002

theorem line_segment_endpoint (y : ℝ) : y > 0 →
  (Real.sqrt (((-7) - 3)^2 + (y - (-2))^2) = 13) →
  y = -2 + Real.sqrt 69 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l710_71002


namespace NUMINAMATH_CALUDE_hyperbola_intersection_l710_71004

/-- Given a triangle AOB with A on the positive y-axis, B on the positive x-axis, and area 9,
    and a hyperbolic function y = k/x intersecting AB at C and D such that CD = 1/3 AB and AC = BD,
    prove that k = 4 -/
theorem hyperbola_intersection (y_A x_B : ℝ) (k : ℝ) : 
  y_A > 0 → x_B > 0 → -- A and B are on positive axes
  1/2 * x_B * y_A = 9 → -- Area of triangle AOB is 9
  ∃ (x_C y_C : ℝ), -- C exists on the line AB and the hyperbola
    0 < x_C ∧ x_C < x_B ∧
    y_C = (y_A / x_B) * (x_B - x_C) ∧ -- C is on line AB
    y_C = k / x_C ∧ -- C is on the hyperbola
    x_C = 1/3 * x_B ∧ -- C is a trisection point
    y_C = 2/3 * y_A → -- C is a trisection point
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_l710_71004


namespace NUMINAMATH_CALUDE_unique_solution_k_l710_71093

theorem unique_solution_k (k : ℝ) : 
  (∃! x : ℝ, (1 / (3 * x) = (k - x) / 8)) ↔ k = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_k_l710_71093


namespace NUMINAMATH_CALUDE_arrangement_theorem_l710_71062

/-- Represents the number of ways to arrange people in two rows -/
def arrangement_count (total_people : ℕ) (front_row : ℕ) (back_row : ℕ) : ℕ := sorry

/-- Represents whether two people are standing next to each other -/
def standing_next_to (person1 : ℕ) (person2 : ℕ) : Prop := sorry

/-- Represents whether two people are standing apart -/
def standing_apart (person1 : ℕ) (person2 : ℕ) : Prop := sorry

theorem arrangement_theorem :
  ∀ (total_people front_row back_row : ℕ) 
    (person_a person_b person_c : ℕ),
  total_people = 7 →
  front_row = 3 →
  back_row = 4 →
  standing_next_to person_a person_b →
  standing_apart person_a person_c →
  arrangement_count total_people front_row back_row = 1056 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l710_71062


namespace NUMINAMATH_CALUDE_tangent_line_reciprocal_function_l710_71054

/-- The equation of the tangent line to y = 1/x at (1,1) is x + y - 2 = 0 -/
theorem tangent_line_reciprocal_function (x y : ℝ) : 
  (∀ t, t ≠ 0 → y = 1 / t) →  -- Condition: the curve is y = 1/x
  (x = 1 ∧ y = 1) →           -- Condition: the point of tangency is (1,1)
  x + y - 2 = 0               -- Conclusion: equation of the tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_reciprocal_function_l710_71054


namespace NUMINAMATH_CALUDE_north_american_stamps_cost_is_91_cents_l710_71046

/-- Represents a country --/
inductive Country
| China
| Japan
| Canada
| Mexico

/-- Represents a continent --/
inductive Continent
| Asia
| NorthAmerica

/-- Represents a decade --/
inductive Decade
| D1960s
| D1970s

/-- Maps a country to its continent --/
def country_continent : Country → Continent
| Country.China => Continent.Asia
| Country.Japan => Continent.Asia
| Country.Canada => Continent.NorthAmerica
| Country.Mexico => Continent.NorthAmerica

/-- Cost of stamps in cents for each country --/
def stamp_cost : Country → ℕ
| Country.China => 7
| Country.Japan => 7
| Country.Canada => 3
| Country.Mexico => 4

/-- Number of stamps for each country and decade --/
def stamp_count : Country → Decade → ℕ
| Country.China => fun
  | Decade.D1960s => 5
  | Decade.D1970s => 9
| Country.Japan => fun
  | Decade.D1960s => 6
  | Decade.D1970s => 7
| Country.Canada => fun
  | Decade.D1960s => 7
  | Decade.D1970s => 6
| Country.Mexico => fun
  | Decade.D1960s => 8
  | Decade.D1970s => 5

/-- Total cost of North American stamps from 1960s and 1970s --/
def north_american_stamps_cost : ℚ :=
  let north_american_countries := [Country.Canada, Country.Mexico]
  let decades := [Decade.D1960s, Decade.D1970s]
  (north_american_countries.map fun country =>
    (decades.map fun decade =>
      (stamp_count country decade) * (stamp_cost country)
    ).sum
  ).sum / 100

theorem north_american_stamps_cost_is_91_cents :
  north_american_stamps_cost = 91 / 100 := by sorry

end NUMINAMATH_CALUDE_north_american_stamps_cost_is_91_cents_l710_71046


namespace NUMINAMATH_CALUDE_correct_statement_l710_71041

def p : Prop := 2017 % 2 = 1
def q : Prop := 2016 % 2 = 0

theorem correct_statement : p ∨ q := by sorry

end NUMINAMATH_CALUDE_correct_statement_l710_71041


namespace NUMINAMATH_CALUDE_unique_solution_equation_l710_71049

theorem unique_solution_equation :
  ∃! x : ℝ, 2017 * x^2017 - 2017 + x = (2018 - 2017*x)^(1/2017) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l710_71049


namespace NUMINAMATH_CALUDE_two_heads_probability_l710_71081

/-- Represents the outcome of a single coin toss -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the outcome of tossing two coins simultaneously -/
def TwoCoinsOutcome := (CoinOutcome × CoinOutcome)

/-- The set of all possible outcomes when tossing two coins -/
def allOutcomes : Finset TwoCoinsOutcome := sorry

/-- The set of outcomes where both coins show heads -/
def twoHeadsOutcomes : Finset TwoCoinsOutcome := sorry

/-- Proposition: The probability of getting two heads when tossing two fair coins simultaneously is 1/4 -/
theorem two_heads_probability :
  (Finset.card twoHeadsOutcomes) / (Finset.card allOutcomes : ℚ) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_two_heads_probability_l710_71081


namespace NUMINAMATH_CALUDE_train_crossing_time_l710_71033

/-- Given a train and a platform with specific dimensions, calculate the time taken for the train to cross the platform. -/
theorem train_crossing_time (train_length : ℝ) (signal_crossing_time : ℝ) (platform_length : ℝ)
  (h1 : train_length = 450)
  (h2 : signal_crossing_time = 18)
  (h3 : platform_length = 525) :
  (train_length + platform_length) / (train_length / signal_crossing_time) = 39 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l710_71033


namespace NUMINAMATH_CALUDE_sets_intersection_and_complement_l710_71013

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def B : Set ℝ := {x | (x - 2) / (x + 3) > 0}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- State the theorem
theorem sets_intersection_and_complement (a : ℝ) 
  (h : A ∩ C a = C a) : 
  (A ∩ B = Set.Ioc 2 3) ∧ 
  ((Set.univ \ A) ∪ (Set.univ \ B) = Set.Iic 2 ∪ Set.Ioi 3) ∧
  (a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_sets_intersection_and_complement_l710_71013


namespace NUMINAMATH_CALUDE_cubic_equation_with_geometric_roots_l710_71021

/-- Given a cubic equation x^3 - 14x^2 + ax - 27 = 0 with three distinct real roots in geometric progression, prove that a = 42 -/
theorem cubic_equation_with_geometric_roots (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧  -- distinct roots
    (∃ r : ℝ, r ≠ 0 ∧ x₂ = x₁ * r ∧ x₃ = x₂ * r) ∧  -- geometric progression
    (x₁^3 - 14*x₁^2 + a*x₁ - 27 = 0) ∧
    (x₂^3 - 14*x₂^2 + a*x₂ - 27 = 0) ∧
    (x₃^3 - 14*x₃^2 + a*x₃ - 27 = 0)) →
  a = 42 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_with_geometric_roots_l710_71021


namespace NUMINAMATH_CALUDE_garden_length_l710_71061

/-- Proves that a rectangular garden with length twice its width and perimeter 240 yards has a length of 80 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width → -- length is twice the width
  2 * length + 2 * width = 240 → -- perimeter is 240 yards
  length = 80 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l710_71061


namespace NUMINAMATH_CALUDE_martha_cakes_l710_71074

/-- The number of cakes Martha needs to buy -/
def total_cakes (num_children : ℝ) (cakes_per_child : ℝ) : ℝ :=
  num_children * cakes_per_child

/-- Theorem: Martha needs to buy 54 cakes -/
theorem martha_cakes : total_cakes 3 18 = 54 := by
  sorry

end NUMINAMATH_CALUDE_martha_cakes_l710_71074


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l710_71067

theorem right_triangle_leg_square (a b c : ℝ) : 
  (a^2 + b^2 = c^2) →  -- right triangle condition
  (c = a + 2) →        -- hypotenuse is 2 units longer than leg a
  b^2 = 4*(a + 1) :=   -- square of other leg b
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l710_71067


namespace NUMINAMATH_CALUDE_nell_gave_136_cards_to_jeff_l710_71051

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff (original_cards : ℕ) (cards_left : ℕ) : ℕ :=
  original_cards - cards_left

/-- Proof that Nell gave 136 cards to Jeff -/
theorem nell_gave_136_cards_to_jeff :
  cards_given_to_jeff 242 106 = 136 := by
  sorry

end NUMINAMATH_CALUDE_nell_gave_136_cards_to_jeff_l710_71051


namespace NUMINAMATH_CALUDE_range_of_a_for_two_zeros_l710_71071

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then Real.log (1 - x) else Real.sqrt x - a

-- State the theorem
theorem range_of_a_for_two_zeros (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
   (∀ z : ℝ, f a z = 0 → z = x ∨ z = y)) →
  a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_zeros_l710_71071


namespace NUMINAMATH_CALUDE_equation_solutions_l710_71003

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 13*x - 12) + 1 / (x^2 + 3*x - 12) + 1 / (x^2 - 16*x - 12) = 0)} = 
  {1, -12, 3, -4} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l710_71003


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l710_71043

theorem complex_number_in_second_quadrant :
  let z : ℂ := Complex.I / (1 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l710_71043


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l710_71069

theorem intersection_nonempty_implies_a_value (P Q : Set ℕ) (a : ℕ) :
  P = {0, a} →
  Q = {1, 2} →
  P ∩ Q ≠ ∅ →
  a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l710_71069


namespace NUMINAMATH_CALUDE_students_like_both_desserts_l710_71022

/-- Proves the number of students who like both apple pie and chocolate cake -/
theorem students_like_both_desserts 
  (total : ℕ) 
  (like_apple : ℕ) 
  (like_chocolate : ℕ) 
  (like_neither : ℕ) 
  (h1 : total = 50)
  (h2 : like_apple = 22)
  (h3 : like_chocolate = 20)
  (h4 : like_neither = 15) :
  total - like_neither - (like_apple + like_chocolate - (total - like_neither)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_like_both_desserts_l710_71022


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l710_71035

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l710_71035


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l710_71009

theorem simplify_fraction_product : 
  (36 : ℚ) / 51 * 35 / 24 * 68 / 49 = 20 / 7 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l710_71009


namespace NUMINAMATH_CALUDE_perimeter_APR_is_50_l710_71048

/-- A circle with two tangents from an exterior point A touching at B and C,
    and a third tangent touching at Q and intersecting AB at P and AC at R. -/
structure TangentCircle where
  /-- The length of tangent AB -/
  AB : ℝ
  /-- The distance from A to Q along the tangent -/
  AQ : ℝ

/-- The perimeter of triangle APR in the TangentCircle configuration -/
def perimeterAPR (tc : TangentCircle) : ℝ :=
  tc.AB - tc.AQ + tc.AQ + tc.AQ

/-- Theorem stating that for a TangentCircle with AB = 25 and AQ = 12.5,
    the perimeter of triangle APR is 50 -/
theorem perimeter_APR_is_50 (tc : TangentCircle)
    (h1 : tc.AB = 25) (h2 : tc.AQ = 12.5) :
    perimeterAPR tc = 50 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_APR_is_50_l710_71048


namespace NUMINAMATH_CALUDE_linear_function_point_value_l710_71039

theorem linear_function_point_value (m n : ℝ) : 
  n = 3 - 5 * m → 10 * m + 2 * n - 3 = 3 := by sorry

end NUMINAMATH_CALUDE_linear_function_point_value_l710_71039


namespace NUMINAMATH_CALUDE_building_units_count_l710_71058

/-- Represents the number of units in a building -/
structure Building where
  oneBedroom : ℕ
  twoBedroom : ℕ

/-- The total cost of all units in the building -/
def totalCost (b : Building) : ℕ := 360 * b.oneBedroom + 450 * b.twoBedroom

/-- The total number of units in the building -/
def totalUnits (b : Building) : ℕ := b.oneBedroom + b.twoBedroom

theorem building_units_count :
  ∃ (b : Building),
    totalCost b = 4950 ∧
    b.twoBedroom = 7 ∧
    totalUnits b = 12 := by
  sorry

end NUMINAMATH_CALUDE_building_units_count_l710_71058


namespace NUMINAMATH_CALUDE_cube_coloring_ways_octahedron_coloring_ways_l710_71083

-- Define the number of colors for each shape
def cube_colors : ℕ := 6
def octahedron_colors : ℕ := 8

-- Define the number of faces for each shape
def cube_faces : ℕ := 6
def octahedron_faces : ℕ := 8

-- Theorem for coloring the cube
theorem cube_coloring_ways :
  (cube_colors.factorial / (cube_colors - cube_faces).factorial) = 30 := by
  sorry

-- Theorem for coloring the octahedron
theorem octahedron_coloring_ways :
  (octahedron_colors.factorial / (octahedron_colors - octahedron_faces).factorial) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_cube_coloring_ways_octahedron_coloring_ways_l710_71083


namespace NUMINAMATH_CALUDE_kah_to_zah_conversion_l710_71008

/-- Conversion rate between zahs and tols -/
def zah_to_tol : ℚ := 24 / 15

/-- Conversion rate between tols and kahs -/
def tol_to_kah : ℚ := 15 / 9

/-- The number of kahs we want to convert -/
def kahs_to_convert : ℕ := 2000

/-- The expected number of zahs after conversion -/
def expected_zahs : ℕ := 750

theorem kah_to_zah_conversion :
  (kahs_to_convert : ℚ) / (zah_to_tol * tol_to_kah) = expected_zahs := by
  sorry

end NUMINAMATH_CALUDE_kah_to_zah_conversion_l710_71008


namespace NUMINAMATH_CALUDE_largest_palindrome_multiple_of_6_l710_71042

def is_palindrome (n : ℕ) : Prop :=
  n ≥ 100 ∧ n ≤ 999 ∧ (n / 100 = n % 10)

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_palindrome_multiple_of_6 :
  ∀ n : ℕ, is_palindrome n → n % 6 = 0 → n ≤ 888 ∧
  (∃ m : ℕ, is_palindrome m ∧ m % 6 = 0 ∧ m = 888) ∧
  sum_of_digits 888 = 24 :=
sorry

end NUMINAMATH_CALUDE_largest_palindrome_multiple_of_6_l710_71042


namespace NUMINAMATH_CALUDE_sector_area_l710_71030

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 24) (h2 : θ = 110 * π / 180) :
  r^2 * θ / 2 = 176 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l710_71030


namespace NUMINAMATH_CALUDE_triangle_ambiguous_case_l710_71092

theorem triangle_ambiguous_case (a b : ℝ) (A : ℝ) : 
  a = 12 → A = π / 3 → (b * Real.sin A < a ∧ a < b) ↔ (12 < b ∧ b < 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ambiguous_case_l710_71092


namespace NUMINAMATH_CALUDE_can_determine_ten_gram_coins_can_determine_coin_weight_l710_71088

/-- Represents the weight of coins in grams -/
inductive CoinWeight
  | Ten
  | Eleven
  | Twelve
  | Thirteen
  | Fourteen

/-- Represents a bag of coins -/
structure Bag where
  weight : CoinWeight
  count : Nat
  h_count : count = 100

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal
  | LeftHeavier
  | RightHeavier

/-- Represents a collection of bags -/
structure BagCollection where
  bags : Fin 5 → Bag
  h_distinct : ∀ i j, i ≠ j → (bags i).weight ≠ (bags j).weight

/-- Function to perform a weighing -/
noncomputable def weigh (left right : List Nat) : WeighingResult :=
  sorry

/-- Theorem stating that it's possible to determine if a specific bag contains 10g coins with one weighing -/
theorem can_determine_ten_gram_coins (bags : BagCollection) (pointed : Fin 5) : 
  ∃ (left right : List Nat), 
    (∀ n ∈ left ∪ right, n ≤ 100) ∧ 
    (weigh left right = WeighingResult.Equal ↔ (bags.bags pointed).weight = CoinWeight.Ten) :=
  sorry

/-- Theorem stating that it's possible to determine the weight of coins in a specific bag with at most two weighings -/
theorem can_determine_coin_weight (bags : BagCollection) (pointed : Fin 5) :
  ∃ (left1 right1 left2 right2 : List Nat),
    (∀ n ∈ left1 ∪ right1 ∪ left2 ∪ right2, n ≤ 100) ∧
    (∃ f : WeighingResult → WeighingResult → CoinWeight,
      f (weigh left1 right1) (weigh left2 right2) = (bags.bags pointed).weight) :=
  sorry

end NUMINAMATH_CALUDE_can_determine_ten_gram_coins_can_determine_coin_weight_l710_71088


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l710_71097

theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l710_71097


namespace NUMINAMATH_CALUDE_labourer_savings_is_30_l710_71019

/-- Calculates the savings of a labourer after 10 months, given specific spending patterns -/
def labourerSavings (monthlyIncome : ℕ) (expenseFirst6Months : ℕ) (expenseLast4Months : ℕ) : ℤ :=
  let totalIncome : ℕ := monthlyIncome * 10
  let totalExpense : ℕ := expenseFirst6Months * 6 + expenseLast4Months * 4
  (totalIncome : ℤ) - (totalExpense : ℤ)

/-- Theorem stating that the labourer's savings after 10 months is 30 -/
theorem labourer_savings_is_30 :
  labourerSavings 75 80 60 = 30 := by
  sorry

#eval labourerSavings 75 80 60

end NUMINAMATH_CALUDE_labourer_savings_is_30_l710_71019


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l710_71034

theorem complex_magnitude_problem (z : ℂ) : z = (2 + I) / (1 - I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l710_71034


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l710_71070

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 11 ∧ 
  (∀ m : ℕ, m < k → (128 : ℝ)^m ≤ 8^25 + 1000) ∧
  (128 : ℝ)^k > 8^25 + 1000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l710_71070


namespace NUMINAMATH_CALUDE_fourth_person_truthful_l710_71045

/-- Represents a person who can be either a liar or truthful. -/
inductive Person
| Liar
| Truthful

/-- The statements made by each person. -/
def statement (p : Fin 4 → Person) : Prop :=
  (p 0 = Person.Liar ∧ p 1 = Person.Liar ∧ p 2 = Person.Liar ∧ p 3 = Person.Liar) ∨
  (∃! i, p i = Person.Liar) ∨
  (∃ i j, i ≠ j ∧ p i = Person.Liar ∧ p j = Person.Liar ∧ ∀ k, k ≠ i → k ≠ j → p k = Person.Truthful) ∨
  (p 3 = Person.Truthful)

/-- The main theorem stating that the fourth person must be truthful. -/
theorem fourth_person_truthful :
  ∀ p : Fin 4 → Person, statement p → p 3 = Person.Truthful :=
sorry

end NUMINAMATH_CALUDE_fourth_person_truthful_l710_71045


namespace NUMINAMATH_CALUDE_max_probability_at_20_red_balls_l710_71056

/-- The probability of winning in one draw -/
def p (n : ℕ) : ℚ := 10 * n / ((n + 5) * (n + 4))

/-- The probability of winning exactly once in three draws -/
def P (n : ℕ) : ℚ := 3 * p n * (1 - p n)^2

theorem max_probability_at_20_red_balls (n : ℕ) (h : n ≥ 5) :
  P n ≤ P 20 ∧ ∃ (m : ℕ), m ≥ 5 ∧ P m = P 20 → m = 20 :=
sorry

end NUMINAMATH_CALUDE_max_probability_at_20_red_balls_l710_71056


namespace NUMINAMATH_CALUDE_stone_123_is_3_l710_71007

/-- The number of stones in the sequence -/
def num_stones : ℕ := 12

/-- The length of the counting pattern before it repeats -/
def pattern_length : ℕ := 22

/-- The target count we're interested in -/
def target_count : ℕ := 123

/-- The original stone number we claim is counted as the target_count -/
def claimed_stone : ℕ := 3

/-- Function to determine which stone is counted as a given number -/
def stone_at_count (count : ℕ) : ℕ :=
  (count - 1) % pattern_length + 1

theorem stone_123_is_3 : 
  stone_at_count target_count = claimed_stone := by
  sorry

end NUMINAMATH_CALUDE_stone_123_is_3_l710_71007


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l710_71052

/-- The equation of a circle given the endpoints of its diameter -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  A = (-3, -1) →
  B = (5, 5) →
  ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 25 ↔ 
    ∃ t : ℝ, (x, y) = ((1 - t) • A.1 + t • B.1, (1 - t) • A.2 + t • B.2) ∧ 0 ≤ t ∧ t ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l710_71052


namespace NUMINAMATH_CALUDE_james_hourly_wage_l710_71079

theorem james_hourly_wage (main_wage : ℝ) (second_wage : ℝ) (main_hours : ℝ) (second_hours : ℝ) (total_earnings : ℝ) :
  second_wage = 0.8 * main_wage →
  main_hours = 30 →
  second_hours = main_hours / 2 →
  total_earnings = main_wage * main_hours + second_wage * second_hours →
  total_earnings = 840 →
  main_wage = 20 := by
sorry

end NUMINAMATH_CALUDE_james_hourly_wage_l710_71079


namespace NUMINAMATH_CALUDE_red_crayon_boxes_l710_71038

/-- The number of boxes of red crayons given the following conditions:
  * 6 boxes of 8 orange crayons each
  * 7 boxes of 5 blue crayons each
  * Each box of red crayons contains 11 crayons
  * Total number of crayons is 94
-/
theorem red_crayon_boxes : ℕ := by
  sorry

#check red_crayon_boxes

end NUMINAMATH_CALUDE_red_crayon_boxes_l710_71038


namespace NUMINAMATH_CALUDE_ram_krish_efficiency_ratio_l710_71084

/-- Ram's efficiency -/
def ram_efficiency : ℝ := 1

/-- Krish's efficiency -/
def krish_efficiency : ℝ := 2

/-- Time taken by Ram alone to complete the task -/
def ram_alone_time : ℝ := 30

/-- Time taken by Ram and Krish together to complete the task -/
def combined_time : ℝ := 10

/-- The amount of work to be done -/
def work : ℝ := ram_efficiency * ram_alone_time

theorem ram_krish_efficiency_ratio :
  ram_efficiency / krish_efficiency = 1 / 2 ∧
  work = ram_efficiency * ram_alone_time ∧
  work = (ram_efficiency + krish_efficiency) * combined_time :=
by sorry

end NUMINAMATH_CALUDE_ram_krish_efficiency_ratio_l710_71084


namespace NUMINAMATH_CALUDE_die_roll_frequency_l710_71011

/-- The frequency of an event in an experiment -/
def frequency (occurrences : ℕ) (totalTrials : ℕ) : ℚ :=
  occurrences / totalTrials

/-- The number of times the die was rolled -/
def totalRolls : ℕ := 100

/-- The number of times "even numbers facing up" occurred -/
def evenOccurrences : ℕ := 47

/-- The expected frequency of "even numbers facing up" -/
def expectedFrequency : ℚ := 47 / 100

theorem die_roll_frequency :
  frequency evenOccurrences totalRolls = expectedFrequency := by
  sorry

end NUMINAMATH_CALUDE_die_roll_frequency_l710_71011


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2014_l710_71068

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2014 :
  arithmetic_sequence 4 3 671 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2014_l710_71068


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l710_71026

theorem beef_weight_before_processing 
  (weight_after : ℝ) 
  (percent_lost : ℝ) 
  (h1 : weight_after = 546) 
  (h2 : percent_lost = 35) : 
  ∃ weight_before : ℝ, 
    weight_before * (1 - percent_lost / 100) = weight_after ∧ 
    weight_before = 840 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l710_71026


namespace NUMINAMATH_CALUDE_complement_of_supplement_35_l710_71018

/-- The supplement of an angle in degrees -/
def supplement (x : ℝ) : ℝ := 180 - x

/-- The complement of an angle in degrees -/
def complement (x : ℝ) : ℝ := 90 - x

/-- Theorem: The degree measure of the complement of the supplement of a 35-degree angle is -55 degrees -/
theorem complement_of_supplement_35 : complement (supplement 35) = -55 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_supplement_35_l710_71018


namespace NUMINAMATH_CALUDE_complex_fraction_power_l710_71065

theorem complex_fraction_power (i : ℂ) : i * i = -1 → (((1 + i) / (1 - i)) ^ 2018 : ℂ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l710_71065


namespace NUMINAMATH_CALUDE_tylenol_dosage_l710_71023

/-- Calculates the mg per pill given the total dosage and number of pills -/
def mg_per_pill (dosage_mg : ℕ) (dosage_interval_hours : ℕ) (duration_days : ℕ) (total_pills : ℕ) : ℚ :=
  let doses_per_day := 24 / dosage_interval_hours
  let total_doses := doses_per_day * duration_days
  let total_mg := dosage_mg * total_doses
  (total_mg : ℚ) / total_pills

theorem tylenol_dosage :
  mg_per_pill 1000 6 14 112 = 500 := by
  sorry

end NUMINAMATH_CALUDE_tylenol_dosage_l710_71023


namespace NUMINAMATH_CALUDE_juniper_bones_l710_71014

theorem juniper_bones (initial_bones doubled_bones stolen_bones : ℕ) 
  (h1 : initial_bones = 4)
  (h2 : doubled_bones = initial_bones * 2)
  (h3 : stolen_bones = 2) :
  doubled_bones - stolen_bones = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_juniper_bones_l710_71014


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l710_71089

/-- Given that if a person walks at 16 km/hr instead of 12 km/hr, they would have walked 20 km more,
    prove that the actual distance traveled is 60 km. -/
theorem actual_distance_traveled (D : ℝ) : 
  (D / 12 = (D + 20) / 16) → D = 60 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l710_71089


namespace NUMINAMATH_CALUDE_equation_solution_l710_71086

theorem equation_solution : ∃! x : ℝ, 4*x - 2*x + 1 - 3 = 0 :=
by
  use 1
  constructor
  · -- Prove that 1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l710_71086


namespace NUMINAMATH_CALUDE_infinite_omega_increasing_sequence_l710_71027

/-- The number of distinct prime divisors of a positive integer -/
def omega (n : ℕ) : ℕ := sorry

/-- The set of integers n > 1 satisfying ω(n) < ω(n+1) < ω(n+2) is infinite -/
theorem infinite_omega_increasing_sequence :
  Set.Infinite {n : ℕ | n > 1 ∧ omega n < omega (n + 1) ∧ omega (n + 1) < omega (n + 2)} :=
sorry

end NUMINAMATH_CALUDE_infinite_omega_increasing_sequence_l710_71027


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l710_71029

theorem reciprocal_of_negative_half : ((-1/2)⁻¹ : ℚ) = -2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l710_71029
