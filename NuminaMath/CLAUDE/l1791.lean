import Mathlib

namespace NUMINAMATH_CALUDE_cd_length_calculation_l1791_179167

theorem cd_length_calculation : 
  let first_cd_length : ℝ := 1.5
  let second_cd_length : ℝ := 1.5
  let third_cd_length : ℝ := 2 * first_cd_length
  first_cd_length + second_cd_length + third_cd_length = 6 := by sorry

end NUMINAMATH_CALUDE_cd_length_calculation_l1791_179167


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1791_179137

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 180 * r = b ∧ b * r = 36 / 25) → 
  b = Real.sqrt (6480 / 25) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1791_179137


namespace NUMINAMATH_CALUDE_security_system_connections_l1791_179195

/-- 
Given a security system with 25 switches where each switch is connected to exactly 4 other switches,
the total number of connections is 50.
-/
theorem security_system_connections (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n * k) / 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_security_system_connections_l1791_179195


namespace NUMINAMATH_CALUDE_special_function_property_l1791_179142

/-- A function f with specific properties -/
structure SpecialFunction (f : ℝ → ℝ) : Prop where
  deriv : ∀ x, deriv f x = 2 * x - 5
  f0_int : ∃ k : ℤ, f 0 = k

/-- The property that f has exactly one integer value in (n, n+1] -/
def ExactlyOneIntegerValue (f : ℝ → ℝ) (n : ℕ) : Prop :=
  ∃! k : ℤ, ∃ x : ℝ, n < x ∧ x ≤ n + 1 ∧ f x = k

/-- The main theorem -/
theorem special_function_property {f : ℝ → ℝ} (hf : SpecialFunction f) :
  (∃ n : ℕ, n > 0 ∧ ExactlyOneIntegerValue f n) → 
  (∃ n : ℕ, n > 0 ∧ ExactlyOneIntegerValue f n ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_special_function_property_l1791_179142


namespace NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l1791_179100

theorem unique_solution_diophantine_equation :
  ∃! (x y z t : ℕ+), 1 + 5^x.val = 2^y.val + 2^z.val * 5^t.val :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l1791_179100


namespace NUMINAMATH_CALUDE_ice_skate_rental_fee_l1791_179164

/-- The rental fee for ice skates at a rink, given the admission fee, cost of new skates, and number of visits to justify buying. -/
theorem ice_skate_rental_fee 
  (admission_fee : ℚ) 
  (new_skates_cost : ℚ) 
  (visits_to_justify : ℕ) 
  (h1 : admission_fee = 5)
  (h2 : new_skates_cost = 65)
  (h3 : visits_to_justify = 26) :
  let rental_fee := (new_skates_cost + admission_fee * visits_to_justify) / visits_to_justify - admission_fee
  rental_fee = (5/2 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_ice_skate_rental_fee_l1791_179164


namespace NUMINAMATH_CALUDE_power_multiplication_l1791_179170

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1791_179170


namespace NUMINAMATH_CALUDE_last_locker_opened_l1791_179198

/-- Represents the process of opening lockers in a hall -/
def openLockers (n : ℕ) : ℕ :=
  n - 2

/-- Theorem stating that the last locker opened is number 727 -/
theorem last_locker_opened (total_lockers : ℕ) (h : total_lockers = 729) :
  openLockers total_lockers = 727 :=
by sorry

end NUMINAMATH_CALUDE_last_locker_opened_l1791_179198


namespace NUMINAMATH_CALUDE_jim_bought_three_pictures_l1791_179199

def total_pictures : ℕ := 10
def probability_not_bought : ℚ := 21/45

theorem jim_bought_three_pictures :
  ∀ x : ℕ,
  x ≤ total_pictures →
  (total_pictures - x : ℚ) * (total_pictures - 1 - x) / (total_pictures * (total_pictures - 1)) = probability_not_bought →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_jim_bought_three_pictures_l1791_179199


namespace NUMINAMATH_CALUDE_palace_visitors_l1791_179163

/-- The number of visitors to Buckingham Palace over two days -/
def total_visitors (day1 : ℕ) (day2 : ℕ) : ℕ := day1 + day2

/-- Theorem stating the total number of visitors over two days -/
theorem palace_visitors : total_visitors 583 246 = 829 := by
  sorry

end NUMINAMATH_CALUDE_palace_visitors_l1791_179163


namespace NUMINAMATH_CALUDE_intersection_symmetry_l1791_179106

/-- Prove that if a line y = kx intersects a circle (x-1)^2 + y^2 = 1 at two points 
    symmetric with respect to the line x - y + b = 0, then k = -1 and b = -1. -/
theorem intersection_symmetry (k b : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    -- Line equation
    y₁ = k * x₁ ∧ y₂ = k * x₂ ∧
    -- Circle equation
    (x₁ - 1)^2 + y₁^2 = 1 ∧ (x₂ - 1)^2 + y₂^2 = 1 ∧
    -- Symmetry condition
    (x₁ + x₂) / 2 - (y₁ + y₂) / 2 + b = 0) →
  k = -1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_symmetry_l1791_179106


namespace NUMINAMATH_CALUDE_competition_probabilities_l1791_179184

-- Define the possible grades
inductive Grade : Type
  | Qualified
  | Good
  | Excellent

-- Define the probabilities for each participant
def probA : Grade → ℝ
  | Grade.Qualified => 0.6
  | Grade.Good => 0.3
  | Grade.Excellent => 0.1

def probB : Grade → ℝ
  | Grade.Qualified => 0.4
  | Grade.Good => 0.4
  | Grade.Excellent => 0.2

-- Define a function to check if one grade is higher than another
def isHigher : Grade → Grade → Bool
  | Grade.Excellent, Grade.Excellent => false
  | Grade.Excellent, _ => true
  | Grade.Good, Grade.Excellent => false
  | Grade.Good, _ => true
  | Grade.Qualified, Grade.Qualified => false
  | Grade.Qualified, _ => false

-- Define the probability that A's grade is higher than B's in one round
def probAHigherThanB : ℝ := 0.2

-- Define the probability that A's grade is higher than B's in at least two out of three rounds
def probAHigherThanBTwiceInThree : ℝ := 0.104

theorem competition_probabilities :
  (probAHigherThanB = 0.2) ∧
  (probAHigherThanBTwiceInThree = 0.104) := by
  sorry


end NUMINAMATH_CALUDE_competition_probabilities_l1791_179184


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l1791_179160

theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < 0 ∧ 0 < x₂ ∧ y₁ = 3 / x₁ ∧ y₂ = 3 / x₂ → y₁ < 0 ∧ 0 < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l1791_179160


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l1791_179127

/-- The amount Joan spent on shorts -/
def shorts_cost : ℚ := 15

/-- The amount Joan spent on a shirt -/
def shirt_cost : ℚ := 12.51

/-- The total amount Joan spent on clothing -/
def total_cost : ℚ := 42.33

/-- The amount Joan spent on the jacket -/
def jacket_cost : ℚ := total_cost - shorts_cost - shirt_cost

theorem jacket_cost_calculation : jacket_cost = 14.82 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l1791_179127


namespace NUMINAMATH_CALUDE_prime_square_sum_equation_l1791_179111

theorem prime_square_sum_equation :
  ∃ (p q r : ℕ), Prime p ∧ Prime q ∧ Prime r ∧ p^2 + 1 = q^2 + r^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_sum_equation_l1791_179111


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l1791_179132

theorem unique_solution_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 5*a| ≤ 3) ↔ (a = 3/4 ∨ a = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l1791_179132


namespace NUMINAMATH_CALUDE_highest_power_of_five_l1791_179166

def M : ℕ := sorry

/-- The highest power of 5 that divides M -/
def j : ℕ := sorry

/-- M is formed by concatenating all 2-digit integers from 21 to 87 -/
axiom M_def : M = sorry

/-- j is the highest power of 5 that divides M -/
axiom j_def : ∀ k : ℕ, (5^k ∣ M) → k ≤ j

/-- There exists a number that when multiplied by 5^j equals M -/
axiom j_divides : ∃ n : ℕ, M = 5^j * n

/-- 5^(j+1) does not divide M -/
axiom j_highest : ¬ (5^(j+1) ∣ M)

theorem highest_power_of_five : j = 1 := sorry

end NUMINAMATH_CALUDE_highest_power_of_five_l1791_179166


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1791_179149

theorem hyperbola_real_axis_length
  (p : ℝ)
  (a b : ℝ)
  (h_p_pos : p > 0)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_directrix_tangent : 3 + p / 2 = 15)
  (h_asymptote : b / a = Real.sqrt 3)
  (h_focus : a^2 + b^2 = 144) :
  2 * a = 12 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1791_179149


namespace NUMINAMATH_CALUDE_original_equals_scientific_l1791_179136

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 7003000

/-- The scientific notation representation of the number -/
def scientific_form : ScientificNotation :=
  { coefficient := 7.003
    exponent := 6
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l1791_179136


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1791_179105

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane defined by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is on a line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The specific line l: x + y - 1 = 0 -/
def lineL : Line := { a := 1, b := 1, c := -1 }

/-- The specific point condition x = 2 and y = -1 -/
def specificPoint : Point := { x := 2, y := -1 }

theorem sufficient_not_necessary_condition :
  (∀ p : Point, p = specificPoint → isOnLine p lineL) ∧
  ¬(∀ p : Point, isOnLine p lineL → p = specificPoint) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1791_179105


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1791_179159

open Real

theorem negation_of_universal_proposition :
  (¬ ∀ m : ℝ, ∃ x : ℝ, x^2 + x + m = 0) ↔ 
  (∃ m : ℝ, ∀ x : ℝ, x^2 + x + m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1791_179159


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l1791_179140

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - x - 1) * Real.exp (a * x)

theorem tangent_line_at_origin (a : ℝ) (h : a ≠ 0) :
  let tangent_line (x : ℝ) := -x - 1
  ∀ x, tangent_line x = f a 0 + (deriv (f a)) 0 * x :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l1791_179140


namespace NUMINAMATH_CALUDE_ordered_pair_solution_l1791_179180

theorem ordered_pair_solution :
  ∀ (c d : ℤ),
  Real.sqrt (16 - 12 * Real.cos (30 * π / 180)) = c + d * (1 / Real.cos (30 * π / 180)) →
  c = 4 ∧ d = -1 := by
sorry

end NUMINAMATH_CALUDE_ordered_pair_solution_l1791_179180


namespace NUMINAMATH_CALUDE_profit_maximization_l1791_179123

noncomputable def profit (x : ℝ) : ℝ := 20 - x - 4 / (x + 1)

theorem profit_maximization (a : ℝ) (h_a : a > 0) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ a^2 - 3*a + 3 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ a^2 - 3*a + 3 → profit x ≥ profit y) ∧
  ((a ≥ 2 ∨ 0 < a ∧ a ≤ 1) → x = 1) ∧
  (1 < a ∧ a < 2 → x = a^2 - 3*a + 3) :=
sorry

end NUMINAMATH_CALUDE_profit_maximization_l1791_179123


namespace NUMINAMATH_CALUDE_solve_for_y_l1791_179133

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = -8) : y = 82 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1791_179133


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l1791_179102

theorem least_sum_of_bases : ∃ (c d : ℕ+), 
  (∀ (c' d' : ℕ+), (2 * c' + 9 = 9 * d' + 2) → (c'.val + d'.val ≥ c.val + d.val)) ∧ 
  (2 * c + 9 = 9 * d + 2) ∧
  (c.val + d.val = 13) := by
  sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l1791_179102


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1791_179122

theorem sum_of_roots_quadratic (m n : ℝ) : 
  (m ^ 2 - 3 * m - 1 = 0) → (n ^ 2 - 3 * n - 1 = 0) → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1791_179122


namespace NUMINAMATH_CALUDE_min_tetrahedra_decomposition_l1791_179129

/-- A tetrahedron is a polyhedron with four triangular faces -/
structure Tetrahedron

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube

/-- Represents a decomposition of a cube into tetrahedra -/
structure CubeDecomposition (c : Cube) where
  tetrahedra : Finset Tetrahedron
  is_valid : Bool  -- This would be a complex condition in practice

/-- The number of tetrahedra in a decomposition -/
def num_tetrahedra (d : CubeDecomposition c) : Nat :=
  d.tetrahedra.card

/-- A predicate that checks if a decomposition is minimal -/
def is_minimal_decomposition (d : CubeDecomposition c) : Prop :=
  ∀ d' : CubeDecomposition c, num_tetrahedra d ≤ num_tetrahedra d'

theorem min_tetrahedra_decomposition (c : Cube) :
  ∃ (d : CubeDecomposition c), is_minimal_decomposition d ∧ num_tetrahedra d = 5 :=
sorry

end NUMINAMATH_CALUDE_min_tetrahedra_decomposition_l1791_179129


namespace NUMINAMATH_CALUDE_becketts_age_l1791_179154

theorem becketts_age (beckett olaf shannen jack : ℕ) 
  (h1 : beckett = olaf - 3)
  (h2 : shannen = olaf - 2)
  (h3 : jack = 2 * shannen + 5)
  (h4 : beckett + olaf + shannen + jack = 71) :
  beckett = 12 := by
  sorry

end NUMINAMATH_CALUDE_becketts_age_l1791_179154


namespace NUMINAMATH_CALUDE_largest_gold_coins_l1791_179113

theorem largest_gold_coins (n : ℕ) : 
  (∃ k : ℕ, n = 13 * k + 3) ∧ 
  n < 150 → 
  n ≤ 146 ∧ 
  (∃ m : ℕ, m > n ∧ (∃ j : ℕ, m = 13 * j + 3) → m ≥ 150) := by
sorry

end NUMINAMATH_CALUDE_largest_gold_coins_l1791_179113


namespace NUMINAMATH_CALUDE_budget_increase_is_twenty_percent_l1791_179147

/-- The percentage increase in the gym budget -/
def budget_increase_percentage (original_dodgeball_count : ℕ) (dodgeball_price : ℚ)
  (new_softball_count : ℕ) (softball_price : ℚ) : ℚ :=
  let original_budget := original_dodgeball_count * dodgeball_price
  let new_budget := new_softball_count * softball_price
  ((new_budget - original_budget) / original_budget) * 100

/-- Theorem stating that the budget increase percentage is 20% -/
theorem budget_increase_is_twenty_percent :
  budget_increase_percentage 15 5 10 9 = 20 := by
  sorry

end NUMINAMATH_CALUDE_budget_increase_is_twenty_percent_l1791_179147


namespace NUMINAMATH_CALUDE_exponent_difference_equals_204_l1791_179103

theorem exponent_difference_equals_204 : 3^(1*(2+3)) - (3^1 + 3^2 + 3^3) = 204 := by
  sorry

end NUMINAMATH_CALUDE_exponent_difference_equals_204_l1791_179103


namespace NUMINAMATH_CALUDE_tan_equality_solution_l1791_179143

open Real

theorem tan_equality_solution (x : ℝ) : 
  0 ≤ x ∧ x ≤ 180 ∧ 
  tan (150 * π / 180 - x * π / 180) = 
    (sin (150 * π / 180) - sin (x * π / 180)) / 
    (cos (150 * π / 180) - cos (x * π / 180)) →
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_solution_l1791_179143


namespace NUMINAMATH_CALUDE_right_triangle_ratio_square_l1791_179121

theorem right_triangle_ratio_square (a c p : ℝ) (h1 : a > 0) (h2 : c > 0) (h3 : p > 0) : 
  (c / a = a / p) → (c^2 = a^2 + p^2) → ((c / a)^2 = (1 + Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_square_l1791_179121


namespace NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l1791_179120

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 175 → ∃ (a b : ℕ), a^2 - b^2 = 175 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 625 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l1791_179120


namespace NUMINAMATH_CALUDE_smallest_inverse_mod_2100_eleven_has_inverse_mod_2100_eleven_is_smallest_with_inverse_mod_2100_l1791_179169

theorem smallest_inverse_mod_2100 : 
  ∀ n : ℕ, n > 1 → n < 11 → ¬(Nat.gcd n 2100 = 1) :=
sorry

theorem eleven_has_inverse_mod_2100 : Nat.gcd 11 2100 = 1 :=
sorry

theorem eleven_is_smallest_with_inverse_mod_2100 : 
  ∀ n : ℕ, n > 1 → Nat.gcd n 2100 = 1 → n ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_mod_2100_eleven_has_inverse_mod_2100_eleven_is_smallest_with_inverse_mod_2100_l1791_179169


namespace NUMINAMATH_CALUDE_mass_percentage_iodine_value_of_x_l1791_179130

-- Define constants for molar masses
def molar_mass_Al : ℝ := 26.98
def molar_mass_I : ℝ := 126.90
def molar_mass_H2O : ℝ := 18.015

-- Define the sample mass
def sample_mass : ℝ := 50

-- Define variables for masses of AlI₃ and H₂O in the sample
variable (mass_AlI3 : ℝ)
variable (mass_H2O : ℝ)

-- Calculate molar mass of AlI₃
def molar_mass_AlI3 : ℝ := molar_mass_Al + 3 * molar_mass_I

-- Define the theorem for mass percentage of iodine
theorem mass_percentage_iodine :
  let mass_iodine := mass_AlI3 * (3 * molar_mass_I / molar_mass_AlI3)
  (mass_iodine / sample_mass) * 100 = 
  (mass_AlI3 * (3 * molar_mass_I / molar_mass_AlI3) / sample_mass) * 100 :=
by sorry

-- Define the theorem for the value of x
theorem value_of_x :
  let moles_water := mass_H2O / molar_mass_H2O
  let moles_AlI3 := mass_AlI3 / molar_mass_AlI3
  (moles_water / moles_AlI3) = 
  (mass_H2O / molar_mass_H2O) / (mass_AlI3 / molar_mass_AlI3) :=
by sorry

end NUMINAMATH_CALUDE_mass_percentage_iodine_value_of_x_l1791_179130


namespace NUMINAMATH_CALUDE_distance_center_to_point_l1791_179192

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y + 5

/-- The center of the circle -/
def circle_center : ℝ × ℝ := sorry

/-- The given point -/
def given_point : ℝ × ℝ := (8, -3)

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem distance_center_to_point :
  distance circle_center given_point = 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_center_to_point_l1791_179192


namespace NUMINAMATH_CALUDE_combination_equation_solution_l1791_179181

theorem combination_equation_solution : 
  ∃! (n : ℕ), n > 0 ∧ Nat.choose (n + 1) (n - 1) = 21 := by sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l1791_179181


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l1791_179148

theorem dennis_teaching_years 
  (total_years : ℕ) 
  (virginia_adrienne_diff : ℕ) 
  (dennis_virginia_diff : ℕ) 
  (h1 : total_years = 75)
  (h2 : virginia_adrienne_diff = 9)
  (h3 : dennis_virginia_diff = 9) :
  ∃ (adrienne virginia dennis : ℕ),
    adrienne + virginia + dennis = total_years ∧
    virginia = adrienne + virginia_adrienne_diff ∧
    dennis = virginia + dennis_virginia_diff ∧
    dennis = 34 := by
  sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l1791_179148


namespace NUMINAMATH_CALUDE_x_values_in_A_l1791_179162

def A (x : ℝ) : Set ℝ := {-3, x + 2, x^2 - 4*x}

theorem x_values_in_A (x : ℝ) : 5 ∈ A x ↔ x = -1 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_values_in_A_l1791_179162


namespace NUMINAMATH_CALUDE_restaurant_production_l1791_179114

/-- Represents a restaurant's daily production of pizzas and hot dogs -/
structure Restaurant where
  hotdogs : ℕ
  pizza_excess : ℕ

/-- Calculates the total number of pizzas and hot dogs made in a given number of days -/
def total_production (r : Restaurant) (days : ℕ) : ℕ :=
  (r.hotdogs + (r.hotdogs + r.pizza_excess)) * days

/-- Theorem stating that a restaurant making 40 more pizzas than hot dogs daily,
    and 60 hot dogs per day, will produce 4800 pizzas and hot dogs in 30 days -/
theorem restaurant_production :
  ∀ (r : Restaurant),
    r.hotdogs = 60 →
    r.pizza_excess = 40 →
    total_production r 30 = 4800 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_production_l1791_179114


namespace NUMINAMATH_CALUDE_abc_sum_mod_five_l1791_179188

theorem abc_sum_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 5 = 1 →
  (3 * c) % 5 = 1 →
  (4 * b) % 5 = (1 + b) % 5 →
  (a + b + c) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_mod_five_l1791_179188


namespace NUMINAMATH_CALUDE_water_requirement_l1791_179182

/-- Represents the amount of a substance in moles -/
def Moles : Type := ℝ

/-- Represents a chemical reaction between ammonium chloride and water -/
structure Reaction where
  nh4cl : Moles  -- Amount of ammonium chloride
  h2o : Moles    -- Amount of water
  hcl : Moles    -- Amount of hydrochloric acid produced
  nh4oh : Moles  -- Amount of ammonium hydroxide produced

/-- The reaction is balanced when the amounts of reactants and products are in the correct proportion -/
def is_balanced (r : Reaction) : Prop :=
  r.nh4cl = r.h2o ∧ r.nh4cl = r.hcl ∧ r.nh4cl = r.nh4oh

/-- The amount of water required is equal to the amount of ammonium chloride when the reaction is balanced -/
theorem water_requirement (r : Reaction) (h : is_balanced r) : r.h2o = r.nh4cl := by
  sorry

end NUMINAMATH_CALUDE_water_requirement_l1791_179182


namespace NUMINAMATH_CALUDE_digits_of_product_l1791_179175

theorem digits_of_product : ∃ n : ℕ, n > 0 ∧ 10^(n-1) ≤ 2^15 * 5^10 * 3 ∧ 2^15 * 5^10 * 3 < 10^n ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_product_l1791_179175


namespace NUMINAMATH_CALUDE_yellow_yellow_pairs_l1791_179165

theorem yellow_yellow_pairs
  (total_students : ℕ)
  (blue_students : ℕ)
  (yellow_students : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ)
  (h1 : total_students = 150)
  (h2 : blue_students = 65)
  (h3 : yellow_students = 85)
  (h4 : total_students = blue_students + yellow_students)
  (h5 : total_pairs = 75)
  (h6 : blue_blue_pairs = 30) :
  ∃ (yellow_yellow_pairs : ℕ),
    yellow_yellow_pairs = 40 ∧
    total_pairs = blue_blue_pairs + yellow_yellow_pairs + (blue_students - 2 * blue_blue_pairs) :=
by sorry


end NUMINAMATH_CALUDE_yellow_yellow_pairs_l1791_179165


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1791_179187

/-- Given a triangle with two sides of lengths 3 and 4, and the third side being the root
    of x^2 - 12x + 35 = 0 that satisfies the triangle inequality, the perimeter is 12. -/
theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 12*x + 35 = 0 →
  x > 0 →
  x < 3 + 4 →
  x > |3 - 4| →
  3 + 4 + x = 12 := by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l1791_179187


namespace NUMINAMATH_CALUDE_forty_nine_squared_equals_seven_to_zero_l1791_179179

theorem forty_nine_squared_equals_seven_to_zero : 49 * 49 = 7^0 := by
  sorry

end NUMINAMATH_CALUDE_forty_nine_squared_equals_seven_to_zero_l1791_179179


namespace NUMINAMATH_CALUDE_number_of_divisors_of_60_l1791_179108

theorem number_of_divisors_of_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_60_l1791_179108


namespace NUMINAMATH_CALUDE_equation_linearity_implies_m_n_values_l1791_179101

/-- A linear equation in two variables has the form ax + by = c, where a, b, and c are constants -/
def is_linear_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

/-- The equation 3x^(2m+1) - 2y^(n-1) = 7 -/
def equation (m n : ℕ) (x y : ℝ) : ℝ :=
  3 * x^(2*m+1) - 2 * y^(n-1) - 7

theorem equation_linearity_implies_m_n_values (m n : ℕ) :
  is_linear_in_two_variables (equation m n) → m = 0 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_linearity_implies_m_n_values_l1791_179101


namespace NUMINAMATH_CALUDE_tv_sales_decrease_l1791_179174

theorem tv_sales_decrease (original_price original_quantity : ℝ) 
  (original_price_positive : original_price > 0)
  (original_quantity_positive : original_quantity > 0) :
  let new_price := 1.30 * original_price
  let new_revenue := 1.04 * (original_price * original_quantity)
  let sales_decrease_percentage := 
    100 * (1 - (new_revenue / new_price) / original_quantity)
  sales_decrease_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_tv_sales_decrease_l1791_179174


namespace NUMINAMATH_CALUDE_necessary_condition_not_sufficient_condition_necessary_but_not_sufficient_l1791_179144

/-- Represents an ellipse equation with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (6 - m) = 1 ∧ m ≠ 4

/-- The condition 2 < m < 6 is necessary for the equation to represent an ellipse -/
theorem necessary_condition (m : ℝ) :
  is_ellipse m → 2 < m ∧ m < 6 := by sorry

/-- The condition 2 < m < 6 is not sufficient for the equation to represent an ellipse -/
theorem not_sufficient_condition :
  ∃ m : ℝ, 2 < m ∧ m < 6 ∧ ¬(is_ellipse m) := by sorry

/-- The main theorem stating that 2 < m < 6 is necessary but not sufficient -/
theorem necessary_but_not_sufficient :
  (∀ m : ℝ, is_ellipse m → 2 < m ∧ m < 6) ∧
  (∃ m : ℝ, 2 < m ∧ m < 6 ∧ ¬(is_ellipse m)) := by sorry

end NUMINAMATH_CALUDE_necessary_condition_not_sufficient_condition_necessary_but_not_sufficient_l1791_179144


namespace NUMINAMATH_CALUDE_total_crayons_l1791_179119

theorem total_crayons (boxes : ℕ) (crayons_per_box : ℕ) (h1 : boxes = 8) (h2 : crayons_per_box = 7) :
  boxes * crayons_per_box = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l1791_179119


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1791_179157

theorem arithmetic_sequence_problem (d a_n n : ℤ) (h1 : d = 2) (h2 : n = 15) (h3 : a_n = -10) :
  ∃ (a_1 S_n : ℤ),
    a_1 = -38 ∧
    S_n = -360 ∧
    a_n = a_1 + (n - 1) * d ∧
    S_n = n * (a_1 + a_n) / 2 :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1791_179157


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1791_179178

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt (x^2 + (y + 3)^2) + Real.sqrt (x^2 + (y - 3)^2) = 10) ↔
  (x^2 / 25 + y^2 / 16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1791_179178


namespace NUMINAMATH_CALUDE_equation_graph_is_two_lines_l1791_179141

/-- The set of points (x, y) satisfying the equation (x-2y)^2 = x^2 + y^2 is equivalent to the union of two lines: y = 0 and y = 4x/3 -/
theorem equation_graph_is_two_lines :
  {p : ℝ × ℝ | (p.1 - 2 * p.2)^2 = p.1^2 + p.2^2} =
  {p : ℝ × ℝ | p.2 = 0 ∨ p.2 = (4/3) * p.1} := by sorry

end NUMINAMATH_CALUDE_equation_graph_is_two_lines_l1791_179141


namespace NUMINAMATH_CALUDE_hcf_lcm_sum_reciprocal_l1791_179109

theorem hcf_lcm_sum_reciprocal (m n : ℕ+) : 
  Nat.gcd m.val n.val = 6 → 
  Nat.lcm m.val n.val = 210 → 
  m.val + n.val = 60 → 
  (1 : ℚ) / m.val + (1 : ℚ) / n.val = 1 / 21 := by
sorry

end NUMINAMATH_CALUDE_hcf_lcm_sum_reciprocal_l1791_179109


namespace NUMINAMATH_CALUDE_hockey_players_count_l1791_179145

/-- The number of hockey players in a games hour -/
def hockey_players (total players cricket football softball : ℕ) : ℕ :=
  total - (cricket + football + softball)

/-- Theorem: There are 17 hockey players in the ground -/
theorem hockey_players_count : hockey_players 50 12 11 10 = 17 := by
  sorry

end NUMINAMATH_CALUDE_hockey_players_count_l1791_179145


namespace NUMINAMATH_CALUDE_tims_score_is_56_l1791_179155

/-- The sum of the first n even numbers -/
def sum_first_n_even (n : ℕ) : ℕ :=
  (2 * n * (n + 1)) / 2

/-- Tim's math score -/
def tims_score : ℕ := sum_first_n_even 7

theorem tims_score_is_56 : tims_score = 56 := by
  sorry

end NUMINAMATH_CALUDE_tims_score_is_56_l1791_179155


namespace NUMINAMATH_CALUDE_largest_negative_angle_solution_l1791_179173

theorem largest_negative_angle_solution :
  let θ : ℝ := -π/2
  let eq (x : ℝ) := (1 - Real.sin x + Real.cos x) / (1 - Real.sin x - Real.cos x) +
                    (1 - Real.sin x - Real.cos x) / (1 - Real.sin x + Real.cos x) = 2
  (eq θ) ∧ 
  (∀ φ, φ < 0 → φ > θ → ¬(eq φ)) :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_angle_solution_l1791_179173


namespace NUMINAMATH_CALUDE_distance_to_complex_point_l1791_179124

open Complex

theorem distance_to_complex_point :
  let z : ℂ := 3 / (2 - I)^2
  abs z = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_distance_to_complex_point_l1791_179124


namespace NUMINAMATH_CALUDE_problem_statement_l1791_179177

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ x * y > a * b) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → x^2 + y^2 ≥ 1/2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → 4/x + 1/y ≥ 9) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ Real.sqrt x + Real.sqrt y < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1791_179177


namespace NUMINAMATH_CALUDE_rectangle_diagonal_length_l1791_179107

/-- The length of the diagonal of a rectangle with specific properties -/
theorem rectangle_diagonal_length : ∀ (a b d : ℝ), 
  a > 0 → 
  b = 2 * a → 
  a = 40 * Real.sqrt 2 → 
  d^2 = a^2 + b^2 → 
  d = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_length_l1791_179107


namespace NUMINAMATH_CALUDE_max_sum_problem_l1791_179131

theorem max_sum_problem (x y z v w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_v : v > 0) (pos_w : w > 0)
  (sum_cubes : x^3 + y^3 + z^3 + v^3 + w^3 = 2024) : 
  ∃ (M x_M y_M z_M v_M w_M : ℝ),
    (∀ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ 
      a^3 + b^3 + c^3 + d^3 + e^3 = 2024 → 
      a*c + 3*b*c + 4*c*d + 8*c*e ≤ M) ∧
    x_M > 0 ∧ y_M > 0 ∧ z_M > 0 ∧ v_M > 0 ∧ w_M > 0 ∧
    x_M^3 + y_M^3 + z_M^3 + v_M^3 + w_M^3 = 2024 ∧
    x_M*z_M + 3*y_M*z_M + 4*z_M*v_M + 8*z_M*w_M = M ∧
    M + x_M + y_M + z_M + v_M + w_M = 3055 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_problem_l1791_179131


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l1791_179191

theorem farmer_tomatoes (T : ℕ) : 
  T - 53 + 12 = 136 → T = 71 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l1791_179191


namespace NUMINAMATH_CALUDE_simplify_expression_l1791_179104

theorem simplify_expression : (512 : ℝ)^(1/3) * (343 : ℝ)^(1/2) = 56 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1791_179104


namespace NUMINAMATH_CALUDE_circle_construction_l1791_179189

/-- Given a circle k0 with diameter AB and center O0, and additional circles k1, k2, k3, k4, k5, k6
    constructed as described in the problem, prove that their radii are in specific ratios to r0. -/
theorem circle_construction (r0 : ℝ) (r1 r2 r3 r4 r5 r6 : ℝ) 
  (h1 : r0 > 0)
  (h2 : r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0 ∧ r5 > 0 ∧ r6 > 0)
  (h3 : ∃ (A B O0 : ℝ × ℝ), ‖A - B‖ = 2 * r0 ∧ O0 = (A + B) / 2)
  (h4 : ∃ (k1 k1' : Set (ℝ × ℝ)), k1 ∩ k1' = {O0}) :
  r1 = r0 / 2 ∧ r2 = r0 / 3 ∧ r3 = r0 / 6 ∧ r4 = r0 / 4 ∧ r5 = r0 / 7 ∧ r6 = r0 / 8 := by
  sorry


end NUMINAMATH_CALUDE_circle_construction_l1791_179189


namespace NUMINAMATH_CALUDE_max_sum_surrounding_45_l1791_179126

theorem max_sum_surrounding_45 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℕ) : 
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧ a₁ ≠ a₈ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧ a₂ ≠ a₈ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧ a₃ ≠ a₈ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧ a₄ ≠ a₈ ∧
  a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧ a₅ ≠ a₈ ∧
  a₆ ≠ a₇ ∧ a₆ ≠ a₈ ∧
  a₇ ≠ a₈ ∧
  0 < a₁ ∧ 0 < a₂ ∧ 0 < a₃ ∧ 0 < a₄ ∧ 0 < a₅ ∧ 0 < a₆ ∧ 0 < a₇ ∧ 0 < a₈ ∧
  a₁ * 45 * a₅ = 3240 ∧
  a₂ * 45 * a₆ = 3240 ∧
  a₃ * 45 * a₇ = 3240 ∧
  a₄ * 45 * a₈ = 3240 →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ ≤ 160 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_surrounding_45_l1791_179126


namespace NUMINAMATH_CALUDE_rectangle_side_length_l1791_179158

/-- Given a rectangle with area 9a^2 - 6ab + 3a and one side length 3a, 
    the other side length is 3a - 2b + 1 -/
theorem rectangle_side_length (a b : ℝ) : 
  let area := 9*a^2 - 6*a*b + 3*a
  let side1 := 3*a
  let side2 := 3*a - 2*b + 1
  area = side1 * side2 := by sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l1791_179158


namespace NUMINAMATH_CALUDE_find_y_l1791_179183

theorem find_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) 
  (h : (2 * a) ^ (4 * b) = a ^ b * y ^ (3 * b)) : 
  y = 2 ^ (4 / 3) * a :=
sorry

end NUMINAMATH_CALUDE_find_y_l1791_179183


namespace NUMINAMATH_CALUDE_donna_marcia_pencils_l1791_179151

/-- The number of pencils Cindi bought -/
def cindi_pencils : ℕ := 60

/-- The number of pencils Marcia bought -/
def marcia_pencils : ℕ := 2 * cindi_pencils

/-- The number of pencils Donna bought -/
def donna_pencils : ℕ := 3 * marcia_pencils

/-- The total number of pencils bought by Donna and Marcia -/
def total_pencils : ℕ := donna_pencils + marcia_pencils

theorem donna_marcia_pencils :
  total_pencils = 480 :=
sorry

end NUMINAMATH_CALUDE_donna_marcia_pencils_l1791_179151


namespace NUMINAMATH_CALUDE_exact_sixty_possible_greater_than_sixty_possible_l1791_179125

/-- Represents the number of pieces a single piece of paper can be cut into -/
inductive Cut
  | eight : Cut
  | twelve : Cut

/-- Represents a sequence of cuts applied to the original piece of paper -/
def CutSequence := List Cut

/-- Calculates the number of pieces resulting from applying a sequence of cuts -/
def num_pieces (cuts : CutSequence) : ℕ :=
  cuts.foldl (λ acc cut => match cut with
    | Cut.eight => acc * 8
    | Cut.twelve => acc * 12) 1

/-- Theorem stating that it's possible to obtain exactly 60 pieces -/
theorem exact_sixty_possible : ∃ (cuts : CutSequence), num_pieces cuts = 60 := by
  sorry

/-- Theorem stating that it's possible to obtain any number of pieces greater than 60 -/
theorem greater_than_sixty_possible (n : ℕ) (h : n > 60) : 
  ∃ (cuts : CutSequence), num_pieces cuts = n := by
  sorry

end NUMINAMATH_CALUDE_exact_sixty_possible_greater_than_sixty_possible_l1791_179125


namespace NUMINAMATH_CALUDE_cubic_function_has_three_roots_l1791_179112

-- Define the cubic function
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem cubic_function_has_three_roots :
  ∃ (a b c : ℝ), a < b ∧ b < c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x, f x = 0 → x = a ∨ x = b ∨ x = c :=
sorry

end NUMINAMATH_CALUDE_cubic_function_has_three_roots_l1791_179112


namespace NUMINAMATH_CALUDE_smallest_n_is_25_l1791_179197

/-- Represents a student's answers as a 5-tuple of integers from 1 to 4 -/
def Answer := Fin 5 → Fin 4

/-- The set of all possible answer patterns satisfying the modular constraint -/
def S : Set Answer :=
  {a | (a 0).val + (a 1).val + (a 2).val + (a 3).val + (a 4).val ≡ 0 [MOD 4]}

/-- The number of students -/
def num_students : ℕ := 2000

/-- The function that checks if two answers differ in at least two positions -/
def differ_in_two (a b : Answer) : Prop :=
  ∃ i j, i ≠ j ∧ a i ≠ b i ∧ a j ≠ b j

/-- The theorem to be proved -/
theorem smallest_n_is_25 :
  ∀ f : Fin num_students → Answer,
  ∃ n : ℕ, n = 25 ∧
  (∀ subset : Fin n → Fin num_students,
   ∃ a b c d : Fin n,
   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   differ_in_two (f (subset a)) (f (subset b)) ∧
   differ_in_two (f (subset a)) (f (subset c)) ∧
   differ_in_two (f (subset a)) (f (subset d)) ∧
   differ_in_two (f (subset b)) (f (subset c)) ∧
   differ_in_two (f (subset b)) (f (subset d)) ∧
   differ_in_two (f (subset c)) (f (subset d))) ∧
  (∀ m : ℕ, m < 25 →
   ∃ f : Fin num_students → Answer,
   ∀ subset : Fin m → Fin num_students,
   ¬∃ a b c d : Fin m,
   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   differ_in_two (f (subset a)) (f (subset b)) ∧
   differ_in_two (f (subset a)) (f (subset c)) ∧
   differ_in_two (f (subset a)) (f (subset d)) ∧
   differ_in_two (f (subset b)) (f (subset c)) ∧
   differ_in_two (f (subset b)) (f (subset d)) ∧
   differ_in_two (f (subset c)) (f (subset d))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_is_25_l1791_179197


namespace NUMINAMATH_CALUDE_right_to_left_grouping_l1791_179152

/-- A function that represents the right-to-left grouping evaluation of expressions -/
noncomputable def rightToLeftEval (a b c d : ℝ) : ℝ := a * (b + (c - d))

/-- The theorem stating that the right-to-left grouping evaluation is correct -/
theorem right_to_left_grouping (a b c d : ℝ) :
  rightToLeftEval a b c d = a * (b + c - d) := by sorry

end NUMINAMATH_CALUDE_right_to_left_grouping_l1791_179152


namespace NUMINAMATH_CALUDE_parametric_curve_extrema_l1791_179161

open Real

theorem parametric_curve_extrema :
  let x : ℝ → ℝ := λ t ↦ 2 * (1 + cos t) * cos t
  let y : ℝ → ℝ := λ t ↦ 2 * (1 + cos t) * sin t
  let t_domain := {t : ℝ | 0 ≤ t ∧ t ≤ 2 * π}
  (∀ t ∈ t_domain, x t ≤ 4) ∧
  (∃ t ∈ t_domain, x t = 4) ∧
  (∀ t ∈ t_domain, x t ≥ -1/2) ∧
  (∃ t ∈ t_domain, x t = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_parametric_curve_extrema_l1791_179161


namespace NUMINAMATH_CALUDE_inner_circle_radius_l1791_179196

theorem inner_circle_radius : 
  ∀ r : ℝ,
  (r > 0) →
  (π * (9^2) - π * ((0.75 * r)^2) = 3.6 * (π * 6^2 - π * r^2)) →
  r = 4 := by
sorry

end NUMINAMATH_CALUDE_inner_circle_radius_l1791_179196


namespace NUMINAMATH_CALUDE_office_network_connections_l1791_179110

/-- The number of connections in a network of switches where each switch is connected to a fixed number of other switches. -/
def network_connections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 30 switches, where each switch is directly connected to exactly 4 other switches, the total number of connections is 60. -/
theorem office_network_connections :
  network_connections 30 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_office_network_connections_l1791_179110


namespace NUMINAMATH_CALUDE_evaluate_f_l1791_179156

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

-- State the theorem
theorem evaluate_f : 3 * f 2 - 2 * f (-2) = -31 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l1791_179156


namespace NUMINAMATH_CALUDE_circle_properties_l1791_179116

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y - 4 = 0

-- Theorem statement
theorem circle_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_x = -2 ∧
    center_y = 1 ∧
    radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l1791_179116


namespace NUMINAMATH_CALUDE_percent_women_non_union_part_time_l1791_179193

/-- Represents the percentage of employees who are men -/
def percentMen : ℝ := 54

/-- Represents the percentage of employees who are women -/
def percentWomen : ℝ := 46

/-- Represents the percentage of men who work full-time -/
def percentMenFullTime : ℝ := 70

/-- Represents the percentage of men who work part-time -/
def percentMenPartTime : ℝ := 30

/-- Represents the percentage of women who work full-time -/
def percentWomenFullTime : ℝ := 60

/-- Represents the percentage of women who work part-time -/
def percentWomenPartTime : ℝ := 40

/-- Represents the percentage of full-time employees who are unionized -/
def percentFullTimeUnionized : ℝ := 60

/-- Represents the percentage of part-time employees who are unionized -/
def percentPartTimeUnionized : ℝ := 50

/-- The main theorem stating that given the conditions, 
    approximately 52.94% of non-union part-time employees are women -/
theorem percent_women_non_union_part_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧
  abs ((9 : ℝ) / 17 * 100 - 52.94) < ε := by
  sorry


end NUMINAMATH_CALUDE_percent_women_non_union_part_time_l1791_179193


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1791_179115

theorem unique_triple_solution (a b c : ℝ) : 
  a > 5 → b > 5 → c > 5 →
  (a + 3)^2 / (b + c - 3) + (b + 6)^2 / (c + a - 6) + (c + 9)^2 / (a + b - 9) = 81 →
  a = 15 ∧ b = 12 ∧ c = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1791_179115


namespace NUMINAMATH_CALUDE_fair_coin_three_tosses_two_heads_l1791_179118

/-- A fair coin is a coin with equal probability of landing on either side. -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of getting exactly k successes in n independent trials,
    each with probability p of success. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- Theorem: For a fair coin tossed 3 times, the probability of getting
    exactly 2 heads and 1 tail is 3/8. -/
theorem fair_coin_three_tosses_two_heads (p : ℝ) :
  fair_coin p → binomial_probability 3 2 p = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_three_tosses_two_heads_l1791_179118


namespace NUMINAMATH_CALUDE_data_transmission_time_l1791_179150

theorem data_transmission_time : 
  let num_blocks : ℕ := 60
  let chunks_per_block : ℕ := 512
  let transmission_rate : ℕ := 120  -- chunks per second
  let total_chunks : ℕ := num_blocks * chunks_per_block
  let transmission_time_seconds : ℕ := total_chunks / transmission_rate
  let transmission_time_minutes : ℕ := transmission_time_seconds / 60
  transmission_time_minutes = 4 := by
  sorry

end NUMINAMATH_CALUDE_data_transmission_time_l1791_179150


namespace NUMINAMATH_CALUDE_star_value_l1791_179176

def star (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 15) (h2 : a * b = 56) :
  star a b = 15 / 56 := by
sorry

end NUMINAMATH_CALUDE_star_value_l1791_179176


namespace NUMINAMATH_CALUDE_series_sum_eight_l1791_179135

def series_sum : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2^(n + 1) + series_sum n

theorem series_sum_eight : series_sum 8 = 510 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_eight_l1791_179135


namespace NUMINAMATH_CALUDE_winning_numbers_are_correct_l1791_179153

def winning_numbers : Set Nat :=
  {n : Nat | n ≥ 1 ∧ n ≤ 999 ∧ n % 100 = 88}

theorem winning_numbers_are_correct :
  winning_numbers = {88, 188, 288, 388, 488, 588, 688, 788, 888, 988} := by
  sorry

end NUMINAMATH_CALUDE_winning_numbers_are_correct_l1791_179153


namespace NUMINAMATH_CALUDE_garden_length_is_40_l1791_179128

/-- Represents a rectangular garden with given properties -/
structure Garden where
  total_distance : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ
  width_ratio : ℝ
  length : ℝ

/-- Theorem stating that the garden's length is 40 meters given the conditions -/
theorem garden_length_is_40 (g : Garden)
  (h1 : g.total_distance = 960)
  (h2 : g.length_walks = 24)
  (h3 : g.perimeter_walks = 8)
  (h4 : g.width_ratio = 1/2)
  (h5 : g.length * g.length_walks = g.total_distance)
  (h6 : (2 * g.length + 2 * (g.width_ratio * g.length)) * g.perimeter_walks = g.total_distance) :
  g.length = 40 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_is_40_l1791_179128


namespace NUMINAMATH_CALUDE_equation_solution_l1791_179194

theorem equation_solution (x : ℝ) : (x + 6) / (x - 3) = 4 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1791_179194


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l1791_179172

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x + a)(x - 4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

/-- If f(x) = (x + a)(x - 4) is an even function, then a = 4 -/
theorem even_function_implies_a_equals_four :
  ∀ a : ℝ, IsEven (f a) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l1791_179172


namespace NUMINAMATH_CALUDE_lucky_sum_equality_l1791_179139

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to form a sum of s using k distinct natural numbers less than n -/
def sumCombinations (s k n : ℕ) : ℕ := sorry

/-- The probability of event A: "the lucky sum in the main draw is 63" -/
def probA (n : ℕ) : ℚ :=
  (sumCombinations 63 10 n : ℚ) / choose n 10

/-- The probability of event B: "the lucky sum in the additional draw is 44" -/
def probB (n : ℕ) : ℚ :=
  (sumCombinations 44 8 n : ℚ) / choose n 8

theorem lucky_sum_equality :
  ∀ n : ℕ, (n ≥ 10 ∧ probA n = probB n) ↔ n = 18 := by sorry

end NUMINAMATH_CALUDE_lucky_sum_equality_l1791_179139


namespace NUMINAMATH_CALUDE_magic_sum_values_l1791_179171

/-- Represents a triangle configuration with 6 numbers -/
structure TriangleConfig where
  vertices : Fin 6 → Nat
  distinct : ∀ i j, i ≠ j → vertices i ≠ vertices j
  range : ∀ i, vertices i ∈ ({1, 2, 3, 4, 5, 6} : Set Nat)

/-- The sum of numbers on one side of the triangle -/
def sideSum (config : TriangleConfig) : Nat :=
  config.vertices 0 + config.vertices 1 + config.vertices 2

/-- All sides have the same sum -/
def validConfig (config : TriangleConfig) : Prop :=
  sideSum config = config.vertices 0 + config.vertices 3 + config.vertices 5 ∧
  sideSum config = config.vertices 2 + config.vertices 4 + config.vertices 5

theorem magic_sum_values :
  ∃ (config : TriangleConfig), validConfig config ∧
  sideSum config ∈ ({9, 10, 11, 12} : Set Nat) ∧
  ∀ (otherConfig : TriangleConfig),
    validConfig otherConfig →
    sideSum otherConfig ∈ ({9, 10, 11, 12} : Set Nat) := by
  sorry

end NUMINAMATH_CALUDE_magic_sum_values_l1791_179171


namespace NUMINAMATH_CALUDE_extracurricular_materials_selection_l1791_179168

theorem extracurricular_materials_selection (n : Nat) (k : Nat) (m : Nat) : 
  n = 6 → k = 2 → m = 1 → 
  (Nat.choose n m) * (m * (n - m) * (n - m - 1)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_extracurricular_materials_selection_l1791_179168


namespace NUMINAMATH_CALUDE_fruit_seller_problem_l1791_179186

/-- Represents the number of apples whose selling price equals the total gain -/
def reference_apples : ℕ := 50

/-- Represents the gain percent as a rational number -/
def gain_percent : ℚ := 100 / 3

/-- Calculates the number of apples sold given the reference apples and gain percent -/
def apples_sold (reference : ℕ) (gain : ℚ) : ℕ := sorry

theorem fruit_seller_problem :
  apples_sold reference_apples gain_percent = 200 := by sorry

end NUMINAMATH_CALUDE_fruit_seller_problem_l1791_179186


namespace NUMINAMATH_CALUDE_star_symmetric_set_eq_three_lines_l1791_179134

/-- The star operation -/
def star (a b : ℝ) : ℝ := a^2 * b + a * b^2

/-- The set of points (x, y) where x ★ y = y ★ x -/
def star_symmetric_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

/-- The union of three lines: x = 0, y = 0, and x + y = 0 -/
def three_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 + p.2 = 0}

theorem star_symmetric_set_eq_three_lines :
  star_symmetric_set = three_lines := by sorry

end NUMINAMATH_CALUDE_star_symmetric_set_eq_three_lines_l1791_179134


namespace NUMINAMATH_CALUDE_betty_balance_l1791_179138

/-- Betty's account balance given Gina's account information -/
theorem betty_balance (gina_account1 gina_account2 betty_balance : ℚ) : 
  gina_account1 = (1 / 4 : ℚ) * betty_balance →
  gina_account2 = (1 / 4 : ℚ) * betty_balance →
  gina_account1 + gina_account2 = 1728 →
  betty_balance = 3456 := by
  sorry

end NUMINAMATH_CALUDE_betty_balance_l1791_179138


namespace NUMINAMATH_CALUDE_removed_cone_height_l1791_179190

-- Define the frustum
structure Frustum where
  height : ℝ
  lowerBaseArea : ℝ
  upperBaseArea : ℝ

-- Define the theorem
theorem removed_cone_height (f : Frustum) (h1 : f.height = 30)
  (h2 : f.lowerBaseArea = 400 * Real.pi) (h3 : f.upperBaseArea = 100 * Real.pi) :
  ∃ (removedHeight : ℝ), removedHeight = f.height := by
  sorry

end NUMINAMATH_CALUDE_removed_cone_height_l1791_179190


namespace NUMINAMATH_CALUDE_quadratic_product_is_square_l1791_179185

/-- Given quadratic trinomials f and g satisfying the inequality condition,
    prove that their product is the square of some quadratic trinomial. -/
theorem quadratic_product_is_square
  (f g : ℝ → ℝ)
  (hf : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e h : ℝ, ∀ x, g x = d * x^2 + e * x + h)
  (h_ineq : ∀ x, (deriv f x) * (deriv g x) ≥ |f x| + |g x|) :
  ∃ (k : ℝ) (p : ℝ → ℝ),
    (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) ∧
    (∀ x, f x * g x = k * (p x)^2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_product_is_square_l1791_179185


namespace NUMINAMATH_CALUDE_tangent_parallel_to_chord_l1791_179117

/-- The curve function -/
def f (x : ℝ) : ℝ := 4*x - x^2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 - 2*x

theorem tangent_parallel_to_chord :
  let A : ℝ × ℝ := (4, 0)
  let B : ℝ × ℝ := (2, 4)
  let P : ℝ × ℝ := (3, 3)
  let chord_slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  P.2 = f P.1 ∧ f' P.1 = chord_slope := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_chord_l1791_179117


namespace NUMINAMATH_CALUDE_middle_share_in_ratio_l1791_179146

/-- Proves that in a 3:5:7 ratio distribution with a 1200 difference between extremes, the middle value is 1500 -/
theorem middle_share_in_ratio (total : ℕ) : 
  let f := 3 * total / 15
  let v := 5 * total / 15
  let r := 7 * total / 15
  r - f = 1200 → v = 1500 := by
  sorry

end NUMINAMATH_CALUDE_middle_share_in_ratio_l1791_179146
