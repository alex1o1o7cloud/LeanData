import Mathlib

namespace NUMINAMATH_CALUDE_constant_sign_of_root_combination_l918_91855

/-- Represents a polynomial of degree 4 -/
structure Polynomial4 where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ

/-- The roots of P(x) - t for a given t -/
def roots (P : Polynomial4) (t : ℝ) : Fin 4 → ℝ := sorry

/-- Predicate to check if P(x) - t has four distinct real roots -/
def has_four_distinct_real_roots (P : Polynomial4) (t : ℝ) : Prop := sorry

theorem constant_sign_of_root_combination (P : Polynomial4) :
  ∀ t₁ t₂ : ℝ, has_four_distinct_real_roots P t₁ → has_four_distinct_real_roots P t₂ →
  (roots P t₁ 0 + roots P t₁ 3 - roots P t₁ 1 - roots P t₁ 2) *
  (roots P t₂ 0 + roots P t₂ 3 - roots P t₂ 1 - roots P t₂ 2) > 0 :=
sorry

end NUMINAMATH_CALUDE_constant_sign_of_root_combination_l918_91855


namespace NUMINAMATH_CALUDE_gcd_16016_20020_l918_91802

theorem gcd_16016_20020 : Nat.gcd 16016 20020 = 4004 := by
  sorry

end NUMINAMATH_CALUDE_gcd_16016_20020_l918_91802


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l918_91875

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^3 + 1/x^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l918_91875


namespace NUMINAMATH_CALUDE_min_value_x_plus_3y_l918_91896

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 1) + 1 / (y + 3) = 1 / 4) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + 1) + 1 / (b + 3) = 1 / 4 → 
  x + 3 * y ≤ a + 3 * b ∧ x + 3 * y = 6 + 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_3y_l918_91896


namespace NUMINAMATH_CALUDE_bounded_recurrence_periodic_l918_91804

def is_bounded (x : ℕ → ℤ) : Prop :=
  ∃ M : ℕ, ∀ n, |x n| ≤ M

def recurrence_relation (x : ℕ → ℤ) : Prop :=
  ∀ n, x (n + 5) = (5 * (x (n + 4))^3 + x (n + 3) - 3 * x (n + 2) + x n) /
                   (2 * x (n + 2) + (x (n + 1))^2 + x (n + 1) * x n)

def eventually_periodic (x : ℕ → ℤ) : Prop :=
  ∃ k p : ℕ, p > 0 ∧ ∀ n ≥ k, x (n + p) = x n

theorem bounded_recurrence_periodic
  (x : ℕ → ℤ) (h_bounded : is_bounded x) (h_recurrence : recurrence_relation x) :
  eventually_periodic x :=
sorry

end NUMINAMATH_CALUDE_bounded_recurrence_periodic_l918_91804


namespace NUMINAMATH_CALUDE_complex_modulus_of_fraction_l918_91846

theorem complex_modulus_of_fraction (z : ℂ) : z = (2 - I) / (2 + I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_of_fraction_l918_91846


namespace NUMINAMATH_CALUDE_particle_in_semicircle_probability_l918_91861

theorem particle_in_semicircle_probability (AB BC : Real) (h1 : AB = 2) (h2 : BC = 1) :
  let rectangle_area := AB * BC
  let semicircle_radius := AB / 2
  let semicircle_area := π * semicircle_radius^2 / 2
  semicircle_area / rectangle_area = π / 4 := by sorry

end NUMINAMATH_CALUDE_particle_in_semicircle_probability_l918_91861


namespace NUMINAMATH_CALUDE_weeks_to_save_l918_91815

def console_cost : ℕ := 282
def game_cost : ℕ := 75
def initial_savings : ℕ := 42
def weekly_allowance : ℕ := 24

theorem weeks_to_save : ℕ := by
  -- The minimum number of whole weeks required to save enough money
  -- for both the console and the game is 14.
  sorry

end NUMINAMATH_CALUDE_weeks_to_save_l918_91815


namespace NUMINAMATH_CALUDE_units_digit_G_2000_l918_91834

/-- Definition of G_n -/
def G (n : ℕ) : ℕ := 2^(2^n) + 5^(5^n)

/-- Property of units digit for powers of 2 -/
axiom units_digit_power_2 (n : ℕ) : n % 4 = 0 → (2^(2^n)) % 10 = 6

/-- Property of units digit for powers of 5 -/
axiom units_digit_power_5 (n : ℕ) : (5^(5^n)) % 10 = 5

/-- Theorem: The units digit of G_2000 is 1 -/
theorem units_digit_G_2000 : G 2000 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_2000_l918_91834


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l918_91850

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_part_overs
  let first_part_runs := game.first_part_run_rate * game.first_part_overs
  let remaining_runs := game.target_runs - first_part_runs
  remaining_runs / remaining_overs

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.total_overs = 50)
  (h2 : game.first_part_overs = 10)
  (h3 : game.first_part_run_rate = 6.2)
  (h4 : game.target_runs = 282) :
  required_run_rate game = 5.5 := by
  sorry

#eval required_run_rate { 
  total_overs := 50, 
  first_part_overs := 10, 
  first_part_run_rate := 6.2, 
  target_runs := 282 
}

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l918_91850


namespace NUMINAMATH_CALUDE_factorization_proof_l918_91851

theorem factorization_proof (x : ℝ) : 72 * x^2 + 108 * x + 36 = 36 * (2 * x + 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l918_91851


namespace NUMINAMATH_CALUDE_incircle_area_of_triangle_PF1F2_l918_91881

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 24 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Define a point on the hyperbola in the first quadrant
def P : ℝ × ℝ := sorry

-- Distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem incircle_area_of_triangle_PF1F2 :
  hyperbola P.1 P.2 ∧
  P.1 > 0 ∧ P.2 > 0 ∧
  distance P F1 / distance P F2 = 4 / 3 →
  ∃ (r : ℝ), r^2 * π = 4 * π ∧
  r * (distance P F1 + distance P F2 + distance F1 F2) = distance P F1 * distance P F2 :=
by sorry

end NUMINAMATH_CALUDE_incircle_area_of_triangle_PF1F2_l918_91881


namespace NUMINAMATH_CALUDE_min_cylinders_is_81_l918_91827

/-- Represents the number of high-altitude camps -/
def num_camps : Nat := 4

/-- Represents the maximum number of oxygen cylinders a person can carry -/
def max_carry : Nat := 3

/-- Represents the number of oxygen cylinders consumed per day -/
def daily_consumption : Nat := 1

/-- Calculates the minimum number of oxygen cylinders needed to place one additional cylinder in the highest camp -/
def min_cylinders_needed : Nat := (max_carry ^ num_camps)

/-- Theorem stating that the minimum number of oxygen cylinders needed is 81 -/
theorem min_cylinders_is_81 : min_cylinders_needed = 81 := by
  sorry

#eval min_cylinders_needed

end NUMINAMATH_CALUDE_min_cylinders_is_81_l918_91827


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_l918_91841

/-- Given an arithmetic sequence {a_n} with a_1 = 20, a_n = 54, and S_n = 999, prove that n = 27 -/
theorem arithmetic_sequence_n (a : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : 
  (∀ k, a (k + 1) - a k = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 20 →
  a n = 54 →
  S_n = 999 →
  n = 27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_l918_91841


namespace NUMINAMATH_CALUDE_no_real_roots_of_quadratic_l918_91865

theorem no_real_roots_of_quadratic (x : ℝ) : ¬∃x, x^2 - 4*x + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_of_quadratic_l918_91865


namespace NUMINAMATH_CALUDE_max_table_sum_l918_91821

def primes : List Nat := [2, 3, 5, 7, 17, 19]

def is_valid_arrangement (top bottom : List Nat) : Prop :=
  top.length = 3 ∧ bottom.length = 3 ∧ 
  (∀ x ∈ top, x ∈ primes) ∧ (∀ x ∈ bottom, x ∈ primes) ∧
  (∀ x ∈ top, x ∉ bottom) ∧ (∀ x ∈ bottom, x ∉ top)

def table_sum (top bottom : List Nat) : Nat :=
  (top.sum * bottom.sum)

theorem max_table_sum :
  ∀ top bottom : List Nat,
  is_valid_arrangement top bottom →
  table_sum top bottom ≤ 682 :=
sorry

end NUMINAMATH_CALUDE_max_table_sum_l918_91821


namespace NUMINAMATH_CALUDE_expected_value_is_91_div_6_l918_91833

/-- The expected value of rolling a fair 6-sided die where the win is n^2 dollars for rolling n -/
def expected_value : ℚ :=
  (1 / 6 : ℚ) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

/-- Theorem stating that the expected value is equal to 91/6 -/
theorem expected_value_is_91_div_6 : expected_value = 91 / 6 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_91_div_6_l918_91833


namespace NUMINAMATH_CALUDE_drug_molecule_diameter_scientific_notation_l918_91837

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem drug_molecule_diameter_scientific_notation :
  toScientificNotation 0.00000008 = ScientificNotation.mk 8 (-8) sorry := by
  sorry

end NUMINAMATH_CALUDE_drug_molecule_diameter_scientific_notation_l918_91837


namespace NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l918_91812

open Real

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := log x / log 10

-- State the theorem
theorem order_of_logarithmic_expressions (a b : ℝ) (h1 : a > b) (h2 : b > 1) :
  sqrt (lg a * lg b) < (lg a + lg b) / 2 ∧ (lg a + lg b) / 2 < lg ((a + b) / 2) :=
by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l918_91812


namespace NUMINAMATH_CALUDE_expand_polynomial_l918_91847

theorem expand_polynomial (x : ℝ) : (5*x^2 + 3*x - 7) * 4*x^3 = 20*x^5 + 12*x^4 - 28*x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l918_91847


namespace NUMINAMATH_CALUDE_distance_sum_equals_3root2_l918_91808

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- Define the line l
def line_l (t x y : ℝ) : Prop := x = -t ∧ y = 1 + t

-- Define point P in Cartesian coordinates
def point_P : ℝ × ℝ := (0, 1)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    line_l t₁ A.1 A.2 ∧ circle_C A.1 A.2 ∧
    line_l t₂ B.1 B.2 ∧ circle_C B.1 B.2 ∧
    t₁ ≠ t₂

-- State the theorem
theorem distance_sum_equals_3root2 (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
  Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_sum_equals_3root2_l918_91808


namespace NUMINAMATH_CALUDE_painted_area_calculation_l918_91829

/-- Given a rectangular exhibition space with specific dimensions and border widths,
    calculate the area of the painted region inside the border. -/
theorem painted_area_calculation (total_width total_length border_width_standard border_width_door : ℕ)
    (h1 : total_width = 100)
    (h2 : total_length = 150)
    (h3 : border_width_standard = 15)
    (h4 : border_width_door = 20) :
    (total_width - 2 * border_width_standard) * (total_length - border_width_standard - border_width_door) = 8050 :=
by sorry

end NUMINAMATH_CALUDE_painted_area_calculation_l918_91829


namespace NUMINAMATH_CALUDE_quotient_derivative_property_l918_91818

/-- Two differentiable functions satisfying the property that the derivative of their quotient
    is equal to the quotient of their derivatives. -/
theorem quotient_derivative_property (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.exp (4 * x)
  let g : ℝ → ℝ := fun x => Real.exp (2 * x)
  (deriv (f / g)) x = (deriv f x) / (deriv g x) := by sorry

end NUMINAMATH_CALUDE_quotient_derivative_property_l918_91818


namespace NUMINAMATH_CALUDE_equation_solution_l918_91898

theorem equation_solution : ∃ x : ℝ, (16 : ℝ) ^ (2 * x - 3) = (1 / 2 : ℝ) ^ (x + 8) ↔ x = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l918_91898


namespace NUMINAMATH_CALUDE_crystal_mass_problem_l918_91852

theorem crystal_mass_problem (a b : ℝ) : 
  (a > 0) → 
  (b > 0) → 
  (0.04 * a / 4 = 0.05 * b / 3) → 
  ((a + 20) / (b + 20) = 1.5) → 
  (a = 100 ∧ b = 60) := by
  sorry

end NUMINAMATH_CALUDE_crystal_mass_problem_l918_91852


namespace NUMINAMATH_CALUDE_unique_modular_solution_l918_91819

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -2023 [ZMOD 9] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l918_91819


namespace NUMINAMATH_CALUDE_integral_x_squared_sqrt_25_minus_x_squared_l918_91862

theorem integral_x_squared_sqrt_25_minus_x_squared : 
  ∫ x in (0)..(5), x^2 * Real.sqrt (25 - x^2) = (625 * Real.pi) / 16 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_sqrt_25_minus_x_squared_l918_91862


namespace NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l918_91824

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The seventh term of an arithmetic sequence is the average of its third and eleventh terms. -/
theorem seventh_term_of_arithmetic_sequence
  (a : ℕ → ℚ) 
  (h_arithmetic : is_arithmetic_sequence a)
  (h_third_term : a 3 = 2 / 11)
  (h_eleventh_term : a 11 = 5 / 6) :
  a 7 = 67 / 132 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l918_91824


namespace NUMINAMATH_CALUDE_systematic_sample_largest_l918_91869

/-- Represents a systematic sample from a range of numbered products -/
structure SystematicSample where
  total_products : Nat
  smallest : Nat
  second_smallest : Nat
  largest : Nat

/-- Theorem stating the properties of the systematic sample in the problem -/
theorem systematic_sample_largest (sample : SystematicSample) : 
  sample.total_products = 300 ∧ 
  sample.smallest = 2 ∧ 
  sample.second_smallest = 17 →
  sample.largest = 287 := by
  sorry

#check systematic_sample_largest

end NUMINAMATH_CALUDE_systematic_sample_largest_l918_91869


namespace NUMINAMATH_CALUDE_hyperbola_perpendicular_product_l918_91887

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptote₁ (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 0
def asymptote₂ (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 0

/-- A point on the hyperbola -/
def P : ℝ × ℝ → Prop := λ p => hyperbola p.1 p.2

/-- Feet of perpendiculars from P to asymptotes -/
def A : (ℝ × ℝ) → (ℝ × ℝ) → Prop := λ p a => 
  asymptote₁ a.1 a.2 ∧ (p.1 - a.1) * (Real.sqrt 3) + (p.2 - a.2) = 0

def B : (ℝ × ℝ) → (ℝ × ℝ) → Prop := λ p b => 
  asymptote₂ b.1 b.2 ∧ (p.1 - b.1) * (Real.sqrt 3) - (p.2 - b.2) = 0

/-- The theorem to be proved -/
theorem hyperbola_perpendicular_product (p a b : ℝ × ℝ) :
  P p → A p a → B p b → 
  (p.1 - a.1) * (p.1 - b.1) + (p.2 - a.2) * (p.2 - b.2) = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_perpendicular_product_l918_91887


namespace NUMINAMATH_CALUDE_siblings_average_age_l918_91816

theorem siblings_average_age (youngest_age : ℕ) (age_differences : List ℕ) :
  youngest_age = 17 →
  age_differences = [4, 5, 7] →
  (youngest_age + (age_differences.map (λ d => youngest_age + d)).sum) / 4 = 21 :=
by sorry

end NUMINAMATH_CALUDE_siblings_average_age_l918_91816


namespace NUMINAMATH_CALUDE_correct_probability_open_l918_91889

/-- Represents a three-digit combination lock -/
structure CombinationLock :=
  (digits : Fin 3 → Fin 10)

/-- The probability of opening the lock by randomly selecting the last digit -/
def probability_open (lock : CombinationLock) : ℚ :=
  1 / 10

theorem correct_probability_open (lock : CombinationLock) :
  probability_open lock = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_probability_open_l918_91889


namespace NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l918_91878

/-- The volume of a cone formed from a 270-degree sector of a circle with radius 20, when divided by π, is equal to 1125√7. -/
theorem cone_volume_divided_by_pi (r l : Real) (h : Real) : 
  r = 15 → l = 20 → h = 5 * Real.sqrt 7 → (1/3 * π * r^2 * h) / π = 1125 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l918_91878


namespace NUMINAMATH_CALUDE_angela_finished_nine_problems_l918_91883

/-- The number of math problems Angela and her friends are working on -/
def total_problems : ℕ := 20

/-- The number of problems Martha has finished -/
def martha_problems : ℕ := 2

/-- The number of problems Jenna has finished -/
def jenna_problems : ℕ := 4 * martha_problems - 2

/-- The number of problems Mark has finished -/
def mark_problems : ℕ := jenna_problems / 2

/-- The number of problems Angela has finished on her own -/
def angela_problems : ℕ := total_problems - (martha_problems + jenna_problems + mark_problems)

theorem angela_finished_nine_problems : angela_problems = 9 := by
  sorry

end NUMINAMATH_CALUDE_angela_finished_nine_problems_l918_91883


namespace NUMINAMATH_CALUDE_degree_of_3ab_l918_91842

-- Define a monomial type
def Monomial := List (String × Nat)

-- Define a function to calculate the degree of a monomial
def degree (m : Monomial) : Nat :=
  m.foldl (fun acc (_, power) => acc + power) 0

-- Define our specific monomial 3ab
def monomial_3ab : Monomial := [("a", 1), ("b", 1)]

-- Theorem statement
theorem degree_of_3ab : degree monomial_3ab = 2 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_3ab_l918_91842


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l918_91859

theorem real_part_of_complex_number (z : ℂ) : z = (Complex.I^3) / (1 + 2 * Complex.I) → Complex.re z = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l918_91859


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l918_91857

/-- The number of different types of wrapping paper -/
def wrapping_paper_types : ℕ := 10

/-- The number of different colors of ribbon -/
def ribbon_colors : ℕ := 4

/-- The number of different types of gift cards -/
def gift_card_types : ℕ := 5

/-- The number of different styles of gift tags -/
def gift_tag_styles : ℕ := 2

/-- The total number of different combinations for gift wrapping -/
def total_combinations : ℕ := wrapping_paper_types * ribbon_colors * gift_card_types * gift_tag_styles

theorem gift_wrapping_combinations : total_combinations = 400 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l918_91857


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l918_91879

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a placeholder structure

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l₁ l₂ : Line3D) : Prop :=
  sorry

/-- Two lines do not intersect if they have no point in common -/
def do_not_intersect (l₁ l₂ : Line3D) : Prop :=
  sorry

theorem skew_lines_sufficient_not_necessary :
  (∀ l₁ l₂ : Line3D, are_skew l₁ l₂ → do_not_intersect l₁ l₂) ∧
  (∃ l₁ l₂ : Line3D, do_not_intersect l₁ l₂ ∧ ¬are_skew l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l918_91879


namespace NUMINAMATH_CALUDE_equal_variance_implies_arithmetic_square_alternating_sequence_is_equal_variance_equal_variance_subsequence_l918_91893

-- Define the equal variance sequence property
def is_equal_variance_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n : ℕ+, a n ^ 2 - a (n + 1) ^ 2 = p

-- Define arithmetic sequence property
def is_arithmetic_sequence (b : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, b (n + 1) - b n = d

-- Statement 1
theorem equal_variance_implies_arithmetic_square (a : ℕ+ → ℝ) :
  is_equal_variance_sequence a → is_arithmetic_sequence (λ n => a n ^ 2) := by sorry

-- Statement 2
theorem alternating_sequence_is_equal_variance :
  is_equal_variance_sequence (λ n => (-1) ^ (n : ℕ)) := by sorry

-- Statement 3
theorem equal_variance_subsequence (a : ℕ+ → ℝ) (k : ℕ+) :
  is_equal_variance_sequence a → is_equal_variance_sequence (λ n => a (k * n)) := by sorry

end NUMINAMATH_CALUDE_equal_variance_implies_arithmetic_square_alternating_sequence_is_equal_variance_equal_variance_subsequence_l918_91893


namespace NUMINAMATH_CALUDE_new_years_numbers_evenness_l918_91823

theorem new_years_numbers_evenness (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2018) :
  ((2019 - k)^12 + 2018) % 2019 = (k^12 + 2018) % 2019 :=
sorry

end NUMINAMATH_CALUDE_new_years_numbers_evenness_l918_91823


namespace NUMINAMATH_CALUDE_volleyball_managers_l918_91872

theorem volleyball_managers (num_teams : ℕ) (people_per_team : ℕ) (num_employees : ℕ) : 
  num_teams = 6 → people_per_team = 5 → num_employees = 7 →
  num_teams * people_per_team - num_employees = 23 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_managers_l918_91872


namespace NUMINAMATH_CALUDE_adam_savings_l918_91874

theorem adam_savings (x : ℝ) : x + 13 = 92 → x = 79 := by
  sorry

end NUMINAMATH_CALUDE_adam_savings_l918_91874


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l918_91807

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 12) 
  (eq2 : x + 4 * y = 18) : 
  17 * x^2 + 24 * x * y + 17 * y^2 = 532 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l918_91807


namespace NUMINAMATH_CALUDE_cosine_function_vertical_shift_l918_91863

/-- Given a cosine function y = a * cos(b * x + c) + d that oscillates between 5 and -3,
    prove that the vertical shift d equals 1. -/
theorem cosine_function_vertical_shift
  (a b c d : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_oscillation : ∀ x, -3 ≤ a * Real.cos (b * x + c) + d ∧ 
                        a * Real.cos (b * x + c) + d ≤ 5) :
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_cosine_function_vertical_shift_l918_91863


namespace NUMINAMATH_CALUDE_negation_of_existence_is_universal_negation_l918_91839

theorem negation_of_existence_is_universal_negation :
  (¬ ∃ (x : ℝ), x^2 = 1) ↔ (∀ (x : ℝ), x^2 ≠ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_universal_negation_l918_91839


namespace NUMINAMATH_CALUDE_f_properties_l918_91838

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def decreasing_on_8_to_inf (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 8 → y > x → f y < f x

def f_plus_8_is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 8) = f (-x + 8)

-- State the theorem
theorem f_properties (h1 : decreasing_on_8_to_inf f) (h2 : f_plus_8_is_even f) :
  f 7 = f 9 ∧ f 7 > f 10 := by sorry

end NUMINAMATH_CALUDE_f_properties_l918_91838


namespace NUMINAMATH_CALUDE_work_completion_time_l918_91880

theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- Ensure x is positive
  (1/x + 1/15 = 1/6) →  -- Combined work rate equals 1/6
  (x = 10) := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l918_91880


namespace NUMINAMATH_CALUDE_gwen_math_problems_l918_91810

theorem gwen_math_problems 
  (science_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_problems : ℕ) 
  (h1 : science_problems = 11)
  (h2 : finished_problems = 24)
  (h3 : remaining_problems = 5)
  (h4 : finished_problems + remaining_problems = science_problems + math_problems) :
  math_problems = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_gwen_math_problems_l918_91810


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l918_91897

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {a_n}, if a_1 + 3a_8 + a_15 = 120, then a_8 = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l918_91897


namespace NUMINAMATH_CALUDE_four_objects_two_groups_l918_91871

theorem four_objects_two_groups : ∃ (n : ℕ), n = 14 ∧ 
  n = (Nat.choose 4 1) + (Nat.choose 4 2) + (Nat.choose 4 3) :=
sorry

end NUMINAMATH_CALUDE_four_objects_two_groups_l918_91871


namespace NUMINAMATH_CALUDE_ball_probabilities_l918_91848

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of white balls in the bag -/
def num_white_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red_balls + num_white_balls

/-- The probability of picking a red ball in a single draw -/
def prob_red : ℚ := num_red_balls / total_balls

/-- The probability of picking a white ball in a single draw -/
def prob_white : ℚ := num_white_balls / total_balls

theorem ball_probabilities :
  -- Statement A
  (Nat.choose 2 1 * Nat.choose 4 2 : ℚ) / Nat.choose 6 3 = 3 / 5 ∧
  -- Statement B
  (6 : ℚ) * prob_red * (1 - prob_red) = 4 / 3 ∧
  -- Statement C
  (4 : ℚ) / 6 * 3 / 5 = 2 / 5 ∧
  -- Statement D
  1 - (1 - prob_red) ^ 3 = 26 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l918_91848


namespace NUMINAMATH_CALUDE_negative_a_range_l918_91845

theorem negative_a_range (a : ℝ) :
  a < 0 →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) →
  a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_range_l918_91845


namespace NUMINAMATH_CALUDE_inscribed_square_area_ratio_l918_91899

/-- The ratio of the area of a square inscribed in an ellipse to the area of a square inscribed in a circle -/
theorem inscribed_square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let ellipse_square_area := (4 * a^2 * b^2) / (a^2 + b^2)
  let circle_square_area := 2 * b^2
  ellipse_square_area / circle_square_area = 2 * a^2 / (a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_ratio_l918_91899


namespace NUMINAMATH_CALUDE_trig_identity_l918_91822

theorem trig_identity (α : Real) (h : Real.tan α = 3) :
  2 * (Real.sin α)^2 + 4 * (Real.sin α) * (Real.cos α) - 9 * (Real.cos α)^2 = 21/10 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l918_91822


namespace NUMINAMATH_CALUDE_solution_set_k_zero_k_range_two_zeros_l918_91868

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - k * x else k * x^2 - x + 1

-- Theorem for the solution set when k = 0
theorem solution_set_k_zero :
  {x : ℝ | f 0 x < 2} = {x : ℝ | -1 < x ∧ x < Real.log 2} := by sorry

-- Theorem for the range of k when f has exactly two zeros
theorem k_range_two_zeros :
  ∀ k : ℝ, (∃! x y : ℝ, x ≠ y ∧ f k x = 0 ∧ f k y = 0) ↔ k > Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_k_zero_k_range_two_zeros_l918_91868


namespace NUMINAMATH_CALUDE_no_integer_sqrt_representation_l918_91892

theorem no_integer_sqrt_representation : ¬ ∃ (A B : ℤ), 99999 + 111111 * Real.sqrt 3 = (A + B * Real.sqrt 3) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_sqrt_representation_l918_91892


namespace NUMINAMATH_CALUDE_a_formula_l918_91817

def a : ℕ → ℕ
  | 0 => 0
  | 1 => 2
  | (n + 2) => 2 * a (n + 1) - a n + 2

theorem a_formula (n : ℕ) : a n = n^2 + n := by
  sorry

end NUMINAMATH_CALUDE_a_formula_l918_91817


namespace NUMINAMATH_CALUDE_selection_methods_equal_l918_91894

def male_students : ℕ := 20
def female_students : ℕ := 30
def total_students : ℕ := male_students + female_students
def select_count : ℕ := 4

def selection_method_1 : ℕ := Nat.choose total_students select_count - 
                               Nat.choose male_students select_count - 
                               Nat.choose female_students select_count

def selection_method_2 : ℕ := Nat.choose male_students 1 * Nat.choose female_students 3 +
                               Nat.choose male_students 2 * Nat.choose female_students 2 +
                               Nat.choose male_students 3 * Nat.choose female_students 1

theorem selection_methods_equal : selection_method_1 = selection_method_2 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_equal_l918_91894


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l918_91853

/-- Given vectors a and b in ℝ², prove that if (a + x*b) is perpendicular to (a - b), then x = -3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (3, 4))
  (h2 : b = (2, 1))
  (h3 : (a + x • b) • (a - b) = 0) :
  x = -3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l918_91853


namespace NUMINAMATH_CALUDE_call_center_fraction_l918_91814

/-- Represents the fraction of calls processed by team B given the conditions of the problem -/
theorem call_center_fraction (team_a team_b : ℕ) (calls_a calls_b : ℝ) : 
  team_a = (5 : ℝ) / 8 * team_b →
  calls_a = (1 : ℝ) / 5 * calls_b →
  (team_b * calls_b) / (team_a * calls_a + team_b * calls_b) = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_call_center_fraction_l918_91814


namespace NUMINAMATH_CALUDE_a_greater_equal_four_l918_91840

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x - a < 0}

-- State the theorem
theorem a_greater_equal_four (a : ℝ) : A ⊆ B a → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_equal_four_l918_91840


namespace NUMINAMATH_CALUDE_tank_water_problem_l918_91864

theorem tank_water_problem (added_saline : ℝ) (salt_concentration_added : ℝ) 
  (salt_concentration_final : ℝ) (initial_water : ℝ) : 
  added_saline = 66.67 →
  salt_concentration_added = 0.25 →
  salt_concentration_final = 0.10 →
  initial_water = 100 →
  salt_concentration_added * added_saline = 
    salt_concentration_final * (initial_water + added_saline) :=
by sorry

end NUMINAMATH_CALUDE_tank_water_problem_l918_91864


namespace NUMINAMATH_CALUDE_fixed_points_specific_case_range_of_a_for_two_fixed_points_l918_91843

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what a fixed point is
def is_fixed_point (a b x : ℝ) : Prop := f a b x = x

-- Part 1: Fixed points when a = 1 and b = 5
theorem fixed_points_specific_case :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_fixed_point 1 5 x₁ ∧ is_fixed_point 1 5 x₂ ∧ x₁ = -4 ∧ x₂ = -1 :=
sorry

-- Part 2: Range of a for two distinct fixed points
theorem range_of_a_for_two_fixed_points :
  (∀ b : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_fixed_point a b x₁ ∧ is_fixed_point a b x₂) →
  (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_specific_case_range_of_a_for_two_fixed_points_l918_91843


namespace NUMINAMATH_CALUDE_inequality_solution_set_l918_91854

theorem inequality_solution_set : 
  {x : ℝ | x^2 - x - 6 < 0} = Set.Ioo (-2) 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l918_91854


namespace NUMINAMATH_CALUDE_inverse_g_90_l918_91866

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_g_90 : g 3 = 90 := by sorry

end NUMINAMATH_CALUDE_inverse_g_90_l918_91866


namespace NUMINAMATH_CALUDE_distance_is_1000_l918_91856

/-- The distance between Liang Liang's home and school in meters. -/
def distance : ℝ := sorry

/-- The time taken (in minutes) when walking at 40 meters per minute. -/
def time_at_40 : ℝ := sorry

/-- Assertion that distance equals speed multiplied by time for 40 m/min speed. -/
axiom distance_eq_40_times_time : distance = 40 * time_at_40

/-- Assertion that distance equals speed multiplied by time for 50 m/min speed. -/
axiom distance_eq_50_times_time_minus_5 : distance = 50 * (time_at_40 - 5)

theorem distance_is_1000 : distance = 1000 := by sorry

end NUMINAMATH_CALUDE_distance_is_1000_l918_91856


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l918_91844

theorem min_reciprocal_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 3) :
  (1 / a + 1 / b + 1 / c) ≥ 3 ∧ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 3 ∧ 1 / x + 1 / y + 1 / z = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l918_91844


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l918_91895

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5678 * 10 + N) % 6 = 0 → N ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l918_91895


namespace NUMINAMATH_CALUDE_escalator_theorem_l918_91800

def escalator_problem (stationary_time walking_time : ℝ) : Prop :=
  let s := 1 / stationary_time -- Clea's walking speed
  let d := 1 -- normalized distance of the escalator
  let v := d / walking_time - s -- speed of the escalator
  (d / v) = 50

theorem escalator_theorem : 
  escalator_problem 75 30 := by sorry

end NUMINAMATH_CALUDE_escalator_theorem_l918_91800


namespace NUMINAMATH_CALUDE_toys_after_game_purchase_l918_91858

theorem toys_after_game_purchase (initial_amount : ℕ) (game_cost : ℕ) (toy_cost : ℕ) : 
  initial_amount = 63 → game_cost = 48 → toy_cost = 3 → 
  (initial_amount - game_cost) / toy_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_toys_after_game_purchase_l918_91858


namespace NUMINAMATH_CALUDE_smallest_solution_of_quartic_l918_91890

theorem smallest_solution_of_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 →
  x ≥ -Real.sqrt 26 ∧ 
  ∃ y : ℝ, y^4 - 50*y^2 + 576 = 0 ∧ y = -Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_quartic_l918_91890


namespace NUMINAMATH_CALUDE_paint_usage_for_large_canvas_l918_91813

/-- Given an artist who uses L ounces of paint for every large canvas and 2 ounces for every small canvas,
    prove that L = 3 when the artist has completed 3 large paintings and 4 small paintings,
    using a total of 17 ounces of paint. -/
theorem paint_usage_for_large_canvas (L : ℝ) : 
  (3 * L + 4 * 2 = 17) → L = 3 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_for_large_canvas_l918_91813


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l918_91870

theorem complex_fraction_simplification :
  ((1 - Complex.I) * (1 + 2 * Complex.I)) / (1 + Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l918_91870


namespace NUMINAMATH_CALUDE_money_left_after_trip_l918_91811

def initial_savings : ℕ := 6000
def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000

theorem money_left_after_trip : 
  initial_savings - (flight_cost + hotel_cost + food_cost) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_trip_l918_91811


namespace NUMINAMATH_CALUDE_rectangle_area_l918_91836

/-- A rectangle with length thrice its breadth and perimeter 56 meters has an area of 147 square meters. -/
theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 56) : l * b = 147 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l918_91836


namespace NUMINAMATH_CALUDE_gcd_4830_3289_l918_91885

theorem gcd_4830_3289 : Nat.gcd 4830 3289 = 23 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4830_3289_l918_91885


namespace NUMINAMATH_CALUDE_candle_flower_groupings_l918_91825

theorem candle_flower_groupings :
  (Nat.choose 4 2) * (Nat.choose 9 8) = 54 := by
  sorry

end NUMINAMATH_CALUDE_candle_flower_groupings_l918_91825


namespace NUMINAMATH_CALUDE_triangle_angle_c_l918_91877

theorem triangle_angle_c (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = Real.pi) (h5 : B = Real.pi / 4)
  (h6 : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a = c * Real.sqrt 2 ∧
    a / (Real.sin C) = b / (Real.sin A) ∧ b / (Real.sin B) = c / (Real.sin A)) :
  C = 7 * Real.pi / 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l918_91877


namespace NUMINAMATH_CALUDE_student_admission_price_l918_91888

theorem student_admission_price
  (total_tickets : ℕ)
  (adult_price : ℕ)
  (total_amount : ℕ)
  (student_attendees : ℕ)
  (h1 : total_tickets = 1500)
  (h2 : adult_price = 12)
  (h3 : total_amount = 16200)
  (h4 : student_attendees = 300) :
  (total_amount - (total_tickets - student_attendees) * adult_price) / student_attendees = 6 :=
by sorry

end NUMINAMATH_CALUDE_student_admission_price_l918_91888


namespace NUMINAMATH_CALUDE_point_product_theorem_l918_91867

theorem point_product_theorem : 
  ∀ y₁ y₂ : ℝ, 
    ((-2 - 4)^2 + (y₁ - (-1))^2 = 8^2) → 
    ((-2 - 4)^2 + (y₂ - (-1))^2 = 8^2) → 
    y₁ ≠ y₂ →
    y₁ * y₂ = -27 := by
  sorry

end NUMINAMATH_CALUDE_point_product_theorem_l918_91867


namespace NUMINAMATH_CALUDE_total_students_is_3700_l918_91805

/-- Represents a high school with three grades -/
structure HighSchool where
  total_students : ℕ
  senior_students : ℕ
  sample_size : ℕ
  freshman_sample : ℕ
  sophomore_sample : ℕ

/-- The conditions of the problem -/
def problem_conditions (school : HighSchool) : Prop :=
  school.senior_students = 1000 ∧
  school.sample_size = 185 ∧
  school.freshman_sample = 75 ∧
  school.sophomore_sample = 60 ∧
  (school.senior_students : ℚ) / school.total_students = 
    (school.sample_size - school.freshman_sample - school.sophomore_sample : ℚ) / school.sample_size

/-- The theorem stating that under the given conditions, the total number of students is 3700 -/
theorem total_students_is_3700 (school : HighSchool) 
  (h : problem_conditions school) : school.total_students = 3700 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_3700_l918_91805


namespace NUMINAMATH_CALUDE_power_equation_solution_l918_91860

theorem power_equation_solution : 2^5 - 7 = 3^3 + (-2) := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l918_91860


namespace NUMINAMATH_CALUDE_simon_kabob_cost_l918_91832

/-- Represents the cost of making kabob sticks -/
def cost_of_kabobs (cubes_per_stick : ℕ) (cubes_per_slab : ℕ) (cost_per_slab : ℕ) (num_sticks : ℕ) : ℕ :=
  let slabs_needed := (num_sticks * cubes_per_stick + cubes_per_slab - 1) / cubes_per_slab
  slabs_needed * cost_per_slab

/-- Proves that the cost for Simon to make 40 kabob sticks is $50 -/
theorem simon_kabob_cost : cost_of_kabobs 4 80 25 40 = 50 := by
  sorry

end NUMINAMATH_CALUDE_simon_kabob_cost_l918_91832


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l918_91884

theorem modular_congruence_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 103 ∧ (99 * n) % 103 = 73 % 103 ∧ n = 68 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l918_91884


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l918_91849

theorem subtraction_of_fractions :
  (3 : ℚ) / 2 - (3 : ℚ) / 5 = (9 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l918_91849


namespace NUMINAMATH_CALUDE_perpendicular_parallel_relationships_l918_91806

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_relationships 
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β) :
  (parallel α β → line_perpendicular l m) ∧
  (line_parallel l m → plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_relationships_l918_91806


namespace NUMINAMATH_CALUDE_square_mod_five_l918_91826

theorem square_mod_five (n : ℤ) : 
  ¬(5 ∣ n) → ∃ k : ℤ, n^2 = 5*k + 1 ∨ n^2 = 5*k - 1 :=
sorry

end NUMINAMATH_CALUDE_square_mod_five_l918_91826


namespace NUMINAMATH_CALUDE_line_in_quadrants_implies_positive_slope_l918_91831

/-- A line passing through the first and third quadrants -/
structure LineInQuadrants where
  k : ℝ
  k_nonzero : k ≠ 0
  passes_first_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = k * x
  passes_third_quadrant : ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ y = k * x

/-- If a line y = kx passes through the first and third quadrants, then k > 0 -/
theorem line_in_quadrants_implies_positive_slope (l : LineInQuadrants) : l.k > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_in_quadrants_implies_positive_slope_l918_91831


namespace NUMINAMATH_CALUDE_dollar_op_six_three_l918_91830

def dollar_op (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem dollar_op_six_three : dollar_op 6 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_dollar_op_six_three_l918_91830


namespace NUMINAMATH_CALUDE_fraction_sum_l918_91820

theorem fraction_sum : (1 : ℚ) / 6 + (1 : ℚ) / 3 + (5 : ℚ) / 9 = (19 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l918_91820


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l918_91809

theorem imaginary_part_of_z (m : ℝ) :
  let z := (2 + m * Complex.I) / (1 + Complex.I)
  (z.re = 0) → z.im = -2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l918_91809


namespace NUMINAMATH_CALUDE_well_depth_is_30_l918_91882

/-- The depth of a well that a man climbs out of in 27 days -/
def well_depth (daily_climb : ℕ) (daily_slip : ℕ) (total_days : ℕ) (final_climb : ℕ) : ℕ :=
  (total_days - 1) * (daily_climb - daily_slip) + final_climb

/-- Theorem stating the depth of the well is 30 meters -/
theorem well_depth_is_30 :
  well_depth 4 3 27 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_well_depth_is_30_l918_91882


namespace NUMINAMATH_CALUDE_paperclip_capacity_l918_91891

theorem paperclip_capacity (box_volume : ℝ) (box_capacity : ℕ) (cube_volume : ℝ) : 
  box_volume = 24 → 
  box_capacity = 75 → 
  cube_volume = 64 → 
  (cube_volume / box_volume * box_capacity : ℝ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_paperclip_capacity_l918_91891


namespace NUMINAMATH_CALUDE_root_implies_sum_l918_91835

/-- Given that 2 + i is a root of the polynomial x^4 + px^2 + qx + 1 = 0,
    where p and q are real numbers, prove that p + q = 4 -/
theorem root_implies_sum (p q : ℝ) 
  (h : (2 + Complex.I) ^ 4 + p * (2 + Complex.I) ^ 2 + q * (2 + Complex.I) + 1 = 0) : 
  p + q = 4 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_sum_l918_91835


namespace NUMINAMATH_CALUDE_units_digit_of_power_sum_divided_l918_91801

/-- The units digit of (4^503 + 6^503) / 10 is 1 -/
theorem units_digit_of_power_sum_divided : ∃ n : ℕ, (4^503 + 6^503) / 10 = 10 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_sum_divided_l918_91801


namespace NUMINAMATH_CALUDE_tournament_results_count_l918_91873

-- Define the teams
inductive Team : Type
| E : Team
| F : Team
| G : Team
| H : Team

-- Define a match result
inductive MatchResult : Type
| Win : Team → MatchResult
| Loss : Team → MatchResult

-- Define a tournament result
structure TournamentResult : Type :=
(saturday1 : MatchResult)  -- E vs F
(saturday2 : MatchResult)  -- G vs H
(sunday1 : MatchResult)    -- 1st vs 2nd
(sunday2 : MatchResult)    -- 3rd vs 4th

def count_tournament_results : ℕ :=
  -- The actual count will be implemented in the proof
  sorry

theorem tournament_results_count :
  count_tournament_results = 16 := by
  sorry

end NUMINAMATH_CALUDE_tournament_results_count_l918_91873


namespace NUMINAMATH_CALUDE_solution_set_eq_l918_91828

-- Define a decreasing function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_decreasing : ∀ x y, x < y → f x > f y
axiom f_0 : f 0 = -2
axiom f_neg_3 : f (-3) = 2

-- Define the solution set
def solution_set : Set ℝ := {x | |f (x - 2)| > 2}

-- State the theorem
theorem solution_set_eq : solution_set = Set.Iic (-1) ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_eq_l918_91828


namespace NUMINAMATH_CALUDE_book_prices_l918_91886

theorem book_prices (book1_discounted : ℚ) (book2_discounted : ℚ)
  (h1 : book1_discounted = 8)
  (h2 : book2_discounted = 9)
  (h3 : book1_discounted = (1 / 8 : ℚ) * book1_discounted / (1 / 8 : ℚ))
  (h4 : book2_discounted = (1 / 9 : ℚ) * book2_discounted / (1 / 9 : ℚ)) :
  book1_discounted / (1 / 8 : ℚ) + book2_discounted / (1 / 9 : ℚ) = 145 := by
sorry

end NUMINAMATH_CALUDE_book_prices_l918_91886


namespace NUMINAMATH_CALUDE_jenna_round_trip_pay_l918_91803

/-- Calculates the pay for a round trip given the pay rate per mile and one-way distance -/
def round_trip_pay (rate : ℚ) (one_way_distance : ℚ) : ℚ :=
  2 * rate * one_way_distance

/-- Proves that the round trip pay for a rate of $0.40 per mile and 400 miles one-way is $320 -/
theorem jenna_round_trip_pay :
  round_trip_pay (40 / 100) 400 = 320 := by
  sorry

#eval round_trip_pay (40 / 100) 400

end NUMINAMATH_CALUDE_jenna_round_trip_pay_l918_91803


namespace NUMINAMATH_CALUDE_pipe_c_fill_time_l918_91876

/-- The time (in minutes) it takes for pipe a to fill the tank -/
def time_a : ℝ := 20

/-- The time (in minutes) it takes for pipe b to fill the tank -/
def time_b : ℝ := 20

/-- The time (in minutes) it takes for pipe c to fill the tank -/
def time_c : ℝ := 30

/-- The proportion of solution r in the tank after 3 minutes -/
def proportion_r : ℝ := 0.25

/-- The time (in minutes) after which we measure the proportion of solution r -/
def measure_time : ℝ := 3

theorem pipe_c_fill_time :
  (time_c = 30) ∧
  (measure_time * (1 / time_a + 1 / time_b + 1 / time_c) * (1 / time_c) /
   (measure_time * (1 / time_a + 1 / time_b + 1 / time_c)) = proportion_r) :=
sorry

end NUMINAMATH_CALUDE_pipe_c_fill_time_l918_91876
