import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l787_78706

theorem simplify_expression (y : ℝ) : 3*y + 4*y^2 - 2 - (8 - 3*y - 4*y^2) = 8*y^2 + 6*y - 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l787_78706


namespace NUMINAMATH_CALUDE_janets_height_l787_78733

/-- Given the heights of various people, prove Janet's height --/
theorem janets_height :
  ∀ (ruby pablo charlene janet : ℝ),
  ruby = pablo - 2 →
  pablo = charlene + 70 →
  charlene = 2 * janet →
  ruby = 192 →
  janet = 62 := by
sorry

end NUMINAMATH_CALUDE_janets_height_l787_78733


namespace NUMINAMATH_CALUDE_one_meeting_l787_78701

/-- Represents the movement and meeting of a jogger and an aid vehicle --/
structure JoggerVehicleSystem where
  jogger_speed : ℝ
  vehicle_speed : ℝ
  station_distance : ℝ
  vehicle_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between the jogger and the vehicle --/
def number_of_meetings (sys : JoggerVehicleSystem) : ℕ :=
  sorry

/-- The specific scenario described in the problem --/
def problem_scenario : JoggerVehicleSystem :=
  { jogger_speed := 6
  , vehicle_speed := 12
  , station_distance := 300
  , vehicle_stop_time := 20
  , initial_distance := 300 }

/-- Theorem stating that in the given scenario, there is exactly one meeting --/
theorem one_meeting :
  number_of_meetings problem_scenario = 1 :=
sorry

end NUMINAMATH_CALUDE_one_meeting_l787_78701


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l787_78766

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x^2 + x - 2| :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l787_78766


namespace NUMINAMATH_CALUDE_nested_bracket_equals_two_l787_78754

/-- Defines the operation [a,b,c] as (a+b)/c for c ≠ 0 -/
def bracket (a b c : ℚ) : ℚ := (a + b) / c

/-- The main theorem to prove -/
theorem nested_bracket_equals_two :
  bracket (bracket 120 60 180) (bracket 4 2 6) (bracket 20 10 30) = 2 := by
  sorry


end NUMINAMATH_CALUDE_nested_bracket_equals_two_l787_78754


namespace NUMINAMATH_CALUDE_e₁_e₂_divisibility_l787_78795

def e₁ (a : ℕ) : ℕ := a^2 + 3^a + a * 3^((a + 1) / 2)
def e₂ (a : ℕ) : ℕ := a^2 + 3^a - a * 3^((a + 1) / 2)

theorem e₁_e₂_divisibility (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 25) :
  (e₁ a * e₂ a) % 3 = 0 ↔ a % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_e₁_e₂_divisibility_l787_78795


namespace NUMINAMATH_CALUDE_ninth_term_is_seven_l787_78715

/-- A sequence where each term is 1/2 more than the previous term -/
def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = a n + 1/2

/-- The 9th term of the arithmetic sequence is 7 -/
theorem ninth_term_is_seven (a : ℕ → ℚ) (h : arithmeticSequence a) : a 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_seven_l787_78715


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l787_78709

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ :=
  {x | x ≤ -2 ∨ x ≥ 6}

-- Define the quadratic inequality
def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c ≤ 0

-- Theorem statement
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, x ∈ solution_set a b c ↔ quadratic_inequality a b c x) :
  a < 0 ∧
  (∀ x, -1/6 < x ∧ x < 1/2 ↔ c * x^2 - b * x + a < 0) ∧
  a + b + c > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l787_78709


namespace NUMINAMATH_CALUDE_basket_count_l787_78708

theorem basket_count (apples_per_basket : ℕ) (total_apples : ℕ) (h1 : apples_per_basket = 17) (h2 : total_apples = 629) :
  total_apples / apples_per_basket = 37 := by
sorry

end NUMINAMATH_CALUDE_basket_count_l787_78708


namespace NUMINAMATH_CALUDE_henry_twice_jills_age_l787_78775

/-- Given that Henry and Jill's present ages sum to 40, with Henry being 23 and Jill being 17,
    this theorem proves that 11 years ago, Henry was twice the age of Jill. -/
theorem henry_twice_jills_age (henry_age : ℕ) (jill_age : ℕ) :
  henry_age + jill_age = 40 →
  henry_age = 23 →
  jill_age = 17 →
  ∃ (years_ago : ℕ), henry_age - years_ago = 2 * (jill_age - years_ago) ∧ years_ago = 11 := by
  sorry

end NUMINAMATH_CALUDE_henry_twice_jills_age_l787_78775


namespace NUMINAMATH_CALUDE_reflect_center_l787_78765

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflect_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center : ℝ × ℝ := reflect_about_y_eq_neg_x original_center
  reflected_center = (3, -8) := by sorry

end NUMINAMATH_CALUDE_reflect_center_l787_78765


namespace NUMINAMATH_CALUDE_sammy_has_eight_caps_l787_78784

/-- The number of bottle caps Billie has -/
def billies_caps : ℕ := 2

/-- The number of bottle caps Janine has -/
def janines_caps : ℕ := 3 * billies_caps

/-- The number of bottle caps Sammy has -/
def sammys_caps : ℕ := janines_caps + 2

/-- Theorem stating that Sammy has 8 bottle caps -/
theorem sammy_has_eight_caps : sammys_caps = 8 := by
  sorry

end NUMINAMATH_CALUDE_sammy_has_eight_caps_l787_78784


namespace NUMINAMATH_CALUDE_trigonometric_identities_l787_78705

theorem trigonometric_identities :
  (∃ (tan25 tan35 : ℝ),
    tan25 = Real.tan (25 * π / 180) ∧
    tan35 = Real.tan (35 * π / 180) ∧
    tan25 + tan35 + Real.sqrt 3 * tan25 * tan35 = Real.sqrt 3) ∧
  (Real.sin (10 * π / 180))⁻¹ - (Real.sqrt 3) * (Real.cos (10 * π / 180))⁻¹ = 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l787_78705


namespace NUMINAMATH_CALUDE_quadratic_root_square_implies_s_l787_78732

theorem quadratic_root_square_implies_s (r s : ℝ) :
  (∃ x : ℂ, 3 * x^2 + r * x + s = 0 ∧ x^2 = 4 - 3*I) →
  s = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_square_implies_s_l787_78732


namespace NUMINAMATH_CALUDE_fraction_sum_product_equality_l787_78738

theorem fraction_sum_product_equality (a b c : ℝ) 
  (h1 : 1 + b * c ≠ 0) (h2 : 1 + c * a ≠ 0) (h3 : 1 + a * b ≠ 0) : 
  (b - c) / (1 + b * c) + (c - a) / (1 + c * a) + (a - b) / (1 + a * b) = 
  (b - c) / (1 + b * c) * (c - a) / (1 + c * a) * (a - b) / (1 + a * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_product_equality_l787_78738


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_dne_l787_78735

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan x * Real.sin (7 / x) else 0

theorem derivative_f_at_zero_dne :
  ¬ DifferentiableAt ℝ f 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_dne_l787_78735


namespace NUMINAMATH_CALUDE_trigonometric_identity_l787_78719

theorem trigonometric_identity (α β γ : Real) :
  Real.sin α + Real.sin β + Real.sin γ - 
  Real.sin (α + β) * Real.cos γ - Real.cos (α + β) * Real.sin γ = 
  4 * Real.sin ((α + β) / 2) * Real.sin ((β + γ) / 2) * Real.sin ((γ + α) / 2) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l787_78719


namespace NUMINAMATH_CALUDE_weight_difference_l787_78710

theorem weight_difference (steve jim stan : ℕ) : 
  stan = steve + 5 →
  jim = 110 →
  steve + stan + jim = 319 →
  jim - steve = 8 := by
sorry

end NUMINAMATH_CALUDE_weight_difference_l787_78710


namespace NUMINAMATH_CALUDE_six_digit_same_digits_prime_divisor_sum_l787_78774

def is_six_digit_same_digits (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ ∃ d : ℕ, n = d * 111111

def sum_distinct_prime_divisors (n : ℕ) : ℕ := sorry

theorem six_digit_same_digits_prime_divisor_sum 
  (n : ℕ) (h : is_six_digit_same_digits n) : 
  sum_distinct_prime_divisors n ≠ 70 ∧ sum_distinct_prime_divisors n ≠ 80 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_same_digits_prime_divisor_sum_l787_78774


namespace NUMINAMATH_CALUDE_polynomial_value_l787_78717

theorem polynomial_value (x : ℝ) (h : x^2 + 3*x = 1) : 3*x^2 + 9*x - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l787_78717


namespace NUMINAMATH_CALUDE_face_ratio_is_four_thirds_l787_78796

/-- A polyhedron with triangular and square faces -/
structure Polyhedron where
  triangular_faces : ℕ
  square_faces : ℕ
  no_shared_square_edges : Bool
  no_shared_triangle_edges : Bool

/-- The ratio of triangular faces to square faces in a polyhedron -/
def face_ratio (p : Polyhedron) : ℚ :=
  p.triangular_faces / p.square_faces

/-- Theorem: The ratio of triangular faces to square faces is 4:3 -/
theorem face_ratio_is_four_thirds (p : Polyhedron) 
  (h1 : p.no_shared_square_edges = true) 
  (h2 : p.no_shared_triangle_edges = true) : 
  face_ratio p = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_face_ratio_is_four_thirds_l787_78796


namespace NUMINAMATH_CALUDE_odd_prime_sum_divisors_count_l787_78767

/-- Sum of positive integer divisors of n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Predicate for odd prime numbers -/
def is_odd_prime (n : ℕ) : Prop := sorry

/-- Count of numbers with odd prime sum of divisors -/
def count_odd_prime_sum_divisors : ℕ := sorry

theorem odd_prime_sum_divisors_count :
  count_odd_prime_sum_divisors = 5 := by sorry

end NUMINAMATH_CALUDE_odd_prime_sum_divisors_count_l787_78767


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l787_78797

theorem quadratic_solution_product (p q : ℝ) : 
  (3 * p^2 - 2 * p - 8 = 0) → 
  (3 * q^2 - 2 * q - 8 = 0) → 
  p ≠ q →
  (p - 1) * (q - 1) = -7/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l787_78797


namespace NUMINAMATH_CALUDE_total_rainfall_2011_2012_l787_78750

/-- Represents the average monthly rainfall in millimeters for a given year. -/
def AverageMonthlyRainfall : ℕ → ℝ
  | 2010 => 50.0
  | 2011 => AverageMonthlyRainfall 2010 + 3
  | 2012 => AverageMonthlyRainfall 2011 + 4
  | _ => 0  -- Default case for other years

/-- Calculates the total yearly rainfall given the average monthly rainfall. -/
def YearlyRainfall (year : ℕ) : ℝ :=
  AverageMonthlyRainfall year * 12

/-- Theorem stating the total rainfall in Clouddale for 2011 and 2012. -/
theorem total_rainfall_2011_2012 :
  YearlyRainfall 2011 + YearlyRainfall 2012 = 1320.0 := by
  sorry

#eval YearlyRainfall 2011 + YearlyRainfall 2012

end NUMINAMATH_CALUDE_total_rainfall_2011_2012_l787_78750


namespace NUMINAMATH_CALUDE_basketball_team_cutoff_l787_78700

/-- The number of students who didn't make the cut for the basketball team -/
theorem basketball_team_cutoff (girls boys callback : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : callback = 26) :
  girls + boys - callback = 17 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_cutoff_l787_78700


namespace NUMINAMATH_CALUDE_john_final_height_l787_78713

/-- Calculates the final height in feet given initial height, growth rate, and duration -/
def final_height_in_feet (initial_height : ℕ) (growth_rate : ℕ) (duration : ℕ) : ℚ :=
  (initial_height + growth_rate * duration) / 12

/-- Theorem stating that given the specific conditions, the final height is 6 feet -/
theorem john_final_height :
  final_height_in_feet 66 2 3 = 6 := by sorry

end NUMINAMATH_CALUDE_john_final_height_l787_78713


namespace NUMINAMATH_CALUDE_petunias_per_flat_is_8_l787_78707

/-- Represents the number of petunias in each flat -/
def petunias_per_flat : ℕ := sorry

/-- The total number of flats of petunias -/
def petunia_flats : ℕ := 4

/-- The total number of flats of roses -/
def rose_flats : ℕ := 3

/-- The number of roses in each flat -/
def roses_per_flat : ℕ := 6

/-- The number of Venus flytraps -/
def venus_flytraps : ℕ := 2

/-- The amount of fertilizer needed for each petunia (in ounces) -/
def fertilizer_per_petunia : ℕ := 8

/-- The amount of fertilizer needed for each rose (in ounces) -/
def fertilizer_per_rose : ℕ := 3

/-- The amount of fertilizer needed for each Venus flytrap (in ounces) -/
def fertilizer_per_flytrap : ℕ := 2

/-- The total amount of fertilizer needed (in ounces) -/
def total_fertilizer : ℕ := 314

theorem petunias_per_flat_is_8 :
  petunias_per_flat = 8 :=
by sorry

end NUMINAMATH_CALUDE_petunias_per_flat_is_8_l787_78707


namespace NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l787_78712

theorem lucky_lacy_correct_percentage (x : ℕ) : 
  let total_problems : ℕ := 4 * x
  let missed_problems : ℕ := 2 * x
  let correct_problems : ℕ := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l787_78712


namespace NUMINAMATH_CALUDE_inequality_solution_set_l787_78776

theorem inequality_solution_set (x : ℝ) : -x^2 + 4*x - 3 ≥ 0 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l787_78776


namespace NUMINAMATH_CALUDE_expected_value_coin_flip_l787_78773

/-- The expected value of a coin flip game -/
theorem expected_value_coin_flip :
  let p_heads : ℚ := 2/5
  let p_tails : ℚ := 3/5
  let win_amount : ℚ := 5
  let loss_amount : ℚ := 4
  let expected_value := p_heads * win_amount - p_tails * loss_amount
  expected_value = -2/5
:= by sorry

end NUMINAMATH_CALUDE_expected_value_coin_flip_l787_78773


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l787_78731

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = q * a n ∧ a n > 0

/-- The theorem statement -/
theorem geometric_sequence_min_value
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_eq : a 7 = a 6 + 2 * a 5)
  (h_sqrt : ∃ m n : ℕ, Real.sqrt (a m * a n) = 2 * a 1) :
  (∃ m n : ℕ, (1 : ℝ) / m + 9 / n = 4) ∧
  (∀ m n : ℕ, (1 : ℝ) / m + 9 / n ≥ 4) := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_min_value_l787_78731


namespace NUMINAMATH_CALUDE_max_value_theorem_l787_78759

/-- The sum of the first m positive even numbers -/
def sumEvenNumbers (m : ℕ) : ℕ := m * (m + 1)

/-- The sum of the first n positive odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- The constraint that the sum of m even numbers and n odd numbers is 1987 -/
def constraint (m n : ℕ) : Prop := sumEvenNumbers m + sumOddNumbers n = 1987

/-- The objective function to be maximized -/
def objective (m n : ℕ) : ℕ := 3 * m + 4 * n

theorem max_value_theorem :
  ∃ m n : ℕ, constraint m n ∧ 
    ∀ m' n' : ℕ, constraint m' n' → objective m' n' ≤ objective m n ∧
    objective m n = 219 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l787_78759


namespace NUMINAMATH_CALUDE_system_solution_conditions_l787_78758

theorem system_solution_conditions (m : ℝ) :
  let x := (1 + 2*m) / 3
  let y := (1 - m) / 3
  (x + 2*y = 1 ∧ x - y = m) ∧ (x > 1 ∧ y ≥ -1) ↔ 1 < m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_conditions_l787_78758


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l787_78727

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_bounds : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number we want to express in scientific notation -/
def target_number : ℝ := 318000000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 3.18
    exponent := 8
    coeff_bounds := by sorry }

/-- Theorem stating that the proposed notation correctly represents the target number -/
theorem scientific_notation_correct :
  target_number = proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l787_78727


namespace NUMINAMATH_CALUDE_fountain_distance_is_30_l787_78742

/-- The distance from Mrs. Hilt's desk to the water fountain -/
def fountain_distance (total_distance : ℕ) (num_trips : ℕ) : ℚ :=
  (total_distance : ℚ) / (2 * num_trips)

/-- Theorem stating that the fountain distance is 30 feet -/
theorem fountain_distance_is_30 :
  fountain_distance 120 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fountain_distance_is_30_l787_78742


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l787_78734

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l787_78734


namespace NUMINAMATH_CALUDE_even_number_divisibility_property_l787_78786

theorem even_number_divisibility_property (n : ℕ) :
  n % 2 = 0 →
  (∀ p : ℕ, Prime p → p ∣ n → (p - 1) ∣ (n - 1)) →
  ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_even_number_divisibility_property_l787_78786


namespace NUMINAMATH_CALUDE_transform_to_zero_y_l787_78702

-- Define the set M
def M : Set (ℤ × ℤ) := Set.univ

-- Define transformation S
def S (p : ℤ × ℤ) : ℤ × ℤ := (p.1 + p.2, p.2)

-- Define transformation T
def T (p : ℤ × ℤ) : ℤ × ℤ := (-p.2, p.1)

-- Define the type of transformations
inductive Transform
| S : Transform
| T : Transform

-- Define the application of a sequence of transformations
def applyTransforms : List Transform → ℤ × ℤ → ℤ × ℤ
| [], p => p
| (Transform.S :: ts), p => applyTransforms ts (S p)
| (Transform.T :: ts), p => applyTransforms ts (T p)

-- The main theorem
theorem transform_to_zero_y (p : ℤ × ℤ) : 
  ∃ (ts : List Transform) (g : ℤ), applyTransforms ts p = (g, 0) := by
  sorry


end NUMINAMATH_CALUDE_transform_to_zero_y_l787_78702


namespace NUMINAMATH_CALUDE_cube_edge_length_l787_78720

/-- Given a cube with volume V, surface area S, and edge length a, 
    where V = S + 1, prove that a satisfies a³ - 6a² - 1 = 0 
    and the solution is closest to 6 -/
theorem cube_edge_length (V S a : ℝ) (hV : V = a^3) (hS : S = 6*a^2) (hVS : V = S + 1) :
  a^3 - 6*a^2 - 1 = 0 ∧ ∃ ε > 0, ∀ x : ℝ, x ≠ a → |x - 6| > |a - 6| - ε :=
sorry

end NUMINAMATH_CALUDE_cube_edge_length_l787_78720


namespace NUMINAMATH_CALUDE_poodle_bark_count_l787_78779

/-- The number of times the terrier's owner says "hush" -/
def hush_count : ℕ := 6

/-- The ratio of poodle barks to terrier barks -/
def poodle_terrier_ratio : ℕ := 2

/-- The number of barks in a poodle bark set -/
def poodle_bark_set : ℕ := 5

/-- The number of times the terrier barks -/
def terrier_barks : ℕ := hush_count * 2

/-- The number of times the poodle barks -/
def poodle_barks : ℕ := terrier_barks * poodle_terrier_ratio

theorem poodle_bark_count : poodle_barks = 24 := by
  sorry

end NUMINAMATH_CALUDE_poodle_bark_count_l787_78779


namespace NUMINAMATH_CALUDE_max_sum_constrained_squares_l787_78744

theorem max_sum_constrained_squares (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m^2 + n^2 = 100) :
  m + n ≤ 10 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_sum_constrained_squares_l787_78744


namespace NUMINAMATH_CALUDE_difference_1500th_1504th_term_l787_78746

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem difference_1500th_1504th_term : 
  let a₁ := 3
  let d := 6
  |arithmetic_sequence a₁ d 1504 - arithmetic_sequence a₁ d 1500| = 24 := by
  sorry

end NUMINAMATH_CALUDE_difference_1500th_1504th_term_l787_78746


namespace NUMINAMATH_CALUDE_initial_dozens_of_doughnuts_l787_78756

theorem initial_dozens_of_doughnuts (eaten : ℕ) (left : ℕ) (dozen : ℕ) : 
  eaten = 8 → left = 16 → dozen = 12 → (eaten + left) / dozen = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_dozens_of_doughnuts_l787_78756


namespace NUMINAMATH_CALUDE_binary_to_base4_example_l787_78783

/-- Converts a binary (base 2) number to its base 4 representation -/
def binary_to_base4 (binary : ℕ) : ℕ := sorry

/-- The binary number 1011010010₂ -/
def binary_num : ℕ := 722  -- 1011010010₂ in decimal

/-- Theorem stating that the base 4 representation of 1011010010₂ is 3122₄ -/
theorem binary_to_base4_example : binary_to_base4 binary_num = 3122 := by sorry

end NUMINAMATH_CALUDE_binary_to_base4_example_l787_78783


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l787_78792

theorem square_perimeter_ratio (a₁ a₂ p₁ p₂ : ℝ) (h_positive : a₁ > 0 ∧ a₂ > 0) 
  (h_area_ratio : a₁ / a₂ = 49 / 64) (h_perimeter₁ : p₁ = 4 * Real.sqrt a₁) 
  (h_perimeter₂ : p₂ = 4 * Real.sqrt a₂) : p₁ / p₂ = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l787_78792


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l787_78782

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 6)
  (h2 : new_price = 7.5) : 
  (1 - initial_price / new_price) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l787_78782


namespace NUMINAMATH_CALUDE_seventh_term_ratio_l787_78723

/-- Two arithmetic sequences and their properties -/
structure ArithmeticSequencePair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum of first n terms of sequence a
  T : ℕ → ℚ  -- Sum of first n terms of sequence b
  h : ∀ n, S n / T n = (3 * n - 2) / (2 * n + 1)  -- Given condition

/-- Theorem stating the relation between the 7th terms of the sequences -/
theorem seventh_term_ratio (seq : ArithmeticSequencePair) : seq.a 7 / seq.b 7 = 37 / 27 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_ratio_l787_78723


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l787_78737

-- Define the complex number z
def z (a : ℝ) : ℂ := (2 - Complex.I) * (2 + a * Complex.I)

-- Define the condition for z to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem complex_in_fourth_quadrant :
  ∃ a : ℝ, in_fourth_quadrant (z a) ∧ a = -2 :=
sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l787_78737


namespace NUMINAMATH_CALUDE_notebooks_promotion_result_l787_78771

/-- Calculates the maximum number of notebooks obtainable given an initial amount of money,
    the cost per notebook, and a promotion where stickers can be exchanged for free notebooks. -/
def max_notebooks (initial_money : ℕ) (cost_per_notebook : ℕ) (stickers_per_free_notebook : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given 150 rubles, with notebooks costing 4 rubles each,
    and a promotion where 5 stickers can be exchanged for an additional notebook
    (each notebook comes with a sticker), the maximum number of notebooks obtainable is 46. -/
theorem notebooks_promotion_result :
  max_notebooks 150 4 5 = 46 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_promotion_result_l787_78771


namespace NUMINAMATH_CALUDE_set_operations_l787_78725

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}

def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}

def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5/2}

theorem set_operations :
  (A ∩ B = {x | -1 < x ∧ x < 2}) ∧
  ((U \ B) ∪ P = {x | x ≤ 0 ∨ x ≥ 5/2}) ∧
  ((A ∩ B) ∩ (U \ P) = {x | 0 < x ∧ x < 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l787_78725


namespace NUMINAMATH_CALUDE_points_form_hyperbola_l787_78785

/-- The set of points (x,y) defined by x = 2sinh(t) and y = 4cosh(t) for real t forms a hyperbola -/
theorem points_form_hyperbola :
  ∀ (x y t : ℝ), x = 2 * Real.sinh t ∧ y = 4 * Real.cosh t →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_points_form_hyperbola_l787_78785


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l787_78726

theorem cow_chicken_problem (c h : ℕ) : 
  4 * c + 2 * h = 2 * (c + h) + 20 → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l787_78726


namespace NUMINAMATH_CALUDE_combination_permutation_equality_permutation_equation_solution_l787_78761

-- Define the combination function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the permutation function
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Theorem 1: Prove that C₁₀⁴ - C₇³ × A₃³ = 0
theorem combination_permutation_equality : C 10 4 - C 7 3 * A 3 3 = 0 := by
  sorry

-- Theorem 2: Prove that the solution to 3A₈ˣ = 4A₉ˣ⁻¹ is x = 6
theorem permutation_equation_solution :
  ∃ x : ℕ, (3 * A 8 x = 4 * A 9 (x - 1)) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_equality_permutation_equation_solution_l787_78761


namespace NUMINAMATH_CALUDE_rhombus_height_l787_78749

/-- A rhombus with diagonals of length 6 and 8 has a height of 24/5 -/
theorem rhombus_height (d₁ d₂ h : ℝ) (hd₁ : d₁ = 6) (hd₂ : d₂ = 8) :
  d₁ * d₂ = 4 * h * (d₁^2 / 4 + d₂^2 / 4).sqrt → h = 24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_height_l787_78749


namespace NUMINAMATH_CALUDE_cookies_with_seven_cups_l787_78760

/-- The number of cookies Lee can make with a given number of cups of flour -/
def cookies_made (cups : ℕ) : ℝ :=
  if cups ≤ 4 then 36
  else cookies_made (cups - 1) * 1.5

/-- The theorem stating the number of cookies Lee can make with 7 cups of flour -/
theorem cookies_with_seven_cups :
  cookies_made 7 = 121.5 := by
  sorry

end NUMINAMATH_CALUDE_cookies_with_seven_cups_l787_78760


namespace NUMINAMATH_CALUDE_second_stack_height_difference_l787_78799

def stack_problem (h : ℕ) : Prop :=
  let first_stack := 7
  let second_stack := h
  let third_stack := h + 7
  let fallen_blocks := first_stack + (second_stack - 2) + (third_stack - 3)
  (fallen_blocks = 33) ∧ (second_stack > first_stack)

theorem second_stack_height_difference : ∃ h : ℕ, stack_problem h ∧ (h - 7 = 5) :=
sorry

end NUMINAMATH_CALUDE_second_stack_height_difference_l787_78799


namespace NUMINAMATH_CALUDE_cookies_eaten_by_adults_l787_78777

/-- Proves that the number of cookies eaten by adults is 40 --/
theorem cookies_eaten_by_adults (total_cookies : ℕ) (num_children : ℕ) (child_cookies : ℕ) : 
  total_cookies = 120 →
  num_children = 4 →
  child_cookies = 20 →
  (total_cookies - num_children * child_cookies : ℚ) = (1/3 : ℚ) * total_cookies :=
by
  sorry

#check cookies_eaten_by_adults

end NUMINAMATH_CALUDE_cookies_eaten_by_adults_l787_78777


namespace NUMINAMATH_CALUDE_perfect_square_condition_l787_78711

/-- The expression (19a + b)^18 + (a + b)^18 + (a + 19b)^18 is a perfect square if and only if a = 0 and b = 0, where a and b are integers. -/
theorem perfect_square_condition (a b : ℤ) : 
  (∃ (k : ℤ), (19*a + b)^18 + (a + b)^18 + (a + 19*b)^18 = k^2) ↔ (a = 0 ∧ b = 0) := by
  sorry

#check perfect_square_condition

end NUMINAMATH_CALUDE_perfect_square_condition_l787_78711


namespace NUMINAMATH_CALUDE_not_perfect_square_l787_78780

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), 4 * n^2 + 4 * n + 4 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l787_78780


namespace NUMINAMATH_CALUDE_article_pricing_l787_78793

theorem article_pricing (P : ℝ) (P_pos : P > 0) : 
  (2/3 * P = 0.9 * ((2/3 * P) / 0.9)) → 
  ((P - ((2/3 * P) / 0.9)) / ((2/3 * P) / 0.9)) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_article_pricing_l787_78793


namespace NUMINAMATH_CALUDE_no_x_squared_term_l787_78730

theorem no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 2*a*x + a^2) = x^3 + (a^2 - 2*a)*x + a^2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l787_78730


namespace NUMINAMATH_CALUDE_fraction_equality_l787_78769

theorem fraction_equality (a b c : ℝ) (h1 : a = b) (h2 : c ≠ 0) : a / c = b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l787_78769


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l787_78740

-- Define repeating decimals
def repeating_decimal_6 : ℚ := 2/3
def repeating_decimal_4 : ℚ := 4/9
def repeating_decimal_8 : ℚ := 8/9

-- Theorem statement
theorem repeating_decimal_sum :
  repeating_decimal_6 - repeating_decimal_4 + repeating_decimal_8 = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l787_78740


namespace NUMINAMATH_CALUDE_red_light_time_proof_l787_78714

/-- Represents the time added by each red light -/
def time_per_red_light : ℕ := sorry

/-- Time for the first route with all green lights -/
def green_route_time : ℕ := 10

/-- Time for the second route -/
def second_route_time : ℕ := 14

/-- Number of stoplights on the first route -/
def num_stoplights : ℕ := 3

theorem red_light_time_proof :
  (green_route_time + num_stoplights * time_per_red_light = second_route_time + 5) ∧
  (time_per_red_light = 3) := by sorry

end NUMINAMATH_CALUDE_red_light_time_proof_l787_78714


namespace NUMINAMATH_CALUDE_triangle_preserving_characterization_l787_78704

/-- A function satisfying the triangle property -/
def TrianglePreserving (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c →
    (a + b > c ∧ b + c > a ∧ c + a > b ↔ f a + f b > f c ∧ f b + f c > f a ∧ f c + f a > f b)

/-- Main theorem: Characterization of triangle-preserving functions -/
theorem triangle_preserving_characterization (f : ℝ → ℝ) 
    (h₁ : ∀ x, 0 < x → 0 < f x) 
    (h₂ : TrianglePreserving f) :
    ∃ c : ℝ, c > 0 ∧ ∀ x, 0 < x → f x = c * x :=
  sorry

end NUMINAMATH_CALUDE_triangle_preserving_characterization_l787_78704


namespace NUMINAMATH_CALUDE_fraction_inequality_l787_78748

theorem fraction_inequality (a b m : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hab : a < b) : 
  (a + m) / (b + m) > a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l787_78748


namespace NUMINAMATH_CALUDE_mountain_hike_l787_78722

theorem mountain_hike (rate_up : ℝ) (time : ℝ) (rate_down_factor : ℝ) :
  rate_up = 8 →
  time = 2 →
  rate_down_factor = 1.5 →
  (rate_up * time) * rate_down_factor = 24 := by
sorry

end NUMINAMATH_CALUDE_mountain_hike_l787_78722


namespace NUMINAMATH_CALUDE_fraction_evaluation_l787_78768

theorem fraction_evaluation (a b c : ℝ) (h : a^3 - b^3 + c^3 ≠ 0) :
  (a^6 - b^6 + c^6) / (a^3 - b^3 + c^3) = a^3 + b^3 + c^3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l787_78768


namespace NUMINAMATH_CALUDE_A_intersect_B_l787_78747

def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | |x - 2| < 2}

theorem A_intersect_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l787_78747


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l787_78764

open Real

theorem perpendicular_tangents_intersection (a : ℝ) :
  ∃ x ∈ Set.Ioo 0 (π/2),
    (2 * sin x = a * cos x) ∧
    (2 * cos x) * (-a * sin x) = -1 →
  a = 2 * sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l787_78764


namespace NUMINAMATH_CALUDE_fifth_term_geometric_progression_l787_78718

theorem fifth_term_geometric_progression :
  let x : ℝ := -1 + Real.sqrt 5
  let r : ℝ := (1 + Real.sqrt 5) / (-1 + Real.sqrt 5)
  let a₁ : ℝ := x
  let a₂ : ℝ := x + 2
  let a₃ : ℝ := 2 * x + 6
  let a₅ : ℝ := r^4 * a₁
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) →
  a₅ = ((1 + Real.sqrt 5) / (-1 + Real.sqrt 5)) * (4 + 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_geometric_progression_l787_78718


namespace NUMINAMATH_CALUDE_clarissa_photos_count_l787_78790

/-- The number of photos brought by Cristina -/
def cristina_photos : ℕ := 7

/-- The number of photos brought by John -/
def john_photos : ℕ := 10

/-- The number of photos brought by Sarah -/
def sarah_photos : ℕ := 9

/-- The total number of slots in the photo album -/
def album_slots : ℕ := 40

/-- The number of photos Clarissa needs to bring -/
def clarissa_photos : ℕ := album_slots - (cristina_photos + john_photos + sarah_photos)

theorem clarissa_photos_count : clarissa_photos = 14 := by
  sorry

end NUMINAMATH_CALUDE_clarissa_photos_count_l787_78790


namespace NUMINAMATH_CALUDE_refrigerator_production_days_l787_78751

/-- The number of additional days needed to complete refrigerator production -/
def additional_days (total_required : ℕ) (days_worked : ℕ) (initial_rate : ℕ) (increased_rate : ℕ) : ℕ :=
  let produced := days_worked * initial_rate
  let remaining := total_required - produced
  remaining / increased_rate

theorem refrigerator_production_days : 
  additional_days 1590 12 80 90 = 7 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_production_days_l787_78751


namespace NUMINAMATH_CALUDE_smallest_y_cube_sum_l787_78741

theorem smallest_y_cube_sum (v w x y : ℕ+) : 
  v.val + 1 = w.val → w.val + 1 = x.val → x.val + 1 = y.val →
  v^3 + w^3 + x^3 = y^3 →
  ∀ (z : ℕ+), z < y → ¬(∃ (a b c : ℕ+), a.val + 1 = b.val ∧ b.val + 1 = c.val ∧ c.val + 1 = z.val ∧ a^3 + b^3 + c^3 = z^3) →
  y = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_y_cube_sum_l787_78741


namespace NUMINAMATH_CALUDE_alyssa_fruit_spending_l787_78745

theorem alyssa_fruit_spending (total_spent cherries_cost : ℚ)
  (h1 : total_spent = 21.93)
  (h2 : cherries_cost = 9.85) :
  total_spent - cherries_cost = 12.08 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_fruit_spending_l787_78745


namespace NUMINAMATH_CALUDE_hannahs_tshirts_l787_78703

theorem hannahs_tshirts (sweatshirt_count : ℕ) (sweatshirt_price : ℕ) (tshirt_price : ℕ) (total_spent : ℕ) :
  sweatshirt_count = 3 →
  sweatshirt_price = 15 →
  tshirt_price = 10 →
  total_spent = 65 →
  (total_spent - sweatshirt_count * sweatshirt_price) / tshirt_price = 2 := by
sorry

end NUMINAMATH_CALUDE_hannahs_tshirts_l787_78703


namespace NUMINAMATH_CALUDE_root_product_plus_one_l787_78798

theorem root_product_plus_one (r s t : ℂ) : 
  r^3 - 15*r^2 + 26*r - 8 = 0 →
  s^3 - 15*s^2 + 26*s - 8 = 0 →
  t^3 - 15*t^2 + 26*t - 8 = 0 →
  (1 + r) * (1 + s) * (1 + t) = 50 := by
sorry

end NUMINAMATH_CALUDE_root_product_plus_one_l787_78798


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_unknowns_l787_78757

theorem arithmetic_mean_of_unknowns (x y z : ℝ) 
  (h : (1 : ℝ) / (x * y) = y / (z - x + 1) ∧ y / (z - x + 1) = 2 / (z + 1)) : 
  x = (z + y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_unknowns_l787_78757


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_equality_condition_l787_78788

theorem min_value_exponential_sum (a b : ℝ) (h : a + 2 * b + 3 = 0) :
  2^a + 4^b ≥ Real.sqrt 2 / 2 :=
by sorry

theorem equality_condition (a b : ℝ) (h : a + 2 * b + 3 = 0) :
  ∃ (a₀ b₀ : ℝ), a₀ + 2 * b₀ + 3 = 0 ∧ 2^a₀ + 4^b₀ = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_equality_condition_l787_78788


namespace NUMINAMATH_CALUDE_system_consistency_solution_values_l787_78772

def is_consistent (a : ℝ) : Prop :=
  ∃ x : ℝ, 3 * x^2 - x - a - 10 = 0 ∧ (a + 4) * x + a + 12 = 0

theorem system_consistency :
  ∀ a : ℝ, is_consistent a ↔ (a = -10 ∨ a = -8 ∨ a = 4) :=
sorry

theorem solution_values :
  (is_consistent (-10) ∧ ∃ x : ℝ, 3 * x^2 - x - (-10) - 10 = 0 ∧ (-10 + 4) * x + (-10) + 12 = 0 ∧ x = -1/3) ∧
  (is_consistent (-8) ∧ ∃ x : ℝ, 3 * x^2 - x - (-8) - 10 = 0 ∧ (-8 + 4) * x + (-8) + 12 = 0 ∧ x = -1) ∧
  (is_consistent 4 ∧ ∃ x : ℝ, 3 * x^2 - x - 4 - 10 = 0 ∧ (4 + 4) * x + 4 + 12 = 0 ∧ x = -2) :=
sorry

end NUMINAMATH_CALUDE_system_consistency_solution_values_l787_78772


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_fifteen_l787_78781

theorem factorial_fraction_equals_fifteen :
  (Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 8) = 15 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_fifteen_l787_78781


namespace NUMINAMATH_CALUDE_intersection_line_of_planes_l787_78739

/-- Represents a plane with its first trace and angle of inclination -/
structure Plane where
  firstTrace : Line2D
  inclinationAngle : ℝ

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  point1 : Point2D
  point2 : Point2D

/-- Finds the intersection point of two 2D lines -/
def intersectionPoint (l1 l2 : Line2D) : Point2D :=
  sorry

/-- Constructs a point using the angles of inclination -/
def constructPoint (p1 p2 : Plane) : Point2D :=
  sorry

/-- Theorem stating that the intersection line of two planes can be determined
    by connecting two specific points -/
theorem intersection_line_of_planes (p1 p2 : Plane) :
  ∃ (l : Line2D),
    l.point1 = intersectionPoint p1.firstTrace p2.firstTrace ∧
    l.point2 = constructPoint p1 p2 :=
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_planes_l787_78739


namespace NUMINAMATH_CALUDE_balloon_distribution_l787_78716

theorem balloon_distribution (red white green chartreuse : ℕ) (friends : ℕ) : 
  red = 25 → white = 40 → green = 55 → chartreuse = 80 → friends = 10 →
  (red + white + green + chartreuse) % friends = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l787_78716


namespace NUMINAMATH_CALUDE_solve_cafeteria_problem_l787_78794

/-- Represents the amount paid by each friend in kopecks -/
structure Payment where
  misha : ℕ
  sasha : ℕ
  grisha : ℕ

/-- Represents the number of dishes each friend paid for -/
structure Dishes where
  misha : ℕ
  sasha : ℕ
  total : ℕ

def cafeteria_problem (p : Payment) (d : Dishes) : Prop :=
  -- All dishes cost the same
  ∃ (dish_cost : ℕ),
  -- Misha paid for 3 dishes
  p.misha = d.misha * dish_cost ∧
  -- Sasha paid for 2 dishes
  p.sasha = d.sasha * dish_cost ∧
  -- Together they ate 5 dishes
  d.total = d.misha + d.sasha ∧
  -- Grisha should pay his friends a total of 50 kopecks
  p.grisha = 50 ∧
  -- Each friend should receive an equal payment
  p.misha + p.sasha + p.grisha = d.total * dish_cost ∧
  -- Prove that Grisha should pay 40 kopecks to Misha and 10 kopecks to Sasha
  p.misha - (p.misha + p.sasha + p.grisha) / 3 = 40 ∧
  p.sasha - (p.misha + p.sasha + p.grisha) / 3 = 10

theorem solve_cafeteria_problem :
  ∃ (p : Payment) (d : Dishes), cafeteria_problem p d :=
sorry

end NUMINAMATH_CALUDE_solve_cafeteria_problem_l787_78794


namespace NUMINAMATH_CALUDE_angle_b_measure_l787_78762

theorem angle_b_measure (A B C : ℝ) (a b c : ℝ) : 
  0 < B ∧ B < π →
  0 < A ∧ A < π →
  0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = b * Real.cos C + c * Real.sin B →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  B = π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_b_measure_l787_78762


namespace NUMINAMATH_CALUDE_sequence_problem_l787_78778

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An arithmetic sequence -/
def IsArithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The problem statement -/
theorem sequence_problem (a : Sequence) 
  (h_arith : IsArithmetic a)
  (h_geom : IsGeometric (fun n => a (n + 1)))
  (h_a5 : a 5 = 1) :
  a 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l787_78778


namespace NUMINAMATH_CALUDE_quadratic_intersects_twice_iff_k_condition_l787_78753

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The discriminant of a quadratic function ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

/-- Predicate for a quadratic function intersecting x-axis at two points -/
def intersects_twice (a b c : ℝ) : Prop :=
  discriminant a b c > 0 ∧ a ≠ 0

theorem quadratic_intersects_twice_iff_k_condition (k : ℝ) :
  intersects_twice (k - 2) (-(2 * k - 1)) k ↔ k > -1/4 ∧ k ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersects_twice_iff_k_condition_l787_78753


namespace NUMINAMATH_CALUDE_increasing_condition_l787_78770

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x + 5

theorem increasing_condition (m : ℝ) :
  (∀ x ∈ Set.Ici (-2 : ℝ), Monotone (fun x => f m x)) ↔ m ∈ Set.Icc 0 (1/4) :=
sorry

end NUMINAMATH_CALUDE_increasing_condition_l787_78770


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l787_78787

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) / z = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l787_78787


namespace NUMINAMATH_CALUDE_probability_nine_correct_l787_78752

/-- The number of English-Russian expression pairs to be matched -/
def total_pairs : ℕ := 10

/-- The number of correctly matched pairs we're interested in -/
def correct_matches : ℕ := 9

/-- Represents the probability of getting exactly 9 out of 10 matches correct when choosing randomly -/
def prob_nine_correct : ℝ := 0

/-- Theorem stating that the probability of getting exactly 9 out of 10 matches correct when choosing randomly is 0 -/
theorem probability_nine_correct :
  prob_nine_correct = 0 := by sorry

end NUMINAMATH_CALUDE_probability_nine_correct_l787_78752


namespace NUMINAMATH_CALUDE_positive_integers_satisfying_conditions_l787_78743

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_8 n ∧ sum_of_digits n = 7 ∧ product_of_digits n = 6

theorem positive_integers_satisfying_conditions :
  {n : ℕ | n > 0 ∧ satisfies_conditions n} = {1312, 3112} :=
sorry

end NUMINAMATH_CALUDE_positive_integers_satisfying_conditions_l787_78743


namespace NUMINAMATH_CALUDE_janous_inequality_l787_78721

theorem janous_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l787_78721


namespace NUMINAMATH_CALUDE_peak_speed_scientific_notation_l787_78729

/-- The peak computing speed of a certain server in operations per second. -/
def peak_speed : ℕ := 403200000000

/-- The scientific notation representation of the peak speed. -/
def scientific_notation : ℝ := 4.032 * (10 ^ 11)

/-- Theorem stating that the peak speed is equal to its scientific notation representation. -/
theorem peak_speed_scientific_notation : (peak_speed : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_peak_speed_scientific_notation_l787_78729


namespace NUMINAMATH_CALUDE_cone_surface_area_ratio_l787_78763

theorem cone_surface_area_ratio (r l : ℝ) (h : l = 4 * r) :
  let side_area := (1 / 2) * Real.pi * l ^ 2
  let base_area := Real.pi * r ^ 2
  let total_area := side_area + base_area
  (total_area / side_area) = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_ratio_l787_78763


namespace NUMINAMATH_CALUDE_great_pyramid_height_l787_78736

theorem great_pyramid_height (h w : ℝ) : 
  h > 500 → 
  w = h + 234 → 
  h + w = 1274 → 
  h - 500 = 20 := by
sorry

end NUMINAMATH_CALUDE_great_pyramid_height_l787_78736


namespace NUMINAMATH_CALUDE_sum_divisible_by_five_l787_78791

theorem sum_divisible_by_five (m : ℤ) : 5 ∣ ((10 - m) + (m + 5)) := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_five_l787_78791


namespace NUMINAMATH_CALUDE_sheridan_fish_proof_l787_78789

/-- The number of fish Mrs. Sheridan gave to her sister -/
def fish_given_away : ℝ := 22.0

/-- The number of fish Mrs. Sheridan has now -/
def fish_remaining : ℕ := 25

/-- The initial number of fish Mrs. Sheridan had -/
def initial_fish : ℝ := fish_given_away + fish_remaining

theorem sheridan_fish_proof : initial_fish = 47.0 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_fish_proof_l787_78789


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l787_78728

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) = 11) ∧
  (∀ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 11) :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l787_78728


namespace NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l787_78755

theorem inscribed_circles_area_ratio (s : ℝ) (hs : s > 0) :
  let square_area := s^2
  let semicircle_area := (π * s^2) / 8
  let quarter_circle_area := (π * s^2) / 16
  let combined_area := semicircle_area + quarter_circle_area
  combined_area / square_area = 3 * π / 16 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l787_78755


namespace NUMINAMATH_CALUDE_train_journey_distance_l787_78724

/-- Represents the train journey problem -/
def TrainJourney (x v : ℝ) : Prop :=
  -- Train stops after 1 hour and remains halted for 0.5 hours
  let initial_stop_time : ℝ := 1.5
  -- Train continues at 3/4 of original speed
  let reduced_speed : ℝ := 3/4 * v
  -- Total delay equation
  let delay_equation : Prop := (x/v + initial_stop_time + (x-v)/reduced_speed - x/v = 3.5)
  -- Equation for incident 90 miles further
  let further_incident_equation : Prop := 
    ((x-90)/v + initial_stop_time + (x-90)/reduced_speed - x/v + 90/v = 3)
  
  -- All conditions must be satisfied
  delay_equation ∧ further_incident_equation

/-- The theorem to be proved -/
theorem train_journey_distance : 
  ∃ (v : ℝ), TrainJourney 600 v := by sorry

end NUMINAMATH_CALUDE_train_journey_distance_l787_78724
