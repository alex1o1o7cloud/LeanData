import Mathlib

namespace NUMINAMATH_CALUDE_parabola_shift_l3971_397186

def original_parabola (x : ℝ) : ℝ := -x^2

def shifted_parabola (x : ℝ) : ℝ := -(x - 2)^2

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l3971_397186


namespace NUMINAMATH_CALUDE_box_value_l3971_397122

theorem box_value (x : ℝ) : x * (-2) = 4 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_box_value_l3971_397122


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l3971_397169

theorem opposite_of_negative_six : -((-6 : ℝ)) = (6 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l3971_397169


namespace NUMINAMATH_CALUDE_max_value_less_than_two_l3971_397180

theorem max_value_less_than_two (m : ℝ) (hm1 : 1 < m) (hm2 : m < 1 + Real.sqrt 2) :
  ∀ x y : ℝ, y ≥ x → y ≤ m * x → x + y ≤ 1 → x + m * y < 2 := by
  sorry

#check max_value_less_than_two

end NUMINAMATH_CALUDE_max_value_less_than_two_l3971_397180


namespace NUMINAMATH_CALUDE_xiao_ming_correct_answers_l3971_397132

theorem xiao_ming_correct_answers 
  (total_questions : ℕ) 
  (correct_points : ℤ) 
  (wrong_points : ℤ) 
  (total_score : ℤ) 
  (h1 : total_questions = 20)
  (h2 : correct_points = 5)
  (h3 : wrong_points = -1)
  (h4 : total_score = 76) :
  ∃ (correct_answers : ℕ), 
    correct_answers ≤ total_questions ∧ 
    correct_points * correct_answers + wrong_points * (total_questions - correct_answers) = total_score ∧
    correct_answers = 16 := by
  sorry

#check xiao_ming_correct_answers

end NUMINAMATH_CALUDE_xiao_ming_correct_answers_l3971_397132


namespace NUMINAMATH_CALUDE_simplify_fraction_l3971_397174

theorem simplify_fraction : (45 : ℚ) / 75 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3971_397174


namespace NUMINAMATH_CALUDE_greg_savings_l3971_397194

theorem greg_savings (scooter_cost : ℕ) (amount_needed : ℕ) (amount_saved : ℕ) : 
  scooter_cost = 90 → amount_needed = 33 → amount_saved = scooter_cost - amount_needed → amount_saved = 57 := by
sorry

end NUMINAMATH_CALUDE_greg_savings_l3971_397194


namespace NUMINAMATH_CALUDE_diamond_calculation_l3971_397121

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation :
  (diamond (diamond 2 (1/2)) (-4)) - (diamond 2 (diamond (1/2) (-4))) = -5/12 :=
by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l3971_397121


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3971_397126

theorem arithmetic_expression_equality : 12 - 7 * (-32) + 16 / (-4) = 232 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3971_397126


namespace NUMINAMATH_CALUDE_cone_height_ratio_l3971_397129

theorem cone_height_ratio (base_circumference : ℝ) (original_height : ℝ) (shorter_volume : ℝ) :
  base_circumference = 20 * Real.pi →
  original_height = 24 →
  shorter_volume = 500 * Real.pi →
  ∃ (shorter_height : ℝ),
    shorter_volume = (1 / 3) * Real.pi * (base_circumference / (2 * Real.pi))^2 * shorter_height ∧
    shorter_height / original_height = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l3971_397129


namespace NUMINAMATH_CALUDE_total_hamburgers_for_lunch_l3971_397157

theorem total_hamburgers_for_lunch : 
  let initial_beef : ℕ := 15
  let initial_veggie : ℕ := 12
  let additional_beef : ℕ := 5
  let additional_veggie : ℕ := 7
  initial_beef + initial_veggie + additional_beef + additional_veggie = 39
  := by sorry

end NUMINAMATH_CALUDE_total_hamburgers_for_lunch_l3971_397157


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3971_397170

theorem interest_rate_calculation (initial_charge : ℝ) (final_amount : ℝ) (time : ℝ) :
  initial_charge = 75 →
  final_amount = 80.25 →
  time = 1 →
  (final_amount - initial_charge) / (initial_charge * time) = 0.07 :=
by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3971_397170


namespace NUMINAMATH_CALUDE_athena_spent_14_dollars_l3971_397116

/-- The total amount Athena spent on snacks for her friends -/
def total_spent (sandwich_price : ℝ) (sandwich_quantity : ℕ) (drink_price : ℝ) (drink_quantity : ℕ) : ℝ :=
  sandwich_price * sandwich_quantity + drink_price * drink_quantity

/-- Theorem stating that Athena spent $14 in total -/
theorem athena_spent_14_dollars :
  let sandwich_price : ℝ := 3
  let sandwich_quantity : ℕ := 3
  let drink_price : ℝ := 2.5
  let drink_quantity : ℕ := 2
  total_spent sandwich_price sandwich_quantity drink_price drink_quantity = 14 := by
sorry

end NUMINAMATH_CALUDE_athena_spent_14_dollars_l3971_397116


namespace NUMINAMATH_CALUDE_matrix_commutation_fraction_l3971_397124

/-- Given two matrices A and B, where A is fixed and B has variable entries,
    if AB = BA and 4b ≠ c, then (a - d) / (c - 4b) = 3/8 -/
theorem matrix_commutation_fraction (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (4 * b ≠ c) → ((a - d) / (c - 4 * b) = 3 / 8) := by
  sorry

end NUMINAMATH_CALUDE_matrix_commutation_fraction_l3971_397124


namespace NUMINAMATH_CALUDE_max_area_folded_rectangle_l3971_397197

/-- Given a rectangle ABCD with perimeter 24 and AB > AD, when folded along its diagonal AC
    such that AB meets DC at point P, the maximum area of triangle ADP is 72√2. -/
theorem max_area_folded_rectangle (AB AD : ℝ) (h1 : AB > AD) (h2 : AB + AD = 12) :
  let x := AB
  let a := (x^2 - 12*x + 72) / x
  let DP := (12*x - 72) / x
  let area := 3 * (12 - x) * ((12*x - 72) / x)
  ∃ (max_area : ℝ), (∀ x, 0 < x → x < 12 → area ≤ max_area) ∧ max_area = 72 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_area_folded_rectangle_l3971_397197


namespace NUMINAMATH_CALUDE_not_prime_expression_l3971_397156

theorem not_prime_expression (n k : ℤ) (h1 : n > 2) (h2 : k ≠ n) :
  ¬ Prime (n^2 - k*n + k - 1) :=
by sorry

end NUMINAMATH_CALUDE_not_prime_expression_l3971_397156


namespace NUMINAMATH_CALUDE_similar_quadrilateral_longest_side_l3971_397164

/-- Given a quadrilateral Q1 with side lengths a, b, c, d, and a similar quadrilateral Q2
    where the minimum side length of Q2 is equal to twice the minimum side length of Q1,
    prove that the longest side of Q2 is twice the longest side of Q1. -/
theorem similar_quadrilateral_longest_side
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hmin : a ≤ b ∧ a ≤ c ∧ a ≤ d)
  (hmax : b ≤ d ∧ c ≤ d)
  (h_similar : ∃ (k : ℝ), k > 0 ∧ k * a = 2 * a) :
  ∃ (l : ℝ), l = 2 * d ∧ l = max (k * a) (max (k * b) (max (k * c) (k * d))) :=
sorry

end NUMINAMATH_CALUDE_similar_quadrilateral_longest_side_l3971_397164


namespace NUMINAMATH_CALUDE_expression_equality_l3971_397143

theorem expression_equality (x : ℝ) : 
  (Real.sqrt x / Real.sqrt 0.81 + Real.sqrt 0.81 / Real.sqrt 0.49 = 2.507936507936508) → 
  x = 1.21 := by
sorry

end NUMINAMATH_CALUDE_expression_equality_l3971_397143


namespace NUMINAMATH_CALUDE_library_shelves_l3971_397172

theorem library_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 14240) (h2 : books_per_shelf = 8) :
  total_books / books_per_shelf = 1780 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_l3971_397172


namespace NUMINAMATH_CALUDE_scarf_problem_l3971_397144

theorem scarf_problem (initial_scarves : ℕ) (num_girls : ℕ) (final_scarves : ℕ) : 
  initial_scarves = 20 →
  num_girls = 17 →
  final_scarves ≠ 10 :=
by
  intro h_initial h_girls
  sorry


end NUMINAMATH_CALUDE_scarf_problem_l3971_397144


namespace NUMINAMATH_CALUDE_complex_number_location_l3971_397195

open Complex

theorem complex_number_location (z : ℂ) (h : z / (1 + I) = 2 - I) :
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3971_397195


namespace NUMINAMATH_CALUDE_irreducible_fractions_count_l3971_397106

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem irreducible_fractions_count : 
  (∃! (count : ℕ), ∃ (S : Finset ℕ), 
    S.card = count ∧ 
    (∀ n ∈ S, (1 : ℚ) / 16 < (n : ℚ) / 15 ∧ (n : ℚ) / 15 < (1 : ℚ) / 15) ∧
    (∀ n ∈ S, is_coprime n 15) ∧
    (∀ n : ℕ, (1 : ℚ) / 16 < (n : ℚ) / 15 ∧ (n : ℚ) / 15 < (1 : ℚ) / 15 → 
      is_coprime n 15 → n ∈ S)) ∧
  count = 8 :=
sorry

end NUMINAMATH_CALUDE_irreducible_fractions_count_l3971_397106


namespace NUMINAMATH_CALUDE_decimal_representation_nonzero_digits_l3971_397173

theorem decimal_representation_nonzero_digits :
  let x : ℚ := 720 / (2^6 * 3^5)
  ∃ (a b c d : ℕ) (r : ℚ),
    0 < a ∧ a < 10 ∧
    0 < b ∧ b < 10 ∧
    0 < c ∧ c < 10 ∧
    0 < d ∧ d < 10 ∧
    x = (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000 + (d : ℚ) / 10000 + r ∧
    0 ≤ r ∧ r < 1/10000 :=
by sorry

end NUMINAMATH_CALUDE_decimal_representation_nonzero_digits_l3971_397173


namespace NUMINAMATH_CALUDE_four_circles_max_regions_l3971_397130

/-- The maximum number of regions that n circles can divide a plane into -/
def max_regions (n : ℕ) : ℕ :=
  n * (n - 1) + 2

/-- Assumption that for n = 1, 2, 3, n circles divide the plane into at most 2^n parts -/
axiom max_regions_small (n : ℕ) (h : n ≤ 3) : max_regions n ≤ 2^n

/-- Theorem: The maximum number of regions that four circles can divide a plane into is 14 -/
theorem four_circles_max_regions : max_regions 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_four_circles_max_regions_l3971_397130


namespace NUMINAMATH_CALUDE_waiter_customers_theorem_l3971_397181

def final_customers (initial new left : ℕ) : ℕ :=
  initial - left + new

theorem waiter_customers_theorem (initial new left : ℕ) 
  (h1 : initial ≥ left) : 
  final_customers initial new left = initial - left + new :=
by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_theorem_l3971_397181


namespace NUMINAMATH_CALUDE_james_waiting_time_l3971_397109

/-- The number of days it took for James' pain to subside -/
def pain_subsided_days : ℕ := 3

/-- The factor by which the full healing time is longer than the pain subsidence time -/
def healing_factor : ℕ := 5

/-- The number of days James waits after healing before working out -/
def wait_before_workout_days : ℕ := 3

/-- The total number of days until James can lift heavy again -/
def total_days_until_heavy_lifting : ℕ := 39

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem james_waiting_time :
  (total_days_until_heavy_lifting - (pain_subsided_days * healing_factor + wait_before_workout_days)) / days_per_week = 3 := by
  sorry

end NUMINAMATH_CALUDE_james_waiting_time_l3971_397109


namespace NUMINAMATH_CALUDE_binomial_coefficient_x4_in_expansion_l3971_397189

/-- The binomial coefficient of the term containing x^4 in the expansion of (x^2 + 1/x)^5 is 10 -/
theorem binomial_coefficient_x4_in_expansion : 
  ∃ k : ℕ, (Nat.choose 5 k) * (4 : ℤ) = (10 : ℤ) ∧ 
    10 - 3 * k = 4 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x4_in_expansion_l3971_397189


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l3971_397188

/-- A natural number is composite if it has a proper factor -/
def IsComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A natural number can be represented as the sum of two composite numbers -/
def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

/-- 11 is the largest natural number that cannot be represented as the sum of two composite numbers -/
theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l3971_397188


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3971_397167

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3971_397167


namespace NUMINAMATH_CALUDE_percentage_of_difference_l3971_397182

theorem percentage_of_difference (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (40 / 100) * (x + y) →
  y = (11.11111111111111 / 100) * x →
  P = 6.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_difference_l3971_397182


namespace NUMINAMATH_CALUDE_system_solution_l3971_397119

theorem system_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) :
  (x₁ + x₂ + x₃ = 6) ∧
  (x₂ + x₃ + x₄ = 9) ∧
  (x₃ + x₄ + x₅ = 3) ∧
  (x₄ + x₅ + x₆ = -3) ∧
  (x₅ + x₆ + x₇ = -9) ∧
  (x₆ + x₇ + x₈ = -6) ∧
  (x₇ + x₈ + x₁ = -2) ∧
  (x₈ + x₁ + x₂ = 2) →
  x₁ = 1 ∧ x₂ = 2 ∧ x₃ = 3 ∧ x₄ = 4 ∧ x₅ = -4 ∧ x₆ = -3 ∧ x₇ = -2 ∧ x₈ = -1 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3971_397119


namespace NUMINAMATH_CALUDE_smallest_cross_family_bound_l3971_397171

/-- A family of subsets A of a finite set X is a cross family if for every subset B of X,
    B is comparable with at least one subset in A. -/
def IsCrossFamily (X : Finset α) (A : Finset (Finset α)) : Prop :=
  ∀ B : Finset α, B ⊆ X → ∃ A' ∈ A, A' ⊆ B ∨ B ⊆ A'

/-- A is the smallest cross family if no proper subfamily of A is a cross family. -/
def IsSmallestCrossFamily (X : Finset α) (A : Finset (Finset α)) : Prop :=
  IsCrossFamily X A ∧ ∀ A' ⊂ A, ¬IsCrossFamily X A'

theorem smallest_cross_family_bound {α : Type*} [DecidableEq α] (X : Finset α) (A : Finset (Finset α)) :
  IsSmallestCrossFamily X A → A.card ≤ Nat.choose X.card (X.card / 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_cross_family_bound_l3971_397171


namespace NUMINAMATH_CALUDE_initial_cookies_l3971_397100

/-- The number of basketball team members -/
def team_members : ℕ := 8

/-- The number of cookies Andy ate -/
def andy_ate : ℕ := 3

/-- The number of cookies Andy gave to his brother -/
def brother_got : ℕ := 5

/-- The number of cookies the first player took -/
def first_player_cookies : ℕ := 1

/-- The increase in cookies taken by each subsequent player -/
def cookie_increase : ℕ := 2

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a₁ aₙ : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- The total number of cookies taken by the basketball team -/
def team_cookies : ℕ :=
  arithmetic_sum team_members first_player_cookies (first_player_cookies + cookie_increase * (team_members - 1))

/-- The theorem stating the initial number of cookies -/
theorem initial_cookies : 
  andy_ate + brother_got + team_cookies = 72 := by sorry

end NUMINAMATH_CALUDE_initial_cookies_l3971_397100


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3971_397110

theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 7) (h2 : DF = 8) (h3 : EF = 9) :
  let s := (DE + DF + EF) / 2
  let A := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  A / s = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3971_397110


namespace NUMINAMATH_CALUDE_quadratic_always_negative_l3971_397192

theorem quadratic_always_negative :
  ∀ x : ℝ, -15 * x^2 + 4 * x - 6 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_negative_l3971_397192


namespace NUMINAMATH_CALUDE_min_handshakes_coach_l3971_397118

/-- Represents the number of handshakes in a volleyball tournament --/
structure VolleyballTournament where
  n : ℕ  -- Total number of players
  m : ℕ  -- Number of players in the smaller team
  k₁ : ℕ -- Number of handshakes by the coach with fewer players
  h : ℕ  -- Total number of handshakes

/-- Conditions for the volleyball tournament --/
def tournament_conditions (t : VolleyballTournament) : Prop :=
  t.n = 3 * t.m ∧                                  -- Total players is 3 times the smaller team
  t.h = (t.n * (t.n - 1)) / 2 + 3 * t.k₁ ∧         -- Total handshakes equation
  t.h = 435                                        -- Given total handshakes

/-- Theorem stating the minimum number of handshakes for the coach with fewer players --/
theorem min_handshakes_coach (t : VolleyballTournament) :
  tournament_conditions t → t.k₁ ≥ 0 → t.k₁ = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_min_handshakes_coach_l3971_397118


namespace NUMINAMATH_CALUDE_triangle_problem_l3971_397148

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.b * Real.cos t.C + t.c * Real.cos t.B = 2 * t.a * Real.cos t.A)
  (h2 : t.b * t.c * Real.cos t.A = Real.sqrt 3) :
  t.A = π / 3 ∧ (1 / 2) * t.b * t.c * Real.sin t.A = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3971_397148


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_9_6_l3971_397101

theorem sqrt_sum_equals_9_6 (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (16 - y^2) = 5) : 
  Real.sqrt (64 - y^2) + Real.sqrt (16 - y^2) = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_9_6_l3971_397101


namespace NUMINAMATH_CALUDE_lost_sea_creatures_l3971_397102

/-- Represents the count of sea creatures Harry collected --/
structure SeaCreatures where
  seaStars : ℕ
  seashells : ℕ
  snails : ℕ
  crabs : ℕ

/-- Represents the number of each type of sea creature that reproduced --/
structure Reproduction where
  seaStars : ℕ
  seashells : ℕ
  snails : ℕ

def initialCount : SeaCreatures :=
  { seaStars := 34, seashells := 21, snails := 29, crabs := 17 }

def reproductionCount : Reproduction :=
  { seaStars := 5, seashells := 3, snails := 4 }

def finalCount : ℕ := 105

def totalAfterReproduction (initial : SeaCreatures) (reproduction : Reproduction) : ℕ :=
  (initial.seaStars + reproduction.seaStars) +
  (initial.seashells + reproduction.seashells) +
  (initial.snails + reproduction.snails) +
  initial.crabs

theorem lost_sea_creatures : 
  totalAfterReproduction initialCount reproductionCount - finalCount = 8 := by
  sorry

end NUMINAMATH_CALUDE_lost_sea_creatures_l3971_397102


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3971_397105

theorem complex_equation_solution :
  let z : ℂ := (1 - I)^2 + 1 + 3*I
  ∀ a b : ℝ, z^2 + a*z + b = 1 - I → a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3971_397105


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3971_397198

theorem triangle_angle_measure (D E F : ℝ) : 
  -- DEF is a triangle
  D + E + F = 180 →
  -- Measure of angle E is three times the measure of angle F
  E = 3 * F →
  -- Angle F is 15°
  F = 15 →
  -- Then the measure of angle D is 120°
  D = 120 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3971_397198


namespace NUMINAMATH_CALUDE_arlo_books_count_l3971_397193

/-- Represents the ratio of school supplies -/
structure SupplyRatio where
  books : Nat
  pens : Nat
  notebooks : Nat

/-- Calculates the number of books bought given a total number of items and a supply ratio -/
def calculate_books (total_items : Nat) (ratio : SupplyRatio) : Nat :=
  let total_ratio := ratio.books + ratio.pens + ratio.notebooks
  let sets := total_items / total_ratio
  sets * ratio.books

/-- Theorem: Given the conditions, Arlo bought 350 books -/
theorem arlo_books_count :
  let ratio : SupplyRatio := { books := 7, pens := 3, notebooks := 2 }
  let total_items : Nat := 600
  calculate_books total_items ratio = 350 := by
  sorry

#eval calculate_books 600 { books := 7, pens := 3, notebooks := 2 }

end NUMINAMATH_CALUDE_arlo_books_count_l3971_397193


namespace NUMINAMATH_CALUDE_sin_theta_value_l3971_397128

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 5 * Real.cos θ) 
  (h2 : 0 < θ) 
  (h3 : θ < Real.pi / 2) : 
  Real.sin θ = 1 := by sorry

end NUMINAMATH_CALUDE_sin_theta_value_l3971_397128


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l3971_397191

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (average_age : ℝ) 
  (wicket_keeper_age_diff : ℝ) 
  (remaining_average_diff : ℝ) :
  team_size = 11 →
  average_age = 29 →
  wicket_keeper_age_diff = 3 →
  remaining_average_diff = 1 →
  ∃ (captain_age : ℝ),
    team_size * average_age = 
      (team_size - 2) * (average_age - remaining_average_diff) + 
      captain_age + 
      (average_age + wicket_keeper_age_diff) :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l3971_397191


namespace NUMINAMATH_CALUDE_seashell_collection_problem_l3971_397120

/-- Calculates the total number of seashells after Leo gives away a quarter of his collection. -/
def final_seashell_count (henry_shells : ℕ) (paul_shells : ℕ) (initial_total : ℕ) : ℕ :=
  let leo_shells := initial_total - (henry_shells + paul_shells)
  let leo_remaining := leo_shells - (leo_shells / 4)
  henry_shells + paul_shells + leo_remaining

/-- Theorem stating that given the initial conditions, the final seashell count is 53. -/
theorem seashell_collection_problem :
  final_seashell_count 11 24 59 = 53 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_problem_l3971_397120


namespace NUMINAMATH_CALUDE_num_perfect_square_factors_is_525_l3971_397113

/-- The number of positive perfect square factors of 2^8 * 3^9 * 5^12 * 7^4 -/
def num_perfect_square_factors : ℕ := 525

/-- The exponents of prime factors in the given product -/
def prime_exponents : List ℕ := [8, 9, 12, 4]

/-- Counts the number of even numbers (including 0) up to and including a given number -/
def count_even_numbers_up_to (n : ℕ) : ℕ :=
  (n / 2) + 1

/-- Theorem: The number of positive perfect square factors of 2^8 * 3^9 * 5^12 * 7^4 is 525 -/
theorem num_perfect_square_factors_is_525 :
  num_perfect_square_factors = (prime_exponents.map count_even_numbers_up_to).prod :=
sorry

end NUMINAMATH_CALUDE_num_perfect_square_factors_is_525_l3971_397113


namespace NUMINAMATH_CALUDE_dagger_example_l3971_397187

-- Define the ⋄ operation
def dagger (m n p q : ℚ) : ℚ := m^2 * p * (q / n)

-- Theorem statement
theorem dagger_example : dagger 5 9 4 6 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l3971_397187


namespace NUMINAMATH_CALUDE_three_digit_integer_with_specific_remainders_l3971_397177

theorem three_digit_integer_with_specific_remainders :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
    n % 7 = 3 ∧ n % 8 = 6 ∧ n % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_integer_with_specific_remainders_l3971_397177


namespace NUMINAMATH_CALUDE_lcm_of_462_and_150_l3971_397153

theorem lcm_of_462_and_150 :
  let a : ℕ := 462
  let b : ℕ := 150
  let hcf : ℕ := 30
  Nat.lcm a b = 2310 :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_462_and_150_l3971_397153


namespace NUMINAMATH_CALUDE_johns_number_is_1500_l3971_397161

/-- John's number satisfies the given conditions -/
def is_johns_number (n : ℕ) : Prop :=
  n % 125 = 0 ∧ n % 30 = 0 ∧ 800 ≤ n ∧ n ≤ 2000

/-- There exists a unique number satisfying John's conditions, and it is 1500 -/
theorem johns_number_is_1500 : ∃! n : ℕ, is_johns_number n ∧ n = 1500 :=
sorry

end NUMINAMATH_CALUDE_johns_number_is_1500_l3971_397161


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l3971_397175

theorem mean_equality_implies_z_value : ∃ z : ℝ,
  (6 + 15 + 9 + 20) / 4 = (13 + z) / 2 → z = 12 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l3971_397175


namespace NUMINAMATH_CALUDE_subset_P_l3971_397139

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_subset_P_l3971_397139


namespace NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l3971_397185

/-- Proves that the difference between Alberto's and Bjorn's biking distances is 10 miles -/
theorem alberto_bjorn_distance_difference : 
  ∀ (alberto_distance bjorn_distance : ℕ), 
  alberto_distance = 75 → 
  bjorn_distance = 65 → 
  alberto_distance - bjorn_distance = 10 := by
sorry

end NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l3971_397185


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3971_397141

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_condition : a * b + b * c + c * a + 2 * a * b * c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ 2 ∧ 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c = 2 ↔ 
    a = (-3 + Real.sqrt 17) / 4 ∧ 
    b = (-3 + Real.sqrt 17) / 4 ∧ 
    c = (-3 + Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3971_397141


namespace NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_l3971_397117

theorem xy_squared_minus_x_squared_y (x y : ℝ) 
  (h1 : x - y = 1/2) 
  (h2 : x * y = 4/3) : 
  x * y^2 - x^2 * y = -2/3 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_l3971_397117


namespace NUMINAMATH_CALUDE_diamond_op_five_three_l3971_397134

def diamond_op (x y : ℝ) : ℝ := 4 * x + 6 * y

theorem diamond_op_five_three : diamond_op 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_diamond_op_five_three_l3971_397134


namespace NUMINAMATH_CALUDE_symmetric_parabolas_product_l3971_397178

/-- Given two parabolas that are symmetric with respect to a line, 
    prove that the product of their parameters is -3 -/
theorem symmetric_parabolas_product (a p m : ℝ) : 
  a ≠ 0 → p > 0 → 
  (∀ x y : ℝ, y = a * x^2 - 3 * x + 3 ↔ 
    ∃ x' y', y' = x + m ∧ x = y' - m ∧ y'^2 = 2 * p * x') →
  a * p * m = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_parabolas_product_l3971_397178


namespace NUMINAMATH_CALUDE_simplify_expression_l3971_397165

theorem simplify_expression (m : ℝ) : 150*m - 72*m + 3*(5*m) = 93*m := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3971_397165


namespace NUMINAMATH_CALUDE_necessary_condition_abs_l3971_397114

theorem necessary_condition_abs (x y : ℝ) (hx : x > 0) : x > |y| → x > y := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_abs_l3971_397114


namespace NUMINAMATH_CALUDE_beef_not_used_in_soup_l3971_397123

-- Define the variables
def total_beef : ℝ := 4
def vegetables_used : ℝ := 6

-- Define the theorem
theorem beef_not_used_in_soup :
  ∃ (beef_used beef_not_used : ℝ),
    beef_used = vegetables_used / 2 ∧
    beef_not_used = total_beef - beef_used ∧
    beef_not_used = 1 := by
  sorry

end NUMINAMATH_CALUDE_beef_not_used_in_soup_l3971_397123


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l3971_397159

/-- Given vectors in R², prove that if they satisfy certain conditions, then x = 1/2 -/
theorem vector_parallel_condition (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  let u : Fin 2 → ℝ := a + 2 • b
  let v : Fin 2 → ℝ := 2 • a - b
  (∃ (k : ℝ), k ≠ 0 ∧ u = k • v) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l3971_397159


namespace NUMINAMATH_CALUDE_sandbox_width_l3971_397190

/-- A rectangular sandbox with perimeter 30 feet and length twice the width has a width of 5 feet. -/
theorem sandbox_width :
  ∀ (width length : ℝ),
  width > 0 →
  length > 0 →
  length = 2 * width →
  2 * length + 2 * width = 30 →
  width = 5 := by
sorry

end NUMINAMATH_CALUDE_sandbox_width_l3971_397190


namespace NUMINAMATH_CALUDE_mika_stickers_problem_l3971_397150

/-- The number of stickers Mika's mother gave her -/
def mothers_stickers (initial : Float) (bought : Float) (birthday : Float) (sister : Float) (final_total : Float) : Float :=
  final_total - (initial + bought + birthday + sister)

theorem mika_stickers_problem (initial : Float) (bought : Float) (birthday : Float) (sister : Float) (final_total : Float)
  (h1 : initial = 20.0)
  (h2 : bought = 26.0)
  (h3 : birthday = 20.0)
  (h4 : sister = 6.0)
  (h5 : final_total = 130.0) :
  mothers_stickers initial bought birthday sister final_total = 58.0 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_problem_l3971_397150


namespace NUMINAMATH_CALUDE_greatest_integer_abs_inequality_l3971_397112

theorem greatest_integer_abs_inequality :
  (∃ (x : ℤ), ∀ (y : ℤ), |3*y - 2| ≤ 21 → y ≤ x) ∧
  (∀ (x : ℤ), |3*x - 2| ≤ 21 → x ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_abs_inequality_l3971_397112


namespace NUMINAMATH_CALUDE_pet_store_combinations_l3971_397199

def num_puppies : ℕ := 10
def num_kittens : ℕ := 8
def num_hamsters : ℕ := 12
def num_rabbits : ℕ := 4

def alice_choice : ℕ := num_puppies + num_rabbits

/-- The number of ways Alice, Bob, Charlie, and Dana can buy pets and leave the store satisfied. -/
theorem pet_store_combinations : ℕ := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l3971_397199


namespace NUMINAMATH_CALUDE_tan_theta_equals_five_twelfths_l3971_397179

/-- Given a dilation matrix D and a rotation matrix R, prove that tan θ = 5/12 -/
theorem tan_theta_equals_five_twelfths 
  (k : ℝ) 
  (θ : ℝ) 
  (hk : k > 0) 
  (D : Matrix (Fin 2) (Fin 2) ℝ) 
  (R : Matrix (Fin 2) (Fin 2) ℝ) 
  (hD : D = ![![k, 0], ![0, k]]) 
  (hR : R = ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]) 
  (h_prod : R * D = ![![12, -5], ![5, 12]]) : 
  Real.tan θ = 5/12 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_equals_five_twelfths_l3971_397179


namespace NUMINAMATH_CALUDE_subset_pairs_count_l3971_397138

/-- Given a fixed set S with n elements, this theorem states that the number of ordered pairs (A, B) 
    where A and B are subsets of S and A ⊆ B is equal to 3^n. -/
theorem subset_pairs_count (n : ℕ) : 
  (Finset.powerset (Finset.range n)).card = 3^n := by sorry

end NUMINAMATH_CALUDE_subset_pairs_count_l3971_397138


namespace NUMINAMATH_CALUDE_equation_solution_l3971_397163

theorem equation_solution : ∃ x : ℚ, (1 / 3 + 1 / x = 7 / 9 + 1) ∧ (x = 9 / 13) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3971_397163


namespace NUMINAMATH_CALUDE_bank_charge_increase_l3971_397166

/-- The percentage increase in the ratio of price to transactions from the old
    charging system to the new charging system -/
theorem bank_charge_increase (old_price : ℝ) (old_transactions : ℕ)
    (new_price : ℝ) (new_transactions : ℕ) :
    old_price = 1 →
    old_transactions = 5 →
    new_price = 0.75 →
    new_transactions = 3 →
    (((new_price / new_transactions) - (old_price / old_transactions)) /
     (old_price / old_transactions)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bank_charge_increase_l3971_397166


namespace NUMINAMATH_CALUDE_harvard_attendance_l3971_397125

def total_applicants : ℕ := 20000
def acceptance_rate : ℚ := 5 / 100
def attendance_rate : ℚ := 90 / 100

theorem harvard_attendance : 
  ⌊(total_applicants : ℚ) * acceptance_rate * attendance_rate⌋ = 900 := by
  sorry

end NUMINAMATH_CALUDE_harvard_attendance_l3971_397125


namespace NUMINAMATH_CALUDE_rectangle_construction_solutions_l3971_397151

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomRight : Point
  bottomLeft : Point

/-- Check if a point lies on any side of the rectangle -/
def pointOnRectangle (p : Point) (r : Rectangle) : Prop :=
  (p.x = r.topLeft.x ∧ p.y ≥ r.bottomLeft.y ∧ p.y ≤ r.topLeft.y) ∨
  (p.x = r.topRight.x ∧ p.y ≥ r.bottomRight.y ∧ p.y ≤ r.topRight.y) ∨
  (p.y = r.topLeft.y ∧ p.x ≥ r.topLeft.x ∧ p.x ≤ r.topRight.x) ∨
  (p.y = r.bottomLeft.y ∧ p.x ≥ r.bottomLeft.x ∧ p.x ≤ r.bottomRight.x)

/-- Check if the rectangle has a side of length 'a' -/
def hasLengthA (r : Rectangle) (a : ℝ) : Prop :=
  (r.topRight.x - r.topLeft.x = a) ∨
  (r.topRight.y - r.bottomRight.y = a)

/-- The main theorem -/
theorem rectangle_construction_solutions 
  (A B C D : Point) (a : ℝ) (h : a > 0) :
  ∃ (solutions : Finset Rectangle), 
    solutions.card = 12 ∧
    ∀ r ∈ solutions, 
      pointOnRectangle A r ∧
      pointOnRectangle B r ∧
      pointOnRectangle C r ∧
      pointOnRectangle D r ∧
      hasLengthA r a :=
sorry

end NUMINAMATH_CALUDE_rectangle_construction_solutions_l3971_397151


namespace NUMINAMATH_CALUDE_oranges_in_sack_l3971_397146

/-- The number of sacks harvested per day -/
def sacks_per_day : ℕ := 66

/-- The number of days of harvest -/
def harvest_days : ℕ := 87

/-- The total number of oranges after the harvest -/
def total_oranges : ℕ := 143550

/-- The number of oranges in each sack -/
def oranges_per_sack : ℕ := total_oranges / (sacks_per_day * harvest_days)

theorem oranges_in_sack : oranges_per_sack = 25 := by
  sorry

end NUMINAMATH_CALUDE_oranges_in_sack_l3971_397146


namespace NUMINAMATH_CALUDE_conditional_probability_rain_wind_l3971_397127

theorem conditional_probability_rain_wind (P_rain P_wind_and_rain : ℚ) 
  (h1 : P_rain = 4 / 15)
  (h2 : P_wind_and_rain = 1 / 10) :
  P_wind_and_rain / P_rain = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_wind_l3971_397127


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3971_397137

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3971_397137


namespace NUMINAMATH_CALUDE_seashell_collection_l3971_397135

/-- Theorem: Given an initial collection of 19 seashells and adding 6 more,
    the total number of seashells is 25. -/
theorem seashell_collection (initial : Nat) (added : Nat) (total : Nat) : 
  initial = 19 → added = 6 → total = initial + added → total = 25 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_l3971_397135


namespace NUMINAMATH_CALUDE_arithmetic_mean_log_implies_geometric_mean_but_not_conversely_l3971_397158

open Real

theorem arithmetic_mean_log_implies_geometric_mean_but_not_conversely 
  (x y z : ℝ) : 
  (2 * log y = log x + log z → y ^ 2 = x * z) ∧
  ¬(y ^ 2 = x * z → 2 * log y = log x + log z) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_log_implies_geometric_mean_but_not_conversely_l3971_397158


namespace NUMINAMATH_CALUDE_swimmer_downstream_distance_l3971_397108

/-- Proves that a swimmer travels 32 km downstream given specific conditions -/
theorem swimmer_downstream_distance 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : upstream_distance = 24) 
  (h2 : time = 4) 
  (h3 : still_water_speed = 7) : 
  ∃ (downstream_distance : ℝ), downstream_distance = 32 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_downstream_distance_l3971_397108


namespace NUMINAMATH_CALUDE_ann_speed_l3971_397140

/-- Given cyclists' speeds, prove Ann's speed -/
theorem ann_speed (tom_speed : ℚ) (jerry_speed : ℚ) (ann_speed : ℚ) : 
  tom_speed = 6 →
  jerry_speed = 3/4 * tom_speed →
  ann_speed = 4/3 * jerry_speed →
  ann_speed = 6 := by
sorry

end NUMINAMATH_CALUDE_ann_speed_l3971_397140


namespace NUMINAMATH_CALUDE_function_F_theorem_l3971_397111

theorem function_F_theorem (F : ℝ → ℝ) 
  (h_diff : Differentiable ℝ F) 
  (h_init : F 0 = -1)
  (h_deriv : ∀ x, deriv F x = Real.sin (Real.sin (Real.sin (Real.sin x))) * 
    Real.cos (Real.sin (Real.sin x)) * Real.cos (Real.sin x) * Real.cos x) :
  ∀ x, F x = -Real.cos (Real.sin (Real.sin (Real.sin x))) := by
sorry

end NUMINAMATH_CALUDE_function_F_theorem_l3971_397111


namespace NUMINAMATH_CALUDE_min_product_value_l3971_397147

def S (n : ℕ+) : ℚ := n / (n + 1)

def b (n : ℕ+) : ℤ := n - 8

def product (n : ℕ+) : ℚ := (b n : ℚ) * S n

theorem min_product_value :
  ∃ (m : ℕ+), ∀ (n : ℕ+), product m ≤ product n ∧ product m = -4 :=
sorry

end NUMINAMATH_CALUDE_min_product_value_l3971_397147


namespace NUMINAMATH_CALUDE_find_second_number_l3971_397104

theorem find_second_number (G N : ℕ) (h1 : G = 101) (h2 : 4351 % G = 8) (h3 : N % G = 10) :
  N = 4359 := by
  sorry

end NUMINAMATH_CALUDE_find_second_number_l3971_397104


namespace NUMINAMATH_CALUDE_denominator_value_l3971_397162

theorem denominator_value (x : ℝ) (h : (1 / x) ^ 1 = 0.25) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_denominator_value_l3971_397162


namespace NUMINAMATH_CALUDE_inequality_not_hold_l3971_397115

theorem inequality_not_hold (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a < 1) :
  ¬(a * b < b^2 ∧ b^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_hold_l3971_397115


namespace NUMINAMATH_CALUDE_trivia_team_selection_l3971_397183

theorem trivia_team_selection (total_students : ℕ) (num_groups : ℕ) (students_per_group : ℕ) :
  total_students = 17 →
  num_groups = 3 →
  students_per_group = 4 →
  total_students - (num_groups * students_per_group) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_selection_l3971_397183


namespace NUMINAMATH_CALUDE_cos_shift_equivalence_l3971_397160

open Real

theorem cos_shift_equivalence (x : ℝ) :
  cos (2 * x - π / 6) = sin (2 * (x - π / 6) + π / 2) := by sorry

end NUMINAMATH_CALUDE_cos_shift_equivalence_l3971_397160


namespace NUMINAMATH_CALUDE_arithmetic_mean_arrangement_l3971_397196

theorem arithmetic_mean_arrangement (n : ℕ+) :
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ (i j : Fin n), i ≠ j →
      ∀ (k : Fin n), (p i < p k ∧ p k < p j) ∨ (p j < p k ∧ p k < p i) →
        (p i + p j : ℚ) / 2 ≠ p k := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_arrangement_l3971_397196


namespace NUMINAMATH_CALUDE_west_distance_calculation_l3971_397107

-- Define the given constants
def total_distance : ℝ := 150
def north_distance : ℝ := 55

-- Theorem statement
theorem west_distance_calculation :
  total_distance - north_distance = 95 := by sorry

end NUMINAMATH_CALUDE_west_distance_calculation_l3971_397107


namespace NUMINAMATH_CALUDE_bananas_left_l3971_397103

theorem bananas_left (initial : ℕ) (eaten : ℕ) : 
  initial = 12 → eaten = 4 → initial - eaten = 8 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l3971_397103


namespace NUMINAMATH_CALUDE_two_digit_sum_square_property_l3971_397152

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def satisfiesCondition (A : ℕ) : Prop :=
  (sumOfDigits A)^2 = sumOfDigits (A^2)

def isTwoDigit (A : ℕ) : Prop :=
  10 ≤ A ∧ A ≤ 99

theorem two_digit_sum_square_property :
  ∀ A : ℕ, isTwoDigit A →
    (satisfiesCondition A ↔ 
      A = 10 ∨ A = 11 ∨ A = 12 ∨ A = 13 ∨ A = 20 ∨ A = 21 ∨ A = 22 ∨ A = 30 ∨ A = 31) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_sum_square_property_l3971_397152


namespace NUMINAMATH_CALUDE_max_product_of_three_l3971_397154

def S : Set Int := {-9, -5, -3, 0, 2, 6, 8}

theorem max_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a * b * c ≤ 360 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_three_l3971_397154


namespace NUMINAMATH_CALUDE_initial_average_mark_l3971_397184

/-- Proves that the initial average mark of a class is 80, given specific conditions --/
theorem initial_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 10 →
  excluded_students = 5 →
  excluded_avg = 70 →
  remaining_avg = 90 →
  (total_students * (total_students * remaining_avg - excluded_students * excluded_avg)) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_mark_l3971_397184


namespace NUMINAMATH_CALUDE_problem_statement_l3971_397155

theorem problem_statement (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = 3) :
  3 * a^2 * b + 3 * a * b^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3971_397155


namespace NUMINAMATH_CALUDE_solve_for_N_l3971_397133

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Represents the grid of numbers -/
structure NumberGrid where
  row : ArithmeticSequence
  col1 : ArithmeticSequence
  col2 : ArithmeticSequence

/-- The problem setup -/
def problem_setup : NumberGrid where
  row := { first := 21, diff := -5 }
  col1 := { first := 6, diff := 4 }
  col2 := { first := -7, diff := -2 }

/-- The theorem to prove -/
theorem solve_for_N (grid : NumberGrid) : 
  grid.row.first = 21 ∧ 
  (grid.col1.first + 3 * grid.col1.diff = 14) ∧
  (grid.col1.first + 4 * grid.col1.diff = 18) ∧
  (grid.col2.first + 4 * grid.col2.diff = -17) →
  grid.col2.first = -7 := by
  sorry

#eval problem_setup.col2.first

end NUMINAMATH_CALUDE_solve_for_N_l3971_397133


namespace NUMINAMATH_CALUDE_reflect_x_axis_l3971_397136

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflect_x_axis (p : Point) (h : p = Point.mk 3 1) :
  reflectX p = Point.mk 3 (-1) := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_axis_l3971_397136


namespace NUMINAMATH_CALUDE_inequality_addition_l3971_397142

theorem inequality_addition (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l3971_397142


namespace NUMINAMATH_CALUDE_acme_vowel_soup_combinations_l3971_397131

/-- Represents the number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- Represents the number of times each vowel appears in the soup -/
def vowel_occurrences : ℕ := 6

/-- Represents the number of wildcard characters in the soup -/
def num_wildcards : ℕ := 1

/-- Represents the length of the words to be formed -/
def word_length : ℕ := 6

/-- Represents the total number of character choices for each position in the word -/
def choices_per_position : ℕ := num_vowels + num_wildcards

/-- Theorem stating that the number of possible six-letter words is 46656 -/
theorem acme_vowel_soup_combinations :
  choices_per_position ^ word_length = 46656 := by
  sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_combinations_l3971_397131


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3971_397149

theorem sum_of_two_numbers (x y : ℝ) : 
  0.5 * x + 0.3333 * y = 11 → 
  max x y = 15 → 
  x + y = 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3971_397149


namespace NUMINAMATH_CALUDE_closer_to_one_than_four_closer_to_zero_than_ax_l3971_397176

-- Part 1
theorem closer_to_one_than_four (x : ℝ) :
  |x^2 - 1| < |4 - 1| → x ∈ Set.Ioo (-2 : ℝ) 2 :=
sorry

-- Part 2
theorem closer_to_zero_than_ax (x a : ℝ) :
  a > 0 → |x^2 + a| < |(a + 1) * x| →
  (0 < a ∧ a < 1 → x ∈ Set.Ioo (-1 : ℝ) (-a) ∪ Set.Ioo a 1) ∧
  (a = 1 → False) ∧
  (a > 1 → x ∈ Set.Ioo (-a : ℝ) (-1) ∪ Set.Ioo 1 a) :=
sorry

end NUMINAMATH_CALUDE_closer_to_one_than_four_closer_to_zero_than_ax_l3971_397176


namespace NUMINAMATH_CALUDE_game_points_l3971_397168

/-- The number of points earned in a video game level --/
def points_earned (total_enemies : ℕ) (enemies_left : ℕ) (points_per_enemy : ℕ) : ℕ :=
  (total_enemies - enemies_left) * points_per_enemy

/-- Theorem: In a level with 8 enemies, destroying all but 6 of them, with 5 points per enemy, results in 10 points --/
theorem game_points : points_earned 8 6 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_game_points_l3971_397168


namespace NUMINAMATH_CALUDE_expression_evaluation_l3971_397145

theorem expression_evaluation : (900^2 : ℝ) / (306^2 - 294^2) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3971_397145
