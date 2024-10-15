import Mathlib

namespace NUMINAMATH_CALUDE_andy_wrong_questions_l1600_160019

theorem andy_wrong_questions (a b c d : ℕ) : 
  a + b = c + d →
  a + d = b + c + 6 →
  c = 6 →
  a = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_andy_wrong_questions_l1600_160019


namespace NUMINAMATH_CALUDE_drawBalls_18_4_l1600_160048

/-- The number of balls in the bin -/
def n : ℕ := 18

/-- The number of balls to be drawn -/
def k : ℕ := 4

/-- The number of ways to draw k balls from n balls, 
    where the first ball is returned and the rest are not -/
def drawBalls (n k : ℕ) : ℕ := n * n * (n - 1) * (n - 2)

/-- Theorem stating that drawing 4 balls from 18 balls, 
    where the first ball is returned and the rest are not, 
    can be done in 87984 ways -/
theorem drawBalls_18_4 : drawBalls n k = 87984 := by sorry

end NUMINAMATH_CALUDE_drawBalls_18_4_l1600_160048


namespace NUMINAMATH_CALUDE_unique_intersection_bounded_difference_l1600_160064

-- Define the set U of functions satisfying the conditions
def U : Set (ℝ → ℝ) :=
  {f | ∃ x, f x = 2 * x ∧ ∀ x, 0 < (deriv f) x ∧ (deriv f) x < 2}

-- Statement 1: For any f in U, f(x) = 2x has exactly one solution
theorem unique_intersection (f : ℝ → ℝ) (hf : f ∈ U) :
  ∃! x, f x = 2 * x :=
sorry

-- Statement 2: For any h in U and x₁, x₂ close to 2023, |h(x₁) - h(x₂)| < 4
theorem bounded_difference (h : ℝ → ℝ) (hh : h ∈ U) :
  ∀ x₁ x₂, |x₁ - 2023| < 1 → |x₂ - 2023| < 1 → |h x₁ - h x₂| < 4 :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_bounded_difference_l1600_160064


namespace NUMINAMATH_CALUDE_power_calculation_l1600_160003

theorem power_calculation : 16^12 * 8^8 / 2^60 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1600_160003


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_average_l1600_160013

theorem arithmetic_sequence_middle_average (a : ℕ → ℕ) :
  (∀ i j, i < j → a i < a j) →  -- ascending order
  (∀ i, a (i + 1) - a i = a (i + 2) - a (i + 1)) →  -- arithmetic sequence
  (a 1 + a 2 + a 3) / 3 = 20 →  -- average of first three
  (a 5 + a 6 + a 7) / 3 = 24 →  -- average of last three
  (a 3 + a 4 + a 5) / 3 = 22 :=  -- average of middle three
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_average_l1600_160013


namespace NUMINAMATH_CALUDE_circle_equation_through_points_l1600_160076

theorem circle_equation_through_points : 
  let equation (x y : ℝ) := x^2 + y^2 - 4*x - 6*y
  ∀ (x y : ℝ), 
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) → 
    equation x y = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_through_points_l1600_160076


namespace NUMINAMATH_CALUDE_lawrence_county_houses_l1600_160077

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 574

/-- The total number of houses in Lawrence County after the housing boom -/
def total_houses : ℕ := houses_before + houses_built

theorem lawrence_county_houses : total_houses = 2000 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_houses_l1600_160077


namespace NUMINAMATH_CALUDE_negation_equivalence_l1600_160051

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 ≥ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1600_160051


namespace NUMINAMATH_CALUDE_orchids_planted_today_calculation_l1600_160029

/-- The number of orchid bushes planted today in the park. -/
def orchids_planted_today (current : ℕ) (tomorrow : ℕ) (final : ℕ) : ℕ :=
  final - current - tomorrow

/-- Theorem stating the number of orchid bushes planted today. -/
theorem orchids_planted_today_calculation :
  orchids_planted_today 47 25 109 = 37 := by
  sorry

end NUMINAMATH_CALUDE_orchids_planted_today_calculation_l1600_160029


namespace NUMINAMATH_CALUDE_erased_number_proof_l1600_160015

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1) = 866 / 19 →
  x = 326 :=
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l1600_160015


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1600_160000

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1600_160000


namespace NUMINAMATH_CALUDE_contest_ranking_l1600_160041

theorem contest_ranking (A B C D : ℝ) 
  (eq1 : B + D = 2*(A + C) - 20)
  (ineq1 : A + 2*C < 2*B + D)
  (ineq2 : D > 2*(B + C)) :
  D > B ∧ B > A ∧ A > C :=
sorry

end NUMINAMATH_CALUDE_contest_ranking_l1600_160041


namespace NUMINAMATH_CALUDE_square_sum_primes_l1600_160098

theorem square_sum_primes (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (h1 : ∃ a : ℕ, pq + 1 = a^2)
  (h2 : ∃ b : ℕ, pr + 1 = b^2)
  (h3 : ∃ c : ℕ, qr - p = c^2) :
  ∃ d : ℕ, p + 2*q*r + 2 = d^2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_primes_l1600_160098


namespace NUMINAMATH_CALUDE_line_points_k_value_l1600_160025

/-- Given a line with equation x - 5/2y + 1 = 0 and two points (m, n) and (m + 1/2, n + 1/k) on this line,
    prove that k = 3/5 -/
theorem line_points_k_value (m n k : ℝ) :
  (m - 5/2 * n + 1 = 0) →
  (m + 1/2 - 5/2 * (n + 1/k) + 1 = 0) →
  k = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_line_points_k_value_l1600_160025


namespace NUMINAMATH_CALUDE_jumping_contest_l1600_160036

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump squirrel_jump : ℕ)
  (grasshopper_obstacle frog_obstacle mouse_obstacle squirrel_obstacle : ℕ)
  (h1 : grasshopper_jump = 19)
  (h2 : grasshopper_obstacle = 3)
  (h3 : frog_jump = grasshopper_jump + 10)
  (h4 : frog_obstacle = 0)
  (h5 : mouse_jump = frog_jump + 20)
  (h6 : mouse_obstacle = 5)
  (h7 : squirrel_jump = mouse_jump - 7)
  (h8 : squirrel_obstacle = 2) :
  (mouse_jump - mouse_obstacle) - (grasshopper_jump - grasshopper_obstacle) = 28 := by
  sorry

#check jumping_contest

end NUMINAMATH_CALUDE_jumping_contest_l1600_160036


namespace NUMINAMATH_CALUDE_sodas_sold_in_afternoon_l1600_160042

theorem sodas_sold_in_afternoon (morning_sodas : ℕ) (total_sodas : ℕ) 
  (h1 : morning_sodas = 77) 
  (h2 : total_sodas = 96) : 
  total_sodas - morning_sodas = 19 := by
  sorry

end NUMINAMATH_CALUDE_sodas_sold_in_afternoon_l1600_160042


namespace NUMINAMATH_CALUDE_difference_between_point_eight_and_half_l1600_160096

theorem difference_between_point_eight_and_half : (0.8 : ℝ) - (1/2 : ℝ) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_point_eight_and_half_l1600_160096


namespace NUMINAMATH_CALUDE_sandy_shopping_l1600_160018

def shopping_equation (X Y : ℝ) : Prop :=
  let pie_cost : ℝ := 6
  let sandwich_cost : ℝ := 3
  let book_cost : ℝ := 10
  let book_discount : ℝ := 0.2
  let sales_tax : ℝ := 0.05
  let discounted_book_cost : ℝ := book_cost * (1 - book_discount)
  let subtotal : ℝ := pie_cost + sandwich_cost + discounted_book_cost
  let total_cost : ℝ := subtotal * (1 + sales_tax)
  Y = X - total_cost

theorem sandy_shopping : 
  ∀ X Y : ℝ, shopping_equation X Y ↔ Y = X - 17.85 := by sorry

end NUMINAMATH_CALUDE_sandy_shopping_l1600_160018


namespace NUMINAMATH_CALUDE_linear_function_through_zero_one_l1600_160081

/-- A linear function is a function of the form f(x) = kx + b where k and b are real numbers. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, ∀ x, f x = k * x + b

/-- Theorem: There exists a linear function that passes through the point (0,1). -/
theorem linear_function_through_zero_one : ∃ f : ℝ → ℝ, LinearFunction f ∧ f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_zero_one_l1600_160081


namespace NUMINAMATH_CALUDE_cristina_speed_l1600_160055

/-- Cristina's running speed in a race with Nicky -/
theorem cristina_speed (head_start : ℝ) (nicky_speed : ℝ) (catch_up_time : ℝ) 
  (h1 : head_start = 36)
  (h2 : nicky_speed = 3)
  (h3 : catch_up_time = 12) :
  (head_start + nicky_speed * catch_up_time) / catch_up_time = 6 :=
by sorry

end NUMINAMATH_CALUDE_cristina_speed_l1600_160055


namespace NUMINAMATH_CALUDE_equation_solution_l1600_160022

theorem equation_solution (y : ℝ) : 
  (|y - 4|^2 + 3*y = 14) ↔ (y = (5 + Real.sqrt 17)/2 ∨ y = (5 - Real.sqrt 17)/2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1600_160022


namespace NUMINAMATH_CALUDE_opposites_sum_zero_l1600_160097

theorem opposites_sum_zero (a b : ℚ) : a + b = 0 → a = -b := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_zero_l1600_160097


namespace NUMINAMATH_CALUDE_horner_v₂_for_specific_polynomial_v₂_value_at_10_l1600_160065

/-- Horner's Rule for a polynomial of degree 4 -/
def horner_rule (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₄ * x + a₃) * x + a₂) * x + a₁ * x + a₀

/-- The second intermediate value in Horner's Rule calculation -/
def v₂ (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  (a₄ * x + a₃) * x + a₂

theorem horner_v₂_for_specific_polynomial (x : ℝ) :
  v₂ 3 1 0 2 4 x = 3 * x * x + x := by sorry

theorem v₂_value_at_10 :
  v₂ 3 1 0 2 4 10 = 310 := by sorry

end NUMINAMATH_CALUDE_horner_v₂_for_specific_polynomial_v₂_value_at_10_l1600_160065


namespace NUMINAMATH_CALUDE_level3_available_spots_l1600_160035

/-- Represents a parking level in a multi-story parking lot -/
structure ParkingLevel where
  totalSpots : ℕ
  parkedCars : ℕ
  reservedParkedCars : ℕ

/-- Calculates the available non-reserved parking spots on a given level -/
def availableNonReservedSpots (level : ParkingLevel) : ℕ :=
  level.totalSpots - (level.parkedCars - level.reservedParkedCars)

/-- Theorem stating that the available non-reserved parking spots on level 3 is 450 -/
theorem level3_available_spots :
  let level3 : ParkingLevel := {
    totalSpots := 480,
    parkedCars := 45,
    reservedParkedCars := 15
  }
  availableNonReservedSpots level3 = 450 := by
  sorry

end NUMINAMATH_CALUDE_level3_available_spots_l1600_160035


namespace NUMINAMATH_CALUDE_coin_circumference_diameter_ratio_l1600_160004

theorem coin_circumference_diameter_ratio :
  let diameter : ℝ := 100
  let circumference : ℝ := 314
  circumference / diameter = 3.14 := by sorry

end NUMINAMATH_CALUDE_coin_circumference_diameter_ratio_l1600_160004


namespace NUMINAMATH_CALUDE_cross_country_winning_scores_l1600_160085

/-- Represents a cross country race between two teams -/
structure CrossCountryRace where
  num_runners_per_team : ℕ
  total_runners : ℕ
  min_score : ℕ
  max_score : ℕ

/-- Calculates the number of different winning scores in a cross country race -/
def num_winning_scores (race : CrossCountryRace) : ℕ :=
  race.max_score - race.min_score + 1

/-- The specific cross country race described in the problem -/
def specific_race : CrossCountryRace :=
  { num_runners_per_team := 6
  , total_runners := 12
  , min_score := 21
  , max_score := 39 }

theorem cross_country_winning_scores :
  num_winning_scores specific_race = 19 := by
  sorry

end NUMINAMATH_CALUDE_cross_country_winning_scores_l1600_160085


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l1600_160008

theorem inverse_proposition_false : 
  ¬ (∀ a b c : ℝ, a > b → a * c^2 > b * c^2) := by
sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l1600_160008


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1600_160031

theorem constant_term_expansion (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 10) :
  (∃ k : ℕ, ∃ r : ℕ, n = 3 * r ∧ n = 2 * k) ↔ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1600_160031


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l1600_160093

/-- Given a line y = ax + 1 and a hyperbola 3x^2 - y^2 = 1 that intersect at points A and B,
    if a circle with AB as its diameter passes through the origin,
    then a = 1 or a = -1 -/
theorem line_hyperbola_intersection (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = a * A.1 + 1 ∧ 3 * A.1^2 - A.2^2 = 1) ∧ 
    (B.2 = a * B.1 + 1 ∧ 3 * B.1^2 - B.2^2 = 1) ∧ 
    A ≠ B ∧
    (A.1 * B.1 + A.2 * B.2 = 0)) →
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l1600_160093


namespace NUMINAMATH_CALUDE_factorization_proof_l1600_160030

theorem factorization_proof (x : ℝ) : 
  (x^2 - 1) * (x^4 + x^2 + 1) - (x^3 + 1)^2 = -2 * (x + 1) * (x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1600_160030


namespace NUMINAMATH_CALUDE_ratio_problem_l1600_160083

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 2) (h2 : c/b = 3) : (a+b)/(b+c) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1600_160083


namespace NUMINAMATH_CALUDE_people_per_column_l1600_160014

theorem people_per_column (total_people : ℕ) 
  (h1 : total_people = 30 * 16) 
  (h2 : total_people = 15 * (total_people / 15)) : 
  total_people / 15 = 32 := by
  sorry

end NUMINAMATH_CALUDE_people_per_column_l1600_160014


namespace NUMINAMATH_CALUDE_smallest_nonprime_no_small_factors_is_529_l1600_160066

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p, is_prime p → p < k → ¬(n % p = 0)

def smallest_nonprime_no_small_factors : ℕ → Prop
| n => ¬(is_prime n) ∧ 
       n > 1 ∧ 
       has_no_prime_factors_less_than n 20 ∧
       ∀ m, 1 < m → m < n → ¬(¬(is_prime m) ∧ has_no_prime_factors_less_than m 20)

theorem smallest_nonprime_no_small_factors_is_529 :
  ∃ n, smallest_nonprime_no_small_factors n ∧ n = 529 :=
sorry

end NUMINAMATH_CALUDE_smallest_nonprime_no_small_factors_is_529_l1600_160066


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1600_160072

-- Define the sets M and N
def M : Set ℝ := {x | (x - 1) * (x - 4) = 0}
def N : Set ℝ := {x | (x + 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1600_160072


namespace NUMINAMATH_CALUDE_union_A_B_range_of_a_l1600_160033

-- Define sets A, B, and C
def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x | x > -2}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem 1: A ∪ B = {x | x ≥ -4}
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -4} := by sorry

-- Theorem 2: Range of a is [-4, -1]
theorem range_of_a : 
  (∀ a : ℝ, C a ∩ A = C a) ↔ a ∈ Set.Icc (-4) (-1) := by sorry

end NUMINAMATH_CALUDE_union_A_B_range_of_a_l1600_160033


namespace NUMINAMATH_CALUDE_gcd_1729_1314_l1600_160095

theorem gcd_1729_1314 : Nat.gcd 1729 1314 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_1314_l1600_160095


namespace NUMINAMATH_CALUDE_parabola_properties_l1600_160073

/-- Parabola passing through a specific point -/
structure Parabola where
  a : ℝ
  passes_through : a * (2 - 3)^2 - 1 = 1

/-- The number of units to move the parabola up for one x-axis intersection -/
def move_up_units (p : Parabola) : ℝ := 1

/-- Theorem stating the properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  p.a = 2 ∧ 
  (∃! x : ℝ, 2 * (x - 3)^2 - 1 + move_up_units p = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1600_160073


namespace NUMINAMATH_CALUDE_fourth_power_one_fourth_equals_decimal_l1600_160094

theorem fourth_power_one_fourth_equals_decimal : (1 / 4 : ℚ) ^ 4 = 390625 / 100000000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_one_fourth_equals_decimal_l1600_160094


namespace NUMINAMATH_CALUDE_alex_class_size_l1600_160088

/-- In a class, given a student who is both the 30th best and 30th worst, 
    the total number of students in the class is 59. -/
theorem alex_class_size (n : ℕ) 
  (h1 : ∃ (alex : ℕ), alex ≤ n ∧ alex = 30)  -- Alex is 30th best
  (h2 : ∃ (alex : ℕ), alex ≤ n ∧ alex = 30)  -- Alex is 30th worst
  : n = 59 := by
  sorry

end NUMINAMATH_CALUDE_alex_class_size_l1600_160088


namespace NUMINAMATH_CALUDE_negation_statement_1_negation_statement_2_negation_statement_3_l1600_160069

-- Define the set of prime numbers
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)
def P : Set ℕ := {p : ℕ | isPrime p}

-- Statement 1
theorem negation_statement_1 :
  (∀ n : ℕ, ∃ p ∈ P, n ≤ p) ↔ (∃ n : ℕ, ∀ p ∈ P, p ≤ n) :=
sorry

-- Statement 2
theorem negation_statement_2 :
  (∀ n : ℤ, ∃! p : ℤ, n + p = 0) ↔ (∃ n : ℤ, ∀ p : ℤ, n + p ≠ 0) :=
sorry

-- Statement 3
theorem negation_statement_3 :
  (∃ y : ℝ, ∀ x : ℝ, ∃ c : ℝ, x * y = c) ↔
  (∀ y : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * y ≠ x₂ * y) :=
sorry

end NUMINAMATH_CALUDE_negation_statement_1_negation_statement_2_negation_statement_3_l1600_160069


namespace NUMINAMATH_CALUDE_lollipop_distribution_l1600_160086

theorem lollipop_distribution (num_kids : ℕ) (additional_lollipops : ℕ) (initial_lollipops : ℕ) : 
  num_kids = 42 → 
  additional_lollipops = 22 → 
  (initial_lollipops + additional_lollipops) % num_kids = 0 → 
  initial_lollipops < num_kids → 
  initial_lollipops = 62 := by
sorry

end NUMINAMATH_CALUDE_lollipop_distribution_l1600_160086


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1600_160002

theorem solution_set_inequality (x : ℝ) : 
  x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1600_160002


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1600_160082

theorem constant_term_binomial_expansion :
  ∀ x : ℝ, ∃ t : ℕ → ℝ,
    (∀ r, t r = (-1)^r * (Nat.choose 6 r) * (2^((12 - 3*r) * x))) ∧
    (∃ k, t k = 15 ∧ ∀ r ≠ k, ∃ n : ℤ, t r = 2^(n*x)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1600_160082


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1600_160079

theorem greatest_divisor_with_remainders (n : ℕ) : 
  (∃ q₁ : ℕ, 6215 = 144 * q₁ + 23) ∧
  (∃ q₂ : ℕ, 7373 = 144 * q₂ + 29) ∧
  (∀ m : ℕ, m > 144 → 
    (∃ q₃ q₄ : ℕ, 6215 = m * q₃ + 23 ∧ 7373 = m * q₄ + 29) → False) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1600_160079


namespace NUMINAMATH_CALUDE_currency_notes_existence_l1600_160075

theorem currency_notes_existence : 
  ∃ (x y z : ℕ), x + 5*y + 10*z = 480 ∧ x + y + z = 90 := by
  sorry

end NUMINAMATH_CALUDE_currency_notes_existence_l1600_160075


namespace NUMINAMATH_CALUDE_other_x_axis_point_on_circle_l1600_160037

def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

theorem other_x_axis_point_on_circle :
  let C : Set (ℝ × ℝ) := Circle (0, 0) 16
  (16, 0) ∈ C →
  (-16, 0) ∈ C ∧ (-16, 0).2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_x_axis_point_on_circle_l1600_160037


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1600_160038

theorem largest_angle_in_triangle (x y z : ℝ) : 
  x = 60 → y = 70 → x + y + z = 180 → 
  ∃ max_angle : ℝ, max_angle = 70 ∧ max_angle ≥ x ∧ max_angle ≥ y ∧ max_angle ≥ z :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1600_160038


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l1600_160080

/-- Represents a point in 1D space -/
structure Point1D where
  x : ℝ

/-- Distance between two points in 1D -/
def distance (p q : Point1D) : ℝ := |p.x - q.x|

/-- Sum of squared distances from a point to multiple points -/
def sumSquaredDistances (p : Point1D) (points : List Point1D) : ℝ :=
  points.foldl (fun sum q => sum + (distance p q)^2) 0

/-- The problem statement -/
theorem min_sum_squared_distances :
  ∃ (a b c d e : Point1D),
    distance a b = 2 ∧
    distance b c = 2 ∧
    distance c d = 3 ∧
    distance d e = 7 ∧
    (∀ p : Point1D, sumSquaredDistances p [a, b, c, d, e] ≥ 133.2) ∧
    (∃ q : Point1D, sumSquaredDistances q [a, b, c, d, e] = 133.2) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l1600_160080


namespace NUMINAMATH_CALUDE_box_volume_l1600_160020

theorem box_volume (side1 side2 upper : ℝ) 
  (h1 : side1 = 120)
  (h2 : side2 = 72)
  (h3 : upper = 60) :
  ∃ (l w h : ℝ), 
    l * w = side1 ∧ 
    w * h = side2 ∧ 
    l * h = upper ∧ 
    l * w * h = 720 :=
sorry

end NUMINAMATH_CALUDE_box_volume_l1600_160020


namespace NUMINAMATH_CALUDE_sequence_ratio_values_l1600_160053

/-- Two sequences where one is arithmetic and the other is geometric -/
structure SequencePair :=
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arithmetic : (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) ∨ 
                  (∃ d : ℝ, ∀ n : ℕ, b (n + 1) - b n = d))
  (h_geometric : (∃ r : ℝ, ∀ n : ℕ, a (n + 1) / a n = r) ∨ 
                 (∃ r : ℝ, ∀ n : ℕ, b (n + 1) / b n = r))

/-- The theorem stating the possible values of a_3 / b_3 -/
theorem sequence_ratio_values (s : SequencePair)
  (h1 : s.a 1 = s.b 1)
  (h2 : s.a 2 / s.b 2 = 2)
  (h4 : s.a 4 / s.b 4 = 8) :
  s.a 3 / s.b 3 = -5 ∨ s.a 3 / s.b 3 = -16/5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_values_l1600_160053


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_diff_fractions_l1600_160032

theorem reciprocal_of_sum_diff_fractions : 
  (1 / (1/3 + 1/4 - 1/12) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_diff_fractions_l1600_160032


namespace NUMINAMATH_CALUDE_tan_alpha_fourth_quadrant_l1600_160010

theorem tan_alpha_fourth_quadrant (α : Real) : 
  (π / 2 < α ∧ α < 2 * π) →  -- α is in the fourth quadrant
  (Real.cos (π / 2 + α) = 4 / 5) → 
  Real.tan α = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_fourth_quadrant_l1600_160010


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1600_160062

theorem expand_and_simplify (a b : ℝ) : (a + b) * (a - 4 * b) = a^2 - 3*a*b - 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1600_160062


namespace NUMINAMATH_CALUDE_marbles_cost_calculation_l1600_160049

/-- The amount spent on marbles when the total spent on toys is known, along with the costs of a football and baseball. -/
def marbles_cost (total_spent football_cost baseball_cost : ℚ) : ℚ :=
  total_spent - (football_cost + baseball_cost)

/-- Theorem stating that the cost of marbles is the difference between the total spent and the sum of football and baseball costs. -/
theorem marbles_cost_calculation (total_spent football_cost baseball_cost : ℚ) 
  (h1 : total_spent = 20.52)
  (h2 : football_cost = 4.95)
  (h3 : baseball_cost = 6.52) : 
  marbles_cost total_spent football_cost baseball_cost = 9.05 := by
sorry

end NUMINAMATH_CALUDE_marbles_cost_calculation_l1600_160049


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1600_160017

theorem cube_equation_solution (a d : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * d) : d = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1600_160017


namespace NUMINAMATH_CALUDE_intersection_chord_length_l1600_160027

/-- Circle in 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line passing through origin in polar form -/
structure PolarLine where
  angle : ℝ

/-- Chord formed by intersection of circle and line -/
def chord_length (c : Circle) (l : PolarLine) : ℝ :=
  sorry

theorem intersection_chord_length :
  let c : Circle := { center := (0, -6), radius := 5 }
  let l : PolarLine := { angle := Real.arctan (Real.sqrt 5 / 2) }
  chord_length c l = 6 := by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l1600_160027


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l1600_160034

theorem chicken_wings_distribution (num_friends : ℕ) (pre_cooked : ℕ) (additional_cooked : ℕ) :
  num_friends = 4 →
  pre_cooked = 9 →
  additional_cooked = 7 →
  (pre_cooked + additional_cooked) / num_friends = 4 :=
by sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l1600_160034


namespace NUMINAMATH_CALUDE_handshake_theorem_l1600_160078

/-- The number of handshakes in a group where each person shakes hands with a fixed number of others -/
def total_handshakes (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a group of 30 people, where each person shakes hands with exactly 3 others, 
    the total number of handshakes is 45 -/
theorem handshake_theorem : 
  total_handshakes 30 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_handshake_theorem_l1600_160078


namespace NUMINAMATH_CALUDE_percent_decrease_l1600_160046

theorem percent_decrease (original_price sale_price : ℝ) (h : original_price > 0) :
  let decrease := original_price - sale_price
  let percent_decrease := (decrease / original_price) * 100
  original_price = 100 ∧ sale_price = 75 → percent_decrease = 25 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_l1600_160046


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_l1600_160047

-- Define a circle
variable (circle : Type) [MetricSpace circle]

-- Define the chords and intersection point
variable (chord1 chord2 : Set circle)
variable (P : circle)

-- Define the segments of the first chord
variable (PA PB : ℝ)

-- Define the ratio of the segments of the second chord
variable (r : ℚ)

-- State the theorem
theorem intersecting_chords_theorem 
  (h1 : P ∈ chord1 ∩ chord2)
  (h2 : PA = 12)
  (h3 : PB = 18)
  (h4 : r = 3 / 8)
  : ∃ (PC PD : ℝ), PC + PD = 33 ∧ PC / PD = r := by
  sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_l1600_160047


namespace NUMINAMATH_CALUDE_problems_per_worksheet_l1600_160016

/-- Given a set of worksheets with the following properties:
    - There are 9 worksheets in total
    - 5 worksheets have been graded
    - 16 problems remain to be graded
    This theorem proves that there are 4 problems on each worksheet. -/
theorem problems_per_worksheet (total_worksheets : Nat) (graded_worksheets : Nat) (remaining_problems : Nat)
    (h1 : total_worksheets = 9)
    (h2 : graded_worksheets = 5)
    (h3 : remaining_problems = 16) :
    (remaining_problems / (total_worksheets - graded_worksheets) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_problems_per_worksheet_l1600_160016


namespace NUMINAMATH_CALUDE_company_fund_problem_l1600_160045

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) :
  (80 * n = initial_fund + 8) →
  (70 * n + 160 = initial_fund) →
  initial_fund = 1352 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l1600_160045


namespace NUMINAMATH_CALUDE_product_a4_b4_l1600_160007

theorem product_a4_b4 (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
sorry

end NUMINAMATH_CALUDE_product_a4_b4_l1600_160007


namespace NUMINAMATH_CALUDE_max_rectangles_in_square_l1600_160001

/-- Given a square with side length 14 cm and rectangles of width 2 cm and length 8 cm,
    the maximum number of whole rectangles that can fit within the square is 12. -/
theorem max_rectangles_in_square : ∀ (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ),
  square_side = 14 →
  rect_width = 2 →
  rect_length = 8 →
  ⌊(square_side * square_side) / (rect_width * rect_length)⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_in_square_l1600_160001


namespace NUMINAMATH_CALUDE_race_average_time_l1600_160058

theorem race_average_time (fastest_time last_three_avg : ℝ) 
  (h1 : fastest_time = 15)
  (h2 : last_three_avg = 35) : 
  (fastest_time + 3 * last_three_avg) / 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_race_average_time_l1600_160058


namespace NUMINAMATH_CALUDE_lottery_probability_l1600_160071

theorem lottery_probability (total_tickets winning_tickets people : ℕ) 
  (h1 : total_tickets = 10)
  (h2 : winning_tickets = 3)
  (h3 : people = 5) :
  let non_winning_tickets := total_tickets - winning_tickets
  1 - (Nat.choose non_winning_tickets people / Nat.choose total_tickets people : ℚ) = 11/12 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l1600_160071


namespace NUMINAMATH_CALUDE_symmetric_line_correct_l1600_160026

/-- Given a line with equation ax + by + c = 0, 
    returns the equation of the line symmetric to it with respect to the origin -/
def symmetric_line (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, -c)

theorem symmetric_line_correct (a b c : ℝ) :
  let (a', b', c') := symmetric_line a b c
  ∀ x y : ℝ, (a * x + b * y + c = 0) ↔ (a' * (-x) + b' * (-y) + c' = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_correct_l1600_160026


namespace NUMINAMATH_CALUDE_deck_width_l1600_160023

/-- Given a rectangular deck with the following properties:
  * length is 30 feet
  * total cost per square foot (including construction and sealant) is $4
  * total payment is $4800
  prove that the width of the deck is 40 feet -/
theorem deck_width (length : ℝ) (cost_per_sqft : ℝ) (total_cost : ℝ) :
  length = 30 →
  cost_per_sqft = 4 →
  total_cost = 4800 →
  (length * (total_cost / cost_per_sqft)) / length = 40 := by
  sorry

end NUMINAMATH_CALUDE_deck_width_l1600_160023


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l1600_160074

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 4

/-- The total number of emails Jack received in the day -/
def total_emails : ℕ := 5

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := total_emails - morning_emails

theorem jack_afternoon_emails :
  afternoon_emails = 1 :=
sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l1600_160074


namespace NUMINAMATH_CALUDE_red_blood_cell_diameter_scientific_notation_l1600_160060

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem red_blood_cell_diameter_scientific_notation :
  toScientificNotation 0.0000077 = ScientificNotation.mk 7.7 (-6) (by sorry) :=
sorry

end NUMINAMATH_CALUDE_red_blood_cell_diameter_scientific_notation_l1600_160060


namespace NUMINAMATH_CALUDE_rug_area_l1600_160099

/-- Calculates the area of a rug on a rectangular floor with uncovered strips along the edges -/
theorem rug_area (floor_length floor_width strip_width : ℝ) 
  (h_floor_length : floor_length = 10)
  (h_floor_width : floor_width = 8)
  (h_strip_width : strip_width = 2)
  (h_positive_length : floor_length > 0)
  (h_positive_width : floor_width > 0)
  (h_positive_strip : strip_width > 0)
  (h_strip_fits : 2 * strip_width < floor_length ∧ 2 * strip_width < floor_width) :
  (floor_length - 2 * strip_width) * (floor_width - 2 * strip_width) = 24 := by
sorry

end NUMINAMATH_CALUDE_rug_area_l1600_160099


namespace NUMINAMATH_CALUDE_triangle_area_range_l1600_160028

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -Real.log x
  else if x > 1 then Real.log x
  else 0  -- undefined for x ≤ 0 and x = 1

-- Define the derivative of f(x)
noncomputable def f_deriv (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -1/x
  else if x > 1 then 1/x
  else 0  -- undefined for x ≤ 0 and x = 1

-- Theorem statement
theorem triangle_area_range (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁ ∧ x₁ < 1) (h₂ : x₂ > 1) 
  (h_perp : f_deriv x₁ * f_deriv x₂ = -1) :
  let y₁ := f x₁
  let y₂ := f x₂
  let m₁ := f_deriv x₁
  let m₂ := f_deriv x₂
  let x_int := (y₂ - y₁ + m₁*x₁ - m₂*x₂) / (m₁ - m₂)
  let area := abs ((1 - Real.log x₁ - (-1 + Real.log x₂)) * x_int / 2)
  0 < area ∧ area < 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_range_l1600_160028


namespace NUMINAMATH_CALUDE_segment_endpoint_l1600_160056

/-- Given a line segment from (1, 3) to (x, 7) with length 15 and x < 0, prove x = 1 - √209 -/
theorem segment_endpoint (x : ℝ) : 
  x < 0 → 
  Real.sqrt ((1 - x)^2 + (3 - 7)^2) = 15 → 
  x = 1 - Real.sqrt 209 := by
sorry

end NUMINAMATH_CALUDE_segment_endpoint_l1600_160056


namespace NUMINAMATH_CALUDE_opposite_solutions_imply_a_equals_one_l1600_160012

theorem opposite_solutions_imply_a_equals_one (x y a : ℝ) :
  (x + 3 * y = 4 - a) →
  (x - y = -3 * a) →
  (x = -y) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_imply_a_equals_one_l1600_160012


namespace NUMINAMATH_CALUDE_length_width_difference_l1600_160052

/-- A rectangular hall with width being half of its length and area of 128 sq. m -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  width_half_length : width = length / 2
  area_128 : length * width = 128

/-- The difference between length and width of the hall is 8 meters -/
theorem length_width_difference (hall : RectangularHall) : hall.length - hall.width = 8 := by
  sorry

end NUMINAMATH_CALUDE_length_width_difference_l1600_160052


namespace NUMINAMATH_CALUDE_lcm_is_perfect_square_l1600_160039

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a*b) % (a*b*(a - b)) = 0) :
  Nat.lcm a b = (Nat.gcd a b)^2 := by
  sorry

end NUMINAMATH_CALUDE_lcm_is_perfect_square_l1600_160039


namespace NUMINAMATH_CALUDE_midpoint_translated_triangle_l1600_160091

/-- Given triangle BIG with vertices B(0, 0), I(3, 3), and G(6, 0),
    translated 3 units left and 4 units up to form triangle B'I'G',
    the midpoint of segment B'G' is (0, 4). -/
theorem midpoint_translated_triangle (B I G B' I' G' : ℝ × ℝ) :
  B = (0, 0) →
  I = (3, 3) →
  G = (6, 0) →
  B' = (B.1 - 3, B.2 + 4) →
  I' = (I.1 - 3, I.2 + 4) →
  G' = (G.1 - 3, G.2 + 4) →
  ((B'.1 + G'.1) / 2, (B'.2 + G'.2) / 2) = (0, 4) := by
sorry

end NUMINAMATH_CALUDE_midpoint_translated_triangle_l1600_160091


namespace NUMINAMATH_CALUDE_age_problem_l1600_160092

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →  -- A is two years older than B
  b = 2 * c →  -- B is twice as old as C
  a + b + c = 47 →  -- Total age of A, B, and C is 47
  b = 18 :=  -- B's age is 18
by sorry

end NUMINAMATH_CALUDE_age_problem_l1600_160092


namespace NUMINAMATH_CALUDE_max_value_of_function_l1600_160050

theorem max_value_of_function (x : ℝ) (h : x < 5/4) :
  ∃ (max_y : ℝ), max_y = 1 ∧ ∀ y, y = 4*x - 2 + 1/(4*x - 5) → y ≤ max_y :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1600_160050


namespace NUMINAMATH_CALUDE_compound_interest_rate_proof_l1600_160070

/-- Proves that the given conditions result in the specified annual interest rate -/
theorem compound_interest_rate_proof 
  (principal : ℝ) 
  (time : ℝ) 
  (compounding_frequency : ℝ) 
  (compound_interest : ℝ) 
  (h1 : principal = 50000)
  (h2 : time = 2)
  (h3 : compounding_frequency = 2)
  (h4 : compound_interest = 4121.608)
  : ∃ (rate : ℝ), 
    (abs (rate - 0.0398) < 0.0001) ∧ 
    (principal * (1 + rate / compounding_frequency) ^ (compounding_frequency * time) = 
     principal + compound_interest) :=
by sorry


end NUMINAMATH_CALUDE_compound_interest_rate_proof_l1600_160070


namespace NUMINAMATH_CALUDE_least_odd_integer_given_mean_l1600_160054

theorem least_odd_integer_given_mean (integers : List Int) : 
  integers.length = 10 ∧ 
  (∀ i ∈ integers, i % 2 = 1) ∧ 
  (∀ i j, i ∈ integers → j ∈ integers → i ≠ j → |i - j| % 2 = 0) ∧
  (integers.sum / integers.length : ℚ) = 154 →
  integers.minimum? = some 144 := by
sorry

end NUMINAMATH_CALUDE_least_odd_integer_given_mean_l1600_160054


namespace NUMINAMATH_CALUDE_ways_to_walk_teaching_building_l1600_160043

/-- Represents a building with a given number of floors and staircases per floor -/
structure Building where
  floors : Nat
  staircases_per_floor : Nat

/-- Calculates the number of ways to walk from the first floor to the top floor -/
def ways_to_walk (b : Building) : Nat :=
  b.staircases_per_floor ^ (b.floors - 1)

/-- The teaching building -/
def teaching_building : Building :=
  { floors := 4, staircases_per_floor := 2 }

theorem ways_to_walk_teaching_building :
  ways_to_walk teaching_building = 2^3 := by
  sorry

#eval ways_to_walk teaching_building

end NUMINAMATH_CALUDE_ways_to_walk_teaching_building_l1600_160043


namespace NUMINAMATH_CALUDE_keith_stored_bales_l1600_160063

/-- The number of bales Keith stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Theorem: Keith stored 67 bales in the barn -/
theorem keith_stored_bales :
  bales_stored 22 89 = 67 := by
  sorry

end NUMINAMATH_CALUDE_keith_stored_bales_l1600_160063


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1600_160021

/-- A polynomial p(x) = x^2 + mx + 9 is a perfect square trinomial if and only if
    there exists a real number a such that p(x) = (x + a)^2 for all x. -/
def IsPerfectSquareTrinomial (m : ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 9 = (x + a)^2

/-- If x^2 + mx + 9 is a perfect square trinomial, then m = 6 or m = -6. -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  IsPerfectSquareTrinomial m → m = 6 ∨ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1600_160021


namespace NUMINAMATH_CALUDE_sum_of_squares_l1600_160040

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_seven_eq : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1600_160040


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1600_160061

-- Define the cubic polynomial
def cubic_poly (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem cubic_root_sum (a b c d : ℝ) : 
  a ≠ 0 → 
  cubic_poly a b c d 4 = 0 →
  cubic_poly a b c d (-3) = 0 →
  (b + c) / a = -13 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1600_160061


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l1600_160084

theorem quadratic_one_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x, 3 * x^2 + b₁ * x + 5 * x + 7 = 0 → (∀ y, 3 * y^2 + b₁ * y + 5 * y + 7 = 0 → x = y)) ∧
  (∀ x, 3 * x^2 + b₂ * x + 5 * x + 7 = 0 → (∀ y, 3 * y^2 + b₂ * y + 5 * y + 7 = 0 → x = y)) ∧
  (∀ b, (∀ x, 3 * x^2 + b * x + 5 * x + 7 = 0 → (∀ y, 3 * y^2 + b * y + 5 * y + 7 = 0 → x = y)) → b = b₁ ∨ b = b₂) →
  b₁ + b₂ = -10 :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l1600_160084


namespace NUMINAMATH_CALUDE_x_value_proof_l1600_160024

theorem x_value_proof (x : ℚ) (h : 2/3 - 1/4 = 4/x) : x = 48/5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1600_160024


namespace NUMINAMATH_CALUDE_third_fraction_numerator_l1600_160009

/-- Given three fractions where the sum is 3.0035428163476343,
    the first fraction is 2007/2999, the second is 8001/5998,
    and the third has a denominator of 3999,
    prove that the numerator of the third fraction is 4002. -/
theorem third_fraction_numerator :
  let sum : ℚ := 3.0035428163476343
  let frac1 : ℚ := 2007 / 2999
  let frac2 : ℚ := 8001 / 5998
  let denom3 : ℕ := 3999
  ∃ (num3 : ℕ), (frac1 + frac2 + (num3 : ℚ) / denom3 = sum) ∧ num3 = 4002 := by
  sorry


end NUMINAMATH_CALUDE_third_fraction_numerator_l1600_160009


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1600_160044

theorem rectangular_box_volume (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (∃ (k : ℕ), k > 0 ∧ a = 2 * k ∧ b = 3 * k ∧ c = 5 * k) →
  a * b * c = 240 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1600_160044


namespace NUMINAMATH_CALUDE_bacon_to_eggs_ratio_l1600_160011

/-- Represents a breakfast plate with eggs and bacon strips -/
structure BreakfastPlate where
  eggs : ℕ
  bacon : ℕ

/-- Represents the cafe's breakfast order -/
structure CafeOrder where
  plates : ℕ
  totalBacon : ℕ

theorem bacon_to_eggs_ratio (order : CafeOrder) (plate : BreakfastPlate) :
  order.plates = 14 →
  order.totalBacon = 56 →
  plate.eggs = 2 →
  (order.totalBacon / order.plates : ℚ) / plate.eggs = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bacon_to_eggs_ratio_l1600_160011


namespace NUMINAMATH_CALUDE_correct_average_l1600_160089

theorem correct_average (n : ℕ) (initial_avg incorrect_num correct_num : ℚ) : 
  n = 10 → 
  initial_avg = 16 → 
  incorrect_num = 26 → 
  correct_num = 46 → 
  (n * initial_avg - incorrect_num + correct_num) / n = 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l1600_160089


namespace NUMINAMATH_CALUDE_fraction_sum_integer_l1600_160068

theorem fraction_sum_integer (n : ℕ) (h1 : n > 0) 
  (h2 : ∃ k : ℤ, (1 : ℚ) / 2 + 1 / 3 + 1 / 5 + 1 / n = k) : n = 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_l1600_160068


namespace NUMINAMATH_CALUDE_steves_gum_pieces_l1600_160090

/-- Given Todd's initial and final number of gum pieces, prove that the number of gum pieces
    Steve gave Todd is equal to the difference between the final and initial numbers. -/
theorem steves_gum_pieces (todd_initial todd_final steve_gave : ℕ) 
    (h1 : todd_initial = 38)
    (h2 : todd_final = 54)
    (h3 : todd_final = todd_initial + steve_gave) :
  steve_gave = todd_final - todd_initial := by
  sorry

end NUMINAMATH_CALUDE_steves_gum_pieces_l1600_160090


namespace NUMINAMATH_CALUDE_haleigh_leggings_count_l1600_160067

/-- The number of dogs Haleigh has -/
def num_dogs : ℕ := 4

/-- The number of cats Haleigh has -/
def num_cats : ℕ := 3

/-- The number of legs each animal (dog or cat) has -/
def legs_per_animal : ℕ := 4

/-- The number of legs covered by one pair of leggings -/
def legs_per_legging : ℕ := 2

/-- The total number of pairs of leggings needed for Haleigh's pets -/
def total_leggings : ℕ := 
  (num_dogs * legs_per_animal + num_cats * legs_per_animal) / legs_per_legging

theorem haleigh_leggings_count : total_leggings = 14 := by
  sorry

end NUMINAMATH_CALUDE_haleigh_leggings_count_l1600_160067


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_l1600_160057

theorem cubic_sum_of_roots (a b r s : ℝ) : 
  (r^2 - a*r + b = 0) → (s^2 - a*s + b = 0) → (r^3 + s^3 = a^3 - 3*a*b) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_l1600_160057


namespace NUMINAMATH_CALUDE_gcd_problem_l1600_160006

def gcd_operation (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_problem :
  gcd_operation (gcd_operation (gcd_operation 20 16) (gcd_operation 18 24)) 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1600_160006


namespace NUMINAMATH_CALUDE_callum_max_score_l1600_160059

/-- Calculates the score for n consecutive wins with a base score and multiplier -/
def consecutiveWinScore (baseScore : ℕ) (n : ℕ) : ℕ :=
  baseScore * 2^(n - 1)

/-- Calculates the total score for a given number of wins -/
def totalScore (wins : ℕ) : ℕ :=
  (List.range wins).map (consecutiveWinScore 10) |> List.sum

theorem callum_max_score (totalMatches : ℕ) (krishnaWins : ℕ) 
    (h1 : totalMatches = 12)
    (h2 : krishnaWins = 2 * totalMatches / 3)
    (h3 : krishnaWins < totalMatches) : 
  totalScore (totalMatches - krishnaWins) = 150 := by
  sorry

end NUMINAMATH_CALUDE_callum_max_score_l1600_160059


namespace NUMINAMATH_CALUDE_max_coefficient_of_expansion_l1600_160087

theorem max_coefficient_of_expansion : 
  ∃ (a b c d : ℕ+), 
    (∀ x : ℝ, (6 * x + 3)^3 = a * x^3 + b * x^2 + c * x + d) ∧ 
    (max a (max b (max c d)) = 324) := by
  sorry

end NUMINAMATH_CALUDE_max_coefficient_of_expansion_l1600_160087


namespace NUMINAMATH_CALUDE_solve_worker_problem_l1600_160005

/-- Represents the work rate of one person -/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers -/
structure WorkerGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work rate of a group -/
def totalWorkRate (g : WorkerGroup) (m w : WorkRate) : ℝ :=
  (g.men : ℝ) * m.rate + (g.women : ℝ) * w.rate

theorem solve_worker_problem (m w : WorkRate) : ∃ x : ℕ,
  let group1 := WorkerGroup.mk 3 8
  let group2 := WorkerGroup.mk x 2
  let group3 := WorkerGroup.mk 3 2
  totalWorkRate group1 m w = totalWorkRate group2 m w ∧
  totalWorkRate group3 m w = (4/7 : ℝ) * totalWorkRate group1 m w ∧
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_worker_problem_l1600_160005
