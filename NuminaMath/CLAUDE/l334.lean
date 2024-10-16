import Mathlib

namespace NUMINAMATH_CALUDE_purely_imaginary_modulus_l334_33459

theorem purely_imaginary_modulus (a : ℝ) :
  (a - 2 : ℂ) + a * I = (0 : ℂ) + (a * I) → Complex.abs (a + I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_modulus_l334_33459


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l334_33413

/-- Represents a rectangular plot with a given breadth and area -/
structure RectangularPlot where
  breadth : ℝ
  area : ℝ
  length_is_thrice_breadth : length = 3 * breadth
  area_formula : area = length * breadth

/-- The breadth of a rectangular plot with thrice length and 2700 sq m area is 30 m -/
theorem rectangular_plot_breadth (plot : RectangularPlot) 
  (h_area : plot.area = 2700) : plot.breadth = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l334_33413


namespace NUMINAMATH_CALUDE_min_abs_z_l334_33401

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 5*Complex.I) + Complex.abs (z - 7) = 10) :
  ∃ (w : ℂ), Complex.abs (z - 5*Complex.I) + Complex.abs (z - 7) = 10 → Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 35 / Real.sqrt 74 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_l334_33401


namespace NUMINAMATH_CALUDE_fraction_equality_l334_33445

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x - 2 * y) / (2 * x + 6 * y) = 3) : 
  (2 * x - 6 * y) / (4 * x + y) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l334_33445


namespace NUMINAMATH_CALUDE_rain_free_paths_l334_33436

/-- The function f representing the amount of rain at point (x,y) -/
def f (x y : ℝ) : ℝ := |x^3 + 2*x^2*y - 5*x*y^2 - 6*y^3|

/-- The theorem stating that the set of m values for which f(x,mx) = 0 for all x
    is exactly {-1, 1/2, -1/3} -/
theorem rain_free_paths (x : ℝ) :
  {m : ℝ | ∀ x, f x (m*x) = 0} = {-1, 1/2, -1/3} := by
  sorry

end NUMINAMATH_CALUDE_rain_free_paths_l334_33436


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l334_33488

def P (x : ℝ) : ℝ := 5*x^6 - 3*x^5 + 4*x^4 - x^3 + 6*x^2 - 5*x + 7

theorem polynomial_remainder_theorem :
  ∃ (Q : ℝ → ℝ), P = λ x ↦ (x - 3) * Q x + 3259 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l334_33488


namespace NUMINAMATH_CALUDE_rectangles_bounded_by_lines_l334_33492

/-- The number of rectangles bounded by p parallel lines and q perpendicular lines -/
def num_rectangles (p q : ℕ) : ℚ :=
  (p * q * (p - 1) * (q - 1)) / 4

/-- Theorem stating the number of rectangles bounded by p parallel lines and q perpendicular lines -/
theorem rectangles_bounded_by_lines (p q : ℕ) :
  num_rectangles p q = (p * q * (p - 1) * (q - 1)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_bounded_by_lines_l334_33492


namespace NUMINAMATH_CALUDE_two_dice_prime_probability_l334_33489

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def dice_outcomes (n : ℕ) : ℕ := 6^n

def prime_outcomes (n : ℕ) : ℕ := 
  if n = 2 then 15 else 0  -- We only define it for 2 dice as per the problem

theorem two_dice_prime_probability :
  (prime_outcomes 2 : ℚ) / (dice_outcomes 2 : ℚ) = 5/12 :=
sorry

end NUMINAMATH_CALUDE_two_dice_prime_probability_l334_33489


namespace NUMINAMATH_CALUDE_tshirt_sale_ratio_l334_33426

/-- Prove that the ratio of black shirts to white shirts is 1:1 given the conditions -/
theorem tshirt_sale_ratio :
  ∀ (black white : ℕ),
  black + white = 200 →
  30 * black + 25 * white = 5500 →
  black = white :=
by sorry

end NUMINAMATH_CALUDE_tshirt_sale_ratio_l334_33426


namespace NUMINAMATH_CALUDE_correct_linear_system_l334_33414

-- Define a structure for a system of two equations
structure EquationSystem where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ

-- Define the four systems of equations
def systemA : EquationSystem := {
  eq1 := fun x y => x + 5*y - 2,
  eq2 := fun x y => x*y - 7
}

def systemB : EquationSystem := {
  eq1 := fun x y => 2*x + 1 - 1,
  eq2 := fun x y => 3*x + 4*y
}

def systemC : EquationSystem := {
  eq1 := fun x y => 3*x^2 - 5*y,
  eq2 := fun x y => x + y - 4
}

def systemD : EquationSystem := {
  eq1 := fun x y => x - 2*y - 8,
  eq2 := fun x y => x + 3*y - 12
}

-- Define a predicate for linear equations with two variables
def isLinearSystem (s : EquationSystem) : Prop :=
  ∃ a b c d e f : ℝ, 
    (∀ x y, s.eq1 x y = a*x + b*y + c) ∧
    (∀ x y, s.eq2 x y = d*x + e*y + f)

-- Theorem statement
theorem correct_linear_system : 
  ¬(isLinearSystem systemA) ∧ 
  ¬(isLinearSystem systemB) ∧ 
  ¬(isLinearSystem systemC) ∧ 
  isLinearSystem systemD := by
  sorry

end NUMINAMATH_CALUDE_correct_linear_system_l334_33414


namespace NUMINAMATH_CALUDE_stating_chess_tournament_players_l334_33422

/-- The number of players in a chess tournament -/
def num_players : ℕ := 17

/-- The total number of games played in the tournament -/
def total_games : ℕ := 272

/-- 
Theorem stating that the number of players in the chess tournament is correct,
given the conditions of the problem.
-/
theorem chess_tournament_players :
  (2 * num_players * (num_players - 1) = total_games) ∧ 
  (∀ n : ℕ, 2 * n * (n - 1) = total_games → n = num_players) := by
  sorry

#check chess_tournament_players

end NUMINAMATH_CALUDE_stating_chess_tournament_players_l334_33422


namespace NUMINAMATH_CALUDE_friend_walking_problem_l334_33498

/-- Two friends walking on a trail problem -/
theorem friend_walking_problem (trail_length : ℝ) (meeting_distance : ℝ) 
  (h1 : trail_length = 43)
  (h2 : meeting_distance = 23)
  (h3 : meeting_distance < trail_length) :
  let rate_ratio := meeting_distance / (trail_length - meeting_distance)
  (rate_ratio - 1) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_friend_walking_problem_l334_33498


namespace NUMINAMATH_CALUDE_ellipse_equation_l334_33446

/-- An ellipse with a line passing through its vertex and focus -/
structure EllipseWithLine where
  /-- The semi-major axis length of the ellipse -/
  a : ℝ
  /-- The semi-minor axis length of the ellipse -/
  b : ℝ
  /-- Condition that a > b > 0 -/
  h1 : a > b ∧ b > 0
  /-- The line equation x - 2y + 4 = 0 -/
  line_eq : ℝ → ℝ → Prop := fun x y => x - 2*y + 4 = 0
  /-- The line passes through a vertex and focus of the ellipse -/
  line_through_vertex_focus : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧
    ((x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧ (x₁ = a ∨ x₁ = -a ∨ y₁ = b ∨ y₁ = -b)) ∨
     (x₂^2 / a^2 + y₂^2 / b^2 > 1 ∧ x₂^2 - y₂^2 = a^2 - b^2))

/-- The theorem stating the standard equation of the ellipse -/
theorem ellipse_equation (e : EllipseWithLine) :
  ∀ (x y : ℝ), x^2/20 + y^2/4 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l334_33446


namespace NUMINAMATH_CALUDE_coffee_consumption_l334_33463

theorem coffee_consumption (people : ℕ) (coffee_per_cup : ℚ) (coffee_cost : ℚ) (weekly_spend : ℚ) :
  people = 4 →
  coffee_per_cup = 1/2 →
  coffee_cost = 5/4 →
  weekly_spend = 35 →
  (weekly_spend / coffee_cost / coffee_per_cup / people / 7 : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_coffee_consumption_l334_33463


namespace NUMINAMATH_CALUDE_small_circle_radius_l334_33495

/-- Given a configuration of circles where:
    - There is one large circle with radius 10 meters
    - There are six congruent smaller circles
    - The smaller circles are aligned in a straight line
    - The smaller circles touch each other and the perimeter of the larger circle
    This theorem proves that the radius of each smaller circle is 5/3 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) (h1 : R = 10) (h2 : 6 * (2 * r) = 2 * R) :
  r = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_small_circle_radius_l334_33495


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l334_33443

theorem cylinder_volume_equality (r h x : ℝ) : 
  r = 5 ∧ h = 7 ∧ x > 0 ∧ 
  π * (2 * r + x)^2 * h = π * r^2 * (3 * h + x) → 
  x = (5 + Real.sqrt 9125) / 14 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l334_33443


namespace NUMINAMATH_CALUDE_equation_solution_l334_33485

theorem equation_solution : ∃ x : ℚ, x + 5/8 = 7/24 + 1/4 ∧ x = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l334_33485


namespace NUMINAMATH_CALUDE_profit_at_selling_price_185_l334_33432

/-- Represents the daily sales volume as a function of price reduction --/
def sales_volume (x : ℝ) : ℝ := 4 * x + 100

/-- Represents the selling price as a function of price reduction --/
def selling_price (x : ℝ) : ℝ := 200 - x

/-- Represents the daily profit as a function of price reduction --/
def daily_profit (x : ℝ) : ℝ := (selling_price x - 100) * sales_volume x

theorem profit_at_selling_price_185 :
  ∃ x : ℝ, 
    daily_profit x = 13600 ∧ 
    selling_price x = 185 ∧ 
    selling_price x ≥ 150 := by sorry

end NUMINAMATH_CALUDE_profit_at_selling_price_185_l334_33432


namespace NUMINAMATH_CALUDE_constant_function_proof_l334_33408

theorem constant_function_proof (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x * y) + f (x * z) ≥ 1 + f x * f (y * z)) : 
  ∀ x : ℝ, f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l334_33408


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l334_33407

-- Define the sets A and B
def A : Set ℝ := {x | x - 3 > 0}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l334_33407


namespace NUMINAMATH_CALUDE_factorial_ratio_l334_33457

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_ratio : factorial 4 / factorial (4 - 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l334_33457


namespace NUMINAMATH_CALUDE_crescent_moon_division_l334_33428

/-- The maximum number of parts a crescent moon can be divided into with n straight cuts -/
def max_parts (n : ℕ) : ℕ := (n^2 + 3*n) / 2 + 1

/-- The number of straight cuts used -/
def num_cuts : ℕ := 5

theorem crescent_moon_division :
  max_parts num_cuts = 21 :=
sorry

end NUMINAMATH_CALUDE_crescent_moon_division_l334_33428


namespace NUMINAMATH_CALUDE_count_triples_eq_12_l334_33430

/-- Least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given conditions -/
def count_triples : ℕ := sorry

theorem count_triples_eq_12 :
  count_triples = 12 := by sorry

end NUMINAMATH_CALUDE_count_triples_eq_12_l334_33430


namespace NUMINAMATH_CALUDE_half_of_large_number_l334_33482

theorem half_of_large_number : (1.2 * 10^30) / 2 = 6.0 * 10^29 := by
  sorry

end NUMINAMATH_CALUDE_half_of_large_number_l334_33482


namespace NUMINAMATH_CALUDE_function_composition_ratio_l334_33405

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l334_33405


namespace NUMINAMATH_CALUDE_roots_of_h_l334_33468

/-- Given that x = 1 is a root of f(x) = a/x + b and a ≠ 0, 
    prove that the roots of h(x) = ax^2 + bx are 0 and 1. -/
theorem roots_of_h (a b : ℝ) (ha : a ≠ 0) 
  (hf : a / 1 + b = 0) : 
  ∀ x : ℝ, ax^2 + bx = 0 ↔ x = 0 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_of_h_l334_33468


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l334_33481

/-- Definition of the quadratic function g(x) -/
def g (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating that under the given conditions, k = 0 -/
theorem quadratic_function_k_value
  (a b c : ℤ)
  (h1 : g a b c 2 = 0)
  (h2 : 60 < g a b c 6 ∧ g a b c 6 < 70)
  (h3 : 90 < g a b c 9 ∧ g a b c 9 < 100)
  (k : ℤ)
  (h4 : 10000 * ↑k < g a b c 50 ∧ g a b c 50 < 10000 * ↑(k + 1)) :
  k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l334_33481


namespace NUMINAMATH_CALUDE_base_8_properties_l334_33448

-- Define the base 10 number
def base_10_num : ℕ := 9257

-- Define the base 8 representation as a list of digits
def base_8_rep : List ℕ := [2, 2, 0, 5, 1]

-- Theorem stating the properties we want to prove
theorem base_8_properties :
  -- The base 8 representation is correct
  (List.foldl (λ acc d => acc * 8 + d) 0 base_8_rep = base_10_num) ∧
  -- The product of the digits is 0
  (List.foldl (· * ·) 1 base_8_rep = 0) ∧
  -- The sum of the digits is 10
  (List.sum base_8_rep = 10) := by
  sorry

end NUMINAMATH_CALUDE_base_8_properties_l334_33448


namespace NUMINAMATH_CALUDE_fraction_inequality_l334_33424

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) : 
  c / a - d / b > 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l334_33424


namespace NUMINAMATH_CALUDE_part_one_part_two_l334_33476

-- Define the solution set M
def M (a : ℝ) := {x : ℝ | a * x^2 + 5 * x - 2 > 0}

-- Part 1
theorem part_one (a : ℝ) : 2 ∈ M a → a > -2 := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  M a = {x : ℝ | 1/2 < x ∧ x < 2} →
  {x : ℝ | a * x^2 - 5 * x + a^2 - 1 > 0} = {x : ℝ | -3 < x ∧ x < 1/2} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l334_33476


namespace NUMINAMATH_CALUDE_smallest_scalene_triangle_with_prime_perimeter_l334_33452

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧
  b = a + 2 ∧ c = b + 2

theorem smallest_scalene_triangle_with_prime_perimeter :
  ∃ (a b c : ℕ),
    areConsecutiveOddPrimes a b c ∧
    isValidTriangle a b c ∧
    isPrime (a + b + c) ∧
    a + b + c = 23 ∧
    (∀ (x y z : ℕ),
      areConsecutiveOddPrimes x y z →
      isValidTriangle x y z →
      isPrime (x + y + z) →
      x + y + z ≥ 23) :=
sorry

end NUMINAMATH_CALUDE_smallest_scalene_triangle_with_prime_perimeter_l334_33452


namespace NUMINAMATH_CALUDE_number_of_workers_l334_33470

/-- Proves that the number of men working on the jobs is 3 --/
theorem number_of_workers (time_per_job : ℝ) (num_jobs : ℕ) (hourly_rate : ℝ) (total_earned : ℝ) : ℕ :=
  by
  -- Assume the given conditions
  have h1 : time_per_job = 1 := by sorry
  have h2 : num_jobs = 5 := by sorry
  have h3 : hourly_rate = 10 := by sorry
  have h4 : total_earned = 150 := by sorry

  -- Define the number of workers
  let num_workers : ℕ := 3

  -- Prove that num_workers satisfies the conditions
  have h5 : (↑num_workers : ℝ) * num_jobs * hourly_rate = total_earned := by sorry

  -- Return the number of workers
  exact num_workers

end NUMINAMATH_CALUDE_number_of_workers_l334_33470


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l334_33494

/-- The difference between the larger and smaller x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference : ∃ (a c : ℝ),
  (∀ x : ℝ, 3 * x^2 - 6 * x + 6 = -x^2 - 4 * x + 6 → (x = a ∨ x = c)) ∧
  c ≥ a ∧
  c - a = (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l334_33494


namespace NUMINAMATH_CALUDE_blueberries_count_l334_33438

/-- Represents the number of blueberries in each blue box -/
def blueberries : ℕ := sorry

/-- Represents the number of strawberries in each red box -/
def strawberries : ℕ := sorry

/-- The increase in total berries when replacing a blue box with a red box -/
def total_increase : ℕ := 20

/-- The increase in difference between strawberries and blueberries after replacement -/
def difference_increase : ℕ := 80

theorem blueberries_count : blueberries = 60 :=
  by sorry

end NUMINAMATH_CALUDE_blueberries_count_l334_33438


namespace NUMINAMATH_CALUDE_smallest_with_18_divisors_l334_33499

/-- Count the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number has exactly 18 positive divisors -/
def has18Divisors (n : ℕ) : Prop := countDivisors n = 18

/-- The smallest positive integer with exactly 18 positive divisors -/
def smallestWith18Divisors : ℕ := 288

theorem smallest_with_18_divisors :
  (has18Divisors smallestWith18Divisors) ∧
  (∀ m : ℕ, m < smallestWith18Divisors → ¬(has18Divisors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_18_divisors_l334_33499


namespace NUMINAMATH_CALUDE_vector_magnitude_l334_33417

def unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem vector_magnitude (e₁ e₂ : ℝ × ℝ) (h₁ : unit_vector e₁) (h₂ : unit_vector e₂)
  (h₃ : e₁.1 * e₂.1 + e₁.2 * e₂.2 = 1/2) : 
  let a := (2 * e₁.1 + e₂.1, 2 * e₁.2 + e₂.2)
  (a.1^2 + a.2^2) = 7 := by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l334_33417


namespace NUMINAMATH_CALUDE_complex_equation_implication_l334_33441

theorem complex_equation_implication (a b : ℝ) :
  let z : ℂ := a + b * Complex.I
  (z * (z + 2 * Complex.I) * (z + 4 * Complex.I) = 5000 * Complex.I) →
  (a^3 - a * (b^2 + 6*b + 8) - (b+6) * (b^2 + 6*b + 8) = 0 ∧
   a * (b+6) - b * (b^2 + 6*b + 8) = 5000) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_implication_l334_33441


namespace NUMINAMATH_CALUDE_inequality_equivalence_l334_33425

theorem inequality_equivalence (x : ℝ) :
  (x + 1) / (x - 5) ≥ 3 ↔ x ≥ 8 ∧ x ≠ 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l334_33425


namespace NUMINAMATH_CALUDE_all_sides_equal_l334_33406

/-- A convex n-gon with equal interior angles and ordered sides -/
structure ConvexNGon (n : ℕ) where
  -- The sides of the n-gon
  sides : Fin n → ℝ
  -- All sides are non-negative
  sides_nonneg : ∀ i, 0 ≤ sides i
  -- The sides are ordered in descending order
  sides_ordered : ∀ i j, i ≤ j → sides j ≤ sides i
  -- The n-gon is convex
  convex : True
  -- All interior angles are equal
  equal_angles : True

/-- Theorem: In a convex n-gon with equal interior angles and ordered sides, all sides are equal -/
theorem all_sides_equal (n : ℕ) (ngon : ConvexNGon n) :
  ∀ i j : Fin n, ngon.sides i = ngon.sides j :=
sorry

end NUMINAMATH_CALUDE_all_sides_equal_l334_33406


namespace NUMINAMATH_CALUDE_set_equality_implies_y_zero_l334_33487

theorem set_equality_implies_y_zero (x y : ℝ) :
  ({0, 1, x} : Set ℝ) = {x^2, y, -1} → y = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_y_zero_l334_33487


namespace NUMINAMATH_CALUDE_sqrt_10_parts_product_l334_33440

theorem sqrt_10_parts_product (x y : ℝ) : 
  (x = ⌊Real.sqrt 10⌋) → 
  (y = Real.sqrt 10 - ⌊Real.sqrt 10⌋) → 
  y * (x + Real.sqrt 10) = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_10_parts_product_l334_33440


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l334_33444

theorem area_between_concentric_circles :
  ∀ (r₁ r₂ : ℝ),
  r₁ > 0 →
  r₂ = 5 * r₁ →
  2 * r₁ = 4 →
  (π * r₂^2 - π * r₁^2) = 96 * π :=
by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l334_33444


namespace NUMINAMATH_CALUDE_cricket_game_target_runs_l334_33412

/-- Calculates the target number of runs in a cricket game -/
def targetRuns (totalOvers runRateFirst10 runRateRemaining : ℝ) : ℝ :=
  10 * runRateFirst10 + (totalOvers - 10) * runRateRemaining

/-- Theorem stating the target number of runs in the given cricket game -/
theorem cricket_game_target_runs :
  targetRuns 50 6.2 5.5 = 282 := by
  sorry

end NUMINAMATH_CALUDE_cricket_game_target_runs_l334_33412


namespace NUMINAMATH_CALUDE_power_three_multiplication_l334_33433

theorem power_three_multiplication : 6^3 * 7^3 = 74088 := by
  sorry

end NUMINAMATH_CALUDE_power_three_multiplication_l334_33433


namespace NUMINAMATH_CALUDE_abc_unique_solution_l334_33420

/-- Represents a base-7 number with two digits --/
def Base7TwoDigit (a b : ℕ) : ℕ := 7 * a + b

/-- Converts a three-digit decimal number to its numeric value --/
def ThreeDigitToNum (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem abc_unique_solution :
  ∀ A B C : ℕ,
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A < 7 → B < 7 → C < 7 →
    A ≠ B → B ≠ C → A ≠ C →
    Base7TwoDigit A B + C = Base7TwoDigit C 0 →
    Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit B 6 →
    ThreeDigitToNum A B C = 425 :=
by sorry

end NUMINAMATH_CALUDE_abc_unique_solution_l334_33420


namespace NUMINAMATH_CALUDE_allison_total_supplies_l334_33450

/-- Represents the number of craft supplies bought by a person -/
structure CraftSupplies where
  glueSticks : ℕ
  constructionPaper : ℕ

/-- The total number of craft supplies -/
def CraftSupplies.total (cs : CraftSupplies) : ℕ :=
  cs.glueSticks + cs.constructionPaper

/-- Given information about Marie's purchases -/
def marie : CraftSupplies :=
  { glueSticks := 15
    constructionPaper := 30 }

/-- Theorem stating the total number of craft supplies Allison bought -/
theorem allison_total_supplies : 
  ∃ (allison : CraftSupplies), 
    (allison.glueSticks = marie.glueSticks + 8) ∧ 
    (allison.constructionPaper * 6 = marie.constructionPaper) ∧ 
    (allison.total = 28) := by
  sorry

end NUMINAMATH_CALUDE_allison_total_supplies_l334_33450


namespace NUMINAMATH_CALUDE_tan_alpha_values_l334_33415

theorem tan_alpha_values (α : Real) :
  5 * Real.sin (2 * α) + 5 * Real.cos (2 * α) + 1 = 0 →
  Real.tan α = 3 ∨ Real.tan α = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l334_33415


namespace NUMINAMATH_CALUDE_side_face_area_l334_33453

/-- A rectangular box with specific proportions and volume -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  front_half_top : length * width = 0.5 * (length * height)
  top_1_5_side : length * height = 1.5 * (width * height)
  volume : length * width * height = 3000

/-- The area of the side face of the box is 200 -/
theorem side_face_area (b : Box) : b.width * b.height = 200 := by
  sorry

end NUMINAMATH_CALUDE_side_face_area_l334_33453


namespace NUMINAMATH_CALUDE_percent_product_theorem_l334_33403

theorem percent_product_theorem :
  let p1 : ℝ := 15
  let p2 : ℝ := 20
  let p3 : ℝ := 25
  (p1 / 100) * (p2 / 100) * (p3 / 100) * 100 = 0.75
  := by sorry

end NUMINAMATH_CALUDE_percent_product_theorem_l334_33403


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l334_33473

theorem percentage_of_sikh_boys (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other : ℕ) :
  total = 400 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  other = 72 →
  (total - (muslim_percent * total).num - (hindu_percent * total).num - other) / total * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l334_33473


namespace NUMINAMATH_CALUDE_age_difference_z_younger_than_x_l334_33467

-- Define variables for ages
variable (X Y Z : ℕ)

-- Define the condition from the problem
def age_condition (X Y Z : ℕ) : Prop := X + Y = Y + Z + 19

-- Theorem to prove
theorem age_difference (h : age_condition X Y Z) : X - Z = 19 :=
by sorry

-- Convert years to decades
def years_to_decades (years : ℕ) : ℚ := (years : ℚ) / 10

-- Theorem to prove the final result
theorem z_younger_than_x (h : age_condition X Y Z) : 
  years_to_decades (X - Z) = 1.9 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_z_younger_than_x_l334_33467


namespace NUMINAMATH_CALUDE_amp_five_two_l334_33421

-- Define the & operation
def amp (a b : ℤ) : ℤ := ((a + b) * (a - b))^2

-- Theorem statement
theorem amp_five_two : amp 5 2 = 441 := by
  sorry

end NUMINAMATH_CALUDE_amp_five_two_l334_33421


namespace NUMINAMATH_CALUDE_function_properties_l334_33437

noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := (x^2 + 1) / (b * x + c)

theorem function_properties (b c : ℝ) :
  (∀ x : ℝ, x ≠ 0 → b * x + c ≠ 0) →
  f b c 1 = 2 →
  (∃ g : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f b c x = g x) ∧ 
                (∀ x : ℝ, x ≠ 0 → g x = x + 1/x) ∧
                (∀ x y : ℝ, 1 ≤ x ∧ x < y → g x < g y) ∧
                (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → g x ≤ 5/2) ∧
                (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → 2 ≤ g x) ∧
                g 2 = 5/2 ∧
                g 1 = 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l334_33437


namespace NUMINAMATH_CALUDE_product_of_diff_of_squares_l334_33410

-- Define the property of being a difference of two squares
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, n = x^2 - y^2 ∧ x > y

-- Theorem statement
theorem product_of_diff_of_squares (a b c d : ℕ) 
  (ha : is_diff_of_squares a)
  (hb : is_diff_of_squares b)
  (hc : is_diff_of_squares c)
  (hd : is_diff_of_squares d)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (pos_d : d > 0) :
  is_diff_of_squares (a * b * c * d) :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_diff_of_squares_l334_33410


namespace NUMINAMATH_CALUDE_car_speed_equality_l334_33465

/-- Proves that given the conditions of the car problem, the average speed of Car Y equals that of Car X -/
theorem car_speed_equality (speed_x : ℝ) (start_delay : ℝ) (distance_after_y_start : ℝ) : 
  speed_x = 35 →
  start_delay = 72 / 60 →
  distance_after_y_start = 210 →
  ∃ (time_after_y_start : ℝ), 
    time_after_y_start > 0 ∧ 
    speed_x * time_after_y_start = distance_after_y_start ∧
    distance_after_y_start / time_after_y_start = speed_x := by
  sorry

#check car_speed_equality

end NUMINAMATH_CALUDE_car_speed_equality_l334_33465


namespace NUMINAMATH_CALUDE_bus_ride_cost_l334_33404

theorem bus_ride_cost (train_cost bus_cost : ℝ) : 
  train_cost = bus_cost + 6.35 →
  train_cost + bus_cost = 9.85 →
  bus_cost = 1.75 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l334_33404


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l334_33442

/-- The x-coordinate of the vertex of a quadratic function f(x) = x^2 + 2px + 3q -/
def vertex_x_coord (p q : ℝ) : ℝ := -p

/-- The quadratic function f(x) = x^2 + 2px + 3q -/
def f (p q x : ℝ) : ℝ := x^2 + 2*p*x + 3*q

theorem vertex_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∀ x : ℝ, f p q x ≥ f p q (vertex_x_coord p q) :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l334_33442


namespace NUMINAMATH_CALUDE_intersection_M_N_l334_33491

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l334_33491


namespace NUMINAMATH_CALUDE_sets_problem_l334_33460

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3)^2 ≤ 0}
def N : Set ℝ := {x | x^2 + x - 6 = 0}

-- Define set A as (complement_I M) ∩ N
def A : Set ℝ := (Set.univ \ M) ∩ N

-- Define set B
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 5 - a}

-- Theorem statement
theorem sets_problem :
  (A = {2}) ∧
  ({a : ℝ | B a ∪ A = A} = {a : ℝ | a ≥ 3}) := by
  sorry

end NUMINAMATH_CALUDE_sets_problem_l334_33460


namespace NUMINAMATH_CALUDE_three_card_draw_probability_l334_33474

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Finset (Fin 52) := Finset.univ

/-- The number of diamonds in a standard deck -/
def numDiamonds : Nat := 13

/-- The number of kings in a standard deck -/
def numKings : Nat := 4

/-- The number of aces in a standard deck -/
def numAces : Nat := 4

/-- The probability of drawing a diamond as the first card, 
    a king as the second card, and an ace as the third card 
    from a standard 52-card deck -/
theorem three_card_draw_probability : 
  (numDiamonds * numKings * numAces : ℚ) / (52 * 51 * 50) = 142 / 66300 := by
  sorry

end NUMINAMATH_CALUDE_three_card_draw_probability_l334_33474


namespace NUMINAMATH_CALUDE_valid_committee_count_l334_33429

/-- Represents the number of male professors in each department -/
def male_profs : Fin 3 → Nat
  | 0 => 3  -- Physics
  | 1 => 2  -- Chemistry
  | 2 => 2  -- Biology

/-- Represents the number of female professors in each department -/
def female_profs : Fin 3 → Nat
  | 0 => 3  -- Physics
  | 1 => 2  -- Chemistry
  | 2 => 3  -- Biology

/-- The total number of departments -/
def num_departments : Nat := 3

/-- The required committee size -/
def committee_size : Nat := 6

/-- The required number of male professors in the committee -/
def required_males : Nat := 3

/-- The required number of female professors in the committee -/
def required_females : Nat := 3

/-- Calculates the number of valid committee formations -/
def count_valid_committees : Nat := sorry

theorem valid_committee_count :
  count_valid_committees = 864 := by sorry

end NUMINAMATH_CALUDE_valid_committee_count_l334_33429


namespace NUMINAMATH_CALUDE_six_distinct_one_repeat_probability_l334_33454

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 7

/-- The probability of rolling seven standard six-sided dice and getting exactly six distinct numbers, with one number repeating once -/
theorem six_distinct_one_repeat_probability : 
  (num_sides.choose 1 * (num_sides - 1).factorial * num_dice.choose 2) / num_sides ^ num_dice = 5 / 186 := by
  sorry

end NUMINAMATH_CALUDE_six_distinct_one_repeat_probability_l334_33454


namespace NUMINAMATH_CALUDE_sum_of_roots_is_18_l334_33496

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry property of g
def is_symmetric_about_3 (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

-- Define the property of having exactly six distinct real roots
def has_six_distinct_roots (g : ℝ → ℝ) : Prop :=
  ∃ (a b c d e f : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                        d ≠ e ∧ d ≠ f ∧
                        e ≠ f) ∧
  (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧ g e = 0 ∧ g f = 0) ∧
  (∀ x : ℝ, g x = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f))

-- Theorem statement
theorem sum_of_roots_is_18 (g : ℝ → ℝ) 
  (h1 : is_symmetric_about_3 g) 
  (h2 : has_six_distinct_roots g) :
  ∃ (a b c d e f : ℝ), 
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧ g e = 0 ∧ g f = 0) ∧
    (a + b + c + d + e + f = 18) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_18_l334_33496


namespace NUMINAMATH_CALUDE_haley_seeds_l334_33497

/-- The number of seeds Haley planted in the big garden -/
def big_garden_seeds : ℕ := 35

/-- The number of small gardens Haley had -/
def small_gardens : ℕ := 7

/-- The number of seeds Haley planted in each small garden -/
def seeds_per_small_garden : ℕ := 3

/-- The total number of seeds Haley started with -/
def total_seeds : ℕ := big_garden_seeds + small_gardens * seeds_per_small_garden

theorem haley_seeds : total_seeds = 56 := by
  sorry

end NUMINAMATH_CALUDE_haley_seeds_l334_33497


namespace NUMINAMATH_CALUDE_sqrt_fraction_inequality_l334_33461

theorem sqrt_fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_inequality_l334_33461


namespace NUMINAMATH_CALUDE_fixed_point_on_curve_l334_33451

/-- The curve equation for any real m and n -/
def curve_equation (x y m n : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 2*n*y + 4*(m - n - 2) = 0

/-- Theorem stating that the point (2, -2) lies on the curve for all real m and n -/
theorem fixed_point_on_curve :
  ∀ (m n : ℝ), curve_equation 2 (-2) m n := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_curve_l334_33451


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l334_33416

theorem system_of_equations_solution 
  (x y z a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (eq1 : y * z / (y + z) = a)
  (eq2 : x * z / (x + z) = b)
  (eq3 : x * y / (x + y) = c) :
  x = 2 * a * b * c / (a * c + a * b - b * c) ∧
  y = 2 * a * b * c / (a * b + b * c - a * c) ∧
  z = 2 * a * b * c / (a * c + b * c - a * b) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l334_33416


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l334_33472

/-- Given a line L1 with equation x - 2y = 0 and a point P (-3, -1),
    prove that the line L2 with equation x - 2y + 1 = 0 passes through P
    and is parallel to L1. -/
theorem parallel_line_through_point (L1 L2 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  L1 = {(x, y) | x - 2*y = 0} →
  L2 = {(x, y) | x - 2*y + 1 = 0} →
  P = (-3, -1) →
  (P ∈ L2) ∧ (∀ (x y : ℝ), (x, y) ∈ L1 ↔ ∃ (k : ℝ), (x + k, y + k/2) ∈ L2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l334_33472


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l334_33490

/-- The time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 375) 
  (h2 : train_speed_kmph = 90) 
  (h3 : bridge_length = 1250) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 65 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l334_33490


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l334_33493

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and eccentricity e = √5/2, prove that its asymptotes are y = ±(1/2)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (he : Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 / 2) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = (1/2) * x ∨ f x = -(1/2) * x) ∧
  (∀ ε > 0, ∃ M > 0, ∀ x y, x^2/a^2 - y^2/b^2 = 1 → abs x > M →
    abs (y - f x) < ε * abs x) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l334_33493


namespace NUMINAMATH_CALUDE_stratified_sample_female_result_l334_33469

/-- Represents the number of female athletes to be selected in a stratified sample -/
def stratified_sample_female (total_male : ℕ) (total_female : ℕ) (sample_size : ℕ) : ℕ :=
  (total_female * sample_size) / (total_male + total_female)

/-- Theorem: In a stratified sampling of 28 people from a population of 98 athletes 
    (56 male and 42 female), the number of female athletes that should be selected is 12 -/
theorem stratified_sample_female_result : 
  stratified_sample_female 56 42 28 = 12 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_female_result_l334_33469


namespace NUMINAMATH_CALUDE_natural_raisin_cost_l334_33455

/-- The cost per scoop of golden seedless raisins in dollars -/
def golden_cost : ℚ := 255/100

/-- The number of scoops of golden seedless raisins -/
def golden_scoops : ℕ := 20

/-- The number of scoops of natural seedless raisins -/
def natural_scoops : ℕ := 20

/-- The cost per scoop of the mixture in dollars -/
def mixture_cost : ℚ := 3

/-- The cost per scoop of natural seedless raisins in dollars -/
def natural_cost : ℚ := 345/100

theorem natural_raisin_cost : 
  (golden_cost * golden_scoops + natural_cost * natural_scoops) / (golden_scoops + natural_scoops) = mixture_cost :=
sorry

end NUMINAMATH_CALUDE_natural_raisin_cost_l334_33455


namespace NUMINAMATH_CALUDE_min_value_on_circle_l334_33423

theorem min_value_on_circle (a b : ℝ) (h : a^2 + b^2 - 4*a + 3 = 0) :
  2 ≤ Real.sqrt (a^2 + b^2) + 1 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀^2 + b₀^2 - 4*a₀ + 3 = 0 ∧ Real.sqrt (a₀^2 + b₀^2) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l334_33423


namespace NUMINAMATH_CALUDE_black_car_speed_proof_l334_33409

/-- The speed of the red car in miles per hour -/
def red_car_speed : ℝ := 10

/-- The initial distance between the cars in miles -/
def initial_distance : ℝ := 20

/-- The time it takes for the black car to overtake the red car in hours -/
def overtake_time : ℝ := 0.5

/-- The speed of the black car in miles per hour -/
def black_car_speed : ℝ := 50

theorem black_car_speed_proof :
  red_car_speed * overtake_time + initial_distance = black_car_speed * overtake_time :=
by sorry

end NUMINAMATH_CALUDE_black_car_speed_proof_l334_33409


namespace NUMINAMATH_CALUDE_circus_illumination_theorem_l334_33479

/-- A convex figure in a plane -/
structure ConvexFigure where
  -- Define properties of a convex figure

/-- The plane -/
structure Plane where
  -- Define properties of a plane

/-- Represents the illumination of the arena -/
def Illumination (n : ℕ) := Fin n → ConvexFigure

/-- The union of a subset of convex figures -/
def UnionOfFigures (i : Illumination n) (s : Finset (Fin n)) : Set Plane :=
  sorry

/-- The entire plane is covered -/
def CoversPlaane (s : Set Plane) : Prop :=
  sorry

/-- Main theorem: For any n ≥ 2, there exists an illumination arrangement satisfying the conditions -/
theorem circus_illumination_theorem (n : ℕ) (h : n ≥ 2) :
  ∃ (i : Illumination n),
    (∀ (k : Fin n), CoversPlaane (UnionOfFigures i (Finset.erase (Finset.univ : Finset (Fin n)) k))) ∧
    (∀ (j k : Fin n), j ≠ k → ¬CoversPlaane (UnionOfFigures i (Finset.erase (Finset.erase (Finset.univ : Finset (Fin n)) j) k))) :=
  sorry

end NUMINAMATH_CALUDE_circus_illumination_theorem_l334_33479


namespace NUMINAMATH_CALUDE_total_passengers_is_420_l334_33435

/-- Represents the number of carriages in a train -/
def carriages_per_train : ℕ := 4

/-- Represents the original number of seats in each carriage -/
def original_seats_per_carriage : ℕ := 25

/-- Represents the additional number of passengers each carriage can accommodate -/
def additional_passengers_per_carriage : ℕ := 10

/-- Represents the number of trains -/
def number_of_trains : ℕ := 3

/-- Calculates the total number of passengers that can fill up the given number of trains -/
def total_passengers : ℕ :=
  number_of_trains * carriages_per_train * (original_seats_per_carriage + additional_passengers_per_carriage)

theorem total_passengers_is_420 : total_passengers = 420 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_is_420_l334_33435


namespace NUMINAMATH_CALUDE_lemming_average_distance_l334_33462

/-- Given a square with side length 12 meters, if a point moves 7.2 meters along the diagonal
    from one corner and then 3 meters perpendicular to that diagonal, the average distance
    from the resulting point to all four sides of the square is 6 meters. -/
theorem lemming_average_distance (square_side : ℝ) (diagonal_move : ℝ) (perpendicular_move : ℝ) :
  square_side = 12 →
  diagonal_move = 7.2 →
  perpendicular_move = 3 →
  let diagonal_length := square_side * Real.sqrt 2
  let diagonal_fraction := diagonal_move / diagonal_length
  let x := diagonal_fraction * square_side + perpendicular_move
  let y := diagonal_fraction * square_side
  let dist_left := x
  let dist_bottom := y
  let dist_right := square_side - x
  let dist_top := square_side - y
  let avg_distance := (dist_left + dist_bottom + dist_right + dist_top) / 4
  avg_distance = 6 := by sorry

end NUMINAMATH_CALUDE_lemming_average_distance_l334_33462


namespace NUMINAMATH_CALUDE_opposite_numbers_l334_33480

theorem opposite_numbers (x y z : ℝ) (h : 1/x + 1/y + 1/z = 1/(x+y+z)) :
  x + y = 0 ∨ y + z = 0 ∨ x + z = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_l334_33480


namespace NUMINAMATH_CALUDE_inequality_preservation_l334_33447

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l334_33447


namespace NUMINAMATH_CALUDE_floor_sqrt_150_l334_33402

theorem floor_sqrt_150 : ⌊Real.sqrt 150⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_150_l334_33402


namespace NUMINAMATH_CALUDE_inequality_contradiction_l334_33456

theorem inequality_contradiction (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ 
    (a + b) * (c + d) < a * b + c * d ∧ 
    (a + b) * c * d < a * b * (c + d)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l334_33456


namespace NUMINAMATH_CALUDE_base7_sum_equality_l334_33466

-- Define a type for base 7 digits
def Base7Digit := { n : Nat // n > 0 ∧ n < 7 }

-- Function to convert a three-digit base 7 number to natural number
def base7ToNat (a b c : Base7Digit) : Nat :=
  49 * a.val + 7 * b.val + c.val

-- Statement of the theorem
theorem base7_sum_equality 
  (A B C : Base7Digit) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (hSum : base7ToNat A B C + base7ToNat B C A + base7ToNat C A B = 
          343 * A.val + 49 * A.val + 7 * A.val) : 
  B.val + C.val = 6 := by
sorry

end NUMINAMATH_CALUDE_base7_sum_equality_l334_33466


namespace NUMINAMATH_CALUDE_ninas_run_l334_33419

theorem ninas_run (x : ℝ) : x + x + 0.67 = 0.83 → x = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_ninas_run_l334_33419


namespace NUMINAMATH_CALUDE_six_digit_scrambled_divisibility_l334_33434

theorem six_digit_scrambled_divisibility (a b c : Nat) 
  (ha : a ∈ Finset.range 10) 
  (hb : b ∈ Finset.range 10) 
  (hc : c ∈ Finset.range 10) 
  (hpos : 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * c + b > 0) :
  let Z := 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * c + b
  ∃ k : Nat, Z = 101 * k := by
  sorry

end NUMINAMATH_CALUDE_six_digit_scrambled_divisibility_l334_33434


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l334_33431

variables (μ σ : ℝ) (ξ : ℝ → ℝ)

-- Define the normal distribution
def normal_dist (μ σ : ℝ) (ξ : ℝ → ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, f x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ)^2) / (2 * σ^2))

-- Define the probability function
noncomputable def P (A : Set ℝ) : ℝ := sorry

theorem normal_distribution_probability 
  (h1 : normal_dist μ σ ξ)
  (h2 : P {x | ξ x < -1} = 0.3)
  (h3 : P {x | ξ x > 2} = 0.3) :
  P {x | ξ x < 2*μ + 1} = 0.7 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l334_33431


namespace NUMINAMATH_CALUDE_conjunction_implies_disjunction_l334_33400

theorem conjunction_implies_disjunction (p q : Prop) : (p ∧ q) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_conjunction_implies_disjunction_l334_33400


namespace NUMINAMATH_CALUDE_tan_two_alpha_l334_33464

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem tan_two_alpha (α : ℝ) (h : (deriv f) α = 3 * f α) : Real.tan (2 * α) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_alpha_l334_33464


namespace NUMINAMATH_CALUDE_fly_ceiling_distance_l334_33439

def fly_distance (x y z : ℝ) : Prop :=
  x = 2 ∧ y = 7 ∧ x^2 + y^2 + z^2 = 10^2

theorem fly_ceiling_distance :
  ∀ x y z : ℝ, fly_distance x y z → z = Real.sqrt 47 :=
by
  sorry

end NUMINAMATH_CALUDE_fly_ceiling_distance_l334_33439


namespace NUMINAMATH_CALUDE_bug_eating_ratio_l334_33427

theorem bug_eating_ratio (gecko lizard frog toad : ℕ) : 
  gecko = 12 →
  lizard = gecko / 2 →
  toad = frog + frog / 2 →
  gecko + lizard + frog + toad = 63 →
  frog / lizard = 3 := by
  sorry

end NUMINAMATH_CALUDE_bug_eating_ratio_l334_33427


namespace NUMINAMATH_CALUDE_johnson_smith_tied_may_l334_33475

/-- Represents the months of a baseball season --/
inductive Month
| Jan | Feb | Mar | Apr | May | Jul | Aug | Sep

/-- Represents a baseball player --/
structure Player where
  name : String
  homeRuns : Month → Nat

def johnson : Player :=
  { name := "Johnson"
  , homeRuns := fun
    | Month.Jan => 2
    | Month.Feb => 12
    | Month.Mar => 15
    | Month.Apr => 8
    | Month.May => 14
    | Month.Jul => 11
    | Month.Aug => 9
    | Month.Sep => 16 }

def smith : Player :=
  { name := "Smith"
  , homeRuns := fun
    | Month.Jan => 5
    | Month.Feb => 9
    | Month.Mar => 10
    | Month.Apr => 12
    | Month.May => 15
    | Month.Jul => 12
    | Month.Aug => 10
    | Month.Sep => 17 }

def totalHomeRunsUpTo (p : Player) (m : Month) : Nat :=
  match m with
  | Month.Jan => p.homeRuns Month.Jan
  | Month.Feb => p.homeRuns Month.Jan + p.homeRuns Month.Feb
  | Month.Mar => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar
  | Month.Apr => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr
  | Month.May => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May
  | Month.Jul => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May + p.homeRuns Month.Jul
  | Month.Aug => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May + p.homeRuns Month.Jul + p.homeRuns Month.Aug
  | Month.Sep => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May + p.homeRuns Month.Jul + p.homeRuns Month.Aug + p.homeRuns Month.Sep

theorem johnson_smith_tied_may :
  totalHomeRunsUpTo johnson Month.May = totalHomeRunsUpTo smith Month.May :=
by sorry

end NUMINAMATH_CALUDE_johnson_smith_tied_may_l334_33475


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l334_33477

theorem quadratic_roots_relation (a b : ℝ) (p : ℝ) : 
  (3 * a^2 + 7 * a + 6 = 0) →
  (3 * b^2 + 7 * b + 6 = 0) →
  (a^3 + b^3 = -p) →
  (p = -35/27) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l334_33477


namespace NUMINAMATH_CALUDE_first_term_is_one_l334_33411

/-- A geometric sequence with its sum function -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1
  sum_formula : ∀ n : ℕ, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)

/-- Theorem: For a geometric sequence with specific sum values, the first term is 1 -/
theorem first_term_is_one (seq : GeometricSequence) (m : ℕ) 
    (h1 : seq.S (m - 2) = 1)
    (h2 : seq.S m = 3)
    (h3 : seq.S (m + 2) = 5) :
  seq.a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_one_l334_33411


namespace NUMINAMATH_CALUDE_max_moves_card_game_l334_33471

/-- Represents the state of the cards as a natural number -/
def initial_state : Nat := 43690

/-- Represents a valid move in the game -/
def is_valid_move (n : Nat) : Prop :=
  ∃ k, 0 < k ∧ k ≤ 16 ∧ n.mod (2^k) = 2^(k-1)

/-- The game ends when no valid move can be made -/
def game_ended (n : Nat) : Prop :=
  ¬∃ m, is_valid_move n ∧ m < n

/-- Theorem stating the maximum number of moves in the game -/
theorem max_moves_card_game :
  ∃ moves : Nat, moves = initial_state ∧
  (∀ n, n > moves → ¬∃ seq : Nat → Nat, seq 0 = initial_state ∧
    (∀ i < n, is_valid_move (seq i) ∧ seq (i+1) < seq i) ∧
    game_ended (seq n)) :=
sorry

end NUMINAMATH_CALUDE_max_moves_card_game_l334_33471


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_product_l334_33458

theorem seventh_root_of_unity_product (z : ℂ) (h1 : z^7 = 1) (h2 : z ≠ 1) :
  (z - 1) * (z^2 - 1) * (z^3 - 1) * (z^4 - 1) * (z^5 - 1) * (z^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_product_l334_33458


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_line_l334_33483

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_line 
  (α β γ : Plane) (l : Line) 
  (distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  perpendicular_line_plane l α → 
  parallel_line_plane l β → 
  perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_line_l334_33483


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l334_33486

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 50 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l334_33486


namespace NUMINAMATH_CALUDE_f_properties_l334_33478

noncomputable def f (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

theorem f_properties (a b c : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 0 → f a b c x₁ > f a b c x₂) →
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f a b c x₁ < f a b c x₂) →
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ f a b c x₁ = 0 ∧ f a b c x₂ = 0 ∧ f a b c x₃ = 0) →
  f a b c 1 = 0 →
  b = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l334_33478


namespace NUMINAMATH_CALUDE_line_intersects_circle_twice_l334_33449

/-- The circle C with equation x^2 + y^2 - 2x - 6y - 15 = 0 -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y - 15 = 0

/-- The line l with equation (1+3k)x + (3-2k)y + 4k - 17 = 0 for any real k -/
def Line (k x y : ℝ) : Prop :=
  (1+3*k)*x + (3-2*k)*y + 4*k - 17 = 0

/-- The theorem stating that the line intersects the circle at exactly two points for any real k -/
theorem line_intersects_circle_twice :
  ∀ k : ℝ, ∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ 
    Circle p1.1 p1.2 ∧ Circle p2.1 p2.2 ∧
    Line k p1.1 p1.2 ∧ Line k p2.1 p2.2 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_twice_l334_33449


namespace NUMINAMATH_CALUDE_mean_not_imply_concentration_stability_range_imply_concentration_stability_std_dev_imply_concentration_stability_variance_imply_concentration_stability_l334_33484

/- Define a dataset as a list of real numbers -/
def Dataset := List ℝ

/- Define statistical measures -/
def range (data : Dataset) : ℝ := sorry
def mean (data : Dataset) : ℝ := sorry
def standardDeviation (data : Dataset) : ℝ := sorry
def variance (data : Dataset) : ℝ := sorry

/- Define a measure of concentration and stability -/
def isConcentratedAndStable (data : Dataset) : Prop := sorry

/- Theorem stating that mean does not imply concentration and stability -/
theorem mean_not_imply_concentration_stability :
  ∃ (data1 data2 : Dataset), mean data1 < mean data2 ∧
    (isConcentratedAndStable data1 ↔ ¬isConcentratedAndStable data2) := by sorry

/- Theorems stating that other measures imply concentration and stability -/
theorem range_imply_concentration_stability :
  ∀ (data1 data2 : Dataset), range data1 < range data2 →
    (isConcentratedAndStable data1 → isConcentratedAndStable data2) := by sorry

theorem std_dev_imply_concentration_stability :
  ∀ (data1 data2 : Dataset), standardDeviation data1 < standardDeviation data2 →
    (isConcentratedAndStable data1 → isConcentratedAndStable data2) := by sorry

theorem variance_imply_concentration_stability :
  ∀ (data1 data2 : Dataset), variance data1 < variance data2 →
    (isConcentratedAndStable data1 → isConcentratedAndStable data2) := by sorry

end NUMINAMATH_CALUDE_mean_not_imply_concentration_stability_range_imply_concentration_stability_std_dev_imply_concentration_stability_variance_imply_concentration_stability_l334_33484


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_15_l334_33418

def f (n : ℕ+) : ℕ := sorry

theorem smallest_n_exceeding_15 :
  (∀ k : ℕ+, k < 3 → f k ≤ 15) ∧ f 3 > 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_15_l334_33418
