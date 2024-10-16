import Mathlib

namespace NUMINAMATH_CALUDE_square_diagonal_property_l1037_103735

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square -/
structure Square :=
  (p q r s : Point)

/-- The small square PQRS is contained in the big square -/
def small_square_in_big_square (small : Square) (big : Square) : Prop := sorry

/-- Point A lies on the extension of PQ -/
def point_on_extension (p q a : Point) : Prop := sorry

/-- Points A, B, C, D lie on the sides of the big square in order -/
def points_on_sides (a b c d : Point) (big : Square) : Prop := sorry

/-- Two line segments are equal -/
def segments_equal (p1 q1 p2 q2 : Point) : Prop := sorry

/-- Two line segments are perpendicular -/
def segments_perpendicular (p1 q1 p2 q2 : Point) : Prop := sorry

theorem square_diagonal_property (small big : Square) (a b c d : Point) :
  small_square_in_big_square small big →
  point_on_extension small.p small.q a →
  point_on_extension small.q small.r b →
  point_on_extension small.r small.s c →
  point_on_extension small.s small.p d →
  points_on_sides a b c d big →
  segments_equal a c b d ∧ segments_perpendicular a c b d := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_property_l1037_103735


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1037_103707

theorem cyclic_sum_inequality (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) :
  Real.sqrt (1 - a*b) + Real.sqrt (1 - b*c) + Real.sqrt (1 - c*d) + 
  Real.sqrt (1 - d*a) + Real.sqrt (1 - a*c) + Real.sqrt (1 - b*d) ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1037_103707


namespace NUMINAMATH_CALUDE_production_normality_l1037_103770

-- Define the parameters of the normal distribution
def μ : ℝ := 8.0
def σ : ℝ := 0.15

-- Define the 3-sigma range
def lower_bound : ℝ := μ - 3 * σ
def upper_bound : ℝ := μ + 3 * σ

-- Define the observed diameters
def morning_diameter : ℝ := 7.9
def afternoon_diameter : ℝ := 7.5

-- Define what it means for a production to be normal
def is_normal (x : ℝ) : Prop := lower_bound ≤ x ∧ x ≤ upper_bound

-- Theorem statement
theorem production_normality :
  is_normal morning_diameter ∧ ¬is_normal afternoon_diameter :=
sorry

end NUMINAMATH_CALUDE_production_normality_l1037_103770


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l1037_103725

open Real

theorem tangent_equation_solution (x : ℝ) : 
  8.482 * (3 * tan x - tan x ^ 3) / (1 - tan x ^ 2) * (cos (3 * x) + cos x) = 2 * sin (5 * x) ↔ 
  (∃ k : ℤ, x = k * π) ∨ (∃ k : ℤ, x = π / 8 * (2 * k + 1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l1037_103725


namespace NUMINAMATH_CALUDE_prob_one_good_product_prob_one_good_product_proof_l1037_103785

/-- The probability of selecting exactly one good product when randomly selecting
    two products from a set of five products, where three are good and two are defective. -/
theorem prob_one_good_product : ℚ :=
  let total_products : ℕ := 5
  let good_products : ℕ := 3
  let defective_products : ℕ := 2
  let selected_products : ℕ := 2
  3 / 5

/-- Proof that the probability of selecting exactly one good product is 3/5. -/
theorem prob_one_good_product_proof :
  prob_one_good_product = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_good_product_prob_one_good_product_proof_l1037_103785


namespace NUMINAMATH_CALUDE_triangle_area_is_16_l1037_103775

/-- The area of a triangle formed by three lines in a 2D plane --/
def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ :=
  sorry

/-- The first line: y = 6 --/
def line1 (x : ℝ) : ℝ := 6

/-- The second line: y = 2 + x --/
def line2 (x : ℝ) : ℝ := 2 + x

/-- The third line: y = 2 - x --/
def line3 (x : ℝ) : ℝ := 2 - x

theorem triangle_area_is_16 : triangleArea line1 line2 line3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_16_l1037_103775


namespace NUMINAMATH_CALUDE_tan_570_degrees_l1037_103745

theorem tan_570_degrees : Real.tan (570 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_570_degrees_l1037_103745


namespace NUMINAMATH_CALUDE_quadrilaterals_from_circle_points_l1037_103798

/-- The number of distinct points on the circumference of a circle -/
def num_points : ℕ := 10

/-- The number of vertices required to form a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

/-- The number of different quadrilaterals that can be formed -/
def num_quadrilaterals : ℕ := Nat.choose num_points vertices_per_quadrilateral

theorem quadrilaterals_from_circle_points : num_quadrilaterals = 300 := by
  sorry

end NUMINAMATH_CALUDE_quadrilaterals_from_circle_points_l1037_103798


namespace NUMINAMATH_CALUDE_tan_double_angle_problem_l1037_103723

theorem tan_double_angle_problem (θ : Real) 
  (h1 : Real.tan (2 * θ) = -2) 
  (h2 : π < 2 * θ) 
  (h3 : 2 * θ < 2 * π) : 
  Real.sin θ ^ 4 - Real.cos θ ^ 4 = -1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_problem_l1037_103723


namespace NUMINAMATH_CALUDE_complex_product_quadrant_l1037_103755

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (0 < z.re) ∧ (0 < z.im) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_quadrant_l1037_103755


namespace NUMINAMATH_CALUDE_tall_min_voters_to_win_l1037_103796

/-- Represents the voting structure and results of the giraffe beauty contest -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat
  winner : String

/-- Calculates the minimum number of voters required for a giraffe to win the contest -/
def min_voters_to_win (contest : GiraffeContest) : Nat :=
  sorry

/-- The theorem stating the minimum number of voters required for Tall to win -/
theorem tall_min_voters_to_win (contest : GiraffeContest) 
  (h1 : contest.total_voters = 105)
  (h2 : contest.num_districts = 5)
  (h3 : contest.sections_per_district = 7)
  (h4 : contest.voters_per_section = 3)
  (h5 : contest.winner = "Tall") :
  min_voters_to_win contest = 24 := by
  sorry

#check tall_min_voters_to_win

end NUMINAMATH_CALUDE_tall_min_voters_to_win_l1037_103796


namespace NUMINAMATH_CALUDE_nikolai_is_petrs_son_l1037_103789

/-- Represents a person who went fishing -/
structure Fisher where
  name : String
  fish_caught : ℕ

/-- Represents a father-son pair who went fishing -/
structure FishingPair where
  father : Fisher
  son : Fisher

/-- The total number of fish caught by all fishers -/
def total_fish : ℕ := 25

/-- Theorem stating that given the conditions, Nikolai must be Petr's son -/
theorem nikolai_is_petrs_son (pair1 pair2 : FishingPair) 
  (h1 : pair1.father.name = "Petr")
  (h2 : pair1.father.fish_caught = 3 * pair1.son.fish_caught)
  (h3 : pair2.father.fish_caught = pair2.son.fish_caught)
  (h4 : pair1.father.fish_caught + pair1.son.fish_caught + 
        pair2.father.fish_caught + pair2.son.fish_caught = total_fish)
  : pair1.son.name = "Nikolai" := by
  sorry

end NUMINAMATH_CALUDE_nikolai_is_petrs_son_l1037_103789


namespace NUMINAMATH_CALUDE_bell_rings_theorem_l1037_103713

/-- Represents the number of times a bell rings for a single class -/
def bell_rings_per_class : ℕ := 2

/-- Represents the total number of classes in a day -/
def total_classes : ℕ := 5

/-- Represents the current class number (1-indexed) -/
def current_class : ℕ := 5

/-- Calculates the total number of bell rings up to and including the current class -/
def total_bell_rings (completed_classes : ℕ) (current_class : ℕ) : ℕ :=
  completed_classes * bell_rings_per_class + 1

/-- Theorem: Given 5 classes where the bell rings twice for each completed class 
    and once for the current class (Music), the total number of bell rings is 9 -/
theorem bell_rings_theorem : 
  total_bell_rings (current_class - 1) current_class = 9 := by
  sorry

end NUMINAMATH_CALUDE_bell_rings_theorem_l1037_103713


namespace NUMINAMATH_CALUDE_age_of_b_l1037_103790

theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 29 →
  (a + c) / 2 = 32 →
  b = 23 := by
sorry

end NUMINAMATH_CALUDE_age_of_b_l1037_103790


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1037_103729

/-- The set M of real numbers a such that x² + ax + 1 > 0 for all real x -/
def M : Set ℝ := {a | ∀ x : ℝ, x^2 + a*x + 1 > 0}

/-- The set N of real numbers a such that there exists a real x where (a-3)x + 1 = 0 -/
def N : Set ℝ := {a | ∃ x : ℝ, (a-3)*x + 1 = 0}

/-- Proposition p: a is in set M -/
def p (a : ℝ) : Prop := a ∈ M

/-- Proposition q: a is in set N -/
def q (a : ℝ) : Prop := a ∈ N

theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬p a) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1037_103729


namespace NUMINAMATH_CALUDE_complex_simplification_l1037_103768

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  (1 - 3*i) / (1 - i) = 2 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l1037_103768


namespace NUMINAMATH_CALUDE_circle_center_l1037_103773

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 12 = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : ℝ × ℝ := (h, k)

/-- Theorem: The center of the circle given by x^2 - 6x + y^2 + 2y - 12 = 0 is (3, -1) -/
theorem circle_center : 
  ∃ (h k : ℝ), CircleCenter h k = (3, -1) ∧ 
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 12 + 6*h - 2*k) :=
sorry

end NUMINAMATH_CALUDE_circle_center_l1037_103773


namespace NUMINAMATH_CALUDE_consecutive_cube_diff_square_l1037_103717

theorem consecutive_cube_diff_square (x : ℤ) :
  ∃ y : ℤ, (x + 1)^3 - x^3 = y^2 →
  ∃ a b : ℤ, y = a^2 + b^2 ∧ b = a + 1 := by
sorry

end NUMINAMATH_CALUDE_consecutive_cube_diff_square_l1037_103717


namespace NUMINAMATH_CALUDE_nested_circles_radius_l1037_103799

theorem nested_circles_radius (B₁ B₃ : ℝ) : 
  B₁ > 0 →
  B₃ > 0 →
  (B₁ + B₃ = π * 6^2) →
  (B₃ - B₁ = (B₁ + B₃) - B₁) →
  (B₁ = π * (3 * Real.sqrt 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_nested_circles_radius_l1037_103799


namespace NUMINAMATH_CALUDE_log_condition_equivalence_l1037_103724

theorem log_condition_equivalence (m n : ℝ) 
  (hm_pos : m > 0) (hm_neq_one : m ≠ 1) (hn_pos : n > 0) :
  (Real.log n / Real.log m < 0) ↔ ((m - 1) * (n - 1) < 0) := by
  sorry

end NUMINAMATH_CALUDE_log_condition_equivalence_l1037_103724


namespace NUMINAMATH_CALUDE_fair_ride_cost_l1037_103736

/-- Represents the fair entrance and ride costs --/
structure FairCosts where
  under18Fee : ℚ
  adultFeeIncrease : ℚ
  totalSpent : ℚ
  numRides : ℕ
  numUnder18 : ℕ
  numAdults : ℕ

/-- Calculates the cost per ride given the fair costs --/
def costPerRide (costs : FairCosts) : ℚ :=
  let adultFee := costs.under18Fee * (1 + costs.adultFeeIncrease)
  let totalEntrance := costs.under18Fee * costs.numUnder18 + adultFee * costs.numAdults
  let totalRideCost := costs.totalSpent - totalEntrance
  totalRideCost / costs.numRides

/-- Theorem stating that the cost per ride is $0.50 given the problem conditions --/
theorem fair_ride_cost :
  let costs : FairCosts := {
    under18Fee := 5,
    adultFeeIncrease := 1/5,
    totalSpent := 41/2,
    numRides := 9,
    numUnder18 := 2,
    numAdults := 1
  }
  costPerRide costs = 1/2 := by sorry


end NUMINAMATH_CALUDE_fair_ride_cost_l1037_103736


namespace NUMINAMATH_CALUDE_sum_first_10_odd_integers_l1037_103784

theorem sum_first_10_odd_integers : 
  (Finset.range 10).sum (fun i => 2 * i + 1) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_10_odd_integers_l1037_103784


namespace NUMINAMATH_CALUDE_sum_of_angles_with_given_tangents_l1037_103757

theorem sum_of_angles_with_given_tangents (A B C : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  Real.tan A = 1 →
  Real.tan B = 2 →
  Real.tan C = 3 →
  A + B + C = π := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_with_given_tangents_l1037_103757


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l1037_103738

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l1037_103738


namespace NUMINAMATH_CALUDE_expansion_coefficients_l1037_103780

theorem expansion_coefficients :
  let expr := (1 + X^5 + X^7)^20
  ∃ (p : Polynomial ℤ),
    p = expr ∧
    p.coeff 18 = 0 ∧
    p.coeff 17 = 3420 :=
by sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l1037_103780


namespace NUMINAMATH_CALUDE_triangle_ratio_bound_l1037_103708

/-- For any triangle with perimeter p, circumradius R, and inradius r,
    the expression p/R * (1 - r/(3R)) is at most 5√3/2 -/
theorem triangle_ratio_bound (p R r : ℝ) (hp : p > 0) (hR : R > 0) (hr : r > 0)
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    p = a + b + c ∧
    R = (a * b * c) / (4 * (a + b - c) * (b + c - a) * (c + a - b))^(1/2) ∧
    r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * p)) :
  p / R * (1 - r / (3 * R)) ≤ 5 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_bound_l1037_103708


namespace NUMINAMATH_CALUDE_pokemon_card_collection_l1037_103771

def cards_needed (michael_cards : ℕ) (mark_diff : ℕ) (lloyd_ratio : ℕ) (total_goal : ℕ) : ℕ :=
  let mark_cards := michael_cards - mark_diff
  let lloyd_cards := mark_cards / lloyd_ratio
  total_goal - (michael_cards + mark_cards + lloyd_cards)

theorem pokemon_card_collection : 
  cards_needed 100 10 3 300 = 80 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_collection_l1037_103771


namespace NUMINAMATH_CALUDE_estate_distribution_l1037_103750

/-- Represents the estate distribution problem --/
theorem estate_distribution (E : ℕ) : 
  -- Daughter and son together receive half the estate
  (∃ x : ℕ, 5 * x = E / 2) →
  -- Wife receives three times as much as the son
  (∃ y : ℕ, y = 6 * x) →
  -- First cook receives $800
  (∃ z₁ : ℕ, z₁ = 800) →
  -- Second cook receives $1200
  (∃ z₂ : ℕ, z₂ = 1200) →
  -- Total estate equals sum of all shares
  (E = 11 * x + 2000) →
  -- The estate value is $20000
  E = 20000 := by
sorry

end NUMINAMATH_CALUDE_estate_distribution_l1037_103750


namespace NUMINAMATH_CALUDE_sum_digits_base7_999_l1037_103711

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Computes the sum of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  sorry

theorem sum_digits_base7_999 : sumList (toBase7 999) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_base7_999_l1037_103711


namespace NUMINAMATH_CALUDE_dice_probability_l1037_103710

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- Predicate to check if a number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Predicate to check if a number is a multiple of 5 -/
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k, n = 5 * k

/-- The set of all possible outcomes when rolling num_dice dice -/
def all_outcomes : Finset (Fin num_dice → Fin num_sides) := sorry

/-- The set of outcomes where at least one die shows an even number -/
def even_product_outcomes : Finset (Fin num_dice → Fin num_sides) := sorry

/-- The set of outcomes where the sum of dice is a multiple of 5 -/
def sum_multiple_of_5_outcomes : Finset (Fin num_dice → Fin num_sides) := sorry

/-- The number of favorable outcomes (sum is multiple of 5 given product is even) -/
def a : ℕ := sorry

/-- The probability of the sum being a multiple of 5 given the product is even -/
theorem dice_probability : 
  (Finset.card (even_product_outcomes ∩ sum_multiple_of_5_outcomes) : ℚ) / 
  (Finset.card even_product_outcomes : ℚ) = 
  (a : ℚ) / ((num_sides ^ num_dice - (num_sides / 2) ^ num_dice) : ℚ) :=
sorry

end NUMINAMATH_CALUDE_dice_probability_l1037_103710


namespace NUMINAMATH_CALUDE_cost_of_jeans_and_shirts_l1037_103766

/-- The cost of a pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of a shirt -/
def shirt_cost : ℝ := 9

theorem cost_of_jeans_and_shirts :
  3 * jeans_cost + 2 * shirt_cost = 69 :=
by
  have h1 : 2 * jeans_cost + 3 * shirt_cost = 61 := sorry
  sorry

end NUMINAMATH_CALUDE_cost_of_jeans_and_shirts_l1037_103766


namespace NUMINAMATH_CALUDE_extremum_and_inequality_l1037_103763

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x - 1) * Real.log (x + a)

theorem extremum_and_inequality (h : ∀ a > 0, ∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f a x ≤ f a 0) :
  (∃ a > 0, (deriv (f a)) 0 = 0) ∧
  (∀ x ≥ 0, f 1 x ≥ x^2) :=
sorry

end NUMINAMATH_CALUDE_extremum_and_inequality_l1037_103763


namespace NUMINAMATH_CALUDE_equation_solution_l1037_103776

theorem equation_solution (x : ℝ) (h : x ≠ 3) :
  (x^2 - 9) / (x - 3) = 3 * x ↔ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1037_103776


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l1037_103749

-- Define the functions
def f (x : ℝ) : ℝ := (x - 2) * (x - 5)
def g (x : ℝ) : ℝ := 2 * f x
def h (x : ℝ) : ℝ := f (-x) + 2

-- Define the number of intersection points
def a : ℕ := 2  -- number of intersection points between y=f(x) and y=g(x)
def b : ℕ := 1  -- number of intersection points between y=f(x) and y=h(x)

-- Theorem statement
theorem intersection_points_theorem : 10 * a + b = 21 := by sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l1037_103749


namespace NUMINAMATH_CALUDE_intersection_M_N_l1037_103774

open Set

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complementN : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := U \ complementN

-- Theorem statement
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1037_103774


namespace NUMINAMATH_CALUDE_cover_ways_eq_fib_cover_2x10_l1037_103759

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of ways to cover a 2 × n grid with 1 × 2 tiles -/
def cover_ways : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => cover_ways (n + 1) + cover_ways n

/-- Theorem: The number of ways to cover a 2 × n grid with 1 × 2 tiles 
    is equal to the (n+1)th Fibonacci number -/
theorem cover_ways_eq_fib (n : ℕ) : cover_ways n = fib (n + 1) := by
  sorry

/-- Corollary: There are 89 ways to cover a 2 × 10 grid with 1 × 2 tiles -/
theorem cover_2x10 : cover_ways 10 = 89 := by
  sorry

end NUMINAMATH_CALUDE_cover_ways_eq_fib_cover_2x10_l1037_103759


namespace NUMINAMATH_CALUDE_total_height_increase_l1037_103782

-- Define the increase in height per decade
def height_increase_per_decade : ℝ := 75

-- Define the number of centuries
def num_centuries : ℕ := 4

-- Define the number of decades in a century
def decades_per_century : ℕ := 10

-- Theorem statement
theorem total_height_increase : 
  height_increase_per_decade * (num_centuries * decades_per_century) = 3000 :=
by sorry

end NUMINAMATH_CALUDE_total_height_increase_l1037_103782


namespace NUMINAMATH_CALUDE_count_flippable_numbers_is_1500_l1037_103791

/-- A digit that remains valid when flipped -/
inductive ValidDigit
| Zero
| One
| Eight
| Six
| Nine

/-- A nine-digit number that remains unchanged when flipped -/
structure FlippableNumber :=
(d1 d2 d3 d4 d5 : ValidDigit)

/-- The count of FlippableNumbers -/
def count_flippable_numbers : ℕ := sorry

/-- The first digit cannot be zero -/
axiom first_digit_nonzero :
  ∀ (n : FlippableNumber), n.d1 ≠ ValidDigit.Zero

/-- The theorem to be proved -/
theorem count_flippable_numbers_is_1500 :
  count_flippable_numbers = 1500 := by sorry

end NUMINAMATH_CALUDE_count_flippable_numbers_is_1500_l1037_103791


namespace NUMINAMATH_CALUDE_radiator_water_fraction_l1037_103719

/-- Represents the fraction of water remaining after a number of replacements -/
def waterFraction (initialVolume : ℚ) (replacementVolume : ℚ) (replacements : ℕ) : ℚ :=
  (1 - replacementVolume / initialVolume) ^ replacements

/-- Proves that the fraction of water in the radiator after 5 replacements is 243/1024 -/
theorem radiator_water_fraction :
  waterFraction 20 5 5 = 243 / 1024 := by
  sorry

#eval waterFraction 20 5 5

end NUMINAMATH_CALUDE_radiator_water_fraction_l1037_103719


namespace NUMINAMATH_CALUDE_max_min_values_on_interval_l1037_103765

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values_on_interval :
  ∃ (a b : ℝ), a ∈ Set.Icc 0 3 ∧ b ∈ Set.Icc 0 3 ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f a) ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 :=
sorry

end NUMINAMATH_CALUDE_max_min_values_on_interval_l1037_103765


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l1037_103746

theorem quadratic_roots_nature (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a * x^2 + b * x + c = 0 ∧ a = 1 ∧ b = -4 * Real.sqrt 3 ∧ c = 12 →
  discriminant = 0 ∧ ∃! x, a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l1037_103746


namespace NUMINAMATH_CALUDE_hotel_charge_difference_l1037_103756

theorem hotel_charge_difference (G R P : ℝ) 
  (hR : R = G * (1 + 0.125))
  (hP : P = R * (1 - 0.2)) :
  P = G * 0.9 := by
sorry

end NUMINAMATH_CALUDE_hotel_charge_difference_l1037_103756


namespace NUMINAMATH_CALUDE_inequality_range_l1037_103754

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ≤ 0 → m * (x^2 - 2*x) * Real.exp x + 1 ≥ Real.exp x) ↔ 
  m ≥ -1/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1037_103754


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1037_103751

/-- For a parabola with equation y² = -x, the distance from its focus to its directrix is 1/2. -/
theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), y^2 = -x → 
  ∃ (focus_x focus_y directrix_x : ℝ),
    (focus_x = -1/4 ∧ focus_y = 0) ∧
    directrix_x = 1/4 ∧
    |focus_x - directrix_x| = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1037_103751


namespace NUMINAMATH_CALUDE_rational_sqrt_equation_zero_l1037_103702

theorem rational_sqrt_equation_zero (a b c : ℚ) 
  (h : a + b * Real.sqrt 32 + c * Real.sqrt 34 = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt_equation_zero_l1037_103702


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l1037_103733

theorem gcd_special_numbers : Nat.gcd (2^2010 - 3) (2^2001 - 3) = 1533 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l1037_103733


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1037_103709

/-- Theorem: Area change of a rectangle after length decrease and width increase -/
theorem rectangle_area_change
  (l w : ℝ)  -- l: original length, w: original width
  (hl : l > 0)  -- length is positive
  (hw : w > 0)  -- width is positive
  : (0.9 * l) * (1.3 * w) = 1.17 * (l * w) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1037_103709


namespace NUMINAMATH_CALUDE_couponA_greatest_discount_specific_prices_l1037_103705

/-- Represents the discount amount for Coupon A -/
def couponA (p : ℝ) : ℝ := 0.15 * p

/-- Represents the discount amount for Coupon B -/
def couponB : ℝ := 30

/-- Represents the discount amount for Coupon C -/
def couponC (p : ℝ) : ℝ := 0.25 * (p - 150)

/-- Theorem stating when Coupon A offers the greatest discount -/
theorem couponA_greatest_discount (p : ℝ) :
  (couponA p > couponB ∧ couponA p > couponC p) ↔ (200 < p ∧ p < 375) :=
sorry

/-- Function to check if a price satisfies the condition for Coupon A being the best -/
def is_couponA_best (p : ℝ) : Prop := 200 < p ∧ p < 375

/-- Theorem for the specific price points given in the problem -/
theorem specific_prices :
  is_couponA_best 209.95 ∧
  is_couponA_best 229.95 ∧
  is_couponA_best 249.95 ∧
  ¬is_couponA_best 169.95 ∧
  ¬is_couponA_best 189.95 :=
sorry

end NUMINAMATH_CALUDE_couponA_greatest_discount_specific_prices_l1037_103705


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l1037_103764

/-- The area of a stripe wrapped around a cylindrical silo. -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 40) 
  (h2 : stripe_width = 4) 
  (h3 : revolutions = 3) : 
  stripe_width * revolutions * Real.pi * diameter = 480 * Real.pi := by
  sorry

#check stripe_area_on_cylindrical_silo

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l1037_103764


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l1037_103734

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ballsInBoxes (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 61 ways to put 5 distinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : ballsInBoxes 5 4 = 61 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l1037_103734


namespace NUMINAMATH_CALUDE_tv_sale_value_increase_l1037_103762

theorem tv_sale_value_increase 
  (original_price original_quantity : ℝ) 
  (original_price_positive : 0 < original_price)
  (original_quantity_positive : 0 < original_quantity) :
  let price_reduction_factor := 0.8
  let sales_increase_factor := 1.8
  let new_price := price_reduction_factor * original_price
  let new_quantity := sales_increase_factor * original_quantity
  let original_sale_value := original_price * original_quantity
  let new_sale_value := new_price * new_quantity
  (new_sale_value - original_sale_value) / original_sale_value = 0.44 := by
sorry

end NUMINAMATH_CALUDE_tv_sale_value_increase_l1037_103762


namespace NUMINAMATH_CALUDE_spherical_equation_describes_cone_l1037_103758

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Defines a cone in 3D space -/
structure Cone where
  vertex : ℝ × ℝ × ℝ
  axis : ℝ × ℝ × ℝ
  openingAngle : ℝ

/-- 
Given a constant c, prove that the equation φ = c in spherical coordinates (ρ,θ,φ) 
describes a cone with vertex at the origin, axis along the z-axis, and opening angle φ = c.
-/
theorem spherical_equation_describes_cone (c : ℝ) :
  ∃ (cone : Cone), 
    (∀ (p : SphericalPoint), p.φ = c ↔ 
      cone.vertex = (0, 0, 0) ∧ 
      cone.axis = (0, 0, 1) ∧ 
      cone.openingAngle = c) :=
by sorry

end NUMINAMATH_CALUDE_spherical_equation_describes_cone_l1037_103758


namespace NUMINAMATH_CALUDE_distributive_property_implication_l1037_103752

theorem distributive_property_implication (a b c : ℝ) (h : c ≠ 0) :
  (∀ x y z : ℝ, (x + y) * z = x * z + y * z) →
  (a + b) / c = a / c + b / c :=
by sorry

end NUMINAMATH_CALUDE_distributive_property_implication_l1037_103752


namespace NUMINAMATH_CALUDE_ace_spades_probability_after_removal_l1037_103716

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (black_suits : ℕ)

/-- Defines the properties of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    black_suits := 2 }

/-- Calculates the probability of drawing the Ace of Spades after removing some black cards -/
def probability_ace_spades (d : Deck) (removed_black_cards : ℕ) : ℚ :=
  1 / (d.total_cards - removed_black_cards)

/-- Theorem stating the probability of drawing the Ace of Spades after removing 12 black cards -/
theorem ace_spades_probability_after_removal :
  probability_ace_spades standard_deck 12 = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_ace_spades_probability_after_removal_l1037_103716


namespace NUMINAMATH_CALUDE_s_5_l1037_103794

/-- s(n) is a function that attaches the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- Examples of s(n) for n = 1, 2, 3, 4 -/
axiom s_examples : s 1 = 1 ∧ s 2 = 14 ∧ s 3 = 149 ∧ s 4 = 14916

/-- Theorem: s(5) equals 1491625 -/
theorem s_5 : s 5 = 1491625 := sorry

end NUMINAMATH_CALUDE_s_5_l1037_103794


namespace NUMINAMATH_CALUDE_p_minus_q_value_l1037_103748

theorem p_minus_q_value (p q : ℚ) 
  (h1 : -6 / p = 3/2) 
  (h2 : 8 / q = -1/4) : 
  p - q = 28 := by
sorry

end NUMINAMATH_CALUDE_p_minus_q_value_l1037_103748


namespace NUMINAMATH_CALUDE_school_transfer_percentage_l1037_103715

/-- Proves the percentage of students from school A going to school C -/
theorem school_transfer_percentage
  (total_students : ℕ)
  (school_A_percentage : ℚ)
  (school_B_to_C_percentage : ℚ)
  (total_to_C_percentage : ℚ)
  (h1 : school_A_percentage = 60 / 100)
  (h2 : school_B_to_C_percentage = 40 / 100)
  (h3 : total_to_C_percentage = 34 / 100)
  : ∃ (school_A_to_C_percentage : ℚ),
    school_A_to_C_percentage = 30 / 100 ∧
    (school_A_percentage * total_students * school_A_to_C_percentage +
     (1 - school_A_percentage) * total_students * school_B_to_C_percentage =
     total_students * total_to_C_percentage) := by
  sorry


end NUMINAMATH_CALUDE_school_transfer_percentage_l1037_103715


namespace NUMINAMATH_CALUDE_davis_remaining_sticks_l1037_103721

/-- The number of popsicle sticks Miss Davis had initially -/
def initial_sticks : ℕ := 170

/-- The number of popsicle sticks given to each group -/
def sticks_per_group : ℕ := 15

/-- The number of groups in Miss Davis's class -/
def number_of_groups : ℕ := 10

/-- The number of popsicle sticks Miss Davis has left -/
def remaining_sticks : ℕ := initial_sticks - (sticks_per_group * number_of_groups)

theorem davis_remaining_sticks : remaining_sticks = 20 := by
  sorry

end NUMINAMATH_CALUDE_davis_remaining_sticks_l1037_103721


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1037_103761

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * y * f x

/-- The main theorem stating that functions satisfying the equation are either constant zero or square. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesFunctionalEquation f) :
    (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1037_103761


namespace NUMINAMATH_CALUDE_area_of_ring_l1037_103720

/-- The area of a ring formed between two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 7) :
  π * r₁^2 - π * r₂^2 = 95 * π := by
  sorry

end NUMINAMATH_CALUDE_area_of_ring_l1037_103720


namespace NUMINAMATH_CALUDE_marble_group_size_l1037_103726

theorem marble_group_size :
  ∀ (x : ℕ),
  (144 / x : ℚ) = (144 / (x + 2) : ℚ) + 1 →
  x = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_group_size_l1037_103726


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l1037_103714

theorem sum_of_x_solutions_is_zero :
  ∀ x₁ x₂ : ℝ,
  (∃ y : ℝ, y = 7 ∧ x₁^2 + y^2 = 100 ∧ x₂^2 + y^2 = 100) →
  x₁ + x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l1037_103714


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_5_l1037_103779

theorem smallest_perfect_square_divisible_by_4_and_5 : ∃ n : ℕ, 
  (n > 0) ∧ 
  (∃ m : ℕ, n = m^2) ∧ 
  (n % 4 = 0) ∧ 
  (n % 5 = 0) ∧ 
  (∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l^2) → (k % 4 = 0) → (k % 5 = 0) → k ≥ n) ∧
  n = 400 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_5_l1037_103779


namespace NUMINAMATH_CALUDE_largest_k_for_g_range_l1037_103787

/-- The function g(x) defined as x^2 + 5x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + k

/-- The property that -5 is in the range of g(x) -/
def inRange (k : ℝ) : Prop := ∃ x, g k x = -5

/-- The theorem stating that 5/4 is the largest value of k such that -5 is in the range of g(x) -/
theorem largest_k_for_g_range :
  (∀ k > 5/4, ¬ inRange k) ∧ inRange (5/4) :=
sorry

end NUMINAMATH_CALUDE_largest_k_for_g_range_l1037_103787


namespace NUMINAMATH_CALUDE_smallest_positive_root_floor_l1037_103788

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x - 3 * Real.cos x + Real.tan x

def is_smallest_positive_root (s : ℝ) : Prop :=
  s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem smallest_positive_root_floor :
  ∃ s, is_smallest_positive_root s ∧ ⌊s⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_root_floor_l1037_103788


namespace NUMINAMATH_CALUDE_P_superset_Q_l1037_103712

-- Define the sets P and Q
def P : Set ℝ := {x | x ≥ 5}
def Q : Set ℝ := {x | 5 ≤ x ∧ x ≤ 7}

-- Theorem statement
theorem P_superset_Q : Q ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_P_superset_Q_l1037_103712


namespace NUMINAMATH_CALUDE_gift_wrapping_theorem_l1037_103777

/-- Cagney's gift wrapping rate in gifts per second -/
def cagney_rate : ℚ := 1 / 45

/-- Lacey's gift wrapping rate in gifts per second -/
def lacey_rate : ℚ := 1 / 60

/-- Total time available in seconds -/
def total_time : ℚ := 15 * 60

/-- The number of gifts that can be wrapped collectively -/
def total_gifts : ℕ := 35

theorem gift_wrapping_theorem :
  (cagney_rate + lacey_rate) * total_time = total_gifts := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_theorem_l1037_103777


namespace NUMINAMATH_CALUDE_total_flowers_planted_l1037_103728

theorem total_flowers_planted (num_people : ℕ) (num_days : ℕ) (flowers_per_day : ℕ) : 
  num_people = 5 → num_days = 2 → flowers_per_day = 20 → 
  num_people * num_days * flowers_per_day = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_planted_l1037_103728


namespace NUMINAMATH_CALUDE_pet_store_birds_count_l1037_103704

theorem pet_store_birds_count :
  let num_cages : ℕ := 8
  let parrots_per_cage : ℕ := 2
  let parakeets_per_cage : ℕ := 7
  let total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)
  total_birds = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_count_l1037_103704


namespace NUMINAMATH_CALUDE_spencer_jump_rope_l1037_103741

def initial_speed : ℕ := 4
def practice_days : List ℕ := [1, 2, 4, 5, 6]
def first_session_duration : ℕ := 10
def second_session_initial : ℕ := 10
def second_session_increase : ℕ := 5

def speed_on_day (day : ℕ) : ℕ :=
  initial_speed * (2^(day - 1))

def second_session_duration (day : ℕ) : ℕ :=
  second_session_initial + (day - 1) * second_session_increase

def jumps_on_day (day : ℕ) : ℕ :=
  speed_on_day day * (first_session_duration + second_session_duration day)

def total_jumps : ℕ :=
  practice_days.map jumps_on_day |>.sum

theorem spencer_jump_rope : total_jumps = 8600 := by
  sorry

end NUMINAMATH_CALUDE_spencer_jump_rope_l1037_103741


namespace NUMINAMATH_CALUDE_garden_expansion_l1037_103781

/-- Given a rectangular garden with dimensions 50 feet by 20 feet, 
    prove that adding 40 feet of fencing and reshaping into a square 
    results in a garden 1025 square feet larger than the original. -/
theorem garden_expansion (original_length : ℝ) (original_width : ℝ) 
  (additional_fence : ℝ) (h1 : original_length = 50) 
  (h2 : original_width = 20) (h3 : additional_fence = 40) : 
  let original_area := original_length * original_width
  let original_perimeter := 2 * (original_length + original_width)
  let new_perimeter := original_perimeter + additional_fence
  let new_side := new_perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 1025 := by
sorry

end NUMINAMATH_CALUDE_garden_expansion_l1037_103781


namespace NUMINAMATH_CALUDE_division_problem_l1037_103722

theorem division_problem : 
  10 / (2 / 0.3) / (0.3 / 0.04) / (0.04 / 0.05) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1037_103722


namespace NUMINAMATH_CALUDE_positive_sum_of_odd_monotonic_increasing_l1037_103795

-- Define a monotonic increasing function
def MonotonicIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem positive_sum_of_odd_monotonic_increasing 
  (f : ℝ → ℝ) 
  (a : ℕ → ℝ) 
  (h_mono : MonotonicIncreasing f) 
  (h_odd : OddFunction f) 
  (h_arith : ArithmeticSequence a) 
  (h_a3_pos : a 3 > 0) : 
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by sorry

end NUMINAMATH_CALUDE_positive_sum_of_odd_monotonic_increasing_l1037_103795


namespace NUMINAMATH_CALUDE_print_shop_cost_difference_l1037_103767

/-- Calculates the total cost for color copies at a print shop --/
def calculate_total_cost (base_price : ℝ) (quantity : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let base_cost := base_price * quantity
  let discounted_cost := if quantity > discount_threshold then base_cost * (1 - discount_rate) else base_cost
  discounted_cost * (1 + tax_rate)

/-- Proves that the difference in cost for 40 color copies between Print Shop Y and Print Shop X is $27.40 --/
theorem print_shop_cost_difference : 
  let shop_x_cost := calculate_total_cost 1.20 40 30 0.10 0.05
  let shop_y_cost := calculate_total_cost 1.70 40 50 0.15 0.07
  shop_y_cost - shop_x_cost = 27.40 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_cost_difference_l1037_103767


namespace NUMINAMATH_CALUDE_positive_difference_of_roots_l1037_103769

theorem positive_difference_of_roots : ∃ (r₁ r₂ : ℝ),
  (r₁^2 - 5*r₁ - 26) / (r₁ + 5) = 3*r₁ + 8 ∧
  (r₂^2 - 5*r₂ - 26) / (r₂ + 5) = 3*r₂ + 8 ∧
  r₁ ≠ r₂ ∧
  |r₁ - r₂| = 8 :=
by sorry

end NUMINAMATH_CALUDE_positive_difference_of_roots_l1037_103769


namespace NUMINAMATH_CALUDE_remainder_of_3_600_mod_17_l1037_103739

theorem remainder_of_3_600_mod_17 : 3^600 % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_600_mod_17_l1037_103739


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_neg_two_l1037_103742

theorem sqrt_meaningful_iff_geq_neg_two (a : ℝ) : 
  (∃ x : ℝ, x^2 = a + 2) ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_neg_two_l1037_103742


namespace NUMINAMATH_CALUDE_distance_along_stream_is_16_l1037_103792

-- Define the boat's speed in still water
def boat_speed : ℝ := 11

-- Define the distance traveled against the stream in one hour
def distance_against_stream : ℝ := 6

-- Define the stream speed
def stream_speed : ℝ := boat_speed - distance_against_stream

-- Define the boat's speed along the stream
def speed_along_stream : ℝ := boat_speed + stream_speed

-- Theorem to prove
theorem distance_along_stream_is_16 : speed_along_stream = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_along_stream_is_16_l1037_103792


namespace NUMINAMATH_CALUDE_cosine_symmetry_and_monotonicity_l1037_103727

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem cosine_symmetry_and_monotonicity (ω : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω x = f ω (3 * Real.pi / 2 - x)) →
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → f ω x ≥ f ω y) →
  ω = 2/3 ∨ ω = 2 := by sorry

end NUMINAMATH_CALUDE_cosine_symmetry_and_monotonicity_l1037_103727


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l1037_103783

def f (x : ℝ) : ℝ := x^5 - x - 1

theorem root_exists_in_interval :
  ∃ r ∈ Set.Ioo 1 2, f r = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l1037_103783


namespace NUMINAMATH_CALUDE_white_balls_count_l1037_103744

theorem white_balls_count (a : ℕ) : 
  (a : ℝ) / (a + 3) = 4/5 → a = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l1037_103744


namespace NUMINAMATH_CALUDE_inequality_preservation_l1037_103786

theorem inequality_preservation (a b c : ℝ) (h : a < b) (h' : b < 0) : a - c < b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1037_103786


namespace NUMINAMATH_CALUDE_alpha_values_l1037_103737

theorem alpha_values (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^3 - 1) = 5 * Complex.abs (α - 1)) :
  α = Complex.I * Real.sqrt 8 ∨ α = -Complex.I * Real.sqrt 8 :=
by sorry

end NUMINAMATH_CALUDE_alpha_values_l1037_103737


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1037_103778

theorem solution_satisfies_system :
  let x : ℝ := 0
  let y : ℝ := 6
  let z : ℝ := 7
  let u : ℝ := 3
  let v : ℝ := -1
  (x - y + z = 1) ∧
  (y - z + u = 2) ∧
  (z - u + v = 3) ∧
  (u - v + x = 4) ∧
  (v - x + y = 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1037_103778


namespace NUMINAMATH_CALUDE_calculation_proof_l1037_103793

theorem calculation_proof :
  (Real.sqrt 8 - abs (-2) + (1/3)⁻¹ - 4 * Real.cos (45 * π / 180)) = 1 ∧
  ∀ x : ℝ, (x - 2)^2 - x*(x - 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1037_103793


namespace NUMINAMATH_CALUDE_rectangle_perimeter_proof_l1037_103700

def square_perimeter : ℝ := 24
def rectangle_width : ℝ := 4

theorem rectangle_perimeter_proof :
  let square_side := square_perimeter / 4
  let square_area := square_side ^ 2
  let rectangle_length := square_area / rectangle_width
  2 * (rectangle_length + rectangle_width) = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_proof_l1037_103700


namespace NUMINAMATH_CALUDE_tangerines_taken_l1037_103772

/-- Represents the contents of Tina's fruit bag -/
structure FruitBag where
  apples : Nat
  oranges : Nat
  tangerines : Nat

/-- Represents the number of fruits taken away -/
structure FruitsTaken where
  oranges : Nat
  tangerines : Nat

def initialBag : FruitBag := { apples := 9, oranges := 5, tangerines := 17 }

def orangesTaken : Nat := 2

theorem tangerines_taken (bag : FruitBag) (taken : FruitsTaken) : 
  bag.oranges - taken.oranges + 4 = bag.tangerines - taken.tangerines →
  taken.tangerines = 10 := by
  sorry

#check tangerines_taken initialBag { oranges := orangesTaken, tangerines := 10 }

end NUMINAMATH_CALUDE_tangerines_taken_l1037_103772


namespace NUMINAMATH_CALUDE_initial_average_production_l1037_103797

/-- Given a company's production data, calculate the initial average daily production. -/
theorem initial_average_production
  (n : ℕ) -- number of days of initial production
  (today_production : ℕ) -- today's production in units
  (new_average : ℚ) -- new average including today's production
  (hn : n = 11)
  (ht : today_production = 110)
  (ha : new_average = 55)
  : (n : ℚ) * (n + 1 : ℚ) * new_average - (n + 1 : ℚ) * today_production = n * 50
  := by sorry

end NUMINAMATH_CALUDE_initial_average_production_l1037_103797


namespace NUMINAMATH_CALUDE_lukes_mother_ten_bills_l1037_103753

def school_fee : ℕ := 350

def mother_fifty : ℕ := 1
def mother_twenty : ℕ := 2

def father_fifty : ℕ := 4
def father_twenty : ℕ := 1
def father_ten : ℕ := 1

theorem lukes_mother_ten_bills (mother_ten : ℕ) :
  mother_fifty * 50 + mother_twenty * 20 + mother_ten * 10 +
  father_fifty * 50 + father_twenty * 20 + father_ten * 10 = school_fee →
  mother_ten = 3 := by
  sorry

end NUMINAMATH_CALUDE_lukes_mother_ten_bills_l1037_103753


namespace NUMINAMATH_CALUDE_two_intersection_points_l1037_103731

/-- Represents a line in the 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a * X + b * Y = c

/-- Represents a point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- The three lines given in the problem --/
def line1 : Line := ⟨4, -3, 2, by sorry⟩
def line2 : Line := ⟨1, 3, 3, by sorry⟩
def line3 : Line := ⟨3, -4, 3, by sorry⟩

/-- Theorem stating that there are exactly two distinct intersection points --/
theorem two_intersection_points : 
  ∃! (p1 p2 : Point), 
    (pointOnLine p1 line1 ∧ pointOnLine p1 line2) ∨ 
    (pointOnLine p1 line1 ∧ pointOnLine p1 line3) ∨ 
    (pointOnLine p1 line2 ∧ pointOnLine p1 line3) ∧
    (pointOnLine p2 line1 ∧ pointOnLine p2 line2) ∨ 
    (pointOnLine p2 line1 ∧ pointOnLine p2 line3) ∨ 
    (pointOnLine p2 line2 ∧ pointOnLine p2 line3) ∧
    p1 ≠ p2 :=
  sorry

end NUMINAMATH_CALUDE_two_intersection_points_l1037_103731


namespace NUMINAMATH_CALUDE_price_increase_percentage_l1037_103718

def lowest_price : ℝ := 18
def highest_price : ℝ := 24

theorem price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 33.33333333333333 := by
sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l1037_103718


namespace NUMINAMATH_CALUDE_A_intersect_B_l1037_103760

def A : Set (ℤ × ℤ) := {(1, 2), (2, 1)}
def B : Set (ℤ × ℤ) := {p : ℤ × ℤ | p.1 - p.2 = 1}

theorem A_intersect_B : A ∩ B = {(2, 1)} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1037_103760


namespace NUMINAMATH_CALUDE_melanie_selling_four_gumballs_l1037_103732

/-- The number of gumballs Melanie is selling -/
def num_gumballs : ℕ := 32 / 8

/-- The price of each gumball in cents -/
def price_per_gumball : ℕ := 8

/-- The total amount Melanie gets from selling gumballs in cents -/
def total_amount : ℕ := 32

/-- Theorem stating that Melanie is selling 4 gumballs -/
theorem melanie_selling_four_gumballs :
  num_gumballs = 4 :=
by sorry

end NUMINAMATH_CALUDE_melanie_selling_four_gumballs_l1037_103732


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l1037_103747

theorem binomial_coefficient_problem (a : ℝ) : 
  (6 : ℕ).choose 1 * a^5 * (Real.sqrt 3 / 6) = -Real.sqrt 3 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l1037_103747


namespace NUMINAMATH_CALUDE_replacement_results_in_four_terms_l1037_103743

-- Define the expression as a function of x and the replacement term
def expression (x : ℝ) (replacement : ℝ → ℝ) : ℝ := 
  (x^3 - 2)^2 + (x^2 + replacement x)^2

-- Define the expansion of the expression
def expanded_expression (x : ℝ) : ℝ := 
  x^6 + x^4 + 4*x^2 + 4

-- Theorem statement
theorem replacement_results_in_four_terms :
  ∀ x : ℝ, expression x (λ y => 2*y) = expanded_expression x :=
by sorry

end NUMINAMATH_CALUDE_replacement_results_in_four_terms_l1037_103743


namespace NUMINAMATH_CALUDE_square_root_sum_l1037_103706

theorem square_root_sum (x : ℝ) :
  (Real.sqrt (64 - x^2) - Real.sqrt (16 - x^2) = 4) →
  (Real.sqrt (64 - x^2) + Real.sqrt (16 - x^2) = 12) :=
by sorry

end NUMINAMATH_CALUDE_square_root_sum_l1037_103706


namespace NUMINAMATH_CALUDE_expand_expression_l1037_103701

theorem expand_expression (x : ℝ) : (7*x + 11 - 3) * 4*x = 28*x^2 + 32*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1037_103701


namespace NUMINAMATH_CALUDE_star_operation_simplification_l1037_103740

/-- The star operation defined as x ★ y = 2x^2 - y -/
def star (x y : ℝ) : ℝ := 2 * x^2 - y

/-- Theorem stating that k ★ (k ★ k) = k -/
theorem star_operation_simplification (k : ℝ) : star k (star k k) = k := by
  sorry

end NUMINAMATH_CALUDE_star_operation_simplification_l1037_103740


namespace NUMINAMATH_CALUDE_max_p_plus_q_l1037_103730

theorem max_p_plus_q (p q : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → 2*p*x^2 + q*x - p + 1 ≥ 0) → 
  p + q ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_p_plus_q_l1037_103730


namespace NUMINAMATH_CALUDE_days_missed_difference_l1037_103703

/-- Represents the number of students who missed a certain number of days -/
structure DaysMissed where
  days : ℕ
  count : ℕ

/-- The histogram data -/
def histogram : List DaysMissed := [
  ⟨0, 3⟩, ⟨1, 1⟩, ⟨2, 4⟩, ⟨3, 1⟩, ⟨4, 1⟩, ⟨5, 5⟩
]

/-- The total number of students -/
def totalStudents : ℕ := 15

/-- Calculates the median number of days missed -/
def median (h : List DaysMissed) (total : ℕ) : ℚ := sorry

/-- Calculates the mean number of days missed -/
def mean (h : List DaysMissed) (total : ℕ) : ℚ := sorry

/-- The main theorem -/
theorem days_missed_difference :
  mean histogram totalStudents - median histogram totalStudents = 11 / 15 := by sorry

end NUMINAMATH_CALUDE_days_missed_difference_l1037_103703
