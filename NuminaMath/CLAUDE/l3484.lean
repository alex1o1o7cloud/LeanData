import Mathlib

namespace NUMINAMATH_CALUDE_small_kite_area_l3484_348484

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a kite given its vertices -/
def kiteArea (a b c d : Point) : ℝ :=
  let base := c.x - a.x
  let height := b.y - a.y
  base * height

/-- The grid spacing in inches -/
def gridSpacing : ℝ := 2

theorem small_kite_area :
  let a := Point.mk 0 6
  let b := Point.mk 3 10
  let c := Point.mk 6 6
  let d := Point.mk 3 0
  kiteArea a b c d * gridSpacing^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_small_kite_area_l3484_348484


namespace NUMINAMATH_CALUDE_ratio_equality_solution_l3484_348460

theorem ratio_equality_solution (x : ℝ) : 
  (4 + 2*x) / (6 + 3*x) = (2 + x) / (3 + 2*x) → x = 0 ∨ x = 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_solution_l3484_348460


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l3484_348421

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation (87.65 * 10^9) = ScientificNotation.mk 8.765 10 sorry := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l3484_348421


namespace NUMINAMATH_CALUDE_greatest_k_for_root_difference_l3484_348479

theorem greatest_k_for_root_difference (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 8 = 0 ∧ 
    x₂^2 + k*x₂ + 8 = 0 ∧ 
    |x₁ - x₂| = 2*Real.sqrt 15) →
  k ≤ Real.sqrt 92 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_for_root_difference_l3484_348479


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3484_348467

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_eq_3 : a + b + c = 3) : 
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3484_348467


namespace NUMINAMATH_CALUDE_quadratic_range_theorem_l3484_348496

/-- The quadratic function f(x) = x^2 + 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 2

/-- A point P with coordinates (m, n) -/
structure Point where
  m : ℝ
  n : ℝ

theorem quadratic_range_theorem (P : Point) 
  (h1 : P.n = f P.m)  -- P lies on the graph of f
  (h2 : |P.m| < 2)    -- distance from P to y-axis is less than 2
  : -1 ≤ P.n ∧ P.n < 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_theorem_l3484_348496


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l3484_348470

theorem gcd_lcm_problem (x y : ℕ+) (u v : ℕ) : 
  (u = Nat.gcd x y ∧ v = Nat.lcm x y) → 
  (x * y * u * v = 3600 ∧ u + v = 32) → 
  ((x = 6 ∧ y = 10) ∨ (x = 10 ∧ y = 6)) ∧ u = 2 ∧ v = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l3484_348470


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l3484_348461

theorem quadratic_roots_sum_and_product :
  let a : ℝ := 2
  let b : ℝ := -10
  let c : ℝ := 12
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  sum_of_roots = 5 ∧ product_of_roots = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l3484_348461


namespace NUMINAMATH_CALUDE_sum_of_products_l3484_348444

theorem sum_of_products : 64 * 46 + 73 * 37 + 82 * 28 + 91 * 19 = 9670 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l3484_348444


namespace NUMINAMATH_CALUDE_price_after_discount_l3484_348404

def original_price : ℕ := 76
def discount : ℕ := 25

theorem price_after_discount :
  original_price - discount = 51 := by sorry

end NUMINAMATH_CALUDE_price_after_discount_l3484_348404


namespace NUMINAMATH_CALUDE_roots_pure_imaginary_for_pure_imaginary_k_l3484_348426

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z k : ℂ) : Prop :=
  8 * z^2 + 6 * i * z - k = 0

-- Define a pure imaginary number
def is_pure_imaginary (x : ℂ) : Prop :=
  x.re = 0 ∧ x.im ≠ 0

-- Define the nature of roots
def roots_are_pure_imaginary (k : ℂ) : Prop :=
  ∀ z : ℂ, quadratic_equation z k → is_pure_imaginary z

-- Theorem statement
theorem roots_pure_imaginary_for_pure_imaginary_k :
  ∀ k : ℂ, is_pure_imaginary k → roots_are_pure_imaginary k :=
by sorry

end NUMINAMATH_CALUDE_roots_pure_imaginary_for_pure_imaginary_k_l3484_348426


namespace NUMINAMATH_CALUDE_cat_owners_count_l3484_348447

/-- Proves that the number of cat owners in a town is 30 -/
theorem cat_owners_count (total_citizens : ℕ) (pet_ownership_rate : ℚ) (dog_ownership_rate : ℚ) : ℕ :=
  by
  sorry

#check cat_owners_count 100 (60/100) (1/2)

end NUMINAMATH_CALUDE_cat_owners_count_l3484_348447


namespace NUMINAMATH_CALUDE_intersection_point_value_l3484_348402

/-- Given three lines that intersect at a single point, prove that the value of a is -1 --/
theorem intersection_point_value (a : ℝ) :
  (∃! p : ℝ × ℝ, (a * p.1 + 2 * p.2 + 8 = 0) ∧
                 (4 * p.1 + 3 * p.2 = 10) ∧
                 (2 * p.1 - p.2 = 10)) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_value_l3484_348402


namespace NUMINAMATH_CALUDE_divisibility_of_product_difference_l3484_348440

theorem divisibility_of_product_difference (a₁ a₂ b₁ b₂ c₁ c₂ d : ℤ) 
  (h1 : d ∣ (a₁ - a₂)) 
  (h2 : d ∣ (b₁ - b₂)) 
  (h3 : d ∣ (c₁ - c₂)) : 
  d ∣ (a₁ * b₁ * c₁ - a₂ * b₂ * c₂) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_product_difference_l3484_348440


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3484_348485

theorem sufficient_not_necessary (a b : ℝ → ℝ) : 
  (∀ x, |a x + b x| + |a x - b x| ≤ 1 → (a x)^2 + (b x)^2 ≤ 1) ∧ 
  (∃ x, (a x)^2 + (b x)^2 ≤ 1 ∧ |a x + b x| + |a x - b x| > 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3484_348485


namespace NUMINAMATH_CALUDE_min_sum_of_radii_l3484_348412

/-
  Define a regular tetrahedron with edge length 1
-/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_unit : edge_length = 1)

/-
  Define a sphere inside the tetrahedron
-/
structure Sphere :=
  (center : ℝ × ℝ × ℝ)
  (radius : ℝ)

/-
  Define the property of a sphere being tangent to three faces of the tetrahedron
-/
def is_tangent_to_three_faces (s : Sphere) (t : RegularTetrahedron) (vertex : ℝ × ℝ × ℝ) : Prop :=
  sorry  -- This would involve complex geometric conditions

/-
  State the theorem
-/
theorem min_sum_of_radii (t : RegularTetrahedron) 
  (s1 s2 : Sphere) 
  (h1 : is_tangent_to_three_faces s1 t (0, 0, 0))  -- Assume A is at (0,0,0)
  (h2 : is_tangent_to_three_faces s2 t (1, 0, 0))  -- Assume B is at (1,0,0)
  : 
  s1.radius + s2.radius ≥ (Real.sqrt 6 - 1) / 5 := by
  sorry


end NUMINAMATH_CALUDE_min_sum_of_radii_l3484_348412


namespace NUMINAMATH_CALUDE_intersection_condition_implies_a_values_l3484_348435

theorem intersection_condition_implies_a_values (a : ℝ) : 
  let M : Set ℝ := {5, a^2 - 3*a + 5}
  let N : Set ℝ := {1, 3}
  (M ∩ N).Nonempty → a = 1 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_implies_a_values_l3484_348435


namespace NUMINAMATH_CALUDE_herman_feeding_months_l3484_348459

/-- The number of months Herman feeds the birds -/
def feeding_months (cups_per_day : ℚ) (total_cups : ℚ) (days_per_month : ℚ) : ℚ :=
  (total_cups / cups_per_day) / days_per_month

theorem herman_feeding_months :
  feeding_months 1 90 30 = 3 := by
  sorry

end NUMINAMATH_CALUDE_herman_feeding_months_l3484_348459


namespace NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l3484_348469

-- Define the basic types
variable (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (at_least_parallel_to_one : Line → Plane → Prop)

-- Define the given conditions
variable (a : Line) (β : Plane)

-- State the theorem
theorem parallel_sufficient_not_necessary :
  (∀ (l : Line), parallel a l → in_plane l β → at_least_parallel_to_one a β) ∧
  (∃ (a : Line) (β : Plane), at_least_parallel_to_one a β ∧ ¬(∃ (l : Line), in_plane l β ∧ parallel a l)) :=
sorry

end NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l3484_348469


namespace NUMINAMATH_CALUDE_min_value_theorem_l3484_348448

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/(y + 1) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 2/(b + 1) = 2 → 2*x + y ≤ 2*a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3484_348448


namespace NUMINAMATH_CALUDE_gcd_lcm_perfect_square_l3484_348495

theorem gcd_lcm_perfect_square (a b c : ℕ+) 
  (h : ∃ k : ℕ, (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a : ℕ) = k^2) : 
  ∃ m : ℕ, (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a : ℕ) = m^2 := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_perfect_square_l3484_348495


namespace NUMINAMATH_CALUDE_packages_to_deliver_l3484_348489

/-- The number of packages received yesterday -/
def packages_yesterday : ℕ := 80

/-- The number of packages received today -/
def packages_today : ℕ := 2 * packages_yesterday

/-- The total number of packages to be delivered tomorrow -/
def total_packages : ℕ := packages_yesterday + packages_today

theorem packages_to_deliver :
  total_packages = 240 :=
sorry

end NUMINAMATH_CALUDE_packages_to_deliver_l3484_348489


namespace NUMINAMATH_CALUDE_market_demand_growth_rate_bound_l3484_348405

theorem market_demand_growth_rate_bound
  (a : Fin 4 → ℝ)  -- Market demand sequence for 4 years
  (p₁ p₂ p₃ : ℝ)   -- Percentage increases between consecutive years
  (h₁ : p₁ + p₂ + p₃ = 1)  -- Condition on percentage increases
  (h₂ : ∀ i : Fin 3, a (i + 1) = a i * (1 + [p₁, p₂, p₃].get i))  -- Relation between consecutive demands
  : ∃ p : ℝ, (∀ i : Fin 3, a (i + 1) = a i * (1 + p)) ∧ p ≤ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_market_demand_growth_rate_bound_l3484_348405


namespace NUMINAMATH_CALUDE_max_product_l3484_348488

theorem max_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 40) :
  x^4 * y^3 ≤ (160/7)^4 * (120/7)^3 ∧
  x^4 * y^3 = (160/7)^4 * (120/7)^3 ↔ x = 160/7 ∧ y = 120/7 := by
  sorry

end NUMINAMATH_CALUDE_max_product_l3484_348488


namespace NUMINAMATH_CALUDE_total_balls_count_l3484_348409

/-- The number of different colors of balls -/
def num_colors : ℕ := 10

/-- The number of balls for each color -/
def balls_per_color : ℕ := 35

/-- The total number of balls -/
def total_balls : ℕ := num_colors * balls_per_color

theorem total_balls_count : total_balls = 350 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_count_l3484_348409


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l3484_348465

/-- The number of weeks it takes to finish a given number of candies with a specific eating pattern -/
def weeks_to_finish (packets : ℕ) (candies_per_packet : ℕ) (weekday_consumption : ℕ) (weekend_consumption : ℕ) : ℕ :=
  let total_candies := packets * candies_per_packet
  let weekly_consumption := weekday_consumption * 5 + weekend_consumption * 2
  total_candies / weekly_consumption

/-- Theorem stating that it takes Bobby 3 weeks to finish his candies given the specified conditions -/
theorem bobby_candy_consumption : 
  weeks_to_finish 2 18 2 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l3484_348465


namespace NUMINAMATH_CALUDE_rotated_line_equation_l3484_348482

/-- Given a line with equation x - y + 1 = 0 and a point P(3, 4) on this line,
    rotating the line 90° counterclockwise around P results in a line with equation x + y - 7 = 0 -/
theorem rotated_line_equation (x y : ℝ) : 
  (x - y + 1 = 0 ∧ 3 - 4 + 1 = 0) → 
  (∃ (m : ℝ), m * (x - 3) + (y - 4) = 0 ∧ m = 1) →
  x + y - 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l3484_348482


namespace NUMINAMATH_CALUDE_mike_ride_distance_l3484_348477

/-- Represents the taxi fare structure -/
structure TaxiFare where
  base_fare : ℚ
  per_mile_rate : ℚ
  toll : ℚ

/-- Calculates the total fare for a given distance -/
def calculate_fare (fare_structure : TaxiFare) (distance : ℚ) : ℚ :=
  fare_structure.base_fare + fare_structure.toll + fare_structure.per_mile_rate * distance

theorem mike_ride_distance (mike_fare annie_fare : TaxiFare) 
  (h1 : mike_fare.base_fare = 2.5)
  (h2 : mike_fare.per_mile_rate = 0.25)
  (h3 : mike_fare.toll = 0)
  (h4 : annie_fare.base_fare = 2.5)
  (h5 : annie_fare.per_mile_rate = 0.25)
  (h6 : annie_fare.toll = 5)
  (h7 : calculate_fare annie_fare 26 = calculate_fare mike_fare (46 : ℚ)) :
  ∃ x : ℚ, calculate_fare mike_fare x = calculate_fare annie_fare 26 ∧ x = 46 := by
  sorry

#eval (46 : ℚ)

end NUMINAMATH_CALUDE_mike_ride_distance_l3484_348477


namespace NUMINAMATH_CALUDE_total_leaves_l3484_348408

/-- The number of leaves Sabrina needs for her poultice --/
structure HerbLeaves where
  basil : ℕ
  sage : ℕ
  verbena : ℕ
  chamomile : ℕ
  lavender : ℕ

/-- The conditions for Sabrina's herb collection --/
def validHerbCollection (h : HerbLeaves) : Prop :=
  h.basil = 3 * h.sage ∧
  h.verbena = h.sage + 8 ∧
  h.chamomile = 2 * h.sage + 7 ∧
  h.lavender = (h.basil + h.chamomile + 1) / 2 ∧
  h.basil = 48

/-- The theorem stating the total number of leaves needed --/
theorem total_leaves (h : HerbLeaves) (hvalid : validHerbCollection h) :
  h.basil + h.sage + h.verbena + h.chamomile + h.lavender = 171 := by
  sorry

#check total_leaves

end NUMINAMATH_CALUDE_total_leaves_l3484_348408


namespace NUMINAMATH_CALUDE_ratio_simplification_and_increase_l3484_348449

def original_ratio : List Nat := [4, 16, 20, 12]

def gcd_list (l : List Nat) : Nat :=
  l.foldl Nat.gcd 0

def simplify_ratio (l : List Nat) : List Nat :=
  let gcd := gcd_list l
  l.map (·/gcd)

def percentage_increase (first last : Nat) : Nat :=
  ((last - first) * 100) / first

theorem ratio_simplification_and_increase :
  let simplified := simplify_ratio original_ratio
  simplified = [1, 4, 5, 3] ∧
  percentage_increase simplified.head! simplified.getLast! = 200 := by
  sorry

end NUMINAMATH_CALUDE_ratio_simplification_and_increase_l3484_348449


namespace NUMINAMATH_CALUDE_lowry_earnings_l3484_348480

/-- Calculates the total earnings from bonsai sales with discounts applied --/
def bonsai_earnings (small_price medium_price big_price : ℚ)
                    (small_discount medium_discount big_discount : ℚ)
                    (small_count medium_count big_count : ℕ)
                    (small_discount_threshold medium_discount_threshold big_discount_threshold : ℕ) : ℚ :=
  let small_total := small_price * small_count
  let medium_total := medium_price * medium_count
  let big_total := big_price * big_count
  let small_discounted := if small_count ≥ small_discount_threshold then small_total * (1 - small_discount) else small_total
  let medium_discounted := if medium_count ≥ medium_discount_threshold then medium_total * (1 - medium_discount) else medium_total
  let big_discounted := if big_count > big_discount_threshold then big_total * (1 - big_discount) else big_total
  small_discounted + medium_discounted + big_discounted

theorem lowry_earnings :
  bonsai_earnings 30 45 60 0.1 0.15 0.05 8 5 7 4 3 5 = 806.25 := by
  sorry

end NUMINAMATH_CALUDE_lowry_earnings_l3484_348480


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_solution_set_correct_l3484_348454

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- The vectors a and b as functions of x -/
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (x - 1, 2)

/-- Theorem stating the conditions for a and b to be parallel -/
theorem parallel_vectors_condition :
  ∀ x : ℝ, are_parallel (a x) (b x) ↔ x = 2 ∨ x = -1 :=
by
  sorry

/-- The solution set for x -/
def solution_set : Set ℝ := {2, -1}

/-- Theorem stating that the solution set is correct -/
theorem solution_set_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ are_parallel (a x) (b x) :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_solution_set_correct_l3484_348454


namespace NUMINAMATH_CALUDE_chipped_marbles_bag_l3484_348453

/-- Represents the number of marbles in each bag -/
def bags : List Nat := [15, 18, 22, 24, 30]

/-- Represents the total number of marbles -/
def total : Nat := bags.sum

/-- Predicate to check if a list of two numbers from the bags list sums to a given value -/
def hasTwoSum (s : Nat) : Prop := ∃ (a b : Nat), a ∈ bags ∧ b ∈ bags ∧ a ≠ b ∧ a + b = s

/-- The main theorem stating that the bag with chipped marbles contains 24 marbles -/
theorem chipped_marbles_bag : 
  ∃ (jane george : Nat), 
    jane ∈ bags ∧ 
    george ∈ bags ∧ 
    jane ≠ george ∧
    hasTwoSum jane ∧ 
    hasTwoSum george ∧ 
    jane = 3 * george ∧ 
    total - jane - george = 24 := by
  sorry

end NUMINAMATH_CALUDE_chipped_marbles_bag_l3484_348453


namespace NUMINAMATH_CALUDE_cube_root_plus_sqrt_minus_sqrt_l3484_348413

theorem cube_root_plus_sqrt_minus_sqrt : ∃ x y z : ℝ, x^3 = -64 ∧ y^2 = 9 ∧ z^2 = 25/16 ∧ x + y - z = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_plus_sqrt_minus_sqrt_l3484_348413


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l3484_348494

theorem fractional_inequality_solution_set (x : ℝ) :
  1 / (x - 1) < -1 ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l3484_348494


namespace NUMINAMATH_CALUDE_peanut_cluster_probability_l3484_348441

def total_chocolates : ℕ := 50
def caramels : ℕ := 3
def nougats : ℕ := 2 * caramels
def truffles : ℕ := caramels + 6
def peanut_clusters : ℕ := total_chocolates - (caramels + nougats + truffles)

theorem peanut_cluster_probability : 
  (peanut_clusters : ℚ) / total_chocolates = 32 / 50 := by sorry

end NUMINAMATH_CALUDE_peanut_cluster_probability_l3484_348441


namespace NUMINAMATH_CALUDE_calculator_theorem_l3484_348407

/-- Represents the state of the calculator as a 4-tuple of real numbers -/
def CalculatorState := Fin 4 → ℝ

/-- Applies the transformation to a given state -/
def transform (s : CalculatorState) : CalculatorState :=
  fun i => match i with
  | 0 => s 0 - s 1
  | 1 => s 1 - s 2
  | 2 => s 2 - s 3
  | 3 => s 3 - s 0

/-- Applies the transformation n times to a given state -/
def transformN (s : CalculatorState) (n : ℕ) : CalculatorState :=
  match n with
  | 0 => s
  | n + 1 => transform (transformN s n)

/-- Checks if any number in the state is greater than 1985 -/
def hasLargeNumber (s : CalculatorState) : Prop :=
  ∃ i : Fin 4, s i > 1985

/-- Main theorem statement -/
theorem calculator_theorem (s : CalculatorState) 
  (h : ∃ i j : Fin 4, s i ≠ s j) : 
  ∃ n : ℕ, hasLargeNumber (transformN s n) := by
sorry

end NUMINAMATH_CALUDE_calculator_theorem_l3484_348407


namespace NUMINAMATH_CALUDE_fraction_equality_l3484_348427

theorem fraction_equality (a b : ℝ) (h : 2 * a = 5 * b) : a / b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3484_348427


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3484_348491

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  students_per_group : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Theorem: In a systematic sampling of 50 students into 5 groups of 10 each,
    if the student with number 22 is selected from the third group,
    then the student with number 42 will be selected from the fifth group. -/
theorem systematic_sampling_theorem (s : SystematicSampling)
    (h1 : s.total_students = 50)
    (h2 : s.num_groups = 5)
    (h3 : s.students_per_group = 10)
    (h4 : s.selected_number = 22)
    (h5 : s.selected_group = 3) :
    s.selected_number + (s.num_groups - s.selected_group) * s.students_per_group = 42 :=
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3484_348491


namespace NUMINAMATH_CALUDE_sum_remainder_l3484_348400

theorem sum_remainder (n : ℤ) : (7 - 2*n + (n + 5)) % 8 = (4 - n) % 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l3484_348400


namespace NUMINAMATH_CALUDE_squirrel_climb_l3484_348437

theorem squirrel_climb (x : ℝ) : 
  (∀ n : ℕ, n > 0 → (2 * n - 1) * x - 2 * (n - 1) = 26) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_climb_l3484_348437


namespace NUMINAMATH_CALUDE_accuracy_of_3_145e8_l3484_348473

/-- Represents the level of accuracy for a number -/
inductive Accuracy
  | HundredThousand
  | Million
  | TenMillion
  | HundredMillion

/-- Determines the accuracy of a number in scientific notation -/
def accuracy_of_scientific_notation (mantissa : Float) (exponent : Int) : Accuracy :=
  match exponent with
  | 8 => Accuracy.HundredThousand
  | 9 => Accuracy.Million
  | 10 => Accuracy.TenMillion
  | 11 => Accuracy.HundredMillion
  | _ => Accuracy.HundredThousand  -- Default case

theorem accuracy_of_3_145e8 :
  accuracy_of_scientific_notation 3.145 8 = Accuracy.HundredThousand :=
by sorry

end NUMINAMATH_CALUDE_accuracy_of_3_145e8_l3484_348473


namespace NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_proof_l3484_348410

/-- A parallelogram with vertices at (0, 0), (7, 0), (3, 5), and (10, 5) has an area of 35 square units. -/
theorem parallelogram_area : ℝ → Prop := fun area =>
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (7, 0)
  let v3 : ℝ × ℝ := (3, 5)
  let v4 : ℝ × ℝ := (10, 5)
  area = 35

/-- The proof of the parallelogram area theorem. -/
theorem parallelogram_area_proof : parallelogram_area 35 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_proof_l3484_348410


namespace NUMINAMATH_CALUDE_square_1600_product_l3484_348417

theorem square_1600_product (x : ℤ) (h : x^2 = 1600) : (x + 2) * (x - 2) = 1596 := by
  sorry

end NUMINAMATH_CALUDE_square_1600_product_l3484_348417


namespace NUMINAMATH_CALUDE_initial_games_count_l3484_348481

theorem initial_games_count (initial remaining given : ℕ) : 
  remaining = initial - given → 
  given = 7 → 
  remaining = 91 → 
  initial = 98 := by sorry

end NUMINAMATH_CALUDE_initial_games_count_l3484_348481


namespace NUMINAMATH_CALUDE_clock_distance_theorem_l3484_348432

/-- Represents a clock on the table -/
structure Clock where
  center : ℝ × ℝ
  radius : ℝ

/-- The state of all clocks at a given time -/
def ClockState := List Clock

/-- Calculate the position of the minute hand at a given time -/
def minuteHandPosition (clock : Clock) (time : ℝ) : ℝ × ℝ :=
  sorry

/-- Calculate the sum of distances from the table center to clock centers -/
def sumDistancesToCenters (clocks : ClockState) : ℝ :=
  sorry

/-- Calculate the sum of distances from the table center to minute hand ends at a given time -/
def sumDistancesToMinuteHands (clocks : ClockState) (time : ℝ) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem clock_distance_theorem (clocks : ClockState) (h : clocks.length = 50) :
  ∃ t : ℝ, sumDistancesToMinuteHands clocks t > sumDistancesToCenters clocks :=
sorry

end NUMINAMATH_CALUDE_clock_distance_theorem_l3484_348432


namespace NUMINAMATH_CALUDE_riddle_guessing_probabilities_l3484_348476

-- Define the probabilities of A and B guessing correctly
def prob_A_correct : ℚ := 5/6
def prob_B_correct : ℚ := 3/5

-- Define the probability of A winning in one activity
def prob_A_wins_one : ℚ := prob_A_correct * (1 - prob_B_correct)

-- Define the probability of A winning at least 2 out of 3 activities
def prob_A_wins_two_out_of_three : ℚ :=
  3 * (prob_A_wins_one^2 * (1 - prob_A_wins_one)) + prob_A_wins_one^3

-- State the theorem
theorem riddle_guessing_probabilities :
  prob_A_wins_one = 1/3 ∧ prob_A_wins_two_out_of_three = 7/27 := by
  sorry

end NUMINAMATH_CALUDE_riddle_guessing_probabilities_l3484_348476


namespace NUMINAMATH_CALUDE_sprint_race_losing_distance_l3484_348433

/-- Represents a sprint race between Kelly and Abel -/
structure SprintRace where
  raceLength : ℝ
  headStart : ℝ
  extraDistanceToOvertake : ℝ

/-- Calculates the distance by which Abel lost the race to Kelly -/
def losingDistance (race : SprintRace) : ℝ :=
  race.headStart + race.extraDistanceToOvertake

theorem sprint_race_losing_distance : 
  let race : SprintRace := {
    raceLength := 100,
    headStart := 3,
    extraDistanceToOvertake := 19.9
  }
  losingDistance race = 22.9 := by sorry

end NUMINAMATH_CALUDE_sprint_race_losing_distance_l3484_348433


namespace NUMINAMATH_CALUDE_jeans_cost_proof_l3484_348478

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.50
def quarters_left : ℕ := 97

def jeans_cost : ℚ := initial_amount - pizza_cost - soda_cost - (quarters_left : ℚ) * (1 / 4)

theorem jeans_cost_proof : jeans_cost = 11.50 := by
  sorry

end NUMINAMATH_CALUDE_jeans_cost_proof_l3484_348478


namespace NUMINAMATH_CALUDE_existence_of_special_set_l3484_348443

theorem existence_of_special_set (n : ℕ) (p : ℕ) (h_n : n ≥ 2) (h_p : Nat.Prime p) (h_div : p ∣ n) :
  ∃ (A : Fin n → ℕ), ∀ (i j : Fin n) (S : Finset (Fin n)), S.card = p →
    (A i * A j) ∣ (S.sum A) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l3484_348443


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l3484_348456

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis. -/
def symmetricToYAxis (a b : Point) : Prop :=
  b.x = -a.x ∧ b.y = a.y

/-- The theorem stating that if A(2, -5) is symmetric to B with respect to the y-axis,
    then B has coordinates (-2, -5). -/
theorem symmetry_coordinates :
  let a : Point := ⟨2, -5⟩
  let b : Point := ⟨-2, -5⟩
  symmetricToYAxis a b → b = ⟨-2, -5⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l3484_348456


namespace NUMINAMATH_CALUDE_exactlyOnePen_exactlyTwoPencils_mutually_exclusive_not_opposite_l3484_348468

/-- Represents the contents of a pencil case -/
structure PencilCase where
  pencils : ℕ
  pens : ℕ

/-- Represents the outcome of selecting two items from a pencil case -/
inductive Selection
  | TwoPencils
  | OnePencilOnePen
  | TwoPens

/-- Defines the pencil case with 2 pencils and 2 pens -/
def myPencilCase : PencilCase := { pencils := 2, pens := 2 }

/-- Event: Exactly 1 pen is selected -/
def exactlyOnePen (s : Selection) : Prop :=
  s = Selection.OnePencilOnePen

/-- Event: Exactly 2 pencils are selected -/
def exactlyTwoPencils (s : Selection) : Prop :=
  s = Selection.TwoPencils

/-- Two events are mutually exclusive -/
def mutuallyExclusive (e1 e2 : Selection → Prop) : Prop :=
  ∀ s, ¬(e1 s ∧ e2 s)

/-- Two events are opposite -/
def opposite (e1 e2 : Selection → Prop) : Prop :=
  ∀ s, e1 s ↔ ¬(e2 s)

theorem exactlyOnePen_exactlyTwoPencils_mutually_exclusive_not_opposite :
  mutuallyExclusive exactlyOnePen exactlyTwoPencils ∧
  ¬(opposite exactlyOnePen exactlyTwoPencils) :=
by sorry

end NUMINAMATH_CALUDE_exactlyOnePen_exactlyTwoPencils_mutually_exclusive_not_opposite_l3484_348468


namespace NUMINAMATH_CALUDE_factor_polynomial_l3484_348486

theorem factor_polynomial (x : ℝ) : 80 * x^5 - 250 * x^9 = -10 * x^5 * (25 * x^4 - 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3484_348486


namespace NUMINAMATH_CALUDE_valid_placements_count_l3484_348424

/-- Represents a grid with rows and columns -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the placement of crosses in a grid -/
structure CrossPlacement :=
  (grid : Grid)
  (num_crosses : ℕ)

/-- Counts the number of valid cross placements in a grid -/
def count_valid_placements (cp : CrossPlacement) : ℕ :=
  sorry

/-- The specific grid and cross placement for our problem -/
def our_problem : CrossPlacement :=
  { grid := { rows := 3, cols := 4 },
    num_crosses := 4 }

/-- Theorem stating that the number of valid placements for our problem is 36 -/
theorem valid_placements_count :
  count_valid_placements our_problem = 36 :=
sorry

end NUMINAMATH_CALUDE_valid_placements_count_l3484_348424


namespace NUMINAMATH_CALUDE_total_caps_produced_l3484_348463

def week1_production : ℕ := 320
def week2_production : ℕ := 400
def week3_production : ℕ := 300

def average_production : ℕ := (week1_production + week2_production + week3_production) / 3

def total_production : ℕ := week1_production + week2_production + week3_production + average_production

theorem total_caps_produced : total_production = 1360 := by
  sorry

end NUMINAMATH_CALUDE_total_caps_produced_l3484_348463


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l3484_348457

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def total_cost : ℚ := 33.56

theorem jacket_cost_calculation :
  total_cost - shorts_cost - shirt_cost = 7.43 := by sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l3484_348457


namespace NUMINAMATH_CALUDE_max_consecutive_sum_less_than_1000_l3484_348462

theorem max_consecutive_sum_less_than_1000 :
  ∀ n : ℕ, (n * (n + 1)) / 2 < 1000 ↔ n ≤ 44 :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_less_than_1000_l3484_348462


namespace NUMINAMATH_CALUDE_edges_ge_twice_faces_l3484_348429

/-- A bipartite planar graph. -/
structure BipartitePlanarGraph where
  V : Type* -- Vertices
  E : Type* -- Edges
  F : Type* -- Faces
  edge_count : ℕ
  face_count : ℕ
  is_bipartite : Prop
  is_planar : Prop
  edge_count_ge_two : edge_count ≥ 2

/-- Theorem: In a bipartite planar graph with at least 2 edges, 
    the number of edges is at least twice the number of faces. -/
theorem edges_ge_twice_faces (G : BipartitePlanarGraph) : 
  G.edge_count ≥ 2 * G.face_count := by
  sorry

end NUMINAMATH_CALUDE_edges_ge_twice_faces_l3484_348429


namespace NUMINAMATH_CALUDE_probability_of_valid_selection_l3484_348442

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def valid_selection (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 12 ∧
  1 ≤ b ∧ b ≤ 12 ∧
  1 ≤ c ∧ c ≤ 12 ∧
  1 ≤ d ∧ d ≤ 12 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  is_multiple a b ∧ is_multiple b c ∧ is_multiple c d

def count_valid_selections : ℕ := sorry

theorem probability_of_valid_selection :
  (count_valid_selections : ℚ) / (12 * 11 * 10 * 9) = 13 / 845 := by sorry

end NUMINAMATH_CALUDE_probability_of_valid_selection_l3484_348442


namespace NUMINAMATH_CALUDE_fraction_irreducible_l3484_348466

theorem fraction_irreducible (n : ℤ) : Int.gcd (21*n + 4) (14*n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l3484_348466


namespace NUMINAMATH_CALUDE_min_sheets_for_boats_is_one_l3484_348419

/-- The minimum number of sheets needed to make paper boats -/
def min_sheets_for_boats : ℕ := 1

/-- The total number of paper toys to be made -/
def total_toys : ℕ := 250

/-- The number of paper boats that can be made from one sheet -/
def boats_per_sheet : ℕ := 9

/-- The number of paper planes that can be made from one sheet -/
def planes_per_sheet : ℕ := 5

/-- The number of paper helicopters that can be made from one sheet -/
def helicopters_per_sheet : ℕ := 3

/-- Theorem stating that the minimum number of sheets needed for paper boats is 1 -/
theorem min_sheets_for_boats_is_one :
  ∃ (boats planes helicopters : ℕ),
    boats + planes + helicopters = total_toys ∧
    boats ≤ min_sheets_for_boats * boats_per_sheet ∧
    planes ≤ (total_toys / helicopters_per_sheet) * planes_per_sheet ∧
    helicopters = (total_toys / helicopters_per_sheet) * helicopters_per_sheet :=
by sorry

end NUMINAMATH_CALUDE_min_sheets_for_boats_is_one_l3484_348419


namespace NUMINAMATH_CALUDE_power_product_equality_l3484_348474

theorem power_product_equality (x : ℝ) (h : x > 0) : 
  x^x * x^x = x^(2*x) ∧ x^x * x^x = (x^2)^x :=
by sorry

end NUMINAMATH_CALUDE_power_product_equality_l3484_348474


namespace NUMINAMATH_CALUDE_vector_equation_l3484_348446

def planar_vector := ℝ × ℝ

def scalar_mult (k : ℝ) (v : planar_vector) : planar_vector :=
  (k * v.1, k * v.2)

theorem vector_equation (a b : planar_vector) :
  b = scalar_mult 2 a → a = (1, 2) → b = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_l3484_348446


namespace NUMINAMATH_CALUDE_hyperbola_quadrants_l3484_348403

theorem hyperbola_quadrants (k : ℝ) : k < 0 ∧ 2 * k^2 + k - 2 = -1 → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_quadrants_l3484_348403


namespace NUMINAMATH_CALUDE_total_donation_is_65_inches_l3484_348416

/-- Represents the hair donation of a person -/
structure HairDonation where
  initialLength : ℕ
  keptLength : ℕ
  donatedLength : ℕ
  donation_calculation : donatedLength = initialLength - keptLength

/-- The total hair donation of the five friends -/
def totalDonation (isabella damien ella toby lisa : HairDonation) : ℕ :=
  isabella.donatedLength + damien.donatedLength + ella.donatedLength + toby.donatedLength + lisa.donatedLength

/-- Theorem stating the total hair donation is 65 inches -/
theorem total_donation_is_65_inches : 
  ∃ (isabella damien ella toby lisa : HairDonation),
    isabella.initialLength = 18 ∧ isabella.keptLength = 9 ∧
    damien.initialLength = 24 ∧ damien.keptLength = 12 ∧
    ella.initialLength = 30 ∧ ella.keptLength = 10 ∧
    toby.initialLength = 16 ∧ toby.keptLength = 0 ∧
    lisa.initialLength = 28 ∧ lisa.donatedLength = 8 ∧
    totalDonation isabella damien ella toby lisa = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_donation_is_65_inches_l3484_348416


namespace NUMINAMATH_CALUDE_expression_evaluation_l3484_348422

theorem expression_evaluation :
  let a : ℝ := Real.sqrt 2 - 3
  (2*a + Real.sqrt 3) * (2*a - Real.sqrt 3) - 3*a*(a - 2) + 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3484_348422


namespace NUMINAMATH_CALUDE_solution_interval_l3484_348498

theorem solution_interval (x : ℝ) : (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Ici (-4) ∩ Set.Iio (-1) := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l3484_348498


namespace NUMINAMATH_CALUDE_salary_changes_l3484_348415

/-- Represents the series of salary changes and calculates the final salary --/
def final_salary (original : ℝ) : ℝ :=
  let after_first_raise := original * 1.12
  let after_reduction := after_first_raise * 0.93
  let after_bonus := after_reduction * 1.15
  let fixed_component := after_bonus * 0.7
  let variable_component := after_bonus * 0.3 * 0.9
  fixed_component + variable_component

/-- Theorem stating that an original salary of approximately 7041.77 results in a final salary of 7600.35 --/
theorem salary_changes (ε : ℝ) (hε : ε > 0) :
  ∃ (original : ℝ), abs (original - 7041.77) < ε ∧ final_salary original = 7600.35 := by
  sorry

end NUMINAMATH_CALUDE_salary_changes_l3484_348415


namespace NUMINAMATH_CALUDE_mary_marbles_l3484_348472

def dan_marbles : ℕ := 5
def mary_multiplier : ℕ := 2

theorem mary_marbles : 
  dan_marbles * mary_multiplier = 10 := by sorry

end NUMINAMATH_CALUDE_mary_marbles_l3484_348472


namespace NUMINAMATH_CALUDE_solve_for_m_l3484_348428

theorem solve_for_m : ∃ m : ℚ, 
  (∃ x y : ℚ, 3 * x - 4 * (m - 1) * y + 30 = 0 ∧ x = 2 ∧ y = -3) → 
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3484_348428


namespace NUMINAMATH_CALUDE_amy_work_hours_l3484_348487

theorem amy_work_hours (hourly_wage : ℝ) (tips : ℝ) (total_earnings : ℝ) : 
  hourly_wage = 2 → tips = 9 → total_earnings = 23 → 
  ∃ h : ℝ, h * hourly_wage + tips = total_earnings ∧ h = 7 := by
sorry

end NUMINAMATH_CALUDE_amy_work_hours_l3484_348487


namespace NUMINAMATH_CALUDE_complex_alpha_value_l3484_348451

theorem complex_alpha_value (α β : ℂ) 
  (h1 : (α + 2*β).im = 0)
  (h2 : (α - Complex.I * (3*β - α)).im = 0)
  (h3 : β = 2 + 3*Complex.I) : 
  α = 6 - 6*Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_alpha_value_l3484_348451


namespace NUMINAMATH_CALUDE_manufacturing_cost_is_210_l3484_348418

/-- Calculates the manufacturing cost of a shoe given transportation cost, selling price, and gain percentage. -/
def manufacturing_cost (transportation_cost : ℚ) (shoes_per_transport : ℕ) (selling_price : ℚ) (gain_percentage : ℚ) : ℚ :=
  let transportation_cost_per_shoe := transportation_cost / shoes_per_transport
  let cost_price := selling_price / (1 + gain_percentage)
  cost_price - transportation_cost_per_shoe

/-- Proves that the manufacturing cost of a shoe is 210, given the specified conditions. -/
theorem manufacturing_cost_is_210 :
  manufacturing_cost 500 100 258 (20/100) = 210 := by
  sorry

#eval manufacturing_cost 500 100 258 (20/100)

end NUMINAMATH_CALUDE_manufacturing_cost_is_210_l3484_348418


namespace NUMINAMATH_CALUDE_fraction_equality_l3484_348420

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 40)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 8) :
  m / q = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3484_348420


namespace NUMINAMATH_CALUDE_junior_trip_fraction_l3484_348438

theorem junior_trip_fraction (S J : ℚ) 
  (h1 : J = 2/3 * S) 
  (h2 : 2/3 * S + x * J = 1/2 * (S + J)) 
  (h3 : S > 0) 
  (h4 : J > 0) : 
  x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_junior_trip_fraction_l3484_348438


namespace NUMINAMATH_CALUDE_apple_boxes_count_apple_boxes_count_specific_l3484_348434

theorem apple_boxes_count (apples_per_crate : ℕ) (crates_delivered : ℕ) 
  (rotten_apples : ℕ) (apples_per_box : ℕ) : ℕ :=
  let total_apples := apples_per_crate * crates_delivered
  let remaining_apples := total_apples - rotten_apples
  remaining_apples / apples_per_box

theorem apple_boxes_count_specific : 
  apple_boxes_count 180 12 160 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_apple_boxes_count_apple_boxes_count_specific_l3484_348434


namespace NUMINAMATH_CALUDE_library_visit_theorem_l3484_348430

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a library visitation schedule -/
structure Schedule where
  interval : Nat
  startDay : DayOfWeek

/-- Calculates the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday    => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday   => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday  => DayOfWeek.Friday
  | DayOfWeek.Friday    => DayOfWeek.Saturday
  | DayOfWeek.Saturday  => DayOfWeek.Sunday
  | DayOfWeek.Sunday    => DayOfWeek.Monday

/-- Calculates the day of the week after a given number of days -/
def addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (addDays d m)

/-- Theorem: Given the conditions, the initial conversation occurred on a Thursday -/
theorem library_visit_theorem 
  (boy1 : Schedule) 
  (boy2 : Schedule) 
  (boy3 : Schedule) 
  (h1 : boy1.interval = 2)
  (h2 : boy2.interval = 3)
  (h3 : boy3.interval = 4)
  (h4 : addDays boy1.startDay 12 = DayOfWeek.Monday)
  (h5 : addDays boy2.startDay 12 = DayOfWeek.Monday)
  (h6 : addDays boy3.startDay 12 = DayOfWeek.Monday) :
  boy1.startDay = DayOfWeek.Thursday ∧ 
  boy2.startDay = DayOfWeek.Thursday ∧ 
  boy3.startDay = DayOfWeek.Thursday :=
sorry


end NUMINAMATH_CALUDE_library_visit_theorem_l3484_348430


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3484_348431

theorem inequality_solution_implies_m_value (m : ℝ) :
  (∀ x : ℝ, mx + 2 > 0 ↔ x < 2) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3484_348431


namespace NUMINAMATH_CALUDE_jar_water_problem_l3484_348423

theorem jar_water_problem (s l w : ℚ) : 
  s > 0 ∧ l > 0 ∧ s < l ∧ w > 0 →  -- s: smaller jar capacity, l: larger jar capacity, w: water amount
  w = (1/6) * s ∧ w = (1/5) * l → 
  (2 * w) / l = 2/5 := by sorry

end NUMINAMATH_CALUDE_jar_water_problem_l3484_348423


namespace NUMINAMATH_CALUDE_simplify_expression_l3484_348455

theorem simplify_expression (x : ℝ) (h : x ≠ -1) :
  (x - 1 - 8 / (x + 1)) / ((x + 3) / (x + 1)) = x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3484_348455


namespace NUMINAMATH_CALUDE_log_cos_sum_squared_l3484_348492

theorem log_cos_sum_squared : 
  (Real.log (Real.cos (20 * π / 180)) / Real.log (Real.sqrt 2) + 
   Real.log (Real.cos (40 * π / 180)) / Real.log (Real.sqrt 2) + 
   Real.log (Real.cos (80 * π / 180)) / Real.log (Real.sqrt 2)) ^ 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_log_cos_sum_squared_l3484_348492


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l3484_348497

theorem vector_subtraction_and_scalar_multiplication :
  (⟨2, -5⟩ : ℝ × ℝ) - 4 • (⟨-1, 7⟩ : ℝ × ℝ) = (⟨6, -33⟩ : ℝ × ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l3484_348497


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l3484_348475

/-- The cost of one chocolate bar given the conditions of the problem -/
theorem chocolate_bar_cost : 
  ∀ (scouts : ℕ) (smores_per_scout : ℕ) (smores_per_bar : ℕ) (total_cost : ℚ),
  scouts = 15 →
  smores_per_scout = 2 →
  smores_per_bar = 3 →
  total_cost = 15 →
  (total_cost / (scouts * smores_per_scout / smores_per_bar : ℚ)) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l3484_348475


namespace NUMINAMATH_CALUDE_wall_width_l3484_348483

theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 4 * w →
  l = 3 * h →
  volume = w * h * l →
  volume = 10368 →
  w = 6 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_l3484_348483


namespace NUMINAMATH_CALUDE_cos_arithmetic_sequence_product_l3484_348411

theorem cos_arithmetic_sequence_product (a₁ : ℝ) : 
  let a : ℕ+ → ℝ := λ n => a₁ + (2 * π / 3) * (n.val - 1)
  let S : Set ℝ := {x | ∃ n : ℕ+, x = Real.cos (a n)}
  (∃ a b : ℝ, S = {a, b} ∧ a ≠ b) → 
  ∃ a b : ℝ, S = {a, b} ∧ a * b = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_cos_arithmetic_sequence_product_l3484_348411


namespace NUMINAMATH_CALUDE_office_meeting_reduction_l3484_348450

theorem office_meeting_reduction (total_people : ℕ) (women_in_meeting : ℕ) : 
  total_people = 60 → 
  women_in_meeting = 6 → 
  (women_in_meeting : ℚ) / (total_people / 2 : ℚ) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_office_meeting_reduction_l3484_348450


namespace NUMINAMATH_CALUDE_dress_shop_inventory_l3484_348471

/-- Proves that given a total space of 200 dresses and 83 red dresses,
    the number of additional blue dresses compared to red dresses is 34. -/
theorem dress_shop_inventory (total_space : Nat) (red_dresses : Nat)
    (h1 : total_space = 200)
    (h2 : red_dresses = 83) :
    total_space - red_dresses - red_dresses = 34 := by
  sorry

end NUMINAMATH_CALUDE_dress_shop_inventory_l3484_348471


namespace NUMINAMATH_CALUDE_city_population_l3484_348493

/-- Proves that given the conditions about women and retail workers in a city,
    the total population is 6,000,000 -/
theorem city_population (total_population : ℕ) : 
  (total_population / 2 : ℚ) = (total_population : ℚ) * (1 / 2 : ℚ) →
  ((total_population / 2 : ℚ) * (1 / 3 : ℚ) : ℚ) = 1000000 →
  total_population = 6000000 := by
  sorry

end NUMINAMATH_CALUDE_city_population_l3484_348493


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l3484_348406

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (40 * π / 180) + Real.tan (60 * π / 180)) / Real.cos (10 * π / 180) =
  Real.sqrt 3 * ((1/2 * Real.cos (10 * π / 180) + Real.sqrt 3 / 2) / (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (40 * π / 180))) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l3484_348406


namespace NUMINAMATH_CALUDE_number_of_puppies_number_of_puppies_proof_l3484_348436

/-- The number of puppies at a camp, given specific feeding conditions for dogs and puppies -/
theorem number_of_puppies : ℕ :=
  let num_dogs : ℕ := 3
  let dog_meal_size : ℚ := 4
  let dog_meals_per_day : ℕ := 3
  let total_food_per_day : ℚ := 108
  let dog_to_puppy_meal_ratio : ℚ := 2
  let puppy_to_dog_meal_frequency_ratio : ℕ := 3
  4

/-- Proof that the number of puppies is correct -/
theorem number_of_puppies_proof :
  let num_dogs : ℕ := 3
  let dog_meal_size : ℚ := 4
  let dog_meals_per_day : ℕ := 3
  let total_food_per_day : ℚ := 108
  let dog_to_puppy_meal_ratio : ℚ := 2
  let puppy_to_dog_meal_frequency_ratio : ℕ := 3
  number_of_puppies = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_puppies_number_of_puppies_proof_l3484_348436


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3484_348445

/-- Given a geometric sequence with first term 2 and fifth term 18, the third term is 6 -/
theorem geometric_sequence_third_term : ∀ (x y z : ℝ),
  (∃ (q : ℝ), q ≠ 0 ∧ 
    2 * q = x ∧
    2 * q^2 = y ∧
    2 * q^3 = z ∧
    2 * q^4 = 18) →
  y = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3484_348445


namespace NUMINAMATH_CALUDE_fibFactLastTwoDigitsSum_l3484_348499

/-- The Fibonacci Factorial Series up to 144 -/
def fibFactSeries : List ℕ := [1, 1, 2, 3, 8, 13, 21, 34, 55, 89, 144]

/-- Function to calculate the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Function to get the last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ :=
  n % 100

/-- Theorem stating that the sum of the last two digits of the Fibonacci Factorial Series is 30 -/
theorem fibFactLastTwoDigitsSum :
  (fibFactSeries.map (λ n => lastTwoDigits (factorial n))).sum = 30 := by
  sorry

/-- Lemma stating that factorials of numbers greater than 10 end with 00 -/
lemma factorialEndsWith00 (n : ℕ) (h : n > 10) :
  lastTwoDigits (factorial n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fibFactLastTwoDigitsSum_l3484_348499


namespace NUMINAMATH_CALUDE_roots_fourth_power_sum_lower_bound_l3484_348401

theorem roots_fourth_power_sum_lower_bound (p : ℝ) (hp : p ≠ 0) :
  let x₁ := (-p + Real.sqrt (p^2 + 2/p^2)) / 2
  let x₂ := (-p - Real.sqrt (p^2 + 2/p^2)) / 2
  x₁^4 + x₂^4 ≥ 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_roots_fourth_power_sum_lower_bound_l3484_348401


namespace NUMINAMATH_CALUDE_chef_cherries_l3484_348439

theorem chef_cherries (used_for_pie : ℕ) (left_over : ℕ) (initial : ℕ) : 
  used_for_pie = 60 → left_over = 17 → initial = used_for_pie + left_over → initial = 77 :=
by sorry

end NUMINAMATH_CALUDE_chef_cherries_l3484_348439


namespace NUMINAMATH_CALUDE_unicorn_stitches_unicorn_stitches_proof_l3484_348458

/-- Proves that the number of stitches required to embroider a unicorn is 180 --/
theorem unicorn_stitches : ℕ → Prop :=
  fun (unicorn_stitches : ℕ) =>
    let stitches_per_minute : ℕ := 4
    let flower_stitches : ℕ := 60
    let godzilla_stitches : ℕ := 800
    let total_flowers : ℕ := 50
    let total_unicorns : ℕ := 3
    let total_minutes : ℕ := 1085
    let total_stitches : ℕ := total_minutes * stitches_per_minute
    let flower_and_godzilla_stitches : ℕ := total_flowers * flower_stitches + godzilla_stitches
    let remaining_stitches : ℕ := total_stitches - flower_and_godzilla_stitches
    remaining_stitches = total_unicorns * unicorn_stitches → unicorn_stitches = 180

/-- Proof of the theorem --/
theorem unicorn_stitches_proof : unicorn_stitches 180 :=
  by sorry

end NUMINAMATH_CALUDE_unicorn_stitches_unicorn_stitches_proof_l3484_348458


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l3484_348464

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the intersection points A and B
def intersectionPoints (A B : ℝ × ℝ) : Prop :=
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B

-- Define the perpendicular bisector
def perpendicularBisector (x y : ℝ) : Prop := x + y - 1 = 0

theorem perpendicular_bisector_of_intersecting_circles 
  (A B : ℝ × ℝ) (h : intersectionPoints A B) :
  ∀ x y : ℝ, perpendicularBisector x y ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l3484_348464


namespace NUMINAMATH_CALUDE_probability_of_seven_in_three_eighths_l3484_348490

theorem probability_of_seven_in_three_eighths : 
  let decimal_rep := [3, 7, 5]
  let count_sevens := (decimal_rep.filter (· = 7)).length
  let total_digits := decimal_rep.length
  (count_sevens : ℚ) / total_digits = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_probability_of_seven_in_three_eighths_l3484_348490


namespace NUMINAMATH_CALUDE_transformed_function_point_l3484_348425

def f : ℝ → ℝ := fun _ ↦ 8

theorem transformed_function_point (h : f 3 = 8) :
  let g : ℝ → ℝ := fun x ↦ 2 * (4 * f (3 * x - 1) + 6)
  g 2 = 38 ∧ 2 + 19 = 21 := by
  sorry

end NUMINAMATH_CALUDE_transformed_function_point_l3484_348425


namespace NUMINAMATH_CALUDE_hypotenuse_length_l3484_348452

/-- Represents a right triangle with a 45° angle -/
structure RightTriangle45 where
  leg : ℝ
  hypotenuse : ℝ

/-- The hypotenuse of a right triangle with a 45° angle is √2 times the leg -/
axiom hypotenuse_formula (t : RightTriangle45) : t.hypotenuse = t.leg * Real.sqrt 2

/-- Theorem: In a right triangle with one leg of 10 inches and an opposite angle of 45°,
    the length of the hypotenuse is 10√2 inches -/
theorem hypotenuse_length : 
  let t : RightTriangle45 := { leg := 10, hypotenuse := 10 * Real.sqrt 2 }
  t.hypotenuse = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l3484_348452


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l3484_348414

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), Prime p ∧ 10 ≤ p ∧ p < 100 ∧ 
  p ∣ binomial_coefficient 210 105 ∧
  ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ binomial_coefficient 210 105 → q ≤ p ∧
  p = 67 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l3484_348414
