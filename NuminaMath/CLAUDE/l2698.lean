import Mathlib

namespace NUMINAMATH_CALUDE_simplify_sum_of_fractions_l2698_269897

theorem simplify_sum_of_fractions (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hsum : x + y + z = 3) :
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2) =
  1 / (9 - 2*y*z) + 1 / (9 - 2*x*z) + 1 / (9 - 2*x*y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sum_of_fractions_l2698_269897


namespace NUMINAMATH_CALUDE_equation_solution_l2698_269811

theorem equation_solution :
  ∃ x : ℝ, (x^2 + x ≠ 0) ∧ (x^2 - x ≠ 0) ∧
  (4 / (x^2 + x) - 3 / (x^2 - x) = 0) ∧ 
  (x = 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2698_269811


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l2698_269819

-- Define the function
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- State the theorem
theorem f_strictly_increasing :
  (∀ x y, x < y ∧ ((x < -1/3 ∧ y ≤ -1/3) ∨ (x ≥ 1 ∧ y > 1)) → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l2698_269819


namespace NUMINAMATH_CALUDE_debut_attendance_is_200_l2698_269879

/-- The number of people who bought tickets for the debut show -/
def debut_attendance : ℕ := sorry

/-- The number of people who attended the second showing -/
def second_showing_attendance : ℕ := 3 * debut_attendance

/-- The cost of each ticket in dollars -/
def ticket_cost : ℕ := 25

/-- The total revenue from both shows in dollars -/
def total_revenue : ℕ := 20000

theorem debut_attendance_is_200 :
  debut_attendance = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_debut_attendance_is_200_l2698_269879


namespace NUMINAMATH_CALUDE_muffins_baked_by_macadams_class_l2698_269880

theorem muffins_baked_by_macadams_class (brier_muffins flannery_muffins total_muffins : ℕ) 
  (h1 : brier_muffins = 18)
  (h2 : flannery_muffins = 17)
  (h3 : total_muffins = 55) :
  total_muffins - (brier_muffins + flannery_muffins) = 20 := by
  sorry

end NUMINAMATH_CALUDE_muffins_baked_by_macadams_class_l2698_269880


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2698_269858

theorem sum_of_squares_of_roots (r₁ r₂ : ℝ) : 
  r₁^2 - 10*r₁ + 9 = 0 →
  r₂^2 - 10*r₂ + 9 = 0 →
  (r₁ > 5 ∨ r₂ > 5) →
  r₁^2 + r₂^2 = 82 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2698_269858


namespace NUMINAMATH_CALUDE_wood_rope_measurement_l2698_269825

/-- Represents the relationship between the length of a piece of wood and a rope used to measure it. -/
theorem wood_rope_measurement (x y : ℝ) :
  y = x + 4.5 ∧ 0.5 * y = x - 1 →
  (y - x = 4.5 ∧ y / 2 - x = -1) :=
by sorry

end NUMINAMATH_CALUDE_wood_rope_measurement_l2698_269825


namespace NUMINAMATH_CALUDE_satellite_height_scientific_notation_l2698_269898

/-- The height of a medium-high orbit satellite in China's Beidou navigation system. -/
def satellite_height : ℝ := 21500000

/-- Scientific notation representation of the satellite height. -/
def satellite_height_scientific : ℝ := 2.15 * (10 ^ 7)

/-- Theorem stating that the satellite height is equal to its scientific notation representation. -/
theorem satellite_height_scientific_notation :
  satellite_height = satellite_height_scientific := by sorry

end NUMINAMATH_CALUDE_satellite_height_scientific_notation_l2698_269898


namespace NUMINAMATH_CALUDE_min_trig_expression_l2698_269863

theorem min_trig_expression (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.sin x + 1 / Real.sin x)^3 + (Real.cos x + 1 / Real.cos x)^3 ≥ 729 * Real.sqrt 2 / 16 := by
  sorry

end NUMINAMATH_CALUDE_min_trig_expression_l2698_269863


namespace NUMINAMATH_CALUDE_larger_number_proof_l2698_269856

/-- Given two positive integers with HCF 23 and LCM factors 13 and 19, prove the larger is 437 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf : Nat.gcd a b = 23)
  (lcm : Nat.lcm a b = 23 * 13 * 19) :
  max a b = 437 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2698_269856


namespace NUMINAMATH_CALUDE_probability_hit_at_least_once_l2698_269814

-- Define the probability of hitting the target in a single shot
def hit_probability : ℚ := 2/3

-- Define the number of shots
def num_shots : ℕ := 3

-- Theorem statement
theorem probability_hit_at_least_once :
  1 - (1 - hit_probability) ^ num_shots = 26/27 := by
  sorry

end NUMINAMATH_CALUDE_probability_hit_at_least_once_l2698_269814


namespace NUMINAMATH_CALUDE_equal_consecutive_subgroup_exists_l2698_269869

/-- A person can be either of type A or type B -/
inductive PersonType
| A
| B

/-- A circular arrangement of people -/
def CircularArrangement := List PersonType

/-- Count the number of type A persons in a list -/
def countTypeA : List PersonType → Nat
| [] => 0
| (PersonType.A :: rest) => 1 + countTypeA rest
| (_ :: rest) => countTypeA rest

/-- Take n consecutive elements from a circular list, starting from index i -/
def takeCircular (n : Nat) (i : Nat) (l : List α) : List α :=
  (List.drop i l ++ l).take n

/-- Main theorem -/
theorem equal_consecutive_subgroup_exists (arrangement : CircularArrangement) 
    (h1 : arrangement.length = 8)
    (h2 : countTypeA arrangement = 4) :
    ∃ i, countTypeA (takeCircular 4 i arrangement) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_consecutive_subgroup_exists_l2698_269869


namespace NUMINAMATH_CALUDE_unique_balance_point_condition_l2698_269859

/-- A function f has a unique balance point if there exists a unique t such that f(t) = t -/
def has_unique_balance_point (f : ℝ → ℝ) : Prop :=
  ∃! t : ℝ, f t = t

/-- The quadratic function we're analyzing -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 3 * x + 2 * m

/-- Theorem stating the conditions for a unique balance point -/
theorem unique_balance_point_condition (m : ℝ) :
  has_unique_balance_point (f m) ↔ m = 2 ∨ m = -1 ∨ m = 1 := by sorry

end NUMINAMATH_CALUDE_unique_balance_point_condition_l2698_269859


namespace NUMINAMATH_CALUDE_watermelon_pricing_l2698_269810

/-- Represents the number of watermelons sold by each student in the morning -/
structure MorningSales where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the prices of watermelons -/
structure Prices where
  morning : ℚ
  afternoon : ℚ

/-- Theorem statement for the watermelon pricing problem -/
theorem watermelon_pricing
  (sales : MorningSales)
  (prices : Prices)
  (h1 : prices.morning > prices.afternoon)
  (h2 : prices.afternoon > 0)
  (h3 : sales.first < 10)
  (h4 : sales.second < 16)
  (h5 : sales.third < 26)
  (h6 : prices.morning * sales.first + prices.afternoon * (10 - sales.first) = 42)
  (h7 : prices.morning * sales.second + prices.afternoon * (16 - sales.second) = 42)
  (h8 : prices.morning * sales.third + prices.afternoon * (26 - sales.third) = 42)
  : prices.morning = 4.5 ∧ prices.afternoon = 1.5 := by
  sorry

#check watermelon_pricing

end NUMINAMATH_CALUDE_watermelon_pricing_l2698_269810


namespace NUMINAMATH_CALUDE_max_puzzle_sets_l2698_269806

/-- Represents the number of puzzles in a set -/
structure PuzzleSet where
  logic : ℕ
  visual : ℕ
  word : ℕ

/-- Checks if a PuzzleSet is valid according to the given conditions -/
def isValidSet (s : PuzzleSet) : Prop :=
  s.logic + s.visual + s.word ≥ 5 ∧ 2 * s.visual = s.logic

/-- The theorem to be proved -/
theorem max_puzzle_sets :
  ∀ (n : ℕ),
    (∃ (s : PuzzleSet),
      isValidSet s ∧
      n * s.logic ≤ 30 ∧
      n * s.visual ≤ 18 ∧
      n * s.word ≤ 12 ∧
      n * s.logic + n * s.visual + n * s.word = 30 + 18 + 12) →
    n ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_puzzle_sets_l2698_269806


namespace NUMINAMATH_CALUDE_german_team_goals_l2698_269843

def journalist1 (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2 (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3 (x : ℕ) : Prop := x % 2 = 1

def twoCorrect (x : ℕ) : Prop :=
  (journalist1 x ∧ journalist2 x ∧ ¬journalist3 x) ∨
  (journalist1 x ∧ ¬journalist2 x ∧ journalist3 x) ∨
  (¬journalist1 x ∧ journalist2 x ∧ journalist3 x)

theorem german_team_goals :
  ∀ x : ℕ, twoCorrect x ↔ x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
by sorry

end NUMINAMATH_CALUDE_german_team_goals_l2698_269843


namespace NUMINAMATH_CALUDE_symmetry_implies_constant_l2698_269839

/-- A bivariate real-coefficient polynomial -/
structure BivariatePolynomial where
  (p : ℝ → ℝ → ℝ)

/-- The property that P(X, Y) = P(X+Y, X-Y) for all real X and Y -/
def has_symmetry (P : BivariatePolynomial) : Prop :=
  ∀ (X Y : ℝ), P.p X Y = P.p (X + Y) (X - Y)

/-- Main theorem: If P has the symmetry property, then it is constant -/
theorem symmetry_implies_constant (P : BivariatePolynomial) 
  (h : has_symmetry P) : 
  ∃ (c : ℝ), ∀ (X Y : ℝ), P.p X Y = c := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_constant_l2698_269839


namespace NUMINAMATH_CALUDE_orange_ribbons_l2698_269877

theorem orange_ribbons (total : ℕ) (yellow purple orange black : ℕ) : 
  yellow + purple + orange + black = total →
  4 * yellow = total →
  3 * purple = total →
  6 * orange = total →
  black = 40 →
  orange = 27 :=
by sorry

end NUMINAMATH_CALUDE_orange_ribbons_l2698_269877


namespace NUMINAMATH_CALUDE_complement_of_P_l2698_269861

def U : Set ℝ := Set.univ

def P : Set ℝ := {x : ℝ | x^2 ≤ 1}

theorem complement_of_P : Set.compl P = {x : ℝ | x < -1 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_P_l2698_269861


namespace NUMINAMATH_CALUDE_oxygen_weight_value_l2698_269890

/-- The atomic weight of sodium -/
def sodium_weight : ℝ := 22.99

/-- The atomic weight of chlorine -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of the compound -/
def compound_weight : ℝ := 74

/-- The atomic weight of oxygen -/
def oxygen_weight : ℝ := compound_weight - (sodium_weight + chlorine_weight)

theorem oxygen_weight_value : oxygen_weight = 15.56 := by
  sorry

end NUMINAMATH_CALUDE_oxygen_weight_value_l2698_269890


namespace NUMINAMATH_CALUDE_parabola_greatest_a_l2698_269822

/-- The greatest possible value of a for a parabola with given conditions -/
theorem parabola_greatest_a (a b c : ℝ) : 
  (∃ (x y : ℝ), y = a * x^2 + b * x + c ∧ x = 3/5 ∧ y = -1/5) → -- vertex condition
  a < 0 → -- a is negative
  (∃ (k : ℤ), b + 2*c = k) → -- b + 2c is an integer
  (∀ (a' : ℝ), (∃ (b' c' : ℝ), 
    (∃ (x y : ℝ), y = a' * x^2 + b' * x + c' ∧ x = 3/5 ∧ y = -1/5) ∧
    a' < 0 ∧
    (∃ (k : ℤ), b' + 2*c' = k)) →
    a' ≤ a) →
  a = -5/6 := by
sorry

end NUMINAMATH_CALUDE_parabola_greatest_a_l2698_269822


namespace NUMINAMATH_CALUDE_range_of_a_l2698_269899

def p (x : ℝ) : Prop := 0 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a :
  (∀ x a : ℝ, q x a → p x) ∧
  (∃ x : ℝ, p x ∧ ∀ a : ℝ, ¬(q x a)) →
  ∀ a : ℝ, (0 ≤ a ∧ a ≤ 1/2) ↔ (∃ x : ℝ, q x a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2698_269899


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_is_linear_in_two_variables_l2698_269835

-- Define the solution point
def solution_x : ℝ := 2
def solution_y : ℝ := -3

-- Define the linear equation
def linear_equation (x y : ℝ) : Prop := x + y = -1

-- Theorem statement
theorem solution_satisfies_equation : 
  linear_equation solution_x solution_y := by
  sorry

-- Theorem to prove the equation is linear in two variables
theorem is_linear_in_two_variables : 
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ 
  ∀ (x y : ℝ), linear_equation x y ↔ a * x + b * y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_is_linear_in_two_variables_l2698_269835


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l2698_269885

/-- A parabola with vertex at the origin, axis of symmetry along the x-axis,
    and passing through the point (-4, 4) has the standard equation y² = -4x. -/
theorem parabola_standard_equation :
  ∀ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ y^2 = -4*x) →  -- Standard equation of the parabola
  f 0 = 0 →                            -- Vertex at the origin
  (∀ x : ℝ, f x = f (-x)) →            -- Axis of symmetry along x-axis
  f (-4) = 4 →                         -- Passes through (-4, 4)
  ∀ x y : ℝ, f x = y ↔ y^2 = -4*x :=   -- Conclusion: standard equation
by sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l2698_269885


namespace NUMINAMATH_CALUDE_fair_cake_distribution_l2698_269845

/-- Represents a cake flavor -/
inductive Flavor
  | Chocolate
  | Strawberry
  | Vanilla

/-- Represents a child's flavor preferences -/
structure ChildPreference where
  flavor1 : Flavor
  flavor2 : Flavor
  different : flavor1 ≠ flavor2

/-- Represents the distribution of cakes -/
structure CakeDistribution where
  totalCakes : Nat
  numChildren : Nat
  numFlavors : Nat
  childPreferences : Fin numChildren → ChildPreference
  cakesPerChild : Nat
  cakesPerFlavor : Fin numFlavors → Nat

/-- Theorem stating that a fair distribution is possible -/
theorem fair_cake_distribution 
  (d : CakeDistribution) 
  (h_total : d.totalCakes = 18) 
  (h_children : d.numChildren = 3) 
  (h_flavors : d.numFlavors = 3) 
  (h_preferences : ∀ i, (d.childPreferences i).flavor1 ≠ (d.childPreferences i).flavor2) 
  (h_distribution : ∀ i, d.cakesPerFlavor i = 6) :
  d.cakesPerChild = 6 ∧ 
  (∀ i : Fin d.numChildren, ∃ f1 f2 : Fin d.numFlavors, 
    f1 ≠ f2 ∧ 
    d.cakesPerFlavor f1 + d.cakesPerFlavor f2 = d.cakesPerChild) :=
by sorry

end NUMINAMATH_CALUDE_fair_cake_distribution_l2698_269845


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l2698_269804

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 8) ^ x) + Real.sqrt ((3 - Real.sqrt 8) ^ x) = 6) ↔ (x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l2698_269804


namespace NUMINAMATH_CALUDE_initial_walnut_count_l2698_269878

/-- The number of walnut trees initially in the park -/
def initial_walnut_trees : ℕ := sorry

/-- The number of walnut trees cut down -/
def cut_trees : ℕ := 13

/-- The number of walnut trees remaining after cutting -/
def remaining_trees : ℕ := 29

/-- The number of orange trees in the park -/
def orange_trees : ℕ := 12

theorem initial_walnut_count :
  initial_walnut_trees = remaining_trees + cut_trees :=
by sorry

end NUMINAMATH_CALUDE_initial_walnut_count_l2698_269878


namespace NUMINAMATH_CALUDE_watermelon_sale_proof_l2698_269894

/-- Calculates the total money made from selling watermelons -/
def total_money_from_watermelons (weight : ℕ) (price_per_pound : ℕ) (num_watermelons : ℕ) : ℕ :=
  weight * price_per_pound * num_watermelons

/-- Proves that selling 18 watermelons weighing 23 pounds each at $2 per pound yields $828 -/
theorem watermelon_sale_proof :
  total_money_from_watermelons 23 2 18 = 828 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_sale_proof_l2698_269894


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_theorem_l2698_269842

/-- An isosceles trapezoid with given midline length and height -/
structure IsoscelesTrapezoid where
  midline : ℝ
  height : ℝ

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := t.midline * t.height

/-- Theorem: The area of an isosceles trapezoid with midline 15 and height 3 is 45 -/
theorem isosceles_trapezoid_area_theorem :
  ∀ t : IsoscelesTrapezoid, t.midline = 15 ∧ t.height = 3 → area t = 45 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_theorem_l2698_269842


namespace NUMINAMATH_CALUDE_ellipse_condition_l2698_269840

def is_ellipse (k : ℝ) : Prop :=
  1 < k ∧ k < 5 ∧ k ≠ 3

theorem ellipse_condition (k : ℝ) :
  (∀ x y : ℝ, x^2 / (k - 1) + y^2 / (5 - k) = 1 → is_ellipse k) ∧
  ¬(∀ k : ℝ, 1 < k ∧ k < 5 → is_ellipse k) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2698_269840


namespace NUMINAMATH_CALUDE_units_digit_of_powers_l2698_269865

theorem units_digit_of_powers : 
  (31^2020 % 10 = 1) ∧ (37^2020 % 10 = 1) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_powers_l2698_269865


namespace NUMINAMATH_CALUDE_f_neg_one_eq_zero_l2698_269855

def f (x : ℝ) : ℝ := x^2 - 1

theorem f_neg_one_eq_zero : f (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_zero_l2698_269855


namespace NUMINAMATH_CALUDE_initial_marbles_l2698_269853

theorem initial_marbles (initial remaining given : ℕ) : 
  remaining = initial - given → 
  given = 8 → 
  remaining = 79 → 
  initial = 87 := by sorry

end NUMINAMATH_CALUDE_initial_marbles_l2698_269853


namespace NUMINAMATH_CALUDE_square_area_l2698_269836

/-- Given a square with one vertex at (-6, -4) and diagonals intersecting at (3, 2),
    prove that its area is 234 square units. -/
theorem square_area (v : ℝ × ℝ) (c : ℝ × ℝ) (h1 : v = (-6, -4)) (h2 : c = (3, 2)) : 
  let d := 2 * Real.sqrt ((c.1 - v.1)^2 + (c.2 - v.2)^2)
  let s := d / Real.sqrt 2
  s^2 = 234 := by sorry

end NUMINAMATH_CALUDE_square_area_l2698_269836


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2698_269867

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - x < 0.1) → (-0.1 < x ∧ x < 1.1) ∧
  ¬(∀ x : ℝ, (-0.1 < x ∧ x < 1.1) → (x^2 - x < 0.1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2698_269867


namespace NUMINAMATH_CALUDE_squared_gt_iff_abs_gt_l2698_269809

theorem squared_gt_iff_abs_gt (a b : ℝ) : a^2 > b^2 ↔ |a| > |b| := by sorry

end NUMINAMATH_CALUDE_squared_gt_iff_abs_gt_l2698_269809


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_line_through_point_with_segment_l2698_269850

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0
def l₂ (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 3 = 0

-- Define the perpendicular line n
def n (x y : ℝ) : Prop := y = -(Real.sqrt 3 / 3) * x + 2 ∨ y = -(Real.sqrt 3 / 3) * x - 2

-- Define the line m
def m (x y : ℝ) : Prop := x = Real.sqrt 3 ∨ y = (Real.sqrt 3 / 3) * x + 3

-- Theorem for part (1)
theorem perpendicular_line_equation
  (h_parallel : ∀ x y, l₁ x y ↔ l₂ x y)
  (h_perp : ∀ x y, n x y → (∀ x' y', l₁ x' y' → (y - y') = (Real.sqrt 3 / 3) * (x - x')))
  (h_area : ∃ a b, n a 0 ∧ n 0 b ∧ a * b / 2 = 2 * Real.sqrt 3) :
  ∀ x y, n x y :=
sorry

-- Theorem for part (2)
theorem line_through_point_with_segment
  (h_parallel : ∀ x y, l₁ x y ↔ l₂ x y)
  (h_point : m (Real.sqrt 3) 4)
  (h_segment : ∃ x₁ y₁ x₂ y₂,
    m x₁ y₁ ∧ m x₂ y₂ ∧ l₁ x₁ y₁ ∧ l₂ x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 2) :
  ∀ x y, m x y :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_line_through_point_with_segment_l2698_269850


namespace NUMINAMATH_CALUDE_red_balls_count_l2698_269883

theorem red_balls_count (total : ℕ) (prob : ℚ) (red : ℕ) : 
  total = 20 → prob = 1/4 → (red : ℚ)/total = prob → red = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2698_269883


namespace NUMINAMATH_CALUDE_sum_divisible_by_nine_l2698_269831

theorem sum_divisible_by_nine : 
  ∃ k : ℕ, 8230 + 8231 + 8232 + 8233 + 8234 + 8235 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_nine_l2698_269831


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_l2698_269816

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and intersection relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line → Prop)

-- State the theorem
theorem line_parallel_to_intersection
  (m n : Line)
  (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_intersection : intersection α β n)
  (h_m_parallel_α : parallel_plane m α)
  (h_m_parallel_β : parallel_plane m β) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_l2698_269816


namespace NUMINAMATH_CALUDE_max_cash_prize_value_l2698_269846

/-- Represents the promotion setup in a shopping mall -/
structure PromotionSetup where
  total_items : Nat
  daily_necessities : Nat
  chosen_items : Nat
  price_increase : ℝ
  lottery_chances : Nat
  win_probability : ℝ

/-- Calculates the expected value of the total cash prize -/
def expected_cash_prize (m : ℝ) (setup : PromotionSetup) : ℝ :=
  setup.lottery_chances * setup.win_probability * m

/-- Theorem stating the maximum value of m for an advantageous promotion -/
theorem max_cash_prize_value (setup : PromotionSetup) :
  setup.total_items = 7 →
  setup.daily_necessities = 3 →
  setup.chosen_items = 3 →
  setup.price_increase = 150 →
  setup.lottery_chances = 3 →
  setup.win_probability = 1/2 →
  ∃ (m : ℝ), m = 100 ∧ 
    ∀ (x : ℝ), expected_cash_prize x setup ≤ setup.price_increase → x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_cash_prize_value_l2698_269846


namespace NUMINAMATH_CALUDE_storks_on_fence_l2698_269826

theorem storks_on_fence (initial_birds : ℕ) (new_birds : ℕ) (total_birds : ℕ) :
  initial_birds = 4 →
  new_birds = 6 →
  total_birds = 10 →
  initial_birds + new_birds = total_birds →
  ∃ storks : ℕ, storks = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_on_fence_l2698_269826


namespace NUMINAMATH_CALUDE_heart_ratio_two_four_four_two_l2698_269854

def heart (n m : ℕ) : ℕ := n^2 * m^3

theorem heart_ratio_two_four_four_two :
  (heart 2 4) / (heart 4 2) = 2 := by sorry

end NUMINAMATH_CALUDE_heart_ratio_two_four_four_two_l2698_269854


namespace NUMINAMATH_CALUDE_movie_theater_shows_24_movies_l2698_269834

/-- Calculates the total number of movies shown in a movie theater. -/
def total_movies (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  screens * (open_hours / movie_duration)

/-- Theorem: A movie theater with 6 screens, open for 8 hours, where each movie lasts 2 hours,
    shows 24 movies throughout the day. -/
theorem movie_theater_shows_24_movies :
  total_movies 6 8 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_shows_24_movies_l2698_269834


namespace NUMINAMATH_CALUDE_identify_genuine_coins_l2698_269891

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a group of coins -/
def CoinGroup := List Nat

/-- Represents a weighing operation -/
def Weighing := CoinGroup → CoinGroup → WeighResult

/-- The total number of coins -/
def totalCoins : Nat := 11

/-- The minimum number of genuine coins to be identified -/
def minGenuineCoins : Nat := 8

/-- The maximum number of weighings allowed -/
def maxWeighings : Nat := 2

/-- Represents the state of knowledge about the coins -/
structure CoinState where
  genuineCoins : List Nat
  suspectCoins : List Nat

/-- Function to perform a weighing and update the coin state -/
def performWeighing (state : CoinState) (w : Weighing) (left right : CoinGroup) : CoinState :=
  sorry

/-- Theorem stating that it's possible to identify at least 8 genuine coins in two weighings -/
theorem identify_genuine_coins 
  (coins : List Nat) 
  (h_coins : coins.length = totalCoins) 
  : ∃ (w₁ w₂ : Weighing) (g₁ g₂ d₁ d₂ : CoinGroup),
    let s₁ := performWeighing ⟨[], coins⟩ w₁ g₁ g₂ 
    let s₂ := performWeighing s₁ w₂ d₁ d₂
    s₂.genuineCoins.length ≥ minGenuineCoins :=
  sorry

end NUMINAMATH_CALUDE_identify_genuine_coins_l2698_269891


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2698_269844

/-- Two circles are externally tangent if and only if the distance between their centers
    is equal to the sum of their radii. -/
def externally_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂

/-- The theorem stating that two circles with radii 2 and 3, whose centers are 5 units apart,
    are externally tangent. -/
theorem circles_externally_tangent :
  let r₁ : ℝ := 2
  let r₂ : ℝ := 3
  let d : ℝ := 5
  externally_tangent r₁ r₂ d :=
by
  sorry


end NUMINAMATH_CALUDE_circles_externally_tangent_l2698_269844


namespace NUMINAMATH_CALUDE_binomial_inequality_l2698_269813

theorem binomial_inequality (n : ℕ) : 2 ≤ (1 + 1 / n : ℝ) ^ n ∧ (1 + 1 / n : ℝ) ^ n < 3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l2698_269813


namespace NUMINAMATH_CALUDE_max_square_field_size_l2698_269802

/-- The maximum size of a square field that can be fully fenced given the specified conditions -/
theorem max_square_field_size (wire_cost : ℝ) (budget : ℝ) : 
  wire_cost = 30 → 
  budget = 120000 → 
  (budget / wire_cost : ℝ) < 4000 → 
  (budget / wire_cost / 4 : ℝ) ^ 2 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_max_square_field_size_l2698_269802


namespace NUMINAMATH_CALUDE_reappearance_line_l2698_269868

def letter_cycle : List Char := ['B', 'K', 'I', 'G', 'N', 'O']
def digit_cycle : List Nat := [3, 0, 7, 2, 0]

theorem reappearance_line : 
  Nat.lcm (List.length letter_cycle) (List.length digit_cycle) = 30 := by
  sorry

end NUMINAMATH_CALUDE_reappearance_line_l2698_269868


namespace NUMINAMATH_CALUDE_cone_height_equals_radius_l2698_269800

/-- The height of a cone formed by rolling a semicircular sheet of iron -/
def coneHeight (R : ℝ) : ℝ := R

/-- Theorem stating that the height of the cone is equal to the radius of the semicircular sheet -/
theorem cone_height_equals_radius (R : ℝ) (h : R > 0) : 
  coneHeight R = R := by sorry

end NUMINAMATH_CALUDE_cone_height_equals_radius_l2698_269800


namespace NUMINAMATH_CALUDE_truck_kinetic_energy_l2698_269895

/-- The initial kinetic energy of a truck with mass m, initial velocity v, and braking force F
    that stops after traveling a distance x, is equal to Fx. -/
theorem truck_kinetic_energy
  (m : ℝ) (v : ℝ) (F : ℝ) (x : ℝ) (t : ℝ)
  (h1 : m > 0)
  (h2 : v > 0)
  (h3 : F > 0)
  (h4 : x > 0)
  (h5 : t > 0)
  (h6 : F * x = (1/2) * m * v^2) :
  (1/2) * m * v^2 = F * x := by
sorry

end NUMINAMATH_CALUDE_truck_kinetic_energy_l2698_269895


namespace NUMINAMATH_CALUDE_oranges_left_to_sell_l2698_269829

theorem oranges_left_to_sell (x : ℕ) (h : x ≥ 7) :
  let total := 12 * x
  let friend1 := (1 / 4 : ℚ) * total
  let friend2 := (1 / 6 : ℚ) * total
  let charity := (1 / 8 : ℚ) * total
  let remaining_after_giving := total - friend1 - friend2 - charity
  let sold_yesterday := (3 / 7 : ℚ) * remaining_after_giving
  let remaining_after_selling := remaining_after_giving - sold_yesterday
  let eaten_by_birds := (1 / 10 : ℚ) * remaining_after_selling
  let remaining_after_birds := remaining_after_selling - eaten_by_birds
  remaining_after_birds - 4 = (3.0214287 : ℚ) * x - 4 := by sorry

end NUMINAMATH_CALUDE_oranges_left_to_sell_l2698_269829


namespace NUMINAMATH_CALUDE_range_of_a_for_p_and_q_l2698_269828

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + a*x

def is_monotonically_increasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → g x < g y

def represents_hyperbola (a : ℝ) : Prop :=
  (a + 2) * (a - 2) < 0

theorem range_of_a_for_p_and_q :
  {a : ℝ | is_monotonically_increasing (f a) ∧ represents_hyperbola a} = Set.Icc 0 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_p_and_q_l2698_269828


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l2698_269857

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  altitude : ℝ
  perimeter : ℝ

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, t.altitude = 10 ∧ t.perimeter = 40 → area t = 75 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l2698_269857


namespace NUMINAMATH_CALUDE_remaining_soup_feeds_twenty_adults_l2698_269832

/-- Represents the number of people a can of soup can feed -/
def people_per_can : ℕ := 4

/-- Represents the total number of cans of soup -/
def total_cans : ℕ := 10

/-- Represents the number of children fed -/
def children_fed : ℕ := 20

/-- Theorem: Given the conditions, prove that 20 adults can be fed with the remaining soup -/
theorem remaining_soup_feeds_twenty_adults :
  let cans_for_children := children_fed / people_per_can
  let remaining_cans := total_cans - cans_for_children
  remaining_cans * people_per_can = 20 := by
  sorry

#check remaining_soup_feeds_twenty_adults

end NUMINAMATH_CALUDE_remaining_soup_feeds_twenty_adults_l2698_269832


namespace NUMINAMATH_CALUDE_overhead_percentage_example_l2698_269870

/-- Given the purchase price, markup, and net profit of an article, 
    calculate the percentage of cost for overhead. -/
def overhead_percentage (purchase_price markup net_profit : ℚ) : ℚ :=
  let overhead := markup - net_profit
  (overhead / purchase_price) * 100

/-- Theorem stating that for the given values, the overhead percentage is 58.33% -/
theorem overhead_percentage_example : 
  overhead_percentage 48 40 12 = 58.33 := by
  sorry

end NUMINAMATH_CALUDE_overhead_percentage_example_l2698_269870


namespace NUMINAMATH_CALUDE_three_intersections_implies_a_value_l2698_269874

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then Real.sin x else x^3 - 9*x^2 + 25*x + a

theorem three_intersections_implies_a_value (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = x₁ ∧ f a x₂ = x₂ ∧ f a x₃ = x₃) →
  a = -20 ∨ a = -16 := by
  sorry

end NUMINAMATH_CALUDE_three_intersections_implies_a_value_l2698_269874


namespace NUMINAMATH_CALUDE_time_period_is_12_hours_l2698_269881

/-- The time period in hours for a given population net increase -/
def time_period (birth_rate : ℚ) (death_rate : ℚ) (net_increase : ℕ) : ℚ :=
  let net_rate_per_second : ℚ := (birth_rate - death_rate) / 2
  let seconds : ℚ := net_increase / net_rate_per_second
  seconds / 3600

/-- Theorem stating that given the problem conditions, the time period is 12 hours -/
theorem time_period_is_12_hours :
  time_period 8 6 86400 = 12 := by
  sorry

end NUMINAMATH_CALUDE_time_period_is_12_hours_l2698_269881


namespace NUMINAMATH_CALUDE_remainder_3456_div_97_l2698_269821

theorem remainder_3456_div_97 : 3456 % 97 = 61 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3456_div_97_l2698_269821


namespace NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l2698_269827

theorem at_least_one_fraction_less_than_two (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l2698_269827


namespace NUMINAMATH_CALUDE_girls_joined_l2698_269837

theorem girls_joined (initial_girls final_girls : ℕ) : 
  initial_girls = 732 → final_girls = 1414 → final_girls - initial_girls = 682 :=
by
  sorry

#check girls_joined

end NUMINAMATH_CALUDE_girls_joined_l2698_269837


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2698_269838

theorem contrapositive_equivalence (a b : ℝ) :
  (¬((a - b) * (a + b) = 0) → ¬(a - b = 0)) ↔
  ((a - b = 0) → ((a - b) * (a + b) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2698_269838


namespace NUMINAMATH_CALUDE_divisibility_by_6p_l2698_269851

theorem divisibility_by_6p (p : ℕ) (hp : Prime p) (hp2 : p > 2) :
  ∃ k : ℤ, 7^p - 5^p - 2 = 6 * p * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_6p_l2698_269851


namespace NUMINAMATH_CALUDE_probability_woman_lawyer_l2698_269807

theorem probability_woman_lawyer (total_members : ℕ) 
  (women_percentage : ℝ) (young_lawyer_percentage : ℝ) (old_lawyer_percentage : ℝ) 
  (h1 : women_percentage = 0.4)
  (h2 : young_lawyer_percentage = 0.3)
  (h3 : old_lawyer_percentage = 0.1)
  (h4 : young_lawyer_percentage + old_lawyer_percentage + 0.6 = 1) :
  (women_percentage * (young_lawyer_percentage + old_lawyer_percentage)) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_probability_woman_lawyer_l2698_269807


namespace NUMINAMATH_CALUDE_problem_solution_l2698_269876

theorem problem_solution :
  ∃ (a b : ℤ) (c d : ℚ),
    (∀ n : ℤ, n > 0 → a ≤ n) ∧
    (∀ n : ℤ, n < 0 → n ≤ b) ∧
    (∀ q : ℚ, q ≠ 0 → |c| ≤ |q|) ∧
    (d⁻¹ = d) ∧
    ((a^2 : ℚ) - (b^2 : ℚ) + 2*d - c = 2 ∨ (a^2 : ℚ) - (b^2 : ℚ) + 2*d - c = -2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2698_269876


namespace NUMINAMATH_CALUDE_quadratic_roots_l2698_269875

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) 
  (h1 : a + b + c = 0) (h2 : a - b + c = 0) :
  ∃ (x y : ℝ), x = 1 ∧ y = -1 ∧ 
  (∀ z : ℝ, a * z^2 + b * z + c = 0 ↔ z = x ∨ z = y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2698_269875


namespace NUMINAMATH_CALUDE_amazon_profit_per_package_l2698_269871

/-- Profit per package for Amazon distribution centers -/
theorem amazon_profit_per_package :
  ∀ (centers : ℕ)
    (first_center_daily_packages : ℕ)
    (second_center_multiplier : ℕ)
    (combined_weekly_profit : ℚ)
    (days_per_week : ℕ),
  centers = 2 →
  first_center_daily_packages = 10000 →
  second_center_multiplier = 3 →
  combined_weekly_profit = 14000 →
  days_per_week = 7 →
  let total_weekly_packages := first_center_daily_packages * days_per_week * (1 + second_center_multiplier)
  (combined_weekly_profit / total_weekly_packages : ℚ) = 1/20 := by
sorry

end NUMINAMATH_CALUDE_amazon_profit_per_package_l2698_269871


namespace NUMINAMATH_CALUDE_loaves_needed_l2698_269815

/-- The number of first-year students -/
def first_year_students : ℕ := 247

/-- The difference between the number of sophomores and first-year students -/
def sophomore_difference : ℕ := 131

/-- The number of sophomores -/
def sophomores : ℕ := first_year_students + sophomore_difference

/-- The total number of students (first-year and sophomores) -/
def total_students : ℕ := first_year_students + sophomores

theorem loaves_needed : total_students = 625 := by sorry

end NUMINAMATH_CALUDE_loaves_needed_l2698_269815


namespace NUMINAMATH_CALUDE_elise_savings_elise_savings_proof_l2698_269847

/-- Proves that Elise saved $13 from her allowance -/
theorem elise_savings : ℕ → Prop :=
  fun (saved : ℕ) =>
    let initial : ℕ := 8
    let comic_cost : ℕ := 2
    let puzzle_cost : ℕ := 18
    let final : ℕ := 1
    initial + saved - (comic_cost + puzzle_cost) = final →
    saved = 13

/-- The proof of the theorem -/
theorem elise_savings_proof : elise_savings 13 := by
  sorry

end NUMINAMATH_CALUDE_elise_savings_elise_savings_proof_l2698_269847


namespace NUMINAMATH_CALUDE_book_price_relationship_l2698_269889

/-- Represents a collection of books with linearly increasing prices -/
structure BookCollection where
  basePrice : ℕ
  count : ℕ

/-- Get the price of a book at a specific position -/
def BookCollection.priceAt (bc : BookCollection) (position : ℕ) : ℕ :=
  bc.basePrice + position - 1

/-- The main theorem about the book price relationship -/
theorem book_price_relationship (bc : BookCollection) 
  (h1 : bc.count = 49) : 
  (bc.priceAt 49)^2 = (bc.priceAt 25)^2 + (bc.priceAt 26)^2 := by
  sorry

/-- Helper lemma: The price difference between adjacent books is 1 -/
lemma price_difference (bc : BookCollection) (i : ℕ) 
  (h : i < bc.count) :
  bc.priceAt (i + 1) = bc.priceAt i + 1 := by
  sorry

end NUMINAMATH_CALUDE_book_price_relationship_l2698_269889


namespace NUMINAMATH_CALUDE_fraction_sum_equals_two_thirds_l2698_269892

theorem fraction_sum_equals_two_thirds : 
  2 / 10 + 4 / 40 + 6 / 60 + 8 / 30 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_two_thirds_l2698_269892


namespace NUMINAMATH_CALUDE_root_range_implies_k_range_l2698_269805

theorem root_range_implies_k_range :
  ∀ k : ℝ,
  (∃ x₁ x₂ : ℝ, 
    x₁^2 + (k-3)*x₁ + k^2 = 0 ∧
    x₂^2 + (k-3)*x₂ + k^2 = 0 ∧
    x₁ < 1 ∧ x₂ > 1 ∧ x₁ ≠ x₂) →
  k > -2 ∧ k < 1 :=
by sorry

end NUMINAMATH_CALUDE_root_range_implies_k_range_l2698_269805


namespace NUMINAMATH_CALUDE_two_digit_sum_divisible_by_11_four_digit_divisible_by_11_l2698_269801

-- Problem 1
theorem two_digit_sum_divisible_by_11 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) :
  ∃ k : ℤ, (10 * a + b) + (10 * b + a) = 11 * k :=
sorry

-- Problem 2
theorem four_digit_divisible_by_11 (m n : ℕ) (h1 : m < 10) (h2 : n < 10) :
  ∃ k : ℤ, 1000 * m + 100 * n + 10 * n + m = 11 * k :=
sorry

end NUMINAMATH_CALUDE_two_digit_sum_divisible_by_11_four_digit_divisible_by_11_l2698_269801


namespace NUMINAMATH_CALUDE_sin_plus_cos_equals_sqrt_a_plus_one_l2698_269817

theorem sin_plus_cos_equals_sqrt_a_plus_one (θ : Real) (a : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sin (2 * θ) = a) : 
  Real.sin θ + Real.cos θ = Real.sqrt (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_equals_sqrt_a_plus_one_l2698_269817


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2698_269893

theorem chess_tournament_participants : ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2698_269893


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_five_l2698_269848

theorem x_squared_minus_y_squared_equals_five
  (a : ℝ) (x y : ℝ) (h1 : a^x * a^y = a^5) (h2 : a^x / a^y = a) :
  x^2 - y^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_five_l2698_269848


namespace NUMINAMATH_CALUDE_remainder_plus_fraction_equals_result_l2698_269852

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem remainder_plus_fraction_equals_result :
  rem (5/7) (-3/4) + 1/14 = 1/28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_plus_fraction_equals_result_l2698_269852


namespace NUMINAMATH_CALUDE_locus_of_centers_l2698_269818

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 2)^2 + b^2 = (3 - r)^2

/-- The locus of centers (a, b) of circles externally tangent to C₁ and internally tangent to C₂ -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) ↔ 
  84 * a^2 + 100 * b^2 - 64 * a - 64 = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l2698_269818


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l2698_269849

theorem smallest_lcm_with_gcd_five (a b : ℕ) : 
  1000 ≤ a ∧ a < 10000 ∧ 
  1000 ≤ b ∧ b < 10000 ∧ 
  Nat.gcd a b = 5 →
  201000 ≤ Nat.lcm a b :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l2698_269849


namespace NUMINAMATH_CALUDE_probability_two_females_l2698_269896

/-- The probability of selecting 2 female students from a group of 5 students (2 males and 3 females) -/
theorem probability_two_females (total_students : Nat) (male_students : Nat) (female_students : Nat) 
  (group_size : Nat) : 
  total_students = 5 → 
  male_students = 2 → 
  female_students = 3 → 
  group_size = 2 → 
  (Nat.choose female_students group_size : Rat) / (Nat.choose total_students group_size : Rat) = 3/10 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_females_l2698_269896


namespace NUMINAMATH_CALUDE_thompson_class_median_l2698_269808

/-- Represents the number of families with a specific number of children -/
structure FamilyCount where
  childCount : ℕ
  familyCount : ℕ

/-- Calculates the median of a list of natural numbers -/
def median (l : List ℕ) : ℚ :=
  sorry

/-- Expands a list of FamilyCount into a list of individual family sizes -/
def expandCounts (counts : List FamilyCount) : List ℕ :=
  sorry

theorem thompson_class_median :
  let familyCounts : List FamilyCount := [
    ⟨1, 4⟩, ⟨2, 3⟩, ⟨3, 5⟩, ⟨4, 2⟩, ⟨5, 1⟩
  ]
  let expandedList := expandCounts familyCounts
  median expandedList = 3 := by
  sorry

end NUMINAMATH_CALUDE_thompson_class_median_l2698_269808


namespace NUMINAMATH_CALUDE_cookie_radius_l2698_269823

theorem cookie_radius (x y : ℝ) :
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 6*x + 10*y + 12 = 0 ↔ (x - 3)^2 + (y + 5)^2 = r^2) →
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 6*x + 10*y + 12 = 0 ↔ (x - 3)^2 + (y + 5)^2 = r^2 ∧ r = Real.sqrt 22) :=
by sorry

end NUMINAMATH_CALUDE_cookie_radius_l2698_269823


namespace NUMINAMATH_CALUDE_alternating_arithmetic_series_sum_l2698_269872

def alternatingArithmeticSeries (a₁ : ℤ) (d : ℤ) (aₙ : ℤ) : ℤ :=
  let n := (aₙ - a₁) / d + 1
  let pairs := (n - 1) / 2
  let pairSum := -d
  let leftover := if n % 2 = 0 then 0 else aₙ
  pairs * pairSum + leftover

theorem alternating_arithmetic_series_sum :
  alternatingArithmeticSeries 2 3 56 = 29 := by sorry

end NUMINAMATH_CALUDE_alternating_arithmetic_series_sum_l2698_269872


namespace NUMINAMATH_CALUDE_greene_nursery_flower_count_l2698_269860

theorem greene_nursery_flower_count : 
  let red_roses : ℕ := 1491
  let yellow_carnations : ℕ := 3025
  let white_roses : ℕ := 1768
  red_roses + yellow_carnations + white_roses = 6284 :=
by sorry

end NUMINAMATH_CALUDE_greene_nursery_flower_count_l2698_269860


namespace NUMINAMATH_CALUDE_no_odd_three_digit_div_five_without_five_l2698_269824

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def divisible_by_five (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

def does_not_contain_five (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 5

theorem no_odd_three_digit_div_five_without_five :
  {n : ℕ | is_odd n ∧ is_three_digit n ∧ divisible_by_five n ∧ does_not_contain_five n} = ∅ :=
sorry

end NUMINAMATH_CALUDE_no_odd_three_digit_div_five_without_five_l2698_269824


namespace NUMINAMATH_CALUDE_min_three_types_l2698_269886

/-- Represents a type of tree in the grove -/
inductive TreeType
  | Birch
  | Spruce
  | Pine
  | Aspen

/-- Represents the grove of trees -/
structure Grove where
  trees : Finset Nat
  types : Nat → TreeType
  total_count : trees.card = 100
  diverse_85 : ∀ s : Finset Nat, s ⊆ trees → s.card = 85 → 
    (TreeType.Birch ∈ types '' s) ∧ 
    (TreeType.Spruce ∈ types '' s) ∧ 
    (TreeType.Pine ∈ types '' s) ∧ 
    (TreeType.Aspen ∈ types '' s)

/-- The main theorem to be proved -/
theorem min_three_types (g : Grove) : 
  ∃ n : Nat, n = 69 ∧ 
  (∀ s : Finset Nat, s ⊆ g.trees → s.card ≥ n → 
    ∃ t1 t2 t3 : TreeType, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    t1 ∈ g.types '' s ∧ t2 ∈ g.types '' s ∧ t3 ∈ g.types '' s) ∧
  (∃ s : Finset Nat, s ⊆ g.trees ∧ s.card = n - 1 ∧ 
    ∃ t1 t2 : TreeType, t1 ≠ t2 ∧ 
    (∀ t ∈ g.types '' s, t = t1 ∨ t = t2)) :=
sorry

end NUMINAMATH_CALUDE_min_three_types_l2698_269886


namespace NUMINAMATH_CALUDE_equation_solution_l2698_269803

theorem equation_solution :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2698_269803


namespace NUMINAMATH_CALUDE_trishul_investment_percentage_l2698_269884

/-- Represents the investment amounts in Rupees -/
structure Investment where
  vishal : ℝ
  trishul : ℝ
  raghu : ℝ

/-- The given conditions of the investment problem -/
def investment_conditions (i : Investment) : Prop :=
  i.vishal = 1.1 * i.trishul ∧
  i.vishal + i.trishul + i.raghu = 6936 ∧
  i.raghu = 2400

/-- The theorem stating that Trishul invested 10% less than Raghu -/
theorem trishul_investment_percentage (i : Investment) 
  (h : investment_conditions i) : 
  (i.raghu - i.trishul) / i.raghu = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_trishul_investment_percentage_l2698_269884


namespace NUMINAMATH_CALUDE_rainwater_chickens_l2698_269887

/-- Mr. Rainwater's farm animals -/
structure Farm where
  cows : ℕ
  goats : ℕ
  chickens : ℕ

/-- The conditions of Mr. Rainwater's farm -/
def rainwater_farm (f : Farm) : Prop :=
  f.cows = 9 ∧ f.goats = 4 * f.cows ∧ f.goats = 2 * f.chickens

/-- Theorem: Mr. Rainwater has 18 chickens -/
theorem rainwater_chickens (f : Farm) (h : rainwater_farm f) : f.chickens = 18 := by
  sorry

end NUMINAMATH_CALUDE_rainwater_chickens_l2698_269887


namespace NUMINAMATH_CALUDE_cubic_sum_reciprocal_l2698_269873

theorem cubic_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^3 + 1/a^3 = 2 * Real.sqrt 5 ∨ a^3 + 1/a^3 = -2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_cubic_sum_reciprocal_l2698_269873


namespace NUMINAMATH_CALUDE_johns_number_l2698_269862

theorem johns_number (n : ℕ) : n = 1500 ↔ 
  (125 ∣ n) ∧ 
  (30 ∣ n) ∧ 
  Even n ∧ 
  1000 < n ∧ 
  n < 3000 ∧ 
  (∀ m : ℕ, m < n → ¬((125 ∣ m) ∧ (30 ∣ m) ∧ Even m ∧ 1000 < m ∧ m < 3000)) :=
by sorry

#check johns_number

end NUMINAMATH_CALUDE_johns_number_l2698_269862


namespace NUMINAMATH_CALUDE_average_of_XYZ_l2698_269864

theorem average_of_XYZ (X Y Z : ℝ) 
  (eq1 : 2001 * Z - 4002 * X = 8008)
  (eq2 : 2001 * Y + 5005 * X = 10010) : 
  (X + Y + Z) / 3 = 0.1667 * X + 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_XYZ_l2698_269864


namespace NUMINAMATH_CALUDE_multiples_of_15_between_20_and_205_l2698_269820

theorem multiples_of_15_between_20_and_205 : 
  (Finset.filter (fun x => x % 15 = 0 ∧ x > 20 ∧ x ≤ 205) (Finset.range 206)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_20_and_205_l2698_269820


namespace NUMINAMATH_CALUDE_smallest_other_integer_l2698_269812

theorem smallest_other_integer (a b x : ℕ+) : 
  (a = 36 ∨ b = 36) →
  Nat.gcd a b = x + 6 →
  Nat.lcm a b = x * (x + 6) →
  (a ≠ 36 → a ≥ 24) ∧ (b ≠ 36 → b ≥ 24) :=
sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l2698_269812


namespace NUMINAMATH_CALUDE_ellipse_max_dot_product_l2698_269866

/-- Definition of the ellipse M -/
def ellipse_M (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the circle N -/
def circle_N (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 1

/-- The ellipse passes through the point (2, √6) -/
def ellipse_point (a b : ℝ) : Prop :=
  ellipse_M a b 2 (Real.sqrt 6)

/-- The eccentricity of the ellipse is √2/2 -/
def ellipse_eccentricity (a b : ℝ) : Prop :=
  (Real.sqrt (a^2 - b^2)) / a = Real.sqrt 2 / 2

/-- Definition of the dot product PA · PB -/
def dot_product (x y : ℝ) : ℝ :=
  x^2 + y^2 - 4*y + 3

/-- Main theorem -/
theorem ellipse_max_dot_product (a b : ℝ) :
  ellipse_M a b 2 (Real.sqrt 6) →
  ellipse_eccentricity a b →
  (∀ x y : ℝ, ellipse_M a b x y → dot_product x y ≤ 23) ∧
  (∃ x y : ℝ, ellipse_M a b x y ∧ dot_product x y = 23) :=
sorry

end NUMINAMATH_CALUDE_ellipse_max_dot_product_l2698_269866


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l2698_269841

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop := 2 * x^2 + 4 * x + 6 * y = 24

/-- The slope of the tangent line at a given point -/
noncomputable def tangent_slope (x : ℝ) : ℝ := 
  -(2/3 * x + 2/3)

/-- Theorem: The slope of the tangent line to the curve at x = 1 is -4/3 -/
theorem tangent_slope_at_one : 
  tangent_slope 1 = -4/3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l2698_269841


namespace NUMINAMATH_CALUDE_correct_inequality_l2698_269833

theorem correct_inequality : 
  (-3 > -5) ∧ 
  ¬(-3 > -2) ∧ 
  ¬((1:ℚ)/2 < (1:ℚ)/3) ∧ 
  ¬(-(1:ℚ)/2 < -(2:ℚ)/3) := by
  sorry

end NUMINAMATH_CALUDE_correct_inequality_l2698_269833


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l2698_269888

-- Define the heights of the sandcastles
def miki_height : ℝ := 0.8333333333333334
def sister_height : ℝ := 0.5

-- Theorem to prove
theorem sandcastle_height_difference :
  miki_height - sister_height = 0.3333333333333334 := by
  sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l2698_269888


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l2698_269830

/-- The number of walnut trees in the park after planting is equal to the sum of 
    the initial number of trees and the number of trees planted. -/
theorem walnut_trees_after_planting 
  (initial_trees : ℕ) 
  (planted_trees : ℕ) : 
  initial_trees + planted_trees = 
    initial_trees + planted_trees :=
by sorry

/-- Specific instance of the theorem with 4 initial trees and 6 planted trees -/
example : 4 + 6 = 10 :=
by sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l2698_269830


namespace NUMINAMATH_CALUDE_pet_store_cats_l2698_269882

theorem pet_store_cats (initial_siamese : ℕ) (cats_sold : ℕ) (cats_remaining : ℕ) 
  (h1 : initial_siamese = 13)
  (h2 : cats_sold = 10)
  (h3 : cats_remaining = 8) :
  ∃ initial_house : ℕ, 
    initial_house = 5 ∧ 
    initial_siamese + initial_house - cats_sold = cats_remaining :=
  sorry

end NUMINAMATH_CALUDE_pet_store_cats_l2698_269882
