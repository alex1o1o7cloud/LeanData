import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_l69_6981

/-- The area of a regular octagon inscribed in a circle with radius 4 units -/
noncomputable def octagonArea : ℝ := 64 * Real.sqrt 2

/-- Theorem: The area of a regular octagon inscribed in a circle with radius 4 units is 64√2 square units -/
theorem regular_octagon_area (r : ℝ) (h : r = 4) : octagonArea = 64 * Real.sqrt 2 := by
  sorry

#check regular_octagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_l69_6981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_equals_one_l69_6985

/-- A function f defined as f(x) = a * sin(πx + α) + b * cos(πx - β) -/
noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x - β)

/-- Theorem stating that if f(2016) = -1, then f(2017) = 1 -/
theorem f_2017_equals_one
  (a b α β : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hα : α ≠ 0)
  (hβ : β ≠ 0)
  (h : f a b α β 2016 = -1) :
  f a b α β 2017 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_equals_one_l69_6985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_real_solution_l69_6914

theorem unique_real_solution :
  ∃! x : ℝ, (4 - x^3 / 2)^(1/3 : ℝ) = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_real_solution_l69_6914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_with_equal_pair_sums_l69_6984

def S : Finset ℕ := Finset.range 16

def is_valid_partition (A B : Finset ℕ) : Prop :=
  A ∪ B = S ∧ A ∩ B = ∅ ∧ A.card = 8 ∧ B.card = 8

def pair_sums (X : Finset ℕ) : Finset ℕ :=
  Finset.image (λ p => p.1 + p.2) (X.product X)

theorem partition_with_equal_pair_sums : ∃ A B : Finset ℕ,
  is_valid_partition A B ∧ pair_sums A = pair_sums B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_with_equal_pair_sums_l69_6984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l69_6934

-- Define the two curves in polar coordinates
noncomputable def curve1 (θ : ℝ) : ℝ := 3 * Real.cos θ
noncomputable def curve2 (θ : ℝ) : ℝ := 6 * Real.sin θ

-- Define the Cartesian equations of the two curves
def circle1 (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = 9/4
def circle2 (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9

-- Define the number of intersection points
def num_intersections : ℕ := 2

-- Theorem statement
theorem intersection_count :
  ∃ (points : Finset (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ points ↔ circle1 x y ∧ circle2 x y) ∧
    points.card = num_intersections := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l69_6934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_ten_in_factorial_20_l69_6903

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem highest_power_of_ten_in_factorial_20 : 
  ∃ k : ℕ, k = 4 ∧ (10^k : ℕ) ∣ factorial 20 ∧ ∀ m : ℕ, m > k → ¬((10^m : ℕ) ∣ factorial 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_ten_in_factorial_20_l69_6903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basement_flood_pump_time_l69_6999

/-- Represents the time required to pump out water from a flooded basement -/
noncomputable def pumpOutTime (roomLength roomWidth waterDepth : ℝ) 
                (numPumps : ℕ) 
                (initialPumpRate decreasedPumpRate : ℝ) 
                (decreaseTime : ℝ) 
                (conversionFactor : ℝ) : ℝ :=
  let totalWaterVolume := roomLength * roomWidth * waterDepth * conversionFactor
  let initialPumpedVolume := (numPumps : ℝ) * initialPumpRate * decreaseTime
  let remainingVolume := totalWaterVolume - initialPumpedVolume
  let reducedPumpRate := (numPumps : ℝ) * decreasedPumpRate
  decreaseTime + remainingVolume / reducedPumpRate

/-- Theorem stating that the time to pump out the water is 345 minutes -/
theorem basement_flood_pump_time :
  pumpOutTime 30 20 2 3 10 8 120 7.5 = 345 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basement_flood_pump_time_l69_6999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_problem_l69_6979

/-- A random variable following a binomial distribution -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  value : ℝ

/-- The expectation of a binomial random variable -/
noncomputable def expectation (n : ℕ) (p : ℝ) (X : BinomialRV n p) : ℝ := n * p

/-- The variance of a binomial random variable -/
noncomputable def variance (n : ℕ) (p : ℝ) (X : BinomialRV n p) : ℝ := n * p * (1 - p)

/-- Given random variables ξ and η satisfying the conditions -/
theorem random_variable_problem 
  (ξ : BinomialRV 10 0.6) 
  (η : ℝ) 
  (h : ξ.value + η = 8) : 
  expectation 10 0.6 ξ - η = 2 ∧ variance 10 0.6 ξ = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_problem_l69_6979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_2009_l69_6983

/-- Definition of φ_d -/
noncomputable def phi (d : ℝ) : ℝ := (d + Real.sqrt (d^2 + 4)) / 2

theorem golden_ratio_2009 :
  ∃ (a b c : ℕ), 
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
    (Nat.gcd a c = 1) ∧
    (phi 2009 = (a + Real.sqrt b : ℝ) / c) ∧
    (a + b + c = 4038096) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_2009_l69_6983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_range_l69_6944

noncomputable section

-- Define the circle C
def circle_C (α : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos α, 2 * Real.sin α)

-- Define the line l in polar form
noncomputable def line_l (θ : ℝ) : ℝ :=
  Real.sqrt 3 / (Real.sin θ + Real.sqrt 3 * Real.cos θ)

-- Define the range of θ₁
def θ₁_range (θ : ℝ) : Prop :=
  Real.pi / 6 ≤ θ ∧ θ ≤ Real.pi / 3

-- Define the distance |OP|
noncomputable def distance_OP (θ : ℝ) : ℝ :=
  4 * Real.cos θ

-- Define the distance |OQ|
noncomputable def distance_OQ (θ : ℝ) : ℝ :=
  line_l θ

-- State the theorem
theorem product_range :
  ∀ θ, θ₁_range θ →
    2 ≤ distance_OP θ * distance_OQ θ ∧
    distance_OP θ * distance_OQ θ ≤ 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_range_l69_6944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_exists_l69_6937

-- Define the circle and points
variable (C : Set (ℝ × ℝ)) -- Circle as a set of points in ℝ²
variable (P B D : ℝ × ℝ) -- Points in ℝ²

-- Define the properties of the circle and points
variable (h_circle : ∀ X ∈ C, dist X P = dist B P)
variable (h_B_on_C : B ∈ C)
variable (h_D_on_C : D ∈ C)

-- Define the ray from P through B
def ray_PB (P B : ℝ × ℝ) (X : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t ≥ 0 ∧ X = P + t • (B - P)

-- State the theorem
theorem unique_point_exists :
  ∃! A : ℝ × ℝ,
    ray_PB P B A ∧
    dist A B = dist A D ∧
    ∀ X ∈ C, dist A B ≤ dist A X :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_exists_l69_6937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_less_than_twenty_percent_is_minority_l69_6904

/-- A group within a population -/
structure MyGroup where
  size : ℝ
  size_nonneg : 0 ≤ size

/-- A population consisting of two groups -/
structure Population where
  total : ℝ
  total_pos : 0 < total
  group1 : MyGroup
  group2 : MyGroup
  sum_eq_total : group1.size + group2.size = total

/-- Definition of minority: a group that comprises less than half of the total population -/
def is_minority (pop : Population) (g : MyGroup) : Prop :=
  g.size < pop.total / 2

/-- Theorem: If a group comprises less than 20% of the total population, it is a minority -/
theorem less_than_twenty_percent_is_minority (pop : Population) (g : MyGroup)
  (h : g.size < 0.2 * pop.total) : is_minority pop g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_less_than_twenty_percent_is_minority_l69_6904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_four_point_five_l69_6956

-- Define the points A and B
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 3)

-- Define the line on which C lies
def line (p : ℝ × ℝ) : Prop := p.1 + 2*p.2 = 8

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

-- Theorem statement
theorem triangle_area_is_four_point_five :
  ∃ C : ℝ × ℝ, line C ∧ triangleArea A B C = 4.5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_four_point_five_l69_6956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l69_6922

/-- 
Given an angle α in the second quadrant and a point P(x, √5) on its terminal side,
if cos α = (√2/4)x, then sin α = √10/4.
-/
theorem sin_alpha_value (α : Real) (x : Real) :
  α ∈ Set.Ioo (π/2) π →  -- α is in the second quadrant
  (∃ P : Real × Real, P.1 = x ∧ P.2 = Real.sqrt 5) →  -- P(x, √5) exists
  Real.cos α = (Real.sqrt 2 / 4) * x →  -- cos α = (√2/4)x
  Real.sin α = Real.sqrt 10 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l69_6922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_percentage_calculation_l69_6942

/-- Calculates the dividend percentage given the investment details and dividend received --/
theorem dividend_percentage_calculation 
  (investment : ℝ) 
  (share_face_value : ℝ) 
  (premium_percentage : ℝ) 
  (dividend_received : ℝ) 
  (h1 : investment = 14400) 
  (h2 : share_face_value = 100) 
  (h3 : premium_percentage = 20) 
  (h4 : dividend_received = 840) : 
  (dividend_received / (investment / (share_face_value * (1 + premium_percentage / 100))) / share_face_value) * 100 = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_percentage_calculation_l69_6942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_system_of_equations_proof_l69_6990

-- Part 1: Calculation proof
theorem calculation_proof : 
  ((-8 : ℝ) ^ (1/3 : ℝ)) - Real.sqrt 2 + (Real.sqrt 3)^2 + |1 - Real.sqrt 2| - (-1)^2023 = 1 := by sorry

-- Part 2: System of equations proof
theorem system_of_equations_proof (x y : ℝ) :
  (1/2 * x - 3/2 * y = -1) ∧ (2 * x + y = 3) → x = 1 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_system_of_equations_proof_l69_6990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_participants_theorem_l69_6940

/-- The number of problems in the mathematical match -/
def num_problems : ℕ := 15

/-- The minimum sum of scores for any group of 12 participants -/
def min_sum_12 : ℕ := 36

/-- The minimum number of participants who must solve the same problem -/
def min_same_problem : ℕ := 3

/-- The size of the group for which we check the minimum sum -/
def group_size : ℕ := 12

theorem min_participants_theorem (n : ℕ) (h1 : n > 12) :
  (∀ (scores : Fin n → Fin (num_problems + 1)),
    (∀ (subset : Finset (Fin n)),
      subset.card = group_size →
      (subset.sum fun i ↦ scores i) ≥ min_sum_12) →
    ∃ (problem : Fin num_problems),
      (Finset.filter (fun i ↦ scores i = problem) (Finset.univ : Finset (Fin n))).card ≥ min_same_problem) →
  n ≥ 15 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_participants_theorem_l69_6940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_first_term_l69_6967

/-- The sum of an infinite geometric series with first term a and common ratio r, where |r| < 1 -/
noncomputable def infiniteGeometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: For an infinite geometric series with common ratio -1/3 and sum 18, the first term is 24 -/
theorem infinite_geometric_series_first_term :
  ∃ (a : ℝ), infiniteGeometricSum a (-1/3) = 18 ∧ a = 24 := by
  -- We'll use 24 as our witness for 'a'
  use 24
  -- Split the goal into two parts
  constructor
  -- Prove that infiniteGeometricSum 24 (-1/3) = 18
  · sorry
  -- Prove that a = 24 (which is trivially true given our choice of 'a')
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_first_term_l69_6967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_and_minimum_value_l69_6982

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a/2) * x^2 + 2*x - Real.log x

theorem tangent_perpendicular_and_minimum_value 
  (a : ℝ) (h : a ≥ 0) : 
  (((deriv (f a)) 1 = 3) → a = 2) ∧ 
  (∃ (m : ℝ), (∀ (x : ℝ), x > 0 → f a x ≥ m) ∧ m > 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_and_minimum_value_l69_6982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_approximation_l69_6953

noncomputable def kimiko_age : ℝ := 28

noncomputable def omi_age : ℝ := 2 * kimiko_age

noncomputable def arlette_age : ℝ := 3/4 * kimiko_age

noncomputable def xander_age : ℝ := kimiko_age^2 - 5

noncomputable def yolanda_age : ℝ := xander_age^(1/3)

noncomputable def average_age : ℝ := (kimiko_age + omi_age + arlette_age + xander_age + yolanda_age) / 5

theorem average_age_approximation :
  abs (average_age - 178.64) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_approximation_l69_6953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_special_polynomial_l69_6915

theorem gcd_special_polynomial (y : ℤ) (h : 56790 ∣ y) :
  Nat.gcd ((3 * y + 2) * (5 * y + 3) * (11 * y + 7) * (y + 17)).natAbs y.natAbs = 714 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_special_polynomial_l69_6915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_coffee_p_used_l69_6992

/-- Representation of a coffee mixture -/
structure CoffeeMixture where
  p : ℝ  -- amount of coffee p
  v : ℝ  -- amount of coffee v

/-- Given two coffee mixtures x and y, prove that the total amount of coffee p used is 24 lbs -/
theorem total_coffee_p_used (x y : CoffeeMixture) (v : ℝ) : 
  v = 25 ∧ 
  (x.p / x.v = 4 / 1) ∧ 
  (y.p / y.v = 1 / 5) ∧ 
  x.p = 20 → 
  x.p + y.p = 24 :=
by
  sorry

/-- Total amount of coffee p used in both mixtures -/
def total_p (x y : CoffeeMixture) : ℝ :=
  x.p + y.p

#check total_coffee_p_used

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_coffee_p_used_l69_6992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_consecutive_sides_and_angle_relation_l69_6962

/-- Given a triangle ABC with sides that are consecutive positive integers
    and where the largest angle is twice the smallest angle,
    prove that its area is 15√7/4 -/
theorem triangle_area_with_consecutive_sides_and_angle_relation :
  ∀ (a b c : ℕ+) (A B C : ℝ),
    (a : ℝ) + 1 = b ∧ b + 1 = c →  -- Consecutive integer sides
    0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Valid angles
    A + B + C = π →  -- Angle sum in a triangle
    max C (max A B) = 2 * min C (min A B) →  -- Largest angle is twice the smallest
    (a : ℝ) / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →  -- Law of sines
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →  -- Law of cosines
    (1 / 2 : ℝ) * b * c * Real.sin A = 15 * Real.sqrt 7 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_consecutive_sides_and_angle_relation_l69_6962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_more_points_specific_l69_6977

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  games_per_team : Nat
  win_probability : ℚ
  first_game_winner : Nat
  first_game_loser : Nat

/-- The probability that a team finishes with more points than another team -/
noncomputable def probability_more_points (t : SoccerTournament) : ℚ :=
  sorry

/-- The specific tournament described in the problem -/
def specific_tournament : SoccerTournament :=
  { num_teams := 8
  , games_per_team := 7
  , win_probability := 1/2
  , first_game_winner := 1  -- Representing team A
  , first_game_loser := 2   -- Representing team B
  }

/-- The main theorem to prove -/
theorem probability_more_points_specific : 
  probability_more_points specific_tournament = 1087/2048 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_more_points_specific_l69_6977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_value_range_of_a_l69_6961

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := (x - a)^2 * Real.exp x
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

-- Define the closed interval [0, 2]
def I : Set ℝ := Set.Icc 0 2

-- Theorem 1: Maximum value of M
theorem max_M_value :
  (∃ M : ℝ, ∀ x₁ x₂ : ℝ, x₁ ∈ I → x₂ ∈ I → g x₁ - g x₂ ≤ M) ∧
  (∀ M : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ I → x₂ ∈ I → g x₁ - g x₂ ≤ M) → M ≥ 112/27) :=
by sorry

-- Theorem 2: Range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ s t : ℝ, s ∈ I → t ∈ I → f a s ≥ g t) ↔
  (a ≤ -1 ∨ a ≥ 2 + Real.exp (-1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_value_range_of_a_l69_6961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_playground_triangle_leg_length_l69_6918

/-- Given a square playground of side length 2 meters divided into four congruent squares,
    each containing an isosceles right triangle with its hypotenuse along one side of the square,
    prove that if the sum of the areas of these four triangles equals the area of the large square,
    then the length of one leg of one of the isosceles triangles is √2 meters. -/
theorem playground_triangle_leg_length
  (playground_side_length : ℝ)
  (small_square_side_length : ℝ)
  (triangle_leg_length : ℝ)
  (h1 : playground_side_length = 2)
  (h2 : small_square_side_length = playground_side_length / 2)
  (h3 : 4 * (1/2 * triangle_leg_length^2) = playground_side_length^2) :
  triangle_leg_length = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_playground_triangle_leg_length_l69_6918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_solutions_l69_6965

theorem product_of_solutions : 
  ∃ x y : ℝ, (45 = 2 * x^2 + 8 * x) ∧ 
             (45 = 2 * y^2 + 8 * y) ∧ 
             (x ≠ y) ∧ 
             (x * y = -22.5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_solutions_l69_6965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_transformation_l69_6923

/-- Given a continuous function f: ℝ → ℝ, area_between_curve_and_axis f a b
    represents the area between the curve y = f(x) and the x-axis
    from x = a to x = b. --/
noncomputable def area_between_curve_and_axis (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

theorem area_transformation (f : ℝ → ℝ) (a b : ℝ) 
  (h : area_between_curve_and_axis f a b = 15) :
  area_between_curve_and_axis (fun x ↦ -2 * f (x + 4)) a b = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_transformation_l69_6923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s₂_equals_half_l69_6998

-- Define the polynomial division operation
noncomputable def poly_div (p q : ℝ → ℝ) : (ℝ → ℝ) × ℝ := sorry

-- Define x^9
noncomputable def x_ninth (x : ℝ) : ℝ := x^9

-- Define (x - 1/3)
noncomputable def x_minus_third (x : ℝ) : ℝ := x - 1/3

-- First division
noncomputable def first_division : (ℝ → ℝ) × ℝ := poly_div x_ninth x_minus_third

noncomputable def p₁ : ℝ → ℝ := (first_division).1
noncomputable def s₁ : ℝ := (first_division).2

-- Second division
noncomputable def second_division : (ℝ → ℝ) × ℝ := poly_div p₁ x_minus_third

noncomputable def p₂ : ℝ → ℝ := (second_division).1
noncomputable def s₂ : ℝ := (second_division).2

theorem s₂_equals_half : s₂ = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s₂_equals_half_l69_6998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_to_common_diff_ratio_l69_6951

/-- An arithmetic progression with first term a and common difference d -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

/-- Theorem: In an arithmetic progression where the sum of the first 30 terms
    is three times the sum of the first 15 terms, the ratio of the first term
    to the common difference is 8:1 -/
theorem first_term_to_common_diff_ratio
  (ap : ArithmeticProgression)
  (h : sum_n_terms ap 30 = 3 * sum_n_terms ap 15) :
  ap.a / ap.d = 8 := by
  sorry

#eval sum_n_terms { a := 1, d := 2 } 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_to_common_diff_ratio_l69_6951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_from_volume_l69_6997

open Real

/-- The volume of a cylinder with hemispheres at both ends -/
noncomputable def cylinderWithHemispheresVolume (radius : ℝ) (length : ℝ) : ℝ :=
  π * radius^2 * length + (4/3) * π * radius^3

/-- Theorem: Length of segment CD given volume of surrounding region -/
theorem segment_length_from_volume (radius : ℝ) (volume : ℝ) :
  radius = 4 →
  volume = 352 * π →
  cylinderWithHemispheresVolume radius (50/3) = volume :=
by
  intros h_radius h_volume
  simp [cylinderWithHemispheresVolume, h_radius, h_volume]
  -- The actual proof would go here
  sorry

#check segment_length_from_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_from_volume_l69_6997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersecting_triangle_ratios_l69_6928

/-- Triangle ACN with a circle passing through vertices A and N -/
structure CircleIntersectingTriangle where
  /-- The triangle ACN -/
  A : ℝ × ℝ
  C : ℝ × ℝ
  N : ℝ × ℝ
  /-- Point B on AC -/
  B : ℝ × ℝ
  /-- Point K on CN -/
  K : ℝ × ℝ
  /-- The circle passes through A and N -/
  circle_through_A_N : sorry
  /-- The circle intersects AC at B -/
  B_on_AC : sorry
  /-- The circle intersects CN at K -/
  K_on_CN : sorry
  /-- The ratio of areas of BCK to ACN is 1/4 -/
  area_ratio_BCK_ACN : sorry
  /-- The ratio of areas of BCN to ACK is 9/16 -/
  area_ratio_BCN_ACK : sorry

/-- Main theorem -/
theorem circle_intersecting_triangle_ratios 
  (t : CircleIntersectingTriangle) : 
  (∃ (x y : ℝ), x / y = 2 ∧ sorry) ∧ 
  (∃ (u v : ℝ), u / v = 2 / 5 ∧ sorry) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersecting_triangle_ratios_l69_6928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_special_trig_matrix_is_zero_l69_6929

open Matrix Real

theorem det_special_trig_matrix_is_zero (a b : ℝ) : 
  det !![1, sin (a + b), sin a; 
         sin (a + b), 1, sin b; 
         sin a, sin b, 1] = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_special_trig_matrix_is_zero_l69_6929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_c_is_9_l69_6925

noncomputable def digits : List ℕ := [1, 9, 8, 5]

noncomputable def expression (a b c d : ℕ) : ℝ := (a : ℝ) ^ ((b : ℝ) ^ ((c : ℝ) ^ (d : ℝ)))

def is_permutation (a b c d : ℕ) : Prop := a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem max_value_when_c_is_9 :
  ∀ a b c d : ℕ, is_permutation a b c d →
    ∃ a' b' d' : ℕ, is_permutation a' b' 9 d' ∧
      expression a' b' 9 d' ≥ expression a b c d :=
by
  sorry

#check max_value_when_c_is_9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_c_is_9_l69_6925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_possible_x_l69_6920

theorem greatest_possible_x : ∃ (x : ℕ), 
  (∀ (y : ℕ), Nat.lcm (Nat.lcm 12 18) y = 108 → y ≤ x) ∧ 
  Nat.lcm (Nat.lcm 12 18) x = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_possible_x_l69_6920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_absolute_difference_theorem_l69_6963

def is_valid_permutation (n : ℕ) (a : Fin n → Fin n) : Prop :=
  Function.Bijective a ∧ 
  (∀ i j : Fin n, i ≠ j → (a i : ℕ).dist i ≠ (a j : ℕ).dist j)

theorem permutation_absolute_difference_theorem (n : ℕ) : 
  (∃ a : Fin n → Fin n, is_valid_permutation n a) ↔ 
  (∃ k : ℕ, n = 4 * k ∨ n = 4 * k + 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_absolute_difference_theorem_l69_6963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_at_most_one_element_l69_6950

def A (a : ℝ) : Set ℝ := {x | a * x^2 - 3 * x - 4 = 0}

theorem set_A_at_most_one_element (a : ℝ) : 
  (∀ x y, x ∈ A a → y ∈ A a → x = y) → a ≤ -9/16 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_at_most_one_element_l69_6950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l69_6902

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Focal length of the ellipse -/
def focal_length : ℝ := 4

/-- Ratio of major to minor axis -/
noncomputable def axis_ratio : ℝ := Real.sqrt 3

/-- Right focus of the ellipse -/
noncomputable def right_focus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

/-- Point T on the line x = t -/
def point_T (t y : ℝ) : ℝ × ℝ := (t, y)

/-- Theorem: Standard equation of ellipse C and the value of t -/
theorem ellipse_theorem (a b t : ℝ) (y : ℝ) (h1 : y ≠ 0) (h2 : t ≠ 2) :
  ellipse_C a b a b ∧
  2 * Real.sqrt (a^2 - b^2) = focal_length ∧
  a / b = axis_ratio →
  (∀ x y, ellipse_C x y (Real.sqrt 6) (Real.sqrt 2)) ∧
  (∃ P Q : ℝ × ℝ, 
    (P.1 - Q.1) * (point_T t y).2 = (P.2 - Q.2) * (point_T t y).1 ∧
    (P.1 + Q.1) / 2 = (point_T t y).1 / 2 ∧
    (P.2 + Q.2) / 2 = (point_T t y).2 / 2 →
    t = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l69_6902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_fraction_of_internal_points_l69_6975

-- Define a triangle ABC with sides a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define an internal point P
structure InternalPoint (t : Triangle) where
  d_a : ℝ
  d_b : ℝ
  d_c : ℝ
  pos_d_a : d_a > 0
  pos_d_b : d_b > 0
  pos_d_c : d_c > 0
  in_triangle : d_a + d_b + d_c < t.a + t.b + t.c

-- Define the condition for distances forming a triangle
def CanFormTriangle (t : Triangle) (p : InternalPoint t) : Prop :=
  p.d_a < p.d_b + p.d_c ∧ p.d_b < p.d_a + p.d_c ∧ p.d_c < p.d_a + p.d_b

-- Define area functions (we'll leave them unimplemented for now)
noncomputable def AreaOfInternalPoints (t : Triangle) : ℝ := sorry

noncomputable def AreaOfTriangle (t : Triangle) : ℝ := sorry

-- Main theorem
theorem area_fraction_of_internal_points (t : Triangle) :
  ∃ (f : ℝ), f = (2 * t.a * t.b * t.c) / ((t.a + t.b) * (t.b + t.c) * (t.c + t.a)) ∧
  f = (AreaOfInternalPoints t) / (AreaOfTriangle t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_fraction_of_internal_points_l69_6975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l69_6930

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/4)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x > f (8*x - 16)} = {x : ℝ | 2 ≤ x ∧ x < 16/7} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l69_6930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_property_l69_6949

/-- A quadrilateral with specific properties -/
structure Quadrilateral where
  FG : ℝ
  GH : ℝ
  EH : ℝ
  angleE : ℝ
  angleF : ℝ
  EF : ℝ

/-- The specific quadrilateral from the problem -/
noncomputable def specialQuadrilateral : Quadrilateral where
  FG := 10
  GH := 15
  EH := 12
  angleE := 45
  angleF := 45
  EF := Real.sqrt 5637 + 12

/-- Theorem stating the properties of s and t -/
theorem quadrilateral_property (s t : ℕ) (h : s > 0 ∧ t > 0) :
  (s : ℝ) + Real.sqrt (t : ℝ) = specialQuadrilateral.EF →
  s + t = 5637 := by
  sorry

#check quadrilateral_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_property_l69_6949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_is_20_l69_6955

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1/2) * t.base * t.height

/-- Represents the park configuration -/
structure Park where
  outer : Triangle
  inner : Triangle
  inner_base_ratio : ℝ

/-- The park satisfies the given conditions -/
def parkSatisfiesConditions (p : Park) : Prop :=
  p.outer.base = 10 ∧
  p.outer.height = 5 ∧
  p.inner.base = 2 ∧
  p.inner.height = p.outer.height ∧
  p.inner_base_ratio = p.inner.base / p.outer.base

/-- Calculates the area of the park -/
noncomputable def parkArea (p : Park) : ℝ :=
  triangleArea p.outer - triangleArea p.inner

/-- Theorem: The area of the park is 20 square miles -/
theorem park_area_is_20 (p : Park) (h : parkSatisfiesConditions p) :
  parkArea p = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_is_20_l69_6955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_monotonicity_l69_6910

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^3 / 2) * x^2 - a * Real.log x

theorem extreme_value_and_monotonicity (a : ℝ) (h : a ≠ 0) :
  -- Part 1: Extreme value when a = 1
  (∀ x > 0, f 1 x ≥ 1/2) ∧ 
  (∃ x > 0, f 1 x = 1/2) ∧
  -- Part 2: Monotonicity for general a
  (a < 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < -1/a → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, -1/a < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂)) ∧
  (a > 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/a → f a x₁ > f a x₂) ∧
           (∀ x₁ x₂, 1/a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_monotonicity_l69_6910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l69_6960

def P : Finset ℤ := {4, 5, 6}
def Q : Finset ℤ := {1, 2, 3}

def set_difference (P Q : Finset ℤ) : Finset ℤ :=
  Finset.biUnion P (fun p => Finset.image (fun q => p - q) Q)

theorem proper_subsets_count :
  Finset.card (Finset.powerset (set_difference P Q) \ {set_difference P Q}) = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l69_6960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l69_6976

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x

-- State the theorem
theorem function_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : f a x₁ = 0) 
  (h2 : f a x₂ = 0) 
  (h3 : x₁ < x₂) :
  (0 < a ∧ a < 1 / Real.exp 1) ∧ (2 / (x₁ + x₂) < a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l69_6976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_l69_6969

noncomputable section

variable (a b : ℝ × ℝ)

/-- Angle between two 2D vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := 
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

/-- Theorem stating the range of the angle between vectors a and b -/
theorem angle_range (h1 : Real.sqrt (a.1^2 + a.2^2) = 2 * Real.sqrt (b.1^2 + b.2^2))
  (h2 : Real.sqrt (a.1^2 + a.2^2) ≠ 0)
  (h3 : (Real.sqrt (a.1^2 + a.2^2))^2 - 4 * (a.1 * b.1 + a.2 * b.2) ≥ 0) :
  π / 3 ≤ angle a b ∧ angle a b ≤ π :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_l69_6969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_naturals_12_to_53_l69_6931

theorem average_of_naturals_12_to_53 :
  (12 + 53 : ℚ) / 2 = 32.5 := by
  -- Convert natural numbers to rationals
  have h1 : (12 : ℚ) + (53 : ℚ) = 65
  · norm_num
  
  -- Perform the division
  have h2 : (65 : ℚ) / 2 = 32.5
  · norm_num
  
  -- Combine the steps
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_naturals_12_to_53_l69_6931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_chain_l69_6973

theorem log_chain (n x y z a : ℝ) (hn : 0 < n) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ha : 0 < a) :
  (Real.log x / Real.log n) * (Real.log y / Real.log x) * (Real.log z / Real.log y) * (Real.log a / Real.log z) = Real.log a / Real.log n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_chain_l69_6973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_area_l69_6908

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Semi-perimeter of a triangle -/
noncomputable def semiperimeter (t : Triangle) : ℝ := (perimeter t) / 2

/-- Area of a triangle using Heron's formula -/
noncomputable def area (t : Triangle) : ℝ :=
  let s := semiperimeter t
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

theorem triangle_perimeter_and_area :
  ∃ (t : Triangle), t.a = 10 ∧ t.b = 13 ∧ t.c = 7 ∧
    perimeter t = 30 ∧ area t = 20 * Real.sqrt 3 :=
by sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_area_l69_6908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_workers_needed_approx_75_l69_6957

/-- Represents the road construction project -/
structure RoadProject where
  totalLength : ℝ
  originalDuration : ℝ
  originalWorkers : ℝ
  elapsedTime : ℝ
  completedLength : ℝ

/-- Calculates the number of extra workers needed to complete the project on time -/
noncomputable def extraWorkersNeeded (project : RoadProject) : ℝ :=
  let remainingLength := project.totalLength - project.completedLength
  let remainingTime := project.originalDuration - project.elapsedTime
  let currentRate := project.completedLength / project.elapsedTime
  let requiredRate := remainingLength / remainingTime
  let totalWorkersNeeded := (requiredRate * project.originalWorkers) / currentRate
  totalWorkersNeeded - project.originalWorkers

/-- Theorem stating that approximately 75 extra workers are needed for the given project -/
theorem extra_workers_needed_approx_75 (project : RoadProject)
  (h1 : project.totalLength = 15)
  (h2 : project.originalDuration = 300)
  (h3 : project.originalWorkers = 50)
  (h4 : project.elapsedTime = 100)
  (h5 : project.completedLength = 2.5) :
  ∃ ε > 0, |extraWorkersNeeded project - 75| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_workers_needed_approx_75_l69_6957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l69_6970

theorem cos_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : π/2 < β ∧ β < π)
  (h3 : Real.cos α = 1/3)
  (h4 : Real.sin (α + β) = -3/5) :
  Real.cos β = -(6*Real.sqrt 2 + 4)/15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l69_6970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_I_maximum_mark_l69_6954

theorem paper_I_maximum_mark :
  ∀ (max_mark : ℕ) (passing_percentage : ℚ) (scored_marks failed_by : ℕ),
    passing_percentage = 42 / 100 →
    scored_marks = 60 →
    failed_by = 20 →
    (passing_percentage * max_mark) = (scored_marks + failed_by) →
    max_mark = 190 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_I_maximum_mark_l69_6954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_property_l69_6941

noncomputable def g (n : ℕ) : ℝ := 
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n + 
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem g_property (n : ℕ) : g (n + 2) - g n = g n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_property_l69_6941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_is_twenty_percent_l69_6913

/-- Calculates the percentage of loss given the cost price and selling price. -/
noncomputable def percentageLoss (costPrice sellingPrice : ℝ) : ℝ :=
  ((costPrice - sellingPrice) / costPrice) * 100

/-- Proves that the percentage of loss is 20% for the given cost and selling prices. -/
theorem loss_percentage_is_twenty_percent (costPrice sellingPrice : ℝ) 
    (h1 : costPrice = 1200)
    (h2 : sellingPrice = 960) :
    percentageLoss costPrice sellingPrice = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_is_twenty_percent_l69_6913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_final_price_l69_6943

/-- Calculates the final price of an item after discounts and compound interest --/
noncomputable def final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (weekly_interest : ℝ) (weeks : ℕ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let annual_interest := weekly_interest * 52
  price_after_discount2 * (1 + weekly_interest / 52) ^ (weeks * 52 / 12)

/-- The final price of the dress is approximately $28.30 --/
theorem dress_final_price :
  ∃ ε > 0, |final_price 50 0.3 0.2 0.03 4 - 28.30| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_final_price_l69_6943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_travel_time_is_25_seconds_l69_6991

/-- Calculates the time for a rabbit to travel from the end to the front and back to the end of a moving line of animals. -/
noncomputable def rabbitTravelTime (lineLength : ℝ) (lineSpeed : ℝ) (rabbitSpeed : ℝ) : ℝ :=
  let forwardTime := lineLength / (rabbitSpeed - lineSpeed)
  let backwardTime := lineLength / (rabbitSpeed + lineSpeed)
  forwardTime + backwardTime

/-- The time taken for a rabbit to travel from the end to the front and back to the end of a moving line of animals is 25 seconds, given the specified conditions. -/
theorem rabbit_travel_time_is_25_seconds :
  rabbitTravelTime 40 3 5 = 25 := by
  unfold rabbitTravelTime
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_travel_time_is_25_seconds_l69_6991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l69_6971

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

variable (O A B P : V)
variable (x y : ℝ)
variable (lambda : ℝ)  -- Changed 'λ' to 'lambda' to avoid syntax issues

-- Non-zero vectors OA and OB are not collinear
axiom h1 : A - O ≠ 0
axiom h2 : B - O ≠ 0
axiom h3 : ¬ ∃ (k : ℝ), A - O = k • (B - O)

-- 2OP = xOA + yOB
axiom h4 : (2 : ℝ) • (P - O) = x • (A - O) + y • (B - O)

-- PA = λAB
axiom h5 : A - P = lambda • (B - A)

-- Theorem to prove
theorem vector_relation : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l69_6971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l69_6993

open Real

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 
  (arctan (x / 3))^2 + Real.pi * arctan (x / 3) - 
  (arctan (x / 3))^2 + (Real.pi^2 / 18) * (x^2 - 3*x + 9)

-- State the theorem
theorem g_range : 
  (∀ x : ℝ, g x ≥ 19 * Real.pi^2 / 48) ∧ 
  (∀ y : ℝ, y ≥ 19 * Real.pi^2 / 48 → ∃ x : ℝ, g x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l69_6993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_apollonius_circle_l69_6980

-- Define the circles and points
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the given setup
def setup (O₁ O₂ A B P C D : EuclideanSpace ℝ (Fin 2)) (r₁ r₂ : ℝ) : Prop :=
  ∃ (circle₁ : Circle) (circle₂ : Circle),
    circle₁ = ⟨O₁, r₁⟩ ∧
    circle₂ = ⟨O₂, r₂⟩ ∧
    dist A O₁ = r₁ ∧
    dist A O₂ = r₂ ∧
    dist B O₁ = r₁ ∧
    dist B O₂ = r₂ ∧
    (dist P C) / (dist P D) = (dist C O₁) / (dist D O₂)

-- Define the Apollonius circle
def apollonius_circle (O₁ O₂ A : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {P | (dist P O₁) / (dist P O₂) = (dist A O₁) / (dist A O₂)}

-- State the theorem
theorem point_on_apollonius_circle
  (O₁ O₂ A B P C D : EuclideanSpace ℝ (Fin 2)) (r₁ r₂ : ℝ)
  (h : setup O₁ O₂ A B P C D r₁ r₂) :
  P ∈ apollonius_circle O₁ O₂ A :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_apollonius_circle_l69_6980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_four_root_three_thirds_l69_6927

/-- A regular hexagon divided into 12 equal triangular slices -/
structure HexagonDartboard where
  /-- Side length of the hexagon -/
  side_length : ℝ
  /-- Assumption that side_length is positive -/
  side_length_pos : 0 < side_length

/-- The ratio of areas of larger to smaller triangular regions in the hexagon dartboard -/
noncomputable def area_ratio (h : HexagonDartboard) : ℝ :=
  let larger_triangle_area := h.side_length * (h.side_length * Real.sqrt 3) / 2
  let smaller_triangle_area := (h.side_length * Real.sqrt 3 / 2) * (h.side_length * Real.sqrt 3 / 2) / 2
  larger_triangle_area / smaller_triangle_area

/-- Theorem stating that the area ratio is 4√3/3 -/
theorem area_ratio_is_four_root_three_thirds (h : HexagonDartboard) :
  area_ratio h = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_four_root_three_thirds_l69_6927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l69_6966

theorem problem_solution : 
  ∀ (x : ℝ) (a m n : ℝ),
  (3 * (9:ℝ)^x * 81 = (3:ℝ)^21) →
  (a^m = 2) →
  (a^n = 5) →
  (x = 8 ∧ a^(3*m + 2*n) = 200) :=
by
  intros x a m n h1 h2 h3
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l69_6966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_ratio_vector_combination_l69_6946

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points A, B, C, and P
variable (A B C P : V)

-- Define the conditions
variable (h1 : C = (1/2 : ℝ) • A + (1/2 : ℝ) • B)  -- C is the midpoint of AB
variable (h2 : ∃ t : ℝ, t ∈ Set.Icc (0 : ℝ) (1 : ℝ) ∧ P = t • A + (1 - t) • C)  -- P is on AC
variable (h3 : ∃ k : ℝ, k > 0 ∧ P - A = (4 * k) • (C - P))  -- AP:PC = 4:1

-- State the theorem
theorem midpoint_ratio_vector_combination :
  P = (9/10 : ℝ) • A + (1/10 : ℝ) • B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_ratio_vector_combination_l69_6946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_QF_is_eight_thirds_l69_6952

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix
def directrix : ℝ → ℝ := λ x ↦ -2

-- Define a point on the directrix
variable (P : ℝ × ℝ)

-- Define Q as the intersection of PF and the parabola
variable (Q : ℝ × ℝ)

-- State that P is on the directrix
axiom P_on_directrix : P.2 = directrix P.1

-- State that Q is on the parabola
axiom Q_on_parabola : parabola Q.1 Q.2

-- State that P, F, and Q are collinear
axiom PFQ_collinear : ∃ t : ℝ, Q - focus = t • (P - focus)

-- State the vector relationship
axiom vector_relation : P - focus = 3 • (Q - focus)

-- Theorem to prove
theorem distance_QF_is_eight_thirds :
  ‖Q - focus‖ = 8/3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_QF_is_eight_thirds_l69_6952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_correct_l69_6945

/-- The function for which we're finding the symmetry axis -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x + 3 * Real.pi / 4)

/-- The proposed symmetry axis -/
noncomputable def symmetry_axis : ℝ := -Real.pi / 12

/-- Theorem stating that the symmetry axis of f is correct -/
theorem symmetry_axis_correct :
  ∀ x : ℝ, f (symmetry_axis - x) = f (symmetry_axis + x) := by
  sorry

#check symmetry_axis_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_correct_l69_6945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_product_l69_6938

/-- Predicate indicating if a set is an isosceles right triangle -/
def IsIsoscelesRightTriangle (triangle : Set ℝ × Set ℝ) : Prop := sorry

/-- Predicate indicating if a triangle has a given inradius -/
def HasInradius (triangle : Set ℝ × Set ℝ) (r : ℝ) : Prop := sorry

/-- Predicate indicating if a circle of radius t is tangent to the hypotenuse, 
    the incircle, and one leg of the triangle -/
def IsTangentCircle (triangle : Set ℝ × Set ℝ) (r t : ℝ) : Prop := sorry

theorem tangent_circle_product (r t : ℝ) : 
  r = 1 + Real.sin (π / 8) →
  t > 0 →
  (∃ (triangle : Set ℝ × Set ℝ), 
    IsIsoscelesRightTriangle triangle ∧ 
    HasInradius triangle r ∧
    IsTangentCircle triangle r t) →
  r * t = (2 + Real.sqrt 2) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_product_l69_6938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l69_6964

-- Define the tangent line and the curve
noncomputable def tangent_line (x : ℝ) (b : ℝ) : ℝ := (1/2) * x + b
noncomputable def ln_curve (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem tangent_line_to_ln_curve (b : ℝ) :
  (∃ x : ℝ, x > 0 ∧ tangent_line x b = ln_curve x ∧
    (∀ y : ℝ, y > 0 → tangent_line y b ≥ ln_curve y)) →
  b = Real.log 2 - 1 := by
  sorry

-- Additional lemma to help with the proof
lemma tangent_point_is_two :
  ∀ b : ℝ, (∃ x : ℝ, x > 0 ∧ tangent_line x b = ln_curve x ∧
    (∀ y : ℝ, y > 0 → tangent_line y b ≥ ln_curve y)) →
  ∃ x : ℝ, x = 2 ∧ tangent_line x b = ln_curve x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l69_6964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l69_6905

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := f ((2 * x) + Real.pi / 3)

theorem g_properties :
  (∀ x, g x = 2 * Real.cos (4 * x)) ∧
  (∀ x, g (-x) = g x) ∧
  (∀ k : ℤ, g ((2 * k - 1) * Real.pi / 8) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l69_6905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_sum_l69_6924

theorem triangle_altitude_sum (a b : ℕ) (h : Nat.Coprime a b) : 
  let triangle_sides : Fin 3 → ℕ := ![48, 55, 73]
  let shortest_altitude : ℚ := a / b
  (∀ (i : Fin 3), 
    2 * (triangle_sides i) * shortest_altitude ≤ 
    (triangle_sides ((i + 1) % 3)) * (triangle_sides ((i + 2) % 3))) →
  a + b = 2713 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_sum_l69_6924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_monochromatic_inscribed_trapezoid_l69_6996

-- Define a type for colors
inductive Color
| Red | Blue | Green | Yellow | Orange | Purple | Pink

-- Define a point on a circle
structure Point where
  angle : ℝ
  color : Color

-- Define a circle with colored points
structure ColoredCircle where
  radius : ℝ
  points : Set Point

-- Define a trapezoid
structure Trapezoid where
  vertices : Fin 4 → Point

-- Function to check if a trapezoid is inscribed in a circle
def isInscribed (t : Trapezoid) (c : ColoredCircle) : Prop :=
  ∀ v : Fin 4, t.vertices v ∈ c.points

-- Function to check if all vertices of a trapezoid have the same color
def sameColor (t : Trapezoid) : Prop :=
  ∀ i j : Fin 4, (t.vertices i).color = (t.vertices j).color

-- Theorem statement
theorem exists_monochromatic_inscribed_trapezoid (c : ColoredCircle) :
  ∃ t : Trapezoid, isInscribed t c ∧ sameColor t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_monochromatic_inscribed_trapezoid_l69_6996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_inequality_l69_6947

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_inequality (A B C : Circle)
    (h1 : A.radius > B.radius)
    (h2 : B.radius > C.radius) :
    ¬(A.radius - B.radius + C.radius = distance A.center B.center) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_inequality_l69_6947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_to_intersection_distance_is_sqrt2_l69_6959

/-- A regular octagon inscribed in a unit circle -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : ∀ i : Fin 8, (vertices i).1^2 + (vertices i).2^2 = 1
  is_octagon : ∀ i : Fin 8, vertices (i + 1) ≠ vertices i

/-- The intersection point of two diagonals in a regular octagon -/
noncomputable def diagonalIntersection (octagon : RegularOctagon) : ℝ × ℝ := sorry

/-- The distance from a vertex to the diagonal intersection point -/
noncomputable def vertexToIntersectionDistance (octagon : RegularOctagon) (v : Fin 8) : ℝ :=
  let I := diagonalIntersection octagon
  Real.sqrt ((octagon.vertices v).1 - I.1)^2 + ((octagon.vertices v).2 - I.2)^2

theorem vertex_to_intersection_distance_is_sqrt2 (octagon : RegularOctagon) (v : Fin 8) :
  vertexToIntersectionDistance octagon v = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_to_intersection_distance_is_sqrt2_l69_6959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_inscribed_triangle_l69_6958

/-- A triangle inscribed in a circle of radius 1 -/
structure InscribedTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  inscribed : (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (C.1^2 + C.2^2 = 1)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: There exists a non-rectangular triangle inscribed in a circle of radius 1,
    where the sum of the squares of the lengths of two sides is equal to 4 -/
theorem exists_special_inscribed_triangle :
  ∃ (t : InscribedTriangle),
    (distance t.A t.B)^2 + (distance t.A t.C)^2 = 4 ∧
    ¬((distance t.A t.B)^2 + (distance t.B t.C)^2 = (distance t.A t.C)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_inscribed_triangle_l69_6958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_medians_l69_6917

/-- A right triangle with specific median lengths has a hypotenuse of length 2√14 -/
theorem right_triangle_with_medians (X Y Z : ℝ × ℝ) : 
  -- X, Y, Z form a right triangle with right angle at Z
  (X.1 - Z.1) * (Y.1 - Z.1) + (X.2 - Z.2) * (Y.2 - Z.2) = 0 →
  -- The median from X to YZ has length 5
  ((X.1 - (Y.1 + Z.1) / 2)^2 + (X.2 - (Y.2 + Z.2) / 2)^2 = 25) →
  -- The median from Y to XZ has length 3√5
  ((Y.1 - (X.1 + Z.1) / 2)^2 + (Y.2 - (X.2 + Z.2) / 2)^2 = 45) →
  -- Then the length of hypotenuse XY is 2√14
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_medians_l69_6917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_thermos_days_l69_6919

/-- Represents the number of days a teacher fills her thermos in a week -/
def days_filling_thermos (thermos_capacity : ℚ) (fills_per_day : ℚ) (current_weekly_consumption : ℚ) (consumption_ratio : ℚ) : ℚ :=
  let previous_weekly_consumption := current_weekly_consumption / consumption_ratio
  let fills_per_week := previous_weekly_consumption / thermos_capacity
  fills_per_week / fills_per_day

/-- Theorem stating that given the problem conditions, the teacher fills her thermos 4 days a week -/
theorem teacher_thermos_days :
  days_filling_thermos 20 2 40 (1/4) = 4 :=
by
  -- Unfold the definition of days_filling_thermos
  unfold days_filling_thermos
  -- Simplify the arithmetic expressions
  simp [Rat.div_def, Rat.mul_def]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_thermos_days_l69_6919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_l69_6939

theorem triangle_angle_A (A B C : ℝ) (b : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = Real.sqrt 2 ∧  -- Given condition
  B = π / 4 ∧  -- Given condition
  b = 2 ∧  -- Given condition
  Real.sin C / C = Real.sin B / b  -- Law of Sines
  → A = 7 * π / 12 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_l69_6939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_probability_is_75_196_l69_6974

/-- A triangle with side lengths 13, 20, and 21 -/
structure Triangle where
  a : ℚ
  b : ℚ
  c : ℚ
  ha : a = 13
  hb : b = 20
  hc : c = 21

/-- The probability that a circle with radius 1 centered at a random point inside the triangle
    intersects at least one side of the triangle -/
def intersectionProbability (t : Triangle) : ℚ :=
  75 / 196

/-- Theorem stating that the intersection probability for the given triangle is 75/196 -/
theorem intersection_probability_is_75_196 (t : Triangle) :
  intersectionProbability t = 75 / 196 := by
  sorry

#check intersection_probability_is_75_196

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_probability_is_75_196_l69_6974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_halima_beckham_l69_6900

/-- The age difference between Halima and Beckham given their age ratio and total age -/
theorem age_difference_halima_beckham (total_age : ℕ) (ratio_halima : ℕ) (ratio_beckham : ℕ) (ratio_michelle : ℕ) : 
  total_age = 126 → 
  ratio_halima = 4 →
  ratio_beckham = 3 →
  ratio_michelle = 7 →
  (ratio_halima * total_age) / (ratio_halima + ratio_beckham + ratio_michelle) - 
  (ratio_beckham * total_age) / (ratio_halima + ratio_beckham + ratio_michelle) = 9 := by
  intro h1 h2 h3 h4
  sorry

#check age_difference_halima_beckham

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_halima_beckham_l69_6900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_in_fourth_quadrant_l69_6901

/-- A quadratic function that does not intersect the x-axis -/
def no_intersection (k : ℝ) : Prop :=
  ∀ x, x^2 - 2*x - k ≠ 0

/-- The vertex of a quadratic function -/
noncomputable def vertex (k : ℝ) : ℝ × ℝ :=
  let x := -(k + 1) / 2
  (x, x^2 + (k + 1)*x + k)

/-- The fourth quadrant -/
def fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

/-- Theorem: If y = x^2 - 2x - k does not intersect the x-axis,
    then the vertex of y = x^2 + (k+1)x + k is in the fourth quadrant -/
theorem vertex_in_fourth_quadrant (k : ℝ) :
  no_intersection k → fourth_quadrant (vertex k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_in_fourth_quadrant_l69_6901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_given_parabola_l69_6978

/-- A parabola with equation y = ax² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ :=
  (0, 1 / (4 * p.a) + p.b)

/-- The given parabola y = 9x² + 6 -/
def given_parabola : Parabola :=
  { a := 9, b := 6 }

theorem focus_of_given_parabola :
  focus given_parabola = (0, 217 / 36) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_given_parabola_l69_6978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l69_6906

-- Define the ellipse E
noncomputable def E (x y : ℝ) (a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

-- Define the slope of AF
noncomputable def slope_AF : ℝ := Real.sqrt 6 / 3

-- Define the point A
def A : ℝ × ℝ := (0, -2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the line l
noncomputable def line_l (k : ℝ) (x : ℝ) : ℝ := k * x - 2

-- Define the area of triangle OPQ
noncomputable def area_OPQ (k : ℝ) : ℝ := 4 * Real.sqrt 2 * (Real.sqrt (4 * k^2 - 1)) / (1 + 4 * k^2)

theorem ellipse_and_max_area_line 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 3 / 2) 
  (h4 : ∃ c : ℝ, c > 0 ∧ 2 / c = slope_AF) :
  (∀ x y : ℝ, E x y a b ↔ x^2/8 + y^2/2 = 1) ∧ 
  (∃ k : ℝ, k = Real.sqrt 3 / 2 ∧ 
    (∀ k' : ℝ, area_OPQ k ≥ area_OPQ k') ∧
    (∀ x : ℝ, line_l k x = Real.sqrt 3 / 2 * x - 2 ∨ line_l k x = -Real.sqrt 3 / 2 * x - 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l69_6906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_ratio_fourth_term_l69_6948

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n ↦ a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geometric_sum_ratio_fourth_term (a₁ : ℝ) :
  let q := (2 : ℝ)
  let seq := geometric_sequence a₁ q
  let S₅ := geometric_sum a₁ q 5
  S₅ / seq 4 = 31 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_ratio_fourth_term_l69_6948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_and_condition_l69_6935

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

-- Define function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

theorem solution_sets_and_condition (a : ℝ) :
  (a ≠ 0 →
    ((a > 0 → {x : ℝ | f a x ≤ 3*a^2 + 1} = Set.Icc (-a) (3*a)) ∧
     (a < 0 → {x : ℝ | f a x ≤ 3*a^2 + 1} = Set.Icc (-3*a) a))) ∧
  (∀ x ∈ A, f a x > 0) ↔ a < 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_and_condition_l69_6935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_specific_repeating_decimals_l69_6988

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingDenominator : ℕ
  nonNegRepeating : 0 ≤ repeating
  lessThanOne : repeating < 1
  positiveDenominator : repeatingDenominator > 0

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (r : RepeatingDecimal) : ℚ :=
  r.nonRepeating + r.repeating / (1 - (1 : ℚ) / r.repeatingDenominator)

/-- The sum of specific repeating decimals equals 1315/9999 -/
theorem sum_specific_repeating_decimals :
  let r1 := RepeatingDecimal.toRational {
    nonRepeating := 0,
    repeating := 1/10,
    repeatingDenominator := 10,
    nonNegRepeating := by norm_num,
    lessThanOne := by norm_num,
    positiveDenominator := by norm_num
  }
  let r2 := RepeatingDecimal.toRational {
    nonRepeating := 0,
    repeating := 2/100,
    repeatingDenominator := 100,
    nonNegRepeating := by norm_num,
    lessThanOne := by norm_num,
    positiveDenominator := by norm_num
  }
  let r3 := RepeatingDecimal.toRational {
    nonRepeating := 0,
    repeating := 2/10000,
    repeatingDenominator := 10000,
    nonNegRepeating := by norm_num,
    lessThanOne := by norm_num,
    positiveDenominator := by norm_num
  }
  r1 + r2 + r3 = 1315 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_specific_repeating_decimals_l69_6988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l69_6916

/-- The function f(x) = √(x-1) + √(x(3-x)) -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) + Real.sqrt (x * (3 - x))

/-- The domain of f is {x | 1 ≤ x ≤ 3} -/
def f_domain : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

/-- Proof that the domain of f is {x | 1 ≤ x ≤ 3} -/
theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = f_domain := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l69_6916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_correct_l69_6987

/-- The x-coordinate of the point on the x-axis that is equidistant from A(-3,0) and B(2,5) -/
def equidistant_point : ℝ := 2

/-- Point A coordinates -/
def A : ℝ × ℝ := (-3, 0)

/-- Point B coordinates -/
def B : ℝ × ℝ := (2, 5)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem equidistant_point_correct :
  distance (equidistant_point, 0) A = distance (equidistant_point, 0) B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_correct_l69_6987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_relationships_l69_6911

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the theorem
theorem geometric_relationships 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (perpendicular_line_plane m α → perpendicular_line_plane n β → perpendicular m n → perpendicular_plane α β) ∧ 
  (perpendicular_line_plane m α → parallel_line_plane n β → parallel_plane α β → perpendicular m n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_relationships_l69_6911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_reciprocal_sum_l69_6926

noncomputable section

/-- Parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Point D -/
def D : ℝ × ℝ := (1, 1)

/-- Distance squared between two points -/
def distanceSquared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Sum of reciprocals of squared distances -/
def s (P Q : ℝ × ℝ) : ℝ :=
  1 / distanceSquared P D + 1 / distanceSquared Q D

/-- Theorem statement -/
theorem chord_reciprocal_sum :
  ∀ (P Q : ℝ × ℝ),
    P.2 = parabola P.1 →
    Q.2 = parabola Q.1 →
    (Q.2 - P.2) / (Q.1 - P.1) = (D.2 - P.2) / (D.1 - P.1) →
    s P Q = 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_reciprocal_sum_l69_6926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_P_abscissa_l69_6972

-- Define the line l: x - y + 1 = 0
def line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the circle C: (x-2)² + (y-1)² = 1
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Define the condition for point P
def point_P (x y : ℝ) : Prop := line x y

-- Define the condition for points M and N
def point_M_N (xm ym xn yn : ℝ) : Prop := circle_C xm ym ∧ circle_C xn yn

-- Define the angle condition
noncomputable def angle_condition (x y xm ym xn yn : ℝ) : Prop :=
  let angle := Real.arccos ((xm - x) * (xn - x) + (ym - y) * (yn - y)) /
    (((xm - x)^2 + (ym - y)^2)^(1/2) * ((xn - x)^2 + (yn - y)^2)^(1/2))
  angle = Real.pi / 3  -- 60 degrees in radians

-- Theorem statement
theorem range_of_P_abscissa :
  ∀ x y xm ym xn yn : ℝ,
    point_P x y →
    point_M_N xm ym xn yn →
    angle_condition x y xm ym xn yn →
    0 ≤ x ∧ x ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_P_abscissa_l69_6972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l69_6907

/-- Represents the speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 120

/-- Represents the time taken by the train to cross a pole in seconds -/
noncomputable def crossing_time : ℝ := 15

/-- Converts km/hr to m/s -/
noncomputable def km_hr_to_m_s (speed : ℝ) : ℝ := speed * 1000 / 3600

/-- Calculates the length of the train in meters -/
noncomputable def train_length : ℝ := km_hr_to_m_s train_speed * crossing_time

/-- Theorem stating that the length of the train is approximately 499.95 meters -/
theorem train_length_approx :
  |train_length - 499.95| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l69_6907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l69_6936

-- Define the function f(x) = 2^x
noncomputable def f (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem range_of_f :
  let S := {y | ∃ x ∈ Set.Icc 0 3, f x = y}
  S = Set.Icc 1 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l69_6936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l69_6995

theorem cos_alpha_value (α : ℝ) 
  (h : ∃ (r : ℝ), r ≠ 0 ∧ Real.sin (2*α) = r * Real.sin α ∧ Real.sin (4*α) = r * Real.sin (2*α)) : 
  Real.cos α = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l69_6995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l69_6912

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(q x a) → ¬(p x)) ∧ (∃ x : ℝ, ¬(p x) ∧ q x a)) →
  (∀ a : ℝ, a ∈ Set.Ici 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l69_6912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abby_in_seat_three_l69_6933

-- Define the people
inductive Person : Type
| Abby : Person
| Bret : Person
| Carl : Person
| Dana : Person

-- Define the seats
def Seat := Fin 4

-- Define a seating arrangement as a function from seats to people
def SeatingArrangement := Seat → Person

def is_next_to (arr : SeatingArrangement) (p1 p2 : Person) : Prop :=
  ∃ (s1 s2 : Seat), (s1.val + 1 = s2.val ∨ s2.val + 1 = s1.val) ∧ arr s1 = p1 ∧ arr s2 = p2

def is_between (arr : SeatingArrangement) (p1 p2 p3 : Person) : Prop :=
  ∃ (s1 s2 s3 : Seat), s1.val < s2.val ∧ s2.val < s3.val ∧ 
    arr s1 = p1 ∧ arr s2 = p2 ∧ arr s3 = p3

theorem abby_in_seat_three :
  ∀ (arr : SeatingArrangement),
    (is_next_to arr Person.Bret Person.Carl) ∧
    ¬(is_between arr Person.Dana Person.Abby Person.Carl) ∧
    (arr ⟨0, by norm_num⟩ = Person.Bret) →
    (arr ⟨2, by norm_num⟩ = Person.Abby) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abby_in_seat_three_l69_6933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_A_production_rate_l69_6986

noncomputable section

-- Define the production rates and times for each machine
def sprockets_produced : ℝ := 990
def machine_A_rate : ℝ := 1.105 -- This is what we want to prove
def machine_Q_rate : ℝ := machine_A_rate * 1.12
def machine_R_rate : ℝ := machine_A_rate * 1.08

-- Define the time differences
def time_diff_P_Q : ℝ := 12
def time_diff_P_R : ℝ := 8

-- Define the production times for each machine
noncomputable def machine_Q_time : ℝ := sprockets_produced / machine_Q_rate
noncomputable def machine_P_time : ℝ := machine_Q_time + time_diff_P_Q
noncomputable def machine_R_time : ℝ := machine_P_time - time_diff_P_R

-- Theorem to prove
theorem machine_A_production_rate : 
  sprockets_produced = machine_P_time * machine_A_rate ∧
  sprockets_produced = machine_Q_time * machine_Q_rate ∧
  sprockets_produced = machine_R_time * machine_R_rate := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_A_production_rate_l69_6986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_fish_population_l69_6968

/-- Approximates the total number of fish in a pond given tagging and recapture data. -/
theorem approximate_fish_population (tagged_fish : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  tagged_fish = 60 →
  second_catch = 50 →
  tagged_in_second = 2 →
  (tagged_fish : ℚ) / ((tagged_fish : ℚ) + (second_catch - tagged_in_second : ℚ)) = 
    (tagged_in_second : ℚ) / (second_catch : ℚ) →
  ⌊(tagged_fish : ℚ) * (second_catch : ℚ) / (tagged_in_second : ℚ)⌋ = 1500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_fish_population_l69_6968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_2016_l69_6994

-- Define the function f
noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (1/2) * x^2 + 2 * x * k - 2016 * Real.log x

-- State the theorem
theorem derivative_at_2016 : 
  ∃ k : ℝ, (deriv (f · k)) 2016 = -2015 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_2016_l69_6994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_eq_half_diff_AE_BE_l69_6989

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))
-- Define point E on AB
variable (E : EuclideanSpace ℝ (Fin 2))
-- Define points M and N
variable (M N : EuclideanSpace ℝ (Fin 2))

-- ABC is isosceles
variable (h_isosceles : ‖A - C‖ = ‖B - C‖)

-- E lies on AB
variable (h_E_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • B)

-- M is the point where incircle of ACE touches CE
variable (h_M_on_CE : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ M = (1 - s) • C + s • E)

-- N is the point where incircle of ECB touches CE
variable (h_N_on_CE : ∃ u : ℝ, 0 < u ∧ u < 1 ∧ N = (1 - u) • C + u • E)

-- Theorem statement
theorem length_MN_eq_half_diff_AE_BE :
  ‖M - N‖ = |‖A - E‖ - ‖B - E‖| / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_eq_half_diff_AE_BE_l69_6989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_eq_1_range_of_a_non_empty_solution_min_value_of_f_l69_6921

-- Define the function f(x) = 2|x-3|+|x-4|
def f (x : ℝ) : ℝ := 2 * abs (x - 3) + abs (x - 4)

-- Theorem for the solution set when a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | f x < 2} = {x : ℝ | 8/3 < x ∧ x < 4} := by sorry

-- Theorem for the range of a when the solution set is non-empty
theorem range_of_a_non_empty_solution :
  {a : ℝ | ∃ x, f x < 2*a} = {a : ℝ | 1/2 < a} := by sorry

-- Auxiliary theorem: minimum value of f(x) is 1
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_eq_1_range_of_a_non_empty_solution_min_value_of_f_l69_6921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_branch_profit_maximization_l69_6932

noncomputable def annual_sales_volume (x : ℝ) : ℝ := (12 - x)^2 * 10000

noncomputable def annual_profit (x : ℝ) : ℝ := (x - 6) * annual_sales_volume x / 10000

theorem branch_profit_maximization (x : ℝ) (h1 : 9 ≤ x) (h2 : x ≤ 11) :
  annual_profit x = x^3 - 30*x^2 + 288*x - 864 ∧
  (∀ y, 9 ≤ y → y ≤ 11 → annual_profit y ≤ annual_profit 9) ∧
  annual_profit 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_branch_profit_maximization_l69_6932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_P_travel_time_l69_6909

/-- The travel time of Ferry P in hours -/
def T : ℝ := sorry

/-- The distance traveled by Ferry P in kilometers -/
def D : ℝ := sorry

/-- The speed of Ferry P in kilometers per hour -/
def speed_P : ℝ := 8

/-- The speed difference between Ferry Q and Ferry P in kilometers per hour -/
def speed_diff : ℝ := 4

/-- The time difference between Ferry Q and Ferry P in hours -/
def time_diff : ℝ := 1

/-- Ferry P's distance-speed-time relationship -/
axiom ferry_P_equation : D = speed_P * T

/-- Ferry Q's distance is twice that of Ferry P -/
axiom ferry_Q_distance : 2 * D = (speed_P + speed_diff) * (T + time_diff)

theorem ferry_P_travel_time : T = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_P_travel_time_l69_6909
