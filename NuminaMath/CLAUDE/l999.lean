import Mathlib

namespace NUMINAMATH_CALUDE_max_triangle_area_l999_99979

-- Define the ellipse C₁
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola C₂
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define a point on the parabola (excluding origin)
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2 ∧ P ≠ (0, 0)

-- Define the tangent line from a point on the parabola
def tangent_line (P : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  point_on_parabola P ∧ ∃ (m b : ℝ), ∀ x, l x = m * x + b

-- Define the intersection points of the tangent with the ellipse
def intersection_points (A B : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
  l A.1 = A.2 ∧ l B.1 = B.2

-- Theorem statement
theorem max_triangle_area 
  (P : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  tangent_line P l → intersection_points A B l →
  ∃ (S : ℝ), S ≤ 8 * Real.sqrt 3 ∧ 
  (∃ (P' : ℝ × ℝ) (l' : ℝ → ℝ) (A' B' : ℝ × ℝ),
    tangent_line P' l' ∧ intersection_points A' B' l' ∧
    S = 8 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l999_99979


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l999_99973

def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l999_99973


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l999_99964

theorem arithmetic_sequence_length : ∀ (a₁ d : ℤ) (n : ℕ),
  a₁ = 165 ∧ d = -6 ∧ (a₁ + d * (n - 1 : ℤ) ≤ 24) ∧ (a₁ + d * ((n - 1) - 1 : ℤ) > 24) →
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l999_99964


namespace NUMINAMATH_CALUDE_damaged_tins_percentage_l999_99930

theorem damaged_tins_percentage (cases : ℕ) (tins_per_case : ℕ) (remaining_tins : ℕ) : 
  cases = 15 → tins_per_case = 24 → remaining_tins = 342 →
  (cases * tins_per_case - remaining_tins) / (cases * tins_per_case) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_damaged_tins_percentage_l999_99930


namespace NUMINAMATH_CALUDE_min_sum_of_four_primes_l999_99901

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem min_sum_of_four_primes :
  ∀ a b c d s : ℕ,
  is_prime a → is_prime b → is_prime c → is_prime d → is_prime s →
  s = a + b + c + d →
  s ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_four_primes_l999_99901


namespace NUMINAMATH_CALUDE_probability_two_diamonds_one_ace_l999_99915

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of diamonds in a standard deck -/
def DiamondCount : ℕ := 13

/-- Number of aces in a standard deck -/
def AceCount : ℕ := 4

/-- Probability of drawing two diamonds followed by an ace from a standard deck -/
def probabilityTwoDiamondsOneAce : ℚ :=
  (DiamondCount : ℚ) / StandardDeck *
  (DiamondCount - 1) / (StandardDeck - 1) *
  ((DiamondCount : ℚ) / StandardDeck * (AceCount - 1) / (StandardDeck - 2) +
   (StandardDeck - DiamondCount : ℚ) / StandardDeck * AceCount / (StandardDeck - 2))

theorem probability_two_diamonds_one_ace :
  probabilityTwoDiamondsOneAce = 29 / 11050 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_diamonds_one_ace_l999_99915


namespace NUMINAMATH_CALUDE_pats_calculation_l999_99919

theorem pats_calculation (x : ℝ) : (x / 8 - 20 = 12) → (x * 8 + 20 = 2068) := by
  sorry

end NUMINAMATH_CALUDE_pats_calculation_l999_99919


namespace NUMINAMATH_CALUDE_spinsters_and_cats_l999_99959

theorem spinsters_and_cats (spinsters : ℕ) (cats : ℕ) : 
  spinsters = 12 →
  (spinsters : ℚ) / cats = 2 / 9 →
  cats > spinsters →
  cats - spinsters = 42 := by
sorry

end NUMINAMATH_CALUDE_spinsters_and_cats_l999_99959


namespace NUMINAMATH_CALUDE_max_non_managers_l999_99981

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 → 
  (managers : ℚ) / non_managers > 7 / 32 → 
  non_managers ≤ 36 :=
by
  sorry

end NUMINAMATH_CALUDE_max_non_managers_l999_99981


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l999_99957

noncomputable def s (t : ℝ) : ℝ := (1/4) * t^4 - 3

theorem instantaneous_velocity_at_5 :
  (deriv s) 5 = 125 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l999_99957


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_power_l999_99943

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other. -/
def symmetric_x_axis (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.1 ∧ P.2 = -Q.2

theorem symmetry_implies_sum_power (a b : ℝ) :
  symmetric_x_axis (a, 3) (4, b) → (a + b)^2021 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_power_l999_99943


namespace NUMINAMATH_CALUDE_motorcycle_speed_l999_99944

/-- Motorcycle trip problem -/
theorem motorcycle_speed (total_distance : ℝ) (ab_distance : ℝ) (bc_distance : ℝ)
  (inclination_angle : ℝ) (total_avg_speed : ℝ) (ab_time_ratio : ℝ) :
  total_distance = ab_distance + bc_distance →
  bc_distance = ab_distance / 2 →
  ab_distance = 120 →
  inclination_angle = 10 →
  total_avg_speed = 30 →
  ab_time_ratio = 3 →
  ∃ (bc_avg_speed : ℝ), bc_avg_speed = 40 := by
  sorry

#check motorcycle_speed

end NUMINAMATH_CALUDE_motorcycle_speed_l999_99944


namespace NUMINAMATH_CALUDE_standard_deviation_calculation_l999_99980

/-- A normal distribution with mean μ and standard deviation σ -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ

/-- The value that is exactly k standard deviations away from the mean -/
def value_k_std_dev_from_mean (d : NormalDistribution) (k : ℝ) : ℝ :=
  d.μ - k * d.σ

theorem standard_deviation_calculation (d : NormalDistribution) 
  (h1 : d.μ = 16.2)
  (h2 : value_k_std_dev_from_mean d 2 = 11.6) : 
  d.σ = 2.3 := by
sorry

end NUMINAMATH_CALUDE_standard_deviation_calculation_l999_99980


namespace NUMINAMATH_CALUDE_tablet_diagonal_problem_l999_99924

/-- Given two square tablets, if the larger tablet has an 8-inch diagonal and its screen area is 7.5 square inches greater than the smaller tablet's, then the diagonal of the smaller tablet is 7 inches. -/
theorem tablet_diagonal_problem (d : ℝ) : 
  let large_diagonal : ℝ := 8
  let large_area : ℝ := (large_diagonal / Real.sqrt 2) ^ 2
  let small_area : ℝ := (d / Real.sqrt 2) ^ 2
  large_area = small_area + 7.5 → d = 7 := by sorry

end NUMINAMATH_CALUDE_tablet_diagonal_problem_l999_99924


namespace NUMINAMATH_CALUDE_age_difference_theorem_l999_99905

/-- Represents a two-digit age --/
structure TwoDigitAge where
  tens : Nat
  ones : Nat
  h1 : tens ≤ 9
  h2 : ones ≤ 9

def TwoDigitAge.toNat (age : TwoDigitAge) : Nat :=
  10 * age.tens + age.ones

theorem age_difference_theorem (anna ella : TwoDigitAge) 
  (h : anna.tens = ella.ones ∧ anna.ones = ella.tens) 
  (future_relation : (anna.toNat + 10) = 3 * (ella.toNat + 10)) :
  anna.toNat - ella.toNat = 54 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_theorem_l999_99905


namespace NUMINAMATH_CALUDE_polynomial_coefficient_product_l999_99909

/-- Given a polynomial x^4 - (a-2)x^3 + 5x^2 + (b+3)x - 1 where the coefficients of x^3 and x are zero, prove that ab = -6 -/
theorem polynomial_coefficient_product (a b : ℝ) : 
  (a - 2 = 0) → (b + 3 = 0) → a * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_product_l999_99909


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l999_99913

def cost_price : ℝ := 120
def selling_price : ℝ := 102
def gain_price : ℝ := 144
def gain_percentage : ℝ := 20

theorem loss_percentage_calculation :
  (cost_price - selling_price) / cost_price * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l999_99913


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l999_99926

/-- The function f(x) = x³ + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  {x : ℝ | f' x = 4} = {1, -1} ∧
  f 1 = 0 ∧ f (-1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l999_99926


namespace NUMINAMATH_CALUDE_determinant_zero_l999_99977

theorem determinant_zero (α β : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![0, Real.cos α, -Real.sin α],
                                        ![-Real.cos α, 0, Real.cos β],
                                        ![Real.sin α, -Real.cos β, 0]]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_determinant_zero_l999_99977


namespace NUMINAMATH_CALUDE_fraction_simplification_l999_99906

theorem fraction_simplification (x : ℝ) : 
  (x + 2) / 4 + (3 - 4*x) / 5 + (7*x - 1) / 10 = (3*x + 20) / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l999_99906


namespace NUMINAMATH_CALUDE_max_vovochka_candies_l999_99953

/-- Represents the distribution of candies to classmates -/
def CandyDistribution := Fin 25 → ℕ

/-- The total number of candies -/
def totalCandies : ℕ := 200

/-- Checks if a candy distribution satisfies the condition that any 16 classmates have at least 100 candies -/
def isValidDistribution (d : CandyDistribution) : Prop :=
  ∀ (s : Finset (Fin 25)), s.card = 16 → (s.sum d) ≥ 100

/-- Calculates the number of candies Vovochka keeps for himself given a distribution -/
def vovochkaCandies (d : CandyDistribution) : ℕ :=
  totalCandies - (Finset.univ.sum d)

/-- Theorem stating that the maximum number of candies Vovochka can keep is 37 -/
theorem max_vovochka_candies :
  (∃ (d : CandyDistribution), isValidDistribution d ∧ vovochkaCandies d = 37) ∧
  (∀ (d : CandyDistribution), isValidDistribution d → vovochkaCandies d ≤ 37) :=
sorry

end NUMINAMATH_CALUDE_max_vovochka_candies_l999_99953


namespace NUMINAMATH_CALUDE_complex_expression_value_l999_99962

theorem complex_expression_value : 
  (10 - (10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)))) * 20 = 192.6 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l999_99962


namespace NUMINAMATH_CALUDE_water_height_after_sphere_removal_l999_99978

-- Define the container
structure Container where
  is_inverted_truncated_cone : Bool
  axial_section_is_equilateral_triangle : Bool

-- Define the sphere
structure Sphere where
  radius : ℝ

-- Define the configuration
structure Configuration where
  container : Container
  sphere : Sphere
  sphere_tangent_to_walls : Bool
  sphere_tangent_to_water_surface : Bool

-- Define the theorem
theorem water_height_after_sphere_removal 
  (config : Configuration) 
  (h : config.container.is_inverted_truncated_cone = true)
  (h' : config.container.axial_section_is_equilateral_triangle = true)
  (h'' : config.sphere_tangent_to_walls = true)
  (h''' : config.sphere_tangent_to_water_surface = true) :
  ∃ (water_height : ℝ), water_height = (15 ^ (1/3 : ℝ)) * config.sphere.radius :=
sorry

end NUMINAMATH_CALUDE_water_height_after_sphere_removal_l999_99978


namespace NUMINAMATH_CALUDE_tan_60_plus_inv_sqrt_3_l999_99911

theorem tan_60_plus_inv_sqrt_3 :
  Real.tan (π / 3) + (Real.sqrt 3)⁻¹ = (4 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_60_plus_inv_sqrt_3_l999_99911


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l999_99927

theorem cubic_sum_problem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) :
  x^3 + y^3 = 65 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l999_99927


namespace NUMINAMATH_CALUDE_volume_of_specific_open_box_l999_99916

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutSize : ℝ) : ℝ :=
  (sheetLength - 2 * cutSize) * (sheetWidth - 2 * cutSize) * cutSize

/-- Theorem stating that the volume of the specific open box is 5120 cubic meters. -/
theorem volume_of_specific_open_box :
  openBoxVolume 48 36 8 = 5120 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_open_box_l999_99916


namespace NUMINAMATH_CALUDE_orange_distribution_ratio_l999_99928

/-- Proves the ratio of oranges given to the brother to the total number of oranges --/
theorem orange_distribution_ratio :
  let total_oranges : ℕ := 12
  let friend_oranges : ℕ := 2
  ∀ brother_fraction : ℚ,
    (1 / 4 : ℚ) * ((1 : ℚ) - brother_fraction) * total_oranges = friend_oranges →
    (brother_fraction * total_oranges : ℚ) / total_oranges = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_ratio_l999_99928


namespace NUMINAMATH_CALUDE_linear_inequality_solution_set_l999_99999

theorem linear_inequality_solution_set 
  (m n : ℝ) 
  (h1 : m = -1) 
  (h2 : n = -1) : 
  {x : ℝ | m * x - n ≤ 2} = {x : ℝ | x ≥ -1} := by
sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_set_l999_99999


namespace NUMINAMATH_CALUDE_hexagon_area_is_19444_l999_99907

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (positive_a : a > 0)
  (positive_b : b > 0)
  (positive_c : c > 0)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

-- Define the specific triangle with sides 13, 14, and 15
def specific_triangle : Triangle :=
  { a := 13
  , b := 14
  , c := 15
  , positive_a := by norm_num
  , positive_b := by norm_num
  , positive_c := by norm_num
  , triangle_inequality_ab := by norm_num
  , triangle_inequality_bc := by norm_num
  , triangle_inequality_ca := by norm_num }

-- Define the area of the hexagon A₅A₆B₅B₆C₅C₆
def hexagon_area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem hexagon_area_is_19444 :
  hexagon_area specific_triangle = 19444 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_is_19444_l999_99907


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l999_99945

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m - 2) * x^(|m|) + x - 1 = a * x^2 + b * x + c) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l999_99945


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l999_99993

-- Define the pyramid
structure RectangularPyramid where
  baseSideLength : ℝ
  sideEdgeLength : ℝ

-- Define the properties of the pyramid
def isPyramidOnSphere (p : RectangularPyramid) : Prop :=
  (p.baseSideLength ^ 2 + p.baseSideLength ^ 2 + p.sideEdgeLength ^ 2) / 4 = 1

def hasSquareBase (p : RectangularPyramid) : Prop :=
  p.baseSideLength ^ 2 + p.baseSideLength ^ 2 = 2

def sideEdgesPerpendicular (p : RectangularPyramid) : Prop :=
  p.sideEdgeLength ^ 2 + p.baseSideLength ^ 2 / 2 = 1

-- Define the surface area calculation
def surfaceArea (p : RectangularPyramid) : ℝ :=
  p.baseSideLength ^ 2 + 4 * p.baseSideLength * p.sideEdgeLength

-- Theorem statement
theorem pyramid_surface_area (p : RectangularPyramid) :
  isPyramidOnSphere p → hasSquareBase p → sideEdgesPerpendicular p →
  surfaceArea p = 2 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_l999_99993


namespace NUMINAMATH_CALUDE_assignment_satisfies_conditions_l999_99967

-- Define the set of people
inductive Person : Type
| Arthur : Person
| Burton : Person
| Congreve : Person
| Downs : Person
| Ewald : Person
| Flynn : Person

-- Define the set of positions
inductive Position : Type
| President : Position
| VicePresident : Position
| Secretary : Position
| Treasurer : Position

-- Define the assignment function
def assignment : Position → Person
| Position.President => Person.Flynn
| Position.VicePresident => Person.Ewald
| Position.Secretary => Person.Congreve
| Position.Treasurer => Person.Burton

-- Define the conditions
def arthur_condition (a : Position → Person) : Prop :=
  (a Position.VicePresident ≠ Person.Arthur) ∨ (a Position.President ≠ Person.Burton ∧ a Position.VicePresident ≠ Person.Burton ∧ a Position.Secretary ≠ Person.Burton ∧ a Position.Treasurer ≠ Person.Burton)

def burton_condition (a : Position → Person) : Prop :=
  a Position.VicePresident ≠ Person.Burton ∧ a Position.Secretary ≠ Person.Burton

def congreve_condition (a : Position → Person) : Prop :=
  (a Position.President ≠ Person.Burton ∧ a Position.VicePresident ≠ Person.Burton ∧ a Position.Secretary ≠ Person.Burton ∧ a Position.Treasurer ≠ Person.Burton) ∨
  (a Position.President = Person.Flynn ∨ a Position.VicePresident = Person.Flynn ∨ a Position.Secretary = Person.Flynn ∨ a Position.Treasurer = Person.Flynn)

def downs_condition (a : Position → Person) : Prop :=
  (a Position.President ≠ Person.Ewald ∧ a Position.VicePresident ≠ Person.Ewald ∧ a Position.Secretary ≠ Person.Ewald ∧ a Position.Treasurer ≠ Person.Ewald) ∧
  (a Position.President ≠ Person.Flynn ∧ a Position.VicePresident ≠ Person.Flynn ∧ a Position.Secretary ≠ Person.Flynn ∧ a Position.Treasurer ≠ Person.Flynn)

def ewald_condition (a : Position → Person) : Prop :=
  ¬(a Position.President = Person.Arthur ∧ (a Position.VicePresident = Person.Burton ∨ a Position.Secretary = Person.Burton ∨ a Position.Treasurer = Person.Burton)) ∧
  ¬(a Position.VicePresident = Person.Arthur ∧ (a Position.President = Person.Burton ∨ a Position.Secretary = Person.Burton ∨ a Position.Treasurer = Person.Burton)) ∧
  ¬(a Position.Secretary = Person.Arthur ∧ (a Position.President = Person.Burton ∨ a Position.VicePresident = Person.Burton ∨ a Position.Treasurer = Person.Burton)) ∧
  ¬(a Position.Treasurer = Person.Arthur ∧ (a Position.President = Person.Burton ∨ a Position.VicePresident = Person.Burton ∨ a Position.Secretary = Person.Burton))

def flynn_condition (a : Position → Person) : Prop :=
  (a Position.President = Person.Flynn) → (a Position.VicePresident ≠ Person.Congreve)

-- Theorem statement
theorem assignment_satisfies_conditions :
  arthur_condition assignment ∧
  burton_condition assignment ∧
  congreve_condition assignment ∧
  downs_condition assignment ∧
  ewald_condition assignment ∧
  flynn_condition assignment :=
sorry

end NUMINAMATH_CALUDE_assignment_satisfies_conditions_l999_99967


namespace NUMINAMATH_CALUDE_scavenger_hunt_items_l999_99934

theorem scavenger_hunt_items (lewis samantha tanya : ℕ) : 
  lewis = samantha + 4 →
  samantha = 4 * tanya →
  lewis = 20 →
  tanya = 4 := by sorry

end NUMINAMATH_CALUDE_scavenger_hunt_items_l999_99934


namespace NUMINAMATH_CALUDE_possible_values_of_a_l999_99974

-- Define the sets A and B
def A : Set ℝ := {0, 1}
def B (a : ℝ) : Set ℝ := {x | a * x^2 + x - 1 = 0}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : A ⊇ B a → a = 0 ∨ a < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l999_99974


namespace NUMINAMATH_CALUDE_cos_135_degrees_l999_99939

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l999_99939


namespace NUMINAMATH_CALUDE_johns_work_hours_l999_99922

/-- Calculates the number of hours John works every other day given his wage, raise percentage, total earnings, and days in a month. -/
theorem johns_work_hours 
  (former_wage : ℝ)
  (raise_percentage : ℝ)
  (total_earnings : ℝ)
  (days_in_month : ℕ)
  (h1 : former_wage = 20)
  (h2 : raise_percentage = 30)
  (h3 : total_earnings = 4680)
  (h4 : days_in_month = 30) :
  let new_wage := former_wage * (1 + raise_percentage / 100)
  let working_days := days_in_month / 2
  let total_hours := total_earnings / new_wage
  let hours_per_working_day := total_hours / working_days
  hours_per_working_day = 12 := by sorry

end NUMINAMATH_CALUDE_johns_work_hours_l999_99922


namespace NUMINAMATH_CALUDE_percentage_difference_l999_99969

theorem percentage_difference : (70 : ℝ) / 100 * 100 - (60 : ℝ) / 100 * 80 = 22 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l999_99969


namespace NUMINAMATH_CALUDE_denmark_pizza_combinations_l999_99990

/-- Represents the number of topping combinations for Denmark's pizza order --/
def toppingCombinations (cheeseOptions : Nat) (meatOptions : Nat) (vegetableOptions : Nat) : Nat :=
  let totalCombinations := cheeseOptions * meatOptions * vegetableOptions
  let restrictedCombinations := cheeseOptions * 1 * 1
  totalCombinations - restrictedCombinations

/-- Theorem: Denmark has 57 different topping combinations for his pizza --/
theorem denmark_pizza_combinations :
  toppingCombinations 3 4 5 = 57 := by
  sorry

#eval toppingCombinations 3 4 5

end NUMINAMATH_CALUDE_denmark_pizza_combinations_l999_99990


namespace NUMINAMATH_CALUDE_max_profit_theorem_l999_99932

def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

def profit_B (x : ℝ) : ℝ := 2 * x

def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (15 - x)

theorem max_profit_theorem :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 15 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 15 → total_profit x ≥ total_profit y ∧
  total_profit x = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l999_99932


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l999_99961

theorem cubic_equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  (∃ a b : ℝ, ∀ x : ℝ, x = a * m + b * n → (x + m)^3 - (x + n)^3 = (m - n)^3) ↔
  (∀ x : ℝ, (x + m)^3 - (x + n)^3 = (m - n)^3 ↔ x = -m + n) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l999_99961


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l999_99935

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 7) + 5 = 14 → x = 74 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l999_99935


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l999_99942

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = !![3, 4; -2, -2]) : 
  (A^2)⁻¹ = !![1, 4; -2, -4] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l999_99942


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l999_99982

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Ensure positive side lengths
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a^2 + b^2 + c^2 = 980 →  -- Given condition
  c = 70 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l999_99982


namespace NUMINAMATH_CALUDE_elaine_rent_percentage_l999_99971

/-- Represents Elaine's financial situation over two years -/
structure ElaineFinances where
  last_year_earnings : ℝ
  last_year_rent_percentage : ℝ
  this_year_earnings_increase : ℝ
  this_year_rent_percentage : ℝ
  rent_increase_percentage : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem elaine_rent_percentage
  (e : ElaineFinances)
  (h1 : e.this_year_earnings_increase = 0.15)
  (h2 : e.this_year_rent_percentage = 0.30)
  (h3 : e.rent_increase_percentage = 3.45)
  : e.last_year_rent_percentage = 0.10 := by
  sorry

#check elaine_rent_percentage

end NUMINAMATH_CALUDE_elaine_rent_percentage_l999_99971


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l999_99975

theorem simplify_fraction_product : 8 * (15 / 14) * (-49 / 45) = -28 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l999_99975


namespace NUMINAMATH_CALUDE_exists_real_for_special_sequence_l999_99972

/-- A sequence of non-negative integers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, n ≤ 1999 → a n ≥ 0) ∧
  (∀ i j, i + j ≤ 1999 → a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1)

/-- The main theorem -/
theorem exists_real_for_special_sequence (a : ℕ → ℕ) (h : SpecialSequence a) :
  ∃ x : ℝ, ∀ n : ℕ, n ≤ 1999 → a n = ⌊n * x⌋ := by
  sorry

end NUMINAMATH_CALUDE_exists_real_for_special_sequence_l999_99972


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l999_99908

theorem fourth_root_simplification :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 
  (2^8 * 3^5)^(1/4 : ℝ) = a * (b : ℝ)^(1/4 : ℝ) ∧
  a + b = 15 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l999_99908


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l999_99938

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l999_99938


namespace NUMINAMATH_CALUDE_three_statements_true_l999_99949

open Function

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def isPeriodic (f : ℝ → ℝ) : Prop := ∃ T ≠ 0, ∀ x, f (x + T) = f x
def isMonoDecreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y
def hasInverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, LeftInverse g f ∧ RightInverse g f

-- The main theorem
theorem three_statements_true (f : ℝ → ℝ) : 
  (isOdd f → isOdd (f ∘ f)) ∧
  (isPeriodic f → isPeriodic (f ∘ f)) ∧
  ¬(isMonoDecreasing f → isMonoDecreasing (f ∘ f)) ∧
  (hasInverse f → (∃ x, f x = x)) :=
sorry

end NUMINAMATH_CALUDE_three_statements_true_l999_99949


namespace NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l999_99987

theorem fgh_supermarkets_in_us (total : ℕ) (difference : ℕ) 
  (h_total : total = 84)
  (h_difference : difference = 14) :
  let us := (total + difference) / 2
  us = 49 := by
sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l999_99987


namespace NUMINAMATH_CALUDE_circles_intersect_l999_99904

-- Define Circle C1
def C1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 1

-- Define Circle C2
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 1 = 0

-- Theorem stating that C1 and C2 intersect
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l999_99904


namespace NUMINAMATH_CALUDE_loot_box_average_loss_l999_99941

/-- Represents the expected value calculation for a loot box system -/
def loot_box_expected_value (standard_value : ℝ) (rare_a_prob : ℝ) (rare_a_value : ℝ)
  (rare_b_prob : ℝ) (rare_b_value : ℝ) (rare_c_prob : ℝ) (rare_c_value : ℝ) : ℝ :=
  let standard_prob := 1 - (rare_a_prob + rare_b_prob + rare_c_prob)
  standard_prob * standard_value + rare_a_prob * rare_a_value +
  rare_b_prob * rare_b_value + rare_c_prob * rare_c_value

/-- Calculates the average loss per loot box -/
def average_loss_per_loot_box (box_cost : ℝ) (expected_value : ℝ) : ℝ :=
  box_cost - expected_value

/-- Theorem stating the average loss per loot box in the given scenario -/
theorem loot_box_average_loss :
  let box_cost : ℝ := 5
  let standard_value : ℝ := 3.5
  let rare_a_prob : ℝ := 0.05
  let rare_a_value : ℝ := 10
  let rare_b_prob : ℝ := 0.03
  let rare_b_value : ℝ := 15
  let rare_c_prob : ℝ := 0.02
  let rare_c_value : ℝ := 20
  let expected_value := loot_box_expected_value standard_value rare_a_prob rare_a_value
    rare_b_prob rare_b_value rare_c_prob rare_c_value
  average_loss_per_loot_box box_cost expected_value = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_loot_box_average_loss_l999_99941


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l999_99902

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l999_99902


namespace NUMINAMATH_CALUDE_second_quadrant_characterization_l999_99920

/-- The set of points in the second quadrant of the Cartesian coordinate system -/
def second_quadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

/-- Theorem stating that the second quadrant is equivalent to the set of points (x, y) where x < 0 and y > 0 -/
theorem second_quadrant_characterization :
  second_quadrant = {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0} := by
sorry

end NUMINAMATH_CALUDE_second_quadrant_characterization_l999_99920


namespace NUMINAMATH_CALUDE_f_is_quadratic_l999_99940

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 2x + 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l999_99940


namespace NUMINAMATH_CALUDE_price_per_pack_is_one_l999_99997

/-- Represents the number of boxes in a carton -/
def boxes_per_carton : ℕ := 12

/-- Represents the number of packs of cheese cookies in a box -/
def packs_per_box : ℕ := 10

/-- Represents the cost of a dozen cartons in dollars -/
def cost_dozen_cartons : ℕ := 1440

/-- Represents the number of cartons in a dozen -/
def cartons_in_dozen : ℕ := 12

/-- Theorem stating that the price of a pack of cheese cookies is $1 -/
theorem price_per_pack_is_one :
  (cost_dozen_cartons : ℚ) / (cartons_in_dozen * boxes_per_carton * packs_per_box) = 1 := by
  sorry

end NUMINAMATH_CALUDE_price_per_pack_is_one_l999_99997


namespace NUMINAMATH_CALUDE_inverse_proportionality_example_l999_99931

/-- Definition of inverse proportionality -/
def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function y = 6/x is inversely proportional -/
theorem inverse_proportionality_example :
  is_inversely_proportional (λ x : ℝ => 6 / x) :=
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportionality_example_l999_99931


namespace NUMINAMATH_CALUDE_like_terms_characterization_l999_99992

/-- Represents a term in an algebraic expression -/
structure Term where
  letters : List Char
  exponents : List Nat
  deriving Repr

/-- Defines when two terms are considered like terms -/
def like_terms (t1 t2 : Term) : Prop :=
  t1.letters = t2.letters ∧ t1.exponents = t2.exponents

theorem like_terms_characterization (t1 t2 : Term) :
  like_terms t1 t2 ↔ t1.letters = t2.letters ∧ t1.exponents = t2.exponents :=
by sorry

end NUMINAMATH_CALUDE_like_terms_characterization_l999_99992


namespace NUMINAMATH_CALUDE_value_of_expression_l999_99948

theorem value_of_expression (x : ℝ) (h : 6 * x^2 - 4 * x - 3 = 0) :
  (x - 1)^2 + x * (x + 2/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l999_99948


namespace NUMINAMATH_CALUDE_skating_minutes_on_eleventh_day_l999_99929

def minutes_per_day_first_period : ℕ := 80
def days_first_period : ℕ := 6
def minutes_per_day_second_period : ℕ := 105
def days_second_period : ℕ := 4
def target_average : ℕ := 95
def total_days : ℕ := 11

theorem skating_minutes_on_eleventh_day :
  (minutes_per_day_first_period * days_first_period +
   minutes_per_day_second_period * days_second_period +
   145) / total_days = target_average :=
by sorry

end NUMINAMATH_CALUDE_skating_minutes_on_eleventh_day_l999_99929


namespace NUMINAMATH_CALUDE_cookies_per_child_l999_99960

theorem cookies_per_child (total_cookies : ℕ) (num_adults num_children : ℕ) (adult_fraction : ℚ) : 
  total_cookies = 240 →
  num_adults = 4 →
  num_children = 6 →
  adult_fraction = 1/4 →
  (total_cookies - (adult_fraction * total_cookies).num) / num_children = 30 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_child_l999_99960


namespace NUMINAMATH_CALUDE_problem_solution_l999_99900

theorem problem_solution (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  w = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l999_99900


namespace NUMINAMATH_CALUDE_smartphone_customers_l999_99933

/-- Represents the relationship between number of customers and smartphone price -/
def inversely_proportional (p c : ℝ) := ∃ k : ℝ, p * c = k

theorem smartphone_customers : 
  ∀ (p₁ p₂ c₁ c₂ : ℝ),
  inversely_proportional p₁ c₁ →
  inversely_proportional p₂ c₂ →
  p₁ = 20 →
  c₁ = 200 →
  c₂ = 400 →
  p₂ = 10 :=
by sorry

end NUMINAMATH_CALUDE_smartphone_customers_l999_99933


namespace NUMINAMATH_CALUDE_line_l_passes_through_M_line_l1_properties_l999_99983

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  (2 + m) * x + (1 - 2 * m) * y + 4 - 3 * m = 0

-- Define the point M
def point_M : ℝ × ℝ := (-1, -2)

-- Define the line l1
def line_l1 (x y : ℝ) : Prop :=
  2 * x + y + 4 = 0

-- Theorem 1: Line l passes through point M for all real m
theorem line_l_passes_through_M :
  ∀ m : ℝ, line_l m (point_M.1) (point_M.2) := by sorry

-- Theorem 2: Line l1 passes through point M and is bisected by M
theorem line_l1_properties :
  line_l1 (point_M.1) (point_M.2) ∧
  ∃ (A B : ℝ × ℝ),
    (A.1 = 0 ∨ A.2 = 0) ∧
    (B.1 = 0 ∨ B.2 = 0) ∧
    line_l1 A.1 A.2 ∧
    line_l1 B.1 B.2 ∧
    ((A.1 + B.1) / 2 = point_M.1 ∧ (A.2 + B.2) / 2 = point_M.2) := by sorry

end NUMINAMATH_CALUDE_line_l_passes_through_M_line_l1_properties_l999_99983


namespace NUMINAMATH_CALUDE_sequence_sum_bounded_l999_99955

theorem sequence_sum_bounded (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_a1 : 0 ≤ a 1)
  (h_a : ∀ i ∈ Finset.range (n - 1), a i ≤ a (i + 1) ∧ a (i + 1) ≤ 2 * a i) :
  ∃ ε : ℕ → ℝ, (∀ i ∈ Finset.range n, ε i = 1 ∨ ε i = -1) ∧ 
    0 ≤ (Finset.range n).sum (λ i => ε i * a (i + 1)) ∧
    (Finset.range n).sum (λ i => ε i * a (i + 1)) ≤ a 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_bounded_l999_99955


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l999_99963

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define variables a and b
def a : ℚ := sorry
def b : ℚ := sorry

-- State the theorem
theorem sum_of_a_and_b : 
  (0.5 / 100 * a = paise_to_rupees 65) → 
  (1.25 / 100 * b = paise_to_rupees 104) → 
  (a + b = 213.2) := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l999_99963


namespace NUMINAMATH_CALUDE_binary_to_decimal_conversion_l999_99994

/-- Converts a binary digit to its decimal value -/
def binaryToDecimal (digit : ℕ) (position : ℤ) : ℚ :=
  (digit : ℚ) * (2 : ℚ) ^ position

/-- Represents the binary number 111.11 -/
def binaryNumber : List (ℕ × ℤ) :=
  [(1, 2), (1, 1), (1, 0), (1, -1), (1, -2)]

/-- Theorem: The binary number 111.11 is equal to 7.75 in decimal -/
theorem binary_to_decimal_conversion :
  (binaryNumber.map (fun (digit, position) => binaryToDecimal digit position)).sum = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_conversion_l999_99994


namespace NUMINAMATH_CALUDE_ellipse_tangent_quadrilateral_area_l999_99946

/-- Given an ellipse with equation 9x^2 + 25y^2 = 225, 
    the area of the quadrilateral formed by tangents at the parameter endpoints is 62.5 -/
theorem ellipse_tangent_quadrilateral_area :
  let a : ℝ := 5  -- semi-major axis
  let b : ℝ := 3  -- semi-minor axis
  ∀ x y : ℝ, 9 * x^2 + 25 * y^2 = 225 →
  let area := 2 * a^3 / Real.sqrt (a^2 - b^2)
  area = 62.5 := by
sorry


end NUMINAMATH_CALUDE_ellipse_tangent_quadrilateral_area_l999_99946


namespace NUMINAMATH_CALUDE_estimate_smaller_than_exact_l999_99917

theorem estimate_smaller_than_exact (a b c d a' b' c' d' : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (ha' : 0 < a' ∧ a' ≤ a) (hb' : 0 < b' ∧ b ≤ b')
  (hc' : 0 < c' ∧ c' ≤ c) (hd' : 0 < d' ∧ d ≤ d') :
  d' * (a' / b') + c' < d * (a / b) + c := by
  sorry

end NUMINAMATH_CALUDE_estimate_smaller_than_exact_l999_99917


namespace NUMINAMATH_CALUDE_profit_share_difference_example_l999_99950

/-- Calculates the difference between profit shares of two partners given investments and one partner's profit share. -/
def profit_share_difference (inv_a inv_b inv_c b_profit : ℚ) : ℚ :=
  let total_inv := inv_a + inv_b + inv_c
  let total_profit := (total_inv / inv_b) * b_profit
  let a_share := (inv_a / total_inv) * total_profit
  let c_share := (inv_c / total_inv) * total_profit
  c_share - a_share

/-- Proves that the difference between profit shares of a and c is 600 given the specified investments and b's profit share. -/
theorem profit_share_difference_example : 
  profit_share_difference 8000 10000 12000 1500 = 600 := by
  sorry


end NUMINAMATH_CALUDE_profit_share_difference_example_l999_99950


namespace NUMINAMATH_CALUDE_collinear_dots_probability_l999_99976

/-- Represents a 5x5 grid of dots -/
def Grid := Fin 5 × Fin 5

/-- The total number of dots in the grid -/
def total_dots : Nat := 25

/-- The number of ways to choose 4 dots from the total dots -/
def total_choices : Nat := Nat.choose total_dots 4

/-- The number of sets of 4 collinear dots in the grid -/
def collinear_sets : Nat := 28

/-- The probability of choosing 4 collinear dots -/
def collinear_probability : Rat := collinear_sets / total_choices

theorem collinear_dots_probability :
  collinear_probability = 4 / 1807 := by sorry

end NUMINAMATH_CALUDE_collinear_dots_probability_l999_99976


namespace NUMINAMATH_CALUDE_equation_solution_l999_99914

theorem equation_solution (x : ℚ) : 
  (3 : ℚ) / 4 + 1 / x = (7 : ℚ) / 8 → x = 8 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l999_99914


namespace NUMINAMATH_CALUDE_freddys_age_l999_99985

theorem freddys_age (job_age stephanie_age freddy_age : ℕ) : 
  job_age = 5 →
  stephanie_age = 4 * job_age →
  freddy_age = stephanie_age - 2 →
  freddy_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_freddys_age_l999_99985


namespace NUMINAMATH_CALUDE_square_kilometer_conversion_time_conversion_l999_99921

-- Define the conversion rates
def sq_km_to_hectares : ℝ := 100
def hour_to_minutes : ℝ := 60

-- Define the problem statements
def problem1 (sq_km : ℝ) (whole_sq_km : ℕ) (hectares : ℕ) : Prop :=
  sq_km = whole_sq_km + hectares / sq_km_to_hectares

def problem2 (hours : ℝ) (whole_hours : ℕ) (minutes : ℕ) : Prop :=
  hours = whole_hours + minutes / hour_to_minutes

-- Theorem statements
theorem square_kilometer_conversion :
  problem1 7.05 7 500 := by sorry

theorem time_conversion :
  problem2 6.7 6 42 := by sorry

end NUMINAMATH_CALUDE_square_kilometer_conversion_time_conversion_l999_99921


namespace NUMINAMATH_CALUDE_greatest_value_is_product_of_zeros_l999_99910

def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - x^2 - 4*x + 4

theorem greatest_value_is_product_of_zeros :
  let product_of_zeros : ℝ := 4
  let q_of_one : ℝ := Q 1
  let sum_of_coefficients : ℝ := 1 + 2 - 1 - 4 + 4
  let sum_of_real_zeros : ℝ := 0  -- Assumption based on estimated real zeros
  product_of_zeros > q_of_one ∧
  product_of_zeros > sum_of_coefficients ∧
  product_of_zeros > sum_of_real_zeros :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_is_product_of_zeros_l999_99910


namespace NUMINAMATH_CALUDE_softball_team_size_l999_99952

theorem softball_team_size (men women : ℕ) : 
  women = men + 6 →
  (men : ℝ) / (women : ℝ) = 0.45454545454545453 →
  men + women = 16 := by
sorry

end NUMINAMATH_CALUDE_softball_team_size_l999_99952


namespace NUMINAMATH_CALUDE_square_roots_and_cube_root_problem_l999_99996

theorem square_roots_and_cube_root_problem (x y a : ℝ) :
  x > 0 ∧
  (a + 3)^2 = x ∧
  (2*a - 15)^2 = x ∧
  (x + y - 2)^(1/3) = 4 →
  x - 2*y + 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_and_cube_root_problem_l999_99996


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l999_99991

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l999_99991


namespace NUMINAMATH_CALUDE_cake_mass_proof_l999_99998

/-- The original mass of the cake in grams -/
def original_mass : ℝ := 750

/-- The mass of cake eaten by Carlson as a fraction -/
def carlson_fraction : ℝ := 0.4

/-- The mass of cake eaten by Little Man in grams -/
def little_man_mass : ℝ := 150

/-- The fraction of remaining cake eaten by Freken Bok -/
def freken_bok_fraction : ℝ := 0.3

/-- The additional mass of cake eaten by Freken Bok in grams -/
def freken_bok_additional : ℝ := 120

/-- The mass of cake crumbs eaten by Matilda in grams -/
def matilda_crumbs : ℝ := 90

theorem cake_mass_proof :
  let remaining_after_carlson := original_mass * (1 - carlson_fraction)
  let remaining_after_little_man := remaining_after_carlson - little_man_mass
  let remaining_after_freken_bok := remaining_after_little_man * (1 - freken_bok_fraction) - freken_bok_additional
  remaining_after_freken_bok = matilda_crumbs :=
by sorry

end NUMINAMATH_CALUDE_cake_mass_proof_l999_99998


namespace NUMINAMATH_CALUDE_quadrilateral_properties_l999_99995

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral where
  O : Point
  P : Point
  Q : Point
  R : Point

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (quad : Quadrilateral) : Prop :=
  (quad.R.x - quad.P.x = quad.Q.x - quad.O.x) ∧
  (quad.R.y - quad.P.y = quad.Q.y - quad.O.y)

/-- Checks if a quadrilateral is a rhombus -/
def isRhombus (quad : Quadrilateral) : Prop :=
  let OP := (quad.P.x - quad.O.x)^2 + (quad.P.y - quad.O.y)^2
  let OQ := (quad.Q.x - quad.O.x)^2 + (quad.Q.y - quad.O.y)^2
  let OR := (quad.R.x - quad.O.x)^2 + (quad.R.y - quad.O.y)^2
  let PQ := (quad.Q.x - quad.P.x)^2 + (quad.Q.y - quad.P.y)^2
  OP = OQ ∧ OQ = OR ∧ OR = PQ

theorem quadrilateral_properties (x₁ y₁ x₂ y₂ : ℝ) :
  let quad := Quadrilateral.mk
    (Point.mk 0 0)
    (Point.mk x₁ y₁)
    (Point.mk x₂ y₂)
    (Point.mk (2*x₁ - x₂) (2*y₁ - y₂))
  isParallelogram quad ∧ (∃ (x₁ y₁ x₂ y₂ : ℝ), isRhombus quad) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_l999_99995


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l999_99936

/-- Proves that the initial number of bananas per child is 2 --/
theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) : 
  total_children = 320 →
  absent_children = 160 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ), 
    (total_children - absent_children) * (initial_bananas + extra_bananas) = 
    total_children * initial_bananas ∧
    initial_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l999_99936


namespace NUMINAMATH_CALUDE_investment_difference_proof_l999_99988

/-- Represents an investment scheme with an initial investment and a yield rate -/
structure Scheme where
  investment : ℝ
  yieldRate : ℝ

/-- Calculates the total amount in a scheme after a year -/
def totalAfterYear (s : Scheme) : ℝ :=
  s.investment + s.investment * s.yieldRate

/-- The difference in total amounts between two schemes after a year -/
def schemeDifference (s1 s2 : Scheme) : ℝ :=
  totalAfterYear s1 - totalAfterYear s2

theorem investment_difference_proof (schemeA schemeB : Scheme) 
  (h1 : schemeA.investment = 300)
  (h2 : schemeB.investment = 200)
  (h3 : schemeA.yieldRate = 0.3)
  (h4 : schemeB.yieldRate = 0.5) :
  schemeDifference schemeA schemeB = 90 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_proof_l999_99988


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l999_99989

/-- A hyperbola is defined by its equation in the form ax² + by² = c, where a, b, and c are constants and a and b have opposite signs. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  opposite_signs : a * b < 0

/-- Two hyperbolas share the same asymptotes if they have the same ratio of coefficients for x² and y². -/
def share_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a / h1.b = h2.a / h2.b

/-- A point (x, y) is on a hyperbola if it satisfies the hyperbola's equation. -/
def point_on_hyperbola (h : Hyperbola) (x y : ℝ) : Prop :=
  h.a * x^2 + h.b * y^2 = h.c

/-- The main theorem to be proved -/
theorem hyperbola_theorem (h1 h2 : Hyperbola) :
  h1.a = 1/4 ∧ h1.b = -1 ∧ h1.c = 1 ∧
  h2.a = -1/16 ∧ h2.b = 1/4 ∧ h2.c = 1 →
  share_asymptotes h1 h2 ∧ point_on_hyperbola h2 2 (Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l999_99989


namespace NUMINAMATH_CALUDE_five_fourths_of_eight_thirds_l999_99966

theorem five_fourths_of_eight_thirds (x : ℚ) : x = 8/3 → (5/4) * x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_eight_thirds_l999_99966


namespace NUMINAMATH_CALUDE_isabelle_bubble_bath_amount_l999_99970

/-- Represents the configuration of a hotel --/
structure HotelConfig where
  double_suites : Nat
  couple_rooms : Nat
  single_rooms : Nat
  family_rooms : Nat
  double_suite_capacity : Nat
  couple_room_capacity : Nat
  single_room_capacity : Nat
  family_room_capacity : Nat
  bubble_bath_per_guest : Nat

/-- Calculates the total bubble bath needed for a given hotel configuration --/
def total_bubble_bath (config : HotelConfig) : Nat :=
  (config.double_suites * config.double_suite_capacity +
   config.couple_rooms * config.couple_room_capacity +
   config.single_rooms * config.single_room_capacity +
   config.family_rooms * config.family_room_capacity) *
  config.bubble_bath_per_guest

/-- The specific hotel configuration from the problem --/
def isabelle_hotel : HotelConfig :=
  { double_suites := 5
  , couple_rooms := 13
  , single_rooms := 14
  , family_rooms := 3
  , double_suite_capacity := 4
  , couple_room_capacity := 2
  , single_room_capacity := 1
  , family_room_capacity := 6
  , bubble_bath_per_guest := 25
  }

/-- Theorem stating that the total bubble bath needed for Isabelle's hotel is 1950 ml --/
theorem isabelle_bubble_bath_amount :
  total_bubble_bath isabelle_hotel = 1950 := by
  sorry

end NUMINAMATH_CALUDE_isabelle_bubble_bath_amount_l999_99970


namespace NUMINAMATH_CALUDE_correct_recommendation_plans_l999_99925

/-- Represents the number of recommendation spots for each language -/
structure RecommendationSpots where
  russian : Nat
  japanese : Nat
  spanish : Nat

/-- Represents the number of male and female candidates -/
structure Candidates where
  male : Nat
  female : Nat

/-- Calculates the number of different recommendation plans -/
def recommendationPlans (spots : RecommendationSpots) (candidates : Candidates) : Nat :=
  sorry

/-- Theorem stating the number of different recommendation plans -/
theorem correct_recommendation_plans :
  let spots := RecommendationSpots.mk 2 2 1
  let candidates := Candidates.mk 3 2
  recommendationPlans spots candidates = 24 := by sorry

end NUMINAMATH_CALUDE_correct_recommendation_plans_l999_99925


namespace NUMINAMATH_CALUDE_different_color_probability_l999_99937

def total_chips : ℕ := 7 + 5
def red_chips : ℕ := 7
def green_chips : ℕ := 5

theorem different_color_probability :
  (red_chips * green_chips : ℚ) / (total_chips * (total_chips - 1) / 2) = 35 / 66 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l999_99937


namespace NUMINAMATH_CALUDE_farm_feet_count_l999_99986

/-- Given a farm with hens and cows, calculates the total number of feet -/
def total_feet (total_heads : ℕ) (num_hens : ℕ) : ℕ :=
  let num_cows := total_heads - num_hens
  let hen_feet := num_hens * 2
  let cow_feet := num_cows * 4
  hen_feet + cow_feet

/-- Theorem: In a farm with 46 total heads and 24 hens, there are 136 feet in total -/
theorem farm_feet_count : total_feet 46 24 = 136 := by
  sorry

end NUMINAMATH_CALUDE_farm_feet_count_l999_99986


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l999_99965

/-- Two lines y = ax - 2 and y = (a+2)x + 1 are perpendicular -/
def are_perpendicular (a : ℝ) : Prop :=
  a * (a + 2) + 1 = 0

/-- Theorem: If the lines y = ax - 2 and y = (a+2)x + 1 are perpendicular, then a = -1 -/
theorem perpendicular_lines_a_value :
  ∀ a : ℝ, are_perpendicular a → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l999_99965


namespace NUMINAMATH_CALUDE_sandys_age_l999_99951

theorem sandys_age (sandy_age molly_age : ℕ) 
  (h1 : molly_age = sandy_age + 18) 
  (h2 : sandy_age * 9 = molly_age * 7) : 
  sandy_age = 63 := by
  sorry

end NUMINAMATH_CALUDE_sandys_age_l999_99951


namespace NUMINAMATH_CALUDE_sandy_earnings_l999_99956

/-- Calculates the total earnings for Sandy given her hourly rate and hours worked each day -/
def total_earnings (hourly_rate : ℕ) (hours_friday : ℕ) (hours_saturday : ℕ) (hours_sunday : ℕ) : ℕ :=
  hourly_rate * (hours_friday + hours_saturday + hours_sunday)

/-- Theorem stating that Sandy's total earnings for the three days is $450 -/
theorem sandy_earnings : 
  total_earnings 15 10 6 14 = 450 := by
  sorry

end NUMINAMATH_CALUDE_sandy_earnings_l999_99956


namespace NUMINAMATH_CALUDE_f_properties_l999_99923

noncomputable section

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := Real.exp (t * (x - 1)) - t * Real.log x

-- State the theorem
theorem f_properties (t : ℝ) (h_t : t > 0) :
  -- Part I: When t = 1, x = 1 is a local minimum point of f
  (t = 1 → ∃ ε > 0, ∀ x, x > 0 → |x - 1| < ε → f 1 x ≥ f 1 1) ∧
  -- Part II: For all x > 0 and t > 0, f(x) ≥ 0
  (∀ x > 0, f t x ≥ 0) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l999_99923


namespace NUMINAMATH_CALUDE_quadratic_point_order_l999_99984

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 2*x + 2

-- Define the points A, B, and C
def A : ℝ × ℝ := (-2, f (-2))
def B : ℝ × ℝ := (-1, f (-1))
def C : ℝ × ℝ := (2, f 2)

-- State the theorem
theorem quadratic_point_order :
  A.2 < B.2 ∧ B.2 < C.2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_point_order_l999_99984


namespace NUMINAMATH_CALUDE_line_intersection_y_coordinate_l999_99958

/-- Given a line with slope 3/4 passing through (400, 0), 
    prove that the y-coordinate at x = -12 is -309 -/
theorem line_intersection_y_coordinate 
  (slope : ℚ) 
  (x_intercept : ℝ) 
  (x_coord : ℝ) :
  slope = 3/4 →
  x_intercept = 400 →
  x_coord = -12 →
  let y_intercept := -(slope * x_intercept)
  let y_coord := slope * x_coord + y_intercept
  y_coord = -309 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_y_coordinate_l999_99958


namespace NUMINAMATH_CALUDE_sphere_volume_reduction_line_tangent_to_circle_l999_99968

-- Proposition 1
theorem sphere_volume_reduction (r : ℝ) (V : ℝ → ℝ) (h : V r = (4/3) * π * r^3) :
  V (r/2) = (1/8) * V r := by sorry

-- Proposition 3
theorem line_tangent_to_circle :
  let d := (1 : ℝ) / Real.sqrt 2
  (d = Real.sqrt ((1/2) : ℝ)) ∧ 
  (∀ x y : ℝ, x + y + 1 = 0 → x^2 + y^2 = 1/2 → 
    (x^2 + y^2 = d^2 ∨ x^2 + y^2 > d^2)) := by sorry

end NUMINAMATH_CALUDE_sphere_volume_reduction_line_tangent_to_circle_l999_99968


namespace NUMINAMATH_CALUDE_theta_value_l999_99954

theorem theta_value (θ : Real) (h1 : 1 / Real.sin θ + 1 / Real.cos θ = 35 / 12) 
  (h2 : θ ∈ Set.Ioo 0 (Real.pi / 2)) :
  θ = Real.arcsin (3 / 5) ∨ θ = Real.arcsin (4 / 5) := by
  sorry

end NUMINAMATH_CALUDE_theta_value_l999_99954


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l999_99912

/-- An arithmetic sequence is a sequence where the difference between 
    each consecutive term is constant. -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + seq.diff * (n - 1)

theorem arithmetic_sequence_eighth_term 
  (seq : ArithmeticSequence) 
  (h4 : seq.nthTerm 4 = 23) 
  (h6 : seq.nthTerm 6 = 47) : 
  seq.nthTerm 8 = 71 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l999_99912


namespace NUMINAMATH_CALUDE_sum_of_exponents_2023_l999_99918

/-- Represents 2023 as a sum of distinct powers of 2 -/
def representation_2023 : List ℕ :=
  [10, 9, 8, 7, 6, 5, 2, 1, 0]

/-- The sum of the exponents in the representation of 2023 -/
def sum_of_exponents : ℕ :=
  representation_2023.sum

/-- Checks if the representation is valid -/
def is_valid_representation (n : ℕ) (rep : List ℕ) : Prop :=
  n = (rep.map (fun x => 2^x)).sum ∧ rep.Nodup

theorem sum_of_exponents_2023 :
  is_valid_representation 2023 representation_2023 ∧
  sum_of_exponents = 48 := by
  sorry

#eval sum_of_exponents -- Should output 48

end NUMINAMATH_CALUDE_sum_of_exponents_2023_l999_99918


namespace NUMINAMATH_CALUDE_certain_number_proof_l999_99903

theorem certain_number_proof : ∃ x : ℝ, 0.45 * 60 = 0.35 * x + 13 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l999_99903


namespace NUMINAMATH_CALUDE_reflection_about_x_axis_l999_99947

theorem reflection_about_x_axis (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (3, a) ∧ 
    B = (3, 4) ∧ 
    A.1 = B.1 ∧ 
    A.2 = -B.2) → 
  a = -4 := by sorry

end NUMINAMATH_CALUDE_reflection_about_x_axis_l999_99947
