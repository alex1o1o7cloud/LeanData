import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_radicals_same_type_l1290_129084

theorem quadratic_radicals_same_type (x y : ℝ) (h : 3 * y = x + 2 * y + 2) : x - y = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_same_type_l1290_129084


namespace NUMINAMATH_CALUDE_regular_quad_pyramid_angle_relation_l1290_129090

/-- A regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- The dihedral angle between a lateral face and the base -/
  α : ℝ
  /-- The dihedral angle between two adjacent lateral faces -/
  β : ℝ

/-- Theorem: For a regular quadrilateral pyramid, 2 cos β + cos 2α = -1 -/
theorem regular_quad_pyramid_angle_relation (P : RegularQuadPyramid) : 
  2 * Real.cos P.β + Real.cos (2 * P.α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_regular_quad_pyramid_angle_relation_l1290_129090


namespace NUMINAMATH_CALUDE_mrs_blue_orchard_yield_l1290_129041

/-- Calculates the expected apple yield from a rectangular orchard -/
def expected_apple_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let length_ft := length_steps * step_length
  let width_ft := width_steps * step_length
  let area_sqft := length_ft * width_ft
  area_sqft * yield_per_sqft

/-- Theorem stating the expected apple yield for Mrs. Blue's orchard -/
theorem mrs_blue_orchard_yield :
  expected_apple_yield 25 20 2.5 0.75 = 2343.75 := by
  sorry

end NUMINAMATH_CALUDE_mrs_blue_orchard_yield_l1290_129041


namespace NUMINAMATH_CALUDE_necessary_unique_letters_count_l1290_129047

def word : String := "necessary"

def unique_letters (s : String) : Finset Char :=
  s.toList.toFinset

theorem necessary_unique_letters_count :
  (unique_letters word).card = 7 := by sorry

end NUMINAMATH_CALUDE_necessary_unique_letters_count_l1290_129047


namespace NUMINAMATH_CALUDE_scale_length_l1290_129082

/-- The total length of a scale divided into equal parts -/
def total_length (num_parts : ℕ) (part_length : ℝ) : ℝ :=
  num_parts * part_length

/-- Theorem: The total length of a scale is 80 inches -/
theorem scale_length : total_length 4 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_l1290_129082


namespace NUMINAMATH_CALUDE_final_position_theorem_l1290_129042

/-- Represents the position of the letter L --/
inductive LPosition
  | PosXPosY  -- Base along positive x-axis, stem along positive y-axis
  | NegXPosY  -- Base along negative x-axis, stem along positive y-axis
  | PosXNegY  -- Base along positive x-axis, stem along negative y-axis
  | NegXNegY  -- Base along negative x-axis, stem along negative y-axis

/-- Represents the transformations --/
inductive Transformation
  | RotateClockwise180
  | ReflectXAxis
  | RotateHalfTurn
  | ReflectYAxis

/-- Applies a single transformation to a given position --/
def applyTransformation (pos : LPosition) (t : Transformation) : LPosition :=
  sorry

/-- Applies a sequence of transformations to a given position --/
def applyTransformations (pos : LPosition) (ts : List Transformation) : LPosition :=
  sorry

theorem final_position_theorem :
  let initialPos := LPosition.PosXPosY
  let transformations := [
    Transformation.RotateClockwise180,
    Transformation.ReflectXAxis,
    Transformation.RotateHalfTurn,
    Transformation.ReflectYAxis
  ]
  applyTransformations initialPos transformations = LPosition.NegXNegY :=
sorry

end NUMINAMATH_CALUDE_final_position_theorem_l1290_129042


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1290_129078

theorem right_triangle_hypotenuse (DE DF : ℝ) (P Q : ℝ × ℝ) :
  DE > 0 →
  DF > 0 →
  P.1 = DE / 4 →
  P.2 = 0 →
  Q.1 = 0 →
  Q.2 = DF / 4 →
  (DE - P.1)^2 + DF^2 = 18^2 →
  DE^2 + (DF - Q.2)^2 = 30^2 →
  DE^2 + DF^2 = (24 * Real.sqrt 3)^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1290_129078


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_property_l1290_129027

/-- Definition of the sum of the first n terms of the sequence -/
def S (n : ℕ) (k : ℝ) : ℝ := k + 3^n

/-- Definition of a term in the sequence -/
def a (n : ℕ) (k : ℝ) : ℝ := S n k - S (n-1) k

/-- Theorem stating that k = -1 for the given conditions -/
theorem geometric_sequence_sum_property (k : ℝ) :
  (∀ n : ℕ, n ≥ 1 → a (n+1) k / a n k = a (n+2) k / a (n+1) k) →
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_property_l1290_129027


namespace NUMINAMATH_CALUDE_bangle_packing_optimal_solution_l1290_129089

/-- Represents the number of dozens of bangles that can be packed in each box size -/
structure BoxCapacity where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the maximum number of boxes available for each size -/
structure MaxBoxes where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of boxes used for packing -/
structure UsedBoxes where
  small : Nat
  medium : Nat
  large : Nat

/-- Check if the given number of used boxes is within the maximum allowed -/
def isValidBoxCount (used : UsedBoxes) (max : MaxBoxes) : Prop :=
  used.small ≤ max.small ∧ used.medium ≤ max.medium ∧ used.large ≤ max.large

/-- Calculate the total number of dozens packed given the box capacities and used boxes -/
def totalPacked (capacity : BoxCapacity) (used : UsedBoxes) : Nat :=
  used.small * capacity.small + used.medium * capacity.medium + used.large * capacity.large

/-- Check if the given solution packs all bangles and uses the minimum number of boxes -/
def isOptimalSolution (totalDozens : Nat) (capacity : BoxCapacity) (max : MaxBoxes) (solution : UsedBoxes) : Prop :=
  isValidBoxCount solution max ∧
  totalPacked capacity solution = totalDozens ∧
  ∀ (other : UsedBoxes), isValidBoxCount other max → totalPacked capacity other = totalDozens →
    solution.small + solution.medium + solution.large ≤ other.small + other.medium + other.large

theorem bangle_packing_optimal_solution :
  let totalDozens : Nat := 40
  let capacity : BoxCapacity := { small := 2, medium := 3, large := 4 }
  let max : MaxBoxes := { small := 6, medium := 5, large := 4 }
  let solution : UsedBoxes := { small := 5, medium := 5, large := 4 }
  isOptimalSolution totalDozens capacity max solution := by
  sorry

end NUMINAMATH_CALUDE_bangle_packing_optimal_solution_l1290_129089


namespace NUMINAMATH_CALUDE_alpha_monogram_count_l1290_129060

/-- The number of letters in the alphabet excluding 'A' -/
def n : ℕ := 25

/-- The number of initials to choose (first and middle) -/
def k : ℕ := 2

/-- The number of possible monograms for baby Alpha -/
def num_monograms : ℕ := n.choose k

theorem alpha_monogram_count : num_monograms = 300 := by
  sorry

end NUMINAMATH_CALUDE_alpha_monogram_count_l1290_129060


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1290_129094

theorem necessary_but_not_sufficient :
  (∃ x : ℝ, x > 1 ∧ ¬(Real.log (2^x) > 1)) ∧
  (∀ x : ℝ, Real.log (2^x) > 1 → x > 1) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1290_129094


namespace NUMINAMATH_CALUDE_total_yield_before_change_l1290_129071

theorem total_yield_before_change (x y z : ℝ) 
  (h1 : 0.4 * x + 0.2 * y = 5)
  (h2 : 0.4 * y + 0.2 * z = 10)
  (h3 : 0.4 * z + 0.2 * x = 9) :
  x + y + z = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_yield_before_change_l1290_129071


namespace NUMINAMATH_CALUDE_a_neq_1_necessary_not_sufficient_for_a_squared_neq_1_l1290_129009

theorem a_neq_1_necessary_not_sufficient_for_a_squared_neq_1 :
  (∀ a : ℝ, a^2 ≠ 1 → a ≠ 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ a^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_a_neq_1_necessary_not_sufficient_for_a_squared_neq_1_l1290_129009


namespace NUMINAMATH_CALUDE_ap_has_twelve_terms_l1290_129028

/-- Represents an arithmetic progression with specific properties -/
structure ArithmeticProgression where
  n : ℕ                  -- number of terms
  first_term : ℝ         -- first term
  last_term : ℝ          -- last term
  odd_sum : ℝ            -- sum of odd-numbered terms
  even_sum : ℝ           -- sum of even-numbered terms
  even_terms : Even n    -- n is even
  first_term_eq : first_term = 3
  last_term_diff : last_term = first_term + 22.5
  odd_sum_eq : odd_sum = 42
  even_sum_eq : even_sum = 48

/-- Theorem stating that the arithmetic progression satisfying given conditions has 12 terms -/
theorem ap_has_twelve_terms (ap : ArithmeticProgression) : ap.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ap_has_twelve_terms_l1290_129028


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1290_129004

-- Define the hyperbola
def Hyperbola (x y : ℝ) := y^2 - x^2/2 = 1

-- Define the asymptotes
def Asymptotes (x y : ℝ) := (x + Real.sqrt 2 * y = 0) ∨ (x - Real.sqrt 2 * y = 0)

theorem hyperbola_equation :
  ∀ (x y : ℝ),
  Asymptotes x y →
  Hyperbola (-2) (Real.sqrt 3) →
  Hyperbola x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1290_129004


namespace NUMINAMATH_CALUDE_arcsin_neg_sqrt3_over_2_l1290_129093

theorem arcsin_neg_sqrt3_over_2 : Real.arcsin (-Real.sqrt 3 / 2) = -π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_neg_sqrt3_over_2_l1290_129093


namespace NUMINAMATH_CALUDE_largest_power_dividing_product_l1290_129037

-- Define pow function
def pow (n : ℕ) : ℕ :=
  sorry

-- Define the product of pow(n) from 2 to 7000
def product : ℕ :=
  sorry

-- State the theorem
theorem largest_power_dividing_product :
  ∃ m : ℕ, (4620 ^ m : ℕ) ∣ product ∧
  ∀ k : ℕ, (4620 ^ k : ℕ) ∣ product → k ≤ m ∧
  m = 698 :=
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_product_l1290_129037


namespace NUMINAMATH_CALUDE_sum_of_absolute_roots_l1290_129091

theorem sum_of_absolute_roots (x : ℂ) : 
  x^4 - 6*x^3 + 13*x^2 - 12*x + 4 = 0 →
  ∃ r1 r2 r3 r4 : ℂ, 
    (x = r1 ∨ x = r2 ∨ x = r3 ∨ x = r4) ∧
    (Complex.abs r1 + Complex.abs r2 + Complex.abs r3 + Complex.abs r4 = 2 * Real.sqrt 6 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_absolute_roots_l1290_129091


namespace NUMINAMATH_CALUDE_stating_assignment_ways_l1290_129033

/-- Represents the number of student volunteers -/
def num_volunteers : ℕ := 5

/-- Represents the number of posts -/
def num_posts : ℕ := 4

/-- Represents the number of ways A and B can be assigned to posts -/
def ways_to_assign_A_and_B : ℕ := num_posts * (num_posts - 1)

/-- 
Theorem stating that the number of ways for A and B to each independently 
take charge of one post, while ensuring each post is staffed by at least 
one volunteer, is equal to 72.
-/
theorem assignment_ways : 
  ∃ (f : ℕ → ℕ → ℕ), 
    f ways_to_assign_A_and_B (num_volunteers - 2) = 72 ∧ 
    (∀ x y, f x y ≤ x * (y^(num_posts - 2))) := by
  sorry

end NUMINAMATH_CALUDE_stating_assignment_ways_l1290_129033


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1290_129017

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  Real.log (a 3) + Real.log (a 6) + Real.log (a 9) = 3 →
  a 1 * a 11 = 100 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1290_129017


namespace NUMINAMATH_CALUDE_telescope_visual_range_l1290_129000

/-- 
Given a telescope that increases the visual range by 50% to reach 150 kilometers,
prove that the initial visual range without the telescope was 100 kilometers.
-/
theorem telescope_visual_range 
  (increased_range : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : increased_range = 150) 
  (h2 : increase_percentage = 0.5) : 
  increased_range / (1 + increase_percentage) = 100 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l1290_129000


namespace NUMINAMATH_CALUDE_james_height_fraction_l1290_129098

/-- Proves that James was 2/3 as tall as his uncle before the growth spurt -/
theorem james_height_fraction (uncle_height : ℝ) (james_growth : ℝ) (height_difference : ℝ) :
  uncle_height = 72 →
  james_growth = 10 →
  height_difference = 14 →
  (uncle_height - (james_growth + height_difference)) / uncle_height = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_james_height_fraction_l1290_129098


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l1290_129092

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  geometric_sequence a → a 1 = 1 → a 5 = 5 → a 3 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l1290_129092


namespace NUMINAMATH_CALUDE_train_length_l1290_129080

/-- The length of a train given specific crossing times -/
theorem train_length (tree_crossing_time platform_crossing_time platform_length : ℝ) 
  (h1 : tree_crossing_time = 120)
  (h2 : platform_crossing_time = 240)
  (h3 : platform_length = 1200) : 
  ∃ (train_length : ℝ), train_length = 1200 ∧ 
    (train_length / tree_crossing_time) * platform_crossing_time = train_length + platform_length :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l1290_129080


namespace NUMINAMATH_CALUDE_hockey_players_count_l1290_129031

theorem hockey_players_count (total_players cricket_players football_players softball_players : ℕ) 
  (h1 : total_players = 55)
  (h2 : cricket_players = 15)
  (h3 : football_players = 13)
  (h4 : softball_players = 15) :
  total_players - (cricket_players + football_players + softball_players) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_hockey_players_count_l1290_129031


namespace NUMINAMATH_CALUDE_intersection_M_N_l1290_129008

def M : Set ℝ := {x | (x + 1) * (x - 3) < 0}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1290_129008


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1290_129007

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 4) → x ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1290_129007


namespace NUMINAMATH_CALUDE_elmer_pond_maturation_rate_l1290_129065

/-- The rate at which pollywogs mature and leave the pond -/
def maturation_rate (
  initial_pollywogs : ℕ
  ) (
  days_to_disappear : ℕ
  ) (
  catch_rate : ℕ
  ) (
  catch_days : ℕ
  ) : ℚ :=
  (initial_pollywogs - catch_rate * catch_days) / days_to_disappear

/-- Theorem stating the maturation rate of pollywogs in Elmer's pond -/
theorem elmer_pond_maturation_rate :
  maturation_rate 2400 44 10 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_elmer_pond_maturation_rate_l1290_129065


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1290_129003

/-- Given two adjacent vertices of a rectangle at (-3, 2) and (1, -6),
    with the third vertex forming a right angle at (-3, 2) and
    the fourth vertex aligning vertically with (-3, 2),
    prove that the area of the rectangle is 32√5. -/
theorem rectangle_area : ℝ → Prop :=
  fun area =>
    let v1 : ℝ × ℝ := (-3, 2)
    let v2 : ℝ × ℝ := (1, -6)
    let v3 : ℝ × ℝ := (-3, 2 - Real.sqrt 80)
    let v4 : ℝ × ℝ := (-3, -6)
    (v1.1 = v3.1 ∧ v1.1 = v4.1) →  -- fourth vertex aligns vertically with (-3, 2)
    (v1.2 - v3.2)^2 + (v1.1 - v2.1)^2 = (v1.2 - v2.2)^2 →  -- right angle at (-3, 2)
    area = 32 * Real.sqrt 5

-- The proof of this theorem
theorem rectangle_area_proof : rectangle_area (32 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1290_129003


namespace NUMINAMATH_CALUDE_median_eq_twelve_l1290_129096

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  -- The height of the trapezoid
  height : ℝ
  -- The angle AOD, where O is the intersection of diagonals
  angle_AOD : ℝ
  -- Assumption that the height is 4√3
  height_eq : height = 4 * Real.sqrt 3
  -- Assumption that ∠AOD is 120°
  angle_AOD_eq : angle_AOD = 120

/-- The median of an isosceles trapezoid -/
def median (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem: The median of the given isosceles trapezoid is 12 -/
theorem median_eq_twelve (t : IsoscelesTrapezoid) : median t = 12 := by sorry

end NUMINAMATH_CALUDE_median_eq_twelve_l1290_129096


namespace NUMINAMATH_CALUDE_definite_integrals_l1290_129057

theorem definite_integrals :
  (∫ (x : ℝ) in (-1)..(1), x^3) = 0 ∧
  (∫ (x : ℝ) in (2)..(ℯ + 1), 1 / (x - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integrals_l1290_129057


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1290_129005

theorem inequality_equivalence (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -4/3) :
  (x + 3) / (x - 1) > (4 * x + 5) / (3 * x + 4) ↔ 7 - Real.sqrt 66 < x ∧ x < 7 + Real.sqrt 66 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1290_129005


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1290_129048

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1290_129048


namespace NUMINAMATH_CALUDE_min_value_expression_l1290_129077

theorem min_value_expression :
  (∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ -1.125) ∧
  (∃ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = -1.125) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1290_129077


namespace NUMINAMATH_CALUDE_arithmetic_sequence_twelfth_term_l1290_129051

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence. -/
def ArithmeticSequenceTerm (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_twelfth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_third : a 3 = 13)
  (h_seventh : a 7 = 25) :
  a 12 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_twelfth_term_l1290_129051


namespace NUMINAMATH_CALUDE_chef_eggs_proof_l1290_129025

def initial_eggs (eggs_in_fridge : ℕ) (eggs_per_cake : ℕ) (num_cakes : ℕ) : ℕ :=
  eggs_in_fridge + eggs_per_cake * num_cakes

theorem chef_eggs_proof :
  initial_eggs 10 5 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_chef_eggs_proof_l1290_129025


namespace NUMINAMATH_CALUDE_margo_pairing_probability_l1290_129016

/-- The probability of a specific pairing in a class with random pairings -/
def pairingProbability (totalStudents : ℕ) (favorableOutcomes : ℕ) : ℚ :=
  favorableOutcomes / (totalStudents - 1)

/-- Theorem: The probability of Margo being paired with either Irma or Julia -/
theorem margo_pairing_probability :
  let totalStudents : ℕ := 32
  let favorableOutcomes : ℕ := 2
  pairingProbability totalStudents favorableOutcomes = 2 / 31 := by
  sorry

end NUMINAMATH_CALUDE_margo_pairing_probability_l1290_129016


namespace NUMINAMATH_CALUDE_tom_profit_l1290_129049

/-- Calculates the profit from a stock transaction -/
def calculate_profit (
  initial_shares : ℕ
  ) (initial_price : ℚ
  ) (sold_shares : ℕ
  ) (selling_price : ℚ
  ) (remaining_shares_value_multiplier : ℚ
  ) : ℚ :=
  let total_cost := initial_shares * initial_price
  let revenue_from_sold := sold_shares * selling_price
  let revenue_from_remaining := (initial_shares - sold_shares) * (initial_price * remaining_shares_value_multiplier)
  let total_revenue := revenue_from_sold + revenue_from_remaining
  total_revenue - total_cost

/-- Tom's stock transaction profit is $40 -/
theorem tom_profit : 
  calculate_profit 20 3 10 4 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_tom_profit_l1290_129049


namespace NUMINAMATH_CALUDE_geometric_series_sum_specific_geometric_series_sum_l1290_129070

theorem geometric_series_sum : ∀ (a r : ℝ), 
  a ≠ 0 → 
  |r| < 1 → 
  (∑' n, a * r^n) = a / (1 - r) :=
sorry

theorem specific_geometric_series_sum : 
  (∑' n, (1 : ℝ) * (1/4 : ℝ)^n) = 4/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_specific_geometric_series_sum_l1290_129070


namespace NUMINAMATH_CALUDE_max_boxes_in_lot_l1290_129079

theorem max_boxes_in_lot (lot_width lot_length box_width box_length : ℕ) 
  (hw : lot_width = 36)
  (hl : lot_length = 72)
  (bw : box_width = 3)
  (bl : box_length = 4) :
  (lot_width / box_width) * (lot_length / box_length) = 216 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_in_lot_l1290_129079


namespace NUMINAMATH_CALUDE_solve_for_a_l1290_129095

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 8*a^2

-- Define the theorem
theorem solve_for_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a > 0)
  (h₂ : ∀ x, f a x < 0 ↔ x₁ < x ∧ x < x₂)
  (h₃ : x₂ - x₁ = 15) :
  a = 5/2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l1290_129095


namespace NUMINAMATH_CALUDE_range_of_m_l1290_129001

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - 2*x + 2 ≠ m) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1290_129001


namespace NUMINAMATH_CALUDE_sum_three_numbers_l1290_129040

theorem sum_three_numbers (a b c M : ℤ) : 
  a + b + c = 75 ∧ 
  a + 4 = M ∧ 
  b - 5 = M ∧ 
  3 * c = M → 
  M = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l1290_129040


namespace NUMINAMATH_CALUDE_leader_assistant_selection_l1290_129063

theorem leader_assistant_selection (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_leader_assistant_selection_l1290_129063


namespace NUMINAMATH_CALUDE_min_type_A_buses_l1290_129013

/-- Represents the number of Type A buses -/
def x : ℕ := sorry

/-- The capacity of a Type A bus -/
def capacity_A : ℕ := 45

/-- The capacity of a Type B bus -/
def capacity_B : ℕ := 30

/-- The total number of people to transport -/
def total_people : ℕ := 300

/-- The total number of buses to be rented -/
def total_buses : ℕ := 8

/-- The minimum number of Type A buses needed -/
def min_buses_A : ℕ := 4

theorem min_type_A_buses :
  (∀ n : ℕ, n ≥ min_buses_A →
    capacity_A * n + capacity_B * (total_buses - n) ≥ total_people) ∧
  (∀ m : ℕ, m < min_buses_A →
    capacity_A * m + capacity_B * (total_buses - m) < total_people) :=
by sorry

end NUMINAMATH_CALUDE_min_type_A_buses_l1290_129013


namespace NUMINAMATH_CALUDE_prism_volume_l1290_129087

/-- The volume of a prism with an isosceles right triangular base and given dimensions -/
theorem prism_volume (leg : ℝ) (height : ℝ) (h_leg : leg = Real.sqrt 5) (h_height : height = 10) :
  (1 / 2) * leg * leg * height = 25 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1290_129087


namespace NUMINAMATH_CALUDE_expression_value_l1290_129099

theorem expression_value (m : ℝ) (h : m^2 + m - 1 = 0) :
  2 / (m^2 + m) - (m + 2) / (m^2 + 2*m + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1290_129099


namespace NUMINAMATH_CALUDE_max_expensive_product_price_is_30900_l1290_129019

/-- Represents a company's product line -/
structure ProductLine where
  total_products : Nat
  average_price : ℝ
  min_price : ℝ
  num_below_threshold : Nat
  threshold : ℝ

/-- Calculates the maximum possible price for the most expensive product -/
def max_expensive_product_price (pl : ProductLine) : ℝ :=
  let total_value := pl.total_products * pl.average_price
  let min_value_below_threshold := pl.num_below_threshold * pl.min_price
  let remaining_products := pl.total_products - pl.num_below_threshold
  let remaining_value := total_value - min_value_below_threshold
  let value_at_threshold := (remaining_products - 1) * pl.threshold
  remaining_value - value_at_threshold

/-- Theorem stating the maximum price of the most expensive product -/
theorem max_expensive_product_price_is_30900 :
  let pl := ProductLine.mk 40 1800 500 15 1400
  max_expensive_product_price pl = 30900 := by
  sorry

end NUMINAMATH_CALUDE_max_expensive_product_price_is_30900_l1290_129019


namespace NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l1290_129059

noncomputable section

def f (x : ℝ) : ℝ := Real.log x + 1 / x - 1

theorem f_monotonicity_and_m_range :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂) ∧ 
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ m : ℝ, (∀ a : ℝ, -1 < a ∧ a < 1 → ∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ Real.exp 1 ∧ m * a - f x₀ < 0) ↔ 
    -1 / Real.exp 1 ≤ m ∧ m ≤ 1 / Real.exp 1) :=
sorry

end

end NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l1290_129059


namespace NUMINAMATH_CALUDE_total_children_l1290_129024

theorem total_children (happy sad neutral boys girls happy_boys sad_girls : ℕ) :
  happy = 30 →
  sad = 10 →
  neutral = 20 →
  boys = 18 →
  girls = 42 →
  happy_boys = 6 →
  sad_girls = 4 →
  happy + sad + neutral = boys + girls :=
by sorry

end NUMINAMATH_CALUDE_total_children_l1290_129024


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1290_129011

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The sum of specific terms in the sequence equals 20 -/
def SumEquals20 (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 + a 7 + a 9 + a 11 = 20

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumEquals20 a) : 
  a 1 + a 13 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1290_129011


namespace NUMINAMATH_CALUDE_inverse_quadratic_equation_l1290_129029

theorem inverse_quadratic_equation (x : ℝ) :
  (1 : ℝ) = 1 / (3 * x^2 + 2 * x + 1) → x = 0 ∨ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_quadratic_equation_l1290_129029


namespace NUMINAMATH_CALUDE_equivalent_operation_l1290_129023

theorem equivalent_operation (x : ℚ) : 
  (x * (2 / 3)) / (4 / 7) = x * (7 / 6) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_operation_l1290_129023


namespace NUMINAMATH_CALUDE_unit_digit_of_fraction_l1290_129039

theorem unit_digit_of_fraction (n : ℕ) :
  (33 * 10) / (2^1984) % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_unit_digit_of_fraction_l1290_129039


namespace NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l1290_129086

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    d < 10 ∧
    e < 10 ∧
    is_prime d ∧
    is_prime e ∧
    is_prime (10 * d + e) ∧
    n = d * e * (10 * d + e) ∧
    (∀ (m : ℕ), m = d' * e' * (10 * d' + e') → 
      is_prime d' ∧ 
      is_prime e' ∧ 
      is_prime (10 * d' + e') ∧ 
      d' < 10 ∧ 
      e' < 10 → 
      m ≤ n) ∧
    sum_of_digits n = 12 :=
sorry

end NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l1290_129086


namespace NUMINAMATH_CALUDE_track_duration_in_seconds_l1290_129073

/-- Converts minutes to seconds -/
def minutesToSeconds (minutes : ℚ) : ℚ := minutes * 60

/-- The duration of the music track in minutes -/
def trackDurationMinutes : ℚ := 12.5

/-- Theorem: A music track playing for 12.5 minutes lasts 750 seconds -/
theorem track_duration_in_seconds : 
  minutesToSeconds trackDurationMinutes = 750 := by sorry

end NUMINAMATH_CALUDE_track_duration_in_seconds_l1290_129073


namespace NUMINAMATH_CALUDE_max_checkers_on_6x6_board_l1290_129058

/-- A checker placement on a 6x6 board is represented as a list of 36 booleans -/
def CheckerPlacement := List Bool

/-- A function to check if three points are collinear on a 6x6 board -/
def areCollinear (p1 p2 p3 : Nat × Nat) : Bool :=
  sorry

/-- A function to check if a placement is valid (no three checkers are collinear) -/
def isValidPlacement (placement : CheckerPlacement) : Bool :=
  sorry

/-- The maximum number of checkers that can be placed on a 6x6 board
    such that no three checkers are collinear -/
def maxCheckers : Nat := 12

/-- Theorem stating that 12 is the maximum number of checkers
    that can be placed on a 6x6 board with no three collinear -/
theorem max_checkers_on_6x6_board :
  (∀ placement : CheckerPlacement,
    isValidPlacement placement → placement.length ≤ maxCheckers) ∧
  (∃ placement : CheckerPlacement,
    isValidPlacement placement ∧ placement.length = maxCheckers) :=
sorry

end NUMINAMATH_CALUDE_max_checkers_on_6x6_board_l1290_129058


namespace NUMINAMATH_CALUDE_bacteria_growth_l1290_129076

/-- Calculates the bacteria population after a given time -/
def bacteria_population (initial_count : ℕ) (doubling_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  initial_count * 2^(elapsed_time / doubling_time)

/-- Theorem: The bacteria population after 20 minutes is 240 -/
theorem bacteria_growth : bacteria_population 15 5 20 = 240 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1290_129076


namespace NUMINAMATH_CALUDE_force_on_smooth_surface_with_pulleys_l1290_129066

/-- The force required to move a mass on a smooth horizontal surface using a pulley system -/
theorem force_on_smooth_surface_with_pulleys 
  (m : ℝ) -- mass in kg
  (g : ℝ) -- acceleration due to gravity in m/s²
  (h_m_pos : m > 0)
  (h_g_pos : g > 0) :
  ∃ F : ℝ, F = 4 * m * g :=
by sorry

end NUMINAMATH_CALUDE_force_on_smooth_surface_with_pulleys_l1290_129066


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1290_129053

theorem absolute_value_equation (a b c : ℝ) :
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) →
  ((a = 0 ∧ b = 0 ∧ (c = 1 ∨ c = -1)) ∨
   (a = 0 ∧ c = 0 ∧ (b = 1 ∨ b = -1)) ∨
   (b = 0 ∧ c = 0 ∧ (a = 1 ∨ a = -1))) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1290_129053


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1290_129035

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (3672 * 10 + N) % 6 = 0 ∧
  ∀ (M : ℕ), M ≤ 9 → (3672 * 10 + M) % 6 = 0 → M ≤ N :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1290_129035


namespace NUMINAMATH_CALUDE_remainder_2365947_div_8_l1290_129006

theorem remainder_2365947_div_8 : 2365947 % 8 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_2365947_div_8_l1290_129006


namespace NUMINAMATH_CALUDE_tan_sum_pi_12_pi_4_l1290_129056

theorem tan_sum_pi_12_pi_4 : 
  Real.tan (π / 12) + Real.tan (π / 4) = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_pi_12_pi_4_l1290_129056


namespace NUMINAMATH_CALUDE_stock_investment_income_l1290_129088

theorem stock_investment_income 
  (investment : ℝ) 
  (stock_percentage : ℝ) 
  (stock_price : ℝ) 
  (face_value : ℝ) 
  (h1 : investment = 6800) 
  (h2 : stock_percentage = 0.20) 
  (h3 : stock_price = 136) 
  (h4 : face_value = 100) : 
  ∃ (annual_income : ℝ), 
    annual_income = 1000 ∧ 
    annual_income = (investment / stock_price) * (stock_percentage * face_value) :=
by
  sorry

end NUMINAMATH_CALUDE_stock_investment_income_l1290_129088


namespace NUMINAMATH_CALUDE_lowest_cost_plan_l1290_129050

/-- Represents a plan for setting up reading corners --/
structure ReadingCornerPlan where
  medium : ℕ
  small : ℕ

/-- Checks if a plan satisfies the book constraints --/
def satisfiesBookConstraints (plan : ReadingCornerPlan) : Prop :=
  plan.medium * 80 + plan.small * 30 ≤ 1900 ∧
  plan.medium * 50 + plan.small * 60 ≤ 1620

/-- Checks if a plan satisfies the total number of corners constraint --/
def satisfiesTotalCorners (plan : ReadingCornerPlan) : Prop :=
  plan.medium + plan.small = 30

/-- Calculates the total cost of a plan --/
def totalCost (plan : ReadingCornerPlan) : ℕ :=
  plan.medium * 860 + plan.small * 570

/-- The theorem to be proved --/
theorem lowest_cost_plan :
  ∃ (plan : ReadingCornerPlan),
    satisfiesBookConstraints plan ∧
    satisfiesTotalCorners plan ∧
    plan.medium = 18 ∧
    plan.small = 12 ∧
    totalCost plan = 22320 ∧
    ∀ (other : ReadingCornerPlan),
      satisfiesBookConstraints other →
      satisfiesTotalCorners other →
      totalCost plan ≤ totalCost other :=
  sorry

end NUMINAMATH_CALUDE_lowest_cost_plan_l1290_129050


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1290_129061

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 1) * (x + 2) < 0

-- Define the solution set
def solution_set : Set ℝ := { x | -2 < x ∧ x < 1 }

-- Theorem statement
theorem inequality_solution_set : 
  { x : ℝ | inequality x } = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1290_129061


namespace NUMINAMATH_CALUDE_employees_in_all_restaurants_l1290_129068

/-- The number of employees trained to work in all three restaurants at a resort --/
theorem employees_in_all_restaurants (total_employees : ℕ) 
  (family_buffet dining_room snack_bar in_two_restaurants : ℕ) : 
  total_employees = 39 →
  family_buffet = 19 →
  dining_room = 18 →
  snack_bar = 12 →
  in_two_restaurants = 4 →
  ∃ (in_all_restaurants : ℕ),
    family_buffet + dining_room + snack_bar - in_two_restaurants - 2 * in_all_restaurants = total_employees ∧
    in_all_restaurants = 5 :=
by sorry

end NUMINAMATH_CALUDE_employees_in_all_restaurants_l1290_129068


namespace NUMINAMATH_CALUDE_longer_show_episode_length_l1290_129045

/-- Given two TV shows, prove the length of each episode of the longer show -/
theorem longer_show_episode_length 
  (total_watch_time : ℝ)
  (short_show_episode_length : ℝ)
  (short_show_episodes : ℕ)
  (long_show_episodes : ℕ)
  (h1 : total_watch_time = 24)
  (h2 : short_show_episode_length = 0.5)
  (h3 : short_show_episodes = 24)
  (h4 : long_show_episodes = 12) :
  (total_watch_time - short_show_episode_length * short_show_episodes) / long_show_episodes = 1 := by
sorry

end NUMINAMATH_CALUDE_longer_show_episode_length_l1290_129045


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_singleton_one_l1290_129046

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem A_intersect_B_eq_singleton_one : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_singleton_one_l1290_129046


namespace NUMINAMATH_CALUDE_henrys_scores_l1290_129032

theorem henrys_scores (G M : ℝ) (h1 : G + M + 66 + (G + M + 66) / 3 = 248) : G + M = 120 := by
  sorry

end NUMINAMATH_CALUDE_henrys_scores_l1290_129032


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_of_a_for_two_fixed_points_l1290_129044

/-- Definition of the function f(x) -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 2

/-- Definition of a fixed point -/
def is_fixed_point (a b x : ℝ) : Prop := f a b x = x

theorem fixed_points_for_specific_values :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ is_fixed_point 2 (-2) x₁ ∧ is_fixed_point 2 (-2) x₂ ∧ x₁ = -1 ∧ x₂ = 2 := by
  sorry

theorem range_of_a_for_two_fixed_points :
  ∀ (a : ℝ), (∀ (b : ℝ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ is_fixed_point a b x₁ ∧ is_fixed_point a b x₂) →
  (0 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_of_a_for_two_fixed_points_l1290_129044


namespace NUMINAMATH_CALUDE_total_legs_calculation_l1290_129085

/-- The total number of legs of Camden's dogs, Rico's dogs, and Samantha's cats -/
def totalLegs : ℕ := by sorry

theorem total_legs_calculation :
  let justin_dogs : ℕ := 14
  let rico_dogs : ℕ := justin_dogs + 10
  let camden_dogs : ℕ := (3 * rico_dogs) / 4
  let camden_legs : ℕ := 5 * 3 + 7 * 4 + 2 * 2
  let rico_legs : ℕ := rico_dogs * 4
  let samantha_cats : ℕ := 8
  let samantha_legs : ℕ := 6 * 4 + 2 * 3
  totalLegs = camden_legs + rico_legs + samantha_legs := by sorry

end NUMINAMATH_CALUDE_total_legs_calculation_l1290_129085


namespace NUMINAMATH_CALUDE_square_area_ratio_l1290_129074

theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b = a * Real.sqrt 3) :
  b ^ 2 = 3 * a ^ 2 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1290_129074


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l1290_129055

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

theorem product_trailing_zeros :
  trailing_zeros 100 = 24 :=
sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l1290_129055


namespace NUMINAMATH_CALUDE_fraction_simplification_l1290_129097

theorem fraction_simplification (x : ℝ) (h : x ≠ -2 ∧ x ≠ 2) :
  (x^2 - 4) / (x^2 - 4*x + 4) / ((x^2 + 4*x + 4) / (2*x - x^2)) = -x / (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1290_129097


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1290_129030

/-- Given a quadratic equation 5x^2 - 11x - 14 = 0, prove that the positive difference
    between its roots is √401/5 and that p + q = 406 --/
theorem quadratic_root_difference (x : ℝ) : 
  let a : ℝ := 5
  let b : ℝ := -11
  let c : ℝ := -14
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  let difference := |root1 - root2|
  difference = Real.sqrt 401 / 5 ∧ 401 + 5 = 406 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1290_129030


namespace NUMINAMATH_CALUDE_sample_size_is_200_l1290_129075

/-- Represents a statistical survey of students -/
structure StudentSurvey where
  total_students : ℕ
  selected_students : ℕ

/-- Definition of sample size for a student survey -/
def sample_size (survey : StudentSurvey) : ℕ := survey.selected_students

/-- Theorem stating that for the given survey, the sample size is 200 -/
theorem sample_size_is_200 (survey : StudentSurvey) 
  (h1 : survey.total_students = 2000) 
  (h2 : survey.selected_students = 200) : 
  sample_size survey = 200 := by
  sorry

#check sample_size_is_200

end NUMINAMATH_CALUDE_sample_size_is_200_l1290_129075


namespace NUMINAMATH_CALUDE_expression_equality_l1290_129062

theorem expression_equality : (34 + 7)^2 - (7^2 + 34^2 + 7 * 34) = 238 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1290_129062


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l1290_129054

/-- Given a triangle with side lengths a, b, c, 
    prove that a^2(b+c-a) + b^2(c+a-b) + c^2(a+b-c) ≤ 3abc -/
theorem triangle_inequality_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l1290_129054


namespace NUMINAMATH_CALUDE_smallest_integer_solution_inequality_neg_two_satisfies_inequality_neg_two_is_smallest_integer_solution_l1290_129026

theorem smallest_integer_solution_inequality :
  ∀ x : ℤ, (9*x + 8)/6 - x/3 ≥ -1 → x ≥ -2 :=
by
  sorry

theorem neg_two_satisfies_inequality :
  (9*(-2) + 8)/6 - (-2)/3 ≥ -1 :=
by
  sorry

theorem neg_two_is_smallest_integer_solution :
  ∀ y : ℤ, y < -2 → (9*y + 8)/6 - y/3 < -1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_inequality_neg_two_satisfies_inequality_neg_two_is_smallest_integer_solution_l1290_129026


namespace NUMINAMATH_CALUDE_conference_handshakes_l1290_129083

/-- The number of companies at the conference -/
def num_companies : ℕ := 5

/-- The number of representatives per company -/
def reps_per_company : ℕ := 5

/-- The total number of people at the conference -/
def total_people : ℕ := num_companies * reps_per_company

/-- The number of people each person shakes hands with -/
def handshakes_per_person : ℕ := total_people - reps_per_company

/-- The total number of handshakes at the conference -/
def total_handshakes : ℕ := (total_people * handshakes_per_person) / 2

theorem conference_handshakes : total_handshakes = 250 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1290_129083


namespace NUMINAMATH_CALUDE_power_of_i_sum_l1290_129014

theorem power_of_i_sum (i : ℂ) : i^2 = -1 → i^44 + i^444 + 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_i_sum_l1290_129014


namespace NUMINAMATH_CALUDE_greg_pages_per_day_l1290_129038

/-- 
Given that Brad reads 26 pages per day and 8 more pages than Greg each day,
prove that Greg reads 18 pages per day.
-/
theorem greg_pages_per_day 
  (brad_pages : ℕ) 
  (difference : ℕ) 
  (h1 : brad_pages = 26)
  (h2 : difference = 8)
  : brad_pages - difference = 18 := by
  sorry

end NUMINAMATH_CALUDE_greg_pages_per_day_l1290_129038


namespace NUMINAMATH_CALUDE_volume_per_balloon_l1290_129012

/-- Given the number of balloons, volume of each gas tank, and number of tanks needed,
    prove that the volume of air per balloon is 10 liters. -/
theorem volume_per_balloon
  (num_balloons : ℕ)
  (tank_volume : ℕ)
  (num_tanks : ℕ)
  (h1 : num_balloons = 1000)
  (h2 : tank_volume = 500)
  (h3 : num_tanks = 20) :
  (num_tanks * tank_volume) / num_balloons = 10 :=
by sorry

end NUMINAMATH_CALUDE_volume_per_balloon_l1290_129012


namespace NUMINAMATH_CALUDE_difference_is_nine_l1290_129036

theorem difference_is_nine (a b c d : ℝ) 
  (h1 : ∃ x, a - b = c + d + x)
  (h2 : a + b = c - d - 3)
  (h3 : a - c = 3) :
  (a - b) - (c + d) = 9 := by
  sorry

end NUMINAMATH_CALUDE_difference_is_nine_l1290_129036


namespace NUMINAMATH_CALUDE_expression_factorization_l1290_129034

theorem expression_factorization (x : ℝ) : 
  (20 * x^3 - 100 * x^2 + 90 * x - 10) - (5 * x^3 - 10 * x^2 + 5) = 
  15 * (x^3 - 6 * x^2 + 6 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1290_129034


namespace NUMINAMATH_CALUDE_garden_plants_correct_l1290_129069

/-- Calculates the total number of plants in Papi Calot's garden -/
def garden_plants : ℕ × ℕ × ℕ :=
  let potato_rows := 8
  let potato_alt1 := 22
  let potato_alt2 := 25
  let potato_extra := 18

  let carrot_rows := 12
  let carrot_start := 30
  let carrot_increment := 5
  let carrot_extra := 24

  let onion_repetitions := 4
  let onion_row1 := 15
  let onion_row2 := 20
  let onion_row3 := 25
  let onion_extra := 12

  let potatoes := (potato_rows / 2 * potato_alt1 + potato_rows / 2 * potato_alt2) + potato_extra
  let carrots := (carrot_rows * (2 * carrot_start + (carrot_rows - 1) * carrot_increment)) / 2 + carrot_extra
  let onions := onion_repetitions * (onion_row1 + onion_row2 + onion_row3) + onion_extra

  (potatoes, carrots, onions)

theorem garden_plants_correct :
  garden_plants = (206, 714, 252) :=
by sorry

end NUMINAMATH_CALUDE_garden_plants_correct_l1290_129069


namespace NUMINAMATH_CALUDE_factorial_sum_quotient_l1290_129021

theorem factorial_sum_quotient (n : ℕ) (h : n ≥ 2) :
  (Nat.factorial (n + 2) + Nat.factorial (n + 1)) / Nat.factorial (n + 1) = n + 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_quotient_l1290_129021


namespace NUMINAMATH_CALUDE_student_count_l1290_129064

theorem student_count (total_eggs : ℕ) (eggs_per_student : ℕ) (h1 : total_eggs = 56) (h2 : eggs_per_student = 8) :
  total_eggs / eggs_per_student = 7 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1290_129064


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_specific_case_l1290_129022

theorem simplify_and_evaluate (a b : ℝ) :
  (a - b)^2 + (a + 3*b)*(a - 3*b) - a*(a - 2*b) = a^2 - 8*b^2 :=
by sorry

theorem specific_case : 
  let a : ℝ := -1
  let b : ℝ := 2
  (a - b)^2 + (a + 3*b)*(a - 3*b) - a*(a - 2*b) = -31 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_specific_case_l1290_129022


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l1290_129002

theorem arithmetic_sequence_count (a₁ aₙ d : ℤ) (n : ℕ) : 
  a₁ = 165 ∧ aₙ = 40 ∧ d = -5 ∧ aₙ = a₁ + (n - 1) * d → n = 26 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l1290_129002


namespace NUMINAMATH_CALUDE_quadratic_no_roots_if_geometric_sequence_l1290_129018

/-- A geometric sequence is a sequence where each term after the first is found by 
    multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

/-- A quadratic function f(x) = ax² + bx + c has no real roots if and only if
    its discriminant is negative. -/
def HasNoRealRoots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

theorem quadratic_no_roots_if_geometric_sequence (a b c : ℝ) (ha : a ≠ 0) :
  IsGeometricSequence a b c → HasNoRealRoots a b c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_if_geometric_sequence_l1290_129018


namespace NUMINAMATH_CALUDE_minimum_fifth_quarter_score_l1290_129081

def required_average : ℚ := 85
def num_quarters : ℕ := 5
def first_four_scores : List ℚ := [84, 80, 78, 82]

theorem minimum_fifth_quarter_score :
  let total_required := required_average * num_quarters
  let sum_first_four := first_four_scores.sum
  let min_fifth_score := total_required - sum_first_four
  min_fifth_score = 101 := by sorry

end NUMINAMATH_CALUDE_minimum_fifth_quarter_score_l1290_129081


namespace NUMINAMATH_CALUDE_sqrt_three_product_l1290_129010

theorem sqrt_three_product : 5 * Real.sqrt 3 * (2 * Real.sqrt 3) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_product_l1290_129010


namespace NUMINAMATH_CALUDE_used_car_clients_l1290_129072

theorem used_car_clients (total_cars : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) :
  total_cars = 12 →
  cars_per_client = 4 →
  selections_per_car = 3 →
  (total_cars * selections_per_car) / cars_per_client = 9 :=
by sorry

end NUMINAMATH_CALUDE_used_car_clients_l1290_129072


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1290_129020

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (AB : ℝ) (BC : ℝ) (AD : ℝ) (DC : ℝ)
  (AB_perp_BC : AB = BC)
  (AD_perp_DC : AD = DC)
  (AB_eq_9 : AB = 9)
  (AD_eq_8 : AD = 8)

/-- The area of the quadrilateral ABCD is 82.5 square units -/
theorem quadrilateral_area (q : Quadrilateral) : Real.sqrt ((q.AB ^ 2 + q.BC ^ 2) * (q.AD ^ 2 + q.DC ^ 2)) / 2 = 82.5 := by
  sorry

#check quadrilateral_area

end NUMINAMATH_CALUDE_quadrilateral_area_l1290_129020


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1290_129067

/-- Proves that the line y - 2 = mx + m passes through the point (-1, 2) for any real m -/
theorem line_passes_through_fixed_point (m : ℝ) : 
  2 - 2 = m * (-1) + m := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1290_129067


namespace NUMINAMATH_CALUDE_range_of_m_l1290_129052

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (25 - m) + y^2 / (m - 7) = 1 ∧ 
  (25 - m > 0) ∧ (m - 7 > 0) ∧ (25 - m > m - 7)

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∃ (e : ℝ), (∃ (x y : ℝ), y^2 / 5 - x^2 / m = 1) ∧ 
  1 < e ∧ e < 2 ∧ e^2 = (5 + m) / 5

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (¬(¬(p m) ∨ ¬(q m))) → (7 < m ∧ m < 15) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1290_129052


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1290_129015

theorem sum_of_three_numbers (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 →
  (a + b + c) / 3 = a + 15 →
  (a + b + c) / 3 = c - 20 →
  c = 2 * a →
  a + b + c = 115 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1290_129015


namespace NUMINAMATH_CALUDE_candy_bar_cost_after_tax_l1290_129043

-- Define the initial amount Peter has
def initial_amount : ℝ := 10

-- Define the cost per ounce of soda
def soda_cost_per_ounce : ℝ := 0.25

-- Define the number of ounces of soda bought
def soda_ounces : ℝ := 16

-- Define the original price of chips
def chips_original_price : ℝ := 2.50

-- Define the discount rate for chips
def chips_discount_rate : ℝ := 0.1

-- Define the price of the candy bar
def candy_bar_price : ℝ := 1.25

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.08

-- Define the function to calculate the discounted price of chips
def discounted_chips_price : ℝ := chips_original_price * (1 - chips_discount_rate)

-- Define the function to calculate the total cost before tax
def total_cost_before_tax : ℝ := soda_cost_per_ounce * soda_ounces + discounted_chips_price + candy_bar_price

-- Define the function to calculate the total cost after tax
def total_cost_after_tax : ℝ := total_cost_before_tax * (1 + sales_tax_rate)

-- Theorem: The cost of the candy bar after tax is $1.35
theorem candy_bar_cost_after_tax :
  candy_bar_price * (1 + sales_tax_rate) = 1.35 ∧ total_cost_after_tax = initial_amount :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_after_tax_l1290_129043
