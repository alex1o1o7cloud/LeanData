import Mathlib

namespace NUMINAMATH_CALUDE_real_complex_intersection_l3879_387942

-- Define the set of real numbers
def RealNumbers : Set ℂ := {z : ℂ | z.im = 0}

-- Define the set of complex numbers
def ComplexNumbers : Set ℂ := Set.univ

-- Theorem statement
theorem real_complex_intersection :
  RealNumbers ∩ ComplexNumbers = RealNumbers := by sorry

end NUMINAMATH_CALUDE_real_complex_intersection_l3879_387942


namespace NUMINAMATH_CALUDE_graph_vertical_shift_l3879_387916

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the vertical translation
def verticalShift (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := fun x ↦ f x + c

-- Theorem statement
theorem graph_vertical_shift :
  ∀ (x y : ℝ), y = f x ↔ y + 1 = verticalShift f 1 x := by
  sorry

end NUMINAMATH_CALUDE_graph_vertical_shift_l3879_387916


namespace NUMINAMATH_CALUDE_max_grain_mass_on_platform_l3879_387928

/-- Represents a rectangular platform with grain piled on it. -/
structure GrainPlatform where
  length : ℝ
  width : ℝ
  grainDensity : ℝ
  maxAngle : ℝ

/-- Calculates the maximum mass of grain on the platform. -/
def maxGrainMass (platform : GrainPlatform) : ℝ :=
  sorry

/-- Theorem stating the maximum mass of grain on the given platform. -/
theorem max_grain_mass_on_platform :
  let platform : GrainPlatform := {
    length := 8,
    width := 5,
    grainDensity := 1200,
    maxAngle := π/4
  }
  maxGrainMass platform = 47500 := by sorry

end NUMINAMATH_CALUDE_max_grain_mass_on_platform_l3879_387928


namespace NUMINAMATH_CALUDE_bd_length_is_ten_l3879_387901

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the right angle at C
def isRightAngleAtC (t : Triangle) : Prop :=
  let (xa, ya) := t.A
  let (xb, yb) := t.B
  let (xc, yc) := t.C
  (xc - xa) * (xc - xb) + (yc - ya) * (yc - yb) = 0

-- Define the lengths of AC and BC
def AC_length (t : Triangle) : ℝ := 5
def BC_length (t : Triangle) : ℝ := 12

-- Define points D, E, F
def D (t : Triangle) : ℝ × ℝ := sorry
def E (t : Triangle) : ℝ × ℝ := sorry
def F (t : Triangle) : ℝ × ℝ := sorry

-- Define the right angle at FED
def isRightAngleAtFED (t : Triangle) : Prop :=
  let (xd, yd) := D t
  let (xe, ye) := E t
  let (xf, yf) := F t
  (xf - xd) * (xe - xd) + (yf - yd) * (ye - yd) = 0

-- Define the lengths of DE and DF
def DE_length (t : Triangle) : ℝ := 5
def DF_length (t : Triangle) : ℝ := 3

-- Define the length of BD
def BD_length (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem bd_length_is_ten (t : Triangle) :
  isRightAngleAtC t →
  isRightAngleAtFED t →
  BD_length t = 10 := by sorry

end NUMINAMATH_CALUDE_bd_length_is_ten_l3879_387901


namespace NUMINAMATH_CALUDE_waynes_blocks_l3879_387944

theorem waynes_blocks (initial_blocks additional_blocks : ℕ) 
  (h1 : initial_blocks = 9)
  (h2 : additional_blocks = 6) :
  initial_blocks + additional_blocks = 15 := by
  sorry

end NUMINAMATH_CALUDE_waynes_blocks_l3879_387944


namespace NUMINAMATH_CALUDE_gcd_of_36_and_60_l3879_387965

theorem gcd_of_36_and_60 : Nat.gcd 36 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_36_and_60_l3879_387965


namespace NUMINAMATH_CALUDE_adults_attending_play_l3879_387972

/-- Proves the number of adults attending a play given the total attendance,
    admission prices, and total receipts. -/
theorem adults_attending_play
  (total_people : ℕ)
  (adult_price child_price : ℕ)
  (total_receipts : ℕ)
  (h1 : total_people = 610)
  (h2 : adult_price = 2)
  (h3 : child_price = 1)
  (h4 : total_receipts = 960) :
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_receipts ∧
    adults = 350 :=
by sorry

end NUMINAMATH_CALUDE_adults_attending_play_l3879_387972


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3879_387931

theorem product_of_three_numbers (x y z : ℝ) : 
  x + y + z = 210 → 
  9 * x = y - 12 → 
  9 * x = z + 12 → 
  x < y → 
  x < z → 
  x * y * z = 746397 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3879_387931


namespace NUMINAMATH_CALUDE_emma_age_when_sister_is_56_l3879_387940

theorem emma_age_when_sister_is_56 (emma_current_age : ℕ) (age_difference : ℕ) (sister_future_age : ℕ) :
  emma_current_age = 7 →
  age_difference = 9 →
  sister_future_age = 56 →
  sister_future_age - age_difference = 47 :=
by sorry

end NUMINAMATH_CALUDE_emma_age_when_sister_is_56_l3879_387940


namespace NUMINAMATH_CALUDE_f_composition_equals_pi_plus_two_l3879_387903

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 2
  else if x = 0 then Real.pi
  else 0

-- State the theorem
theorem f_composition_equals_pi_plus_two :
  f (f (f (-2))) = Real.pi + 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_pi_plus_two_l3879_387903


namespace NUMINAMATH_CALUDE_vehicle_value_last_year_l3879_387907

theorem vehicle_value_last_year 
  (value_this_year : ℝ)
  (ratio : ℝ)
  (h1 : value_this_year = 16000)
  (h2 : ratio = 0.8)
  (h3 : value_this_year = ratio * value_last_year) :
  value_last_year = 20000 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_last_year_l3879_387907


namespace NUMINAMATH_CALUDE_value_range_of_f_l3879_387981

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem value_range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -3 ≤ y ∧ y ≤ 1} := by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l3879_387981


namespace NUMINAMATH_CALUDE_stake_B_maximizes_grazing_area_l3879_387953

/-- Represents a stake on the edge of the pond -/
inductive Stake
| A
| B
| C
| D

/-- The side length of the square pond in meters -/
def pondSideLength : ℝ := 12

/-- The distance between adjacent stakes in meters -/
def stakesDistance : ℝ := 3

/-- The length of the rope in meters -/
def ropeLength : ℝ := 4

/-- Calculates the grazing area for a given stake -/
noncomputable def grazingArea (s : Stake) : ℝ :=
  match s with
  | Stake.A => 4.25 * Real.pi
  | Stake.B => 8 * Real.pi
  | Stake.C => 4.25 * Real.pi
  | Stake.D => 4.25 * Real.pi

/-- Theorem stating that stake B maximizes the grazing area -/
theorem stake_B_maximizes_grazing_area :
  ∀ s : Stake, grazingArea Stake.B ≥ grazingArea s :=
sorry


end NUMINAMATH_CALUDE_stake_B_maximizes_grazing_area_l3879_387953


namespace NUMINAMATH_CALUDE_sandwich_combinations_l3879_387957

theorem sandwich_combinations (salami_types : Nat) (cheese_types : Nat) (sauce_types : Nat) :
  salami_types = 8 →
  cheese_types = 7 →
  sauce_types = 3 →
  (salami_types * Nat.choose cheese_types 2 * sauce_types) = 504 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l3879_387957


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l3879_387994

theorem min_sum_absolute_values :
  ∀ x : ℝ, |x + 3| + |x + 5| + |x + 6| ≥ 5 ∧ ∃ x : ℝ, |x + 3| + |x + 5| + |x + 6| = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l3879_387994


namespace NUMINAMATH_CALUDE_triangle_value_l3879_387993

theorem triangle_value (q : ℝ) (h1 : 2 * triangle + q = 134) (h2 : 2 * (triangle + q) + q = 230) :
  triangle = 43 := by sorry

end NUMINAMATH_CALUDE_triangle_value_l3879_387993


namespace NUMINAMATH_CALUDE_flu_infection_rate_l3879_387983

theorem flu_infection_rate : ∃ x : ℝ, x > 0 ∧ 1 + x + x^2 = 121 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_flu_infection_rate_l3879_387983


namespace NUMINAMATH_CALUDE_world_grain_ratio_l3879_387927

def world_grain_supply : ℕ := 1800000
def world_grain_demand : ℕ := 2400000

theorem world_grain_ratio : 
  (world_grain_supply : ℚ) / world_grain_demand = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_world_grain_ratio_l3879_387927


namespace NUMINAMATH_CALUDE_inverse_composition_l3879_387973

-- Define the function f
def f : ℕ → ℕ
| 3 => 10
| 4 => 17
| 5 => 26
| 6 => 37
| 7 => 50
| _ => 0  -- Default case for other inputs

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 10 => 3
| 17 => 4
| 26 => 5
| 37 => 6
| 50 => 7
| _ => 0  -- Default case for other inputs

-- Theorem statement
theorem inverse_composition :
  f_inv (f_inv 50 * f_inv 10 + f_inv 26) = 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_l3879_387973


namespace NUMINAMATH_CALUDE_m_value_theorem_l3879_387954

theorem m_value_theorem (m : ℕ) : 
  2^2000 - 3 * 2^1999 + 5 * 2^1998 - 2^1997 = m * 2^1997 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_m_value_theorem_l3879_387954


namespace NUMINAMATH_CALUDE_can_identify_80_weights_l3879_387952

/-- Represents a comparison between two sets of weights -/
def Comparison := List ℕ → List ℕ → Bool

/-- Given a list of weights and a number of comparisons, 
    determines if it's possible to uniquely identify all weights -/
def can_identify (weights : List ℕ) (num_comparisons : ℕ) : Prop :=
  ∃ (comparisons : List Comparison), 
    comparisons.length = num_comparisons ∧ 
    ∀ (w1 w2 : List ℕ), w1 ≠ w2 → w1.length = weights.length → w2.length = weights.length →
      ∃ (c : Comparison), c ∈ comparisons ∧ c w1 ≠ c w2

theorem can_identify_80_weights :
  ∀ (weights : List ℕ), 
    weights.length = 80 → 
    weights.Nodup → 
    (can_identify weights 4 ∧ ¬can_identify weights 3) := by
  sorry

#check can_identify_80_weights

end NUMINAMATH_CALUDE_can_identify_80_weights_l3879_387952


namespace NUMINAMATH_CALUDE_intercepted_line_with_midpoint_at_origin_l3879_387998

/-- Given two lines l₁ and l₂, prove that the line x + 6y = 0 is intercepted by both lines
    and has its midpoint at the origin. -/
theorem intercepted_line_with_midpoint_at_origin :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 4*x + y + 6 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ 3*x - 5*y - 6 = 0
  let intercepted_line : ℝ → ℝ → Prop := λ x y ↦ x + 6*y = 0
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    l₁ x₁ y₁ ∧ l₂ x₂ y₂ ∧
    intercepted_line x₁ y₁ ∧ intercepted_line x₂ y₂ ∧
    (x₁ + x₂) / 2 = 0 ∧ (y₁ + y₂) / 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_intercepted_line_with_midpoint_at_origin_l3879_387998


namespace NUMINAMATH_CALUDE_emily_egg_collection_l3879_387974

/-- The number of baskets Emily used --/
def num_baskets : ℕ := 1525

/-- The average number of eggs per basket --/
def eggs_per_basket : ℚ := 37.5

/-- The total number of eggs collected --/
def total_eggs : ℚ := num_baskets * eggs_per_basket

/-- Theorem stating that the total number of eggs is 57,187.5 --/
theorem emily_egg_collection :
  total_eggs = 57187.5 := by sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l3879_387974


namespace NUMINAMATH_CALUDE_arithmetic_problem_l3879_387986

theorem arithmetic_problem : 40 + 5 * 12 / (180 / 3) = 41 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l3879_387986


namespace NUMINAMATH_CALUDE_additional_blurays_is_six_l3879_387997

/-- Represents the movie collection and purchase scenario -/
structure MovieCollection where
  initialDVDRatio : Nat
  initialBluRayRatio : Nat
  newDVDRatio : Nat
  newBluRayRatio : Nat
  totalInitialMovies : Nat

/-- Calculates the number of additional Blu-ray movies purchased -/
def additionalBluRays (mc : MovieCollection) : Nat :=
  let initialX := mc.totalInitialMovies / (mc.initialDVDRatio + mc.initialBluRayRatio)
  let initialDVD := mc.initialDVDRatio * initialX
  let initialBluRay := mc.initialBluRayRatio * initialX
  ((initialDVD * mc.newBluRayRatio) - (initialBluRay * mc.newDVDRatio)) / mc.newDVDRatio

/-- Theorem stating that the number of additional Blu-ray movies purchased is 6 -/
theorem additional_blurays_is_six (mc : MovieCollection) 
  (h1 : mc.initialDVDRatio = 7)
  (h2 : mc.initialBluRayRatio = 2)
  (h3 : mc.newDVDRatio = 13)
  (h4 : mc.newBluRayRatio = 4)
  (h5 : mc.totalInitialMovies = 351) :
  additionalBluRays mc = 6 := by
  sorry

#eval additionalBluRays { initialDVDRatio := 7, initialBluRayRatio := 2, 
                          newDVDRatio := 13, newBluRayRatio := 4, 
                          totalInitialMovies := 351 }

end NUMINAMATH_CALUDE_additional_blurays_is_six_l3879_387997


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3879_387977

/-- A structure made of unit cubes -/
structure CubeStructure where
  /-- The number of unit cubes in the structure -/
  num_cubes : ℕ
  /-- The volume of the structure in cubic units -/
  volume : ℕ
  /-- The surface area of the structure in square units -/
  surface_area : ℕ
  /-- The structure has a central cube surrounded symmetrically on all faces except the bottom -/
  has_central_cube : Prop
  /-- The structure forms a large plus sign when viewed from the top -/
  is_plus_shaped : Prop

/-- The specific cube structure described in the problem -/
def plus_structure : CubeStructure :=
  { num_cubes := 9
  , volume := 9
  , surface_area := 31
  , has_central_cube := True
  , is_plus_shaped := True }

/-- The theorem stating that the ratio of volume to surface area for the plus_structure is 9/31 -/
theorem volume_to_surface_area_ratio (s : CubeStructure) (h1 : s = plus_structure) :
  (s.volume : ℚ) / s.surface_area = 9 / 31 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3879_387977


namespace NUMINAMATH_CALUDE_quadratic_intersection_on_y_axis_l3879_387966

theorem quadratic_intersection_on_y_axis (m : ℝ) : 
  (∃ y : ℝ, x^2 - 2*(m-1)*x + m^2 - 2*m - 3 = -x^2 + 6*x ∧ x = 0) ↔ 
  (m = -1 ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_on_y_axis_l3879_387966


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l3879_387926

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then
    []
  else
    (n % 2 = 1) :: nat_to_binary (n / 2)

def a : List Bool := [true, false, true, true, false, true]  -- 101101₂
def b : List Bool := [true, true, true]  -- 111₂
def c : List Bool := [false, true, true, false, false, true, true]  -- 1100110₂
def d : List Bool := [false, true, false, true]  -- 1010₂
def result : List Bool := [true, false, true, true, true, false, true, true]  -- 11011101₂

theorem binary_addition_subtraction :
  nat_to_binary ((binary_to_nat a + binary_to_nat b + binary_to_nat c) - binary_to_nat d) = result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l3879_387926


namespace NUMINAMATH_CALUDE_fraction_simplification_l3879_387924

theorem fraction_simplification : (200 + 10) / (20 + 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3879_387924


namespace NUMINAMATH_CALUDE_yulia_number_l3879_387918

theorem yulia_number (x : ℝ) : x + 13 = 4 * (x + 1) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_yulia_number_l3879_387918


namespace NUMINAMATH_CALUDE_initial_men_count_l3879_387988

/-- The initial number of men working on the construction job -/
def initial_men : ℕ := sorry

/-- The total amount of work to be done -/
def total_work : ℝ := sorry

/-- Half of the job is finished in 15 days with the initial number of men -/
axiom half_job_rate : (initial_men : ℝ) * 15 = total_work / 2

/-- The remaining job is completed in 25 days with two fewer men -/
axiom remaining_job_rate : (initial_men - 2 : ℝ) * 25 = total_work / 2

/-- The initial number of men is 5 -/
theorem initial_men_count : initial_men = 5 := by sorry

end NUMINAMATH_CALUDE_initial_men_count_l3879_387988


namespace NUMINAMATH_CALUDE_draw_XS_count_l3879_387967

/-- The number of X cards in the set -/
def num_X : ℕ := 3

/-- The number of S cards in the set -/
def num_S : ℕ := 2

/-- The total number of cards in the set -/
def total_cards : ℕ := 10

/-- The number of ways to draw "XS" in order from the given set of cards -/
def ways_to_draw_XS : ℕ := num_X * num_S

theorem draw_XS_count : ways_to_draw_XS = 6 := by
  sorry

end NUMINAMATH_CALUDE_draw_XS_count_l3879_387967


namespace NUMINAMATH_CALUDE_combined_tax_rate_l3879_387902

/-- Calculates the combined tax rate for Mork and Mindy -/
theorem combined_tax_rate (mork_rate : ℚ) (mindy_rate : ℚ) (income_ratio : ℚ) : 
  mork_rate = 40 / 100 →
  mindy_rate = 30 / 100 →
  income_ratio = 3 →
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 325 / 1000 := by
  sorry

#eval (40 / 100 + 30 / 100 * 3) / (1 + 3)

end NUMINAMATH_CALUDE_combined_tax_rate_l3879_387902


namespace NUMINAMATH_CALUDE_largest_number_proof_l3879_387930

def largest_number (a b : ℕ+) : Prop :=
  let hcf := Nat.gcd a b
  let lcm := Nat.lcm a b
  hcf = 154 ∧
  ∃ (k : ℕ), lcm = hcf * 19 * 23 * 37 * k ∧
  max a b = hcf * 19 * 23 * 37

theorem largest_number_proof (a b : ℕ+) (h : largest_number a b) :
  max a b = 2493726 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_proof_l3879_387930


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3879_387913

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 + (x + 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a₀| + |a₁| + |a₂| + |a₅| = 105 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3879_387913


namespace NUMINAMATH_CALUDE_angle_measure_l3879_387909

theorem angle_measure : ∃ x : ℝ, 
  (x + (5 * x + 12) = 180) ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3879_387909


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3879_387922

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), 
    (x^6 + x^3 + x^3*y + y = 147^157) ∧ 
    (x^3 + x^3*y + y^2 + y + z^9 = 157^147) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3879_387922


namespace NUMINAMATH_CALUDE_gcd_272_595_l3879_387919

theorem gcd_272_595 : Nat.gcd 272 595 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_272_595_l3879_387919


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l3879_387917

theorem set_equality_implies_a_equals_three (a : ℝ) : 
  ({0, 1, a^2} : Set ℝ) = ({1, 0, 2*a + 3} : Set ℝ) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l3879_387917


namespace NUMINAMATH_CALUDE_wall_ratio_l3879_387938

/-- Given a wall with specific dimensions, prove that the ratio of its length to its height is 7:1 -/
theorem wall_ratio (w h l : ℝ) : 
  w = 3 →                 -- width is 3 meters
  h = 6 * w →             -- height is 6 times the width
  w * h * l = 6804 →      -- volume is 6804 cubic meters
  l / h = 7 := by
sorry

end NUMINAMATH_CALUDE_wall_ratio_l3879_387938


namespace NUMINAMATH_CALUDE_bd_squared_equals_36_l3879_387914

theorem bd_squared_equals_36 
  (a b c d : ℤ) 
  (h1 : a - b - c + d = 18) 
  (h2 : a + b - c - d = 6) : 
  (b - d)^2 = 36 := by
sorry

end NUMINAMATH_CALUDE_bd_squared_equals_36_l3879_387914


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3879_387956

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ ((x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3879_387956


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l3879_387958

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 49
def C2 (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 64

-- Define the tangent line segment
def is_tangent (P Q : ℝ × ℝ) : Prop :=
  C1 P.1 P.2 ∧ C2 Q.1 Q.2 ∧ 
  ∀ R : ℝ × ℝ, (C1 R.1 R.2 ∨ C2 R.1 R.2) → 
    (R.1 - P.1)^2 + (R.2 - P.2)^2 ≤ (Q.1 - P.1)^2 + (Q.2 - P.2)^2

-- State the theorem
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, is_tangent P Q ∧
    ∀ P' Q' : ℝ × ℝ, is_tangent P' Q' →
      Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) ≤ 
      Real.sqrt ((Q'.1 - P'.1)^2 + (Q'.2 - P'.2)^2) ∧
    Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = Real.sqrt 207 + Real.sqrt 132 :=
sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l3879_387958


namespace NUMINAMATH_CALUDE_tan_beta_value_l3879_387989

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = -2/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3879_387989


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_l3879_387941

/-- The first term of an infinite geometric series with common ratio 1/4 and sum 80 is 60 -/
theorem first_term_of_geometric_series : 
  ∀ (a : ℝ), 
  (a * (1 - (1/4)⁻¹) = 80) → 
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_l3879_387941


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3879_387959

/-- 
Given an article with a selling price of 600 and a cost price of 375,
prove that the profit percentage is 60%.
-/
theorem profit_percentage_calculation (selling_price cost_price : ℝ) 
  (h1 : selling_price = 600)
  (h2 : cost_price = 375) : 
  (selling_price - cost_price) / cost_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3879_387959


namespace NUMINAMATH_CALUDE_prism_faces_l3879_387968

theorem prism_faces (E V : ℕ) (h : E + V = 30) : ∃ (F : ℕ), F = 8 ∧ F + V = E + 2 := by
  sorry

end NUMINAMATH_CALUDE_prism_faces_l3879_387968


namespace NUMINAMATH_CALUDE_condition_D_not_sufficient_condition_A_sufficient_condition_B_sufficient_condition_C_sufficient_l3879_387908

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles

-- Define similarity relation between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the four conditions
def condition_A (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β

def condition_B (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.γ = t2.γ

def condition_C (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

def condition_D (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b

-- Theorem stating that condition D is not sufficient for similarity
theorem condition_D_not_sufficient :
  ∃ t1 t2 : Triangle, condition_D t1 t2 ∧ ¬(similar t1 t2) := by sorry

-- Theorems stating that the other conditions are sufficient for similarity
theorem condition_A_sufficient (t1 t2 : Triangle) :
  condition_A t1 t2 → similar t1 t2 := by sorry

theorem condition_B_sufficient (t1 t2 : Triangle) :
  condition_B t1 t2 → similar t1 t2 := by sorry

theorem condition_C_sufficient (t1 t2 : Triangle) :
  condition_C t1 t2 → similar t1 t2 := by sorry

end NUMINAMATH_CALUDE_condition_D_not_sufficient_condition_A_sufficient_condition_B_sufficient_condition_C_sufficient_l3879_387908


namespace NUMINAMATH_CALUDE_total_apples_calculation_total_apples_is_210_l3879_387946

/-- The number of apples bought by two men and three women -/
def total_apples : ℕ := by sorry

/-- The number of men -/
def num_men : ℕ := 2

/-- The number of women -/
def num_women : ℕ := 3

/-- The number of apples bought by each man -/
def apples_per_man : ℕ := 30

/-- The additional number of apples bought by each woman compared to each man -/
def additional_apples_per_woman : ℕ := 20

/-- The number of apples bought by each woman -/
def apples_per_woman : ℕ := apples_per_man + additional_apples_per_woman

theorem total_apples_calculation :
  total_apples = num_men * apples_per_man + num_women * apples_per_woman :=
by sorry

theorem total_apples_is_210 : total_apples = 210 := by sorry

end NUMINAMATH_CALUDE_total_apples_calculation_total_apples_is_210_l3879_387946


namespace NUMINAMATH_CALUDE_clock_strike_theorem_l3879_387960

/-- Represents the number of seconds it takes for a clock to strike a given number of times. -/
def strike_time (num_strikes : ℕ) (seconds : ℝ) : Prop :=
  num_strikes > 0 ∧ seconds > 0 ∧ 
  (seconds / (num_strikes - 1 : ℝ)) = (8 : ℝ) / (5 - 1 : ℝ)

/-- Theorem stating that if a clock takes 8 seconds to strike 5 times, 
    it will take 18 seconds to strike 10 times. -/
theorem clock_strike_theorem :
  strike_time 5 8 → strike_time 10 18 :=
by
  sorry

end NUMINAMATH_CALUDE_clock_strike_theorem_l3879_387960


namespace NUMINAMATH_CALUDE_complex_equation_modulus_l3879_387950

theorem complex_equation_modulus : ∀ (x y : ℝ), 
  (Complex.I : ℂ) * x + 2 * (Complex.I : ℂ) * x = (2 : ℂ) + (Complex.I : ℂ) * y → 
  Complex.abs (x + (Complex.I : ℂ) * y) = 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_modulus_l3879_387950


namespace NUMINAMATH_CALUDE_total_students_is_90_l3879_387934

/-- Represents a class with its exam statistics -/
structure ClassStats where
  totalStudents : ℕ
  averageMark : ℚ
  excludedStudents : ℕ
  excludedAverage : ℚ
  newAverage : ℚ

/-- Calculate the total number of students across all classes -/
def totalStudents (classA classB classC : ClassStats) : ℕ :=
  classA.totalStudents + classB.totalStudents + classC.totalStudents

/-- Theorem stating that the total number of students is 90 -/
theorem total_students_is_90 (classA classB classC : ClassStats)
  (hA : classA.averageMark = 80 ∧ classA.excludedStudents = 5 ∧
        classA.excludedAverage = 20 ∧ classA.newAverage = 92)
  (hB : classB.averageMark = 75 ∧ classB.excludedStudents = 6 ∧
        classB.excludedAverage = 25 ∧ classB.newAverage = 85)
  (hC : classC.averageMark = 70 ∧ classC.excludedStudents = 4 ∧
        classC.excludedAverage = 30 ∧ classC.newAverage = 78) :
  totalStudents classA classB classC = 90 := by
  sorry


end NUMINAMATH_CALUDE_total_students_is_90_l3879_387934


namespace NUMINAMATH_CALUDE_ellipse_outside_circle_l3879_387911

theorem ellipse_outside_circle (b : ℝ) (m : ℝ) (x y : ℝ) 
  (h_b : b > 0) (h_m : -1 < m ∧ m < 1) 
  (h_ellipse : x^2 / (b^2 + 1) + y^2 / b^2 = 1) :
  (x - m)^2 + y^2 ≥ 1 - m^2 := by sorry

end NUMINAMATH_CALUDE_ellipse_outside_circle_l3879_387911


namespace NUMINAMATH_CALUDE_min_value_sum_of_fractions_l3879_387987

theorem min_value_sum_of_fractions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 3) : 
  (4 / x) + (9 / y) + (16 / z) ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_fractions_l3879_387987


namespace NUMINAMATH_CALUDE_tan_geq_one_range_l3879_387978

open Set
open Real

theorem tan_geq_one_range (f : ℝ → ℝ) (h : ∀ x ∈ Ioo (-π/2) (π/2), f x = tan x) :
  {x ∈ Ioo (-π/2) (π/2) | f x ≥ 1} = Ico (π/4) (π/2) := by
  sorry

end NUMINAMATH_CALUDE_tan_geq_one_range_l3879_387978


namespace NUMINAMATH_CALUDE_window_screen_sales_l3879_387948

/-- Represents the monthly sales of window screens -/
structure MonthlySales where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ

/-- Calculates the total sales from January to April -/
def totalSales (sales : MonthlySales) : ℕ :=
  sales.january + sales.february + sales.march + sales.april

/-- Theorem stating the total sales given the conditions -/
theorem window_screen_sales : ∃ (sales : MonthlySales),
  sales.february = 2 * sales.january ∧
  sales.march = (5 / 4 : ℚ) * sales.february ∧
  sales.april = (9 / 10 : ℚ) * sales.march ∧
  sales.march = 12100 ∧
  totalSales sales = 37510 := by
  sorry


end NUMINAMATH_CALUDE_window_screen_sales_l3879_387948


namespace NUMINAMATH_CALUDE_probability_three_consecutive_beliy_naliv_l3879_387949

/-- The probability of selecting 3 "Beliy Naliv" bushes consecutively -/
def probability_three_consecutive (beliy_naliv : ℕ) (verlioka : ℕ) : ℚ :=
  (beliy_naliv / (beliy_naliv + verlioka)) *
  ((beliy_naliv - 1) / (beliy_naliv + verlioka - 1)) *
  ((beliy_naliv - 2) / (beliy_naliv + verlioka - 2))

/-- Theorem stating the probability of selecting 3 "Beliy Naliv" bushes consecutively is 1/8 -/
theorem probability_three_consecutive_beliy_naliv :
  probability_three_consecutive 9 7 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_consecutive_beliy_naliv_l3879_387949


namespace NUMINAMATH_CALUDE_sum_equals_three_halves_l3879_387939

theorem sum_equals_three_halves : 
  let original_sum := (1:ℚ)/3 + 1/5 + 1/7 + 1/9 + 1/11 + 1/13
  let reduced_sum := (1:ℚ)/3 + 1/7 + 1/9 + 1/11
  reduced_sum = 3/2 := by sorry

end NUMINAMATH_CALUDE_sum_equals_three_halves_l3879_387939


namespace NUMINAMATH_CALUDE_average_income_of_A_and_B_l3879_387937

/-- Given the average monthly incomes of different pairs of people and the income of one person,
    prove that the average monthly income of A and B is 5050. -/
theorem average_income_of_A_and_B (A B C : ℕ) : 
  A = 4000 →
  (B + C) / 2 = 6250 →
  (A + C) / 2 = 5200 →
  (A + B) / 2 = 5050 := by
  sorry


end NUMINAMATH_CALUDE_average_income_of_A_and_B_l3879_387937


namespace NUMINAMATH_CALUDE_intersection_condition_l3879_387992

/-- Set M in R^2 -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

/-- Set N in R^2 parameterized by r -/
def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

/-- The theorem stating the condition for M ∩ N = N -/
theorem intersection_condition (r : ℝ) : 
  (M ∩ N r = N r) ↔ (0 < r ∧ r ≤ 2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3879_387992


namespace NUMINAMATH_CALUDE_lung_cancer_probability_l3879_387985

theorem lung_cancer_probability (overall_prob : ℝ) (smoker_ratio : ℝ) (smoker_prob : ℝ) :
  overall_prob = 0.001 →
  smoker_ratio = 0.2 →
  smoker_prob = 0.004 →
  ∃ (nonsmoker_prob : ℝ),
    nonsmoker_prob = 0.00025 ∧
    overall_prob = smoker_ratio * smoker_prob + (1 - smoker_ratio) * nonsmoker_prob :=
by sorry

end NUMINAMATH_CALUDE_lung_cancer_probability_l3879_387985


namespace NUMINAMATH_CALUDE_percentage_decrease_l3879_387936

theorem percentage_decrease (original : ℝ) (increase_percent : ℝ) (difference : ℝ) :
  original = 80 →
  increase_percent = 12.5 →
  difference = 30 →
  let increased_value := original * (1 + increase_percent / 100)
  let decrease_percent := (increased_value - original - difference) / original * 100
  decrease_percent = 25 := by sorry

end NUMINAMATH_CALUDE_percentage_decrease_l3879_387936


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_has_20_sides_l3879_387971

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_has_20_sides :
  ∀ n : ℕ,
  n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_has_20_sides_l3879_387971


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l3879_387947

def f (x : ℝ) : ℝ := 2 * x^3

theorem tangent_slope_at_point_one (x : ℝ) :
  HasDerivAt f 6 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l3879_387947


namespace NUMINAMATH_CALUDE_equation_solution_l3879_387996

theorem equation_solution : ∃ x : ℝ, 45 - (x - (37 - (15 - 15))) = 54 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3879_387996


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3879_387915

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 11 = -26 →
  arithmetic_sequence a₁ d 51 = 54 →
  (arithmetic_sequence a₁ d 14 = -20) ∧
  (∀ n : ℕ, n < 25 → arithmetic_sequence a₁ d n ≤ 0) ∧
  (arithmetic_sequence a₁ d 25 > 0) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3879_387915


namespace NUMINAMATH_CALUDE_dans_money_was_three_l3879_387980

/-- Dan's initial amount of money, given he bought a candy bar and has some money left -/
def dans_initial_money (candy_bar_cost : ℝ) (money_left : ℝ) : ℝ :=
  candy_bar_cost + money_left

/-- Theorem stating Dan's initial money was $3 -/
theorem dans_money_was_three :
  dans_initial_money 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_money_was_three_l3879_387980


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l3879_387969

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 6)
  (h₃ : contact_distance = 26) :
  let center_distance := Real.sqrt ((contact_distance ^ 2) + ((r₁ - r₂) ^ 2))
  center_distance = 2 * Real.sqrt 173 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l3879_387969


namespace NUMINAMATH_CALUDE_power_multiplication_l3879_387932

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l3879_387932


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3879_387905

theorem arithmetic_expression_equality : 15 - 14 * 3 + 11 / 2 - 9 * 4 + 18 = -39.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3879_387905


namespace NUMINAMATH_CALUDE_range_of_a_l3879_387945

theorem range_of_a (a b : ℝ) 
  (h1 : 0 ≤ a - b ∧ a - b ≤ 1) 
  (h2 : 1 ≤ a + b ∧ a + b ≤ 4) : 
  1/2 ≤ a ∧ a ≤ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3879_387945


namespace NUMINAMATH_CALUDE_distance_to_line_l3879_387970

/-- The smallest distance from (0, 0) to the line y = 4/3 * x - 100 -/
def smallest_distance : ℝ := 60

/-- The equation of the line in the form Ax + By + C = 0 -/
def line_equation (x y : ℝ) : Prop := -4 * x + 3 * y + 300 = 0

/-- The point from which we're measuring the distance -/
def origin : ℝ × ℝ := (0, 0)

theorem distance_to_line :
  smallest_distance = 
    (‖-4 * origin.1 + 3 * origin.2 + 300‖ : ℝ) / Real.sqrt ((-4)^2 + 3^2) :=
sorry

end NUMINAMATH_CALUDE_distance_to_line_l3879_387970


namespace NUMINAMATH_CALUDE_min_value_exponential_quadratic_min_value_achieved_at_zero_l3879_387910

theorem min_value_exponential_quadratic (x : ℝ) : 16^x - 2^x + x^2 + 1 ≥ 1 :=
by
  sorry

theorem min_value_achieved_at_zero : 16^0 - 2^0 + 0^2 + 1 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_quadratic_min_value_achieved_at_zero_l3879_387910


namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l3879_387900

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48)
  (h2 : b + d = 6) : 
  a + c = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l3879_387900


namespace NUMINAMATH_CALUDE_P_equals_Q_l3879_387975

-- Define a one-to-one, strictly increasing function f: R → R
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_injective : Function.Injective f
axiom f_strictly_increasing : ∀ x y, x < y → f x < f y

-- Define the sets P and Q
def P : Set ℝ := {x | x > f x}
def Q : Set ℝ := {x | x > f (f x)}

-- State the theorem
theorem P_equals_Q : P = Q := by sorry

end NUMINAMATH_CALUDE_P_equals_Q_l3879_387975


namespace NUMINAMATH_CALUDE_opposite_of_neg_2020_l3879_387929

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- Theorem stating that the opposite of -2020 is 2020. -/
theorem opposite_of_neg_2020 : opposite (-2020) = 2020 := by sorry

end NUMINAMATH_CALUDE_opposite_of_neg_2020_l3879_387929


namespace NUMINAMATH_CALUDE_number_problem_l3879_387995

theorem number_problem (x : ℝ) : 42 + 3 * x - 10 = 65 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3879_387995


namespace NUMINAMATH_CALUDE_triangle_angle_leq_60_l3879_387955

/-- Theorem: In any triangle, at least one angle is less than or equal to 60 degrees. -/
theorem triangle_angle_leq_60 (A B C : ℝ) (h_triangle : A + B + C = 180) :
  A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_leq_60_l3879_387955


namespace NUMINAMATH_CALUDE_max_sum_of_product_107_l3879_387933

theorem max_sum_of_product_107 (a b : ℤ) (h : a * b = 107) :
  ∃ (c d : ℤ), c * d = 107 ∧ c + d ≥ a + b ∧ c + d = 108 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_product_107_l3879_387933


namespace NUMINAMATH_CALUDE_g_increasing_iff_a_in_range_l3879_387921

-- Define the piecewise function g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ -1 then -a / (x - 1) else (3 - 3*a) * x + 1

-- State the theorem
theorem g_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → g a x < g a y) ↔ (4/5 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_g_increasing_iff_a_in_range_l3879_387921


namespace NUMINAMATH_CALUDE_log_equation_solution_l3879_387923

theorem log_equation_solution (x : ℝ) : Real.log (729 : ℝ) / Real.log (3 * x) = x → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3879_387923


namespace NUMINAMATH_CALUDE_only_eleven_not_sum_of_two_primes_l3879_387979

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

def numbers_to_check : List ℕ := [5, 7, 9, 10, 11]

theorem only_eleven_not_sum_of_two_primes :
  ∀ n ∈ numbers_to_check, n ≠ 11 → is_sum_of_two_primes n ∧
  ¬(is_sum_of_two_primes 11) :=
sorry

end NUMINAMATH_CALUDE_only_eleven_not_sum_of_two_primes_l3879_387979


namespace NUMINAMATH_CALUDE_max_k_value_l3879_387982

open Real

noncomputable def f (x : ℝ) : ℝ := x * (1 + log x)

theorem max_k_value (k : ℤ) :
  (∀ x > 2, k * (x - 2) < f x) → k ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l3879_387982


namespace NUMINAMATH_CALUDE_reflection_result_l3879_387964

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 2)
  let reflected := (-p'.2, -p'.1)
  (reflected.1, reflected.2 + 2)

def D : ℝ × ℝ := (5, 0)

theorem reflection_result :
  let D' := reflect_x D
  let D'' := reflect_line D'
  D'' = (2, -3) := by sorry

end NUMINAMATH_CALUDE_reflection_result_l3879_387964


namespace NUMINAMATH_CALUDE_triangle_max_area_l3879_387961

theorem triangle_max_area (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  3 * a * b = 25 - c^2 →
  let angle_C := 60 * π / 180
  let area := (1 / 2) * a * b * Real.sin angle_C
  area ≤ 25 * Real.sqrt 3 / 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3879_387961


namespace NUMINAMATH_CALUDE_circle_radius_is_5_l3879_387906

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (-1, 1)

-- Define the tangent line
def TangentLine (x y : ℝ) : Prop := 3 * x - 4 * y + 7 = 0

-- State the theorem
theorem circle_radius_is_5 :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- Circle C passes through point A
    A ∈ Circle center radius ∧
    -- Circle C is tangent to the line 3x-4y+7=0 at point B
    B ∈ Circle center radius ∧
    TangentLine B.1 B.2 ∧
    -- The radius of circle C is 5
    radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_5_l3879_387906


namespace NUMINAMATH_CALUDE_regression_for_related_variables_l3879_387963

/-- A type representing a statistical variable -/
structure StatVariable where
  name : String

/-- A type representing a statistical analysis method -/
inductive AnalysisMethod
  | ErrorAnalysis
  | RegressionAnalysis
  | IndependenceTest

/-- A relation indicating that two variables are related -/
def are_related (v1 v2 : StatVariable) : Prop := sorry

/-- The correct method to analyze related variables -/
def analyze_related_variables (v1 v2 : StatVariable) : AnalysisMethod :=
  AnalysisMethod.RegressionAnalysis

/-- Theorem stating that regression analysis is the correct method for analyzing related variables -/
theorem regression_for_related_variables (height weight : StatVariable) 
    (h : are_related height weight) : 
    analyze_related_variables height weight = AnalysisMethod.RegressionAnalysis := by
  sorry

end NUMINAMATH_CALUDE_regression_for_related_variables_l3879_387963


namespace NUMINAMATH_CALUDE_united_charge_per_minute_is_correct_l3879_387935

/-- Additional charge per minute for United Telephone -/
def united_charge_per_minute : ℚ := 25 / 100

/-- Base rate for United Telephone -/
def united_base_rate : ℚ := 7

/-- Base rate for Atlantic Call -/
def atlantic_base_rate : ℚ := 12

/-- Additional charge per minute for Atlantic Call -/
def atlantic_charge_per_minute : ℚ := 1 / 5

/-- Number of minutes for which the bills are equal -/
def equal_minutes : ℕ := 100

theorem united_charge_per_minute_is_correct :
  united_base_rate + equal_minutes * united_charge_per_minute =
  atlantic_base_rate + equal_minutes * atlantic_charge_per_minute :=
by sorry

end NUMINAMATH_CALUDE_united_charge_per_minute_is_correct_l3879_387935


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l3879_387999

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![9, -3; 2, 2]
def C : Matrix (Fin 2) (Fin 2) ℝ := !![27, -9; 32, -16]

theorem matrix_multiplication_result : A * B = C := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l3879_387999


namespace NUMINAMATH_CALUDE_partner_investment_time_l3879_387951

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where p invested for 20 months, prove that q invested for 40 months. -/
theorem partner_investment_time (x : ℝ) (t : ℝ) : 
  (7 : ℝ) / 5 = 7 * x / (5 * x) →  -- investment ratio
  (7 : ℝ) / 10 = (7 * x * 20) / (5 * x * t) →  -- profit ratio
  t = 40 := by
sorry

end NUMINAMATH_CALUDE_partner_investment_time_l3879_387951


namespace NUMINAMATH_CALUDE_parabola_vertex_l3879_387925

/-- The vertex of the parabola defined by y² + 10y + 3x + 9 = 0 is (16/3, -5) -/
theorem parabola_vertex :
  let f : ℝ → ℝ → ℝ := λ x y ↦ y^2 + 10*y + 3*x + 9
  ∃! (x₀ y₀ : ℝ), (∀ x y, f x y = 0 → y ≥ y₀) ∧ f x₀ y₀ = 0 ∧ x₀ = 16/3 ∧ y₀ = -5 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3879_387925


namespace NUMINAMATH_CALUDE_enrollment_increase_l3879_387976

theorem enrollment_increase (e1991 e1992 e1993 : ℝ) 
  (h1 : e1993 = e1991 * (1 + 0.38))
  (h2 : e1993 = e1992 * (1 + 0.15)) :
  e1992 = e1991 * (1 + 0.2) := by
  sorry

end NUMINAMATH_CALUDE_enrollment_increase_l3879_387976


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3879_387943

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

theorem parallel_vectors_k_value :
  ∀ (k : ℝ),
  (∃ (c : ℝ), c ≠ 0 ∧ (k * a.1 + b.1, k * a.2 + b.2) = c • (a.1 - 2 * b.1, a.2 - 2 * b.2)) →
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3879_387943


namespace NUMINAMATH_CALUDE_moore_law_gpu_transistors_l3879_387920

def initial_year : Nat := 1992
def final_year : Nat := 2011
def initial_transistors : Nat := 500000
def doubling_period : Nat := 3

def moore_law_prediction (initial : Nat) (years : Nat) (period : Nat) : Nat :=
  initial * (2 ^ (years / period))

theorem moore_law_gpu_transistors :
  moore_law_prediction initial_transistors (final_year - initial_year) doubling_period = 32000000 := by
  sorry

end NUMINAMATH_CALUDE_moore_law_gpu_transistors_l3879_387920


namespace NUMINAMATH_CALUDE_complex_modulus_power_l3879_387912

theorem complex_modulus_power : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_power_l3879_387912


namespace NUMINAMATH_CALUDE_room_width_calculation_l3879_387962

/-- Given a room with specified length, total paving cost, and paving rate per square meter,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  length = 5.5 →
  total_cost = 28875 →
  rate_per_sqm = 1400 →
  (total_cost / rate_per_sqm) / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l3879_387962


namespace NUMINAMATH_CALUDE_carries_payment_l3879_387991

/-- Calculate Carrie's payment for clothes shopping --/
theorem carries_payment (shirt_quantity : ℕ) (pants_quantity : ℕ) (jacket_quantity : ℕ) 
  (skirt_quantity : ℕ) (shoes_quantity : ℕ) (shirt_price : ℚ) (pants_price : ℚ) 
  (jacket_price : ℚ) (skirt_price : ℚ) (shoes_price : ℚ) (shirt_discount : ℚ) 
  (jacket_discount : ℚ) (skirt_discount : ℚ) (mom_payment_ratio : ℚ) :
  shirt_quantity = 8 →
  pants_quantity = 4 →
  jacket_quantity = 4 →
  skirt_quantity = 3 →
  shoes_quantity = 2 →
  shirt_price = 12 →
  pants_price = 25 →
  jacket_price = 75 →
  skirt_price = 30 →
  shoes_price = 50 →
  shirt_discount = 0.2 →
  jacket_discount = 0.2 →
  skirt_discount = 0.1 →
  mom_payment_ratio = 2/3 →
  let total_cost := 
    (shirt_quantity : ℚ) * shirt_price * (1 - shirt_discount) +
    (pants_quantity : ℚ) * pants_price +
    (jacket_quantity : ℚ) * jacket_price * (1 - jacket_discount) +
    (skirt_quantity : ℚ) * skirt_price * (1 - skirt_discount) +
    (shoes_quantity : ℚ) * shoes_price
  (1 - mom_payment_ratio) * total_cost = 199.27 := by
  sorry

end NUMINAMATH_CALUDE_carries_payment_l3879_387991


namespace NUMINAMATH_CALUDE_cat_food_finished_on_sunday_l3879_387984

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the number of days from Monday to a given day -/
def daysFromMonday (day : DayOfWeek) : Nat :=
  match day with
  | .Monday => 0
  | .Tuesday => 1
  | .Wednesday => 2
  | .Thursday => 3
  | .Friday => 4
  | .Saturday => 5
  | .Sunday => 6

def dailyConsumption : Rat := 3/5
def initialCans : Nat := 8

theorem cat_food_finished_on_sunday :
  ∃ (day : DayOfWeek),
    (daysFromMonday day + 1) * dailyConsumption > initialCans ∧
    (daysFromMonday day) * dailyConsumption ≤ initialCans ∧
    day = DayOfWeek.Sunday := by
  sorry


end NUMINAMATH_CALUDE_cat_food_finished_on_sunday_l3879_387984


namespace NUMINAMATH_CALUDE_isosceles_triangle_most_stable_isosceles_triangle_stable_other_shapes_not_stable_l3879_387990

/-- Represents a geometric shape -/
inductive Shape
  | IsoscelesTriangle
  | Rectangle
  | Square
  | Parallelogram

/-- Stability measure of a shape -/
def stability (s : Shape) : ℕ :=
  match s with
  | Shape.IsoscelesTriangle => 3
  | Shape.Rectangle => 2
  | Shape.Square => 2
  | Shape.Parallelogram => 1

/-- A shape is considered stable if its stability measure is greater than 2 -/
def is_stable (s : Shape) : Prop := stability s > 2

theorem isosceles_triangle_most_stable :
  ∀ s : Shape, s ≠ Shape.IsoscelesTriangle → stability Shape.IsoscelesTriangle > stability s :=
by sorry

theorem isosceles_triangle_stable :
  is_stable Shape.IsoscelesTriangle :=
by sorry

theorem other_shapes_not_stable :
  ¬ is_stable Shape.Rectangle ∧
  ¬ is_stable Shape.Square ∧
  ¬ is_stable Shape.Parallelogram :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_most_stable_isosceles_triangle_stable_other_shapes_not_stable_l3879_387990


namespace NUMINAMATH_CALUDE_percentage_equals_1000_l3879_387904

theorem percentage_equals_1000 (x : ℝ) (p : ℝ) : 
  (p / 100) * x = 1000 → 
  (120 / 100) * x = 6000 → 
  p = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_equals_1000_l3879_387904
