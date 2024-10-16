import Mathlib

namespace NUMINAMATH_CALUDE_unique_triples_l3572_357268

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_triples : 
  ∀ a b c : ℕ, 
    (is_prime (a^2 - 23)) → 
    (is_prime (b^2 - 23)) → 
    ((a^2 - 23) * (b^2 - 23) = c^2 - 23) → 
    ((a = 5 ∧ b = 6 ∧ c = 7) ∨ (a = 6 ∧ b = 5 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_unique_triples_l3572_357268


namespace NUMINAMATH_CALUDE_vector_equation_result_l3572_357280

def a : ℝ × ℝ := (2, -3)
def b : ℝ × ℝ := (1, 2)
def c : ℝ × ℝ := (9, 4)

theorem vector_equation_result (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : c = (m * a.1 + n * b.1, m * a.2 + n * b.2)) : 
  1/m + 1/n = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_result_l3572_357280


namespace NUMINAMATH_CALUDE_angle_covered_in_three_layers_l3572_357297

/-- Given a 90-degree angle covered by some angles with the same vertex in two or three layers,
    if the sum of the angles is 290 degrees, then the measure of the angle covered in three layers is 20 degrees. -/
theorem angle_covered_in_three_layers 
  (total_angle : ℝ) 
  (sum_of_angles : ℝ) 
  (angle_covered_three_layers : ℝ) 
  (angle_covered_two_layers : ℝ) 
  (h1 : total_angle = 90)
  (h2 : sum_of_angles = 290)
  (h3 : angle_covered_three_layers + angle_covered_two_layers = total_angle)
  (h4 : 3 * angle_covered_three_layers + 2 * angle_covered_two_layers = sum_of_angles) :
  angle_covered_three_layers = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_covered_in_three_layers_l3572_357297


namespace NUMINAMATH_CALUDE_stating_mans_speed_with_current_l3572_357232

/-- 
Given a man's speed against a current and the speed of the current,
this function calculates the man's speed with the current.
-/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- 
Theorem stating that given the specific conditions in the problem,
the man's speed with the current is 20 kmph.
-/
theorem mans_speed_with_current : 
  speed_with_current 14 3 = 20 := by
  sorry

#eval speed_with_current 14 3

end NUMINAMATH_CALUDE_stating_mans_speed_with_current_l3572_357232


namespace NUMINAMATH_CALUDE_min_k_value_l3572_357210

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- AB is the length of side AB
  AB : ℝ
  -- h is the height of the trapezoid
  h : ℝ
  -- E and F are midpoints of AD and AB respectively
  -- CD = 2AB (implied by the structure)

/-- The area difference between triangle CDG and quadrilateral AEGF -/
def areaDifference (t : Trapezoid) : ℝ :=
  2 * t.AB * t.h - t.AB * t.h

/-- The area of the trapezoid ABCD -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  3 * t.AB * t.h

/-- Main theorem: The minimum value of k is 8 -/
theorem min_k_value (t : Trapezoid) (k : ℕ+) 
    (h1 : areaDifference t = k / 24)
    (h2 : ∃ n : ℕ, trapezoidArea t = n) : 
  k ≥ 8 ∧ ∃ (t : Trapezoid) (k : ℕ+), k = 8 ∧ areaDifference t = k / 24 ∧ ∃ (n : ℕ), trapezoidArea t = n :=
by
  sorry

end NUMINAMATH_CALUDE_min_k_value_l3572_357210


namespace NUMINAMATH_CALUDE_monthly_income_a_l3572_357243

/-- Proves that given the average incomes of (A,B), (B,C), and (A,C), the monthly income of A is Rs. 3000 -/
theorem monthly_income_a (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 4050)
  (avg_bc : (b + c) / 2 = 5250)
  (avg_ac : (a + c) / 2 = 4200) :
  a = 3000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_a_l3572_357243


namespace NUMINAMATH_CALUDE_difference_even_plus_five_minus_odd_l3572_357221

/-- Sum of the first n odd counting numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * (2 * n - 1)

/-- Sum of the first n even counting numbers plus 5 added to each number -/
def sumEvenNumbersPlusFive (n : ℕ) : ℕ := n * (2 * n + 5)

/-- The difference between the sum of the first 3000 even counting numbers plus 5 
    added to each number and the sum of the first 3000 odd counting numbers is 18000 -/
theorem difference_even_plus_five_minus_odd : 
  sumEvenNumbersPlusFive 3000 - sumOddNumbers 3000 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_difference_even_plus_five_minus_odd_l3572_357221


namespace NUMINAMATH_CALUDE_full_bucket_weight_formula_l3572_357278

/-- Represents the weight of a bucket with water -/
structure BucketWeight where
  twoThirdsFull : ℝ  -- Weight when 2/3 full
  halfFull : ℝ       -- Weight when 1/2 full

/-- Calculates the weight of a bucket when it's full of water -/
def fullBucketWeight (bw : BucketWeight) : ℝ :=
  3 * bw.twoThirdsFull - 2 * bw.halfFull

/-- Theorem stating that the weight of a full bucket is 3a - 2b given the weights at 2/3 and 1/2 full -/
theorem full_bucket_weight_formula (bw : BucketWeight) :
  fullBucketWeight bw = 3 * bw.twoThirdsFull - 2 * bw.halfFull := by
  sorry

end NUMINAMATH_CALUDE_full_bucket_weight_formula_l3572_357278


namespace NUMINAMATH_CALUDE_initial_bottle_caps_l3572_357245

theorem initial_bottle_caps (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 7 → total = 14 → total = initial + added → initial = 7 := by sorry

end NUMINAMATH_CALUDE_initial_bottle_caps_l3572_357245


namespace NUMINAMATH_CALUDE_last_term_is_1344_l3572_357236

/-- Defines the nth term of the sequence -/
def sequenceTerm (n : ℕ) : ℕ :=
  if n % 3 = 1 then (n + 2) / 3 else (n + 1) / 3

/-- The last term of the sequence with 2015 elements -/
def lastTerm : ℕ := sequenceTerm 2015

theorem last_term_is_1344 : lastTerm = 1344 := by
  sorry

end NUMINAMATH_CALUDE_last_term_is_1344_l3572_357236


namespace NUMINAMATH_CALUDE_profit_achieved_min_disks_optimal_l3572_357281

/-- The number of disks Maria buys for $6 -/
def buy_rate : ℕ := 5

/-- The price Maria pays for buy_rate disks -/
def buy_price : ℚ := 6

/-- The number of disks Maria sells for $7 -/
def sell_rate : ℕ := 4

/-- The price Maria receives for sell_rate disks -/
def sell_price : ℚ := 7

/-- The target profit Maria wants to achieve -/
def target_profit : ℚ := 120

/-- The minimum number of disks Maria must sell to make the target profit -/
def min_disks_to_sell : ℕ := 219

theorem profit_achieved (n : ℕ) : 
  n ≥ min_disks_to_sell → 
  n * (sell_price / sell_rate - buy_price / buy_rate) ≥ target_profit :=
by sorry

theorem min_disks_optimal : 
  ∀ m : ℕ, m < min_disks_to_sell → 
  m * (sell_price / sell_rate - buy_price / buy_rate) < target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_achieved_min_disks_optimal_l3572_357281


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3572_357294

-- Define the cyclic sum function
def cyclicSum (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c + f b c a + f c a b

-- State the theorem
theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  cyclicSum (fun x y z => (y + z - x)^2 / (x^2 + (y + z)^2)) a b c ≥ 3/5 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3572_357294


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l3572_357234

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
theorem mans_speed_against_current
  (speed_with_current : ℝ)
  (current_speed : ℝ)
  (h1 : speed_with_current = 12)
  (h2 : current_speed = 2) :
  speed_with_current - 2 * current_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l3572_357234


namespace NUMINAMATH_CALUDE_angle_ABF_measure_l3572_357217

/-- A regular octagon is a polygon with 8 sides of equal length and 8 equal angles -/
structure RegularOctagon where
  vertices : Fin 8 → Point

/-- The measure of an angle in a regular octagon -/
def regular_octagon_angle : ℝ := 135

/-- The measure of angle ABF in a regular octagon -/
def angle_ABF (octagon : RegularOctagon) : ℝ := 22.5

theorem angle_ABF_measure (octagon : RegularOctagon) :
  angle_ABF octagon = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABF_measure_l3572_357217


namespace NUMINAMATH_CALUDE_equation_solution_l3572_357285

theorem equation_solution : ∀ (x : ℝ) (number : ℝ),
  x = 4 →
  7 * (x - 1) = number →
  number = 21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3572_357285


namespace NUMINAMATH_CALUDE_trees_in_column_l3572_357292

/-- Proves the number of trees in one column of Jack's grove --/
theorem trees_in_column (trees_per_row : ℕ) (cleaning_time_per_tree : ℕ) (total_cleaning_time : ℕ) 
  (h1 : trees_per_row = 4)
  (h2 : cleaning_time_per_tree = 3)
  (h3 : total_cleaning_time = 60)
  (h4 : total_cleaning_time / cleaning_time_per_tree = trees_per_row * (total_cleaning_time / cleaning_time_per_tree / trees_per_row)) :
  total_cleaning_time / cleaning_time_per_tree / trees_per_row = 5 := by
  sorry

#check trees_in_column

end NUMINAMATH_CALUDE_trees_in_column_l3572_357292


namespace NUMINAMATH_CALUDE_max_abs_cexp_minus_two_l3572_357273

-- Define the complex exponential function
noncomputable def cexp (x : ℝ) : ℂ := Complex.exp (Complex.I * x)

-- State Euler's formula
axiom euler_formula (x : ℝ) : cexp x = Complex.cos x + Complex.I * Complex.sin x

-- State the theorem
theorem max_abs_cexp_minus_two :
  ∃ (M : ℝ), M = 3 ∧ ∀ (x : ℝ), Complex.abs (cexp x - 2) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_abs_cexp_minus_two_l3572_357273


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3572_357277

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^3 - 18 * x^2 + 24 * x - 26) % (4 * x - 8) = 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3572_357277


namespace NUMINAMATH_CALUDE_park_rose_bushes_l3572_357296

/-- Calculate the final number of rose bushes in the park -/
def final_rose_bushes (initial : ℕ) (planned : ℕ) (rate : ℕ) (removed : ℕ) : ℕ :=
  initial + planned * rate - removed

/-- Theorem stating the final number of rose bushes in the park -/
theorem park_rose_bushes : final_rose_bushes 2 4 3 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_park_rose_bushes_l3572_357296


namespace NUMINAMATH_CALUDE_base5_1204_eq_179_l3572_357204

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 5^3 + d₂ * 5^2 + d₁ * 5^1 + d₀ * 5^0

/-- Proves that 1204₍₅₎ is equal to 179 in decimal --/
theorem base5_1204_eq_179 : base5ToDecimal 1 2 0 4 = 179 := by
  sorry

end NUMINAMATH_CALUDE_base5_1204_eq_179_l3572_357204


namespace NUMINAMATH_CALUDE_jung_min_wire_purchase_l3572_357252

/-- The length of wire needed to make a regular pentagon with given side length -/
def pentagonWireLength (sideLength : ℝ) : ℝ := 5 * sideLength

/-- The total length of wire bought given the side length of the pentagon and the leftover wire -/
def totalWireBought (sideLength leftover : ℝ) : ℝ := pentagonWireLength sideLength + leftover

theorem jung_min_wire_purchase :
  totalWireBought 13 8 = 73 := by
  sorry

end NUMINAMATH_CALUDE_jung_min_wire_purchase_l3572_357252


namespace NUMINAMATH_CALUDE_intersection_reciprocal_sum_l3572_357264

/-- Given a line intersecting y = x^2 at (x₁, x₁²) and (x₂, x₂²), and the x-axis at (x₃, 0), 
    prove that 1/x₁ + 1/x₂ = 1/x₃ -/
theorem intersection_reciprocal_sum (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0) (h₃ : x₃ ≠ 0)
  (line_eq : ∃ (k m : ℝ), ∀ x y, y = k * x + m ↔ (x = x₁ ∧ y = x₁^2) ∨ (x = x₂ ∧ y = x₂^2) ∨ (x = x₃ ∧ y = 0)) :
  1 / x₁ + 1 / x₂ = 1 / x₃ := by
  sorry

end NUMINAMATH_CALUDE_intersection_reciprocal_sum_l3572_357264


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3572_357299

theorem quadratic_root_problem (m : ℝ) : 
  (3 * (1 : ℝ)^2 + m * 1 - 7 = 0) → 
  (∃ x : ℝ, x ≠ 1 ∧ 3 * x^2 + m * x - 7 = 0 ∧ x = -7/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3572_357299


namespace NUMINAMATH_CALUDE_fraction_problem_l3572_357288

theorem fraction_problem :
  ∃ (n d : ℚ), n + d = 5.25 ∧ (n + 3) / (2 * d) = 1/3 ∧ n/d = 2/33 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3572_357288


namespace NUMINAMATH_CALUDE_smallest_division_is_six_l3572_357209

/-- A typical rectangular parallelepiped has all dimensions different -/
structure TypicalParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  all_different : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- A cube has all sides equal -/
structure Cube where
  side : ℝ

/-- The division of a cube into typical parallelepipeds -/
def CubeDivision (c : Cube) := List TypicalParallelepiped

/-- Predicate to check if a division is valid (i.e., the parallelepipeds fill the cube exactly) -/
def IsValidDivision (c : Cube) (d : CubeDivision c) : Prop := sorry

/-- The smallest number of typical parallelepipeds into which a cube can be divided is 6 -/
theorem smallest_division_is_six (c : Cube) : 
  (∃ (d : CubeDivision c), IsValidDivision c d ∧ d.length = 6) ∧
  (∀ (d : CubeDivision c), IsValidDivision c d → d.length ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_smallest_division_is_six_l3572_357209


namespace NUMINAMATH_CALUDE_f_monotonic_increase_l3572_357244

def f (x : ℝ) : ℝ := x * |x| - 2 * x

theorem f_monotonic_increase :
  ∀ (a b : ℝ), (a < b ∧ ((a < -1 ∧ b ≤ -1) ∨ (a ≥ 1 ∧ b > 1))) →
  ∀ (x y : ℝ), a ≤ x ∧ x < y ∧ y ≤ b → f x < f y :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_increase_l3572_357244


namespace NUMINAMATH_CALUDE_smallest_perimeter_is_30_l3572_357258

/-- Represents the side lengths of squares in the rectangle --/
structure SquareSides where
  a : ℕ  -- side length of two squares
  b : ℕ  -- side length of one square

/-- Represents the dimensions of the rectangle --/
structure RectangleDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : RectangleDimensions) : ℕ :=
  2 * (r.width + r.height)

/-- Checks if the given square sides form a valid configuration --/
def isValidConfiguration (s : SquareSides) : Prop :=
  s.b = 3 * s.a

/-- Calculates the rectangle dimensions from square sides --/
def calculateRectangleDimensions (s : SquareSides) : RectangleDimensions :=
  { width := 3 * s.a + s.b
  , height := 12 * s.a - s.b }

theorem smallest_perimeter_is_30 :
  ∀ s : SquareSides,
    s.a > 0 →
    isValidConfiguration s →
    perimeter (calculateRectangleDimensions s) ≥ 30 ∧
    (∃ s' : SquareSides, 
      s'.a > 0 ∧ 
      isValidConfiguration s' ∧ 
      perimeter (calculateRectangleDimensions s') = 30) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_is_30_l3572_357258


namespace NUMINAMATH_CALUDE_T_100_value_l3572_357256

/-- The original sequence a_n -/
def a (n : ℕ) : ℕ := 2^(n-1)

/-- The number of inserted terms between a_k and a_{k+1} -/
def inserted_count (k : ℕ) : ℕ := k

/-- The value of inserted terms between a_k and a_{k+1} -/
def inserted_value (k : ℕ) : ℤ := (-1)^k * k

/-- The sum of the first n terms of the new sequence b_n -/
noncomputable def T (n : ℕ) : ℤ := sorry

/-- The theorem to prove -/
theorem T_100_value : T 100 = 8152 := by sorry

end NUMINAMATH_CALUDE_T_100_value_l3572_357256


namespace NUMINAMATH_CALUDE_calculation_proof_l3572_357254

theorem calculation_proof : (2.5 - 0.3) * 0.25 = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3572_357254


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3572_357207

theorem geometric_series_sum (y : ℚ) : y = 23 / 13 ↔ 
  (∑' n, (1 / 3 : ℚ) ^ n) + (∑' n, (-1/4 : ℚ) ^ n) = ∑' n, (1 / y : ℚ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3572_357207


namespace NUMINAMATH_CALUDE_surface_area_of_specific_block_l3572_357289

/-- Represents a rectangular solid block made of unit cubes -/
structure RectangularBlock where
  length : Nat
  width : Nat
  height : Nat
  total_cubes : Nat

/-- Calculates the surface area of a rectangular block -/
def surface_area (block : RectangularBlock) : Nat :=
  2 * (block.length * block.width + block.length * block.height + block.width * block.height)

/-- Theorem stating that the surface area of the specific block is 66 square units -/
theorem surface_area_of_specific_block :
  ∃ (block : RectangularBlock),
    block.length = 5 ∧
    block.width = 3 ∧
    block.height = 1 ∧
    block.total_cubes = 15 ∧
    surface_area block = 66 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_specific_block_l3572_357289


namespace NUMINAMATH_CALUDE_net_change_is_correct_l3572_357240

/-- Calculates the final price after applying two percentage changes -/
def apply_price_changes (original_price : ℚ) (change1 : ℚ) (change2 : ℚ) : ℚ :=
  original_price * (1 + change1) * (1 + change2)

/-- Represents the store inventory with original prices and price changes -/
structure Inventory where
  tv_price : ℚ
  tv_change1 : ℚ
  tv_change2 : ℚ
  fridge_price : ℚ
  fridge_change1 : ℚ
  fridge_change2 : ℚ
  washer_price : ℚ
  washer_change1 : ℚ
  washer_change2 : ℚ

/-- Calculates the net change in total prices -/
def net_change (inv : Inventory) : ℚ :=
  let final_tv_price := apply_price_changes inv.tv_price inv.tv_change1 inv.tv_change2
  let final_fridge_price := apply_price_changes inv.fridge_price inv.fridge_change1 inv.fridge_change2
  let final_washer_price := apply_price_changes inv.washer_price inv.washer_change1 inv.washer_change2
  let total_final_price := final_tv_price + final_fridge_price + final_washer_price
  let total_original_price := inv.tv_price + inv.fridge_price + inv.washer_price
  total_final_price - total_original_price

theorem net_change_is_correct (inv : Inventory) : 
  inv.tv_price = 500 ∧ 
  inv.tv_change1 = -1/5 ∧ 
  inv.tv_change2 = 9/20 ∧
  inv.fridge_price = 1000 ∧ 
  inv.fridge_change1 = 7/20 ∧ 
  inv.fridge_change2 = -3/20 ∧
  inv.washer_price = 750 ∧ 
  inv.washer_change1 = 1/10 ∧ 
  inv.washer_change2 = -1/5 
  → net_change inv = 275/2 := by
  sorry

#eval net_change { 
  tv_price := 500, tv_change1 := -1/5, tv_change2 := 9/20,
  fridge_price := 1000, fridge_change1 := 7/20, fridge_change2 := -3/20,
  washer_price := 750, washer_change1 := 1/10, washer_change2 := -1/5
}

end NUMINAMATH_CALUDE_net_change_is_correct_l3572_357240


namespace NUMINAMATH_CALUDE_weaving_productivity_l3572_357206

/-- Represents the daily increase in fabric production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the number of days -/
def days : ℕ := 30

/-- Represents the initial daily production -/
def initial_production : ℚ := 5

/-- Represents the total production over the given period -/
def total_production : ℚ := 390

/-- Theorem stating the relationship between the daily increase and total production -/
theorem weaving_productivity :
  days * initial_production + (days * (days - 1) / 2) * daily_increase = total_production :=
sorry

end NUMINAMATH_CALUDE_weaving_productivity_l3572_357206


namespace NUMINAMATH_CALUDE_eighth_term_equals_general_term_l3572_357223

/-- The general term of the sequence -/
def generalTerm (n : ℕ) (a : ℝ) : ℝ := (-1)^n * n^2 * a^(n+1)

/-- The 8th term of the sequence -/
def eighthTerm (a : ℝ) : ℝ := 64 * a^9

theorem eighth_term_equals_general_term : 
  ∀ a : ℝ, generalTerm 8 a = eighthTerm a := by sorry

end NUMINAMATH_CALUDE_eighth_term_equals_general_term_l3572_357223


namespace NUMINAMATH_CALUDE_function_value_at_negative_a_l3572_357267

/-- Given a function f(x) = ax² + bx, if f(a) = 8, then f(-a) = 8 - 2ab -/
theorem function_value_at_negative_a (a b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 + b * x
  f a = 8 → f (-a) = 8 - 2 * a * b := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_a_l3572_357267


namespace NUMINAMATH_CALUDE_blue_balls_removal_l3572_357249

theorem blue_balls_removal (total : ℕ) (red_percent : ℚ) (target_red_percent : ℚ) 
  (h1 : total = 120) 
  (h2 : red_percent = 2/5) 
  (h3 : target_red_percent = 3/4) : 
  ∃ (removed : ℕ), 
    removed = 56 ∧ 
    (red_percent * total : ℚ) / (total - removed : ℚ) = target_red_percent := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_removal_l3572_357249


namespace NUMINAMATH_CALUDE_pets_count_l3572_357274

/-- The total number of pets owned by Teddy, Ben, and Dave -/
def total_pets (x y z a b c d e f : ℕ) : ℕ := x + y + z + a + b + c + d + e + f

/-- Theorem stating the total number of pets is 118 -/
theorem pets_count (x y z a b c d e f : ℕ) 
  (eq1 : x = 9)
  (eq2 : y = 8)
  (eq3 : z = 10)
  (eq4 : a = 21)
  (eq5 : b = 2 * y)
  (eq6 : c = z)
  (eq7 : d = x - 4)
  (eq8 : e = y + 13)
  (eq9 : f = 18) :
  total_pets x y z a b c d e f = 118 := by
  sorry


end NUMINAMATH_CALUDE_pets_count_l3572_357274


namespace NUMINAMATH_CALUDE_area_of_median_triangle_l3572_357200

/-- Given a triangle ABC with area S, the area of a triangle whose sides are equal to the medians of ABC is 3/4 * S -/
theorem area_of_median_triangle (A B C : ℝ × ℝ) (S : ℝ) : 
  let triangle_area := S
  let median_triangle_area := (3/4 : ℝ) * S
  triangle_area = S → median_triangle_area = (3/4 : ℝ) * triangle_area := by
sorry

end NUMINAMATH_CALUDE_area_of_median_triangle_l3572_357200


namespace NUMINAMATH_CALUDE_inequality_problem_l3572_357248

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a * c < b * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l3572_357248


namespace NUMINAMATH_CALUDE_wage_ratio_is_two_to_one_l3572_357208

/-- The ratio of a man's daily wage to a woman's daily wage -/
def wage_ratio (men_wage women_wage : ℚ) : ℚ := men_wage / women_wage

/-- The total earnings of a group of workers over a period of time -/
def total_earnings (num_workers : ℕ) (days : ℕ) (daily_wage : ℚ) : ℚ :=
  (num_workers : ℚ) * (days : ℚ) * daily_wage

theorem wage_ratio_is_two_to_one 
  (men_wage women_wage : ℚ)
  (h1 : total_earnings 16 25 men_wage = 14400)
  (h2 : total_earnings 40 30 women_wage = 21600) :
  wage_ratio men_wage women_wage = 2 := by
  sorry

#eval wage_ratio 36 18  -- Expected output: 2

end NUMINAMATH_CALUDE_wage_ratio_is_two_to_one_l3572_357208


namespace NUMINAMATH_CALUDE_new_person_weight_l3572_357286

/-- Given a group of 8 people where one person weighing 45 kg is replaced by a new person,
    and the average weight increases by 6 kg, the weight of the new person is 93 kg. -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 45 →
  avg_increase = 6 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 93 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3572_357286


namespace NUMINAMATH_CALUDE_complex_equations_solutions_l3572_357266

theorem complex_equations_solutions :
  let x₁ : ℚ := -7/5
  let y₁ : ℚ := 5
  let x₂ : ℚ := 5
  let y₂ : ℚ := -1
  (3 * y₁ : ℂ) + (5 * x₁ * I) = 15 - 7 * I ∧
  (2 * x₂ + 3 * y₂ : ℂ) + ((x₂ - y₂) * I) = 7 + 6 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_equations_solutions_l3572_357266


namespace NUMINAMATH_CALUDE_tonys_correct_score_l3572_357224

def class_size : ℕ := 20
def initial_average : ℚ := 73
def final_average : ℚ := 74
def score_increase : ℕ := 16

theorem tonys_correct_score :
  ∀ (initial_score final_score : ℕ),
  (class_size - 1 : ℚ) * initial_average + (initial_score : ℚ) / class_size = initial_average →
  (class_size - 1 : ℚ) * initial_average + (final_score : ℚ) / class_size = final_average →
  final_score = initial_score + score_increase →
  final_score = 36 := by
sorry

end NUMINAMATH_CALUDE_tonys_correct_score_l3572_357224


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l3572_357284

theorem factorization_of_2x_squared_minus_2 (x : ℝ) : 2 * x^2 - 2 = 2 * (x - 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l3572_357284


namespace NUMINAMATH_CALUDE_lecture_slides_theorem_our_lecture_slides_l3572_357265

/-- Represents a lecture with slides -/
structure Lecture where
  duration : ℕ  -- Duration of the lecture in minutes
  initial_slides : ℕ  -- Number of slides changed in the initial period
  initial_period : ℕ  -- Initial period in minutes
  total_slides : ℕ  -- Total number of slides used

/-- Calculates the total number of slides used in a lecture -/
def calculate_total_slides (l : Lecture) : ℕ :=
  (l.duration * l.initial_slides) / l.initial_period

/-- Theorem stating that for the given lecture conditions, the total slides used is 100 -/
theorem lecture_slides_theorem (l : Lecture) 
  (h1 : l.duration = 50)
  (h2 : l.initial_slides = 4)
  (h3 : l.initial_period = 2) :
  calculate_total_slides l = 100 := by
  sorry

/-- The specific lecture instance -/
def our_lecture : Lecture := {
  duration := 50,
  initial_slides := 4,
  initial_period := 2,
  total_slides := 100
}

/-- Proof that our specific lecture uses 100 slides -/
theorem our_lecture_slides : 
  calculate_total_slides our_lecture = 100 := by
  sorry

end NUMINAMATH_CALUDE_lecture_slides_theorem_our_lecture_slides_l3572_357265


namespace NUMINAMATH_CALUDE_property_rent_calculation_l3572_357291

theorem property_rent_calculation (purchase_price : ℝ) (maintenance_rate : ℝ) 
  (annual_tax : ℝ) (target_return_rate : ℝ) (monthly_rent : ℝ) : 
  purchase_price = 12000 ∧ 
  maintenance_rate = 0.15 ∧ 
  annual_tax = 400 ∧ 
  target_return_rate = 0.06 ∧ 
  monthly_rent = 109.80 →
  monthly_rent * 12 * (1 - maintenance_rate) = 
    purchase_price * target_return_rate + annual_tax :=
by
  sorry

#check property_rent_calculation

end NUMINAMATH_CALUDE_property_rent_calculation_l3572_357291


namespace NUMINAMATH_CALUDE_sine_phase_shift_specific_sine_phase_shift_l3572_357257

/-- The phase shift of a sine function y = A * sin(B * x + C) is -C/B -/
theorem sine_phase_shift (A B C : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ A * Real.sin (B * x + C)
  let phase_shift := -C / B
  ∀ x, f (x + phase_shift) = A * Real.sin (B * x)
  := by sorry

/-- The phase shift of y = 3 * sin(4x + π/4) is -π/16 -/
theorem specific_sine_phase_shift : 
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (4 * x + π/4)
  let phase_shift := -π/16
  ∀ x, f (x + phase_shift) = 3 * Real.sin (4 * x)
  := by sorry

end NUMINAMATH_CALUDE_sine_phase_shift_specific_sine_phase_shift_l3572_357257


namespace NUMINAMATH_CALUDE_sum_prime_factors_2310_l3572_357263

def prime_factors (n : ℕ) : List ℕ := sorry

theorem sum_prime_factors_2310 : (prime_factors 2310).sum = 28 := by sorry

end NUMINAMATH_CALUDE_sum_prime_factors_2310_l3572_357263


namespace NUMINAMATH_CALUDE_dawson_group_size_l3572_357202

/-- The number of people in a group given the total cost and cost per person -/
def group_size (total_cost : ℕ) (cost_per_person : ℕ) : ℕ :=
  total_cost / cost_per_person

/-- Proof that the group size is 15 given the specific costs -/
theorem dawson_group_size :
  group_size 13500 900 = 15 := by
  sorry

end NUMINAMATH_CALUDE_dawson_group_size_l3572_357202


namespace NUMINAMATH_CALUDE_graph_transformation_l3572_357218

/-- Given a function f, prove that (1/3)f(x) + 2 is equivalent to scaling f(x) vertically by 1/3 and shifting up by 2 -/
theorem graph_transformation (f : ℝ → ℝ) (x : ℝ) :
  (1/3) * (f x) + 2 = ((1/3) * f x) + 2 := by sorry

end NUMINAMATH_CALUDE_graph_transformation_l3572_357218


namespace NUMINAMATH_CALUDE_packages_per_truck_l3572_357271

theorem packages_per_truck (total_packages : ℕ) (num_trucks : ℕ) 
  (h1 : total_packages = 490) (h2 : num_trucks = 7) :
  total_packages / num_trucks = 70 := by
  sorry

end NUMINAMATH_CALUDE_packages_per_truck_l3572_357271


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l3572_357247

theorem sum_of_fourth_powers_of_roots (P : ℝ → ℝ) (r₁ r₂ : ℝ) : 
  P = (fun x ↦ x^2 + 2*x + 3) →
  P r₁ = 0 →
  P r₂ = 0 →
  r₁^4 + r₂^4 = -14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l3572_357247


namespace NUMINAMATH_CALUDE_ellipse_higher_focus_coordinates_l3572_357226

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  majorAxis : Point × Point
  minorAxis : Point × Point

/-- The focus of an ellipse with higher y-coordinate -/
def higherFocus (e : Ellipse) : Point :=
  sorry

theorem ellipse_higher_focus_coordinates :
  let e : Ellipse := {
    majorAxis := (⟨3, 0⟩, ⟨3, 8⟩),
    minorAxis := (⟨1, 4⟩, ⟨5, 4⟩)
  }
  let focus := higherFocus e
  focus.x = 3 ∧ focus.y = 4 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_higher_focus_coordinates_l3572_357226


namespace NUMINAMATH_CALUDE_map_scale_l3572_357235

theorem map_scale (map_length : ℝ) (real_distance : ℝ) (query_length : ℝ) :
  map_length > 0 →
  real_distance > 0 →
  query_length > 0 →
  (15 : ℝ) * real_distance = 45 * map_length →
  25 * real_distance = 75 * map_length := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l3572_357235


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3572_357275

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 + 5 * i) / (3 - 5 * i) + (3 - 5 * i) / (3 + 5 * i) = (-16 : ℂ) / 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3572_357275


namespace NUMINAMATH_CALUDE_touching_circle_radius_l3572_357282

/-- A circle touching two semicircles and a line segment --/
structure TouchingCircle where
  /-- Radius of the larger semicircle -/
  R : ℝ
  /-- Radius of the smaller semicircle -/
  r : ℝ
  /-- Radius of the touching circle -/
  x : ℝ
  /-- The smaller semicircle's diameter is half of the larger one -/
  h1 : r = R / 2
  /-- The touching circle is tangent to both semicircles and the line segment -/
  h2 : x > 0 ∧ x < r

/-- The radius of the touching circle is 8 when the larger semicircle has diameter 36 -/
theorem touching_circle_radius (c : TouchingCircle) (h : c.R = 18) : c.x = 8 := by
  sorry

end NUMINAMATH_CALUDE_touching_circle_radius_l3572_357282


namespace NUMINAMATH_CALUDE_randy_biscuits_l3572_357222

/-- The number of biscuits Randy has after receiving and losing some -/
def final_biscuits (initial : ℕ) (from_father : ℕ) (from_mother : ℕ) (eaten_by_brother : ℕ) : ℕ :=
  initial + from_father + from_mother - eaten_by_brother

/-- Theorem stating that Randy ends up with 40 biscuits -/
theorem randy_biscuits : final_biscuits 32 13 15 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_randy_biscuits_l3572_357222


namespace NUMINAMATH_CALUDE_circle_center_l3572_357272

/-- The center of a circle defined by the equation 4x^2 - 8x + 4y^2 - 16y + 20 = 0 is (1, 2) -/
theorem circle_center (x y : ℝ) : 
  (4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0) → 
  (∃ (h : ℝ), h = 0 ∧ (x - 1)^2 + (y - 2)^2 = h) := by
sorry

end NUMINAMATH_CALUDE_circle_center_l3572_357272


namespace NUMINAMATH_CALUDE_team7_cups_l3572_357211

-- Define the number of teams
def num_teams : Nat := 7

-- Define the total amount of soup required
def total_soup : Nat := 2500

-- Define the amount made by the first team
def first_team : Nat := 450

-- Define the amount made by the second team
def second_team : Nat := 300

-- Define the relationship between teams 3+4 and team 7
def teams_34_7_relation (team7 : Nat) : Nat := 2 * team7

-- Define the relationship between teams 1+2 and teams 5+6
def teams_12_56_relation : Nat := first_team + second_team

-- Define the function to calculate the total soup made by all teams
def total_soup_made (team7 : Nat) : Nat :=
  first_team + second_team + teams_34_7_relation team7 + teams_12_56_relation + team7

-- Theorem stating that team 7 should prepare 334 cups to meet the total required
theorem team7_cups : ∃ (team7 : Nat), team7 = 334 ∧ total_soup_made team7 = total_soup := by
  sorry

end NUMINAMATH_CALUDE_team7_cups_l3572_357211


namespace NUMINAMATH_CALUDE_prob_second_unqualified_given_first_is_one_fifth_l3572_357269

/-- A box containing disinfectant bottles -/
structure DisinfectantBox where
  total : ℕ
  qualified : ℕ
  unqualified : ℕ

/-- The probability of drawing an unqualified bottle for the second time,
    given that an unqualified bottle was drawn for the first time -/
def prob_second_unqualified_given_first (box : DisinfectantBox) : ℚ :=
  (box.unqualified - 1 : ℚ) / (box.total - 1)

/-- The main theorem -/
theorem prob_second_unqualified_given_first_is_one_fifth
  (box : DisinfectantBox)
  (h_total : box.total = 6)
  (h_qualified : box.qualified = 4)
  (h_unqualified : box.unqualified = 2) :
  prob_second_unqualified_given_first box = 1/5 :=
sorry

end NUMINAMATH_CALUDE_prob_second_unqualified_given_first_is_one_fifth_l3572_357269


namespace NUMINAMATH_CALUDE_circle_area_increase_l3572_357230

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l3572_357230


namespace NUMINAMATH_CALUDE_percentage_of_A_students_l3572_357201

theorem percentage_of_A_students (total_students : ℕ) (failed_students : ℕ) 
  (h1 : total_students = 32)
  (h2 : failed_students = 18)
  (h3 : ∃ (A : ℕ) (B_C : ℕ), 
    A + B_C + failed_students = total_students ∧ 
    B_C = (total_students - failed_students - A) / 4) :
  (((total_students - failed_students) : ℚ) / total_students) * 100 = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_A_students_l3572_357201


namespace NUMINAMATH_CALUDE_product_sum_equality_l3572_357259

theorem product_sum_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b * c = 1) (h2 : a + 1 / c = 7) (h3 : b + 1 / a = 31) :
  c + 1 / b = 5 / 27 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equality_l3572_357259


namespace NUMINAMATH_CALUDE_zeros_in_concatenated_number_l3572_357251

/-- Counts the number of zeros in a given positive integer -/
def countZeros (n : ℕ) : ℕ := sorry

/-- Counts the total number of zeros in all integers from 1 to n -/
def totalZeros (n : ℕ) : ℕ := sorry

/-- The concatenated number formed by all integers from 1 to 2007 -/
def concatenatedNumber : ℕ := sorry

theorem zeros_in_concatenated_number :
  countZeros concatenatedNumber = 506 := by sorry

end NUMINAMATH_CALUDE_zeros_in_concatenated_number_l3572_357251


namespace NUMINAMATH_CALUDE_xiaoman_dumpling_probability_l3572_357213

theorem xiaoman_dumpling_probability :
  let total_dumplings : ℕ := 10
  let egg_dumplings : ℕ := 3
  let probability : ℚ := egg_dumplings / total_dumplings
  probability = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_xiaoman_dumpling_probability_l3572_357213


namespace NUMINAMATH_CALUDE_intersection_M_N_l3572_357262

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3572_357262


namespace NUMINAMATH_CALUDE_pair_sequence_existence_l3572_357237

theorem pair_sequence_existence (n q : ℕ) (h : n > 0) (h2 : q > 0) :
  ∃ (m : ℕ) (seq : List (Fin n × Fin n)),
    m = ⌈(2 * q : ℚ) / n⌉ ∧
    seq.length = m ∧
    seq.Nodup ∧
    (∀ i < m - 1, ∃ x, (seq.get ⟨i, by sorry⟩).1 = x ∨ (seq.get ⟨i, by sorry⟩).2 = x) ∧
    (∀ i < m - 1, (seq.get ⟨i, by sorry⟩).1.val < (seq.get ⟨i + 1, by sorry⟩).1.val) :=
by sorry

end NUMINAMATH_CALUDE_pair_sequence_existence_l3572_357237


namespace NUMINAMATH_CALUDE_black_area_after_four_changes_l3572_357298

/-- Represents the fraction of black area remaining after a certain number of changes --/
def blackAreaFraction (changes : ℕ) : ℚ :=
  (3/4) ^ changes

/-- The number of changes applied to the triangle --/
def totalChanges : ℕ := 4

/-- Theorem stating that after four changes, the fraction of the original area that remains black is 81/256 --/
theorem black_area_after_four_changes :
  blackAreaFraction totalChanges = 81/256 := by
  sorry

#eval blackAreaFraction totalChanges

end NUMINAMATH_CALUDE_black_area_after_four_changes_l3572_357298


namespace NUMINAMATH_CALUDE_bracelet_pairing_impossibility_l3572_357241

theorem bracelet_pairing_impossibility (n : ℕ) (h : n = 100) :
  ¬ ∃ (arrangement : List (Finset (Fin n))),
    (∀ s ∈ arrangement, s.card = 3) ∧
    (∀ i j : Fin n, i ≠ j → 
      (arrangement.filter (λ s => i ∈ s ∧ j ∈ s)).length = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_bracelet_pairing_impossibility_l3572_357241


namespace NUMINAMATH_CALUDE_average_habitable_land_per_person_approx_l3572_357228

-- Define the given constants
def total_population : ℕ := 281000000
def total_land_area : ℝ := 3797000
def habitable_land_percentage : ℝ := 0.8
def feet_per_mile : ℕ := 5280

-- Theorem statement
theorem average_habitable_land_per_person_approx :
  let habitable_land_area : ℝ := total_land_area * habitable_land_percentage
  let total_habitable_sq_feet : ℝ := habitable_land_area * (feet_per_mile ^ 2 : ℝ)
  let avg_sq_feet_per_person : ℝ := total_habitable_sq_feet / total_population
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1000 ∧ |avg_sq_feet_per_person - 300000| < ε :=
by sorry

end NUMINAMATH_CALUDE_average_habitable_land_per_person_approx_l3572_357228


namespace NUMINAMATH_CALUDE_remainder_9053_div_98_l3572_357216

theorem remainder_9053_div_98 : 9053 % 98 = 37 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9053_div_98_l3572_357216


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3572_357253

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, (-x^2 + c*x - 9 < -4) ↔ (x < 2 ∨ x > 7)) → c = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3572_357253


namespace NUMINAMATH_CALUDE_hospital_bill_proof_l3572_357214

theorem hospital_bill_proof (total_bill : ℝ) (medication_percentage : ℝ) 
  (food_cost : ℝ) (ambulance_cost : ℝ) :
  total_bill = 5000 →
  medication_percentage = 50 →
  food_cost = 175 →
  ambulance_cost = 1700 →
  let remaining_bill := total_bill - (medication_percentage / 100 * total_bill)
  let overnight_cost := remaining_bill - food_cost - ambulance_cost
  overnight_cost / remaining_bill * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_hospital_bill_proof_l3572_357214


namespace NUMINAMATH_CALUDE_inverse_of_A_l3572_357238

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 7; 2, 3]

theorem inverse_of_A :
  A⁻¹ = !![-(3/2), 7/2; 1, -2] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_l3572_357238


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3572_357229

theorem angle_sum_is_pi_over_two (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2) (h_eq : Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β)) : α + β = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3572_357229


namespace NUMINAMATH_CALUDE_sqrt_x_minus_9_meaningful_l3572_357212

theorem sqrt_x_minus_9_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 9) ↔ x ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_9_meaningful_l3572_357212


namespace NUMINAMATH_CALUDE_existence_of_twenty_problem_sequence_l3572_357233

theorem existence_of_twenty_problem_sequence (a : ℕ → ℕ) 
  (h1 : ∀ n, a (n + 1) ≥ a n + 1)
  (h2 : ∀ n, a (n + 7) - a n ≤ 12) :
  ∃ i j, i < j ∧ j ≤ 77 ∧ a j - a i = 20 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_twenty_problem_sequence_l3572_357233


namespace NUMINAMATH_CALUDE_sequence_length_6_to_202_l3572_357293

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Proof that the arithmetic sequence from 6 to 202 with step 2 has 99 terms -/
theorem sequence_length_6_to_202 : 
  arithmeticSequenceLength 6 202 2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_6_to_202_l3572_357293


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l3572_357220

theorem quadratic_equation_transformation (x : ℝ) :
  (x^2 + 2*x - 2 = 0) ↔ ((x + 1)^2 = 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l3572_357220


namespace NUMINAMATH_CALUDE_a_equals_2a_is_valid_assignment_l3572_357290

/-- Definition of a valid assignment statement -/
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (expr : String),
    stmt = var ++ " = " ++ expr ∧
    var.length > 0 ∧
    (∀ c, c ∈ var.data → c.isAlpha)

/-- The statement "a = 2*a" is a valid assignment -/
theorem a_equals_2a_is_valid_assignment :
  is_valid_assignment "a = 2*a" := by
  sorry

#check a_equals_2a_is_valid_assignment

end NUMINAMATH_CALUDE_a_equals_2a_is_valid_assignment_l3572_357290


namespace NUMINAMATH_CALUDE_T_properties_l3572_357225

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (3*x + 2) / (x + 1)}

theorem T_properties :
  ∃ (n N : ℝ),
    n ∈ T ∧
    (∀ y ∈ T, n ≤ y) ∧
    (∀ y ∈ T, y < N) ∧
    N ∉ T ∧
    (∀ ε > 0, ∃ y ∈ T, N - ε < y) :=
  sorry

end NUMINAMATH_CALUDE_T_properties_l3572_357225


namespace NUMINAMATH_CALUDE_bill_difference_l3572_357261

theorem bill_difference (john_tip peter_tip : ℝ) (john_percent peter_percent : ℝ) :
  john_tip = 4 →
  peter_tip = 3 →
  john_percent = 20 / 100 →
  peter_percent = 15 / 100 →
  john_tip = john_percent * (john_tip / john_percent) →
  peter_tip = peter_percent * (peter_tip / peter_percent) →
  (john_tip / john_percent) - (peter_tip / peter_percent) = 0 :=
by
  sorry

#check bill_difference

end NUMINAMATH_CALUDE_bill_difference_l3572_357261


namespace NUMINAMATH_CALUDE_fibonacci_congruence_existence_and_uniqueness_l3572_357287

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_congruence_existence_and_uniqueness :
  ∃! (a b m : ℕ), 0 < a ∧ a < m ∧ 0 < b ∧ b < m ∧
    (∀ n : ℕ, n > 0 → (fibonacci n - a * n * (b ^ n)) % m = 0) ∧
    a = 2 ∧ b = 3 ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_congruence_existence_and_uniqueness_l3572_357287


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3572_357276

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ (1/a + 1/b = 2 ↔ a = 1 ∧ b = 1) :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3572_357276


namespace NUMINAMATH_CALUDE_points_needed_for_average_increase_l3572_357250

/-- Represents a basketball player's scoring history -/
structure PlayerStats where
  gamesPlayed : ℕ
  totalPoints : ℕ

/-- Calculates the average points per game -/
def averagePoints (stats : PlayerStats) : ℚ :=
  stats.totalPoints / stats.gamesPlayed

/-- Updates player stats after a game -/
def updateStats (stats : PlayerStats) (points : ℕ) : PlayerStats :=
  { gamesPlayed := stats.gamesPlayed + 1
  , totalPoints := stats.totalPoints + points }

/-- Theorem: A player who raised their average from 20 to 21 by scoring 36 points
    must score 38 points to raise their average to 22 -/
theorem points_needed_for_average_increase 
  (initialStats : PlayerStats)
  (h1 : averagePoints initialStats = 20)
  (h2 : averagePoints (updateStats initialStats 36) = 21) :
  averagePoints (updateStats (updateStats initialStats 36) 38) = 22 := by
  sorry


end NUMINAMATH_CALUDE_points_needed_for_average_increase_l3572_357250


namespace NUMINAMATH_CALUDE_least_k_value_l3572_357205

theorem least_k_value (k : ℤ) : 
  (0.00010101 * (10 : ℝ)^k > 100) → k ≥ 7 ∧ ∀ m : ℤ, m < 7 → (0.00010101 * (10 : ℝ)^m ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_least_k_value_l3572_357205


namespace NUMINAMATH_CALUDE_min_sum_squares_roots_l3572_357239

/-- The sum of squares of roots of x^2 - (m+1)x + m - 1 = 0 is minimized when m = 0 -/
theorem min_sum_squares_roots (m : ℝ) : 
  let sum_squares := (m + 1)^2 - 2*(m - 1)
  ∀ k : ℝ, sum_squares ≤ (k + 1)^2 - 2*(k - 1) → m = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_roots_l3572_357239


namespace NUMINAMATH_CALUDE_shifted_function_equals_g_l3572_357283

-- Define the original function
def f (x : ℝ) : ℝ := -3 * x + 2

-- Define the shifted function
def g (x : ℝ) : ℝ := -3 * x - 1

-- Define the vertical shift
def shift : ℝ := 3

-- Theorem statement
theorem shifted_function_equals_g :
  ∀ x : ℝ, f x - shift = g x :=
by
  sorry

end NUMINAMATH_CALUDE_shifted_function_equals_g_l3572_357283


namespace NUMINAMATH_CALUDE_inequality_proof_l3572_357270

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3572_357270


namespace NUMINAMATH_CALUDE_sum_of_fractions_minus_eight_l3572_357295

theorem sum_of_fractions_minus_eight (a b c d e f : ℚ) : 
  a = 4 / 2 →
  b = 7 / 4 →
  c = 11 / 8 →
  d = 21 / 16 →
  e = 41 / 32 →
  f = 81 / 64 →
  a + b + c + d + e + f - 8 = 63 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_minus_eight_l3572_357295


namespace NUMINAMATH_CALUDE_max_value_expressions_l3572_357260

theorem max_value_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt (a / (2 * a + b)) + Real.sqrt (b / (2 * b + a)) ≤ 2 * Real.sqrt 3 / 3) ∧
  (Real.sqrt (a / (a + 2 * b)) + Real.sqrt (b / (b + 2 * a)) ≤ 2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expressions_l3572_357260


namespace NUMINAMATH_CALUDE_highlighter_profit_l3572_357227

/-- Calculates the profit from selling highlighter pens under specific conditions --/
theorem highlighter_profit : 
  let total_boxes : ℕ := 12
  let pens_per_box : ℕ := 30
  let cost_per_box : ℕ := 10
  let rearranged_boxes : ℕ := 5
  let pens_per_package : ℕ := 6
  let price_per_package : ℕ := 3
  let pens_per_group : ℕ := 3
  let price_per_group : ℕ := 2

  let total_cost : ℕ := total_boxes * cost_per_box
  let total_pens : ℕ := total_boxes * pens_per_box
  let packages : ℕ := rearranged_boxes * (pens_per_box / pens_per_package)
  let revenue_packages : ℕ := packages * price_per_package
  let remaining_pens : ℕ := total_pens - (rearranged_boxes * pens_per_box)
  let groups : ℕ := remaining_pens / pens_per_group
  let revenue_groups : ℕ := groups * price_per_group
  let total_revenue : ℕ := revenue_packages + revenue_groups
  let profit : ℕ := total_revenue - total_cost

  profit = 115 := by sorry

end NUMINAMATH_CALUDE_highlighter_profit_l3572_357227


namespace NUMINAMATH_CALUDE_square_of_difference_101_minus_2_l3572_357203

theorem square_of_difference_101_minus_2 :
  (101 - 2)^2 = 9801 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_101_minus_2_l3572_357203


namespace NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3572_357255

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- Define the property of arithmetic sequence for Fibonacci numbers
def is_arithmetic_sequence (a : ℕ) : Prop :=
  fib (a + 4) = 2 * fib (a + 2) - fib a

-- Define the sum condition
def sum_condition (a : ℕ) : Prop :=
  a + (a + 2) + (a + 4) = 2500

-- Theorem statement
theorem fibonacci_arithmetic_sequence :
  ∃ a : ℕ, is_arithmetic_sequence a ∧ sum_condition a ∧ a = 831 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3572_357255


namespace NUMINAMATH_CALUDE_marbles_lost_l3572_357242

/-- 
Given that Josh initially had 9 marbles and now has 4 marbles,
prove that the number of marbles he lost is 5.
-/
theorem marbles_lost (initial : ℕ) (current : ℕ) (lost : ℕ) : 
  initial = 9 → current = 4 → lost = initial - current → lost = 5 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l3572_357242


namespace NUMINAMATH_CALUDE_highest_lowest_difference_l3572_357215

/-- Represents the scores of four participants in an exam -/
structure ExamScores where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The conditions of the exam scores -/
def validExamScores (scores : ExamScores) : Prop :=
  scores.A + scores.B = scores.C + scores.D + 17 ∧
  scores.A = scores.B - 4 ∧
  scores.C = scores.D + 5

/-- The theorem stating the difference between the highest and lowest scores -/
theorem highest_lowest_difference (scores : ExamScores) 
  (h : validExamScores scores) : 
  max scores.A (max scores.B (max scores.C scores.D)) - 
  min scores.A (min scores.B (min scores.C scores.D)) = 13 := by
  sorry

#check highest_lowest_difference

end NUMINAMATH_CALUDE_highest_lowest_difference_l3572_357215


namespace NUMINAMATH_CALUDE_same_suit_in_rows_l3572_357231

/-- Represents a playing card suit -/
inductive Suit
| clubs
| diamonds
| hearts
| spades

/-- Represents a card in the grid -/
structure Card where
  suit : Suit
  rank : Nat

/-- Represents the 13 × 4 grid of cards -/
def CardGrid := Fin 13 → Fin 4 → Card

/-- Checks if two cards are adjacent -/
def adjacent (c1 c2 : Card) : Prop :=
  c1.suit = c2.suit ∨ c1.rank = c2.rank

/-- The condition that adjacent cards in the grid are of the same suit or rank -/
def adjacency_condition (grid : CardGrid) : Prop :=
  ∀ i j, (i.val < 12 → adjacent (grid i j) (grid (i + 1) j)) ∧
         (j.val < 3 → adjacent (grid i j) (grid i (j + 1)))

/-- The statement to be proved -/
theorem same_suit_in_rows (grid : CardGrid) 
  (h : adjacency_condition grid) : 
  ∀ j, ∀ i1 i2, (grid i1 j).suit = (grid i2 j).suit :=
sorry

end NUMINAMATH_CALUDE_same_suit_in_rows_l3572_357231


namespace NUMINAMATH_CALUDE_shoe_comparison_l3572_357279

theorem shoe_comparison (bobby_shoes : ℕ) (bonny_shoes : ℕ) : 
  bobby_shoes = 27 →
  bonny_shoes = 13 →
  ∃ (becky_shoes : ℕ), 
    bobby_shoes = 3 * becky_shoes ∧
    2 * becky_shoes - bonny_shoes = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_shoe_comparison_l3572_357279


namespace NUMINAMATH_CALUDE_tanya_bought_eleven_pears_l3572_357219

/-- Represents the number of pears Tanya bought -/
def num_pears : ℕ := sorry

/-- Represents the number of Granny Smith apples Tanya bought -/
def num_apples : ℕ := 4

/-- Represents the number of pineapples Tanya bought -/
def num_pineapples : ℕ := 2

/-- Represents the basket of plums as a single item -/
def num_plum_baskets : ℕ := 1

/-- Represents the total number of fruit items Tanya bought -/
def total_fruits : ℕ := num_pears + num_apples + num_pineapples + num_plum_baskets

/-- Represents the number of fruits remaining in the bag after half fell out -/
def remaining_fruits : ℕ := 9

theorem tanya_bought_eleven_pears :
  num_pears = 11 ∧
  total_fruits = 2 * remaining_fruits :=
by sorry

end NUMINAMATH_CALUDE_tanya_bought_eleven_pears_l3572_357219


namespace NUMINAMATH_CALUDE_probability_specific_arrangement_l3572_357246

theorem probability_specific_arrangement (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  (1 : ℚ) / (n.choose k) = (1 : ℚ) / 35 :=
sorry

end NUMINAMATH_CALUDE_probability_specific_arrangement_l3572_357246
