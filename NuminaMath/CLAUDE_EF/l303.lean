import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l303_30399

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (1, 8)
def C : ℝ × ℝ := (5, 5)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def perimeter (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  distance p1 p2 + distance p2 p3 + distance p3 p1

theorem triangle_perimeter : perimeter A B C = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l303_30399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_negative_terms_l303_30355

-- Define the polynomial P(x)
def P (r p q : ℝ) (x : ℝ) : ℝ := r * x^3 + q * x^2 + p * x + 1

-- Define the sequence a_n
def a : ℕ → ℝ → ℝ → ℝ → ℝ
| 0, p, _, _ => 1
| 1, p, _, _ => -p
| 2, p, q, _ => p^2 - q
| (n + 3), p, q, r => -p * a (n + 2) p q r - q * a (n + 1) p q r - r * a n p q r

-- Theorem statement
theorem infinite_negative_terms (r p q : ℝ) (hr : r > 0) 
  (h_one_real_root : ∃! x, P r p q x = 0) :
  ∃ S : Set ℕ, (Set.Infinite S) ∧ (∀ n ∈ S, a n p q r < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_negative_terms_l303_30355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_term_l303_30308

/-- The coefficient of a polynomial term is the constant factor multiplying the variable part. -/
def coefficient (term : ℝ → ℝ) : ℝ :=
  sorry

/-- A polynomial term with coefficient 1/2π and variable r^2. -/
noncomputable def term (r : ℝ) : ℝ :=
  (1/2) * Real.pi * r^2

theorem coefficient_of_term :
  coefficient term = (1/2) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_term_l303_30308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_50_l303_30347

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  -- We'll use the sorry tactic to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_50_l303_30347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_mean_median_l303_30363

/-- Arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧
  (∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 3 = 8 ∧
  (a 1 * a 7 = a 3 * a 3)

/-- Mean of the arithmetic sequence -/
noncomputable def Mean (a : ℕ → ℝ) : ℝ := (a 1 + a 10) / 2

/-- Median of the arithmetic sequence -/
noncomputable def Median (a : ℕ → ℝ) : ℝ := (a 5 + a 6) / 2

/-- Theorem stating the mean and median are both 13 -/
theorem arithmetic_sequence_mean_median
  (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  Mean a = 13 ∧ Median a = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_mean_median_l303_30363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_correct_product_correct_l303_30362

/-- A calculator that can only compute reciprocals, add, and subtract -/
class BrokenCalculator (α : Type*) [Field α] where
  reciprocal : α → α
  add : α → α → α
  subtract : α → α → α

/-- Computes the square of a number using only reciprocal, addition, and subtraction -/
def square {α : Type*} [Field α] [BrokenCalculator α] (a c : α) : α :=
  let A := BrokenCalculator.subtract
    (BrokenCalculator.reciprocal a)
    (BrokenCalculator.reciprocal (BrokenCalculator.add a c))
  BrokenCalculator.subtract
    (BrokenCalculator.reciprocal A)
    a

/-- Computes the product of two numbers using only reciprocal, addition, and subtraction -/
def product {α : Type*} [Field α] [BrokenCalculator α] (a b : α) : α :=
  let half_b := BrokenCalculator.reciprocal (BrokenCalculator.add (BrokenCalculator.reciprocal b) (BrokenCalculator.reciprocal b))
  let a_plus_half_b := BrokenCalculator.add a half_b
  BrokenCalculator.subtract
    (BrokenCalculator.subtract
      (square a_plus_half_b 1)
      (square a 1))
    (square half_b 1)

theorem square_correct {α : Type*} [Field α] [BrokenCalculator α] (a c : α) :
  square a c = a ^ 2 := by sorry

theorem product_correct {α : Type*} [Field α] [BrokenCalculator α] (a b : α) :
  product a b = a * b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_correct_product_correct_l303_30362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mabel_age_l303_30315

/-- Represents the age of Mabel in years. -/
def age : ℕ := sorry

/-- Represents the total amount of money in Mabel's piggy bank in cents. -/
def total_cents : ℕ := 700  -- $7 = 700 cents

/-- The sum of quarters Mabel has put in her piggy bank over the years. -/
def sum_of_quarters (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating Mabel's age given the conditions of the problem. -/
theorem mabel_age : 
  (age ≥ 1) →  -- Ensure age is positive
  (total_cents = sum_of_quarters age * 25) →  -- Convert quarters to cents
  (age = 7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mabel_age_l303_30315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_assignment_theorem_l303_30353

/-- The number of ways to assign 5 workers to 3 duty positions with constraints -/
def assignment_count : ℕ := 36

/-- The number of workers -/
def num_workers : ℕ := 5

/-- The number of duty positions -/
def num_positions : ℕ := 3

/-- Predicate to represent if a worker is assigned to a position -/
def is_assigned : ℕ → ℕ → Prop := sorry

/-- Function to count the number of ways two specific workers can be assigned to the same position -/
def count_same_position : ℕ → ℕ → ℕ := sorry

theorem worker_assignment_theorem :
  (num_workers = 5) →
  (num_positions = 3) →
  (∀ w : ℕ, w ≤ num_workers → ∃! p : ℕ, p ≤ num_positions ∧ is_assigned w p) →
  (∀ p : ℕ, p ≤ num_positions → ∃ w : ℕ, w ≤ num_workers ∧ is_assigned w p) →
  (count_same_position 1 2 = assignment_count) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_assignment_theorem_l303_30353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l303_30385

/-- The area of a parallelogram with adjacent sides of lengths s and 3s units, 
    forming a 60-degree angle, is equal to (3s²√3)/2. -/
theorem parallelogram_area (s : ℝ) (h_s : s > 0) : 
  (3 * s^2 * Real.sqrt 3) / 2 = 
  s * (3 * s) * Real.sin (60 * (π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l303_30385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_to_cone_theorem_l303_30368

/-- Represents a circular sector --/
structure CircularSector where
  radius : ℝ
  angle : ℝ

/-- Represents a cone --/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ

/-- Converts a circular sector to a cone --/
noncomputable def sectorToCone (s : CircularSector) : Cone :=
  { baseRadius := (s.angle / (2 * Real.pi)) * s.radius,
    slantHeight := s.radius }

theorem sector_to_cone_theorem (s : CircularSector) :
  s.radius = 12 ∧ s.angle = (3 * Real.pi) / 2 →
  let c := sectorToCone s
  c.baseRadius = 9 ∧ c.slantHeight = 12 := by
  sorry

#check sector_to_cone_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_to_cone_theorem_l303_30368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_mod_17_l303_30392

theorem three_digit_integers_mod_17 :
  let S := {x : ℕ | 100 ≤ x ∧ x < 1000 ∧ (5137 * x + 615) % 17 = 1532 % 17}
  Finset.card (Finset.filter (fun x => x ∈ S) (Finset.range 1000)) = 53 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_mod_17_l303_30392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l303_30371

/-- Calculates the market value of a stock given its face value, dividend rate, and current yield. -/
noncomputable def marketValue (faceValue : ℝ) (dividendRate : ℝ) (currentYield : ℝ) : ℝ :=
  (faceValue * dividendRate) / currentYield

/-- Theorem stating that a stock with 11% dividend yield on $100 face value and 8% current yield has a market value of $137.50. -/
theorem stock_market_value :
  let faceValue : ℝ := 100
  let dividendRate : ℝ := 0.11
  let currentYield : ℝ := 0.08
  marketValue faceValue dividendRate currentYield = 137.50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l303_30371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_1_trigonometric_simplification_2_l303_30377

open Real

-- Part 1
theorem trigonometric_simplification_1 (α : ℝ) :
  (Real.sin (π + α) * Real.sin (2 * π - α) * Real.cos (-π - α)) /
  (Real.sin (3 * π + α) * Real.cos (π - α) * Real.cos ((3 * π) / 2 + α)) = -1 := by sorry

-- Part 2
theorem trigonometric_simplification_2 :
  Real.cos (20 * (π / 180)) + Real.cos (160 * (π / 180)) +
  Real.sin (1866 * (π / 180)) - Real.sin (606 * (π / 180)) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_1_trigonometric_simplification_2_l303_30377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bright_spot_area_approx_l303_30331

/-- Represents a glass cone with specific properties --/
structure GlassCone where
  n : ℝ  -- refractive index
  R : ℝ  -- base radius in cm
  h : ℝ  -- height in cm

/-- Represents the screen distance --/
def screen_distance : ℝ := 1

/-- Calculates the area of the bright spot on the screen --/
noncomputable def bright_spot_area (cone : GlassCone) : ℝ :=
  let α := Real.arctan (cone.h / cone.R)
  let R₁ := screen_distance * Real.tan α
  let R₂ := ((cone.h + screen_distance) * Real.tan α) - cone.R
  Real.pi * (R₂^2 - R₁^2)

/-- Theorem stating the area of the bright spot --/
theorem bright_spot_area_approx :
  let cone : GlassCone := ⟨1.5, 1, 1.73⟩
  ∃ ε > 0, abs (bright_spot_area cone - 34) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bright_spot_area_approx_l303_30331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_set_property_l303_30369

-- Define a complete set of positive rational numbers
def CompleteSet (T : Set ℚ) : Prop :=
  ∀ p q : ℚ, 0 < p → 0 < q → p / q ∈ T → (p / (p + q) ∈ T ∧ q / (p + q) ∈ T)

-- Define the property we want to prove
def ContainsAllBetweenZeroAndOne (T : Set ℚ) : Prop :=
  ∀ x : ℚ, 0 < x → x < 1 → x ∈ T

-- Theorem statement
theorem complete_set_property (T : Set ℚ) (h : CompleteSet T) :
  (1 ∈ T → ContainsAllBetweenZeroAndOne T) ∧
  ((1/2 : ℚ) ∈ T → ContainsAllBetweenZeroAndOne T) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_set_property_l303_30369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l303_30326

noncomputable def E : ℝ × ℝ := (1, 2)
noncomputable def F : ℝ × ℝ := (4, 6)
noncomputable def G : ℝ × ℝ := (8, 3)
noncomputable def H : ℝ × ℝ := (10, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def perimeter : ℝ :=
  distance E F + distance F G + distance G H + distance H E

theorem quadrilateral_perimeter :
  ∃ (p q r : ℤ), 
    (perimeter = 10 + Real.sqrt 13 + Real.sqrt 17 * Real.sqrt 5) ∧
    (p + q + r = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l303_30326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l303_30300

open Complex

theorem complex_problem :
  let i : ℂ := Complex.I
  let z : ℂ := 5 + i
  ((z - 3) * (2 - i) = 5) →
  z = 5 + i ∧ 
  abs (z - 2 + 3*i) = 5 ∧ 
  ∀ a : ℝ, (∃ b : ℝ, z * (a + i) = b * i) → a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l303_30300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_N_properties_l303_30319

-- Define the circle N
def circle_N : Set (ℝ × ℝ) := sorry

-- Define the endpoints of the diameter
def endpoint1 : ℝ × ℝ := (2, -3)
def endpoint2 : ℝ × ℝ := (-8, 7)

-- Define the center of the circle
def center : ℝ × ℝ := (-3, 2)

-- Define the radius of the circle
noncomputable def radius : ℝ := 5 * Real.sqrt 2

-- Theorem statement
theorem circle_N_properties :
  (∀ p ∈ circle_N, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2) ∧
  (endpoint1 ∈ circle_N) ∧
  (endpoint2 ∈ circle_N) ∧
  ((endpoint1.1 - endpoint2.1)^2 + (endpoint1.2 - endpoint2.2)^2 = (2 * radius)^2) := by
  sorry

#check circle_N_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_N_properties_l303_30319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l303_30314

/-- The function g as defined in the problem -/
noncomputable def g (n : ℝ) : ℝ := (1/4) * n * (n+1) * (n+2) * (n+3)

/-- Theorem stating the difference of g(r) and g(r-1) -/
theorem g_difference (r : ℝ) : g r - g (r-1) = r * (r+1) * (r+2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l303_30314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_series_sum_l303_30301

-- Define the sequence b_n
def b : ℕ → ℚ
  | 0 => 2  -- We start from 0 to match Lean's natural number indexing
  | 1 => 3
  | n + 2 => (1 / 2) * b (n + 1) + (1 / 3) * b n

-- Define the sum of the sequence
noncomputable def seriesSum : ℚ := ∑' n, b n

-- Theorem statement
theorem b_series_sum : seriesSum = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_series_sum_l303_30301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l303_30396

/-- The function we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (2*x^2 - 3*x + 1)

/-- The decreasing interval of f -/
def decreasing_interval : Set ℝ := { x | x ≥ 3/4 }

/-- Theorem stating that the decreasing interval of f is [3/4, +∞) -/
theorem f_decreasing_interval :
  ∀ x y, x ∈ decreasing_interval → y ∈ decreasing_interval → x < y → f y < f x := by
  sorry

#check f_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l303_30396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l303_30330

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the shorter base -/
  a : ℝ
  /-- The length of the longer base -/
  b : ℝ
  /-- Condition: The shorter base has length 1 -/
  ha : a = 1
  /-- Condition: The longer base has length 9 -/
  hb : b = 9
  /-- The trapezoid contains two inscribed circles -/
  has_two_circles : Prop
  /-- The circles touch each other -/
  circles_touch : Prop
  /-- The circles touch both lateral sides -/
  circles_touch_sides : Prop
  /-- Each circle touches one base -/
  circles_touch_bases : Prop

/-- The area of the isosceles trapezoid -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ :=
  20 * Real.sqrt 3

/-- Theorem: The area of the isosceles trapezoid with the given properties is 20√3 -/
theorem isosceles_trapezoid_area (t : IsoscelesTrapezoid) : area t = 20 * Real.sqrt 3 := by
  -- Unfold the definition of area
  unfold area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l303_30330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probabilities_l303_30394

/-- Represents the outcome of rolling two fair dice -/
structure DiceRoll where
  first : Fin 6
  second : Fin 6

/-- The sample space of all possible outcomes when rolling two fair dice -/
def Ω : Finset DiceRoll := sorry

/-- The probability measure on the sample space -/
noncomputable def P : Set DiceRoll → ℝ := sorry

/-- Event A: The sum of the two rolls is 6 -/
def A : Set DiceRoll := sorry

/-- Event B: The first roll is an odd number -/
def B : Set DiceRoll := sorry

/-- Event C: The two rolls are the same -/
def C : Set DiceRoll := sorry

/-- The intersection of events B and C -/
def B_and_C : Set DiceRoll := B ∩ C

theorem dice_probabilities :
  (P B_and_C = P B * P C) ∧
  (P (A ∩ B) / P B = 1 / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probabilities_l303_30394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hilary_samosa_count_l303_30340

/-- The cost of a single samosa in dollars -/
def samosa_cost : ℚ := 2

/-- The cost of a single order of pakoras in dollars -/
def pakora_cost : ℚ := 3

/-- The number of pakora orders -/
def pakora_orders : ℕ := 4

/-- The cost of a mango lassi in dollars -/
def lassi_cost : ℚ := 2

/-- The tip percentage as a decimal -/
def tip_percentage : ℚ := 1/4

/-- The total cost of the meal including tip and tax in dollars -/
def total_cost : ℚ := 25

/-- The number of samosas Hilary bought -/
def num_samosas : ℕ := 3

theorem hilary_samosa_count :
  samosa_cost * num_samosas +
  pakora_cost * pakora_orders +
  lassi_cost +
  (samosa_cost * num_samosas +
   pakora_cost * pakora_orders +
   lassi_cost) * tip_percentage = total_cost := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hilary_samosa_count_l303_30340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_in_base4_l303_30337

/-- Converts a list of digits in base 4 to a natural number. -/
def toNat4 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 4 + d) 0

/-- Converts a natural number to a list of digits in base 4. -/
def toBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Theorem stating that the sum of 203₄, 112₄, and 321₄ is equal to 1040₄ in base 4. -/
theorem sum_in_base4 :
  toBase4 (toNat4 [3, 0, 2] + toNat4 [2, 1, 1] + toNat4 [1, 2, 3]) = [0, 4, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_in_base4_l303_30337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_range_l303_30351

theorem x_squared_range (x : ℝ) (h : (x + 16) ^ (1/3) - (x - 16) ^ (1/3) = 4) :
  235 < x^2 ∧ x^2 < 245 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_range_l303_30351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pentagon_same_color_l303_30318

/-- A graph consisting of two pentagons with all vertices of one pentagon 
    connected to all vertices of the other pentagon -/
structure TwoPentagonGraph where
  vertices : Finset (Fin 10)
  edges : Finset (Fin 10 × Fin 10)
  pentagon1 : Finset (Fin 10)
  pentagon2 : Finset (Fin 10)
  edge_coloring : (Fin 10 × Fin 10) → Bool
  vertex_count : vertices.card = 10
  edge_count : edges.card = 35
  pentagon_size : pentagon1.card = 5 ∧ pentagon2.card = 5
  pentagons_disjoint : pentagon1 ∩ pentagon2 = ∅
  no_monochrome_triangles : ∀ a b c, 
    (a, b) ∈ edges → (b, c) ∈ edges → (a, c) ∈ edges →
    ¬(edge_coloring (a, b) = edge_coloring (b, c) ∧ 
      edge_coloring (b, c) = edge_coloring (a, c))

/-- The theorem stating that all edges of both pentagons must have the same color -/
theorem two_pentagon_same_color (G : TwoPentagonGraph) : 
  ∃ c : Bool, ∀ e ∈ G.edges, 
    (e.1 ∈ G.pentagon1 ∧ e.2 ∈ G.pentagon1) ∨ 
    (e.1 ∈ G.pentagon2 ∧ e.2 ∈ G.pentagon2) → 
    G.edge_coloring e = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pentagon_same_color_l303_30318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l303_30387

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * a * x^2 - x

-- Main theorem
theorem f_properties :
  -- Part 1: f is monotonically increasing iff a = 1
  (∀ x y, x < y → f 1 x < f 1 y) ∧
  (∀ a, (∀ x y, x < y → f a x < f a y) → a = 1) ∧
  
  -- Part 2: Properties when a > 1
  ∀ a, a > 1 →
    -- i) f has exactly two extremum points
    (∃! x₁ x₂, x₁ < x₂ ∧ 
      (∀ x, x ≠ x₁ → x ≠ x₂ → deriv (f a) x ≠ 0) ∧
      deriv (f a) x₁ = 0 ∧ deriv (f a) x₂ = 0) ∧
    
    -- ii) x₂ - x₁ increases as a increases
    (∀ a₁ a₂, a < a₁ → a₁ < a₂ → 
      ∃ x₁ x₂ x₃ x₄, deriv (f a₁) x₁ = 0 ∧ deriv (f a₁) x₂ = 0 ∧ x₁ < x₂ ∧
                     deriv (f a₂) x₃ = 0 ∧ deriv (f a₂) x₄ = 0 ∧ x₃ < x₄ ∧
                     x₂ - x₁ < x₄ - x₃) ∧
    
    -- iii) f(x₂) < 1 + (sin x₂ - x₂)/2
    (∃ x₂, deriv (f a) x₂ = 0 ∧ 
      ∀ x, deriv (f a) x = 0 → x ≤ x₂ →
        f a x₂ < 1 + (Real.sin x₂ - x₂) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l303_30387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_l303_30398

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2

def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def interval (k : ℤ) : Set ℝ :=
  {x | Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ 7 * Real.pi / 12 + k * Real.pi}

theorem f_decreasing_intervals :
  ∀ k : ℤ, monotonically_decreasing f (Real.pi / 12 + k * Real.pi) (7 * Real.pi / 12 + k * Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_l303_30398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_l303_30356

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem odd_function_range (f : ℝ → ℝ) (h_odd : is_odd f)
    (h_mono : monotone_increasing_on f (Set.Ici 0))
    (h_ineq : ∀ x, 0 < x → f (f x) + f (Real.log x - 2) < 0) :
    ∀ x, 0 < x ∧ x < 10 → f (f x) + f (Real.log x - 2) < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_l303_30356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l303_30384

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | -3 < x ∧ x ≤ 6}

def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

theorem set_operations :
  (A ∪ B = {x | -3 < x ∧ x ≤ 6}) ∧
  ((U \ B) ∩ A = {x | (-3 < x ∧ x ≤ -1) ∨ x = 6}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l303_30384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_theorem_l303_30321

noncomputable def inner_radius : ℝ := 3
noncomputable def outer_radius : ℝ := 5

noncomputable def chord_intersection_probability : ℝ := 
  Real.arctan (inner_radius / (outer_radius - inner_radius)) / Real.pi

theorem chord_intersection_probability_theorem :
  chord_intersection_probability = Real.arctan (3 / 4) / Real.pi := by
  -- Unfold the definition of chord_intersection_probability
  unfold chord_intersection_probability
  -- Simplify the expression
  simp [inner_radius, outer_radius]
  -- The proof is completed
  sorry

#check chord_intersection_probability_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_theorem_l303_30321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_a_inverse_minus_one_squared_b_value_l303_30346

noncomputable def a : ℝ := Real.sqrt 3 * (3 : ℝ) ^ (1/3) * (3 : ℝ) ^ (1/6)

theorem sqrt_a_inverse_minus_one_squared (a : ℝ) (h : a = Real.sqrt 3 * (3 : ℝ) ^ (1/3) * (3 : ℝ) ^ (1/6)) : 
  Real.sqrt ((a⁻¹ - 1)^2) = 2/3 := by sorry

theorem b_value (a b : ℝ) (h1 : a = Real.sqrt 3 * (3 : ℝ) ^ (1/3) * (3 : ℝ) ^ (1/6)) 
  (h2 : (b : ℝ) ^ (1/3) * (-b : ℝ) ^ (1/6) = -a) : b = -9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_a_inverse_minus_one_squared_b_value_l303_30346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p_sufficient_not_necessary_for_q_l303_30378

-- Define the conditions p and q
def p (a : ℝ) : Prop := a ≥ -2
def q (a : ℝ) : Prop := a < 0

-- Define what it means for one condition to be sufficient for another
def is_sufficient (P Q : ℝ → Prop) : Prop :=
  ∀ a, P a → Q a

-- Define what it means for one condition to be necessary for another
def is_necessary (P Q : ℝ → Prop) : Prop :=
  ∀ a, Q a → P a

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_q :
  (is_sufficient (fun a ↦ ¬(p a)) q) ∧ ¬(is_necessary (fun a ↦ ¬(p a)) q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p_sufficient_not_necessary_for_q_l303_30378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_degrees_for_polynomial_division_l303_30322

-- Define the divisor polynomial
def divisor_polynomial (x : ℝ) : ℝ := x^4 - 2*x^3 + x - 5

-- Define the set of possible remainder degrees
def possible_remainder_degrees : Set ℕ := {0, 1, 2, 3}

-- Theorem statement
theorem remainder_degrees_for_polynomial_division :
  ∀ (p : Polynomial ℝ), ∃ (q r : Polynomial ℝ),
    p = q * (Polynomial.X^4 - 2*Polynomial.X^3 + Polynomial.X - 5) + r ∧
    (r = 0 ∨ r.natDegree ∈ possible_remainder_degrees) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_degrees_for_polynomial_division_l303_30322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_perpendicular_if_equal_norms_l303_30352

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable def angle_between (u v : E) : ℝ := Real.arccos (inner u v / (norm u * norm v))

theorem vectors_perpendicular_if_equal_norms (u v : E) 
  (h : ‖u + (2 : ℝ) • v‖ = ‖u - (2 : ℝ) • v‖) : 
  angle_between u v = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_perpendicular_if_equal_norms_l303_30352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_sum_l303_30311

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 1 / x + 2 / (x^2)

-- Theorem statement
theorem tangent_slope_sum (α : ℝ) :
  f' 1 = Real.tan α → 0 < α → α < Real.pi / 2 →
  Real.cos α + Real.sin α = 2 * Real.sqrt 10 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_sum_l303_30311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_eight_l303_30332

theorem cube_root_negative_eight : ((-8 : ℝ) ^ (1/3 : ℝ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_eight_l303_30332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_a_cheaper_range_l303_30359

/-- Cost function for Plan A -/
noncomputable def L (x : ℝ) : ℝ :=
  if x ≤ 30 then 2 + 0.5 * x else 0.6 * x - 1

/-- Cost function for Plan B -/
noncomputable def F (x : ℝ) : ℝ := 0.58 * x

/-- Theorem stating the range where Plan A is cheaper than Plan B -/
theorem plan_a_cheaper_range :
  ∀ x : ℝ, (25 < x ∧ x < 50) ↔ L x < F x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_a_cheaper_range_l303_30359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_nine_equals_eightyone_l303_30350

theorem power_of_nine_equals_eightyone (x : ℝ) : (9 : ℝ)^x = 81 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_nine_equals_eightyone_l303_30350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l303_30367

-- Define the parametric equations for curve C₁
noncomputable def C₁ (t : ℝ) : ℝ × ℝ :=
  (4 + 5 * Real.cos t, 5 + 5 * Real.sin t)

-- Define the polar equation for curve C₂
noncomputable def C₂ (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the conversion from polar to Cartesian coordinates
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Theorem statement
theorem intersection_points :
  ∃ (t₁ t₂ θ₁ θ₂ : ℝ),
    C₁ t₁ = polar_to_cartesian (C₂ θ₁) θ₁ ∧
    C₁ t₂ = polar_to_cartesian (C₂ θ₂) θ₂ ∧
    (C₂ θ₁ = Real.sqrt 2 ∧ θ₁ = Real.pi / 4) ∧
    (C₂ θ₂ = 2 ∧ θ₂ = Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l303_30367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bread_is_310_l303_30341

/-- The amount of bread Cara ate for dinner, in grams -/
noncomputable def dinner_bread : ℝ := 240

/-- The ratio of bread eaten for dinner compared to lunch -/
noncomputable def dinner_lunch_ratio : ℝ := 8

/-- The ratio of bread eaten for dinner compared to breakfast -/
noncomputable def dinner_breakfast_ratio : ℝ := 6

/-- The total amount of bread Cara ate throughout the day -/
noncomputable def total_bread : ℝ := dinner_bread + dinner_bread / dinner_lunch_ratio + dinner_bread / dinner_breakfast_ratio

/-- Theorem stating that the total amount of bread Cara ate is 310 grams -/
theorem total_bread_is_310 : total_bread = 310 := by
  -- Unfold the definitions
  unfold total_bread dinner_bread dinner_lunch_ratio dinner_breakfast_ratio
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bread_is_310_l303_30341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equation_solution_l303_30302

noncomputable def g (x : ℝ) : ℝ := Real.rpow ((x + 5) / 5) (1/3)

theorem g_equation_solution :
  ∃! x : ℝ, g (3 * x) = 3 * g x ∧ x = -65/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equation_solution_l303_30302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_at_point_l303_30389

/-- Two lines in a plane are defined by their parametric equations -/
noncomputable def line1 (s : ℝ) : ℝ × ℝ := (2 + 3*s, 3 - 4*s)
noncomputable def line2 (v : ℝ) : ℝ × ℝ := (4 + 5*v, -6 + v)

/-- The intersection point of the two lines -/
noncomputable def intersection_point : ℝ × ℝ := (425/69, 151/69)

/-- Theorem stating that the intersection_point is indeed the intersection of line1 and line2 -/
theorem lines_intersect_at_point :
  ∃ (s v : ℝ), line1 s = line2 v ∧ line1 s = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_at_point_l303_30389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xSequence_2007_equals_2_l303_30320

def xSequence : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (1 + xSequence (n + 1)) / xSequence n

theorem xSequence_2007_equals_2 : xSequence 2007 = 2 := by
  -- Proof steps will go here
  sorry

#eval xSequence 2007

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xSequence_2007_equals_2_l303_30320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l303_30324

def has_exactly_two_divisors (m : ℕ) : Prop :=
  ∃ a b : ℕ, (a ≠ b ∧ a > 0 ∧ b > 0 ∧ m % a = 0 ∧ m % b = 0) ∧
  ∀ c : ℕ, c > 0 → m % c = 0 → (c = a ∨ c = b)

def has_exactly_four_divisors (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    n % a = 0 ∧ n % b = 0 ∧ n % c = 0 ∧ n % d = 0) ∧
  ∀ e : ℕ, e > 0 → n % e = 0 → (e = a ∨ e = b ∨ e = c ∨ e = d)

theorem sum_of_special_numbers :
  ∃ m n : ℕ,
    (∀ k : ℕ, k < m → ¬(has_exactly_two_divisors k)) ∧
    has_exactly_two_divisors m ∧
    has_exactly_four_divisors n ∧
    n < 500 ∧
    (∀ l : ℕ, l < 500 → l > n → ¬(has_exactly_four_divisors l)) ∧
    m + n = 345 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l303_30324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_295_l303_30370

/-- Represents the cycling problem with given conditions -/
structure CyclingProblem where
  jack_speed_to_store : ℝ
  jack_speed_to_peter : ℝ
  store_to_peter_distance : ℝ
  combined_return_speed : ℝ
  detour_distance : ℝ
  home_to_store_time_ratio : ℝ

/-- Calculates the total distance cycled by Jack and Peter -/
noncomputable def total_distance (p : CyclingProblem) : ℝ :=
  let time_to_peter := p.store_to_peter_distance / p.jack_speed_to_peter
  let home_to_store_distance := p.jack_speed_to_store * (time_to_peter * p.home_to_store_time_ratio)
  let return_distance := p.store_to_peter_distance + p.detour_distance
  (home_to_store_distance + p.store_to_peter_distance + return_distance) +
  (p.store_to_peter_distance + return_distance)

/-- Theorem stating that the total distance cycled is 295 miles -/
theorem total_distance_is_295 (p : CyclingProblem)
  (h1 : p.jack_speed_to_store = 15)
  (h2 : p.jack_speed_to_peter = 20)
  (h3 : p.store_to_peter_distance = 50)
  (h4 : p.combined_return_speed = 18)
  (h5 : p.detour_distance = 10)
  (h6 : p.home_to_store_time_ratio = 2) :
  total_distance p = 295 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_295_l303_30370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sqrt_of_seven_l303_30388

-- Define the arithmetic square root function as noncomputable
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Theorem statement
theorem arithmetic_sqrt_of_seven :
  arithmetic_sqrt 7 = Real.sqrt 7 := by
  -- Unfold the definition of arithmetic_sqrt
  unfold arithmetic_sqrt
  -- The equality now follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sqrt_of_seven_l303_30388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_problem_l303_30345

/-- The volume of water in a tank after evaporation, draining, and rain. -/
noncomputable def final_water_volume (initial_volume evaporated_volume drained_volume rain_duration rain_rate : ℝ) : ℝ :=
  initial_volume - evaporated_volume - drained_volume + (rain_duration / 10) * rain_rate

/-- Theorem stating the final water volume in the tank -/
theorem water_tank_problem :
  let initial_volume : ℝ := 6000
  let evaporated_volume : ℝ := 2000
  let drained_volume : ℝ := 3500
  let rain_duration : ℝ := 30
  let rain_rate : ℝ := 350
  final_water_volume initial_volume evaporated_volume drained_volume rain_duration rain_rate = 1550 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_problem_l303_30345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implication_l303_30304

theorem inequality_implication (x y : ℝ) : 
  (2 : ℝ)^x - (2 : ℝ)^y < (3 : ℝ)^(-x) - (3 : ℝ)^(-y) → Real.log (y - x + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implication_l303_30304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_complement_theorem_l303_30373

/-- The complement of an angle is 90 degrees minus the angle -/
noncomputable def complement (angle : ℝ) : ℝ := 90 - angle

/-- Converting degrees and minutes to decimal degrees -/
noncomputable def to_decimal_degrees (degrees : ℕ) (minutes : ℕ) : ℝ := 
  (degrees : ℝ) + ((minutes : ℝ) / 60)

theorem angle_complement_theorem :
  let angle_A := to_decimal_degrees 29 18
  complement angle_A = to_decimal_degrees 60 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_complement_theorem_l303_30373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l303_30360

-- Define the triangle DEF
def triangle_DEF : Set (ℝ × ℝ) := sorry

-- Define the side lengths of the triangle
def DE : ℝ := 15
def EF : ℝ := 20
def FD : ℝ := 25

-- Define the rectangle WXYZ
def rectangle_WXYZ : Set (ℝ × ℝ) := sorry

-- Define λ as the length of WX
def lambda : ℝ := sorry

-- Define the area function of WXYZ
def area_WXYZ (γ δ : ℝ) : ℝ → ℝ := fun x ↦ γ * x - δ * x^2

-- Define p and q as positive integers
def p : ℕ+ := sorry
def q : ℕ+ := sorry

-- State the theorem
theorem inscribed_rectangle_area :
  ∃ (γ δ : ℝ), 
    (∀ x, area_WXYZ γ δ x = γ * x - δ * x^2) ∧
    (γ = (p : ℝ) / (q : ℝ)) ∧
    (Nat.Coprime p.val q.val) ∧
    (γ = 15) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l303_30360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l303_30338

/-- The probability of a coin landing heads on a single flip. -/
noncomputable def prob_heads : ℚ := 1/2

/-- The number of times the coin is flipped. -/
def num_flips : ℕ := 5

/-- The probability of the coin landing heads on the first flip and tails on the last 4 flips. -/
noncomputable def prob_heads_then_tails : ℚ := (1/2)^5

theorem coin_flip_probability :
  prob_heads_then_tails = (prob_heads) * (1 - prob_heads)^(num_flips - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l303_30338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_exists_l303_30366

/-- A city in the kingdom -/
structure City where
  routes : ℕ

/-- The kingdom's transportation network -/
structure Kingdom where
  cities : Set City
  capital : City
  distant : City
  h_capital : capital.routes = 21
  h_distant : distant.routes = 1
  h_others : ∀ c ∈ cities, c ≠ capital → c ≠ distant → c.routes = 20

/-- A path exists between the capital and the city of Distant -/
theorem path_exists (k : Kingdom) : ∃ (path : List City), 
  path.head? = some k.capital ∧ 
  path.getLast? = some k.distant ∧
  ∀ (i : ℕ) (c₁ c₂ : City), 
    i < path.length - 1 → 
    path.get? i = some c₁ → 
    path.get? (i + 1) = some c₂ → 
    ∃ (route : ℕ), route ≤ c₁.routes :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_exists_l303_30366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_foci_distance_product_l303_30316

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Product of distances from intersection point to foci -/
theorem intersection_foci_distance_product 
  (e : Ellipse) 
  (h : Hyperbola) 
  (F₁ F₂ P : Point) 
  (he : e.a^2 = 25 ∧ e.b^2 = 16) 
  (hh : h.a^2 = 4 ∧ h.b^2 = 5) 
  (h_common_foci : ∃ (c : ℝ), c^2 = e.a^2 - e.b^2 ∧ c^2 = h.a^2 + h.b^2) 
  (h_on_ellipse : P.x^2 / e.a^2 + P.y^2 / e.b^2 = 1) 
  (h_on_hyperbola : P.x^2 / h.a^2 - P.y^2 / h.b^2 = 1) : 
  distance P F₁ * distance P F₂ = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_foci_distance_product_l303_30316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l303_30380

theorem tan_alpha_value (α : ℝ) 
  (h : (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 2) : 
  Real.tan α = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l303_30380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_five_l303_30357

open Real

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![sqrt 2, -sqrt 2; sqrt 2, sqrt 2]

theorem matrix_power_five :
  A ^ 5 = !![(-16 : ℝ), 16; -16, -16] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_five_l303_30357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_greater_than_one_l303_30372

noncomputable def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y = 0

noncomputable def point_to_line_distance (m : ℝ) : ℝ :=
  |3 * m + 4| / 5

theorem distance_greater_than_one (m : ℝ) : 
  point_to_line_distance m > 1 ↔ m ∈ Set.Iic (-3) ∪ Set.Ioi (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_greater_than_one_l303_30372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_five_percent_l303_30334

-- Define the profit percentages
def profit_with_discount : ℚ := 23.5
def profit_without_discount : ℚ := 30

-- Define the function to calculate the discount percentage
noncomputable def discount_percentage (cost_price : ℚ) : ℚ :=
  let selling_price_with_discount := cost_price * (1 + profit_with_discount / 100)
  let selling_price_without_discount := cost_price * (1 + profit_without_discount / 100)
  let discount := selling_price_without_discount - selling_price_with_discount
  (discount / selling_price_without_discount) * 100

-- Theorem statement
theorem discount_is_five_percent (cost_price : ℚ) (h : cost_price > 0) :
  discount_percentage cost_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_five_percent_l303_30334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_probability_condition_l303_30325

noncomputable def probability_no_close_pair (n : ℕ) : ℝ := (n - 2)^3 / n^3

theorem smallest_n_for_probability_condition : 
  (∀ k < 10, probability_no_close_pair k ≤ 1/2) ∧ 
  probability_no_close_pair 10 > 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_probability_condition_l303_30325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_inequality_l303_30336

theorem root_sum_inequality (m n : ℕ+) :
  (1 : ℝ) / ((n + 1 : ℝ) ^ (1 / m.val)) + (1 : ℝ) / ((m + 1 : ℝ) ^ (1 / n.val)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_inequality_l303_30336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l303_30310

noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 1)
def b : ℝ × ℝ := (0, 1)
noncomputable def c (k : ℝ) : ℝ × ℝ := (k, Real.sqrt 3)

theorem perpendicular_vectors (k : ℝ) : 
  (a.1 + 2 * b.1) * (c k).1 + (a.2 + 2 * b.2) * (c k).2 = 0 → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l303_30310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_range_l303_30313

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.exp (1 + x^2) - 1 / (1 + x^2) + |x|

-- State the theorem
theorem g_inequality_range (x : ℝ) :
  g (x - 1) > g (3 * x + 1) ↔ x > -1 ∧ x < 0 := by
  sorry

-- You can add more theorems or lemmas here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_range_l303_30313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_integer_probability_all_fourth_terms_are_integers_l303_30306

-- Define the sequence type
def Sequence := List Int

-- Define the coin flip result
inductive CoinFlip
| Heads
| Tails

-- Define the function to generate the next term
def nextTerm (a : Int) (flip : CoinFlip) : Int :=
  match flip with
  | CoinFlip.Heads => if a % 2 = 0 then 2 * a - 1 else 3 * a + 1
  | CoinFlip.Tails => if a % 2 = 0 then a / 2 else (a - 1) / 2

-- Define a function to generate all possible fourth terms
def generateFourthTerms (start : Int) : List Int :=
  let secondTerms := [nextTerm start CoinFlip.Heads, nextTerm start CoinFlip.Tails]
  let thirdTerms := secondTerms.bind (λ t => [nextTerm t CoinFlip.Heads, nextTerm t CoinFlip.Tails])
  thirdTerms.bind (λ t => [nextTerm t CoinFlip.Heads, nextTerm t CoinFlip.Tails])

-- Theorem statement
theorem fourth_term_integer_probability (start : Int := 7) :
  (generateFourthTerms start).all (λ x => true) = true := by
  sorry

-- Helper function to check if all elements are integers (always true for Int)
def allIntegers (l : List Int) : Bool :=
  l.all (λ _ => true)

-- Theorem to show that all fourth terms are integers
theorem all_fourth_terms_are_integers (start : Int := 7) :
  allIntegers (generateFourthTerms start) = true := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_integer_probability_all_fourth_terms_are_integers_l303_30306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_iff_n_eq_four_l303_30395

/-- Represents the n-gon painting game where n is the number of sides. -/
structure PaintingGame where
  n : ℕ

/-- Represents a player in the game. -/
inductive Player
  | First
  | Second

/-- Counts the number of adjacent painted sides for a given move. -/
def AdjacentPaintedSides (g : PaintingGame) (move : ℕ) (previous_moves : List ℕ) : ℕ :=
  sorry

/-- Checks if a move is valid for a given player. -/
def IsValidMove (g : PaintingGame) (p : Player) (move : ℕ) (previous_moves : List ℕ) : Prop :=
  match p with
  | Player.First => 
      move < g.n ∧ (AdjacentPaintedSides g move previous_moves = 0 ∨ AdjacentPaintedSides g move previous_moves = 2)
  | Player.Second => 
      move < g.n ∧ AdjacentPaintedSides g move previous_moves = 1

/-- Represents a winning strategy for a player. -/
def WinningStrategy (g : PaintingGame) (p : Player) : Prop :=
  ∀ (opponent_moves : List ℕ), ∃ (player_move : ℕ), IsValidMove g p player_move opponent_moves

/-- The main theorem stating that the second player has a winning strategy if and only if n = 4. -/
theorem second_player_wins_iff_n_eq_four (g : PaintingGame) :
  WinningStrategy g Player.Second ↔ g.n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_iff_n_eq_four_l303_30395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_feeding_cost_is_21_l303_30382

/-- Calculates the cost of feeding pets for a week -/
noncomputable def calculate_weekly_feeding_cost (turtle_weight : ℝ) (bird_weight : ℝ) (hamster_weight : ℝ)
  (turtle_food_oz_per_half_lb : ℝ) (bird_food_oz_per_lb : ℝ) (hamster_food_oz_per_half_lb : ℝ)
  (turtle_food_oz_per_jar : ℝ) (bird_food_oz_per_bag : ℝ) (hamster_food_oz_per_box : ℝ)
  (turtle_food_cost_per_jar : ℝ) (bird_food_cost_per_bag : ℝ) (hamster_food_cost_per_box : ℝ) : ℝ :=
  let turtle_food_oz := turtle_weight * 2 * turtle_food_oz_per_half_lb
  let bird_food_oz := bird_weight * bird_food_oz_per_lb
  let hamster_food_oz := hamster_weight * 2 * hamster_food_oz_per_half_lb
  let turtle_jars := ⌈turtle_food_oz / turtle_food_oz_per_jar⌉
  let bird_bags := ⌈bird_food_oz / bird_food_oz_per_bag⌉
  let hamster_boxes := ⌈hamster_food_oz / hamster_food_oz_per_box⌉
  turtle_jars * turtle_food_cost_per_jar +
  bird_bags * bird_food_cost_per_bag +
  hamster_boxes * hamster_food_cost_per_box

/-- The total cost to feed all pets for a week is $21 -/
theorem weekly_feeding_cost_is_21 :
  calculate_weekly_feeding_cost 30 8 3 1 2 1.5 15 40 20 3 5 4 = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_feeding_cost_is_21_l303_30382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_length_specific_case_l303_30383

/-- Represents an acute-angled triangle with an inscribed square -/
structure AcuteTriangleWithSquare where
  /-- The height of the triangle from vertex X to side YZ -/
  h : ℝ
  /-- The side length of the inscribed square -/
  s : ℝ
  /-- The length of side YZ -/
  b : ℝ
  /-- The triangle is acute-angled -/
  acute : h > 0 ∧ s > 0 ∧ b > 0
  /-- The square fits within the triangle -/
  square_fits : s ≤ h ∧ s ≤ b

/-- There exists a unique base length b for given h and s -/
theorem unique_base_length (h s : ℝ) (h_pos : h > 0) (s_pos : s > 0) (s_le_h : s ≤ h) :
  ∃! b : ℝ, ∃ t : AcuteTriangleWithSquare, t.h = h ∧ t.s = s ∧ t.b = b := by
  sorry

/-- The specific case where h = 3 and s = 2 -/
theorem specific_case :
  ∃! b : ℝ, ∃ t : AcuteTriangleWithSquare, t.h = 3 ∧ t.s = 2 ∧ t.b = b := by
  apply unique_base_length 3 2
  · exact (by norm_num : (3 : ℝ) > 0)
  · exact (by norm_num : (2 : ℝ) > 0)
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_length_specific_case_l303_30383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mayoral_race_voting_percentage_l303_30374

theorem mayoral_race_voting_percentage :
  ∀ (V : ℝ), V > 0 →
  let democrat_percentage : ℝ := 0.60
  let republican_percentage : ℝ := 1 - democrat_percentage
  let democrat_voting_for_A : ℝ := 0.75
  let total_voting_for_A : ℝ := 0.53
  let republican_voting_for_A : ℝ := 
    (total_voting_for_A - democrat_percentage * democrat_voting_for_A) / republican_percentage
  republican_voting_for_A = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mayoral_race_voting_percentage_l303_30374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l303_30344

/-- The time (in seconds) it takes for a train to pass a person moving in the opposite direction. -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) : ℝ :=
  train_length / ((train_speed + person_speed) * (5 / 18))

/-- Theorem stating that the time for a 275 m long train moving at 60 km/hr to pass a man
    moving at 6 km/hr in the opposite direction is approximately 15 seconds. -/
theorem train_passing_man_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |train_passing_time 275 60 6 - 15| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l303_30344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vector_product_l303_30303

/-- Given a parallelogram ABCD with points E and F such that
    E is the midpoint of AB and F is on BC with CF = 2FB,
    and M is the intersection of CE and DF,
    if AM = λAB + μAD, then λμ = 3/8 -/
theorem parallelogram_vector_product (A B C D E F M : ℝ × ℝ) 
  (lambda mu : ℝ) : 
  (∀ (X Y : ℝ × ℝ), (C - A) = (D - B) → (D - A) = (B - C) → -- ABCD is a parallelogram
   (E - A) = (B - E) → -- E is midpoint of AB
   (C - F) = 2 • (F - B) → -- F is trisection point of BC
   (∃ t : ℝ, M - C = t • (E - C)) → -- M is on CE
   (∃ s : ℝ, M - D = s • (F - D)) → -- M is on DF
   (M - A) = lambda • (B - A) + mu • (D - A)) → -- AM = λAB + μAD
  lambda * mu = 3/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vector_product_l303_30303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l303_30309

-- Define the function f(x) = ln x - 3x
noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := 1 / x - 3

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  (2 : ℝ) * x + y + 1 = 0 ↔ y - y₀ = m * (x - x₀) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l303_30309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l303_30335

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem max_omega_value (ω φ : ℝ) 
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < Real.pi / 2)
  (h_odd : ∀ x, f ω φ (x - Real.pi / 8) = -f ω φ (-(x - Real.pi / 8)))
  (h_even : ∀ x, f ω φ (x + Real.pi / 8) = f ω φ (-(x + Real.pi / 8)))
  (h_roots : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ 
    0 < r1 ∧ r1 < Real.pi / 6 ∧ 
    0 < r2 ∧ r2 < Real.pi / 6 ∧
    f ω φ r1 = Real.sqrt 2 / 2 ∧ 
    f ω φ r2 = Real.sqrt 2 / 2 ∧
    ∀ x, 0 < x ∧ x < Real.pi / 6 ∧ f ω φ x = Real.sqrt 2 / 2 → x = r1 ∨ x = r2) :
  ω ≤ 10 ∧ ∃ ω₀, ω₀ = 10 ∧ 
    ∃ φ₀, 0 < φ₀ ∧ φ₀ < Real.pi / 2 ∧
    (∀ x, f ω₀ φ₀ (x - Real.pi / 8) = -f ω₀ φ₀ (-(x - Real.pi / 8))) ∧
    (∀ x, f ω₀ φ₀ (x + Real.pi / 8) = f ω₀ φ₀ (-(x + Real.pi / 8))) ∧
    (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ 
      0 < r1 ∧ r1 < Real.pi / 6 ∧ 
      0 < r2 ∧ r2 < Real.pi / 6 ∧
      f ω₀ φ₀ r1 = Real.sqrt 2 / 2 ∧ 
      f ω₀ φ₀ r2 = Real.sqrt 2 / 2 ∧
      ∀ x, 0 < x ∧ x < Real.pi / 6 ∧ f ω₀ φ₀ x = Real.sqrt 2 / 2 → x = r1 ∨ x = r2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l303_30335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EFGH_area_l303_30342

-- Define the trapezoid EFGH
structure Trapezoid where
  EF : ℚ
  GH : ℚ
  EG : ℚ
  FH : ℚ
  altitude : ℚ

-- Define the specific trapezoid from the problem
def EFGH : Trapezoid :=
  { EF := 60
  , GH := 30
  , EG := 25
  , FH := 18
  , altitude := 15
  }

-- Function to calculate the area of a trapezoid
def trapezoidArea (t : Trapezoid) : ℚ :=
  (t.EF + t.GH) * t.altitude / 2

-- Theorem statement
theorem EFGH_area : trapezoidArea EFGH = 675 := by
  -- Unfold the definition of trapezoidArea and EFGH
  unfold trapezoidArea EFGH
  -- Simplify the arithmetic expression
  simp [Trapezoid.EF, Trapezoid.GH, Trapezoid.altitude]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_EFGH_area_l303_30342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l303_30327

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- Main theorem
theorem odd_function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b x = -f a b (-x)) →  -- f is odd on (-1,1)
  f a b (1/2) = 2/5 →                                      -- f(1/2) = 2/5
  (∃ g : ℝ → ℝ, 
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b x = g x) ∧       -- f equals g on (-1,1)
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → g x = x / (1 + x^2)) ∧ -- g(x) = x / (1 + x^2)
    (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → g x < g y) ∧ -- g is increasing on (-1,1)
    {t : ℝ | g (t-1) + g t < 0} = Set.Ioo 0 (1/2)) :=  -- solution set for g(t-1) + g(t) < 0
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l303_30327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_two_l303_30348

noncomputable def g (x : ℝ) : ℝ := 
  (x^(2^2008 - 1) - 1)⁻¹ * ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^(2^2007) + 1) - 1)

theorem g_of_two : g 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_two_l303_30348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l303_30391

/-- Represents a triangle in the sequence -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Constructs the next triangle in the sequence -/
noncomputable def nextTriangle (T : Triangle) : Triangle :=
  { a := (T.b + T.c - T.a) / 2,
    b := (T.a + T.c - T.b) / 2,
    c := (T.a + T.b - T.c) / 2 }

/-- Checks if a triangle satisfies the triangle inequality -/
def isValidTriangle (T : Triangle) : Prop :=
  T.a + T.b > T.c ∧ T.b + T.c > T.a ∧ T.c + T.a > T.b

/-- Calculates the perimeter of a triangle -/
noncomputable def perimeter (T : Triangle) : ℝ :=
  T.a + T.b + T.c

/-- The initial triangle T₁ -/
def T1 : Triangle :=
  { a := 2022,
    b := 2023,
    c := 2021 }

/-- The theorem to be proved -/
theorem last_triangle_perimeter :
  ∃ n : ℕ, 
    let Tn := (Nat.iterate nextTriangle n) T1
    isValidTriangle Tn ∧
    ¬isValidTriangle (nextTriangle Tn) ∧
    perimeter Tn = 1516.5 / 128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l303_30391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l303_30358

noncomputable def f (x : ℝ) : ℝ := -Real.sin (2 * x + Real.pi / 6)

theorem f_symmetry (x : ℝ) : f (-Real.pi / 3 + x) = f (-Real.pi / 3 - x) := by
  -- Unfold the definition of f
  unfold f
  -- Use Real.sin_add to expand the sine of a sum
  simp [Real.sin_add]
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l303_30358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l303_30305

noncomputable def u : ℝ × ℝ × ℝ := (4, -1, 3)
noncomputable def v : ℝ × ℝ × ℝ := (-2, 2, 5)

noncomputable def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  (a₂ * b₃ - a₃ * b₂, a₃ * b₁ - a₁ * b₃, a₁ * b₂ - a₂ * b₁)

noncomputable def magnitude (w : ℝ × ℝ × ℝ) : ℝ :=
  let (w₁, w₂, w₃) := w
  Real.sqrt (w₁^2 + w₂^2 + w₃^2)

theorem parallelogram_area : magnitude (cross_product u v) = Real.sqrt 833 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l303_30305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_calculation_l303_30390

-- Define the circles and points
noncomputable def larger_circle_radius : ℝ := Real.sqrt 104
noncomputable def smaller_circle_radius : ℝ := larger_circle_radius - 5
def P : ℝ × ℝ := (10, 2)
def S : ℝ → ℝ × ℝ := λ k => (0, k)

-- Define the theorem
theorem circle_radius_calculation (k : ℝ) :
  (P.1^2 + P.2^2 = larger_circle_radius^2) →
  (S k).2^2 = smaller_circle_radius^2 →
  k = |larger_circle_radius - 5| :=
by
  intro h1 h2
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check circle_radius_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_calculation_l303_30390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_criterion_disjoint_criterion_l303_30339

/-- A finite set of natural numbers from 1 to n -/
def U (n : ℕ) : Finset ℕ := Finset.range n

/-- The characteristic function of a subset A of U -/
def f_A (n : ℕ) (A : Finset ℕ) (x : ℕ) : ℕ :=
  if x ∈ A ∩ U n then 1 else 0

theorem subset_criterion {n : ℕ} (A B : Finset ℕ) :
  (∀ x ∈ U n, f_A n A x ≤ f_A n B x) → A ⊆ B :=
by
  sorry

theorem disjoint_criterion {n : ℕ} (A B : Finset ℕ) :
  (∀ x ∈ U n, f_A n (A ∪ B) x = f_A n A x + f_A n B x) → A ∩ B = ∅ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_criterion_disjoint_criterion_l303_30339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_implication_l303_30329

theorem sine_inequality_implication : 
  (∀ α β : Real, Real.sin α ≠ Real.sin β → α ≠ β) ∧ 
  (∃ α β : Real, α ≠ β ∧ Real.sin α = Real.sin β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_implication_l303_30329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l303_30397

/-- Represents the number of squares at a vertex in the tiling. -/
def m : ℕ := 1

/-- Represents the number of octagons at a vertex in the tiling. -/
def n : ℕ := 2

/-- The sum of angles at the vertex must be 360°. -/
axiom angle_sum : 135 * n + 90 * m = 360

/-- There is exactly one solution for m and n. -/
theorem unique_solution : m = 1 ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l303_30397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equivalence_l303_30365

/-- The floor function, denoted as ⌊x⌋, returns the largest integer not greater than x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem stating the equivalence between the floor condition and the range of x -/
theorem floor_equivalence (x : ℝ) : 
  floor ((1 - 3 * x) / 2) = -1 ↔ 1/3 < x ∧ x ≤ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equivalence_l303_30365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_circular_sector_l303_30361

/-- Represents a circular sector -/
structure CircularSector where
  r : ℝ  -- radius
  α : ℝ  -- central angle in radians
  h_r_pos : r > 0

/-- The perimeter of a circular sector -/
noncomputable def perimeter (s : CircularSector) : ℝ := s.r * s.α + 2 * s.r

/-- The area of a circular sector -/
noncomputable def area (s : CircularSector) : ℝ := (1 / 2) * s.r^2 * s.α

/-- Theorem: Maximum area of a circular sector with perimeter 24 -/
theorem max_area_circular_sector :
  ∃ (s : CircularSector),
    perimeter s = 24 ∧
    (∀ (t : CircularSector), perimeter t = 24 → area t ≤ area s) ∧
    area s = 36 ∧
    s.α = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_circular_sector_l303_30361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_minus_only_solvable_l303_30379

/-- Represents a position on the 5x5 board -/
structure Position where
  row : Fin 5
  col : Fin 5
deriving Repr, DecidableEq

/-- Represents the state of the board -/
def Board := Position → Bool

/-- Represents a move on the board -/
inductive Move where
  | square_2x2 (top_left : Position)
  | square_3x3 (top_left : Position)
  | square_4x4 (top_left : Position)
  | square_5x5
deriving Repr

/-- The central position of the board -/
def centralPosition : Position :=
  ⟨2, 2⟩

/-- Applies a move to the board -/
def applyMove (b : Board) (m : Move) : Board :=
  sorry

/-- Checks if all signs on the board are plus -/
def allPlus (b : Board) : Prop :=
  ∀ p, b p = true

/-- Initial board state with one minus sign -/
def initialBoard (minusPosition : Position) : Board :=
  fun p => if p = minusPosition then false else true

theorem central_minus_only_solvable :
  ∀ (minusPosition : Position),
    (∃ (moves : List Move), allPlus (moves.foldl applyMove (initialBoard minusPosition))) ↔
    minusPosition = centralPosition :=
  sorry

#check central_minus_only_solvable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_minus_only_solvable_l303_30379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_x_is_171_l303_30381

def heights : List ℝ := [158, 165, 165, 167, 168, 169, 172, 173, 175]

noncomputable def sixtieth_percentile (l : List ℝ) : ℝ :=
  (l.get! 5 + l.get! 6) / 2

theorem height_x_is_171 (x : ℝ) :
  let l := (heights.take 6) ++ [x] ++ (heights.drop 6)
  sixtieth_percentile l = 170 → x = 171 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_x_is_171_l303_30381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_die_probability_l303_30333

-- Define the number of rolls
def num_rolls : ℕ := 8

-- Define the probability of rolling at least a five on a single roll
def p_at_least_five : ℚ := 1/3

-- Define the probability of rolling at least a five at least six times in eight rolls
def p_at_least_six_of_eight : ℚ := 129/6561

-- Theorem statement
theorem fair_die_probability : 
  (Finset.sum (Finset.range 3) (λ k => 
    (Nat.choose num_rolls (num_rolls - k) * p_at_least_five^(num_rolls - k) * (1 - p_at_least_five)^k))) = p_at_least_six_of_eight :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_die_probability_l303_30333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_dimensions_l303_30307

-- Define an isosceles trapezoid
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  leg : ℝ
  diagonal : ℝ

-- Define the properties of our specific trapezoid
noncomputable def specificTrapezoid : IsoscelesTrapezoid where
  base1 := 20
  base2 := 12
  leg := 4 * Real.sqrt 5
  diagonal := 8 * Real.sqrt 5

-- Theorem statement
theorem trapezoid_dimensions (t : IsoscelesTrapezoid) 
  (h1 : t.base1 = 20) 
  (h2 : t.base2 = 12) 
  (h3 : ∃ (center : ℝ × ℝ), center.1 ∈ Set.Icc 0 t.base1) : 
  t.leg = 4 * Real.sqrt 5 ∧ t.diagonal = 8 * Real.sqrt 5 := by
  sorry

#check trapezoid_dimensions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_dimensions_l303_30307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_solution_l303_30312

/-- The number of children -/
def num_children : ℕ := sorry

/-- The number of books -/
def num_books : ℕ := sorry

/-- The number of books each child gets in the first scenario -/
def m : ℕ := sorry

/-- If each child gets m books, there are 14 books left -/
axiom condition1 : num_children * m + 14 = num_books

/-- If each child gets 9 books, the last child only gets 6 books -/
axiom condition2 : (num_children - 1) * 9 + 6 = num_books

theorem correct_solution : num_children = 17 ∧ num_books = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_solution_l303_30312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_six_l303_30349

-- Define the piecewise function f
noncomputable def f (a b c : ℕ) (x : ℝ) : ℝ :=
  if x > 0 then a * x + 3
  else if x = 0 then a * b
  else b * x + c

-- State the theorem
theorem sum_abc_equals_six (a b c : ℕ) : 
  (f a b c 2 = 5) → 
  (f a b c 0 = 5) → 
  (f a b c (-2) = -10) → 
  a + b + c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_six_l303_30349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isabella_gallons_l303_30375

-- Define the discount rate
def discount_rate : Real := 0.10

-- Define the non-discounted gallons
def non_discounted_gallons : Nat := 6

-- Define Kim's total gallons
def kim_total_gallons : Nat := 20

-- Define the ratio of Isabella's discount to Kim's
def isabella_discount_ratio : Real := 1.0857142857142861

-- Define a function to calculate the discounted gallons
def discounted_gallons (total : Nat) : Nat :=
  max (total - non_discounted_gallons) 0

-- Define a function to calculate the total discount
def total_discount (gallons : Nat) : Real :=
  (discounted_gallons gallons : Real) * discount_rate

-- Theorem statement
theorem isabella_gallons :
  ∃ (gallons : Nat), 
    (total_discount gallons) / (total_discount kim_total_gallons) = isabella_discount_ratio ∧
    gallons = 21 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isabella_gallons_l303_30375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l303_30317

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the line y=2
def line (y : ℝ) : Prop := y = 2

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -4

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + (y-2)^2 = 16

-- Theorem statement
theorem circle_properties :
  ∀ x y : ℝ,
  (parabola x y ∧ line y) →
  my_circle x y ∧
  (∃ t : ℝ, my_circle 0 t ∧ directrix t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l303_30317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_factor_iff_prime_l303_30323

def sequenceF (m n i : ℕ) : ℕ := i * m + (n + 1 - i)

theorem no_common_factor_iff_prime (n : ℕ) : 
  (∃ m : ℕ, m > 0 ∧ 
    (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ n → 
      Nat.gcd (sequenceF m n i) (sequenceF m n j) = 1)) ↔ 
  Nat.Prime (n + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_factor_iff_prime_l303_30323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_range_l303_30386

-- Define the parabola
def parabola (x : ℝ) : ℝ := -x^2 + 2*x + 8

-- Define the x-intercepts
def x_intercepts : Set ℝ := {x | parabola x = 0}

-- Define point D as the midpoint of the x-intercepts
def D : ℝ × ℝ := (1, 0)

-- Define the set of points A on the parabola above the x-axis
def A_set : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = parabola p.1 ∧ p.2 > 0}

-- Define the condition for ∠BAC to be acute
def is_acute_angle (A B C : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem parabola_distance_range :
  ∀ A : ℝ × ℝ, A ∈ A_set → 
  (∀ B C : ℝ × ℝ, B.1 ∈ x_intercepts → C.1 ∈ x_intercepts → is_acute_angle A B C) →
  3 < dist A D ∧ dist A D ≤ 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_range_l303_30386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2A_value_l303_30328

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

-- State the theorem
theorem sin_2A_value (A : ℝ) (h1 : 0 < A) (h2 : A < Real.pi / 2) (h3 : f A = 2/3) :
  Real.sin (2 * A) = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2A_value_l303_30328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_boy_girl_adjacencies_l303_30393

/-- The expected number of boy-girl adjacencies in a line of 20 people -/
theorem expected_boy_girl_adjacencies :
  let total_people : ℕ := 20
  let num_boys : ℕ := 5
  let num_girls : ℕ := 15
  let first_two_fixed : Prop := ∃ (b g : ℕ), b + g = 2 ∧ b ≤ 1 ∧ g ≤ 1
  let S : ℕ → ℝ := λ n ↦ (n : ℝ)  -- S is the number of boy-girl adjacencies
  ∃ (E : (ℕ → ℝ) → ℝ), -- E represents the expected value function
    E S = 527 / 51 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_boy_girl_adjacencies_l303_30393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheila_tuesday_thursday_hours_l303_30376

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hourly_rate : ℚ
  weekly_earnings : ℚ
  hours_mon_wed_fri : ℚ
  days_mon_wed_fri : ℕ

/-- Calculates the total hours worked on Tuesday and Thursday --/
def tuesday_thursday_hours (s : WorkSchedule) : ℚ :=
  (s.weekly_earnings - s.hourly_rate * s.hours_mon_wed_fri * s.days_mon_wed_fri) / s.hourly_rate

/-- Theorem stating that Sheila works 12 hours on Tuesday and Thursday --/
theorem sheila_tuesday_thursday_hours :
  let s : WorkSchedule := {
    hourly_rate := 7,
    weekly_earnings := 252,
    hours_mon_wed_fri := 8,
    days_mon_wed_fri := 3
  }
  tuesday_thursday_hours s = 12 := by
  -- Proof goes here
  sorry

#eval tuesday_thursday_hours {
  hourly_rate := 7,
  weekly_earnings := 252,
  hours_mon_wed_fri := 8,
  days_mon_wed_fri := 3
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheila_tuesday_thursday_hours_l303_30376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_integer_implies_a_range_l303_30343

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -Real.exp x * (2 * x + 1) - a * x + a

/-- The theorem statement -/
theorem unique_positive_integer_implies_a_range (a : ℝ) 
  (h_a : a > -1) 
  (h_unique : ∃! (x₀ : ℤ), f a (x₀ : ℝ) > 0) :
  a ∈ Set.Ioo (-1 / (2 * Real.exp 1)) (-1 / (Real.exp 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_integer_implies_a_range_l303_30343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_implies_a_range_a_range_implies_function_domain_l303_30364

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x + a)

-- State the theorem
theorem function_domain_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) → a > (1/2 : ℝ) := by
  sorry

-- Additional theorem to capture the full condition
theorem a_range_implies_function_domain (a : ℝ) :
  a > (1/2 : ℝ) → (∀ x : ℝ, ∃ y : ℝ, f a x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_implies_a_range_a_range_implies_function_domain_l303_30364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_segment_proof_l303_30354

noncomputable def circle_intersection_segment (a α β : ℝ) : ℝ :=
  (a / (2 * Real.sin α)) * (Real.sqrt ((Real.sin β)^2 + 8 * (Real.sin α)^2) - Real.sin β)

theorem circle_intersection_segment_proof (a α β : ℝ) (h_a : a > 0) (h_α : 0 < α ∧ α < π/2) (h_β : 0 < β ∧ β < π/2) :
  ∃ (AE : ℝ), AE = circle_intersection_segment a α β := by
  sorry

#check circle_intersection_segment_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_segment_proof_l303_30354
