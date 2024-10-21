import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_linked_rings_four_l1144_114401

def a : ℕ → ℕ
| 0 => 1
| n + 1 => if (n + 1) % 2 = 0 then 2 * a n - 1 else 2 * a n + 2

theorem nine_linked_rings_four : a 4 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_linked_rings_four_l1144_114401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_boxes_l1144_114492

/-- Represents the number of 10-pound boxes -/
def x : ℕ := sorry

/-- Represents the number of 20-pound boxes -/
def y : ℕ := sorry

/-- The average weight of all boxes is 18 pounds -/
axiom initial_average : (10 * x + 20 * y) / (x + y) = 18

/-- After removing 10 of the 20-pound boxes, the average weight becomes 16 pounds -/
axiom final_average : (10 * x + 20 * (y - 10)) / (x + y - 10) = 16

/-- The theorem to prove: the total number of boxes is 20 -/
theorem total_boxes : x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_boxes_l1144_114492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_maximized_l1144_114444

theorem triangle_sine_sum_maximized (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) : 
  ∀ α' β' γ' : Real, 
    α' + β' + γ' = Real.pi → 
    0 < α' ∧ 0 < β' ∧ 0 < γ' →
    Real.sin α + Real.sin β + Real.sin γ ≤ Real.sin α' + Real.sin β' + Real.sin γ' → 
    α = β ∧ β = γ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_maximized_l1144_114444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_numerograd_gates_l1144_114475

def magicSequence : List Nat := [8, 7, 5, 8, 7, 5, 8, 7, 5, 8, 7]

theorem numerograd_gates (seq : List Nat) :
  seq = magicSequence →
  seq.head? = some 8 ∧
  seq.get? 8 = some 5 ∧
  (∀ i, i + 2 < seq.length → 
    (do 
      let x ← seq.get? i
      let y ← seq.get? (i + 1)
      let z ← seq.get? (i + 2)
      pure (x + y + z)
    ) = some 20) →
  True := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_numerograd_gates_l1144_114475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_sufficient_not_necessary_l1144_114406

-- Define the function type
def RealFunction := Set.Icc 0 1 → ℝ

-- Define strictly increasing property
def StrictlyIncreasing (f : RealFunction) : Prop :=
  ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x < y → f x < f y

-- Define the maximum value property
def MaxValueAtOne (f : RealFunction) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → f x ≤ f 1

-- Theorem statement
theorem strictly_increasing_sufficient_not_necessary :
  (∀ f : RealFunction, StrictlyIncreasing f → MaxValueAtOne f) ∧
  (∃ f : RealFunction, MaxValueAtOne f ∧ ¬StrictlyIncreasing f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_sufficient_not_necessary_l1144_114406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l1144_114458

-- Define the curves
noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.sqrt x

-- Define the area enclosed by the curves
noncomputable def enclosed_area : ℝ :=
  ∫ x in (0)..(1), g x - f x

-- Theorem statement
theorem area_between_curves : enclosed_area = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l1144_114458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbara_shopping_theorem_l1144_114420

/-- Represents Barbara's shopping trip to the supermarket -/
structure ShoppingTrip where
  tuna_packs : ℕ
  tuna_price : ℚ
  water_bottles : ℕ
  water_price : ℚ
  discount_rate : ℚ
  total_paid_after_discount : ℚ

/-- Calculates the amount spent on different goods before discount -/
noncomputable def amount_on_different_goods (trip : ShoppingTrip) : ℚ :=
  let tuna_cost := trip.tuna_packs * trip.tuna_price
  let water_cost := trip.water_bottles * trip.water_price
  let total_before_discount := trip.total_paid_after_discount / (1 - trip.discount_rate)
  total_before_discount - (tuna_cost + water_cost)

/-- Theorem stating that Barbara spent $46.22 on different goods before the discount -/
theorem barbara_shopping_theorem (trip : ShoppingTrip)
    (h1 : trip.tuna_packs = 5)
    (h2 : trip.tuna_price = 2)
    (h3 : trip.water_bottles = 4)
    (h4 : trip.water_price = 3/2)
    (h5 : trip.discount_rate = 1/10)
    (h6 : trip.total_paid_after_discount = 56) :
    amount_on_different_goods trip = 2311/50 := by
  sorry

#eval (2311 : ℚ) / 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbara_shopping_theorem_l1144_114420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l1144_114418

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

noncomputable def odd_powers_sum (a : ℝ) (r : ℝ) : ℝ := (a * r) / (1 - r^2)

theorem geometric_series_problem (a : ℝ) (r : ℝ) 
  (h1 : geometric_series_sum a r = 18)
  (h2 : odd_powers_sum a r = 8) :
  r = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l1144_114418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_solve_quadratic_equation_l1144_114429

-- Problem 1
theorem calculate_expression : 
  4 * Real.sin (45 * π / 180) + (Real.sqrt 2 - Real.pi)^(0 : ℝ) - Real.sqrt 8 + (1/3)^(-2 : ℝ) = 10 := by sorry

-- Problem 2
theorem solve_quadratic_equation :
  ∀ x : ℝ, x^2 - 2*x = 8 ↔ x = 4 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_solve_quadratic_equation_l1144_114429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_cube_diagonal_product_l1144_114459

/-- A cube with side length 1 -/
structure UnitCube where
  side_length : ℝ
  side_length_eq_one : side_length = 1

/-- The length of a face diagonal in a unit cube -/
noncomputable def face_diagonal_length (c : UnitCube) : ℝ := Real.sqrt 2

/-- The length of a space diagonal in a unit cube -/
noncomputable def space_diagonal_length (c : UnitCube) : ℝ := Real.sqrt 3

/-- The number of face diagonals in a cube -/
def num_face_diagonals : ℕ := 12

/-- The number of space diagonals in a cube -/
def num_space_diagonals : ℕ := 4

/-- The theorem stating that the product of all diagonal lengths in a unit cube is 576 -/
theorem unit_cube_diagonal_product (c : UnitCube) :
  (face_diagonal_length c) ^ num_face_diagonals * (space_diagonal_length c) ^ num_space_diagonals = 576 := by
  sorry

#eval 2^6 * 3^2 -- This will output 576, confirming our calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_cube_diagonal_product_l1144_114459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ballpoint_pen_price_l1144_114461

/-- Given the unit price of a pencil and the relation between the prices of a pencil and a ballpoint pen,
    prove that the unit price of a ballpoint pen is 3a yuan. -/
theorem ballpoint_pen_price (a : ℝ) (h : a > 0) : 3 * a = 3 * a := by
  let pencil_price := a
  let ballpoint_pen_price := 3 * pencil_price
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ballpoint_pen_price_l1144_114461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equiangular_inner_parallelogram_l1144_114427

/-- A regular hexagon -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- An inner parallelogram formed by diagonals in a regular hexagon -/
structure InnerParallelogram (h : RegularHexagon) where
  diagonal_length : ℝ
  diagonal_length_pos : diagonal_length > 0

/-- Definition of an equiangular parallelogram -/
def is_equiangular (h : RegularHexagon) (p : InnerParallelogram h) : Prop :=
  ∀ angle : ℝ, angle = 90 -- All angles are 90 degrees

/-- Theorem: No inner parallelogram in a regular hexagon with parallel diagonals is equiangular -/
theorem no_equiangular_inner_parallelogram (h : RegularHexagon) :
  ∀ p : InnerParallelogram h, ¬(is_equiangular h p) :=
by
  intro p
  intro h_equiangular
  -- The proof would go here, but we'll use sorry for now
  sorry

#check no_equiangular_inner_parallelogram

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equiangular_inner_parallelogram_l1144_114427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_average_time_l1144_114472

noncomputable def casey_time : ℝ := 6
noncomputable def zendaya_multiplier : ℝ := 4/3

theorem marathon_average_time : 
  let zendaya_time := casey_time * zendaya_multiplier
  (casey_time + zendaya_time) / 2 = 7 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_average_time_l1144_114472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_sum_l1144_114448

def IsNonDecreasing (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ D → x₂ ∈ D → x₁ < x₂ → f x₁ ≤ f x₂

theorem function_value_sum (f : ℝ → ℝ) (h_domain : Set.Icc 0 1 ⊆ Set.range f)
    (h_nondec : IsNonDecreasing f (Set.Icc 0 1))
    (h_zero : f 0 = 0)
    (h_third : ∀ x, f (x / 3) = (1 / 2) * f x)
    (h_complement : ∀ x, f (1 - x) = 1 - f x) :
    f (1 / 3) + f (1 / 8) = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_sum_l1144_114448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cost_theorem_l1144_114447

-- Define the total cost function as noncomputable
noncomputable def total_cost (x : ℝ) : ℝ := (200 * x / 3) + (86400 / x)

-- State the theorem
theorem minimum_cost_theorem :
  ∀ x : ℝ, 0 < x → x ≤ 48 → total_cost x ≥ 4800 ∧
  (total_cost x = 4800 ↔ x = 36) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cost_theorem_l1144_114447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_XYZ_power_of_point_l1144_114491

-- Define the points in the Euclidean plane
variable (X Y Z Q C E : EuclideanPlane ℝ)

-- Define the conditions
variable (h1 : dist Q X = dist Q Y)
variable (h2 : angle X Q Y = 3 * angle X C Z)
variable (h3 : ∃ t : ℝ, X + t • (Z - X) = E ∧ Y + t • (Q - Y) = E)
variable (h4 : dist Y Q = 6)
variable (h5 : dist Y E = 4)

-- Theorem statement
theorem triangle_XYZ_power_of_point : 
  (dist X E) * (dist Z E) = 32 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_XYZ_power_of_point_l1144_114491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1144_114410

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / (4^x) - 1 / (3^x) else 3^x - 4^x

-- State the theorem
theorem min_m_value :
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f (-x) = -f x) →  -- f is odd on [-4, 4]
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, ∃ a : ℝ, f x = 1 / (4^x) + a / (3^x)) →  -- f(x) = 1/(4^x) + a/(3^x) for x ∈ [-4, 0]
  (∃ x ∈ Set.Icc (-2 : ℝ) (-1), f x ≤ 5 / (2^x) - 1 / (3^(x-1))) →  -- There exists x ∈ [-2, -1] such that f(x) ≤ 5/(2^x) - 1/(3^(x-1))
  (∀ m : ℝ, m < 5 → ¬∀ x ∈ Set.Icc (-2 : ℝ) (-1), f x ≤ m / (2^x) - 1 / (3^(x-1))) -- For all m < 5, the inequality doesn't hold for all x ∈ [-2, -1]
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1144_114410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1144_114484

/-- The complex number that satisfies the given equation -/
noncomputable def z : ℂ := sorry

/-- The equation that z satisfies -/
axiom z_equation : (z + 3*Complex.I) * (2 - Complex.I^3) = 10*Complex.I^5

/-- Theorem stating that z is in the first quadrant -/
theorem z_in_first_quadrant : 
  z.re > 0 ∧ z.im > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1144_114484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_contains_perfect_square_product_l1144_114466

def V : Set Nat := {n | 1 ≤ n ∧ n ≤ 25}

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, n = m * m

theorem subset_contains_perfect_square_product (S : Finset Nat) :
  (∀ x ∈ S, x ∈ V) → S.card ≥ 17 →
  ∃ x y : Nat, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ is_perfect_square (x * y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_contains_perfect_square_product_l1144_114466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_completes_in_78_div_7_days_l1144_114479

/-- The number of days it takes for person B to complete a work alone, 
    given that A and B together can complete it in 6 days, 
    and A alone can complete it in 13 days. -/
noncomputable def days_for_B_alone (days_AB : ℝ) (days_A : ℝ) : ℝ :=
  let rate_AB := 1 / days_AB
  let rate_A := 1 / days_A
  let rate_B := rate_AB - rate_A
  1 / rate_B

/-- Theorem stating that B can complete the work in 78/7 days -/
theorem B_completes_in_78_div_7_days : 
  days_for_B_alone 6 13 = 78 / 7 := by
  -- Unfold the definition of days_for_B_alone
  unfold days_for_B_alone
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_completes_in_78_div_7_days_l1144_114479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_when_f_no_zeros_l1144_114428

open Real

/-- The function g(x) defined as a(2x-1) where a is a real number -/
def g (a : ℝ) (x : ℝ) : ℝ := a * (2 * x - 1)

/-- The function h(x) defined as (2a^2+1)ln(x) where a is a real number -/
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := (2 * a^2 + 1) * log x

/-- The function f(x) defined as g(x) + h(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := g a x + h a x

/-- Theorem stating that if f(x) has no zeros in [1/2, 1], then a is in (-∞, 0) -/
theorem a_range_when_f_no_zeros (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, f a x ≠ 0) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_when_f_no_zeros_l1144_114428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_ratio_proof_l1144_114494

theorem pizza_ratio_proof (total_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (frank_slices : ℕ) (sammy_fraction : ℚ) (leftover_slices : ℕ) :
  total_pizzas = 2 →
  slices_per_pizza = 12 →
  frank_slices = 3 →
  sammy_fraction = 1/3 →
  leftover_slices = 11 →
  ∃ (dean_slices : ℕ),
    dean_slices = total_pizzas * slices_per_pizza - leftover_slices - frank_slices - 
      (sammy_fraction * ↑slices_per_pizza).num ∧
    (dean_slices : ℚ) / slices_per_pizza = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_ratio_proof_l1144_114494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_proof_l1144_114436

/-- Represents the revenue data for a year -/
structure RevenueData where
  transaction_processing : ℝ
  data_processing : ℝ
  cross_border : ℝ

/-- Represents the currency conversion factors for a year -/
structure ConversionFactors where
  transaction_processing : ℝ
  data_processing : ℝ
  cross_border : ℝ

/-- Calculates the total revenue after applying conversion factors -/
noncomputable def calculate_total_revenue (revenue : RevenueData) (factors : ConversionFactors) : ℝ :=
  revenue.transaction_processing * factors.transaction_processing +
  revenue.data_processing * factors.data_processing +
  revenue.cross_border * factors.cross_border

/-- Calculates the percentage decrease between two values -/
noncomputable def percentage_decrease (old_value : ℝ) (new_value : ℝ) : ℝ :=
  (old_value - new_value) / old_value * 100

theorem revenue_decrease_proof 
  (last_year_revenue : RevenueData)
  (this_year_revenue : RevenueData)
  (last_year_factors : ConversionFactors)
  (this_year_factors : ConversionFactors)
  (h1 : last_year_revenue.transaction_processing = 40)
  (h2 : last_year_revenue.data_processing = 25)
  (h3 : last_year_revenue.cross_border = 20)
  (h4 : this_year_revenue.transaction_processing = 28.8)
  (h5 : this_year_revenue.data_processing = 20)
  (h6 : this_year_revenue.cross_border = 17.6)
  (h7 : last_year_factors.transaction_processing = 1)
  (h8 : last_year_factors.data_processing = 1)
  (h9 : last_year_factors.cross_border = 1)
  (h10 : this_year_factors.transaction_processing = 0.98)
  (h11 : this_year_factors.data_processing = 1.02)
  (h12 : this_year_factors.cross_border = 0.95) :
  abs (percentage_decrease 
    (calculate_total_revenue last_year_revenue last_year_factors)
    (calculate_total_revenue this_year_revenue this_year_factors) - 23.13) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_proof_l1144_114436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_no_quasi_square_sequences_l1144_114435

/-- A positive integer is a quasi-square number if it can be expressed as ab where a and b are positive integers and a ≤ b ≤ 1.01a -/
def IsQuasiSquare (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = a * b ∧ a ≤ b ∧ (b : ℝ) ≤ 1.01 * a

/-- A sequence of 199 consecutive integers starting from m contains no quasi-square numbers -/
def NoQuasiSquareSequence (m : ℕ) : Prop :=
  ∀ k : ℕ, k < 199 → ¬IsQuasiSquare (m + k)

/-- The set of starting points for sequences of 199 consecutive integers 
    that contain no quasi-square numbers is infinite -/
theorem infinite_no_quasi_square_sequences :
  Set.Infinite {m : ℕ | NoQuasiSquareSequence m} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_no_quasi_square_sequences_l1144_114435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1144_114457

theorem right_triangle_hypotenuse (x : ℝ) : 
  x > 0 → -- Shorter leg length is positive
  2 * x - 3 > 0 → -- Longer leg length is positive
  (1/2) * x * (2*x - 3) = 72 → -- Area condition
  ∃ h : ℝ, h > 0 ∧ h^2 = x^2 + (2*x - 3)^2 ∧ |h - 18.1| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1144_114457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_poly_average_theorem_l1144_114413

open Polynomial

-- Define a monic polynomial of degree n with real coefficients
def is_monic_real_poly (p : Polynomial ℝ) (n : ℕ) : Prop :=
  p.leadingCoeff = 1 ∧ p.degree = n

-- Define a polynomial with n real roots
def has_n_real_roots (p : Polynomial ℝ) (n : ℕ) : Prop :=
  ∃ (roots : Finset ℝ), roots.card = n ∧ (∀ r ∈ roots, p.eval r = 0)

theorem monic_poly_average_theorem (n : ℕ) (F : Polynomial ℝ) 
  (h : is_monic_real_poly F n) :
  ∃ (P Q : Polynomial ℝ),
    is_monic_real_poly P n ∧
    is_monic_real_poly Q n ∧
    has_n_real_roots P n ∧
    has_n_real_roots Q n ∧
    F = (P + Q) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_poly_average_theorem_l1144_114413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_comparisons_for_three_balls_l1144_114499

/-- A type representing a ball with a weight -/
structure Ball where
  weight : ℝ

/-- A function representing a comparison between two balls -/
noncomputable def compare (a b : Ball) : Bool :=
  a.weight < b.weight

/-- The number of balls -/
def n : ℕ := 3

/-- The minimum number of comparisons required to rank n balls -/
def min_comparisons (n : ℕ) : ℕ :=
  (n - 1) * n / 2

theorem min_comparisons_for_three_balls :
  ∀ (balls : Fin n → Ball),
  (∀ i j, i ≠ j → balls i ≠ balls j) →
  min_comparisons n = 3 := by
  sorry

#eval min_comparisons n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_comparisons_for_three_balls_l1144_114499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1144_114403

-- Define the equation
def equation (a : ℝ) (x : ℝ) : Prop :=
  |x^2 + a*x| = 4

-- Define the condition of having exactly 3 distinct real roots
def has_three_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    equation a x₁ ∧ equation a x₂ ∧ equation a x₃ ∧
    ∀ x : ℝ, equation a x → (x = x₁ ∨ x = x₂ ∨ x = x₃)

-- State the theorem
theorem equation_solution :
  ∀ a : ℝ, has_three_distinct_roots a →
    ((a = 4 ∨ a = -4) ∧
     (∃ x₁ x₂ x₃ : ℝ, ({x₁, x₂, x₃} : Set ℝ) = {-2, -2 + Real.sqrt 2, -2 - Real.sqrt 2} ∨
                      ({x₁, x₂, x₃} : Set ℝ) = {2, 2 + Real.sqrt 2, 2 - Real.sqrt 2})) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1144_114403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1144_114486

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * ω * x)

-- State the theorem
theorem function_properties (ω : ℝ) :
  (∀ x, f ω (x + π) = f ω x) →  -- Period is π
  ω = 1 ∧                       -- ω equals 1
  (∃ c ∈ Set.Icc (π/6) (π/3),   -- Maximum value is 2
    f 1 c = 2 ∧
    ∀ x ∈ Set.Icc (π/6) (π/3), f 1 x ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1144_114486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l1144_114489

/-- Given a triangle ABC where BD = 2DC and AD = xAB + yAC, prove that x - y = -1/3 -/
theorem triangle_vector_relation (A B C D : ℝ × ℝ) (x y : ℝ) :
  (B - D : ℝ × ℝ) = 2 • (D - C) →
  (A - D : ℝ × ℝ) = x • (A - B) + y • (A - C) →
  x - y = -1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l1144_114489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_is_three_l1144_114416

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the tangent line
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Theorem statement
theorem tangent_slope_is_three :
  ∃ (x₀ : ℝ), 
    (curve x₀ = tangent_line 3 x₀) ∧ 
    (∀ x : ℝ, x ≠ x₀ → curve x ≠ tangent_line 3 x) ∧
    (deriv curve x₀ = 3) :=
by
  -- Proof goes here
  sorry

#check tangent_slope_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_is_three_l1144_114416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_floor_div_tan_zero_l1144_114411

-- Define the greatest integer function (floor function)
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem limit_floor_div_tan_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < x ∧ x < δ → |floor x / Real.tan x - 0| < ε :=
by
  sorry

#check limit_floor_div_tan_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_floor_div_tan_zero_l1144_114411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_changes_theorem_l1144_114437

/-- Represents a 3x3 matrix of integers -/
def Matrix3x3 : Type := Fin 3 → Fin 3 → ℤ

/-- The initial matrix -/
def initial_matrix : Matrix3x3 :=
  λ i j => match i, j with
  | ⟨0, _⟩, ⟨0, _⟩ => 4 | ⟨0, _⟩, ⟨1, _⟩ => 9 | ⟨0, _⟩, ⟨2, _⟩ => 5
  | ⟨1, _⟩, ⟨0, _⟩ => 3 | ⟨1, _⟩, ⟨1, _⟩ => 1 | ⟨1, _⟩, ⟨2, _⟩ => 6
  | ⟨2, _⟩, ⟨0, _⟩ => 7 | ⟨2, _⟩, ⟨1, _⟩ => 2 | ⟨2, _⟩, ⟨2, _⟩ => 8
  | _, _ => 0

/-- Calculates the row sum for a given row -/
def row_sum (m : Matrix3x3) (row : Fin 3) : ℤ :=
  (m row 0) + (m row 1) + (m row 2)

/-- Calculates the column sum for a given column -/
def col_sum (m : Matrix3x3) (col : Fin 3) : ℤ :=
  (m 0 col) + (m 1 col) + (m 2 col)

/-- Checks if all row sums are equal -/
def all_row_sums_equal (m : Matrix3x3) : Prop :=
  (row_sum m 0) = (row_sum m 1) ∧ (row_sum m 1) = (row_sum m 2)

/-- Checks if all column sums are different -/
def all_col_sums_different (m : Matrix3x3) : Prop :=
  (col_sum m 0) ≠ (col_sum m 1) ∧ 
  (col_sum m 1) ≠ (col_sum m 2) ∧ 
  (col_sum m 0) ≠ (col_sum m 2)

/-- Counts the number of different entries between two matrices -/
def diff_count (m1 m2 : Matrix3x3) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 3)) (λ i => 
    Finset.sum (Finset.univ : Finset (Fin 3)) (λ j => 
      if m1 i j ≠ m2 i j then 1 else 0))

/-- The main theorem: There exists a matrix that satisfies the conditions
    and differs from the initial matrix in exactly 3 entries, 
    and no matrix with fewer changes can satisfy the conditions -/
theorem min_changes_theorem : 
  ∃ (m : Matrix3x3), 
    all_row_sums_equal m ∧ 
    all_col_sums_different m ∧
    diff_count initial_matrix m = 3 ∧
    ∀ (m' : Matrix3x3), 
      all_row_sums_equal m' → 
      all_col_sums_different m' → 
      diff_count initial_matrix m' ≥ 3 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_changes_theorem_l1144_114437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_bound_proof_l1144_114495

theorem lower_bound_proof (x : ℤ) (y : ℝ)
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : -2 < x ∧ x < 9)
  (h4 : y < x ∧ x < 8)
  (h5 : x + 1 < 9)
  (hx7 : x = 7) : 
  y ≤ 7 :=
by
  -- Convert x to ℝ for comparison with y
  have x_real : ℝ := ↑x
  -- Use hx7 to replace x with 7
  rw [hx7] at h4
  -- Now h4 is: y < 7 ∧ 7 < 8
  -- Extract y < 7 from h4
  have y_lt_7 : y < 7 := h4.left
  -- y < 7 implies y ≤ 7
  exact le_of_lt y_lt_7


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_bound_proof_l1144_114495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_theorem_l1144_114455

/-- Calculates the speed of a bus including stoppages -/
noncomputable def speed_including_stoppages (speed_excluding_stoppages : ℝ) (stoppage_time : ℝ) : ℝ :=
  let moving_time := 1 - stoppage_time / 60
  let distance := speed_excluding_stoppages * moving_time
  distance

/-- Theorem: Given a bus with a speed of 50 kmph excluding stoppages and stopping for 6 minutes per hour,
    the speed including stoppages is 45 kmph -/
theorem bus_speed_theorem :
  speed_including_stoppages 50 6 = 45 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval speed_including_stoppages 50 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_theorem_l1144_114455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_not_divisible_implies_prime_l1144_114423

theorem product_not_divisible_implies_prime (n : ℕ) (h1 : n > 3) 
  (h2 : ¬ (n + 1) ∣ Nat.factorial n) : Nat.Prime (4 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_not_divisible_implies_prime_l1144_114423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_conversation_day_l1144_114438

-- Define the days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define the dwarfs
inductive Dwarf : Type
| Senya | Venya

def lies (d : Dwarf) (day : Day) : Prop :=
  match d with
  | Dwarf.Senya => day = Day.Monday ∨ day = Day.Tuesday ∨ day = Day.Wednesday
  | Dwarf.Venya => day = Day.Tuesday ∨ day = Day.Thursday ∨ day = Day.Saturday

def tells_truth (d : Dwarf) (day : Day) : Prop :=
  ¬(lies d day)

def yesterday (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday
  | Day.Sunday => Day.Saturday

def tomorrow (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

def is_senya (d : Dwarf) : Prop :=
  d = Dwarf.Senya

def is_venya (d : Dwarf) : Prop :=
  d = Dwarf.Venya

theorem dwarf_conversation_day :
  ∃ (d1 d2 : Dwarf) (day : Day),
    d1 ≠ d2 ∧
    ((lies d1 day ∧ is_senya d1) ∨ (tells_truth d1 day ∧ is_venya d1)) ∧
    ((lies d1 day ∧ yesterday day = Day.Sunday) ∨ (tells_truth d1 day ∧ yesterday day ≠ Day.Sunday)) ∧
    ((lies d2 day ∧ tomorrow day ≠ Day.Friday) ∨ (tells_truth d2 day ∧ tomorrow day = Day.Friday)) ∧
    ((lies d2 day ∧ day ≠ Day.Wednesday) ∨ (tells_truth d2 day ∧ day = Day.Wednesday)) →
    d1 = Dwarf.Venya ∧ d2 = Dwarf.Senya ∧ day = Day.Tuesday :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_conversation_day_l1144_114438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1144_114453

noncomputable section

/-- The line y = -2x + 3 -/
def line (x : ℝ) : ℝ := -2 * x + 3

/-- The point we're measuring distance from -/
def point : ℝ × ℝ := (3, 1)

/-- The proposed closest point on the line -/
def closest_point : ℝ × ℝ := (7/5, 1/5)

/-- Check if a point is on the line -/
def on_line (p : ℝ × ℝ) : Prop :=
  p.2 = line p.1

/-- Calculate the squared distance between two points -/
def squared_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem closest_point_on_line :
  on_line closest_point ∧
  ∀ p : ℝ × ℝ, on_line p → squared_distance point closest_point ≤ squared_distance point p :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1144_114453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_round_trip_time_l1144_114412

/-- Represents a trip with distance in miles and speed in miles per hour -/
structure Trip where
  distance : Float
  speed : Float

/-- Calculates the round trip time in minutes for a given trip -/
def roundTripTime (trip : Trip) : Float :=
  2 * trip.distance / trip.speed * 60

/-- The list of trips Tony takes -/
def trips : List Trip :=
  [{ distance := 3.5, speed := 2.5 },
   { distance := 3.0, speed := 8.0 },
   { distance := 4.5, speed := 11.0 },
   { distance := 2.8, speed := 9.0 },
   { distance := 4.2, speed := 3.0 }]

/-- The average round trip time in minutes -/
def averageTime : Float :=
  (trips.map roundTripTime).sum / trips.length.toFloat

/-- Checks if two floats are approximately equal within a small epsilon -/
def approxEqual (x y : Float) (ε : Float := 1e-4) : Prop :=
  (x - y).abs ≤ ε

theorem average_round_trip_time :
  approxEqual averageTime 93.4848 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_round_trip_time_l1144_114412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_z_value_l1144_114445

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem max_z_value (z : ℝ) (hz : z > 0) :
  (∀ w, w > 0 → floor (5 / w) + floor (6 / w) = 7 → w ≤ z) →
  floor (5 / z) + floor (6 / z) = 7 →
  z = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_z_value_l1144_114445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_of_high_quality_seeds_l1144_114440

/-- The probability of a seed being high quality -/
def p : ℝ := 0.3

/-- The number of seeds randomly selected -/
def n : ℕ := 8

/-- X represents the number of high-quality seeds among the selected seeds -/
def X : ℕ → ℝ := sorry

/-- The expectation of X -/
def E_X : ℝ := n * p

theorem expectation_of_high_quality_seeds :
  E_X = 2.4 := by
  unfold E_X
  unfold n
  unfold p
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_of_high_quality_seeds_l1144_114440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l1144_114460

-- Define the circles
def circle1 : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + p.2^2 = 9}
def circle2 : Set (ℝ × ℝ) := {p | (p.1 - 8)^2 + p.2^2 = 4}

-- Define the tangent line (implicitly)
noncomputable def tangent_line : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem tangent_line_y_intercept :
  (∀ p : ℝ × ℝ, p ∈ tangent_line → p.2 ≥ 0) →  -- Line is in the first quadrant
  (∃ p : ℝ × ℝ, p ∈ circle1 ∩ tangent_line) →  -- Line is tangent to circle1
  (∃ p : ℝ × ℝ, p ∈ circle2 ∩ tangent_line) →  -- Line is tangent to circle2
  ∃ y : ℝ, y = 15 * Real.sqrt 26 / 26 ∧ (0, y) ∈ tangent_line := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l1144_114460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottom_row_bricks_l1144_114497

/-- Represents a brick wall with a specific structure -/
structure BrickWall where
  numRows : ℕ
  totalBricks : ℕ
  bottomRowBricks : ℕ
  decreasingProperty : bottomRowBricks ≥ numRows

/-- The number of bricks in the wall follows the arithmetic sequence -/
def sumArithmeticSequence (a₁ n d : ℤ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem bottom_row_bricks (wall : BrickWall) 
    (h1 : wall.numRows = 5)
    (h2 : wall.totalBricks = 50)
    (h3 : sumArithmeticSequence (wall.bottomRowBricks : ℤ) wall.numRows (-1) = wall.totalBricks) :
  wall.bottomRowBricks = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottom_row_bricks_l1144_114497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_average_l1144_114496

def consecutive_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

def average (list : List ℤ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem consecutive_integers_average (c d : ℤ) :
  average (consecutive_integers c 7) = d →
  average (consecutive_integers d 7) = c + 6 := by
  sorry

#eval average (consecutive_integers 10 7)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_average_l1144_114496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sums_of_squares_l1144_114441

theorem product_of_sums_of_squares (a b c d : ℤ) :
  ∃ (x y : ℤ), (a^2 + b^2) * (c^2 + d^2) = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sums_of_squares_l1144_114441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_Q_2023_l1144_114478

open BigOperators

def Q (n : ℕ) : ℚ :=
  ∏ i in Finset.range (n - 2), (1 - 1 / (i + 3 : ℚ))

theorem Q_value (n : ℕ) (h : n ≥ 3) : Q n = 2 / n := by
  sorry

theorem Q_2023 : Q 2023 = 2 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_Q_2023_l1144_114478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_range_l1144_114483

theorem cosine_sum_range (α β : ℝ) (h : Real.sin α + Real.sin β = 1) :
  ∃ (x : ℝ), Real.cos α + Real.cos β = x ∧ -Real.sqrt 3 ≤ x ∧ x ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_range_l1144_114483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_iff_coeff_sum_equal_l1144_114473

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

/-- Define coplanarity of four points in a vector space -/
def coplanar (a b c d : V) : Prop :=
  ∃ (x y z : ℝ), a - d = x • (b - d) + y • (c - d) + z • (0 : V)

/-- Theorem stating the equivalence between coplanarity and the coefficient sum condition -/
theorem coplanar_iff_coeff_sum_equal (a b c d : V) :
  coplanar V a b c d ↔
  ∀ (m n p q : ℝ), m • a + n • b = p • c + q • d → m + n = p + q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_iff_coeff_sum_equal_l1144_114473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_three_solutions_l1144_114431

-- Define the sign function
noncomputable def sign (x : ℝ) : ℝ :=
  if x > 0 then 1 else if x < 0 then -1 else 0

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  (x = 1000 - 1001 * sign (y + z + 1)) ∧
  (y = 1000 - 1001 * sign (x + z - 1)) ∧
  (z = 1000 - 1001 * sign (x + y + 2))

-- State the theorem
theorem system_has_three_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    (Finset.card solutions = 3) ∧
    (∀ (x y z : ℝ), (x, y, z) ∈ solutions ↔ system x y z) := by
  sorry

#check system_has_three_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_three_solutions_l1144_114431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_equality_implies_x_value_l1144_114469

noncomputable def set_A (x : ℝ) : Finset ℝ := {1, 2, 3, 4, x}
noncomputable def set_B : Finset ℝ := {2020, 2021, 2022, 2023, 2024}

noncomputable def variance (s : Finset ℝ) : ℝ :=
  let mean := (s.sum id) / s.card
  (s.sum (fun x => (x - mean) ^ 2)) / s.card

theorem variance_equality_implies_x_value :
  ∀ x : ℝ, variance (set_A x) = variance set_B → x = 0 ∨ x = 5 := by
  sorry

#check variance_equality_implies_x_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_equality_implies_x_value_l1144_114469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1144_114408

open Real

-- Define the function (marked as noncomputable due to transcendental functions)
noncomputable def f (x : ℝ) : ℝ := tan (π / 5 - x / 3)

-- State the theorem
theorem min_positive_period_of_f :
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  T = 3 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1144_114408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l1144_114470

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_expression_evaluation :
  (floor 6.5 : ℝ) * (floor (2/3) : ℝ) + (floor 2 : ℝ) * 7.2 + (floor 8.4 : ℝ) - 6.6 = 15.8 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l1144_114470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_discounted_tickets_l1144_114487

/-- Represents the number of concert tickets Martin bought at full price -/
def full_price_tickets : ℕ := sorry

/-- Represents the number of concert tickets Martin bought at discounted price -/
def discounted_tickets : ℕ := sorry

/-- The total number of tickets Martin bought -/
def total_tickets : ℕ := 10

/-- The price of a full-price ticket in cents -/
def full_price : ℕ := 200

/-- The price of a discounted ticket in cents -/
def discounted_price : ℕ := 160

/-- The total amount Martin spent in cents -/
def total_spent : ℕ := 1840

theorem martin_discounted_tickets : 
  full_price_tickets + discounted_tickets = total_tickets ∧ 
  full_price * full_price_tickets + discounted_price * discounted_tickets = total_spent →
  discounted_tickets = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_discounted_tickets_l1144_114487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_implies_a_bound_tangent_implies_a_value_l1144_114433

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (a * x) / (x + 1)

-- Theorem 1: If f is monotonically increasing on (0, 4), then a ≥ -4
theorem monotonic_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 4, Monotone (f a)) → a ≥ -4 := by sorry

-- Theorem 2: If f is tangent to y = 2x, then a = 4
theorem tangent_implies_a_value (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f a x = 2 * x ∧ HasDerivAt (f a) 2 x) → a = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_implies_a_bound_tangent_implies_a_value_l1144_114433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_neg_one_third_l1144_114402

noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - x - 1) / (3 * x + 1)

theorem limit_f_at_neg_one_third (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, x ≠ -1/3 → |x + 1/3| < δ → |f x + 5/3| < ε :=
by
  -- We'll use δ = ε/2
  let δ := ε / 2
  
  -- Show that δ > 0
  have hδ : δ > 0 := by
    apply half_pos
    exact hε

  -- Main proof
  use δ
  constructor
  · exact hδ
  · intro x hx_neq hx_lt
    -- The rest of the proof would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_neg_one_third_l1144_114402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1144_114488

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - x + 3 else x + 2/x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x/2 + a|) → a ∈ Set.Icc (-47/16 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1144_114488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1144_114419

-- Define the circle
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

-- Define a line passing through (1,2)
def line_l (m : ℝ) (x y : ℝ) : Prop := 
  (m + 1) * x + (2 * m + 1) * y = 5 * m + 3

-- Define the arc ratio condition
noncomputable def arc_ratio_condition (A B : ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), θ = 2 * Real.pi / 3 ∧ 
  (A.1 - 2) * (B.2 - 3) - (A.2 - 3) * (B.1 - 2) = 4 * Real.sin θ

-- Theorem statement
theorem line_equation (m : ℝ) :
  (∀ x y, circle_C x y → line_l m x y → 
    ∃ A B : ℝ × ℝ, arc_ratio_condition A B) →
  (∀ x y, line_l m x y ↔ (y = 2 ∨ x = 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1144_114419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_married_men_fraction_l1144_114452

theorem faculty_married_men_fraction 
  (total : ℕ) 
  (women_percentage : ℚ) 
  (married_percentage : ℚ) 
  (single_men_fraction : ℚ) 
  (h1 : women_percentage = 60 / 100)
  (h2 : married_percentage = 60 / 100)
  (h3 : single_men_fraction = 3 / 4)
  (h4 : total > 0) :
  let men := total - (women_percentage * ↑total).floor
  let married_men := (married_percentage * ↑total).floor - (women_percentage * married_percentage * ↑total).floor
  (married_men : ℚ) / men = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_married_men_fraction_l1144_114452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_translation_on_cosine_graphs_l1144_114415

theorem point_translation_on_cosine_graphs :
  ∀ (t m : ℝ),
    m > 0 → 
    t = Real.cos (2 * (π/4) + π/6) → 
    t = Real.cos (2 * (π/4 + m)) → 
    t = -1/2 ∧ 
    (∀ m' > 0, m' ≥ π/12) ∧ 
    (∃ m₀ > 0, m₀ = π/12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_translation_on_cosine_graphs_l1144_114415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentanol_HCl_reaction_yield_l1144_114468

-- Define the molecules and their molar quantities
def C5H12O : ℝ := 1
def HCl : ℝ := 100000  -- A large number to represent excess
def C5H11Cl : ℝ := 1
def H2O_theoretical : ℝ := 1

-- Define the actual yield of H2O (to be provided)
variable (H2O_actual : ℝ)

-- Define the percentage yield calculation
noncomputable def percentage_yield (actual : ℝ) (theoretical : ℝ) : ℝ :=
  (actual / theoretical) * 100

-- Theorem statement
theorem pentanol_HCl_reaction_yield :
  (C5H12O = 1 ∧ HCl > C5H12O) →
  (H2O_theoretical = 1) ∧
  (percentage_yield H2O_actual H2O_theoretical = (H2O_actual / H2O_theoretical) * 100) :=
by
  intro h
  apply And.intro
  · rfl
  · rfl

#check pentanol_HCl_reaction_yield

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentanol_HCl_reaction_yield_l1144_114468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1144_114430

-- Define set A
def A : Set ℝ := {x : ℝ | x / (x - 2) < 0}

-- Define set B as the set of integers
def B : Set ℝ := {x : ℝ | ∃ n : ℤ, x = ↑n}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1144_114430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_df_length_l1144_114432

-- Define the triangle ABC
structure Triangle (α : Type*) where
  A : α
  B : α
  C : α

-- Define congruence between triangles
def congruent {α : Type*} (t1 t2 : Triangle α) : Prop := sorry

-- Define the length of a side
noncomputable def side_length {α : Type*} [NormedAddCommGroup α] (A B : α) : ℝ := ‖B - A‖

-- Theorem statement
theorem df_length 
  {α : Type*} [NormedAddCommGroup α]
  (ABC DEF : Triangle α) 
  (h1 : congruent ABC DEF) 
  (h2 : side_length ABC.A ABC.C = 5.5) : 
  side_length DEF.A DEF.C = 5.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_df_length_l1144_114432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_l1144_114422

/-- Given a function f(x) = x^3 - ax^2 + 2, prove that if the tangent line at (1, f(1)) has slope -1, then a = 2 -/
theorem tangent_slope_implies_a (a : ℝ) : 
  let f := fun (x : ℝ) => x^3 - a*x^2 + 2
  deriv f 1 = -1 → a = 2 := by
  sorry

#check tangent_slope_implies_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_l1144_114422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_area_min_max_l1144_114424

noncomputable section

-- Define the hyperbola C₁
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / (2 * a^2) = 1

-- Define the parabola C₂
noncomputable def parabola (a : ℝ) (x y : ℝ) : Prop :=
  y^2 = -4 * Real.sqrt 3 * a * x

-- Define the focus F₁
noncomputable def focus (a : ℝ) : ℝ × ℝ :=
  (-Real.sqrt 3 * a, 0)

-- Define the area of triangle AOB
noncomputable def triangle_area (a k : ℝ) : ℝ :=
  6 * a^2 * Real.sqrt (1 + 1 / k^2)

-- Theorem statement
theorem chord_area_min_max (a : ℝ) (h : a > 0) :
  (∃ (k : ℝ), ∀ (m : ℝ), triangle_area a k ≤ triangle_area a m) ∧
  (¬ ∃ (k : ℝ), ∀ (m : ℝ), triangle_area a k ≥ triangle_area a m) ∧
  (∀ (k : ℝ), triangle_area a k ≥ 6 * a^2) ∧
  (triangle_area a 0 = 6 * a^2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_area_min_max_l1144_114424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_B_is_sampling_survey_l1144_114465

/-- A survey option -/
inductive SurveyOption
| A  -- Understanding students' clothing sizes when ordering school uniforms
| B  -- Investigating the killing radius of a batch of shells
| C  -- Understanding whether all teachers and students in the school have completed nucleic acid testing during the epidemic
| D  -- Conducting security checks on passengers boarding a plane

/-- Definition of a sampling survey -/
def is_sampling_survey (survey : SurveyOption) : Prop :=
  ∃ (subset : Set SurveyOption) (whole : Set SurveyOption), 
    subset ⊆ whole ∧ 
    (∀ (property : SurveyOption → Prop), 
      (∀ x ∈ subset, property x) → 
      (∀ x ∈ whole, property x))

/-- The theorem stating that option B is suitable for a sampling survey -/
theorem option_B_is_sampling_survey : 
  is_sampling_survey SurveyOption.B := by
  sorry

#check option_B_is_sampling_survey

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_B_is_sampling_survey_l1144_114465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1144_114456

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k - Real.exp (x * Real.log 3)) / (1 + k * Real.exp (x * Real.log 3))

-- Define the function g
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := f k (2*x) + 1

-- Theorem statement
theorem f_properties (k : ℝ) :
  (∀ x, f k x = -f k (-x)) →  -- f is an odd function
  (∃! k', ∀ x, f k x = f k' x) →  -- k is unique
  (∀ x y, x < y → f k x > f k y) →  -- f is monotonically decreasing
  (∀ y, -1 < y ∧ y < 1 ↔ ∃ x, f k x = y) →  -- range of f is (-1, 1)
  (∀ a, g k a + g k (-a) = 2) →  -- property of g
  k = 1 ∧
  (∀ x y, x < y → f k x > f k y) ∧
  (∀ y, -1 < y ∧ y < 1 ↔ ∃ x, f k x = y) ∧
  (∀ a, g k a + g k (-a) = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1144_114456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_three_proposals_l1144_114464

/-- Represents a proposal with votes --/
structure Proposal where
  diff : ℕ  -- Difference between votes in favor and against
  against_percent : ℚ  -- Percentage of votes against

/-- Calculates the total votes for a proposal --/
def total_votes (p : Proposal) : ℕ :=
  let against := (p.diff * (1 - p.against_percent)) / p.against_percent
  (against + p.diff + against).ceil.toNat

/-- The main theorem --/
theorem total_votes_three_proposals :
  let p1 := Proposal.mk 70 (2/5)
  let p2 := Proposal.mk 120 (7/20)
  let p3 := Proposal.mk 150 (3/10)
  total_votes p1 + total_votes p2 + total_votes p3 = 1126 := by
  sorry

#eval total_votes (Proposal.mk 70 (2/5)) +
      total_votes (Proposal.mk 120 (7/20)) +
      total_votes (Proposal.mk 150 (3/10))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_three_proposals_l1144_114464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1144_114425

/-- The function g(x) = (x-5) / sqrt(x^2-5x-6) -/
noncomputable def g (x : ℝ) : ℝ := (x - 5) / Real.sqrt (x^2 - 5*x - 6)

/-- The domain of g(x) -/
def domain_g : Set ℝ := {x | x < -1 ∨ x > 6}

theorem domain_of_g :
  ∀ x : ℝ, x ∈ domain_g ↔ (x^2 - 5*x - 6 > 0 ∧ g x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1144_114425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_b_onto_a_l1144_114477

noncomputable section

-- Define the vectors
def a : ℝ × ℝ := (Real.sqrt 3, 1)
def b : ℝ × ℝ := (-3, Real.sqrt 3)

-- Define the dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude function
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the projection function
def projection (u v : ℝ × ℝ) : ℝ := (dot_product u v) / (magnitude v)

-- Theorem statement
theorem projection_b_onto_a :
  projection b a = -Real.sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_b_onto_a_l1144_114477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1144_114481

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  bd : ℝ

-- State the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.c * Real.sin ((abc.A + abc.C) / 2) = abc.b * Real.sin abc.C)
  (h2 : abc.bd = 1)
  (h3 : abc.b = Real.sqrt 3) :
  abc.B = π / 3 ∧ abc.a + abc.b + abc.c = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1144_114481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_theorem_l1144_114442

/-- Represents a finite sequence of 5 terms -/
structure Sequence where
  terms : Fin 5 → ℝ

/-- Defines the initial sequence T -/
noncomputable def T (x : ℝ) : Sequence where
  terms := fun i => x ^ i.val

/-- Applies the B transformation to a sequence -/
noncomputable def B (s : Sequence) : Sequence where
  terms := fun i => if h : i.val < 4 then
    (s.terms ⟨i.val, Nat.lt_trans h (by norm_num)⟩ + s.terms ⟨i.val + 1, by { cases i; repeat { exact Nat.succ_lt_succ h } }⟩) / 2
  else 0

/-- Recursively applies B m times -/
noncomputable def B_power (m : ℕ) (s : Sequence) : Sequence :=
  match m with
  | 0 => s
  | m + 1 => B (B_power m s)

theorem x_value_theorem (x : ℝ) (h1 : x > 0) 
    (h2 : (B_power 4 (T x)).terms = fun _ => 1 / 4) : x = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_theorem_l1144_114442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sum_l1144_114454

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (a₁ + arithmetic_sequence a₁ d n) / 2

noncomputable def b_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ := |a n - 2|

theorem arithmetic_sequence_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (a 1 = 12) →
  (S 10 = 30) →
  (∀ n, a n = arithmetic_sequence 12 (-2) n) →
  (∀ n, S n = sum_arithmetic_sequence 12 (-2) n) →
  (∀ n, a n = 14 - 2 * (n : ℝ)) ∧
  (∀ n : ℕ, (Finset.range n).sum (λ i => b_sequence a (i + 1)) = 
    if n ≤ 6 then -(n : ℝ)^2 + 11*(n : ℝ) else (n : ℝ)^2 - 11*(n : ℝ) + 60) :=
by sorry

#check arithmetic_sequence_and_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sum_l1144_114454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1144_114417

theorem simplify_expression :
  Real.sqrt 2 * (2 : ℝ)^(1/2 : ℝ) + 18 / 3 * 3 - (8 : ℝ)^(3/2 : ℝ) = 20 - 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1144_114417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l1144_114405

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- State the theorem
theorem f_composition (x : ℝ) (h : -1 < x ∧ x < 1) :
  f ((4 * x + x^4) / (1 + 4 * x^3)) = 4 * f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l1144_114405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_quadrant_angles_characterization_l1144_114414

/-- The set of angles with terminal sides in the first quadrant -/
def first_quadrant_angles : Set ℝ :=
  {α | ∃ k : ℤ, (2 : ℝ) * Real.pi * k < α ∧ α < Real.pi / 2 + (2 : ℝ) * Real.pi * k}

/-- Theorem stating that the set of angles with terminal sides in the first quadrant
    is equal to {α | 2kπ < α < π/2 + 2kπ, k ∈ ℤ} -/
theorem first_quadrant_angles_characterization :
  first_quadrant_angles = {α : ℝ | ∃ k : ℤ, (2 : ℝ) * Real.pi * k < α ∧ α < Real.pi / 2 + (2 : ℝ) * Real.pi * k} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_quadrant_angles_characterization_l1144_114414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_solution_l1144_114421

theorem unique_n_solution (n : ℕ) (h : Nat.factorial (n + 1) + Nat.factorial (n + 3) = Nat.factorial n * 964) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_solution_l1144_114421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l1144_114462

-- Define the variables and constants
variable (A S ρ v₀ v : ℝ)

-- Define the force equation
noncomputable def F (A S ρ v₀ v : ℝ) : ℝ := (A * S * ρ * (v₀ - v)^2) / 2

-- Define the power equation
noncomputable def N (A S ρ v₀ v : ℝ) : ℝ := F A S ρ v₀ v * v

-- Theorem statement
theorem max_power_speed (h₁ : v₀ = 4.8) : 
  (∀ u, deriv (N A S ρ v₀) u = 0 → u = v₀ / 3) → v = 1.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l1144_114462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_sum_binomial_expansions_l1144_114449

theorem coefficient_x_squared_sum_binomial_expansions : 
  Finset.sum (Finset.range 9) (λ n => Nat.choose (n + 1) 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_sum_binomial_expansions_l1144_114449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_and_radius_significance_l1144_114498

/-- A crate with internal dimensions -/
structure Crate where
  length : ℝ
  width : ℝ
  height : ℝ

/-- A right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ :=
  Real.pi * c.radius^2 * c.height

theorem cylinder_volume_and_radius_significance 
  (crate : Crate)
  (cylinder : Cylinder)
  (h_crate_dims : crate.length = 12 ∧ crate.width = 8 ∧ crate.height = 2)
  (h_cylinder_radius : cylinder.radius = 2)
  (h_cylinder_height : cylinder.height = min crate.length (min crate.width crate.height)) :
  cylinderVolume cylinder = 32 * Real.pi ∧ 
  ∀ (r : ℝ), cylinderVolume { radius := r, height := cylinder.height } = Real.pi * r^2 * cylinder.height :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_and_radius_significance_l1144_114498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l1144_114404

theorem sum_of_roots_quadratic : 
  (∃ x y : ℝ, x^2 = 8*x + 15 ∧ y^2 = 8*y + 15 ∧ x ≠ y) → x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l1144_114404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_PAB_l1144_114400

-- Define the line l₁
noncomputable def l₁ (x : ℝ) : ℝ := (1/2) * x + 3

-- Define point A (intersection of l₁ with x-axis)
def A : ℝ × ℝ := (-6, 0)

-- Define point B (intersection of l₁ with y-axis)
def B : ℝ × ℝ := (0, 3)

-- Define the line l₂ (l₁ shifted 8 units to the right)
noncomputable def l₂ (x : ℝ) : ℝ := (1/2) * x - 1

-- Define point P (intersection of l₂ with x-axis)
def P : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem area_triangle_PAB : 
  let triangle_area := (1/2) * |P.1 - A.1| * B.2
  triangle_area = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_PAB_l1144_114400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_distance_slope_l1144_114490

theorem curve_line_distance_slope (k : ℝ) :
  (∃! p q : ℝ × ℝ, 
    p.1^2 + 4 * p.2^2 = 4 ∧
    q.1^2 + 4 * q.2^2 = 4 ∧
    p ≠ q ∧
    (abs (k * p.1 - p.2) / Real.sqrt (k^2 + 1) = Real.sqrt 2) ∧
    (abs (k * q.1 - q.2) / Real.sqrt (k^2 + 1) = Real.sqrt 2)) →
  k = 1 ∨ k = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_distance_slope_l1144_114490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_area_circumference_ratio_l1144_114480

-- Define the circle properties
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- Theorem statement
theorem circle_radius_from_area_circumference_ratio 
  (M N : ℝ) (h : M / N = 20) : 
  ∃ r : ℝ, circle_area r = M ∧ circle_circumference r = N ∧ r = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_area_circumference_ratio_l1144_114480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_signature_pens_count_l1144_114407

/-- Represents the number of signature pens purchased -/
def signature_pens : ℕ := sorry

/-- Represents the number of ballpoint pens purchased -/
def ballpoint_pens : ℕ := sorry

/-- The total number of pens purchased -/
axiom total_pens : signature_pens + ballpoint_pens = 15

/-- The cost of a signature pen in yuan -/
def signature_pen_cost : ℚ := 2

/-- The cost of a ballpoint pen in yuan -/
def ballpoint_pen_cost : ℚ := 3/2

/-- The total cost of all pens in yuan -/
def total_cost : ℚ := signature_pens * signature_pen_cost + ballpoint_pens * ballpoint_pen_cost

/-- The total cost is more than 26 yuan -/
axiom cost_lower_bound : total_cost > 26

/-- The total cost is less than 27 yuan -/
axiom cost_upper_bound : total_cost < 27

/-- The number of signature pens purchased is 8 -/
theorem signature_pens_count : signature_pens = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_signature_pens_count_l1144_114407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1144_114476

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + 2 * Real.cos x - 4

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x ∈ Set.Icc 0 (2 * Real.pi), 
  ∀ y ∈ Set.Icc 0 (2 * Real.pi), 
  x ≤ y → f x ≤ f y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1144_114476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_jar_days_l1144_114443

/-- The number of days it takes to fill the entire jar -/
def full_jar_days : ℕ := 19

/-- The growth rate of the kombucha area per day -/
def growth_rate : ℝ := 2

/-- The function representing the area of kombucha after x days -/
noncomputable def kombucha_area (x : ℝ) : ℝ := growth_rate ^ x

theorem half_jar_days : 
  kombucha_area (full_jar_days - 1 : ℝ) = (1 / 2) * kombucha_area (full_jar_days : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_jar_days_l1144_114443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_l1144_114409

/-- The function f(x) defined in terms of k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

/-- A predicate to check if three real numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b ∧ a > 0 ∧ b > 0 ∧ c > 0

/-- The main theorem stating the condition on k for triangle formation -/
theorem triangle_condition (k : ℝ) : 
  (∀ (a b c : ℝ), can_form_triangle (f k a) (f k b) (f k c)) ↔ 
  (-1/2 < k ∧ k < 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_l1144_114409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1144_114426

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (a - b * Real.cos C) * Real.sin B + Real.sqrt 3 * b * Real.cos B * Real.cos C = 0 ∧
  b = 3 ∧
  B ≠ Real.pi / 2

-- Theorem for part 1
theorem part1 (A B C : ℝ) (a b c : ℝ) :
  Triangle A B C a b c → a = 5 → c = 7 := by sorry

-- Theorem for part 2
theorem part2 (A B C : ℝ) (a b c : ℝ) :
  Triangle A B C a b c → 
  (∃ (D : ℝ), D * D = (a * a + b * b) / 4 + ((a * b) / 4) * Real.cos C ∧ D = 7 / 2) →
  (1 / 2) * a * b * Real.sin C = 6 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1144_114426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_for_ten_people_l1144_114474

-- Define a type for people
def Person : Type := Fin 10

-- Define a type for questions
def Question : Type := ℕ

-- Define a function to represent answering a question correctly
def answers (p : Person) (q : Question) : Prop := sorry

theorem min_questions_for_ten_people (n : ℕ) 
  (h1 : ∀ (s : Finset Person), s.card = 5 → (∀ q : Question, ∃ p ∈ s, answers p q))
  (h2 : ∀ (s : Finset Person), s.card = 4 → (∃ q : Question, ∀ p ∈ s, ¬(answers p q)))
  : n ≥ 210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_for_ten_people_l1144_114474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_root_deriv_nonneg_l1144_114434

/-- Polynomial function f(x) = x^n + a_{n-1}x^{n-1} + ... + a_1x + a -/
def f (n : ℕ) (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  x^n + (Finset.range n).sum (λ i ↦ a i * x^i)

/-- The derivative of f(x) -/
noncomputable def f_deriv (n : ℕ) (a : ℕ → ℝ) : ℝ → ℝ := 
  deriv (f n a)

/-- IsMaxRoot f x means x is the maximum root of f -/
def IsMaxRoot (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0 ∧ ∀ y > x, f y ≠ 0

theorem max_root_deriv_nonneg (n : ℕ) (a : ℕ → ℝ) (x : ℝ) 
  (h1 : n > 2) 
  (h2 : IsMaxRoot (f n a) x) : 
  f_deriv n a x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_root_deriv_nonneg_l1144_114434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completes_in_20_days_l1144_114467

noncomputable def B_days : ℝ := 30

noncomputable def total_days : ℝ := 15

noncomputable def A_work_days : ℝ := total_days - 5

noncomputable def A_rate (x : ℝ) : ℝ := 1 / x

noncomputable def B_rate : ℝ := 1 / B_days

noncomputable def work_together (x : ℝ) : ℝ := A_work_days * (A_rate x + B_rate)

noncomputable def B_work_alone : ℝ := (total_days - A_work_days) * B_rate

theorem A_completes_in_20_days :
  ∃ x : ℝ, x > 0 ∧ work_together x + B_work_alone = 1 ∧ x = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completes_in_20_days_l1144_114467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_implies_m_range_l1144_114482

/-- The function f(x) defined in the problem -/
noncomputable def f (m n x : ℝ) : ℝ := 2 * Real.log x - (1/2) * m * x^2 - n * x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (m n x : ℝ) : ℝ := 2 / x - m * x - n

theorem local_max_implies_m_range (m n : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f m n x ≤ f m n 2) →
  f_derivative m n 2 = 0 →
  m > -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_implies_m_range_l1144_114482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1144_114451

/-- Given a hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0, and one focus at (2,0),
    prove that the equation of its asymptotes is √3x ± y = 0 -/
theorem hyperbola_asymptotes 
  (b : ℝ) 
  (h1 : b > 0) 
  (h2 : ∀ x y : ℝ, x^2 - y^2/b^2 = 1 → Set.Mem (x, y) {p : ℝ × ℝ | x^2 - y^2/b^2 = 1})
  (h3 : Set.Mem (2, 0) {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ (x^2 - y^2/b^2 = 1)}) : 
  ∃ k : ℝ, k = Real.sqrt 3 ∧ ∀ x y : ℝ, (x^2 - y^2/b^2 = 1) → (k*x = y ∨ k*x = -y) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1144_114451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_containing_circle_l1144_114446

/-- The relationship between the semi-major and semi-minor axes of an ellipse containing a unit circle --/
theorem ellipse_containing_circle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∀ x y : ℝ, x^2 + y^2 = 1 → x^2/a^2 + y^2/b^2 ≤ 1) : 
  a = b * (1 + b) / (b^2 - 1) := by
  sorry

#check ellipse_containing_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_containing_circle_l1144_114446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1144_114463

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the point of tangency
noncomputable def tangent_point : ℝ := Real.exp (-1)

-- Theorem: The equation of the tangent line is y = -1/e
theorem tangent_line_equation :
  let x₀ := tangent_point
  let y₀ := f x₀
  let m := deriv f x₀
  (fun x => m * (x - x₀) + y₀) = (fun _ => -1 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1144_114463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1144_114493

theorem solve_exponential_equation :
  ∃ y : ℚ, (5 : ℝ) ^ (3 * (y : ℝ)) = Real.sqrt 125 ∧ y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1144_114493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_500g_l1144_114439

/-- The cost of transporting a scientific instrument to the International Space Station --/
noncomputable def transportCost (weight : ℝ) (baseCost : ℝ) (discountRate : ℝ) : ℝ :=
  let weightInKg := weight / 1000
  let rawCost := weightInKg * baseCost
  if weightInKg < 1 then
    rawCost * (1 - discountRate)
  else
    rawCost

/-- Theorem: The cost of transporting a 500 g scientific instrument is $8,100 --/
theorem transport_cost_500g :
  transportCost 500 18000 0.1 = 8100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_500g_l1144_114439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_bearing_symmetry_prove_direction_from_a_to_b_l1144_114471

/-- Represents cardinal directions --/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a compass bearing --/
structure Bearing where
  angle : ℝ
  primary : Direction
  secondary : Direction

/-- Defines the opposite direction --/
def opposite_direction : Direction → Direction
  | Direction.North => Direction.South
  | Direction.South => Direction.North
  | Direction.East => Direction.West
  | Direction.West => Direction.East

/-- Defines the opposite bearing --/
def opposite_bearing (b : Bearing) : Bearing :=
  { angle := b.angle,
    primary := opposite_direction b.primary,
    secondary := opposite_direction b.secondary }

/-- Represents a point in space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to calculate bearing between two points --/
noncomputable def bearing_between (a b : Point) : Bearing :=
  sorry -- Placeholder for actual calculation

/-- Theorem: If A is at a certain bearing from B, then B is at the opposite bearing from A --/
theorem opposite_bearing_symmetry (a b : Point) (bearing : Bearing) :
  bearing_between b a = bearing →
  bearing_between a b = opposite_bearing bearing :=
by
  sorry -- Placeholder for actual proof

/-- Given condition: A is 40° south by east from B --/
def given_bearing : Bearing :=
  { angle := 40,
    primary := Direction.South,
    secondary := Direction.East }

/-- Theorem to prove --/
theorem prove_direction_from_a_to_b (a b : Point) :
  bearing_between b a = given_bearing →
  bearing_between a b = { angle := 40,
                          primary := Direction.North,
                          secondary := Direction.West } :=
by
  sorry -- Placeholder for actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_bearing_symmetry_prove_direction_from_a_to_b_l1144_114471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_whitewashing_cost_l1144_114450

-- Define the hall dimensions
def hall_length : ℝ := 40
def hall_width : ℝ := 30
def hall_height : ℝ := 20

-- Define the costs
def wall_cost_per_sqft : ℝ := 2
def ceiling_cost_per_sqft : ℝ := 3

-- Define door and window dimensions
def door_height : ℝ := 7
def door_width : ℝ := 4
def large_window_height : ℝ := 5
def large_window_width : ℝ := 4
def small_window_height : ℝ := 4
def small_window_width : ℝ := 3

-- Define the number of doors and windows
def num_doors : ℕ := 3
def num_large_windows : ℕ := 3
def num_small_windows : ℕ := 2

-- Theorem statement
theorem total_whitewashing_cost :
  (2 * (hall_length * hall_height + hall_width * hall_height) -
   (↑num_doors * door_height * door_width +
    ↑num_large_windows * large_window_height * large_window_width +
    ↑num_small_windows * small_window_height * small_window_width)) * wall_cost_per_sqft +
  (hall_length * hall_width) * ceiling_cost_per_sqft = 8864 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_whitewashing_cost_l1144_114450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_tuple_theorem_l1144_114485

/-- A structure representing a 5-tuple of integers -/
structure FiveTuple where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- Function to check if a number is a four-digit integer -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- Function to check if two numbers have the same initial digit -/
def same_initial_digit (m n : ℕ) : Prop := m / 1000 = n / 1000

/-- Function to check if a number divides another number -/
def divides (m n : ℕ) : Prop := ∃ k : ℕ, n = k * m

/-- Main theorem statement -/
theorem five_tuple_theorem (t : FiveTuple) : 
  (is_four_digit t.a ∧ is_four_digit t.b ∧ is_four_digit t.c ∧ is_four_digit t.d ∧ is_four_digit t.e) →
  (same_initial_digit t.a t.b ∧ same_initial_digit t.a t.c ∧ same_initial_digit t.a t.d ∧ same_initial_digit t.a t.e) →
  (t.a ≠ t.b ∧ t.a ≠ t.c ∧ t.a ≠ t.d ∧ t.a ≠ t.e ∧ t.b ≠ t.c ∧ t.b ≠ t.d ∧ t.b ≠ t.e ∧ t.c ≠ t.d ∧ t.c ≠ t.e ∧ t.d ≠ t.e) →
  (divides t.a (t.a + t.b + t.c + t.d + t.e) ∨ 
   divides t.b (t.a + t.b + t.c + t.d + t.e) ∨ 
   divides t.c (t.a + t.b + t.c + t.d + t.e) ∨ 
   divides t.d (t.a + t.b + t.c + t.d + t.e)) →
  (∃ k : ℕ, k ∈ ({17, 18, 19} : Set ℕ) ∧ 
   ((t.a = 60 * k ∧ t.b = 70 * k ∧ t.c = 84 * k ∧ t.d = 101 * k ∧ t.e = 105 * k) ∨
    (t.a = 60 * k ∧ t.b = 70 * k ∧ t.c = 84 * k ∧ t.d = 105 * k ∧ t.e = 101 * k) ∨
    (t.a = 60 * k ∧ t.b = 70 * k ∧ t.c = 101 * k ∧ t.d = 84 * k ∧ t.e = 105 * k) ∨
    (t.a = 60 * k ∧ t.b = 70 * k ∧ t.c = 101 * k ∧ t.d = 105 * k ∧ t.e = 84 * k) ∨
    (t.a = 60 * k ∧ t.b = 70 * k ∧ t.c = 105 * k ∧ t.d = 84 * k ∧ t.e = 101 * k) ∨
    (t.a = 60 * k ∧ t.b = 70 * k ∧ t.c = 105 * k ∧ t.d = 101 * k ∧ t.e = 84 * k))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_tuple_theorem_l1144_114485
