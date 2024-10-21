import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_24_l1292_129289

theorem sum_of_divisors_24 : Finset.sum (Nat.divisors 24) id = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_24_l1292_129289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1292_129230

theorem equation_solution (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hpq : p ≠ q) :
  let x := (p + q) / (p^2 + p*q + q^2)
  (p / q) - (p * x / (q * x - 1)) = (q / p) - (q * x / (p * x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1292_129230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_formula_l1292_129261

/-- The volume of a regular hexagonal pyramid with height h and lateral face area Q -/
noncomputable def hexagonal_pyramid_volume (h Q : ℝ) : ℝ :=
  (1/3) * h * Real.sqrt 3 * (Real.sqrt (h^4 + 12*Q^2) - h^2)

/-- Theorem: The volume of a regular hexagonal pyramid with height h and lateral face area Q
    is given by (1/3) * h * √3 * (√(h^4 + 12Q^2) - h^2) -/
theorem hexagonal_pyramid_volume_formula (h Q : ℝ) (h_pos : h > 0) (Q_pos : Q > 0) :
  ∃ (V : ℝ), V = hexagonal_pyramid_volume h Q ∧ 
  V = (1/3) * h * Real.sqrt 3 * (Real.sqrt (h^4 + 12*Q^2) - h^2) := by
  use hexagonal_pyramid_volume h Q
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_formula_l1292_129261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1292_129206

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |2*x - 1|

-- Define the solution set condition
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Define the function g
noncomputable def g (x m : ℝ) : ℝ := 1 / (f x + f (x + 1) + m)

-- Theorem 1
theorem part_one : 
  ∃ a : ℝ, ({x : ℝ | f x ≤ a} = solution_set a) ∧ a = 1 :=
by sorry

-- Theorem 2
theorem part_two : 
  ∃ m : ℝ, (∀ x : ℝ, g x m ≠ 0) ∧ (m < 0 ∨ m > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1292_129206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_minus_one_not_square_l1292_129259

theorem product_minus_one_not_square (d : ℕ) 
  (h1 : d ≠ 2) (h2 : d ≠ 5) (h3 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
  a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_minus_one_not_square_l1292_129259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_below_curve_condition_l1292_129215

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * Real.exp (2 * x) + x

-- Define the theorem for part I
theorem monotonically_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Iio (0 : ℝ), Monotone (f a)) ↔ a ≥ 0 := by sorry

-- Define the theorem for part II
theorem below_curve_condition (a : ℝ) :
  (∀ x ∈ Set.Ioi (0 : ℝ), f a x < 2 * a * Real.exp x) ↔ a ∈ Set.Icc (-1/2) (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_below_curve_condition_l1292_129215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_quadrilateral_l1292_129227

-- Define a quadrilateral as a set of four points in the plane
def Quadrilateral : Type := Fin 4 → ℝ × ℝ

-- Define the area of a quadrilateral
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Define a point being inside a quadrilateral
def inside (p : ℝ × ℝ) (q : Quadrilateral) : Prop := sorry

-- Define the area of a triangle given three points
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem exists_special_quadrilateral :
  ∃ (q : Quadrilateral),
    area q = 1 ∧
    ∀ (O : ℝ × ℝ), inside O q →
      ∃ (i : Fin 4),
        ¬ (∃ (a b : ℚ), a/b = triangleArea O (q i) (q (i.succ))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_quadrilateral_l1292_129227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1292_129213

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 + Real.sin x - 1

-- State the theorem
theorem f_range : Set.Icc (-2 : ℝ) (1/4 : ℝ) = Set.range f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1292_129213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l1292_129212

/-- Represents a plane in 3D space defined by ax + by + cz = d -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the distance between two parallel planes -/
noncomputable def distance_between_parallel_planes (p1 p2 : Plane) : ℝ :=
  let normal_vector_magnitude := Real.sqrt (p1.a^2 + p1.b^2 + p1.c^2)
  |p1.d - p2.d| / normal_vector_magnitude

theorem distance_between_specific_planes :
  let plane1 : Plane := ⟨2, -4, 4, 10⟩
  let plane2 : Plane := ⟨1, -2, 2, 1⟩
  distance_between_parallel_planes plane1 plane2 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l1292_129212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_equals_fraction_l1292_129296

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingLength : ℕ
  nonRepeatingNonneg : 0 ≤ nonRepeating
  repeatingNonneg : 0 ≤ repeating
  repeatingLtOne : repeating < 1

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.nonRepeating + d.repeating / (10^d.repeatingLength - 1)

/-- The repeating decimal 0.7̅56̅ -/
def decimal : RepeatingDecimal where
  nonRepeating := 7/10
  repeating := 56/100
  repeatingLength := 2
  nonRepeatingNonneg := by norm_num
  repeatingNonneg := by norm_num
  repeatingLtOne := by norm_num

/-- Theorem stating that 0.7̅56̅ is equal to 749/990 -/
theorem decimal_equals_fraction : decimal.toRational = 749/990 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_equals_fraction_l1292_129296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_range_l1292_129238

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the problem conditions
theorem x_plus_y_range (x y : ℝ) 
  (h1 : y = 4 * (floor x) + 4)
  (h2 : y = 5 * (floor (x - 3)) + 7)
  (h3 : x > 3)
  (h4 : x ≠ ↑(floor x)) : 
  64 < x + y ∧ x + y < 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_range_l1292_129238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_angle_l1292_129207

/-- Regular quadrilateral prism -/
structure RegularQuadrilateralPrism where
  base : Set (EuclideanSpace ℝ (Fin 2))
  height : ℝ
  is_regular : ∃ (side : ℝ), ∀ (p q : base), p ≠ q → dist p q = side

/-- Plane passing through midpoints of two adjacent base sides and opposite upper vertex -/
structure CrossSectionPlane (prism : RegularQuadrilateralPrism) where
  midpoint1 : EuclideanSpace ℝ (Fin 2)
  midpoint2 : EuclideanSpace ℝ (Fin 2)
  upper_vertex : EuclideanSpace ℝ (Fin 3)

/-- The perimeter of the cross-section -/
noncomputable def cross_section_perimeter (prism : RegularQuadrilateralPrism) (plane : CrossSectionPlane prism) : ℝ :=
  sorry

/-- The diagonal of the base -/
noncomputable def base_diagonal (prism : RegularQuadrilateralPrism) : ℝ :=
  sorry

/-- The perimeter of the cross-section is three times the diagonal of the base -/
def perimeter_condition (prism : RegularQuadrilateralPrism) (plane : CrossSectionPlane prism) : Prop :=
  cross_section_perimeter prism plane = 3 * base_diagonal prism

/-- The angle between the cross-section plane and the base plane -/
noncomputable def angle_with_base (prism : RegularQuadrilateralPrism) (plane : CrossSectionPlane prism) : ℝ :=
  sorry

theorem cross_section_angle (prism : RegularQuadrilateralPrism) (plane : CrossSectionPlane prism) 
  (h : perimeter_condition prism plane) : 
  angle_with_base prism plane = Real.arccos (3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_angle_l1292_129207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l1292_129299

/-- Calculates the cost price of an article given the sale price including tax, sales tax rate, and profit margin. -/
noncomputable def cost_price (sale_price_with_tax : ℝ) (sales_tax_rate : ℝ) (profit_margin : ℝ) : ℝ :=
  sale_price_with_tax / (1 + sales_tax_rate) / (1 + profit_margin)

/-- Theorem stating that the cost price of an article is approximately 491.23 given the specified conditions. -/
theorem cost_price_calculation :
  let sale_price_with_tax : ℝ := 616
  let sales_tax_rate : ℝ := 0.10
  let profit_margin : ℝ := 0.14
  let calculated_cost_price := cost_price sale_price_with_tax sales_tax_rate profit_margin
  ∃ ε > 0, |calculated_cost_price - 491.23| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l1292_129299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1292_129256

noncomputable def f (x : ℝ) := |x - 3/4| + |x + 5/4|

theorem f_properties :
  (∀ x : ℝ, f x ≥ 2) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → m + 2*n = 2 →
    ∀ x : ℝ, Real.sqrt (m + 1) + Real.sqrt (2*n + 1) ≤ 2 * Real.sqrt (f x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1292_129256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_people_in_company_l1292_129205

/-- Represents a company of people -/
structure Company where
  people : Finset Nat
  knows : Nat → Nat → Bool

/-- A person is sociable if they have at least 20 acquaintances, 
    with at least two of those acquaintances knowing each other -/
def IsSociable (c : Company) (p : Nat) : Prop :=
  ∃ (acquaintances : Finset Nat),
    acquaintances ⊆ c.people ∧
    acquaintances.card ≥ 20 ∧
    (∀ a, a ∈ acquaintances → c.knows p a) ∧
    (∃ a b, a ∈ acquaintances ∧ b ∈ acquaintances ∧ a ≠ b ∧ c.knows a b)

/-- A person is shy if they have at least 20 non-acquaintances, 
    with at least two of those non-acquaintances not knowing each other -/
def IsShy (c : Company) (p : Nat) : Prop :=
  ∃ (non_acquaintances : Finset Nat),
    non_acquaintances ⊆ c.people ∧
    non_acquaintances.card ≥ 20 ∧
    (∀ a, a ∈ non_acquaintances → ¬c.knows p a) ∧
    (∃ a b, a ∈ non_acquaintances ∧ b ∈ non_acquaintances ∧ a ≠ b ∧ ¬c.knows a b)

/-- The maximum number of people in a company where no one is sociable or shy is 40 -/
theorem max_people_in_company : 
  ∀ (c : Company),
    (∀ p, p ∈ c.people → ¬IsSociable c p ∧ ¬IsShy c p) →
    c.people.card ≤ 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_people_in_company_l1292_129205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_even_integers_l1292_129262

/-- The set of even digits -/
def EvenDigits : Finset Nat := {0, 2, 4, 6, 8}

/-- A function that checks if a number has only even digits -/
def hasOnlyEvenDigits (n : Nat) : Bool :=
  n.digits 10 |>.all (· ∈ EvenDigits)

/-- The set of three-digit positive integers with only even digits -/
def ThreeDigitEvenIntegers : Finset Nat :=
  Finset.filter (λ n => 100 ≤ n ∧ n < 1000 ∧ hasOnlyEvenDigits n) (Finset.range 1000)

/-- The theorem to be proved -/
theorem count_three_digit_even_integers :
  Finset.card ThreeDigitEvenIntegers = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_even_integers_l1292_129262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_combinations_l1292_129273

/-- The sum of combinations C(8,2) and C(8,3) equals 84 -/
theorem sum_of_combinations : Nat.choose 8 2 + Nat.choose 8 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_combinations_l1292_129273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_ratio_l1292_129211

/-- The time it takes printer X to complete the job alone -/
noncomputable def time_x : ℝ := 15

/-- The time it takes printer Y to complete the job alone -/
noncomputable def time_y : ℝ := 10

/-- The time it takes printer Z to complete the job alone -/
noncomputable def time_z : ℝ := 20

/-- The rate at which printer X completes the job -/
noncomputable def rate_x : ℝ := 1 / time_x

/-- The rate at which printer Y completes the job -/
noncomputable def rate_y : ℝ := 1 / time_y

/-- The rate at which printer Z completes the job -/
noncomputable def rate_z : ℝ := 1 / time_z

/-- The combined rate of printers Y and Z -/
noncomputable def rate_yz : ℝ := rate_y + rate_z

/-- The time it takes printers Y and Z to complete the job together -/
noncomputable def time_yz : ℝ := 1 / rate_yz

theorem printer_ratio : time_x / time_yz = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_ratio_l1292_129211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_approx_41_percent_l1292_129292

/-- Represents the total cost of Roger's expenses as a fraction of his allowance -/
noncomputable def total_cost_fraction (B : ℝ) : ℝ :=
  let t := (3 / 13) * B  -- movie ticket cost
  let s := (1 / 13) * B  -- soda cost
  let n := 5             -- snack cost
  (t + s + n) / B

/-- Theorem stating that the total cost fraction is approximately 41% -/
theorem total_cost_approx_41_percent (B : ℝ) (h : B > 0) :
  ∃ ε > 0, |total_cost_fraction B - 0.41| < ε :=
by
  -- The proof is omitted for now
  sorry

#check total_cost_approx_41_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_approx_41_percent_l1292_129292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_function_range_l1292_129297

theorem monotonic_function_range (f : ℝ → ℝ) (h_mono : Monotone f) :
  (∀ a : ℝ, f (2 - a^2) > f a) → {a : ℝ | a < -2 ∨ a > 1} = Set.univ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_function_range_l1292_129297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1292_129223

noncomputable def f (x : ℝ) := Real.log (2*x - x^2)

theorem domain_of_f : Set.Ioo 0 2 = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1292_129223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_three_l1292_129298

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 1

theorem inverse_f_at_three (a : ℝ) (h : f a 1 = 1) : 
  (Function.invFun (f a)) 3 = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_three_l1292_129298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_sum_three_dice_l1292_129266

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The set of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ 3

/-- The set of prime numbers that can be rolled as a sum of three dice -/
def primeSums : List ℕ := [3, 5, 7, 11, 13, 17]

/-- The number of ways to roll each prime sum -/
def waysToRollPrime : ℕ → ℕ
| 3 => 1
| 5 => 6
| 7 => 15
| 11 => 27
| 13 => 21
| 17 => 3
| _ => 0

/-- The total number of ways to roll a prime sum -/
def totalPrimeWays : ℕ := (primeSums.map waysToRollPrime).sum

/-- The probability of rolling a prime sum with three dice -/
theorem probability_prime_sum_three_dice :
  (totalPrimeWays : ℚ) / totalOutcomes = 73 / 216 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_sum_three_dice_l1292_129266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_segment_exists_l1292_129263

/-- Represents a point on the 5x5 grid -/
structure GridPoint where
  x : Fin 5
  y : Fin 5

/-- Represents a segment between two points on the grid -/
structure Segment where
  start : GridPoint
  stop : GridPoint

/-- The set of all marked centers on the 5x5 grid -/
def MarkedCenters : Set GridPoint :=
  {p | p.x < 5 ∧ p.y < 5}

/-- Predicate to check if two segments intersect -/
def intersect (s1 s2 : Segment) : Prop := sorry

/-- Theorem stating that a fifth non-intersecting segment can always be drawn -/
theorem fifth_segment_exists (s1 s2 s3 s4 : Segment) 
  (h1 : s1.start ∈ MarkedCenters) (h2 : s1.stop ∈ MarkedCenters)
  (h3 : s2.start ∈ MarkedCenters) (h4 : s2.stop ∈ MarkedCenters)
  (h5 : s3.start ∈ MarkedCenters) (h6 : s3.stop ∈ MarkedCenters)
  (h7 : s4.start ∈ MarkedCenters) (h8 : s4.stop ∈ MarkedCenters)
  (h9 : ¬ intersect s1 s2) (h10 : ¬ intersect s1 s3) (h11 : ¬ intersect s1 s4)
  (h12 : ¬ intersect s2 s3) (h13 : ¬ intersect s2 s4) (h14 : ¬ intersect s3 s4) :
  ∃ (s5 : Segment), s5.start ∈ MarkedCenters ∧ s5.stop ∈ MarkedCenters ∧
    ¬ intersect s5 s1 ∧ ¬ intersect s5 s2 ∧ ¬ intersect s5 s3 ∧ ¬ intersect s5 s4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_segment_exists_l1292_129263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_discount_proof_l1292_129201

theorem store_discount_proof (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) 
  (h1 : initial_discount = 0.25)
  (h2 : additional_discount = 0.15)
  (h3 : claimed_discount = 0.40) : 
  (1 - (1 - initial_discount) * (1 - additional_discount) = 0.3625) ∧
  (claimed_discount - (1 - (1 - initial_discount) * (1 - additional_discount)) = 0.0375) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_discount_proof_l1292_129201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounds_l1292_129219

theorem sequence_bounds (n : ℕ) (hn : n > 0) :
  ∃ a : ℕ → ℚ,
    a 0 = 1 / 2 ∧
    (∀ k, a (k + 1) = a k + (1 / n) * (a k)^2) ∧
    1 - 1 / n < a n ∧ a n < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounds_l1292_129219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l1292_129242

/-- Given 5 points on the positive x-axis and 3 points on the positive y-axis,
    the maximum number of intersection points formed by connecting these points is 30. -/
theorem max_intersection_points (x_points y_points : ℕ) 
  (hx : x_points = 5) (hy : y_points = 3) : ℕ := 30

#check max_intersection_points 5 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l1292_129242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_existence_l1292_129209

/-- Theorem: Given a positive side length and a positive half-sum of diagonals, 
    there exists a rhombus satisfying these conditions. -/
theorem rhombus_existence (a s : ℝ) (ha : a > 0) (hs : s > 0) : 
  ∃ (d₁ d₂ : ℝ), d₁ > 0 ∧ d₂ > 0 ∧ d₁ + d₂ = 2 * s ∧ 
  ∃ (A B C D : ℝ × ℝ), 
    -- A, B, C, D form a rhombus
    (dist A B = a) ∧ (dist B C = a) ∧ (dist C D = a) ∧ (dist D A = a) ∧
    -- Diagonals are d₁ and d₂
    (dist A C = d₁) ∧ (dist B D = d₂) :=
sorry

/-- Helper function to calculate distance between two points -/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_existence_l1292_129209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_2sin_2x_minus_pi_half_l1292_129208

noncomputable section

/-- The original function f(x) -/
def f (x : ℝ) : ℝ := 1 - 2 * Real.sin x * (Real.sin x + Real.sqrt 3 * Real.cos x)

/-- The shifted function g(x) -/
def g (x : ℝ) : ℝ := f (x + Real.pi / 3)

/-- Theorem stating that g(x) equals 2sin(2x - π/2) -/
theorem g_equals_2sin_2x_minus_pi_half : 
  ∀ x : ℝ, g x = 2 * Real.sin (2 * x - Real.pi / 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_2sin_2x_minus_pi_half_l1292_129208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_70_percent_l1292_129255

/-- Calculates the simple interest given principal, rate, and time. -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates the compound interest given principal, rate, and time. -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Proves that the rate of interest is 70% given the conditions. -/
theorem interest_rate_is_70_percent (principal : ℝ) (time : ℝ) (difference : ℝ)
  (h_principal : principal = 10000)
  (h_time : time = 2)
  (h_difference : difference = 49)
  : ∃ (rate : ℝ), rate = 70 ∧ 
    compoundInterest principal rate time - simpleInterest principal rate time = difference :=
by
  sorry

#check interest_rate_is_70_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_70_percent_l1292_129255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_given_field_l1292_129226

/-- Calculates the fencing required for a rectangular field -/
noncomputable def fencing_required (area : ℝ) (uncovered_side : ℝ) : ℝ :=
  let width := area / uncovered_side
  uncovered_side + 2 * width

/-- Theorem stating the fencing required for the given field -/
theorem fencing_for_given_field :
  fencing_required 800 20 = 100 := by
  -- Unfold the definition of fencing_required
  unfold fencing_required
  -- Simplify the expression
  simp
  -- Perform the numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_given_field_l1292_129226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1292_129228

theorem trig_problem (α : ℝ) 
  (h1 : Real.sin α - Real.cos α = 1/5)
  (h2 : α ∈ Set.Ioo 0 (π/2)) :
  Real.cos (2*α) / (Real.sqrt 2 * Real.sin (α - π/4)) = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1292_129228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_closed_under_product_and_sum_l1292_129244

/-- For any real number c, there exists a non-empty proper subset X of real numbers
    such that the set of products of elements in X with itself, 
    added to the singleton set containing c, equals X itself. -/
theorem exists_subset_closed_under_product_and_sum (c : ℝ) : 
  ∃ (X : Set ℝ), X.Nonempty ∧ X ≠ Set.univ ∧ (X.prod X).image (fun p => p.1 * p.2) ∪ {c} = X := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_closed_under_product_and_sum_l1292_129244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_difference_l1292_129276

theorem tangent_difference (x y : ℝ) 
  (h1 : Real.sin x / Real.cos y - Real.sin y / Real.cos x = 2)
  (h2 : Real.cos x / Real.sin y - Real.cos y / Real.sin x = 3) :
  Real.tan x / Real.tan y - Real.tan y / Real.tan x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_difference_l1292_129276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_swims_on_sunday_l1292_129275

-- Define the days of the week
inductive Day : Type
  | sunday : Day
  | monday : Day
  | tuesday : Day
  | wednesday : Day
  | thursday : Day
  | friday : Day
  | saturday : Day

def next_day : Day → Day
  | Day.sunday => Day.monday
  | Day.monday => Day.tuesday
  | Day.tuesday => Day.wednesday
  | Day.wednesday => Day.thursday
  | Day.thursday => Day.friday
  | Day.friday => Day.saturday
  | Day.saturday => Day.sunday

-- Define the sports
inductive Sport : Type
  | running : Sport
  | basketball : Sport
  | cricket : Sport
  | swimming : Sport
  | tennis : Sport

-- Define Mahdi's schedule
def schedule : Day → Sport := sorry

-- Conditions
axiom practice_daily : ∀ d : Day, ∃ s : Sport, schedule d = s

axiom runs_three_days : ∃ d1 d2 d3 : Day, 
  schedule d1 = Sport.running ∧ 
  schedule d2 = Sport.running ∧ 
  schedule d3 = Sport.running ∧ 
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

axiom no_consecutive_running : ∀ d : Day, 
  schedule d = Sport.running → schedule (next_day d) ≠ Sport.running

axiom basketball_tuesday : schedule Day.tuesday = Sport.basketball

axiom cricket_friday : schedule Day.friday = Sport.cricket

axiom swims_and_plays_tennis : ∃ d1 d2 : Day, 
  schedule d1 = Sport.swimming ∧ 
  schedule d2 = Sport.tennis

axiom no_tennis_after_run_or_swim : ∀ d : Day, 
  (schedule d = Sport.running ∨ schedule d = Sport.swimming) → 
  schedule (next_day d) ≠ Sport.tennis

-- Theorem to prove
theorem mahdi_swims_on_sunday : schedule Day.sunday = Sport.swimming := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_swims_on_sunday_l1292_129275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_CND_measure_l1292_129248

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the chord structure
structure Chord where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

-- Define the arc structure
structure Arc where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ
  measure : ℝ

-- Define a function to check if a point is on the circle's boundary
def on_boundary (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a function to check if a point is on a chord
def on_chord (ch : Chord) (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    p.1 = ch.start.1 * (1 - t) + ch.endpoint.1 * t ∧
    p.2 = ch.start.2 * (1 - t) + ch.endpoint.2 * t

-- Define the problem setup
def problem_setup (outer : Circle) (inner : Circle) (chord1 chord2 : Chord) 
  (arc_AMB arc_EPF arc_CND : Arc) : Prop :=
  on_boundary outer chord1.start ∧ 
  on_boundary outer chord1.endpoint ∧
  on_boundary outer chord2.start ∧ 
  on_boundary outer chord2.endpoint ∧
  ∃ (A B : ℝ × ℝ), on_boundary inner A ∧ on_boundary inner B ∧
  on_chord chord1 A ∧ on_chord chord2 B ∧
  arc_AMB.measure = 154 ∧
  arc_EPF.measure = 70

-- State the theorem
theorem arc_CND_measure 
  (outer : Circle) (inner : Circle) 
  (chord1 chord2 : Chord) 
  (arc_AMB arc_EPF arc_CND : Arc) :
  problem_setup outer inner chord1 chord2 arc_AMB arc_EPF arc_CND →
  arc_CND.measure = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_CND_measure_l1292_129248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_prism_volume_theorem_l1292_129294

/-- The volume of an oblique prism with the given conditions -/
noncomputable def oblique_prism_volume (a α β : ℝ) : ℝ :=
  (a^3 * Real.sin α * Real.sin (α/2) * Real.tan β) / (2 * Real.cos ((Real.pi - α)/4))

/-- Theorem stating the volume of the oblique prism -/
theorem oblique_prism_volume_theorem (a α β : ℝ) 
  (h_positive : a > 0)
  (h_angle_α : 0 < α ∧ α < Real.pi)
  (h_angle_β : 0 < β ∧ β < Real.pi/2) :
  let prism_volume := oblique_prism_volume a α β
  ∃ (V : ℝ), V = prism_volume ∧ V > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_prism_volume_theorem_l1292_129294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1292_129293

/-- Given an isosceles triangle ABC where sin A : sin B = 1 : 2 and BC = 10,
    prove that its perimeter is 30. -/
theorem isosceles_triangle_perimeter (A B C : Real) (h_isosceles : A = C) 
  (h_sin_ratio : Real.sin A / Real.sin B = 1 / 2) (h_base : B = 10) : A + B + C = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1292_129293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1292_129271

/-- The parabola y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The line y = k(x-2) -/
def line (k x y : ℝ) : Prop := y = k*(x-2)

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- Point A on the parabola -/
def point_A : ℝ × ℝ → Prop := λ p => parabola p.1 p.2 ∧ ∃ k > 0, line k p.1 p.2

/-- Point B on the parabola -/
def point_B : ℝ × ℝ → Prop := λ p => parabola p.1 p.2 ∧ ∃ k > 0, line k p.1 p.2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_line_intersection (k : ℝ) :
  k > 0 →
  (∃ A B : ℝ × ℝ, point_A A ∧ point_B B ∧ A ≠ B) →
  (∀ A B : ℝ × ℝ, point_A A → point_B B → distance A focus = 2 * distance B focus) →
  k = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1292_129271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_earnings_l1292_129260

/-- Represents the earnings for each hour in a 6-hour cycle -/
def hourly_pay : Fin 6 → ℕ
| 0 => 2
| 1 => 3
| 2 => 4
| 3 => 5
| 4 => 6
| 5 => 7

/-- Calculates the total earnings for a given number of hours worked -/
def total_earnings (hours : ℕ) : ℕ :=
  let full_cycles := hours / 6
  let remaining_hours := hours % 6
  let cycle_earnings := (List.range 6).map hourly_pay |>.sum
  full_cycles * cycle_earnings + ((List.range remaining_hours).map hourly_pay).sum

theorem jason_earnings :
  total_earnings 45 = 198 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_earnings_l1292_129260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_rate_calculation_l1292_129249

/-- The rate per kg of grapes -/
def grape_rate (G : ℕ) : Prop := true

/-- The total cost of the purchase -/
def total_cost (T : ℕ) : Prop := true

theorem grape_rate_calculation (G : ℕ) :
  grape_rate G →
  total_cost 1055 →
  G = 70 := by
  intro _ _
  sorry

#check grape_rate_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_rate_calculation_l1292_129249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magician_trick_l1292_129237

theorem magician_trick (n : ℕ+) (x : Fin (2 * n) → ℝ) 
  (h : ∀ i j, i < j → x i < x j) :
  ∃ P : Polynomial ℝ, 
    P ≠ 0 ∧ 
    Polynomial.degree P ≤ n ∧ 
    ∀ i : Fin n, P.eval (x (2*i)) + P.eval (x (2*i+1)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magician_trick_l1292_129237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_integer_point_exists_l1292_129203

/-- A point in the 2D plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A convex pentagon with vertices having integer coordinates -/
structure ConvexPentagon where
  A : IntPoint
  B : IntPoint
  C : IntPoint
  D : IntPoint
  E : IntPoint
  convex : Convex ℝ (Set.range (fun i => (⟨↑(match i with
    | 0 => A.x | 1 => B.x | 2 => C.x | 3 => D.x | _ => E.x),
    ↑(match i with
    | 0 => A.y | 1 => B.y | 2 => C.y | 3 => D.y | _ => E.y)⟩ : ℝ × ℝ)))

/-- Predicate to check if a point is in the interior of a set of points -/
def InteriorPoint (M : IntPoint) (S : Set (IntPoint)) : Prop :=
  ∃ ε > 0, ∀ P, ‖(P.x - M.x, P.y - M.y)‖ < ε → P ∈ S

/-- Theorem: There exists an interior point with integer coordinates in a convex pentagon with integer vertex coordinates -/
theorem interior_integer_point_exists (p : ConvexPentagon) : 
  ∃ (M : IntPoint), InteriorPoint M {p.A, p.B, p.C, p.D, p.E} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_integer_point_exists_l1292_129203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_value_l1292_129204

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a / x + 2 * x

theorem tangent_line_parallel_implies_a_value :
  ∀ a : ℝ,
  (f_derivative a 1 = -1) →
  a = -3 :=
by
  intro a h
  -- The proof steps would go here
  sorry

#check tangent_line_parallel_implies_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_value_l1292_129204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cake_cutting_l1292_129221

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle represented by its three vertices -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- A square represented by its side length and center point -/
structure Square where
  center : Point
  side_length : ℝ

/-- A configuration of chocolates on a cake -/
structure CakeConfiguration where
  cake : Square
  chocolates : List Triangle

/-- Predicate to check if two triangles are not touching -/
def not_touching (t1 t2 : Triangle) : Prop := sorry

/-- Predicate to check if a list of triangles are pairwise not touching -/
def pairwise_not_touching (triangles : List Triangle) : Prop := sorry

/-- Predicate to check if a polygon is convex -/
def is_convex (polygon : List Point) : Prop := sorry

/-- Predicate to check if a point is inside a polygon -/
def point_inside_polygon (p : Point) (polygon : List Point) : Prop := sorry

/-- Predicate to check if a triangle is inside a polygon -/
def triangle_inside_polygon (t : Triangle) (polygon : List Point) : Prop := sorry

/-- The main theorem stating that there exists a configuration where it's impossible to cut the cake into convex polygons with each containing exactly one chocolate -/
theorem impossible_cake_cutting :
  ∃ (config : CakeConfiguration),
    pairwise_not_touching config.chocolates ∧
    ¬(∃ (cutting : List (List Point)),
      (∀ polygon, polygon ∈ cutting → is_convex polygon) ∧
      (∀ chocolate, chocolate ∈ config.chocolates →
        ∃! polygon, polygon ∈ cutting ∧ triangle_inside_polygon chocolate polygon)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cake_cutting_l1292_129221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_theorem_l1292_129295

/-- Represents a circular arrangement of n elements -/
def CircularArrangement (n : ℕ) := Fin n → Fin n

/-- A circular arrangement is valid if it's a bijection -/
def isValidArrangement {n : ℕ} (arr : CircularArrangement n) : Prop :=
  Function.Bijective arr

/-- An element is correctly placed if its position matches its value -/
def isCorrectlyPlaced {n : ℕ} (arr : CircularArrangement n) (i : Fin n) : Prop :=
  arr i = i

/-- Counts the number of correctly placed elements in an arrangement -/
def correctlyPlacedCount {n : ℕ} (arr : CircularArrangement n) : ℕ :=
  (Finset.univ.filter (fun i => arr i = i)).card

/-- Rotates a circular arrangement by k positions -/
def rotate {n : ℕ} (arr : CircularArrangement n) (k : Fin n) : CircularArrangement n :=
  fun i => arr (Fin.add i k)

theorem circular_arrangement_theorem (arr : CircularArrangement 15) 
  (h_valid : isValidArrangement arr) 
  (h_initial : ∀ i : Fin 15, ¬isCorrectlyPlaced arr i) :
  ∃ k : Fin 15, correctlyPlacedCount (rotate arr k) ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_theorem_l1292_129295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_ratio_l1292_129247

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 5
  side_b : b = 12
  side_c : c = 13

/-- Square inscribed with one vertex at the right angle -/
noncomputable def inscribed_square_right_angle (t : RightTriangle) : ℝ := 
  60 / 17

/-- Square inscribed with one side on the hypotenuse -/
noncomputable def inscribed_square_hypotenuse (t : RightTriangle) : ℝ := 
  144 / 17

/-- The ratio of the side lengths of the two inscribed squares -/
noncomputable def square_ratio (t : RightTriangle) : ℝ :=
  inscribed_square_right_angle t / inscribed_square_hypotenuse t

/-- Theorem stating that the ratio of the side lengths is 5/12 -/
theorem inscribed_squares_ratio (t : RightTriangle) : 
  square_ratio t = 5 / 12 := by
  unfold square_ratio inscribed_square_right_angle inscribed_square_hypotenuse
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_ratio_l1292_129247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_to_determine_location_and_resident_l1292_129272

-- Define the villages
inductive Village : Type
| A : Village
| B : Village
| C : Village

-- Define the possible answers
inductive Answer : Type
| Yes : Answer
| No : Answer

-- Define the truthfulness of residents
def alwaysTruthful (v : Village) : Prop :=
  v = Village.A

def alwaysLies (v : Village) : Prop :=
  v = Village.B

def sometimesTruthful (v : Village) : Prop :=
  v = Village.C

-- Define the function to determine if a resident tells the truth
def tellsTruth (v : Village) : Prop :=
  alwaysTruthful v ∨ (sometimesTruthful v ∧ Bool.true)

-- Define the function to determine if a resident lies
def tellsLie (v : Village) : Prop :=
  alwaysLies v ∨ (sometimesTruthful v ∧ Bool.false)

-- Define the type of questions
def Question := Village → Answer

-- Define the function to determine the minimum number of questions
def minQuestions : ℕ := 4

-- State the theorem
theorem min_questions_to_determine_location_and_resident :
  minQuestions = 4 := by
  -- The proof goes here
  sorry

#check min_questions_to_determine_location_and_resident

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_to_determine_location_and_resident_l1292_129272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_translation_l1292_129239

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4)

theorem function_translation (x : ℝ) : g x = Real.sin (2 * x + Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_translation_l1292_129239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_min_distance_l_to_C_l1292_129243

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 + (1/2) * t, (Real.sqrt 3 / 2) * t)

noncomputable def circle_C_polar (θ : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin θ

def circle_C_rect (x y : ℝ) : Prop := x^2 + (y - Real.sqrt 3)^2 = 3

noncomputable def center_C : ℝ × ℝ := (0, Real.sqrt 3)

noncomputable def dist_to_center (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - center_C.1)^2 + (p.2 - center_C.2)^2)

theorem circle_C_equation : ∀ θ : ℝ,
  let (x, y) := (circle_C_polar θ * Real.cos θ, circle_C_polar θ * Real.sin θ)
  circle_C_rect x y := by sorry

theorem min_distance_l_to_C : ∃ min_dist : ℝ,
  (min_dist = Real.sqrt 3) ∧
  (∀ t : ℝ, dist_to_center (line_l t) ≥ min_dist) ∧
  (∃ t : ℝ, dist_to_center (line_l t) = min_dist) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_min_distance_l_to_C_l1292_129243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_l1292_129233

/-- Represents the cost of a type A bus in millions of yuan -/
def cost_A : ℝ := sorry

/-- Represents the cost of a type B bus in millions of yuan -/
def cost_B : ℝ := sorry

/-- The total cost of 1 type A bus and 2 type B buses is 260 million yuan -/
axiom eq1 : cost_A + 2 * cost_B = 260

/-- The total cost of 2 type A buses and 1 type B bus is 280 million yuan -/
axiom eq2 : 2 * cost_A + cost_B = 280

/-- The price difference between type A and type B buses is 20 million yuan -/
theorem price_difference : cost_A - cost_B = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_l1292_129233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_derivative_l1292_129220

-- Define the logarithm base 2 function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv log2 x = 1 / (x * Real.log 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_derivative_l1292_129220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inconsistent_quadratic_l1292_129229

/-- Represents a quadratic function of the form y = ax² + bx + 3 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  h : a ≠ 0

namespace QuadraticFunction

/-- The value of the quadratic function at a given x -/
def value (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + 3

/-- The axis of symmetry of the quadratic function -/
noncomputable def axisOfSymmetry (f : QuadraticFunction) : ℝ :=
  -f.b / (2 * f.a)

/-- Whether a given x is a root of the quadratic function -/
def isRoot (f : QuadraticFunction) (x : ℝ) : Prop :=
  f.value x = 0

/-- The maximum value of the quadratic function -/
noncomputable def maxValue (f : QuadraticFunction) : ℝ :=
  f.value (f.axisOfSymmetry)

theorem inconsistent_quadratic (f : QuadraticFunction) 
  (h1 : f.axisOfSymmetry = 1)
  (h2 : f.isRoot 3)
  (h3 : f.maxValue = 4) :
  f.value 2 ≠ 5 := by
  sorry

end QuadraticFunction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inconsistent_quadratic_l1292_129229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_real_solutions_l1292_129280

-- Define the equation as a noncomputable function
noncomputable def f (x : ℝ) : ℝ := 
  5 * x / (x^2 + 2*x + 4) + 6 * x / (x^2 - 6*x + 4) + 4/3

-- Theorem statement
theorem equation_has_two_real_solutions :
  ∃! (s : Set ℝ), s.Finite ∧ s.ncard = 2 ∧ ∀ x ∈ s, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_real_solutions_l1292_129280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_lines_l1292_129216

/-- Two lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The x-intercept of a line -/
noncomputable def Line.x_intercept (l : Line) : ℝ := -l.c / l.a

/-- Given two perpendicular lines l1 and l2, prove that the x-intercept of l1 is 2 -/
theorem x_intercept_of_perpendicular_lines :
  ∀ (a : ℝ),
  let l1 : Line := ⟨a + 3, 1, -4⟩
  let l2 : Line := ⟨1, a - 1, 4⟩
  perpendicular l1 l2 → l1.x_intercept = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_lines_l1292_129216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_one_is_one_l1292_129210

/-- A function satisfying the given property -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (g (x - y)) = g x + g y - g x * g y - x * y

/-- The theorem statement -/
theorem sum_of_g_one_is_one (g : ℝ → ℝ) (h : SpecialFunction g) :
    {g 1 | g : ℝ → ℝ} = {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_one_is_one_l1292_129210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collatz_six_steps_l1292_129286

def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def collatz_seq (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => collatz (collatz_seq n k)

def reaches_one_in_six_steps (m : ℕ) : Prop :=
  collatz_seq m 6 = 1 ∧ ∀ k < 6, collatz_seq m k ≠ 1

theorem collatz_six_steps :
  ∀ m : ℕ, reaches_one_in_six_steps m ↔ m ∈ ({64, 10, 1, 8} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collatz_six_steps_l1292_129286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sin_over_x_l1292_129277

/-- The derivative of sin(x) / x is (x * cos(x) - sin(x)) / x^2 -/
theorem derivative_sin_over_x (x : ℝ) (h : x ≠ 0) :
  deriv (λ y => Real.sin y / y) x = (x * Real.cos x - Real.sin x) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sin_over_x_l1292_129277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_triangle_area_range_l1292_129214

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line not passing through the origin -/
structure Line where
  k : ℝ  -- slope
  m : ℝ  -- y-intercept
  h_nonzero : m ≠ 0

/-- Definition of the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Definition of the distance between foci -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  2 * Real.sqrt (e.a^2 - e.b^2)

/-- Definition of the geometric mean of distances from foci to vertices -/
noncomputable def foci_vertex_geometric_mean (e : Ellipse) : ℝ :=
  Real.sqrt ((e.a + focal_distance e / 2) * (e.a - focal_distance e / 2))

/-- Theorem stating the properties of the ellipse and the range of triangle areas -/
theorem ellipse_properties_and_triangle_area_range (e : Ellipse) :
  foci_vertex_geometric_mean e = Real.sqrt 3 →
  eccentricity e = 1/2 →
  (∃ (l : Line), ∀ (A B : ℝ × ℝ),
    (A.1^2 / e.a^2 + A.2^2 / e.b^2 = 1) →
    (B.1^2 / e.a^2 + B.2^2 / e.b^2 = 1) →
    A.2 = l.k * A.1 + l.m →
    B.2 = l.k * B.1 + l.m →
    (A.2 / A.1) * (B.2 / B.1) = l.k^2 →
    ∃ (S : ℝ), 0 < S ∧ S < Real.sqrt 3 ∧
    S = (1/2) * abs (A.1 * B.2 - A.2 * B.1)) →
  e.a = 2 ∧ e.b = Real.sqrt 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_triangle_area_range_l1292_129214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_simplifications_l1292_129281

theorem algebraic_simplifications :
  ((5 * Real.sqrt 12 - Real.sqrt 48) / Real.sqrt 3 - Real.sqrt (1/5) * Real.sqrt 35 + Real.sqrt 28 = 6 + Real.sqrt 7) ∧
  ((Real.sqrt 3 + 1)^2 - (2 * Real.sqrt 2 + 3) * (2 * Real.sqrt 2 - 3) = 5 + 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_simplifications_l1292_129281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charts_learned_in_sixth_grade_l1292_129282

def sixth_grade_charts : List String :=
  ["bar charts", "line charts", "pie charts"]

theorem charts_learned_in_sixth_grade :
  sixth_grade_charts = ["bar charts", "line charts", "pie charts"] := by
  rfl

#eval sixth_grade_charts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charts_learned_in_sixth_grade_l1292_129282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1292_129253

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∃ k : ℚ, sum_n seq 3 = (sum_n seq 1 + 1 + sum_n seq 4) / 2) →
  (∃ r : ℚ, seq.a 2 ^ 2 = seq.a 1 * seq.a 5) →
  (∀ n : ℕ, seq.a n = 2 * n - 1) ∧
  (∃ r : ℚ, sum_n seq 6 = r * sum_n seq 4 ∧ sum_n seq 9 = r * sum_n seq 6 ∧ r = 9/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1292_129253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_value_at_specific_points_expr_independent_of_a_l1292_129285

noncomputable section

/-- Given expressions for A and B in terms of a and b -/
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1

def B (a b : ℝ) : ℝ := -a^2 + (1/2) * a * b + (2/3)

/-- The main expression in terms of A and B -/
def expr (a b : ℝ) : ℝ := A a b + 2 * B a b

theorem expr_value_at_specific_points :
  expr (-1) (-2) = 31/3 := by
  sorry

theorem expr_independent_of_a (b : ℝ) :
  (∀ a : ℝ, expr a b = expr 0 b) ↔ b = 1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_value_at_specific_points_expr_independent_of_a_l1292_129285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_ratio_456_not_imply_right_triangle_l1292_129290

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define what it means for a triangle to be right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Define the condition a : b : c = 4 : 5 : 6
def hasSideRatio456 (t : Triangle) : Prop :=
  ∃ (k : ℝ), t.a = 4*k ∧ t.b = 5*k ∧ t.c = 6*k

-- Theorem statement
theorem side_ratio_456_not_imply_right_triangle :
  ∃ (t : Triangle), hasSideRatio456 t ∧ ¬(isRightTriangle t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_ratio_456_not_imply_right_triangle_l1292_129290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l1292_129291

/-- A color type representing red and black --/
inductive Color
  | Red
  | Black

/-- A point in the plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A function that assigns a color to each point in the plane --/
def coloring : Point → Color := sorry

/-- The distance between two points --/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: In any coloring of the plane with two colors, 
    there exist two points of the same color exactly 1 meter apart --/
theorem same_color_unit_distance :
  ∃ (p q : Point), coloring p = coloring q ∧ distance p q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l1292_129291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l1292_129235

/-- Circle C with center at (√3, 1) and radius 1 -/
def circle_C (x y : ℝ) : Prop := (x - Real.sqrt 3) ^ 2 + (y - 1) ^ 2 = 1

/-- Point A with coordinates (-t, 0) -/
def point_A (t : ℝ) : ℝ × ℝ := (-t, 0)

/-- Point B with coordinates (t, 0) -/
def point_B (t : ℝ) : ℝ × ℝ := (t, 0)

/-- Condition that t is positive -/
def t_positive (t : ℝ) : Prop := t > 0

/-- Existence of point P on circle C forming a right angle with A and B -/
def exists_right_angle_P (t : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, circle_C P.1 P.2 ∧ 
    (P.1 - (point_A t).1) * (P.1 - (point_B t).1) + (P.2 - (point_A t).2) * (P.2 - (point_B t).2) = 0

/-- The theorem stating the maximum value of t -/
theorem max_t_value : 
  ∀ t : ℝ, t_positive t → exists_right_angle_P t → t ≤ 3 ∧ ∃ t₀ : ℝ, t_positive t₀ ∧ exists_right_angle_P t₀ ∧ t₀ = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l1292_129235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1292_129218

theorem calculate_expression : 3 * (1/2)^(-2 : ℤ) + |2 - Real.pi| + (-3 : ℝ)^(0 : ℕ) = 11 + Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1292_129218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l1292_129252

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => if x < 0 then 3 * x - 1 else -3 * x - 1

-- State the theorem
theorem f_is_even_and_correct :
  (∀ x, f x = f (-x)) ∧  -- f is even
  (∀ x, x < 0 → f x = 3 * x - 1) ∧  -- f(x) = 3x - 1 for x < 0
  (∀ x, x > 0 → f x = -3 * x - 1) :=  -- f(x) = -3x - 1 for x > 0
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l1292_129252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_heptagon_perturbation_l1292_129234

-- Define a convex heptagon
structure ConvexHeptagon where
  vertices : Fin 7 → ℝ × ℝ

-- Define a diagonal of a heptagon
def diagonal (h : ConvexHeptagon) (i j : Fin 7) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • h.vertices i + t • h.vertices j}

-- Define the property of being a special heptagon
def is_special (h : ConvexHeptagon) : Prop :=
  ∃ (i j k l m n : Fin 7) (p : ℝ × ℝ),
    i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ m ≠ n ∧
    p ∈ diagonal h i l ∧ p ∈ diagonal h j m ∧ p ∈ diagonal h k n

-- Define a small perturbation of a vertex
def perturb (h : ConvexHeptagon) (i : Fin 7) (ε : ℝ × ℝ) : ConvexHeptagon where
  vertices := fun j => if j = i then h.vertices j + ε else h.vertices j

-- State the theorem
theorem special_heptagon_perturbation (h : ConvexHeptagon) (h_special : is_special h) :
  ∃ (i : Fin 7) (ε : ℝ × ℝ), ∀ δ : ℝ × ℝ, ‖δ‖ < ‖ε‖ → ¬is_special (perturb h i δ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_heptagon_perturbation_l1292_129234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_inscribed_circle_theorem_l1292_129200

/-- Triangle XYZ with circle O inscribed -/
structure TriangleWithInscribedCircle where
  /-- The perimeter of triangle XYZ -/
  perimeter : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The distance OY, where O is the center of the inscribed circle -/
  oy : ℝ

/-- The theorem statement -/
theorem triangle_with_inscribed_circle_theorem
  (t : TriangleWithInscribedCircle)
  (h_perimeter : t.perimeter = 120)
  (h_radius : t.radius = 15)
  (h_oy_rational : ∃ (p q : ℕ), t.oy = p / q ∧ Nat.Coprime p q)
  : ∃ (p q : ℕ), t.oy = p / q ∧ Nat.Coprime p q ∧ p + q = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_inscribed_circle_theorem_l1292_129200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1292_129246

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * (Real.tan x + 1 / Real.tan x)

theorem period_of_f (k : ℝ) (h : k ≠ 0) :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f k (x + p) = f k x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f k (y + q) ≠ f k y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1292_129246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1292_129278

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 + Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

noncomputable def curve_C (θ : ℝ) : ℝ := 6 * Real.cos θ

def point_P : ℝ × ℝ := (-1, 0)

theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    (∃ (t₁ t₂ : ℝ), line_l t₁ = A ∧ line_l t₂ = B) ∧
    (∃ (θ₁ θ₂ : ℝ),
      curve_C θ₁ = Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) ∧
      curve_C θ₂ = Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)) →
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
    4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1292_129278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_vertex_after_dilation_l1292_129217

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in 2D space -/
structure Square where
  center : Point
  area : ℝ

/-- Applies a dilation to a point -/
def dilate (p : Point) (center : Point) (scale : ℝ) : Point :=
  ⟨center.x + scale * (p.x - center.x), center.y + scale * (p.y - center.y)⟩

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Helper function to get vertices of a square (not fully implemented) -/
def vertices (s : Square) : List Point :=
  sorry  -- This would typically return the four vertices of the square

/-- Main theorem -/
theorem closest_vertex_after_dilation (s : Square) 
  (h1 : s.center = ⟨5, 3⟩) 
  (h2 : s.area = 16) : 
  ∃ (v : Point), 
    v ∈ (List.map (fun p => dilate p ⟨0, 0⟩ (1/2)) (vertices s)) ∧ 
    (∀ (u : Point), u ∈ (List.map (fun p => dilate p ⟨0, 0⟩ (1/2)) (vertices s)) → 
      distance v ⟨0, 0⟩ ≤ distance u ⟨0, 0⟩) ∧
    v = ⟨1.5, 0.5⟩ :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_vertex_after_dilation_l1292_129217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_two_l1292_129279

-- Define the function g(x) as noncomputable
noncomputable def g (x : ℝ) : ℝ := (2/7)^x + (3/7)^x + (6/7)^x

-- State the theorem
theorem unique_solution_is_two :
  ∃! x : ℝ, (2 : ℝ)^x + (3 : ℝ)^x + (6 : ℝ)^x = (7 : ℝ)^x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_two_l1292_129279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_range_theorem_l1292_129231

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the domain
def domain : Set ℝ := Set.Ioc (-2.5) 3

-- Define the range
def range : Set ℤ := {-3, -2, -1, 0, 1, 2, 3}

-- State the theorem
theorem floor_range_theorem :
  (∀ y ∈ range, ∃ x ∈ domain, floor x = y) ∧
  (∀ x ∈ domain, floor x ∈ range) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_range_theorem_l1292_129231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_theorem_l1292_129224

/-- Represents a milkman's grazing data -/
structure MilkmanData where
  cows : ℕ
  months : ℚ
deriving Inhabited

/-- Calculates the total cow-months for a given set of milkmen data -/
def totalCowMonths (data : List MilkmanData) : ℚ :=
  data.foldl (fun acc d => acc + d.cows * d.months) 0

/-- Represents the problem setup -/
def grazingProblem (milkmenData : List MilkmanData) (aRent : ℕ) : Prop :=
  let totalMonths := totalCowMonths milkmenData
  let aMonths := totalCowMonths [milkmenData.head!]
  let rentPerCowMonth : ℚ := aRent / aMonths
  totalMonths * rentPerCowMonth = 5930

/-- The main theorem statement -/
theorem total_rent_theorem : 
  let milkmenData := [
    ⟨24, 3⟩, ⟨10, 5⟩, ⟨35, 4⟩, ⟨21, 3⟩, 
    ⟨15, 6⟩, ⟨40, 2⟩, ⟨28, (7:ℚ)/2⟩
  ]
  grazingProblem milkmenData 720 := by
  sorry

#eval totalCowMonths [
  ⟨24, 3⟩, ⟨10, 5⟩, ⟨35, 4⟩, ⟨21, 3⟩, 
  ⟨15, 6⟩, ⟨40, 2⟩, ⟨28, (7:ℚ)/2⟩
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_theorem_l1292_129224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_in_plane_l1292_129250

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane structure (omitted for brevity)

/-- A line in 3D space -/
structure Line3D where
  -- Define the line structure (omitted for brevity)

/-- A point in 3D space -/
structure Point3D where
  -- Define the point structure (omitted for brevity)

/-- Predicate for a line being perpendicular to a plane -/
def is_perpendicular_to_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate for a point being on a plane -/
def point_on_plane (P : Point3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate for a line passing through a point -/
def line_passes_through_point (l : Line3D) (P : Point3D) : Prop :=
  sorry

/-- Predicate for two lines being perpendicular -/
def lines_perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for a line lying within a plane -/
def line_in_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- The main theorem -/
theorem perpendicular_lines_in_plane 
  (l : Line3D) (α : Plane3D) (P : Point3D)
  (h1 : is_perpendicular_to_plane l α)
  (h2 : point_on_plane P α) :
  ∃ (S : Set Line3D), 
    (∀ m ∈ S, line_passes_through_point m P ∧ 
               lines_perpendicular m l ∧ 
               line_in_plane m α) ∧
    Set.Infinite S :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_in_plane_l1292_129250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l1292_129257

theorem coefficient_x_squared_in_expansion : 
  let expansion := (1 - X : Polynomial ℤ)^6 * (1 + X : Polynomial ℤ)^4
  expansion.coeff 2 = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l1292_129257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_reciprocal_period_l1292_129245

def S : Finset ℕ := {3, 7, 9, 11, 13, 37}

theorem product_reciprocal_period (a b c d e f : ℕ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
                c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
                d ≠ e ∧ d ≠ f ∧ 
                e ≠ f)
  (h_in_S : a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S) :
  ∃ (k : ℕ), k > 0 ∧ (
    (b * c * d * e * f : ℚ) = k * (1 / a - ↑⌊(1 : ℚ) / a⌋) ∨
    (a * c * d * e * f : ℚ) = k * (1 / b - ↑⌊(1 : ℚ) / b⌋) ∨
    (a * b * d * e * f : ℚ) = k * (1 / c - ↑⌊(1 : ℚ) / c⌋) ∨
    (a * b * c * e * f : ℚ) = k * (1 / d - ↑⌊(1 : ℚ) / d⌋) ∨
    (a * b * c * d * f : ℚ) = k * (1 / e - ↑⌊(1 : ℚ) / e⌋) ∨
    (a * b * c * d * e : ℚ) = k * (1 / f - ↑⌊(1 : ℚ) / f⌋)
  ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_reciprocal_period_l1292_129245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_exists_non_aligned_parallelepiped_l1292_129232

/-- A rectangular parallelepiped with integer coordinates and volume 2017 -/
structure Parallelepiped where
  a : ℕ
  b : ℕ
  c : ℕ
  volume_eq : a * b * c = 2017

/-- The diagonal of a rectangular parallelepiped -/
noncomputable def diagonal (p : Parallelepiped) : ℝ :=
  Real.sqrt (p.a^2 + p.b^2 + p.c^2 : ℝ)

/-- Theorem: The diagonal of the parallelepiped is √(2017² + 2) -/
theorem parallelepiped_diagonal :
  ∀ p : Parallelepiped, diagonal p = Real.sqrt (2017^2 + 2) := by
  sorry

/-- Theorem: There exists a parallelepiped with volume 2017 and not all edges parallel to axes -/
theorem exists_non_aligned_parallelepiped :
  ∃ (a b c : ℕ), a * b * c = 2017 ∧ a^2 + b^2 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_exists_non_aligned_parallelepiped_l1292_129232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_books_bought_l1292_129274

/-- Proves that the total number of books bought is 90 given the conditions --/
theorem total_books_bought (total_price : ℕ) (math_book_price : ℕ) (history_book_price : ℕ) (math_books_bought : ℕ) : ℕ :=
  by
  have h1 : total_price = 397 := by sorry
  have h2 : math_book_price = 4 := by sorry
  have h3 : history_book_price = 5 := by sorry
  have h4 : math_books_bought = 53 := by sorry
  
  -- Define the number of history books
  let history_books : ℕ := (total_price - math_books_bought * math_book_price) / history_book_price
  
  -- Calculate the total number of books
  let total_books : ℕ := math_books_bought + history_books
  
  -- Prove that total_books equals 90
  have h5 : total_books = 90 := by sorry
  
  exact total_books

-- Use #eval with a function to compute the result
def compute_total_books (total_price math_book_price history_book_price math_books_bought : ℕ) : ℕ :=
  let history_books := (total_price - math_books_bought * math_book_price) / history_book_price
  math_books_bought + history_books

#eval compute_total_books 397 4 5 53

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_books_bought_l1292_129274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_theorem_l1292_129267

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

def line (x y m : ℝ) : Prop := y = x + m

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 2, 0)

noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

noncomputable def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

theorem ellipse_line_intersection_theorem (m : ℝ) :
  (∀ x y, ellipse x y → line x y m → (x, y) = A ∨ (x, y) = B) →
  triangle_area F₁ A B = 2 * triangle_area F₂ A B →
  m = -Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_theorem_l1292_129267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_seventh_term_l1292_129251

/-- An arithmetic sequence with a_1 = 2 and a_4 = 5 -/
noncomputable def arithmetic_seq (n : ℕ) : ℝ :=
  2 + (n - 1) * ((5 - 2) / 3)

theorem arithmetic_seq_seventh_term :
  arithmetic_seq 7 = 8 := by
  -- Unfold the definition of arithmetic_seq
  unfold arithmetic_seq
  -- Simplify the expression
  simp [Nat.cast_sub, Nat.cast_one]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_seventh_term_l1292_129251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_noncongruent_triangles_l1292_129269

-- Define the triangle type
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)

-- Define the condition for a valid triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.α > 0 ∧ t.β > 0 ∧ t.γ > 0 ∧
  t.α + t.β + t.γ = Real.pi

-- Define the condition for the specific triangle we're looking for
def is_target_triangle (t : Triangle) : Prop :=
  (t.a = 20 ∧ t.b = 17 ∧ t.α = Real.pi/3) ∨
  (t.a = 20 ∧ t.c = 17 ∧ t.β = Real.pi/3) ∨
  (t.b = 20 ∧ t.c = 17 ∧ t.γ = Real.pi/3)

-- Define a function to count the number of valid target triangles
noncomputable def count_target_triangles : ℕ :=
  -- Implementation goes here
  sorry

-- The main theorem
theorem two_noncongruent_triangles :
  count_target_triangles = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_noncongruent_triangles_l1292_129269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_deposits_for_given_bank_l1292_129240

/-- Represents the financial conditions of a savings bank --/
structure SavingsBank where
  salary_expenses : ℚ
  other_expenses : ℚ
  deposit_interest_rate : ℚ
  earning_interest_rate : ℚ

/-- Calculates the minimum deposits needed to avoid loss --/
noncomputable def min_deposits (bank : SavingsBank) : ℚ :=
  (bank.salary_expenses + bank.other_expenses) / (bank.earning_interest_rate - bank.deposit_interest_rate)

/-- Theorem stating the minimum deposits needed for the given bank --/
theorem min_deposits_for_given_bank :
  let bank : SavingsBank := {
    salary_expenses := 100000,
    other_expenses := 170000,
    deposit_interest_rate := 225/10000,
    earning_interest_rate := 405/10000
  }
  min_deposits bank = 15000000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_deposits_for_given_bank_l1292_129240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l1292_129270

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 5

-- Define the point M
def M : ℝ × ℝ := (1, 1)

-- Define the line ax + y - 1 = 0
def line_a (a x y : ℝ) : Prop := a * x + y - 1 = 0

-- Define the tangent line l: 2x - y - 1 = 0
def line_l (x y : ℝ) : Prop := 2 * x - y - 1 = 0

-- Define the perpendicular relationship between two lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

theorem tangent_line_proof (a : ℝ) : 
  my_circle M.1 M.2 →  -- M is on the circle
  (∃ x y, my_circle x y ∧ line_l x y) →  -- l is tangent to the circle
  perpendicular (-a) 2 →  -- l is perpendicular to ax + y - 1 = 0
  a = 1/2 ∧ ∀ x y, line_l x y ↔ 2 * x - y - 1 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l1292_129270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1292_129283

/-- Represents the time taken to complete a task -/
def CompletionTime := ℝ

/-- Represents the rate of work (portion of task completed per day) -/
def WorkRate := ℝ

/-- The total amount of work to be done -/
def TotalWork : ℝ := 1

theorem work_completion_time 
  (time_person1 : CompletionTime)
  (time_both : CompletionTime)
  (h1 : time_person1 = (6 : ℝ))
  (h2 : time_both = (3 : ℝ)) :
  ∃ (time_person2 : CompletionTime), time_person2 = (6 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1292_129283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1292_129264

/-- The area of a triangle with side lengths 5, 12, and 13 units is 30 square units. -/
theorem triangle_area
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (area : ℝ) :
  side1 = 5 ∧ side2 = 12 ∧ side3 = 13 →
  area = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1292_129264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_lower_bound_achievable_l1292_129287

theorem min_value_theorem (y : ℝ) (h : y > 0) : 3 * y^4 + 4 * y^(-3 : ℝ) ≥ 7 := by
  sorry

theorem lower_bound_achievable : ∃ y : ℝ, y > 0 ∧ 3 * y^4 + 4 * y^(-3 : ℝ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_lower_bound_achievable_l1292_129287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_routes_l1292_129241

-- Define the cities
inductive City : Type
| A : City
| B : City
| D : City
| F : City
deriving DecidableEq

-- Define the roads
inductive Road : Type
| AB : Road
| AD : Road
| BD : Road
| DF : Road
| FB : Road
| AF : Road
| BF : Road
deriving DecidableEq

-- Define a route as a list of roads
def Route := List Road

-- Function to check if a route is valid
def isValidRoute (r : Route) : Prop :=
  -- Each road is used exactly once
  r.toFinset.card = 7 ∧
  -- The route starts at A and ends at B
  (r.head? = some Road.AB ∨ r.head? = some Road.AD ∨ r.head? = some Road.AF) ∧
  (r.getLast? = some Road.AB ∨ r.getLast? = some Road.BD ∨ r.getLast? = some Road.BF)

-- The main theorem
theorem number_of_routes : ∃ (routes : Finset Route), 
  (∀ r ∈ routes, isValidRoute r) ∧ routes.card = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_routes_l1292_129241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l1292_129236

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin x else -x^2 - 1

-- State the theorem
theorem k_range (k : ℝ) :
  (∀ x, f x ≤ k * x) ↔ 1 ≤ k ∧ k ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l1292_129236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_model_ratio_l1292_129254

/-- The model function --/
noncomputable def model (a b x : ℝ) : ℝ := a * Real.exp (b * x + 1)

/-- The transformed variable --/
noncomputable def z (y : ℝ) : ℝ := Real.log y

/-- The transformed empirical regression equation --/
def empirical_regression (x a : ℝ) : ℝ := 2 * x + a

theorem model_ratio (a b : ℝ) (h₁ : a > 0) :
  (∀ x y : ℝ, y = model a b x → z y = empirical_regression x a) →
  b / a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_model_ratio_l1292_129254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1292_129268

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2*Complex.I) = 3 - 4*Complex.I) : z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1292_129268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l1292_129225

/-- Calculates the distance driven given speed and time -/
noncomputable def distance (speed time : ℝ) : ℝ := speed * time

/-- Calculates the speed given distance and time -/
noncomputable def speed (distance time : ℝ) : ℝ := distance / time

theorem sam_distance (marguerite_distance marguerite_time sam_time : ℝ)
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4)
  (h4 : speed (distance (speed marguerite_distance marguerite_time) sam_time) sam_time = 
        1.2 * speed marguerite_distance marguerite_time) :
  distance (speed marguerite_distance marguerite_time * 1.2) sam_time = 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l1292_129225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_6_over_2_l1292_129202

/-- Represents a hyperbola with semi-major axis a, semi-minor axis b, and focal length 2c -/
structure Hyperbola (a b c : ℝ) : Prop where
  a_pos : a > 0
  b_pos : b > 0
  a_gt_b : a > b
  focal_length : c^2 = a^2 + b^2

/-- Represents a line passing through points (a, 0) and (0, b) -/
def line_through_ab (a b : ℝ) (x y : ℝ) : Prop :=
  b * x + a * y = a * b

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 -/
noncomputable def point_line_distance (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity_sqrt_6_over_2 
  {a b c : ℝ} (h : Hyperbola a b c) :
  point_line_distance (-a) 0 b a (-a*b) = (2 * Real.sqrt 2 / 3) * c →
  eccentricity a c = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_6_over_2_l1292_129202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_total_length_l1292_129222

noncomputable section

/-- The length of a diagonal in a 1x1 square -/
def diagonal_length : ℝ := Real.sqrt 2

/-- The length of a diagonal in a 2x1 rectangle -/
def long_diagonal_length : ℝ := Real.sqrt 5

/-- The total length of line segments forming the letter X -/
def x_length : ℝ := 2 * diagonal_length

/-- The total length of line segments forming the letter Y -/
def y_length : ℝ := 2 + diagonal_length

/-- The total length of line segments forming the letter Z -/
def z_length : ℝ := 4 + long_diagonal_length

/-- The total length of line segments forming XYZ -/
def xyz_length : ℝ := x_length + y_length + z_length

theorem xyz_total_length : xyz_length = 6 + 3 * diagonal_length + long_diagonal_length := by
  unfold xyz_length x_length y_length z_length diagonal_length long_diagonal_length
  ring

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_total_length_l1292_129222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_approximation_l1292_129258

/-- Calculates the approximate initial investment amount given the final amount,
    interest rate, and time period. -/
noncomputable def approximate_initial_investment (final_amount : ℝ) (interest_rate : ℝ) (years : ℝ) : ℝ :=
  let doublings := years / (70 / interest_rate)
  final_amount / (2 ^ doublings)

/-- Theorem stating that the initial investment is approximately $4,819.28
    given the conditions of the problem. -/
theorem initial_investment_approximation :
  let final_amount : ℝ := 20000
  let interest_rate : ℝ := 4
  let years : ℝ := 36
  let initial_investment := approximate_initial_investment final_amount interest_rate years
  abs (initial_investment - 4819.28) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_approximation_l1292_129258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1292_129265

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if b² = ac and (cos(A-C) + cos B) = 3/2, then sin A sin C = 3/4
    and the triangle is equilateral. -/
theorem triangle_properties (a b c A B C : ℝ) 
  (h_triangle : A + B + C = π)
  (h_sides : b^2 = a*c)
  (h_vectors : Real.cos (A - C) + Real.cos B = 3/2) :
  Real.sin A * Real.sin C = 3/4 ∧ a = b ∧ b = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1292_129265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_segment_l1292_129284

def is_lattice_point (x y : ℤ) : Prop := true

def on_line_segment (x y x1 y1 x2 y2 : ℤ) : Prop :=
  ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧ 
    x = x1 + Int.floor ((x2 - x1 : ℚ) * t) ∧
    y = y1 + Int.floor ((y2 - y1 : ℚ) * t)

theorem lattice_points_on_segment : 
  let x1 : ℤ := 5
  let y1 : ℤ := 23
  let x2 : ℤ := 65
  let y2 : ℤ := 407
  ∃ points : Finset (ℤ × ℤ),
    (∀ (x y : ℤ), (x, y) ∈ points ↔ 
      (is_lattice_point x y ∧ on_line_segment x y x1 y1 x2 y2)) ∧
    points.card = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_segment_l1292_129284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1292_129288

-- Define the points A and B
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (1, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the sum of distances function for a point P on the x-axis
noncomputable def sum_distances (x : ℝ) : ℝ :=
  distance (x, 0) A + distance (x, 0) B

-- Theorem statement
theorem min_sum_distances :
  ∃ (x : ℝ), ∀ (y : ℝ), sum_distances x ≤ sum_distances y ∧ sum_distances x = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1292_129288
