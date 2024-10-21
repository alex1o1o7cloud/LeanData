import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l224_22481

/-- Calculates the selling price of an item given its original price and loss percentage. -/
noncomputable def sellingPrice (originalPrice : ℝ) (lossPercentage : ℝ) : ℝ :=
  originalPrice * (1 - lossPercentage / 100)

/-- Theorem stating that the selling price of a cycle bought for Rs. 2000 and sold at a 10% loss is Rs. 1800. -/
theorem cycle_selling_price :
  sellingPrice 2000 10 = 1800 := by
  -- Unfold the definition of sellingPrice
  unfold sellingPrice
  -- Simplify the arithmetic
  simp [mul_sub, mul_div_cancel']
  -- Check that 2000 * (1 - 10 / 100) = 1800
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l224_22481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l224_22450

/-- Function to calculate the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Function to check if a number satisfies the given conditions -/
def satisfiesConditions (n : ℕ) : Bool :=
  n < 1000 && n > 0 && sumOfDigits n % 7 = 0 && n % 3 = 0

/-- The main theorem -/
theorem count_satisfying_numbers :
  (Finset.filter (fun n => satisfiesConditions n) (Finset.range 1000)).card = 33 := by
  sorry

#eval (Finset.filter (fun n => satisfiesConditions n) (Finset.range 1000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l224_22450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pb_length_l224_22404

/-- Rectangle ABCD with point P inside -/
structure Rectangle :=
  (A B C D P : ℝ × ℝ)

/-- The length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating the relationship between PB and other given lengths -/
theorem pb_length (rect : Rectangle) : 
  distance rect.A rect.B = 2 * distance rect.B rect.C →
  distance rect.P rect.A = 5 →
  distance rect.P rect.D = 12 →
  distance rect.P rect.C = 13 →
  distance rect.P rect.B = Real.sqrt (551/5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pb_length_l224_22404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mo_tea_consumption_l224_22417

/-- Represents the number of cups of hot chocolate Mo drinks on a rainy morning -/
def n : ℕ := sorry

/-- Represents the number of cups of tea Mo drinks on a non-rainy morning -/
def t : ℕ := sorry

/-- The total number of cups Mo drank last week -/
def total_cups : ℕ := 22

/-- The difference between tea and hot chocolate cups -/
def tea_hc_difference : ℕ := 8

/-- The number of rainy days last week -/
def rainy_days : ℕ := 4

/-- The number of non-rainy days last week -/
def non_rainy_days : ℕ := 7 - rainy_days

theorem mo_tea_consumption :
  (rainy_days * n + non_rainy_days * t = total_cups) →
  (non_rainy_days * t = rainy_days * n + tea_hc_difference) →
  t = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mo_tea_consumption_l224_22417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_points_l224_22429

theorem chess_tournament_points (x : ℕ) (p : ℕ) : 
  x > 0 →  -- There is at least one 9th grader
  10 * x = 10 * x →  -- Number of 10th graders is 10 times the number of 9th graders
  p + (9 * p / 2) = (11 * p / 2) →  -- Total points is 5.5 times the points scored by 9th graders
  p = 10  -- 9th graders scored 10 points
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_points_l224_22429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_greater_sin_l224_22462

theorem tan_half_greater_sin (x : ℝ) : 
  (∃ k : ℤ, x ∈ Set.Ioo ((k : ℝ) * π - π / 2) (k * π)) ↔ 
  (Real.tan (x / 2) > Real.sin x ∧ x ∉ {y | ∃ k : ℤ, y = (2 * k + 1) * π}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_greater_sin_l224_22462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l224_22454

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.sqrt x + 4 / Real.sqrt x - 2

-- State the theorem
theorem min_value_of_f :
  (∀ x > 0, f x ≥ 2) ∧ (∃ x > 0, f x = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l224_22454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_exactly_sqrt_49_eq_7_eleven_divides_15_plus_sqrt_49_l224_22442

theorem divide_exactly (a b : ℕ) : a ∣ b ↔ b % a = 0 := by sorry

theorem sqrt_49_eq_7 : Real.sqrt 49 = 7 := by sorry

theorem eleven_divides_15_plus_sqrt_49 :
  11 ∣ Int.floor (15 + Real.sqrt 49) :=
by
  have h1 : Real.sqrt 49 = 7 := sqrt_49_eq_7
  have h2 : 15 + Real.sqrt 49 = 22 := by
    rw [h1]
    norm_num
  have h3 : Int.floor (15 + Real.sqrt 49) = 22 := by
    rw [h2]
    norm_num
  rw [h3]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_exactly_sqrt_49_eq_7_eleven_divides_15_plus_sqrt_49_l224_22442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_selling_price_before_sale_l224_22419

/-- Represents the price of a kite -/
structure KitePrice where
  cost : ℝ
  initialGainPercent : ℝ
  saleDiscountPercent : ℝ
  saleGainPercent : ℝ

/-- Calculates the selling price before the clearance sale -/
noncomputable def sellingPriceBeforeSale (p : KitePrice) : ℝ :=
  p.cost * (1 + p.initialGainPercent / 100)

/-- Calculates the selling price during the clearance sale -/
noncomputable def sellingPriceDuringSale (p : KitePrice) : ℝ :=
  sellingPriceBeforeSale p * (1 - p.saleDiscountPercent / 100)

/-- Theorem stating the selling price before the clearance sale is $135 -/
theorem kite_selling_price_before_sale (p : KitePrice)
  (h1 : p.initialGainPercent = 35)
  (h2 : p.saleDiscountPercent = 10)
  (h3 : p.saleGainPercent = 21.5)
  (h4 : sellingPriceDuringSale p = p.cost * (1 + p.saleGainPercent / 100)) :
  sellingPriceBeforeSale p = 135 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_selling_price_before_sale_l224_22419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_addition_formula_l224_22459

theorem tan_addition_formula (x : Real) (h : Real.tan x = Real.sqrt 3) :
  Real.tan (x + π / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_addition_formula_l224_22459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_identity_l224_22458

noncomputable def g (n : ℤ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem g_identity (n : ℤ) : g (n + 2) - g (n - 2) = 3 * g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_identity_l224_22458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l224_22412

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the point P
def point_P : ℝ × ℝ := (3, 1)

-- Define a line passing through P and tangent to C at two points
def tangent_line (A B : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
  (∃ (m : ℝ), (A.2 - point_P.2) = m * (A.1 - point_P.1) ∧
               (B.2 - point_P.2) = m * (B.1 - point_P.1))

-- Helper definition for a line through two points
def line_through (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (t : ℝ), (x, y) = (1 - t) • A + t • B}

-- The theorem to be proved
theorem tangent_line_equation :
  ∀ (A B : ℝ × ℝ), tangent_line A B →
  (∀ (x y : ℝ), (x, y) ∈ line_through A B ↔ 2*x + y - 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l224_22412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l224_22438

/-- An inverse proportion function passing through a specific point -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

/-- The point through which the function passes -/
def point (k n : ℝ) : ℝ × ℝ := (2, k - n^2 - 2)

/-- Theorem stating the minimum value of k -/
theorem min_k_value (k n : ℝ) (h1 : k ≠ 0) 
  (h2 : inverse_proportion k (point k n).1 = (point k n).2) :
  k ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l224_22438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_result_l224_22446

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; -1, 5]
def B : Matrix (Fin 2) (Fin 1) ℤ := !![4; -3]

theorem matrix_product_result : A * B = !![18; -19] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_result_l224_22446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l224_22445

-- Define the sample space for throwing a fair dice twice
def Ω : Type := Fin 6 × Fin 6

-- Define event A: both numbers are odd
def A (ω : Ω) : Prop := Odd ω.1.val ∧ Odd ω.2.val

-- Define event B: the sum of the two numbers is 4
def B (ω : Ω) : Prop := ω.1.val + ω.2.val = 4

-- Define the probability measure for a fair dice throw
noncomputable def P : Set Ω → ℝ := sorry

-- Theorem statement
theorem conditional_probability_B_given_A :
  P {ω | B ω ∧ A ω} / P {ω | A ω} = 2 / 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l224_22445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_transitive_property_l224_22405

/-- Represents a tournament between a finite number of people -/
structure Tournament where
  people : Finset Nat
  beats : Nat → Nat → Bool
  beat_antisym : ∀ x y, x ∈ people → y ∈ people → x ≠ y → beats x y ≠ beats y x
  beat_total : ∀ x y, x ∈ people → y ∈ people → x ≠ y → beats x y ∨ beats y x

/-- The main theorem: there exists a person X who either beats Y directly or through an intermediary -/
theorem tournament_transitive_property (t : Tournament) :
  ∃ x ∈ t.people, ∀ y ∈ t.people, x ≠ y →
    t.beats x y ∨ ∃ z ∈ t.people, t.beats x z ∧ t.beats z y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_transitive_property_l224_22405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_proof_l224_22403

theorem base_number_proof (n : ℝ) (x : ℝ) (b : ℝ) : 
  n = x^(3/20) → 
  n^b = 8 → 
  b = 20 →
  x = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_proof_l224_22403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l224_22418

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧ 
  Real.sin A ^ 2 - Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sin B * Real.sin C ∧ 
  c = 3 → 
  (A = 2 * π / 3 ∧ 
   ∃ (p : ℝ), p ≤ 3 + 2 * Real.sqrt 3 ∧ 
     p = a + b + c ∧
     ∀ (a' b' c' : ℝ), 
       0 < a' ∧ 0 < b' ∧ 0 < c' ∧
       a' + b' + c' > p →
       ¬(∃ (A' B' C' : ℝ), 
         0 < A' ∧ 0 < B' ∧ 0 < C' ∧
         A' + B' + C' = π ∧
         Real.sin A' ^ 2 - Real.sin B' ^ 2 - Real.sin C' ^ 2 = Real.sin B' * Real.sin C' ∧
         c' = 3 ∧
         a' / Real.sin A' = b' / Real.sin B' ∧ b' / Real.sin B' = c' / Real.sin C')) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l224_22418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l224_22460

/-- Represents a cone with its unfolded diagram as a sector -/
structure Cone where
  sector_angle : ℝ
  sector_radius : ℝ

/-- Calculate the surface area of a cone given its unfolded sector properties -/
noncomputable def surface_area (c : Cone) : ℝ :=
  let base_radius := c.sector_angle * c.sector_radius / (2 * Real.pi)
  let lateral_area := Real.pi * base_radius * c.sector_radius
  let base_area := Real.pi * base_radius ^ 2
  lateral_area + base_area

/-- Theorem: The surface area of a cone with sector angle 2π/3 and radius 2 is 16π/9 -/
theorem cone_surface_area :
  let c : Cone := { sector_angle := 2 * Real.pi / 3, sector_radius := 2 }
  surface_area c = 16 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l224_22460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l224_22420

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.log (x^2) / Real.log (1/2))

-- State the theorem
theorem f_monotonic_increasing_interval :
  {x : ℝ | ∀ y, -1 ≤ x ∧ x < y ∧ y < 0 → f x < f y} = Set.Icc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l224_22420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_times_inverse_domain_interval_of_g_l224_22426

noncomputable section

variable (m : ℝ)

def g (m : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + (m + 2) * x + m - 2
  else -(x^2) + (m + 2) * x - m + 2

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def k_times_inverse_domain_interval (f : ℝ → ℝ) (k : ℝ) (a b : ℝ) : Prop :=
  a < b ∧ ∀ x ∈ Set.Icc a b, k / b ≤ f x ∧ f x ≤ k / a

theorem eight_times_inverse_domain_interval_of_g (m : ℝ) :
  is_odd_function (g m) ∧
  (∀ x ≤ 0, g m x = x^2 + (m + 2) * x + m - 2) →
  k_times_inverse_domain_interval (g m) 8 2 (Real.sqrt 5 + 1) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_times_inverse_domain_interval_of_g_l224_22426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l224_22466

/-- 
Given a triangle ABC where:
- a, b, c are the lengths of the sides opposite to angles A, B, C respectively
- a + b = 11
- c = 7
- cos A = -1/7

Prove that:
1. a = 8
2. sin C = √3/2
3. Area of triangle ABC = 6√3
-/
theorem triangle_properties (a b c : ℝ) (A B C : Real) :
  a + b = 11 →
  c = 7 →
  Real.cos A = -1/7 →
  a = 8 ∧ 
  Real.sin C = Real.sqrt 3 / 2 ∧ 
  (1/2 * a * b * Real.sin C) = 6 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l224_22466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_quadrilateral_probability_l224_22401

/-- Given 8 points on a circle, the probability that 4 randomly selected chords 
    form a convex quadrilateral is 2/585 -/
theorem chord_quadrilateral_probability : 
  let n : ℕ := 8  -- number of points on the circle
  let k : ℕ := 4  -- number of chords to select
  let total_chords : ℕ := Nat.choose n 2  -- total number of possible chords
  let convex_quads : ℕ := Nat.choose n k  -- number of ways to choose 4 points (convex quads)
  (convex_quads : ℚ) / (Nat.choose total_chords k) = 2 / 585 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_quadrilateral_probability_l224_22401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l224_22496

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := 1 / (3 * x + 2)

/-- The point of tangency -/
def x₀ : ℝ := 2

/-- The slope of the tangent line -/
noncomputable def m : ℝ := -3 / 64

/-- The y-intercept of the tangent line -/
noncomputable def b : ℝ := 7 / 32

/-- Theorem: The equation of the tangent line to the curve y = 1 / (3x + 2) 
    at the point with x₀ = 2 is y = -3/64 x + 7/32 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * x + b ↔ 
  y - f x₀ = (deriv f x₀) * (x - x₀) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l224_22496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_square_range_l224_22432

theorem count_integers_in_square_range : 
  (Finset.filter (fun n : ℕ => 300 < n ^ 2 ∧ n ^ 2 < 1800) (Finset.range 43)).card = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_square_range_l224_22432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_AB_l224_22448

/-- Theorem: The slope of the line passing through points A(2,3) and B(3,5) is 2 -/
theorem slope_of_line_AB : (5 - 3) / (3 - 2) = 2 := by
  norm_num

#eval (5 - 3) / (3 - 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_AB_l224_22448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_product_l224_22488

theorem vector_angle_product (m n : ℝ) : 
  let OA : Fin 3 → ℝ := ![m, n, 0]
  let OB : Fin 3 → ℝ := ![1, 1, 1]
  (m^2 + n^2 = 1) →  -- OA is a unit vector
  (OA 0 * OB 0 + OA 1 * OB 1 + OA 2 * OB 2) / (Real.sqrt ((OA 0)^2 + (OA 1)^2 + (OA 2)^2) * Real.sqrt ((OB 0)^2 + (OB 1)^2 + (OB 2)^2)) = Real.sqrt 2 / 2 →  -- angle between OA and OB is π/4
  m * n = 1/4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_product_l224_22488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_when_k_zero_k_range_for_f_geq_one_l224_22449

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k * x^2 + 2

-- Theorem for part (1)
theorem extreme_values_when_k_zero :
  (∀ x : ℝ, f 0 x ≥ f 0 0) ∧ 
  (f 0 0 = 1) ∧
  (¬ ∃ M : ℝ, ∀ x : ℝ, f 0 x ≤ M) := by sorry

-- Theorem for part (2)
theorem k_range_for_f_geq_one :
  (∀ k : ℝ, (∀ x : ℝ, x ≥ 0 → f k x ≥ 1) ↔ k ≤ (1/2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_when_k_zero_k_range_for_f_geq_one_l224_22449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_agency_comparison_l224_22407

/-- The charge function for Agency A -/
noncomputable def charge_a (x : ℝ) : ℝ :=
  if x < 4 then 200 * x else 100 * x + 400

/-- The charge function for Agency B -/
def charge_b (x : ℝ) : ℝ := 140 * x

/-- Theorem stating the relationship between charges of Agency A and B based on number of travelers -/
theorem agency_comparison (x : ℝ) (h : x > 0) :
  (charge_a x < charge_b x ↔ x > 10) ∧
  (charge_a x = charge_b x ↔ x = 10) ∧
  (charge_a x > charge_b x ↔ x < 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_agency_comparison_l224_22407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_value_l224_22479

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def IsParallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem vector_parallel_value (a b : ℝ × ℝ) (lambda : ℝ) :
  ¬IsParallel a b →
  IsParallel (lambda * a.1 + b.1, lambda * a.2 + b.2) (a.1 + 2 * b.1, a.2 + 2 * b.2) →
  lambda = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_value_l224_22479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l224_22473

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  2 * (Real.log a / Real.log b / (a + b) + Real.log b / Real.log c / (b + c) + Real.log c / Real.log a / (c + a)) ≥ 9 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l224_22473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_is_top_leftmost_l224_22491

-- Define a structure for rectangles
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the five rectangles
def A : Rectangle := ⟨5, 1, 9, 2⟩
def B : Rectangle := ⟨2, 0, 6, 3⟩
def C : Rectangle := ⟨6, 7, 4, 1⟩
def D : Rectangle := ⟨8, 4, 3, 5⟩
def E : Rectangle := ⟨7, 3, 8, 0⟩

-- Define a list of all rectangles
def rectangles : List Rectangle := [A, B, C, D, E]

-- Function to check if a value is unique among rectangles
def isUnique (getValue : Rectangle → Int) (r : Rectangle) (rs : List Rectangle) : Bool :=
  (rs.filter (fun r' => getValue r' = getValue r)).length = 1

-- Function to determine if a rectangle should be on the left
def shouldBeLeft (r : Rectangle) (rs : List Rectangle) : Bool :=
  isUnique Rectangle.w r rs

-- Function to determine if a rectangle should be on the top
def shouldBeTop (r : Rectangle) (rs : List Rectangle) : Bool :=
  match (rs.map Rectangle.w).minimum? with
  | some min => r.w = min
  | none => false

-- Theorem stating that B should be the top leftmost rectangle
theorem B_is_top_leftmost :
  shouldBeLeft B rectangles ∧ shouldBeTop B rectangles := by
  sorry

#eval shouldBeLeft B rectangles
#eval shouldBeTop B rectangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_is_top_leftmost_l224_22491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l224_22499

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-8 * x^2 - 10 * x + 12)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | -2 ≤ x ∧ x ≤ 3/4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l224_22499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_distance_l224_22436

/-- Two distinct points on a parabola, symmetric with respect to a line -/
structure SymmetricPointsPair where
  A : ℝ × ℝ
  B : ℝ × ℝ
  distinct : A ≠ B
  on_parabola : (λ (x, y) => y = -x^2 + 3) A ∧ (λ (x, y) => y = -x^2 + 3) B
  symmetric : (A.1 + A.2 = 0 ∧ B.1 + B.2 = 0) ∨ 
              ((A.1 + B.1) / 2 + (A.2 + B.2) / 2 = 0)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The distance between symmetric points on the parabola is 3√2 -/
theorem symmetric_points_distance (pair : SymmetricPointsPair) :
  distance pair.A pair.B = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_distance_l224_22436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_track_distance_is_380pi_l224_22498

/-- The distance traveled by the center of a cylinder along a track of semicircular arcs -/
noncomputable def cylinderTrackDistance (cylinderDiameter : ℝ) (R₁ R₂ R₃ R₄ : ℝ) : ℝ :=
  let cylinderRadius := cylinderDiameter / 2
  let R₁' := R₁ - cylinderRadius
  let R₂' := R₂ + cylinderRadius
  let R₃' := R₃ - cylinderRadius
  let R₄' := R₄ + cylinderRadius
  Real.pi * (R₁' + R₂' + R₃' + R₄')

/-- Theorem stating that the distance traveled by the center of the cylinder is 380π inches -/
theorem cylinder_track_distance_is_380pi :
  cylinderTrackDistance 6 120 90 100 70 = 380 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_track_distance_is_380pi_l224_22498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_is_cone_l224_22474

/-- A solid shape with specific properties -/
structure Solid where
  front_view : Type
  side_view : Type
  top_view : Type

/-- Predicate to check if a shape is an isosceles triangle -/
def is_isosceles_triangle (shape : Type) : Prop := sorry

/-- Predicate to check if two shapes are congruent -/
def are_congruent (s1 s2 : Type) : Prop := sorry

/-- Predicate to check if a shape is a circle -/
def is_circle (shape : Type) : Prop := sorry

/-- Predicate to check if a circle has a center -/
def has_center (shape : Type) : Prop := sorry

/-- Predicate to check if a solid is a cone -/
def is_cone (s : Solid) : Prop := sorry

/-- Theorem stating that a solid with specific properties is a cone -/
theorem solid_is_cone (s : Solid)
  (h1 : is_isosceles_triangle s.front_view)
  (h2 : is_isosceles_triangle s.side_view)
  (h3 : are_congruent s.front_view s.side_view)
  (h4 : is_circle s.top_view)
  (h5 : has_center s.top_view) :
  is_cone s := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_is_cone_l224_22474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_head_start_distance_l224_22416

/-- Calculates the head start distance in a race between two runners --/
theorem head_start_distance 
  (cristina_speed : ℝ) 
  (nicky_speed : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : cristina_speed = 6) 
  (h2 : nicky_speed = 3) 
  (h3 : catch_up_time = 12) : 
  cristina_speed * catch_up_time - nicky_speed * catch_up_time = 36 := by
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Simplify the expression
  ring
  -- The proof is complete
  done

#check head_start_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_head_start_distance_l224_22416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_half_times_one_plus_sin_l224_22482

theorem max_cos_half_times_one_plus_sin :
  ∃ (max : ℝ), max = Real.sqrt 2 ∧ 
    ∀ θ : ℝ, 0 < θ ∧ θ < Real.pi → Real.cos (θ / 2) * (1 + Real.sin θ) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_half_times_one_plus_sin_l224_22482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_on_line_l224_22434

theorem sin_double_angle_on_line (α : ℝ) :
  (∃ (x y : ℝ), y = -2 * x ∧ x * Real.cos α = y * Real.sin α) →
  Real.sin (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_on_line_l224_22434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_sine_inequality_l224_22447

theorem negation_of_cosine_sine_inequality :
  (¬ ∀ x : ℝ, Real.cos x > Real.sin x - 1) ↔ (∃ x : ℝ, Real.cos x ≤ Real.sin x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_sine_inequality_l224_22447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_properties_l224_22452

/-- An ellipse with eccentricity 1/2 and a tangent line -/
structure EllipseWithTangent where
  a : ℝ
  b : ℝ
  h₁ : a > b ∧ b > 0
  h₂ : ((a^2 - b^2) / a^2)^(1/2 : ℝ) = 1/2

/-- The theorem stating properties of the ellipse and its tangent line -/
theorem ellipse_tangent_properties (E : EllipseWithTangent) :
  (E.a = 2 ∧ E.b = Real.sqrt 3) ∧
  (∃ (x y : ℝ), x = 1 ∧ y = 3/2 ∧ x + 2*y = 4 ∧ x^2/4 + y^2/3 = 1) ∧
  (∃ (t : ℝ), t = Real.sqrt 6 ∧ 
    (∀ (x y : ℝ), y = 3/2*x + t ∨ y = 3/2*x - t → 
      ∀ (s : ℝ), (∃ (x₁ y₁ x₂ y₂ : ℝ), 
        x₁^2/4 + y₁^2/3 = 1 ∧ 
        x₂^2/4 + y₂^2/3 = 1 ∧ 
        y₁ = 3/2*x₁ + s ∧ 
        y₂ = 3/2*x₂ + s ∧ 
        x₁ ≠ x₂ → 
        abs (x₁*y₂ - x₂*y₁) ≤ abs (x*Real.sqrt 3 - y*2)))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_properties_l224_22452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_or_power_of_two_l224_22464

theorem prime_or_power_of_two (n : ℕ) (h_n : n > 6)
  (a : ℕ → ℕ) (k : ℕ) (h_k : k > 0)
  (h_a : ∀ i, i > 0 → i ≤ k → a i < n ∧ Nat.Coprime (a i) n)
  (h_diff : ∀ i, i > 0 → i < k → a (i + 1) - a i = a 2 - a 1)
  (h_pos_diff : a 2 > a 1) :
  Nat.Prime n ∨ ∃ m : ℕ, n = 2^m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_or_power_of_two_l224_22464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_wins_probability_l224_22444

theorem tom_wins_probability (tom_hit_prob geri_hit_prob : ℚ) : 
  tom_hit_prob = 4/5 →
  geri_hit_prob = 2/3 →
  (tom_hit_prob * (1 - geri_hit_prob) + 
   (tom_hit_prob * geri_hit_prob + (1 - tom_hit_prob) * (1 - geri_hit_prob)) * 
   (tom_hit_prob * (1 - geri_hit_prob) / 
    (1 - (tom_hit_prob * geri_hit_prob + (1 - tom_hit_prob) * (1 - geri_hit_prob))))) = 2/3 :=
by
  intros h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_wins_probability_l224_22444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l224_22406

-- Define the plane
variable (P : ℝ × ℝ)

-- Define point F
def F : ℝ × ℝ := (1, 0)

-- Define the line x = -1
def line_neg_one (y : ℝ) : ℝ × ℝ := (-1, y)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the trajectory C
def on_trajectory (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define perpendicular lines through F
def perpendicular_lines (l1 l2 : ℝ → ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (∀ x, l1 x = (x, k*(x-1))) ∧ (∀ x, l2 x = (x, -1/k*(x-1)))

-- Define intersection points
def intersect_trajectory (l : ℝ → ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  on_trajectory A ∧ on_trajectory B ∧ ∃ t1 t2, l t1 = A ∧ l t2 = B

-- Theorem statement
theorem min_dot_product :
  ∀ (A B D E : ℝ × ℝ) (l1 l2 : ℝ → ℝ × ℝ),
    (∀ p : ℝ × ℝ, on_trajectory p → distance p F = distance p (line_neg_one p.2)) →
    perpendicular_lines l1 l2 →
    intersect_trajectory l1 A B →
    intersect_trajectory l2 D E →
    (∀ A' B' D' E' : ℝ × ℝ,
      intersect_trajectory l1 A' B' →
      intersect_trajectory l2 D' E' →
      ((A'.1 - D'.1) * (E'.1 - B'.1) + (A'.2 - D'.2) * (E'.2 - B'.2)) ≥ 16) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l224_22406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l224_22408

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l224_22408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_product_sum_theorem_l224_22486

def subset_product_sum (n : ℕ) : ℕ :=
  sorry

theorem subset_product_sum_theorem (n : ℕ) :
  subset_product_sum n = (n + 1).factorial - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_product_sum_theorem_l224_22486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l224_22493

theorem triangle_proof (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  Real.cos ((B + C) / 2) = 1 - Real.cos A ∧
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 ∧
  b + c = 7 ∧
  b > c →
  A = π / 3 ∧ a = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l224_22493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l224_22443

/-- Given a quadratic function f(x) = -x^2 + cx + d where f(x) ≤ 0 has the solution [-4, 6],
    the vertex of the parabola y = f(x) is (5, -23). -/
theorem parabola_vertex (c d : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = -x^2 + c*x + d)
    (h2 : Set.Icc (-4) 6 = {x | f x ≤ 0}) : 
    ∃ (v : ℝ × ℝ), v = (5, -23) ∧ IsLocalMax f v.1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l224_22443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_of_unity_with_real_sixth_power_l224_22477

theorem complex_roots_of_unity_with_real_sixth_power :
  (∃! (S : Finset ℂ), 
    (∀ z ∈ S, z^24 = 1 ∧ (z^6 : ℂ).re = (z^6 : ℂ).im) ∧ 
    (∀ z : ℂ, z^24 = 1 ∧ (z^6 : ℂ).re = (z^6 : ℂ).im → z ∈ S) ∧
    S.card = 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_of_unity_with_real_sixth_power_l224_22477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l224_22461

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.log 0.2 / Real.log 0.5)
  (hb : b = Real.log 0.2 / Real.log 2)
  (hc : c = (2 : ℝ)^(0.2 : ℝ)) : 
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l224_22461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l224_22427

-- Define the ellipse
structure Ellipse where
  focus : ℝ × ℝ
  eccentricity : ℝ

-- Define our specific ellipse
noncomputable def our_ellipse : Ellipse :=
  { focus := (0, 1),
    eccentricity := 1/2 }

-- Define the standard form of an ellipse equation
def standard_form (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem ellipse_equation (e : Ellipse) (x y : ℝ) :
  e = our_ellipse →
  standard_form 4 3 x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l224_22427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_existence_l224_22453

/-- A polygon on a grid --/
structure GridPolygon where
  sides : ℕ
  area : ℕ
  vertices : List (ℕ × ℕ)

/-- Predicate to check if a polygon is valid on the grid --/
def is_valid_grid_polygon (p : GridPolygon) : Prop :=
  p.vertices.length = p.sides ∧
  ∀ i j, i < p.vertices.length → j < p.vertices.length →
    (p.vertices.get ⟨i, by sorry⟩).1 = (p.vertices.get ⟨j, by sorry⟩).1 ∨
    (p.vertices.get ⟨i, by sorry⟩).2 = (p.vertices.get ⟨j, by sorry⟩).2

/-- The main theorem to prove --/
theorem polygon_existence :
  (∃ p : GridPolygon, p.sides = 20 ∧ p.area = 9 ∧ is_valid_grid_polygon p) ∧
  (∃ p : GridPolygon, p.sides = 100 ∧ p.area = 49 ∧ is_valid_grid_polygon p) :=
by sorry

#check polygon_existence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_existence_l224_22453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_25_5_l224_22495

/-- The area of a pentagon given its vertices -/
noncomputable def pentagonArea (v1 v2 v3 v4 v5 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  let (x5, y5) := v5
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y5 + x5*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x5 + y5*x1))

/-- The theorem stating that the area of the given pentagon is 25.5 -/
theorem pentagon_area_is_25_5 :
  pentagonArea (4, 1) (2, 6) (6, 7) (9, 5) (7, 2) = 25.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval pentagonArea (4, 1) (2, 6) (6, 7) (9, 5) (7, 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_25_5_l224_22495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_M_l224_22414

/-- The equation of a circle with radius sqrt(5) centered at the origin -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- The point M on the circle -/
def point_M : ℝ × ℝ := (-1, 2)

/-- The equation of the tangent line -/
def tangent_line_equation (x y : ℝ) : Prop := x - 2*y + 5 = 0

/-- Theorem: The tangent line to the circle x^2 + y^2 = 5 at the point (-1, 2) has the equation x - 2y + 5 = 0 -/
theorem tangent_line_at_M :
  circle_equation point_M.1 point_M.2 →
  ∀ x y : ℝ, (tangent_line_equation x y ↔ (x - point_M.1) * point_M.1 + (y - point_M.2) * point_M.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_M_l224_22414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_proof_l224_22422

-- Define the commute distance in miles
noncomputable def commute_distance : ℝ := 40

-- Define the actual speed in mph
noncomputable def actual_speed : ℝ := 60

-- Define the slower speed in mph
noncomputable def slower_speed : ℝ := 55

-- Function to calculate travel time in hours
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- Function to convert hours to minutes
noncomputable def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

-- Theorem to prove the time difference
theorem time_difference_proof :
  let actual_time := travel_time commute_distance actual_speed
  let slower_time := travel_time commute_distance slower_speed
  let time_diff := hours_to_minutes (slower_time - actual_time)
  abs (time_diff - 3.64) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_proof_l224_22422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_cartesian_equation_intersection_distance_product_l224_22439

noncomputable def curve_C (θ : ℝ) : ℝ := 4 * (Real.cos θ + Real.sin θ)

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t / 2, 1 + (Real.sqrt 3 / 2) * t)

def point_E : ℝ × ℝ := (0, 1)

theorem curve_C_cartesian_equation :
  ∀ x y : ℝ, (x - 2)^2 + (y - 2)^2 = 8 ↔ 
  ∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ :=
by sorry

theorem intersection_distance_product :
  ∃ A B : ℝ × ℝ,
  (∃ t₁ t₂ : ℝ, line_l t₁ = A ∧ line_l t₂ = B) ∧
  (∃ θ₁ θ₂ : ℝ, 
    A.1 = curve_C θ₁ * Real.cos θ₁ ∧ A.2 = curve_C θ₁ * Real.sin θ₁ ∧
    B.1 = curve_C θ₂ * Real.cos θ₂ ∧ B.2 = curve_C θ₂ * Real.sin θ₂) ∧
  (A.1 - point_E.1)^2 + (A.2 - point_E.2)^2 *
  (B.1 - point_E.1)^2 + (B.2 - point_E.2)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_cartesian_equation_intersection_distance_product_l224_22439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_hexagon_side_ratio_l224_22478

/-- Given a regular pentagon and a regular hexagon with equal perimeters,
    the ratio of their side lengths is 6/5. -/
theorem pentagon_hexagon_side_ratio (perimeter : ℝ) (perimeter_pos : 0 < perimeter) :
  (perimeter / 5) / (perimeter / 6) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_hexagon_side_ratio_l224_22478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_elements_theorem_l224_22409

/-- A symmetric n × n matrix with rows containing 1 to n has diagonal elements 1 to n when n is odd -/
theorem diagonal_elements_theorem (n : ℕ) (hn : Odd n) 
  (A : Matrix (Fin n) (Fin n) ℕ)
  (h_sym : ∀ i j, A i j = A j i)
  (h_rows : ∀ i, Multiset.ofList (List.map (A i) (List.finRange n)) = Multiset.ofList (List.range n)) :
  Multiset.ofList (List.map (λ i ↦ A i i) (List.finRange n)) = Multiset.ofList (List.range n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_elements_theorem_l224_22409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_a_encoded_as_k_l224_22489

/-- Calculates the sum of natural numbers from 1 to n -/
def sumToN (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents a letter in the alphabet (A-Z) -/
inductive Letter : Type where
| A | B | C | D | E | F | G | H | I | J | K | L | M
| N | O | P | Q | R | S | T | U | V | W | X | Y | Z

/-- Shifts a letter by a given amount (with wrapping) -/
def shiftLetter (l : Letter) (shift : ℕ) : Letter :=
  match l with
  | Letter.A => sorry
  | Letter.B => sorry
  | Letter.C => sorry
  | Letter.D => sorry
  | Letter.E => sorry
  | Letter.F => sorry
  | Letter.G => sorry
  | Letter.H => sorry
  | Letter.I => sorry
  | Letter.J => sorry
  | Letter.K => sorry
  | Letter.L => sorry
  | Letter.M => sorry
  | Letter.N => sorry
  | Letter.O => sorry
  | Letter.P => sorry
  | Letter.Q => sorry
  | Letter.R => sorry
  | Letter.S => sorry
  | Letter.T => sorry
  | Letter.U => sorry
  | Letter.V => sorry
  | Letter.W => sorry
  | Letter.X => sorry
  | Letter.Y => sorry
  | Letter.Z => sorry

/-- Counts occurrences of a letter in a string -/
def countOccurrences (s : String) (c : Char) : ℕ := sorry

/-- The text to be encrypted -/
def text : String := "Cassandra dares radar advancement"

theorem last_a_encoded_as_k :
  let aCount := countOccurrences text 'a'
  let shift := sumToN aCount % 26
  shiftLetter Letter.A shift = Letter.K := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_a_encoded_as_k_l224_22489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l224_22430

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

/-- The left focus of the hyperbola -/
noncomputable def F : ℝ × ℝ := sorry

/-- Point A -/
def A : ℝ × ℝ := (1, 4)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem min_distance_sum :
  ∃ (min : ℝ), min = 9 ∧
    ∀ P : ℝ × ℝ, hyperbola P.1 P.2 → P.1 > F.1 →
      distance P F + distance P A ≥ min := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l224_22430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_from_country_l224_22440

def total_cards : ℝ := 403.0
def cards_from_home : ℝ := 287.0

theorem cards_from_country : total_cards - cards_from_home = 116.0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_from_country_l224_22440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_total_is_2711_l224_22483

/-- Represents a field with a number of rows and items per row -/
structure MyField where
  rows : ℕ
  itemsPerRow : ℕ

/-- Calculates the total number of items in a field -/
def fieldTotal (f : MyField) : ℕ := f.rows * f.itemsPerRow

/-- Represents the farm with its corn, pea, and carrot fields -/
structure Farm where
  cornFields : List MyField
  peaFields : List MyField
  carrotFields : List MyField

/-- Calculates the total number of items across all fields of a given type -/
def cropTotal (fields : List MyField) : ℕ := (fields.map fieldTotal).sum

/-- The specific farm described in the problem -/
def exampleFarm : Farm := {
  cornFields := [
    { rows := 13, itemsPerRow := 8 },
    { rows := 16, itemsPerRow := 12 },
    { rows := 9, itemsPerRow := 10 },
    { rows := 20, itemsPerRow := 6 }
  ],
  peaFields := [
    { rows := 12, itemsPerRow := 20 },
    { rows := 15, itemsPerRow := 18 },
    { rows := 7, itemsPerRow := 25 }
  ],
  carrotFields := [
    { rows := 10, itemsPerRow := 30 },
    { rows := 8, itemsPerRow := 25 },
    { rows := 15, itemsPerRow := 20 },
    { rows := 12, itemsPerRow := 35 },
    { rows := 20, itemsPerRow := 15 }
  ]
}

/-- Calculates the total number of items grown on the farm -/
def farmTotal (f : Farm) : ℕ :=
  cropTotal f.cornFields + cropTotal f.peaFields + cropTotal f.carrotFields

theorem farm_total_is_2711 : farmTotal exampleFarm = 2711 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_total_is_2711_l224_22483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_prime_path_length_correct_l224_22492

/-- Correspondence rule f that maps a point (m,n) to (√m, √n) -/
noncomputable def f (m n : ℝ) : ℝ × ℝ := (Real.sqrt m, Real.sqrt n)

/-- The line segment AB -/
def line_segment (t : ℝ) : ℝ × ℝ := (2 + 4*t, 6 - 4*t)

theorem locus_of_M_prime (t : ℝ) (h : 0 ≤ t ∧ t ≤ 1) :
  ∃ (θ : ℝ), f (line_segment t).1 (line_segment t).2 = (Real.sqrt 8 * Real.cos θ, Real.sqrt 8 * Real.sin θ) := by
  sorry

/-- The length of the path that M' travels -/
noncomputable def path_length : ℝ := 
  (Real.sqrt 8) * (Real.arctan (Real.sqrt 3) - Real.arctan (1 / Real.sqrt 3))

theorem path_length_correct : 
  path_length = (Real.sqrt 8) * (Real.arctan (Real.sqrt 3) - Real.arctan (1 / Real.sqrt 3)) := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_prime_path_length_correct_l224_22492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_balanceable_x_squared_and_exp_same_balance_pair_cos_squared_balance_pairs_range_l224_22437

-- Definition of "balanceable" function
def balanceable (f : ℝ → ℝ) : Prop :=
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x, m * f x = f (x + k) + f (x - k)

-- Definition of "balance" pair
def balance_pair (f : ℝ → ℝ) (m k : ℝ) : Prop :=
  m ≠ 0 ∧ ∀ x, m * f x = f (x + k) + f (x - k)

theorem sin_balanceable : ∃ k : ℝ, ∀ x : ℝ, Real.sin x = Real.sin (x + k) + Real.sin (x - k) := by sorry

theorem x_squared_and_exp_same_balance_pair :
  ∀ a : ℝ, a ≠ 0 → (∀ x : ℝ, 2 * x^2 = (x + 0)^2 + (x - 0)^2) ∧
  (∀ x : ℝ, 2 * (a + Real.exp (x * Real.log 2)) = (a + Real.exp ((x + 0) * Real.log 2)) + (a + Real.exp ((x - 0) * Real.log 2))) := by sorry

theorem cos_squared_balance_pairs_range :
  ∀ x : ℝ, 0 < x → x ≤ π/4 → 1 ≤ 4 * (Real.tan x)^4 + 1 / (Real.cos x)^4 ∧
  4 * (Real.tan x)^4 + 1 / (Real.cos x)^4 ≤ 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_balanceable_x_squared_and_exp_same_balance_pair_cos_squared_balance_pairs_range_l224_22437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_weights_l224_22411

/-- Represents the properties of a metal cube -/
structure MetalCube where
  side_length : ℝ
  weight : ℝ

/-- The density of the metal -/
noncomputable def metal_density (cube : MetalCube) : ℝ :=
  cube.weight / (cube.side_length ^ 3)

/-- Theorem about the weights of different sized cubes made of the same metal -/
theorem cube_weights (original : MetalCube)
    (h_original_weight : original.weight = 8)
    (second_cube : MetalCube)
    (h_second_side : second_cube.side_length = 4 * original.side_length)
    (third_cube : MetalCube)
    (h_third_height : third_cube.side_length = 2 * original.side_length)
    (h_third_base : third_cube.weight = metal_density original * (original.side_length ^ 2) * (2 * original.side_length)) :
  second_cube.weight = 512 ∧ third_cube.weight = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_weights_l224_22411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_bound_l224_22494

theorem min_lambda_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∃ lambda_min : ℝ, lambda_min = 1 ∧ 
  (∀ lambda : ℝ, (a / (a^2 + 1) + b / (b^2 + 1) ≤ lambda) ↔ lambda ≥ lambda_min) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_bound_l224_22494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_symmetry_and_monotonicity_l224_22421

-- Define the power function
noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x ^ (m^2 - 2*m - 3)

-- State the theorem
theorem power_function_symmetry_and_monotonicity (m : ℤ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- Symmetry about y-axis
  (∀ x y : ℝ, 0 < x ∧ x < y → f m y < f m x) →  -- Monotonically decreasing in first quadrant
  m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_symmetry_and_monotonicity_l224_22421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_k_equals_4_l224_22435

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the line of symmetry
def line_of_symmetry (k : ℝ) (x y : ℝ) : Prop := k * x - y - 2 = 0

-- Define symmetry with respect to a line
def symmetric_points (P Q : ℝ × ℝ) (k : ℝ) : Prop :=
  line_of_symmetry k ((P.1 + Q.1) / 2) ((P.2 + Q.2) / 2)

-- Theorem statement
theorem symmetry_implies_k_equals_4 (P Q : ℝ × ℝ) (k : ℝ)
  (h_distinct : P ≠ Q)
  (h_on_circle_P : my_circle P.1 P.2)
  (h_on_circle_Q : my_circle Q.1 Q.2)
  (h_symmetric : symmetric_points P Q k) :
  k = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_k_equals_4_l224_22435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_bakes_cherry_pies_three_days_l224_22400

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of pies Steve bakes per day -/
def pies_per_day : ℕ := 12

/-- The number of days Steve bakes apple pies in a week -/
def apple_days : ℕ := 4

/-- The number of days Steve bakes cherry pies in a week -/
def cherry_days : ℕ := 3

/-- Steve bakes either apple or cherry pies each day of the week -/
axiom total_days : apple_days + cherry_days = days_in_week

/-- Steve bakes 12 more apple pies than cherry pies in a week -/
axiom pie_difference : apple_days * pies_per_day = cherry_days * pies_per_day + pies_per_day

theorem steve_bakes_cherry_pies_three_days :
  cherry_days = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_bakes_cherry_pies_three_days_l224_22400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_circle_l224_22468

-- Define the parabola and circle
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1
def circle' (q : ℝ × ℝ) : Prop := (q.1 - 3)^2 + q.2^2 = 1

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem min_distance_parabola_circle :
  ∃ (min_dist : ℝ),
    (∀ (p q : ℝ × ℝ), parabola p → circle' q → distance p q ≥ min_dist) ∧
    (∃ (p q : ℝ × ℝ), parabola p ∧ circle' q ∧ distance p q = min_dist) ∧
    (min_dist = Real.sqrt 11 / 2 - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_circle_l224_22468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l224_22410

noncomputable def f (x : ℝ) := Real.sin (Real.pi * x + 1/3)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l224_22410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sliced_face_area_of_specific_cylinder_l224_22480

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  height : ℝ
  radius : ℝ

/-- Represents two points on the circumference of the top face of a cylinder -/
structure CircumferencePoints where
  arcMeasure : ℝ  -- in degrees

/-- Calculates the area of the face created by slicing a cylinder through two points on its top circumference and its axis -/
noncomputable def slicedFaceArea (c : RightCircularCylinder) (p : CircumferencePoints) : ℝ :=
  (p.arcMeasure / 360) * Real.pi * c.radius^2

theorem sliced_face_area_of_specific_cylinder :
  let c : RightCircularCylinder := ⟨8, 5⟩
  let p : CircumferencePoints := ⟨90⟩
  slicedFaceArea c p = 25 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sliced_face_area_of_specific_cylinder_l224_22480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt3_irrational_l224_22471

-- Define the numbers
noncomputable def sqrt3 : ℝ := Real.sqrt 3
noncomputable def sqrt4 : ℝ := Real.sqrt 4
noncomputable def cbrt8 : ℝ := Real.rpow 8 (1/3)
def pi_approx : ℝ := 3.14

-- State the theorem
theorem sqrt3_irrational :
  Irrational sqrt3 ∧ ¬Irrational sqrt4 ∧ ¬Irrational cbrt8 ∧ ¬Irrational pi_approx := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt3_irrational_l224_22471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sos_properties_l224_22433

-- Define the sos function
noncomputable def sos (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- State the theorem
theorem sos_properties :
  (∀ y ∈ Set.range sos, -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2) ∧
  (∀ x : ℝ, sos (x + 2 * Real.pi) = sos x) ∧
  (∀ p : ℝ, p > 0 → (∀ x : ℝ, sos (x + p) = sos x) → p ≥ 2 * Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sos_properties_l224_22433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_transformed_curve_is_unit_circle_l224_22484

/-- Transformation function φ -/
noncomputable def φ (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 / 3, p.2 / 2)

/-- Original curve -/
def original_curve (p : ℝ × ℝ) : Prop :=
  p.1^2 / 9 + p.2^2 / 4 = 1

/-- Transformed curve -/
def transformed_curve (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 = 1

theorem transformation_result :
  ∀ p : ℝ × ℝ, original_curve p ↔ transformed_curve (φ p) := by
  sorry

theorem transformed_curve_is_unit_circle :
  ∀ p : ℝ × ℝ, transformed_curve p ↔ Metric.ball (0 : ℝ × ℝ) 1 p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_transformed_curve_is_unit_circle_l224_22484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_and_minimum_a_l224_22451

noncomputable def f (x : ℝ) := Real.log x - x^2 + x

theorem f_monotone_decreasing_and_minimum_a (a : ℕ) :
  (∀ x : ℝ, x ≥ 1 → (deriv f) x < 0) ∧
  (∀ x : ℝ, x > 0 → f x ≤ (a / 2 - 1) * x^2 + a * x - 1) ↔ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_and_minimum_a_l224_22451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equivalent_to_cos_2x_l224_22497

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.sin x ^ 2) - Real.sqrt (Real.cos x ^ 4 + 4 * Real.cos x ^ 2)

theorem g_equivalent_to_cos_2x : ∀ x : ℝ, g x = Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equivalent_to_cos_2x_l224_22497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_intersection_of_roots_l224_22476

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 10*x^2 + 29*x - 25

-- Define the sequence S(x)
def S (x : ℝ) : Set ℕ := {n : ℕ | ∃ m : ℕ, n = ⌊m * x⌋}

theorem infinite_intersection_of_roots : 
  ∃ (α β : ℝ), α ≠ β ∧ f α = 0 ∧ f β = 0 ∧ 
  Set.Infinite (S α ∩ S β) ∧ (∀ n ∈ S α ∩ S β, n > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_intersection_of_roots_l224_22476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_cosine_csc_l224_22425

open Real BigOperators

theorem sum_cosine_csc : 
  ∑ x in Finset.range 41, 2 * cos (x + 3 : ℝ) * cos 1 * (1 + 1 / sin (x + 2 : ℝ) * 1 / sin (x + 4 : ℝ)) = 2 * cos 2 + 2 / sin 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_cosine_csc_l224_22425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_properties_l224_22472

/-- Arithmetic sequence a_n -/
noncomputable def a (n : ℕ) : ℝ := 3 * n - 2

/-- Geometric sequence b_n -/
noncomputable def b (n : ℕ) : ℝ := 2^(n - 1)

/-- Sequence c_n defined in terms of a_n -/
noncomputable def c (n : ℕ) : ℝ := 3 / (a n * a (n + 1))

/-- Sum of the first n terms of c_n -/
noncomputable def S (n : ℕ) : ℝ := 3 * n / (3 * n + 1)

theorem sequences_properties :
  (∀ n : ℕ, n ≥ 2 → a n = 3 * n - 2) ∧
  (a 2 = 4) ∧
  (2 * a 4 - a 5 = 7) ∧
  (∀ n : ℕ, n ≥ 1 → b n = 2^(n - 1)) ∧
  (b 3 = 4) ∧
  (b 4 + b 5 = 8 * (b 1 + b 2)) ∧
  (∀ n : ℕ, n ≥ 1 → c n = 3 / (a n * a (n + 1))) ∧
  (∀ n : ℕ, n ≥ 1 → S n = 3 * n / (3 * n + 1)) :=
by sorry

#check sequences_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_properties_l224_22472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_scalar_l224_22415

/-- Given a vector and two points, calculate the scalar of the projection vector. -/
theorem projection_scalar (a : ℝ × ℝ) (A B : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let scalar := (AB.1 * a.1 + AB.2 * a.2) / (a.1^2 + a.2^2)
  a = (-4, 3) → A = (1, 1) → B = (2, -1) → scalar = -2/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_scalar_l224_22415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_negative_correct_point_slope_form_l224_22467

-- Define a line l: ax + by - 2 = 0
def line_equation (a b : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y - 2 = 0

-- Define the slope of a line ax + by + c = 0
noncomputable def line_slope (a b : ℝ) : ℝ := -a / b

-- Statement 1
theorem slope_negative (a b : ℝ) (h : a * b > 0) :
  line_slope a b < 0 := by sorry

-- Define a point
def point := ℝ × ℝ

-- Define the point-slope form of a line
def point_slope_form (p : point) (m : ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

-- Statement 2
theorem correct_point_slope_form (x y : ℝ) :
  point_slope_form (2, -1) (-Real.sqrt 3) x y ↔ y + 1 = -Real.sqrt 3 * (x - 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_negative_correct_point_slope_form_l224_22467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l224_22456

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem tangent_line_at_one (x y : ℝ) :
  (y - f 1) = (deriv f 1) * (x - 1) ↔ x - y - 1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l224_22456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_inv_sequence_l224_22428

/-- Piecewise function f -/
noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then 2*x - 2 else Real.sqrt (2*x)

/-- Inverse function of f -/
noncomputable def f_inv (x : ℝ) : ℝ :=
  if x < 4 then (x + 2) / 2 else x^2 / 2

/-- f_inv is the inverse of f -/
axiom f_inv_is_inverse : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f

/-- Theorem stating the sum of f_inv applied to the sequence -4, -2, 0, 2, 4, 6 equals 8 -/
theorem sum_of_f_inv_sequence : 
  f_inv (-4) + f_inv (-2) + f_inv 0 + f_inv 2 + f_inv 4 + f_inv 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_inv_sequence_l224_22428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_Z_l224_22475

noncomputable def i : ℂ := Complex.I

noncomputable def Z : ℂ := (1 : ℂ) / (1 + i) + i^3

theorem magnitude_of_Z : Complex.abs Z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_Z_l224_22475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_min_F_equals_one_l224_22441

-- Define the real numbers x and y
variable (x y : ℝ)

-- Define m and n with their constraint
variable (m n : ℝ)
axiom mn_sum : m + n = 7

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 1|

-- Define the max function
noncomputable def max' (a b : ℝ) : ℝ := if a ≥ b then a else b

-- Statement 1: The solution set of f(x) ≥ 7x is {x | x ≤ 0}
theorem f_inequality_solution_set :
  {x : ℝ | f x ≥ 7 * x} = {x : ℝ | x ≤ 0} := by
  sorry

-- Statement 2: For F = max{|x² - 4y + m|, |y² - 2x + n|}, min F = 1
theorem min_F_equals_one :
  ∃ (x y : ℝ), ∀ (x' y' : ℝ),
    max' (|x'^2 - 4*y' + m|) (|y'^2 - 2*x' + n|) ≥
    max' (|x^2 - 4*y + m|) (|y^2 - 2*x + n|) ∧
    max' (|x^2 - 4*y + m|) (|y^2 - 2*x + n|) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_min_F_equals_one_l224_22441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sales_profit_loss_l224_22470

theorem car_sales_profit_loss (selling_price : ℝ) (gain_percent : ℝ) (loss_percent : ℝ) :
  selling_price = 404415 ∧ 
  gain_percent = 15 ∧ 
  loss_percent = 15 →
  let cp1 := selling_price / (1 + gain_percent / 100)
  let cp2 := selling_price / (1 - loss_percent / 100)
  let total_cost := cp1 + cp2
  let total_selling := 2 * selling_price
  let profit_loss_percent := (total_selling - total_cost) / total_cost * 100
  abs (profit_loss_percent + 2.25) < 0.01 := by
  sorry

#eval (808830 - 827447.35) / 827447.35 * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sales_profit_loss_l224_22470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowler_last_match_runs_l224_22485

/-- Represents a bowler's statistics --/
structure BowlerStats where
  initialAverage : ℚ
  wicketsLastMatch : ℕ
  averageDecrease : ℚ
  approximateWicketsBefore : ℕ

/-- Calculates the runs given in the last match --/
def runsLastMatch (stats : BowlerStats) : ℚ :=
  let newAverage := stats.initialAverage - stats.averageDecrease
  let totalWickets := stats.approximateWicketsBefore + stats.wicketsLastMatch
  (newAverage * totalWickets) - (stats.initialAverage * stats.approximateWicketsBefore)

/-- Theorem: The bowler gave 26 runs in the last match --/
theorem bowler_last_match_runs :
  let stats : BowlerStats := {
    initialAverage := 124/10,
    wicketsLastMatch := 5,
    averageDecrease := 4/10,
    approximateWicketsBefore := 85
  }
  runsLastMatch stats = 26 := by
  -- Proof goes here
  sorry

#eval runsLastMatch {
  initialAverage := 124/10,
  wicketsLastMatch := 5,
  averageDecrease := 4/10,
  approximateWicketsBefore := 85
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowler_last_match_runs_l224_22485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l224_22431

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := x / (x + 2)

-- Define the point of tangency
def point : ℝ × ℝ := (-1, -1)

-- Define the slope of the tangent line at the point
noncomputable def tangent_slope : ℝ := (deriv f) point.1

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := tangent_slope * (x - point.1) + point.2

-- Theorem statement
theorem tangent_line_equation :
  ∀ x, tangent_line x = 2 * x + 1 :=
by
  intro x
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l224_22431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_period_with_linear_transform_period_of_tan_x_div_3_plus_pi_div_4_l224_22465

open Real

/-- The period of the tangent function with a linear transformation of its argument -/
theorem tangent_period_with_linear_transform (a b : ℝ) (ha : a ≠ 0) :
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, Real.tan (a * x + b) = Real.tan (a * (x + p) + b) :=
sorry

/-- The period of y = tan(x/3 + π/4) is 3π -/
theorem period_of_tan_x_div_3_plus_pi_div_4 :
  ∃ p : ℝ, p = 3 * π ∧ p > 0 ∧ 
    ∀ x : ℝ, Real.tan (x / 3 + π / 4) = Real.tan ((x + p) / 3 + π / 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_period_with_linear_transform_period_of_tan_x_div_3_plus_pi_div_4_l224_22465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l224_22457

/-- Calculates the length of a train given its speed and time to cross a point. -/
noncomputable def train_length (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * (1000 / 3600)
  speed_m_s * time_seconds

/-- The length of a train with given speed and crossing time is approximately 99.992 meters. -/
theorem train_length_calculation :
  let speed := (126 : ℝ)
  let time := (2.856914303998537 : ℝ)
  abs (train_length speed time - 99.992) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l224_22457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_l224_22402

-- Define the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (7, 3)
def C : ℝ × ℝ := (4, 4)
def D : ℝ × ℝ := (2, 3)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def DC : ℝ × ℝ := (C.1 - D.1, C.2 - D.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define cosine of angle between vectors
noncomputable def cos_angle (v w : ℝ × ℝ) : ℝ := (dot_product v w) / (magnitude v * magnitude w)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = (k * w.1, k * w.2)

-- Define trapezoid
def is_trapezoid (A B C D : ℝ × ℝ) : Prop :=
  parallel (B.1 - A.1, B.2 - A.2) (C.1 - D.1, C.2 - D.2) ∧
  magnitude (C.1 - B.1, C.2 - B.2) = magnitude (D.1 - A.1, D.2 - A.2)

theorem quadrilateral_properties : 
  cos_angle AB AC = 2 * Real.sqrt 5 / 5 ∧ 
  is_trapezoid A B C D := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_l224_22402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l224_22424

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : det (B^2 - 3 • B) = 88 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l224_22424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_in_cube_l224_22487

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem: Light path length in a cube -/
theorem light_path_length_in_cube (cube_side : ℝ) (reflection_point : Point3D) : 
  cube_side = 10 →
  reflection_point = ⟨0, 6, 4⟩ →
  ∃ (p q : ℕ), 
    (distance ⟨0, 0, 0⟩ reflection_point * (10 : ℝ)) = p * Real.sqrt q ∧
    p = 10 ∧
    q = 152 := by
  sorry

#check light_path_length_in_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_in_cube_l224_22487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_proof_l224_22423

noncomputable def sample_mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def sample_variance (xs : List ℝ) : ℝ :=
  let mean := sample_mean xs
  (xs.map (fun x => (x - mean) ^ 2)).sum / xs.length

theorem sample_variance_proof (a : ℝ) :
  let xs := [a, 0, 1, 2, 3]
  sample_mean xs = 1 → sample_variance xs = 2 := by
  sorry

#check sample_variance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_proof_l224_22423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_lambda_l224_22469

/-- Given vectors a, b, and c in ℝ², prove that if c is parallel to (a + b), then λ = 0 -/
theorem parallel_vector_lambda (a b c : ℝ × ℝ) (l : ℝ) :
  a = (1, 2) →
  b = (2, -2) →
  c = (1, l) →
  ∃ (k : ℝ), c = k • (a + b) →
  l = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_lambda_l224_22469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_jam_speed_is_12_l224_22463

/-- The speed of the car in the traffic jam in km/h -/
def speed_in_jam : ℝ := 10

/-- The speed of the car outside the traffic jam in km/h -/
def speed_outside_jam : ℝ := 60

/-- The speed at which the traffic jam is extending in km/h -/
def traffic_jam_speed : ℝ → ℝ := λ v => v

/-- The theorem stating that the traffic jam speed is 12 km/h -/
theorem traffic_jam_speed_is_12 :
  ∃ v : ℝ, traffic_jam_speed v = 12 ∧
  (speed_outside_jam + v) / speed_outside_jam = v / speed_in_jam := by
  -- Proof goes here
  sorry

#check traffic_jam_speed_is_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_jam_speed_is_12_l224_22463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sine_inequality_is_false_l224_22490

theorem negation_of_sine_inequality_is_false :
  (∀ x : ℝ, x ∈ Set.Ioo 0 (π/2) → Real.sin x < 1) →
  ¬(∃ x : ℝ, x ∈ Set.Ioo 0 (π/2) ∧ Real.sin x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sine_inequality_is_false_l224_22490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_x2_greater_than_e_range_of_a_l224_22455

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log x
def h (k b x : ℝ) : ℝ := k * x + b

-- Theorem 1
theorem range_of_k (k : ℝ) :
  (∀ x > 0, f x ≥ h k 0 x ∧ h k 0 x ≥ g x) →
  k ≥ 1 / Real.exp 1 ∧ k ≤ Real.exp 1 :=
by sorry

-- Theorem 2
theorem x2_greater_than_e (x₁ x₂ k b : ℝ) :
  x₁ < 0 →
  (∃ y₁, h k b x₁ = y₁ ∧ f x₁ = y₁ ∧ k = (deriv f) x₁) →
  (∃ y₂, h k b x₂ = y₂ ∧ g x₂ = y₂ ∧ k = (deriv g) x₂) →
  x₂ > Real.exp 1 :=
by sorry

-- Theorem 3
theorem range_of_a (x₁ x₂ k b a : ℝ) :
  x₁ < 0 →
  (∃ y₁, h k b x₁ = y₁ ∧ f x₁ = y₁ ∧ k = (deriv f) x₁) →
  (∃ y₂, h k b x₂ = y₂ ∧ g x₂ = y₂ ∧ k = (deriv g) x₂) →
  (∀ x ≥ x₂, a * (x₁ - 1) + x * Real.log x - x ≥ 0) →
  a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_x2_greater_than_e_range_of_a_l224_22455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_reals_l224_22413

/-- A power function that passes through the point (2, 1/4) -/
noncomputable def f (x : ℝ) : ℝ := x^(-2 : ℝ)

/-- The function f passes through the point (2, 1/4) -/
axiom f_passes_through : f 2 = 1/4

/-- The interval (0, +∞) is monotonically decreasing for f -/
theorem f_decreasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → x < y → f y < f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_reals_l224_22413
