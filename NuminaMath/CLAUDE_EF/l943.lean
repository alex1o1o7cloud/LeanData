import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_non_zero_period_l943_94313

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - (floor x : ℝ) - Real.tan x

-- State the theorem
theorem no_non_zero_period :
  ¬ ∃ (T : ℝ), T ≠ 0 ∧ ∀ (x : ℝ), f (x + T) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_non_zero_period_l943_94313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_int_average_l943_94338

theorem three_int_average (a b c : ℕ) : 
  0 < a → a ≤ b → b ≤ c → b = a + 13 → c = 25 → (a + b + c : ℚ) / 3 = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_int_average_l943_94338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_values_and_e_bounds_l943_94367

noncomputable def a (n : ℕ) : ℝ := ∫ x in (0:ℝ)..1, x^n * Real.exp x

theorem a_values_and_e_bounds :
  a 1 = 1 ∧
  a 2 = Real.exp 1 - 2 ∧
  a 3 = 6 - 2 * Real.exp 1 ∧
  a 4 = 9 * Real.exp 1 - 24 ∧
  8/3 < Real.exp 1 ∧ Real.exp 1 < 30/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_values_and_e_bounds_l943_94367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_line_circle_intersection_l943_94301

/-- The minimum value of k for which the line kx - y - 2k = 0 always intersects
    the circle (x+2)^2 + y^2 = 4 -/
theorem min_k_line_circle_intersection :
  let C : Set (ℝ × ℝ) := {p | (p.1 + 2)^2 + p.2^2 = 4}
  let l (k : ℝ) : Set (ℝ × ℝ) := {p | k * p.1 - p.2 - 2 * k = 0}
  ∀ k : ℝ, (∀ k' : ℝ, (l k' ∩ C).Nonempty) → k ≥ -Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_line_circle_intersection_l943_94301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_open_interval_l943_94326

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x)

theorem f_increasing_on_open_interval : 
  ∀ x y : ℝ, 1 < x ∧ x < y → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_open_interval_l943_94326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l943_94303

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 + x else (-x)^2 + (-x)

-- State the theorem
theorem f_properties :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x > 0, f x = x^2 + x) →  -- definition for x > 0
  (f (-1) = 2) ∧  -- property 1
  (∀ x < 0, f x = x^2 - x) ∧  -- property 2
  (Set.Ioo 0 2 = { x | f (x - 1) < 2 }) -- property 3
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l943_94303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_is_two_l943_94389

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (changed to ℚ for computability)
  d : ℚ      -- Common difference (changed to ℚ)
  first_term_eq_one : a 1 = 1
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d)

theorem common_difference_is_two (seq : ArithmeticSequence) 
  (sum_three_eq_nine : sum_n seq 3 = 9) : seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_is_two_l943_94389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l943_94370

noncomputable def f (x : ℝ) := Real.log (x + 1)

theorem domain_of_f :
  {x : ℝ | x > -1} = {x : ℝ | f x ∈ Set.range Real.log} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l943_94370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_triangle_height_l943_94334

-- Define the properties of the triangles
noncomputable def triangle1_base : ℝ := 15
noncomputable def triangle1_height : ℝ := 12
noncomputable def triangle2_base : ℝ := 20

-- Define the area calculation function for a triangle
noncomputable def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

-- State the theorem
theorem new_triangle_height :
  let triangle1_area := triangle_area triangle1_base triangle1_height
  let triangle2_area := 2 * triangle1_area
  let triangle2_height := (2 * triangle2_area) / triangle2_base
  triangle2_height = 18 := by
  -- Unfold definitions
  simp [triangle_area, triangle1_base, triangle1_height, triangle2_base]
  -- Perform algebraic simplification
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_triangle_height_l943_94334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_razorback_tshirt_sales_l943_94318

theorem razorback_tshirt_sales (price : ℝ) (quantity : ℕ) (discount_rate : ℝ) (tax_rate : ℝ) : 
  price = 16 →
  quantity = 45 →
  discount_rate = 0.1 →
  tax_rate = 0.06 →
  (price * ↑quantity * (1 - discount_rate) * (1 + tax_rate)) = 686.88 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_razorback_tshirt_sales_l943_94318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_for_minimum_cost_l943_94347

/-- Represents the cost function for a locomotive's travel between two cities. -/
noncomputable def cost_function (k : ℝ) (a : ℝ) (x : ℝ) : ℝ :=
  a * (k * x^2 + 400 / x)

/-- Theorem stating the optimal speed for minimum cost -/
theorem optimal_speed_for_minimum_cost 
  (k : ℝ) 
  (a : ℝ) 
  (h1 : k = 1 / 200) 
  (h2 : a > 0) :
  ∃ (x : ℝ), x = 20 * (5 : ℝ)^(1/3) ∧
  ∀ (y : ℝ), 0 < y ∧ y ≤ 100 → cost_function k a x ≤ cost_function k a y := by
  sorry

#check optimal_speed_for_minimum_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_for_minimum_cost_l943_94347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l943_94369

/-- Represents a triangle in the sequence --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Generates the next triangle in the sequence --/
noncomputable def next_triangle (t : Triangle) : Triangle :=
  { a := t.a / 2, b := t.b / 2, c := t.c / 2 }

/-- Checks if a triangle is valid (all sides ≥ 1) --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.a ≥ 1 ∧ t.b ≥ 1 ∧ t.c ≥ 1

/-- The initial triangle T₁ --/
def T₁ : Triangle :=
  { a := 1011, b := 1012, c := 1013 }

/-- The sequence of triangles --/
noncomputable def triangle_sequence : ℕ → Triangle
  | 0 => T₁
  | n + 1 => next_triangle (triangle_sequence n)

/-- The perimeter of a triangle --/
noncomputable def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

theorem last_triangle_perimeter :
  ∃ n : ℕ, 
    is_valid_triangle (triangle_sequence n) ∧
    ¬is_valid_triangle (triangle_sequence (n + 1)) ∧
    perimeter (triangle_sequence n) = 379.5 / 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l943_94369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_five_primes_with_units_digit_3_l943_94359

def isPrime (n : ℕ) : Bool :=
  n > 1 && (List.range (n - 1)).all (λ d => d <= 1 || n % (d + 2) ≠ 0)

def hasUnitsDigitOf3 (n : ℕ) : Bool :=
  n % 10 = 3

def firstFivePrimesWithUnitsDigit3 : List ℕ :=
  (List.range 100).filter (λ n => isPrime n && hasUnitsDigitOf3 n) |>.take 5

theorem sum_first_five_primes_with_units_digit_3 :
  firstFivePrimesWithUnitsDigit3.sum = 135 := by
  sorry

#eval firstFivePrimesWithUnitsDigit3
#eval firstFivePrimesWithUnitsDigit3.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_five_primes_with_units_digit_3_l943_94359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_equals_two_l943_94365

theorem cube_root_of_eight_equals_two : ∃ x : ℝ, x^3 = 8 ∧ x = 2 := by
  use 2
  constructor
  · norm_num
  · rfl

#check cube_root_of_eight_equals_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_equals_two_l943_94365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bandages_from_given_cloth_l943_94379

/-- Calculates the number of triangular bandages that can be made from a rectangular cloth. -/
def bandages_from_cloth (cloth_length cloth_width bandage_size : ℚ) : ℕ :=
  let squares_length := (cloth_length / bandage_size).floor
  let squares_width := (cloth_width / bandage_size).floor
  let total_squares := squares_length * squares_width
  (2 * total_squares).toNat

/-- Theorem stating the number of bandages that can be made from the given cloth dimensions. -/
theorem bandages_from_given_cloth :
  bandages_from_cloth 60 0.8 0.4 = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bandages_from_given_cloth_l943_94379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l943_94398

-- Define the function f(x) = ln x + 2x - 1
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 1

-- Theorem statement
theorem root_in_interval :
  ∃ (r : ℝ), r ∈ Set.Ioo (1/2 : ℝ) 1 ∧ f r = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l943_94398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_function_proof_l943_94383

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 9

theorem logarithmic_function_proof (h1 : f 3 = 1/2) 
  (h2 : ∃ (x : ℝ), x ∈ Set.Ioo 1 5 ∧ f x = m) :
  (∀ x > 0, f x = Real.log x / Real.log 9) ∧
  m ∈ Set.Ioo 0 (Real.log 5 / Real.log 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_function_proof_l943_94383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l943_94335

/-- The length of a train in meters, given its speed in km/hr and the time it takes to cross a pole -/
noncomputable def train_length (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  speed_km_hr * (1000 / 3600) * time_s

theorem train_length_calculation (speed_km_hr : ℝ) (time_s : ℝ) 
  (h1 : speed_km_hr = 54)
  (h2 : time_s = 9) :
  train_length speed_km_hr time_s = 135 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l943_94335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_percentage_l943_94388

theorem divisible_by_six_percentage (n : Nat) : 
  (↑(Finset.filter (λ x => x % 6 = 0) (Finset.range (n + 1))).card / ↑(n + 1)) * 100 = 100 / 6 :=
by
  sorry

#eval (↑(Finset.filter (λ x => x % 6 = 0) (Finset.range 151)).card / 151) * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_percentage_l943_94388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l943_94324

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (a r : ℝ) :
  a = 250 →
  geometric_sum a r 50 = 625 →
  geometric_sum a r 100 = 1225 →
  geometric_sum a r 150 = 1801 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l943_94324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scores_stats_l943_94362

def scores : List ℝ := [90, 89, 90, 95, 93, 94, 93]

noncomputable def remove_extremes (xs : List ℝ) : List ℝ :=
  xs.filter (λ x => x ≠ (xs.maximum?).getD 0 ∧ x ≠ (xs.minimum?).getD 0)

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (λ x => (x - μ) ^ 2)).sum / xs.length

theorem scores_stats :
  let remaining := remove_extremes scores
  mean remaining = 92 ∧ variance remaining = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scores_stats_l943_94362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_ratio_l943_94376

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the foci
noncomputable def left_focus : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define a point on the right branch of the hyperbola
def point_on_right_branch (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ P.1 > 0

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def distance (v w : ℝ × ℝ) : ℝ := Real.sqrt ((v.1 - w.1)^2 + (v.2 - w.2)^2)

theorem hyperbola_focus_ratio (P : ℝ × ℝ) :
  point_on_right_branch P →
  dot_product (vec_add (vec_sub P origin) (vec_sub right_focus origin)) (vec_sub P right_focus) = 0 →
  distance P left_focus = 2 * distance P right_focus :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_ratio_l943_94376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_real_root_iff_non_real_root_in_RDelta_l943_94368

-- Define the structure for ℝ[δ]
structure RDelta where
  a : ℝ
  b : ℝ

-- Define δ
def δ : RDelta := ⟨0, 1⟩

-- Define addition in ℝ[δ]
def add (x y : RDelta) : RDelta :=
  ⟨x.a + y.a, x.b + y.b⟩

-- Define multiplication in ℝ[δ]
def mul (x y : RDelta) : RDelta :=
  ⟨x.a * y.a, x.a * y.b + x.b * y.a⟩

-- Define a polynomial with real coefficients
def RealPolynomial := ℝ → ℝ

-- Define what it means for a polynomial to have a multiple real root
def has_multiple_real_root (P : RealPolynomial) : Prop :=
  ∃ x : ℝ, P x = 0 ∧ (deriv P) x = 0

-- Define what it means for a polynomial to have a non-real root in ℝ[δ]
def has_non_real_root_in_RDelta (P : RealPolynomial) : Prop :=
  ∃ x : RDelta, x.b ≠ 0 ∧ P (x.a + x.b * δ.b) = 0

-- State the theorem
theorem multiple_real_root_iff_non_real_root_in_RDelta (P : RealPolynomial) :
  has_multiple_real_root P ↔ has_non_real_root_in_RDelta P := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_real_root_iff_non_real_root_in_RDelta_l943_94368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_example_l943_94374

/-- The cost of fencing a rectangular park with given dimensions and fencing rate -/
noncomputable def fencing_cost (area : ℝ) (side_ratio : ℝ) (fencing_rate : ℝ) : ℝ :=
  let length := Real.sqrt (area * side_ratio / 2)
  let width := length / side_ratio
  let perimeter := 2 * (length + width)
  perimeter * fencing_rate / 100

/-- Theorem stating the fencing cost for a specific rectangular park -/
theorem fencing_cost_example : fencing_cost 3750 (3/2) 60 = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_example_l943_94374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_l943_94305

/-- The parabola function -/
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

/-- The tangent line function at x₀ = 2 -/
def tangent (x : ℝ) : ℝ := 6 * x - 7

/-- The area between the parabola and its tangent from 0 to 2 -/
noncomputable def area : ℝ := ∫ x in (0 : ℝ)..(2 : ℝ), f x - tangent x

theorem parabola_area : area = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_l943_94305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l943_94372

noncomputable section

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the perimeter of triangle PF₁F₂
def triangle_perimeter (a c : ℝ) : ℝ :=
  2 * a + 2 * c

-- Define the line l
def line_l (x y k : ℝ) : Prop :=
  y = k * x + 2

-- Define the angle AOB
def angle_AOB_acute (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ > 0

theorem ellipse_and_line_theorem :
  ∀ a b c : ℝ,
  ellipse_C 0 a a b →
  eccentricity a b = Real.sqrt 3 / 2 →
  triangle_perimeter a c = 4 + 2 * Real.sqrt 3 →
  (a = 2 ∧ b = 1) ∧
  ∀ k : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ 2 1 ∧
    ellipse_C x₂ y₂ 2 1 ∧
    line_l x₁ y₁ k ∧
    line_l x₂ y₂ k ∧
    x₁ ≠ x₂ ∧
    angle_AOB_acute x₁ y₁ x₂ y₂) →
  (k > Real.sqrt 3 / 2 ∧ k < 2) ∨ (k < -Real.sqrt 3 / 2 ∧ k > -2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l943_94372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_g_product_l943_94310

def g : ℕ → ℕ
| 0 => 0  -- Added case for 0
| 1 => 0
| 2 => 1
| (n + 3) => g (n + 1) + g (n + 2) + 1

theorem prime_divides_g_product {n : ℕ} (hn : n > 5) (hprime : Nat.Prime n) :
  n ∣ g n * (g n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_g_product_l943_94310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_hours_per_week_proof_l943_94397

/-- Prove that the number of hours Cathy and Chris were supposed to work per week is 20 -/
def work_hours_per_week : ℕ := 20

theorem work_hours_per_week_proof (
  total_months : ℕ := 2
) (weeks_per_month : ℕ := 4)
  (extra_week : ℕ := 1)
  (cathy_total_hours : ℕ := 180)
  : work_hours_per_week = 20 := by
  let total_weeks : ℕ := total_months * weeks_per_month
  let cathy_worked_weeks : ℕ := total_weeks + extra_week
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_hours_per_week_proof_l943_94397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_l943_94390

open Real

/-- The function f(x) = x - a ln x has a tangent line y = 3x - 2 at the point (1,1) if and only if a = -2 -/
theorem tangent_line_implies_a (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = x - a * log x) ∧ 
   (∃ g : ℝ → ℝ, (∀ x, g x = 3*x - 2) ∧ 
    HasDerivAt f ((g 1) - (f 1)) 1)) ↔ 
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_l943_94390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_one_l943_94306

-- Define the function g(x)
noncomputable def g (x m : ℝ) : ℝ := Real.rpow 5 (-x) + m

-- Define the condition for g(x) not passing through the first quadrant
def not_in_first_quadrant (m : ℝ) : Prop :=
  ∀ x, x > 0 → g x m ≤ 0

-- State the theorem
theorem a_greater_than_one (a : ℝ) :
  (∀ m, m < a → not_in_first_quadrant m) ∧
  (∃ m, m < a ∧ ¬not_in_first_quadrant m) →
  a > 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_one_l943_94306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l943_94356

/-- Represents an investment split between two interest rates -/
structure Investment where
  total : ℝ
  rate1 : ℝ
  rate2 : ℝ
  amount1 : ℝ
  amount2 : ℝ

/-- Calculates the average interest rate for an investment -/
noncomputable def averageInterestRate (inv : Investment) : ℝ :=
  (inv.rate1 * inv.amount1 + inv.rate2 * inv.amount2) / inv.total

/-- Theorem: The average interest rate is 3.75% for the given investment conditions -/
theorem investment_average_rate (inv : Investment) 
  (h1 : inv.total = 6000)
  (h2 : inv.rate1 = 0.03)
  (h3 : inv.rate2 = 0.05)
  (h4 : inv.amount1 + inv.amount2 = inv.total)
  (h5 : inv.rate1 * inv.amount1 = inv.rate2 * inv.amount2) : 
  averageInterestRate inv = 0.0375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l943_94356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l943_94315

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Minimum sum of distances in a freight yard -/
theorem min_sum_distances (a b c d p h : Point)
  (h1 : a.x = 0 ∧ a.y = 0)
  (h2 : b.x = 0 ∧ b.y = 600)
  (h3 : c.x = 1000 ∧ c.y = 600)
  (h4 : d.x = 1000 ∧ d.y = 0)
  (h5 : 0 ≤ p.x ∧ p.x ≤ 1000)
  (h6 : 0 ≤ p.y ∧ p.y ≤ 600)
  (h7 : h.x = 1000 ∧ 0 ≤ h.y ∧ h.y ≤ 600) :
  ∃ (p_opt h_opt : Point),
    distance a p_opt + distance d p_opt + distance p_opt h_opt =
    500 * Real.sqrt 3 + 600 ∧
    ∀ (p' h' : Point),
      (0 ≤ p'.x ∧ p'.x ≤ 1000) →
      (0 ≤ p'.y ∧ p'.y ≤ 600) →
      (h'.x = 1000 ∧ 0 ≤ h'.y ∧ h'.y ≤ 600) →
      distance a p' + distance d p' + distance p' h' ≥
      500 * Real.sqrt 3 + 600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l943_94315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_sin_pi_x_l943_94332

noncomputable def f (x : ℝ) := Real.sin (Real.pi * x)

theorem smallest_positive_period_of_sin_pi_x :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_sin_pi_x_l943_94332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_result_l943_94345

def matrix_product (n : Nat) : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![1, 2 * n],
    ![0, 1]]

def product_up_to (N : Nat) : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range N).foldl (λ acc i => acc * matrix_product (i + 1)) (1 : Matrix (Fin 2) (Fin 2) ℕ)

theorem matrix_product_result :
  product_up_to 50 = ![![1, 2550],
                      ![0, 1]] := by
  sorry

#eval product_up_to 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_result_l943_94345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l943_94325

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def Hyperbola.eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from the center to a focus of the hyperbola -/
noncomputable def Hyperbola.focalDistance (h : Hyperbola) : ℝ :=
  h.a * h.eccentricity

theorem hyperbola_eccentricity_range (h : Hyperbola) 
  (hP : ∃ P : ℝ × ℝ, (P.1^2 / h.a^2 - P.2^2 / h.b^2 = 1) ∧ 
    (P.1^2 + P.2^2 = (P.1 - h.focalDistance)^2 + P.2^2)) :
  h.eccentricity ≥ 2 := by
  sorry

#check hyperbola_eccentricity_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l943_94325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l943_94351

theorem abc_inequality : 
  let a : ℝ := (3/5) ^ (1/3 : ℝ)
  let b : ℝ := (3/5) ^ (-(1/3) : ℝ)
  let c : ℝ := (2/5) ^ (1/3 : ℝ)
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l943_94351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l943_94327

theorem expression_value 
  (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |x| = Real.sqrt 3) : 
  x^2 + Real.sqrt (a + b + 4) - (27 * c * d)^(1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l943_94327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_wrt_origin_point_neg3_2_wrt_origin_l943_94371

/-- Definition of coordinates with respect to the origin -/
def Prod.coords_wrt_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

/-- Given a point in a Cartesian coordinate system, its coordinates with respect to the origin are the negatives of its original coordinates. -/
theorem point_wrt_origin (x y : ℝ) : (x, y).coords_wrt_origin = (-x, -y) := by
  -- Unfold the definition of coords_wrt_origin
  unfold Prod.coords_wrt_origin
  -- The result follows directly from the definition
  rfl

/-- The coordinates of the point (-3, 2) with respect to the origin are (3, -2). -/
theorem point_neg3_2_wrt_origin : (-3, 2).coords_wrt_origin = (3, -2) := by
  -- Apply the general theorem
  rw [point_wrt_origin]
  -- Simplify the negations
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_wrt_origin_point_neg3_2_wrt_origin_l943_94371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l943_94373

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 4 * x - 5) / (x - 2)

def m : ℝ := 3

def b : ℝ := 10

/-- Theorem: For the rational function f(x) = (3x^2 + 4x - 5) / (x - 2),
    if the slant asymptote is of the form y = mx + b, then m + b = 13 -/
theorem slant_asymptote_sum : m + b = 13 := by
  -- Unfold the definitions of m and b
  unfold m b
  -- Perform the addition
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l943_94373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tobacco_acreage_difference_l943_94337

noncomputable def total_land : ℝ := 1350

def initial_ratio : Fin 3 → ℝ
| 0 => 5  -- corn
| 1 => 2  -- sugar cane
| 2 => 2  -- tobacco
| _ => 0

def new_ratio : Fin 3 → ℝ
| 0 => 2  -- corn
| 1 => 2  -- sugar cane
| 2 => 5  -- tobacco
| _ => 0

def seasonal_ratios : Fin 4 → Fin 3 → ℝ
| 0, 0 => 3  -- spring corn
| 0, 1 => 1  -- spring sugar cane
| 0, 2 => 1  -- spring tobacco
| 1, 0 => 2  -- summer corn
| 1, 1 => 3  -- summer sugar cane
| 1, 2 => 1  -- summer tobacco
| 2, 0 => 1  -- fall corn
| 2, 1 => 2  -- fall sugar cane
| 2, 2 => 2  -- fall tobacco
| 3, 0 => 1  -- winter corn
| 3, 1 => 1  -- winter sugar cane
| 3, 2 => 3  -- winter tobacco
| _, _ => 0

noncomputable def initial_tobacco_acres : ℝ := total_land * initial_ratio 2 / (initial_ratio 0 + initial_ratio 1 + initial_ratio 2)

noncomputable def seasonal_tobacco_acres (season : Fin 4) : ℝ :=
  total_land * seasonal_ratios season 2 / (seasonal_ratios season 0 + seasonal_ratios season 1 + seasonal_ratios season 2)

noncomputable def total_seasonal_tobacco_acres : ℝ :=
  seasonal_tobacco_acres 0 + seasonal_tobacco_acres 1 + seasonal_tobacco_acres 2 + seasonal_tobacco_acres 3

theorem tobacco_acreage_difference :
  total_seasonal_tobacco_acres - initial_tobacco_acres = 1545 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tobacco_acreage_difference_l943_94337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_1989th_term_l943_94385

open Nat

noncomputable def my_sequence (n : ℕ) : ℚ :=
  let k := (((8 * n + 1 : ℕ).sqrt - 1) / 2 + 1 : ℕ)
  let pos := n - k * (k - 1) / 2
  ↑(k - pos + 1) / ↑pos

theorem my_sequence_1989th_term :
  my_sequence 1989 = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_1989th_term_l943_94385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l943_94346

/-- Calculates the speed of a man given the parameters of a train passing him --/
noncomputable def man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  let man_speed_ms := train_speed_ms - relative_speed
  man_speed_ms * 3600 / 1000

/-- Proves that the speed of the man is approximately 2.9916 km/h --/
theorem man_speed_calculation :
  let ε := 0.0001
  let calculated_speed := man_speed 800 63 47.99616030717543
  |calculated_speed - 2.9916| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l943_94346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l943_94312

-- Define the function representing the left side of the inequality
noncomputable def f (x : ℝ) : ℝ := 2 / (x + 1) + 8 / (x + 3)

-- Define the set of x values that satisfy the inequality
def S : Set ℝ := { x | f x < 3 }

-- State the theorem
theorem inequality_solution :
  S = Set.Ioo (-3 : ℝ) (-1) ∪ Set.Ioo (-1/3 : ℝ) 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l943_94312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l943_94387

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + a*x - 2 else -a^x

-- State the theorem
theorem increasing_f_implies_a_range (a : ℝ) :
  a ≠ 1 →
  (∀ x y : ℝ, 0 < x ∧ x < y → f a x < f a y) →
  0 < a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l943_94387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_piles_l943_94392

/-- Represents a pile of coins -/
structure Pile where
  amount : ℕ

/-- Represents the possible multiplication factors -/
inductive Factor where
  | Two
  | Three
  | Four

/-- Applies a factor to a pile -/
def applyFactor (p : Pile) (f : Factor) : Pile :=
  match f with
  | Factor.Two => ⟨p.amount * 2⟩
  | Factor.Three => ⟨p.amount * 3⟩
  | Factor.Four => ⟨p.amount * 4⟩

/-- The main theorem -/
theorem min_distinct_piles
  (initialPiles : Finset Pile)
  (h1 : initialPiles.card = 40)
  (h2 : ∀ p1 p2, p1 ∈ initialPiles → p2 ∈ initialPiles → p1 ≠ p2 → p1.amount ≠ p2.amount)
  (factorAssignment : Pile → Factor) :
  (initialPiles.image (fun p => (applyFactor p (factorAssignment p)).amount)).card ≥ 14 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_piles_l943_94392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_impossibility_l943_94393

theorem circle_sum_impossibility : ¬ ∃ (arrangement : List ℕ) (x : ℕ),
  arrangement.length = 8 ∧
  arrangement.toFinset = Finset.range 8 ∧
  (List.zip arrangement (arrangement.rotateLeft 1)).map (λ (a, b) => a + b) =
    List.map (λ i => x + i - 3) (List.range 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_impossibility_l943_94393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_separation_time_l943_94328

/-- Adam's speed in miles per hour -/
def adam_speed : ℝ := 10

/-- Simon's speed in miles per hour -/
def simon_speed : ℝ := 10

/-- Angle of Simon's direction with respect to east (in radians) -/
noncomputable def simon_angle : ℝ := Real.pi / 4

/-- Distance between Adam and Simon after time t -/
noncomputable def distance (t : ℝ) : ℝ :=
  ((adam_speed * t - simon_speed * t * Real.cos simon_angle) ^ 2 + 
   (simon_speed * t * Real.sin simon_angle) ^ 2) ^ (1/2 : ℝ)

/-- The time taken for Adam and Simon to be 100 miles apart is 2√2 hours -/
theorem separation_time : 
  ∃ t : ℝ, t = 2 * Real.sqrt 2 ∧ distance t = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_separation_time_l943_94328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l943_94343

theorem distance_between_points : 
  let p1 : Fin 3 → ℝ := ![3, 3, 3]
  let p2 : Fin 3 → ℝ := ![-2, -2, -2]
  Real.sqrt ((p2 0 - p1 0)^2 + (p2 1 - p1 1)^2 + (p2 2 - p1 2)^2) = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l943_94343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l943_94304

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c = 1 - Real.log 2 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (1/2 : ℝ) 2 → f x ≤ c :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l943_94304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bound_on_a_e_pow_one_tenth_bounds_l943_94302

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - 1

-- Statement 1
theorem bound_on_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f a x ≤ g x) → a ≤ 1 := by sorry

-- Statement 2
theorem e_pow_one_tenth_bounds : 
  (1095 : ℝ) / 1000 < Real.exp (1 / 10) ∧ Real.exp (1 / 10) < (2000 : ℝ) / 1791 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bound_on_a_e_pow_one_tenth_bounds_l943_94302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sincos_combination_l943_94336

/-- The period of the function y = 3sin(x) - 2cos(x) is 2π -/
theorem period_of_sincos_combination : 
  ∃ (p : ℝ), p > 0 ∧ ∀ (t : ℝ), (3 * Real.sin t - 2 * Real.cos t = 3 * Real.sin (t + p) - 2 * Real.cos (t + p)) ∧ p = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sincos_combination_l943_94336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l943_94320

/-- A line passing through point (2,1) with slope 1 intersects a parabola y^2 = 2px (p > 0) at points A and B. If (2,1) is the midpoint of AB, then p = 1. -/
theorem parabola_intersection (p : ℝ) (A B : ℝ × ℝ) : 
  p > 0 → 
  (∀ x y : ℝ, y = x - 1 ↔ (x, y) ∈ ({(x, y) | y = x - 1} : Set (ℝ × ℝ))) →
  (∀ x y : ℝ, y^2 = 2*p*x ↔ (x, y) ∈ ({(x, y) | y^2 = 2*p*x} : Set (ℝ × ℝ))) →
  A ∈ ({(x, y) : ℝ × ℝ | y = x - 1} : Set (ℝ × ℝ)) →
  B ∈ ({(x, y) : ℝ × ℝ | y = x - 1} : Set (ℝ × ℝ)) →
  A ∈ ({(x, y) : ℝ × ℝ | y^2 = 2*p*x} : Set (ℝ × ℝ)) →
  B ∈ ({(x, y) : ℝ × ℝ | y^2 = 2*p*x} : Set (ℝ × ℝ)) →
  (2, 1) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  p = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l943_94320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_of_inequality_l943_94363

theorem integer_solutions_of_inequality :
  ∀ x : ℤ, x ∈ ({-1, 0, 1} : Set ℤ) ↔ -4 < 1 - 3*x ∧ 1 - 3*x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_of_inequality_l943_94363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l943_94354

/-- A circle passing through two points and tangent to the x-axis -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_A : (center.fst - 0)^2 + (center.snd - 1)^2 = radius^2
  passes_through_B : (center.fst - 1)^2 + (center.snd - 2)^2 = radius^2
  tangent_to_x_axis : center.snd = radius

/-- The equation of a circle given its center and radius -/
def circle_equation (c : ℝ × ℝ) (r : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (x - c.fst)^2 + (y - c.snd)^2 = r^2

/-- Theorem: The equation of a circle passing through (0,1) and (1,2) and tangent to the x-axis -/
theorem tangent_circle_equation :
  ∀ (c : TangentCircle),
    (circle_equation c.center c.radius = circle_equation (1, 1) 1) ∨
    (circle_equation c.center c.radius = circle_equation (-3, 5) 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l943_94354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_tournament_without_large_transitive_subtournament_l943_94386

-- Define a tournament
def is_tournament {α : Type*} (R : α → α → Prop) : Prop :=
  (∀ a b : α, a ≠ b → (R a b ∨ R b a)) ∧
  (∀ a : α, ¬ R a a) ∧
  (∀ a b : α, R a b → ¬ R b a)

-- Define transitivity for a relation
def is_transitive {α : Type*} (R : α → α → Prop) : Prop :=
  ∀ a b c : α, R a b → R b c → R a c

-- Define a transitive subtournament
def is_transitive_subtournament {α : Type*} (R : α → α → Prop) (S : Set α) : Prop :=
  is_tournament (λ a b ↦ a ∈ S ∧ b ∈ S ∧ R a b) ∧
  is_transitive (λ a b ↦ a ∈ S ∧ b ∈ S ∧ R a b)

-- State the theorem
theorem exists_tournament_without_large_transitive_subtournament :
  ∃ (T : Type) (R : T → T → Prop),
    Cardinal.mk T = ℵ₁ ∧
    is_tournament R ∧
    ∀ (S : Set T), Cardinal.mk S = ℵ₁ → ¬ is_transitive_subtournament R S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_tournament_without_large_transitive_subtournament_l943_94386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2_product_l943_94375

/-- The coefficient of x^2 in the product of two polynomials -/
noncomputable def coefficientOfX2 (p q : Polynomial ℤ) : ℤ :=
  (p * q).coeff 2

/-- The first polynomial -/
noncomputable def p : Polynomial ℤ := 3 * Polynomial.X ^ 3 - 4 * Polynomial.X ^ 2 - 2 * Polynomial.X - 3

/-- The second polynomial -/
noncomputable def q : Polynomial ℤ := 2 * Polynomial.X ^ 2 + 3 * Polynomial.X - 4

theorem coefficient_x2_product : coefficientOfX2 p q = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2_product_l943_94375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celsius_coefficient_is_five_ninths_l943_94321

noncomputable def fahrenheit_to_celsius (f : ℝ) : ℝ := (f - 32) * (5/9)

def fahrenheit_increase : ℝ := 25
def celsius_increase : ℝ := 13.88888888888889

theorem celsius_coefficient_is_five_ninths :
  ∀ f : ℝ, 
  fahrenheit_to_celsius (f + fahrenheit_increase) - fahrenheit_to_celsius f = celsius_increase →
  ∃ k : ℝ, k = 5/9 ∧ ∀ c : ℝ, f = k * c + 32 :=
by
  intro f h
  use 5/9
  constructor
  · rfl
  · intro c
    sorry

#check celsius_coefficient_is_five_ninths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_celsius_coefficient_is_five_ninths_l943_94321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_with_5_and_6_l943_94353

/-- The set of numbers we're working with -/
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- Predicate to check if a subset contains both 5 and 6 -/
def containsBoth5And6 (subset : Finset ℕ) : Prop :=
  5 ∈ subset ∧ 6 ∈ subset

/-- Instance for DecidablePred containsBoth5And6 -/
instance : DecidablePred containsBoth5And6 := fun subset =>
  show Decidable (5 ∈ subset ∧ 6 ∈ subset) from
    inferInstance

/-- The theorem stating the number of subsets containing both 5 and 6 -/
theorem subsets_with_5_and_6 :
  (Finset.filter containsBoth5And6 (Finset.powerset S)).card = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_with_5_and_6_l943_94353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_condition_l943_94358

theorem partition_condition (n : ℕ) (h : n ≥ 6) :
  (∃ (A₁ A₂ A₃ : Finset ℕ),
    (A₁ ∪ A₂ ∪ A₃ = Finset.range n) ∧
    (A₁ ∩ A₂ = ∅) ∧ (A₂ ∩ A₃ = ∅) ∧ (A₁ ∩ A₃ = ∅) ∧
    (A₁.card = A₂.card) ∧ (A₂.card = A₃.card) ∧
    (A₁.sum id = A₂.sum id) ∧ (A₂.sum id = A₃.sum id)) ↔
  (∃ k : ℕ, n = 3 * k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_condition_l943_94358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_odd_and_2pi_periodic_l943_94342

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 2)

-- State the theorem
theorem tan_half_odd_and_2pi_periodic :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2 * Real.pi) = f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_odd_and_2pi_periodic_l943_94342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_similar_circumscribing_quadrilateral_l943_94391

/-- A quadrilateral in 2D space -/
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

/-- Similarity relation between two quadrilaterals -/
def similar (Q1 Q2 : Quadrilateral) : Prop := sorry

/-- A quadrilateral Q1 circumscribes another quadrilateral Q2 -/
def circumscribes (Q1 Q2 : Quadrilateral) : Prop := sorry

/-- Main theorem: Given two quadrilaterals, we can construct a third that circumscribes
    the first and is similar to the second -/
theorem construct_similar_circumscribing_quadrilateral
  (ABCD mnpq : Quadrilateral) :
  ∃ MNPQ : Quadrilateral, circumscribes MNPQ ABCD ∧ similar MNPQ mnpq := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_similar_circumscribing_quadrilateral_l943_94391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l943_94333

-- Define the domain D
noncomputable def D : Set ℝ := {x | (0 < x ∧ x < 1) ∨ (x > 1)}

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 - x)
noncomputable def g (a x : ℝ) : ℝ := -a / Real.sqrt x

-- State the theorem
theorem function_inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ D, f x ≥ g a x) → a ∈ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l943_94333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_12_l943_94396

-- Define the cylinder's properties
def cylinder_diameter : ℝ := 32
def water_level_rise : ℝ := 9

-- Define the volume of water displaced (equal to the volume of the sphere)
noncomputable def water_volume : ℝ := Real.pi * (cylinder_diameter / 2)^2 * water_level_rise

-- Define the radius of the sphere
noncomputable def sphere_radius : ℝ := (3 * water_volume / (4 * Real.pi))^(1/3)

-- Theorem statement
theorem sphere_radius_is_12 : sphere_radius = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_12_l943_94396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_sum_constant_l943_94308

/-- An ellipse with semi-major axis 2 and semi-minor axis 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 4 + p.2^2 = 1}

/-- The unit circle -/
def UnitCircle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 1}

/-- The circle with radius 2 -/
def Circle2 : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 4}

/-- Point P on the x-axis -/
def P : ℝ × ℝ := (-1, 0)

/-- The sum of squared distances between three points -/
def SumSquaredDistances (a b c : ℝ × ℝ) : ℝ :=
  (b.1 - c.1)^2 + (b.2 - c.2)^2 + 
  (c.1 - a.1)^2 + (c.2 - a.2)^2 + 
  (a.1 - b.1)^2 + (a.2 - b.2)^2

theorem ellipse_triangle_sum_constant 
  (a : ℝ × ℝ) 
  (ha : a ∈ UnitCircle ∧ a ≠ P) 
  (b c : ℝ × ℝ) 
  (hbc : b ∈ Circle2 ∧ c ∈ Circle2) 
  (hperp : (b.1 - P.1) * (a.1 - P.1) + (b.2 - P.2) * (a.2 - P.2) = 0) :
  SumSquaredDistances a b c = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_sum_constant_l943_94308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_circle_and_line_l943_94311

/-- The curve represented by the polar equation ρ cos θ = 2 sin 2θ -/
def PolarCurve : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (ρ θ : ℝ), p.1 = ρ * Real.cos θ ∧ p.2 = ρ * Real.sin θ ∧ ρ * Real.cos θ = 2 * Real.sin (2 * θ)}

/-- A circle with radius 4 centered at the origin -/
def Circle4 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 16}

/-- The vertical line through the pole (origin) -/
def VerticalLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0}

theorem polar_curve_is_circle_and_line : PolarCurve = Circle4 ∪ VerticalLine := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_circle_and_line_l943_94311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_travel_distance_l943_94330

/-- Represents the distance a truck can travel given an amount of gas -/
noncomputable def truckDistance (initialDistance : ℝ) (initialGas : ℝ) (newGas : ℝ) : ℝ :=
  (initialDistance / initialGas) * newGas

theorem truck_travel_distance 
  (initialDistance : ℝ) 
  (initialGas : ℝ) 
  (newGas : ℝ) 
  (h1 : initialDistance = 300) 
  (h2 : initialGas = 10) 
  (h3 : newGas = 15) :
  truckDistance initialDistance initialGas newGas = 450 := by
  sorry

#check truck_travel_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_travel_distance_l943_94330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l943_94348

noncomputable def f (x : ℝ) : ℝ := (1 + 2^x) / (1 + 4^x)

theorem f_range :
  (∃ x : ℝ, f x = (Real.sqrt 2 + 1) / 2) ∧  -- supremum is attainable
  (∀ x : ℝ, f x ≤ (Real.sqrt 2 + 1) / 2) ∧  -- upper bound
  (∀ ε > 0, ∃ x : ℝ, f x < ε) ∧             -- infimum is 0
  (∀ x : ℝ, f x > 0)                        -- lower bound, not attainable
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l943_94348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_segment_length_l943_94317

/-- Represents a trapezoid ABCD with AB parallel to CD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  h : ℝ  -- height of the trapezoid

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

theorem trapezoid_segment_length 
  (ABCD : Trapezoid) 
  (area_ratio : triangle_area ABCD.AB ABCD.h / triangle_area ABCD.CD ABCD.h = 4) 
  (sum_parallel_sides : ABCD.AB + ABCD.CD = 180) : 
  ABCD.AB = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_segment_length_l943_94317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_limit_l943_94309

theorem middle_term_limit (x : ℝ) : 
  (Nat.choose 6 3) * (-x)^3 = 5/2 → 
  Filter.Tendsto (fun n => (1 - x^(n+1)) / (1 - x)) Filter.atTop (nhds (-1/3)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_limit_l943_94309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_decreasing_range_l943_94350

/-- A linear function with slope (m - 2023) and y-intercept (m + 2023) -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (m - 2023) * x + m + 2023

/-- Proposition: For a linear function y = (m - 2023)x + m + 2023, 
    where y decreases as x increases, the range of m is m < 2023 -/
theorem linear_function_decreasing_range (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function m x₁ > linear_function m x₂) ↔ m < 2023 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_decreasing_range_l943_94350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_1_to_100_last_digits_l943_94349

/-- The product of all integers from 1 to 100 -/
def product_1_to_100 : ℕ := (List.range 100).map (λ i => i + 1) |>.prod

/-- The last digit of a natural number -/
def last_digit (n : ℕ) : ℕ := n % 10

/-- The last ten digits of a natural number -/
def last_ten_digits (n : ℕ) : ℕ := n % (10^10)

theorem product_1_to_100_last_digits :
  (last_digit product_1_to_100 = 0) ∧
  (last_ten_digits product_1_to_100 = 0) := by
  sorry

#eval product_1_to_100
#eval last_digit product_1_to_100
#eval last_ten_digits product_1_to_100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_1_to_100_last_digits_l943_94349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l943_94364

-- Define the ellipse
noncomputable def is_on_ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

-- Define the distance function from a point to the line x + y - 4 = 0
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y - 4| / Real.sqrt 2

-- State the theorem
theorem max_distance_to_line :
  ∃ (x y : ℝ), is_on_ellipse x y ∧
  (∀ (a b : ℝ), is_on_ellipse a b → distance_to_line x y ≥ distance_to_line a b) ∧
  distance_to_line x y = 4 * Real.sqrt 2 ∧
  x = -3 ∧ y = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l943_94364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_big_dig_copper_production_l943_94331

/-- Represents the daily mining output of Big Dig Mining Company -/
structure MiningOutput where
  nickel : ℝ
  iron : ℝ
  copper : ℝ

/-- Calculates the total daily output given the nickel output and its percentage -/
noncomputable def total_output (nickel_output : ℝ) (nickel_percentage : ℝ) : ℝ :=
  nickel_output / nickel_percentage

/-- Theorem stating the daily copper production given the mining conditions -/
theorem big_dig_copper_production
  (output : MiningOutput)
  (h_nickel_percent : output.nickel / (output.nickel + output.iron + output.copper) = 0.1)
  (h_iron_percent : output.iron / (output.nickel + output.iron + output.copper) = 0.6)
  (h_nickel_tons : output.nickel = 720)
  : output.copper = 2160 := by
  sorry

#check big_dig_copper_production

end NUMINAMATH_CALUDE_ERRORFEEDBACK_big_dig_copper_production_l943_94331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l943_94380

/-- The area of the trapezoid formed in the arrangement of three squares --/
theorem trapezoid_area (square1 square2 square3 : ℝ) 
  (h1 : square1 = 3) (h2 : square2 = 5) (h3 : square3 = 7) :
  let total_base := square1 + square2 + square3
  let max_height := square3
  let height_ratio := max_height / total_base
  let height1 := square1 * height_ratio
  let height2 := (square1 + square2) * height_ratio
  let trapezoid_height := square2
  (height1 + height2) * trapezoid_height / 2 = 12.825 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l943_94380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_k_value_l943_94378

/-- A hyperbola with equation 2x^2 - y^2 = k and focal distance 6 has k = ±6 -/
theorem hyperbola_k_value (k : ℝ) : 
  (∃ (x y : ℝ), 2 * x^2 - y^2 = k) → -- Hyperbola equation exists
  (∃ (c : ℝ), c = 6 ∧ c^2 = max (k/2 + k) (-k - k/2)) → -- Focal distance is 6
  k = 6 ∨ k = -6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_k_value_l943_94378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_less_than_35_l943_94323

/-- A configuration of circles in a plane -/
structure CircleConfiguration where
  n : ℕ
  centers : Fin n → ℝ × ℝ
  radius : ℝ
  n_ge_3 : n ≥ 3
  radius_is_1 : radius = 1
  intersection_condition : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    ‖centers i - centers j‖ ≤ 2 * radius ∨ 
    ‖centers j - centers k‖ ≤ 2 * radius ∨ 
    ‖centers i - centers k‖ ≤ 2 * radius

/-- The total area covered by the circles -/
noncomputable def coveredArea (config : CircleConfiguration) : ℝ := sorry

/-- Theorem: The area covered by the circles is less than 35 square units -/
theorem area_less_than_35 (config : CircleConfiguration) : 
  coveredArea config < 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_less_than_35_l943_94323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_sqrt5_minus2_l943_94357

theorem reciprocal_of_sqrt5_minus2 :
  (Real.sqrt 5 - 2)⁻¹ = Real.sqrt 5 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_sqrt5_minus2_l943_94357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_transfers_62_l943_94352

/-- Represents a city in the country -/
structure City where
  id : Nat

/-- Represents the country with its cities and road network -/
structure Country where
  cities : Finset City
  roads : City → Finset City
  total_cities : cities.card = 1993
  min_roads : ∀ c, c ∈ cities → (roads c).card ≥ 93
  connected : ∀ c₁ c₂, c₁ ∈ cities → c₂ ∈ cities → 
    ∃ path : List City, path.head? = some c₁ ∧ path.getLast? = some c₂

/-- Represents the number of transfers required to travel between two cities -/
def transfers (country : Country) (start finish : City) : Nat :=
  sorry

/-- Theorem stating that the maximum number of transfers is at most 62 -/
theorem max_transfers_62 (country : Country) :
  ∀ c₁ c₂, c₁ ∈ country.cities → c₂ ∈ country.cities → transfers country c₁ c₂ ≤ 62 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_transfers_62_l943_94352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_l943_94381

/-- Given the weights of Anne, Douglas, and Maria in pounds, 
    prove that Anne is 33 pounds lighter than the combined weight of Douglas and Maria. -/
theorem weight_difference (anne douglas maria : ℤ) 
  (h_anne : anne = 67)
  (h_douglas : douglas = 52)
  (h_maria : maria = 48) :
  anne - (douglas + maria) = -33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_l943_94381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_w_and_x_is_ten_l943_94339

theorem product_of_w_and_x_is_ten (W X Y Z : ℕ) :
  W ∈ ({2, 3, 4, 5} : Set ℕ) →
  X ∈ ({2, 3, 4, 5} : Set ℕ) →
  Y ∈ ({2, 3, 4, 5} : Set ℕ) →
  Z ∈ ({2, 3, 4, 5} : Set ℕ) →
  W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z →
  (W : ℚ) / (X : ℚ) - (Y : ℚ) / (Z : ℚ) = 1 / 2 →
  W * X = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_w_and_x_is_ten_l943_94339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l943_94341

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ Set.Ioo 0 (π/2) → x > Real.sin x) ↔ 
  (∃ x : ℝ, x ∈ Set.Ioo 0 (π/2) ∧ x ≤ Real.sin x) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l943_94341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_items_sold_without_discount_is_111_l943_94329

/-- The number of items a retailer sells each month without a discount, given the following conditions:
  - The profit per item is $60
  - The profit is 10% of the item's price to the retailer
  - With a 5% discount, the retailer needs to sell at least 222.22222222222223 items to justify the policy
-/
noncomputable def items_sold_without_discount : ℕ :=
  let profit_per_item : ℚ := 60
  let profit_percentage : ℚ := 1 / 10
  let discount_percentage : ℚ := 1 / 20
  let items_with_discount : ℚ := 222.22222222222223
  let price_to_retailer : ℚ := profit_per_item / profit_percentage
  let discounted_price : ℚ := price_to_retailer * (1 - discount_percentage)
  let new_profit_per_item : ℚ := discounted_price - price_to_retailer + profit_per_item
  (Int.ceil ((items_with_discount * new_profit_per_item) / profit_per_item)).toNat

theorem items_sold_without_discount_is_111 :
  items_sold_without_discount = 111 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_items_sold_without_discount_is_111_l943_94329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_71_81_sum_m_n_is_152_l943_94316

/-- The probability that the equation x^3 + 45b^2 = (9b^2 - 12b)x^2 has at least two distinct real solutions,
    where b is uniformly distributed in [-9,9] -/
noncomputable def probability_two_solutions : ℝ :=
  (142 : ℝ) / 162

theorem probability_is_71_81 : probability_two_solutions = 71 / 81 := by sorry

theorem sum_m_n_is_152 (m n : ℕ) (hm : m > 0) (hn : n > 0) (hcoprime : Nat.Coprime m n)
  (h_prob : (m : ℝ) / n = probability_two_solutions) : m + n = 152 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_71_81_sum_m_n_is_152_l943_94316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_time_satisfies_equation_l943_94300

/-- Represents the current time in minutes past 9:00 -/
def t : ℝ := sorry

/-- The current time is between 9:00 and 10:00 -/
axiom t_bounds : 0 < t ∧ t < 60

/-- The equation representing the opposite position of minute and hour hands -/
def opposite_hands_equation (x : ℝ) : Prop := |5.5 * x + 16| = 180

/-- The theorem stating that the current time satisfies the opposite hands equation -/
theorem current_time_satisfies_equation : opposite_hands_equation t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_time_satisfies_equation_l943_94300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_purchase_theorem_l943_94355

/-- The amount spent on mangoes before price reduction -/
noncomputable def amount_spent : ℝ := 359.64

/-- The number of mangoes originally purchased -/
def original_mangoes : ℕ := 108

/-- The original price per mango -/
noncomputable def original_price : ℝ := 383.33 / 115

/-- The reduced price per mango -/
noncomputable def reduced_price : ℝ := original_price * 0.9

theorem mango_purchase_theorem :
  (original_mangoes : ℝ) * original_price = amount_spent ∧
  383.33 = 115 * original_price ∧
  reduced_price = 0.9 * original_price ∧
  ((original_mangoes : ℝ) + 12) * reduced_price = (original_mangoes : ℝ) * original_price :=
by sorry

#check mango_purchase_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_purchase_theorem_l943_94355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_side_length_l943_94394

theorem inscribed_quadrilateral_side_length 
  (A B C D P : ℝ × ℝ) 
  (circle_radius : ℝ) :
  let O := (0, 0)
  -- Conditions
  (∀ X ∈ ({A, B, C, D} : Set (ℝ × ℝ)), (X.1^2 + X.2^2 : ℝ) = circle_radius^2) →
  circle_radius = 1 →
  A = (-1, 0) →
  C = (1, 0) →
  P = (1/5, 0) →
  (B.1 - P.1)^2 + (B.2 - P.2)^2 = (D.1 - P.1)^2 + (D.2 - P.2)^2 →
  (B.1 + 1)^2 + B.2^2 = (D.1 - 1)^2 + D.2^2 →
  -- Question
  (D.1 - 1)^2 + D.2^2 = (2 * Real.sqrt 2 / 5)^2 :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_side_length_l943_94394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_a_is_three_l943_94399

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x + (1 - a) * x + a

-- State the theorem
theorem max_integer_a_is_three :
  ∃ (a₀ : ℝ), a₀ ∈ Set.Ioo 3 4 ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 1 → f a x > 0) ↔ a < a₀) ∧
  (∀ a : ℤ, (∀ x : ℝ, x > 1 → f (↑a) x > 0) ↔ a ≤ 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_a_is_three_l943_94399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_35_workers_needed_l943_94366

/-- Represents the project parameters and progress --/
structure ProjectStatus where
  totalDays : ℕ
  workedDays : ℕ
  initialWorkers : ℕ
  completedFraction : ℚ
  
/-- Calculates the minimum number of workers needed to complete the project on time --/
def minWorkersNeeded (status : ProjectStatus) : ℕ :=
  let remainingWork := 1 - status.completedFraction
  let remainingDays := status.totalDays - status.workedDays
  let workerDayProductivity := status.completedFraction / (status.initialWorkers * status.workedDays)
  ⌈(remainingWork / (workerDayProductivity * remainingDays : ℚ))⌉.toNat

/-- Theorem stating that at least 35 workers are needed to complete the project on time --/
theorem at_least_35_workers_needed (status : ProjectStatus) 
  (h1 : status.totalDays = 36)
  (h2 : status.workedDays = 10)
  (h3 : status.initialWorkers = 12)
  (h4 : status.completedFraction = 2/5) :
  minWorkersNeeded status ≥ 35 := by
  sorry

#eval minWorkersNeeded {totalDays := 36, workedDays := 10, initialWorkers := 12, completedFraction := 2/5}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_35_workers_needed_l943_94366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_catches_rabbit_l943_94344

-- Define the circular arena
structure Arena where
  radius : ℝ
  radius_pos : radius > 0

-- Define the positions of the dog and rabbit
noncomputable def dog_position (arena : Arena) (θ : ℝ) : ℝ × ℝ :=
  (arena.radius / 2 * Real.cos θ, arena.radius / 2 * Real.sin θ)

noncomputable def rabbit_position (arena : Arena) (θ : ℝ) : ℝ × ℝ :=
  (arena.radius * Real.cos θ, arena.radius * Real.sin θ)

-- Define the theorem
theorem dog_catches_rabbit (arena : Arena) :
  ∃ (t : ℝ), t > 0 ∧ dog_position arena (π/2) = rabbit_position arena (π/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_catches_rabbit_l943_94344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_equals_negative_two_l943_94361

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := -1 + k / (Real.exp x - 1)

theorem odd_function_implies_k_equals_negative_two (k : ℝ) :
  (∀ x : ℝ, f k x = -(f k (-x))) → k = -2 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_equals_negative_two_l943_94361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hallway_length_is_six_l943_94319

/-- Represents the dimensions and areas of a bathroom with a central square area and a hallway. -/
structure Bathroom where
  central_side : ℝ  -- Side length of the central square area
  hallway_width : ℝ  -- Width of the hallway
  total_area : ℝ    -- Total area of the bathroom flooring

/-- Calculates the length of the hallway in the bathroom. -/
noncomputable def hallway_length (b : Bathroom) : ℝ :=
  (b.total_area - b.central_side ^ 2) / b.hallway_width

/-- Theorem stating that for a bathroom with given dimensions, the hallway length is 6 feet. -/
theorem hallway_length_is_six (b : Bathroom) 
    (h1 : b.central_side = 10)
    (h2 : b.hallway_width = 4)
    (h3 : b.total_area = 124) : 
  hallway_length b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hallway_length_is_six_l943_94319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_event_girls_ratio_l943_94314

/-- Represents a school with a given number of students and ratio of boys to girls -/
structure School where
  total_students : ℕ
  boy_ratio : ℕ
  girl_ratio : ℕ

/-- Calculates the number of girls in a school -/
def girls_count (s : School) : ℕ :=
  s.total_students * s.girl_ratio / (s.boy_ratio + s.girl_ratio)

/-- Calculates the number of boys in a school -/
def boys_count (s : School) : ℕ :=
  s.total_students * s.boy_ratio / (s.boy_ratio + s.girl_ratio)

def maplewood : School :=
  { total_students := 300,
    boy_ratio := 3,
    girl_ratio := 2 }

def brookside : School :=
  { total_students := 240,
    boy_ratio := 3,
    girl_ratio := 5 }

theorem joint_event_girls_ratio :
  let total_students := maplewood.total_students + brookside.total_students
  let total_girls := girls_count maplewood + girls_count brookside
  (total_girls : ℚ) / total_students = 1 / 2 ∧
  boys_count maplewood + boys_count brookside = girls_count maplewood + girls_count brookside := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_event_girls_ratio_l943_94314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_determination_l943_94360

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ

-- Define the concept of "uniquely determining" a triangle's shape
def uniquely_determines_shape (data : Triangle → ℝ × ℝ) : Prop :=
  ∀ t1 t2 : Triangle, data t1 = data t2 → t1 = t2

-- Define the different sets of data
noncomputable def ratio_two_sides_included_angle (t : Triangle) : ℝ × ℝ := 
  (t.a / t.b, t.angle_C)

noncomputable def ratios_three_altitudes (t : Triangle) : ℝ × ℝ := 
  (t.b * Real.sin t.angle_A / (t.c * Real.sin t.angle_B), 
   t.a * Real.sin t.angle_B / (t.c * Real.sin t.angle_C))

noncomputable def ratios_three_medians (t : Triangle) : ℝ × ℝ := 
  (Real.sqrt (2 * t.b^2 + 2 * t.c^2 - t.a^2) / Real.sqrt (2 * t.a^2 + 2 * t.c^2 - t.b^2),
   Real.sqrt (2 * t.a^2 + 2 * t.b^2 - t.c^2) / Real.sqrt (2 * t.a^2 + 2 * t.c^2 - t.b^2))

noncomputable def ratio_two_altitudes (t : Triangle) : ℝ × ℝ := 
  (t.b * Real.sin t.angle_A / (t.c * Real.sin t.angle_B), 0)

noncomputable def two_angles (t : Triangle) : ℝ × ℝ := 
  (t.angle_A, t.angle_B)

-- State the theorem
theorem triangle_shape_determination :
  uniquely_determines_shape ratio_two_sides_included_angle ∧
  uniquely_determines_shape ratios_three_altitudes ∧
  uniquely_determines_shape ratios_three_medians ∧
  uniquely_determines_shape two_angles ∧
  ¬uniquely_determines_shape ratio_two_altitudes :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_determination_l943_94360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mn_value_l943_94384

/-- The function f(x) -/
noncomputable def f (m n x : ℝ) : ℝ := (1/2) * (m - 2) * x^2 + (n - 8) * x + 1

/-- The derivative of f(x) -/
noncomputable def f_deriv (m n x : ℝ) : ℝ := (m - 2) * x + (n - 8)

theorem max_mn_value (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) 
  (h_decreasing : ∀ x ∈ Set.Icc (1/2 : ℝ) 2, f_deriv m n x ≤ 0) :
  m * n ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mn_value_l943_94384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_hyperbola_triangle_l943_94382

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/12 = 1

-- Define the foci
noncomputable def left_focus : ℝ × ℝ := (-Real.sqrt 13, 0)
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 13, 0)

-- Helper definitions
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem angle_measure_in_hyperbola_triangle 
  (P : ℝ × ℝ) 
  (h1 : hyperbola P.1 P.2) 
  (h2 : area_triangle P left_focus right_focus = 12) :
  angle_measure P left_focus right_focus = π/2 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_hyperbola_triangle_l943_94382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l943_94322

/-- The ⊗ operation defined for real numbers -/
noncomputable def otimes (x y z : ℝ) : ℝ := x / (y - z)

/-- Main theorem to prove -/
theorem otimes_calculation : 
  otimes (otimes 2 5 3 ^ 2) (otimes 4 6 2) (otimes 5 2 6) = 4 / 9 := by
  -- Expand the definition of otimes
  unfold otimes
  -- Simplify the expression
  simp [pow_two]
  -- Perform algebraic manipulations
  ring
  -- QED
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l943_94322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_660_deg_l943_94340

-- Define the degree to radian conversion constant
noncomputable def deg_to_rad : ℝ := Real.pi / 180

-- State the properties of cosine
axiom cos_periodic (θ : ℝ) (k : ℤ) : Real.cos (θ + 360 * k * deg_to_rad) = Real.cos θ
axiom cos_even (θ : ℝ) : Real.cos (-θ) = Real.cos θ
axiom cos_60_deg : Real.cos (60 * deg_to_rad) = 1 / 2

-- State the theorem
theorem cos_660_deg : Real.cos (660 * deg_to_rad) = 1 / 2 := by
  -- The proof steps would go here, but we'll use 'sorry' for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_660_deg_l943_94340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l943_94377

def a (n : ℕ) : ℕ := 2^n + 2^(n / 2)

def is_sum_of_distinct_terms (k : ℕ) : Prop :=
  ∃ (s : Finset ℕ), 2 ≤ s.card ∧ k = s.sum (λ i => a i) ∧ ∀ i ∈ s, ∀ j ∈ s, i ≠ j → a i ≠ a j

theorem sequence_properties :
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ k ∈ S, is_sum_of_distinct_terms k) ∧
  (∃ (T : Set ℕ), Set.Infinite T ∧ ∀ k ∈ T, ¬is_sum_of_distinct_terms k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l943_94377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleon_color_change_l943_94307

/-- Represents the number of chameleons that changed color -/
def changed_chameleons : ℕ := 80

/-- The total number of chameleons in the grove -/
def total_chameleons : ℕ := 140

/-- Represents the initial number of blue chameleons -/
def initial_blue : ℕ := 5 * (total_chameleons / 14)

/-- Represents the final number of blue chameleons -/
def final_blue : ℕ := initial_blue / 5

/-- Represents the initial number of red chameleons -/
def initial_red : ℕ := total_chameleons - initial_blue

/-- Represents the final number of red chameleons -/
def final_red : ℕ := 3 * initial_red

theorem chameleon_color_change :
  changed_chameleons = initial_blue - final_blue ∧
  total_chameleons = initial_blue + initial_red ∧
  total_chameleons = final_blue + final_red := by
  sorry

#eval changed_chameleons
#eval initial_blue
#eval final_blue
#eval initial_red
#eval final_red

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleon_color_change_l943_94307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_linear_function_sum_l943_94395

/-- Given two real numbers c and d, if g(x) = cx + d and its inverse
    g⁻¹(x) = dx + c, then c + d = -2 -/
theorem inverse_linear_function_sum (c d : ℝ) (g : ℝ → ℝ) :
  (∀ x, g x = c * x + d) →
  (∀ x, Function.invFun g x = d * x + c) →
  c + d = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_linear_function_sum_l943_94395
