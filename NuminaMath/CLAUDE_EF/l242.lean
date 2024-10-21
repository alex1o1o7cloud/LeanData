import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_order_isomorphism_between_Z_and_Q_l242_24229

-- Define the concept of order-isomorphism
def OrderIsomorphic (S T : Set ℝ) : Prop :=
  ∃ f : S → T,
    (∀ y : T, ∃ x : S, f x = y) ∧
    (∀ x₁ x₂ : S, x₁ < x₂ → f x₁ < f x₂)

-- Theorem statement
theorem no_order_isomorphism_between_Z_and_Q :
  ¬ OrderIsomorphic (Set.range (Int.cast : ℤ → ℝ)) (Set.range (Rat.cast : ℚ → ℝ)) := by
  sorry

#check no_order_isomorphism_between_Z_and_Q

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_order_isomorphism_between_Z_and_Q_l242_24229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l242_24252

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.cos ((2/3) * x + (7 * Real.pi) / 2)
noncomputable def g (x : ℝ) : ℝ := Real.sin (-2 * x)
noncomputable def h (x : ℝ) : ℝ := Real.sin (-2 * x + Real.pi / 4)

-- Theorem statement
theorem problem_solution :
  (∀ x, f x = -f (-x)) ∧  -- f is an odd function
  (∀ p q : Prop, (¬(p ∨ q) → (¬p ∧ ¬q))) ∧  -- logical implication
  (∀ x, g (x - Real.pi / 8) = h x) := by  -- graph shift
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l242_24252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_calculation_l242_24284

/-- Represents a vessel containing an alcohol mixture -/
structure Vessel where
  capacity : ℚ
  alcoholPercentage : ℚ

/-- Calculates the new alcohol concentration after mixing and diluting -/
def newConcentration (v1 v2 : Vessel) (finalVolume : ℚ) : ℚ :=
  let alcoholVolume1 := v1.capacity * v1.alcoholPercentage
  let alcoholVolume2 := v2.capacity * v2.alcoholPercentage
  let totalAlcohol := alcoholVolume1 + alcoholVolume2
  totalAlcohol / finalVolume

/-- Theorem stating that the new concentration matches the expected result -/
theorem concentration_calculation (v1 v2 : Vessel) (finalVolume : ℚ) :
  v1.capacity = 2 →
  v1.alcoholPercentage = 35 / 100 →
  v2.capacity = 6 →
  v2.alcoholPercentage = 1 / 2 →
  finalVolume = 10 →
  newConcentration v1 v2 finalVolume = 37 / 100 := by
  intro h1 h2 h3 h4 h5
  simp [newConcentration, h1, h2, h3, h4, h5]
  norm_num

#eval newConcentration ⟨2, 35/100⟩ ⟨6, 1/2⟩ 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_calculation_l242_24284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_sequence_l242_24253

def factorial_plus_n (n : ℕ) : ℕ := n.factorial + n

def sum_sequence : ℕ := (List.range 10).map (λ i => factorial_plus_n (i + 1)) |>.sum

theorem units_digit_of_sum_sequence :
  sum_sequence % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_sequence_l242_24253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_range_l242_24247

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3 / Real.sqrt (x - 2)

-- Theorem stating the range of x for which f is meaningful
theorem f_meaningful_range :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_range_l242_24247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_conics_and_intersection_angle_l242_24278

-- Define the family of conics C
def C (a : ℝ) (x y : ℝ) : Prop :=
  (2 * y + x)^2 = a * (y + x)

-- Define the orthogonal family of conics C'
def C' (b : ℝ) (x y : ℝ) : Prop :=
  y = b * x^2 - 3 * x

-- Define the angle between two curves at a point
noncomputable def angle_between_curves (f g : ℝ → ℝ) (x : ℝ) : ℝ :=
  Real.arctan ((deriv f x - deriv g x) / (1 + deriv f x * deriv g x))

theorem orthogonal_conics_and_intersection_angle :
  ∃ (f g : ℝ → ℝ),
    (∀ x, C (f 0) x (f x)) ∧
    (∀ x, C' (g 0) x (g x)) ∧
    (∀ a b, C a 0 0 → C' b 0 0 → angle_between_curves f g 0 = Real.arctan (1/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_conics_and_intersection_angle_l242_24278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_root_f_has_root_in_interval_quadratic_one_solution_in_interval_l242_24295

-- Part 1
theorem existence_of_root (f : ℝ → ℝ) (h : Continuous f) :
  f (-1) ≤ 0 ∧ f 0 ≥ 0 → ∃ x ∈ Set.Icc (-1) 0, f x = 0 :=
by sorry

noncomputable def f (x : ℝ) : ℝ := 2^x - x^2

theorem f_has_root_in_interval :
  ∃ x ∈ Set.Icc (-1) 0, f x = 0 :=
by sorry

-- Part 2
theorem quadratic_one_solution_in_interval (a : ℝ) :
  (∃! x, x ∈ Set.Ioo 0 1 ∧ a * x^2 - x - 1 = 0) ↔ a > 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_root_f_has_root_in_interval_quadratic_one_solution_in_interval_l242_24295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shift_l242_24239

-- Define the original function g
noncomputable def g (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 1 then -x
  else if 1 < x ∧ x ≤ 3 then Real.sqrt (4 - (x - 1)^2)
  else if 3 < x ∧ x ≤ 5 then x - 3
  else 0  -- undefined outside the given intervals

-- Define the shifted function g'
noncomputable def g' (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 4 then -(x - 3)
  else if 4 < x ∧ x ≤ 6 then Real.sqrt (4 - (x - 4)^2)
  else if 6 < x ∧ x ≤ 8 then x - 6
  else 0  -- undefined outside the given intervals

-- Theorem statement
theorem g_shift (x : ℝ) : g (x - 3) = g' x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shift_l242_24239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_height_example_l242_24240

/-- The height of a parallelogram given its area and base -/
noncomputable def parallelogram_height (area : ℝ) (base : ℝ) : ℝ :=
  area / base

/-- Theorem: The height of a parallelogram with area 240 cm² and base 24 cm is 10 cm -/
theorem parallelogram_height_example : parallelogram_height 240 24 = 10 := by
  -- Unfold the definition of parallelogram_height
  unfold parallelogram_height
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_height_example_l242_24240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_x_coordinates_of_A_l242_24246

/-- Calculates the area of a triangle given three points in 2D space. -/
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))

/-- Given two triangles ABC and ADE with specified areas and coordinates for points B, C, D, and E,
    prove that the sum of all possible x-coordinates of point A is 606. -/
theorem sum_of_possible_x_coordinates_of_A : 
  ∃ (A : ℝ × ℝ),
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (229, 0)
  let D : ℝ × ℝ := (680, 380)
  let E : ℝ × ℝ := (695, 388)
  let area_ABC : ℝ := 3011
  let area_ADE : ℝ := 9033
  (area_triangle A B C = area_ABC ∧ area_triangle A D E = area_ADE) ∧
  (∃ (x₁ x₂ : ℝ), 
    (A.1 = x₁ ∨ A.1 = x₂) ∧ 
    x₁ + x₂ = 606) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_x_coordinates_of_A_l242_24246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l242_24279

/-- The function f(x) as defined in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt 2 * Real.sin (ω * x) * Real.cos (ω * x) + 
  Real.sqrt 2 * (Real.cos (ω * x))^2 - Real.sqrt 2 / 2

/-- The theorem stating the range of ω -/
theorem omega_range (ω : ℝ) : 
  (ω > 0) → 
  (∀ x ∈ Set.Ioo (π/2) π, StrictMonoOn (fun x => -(f ω x)) (Set.Ioo (π/2) π)) → 
  ω ∈ Set.Icc (1/4) (5/8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l242_24279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_arrangement_property_l242_24271

theorem ball_arrangement_property (arrangement : List Bool) : 
  (arrangement.count true = 5) → 
  (arrangement.count false = 4) →
  ∃ (i : Nat), i < arrangement.length ∧ arrangement.get ⟨i, by sorry⟩ = true ∧ 
    (arrangement.drop (i + 1)).count true = (arrangement.drop (i + 1)).count false := by
  sorry

-- In this statement:
-- true represents a black ball
-- false represents a white ball
-- The theorem states that for any list of booleans (arrangement) where:
--   1. The count of true (black balls) is 5
--   2. The count of false (white balls) is 4
-- There exists an index i where:
--   1. i is less than the length of the arrangement
--   2. The ball at index i is black (true)
--   3. The number of black balls (true) after index i
--      is equal to the number of white balls (false) after index i

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_arrangement_property_l242_24271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maintenance_check_increase_l242_24218

/-- Calculates the percent increase between two values -/
noncomputable def percent_increase (original new : ℝ) : ℝ :=
  (new - original) / original * 100

/-- The original time between maintenance checks in days -/
def original_time : ℝ := 20

/-- The new time between maintenance checks in days -/
def new_time : ℝ := 25

/-- Theorem stating that the percent increase in maintenance check interval is 25% -/
theorem maintenance_check_increase :
  percent_increase original_time new_time = 25 := by
  -- Unfold the definition of percent_increase
  unfold percent_increase
  -- Simplify the expression
  simp [original_time, new_time]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maintenance_check_increase_l242_24218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_composite_function_l242_24270

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then |x + 1|
  else -x^2 + 3*x

-- Define the composite function g(x) = f(f(x) - 1)
noncomputable def g (x : ℝ) : ℝ := f (f x - 1)

-- Theorem statement
theorem zeros_of_composite_function :
  ∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0) ∧
    (∀ x : ℝ, g x = 0 → x = a ∨ x = b ∨ x = c) := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_composite_function_l242_24270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_transformation_graph_transformation_l242_24293

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

theorem derivative_transformation (x : ℝ) :
  deriv f x = 2 * Real.sin (2 * (x + Real.pi / 4)) :=
by sorry

theorem graph_transformation (x : ℝ) :
  deriv f x = 2 * f (x + Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_transformation_graph_transformation_l242_24293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_is_three_l242_24275

-- Define the cost of a bunch of bananas
def banana_cost : ℚ := 3

-- Define Tony's purchase
def tony_apples (apple_cost : ℚ) (apple_count : ℚ) : ℚ := apple_cost * apple_count
def tony_total : ℚ := 7

-- Define Arnold's purchase
def arnold_apples (apple_cost : ℚ) (tony_apple_count : ℚ) : ℚ := apple_cost * (tony_apple_count / 2)
def arnold_total : ℚ := 5

-- Theorem statement
theorem banana_cost_is_three :
  (∃ (apple_cost : ℚ) (tony_apple_count : ℚ),
    tony_total = tony_apples apple_cost tony_apple_count + banana_cost ∧
    arnold_total = arnold_apples apple_cost tony_apple_count + banana_cost) →
  banana_cost = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_is_three_l242_24275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BEIH_l242_24250

noncomputable section

-- Define the points
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (3, 0)
def D : ℝ × ℝ := (3, 3)
def E : ℝ × ℝ := (0, 1.5)
def F : ℝ × ℝ := (1.5, 0)

-- Define the lines
def line_AF (x : ℝ) : ℝ := -2 * x + 3
def line_DE (x : ℝ) : ℝ := 0.5 * x + 1.5
def line_BD (x : ℝ) : ℝ := x

-- Define intersection points
def I : ℝ × ℝ := (3/5, 9/5)
def H : ℝ × ℝ := (1, 1)

-- Define the quadrilateral area function
def quad_area (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  let (x4, y4) := p4
  0.5 * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

theorem area_of_BEIH :
  quad_area B E I H = 6/5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BEIH_l242_24250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_equals_991_l242_24259

def product_with_k_eights (k : ℕ) : ℕ :=
  8 * (Finset.sum (Finset.range k) (λ i => 8 * 10^i))

def sum_of_digits : ℕ → ℕ
| n => if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem k_equals_991 :
  ∃ k : ℕ, k > 0 ∧ sum_of_digits (product_with_k_eights k) = 1000 ∧ k = 991 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_equals_991_l242_24259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_good_coloring_l242_24262

/-- A set of points on a circle. -/
structure CirclePoints where
  n : ℕ
  E : Finset ℕ
  h1 : n ≥ 3
  h2 : E.card = 2 * n - 1

/-- A coloring of points on a circle. -/
structure Coloring (cp : CirclePoints) where
  black : Finset ℕ
  h : black ⊆ cp.E

/-- A predicate indicating whether a coloring is "good". -/
def is_good_coloring (cp : CirclePoints) (c : Coloring cp) : Prop :=
  ∃ (p q : ℕ), p ∈ c.black ∧ q ∈ c.black ∧
    ∃ (arc : Finset ℕ), arc ⊆ cp.E ∧ arc.card = cp.n ∧
    (∀ x ∈ arc, x ≠ p ∧ x ≠ q)

/-- The theorem stating the smallest number of black points for a good coloring. -/
theorem smallest_good_coloring (cp : CirclePoints) :
  ∃ (k : ℕ), k = cp.n ∧
    (∀ (c : Coloring cp), c.black.card = k → is_good_coloring cp c) ∧
    (∀ (k' : ℕ), k' < k →
      ∃ (c : Coloring cp), c.black.card = k' ∧ ¬is_good_coloring cp c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_good_coloring_l242_24262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bah_yah_equivalence_l242_24244

-- Define our types as aliases for ℚ
abbrev bah := ℚ
abbrev rah := ℚ
abbrev yah := ℚ

-- Exchange rates
def bah_to_rah_rate : ℚ := 36 / 20
def rah_to_yah_rate : ℚ := 20 / 12

-- Conversion functions
def bah_to_rah (x : bah) : rah := x * bah_to_rah_rate
def rah_to_yah (x : rah) : yah := x * rah_to_yah_rate
def yah_to_rah (x : yah) : rah := x / rah_to_yah_rate
def rah_to_bah (x : rah) : bah := x / bah_to_rah_rate

theorem bah_yah_equivalence : 
  rah_to_bah (yah_to_rah 1200) = 400 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bah_yah_equivalence_l242_24244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_power_function_passes_through_point_power_function_at_half_l242_24241

/-- A power function that passes through the point (2, 1/4) -/
noncomputable def f (x : ℝ) : ℝ :=
  x^(Real.log (1/4) / Real.log 2)

theorem power_function_value (x : ℝ) (hx : x > 0) :
  f x = x^(Real.log (1/4) / Real.log 2) :=
by sorry

theorem power_function_passes_through_point :
  f 2 = 1/4 :=
by sorry

theorem power_function_at_half : f (1/2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_power_function_passes_through_point_power_function_at_half_l242_24241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_minimum_l242_24248

theorem quadratic_inequality_minimum (a b : ℝ) (h1 : a > b)
  (h2 : ∀ x : ℝ, a * x^2 + 2*x + b ≥ 0)
  (h3 : ∃ x_0 : ℝ, a * x_0^2 + 2*x_0 + b = 0) :
  (∀ a b : ℝ, 2*a^2 + b^2 ≥ 2*Real.sqrt 2) ∧ (∃ a b : ℝ, 2*a^2 + b^2 = 2*Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_minimum_l242_24248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harolds_finances_l242_24297

noncomputable def monthly_income : ℝ := 2500
noncomputable def rent : ℝ := 700
noncomputable def car_payment : ℝ := 300
noncomputable def groceries : ℝ := 50

noncomputable def utilities : ℝ := car_payment / 2

noncomputable def total_expenses : ℝ := rent + car_payment + utilities + groceries

noncomputable def remaining_money : ℝ := monthly_income - total_expenses

noncomputable def retirement_contribution : ℝ := remaining_money / 2

noncomputable def final_remaining : ℝ := remaining_money - retirement_contribution

theorem harolds_finances : final_remaining = 650 := by
  -- Expand definitions
  unfold final_remaining retirement_contribution remaining_money total_expenses utilities
  -- Perform algebraic simplifications
  simp [monthly_income, rent, car_payment, groceries]
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harolds_finances_l242_24297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_contains_all_numbers_l242_24231

/-- Represents a sector in the circle -/
structure Sector where
  color : Bool  -- true for blue, false for red
  number : Nat

/-- Represents the circle with 2n sectors -/
def Circle (n : Nat) := Vector Sector (2 * n)

/-- Predicate to check if a circle is valid according to the problem description -/
def is_valid_circle (c : Circle n) : Prop :=
  (c.toList.filter (·.color)).length = n ∧
  (c.toList.filter (¬·.color)).length = n ∧
  (List.zip c.toList (c.toList.rotate 1)).all (λ (s1, s2) => s1.color ≠ s2.color) ∧
  (c.toList.filter (·.color)).map (·.number) = List.range n ∧
  (c.toList.filter (¬·.color)).reverse.map (·.number) = List.range n

/-- Predicate to check if a list of sectors contains all numbers from 1 to n -/
def contains_all_numbers (l : List Sector) (n : Nat) : Prop :=
  (l.map (·.number)).toFinset = Finset.range n

/-- Main theorem: There exists a semicircle containing all numbers from 1 to n -/
theorem semicircle_contains_all_numbers {n : Nat} (c : Circle n) 
  (h : is_valid_circle c) : 
  ∃ (i : Fin (2 * n)), contains_all_numbers (c.toList.take n) n ∨ 
                       contains_all_numbers ((c.toList.drop i.val).take n) n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_contains_all_numbers_l242_24231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_average_score_l242_24228

/-- Represents the average score of a cricketer after a number of innings -/
def AverageScore (totalRuns : ℚ) (innings : ℕ) : ℚ :=
  totalRuns / innings

theorem cricketer_average_score 
  (initialAverage : ℚ) 
  (runsIn19thInning : ℕ) 
  (averageIncrease : ℚ) :
  runsIn19thInning = 96 →
  averageIncrease = 4 →
  AverageScore (18 * initialAverage + runsIn19thInning) 19 = 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_average_score_l242_24228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_period_implies_omega_one_l242_24238

/-- The function f(x) defined in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x) * Real.cos (Real.pi / 2 - ω * x)

/-- The theorem statement -/
theorem minimum_period_implies_omega_one (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ T : ℝ, T > 0 → (∀ x : ℝ, f ω (x + T) = f ω x) → T ≥ Real.pi) 
  (h3 : ∀ x : ℝ, f ω (x + Real.pi) = f ω x) : ω = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_period_implies_omega_one_l242_24238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_max_value_l242_24208

-- Define the set (0, 2]
def OpenZeroTwoClosed : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the expression A
noncomputable def A (x y z : ℝ) : ℝ := 
  ((x^3 - 6) * (x + 6)^(1/3) + (y^3 - 6) * (y + 6)^(1/3) + (z^3 - 6) * (z + 6)^(1/3)) / (x^2 + y^2 + z^2)

-- State the theorem
theorem A_max_value (x y z : ℝ) (hx : x ∈ OpenZeroTwoClosed) (hy : y ∈ OpenZeroTwoClosed) (hz : z ∈ OpenZeroTwoClosed) :
  A x y z ≤ 1 ∧ ∃ (a b c : ℝ), a ∈ OpenZeroTwoClosed ∧ b ∈ OpenZeroTwoClosed ∧ c ∈ OpenZeroTwoClosed ∧ A a b c = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_max_value_l242_24208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l242_24243

-- Define the rectangle's dimensions
noncomputable def rectangle_length : ℝ := 7
noncomputable def rectangle_width : ℝ := 5

-- Define the perimeter of the rectangle
noncomputable def rectangle_perimeter : ℝ := 2 * (rectangle_length + rectangle_width)

-- Define the side length of the square
noncomputable def square_side : ℝ := rectangle_perimeter / 4

-- Theorem statement
theorem square_side_length :
  square_side = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l242_24243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_win_probability_l242_24286

/-- Represents the game state -/
structure GameState where
  n : ℕ  -- Size of the grid

/-- Calculates the probability of the escaping player winning -/
def escape_probability (state : GameState) : ℚ :=
  1 - (1/2)^state.n

/-- Theorem stating the probability of Rachel winning -/
theorem rachel_win_probability :
  let initial_state : GameState := ⟨min 6 8⟩
  escape_probability initial_state = 63/64 := by
  sorry

#eval escape_probability ⟨6⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_win_probability_l242_24286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_total_l242_24294

theorem election_votes_total (total_votes : ℕ) : 
  (0.005 * (total_votes : ℝ) = Geoff's_votes) →
  (0.505 * (total_votes : ℝ) = winning_threshold) →
  (winning_threshold - Geoff's_votes = 3000) →
  total_votes = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_total_l242_24294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_18_l242_24272

/-- The area of a quadrilateral given its vertices -/
noncomputable def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

/-- Theorem: The area of the quadrilateral with vertices (2, 1), (1, 6), (5, 5), and (7, 2) is 18 -/
theorem quadrilateral_area_is_18 :
  quadrilateralArea (2, 1) (1, 6) (5, 5) (7, 2) = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_18_l242_24272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_second_quadrant_l242_24209

theorem tan_half_angle_second_quadrant (θ : Real) : 
  π / 2 < θ ∧ θ < π → Real.tan (θ / 2) > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_second_quadrant_l242_24209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_elements_l242_24267

def A : Set ℕ := {x | x > 0 ∧ (∃ n : ℕ, 6 = n * (6 - x))}

theorem set_A_elements : A = {3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_elements_l242_24267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p_and_not_q_l242_24283

-- Define the function y = 2 - a^(x+1)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 - a^(x + 1)

-- Define an arbitrary function g
noncomputable def g : ℝ → ℝ := sorry

theorem not_p_and_not_q :
  (∃ a : ℝ, f a 1 ≠ 2) ∧
  (∃ h : ℝ → ℝ, (∀ x, h (x - 1) = h (1 - x)) ∧
                ¬(∀ x, g (x + 1) = g (1 - x))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p_and_not_q_l242_24283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_theorem_l242_24273

/-- The radius of the large circular cookie dough -/
def R : ℝ := 3

/-- The radius of each small circular cookie -/
def r : ℝ := 1

/-- The number of small cookies cut from the large cookie dough -/
def n : ℕ := 7

/-- The radius of the new cookie formed from the remaining dough -/
noncomputable def new_radius : ℝ := Real.sqrt 2

/-- Theorem stating that the area of the remaining dough equals the area of the new cookie -/
theorem cookie_theorem : 
  π * R^2 - n * π * r^2 = π * new_radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_theorem_l242_24273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_14_pow_14_equals_1_l242_24280

def a : ℕ → ℕ
  | 0 => 0  -- Add a case for 0 to cover all natural numbers
  | 1 => 11^11
  | 2 => 12^12
  | 3 => 13^13
  | n + 4 => Int.natAbs (a (n + 3) - a (n + 2)) + Int.natAbs (a (n + 2) - a (n + 1))

theorem a_14_pow_14_equals_1 : a (14^14) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_14_pow_14_equals_1_l242_24280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_projection_l242_24204

/-- The side length of an equilateral triangle whose projection has sides 2, 3, and 2√3 -/
noncomputable def equilateral_triangle_side : ℝ := Real.sqrt 13

theorem equilateral_triangle_projection (ABC : Fin 3 → ℝ × ℝ × ℝ) 
  (α : Set (ℝ × ℝ × ℝ)) (proj : (ℝ × ℝ × ℝ) → (ℝ × ℝ)) :
  (∀ i j, i ≠ j → ‖ABC i - ABC j‖ = equilateral_triangle_side) →
  (∃ f : Fin 3 ↪ Fin 3, 
    ‖proj (ABC (f 0)) - proj (ABC (f 1))‖ = 2 ∧
    ‖proj (ABC (f 1)) - proj (ABC (f 2))‖ = 3 ∧
    ‖proj (ABC (f 2)) - proj (ABC (f 0))‖ = 2 * Real.sqrt 3) →
  ∀ i j, i ≠ j → ‖ABC i - ABC j‖ = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_projection_l242_24204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_town_growth_approximately_9_09_percent_l242_24212

/-- Represents the population of a town over time -/
structure TownPopulation where
  initial : Nat  -- Population in 1991
  middle : Nat   -- Population in 2006
  final : Nat    -- Population in 2016

/-- Conditions for the town population -/
def validPopulation (p : TownPopulation) : Prop :=
  ∃ (x y z : Nat),
    p.initial = x^2 ∧
    p.middle = p.initial + 100 ∧
    p.middle = y^2 + 9 ∧
    p.final = p.middle + 175 ∧
    p.final = z^2

/-- Calculate the percentage growth -/
noncomputable def percentGrowth (p : TownPopulation) : ℝ :=
  (p.final - p.initial : ℝ) / p.initial * 100

/-- Theorem stating that the percentage growth is approximately 9.09% -/
theorem town_growth_approximately_9_09_percent (p : TownPopulation) 
  (h : validPopulation p) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |percentGrowth p - 9.09| < ε := by
  sorry

#eval "Theorem statement completed."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_town_growth_approximately_9_09_percent_l242_24212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_concentration_correct_verify_specific_instance_l242_24232

/-- Calculates the final concentration of a solution after repeated dilution -/
noncomputable def final_concentration (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ) : ℝ :=
  C₀ * (1 - v / V) ^ n

/-- Theorem stating the correctness of the final_concentration function -/
theorem final_concentration_correct
  (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ)
  (hV : V > 0)
  (hC₀ : 0 ≤ C₀ ∧ C₀ ≤ 1)
  (hv : 0 < v ∧ v < V) :
  final_concentration V C₀ v n = C₀ * (1 - v / V) ^ n :=
by sorry

/-- Verification of the specific problem instance -/
theorem verify_specific_instance :
  ∃ ε > 0, |final_concentration 0.5 0.4 0.05 5 - 0.236| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_concentration_correct_verify_specific_instance_l242_24232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_perp_necessary_not_sufficient_l242_24216

structure Plane where
  -- Placeholder for plane properties

structure Line where
  -- Placeholder for line properties

def perpendicular (a b : Plane) : Prop :=
  sorry -- Definition of perpendicular planes

def perpendicular_line_plane (l : Line) (p : Plane) : Prop :=
  sorry -- Definition of a line perpendicular to a plane

def line_in_plane (l : Line) (p : Plane) : Prop :=
  sorry -- Definition of a line being in a plane

theorem plane_perp_necessary_not_sufficient 
  (α β : Plane) (m : Line) 
  (h_diff : α ≠ β) 
  (h_in : line_in_plane m α) :
  (∀ m, line_in_plane m α → perpendicular_line_plane m β → perpendicular α β) ∧
  (∃ m, line_in_plane m α ∧ perpendicular α β ∧ ¬perpendicular_line_plane m β) :=
by
  sorry -- Proof to be completed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_perp_necessary_not_sufficient_l242_24216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wendi_chickens_count_l242_24222

theorem wendi_chickens_count : ℕ := by
  let initial_chickens : ℕ := 4
  let doubled_chickens : ℕ := initial_chickens * 2
  let after_dog_ate : ℕ := doubled_chickens - 1
  let additional_chickens : ℕ := 10 - 4
  let final_count : ℕ := after_dog_ate + additional_chickens
  have h : final_count = 13 := by
    -- Proof steps would go here
    sorry
  exact final_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wendi_chickens_count_l242_24222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_function_l242_24223

noncomputable def P : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}
noncomputable def N : Set ℝ := {y | 0 ≤ y ∧ y ≤ 2}

noncomputable def f (x : ℝ) : ℝ := (2 / 3) * x

theorem f_not_function : ¬(∀ x ∈ P, f x ∈ N) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_function_l242_24223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_wrapping_capacity_l242_24236

/-- Given the conditions of Edmund's gift wrapping shop, prove the number of gift boxes he can wrap in 3 days. -/
theorem gift_wrapping_capacity 
  (wrapper_per_box : ℚ) 
  (wrapper_per_day : ℚ) 
  (days : ℕ) 
  (h1 : wrapper_per_box = 18) 
  (h2 : wrapper_per_day = 90) 
  (h3 : days = 3) : 
  ⌊(wrapper_per_day / wrapper_per_box) * days⌋ = 15 := by
  sorry

#eval Int.floor ((90 / 18 : ℚ) * 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_wrapping_capacity_l242_24236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_of_special_arithmetic_sequence_l242_24298

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

/-- The 5th term of an arithmetic sequence with a₁ = 1 and S₄ = 16 is 9 -/
theorem fifth_term_of_special_arithmetic_sequence :
  ∃ d : ℝ, arithmetic_sequence 1 d 5 = 9 ∧ arithmetic_sum 1 d 4 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_of_special_arithmetic_sequence_l242_24298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l242_24242

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/3) * Real.sin (x + Real.pi/2)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l242_24242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l242_24255

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+1) + 3

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l242_24255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_tan_squared_equals_twelve_l242_24268

/-- A regular triangular prism with all edges of equal length -/
structure RegularTriangularPrism where
  -- All edges have equal length
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- Plane intersecting the lateral faces of the prism -/
structure IntersectingPlane (prism : RegularTriangularPrism) where
  -- Angles between intersection lines and base plane
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem: Sum of squared tangents of intersection angles equals 12 -/
theorem sum_tan_squared_equals_twelve (prism : RegularTriangularPrism) 
  (plane : IntersectingPlane prism) : 
  Real.tan plane.α ^ 2 + Real.tan plane.β ^ 2 + Real.tan plane.γ ^ 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_tan_squared_equals_twelve_l242_24268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_centroid_distance_l242_24202

noncomputable def RightTriangle (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0

noncomputable def Incenter (a b : ℝ) : ℝ × ℝ :=
  let r := (a * b) / (a + b + Real.sqrt (a^2 + b^2))
  (r, r)

noncomputable def Centroid (a b : ℝ) : ℝ × ℝ :=
  (a / 3, b / 3)

noncomputable def Distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem incenter_centroid_distance (a b : ℝ) 
  (h : RightTriangle a b) (ha : a = 9) (hb : b = 12) : 
  Distance (Incenter a b) (Centroid a b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_centroid_distance_l242_24202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_price_calculation_l242_24251

theorem shoe_price_calculation (initial_price increase_percentage discount_percentage : ℝ) : 
  initial_price = 50 → 
  increase_percentage = 20 → 
  discount_percentage = 15 → 
  (initial_price * (1 + increase_percentage / 100)) * (1 - discount_percentage / 100) = 51 := by
  intros h1 h2 h3
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_price_calculation_l242_24251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_rectangle_l242_24291

/-- A rectangle with one side double the other and area at least 800 sq ft has minimum perimeter when 20 ft by 40 ft -/
theorem min_perimeter_rectangle (x : ℝ) (h1 : x > 0) (h2 : 2 * x^2 ≥ 800) : 
  2 * (x + 2*x) ≥ 2 * (20 + 40) := by
  sorry

#check min_perimeter_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_rectangle_l242_24291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_ten_points_exist_l242_24266

/-- A point in the xy-plane with rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- Predicate to check if a point satisfies the given conditions -/
def satisfiesConditions (p : RationalPoint) : Prop :=
  p.x > 0 ∧ p.y > 0 ∧ p.x + p.y ≤ 4 ∧ p.x ≥ 1

/-- Theorem stating that there exist at least 10 distinct points satisfying the conditions -/
theorem at_least_ten_points_exist :
  ∃ (points : Finset RationalPoint),
    (points.card ≥ 10) ∧
    (∀ p, p ∈ points → satisfiesConditions p) ∧
    (∀ p q, p ∈ points → q ∈ points → p ≠ q → p.x ≠ q.x ∨ p.y ≠ q.y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_ten_points_exist_l242_24266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_l242_24281

/-- A square centered at the origin with side length 6 -/
structure Square where
  side : ℝ
  center : ℝ × ℝ
  h_side : side = 6
  h_center : center = (0, 0)

/-- A line passing through the center of the square -/
structure CuttingLine where
  slope : ℝ
  h_slope : -1 ≤ slope ∧ slope ≤ 1

/-- The centroid of a part formed by the cutting line -/
noncomputable def centroid (s : Square) (l : CuttingLine) : ℝ × ℝ := sorry

/-- The locus of centroids is a parabola -/
theorem centroid_locus (s : Square) :
  ∀ (l : CuttingLine), 
    let (x, y) := centroid s l
    y = (x^2)/2 - 3/2 ∧ -1 ≤ x ∧ x ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_l242_24281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldbach_difference_140_l242_24274

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem goldbach_difference_140 :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p + q = 140 ∧ 
    p ≠ q ∧ 
    (p < 20 ∨ q < 20) ∧ 
    ∀ (r s : ℕ), 
      is_prime r → 
      is_prime s → 
      r + s = 140 → 
      r ≠ s → 
      (r < 20 ∨ s < 20) → 
      (s - r) ≤ (q - p) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldbach_difference_140_l242_24274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l242_24214

theorem car_average_speed : 
  let d1 : ℝ := 30 -- distance in km for segment 1
  let v1 : ℝ := 35 -- speed in kph for segment 1
  let d2 : ℝ := 40 -- distance in km for segment 2
  let v2 : ℝ := 55 -- speed in kph for segment 2
  let t3 : ℝ := 0.5 -- time in hours for segment 3
  let v3 : ℝ := 75 -- speed in kph for segment 3
  let t4 : ℝ := 1/6 -- time in hours for segment 4
  let v4 : ℝ := 42 -- speed in kph for segment 4
  
  let total_distance : ℝ := d1 + d2 + v3 * t3 + v4 * t4
  let total_time : ℝ := d1 / v1 + d2 / v2 + t3 + t4
  
  let average_speed : ℝ := total_distance / total_time
  
  ∃ ε : ℝ, ε > 0 ∧ |average_speed - 51| < ε :=
by
  sorry

#check car_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l242_24214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biology_exam_students_l242_24235

theorem biology_exam_students : ∃ T : ℕ,
  -- Condition 1: 2/5 of students scored 100%
  T * 2 = T * 5 * 2 / 5 ∧
  -- Condition 3: 50% of the remaining students scored over 80%
  (T - T * 2 / 5) / 2 = T * 3 / 10 ∧
  -- Condition 4: 24 students failed the exam
  T * 3 / 10 = 24 ∧
  -- Prove that T = 80
  T = 80 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_biology_exam_students_l242_24235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_l242_24210

/-- Given two spheres inscribed in a dihedral angle, this theorem proves
    that the cosine of the dihedral angle is 0.04 under specific conditions. -/
theorem dihedral_angle_cosine (r θ : ℝ) : 
  r > 0 → -- The radius of the smaller sphere is positive
  let R := 4 * r -- The radius of the larger sphere is 4 times the smaller one
  let d := r + R -- The distance between the centers is the sum of the radii
  let α := π / 3 -- The angle between the line connecting centers and the edge is 60°
  let d_x := d * Real.cos α -- Projection of d onto the edge plane
  d_x = 2 * r * Real.cos (θ / 2) → -- Relationship between d_x, r, and θ
  Real.cos θ = 0.04 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_l242_24210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l242_24200

/-- The length of a bridge given train parameters -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 100)
  (h2 : train_speed_kmh = 65)
  (h3 : crossing_time = 13.568145317605362) :
  ∃ (bridge_length : ℝ),
    bridge_length = train_speed_kmh * (1000 / 3600) * crossing_time - train_length ∧
    abs (bridge_length - 145) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l242_24200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equilateral_triangle_l242_24205

-- Define the line
def line (x y a : ℝ) : Prop := x - y - a = 0

-- Define the circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define an equilateral triangle
def is_equilateral_triangle (A B : ℝ × ℝ) : Prop :=
  let O := origin
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - O.1)^2 + (B.2 - O.2)^2 ∧
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2

theorem intersection_equilateral_triangle (a : ℝ) :
  (∃ A B : ℝ × ℝ, 
    line A.1 A.2 a ∧ 
    line B.1 B.2 a ∧ 
    unit_circle A.1 A.2 ∧ 
    unit_circle B.1 B.2 ∧ 
    is_equilateral_triangle A B) →
  a = Real.sqrt 6 ∨ a = -Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equilateral_triangle_l242_24205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_not_always_contains_loop_l242_24226

/-- Represents the types of structures in algorithms -/
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents an algorithm as a set of structures -/
def Algorithm := Set AlgorithmStructure

/-- Every algorithm must contain a sequential structure -/
axiom sequential_in_all (a : Algorithm) : a AlgorithmStructure.Sequential

/-- A loop structure must contain a conditional structure -/
axiom loop_contains_conditional (a : Algorithm) :
  a AlgorithmStructure.Loop → a AlgorithmStructure.Conditional

/-- Theorem: It is not true that a conditional structure must contain a loop structure -/
theorem conditional_not_always_contains_loop :
  ∃ (a : Algorithm), a AlgorithmStructure.Conditional ∧ ¬a AlgorithmStructure.Loop :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_not_always_contains_loop_l242_24226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_exponent_relation_l242_24203

theorem monomial_sum_exponent_relation (m n : ℤ) : 
  (∃ (a : ℝ) (p q : ℤ), -2 * X^m * Y^2 + (1/2) * X^3 * Y^n = a * X^p * Y^q) → 
  m - 2*n = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_exponent_relation_l242_24203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_divisible_by_13_l242_24230

def f : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | (v + 2) => 4^(v+2) * f (v+1) - 16^(v+1) * f v + v * 2^(v^2)

theorem f_divisible_by_13 : ∀ v ∈ ({1989, 1990, 1991} : Set ℕ), 13 ∣ f v := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_divisible_by_13_l242_24230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l242_24260

/-- The molecular weight of a compound given its total weight and number of moles -/
noncomputable def molecular_weight (total_weight : ℝ) (num_moles : ℝ) : ℝ :=
  total_weight / num_moles

/-- Theorem stating the molecular weight of a compound is 68 grams/mole -/
theorem compound_molecular_weight :
  let total_weight : ℝ := 204
  let num_moles : ℝ := 3
  molecular_weight total_weight num_moles = 68 := by
  -- Unfold the definition of molecular_weight
  unfold molecular_weight
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l242_24260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_volume_is_five_fourths_l242_24220

/-- A cube with side length 1 -/
def UnitCube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 1}

/-- The space diagonal of a unit cube -/
def SpaceDiagonal : Fin 3 → ℝ :=
  λ _ ↦ 1

/-- Reflection of a point across the space diagonal -/
def ReflectAcrossDiagonal (p : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i ↦ SpaceDiagonal i - p i

/-- The reflected cube -/
def ReflectedCube : Set (Fin 3 → ℝ) :=
  {p | ∃ q ∈ UnitCube, p = ReflectAcrossDiagonal q}

/-- The volume of a set in ℝ³ -/
noncomputable def Volume (s : Set (Fin 3 → ℝ)) : ℝ := sorry

/-- The theorem stating the combined volume of the original and reflected cube -/
theorem combined_volume_is_five_fourths :
  Volume (UnitCube ∪ ReflectedCube) = 5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_volume_is_five_fourths_l242_24220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_and_parametric_representations_l242_24257

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = Real.cos θ

-- Define the parametric equations
def parametric_equations (x y t : ℝ) : Prop := x = -1 - t ∧ y = 2 + 3*t

-- State the theorem
theorem polar_and_parametric_representations :
  (∃ (a b r : ℝ), ∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2 ↔ ∃ (ρ θ : ℝ), polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∃ (A B C : ℝ), (A ≠ 0 ∨ B ≠ 0) ∧ (∀ (x y : ℝ), A*x + B*y + C = 0 ↔ ∃ (t : ℝ), parametric_equations x y t)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_and_parametric_representations_l242_24257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_l242_24256

noncomputable def f (x : ℝ) : ℝ := |Real.sin (2 * x) - Real.cos (2 * x)|

theorem min_positive_period_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_l242_24256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_PS_distance_l242_24245

/-- Represents a village on a straight road -/
structure Village where
  position : ℝ

/-- The setup of villages P, Q, R, S on a straight road -/
structure VillageSetup where
  P : Village
  Q : Village
  R : Village
  S : Village
  PR_distance : |P.position - R.position| = 75
  QS_distance : |Q.position - S.position| = 45
  QR_distance : |Q.position - R.position| = 20

/-- Theorem stating that given the setup, the distance between P and S cannot be 80 km -/
theorem impossible_PS_distance (setup : VillageSetup) : 
  |setup.P.position - setup.S.position| ≠ 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_PS_distance_l242_24245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l242_24269

/-- Parabola type representing y^2 = 2x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focal_point : ℝ × ℝ

/-- Line type representing a line passing through two points -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Point type representing a point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The theorem statement -/
theorem parabola_line_intersection
  (C : Parabola)
  (F : Point)
  (l : Line)
  (A B : Point)
  (h1 : C.equation = fun x y => y^2 = 2*x)
  (h2 : C.focal_point = (1/2, 0))
  (h3 : l.equation F.x F.y)
  (h4 : l.equation A.x A.y ∧ C.equation A.x A.y)
  (h5 : l.equation B.x B.y ∧ C.equation B.x B.y)
  (h6 : (F.x - A.x, F.y - A.y) = (3*(B.x - F.x), 3*(B.y - F.y))) :
  ∃ (m : ℝ), l.equation = fun x y => y = m*(x - 1/2) ∧ m^2 = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l242_24269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l242_24211

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

-- State the theorem
theorem range_of_a_for_increasing_f (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (-3) (-2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l242_24211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l242_24254

noncomputable section

open Real

variable (a b c A B C : ℝ)

-- Define the triangle ABC
def triangle_ABC (a b c A B C : ℝ) : Prop :=
  c = 2 ∧ C = π/3

-- Define the area of the triangle
def area (a b : ℝ) : ℝ := (1/2) * a * b * sqrt 3 / 2

-- Part I
theorem part_I (h : triangle_ABC a b c A B C) (area_eq : area a b = sqrt 3) :
  a = 2 ∧ b = 2 := by sorry

-- Part II
theorem part_II (h : triangle_ABC a b c A B C) (h_sin : sin A = 2 * sin B) :
  area a b = (2 * sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l242_24254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l242_24224

/-- Represents a rectangular garden with flower beds and a trapezoidal remainder -/
structure Garden where
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  flower_bed_count : ℕ
  (short_side_positive : 0 < trapezoid_short_side)
  (long_side_positive : 0 < trapezoid_long_side)
  (short_less_than_long : trapezoid_short_side < trapezoid_long_side)
  (flower_bed_count_is_two : flower_bed_count = 2)

/-- Calculates the area of an isosceles right triangle given its leg length -/
noncomputable def isosceles_right_triangle_area (leg_length : ℝ) : ℝ :=
  (1 / 2) * leg_length ^ 2

/-- Calculates the total area of flower beds in the garden -/
noncomputable def flower_beds_area (g : Garden) : ℝ :=
  let leg_length := (g.trapezoid_long_side - g.trapezoid_short_side) / 2
  g.flower_bed_count * isosceles_right_triangle_area leg_length

/-- Calculates the total area of the garden -/
noncomputable def garden_area (g : Garden) : ℝ :=
  g.trapezoid_long_side * ((g.trapezoid_long_side - g.trapezoid_short_side) / 2)

/-- Theorem stating that the fraction of the garden occupied by flower beds is 1/5 -/
theorem flower_beds_fraction (g : Garden) :
    flower_beds_area g / garden_area g = 1 / 5 := by
  sorry

#check flower_beds_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l242_24224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_bound_count_l242_24292

theorem sqrt_bound_count : 
  (Finset.filter (fun x : ℕ => 25 < x ∧ x < 49) (Finset.range 50)).card = 23 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_bound_count_l242_24292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_AC_length_l242_24221

-- Define the circle U
noncomputable def circle_U : ℝ := 16 * Real.pi

-- Define the angle UAC in radians
noncomputable def angle_UAC : ℝ := 30 * (Real.pi / 180)

-- Theorem statement
theorem segment_AC_length :
  ∀ (radius : ℝ) (diameter : ℝ) (AC : ℝ),
  circle_U = 2 * Real.pi * radius →
  diameter = 2 * radius →
  AC = 2 * radius * Real.sin angle_UAC →
  AC = 8 := by
  intros radius diameter AC h1 h2 h3
  sorry

#check segment_AC_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_AC_length_l242_24221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_selling_price_l242_24217

def purchase_price : ℕ := 10000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℚ := 50 / 100

def total_cost : ℕ := purchase_price + repair_cost + transportation_charges

def selling_price : ℕ := total_cost + (profit_percentage * total_cost).floor.toNat

theorem machine_selling_price :
  selling_price = 24000 := by
  -- Unfold definitions
  unfold selling_price
  unfold total_cost
  unfold profit_percentage
  -- Simplify arithmetic
  simp [Nat.add_assoc, Nat.mul_comm]
  -- The rest of the proof
  sorry

#eval selling_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_selling_price_l242_24217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fizz_survey_result_l242_24215

/-- The number of participants who chose "Fizz" in a survey -/
def fizz_participants (total : ℕ) (central_angle : ℚ) : ℚ :=
  (total : ℚ) * (central_angle / 360)

/-- Theorem: Given 500 total participants and a 270° central angle for "Fizz",
    375 participants chose "Fizz" -/
theorem fizz_survey_result :
  fizz_participants 500 270 = 375 := by
  -- Unfold the definition of fizz_participants
  unfold fizz_participants
  -- Perform the calculation
  norm_num
  -- QED

#eval Int.floor (fizz_participants 500 270)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fizz_survey_result_l242_24215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_one_necessary_not_sufficient_for_sin_zero_l242_24207

theorem cos_one_necessary_not_sufficient_for_sin_zero :
  (∀ x : ℝ, Real.sin x = 0 → Real.cos x = 1) ∧
  (∃ x : ℝ, Real.cos x = 1 ∧ Real.sin x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_one_necessary_not_sufficient_for_sin_zero_l242_24207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l242_24289

theorem ticket_price_possibilities : 
  let possible_prices := {x : ℕ | x > 0 ∧ 60 % x = 0 ∧ 80 % x = 0}
  Finset.card (Finset.filter (λ x => x > 0 ∧ 60 % x = 0 ∧ 80 % x = 0) (Finset.range 81)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l242_24289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_division_theorem_l242_24225

/-- Represents a regular hexagon with diagonals marked --/
structure RegularHexagonWithDiagonals where
  area : ℝ
  diagonals_marked : Bool

/-- The number of parts a regular hexagon is divided into when all diagonals are marked --/
def num_parts : ℕ := 24

/-- The area of the inner regular hexagon formed by quadrilateral parts --/
noncomputable def inner_hexagon_area (h : RegularHexagonWithDiagonals) : ℝ := h.area / 3

/-- Theorem stating the number of parts and the area of the inner hexagon --/
theorem hexagon_division_theorem (h : RegularHexagonWithDiagonals) 
  (h_area : h.area = 144) 
  (h_marked : h.diagonals_marked = true) : 
  num_parts = 24 ∧ inner_hexagon_area h = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_division_theorem_l242_24225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_list_mean_mode_relation_l242_24258

theorem integer_list_mean_mode_relation (y : ℤ) : 
  y > 0 ∧ y ≤ 100 →
  (42 + 74 + 87 + 3 * y : ℚ) / 6 = 1.25 * y →
  y = 45 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_list_mean_mode_relation_l242_24258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_defect_analysis_l242_24277

/-- Defective rates for three processes -/
def p1 : ℚ := 1/10
def p2 : ℚ := 1/11
def p3 : ℚ := 1/12

/-- Number of products selected -/
def n : ℕ := 3

/-- Overall defective rate -/
def p : ℚ := 1 - (1 - p1) * (1 - p2) * (1 - p3)

/-- Random variable X: number of defective items in n selected products -/
def X : Fin (n + 1) → ℚ := λ k ↦
  (n.choose k.val) * p^k.val * (1 - p)^(n - k.val)

/-- Theorem stating the main results -/
theorem product_defect_analysis :
  (p = 1/4) ∧
  (∀ k : Fin (n + 1), X k = if k.val = 0 then 27/64
                            else if k.val = 1 then 27/64
                            else if k.val = 2 then 9/64
                            else 1/64) ∧
  (n * p * (1 - p) = 9/16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_defect_analysis_l242_24277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_ratio_l242_24201

theorem cylinder_sphere_volume_ratio (r : ℝ) (hr : r > 0) :
  (2 * Real.pi * r^3) / ((4/3) * Real.pi * r^3) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_ratio_l242_24201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l242_24206

open Real

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (cos t.B / t.b + cos t.C / (2 * t.a + t.c) = 0) ∧
  (t.b = Real.sqrt 13) ∧
  (t.a + t.c = 4)

-- Helper function for area
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.B = 2 * π / 3 ∧ 
  area t = 3 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l242_24206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_win_streak_l242_24282

theorem soccer_win_streak (initial_games : ℕ) (initial_win_rate : ℚ) (final_win_rate : ℚ) (n : ℕ) :
  initial_games = 40 →
  initial_win_rate = 2/5 →
  final_win_rate = 1/2 →
  (initial_win_rate * initial_games + n) / (initial_games + n) = final_win_rate →
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_win_streak_l242_24282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l242_24263

-- Define the equation
def equation (x : ℝ) : Prop := x * Real.sqrt x - 7 * x + 8 * Real.sqrt x + 5 = 0

-- Define the set of roots
def roots : Set ℝ := {x | equation x ∧ x ≥ 0}

theorem sum_of_squares_of_roots :
  ∃ (r s t : ℝ), r ∈ roots ∧ s ∈ roots ∧ t ∈ roots ∧
  (∀ x, x ∈ roots → x = r ∨ x = s ∨ x = t) →
  r^2 + s^2 + t^2 = 17/4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l242_24263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l242_24276

-- Define the triangle XYZ
structure Triangle :=
  (x y z : ℝ)
  (xy_length : ℝ)
  (yz_length : ℝ)
  (zx_length : ℝ)

-- Define the points P, Q, R
structure DividingPoints (t : Triangle) :=
  (s t u : ℝ)
  (positive_s : 0 < s)
  (positive_t : 0 < t)
  (positive_u : 0 < u)
  (sum_condition : s + t + u = 3/4)
  (square_sum_condition : s^2 + t^2 + u^2 = 3/7)

-- Define area functions (these are placeholders and should be properly defined)
def area_PQR (t : Triangle) (d : DividingPoints t) : ℝ := sorry

def area_XYZ (t : Triangle) : ℝ := sorry

-- Define the theorem
theorem area_ratio_theorem (t : Triangle) (d : DividingPoints t) 
  (h1 : t.xy_length = 14)
  (h2 : t.yz_length = 16)
  (h3 : t.zx_length = 18) :
  (area_PQR t d) / (area_XYZ t) = 59 / 112 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l242_24276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_and_function_l242_24264

theorem trigonometric_identities_and_function (α : ℝ) :
  α ∈ Set.Ioo π (3 * π / 2) →
  Real.sin α = -3/5 →
  (Real.cos α = -4/5 ∧ 
   Real.tan α = 3/4 ∧
   (2 * Real.sin (π + α) + Real.cos (2 * π + α)) / (Real.cos (α - π/2) + Real.sin (π/2 + α)) = -2/7) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_and_function_l242_24264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_cube_relation_l242_24213

/-- A function representing the inverse relationship between x and y^3 -/
noncomputable def inverse_relation (k : ℝ) (y : ℝ) : ℝ := k / (y^3)

/-- Theorem stating that given the inverse relationship between x and y^3,
    if x = 4 when y = 1, then x = 1/2 when y = 2 -/
theorem inverse_cube_relation (k : ℝ) :
  (inverse_relation k 1 = 4) → (inverse_relation k 2 = 1/2) := by
  intro h
  have k_eq : k = 4 := by
    rw [inverse_relation] at h
    simp at h
    exact h
  rw [inverse_relation]
  simp [k_eq]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_cube_relation_l242_24213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_point_l242_24237

/-- 
Given an angle α with vertex at the origin and initial side on the positive x-axis,
and a point P on its terminal side, prove that cos(2α) = -7/25.
-/
theorem cos_double_angle_special_point (α : ℝ) (P : ℝ × ℝ) :
  P.1 = -3/5 ∧ P.2 = 4/5 ∧ P ≠ (0, 0) →
  Real.cos (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_point_l242_24237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_to_line_l242_24296

/-- Given a triangle ABC with perpendicular distances from its vertices to a line,
    prove that the perpendicular distance from the centroid to the line
    is the average of the distances from the vertices. -/
theorem centroid_distance_to_line (a b c : ℝ) :
  a = 10 ∧ b = 6 ∧ c = 24 →
  (a + b + c) / 3 = 40 / 3 := by
  intro h
  sorry

#check centroid_distance_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_to_line_l242_24296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_A_is_right_triangle_l242_24287

/-- Checks if three numbers form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The sets of numbers to check --/
noncomputable def set_A : List ℝ := [1, 2, Real.sqrt 5]
def set_B : List ℝ := [6, 8, 9]
noncomputable def set_C : List ℝ := [Real.sqrt 3, Real.sqrt 2, 5]
def set_D : List ℝ := [3^2, 4^2, 5^2]

/-- Theorem stating that only set A forms a right-angled triangle --/
theorem only_set_A_is_right_triangle :
  (∃ (a b c : ℝ), a ∈ set_A ∧ b ∈ set_A ∧ c ∈ set_A ∧ is_right_triangle a b c) ∧
  (¬ ∃ (a b c : ℝ), a ∈ set_B ∧ b ∈ set_B ∧ c ∈ set_B ∧ is_right_triangle a b c) ∧
  (¬ ∃ (a b c : ℝ), a ∈ set_C ∧ b ∈ set_C ∧ c ∈ set_C ∧ is_right_triangle a b c) ∧
  (¬ ∃ (a b c : ℝ), a ∈ set_D ∧ b ∈ set_D ∧ c ∈ set_D ∧ is_right_triangle a b c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_A_is_right_triangle_l242_24287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_l242_24290

noncomputable def f (x : ℝ) : ℝ := x^3 + (1/2) * x^2 - 4*x

noncomputable def f' (x : ℝ) : ℝ := 3*x^2 + x - 4

theorem f_extrema_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (-2) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2) 2, f x = min) ∧
    max = 104/27 ∧ min = -5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_l242_24290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l242_24249

-- Define the propositions P and Q as functions of m
def P (m : ℝ) : Prop := ∀ x y, x < y → Real.log (x + 1) / Real.log (2 * m) < Real.log (y + 1) / Real.log (2 * m)

def Q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 ≥ 0

-- Define the range of m
def range_m (m : ℝ) : Prop := m ∈ Set.Icc (-2) (1/2) ∪ Set.Ioi 2

-- State the theorem
theorem m_range_theorem (m : ℝ) : 
  ((P m ∨ Q m) ∧ ¬(P m ∧ Q m)) ↔ range_m m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l242_24249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l242_24299

theorem inequality_theorem (a b n x : ℝ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  a / (Real.sin x)^n + b / (Real.cos x)^n ≥ (a^(2/(n+2)) + b^(2/(n+2)))^((n+2)/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l242_24299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_planes_partition_max_parts_correct_l242_24233

/-- The number of parts into which four planes can divide space -/
def max_parts : ℕ := 15

/-- Formula for the maximum number of parts created by x planes -/
def plane_partition (x : ℕ) : ℚ :=
  (x^3 + 5*x + 6) / 6

theorem four_planes_partition :
  plane_partition 4 = max_parts := by
  unfold plane_partition max_parts
  norm_num

theorem max_parts_correct :
  max_parts = 15 := by rfl

#eval max_parts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_planes_partition_max_parts_correct_l242_24233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_trapezoid_l242_24285

-- Helper definitions
def is_trapezoid_sides (a b c d : ℝ) : Prop := sorry

def is_inscribed_circle_trapezoid (r chord_length a b c d : ℝ) : Prop := sorry

theorem inscribed_circle_trapezoid (r : ℝ) (chord_length : ℝ) 
  (h1 : r = 5)
  (h2 : chord_length = 8) :
  ∃ (a b c d : ℝ),
    a = 20 ∧ b = 5 ∧ c = 12.5 ∧ d = 12.5 ∧
    is_trapezoid_sides a b c d ∧
    is_inscribed_circle_trapezoid r chord_length a b c d :=
by
  sorry

-- Additional definitions or lemmas can be added here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_trapezoid_l242_24285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_in_U_l242_24234

-- Define the universal set U
def U : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≤ 0}

-- Define the set M
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, x^2 + y^2 = 1}

-- Theorem statement
theorem complement_M_in_U : 
  (U \ M) = Set.Ioc 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_in_U_l242_24234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minus_one_x_one_x_minus_one_equals_25_l242_24265

-- Define the operation x
noncomputable def x (a b : ℝ) : ℝ := (b - a)^2 / a^2

-- Theorem statement
theorem minus_one_x_one_x_minus_one_equals_25 :
  x (-1) (x 1 (-1)) = 25 := by
  -- Expand the definition of x
  unfold x
  -- Simplify the expression
  simp
  -- Perform algebraic manipulations
  ring
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minus_one_x_one_x_minus_one_equals_25_l242_24265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_face_angle_formula_l242_24261

/-- A regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- Side length of the base -/
  a : ℝ
  /-- Lateral surface area -/
  S : ℝ
  /-- a and S are positive -/
  a_pos : 0 < a
  S_pos : 0 < S

/-- The angle between adjacent lateral faces of a regular quadrilateral pyramid -/
noncomputable def lateral_face_angle (p : RegularQuadPyramid) : ℝ :=
  Real.arccos (-p.a^4 / p.S^2)

/-- Theorem: The angle between adjacent lateral faces of a regular quadrilateral pyramid
    is equal to arccos(-a^4/S^2), where a is the side length of the base and S is the
    lateral surface area -/
theorem lateral_face_angle_formula (p : RegularQuadPyramid) :
  lateral_face_angle p = Real.arccos (-p.a^4 / p.S^2) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_face_angle_formula_l242_24261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l242_24288

theorem polynomial_value_bound (n : ℕ) (p : Polynomial ℝ) (b : Fin (n + 1) → ℤ) :
  p.natDegree = n →
  (∀ i j : Fin (n + 1), i ≠ j → b i ≠ b j) →
  ∃ j : Fin (n + 1), |p.eval (↑(b j))| ≥ (n.factorial : ℝ) / 2^n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l242_24288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l242_24227

def a : ℕ → ℚ
  | 0 => 1/2  -- Add this case for n = 0
  | 1 => 1/2
  | n + 1 => 3 * a n / (a n + 3)

theorem a_formula (n : ℕ) : n ≥ 1 → a n = 3 / (n + 5) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l242_24227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l242_24219

/-- A complex number is pure imaginary if its real part is zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0

theorem omega_values (z ω : ℂ) 
  (h1 : IsPureImaginary ((1 + 3*Complex.I) * z))
  (h2 : ω = z / (2 + Complex.I))
  (h3 : Complex.abs ω = 5 * Real.sqrt 2) :
  ω = 7 - Complex.I ∨ ω = -7 + Complex.I := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l242_24219
