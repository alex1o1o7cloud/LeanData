import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_decrease_notation_l514_51405

/-- Represents a temperature change in degrees Celsius -/
structure TempChange where
  value : Int
  unit : String

/-- Defines how temperature changes are denoted -/
def denoteTempChange (change : TempChange) : String :=
  if change.value ≥ 0 then
    s!"+{change.value}{change.unit}"
  else
    s!"{change.value}{change.unit}"

theorem temp_decrease_notation (rise : TempChange) (decrease : TempChange) 
    (h_rise : rise.value = 8 ∧ rise.unit = "°C" ∧ denoteTempChange rise = "+8°C")
    (h_decrease : decrease.value = -5 ∧ decrease.unit = "°C") :
  denoteTempChange decrease = "-5°C" := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_decrease_notation_l514_51405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_repair_cost_is_360_l514_51442

/-- Represents the financial details of Anil's scooter purchase and repairs -/
structure ScooterFinancials where
  originalCost : ℝ
  firstRepairPercent : ℝ
  secondRepairPercent : ℝ
  thirdRepairPercent : ℝ
  taxPercent : ℝ
  profitAmount : ℝ
  profitPercent : ℝ

/-- Calculates the amount spent on the third repair -/
def thirdRepairCost (sf : ScooterFinancials) : ℝ :=
  sf.originalCost * sf.thirdRepairPercent

/-- Theorem stating that the third repair cost is 360 given the problem conditions -/
theorem third_repair_cost_is_360 (sf : ScooterFinancials)
  (h1 : sf.firstRepairPercent = 0.08)
  (h2 : sf.secondRepairPercent = 0.12)
  (h3 : sf.thirdRepairPercent = 0.05)
  (h4 : sf.taxPercent = 0.07)
  (h5 : sf.profitAmount = 1800)
  (h6 : sf.profitPercent = 0.25)
  (h7 : sf.profitAmount = sf.originalCost * sf.profitPercent) :
  thirdRepairCost sf = 360 := by
  sorry

-- Remove the #eval line as it's not necessary and can cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_repair_cost_is_360_l514_51442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_points_in_unit_triangle_l514_51481

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (side : ℝ)
  (is_positive : side > 0)

-- Define a set of points inside a triangle
def PointsInsideTriangle (t : EquilateralTriangle) (n : ℕ) := 
  Fin n → Set (ℝ × ℝ)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem five_points_in_unit_triangle (t : EquilateralTriangle) 
  (h : t.side = 1) (points : PointsInsideTriangle t 5) :
  ∃ (i j : Fin 5), i ≠ j ∧ 
    ∀ (p1 p2 : ℝ × ℝ), p1 ∈ points i → p2 ∈ points j → distance p1 p2 ≤ 1/2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_points_in_unit_triangle_l514_51481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_of_four_boxes_l514_51468

def box_volume (edge_length : ℝ) : ℝ := edge_length ^ 3

theorem total_volume_of_four_boxes (edge_length : ℝ) (h : edge_length = 5) :
  4 * box_volume edge_length = 500 := by
  have h1 : box_volume edge_length = 125 := by
    rw [box_volume, h]
    norm_num
  calc
    4 * box_volume edge_length = 4 * 125 := by rw [h1]
    _ = 500 := by norm_num

#eval box_volume 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_of_four_boxes_l514_51468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_sum_l514_51429

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem min_abs_sum (a : ℝ) :
  (∃ x₁ x₂, f a x₁ - f a x₂ = -4) →
  (∀ x, f a (x + π/3) = -f a (-x + π/3)) →
  (∃ x₁ x₂, f a x₁ - f a x₂ = -4 ∧ |x₁ + x₂| = 2*π/3 ∧ 
    ∀ y₁ y₂, f a y₁ - f a y₂ = -4 → |y₁ + y₂| ≥ 2*π/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_sum_l514_51429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_tangent_to_x_axis_f_has_two_zero_points_l514_51443

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.log x + a * x - 1

-- Theorem 1: When a = 1, the graph of f(x) is tangent to the x-axis
theorem f_tangent_to_x_axis :
  ∃ x₀ : ℝ, x₀ > 0 ∧ f 1 x₀ = 0 ∧ (deriv (f 1)) x₀ = 0 := by
  sorry

-- Theorem 2: When a < 1, f(x) has two zero points
theorem f_has_two_zero_points (a : ℝ) (h : a < 1) :
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_tangent_to_x_axis_f_has_two_zero_points_l514_51443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_sum_proof_l514_51482

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem investment_sum_proof (P : ℝ) :
  simple_interest P 18 2 - simple_interest P 12 2 = 300 →
  P = 2500 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check investment_sum_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_sum_proof_l514_51482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_OP_l514_51453

-- Define the curve C in polar coordinates
def curve_C (p θ : ℝ) : Prop :=
  p^2 + 2 * Real.sqrt 2 * p * Real.cos (θ + Real.pi / 4) = 2

-- Define the line l
def line_l (α : ℝ) : Prop :=
  0 ≤ α ∧ α < Real.pi

-- Define the distance from origin to point P
noncomputable def distance_OP (α : ℝ) : ℝ :=
  Real.sqrt 2 * |Real.sin (α - Real.pi / 4)|

-- State the theorem
theorem max_distance_OP :
  ∀ α, line_l α → distance_OP α ≤ Real.sqrt 2 ∧
  ∃ α₀, line_l α₀ ∧ distance_OP α₀ = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_OP_l514_51453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_equidistant_point_l514_51401

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- A tangent line to a circle -/
structure Tangent where
  line : ℝ → ℝ  -- Changed to a function ℝ → ℝ to represent a line
  distance_from_center : ℝ

/-- The configuration of the circle and three parallel tangents -/
structure CircleWithTangents where
  circle : Circle
  tangent1 : Tangent
  tangent2 : Tangent
  tangent3 : Tangent
  parallel : tangent1.line = tangent2.line ∧ tangent2.line = tangent3.line
  equidistant : tangent1.distance_from_center = circle.r ∧ tangent2.distance_from_center = circle.r
  third_tangent : tangent3.distance_from_center = circle.r + 3

/-- A point is equidistant from the circle and all tangents -/
def is_equidistant (p : ℝ × ℝ) (config : CircleWithTangents) : Prop :=
  let d_circle := Real.sqrt ((p.1 - config.circle.O.1)^2 + (p.2 - config.circle.O.2)^2) - config.circle.r
  let d_tangent1 := |config.tangent1.line p.2 - p.1|
  let d_tangent2 := |config.tangent2.line p.2 - p.1|
  let d_tangent3 := |config.tangent3.line p.2 - p.1|
  d_circle = d_tangent1 ∧ d_circle = d_tangent2 ∧ d_circle = d_tangent3

/-- The main theorem -/
theorem unique_equidistant_point (config : CircleWithTangents) :
  ∃! p, is_equidistant p config :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_equidistant_point_l514_51401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l514_51459

noncomputable def a (x m : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, m + Real.cos x)

noncomputable def b (x m : ℝ) : ℝ × ℝ := (Real.cos x, -m + Real.cos x)

noncomputable def f (x m : ℝ) : ℝ := (a x m).1 * (b x m).1 + (a x m).2 * (b x m).2

theorem f_properties (m : ℝ) :
  ∃ (period : ℝ) (max_val : ℝ) (max_x : ℝ),
    (∀ x, f (x + period) m = f x m) ∧
    period = Real.pi ∧
    (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/3), f x m ≤ max_val) ∧
    max_val = -5/2 ∧
    max_x = Real.pi/6 ∧
    f max_x m = max_val ∧
    (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/3), f x m = -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l514_51459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_sum_proof_l514_51475

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem investment_sum_proof (P : ℝ) :
  simple_interest P 18 2 - simple_interest P 12 2 = 840 →
  P = 7000 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_sum_proof_l514_51475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_parabola_line_l514_51404

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2

/-- The line function -/
def line (x : ℝ) : ℝ := -2*x + 3

/-- Theorem: The area enclosed by the parabola y = x^2 and the line 2x + y - 3 = 0 is equal to 32/3 -/
theorem area_enclosed_parabola_line : 
  (∫ x in (-3)..1, line x - parabola x) = 32/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_parabola_line_l514_51404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lopez_seating_arrangements_l514_51417

/-- Represents the Lopez family members -/
inductive FamilyMember
| MrLopez
| MrsLopez
| Child1
| Child2
| Child3

/-- Represents the seats in the car -/
inductive Seat
| Driver
| FrontPassenger
| BackLeft
| BackMiddle
| BackRight

/-- A seating arrangement is a function from Seat to FamilyMember -/
def SeatingArrangement := Seat → FamilyMember

/-- Check if a seating arrangement is valid -/
def isValidArrangement (arrangement : SeatingArrangement) : Prop :=
  (arrangement Seat.Driver = FamilyMember.MrLopez ∨ 
   arrangement Seat.Driver = FamilyMember.MrsLopez) ∧
  (∀ s1 s2 : Seat, s1 ≠ s2 → arrangement s1 ≠ arrangement s2)

/-- The theorem stating the number of valid seating arrangements -/
theorem lopez_seating_arrangements :
  ∃ (arrangements : Finset SeatingArrangement), 
    (∀ a ∈ arrangements, isValidArrangement a) ∧ 
    arrangements.card = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lopez_seating_arrangements_l514_51417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_students_count_l514_51456

theorem exam_students_count : ∃ (N : ℕ), N = 10 :=
  let avg_mark : ℚ := 70
  let excluded_avg : ℚ := 50
  let excluded_count : ℕ := 5
  let remaining_avg : ℚ := 90

  have h1 : avg_mark = 70 := rfl
  have h2 : excluded_count = 5 := rfl
  have h3 : excluded_avg = 50 := rfl
  have h4 : remaining_avg = 90 := rfl

  sorry

#check exam_students_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_students_count_l514_51456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l514_51410

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := 3^x

-- Define the proposed inverse function g
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Theorem statement
theorem inverse_function_proof (x : ℝ) (hx : x ≥ 1) (y : ℝ) (hy : y ≥ 3) :
  (f ∘ g) y = y ∧ (g ∘ f) x = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l514_51410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_B_opposite_l514_51421

-- Define the real number x with the condition x ≠ ±2
variable (x : ℝ) (hx : x ≠ 2 ∧ x ≠ -2)

-- Define A and B as functions of x
noncomputable def A (x : ℝ) : ℝ := 4 / (x^2 - 4)
noncomputable def B (x : ℝ) : ℝ := 1 / (x + 2) + 1 / (2 - x)

-- State the theorem
theorem A_B_opposite (x : ℝ) (hx : x ≠ 2 ∧ x ≠ -2) : 
  A x + B x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_B_opposite_l514_51421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_upper_torso_spots_l514_51439

/-- Represents the total number of spots on Jean the jaguar. -/
def total_spots : ℕ := sorry

/-- Represents the number of spots on Jean's upper torso. -/
def upper_torso_spots : ℕ := sorry

/-- Represents the number of spots on Jean's back and hindquarters. -/
def back_hindquarters_spots : ℕ := sorry

/-- Represents the number of spots on Jean's sides. -/
def side_spots : ℕ := sorry

/-- The number of spots on Jean's sides is 10. -/
axiom side_spots_count : side_spots = 10

/-- Half of Jean's total spots are on her upper torso. -/
axiom upper_torso_half : upper_torso_spots = total_spots / 2

/-- One-third of Jean's total spots are on her back and hindquarters. -/
axiom back_hindquarters_third : back_hindquarters_spots = total_spots / 3

/-- The remaining spots are on Jean's sides. -/
axiom remaining_on_sides : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots

/-- Theorem: The number of spots on Jean's upper torso is 30. -/
theorem jean_upper_torso_spots : upper_torso_spots = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_upper_torso_spots_l514_51439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_angle_pi_over_three_l514_51431

noncomputable section

-- Define the vectors
def a : ℝ × ℝ := (1, -1)
def b (k : ℝ) : ℝ × ℝ := (1, k)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the angle between two vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ := Real.arccos ((dot_product v w) / (magnitude v * magnitude w))

-- Theorem 1: If a ⊥ b, then k = 1
theorem perpendicular_vectors (k : ℝ) : 
  dot_product a (b k) = 0 → k = 1 := by sorry

-- Theorem 2: If ⟨a, b⟩ = π/3, then k = 2 - √3
theorem angle_pi_over_three (k : ℝ) : 
  angle a (b k) = π/3 → k = 2 - Real.sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_angle_pi_over_three_l514_51431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_in_triangle_l514_51418

noncomputable section

-- Define a triangle ABC
def Triangle (A B C : ℂ) : Prop := A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define a point P inside the triangle
def InsideTriangle (P A B C : ℂ) : Prop :=
  ∃ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧
  P = α • A + β • B + γ • C

-- Define the function we want to minimize
noncomputable def f (P A B C : ℂ) : ℝ :=
  Complex.abs (P - A) / Complex.abs (B - C) +
  Complex.abs (P - B) / Complex.abs (A - C) +
  Complex.abs (P - C) / Complex.abs (A - B)

-- State the theorem
theorem min_value_in_triangle (A B C : ℂ) (h : Triangle A B C) :
  ∀ P, InsideTriangle P A B C → f P A B C ≥ Real.sqrt 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_in_triangle_l514_51418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cosine_value_of_triangle_l514_51470

noncomputable def wire_lengths : List ℝ := [2, 3, 4, 6]

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def cos_angle (a b c : ℝ) : ℝ :=
  (a^2 + b^2 - c^2) / (2 * a * b)

noncomputable def min_cos_angle (a b c : ℝ) : ℝ :=
  min (cos_angle a b c) (min (cos_angle b c a) (cos_angle c a b))

theorem min_cosine_value_of_triangle (l₁ l₂ l₃ : ℝ) :
  l₁ ∈ wire_lengths → l₂ ∈ wire_lengths → l₃ ∈ wire_lengths →
  l₁ ≠ l₂ → l₂ ≠ l₃ → l₃ ≠ l₁ →
  is_triangle l₁ l₂ l₃ →
  min_cos_angle l₁ l₂ l₃ ≥ 43/48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cosine_value_of_triangle_l514_51470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_angles_l514_51461

noncomputable def is_geometric_sequence (a b c : ℝ) : Prop :=
  (a * c = b^2) ∨ (a * b = c^2) ∨ (b * c = a^2)

noncomputable def valid_angle (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 2 * Real.pi ∧
  ¬∃ (k : ℤ), θ = (k : ℝ) * Real.pi / 3 ∧
  is_geometric_sequence (Real.sin θ) (Real.cos θ) (1 / Real.tan θ)

theorem count_valid_angles : 
  ∃ (s : Finset ℝ), (∀ θ ∈ s, valid_angle θ) ∧ s.card = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_angles_l514_51461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l514_51432

theorem quadratic_roots (b c : ℝ) (h : b^2 - 4*c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ = (-b + Real.sqrt (b^2 - 4*c)) / 2 ∧
              x₂ = (-b - Real.sqrt (b^2 - 4*c)) / 2 ∧
              (x₁^2 + b*x₁ + c = 0) ∧ (x₂^2 + b*x₂ + c = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l514_51432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_divisors_90_l514_51480

def is_odd_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ Odd d

def sum_odd_divisors_less_than_10 (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d ∣ n ∧ Odd d ∧ d < 10) (Finset.range (n + 1))).sum id

theorem sum_odd_divisors_90 :
  sum_odd_divisors_less_than_10 90 = 18 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_divisors_90_l514_51480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l514_51490

theorem power_equation_solution (y : ℝ) : (9 : ℝ)^y = (3 : ℝ)^16 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l514_51490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_purchase_cost_l514_51433

/-- Calculates the final cost of a laptop and accessories purchase with discounts and tax -/
theorem laptop_purchase_cost
  (laptop_cost : ℝ)
  (accessories_cost : ℝ)
  (laptop_discount_rate : ℝ)
  (accessories_discount_rate : ℝ)
  (tax_rate : ℝ)
  (h1 : laptop_cost = 800)
  (h2 : accessories_cost = 200)
  (h3 : laptop_discount_rate = 0.15)
  (h4 : accessories_discount_rate = 0.10)
  (h5 : tax_rate = 0.07) :
  (laptop_cost * (1 - laptop_discount_rate) +
   accessories_cost * (1 - accessories_discount_rate)) *
  (1 + tax_rate) = 920.20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_purchase_cost_l514_51433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_soda_price_l514_51476

/-- The price of discounted soda cans -/
theorem discounted_soda_price 
  (regular_price : ℝ) 
  (discount_percent : ℝ) 
  (case_size : ℕ) 
  (total_cans : ℕ) :
  regular_price = 0.60 →
  discount_percent = 20 →
  case_size = 24 →
  total_cans = 72 →
  let discounted_price := regular_price * (1 - discount_percent / 100)
  let total_price := (total_cans / case_size : ℝ) * (case_size : ℝ) * discounted_price
  total_price = 34.56 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check discounted_soda_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_soda_price_l514_51476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_cost_graph_l514_51488

-- Define the cost function
def cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 20 * n else 18 * n

-- Define the set of points on the graph
def graph : Set (ℕ × ℚ) :=
  {(n, cost n) | n ∈ Finset.range 16 \ {0}}

-- Theorem statement
theorem goldfish_cost_graph :
  ∃ (p₁ p₂ p₃ : ℕ × ℚ),
    p₁ ∈ graph ∧ p₂ ∈ graph ∧ p₃ ∈ graph ∧
    p₁.1 < p₂.1 ∧ p₂.1 < p₃.1 ∧
    (∀ q ∈ graph, q.1 ≤ p₂.1 → ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧ q.2 = (1 - t) * p₁.2 + t * p₂.2) ∧
    (∀ q ∈ graph, q.1 ≥ p₂.1 → ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧ q.2 = (1 - t) * p₂.2 + t * p₃.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_cost_graph_l514_51488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_3_in_695_factorial_base_l514_51498

def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

def factorial_base_representation (n : ℕ) : List ℕ :=
  sorry

theorem a_4_equals_3_in_695_factorial_base :
  let repr := factorial_base_representation 695
  ∀ k, k < repr.length → 0 ≤ repr[k]! ∧ repr[k]! ≤ k + 1 →
  695 = (List.sum (List.zipWith (· * ·) repr (List.map factorial (List.range repr.length)))) →
  repr[3]! = 3 :=
by sorry

#eval factorial 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_3_in_695_factorial_base_l514_51498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_unit_interval_range_of_a_l514_51440

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x^2 + 2/x
noncomputable def g (x a : ℝ) : ℝ := x^2 / (x^2 + 2*x + 1) + (4*x + 10) / (9*x + 9) - a

-- Part 1: Monotonicity of f on (0,1)
theorem f_decreasing_on_unit_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → f x₁ > f x₂ := by sorry

-- Part 2: Range of a
theorem range_of_a :
  ∀ a : ℝ, 
    (∀ x₁ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 1 → 
      ∃ x₂ : ℝ, 2/3 ≤ x₂ ∧ x₂ ≤ 2 ∧ g x₁ a = f x₂) ↔ 
    (-35/9 ≤ a ∧ a ≤ -2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_unit_interval_range_of_a_l514_51440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_formula_l514_51477

/-- Right triangle with inscribed circle -/
structure RightTriangleWithInscribedCircle where
  h : ℝ  -- length of hypotenuse
  r : ℝ  -- radius of inscribed circle (inradius)
  d : ℝ  -- length of altitude from hypotenuse to right angle vertex
  h_pos : 0 < h  -- hypotenuse is positive
  r_pos : 0 < r  -- inradius is positive
  d_pos : 0 < d  -- altitude is positive

/-- The ratio of the area of the inscribed circle to the area of the right triangle -/
noncomputable def areaRatio (t : RightTriangleWithInscribedCircle) : ℝ :=
  (Real.pi * t.r) / (t.h + t.r + t.d)

/-- Theorem: The ratio of the area of the inscribed circle to the area of the right triangle
    is equal to πr / (h + r + d) -/
theorem area_ratio_formula (t : RightTriangleWithInscribedCircle) :
  areaRatio t = (Real.pi * t.r) / (t.h + t.r + t.d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_formula_l514_51477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_special_n_l514_51451

theorem digit_sum_of_special_n : ∃ (n : ℕ), 
  (Nat.factorial (n + 1) + Nat.factorial (n + 3) = 440 * Nat.factorial n) ∧ 
  (Nat.digits 10 n).sum = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_special_n_l514_51451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l514_51402

theorem cosine_sum_product_form :
  ∃ (a b c d : ℕ+),
    (∀ x : ℝ, Real.cos (2 * x) + Real.cos (4 * x) + Real.cos (8 * x) + Real.cos (10 * x) = 
      (a : ℝ) * Real.cos (↑b * x) * Real.cos (↑c * x) * Real.cos (↑d * x)) ∧
    a + b + c + d = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l514_51402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_haiku_probability_sum_l514_51485

theorem haiku_probability_sum : ∃ (m n : ℕ), 
  m > 0 ∧ n > 0 ∧
  (m * 3003 = n) ∧ 
  (Nat.gcd m n = 1) ∧
  ((Nat.factorial 10 * Nat.factorial 5) * m = Nat.factorial 15) ∧
  (m + n = 3004) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_haiku_probability_sum_l514_51485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_attendance_proof_l514_51434

def theater_attendance 
  (adult_ticket_price : ℕ) 
  (child_ticket_price : ℕ) 
  (total_revenue : ℕ) 
  (num_children : ℕ) : ℕ :=
  let num_adults := (total_revenue - child_ticket_price * num_children) / adult_ticket_price
  num_adults + num_children

theorem theater_attendance_proof 
  (h1 : adult_ticket_price = 11) 
  (h2 : child_ticket_price = 10) 
  (h3 : total_revenue = 246) 
  (h4 : num_children = 7) : 
  theater_attendance 11 10 246 7 = 23 := by
  sorry

#eval theater_attendance 11 10 246 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_attendance_proof_l514_51434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l514_51449

/-- The time (in hours) it takes for machines P, Q, and R to complete the job together -/
noncomputable def x : ℝ := 2

/-- The time (in hours) it takes for machine P to complete the job alone -/
noncomputable def time_P : ℝ := x + 8

/-- The time (in hours) it takes for machine Q to complete the job alone -/
noncomputable def time_Q : ℝ := x + 2

/-- The time (in hours) it takes for machine R to complete the job alone -/
noncomputable def time_R : ℝ := 2 * x

/-- The combined work rate of machines P, Q, and R -/
noncomputable def combined_rate : ℝ := 1 / x

/-- The theorem stating that x = 2 given the conditions of the problem -/
theorem job_completion_time : 
  (1 / time_P + 1 / time_Q + 1 / time_R = combined_rate) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l514_51449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l514_51497

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℝ, (X : Polynomial ℝ)^5 + 3 = (X - 3)^2 * q + (27 * X - 78) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l514_51497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subset_count_complement_intersection_empty_l514_51406

-- Define the sets A and B
def A (b : ℝ) : Set ℝ := {x : ℝ | x^2 - 3*x + b = 0}
def B : Set ℝ := {x : ℝ | (x-2)*(x^2+3*x-4) = 0}

-- Part 1
theorem proper_subset_count (b : ℝ) : 
  b = 4 → ∃! (count : ℕ), ∃ (S : Finset (Set ℝ)), 
    (∀ M ∈ S, A b ⊂ M ∧ M ⊂ B) ∧ S.card = count ∧ count = 7 := by sorry

-- Part 2
theorem complement_intersection_empty : 
  ∀ b : ℝ, (Set.compl B ∩ A b = ∅) ↔ (b > 9/4 ∨ b = 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subset_count_complement_intersection_empty_l514_51406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_form_fraction_l514_51403

/-- A fraction is in its simplest form if the numerator and denominator do not share any common factors other than 1 and -1. -/
def IsSimplestForm {R : Type*} [CommRing R] (n d : R) : Prop :=
  ∀ f : R, f ∣ n ∧ f ∣ d → IsUnit f ∨ IsUnit (-f)

/-- The fraction (x^2 + y^2) / (x + y) is in its simplest form. -/
theorem simplest_form_fraction {R : Type*} [CommRing R] (x y : R) (h : x ≠ -y) :
  IsSimplestForm (x^2 + y^2) (x + y) := by
  sorry

#check simplest_form_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_form_fraction_l514_51403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_path_length_l514_51422

/-- Represents a square with side length 5 inches -/
def Square : ℝ := 5

/-- Represents the length of the hypotenuse AB of the right triangle ABP -/
def Hypotenuse : ℝ := 3

/-- Represents the number of rotations (one for each side of the square) -/
def NumRotations : ℕ := 4

/-- Represents the angle of rotation for each side (90 degrees or π/2 radians) -/
noncomputable def RotationAngle : ℝ := Real.pi / 2

/-- Calculates the length of the path traversed by vertex P -/
noncomputable def pathLength : ℝ :=
  (NumRotations : ℝ) * (Hypotenuse * RotationAngle)

/-- Theorem stating that the total path length is 6π inches -/
theorem total_path_length :
  pathLength = 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_path_length_l514_51422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_cone_l514_51423

-- Define the cone parameters
def cone_height : ℝ := 4
def cone_slant_height : ℝ := 8

-- Define the Sphere structure
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Helper definitions (not proved)
def sphere_in_cone (s : Sphere) (h : ℝ) (sl : ℝ) : Prop := sorry
def spheres_touch_externally (s1 s2 : Sphere) : Prop := sorry
def sphere_touches_cone_lateral (s : Sphere) (h : ℝ) (sl : ℝ) : Prop := sorry
def sphere_touches_cone_base (s : Sphere) (h : ℝ) (sl : ℝ) : Prop := sorry

-- Define the theorem
theorem max_sphere_radius_in_cone :
  let max_radius := (12 : ℝ) / (5 + 2 * Real.sqrt 3)
  ∀ r : ℝ,
    r > 0 →
    (∃ (s₁ s₂ s₃ : Sphere),
      -- Spheres have radius r
      s₁.radius = r ∧ s₂.radius = r ∧ s₃.radius = r ∧
      -- Spheres are inside the cone
      sphere_in_cone s₁ cone_height cone_slant_height ∧
      sphere_in_cone s₂ cone_height cone_slant_height ∧
      sphere_in_cone s₃ cone_height cone_slant_height ∧
      -- Spheres touch each other externally
      spheres_touch_externally s₁ s₂ ∧
      spheres_touch_externally s₁ s₃ ∧
      spheres_touch_externally s₂ s₃ ∧
      -- Spheres touch the lateral surface of the cone
      sphere_touches_cone_lateral s₁ cone_height cone_slant_height ∧
      sphere_touches_cone_lateral s₂ cone_height cone_slant_height ∧
      sphere_touches_cone_lateral s₃ cone_height cone_slant_height ∧
      -- First two spheres touch the base of the cone
      sphere_touches_cone_base s₁ cone_height cone_slant_height ∧
      sphere_touches_cone_base s₂ cone_height cone_slant_height) →
    r ≤ max_radius :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_cone_l514_51423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_pi_over_six_l514_51464

theorem tan_a_pi_over_six (a : ℝ) (h : (3 : ℝ)^a = 9) : Real.tan (a * π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_pi_over_six_l514_51464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_at_point_l514_51428

noncomputable section

/-- The function f(x) = x^4 - 2x --/
def f (x : ℝ) : ℝ := x^4 - 2*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 4*x^3 - 2

/-- The slope of the line x + 2y + 1 = 0 --/
def m : ℝ := -1/2

theorem tangent_perpendicular_at_point :
  ∃ x₀ y₀ : ℝ, x₀ = 1 ∧ y₀ = -1 ∧ f x₀ = y₀ ∧ f' x₀ * m = -1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_at_point_l514_51428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l514_51496

noncomputable def f (x : ℝ) := Real.log x + (1/2) * x - 2

theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l514_51496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_is_20_l514_51473

-- Define the heights of the poles and the distance between them
noncomputable def pole1_height : ℝ := 30
noncomputable def pole2_height : ℝ := 60
noncomputable def distance_between_poles : ℝ := 120

-- Define the slopes of the lines
noncomputable def slope1 : ℝ := (0 - pole1_height) / distance_between_poles
noncomputable def slope2 : ℝ := (0 - pole2_height) / (-distance_between_poles)

-- Define the y-intercepts of the lines
noncomputable def y_intercept1 : ℝ := pole1_height
noncomputable def y_intercept2 : ℝ := 0

-- Define the function to calculate the intersection point
noncomputable def intersection_point : ℝ × ℝ :=
  let x := (y_intercept2 - y_intercept1) / (slope1 - slope2)
  let y := slope1 * x + y_intercept1
  (x, y)

-- Theorem stating that the height of intersection is 20''
theorem intersection_height_is_20 :
  (intersection_point.2 : ℝ) = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_is_20_l514_51473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l514_51462

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

noncomputable def sum_arithmetic_sequence (a₁ aₙ : ℝ) (n : ℕ) : ℝ := (a₁ + aₙ) / 2 * n

noncomputable def f (a b c : ℝ) (n : ℕ) : ℝ := a * (2 * n + 1)^2 + b * (2 * n + 1) + c

theorem solution_exists : ∃ (a b c : ℝ),
  (∀ n : ℕ, f a b c n = sum_arithmetic_sequence 3 (arithmetic_sequence 3 2 n) n) ∧
  a = (1 : ℝ) / 4 ∧ b = (1 : ℝ) / 2 ∧ c = -(3 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l514_51462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eisenstein_criterion_l514_51491

theorem eisenstein_criterion (f : Polynomial ℤ) (p : ℕ) (hp : Nat.Prime p) :
  (∃ n : ℕ, n > 0 ∧ f.degree = n) →
  (∀ i : ℕ, i < f.degree → (p : ℤ) ∣ f.coeff i) →
  ¬((p : ℤ) ∣ f.leadingCoeff) →
  ¬((p : ℤ)^2 ∣ f.coeff 0) →
  Irreducible (f.map (algebraMap ℤ ℚ)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eisenstein_criterion_l514_51491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_ones_value_probability_two_ones_decimal_value_l514_51420

/-- The probability of rolling exactly two 1s when rolling twelve standard 6-sided dice -/
def probability_two_ones : ℚ :=
  (66 * 5^10 : ℚ) / 6^12

/-- Theorem stating that the probability of rolling exactly two 1s
    when rolling twelve standard 6-sided dice is equal to (66 * 5^10) / 6^12 -/
theorem probability_two_ones_value :
  probability_two_ones = (66 * 5^10 : ℚ) / 6^12 := by
  rfl

/-- The decimal approximation of the probability, rounded to three decimal places -/
def probability_two_ones_decimal : ℚ :=
  294/1000

/-- Theorem stating that the decimal approximation of the probability,
    rounded to three decimal places, is equal to 0.294 -/
theorem probability_two_ones_decimal_value :
  (probability_two_ones * 1000).floor / 1000 = probability_two_ones_decimal := by
  sorry

#eval (probability_two_ones * 1000).floor / 1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_ones_value_probability_two_ones_decimal_value_l514_51420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_with_at_least_6_black_l514_51426

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  topLeft : Nat × Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 9

/-- Checks if a square contains at least 6 black squares -/
def hasAtLeast6BlackSquares (s : Square) : Bool :=
  s.size ≥ 4

/-- Counts the number of valid positions for a square of given size -/
def countValidPositions (size : Nat) : Nat :=
  (boardSize - size + 1) * (boardSize - size + 1)

/-- Counts the total number of squares with at least 6 black squares -/
def countSquaresWithAtLeast6Black : Nat :=
  List.sum (List.map (fun i => countValidPositions (i + 4)) (List.range 6))

theorem count_squares_with_at_least_6_black :
  countSquaresWithAtLeast6Black = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_with_at_least_6_black_l514_51426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_specific_angles_l514_51447

theorem cos_sum_specific_angles (α β : Real) :
  Real.sin α = 4/5 →
  α ∈ Set.Ioo (π/2) π →
  Real.cos β = -5/13 →
  β ∈ Set.Ioo π (3*π/2) →
  Real.cos (α + β) = -33/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_specific_angles_l514_51447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_rational_unit_polygon_l514_51400

/-- A point on the coordinate plane with rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- A closed polygonal chain on the coordinate plane -/
structure ClosedPolygonalChain where
  vertices : List RationalPoint
  closed : vertices.head? = vertices.getLast?
  unitLength : ∀ i : Fin (vertices.length - 1), 
    (vertices[i.val].x - vertices[i.val.succ].x)^2 + (vertices[i.val].y - vertices[i.val.succ].y)^2 = 1

theorem no_odd_rational_unit_polygon : 
  ∀ n : ℕ, Odd n → n ≥ 3 → ¬ ∃ chain : ClosedPolygonalChain, chain.vertices.length = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_rational_unit_polygon_l514_51400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_between_chords_proof_l514_51472

noncomputable def circle_area_between_chords (R : ℝ) : ℝ :=
  (R^2 * (Real.pi + Real.sqrt 3)) / 2

theorem circle_area_between_chords_proof (R : ℝ) (h : R > 0) :
  circle_area_between_chords R = (R^2 * (Real.pi + Real.sqrt 3)) / 2 := by
  -- Unfold the definition
  unfold circle_area_between_chords
  -- The rest of the proof would go here
  sorry

#check circle_area_between_chords_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_between_chords_proof_l514_51472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_magnitude_l514_51465

variable (a b : ℝ × ℝ)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_b_magnitude
  (h1 : magnitude (a.1 - b.1, a.2 - b.2) = 3)
  (h2 : magnitude (a.1 + 2 * b.1, a.2 + 2 * b.2) = 6)
  (h3 : a.1 ^ 2 + a.2 ^ 2 + a.1 * b.1 + a.2 * b.2 - 2 * (b.1 ^ 2 + b.2 ^ 2) = -9) :
  magnitude b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_magnitude_l514_51465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l514_51463

-- Define the function
noncomputable def y (x : ℝ) : ℝ := 
  -Real.tan (x + 2*Real.pi/3) - Real.tan (x + Real.pi/6) + Real.cos (x + Real.pi/6)

-- Define the interval
def I : Set ℝ := {x | -Real.pi/12 ≤ x ∧ x ≤ -Real.pi/3}

-- State the theorem
theorem max_value_of_y :
  ∃ (x : ℝ), x ∈ I ∧ y x = (11/6) * Real.sqrt 3 ∧ ∀ (z : ℝ), z ∈ I → y z ≤ y x := by
  sorry

#check max_value_of_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l514_51463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l514_51452

-- Define the logarithm values given in the problem
noncomputable def log2 : ℝ := 0.3010
noncomputable def log3 : ℝ := 0.4771

-- Define the equation
def equation (x : ℝ) : Prop := (3 : ℝ)^(x + 3) = 135

-- State the theorem
theorem equation_solution :
  ∃ x : ℝ, equation x ∧ abs (x - 1.47) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l514_51452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_six_l514_51486

def modifiedFibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => modifiedFibonacci n + modifiedFibonacci (n + 1)

def unitDigit (n : ℕ) : ℕ := n % 10

def allDigitsAppearBefore (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → d ≠ 6 → ∃ k : ℕ, k < n ∧ unitDigit (modifiedFibonacci k) = d

theorem last_digit_is_six :
  ∃ n : ℕ, 
    (unitDigit (modifiedFibonacci n) = 6) ∧ 
    (allDigitsAppearBefore n) ∧ 
    (∀ k : ℕ, k < n → unitDigit (modifiedFibonacci k) ≠ 6) :=
by
  sorry

#eval modifiedFibonacci 19
#eval unitDigit (modifiedFibonacci 19)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_six_l514_51486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_nonempty_proper_subsets_l514_51427

open Set Finset

def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {4, 5}
def M : Finset ℕ := A.biUnion (λ a => B.image (λ b => a + b))

theorem number_of_nonempty_proper_subsets : (M.powerset.filter (λ s => s ≠ ∅ ∧ s ≠ M)).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_nonempty_proper_subsets_l514_51427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_visibility_l514_51430

/-- The radius of a circle concentric with and outside a regular hexagon,
    such that the probability of seeing three sides from a random point on the circle is 1/2 -/
noncomputable def circle_radius (hexagon_side : ℝ) : ℝ :=
  3 * Real.sqrt 2 + Real.sqrt 6

/-- Axioms for necessary concepts -/
axiom IsRegularPolygon (n : ℕ) (side : ℝ) : Prop
axiom ConcentricWithHexagon (side : ℝ) : Prop
axiom CircleOutsideHexagon (side : ℝ) : Prop
axiom ProbabilityOfSeeingThreeSides (side : ℝ) : ℝ

/-- Theorem stating the radius of the circle given the conditions -/
theorem hexagon_circle_visibility
  (hexagon_side : ℝ)
  (hexagon_regular : IsRegularPolygon 6 hexagon_side)
  (circle_concentric : ConcentricWithHexagon hexagon_side)
  (circle_outside : CircleOutsideHexagon hexagon_side)
  (visibility_prob : ProbabilityOfSeeingThreeSides hexagon_side = 1/2) :
  circle_radius hexagon_side = 3 * Real.sqrt 2 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_visibility_l514_51430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_for_given_circle_l514_51492

noncomputable def arcLength (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (centralAngle * Real.pi * radius) / 180

theorem arc_length_for_given_circle : 
  arcLength 10 240 = (40 / 3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_for_given_circle_l514_51492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probability_theorem_l514_51450

/-- The total number of balls in the box -/
def total_balls : ℕ := 60

/-- The observed frequency of picking a white ball -/
def observed_frequency : ℚ := 1/4

/-- The desired probability of picking a white ball after adding more white balls -/
def desired_probability : ℚ := 2/5

/-- The estimated probability of picking a white ball based on the observed frequency -/
def estimated_probability : ℚ := observed_frequency

/-- The number of additional white balls needed to achieve the desired probability -/
def additional_white_balls : ℕ := 15

/-- Theorem stating the estimated probability and the number of additional white balls needed -/
theorem ball_probability_theorem :
  (estimated_probability = observed_frequency) ∧
  ((↑((total_balls * observed_frequency).floor + additional_white_balls) / ↑(total_balls + additional_white_balls) : ℚ) = desired_probability) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probability_theorem_l514_51450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l514_51493

open Real

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x - y) * (f (x + y)) - (x + y) * (f (x - y)) = 4 * x * y * (x^2 - y^2)

/-- The theorem stating that any function satisfying the functional equation
    must be of the form f(x) = x³ + cx for some constant c -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f →
  ∃ c : ℝ, ∀ x : ℝ, f x = x^3 + c * x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l514_51493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_min_workers_l514_51424

/-- Represents a construction project with its parameters and progress -/
structure Project where
  totalDays : ℕ
  workedDays : ℕ
  completionPercentage : ℚ
  initialWorkers : ℕ

/-- Calculates the minimum number of workers needed to complete the project on time -/
def minWorkersNeeded (p : Project) : ℕ :=
  let remainingWork := 1 - p.completionPercentage
  let remainingDays := p.totalDays - p.workedDays
  let dailyProgressNeeded := remainingWork / remainingDays
  let initialDailyProgress := p.completionPercentage / p.workedDays
  (((dailyProgressNeeded / initialDailyProgress) * p.initialWorkers).ceil).toNat

/-- Theorem stating that for the given project parameters, the minimum number of workers needed is 6 -/
theorem project_min_workers :
  let p : Project := {
    totalDays := 40,
    workedDays := 10,
    completionPercentage := 2/5,
    initialWorkers := 12
  }
  minWorkersNeeded p = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_min_workers_l514_51424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_zero_l514_51411

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (Real.cos x ^ 4 + 9 * Real.sin x ^ 2) - Real.sqrt (Real.sin x ^ 4 + 9 * Real.cos x ^ 2)

theorem g_is_zero : ∀ x : ℝ, g x = 0 := by
  intro x
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_zero_l514_51411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_l514_51471

noncomputable section

-- Define the coordinate systems
def CartesianSystem : Type := ℝ × ℝ
def PolarSystem : Type := ℝ × ℝ  -- (ρ, θ)

-- Define the circle C
def CircleC (p : CartesianSystem) : Prop :=
  let (x, y) := p
  x^2 + y^2 + 2*x - 2*y = 0

-- Define the line l
def LineL (p : CartesianSystem) : Prop :=
  let (x, y) := p
  x - y + 1 = 0

-- Define the ray OM
def RayOM (p : PolarSystem) : Prop :=
  let (_, θ) := p
  θ = 3*Real.pi/4

-- Define the point P
noncomputable def PointP : PolarSystem :=
  (2*Real.sqrt 2, 3*Real.pi/4)

-- Define the point Q
noncomputable def PointQ : PolarSystem :=
  (Real.sqrt 2/2, 3*Real.pi/4)

-- Theorem statement
theorem length_PQ :
  let (ρ_P, θ_P) := PointP
  let (ρ_Q, θ_Q) := PointQ
  Real.sqrt ((ρ_P * Real.cos θ_P - ρ_Q * Real.cos θ_Q)^2 +
             (ρ_P * Real.sin θ_P - ρ_Q * Real.sin θ_Q)^2) = 3*Real.sqrt 2/2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_l514_51471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_quantity_l514_51484

/-- The total annual cost function for a company's purchases and storage. -/
noncomputable def C (x : ℝ) : ℝ := (16000000 / x) + 40000 * x

/-- The optimal purchase quantity minimizes the total annual cost. -/
theorem optimal_purchase_quantity :
  ∃ (x : ℝ), x > 0 ∧ (∀ (y : ℝ), y > 0 → C y ≥ C x) ∧ x = 20 := by
  sorry

#check optimal_purchase_quantity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_quantity_l514_51484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_phase_shift_phase_shift_equivalence_l514_51438

/-- The phase shift of the cosine function y = cos(5x - 5π/6) -/
noncomputable def phase_shift : ℝ := Real.pi / 6

/-- The cosine function with given coefficients -/
noncomputable def f (x : ℝ) : ℝ := Real.cos (5 * x - 5 * Real.pi / 6)

theorem cosine_phase_shift :
  ∀ x : ℝ, f (x + phase_shift) = Real.cos (5 * x) := by
  sorry

theorem phase_shift_equivalence :
  ∀ x : ℝ, f (x + phase_shift) = f (x - phase_shift) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_phase_shift_phase_shift_equivalence_l514_51438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_equals_sqrt_58_l514_51446

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 0]
def b : Fin 2 → ℝ := ![2, 1]

-- Define the vector addition and scalar multiplication
def vector_add (u v : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => u i + v i
def scalar_mult (c : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => c * v i

-- Define the magnitude (norm) of a 2D vector
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ := Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

theorem magnitude_of_sum_equals_sqrt_58 :
  magnitude (vector_add a (scalar_mult 3 b)) = Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_equals_sqrt_58_l514_51446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_never_zero_l514_51436

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => if a (n + 1) % 2 ≠ 0 ∧ a n % 2 ≠ 0 then
                 5 * a (n + 1) - 3 * a n
               else
                 a (n + 1) - a n

theorem a_never_zero : ∀ n : ℕ, a n ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_never_zero_l514_51436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_balance_possible_l514_51458

/-- A pair of weights -/
structure WeightPair where
  first : ℕ
  second : ℕ
  diff_bound : second - first ≤ 20

/-- A collection of weight pairs -/
def WeightCollection := List WeightPair

/-- The state of the balance scale -/
structure BalanceScale where
  left : ℕ
  right : ℕ

/-- Function to place weights on the balance scale -/
def place_weights : WeightCollection → BalanceScale
  | _ => { left := 0, right := 0 }  -- Placeholder implementation

/-- Theorem stating that it's always possible to place weights
    such that the difference never exceeds 20 -/
theorem weight_balance_possible (weights : WeightCollection) :
  let final_state := place_weights weights
  (final_state.left : Int) - (final_state.right : Int) ≤ 20 ∧
  (final_state.right : Int) - (final_state.left : Int) ≤ 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_balance_possible_l514_51458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_l514_51467

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) : Prop :=
  ∀ x, deriv y x + y x * Real.tan x = 0

-- Define the initial condition
def initial_condition (y : ℝ → ℝ) : Prop :=
  y 0 = 2

-- State the theorem
theorem particular_solution :
  ∃ y : ℝ → ℝ, diff_eq y ∧ initial_condition y ∧ (∀ x, y x = 2 * Real.cos x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_l514_51467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l514_51479

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a ≥ 1
  h2 : a ≤ 9
  h3 : b ≤ 9
  h4 : c ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def toNum (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- Defines the function f(n) as specified in the problem -/
def f (n : ThreeDigitNumber) : Nat :=
  (n.a + n.b + n.c) + (n.a * n.b + n.b * n.c + n.c * n.a) + n.a * n.b * n.c

/-- The first part of the theorem -/
theorem part1 :
  ∃ (n : ThreeDigitNumber), toNum n = 625 ∧ (toNum n / f n = 5) := by sorry

/-- The second part of the theorem -/
theorem part2 :
  ∀ (n : ThreeDigitNumber), (toNum n / f n = 1) ↔
    toNum n ∈ ({199, 299, 399, 499, 599, 699, 799, 899, 999} : Set Nat) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l514_51479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_weight_change_l514_51415

/-- Calculates the final weight of a glass with solution after water evaporation --/
noncomputable def final_weight (initial_weight : ℝ) (glass_weight : ℝ) (initial_water_percent : ℝ) (final_water_percent : ℝ) : ℝ :=
  let initial_solution_weight := initial_weight - glass_weight
  let initial_water_weight := initial_solution_weight * initial_water_percent
  let solute_weight := initial_solution_weight - initial_water_weight
  let final_solution_weight := solute_weight / (1 - final_water_percent)
  final_solution_weight + glass_weight

/-- Theorem stating that under given conditions, the final weight is 400 grams --/
theorem evaporation_weight_change :
  final_weight 500 300 0.99 0.98 = 400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_weight_change_l514_51415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l514_51425

/-- The function f(x) = x^2 + 11x - 5 -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 11*x - 5

/-- The point where f attains its minimum -/
noncomputable def min_point : ℝ := -11/2

theorem f_minimum :
  ∀ x : ℝ, f x ≥ f min_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l514_51425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_implies_y_equals_81_l514_51457

theorem log_equation_implies_y_equals_81 (m y : ℝ) (hm : m > 0) (hy : y > 0) :
  (Real.log y / Real.log m) * (Real.log m / Real.log 3) = 4 → y = 81 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_implies_y_equals_81_l514_51457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l514_51437

-- Define the set of x-values satisfying the given conditions
def S : Set ℝ := {x | 3 ≤ |x - 2| ∧ |x - 2| ≤ 7 ∧ x^2 ≤ 36}

-- Define the expected result set
def R : Set ℝ := Set.Icc (-5) (-1) ∪ Set.Icc 5 6

-- Theorem stating that S equals R
theorem solution_set_equality : S = R := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l514_51437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_product_l514_51413

noncomputable def product_of_real_imaginary_parts (z : ℂ) : ℝ :=
  z.re * z.im

theorem complex_fraction_product : 
  let z : ℂ := (2 + 3*Complex.I) / (1 + Complex.I)
  product_of_real_imaginary_parts z = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_product_l514_51413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l514_51412

theorem inscribed_square_side_length 
  (a h : ℝ) (ha : 0 < a) (hh : 0 < h) :
  ∃ x : ℝ, x = (a * h) / (a + h) ∧ 
  x > 0 ∧ 
  x < a ∧ 
  x < h := by
  -- Let x be the side length of the inscribed square
  let x := (a * h) / (a + h)
  
  -- Show that x satisfies the equation
  have hx : x = (a * h) / (a + h) := rfl
  
  -- Show that 0 < x < a and 0 < x < h
  have hx_pos : 0 < x := by
    apply div_pos
    · exact mul_pos ha hh
    · exact add_pos ha hh
  
  have hx_lt_a : x < a := by
    sorry -- Proof to be completed
  
  have hx_lt_h : x < h := by
    sorry -- Proof to be completed
  
  -- Conclude the proof
  exact ⟨x, hx, hx_pos, hx_lt_a, hx_lt_h⟩

#check inscribed_square_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l514_51412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_known_child_is_boy_l514_51409

/-- Represents the gender of a child -/
inductive Gender
| Boy
| Girl

/-- Represents a pair of children -/
structure ChildPair :=
  (first : Gender)
  (second : Gender)

/-- The probability of a specific child pair -/
noncomputable def probability (pair : ChildPair) : ℝ := sorry

/-- Assumption: The probability of having two boys is 0.5 -/
axiom prob_two_boys : probability (ChildPair.mk Gender.Boy Gender.Boy) = 0.5

/-- Assumption: The total probability of all possible child pairs is 1 -/
axiom total_probability :
  probability (ChildPair.mk Gender.Boy Gender.Boy) +
  probability (ChildPair.mk Gender.Boy Gender.Girl) +
  probability (ChildPair.mk Gender.Girl Gender.Boy) +
  probability (ChildPair.mk Gender.Girl Gender.Girl) = 1

/-- The known child's gender -/
def known_child : Gender := sorry

/-- Theorem: Given the conditions, the known child must be a boy -/
theorem known_child_is_boy : known_child = Gender.Boy := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_known_child_is_boy_l514_51409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_value_l514_51478

theorem cos_sin_value (α : ℝ) (h : Real.tan α = Real.sqrt 2) : 
  Real.cos α * Real.sin α = (Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_value_l514_51478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_of_7200_l514_51441

theorem gcd_of_polynomial_and_multiple_of_7200 (x : ℤ) (h : 7200 ∣ x) :
  Int.gcd ((5 * x + 3) * (11 * x + 2) * (17 * x + 5) * (4 * x + 7)) x.natAbs = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_of_7200_l514_51441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_midpoint_implies_p_eq_four_l514_51445

/-- A parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- The focus of a parabola -/
noncomputable def focus (c : Parabola) : ℝ × ℝ := (c.p / 2, 0)

/-- A point on the parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * c.p * x

/-- The midpoint of two points -/
noncomputable def midpoint_parabola (a b : ℝ × ℝ) : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

theorem parabola_focus_midpoint_implies_p_eq_four (c : Parabola) 
  (m : PointOnParabola c) :
  midpoint_parabola (m.x, m.y) (focus c) = (2, 2) → c.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_midpoint_implies_p_eq_four_l514_51445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l514_51489

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1

def g (x : ℝ) : ℝ := -x^2 + 4*x - 3

theorem range_of_b (a b : ℝ) (h : f a = g b) :
  b ∈ Set.Icc (2 - Real.sqrt 2) (2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l514_51489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_dynamics_index_difference_l514_51487

theorem group_dynamics_index_difference :
  let n : ℚ := 20  -- Total number of people
  let k : ℚ := 7   -- Number of females
  let index (total : ℚ) (subgroup : ℚ) := (total - subgroup) / total
  (index n k - index n (n - k)) = 0.30 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_dynamics_index_difference_l514_51487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l514_51455

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states properties of a specific triangle. -/
theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos t.B = 3/5)
  (h2 : t.a * t.c * Real.cos t.B = -21)
  : 
  /- The area of the triangle is 14 -/
  (1/2 * t.a * t.c * Real.sin t.B = 14) ∧ 
  /- If a = 7, then angle C = π/4 -/
  (t.a = 7 → t.C = π/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l514_51455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_higher_prob_l514_51494

/-- The probability that a ball lands in bin k -/
noncomputable def prob_in_bin (base : ℝ) (k : ℕ+) : ℝ := base^(-(k : ℝ))

/-- The probability that both balls land in the same bin k -/
noncomputable def prob_same_bin (k : ℕ+) : ℝ := (prob_in_bin 2 k) * (prob_in_bin 3 k)

theorem yellow_higher_prob :
  let prob_yellow_higher : ℝ := (1 - ∑' k, prob_same_bin k) / 2
  prob_yellow_higher = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_higher_prob_l514_51494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dotted_square_area_l514_51483

/-- A square with 7 dots on each side, divided into 6 equal segments --/
structure DottedSquare where
  side_length : ℝ
  segment_count : ℕ
  dot_count : ℕ
  segment_length : ℝ
  h_segment_count : segment_count = 6
  h_dot_count : dot_count = 7
  h_side_length : side_length = segment_count * segment_length

/-- The shaded squares formed by 45° line segments --/
noncomputable def shaded_squares (ds : DottedSquare) : ℝ :=
  (ds.dot_count - 1) * (ds.dot_count - 2) / 2

/-- The area of a single shaded square --/
noncomputable def shaded_square_area (ds : DottedSquare) : ℝ :=
  ds.segment_length^2 / 2

/-- The theorem to be proved --/
theorem dotted_square_area (ds : DottedSquare) 
  (h_shaded_area : shaded_squares ds * shaded_square_area ds = 75) :
  ds.side_length^2 = 360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dotted_square_area_l514_51483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l514_51495

-- Define the function f(x) = 1 / √(x-1)
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x - 1)

-- Theorem: The domain of f is x > 1
theorem f_domain : ∀ x : ℝ, x > 1 ↔ ∃ y : ℝ, f x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l514_51495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_next_point_bisection_next_eval_correct_l514_51414

/-- Bisection method next point theorem -/
theorem bisection_next_point (f : ℝ → ℝ) (a b : ℝ) 
  (ha : f a > 0) (hb : f b < 0) (hmid : f ((a + b) / 2) < 0) :
  ∃ x ∈ Set.Icc ((a + b) / 2) a, f x = 0 :=
by sorry

/-- The next point to evaluate in the bisection method -/
noncomputable def bisection_next_eval (a b : ℝ) : ℝ := (3 * a + b) / 4

/-- Theorem stating that (3a+b)/4 is the next point to evaluate -/
theorem bisection_next_eval_correct (f : ℝ → ℝ) (a b : ℝ) 
  (ha : f a > 0) (hb : f b < 0) (hmid : f ((a + b) / 2) < 0) :
  bisection_next_eval a b ∈ Set.Icc ((a + b) / 2) a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_next_point_bisection_next_eval_correct_l514_51414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l514_51448

-- Define the right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

-- Define the circles
structure CirclePair where
  r : ℝ
  triangle : RightTriangle
  circumscribed : r = triangle.c / 2
  tangent_to_legs : r = triangle.a

-- Define the ratio
noncomputable def area_ratio (cp : CirclePair) : ℝ :=
  (3 * Real.sqrt 3) / (5 * Real.pi - 3)

-- State the theorem
theorem area_ratio_theorem (cp : CirclePair) :
  area_ratio cp = (cp.triangle.a * cp.triangle.b / 2) /
    (cp.r^2 * ((5 * Real.pi / 12) - 1/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l514_51448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_field_dimensions_optimal_field_area_l514_51474

/-- 
Theorem: For a rectangular field bounded by a river on one side, 
the maximum area is achieved when the length along the river is 
twice the width perpendicular to the river.
-/
theorem optimal_field_dimensions (p : ℝ) (h : p > 0) :
  let width := p / 4
  let length := p / 2
  ∀ w l, w > 0 → l > 0 → w + 2 * l = p → w * l ≤ width * length := by
  sorry  -- Proof to be filled in

/--
The area of the optimal field is p^2 / 8.
-/
theorem optimal_field_area (p : ℝ) (h : p > 0) :
  (p / 4) * (p / 2) = p^2 / 8 := by
  sorry  -- Proof to be filled in

#check optimal_field_dimensions
#check optimal_field_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_field_dimensions_optimal_field_area_l514_51474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l514_51466

-- Part 1
theorem sin_bounds (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := by
  sorry

-- Part 2
noncomputable def f (a x : ℝ) : ℝ := Real.cos (a * x) - Real.log (1 - x^2)

theorem local_max_condition (a : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → f a x < f a 0) ↔
  a < -Real.sqrt 2 ∨ a > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l514_51466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pirates_l514_51499

/-- The number of raiders -/
def R : ℕ := sorry

/-- The number of sailors -/
def S : ℕ := sorry

/-- The number of cabin boys -/
def C : ℕ := sorry

/-- The total number of gold coins -/
def total_gold : ℕ := 200

/-- The total number of silver coins -/
def total_silver : ℕ := 600

/-- Gold coins distribution equation -/
axiom gold_distribution : 5 * R + 3 * S + C = total_gold

/-- Silver coins distribution equation -/
axiom silver_distribution : 10 * R + 8 * S + 6 * C = total_silver

/-- Theorem: The total number of pirates is 80 -/
theorem total_pirates : R + S + C = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pirates_l514_51499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_number_property_l514_51444

-- Define the harmonic number sequence
def H : ℕ → ℚ
  | 0 => 0
  | n + 1 => H n + 1 / (n + 1)

-- Define a predicate for irreducible fractions with odd numerator and even denominator
def OddNumeratorEvenDenominator (q : ℚ) : Prop :=
  ∃ (a b : ℕ), q = a / b ∧ Odd a ∧ Even b ∧ Nat.Coprime a b

-- State the theorem
theorem harmonic_number_property : ∀ n : ℕ, n ≥ 2 → OddNumeratorEvenDenominator (H n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_number_property_l514_51444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charlyn_visible_area_l514_51407

noncomputable section

/-- The area of the region visible to a person walking around a rectangle --/
noncomputable def visibleArea (length width viewDistance : ℝ) : ℝ :=
  let interiorArea := (length - 2 * viewDistance) * (width - 2 * viewDistance)
  let stripArea := 2 * (length * viewDistance) + 2 * (width * viewDistance)
  let cornerArea := 4 * (Real.pi * viewDistance^2 / 4)
  interiorArea + stripArea + cornerArea

/-- Theorem stating the visible area for Charlyn's walk --/
theorem charlyn_visible_area :
  visibleArea 8 4 1.5 = 5 + 36 + 2.25 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charlyn_visible_area_l514_51407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sine_curve_l514_51460

noncomputable def T (ω : ℝ) : ℝ := 2 * Real.pi / ω

noncomputable def curve (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem period_of_sine_curve (ω : ℝ) (x₀ : ℝ) :
  ω > 0 →
  curve ω x₀ = 2 →
  x₀ * 1 + 2 * 0 = 1 →
  0 < x₀ →
  x₀ < T ω →
  T ω = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sine_curve_l514_51460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_A_winning_l514_51469

/-- Represents the outcome of a single chess game -/
inductive GameResult
| Win
| Draw
| Loss

/-- Calculates the points for a given game result -/
def pointsForResult (result : GameResult) : ℕ :=
  match result with
  | GameResult.Win => 2
  | GameResult.Draw => 1
  | GameResult.Loss => 0

/-- Represents the results of a three-game chess match -/
structure MatchResult where
  game1 : GameResult
  game2 : GameResult
  game3 : GameResult

/-- Calculates the total points for a match result -/
def totalPoints (m : MatchResult) : ℕ :=
  pointsForResult m.game1 + pointsForResult m.game2 + pointsForResult m.game3

/-- Determines if player A wins the match -/
def playerAWins (m : MatchResult) : Prop :=
  totalPoints m > 3

/-- The set of all possible match results -/
def allMatchResults : Finset MatchResult := sorry

/-- The number of match results where player A wins -/
def winningOutcomes : ℕ := sorry

/-- The total number of possible match results -/
def totalOutcomes : ℕ := sorry

theorem probability_of_A_winning :
  (winningOutcomes : ℚ) / totalOutcomes = 10 / 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_A_winning_l514_51469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_sum_l514_51419

theorem cos_double_sum (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_sum_l514_51419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_rectangle_l514_51416

/-- A quadrilateral with vertices A, B, C, and D -/
structure Quadrilateral (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] :=
  (A B C D : P)

/-- Predicate to check if two line segments are parallel -/
def IsParallel {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (seg1 seg2 : P × P) : Prop :=
  sorry

/-- Predicate to check if two line segments have equal length -/
def EqualLength {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (seg1 seg2 : P × P) : Prop :=
  sorry

/-- Predicate to check if two angles are equal -/
def EqualAngles {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (ang1 ang2 : P × P × P) : Prop :=
  sorry

/-- Predicate to check if a quadrilateral is a rectangle -/
def IsRectangle {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (quad : Quadrilateral P) : Prop :=
  sorry

/-- Theorem: If in quadrilateral ABCD, AD is parallel to BC, AB = CD, and ∠A = ∠B, then ABCD is a rectangle -/
theorem quadrilateral_is_rectangle {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (quad : Quadrilateral P) :
  IsParallel (quad.A, quad.D) (quad.B, quad.C) →
  EqualLength (quad.A, quad.B) (quad.C, quad.D) →
  EqualAngles (quad.B, quad.A, quad.D) (quad.A, quad.B, quad.C) →
  IsRectangle quad :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_rectangle_l514_51416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_fourth_power_l514_51435

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom fg_condition : ∀ x, x ≥ 1 → f (g x) = x^3
axiom gf_condition : ∀ x, x ≥ 1 → g (f x) = x^4
axiom g_81 : g 81 = 81

-- State the theorem to be proved
theorem g_3_fourth_power : (g 3)^4 = 531441 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_fourth_power_l514_51435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_augmented_matrix_of_system_l514_51454

def system_of_equations : List (List ℝ) := [[2, -1, 1], [1, 3, 2]]

def augmented_matrix : List (List ℝ) := [[2, -1, 1], [1, 3, -2]]

theorem augmented_matrix_of_system :
  system_of_equations.map (λ eq => eq.take 2 ++ [eq.getLast!]) = augmented_matrix := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_augmented_matrix_of_system_l514_51454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_girls_at_outing_l514_51408

theorem fraction_of_girls_at_outing : 
  let total_a := 300
  let total_b := 240
  let ratio_boys_girls_a := 3/2
  let ratio_girls_boys_b := 3/2
  (total_a * (1 / (1 + ratio_boys_girls_a)) + 
   total_b * (ratio_girls_boys_b / (1 + ratio_girls_boys_b))) / 
  (total_a + total_b) = 22/45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_girls_at_outing_l514_51408
