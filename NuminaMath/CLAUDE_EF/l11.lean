import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_l11_1154

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line passing through M(-2,0) with slope k₁
def line (x y k₁ : ℝ) : Prop := y = k₁ * (x + 2)

-- Define the intersection points
def intersection (x₁ y₁ x₂ y₂ k₁ : ℝ) : Prop :=
  line x₁ y₁ k₁ ∧ line x₂ y₂ k₁ ∧ ellipse x₁ y₁ ∧ ellipse x₂ y₂

-- Define the midpoint of the intersection
def midpoint_of_intersection (x y x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

-- The main theorem
theorem slope_product (k₁ k₂ x₁ y₁ x₂ y₂ x y : ℝ) :
  k₁ ≠ 0 →
  intersection x₁ y₁ x₂ y₂ k₁ →
  midpoint_of_intersection x y x₁ y₁ x₂ y₂ →
  k₂ = y / x →
  k₁ * k₂ = -1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_l11_1154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_total_spend_l11_1189

noncomputable def peach_price : ℝ := 12.32 + 3
def peach_weight : ℝ := 3

def cherry_price : ℝ := 11.54
def cherry_weight : ℝ := 2

def apple_price : ℝ := 5
def apple_weight : ℝ := 4
def apple_discount : ℝ := 0.15

def orange_price : ℝ := 1.25
def orange_count : ℕ := 6

noncomputable def total_cost : ℝ := peach_price + 
                      cherry_price * cherry_weight + 
                      apple_price * apple_weight * (1 - apple_discount) + 
                      orange_price * (2 * (orange_count / 3))

theorem sally_total_spend : total_cost = 60.40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_total_spend_l11_1189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_divisible_into_three_trapezoids_l11_1164

-- Define a point in 2D space
structure Point :=
  (x y : ℝ)

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a trapezoid
structure Trapezoid :=
  (A B C D : Point)
  (is_trapezoid : Bool) -- We'll assume this property for now

-- Define a set of points for a shape
def points_of (shape : Quadrilateral ⊕ Trapezoid) : Set Point :=
  match shape with
  | Sum.inl q => {q.A, q.B, q.C, q.D}
  | Sum.inr t => {t.A, t.B, t.C, t.D}

-- Theorem statement
theorem quadrilateral_divisible_into_three_trapezoids (Q : Quadrilateral) :
  ∃ (T1 T2 T3 : Trapezoid), 
    (points_of (Sum.inr T1) ∪ points_of (Sum.inr T2) ∪ points_of (Sum.inr T3)) = 
    points_of (Sum.inl Q) :=
by
  sorry -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_divisible_into_three_trapezoids_l11_1164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_work_hours_l11_1134

theorem total_work_hours (amber_hours : ℕ) (armand_fraction : ℚ) (ella_multiplier : ℕ) : 
  amber_hours = 12 →
  armand_fraction = 1/3 →
  ella_multiplier = 2 →
  amber_hours + (armand_fraction * ↑amber_hours).floor + ella_multiplier * amber_hours = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_work_hours_l11_1134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_m_eq_one_l11_1155

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m / x - m + Real.log x

-- State the theorem
theorem f_nonnegative_iff_m_eq_one :
  ∀ m : ℝ, (∀ x : ℝ, x > 0 → f m x ≥ 0) ↔ m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_m_eq_one_l11_1155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_theorem_l11_1126

/-- Represents the journey of a car -/
structure CarJourney where
  totalDistance : ℝ
  initialSpeed : ℝ
  speedUpFactor : ℝ
  earlierArrival : ℝ

/-- Calculates the time to complete the remaining distance after speeding up -/
noncomputable def remainingTime (j : CarJourney) : ℝ :=
  (j.totalDistance - j.initialSpeed) / (j.speedUpFactor * j.initialSpeed)

/-- Calculates the actual time for the entire journey -/
noncomputable def actualJourneyTime (j : CarJourney) : ℝ :=
  j.totalDistance / j.initialSpeed - j.earlierArrival

/-- Compares two different driving strategies for the return journey -/
def returnJourneyComparison (a b : ℝ) : Prop :=
  a ≠ b → (90 / a + 90 / b) > 360 / (a + b)

/-- Theorem about the car journey -/
theorem car_journey_theorem (j : CarJourney) 
    (h1 : j.totalDistance = 180)
    (h2 : j.speedUpFactor = 1.5)
    (h3 : j.earlierArrival = 2/3) : 
  remainingTime j = (360 - 2 * j.initialSpeed) / (3 * j.initialSpeed) ∧
  actualJourneyTime j = 7/3 ∧
  ∀ a b, returnJourneyComparison a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_theorem_l11_1126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_property_l11_1174

theorem triangle_angle_property (a b c : ℝ) (h : (a^2 + b^2 + c^2)^2 = 4*b^2*(a^2 + c^2) + 3*a^2*c^2) :
  ∃ β : ℝ, (Real.cos β)^2 = 3/4 ∧ 
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_property_l11_1174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_rectangle_dimensions_for_specific_frustum_minimal_rectangle_length_approximation_l11_1182

/-- Represents the dimensions of a rectangular piece of material. -/
structure RectangleDimensions where
  length : ℝ
  width : ℝ

/-- Represents the properties of a conical frustum. -/
structure ConicalFrustum where
  baseDiameter : ℝ
  slantHeight : ℝ

/-- Calculates the minimum dimensions of a rectangular piece of material
    needed for the lateral surface of a conical frustum. -/
noncomputable def minimalRectangleDimensions (frustum : ConicalFrustum) : RectangleDimensions :=
  { length := 2 * Real.pi * frustum.baseDiameter / 2,
    width := frustum.slantHeight }

/-- Theorem stating that for a conical frustum with a base diameter of 4 cm
    and a slant height of 6 cm, the minimum dimensions of the rectangular
    piece of material needed for its lateral surface are 12 cm in length
    and 6 cm in width. -/
theorem minimal_rectangle_dimensions_for_specific_frustum :
  let frustum : ConicalFrustum := { baseDiameter := 4, slantHeight := 6 }
  let rect := minimalRectangleDimensions frustum
  rect.length = 4 * Real.pi ∧ rect.width = 6 := by
  sorry

/-- Corollary stating that the length is approximately 12 cm when π is approximated to 3. -/
theorem minimal_rectangle_length_approximation :
  let frustum : ConicalFrustum := { baseDiameter := 4, slantHeight := 6 }
  let rect := minimalRectangleDimensions frustum
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |rect.length - 12| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_rectangle_dimensions_for_specific_frustum_minimal_rectangle_length_approximation_l11_1182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_true_discount_l11_1180

/-- Calculate the true discount on a bill -/
noncomputable def trueDiscount (amount : ℝ) (ratePerAnnum : ℝ) (timeInMonths : ℝ) : ℝ :=
  let timeInYears := timeInMonths / 12
  (amount * ratePerAnnum * timeInYears) / (100 + (ratePerAnnum * timeInYears))

/-- Theorem: The true discount on a bill of Rs. 1764 due in 9 months with an annual interest rate of 16% is Rs. 189 -/
theorem bill_true_discount :
  let amount : ℝ := 1764
  let ratePerAnnum : ℝ := 16
  let timeInMonths : ℝ := 9
  trueDiscount amount ratePerAnnum timeInMonths = 189 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_true_discount_l11_1180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l11_1185

def problem (g : ℝ → ℝ) (g_inv : ℝ → ℝ) : Prop :=
  (Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g) ∧
  (g 1 = 2) ∧
  (g 4 = 7) ∧
  (g 3 = 8) →
  g_inv (g_inv 8 * g_inv 2) = 3

theorem problem_solution :
  ∀ (g : ℝ → ℝ) (g_inv : ℝ → ℝ),
  problem g g_inv :=
by
  intro g g_inv
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l11_1185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l11_1199

-- Define the custom operation
noncomputable def circle_slash (a b : ℝ) : ℝ := (Real.sqrt (3 * a + b))^3

-- State the theorem
theorem solve_equation (x : ℝ) : circle_slash 6 x = 64 → x = -2 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l11_1199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_squares_to_remove_l11_1179

/-- Represents a T-tetromino -/
structure TTetromino :=
  (shape : Set (ℕ × ℕ))
  (is_valid : shape = {(0,0), (1,0), (2,0), (1,1)})

/-- Represents the grid -/
def Grid := Fin 202 → Fin 202 → Bool

/-- A tiling of the grid with T-tetrominoes -/
def Tiling := List (TTetromino × ℕ × ℕ)

/-- Checks if a tiling is valid for a given grid -/
def is_valid_tiling (g : Grid) (t : Tiling) : Prop :=
  sorry

/-- The number of squares in the grid -/
def grid_size : ℕ := 202 * 202

/-- The main theorem -/
theorem min_squares_to_remove :
  ∃ (removed : Finset (Fin 202 × Fin 202)),
    (Finset.card removed = 4) ∧
    (∃ (g : Grid),
      (∀ (i j : Fin 202), g i j = false ↔ (i, j) ∈ removed) ∧
      (∃ (t : Tiling), is_valid_tiling g t)) ∧
    (∀ (removed' : Finset (Fin 202 × Fin 202)),
      Finset.card removed' < 4 →
      ∀ (g : Grid),
        (∀ (i j : Fin 202), g i j = false ↔ (i, j) ∈ removed') →
        ¬∃ (t : Tiling), is_valid_tiling g t) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_squares_to_remove_l11_1179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_reciprocal_problem_l11_1153

theorem lcm_reciprocal_problem (a b : ℕ+) (h1 : a = 24) (h2 : b = 208) 
  (h3 : (1 : ℚ) / (Nat.gcd a b : ℚ) = (1 : ℚ) / 16) : 
  (1 : ℚ) / (Nat.lcm a b : ℚ) = (1 : ℚ) / 312 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_reciprocal_problem_l11_1153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_remainder_main_theorem_l11_1186

theorem geometric_sum_remainder (n : ℕ) (a r : ℤ) (m : ℕ) : 
  (r ^ (n + 1) - 1) / (r - 1) ≡ (r ^ ((n + 1) % (Nat.totient m)) - 1) / (r - 1) [ZMOD m] :=
sorry

theorem main_theorem : (8^101 - 1) / 7 ≡ 1 [ZMOD 500] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_remainder_main_theorem_l11_1186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_80_l11_1173

noncomputable section

def line_y : ℝ := 13

def center : ℝ × ℝ := (7, 13)

def distance_to_line (p : ℝ × ℝ) : ℝ := |p.2 - line_y|

noncomputable def distance_to_center (p : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2)

def is_valid_point (p : ℝ × ℝ) : Prop :=
  distance_to_line p = 4 ∧ distance_to_center p = 10

theorem sum_of_coordinates_is_80 :
  ∃ (p1 p2 p3 p4 : ℝ × ℝ),
    is_valid_point p1 ∧ is_valid_point p2 ∧ is_valid_point p3 ∧ is_valid_point p4 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    p1.1 + p1.2 + p2.1 + p2.2 + p3.1 + p3.2 + p4.1 + p4.2 = 80 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_80_l11_1173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l11_1160

theorem xy_value (x y : ℝ) (h1 : (2:ℝ)^x = (16:ℝ)^(y+3)) (h2 : (27:ℝ)^y = (3:ℝ)^(x-2)) : x * y = 280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l11_1160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_sum_l11_1125

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x) + Real.pi / 6

theorem symmetry_point_sum (x₀ y₀ : ℝ) :
  (∀ x, f (2 * x₀ - x) = f x) →  -- symmetry condition
  x₀ ∈ Set.Ioo (Real.pi / 2) Real.pi →  -- x₀ in (π/2, π)
  y₀ = f x₀ →                    -- y₀ is the function value at x₀
  x₀ + y₀ = Real.pi := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_sum_l11_1125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_length_is_34_l11_1159

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ aₙ d : ℤ) : ℕ :=
  (((aₙ - a₁) / d).toNat + 1)

/-- Theorem: The arithmetic sequence with first term 250, last term 22, and common difference -7 has 34 terms -/
theorem sequence_length_is_34 :
  arithmetic_sequence_length 250 22 (-7) = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_length_is_34_l11_1159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemys_theorem_l11_1184

/-- A predicate stating that four points form a cyclic quadrilateral -/
def is_cyclic_quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    ‖A - center‖ = radius ∧
    ‖B - center‖ = radius ∧
    ‖C - center‖ = radius ∧
    ‖D - center‖ = radius

/-- Ptolemy's Theorem -/
theorem ptolemys_theorem (A B C D : ℝ × ℝ) (h : is_cyclic_quadrilateral A B C D) :
  ‖A - C‖ * ‖B - D‖ = ‖A - B‖ * ‖C - D‖ + ‖B - C‖ * ‖D - A‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemys_theorem_l11_1184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_5_equals_5_range_of_a_for_three_intersections_range_of_t_for_ratio_condition_l11_1107

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then -x^2 + 2*a*x - a^2 + 2*a
  else x^2 + 2*a*x + a^2 - 2*a

-- Part 1
theorem f_f_5_equals_5 :
  f 2 (f 2 5) = 5 := by
  sorry

-- Part 2
theorem range_of_a_for_three_intersections :
  ∀ a : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f a x₁ = 3 ∧ f a x₂ = 3 ∧ f a x₃ = 3) ↔ 
  (3/2 < a ∧ a ≤ 3) := by
  sorry

-- Part 3
theorem range_of_t_for_ratio_condition :
  ∀ t : ℝ, (∀ a : ℝ, 3/2 < a ∧ a ≤ 3 →
    ∀ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    f a x₁ = 3 ∧ f a x₂ = 3 ∧ f a x₃ = 3 →
    x₁ / (x₂ + x₃) < t) ↔
  (-1 < t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_5_equals_5_range_of_a_for_three_intersections_range_of_t_for_ratio_condition_l11_1107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_theorem_l11_1139

theorem game_probability_theorem (total_rounds : ℕ) 
  (alex_prob : ℚ) (mel_chelsea_ratio : ℚ) :
  total_rounds = 8 →
  alex_prob = 1/3 →
  mel_chelsea_ratio = 3 →
  let mel_prob := (1 - alex_prob) * mel_chelsea_ratio / (1 + mel_chelsea_ratio)
  let chelsea_prob := (1 - alex_prob) / (1 + mel_chelsea_ratio)
  let specific_outcome_prob := (alex_prob ^ 4) * (mel_prob ^ 3) * chelsea_prob
  let arrangements := Nat.choose total_rounds 4 * Nat.choose 4 3
  (arrangements : ℚ) * specific_outcome_prob = 35/486 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_theorem_l11_1139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l11_1171

theorem tan_ratio_from_sin_sum_diff (x y : ℝ) 
  (h1 : Real.sin (x + y) = 5/8) 
  (h2 : Real.sin (x - y) = 1/4) : 
  Real.tan x / Real.tan y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l11_1171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l11_1128

-- Define the function f as noncomputable
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

-- State the theorem
theorem min_omega_value (ω φ : ℝ) (h1 : ω > 0) (h2 : 0 < φ) (h3 : φ < Real.pi) :
  let T := 2 * Real.pi / ω
  (f ω φ T = Real.sqrt 3 / 2) →
  (f ω φ (Real.pi / 9) = 0) →
  (∀ ω' > 0, (f ω' φ T = Real.sqrt 3 / 2) → (f ω' φ (Real.pi / 9) = 0) → ω ≤ ω') →
  ω = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l11_1128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l11_1176

theorem equation_solution (x : ℝ) (n : ℤ) : 
  (Real.sqrt (6 * Real.cos (4 * x) + 15 * Real.sin (2 * x)) = 2 * Real.cos (2 * x)) ↔ 
  (x = -1/2 * Real.arcsin (1/8) + π * (n : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l11_1176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_sum_l11_1119

theorem trigonometric_ratio_sum (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 4)
  (h2 : Real.cos x / Real.cos y = 1/3) :
  Real.sin (2*x) / Real.sin (2*y) + Real.cos (2*x) / Real.cos (2*y) = 911/429 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_sum_l11_1119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_fraction_l11_1168

theorem periodic_decimal_fraction : 
  (238 : ℚ) / 333 / ((2855 : ℚ) / 999) = 714 / 2855 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_fraction_l11_1168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l11_1116

/-- The length of a marathon in kilometers -/
noncomputable def marathonLength : ℝ := 42

/-- Jack's time to complete a marathon in hours -/
noncomputable def jackTime : ℝ := 5

/-- Jill's time to complete a marathon in hours -/
noncomputable def jillTime : ℝ := 4.2

/-- Calculate the average speed given distance and time -/
noncomputable def averageSpeed (distance time : ℝ) : ℝ := distance / time

/-- Jack's average speed for the marathon -/
noncomputable def jackSpeed : ℝ := averageSpeed marathonLength jackTime

/-- Jill's average speed for the marathon -/
noncomputable def jillSpeed : ℝ := averageSpeed marathonLength jillTime

/-- Theorem stating the ratio of Jack's speed to Jill's speed -/
theorem speed_ratio : ∃ (n m : ℕ), n = 84 ∧ m = 100 ∧ jackSpeed / jillSpeed = n / m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l11_1116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_hugo_first_roll_5_given_win_l11_1141

/-- Represents the outcome of a die roll -/
def DieRoll := Fin 6

/-- Represents a player in the game -/
structure Player where
  name : String
  roll : DieRoll

/-- The game with 4 players -/
structure Game where
  players : Fin 4 → Player

/-- Hugo is one of the players in the game -/
def hugo : Player := { name := "Hugo", roll := ⟨0, sorry⟩ }

/-- The event that Hugo wins the game -/
def hugo_wins (g : Game) : Prop := sorry

/-- The probability that Hugo wins the game -/
def prob_hugo_wins : ℚ := 1 / 4

/-- The probability that Hugo's first roll is 5 -/
def prob_hugo_rolls_5 : ℚ := 1 / 6

/-- The probability that Hugo wins given his first roll was 5 -/
def prob_hugo_wins_given_5 : ℚ := 41 / 96

/-- The main theorem: The probability that Hugo's first roll was 5, given that he won the game -/
theorem prob_hugo_first_roll_5_given_win :
  (prob_hugo_rolls_5 * prob_hugo_wins_given_5) / prob_hugo_wins = 41 / 144 := by
  -- Proof steps go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_hugo_first_roll_5_given_win_l11_1141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_queen_placement_l11_1150

def ChessBoard := Fin 2017 × Fin 2017

def attacks (queen : ChessBoard) (square : ChessBoard) : Prop :=
  queen.1 = square.1 ∨ queen.2 = square.2 ∨ 
  (queen.1 : ℤ) - (square.1 : ℤ) = (queen.2 : ℤ) - (square.2 : ℤ) ∨
  (queen.1 : ℤ) - (square.1 : ℤ) = (square.2 : ℤ) - (queen.2 : ℤ)

theorem impossible_queen_placement :
  ¬ ∃ (queens : Finset ChessBoard),
    queens.card = 1000 ∧
    ∀ (square : ChessBoard), ∃ (queen : ChessBoard), queen ∈ queens ∧ attacks queen square :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_queen_placement_l11_1150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_q_over_s_value_l11_1124

noncomputable def g (x : ℝ) : ℝ := (3*x - 2) / (x + 4)

noncomputable def g_inverse (x : ℝ) : ℝ := (4*x + 2) / (3 - x)

theorem inverse_function_theorem (x : ℝ) :
  g (g_inverse x) = x ∧ g_inverse (g x) = x :=
by sorry

theorem q_over_s_value :
  2 / 3 = (4 * 0 + 2) / (3 - 0) :=
by
  norm_num

#eval (4 * 0 + 2) / (3 - 0)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_q_over_s_value_l11_1124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l11_1191

theorem arithmetic_sequence_sum : ∀ (a₁ d n : ℤ),
  a₁ = -48 ∧ d = 3 ∧ a₁ + (n - 1) * d = 0 →
  (n * (a₁ + (a₁ + (n - 1) * d))) / 2 = -408 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l11_1191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l11_1198

open Real

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides opposite to angles A, B, C respectively

-- Define the conditions
axiom angle_side_correspondence : 
  (A > 0 ∧ B > 0 ∧ C > 0) ∧ (a > 0 ∧ b > 0 ∧ c > 0)

axiom cos_A : cos A = 1/2
axiom side_a : a = sqrt 3

-- Define the sine rule
axiom sine_rule : a / (sin A) = b / (sin B) ∧ b / (sin B) = c / (sin C)

-- State the theorem to be proved
theorem triangle_ratio_theorem :
  (sin A + sin B + sin C) / (a + b + c) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l11_1198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_capital_l11_1114

/-- The capital of partner a in a three-partner business arrangement -/
noncomputable def capital_of_a (total_capital : ℝ) : ℝ := (2/3) * total_capital

/-- The profit at a given rate -/
noncomputable def profit_at_rate (total_capital : ℝ) (rate : ℝ) : ℝ := rate * total_capital

/-- Partner a's income at a given profit rate -/
noncomputable def a_income (total_capital : ℝ) (rate : ℝ) : ℝ := (2/3) * profit_at_rate total_capital rate

theorem partner_a_capital : 
  ∃ (total_capital : ℝ),
    a_income total_capital 0.07 - a_income total_capital 0.05 = 200 ∧
    capital_of_a total_capital = 10000 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_capital_l11_1114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identification_l11_1108

/-- Represents a coin, which can be either genuine or counterfeit -/
inductive Coin
| genuine
| counterfeit

/-- Represents the result of weighing two groups of coins -/
inductive WeighResult
| equal
| left_heavier
| right_heavier

/-- Represents a weighing operation on a balance scale -/
def weigh (left_group right_group : List Coin) : WeighResult :=
  sorry

/-- Represents the state of our coin arrangement -/
structure CoinArrangement where
  coins : List Coin
  counterfeit_start : Nat
  counterfeit_end : Nat

/-- Checks if a given coin arrangement is valid -/
def is_valid_arrangement (arr : CoinArrangement) : Prop :=
  arr.coins.length = 23 ∧
  arr.counterfeit_end - arr.counterfeit_start + 1 = 6 ∧
  arr.counterfeit_end < arr.coins.length

/-- Represents a strategy for identifying a counterfeit coin -/
def identification_strategy (arr : CoinArrangement) : Prop :=
  ∃ (first_left first_right : List Coin)
    (second_left second_right : List Coin),
    first_left ⊆ arr.coins ∧
    first_right ⊆ arr.coins ∧
    second_left ⊆ arr.coins ∧
    second_right ⊆ arr.coins ∧
    first_left.length = first_right.length ∧
    second_left.length = second_right.length ∧
    (∃ (c : Coin), (c ∈ second_left ∨ c ∈ second_right) ∧ c = Coin.counterfeit)

/-- The main theorem: There exists a strategy to identify at least one counterfeit coin
    using two weighings for any valid coin arrangement -/
theorem counterfeit_coin_identification
  (arr : CoinArrangement) (h : is_valid_arrangement arr) :
  identification_strategy arr :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identification_l11_1108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiplier_l11_1188

theorem largest_multiplier (x : ℕ) (N : ℕ) (h1 : N * (10 : ℕ)^x < 31000) (h2 : x ≤ 3) :
  ∃ (M : ℕ), M = 30 ∧ M * (10 : ℕ)^3 < 31000 ∧ ∀ (K : ℕ), K * (10 : ℕ)^3 < 31000 → K ≤ M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiplier_l11_1188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_sum_zero_l11_1163

noncomputable section

/-- The curve defined by y = (ax + b) / (cx + d) -/
def curve (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- The axis of symmetry y = -x -/
def axis_of_symmetry (x : ℝ) : ℝ := -x

theorem symmetry_implies_sum_zero (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x y : ℝ, y = curve a b c d x → axis_of_symmetry y = curve a b c d (axis_of_symmetry x)) →
  a + d = 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_sum_zero_l11_1163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sum_product_l11_1181

theorem cubic_sum_product (x₁ x₂ x₃ p q : ℝ) 
  (h₁ : x₁^3 + x₁*p + q = 0)
  (h₂ : x₂^3 + x₂*p + q = 0)
  (h₃ : x₃^3 + x₃*p + q = 0)
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) :
  x₁^3 + x₂^3 + x₃^3 = 3*x₁*x₂*x₃ := by
  sorry

#check cubic_sum_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sum_product_l11_1181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grandfather_age_theorem_l11_1110

/-- Calculates the age of Yuna's grandfather based on Yuna's age and the age differences. -/
def grandfather_age (yuna_age : ℕ) (father_diff : ℕ) (grandfather_diff : ℕ) : ℕ :=
  yuna_age + father_diff + grandfather_diff

/-- Given Yuna's age, the age difference between Yuna and her father, and the age difference
    between her father and grandfather, proves that the grandfather's age is equal to
    Yuna's age plus the sum of both age differences. -/
theorem grandfather_age_theorem (yuna_age : ℕ) (father_diff : ℕ) (grandfather_diff : ℕ) :
  grandfather_age yuna_age father_diff grandfather_diff = yuna_age + father_diff + grandfather_diff :=
by
  rfl  -- reflexivity proves this trivial equality

#eval grandfather_age 8 20 25  -- Should output 53

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grandfather_age_theorem_l11_1110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_sqrt_prob_l11_1145

theorem two_digit_sqrt_prob : 
  (Finset.card (Finset.filter (fun n => (n + 10) < 49) (Finset.range 90)) : ℚ) / 
  (Finset.card (Finset.range 90)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_sqrt_prob_l11_1145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_to_line_l11_1151

/-- The circle equation in polar coordinates -/
def circle_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The line equation in polar coordinates -/
def line_eq (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- The center of the circle in Cartesian coordinates -/
def circle_center : ℝ × ℝ := (1, 0)

/-- The distance from the center of the circle to the line -/
theorem distance_from_circle_to_line :
  let d := Real.sqrt ((circle_center.1 - 2) ^ 2 + circle_center.2 ^ 2)
  d = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_to_line_l11_1151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l11_1127

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem stating that the eccentricity of the given hyperbola is 2 -/
theorem hyperbola_eccentricity_is_two (h : Hyperbola) (c : Circle)
  (h_circle : c.h = 2 ∧ c.k = 0 ∧ c.r = 2)
  (h_chord : ∃ (x y : ℝ), (x - c.h)^2 + y^2 = c.r^2 ∧ 
    (x / h.a)^2 - (y / h.b)^2 = 1 ∧
    (∃ (t : ℝ), x = h.a * t ∧ y = h.b * t) ∧
    (x - 2)^2 + y^2 = 4) :
  eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l11_1127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_squared_specific_trapezoid_l11_1104

/-- Represents an isosceles trapezoid ABCD with a circle tangent to its sides --/
structure IsoscelesTrapezoidWithCircle where
  AB : ℝ
  CD : ℝ
  x : ℝ  -- length of AD and BC
  h : AB > CD  -- AB is the longer base

/-- The minimum possible value of x^2 for the given trapezoid configuration --/
noncomputable def min_x_squared (t : IsoscelesTrapezoidWithCircle) : ℝ := 
  (t.AB / 2)^2 - ((t.AB - t.CD) / 4)^2

/-- Theorem stating the minimum value of x^2 for the specific trapezoid --/
theorem min_x_squared_specific_trapezoid :
  let t : IsoscelesTrapezoidWithCircle := ⟨100, 25, 0, by norm_num⟩
  min_x_squared t = 1875 := by sorry

#eval (100 / 2)^2 - ((100 - 25) / 4)^2  -- This should evaluate to 1875.0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_squared_specific_trapezoid_l11_1104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_l11_1175

def divides (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

def is_valid_pair (n m : ℕ) : Prop :=
  n > 0 ∧ m > 0 ∧ divides (n^m - m) (m^2 + 2*m)

theorem valid_pairs :
  ∀ n m : ℕ, is_valid_pair n m ↔ (n = 2 ∧ m ∈ ({1, 2, 3, 4} : Set ℕ)) ∨ (n = 4 ∧ m = 1) :=
by
  sorry

#check valid_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_l11_1175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_theorem_l11_1165

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 8 = 0
def curve2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 4 = 0

-- Define the tangent length function
noncomputable def tangentLength (x y : ℝ) (h k r : ℝ) : ℝ := 
  Real.sqrt ((x - h)^2 + (y - k)^2 - r^2)

-- Theorem statement
theorem tangent_points_theorem : 
  ∀ x y : ℝ, 
    (tangentLength x y (-1) 0 3 = 6 ∧ 
     tangentLength x y 2 3 3 = 6) ↔ 
    ((x = -4 ∧ y = 6) ∨ (x = 5 ∧ y = -3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_theorem_l11_1165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_iteration_theorem_l11_1167

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

def Line (p q : Point) : Set Point :=
  {r : Point | ∃ t : ℝ, r.x = p.x + t * (q.x - p.x) ∧ r.y = p.y + t * (q.y - p.y)}

-- Define the sequence of circles
def circle_sequence : ℕ → Circle := sorry

-- Define the original circles
def original_circle1 : Circle := sorry
def original_circle2 : Circle := sorry

-- Define point P (tangent point of original circles)
def P : Point := sorry

-- Define points A and B
def A : Point := sorry
def B : Point := sorry

-- Define the initial circle
def initial_circle : Circle := sorry

-- Define the line AB
def AB : Set Point := Line A B

-- Distance function
noncomputable def distance (p : Point) (l : Set Point) : ℝ := sorry

-- Theorem statement
theorem circle_iteration_theorem (n : ℕ) :
  distance (circle_sequence n).center AB = n * (2 * (circle_sequence n).radius) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_iteration_theorem_l11_1167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_theorem_l11_1109

/-- Calculates the initial distance between two trains given their lengths, speeds, and meeting time. -/
noncomputable def initial_distance (length1 length2 : ℝ) (speed1 speed2 : ℝ) (meeting_time : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (1000 / 3600)  -- Convert km/h to m/s
  let total_distance := relative_speed * (meeting_time * 60)  -- Convert minutes to seconds
  total_distance - (length1 + length2)

/-- Theorem stating the initial distance between two trains given specific conditions. -/
theorem train_distance_theorem :
  let length1 : ℝ := 100
  let length2 : ℝ := 200
  let speed1 : ℝ := 54
  let speed2 : ℝ := 72
  let meeting_time : ℝ := 2.856914303998537
  abs (initial_distance length1 length2 speed1 speed2 meeting_time - 5699.52) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_theorem_l11_1109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_g_l11_1102

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := 5 / (2 * x^2 - 4 * x + 3)

-- State the theorem
theorem range_of_f : Set.range f = Set.Ioo 0 5 := by
  sorry

-- Define the second function
noncomputable def g (x : ℝ) := x + Real.sqrt (1 - 2 * x)

-- State the theorem for the second function
theorem range_of_g : Set.range g = Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_g_l11_1102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_subdivision_property_l11_1195

/-- A polygon in a plane -/
structure Polygon where
  sides : ℕ
  is_convex : Bool

/-- A subdivision of a triangle into polygons -/
structure TriangleSubdivision where
  polygons : List Polygon
  is_valid : Bool  -- Ensures the subdivision is valid

/-- 
Given a valid subdivision of a triangle into convex polygons,
there exists either a triangle or two polygons with the same number of sides
-/
theorem triangle_subdivision_property (sub : TriangleSubdivision) 
  (h_valid : sub.is_valid = true) 
  (h_convex : ∀ p, p ∈ sub.polygons → p.is_convex = true) :
  (∃ p, p ∈ sub.polygons ∧ p.sides = 3) ∨ 
  (∃ p1 p2, p1 ∈ sub.polygons ∧ p2 ∈ sub.polygons ∧ p1 ≠ p2 ∧ p1.sides = p2.sides) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_subdivision_property_l11_1195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_45_l11_1142

theorem sum_of_divisors_45 : (Finset.sum (Nat.divisors 45) id) = 78 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_45_l11_1142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_square_side_approx_173_l11_1130

/-- The side length of a square -/
noncomputable def side_length : ℝ := 200

/-- The total area of the large square -/
noncomputable def total_area : ℝ := side_length ^ 2

/-- The fraction of the total area occupied by one L-shaped region -/
noncomputable def l_shape_fraction : ℝ := 1 / 16

/-- The number of L-shaped regions -/
def num_l_shapes : ℕ := 4

/-- The side length of the center square -/
noncomputable def center_square_side : ℝ := Real.sqrt (total_area * (1 - ↑num_l_shapes * l_shape_fraction))

theorem center_square_side_approx_173 :
  ⌊center_square_side⌋ = 173 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_square_side_approx_173_l11_1130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_solution_l11_1197

-- Define the polynomial type
def MyPolynomial (α : Type) := α → α

-- Define the equation that p must satisfy
def SatisfiesEquation (p : MyPolynomial ℝ) : Prop :=
  ∀ x, p (p x) = 2 * x * p x + x^2

-- State the theorem
theorem polynomial_solution :
  ∀ p : MyPolynomial ℝ,
    SatisfiesEquation p ↔ 
      (∀ x, p x = (1 + Real.sqrt 2) * x) ∨
      (∀ x, p x = (1 - Real.sqrt 2) * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_solution_l11_1197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_morning_kibble_amount_l11_1190

/-- The amount of kibble Mary gave Luna in the morning -/
def morning_kibble : ℝ := 1

/-- The total amount of kibble in the bag at the start -/
def total_kibble : ℝ := 12

/-- The amount of kibble remaining in the bag the next morning -/
def remaining_kibble : ℝ := 7

/-- The amount of kibble Frank gave Luna in the afternoon -/
def afternoon_kibble : ℝ := 1

/-- The amount of kibble Frank gave Luna in the late evening -/
def late_evening_kibble : ℝ := 2 * afternoon_kibble

theorem morning_kibble_amount :
  morning_kibble = 1 := by
  -- The proof goes here
  sorry

#eval morning_kibble

end NUMINAMATH_CALUDE_ERRORFEEDBACK_morning_kibble_amount_l11_1190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_sum_l11_1162

-- Define the sum of coordinates of a point
def sum_coordinates (p : ℝ × ℝ) : ℝ := p.1 + p.2

-- Theorem statement
theorem midpoint_coordinate_sum :
  let M : ℝ × ℝ := (3, 5)
  let C : ℝ × ℝ := (5, 3)
  ∀ D : ℝ × ℝ, ((C.1 + D.1) / 2, (C.2 + D.2) / 2) = M → sum_coordinates D = 8 :=
by
  -- Introduce the points and the hypothesis
  intro M C D h
  -- Expand the definition of sum_coordinates
  unfold sum_coordinates
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_sum_l11_1162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_range_l11_1123

-- Define the function y = ae^x + 3x
noncomputable def y (a x : ℝ) : ℝ := a * Real.exp x + 3 * x

-- Define the derivative of y with respect to x
noncomputable def y_derivative (a x : ℝ) : ℝ := a * Real.exp x + 3

-- Define what it means for the function to have an extremum point greater than zero
def has_extremum_point_gt_zero (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ y_derivative a x = 0

-- State the theorem
theorem extremum_point_implies_a_range (a : ℝ) :
  has_extremum_point_gt_zero a → a ∈ Set.Ioo (-3) 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_range_l11_1123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_l11_1170

-- Define the curve C
noncomputable def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the intersection points A and B
noncomputable def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  curve_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Define the area of triangle AOB
noncomputable def triangle_area (A B : ℝ × ℝ) : ℝ :=
  abs ((A.1 * B.2 - B.1 * A.2) / 2)

-- Theorem statement
theorem area_of_triangle_AOB :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  triangle_area A B = 8 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_l11_1170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_1090_eq_cos_80_l11_1172

open Real

-- Define the periodicity of sine
axiom sine_periodicity (x : ℝ) : sin x = sin (x + 2 * π)

-- Define the co-function identity
axiom cofunction_identity (x : ℝ) : sin x = cos (π / 2 - x)

-- Theorem to prove
theorem sin_1090_eq_cos_80 : 
  sin (1090 * π / 180) = cos (80 * π / 180) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_1090_eq_cos_80_l11_1172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_of_given_fractions_l11_1169

def hcf_fractions (a b c : ℚ) : ℚ :=
  let num_hcf := Nat.gcd (Nat.gcd (Int.natAbs a.num) (Int.natAbs b.num)) (Int.natAbs c.num)
  let den_lcm := Nat.lcm (Nat.lcm a.den b.den) c.den
  (num_hcf : ℚ) / den_lcm

theorem hcf_of_given_fractions :
  hcf_fractions (2/3) (4/9) (6/18) = 1/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_of_given_fractions_l11_1169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_three_of_six_l11_1101

/-- The probability that a baby speaks within two days -/
def p : ℚ := 1/3

/-- The number of babies in the group -/
def n : ℕ := 6

/-- The minimum number of babies we want to speak -/
def k : ℕ := 3

/-- The probability that at least k out of n babies speak within two days -/
def prob_at_least (p : ℚ) (n k : ℕ) : ℚ :=
  1 - (Finset.range k).sum (λ i ↦ (n.choose i : ℚ) * p^i * (1-p)^(n-i))

theorem prob_at_least_three_of_six :
  prob_at_least p n k = 353/729 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_three_of_six_l11_1101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_300_degree_angle_l11_1132

/-- Given a point P(1, m) on the terminal side of a 300° angle in the Cartesian coordinate plane,
    prove that m = -√3. -/
theorem point_on_300_degree_angle (m : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = 1 ∧ P.2 = m ∧ (P.1 : ℝ) * Real.cos (300 * π / 180) = 1 ∧ (P.2 : ℝ) * Real.sin (300 * π / 180) = m) →
  m = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_300_degree_angle_l11_1132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_product_l11_1177

theorem cos_sum_product (a b : ℝ) : 
  Real.cos (a + b) + Real.cos (a - b) = 2 * Real.cos a * Real.cos b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_product_l11_1177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l11_1143

noncomputable section

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 / 2 = 1

-- Define the foci of the ellipse
def left_focus : ℝ × ℝ := (-2, 0)
def right_focus : ℝ × ℝ := (2, 0)

-- Define the point P that the ellipse passes through
def point_P : ℝ × ℝ := (2, Real.sqrt 6 / 3)

-- Define the line l passing through the right focus
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y + 2

-- Define the area of triangle OAB
noncomputable def triangle_area (m : ℝ) : ℝ :=
  2 * Real.sqrt 6 * Real.sqrt (m^2 + 1) / (m^2 + 3)

theorem ellipse_and_line_theorem :
  -- The standard equation of ellipse C is correct
  (∀ x y : ℝ, ellipse_C x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  -- The ellipse passes through point P
  ellipse_C point_P.1 point_P.2 ∧
  -- The line l that maximizes the area of triangle OAB has slope ±1
  (∃ m : ℝ, triangle_area m = Real.sqrt 3 ∧ (m = 1 ∨ m = -1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l11_1143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_b_cycle_ten_is_smallest_l11_1112

noncomputable def b : ℕ → ℝ
  | 0 => Real.cos (Real.pi / 30) ^ 2
  | n + 1 => 4 * b n * (1 - b n)

theorem smallest_n_for_b_cycle : ∃ n : ℕ, n > 0 ∧ b n = b 0 ∧ ∀ m : ℕ, 0 < m ∧ m < n → b m ≠ b 0 := by
  -- We claim that n = 10 satisfies the conditions
  use 10
  apply And.intro
  · -- Prove 10 > 0
    simp
  apply And.intro
  · -- Prove b 10 = b 0
    sorry -- This requires computation and algebraic manipulation
  · -- Prove ∀ m : ℕ, 0 < m ∧ m < 10 → b m ≠ b 0
    intros m hm
    sorry -- This requires checking all values of m from 1 to 9

-- Additional theorem to show that 10 is indeed the smallest such n
theorem ten_is_smallest : ∀ n : ℕ, 0 < n ∧ n < 10 → b n ≠ b 0 := by
  sorry -- This requires checking all values of n from 1 to 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_b_cycle_ten_is_smallest_l11_1112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doug_initial_marbles_l11_1157

theorem doug_initial_marbles (ed_marbles : ℕ) (difference : ℕ) (doug_initial_marbles : ℕ) : 
  ed_marbles = 27 → difference = 5 → ed_marbles = difference + doug_initial_marbles → doug_initial_marbles = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_doug_initial_marbles_l11_1157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_less_than_two_l11_1111

theorem a_less_than_b_less_than_two (a b : ℝ) (h1 : (3 : ℝ)^a = 2) (h2 : b^3 = 2) : a < b ∧ b < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_less_than_two_l11_1111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l11_1113

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 + 2 * x - 1) / Real.log (1/2)

noncomputable def g (x : ℝ) : ℝ := (2 + 2 * Real.sin (2 * x + Real.pi / 6)) / (Real.sin x + Real.sqrt 3 * Real.cos x)

def x_range : Set ℝ := { x | 7/10 ≤ x ∧ x ≤ 3/2 }

theorem a_range_theorem (a : ℝ) :
  (∀ x₁ ∈ x_range, ∀ x₂, f a x₁ > g x₂) ↔ -40/49 < a ∧ a < -4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l11_1113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_correct_l11_1158

/-- Given two intersecting circles with radii R and r, and the distance between their centers a,
    this function calculates the radius of the largest circle that can be inscribed to touch
    both circles internally. -/
noncomputable def largest_inscribed_circle_radius (R r a : ℝ) : ℝ :=
  (R + r - a) / 2

/-- Theorem stating that the radius of the largest inscribed circle is (R + r - a) / 2 -/
theorem largest_inscribed_circle_radius_correct
  (R r a : ℝ) 
  (h_positive : R > 0 ∧ r > 0 ∧ a > 0) 
  (h_intersect : a < R + r) :
  ∃ x : ℝ, x = largest_inscribed_circle_radius R r a ∧ 
    x > 0 ∧ 
    ∀ y : ℝ, y > 0 → y ≤ x → 
      (R - y) + (r - y) ≥ a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_correct_l11_1158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_problem_l11_1156

-- Define the setup
def chord_length : ℝ := 100
def small_radius : ℝ := 40

-- Define the radius of the larger circle
noncomputable def large_radius : ℝ := Real.sqrt 4100

-- Define the area of the shaded region
noncomputable def shaded_area : ℝ := 2500 * Real.pi

-- Theorem statement
theorem concentric_circles_problem :
  let half_chord := chord_length / 2
  large_radius ^ 2 = half_chord ^ 2 + small_radius ^ 2 ∧
  shaded_area = Real.pi * (large_radius ^ 2 - small_radius ^ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_problem_l11_1156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solution_l11_1106

theorem cos_equation_solution (x : ℝ) : 
  (1/2 * |Real.cos (2*x) - 1/2| = Real.cos x ^ 2 + Real.cos x * Real.cos (5*x)) ↔ 
  (∃ k : ℤ, x = π/6 + k*π/2 ∨ x = -π/6 + k*π/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solution_l11_1106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_problem_l11_1137

/-- Represents the scenario of a person walking on a moving walkway -/
structure WalkwayScenario where
  length : ℝ
  time_with : ℝ
  time_against : ℝ

/-- Calculates the time to walk when the walkway is not moving -/
noncomputable def time_without_movement (scenario : WalkwayScenario) : ℝ :=
  2 * scenario.length * scenario.time_with * scenario.time_against /
  (scenario.time_with * scenario.time_against + (scenario.time_with + scenario.time_against) * scenario.length)

/-- Theorem stating that for the given scenario, the time to walk without movement is 60 seconds -/
theorem walkway_problem (scenario : WalkwayScenario)
  (h1 : scenario.length = 80)
  (h2 : scenario.time_with = 40)
  (h3 : scenario.time_against = 120) :
  time_without_movement scenario = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_problem_l11_1137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recover_startup_capital_l11_1117

/-- Represents the charity event organized by the company -/
structure CharityEvent where
  initialParticipants : ℕ
  dailyIncreaseRate : ℚ
  stabilizationDay : ℕ
  startupCapital : ℚ
  profitPerParticipant : ℚ

/-- Calculate the number of participants on a given day -/
noncomputable def participantsOnDay (event : CharityEvent) (day : ℕ) : ℚ :=
  if day ≤ event.stabilizationDay then
    (event.initialParticipants : ℚ) * (1 + event.dailyIncreaseRate) ^ (day - 1)
  else
    (event.initialParticipants : ℚ) * (1 + event.dailyIncreaseRate) ^ (event.stabilizationDay - 1)

/-- Calculate the total profit up to a given day -/
noncomputable def totalProfitUpToDay (event : CharityEvent) (day : ℕ) : ℚ :=
  if day ≤ event.stabilizationDay then
    (event.initialParticipants : ℚ) * event.profitPerParticipant * 
      ((1 - (1 + event.dailyIncreaseRate)^day) / (1 - (1 + event.dailyIncreaseRate)))
  else
    let stabilizedProfit := (event.initialParticipants : ℚ) * event.profitPerParticipant * 
      ((1 - (1 + event.dailyIncreaseRate)^event.stabilizationDay) / (1 - (1 + event.dailyIncreaseRate)))
    let extraDays := day - event.stabilizationDay
    let dailyProfitAfterStabilization := participantsOnDay event event.stabilizationDay * event.profitPerParticipant
    stabilizedProfit + (extraDays : ℚ) * dailyProfitAfterStabilization

/-- The main theorem to prove -/
theorem recover_startup_capital (event : CharityEvent) 
  (h1 : event.initialParticipants = 5000)
  (h2 : event.dailyIncreaseRate = 15/100)
  (h3 : event.stabilizationDay = 30)
  (h4 : event.startupCapital = 200000)
  (h5 : event.profitPerParticipant = 1/20) :
  ∃ d : ℕ, d = 37 ∧ totalProfitUpToDay event d > event.startupCapital := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recover_startup_capital_l11_1117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_number_identification_l11_1140

theorem negative_number_identification : ∀ x : ℤ, x ∈ ({-1, 0, 1, 2} : Set ℤ) → (x < 0 ↔ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_number_identification_l11_1140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleAllPositive_l11_1196

/-- Represents a grid of signs (+ or -) -/
def Grid (n : ℕ) := Fin n → Fin n → Bool

/-- Represents the allowed operations on the grid -/
inductive Operation (n : ℕ)
  | row : Fin n → Operation n
  | column : Fin n → Operation n
  | diagonal : Bool → Fin n → Operation n

/-- Applies an operation to a grid -/
def applyOperation {n : ℕ} (g : Grid n) (op : Operation n) : Grid n :=
  sorry

/-- The initial configuration of the grid -/
def initialGrid (n : ℕ) : Grid n :=
  sorry

/-- Checks if all cells in the grid are positive -/
def allPositive {n : ℕ} (g : Grid n) : Prop :=
  ∀ i j, g i j = true

theorem impossibleAllPositive (n : ℕ) (h : n = 4 ∨ n = 8) :
  ¬∃ (ops : List (Operation n)), allPositive (ops.foldl applyOperation (initialGrid n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleAllPositive_l11_1196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocal_distances_l11_1147

-- Define the circle C
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y = 0

-- Define the line l
def line_equation (x y : ℝ) : Prop := y = x - 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_equation A.1 A.2 ∧ circle_equation B.1 B.2 ∧
  line_equation A.1 A.2 ∧ line_equation B.1 B.2

-- Define the point P
def P : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem intersection_sum_reciprocal_distances
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  1 / distance P A + 1 / distance P B = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocal_distances_l11_1147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l11_1121

theorem constant_term_expansion (x : ℝ) (h : x > 0) : 
  ∃ (terms : List ℝ), 
    (λ y => y * (1 - 2 / Real.sqrt x)^6) x = terms.sum ∧ 
    (∃ c ∈ terms, c = 60 ∧ ∀ t ∈ terms, t ≠ c → ∃ n : ℤ, t = x^n ∧ n ≠ 0) := by
  sorry

#check constant_term_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l11_1121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_negative_one_and_sqrt_three_l11_1131

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def A : Set ℝ := {x | x^2 - (floor x) = 2}

def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem A_intersect_B_equals_negative_one_and_sqrt_three :
  A ∩ B = {-1, Real.sqrt 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_negative_one_and_sqrt_three_l11_1131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l11_1194

theorem log_sum_upper_bound (x y : ℝ) (hx : x > y) (hy : y > 2) :
  (∃ (ε : ℝ), ε > 0 ∧ Real.log (x / y) / Real.log x + Real.log (y / x) / Real.log y < ε) ∧
  (∃ (z w : ℝ), z > w ∧ w > 2 ∧ Real.log (z / w) / Real.log z + Real.log (w / z) / Real.log w = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l11_1194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_property_l11_1146

theorem series_property (r : ℝ) (h_r_pos : r > 0) : 
  ∃ (s a : ℝ) (seq : ℕ → ℝ), 
    (∀ n, seq n > 0) ∧ 
    seq 0 = s ∧ 
    seq 1 = a ∧ 
    (∀ n ≥ 2, seq n = seq (n-1) * seq (n-2)) ∧ 
    seq 4 = 1/r ∧ 
    seq 7 = 1/(r^4) → 
    s = 1/Real.sqrt r :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_property_l11_1146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_chris_time_difference_l11_1193

/-- The walking speed of both Mark and Chris in miles per hour -/
noncomputable def walking_speed : ℝ := 3

/-- The distance from their house to school in miles -/
noncomputable def school_distance : ℝ := 9

/-- The distance Mark walks before turning back for his lunch in miles -/
noncomputable def mark_initial_distance : ℝ := 3

/-- Calculates the total distance Mark walks -/
noncomputable def mark_total_distance : ℝ := 2 * mark_initial_distance + school_distance

/-- Calculates the time Mark spends walking -/
noncomputable def mark_time : ℝ := mark_total_distance / walking_speed

/-- Calculates the time Chris spends walking -/
noncomputable def chris_time : ℝ := school_distance / walking_speed

/-- The difference in walking time between Mark and Chris -/
noncomputable def time_difference : ℝ := mark_time - chris_time

theorem mark_chris_time_difference :
  time_difference = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_chris_time_difference_l11_1193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_036_product_l11_1135

/-- Represents a repeating decimal with a repeating part of length 3 -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_036_product : 
  let x := RepeatingDecimal 0 3 6
  let n := x.num
  let d := x.den
  n * d = 444 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_036_product_l11_1135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_police_catch_thief_l11_1120

/-- Represents a cell in the m × n table -/
structure Cell where
  x : Nat
  y : Nat

/-- Determines if two cells are neighbors -/
def isNeighbor (c1 c2 : Cell) : Prop :=
  (c1.x = c2.x ∧ (c1.y + 1 = c2.y ∨ c2.y + 1 = c1.y)) ∨
  (c1.y = c2.y ∧ (c1.x + 1 = c2.x ∨ c2.x + 1 = c1.x))

/-- Determines if a cell has the same parity coordinates -/
def hasSameParity (c : Cell) : Prop :=
  c.x % 2 = c.y % 2

/-- Represents the game state -/
structure GameState where
  m : Nat
  n : Nat
  policeman : Cell
  thief : Cell

/-- Determines if the policeman can catch the thief -/
def canCatch (state : GameState) : Prop :=
  hasSameParity state.thief

theorem police_catch_thief (m n i j : Nat) (h1 : m > 0) (h2 : n > 0) (h3 : i ≤ m) (h4 : j ≤ n) :
  let initialState : GameState := {
    m := m,
    n := n,
    policeman := { x := 1, y := 1 },
    thief := { x := i, y := j }
  }
  canCatch initialState ↔ hasSameParity initialState.thief :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_police_catch_thief_l11_1120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_translation_on_sine_graphs_l11_1161

theorem point_translation_on_sine_graphs (t s : ℝ) (h_s : s > 0) : 
  Real.sin (2 * (π/4) - π/3) = t ∧ 
  (∃ x', Real.sin (2 * x') = t ∧ x' = π/4 - s) →
  t = 1/2 ∧ (∀ s' > 0, (∃ x', Real.sin (2 * x') = t ∧ x' = π/4 - s') → s' ≥ π/6) ∧
  (∃ x', Real.sin (2 * x') = t ∧ x' = π/4 - π/6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_translation_on_sine_graphs_l11_1161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_18_seconds_l11_1105

-- Define the train length in meters
noncomputable def train_length : ℝ := 160

-- Define the train speed in km/h
noncomputable def train_speed_kmh : ℝ := 32

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Calculate the train speed in m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmh * kmh_to_ms

-- Define the time taken for the train to cross a stationary point
noncomputable def crossing_time : ℝ := train_length / train_speed_ms

-- Theorem to prove
theorem train_crossing_time_approx_18_seconds :
  (Int.floor crossing_time) = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_18_seconds_l11_1105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l11_1133

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ := a + (b + (b + b^(1/3))^(1/3))^(1/3)

-- Theorem statement
theorem star_equation_solution :
  ∃ h : ℝ, star 8 h = 10 ∧ h = 6 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l11_1133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l11_1149

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x + (2*x)/(x^2 + 1) + (x*(x + 3))/(x^2 + 2) + (3*(x + 1))/(x*(x^2 + 2))

/-- Theorem stating that 3 is the minimum value of f(x) for x > 0 -/
theorem min_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ 3) ∧ (∃ x : ℝ, x > 0 ∧ f x = 3) := by
  sorry

-- Example of how to use the theorem
example : ∃ x : ℝ, x > 0 ∧ f x = 3 := by
  exact (min_value_of_f.2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l11_1149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l11_1129

open Real

-- Define the second quadrant
def second_quadrant (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ ∈ Set.Ioo ((2 * k + 1/2) * π) ((2 * k + 1) * π)

-- Define the third quadrant
def third_quadrant (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ ∈ Set.Ioo ((2 * k + 1) * π) ((2 * k + 3/2) * π)

-- Theorem statement
theorem angle_in_third_quadrant (θ : ℝ) 
  (h1 : second_quadrant θ) 
  (h2 : |sin (θ / 2)| = -sin (θ / 2)) : 
  third_quadrant (θ / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l11_1129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l11_1183

/-- Given the equation (sin²x + 1/sin²x)³ + (cos²y + 1/cos²y)³ = 16 cos z,
    this theorem states that the equation is satisfied if and only if
    x = π/2 + πm, y = πn, and z = 2πm, where m and n are integers,
    under the conditions that sin x ≠ 0 and cos y ≠ 0. -/
theorem trigonometric_equation_solution
  (x y z : Real) (hx : Real.sin x ≠ 0) (hy : Real.cos y ≠ 0) :
  (Real.sin x ^ 2 + 1 / Real.sin x ^ 2) ^ 3 + (Real.cos y ^ 2 + 1 / Real.cos y ^ 2) ^ 3 = 16 * Real.cos z ↔
  ∃ (m n : Int), x = Real.pi / 2 + Real.pi * ↑m ∧ y = Real.pi * ↑n ∧ z = 2 * Real.pi * ↑m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l11_1183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l11_1148

theorem proposition_equivalence (m : ℝ) : 
  (∀ x : ℝ, (Real.sin x + Real.cos x > m) ≠ (x^2 + m*x + 1 > 0)) ↔ 
  (m ≤ -2 ∨ (-Real.sqrt 2 ≤ m ∧ m < 2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l11_1148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l11_1115

-- Define the line l
def line_l (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x + 2

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Define the midpoint of a line segment
def segment_midpoint (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem midpoint_coordinates :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  ∃ M : ℝ × ℝ, segment_midpoint A B M ∧ M = (-Real.sqrt 3 / 2, 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l11_1115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_legacy_cleaning_earnings_l11_1187

/-- Calculate the total amount Legacy makes from cleaning a building with given conditions -/
theorem legacy_cleaning_earnings : 
  let floors : ℕ := 12
  let rooms_per_floor : ℕ := 25
  let cleaning_hours_per_room : ℕ := 8
  let rate_first_4_hours : ℕ := 20
  let rate_next_4_hours : ℕ := 25
  let supply_cost : ℕ := 1200
  let total_rooms : ℕ := floors * rooms_per_floor
  let earnings_per_room : ℕ := (rate_first_4_hours * 4) + (rate_next_4_hours * 4)
  let total_earnings : ℕ := total_rooms * earnings_per_room
  total_earnings - supply_cost = 52800 := by
  -- Proof steps would go here
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_legacy_cleaning_earnings_l11_1187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_theorem_l11_1192

def scores : List ℕ := [65, 70, 75, 85, 90, 100]

def is_integer_average (sublist : List ℕ) : Prop :=
  (sublist.sum % sublist.length = 0)

def all_averages_integer (list : List ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → n ≤ list.length → is_integer_average (list.take n)

theorem last_score_theorem (list : List ℕ) (h : list = scores) :
  all_averages_integer list → (list.getLast? = some 65 ∨ list.getLast? = some 85) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_theorem_l11_1192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_average_speed_with_stoppages_l11_1178

/-- The average speed of a bus including stoppages, given its speed excluding stoppages and stop duration. -/
noncomputable def average_speed_with_stoppages (speed_without_stoppages : ℝ) (stop_duration : ℝ) : ℝ :=
  speed_without_stoppages * (1 - stop_duration / 60)

/-- Theorem: The average speed of a bus including stoppages is 40 km/hr. -/
theorem bus_average_speed_with_stoppages :
  let speed_without_stoppages : ℝ := 80
  let stop_duration : ℝ := 30
  average_speed_with_stoppages speed_without_stoppages stop_duration = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_average_speed_with_stoppages_l11_1178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_camp_sizes_l11_1152

def total_students : ℕ := 500
def sample_size : ℕ := 50
def start_number : ℕ := 3
def step_size : ℕ := total_students / sample_size

def camp1_end : ℕ := 200
def camp2_end : ℕ := 355

def selected_in_range (start_range end_range : ℕ) : ℕ :=
  ((end_range - start_range + 1) + (step_size - (start_range - start_number) % step_size) % step_size) / step_size

theorem systematic_sampling_camp_sizes :
  (selected_in_range 1 camp1_end,
   selected_in_range (camp1_end + 1) camp2_end,
   selected_in_range (camp2_end + 1) total_students) = (20, 16, 14) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_camp_sizes_l11_1152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l11_1118

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Define the set M
def M : Set ℝ := {x | x < -1 ∨ x > 1}

-- State the theorem
theorem f_properties :
  -- f is monotonically increasing on [-1, +∞)
  (∀ x y, -1 ≤ x ∧ x ≤ y → f x ≤ f y) →
  -- M is the solution set for |x + 1| + 1 < |2x + 1|
  (M = {x | f x + 1 < |2*x + 1|}) ∧
  -- For all a, b in M, |ab + 1| > |a + b|
  (∀ a b, a ∈ M → b ∈ M → |a*b + 1| > |a + b|) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l11_1118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_multiples_with_n_divisors_l11_1166

/-- A positive integer is squarefree if its prime factorization contains no repeated prime factors. -/
def IsSquarefree (n : ℕ+) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ^ 2 : ℕ) ∣ (n : ℕ) → False

/-- A number m is a multiple of n with exactly n positive divisors. -/
def IsMultipleWithNDivisors (n m : ℕ+) : Prop :=
  (n : ℕ) ∣ (m : ℕ) ∧ (Finset.card (Finset.filter (fun d => (d : ℕ) ∣ (m : ℕ)) (Finset.range m.val.succ)) = n)

/-- The set of multiples of n with exactly n positive divisors is finite. -/
def HasFiniteMultiplesWithNDivisors (n : ℕ+) : Prop :=
  Set.Finite {m : ℕ+ | IsMultipleWithNDivisors n m}

/-- The main theorem: a positive integer n has only finitely many multiples with exactly n divisors
    if and only if n is squarefree or n = 4. -/
theorem finite_multiples_with_n_divisors (n : ℕ+) :
  HasFiniteMultiplesWithNDivisors n ↔ IsSquarefree n ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_multiples_with_n_divisors_l11_1166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_above_line_l11_1144

noncomputable def points : List (ℝ × ℝ) := [(4, 15), (8, 25), (10, 30), (14, 40), (18, 45), (22, 55)]

noncomputable def isAboveLine (point : ℝ × ℝ) : Bool :=
  point.2 > 3 * point.1 + 5

noncomputable def sumOfXCoordinatesAboveLine (points : List (ℝ × ℝ)) : ℝ :=
  (points.filter isAboveLine).map (·.1) |>.sum

theorem sum_of_x_coordinates_above_line :
  sumOfXCoordinatesAboveLine points = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_above_line_l11_1144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_xy_value_l11_1122

theorem min_xy_value (x y : ℝ) 
  (h : 1 + Real.cos (2*x + 3*y - 1)^2 = (x^2 + y^2 + 2*(x+1)*(1-y)) / (x-y+1)) : 
  ∃ (z : ℝ), z = x*y ∧ z ≥ 1/25 ∧ (∀ (w : ℝ), w = x*y → w ≥ z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_xy_value_l11_1122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_lovers_count_is_nine_l11_1136

/-- Represents the purchase of pizzas at Piazzanos Pizzeria -/
structure PizzaPurchase where
  standard_price : ℕ
  triple_cheese_count : ℕ
  meat_lovers_count : ℕ
  total_cost : ℕ

/-- Calculates the effective cost of triple cheese pizzas -/
def triple_cheese_cost (p : PizzaPurchase) : ℕ :=
  p.standard_price * (p.triple_cheese_count / 2)

/-- Calculates the effective cost of meat lovers pizzas -/
def meat_lovers_cost (p : PizzaPurchase) : ℕ :=
  p.standard_price * ((p.meat_lovers_count + 1) / 3 * 2)

/-- Theorem stating the number of meat lovers pizzas purchased -/
theorem meat_lovers_count_is_nine (p : PizzaPurchase) 
  (h1 : p.standard_price = 5)
  (h2 : p.triple_cheese_count = 10)
  (h3 : p.total_cost = 55) : 
  p.meat_lovers_count = 9 := by
  sorry

#check meat_lovers_count_is_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_lovers_count_is_nine_l11_1136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_solution_l11_1103

-- Define the function f(x) = x^3 + 2x - 9
def f (x : ℝ) : ℝ := x^3 + 2*x - 9

-- Define the partial function values
def partial_values : List (ℝ × ℝ) := [
  (1, -6),
  (2, 3),
  (1.5, -2.625),
  (1.625, -1.459),
  (1.75, -0.14),
  (1.875, 1.3418),
  (1.8125, 0.5793)
]

-- Define the accuracy
def accuracy : ℝ := 0.1

-- Theorem statement
theorem approximate_solution :
  ∃ (x : ℝ), (x ≥ 1.75 ∧ x ≤ 1.85) ∧ 
  (∀ (y : ℝ), y ∈ (partial_values.map Prod.fst) → |x - y| ≤ accuracy) ∧
  (∃ (a b : ℝ), (a, f a) ∈ partial_values ∧ (b, f b) ∈ partial_values ∧ f a * f b ≤ 0 ∧ |a - b| ≤ accuracy) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_solution_l11_1103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_to_line_l11_1100

noncomputable section

/-- The parabola y = x^2 - 4x + 4 -/
def parabola (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- The line y = 2x - 3 -/
def line (x : ℝ) : ℝ := 2*x - 3

/-- Distance between a point (x, parabola x) and the line y = 2x - 3 -/
def distance_to_line (x : ℝ) : ℝ := 
  abs (2*x - (parabola x) - 3) / Real.sqrt 5

/-- Theorem stating the existence of the shortest distance and its value -/
theorem shortest_distance_parabola_to_line :
  ∃ (min_dist : ℝ), min_dist = 2 * Real.sqrt 5 / 5 ∧
  ∀ (x : ℝ), distance_to_line x ≥ min_dist := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_to_line_l11_1100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_increase_first_digit_l11_1138

def original_decimal : ℚ := 25678 / 100000

theorem max_increase_first_digit :
  ∀ i : Fin 5,
  (80000 + (original_decimal * 100000 - 20000)) / 100000 ≥
  (if i.val = 0 then (original_decimal * 100000 + 60000) / 100000
   else if i.val = 1 then (original_decimal * 100000 + 3000) / 100000
   else if i.val = 2 then (original_decimal * 100000 + 200) / 100000
   else if i.val = 3 then (original_decimal * 100000 + 10) / 100000
   else (original_decimal * 100000) / 100000) :=
by
  intro i
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_increase_first_digit_l11_1138
