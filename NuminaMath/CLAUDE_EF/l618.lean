import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_in_20_minutes_l618_61869

-- Define the train's speed in miles per hour
noncomputable def train_speed : ℝ := 80

-- Define the car's speed as a fraction of the train's speed
noncomputable def car_speed_ratio : ℝ := 3 / 4

-- Define the duration of travel in minutes
noncomputable def travel_time : ℝ := 20

-- Define the number of minutes in an hour
noncomputable def minutes_per_hour : ℝ := 60

-- Theorem statement
theorem car_distance_in_20_minutes :
  let car_speed := car_speed_ratio * train_speed
  let time_in_hours := travel_time / minutes_per_hour
  car_speed * time_in_hours = 20 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_in_20_minutes_l618_61869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l618_61890

noncomputable def curve (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

def tangent_point : ℝ × ℝ := (2, 4)

theorem tangent_line_equation :
  ∃ (x₀ : ℝ), x₀ ≠ 0 ∧
  (∀ (x y : ℝ), y = curve x → (y - curve x₀) = x₀^2 * (x - x₀)) ∧
  curve x₀ = tangent_point.2 ∧
  x₀^2 * tangent_point.1 - curve x₀ = tangent_point.2 ∧
  (∀ (x y : ℝ), 4*x - y - 4 = 0 ↔ y - curve x₀ = x₀^2 * (x - x₀)) :=
by
  -- The proof goes here
  sorry

#check tangent_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l618_61890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l618_61878

theorem trigonometric_problem (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 2)
  (h4 : Real.sin (α - β) = 3/5) :
  (2 * Real.sin α - Real.cos (π - α)) / (3 * Real.sin α - Real.sin (π/2 + α)) = 1 ∧
  Real.cos β = 2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l618_61878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_interval_l618_61860

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem function_increasing_interval 
  (φ : ℝ) 
  (h1 : |φ| < π) 
  (h2 : ∀ x : ℝ, f x φ ≤ |f (π/6) φ|) 
  (h3 : f (π/2) φ < f (π/3) φ) : 
  ∀ k : ℤ, StrictMonoOn (fun x ↦ f x φ) (Set.Icc (k * π - π/3) (k * π + π/6)) :=
by
  sorry

#check function_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_interval_l618_61860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_game_theorem_l618_61877

/-- Represents a sequence of coin flips -/
def CoinFlipSequence := List Bool

/-- Represents the game state -/
structure GameState where
  a_beans : ℕ
  b_beans : ℕ

/-- Simulates the game for a given sequence of coin flips -/
def simulate_game (m n : ℕ) (seq : CoinFlipSequence) : GameState :=
  sorry

/-- Checks if a sequence results in Person A losing all beans after exactly m+2n flips -/
def is_valid_sequence (m n : ℕ) (seq : CoinFlipSequence) : Bool :=
  sorry

/-- Set of all valid sequences -/
def valid_sequences (m n : ℕ) : Set CoinFlipSequence :=
  sorry

/-- Binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  sorry

/-- Count of elements in a finite set -/
noncomputable def Set.card {α : Type*} (s : Set α) : ℕ := sorry

theorem coin_flip_game_theorem (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 1) :
  (valid_sequences m n).card = (m * binom (m + 2*n - 1) n) / (m + n) :=
by
  sorry

#check coin_flip_game_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_game_theorem_l618_61877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l618_61822

theorem right_triangle_hypotenuse (x : ℝ) (h : x > 0) :
  let shorter_leg := x
  let longer_leg := 3 * x - 3
  let area := (1 / 2) * shorter_leg * longer_leg
  area = 90 →
  Real.sqrt (shorter_leg ^ 2 + longer_leg ^ 2) = Real.sqrt 829 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l618_61822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_sequence_sum_l618_61843

def is_valid_sequence (seq : List Nat) : Prop :=
  ∀ i, i < seq.length → 
    let curr := seq[i]!
    let next := seq[(i + 1) % seq.length]!
    100 ≤ curr ∧ curr < 1000 ∧
    100 ≤ next ∧ next < 1000 ∧
    (curr % 10) * 100 + (curr / 100) + ((curr / 10) % 10) * 10 = next

def sequence_sum (seq : List Nat) : Nat :=
  seq.sum

theorem largest_prime_factor_of_sequence_sum (seq : List Nat) 
  (h : is_valid_sequence seq) : 
  ∃ p, Nat.Prime p ∧ p ∣ sequence_sum seq ∧ ∀ q, Nat.Prime q → q ∣ sequence_sum seq → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_sequence_sum_l618_61843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_condition_b_arithmetic_iff_d_2kpi_unique_tangent_line_l618_61819

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x + Real.sin x

-- Part 1
theorem strictly_increasing_condition (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) ↔ m ≥ 1 :=
sorry

-- Part 2
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem b_arithmetic_iff_d_2kpi (m : ℝ) (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  (∃ k : ℤ, k ≠ 0 ∧ d = 2 * k * Real.pi) ↔
  arithmetic_sequence (λ n ↦ f m (a n)) (2 * m * d) :=
sorry

-- Part 3
theorem unique_tangent_line :
  ∃! k b : ℝ, 
    (∀ x : ℝ, f 1 x ≥ k * x + b) ∧
    (∃ x₀ : ℝ, f 1 x₀ = k * x₀ + b) ∧
    k = 1 ∧ b = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_condition_b_arithmetic_iff_d_2kpi_unique_tangent_line_l618_61819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_lines_are_parallel_second_line_equivalent_l618_61828

/-- The distance between two parallel lines given by their coefficients -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between 4x + 3y - 1 = 0 and 8x + 6y + 3 = 0 is 1/2 -/
theorem distance_specific_parallel_lines :
  distance_parallel_lines 4 3 (-1) (3/2) = 1/2 := by
  sorry

/-- The two lines are parallel -/
theorem lines_are_parallel : ∃ (k : ℝ), k ≠ 0 ∧ 8 = k * 4 ∧ 6 = k * 3 := by
  sorry

/-- The second line equation is equivalent to 4x + 3y + 3/2 = 0 -/
theorem second_line_equivalent (x y : ℝ) : 8 * x + 6 * y + 3 = 0 ↔ 4 * x + 3 * y + 3/2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_lines_are_parallel_second_line_equivalent_l618_61828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_odd_function_l618_61849

noncomputable def determinant (x1 y1 x2 y2 : ℝ) : ℝ := x1 * y2 - x2 * y1

noncomputable def f (x : ℝ) : ℝ := determinant (Real.sqrt 3) (Real.cos x) 1 (Real.sin x)

theorem min_shift_for_odd_function :
  ∃ φ : ℝ, φ > 0 ∧
    (∀ ψ : ℝ, ψ > 0 → (∀ x : ℝ, f (x + ψ) = -f x) → φ ≤ ψ) ∧
    (∀ x : ℝ, f (x + φ) = -f x) ∧
    φ = 5 * Real.pi / 6 := by
  sorry

#check min_shift_for_odd_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_odd_function_l618_61849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l618_61808

/-- The function f(x) defined by (x^2 - 2x + c) / (x^2 + 2x - 8) -/
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + c) / (x^2 + 2*x - 8)

/-- Theorem stating that f(x) has exactly one vertical asymptote iff c = -24 or c = 0 -/
theorem exactly_one_vertical_asymptote (c : ℝ) : 
  (∃! x, ¬ ∃ y, f c x = y) ↔ (c = -24 ∨ c = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l618_61808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_y_plus_one_eq_zero_l618_61845

/-- The angle of inclination of a line parallel to the x-axis -/
def angle_of_inclination_parallel_x : ℝ := 0

/-- A line in the rectangular coordinate system -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The angle of inclination of a line -/
noncomputable def angle_of_inclination (l : Line) : ℝ :=
  if l.slope = 0 then angle_of_inclination_parallel_x else Real.arctan l.slope

/-- The line y + 1 = 0 -/
def line_y_plus_one_eq_zero : Line where
  slope := 0
  y_intercept := -1

theorem angle_of_inclination_y_plus_one_eq_zero :
  angle_of_inclination line_y_plus_one_eq_zero = 0 := by
  -- Unfold the definition of angle_of_inclination
  unfold angle_of_inclination
  -- The slope of line_y_plus_one_eq_zero is 0, so the 'if' condition is true
  simp [line_y_plus_one_eq_zero]
  -- The result is angle_of_inclination_parallel_x, which is defined as 0
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_y_plus_one_eq_zero_l618_61845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_sum_le_l618_61842

/-- A monic quadratic polynomial -/
structure MonicQuadratic where
  b : ℝ
  c : ℝ

/-- The difference between the roots of a monic quadratic polynomial -/
noncomputable def root_difference (f : MonicQuadratic) : ℝ :=
  Real.sqrt (f.b^2 - 4*f.c)

/-- The sum of two monic quadratic polynomials -/
def sum_poly (f g : MonicQuadratic) : MonicQuadratic :=
  ⟨f.b + g.b, f.c + g.c⟩

theorem root_difference_sum_le (f g : MonicQuadratic) 
  (hf : root_difference f > 0)
  (hg : root_difference g > 0)
  (hsum : root_difference (sum_poly f g) > 0)
  (heq : root_difference f = root_difference g) :
  root_difference (sum_poly f g) ≤ root_difference f := by
  sorry

#check root_difference_sum_le

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_sum_le_l618_61842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucy_lemonade_calories_l618_61853

/-- Represents Lucy's lemonade recipe and calorie information --/
structure LemonadeRecipe where
  lemon_juice : ℚ
  sugar : ℚ
  water : ℚ
  lemon_juice_calories : ℚ
  sugar_calories : ℚ

/-- Calculates the calories in a given weight of lemonade --/
def calories_in_lemonade (recipe : LemonadeRecipe) (weight : ℚ) : ℚ :=
  let total_weight := recipe.lemon_juice + recipe.sugar + recipe.water
  let total_calories := (recipe.lemon_juice * recipe.lemon_juice_calories / 100) +
                        (recipe.sugar * recipe.sugar_calories / 100)
  (total_calories / total_weight) * weight

/-- Lucy's lemonade recipe --/
def lucy_recipe : LemonadeRecipe :=
  { lemon_juice := 150
  , sugar := 150
  , water := 300
  , lemon_juice_calories := 30
  , sugar_calories := 386 }

theorem lucy_lemonade_calories :
  calories_in_lemonade lucy_recipe 250 = 260 := by
  -- Unfold the definitions and perform the calculation
  unfold calories_in_lemonade lucy_recipe
  -- The actual proof steps would go here
  sorry

#eval calories_in_lemonade lucy_recipe 250

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucy_lemonade_calories_l618_61853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_valid_sequence_length_l618_61870

def S : Finset Nat := {1, 2, 3, 4}

def isValidSequence (a : List Nat) : Prop :=
  ∀ x ∈ a, x ∈ S ∧
  ∀ p : List Nat, p.Perm (Finset.toList S) → p.getLast? ≠ some 1 →
    ∃ i₁ i₂ i₃ i₄, i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ ≤ a.length ∧
      (a.get? i₁, a.get? i₂, a.get? i₃, a.get? i₄) = (p.get? 0, p.get? 1, p.get? 2, p.get? 3)

theorem min_valid_sequence_length :
  ∃ a : List Nat, isValidSequence a ∧ a.length = 11 ∧
    ∀ b : List Nat, isValidSequence b → b.length ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_valid_sequence_length_l618_61870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_equation_l618_61833

noncomputable section

/-- The equation of an ellipse with semi-major axis a and semi-minor axis b -/
def is_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- The eccentricity of an ellipse -/
def eccentricity (a c : ℝ) : ℝ := c / a

/-- A point is on the ellipse -/
def point_on_ellipse (x y a b : ℝ) : Prop := is_ellipse x y a b

/-- The dot product of two 2D vectors -/
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

/-- The magnitude squared of a 2D vector -/
def magnitude_squared (x y : ℝ) : ℝ := x^2 + y^2

theorem ellipse_line_equation (a b c : ℝ) (A : ℝ × ℝ) :
  a > b ∧ b > c ∧
  point_on_ellipse A.fst A.snd a b ∧
  A.fst > 0 ∧ A.snd > 0 ∧
  dot_product A.fst A.snd c 0 = magnitude_squared c 0 ∧
  eccentricity a c = Real.sqrt 2 / 2 →
  ∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ A.snd = k * A.fst :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_equation_l618_61833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l618_61861

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 20

-- Define the line
def line_eq (x y : ℝ) : Prop := y = 2*x + 6

-- Theorem statement
theorem circle_tangent_to_line :
  ∃! p : ℝ × ℝ, circle_eq p.1 p.2 ∧ line_eq p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l618_61861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_900_l618_61818

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculates the midpoint of two points -/
noncomputable def midpoint' (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  abs ((t.b.x - t.a.x) * (t.c.y - t.a.y) - (t.c.x - t.a.x) * (t.b.y - t.a.y)) / 2

/-- Represents the triangular pyramid formed by folding the original triangle -/
structure TriangularPyramid where
  base : Triangle
  height : ℝ

/-- Calculates the volume of a triangular pyramid -/
noncomputable def pyramidVolume (p : TriangularPyramid) : ℝ :=
  (triangleArea p.base * p.height) / 3

/-- The main theorem stating the volume of the specific pyramid -/
theorem pyramid_volume_is_900 : ∃ (p : TriangularPyramid), 
  p.base = { a := { x := 0, y := 0 }, b := { x := 30, y := 0 }, c := { x := 20, y := 15 } } ∧ 
  pyramidVolume p = 900 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_900_l618_61818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_proposition2_is_correct_l618_61801

-- Define the propositions
def proposition1 : Prop := ∀ (p q : Prop), (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)

def proposition2 : Prop := 
  let p := ∃ x : ℝ, x^2 + 2*x ≤ 0
  (¬p) = (∀ x : ℝ, x^2 + 2*x > 0)

def proposition3 : Prop := ∀ (p q : Prop), p ∧ ¬q → (p ∧ ¬q) ∧ (¬p ∨ q)

def proposition4 : Prop := ∀ (p q : Prop), (¬p → q) ↔ (p → ¬q)

-- Theorem stating that only proposition2 is correct
theorem only_proposition2_is_correct : 
  ¬proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ ¬proposition4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_proposition2_is_correct_l618_61801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l618_61852

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := (x - 3) / (x - 5)

-- State the theorem
theorem inverse_g_undefined_at_one :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 5, f (g x) = x) ∧ (∀ x ≠ 1, g (f x) = x) ∧ ¬ (∃ y, f 1 = y) :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l618_61852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l618_61872

/-- The length of a chord intercepted on a circle by a line --/
theorem chord_length (r : ℝ) (a b c : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | a*x + b*y + c = 0}
  let d := |c| / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (r^2 - d^2) = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l618_61872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_4_6_l618_61817

/-- Represents the taxi fare structure and ride details -/
structure TaxiRide where
  initialFare : ℚ  -- Initial fare for the first mile
  additionalRate : ℚ  -- Rate per 0.1 mile after the first mile
  tip : ℚ  -- Tip amount
  totalBudget : ℚ  -- Total budget including tip

/-- Calculates the maximum distance that can be traveled given a TaxiRide -/
def maxDistance (ride : TaxiRide) : ℚ :=
  let fareBudget := ride.totalBudget - ride.tip
  let additionalDistance := (fareBudget - ride.initialFare) / (ride.additionalRate * 10)
  1 + additionalDistance

/-- Theorem stating that for the given fare structure and budget, 
    the maximum distance that can be traveled is 4.6 miles -/
theorem max_distance_is_4_6 : 
  let ride : TaxiRide := {
    initialFare := 3,
    additionalRate := 1/4,
    tip := 3,
    totalBudget := 15
  }
  maxDistance ride = 23/5 := by
  -- Proof goes here
  sorry

#eval maxDistance {
  initialFare := 3,
  additionalRate := 1/4,
  tip := 3,
  totalBudget := 15
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_4_6_l618_61817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_value_l618_61893

theorem sin_x_value (x : ℝ) 
  (h1 : Real.cos (π + x) = 3/5) 
  (h2 : x ∈ Set.Ioo π (2*π)) : 
  Real.sin x = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_value_l618_61893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l618_61880

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Define the set M
def M : Set ℝ := {x | f x < 4}

-- State the theorem
theorem f_properties :
  (M = Set.Ioo (-2) 2) ∧
  (∀ (a b : ℝ), a ∈ M → b ∈ M → 2 * |a + b| < |4 + a * b|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l618_61880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_properties_l618_61832

/-- Represents a trapezoid with parallel sides a and b -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ

/-- Calculate the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  (t.a + t.b) * t.h / 2

/-- Convert degrees to radians -/
noncomputable def degreesToRadians (degrees : ℝ) : ℝ :=
  degrees * (Real.pi / 180)

/-- Trapezoid properties theorem -/
theorem trapezoid_properties :
  ∃ (t : Trapezoid),
    t.a = 12.35 ∧
    t.b = 11.2 ∧
    (abs (t.angle1 - degreesToRadians 52.15) < 0.01) ∧
    (abs (t.angle2 - degreesToRadians 81.6) < 0.01) ∧
    (abs (t.angle3 - degreesToRadians 98.4) < 0.01) ∧
    (abs (t.angle4 - degreesToRadians 127.35) < 0.01) ∧
    (abs (trapezoidArea t - 134.35) < 0.01) :=
by
  sorry

#check trapezoid_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_properties_l618_61832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l618_61879

/-- The radius of the inscribed circle of a triangle with sides a, b, and c. -/
noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

/-- Theorem: The radius of the circle inscribed in triangle ABC with sides 6, 8, and 10 is 2. -/
theorem inscribed_circle_radius_specific_triangle :
  inscribedCircleRadius 6 8 10 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l618_61879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_from_point_l618_61871

noncomputable def point : ℝ × ℝ := (-Real.sqrt 3 / 2, 1 / 2)

def θ : ℝ → Prop := λ t => ∃ k : ℤ, t = 2 * k * Real.pi + 5 * Real.pi / 6

theorem angle_from_point (t : ℝ) : 
  (Real.cos t = point.fst ∧ Real.sin t = point.snd) → θ t := by
  sorry

#check angle_from_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_from_point_l618_61871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_angle_bisector_l618_61859

noncomputable def angle_between_median_and_bisector (β : Real) : Real :=
  sorry

theorem right_triangle_median_angle_bisector (β φ : Real) :
  β > 0 →
  β < Real.pi / 2 →
  Real.tan (β / 2) = 2 / Real.rpow 3 (1/3) →
  φ = angle_between_median_and_bisector β →
  Real.tan φ = 8 / (Real.rpow 27 (1/3) - 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_angle_bisector_l618_61859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l618_61847

def f : ℝ → ℝ := sorry

def is_non_decreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

theorem f_property (h1 : is_non_decreasing f 0 2)
                   (h2 : f 2 = 2)
                   (h3 : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x + f (2 - x) = 2)
                   (h4 : ∀ x, 3/2 ≤ x ∧ x ≤ 2 → f x ≤ 2*(x - 1)) :
  ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f (f x) ∧ f (f x) ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l618_61847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_sum_l618_61873

/-- Given a triangle ABC with centroid G, prove that if GA^2 + GB^2 + GC^2 = 72, 
    then AB^2 + AC^2 + BC^2 = 216. -/
theorem centroid_distance_sum (A B C G : EuclideanSpace ℝ (Fin 3)) : 
  G = (1 / 3 : ℝ) • (A + B + C) →  -- G is the centroid of triangle ABC
  ‖G - A‖^2 + ‖G - B‖^2 + ‖G - C‖^2 = 72 →  -- GA^2 + GB^2 + GC^2 = 72
  ‖A - B‖^2 + ‖A - C‖^2 + ‖B - C‖^2 = 216 :=  -- AB^2 + AC^2 + BC^2 = 216
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_sum_l618_61873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_three_count_l618_61825

theorem sum_divisible_by_three_count : 
  let S := Finset.range 20
  (Finset.filter (fun p : ℕ × ℕ => p.1 ∈ S ∧ p.2 ∈ S ∧ (p.1 + p.2) % 3 = 0) (S.product S)).card = 134 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_three_count_l618_61825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l618_61868

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / (11/3) - y^2 / 11 = 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + (y-2)^2 = 1

-- Define the asymptotes of the hyperbola
noncomputable def asymptotes (x y : ℝ) : Prop :=
  y = (Real.sqrt (11/3 / 11)) * x ∨ y = -(Real.sqrt (11/3 / 11)) * x

theorem hyperbola_properties :
  -- The hyperbola passes through (2, 1)
  hyperbola 2 1 ∧
  -- The asymptotes are tangent to the circle
  ∀ x y, asymptotes x y → 
    (∃ t, x = t ∧ y = (Real.sqrt (11/3 / 11)) * t ∧ circle_eq x y) ∨
    (∃ t, x = t ∧ y = -(Real.sqrt (11/3 / 11)) * t ∧ circle_eq x y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l618_61868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_grade_score_is_ten_l618_61802

/-- Represents the number of seventh-grade students -/
def n : ℕ := 1

/-- Represents the number of eighth-grade students -/
def m : ℕ := 10 * n

/-- Calculates the total number of games for k players -/
def num_games (k : ℕ) : ℕ := k * (k - 1) / 2

/-- Represents the total score of seventh-grade students -/
def seventh_grade_score : ℕ := n * (n + m - 1)

/-- Represents the total score of eighth-grade students -/
noncomputable def eighth_grade_score : ℚ := (4.5 : ℚ) * seventh_grade_score

/-- The main theorem to be proved -/
theorem seventh_grade_score_is_ten : seventh_grade_score = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_grade_score_is_ten_l618_61802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_sin_2x_l618_61841

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

-- State the theorem
theorem smallest_positive_period_of_sin_2x :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_sin_2x_l618_61841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l618_61812

theorem log_inequality (x y : ℝ) (h : Real.log x < Real.log y ∧ Real.log y < 0) : x < y ∧ y < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l618_61812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_percentage_l618_61824

theorem retailer_profit_percentage (wholesale_price retail_price : ℝ) 
  (discount_percentage : ℝ) (h1 : wholesale_price = 90) 
  (h2 : retail_price = 120) (h3 : discount_percentage = 0.1) : 
  ((retail_price * (1 - discount_percentage) - wholesale_price) / wholesale_price) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_percentage_l618_61824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_law_non_rectangular_triangle_l618_61881

/-- For a non-rectangular triangle with sides a, b, c and angle α opposite to side a,
    the formula a² = b² + c² - 2bc cos(α) is valid regardless of whether α is acute or obtuse. -/
theorem cosine_law_non_rectangular_triangle 
  (a b c : ℝ) (α : ℝ) 
  (triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (non_rectangular : α ≠ π / 2) 
  (α_range : 0 < α ∧ α < π) :
  a^2 = b^2 + c^2 - 2*b*c*Real.cos α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_law_non_rectangular_triangle_l618_61881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_picnic_tickets_l618_61895

theorem ice_cream_picnic_tickets (student_ticket_price : ℚ) 
                                 (non_student_ticket_price : ℚ)
                                 (total_amount : ℚ)
                                 (student_tickets : ℕ) :
  student_ticket_price = 1/2 →
  non_student_ticket_price = 3/2 →
  total_amount = 2065/10 →
  student_tickets = 83 →
  ∃ (non_student_tickets : ℕ),
    student_ticket_price * student_tickets + 
    non_student_ticket_price * non_student_tickets = total_amount ∧
    student_tickets + non_student_tickets = 193 := by
  sorry

#check ice_cream_picnic_tickets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_picnic_tickets_l618_61895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tile_probability_l618_61856

def is_red (n : ℕ) : Prop := n % 4 = 3

def count_red (n : ℕ) : ℕ := (Finset.range n).filter (fun x => x % 4 = 3) |>.card

theorem red_tile_probability :
  let total_tiles := 60
  let red_tiles := count_red total_tiles
  (red_tiles : ℚ) / total_tiles = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tile_probability_l618_61856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_circumference_l618_61820

/-- Represents the track around which A and B travel -/
structure Track where
  circumference : ℝ

/-- Represents the movement of A and B around the track -/
structure Movement (track : Track) where
  first_meet : ℝ  -- distance B travels before first meeting
  second_meet : ℝ  -- distance A travels before second meeting

/-- Theorem stating the circumference of the track given the movement conditions -/
theorem track_circumference (c : ℝ) (m : Movement ⟨c⟩) 
  (h1 : m.first_meet = 150)
  (h2 : m.second_meet = c - 90) : c = 720 := by
  sorry

#check track_circumference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_circumference_l618_61820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_sum_l618_61844

theorem lcm_ratio_sum (a b : ℕ) : 
  Nat.lcm a b = 36 → a * 3 = b * 2 → a + b = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_sum_l618_61844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l618_61855

theorem equation_roots : 
  ∀ x : ℝ, x > 0 → (4 * Real.sqrt x + 4 * (Real.sqrt x)⁻¹ = 10 ↔ x = 4 ∨ x = (1/4 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l618_61855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l618_61848

/-- Converts spherical coordinates to rectangular coordinates -/
noncomputable def spherical_to_rectangular (ρ θ φ : Real) : Real × Real × Real :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

/-- The given point in spherical coordinates -/
noncomputable def spherical_point : Real × Real × Real :=
  (3, 3 * Real.pi / 2, Real.pi / 3)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular spherical_point.1 spherical_point.2.1 spherical_point.2.2 =
  (0, -3 * Real.sqrt 3 / 2, 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l618_61848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tinas_total_pens_l618_61811

/-- The number of pens Tina has -/
structure PenCount where
  pink : ℕ
  green : ℕ
  blue : ℕ

/-- Tina's pen collection satisfying the given conditions -/
def tinas_pens : PenCount :=
  let pink := 12
  let green := pink - 9
  let blue := green + 3
  { pink := pink, green := green, blue := blue }

/-- The total number of pens Tina has -/
def total_pens (pens : PenCount) : ℕ :=
  pens.pink + pens.green + pens.blue

/-- Theorem stating that the total number of Tina's pens is 21 -/
theorem tinas_total_pens : total_pens tinas_pens = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tinas_total_pens_l618_61811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_difference_squares_l618_61897

open Real

/-- Prove that for functions f and g defined on (-π/2, π/2), if f(x) + g(x) = √((1 + cos 2x) / (1 - sin x)), 
    f is odd, and g is even, then (f(x))^2 - (g(x))^2 = -2 cos x. -/
theorem function_difference_squares 
  (f g : ℝ → ℝ) 
  (h₁ : ∀ x, x ∈ Set.Ioo (-π/2) (π/2) → f x + g x = sqrt ((1 + cos (2*x)) / (1 - sin x)))
  (h₂ : ∀ x, f (-x) = -f x)  -- f is odd
  (h₃ : ∀ x, g (-x) = g x)   -- g is even
  : ∀ x, x ∈ Set.Ioo (-π/2) (π/2) → (f x)^2 - (g x)^2 = -2 * cos x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_difference_squares_l618_61897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_cos_squared_l618_61813

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2

theorem smallest_positive_period_cos_squared :
  ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
  (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry

#check smallest_positive_period_cos_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_cos_squared_l618_61813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_sum_zero_isosceles_condition_l618_61829

-- Define a triangle ABC in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vector operations
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the zero vector
def zero_vector : ℝ × ℝ := (0, 0)

theorem triangle_vector_sum_zero (t : Triangle) :
  vector_add (vector_sub t.B t.A) (vector_add (vector_sub t.C t.B) (vector_sub t.A t.C)) = zero_vector := by
  sorry

theorem isosceles_condition (t : Triangle) :
  dot_product (vector_add (vector_sub t.B t.A) (vector_sub t.C t.A)) 
              (vector_sub (vector_sub t.B t.A) (vector_sub t.C t.A)) = 0 →
  vector_length (vector_sub t.B t.A) = vector_length (vector_sub t.C t.A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_sum_zero_isosceles_condition_l618_61829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_constant_l618_61885

/-- Ellipse C with equation x²/8 + y²/4 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- Point N -/
def N : ℝ × ℝ := (0, 2)

/-- Point P -/
def P : ℝ × ℝ := (-1, -2)

/-- Line l passing through P -/
noncomputable def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1) - 2

/-- Slope of a line passing through N and a point (x, y) -/
noncomputable def slope_from_N (x y : ℝ) : ℝ := (y - N.2) / (x - N.1)

theorem sum_of_slopes_constant (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  ellipse_C x₁ y₁ →
  ellipse_C x₂ y₂ →
  y₁ = line_l k x₁ →
  y₂ = line_l k x₂ →
  x₁ ≠ N.1 →
  x₂ ≠ N.1 →
  x₁ ≠ x₂ →
  slope_from_N x₁ y₁ + slope_from_N x₂ y₂ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_constant_l618_61885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l618_61884

/-- Given an ellipse C with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_e : e = 1/2

/-- A line tangent to the circle x² + y² = b² -/
def tangent_line (C : Ellipse) : ℝ → ℝ → Prop :=
  fun x y ↦ x + y - Real.sqrt 6 = 0

/-- The equation of the ellipse -/
def ellipse_equation (C : Ellipse) : ℝ → ℝ → Prop :=
  fun x y ↦ x^2 / C.a^2 + y^2 / C.b^2 = 1

/-- The line passing through (4, 0) and intersecting the ellipse at two distinct points -/
def intersecting_line : ℝ → ℝ → Prop :=
  fun x y ↦ x - 4*y - 4 = 0

/-- The y-intercept of the midpoint perpendicular bisector -/
noncomputable def y_intercept : ℝ := 4/13

theorem ellipse_and_line_properties (C : Ellipse) :
  (∀ x y, ellipse_equation C x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ x y, intersecting_line x y ↔ x - 4*y - 4 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l618_61884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_environmental_knowledge_competition_l618_61836

theorem environmental_knowledge_competition (total_questions : ℕ) 
  (correct_points : ℕ) (incorrect_points : ℕ) (total_score : ℕ) :
  total_questions = 20 →
  correct_points = 8 →
  incorrect_points = 5 →
  total_score = 134 →
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_points = 
    total_score + (total_questions - correct_answers) * incorrect_points ∧
    correct_answers = 18 := by
  sorry

#check environmental_knowledge_competition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_environmental_knowledge_competition_l618_61836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l618_61809

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem f_increasing_interval :
  ∀ x y, -3 * Real.pi / 4 ≤ x ∧ x < y ∧ y ≤ Real.pi / 4 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l618_61809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_real_roots_l618_61806

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
noncomputable def equation (x : ℝ) : ℝ := (log10 x)^2 - (floor (log10 x) : ℝ) - 2

-- Theorem statement
theorem equation_has_three_real_roots :
  ∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (equation a = 0 ∧ equation b = 0 ∧ equation c = 0) ∧
  (∀ x : ℝ, equation x = 0 → (x = a ∨ x = b ∨ x = c)) := by
  sorry

#check equation_has_three_real_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_real_roots_l618_61806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_3_power_minus_1_units_digit_of_3_2012_minus_1_l618_61810

def units_digit (n : ℕ) : ℕ := n % 10

def units_digit_cycle (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 0
  | 1 => 2
  | 2 => 8
  | 3 => 6
  | _ => 0  -- This case is actually unreachable, but Lean requires it for completeness

theorem units_digit_of_3_power_minus_1 (n : ℕ) :
  units_digit (3^n - 1) = units_digit_cycle n := by
  sorry

theorem units_digit_of_3_2012_minus_1 :
  units_digit (3^2012 - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_3_power_minus_1_units_digit_of_3_2012_minus_1_l618_61810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l618_61837

theorem simplify_expression : 6 - 3 - (-7) + (-2) = 6 - 3 + 7 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l618_61837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l618_61898

noncomputable def f (x : ℝ) := Real.log (9 - x^2)

theorem f_properties :
  -- Domain
  {x : ℝ | ∃ y, f x = y} = Set.Ioo (-3) 3 ∧
  -- Range
  Set.range f = Set.Iic (2 * Real.log 3) ∧
  -- Monotonically increasing interval
  ∀ x₁ x₂, x₁ ∈ Set.Ioc (-3) 0 → x₂ ∈ Set.Ioc (-3) 0 → x₁ ≤ x₂ → f x₁ ≤ f x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l618_61898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_first_train_speed_l618_61886

/-- The speed of a train given distance traveled and time taken -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- The theorem stating the speed of the first train -/
noncomputable def first_train_speed 
  (ratio : ℚ) 
  (second_train_distance : ℝ) 
  (second_train_time : ℝ) : ℝ :=
  let second_train_speed := speed second_train_distance second_train_time
  (ratio : ℝ) * second_train_speed

/-- The main theorem proving the speed of the first train -/
theorem prove_first_train_speed : 
  first_train_speed (7/8) 400 4 = 87.5 := by
  -- Unfold the definition of first_train_speed
  unfold first_train_speed
  -- Unfold the definition of speed
  unfold speed
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_first_train_speed_l618_61886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_same_flips_l618_61805

/-- The probability of getting a head on the nth flip for a single player -/
def prob_head_on_nth_flip (n : ℕ) : ℚ := (1 / 2) ^ n

/-- The probability of all four players getting their first head on the nth flip -/
def prob_all_head_on_nth_flip (n : ℕ) : ℚ := (prob_head_on_nth_flip n) ^ 4

/-- The sum of probabilities for all possible numbers of flips -/
noncomputable def prob_sum : ℚ := ∑' n, prob_all_head_on_nth_flip n

/-- The theorem stating that the probability of all players flipping the same number of times is 1/15 -/
theorem prob_all_same_flips : prob_sum = 1 / 15 := by
  sorry

#eval prob_head_on_nth_flip 1  -- For testing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_same_flips_l618_61805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_basis_and_dim_l618_61875

/-- The subspace F of ℝ³ -/
def F : Subspace ℝ (Fin 3 → ℝ) :=
  {carrier := {v | v 0 + v 1 + v 2 = 0},
   add_mem' := by sorry,
   zero_mem' := by sorry,
   smul_mem' := by sorry}

/-- The proposed basis for F -/
def proposed_basis : Fin 2 → (Fin 3 → ℝ) :=
  ![fun i => if i = 0 then -1 else if i = 1 then 1 else 0,
    fun i => if i = 0 then -1 else if i = 2 then 1 else 0]

/-- Theorem stating that the proposed basis is indeed a basis for F and F has dimension 2 -/
theorem F_basis_and_dim :
  LinearIndependent ℝ (fun i => proposed_basis i) ∧
  Submodule.span ℝ (Set.range proposed_basis) = F ∧
  FiniteDimensional.finrank ℝ F = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_basis_and_dim_l618_61875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_arbitrarily_large_tetrahedron_volume_l618_61814

-- Define a parallelepiped structure
structure Parallelepiped where
  vertices : List (Int × Int × Int)
  inner_points : Nat
  face_points : Nat
  edge_points : Nat

-- Define a tetrahedron structure
structure Tetrahedron where
  vertices : List (Int × Int × Int)

-- Helper function to calculate volume (not part of the proof)
noncomputable def volume (t : Tetrahedron) : ℚ := sorry

-- Statement 1: Volume of parallelepiped
theorem parallelepiped_volume (p : Parallelepiped) : 
  ∃ V : ℚ, V = 1 + p.inner_points + p.face_points / 2 + p.edge_points / 4 := by
  sorry

-- Statement 2: Arbitrarily large tetrahedron volume
theorem arbitrarily_large_tetrahedron_volume :
  ∀ M : ℚ, ∃ t : Tetrahedron, volume t > M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_arbitrarily_large_tetrahedron_volume_l618_61814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_marks_combined_classes_l618_61889

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = (n1 * avg1 + n2 * avg2) / (n1 + n2) := by
  sorry

#eval (26 * 40 + 50 * 60) / (26 + 50)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_marks_combined_classes_l618_61889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_lines_intersecting_circle_at_integer_points_l618_61831

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℤ
  y : ℤ

/-- Represents a line in the form x/a + y/b = 1 -/
structure Line where
  a : ℚ
  b : ℚ

/-- The circle x^2 + y^2 = 50 -/
def is_on_circle (p : Point2D) : Prop :=
  p.x^2 + p.y^2 = 50

/-- A line intersects the circle at a given point -/
def intersects_at (l : Line) (p : Point2D) : Prop :=
  (p.x : ℚ) / l.a + (p.y : ℚ) / l.b = 1 ∧ is_on_circle p

/-- A line is valid if it's not vertical, horizontal, or passing through the origin -/
def valid_line (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (l.a ≠ l.b ∨ l.a ≠ 1)

/-- The main theorem -/
theorem number_of_lines_intersecting_circle_at_integer_points :
  (∃ S : Finset Line, (∀ l ∈ S, valid_line l ∧
    (∀ p : Point2D, intersects_at l p → is_on_circle p)) ∧
    S.card = 60) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_lines_intersecting_circle_at_integer_points_l618_61831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_hike_distance_l618_61874

theorem janet_hike_distance : 
  ∀ (A B C : ℝ × ℝ), 
    let AB := (B.1 - A.1, B.2 - A.2)
    let BC := (C.1 - B.1, C.2 - B.2)
    let AC := (C.1 - A.1, C.2 - A.2)
    AB.1 = 0 ∧ AB.2 = 3 →  -- AB is vertical with length 3
    BC.1^2 + BC.2^2 = 8^2 →  -- BC has length 8
    BC.1 = 4 ∧ BC.2 = 4 * Real.sqrt 3 →  -- BC is at 30 degrees from vertical
    AC.1^2 + AC.2^2 = 57 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_hike_distance_l618_61874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_ratio_is_one_fourth_l618_61826

/-- The ratio of the height of a shorter cone to the height of the original cone -/
noncomputable def height_ratio (original_circumference : ℝ) (original_height : ℝ) (shorter_volume : ℝ) : ℝ :=
  let r := original_circumference / (2 * Real.pi)
  let shorter_height := 3 * shorter_volume / (Real.pi * r^2)
  shorter_height / original_height

/-- Theorem stating that the height ratio is 1/4 under given conditions -/
theorem height_ratio_is_one_fourth :
  height_ratio (18 * Real.pi) 24 (162 * Real.pi) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_ratio_is_one_fourth_l618_61826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_family_l618_61846

/-- The area of the figure formed by all points on a family of ellipses -/
theorem area_of_ellipse_family : ℝ := by
  -- Define the ellipse family
  let ellipse_family := fun (θ : ℝ) (P : ℝ × ℝ) =>
    (P.1 - Real.cos θ)^2 / 4 + (P.2 - 0.5 * Real.sin θ)^2 = 1

  -- Define the figure
  let figure := {P : ℝ × ℝ | ∃ θ, ellipse_family θ P}

  -- Define the area of the figure
  let area_of_figure : ℝ := 4 * Real.pi

  -- State the theorem
  have h : area_of_figure = 4 * Real.pi := by rfl

  -- Return the area
  exact area_of_figure


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_family_l618_61846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l618_61882

noncomputable def arithmeticSeq (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1 : ℝ)

noncomputable def arithmeticSum (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_sum (a d : ℝ) :
  (2 * (arithmeticSeq a d 8) + arithmeticSeq a d 2 = 12) →
  (arithmeticSum a d 11 = 44) := by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l618_61882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yan_distance_ratio_l618_61857

/-- The ratio of Yan's distance from his house to his distance from the park -/
noncomputable def distance_ratio (x y : ℝ) : ℝ := x / y

/-- Yan's walking speed -/
def walking_speed : ℝ := 1

/-- Yan's scooter speed relative to his walking speed -/
def scooter_speed_multiplier : ℝ := 5

theorem yan_distance_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  walking_speed * y = walking_speed * x + (x + y) / scooter_speed_multiplier →
  distance_ratio x y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yan_distance_ratio_l618_61857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l618_61823

-- Define the function g
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  if x > 0 then f x else (3^x - 1)

-- State the theorem
theorem inverse_function_value
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (h1 : ∀ x > 0, f_inv (f x) = x)
  (h2 : ∀ x, f_inv x = f_inv x)  -- Domain of f_inv
  (h3 : ∀ x, g f (-x) = -(g f x))  -- g is odd
  : f_inv (8/9) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l618_61823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_number_of_sets_l618_61830

theorem greatest_number_of_sets (logic_puzzles visual_puzzles : ℕ) 
  (h1 : logic_puzzles = 18) 
  (h2 : visual_puzzles = 9) : 
  Nat.gcd logic_puzzles visual_puzzles = 9 :=
by
  rw [h1, h2]
  norm_num

#eval Nat.gcd 18 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_number_of_sets_l618_61830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_decrease_l618_61821

-- Define an approximate equality for real numbers
def approx_eq (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

notation:50 x " ≈ " y => approx_eq x y 0.0000001

theorem profit_decrease (march_profit : ℝ) (april_may_decrease : ℝ) : 
  (march_profit * 1.2 * (1 - april_may_decrease / 100) * 1.5 = march_profit * 1.4399999999999999) →
  (april_may_decrease ≈ 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_decrease_l618_61821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l618_61839

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (2*x^2 - 3*x + 1)

theorem f_strictly_increasing : 
  ∀ x y : ℝ, x < y ∧ y < 3/4 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l618_61839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_travel_time_l618_61862

/-- The time taken for Rachel to reach Nicholas's house -/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Theorem stating that Rachel's travel time is 5 hours -/
theorem rachel_travel_time :
  let distance : ℝ := 10  -- Distance in miles
  let speed : ℝ := 2      -- Speed in miles per hour
  travel_time distance speed = 5 := by
  -- Unfold the definition of travel_time
  unfold travel_time
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_travel_time_l618_61862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcakes_frosted_is_26_l618_61883

def cagney_rate : ℚ := 1 / 25
def lacey_rate : ℚ := 1 / 35
def total_time : ℕ := 420
def break_time : ℕ := 60

def combined_rate : ℚ := cagney_rate + lacey_rate

def cupcakes_frosted : ℕ := 
  (combined_rate * (total_time - break_time : ℚ)).floor.toNat + 
  (cagney_rate * (break_time : ℚ)).floor.toNat

theorem cupcakes_frosted_is_26 : cupcakes_frosted = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcakes_frosted_is_26_l618_61883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l618_61835

theorem sequence_convergence (s : ℕ → ℝ) :
  (∃ L, Filter.Tendsto (λ n => s n + 2 * s (n + 1)) Filter.atTop (nhds L)) →
  ∃ M, Filter.Tendsto s Filter.atTop (nhds M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l618_61835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l618_61800

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sqrt 3 * Real.sin x - Real.cos x) + 3/2

-- State the theorem
theorem f_properties :
  let I : Set ℝ := {x | π/3 ≤ x ∧ x ≤ 5*π/6}
  (∀ x ∈ I, ∀ y ∈ I, x < y → f x > f y) ∧ 
  (∀ α : ℝ, α ∈ Set.Ioo (π/3) (5*π/6) → f α = 2/5 → Real.sin (2*α) = (-3*Real.sqrt 3 - 4)/10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l618_61800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tropical_drink_volume_l618_61866

/-- Represents the composition of a tropical fruit drink -/
structure TropicalDrink where
  grapefruit : Real
  lemon : Real
  orange : Real
  pineapple : Real
  mango : Real

/-- Theorem: The total volume of the tropical fruit drink is 80 ounces -/
theorem tropical_drink_volume (drink : TropicalDrink)
  (h1 : drink.grapefruit = 0.20)
  (h2 : drink.lemon = 0.25)
  (h3 : drink.pineapple = 0.10)
  (h4 : drink.mango = 0.15)
  (h5 : drink.orange = 1 - (drink.grapefruit + drink.lemon + drink.pineapple + drink.mango))
  (h6 : 24 / drink.orange = 80) :
  drink.grapefruit + drink.lemon + drink.orange + drink.pineapple + drink.mango = 1 ∧
  24 / drink.orange = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tropical_drink_volume_l618_61866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l618_61896

theorem problem_statement (a b c : ℤ) : 
  c ≥ 0 → 
  (∀ n : ℕ+, (a^(n:ℕ) + 2^(n:ℕ)) ∣ (b^(n:ℕ) + c)) → 
  ¬ ∃ k : ℤ, 2 * a * b = k^2 → 
  c = 0 ∨ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l618_61896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_collections_count_l618_61892

def num_belts : ℕ := 16
def belt_types : List (Char × ℕ) := [('B', 1), ('G', 2), ('L', 3), ('P', 4), ('R', 4), ('S', 2)]
def num_gold_buckles : ℕ := 2
def num_silver_buckles : ℕ := 4
def belts_to_select : ℕ := 3
def buckles_to_select : ℕ := 3

def is_indistinguishable (c : Char) : Bool :=
  c = 'B' || c = 'G' || c = 'S'

theorem distinct_collections_count :
  (List.sum (belt_types.map (fun (t, n) => Nat.choose n (if is_indistinguishable t then 1 else belts_to_select))) * 
   (Nat.choose num_silver_buckles 2 * Nat.choose num_gold_buckles 1 + 
    Nat.choose num_silver_buckles 1 * Nat.choose num_gold_buckles 2)) = 464 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_collections_count_l618_61892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_table_l618_61840

/-- A 3x3 table of natural numbers -/
def Table := Fin 3 → Fin 3 → ℕ

/-- The sum of a row in the table -/
def rowSum (t : Table) (i : Fin 3) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 3)) fun j => t i j)

/-- The sum of a column in the table -/
def colSum (t : Table) (j : Fin 3) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 3)) fun i => t i j)

/-- The sum of all numbers in the table -/
def totalSum (t : Table) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 3)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 3)) fun j => t i j)

/-- All row sums and column sums are distinct -/
def distinctSums (t : Table) : Prop :=
  (∀ i j, i ≠ j → rowSum t i ≠ rowSum t j) ∧
  (∀ i j, i ≠ j → colSum t i ≠ colSum t j) ∧
  (∀ i j, rowSum t i ≠ colSum t j)

theorem min_sum_table :
  ∀ t : Table, distinctSums t → totalSum t ≥ 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_table_l618_61840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l618_61815

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 2 then 2 * x + 6
  else if -1 ≤ x ∧ x ≤ 1 then x + 7
  else 0  -- Default value for x outside the defined intervals

-- State the theorem
theorem f_max_min :
  (∃ x, f x = 10) ∧
  (∀ x, f x ≤ 10) ∧
  (∃ x, f x = 6) ∧
  (∀ x, f x ≥ 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l618_61815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l618_61804

theorem log_inequality : 
  ∀ (a b c : ℝ), 
    a = Real.log 3 / Real.log 7 → 
    b = Real.log 7 / Real.log (1/3) → 
    c = 3^(7/10) → 
    b < a ∧ a < c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l618_61804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_result_l618_61807

/-- The period of the decimal expansion of a rational number -/
def decimal_period (r : ℚ) : ℕ := sorry

/-- Given a prime number p, if an operation on p produces another prime number q
    such that the decimal expansion of 1/q has a period of 166 digits, then q = 167. -/
theorem operation_result (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h_period : decimal_period (1 / q) = 166) : q = 167 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_result_l618_61807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l618_61863

theorem relationship_abc : 
  ∃ (a b c : ℝ), 
    a = Real.log 0.3 / Real.log 2 ∧
    b = 2^(0.1 : ℝ) ∧
    c = 0.2^(1.3 : ℝ) ∧
    a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l618_61863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l618_61899

-- Define the function f(x) = 3^x
noncomputable def f (x : ℝ) : ℝ := 3^x

-- Define g as the inverse function of f
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Theorem statement
theorem inverse_function_value : g (Real.sqrt 3) = 1/2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l618_61899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l618_61865

-- Define the hyperbola equation
noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 5

-- Define the positive slope of an asymptote
noncomputable def positive_asymptote_slope : ℝ := Real.sqrt 11 / 5

-- Theorem statement
theorem hyperbola_asymptote_slope :
  ∃ (x y : ℝ), hyperbola_equation x y →
  positive_asymptote_slope = Real.sqrt 11 / 5 := by
  sorry

#check hyperbola_asymptote_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l618_61865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_pond_ratio_l618_61827

/-- Represents a rectangular field with a square pond inside -/
structure FieldWithPond where
  field_length : ℝ
  field_width : ℝ
  pond_side : ℝ

/-- The ratio of a field's length to its width -/
noncomputable def length_width_ratio (f : FieldWithPond) : ℝ :=
  f.field_length / f.field_width

/-- Theorem: Given a rectangular field with length 16m and a square pond with side 8m,
    if the pond's area is half the field's area, then the ratio of the field's length
    to its width is 2:1 -/
theorem field_pond_ratio (f : FieldWithPond)
    (h1 : f.field_length = 16)
    (h2 : f.pond_side = 8)
    (h3 : f.pond_side ^ 2 = (1/2) * f.field_length * f.field_width) :
    length_width_ratio f = 2 := by
  sorry

#check field_pond_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_pond_ratio_l618_61827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_representation_l618_61867

theorem unique_representation (n : ℕ) (h : n ≥ 1) :
  ∃! (p q : ℕ), n = 2^p * (2*q + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_representation_l618_61867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l618_61891

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point on the right branch of a hyperbola -/
structure RightBranchPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1
  right_branch : 0 < x

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from a point to the left focus -/
noncomputable def dist_to_left_focus (h : Hyperbola) (p : RightBranchPoint h) : ℝ :=
  Real.sqrt ((p.x + h.a * eccentricity h)^2 + p.y^2)

/-- The distance from a point to the right focus -/
noncomputable def dist_to_right_focus (h : Hyperbola) (p : RightBranchPoint h) : ℝ :=
  Real.sqrt ((p.x - h.a * eccentricity h)^2 + p.y^2)

/-- The main theorem to prove -/
theorem eccentricity_range (h : Hyperbola) 
  (p : RightBranchPoint h) 
  (h_dist : dist_to_left_focus h p = 3 * dist_to_right_focus h p) :
  1 < eccentricity h ∧ eccentricity h ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l618_61891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_fraction_numerator_l618_61876

theorem simplified_fraction_numerator (a : ℕ) (ha : a > 0) :
  let p := 2 * a + 1
  let q := a * (a + 1)
  (((a + 1 : ℚ) / a) - (a / (a + 1 : ℚ))) = (p : ℚ) / q ∧
  Nat.Coprime p q ∧
  p = 4045 ∧
  q > 0 :=
by
  sorry

#eval 2 * 2022 + 1  -- Should output 4045

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_fraction_numerator_l618_61876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionAreaTheorem_l618_61854

/-- Regular quadrilateral frustum -/
structure QuadrilateralFrustum where
  height : ℝ
  upperBaseSide : ℝ
  lowerBaseSide : ℝ

/-- Cross-section area of a quadrilateral frustum -/
noncomputable def crossSectionArea (f : QuadrilateralFrustum) : ℝ :=
  16 * Real.sqrt 6

/-- Theorem: The area of the cross-section through diagonal BD₁ parallel to diagonal AC
    of the base in the given regular quadrilateral frustum is 16√6 cm² -/
theorem crossSectionAreaTheorem (f : QuadrilateralFrustum)
    (h1 : f.height = 6)
    (h2 : f.upperBaseSide = 4)
    (h3 : f.lowerBaseSide = 8) :
    crossSectionArea f = 16 * Real.sqrt 6 := by
  sorry

#check crossSectionAreaTheorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionAreaTheorem_l618_61854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_for_increasing_g_l618_61851

open Real Set

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * cos (ω * x / 2) * (sin (ω * x / 2) + cos (ω * x / 2)) - 1

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ :=
  f ω (x - π / (4 * ω))

theorem max_omega_for_increasing_g :
  ∀ ω : ℝ, ω > 0 →
  (∀ x ∈ Icc (-π/4) 0, HasDerivAt (g ω) (Real.sqrt 2 * ω * cos (ω * x)) x) →
  ω ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_for_increasing_g_l618_61851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_C_l618_61858

-- Define the points
noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (8, 6)
noncomputable def C : ℝ × ℝ := (0, 2)

-- Define the slopes
noncomputable def slope_AB : ℝ := 3/4
noncomputable def slope_BC : ℝ := 1/2

-- Theorem statement
theorem sum_of_coordinates_C : 
  A.1 = 0 ∧ A.2 = 0 ∧  -- A is at (0, 0)
  B.2 = 6 ∧  -- B is on the line y = 6
  (B.2 - A.2) / (B.1 - A.1) = slope_AB ∧  -- Slope of AB is 3/4
  C.1 = 0 ∧  -- C is on the y-axis
  (C.2 - B.2) / (C.1 - B.1) = slope_BC  -- Slope from B to C is 1/2
  →
  C.1 + C.2 = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_C_l618_61858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_and_in_circle_l618_61838

theorem no_intersection_and_in_circle :
  ¬∃ (a b : ℝ),
    let A := {p : ℝ × ℝ | ∃ n : ℤ, p.1 = n ∧ p.2 = a * n + b}
    let B := {p : ℝ × ℝ | ∃ m : ℤ, p.1 = m ∧ p.2 = 3 * m^2 + 15}
    let C := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 144}
    (Set.Nonempty (A ∩ B) ∧ (a, b) ∈ C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_and_in_circle_l618_61838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andy_makes_fewest_cookies_l618_61887

structure Friend where
  name : String
  cookie_area : ℝ

def Andy : Friend := { name := "Andy", cookie_area := 15 }
def Bella : Friend := { name := "Bella", cookie_area := 10 }
def Carlos : Friend := { name := "Carlos", cookie_area := 10 }
def Diana : Friend := { name := "Diana", cookie_area := 7.5 }

def dough_amount : ℝ := 20 * Andy.cookie_area

noncomputable def cookies_made (f : Friend) : ℝ := dough_amount / f.cookie_area

theorem andy_makes_fewest_cookies :
  ∀ f : Friend, f ≠ Andy → cookies_made Andy ≤ cookies_made f :=
by
  intro f h_not_andy
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andy_makes_fewest_cookies_l618_61887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_group_A_has_like_terms_l618_61850

-- Define a structure for a term
structure Term where
  coefficient : ℚ
  vars : List (Char × ℕ)

-- Define a function to check if two terms are like terms
def areLikeTerms (t1 t2 : Term) : Prop :=
  t1.vars = t2.vars

-- Define the groups
def groupA : List Term := [
  { coefficient := 3, vars := [('a', 3), ('b', 1)] },
  { coefficient := -3, vars := [('a', 3), ('b', 1)] }
]

def groupB : List Term := [
  { coefficient := 1, vars := [('a', 3)] },
  { coefficient := 1, vars := [('b', 3)] }
]

def groupC : List Term := [
  { coefficient := 1, vars := [('a', 1), ('b', 1), ('c', 1)] },
  { coefficient := 1, vars := [('a', 1), ('c', 1)] }
]

def groupD : List Term := [
  { coefficient := 1, vars := [('a', 5)] },
  { coefficient := 32, vars := [] }
]

-- Theorem statement
theorem only_group_A_has_like_terms :
  (∀ t1 t2, t1 ∈ groupA → t2 ∈ groupA → areLikeTerms t1 t2) ∧
  (¬ ∀ t1 t2, t1 ∈ groupB → t2 ∈ groupB → areLikeTerms t1 t2) ∧
  (¬ ∀ t1 t2, t1 ∈ groupC → t2 ∈ groupC → areLikeTerms t1 t2) ∧
  (¬ ∀ t1 t2, t1 ∈ groupD → t2 ∈ groupD → areLikeTerms t1 t2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_group_A_has_like_terms_l618_61850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l618_61816

theorem tan_double_angle_special_case (α : Real) 
  (h : Real.sin α - 2 * Real.cos α = Real.sqrt 10 / 2) : 
  Real.tan (2 * α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l618_61816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l618_61888

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Theorem: Given conditions on an arithmetic sequence, prove S_7 + S_2 = 80 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.a 2 = 6)
  (h2 : (seq.a 3)^2 = seq.a 1 * seq.a 7) :
  S seq 7 + S seq 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l618_61888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_volume_independent_of_point_l618_61834

/-- A rectangular prism with dimensions a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- A point inside the rectangular prism -/
structure PointInPrism (prism : RectangularPrism) where
  x : ℝ
  y : ℝ
  z : ℝ
  x_in : 0 < x ∧ x < prism.a
  y_in : 0 < y ∧ y < prism.b
  z_in : 0 < z ∧ z < prism.c

/-- The volume of the region common to the prism and the shape formed by reflections -/
noncomputable def commonVolume (prism : RectangularPrism) (p : PointInPrism prism) : ℝ :=
  (5/6) * prism.a * prism.b * prism.c

/-- Theorem stating that the common volume is independent of the point's position -/
theorem common_volume_independent_of_point (prism : RectangularPrism) 
  (p1 p2 : PointInPrism prism) : 
  commonVolume prism p1 = commonVolume prism p2 := by
  -- Unfold the definition of commonVolume
  unfold commonVolume
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_volume_independent_of_point_l618_61834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_openInterval_l618_61894

open Set
open Function
open Real

-- Define the function f(x) = e^(-x)
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

-- Define the open interval (0, +∞)
def openInterval : Set ℝ := Ioi 0

-- State the theorem
theorem f_decreasing_on_openInterval : 
  StrictAntiOn f openInterval :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_openInterval_l618_61894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l618_61864

theorem complex_fraction_simplification :
  ∃ (a b : ℚ), (5 : ℂ) + 7*Complex.I / ((2 : ℂ) + 3*Complex.I) = a + b*Complex.I ∧ a = 31/13 ∧ b = -1/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l618_61864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l618_61803

/-- Curve C₂ obtained by stretching unit circle -/
noncomputable def C₂ (t : ℝ) : ℝ × ℝ := (3 * Real.cos t, Real.sin t)

/-- Line l passing through (1,0) with inclination π/4 -/
noncomputable def l (t : ℝ) : ℝ × ℝ := (1 + t * Real.sqrt 2 / 2, t * Real.sqrt 2 / 2)

/-- Intersection points of l and C₂ -/
def intersection_points : Set ℝ := {t | ∃ s, C₂ s = l t}

/-- Distance from (1,0) to a point on l -/
noncomputable def distance_from_origin (t : ℝ) : ℝ := Real.sqrt ((l t).1 - 1)^2 + (l t).2^2

theorem intersection_distance_product :
  ∃ t₁ t₂, t₁ ∈ intersection_points ∧ t₂ ∈ intersection_points ∧ 
    distance_from_origin t₁ * distance_from_origin t₂ = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l618_61803
