import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l73_7372

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 0

-- Define the intersection points
def intersects (k : ℝ) : Prop := ∃ (A B : ℝ × ℝ), 
  circleEq A.1 A.2 ∧ circleEq B.1 B.2 ∧ 
  line k A.1 = A.2 ∧ line k B.1 = B.2

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ := 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The main theorem
theorem line_intersects_circle (k : ℝ) :
  intersects k → (∃ (A B : ℝ × ℝ), distance A B = 2 * Real.sqrt 2) → k = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l73_7372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l73_7365

/-- The range of m that satisfies the given conditions -/
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|x - 1| ≤ 2 → x^2 - 2*x + 1 - m^2 ≤ 0)) → 
  (m > 0) →
  (∀ x : ℝ, (x^2 - 2*x + 1 - m^2 > 0 → |x - 1| > 2)) →
  (∃ x : ℝ, |x - 1| ≤ 2 ∧ x^2 - 2*x + 1 - m^2 > 0) →
  m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l73_7365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_coefficients_function_below_line_l73_7321

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

-- Part I
theorem tangent_line_coefficients (a b : ℝ) :
  (∀ x, (2 * x + f a x + b = 0) ↔ (x = 1)) →
  a = -1 ∧ b = -1/2 := by sorry

-- Part II
theorem function_below_line (a : ℝ) :
  (∀ x > 1, f a x < 2 * a * x) →
  -1/2 ≤ a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_coefficients_function_below_line_l73_7321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angles_adjustment_l73_7349

/-- Two complementary angles with measures in the ratio 3:6 -/
def complementary_angles (a b : ℝ) : Prop :=
  a + b = 90 ∧ a / b = 1 / 2

/-- The smaller angle increased by 20% -/
noncomputable def increased_smaller_angle (a : ℝ) : ℝ := a * 1.2

/-- The required decrease in the larger angle -/
noncomputable def required_decrease (b : ℝ) : ℝ := (90 - increased_smaller_angle (90 / 3)) / b

theorem complementary_angles_adjustment (a b : ℝ) 
  (h : complementary_angles a b) : 
  required_decrease b = 0.9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angles_adjustment_l73_7349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_cost_price_l73_7368

/-- Represents the cost price of a toy -/
def cost_price : ℝ → Prop := sorry

/-- Represents the selling price of a toy -/
def selling_price : ℝ → Prop := sorry

/-- Represents the number of toys sold -/
def num_toys_sold : ℕ := 18

/-- Represents the total selling price of all toys -/
def total_selling_price : ℝ := 21000

/-- Represents the gain in terms of number of toys -/
def gain_in_toys : ℕ := 3

theorem toy_cost_price :
  cost_price 1000 ∧
  (∀ x, cost_price x → selling_price (x + x * gain_in_toys / num_toys_sold)) ∧
  (∀ x, selling_price x → x * num_toys_sold = total_selling_price) := by
  sorry

#check toy_cost_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_cost_price_l73_7368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petes_total_miles_rounded_l73_7340

/-- Represents a pedometer with a maximum count --/
structure Pedometer where
  max_count : ℕ

/-- Represents a postman's walking data for a year --/
structure PostmanData where
  pedometer : Pedometer
  resets : ℕ
  final_reading : ℕ
  steps_per_mile : ℕ

/-- Calculates the total steps walked based on pedometer data --/
def total_steps (data : PostmanData) : ℕ :=
  data.resets * (data.pedometer.max_count + 1) + data.final_reading

/-- Calculates the total miles walked based on total steps and steps per mile --/
def total_miles (steps : ℕ) (steps_per_mile : ℕ) : ℚ :=
  steps / steps_per_mile

/-- Rounds a rational number to the nearest hundred --/
def round_to_nearest_hundred (x : ℚ) : ℕ :=
  (((x + 50) / 100).floor : ℤ).toNat * 100

/-- Theorem stating that Pete walked 2200 miles (rounded to nearest hundred) --/
theorem petes_total_miles_rounded (data : PostmanData) 
  (h1 : data.pedometer.max_count = 99999)
  (h2 : data.resets = 35)
  (h3 : data.final_reading = 75500)
  (h4 : data.steps_per_mile = 1600) :
  round_to_nearest_hundred (total_miles (total_steps data) data.steps_per_mile) = 2200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petes_total_miles_rounded_l73_7340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_side_c_length_triangle_area_l73_7395

-- Define triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  Real.cos t.B * Real.cos t.C - Real.sin t.B * Real.sin t.C = 1/2 ∧
  t.a = 2 * Real.sqrt 3

-- Theorem 1
theorem angle_A_measure (t : Triangle) 
  (h : triangle_conditions t) : t.A = 2*Real.pi/3 := by sorry

-- Theorem 2
theorem side_c_length (t : Triangle) 
  (h : triangle_conditions t) (hb : t.b = 2) : t.c = 2 := by sorry

-- Theorem 3
theorem triangle_area (t : Triangle) 
  (h : triangle_conditions t) (hbc : t.b + t.c = 4) : 
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_side_c_length_triangle_area_l73_7395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l73_7334

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y - 6| / Real.sqrt 2

-- Theorem statement
theorem min_distance_to_line :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 / 2 ∧ 
    ∀ x y : ℝ, ellipse x y → distance_to_line x y ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l73_7334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_good_coloring_correct_l73_7352

/-- A coloring of points on a circle is "good" if there is at least one pair of black points such that 
    the interior of one of the arcs between them contains exactly n points. -/
def is_good_coloring (n : ℕ) (E : Finset ℕ) (black_points : Finset ℕ) : Prop :=
  ∃ a b, a ∈ black_points ∧ b ∈ black_points ∧ 
    (Finset.card (E.filter (λ x ↦ x > a ∧ x < b)) = n ∨ 
     Finset.card (E.filter (λ x ↦ x > b ∨ x < a)) = n)

/-- The smallest number of black points needed for all colorings to be "good" -/
def min_good_coloring (n : ℕ) : ℕ :=
  if n % 3 = 2 then n - 1 else n

theorem min_good_coloring_correct (n : ℕ) (h : n ≥ 3) :
  ∀ E : Finset ℕ, E.card = 2*n - 1 →
  ∀ k : ℕ, k ≥ min_good_coloring n →
  ∀ black_points : Finset ℕ, black_points ⊆ E ∧ black_points.card = k →
  is_good_coloring n E black_points :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_good_coloring_correct_l73_7352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_digits_convergence_l73_7370

def sumOfSquaresOfDigits (n : ℕ) : ℕ :=
  (n.digits 10).map (· ^ 2) |>.sum

def isCycleNumber (n : ℕ) : Prop :=
  n = 145 ∨ n = 42 ∨ n = 20 ∨ n = 4 ∨ n = 16 ∨ n = 37 ∨ n = 58 ∨ n = 89

def eventuallyReachesOneOrCycle (n : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate sumOfSquaresOfDigits k n = 1) ∨ isCycleNumber (Nat.iterate sumOfSquaresOfDigits k n)

theorem sum_of_squares_of_digits_convergence (n : ℕ) :
  eventuallyReachesOneOrCycle n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_digits_convergence_l73_7370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_percentage_x_l73_7318

/-- Represents the composition of a solution -/
structure Solution :=
  (x : ℚ) -- Percentage of liquid X
  (z : ℚ) -- Percentage of liquid Z
  (w : ℚ) -- Percentage of water

/-- Calculates the amount of a component in a solution given its percentage and total mass -/
def amount_of (percentage : ℚ) (total_mass : ℚ) : ℚ :=
  percentage * total_mass / 100

/-- Theorem stating the final percentage of liquid X in the mixture -/
theorem final_percentage_x 
  (y : Solution) 
  (initial_mass : ℚ) 
  (water_evaporated : ℚ) 
  (remaining_mass : ℚ) 
  (z_evaporated : ℚ) 
  (z_solution : Solution) 
  (z_solution_mass : ℚ) : 
  y.x = 40 ∧ y.z = 30 ∧ y.w = 30 ∧ 
  initial_mass = 10 ∧ 
  water_evaporated = 4 ∧ 
  remaining_mass = 6 ∧ 
  z_evaporated = 3 ∧
  z_solution.x = 60 ∧ z_solution.w = 40 ∧
  z_solution_mass = 5 →
  (amount_of y.x initial_mass + amount_of z_solution.x z_solution_mass) / 
  (remaining_mass - water_evaporated - z_evaporated + z_solution_mass) * 100 = 700 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_percentage_x_l73_7318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_cost_per_game_l73_7353

/-- Proves the manufacturing cost per game given initial investment, selling price, and break-even point -/
theorem manufacturing_cost_per_game 
  (initial_investment : ℚ) 
  (selling_price : ℚ) 
  (break_even_quantity : ℚ) 
  (h1 : initial_investment = 10410)
  (h2 : selling_price = 20)
  (h3 : break_even_quantity = 600) :
  (selling_price * break_even_quantity - initial_investment) / break_even_quantity = 265/100 := by
  sorry

#check manufacturing_cost_per_game

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_cost_per_game_l73_7353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l73_7306

/-- An inverse proportion function passing through (-2, 3) is in the second and fourth quadrants -/
theorem inverse_proportion_quadrants :
  ∀ k : ℝ,
  k ≠ 0 →
  (fun x : ℝ ↦ k / x) (-2) = 3 →
  ∀ x : ℝ,
  x ≠ 0 →
  (x > 0 → k / x < 0) ∧ (x < 0 → k / x > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l73_7306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_experiment_l73_7325

theorem seed_germination_experiment (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 25 / 100 →
  germination_rate2 = 35 / 100 →
  (((seeds_plot1 : ℚ) * germination_rate1 + (seeds_plot2 : ℚ) * germination_rate2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) = 29 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_experiment_l73_7325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l73_7397

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((a^2 - 3*a) * x) / Real.log (3 * a)

-- State the theorem
theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y, x < y → x < 0 → y < 0 → f a x > f a y) →
  a > 1/3 ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l73_7397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_k_l73_7393

/-- The function f(x) defined as a*ln(x) + b*x --/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x

/-- The derivative of f(x) --/
noncomputable def f_deriv (a b : ℝ) (x : ℝ) : ℝ := a / x + b

theorem tangent_line_and_max_k (a b : ℝ) :
  (f_deriv a b 1 = 1/2) →
  (f a b 1 = -1/2) →
  (∃ (k : ℝ), ∀ (x : ℝ), x > 1 → f a b x + k / x < 0) →
  (f a b = λ x ↦ Real.log x - (1/2) * x) ∧
  (∀ k : ℝ, (∀ x : ℝ, x > 1 → f a b x + k / x < 0) → k ≤ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_k_l73_7393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_through_points_unique_intersection_of_lines_central_projection_bijective_l73_7319

-- Define the basic structures
structure ProjectivePoint where

structure ProjectiveLine where

structure ProjectivePlane where

-- Define the axioms of projective geometry
axiom line_through_points : ProjectivePoint → ProjectivePoint → ProjectiveLine
axiom point_on_line : ProjectivePoint → ProjectiveLine → Prop
axiom distinct_points : ProjectivePoint → ProjectivePoint → Prop
axiom distinct_lines : ProjectiveLine → ProjectiveLine → Prop
axiom line_in_plane : ProjectiveLine → ProjectivePlane → Prop

-- Define central projection
def central_projection (π₁ π₂ : ProjectivePlane) : ProjectivePoint → ProjectivePoint :=
  sorry

-- Theorem statements
theorem unique_line_through_points (P Q : ProjectivePoint) (h : distinct_points P Q) :
  ∃! l : ProjectiveLine, point_on_line P l ∧ point_on_line Q l :=
sorry

theorem unique_intersection_of_lines (l m : ProjectiveLine) (π : ProjectivePlane)
  (hl : line_in_plane l π) (hm : line_in_plane m π) (h : distinct_lines l m) :
  ∃! P : ProjectivePoint, point_on_line P l ∧ point_on_line P m :=
sorry

theorem central_projection_bijective (π₁ π₂ : ProjectivePlane) :
  Function.Bijective (central_projection π₁ π₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_through_points_unique_intersection_of_lines_central_projection_bijective_l73_7319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l73_7313

noncomputable def points : List (ℝ × ℝ) := [(4, 15), (7, 25), (13, 42), (19, 57), (21, 65)]

noncomputable def isAboveLine (p : ℝ × ℝ) : Bool :=
  p.2 > 3 * p.1 + 5

noncomputable def sumXCoordinatesAboveLine (pts : List (ℝ × ℝ)) : ℝ :=
  (pts.filter isAboveLine).map (·.1) |>.sum

theorem sum_x_coordinates_above_line :
  sumXCoordinatesAboveLine points = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l73_7313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l73_7341

theorem expression_value (a b c d x : ℝ) : 
  (a + b = 0) →  -- a and b are additive inverses
  (c * d = 1) →  -- c and d are multiplicative inverses
  (abs x = 3) →  -- absolute value of x is 3
  (x^2 + (a + b - c*d)*x + Real.sqrt (a + b) + (c*d)^(1/3 : ℝ) = 7 ∨ 
   x^2 + (a + b - c*d)*x + Real.sqrt (a + b) + (c*d)^(1/3 : ℝ) = 13) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l73_7341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l73_7344

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x)^2 / x
noncomputable def g (x : ℝ) : ℝ := x / (Real.sqrt x)^2

-- State the theorem
theorem f_equals_g : ∀ x > 0, f x = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l73_7344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l73_7376

/-- A tetrahedron with two perpendicular skew edges -/
structure Tetrahedron where
  /-- Length of the first skew edge -/
  edge1 : ℝ
  /-- Length of the second skew edge -/
  edge2 : ℝ
  /-- Distance between the lines of the skew edges -/
  distance : ℝ
  /-- The skew edges are perpendicular -/
  perpendicular : True

/-- Calculate the volume of the tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ :=
  (1 / 6) * t.edge1 * t.edge2 * t.distance

/-- Theorem: The volume of the specific tetrahedron is 504 cubic units -/
theorem tetrahedron_volume :
  ∃ t : Tetrahedron, t.edge1 = 12 ∧ t.edge2 = 13 ∧ t.distance = 14 ∧ volume t = 504 := by
  sorry

#eval "Tetrahedron volume theorem stated successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l73_7376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_stability_l73_7314

/-- A type representing a sequence of integers with a fixed length -/
def IntSequence (n : ℕ) := Fin n → ℤ

/-- Function to count occurrences of an integer in a sequence -/
def countOccurrences {n : ℕ} (seq : IntSequence n) (x : ℤ) : ℕ :=
  (Finset.univ.filter (fun i => seq i = x)).card

/-- Function to perform one iteration of the frequency replacement -/
def frequencyIteration {n : ℕ} (seq : IntSequence n) : IntSequence n :=
  fun i => countOccurrences seq (seq i)

/-- Predicate to check if a sequence is stable under frequency iteration -/
def isStable {n : ℕ} (seq : IntSequence n) : Prop :=
  frequencyIteration seq = seq

theorem eventual_stability {n : ℕ} (seq : IntSequence n) :
  ∃ k : ℕ, isStable ((frequencyIteration^[k]) seq) := by
  sorry

#check eventual_stability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_stability_l73_7314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_perpendicular_point_l73_7301

-- Define the points in the coordinate system
def A : ℝ × ℝ := (2, 5)
def B : ℝ × ℝ := (3, 1)
def C (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the condition for three points to form a triangle
def forms_triangle (p q r : ℝ × ℝ) : Prop :=
  (q.1 - p.1) * (r.2 - p.2) ≠ (r.1 - p.1) * (q.2 - p.2)

-- Define the condition for two vectors to be perpendicular
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Theorem for the range of x
theorem triangle_condition (x : ℝ) :
  forms_triangle A B (C x) ↔ x ≠ 5/2 := by
  sorry

-- Theorem for the coordinates of point M
theorem perpendicular_point (M : ℝ × ℝ) :
  (∃ (t : ℝ), M = (6*t, 3*t)) ∧ 
  perpendicular (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2) ↔ 
  M = (2, 1) ∨ M = (22/5, 11/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_perpendicular_point_l73_7301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_z_value_l73_7347

theorem mean_equality_implies_z_value (z : ℝ) : 
  (8 + 10 + 22) / 3 = (16 + z) / 2 → z = 32 / 3 := by
  intro h
  -- The proof steps would go here, but we'll use sorry as requested
  sorry

#check mean_equality_implies_z_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_z_value_l73_7347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separate_l73_7398

noncomputable def circle1_center : ℝ × ℝ := (-1/2, -1)
noncomputable def circle2_center (θ : ℝ) : ℝ × ℝ := (Real.sin θ, 1)
noncomputable def circle1_radius : ℝ := Real.sqrt (1/2)
noncomputable def circle2_radius : ℝ := 1/4

noncomputable def distance_between_centers (θ : ℝ) : ℝ :=
  Real.sqrt ((Real.sin θ + 1/2)^2 + 4)

theorem circles_are_separate (θ : ℝ) (h : 0 < θ ∧ θ < π/2) :
  distance_between_centers θ > circle1_radius + circle2_radius := by
  sorry

#check circles_are_separate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separate_l73_7398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_l73_7315

/-- The amount of money borrowed and lent -/
def P : ℝ := 20000

/-- The interest rate at which money is borrowed (as a decimal) -/
def borrow_rate : ℝ := 0.08

/-- The interest rate at which money is lent (as a decimal) -/
def lend_rate : ℝ := 0.09

/-- The annual gain from the transaction -/
def annual_gain : ℝ := 200

/-- Theorem stating that if the annual gain is the difference between
    the interest earned and paid, then the borrowed amount is 20000 -/
theorem borrowed_amount : lend_rate * P - borrow_rate * P = annual_gain := by
  simp [P, lend_rate, borrow_rate, annual_gain]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_l73_7315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_with_separation_l73_7303

def number_of_people : ℕ := 6

theorem arrangements_with_separation (n : ℕ) (h : n = number_of_people) : 
  (Nat.factorial n) - (2 * (Nat.factorial (n - 1))) = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_with_separation_l73_7303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l73_7359

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x * (a * x + b) + x^2 + 2 * x

theorem function_properties (a b : ℝ) :
  f a b 0 = 1 ∧ 
  (deriv (f a b)) 0 = 4 →
  a = 1 ∧ b = 1 ∧
  ∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) (-1) → 
    f a b x ≥ x^2 + 2*(k+1)*x + k) →
  k ≥ 1/4 * Real.exp (-3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l73_7359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l73_7300

-- Define the sets A and B
def A : Set ℝ := {x | x + 2 ≥ 0}
def B : Set ℝ := {x | (x - 1) / (x + 1) ≥ 2}

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Theorem statement
theorem set_operations :
  (A ∩ B = Set.Icc (-2) (-1)) ∧
  (A ∪ B = Set.Ici (-3)) ∧
  ((U \ A) ∩ B = Set.Ico (-3) (-2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l73_7300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l73_7355

/-- The directrix equation of a parabola -/
def DirectrixEquation (y : ℝ → ℝ) : ℝ := 
  sorry -- Definition of directrix equation

/-- A parabola with equation y = x^2 has directrix y = -1/4 -/
theorem parabola_directrix : 
  ∃ (d : ℝ), d = -1/4 ∧ d = DirectrixEquation (fun x => x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l73_7355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_coloring_theorem_l73_7390

theorem house_coloring_theorem (n : ℕ) (π : Equiv.Perm (Fin n)) :
  ∃ (f : Fin n → Fin 3), ∀ i : Fin n, f i ≠ f (π i) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_coloring_theorem_l73_7390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_is_twelve_l73_7391

/-- Represents the average number of minutes run per day for each grade --/
structure GradeRunningAverage where
  sixth : ℚ
  seventh : ℚ
  eighth : ℚ

/-- Represents the ratio of students in each grade compared to seventh grade --/
structure GradeRatio where
  sixth : ℚ
  seventh : ℚ
  eighth : ℚ

/-- Calculates the average number of minutes run per day by all students --/
def averageMinutesRun (avg : GradeRunningAverage) (ratio : GradeRatio) : ℚ :=
  (avg.sixth * ratio.sixth + avg.seventh * ratio.seventh + avg.eighth * ratio.eighth) /
  (ratio.sixth + ratio.seventh + ratio.eighth)

/-- Theorem stating that the average number of minutes run per day is 12 --/
theorem average_minutes_run_is_twelve : 
  let avg : GradeRunningAverage := { sixth := 10, seventh := 18, eighth := 12 }
  let ratio : GradeRatio := { sixth := 3, seventh := 1, eighth := 1/2 }
  averageMinutesRun avg ratio = 12 := by
  -- Proof goes here
  sorry

#eval averageMinutesRun 
  { sixth := 10, seventh := 18, eighth := 12 }
  { sixth := 3, seventh := 1, eighth := 1/2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_is_twelve_l73_7391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_duration_l73_7356

-- Define the painting rates and lunch break duration
variable (s : ℝ)  -- Sandra's painting rate (house/hour)
variable (h : ℝ)  -- Combined rate of three helpers (house/hour)
variable (L : ℝ)  -- Lunch break duration (hours)

-- Define the conditions from the problem
def monday_condition (s h L : ℝ) : Prop := (8 - L) * (s + h) = 0.6
def tuesday_condition (h L : ℝ) : Prop := (6 - L) * h = 0.3
def wednesday_condition (s L : ℝ) : Prop := (2 - L) * s = 0.1

-- State the theorem to be proved
theorem lunch_break_duration :
  ∃ (s h L : ℝ), 
    monday_condition s h L ∧ 
    tuesday_condition h L ∧ 
    wednesday_condition s L ∧ 
    L = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_duration_l73_7356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_fixed_point_l73_7375

/-- Hyperbola with given eccentricity and distance from vertex to asymptote -/
def Hyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c/a = 5/3 ∧ a*b/c = 12/5

/-- Line intersecting hyperbola at two points with perpendicular vectors from vertex -/
def IntersectingLine (a b : ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, (x t)^2/a^2 - (y t)^2/b^2 = 1 ∧
  (x t - a) * (x (-t) - a) + (y t) * (y (-t)) = 0

/-- The main theorem -/
theorem hyperbola_fixed_point :
  ∀ a b : ℝ, Hyperbola a b →
  ∀ x y : ℝ → ℝ, IntersectingLine a b x y →
  ∃ t : ℝ, x t = -75/7 ∧ y t = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_fixed_point_l73_7375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circumcircle_area_l73_7328

/-- Given an isosceles triangle with two sides of length 5 and a base of length 4,
    the area of the circle passing through all three vertices is 15625π/1764 -/
theorem isosceles_triangle_circumcircle_area :
  ∀ (A B C : EuclideanSpace ℝ (Fin 2)),
    (dist A B = 5 ∧ dist A C = 5 ∧ dist B C = 4) →
    ∃ (O : EuclideanSpace ℝ (Fin 2)) (r : ℝ),
      (dist O A = r ∧ dist O B = r ∧ dist O C = r) →
      π * r^2 = 15625 / 1764 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circumcircle_area_l73_7328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_equation_l73_7311

noncomputable def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 3 = 0

noncomputable def point_M : ℝ × ℝ := (-Real.sqrt 3, 0)

noncomputable def rotate_line (θ : ℝ) (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  fun x y => sorry

theorem rotated_line_equation :
  ∀ (x y : ℝ),
    (rotate_line (π/6) point_M line_l x y ∨ rotate_line (-π/6) point_M line_l x y) ↔
    (x = -Real.sqrt 3 ∨ x - Real.sqrt 3 * y + Real.sqrt 3 = 0) := by
  sorry

#check rotated_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_line_equation_l73_7311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l73_7323

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points
variable (A A₁ B B₁ C C₁ P P₁ : V)

-- Define the conditions
variable (h1 : ¬ Submodule.span ℝ {A - C, B - C, A₁ - C} = ⊤)
variable (h2 : ∃ (k l : ℝ), B₁ - B = k • (A₁ - A) ∧ C₁ - C = l • (A₁ - A))
variable (h3 : P ∈ (affineSpan ℝ {A, B, C₁} : Set V) ∩ (affineSpan ℝ {A, B₁, C} : Set V) ∩ (affineSpan ℝ {A₁, B, C} : Set V))
variable (h4 : P₁ ∈ (affineSpan ℝ {A₁, B₁, C} : Set V) ∩ (affineSpan ℝ {A₁, B, C₁} : Set V) ∩ (affineSpan ℝ {A, B₁, C₁} : Set V))

-- State the theorem
theorem parallel_vectors :
  ∃ (t : ℝ), P₁ - P = t • (A₁ - A) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l73_7323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_weight_loss_percentage_l73_7354

/-- Calculates the percentage of weight lost during processing for a side of beef. -/
noncomputable def weightLossPercentage (weightBefore : ℝ) (weightAfter : ℝ) : ℝ :=
  ((weightBefore - weightAfter) / weightBefore) * 100

/-- Theorem stating that a side of beef weighing 400 pounds before processing
    and 240 pounds after processing loses 40% of its weight. -/
theorem beef_weight_loss_percentage :
  weightLossPercentage 400 240 = 40 := by
  -- Unfold the definition of weightLossPercentage
  unfold weightLossPercentage
  -- Simplify the arithmetic expression
  simp [div_eq_mul_inv]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_weight_loss_percentage_l73_7354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_fourth_term_l73_7326

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Adding the case for 0 to avoid missing cases error
  | 1 => 3
  | (n + 2) => 3 * sequence_a (n + 1) / (sequence_a (n + 1) + 3)

theorem a_fourth_term : sequence_a 4 = 3/4 := by
  -- The proof will be implemented here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_fourth_term_l73_7326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quartic_at_6_l73_7329

/-- A quartic polynomial satisfying specific conditions -/
def is_special_quartic (p : ℝ → ℝ) : Prop :=
  (∃ a b c d e : ℝ, ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e) ∧
  (∀ n : ℕ, n ∈ ({1, 2, 3, 4, 5} : Set ℕ) → p (n : ℝ) = 1 / (n^2 : ℝ))

/-- The value of the special quartic polynomial at x = 6 -/
theorem special_quartic_at_6 (p : ℝ → ℝ) (h : is_special_quartic p) : p 6 = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quartic_at_6_l73_7329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_gcd_triangular_l73_7333

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The greatest possible value of gcd(6Tₙ, n+1) -/
theorem greatest_gcd_triangular : 
  (Finset.sup (Finset.range 1000) (λ k ↦ Nat.gcd (6 * triangular_number k) (k + 1))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_gcd_triangular_l73_7333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_reciprocal_sum_constant_l73_7351

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  h : a > 0

/-- Chord passing through the focus of a parabola -/
structure Chord (p : Parabola) where
  k : ℝ  -- slope of the chord

/-- Point on the parabola -/
structure Point (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.a * x^2

/-- Distance from a point to the focus -/
noncomputable def distance_to_focus (p : Parabola) (pt : Point p) : ℝ :=
  pt.y + 1 / (4 * p.a)

/-- Theorem: Sum of reciprocals of distances from chord endpoints to focus is constant -/
theorem chord_reciprocal_sum_constant (p : Parabola) (c : Chord p) :
  ∃ (A B : Point p), 
    (1 / distance_to_focus p A + 1 / distance_to_focus p B) = 1 := by
  sorry

#check chord_reciprocal_sum_constant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_reciprocal_sum_constant_l73_7351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_plus_pi_over_3_l73_7371

theorem sin_2theta_plus_pi_over_3 (θ : Real) :
  (∃ (x y : Real), x > 0 ∧ y = 3 * x ∧ (Real.cos θ = x / Real.sqrt (x^2 + y^2)) ∧ (Real.sin θ = y / Real.sqrt (x^2 + y^2))) →
  Real.sin (2 * θ + π / 3) = (3 - 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_plus_pi_over_3_l73_7371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_grid_size_l73_7399

-- Define m as a parameter
variable (m : Nat)

def Grid (m : Nat) := Fin m → Fin 8 → Fin 4

def valid_grid {m : Nat} (g : Grid m) : Prop :=
  ∀ i j : Fin m, i ≠ j →
    (∃! k : Fin 8, g i k = g j k) ∨ (∀ k : Fin 8, g i k ≠ g j k)

theorem largest_valid_grid_size :
  (∃ g : Grid 5, valid_grid g) ∧
  (∀ m' > 5, ¬∃ g : Grid m', valid_grid g) ↔
  5 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_grid_size_l73_7399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coloring_theorem_l73_7327

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a coloring of regions in the plane --/
def Coloring := (ℝ × ℝ) → Bool

/-- Checks if two circles are non-tangent --/
def nonTangent (c1 c2 : Circle) : Prop := sorry

/-- Checks if a coloring is valid for a given set of circles --/
def validColoring (circles : List Circle) (coloring : Coloring) : Prop := sorry

/-- Main theorem: For any list of non-tangent circles, there exists a valid coloring --/
theorem circle_coloring_theorem (circles : List Circle) 
  (h : ∀ (c1 c2 : Circle), c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → nonTangent c1 c2) : 
  ∃ coloring : Coloring, validColoring circles coloring := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coloring_theorem_l73_7327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_9A_l73_7377

def is_strictly_increasing (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  ∀ i j, i < j → i < digits.length → j < digits.length → digits[i]! < digits[j]!

theorem sum_of_digits_of_9A (A : ℕ) (h : is_strictly_increasing A) :
  (Nat.digits 10 (9 * A)).sum = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_9A_l73_7377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_owed_proof_l73_7389

noncomputable def calculate_amount_owed (original_price deposit_percent processing_fee_percent discount_percent : ℝ) : ℝ :=
  let deposit := original_price * (deposit_percent / 100)
  let remaining_balance := original_price - deposit
  let discounted_amount := remaining_balance * (1 - discount_percent / 100)
  let processing_fee := discounted_amount * (processing_fee_percent / 100)
  discounted_amount + processing_fee

theorem total_amount_owed_proof :
  let product_a := calculate_amount_owed 1000 5 3 10
  let product_b := calculate_amount_owed 1000 7 4 0
  let product_c := calculate_amount_owed 1000 10 2 5
  (product_a + product_b + product_c) = 2720.95 := by
  sorry

-- Remove #eval statements as they are not computable
-- #eval calculate_amount_owed 1000 5 3 10 -- Product A
-- #eval calculate_amount_owed 1000 7 4 0  -- Product B
-- #eval calculate_amount_owed 1000 10 2 5 -- Product C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_owed_proof_l73_7389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l73_7384

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) / (x - 2)

-- Define the domain
def domain : Set ℝ := {x : ℝ | x ≥ -2 ∧ x ≠ 2}

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, x ∈ domain ↔ (∃ y : ℝ, f x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l73_7384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_minimum_point_at_two_l73_7317

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / x + log x

-- State the theorem
theorem is_minimum_point_at_two :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x > 0 ∧ x ≠ 2 ∧ |x - 2| < ε → f x > f 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_minimum_point_at_two_l73_7317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_is_2_a_range_when_f_decreasing_l73_7308

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) / (x + 1)

-- Part 1
theorem f_increasing_when_a_is_2 :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < -1 → f 2 x₁ < f 2 x₂ := by
  sorry

-- Part 2
theorem a_range_when_f_decreasing (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < -1 → f a x₁ > f a x₂) → a < -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_is_2_a_range_when_f_decreasing_l73_7308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l73_7366

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*y = 0

-- Define the center and radius of circle1
def center1 : ℝ × ℝ := (2, 0)
def radius1 : ℝ := 2

-- Define the center and radius of circle2
def center2 : ℝ × ℝ := (0, -1)
def radius2 : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 5

-- Theorem: The circles are intersecting
theorem circles_intersect : 
  radius1 - radius2 < distance_between_centers ∧ 
  distance_between_centers < radius1 + radius2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l73_7366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_range_l73_7382

theorem quadratic_inequality_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + m*x + 2 > 0) → 
  m ∈ Set.Ioi (-3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_range_l73_7382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_l73_7310

/-- Represents the position of the lemming -/
structure Position where
  x : ℝ
  y : ℝ

/-- Calculates the average distance to the sides of the rectangle -/
noncomputable def averageDistanceToSides (pos : Position) (width height : ℝ) : ℝ :=
  (pos.x + pos.y + (width - pos.x) + (height - pos.y)) / 4

/-- Theorem stating the average distance to sides after lemming's movement -/
theorem lemming_average_distance 
  (rect_width : ℝ) 
  (rect_height : ℝ) 
  (diagonal_move : ℝ) 
  (vertical_move : ℝ) :
  rect_width = 15 →
  rect_height = 8 →
  diagonal_move = 11.3 →
  vertical_move = 2.6824 →
  let diagonal_length := Real.sqrt (rect_width^2 + rect_height^2)
  let fraction_moved := diagonal_move / diagonal_length
  let final_pos : Position := {
    x := fraction_moved * rect_width,
    y := fraction_moved * rect_height + vertical_move
  }
  averageDistanceToSides final_pos rect_width rect_height = 5.75 := by
  sorry

#eval "Lemming problem theorem defined successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_l73_7310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_ratio_l73_7343

/-- Represents a cone with an inscribed sphere and a surrounding cylinder -/
structure ConeSphereCylinder where
  cone_radius : ℝ
  cone_height : ℝ
  sphere_radius : ℝ

/-- The volume of the cone -/
noncomputable def cone_volume (c : ConeSphereCylinder) : ℝ :=
  (1/3) * Real.pi * c.cone_radius^2 * c.cone_height

/-- The volume of the cylinder -/
noncomputable def cylinder_volume (c : ConeSphereCylinder) : ℝ :=
  2 * Real.pi * c.sphere_radius^3

/-- The ratio of cone volume to cylinder volume -/
noncomputable def volume_ratio (c : ConeSphereCylinder) : ℝ :=
  cone_volume c / cylinder_volume c

theorem cone_cylinder_volume_ratio 
  (c : ConeSphereCylinder) 
  (h1 : c.cone_radius > 0)
  (h2 : c.cone_height > 0)
  (h3 : c.sphere_radius > 0)
  (h4 : c.sphere_radius = (c.cone_radius * c.cone_height) / (c.cone_radius + Real.sqrt (c.cone_radius^2 + c.cone_height^2)))
  (h5 : 2 * c.sphere_radius ≤ c.cone_height) :
  volume_ratio c > 1 ∧ volume_ratio c ≥ 4/3 := by
  sorry

#check cone_cylinder_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_ratio_l73_7343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_and_side_length_l73_7394

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

theorem triangle_arithmetic_sequence_and_side_length 
  (t : Triangle) 
  (h1 : t.b * (Real.cos (t.A / 2))^2 + t.a * (Real.cos (t.B / 2))^2 = 3/2 * t.c) 
  (h2 : t.C = π/3)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3) : 
  (t.a + t.b = 2 * t.c) ∧ (t.c = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_and_side_length_l73_7394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_multiple_of_630_finite_count_factors_multiple_of_630_l73_7339

def m : ℕ := 2^12 * 3^10 * 5^9 * 7^6

def is_multiple_of_630 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 630 * k

def factors_multiple_of_630 (n : ℕ) : Set ℕ :=
  {d : ℕ | d ∣ n ∧ is_multiple_of_630 d}

-- We need to prove that the set is finite before we can count its elements
theorem factors_multiple_of_630_finite (n : ℕ) :
  (factors_multiple_of_630 n).Finite :=
sorry

theorem count_factors_multiple_of_630 :
  Finset.card (Set.Finite.toFinset (factors_multiple_of_630_finite m)) = 5832 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_multiple_of_630_finite_count_factors_multiple_of_630_l73_7339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_l73_7348

-- Define the two fixed circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define a moving circle externally tangent to both fixed circles
def movingCircle (h k r : ℝ) : Prop :=
  ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2

-- Define the condition for external tangency
def externallyTangent (h k r : ℝ) : Prop :=
  (∃ (x1 y1 : ℝ), circle1 x1 y1 ∧ (x1 - h)^2 + (y1 - k)^2 = (r + 1)^2) ∧
  (∃ (x2 y2 : ℝ), circle2 x2 y2 ∧ (x2 - h)^2 + (y2 - k)^2 = (r + 2)^2)

-- Define a hyperbola
def isHyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
    ∀ (x y : ℝ), f x y ↔ (x - c)^2 / a^2 - (y - d)^2 / b^2 = 1

-- Theorem statement
theorem locus_is_hyperbola :
  isHyperbola (λ h k ↦ ∃ r, movingCircle h k r ∧ externallyTangent h k r) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_l73_7348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_average_age_l73_7332

/-- Calculates the average age of a population given the ratio of women to men and their respective average ages. -/
noncomputable def averageAgeOfPopulation (womenRatio menRatio : ℕ) (womenAvgAge menAvgAge : ℝ) : ℝ :=
  let totalWomen := womenRatio
  let totalMen := menRatio
  let totalPopulation := totalWomen + totalMen
  let totalWomenAge := womenAvgAge * (totalWomen : ℝ)
  let totalMenAge := menAvgAge * (totalMen : ℝ)
  let totalAge := totalWomenAge + totalMenAge
  totalAge / (totalPopulation : ℝ)

/-- Theorem stating that the average age of the population is 27.25 years
    given the specified conditions. -/
theorem population_average_age :
  averageAgeOfPopulation 5 3 25 31 = 27.25 := by
  -- Unfold the definition of averageAgeOfPopulation
  unfold averageAgeOfPopulation
  -- Simplify the arithmetic expressions
  simp [Nat.cast_add, Nat.cast_mul]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_average_age_l73_7332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l73_7396

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

-- State the theorem
theorem f_inequality_range : 
  ∀ x : ℝ, f x > f (2*x - 1) ↔ 1/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l73_7396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l73_7373

noncomputable def f (x : ℝ) : ℝ := -5 * Real.cos (x + Real.pi/4)

theorem amplitude_and_phase_shift :
  (∃ (A : ℝ), A > 0 ∧ ∀ x, |f x| ≤ A ∧ (∃ x₀, |f x₀| = A)) ∧
  (∃ (φ : ℝ), ∀ x, f x = -5 * Real.cos (x - φ)) :=
by
  sorry

#check amplitude_and_phase_shift

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l73_7373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l73_7380

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := Real.sin (16 * Real.pi / 180) + Real.cos (16 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem relationship_abc : a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l73_7380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_water_volume_l73_7320

-- Define the diameters of the pipes
noncomputable def large_diameter : ℝ := 12
noncomputable def small_diameter : ℝ := 3

-- Define the ratio of cross-sectional areas
noncomputable def area_ratio : ℝ := (large_diameter / small_diameter) ^ 2

-- Theorem statement
theorem equivalent_water_volume : area_ratio = 16 := by
  -- Unfold the definitions
  unfold area_ratio large_diameter small_diameter
  -- Simplify the expression
  simp
  -- The proof is completed
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_water_volume_l73_7320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_multiplication_puzzle_l73_7331

theorem unique_multiplication_puzzle : ∃! (a b c d e f g h i : ℕ),
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10) ∧
  (Finset.card (Finset.range 10) = 10) ∧
  (a * (10 * b + c) * (100 * d + 10 * e + f) = 1000 * g + 100 * h + 10 * i + 1) ∧
  a = 1 ∧ b = 2 ∧ c = 6 ∧ d = 3 ∧ e = 4 ∧ f = 5 ∧ g = 8 ∧ h = 9 ∧ i = 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_multiplication_puzzle_l73_7331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_billing_problem_l73_7363

-- Define the billing function for Option 1
noncomputable def L (x : ℝ) : ℝ :=
  if x ≤ 30 then 2 + 0.5 * x else 0.6 * x - 1

-- Define the billing function for Option 2
noncomputable def F (x : ℝ) : ℝ := 0.58 * x

theorem electricity_billing_problem :
  -- 1. Prove the billing function L(x) is correct
  (∀ x : ℝ, x ≤ 30 → L x = 2 + 0.5 * x) ∧
  (∀ x : ℝ, x > 30 → L x = 0.6 * x - 1) ∧
  -- 2. Prove that for a $35 bill under Option 1, the usage is 60 kWh
  L 60 = 35 ∧
  -- 3. Prove Option 1 is more economical than Option 2 when 25 < x < 50
  (∀ x : ℝ, 25 < x ∧ x < 50 → L x < F x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_billing_problem_l73_7363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_rotation_l73_7381

-- Define the original curve
def original_curve (x y : ℝ) : Prop := x + y^2 = 1

-- Define the rotation angle
noncomputable def rotation_angle : ℝ := Real.pi / 4  -- 45° in radians

-- Define the rotated coordinates
noncomputable def rotated_x (x' y' : ℝ) : ℝ := (x' + y') / Real.sqrt 2
noncomputable def rotated_y (x' y' : ℝ) : ℝ := (y' - x') / Real.sqrt 2

-- Define the rotated curve
def rotated_curve (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x*y + Real.sqrt 2 * x + Real.sqrt 2 * y - 2 = 0

-- Theorem statement
theorem curve_rotation :
  ∀ x y x' y' : ℝ,
  original_curve (rotated_x x' y') (rotated_y x' y') →
  rotated_curve x' y' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_rotation_l73_7381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_property_l73_7342

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- The focus of a parabola -/
noncomputable def focus (para : Parabola) : Point :=
  { x := para.p / 2, y := 0 }

/-- A point on the parabola -/
noncomputable def pointOnParabola (para : Parabola) (x₀ : ℝ) : Point :=
  { x := x₀, y := 2 * Real.sqrt 2 }

theorem parabola_point_property (para : Parabola) (x₀ : ℝ) 
    (h₁ : x₀ > para.p / 2)
    (h₂ : (pointOnParabola para x₀).y ^ 2 = 2 * para.p * x₀) :
  let M := pointOnParabola para x₀
  let F := focus para
  ∃ (A : Point),
    -- Circle M intersects MF at point A
    -- Chord length intercepted by x = p/2 is √3|MA|
    -- |MA|/|AF| = 2
    -- These conditions are assumed implicitly
    norm (A.x - F.x, A.y - F.y) = 1 := by
  sorry

-- Define a subtraction operation for Points
instance : HSub Point Point Point where
  hSub a b := { x := a.x - b.x, y := a.y - b.y }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_property_l73_7342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_in_square_configuration_l73_7383

/-- The energy increase when moving one charge from a vertex to the center of a square configuration of four identical point charges. -/
noncomputable def energy_increase (initial_energy : ℝ) : ℝ :=
  15 * Real.sqrt 2 - 10

/-- Theorem stating the energy increase when moving one charge in a square configuration. -/
theorem energy_increase_in_square_configuration (initial_energy : ℝ) :
  initial_energy = 20 →
  energy_increase initial_energy = 15 * Real.sqrt 2 - 10 :=
by
  intro h
  unfold energy_increase
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_in_square_configuration_l73_7383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_sqrt_l73_7385

theorem cosine_product_sqrt (a b c : Real) : 
  (a^3 - 18*a^2 + 81*a - 54 = 0) →
  (b^3 - 18*b^2 + 81*b - 54 = 0) →
  (c^3 - 18*c^2 + 81*c - 54 = 0) →
  a ≠ b → b ≠ c → a ≠ c →
  Real.sqrt ((3 - a) * (3 - b) * (3 - c)) = 0 := by
  sorry

#check cosine_product_sqrt

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_sqrt_l73_7385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l73_7358

/-- Given two workers A and B who can complete a task individually in 8 and 16 days respectively,
    this theorem proves that they can complete the task together in 16/3 days. -/
theorem work_completion_time (work : ℝ) (time_A time_B : ℝ) 
  (h_work : work > 0)
  (h_time_A : time_A = 8)
  (h_time_B : time_B = 16)
  (h_A : work / time_A = work * (1 / time_A))
  (h_B : work / time_B = work * (1 / time_B)) :
  work / (work / time_A + work / time_B) = 16 / 3 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l73_7358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_relation_l73_7338

theorem triangle_tangent_relation (A B C : Real) (hABC : A + B + C = Real.pi) 
  (h_sin : Real.sin A ^ 2 + Real.sin C ^ 2 = 2018 * Real.sin B ^ 2) :
  (Real.tan A + Real.tan C) * Real.tan B ^ 2 / (Real.tan A + Real.tan B + Real.tan C) = 2/2017 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_relation_l73_7338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l73_7379

/-- The distance from a point (x₀, y₀) to the line Ax + By + C = 0 -/
noncomputable def distancePointToLine (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The radius of the circle x^2 + y^2 = 5 -/
noncomputable def circleRadius : ℝ := Real.sqrt 5

theorem line_tangent_to_circle :
  let d := distancePointToLine 0 0 2 (-1) (-5)
  d = circleRadius := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l73_7379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_height_from_AC_l73_7324

noncomputable section

open Real

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of triangle ABC -/
def area (t : Triangle) : ℝ := (1/2) * t.a * t.c * sin t.B

/-- The height from side AC in triangle ABC -/
def height_from_AC (t : Triangle) : ℝ := 2 * (area t) / t.c

theorem max_height_from_AC (t : Triangle) 
  (h1 : t.b = 2 * sqrt 3)
  (h2 : sqrt 3 * sin t.C = (sin t.A + sqrt 3 * cos t.A) * sin t.B) :
  ∃ (h : ℝ), h = height_from_AC t ∧ h ≤ 3 ∧ ∀ (h' : ℝ), h' = height_from_AC t → h' ≤ h :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_height_from_AC_l73_7324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hike_leak_rate_l73_7337

/-- Represents the water consumption and leak during a hike --/
structure HikeWater where
  totalDistance : ℝ
  initialWater : ℝ
  duration : ℝ
  remainingWater : ℝ
  waterPerMileFirst3 : ℝ
  waterLastMile : ℝ

/-- Calculates the rate at which the canteen leaked during the hike --/
noncomputable def leakRate (h : HikeWater) : ℝ :=
  let totalWaterLost := h.initialWater - h.remainingWater
  let waterDrunk := h.waterPerMileFirst3 * 3 + h.waterLastMile
  let waterLeaked := totalWaterLost - waterDrunk
  waterLeaked / h.duration

/-- Theorem stating that for the given hike conditions, the leak rate is 1 cup per hour --/
theorem hike_leak_rate (h : HikeWater) 
  (h_totalDistance : h.totalDistance = 4)
  (h_initialWater : h.initialWater = 10)
  (h_duration : h.duration = 2)
  (h_remainingWater : h.remainingWater = 2)
  (h_waterPerMileFirst3 : h.waterPerMileFirst3 = 1)
  (h_waterLastMile : h.waterLastMile = 3) :
  leakRate h = 1 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hike_leak_rate_l73_7337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_is_negative_one_zero_one_l73_7362

-- Define the Gauss function
noncomputable def gauss (x : ℝ) : ℤ := ⌊x⌋

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 - 3*x + 4

-- Define the set of real numbers between 1 and 4
def open_interval : Set ℝ := {x | 1 < x ∧ x < 4}

-- Define the range of y = [f(x)]
def range_y : Set ℤ := {y | ∃ x ∈ open_interval, y = gauss (f x)}

-- Theorem statement
theorem range_of_y_is_negative_one_zero_one : 
  range_y = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_is_negative_one_zero_one_l73_7362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l73_7336

/-- Represents a statistical proposition -/
inductive Proposition
| RandomError
| ResidualSumSquares
| CorrelationCoefficient
| LeastSquaresLine

/-- Determines if a proposition is true -/
def is_true (p : Proposition) : Bool :=
  match p with
  | Proposition.RandomError => true
  | Proposition.ResidualSumSquares => true
  | Proposition.CorrelationCoefficient => false
  | Proposition.LeastSquaresLine => true

/-- The list of all propositions -/
def all_propositions : List Proposition :=
  [Proposition.RandomError, Proposition.ResidualSumSquares,
   Proposition.CorrelationCoefficient, Proposition.LeastSquaresLine]

/-- Counts the number of true propositions -/
def count_true_propositions (props : List Proposition) : Nat :=
  (props.filter is_true).length

/-- The main theorem stating that the number of true propositions is 3 -/
theorem three_true_propositions :
  count_true_propositions all_propositions = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l73_7336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_routes_count_l73_7335

/-- Represents a city in the road network -/
inductive City : Type
| A | B | C | D | E | F
deriving Repr, DecidableEq

/-- Represents a road between two cities -/
inductive Road : Type
| AB | AD | AE | BC | BD | CD | DE | DF | EF
deriving Repr, DecidableEq

/-- A route is a list of roads -/
def Route := List Road

/-- Function to check if a route is valid -/
def isValidRoute (r : Route) : Bool :=
  r.length = 9 && r.toFinset.card = 9 && r.head? = some Road.AB

/-- Function to count the number of valid routes -/
def countValidRoutes : Nat :=
  (List.filter isValidRoute (List.permutations [Road.AB, Road.AD, Road.AE, Road.BC, Road.BD, Road.CD, Road.DE, Road.DF, Road.EF])).length

/-- The main theorem stating that there are exactly 30 valid routes -/
theorem valid_routes_count : countValidRoutes = 30 := by
  sorry

#eval countValidRoutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_routes_count_l73_7335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_with_log_constraint_l73_7304

theorem min_sum_with_log_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_log : Real.log a / Real.log 2 + Real.log b / Real.log 2 = 3) : 
  a + b ≥ 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_with_log_constraint_l73_7304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_bounds_l73_7307

-- Define the function f(x) = 3^|x|
noncomputable def f (x : ℝ) : ℝ := 3^(abs x)

-- Define the theorem
theorem interval_length_bounds (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc 1 9) →
  (∃ x ∈ Set.Icc a b, f x = 1) →
  (∃ x ∈ Set.Icc a b, f x = 9) →
  b - a ≤ 4 ∧ b - a ≥ 2 := by
  sorry

#check interval_length_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_bounds_l73_7307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_approx_l73_7316

/-- The radius such that the area of a circle with this radius is 1/3 of a unit square's area --/
noncomputable def circle_radius : ℝ :=
  1 / Real.sqrt (3 * Real.pi)

/-- Theorem stating that the circle_radius is approximately 0.3 --/
theorem circle_radius_approx : ∃ ε > 0, |circle_radius - 0.3| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_approx_l73_7316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_journey_l73_7387

/-- Represents the time on an analog clock --/
structure ClockTime where
  hours : ℕ
  minutes : ℕ
  valid : hours < 12 ∧ minutes < 60

/-- Calculates the angle of the hour hand from 12 o'clock position --/
noncomputable def hourHandAngle (t : ClockTime) : ℝ :=
  (t.hours % 12 : ℝ) * 30 + (t.minutes : ℝ) * 0.5

/-- Calculates the angle of the minute hand from 12 o'clock position --/
noncomputable def minuteHandAngle (t : ClockTime) : ℝ :=
  (t.minutes : ℝ) * 6

/-- Checks if the clock hands are symmetrical around a given hour --/
noncomputable def isSymmetricalAround (t : ClockTime) (hour : ℕ) : Prop :=
  let hourAngle := (hour % 12 : ℝ) * 30
  let angleDiff := (minuteHandAngle t - hourHandAngle t + 360) % 360
  angleDiff = (2 * (hourAngle - hourHandAngle t) + 360) % 360

/-- The theorem representing Xiaoming's journey --/
theorem xiaoming_journey 
  (startTime endTime : ClockTime)
  (speed : ℝ)
  (h1 : startTime.hours = 7 ∧ startTime.minutes = 0)
  (h2 : speed = 52)
  (h3 : isSymmetricalAround endTime 7)
  (h4 : endTime.hours - startTime.hours < 1 ∨ 
        (endTime.hours - startTime.hours = 1 ∧ endTime.minutes < startTime.minutes)) :
  ((endTime.hours - startTime.hours) * 60 + (endTime.minutes - startTime.minutes)) * speed = 1680 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_journey_l73_7387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_through_origin_with_intercepts_l73_7361

theorem circle_equation_through_origin_with_intercepts 
  (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  let circle := fun x y ↦ (x - p/2)^2 + (y - q/2)^2 = (p^2 + q^2)/4
  (circle 0 0) ∧ (circle p 0) ∧ (circle 0 q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_through_origin_with_intercepts_l73_7361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l73_7364

/-- Given two 2D vectors a and b, prove that if k*a + b is parallel to a + k*b,
    then k equals either -1 or -7. -/
theorem parallel_vectors_lambda (a b : ℝ × ℝ) (h1 : a = (3, 2)) (h2 : b = (2, -1)) :
  ∃ k : ℝ, (∃ c : ℝ, c • (k • a + b) = a + k • b) → (k = -1 ∨ k = -7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l73_7364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_around_earth_person_can_pass_l73_7345

/-- The radius of the Earth in centimeters -/
noncomputable def R : ℝ := 6e8

/-- The length by which the rope is extended in centimeters -/
def rope_extension : ℝ := 1

/-- The height of the gap between the extended rope and the Earth's surface -/
noncomputable def gap_height (R : ℝ) : ℝ := Real.sqrt (R / 12)

theorem rope_around_earth (R : ℝ) (h : R > 0) :
  gap_height R = Real.sqrt (R / 12) := by
  -- The proof is omitted
  sorry

/-- A person can pass through the gap if it's greater than 2 meters -/
def can_pass_through (height : ℝ) : Prop := height > 200

theorem person_can_pass :
  can_pass_through (gap_height R) := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_around_earth_person_can_pass_l73_7345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l73_7322

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := |1/2 * x| - |1/2 * x - m|

-- State the theorem
theorem function_properties :
  -- Given conditions
  ∃ (m : ℝ), (∀ x, f x m ≤ 4) ∧ (∃ x, f x m = 4) ∧
  -- Conclusions
  (m = 4) ∧
  (∀ x, 0 < x → x < m/2 → (2/|x| + 2/|x-2| ≥ 4)) ∧
  (∃ x, 0 < x ∧ x < m/2 ∧ 2/|x| + 2/|x-2| = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l73_7322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_monitor_selection_l73_7388

theorem class_monitor_selection (n_boys n_girls : ℕ) (h_boys : n_boys = 4) (h_girls : n_girls = 3) :
  (Nat.choose n_boys 1 * Nat.choose n_girls 1 + Nat.choose n_boys 2) * (2 * 1) =
  (Nat.choose (n_boys + n_girls) 2 - Nat.choose n_girls 2) * (2 * 1) ∧
  (Nat.choose n_boys 1 * Nat.choose n_girls 1 + Nat.choose n_boys 2) * (2 * 1) =
  (n_boys + n_girls) * (n_boys + n_girls - 1) - n_girls * (n_girls - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_monitor_selection_l73_7388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_QR_is_sqrt_6_l73_7305

noncomputable section

def cube_edge_length : ℝ := 2

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def Q : Point3D := { x := 1, y := 1, z := 0 }
def R : Point3D := { x := 0, y := 0, z := 2 }

noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

theorem length_QR_is_sqrt_6 : distance Q R = Real.sqrt 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_QR_is_sqrt_6_l73_7305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l73_7312

theorem quadratic_function_properties (a b c : ℤ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)
  (∀ x, (deriv f) x = 0 ↔ x = 1) →  -- 1 is the point of extremum
  f 1 = 3 →                        -- 3 is the extremum value
  f 2 = 8 →                        -- (2, 8) lies on the curve
  (a = 5 ∧ b = -10 ∧ c = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l73_7312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_proof_l73_7374

def total_balls : ℕ := 8
def yellow_balls : ℕ := 4
def red_balls : ℕ := 4
def drawn_balls : ℕ := 3

def winning_amount (red_count : ℕ) : ℚ :=
  match red_count with
  | 1 => 50
  | 2 => 100
  | 3 => 200
  | _ => 0

def prob_one_yellow : ℚ := 3 / 7

def distribution : List (ℚ × ℚ) := [
  (0, 1/14),
  (50, 3/7),
  (100, 3/7),
  (200, 1/14)
]

theorem lottery_proof :
  (prob_one_yellow = 3 / 7) ∧
  (List.sum (List.map (λ x => x.1 * x.2) distribution) = 550 / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_proof_l73_7374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_estimate_is_total_frequency_l73_7392

structure ShootingSet where
  attempts : ℕ
  hits : ℕ
  frequency : ℚ
  hits_le_attempts : hits ≤ attempts

def total_attempts (sets : List ShootingSet) : ℕ :=
  (sets.map (·.attempts)).sum

def total_hits (sets : List ShootingSet) : ℕ :=
  (sets.map (·.hits)).sum

def total_frequency (sets : List ShootingSet) : ℚ :=
  (total_hits sets : ℚ) / (total_attempts sets : ℚ)

theorem best_estimate_is_total_frequency 
  (set1 set2 set3 : ShootingSet)
  (h1 : set1.attempts = 100 ∧ set1.hits = 68)
  (h2 : set2.attempts = 200 ∧ set2.hits = 124)
  (h3 : set3.attempts = 300 ∧ set3.hits = 174)
  (sets : List ShootingSet := [set1, set2, set3]) :
  total_frequency sets = 366 / 600 ∧ 
  (∀ s ∈ sets, (total_frequency sets : ℝ) ≤ (s.frequency : ℝ) → 
    total_attempts sets ≤ s.attempts) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_estimate_is_total_frequency_l73_7392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employees_without_benefits_l73_7386

theorem employees_without_benefits (total_employees : ℕ) 
  (salary_increase_percent : ℚ) 
  (travel_allowance_percent : ℚ) 
  (both_increases_percent : ℚ) 
  (vacation_days_percent : ℚ) 
  (h1 : total_employees = 480) 
  (h2 : salary_increase_percent = 1/10) 
  (h3 : travel_allowance_percent = 1/5) 
  (h4 : both_increases_percent = 1/20) 
  (h5 : vacation_days_percent = 3/20) :
  total_employees - (↑total_employees * (salary_increase_percent + travel_allowance_percent + vacation_days_percent - both_increases_percent)).floor = 288 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_employees_without_benefits_l73_7386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_interval_l73_7369

-- Define the function f(x) = (1/3)^x
noncomputable def f (x : ℝ) : ℝ := (1/3) ^ x

-- State the theorem
theorem min_value_on_interval :
  ∃ (min_val : ℝ), min_val = 1 ∧
  ∀ x ∈ Set.Icc (-1 : ℝ) 0, f x ≥ min_val := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_interval_l73_7369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_nonempty_domain_domain_of_g5_l73_7330

-- Define the sequence of functions
noncomputable def g : ℕ → (ℝ → ℝ)
  | 0 => λ x => Real.sqrt (4 - x)  -- Added case for 0
  | 1 => λ x => Real.sqrt (4 - x)
  | (n + 2) => λ x => g (n + 1) (Real.sqrt ((n + 3)^2 - x))

-- Define the domain of a function
def domain (f : ℝ → ℝ) := {x : ℝ | ∃ y, f x = y}

-- Define a predicate for non-empty domain
def has_nonempty_domain (f : ℝ → ℝ) := (domain f).Nonempty

-- Statement of the theorem
theorem largest_n_with_nonempty_domain :
  (∃ M : ℕ, M = 5 ∧
    has_nonempty_domain (g M) ∧
    ∀ n > M, ¬ has_nonempty_domain (g n)) ∧
  domain (g 5) = {-589} := by
  sorry

-- Additional theorem to clarify the domain of g₅
theorem domain_of_g5 :
  domain (g 5) = {-589} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_nonempty_domain_domain_of_g5_l73_7330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_1804_to_hundredth_l73_7357

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The statement that 1.804 rounded to the nearest hundredth equals 1.80 -/
theorem round_1804_to_hundredth :
  roundToHundredth 1.804 = 1.80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_1804_to_hundredth_l73_7357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_median_and_bisector_l73_7350

/-- The angle between the median and the bisector in a right-angled triangle
    with acute angle α --/
noncomputable def angle_between_median_and_bisector_in_right_triangle (α : Real) : Real :=
  Real.arctan (Real.tan α / 2) - α / 2

/-- In a right-angled triangle with an acute angle α, the angle between the median
    and the bisector drawn from the vertex of this acute angle is
    arctan(tan(α)/2) - α/2 --/
theorem angle_between_median_and_bisector (α : Real) 
  (h₁ : 0 < α) (h₂ : α < π / 2) : 
  ∃ (θ : Real), θ = Real.arctan (Real.tan α / 2) - α / 2 ∧
  θ = angle_between_median_and_bisector_in_right_triangle α :=
by
  -- Define θ as the angle between the median and bisector
  let θ := angle_between_median_and_bisector_in_right_triangle α
  
  -- Show that θ satisfies both conditions
  have h₃ : θ = Real.arctan (Real.tan α / 2) - α / 2 := rfl
  have h₄ : θ = angle_between_median_and_bisector_in_right_triangle α := rfl
  
  -- Prove the existence of such θ
  exact ⟨θ, ⟨h₃, h₄⟩⟩
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_median_and_bisector_l73_7350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l73_7346

-- Define the foci
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Define the distance difference
def distance_diff : ℝ := 6

-- Define the hyperbola equation
def is_on_hyperbola (x y : ℝ) : Prop :=
  x < 0 ∧ x^2 / 9 - y^2 / 16 = 1

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem hyperbola_equation (x y : ℝ) :
  (distance x y F2.1 F2.2 - distance x y F1.1 F1.2 = distance_diff) →
  is_on_hyperbola x y := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l73_7346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_proof_l73_7302

/-- Proves that given a tree growing at 2 feet per week, with 4 weeks per month,
    and a total height of 42 feet after 4 months, the current height of the tree is 10 feet. -/
theorem tree_height_proof (growth_rate : ℝ) (weeks_per_month : ℕ) (months : ℕ) (final_height : ℝ) :
  growth_rate = 2 →
  weeks_per_month = 4 →
  months = 4 →
  final_height = 42 →
  final_height - (growth_rate * (weeks_per_month : ℝ) * (months : ℝ)) = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_proof_l73_7302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_monotone_increasing_l73_7309

open Real

/-- The function f(x) defined in the problem -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := Real.exp (1 - x) + t * Real.log x

/-- The theorem stating the minimum value of t for which f is monotonically increasing -/
theorem min_t_for_monotone_increasing :
  (∃ t_min : ℝ, ∀ t : ℝ, t ≥ t_min ↔ 
    (∀ x y : ℝ, x > 0 → y > 0 → x < y → f t x < f t y)) ∧
  (∀ t_min : ℝ, (∀ t : ℝ, t ≥ t_min ↔ 
    (∀ x y : ℝ, x > 0 → y > 0 → x < y → f t x < f t y)) → 
    t_min = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_monotone_increasing_l73_7309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_54pi_l73_7378

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The region described in the problem -/
structure Region where
  part1_height : ℝ
  part1_width : ℝ
  part2_height : ℝ
  part2_width : ℝ

/-- The volume of the solid formed by rotating the region -/
noncomputable def rotatedVolume (r : Region) : ℝ :=
  cylinderVolume r.part1_height r.part1_width + cylinderVolume (r.part2_width / 2) r.part2_height

/-- The specific region described in the problem -/
def problemRegion : Region where
  part1_height := 6
  part1_width := 1
  part2_height := 2
  part2_width := 3

theorem volume_is_54pi : rotatedVolume problemRegion = 54 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_54pi_l73_7378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_to_equalize_l73_7367

/-- Represents the state of gas cylinders with their pressures -/
def CylinderState := List ℝ

/-- Represents a connection operation on cylinders -/
def ConnectOperation := List ℕ

/-- Checks if a list has exactly 40 elements -/
def has_40_elements (l : List α) : Prop := l.length = 40

/-- Checks if all elements in a list are equal -/
def all_equal (l : List α) [DecidableEq α] : Prop := ∀ x y, x ∈ l → y ∈ l → x = y

/-- Checks if a connection operation is valid (connects at most k cylinders) -/
def valid_operation (k : ℕ) (op : ConnectOperation) : Prop := op.length ≤ k

/-- Applies a single connection operation to a cylinder state -/
noncomputable def apply_operation (state : CylinderState) (op : ConnectOperation) : CylinderState :=
  sorry

/-- Checks if a sequence of operations can equalize all pressures -/
def can_equalize (k : ℕ) (initial : CylinderState) : Prop :=
  ∃ (ops : List ConnectOperation),
    (∀ op ∈ ops, valid_operation k op) ∧
    all_equal (ops.foldl apply_operation initial)

/-- The main theorem: 5 is the smallest k that allows equalizing 40 cylinders -/
theorem smallest_k_to_equalize :
  (∀ initial : CylinderState, has_40_elements initial → can_equalize 5 initial) ∧
  (∀ k < 5, ∃ initial : CylinderState, has_40_elements initial ∧ ¬can_equalize k initial) :=
by
  sorry

#check smallest_k_to_equalize

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_to_equalize_l73_7367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_equals_pi_over_two_plus_four_thirds_l73_7360

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then Real.sqrt (1 - x^2)
  else if 1 < x ∧ x ≤ 2 then x^2 - 1
  else 0  -- Define f as 0 outside the given intervals

-- State the theorem
theorem integral_of_f_equals_pi_over_two_plus_four_thirds :
  ∫ x in (-1)..2, f x = π / 2 + 4 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_f_equals_pi_over_two_plus_four_thirds_l73_7360
