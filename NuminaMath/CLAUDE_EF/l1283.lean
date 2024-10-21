import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1283_128337

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

/-- Definition of major axis length -/
def majorAxisLength (a : ℝ) : ℝ := 2 * a

/-- Definition of the line intersecting the ellipse -/
def intersectingLine (x y k : ℝ) : Prop :=
  y = k * x - 1/2

/-- Definition of perpendicularity -/
def perpendicular (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) = 0

/-- Main theorem -/
theorem ellipse_properties :
  ∀ a b : ℝ,
  ellipse x y a b →
  eccentricity a b = Real.sqrt 6 / 3 →
  majorAxisLength a = 2 * Real.sqrt 3 →
  (∀ x y : ℝ, ellipse x y a b ↔ x^2 / 3 + y^2 = 1) ∧
  (∃ M : ℝ × ℝ,
    M.1 = 0 ∧ M.2 = 1 ∧
    ∀ k : ℝ, ∀ A B : ℝ × ℝ,
    intersectingLine A.1 A.2 k →
    intersectingLine B.1 B.2 k →
    ellipse A.1 A.2 a b →
    ellipse B.1 B.2 a b →
    perpendicular M.1 M.2 A.1 A.2 B.1 B.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1283_128337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1283_128377

theorem trigonometric_inequality : 
  Real.tan (7 * π / 4) < Real.sin (17 * π / 12) ∧ Real.sin (17 * π / 12) < Real.cos (4 * π / 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1283_128377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_perimeter_l1283_128319

/-- Represents a staircase-shaped region -/
structure StaircaseRegion where
  base : ℝ
  num_steps : ℕ
  step_size : ℝ
  area : ℝ

/-- Calculates the perimeter of a staircase-shaped region -/
noncomputable def perimeter (s : StaircaseRegion) : ℝ :=
  s.base + s.area / s.base + s.num_steps * s.step_size + s.step_size * s.num_steps

theorem staircase_perimeter (s : StaircaseRegion) 
  (h1 : s.base = 10)
  (h2 : s.num_steps = 7)
  (h3 : s.step_size = 1)
  (h4 : s.area = 60) :
  perimeter s = 34.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_perimeter_l1283_128319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l1283_128397

theorem xy_value (x y : ℝ) (h1 : (16 : ℝ)^x = 4) (h2 : (9 : ℝ)^y = 4) : 
  x * y = Real.log 2 / Real.log 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l1283_128397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_one_l1283_128390

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^2
  else if 1 < x ∧ x ≤ 2 then Real.log x / Real.log 2 + 1
  else 0  -- This else case is added to make the function total

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_sum_equals_one :
  (is_periodic f 4) →
  (is_odd f) →
  f 2014 + f 2015 = 1 := by
    sorry

#check f_sum_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_one_l1283_128390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_line_equation_special_line_equation_l1283_128388

-- Part 1
def line_with_equal_intercepts (l : Set (ℝ × ℝ)) : Prop :=
  (4, 1) ∈ l ∧ ∃ k, k ≠ 0 ∧ (k, 0) ∈ l ∧ (0, k) ∈ l

theorem equal_intercepts_line_equation (l : Set (ℝ × ℝ)) :
  line_with_equal_intercepts l →
  (∀ x y, (x, y) ∈ l ↔ x - 4*y = 0) ∨
  (∀ x y, (x, y) ∈ l ↔ x + y = 5) :=
sorry

-- Part 2
def line_through_points (l : Set (ℝ × ℝ)) (θ : ℝ) : Prop :=
  (3, 4) ∈ l ∧ (Real.cos θ, Real.sin θ) ∈ l ∧ θ ≠ Real.pi/2

theorem special_line_equation (l : Set (ℝ × ℝ)) (θ : ℝ) :
  line_through_points l θ →
  ∀ x y, (x, y) ∈ l ↔ y = (4/3) * x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_line_equation_special_line_equation_l1283_128388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1283_128346

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Theorem statement
theorem power_function_through_point :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f (Real.sqrt 2) = (1 : ℝ) / 2 →
  ∀ x : ℝ, f x = x ^ ((-2) : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1283_128346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l1283_128338

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 4 * x^2 - y^2 + 64 = 0

-- Define a point on the hyperbola
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- State the theorem
theorem hyperbola_focus_distance (P : PointOnHyperbola) (F1 F2 : ℝ × ℝ) :
  distance P.x P.y F1.1 F1.2 = 1 →
  ∃ c : ℝ, c > 0 ∧ F1 = (c, 0) ∧ F2 = (-c, 0) ∧ distance P.x P.y F2.1 F2.2 = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l1283_128338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kylie_earrings_count_l1283_128399

/-- The number of beaded earrings Kylie made on Wednesday -/
def earrings_count (monday_necklaces : ℕ) (tuesday_necklaces : ℕ) (wednesday_bracelets : ℕ)
  (beads_per_necklace : ℕ) (beads_per_bracelet : ℕ) (beads_per_earring : ℕ) : ℕ :=
  let total_beads := 325
  let necklace_beads := (monday_necklaces + tuesday_necklaces) * beads_per_necklace
  let bracelet_beads := wednesday_bracelets * beads_per_bracelet
  let earring_beads := total_beads - necklace_beads - bracelet_beads
  earring_beads / beads_per_earring

theorem kylie_earrings_count :
  earrings_count 10 2 5 20 10 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kylie_earrings_count_l1283_128399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_valid_triple_l1283_128318

/-- A coloring of integers from 1 to n using two colors -/
def Coloring (n : ℕ) := Fin n → Bool

/-- Predicate that checks if three distinct integers of the same color satisfy 2a + b = c -/
def HasValidTriple {n : ℕ} (c : Coloring n) : Prop :=
  ∃ a b d : Fin n, a ≠ b ∧ b ≠ d ∧ a ≠ d ∧
    c a = c b ∧ c b = c d ∧
    (2 * (a.val) + b.val = d.val)

/-- The main theorem stating that 15 is the minimum n satisfying the condition -/
theorem min_n_for_valid_triple : 
  (∀ c : Coloring 15, HasValidTriple c) ∧ 
  (∀ m < 15, ∃ c : Coloring m, ¬HasValidTriple c) := by
  sorry

#check min_n_for_valid_triple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_valid_triple_l1283_128318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_distance_theorem_l1283_128324

/-- The distance between parallel sides of a trapezium -/
noncomputable def distance_between_sides (a b area : ℝ) : ℝ :=
  (2 * area) / (a + b)

/-- Theorem: The distance between parallel sides of a trapezium with given dimensions -/
theorem trapezium_distance_theorem (a b area : ℝ) 
  (ha : a = 20) 
  (hb : b = 18) 
  (harea : area = 266) : 
  distance_between_sides a b area = 14 := by
  sorry

#check trapezium_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_distance_theorem_l1283_128324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_and_alpha_values_l1283_128308

noncomputable section

variable (α : ℝ)
def a : ℝ × ℝ := (Real.cos α, Real.sin α)
def b : ℝ × ℝ := (-1/2, Real.sqrt 3/2)

theorem vector_perpendicular_and_alpha_values :
  (0 ≤ α) ∧ (α < 2*Real.pi) →
  (((a α).1 + b.1) * ((a α).1 - b.1) + ((a α).2 + b.2) * ((a α).2 - b.2) = 0) ∧
  ((3 * ((a α).1^2 + (a α).2^2) + 2*Real.sqrt 3*((a α).1*b.1 + (a α).2*b.2) + (b.1^2 + b.2^2) =
    ((a α).1^2 + (a α).2^2) - 2*Real.sqrt 3*((a α).1*b.1 + (a α).2*b.2) + 3*(b.1^2 + b.2^2)) →
   (α = Real.pi/6 ∨ α = 7*Real.pi/6)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_and_alpha_values_l1283_128308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1283_128364

-- Define the ellipse
def Ellipse (F₁ F₂ : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P F₁ + dist P F₂ = dist F₁ F₂ * 2}

-- Define the arithmetic mean property
def ArithmeticMeanProperty (F₁ F₂ : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  dist F₁ F₂ = (dist P F₁ + dist P F₂) / 2

-- Theorem statement
theorem ellipse_equation (F₁ F₂ : ℝ × ℝ) (h₁ : F₁ = (-1, 0)) (h₂ : F₂ = (1, 0)) :
  ∀ P ∈ Ellipse F₁ F₂, ArithmeticMeanProperty F₁ F₂ P →
  (let (x, y) := P; x^2 / 4 + y^2 / 3 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1283_128364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_hops_needed_l1283_128351

/-- The number of elements in the sequence -/
def n : ℕ := 10

/-- The number of attempts needed to determine the i-th element -/
def attempts (i : ℕ) : ℕ := i * (n - i)

/-- The total number of attempts needed to determine the full sequence -/
def total_attempts : ℕ := (Finset.sum (Finset.range n) attempts) + n + 1

/-- The theorem stating the minimum number of hops needed -/
theorem min_hops_needed : total_attempts = 176 := by
  sorry

#eval total_attempts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_hops_needed_l1283_128351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l1283_128349

open Real

-- Define an acute triangle
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  sides : a > 0 ∧ b > 0 ∧ c > 0

-- State the theorem
theorem acute_triangle_properties (t : AcuteTriangle) 
  (h : sqrt 3 * tan t.A * tan t.C = tan t.A + tan t.C + sqrt 3) :
  t.B = π/3 ∧ 
  ∃ (x : ℝ), x ∈ Set.Ioo (sqrt 3 / 2) 1 ∧ cos t.A + cos t.C = x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l1283_128349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l1283_128383

/-- Represents the distribution of scores in a mathematics test -/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score95 : ℝ
  sum_to_one : score60 + score75 + score85 + score95 = 1
  score60_value : score60 = 0.15
  score75_value : score75 = 0.25
  score85_value : score85 = 0.20

/-- Calculates the mean score given a score distribution -/
def meanScore (dist : ScoreDistribution) : ℝ :=
  60 * dist.score60 + 75 * dist.score75 + 85 * dist.score85 + 95 * dist.score95

/-- Determines the median score given a score distribution -/
noncomputable def medianScore (dist : ScoreDistribution) : ℝ :=
  if dist.score60 + dist.score75 ≤ 0.5 ∧ dist.score60 + dist.score75 + dist.score85 > 0.5
  then 85
  else 0  -- This else case should never occur given our constraints

/-- Theorem stating the difference between median and mean scores -/
theorem median_mean_difference (dist : ScoreDistribution) :
  medianScore dist - meanScore dist = 2.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l1283_128383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1283_128374

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else x^2 - 2*x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (-a) + f a ≤ 2 * f 3) ↔ -3 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1283_128374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_volume_l1283_128325

/-- The volume of a sphere given its radius -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem larger_sphere_volume
  (r : ℝ) -- radius of the smaller sphere
  (h1 : r > 0) -- radius is positive
  (h2 : sphereVolume r = 36) -- volume of smaller sphere is 36
  : sphereVolume (2 * r) = 288 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_volume_l1283_128325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l1283_128357

theorem negation_of_existence {R : Type} [LinearOrderedField R] :
  (¬ ∃ x : R, x + (2 : R) ≤ 0) ↔ (∀ x : R, x + (2 : R) > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l1283_128357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_320_l1283_128322

theorem closest_perfect_square_to_320 : 
  ∀ n : ℤ, n ≠ 0 → n^2 ≠ 324 → |320 - 324| ≤ |320 - n^2| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_320_l1283_128322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_270_deg_l1283_128369

/-- Applies a 270° counter-clockwise rotation to a complex number -/
def rotate270 (z : ℂ) : ℂ := -Complex.I * z

theorem rotation_270_deg (z : ℂ) (h : z = 4 - 7*Complex.I) : 
  rotate270 z = -7 - 4*Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_270_deg_l1283_128369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_abs_power_sum_linear_function_through_points_l1283_128304

-- Problem 1
theorem cube_root_abs_power_sum : (8 : ℝ) ^ (1/3) + |(-5)| + (-1) ^ 2023 = 6 := by sorry

-- Problem 2
theorem linear_function_through_points :
  ∀ k b : ℝ,
  (k * 0 + b = 1) →
  (k * 2 + b = 5) →
  ∀ x : ℝ, k * x + b = 2 * x + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_abs_power_sum_linear_function_through_points_l1283_128304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_number_l1283_128366

theorem smaller_number (a b c d : ℝ) (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x / y = a / (b + 1)) 
  (h4 : 0 < a) 
  (h5 : a < b + 1) 
  (h6 : x + y = c) 
  (h7 : x - y = d) : 
  min x y = a * c / (a + b + 1) := by
  sorry

#check smaller_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_number_l1283_128366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_acquaintances_meeting_size_iff_valid_n_l1283_128305

/-- Represents a person at the meeting -/
structure Person where
  id : Nat

/-- Represents the meeting -/
structure Meeting where
  n : Nat
  people : Finset Person
  acquaintance : Person → Person → Bool

/-- The meeting satisfies the given conditions -/
def MeetingConditions (m : Meeting) : Prop :=
  (∀ p q r : Person, m.acquaintance p q = true ∧ m.acquaintance p r = true → m.acquaintance q r = false) ∧
  (∀ p q : Person, m.acquaintance p q = false →
    ∃! (r s : Person), r ≠ s ∧ m.acquaintance p r = true ∧ m.acquaintance p s = true ∧ m.acquaintance q r = true ∧ m.acquaintance q s = true)

/-- The number of acquaintances for a person -/
def NumAcquaintances (m : Meeting) (p : Person) : Nat :=
  (m.people.filter (fun q => m.acquaintance p q)).card

/-- All attendees have the same number of acquaintances -/
theorem all_same_acquaintances (m : Meeting) (h : MeetingConditions m) :
  ∀ p q : Person, p ∈ m.people → q ∈ m.people → NumAcquaintances m p = NumAcquaintances m q := by
  sorry

/-- The possible values of n -/
def ValidN (n : Nat) : Prop :=
  ∃ k : Nat, k > 0 ∧ n = ((2 * k - 1)^2 + 7) / 8

/-- The meeting size satisfies the conditions if and only if it's a valid n -/
theorem meeting_size_iff_valid_n (m : Meeting) (h : MeetingConditions m) :
  ValidN m.n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_acquaintances_meeting_size_iff_valid_n_l1283_128305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l1283_128372

theorem trig_equation_solution (θ : ℝ) (m : ℝ) 
  (h1 : Real.sin θ = (m - 3) / (m + 5))
  (h2 : Real.cos θ = (4 - 2*m) / (m + 5)) :
  m = 0 ∨ m = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l1283_128372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_people_moving_per_hour_l1283_128387

/-- The number of people moving to Texas -/
def people_moving : ℕ := 3500

/-- The number of days -/
def num_days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Function to round a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- Theorem stating that the average number of people moving to Texas per hour, 
    rounded to the nearest whole number, is 29 -/
theorem average_people_moving_per_hour :
  round_to_nearest ((people_moving : ℝ) / (num_days * hours_per_day)) = 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_people_moving_per_hour_l1283_128387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_opposite_c_is_zero_l1283_128316

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem angle_opposite_c_is_zero (t : Triangle) 
  (h : (t.a + t.b + t.c) * (t.a + t.b - t.c) = 4 * t.a * t.b) : 
  Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_opposite_c_is_zero_l1283_128316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1283_128379

noncomputable def g (x : ℝ) : ℝ := (3*x + 4) / (x + 3)

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 1 ∧ g x = y}

theorem g_properties :
  (∃ n ∈ T, ∀ y ∈ T, n ≤ y) ∧ 
  (∃ N : ℝ, (∀ y ∈ T, y ≤ N) ∧ N ∉ T) := by
  sorry

#check g_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1283_128379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_change_l1283_128392

/-- Represents a configuration of three point charges -/
structure ChargeConfig where
  -- Positions of the three charges
  pos1 : ℝ × ℝ
  pos2 : ℝ × ℝ
  pos3 : ℝ × ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculates the energy stored between two charges given their distance -/
noncomputable def energyBetweenCharges (d : ℝ) : ℝ :=
  5 / d

/-- Calculates the total energy stored in a charge configuration -/
noncomputable def totalEnergy (config : ChargeConfig) : ℝ :=
  energyBetweenCharges (distance config.pos1 config.pos2) +
  energyBetweenCharges (distance config.pos2 config.pos3) +
  energyBetweenCharges (distance config.pos3 config.pos1)

/-- The initial equilateral triangle configuration -/
noncomputable def initialConfig : ChargeConfig :=
  { pos1 := (0, 0), pos2 := (1, 0), pos3 := (0.5, Real.sqrt 3 / 2) }

/-- The configuration after moving one charge -/
noncomputable def newConfig : ChargeConfig :=
  { pos1 := (0, 0), pos2 := (1, 0), pos3 := (1/6, Real.sqrt 3 / 6) }

theorem energy_change :
  totalEnergy initialConfig = 15 →
  totalEnergy newConfig = 27.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_change_l1283_128392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_six_l1283_128313

theorem subsets_containing_six (S : Finset Nat) : S = {1, 2, 3, 4, 5, 6} →
  (Finset.filter (fun s => s ∈ Finset.powerset S ∧ 6 ∈ s) (Finset.powerset S)).card = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_six_l1283_128313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l1283_128331

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 8 * (Real.cos (x / 2))^2

-- State the theorem
theorem sin_theta_value (θ : ℝ) :
  (∀ x, f x ≤ f θ) → Real.sin θ = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l1283_128331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_count_theorem_l1283_128339

def is_simple (n : ℕ) : Prop :=
  n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1)

def valid_knight_count (total : ℕ) (knights : ℕ) : Prop :=
  knights ≤ total ∧
  ∀ i : ℕ, i < total →
    let left := min i knights
    let right := min (total - i - 1) (knights - left)
    is_simple (Int.natAbs (right - left)) ∨ (left + right ≠ knights)

theorem knight_count_theorem (total : ℕ) (h : total = 2019) :
  {k : ℕ | valid_knight_count total k} = {0, 2, 4, 6, 8} := by
  sorry

#check knight_count_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_count_theorem_l1283_128339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_403_l1283_128398

theorem sum_of_divisors_403 (l m : ℕ) :
  let n : ℕ := 2^l * 3^m
  let sum_of_divisors := λ x : ℕ => (Finset.sum (Finset.filter (λ y => x % y = 0) (Finset.range (x + 1))) id)
  sum_of_divisors n = 403 → n = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_403_l1283_128398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_spade_result_l1283_128384

/-- The ♠ operation for positive real numbers -/
noncomputable def spade (x y : ℝ) : ℝ := x - 1 / y

/-- Theorem stating the result of the nested spade operation -/
theorem nested_spade_result :
  spade 3 (spade (5/2) 3) = 33/13 := by
  -- Expand the definition of spade
  unfold spade
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_spade_result_l1283_128384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_formula_u_gcd_l1283_128315

def u : ℕ → ℤ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 1
  | n + 3 => u (n + 2) + 2 * u (n + 1)

theorem u_formula (n p : ℕ) (h : p > 1) :
  u (n + p) = u (n + 1) * u p + 2 * u n * u (p - 1) := by sorry

theorem u_gcd (n : ℕ) :
  Int.gcd (u n) (u (n + 3)) = if n % 3 = 0 then 3 else 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_formula_u_gcd_l1283_128315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_gaming_hobby_duration_l1283_128341

noncomputable def joe_gaming_hobby (initial_funds : ℝ) (max_game_cost : ℝ) (min_resale_price : ℝ) 
  (max_games_per_month : ℕ) (subscription_cost : ℝ) : ℕ :=
  let worst_case_expense := max_game_cost * (max_games_per_month : ℝ)
  let worst_case_income := min_resale_price * (max_games_per_month : ℝ)
  let monthly_net_expense := worst_case_expense - worst_case_income + subscription_cost
  ⌊initial_funds / monthly_net_expense⌋₊

theorem joe_gaming_hobby_duration :
  joe_gaming_hobby 240 60 20 3 15 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_gaming_hobby_duration_l1283_128341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_two_rays_l1283_128361

-- Define the two fixed points
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points M satisfying the condition
def trajectory_set : Set (ℝ × ℝ) :=
  {M | |distance M F₁ - distance M F₂| = 6}

-- Theorem statement
theorem trajectory_is_two_rays :
  ∃ (A B : ℝ × ℝ),
    trajectory_set = {M | ∃ t : ℝ, t ≥ 0 ∧ (M = (A.1 + t * (F₁.1 - A.1), A.2 + t * (F₁.2 - A.2)) ∨
                                           M = (B.1 + t * (F₂.1 - B.1), B.2 + t * (F₂.2 - B.2)))} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_two_rays_l1283_128361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1283_128329

/-- An equilateral triangle with side length 12 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 12

/-- The perimeter of an equilateral triangle -/
def perimeter (t : EquilateralTriangle) : ℝ :=
  3 * t.side_length

/-- The area of an equilateral triangle -/
noncomputable def area (t : EquilateralTriangle) : ℝ :=
  (Real.sqrt 3 / 4) * t.side_length ^ 2

/-- Theorem stating the perimeter and area of the specific equilateral triangle -/
theorem equilateral_triangle_properties (t : EquilateralTriangle) :
  perimeter t = 36 ∧ area t = 36 * Real.sqrt 3 := by
  sorry

#check equilateral_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1283_128329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_satisfying_inequality_three_satisfies_inequality_three_is_greatest_l1283_128365

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ+, (x : ℝ)^4 / (x : ℝ)^2 < 10 → x ≤ 3 :=
by
  sorry

theorem three_satisfies_inequality :
  (3 : ℝ)^4 / (3 : ℝ)^2 < 10 :=
by
  sorry

theorem three_is_greatest :
  ∃ x : ℕ+, (x : ℝ)^4 / (x : ℝ)^2 < 10 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_satisfying_inequality_three_satisfies_inequality_three_is_greatest_l1283_128365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_2x_implies_a_in_range_l1283_128352

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- State the theorem
theorem f_geq_2x_implies_a_in_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 2 * x) → a ∈ Set.Icc (-2) (Real.exp 1 - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_2x_implies_a_in_range_l1283_128352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_one_l1283_128356

-- Define a 3x3 grid
def Grid := Fin 3 → Fin 3 → Nat

-- Define a valid grid
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j k : Fin 3, i ≠ j → g i k ≠ g j k) ∧  -- Each column contains distinct values
  (∀ i j k : Fin 3, j ≠ k → g i j ≠ g i k) ∧  -- Each row contains distinct values
  (∀ i j : Fin 3, g i j ∈ ({1, 2, 3} : Set Nat))  -- All values are 1, 2, or 3

-- Define the initial grid setup
def initial_setup (g : Grid) : Prop :=
  g 0 0 = 2 ∧ g 0 2 = 1 ∧ g 1 2 = 3 ∧ g 2 1 = 2

-- Theorem: The center square must be 1
theorem center_is_one (g : Grid) (h1 : is_valid_grid g) (h2 : initial_setup g) : 
  g 1 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_one_l1283_128356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emma_bike_time_l1283_128353

/-- The time taken for Emma to cover a one-mile stretch of highway -/
theorem emma_bike_time (highway_length : Real) (highway_width : Real) (bike_speed : Real) 
  (h1 : highway_length = 1) -- mile
  (h2 : highway_width = 50 / 5280) -- 50 feet converted to miles
  (h3 : bike_speed = 10) -- miles per hour
  : highway_length * π / bike_speed = π / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emma_bike_time_l1283_128353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_function_l1283_128300

noncomputable def f (ω φ x : Real) : Real := Real.sin (ω * x + φ)

theorem symmetric_sine_function
  (ω φ : Real)
  (h_ω : 0 < ω ∧ ω < 1)
  (h_φ : 0 ≤ φ ∧ φ ≤ Real.pi)
  (h_even : ∀ x, f ω φ x = f ω φ (-x))
  (h_sym : ∀ x, f ω φ (x + 3 * Real.pi / 4) = -f ω φ (-x + 3 * Real.pi / 4)) :
  φ = Real.pi / 2 ∧ ω = 2 / 3 ∧
  (∀ x ∈ Set.Icc (-3 * Real.pi / 4) (Real.pi / 2),
    f ω φ x ≤ 1 ∧ f ω φ x ≥ 0 ∧
    (∃ x₁ ∈ Set.Icc (-3 * Real.pi / 4) (Real.pi / 2), f ω φ x₁ = 1) ∧
    (∃ x₂ ∈ Set.Icc (-3 * Real.pi / 4) (Real.pi / 2), f ω φ x₂ = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_function_l1283_128300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_destination_l1283_128386

noncomputable def total_distance : ℝ := 280

noncomputable def first_stop_fraction : ℝ := 1/2

noncomputable def second_stop_fraction : ℝ := 1/4

noncomputable def distance_after_second_stop (total : ℝ) (first_frac : ℝ) (second_frac : ℝ) : ℝ :=
  total * (1 - first_frac) * (1 - second_frac)

theorem distance_to_destination : 
  distance_after_second_stop total_distance first_stop_fraction second_stop_fraction = 105 := by
  -- Unfold the definitions
  unfold distance_after_second_stop total_distance first_stop_fraction second_stop_fraction
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_destination_l1283_128386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_six_l1283_128306

/-- The exchange rate from USD to yen -/
noncomputable def usd_to_yen : ℚ := 100

/-- The cost of tea in yen -/
noncomputable def tea_cost : ℚ := 250

/-- The cost of sandwich in yen -/
noncomputable def sandwich_cost : ℚ := 350

/-- The total cost of tea and sandwich in USD -/
noncomputable def total_cost_usd : ℚ := (tea_cost + sandwich_cost) / usd_to_yen

theorem total_cost_is_six : total_cost_usd = 6 := by
  -- Unfold the definitions
  unfold total_cost_usd usd_to_yen tea_cost sandwich_cost
  -- Simplify the arithmetic
  simp [add_div]
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_six_l1283_128306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_third_l1283_128333

noncomputable def f (x : ℝ) : ℝ := Real.log (2 - 3 * x)

theorem f_derivative_at_one_third :
  HasDerivAt f (-3) (1/3) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_third_l1283_128333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_l1283_128395

/-- The diameter of a wheel given its revolutions and distance covered -/
noncomputable def wheel_diameter (revolutions : ℝ) (distance : ℝ) : ℝ :=
  distance / (revolutions * Real.pi)

/-- Theorem stating the diameter of the wheel is approximately 27.979 cm -/
theorem wheel_diameter_approx :
  let revolutions : ℝ := 11.010009099181074
  let distance : ℝ := 968
  abs (wheel_diameter revolutions distance - 27.979) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_l1283_128395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_percentage_l1283_128343

/-- Given a mixture of wine and water, prove that the initial percentage of water is 20% -/
theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_volume = 200)
  (h2 : added_water = 13.333333333333334)
  (h3 : final_water_percentage = 25) :
  ∃ initial_water_percentage : ℝ,
    initial_water_percentage = 20 ∧
    final_water_percentage / 100 * (initial_volume + added_water) =
    initial_volume * (initial_water_percentage / 100) + added_water :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_percentage_l1283_128343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_l1283_128311

-- Define the family of integral curves
def IntegralCurve (p q : ℝ → ℝ) := 
  {y : ℝ → ℝ | ∀ x, (deriv y x) + p x * y x = q x}

-- Define the tangent line at a point
noncomputable def TangentLine (p q : ℝ → ℝ) (x y : ℝ) := 
  {ξη : ℝ × ℝ | (ξη.2 : ℝ) - y = (q x - p x * y) * ((ξη.1 : ℝ) - x)}

-- Define the intersection point
noncomputable def IntersectionPoint (p q : ℝ → ℝ) (x : ℝ) := (x + 1 / p x, q x / p x)

-- Theorem statement
theorem tangent_intersection 
  (p q : ℝ → ℝ) 
  (h : ∀ x, p x ≠ 0) :
  ∀ x, ∀ y₁ y₂, y₁ ∈ IntegralCurve p q → y₂ ∈ IntegralCurve p q →
    ∃ S, S ∈ TangentLine p q x (y₁ x) ∧ 
         S ∈ TangentLine p q x (y₂ x) ∧
         S = IntersectionPoint p q x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_l1283_128311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_eyes_l1283_128312

/-- The number of eyes each ant has, given the conditions of Nina's insect collection. -/
theorem ant_eyes (spiders : ℕ) (ants : ℕ) (spider_eyes : ℕ) (total_eyes : ℕ) 
  (h1 : spiders = 3)
  (h2 : ants = 50)
  (h3 : spider_eyes = 8)
  (h4 : total_eyes = 124) :
  ∃ ant_eyes : ℕ, ant_eyes = 2 ∧ total_eyes = spiders * spider_eyes + ants * ant_eyes := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_eyes_l1283_128312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_zeros_min_m_value_l1283_128332

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.cos x + m * (x + Real.pi / 2) * Real.sin x

-- Theorem for the number of zeros
theorem number_of_zeros (m : ℝ) (h : m ≤ 1) :
  ∃! x, x ∈ Set.Ioo (-Real.pi) 0 ∧ f m x = 0 := by sorry

-- Theorem for the minimum value of m
theorem min_m_value :
  ∀ ε > 0, ∃ m, m = -1 ∧
  ∀ t > 0, ∀ x ∈ Set.Ioo (-Real.pi/2 - t) (-Real.pi/2),
    |f m x| < -2*x - Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_zeros_min_m_value_l1283_128332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l1283_128391

/-- The length of a chord on a parabola with specific symmetry properties -/
theorem parabola_chord_length :
  ∀ (A B : ℝ × ℝ),
  (A.2 = -A.1^2 + 3) →
  (B.2 = -B.1^2 + 3) →
  (A ≠ B) →
  (A.1 + A.2 = -(B.1 + B.2)) →
  ‖A - B‖ = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l1283_128391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_problem_l1283_128360

-- Define variables for current ages
variable (john_age : ℕ)
variable (sister_age : ℕ)
variable (cousin_age : ℚ)

-- Define the conditions from the problem
def condition1 (j : ℕ) : Prop := j + 9 = 3 * (j - 11)
def condition2 (j s : ℕ) : Prop := s = 2 * j
def condition3 (j s : ℕ) (c : ℚ) : Prop := c = (j + s) / 2

-- Theorem to prove
theorem age_problem :
  ∃ (j s : ℕ) (c : ℚ), condition1 j ∧ condition2 j s ∧ condition3 j s c ∧
  j = 21 ∧ s = 42 ∧ c = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_problem_l1283_128360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_real_coefficients_binomial_expansion_l1283_128303

theorem sum_real_coefficients_binomial_expansion (i : ℂ) :
  let x : ℂ := 0
  let T : ℝ := (((1 + i*x)^2011 + (1 - i*x)^2011) / 2).re
  T = -2^(2011/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_real_coefficients_binomial_expansion_l1283_128303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_and_S_not_third_l1283_128376

-- Define the set of runners
inductive Runner : Type
| P | Q | R | S | T | U

-- Define the relation "beats" between runners
axiom beats : Runner → Runner → Prop

-- Define the relation "finishes_before" between runners
axiom finishes_before : Runner → Runner → Prop

-- Define the conditions
axiom beats_PQ : beats Runner.P Runner.Q
axiom beats_PR : beats Runner.P Runner.R
axiom beats_QS : beats Runner.Q Runner.S
axiom T_after_P_before_Q : finishes_before Runner.P Runner.T ∧ finishes_before Runner.T Runner.Q
axiom U_before_S_after_Q : finishes_before Runner.Q Runner.U ∧ finishes_before Runner.U Runner.S

-- Define what it means to finish third
def finishes_third (r : Runner) : Prop :=
  ∃ (a b : Runner), (a ≠ r) ∧ (b ≠ r) ∧ (a ≠ b) ∧
  finishes_before a r ∧ finishes_before b r ∧
  ∀ (x : Runner), x ≠ r → x ≠ a → x ≠ b → finishes_before r x

-- State the theorem
theorem P_and_S_not_third :
  ¬(finishes_third Runner.P) ∧ ¬(finishes_third Runner.S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_and_S_not_third_l1283_128376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_speed_problem_l1283_128355

/-- 
Given two students walking in opposite directions, this theorem proves
that if one student's speed is 9 km/hr and they are 60 km apart after 4 hours,
the other student's speed must be 6 km/hr.
-/
theorem student_speed_problem (v : ℝ) : 
  v > 0 →  -- Ensure speed is positive
  (4 : ℝ) * (v + 9) = 60 → 
  v = 6 := by
  intro h_pos h_eq
  -- Proof steps would go here
  sorry

#check student_speed_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_speed_problem_l1283_128355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l1283_128301

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (-10, -4)
def radius1 : ℝ := 13
def center2 : ℝ × ℝ := (3, 9)
noncomputable def radius2 : ℝ := Real.sqrt 65

-- Define the circles
def circle1 (x y : ℝ) : Prop :=
  (x - center1.1)^2 + (y - center1.2)^2 = radius1^2

def circle2 (x y : ℝ) : Prop :=
  (x - center2.1)^2 + (y - center2.2)^2 = radius2^2

-- Theorem statement
theorem intersection_line_equation :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y → x + y = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l1283_128301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1283_128382

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then |x - 1| else 3^x

-- Theorem statement
theorem function_properties :
  (f (f (-2)) = 27) ∧ (∀ a : ℝ, f a = 2 ↔ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1283_128382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_N_value_l1283_128381

theorem unique_N_value (N : ℕ) (h1 : N > 1) 
  (d : ℕ → ℕ) (s : ℕ) 
  (h2 : d 1 = 1 ∧ d s = N)
  (h3 : ∀ i ∈ Finset.range (s - 1), d i < d (i + 1))
  (h4 : ∀ i ∈ Finset.range s, ∃ k : ℕ, N = k * d i)
  (h5 : (Finset.range (s - 1)).sum (λ i => Nat.gcd (d i) (d (i + 1))) = N - 2) :
  N = 3 := by
  sorry

#check unique_N_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_N_value_l1283_128381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_role_assignment_l1283_128307

/-- The number of ways to assign theater roles given specific conditions -/
theorem theater_role_assignment (men women : ℕ) 
  (male_roles female_roles either_roles : ℕ) : 
  men = 7 → women = 4 → male_roles = 3 → female_roles = 2 → either_roles = 1 →
  (men.factorial / (men - male_roles).factorial) * 
  (women.factorial / (women - female_roles).factorial) * 
  (men + women - male_roles - female_roles) = 15120 := by
  intros hmen hwomen hmale hfemale heither
  sorry

#check theater_role_assignment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_role_assignment_l1283_128307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_A_l1283_128359

/-- The locus of point A given an ellipse and equilateral triangle condition -/
theorem locus_of_point_A (x y : ℝ) (B : ℂ) :
  (x^2 / 9 + y^2 / 5 = 1) →  -- Given ellipse equation
  (Complex.abs (B + 2) + Complex.abs (B - 2) = 6) →  -- B is on the ellipse
  (∃ (A : ℂ), Complex.abs (B - 2) = Complex.abs (A - 2) ∧  -- FAB is equilateral
              Complex.arg ((B - 2) / (A - 2)) = π / 3) →  -- F, A, B are counterclockwise
  (∃ (z : ℂ), Complex.abs (z - 2) + Complex.abs (z - 2 * Complex.I * Real.sqrt 3) = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_A_l1283_128359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l1283_128375

/-- The total number of numbers in the lottery -/
def total_numbers : ℕ := 90

/-- The number of numbers drawn in each lottery -/
def drawn_numbers : ℕ := 5

/-- The number of winning numbers from the previous week -/
def previous_winners : ℕ := 5

/-- The probability of drawing at least one number from the previous week's winning numbers -/
def probability : ℚ := 1 - (Nat.choose (total_numbers - previous_winners) drawn_numbers : ℚ) / 
                       (Nat.choose total_numbers drawn_numbers : ℚ)

/-- The theorem statement -/
theorem lottery_probability : probability = 1238639 / 4883252 := by
  -- Expand the definition of probability
  unfold probability
  -- Simplify the rational expression
  norm_num
  -- Complete the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l1283_128375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l1283_128348

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_inequality_solution_set (x : ℝ) :
  (floor x)^2 - 5*(floor x) - 36 ≤ 0 ↔ -4 ≤ x ∧ x < 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l1283_128348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1283_128385

/-- Given a parabola and a hyperbola with specific properties, prove the equation of the hyperbola. -/
theorem hyperbola_equation :
  ∀ (F F₁ : ℝ × ℝ) (a b c : ℝ),
  let M := {(x, y) : ℝ × ℝ | y^2 = 12*x}
  let C := {(x, y) : ℝ × ℝ | y^2/a^2 - x^2/b^2 = 1}
  F = (3, 0) →
  F₁ = (0, c) →
  a > 0 →
  b > 0 →
  (∀ (x y : ℝ), y / x = a / b ∨ y / x = -a / b → 
    (x - 3)^2 + y^2 = (3*Real.sqrt 10/4)^2) →
  (∀ (P : ℝ × ℝ), P ∈ M → 
    dist P F₁ + dist P (-3, P.2) ≥ 5) →
  a^2 = 10 ∧ b^2 = 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1283_128385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_B_fastest_l1283_128345

-- Define the athletes' data
noncomputable def athlete_A_distance : ℝ := 400
noncomputable def athlete_A_time : ℝ := 56
noncomputable def athlete_B_distance : ℝ := 600
noncomputable def athlete_B_time : ℝ := 80
noncomputable def athlete_C_distance : ℝ := 800
noncomputable def athlete_C_time : ℝ := 112

-- Define the speed calculation function
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Theorem statement
theorem athlete_B_fastest :
  let speed_A := speed athlete_A_distance athlete_A_time
  let speed_B := speed athlete_B_distance athlete_B_time
  let speed_C := speed athlete_C_distance athlete_C_time
  speed_B > speed_A ∧ speed_B > speed_C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_B_fastest_l1283_128345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1283_128323

theorem tangent_line_to_circle (m : ℝ) : 
  (∃ (x y : ℝ), (Real.sqrt 3 * x - y + m = 0) ∧ 
   ((x - 1)^2 + y^2 = 3) ∧
   (∀ (x' y' : ℝ), (Real.sqrt 3 * x' - y' + m = 0) → ((x' - 1)^2 + y'^2 ≥ 3))) 
  ↔ (m = Real.sqrt 3 ∨ m = -3 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1283_128323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_difference_l1283_128327

noncomputable def cost_price : ℝ := 200
noncomputable def selling_price_1 : ℝ := 350
noncomputable def selling_price_2 : ℝ := 340

noncomputable def profit_percentage (selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

theorem profit_percentage_difference : 
  profit_percentage selling_price_1 - profit_percentage selling_price_2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_difference_l1283_128327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_side_a_value_l1283_128335

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Side lengths
variable (BD : ℝ) -- Length of angle bisector

-- Define the conditions
axiom triangle_condition : 2 * a * Real.cos C - c = 2 * b
axiom angle_sum : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ A + B + C = Real.pi
axiom side_positive : a > 0 ∧ b > 0 ∧ c > 0

-- Part I
theorem angle_A_value : 2 * a * Real.cos C - c = 2 * b → A = 2 * Real.pi / 3 := by
  sorry

-- Part II
theorem side_a_value (h1 : c = Real.sqrt 2) (h2 : BD = Real.sqrt 3) :
  2 * a * Real.cos C - c = 2 * b → a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_side_a_value_l1283_128335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adam_grace_separation_l1283_128314

/-- The speed of Adam in miles per hour -/
noncomputable def adam_speed : ℝ := 10

/-- The speed of Grace in miles per hour -/
noncomputable def grace_speed : ℝ := 12

/-- The distance between Adam and Grace after time t -/
noncomputable def distance (t : ℝ) : ℝ := 
  Real.sqrt ((grace_speed * t)^2 + (adam_speed * t / Real.sqrt 2)^2)

/-- The time when Adam and Grace are 100 miles apart -/
noncomputable def separation_time : ℝ := 100 / Real.sqrt 194

theorem adam_grace_separation :
  distance separation_time = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adam_grace_separation_l1283_128314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_profit_percentage_l1283_128328

/-- Profit percentage calculation -/
noncomputable def profit_percentage (cost : ℝ) (sell : ℝ) : ℝ :=
  (sell - cost) / cost * 100

/-- Theorem for bicycle profit percentage calculation -/
theorem bicycle_profit_percentage 
  (a_cost : ℝ) 
  (a_profit_percent : ℝ) 
  (final_price : ℝ) 
  (h1 : a_cost = 120)
  (h2 : a_profit_percent = 25)
  (h3 : final_price = 225) :
  profit_percentage (a_cost * (1 + a_profit_percent / 100)) final_price = 50 := by
  sorry

#check bicycle_profit_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_profit_percentage_l1283_128328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_l1283_128362

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^4 - 10*x^3 + 35*x^2 - 51*x + 26

-- Define the roots of P(x)
noncomputable def roots : Finset ℝ := sorry

-- Helper definition for the area of a quadrilateral
def is_area_of_quadrilateral (A : ℝ) (sides : Finset ℝ) : Prop := sorry

-- Theorem statement
theorem max_area_quadrilateral : 
  (∀ r ∈ roots, r > 0) →  -- Ensure all roots are positive
  (roots.card = 4) →      -- Ensure there are exactly 4 roots
  (∃ a b c d : ℝ, {a, b, c, d} = roots) →  -- The roots form a quadrilateral
  (∃ A : ℝ, A = Real.sqrt 224.5 ∧ 
    ∀ A' : ℝ, is_area_of_quadrilateral A' roots → A' ≤ A) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_l1283_128362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l1283_128367

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3)

theorem graph_translation (x : ℝ) : 
  f (x + Real.pi / 6) = Real.cos (2 * x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l1283_128367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1283_128378

/-- An arithmetic sequence with first term a and common difference d -/
noncomputable def arithmeticSequence (a d : ℝ) : ℕ → ℝ := λ n => a + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_ratio (a d : ℝ) :
  let an := arithmeticSequence a d
  let Sn := arithmeticSum a d
  (an 8 = 2 * an 3) → (Sn 15 / Sn 5 = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1283_128378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_relationship_l1283_128336

/-- Inverse proportion function -/
noncomputable def inverse_prop (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_prop_relationship (k y₁ y₂ y₃ : ℝ) :
  k < 0 →
  inverse_prop k (-4) = y₁ →
  inverse_prop k 2 = y₂ →
  inverse_prop k 3 = y₃ →
  y₁ > y₃ ∧ y₃ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_relationship_l1283_128336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_sign_white_area_l1283_128326

/-- The area of the white portion of a rectangular sign with "FOOD" painted on it -/
theorem food_sign_white_area 
  (sign_width : ℝ) (sign_height : ℝ) 
  (f_area : ℝ) (o_area : ℝ) (d_area : ℝ)
  (stroke_width : ℝ) : 
  sign_width = 18 ∧ 
  sign_height = 8 ∧
  stroke_width = 1 ∧
  f_area = 6 * stroke_width + 2 * (3 * stroke_width) ∧
  o_area = 2 * (5 * stroke_width + 2 * (3 * stroke_width)) ∧
  d_area = 6 * stroke_width + 5 * stroke_width + (Real.pi / 2) * stroke_width^2 →
  abs ((sign_width * sign_height) - (f_area + o_area + d_area) - 88.43) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_sign_white_area_l1283_128326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_condition_l1283_128330

/-- A point on a parabola -/
structure ParabolaPoint (p : ℝ) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2*p*x

/-- The theorem statement -/
theorem parabola_intersection_condition {p : ℝ} (hp : p > 0) 
  (A B : ParabolaPoint p) (c : ℝ) :
  (∃ (m : ℝ), ∀ (t : ℝ), 
    t * A.x + (1 - t) * B.x = m ∧ 
    t * A.y + (1 - t) * B.y = 0) ↔ 
  A.y * B.y = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_condition_l1283_128330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_l1283_128340

/-- Proves that a car traveling from A to B spends 1 hour on the regular road given the specified conditions -/
theorem car_travel_time (total_time regular_time regular_speed highway_speed : Real) :
  total_time = 2.2 ∧ 
  regular_speed = 60 ∧ 
  highway_speed = 100 ∧ 
  (highway_speed * (total_time - regular_time) = 2 * regular_speed * regular_time) →
  regular_time = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_l1283_128340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omar_coffee_remaining_l1283_128310

/-- Calculates the remaining coffee amount after Omar's consumption pattern --/
noncomputable def remaining_coffee (initial : ℝ) (espresso : ℝ) : ℝ :=
  let after_work := initial * (1 - 1/4)
  let after_office := after_work * (1 - 1/3)
  let after_espresso := after_office + espresso
  let after_lunch := after_espresso * (1 - 0.75)
  after_lunch * (1 - 0.6)

/-- Theorem stating that the remaining coffee is 0.85 ounces --/
theorem omar_coffee_remaining :
  remaining_coffee 12 2.5 = 0.85 := by
  -- Unfold the definition of remaining_coffee
  unfold remaining_coffee
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omar_coffee_remaining_l1283_128310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_bound_l1283_128321

/-- The function f(x) defined as (1-x) / (1+ax^2) where a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (1 + a * x^2)

/-- Theorem stating that there exists a bound m for the function f -/
theorem exists_bound (a : ℝ) (ha : a > 0) : 
  ∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, -m ≤ f a x ∧ f a x ≤ m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_bound_l1283_128321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1283_128363

theorem sufficient_not_necessary :
  (∀ x : ℝ, x^2 - 2*x < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ x^2 - 2*x ≥ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1283_128363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_six_times_30_l1283_128394

-- Define the function r
noncomputable def r (θ : ℝ) : ℝ := 1 / (1 + θ)

-- Define the composition of r with itself n times
noncomputable def r_compose : ℕ → ℝ → ℝ
| 0 => id
| n + 1 => r ∘ (r_compose n)

-- State the theorem
theorem r_six_times_30 : r_compose 6 30 = 158 / 253 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_six_times_30_l1283_128394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gerald_speed_l1283_128371

-- Define the track length in miles
noncomputable def track_length : ℚ := 1/4

-- Define the number of laps completed by person A
def laps_A : ℕ := 12

-- Define the time taken by person A in hours
noncomputable def time_A : ℚ := 1/2

-- Define the speed ratio of person B to person A
noncomputable def speed_ratio : ℚ := 1/2

-- Theorem statement
theorem gerald_speed :
  let distance_A := track_length * (laps_A : ℚ)
  let speed_A := distance_A / time_A
  let speed_B := speed_A * speed_ratio
  speed_B = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gerald_speed_l1283_128371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1283_128309

def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ := fun n ↦ a₁ * r^(n - 1)

theorem geometric_sequence_sum (a : ℝ) :
  let s := geometric_sequence (1/2) (a - 1/2)
  (∑' n, s n) = a →
  a = 1 := by
  intro h
  sorry

#check geometric_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1283_128309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1283_128347

theorem equation_solution (x : ℝ) (h : x > 0) :
  (3 * Real.sqrt x + 3 * x^(-(1/2 : ℝ)) = 8) ↔ 
  (x = ((8 + Real.sqrt 28) / 6)^2 ∨ x = ((8 - Real.sqrt 28) / 6)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1283_128347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_tan_result_when_f_equals_two_l1283_128380

-- Define the function f
noncomputable def f (α : ℝ) : ℝ :=
  (1 + Real.sin (-α) + Real.sin (2 * Real.pi - α)^2 - Real.sin (Real.pi / 2 + α)^2) /
  (2 * Real.sin (Real.pi - α) * Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi + α))

-- Theorem 1: f(α) = tan(α)
theorem f_equals_tan (α : ℝ) : f α = Real.tan α := by
  sorry

-- Theorem 2: If f(α) = 2, then (sin α + cos α)cos α = 3/5
theorem result_when_f_equals_two (α : ℝ) (h : f α = 2) :
  (Real.sin α + Real.cos α) * Real.cos α = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_tan_result_when_f_equals_two_l1283_128380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_interval_iff_a_between_zero_and_one_l1283_128396

/-- Definition of a harmonious interval for a function -/
noncomputable def is_harmonious_interval (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  (∀ x y, m ≤ x ∧ x < y ∧ y ≤ n → f x < f y) ∧
  (∀ y, m ≤ y ∧ y ≤ n → ∃ x, m ≤ x ∧ x ≤ n ∧ f x = y)

/-- The function f(x) = (a+1)/a - 1/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) / a - 1 / x

/-- Main theorem: f(x) has a harmonious interval iff 0 < a < 1 -/
theorem harmonious_interval_iff_a_between_zero_and_one (a : ℝ) :
  (∃ m n : ℝ, is_harmonious_interval (f a) m n) ↔ 0 < a ∧ a < 1 := by
  sorry

/-- Auxiliary lemma: f(x) is monotonic increasing on (0, +∞) -/
lemma f_monotonic_positive (a : ℝ) (h : a > 0) :
  ∀ x y, 0 < x ∧ x < y → f a x < f a y := by
  sorry

/-- Auxiliary lemma: f(x) is monotonic increasing on (-∞, 0) -/
lemma f_monotonic_negative (a : ℝ) (h : a > 0) :
  ∀ x y, x < y ∧ y < 0 → f a x < f a y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_interval_iff_a_between_zero_and_one_l1283_128396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_twelve_gon_l1283_128344

/-- A convex 12-gon with equal angles and specific side lengths -/
structure RegularTwelveGon where
  -- The polygon is convex
  isConvex : Bool
  -- The polygon has 12 sides
  numSides : Nat
  numSides_eq : numSides = 12
  -- All angles are equal
  equalAngles : Bool
  -- Ten sides have length 1
  tenSidesLength : ℝ
  tenSidesLength_eq : tenSidesLength = 1
  -- Two sides have length 2
  twoSidesLength : ℝ
  twoSidesLength_eq : twoSidesLength = 2

/-- The area of the specific 12-gon -/
noncomputable def areaOfTwelveGon (p : RegularTwelveGon) : ℝ :=
  8 + 4 * Real.sqrt 3

/-- Theorem stating the area of the specific 12-gon -/
theorem area_of_specific_twelve_gon (p : RegularTwelveGon) :
  areaOfTwelveGon p = 8 + 4 * Real.sqrt 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_twelve_gon_l1283_128344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1283_128334

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_inequality : f (Real.exp 1) > f 3 ∧ f 3 > f 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1283_128334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_match_odd_numbers_l1283_128354

/-- Represents a tennis tournament between n girls and n boys -/
structure TennisTournament (n : ℕ) where
  /-- n is odd -/
  n_odd : Odd n
  /-- Each girl plays exactly one match with each boy -/
  all_matches_played : ∀ (g b : Fin n), ∃! m : ℕ, True

/-- The last match in the tournament -/
noncomputable def last_match (n : ℕ) (t : TennisTournament n) : Fin n × Fin n :=
  sorry

/-- Theorem: In a tennis tournament where n is odd and each girl plays exactly one match with each boy,
    the girl and boy in the last match have odd numbers -/
theorem last_match_odd_numbers {n : ℕ} (t : TennisTournament n) :
  Odd ((last_match n t).1.val) ∧ Odd ((last_match n t).2.val) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_match_odd_numbers_l1283_128354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_five_simplification_l1283_128370

theorem power_five_simplification (x y z : ℝ) (h : x * y * z = 1) :
  (5 : ℝ) ^ ((x + y + z) ^ 2) / (5 : ℝ) ^ (x - y + 2 * z) = 5 ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_five_simplification_l1283_128370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1283_128320

def mySequence (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n : ℕ, n > 0 → n * a n = (n + 1) * a (n + 1)

theorem sequence_formula (a : ℕ → ℚ) (h : mySequence a) :
  ∀ n : ℕ, n > 0 → a n = 3 / n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1283_128320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_after_detaching_l1283_128342

/-- Represents a train with boggies -/
structure Train where
  num_boggies : ℕ
  boggy_length : ℚ
  crossing_time : ℚ

/-- Calculates the time taken for a train to cross a telegraph post after detaching one boggy -/
def time_after_detaching (t : Train) : ℚ :=
  let initial_length := t.num_boggies * t.boggy_length
  let initial_speed := initial_length / t.crossing_time
  let new_length := (t.num_boggies - 1) * t.boggy_length
  new_length / initial_speed

/-- Theorem stating that for a train with 12 boggies, each 15 meters long, 
    initially crossing in 9 seconds, the time taken after detaching one boggy is 8.25 seconds -/
theorem train_crossing_time_after_detaching :
  let initial_train := Train.mk 12 (15 : ℚ) (9 : ℚ)
  time_after_detaching initial_train = (33 : ℚ) / 4 := by
  sorry

#eval time_after_detaching (Train.mk 12 15 9)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_after_detaching_l1283_128342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_shop_dogs_l1283_128350

/-- The ratio of dogs in the pet shop -/
def dog_ratio : ℝ := 4.5

/-- The ratio of bunnies in the pet shop -/
def bunny_ratio : ℝ := 9.8

/-- The ratio of parrots in the pet shop -/
def parrot_ratio : ℝ := 12.2

/-- The total number of dogs, bunnies, and parrots in the pet shop -/
def total_animals : ℕ := 815

/-- The number of dogs in the pet shop -/
def num_dogs : ℕ := 138

theorem pet_shop_dogs :
  let total_ratio := dog_ratio + bunny_ratio + parrot_ratio
  let animals_per_ratio := (total_animals : ℝ) / total_ratio
  Int.floor (dog_ratio * animals_per_ratio) = num_dogs := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_shop_dogs_l1283_128350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1283_128302

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (a / (2 * b)) + (2 * b / c) + (c / a) ≥ 3 ∧
  ((a / (2 * b)) + (2 * b / c) + (c / a) = 3 ↔ a = (2 : ℝ) ^ (1/3) ∧ b = (2 : ℝ) ^ (1/3) ∧ c = (2 : ℝ) ^ (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1283_128302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_and_planes_relations_l1283_128393

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Theorem statement
theorem lines_and_planes_relations 
  (m n : Line) (α β : Plane) (l : Line) :
  (perpendicular_plane_plane α β → perpendicular_line_plane m β → ¬in_plane m α → parallel_line_plane m α) ∧
  (perpendicular_plane_plane α β → intersect α β l → parallel_line_plane m α → perpendicular_line_line m l → perpendicular_line_plane m β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_and_planes_relations_l1283_128393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1283_128358

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |x| * (10^x - 10^(-x))

-- State the theorem
theorem inequality_solution_set :
  ∀ x : ℝ, f (1 - 2*x) + f 3 > 0 ↔ x < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1283_128358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fitness_center_cost_effectiveness_l1283_128368

/-- Cost function for Center A -/
def f (x : ℝ) : ℝ := 5 * x

/-- Cost function for Center B -/
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 30 then 90 else 30 + 2 * x

/-- The range of exercise hours -/
def valid_hours (x : ℝ) : Prop := 15 ≤ x ∧ x ≤ 40

theorem fitness_center_cost_effectiveness :
  ∀ x, valid_hours x →
    (x < 18 → f x < g x) ∧
    (x = 18 → f x = g x) ∧
    (x > 18 → f x > g x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fitness_center_cost_effectiveness_l1283_128368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l1283_128373

/-- The minimum distance between a fixed point A(0, a) and a moving point M on a curve -/
noncomputable def min_distance (a : ℝ) : ℝ :=
  if 1 < a ∧ a ≤ 4 then a - 1 else Real.sqrt (2 * a + 1)

/-- The curve on which point M moves -/
noncomputable def curve (x : ℝ) : ℝ := |1/2 * x^2 - 1|

theorem min_distance_theorem (a : ℝ) (h : a > 1) :
  ∀ x y : ℝ, y = curve x →
  (x^2 + (y - a)^2) ≥ (min_distance a)^2 := by
  sorry

#check min_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l1283_128373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_specific_primes_l1283_128317

-- Define the set of one-digit primes
def OneDigitPrimes : Set Nat := {p | Nat.Prime p ∧ p < 10}

-- Define the set of two-digit primes
def TwoDigitPrimes : Set Nat := {p | Nat.Prime p ∧ p ≥ 10 ∧ p < 100}

-- State the theorem
theorem product_of_specific_primes :
  ∃ (p₁ p₂ p₃ : Nat),
    p₁ ∈ OneDigitPrimes ∧
    p₂ ∈ OneDigitPrimes ∧
    p₃ ∈ TwoDigitPrimes ∧
    (∀ q, q ∈ OneDigitPrimes → p₁ ≤ q) ∧
    (∃! r, r ∈ OneDigitPrimes ∧ r > p₂) ∧
    (∀ s, s ∈ TwoDigitPrimes → p₃ ≤ s) ∧
    p₁ * p₂ * p₃ = 110 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_specific_primes_l1283_128317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_l1283_128389

/-- Represents the weight in grams used by the shopkeeper --/
noncomputable def actual_weight : ℚ := 750

/-- Represents the standard weight of a kilogram in grams --/
noncomputable def standard_weight : ℚ := 1000

/-- Calculates the gain percentage of the shopkeeper --/
noncomputable def gain_percentage (actual : ℚ) (standard : ℚ) : ℚ :=
  (standard - actual) / actual

/-- 
Theorem: If a shopkeeper uses a 750 grams weight instead of 1000 grams (1 kilogram) 
while selling at apparent cost price, their gain percentage is 1/3.
-/
theorem shopkeeper_gain : 
  gain_percentage actual_weight standard_weight = 1/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_l1283_128389
