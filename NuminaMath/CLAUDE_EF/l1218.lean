import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l1218_121861

/-- Calculates the distance to a place given rowing conditions -/
noncomputable def distance_to_place (still_water_speed : ℝ) (current_to_place : ℝ) (current_from_place : ℝ) (total_time : ℝ) : ℝ :=
  let speed_to_place := still_water_speed - current_to_place
  let speed_from_place := still_water_speed + current_from_place
  (speed_to_place * speed_from_place * total_time) / (speed_to_place + speed_from_place)

/-- Theorem stating the distance to the place under given conditions -/
theorem distance_calculation :
  distance_to_place 5 1 2 1 = 28 / 11 := by
  -- Unfold the definition of distance_to_place
  unfold distance_to_place
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l1218_121861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roberto_stairs_l1218_121884

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a + (n - 1 : ℝ) * d)

/-- Roberto's stair climbing problem -/
theorem roberto_stairs :
  let a : ℝ := 15  -- time for the first flight
  let d : ℝ := 10  -- additional time for each successive flight
  let n : ℕ := 6   -- number of flights
  arithmetic_sum a d n = 240 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roberto_stairs_l1218_121884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_approx_65_minutes_l1218_121818

/-- Represents a cyclist with their speeds on different terrains -/
structure Cyclist where
  flatSpeed : ℝ
  downhillSpeed : ℝ
  uphillSpeed : ℝ

/-- Represents the journey with distances for each terrain type -/
structure Journey where
  flatDistance : ℝ
  downhillDistance : ℝ
  uphillDistance : ℝ

/-- Calculates the time taken for a cyclist to complete a journey -/
noncomputable def timeTaken (c : Cyclist) (j : Journey) : ℝ :=
  j.flatDistance / c.flatSpeed +
  j.downhillDistance / c.downhillSpeed +
  j.uphillDistance / c.uphillSpeed

/-- The main theorem stating the time difference between Minnie and Penny -/
theorem time_difference_approx_65_minutes 
  (minnie penny : Cyclist) 
  (journey : Journey) :
  minnie.flatSpeed = 20 →
  minnie.downhillSpeed = 30 →
  minnie.uphillSpeed = 5 →
  penny.flatSpeed = 30 →
  penny.downhillSpeed = 40 →
  penny.uphillSpeed = 10 →
  journey.flatDistance = 20 →
  journey.downhillDistance = 15 →
  journey.uphillDistance = 10 →
  ∃ (diff : ℝ), abs (diff - 65/60) < 0.01 ∧ 
  diff = timeTaken minnie journey - timeTaken penny journey := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_approx_65_minutes_l1218_121818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_pile_volume_calculation_l1218_121802

/-- The volume of a conical pile of sand -/
noncomputable def sand_pile_volume (diameter : ℝ) (height_ratio : ℝ) : ℝ :=
  let radius := diameter / 2
  let height := height_ratio * diameter
  (1 / 3) * Real.pi * radius^2 * height

/-- Theorem: The volume of a conical pile of sand with diameter 10 feet
    and height 60% of the diameter is 50π cubic feet -/
theorem sand_pile_volume_calculation :
  sand_pile_volume 10 0.6 = 50 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_pile_volume_calculation_l1218_121802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_v_at_points_l1218_121816

/-- The function v(x) as defined in the problem -/
noncomputable def v (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x / 2) - x

/-- Theorem stating the sum of v at specific points equals -2 -/
theorem sum_of_v_at_points : 
  v (-3.14) + v (-1.57) + v 1.57 + v 3.14 = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_v_at_points_l1218_121816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l1218_121801

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (2 * x^2 - 3) / x

-- State the theorem
theorem f_increasing :
  (∀ x₁ x₂, x₁ < x₂ → x₁ < 0 → x₂ < 0 → f x₁ < f x₂) ∧
  (∀ x₁ x₂, x₁ < x₂ → 0 < x₁ → 0 < x₂ → f x₁ < f x₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l1218_121801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_zeros_of_g_zeros_conditions_l1218_121891

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x + (1/2) * x^2 - 4*x + 1

noncomputable def g (x m : ℝ) : ℝ := f x - m

-- Statement for the tangent line equation
theorem tangent_line_at_2 :
  ∃ (y : ℝ → ℝ), ∀ x, x + 2 * y x - 6 * Real.log 2 + 8 = 0 :=
sorry

-- Statement for the number of zeros of g
theorem zeros_of_g (m : ℝ) :
  (∃! x, g x m = 0) ∨
  (∃! x₁ x₂, x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0) ∨
  (∃! x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ g x₁ m = 0 ∧ g x₂ m = 0 ∧ g x₃ m = 0) :=
sorry

-- Conditions for the number of zeros
theorem zeros_conditions (m : ℝ) :
  ((m > -5/2 ∨ m < 3 * Real.log 3 - 13/2) →
    (∃! x, g x m = 0)) ∧
  ((m = -5/2 ∨ m = 3 * Real.log 3 - 13/2) →
    (∃! x₁ x₂, x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0)) ∧
  ((3 * Real.log 3 - 13/2 < m ∧ m < -5/2) →
    (∃! x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ g x₁ m = 0 ∧ g x₂ m = 0 ∧ g x₃ m = 0)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_zeros_of_g_zeros_conditions_l1218_121891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l1218_121824

/-- A quadratic polynomial q(x) satisfying specific conditions -/
noncomputable def q (x : ℝ) : ℝ := -2/3 * x^2 + 1/3 * x + 5

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q (-1) = 4 ∧ q 2 = 3 ∧ q 0 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l1218_121824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l1218_121805

theorem cos_minus_sin_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = 1/8)
  (h2 : π/4 < α)
  (h3 : α < π/2) :
  Real.cos α - Real.sin α = -Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l1218_121805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_skew_median_distance_l1218_121821

/-- A regular tetrahedron with edge length 1 -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_regular : Bool
  edge_length_is_one : edge_length = 1

/-- A median of a face in a regular tetrahedron -/
structure FaceMedian (t : RegularTetrahedron)

/-- The distance between two skew medians -/
def skew_median_distance (t : RegularTetrahedron) (m1 m2 : FaceMedian t) : ℝ :=
  sorry

/-- The theorem stating the minimum distance between skew medians -/
theorem min_skew_median_distance (t : RegularTetrahedron) :
  ∃ (m1 m2 : FaceMedian t), 
    skew_median_distance t m1 m2 = Real.sqrt (1 / 10) ∧
    ∀ (n1 n2 : FaceMedian t), skew_median_distance t n1 n2 ≥ Real.sqrt (1 / 10) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_skew_median_distance_l1218_121821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_l1218_121894

noncomputable section

-- Define the ellipse C and circle D
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def circle_D (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define points A, B, and E
def point_on_ellipse (x y : ℝ) : Prop := ellipse_C x y
def point_on_circle (x y : ℝ) : Prop := circle_D x y

-- Define the distances
def max_distance : ℝ := 5
def min_distance : ℝ := 1

-- Define the slopes
def slope_AB : ℝ := Real.sqrt 2 / 2

-- Define the parallelogram property
def is_parallelogram (xa ya xb yb xe ye : ℝ) : Prop :=
  xe = xa + xb ∧ ye = ya + yb

-- Main theorem
theorem sum_of_slopes (xa ya xb yb xe ye k1 k2 : ℝ) :
  point_on_ellipse xa ya →
  point_on_ellipse xb yb →
  point_on_circle xe ye →
  k1 = ya / xa →
  k2 = yb / xb →
  (yb - ya) / (xb - xa) = slope_AB →
  is_parallelogram xa ya xb yb xe ye →
  k1 + k2 = -4 * Real.sqrt 2 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_l1218_121894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_three_rational_others_l1218_121875

theorem irrational_sqrt_three_rational_others : 
  (Irrational (Real.sqrt 3)) ∧ 
  (¬ Irrational (0 : ℝ)) ∧ 
  (¬ Irrational (-3 : ℝ)) ∧ 
  (¬ Irrational (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_three_rational_others_l1218_121875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_66_degrees_l1218_121859

theorem sin_66_degrees (a : ℝ) (h : Real.sin (12 * π / 180) = a) :
  Real.sin (66 * π / 180) = 1 - 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_66_degrees_l1218_121859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_moles_in_reaction_l1218_121829

/-- Represents the number of moles of a substance -/
def Moles := ℝ

/-- Represents the chemical reaction Fe + H₂SO₄ → FeSO₄ + H₂ -/
structure Reaction where
  iron : Moles
  sulfuric_acid : Moles
  hydrogen : Moles

/-- The reaction is balanced when the moles of reactants and products are in the correct ratio -/
def is_balanced (r : Reaction) : Prop :=
  r.iron = r.sulfuric_acid ∧ r.iron = r.hydrogen

theorem iron_moles_in_reaction (r : Reaction) 
  (h1 : is_balanced r) 
  (h2 : r.sulfuric_acid = (2 : ℝ)) 
  (h3 : r.hydrogen = (2 : ℝ)) : 
  r.iron = (2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_moles_in_reaction_l1218_121829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_statement_l1218_121893

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, x > 0 ∧ (2 : ℝ)^x > (3 : ℝ)^x) ↔ (∀ x : ℝ, x > 0 → (2 : ℝ)^x ≤ (3 : ℝ)^x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_statement_l1218_121893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equal_if_floor_equal_l1218_121841

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_equal_if_floor_equal
  (f g : ℝ → ℝ)
  (hf : is_quadratic f)
  (hg : is_quadratic g)
  (h : ∀ x, floor (f x) = floor (g x)) :
  ∀ x, f x = g x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equal_if_floor_equal_l1218_121841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sets_imply_value_l1218_121817

theorem equal_sets_imply_value (a b : ℝ) : 
  let M : Set ℝ := {1, a + b, a}
  let N : Set ℝ := {0, b / a, b}
  M = N → b^2014 - a^2013 = 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sets_imply_value_l1218_121817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_is_correct_l1218_121845

/-- Conversion factor from grams to pounds -/
noncomputable def gram_to_pound : ℝ := 0.00220462

/-- Conversion factor from ounces to pounds -/
noncomputable def ounce_to_pound : ℝ := 0.0625

/-- Conversion factor from liters to pounds -/
noncomputable def liter_to_pound : ℝ := 2.20462

/-- Weight of brie cheese in ounces -/
noncomputable def brie_weight : ℝ := 8

/-- Weight of bread in pounds -/
noncomputable def bread_weight : ℝ := 1

/-- Weight of tomatoes in pounds -/
noncomputable def tomato_weight : ℝ := 1

/-- Weight of zucchini in pounds -/
noncomputable def zucchini_weight : ℝ := 2

/-- Weight of chicken breasts in pounds -/
noncomputable def chicken_weight : ℝ := 1.5

/-- Weight of raspberries in ounces -/
noncomputable def raspberry_weight : ℝ := 8

/-- Weight of blueberries in ounces -/
noncomputable def blueberry_weight : ℝ := 8

/-- Weight of asparagus in grams -/
noncomputable def asparagus_weight : ℝ := 500

/-- Weight of oranges in grams -/
noncomputable def orange_weight : ℝ := 1000

/-- Volume of olive oil in milliliters -/
noncomputable def olive_oil_volume : ℝ := 750

/-- The total weight of all food items in pounds -/
noncomputable def total_weight : ℝ :=
  brie_weight * ounce_to_pound +
  bread_weight +
  tomato_weight +
  zucchini_weight +
  chicken_weight +
  raspberry_weight * ounce_to_pound +
  blueberry_weight * ounce_to_pound +
  asparagus_weight * gram_to_pound +
  orange_weight * gram_to_pound +
  (olive_oil_volume / 1000) * liter_to_pound

theorem total_weight_is_correct : total_weight = 11.960895 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_is_correct_l1218_121845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_a_bound_l1218_121855

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2/x

theorem f_increasing_and_a_bound (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∀ x : ℝ, 1 < x → f a x < 2*x) →
  a ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_a_bound_l1218_121855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_colony_limit_time_l1218_121850

/-- Represents the number of days it takes for a colony to reach the habitat's limit -/
def days_to_limit (n : ℕ) : ℕ := sorry

/-- The size of a colony on a given day -/
def colony_size (days : ℕ) : ℕ := 2^days

/-- The habitat's limit (maximum capacity) -/
def habitat_limit : ℕ := sorry

theorem single_colony_limit_time :
  (∀ d : ℕ, colony_size d = habitat_limit → days_to_limit 1 = d) →
  (colony_size 19 + colony_size 19 = habitat_limit) →
  days_to_limit 1 = 20 :=
by sorry

#check single_colony_limit_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_colony_limit_time_l1218_121850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_solution_l1218_121877

open Real

theorem system_has_solution (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (Real.sin x₁ + a = b * x₁) ∧ (Real.cos x₁ = b) ∧
    (Real.sin x₂ + a = b * x₂) ∧ (Real.cos x₂ = b) ∧
    (∀ x : ℝ, (Real.sin x + a = b * x ∧ Real.cos x = b) → (x = x₁ ∨ x = x₂))) →
  (∃ x : ℝ, Real.sin x + a = b * x ∧ Real.cos x = b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_solution_l1218_121877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ian_says_591_l1218_121872

/-- Represents a student in the counting game -/
structure Student where
  name : String
  skip_pattern : ℕ → Bool

/-- The counting game with 9 students -/
structure CountingGame where
  students : List Student
  max_number : ℕ
  h_students : students.length = 9
  h_max_number : max_number = 1024

/-- Determines if a number should be skipped by a student based on their pattern -/
def should_skip (s : Student) (n : ℕ) : Bool :=
  s.skip_pattern n

/-- Finds the number spoken by the last student -/
def last_student_number (game : CountingGame) : ℕ :=
  sorry

/-- The main theorem: The last student (Ian) says 591 -/
theorem ian_says_591 (game : CountingGame) : last_student_number game = 591 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ian_says_591_l1218_121872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_available_seats_l1218_121882

/-- Calculates the number of available seats in an auditorium --/
theorem available_seats (total : ℕ) (taken_fraction : ℚ) (broken_fraction : ℚ) 
  (h1 : total = 500)
  (h2 : taken_fraction = 2/5)
  (h3 : broken_fraction = 1/10) : 
  total - (total * taken_fraction).floor - (total * broken_fraction).floor = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_available_seats_l1218_121882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_g_eq_3_has_three_solutions_l1218_121839

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then -x + 2 else 3*x - 7

-- State the theorem
theorem g_g_eq_3_has_three_solutions :
  ∃ (a b c : ℝ), (∀ x : ℝ, g (g x) = 3 ↔ x = a ∨ x = b ∨ x = c) ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_g_eq_3_has_three_solutions_l1218_121839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_classification_l1218_121803

noncomputable def given_numbers : List ℚ := [-1/3, 22/7, -1, -7/10, 11, -25, 0, 85/100]

def is_positive (x : ℚ) : Prop := x > 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_non_negative (x : ℚ) : Prop := x ≥ 0
def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ ¬(is_integer x)

def positive_set : Set ℚ := {x | is_positive x}
def integer_set : Set ℚ := {x | is_integer x}
def non_negative_set : Set ℚ := {x | is_non_negative x}
def negative_fraction_set : Set ℚ := {x | is_negative_fraction x}

theorem number_classification :
  (22/7 ∈ positive_set ∧ 11 ∈ positive_set ∧ 85/100 ∈ positive_set) ∧
  (-1 ∈ integer_set ∧ 11 ∈ integer_set ∧ -25 ∈ integer_set ∧ 0 ∈ integer_set) ∧
  (22/7 ∈ non_negative_set ∧ 11 ∈ non_negative_set ∧ 0 ∈ non_negative_set ∧ 85/100 ∈ non_negative_set) ∧
  (-1/3 ∈ negative_fraction_set ∧ -7/10 ∈ negative_fraction_set) :=
by sorry

#check number_classification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_classification_l1218_121803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_focus_l1218_121834

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the focus of the parabola
noncomputable def focus : ℝ × ℝ := (0, 1/8)

-- Define the distance between a point on the parabola and the focus
noncomputable def distance_to_focus (x : ℝ) : ℝ :=
  let y := parabola x
  Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2)

-- Theorem statement
theorem min_distance_to_focus :
  ∃ (min_dist : ℝ), min_dist = 1/8 ∧
  ∀ (x : ℝ), distance_to_focus x ≥ min_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_focus_l1218_121834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l1218_121843

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det (B^2 - 3 • B) = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l1218_121843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_triangle_l1218_121815

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  perimeter : ℝ

/-- The trajectory of point A -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 5 = 1 ∧ y ≠ 0

/-- Theorem stating the trajectory of point A given the triangle properties -/
theorem trajectory_of_triangle (ABC : Triangle) 
  (hB : ABC.B = (-2, 0)) 
  (hC : ABC.C = (2, 0)) 
  (hP : ABC.perimeter = 10) : 
  trajectory ABC.A.1 ABC.A.2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_triangle_l1218_121815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_60_degrees_l1218_121878

/-- Represents a triangle with its three angles -/
structure Triangle where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- Generates the next triangle in the sequence -/
noncomputable def nextTriangle (t : Triangle) : Triangle :=
  { angleA := (t.angleB + t.angleC) / 2,
    angleB := (t.angleA + t.angleC) / 2,
    angleC := (t.angleA + t.angleB) / 2 }

/-- Checks if all angles in a triangle are less than or equal to 60° -/
def allAnglesLEQ60 (t : Triangle) : Prop :=
  t.angleA ≤ 60 ∧ t.angleB ≤ 60 ∧ t.angleC ≤ 60

/-- The initial triangle A₀B₀C₀ -/
def A₀B₀C₀ : Triangle :=
  { angleA := 50,
    angleB := 65,
    angleC := 65 }

/-- Generates the nth triangle in the sequence -/
noncomputable def nthTriangle : ℕ → Triangle
  | 0 => A₀B₀C₀
  | n+1 => nextTriangle (nthTriangle n)

/-- The main theorem to be proved -/
theorem smallest_n_for_60_degrees :
  ∃ n : ℕ, (∀ k < n, ¬(allAnglesLEQ60 (nthTriangle k))) ∧ 
           allAnglesLEQ60 (nthTriangle n) ∧ 
           n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_60_degrees_l1218_121878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1218_121868

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 17)

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Iic (Real.log 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1218_121868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_unexpressible_l1218_121853

def expressible (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, n * (2^c - 2^d) = 2^a - 2^b

theorem smallest_unexpressible : 
  (∀ m : ℕ, m < 11 → expressible m) ∧ 
  ¬expressible 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_unexpressible_l1218_121853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subtract_add_subtract_equals_negative_twentyeight_l1218_121895

theorem subtract_add_subtract_equals_negative_twentyeight :
  -15 - 21 - (-8) = -28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subtract_add_subtract_equals_negative_twentyeight_l1218_121895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_phi_value_l1218_121889

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)
noncomputable def g (x φ : ℝ) : ℝ := 2 * Real.cos (2 * x - 2 * φ)

-- State the theorem
theorem translation_phi_value (φ : ℝ) :
  (0 < φ ∧ φ < Real.pi / 2) →
  (∃ x₁ x₂ : ℝ, |f x₁ - g x₂ φ| = 4 ∧ |x₁ - x₂| = Real.pi / 6) →
  φ = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_phi_value_l1218_121889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_factorial_sum_l1218_121842

theorem greatest_prime_factor_of_factorial_sum : 
  (Nat.factorial 15 + Nat.factorial 17).factors.maximum? = some 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_factorial_sum_l1218_121842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_theorem_l1218_121806

-- Define the angle α
variable (α : Real)

-- Define the point through which the terminal side of α passes
def terminal_point (a : Real) : Real × Real := (3*a - 9, a + 2)

-- Define the theorem
theorem angle_range_theorem (a : Real) :
  (∃ α, terminal_point a = (3*a - 9, a + 2) ∧ Real.cos α ≤ 0 ∧ Real.sin α > 0) →
  a ∈ Set.Ioc (-2) 3 :=
by
  sorry

#check angle_range_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_theorem_l1218_121806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_measure_l1218_121883

-- Define the triangle side lengths as functions of v
noncomputable def side1 (v : ℝ) : ℝ := Real.sqrt (3 * v - 2)
noncomputable def side2 (v : ℝ) : ℝ := Real.sqrt (3 * v + 2)
noncomputable def side3 (v : ℝ) : ℝ := 2 * Real.sqrt v

-- Define the condition for v
def v_condition (v : ℝ) : Prop := v > 2/3

-- Define the largest angle measure
noncomputable def largest_angle (v : ℝ) : ℝ := 
  Real.arccos (1 / Real.sqrt ((3 * v - 2) * (3 * v + 2)))

-- Theorem statement
theorem largest_angle_measure (v : ℝ) (h : v_condition v) :
  largest_angle v = Real.arccos (1 / Real.sqrt ((3 * v - 2) * (3 * v + 2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_measure_l1218_121883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_1000_l1218_121852

/-- Calculate simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculate compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Theorem stating that the principal is 1000 given the conditions -/
theorem principal_is_1000 :
  ∃ (principal : ℝ),
    let rate : ℝ := 10
    let time : ℝ := 4
    compoundInterest principal rate time - simpleInterest principal rate time = 64.10 ∧
    principal = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_1000_l1218_121852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l1218_121808

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / (Real.exp x)

-- State the theorem
theorem tangent_line_at_zero :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ x + y - 1 = 0) ∧
    (deriv f 0 = m) ∧
    (f 0 = b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l1218_121808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_tangent_problem_l1218_121830

theorem circle_chord_tangent_problem (R k : ℝ) (h : R > 0) :
  let x := λ (sign : Bool) ↦ (k + 4*R + (if sign then 1 else -1) * Real.sqrt ((k + 4*R)^2 - 5*k^2)) / 5
  let solution_exists := k ≤ R * (Real.sqrt 5 + 1)
  let one_solution := k ≤ 2*R
  let two_solutions := 2*R < k ∧ k < R * (Real.sqrt 5 + 1)
  (∀ sign, 5 * x sign^2 - 2*(k + 4*R) * x sign + k^2 = 0) ∧
  solution_exists ∧
  (one_solution → ∃! sign, 0 < x sign ∧ x sign < k) ∧
  (two_solutions → ∃ sign₁ sign₂, sign₁ ≠ sign₂ ∧ 0 < x sign₁ ∧ x sign₁ < k ∧ 0 < x sign₂ ∧ x sign₂ < k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_tangent_problem_l1218_121830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_without_zeros_in_product_with_power_of_two_l1218_121858

/-- Given a natural number, return its decimal digits as a list of natural numbers. -/
def decimal_digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: decimal_digits (n / 10)

theorem exists_number_without_zeros_in_product_with_power_of_two :
  ∃ q : ℕ, ∀ d : ℕ, d ∈ decimal_digits (q * 2^1000) → d ≠ 0 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_without_zeros_in_product_with_power_of_two_l1218_121858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_polar_sum_l1218_121898

/-- The ellipse x²/4 + y² = 1 in polar coordinates -/
def Ellipse (ρ θ : ℝ) : Prop :=
  (ρ * Real.cos θ)^2 / 4 + (ρ * Real.sin θ)^2 = 1

theorem ellipse_polar_sum (ρ₁ ρ₂ θ : ℝ) :
  Ellipse ρ₁ θ → Ellipse ρ₂ (θ + π/2) → 1/ρ₁^2 + 1/ρ₂^2 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_polar_sum_l1218_121898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_20_miles_l1218_121844

/-- The distance Bob walked when he met Yolanda -/
noncomputable def bobsDistance (totalDistance : ℝ) (yolandaRate : ℝ) (bobRate : ℝ) (headStart : ℝ) : ℝ :=
  let meetingTime := (totalDistance - yolandaRate * headStart) / (yolandaRate + bobRate)
  bobRate * meetingTime

/-- Theorem stating that Bob walked 20 miles when he met Yolanda -/
theorem bob_walked_20_miles :
  let totalDistance : ℝ := 31
  let yolandaRate : ℝ := 1
  let bobRate : ℝ := 2
  let headStart : ℝ := 1
  bobsDistance totalDistance yolandaRate bobRate headStart = 20 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_20_miles_l1218_121844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1218_121856

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x)^2 - 1

-- Define the interval
def interval : Set ℝ := { x | -Real.pi/6 ≤ x ∧ x ≤ Real.pi/4 }

-- Theorem statement
theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  (∃ (max min : ℝ), max = 2 ∧ min = -1 ∧
    (∀ (x : ℝ), x ∈ interval → min ≤ f x ∧ f x ≤ max) ∧
    (∃ (x1 x2 : ℝ), x1 ∈ interval ∧ x2 ∈ interval ∧ f x1 = max ∧ f x2 = min)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1218_121856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_price_l1218_121879

/-- The price of an adult ticket -/
noncomputable def adult_price : ℝ := 3.75

/-- The price of a child ticket -/
noncomputable def child_price : ℝ := (2/3) * adult_price

/-- The total price for 6 adult tickets and 5 child tickets -/
noncomputable def total_price_6a5c : ℝ := 6 * adult_price + 5 * child_price

/-- The condition that the total price for 6 adult tickets and 5 child tickets is $35 -/
axiom total_price_condition : total_price_6a5c = 35

/-- The theorem to prove -/
theorem concert_ticket_price : 
  9 * adult_price + 7 * child_price = 51.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_price_l1218_121879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_faces_theorem_l1218_121840

/-- Represents a configuration of painted cubes -/
structure CubeConfiguration where
  totalCubes : Nat
  facesPerCube : Nat
  blackFaces : Nat
  largerCubeSurfaceFaces : Nat

/-- Checks if a given configuration is valid -/
def isValidConfiguration (config : CubeConfiguration) : Prop :=
  config.totalCubes = 8 ∧
  config.facesPerCube = 6 ∧
  config.largerCubeSurfaceFaces = 24 ∧
  config.blackFaces ≤ config.totalCubes * config.facesPerCube

/-- Checks if a configuration can be assembled into a larger cube with equal black and white faces -/
def canAssembleBalanced (config : CubeConfiguration) : Prop :=
  ∃ (blackOnSurface : Nat),
    blackOnSurface = config.largerCubeSurfaceFaces / 2 ∧
    blackOnSurface ≤ config.blackFaces ∧
    (config.largerCubeSurfaceFaces - blackOnSurface) ≤ (config.totalCubes * config.facesPerCube - config.blackFaces)

/-- The main theorem stating the possible values for black faces -/
theorem black_faces_theorem :
  ∀ (n : Nat),
    (∃ (config : CubeConfiguration),
      isValidConfiguration config ∧
      config.blackFaces = n ∧
      canAssembleBalanced config) ↔
    n = 23 ∨ n = 24 ∨ n = 25 := by
  sorry

#check black_faces_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_faces_theorem_l1218_121840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1218_121804

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Checks if three circles touch each other externally -/
def circles_touch_externally (ω₁ ω₂ ω₃ : Circle) : Prop := sorry

/-- Checks if a point lies on a circle -/
def point_on_circle (P : Point) (ω : Circle) : Prop := sorry

/-- Calculates the distance between two points -/
noncomputable def distance (P₁ P₂ : Point) : ℝ := sorry

/-- Checks if a line segment is tangent to a circle -/
def line_tangent_to_circle (line : Point × Point) (ω : Circle) : Prop := sorry

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangle_area (P₁ P₂ P₃ : Point) : ℝ := sorry

/-- Given three externally touching circles of radius 36 and an equilateral triangle
    formed by points on these circles, prove that the area of the triangle
    can be expressed as √a + √b where a and b are natural numbers,
    and a + b = 283,435,776 -/
theorem triangle_area_proof (ω₁ ω₂ ω₃ : Circle) (P₁ P₂ P₃ : Point) :
  ω₁.radius = 36 ∧ ω₂.radius = 36 ∧ ω₃.radius = 36 →
  circles_touch_externally ω₁ ω₂ ω₃ →
  point_on_circle P₁ ω₁ ∧ point_on_circle P₂ ω₂ ∧ point_on_circle P₃ ω₃ →
  distance P₁ P₂ = distance P₂ P₃ ∧ distance P₂ P₃ = distance P₃ P₁ →
  line_tangent_to_circle (P₁, P₂) ω₃ ∧
  line_tangent_to_circle (P₂, P₃) ω₁ ∧
  line_tangent_to_circle (P₃, P₁) ω₂ →
  ∃ (a b : ℕ), triangle_area P₁ P₂ P₃ = Real.sqrt a + Real.sqrt b ∧ a + b = 283435776 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1218_121804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_loss_correct_min_purchase_price_correct_l1218_121837

/-- Represents the weather conditions --/
inductive Weather
| Normal
| Less

/-- Represents the price of vegetable A --/
inductive Price
| High
| Low

/-- The production cost per acre --/
noncomputable def productionCost : ℝ := 7000

/-- The probability of normal rainfall --/
noncomputable def probNormalRainfall : ℝ := 2/3

/-- The probability of less rainfall --/
noncomputable def probLessRainfall : ℝ := 1/3

/-- The yield per acre under normal rainfall --/
noncomputable def yieldNormal : ℝ := 2000

/-- The yield per acre under less rainfall --/
noncomputable def yieldLess : ℝ := 1500

/-- The high price of vegetable A --/
noncomputable def priceHigh : ℝ := 6

/-- The low price of vegetable A --/
noncomputable def priceLow : ℝ := 3

/-- The probability of high price under normal rainfall --/
noncomputable def probHighPriceNormal : ℝ := 1/4

/-- The probability of high price under less rainfall --/
noncomputable def probHighPriceLess : ℝ := 2/3

/-- The yield per acre with greenhouse --/
noncomputable def yieldGreenhouse : ℝ := 2500

/-- The expected income increase with greenhouse --/
noncomputable def incomeIncrease : ℝ := 1000

/-- The probability of not losing money --/
noncomputable def probNoLoss : ℝ := 7/18

/-- The minimum purchase price to increase expected income --/
noncomputable def minPurchasePrice : ℝ := 3.4

/-- Theorem: The probability of not losing money is 7/18 --/
theorem prob_no_loss_correct :
  probNoLoss = probNormalRainfall * probHighPriceNormal + probLessRainfall * probHighPriceLess :=
by sorry

/-- Theorem: The minimum purchase price to increase expected income by 1000 yuan is 3.4 yuan/kg --/
theorem min_purchase_price_correct :
  minPurchasePrice = (productionCost + incomeIncrease) / yieldGreenhouse :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_loss_correct_min_purchase_price_correct_l1218_121837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_3_over_1240_l1218_121831

noncomputable def g (x : ℝ) : ℝ := (x^7 - 1) / 5

theorem inverse_g_at_3_over_1240 :
  g⁻¹ (3/1240) = (1255/1240)^(1/7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_3_over_1240_l1218_121831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PSR_l1218_121820

-- Define the points
def P : ℝ × ℝ := (2, 9)
def S : ℝ × ℝ := (4, 0)

-- Define the slopes
def slope1 : ℝ := 3
def slope2 : ℝ := -1

-- Define the line equations
noncomputable def line1 (x : ℝ) : ℝ := slope1 * (x - P.1) + P.2
noncomputable def line2 (x : ℝ) : ℝ := slope2 * (x - S.1) + S.2

-- Define point R
noncomputable def R : ℝ × ℝ := (-(line1 0) / slope1, 0)

-- Calculate the area of triangle PSR
noncomputable def area_PSR : ℝ := (1/2) * (S.1 - R.1) * P.2

-- Theorem statement
theorem area_of_triangle_PSR : 
  area_PSR = 22.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PSR_l1218_121820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_odd_f_upper_bound_f_unbounded_l1218_121846

/-- The function f(m,n) represents the absolute difference between 
    the areas of black and white parts of a right-angled triangle 
    on a chessboard with legs of length m and n. -/
def f (m n : ℕ) : ℕ := sorry

/-- Theorem 1: f(m,n) = 0 if m and n are both even -/
theorem f_even (m n : ℕ) (hm : Even m) (hn : Even n) : f m n = 0 := by sorry

/-- Theorem 2: f(m,n) = 1 if m and n are both odd -/
theorem f_odd (m n : ℕ) (hm : Odd m) (hn : Odd n) : f m n = 1 := by sorry

/-- Theorem 3: f(m,n) ≤ max(m,n)/2 for all m, n -/
theorem f_upper_bound (m n : ℕ) : f m n ≤ max m n / 2 := by sorry

/-- Theorem 4: For any constant C, there exist m and n such that f(m,n) ≥ C -/
theorem f_unbounded : ∀ C : ℕ, ∃ m n : ℕ, f m n ≥ C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_odd_f_upper_bound_f_unbounded_l1218_121846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_is_subset_of_population_sleep_time_is_individual_l1218_121826

-- Define the total number of students
def total_students : Nat := 520

-- Define the number of students in the sample
def sample_size : Nat := 100

-- Define a structure for a student's sleep data
structure SleepData where
  hours : Nat

-- Define the type for the population and sample
def Population := Fin total_students → SleepData
def Sample := Fin sample_size → SleepData

-- Define a function to randomly select a sample from the population
noncomputable def select_sample (pop : Population) : Sample :=
  fun i => pop ⟨i.val, by sorry⟩

-- Theorem: The selected students form a sample of the population
theorem sample_is_subset_of_population (pop : Population) (sample : Sample) :
  ∃ (f : Fin sample_size → Fin total_students), Function.Injective f ∧ 
  (∀ i : Fin sample_size, sample i = pop (f i)) := by
  sorry

-- Theorem: Each student's sleep time is an individual data point
theorem sleep_time_is_individual (pop : Population) (i : Fin total_students) :
  ∃ (data : SleepData), pop i = data := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_is_subset_of_population_sleep_time_is_individual_l1218_121826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1218_121896

/-- The polygonal region defined by the given system of inequalities -/
def PolygonalRegion : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 ≤ 4 ∧ 3 * p.1 + p.2 ≥ 3 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- The vertices of the polygonal region -/
def Vertices : Set (ℝ × ℝ) :=
  {(0, 0), (1, 0), (0, 3), (4, 0)}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that the longest side of the polygonal region has length √10 -/
theorem longest_side_length :
  ∃ (p q : ℝ × ℝ), p ∈ Vertices ∧ q ∈ Vertices ∧
  distance p q = Real.sqrt 10 ∧
  ∀ (r s : ℝ × ℝ), r ∈ Vertices → s ∈ Vertices → distance r s ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1218_121896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_a_range_l1218_121849

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

theorem tangent_line_at_one (x y : ℝ) :
  f 2 1 = 2 ∧ (deriv (f 2)) 1 = 3 → 3 * x - y - 1 = 0 := by sorry

theorem monotonicity (a : ℝ) :
  (a ≥ 0 → ∀ x > 0, deriv (f a) x > 0) ∧
  (a < 0 → ∀ x ∈ Set.Ioo 0 (-1/a), deriv (f a) x > 0) ∧
  (a < 0 → ∀ x > -1/a, deriv (f a) x < 0) := by sorry

theorem a_range (a : ℝ) :
  (∀ x > 0, f a x < 2) → a < -1 / Real.exp 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_a_range_l1218_121849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_and_triangle_side_l1218_121885

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x + Real.sqrt 3 * (Real.sin x)^2 - Real.sqrt 3 / 2

theorem range_and_triangle_side :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-(Real.sqrt 3) / 2) 1) ∧
  (∀ A B C : ℝ, 
    0 < C ∧ C < Real.pi / 2 ∧
    f (C / 2) = -1/2 ∧
    A = Real.pi / 4 ∧
    (∃ (BC : ℝ), BC = 3 * Real.sqrt 2 ∧
      BC / Real.sin A = 3 / Real.sin C)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_and_triangle_side_l1218_121885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_fourth_id_l1218_121874

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  known_ids : Fin 3 → Nat

/-- Checks if a list of numbers forms an arithmetic sequence -/
def is_arithmetic_sequence (seq : List Nat) : Prop :=
  seq.length > 1 ∧ ∃ d, ∀ i, i + 1 < seq.length → seq[i+1]! - seq[i]! = d

/-- The theorem to be proved -/
theorem systematic_sample_fourth_id
  (sample : SystematicSample)
  (h1 : sample.total_students = 52)
  (h2 : sample.sample_size = 4)
  (h3 : sample.known_ids 0 = 3)
  (h4 : sample.known_ids 1 = 29)
  (h5 : sample.known_ids 2 = 42)
  : ∃ (full_sample : Fin 4 → Nat),
    (∀ i : Fin 3, full_sample i = sample.known_ids i) ∧
    full_sample 3 = 16 ∧
    is_arithmetic_sequence (List.ofFn full_sample) := by
  sorry

#check systematic_sample_fourth_id

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_fourth_id_l1218_121874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_constant_derivative_independent_of_sign_instantaneous_rate_of_change_at_point_delta_y_not_necessarily_zero_l1218_121860

-- Define a constant function
def constant_function (c : ℝ) : ℝ → ℝ := λ _ => c

-- Statement 1
theorem average_rate_of_change_constant (c x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  (constant_function c x₂ - constant_function c x₁) / (x₂ - x₁) = 0 := by sorry

-- Statement 2
theorem derivative_independent_of_sign (f : ℝ → ℝ) (x₀ : ℝ) (f' : ℝ → ℝ) :
  ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| ∧ |h| < δ → |f (x₀ + h) - f x₀ - f' x₀ * h| ≤ ε * |h| := by sorry

-- Statement 3
theorem instantaneous_rate_of_change_at_point (f : ℝ → ℝ) (x₀ : ℝ) :
  ∃ L, ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| ∧ |h| < δ → |(f (x₀ + h) - f x₀) / h - L| < ε := by sorry

-- Statement 4
theorem delta_y_not_necessarily_zero (f : ℝ → ℝ) (x₀ : ℝ) :
  ∃ L, ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| ∧ |h| < δ → |f (x₀ + h) - f x₀ - L * h| ≤ ε * |h| := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_constant_derivative_independent_of_sign_instantaneous_rate_of_change_at_point_delta_y_not_necessarily_zero_l1218_121860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_property_l1218_121833

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- Define the angle bisector of ∠BAC
noncomputable def angleBisectorBAC (t : Triangle) : Real := sorry

-- Define the function to calculate the angle between two lines
noncomputable def angleBetween (a b c : Real) : Real := sorry

-- Define cotangent function
noncomputable def cot (θ : Real) : Real := sorry

-- Theorem statement
theorem angle_bisector_property (t : Triangle) :
  angleBetween (angleBisectorBAC t) t.B t.C = 30 * Real.pi / 180 →
  |cot (angleBetween t.A t.B t.C) - cot (angleBetween t.A t.C t.B)| = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_property_l1218_121833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1218_121864

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then
    -Real.sqrt 2 * Real.sin x
  else if 0 < x ∧ x ≤ 1 then
    Real.tan (Real.pi / 4 * x)
  else
    0  -- undefined for x outside [-1, 1]

theorem f_composition_value : f (f (-Real.pi/4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1218_121864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_marathon_time_l1218_121857

/-- Calculates the time taken to complete a marathon given the runner's speed -/
noncomputable def marathonTime (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Represents the marathon problem and its solution -/
theorem jack_marathon_time 
  (marathonDistance : ℝ) 
  (jillTime : ℝ) 
  (speedRatio : ℝ) :
  marathonDistance = 42 →
  jillTime = 4.2 →
  speedRatio = 0.7 →
  marathonTime marathonDistance (marathonDistance / jillTime * speedRatio) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_marathon_time_l1218_121857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1218_121854

/-- Line l in the plane -/
def line_l (x y : ℝ) : Prop := x - 2*y + 8 = 0

/-- Curve C in the plane -/
def curve_C (x y : ℝ) : Prop := ∃ s : ℝ, x = 2*s^2 ∧ y = 2*Real.sqrt 2*s

/-- Distance from a point (x,y) to line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - 2*y + 8| / Real.sqrt 5

/-- The minimum distance from curve C to line l is 4√5 / 5 -/
theorem min_distance_curve_to_line :
  ∃ x y : ℝ, curve_C x y ∧
    ∀ x' y' : ℝ, curve_C x' y' →
      distance_to_line x y ≤ distance_to_line x' y' ∧
      distance_to_line x y = 4 * Real.sqrt 5 / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1218_121854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_greater_than_beta_l1218_121810

theorem alpha_greater_than_beta
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.sin α * Real.sin α = Real.cos (α - β)) :
  α > β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_greater_than_beta_l1218_121810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_negative_two_two_sqrt_three_l1218_121835

noncomputable section

/-- Converts a point from rectangular coordinates to polar coordinates -/
def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 ∧ y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 ∧ y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 ∧ y > 0 then Real.pi / 2
           else if x = 0 ∧ y < 0 then -Real.pi / 2
           else 0  -- undefined for (0,0)
  (r, θ)

theorem polar_coordinates_of_negative_two_two_sqrt_three :
  let x : ℝ := -2
  let y : ℝ := 2 * Real.sqrt 3
  let (r, θ) := rectangular_to_polar x y
  r = 4 ∧ θ = 2 * Real.pi / 3 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_negative_two_two_sqrt_three_l1218_121835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1218_121848

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a parabola -/
structure Parabola where
  focus : Point2D
  directrix : ℝ

/-- Check if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point2D) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Check if a point is on the parabola -/
def on_parabola (p : Parabola) (point : Point2D) : Prop :=
  point.x = (point.y^2 / (4 * (p.focus.x - p.directrix))) + (p.focus.x + p.directrix) / 2

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Main theorem -/
theorem intersection_distance (h : Hyperbola) (p : Parabola) :
  h.a = 4 →
  h.b = 3 →
  p.focus = ⟨5, 0⟩ →
  p.directrix = 0 →
  ∃ (p1 p2 : Point2D),
    on_hyperbola h p1 ∧
    on_hyperbola h p2 ∧
    on_parabola p p1 ∧
    on_parabola p p2 ∧
    distance p1 p2 = 30 * Real.sqrt 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1218_121848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1218_121873

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + x^3 - x^2 - a * x

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a / (a * x + 1) + 3 * x^2 - 2 * x - a

theorem problem_solution :
  -- Part I
  (∀ a : ℝ, f' a (2/3) = 0 → a = 0) ∧
  -- Part II
  (∀ a : ℝ, (∀ x : ℝ, x ≥ 1 → (f' a x ≥ 0)) ↔ (a > 0 ∧ a ≤ (1 + Real.sqrt 5) / 2)) ∧
  -- Part III
  (∀ b : ℝ, (∃ x : ℝ, f (-1) (1 - x) - (1 - x)^3 = b / x) ↔ b ≤ 0) := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1218_121873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baxter_peanut_purchase_l1218_121897

/-- Represents the peanut purchase scenario -/
structure PeanutPurchase where
  pricePerPound : ℚ
  minimumPurchase : ℚ
  discountThreshold : ℚ
  discountRate : ℚ
  taxRate : ℚ
  totalSpent : ℚ

/-- Calculates the pounds of peanuts purchased given the total spent -/
noncomputable def calculatePoundsPurchased (p : PeanutPurchase) : ℚ :=
  let preTaxAmount := p.totalSpent / (1 + p.taxRate)
  let preDiscountAmount := preTaxAmount / (1 - p.discountRate)
  preDiscountAmount / p.pricePerPound

/-- Theorem: Given the conditions, Baxter purchased 26 pounds over the minimum -/
theorem baxter_peanut_purchase :
  let p : PeanutPurchase := {
    pricePerPound := 3,
    minimumPurchase := 15,
    discountThreshold := 25,
    discountRate := 1/10,
    taxRate := 2/25,
    totalSpent := 12096/100
  }
  let poundsPurchased := calculatePoundsPurchased p
  poundsPurchased ≥ p.discountThreshold ∧
  (⌊(poundsPurchased : ℝ)⌋ : ℚ) - p.minimumPurchase = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baxter_peanut_purchase_l1218_121897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_implies_m_value_l1218_121807

theorem log_equation_implies_m_value (m n b : ℝ) (h : Real.log m = b - Real.log n) :
  m = (Real.exp b) / n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_implies_m_value_l1218_121807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_catches_mouse_l1218_121863

/-- The nonnegative quadrant -/
def N : Set (ℝ × ℝ) := {p | p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- Cat's position function -/
noncomputable def c : ℝ → ℝ × ℝ := sorry

/-- Mouse's position function -/
noncomputable def m : ℝ → ℝ × ℝ := sorry

/-- Cat's velocity function -/
noncomputable def c' : ℝ → ℝ × ℝ := sorry

/-- Mouse's velocity function -/
noncomputable def m' : ℝ → ℝ × ℝ := sorry

/-- The set of times where c and m are not differentiable -/
def S : Set ℝ := sorry

variable (h_S : Finite S)

variable (h_c_cont : Continuous c)
variable (h_m_cont : Continuous m)

variable (h_c_diff : ∀ t ∉ S, HasDerivAt c (c' t) t)
variable (h_m_diff : ∀ t ∉ S, HasDerivAt m (m' t) t)

variable (h_c_speed : ∀ t ∉ S, (c' t).1^2 + (c' t).2^2 = 2)
variable (h_m_speed : ∀ t ∉ S, (m' t).1^2 + (m' t).2^2 = 1)

variable (h_c_init : c 0 = (1, 1))
variable (h_m_init : m 0 = (0, 0))

variable (h_c_in_N : ∀ t ≥ 0, c t ∈ N)
variable (h_m_in_N : ∀ t ≥ 0, m t ∈ N)

theorem cat_catches_mouse : ∃ τ ∈ Set.Icc 0 1, c τ = m τ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_catches_mouse_l1218_121863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l1218_121888

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 2) + Real.cos (x / 2)
noncomputable def g (A ω φ x : ℝ) : ℝ := A * Real.cos (ω * x + φ)

-- Define what it means for a function to be periodic
def is_periodic (h : ℝ → ℝ) : Prop := ∃ T : ℝ, T ≠ 0 ∧ ∀ x, h (x + T) = h x

-- State the theorem
theorem minor_premise_identification :
  (∃ A ω φ : ℝ, ∀ x, f x = g A ω φ x) →  -- f can be transformed into g
  (∀ A ω φ, is_periodic (g A ω φ)) →     -- g is periodic for all A, ω, φ
  (is_periodic f) →                      -- f is periodic
  (∀ A ω φ, is_periodic (g A ω φ)) =     -- g is periodic is the minor premise
  (∃ p q r : Prop, p ∧ q → r ∧ q = (∀ A ω φ, is_periodic (g A ω φ))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l1218_121888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l1218_121809

-- Define the circles and cyclists
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Cyclist where
  circle : Circle
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

-- Define the problem setup
def intersectingCircles (c1 c2 : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
                 (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2

def startAtIntersection (c1 c2 : Circle) (cyclist1 cyclist2 : Cyclist) : Prop :=
  ∃ (p : ℝ × ℝ), (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
                 (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2 ∧
                 cyclist1.circle = c1 ∧ cyclist2.circle = c2

def meetAfterOneLap (cyclist1 cyclist2 : Cyclist) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ 
    (cyclist1.speed * t = 2 * Real.pi * cyclist1.circle.radius) ∧
    (cyclist2.speed * t = 2 * Real.pi * cyclist2.circle.radius)

-- Additional helper function (not part of the problem statement, but needed for the theorem)
noncomputable def cyclistPosition (cyclist : Cyclist) (t : ℝ) : ℝ × ℝ :=
  sorry

-- Define the theorem
theorem equidistant_point_exists 
  (c1 c2 : Circle) (cyclist1 cyclist2 : Cyclist) :
  intersectingCircles c1 c2 →
  startAtIntersection c1 c2 cyclist1 cyclist2 →
  meetAfterOneLap cyclist1 cyclist2 →
  (∃ (p : ℝ × ℝ), ∀ (t : ℝ), 
    Real.sqrt ((p.1 - (cyclistPosition cyclist1 t).1)^2 + (p.2 - (cyclistPosition cyclist1 t).2)^2) =
    Real.sqrt ((p.1 - (cyclistPosition cyclist2 t).1)^2 + (p.2 - (cyclistPosition cyclist2 t).2)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l1218_121809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_vote_count_l1218_121822

theorem election_vote_count : ∀ (V : ℚ),
  -- Conditions
  (V > 0) →  -- Total votes must be positive
  (0.45 * V = V * 45 / 100) →  -- Winner's vote percentage
  (0.20 * V = V * 20 / 100) →  -- Third candidate's vote percentage
  (∃ (second_place fourth_place : ℚ),
    (0.45 * V - second_place = 850) ∧  -- Winner's majority over second place
    (0.20 * V - fourth_place = 400) ∧  -- Third candidate's votes compared to fourth
    (2 * fourth_place + 0.45 * V + second_place + 0.20 * V = V)) →  -- Sum of all votes
  -- Conclusion
  V = 3300 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_vote_count_l1218_121822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1218_121881

/-- The function f as defined in the problem -/
noncomputable def f (x y : ℝ) : ℝ := ((x - y) + (4 + Real.sqrt (1 - x^2) + Real.sqrt (1 - y^2/9)))^2

/-- The theorem stating the maximum value of (x, y) for the function f -/
theorem max_value_of_f :
  ∃ (x y : ℝ), ∀ (a b : ℝ), f x y ≥ f a b ∧ x = 3 * Real.sqrt 3 + 1 ∧ y = -6 * Real.sqrt 3 + 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1218_121881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteer_distribution_plans_l1218_121800

/-- Function to calculate the number of distribution plans --/
def number_of_distribution_plans (n_girls n_boys n_places : ℕ) : ℕ := 
  sorry

/-- Theorem stating the correct number of distribution plans for the given problem --/
theorem volunteer_distribution_plans (n_girls n_boys n_places : ℕ) : 
  n_girls = 5 → n_boys = 2 → n_places = 2 → 
  (∀ place, place ≤ n_places → ∃ (g b : ℕ), g > 0 ∧ b > 0) →
  number_of_distribution_plans n_girls n_boys n_places = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteer_distribution_plans_l1218_121800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1218_121832

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (-2^x + a) / (2^x + 1)

-- State the theorem
theorem odd_function_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is an odd function
  (a = 1) ∧  -- a equals 1
  (∀ x y, x < y → f a x > f a y) ∧  -- f is decreasing
  (∀ k, (∀ t, f a (t^2 - 2*t) + f a (2*t^2 - k) < 0) ↔ k < -1/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1218_121832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_zeros_in_prime_power_l1218_121828

theorem consecutive_zeros_in_prime_power (p : ℕ) (m : ℕ) 
  (hp : Nat.Prime p) (hm : m > 0) : 
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, (p^n : ℕ) = k * (10^m) ∧ k * (10^m) < p^n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_zeros_in_prime_power_l1218_121828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_triangular_array_l1218_121887

theorem modified_triangular_array (N : ℕ) : 
  (∀ n : ℕ, n ≤ N → 3 * n = 3 * n) →
  (3 * (N * (N + 1)) / 2 = 3825) →
  N = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_triangular_array_l1218_121887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_cube_mult_two_l1218_121827

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -2],
    ![3, -1]]

theorem matrix_cube_mult_two :
  (2 : ℝ) • (A ^ 3) = ![![-20, 12],
                       ![-18, -2]] := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_cube_mult_two_l1218_121827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_correct_l1218_121823

/-- Given a boat traveling upstream and downstream, calculates the downstream speed. -/
noncomputable def downstream_speed (upstream_speed : ℝ) (average_speed : ℝ) : ℝ :=
  (2 * upstream_speed * average_speed) / (2 * upstream_speed - average_speed)

/-- Theorem stating that for a boat with given upstream speed and average round-trip speed,
    the calculated downstream speed is correct. -/
theorem downstream_speed_correct (upstream_speed average_speed : ℝ)
  (h_upstream : upstream_speed = 6)
  (h_average : average_speed = 6.857142857142857)
  : downstream_speed upstream_speed average_speed = 8 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_correct_l1218_121823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1218_121865

-- Define the circle C
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

-- Define the line L
def line_equation (x y a : ℝ) : Prop := x - y + a = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 2)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x y a : ℝ) : ℝ :=
  |x - y + a| / Real.sqrt 2

-- Theorem statement
theorem circle_line_intersection (a : ℝ) :
  (∀ x y, circle_equation x y → 
   ∀ x y, line_equation x y a → 
   distance_point_to_line (circle_center.1) (circle_center.2) a = Real.sqrt 2 / 2) →
  a = 2 ∨ a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1218_121865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_pushing_effort_l1218_121825

/-- The force needed to push a car varies inversely with the number of people pushing it. -/
def inverse_relation (force : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ p : ℝ, p > 0 → force p * p = k

/-- Given that 4 people can push a car with an effort of 120 newtons each,
    prove that 6 people pushing the car would each need to exert 80 newtons. -/
theorem car_pushing_effort (force : ℝ → ℝ) 
    (h1 : inverse_relation force)
    (h2 : force 4 = 120) : 
    force 6 = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_pushing_effort_l1218_121825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_sum_of_divisors_gt_30_l1218_121890

def n : ℕ := 360

-- Number of positive divisors
theorem number_of_divisors : (Finset.filter (fun x => n % x = 0) (Finset.range (n + 1))).card = 24 := by sorry

-- Sum of divisors greater than 30
theorem sum_of_divisors_gt_30 : (Finset.filter (fun x => n % x = 0 ∧ x > 30) (Finset.range (n + 1))).sum id = 1003 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_sum_of_divisors_gt_30_l1218_121890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evenHeadsProbabilityTheorem_l1218_121847

/-- The probability of getting an even number of heads when tossing a coin n times -/
noncomputable def evenHeadsProbability (n : ℕ) (p : ℝ) : ℝ :=
  (1 + (1 - 2*p)^n) / 2

/-- Theorem for the probability of even heads in coin tosses -/
theorem evenHeadsProbabilityTheorem (n : ℕ) (p : ℝ) 
  (h1 : 0 < p) (h2 : p < 1) : 
  (p = 1/2 → evenHeadsProbability n p = 1/2) ∧ 
  (p ≠ 1/2 → evenHeadsProbability n p = (1 + (1 - 2*p)^n) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evenHeadsProbabilityTheorem_l1218_121847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt27_star_3_sqrt12_plus_sqrt3_star_sqrt12_solve_x_star_neg_sqrt3_l1218_121838

-- Define the * operator
noncomputable def star_op (a b : ℝ) : ℝ := a * b + 3 / b - Real.sqrt 3

-- Theorem statements
theorem sqrt27_star_3 : star_op (Real.sqrt 27) 3 = 8 * Real.sqrt 3 + 1 := by sorry

theorem sqrt12_plus_sqrt3_star_sqrt12 : 
  star_op (Real.sqrt 12 + Real.sqrt 3) (Real.sqrt 12) = 12 + (5 * Real.sqrt 3) / 2 := by sorry

theorem solve_x_star_neg_sqrt3 : 
  ∃ x : ℝ, star_op x (-Real.sqrt 3) = Real.sqrt 3 ∧ x = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt27_star_3_sqrt12_plus_sqrt3_star_sqrt12_solve_x_star_neg_sqrt3_l1218_121838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_ratio_l1218_121880

/-- Represents the partnership investment scenario --/
structure Partnership where
  a_investment : ℚ
  b_investment : ℚ
  c_investment : ℚ
  annual_gain : ℚ
  a_share : ℚ

/-- Calculates the ratio of C's investment to A's investment --/
def investment_ratio (p : Partnership) : ℚ :=
  p.c_investment / p.a_investment

/-- Theorem stating the ratio of C's investment to A's investment --/
theorem partnership_investment_ratio (p : Partnership)
  (h1 : p.b_investment = 2 * p.a_investment)
  (h2 : p.annual_gain = 18000)
  (h3 : p.a_share = 6000)
  (h4 : p.a_share / p.annual_gain = 1 / 3)
  (h5 : 12 * p.a_investment / (12 * p.a_investment + 6 * p.b_investment + 4 * p.c_investment) = 1 / 3) :
  investment_ratio p = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_ratio_l1218_121880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_common_difference_l1218_121876

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Theorem: If a_3 = 4 and S_9 = 18 in an arithmetic sequence, then d = -1 -/
theorem arithmetic_seq_common_difference 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 3 = 4) 
  (h2 : sum_n seq 9 = 18) : 
  seq.d = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_common_difference_l1218_121876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l1218_121866

-- Define the shapes as propositions instead of types
def Square : Prop := sorry
def Parallelogram : Prop := sorry
def Trapezoid : Prop := sorry

-- Define the relationships
axiom square_is_parallelogram : Square → Parallelogram
axiom trapezoid_not_parallelogram : Trapezoid → ¬Parallelogram
axiom trapezoid_not_square : Trapezoid → ¬Square

-- Define the deduction structure
structure Deduction :=
  (major_premise : Prop)
  (minor_premise : Prop)
  (conclusion : Prop)

-- Define our specific deduction
def our_deduction : Deduction :=
  { major_premise := Square → Parallelogram,
    minor_premise := Trapezoid → ¬Parallelogram,
    conclusion := Trapezoid → ¬Square }

-- Theorem to prove
theorem minor_premise_identification :
  our_deduction.minor_premise = (Trapezoid → ¬Parallelogram) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l1218_121866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_90_minus_sqrt_88_closest_to_0_10_l1218_121819

theorem sqrt_90_minus_sqrt_88_closest_to_0_10 :
  let diff := Real.sqrt 90 - Real.sqrt 88
  ∀ x ∈ ({0.11, 0.12, 0.13, 0.14} : Set ℝ), |diff - 0.10| < |diff - x| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_90_minus_sqrt_88_closest_to_0_10_l1218_121819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_minus_base_nine_l1218_121862

def octal_to_decimal : ℕ := 
  7 * 8^5 + 6 * 8^4 + 5 * 8^3 + 4 * 8^2 + 3 * 8^1 + 2 * 8^0

def base_nine_to_decimal : ℕ := 
  5 * 9^5 + 4 * 9^4 + 3 * 9^3 + 2 * 9^2 + 1 * 9^1 + 0 * 9^0

theorem octal_minus_base_nine : 
  (octal_to_decimal : Int) - (base_nine_to_decimal : Int) = -67053 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_minus_base_nine_l1218_121862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_thirds_pi_plus_two_alpha_l1218_121869

theorem cos_two_thirds_pi_plus_two_alpha (α : ℝ) : 
  Real.sin (π / 6 - α) = 1 / 3 → Real.cos ((2 * π) / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_thirds_pi_plus_two_alpha_l1218_121869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1218_121892

noncomputable def C1 (β : ℝ) : ℝ × ℝ := (1 + Real.cos β, Real.sin β)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ * Real.cos θ, 4 * Real.cos θ * Real.sin θ)

noncomputable def l (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)

theorem intersection_distance (α : ℝ) (h1 : π / 2 < α) (h2 : α < π) :
  ∃ (βA θB tA tB : ℝ),
    tA ≠ 0 ∧ tB ≠ 0 ∧
    C1 βA = l tA α ∧
    C2 θB = l tB α ∧
    Real.sqrt 3 = Real.sqrt ((C2 θB).1 - (C1 βA).1)^2 + ((C2 θB).2 - (C1 βA).2)^2 →
    α = 5 * π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1218_121892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_temperature_is_84_l1218_121836

noncomputable def temperatures : List ℚ := [81, 78, 82, 84, 86, 88, 85, 87, 89, 83]

noncomputable def mean (l : List ℚ) : ℚ := (l.sum) / l.length

theorem mean_temperature_is_84 :
  Int.floor (mean temperatures) = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_temperature_is_84_l1218_121836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_our_monomial_l1218_121871

/-- The degree of a monomial is the sum of the exponents of its variables. -/
def degree_of_monomial (m : MvPolynomial (Fin 2) ℚ) : ℕ :=
  sorry

/-- The monomial x^2 * y / 3 -/
def our_monomial : MvPolynomial (Fin 2) ℚ :=
  sorry

theorem degree_of_our_monomial :
  degree_of_monomial our_monomial = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_our_monomial_l1218_121871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a6_b6_ratio_l1218_121886

/-- Given arithmetic sequences {a_n} and {b_n}, S_n and T_n are the sums of their first n terms -/
noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n : ℝ) * (a 1 + a n) / 2

noncomputable def T (b : ℕ → ℝ) (n : ℕ) : ℝ := (n : ℝ) * (b 1 + b n) / 2

/-- The ratio of S_n to T_n is n / (2n + 1) for all positive integers n -/
axiom ratio_condition {a b : ℕ → ℝ} (n : ℕ) (hn : n > 0) : 
  S a n / T b n = n / (2 * n + 1)

/-- The main theorem: Under the given conditions, a_6 / b_6 = 11 / 23 -/
theorem a6_b6_ratio {a b : ℕ → ℝ} : a 6 / b 6 = 11 / 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a6_b6_ratio_l1218_121886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1218_121813

/-- Line equation: x - y - 1 = 0 -/
def line_equation (x y : ℝ) : Prop := x - y - 1 = 0

/-- Point P coordinates -/
def point_P : ℝ × ℝ := (-2, 2)

/-- Y-intercept of the line -/
noncomputable def y_intercept : ℝ := -1

/-- Distance between point P and the line -/
noncomputable def distance_point_to_line : ℝ := 5 * Real.sqrt 2 / 2

theorem line_properties :
  (∀ x, line_equation x y_intercept) ∧
  (let (x₀, y₀) := point_P
   abs (x₀ - y₀ - 1) / Real.sqrt 2 = distance_point_to_line) := by
  sorry

#check line_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1218_121813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adding_back_5_red_equiv_xi_6_l1218_121811

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 10

/-- Represents the number of black balls in the bag -/
def black_balls : ℕ := 5

/-- Represents the number of draws made -/
def ξ : ℕ → ℕ := sorry

/-- Represents whether a red ball is drawn on the nth draw -/
def red_drawn : ℕ → Prop := sorry

/-- Represents whether a black ball is drawn on the nth draw -/
def black_drawn : ℕ → Prop := sorry

/-- The process continues until a red ball is drawn -/
axiom process_end (n : ℕ) : red_drawn n → ξ n = n

/-- If a black ball is drawn, another red ball is added to the bag -/
axiom black_to_red (n : ℕ) : black_drawn n → red_balls = red_balls + 1

/-- The event of "adding back 5 red balls" -/
def adding_back_5_red : Prop := ∀ n : ℕ, n < 5 → black_drawn n

/-- Theorem: The event of "adding back 5 red balls" is equivalent to ξ = 6 -/
theorem adding_back_5_red_equiv_xi_6 : adding_back_5_red ↔ ξ 6 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adding_back_5_red_equiv_xi_6_l1218_121811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_for_point_one_neg_two_l1218_121870

/-- If the terminal side of angle α passes through point (1,-2), then tan α = -2 -/
theorem tan_alpha_for_point_one_neg_two (α : ℝ) :
  (∃ (k : ℝ), k > 0 ∧ k * (Real.cos α) = 1 ∧ k * (Real.sin α) = -2) →
  Real.tan α = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_for_point_one_neg_two_l1218_121870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1218_121851

open Real

/-- Definition of the function f(x) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  2 * cos (ω * x - Real.pi / 6) * sin (ω * x) - (1 / 2) * cos (2 * ω * x + Real.pi)

/-- Theorem stating the range of f(x) and the maximum value of ω -/
theorem f_properties (ω : ℝ) (h : ω > 0) :
  (∀ x, (1 - sqrt 3) / 2 ≤ f ω x ∧ f ω x ≤ (1 + sqrt 3) / 2) ∧
  (∀ x ∈ Set.Icc (-3 * Real.pi / 4) (Real.pi / 2), Monotone (f ω) → ω ≤ 1 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1218_121851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1218_121867

noncomputable def g (x : ℝ) : ℝ := (3*x + 4)/(x + 3) - (x + 1)/(x + 2)

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ g x = y}

theorem g_properties :
  (∃ M : ℝ, (∀ y ∈ T, y ≤ M) ∧ M = 2) ∧
  (∃ m : ℝ, (∀ y ∈ T, m ≤ y) ∧ m = 1/3) ∧
  (1/3 ∈ T) ∧
  (2 ∉ T) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1218_121867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AE_length_l1218_121814

-- Define the points and circles
variable (A B C D E : ℝ × ℝ)
variable (S₁ S₂ : Set (ℝ × ℝ))

-- Define the conditions
axiom circles_meet : A ∈ S₁ ∧ A ∈ S₂ ∧ B ∈ S₁ ∧ B ∈ S₂
axiom line_through_B : ∃ l : Set (ℝ × ℝ), B ∈ l ∧ C ∈ l ∧ D ∈ l ∧ C ≠ B ∧ D ≠ B
axiom C_in_S₂ : C ∈ S₂
axiom D_in_S₁ : D ∈ S₁
axiom tangent_meet : ∃ t₁ t₂ : Set (ℝ × ℝ), 
  (∀ x ∈ t₁, x ∉ S₁ ∨ x = D) ∧ 
  (∀ x ∈ t₂, x ∉ S₂ ∨ x = C) ∧
  E ∈ t₁ ∧ E ∈ t₂ ∧ D ∈ t₁ ∧ C ∈ t₂

-- Define the distances
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

axiom AD_length : distance A D = 15
axiom AC_length : distance A C = 16
axiom AB_length : distance A B = 10

-- State the theorem
theorem AE_length : distance A E = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AE_length_l1218_121814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_imply_m_range_l1218_121812

-- Define the equation as noncomputable
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (5*x + 5/x) - |4*x - 4/x| - m

-- State the theorem
theorem equation_roots_imply_m_range :
  (∃ (a b c d : ℝ), 0 < a ∧ a < b ∧ b < c ∧ c < d ∧
    (∀ x : ℝ, 0 < x → (f x m = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)))) →
  (m > 6 ∧ m < 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_imply_m_range_l1218_121812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l1218_121899

theorem integer_solutions_equation :
  let f (a b c : ℤ) := (1/2 : ℚ) * ((a + b : ℚ) * (b + c) * (c + a)) + (a + b + c : ℚ)^3
  ∀ a b c : ℤ, f a b c = (1 : ℚ) - (a * b * c : ℚ) ↔
    (a, b, c) ∈ ({(1, 0, 0), (0, 1, 0), (0, 0, 1), (2, -1, -1), (-1, 2, -1), (-1, -1, 2)} : Set (ℤ × ℤ × ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l1218_121899
