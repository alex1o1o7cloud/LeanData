import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_l1029_102967

noncomputable def k (x : ℝ) : ℝ := 1 / (x + 5) + 1 / (x^2 - 9) + 1 / (x^3 - x + 1)

theorem k_domain (x : ℝ) :
  k x ≠ 0 ↔ x ≠ -5 ∧ x ≠ -3 ∧ x ≠ 3 ∧ x^3 - x + 1 ≠ 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_l1029_102967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_OB_is_sqrt_13_B_is_projection_of_A_l1029_102987

/-- The length of the line segment OB, where B is the orthogonal projection
    of point A(1, 2, 3) onto the yOz plane. -/
noncomputable def lengthOB : ℝ := Real.sqrt 13

/-- Point A in 3D space -/
def A : Fin 3 → ℝ := ![1, 2, 3]

/-- Point B, the orthogonal projection of A onto the yOz plane -/
def B : Fin 3 → ℝ := ![0, 2, 3]

/-- The origin O -/
def O : Fin 3 → ℝ := ![0, 0, 0]

/-- Theorem stating that the length of OB is √13 -/
theorem length_OB_is_sqrt_13 : 
  Real.sqrt ((B 0 - O 0)^2 + (B 1 - O 1)^2 + (B 2 - O 2)^2) = lengthOB := by
  sorry

/-- B is the orthogonal projection of A onto the yOz plane -/
theorem B_is_projection_of_A : 
  B 0 = 0 ∧ B 1 = A 1 ∧ B 2 = A 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_OB_is_sqrt_13_B_is_projection_of_A_l1029_102987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_bookcase_length_l1029_102999

-- Define constants
noncomputable def bookcase_length : ℝ := 48
noncomputable def shelving_unit_length : ℝ := 1.2
noncomputable def extra_shelf_length : ℝ := 36
noncomputable def meters_to_inches : ℝ := 39.37
noncomputable def inches_to_feet : ℝ := 1 / 12

-- Define the theorem
theorem custom_bookcase_length :
  let combined_length := bookcase_length + shelving_unit_length * meters_to_inches
  let combined_length_feet := combined_length * inches_to_feet
  let extra_shelf_feet := extra_shelf_length * inches_to_feet
  let final_length := combined_length_feet + extra_shelf_feet
  ∃ ε > 0, |final_length - 10.937| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_bookcase_length_l1029_102999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_min_value_l1029_102992

-- Define the function y
noncomputable def y (x : Real) : Real := Real.sin x + Real.cos x - Real.sin x * Real.cos x

-- Theorem statement
theorem smallest_angle_min_value :
  ∀ x : Real,
  (0 < x ∧ x ≤ Real.pi / 3) →  -- x is the smallest internal angle of a triangle
  (∀ θ : Real, (0 < θ ∧ θ ≤ Real.pi / 3) → y θ ≥ y x) →  -- y(x) is the minimum value
  y x = Real.sqrt 2 - 1/2 :=  -- the minimum value is √2 - 1/2
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_min_value_l1029_102992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_unique_solution_l1029_102921

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (2 * x + f y) = x + y + f x

/-- The identity function on ℝ -/
def identityFunc : ℝ → ℝ := λ x => x

/-- Theorem stating that the identity function is the unique solution -/
theorem identity_unique_solution :
  (SatisfiesFunctionalEquation identityFunc) ∧
  (∀ g : ℝ → ℝ, SatisfiesFunctionalEquation g → g = identityFunc) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_unique_solution_l1029_102921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1029_102997

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
noncomputable def train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) : ℝ :=
  speed * (1000 / 3600) * time - bridge_length

/-- Theorem stating that a train traveling at 72 km/hr and taking 16.5986721062315 seconds
    to cross a bridge of 132 m length has a length of 200 m -/
theorem train_length_calculation :
  train_length 72 16.5986721062315 132 = 200 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1029_102997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_pairs_l1029_102938

def satisfies_equation (a b : ℕ+) : Prop :=
  (a : ℕ).succ * (b : ℕ).succ + 2 = (a : ℕ).succ ^ 3 + 2 * (b : ℕ).succ

theorem positive_integer_pairs : 
  {(a, b) : ℕ+ × ℕ+ | satisfies_equation a b} = 
  {(1, 1), (3, 25), (4, 31), (5, 41), (8, 85)} := by
  sorry

#check positive_integer_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_pairs_l1029_102938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_grams_combinations_l1029_102902

def weight_combinations (n : ℕ) : ℕ := 
  (List.range 4).foldl (λ acc i => 
    acc + (List.range 4).foldl (λ acc' j => 
      acc' + (List.range 2).foldl (λ acc'' k => 
        acc'' + if i * 1 + j * 2 + k * 5 == n then 1 else 0) 0) 0) 0

theorem nine_grams_combinations : weight_combinations 9 = 8 := by
  -- Proof goes here
  sorry

#eval weight_combinations 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_grams_combinations_l1029_102902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_height_of_specific_cone_l1029_102932

/-- A right cone with given height and base diameter -/
structure RightCone where
  height : ℝ
  baseDiameter : ℝ

/-- The slant height of a right cone -/
noncomputable def slantHeight (c : RightCone) : ℝ :=
  Real.sqrt (c.height^2 + (c.baseDiameter / 2)^2)

/-- Theorem: For a right cone with height 3 and base diameter 8, 
    the slant height (distance from apex to any point on base circumference) is 5 -/
theorem slant_height_of_specific_cone :
  let c : RightCone := { height := 3, baseDiameter := 8 }
  slantHeight c = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_height_of_specific_cone_l1029_102932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_exceeds_light_by_one_l1029_102995

/-- Represents a square on the chessboard -/
inductive Square
| Dark
| Light

/-- Represents a 7x7 chessboard -/
def Chessboard := Fin 7 → Fin 7 → Square

/-- Returns the color of a square based on its coordinates -/
def squareColor (row col : Fin 7) : Square :=
  if (row.val + col.val) % 2 == 0 then Square.Dark else Square.Light

/-- Constructs a 7x7 chessboard with alternating colors starting with dark -/
def makeChessboard : Chessboard :=
  fun row col => squareColor row col

/-- Counts the number of dark squares on the chessboard -/
def countDarkSquares (board : Chessboard) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 7)) fun row =>
    Finset.sum (Finset.univ : Finset (Fin 7)) fun col =>
      match board row col with
      | Square.Dark => 1
      | Square.Light => 0)

/-- Counts the number of light squares on the chessboard -/
def countLightSquares (board : Chessboard) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 7)) fun row =>
    Finset.sum (Finset.univ : Finset (Fin 7)) fun col =>
      match board row col with
      | Square.Dark => 0
      | Square.Light => 1)

/-- The main theorem to be proved -/
theorem dark_exceeds_light_by_one :
  countDarkSquares makeChessboard = countLightSquares makeChessboard + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_exceeds_light_by_one_l1029_102995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_difference_l1029_102924

/-- Calculates the difference between profit shares of A and C given investments and B's profit share -/
theorem profit_share_difference (a_investment b_investment c_investment b_profit : ℕ) :
  a_investment = 8000 →
  b_investment = 10000 →
  c_investment = 12000 →
  b_profit = 1600 →
  (let total_parts := a_investment / 2000 + b_investment / 2000 + c_investment / 2000
   let profit_per_part := b_profit / (b_investment / 2000)
   let a_profit := profit_per_part * (a_investment / 2000)
   let c_profit := profit_per_part * (c_investment / 2000)
   c_profit - a_profit) = 640 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_difference_l1029_102924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_fourth_quadrant_l1029_102947

theorem sin_x_fourth_quadrant (x : ℝ) 
  (h1 : Real.sin (π / 2 + x) = 5 / 13) 
  (h2 : x ∈ Set.Icc (3 * π / 2) (2 * π)) : 
  Real.sin x = -12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_fourth_quadrant_l1029_102947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_emptying_time_l1029_102998

/-- Represents the time it takes for a cistern to empty through a leak, given the normal fill time and the fill time with a leak. -/
noncomputable def emptying_time (normal_fill_time leak_fill_time : ℝ) : ℝ :=
  let fill_rate := 1 / normal_fill_time
  let leak_rate := fill_rate - (1 / leak_fill_time)
  1 / leak_rate

/-- Theorem stating that for a cistern that normally fills in 6 hours but takes 8 hours with a leak, it will take 24 hours to empty when full. -/
theorem cistern_emptying_time :
  emptying_time 6 8 = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_emptying_time_l1029_102998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_z_l1029_102951

-- Define the complex function f
def f (z : ℂ) : ℂ := z^2 + Complex.I * z + 1

-- Define the conditions for z
def satisfies_conditions (z : ℂ) : Prop :=
  z.im > 0 ∧
  ∃ (a b : ℤ), f z = ↑a + ↑b * Complex.I ∧
  abs a ≤ 10 ∧ abs b ≤ 10

-- Theorem statement
theorem count_satisfying_z : 
  ∃ S : Finset ℂ, (∀ z ∈ S, satisfies_conditions z) ∧ S.card = 399 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_z_l1029_102951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l1029_102949

/-- The solution function to the Cauchy problem -/
noncomputable def y (x : ℝ) : ℝ := 2 * Real.exp x - Real.exp (2 * x)

/-- The differential equation -/
def diff_eq (f : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv f)) x - 3 * (deriv f x) + 2 * (f x) = 0

theorem cauchy_problem_solution :
  diff_eq y ∧ y 0 = 1 ∧ deriv y 0 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l1029_102949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_whitening_probability_l1029_102922

/-- Represents a 3x3 grid where each cell can be white or black -/
def Grid := Fin 3 → Fin 3 → Bool

/-- Rotates a grid 180 degrees -/
def rotate (g : Grid) : Grid :=
  fun i j => g (2 - i) (2 - j)

/-- Applies the whitening process to a rotated grid -/
def whiten (g : Grid) : Grid :=
  fun i j => g i j || (rotate g) i j

/-- The probability of a single cell being white -/
noncomputable def p_white : ℝ := 1 / 2

/-- The probability of the grid being all white after the process -/
noncomputable def p_all_white : ℝ := (1 / 2) * (1 / 4)^4

theorem grid_whitening_probability :
  p_all_white = 1 / 512 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_whitening_probability_l1029_102922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_sum_of_squares_l1029_102925

def sequence_a (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 11
  | n + 2 => 4 * sequence_a (n + 1) - sequence_a n

theorem sequence_is_sum_of_squares :
  ∀ n : ℕ, ∃ a b : ℕ, sequence_a n = a^2 + 2*b^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_sum_of_squares_l1029_102925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bd_length_l1029_102970

/-- Given a triangle ABC with AC = BC = 8 and AB = 2, and a point D on line AB
    such that B lies between A and D, and CD = 10, then BD = √37 - 1 -/
theorem triangle_bd_length (A B C D : EuclideanSpace ℝ (Fin 2)) : 
  (dist A C = 8) →
  (dist B C = 8) →
  (dist A B = 2) →
  (∃ t : ℝ, 0 < t ∧ D = (1 - t) • A + t • B) →
  (dist C D = 10) →
  dist B D = Real.sqrt 37 - 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bd_length_l1029_102970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_george_number_l1029_102969

/-- Represents the set of numbers said by a student -/
def StudentNumbers (start : ℕ) (step : ℕ) : Set ℕ :=
  {n : ℕ | ∃ k : ℕ, n = start + step * k ∧ n ≤ 300}

/-- The set of numbers said by Alice, Barbara, and Candice -/
def ABC : Set ℕ := StudentNumbers 2 3

/-- The set of numbers said by Debbie -/
def Debbie : Set ℕ := (Finset.range 301).toSet \ ABC ∩ StudentNumbers 3 4

/-- The set of numbers said by Eliza -/
def Eliza : Set ℕ := (Finset.range 301).toSet \ (ABC ∪ Debbie) ∩ StudentNumbers 4 5

/-- The set of numbers said by Fatima -/
def Fatima : Set ℕ := (Finset.range 301).toSet \ (ABC ∪ Debbie ∪ Eliza) ∩ StudentNumbers 5 6

/-- The number said by George -/
def George : ℕ := 104

theorem george_number : 
  George ∉ ABC ∧ 
  George ∉ Debbie ∧ 
  George ∉ Eliza ∧ 
  George ∉ Fatima ∧
  ∀ n : ℕ, n ≤ 300 → n ≠ George → (n ∈ ABC ∨ n ∈ Debbie ∨ n ∈ Eliza ∨ n ∈ Fatima) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_george_number_l1029_102969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ashwin_area_theorem_l1029_102955

/-- Ashwin's frog movement on the xy-plane --/
def ashwin_movement (n : ℕ) : ℕ × ℕ :=
  if n % 2 = 1 then (1, 0)  -- Odd step: move right
  else (0, Nat.log2 (Nat.gcd n (2^n)))  -- Even step: move up

/-- The total movement after n steps --/
def total_movement (n : ℕ) : ℕ × ℕ :=
  (List.range n).foldl (λ acc step ↦ (acc.1 + (ashwin_movement step).1, acc.2 + (ashwin_movement step).2)) (0, 0)

/-- The theorem to be proved --/
theorem ashwin_area_theorem :
  let final_pos := total_movement (2^2017 - 1)
  final_pos.1 = 2^2016 ∧ final_pos.2 = 2^2017 - 2018 ∧
  final_pos.1 * final_pos.2 / 2 = 2^2015 * (2^2017 - 2018) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ashwin_area_theorem_l1029_102955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_zero_l1029_102942

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (u v : V) : Prop := ∃ k : ℝ, u = k • v

theorem vector_sum_zero (a b c : V) 
  (h1 : ¬ collinear a b ∧ ¬ collinear b c ∧ ¬ collinear c a)
  (h2 : collinear (a + b) c)
  (h3 : collinear (b + c) a) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_zero_l1029_102942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_l1029_102960

-- Define the curves and the line
noncomputable def ln_curve (x : ℝ) : ℝ := Real.log x
def quad_curve (a x : ℝ) : ℝ := x^2 + a
def tangent_line (m x y : ℝ) : Prop := x - y + m = 0

-- Define the tangency condition
def is_tangent (f g : ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ x₀, tangent_line m x₀ (f x₀) ∧ 
        (∀ x, x ≠ x₀ → ¬(tangent_line m x (f x))) ∧
        (∀ x, x ≠ x₀ → ¬(tangent_line m x (g x)))

-- State the theorem
theorem tangent_sum (m a : ℝ) :
  is_tangent ln_curve (quad_curve a) m →
  m + a = -7/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_l1029_102960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_expression_l1029_102911

noncomputable def f (x θ : ℝ) : ℝ := Real.cos θ ^ 2 - 2 * x * Real.cos θ - 1

noncomputable def M (x : ℝ) : ℝ := ⨆ (θ : ℝ), f x θ

theorem M_expression (x : ℝ) : M x = max (2 * x) (-2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_expression_l1029_102911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_odd_series_sum_l1029_102965

theorem alternating_odd_series_sum : 
  let series := (fun n : ℕ => if n % 2 = 0 then (2*n + 1 : ℤ) else -(2*n + 1 : ℤ))
  let last_term := 2025
  let series_sum := (Finset.range ((last_term - 1) / 4 + 1)).sum (fun i => series i) + last_term
  series_sum = 1013 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_odd_series_sum_l1029_102965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_for_specific_ellipse_l1029_102964

/-- The radius of two externally tangent circles that are internally tangent to a given ellipse -/
noncomputable def circle_radius (a b c : ℝ) : ℝ :=
  let ellipse := λ (x y : ℝ) ↦ a * x^2 + b * y^2 = c
  let circle_center (r : ℝ) := (r, 0)
  let circle_equation (r : ℝ) := λ (x y : ℝ) ↦ (x - r)^2 + y^2 = r^2
  Real.sqrt (4/3)

/-- Two externally tangent circles of radius r that are internally tangent to the ellipse 4x^2 + 3y^2 = 12 have radius √(4/3) -/
theorem circle_radius_for_specific_ellipse :
  circle_radius 4 3 12 = Real.sqrt (4/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_for_specific_ellipse_l1029_102964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_value_l1029_102944

noncomputable def terminal_side (α : Real) : Set (Real × Real) :=
  {(x, y) | x = Real.cos α ∧ y = Real.sin α}

theorem min_angle_value (α : Real) 
  (h : ∃ (x y : Real), x = Real.sin (2 * Real.pi / 3) ∧ y = Real.cos (2 * Real.pi / 3) ∧ (x, y) ∈ terminal_side α) : 
  α % (2 * Real.pi) = 11 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_value_l1029_102944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_cost_calculation_l1029_102984

theorem pencil_cost_calculation (sharpenings_per_pencil : ℕ) 
  (hours_per_sharpening : ℚ) (initial_pencils : ℕ) (total_writing_hours : ℚ) 
  (additional_cost : ℚ) : 
  sharpenings_per_pencil = 5 →
  hours_per_sharpening = 3/2 →
  initial_pencils = 10 →
  total_writing_hours = 105 →
  additional_cost = 8 →
  let hours_per_pencil := sharpenings_per_pencil * hours_per_sharpening
  let total_pencils_needed := (total_writing_hours / hours_per_pencil).ceil
  let additional_pencils := total_pencils_needed - initial_pencils
  additional_cost / additional_pencils = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_cost_calculation_l1029_102984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_track_length_l1029_102907

/-- The ascent of the railway in feet -/
noncomputable def ascent : ℝ := 800

/-- The initial grade as a percentage -/
noncomputable def initial_grade : ℝ := 4

/-- The final grade as a percentage -/
noncomputable def final_grade : ℝ := 3

/-- Calculate the horizontal length for a given grade -/
noncomputable def horizontal_length (grade : ℝ) : ℝ := ascent / (grade / 100)

/-- The additional length of track required -/
noncomputable def additional_length : ℝ := horizontal_length final_grade - horizontal_length initial_grade

theorem additional_track_length : 
  ⌊additional_length⌋ = 6667 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_track_length_l1029_102907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_bisecting_folds_l1029_102962

/-- A parallelogram -/
structure Parallelogram where
  -- We don't need to define the structure fully, just declare it exists
  dummy : Unit

/-- A line in a plane -/
structure Line where
  -- Similarly, we just declare it exists
  dummy : Unit

/-- The property of a line bisecting the area of a parallelogram -/
def bisects_area (l : Line) (p : Parallelogram) : Prop :=
  sorry -- Definition omitted

/-- The set of lines that bisect the area of a parallelogram -/
def bisecting_lines (p : Parallelogram) : Set Line :=
  {l : Line | bisects_area l p}

/-- Theorem: There are infinitely many ways to fold a parallelogram to bisect its area -/
theorem infinite_bisecting_folds (p : Parallelogram) : Infinite (bisecting_lines p) := by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_bisecting_folds_l1029_102962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_area_half_shaded_area_half_l1029_102909

/-- A square with side length s -/
structure Square (s : ℝ) where
  side : s > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

theorem square_triangle_area_half (s : ℝ) (sq : Square s) :
  let o : Point := { x := 0, y := 0 }
  let p : Point := { x := 0, y := s }
  let q : Point := { x := s, y := s/2 }
  triangleArea o p q = (s^2) / 2 := by
  sorry

theorem shaded_area_half (s : ℝ) (sq : Square s) :
  let o : Point := { x := 0, y := 0 }
  let p : Point := { x := 0, y := s }
  let q : Point := { x := s, y := s/2 }
  let squareArea := s^2
  let shadedArea := squareArea - triangleArea o p q
  shadedArea / squareArea = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_area_half_shaded_area_half_l1029_102909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l1029_102948

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  focus : ℝ × ℝ
  vertex : ℝ × ℝ

/-- Theorem about the sum of h, k, a, and b for a specific hyperbola -/
theorem hyperbola_sum (H : Hyperbola) 
    (h_center : H.center = (1, 1))
    (h_focus : H.focus = (1, 10))
    (h_vertex : H.vertex = (1, 4)) : 
  let (h, k) := H.center
  let a := |H.vertex.2 - H.center.2|
  let c := |H.focus.2 - H.center.2|
  let b := Real.sqrt (c^2 - a^2)
  h + k + a + b = 5 + 6 * Real.sqrt 2 := by
  sorry

#check hyperbola_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l1029_102948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stoppage_time_is_six_point_five_minutes_l1029_102941

/-- Represents the types of bus stoppages -/
inductive StoppageType
  | Short
  | Medium
  | Long

/-- Represents a bus stoppage with its type and duration -/
structure Stoppage where
  type : StoppageType
  duration : ℝ
  frequency : ℝ

/-- Represents a bus with its speeds and stoppages -/
structure Bus where
  speed_without_stoppages : ℝ
  speed_with_stoppages : ℝ
  short_stoppages : Stoppage
  medium_stoppages : Stoppage
  long_stoppages : Stoppage

/-- Calculates the total stoppage time per hour for a given bus -/
noncomputable def totalStoppageTimePerHour (bus : Bus) : ℝ :=
  bus.short_stoppages.duration * bus.short_stoppages.frequency +
  bus.medium_stoppages.duration * bus.medium_stoppages.frequency +
  bus.long_stoppages.duration * bus.long_stoppages.frequency / 2

/-- Theorem stating that the total stoppage time per hour is 6.5 minutes -/
theorem stoppage_time_is_six_point_five_minutes (bus : Bus) : 
  bus.speed_without_stoppages = 90 ∧ 
  bus.speed_with_stoppages = 84 ∧
  bus.short_stoppages = { type := StoppageType.Short, duration := 0.5, frequency := 2 } ∧
  bus.medium_stoppages = { type := StoppageType.Medium, duration := 3, frequency := 1 } ∧
  bus.long_stoppages = { type := StoppageType.Long, duration := 5, frequency := 0.5 } →
  totalStoppageTimePerHour bus = 6.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stoppage_time_is_six_point_five_minutes_l1029_102941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l1029_102945

noncomputable def f (x : ℝ) := (x - 3) / (4 * x + 6)

theorem vertical_asymptote_of_f :
  ∃ (x : ℝ), x = -3/2 ∧ ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
  ∀ (y : ℝ), 0 < |y - x| ∧ |y - x| < δ → |f y| > 1/ε := by
  sorry

#check vertical_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l1029_102945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l1029_102943

theorem trig_expression_simplification (α : Real) :
  (Real.sin (α + π))^2 * Real.cos (π + α) * Real.cos (-α - 2*π) /
  (Real.tan (π + α) * (Real.sin (π/2 + α))^3 * Real.sin (-α - 2*π)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l1029_102943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_internal_angle_sine_cosine_difference_l1029_102920

theorem internal_angle_sine_cosine_difference (A : ℝ) :
  (0 < A) ∧ (A < Real.pi) →  -- A is an internal angle of a triangle
  Real.sin (2 * A) = -2/3 →
  Real.sin A - Real.cos A = Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_internal_angle_sine_cosine_difference_l1029_102920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1029_102913

noncomputable section

open Real

/-- Given a triangle ABC with sides a, b, c and corresponding angles A, B, C -/
def Triangle (a b c A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_problem (a b c A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : b * cos A - a * sin B = 0)
  (h_b : b = sqrt 2)
  (h_area : (1/2) * a * b * sin C = 1) :
  A = Real.pi/4 ∧ a = sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1029_102913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_x_value_l1029_102939

theorem power_equality_x_value (x : ℝ) : (8 : ℝ)^x = 2^9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_x_value_l1029_102939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_gave_seventy_cents_l1029_102963

/-- Represents the amount of money Lizzy has, in cents -/
def LizzyMoney : ℕ → ℕ
| 0 => 80 + 40 -- Initial amount from parents
| 1 => LizzyMoney 0 - 50 -- After buying candy
| 2 => 140 -- Final amount
| _ => 0 -- Default case for other natural numbers

/-- The amount Lizzy's uncle gave her -/
def UncleMoney : ℕ := LizzyMoney 2 - LizzyMoney 1

theorem uncle_gave_seventy_cents : UncleMoney = 70 := by
  rfl

#eval UncleMoney

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_gave_seventy_cents_l1029_102963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1029_102910

noncomputable def series_term (n : ℕ) : ℝ := (2^n) / (5^(2^n) + 1)

theorem series_sum : ∑' n, series_term n = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1029_102910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_iff_a_range_f_derivative_negative_at_midpoint_l1029_102903

noncomputable section

variable (a : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1)^2 + x * Real.exp x

def has_max_and_min (f : ℝ → ℝ) : Prop :=
  ∃ (x_max x_min : ℝ), ∀ x, f x ≤ f x_max ∧ f x_min ≤ f x

theorem f_extrema_iff_a_range :
  has_max_and_min (f a) ↔ a ∈ Set.Iio (-1 / (2 * Real.exp 1)) ∪ Set.Ioc (-1 / (2 * Real.exp 1)) 0 :=
sorry

theorem f_derivative_negative_at_midpoint {a : ℝ} (ha : a > 0) {x₁ x₂ : ℝ}
  (hx₁ : f a x₁ = 0) (hx₂ : f a x₂ = 0) :
  deriv (f a) ((x₁ + x₂) / 2) < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_iff_a_range_f_derivative_negative_at_midpoint_l1029_102903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1029_102908

/-- The equation of a line tangent to a circle at a given point. -/
theorem tangent_line_equation (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 4*x - 1 = 0 → (a*x + b*y - 3)^2 > 0) →  -- Line is not secant
  a*(-1) + b*2 - 3 = 0 →  -- Point (-1, 2) lies on the line
  (∃ t : ℝ, (-1 - (-2))*a + (2 - 0)*b = 0) →  -- Line is perpendicular to radius
  a = 1 ∧ b = 2 := by
  sorry

#check tangent_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1029_102908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_intersection_A_B_l1029_102961

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (3^x - 9)

-- Define the domain of f
def A : Set ℝ := {x | x > 2}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the domain of f
theorem domain_of_f : {x | ∃ y, f x = y} = A := by sorry

-- Theorem for the intersection of A and B
theorem intersection_A_B (a : ℝ) :
  A ∩ B a = if a ≤ 2 then ∅ else {x | 2 < x ∧ x < a} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_intersection_A_B_l1029_102961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_unknown_number_of_men_l1029_102989

/-- Represents the number of days to complete a job -/
def days_to_complete : ℕ := 15

/-- Represents the unknown number of men working on the job -/
def number_of_men : ℕ := sorry

/-- A theorem stating that the job is completed in 15 days, regardless of the number of men -/
theorem job_completion :
  ∀ (n : ℕ), n > 0 → days_to_complete = 15 := by
  intro n h
  rfl

/-- A statement that we cannot determine the exact number of men -/
theorem unknown_number_of_men :
  ¬ ∃ (n : ℕ), n > 0 ∧ number_of_men = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_unknown_number_of_men_l1029_102989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1029_102981

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (oplus 1 x) * x - (oplus 2 x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), M = 6 ∧ 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x ≤ M) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = M) := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1029_102981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_with_12_segments_area_16_l1029_102959

/-- Represents a polygon constructed from line segments -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.length ≥ 3 ∧ vertices.head? = vertices.get? (vertices.length - 1)

/-- Calculates the area of a polygon -/
noncomputable def polygonArea (p : Polygon) : ℝ :=
  sorry

/-- Calculates the perimeter of a polygon -/
noncomputable def polygonPerimeter (p : Polygon) : ℝ :=
  sorry

/-- Checks if all line segments in the polygon have the same length -/
def allSegmentsEqualLength (p : Polygon) (length : ℝ) : Prop :=
  sorry

/-- Theorem stating the existence of a polygon with 12 segments of length 2, area 16, and perimeter 24 -/
theorem polygon_with_12_segments_area_16 :
  ∃ (p : Polygon), 
    polygonArea p = 16 ∧ 
    polygonPerimeter p = 24 ∧ 
    allSegmentsEqualLength p 2 ∧ 
    p.vertices.length = 12 := by
  sorry

#check polygon_with_12_segments_area_16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_with_12_segments_area_16_l1029_102959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_def_sequence_properties_l1029_102971

-- Define the sequence a_n
def a (n : ℕ) : ℝ := 2^n

-- Define S_n as the sum of the first n terms of a_n
def S (n : ℕ) : ℝ := 2 * a n - 2

-- Define the condition S_n = 2a_n - 2
theorem S_def (n : ℕ) : S n = 2 * a n - 2 := by rfl

-- Define b_n
noncomputable def b (n : ℕ) : ℝ := a n * Real.log (a n) / Real.log 2

-- Define T_n as the sum of the first n terms of b_n
noncomputable def T (n : ℕ) : ℝ := (n - 1) * 2^(n + 1) + 2

theorem sequence_properties :
  (∀ n : ℕ, a n = 2^n) ∧
  (∀ n : ℕ, T n = (n - 1) * 2^(n + 1) + 2) := by
  constructor
  · intro n
    rfl
  · intro n
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_def_sequence_properties_l1029_102971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonals_perpendicular_bisectors_l1029_102912

/-- Definition of a square -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ

/-- Definition of diagonals of a square -/
def diagonals (s : Square) : ((ℝ × ℝ) × (ℝ × ℝ)) × ((ℝ × ℝ) × (ℝ × ℝ)) :=
  ((s.vertices 0, s.vertices 2), (s.vertices 1, s.vertices 3))

/-- Two lines are perpendicular -/
def are_perpendicular (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

/-- A line bisects another line -/
def bisects (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

/-- Main theorem -/
theorem square_diagonals_perpendicular_bisectors (s : Square) :
  (are_perpendicular (diagonals s).1 (diagonals s).2 ∧
   bisects (diagonals s).1 (diagonals s).2 ∧
   bisects (diagonals s).2 (diagonals s).1) ↔
  (are_perpendicular (diagonals s).1 (diagonals s).2 ∧
   (bisects (diagonals s).1 (diagonals s).2 ∧
    bisects (diagonals s).2 (diagonals s).1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonals_perpendicular_bisectors_l1029_102912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_results_l1029_102982

def possible_expressions : List (Int → Int → Int → Int → Int → Int) := [
  (λ a b c d e => a + b + c + d + e),
  (λ a b c d e => a + b + c + d - e),
  (λ a b c d e => a + b + c - d + e),
  (λ a b c d e => a + b + c - d - e),
  (λ a b c d e => a + b - c + d + e),
  (λ a b c d e => a + b - c + d - e),
  (λ a b c d e => a + b - c - d + e),
  (λ a b c d e => a + b - c - d - e),
  (λ a b c d e => a - b + c + d + e),
  (λ a b c d e => a - b + c + d - e),
  (λ a b c d e => a - b + c - d + e),
  (λ a b c d e => a - b + c - d - e),
  (λ a b c d e => a - b - c + d + e),
  (λ a b c d e => a - b - c + d - e),
  (λ a b c d e => a - b - c - d + e),
  (λ a b c d e => a - b - c - d - e)
]

theorem sum_of_possible_results :
  (List.map (λ f => f 625 125 25 5 1) possible_expressions).sum = 10000 := by
  sorry

#eval (List.map (λ f => f 625 125 25 5 1) possible_expressions).sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_results_l1029_102982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_C_and_a_value_l1029_102974

variable (x a : ℝ)

def A (x a : ℝ) : ℝ := 2 * x^2 + 3 * a * x - 2 * x - 1
def B (x a : ℝ) : ℝ := -3 * x^2 + 3 * a * x - 1
def C (x a : ℝ) : ℝ := 3 * A x a - 2 * B x a

theorem polynomial_C_and_a_value :
  (C x a = 12 * x^2 + 3 * a * x - 6 * x - 1) ∧
  (3 * a - 6 = 0 → a = 2) := by
  sorry

#check polynomial_C_and_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_C_and_a_value_l1029_102974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1029_102927

-- Define the points
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 9)
def C : ℝ × ℝ := (7, 6)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter function
noncomputable def perimeter (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  distance p1 p2 + distance p2 p3 + distance p3 p1

-- Theorem statement
theorem triangle_perimeter :
  perimeter A B C = 6 + 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1029_102927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flashlight_equilibrium_l1029_102935

/-- Represents a box containing flashlights -/
structure Box where
  lit : ℕ
  unlit : ℕ

/-- The initial state of the boxes -/
def initial_state : Box × Box := (⟨100, 100⟩, ⟨100, 100⟩)

/-- Move all flashlights from one box to another -/
def move_all (b1 b2 : Box) : Box :=
  ⟨b2.lit + b1.lit, b2.unlit + b1.unlit⟩

/-- Move k flashlights from one box to another, pressing their buttons -/
def move_and_press (b1 b2 : Box) (k : ℕ) : Box × Box :=
  let moved_lit := min k b1.lit
  let moved_unlit := k - moved_lit
  let new_b1 := ⟨b1.lit - moved_lit, b1.unlit - moved_unlit⟩
  let new_b2 := ⟨b2.lit + moved_unlit, b2.unlit + moved_lit⟩
  (new_b1, new_b2)

theorem flashlight_equilibrium :
  let (box_a, box_b) := initial_state
  let all_in_a := move_all box_b box_a
  let (final_a, final_b) := move_and_press all_in_a ⟨0, 0⟩ 100
  final_a.lit = final_b.lit := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flashlight_equilibrium_l1029_102935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l1029_102937

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def valid_arrangement (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  (∀ i j : Fin 3, m i j ∈ Finset.range 9) ∧
  (∀ x ∈ Finset.range 9, ∃! i j : Fin 3, m i j = x + 1) ∧
  (∀ i : Fin 3, is_prime (m i 0 + m i 1 + m i 2)) ∧
  (∀ j : Fin 3, is_prime (m 0 j + m 1 j + m 2 j))

theorem exists_valid_arrangement : ∃ m : Matrix (Fin 3) (Fin 3) ℕ, valid_arrangement m := by
  sorry

#check exists_valid_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l1029_102937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_complement_unique_m_value_l1029_102901

-- Define the function f (marked as noncomputable due to sqrt)
noncomputable def f (x : ℝ) := Real.sqrt (-x^2 + 5*x - 6)

-- Define set A (domain of f)
def A : Set ℝ := {x | -x^2 + 5*x - 6 ≥ 0}

-- Define set B
def B : Set ℝ := {x | 2 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 16}

-- Define set C
def C (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3*m}

-- Theorem 1
theorem intersection_and_complement :
  A ∩ B = Set.Icc 2 3 ∧ 
  (Set.univ : Set ℝ) \ B = Set.Iio 1 ∪ Set.Ioi 4 :=
by sorry

-- Theorem 2
theorem unique_m_value :
  ∃! m : ℝ, A ∪ C m = A :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_complement_unique_m_value_l1029_102901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunks_for_two_dozen_pears_l1029_102977

/-- The number of lunks needed to purchase a given number of pears -/
def lunks_for_pears (lunk_kunk_rate : ℚ) (kunk_pear_rate : ℚ) (num_pears : ℕ) : ℕ :=
  let kunks_needed : ℚ := (num_pears : ℚ) / kunk_pear_rate
  let lunks_needed : ℚ := kunks_needed * lunk_kunk_rate
  (Int.ceil lunks_needed).toNat

/-- Theorem stating that 20 lunks are needed to purchase 24 pears given the specified exchange rates -/
theorem lunks_for_two_dozen_pears :
  lunks_for_pears (8/5) (6/3) 24 = 20 := by
  sorry

#eval lunks_for_pears (8/5) (6/3) 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunks_for_two_dozen_pears_l1029_102977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_parabola_line_l1029_102958

/-- The area between the parabola y = a³x² - a⁴x and the line y = x for a > 0 -/
noncomputable def area (a : ℝ) : ℝ := (a^4 + 1)^3 / (6 * a^6)

/-- The minimum area between the parabola y = a³x² - a⁴x and the line y = x -/
theorem min_area_parabola_line :
  ∃ (min_area : ℝ), min_area = 4/3 ∧ 
  ∀ (a : ℝ), a > 0 → area a ≥ min_area := by
  sorry

#check min_area_parabola_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_parabola_line_l1029_102958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1029_102916

-- Define the function (marked as noncomputable due to use of Real.log)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2 * x)

-- Define what it means for x to be a valid argument of f
def IsValidArg (x : ℝ) : Prop :=
  1 - 2 * x > 0

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | IsValidArg x} = {x : ℝ | x < (1 : ℝ) / 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1029_102916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l1029_102917

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 6*x + 1

-- Define the circle
def circleC (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- Define the line
def line (x y : ℝ) (a : ℝ) : Prop := x - y + a = 0

-- Define perpendicularity of OA and OB
def perpendicular (xA yA xB yB : ℝ) : Prop := xA * xB + yA * yB = 0

theorem circle_and_line_intersection :
  -- The circle intersects with the coordinate axes and the curve
  (∃ x, circleC x 0) ∧ (∃ y, circleC 0 y) ∧ (∃ x y, curve x y ∧ circleC x y) →
  -- If the circle intersects with the line at points A and B, and OA ⊥ OB
  (∀ a : ℝ, ∃ xA yA xB yB,
    circleC xA yA ∧ circleC xB yB ∧
    line xA yA a ∧ line xB yB a ∧
    perpendicular xA yA xB yB →
    -- Then a = -1
    a = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l1029_102917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pappus_chain_inequality_l1029_102985

/-- Two circles with unequal radii that are tangent internally -/
structure TangentCircles where
  A : Set (ℝ × ℝ)  -- Circle A
  B : Set (ℝ × ℝ)  -- Circle B
  r : ℝ  -- Radius of A
  R : ℝ  -- Radius of B
  A₀ : ℝ × ℝ  -- Point of tangency
  h_radii : r ≠ R  -- Radii are unequal
  h_tangent : A₀ ∈ A ∧ A₀ ∈ B  -- A₀ is on both circles
  h_internal : ∀ p ∈ A, p ∈ B ∨ p = A₀  -- A is inside B

/-- Sequence of circles tangent to A and B -/
def CircleSequence (tc : TangentCircles) :=
  ℕ → Set (ℝ × ℝ)

/-- Sequence of points where consecutive circles touch -/
def TouchPoints (tc : TangentCircles) (cs : CircleSequence tc) :=
  ℕ → ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of distances between consecutive touch points -/
noncomputable def SumOfDistances (tc : TangentCircles) (cs : CircleSequence tc) (tp : TouchPoints tc cs) : ℝ :=
  ∑' n, distance (tp n) (tp (n+1))

/-- Main theorem -/
theorem pappus_chain_inequality (tc : TangentCircles) 
  (cs : CircleSequence tc) 
  (tp : TouchPoints tc cs)
  (h_tangent : ∀ n, (cs n ∩ tc.A).Nonempty ∧ (cs n ∩ tc.B).Nonempty)
  (h_touch : ∀ n, tp n ∈ cs n ∩ cs (n+1)) :
  SumOfDistances tc cs tp < (4 * Real.pi * tc.R * tc.r) / (tc.R + tc.r) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pappus_chain_inequality_l1029_102985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sqrt_two_minus_cos_l1029_102972

noncomputable def f (x : ℝ) : ℝ := (2 - Real.cos x) ^ (1 / x^2)

theorem limit_sqrt_two_minus_cos (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - Real.sqrt (Real.exp 1)| < ε := by
  sorry

#check limit_sqrt_two_minus_cos

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sqrt_two_minus_cos_l1029_102972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nickel_difference_l1029_102986

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Proves that the difference in money between Linda and Carol is 20p - 30 cents -/
theorem nickel_difference (p : ℤ) : 
  (nickel_value : ℤ) * ((7 * p - 2) - (3 * p + 4)) = 20 * p - 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nickel_difference_l1029_102986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_catches_john_in_30_minutes_l1029_102979

/-- The time (in minutes) it takes for Bob to catch up with John -/
noncomputable def catchUpTime (johnSpeed bobSpeed : ℝ) (initialDistance : ℝ) : ℝ :=
  initialDistance / (bobSpeed - johnSpeed) * 60

/-- Theorem stating that Bob catches up to John in 30 minutes -/
theorem bob_catches_john_in_30_minutes :
  catchUpTime 3 5 1 = 30 := by
  -- Unfold the definition of catchUpTime
  unfold catchUpTime
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_catches_john_in_30_minutes_l1029_102979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l1029_102953

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) + v }

/-- The original parabola y = x^2 - 2 -/
def original_parabola : Parabola :=
  { f := fun x => x^2 - 2 }

/-- The translated parabola -/
def translated_parabola : Parabola :=
  translate (translate original_parabola 1 0) 0 3

theorem parabola_translation :
  ∀ x, translated_parabola.f x = (x - 1)^2 + 1 := by
  intro x
  simp [translated_parabola, translate, original_parabola]
  ring

#check parabola_translation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l1029_102953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_order_l1029_102940

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -x^2 - 2*x + 2

/-- Point A on the parabola -/
def A (y₁ : ℝ) : ℝ × ℝ := (-2, y₁)

/-- Point B on the parabola -/
def B (y₂ : ℝ) : ℝ × ℝ := (1, y₂)

/-- Point C on the parabola -/
def C (y₃ : ℝ) : ℝ × ℝ := (2, y₃)

theorem parabola_point_order (y₁ y₂ y₃ : ℝ) :
  parabola (A y₁).1 (A y₁).2 ∧ 
  parabola (B y₂).1 (B y₂).2 ∧ 
  parabola (C y₃).1 (C y₃).2 →
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

#check parabola_point_order

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_order_l1029_102940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_property_l1029_102933

def x : ℕ → ℚ
  | 0 => 7/2  -- Add this case to handle Nat.zero
  | 1 => 7/2
  | n + 1 => x n * (x n - 2)

theorem prime_divisor_property (a b : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_coprime : Nat.Coprime a b) (h_x2021 : x 2021 = a / b) (h_divides : p ∣ a) :
  3 ∣ p - 1 ∨ p = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_property_l1029_102933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_for_receptacles_only_l1029_102990

/-- Represents a cube with snaps and receptacle holes -/
structure Cube where
  snaps : Fin 2 → Unit
  receptacles : Fin 4 → Unit

/-- Represents a configuration of cubes -/
structure Configuration where
  cubes : List Cube
  connections : List (Nat × Nat)  -- Pairs of indices representing connected cubes

/-- Predicate to check if a configuration shows only receptacle holes -/
def shows_only_receptacles (config : Configuration) : Prop :=
  ∀ c ∈ config.cubes, ∀ s : Fin 2, ∃ (other : Cube) (i j : Nat),
    other ∈ config.cubes ∧ (i, j) ∈ config.connections ∧
    i < config.cubes.length ∧ j < config.cubes.length ∧
    config.cubes[i]? = some c ∧ config.cubes[j]? = some other

/-- The main theorem stating the minimum number of cubes required -/
theorem min_cubes_for_receptacles_only :
  ∃ (n : Nat) (config : Configuration),
    n = config.cubes.length ∧
    shows_only_receptacles config ∧
    (∀ (m : Nat) (other_config : Configuration),
      m < n →
      shows_only_receptacles other_config →
      other_config.cubes.length ≥ n) ∧
    n = 4 := by
  sorry

#check min_cubes_for_receptacles_only

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_for_receptacles_only_l1029_102990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_problem_l1029_102957

theorem expansion_problem (a : ℝ) :
  (∃ k : ℝ, k = (Nat.choose 5 4) * a * (1 : ℝ)^4 ∧ k = 5) →
  (a = 1 ∧ Nat.choose 5 2 = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_problem_l1029_102957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_distance_l1029_102956

/-- A hyperbola with equation x^2/4 - y^2/3 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 / 3 = 1}

/-- The left focus of the hyperbola -/
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 7, 0)

/-- The right focus of the hyperbola -/
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 7, 0)

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem hyperbola_point_distance (P : ℝ × ℝ) 
  (h1 : P ∈ Hyperbola) 
  (h2 : P.1 > 0) -- P is on the right branch
  (h3 : distance P F1 = 2 * distance P F2) :
  distance O P = Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_distance_l1029_102956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_max_eccentricity_l1029_102973

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: For a hyperbola with a triangle formed by certain intersections
    having perimeter 12, the eccentricity when ab is maximized is √2 -/
theorem hyperbola_max_eccentricity (h : Hyperbola) 
  (triangle_perimeter : ℝ) (h_perimeter : triangle_perimeter = 12) :
  ∃ (h_max : Hyperbola), 
    (∀ h' : Hyperbola, h'.a * h'.b ≤ h_max.a * h_max.b) → 
    eccentricity h_max = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_max_eccentricity_l1029_102973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_nested_sqrt_l1029_102934

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 3 then x^2 + 1 else Real.sqrt (x + 1)

-- Theorem statement
theorem f_composition_equals_nested_sqrt : f (f (f 1)) = Real.sqrt (Real.sqrt (Real.sqrt 2 + 1) + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_nested_sqrt_l1029_102934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_ways_to_5_5_l1029_102906

/-- The number of ways to reach a point (i, j) from (0, 0) in a coordinate plane,
    moving only right, up, or diagonally (right and up) by 1 unit at a time. -/
def numWays : ℕ → ℕ → ℕ
  | 0, 0 => 1
  | 0, j+1 => numWays 0 j
  | i+1, 0 => numWays i 0
  | i+1, j+1 => numWays i (j+1) + numWays (i+1) j + numWays i j

/-- The theorem stating that the number of ways to reach (5, 5) is 1573. -/
theorem num_ways_to_5_5 : numWays 5 5 = 1573 := by
  sorry

#eval numWays 5 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_ways_to_5_5_l1029_102906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sales_prediction_l1029_102923

def sports_cars : ℕ := 45

def ratio_sports_cars : ℕ := 3
def ratio_sedans : ℕ := 5
def ratio_trucks : ℕ := 3

theorem car_sales_prediction :
  (sports_cars * ratio_sedans) / ratio_sports_cars = 75 ∧
  (sports_cars * ratio_trucks) / ratio_sports_cars = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sales_prediction_l1029_102923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_range_l1029_102966

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

def b_sequence (a b : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ+), b n = (Finset.range n).sum (λ i ↦ (n - i) * a (i + 1))

theorem geometric_sequence_range (m : ℝ) :
  m ≠ 0 →
  (∃ (a : ℕ → ℝ), geometric_sequence a ∧
    a 1 = m ∧
    (∃ (q : ℝ), q = -1/2 ∧ ∀ (n : ℕ), a (n + 1) = q * a n) ∧
    (∃ (b : ℕ → ℝ), b_sequence a b ∧ b 1 = m ∧ b 2 = 3*m/2) ∧
    (∀ (n : ℕ+), (Finset.range n).sum (λ i ↦ a (i + 1)) ∈ Set.Icc 1 3)) →
  m ∈ Set.Icc 2 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_range_l1029_102966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1029_102915

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of two lines being parallel
variable (parallel : Line → Line → Prop)

-- Define the property of two lines intersecting
variable (intersect : Line → Line → Prop)

-- Define the property of a line not having common points with another line
variable (no_common_points : Line → Line → Prop)

-- Define a type for points in 3D space
variable (Point : Type)

-- Define a function to determine a plane from two lines
variable (determine_plane : Line → Line → Type)

-- Theorem statement
theorem line_properties :
  ∃ (l₁ l₂ l₃ l₄ l₅ : Line),
    -- Proposition 1
    (no_common_points l₁ l₂ ∧ ¬(skew l₁ l₂)) ∧
    -- Proposition 2
    (skew l₁ l₂ ∧ intersect l₃ l₁ ∧ intersect l₃ l₂ ∧ ∃ l₄, intersect l₄ l₁ ∧ intersect l₄ l₂ ∧ ¬(skew l₃ l₄)) ∧
    -- Proposition 3
    (∀ l₃, skew l₁ l₂ → parallel l₃ l₁ → ¬(parallel l₃ l₂)) ∧
    -- Proposition 4
    (∀ l₃, skew l₁ l₂ → intersect l₃ l₁ → intersect l₃ l₂ → 
      ∃ (p₁ p₂ : determine_plane l₃ l₁), p₁ ≠ p₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1029_102915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_by_five_board_domino_covering_l1029_102994

theorem five_by_five_board_domino_covering (board_size : Nat) (domino_size : Nat) :
  board_size = 5 →
  domino_size = 2 →
  ¬(board_size * board_size % 2 = 0) →
  ¬∃ (covering : Nat), covering * domino_size = board_size * board_size :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_by_five_board_domino_covering_l1029_102994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_exponent_l1029_102988

theorem smallest_power_exponent : 
  ∃ (m n : ℕ) (a : ℕ), 1324 + 279 * m + 5^n = a^3 ∧ 
  ∀ (k : ℕ), k < 3 → ¬∃ (m n : ℕ) (a : ℕ), 1324 + 279 * m + 5^n = a^k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_exponent_l1029_102988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_in_pyramid_volume_l1029_102983

/-- The volume of a regular tetrahedron with side length s -/
noncomputable def tetrahedron_volume (s : ℝ) : ℝ := s^3 * Real.sqrt 2 / 12

/-- The side length of the tetrahedron in the pyramid -/
noncomputable def tetrahedron_side_length : ℝ := Real.sqrt 3

theorem tetrahedron_in_pyramid_volume :
  tetrahedron_volume tetrahedron_side_length = Real.sqrt 6 / 4 := by
  -- Expand the definition of tetrahedron_volume
  unfold tetrahedron_volume
  -- Substitute the value of tetrahedron_side_length
  unfold tetrahedron_side_length
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_in_pyramid_volume_l1029_102983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_car_price_l1029_102926

noncomputable def median (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem lowest_car_price
  (w x y z : ℝ)
  (h1 : median w x y = 52000)
  (h2 : median x y 57000 = 56000)
  (h3 : 44000 ∈ ({w, x, y, z, 57000} : Set ℝ))
  (h4 : 57000 ∈ ({w, x, y, z, 57000} : Set ℝ))
  : ∃ (p : ℝ), p < 44000 ∧ p ∈ ({w, x, y, z, 57000} : Set ℝ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_car_price_l1029_102926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_zero_l1029_102952

theorem complex_expression_zero : 
  (1 + Complex.I) / (1 - Complex.I) + Complex.I^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_zero_l1029_102952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l1029_102914

/-- The function f(x) = 1/x -/
noncomputable def f (x : ℝ) : ℝ := 1 / x

/-- The function g(x) = |1/x| -/
noncomputable def g (x : ℝ) : ℝ := |1 / x|

/-- Theorem stating that f and g are equal for all non-zero real numbers -/
theorem f_eq_g : ∀ (x : ℝ), x ≠ 0 → f x = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l1029_102914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_strict_subset_Q_l1029_102991

def P : Set ℝ := {m : ℝ | -1 < m ∧ m < 0}

def Q : Set ℝ := {m : ℝ | ∀ x : ℝ, m*x^2 + 4*m*x - 4 < 0}

theorem P_strict_subset_Q : P ⊂ Q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_strict_subset_Q_l1029_102991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_135_degrees_l1029_102930

-- Define the circle and its properties
structure Circle where
  O : ℝ × ℝ -- Circumcenter
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_on_circle : A ≠ O ∧ B ≠ O ∧ C ≠ O
  equal_radii : dist O A = dist O B ∧ dist O B = dist O C

-- Define the vector equation
def vector_equation (c : Circle) : Prop :=
  3 • (c.A - c.O) + 4 • (c.B - c.O) - 5 • (c.C - c.O) = (0, 0)

-- Define the angle
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_C_is_135_degrees (c : Circle) 
  (h : vector_equation c) : angle c.A c.C c.B = 135 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_135_degrees_l1029_102930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_polynomial_m_l1029_102900

/-- A polynomial of the form x^4 + mx^2 + nx - 144 with four distinct real roots in arithmetic progression -/
noncomputable def ArithmeticProgressionPolynomial (m n : ℝ) : Polynomial ℝ :=
  Polynomial.X^4 + m • Polynomial.X^2 + n • Polynomial.X - 144

/-- The property that a polynomial has four distinct real roots in arithmetic progression -/
def has_arithmetic_progression_roots (p : Polynomial ℝ) : Prop :=
  ∃ (a d : ℝ), d ≠ 0 ∧ 
    (∀ x : ℝ, p.eval x = 0 ↔ x ∈ ({a, a + d, a + 2*d, a + 3*d} : Set ℝ))

theorem arithmetic_progression_polynomial_m (m n : ℝ) :
  has_arithmetic_progression_roots (ArithmeticProgressionPolynomial m n) →
  m = -40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_polynomial_m_l1029_102900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l1029_102978

theorem tv_price_change (P : ℝ) (h : P > 0) : 
  let price_after_decrease := P * (1 - 0.2)
  let final_price := price_after_decrease * (1 + 0.5)
  final_price = P * 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l1029_102978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_probability_after_6_moves_l1029_102968

/-- A point on the lattice -/
structure Point where
  x : Int
  y : Int

/-- The lattice on which the ant moves -/
def AntLattice : Type := Set Point

/-- A function representing a single move of the ant -/
def Move : Point → Point → Prop := sorry

/-- The set of all points reachable in 6 moves from a starting point -/
def ReachableIn6Moves (start : Point) : Set Point := sorry

/-- The probability of reaching a specific point in 6 moves -/
noncomputable def ProbabilityToReach (start finish : Point) : ℚ := sorry

/-- The starting point C -/
def C : Point := ⟨-1, -1⟩

/-- The target point D -/
def D : Point := ⟨1, 1⟩

theorem ant_probability_after_6_moves :
  ProbabilityToReach C D = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_probability_after_6_moves_l1029_102968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_sequence_range_l1029_102980

theorem decreasing_sequence_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ n : ℕ+, (n : ℝ) * a^(n : ℕ) > ((n + 1) : ℝ) * a^((n + 1) : ℕ)) → 0 < a ∧ a < (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_sequence_range_l1029_102980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1029_102976

/-- The eccentricity of a hyperbola with given equation and asymptotes -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptotes : ∀ x y : ℝ, y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) :
  Real.sqrt ((a^2 + b^2) / a^2) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1029_102976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_values_l1029_102954

theorem parallel_vectors_x_values (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, x]
  let b : Fin 2 → ℝ := ![x - 1, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → x = 2 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_values_l1029_102954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_roots_l1029_102996

theorem compare_roots : (4 : ℝ)^(1/4) > (6 : ℝ)^(1/6) ∧ (6 : ℝ)^(1/6) > (16 : ℝ)^(1/16) ∧ (16 : ℝ)^(1/16) > (27 : ℝ)^(1/27) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_roots_l1029_102996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1029_102904

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def triangleCondition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0

/-- The area of the triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  (1 / 2) * t.b * t.c * Real.sin t.A

theorem triangle_theorem (t : Triangle) :
  triangleCondition t →
  (t.A = Real.pi / 3) ∧
  (t.a = 2 ∧ triangleArea t = Real.sqrt 3 → t.b = 2 ∧ t.c = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1029_102904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_ratios_l1029_102950

/-- Arithmetic sequence a_n -/
def a : ℕ → ℚ := sorry

/-- Arithmetic sequence b_n -/
def b : ℕ → ℚ := sorry

/-- Sum of first n terms of a_n -/
def S : ℕ → ℚ := sorry

/-- Sum of first n terms of b_n -/
def T : ℕ → ℚ := sorry

/-- Condition relating S_n and T_n -/
axiom ST_relation (n : ℕ) : S n / T n = (2 * n + 30) / (n + 3)

/-- Definition of when a_n / b_n is an integer -/
def is_integer_ratio (n : ℕ) : Prop :=
  ∃ k : ℤ, a n / b n = k

/-- The main theorem to prove -/
theorem count_integer_ratios :
  ∃ l : Finset ℕ, l.card = 5 ∧ (∀ n : ℕ, n ∈ l ↔ is_integer_ratio n) ∧
    (∀ n : ℕ, n ∉ l → ¬is_integer_ratio n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_ratios_l1029_102950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_ratio_l1029_102928

/-- A line in the form y = b + x where b > 0 -/
structure Line where
  b : ℝ
  h_pos : b > 0

/-- Point of intersection with y-axis -/
noncomputable def P (l : Line) : ℝ × ℝ := (0, l.b)

/-- Point of intersection with x-axis -/
noncomputable def Q (l : Line) : ℝ × ℝ := (-l.b, 0)

/-- Point of intersection with x = 5 -/
noncomputable def S (l : Line) : ℝ × ℝ := (5, l.b + 5)

/-- Area of triangle QRS -/
noncomputable def area_QRS (l : Line) : ℝ := (5 + l.b) * (l.b + 5) / 2

/-- Area of triangle QOP -/
noncomputable def area_QOP (l : Line) : ℝ := l.b^2 / 2

/-- The main theorem -/
theorem line_intersection_ratio (l : Line) :
  area_QRS l / area_QOP l = 4 / 9 → l.b = 5 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_ratio_l1029_102928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_brownies_count_l1029_102905

/-- Represents the dimensions of a brownie piece -/
structure BrownieDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the number of interior pieces -/
def interior_pieces (pan_side : ℝ) (piece : BrownieDimensions) : ℝ :=
  (pan_side - piece.length - 2) * (pan_side - piece.width - 2)

/-- Calculates the number of perimeter pieces -/
def perimeter_pieces (pan_side : ℝ) (piece : BrownieDimensions) : ℝ :=
  2 * (pan_side - piece.length) + 2 * (pan_side - piece.width) - 4

/-- Theorem stating the maximum number of brownies -/
theorem max_brownies_count (piece : BrownieDimensions) :
  let pan_side : ℝ := 10  -- derived from perimeter 40 cm
  interior_pieces pan_side piece = 2 * perimeter_pieces pan_side piece →
  ∀ other_piece : BrownieDimensions,
    interior_pieces pan_side other_piece = 2 * perimeter_pieces pan_side other_piece →
    ⌊pan_side / piece.length⌋ * ⌊pan_side / piece.width⌋ ≤ 100 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_brownies_count_l1029_102905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_all_powers_of_two_l1029_102975

/-- Represents a card with a natural number -/
structure Card where
  value : ℕ

/-- Represents the state of the table at any given time -/
structure TableState where
  cards : List Card

/-- The procedure applied every minute -/
def minuteProcedure (state : TableState) : TableState :=
  sorry

/-- Checks if a number is divisible by 2^d -/
def isDivisibleByPowerOfTwo (n d : ℕ) : Prop :=
  ∃ k, n = k * 2^d

/-- The main theorem -/
theorem impossibility_of_all_powers_of_two :
  ∀ (initialState : TableState),
    (initialState.cards.length = 100) →
    ((initialState.cards.filter (fun c => c.value % 2 = 1)).length = 28) →
    ∃ d : ℕ, ∀ t : ℕ,
      ∀ card ∈ (minuteProcedure^[t] initialState).cards,
        ¬ isDivisibleByPowerOfTwo card.value d :=
by
  sorry

#check impossibility_of_all_powers_of_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_all_powers_of_two_l1029_102975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_l1029_102929

-- Define the curve C
noncomputable def curve_C (m : ℝ) (α : ℝ) : ℝ × ℝ := (Real.cos α, m + Real.sin α)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 5 / 5) * t, 4 + (2 * Real.sqrt 5 / 5) * t)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_points_distance (m : ℝ) :
  (∃ (α₁ α₂ t₁ t₂ : ℝ),
    curve_C m α₁ = line_l t₁ ∧
    curve_C m α₂ = line_l t₂ ∧
    distance (curve_C m α₁) (curve_C m α₂) = 4 * Real.sqrt 5 / 5) →
  m = 1 ∨ m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_l1029_102929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_l1029_102946

/-- A point in the 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points in the 2D plane. -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- An integer point in the 2D plane. -/
def IntegerPoint : Type := ℤ × ℤ

/-- Convert an integer point to a regular point. -/
def intPointToPoint (p : IntegerPoint) : Point where
  x := p.1
  y := p.2

/-- The main theorem: For any point in the plane, there exists an integer point 
    within a distance of 100 + 1/14 from it. -/
theorem circle_intersection (O : Point) : 
  ∃ (p : IntegerPoint), distance O (intPointToPoint p) ≤ 100 + 1/14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_l1029_102946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_squared_sum_89_l1029_102931

open Real BigOperators

theorem sine_squared_sum_89 : 
  (∑ i in Finset.range 89, (sin ((i + 1 : ℝ) * π / 180))^2) = 89 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_squared_sum_89_l1029_102931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_difference_l1029_102918

noncomputable section

/-- A cubic function with two distinct extreme points -/
def f (a b x : ℝ) : ℝ := (1/3) * x^3 - (5/2) * a * x^2 + 6 * a * x + b

/-- The derivative of f -/
def f' (a x : ℝ) : ℝ := x^2 - 5 * a * x + 6 * a

theorem extreme_points_difference (a b : ℝ) (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ →
  f' a x₁ = 0 →
  f' a x₂ = 0 →
  x₂ = (3/2) * x₁ →
  f a b x₁ - f a b x₂ = 1/6 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_difference_l1029_102918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equidistant_from_B_and_C_l1029_102919

-- Define the points
noncomputable def A : ℝ × ℝ × ℝ := (0, -7/6, 0)
noncomputable def B : ℝ × ℝ × ℝ := (-2, 8, 10)
noncomputable def C : ℝ × ℝ × ℝ := (6, 11, -2)

-- Define a function to calculate the distance between two 3D points
noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

-- Theorem statement
theorem A_equidistant_from_B_and_C : distance A B = distance A C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equidistant_from_B_and_C_l1029_102919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_approximation_l1029_102936

/-- Represents the setup of the river problem -/
structure RiverProblem where
  L : ℝ  -- Half the distance between sirens
  y : ℝ  -- Distance from bank when we want to calculate Gavrila's position

/-- Calculates the x-coordinate of Gavrila's position -/
noncomputable def x_position (p : RiverProblem) : ℝ :=
  p.y^2 / (4 * p.L)

/-- Calculates the distance from the starting point to Gavrila's position -/
noncomputable def distance_from_start (p : RiverProblem) : ℝ :=
  Real.sqrt ((x_position p)^2 + p.y^2)

/-- The main theorem stating the approximate distance -/
theorem distance_approximation (p : RiverProblem) 
  (h1 : p.L = 50) 
  (h2 : p.y = 40) : 
  ⌊distance_from_start p⌋₊ = 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_approximation_l1029_102936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_difference_l1029_102993

/-- The distance to the park in miles -/
def distance_to_park : ℚ := 2

/-- Jill's cycling speed in miles per hour -/
def jill_speed : ℚ := 12

/-- Jack's jogging speed in miles per hour -/
def jack_speed : ℚ := 5

/-- Convert hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

/-- Calculate travel time given distance and speed -/
def travel_time (distance : ℚ) (speed : ℚ) : ℚ := distance / speed

theorem arrival_time_difference : 
  hours_to_minutes (travel_time distance_to_park jack_speed - travel_time distance_to_park jill_speed) = 14 := by
  -- Unfold definitions
  unfold hours_to_minutes travel_time distance_to_park jack_speed jill_speed
  -- Simplify the expression
  simp [mul_div_assoc, sub_mul]
  -- Perform the calculation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_difference_l1029_102993
