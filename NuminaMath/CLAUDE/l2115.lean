import Mathlib

namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l2115_211544

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → ¬((2 ∣ m) ∧ (3 ∣ m) ∧ (5 ∣ m) ∧ (7 ∣ m))) ∧
  n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l2115_211544


namespace NUMINAMATH_CALUDE_bus_problem_l2115_211579

/-- The number of people who boarded at the second stop of a bus journey --/
def second_stop_boarders (
  rows : ℕ) (seats_per_row : ℕ) 
  (initial_passengers : ℕ) 
  (first_stop_on : ℕ) (first_stop_off : ℕ)
  (second_stop_off : ℕ) 
  (final_empty_seats : ℕ) : ℕ := by
  sorry

/-- The number of people who boarded at the second stop is 17 --/
theorem bus_problem : 
  second_stop_boarders 23 4 16 15 3 10 57 = 17 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l2115_211579


namespace NUMINAMATH_CALUDE_proposition_truth_l2115_211519

-- Define the propositions P and q
def P : Prop := ∀ x y : ℝ, x > y → -x > -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Define the compound propositions
def prop1 : Prop := P ∧ q
def prop2 : Prop := ¬P ∨ ¬q
def prop3 : Prop := P ∧ ¬q
def prop4 : Prop := ¬P ∨ q

-- Theorem statement
theorem proposition_truth : 
  ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 :=
sorry

end NUMINAMATH_CALUDE_proposition_truth_l2115_211519


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2115_211532

theorem quadratic_inequality_solution (b c : ℝ) :
  (∀ x, x^2 + b*x + c < 0 ↔ 1 < x ∧ x < 2) →
  (∀ x, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = 2) →
  b + c = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2115_211532


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l2115_211557

/-- Represents a 4x4x4 cube composed of smaller cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  diagonal_shaded : Bool

/-- Represents the shading pattern on the faces of the large cube -/
structure ShadingPattern where
  diagonal : Bool
  opposite_faces_identical : Bool

/-- Counts the number of smaller cubes with at least one shaded face -/
def count_shaded_cubes (cube : LargeCube) (pattern : ShadingPattern) : Nat :=
  sorry

/-- The main theorem stating that 32 smaller cubes are shaded -/
theorem shaded_cubes_count (cube : LargeCube) (pattern : ShadingPattern) :
  cube.size = 4 ∧ 
  cube.total_cubes = 64 ∧ 
  cube.diagonal_shaded = true ∧
  pattern.diagonal = true ∧
  pattern.opposite_faces_identical = true →
  count_shaded_cubes cube pattern = 32 :=
sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l2115_211557


namespace NUMINAMATH_CALUDE_gift_shop_combinations_l2115_211536

/-- The number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 8

/-- The number of colors of ribbon -/
def ribbon_colors : ℕ := 3

/-- The number of types of gift cards -/
def gift_card_types : ℕ := 5

/-- The number of varieties of stickers -/
def sticker_varieties : ℕ := 5

/-- The total number of possible combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_card_types * sticker_varieties

theorem gift_shop_combinations : total_combinations = 600 := by
  sorry

end NUMINAMATH_CALUDE_gift_shop_combinations_l2115_211536


namespace NUMINAMATH_CALUDE_x_equation_implies_y_values_l2115_211518

theorem x_equation_implies_y_values (x : ℝ) :
  x^2 + 9 * (x / (x - 3))^2 = 54 →
  let y := ((x - 3)^2 * (x + 4)) / (2 * x - 4)
  y = 11.25 ∨ y = 10.125 := by
sorry

end NUMINAMATH_CALUDE_x_equation_implies_y_values_l2115_211518


namespace NUMINAMATH_CALUDE_existence_of_special_number_l2115_211525

/-- A function that computes the sum of digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number contains only digits 2-9. -/
def contains_only_2_to_9 (n : ℕ) : Prop := sorry

/-- The number of digits in a natural number. -/
def num_digits (n : ℕ) : ℕ := sorry

theorem existence_of_special_number :
  ∃ N : ℕ, 
    num_digits N = 2020 ∧ 
    contains_only_2_to_9 N ∧ 
    N % sum_of_digits N = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l2115_211525


namespace NUMINAMATH_CALUDE_eccentricity_of_parametric_ellipse_l2115_211591

/-- Eccentricity of an ellipse defined by parametric equations -/
theorem eccentricity_of_parametric_ellipse :
  let x : ℝ → ℝ := λ φ ↦ 3 * Real.cos φ
  let y : ℝ → ℝ := λ φ ↦ Real.sqrt 5 * Real.sin φ
  let a : ℝ := 3
  let b : ℝ := Real.sqrt 5
  let c : ℝ := Real.sqrt (a^2 - b^2)
  c / a = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_eccentricity_of_parametric_ellipse_l2115_211591


namespace NUMINAMATH_CALUDE_tangent_slope_at_negative_five_l2115_211508

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem tangent_slope_at_negative_five
  (f : ℝ → ℝ)
  (hf_even : is_even f)
  (hf_diff : Differentiable ℝ f)
  (hf_deriv_one : deriv f 1 = 1)
  (hf_periodic : ∀ x, f (x + 2) = f (x - 2)) :
  deriv f (-5) = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_negative_five_l2115_211508


namespace NUMINAMATH_CALUDE_journey_equations_l2115_211510

/-- Represents a journey between two points with an uphill and a flat section -/
structure Journey where
  uphill_length : ℝ  -- Length of uphill section in km
  flat_length : ℝ    -- Length of flat section in km
  uphill_speed : ℝ   -- Speed on uphill section in km/h
  flat_speed : ℝ     -- Speed on flat section in km/h
  downhill_speed : ℝ -- Speed on downhill section in km/h
  time_ab : ℝ        -- Time from A to B in minutes
  time_ba : ℝ        -- Time from B to A in minutes

/-- The correct system of equations for the journey -/
def correct_equations (j : Journey) : Prop :=
  (j.uphill_length / j.uphill_speed + j.flat_length / j.flat_speed = j.time_ab / 60) ∧
  (j.uphill_length / j.downhill_speed + j.flat_length / j.flat_speed = j.time_ba / 60)

/-- Theorem stating that the given journey satisfies the correct system of equations -/
theorem journey_equations (j : Journey) 
  (h1 : j.uphill_speed = 3)
  (h2 : j.flat_speed = 4)
  (h3 : j.downhill_speed = 5)
  (h4 : j.time_ab = 54)
  (h5 : j.time_ba = 42) :
  correct_equations j := by
  sorry

end NUMINAMATH_CALUDE_journey_equations_l2115_211510


namespace NUMINAMATH_CALUDE_vectors_collinear_l2115_211552

/-- Given vectors a and b in ℝ³, prove that c₁ and c₂ are collinear -/
theorem vectors_collinear (a b : Fin 3 → ℝ) 
  (ha : a = ![1, -2, 5])
  (hb : b = ![3, -1, 0])
  (c₁ : Fin 3 → ℝ) (hc₁ : c₁ = 4 • a - 2 • b)
  (c₂ : Fin 3 → ℝ) (hc₂ : c₂ = b - 2 • a) :
  ∃ k : ℝ, c₁ = k • c₂ := by
sorry

end NUMINAMATH_CALUDE_vectors_collinear_l2115_211552


namespace NUMINAMATH_CALUDE_solve_fish_problem_l2115_211585

def fish_problem (total_spent : ℕ) (cost_per_fish : ℕ) (fish_for_dog : ℕ) : Prop :=
  let total_fish : ℕ := total_spent / cost_per_fish
  let fish_for_cat : ℕ := total_fish - fish_for_dog
  (fish_for_cat : ℚ) / fish_for_dog = 1 / 2

theorem solve_fish_problem :
  fish_problem 240 4 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_fish_problem_l2115_211585


namespace NUMINAMATH_CALUDE_hyperbola_and_intersecting_line_l2115_211516

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1, real axis length 2√3, and one focus at (-√5, 0),
    prove its equation and find the equation of a line intersecting it. -/
theorem hyperbola_and_intersecting_line 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (real_axis_length : ℝ) 
  (focus : ℝ × ℝ) 
  (hreal_axis : real_axis_length = 2 * Real.sqrt 3) 
  (hfocus : focus = (-Real.sqrt 5, 0)) :
  (∃ (x y : ℝ), x^2 / 3 - y^2 / 2 = 1) ∧ 
  (∃ (m : ℝ), (m = Real.sqrt 210 / 3 ∨ m = -Real.sqrt 210 / 3) ∧
    ∀ (x y : ℝ), y = 2 * x + m → 
      (∃ (A B : ℝ × ℝ), A ≠ B ∧ 
        (A.1^2 / 3 - A.2^2 / 2 = 1) ∧ 
        (B.1^2 / 3 - B.2^2 / 2 = 1) ∧
        (A.2 = 2 * A.1 + m) ∧ 
        (B.2 = 2 * B.1 + m) ∧
        (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_intersecting_line_l2115_211516


namespace NUMINAMATH_CALUDE_complement_intersection_equality_l2115_211592

def I : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5}

theorem complement_intersection_equality :
  (I \ N) ∩ M = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equality_l2115_211592


namespace NUMINAMATH_CALUDE_corresponding_angles_equal_l2115_211560

theorem corresponding_angles_equal (α β γ : ℝ) : 
  α + β + γ = 180 → 
  (180 - α) + β + γ = 180 → 
  α = 180 - α ∧ β = β ∧ γ = γ := by
sorry

end NUMINAMATH_CALUDE_corresponding_angles_equal_l2115_211560


namespace NUMINAMATH_CALUDE_lowest_unique_score_above_100_l2115_211513

/-- Represents the scoring system and conditions of the math competition. -/
structure MathCompetition where
  total_questions : Nat
  base_score : Nat
  correct_points : Nat
  wrong_points : Nat
  score : Nat

/-- Checks if a given score is valid for the math competition. -/
def is_valid_score (comp : MathCompetition) (correct wrong : Nat) : Prop :=
  correct + wrong ≤ comp.total_questions ∧
  comp.score = comp.base_score + comp.correct_points * correct - comp.wrong_points * wrong

/-- Checks if a score has a unique solution for correct and wrong answers. -/
def has_unique_solution (comp : MathCompetition) : Prop :=
  ∃! (correct wrong : Nat), is_valid_score comp correct wrong

/-- The main theorem stating that 150 is the lowest score above 100 with a unique solution. -/
theorem lowest_unique_score_above_100 : 
  let comp : MathCompetition := {
    total_questions := 50,
    base_score := 50,
    correct_points := 5,
    wrong_points := 2,
    score := 150
  }
  (comp.score > 100) ∧ 
  has_unique_solution comp ∧
  ∀ (s : Nat), 100 < s ∧ s < comp.score → 
    ¬(has_unique_solution {comp with score := s}) := by
  sorry

end NUMINAMATH_CALUDE_lowest_unique_score_above_100_l2115_211513


namespace NUMINAMATH_CALUDE_seashell_ratio_l2115_211529

theorem seashell_ratio : 
  let henry_shells : ℕ := 11
  let paul_shells : ℕ := 24
  let initial_total : ℕ := 59
  let final_total : ℕ := 53
  let leo_initial : ℕ := initial_total - henry_shells - paul_shells
  let leo_gave_away : ℕ := initial_total - final_total
  (leo_gave_away : ℚ) / leo_initial = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_seashell_ratio_l2115_211529


namespace NUMINAMATH_CALUDE_tiles_per_row_l2115_211548

/-- Proves that a square room with an area of 144 square feet,
    when covered with 8-inch by 8-inch tiles, will have 18 tiles in each row. -/
theorem tiles_per_row (room_area : ℝ) (tile_size : ℝ) :
  room_area = 144 →
  tile_size = 8 →
  (Real.sqrt room_area * 12) / tile_size = 18 := by
  sorry

end NUMINAMATH_CALUDE_tiles_per_row_l2115_211548


namespace NUMINAMATH_CALUDE_hyperbola_intersection_perpendicular_l2115_211543

-- Define the hyperbola C₁
def C₁ (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a line with slope 1
def Line (x y b : ℝ) : Prop := y = x + b

-- Define the tangency condition
def IsTangent (b : ℝ) : Prop := b^2 = 2

-- Define the perpendicularity of two vectors
def Perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_intersection_perpendicular 
  (x₁ y₁ x₂ y₂ b : ℝ) : 
  C₁ x₁ y₁ → C₁ x₂ y₂ → 
  Line x₁ y₁ b → Line x₂ y₂ b → 
  IsTangent b → 
  Perpendicular x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_perpendicular_l2115_211543


namespace NUMINAMATH_CALUDE_catch_up_time_l2115_211574

-- Define the speeds of the girl, young man, and tram
def girl_speed : ℝ := 1
def young_man_speed : ℝ := 2 * girl_speed
def tram_speed : ℝ := 5 * young_man_speed

-- Define the time the young man waits before exiting the tram
def wait_time : ℝ := 8

-- Define the theorem
theorem catch_up_time : 
  ∀ (t : ℝ), 
  (girl_speed * wait_time + tram_speed * wait_time + girl_speed * t = young_man_speed * t) → 
  t = 88 := by
  sorry

end NUMINAMATH_CALUDE_catch_up_time_l2115_211574


namespace NUMINAMATH_CALUDE_circle_sum_bounds_l2115_211588

/-- The circle defined by the equation x² + y² - 4x + 2 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 + 2 = 0}

/-- The sum function x + y for points (x, y) on the circle -/
def sum_func (p : ℝ × ℝ) : ℝ := p.1 + p.2

theorem circle_sum_bounds :
  ∃ (min max : ℝ), min = 0 ∧ max = 4 ∧
  ∀ p ∈ Circle, min ≤ sum_func p ∧ sum_func p ≤ max :=
sorry

end NUMINAMATH_CALUDE_circle_sum_bounds_l2115_211588


namespace NUMINAMATH_CALUDE_largest_number_l2115_211526

theorem largest_number (π : ℝ) (h1 : π > 3) : π = max π (max (Real.sqrt 2) (max (-2) 3)) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2115_211526


namespace NUMINAMATH_CALUDE_initial_books_l2115_211551

theorem initial_books (initial : ℕ) (sold : ℕ) (bought : ℕ) (final : ℕ) : 
  sold = 11 → bought = 23 → final = 45 → initial - sold + bought = final → initial = 33 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_l2115_211551


namespace NUMINAMATH_CALUDE_other_person_speed_l2115_211568

/-- Proves that given Roja's speed and the final distance after a certain time, 
    the other person's speed can be determined. -/
theorem other_person_speed 
  (roja_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : roja_speed = 2) 
  (h2 : time = 4) 
  (h3 : final_distance = 20) : 
  ∃ other_speed : ℝ, 
    other_speed = 3 ∧ 
    final_distance = (roja_speed + other_speed) * time :=
by
  sorry

#check other_person_speed

end NUMINAMATH_CALUDE_other_person_speed_l2115_211568


namespace NUMINAMATH_CALUDE_function_transformation_l2115_211512

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_transformation (x : ℝ) : f (x + 1) = 3 * x + 2 → f x = 3 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l2115_211512


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l2115_211563

/-- 
Given a mixture of 1 liter of pure water and x liters of 30% salt solution,
resulting in a 15% salt solution, prove that x = 1.
-/
theorem salt_solution_mixture (x : ℝ) : 
  (0.30 * x = 0.15 * (1 + x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l2115_211563


namespace NUMINAMATH_CALUDE_translate_sum_zero_l2115_211565

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally and vertically -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem translate_sum_zero :
  let A : Point := ⟨-1, 2⟩
  let B : Point := translate (translate A 1 0) 0 (-2)
  B.x + B.y = 0 := by sorry

end NUMINAMATH_CALUDE_translate_sum_zero_l2115_211565


namespace NUMINAMATH_CALUDE_exists_interior_rectangle_l2115_211562

/-- A rectangle in a square partition -/
structure Rectangle where
  left : ℝ
  right : ℝ
  bottom : ℝ
  top : ℝ
  left_lt_right : left < right
  bottom_lt_top : bottom < top

/-- A partition of a square into rectangles -/
structure SquarePartition where
  rectangles : List Rectangle
  n_gt_one : rectangles.length > 1
  covers_square : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
    ∃ r ∈ rectangles, r.left ≤ x ∧ x ≤ r.right ∧ r.bottom ≤ y ∧ y ≤ r.top
  intersects_line : ∀ l : ℝ, 0 < l ∧ l < 1 →
    (∃ r ∈ rectangles, r.left < l ∧ l < r.right) ∧
    (∃ r ∈ rectangles, r.bottom < l ∧ l < r.top)

/-- A rectangle touches the side of the square if any of its sides coincide with the square's sides -/
def touches_side (r : Rectangle) : Prop :=
  r.left = 0 ∨ r.right = 1 ∨ r.bottom = 0 ∨ r.top = 1

/-- Main theorem: There exists a rectangle that doesn't touch the sides of the square -/
theorem exists_interior_rectangle (p : SquarePartition) :
  ∃ r ∈ p.rectangles, ¬touches_side r := by
  sorry

end NUMINAMATH_CALUDE_exists_interior_rectangle_l2115_211562


namespace NUMINAMATH_CALUDE_perfect_square_problem_l2115_211523

theorem perfect_square_problem :
  (∃ (x : ℕ), 7^2040 = x^2) ∧
  (∀ (x : ℕ), 8^2041 ≠ x^2) ∧
  (∃ (x : ℕ), 9^2042 = x^2) ∧
  (∃ (x : ℕ), 10^2043 = x^2) ∧
  (∃ (x : ℕ), 11^2044 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_problem_l2115_211523


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2115_211547

theorem complex_sum_problem (a b c d e f : ℂ) : 
  b = 1 →
  e = -a - 2*c →
  a + b * Complex.I + c + d * Complex.I + e + f * Complex.I = 3 + 2 * Complex.I →
  d + f = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2115_211547


namespace NUMINAMATH_CALUDE_degree_of_example_monomial_not_six_l2115_211537

/-- The degree of a monomial is the sum of the exponents of its variables -/
def monomial_degree (m : Polynomial ℤ) : ℕ := sorry

/-- A function to represent the monomial -2^2xab^2 -/
def example_monomial : Polynomial ℤ := sorry

theorem degree_of_example_monomial_not_six :
  monomial_degree example_monomial ≠ 6 := by sorry

end NUMINAMATH_CALUDE_degree_of_example_monomial_not_six_l2115_211537


namespace NUMINAMATH_CALUDE_mary_overtime_rate_increase_l2115_211549

/-- Represents Mary's work schedule and pay structure -/
structure WorkSchedule where
  max_hours : ℕ
  regular_hours : ℕ
  regular_rate : ℚ
  total_earnings : ℚ

/-- Calculates the percentage increase in overtime rate compared to regular rate -/
def overtime_rate_increase (w : WorkSchedule) : ℚ :=
  let regular_earnings := w.regular_hours * w.regular_rate
  let overtime_earnings := w.total_earnings - regular_earnings
  let overtime_hours := w.max_hours - w.regular_hours
  let overtime_rate := overtime_earnings / overtime_hours
  ((overtime_rate - w.regular_rate) / w.regular_rate) * 100

/-- Mary's work schedule -/
def mary_schedule : WorkSchedule :=
  { max_hours := 40
  , regular_hours := 20
  , regular_rate := 8
  , total_earnings := 360 }

/-- Theorem stating that Mary's overtime rate increase is 25% -/
theorem mary_overtime_rate_increase :
  overtime_rate_increase mary_schedule = 25 := by
  sorry

end NUMINAMATH_CALUDE_mary_overtime_rate_increase_l2115_211549


namespace NUMINAMATH_CALUDE_list_number_relation_l2115_211589

theorem list_number_relation (n : ℝ) (list : List ℝ) : 
  list.length = 21 ∧ 
  n ∈ list ∧
  n = (1 / 6 : ℝ) * list.sum →
  n = 4 * ((list.sum - n) / 20) := by
sorry

end NUMINAMATH_CALUDE_list_number_relation_l2115_211589


namespace NUMINAMATH_CALUDE_limit_a_over_3n_l2115_211514

def S (n : ℕ) : ℝ := -3 * (n ^ 2 : ℝ) + 2 * n + 1

def a (n : ℕ) : ℝ := S (n + 1) - S n

theorem limit_a_over_3n :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n / (3 * (n + 1)) + 2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_a_over_3n_l2115_211514


namespace NUMINAMATH_CALUDE_fermat_prime_l2115_211559

theorem fermat_prime (m : ℕ) (h : m > 0) :
  (2^(m+1) + 1) ∣ (3^(2^m) + 1) → Nat.Prime (2^(m+1) + 1) :=
by sorry

end NUMINAMATH_CALUDE_fermat_prime_l2115_211559


namespace NUMINAMATH_CALUDE_largest_remaining_circle_l2115_211520

/-- Represents a circle with a given diameter -/
structure Circle where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- The problem setup -/
def plywood_problem (initial : Circle) (cutout1 : Circle) (cutout2 : Circle) : Prop :=
  initial.diameter = 30 ∧ cutout1.diameter = 20 ∧ cutout2.diameter = 10

/-- The theorem to be proved -/
theorem largest_remaining_circle 
  (initial : Circle) (cutout1 : Circle) (cutout2 : Circle) 
  (h : plywood_problem initial cutout1 cutout2) : 
  ∃ (largest : Circle), largest.diameter = 30 / 7 ∧ 
  ∀ (c : Circle), c.diameter ≤ largest.diameter :=
sorry

end NUMINAMATH_CALUDE_largest_remaining_circle_l2115_211520


namespace NUMINAMATH_CALUDE_train_crossing_time_l2115_211599

/-- Given a train and a platform with specific dimensions and time to pass,
    calculate the time it takes for the train to cross a tree. -/
theorem train_crossing_time (train_length platform_length time_to_pass_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 1100)
  (h3 : time_to_pass_platform = 230) :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / time_to_pass_platform
  train_length / train_speed = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2115_211599


namespace NUMINAMATH_CALUDE_solve_inequality_1_solve_inequality_2_l2115_211598

-- Inequality 1
theorem solve_inequality_1 : 
  {x : ℝ | x^2 + x - 6 < 0} = {x : ℝ | -3 < x ∧ x < 2} := by sorry

-- Inequality 2
theorem solve_inequality_2 : 
  {x : ℝ | -6*x^2 - x + 2 ≤ 0} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 1/2} := by sorry

end NUMINAMATH_CALUDE_solve_inequality_1_solve_inequality_2_l2115_211598


namespace NUMINAMATH_CALUDE_contest_winner_l2115_211597

theorem contest_winner (n : ℕ) : 
  (∀ k : ℕ, k > 0 → n % 100 = 0 ∧ n % 40 = 0) → n ≥ 200 :=
sorry

end NUMINAMATH_CALUDE_contest_winner_l2115_211597


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2115_211595

theorem arithmetic_equality : (469138 * 9999) + (876543 * 12345) = 15512230997 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2115_211595


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2115_211527

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 118 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2115_211527


namespace NUMINAMATH_CALUDE_prob_two_co_captains_all_teams_l2115_211539

/-- Represents a math team with a given number of students and co-captains -/
structure MathTeam where
  students : Nat
  coCaptains : Nat
  h : coCaptains ≤ students

/-- Calculates the probability of choosing two co-captains from a given team -/
def probTwoCoCaptains (team : MathTeam) : Rat :=
  (Nat.choose team.coCaptains 2 : Rat) / (Nat.choose team.students 2 : Rat)

/-- The list of math teams in the area -/
def mathTeams : List MathTeam := [
  ⟨6, 3, by norm_num⟩,
  ⟨9, 2, by norm_num⟩,
  ⟨10, 4, by norm_num⟩
]

theorem prob_two_co_captains_all_teams : 
  (List.sum (mathTeams.map probTwoCoCaptains) / (mathTeams.length : Rat)) = 65 / 540 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_co_captains_all_teams_l2115_211539


namespace NUMINAMATH_CALUDE_birthday_stickers_proof_l2115_211504

/-- The number of stickers James had initially -/
def initial_stickers : ℕ := 39

/-- The number of stickers James had after his birthday -/
def final_stickers : ℕ := 61

/-- The number of stickers James got for his birthday -/
def birthday_stickers : ℕ := final_stickers - initial_stickers

theorem birthday_stickers_proof :
  birthday_stickers = final_stickers - initial_stickers :=
by sorry

end NUMINAMATH_CALUDE_birthday_stickers_proof_l2115_211504


namespace NUMINAMATH_CALUDE_area_of_specific_hexagon_l2115_211521

/-- Regular hexagon with vertices A and C -/
structure RegularHexagon where
  A : ℝ × ℝ
  C : ℝ × ℝ
  is_regular : True  -- This is a placeholder for the regularity condition

/-- The area of a regular hexagon -/
def area (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating the area of the specific regular hexagon -/
theorem area_of_specific_hexagon :
  ∃ (h : RegularHexagon),
    h.A = (0, 0) ∧
    h.C = (2 * Real.sqrt 3, 2) ∧
    area h = 6 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_area_of_specific_hexagon_l2115_211521


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2115_211569

theorem complex_equation_solution :
  ∀ z : ℂ, (z - Complex.I) / z = Complex.I → z = -1/2 + Complex.I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2115_211569


namespace NUMINAMATH_CALUDE_third_month_sale_l2115_211505

def sales_data : List ℕ := [8435, 8927, 9230, 8562, 6991]
def average_sale : ℕ := 8500
def num_months : ℕ := 6

theorem third_month_sale :
  ∃ x : ℕ, 
    (List.sum sales_data + x) / num_months = average_sale ∧
    x = 8855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l2115_211505


namespace NUMINAMATH_CALUDE_octal_54321_to_decimal_l2115_211556

/-- Converts a base-8 digit to its base-10 equivalent -/
def octalToDecimal (digit : ℕ) : ℕ := digit

/-- Computes the value of a digit in a specific position in base 8 -/
def octalDigitValue (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (8 ^ position)

/-- Theorem: The base-10 equivalent of 54321 in base-8 is 22737 -/
theorem octal_54321_to_decimal : 
  octalToDecimal 1 + 
  octalDigitValue 2 1 + 
  octalDigitValue 3 2 + 
  octalDigitValue 4 3 + 
  octalDigitValue 5 4 = 22737 :=
by sorry

end NUMINAMATH_CALUDE_octal_54321_to_decimal_l2115_211556


namespace NUMINAMATH_CALUDE_large_cube_edge_approx_l2115_211571

/-- The edge length of a smaller cube in centimeters -/
def small_cube_edge : ℝ := 20

/-- The approximate number of smaller cubes that fit in the larger cubical box -/
def num_small_cubes : ℝ := 125

/-- The approximate edge length of the larger cubical box in centimeters -/
def large_cube_edge : ℝ := 100

/-- Theorem stating that the edge length of the larger cubical box is approximately 100 cm -/
theorem large_cube_edge_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |large_cube_edge ^ 3 - num_small_cubes * small_cube_edge ^ 3| < ε * (num_small_cubes * small_cube_edge ^ 3) :=
sorry

end NUMINAMATH_CALUDE_large_cube_edge_approx_l2115_211571


namespace NUMINAMATH_CALUDE_exists_special_number_l2115_211596

def is_twelve_digit (n : ℕ) : Prop := 10^11 ≤ n ∧ n < 10^12

def is_not_perfect_square (n : ℕ) : Prop := ∃ (d : ℕ), n % 10 = d ∧ (d = 2 ∨ d = 3 ∨ d = 7 ∨ d = 8)

def is_ambiguous_cube (n : ℕ) : Prop := ∀ (d : ℕ), d < 10 → ∃ (k : ℕ), k^3 % 10 = d

theorem exists_special_number : ∃ (n : ℕ), 
  is_twelve_digit n ∧ 
  is_not_perfect_square n ∧ 
  is_ambiguous_cube n := by
  sorry

end NUMINAMATH_CALUDE_exists_special_number_l2115_211596


namespace NUMINAMATH_CALUDE_min_packs_for_144_cans_l2115_211522

/-- Represents the number of cans in each pack size --/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans for a given pack size --/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | .small => 8
  | .medium => 18
  | .large => 30

/-- Represents a combination of packs --/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans for a given pack combination --/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs for a given pack combination --/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- Theorem: The minimum number of packs needed to buy exactly 144 cans is 6 --/
theorem min_packs_for_144_cans :
  ∃ (c : PackCombination),
    totalCans c = 144 ∧
    totalPacks c = 6 ∧
    ∀ (d : PackCombination), totalCans d = 144 → totalPacks d ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_packs_for_144_cans_l2115_211522


namespace NUMINAMATH_CALUDE_integral_square_le_four_integral_derivative_square_l2115_211558

open MeasureTheory Interval RealInnerProductSpace

theorem integral_square_le_four_integral_derivative_square
  (f : ℝ → ℝ) (hf : ContDiff ℝ 1 f) (h : ∃ x₀ ∈ Set.Icc 0 1, f x₀ = 0) :
  ∫ x in Set.Icc 0 1, (f x)^2 ≤ 4 * ∫ x in Set.Icc 0 1, (deriv f x)^2 :=
sorry

end NUMINAMATH_CALUDE_integral_square_le_four_integral_derivative_square_l2115_211558


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l2115_211594

theorem company_picnic_attendance (men_attendance : Real) (women_attendance : Real) 
  (total_attendance : Real) :
  men_attendance = 0.2 →
  women_attendance = 0.4 →
  total_attendance = 0.30000000000000004 →
  ∃ (men_percentage : Real),
    men_percentage * men_attendance + (1 - men_percentage) * women_attendance = total_attendance ∧
    men_percentage = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l2115_211594


namespace NUMINAMATH_CALUDE_prob_all_red_is_one_third_l2115_211576

/-- Represents the number of red chips in the hat -/
def num_red : ℕ := 4

/-- Represents the number of green chips in the hat -/
def num_green : ℕ := 2

/-- Represents the total number of chips in the hat -/
def total_chips : ℕ := num_red + num_green

/-- Represents the probability of drawing all red chips before both green chips -/
def prob_all_red : ℚ := 1 / 3

/-- Theorem stating that the probability of drawing all red chips before both green chips is 1/3 -/
theorem prob_all_red_is_one_third :
  prob_all_red = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_all_red_is_one_third_l2115_211576


namespace NUMINAMATH_CALUDE_largest_a_less_than_l2115_211555

theorem largest_a_less_than (a b : ℤ) : 
  9 < a → 
  19 < b → 
  b < 31 → 
  (a : ℚ) / (b : ℚ) ≤ 2/3 → 
  a < 21 :=
by sorry

end NUMINAMATH_CALUDE_largest_a_less_than_l2115_211555


namespace NUMINAMATH_CALUDE_jewelry_price_increase_is_10_l2115_211564

/-- Represents the increase in price of jewelry -/
def jewelry_price_increase : ℝ := sorry

/-- Original price of jewelry -/
def original_jewelry_price : ℝ := 30

/-- Original price of paintings -/
def original_painting_price : ℝ := 100

/-- New price of paintings after 20% increase -/
def new_painting_price : ℝ := original_painting_price * 1.2

/-- Total cost for 2 pieces of jewelry and 5 paintings -/
def total_cost : ℝ := 680

theorem jewelry_price_increase_is_10 :
  2 * (original_jewelry_price + jewelry_price_increase) + 5 * new_painting_price = total_cost ∧
  jewelry_price_increase = 10 := by sorry

end NUMINAMATH_CALUDE_jewelry_price_increase_is_10_l2115_211564


namespace NUMINAMATH_CALUDE_magnitude_of_3_plus_i_squared_l2115_211561

theorem magnitude_of_3_plus_i_squared : 
  Complex.abs ((3 : ℂ) + Complex.I) ^ 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_3_plus_i_squared_l2115_211561


namespace NUMINAMATH_CALUDE_ellipse_max_sum_l2115_211567

/-- The maximum value of x + y for points on the ellipse x^2/16 + y^2/9 = 1 is 5 -/
theorem ellipse_max_sum (x y : ℝ) : 
  x^2/16 + y^2/9 = 1 → x + y ≤ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀^2/16 + y₀^2/9 = 1 ∧ x₀ + y₀ = 5 := by
  sorry

#check ellipse_max_sum

end NUMINAMATH_CALUDE_ellipse_max_sum_l2115_211567


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_maximum_marks_is_750_l2115_211500

theorem maximum_marks_calculation (passing_percentage : ℝ) (student_score : ℕ) (shortfall : ℕ) : ℝ :=
  let passing_score : ℕ := student_score + shortfall
  let maximum_marks : ℝ := passing_score / passing_percentage
  maximum_marks

-- Proof that the maximum marks is 750 given the conditions
theorem maximum_marks_is_750 :
  maximum_marks_calculation 0.3 212 13 = 750 :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_maximum_marks_is_750_l2115_211500


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2115_211570

/-- The angle between two vectors in degrees -/
def angle_between (u v : ℝ × ℝ) : ℝ := sorry

/-- The dot product of two 2D vectors -/
def dot_product (u v : ℝ × ℝ) : ℝ := sorry

/-- The magnitude (length) of a 2D vector -/
def magnitude (u : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, -4)
  ∀ c : ℝ × ℝ,
    magnitude c = Real.sqrt 5 →
    dot_product (a.1 + b.1, a.2 + b.2) c = 5/2 →
    angle_between a c = 120 := by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2115_211570


namespace NUMINAMATH_CALUDE_s_eight_value_l2115_211535

theorem s_eight_value (x : ℝ) (h : x + 1/x = 4) : 
  let S : ℕ → ℝ := λ m => x^m + 1/(x^m)
  S 8 = 37634 := by
  sorry

end NUMINAMATH_CALUDE_s_eight_value_l2115_211535


namespace NUMINAMATH_CALUDE_candy_probability_l2115_211573

/-- The probability of picking specific candies from a bag --/
theorem candy_probability : 
  let green : ℕ := 8
  let blue : ℕ := 5
  let red : ℕ := 9
  let yellow : ℕ := 10
  let pink : ℕ := 6
  let total : ℕ := green + blue + red + yellow + pink
  
  -- Probability of picking green first
  let p_green : ℚ := green / total
  
  -- Probability of picking yellow second
  let p_yellow : ℚ := yellow / (total - 1)
  
  -- Probability of picking pink third
  let p_pink : ℚ := pink / (total - 2)
  
  -- Overall probability
  let probability : ℚ := p_green * p_yellow * p_pink
  
  probability = 20 / 2109 := by
  sorry

end NUMINAMATH_CALUDE_candy_probability_l2115_211573


namespace NUMINAMATH_CALUDE_number_with_special_average_l2115_211566

theorem number_with_special_average (x : ℝ) (h1 : x ≠ 0) 
  (h2 : (x + x^2) / 2 = 5 * x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_with_special_average_l2115_211566


namespace NUMINAMATH_CALUDE_min_sum_given_product_min_sum_value_l2115_211538

theorem min_sum_given_product (x y : ℝ) (h1 : x * y = 4) (h2 : x > 0) (h3 : y > 0) :
  ∀ a b : ℝ, a * b = 4 ∧ a > 0 ∧ b > 0 → x + y ≤ a + b :=
by
  sorry

theorem min_sum_value (x y : ℝ) (h1 : x * y = 4) (h2 : x > 0) (h3 : y > 0) :
  ∃ M : ℝ, M = 4 ∧ x + y ≥ M :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_min_sum_value_l2115_211538


namespace NUMINAMATH_CALUDE_image_of_square_l2115_211501

/-- Transformation from xy-plane to uv-plane -/
def transform (x y : ℝ) : ℝ × ℝ :=
  (x^3 - y^3, x^2 * y^2)

/-- Square OABC in xy-plane -/
def square_vertices : List (ℝ × ℝ) :=
  [(0, 0), (2, 0), (2, 2), (0, 2)]

/-- Theorem: Image of square OABC in uv-plane -/
theorem image_of_square :
  (square_vertices.map (λ (x, y) => transform x y)) =
  [(0, 0), (8, 0), (0, 16), (-8, 0)] := by
  sorry


end NUMINAMATH_CALUDE_image_of_square_l2115_211501


namespace NUMINAMATH_CALUDE_complex_parts_of_3i_times_1_plus_i_l2115_211511

theorem complex_parts_of_3i_times_1_plus_i :
  let z : ℂ := 3 * Complex.I * (1 + Complex.I)
  (z.re = -3) ∧ (z.im = 3) := by sorry

end NUMINAMATH_CALUDE_complex_parts_of_3i_times_1_plus_i_l2115_211511


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2115_211533

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {2, 3, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2115_211533


namespace NUMINAMATH_CALUDE_sum_divisibility_odd_sum_divisibility_l2115_211578

theorem sum_divisibility (n : ℕ) :
  (∃ k : ℕ, 2 * n ∣ (n * (n + 1) / 2)) ↔ (∃ k : ℕ, n = 4 * k - 1) :=
sorry

theorem odd_sum_divisibility (n : ℕ) :
  (∃ k : ℕ, (2 * n + 1) ∣ (n * (n + 1) / 2)) ↔
  ((2 * n + 1) % 4 = 1 ∨ (2 * n + 1) % 4 = 3) :=
sorry

end NUMINAMATH_CALUDE_sum_divisibility_odd_sum_divisibility_l2115_211578


namespace NUMINAMATH_CALUDE_franks_breakfast_shopping_l2115_211515

/-- Frank's breakfast shopping problem -/
theorem franks_breakfast_shopping
  (num_buns : ℕ)
  (num_milk_bottles : ℕ)
  (milk_price : ℚ)
  (egg_price_multiplier : ℕ)
  (total_paid : ℚ)
  (h_num_buns : num_buns = 10)
  (h_num_milk_bottles : num_milk_bottles = 2)
  (h_milk_price : milk_price = 2)
  (h_egg_price : egg_price_multiplier = 3)
  (h_total_paid : total_paid = 11)
  : (total_paid - (num_milk_bottles * milk_price + egg_price_multiplier * milk_price)) / num_buns = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_franks_breakfast_shopping_l2115_211515


namespace NUMINAMATH_CALUDE_inscribed_square_existence_uniqueness_l2115_211586

/-- A sector in a plane --/
structure Sector where
  center : Point
  p : Point
  q : Point

/-- Angle of a sector --/
def Sector.angle (s : Sector) : ℝ := sorry

/-- A square in a plane --/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Predicate to check if a square is inscribed in a sector according to the problem conditions --/
def isInscribed (sq : Square) (s : Sector) : Prop := sorry

/-- Theorem stating the existence and uniqueness of the inscribed square --/
theorem inscribed_square_existence_uniqueness (s : Sector) :
  (∃! sq : Square, isInscribed sq s) ↔ s.angle ≤ 180 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_existence_uniqueness_l2115_211586


namespace NUMINAMATH_CALUDE_nth_equation_proof_l2115_211590

theorem nth_equation_proof (n : ℕ) : (n + 1) * (n^2 - n + 1) - 1 = n^3 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_proof_l2115_211590


namespace NUMINAMATH_CALUDE_johnsonville_marching_band_max_size_l2115_211550

theorem johnsonville_marching_band_max_size :
  ∀ m : ℕ,
  (∃ k : ℕ, 30 * m = 34 * k + 2) →
  30 * m < 1500 →
  (∀ n : ℕ, (∃ j : ℕ, 30 * n = 34 * j + 2) → 30 * n < 1500 → 30 * n ≤ 30 * m) →
  30 * m = 1260 :=
by sorry

end NUMINAMATH_CALUDE_johnsonville_marching_band_max_size_l2115_211550


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2115_211584

theorem hyperbola_equation (a b c : ℝ) (h1 : c = 4 * Real.sqrt 3) (h2 : a = 1) (h3 : b^2 = c^2 - a^2) :
  ∀ x y : ℝ, x^2 - y^2 / 47 = 1 ↔ x^2 - y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2115_211584


namespace NUMINAMATH_CALUDE_cookie_pie_slices_left_l2115_211531

theorem cookie_pie_slices_left (num_pies : ℕ) (slices_per_pie : ℕ) (num_people : ℕ) : 
  num_pies = 5 → 
  slices_per_pie = 12 → 
  num_people = 33 → 
  num_pies * slices_per_pie - num_people = 27 := by
sorry

end NUMINAMATH_CALUDE_cookie_pie_slices_left_l2115_211531


namespace NUMINAMATH_CALUDE_max_semicircle_intersections_l2115_211502

/-- Given n distinct points on a line, the maximum number of intersection points
    of semicircles drawn on one side of the line with these points as endpoints
    is equal to (n choose 4). -/
theorem max_semicircle_intersections (n : ℕ) : ℕ :=
  Nat.choose n 4

#check max_semicircle_intersections

end NUMINAMATH_CALUDE_max_semicircle_intersections_l2115_211502


namespace NUMINAMATH_CALUDE_sine_function_properties_l2115_211577

theorem sine_function_properties (ω φ : ℝ) (f : ℝ → ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π / 2)
  (h_f_def : ∀ x, f x = Real.sin (ω * x + φ))
  (h_period : ∀ x, f (x + π) = f x)
  (h_f_zero : f 0 = 1 / 2) :
  (ω = 2) ∧ 
  (∀ x, f (π / 3 - x) = f (π / 3 + x)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), 
    ∀ y ∈ Set.Icc (k * π - π / 3) (k * π + π / 6),
    x ≤ y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_sine_function_properties_l2115_211577


namespace NUMINAMATH_CALUDE_remainder_27_power_27_plus_27_mod_28_l2115_211524

theorem remainder_27_power_27_plus_27_mod_28 :
  (27^27 + 27) % 28 = 26 := by
  sorry

end NUMINAMATH_CALUDE_remainder_27_power_27_plus_27_mod_28_l2115_211524


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l2115_211530

/-- Given a class of students with their exam marks, this theorem proves
    the average mark of excluded students based on the given conditions. -/
theorem excluded_students_average_mark
  (total_students : ℕ)
  (all_average : ℝ)
  (excluded_count : ℕ)
  (remaining_average : ℝ)
  (h1 : total_students = 10)
  (h2 : all_average = 70)
  (h3 : excluded_count = 5)
  (h4 : remaining_average = 90) :
  (total_students * all_average - (total_students - excluded_count) * remaining_average) / excluded_count = 50 := by
  sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l2115_211530


namespace NUMINAMATH_CALUDE_units_digit_of_7_cubed_l2115_211540

theorem units_digit_of_7_cubed : (7^3) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_cubed_l2115_211540


namespace NUMINAMATH_CALUDE_fraction_sum_l2115_211507

theorem fraction_sum : (3 : ℚ) / 9 + (7 : ℚ) / 12 = (11 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2115_211507


namespace NUMINAMATH_CALUDE_final_postcard_count_l2115_211583

-- Define the exchange rates
def euro_to_usd : ℚ := 1.20
def gbp_to_usd : ℚ := 1.35
def usd_to_yen : ℚ := 110

-- Define the initial number of postcards and sales
def initial_postcards : ℕ := 18
def sold_euro : ℕ := 6
def sold_gbp : ℕ := 3
def sold_usd : ℕ := 2

-- Define the prices of sold postcards
def price_euro : ℚ := 10
def price_gbp : ℚ := 12
def price_usd : ℚ := 15

-- Define the price of new postcards in USD
def new_postcard_price_usd : ℚ := 8

-- Define the price of additional postcards in Yen
def additional_postcard_price_yen : ℚ := 800

-- Define the percentage of earnings used to buy new postcards
def percentage_for_new_postcards : ℚ := 0.70

-- Define the number of additional postcards bought
def additional_postcards : ℕ := 5

-- Theorem statement
theorem final_postcard_count :
  let total_earnings_usd := sold_euro * price_euro * euro_to_usd + 
                            sold_gbp * price_gbp * gbp_to_usd + 
                            sold_usd * price_usd
  let new_postcards := (total_earnings_usd * percentage_for_new_postcards / new_postcard_price_usd).floor
  let remaining_usd := total_earnings_usd - new_postcards * new_postcard_price_usd
  let additional_postcards_bought := (remaining_usd * usd_to_yen / additional_postcard_price_yen).floor
  initial_postcards - (sold_euro + sold_gbp + sold_usd) + new_postcards + additional_postcards_bought = 26 :=
by sorry

end NUMINAMATH_CALUDE_final_postcard_count_l2115_211583


namespace NUMINAMATH_CALUDE_kevin_kangaroo_four_hops_l2115_211593

def hop_distance (remaining : ℚ) : ℚ := (1 / 4) * remaining

def total_distance (n : ℕ) : ℚ :=
  let goal := 2
  let rec distance_after_hops (k : ℕ) (remaining : ℚ) (acc : ℚ) : ℚ :=
    if k = 0 then acc
    else
      let hop := hop_distance remaining
      distance_after_hops (k - 1) (remaining - hop) (acc + hop)
  distance_after_hops n goal 0

theorem kevin_kangaroo_four_hops :
  total_distance 4 = 175 / 128 := by sorry

end NUMINAMATH_CALUDE_kevin_kangaroo_four_hops_l2115_211593


namespace NUMINAMATH_CALUDE_factor_calculation_l2115_211581

theorem factor_calculation (x : ℝ) (factor : ℝ) : 
  x = 4 → (2 * x + 9) * factor = 51 → factor = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l2115_211581


namespace NUMINAMATH_CALUDE_inverse_proportion_through_point_l2115_211587

/-- An inverse proportion function passing through (-2, 3) has the equation y = -6/x -/
theorem inverse_proportion_through_point (f : ℝ → ℝ) :
  (∀ x ≠ 0, ∃ k, f x = k / x) →  -- f is an inverse proportion function
  f (-2) = 3 →                   -- f passes through the point (-2, 3)
  ∀ x ≠ 0, f x = -6 / x :=       -- The equation of f is y = -6/x
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_through_point_l2115_211587


namespace NUMINAMATH_CALUDE_sin_2theta_value_l2115_211541

theorem sin_2theta_value (θ : Real) (h : Real.tan θ + 1 / Real.tan θ = 4) : 
  Real.sin (2 * θ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l2115_211541


namespace NUMINAMATH_CALUDE_consecutive_even_sum_46_l2115_211528

theorem consecutive_even_sum_46 (n m : ℤ) : 
  (Even n) → (Even m) → (m = n + 2) → (n + m = 46) → (m = 24) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_46_l2115_211528


namespace NUMINAMATH_CALUDE_point_distance_on_number_line_l2115_211534

theorem point_distance_on_number_line :
  ∀ x : ℝ, |x - (-3)| = 4 ↔ x = -7 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_point_distance_on_number_line_l2115_211534


namespace NUMINAMATH_CALUDE_binary_1011_equals_11_l2115_211582

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1011_equals_11 :
  binary_to_decimal [true, true, false, true] = 11 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011_equals_11_l2115_211582


namespace NUMINAMATH_CALUDE_f_minimum_value_l2115_211546

noncomputable def f (x : ℝ) : ℝ := (1 + 4 * x) / Real.sqrt x

theorem f_minimum_value (x : ℝ) (hx : x > 0) : 
  f x ≥ 4 ∧ ∃ x₀ > 0, f x₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l2115_211546


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l2115_211545

/-- The number of students in the class --/
def total_students : ℕ := 6

/-- The number of students needed for the relay race --/
def relay_team_size : ℕ := 4

/-- The possible positions for student A --/
inductive PositionA
| first
| second

/-- The possible positions for student B --/
inductive PositionB
| second
| fourth

/-- A function to calculate the number of arrangements --/
def count_arrangements (total : ℕ) (team_size : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the number of arrangements is 36 --/
theorem relay_race_arrangements :
  count_arrangements total_students relay_team_size = 36 :=
sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l2115_211545


namespace NUMINAMATH_CALUDE_power_of_two_special_case_l2115_211509

theorem power_of_two_special_case :
  let n : ℝ := 2^(0.15 : ℝ)
  let b : ℝ := 33.333333333333314
  n^b = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_special_case_l2115_211509


namespace NUMINAMATH_CALUDE_rectangle_area_l2115_211572

def length (x : ℝ) : ℝ := 5 * x + 3

def width (x : ℝ) : ℝ := x - 7

def area (x : ℝ) : ℝ := length x * width x

theorem rectangle_area (x : ℝ) : area x = 5 * x^2 - 32 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2115_211572


namespace NUMINAMATH_CALUDE_furniture_cost_price_l2115_211554

theorem furniture_cost_price (price : ℝ) (discount : ℝ) (profit : ℝ) :
  price = 132 ∧ 
  discount = 0.1 ∧ 
  profit = 0.1 ∧ 
  price * (1 - discount) = (1 + profit) * (price * (1 - discount) / (1 + profit)) →
  price * (1 - discount) / (1 + profit) = 108 :=
by sorry

end NUMINAMATH_CALUDE_furniture_cost_price_l2115_211554


namespace NUMINAMATH_CALUDE_altitude_and_equidistant_lines_l2115_211580

/-- Given three points in a plane -/
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 4)
def C : ℝ × ℝ := (5, 7)

/-- Line l₁ containing the altitude from B to BC -/
def l₁ (x y : ℝ) : Prop := 7 * x + 3 * y - 10 = 0

/-- Lines l₂ passing through B with equal distances from A and C -/
def l₂₁ (y : ℝ) : Prop := y = 4
def l₂₂ (x y : ℝ) : Prop := 3 * x - 2 * y + 14 = 0

/-- Main theorem -/
theorem altitude_and_equidistant_lines :
  (∀ x y, l₁ x y ↔ (x - B.1) * (C.2 - B.2) = (y - B.2) * (C.1 - B.1)) ∧
  (∀ x y, (l₂₁ y ∨ l₂₂ x y) ↔ 
    ((x = B.1 ∧ y = B.2) ∨ 
     (abs ((y - A.2) - ((y - B.2) / (x - B.1)) * (A.1 - B.1)) = 
      abs ((y - C.2) - ((y - B.2) / (x - B.1)) * (C.1 - B.1))))) :=
sorry

end NUMINAMATH_CALUDE_altitude_and_equidistant_lines_l2115_211580


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2115_211517

/-- The line x + y = 0 is tangent to the circle (x-a)² + (y-b)² = 2 -/
def is_tangent (a b : ℝ) : Prop :=
  (a + b = 2) ∨ (a + b = -2)

/-- a + b = 2 is a sufficient condition for the line to be tangent to the circle -/
theorem sufficient_condition (a b : ℝ) :
  a + b = 2 → is_tangent a b :=
sorry

/-- a + b = 2 is not a necessary condition for the line to be tangent to the circle -/
theorem not_necessary_condition :
  ∃ a b, is_tangent a b ∧ a + b ≠ 2 :=
sorry

/-- a + b = 2 is a sufficient but not necessary condition for the line to be tangent to the circle -/
theorem sufficient_but_not_necessary :
  (∀ a b, a + b = 2 → is_tangent a b) ∧
  (∃ a b, is_tangent a b ∧ a + b ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2115_211517


namespace NUMINAMATH_CALUDE_grandmothers_gift_amount_l2115_211503

theorem grandmothers_gift_amount (num_grandchildren : ℕ) (cards_per_year : ℕ) (money_per_card : ℕ) : 
  num_grandchildren = 3 → cards_per_year = 2 → money_per_card = 80 →
  num_grandchildren * cards_per_year * money_per_card = 480 := by
  sorry

end NUMINAMATH_CALUDE_grandmothers_gift_amount_l2115_211503


namespace NUMINAMATH_CALUDE_unique_integer_square_less_than_triple_l2115_211575

theorem unique_integer_square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_square_less_than_triple_l2115_211575


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l2115_211506

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l2115_211506


namespace NUMINAMATH_CALUDE_abs_difference_equals_sum_of_abs_l2115_211553

theorem abs_difference_equals_sum_of_abs (a b c : ℚ) 
  (h1 : a < b) (h2 : b < 0) (h3 : 0 < c) : 
  |a - c| = |a| + c := by sorry

end NUMINAMATH_CALUDE_abs_difference_equals_sum_of_abs_l2115_211553


namespace NUMINAMATH_CALUDE_tan_sum_special_l2115_211542

theorem tan_sum_special : Real.tan (10 * π / 180) + Real.tan (50 * π / 180) + Real.sqrt 3 * Real.tan (10 * π / 180) * Real.tan (50 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_special_l2115_211542
