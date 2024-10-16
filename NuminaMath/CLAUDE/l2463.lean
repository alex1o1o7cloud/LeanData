import Mathlib

namespace NUMINAMATH_CALUDE_parallelepiped_ring_sum_exists_l2463_246377

/-- Represents a rectangular parallelepiped with dimensions a × b × c -/
structure Parallelepiped (a b c : ℕ) where
  dim_a : a > 0
  dim_b : b > 0
  dim_c : c > 0

/-- Represents an assignment of numbers to the faces of a parallelepiped -/
def FaceAssignment (a b c : ℕ) := Fin 6 → ℕ

/-- Calculates the sum of numbers in a 1-unit-wide ring around the parallelepiped -/
def ringSum (p : Parallelepiped 3 4 5) (assignment : FaceAssignment 3 4 5) : ℕ :=
  2 * (4 * assignment 0 + 5 * assignment 2 +
       3 * assignment 0 + 5 * assignment 4 +
       3 * assignment 2 + 4 * assignment 4)

/-- The main theorem stating that there exists an assignment satisfying the condition -/
theorem parallelepiped_ring_sum_exists :
  ∃ (assignment : FaceAssignment 3 4 5),
    ∀ (p : Parallelepiped 3 4 5), ringSum p assignment = 120 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_ring_sum_exists_l2463_246377


namespace NUMINAMATH_CALUDE_parabola_translation_l2463_246354

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - k }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 8 0 0
  let translated := translate original 3 (-5)
  y = 8 * x^2 → y = translated.a * (x - 3)^2 + translated.b * (x - 3) + translated.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2463_246354


namespace NUMINAMATH_CALUDE_joshua_toy_cars_l2463_246308

theorem joshua_toy_cars (total_boxes : ℕ) (cars_box1 : ℕ) (cars_box2 : ℕ) (total_cars : ℕ)
  (h1 : total_boxes = 3)
  (h2 : cars_box1 = 21)
  (h3 : cars_box2 = 31)
  (h4 : total_cars = 71) :
  total_cars - (cars_box1 + cars_box2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_joshua_toy_cars_l2463_246308


namespace NUMINAMATH_CALUDE_solution_equality_l2463_246386

/-- The solution set of x^2 - 2x - 3 < 0 -/
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

/-- The solution set of x^2 + x - 6 < 0 -/
def B : Set ℝ := {x | x^2 + x - 6 < 0}

/-- The solution set of x^2 + ax + b < 0 -/
def C (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

/-- Theorem stating that if C(a,b) = A ∩ B, then a + b = -3 -/
theorem solution_equality (a b : ℝ) (h : C a b = A ∩ B) : a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_equality_l2463_246386


namespace NUMINAMATH_CALUDE_problem_1_l2463_246378

theorem problem_1 : -3 * (1/4) - (-1/9) + (-3/4) + 1 * (8/9) = -2 := by sorry

end NUMINAMATH_CALUDE_problem_1_l2463_246378


namespace NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l2463_246389

open Real

theorem least_positive_t_for_geometric_progression :
  ∃ (t : ℝ), t > 0 ∧
  (∀ (α : ℝ), 0 < α → α < π / 3 →
    ∃ (r : ℝ), r > 0 ∧
    (arcsin (sin (3 * α)) * r = arcsin (sin (6 * α))) ∧
    (arcsin (sin (6 * α)) * r = arccos (cos (10 * α))) ∧
    (arccos (cos (10 * α)) * r = arcsin (sin (t * α)))) ∧
  (∀ (t' : ℝ), t' > 0 →
    (∀ (α : ℝ), 0 < α → α < π / 3 →
      ∃ (r : ℝ), r > 0 ∧
      (arcsin (sin (3 * α)) * r = arcsin (sin (6 * α))) ∧
      (arcsin (sin (6 * α)) * r = arccos (cos (10 * α))) ∧
      (arccos (cos (10 * α)) * r = arcsin (sin (t' * α)))) →
    t ≤ t') ∧
  t = 10 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l2463_246389


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l2463_246383

def sum_fractions : ℚ :=
  3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/8 + 7 + 1/9

theorem smallest_whole_number_above_sum : 
  ∃ n : ℕ, n = 26 ∧ (∀ m : ℕ, m < n → (m : ℚ) ≤ sum_fractions) ∧ sum_fractions < (n : ℚ) :=
sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l2463_246383


namespace NUMINAMATH_CALUDE_two_digit_number_system_l2463_246342

theorem two_digit_number_system (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 →
  (10 * x + y) - 3 * (x + y) = 13 →
  (10 * x + y) % (x + y) = 6 →
  (10 * x + y) / (x + y) = 4 →
  (10 * x + y - 3 * (x + y) = 13 ∧ 10 * x + y - 6 = 4 * (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_system_l2463_246342


namespace NUMINAMATH_CALUDE_total_campers_is_71_l2463_246337

/-- The total number of campers who went rowing and hiking -/
def total_campers (morning_rowing : ℕ) (morning_hiking : ℕ) (afternoon_rowing : ℕ) : ℕ :=
  morning_rowing + morning_hiking + afternoon_rowing

/-- Theorem stating that the total number of campers who went rowing and hiking is 71 -/
theorem total_campers_is_71 :
  total_campers 41 4 26 = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_is_71_l2463_246337


namespace NUMINAMATH_CALUDE_least_sum_m_n_l2463_246347

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m + n) 231 = 1) ∧ 
  (∃ (k : ℕ), m ^ m.val = k * (n ^ n.val)) ∧ 
  (∀ (k : ℕ+), m ≠ k * n) ∧
  (m + n = 75) ∧
  (∀ (m' n' : ℕ+), 
    (Nat.gcd (m' + n') 231 = 1) → 
    (∃ (k : ℕ), m' ^ m'.val = k * (n' ^ n'.val)) → 
    (∀ (k : ℕ+), m' ≠ k * n') → 
    (m' + n' ≥ 75)) :=
sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l2463_246347


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l2463_246358

def number_of_arrangements (a o b g : ℕ) : ℕ :=
  Nat.factorial 14 / (Nat.factorial a * Nat.factorial o * Nat.factorial b * Nat.factorial g)

theorem fruit_arrangement_count :
  number_of_arrangements 4 3 3 4 = 4204200 :=
by
  sorry

#eval number_of_arrangements 4 3 3 4

end NUMINAMATH_CALUDE_fruit_arrangement_count_l2463_246358


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l2463_246326

def total_shoes : ℕ := 10
def red_shoes : ℕ := 6
def green_shoes : ℕ := 4
def shoes_drawn : ℕ := 2

theorem probability_two_red_shoes :
  (Nat.choose red_shoes shoes_drawn) / (Nat.choose total_shoes shoes_drawn) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l2463_246326


namespace NUMINAMATH_CALUDE_negative_one_greater_than_negative_sqrt_two_l2463_246351

theorem negative_one_greater_than_negative_sqrt_two :
  -1 > -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_greater_than_negative_sqrt_two_l2463_246351


namespace NUMINAMATH_CALUDE_lingling_tourist_growth_l2463_246356

/-- The average annual growth rate of tourists visiting Lingling Ancient City from 2018 to 2020 -/
def average_growth_rate : ℝ := 0.125

/-- The number of tourists (in millions) visiting Lingling Ancient City in 2018 -/
def tourists_2018 : ℝ := 6.4

/-- The number of tourists (in millions) visiting Lingling Ancient City in 2020 -/
def tourists_2020 : ℝ := 8.1

/-- The time period in years -/
def years : ℕ := 2

theorem lingling_tourist_growth :
  tourists_2018 * (1 + average_growth_rate) ^ years = tourists_2020 := by
  sorry

end NUMINAMATH_CALUDE_lingling_tourist_growth_l2463_246356


namespace NUMINAMATH_CALUDE_adams_laundry_l2463_246395

theorem adams_laundry (total_loads : ℕ) (remaining_loads : ℕ) (washed_loads : ℕ) : 
  total_loads = 14 → remaining_loads = 6 → washed_loads = total_loads - remaining_loads → washed_loads = 8 := by
  sorry

end NUMINAMATH_CALUDE_adams_laundry_l2463_246395


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2463_246365

theorem solve_linear_equation (x : ℝ) :
  3 * x - 4 * x + 5 * x = 140 → x = 35 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2463_246365


namespace NUMINAMATH_CALUDE_parallelogram_point_D_l2463_246349

/-- A parallelogram in the complex plane -/
structure ComplexParallelogram where
  A : ℂ
  B : ℂ
  C : ℂ
  D : ℂ
  is_parallelogram : (B - A) = (C - D)

/-- Theorem: Given a parallelogram ABCD in the complex plane with A = 1+3i, B = 2-i, and C = -3+i, then D = -4+5i -/
theorem parallelogram_point_D (ABCD : ComplexParallelogram) 
  (hA : ABCD.A = 1 + 3*I)
  (hB : ABCD.B = 2 - I)
  (hC : ABCD.C = -3 + I) :
  ABCD.D = -4 + 5*I :=
sorry

end NUMINAMATH_CALUDE_parallelogram_point_D_l2463_246349


namespace NUMINAMATH_CALUDE_box_weight_l2463_246310

/-- Given a pallet with boxes, calculate the weight of each box. -/
theorem box_weight (total_weight : ℕ) (num_boxes : ℕ) (h1 : total_weight = 267) (h2 : num_boxes = 3) :
  total_weight / num_boxes = 89 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_l2463_246310


namespace NUMINAMATH_CALUDE_genetic_events_in_both_divisions_l2463_246305

-- Define cell division processes
inductive CellDivision
| mitosis
| meiosis

-- Define genetic events
inductive GeneticEvent
| mutation
| chromosomalVariation

-- Define cellular processes during division
structure CellularProcess where
  chromosomeReplication : Bool
  centromereSplitting : Bool

-- Define the occurrence of genetic events during cell division
def geneticEventOccurs (event : GeneticEvent) (division : CellDivision) : Prop :=
  ∃ (process : CellularProcess), 
    process.chromosomeReplication ∧ 
    process.centromereSplitting

-- Theorem statement
theorem genetic_events_in_both_divisions :
  (∀ (event : GeneticEvent) (division : CellDivision), 
    geneticEventOccurs event division) :=
sorry

end NUMINAMATH_CALUDE_genetic_events_in_both_divisions_l2463_246305


namespace NUMINAMATH_CALUDE_multiply_square_roots_l2463_246319

theorem multiply_square_roots : -2 * Real.sqrt 10 * (3 * Real.sqrt 30) = -60 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_square_roots_l2463_246319


namespace NUMINAMATH_CALUDE_pair_in_six_cascades_valid_coloring_exists_l2463_246320

-- Define a cascade
def cascade (r : ℕ) : Set ℕ := {n : ℕ | ∃ k : ℕ, k ≤ 12 ∧ n = k * r}

-- Part a: Existence of a pair in six cascades
theorem pair_in_six_cascades : ∃ a b : ℕ, ∃ r₁ r₂ r₃ r₄ r₅ r₆ : ℕ,
  r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧ r₁ ≠ r₆ ∧
  r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧ r₂ ≠ r₆ ∧
  r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧ r₃ ≠ r₆ ∧
  r₄ ≠ r₅ ∧ r₄ ≠ r₆ ∧
  r₅ ≠ r₆ ∧
  a ∈ cascade r₁ ∧ b ∈ cascade r₁ ∧
  a ∈ cascade r₂ ∧ b ∈ cascade r₂ ∧
  a ∈ cascade r₃ ∧ b ∈ cascade r₃ ∧
  a ∈ cascade r₄ ∧ b ∈ cascade r₄ ∧
  a ∈ cascade r₅ ∧ b ∈ cascade r₅ ∧
  a ∈ cascade r₆ ∧ b ∈ cascade r₆ := by
  sorry

-- Part b: Existence of a valid coloring function
theorem valid_coloring_exists : ∃ f : ℕ → Fin 12, ∀ r : ℕ, ∀ k₁ k₂ : ℕ,
  k₁ ≤ 12 → k₂ ≤ 12 → k₁ ≠ k₂ → f (k₁ * r) ≠ f (k₂ * r) := by
  sorry

end NUMINAMATH_CALUDE_pair_in_six_cascades_valid_coloring_exists_l2463_246320


namespace NUMINAMATH_CALUDE_partners_count_l2463_246302

/-- Represents the number of employees in each category -/
structure FirmComposition where
  partners : ℕ
  associates : ℕ
  managers : ℕ

/-- The initial ratio of partners : associates : managers -/
def initial_ratio : FirmComposition := ⟨2, 63, 20⟩

/-- The new ratio after hiring more employees -/
def new_ratio : FirmComposition := ⟨1, 34, 15⟩

/-- The number of additional associates hired -/
def additional_associates : ℕ := 35

/-- The number of additional managers hired -/
def additional_managers : ℕ := 10

/-- Theorem stating that the number of partners in the firm is 14 -/
theorem partners_count : ∃ (x : ℕ), 
  x * initial_ratio.partners = 14 ∧
  x * initial_ratio.associates + additional_associates = new_ratio.associates * 14 ∧
  x * initial_ratio.managers + additional_managers = new_ratio.managers * 14 :=
sorry

end NUMINAMATH_CALUDE_partners_count_l2463_246302


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2463_246322

theorem complex_expression_simplification :
  (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) + (1 - Complex.I) * (3 + Complex.I) = 5 - 17 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2463_246322


namespace NUMINAMATH_CALUDE_new_apples_grown_l2463_246360

/-- Given a tree with apples, calculate the number of new apples grown -/
theorem new_apples_grown
  (initial_apples : ℕ)
  (picked_apples : ℕ)
  (current_apples : ℕ)
  (h1 : initial_apples = 4)
  (h2 : picked_apples = 2)
  (h3 : current_apples = 5)
  (h4 : picked_apples ≤ initial_apples) :
  current_apples - (initial_apples - picked_apples) = 3 :=
by sorry

end NUMINAMATH_CALUDE_new_apples_grown_l2463_246360


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2463_246372

noncomputable def f (x : ℝ) : ℝ := x - (Real.exp 1 - 1) * Real.log x

theorem solution_set_of_inequality (x : ℝ) :
  (f (Real.exp x) < 1) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2463_246372


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l2463_246339

/-- A function satisfying the given inequality for all real numbers -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x * z) - f x * f (y * z) ≥ 1

/-- The theorem stating that there is a unique function satisfying the inequality -/
theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfiesInequality f ∧ ∀ x : ℝ, f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l2463_246339


namespace NUMINAMATH_CALUDE_max_min_values_l2463_246376

theorem max_min_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 2 * y = 10) :
  (Real.sqrt (3 * x) + Real.sqrt (2 * y) ≤ 2 * Real.sqrt 5) ∧
  (3 / x + 2 / y ≥ 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_l2463_246376


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l2463_246312

theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (student_marks : ℕ) (failing_margin : ℕ),
    student_marks = 92 →
    failing_margin = 40 →
    (student_marks + failing_margin : ℚ) = (33 / 100) * max_marks →
    max_marks = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l2463_246312


namespace NUMINAMATH_CALUDE_problem_statement_l2463_246334

theorem problem_statement (a b : ℝ) : 
  (Real.sqrt (a - 2) + abs (b + 3) = 0) → ((a + b) ^ 2023 = -1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2463_246334


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2463_246311

theorem smallest_number_with_remainders : ∃! x : ℕ, 
  x % 4 = 1 ∧ 
  x % 3 = 2 ∧ 
  x % 5 = 3 ∧ 
  ∀ y : ℕ, (y % 4 = 1 ∧ y % 3 = 2 ∧ y % 5 = 3) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2463_246311


namespace NUMINAMATH_CALUDE_burj_khalifa_height_is_830_l2463_246303

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height : ℝ := 324

/-- The difference in height between the Burj Khalifa and the Eiffel Tower in meters -/
def height_difference : ℝ := 506

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height : ℝ := eiffel_tower_height + height_difference

theorem burj_khalifa_height_is_830 : burj_khalifa_height = 830 := by
  sorry

end NUMINAMATH_CALUDE_burj_khalifa_height_is_830_l2463_246303


namespace NUMINAMATH_CALUDE_stairs_distance_l2463_246397

theorem stairs_distance (total_time speed_up speed_down : ℝ) 
  (h_total_time : total_time = 4)
  (h_speed_up : speed_up = 2)
  (h_speed_down : speed_down = 3)
  (h_distance_diff : ∀ d : ℝ, d / speed_up + (d + 2) / speed_down = total_time) :
  ∃ d : ℝ, d + 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_stairs_distance_l2463_246397


namespace NUMINAMATH_CALUDE_derivative_of_f_l2463_246385

-- Define the function f(x) = (3x+4)(2x+6)
def f (x : ℝ) : ℝ := (3*x + 4) * (2*x + 6)

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 12*x + 26 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l2463_246385


namespace NUMINAMATH_CALUDE_kolya_cannot_descend_l2463_246388

/-- Represents the possible jump sizes Kolya can make. -/
inductive JumpSize
  | six
  | seven
  | eight

/-- Represents a sequence of jumps Kolya makes. -/
def JumpSequence := List JumpSize

/-- The total number of steps on the ladder. -/
def totalSteps : Nat := 100

/-- Converts a JumpSize to its corresponding natural number. -/
def jumpSizeToNat (j : JumpSize) : Nat :=
  match j with
  | JumpSize.six => 6
  | JumpSize.seven => 7
  | JumpSize.eight => 8

/-- Calculates the position after a sequence of jumps. -/
def finalPosition (jumps : JumpSequence) : Int :=
  totalSteps - (jumps.map jumpSizeToNat).sum

/-- Checks if a sequence of jumps results in unique positions. -/
def hasUniquePositions (jumps : JumpSequence) : Prop :=
  let positions := List.scanl (fun pos jump => pos - jumpSizeToNat jump) totalSteps jumps
  positions.Nodup

/-- Theorem stating that Kolya cannot descend the ladder under the given conditions. -/
theorem kolya_cannot_descend :
  ¬∃ (jumps : JumpSequence), finalPosition jumps = 0 ∧ hasUniquePositions jumps :=
sorry


end NUMINAMATH_CALUDE_kolya_cannot_descend_l2463_246388


namespace NUMINAMATH_CALUDE_burgers_spending_l2463_246345

def total_allowance : ℚ := 50

def movie_fraction : ℚ := 2 / 5
def video_game_fraction : ℚ := 1 / 10
def book_fraction : ℚ := 1 / 4

def spent_on_movies : ℚ := movie_fraction * total_allowance
def spent_on_video_games : ℚ := video_game_fraction * total_allowance
def spent_on_books : ℚ := book_fraction * total_allowance

def total_spent : ℚ := spent_on_movies + spent_on_video_games + spent_on_books

def remaining_for_burgers : ℚ := total_allowance - total_spent

theorem burgers_spending :
  remaining_for_burgers = 12.5 := by sorry

end NUMINAMATH_CALUDE_burgers_spending_l2463_246345


namespace NUMINAMATH_CALUDE_experience_difference_l2463_246341

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions of the problem -/
def problemConditions (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.roger = 50 - 8 ∧
  e.peter = 19 - 7 ∧
  e.tom = 2 * e.robert ∧
  e.robert = e.peter - 4 ∧
  e.robert > e.mike

/-- The theorem to prove -/
theorem experience_difference (e : Experience) :
  problemConditions e → e.robert - e.mike = 2 := by
  sorry

end NUMINAMATH_CALUDE_experience_difference_l2463_246341


namespace NUMINAMATH_CALUDE_dinner_slices_l2463_246375

/-- Represents the number of slices of pie served during different times of the day -/
structure PieSlices where
  lunch : ℕ
  total : ℕ

/-- Proves that the number of slices served during dinner is 5,
    given 7 slices were served during lunch and 12 slices in total -/
theorem dinner_slices (ps : PieSlices) 
  (h_lunch : ps.lunch = 7) 
  (h_total : ps.total = 12) : 
  ps.total - ps.lunch = 5 := by
  sorry

end NUMINAMATH_CALUDE_dinner_slices_l2463_246375


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l2463_246316

/-- Given a rectangle intersecting a circle with specific chord properties,
    prove the ratio of their areas. -/
theorem rectangle_circle_area_ratio :
  ∀ (r : ℝ) (x y : ℝ),
    r > 0 →                 -- radius is positive
    x > 0 →                 -- shorter side is positive
    y > 0 →                 -- longer side is positive
    y = r →                 -- longer side equals radius (chord property)
    x = r / 2 →             -- shorter side equals half radius (chord property)
    y = 2 * x →             -- longer side is twice the shorter side
    (x * y) / (π * r^2) = 1 / (2 * π) :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l2463_246316


namespace NUMINAMATH_CALUDE_fraction_irreducible_l2463_246382

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l2463_246382


namespace NUMINAMATH_CALUDE_probability_five_digit_palindrome_div_11_l2463_246332

-- Define a five-digit palindrome
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ ∃ a b c : ℕ, 
    a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 10000 * a + 1000 * b + 100 * c + 10 * b + a

-- Define divisibility by 11
def divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

-- Count of five-digit palindromes
def count_five_digit_palindromes : ℕ := 900

-- Count of five-digit palindromes divisible by 11
def count_five_digit_palindromes_div_11 : ℕ := 90

-- Theorem statement
theorem probability_five_digit_palindrome_div_11 :
  (count_five_digit_palindromes_div_11 : ℚ) / count_five_digit_palindromes = 1 / 10 :=
sorry

end NUMINAMATH_CALUDE_probability_five_digit_palindrome_div_11_l2463_246332


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l2463_246309

/-- Represents a batsman's statistics -/
structure Batsman where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (score : ℕ) : ℚ :=
  (b.totalScore + score) / (b.innings + 1)

/-- Theorem stating the batsman's new average after the 17th innings -/
theorem batsman_average_after_17th_innings
  (b : Batsman)
  (h1 : b.innings = 16)
  (h2 : newAverage b 85 = b.average + 3) :
  newAverage b 85 = 37 := by
  sorry

#check batsman_average_after_17th_innings

end NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l2463_246309


namespace NUMINAMATH_CALUDE_yang_final_floor_l2463_246366

/-- The number of floors in the building -/
def total_floors : ℕ := 36

/-- The floor Xiao Wu reaches in the initial observation -/
def wu_initial : ℕ := 6

/-- The floor Xiao Yang reaches in the initial observation -/
def yang_initial : ℕ := 5

/-- The starting floor for both climbers -/
def start_floor : ℕ := 1

/-- The floor Xiao Yang reaches when Xiao Wu reaches the top floor -/
def yang_final : ℕ := 29

theorem yang_final_floor :
  (wu_initial - start_floor) / (yang_initial - start_floor) =
  (total_floors - start_floor) / (yang_final - start_floor) :=
sorry

end NUMINAMATH_CALUDE_yang_final_floor_l2463_246366


namespace NUMINAMATH_CALUDE_complex_equality_l2463_246357

theorem complex_equality (z : ℂ) : z = Complex.I ↔ 
  Complex.abs (z - 2) = Complex.abs (z + 1 - Complex.I) ∧ 
  Complex.abs (z - 2) = Complex.abs (z - (1 + 2*Complex.I)) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l2463_246357


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l2463_246300

theorem quadratic_equation_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 6)
  (h_geometric : Real.sqrt (a * b) = 5) :
  ∃ (x : ℝ), x^2 - 12*x + 25 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l2463_246300


namespace NUMINAMATH_CALUDE_quadratic_root_reciprocal_relation_l2463_246352

/-- Given two quadratic equations ax² + bx + c = 0 and cx² + bx + a = 0,
    this theorem states that the roots of the second equation
    are the reciprocals of the roots of the first equation. -/
theorem quadratic_root_reciprocal_relation (a b c : ℝ) (x₁ x₂ : ℝ) :
  (a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) →
  (c * (1/x₁)^2 + b * (1/x₁) + a = 0 ∧ c * (1/x₂)^2 + b * (1/x₂) + a = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_reciprocal_relation_l2463_246352


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l2463_246306

theorem y_in_terms_of_x (x y : ℝ) (h : x + y = -1) : y = -1 - x := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l2463_246306


namespace NUMINAMATH_CALUDE_no_common_points_l2463_246362

/-- Theorem: If a point (x, y) is inside the parabola y^2 = 4x, 
    then the line yy = 2(x + x) and the parabola have no common points. -/
theorem no_common_points (x y : ℝ) (h : y^2 < 4*x) : 
  ∀ (x' y' : ℝ), y'^2 = 4*x' → y'*y = 2*(x + x') → False :=
by sorry

end NUMINAMATH_CALUDE_no_common_points_l2463_246362


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2463_246361

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = π/3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 1) : 
  Real.sqrt (((a.1 + 2*b.1) ^ 2) + ((a.2 + 2*b.2) ^ 2)) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2463_246361


namespace NUMINAMATH_CALUDE_dress_price_l2463_246363

theorem dress_price (total_revenue : ℕ) (num_dresses : ℕ) (num_shirts : ℕ) (shirt_price : ℕ) (dress_price : ℕ) :
  total_revenue = 69 →
  num_dresses = 7 →
  num_shirts = 4 →
  shirt_price = 5 →
  num_dresses * dress_price + num_shirts * shirt_price = total_revenue →
  dress_price = 7 := by
sorry

end NUMINAMATH_CALUDE_dress_price_l2463_246363


namespace NUMINAMATH_CALUDE_paul_fishing_theorem_l2463_246387

/-- Calculates the number of fish caught given a fishing rate and total time -/
def fish_caught (rate : ℚ) (time : ℚ) : ℚ :=
  (time / 2) * rate

/-- Proves that fishing at a rate of 5 fish per 2 hours for 12 hours results in 30 fish -/
theorem paul_fishing_theorem :
  fish_caught 5 12 = 30 := by
  sorry

end NUMINAMATH_CALUDE_paul_fishing_theorem_l2463_246387


namespace NUMINAMATH_CALUDE_units_digit_3_pow_34_l2463_246333

def units_digit (n : ℕ) : ℕ := n % 10

def power_3_cycle (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur due to the modulo operation

theorem units_digit_3_pow_34 :
  units_digit (3^34) = 9 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_3_pow_34_l2463_246333


namespace NUMINAMATH_CALUDE_solve_yellow_balloons_problem_l2463_246399

def yellow_balloons_problem (sam_initial : Real) (sam_gives : Real) (total : Real) : Prop :=
  let sam_remaining : Real := sam_initial - sam_gives
  let mary_balloons : Real := total - sam_remaining
  mary_balloons = 7.0

theorem solve_yellow_balloons_problem :
  yellow_balloons_problem 6.0 5.0 8.0 := by
  sorry

end NUMINAMATH_CALUDE_solve_yellow_balloons_problem_l2463_246399


namespace NUMINAMATH_CALUDE_stepa_multiplication_l2463_246368

theorem stepa_multiplication (sequence : Fin 5 → ℕ) 
  (h1 : ∀ i : Fin 4, sequence (i.succ) = (3 * sequence i) / 2)
  (h2 : sequence 4 = 81) :
  ∃ (a b : ℕ), a * b = sequence 3 ∧ a = 6 ∧ b = 9 :=
by sorry

end NUMINAMATH_CALUDE_stepa_multiplication_l2463_246368


namespace NUMINAMATH_CALUDE_probability_of_selection_l2463_246393

theorem probability_of_selection (total_students : ℕ) (xiao_li_in_group : Prop) : 
  total_students = 5 → xiao_li_in_group → (1 : ℚ) / total_students = (1 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_l2463_246393


namespace NUMINAMATH_CALUDE_min_value_zero_l2463_246348

/-- The quadratic function for which we want to find the minimum value -/
def f (m x y : ℝ) : ℝ := 3*x^2 - 4*m*x*y + (2*m^2 + 3)*y^2 - 6*x - 9*y + 8

/-- The theorem stating the value of m that makes the minimum of f equal to 0 -/
theorem min_value_zero (m : ℝ) : 
  (∀ x y : ℝ, f m x y ≥ 0) ∧ (∃ x y : ℝ, f m x y = 0) ↔ 
  m = (6 + Real.sqrt 67.5) / 9 ∨ m = (6 - Real.sqrt 67.5) / 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_zero_l2463_246348


namespace NUMINAMATH_CALUDE_cylinder_unique_non_trapezoid_cross_section_l2463_246324

-- Define the solids
inductive Solid
| Frustum
| Cylinder
| Cube
| TriangularPrism

-- Define a predicate for whether a solid can have an isosceles trapezoid cross-section
def can_have_isosceles_trapezoid_cross_section : Solid → Prop
| Solid.Frustum => True
| Solid.Cylinder => False
| Solid.Cube => True
| Solid.TriangularPrism => True

-- Theorem statement
theorem cylinder_unique_non_trapezoid_cross_section :
  ∀ s : Solid, ¬(can_have_isosceles_trapezoid_cross_section s) ↔ s = Solid.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_cylinder_unique_non_trapezoid_cross_section_l2463_246324


namespace NUMINAMATH_CALUDE_order_of_xyz_l2463_246325

theorem order_of_xyz (x y z : ℝ) 
  (h : x + 2013 / 2014 = y + 2012 / 2013 ∧ y + 2012 / 2013 = z + 2014 / 2015) : 
  z < y ∧ y < x := by
sorry

end NUMINAMATH_CALUDE_order_of_xyz_l2463_246325


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l2463_246336

/-- Given a man who can row with the stream at 16 km/h and against the stream at 8 km/h,
    his rate in still water is 12 km/h. -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h_with : speed_with_stream = 16)
  (h_against : speed_against_stream = 8) :
  (speed_with_stream + speed_against_stream) / 2 = 12 := by
  sorry

#check mans_rate_in_still_water

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l2463_246336


namespace NUMINAMATH_CALUDE_calculator_mistake_l2463_246369

theorem calculator_mistake (x : ℝ) (h : Real.sqrt x = 9) : x^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_calculator_mistake_l2463_246369


namespace NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_three_l2463_246330

theorem complex_exp_thirteen_pi_over_three : 
  Complex.exp (Complex.I * (13 * Real.pi / 3)) = Complex.ofReal (1 / 2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_three_l2463_246330


namespace NUMINAMATH_CALUDE_mike_tire_change_l2463_246313

def total_tires_changed (num_motorcycles num_cars tires_per_motorcycle tires_per_car : ℕ) : ℕ :=
  num_motorcycles * tires_per_motorcycle + num_cars * tires_per_car

theorem mike_tire_change :
  let num_motorcycles : ℕ := 12
  let num_cars : ℕ := 10
  let tires_per_motorcycle : ℕ := 2
  let tires_per_car : ℕ := 4
  total_tires_changed num_motorcycles num_cars tires_per_motorcycle tires_per_car = 64 := by
  sorry

end NUMINAMATH_CALUDE_mike_tire_change_l2463_246313


namespace NUMINAMATH_CALUDE_triangle_properties_l2463_246331

/-- Given a triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h_area : (1/2) * t.a * t.c * Real.sin t.B = (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2))
  (h_obtuse : t.C > π/2) :
  t.B = π/3 ∧ ∃ (k : ℝ), k > 2 ∧ t.c / t.a = k :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2463_246331


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_when_a_is_2_complement_A_union_B_equals_reals_iff_l2463_246327

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1) + Real.sqrt (2 - x)}
def B (a : ℝ) : Set ℝ := {y | ∃ x ≥ a, y = 2^x}

-- Define the complement of A
def complementA : Set ℝ := {x | x ∉ A}

-- Statement I
theorem complement_A_intersect_B_when_a_is_2 :
  (complementA ∩ B 2) = {x | x ≥ 4} := by sorry

-- Statement II
theorem complement_A_union_B_equals_reals_iff (a : ℝ) :
  (complementA ∪ B a) = Set.univ ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_when_a_is_2_complement_A_union_B_equals_reals_iff_l2463_246327


namespace NUMINAMATH_CALUDE_balanced_colorings_count_l2463_246370

/-- Represents a color in the grid -/
inductive Color
| Red
| Blue
| Yellow
| Green

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat
  color : Color

/-- Represents the grid -/
def Grid := List Cell

/-- Checks if a 2x2 subgrid is balanced -/
def isBalanced2x2 (grid : Grid) (startRow startCol : Nat) : Bool :=
  sorry

/-- Checks if the entire grid is balanced -/
def isBalancedGrid (grid : Grid) : Bool :=
  sorry

/-- Counts the number of balanced colorings for an 8x6 grid -/
def countBalancedColorings : Nat :=
  sorry

/-- The main theorem stating the number of balanced colorings -/
theorem balanced_colorings_count :
  countBalancedColorings = 1896 :=
sorry

end NUMINAMATH_CALUDE_balanced_colorings_count_l2463_246370


namespace NUMINAMATH_CALUDE_four_men_absent_l2463_246340

/-- Represents the work completion scenario with absent workers -/
structure WorkScenario where
  totalMen : ℕ
  originalDays : ℕ
  actualDays : ℕ
  absentMen : ℕ

/-- Calculates the number of absent men given the work scenario -/
def calculateAbsentMen (scenario : WorkScenario) : ℕ :=
  scenario.totalMen - (scenario.totalMen * scenario.originalDays) / scenario.actualDays

/-- Theorem stating that 4 men became absent in the given scenario -/
theorem four_men_absent :
  let scenario := WorkScenario.mk 8 6 12 4
  calculateAbsentMen scenario = 4 := by
  sorry

#eval calculateAbsentMen (WorkScenario.mk 8 6 12 4)

end NUMINAMATH_CALUDE_four_men_absent_l2463_246340


namespace NUMINAMATH_CALUDE_peach_difference_is_eight_l2463_246384

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 14

/-- The number of yellow peaches in the basket -/
def yellow_peaches : ℕ := 6

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 2

/-- The difference between the number of green peaches and yellow peaches -/
def peach_difference : ℕ := green_peaches - yellow_peaches

theorem peach_difference_is_eight : peach_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_is_eight_l2463_246384


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l2463_246343

/-- The quadratic equation 3x^2 + 2x - 5 = 0 has exactly two distinct real solutions. -/
theorem parabola_x_intercepts :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  3 * x₁^2 + 2 * x₁ - 5 = 0 ∧
  3 * x₂^2 + 2 * x₂ - 5 = 0 ∧
  ∀ (x : ℝ), 3 * x^2 + 2 * x - 5 = 0 → (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l2463_246343


namespace NUMINAMATH_CALUDE_red_parrots_count_l2463_246391

theorem red_parrots_count (total : ℕ) (green_fraction : ℚ) (blue_fraction : ℚ) 
  (h_total : total = 160)
  (h_green : green_fraction = 5/8)
  (h_blue : blue_fraction = 1/4)
  (h_sum : green_fraction + blue_fraction < 1) :
  total - (green_fraction * total).num - (blue_fraction * total).num = 20 := by
  sorry

end NUMINAMATH_CALUDE_red_parrots_count_l2463_246391


namespace NUMINAMATH_CALUDE_maria_apple_sales_l2463_246364

/-- Calculate the average revenue per hour for Maria's apple sales -/
theorem maria_apple_sales (a1 a2 b1 b2 pa1 pa2 pb : ℕ) 
  (h1 : a1 = 10) -- kg of type A apples sold in first hour
  (h2 : a2 = 2)  -- kg of type A apples sold in second hour
  (h3 : b1 = 5)  -- kg of type B apples sold in first hour
  (h4 : b2 = 3)  -- kg of type B apples sold in second hour
  (h5 : pa1 = 3) -- price of type A apples in first hour
  (h6 : pa2 = 4) -- price of type A apples in second hour
  (h7 : pb = 2)  -- price of type B apples (constant)
  : (a1 * pa1 + b1 * pb + a2 * pa2 + b2 * pb) / 2 = 27 := by
  sorry

#check maria_apple_sales

end NUMINAMATH_CALUDE_maria_apple_sales_l2463_246364


namespace NUMINAMATH_CALUDE_people_counting_l2463_246328

theorem people_counting (first_day second_day : ℕ) : 
  first_day = 2 * second_day →
  first_day + second_day = 1500 →
  second_day = 500 := by
sorry

end NUMINAMATH_CALUDE_people_counting_l2463_246328


namespace NUMINAMATH_CALUDE_angle_supplement_l2463_246394

theorem angle_supplement (θ : ℝ) : 
  (90 - θ = 30) → (180 - θ = 120) := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_l2463_246394


namespace NUMINAMATH_CALUDE_green_sweets_count_l2463_246398

theorem green_sweets_count (total : ℕ) (red : ℕ) (neither : ℕ) (h1 : total = 285) (h2 : red = 49) (h3 : neither = 177) :
  total - red - neither = 59 := by
  sorry

end NUMINAMATH_CALUDE_green_sweets_count_l2463_246398


namespace NUMINAMATH_CALUDE_line_quadrants_m_range_l2463_246317

theorem line_quadrants_m_range (m : ℝ) : 
  (∀ x y : ℝ, y = (m - 2) * x + m → 
    (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) → 
  0 < m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_line_quadrants_m_range_l2463_246317


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2463_246359

theorem arithmetic_sequence_sum : 
  ∀ (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ),
    a₁ = -25 →
    aₙ = 19 →
    d = 4 →
    aₙ = a₁ + (n - 1) * d →
    (n : ℤ) * (a₁ + aₙ) / 2 = -36 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2463_246359


namespace NUMINAMATH_CALUDE_mrs_hilt_has_more_money_l2463_246315

-- Define the value of each coin type
def penny_value : ℚ := 0.01
def nickel_value : ℚ := 0.05
def dime_value : ℚ := 0.10

-- Define the number of coins each person has
def mrs_hilt_pennies : ℕ := 2
def mrs_hilt_nickels : ℕ := 2
def mrs_hilt_dimes : ℕ := 2

def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1

-- Calculate the total amount for each person
def mrs_hilt_total : ℚ :=
  mrs_hilt_pennies * penny_value +
  mrs_hilt_nickels * nickel_value +
  mrs_hilt_dimes * dime_value

def jacob_total : ℚ :=
  jacob_pennies * penny_value +
  jacob_nickels * nickel_value +
  jacob_dimes * dime_value

-- State the theorem
theorem mrs_hilt_has_more_money :
  mrs_hilt_total - jacob_total = 0.13 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_has_more_money_l2463_246315


namespace NUMINAMATH_CALUDE_inequality_proof_l2463_246307

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + y) / (2 + x + y) < x / (2 + x) + y / (2 + y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2463_246307


namespace NUMINAMATH_CALUDE_deficiency_and_excess_l2463_246380

theorem deficiency_and_excess (people : ℕ) (price : ℕ) : 
  (5 * people + 45 = price) →
  (7 * people + 3 = price) →
  (people = 21 ∧ price = 150) := by
  sorry

end NUMINAMATH_CALUDE_deficiency_and_excess_l2463_246380


namespace NUMINAMATH_CALUDE_hotel_room_charges_l2463_246344

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R - 0.55 * R) 
  (h2 : P = G - 0.10 * G) : 
  R = 2 * G := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l2463_246344


namespace NUMINAMATH_CALUDE_window_purchase_savings_l2463_246329

/-- Calculates the cost of purchasing windows under the given offer -/
def cost_with_offer (num_windows : ℕ) : ℕ :=
  ((num_windows + 4) / 7 * 5 + (num_windows + 4) % 7) * 100

/-- Represents the window purchase problem -/
theorem window_purchase_savings (dave_windows doug_windows : ℕ) 
  (h1 : dave_windows = 10) (h2 : doug_windows = 11) : 
  (dave_windows + doug_windows) * 100 - cost_with_offer (dave_windows + doug_windows) = 
  (dave_windows * 100 - cost_with_offer dave_windows) + 
  (doug_windows * 100 - cost_with_offer doug_windows) :=
sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l2463_246329


namespace NUMINAMATH_CALUDE_smallest_square_longest_ending_sequence_l2463_246350

/-- A function that returns the length of the longest sequence of the same non-zero digit at the end of a number -/
def longestEndingSequence (n : ℕ) : ℕ :=
  sorry

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  sorry

/-- The theorem stating that 1444 is the smallest square with the longest ending sequence of same non-zero digits -/
theorem smallest_square_longest_ending_sequence :
  ∀ n : ℕ, isPerfectSquare n → n ≠ 1444 → longestEndingSequence n ≤ longestEndingSequence 1444 :=
sorry

end NUMINAMATH_CALUDE_smallest_square_longest_ending_sequence_l2463_246350


namespace NUMINAMATH_CALUDE_initial_onions_l2463_246373

theorem initial_onions (sold : ℕ) (left : ℕ) (h1 : sold = 65) (h2 : left = 33) :
  sold + left = 98 := by
  sorry

end NUMINAMATH_CALUDE_initial_onions_l2463_246373


namespace NUMINAMATH_CALUDE_chris_age_l2463_246367

theorem chris_age (a b c : ℚ) : 
  (a + b + c) / 3 = 10 →
  c - 5 = 2 * a →
  b + 4 = 3/4 * (a + 4) →
  c = 283/15 :=
by sorry

end NUMINAMATH_CALUDE_chris_age_l2463_246367


namespace NUMINAMATH_CALUDE_vanessa_deleted_files_l2463_246321

/-- Calculates the number of deleted files given the initial number of music and video files and the number of files left after deletion. -/
def deleted_files (music_files : ℕ) (video_files : ℕ) (files_left : ℕ) : ℕ :=
  music_files + video_files - files_left

/-- Proves that the number of deleted files is 10 given the specific conditions in the problem. -/
theorem vanessa_deleted_files :
  deleted_files 13 30 33 = 10 := by
  sorry

#eval deleted_files 13 30 33

end NUMINAMATH_CALUDE_vanessa_deleted_files_l2463_246321


namespace NUMINAMATH_CALUDE_square_of_107_l2463_246390

theorem square_of_107 : (107 : ℕ)^2 = 11449 := by
  sorry

end NUMINAMATH_CALUDE_square_of_107_l2463_246390


namespace NUMINAMATH_CALUDE_stock_percentage_is_25_percent_l2463_246323

/-- Calculates the percentage of a stock given the investment amount and income. -/
def stock_percentage (investment income : ℚ) : ℚ :=
  (income * 100) / investment

/-- Theorem stating that the stock percentage is 25% given the specified investment and income. -/
theorem stock_percentage_is_25_percent (investment : ℚ) (income : ℚ) 
  (h1 : investment = 15200)
  (h2 : income = 3800) :
  stock_percentage investment income = 25 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_is_25_percent_l2463_246323


namespace NUMINAMATH_CALUDE_ab_equality_l2463_246318

theorem ab_equality (a b : ℝ) : 2 * a * b + 3 * b * a = 5 * a * b := by
  sorry

end NUMINAMATH_CALUDE_ab_equality_l2463_246318


namespace NUMINAMATH_CALUDE_gcd_2024_2048_l2463_246371

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2024_2048_l2463_246371


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2463_246314

theorem sin_cos_identity (α : ℝ) : 
  Real.sin α ^ 6 + Real.cos α ^ 6 + 3 * (Real.sin α ^ 2) * (Real.cos α ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2463_246314


namespace NUMINAMATH_CALUDE_frank_has_two_ten_dollar_bills_l2463_246396

-- Define the problem parameters
def one_dollar_bills : ℕ := 7
def five_dollar_bills : ℕ := 4
def twenty_dollar_bill : ℕ := 1
def peanut_price_per_pound : ℕ := 3
def change : ℕ := 4
def daily_peanut_consumption : ℕ := 3
def days_in_week : ℕ := 7

-- Define the function to calculate the number of ten-dollar bills
def calculate_ten_dollar_bills : ℕ := 
  let total_without_tens : ℕ := one_dollar_bills + 5 * five_dollar_bills + 20 * twenty_dollar_bill
  let total_peanuts_bought : ℕ := daily_peanut_consumption * days_in_week
  let total_spent : ℕ := peanut_price_per_pound * total_peanuts_bought
  let amount_from_tens : ℕ := total_spent - total_without_tens + change
  amount_from_tens / 10

-- Theorem stating that Frank has exactly 2 ten-dollar bills
theorem frank_has_two_ten_dollar_bills : calculate_ten_dollar_bills = 2 := by
  sorry

end NUMINAMATH_CALUDE_frank_has_two_ten_dollar_bills_l2463_246396


namespace NUMINAMATH_CALUDE_square_of_complex_number_l2463_246353

theorem square_of_complex_number : 
  let z : ℂ := 1 - 2*I
  z^2 = -3 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_square_of_complex_number_l2463_246353


namespace NUMINAMATH_CALUDE_expression_bounds_l2463_246392

-- Define the constraint function
def constraint (x y : ℝ) : Prop := (|x| - 3)^2 + (|y| - 2)^2 = 1

-- Define the expression to be minimized/maximized
def expression (x y : ℝ) : ℝ := |x + 2| + |y + 3|

-- Theorem statement
theorem expression_bounds :
  (∃ x y : ℝ, constraint x y) →
  (∃ min max : ℝ,
    (∀ x y : ℝ, constraint x y → expression x y ≥ min) ∧
    (∃ x y : ℝ, constraint x y ∧ expression x y = min) ∧
    (∀ x y : ℝ, constraint x y → expression x y ≤ max) ∧
    (∃ x y : ℝ, constraint x y ∧ expression x y = max) ∧
    min = 2 - Real.sqrt 2 ∧
    max = 10 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_expression_bounds_l2463_246392


namespace NUMINAMATH_CALUDE_sin_585_degrees_l2463_246381

theorem sin_585_degrees :
  let π : ℝ := Real.pi
  let deg_to_rad (x : ℝ) : ℝ := x * π / 180
  ∀ (sin : ℝ → ℝ),
    (∀ x, sin (x + 2 * π) = sin x) →  -- Periodicity of sine
    (∀ x, sin (x + π) = -sin x) →     -- Sine of sum property
    sin (deg_to_rad 45) = Real.sqrt 2 / 2 →  -- Value of sin 45°
    sin (deg_to_rad 585) = -Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l2463_246381


namespace NUMINAMATH_CALUDE_grid_erasing_game_strategies_l2463_246335

/-- Represents the possible outcomes of the grid erasing game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Defines the grid erasing game -/
def GridErasingGame (rows : Nat) (cols : Nat) : GameOutcome :=
  sorry

/-- Theorem stating the winning strategies for different grid sizes -/
theorem grid_erasing_game_strategies :
  (GridErasingGame 10 12 = GameOutcome.SecondPlayerWins) ∧
  (GridErasingGame 9 10 = GameOutcome.FirstPlayerWins) ∧
  (GridErasingGame 9 11 = GameOutcome.SecondPlayerWins) := by
  sorry

/-- Lemma: In a grid with even dimensions, the second player has a winning strategy -/
lemma even_dimensions_second_player_wins (m n : Nat) 
  (hm : Even m) (hn : Even n) : 
  GridErasingGame m n = GameOutcome.SecondPlayerWins := by
  sorry

/-- Lemma: In a grid with one odd and one even dimension, the first player has a winning strategy -/
lemma odd_even_dimensions_first_player_wins (m n : Nat) 
  (hm : Odd m) (hn : Even n) : 
  GridErasingGame m n = GameOutcome.FirstPlayerWins := by
  sorry

/-- Lemma: In a grid with both odd dimensions, the second player has a winning strategy -/
lemma odd_dimensions_second_player_wins (m n : Nat) 
  (hm : Odd m) (hn : Odd n) : 
  GridErasingGame m n = GameOutcome.SecondPlayerWins := by
  sorry

end NUMINAMATH_CALUDE_grid_erasing_game_strategies_l2463_246335


namespace NUMINAMATH_CALUDE_system_solutions_l2463_246374

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  y = x^3 * (3 - 2*x) ∧
  z = y^3 * (3 - 2*y) ∧
  x = z^3 * (3 - 2*z)

/-- The theorem stating the solutions of the system -/
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ 
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨
     (x = 1 ∧ y = 1 ∧ z = 1) ∨
     (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2463_246374


namespace NUMINAMATH_CALUDE_least_whole_number_subtraction_l2463_246301

-- Define the original ratio
def original_ratio : Rat := 6 / 7

-- Define the comparison ratio
def comparison_ratio : Rat := 16 / 21

-- Define the function that creates the new ratio after subtracting x
def new_ratio (x : ℕ) : Rat := (6 - x) / (7 - x)

-- Statement to prove
theorem least_whole_number_subtraction :
  ∀ x : ℕ, x < 3 → new_ratio x ≥ comparison_ratio ∧
  new_ratio 3 < comparison_ratio :=
by sorry

end NUMINAMATH_CALUDE_least_whole_number_subtraction_l2463_246301


namespace NUMINAMATH_CALUDE_exists_non_multiple_4_perimeter_with_valid_subdivision_l2463_246338

/-- A rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculate the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Check if a real number is a multiple of 4 -/
def isMultipleOf4 (x : ℝ) : Prop := ∃ (n : ℤ), x = 4 * n

/-- Represents a subdivision of a rectangle into smaller rectangles -/
structure Subdivision where
  original : Rectangle
  parts : List Rectangle

/-- A subdivision is valid if all parts have perimeters that are multiples of 4 -/
def isValidSubdivision (s : Subdivision) : Prop :=
  ∀ r ∈ s.parts, isMultipleOf4 (perimeter r)

/-- The main theorem to be proved -/
theorem exists_non_multiple_4_perimeter_with_valid_subdivision :
  ∃ (s : Subdivision), isValidSubdivision s ∧ ¬isMultipleOf4 (perimeter s.original) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_multiple_4_perimeter_with_valid_subdivision_l2463_246338


namespace NUMINAMATH_CALUDE_power_sum_equals_power_implies_exponent_one_l2463_246355

theorem power_sum_equals_power_implies_exponent_one (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → (2^p + 3^p = a^n) → n = 1 := by sorry

end NUMINAMATH_CALUDE_power_sum_equals_power_implies_exponent_one_l2463_246355


namespace NUMINAMATH_CALUDE_ferns_total_cost_l2463_246346

/-- Calculates the total cost of Fern's purchase --/
def calculate_total_cost (high_heels_price : ℝ) (ballet_slippers_ratio : ℝ) 
  (ballet_slippers_count : ℕ) (purse_price : ℝ) (scarf_price : ℝ) 
  (high_heels_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let ballet_slippers_price := high_heels_price * ballet_slippers_ratio
  let total_ballet_slippers := ballet_slippers_price * ballet_slippers_count
  let discounted_high_heels := high_heels_price * (1 - high_heels_discount)
  let subtotal := discounted_high_heels + total_ballet_slippers + purse_price + scarf_price
  subtotal * (1 + sales_tax)

/-- Theorem stating that Fern's total cost is $348.30 --/
theorem ferns_total_cost : 
  calculate_total_cost 60 (2/3) 5 45 25 0.1 0.075 = 348.30 := by
  sorry

end NUMINAMATH_CALUDE_ferns_total_cost_l2463_246346


namespace NUMINAMATH_CALUDE_triangle_angle_b_l2463_246304

/-- In a triangle ABC, if side a = 1, side b = √3, and angle A = 30°, then angle B = 60° -/
theorem triangle_angle_b (a b : ℝ) (A B : ℝ) : 
  a = 1 → b = Real.sqrt 3 → A = π / 6 → B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_b_l2463_246304


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2463_246379

theorem quadratic_solution_difference_squared :
  ∀ p q : ℝ,
  (6 * p^2 - 7 * p - 20 = 0) →
  (6 * q^2 - 7 * q - 20 = 0) →
  p ≠ q →
  (p - q)^2 = 529 / 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2463_246379
