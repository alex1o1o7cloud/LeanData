import Mathlib

namespace NUMINAMATH_CALUDE_root_in_interval_l1844_184450

theorem root_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (-4 : ℝ) (-3 : ℝ) ∧ x^3 + 3*x^2 - x + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1844_184450


namespace NUMINAMATH_CALUDE_good_student_count_l1844_184456

/-- Represents a student in the class -/
inductive Student
| Good
| Troublemaker

/-- The total number of students in the class -/
def totalStudents : Nat := 25

/-- The number of students who made the first statement -/
def firstStatementCount : Nat := 5

/-- The number of students who made the second statement -/
def secondStatementCount : Nat := 20

/-- Checks if the first statement is true for a given number of good students -/
def firstStatementTrue (goodCount : Nat) : Prop :=
  totalStudents - goodCount > (totalStudents - 1) / 2

/-- Checks if the second statement is true for a given number of good students -/
def secondStatementTrue (goodCount : Nat) : Prop :=
  totalStudents - goodCount = 3 * (goodCount - 1)

/-- Theorem stating that the number of good students is either 5 or 7 -/
theorem good_student_count :
  ∃ (goodCount : Nat), (goodCount = 5 ∨ goodCount = 7) ∧
    (firstStatementTrue goodCount ∨ ¬firstStatementTrue goodCount) ∧
    (secondStatementTrue goodCount ∨ ¬secondStatementTrue goodCount) ∧
    goodCount ≤ totalStudents :=
  sorry

end NUMINAMATH_CALUDE_good_student_count_l1844_184456


namespace NUMINAMATH_CALUDE_clothing_purchase_problem_l1844_184462

/-- The problem of determining the number of clothing pieces bought --/
theorem clothing_purchase_problem (total_spent : ℕ) (price1 price2 other_price : ℕ) :
  total_spent = 610 →
  price1 = 49 →
  price2 = 81 →
  other_price = 96 →
  ∃ (n : ℕ), total_spent = price1 + price2 + n * other_price ∧ n + 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_clothing_purchase_problem_l1844_184462


namespace NUMINAMATH_CALUDE_inverse_of_complex_l1844_184469

theorem inverse_of_complex (z : ℂ) (h : z = (1 : ℝ) / 2 + (Real.sqrt 3 / 2) * I) : 
  z⁻¹ = (1 : ℝ) / 2 - (Real.sqrt 3 / 2) * I := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_complex_l1844_184469


namespace NUMINAMATH_CALUDE_compare_powers_l1844_184429

theorem compare_powers : 
  let a : ℝ := 2^(4/3)
  let b : ℝ := 4^(2/5)
  let c : ℝ := 25^(1/3)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_compare_powers_l1844_184429


namespace NUMINAMATH_CALUDE_total_candies_l1844_184441

theorem total_candies (red_candies blue_candies : ℕ) 
  (h1 : red_candies = 145) 
  (h2 : blue_candies = 3264) : 
  red_candies + blue_candies = 3409 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l1844_184441


namespace NUMINAMATH_CALUDE_similar_cube_volume_l1844_184466

theorem similar_cube_volume (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 343 → scale_factor = 2 → 
  (scale_factor ^ 3) * original_volume = 2744 := by
  sorry

end NUMINAMATH_CALUDE_similar_cube_volume_l1844_184466


namespace NUMINAMATH_CALUDE_sandra_betty_orange_ratio_l1844_184468

theorem sandra_betty_orange_ratio :
  ∀ (emily sandra betty : ℕ),
    emily = 7 * sandra →
    betty = 12 →
    emily = 252 →
    sandra / betty = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sandra_betty_orange_ratio_l1844_184468


namespace NUMINAMATH_CALUDE_no_solution_exists_l1844_184433

theorem no_solution_exists : ¬∃ (a b c x : ℝ),
  (2 : ℝ)^(x * 0.15) = 5^(a * Real.sin c) ∧
  ((2 : ℝ)^(x * 0.15))^b = 32 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1844_184433


namespace NUMINAMATH_CALUDE_hyperbola_sequence_fixed_point_l1844_184486

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop := y^2 - x^2 = 2

/-- Definition of the line with slope 2 passing through a point -/
def line_slope_2 (x₀ y₀ x y : ℝ) : Prop := y - y₀ = 2 * (x - x₀)

/-- Definition of the next point in the sequence -/
def next_point (x₀ x₁ : ℝ) : Prop :=
  ∃ y₁, hyperbola x₁ y₁ ∧ line_slope_2 x₀ 0 x₁ y₁

/-- Definition of the sequence of points -/
def point_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n, next_point (x n) (x (n+1)) ∨ x n = 0

/-- The main theorem -/
theorem hyperbola_sequence_fixed_point :
  ∃! k : ℕ, k = (2^2048 - 2) ∧
  ∃ x : ℕ → ℝ, point_sequence x ∧ x 0 = x 2048 ∧ x 0 ≠ 0 ∧
  ∀ y : ℕ → ℝ, point_sequence y ∧ y 0 = y 2048 ∧ y 0 ≠ 0 →
    ∃! i : ℕ, i < k ∧ x 0 = y 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_sequence_fixed_point_l1844_184486


namespace NUMINAMATH_CALUDE_valid_pairings_count_l1844_184489

def num_colors : ℕ := 5

def total_pairings (n : ℕ) : ℕ := n * n

def same_color_pairings (n : ℕ) : ℕ := n

theorem valid_pairings_count :
  total_pairings num_colors - same_color_pairings num_colors = 20 := by
  sorry

end NUMINAMATH_CALUDE_valid_pairings_count_l1844_184489


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l1844_184422

theorem smallest_common_multiple_of_6_and_15 (b : ℕ) : 
  (b % 6 = 0 ∧ b % 15 = 0) → b ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l1844_184422


namespace NUMINAMATH_CALUDE_total_time_spent_on_pictures_johns_total_time_is_34_hours_l1844_184474

/-- Calculates the total time spent on drawing and coloring pictures. -/
theorem total_time_spent_on_pictures 
  (num_pictures : ℕ) 
  (drawing_time : ℝ) 
  (coloring_time_reduction : ℝ) : ℝ :=
  let coloring_time := drawing_time * (1 - coloring_time_reduction)
  let time_per_picture := drawing_time + coloring_time
  num_pictures * time_per_picture

/-- Proves that John spends 34 hours on all pictures given the conditions. -/
theorem johns_total_time_is_34_hours : 
  total_time_spent_on_pictures 10 2 0.3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_time_spent_on_pictures_johns_total_time_is_34_hours_l1844_184474


namespace NUMINAMATH_CALUDE_line_through_intersection_parallel_to_l₃_l1844_184447

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := 4 * x + 3 * y - 2 = 0

-- Define the intersection point of l₁ and l₂
def intersection_point (x y : ℝ) : Prop := l₁ x y ∧ l₂ x y

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g (x + k) y

-- Theorem statement
theorem line_through_intersection_parallel_to_l₃ :
  ∃ (a b c : ℝ), 
    (∀ (x y : ℝ), intersection_point x y → a * x + b * y + c = 0) ∧
    parallel (fun x y => a * x + b * y + c = 0) l₃ ∧
    (a = 4 ∧ b = 3 ∧ c = 2) :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_parallel_to_l₃_l1844_184447


namespace NUMINAMATH_CALUDE_max_value_of_f_on_I_l1844_184454

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

-- Define the interval
def I : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem max_value_of_f_on_I :
  ∃ (M : ℝ), M = 2 ∧ ∀ x ∈ I, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_I_l1844_184454


namespace NUMINAMATH_CALUDE_lcm_1404_972_l1844_184402

theorem lcm_1404_972 : Nat.lcm 1404 972 = 88452 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1404_972_l1844_184402


namespace NUMINAMATH_CALUDE_republicans_count_l1844_184464

/-- Given the total number of representatives and the difference between Republicans and Democrats,
    calculate the number of Republicans. -/
def calculateRepublicans (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

theorem republicans_count :
  calculateRepublicans 434 30 = 232 := by
  sorry

end NUMINAMATH_CALUDE_republicans_count_l1844_184464


namespace NUMINAMATH_CALUDE_expression_evaluation_l1844_184442

theorem expression_evaluation (a b c : ℝ) 
  (ha : a = 12) (hb : b = 14) (hc : c = 19) : 
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1844_184442


namespace NUMINAMATH_CALUDE_hawk_percentage_is_25_percent_l1844_184439

/-- Represents the percentage of hawks in the bird population -/
def hawk_percentage : ℝ := sorry

/-- Represents the percentage of paddyfield-warblers in the bird population -/
def paddyfield_warbler_percentage : ℝ := sorry

/-- Represents the percentage of kingfishers in the bird population -/
def kingfisher_percentage : ℝ := sorry

/-- The percentage of non-hawks that are paddyfield-warblers -/
def paddyfield_warbler_ratio : ℝ := 0.4

/-- The ratio of kingfishers to paddyfield-warblers -/
def kingfisher_to_warbler_ratio : ℝ := 0.25

/-- The percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
def other_birds_percentage : ℝ := 0.35

theorem hawk_percentage_is_25_percent :
  hawk_percentage = 0.25 ∧
  paddyfield_warbler_percentage = paddyfield_warbler_ratio * (1 - hawk_percentage) ∧
  kingfisher_percentage = kingfisher_to_warbler_ratio * paddyfield_warbler_percentage ∧
  hawk_percentage + paddyfield_warbler_percentage + kingfisher_percentage + other_birds_percentage = 1 :=
by sorry

end NUMINAMATH_CALUDE_hawk_percentage_is_25_percent_l1844_184439


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1844_184400

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x^2 + 2*x - 3}

-- Define set B
def B : Set ℝ := {y | ∃ x < 0, y = x + 1/x}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Icc (-4) (-2) := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1844_184400


namespace NUMINAMATH_CALUDE_diagonals_bisect_if_equal_areas_l1844_184476

/-- A quadrilateral in a 2D plane. -/
structure Quadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

/-- The area of a triangle given its vertices. -/
noncomputable def triangleArea {V : Type*} [AddCommGroup V] [Module ℝ V] (A B C : V) : ℝ := sorry

/-- Statement that a line segment divides a quadrilateral into two equal areas. -/
def dividesEquallyBy {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) (P Q : V) : Prop :=
  triangleArea q.A P Q + triangleArea Q P q.D = triangleArea q.B P Q + triangleArea Q P q.C

/-- The intersection point of the diagonals of a quadrilateral. -/
noncomputable def diagonalIntersection {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) : V := sorry

/-- Statement that a point is the midpoint of a line segment. -/
def isMidpoint {V : Type*} [AddCommGroup V] [Module ℝ V] (M A B : V) : Prop :=
  2 • M = A + B

theorem diagonals_bisect_if_equal_areas {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) :
  dividesEquallyBy q q.A q.C → dividesEquallyBy q q.B q.D →
  let E := diagonalIntersection q
  isMidpoint E q.A q.C ∧ isMidpoint E q.B q.D :=
sorry

end NUMINAMATH_CALUDE_diagonals_bisect_if_equal_areas_l1844_184476


namespace NUMINAMATH_CALUDE_school_study_sample_size_l1844_184490

/-- Represents a collection of student report cards -/
structure ReportCardCollection where
  total : Nat
  selected : Nat
  h_selected_le_total : selected ≤ total

/-- Defines the sample size of a report card collection -/
def sampleSize (collection : ReportCardCollection) : Nat :=
  collection.selected

/-- Theorem stating that for the given scenario, the sample size is 100 -/
theorem school_study_sample_size :
  ∀ (collection : ReportCardCollection),
    collection.total = 1000 →
    collection.selected = 100 →
    sampleSize collection = 100 := by
  sorry

end NUMINAMATH_CALUDE_school_study_sample_size_l1844_184490


namespace NUMINAMATH_CALUDE_size_ratio_proof_l1844_184467

def anna_size : ℕ := 2

def becky_size (anna_size : ℕ) : ℕ := 3 * anna_size

def ginger_size : ℕ := 8

theorem size_ratio_proof (anna_size : ℕ) (becky_size : ℕ → ℕ) (ginger_size : ℕ)
  (h1 : anna_size = 2)
  (h2 : becky_size anna_size = 3 * anna_size)
  (h3 : ginger_size = 8)
  (h4 : ∃ k : ℕ, ginger_size = k * (becky_size anna_size - 4)) :
  ginger_size / (becky_size anna_size) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_size_ratio_proof_l1844_184467


namespace NUMINAMATH_CALUDE_mikas_height_l1844_184432

/-- Proves that Mika's current height is 66 inches given the problem conditions --/
theorem mikas_height (initial_height : ℝ) : 
  initial_height > 0 →
  initial_height * 1.25 = 75 →
  initial_height * 1.1 = 66 :=
by
  sorry

#check mikas_height

end NUMINAMATH_CALUDE_mikas_height_l1844_184432


namespace NUMINAMATH_CALUDE_solution_ratio_proof_l1844_184465

/-- Proves that the ratio of solutions A and B is 1:1 when mixed to form a 45% alcohol solution --/
theorem solution_ratio_proof (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) : 
  (4/9 * a + 5/11 * b) / (a + b) = 9/20 → a = b :=
by
  sorry

end NUMINAMATH_CALUDE_solution_ratio_proof_l1844_184465


namespace NUMINAMATH_CALUDE_triangular_prism_surface_area_l1844_184481

/-- The surface area of a triangular-based prism created by vertically cutting a rectangular prism -/
theorem triangular_prism_surface_area (l w h : ℝ) (h_l : l = 3) (h_w : w = 5) (h_h : h = 12) :
  let front_area := l * h
  let side_area := l * w
  let triangle_area := w * h / 2
  let back_diagonal := Real.sqrt (w^2 + h^2)
  let back_area := l * back_diagonal
  front_area + side_area + 2 * triangle_area + back_area = 150 :=
sorry

end NUMINAMATH_CALUDE_triangular_prism_surface_area_l1844_184481


namespace NUMINAMATH_CALUDE_oldest_babysat_prime_age_l1844_184499

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def max_babysit_age (current_age : ℕ) (start_age : ℕ) (stop_age : ℕ) (years_since_stop : ℕ) : ℕ :=
  min (stop_age / 2 + years_since_stop) (current_age - 1)

def satisfies_babysit_criteria (age : ℕ) (max_age : ℕ) (gap : ℕ) : Prop :=
  age ≤ max_age ∧ ∃ n : ℕ, n ≤ gap ∧ max_age - n = age

theorem oldest_babysat_prime_age :
  ∀ (current_age : ℕ) (start_age : ℕ) (stop_age : ℕ) (years_since_stop : ℕ),
    current_age = 32 →
    start_age = 20 →
    stop_age = 22 →
    years_since_stop = 10 →
    ∃ (oldest_age : ℕ),
      is_prime oldest_age ∧
      oldest_age = 19 ∧
      satisfies_babysit_criteria oldest_age (max_babysit_age current_age start_age stop_age years_since_stop) 1 ∧
      ∀ (age : ℕ),
        is_prime age →
        satisfies_babysit_criteria age (max_babysit_age current_age start_age stop_age years_since_stop) 1 →
        age ≤ oldest_age :=
by sorry

end NUMINAMATH_CALUDE_oldest_babysat_prime_age_l1844_184499


namespace NUMINAMATH_CALUDE_tim_found_37_shells_l1844_184413

/-- The number of seashells Sally found -/
def sally_shells : ℕ := 13

/-- The total number of seashells Tim and Sally found together -/
def total_shells : ℕ := 50

/-- The number of seashells Tim found -/
def tim_shells : ℕ := total_shells - sally_shells

theorem tim_found_37_shells : tim_shells = 37 := by
  sorry

end NUMINAMATH_CALUDE_tim_found_37_shells_l1844_184413


namespace NUMINAMATH_CALUDE_water_added_fourth_hour_l1844_184409

-- Define the water tank scenario
def water_tank_scenario (initial_water : ℝ) (loss_rate : ℝ) (added_third_hour : ℝ) (added_fourth_hour : ℝ) : ℝ :=
  initial_water - 4 * loss_rate + added_third_hour + added_fourth_hour

-- Theorem statement
theorem water_added_fourth_hour :
  ∃ (added_fourth_hour : ℝ),
    water_tank_scenario 40 2 1 added_fourth_hour = 36 ∧
    added_fourth_hour = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_water_added_fourth_hour_l1844_184409


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1844_184415

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A square inscribed in the region bound by the parabola and the x-axis -/
structure InscribedSquare where
  center : ℝ
  side_half : ℝ
  lower_left_on_axis : f (center - side_half) = 0
  upper_right_on_parabola : f (center + side_half) = 2 * side_half

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area (s : InscribedSquare) : 
  (2 * s.side_half)^2 = 24 - 16 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1844_184415


namespace NUMINAMATH_CALUDE_sin_alpha_terminal_side_l1844_184472

/-- Given a point P on the terminal side of angle α with coordinates (3a, 4a) where a < 0, prove that sin α = -4/5 -/
theorem sin_alpha_terminal_side (a : ℝ) (α : ℝ) (h : a < 0) :
  let P : ℝ × ℝ := (3 * a, 4 * a)
  (P.1 = 3 * a ∧ P.2 = 4 * a) → Real.sin α = -4/5 :=
by sorry

end NUMINAMATH_CALUDE_sin_alpha_terminal_side_l1844_184472


namespace NUMINAMATH_CALUDE_angle_inequality_theorem_l1844_184484

theorem angle_inequality_theorem (θ : Real) : 
  (π / 2 < θ ∧ θ < π) ↔ 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ + x * (1 - x) - (1 - x)^3 * Real.sin θ < 0) ∧
  (∀ φ : Real, 0 ≤ φ ∧ φ ≤ 2*π ∧ φ ≠ θ → 
    ∃ y : Real, 0 ≤ y ∧ y ≤ 1 ∧ y^3 * Real.cos φ + y * (1 - y) - (1 - y)^3 * Real.sin φ ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_theorem_l1844_184484


namespace NUMINAMATH_CALUDE_quadratic_roots_greater_than_half_l1844_184449

theorem quadratic_roots_greater_than_half (a : ℝ) :
  (∀ x : ℝ, (2 - a) * x^2 - 3 * a * x + 2 * a = 0 → x > (1/2 : ℝ)) ↔ 16/17 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_greater_than_half_l1844_184449


namespace NUMINAMATH_CALUDE_timmy_candies_l1844_184404

theorem timmy_candies : ∃ x : ℕ, 
  (x / 2 - 3) / 2 - 5 = 10 ∧ x = 66 := by
  sorry

end NUMINAMATH_CALUDE_timmy_candies_l1844_184404


namespace NUMINAMATH_CALUDE_division_problem_l1844_184416

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1844_184416


namespace NUMINAMATH_CALUDE_min_value_expression_l1844_184444

theorem min_value_expression (a b c : ℕ) (h1 : b > a) (h2 : a > c) (h3 : c > 0) (h4 : b ≠ 0) :
  ((a + b)^2 + (b + c)^2 + (c - a)^2 + (a - c)^2 : ℚ) / (b^2 : ℚ) ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1844_184444


namespace NUMINAMATH_CALUDE_find_c_value_l1844_184485

theorem find_c_value (a b c : ℝ) 
  (eq1 : 2 * a + 3 = 5)
  (eq2 : b - a = 1)
  (eq3 : c = 2 * b) : 
  c = 4 := by sorry

end NUMINAMATH_CALUDE_find_c_value_l1844_184485


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_properties_l1844_184437

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the line
def line (x y k : ℝ) : Prop := y = k * x + 1

-- Define the foci
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Define the intersection points
variable (A B : ℝ × ℝ)

-- Define the parallel condition
def parallel (F1 A F2 B : ℝ × ℝ) : Prop :=
  (A.1 - F1.1) * (B.2 - F2.2) = (A.2 - F1.2) * (B.1 - F2.1)

-- Define the perpendicular condition
def perpendicular (A F1 F2 : ℝ × ℝ) : Prop :=
  (A.1 - F1.1) * (A.1 - F2.1) + (A.2 - F1.2) * (A.2 - F2.2) = 0

-- Theorem statement
theorem ellipse_line_intersection_properties
  (k : ℝ)
  (hA : ellipse A.1 A.2 ∧ line A.1 A.2 k)
  (hB : ellipse B.1 B.2 ∧ line B.1 B.2 k) :
  ¬(parallel F1 A F2 B) ∧ ¬(perpendicular A F1 F2) := by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_properties_l1844_184437


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l1844_184446

-- Define the function f(x) = -x^3
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l1844_184446


namespace NUMINAMATH_CALUDE_onions_in_basket_l1844_184411

/-- Given a basket of onions with initial count S, prove that after
    Sara adds 4, Sally removes 5, and Fred adds F onions, 
    resulting in 8 more onions than the initial count,
    Fred must have added 9 onions. -/
theorem onions_in_basket (S : ℤ) : ∃ F : ℤ, 
  S - 1 + F = S + 8 ∧ F = 9 := by
  sorry

end NUMINAMATH_CALUDE_onions_in_basket_l1844_184411


namespace NUMINAMATH_CALUDE_survivor_quitter_probability_survivor_quitter_probability_proof_l1844_184480

/-- The probability that both quitters are from the same tribe in a Survivor game -/
theorem survivor_quitter_probability : ℚ :=
  let total_participants : ℕ := 32
  let tribe_size : ℕ := 16
  let num_quitters : ℕ := 2

  -- The probability that both quitters are from the same tribe
  15 / 31

/-- Proof of the survivor_quitter_probability theorem -/
theorem survivor_quitter_probability_proof :
  survivor_quitter_probability = 15 / 31 := by
  sorry

end NUMINAMATH_CALUDE_survivor_quitter_probability_survivor_quitter_probability_proof_l1844_184480


namespace NUMINAMATH_CALUDE_problem_solution_l1844_184459

theorem problem_solution (m n : ℝ) 
  (h1 : (m * Real.exp m) / (4 * n^2) = (Real.log n + Real.log 2) / Real.exp m)
  (h2 : Real.exp (2 * m) = 1 / m) :
  (n = Real.exp m / 2) ∧ 
  (m + n < 7/5) ∧ 
  (1 < 2*n - m^2 ∧ 2*n - m^2 < 3/2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1844_184459


namespace NUMINAMATH_CALUDE_sequence_product_l1844_184493

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_product (a b : ℕ → ℝ) :
  (∀ n, a n ≠ 0) →
  arithmetic_sequence a →
  geometric_sequence b →
  a 4 - 2 * (a 7)^2 + 3 * a 8 = 0 →
  b 7 = a 7 →
  b 2 * b 8 * b 11 = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l1844_184493


namespace NUMINAMATH_CALUDE_median_formulas_l1844_184475

/-- Triangle with sides a, b, c and medians ma, mb, mc -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ma : ℝ
  mb : ℝ
  mc : ℝ

/-- Theorem: Median formula and sum of squares of medians -/
theorem median_formulas (t : Triangle) :
  t.ma^2 = (2*t.b^2 + 2*t.c^2 - t.a^2) / 4 ∧
  t.ma^2 + t.mb^2 + t.mc^2 = 3*(t.a^2 + t.b^2 + t.c^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_median_formulas_l1844_184475


namespace NUMINAMATH_CALUDE_pencil_price_theorem_l1844_184403

/-- The price of a pencil in won -/
def pencil_price : ℚ := 5000 + 20

/-- The conversion factor from won to 10,000 won units -/
def conversion_factor : ℚ := 10000

/-- The price of the pencil in units of 10,000 won -/
def pencil_price_in_units : ℚ := pencil_price / conversion_factor

theorem pencil_price_theorem : pencil_price_in_units = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_theorem_l1844_184403


namespace NUMINAMATH_CALUDE_crayons_per_box_l1844_184478

theorem crayons_per_box (total_crayons : ℕ) (num_boxes : ℕ) (h1 : total_crayons = 56) (h2 : num_boxes = 8) :
  total_crayons / num_boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_box_l1844_184478


namespace NUMINAMATH_CALUDE_greg_books_multiple_l1844_184494

/-- The number of books Megan has read -/
def megan_books : ℕ := 32

/-- The number of books Kelcie has read -/
def kelcie_books : ℕ := megan_books / 4

/-- The total number of books read by all three people -/
def total_books : ℕ := 65

/-- The multiple of Kelcie's books that Greg has read -/
def greg_multiple : ℕ := 2

theorem greg_books_multiple : 
  megan_books + kelcie_books + (greg_multiple * kelcie_books + 9) = total_books :=
sorry

end NUMINAMATH_CALUDE_greg_books_multiple_l1844_184494


namespace NUMINAMATH_CALUDE_birthday_count_theorem_l1844_184477

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a birthday -/
structure Birthday where
  month : ℕ
  day : ℕ

def startDate : Date := ⟨2012, 12, 26⟩

def dadBirthday : Birthday := ⟨5, 1⟩
def chunchunBirthday : Birthday := ⟨7, 1⟩

def daysToCount : ℕ := 2013

/-- Counts the number of birthdays between two dates -/
def countBirthdays (start : Date) (days : ℕ) (birthday : Birthday) : ℕ :=
  sorry

theorem birthday_count_theorem :
  countBirthdays startDate daysToCount dadBirthday +
  countBirthdays startDate daysToCount chunchunBirthday = 11 :=
by sorry

end NUMINAMATH_CALUDE_birthday_count_theorem_l1844_184477


namespace NUMINAMATH_CALUDE_y_decreases_as_x_increases_l1844_184498

def tensor (m n : ℝ) : ℝ := -m * n + n

theorem y_decreases_as_x_increases :
  let f : ℝ → ℝ := λ x ↦ tensor x 2
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₂ < f x₁ := by
  sorry

end NUMINAMATH_CALUDE_y_decreases_as_x_increases_l1844_184498


namespace NUMINAMATH_CALUDE_consecutive_ones_count_l1844_184470

def a : ℕ → ℕ
  | 0 => 1  -- We define a(0) = 1 to simplify the recursion
  | 1 => 2
  | 2 => 3
  | (n + 3) => a (n + 2) + a (n + 1)

theorem consecutive_ones_count : 
  (2^8 : ℕ) - a 8 = 201 :=
sorry

end NUMINAMATH_CALUDE_consecutive_ones_count_l1844_184470


namespace NUMINAMATH_CALUDE_polynomial_real_root_l1844_184407

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^4 - a*x^3 - x^2 - a*x + 1 = 0) ↔ a ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l1844_184407


namespace NUMINAMATH_CALUDE_probability_multiple_of_three_l1844_184410

theorem probability_multiple_of_three (n : ℕ) (h : n = 21) :
  (Finset.filter (fun x => x % 3 = 0) (Finset.range n.succ)).card / n = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_of_three_l1844_184410


namespace NUMINAMATH_CALUDE_road_graveling_cost_l1844_184483

/-- Calculate the cost of graveling two intersecting roads on a rectangular lawn. -/
theorem road_graveling_cost
  (lawn_length : ℕ)
  (lawn_width : ℕ)
  (road_width : ℕ)
  (gravel_cost_per_sqm : ℕ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 40)
  (h3 : road_width = 10)
  (h4 : gravel_cost_per_sqm = 3) :
  lawn_length * road_width + lawn_width * road_width - road_width * road_width * gravel_cost_per_sqm = 3900 :=
by sorry

end NUMINAMATH_CALUDE_road_graveling_cost_l1844_184483


namespace NUMINAMATH_CALUDE_michaels_earnings_l1844_184419

/-- Calculates earnings based on hours worked and pay rates -/
def calculate_earnings (regular_hours : ℝ) (overtime_hours : ℝ) (regular_rate : ℝ) : ℝ :=
  regular_hours * regular_rate + overtime_hours * (2 * regular_rate)

theorem michaels_earnings :
  let total_hours : ℝ := 42.857142857142854
  let regular_hours : ℝ := 40
  let overtime_hours : ℝ := total_hours - regular_hours
  let regular_rate : ℝ := 7
  calculate_earnings regular_hours overtime_hours regular_rate = 320 := by
sorry

end NUMINAMATH_CALUDE_michaels_earnings_l1844_184419


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_lcm_l1844_184412

theorem product_of_numbers_with_given_sum_and_lcm :
  ∃ (a b : ℕ+), 
    (a + b : ℕ) = 210 ∧ 
    Nat.lcm a b = 1547 → 
    (a * b : ℕ) = 10829 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_lcm_l1844_184412


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l1844_184426

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem range_of_a_when_not_p_implies_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬(p x a) → (x ≤ 2 ∨ x > 3)) ∧
           (∃ x : ℝ, (x ≤ 2 ∨ x > 3) ∧ p x a) →
  (1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l1844_184426


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1844_184428

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents income levels -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Structure representing a community's income distribution -/
structure CommunityIncome where
  totalFamilies : ℕ
  highIncome : ℕ
  middleIncome : ℕ
  lowIncome : ℕ
  high_income_valid : highIncome ≤ totalFamilies
  middle_income_valid : middleIncome ≤ totalFamilies
  low_income_valid : lowIncome ≤ totalFamilies
  total_sum_valid : highIncome + middleIncome + lowIncome = totalFamilies

/-- Function to determine the most appropriate sampling method -/
def mostAppropriateSamplingMethod (community : CommunityIncome) (sampleSize : ℕ) : SamplingMethod :=
  SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is most appropriate for the given community -/
theorem stratified_sampling_most_appropriate 
  (community : CommunityIncome) 
  (sampleSize : ℕ) 
  (sample_size_valid : sampleSize ≤ community.totalFamilies) :
  mostAppropriateSamplingMethod community sampleSize = SamplingMethod.Stratified :=
by
  sorry

#check stratified_sampling_most_appropriate

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1844_184428


namespace NUMINAMATH_CALUDE_quadratic_nonnegative_conditions_l1844_184453

theorem quadratic_nonnegative_conditions (a b c : ℝ) (ha : a ≠ 0)
  (hf : ∀ x : ℝ, a * x^2 + 2 * b * x + c ≥ 0) :
  a > 0 ∧ c ≥ 0 ∧ a * c - b^2 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_nonnegative_conditions_l1844_184453


namespace NUMINAMATH_CALUDE_intersection_theorem_l1844_184438

/-- A line passing through two points -/
structure Line1 where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- A line described by y = mx + b -/
structure Line2 where
  m : ℝ
  b : ℝ

/-- The intersection point of two lines -/
def intersection_point (l1 : Line1) (l2 : Line2) : ℝ × ℝ :=
  sorry

theorem intersection_theorem :
  let l1 : Line1 := { x1 := 0, y1 := 3, x2 := 4, y2 := 11 }
  let l2 : Line2 := { m := -1, b := 15 }
  intersection_point l1 l2 = (4, 11) := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l1844_184438


namespace NUMINAMATH_CALUDE_halloween_decorations_l1844_184423

/-- Calculates the number of plastic skulls in Danai's Halloween decorations. -/
theorem halloween_decorations (total_decorations : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) 
  (cauldron : ℕ) (budget_left : ℕ) (left_to_put_up : ℕ) 
  (h1 : total_decorations = 83)
  (h2 : broomsticks = 4)
  (h3 : spiderwebs = 12)
  (h4 : cauldron = 1)
  (h5 : budget_left = 20)
  (h6 : left_to_put_up = 10) :
  total_decorations - (broomsticks + spiderwebs + 2 * spiderwebs + cauldron + budget_left + left_to_put_up) = 12 := by
  sorry

end NUMINAMATH_CALUDE_halloween_decorations_l1844_184423


namespace NUMINAMATH_CALUDE_parallelogram_area_l1844_184463

/-- The area of a parallelogram with a diagonal of length 30 meters and a perpendicular height to that diagonal of 20 meters is 600 square meters. -/
theorem parallelogram_area (d h : ℝ) (hd : d = 30) (hh : h = 20) :
  d * h = 600 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1844_184463


namespace NUMINAMATH_CALUDE_after_two_right_turns_l1844_184491

/-- Represents a position in the square formation -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- The size of the square formation -/
def formationSize : Nat := 9

/-- Counts the number of people in front of a given position -/
def peopleInFront (pos : Position) : Nat :=
  formationSize - pos.row - 1

/-- Performs a right turn on a position -/
def rightTurn (pos : Position) : Position :=
  ⟨formationSize - pos.col + 1, pos.row⟩

/-- The main theorem to prove -/
theorem after_two_right_turns 
  (initialPos : Position)
  (h1 : peopleInFront initialPos = 2)
  (h2 : peopleInFront (rightTurn initialPos) = 4) :
  peopleInFront (rightTurn (rightTurn initialPos)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_after_two_right_turns_l1844_184491


namespace NUMINAMATH_CALUDE_calculation_correctness_l1844_184492

theorem calculation_correctness : 
  (4 + (-2) = 2) ∧ 
  (-2 - (-1.5) = -0.5) ∧ 
  (-(-4) + 4 = 8) ∧ 
  (|-6| + |2| ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_calculation_correctness_l1844_184492


namespace NUMINAMATH_CALUDE_fraction_simplification_l1844_184457

theorem fraction_simplification : (10^9 : ℕ) / (2 * 10^5) = 5000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1844_184457


namespace NUMINAMATH_CALUDE_triangle_with_unit_inradius_is_right_angled_l1844_184435

/-- A triangle with integer side lengths and inradius 1 is right-angled with sides (3, 4, 5) -/
theorem triangle_with_unit_inradius_is_right_angled (a b c : ℕ) (r : ℝ) :
  r = 1 →
  (a : ℝ) + (b : ℝ) + (c : ℝ) = 2 * ((a : ℝ) * (b : ℝ) * (c : ℝ)) / ((a : ℝ) + (b : ℝ) + (c : ℝ)) →
  (a = 3 ∧ b = 4 ∧ c = 5) ∨ (a = 3 ∧ b = 5 ∧ c = 4) ∨
  (a = 4 ∧ b = 3 ∧ c = 5) ∨ (a = 4 ∧ b = 5 ∧ c = 3) ∨
  (a = 5 ∧ b = 3 ∧ c = 4) ∨ (a = 5 ∧ b = 4 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_unit_inradius_is_right_angled_l1844_184435


namespace NUMINAMATH_CALUDE_inequality_proof_l1844_184430

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1844_184430


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1844_184455

theorem quadratic_inequality_condition (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (b^2 - 4*a*c < 0) = False := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1844_184455


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1844_184436

/-- Calculates the total wet surface area of a rectangular cistern -/
def cisternWetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem: The wet surface area of a cistern with given dimensions is 134 square meters -/
theorem cistern_wet_surface_area :
  cisternWetSurfaceArea 10 8 1.5 = 134 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1844_184436


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1844_184406

theorem arithmetic_mean_problem (original_list : List ℝ) (x y z : ℝ) :
  (original_list.length = 12) →
  (original_list.sum / original_list.length = 40) →
  ((original_list.sum + x + y + z) / (original_list.length + 3) = 50) →
  (x + y = 100) →
  z = 170 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1844_184406


namespace NUMINAMATH_CALUDE_three_digit_numbers_count_l1844_184452

/-- Represents a card with two distinct numbers -/
structure Card where
  front : Nat
  back : Nat
  distinct : front ≠ back

/-- The set of cards given in the problem -/
def cards : Finset Card := sorry

/-- The number of cards -/
def num_cards : Nat := Finset.card cards

/-- The number of cards used to form a number -/
def cards_used : Nat := 3

/-- Calculates the number of different three-digit numbers that can be formed -/
def num_three_digit_numbers : Nat :=
  (num_cards.choose cards_used) * (2^cards_used) * (cards_used.factorial)

theorem three_digit_numbers_count :
  num_three_digit_numbers = 192 := by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_count_l1844_184452


namespace NUMINAMATH_CALUDE_fiftieth_term_is_199_l1844_184473

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
theorem fiftieth_term_is_199 :
  arithmetic_sequence 3 4 50 = 199 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_199_l1844_184473


namespace NUMINAMATH_CALUDE_congruence_system_solution_l1844_184497

theorem congruence_system_solution (x : ℤ) :
  (9 * x + 3) % 15 = 6 →
  x % 5 = 2 →
  x % 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_congruence_system_solution_l1844_184497


namespace NUMINAMATH_CALUDE_product_zero_in_special_set_l1844_184487

theorem product_zero_in_special_set (n : ℕ) (h : n = 1997) (S : Finset ℝ) 
  (hcard : S.card = n)
  (hsum : ∀ x ∈ S, (S.sum id - x) ∈ S) :
  S.prod id = 0 :=
sorry

end NUMINAMATH_CALUDE_product_zero_in_special_set_l1844_184487


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l1844_184424

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 675)
  (h2 : selling_price = 1080) : 
  (selling_price - cost_price) / cost_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l1844_184424


namespace NUMINAMATH_CALUDE_max_correct_answers_jesse_l1844_184427

/-- Represents a math contest with given parameters -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  unanswered_points : ℤ
  incorrect_points : ℤ

/-- Represents a contestant's performance in the math contest -/
structure ContestPerformance where
  contest : MathContest
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions for a given contest performance -/
def max_correct_answers (performance : ContestPerformance) : ℕ :=
  sorry

/-- The specific contest Jesse participated in -/
def jesses_contest : MathContest := {
  total_questions := 60,
  correct_points := 4,
  unanswered_points := 0,
  incorrect_points := -1
}

/-- Jesse's performance in the contest -/
def jesses_performance : ContestPerformance := {
  contest := jesses_contest,
  total_score := 112
}

theorem max_correct_answers_jesse :
  max_correct_answers jesses_performance = 34 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_answers_jesse_l1844_184427


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l1844_184434

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight : ℕ := bag_weights.sum

theorem turnip_bag_weights (turnip_weight : ℕ) 
  (h_turnip : turnip_weight ∈ bag_weights) :
  (∃ (onion_weights carrot_weights : List ℕ),
    onion_weights ++ carrot_weights ++ [turnip_weight] = bag_weights ∧
    onion_weights.sum * 2 = carrot_weights.sum ∧
    onion_weights.sum + carrot_weights.sum + turnip_weight = total_weight) ↔
  (turnip_weight = 13 ∨ turnip_weight = 16) :=
sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l1844_184434


namespace NUMINAMATH_CALUDE_highest_result_l1844_184488

def alice_calc (x : ℕ) : ℕ := x * 3 - 2 + 3

def bob_calc (x : ℕ) : ℕ := (x - 3) * 3 + 4

def carla_calc (x : ℕ) : ℕ := x * 3 + 4 - 2

theorem highest_result (start : ℕ) (h : start = 12) : 
  carla_calc start > alice_calc start ∧ carla_calc start > bob_calc start := by
  sorry

end NUMINAMATH_CALUDE_highest_result_l1844_184488


namespace NUMINAMATH_CALUDE_heather_aprons_l1844_184417

/-- The number of aprons Heather sewed before today -/
def aprons_before_today : ℕ := by sorry

/-- The total number of aprons to be sewn -/
def total_aprons : ℕ := 150

/-- The number of aprons Heather sewed today -/
def aprons_today : ℕ := 3 * aprons_before_today

/-- The number of aprons Heather will sew tomorrow -/
def aprons_tomorrow : ℕ := 49

/-- The number of remaining aprons after sewing tomorrow -/
def remaining_aprons : ℕ := aprons_tomorrow

theorem heather_aprons : 
  aprons_before_today = 13 ∧
  aprons_before_today + aprons_today + aprons_tomorrow + remaining_aprons = total_aprons := by
  sorry

end NUMINAMATH_CALUDE_heather_aprons_l1844_184417


namespace NUMINAMATH_CALUDE_number_is_forty_l1844_184420

theorem number_is_forty (N : ℝ) (P : ℝ) : 
  (P / 100) * N = 0.25 * 16 + 2 → N = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_is_forty_l1844_184420


namespace NUMINAMATH_CALUDE_factorization_cubic_l1844_184461

theorem factorization_cubic (a : ℝ) : a^3 - 6*a^2 + 9*a = a*(a-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_l1844_184461


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1844_184414

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x + 1) - 3 * x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 * Real.exp 1 - 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1844_184414


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l1844_184418

/-- The area of a circle with radius r -/
def circle_area (r : ℝ) : ℝ := 3.14 * r ^ 2

/-- The theorem stating the area of the shaded region -/
theorem shaded_area_theorem :
  let large_radius : ℝ := 20
  let small_radius : ℝ := 10
  let num_small_circles : ℕ := 7
  let large_circle_area := circle_area large_radius
  let small_circle_area := circle_area small_radius
  large_circle_area - (num_small_circles : ℝ) * small_circle_area = 942 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l1844_184418


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1844_184458

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1844_184458


namespace NUMINAMATH_CALUDE_power_multiplication_result_l1844_184448

theorem power_multiplication_result : 0.25^2023 * 4^2024 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_result_l1844_184448


namespace NUMINAMATH_CALUDE_hockey_league_games_l1844_184496

theorem hockey_league_games (n : ℕ) (games_per_pair : ℕ) : n = 10 ∧ games_per_pair = 4 →
  (n * (n - 1) / 2) * games_per_pair = 180 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l1844_184496


namespace NUMINAMATH_CALUDE_system_solution_l1844_184405

/-- Prove that the given system of linear equations has the specified solution -/
theorem system_solution (x y : ℝ) : 
  (x = 2 ∧ y = -3) → (3 * x + y = 3 ∧ 4 * x - y = 11) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1844_184405


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1844_184443

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 15 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1844_184443


namespace NUMINAMATH_CALUDE_derivative_bounded_l1844_184421

open Real

/-- Given a function f: ℝ → ℝ with continuous second derivative, 
    and both f and f'' are bounded, prove that f' is also bounded. -/
theorem derivative_bounded (f : ℝ → ℝ) (hf'' : Continuous (deriv (deriv f))) 
  (hf_bdd : ∃ M, ∀ x, |f x| ≤ M) (hf''_bdd : ∃ M, ∀ x, |(deriv (deriv f)) x| ≤ M) :
  ∃ K, ∀ x, |deriv f x| ≤ K := by
  sorry

end NUMINAMATH_CALUDE_derivative_bounded_l1844_184421


namespace NUMINAMATH_CALUDE_coat_price_calculation_l1844_184471

/-- Calculates the total selling price of a coat given its original price, discount percentage, and tax percentage. -/
def totalSellingPrice (originalPrice discount tax : ℚ) : ℚ :=
  let salePrice := originalPrice * (1 - discount)
  salePrice * (1 + tax)

/-- Theorem stating that the total selling price of a coat with original price $120, 30% discount, and 15% tax is $96.60. -/
theorem coat_price_calculation :
  totalSellingPrice 120 (30 / 100) (15 / 100) = 966 / 10 := by
  sorry

#eval totalSellingPrice 120 (30 / 100) (15 / 100)

end NUMINAMATH_CALUDE_coat_price_calculation_l1844_184471


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l1844_184408

/-- Alice's walking speed in miles per minute -/
def alice_speed : ℚ := 1 / 20

/-- Bob's jogging speed in miles per minute -/
def bob_speed : ℚ := 3 / 40

/-- Time elapsed in minutes -/
def time_elapsed : ℚ := 120

/-- The distance between Alice and Bob after 2 hours -/
def distance_between : ℚ := alice_speed * time_elapsed + bob_speed * time_elapsed

theorem distance_after_two_hours :
  distance_between = 15 := by sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l1844_184408


namespace NUMINAMATH_CALUDE_cake_eaters_l1844_184401

theorem cake_eaters (n : ℕ) (h1 : n > 0) : 
  (∃ (portions : Fin n → ℚ), 
    (∀ i, portions i > 0) ∧ 
    (∃ i, portions i = 1/11) ∧ 
    (∃ i, portions i = 1/14) ∧ 
    (∀ i, portions i ≤ 1/11) ∧ 
    (∀ i, portions i ≥ 1/14) ∧ 
    (Finset.sum Finset.univ portions = 1)) ↔ 
  (n = 12 ∨ n = 13) :=
sorry

end NUMINAMATH_CALUDE_cake_eaters_l1844_184401


namespace NUMINAMATH_CALUDE_theodore_stone_statues_l1844_184445

/-- The number of stone statues Theodore crafts every month -/
def stone_statues : ℕ := sorry

/-- The number of wooden statues Theodore crafts every month -/
def wooden_statues : ℕ := 20

/-- The cost of a stone statue in dollars -/
def stone_cost : ℕ := 20

/-- The cost of a wooden statue in dollars -/
def wooden_cost : ℕ := 5

/-- The tax rate as a decimal -/
def tax_rate : ℚ := 1/10

/-- Theodore's total monthly earnings after tax in dollars -/
def total_earnings : ℕ := 270

theorem theodore_stone_statues :
  stone_statues = 10 ∧
  (stone_statues * stone_cost + wooden_statues * wooden_cost) * (1 - tax_rate) = total_earnings :=
sorry

end NUMINAMATH_CALUDE_theodore_stone_statues_l1844_184445


namespace NUMINAMATH_CALUDE_charles_stroll_distance_l1844_184425

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Charles strolled 6 miles -/
theorem charles_stroll_distance :
  let speed : ℝ := 3
  let time : ℝ := 2
  distance speed time = 6 := by sorry

end NUMINAMATH_CALUDE_charles_stroll_distance_l1844_184425


namespace NUMINAMATH_CALUDE_soap_brand_usage_ratio_l1844_184440

/-- Given a survey of households and their soap brand usage, prove the ratio of households
    using only brand B to those using both brands A and B. -/
theorem soap_brand_usage_ratio
  (total_households : ℕ)
  (neither_brand : ℕ)
  (only_brand_A : ℕ)
  (both_brands : ℕ)
  (h1 : total_households = 180)
  (h2 : neither_brand = 80)
  (h3 : only_brand_A = 60)
  (h4 : both_brands = 10)
  (h5 : total_households = neither_brand + only_brand_A + (total_households - neither_brand - only_brand_A - both_brands) + both_brands) :
  (total_households - neither_brand - only_brand_A - both_brands) / both_brands = 3 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_usage_ratio_l1844_184440


namespace NUMINAMATH_CALUDE_circle_through_origin_l1844_184495

theorem circle_through_origin (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 3*x + m + 1 = 0 → (x = 0 ∧ y = 0)) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_circle_through_origin_l1844_184495


namespace NUMINAMATH_CALUDE_statement_A_incorrect_l1844_184451

/-- Represents the process of meiosis and fertilization -/
structure MeiosisFertilization where
  sperm_transformation : Bool
  egg_metabolism_increase : Bool
  homologous_chromosomes_appearance : Bool
  fertilization_randomness : Bool

/-- Represents the correctness of statements about meiosis and fertilization -/
structure Statements where
  A : Bool
  B : Bool
  C : Bool
  D : Bool

/-- The given information about meiosis and fertilization -/
def given_info : MeiosisFertilization :=
  { sperm_transformation := true
  , egg_metabolism_increase := true
  , homologous_chromosomes_appearance := true
  , fertilization_randomness := true }

/-- The correctness of statements based on the given information -/
def statement_correctness (info : MeiosisFertilization) : Statements :=
  { A := false  -- Statement A is incorrect
  , B := info.sperm_transformation && info.egg_metabolism_increase
  , C := info.homologous_chromosomes_appearance
  , D := info.fertilization_randomness }

/-- Theorem stating that statement A is incorrect -/
theorem statement_A_incorrect (info : MeiosisFertilization) :
  (statement_correctness info).A = false := by
  sorry

end NUMINAMATH_CALUDE_statement_A_incorrect_l1844_184451


namespace NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l1844_184482

-- Define the function g(x)
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

-- State the theorem
theorem largest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (c' : ℝ), (∃ (x : ℝ), g c' x = -3) → c' ≤ c) ∧
  (∃ (x : ℝ), g 1 x = -3) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l1844_184482


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1844_184460

theorem sufficient_not_necessary_condition :
  (∃ a : ℝ, a > 1 ∧ 1 / a < 1) ∧
  (∃ a : ℝ, ¬(a > 1) ∧ 1 / a < 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1844_184460


namespace NUMINAMATH_CALUDE_triangle_consecutive_integers_l1844_184479

theorem triangle_consecutive_integers (n : ℕ) :
  let a := n - 1
  let b := n
  let c := n + 1
  let s := (a + b + c) / 2
  let area := n + 2
  (area : ℝ)^2 = s * (s - a) * (s - b) * (s - c) → n = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_consecutive_integers_l1844_184479


namespace NUMINAMATH_CALUDE_dan_picked_more_l1844_184431

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- Theorem: Dan picked 7 more apples than Benny -/
theorem dan_picked_more : dan_apples - benny_apples = 7 := by sorry

end NUMINAMATH_CALUDE_dan_picked_more_l1844_184431
