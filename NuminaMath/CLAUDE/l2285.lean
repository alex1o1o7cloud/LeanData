import Mathlib

namespace NUMINAMATH_CALUDE_library_visit_equation_l2285_228517

/-- Represents the growth of library visits over three months -/
def library_visit_growth (initial_visits : ℕ) (final_visits : ℕ) (growth_rate : ℝ) : Prop :=
  initial_visits * (1 + growth_rate)^2 = final_visits

/-- Theorem stating that the given equation accurately represents the library visit growth -/
theorem library_visit_equation : 
  ∃ (x : ℝ), library_visit_growth 560 830 x :=
sorry

end NUMINAMATH_CALUDE_library_visit_equation_l2285_228517


namespace NUMINAMATH_CALUDE_fernanda_savings_calculation_l2285_228511

/-- Calculates the final amount in Fernanda's savings account after receiving payments from debtors and making a withdrawal. -/
theorem fernanda_savings_calculation 
  (aryan_debt kyro_debt jordan_debt imani_debt : ℚ)
  (aryan_payment_percent kyro_payment_percent jordan_payment_percent imani_payment_percent : ℚ)
  (initial_savings withdrawal : ℚ) : 
  aryan_debt = 2 * kyro_debt →
  aryan_debt = 1200 →
  jordan_debt = 800 →
  imani_debt = 500 →
  aryan_payment_percent = 0.6 →
  kyro_payment_percent = 0.8 →
  jordan_payment_percent = 0.5 →
  imani_payment_percent = 0.25 →
  initial_savings = 300 →
  withdrawal = 120 →
  initial_savings + 
    (aryan_debt * aryan_payment_percent +
     kyro_debt * kyro_payment_percent +
     jordan_debt * jordan_payment_percent +
     imani_debt * imani_payment_percent) -
    withdrawal = 1905 :=
by sorry


end NUMINAMATH_CALUDE_fernanda_savings_calculation_l2285_228511


namespace NUMINAMATH_CALUDE_fraction_simplification_l2285_228514

theorem fraction_simplification :
  (6 + 6 + 6 + 6) / ((-2) * (-2) * (-2) * (-2)) = (4 * 6) / ((-2)^4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2285_228514


namespace NUMINAMATH_CALUDE_ingot_growth_theorem_l2285_228525

def gold_good_multiplier : ℝ := 1.3
def silver_good_multiplier : ℝ := 1.2
def gold_bad_multiplier : ℝ := 0.7
def silver_bad_multiplier : ℝ := 0.8
def total_days : ℕ := 7

theorem ingot_growth_theorem (good_days : ℕ) 
  (h1 : good_days ≤ total_days) 
  (h2 : gold_good_multiplier ^ good_days * gold_bad_multiplier ^ (total_days - good_days) < 1)
  (h3 : silver_good_multiplier ^ good_days * silver_bad_multiplier ^ (total_days - good_days) > 1) : 
  good_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_ingot_growth_theorem_l2285_228525


namespace NUMINAMATH_CALUDE_journey_time_ratio_l2285_228504

/-- Proves that the ratio of return journey time to initial journey time is 3:2 
    given specific speed conditions -/
theorem journey_time_ratio 
  (initial_speed : ℝ) 
  (average_speed : ℝ) 
  (h1 : initial_speed = 51)
  (h2 : average_speed = 34) :
  (1 / average_speed) / (1 / initial_speed) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_ratio_l2285_228504


namespace NUMINAMATH_CALUDE_leak_emptying_time_l2285_228545

/-- Given a pipe that fills a tank in 12 hours and a leak that causes the tank to take 20 hours to fill when both are active, prove that the leak alone will empty the full tank in 30 hours. -/
theorem leak_emptying_time (pipe_fill_rate : ℝ) (combined_fill_rate : ℝ) (leak_empty_rate : ℝ) :
  pipe_fill_rate = 1 / 12 →
  combined_fill_rate = 1 / 20 →
  pipe_fill_rate - leak_empty_rate = combined_fill_rate →
  1 / leak_empty_rate = 30 := by
  sorry

#check leak_emptying_time

end NUMINAMATH_CALUDE_leak_emptying_time_l2285_228545


namespace NUMINAMATH_CALUDE_max_value_abs_sum_l2285_228542

theorem max_value_abs_sum (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) : 
  ∃ (m : ℝ), m = 5 ∧ ∀ x, 0 ≤ x ∧ x ≤ 4 → |x - 2| + |3 - x| ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_abs_sum_l2285_228542


namespace NUMINAMATH_CALUDE_min_distance_sum_l2285_228590

theorem min_distance_sum (x : ℝ) : 
  ∃ (min : ℝ) (x_min : ℝ), 
    min = |x_min - 2| + |x_min - 4| + |x_min - 10| ∧
    min = 8 ∧ 
    x_min = 4 ∧
    ∀ y : ℝ, |y - 2| + |y - 4| + |y - 10| ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l2285_228590


namespace NUMINAMATH_CALUDE_curve_transformation_l2285_228515

/-- Given a curve y = sin(2x) and a scaling transformation x' = 2x, y' = 3y,
    prove that the resulting curve has the equation y' = 3sin(x'). -/
theorem curve_transformation (x x' y y' : ℝ) : 
  y = Real.sin (2 * x) → 
  x' = 2 * x → 
  y' = 3 * y → 
  y' = 3 * Real.sin x' :=
by sorry

end NUMINAMATH_CALUDE_curve_transformation_l2285_228515


namespace NUMINAMATH_CALUDE_no_valid_digit_satisfies_condition_l2285_228521

theorem no_valid_digit_satisfies_condition : ¬ ∃ z : ℕ, z < 10 ∧ 
  ∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ 
    ∀ i : ℕ, i < k → (n^9 / 10^i) % 10 = z :=
by sorry

end NUMINAMATH_CALUDE_no_valid_digit_satisfies_condition_l2285_228521


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l2285_228579

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A quadrilateral in a 2D Cartesian coordinate system -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Predicate to check if two line segments are parallel -/
def parallel (p1 p2 p3 p4 : Point2D) : Prop :=
  (p2.x - p1.x) * (p4.y - p3.y) = (p2.y - p1.y) * (p4.x - p3.x)

theorem parallelogram_fourth_vertex 
  (q : Quadrilateral)
  (h1 : parallel q.A q.B q.D q.C)
  (h2 : parallel q.A q.D q.B q.C)
  (h3 : q.A = Point2D.mk (-2) 0)
  (h4 : q.B = Point2D.mk 6 8)
  (h5 : q.C = Point2D.mk 8 6) :
  q.D = Point2D.mk 0 (-2) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l2285_228579


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2285_228556

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2285_228556


namespace NUMINAMATH_CALUDE_volume_cube_with_pyramids_l2285_228567

/-- The volume of a solid formed by a cube with pyramids on each face --/
theorem volume_cube_with_pyramids (a : ℝ) (a_pos : 0 < a) :
  let cube_volume := a^3
  let sphere_radius := a / 2 * Real.sqrt 3
  let pyramid_height := a / 2 * (Real.sqrt 3 - 1)
  let pyramid_volume := 1 / 3 * a^2 * pyramid_height
  let total_pyramid_volume := 6 * pyramid_volume
  cube_volume + total_pyramid_volume = a^3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_volume_cube_with_pyramids_l2285_228567


namespace NUMINAMATH_CALUDE_c_oxen_count_l2285_228585

/-- Represents the number of oxen-months for a person's grazing arrangement -/
def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

/-- Calculates the share of rent based on oxen-months and total rent -/
def rent_share (own_oxen_months : ℕ) (total_oxen_months : ℕ) (total_rent : ℕ) : ℕ :=
  (own_oxen_months * total_rent) / total_oxen_months

theorem c_oxen_count (x : ℕ) : 
  oxen_months 10 7 + oxen_months 12 5 + oxen_months x 3 = 130 + 3 * x →
  rent_share (oxen_months x 3) (130 + 3 * x) 280 = 72 →
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_c_oxen_count_l2285_228585


namespace NUMINAMATH_CALUDE_pipe_length_proof_l2285_228506

theorem pipe_length_proof (shorter_piece longer_piece total_length : ℕ) : 
  shorter_piece = 28 →
  longer_piece = shorter_piece + 12 →
  total_length = shorter_piece + longer_piece →
  total_length = 68 := by
  sorry

end NUMINAMATH_CALUDE_pipe_length_proof_l2285_228506


namespace NUMINAMATH_CALUDE_geometric_series_squared_sum_l2285_228533

/-- For a convergent geometric series with first term a and common ratio r,
    the sum of the series formed by the absolute values of the squares of the terms
    is a^2 / (1 - |r|^2). -/
theorem geometric_series_squared_sum (a r : ℝ) (h : |r| < 1) :
  ∑' n, |a^2 * r^(2*n)| = a^2 / (1 - |r|^2) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_squared_sum_l2285_228533


namespace NUMINAMATH_CALUDE_count_magic_numbers_l2285_228512

def is_magic_number (N : ℕ) : Prop :=
  ∀ m : ℕ, m > 0 → ∃ k : ℕ, k > 0 ∧ (m * 10^k + N) % N = 0

theorem count_magic_numbers :
  (∃! (L : List ℕ), 
    (∀ N ∈ L, N < 600 ∧ is_magic_number N) ∧
    (∀ N < 600, is_magic_number N → N ∈ L) ∧
    L.length = 13) :=
sorry

end NUMINAMATH_CALUDE_count_magic_numbers_l2285_228512


namespace NUMINAMATH_CALUDE_checkerboard_inner_probability_l2285_228548

/-- The size of one side of the square checkerboard -/
def boardSize : ℕ := 10

/-- The total number of squares on the checkerboard -/
def totalSquares : ℕ := boardSize * boardSize

/-- The number of squares on the perimeter of the checkerboard -/
def perimeterSquares : ℕ := 4 * boardSize - 4

/-- The number of squares not touching the outer edge -/
def innerSquares : ℕ := totalSquares - perimeterSquares

/-- The probability of choosing a square not touching the outer edge -/
def innerProbability : ℚ := innerSquares / totalSquares

theorem checkerboard_inner_probability :
  innerProbability = 16 / 25 := by sorry

end NUMINAMATH_CALUDE_checkerboard_inner_probability_l2285_228548


namespace NUMINAMATH_CALUDE_sum_C_D_equals_negative_ten_l2285_228552

variable (x : ℝ)
variable (C D : ℝ)

theorem sum_C_D_equals_negative_ten :
  (∀ x ≠ 3, C / (x - 3) + D * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) →
  C + D = -10 := by
sorry

end NUMINAMATH_CALUDE_sum_C_D_equals_negative_ten_l2285_228552


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2285_228530

theorem sum_of_cubes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 2 / 3) (h4 : b - a = 3) :
  a^3 + b^3 = 945 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2285_228530


namespace NUMINAMATH_CALUDE_gcd_1821_2993_l2285_228582

theorem gcd_1821_2993 : Nat.gcd 1821 2993 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1821_2993_l2285_228582


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2285_228540

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ - 2*m + 5 = 0 ∧ 
    x₂^2 - 4*x₂ - 2*m + 5 = 0 ∧
    x₁*x₂ + x₁ + x₂ = m^2 + 6) → 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2285_228540


namespace NUMINAMATH_CALUDE_chosen_number_proof_l2285_228592

theorem chosen_number_proof :
  ∃! (x : ℝ), x > 0 ∧ (Real.sqrt (x^2) / 6) - 189 = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l2285_228592


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2285_228568

theorem weight_of_new_person (n : ℕ) (initial_weight replaced_weight new_average_increase : ℝ) :
  n = 8 →
  replaced_weight = 65 →
  new_average_increase = 5 →
  (n * new_average_increase + replaced_weight) = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l2285_228568


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2285_228595

theorem right_triangle_arctan_sum (d e f : ℝ) (h : d^2 + e^2 = f^2) :
  Real.arctan (d / (e + 2*f)) + Real.arctan (e / (d + 2*f)) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2285_228595


namespace NUMINAMATH_CALUDE_altitude_length_l2285_228550

-- Define the right triangle DEF
def RightTriangleDEF (DE DF EF : ℝ) : Prop :=
  DE = 15 ∧ DF = 9 ∧ EF = 12 ∧ DE^2 = DF^2 + EF^2

-- Define the altitude from F to DE
def Altitude (DE DF EF h : ℝ) : Prop :=
  h * DE = 2 * (1/2 * DF * EF)

-- Theorem statement
theorem altitude_length (DE DF EF h : ℝ) 
  (hTriangle : RightTriangleDEF DE DF EF) 
  (hAltitude : Altitude DE DF EF h) : 
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_altitude_length_l2285_228550


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2285_228523

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -1
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 3*x + y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2285_228523


namespace NUMINAMATH_CALUDE_line_parallel_plane_necessary_not_sufficient_l2285_228516

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (planeParallel : Plane → Plane → Prop)
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (lineInPlane : Line → Plane → Prop)

theorem line_parallel_plane_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (distinct : α ≠ β)
  (m_in_α : lineInPlane m α) :
  (∀ (α β : Plane) (m : Line), planeParallel α β → lineInPlane m α → lineParallelPlane m β) ∧
  (∃ (α β : Plane) (m : Line), lineParallelPlane m β ∧ lineInPlane m α ∧ ¬planeParallel α β) :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_plane_necessary_not_sufficient_l2285_228516


namespace NUMINAMATH_CALUDE_largest_fraction_l2285_228553

theorem largest_fraction : 
  (200 : ℚ) / 399 > 5 / 11 ∧
  (200 : ℚ) / 399 > 7 / 15 ∧
  (200 : ℚ) / 399 > 29 / 59 ∧
  (200 : ℚ) / 399 > 251 / 501 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l2285_228553


namespace NUMINAMATH_CALUDE_alpha_range_l2285_228503

open Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 2 * sin x

theorem alpha_range (α : ℝ) :
  α > 0 ∧
  (∀ x, x ∈ Set.Icc 0 α → f x ∈ Set.Icc 1 (3/2)) ∧
  (∃ x₁ x₂, x₁ ∈ Set.Icc 0 α ∧ x₂ ∈ Set.Icc 0 α ∧ f x₁ = 1 ∧ f x₂ = 3/2) →
  α ∈ Set.Icc (π/6) π :=
sorry

end NUMINAMATH_CALUDE_alpha_range_l2285_228503


namespace NUMINAMATH_CALUDE_inscribed_triangle_theorem_l2285_228562

-- Define the parabola C: y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point M
def M : ℝ × ℝ := (1, 2)

-- Define the fixed point P
def P : ℝ × ℝ := (5, -2)

-- Define a line passing through two points
def line_through (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • p1 + t • p2}

-- Define a right triangle
def right_triangle (A B M : ℝ × ℝ) : Prop :=
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0

-- Define the theorem
theorem inscribed_triangle_theorem (A B : ℝ × ℝ) :
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  right_triangle A B M →
  (∃ (N : ℝ × ℝ), N ∈ line_through A B ∧ 
    (N.1 - M.1) * (B.1 - A.1) + (N.2 - M.2) * (B.2 - A.2) = 0) →
  (P ∈ line_through A B) ∧
  (∀ (x y : ℝ), x ≠ 1 → ((x - 3)^2 + y^2 = 8 ↔ 
    (∃ (A' B' : ℝ × ℝ), parabola A'.1 A'.2 ∧ parabola B'.1 B'.2 ∧
      right_triangle A' B' M ∧ (x, y) ∈ line_through A' B' ∧
      (x - M.1) * (B'.1 - A'.1) + (y - M.2) * (B'.2 - A'.2) = 0))) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_theorem_l2285_228562


namespace NUMINAMATH_CALUDE_islander_liar_count_l2285_228513

/-- Represents the type of islander: either a knight or a liar -/
inductive IslanderType
| Knight
| Liar

/-- Represents a group of islanders making a statement -/
structure IslanderGroup where
  size : Nat
  statement : Nat

/-- The total number of islanders -/
def totalIslanders : Nat := 19

/-- The three groups of islanders making statements -/
def groups : List IslanderGroup := [
  { size := 3, statement := 3 },
  { size := 6, statement := 6 },
  { size := 9, statement := 9 }
]

/-- Determines if a statement is true given the actual number of liars -/
def isStatementTrue (statement : Nat) (actualLiars : Nat) : Bool :=
  statement == actualLiars

/-- Determines if an islander is telling the truth based on their type and statement -/
def isTellingTruth (type : IslanderType) (statementTrue : Bool) : Bool :=
  match type with
  | IslanderType.Knight => statementTrue
  | IslanderType.Liar => ¬statementTrue

/-- The main theorem to prove -/
theorem islander_liar_count :
  ∀ (liarCount : Nat),
  (liarCount ≤ totalIslanders) →
  (∀ (group : IslanderGroup),
    group ∈ groups →
    (∀ (type : IslanderType),
      (isTellingTruth type (isStatementTrue group.statement liarCount)) →
      (type = IslanderType.Knight ↔ liarCount = group.statement))) →
  (liarCount = 9 ∨ liarCount = 18 ∨ liarCount = 19) :=
sorry

end NUMINAMATH_CALUDE_islander_liar_count_l2285_228513


namespace NUMINAMATH_CALUDE_first_friend_slices_l2285_228539

def burgers : ℕ := 5
def friends : ℕ := 4
def slices_per_burger : ℕ := 2
def second_friend_slices : ℕ := 2
def third_friend_slices : ℕ := 3
def fourth_friend_slices : ℕ := 3
def era_slices : ℕ := 1

theorem first_friend_slices : 
  burgers * slices_per_burger - 
  (second_friend_slices + third_friend_slices + fourth_friend_slices + era_slices) = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_friend_slices_l2285_228539


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2285_228577

theorem no_integer_solutions : 
  ¬∃ (x y z : ℤ), 
    (x^2 - 4*x*y + 3*y^2 - z^2 = 41) ∧ 
    (-x^2 + 4*y*z + 2*z^2 = 52) ∧ 
    (x^2 + x*y + 5*z^2 = 110) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2285_228577


namespace NUMINAMATH_CALUDE_fraction_simplification_l2285_228538

theorem fraction_simplification (x : ℝ) (h : x ≠ 0) : (4 * x) / (x + 2 * x) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2285_228538


namespace NUMINAMATH_CALUDE_no_two_digit_special_number_l2285_228505

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10
def ones_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ := tens_digit n + ones_digit n
def product_of_digits (n : ℕ) : ℕ := tens_digit n * ones_digit n

theorem no_two_digit_special_number :
  ¬∃ n : ℕ, is_two_digit n ∧ 
    (sum_of_digits n + 1) ∣ n ∧ 
    (product_of_digits n + 1) ∣ n :=
by sorry

end NUMINAMATH_CALUDE_no_two_digit_special_number_l2285_228505


namespace NUMINAMATH_CALUDE_direction_vector_of_bisecting_line_l2285_228531

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 2) + 2

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 0

-- Define what it means for a line to bisect a circle
def bisects (k : ℝ) : Prop := ∃ (x₀ y₀ : ℝ), circle_C x₀ y₀ ∧ line_l k x₀ y₀

-- Theorem statement
theorem direction_vector_of_bisecting_line :
  ∃ (k : ℝ), bisects k → ∃ (t : ℝ), t ≠ 0 ∧ (2 = t * 2 ∧ 2 = t * 2) :=
sorry

end NUMINAMATH_CALUDE_direction_vector_of_bisecting_line_l2285_228531


namespace NUMINAMATH_CALUDE_train_platform_length_equality_l2285_228597

/-- Given a train and a platform with specific conditions, prove that the length of the platform equals the length of the train. -/
theorem train_platform_length_equality
  (train_speed : Real) -- Speed of the train in km/hr
  (crossing_time : Real) -- Time to cross the platform in minutes
  (train_length : Real) -- Length of the train in meters
  (h1 : train_speed = 180) -- Train speed is 180 km/hr
  (h2 : crossing_time = 1) -- Time to cross the platform is 1 minute
  (h3 : train_length = 1500) -- Length of the train is 1500 meters
  : Real := -- Length of the platform in meters
by
  sorry

#check train_platform_length_equality

end NUMINAMATH_CALUDE_train_platform_length_equality_l2285_228597


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2285_228575

theorem three_numbers_sum (a b c : ℝ) :
  a ≤ b → b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 10 →
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2285_228575


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2285_228564

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_roots : a 2 * a 6 = 81 ∧ a 2 + a 6 = 34) : 
  a 4 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2285_228564


namespace NUMINAMATH_CALUDE_men_earnings_l2285_228572

/-- The amount earned by a group of workers given their daily wage and work duration -/
def amount_earned (num_workers : ℕ) (days : ℕ) (daily_wage : ℕ) : ℕ :=
  num_workers * days * daily_wage

theorem men_earnings (woman_daily_wage : ℕ) :
  woman_daily_wage * 40 * 30 = 21600 →
  amount_earned 16 25 (2 * woman_daily_wage) = 14400 := by
  sorry

#check men_earnings

end NUMINAMATH_CALUDE_men_earnings_l2285_228572


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2285_228529

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_not_all_equal : ¬(a = b ∧ b = c)) : 
  (((a - b*c)^2 + (b - a*c)^2 + (c - a*b)^2) ≥ 
   (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2)) ∧
  ((a - b*c)^2 + (b - a*c)^2 + (c - a*b)^2 = 
   (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ↔ 
   ((a > 0 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b > 0 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c > 0))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2285_228529


namespace NUMINAMATH_CALUDE_inequality_solution_condition_sum_a_b_l2285_228501

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|
def g (a x : ℝ) : ℝ := a - |x - 2|

-- Theorem for part 1
theorem inequality_solution_condition (a : ℝ) :
  (∃ x, f x < g a x) ↔ a > 4 :=
sorry

-- Theorem for part 2
theorem sum_a_b (a b : ℝ) :
  (∀ x, f x < g a x ↔ b < x ∧ x < 7/2) →
  a + b = 6 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_condition_sum_a_b_l2285_228501


namespace NUMINAMATH_CALUDE_tangent_parallel_point_l2285_228589

theorem tangent_parallel_point (x y : ℝ) : 
  y = x^4 - x →                           -- Curve equation
  (4 * x^3 - 1 : ℝ) = 3 →                 -- Tangent slope equals 3
  (x = 1 ∧ y = 0) :=                      -- Coordinates of point P
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_point_l2285_228589


namespace NUMINAMATH_CALUDE_arcsin_neg_one_l2285_228598

theorem arcsin_neg_one : Real.arcsin (-1) = -π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_neg_one_l2285_228598


namespace NUMINAMATH_CALUDE_root_equation_solution_l2285_228591

theorem root_equation_solution (a b c : ℕ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (∀ N : ℝ, N ≠ 1 → (N^(1/a) * (N^(1/b) * N^(3/c))^(1/a))^a = N^(15/24)) →
  c = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_solution_l2285_228591


namespace NUMINAMATH_CALUDE_toy_bridge_weight_l2285_228580

/-- The weight that a toy bridge must support given the following conditions:
  * There are 6 cans of soda, each containing 12 ounces of soda
  * Each empty can weighs 2 ounces
  * There are 2 additional empty cans
-/
theorem toy_bridge_weight (soda_cans : ℕ) (soda_per_can : ℕ) (empty_can_weight : ℕ) (additional_cans : ℕ) :
  soda_cans = 6 →
  soda_per_can = 12 →
  empty_can_weight = 2 →
  additional_cans = 2 →
  (soda_cans * soda_per_can) + ((soda_cans + additional_cans) * empty_can_weight) = 88 := by
  sorry

end NUMINAMATH_CALUDE_toy_bridge_weight_l2285_228580


namespace NUMINAMATH_CALUDE_square_of_binomial_exclusion_l2285_228510

theorem square_of_binomial_exclusion (a b x m : ℝ) : 
  (∃ p q : ℝ, (x + a) * (x - a) = p^2 - q^2) ∧ 
  (∃ p q : ℝ, (-x - b) * (x - b) = -(p^2 - q^2)) ∧ 
  (∃ p q : ℝ, (b + m) * (m - b) = p^2 - q^2) ∧ 
  ¬(∃ p : ℝ, (a + b) * (-a - b) = p^2) :=
by sorry

end NUMINAMATH_CALUDE_square_of_binomial_exclusion_l2285_228510


namespace NUMINAMATH_CALUDE_dam_building_problem_l2285_228508

/-- Represents the work rate of beavers building a dam -/
def work_rate (beavers : ℕ) (hours : ℝ) : ℝ := beavers * hours

/-- The number of beavers in the second group -/
def second_group_beavers : ℕ := 12

theorem dam_building_problem :
  let first_group_beavers : ℕ := 20
  let first_group_hours : ℝ := 3
  let second_group_hours : ℝ := 5
  work_rate first_group_beavers first_group_hours = work_rate second_group_beavers second_group_hours :=
by
  sorry

end NUMINAMATH_CALUDE_dam_building_problem_l2285_228508


namespace NUMINAMATH_CALUDE_sine_of_supplementary_angles_l2285_228528

theorem sine_of_supplementary_angles (VPQ VPS : Real) 
  (h1 : VPS + VPQ = Real.pi)  -- Supplementary angles
  (h2 : Real.sin VPQ = 3/5) : 
  Real.sin VPS = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_supplementary_angles_l2285_228528


namespace NUMINAMATH_CALUDE_annie_initial_money_l2285_228559

def hamburger_price : ℕ := 4
def cheeseburger_price : ℕ := 5
def fries_price : ℕ := 3
def milkshake_price : ℕ := 5
def smoothie_price : ℕ := 6

def hamburger_count : ℕ := 8
def cheeseburger_count : ℕ := 5
def fries_count : ℕ := 3
def milkshake_count : ℕ := 6
def smoothie_count : ℕ := 4

def discount : ℕ := 10
def money_left : ℕ := 45

def total_cost : ℕ := 
  hamburger_price * hamburger_count +
  cheeseburger_price * cheeseburger_count +
  fries_price * fries_count +
  milkshake_price * milkshake_count +
  smoothie_price * smoothie_count

def discounted_cost : ℕ := total_cost - discount

theorem annie_initial_money : 
  discounted_cost + money_left = 155 := by sorry

end NUMINAMATH_CALUDE_annie_initial_money_l2285_228559


namespace NUMINAMATH_CALUDE_minimize_plates_l2285_228574

/-- Represents the number of units of each product produced by a single plate of each type -/
def PlateProduction := Fin 2 → Fin 2 → ℕ

/-- The required production amounts for products A and B -/
def RequiredProduction := Fin 2 → ℕ

/-- The solution represented as the number of plates of each type used -/
def Solution := Fin 2 → ℕ

/-- Checks if a solution satisfies the production requirements -/
def satisfiesRequirements (plate_prod : PlateProduction) (req_prod : RequiredProduction) (sol : Solution) : Prop :=
  ∀ i, (sol 0 * plate_prod 0 i + sol 1 * plate_prod 1 i) = req_prod i

/-- Calculates the total number of plates used in a solution -/
def totalPlates (sol : Solution) : ℕ :=
  sol 0 + sol 1

theorem minimize_plates (plate_prod : PlateProduction) (req_prod : RequiredProduction) :
  let solution : Solution := ![6, 2]
  satisfiesRequirements plate_prod req_prod solution ∧
  (∀ other : Solution, satisfiesRequirements plate_prod req_prod other →
    totalPlates solution ≤ totalPlates other) :=
by
  sorry

end NUMINAMATH_CALUDE_minimize_plates_l2285_228574


namespace NUMINAMATH_CALUDE_total_blue_balloons_l2285_228551

theorem total_blue_balloons (joan_balloons melanie_balloons : ℕ) 
  (h1 : joan_balloons = 40) 
  (h2 : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l2285_228551


namespace NUMINAMATH_CALUDE_equation_solution_l2285_228549

theorem equation_solution :
  ∃ x : ℚ, (3 / (2 * x - 2) + 1 / (1 - x) = 3) ∧ (x = 7 / 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2285_228549


namespace NUMINAMATH_CALUDE_domino_coloring_l2285_228535

/-- 
Given an m × n board where m and n are natural numbers and mn is even,
prove that the smallest non-negative integer V such that in each row,
the number of squares covered by red dominoes and the number of squares
covered by blue dominoes are each at most V, is equal to n.
-/
theorem domino_coloring (m n : ℕ) (h : Even (m * n)) : 
  ∃ V : ℕ, (∀ row_red row_blue : ℕ, row_red + row_blue = n → row_red ≤ V ∧ row_blue ≤ V) ∧
  (∀ W : ℕ, W < n → ∃ row_red row_blue : ℕ, row_red + row_blue = n ∧ (row_red > W ∨ row_blue > W)) :=
by sorry

end NUMINAMATH_CALUDE_domino_coloring_l2285_228535


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_square_l2285_228502

theorem power_of_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_square_l2285_228502


namespace NUMINAMATH_CALUDE_circle_segment_angle_l2285_228587

theorem circle_segment_angle (r₁ r₂ r₃ : ℝ) (shaded_ratio : ℝ) :
  r₁ = 4 →
  r₂ = 3 →
  r₃ = 2 →
  shaded_ratio = 3 / 5 →
  ∃ θ : ℝ,
    θ > 0 ∧
    θ < π / 2 ∧
    (θ * (r₁^2 + r₂^2 + r₃^2)) / ((π - θ) * (r₁^2 + r₂^2 + r₃^2)) = shaded_ratio ∧
    θ = 3 * π / 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_segment_angle_l2285_228587


namespace NUMINAMATH_CALUDE_parallel_distance_theorem_l2285_228527

/-- Represents a line in a plane -/
structure Line where
  -- We don't need to define the internal structure of a line for this problem

/-- Represents the distance between two lines -/
def distance (l1 l2 : Line) : ℝ := sorry

/-- States that two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

theorem parallel_distance_theorem (a b c : Line) :
  parallel a b ∧ parallel b c ∧ parallel a c →
  distance a b = 5 →
  distance a c = 2 →
  (distance b c = 3 ∨ distance b c = 7) := by sorry

end NUMINAMATH_CALUDE_parallel_distance_theorem_l2285_228527


namespace NUMINAMATH_CALUDE_haunted_mansion_scenarios_l2285_228593

theorem haunted_mansion_scenarios (windows : ℕ) (rooms : ℕ) : windows = 8 → rooms = 3 → windows * (windows - 1) * rooms = 168 := by
  sorry

end NUMINAMATH_CALUDE_haunted_mansion_scenarios_l2285_228593


namespace NUMINAMATH_CALUDE_train_passing_time_l2285_228500

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 150 ∧ 
  train_speed = 62 * (1000 / 3600) ∧ 
  man_speed = 8 * (1000 / 3600) →
  train_length / (train_speed - man_speed) = 10 := by
  sorry


end NUMINAMATH_CALUDE_train_passing_time_l2285_228500


namespace NUMINAMATH_CALUDE_fraction_simplification_l2285_228544

theorem fraction_simplification : 
  (2+4+6+8+10+12+14+16+18+20) / (1+2+3+4+5+6+7+8+9+10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2285_228544


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15_to_100_l2285_228507

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceTerms (a : ℕ) (d : ℕ) (lastTerm : ℕ) : ℕ :=
  (lastTerm - a) / d + 1

/-- Theorem: The arithmetic sequence with first term 15, last term 100, and common difference 5 has 18 terms -/
theorem arithmetic_sequence_15_to_100 :
  arithmeticSequenceTerms 15 5 100 = 18 := by
  sorry

#eval arithmeticSequenceTerms 15 5 100

end NUMINAMATH_CALUDE_arithmetic_sequence_15_to_100_l2285_228507


namespace NUMINAMATH_CALUDE_letterbox_strip_height_calculation_l2285_228547

/-- Represents a screen with width, height, and diagonal measurements -/
structure Screen where
  width : ℝ
  height : ℝ
  diagonal : ℝ

/-- Represents an aspect ratio as a pair of numbers -/
structure AspectRatio where
  horizontal : ℝ
  vertical : ℝ

/-- Calculates the height of letterbox strips when showing a movie on a TV -/
def letterboxStripHeight (tv : Screen) (movieRatio : AspectRatio) : ℝ :=
  sorry

theorem letterbox_strip_height_calculation 
  (tv : Screen)
  (movieRatio : AspectRatio)
  (h1 : tv.diagonal = 27)
  (h2 : tv.width / tv.height = 4 / 3)
  (h3 : movieRatio.horizontal / movieRatio.vertical = 2 / 1)
  (h4 : tv.width^2 + tv.height^2 = tv.diagonal^2) :
  letterboxStripHeight tv movieRatio = 2.7 := by
  sorry

end NUMINAMATH_CALUDE_letterbox_strip_height_calculation_l2285_228547


namespace NUMINAMATH_CALUDE_cranberry_juice_unit_cost_l2285_228565

/-- Given a 12-ounce can of cranberry juice selling for 84 cents, 
    prove that the unit cost is 7 cents per ounce. -/
theorem cranberry_juice_unit_cost 
  (can_size : ℕ) 
  (total_cost : ℕ) 
  (h1 : can_size = 12)
  (h2 : total_cost = 84) :
  total_cost / can_size = 7 :=
sorry

end NUMINAMATH_CALUDE_cranberry_juice_unit_cost_l2285_228565


namespace NUMINAMATH_CALUDE_petpals_center_total_cats_l2285_228541

/-- Represents the PetPals Training Center -/
structure PetPalsCenter where
  jump : ℕ
  fetch : ℕ
  spin : ℕ
  jump_fetch : ℕ
  fetch_spin : ℕ
  jump_spin : ℕ
  all_three : ℕ
  none : ℕ

/-- The number of cats in the PetPals Training Center -/
def total_cats (center : PetPalsCenter) : ℕ :=
  center.all_three +
  (center.jump_fetch - center.all_three) +
  (center.fetch_spin - center.all_three) +
  (center.jump_spin - center.all_three) +
  (center.jump - center.jump_fetch - center.jump_spin + center.all_three) +
  (center.fetch - center.jump_fetch - center.fetch_spin + center.all_three) +
  (center.spin - center.jump_spin - center.fetch_spin + center.all_three) +
  center.none

/-- Theorem stating the total number of cats in the PetPals Training Center -/
theorem petpals_center_total_cats :
  ∀ (center : PetPalsCenter),
    center.jump = 60 →
    center.fetch = 40 →
    center.spin = 50 →
    center.jump_fetch = 25 →
    center.fetch_spin = 20 →
    center.jump_spin = 30 →
    center.all_three = 15 →
    center.none = 5 →
    total_cats center = 95 := by
  sorry

end NUMINAMATH_CALUDE_petpals_center_total_cats_l2285_228541


namespace NUMINAMATH_CALUDE_dog_accessible_area_l2285_228520

/-- Represents the shed's dimensions and rope configuration --/
structure DogTieSetup where
  shedSideLength : ℝ
  ropeLength : ℝ
  attachmentDistance : ℝ

/-- Calculates the area accessible to the dog --/
def accessibleArea (setup : DogTieSetup) : ℝ :=
  sorry

/-- Theorem stating the area accessible to the dog --/
theorem dog_accessible_area (setup : DogTieSetup) 
  (h1 : setup.shedSideLength = 30)
  (h2 : setup.ropeLength = 10)
  (h3 : setup.attachmentDistance = 5) :
  accessibleArea setup = 37.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_dog_accessible_area_l2285_228520


namespace NUMINAMATH_CALUDE_garden_area_difference_l2285_228588

def alice_length : ℝ := 15
def alice_width : ℝ := 30
def bob_length : ℝ := 18
def bob_width : ℝ := 28

theorem garden_area_difference :
  bob_length * bob_width - alice_length * alice_width = 54 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_difference_l2285_228588


namespace NUMINAMATH_CALUDE_value_of_y_l2285_228537

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.25 * y) (h2 : x = 24) : y = 144 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2285_228537


namespace NUMINAMATH_CALUDE_brown_family_probability_l2285_228563

/-- The number of children Mr. Brown has. -/
def num_children : ℕ := 8

/-- The probability of a child being a twin. -/
def twin_probability : ℚ := 1/10

/-- The probability of a child being male (or female). -/
def gender_probability : ℚ := 1/2

/-- Calculates the probability of having an unequal number of sons and daughters
    given the number of children, twin probability, and gender probability. -/
def unequal_gender_probability (n : ℕ) (p_twin : ℚ) (p_gender : ℚ) : ℚ :=
  sorry

theorem brown_family_probability :
  unequal_gender_probability num_children twin_probability gender_probability = 95/128 :=
sorry

end NUMINAMATH_CALUDE_brown_family_probability_l2285_228563


namespace NUMINAMATH_CALUDE_seventh_term_is_13_l2285_228571

def fibonacci_like : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci_like n + fibonacci_like (n + 1)

theorem seventh_term_is_13 : fibonacci_like 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_13_l2285_228571


namespace NUMINAMATH_CALUDE_cuboid_height_is_11_l2285_228576

/-- Represents the dimensions and surface area of a rectangular cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ
  surface_area : ℝ

/-- The surface area formula for a rectangular cuboid -/
def surface_area_formula (c : Cuboid) : ℝ :=
  2 * c.length * c.width + 2 * c.length * c.height + 2 * c.width * c.height

/-- Theorem stating that a cuboid with given dimensions has the specified height -/
theorem cuboid_height_is_11 (c : Cuboid) 
    (h_area : c.surface_area = 442)
    (h_width : c.width = 7)
    (h_length : c.length = 8)
    (h_formula : c.surface_area = surface_area_formula c) :
    c.height = 11 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_is_11_l2285_228576


namespace NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l2285_228581

theorem triangle_circumcircle_diameter 
  (a b c : ℝ) 
  (ha : a = 25) 
  (hb : b = 39) 
  (hc : c = 40) : 
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  2 * (a * b * c) / (4 * area) = 125 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l2285_228581


namespace NUMINAMATH_CALUDE_middle_building_height_l2285_228534

/-- The height of the middle building in feet -/
def middle_height : ℝ := sorry

/-- The height of the left building in feet -/
def left_height : ℝ := 0.8 * middle_height

/-- The height of the right building in feet -/
def right_height : ℝ := middle_height + left_height - 20

/-- The total height of all three buildings in feet -/
def total_height : ℝ := 340

theorem middle_building_height :
  middle_height + left_height + right_height = total_height →
  middle_height = 340 / 5.2 :=
by sorry

end NUMINAMATH_CALUDE_middle_building_height_l2285_228534


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l2285_228532

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 2 / 3)
  (hdb : d / b = 1 / 5) : 
  a / c = 75 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l2285_228532


namespace NUMINAMATH_CALUDE_marching_band_members_l2285_228569

theorem marching_band_members : ∃! n : ℕ, 
  150 < n ∧ n < 250 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 7 = 2 ∧ 
  n = 163 := by
sorry

end NUMINAMATH_CALUDE_marching_band_members_l2285_228569


namespace NUMINAMATH_CALUDE_investment_time_solution_l2285_228578

/-- Represents a partner in the investment problem -/
structure Partner where
  investment : ℝ
  time : ℝ
  profit : ℝ

/-- The investment problem -/
def InvestmentProblem (p q : Partner) : Prop :=
  p.investment / q.investment = 7 / 5 ∧
  p.profit / q.profit = 7 / 10 ∧
  p.time = 7 ∧
  p.investment * p.time / (q.investment * q.time) = p.profit / q.profit

theorem investment_time_solution (p q : Partner) :
  InvestmentProblem p q → q.time = 14 := by
  sorry

end NUMINAMATH_CALUDE_investment_time_solution_l2285_228578


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l2285_228526

theorem complex_equation_real_solution (a : ℝ) :
  (∃ x : ℝ, (1 + Complex.I) * x^2 - 2 * (a + Complex.I) * x + (5 - 3 * Complex.I) = 0) →
  (a = 7/3 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l2285_228526


namespace NUMINAMATH_CALUDE_max_grid_size_is_five_l2285_228554

/-- A coloring of an n × n grid using two colors. -/
def Coloring (n : ℕ) := Fin n → Fin n → Bool

/-- Predicate to check if a rectangle in the grid has all corners of the same color. -/
def hasMonochromaticRectangle (c : Coloring n) : Prop :=
  ∃ (i j k l : Fin n), i < k ∧ j < l ∧
    c i j = c i l ∧ c i l = c k j ∧ c k j = c k l

/-- The maximum size of a grid that can be colored without monochromatic rectangles. -/
def maxGridSize : ℕ := 5

/-- Theorem stating that 5 is the maximum size of a grid that can be colored
    with two colors such that no rectangle has all four corners the same color. -/
theorem max_grid_size_is_five :
  (∀ n : ℕ, n ≤ maxGridSize →
    ∃ c : Coloring n, ¬hasMonochromaticRectangle c) ∧
  (∀ n : ℕ, n > maxGridSize →
    ∀ c : Coloring n, hasMonochromaticRectangle c) :=
sorry

end NUMINAMATH_CALUDE_max_grid_size_is_five_l2285_228554


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l2285_228522

/-- The number of candy pieces Bobby initially had -/
def initial_candy : ℕ := sorry

/-- The number of candy pieces Bobby ate first -/
def first_eaten : ℕ := 17

/-- The number of candy pieces Bobby ate second -/
def second_eaten : ℕ := 15

/-- The number of candy pieces Bobby has left -/
def remaining_candy : ℕ := 4

theorem bobby_candy_problem : 
  initial_candy = first_eaten + second_eaten + remaining_candy := by sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l2285_228522


namespace NUMINAMATH_CALUDE_share_of_y_is_54_l2285_228543

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total amount in rupees -/
def total_amount : ℝ := 234

/-- The share distribution satisfies the given conditions -/
def is_valid_distribution (s : ShareDistribution) : Prop :=
  s.y = 0.45 * s.x ∧ s.z = 0.5 * s.x ∧ s.x + s.y + s.z = total_amount

/-- The share of y in a valid distribution is 54 rupees -/
theorem share_of_y_is_54 (s : ShareDistribution) (h : is_valid_distribution s) : s.y = 54 := by
  sorry


end NUMINAMATH_CALUDE_share_of_y_is_54_l2285_228543


namespace NUMINAMATH_CALUDE_equation_holds_iff_specific_pairs_l2285_228594

def S (r : ℕ) (x y z : ℝ) : ℝ := x^r + y^r + z^r

theorem equation_holds_iff_specific_pairs (m n : ℕ) (x y z : ℝ) 
  (h : x + y + z = 0) :
  (∀ (x y z : ℝ), x + y + z = 0 → 
    S (m + n) x y z / (m + n : ℝ) = (S m x y z / m) * (S n x y z / n)) ↔ 
  ((m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 5) ∨ (m = 5 ∧ n = 2)) :=
sorry

end NUMINAMATH_CALUDE_equation_holds_iff_specific_pairs_l2285_228594


namespace NUMINAMATH_CALUDE_smallest_m_fibonacci_l2285_228555

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def representable (F : ℕ → ℕ) (x : List ℕ) (n : ℕ) : Prop :=
  ∃ (subset : List ℕ), subset.Sublist x ∧ subset.sum = F n

theorem smallest_m_fibonacci :
  ∃ (m : ℕ) (x : List ℕ),
    (∀ i, i ∈ x → i > 0) ∧
    (∀ n, n ≤ 2018 → representable fibonacci x n) ∧
    (∀ m' < m, ¬∃ x' : List ℕ,
      (∀ i, i ∈ x' → i > 0) ∧
      (∀ n, n ≤ 2018 → representable fibonacci x' n)) ∧
    m = 1009 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_fibonacci_l2285_228555


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2285_228560

/-- Given two vectors a and b in ℝ², prove that the angle between them is 2π/3 -/
theorem angle_between_vectors (a b : ℝ × ℝ) : 
  a = (1, Real.sqrt 3) → 
  ‖b‖ = 1 → 
  ‖a + b‖ = Real.sqrt 3 → 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2285_228560


namespace NUMINAMATH_CALUDE_unique_solution_prime_cube_equation_l2285_228519

theorem unique_solution_prime_cube_equation :
  ∀ (p m n : ℕ), 
    Prime p → 
    1 + p^n = m^3 → 
    p = 7 ∧ n = 1 ∧ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_prime_cube_equation_l2285_228519


namespace NUMINAMATH_CALUDE_problem_1_l2285_228573

theorem problem_1 (x y : ℝ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2285_228573


namespace NUMINAMATH_CALUDE_x_minus_y_squared_l2285_228561

theorem x_minus_y_squared (x y : ℝ) (hx : x^2 = 4) (hy : y^2 = 9) :
  (x - y)^2 = 25 ∨ (x - y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_squared_l2285_228561


namespace NUMINAMATH_CALUDE_units_digit_of_seven_power_l2285_228536

theorem units_digit_of_seven_power (n : ℕ) : (7^(6^5) : ℕ) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_power_l2285_228536


namespace NUMINAMATH_CALUDE_unique_solution_l2285_228570

/-- Returns the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- Returns the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Returns the product of digits of a two-digit number -/
def digit_product (n : ℕ) : ℕ := tens_digit n * ones_digit n

/-- Returns the reversed number of a two-digit number -/
def reverse_number (n : ℕ) : ℕ := 10 * (ones_digit n) + tens_digit n

/-- Checks if a number satisfies the given conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n / digit_product n = 3) ∧ (n % digit_product n = 8) ∧
  (reverse_number n / digit_product n = 2) ∧ (reverse_number n % digit_product n = 5)

theorem unique_solution : ∃! n : ℕ, satisfies_conditions n ∧ n = 53 :=
  sorry


end NUMINAMATH_CALUDE_unique_solution_l2285_228570


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2285_228524

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Returns true if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Returns true if a point (x, y) is on the given line -/
def on_line (l : Line) (x y : ℝ) : Prop := y = l.slope * x + l.y_intercept

theorem y_intercept_of_parallel_line 
  (line1 : Line) 
  (hline1 : line1.slope = -3 ∧ line1.y_intercept = 6) 
  (line2 : Line)
  (hparallel : parallel line1 line2)
  (hon_line : on_line line2 3 1) : 
  line2.y_intercept = 10 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2285_228524


namespace NUMINAMATH_CALUDE_cents_left_over_l2285_228546

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the number of pennies in the jar -/
def num_pennies : ℕ := 123

/-- Represents the number of nickels in the jar -/
def num_nickels : ℕ := 85

/-- Represents the number of dimes in the jar -/
def num_dimes : ℕ := 35

/-- Represents the number of quarters in the jar -/
def num_quarters : ℕ := 26

/-- Represents the cost of a double scoop in dollars -/
def double_scoop_cost : ℕ := 3

/-- Represents the number of family members -/
def num_family_members : ℕ := 5

/-- Theorem stating that the number of cents left over after the trip is 48 -/
theorem cents_left_over : 
  (num_pennies * penny_value + 
   num_nickels * nickel_value + 
   num_dimes * dime_value + 
   num_quarters * quarter_value) - 
  (double_scoop_cost * num_family_members * cents_per_dollar) = 48 := by
  sorry


end NUMINAMATH_CALUDE_cents_left_over_l2285_228546


namespace NUMINAMATH_CALUDE_f_fixed_point_l2285_228584

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the repeated application of f
def repeat_f (p q : ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f p q (repeat_f p q n x)

-- State the theorem
theorem f_fixed_point (p q : ℝ) :
  (∀ x ∈ Set.Icc 2 4, |f p q x| ≤ 1/2) →
  repeat_f p q 2017 ((5 - Real.sqrt 11) / 2) = (5 + Real.sqrt 11) / 2 :=
by sorry

end NUMINAMATH_CALUDE_f_fixed_point_l2285_228584


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2285_228599

/-- 
Given a geometric sequence with positive terms, if the sum of the first n terms is 3
and the sum of the first 3n terms is 21, then the sum of the first 2n terms is 9.
-/
theorem geometric_sequence_sum (n : ℕ) (a : ℝ) (r : ℝ) 
  (h_positive : ∀ k, a * r ^ k > 0)
  (h_sum_n : (a * (1 - r^n)) / (1 - r) = 3)
  (h_sum_3n : (a * (1 - r^(3*n))) / (1 - r) = 21) :
  (a * (1 - r^(2*n))) / (1 - r) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2285_228599


namespace NUMINAMATH_CALUDE_max_planes_from_four_lines_l2285_228586

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of lines -/
def num_lines : ℕ := 4

/-- The number of lines needed to define a plane -/
def lines_per_plane : ℕ := 2

/-- The maximum number of planes that can be defined by four lines starting from the same point -/
def max_planes : ℕ := choose num_lines lines_per_plane

theorem max_planes_from_four_lines : 
  max_planes = 6 := by sorry

end NUMINAMATH_CALUDE_max_planes_from_four_lines_l2285_228586


namespace NUMINAMATH_CALUDE_insert_zeros_is_perfect_cube_l2285_228509

/-- Given a non-negative integer n, the function calculates the number
    obtained by inserting n zeros between each digit of 1331. -/
def insert_zeros (n : ℕ) : ℕ :=
  10^(3*n+3) + 3 * 10^(2*n+2) + 3 * 10^(n+1) + 1

/-- Theorem stating that for any non-negative integer n,
    the number obtained by inserting n zeros between each digit of 1331
    is equal to (10^(n+1) + 1)^3. -/
theorem insert_zeros_is_perfect_cube (n : ℕ) :
  insert_zeros n = (10^(n+1) + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_insert_zeros_is_perfect_cube_l2285_228509


namespace NUMINAMATH_CALUDE_subtraction_theorem_l2285_228557

/-- Represents a four-digit number --/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_digits : thousands < 10 ∧ hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- The result of subtracting two four-digit numbers --/
structure SubtractionResult where
  thousands : Int
  hundreds : Int
  tens : Int
  ones : Int

def subtract (minuend subtrahend : FourDigitNumber) : SubtractionResult :=
  sorry

theorem subtraction_theorem (a b c d : Nat) 
  (h_digits : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) :
  let minuend : FourDigitNumber := ⟨a, b, c, d, h_digits⟩
  let subtrahend : FourDigitNumber := ⟨d, b, a, c, sorry⟩
  let result := subtract minuend subtrahend
  (result.hundreds = 7 ∧ minuend.thousands ≥ subtrahend.thousands) →
  result.thousands = 9 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_theorem_l2285_228557


namespace NUMINAMATH_CALUDE_exponent_relationship_l2285_228566

theorem exponent_relationship (m p r s : ℝ) (u v w t : ℝ) 
  (h1 : m ^ u = p ^ v) (h2 : m ^ u = r) (h3 : p ^ w = m ^ t) (h4 : p ^ w = s) :
  u * v = w * t := by
  sorry

end NUMINAMATH_CALUDE_exponent_relationship_l2285_228566


namespace NUMINAMATH_CALUDE_S_equals_seven_l2285_228518

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) -
  1 / (Real.sqrt 15 - Real.sqrt 14) +
  1 / (Real.sqrt 14 - Real.sqrt 13) -
  1 / (Real.sqrt 13 - Real.sqrt 12) +
  1 / (Real.sqrt 12 - 3)

theorem S_equals_seven : S = 7 := by
  sorry

end NUMINAMATH_CALUDE_S_equals_seven_l2285_228518


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2285_228596

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-14/17, 96/17)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 4

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y = -7 * x - 2

theorem intersection_point_is_unique :
  (∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2285_228596


namespace NUMINAMATH_CALUDE_range_of_a_l2285_228583

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) → 
  -3/5 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2285_228583


namespace NUMINAMATH_CALUDE_discounted_shoe_price_l2285_228558

/-- The price paid for a pair of shoes after a discount -/
theorem discounted_shoe_price (original_price : ℝ) (discount_percent : ℝ) 
  (h1 : original_price = 204)
  (h2 : discount_percent = 75) : 
  original_price * (1 - discount_percent / 100) = 51 := by
  sorry

end NUMINAMATH_CALUDE_discounted_shoe_price_l2285_228558
