import Mathlib

namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l3485_348525

/-- Given an inverse proportion function f(x) = -6/x, prove that y₁ < y₂ 
    where (2, y₁) and (-1, y₂) lie on the graph of f. -/
theorem inverse_proportion_inequality (y₁ y₂ : ℝ) : 
  y₁ = -6/2 → y₂ = -6/(-1) → y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l3485_348525


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l3485_348573

theorem solution_set_implies_m_value (m : ℝ) 
  (h : ∀ x : ℝ, x - m > 5 ↔ x > 2) : 
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l3485_348573


namespace NUMINAMATH_CALUDE_highest_uniquely_identifiable_score_l3485_348595

/-- The AHSME scoring system -/
def score (c w : ℕ) : ℕ := 30 + 4 * c - w

/-- The maximum number of questions in AHSME -/
def max_questions : ℕ := 30

/-- Predicate to check if a score is uniquely identifiable -/
def is_uniquely_identifiable (s : ℕ) : Prop :=
  ∃! (c w : ℕ), c ≤ max_questions ∧ w ≤ max_questions ∧ s = score c w

/-- Theorem stating that 130 is the highest possible uniquely identifiable score over 100 -/
theorem highest_uniquely_identifiable_score :
  (∀ s : ℕ, s > 130 → ¬(is_uniquely_identifiable s)) ∧
  (is_uniquely_identifiable 130) ∧
  (130 > 100) :=
sorry

end NUMINAMATH_CALUDE_highest_uniquely_identifiable_score_l3485_348595


namespace NUMINAMATH_CALUDE_berry_count_l3485_348501

theorem berry_count (total : ℕ) (raspberries blackberries blueberries : ℕ) : 
  total = 42 →
  raspberries = total / 2 →
  blackberries = total / 3 →
  total = raspberries + blackberries + blueberries →
  blueberries = 7 := by
sorry

end NUMINAMATH_CALUDE_berry_count_l3485_348501


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3485_348500

/-- For a quadratic equation ax^2 + bx + c = 0, if one root is three times the other,
    then the coefficients a, b, and c satisfy the relationship 3b^2 = 16ac. -/
theorem quadratic_root_relation (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3485_348500


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l3485_348541

/-- The total cost to copy and bind multiple manuscripts. -/
def total_cost (num_copies : ℕ) (pages_per_copy : ℕ) (copy_cost_per_page : ℚ) (binding_cost_per_copy : ℚ) : ℚ :=
  num_copies * (pages_per_copy * copy_cost_per_page + binding_cost_per_copy)

/-- Theorem stating the total cost for the given manuscript copying and binding scenario. -/
theorem manuscript_cost_theorem :
  total_cost 10 400 (5 / 100) 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_theorem_l3485_348541


namespace NUMINAMATH_CALUDE_set_operations_l3485_348567

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set Nat := {4, 5, 6, 7, 8, 9}
def B : Set Nat := {1, 2, 3, 4, 5, 6}

theorem set_operations :
  (A ∪ B = U) ∧
  (A ∩ B = {4, 5, 6}) ∧
  (U \ (A ∩ B) = {1, 2, 3, 7, 8, 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3485_348567


namespace NUMINAMATH_CALUDE_system_solution_l3485_348574

theorem system_solution (x y : ℝ) : 
  0 < x + y → 
  x + y ≠ 1 → 
  2*x - y ≠ 0 → 
  (x + y) * (2 ^ (y - 2*x)) = 6.25 → 
  (x + y) * (1 / (2*x - y)) = 5 → 
  x = 9 ∧ y = 16 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3485_348574


namespace NUMINAMATH_CALUDE_football_progress_l3485_348581

def round1 : Int := -5
def round2 : Int := 9
def round3 : Int := -12
def round4 : Int := 17
def round5 : Int := -15
def round6 : Int := 24
def round7 : Int := -7

def overall_progress : Int := round1 + round2 + round3 + round4 + round5 + round6 + round7

theorem football_progress : overall_progress = 11 := by
  sorry

end NUMINAMATH_CALUDE_football_progress_l3485_348581


namespace NUMINAMATH_CALUDE_symmetry_implies_periodicity_l3485_348548

/-- A function is symmetric about a line x = a -/
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

/-- A function is symmetric about a point (m, n) -/
def SymmetricAboutPoint (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  ∀ x, 2 * n - f x = f (2 * m - x)

/-- A function is periodic with period p -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem symmetry_implies_periodicity
  (f : ℝ → ℝ) (a m n : ℝ) (ha : a ≠ 0) (hm : m ≠ a)
  (h_line : SymmetricAboutLine f a)
  (h_point : SymmetricAboutPoint f m n) :
  IsPeriodic f (4 * (m - a)) :=
sorry

end NUMINAMATH_CALUDE_symmetry_implies_periodicity_l3485_348548


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l3485_348552

/-- Calculates the sampling interval for systematic sampling -/
def samplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (populationSize - populationSize % sampleSize) / sampleSize

theorem systematic_sampling_interval :
  samplingInterval 1003 50 = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l3485_348552


namespace NUMINAMATH_CALUDE_smaller_cuboid_width_l3485_348511

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

theorem smaller_cuboid_width 
  (large : Cuboid)
  (small_length : ℝ)
  (small_height : ℝ)
  (num_small : ℕ)
  (h1 : large.length = 18)
  (h2 : large.width = 15)
  (h3 : large.height = 2)
  (h4 : small_length = 5)
  (h5 : small_height = 3)
  (h6 : num_small = 18)
  (h7 : volume large = num_small * volume { length := small_length, width := 2, height := small_height }) :
  ∃ (small : Cuboid), small.length = small_length ∧ small.height = small_height ∧ small.width = 2 :=
sorry

end NUMINAMATH_CALUDE_smaller_cuboid_width_l3485_348511


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3485_348555

/-- Represents a hyperbola with semi-major axis a, semi-minor axis b, and focal length c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0
  h_equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  h_focal_length : c = 2 * c
  h_foci_to_asymptotes : b = c / 2

/-- The eccentricity of a hyperbola is 2√3/3 given the conditions -/
theorem hyperbola_eccentricity (C : Hyperbola) : 
  Real.sqrt ((C.c^2) / (C.a^2)) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3485_348555


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3485_348583

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x < 0, f x ≥ 0) ↔ (∀ x < 0, f x < 0) :=
by
  sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x < 0, x^2 - 3*x + 1 ≥ 0) ↔ (∀ x < 0, x^2 - 3*x + 1 < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3485_348583


namespace NUMINAMATH_CALUDE_hall_reunion_attendance_l3485_348591

/-- The number of people attending the Hall reunion -/
def hall_attendees (total_guests oates_attendees both_attendees : ℕ) : ℕ :=
  total_guests - (oates_attendees - both_attendees)

/-- Theorem stating the number of people attending the Hall reunion -/
theorem hall_reunion_attendance 
  (total_guests : ℕ) 
  (oates_attendees : ℕ) 
  (both_attendees : ℕ) 
  (h1 : total_guests = 100) 
  (h2 : oates_attendees = 40) 
  (h3 : both_attendees = 10) 
  (h4 : total_guests ≥ oates_attendees) 
  (h5 : oates_attendees ≥ both_attendees) : 
  hall_attendees total_guests oates_attendees both_attendees = 70 := by
  sorry

end NUMINAMATH_CALUDE_hall_reunion_attendance_l3485_348591


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l3485_348580

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + 2*a - 5 = 0) → (b^2 + 2*b - 5 = 0) → (a^2 + a*b + 2*a = 0) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l3485_348580


namespace NUMINAMATH_CALUDE_A_3_2_l3485_348566

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2 : A 3 2 = 29 := by sorry

end NUMINAMATH_CALUDE_A_3_2_l3485_348566


namespace NUMINAMATH_CALUDE_fault_line_movement_l3485_348571

/-- Fault line movement problem -/
theorem fault_line_movement 
  (total_movement : ℝ) 
  (past_year_movement : ℝ) 
  (h1 : total_movement = 6.5)
  (h2 : past_year_movement = 1.25) :
  total_movement - past_year_movement = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_movement_l3485_348571


namespace NUMINAMATH_CALUDE_square_root_of_product_plus_one_l3485_348556

theorem square_root_of_product_plus_one : 
  Real.sqrt ((34 : ℝ) * 32 * 28 * 26 + 1) = 170 := by sorry

end NUMINAMATH_CALUDE_square_root_of_product_plus_one_l3485_348556


namespace NUMINAMATH_CALUDE_carousel_horse_ratio_l3485_348576

/-- The carousel problem -/
theorem carousel_horse_ratio :
  let blue_horses : ℕ := 3
  let purple_horses : ℕ := 3 * blue_horses
  let green_horses : ℕ := 2 * purple_horses
  let total_horses : ℕ := 33
  let gold_horses : ℕ := total_horses - (blue_horses + purple_horses + green_horses)
  (gold_horses : ℚ) / green_horses = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_carousel_horse_ratio_l3485_348576


namespace NUMINAMATH_CALUDE_triangle_properties_l3485_348512

/-- Triangle with vertices A(4, 0), B(8, 10), and C(0, 6) -/
structure Triangle where
  A : Prod ℝ ℝ := (4, 0)
  B : Prod ℝ ℝ := (8, 10)
  C : Prod ℝ ℝ := (0, 6)

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.altitudeFromAtoBC (t : Triangle) : LineEquation :=
  { a := 2, b := -3, c := 14 }

def Triangle.lineParallelToBCThroughA (t : Triangle) : LineEquation :=
  { a := 1, b := -2, c := -4 }

def Triangle.altitudeFromBtoAC (t : Triangle) : LineEquation :=
  { a := 2, b := 1, c := -8 }

theorem triangle_properties (t : Triangle) : 
  (t.altitudeFromAtoBC = { a := 2, b := -3, c := 14 }) ∧ 
  (t.lineParallelToBCThroughA = { a := 1, b := -2, c := -4 }) ∧
  (t.altitudeFromBtoAC = { a := 2, b := 1, c := -8 }) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3485_348512


namespace NUMINAMATH_CALUDE_stratified_sample_composition_l3485_348586

theorem stratified_sample_composition 
  (total_athletes : ℕ) 
  (male_athletes : ℕ) 
  (female_athletes : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = male_athletes + female_athletes)
  (h2 : total_athletes = 98)
  (h3 : male_athletes = 56)
  (h4 : female_athletes = 42)
  (h5 : sample_size = 14) :
  (male_athletes * sample_size / total_athletes : ℚ) = 8 ∧ 
  (female_athletes * sample_size / total_athletes : ℚ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_composition_l3485_348586


namespace NUMINAMATH_CALUDE_negative_m_exponent_division_l3485_348510

theorem negative_m_exponent_division (m : ℝ) :
  ((-m)^7) / ((-m)^2) = -m^5 := by sorry

end NUMINAMATH_CALUDE_negative_m_exponent_division_l3485_348510


namespace NUMINAMATH_CALUDE_monotonic_quadratic_constraint_l3485_348544

/-- A function f is monotonic on an interval [a, b] if and only if
    its derivative f' does not change sign on (a, b) -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ (Set.Icc a b), (∀ y ∈ (Set.Icc a b), x ≤ y → f x ≤ f y) ∨
                       (∀ y ∈ (Set.Icc a b), x ≤ y → f y ≤ f x)

/-- The quadratic function f(x) = 4x² - kx - 8 -/
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

theorem monotonic_quadratic_constraint (k : ℝ) :
  IsMonotonic (f k) 5 8 ↔ k ∈ Set.Iic 40 ∪ Set.Ici 64 := by
  sorry

#check monotonic_quadratic_constraint

end NUMINAMATH_CALUDE_monotonic_quadratic_constraint_l3485_348544


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l3485_348568

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ n

theorem fifth_term_of_sequence (x : ℝ) :
  let a₁ : ℝ := 4
  let a₂ : ℝ := 12 * x^2
  let a₃ : ℝ := 36 * x^4
  let a₄ : ℝ := 108 * x^6
  let r : ℝ := 3 * x^2
  geometric_sequence a₁ r 4 = 324 * x^8 :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l3485_348568


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l3485_348542

theorem triangle_cosine_theorem (A : ℝ) (a m : ℝ) (θ : ℝ) 
  (h1 : A = 40) 
  (h2 : a = 12) 
  (h3 : m = 10) 
  (h4 : A = 1/2 * a * m * Real.sin θ) 
  (h5 : 0 < θ) 
  (h6 : θ < π/2) : 
  Real.cos θ = Real.sqrt 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l3485_348542


namespace NUMINAMATH_CALUDE_angle_r_measure_l3485_348546

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  /-- The measure of angle P in degrees -/
  angle_p : ℝ
  /-- The measure of angle R is 40 degrees more than angle P -/
  angle_r : ℝ := angle_p + 40
  /-- The sum of all angles in the triangle is 180 degrees -/
  angle_sum : angle_p + angle_p + angle_r = 180

/-- The measure of angle R in an isosceles triangle with the given conditions -/
theorem angle_r_measure (t : IsoscelesTriangle) : t.angle_r = 86.67 := by
  sorry

end NUMINAMATH_CALUDE_angle_r_measure_l3485_348546


namespace NUMINAMATH_CALUDE_linear_function_theorem_l3485_348550

/-- A linear function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem linear_function_theorem (x : ℝ) :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) →  -- f is linear
  (∀ x : ℝ, f x = 3 * (f⁻¹ x) + 5) →        -- f(x) = 3f^(-1)(x) + 5
  f 0 = 3 →                                 -- f(0) = 3
  f 3 = 3 * Real.sqrt 3 + 3 :=               -- f(3) = 3√3 + 3
by sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l3485_348550


namespace NUMINAMATH_CALUDE_min_value_of_f_l3485_348578

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3485_348578


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l3485_348508

/-- Given that 7 oranges weigh the same as 5 apples, prove that 28 oranges weigh the same as 20 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
    orange_weight > 0 →
    apple_weight > 0 →
    7 * orange_weight = 5 * apple_weight →
    28 * orange_weight = 20 * apple_weight :=
by sorry

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l3485_348508


namespace NUMINAMATH_CALUDE_common_chord_equation_l3485_348515

/-- Two circles in a plane -/
structure TwoCircles where
  circle1 : (ℝ × ℝ) → Prop
  circle2 : (ℝ × ℝ) → Prop

/-- The equation of a line in a plane -/
structure Line where
  equation : (ℝ × ℝ) → Prop

/-- The common chord of two intersecting circles -/
def commonChord (circles : TwoCircles) : Line :=
  sorry

/-- Combining equations of two circles -/
def combineEquations (circles : TwoCircles) : (ℝ × ℝ) → Prop :=
  sorry

/-- Eliminating quadratic terms from an equation -/
def eliminateQuadraticTerms (eq : (ℝ × ℝ) → Prop) : (ℝ × ℝ) → Prop :=
  sorry

/-- Theorem: The equation of the common chord of two intersecting circles
    is obtained by eliminating the quadratic terms after combining
    the equations of the two circles -/
theorem common_chord_equation (circles : TwoCircles) :
  (commonChord circles).equation =
  eliminateQuadraticTerms (combineEquations circles) :=
sorry

end NUMINAMATH_CALUDE_common_chord_equation_l3485_348515


namespace NUMINAMATH_CALUDE_circle_equation_is_correct_l3485_348565

/-- A circle with center on the y-axis, radius 1, passing through (1, 3) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_y_axis : center.1 = 0
  radius_is_one : radius = 1
  point_on_circle : (passes_through.1 - center.1)^2 + (passes_through.2 - center.2)^2 = radius^2

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_equation_is_correct (c : Circle) (h : c.passes_through = (1, 3)) :
  ∀ x y : ℝ, circle_equation c x y ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_is_correct_l3485_348565


namespace NUMINAMATH_CALUDE_lindas_family_women_without_daughters_l3485_348584

/-- Represents the family structure of Linda and her descendants -/
structure Family where
  total_daughters_and_granddaughters : Nat
  lindas_daughters : Nat
  daughters_with_five_children : Nat

/-- The number of women (daughters and granddaughters) who have no daughters in Linda's family -/
def women_without_daughters (f : Family) : Nat :=
  f.total_daughters_and_granddaughters - f.daughters_with_five_children

/-- Theorem stating the number of women without daughters in Linda's specific family situation -/
theorem lindas_family_women_without_daughters :
  ∀ f : Family,
  f.total_daughters_and_granddaughters = 43 →
  f.lindas_daughters = 8 →
  f.daughters_with_five_children * 5 = f.total_daughters_and_granddaughters - f.lindas_daughters →
  women_without_daughters f = 36 := by
  sorry


end NUMINAMATH_CALUDE_lindas_family_women_without_daughters_l3485_348584


namespace NUMINAMATH_CALUDE_bryan_books_per_continent_l3485_348560

/-- The number of continents Bryan visited -/
def num_continents : ℕ := 4

/-- The total number of books Bryan collected -/
def total_books : ℕ := 488

/-- The number of books Bryan collected per continent -/
def books_per_continent : ℕ := total_books / num_continents

/-- Theorem stating that Bryan collected 122 books per continent -/
theorem bryan_books_per_continent : books_per_continent = 122 := by
  sorry

end NUMINAMATH_CALUDE_bryan_books_per_continent_l3485_348560


namespace NUMINAMATH_CALUDE_roses_cut_l3485_348534

theorem roses_cut (initial_roses final_roses : ℕ) : 
  initial_roses = 6 → final_roses = 16 → final_roses - initial_roses = 10 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l3485_348534


namespace NUMINAMATH_CALUDE_expression_simplification_l3485_348558

theorem expression_simplification (a : ℝ) (h : a = 2) :
  (a^2 / (a - 1) - a) / ((a + a^2) / (1 - 2*a + a^2)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3485_348558


namespace NUMINAMATH_CALUDE_go_stones_problem_l3485_348507

theorem go_stones_problem (x : ℕ) (h1 : (x / 7 + 40) * 5 = 555) (h2 : x ≥ 55) : x - 55 = 442 := by
  sorry

end NUMINAMATH_CALUDE_go_stones_problem_l3485_348507


namespace NUMINAMATH_CALUDE_average_time_per_flower_l3485_348526

/-- Proves that the average time to find a flower is 10 minutes -/
theorem average_time_per_flower 
  (total_time : ℕ) 
  (total_flowers : ℕ) 
  (h1 : total_time = 330) 
  (h2 : total_flowers = 33) 
  (h3 : total_time % total_flowers = 0) :
  total_time / total_flowers = 10 := by
  sorry

#check average_time_per_flower

end NUMINAMATH_CALUDE_average_time_per_flower_l3485_348526


namespace NUMINAMATH_CALUDE_two_zeros_cubic_l3485_348597

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem two_zeros_cubic (c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ f c x = 0 ∧ f c y = 0 ∧ ∀ z : ℝ, f c z = 0 → z = x ∨ z = y) → 
  c = -2 ∨ c = 2 := by
sorry

end NUMINAMATH_CALUDE_two_zeros_cubic_l3485_348597


namespace NUMINAMATH_CALUDE_max_sum_of_product_3003_l3485_348557

theorem max_sum_of_product_3003 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 3003 →
  A + B + C ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_product_3003_l3485_348557


namespace NUMINAMATH_CALUDE_rectangle_area_preservation_l3485_348518

theorem rectangle_area_preservation (L W : ℝ) (x : ℝ) (h : x > 0) :
  L * W = L * (1 - x / 100) * W * (1 + 11.111111111111107 / 100) →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_preservation_l3485_348518


namespace NUMINAMATH_CALUDE_car_speed_problem_l3485_348520

/-- Proves that given the conditions of the car problem, the average speed of Car X is 50 mph -/
theorem car_speed_problem (Vx : ℝ) : 
  (∃ (T : ℝ), 
    T > 0 ∧ 
    Vx * 1.2 + Vx * T = 50 * T ∧ 
    Vx * T = 98) → 
  Vx = 50 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3485_348520


namespace NUMINAMATH_CALUDE_crushers_win_probability_l3485_348509

theorem crushers_win_probability (n : ℕ) (p : ℚ) (h1 : n = 6) (h2 : p = 4/5) :
  p^n = 4096/15625 := by
  sorry

end NUMINAMATH_CALUDE_crushers_win_probability_l3485_348509


namespace NUMINAMATH_CALUDE_field_trip_students_l3485_348527

theorem field_trip_students (van_capacity : ℕ) (num_vans : ℕ) (num_adults : ℕ) :
  van_capacity = 7 →
  num_vans = 6 →
  num_adults = 9 →
  (van_capacity * num_vans - num_adults : ℕ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l3485_348527


namespace NUMINAMATH_CALUDE_waiter_tables_problem_l3485_348522

theorem waiter_tables_problem (initial_tables : ℝ) : 
  (initial_tables - 12.0) * 8.0 = 256 → initial_tables = 44.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_problem_l3485_348522


namespace NUMINAMATH_CALUDE_shirt_price_is_correct_l3485_348569

/-- The price of a shirt and sweater with given conditions -/
def shirt_price (total_cost sweater_price : ℝ) : ℝ :=
  let shirt_price := sweater_price - 7.43
  let discounted_sweater_price := sweater_price * 0.9
  shirt_price

theorem shirt_price_is_correct (total_cost sweater_price : ℝ) :
  total_cost = 80.34 ∧ 
  shirt_price total_cost sweater_price + sweater_price * 0.9 = total_cost →
  shirt_price total_cost sweater_price = 38.76 :=
by
  sorry

#eval shirt_price 80.34 46.19

end NUMINAMATH_CALUDE_shirt_price_is_correct_l3485_348569


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3485_348594

theorem solution_set_of_inequality (x : ℝ) :
  (3 * x^2 - 7 * x > 6) ↔ (x < -2/3 ∨ x > 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3485_348594


namespace NUMINAMATH_CALUDE_precious_stones_count_l3485_348570

theorem precious_stones_count (N : ℕ) (W : ℝ) : 
  (N > 0) →
  (W > 0) →
  (0.35 * W = 3 * (W / N)) →
  (5/13 * (0.65 * W) = 3 * ((0.65 * W) / (N - 3))) →
  N = 10 := by
sorry

end NUMINAMATH_CALUDE_precious_stones_count_l3485_348570


namespace NUMINAMATH_CALUDE_abs_difference_opposite_signs_l3485_348524

theorem abs_difference_opposite_signs (a b : ℝ) 
  (ha : |a| = 4) 
  (hb : |b| = 2) 
  (hab : a * b < 0) : 
  |a - b| = 6 := by
sorry

end NUMINAMATH_CALUDE_abs_difference_opposite_signs_l3485_348524


namespace NUMINAMATH_CALUDE_unique_solution_system_l3485_348549

theorem unique_solution_system (x y u v : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ u > 0 ∧ v > 0)
  (h1 : x + y = u)
  (h2 : v * x * y = u + v)
  (h3 : x * y * u * v = 16) :
  x = 2 ∧ y = 2 ∧ u = 2 ∧ v = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3485_348549


namespace NUMINAMATH_CALUDE_tony_investment_rate_l3485_348593

/-- Calculates the investment rate given the investment amount and annual income. -/
def investment_rate (investment : ℚ) (annual_income : ℚ) : ℚ :=
  (annual_income / investment) * 100

/-- Proves that the investment rate is 7.8125% for the given scenario. -/
theorem tony_investment_rate :
  let investment := 3200
  let annual_income := 250
  investment_rate investment annual_income = 7.8125 := by
  sorry

end NUMINAMATH_CALUDE_tony_investment_rate_l3485_348593


namespace NUMINAMATH_CALUDE_min_value_c_l3485_348538

-- Define the consecutive integers
def consecutive_integers (a b c d e : ℕ) : Prop :=
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e

-- Define perfect square and perfect cube
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

-- Main theorem
theorem min_value_c (a b c d e : ℕ) :
  consecutive_integers a b c d e →
  is_perfect_square (b + c + d) →
  is_perfect_cube (a + b + c + d + e) →
  c ≥ 675 ∧ (∀ c' : ℕ, c' < 675 → 
    ¬(∃ a' b' d' e' : ℕ, consecutive_integers a' b' c' d' e' ∧
      is_perfect_square (b' + c' + d') ∧
      is_perfect_cube (a' + b' + c' + d' + e'))) :=
by sorry

end NUMINAMATH_CALUDE_min_value_c_l3485_348538


namespace NUMINAMATH_CALUDE_prob_even_odd_is_one_fourth_l3485_348561

/-- Represents a six-sided die -/
def Die := Fin 6

/-- The probability of rolling an even number on a six-sided die -/
def prob_even (d : Die) : ℚ := 1/2

/-- The probability of rolling an odd number on a six-sided die -/
def prob_odd (d : Die) : ℚ := 1/2

/-- The probability of rolling an even number on the first die and an odd number on the second die -/
def prob_even_odd (d1 d2 : Die) : ℚ := prob_even d1 * prob_odd d2

theorem prob_even_odd_is_one_fourth (d1 d2 : Die) :
  prob_even_odd d1 d2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_odd_is_one_fourth_l3485_348561


namespace NUMINAMATH_CALUDE_blackboard_problem_l3485_348531

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The operation of replacing a set of numbers with their sum modulo m -/
def replace_with_sum_mod (m : ℕ) (s : Finset ℕ) : ℕ := (s.sum id) % m

theorem blackboard_problem :
  ∀ (s : Finset ℕ),
  s.card = 2 →
  999 ∈ s →
  (∃ (t : Finset ℕ), t.card = 2004 ∧ Finset.range 2004 = t ∧
   replace_with_sum_mod 167 t = replace_with_sum_mod 167 s) →
  ∃ x, x ∈ s ∧ x ≠ 999 ∧ x = 3 := by
  sorry

#check blackboard_problem

end NUMINAMATH_CALUDE_blackboard_problem_l3485_348531


namespace NUMINAMATH_CALUDE_B_subset_A_l3485_348559

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 + 2*x = 0}

theorem B_subset_A : B ⊆ A := by sorry

end NUMINAMATH_CALUDE_B_subset_A_l3485_348559


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3485_348505

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3485_348505


namespace NUMINAMATH_CALUDE_loan_duration_proof_l3485_348528

/-- Represents the annual interest rate as a decimal -/
def interest_rate (percent : ℚ) : ℚ := percent / 100

/-- Calculates the annual interest given a principal and an interest rate -/
def annual_interest (principal : ℚ) (rate : ℚ) : ℚ := principal * rate

/-- Calculates the gain over a period of time -/
def gain (annual_gain : ℚ) (years : ℚ) : ℚ := annual_gain * years

theorem loan_duration_proof (principal : ℚ) (rate_A_to_B rate_B_to_C total_gain : ℚ) :
  principal = 2000 →
  rate_A_to_B = interest_rate 10 →
  rate_B_to_C = interest_rate 11.5 →
  total_gain = 90 →
  gain (annual_interest principal rate_B_to_C - annual_interest principal rate_A_to_B) 3 = total_gain :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_proof_l3485_348528


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l3485_348563

theorem complex_arithmetic_result : 
  let z₁ : ℂ := 2 - 3*I
  let z₂ : ℂ := -1 + 5*I
  let z₃ : ℂ := 1 + I
  (z₁ + z₂) * z₃ = -1 + 3*I := by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l3485_348563


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l3485_348513

/-- Given a geometric progression with the first three terms, find the fourth term -/
theorem geometric_progression_fourth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 5^(1/3 : ℝ)) 
  (h2 : a * r = 5^(1/5 : ℝ)) 
  (h3 : a * r^2 = 5^(1/15 : ℝ)) : 
  a * r^3 = 5^(-1/15 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l3485_348513


namespace NUMINAMATH_CALUDE_rhombus_area_l3485_348587

/-- The area of a rhombus with side length 25 and one diagonal of 30 is 600 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) :
  side = 25 →
  diagonal1 = 30 →
  diagonal2 = 2 * Real.sqrt (side^2 - (diagonal1 / 2)^2) →
  (diagonal1 * diagonal2) / 2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3485_348587


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l3485_348502

theorem largest_divisor_of_consecutive_odd_product :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 0 ∧
  (∀ k : ℕ, k > m → ¬(k ∣ ((2*n - 1) * (2*n + 1) * (2*n + 3)))) ∧
  (3 ∣ ((2*n - 1) * (2*n + 1) * (2*n + 3))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l3485_348502


namespace NUMINAMATH_CALUDE_three_digit_numbers_property_l3485_348575

theorem three_digit_numbers_property : 
  (∃! (l : List Nat), 
    (∀ n ∈ l, 100 ≤ n ∧ n < 1000) ∧ 
    (∀ n ∈ l, let a := n / 100
              let b := (n / 10) % 10
              let c := n % 10
              10 * a + c = (100 * a + 10 * b + c) / 9) ∧
    l.length = 4) := by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_property_l3485_348575


namespace NUMINAMATH_CALUDE_subtraction_relation_l3485_348592

theorem subtraction_relation (minuend subtrahend difference : ℝ) 
  (h : subtrahend + difference = minuend) : 
  (minuend + subtrahend + difference) / minuend = 2 := by
sorry

end NUMINAMATH_CALUDE_subtraction_relation_l3485_348592


namespace NUMINAMATH_CALUDE_tau_prime_factors_divide_l3485_348590

/-- The number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- The sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- For positive integers a and b, if σ(a^n) divides σ(b^n) for all n ∈ ℕ,
    then each prime factor of τ(a) divides τ(b) -/
theorem tau_prime_factors_divide (a b : ℕ+) 
  (h : ∀ n : ℕ, (sigma (a^n) : ℕ) ∣ (sigma (b^n) : ℕ)) :
  ∀ p : ℕ, Prime p → p ∣ tau a → p ∣ tau b := by
  sorry

end NUMINAMATH_CALUDE_tau_prime_factors_divide_l3485_348590


namespace NUMINAMATH_CALUDE_max_value_theorem_l3485_348579

theorem max_value_theorem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  3 * x * z * Real.sqrt 2 + 5 * x * y ≤ Real.sqrt 43 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3485_348579


namespace NUMINAMATH_CALUDE_range_of_a_l3485_348572

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + a * (y - 2 * Real.exp 1 * x) * (Real.log y - Real.log x) = 0) : 
  a < 0 ∨ a ≥ 2 / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3485_348572


namespace NUMINAMATH_CALUDE_armband_break_even_l3485_348516

/-- The cost of an individual ticket in dollars -/
def individual_ticket_cost : ℚ := 3/4

/-- The cost of an armband in dollars -/
def armband_cost : ℚ := 15

/-- The number of rides at which the armband cost equals the cost of individual tickets -/
def break_even_rides : ℕ := 20

theorem armband_break_even :
  (individual_ticket_cost * break_even_rides : ℚ) = armband_cost :=
by sorry

end NUMINAMATH_CALUDE_armband_break_even_l3485_348516


namespace NUMINAMATH_CALUDE_transformation_result_l3485_348547

-- Define the initial point
def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

-- Define the transformations
def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def rotate_x_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, z, -y)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

-- Define the sequence of transformations
def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  rotate_z_90 (reflect_yz (rotate_x_90 (reflect_xz (rotate_z_90 p))))

-- Theorem statement
theorem transformation_result :
  transform initial_point = (2, -2, -2) := by sorry

end NUMINAMATH_CALUDE_transformation_result_l3485_348547


namespace NUMINAMATH_CALUDE_system_equation_ratio_l3485_348596

theorem system_equation_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (eq1 : 8 * x - 6 * y = c) (eq2 : 12 * y - 18 * x = d) : c / d = -4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_ratio_l3485_348596


namespace NUMINAMATH_CALUDE_complex_modulus_l3485_348577

theorem complex_modulus (z : ℂ) (h : (2 - Complex.I) * z = 4 + 3 * Complex.I) : 
  Complex.abs (z - Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3485_348577


namespace NUMINAMATH_CALUDE_circle_division_theorem_l3485_348554

/-- Represents a region in the circle --/
structure Region where
  value : Nat
  deriving Repr

/-- Represents a line dividing the circle --/
structure DividingLine where
  left_regions : List Region
  right_regions : List Region
  deriving Repr

/-- The configuration of regions in the circle --/
def CircleConfiguration := List Region

/-- Checks if the sums on both sides of a line are equal --/
def is_line_balanced (line : DividingLine) : Bool :=
  (line.left_regions.map Region.value).sum = (line.right_regions.map Region.value).sum

/-- Checks if all lines in the configuration are balanced --/
def is_configuration_valid (config : CircleConfiguration) (lines : List DividingLine) : Bool :=
  lines.all is_line_balanced

/-- Theorem: There exists a valid configuration for distributing numbers 1 to 7 in a circle divided by 3 lines --/
theorem circle_division_theorem :
  ∃ (config : CircleConfiguration) (lines : List DividingLine),
    config.length = 7 ∧
    (∀ n, n ∈ config.map Region.value → n ∈ List.range 7) ∧
    lines.length = 3 ∧
    is_configuration_valid config lines :=
sorry

end NUMINAMATH_CALUDE_circle_division_theorem_l3485_348554


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l3485_348588

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a / Real.exp x

theorem tangent_point_x_coordinate 
  (a : ℝ) 
  (h_even : ∀ x, f a x = f a (-x)) 
  (h_slope : ∃ x, (deriv (f a)) x = 3/2) :
  ∃ x, (deriv (f a)) x = 3/2 ∧ x = Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l3485_348588


namespace NUMINAMATH_CALUDE_saree_stripe_ratio_l3485_348589

theorem saree_stripe_ratio (brown_stripes : ℕ) (blue_stripes : ℕ) (gold_stripes : ℕ) :
  brown_stripes = 4 →
  gold_stripes = 3 * brown_stripes →
  blue_stripes = 60 →
  blue_stripes = gold_stripes →
  blue_stripes / gold_stripes = 5 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_saree_stripe_ratio_l3485_348589


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3485_348553

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  sample_size : ℕ
  first_grade_sample : ℕ
  second_grade_sample : ℕ
  third_grade_sample : ℕ

/-- The stratified sampling theorem -/
theorem stratified_sampling_theorem (s : School) 
  (h1 : s.sample_size = 45)
  (h2 : s.first_grade_sample = 20)
  (h3 : s.third_grade_sample = 10)
  (h4 : s.second_grade = 300)
  (h5 : s.sample_size = s.first_grade_sample + s.second_grade_sample + s.third_grade_sample)
  (h6 : s.total_students = s.first_grade + s.second_grade + s.third_grade)
  (h7 : s.second_grade_sample / s.second_grade = s.sample_size / s.total_students) :
  s.total_students = 900 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3485_348553


namespace NUMINAMATH_CALUDE_ratio_to_percent_l3485_348535

theorem ratio_to_percent (a b : ℕ) (h : a = 6 ∧ b = 3) :
  (a : ℚ) / b * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percent_l3485_348535


namespace NUMINAMATH_CALUDE_book_club_members_count_l3485_348537

def annual_snack_fee : ℕ := 150
def hardcover_books_count : ℕ := 6
def hardcover_book_price : ℕ := 30
def paperback_books_count : ℕ := 6
def paperback_book_price : ℕ := 12
def total_collected : ℕ := 2412

theorem book_club_members_count :
  let cost_per_member := annual_snack_fee +
    hardcover_books_count * hardcover_book_price +
    paperback_books_count * paperback_book_price
  total_collected / cost_per_member = 6 :=
by sorry

end NUMINAMATH_CALUDE_book_club_members_count_l3485_348537


namespace NUMINAMATH_CALUDE_running_time_calculation_l3485_348521

/-- Proves that given the conditions, the time taken to cover the same distance while running is [(a + 2b) × (c + d)] / (3a - b) hours. -/
theorem running_time_calculation 
  (a b c d : ℕ+) -- a, b, c, and d are positive integers
  (walking_speed : ℝ := a + 2*b) -- Walking speed = (a + 2b) kmph
  (walking_time : ℝ := c + d) -- Walking time = (c + d) hours
  (running_speed : ℝ := 3*a - b) -- Running speed = (3a - b) kmph
  (k : ℝ := 3) -- Conversion factor k = 3
  (h : k * walking_speed = running_speed) -- Assumption that k * walking_speed = running_speed
  : 
  (walking_speed * walking_time) / running_speed = (a + 2*b) * (c + d) / (3*a - b) := 
by
  sorry


end NUMINAMATH_CALUDE_running_time_calculation_l3485_348521


namespace NUMINAMATH_CALUDE_permutation_sum_l3485_348540

theorem permutation_sum (n : ℕ) : 
  n + 3 ≤ 2 * n ∧ n + 1 ≤ 4 → 
  (Nat.factorial (2 * n)) / (Nat.factorial (2 * n - (n + 3))) + 
  (Nat.factorial 4) / (Nat.factorial (4 - (n + 1))) = 744 := by
  sorry

end NUMINAMATH_CALUDE_permutation_sum_l3485_348540


namespace NUMINAMATH_CALUDE_average_pastry_sales_l3485_348519

/-- Represents the daily sales of pastries over a week -/
def weeklySales : List Nat := [2, 3, 4, 5, 6, 7, 8]

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- Calculates the average of a list of natural numbers -/
def average (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem average_pastry_sales : average weeklySales = 5 := by sorry

end NUMINAMATH_CALUDE_average_pastry_sales_l3485_348519


namespace NUMINAMATH_CALUDE_paint_area_calculation_l3485_348585

/-- The height of the wall in feet -/
def wall_height : ℝ := 10

/-- The length of the wall in feet -/
def wall_length : ℝ := 15

/-- The height of the door in feet -/
def door_height : ℝ := 3

/-- The width of the door in feet -/
def door_width : ℝ := 5

/-- The area to paint in square feet -/
def area_to_paint : ℝ := wall_height * wall_length - door_height * door_width

theorem paint_area_calculation :
  area_to_paint = 135 := by sorry

end NUMINAMATH_CALUDE_paint_area_calculation_l3485_348585


namespace NUMINAMATH_CALUDE_croissant_resting_time_l3485_348504

theorem croissant_resting_time (fold_count : ℕ) (fold_time : ℕ) (mixing_time : ℕ) (baking_time : ℕ) (total_time : ℕ) :
  fold_count = 4 →
  fold_time = 5 →
  mixing_time = 10 →
  baking_time = 30 →
  total_time = 6 * 60 →
  (total_time - (mixing_time + fold_count * fold_time + baking_time)) / fold_count = 75 := by
  sorry

end NUMINAMATH_CALUDE_croissant_resting_time_l3485_348504


namespace NUMINAMATH_CALUDE_jacks_initial_yen_l3485_348582

/-- Represents Jack's currency holdings and exchange rates -/
structure CurrencyHolding where
  pounds : ℕ
  euros : ℕ
  total_yen : ℕ
  pounds_per_euro : ℕ
  yen_per_pound : ℕ

/-- Calculates the initial yen amount given Jack's currency holding -/
def initial_yen (holding : CurrencyHolding) : ℕ :=
  holding.total_yen - (holding.pounds * holding.yen_per_pound + holding.euros * holding.pounds_per_euro * holding.yen_per_pound)

/-- Theorem stating that Jack's initial yen is 3000 given the problem conditions -/
theorem jacks_initial_yen :
  let jack : CurrencyHolding := {
    pounds := 42,
    euros := 11,
    total_yen := 9400,
    pounds_per_euro := 2,
    yen_per_pound := 100
  }
  initial_yen jack = 3000 := by sorry

end NUMINAMATH_CALUDE_jacks_initial_yen_l3485_348582


namespace NUMINAMATH_CALUDE_long_division_puzzle_l3485_348543

theorem long_division_puzzle :
  (631938 : ℚ) / 625 = 1011.1008 := by
  sorry

end NUMINAMATH_CALUDE_long_division_puzzle_l3485_348543


namespace NUMINAMATH_CALUDE_fantasy_ball_handshakes_l3485_348539

/-- The number of goblins attending the Fantasy Creatures Ball -/
def num_goblins : ℕ := 30

/-- The number of pixies attending the Fantasy Creatures Ball -/
def num_pixies : ℕ := 10

/-- Represents whether pixies can shake hands with a given number of goblins -/
def pixie_can_shake (n : ℕ) : Prop := Even n

/-- Calculates the number of handshakes between goblins -/
def goblin_handshakes (n : ℕ) : ℕ := n.choose 2

/-- Calculates the number of handshakes between goblins and pixies -/
def goblin_pixie_handshakes (g p : ℕ) : ℕ := g * p

/-- The total number of handshakes at the Fantasy Creatures Ball -/
def total_handshakes : ℕ := goblin_handshakes num_goblins + goblin_pixie_handshakes num_goblins num_pixies

theorem fantasy_ball_handshakes :
  pixie_can_shake num_goblins →
  total_handshakes = 735 := by
  sorry

end NUMINAMATH_CALUDE_fantasy_ball_handshakes_l3485_348539


namespace NUMINAMATH_CALUDE_power_of_product_l3485_348523

theorem power_of_product (a : ℝ) : (-4 * a^3)^2 = 16 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3485_348523


namespace NUMINAMATH_CALUDE_seller_took_weight_l3485_348514

/-- Given 10 weights with masses n, n+1, ..., n+9, if the sum of 9 of these weights is 1457,
    then the missing weight is 158. -/
theorem seller_took_weight (n : ℕ) (x : ℕ) (h1 : x ≤ 9) 
    (h2 : (10 * n + 45) - (n + x) = 1457) : n + x = 158 := by
  sorry

end NUMINAMATH_CALUDE_seller_took_weight_l3485_348514


namespace NUMINAMATH_CALUDE_square_root_of_2m_minus_n_is_2_l3485_348503

theorem square_root_of_2m_minus_n_is_2 
  (m n : ℝ) 
  (eq1 : m * 2 + n * 1 = 8) 
  (eq2 : n * 2 - m * 1 = 1) : 
  Real.sqrt (2 * m - n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_2m_minus_n_is_2_l3485_348503


namespace NUMINAMATH_CALUDE_largest_A_value_l3485_348530

theorem largest_A_value : ∃ (A : ℝ),
  (∀ (x y : ℝ), x * y = 1 →
    ((x + y)^2 + 4) * ((x + y)^2 - 2) ≥ A * (x - y)^2) ∧
  (∀ (B : ℝ), (∀ (x y : ℝ), x * y = 1 →
    ((x + y)^2 + 4) * ((x + y)^2 - 2) ≥ B * (x - y)^2) → B ≤ A) ∧
  A = 18 :=
by sorry

end NUMINAMATH_CALUDE_largest_A_value_l3485_348530


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3485_348562

/-- Given a geometric sequence with first term a₁ and common ratio r,
    a_n represents the nth term of the sequence. -/
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem fifth_term_of_geometric_sequence :
  let a₁ : ℝ := 5
  let r : ℝ := -2
  geometric_sequence a₁ r 5 = 80 := by sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3485_348562


namespace NUMINAMATH_CALUDE_sum_of_composite_function_l3485_348506

def p (x : ℝ) : ℝ := |x| - 3

def q (x : ℝ) : ℝ := -x^2

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_composite_function :
  (x_values.map (fun x => q (p x))).sum = -29 := by sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_l3485_348506


namespace NUMINAMATH_CALUDE_graduating_class_male_percentage_l3485_348551

theorem graduating_class_male_percentage :
  ∀ (M F : ℝ),
  M + F = 100 →
  0.5 * M + 0.7 * F = 62 →
  M = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_graduating_class_male_percentage_l3485_348551


namespace NUMINAMATH_CALUDE_monday_rainfall_calculation_l3485_348545

/-- The rainfall on Monday in inches -/
def monday_rainfall : ℝ := sorry

/-- The rainfall on Tuesday in inches -/
def tuesday_rainfall : ℝ := 0.2

/-- The difference in rainfall between Monday and Tuesday in inches -/
def rainfall_difference : ℝ := 0.7

theorem monday_rainfall_calculation : monday_rainfall = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_monday_rainfall_calculation_l3485_348545


namespace NUMINAMATH_CALUDE_equation_pattern_find_a_b_l3485_348533

theorem equation_pattern (n : ℕ) (hn : n ≥ 2) :
  Real.sqrt (n + n / (n^2 - 1)) = n * Real.sqrt (n / (n^2 - 1)) := by sorry

theorem find_a_b (a b : ℝ) (h : Real.sqrt (6 + a / b) = 6 * Real.sqrt (a / b)) :
  a = 6 ∧ b = 35 := by sorry

end NUMINAMATH_CALUDE_equation_pattern_find_a_b_l3485_348533


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3485_348529

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, (-x^2 + c*x - 9 < -4) ↔ (x < 2 ∨ x > 7)) → c = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3485_348529


namespace NUMINAMATH_CALUDE_infinitely_many_with_1989_ones_l3485_348536

/-- Count the number of ones in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The theorem stating that there are infinitely many positive integers
    with 1989 ones in their binary representation -/
theorem infinitely_many_with_1989_ones :
  ∀ k : ℕ, ∃ m : ℕ, m > k ∧ countOnes m = 1989 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_with_1989_ones_l3485_348536


namespace NUMINAMATH_CALUDE_expected_value_is_three_l3485_348598

/-- Represents the outcome of rolling a six-sided dice -/
inductive DiceOutcome
  | Two
  | Five
  | Other

/-- The probability of each dice outcome -/
def probability (outcome : DiceOutcome) : ℚ :=
  match outcome with
  | DiceOutcome.Two => 1/4
  | DiceOutcome.Five => 1/2
  | DiceOutcome.Other => 1/12

/-- The payoff for each dice outcome in dollars -/
def payoff (outcome : DiceOutcome) : ℚ :=
  match outcome with
  | DiceOutcome.Two => 4
  | DiceOutcome.Five => 6
  | DiceOutcome.Other => -3

/-- The expected value of rolling the dice once -/
def expectedValue : ℚ :=
  (probability DiceOutcome.Two * payoff DiceOutcome.Two) +
  (probability DiceOutcome.Five * payoff DiceOutcome.Five) +
  (4 * probability DiceOutcome.Other * payoff DiceOutcome.Other)

theorem expected_value_is_three :
  expectedValue = 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_three_l3485_348598


namespace NUMINAMATH_CALUDE_min_value_theorem_l3485_348564

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^3 / (y - 2) + y^3 / (x - 2)) ≥ 54 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 2 ∧ y₀ > 2 ∧ x₀^3 / (y₀ - 2) + y₀^3 / (x₀ - 2) = 54 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3485_348564


namespace NUMINAMATH_CALUDE_road_repaving_l3485_348599

/-- Given that a construction company repaved 4133 inches of road before today
    and 805 inches today, prove that the total length of road repaved is 4938 inches. -/
theorem road_repaving (inches_before : ℕ) (inches_today : ℕ) 
  (h1 : inches_before = 4133) (h2 : inches_today = 805) :
  inches_before + inches_today = 4938 := by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l3485_348599


namespace NUMINAMATH_CALUDE_reciprocal_equality_l3485_348517

theorem reciprocal_equality (a b : ℝ) 
  (ha : a⁻¹ = -8) 
  (hb : (-b)⁻¹ = 8) : 
  a = b := by sorry

end NUMINAMATH_CALUDE_reciprocal_equality_l3485_348517


namespace NUMINAMATH_CALUDE_nonCubeSequence_250th_term_l3485_348532

/-- Function that determines if a positive integer is a perfect cube --/
def isPerfectCube (n : ℕ+) : Prop :=
  ∃ m : ℕ+, n = m^3

/-- The sequence of positive integers omitting perfect cubes --/
def nonCubeSequence : ℕ+ → ℕ+ :=
  sorry

/-- The 250th term of the sequence is 256 --/
theorem nonCubeSequence_250th_term :
  nonCubeSequence 250 = 256 := by
  sorry

end NUMINAMATH_CALUDE_nonCubeSequence_250th_term_l3485_348532
