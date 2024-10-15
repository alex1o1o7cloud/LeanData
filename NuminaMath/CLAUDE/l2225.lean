import Mathlib

namespace NUMINAMATH_CALUDE_parts_cost_calculation_l2225_222536

theorem parts_cost_calculation (total_amount : ℝ) (total_parts : ℕ) 
  (expensive_parts : ℕ) (expensive_cost : ℝ) :
  total_amount = 2380 →
  total_parts = 59 →
  expensive_parts = 40 →
  expensive_cost = 50 →
  ∃ (other_cost : ℝ),
    other_cost = 20 ∧
    total_amount = expensive_parts * expensive_cost + (total_parts - expensive_parts) * other_cost :=
by sorry

end NUMINAMATH_CALUDE_parts_cost_calculation_l2225_222536


namespace NUMINAMATH_CALUDE_combined_travel_time_l2225_222546

/-- 
Given a car that takes 4.5 hours to reach station B, and a train that takes 2 hours longer 
than the car to travel the same distance, the combined time for both to reach station B is 11 hours.
-/
theorem combined_travel_time (car_time train_time : ℝ) : 
  car_time = 4.5 → 
  train_time = car_time + 2 → 
  car_time + train_time = 11 := by
sorry

end NUMINAMATH_CALUDE_combined_travel_time_l2225_222546


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l2225_222530

-- Part 1
theorem simplify_expression (x y : ℝ) :
  (3 * x^2 - 2 * x * y + 5 * y^2) - 2 * (x^2 - x * y - 2 * y^2) = x^2 + 9 * y^2 := by
  sorry

-- Part 2
theorem evaluate_expression (x y : ℝ) (A B : ℝ) 
  (h1 : A = -x - 2*y - 1)
  (h2 : B = x + 2*y + 2)
  (h3 : x + 2*y = 6) :
  A + 3*B = 17 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l2225_222530


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l2225_222524

theorem imaginary_part_of_complex_product (i : ℂ) :
  i * i = -1 →
  (Complex.im ((2 - 3 * i) * i) = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l2225_222524


namespace NUMINAMATH_CALUDE_parallelogram_properties_l2225_222562

-- Define a parallelogram
structure Parallelogram where
  is_quadrilateral : Bool
  opposite_sides_parallel : Bool

-- Define the properties
def has_equal_sides (p : Parallelogram) : Prop := sorry
def is_square (p : Parallelogram) : Prop := sorry

-- Theorem statement
theorem parallelogram_properties (p : Parallelogram) :
  (p.is_quadrilateral ∧ p.opposite_sides_parallel) →
  (∃ p1 : Parallelogram, has_equal_sides p1 ∧ ¬is_square p1) ∧
  (∀ p2 : Parallelogram, is_square p2 → has_equal_sides p2) ∧
  (∃ p3 : Parallelogram, has_equal_sides p3 ∧ is_square p3) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_properties_l2225_222562


namespace NUMINAMATH_CALUDE_quadratic_root_sum_inequality_l2225_222594

theorem quadratic_root_sum_inequality (a b c x₁ : ℝ) (h₁ : x₁ > 0) (h₂ : a * x₁^2 + b * x₁ + c = 0) :
  ∃ x₂ : ℝ, x₂ > 0 ∧ c * x₂^2 + b * x₂ + a = 0 ∧ x₁ + x₂ ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_inequality_l2225_222594


namespace NUMINAMATH_CALUDE_inclination_angle_of_line_l2225_222590

/-- The inclination angle of a line with equation y = x - 3 is 45 degrees. -/
theorem inclination_angle_of_line (x y : ℝ) :
  y = x - 3 → Real.arctan 1 = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_inclination_angle_of_line_l2225_222590


namespace NUMINAMATH_CALUDE_quadratic_transform_h_value_l2225_222595

/-- Given a quadratic equation ax^2 + bx + c that can be expressed as 3(x - 5)^2 + 15,
    prove that when 4ax^2 + 4bx + 4c is expressed as n(x - h)^2 + k, h equals 5. -/
theorem quadratic_transform_h_value
  (a b c : ℝ)
  (h : ∀ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 15) :
  ∃ (n k : ℝ), ∀ x, 4 * a * x^2 + 4 * b * x + 4 * c = n * (x - 5)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transform_h_value_l2225_222595


namespace NUMINAMATH_CALUDE_expression_equality_l2225_222519

theorem expression_equality : -1^4 + |1 - Real.sqrt 2| - (Real.pi - 3.14)^0 = Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2225_222519


namespace NUMINAMATH_CALUDE_transformation_result_l2225_222551

def rotate_180_degrees (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

def reflect_y_equals_x (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.2, point.1)

theorem transformation_result (Q : ℝ × ℝ) :
  let rotated := rotate_180_degrees (2, 3) Q
  let reflected := reflect_y_equals_x rotated
  reflected = (4, -1) → Q.1 - Q.2 = 3 := by
sorry

end NUMINAMATH_CALUDE_transformation_result_l2225_222551


namespace NUMINAMATH_CALUDE_least_months_to_triple_l2225_222549

def interest_rate : ℝ := 1.06

def amount_owed (t : ℕ) : ℝ := interest_rate ^ t

def exceeds_triple (t : ℕ) : Prop := amount_owed t > 3

theorem least_months_to_triple : 
  (∀ n < 20, ¬(exceeds_triple n)) ∧ (exceeds_triple 20) :=
sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l2225_222549


namespace NUMINAMATH_CALUDE_gcd_45_75_l2225_222575

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_75_l2225_222575


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2225_222568

theorem sphere_surface_area (a b c : ℝ) (h1 : a * b * c = Real.sqrt 6) 
  (h2 : a * b = Real.sqrt 2) (h3 : b * c = Real.sqrt 3) : 
  4 * Real.pi * ((a^2 + b^2 + c^2).sqrt / 2)^2 = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2225_222568


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2225_222543

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let a : ℝ := -32
  let b : ℝ := 84
  let c : ℝ := 135
  let eq := a * x^2 + b * x + c = 0
  let sum_of_roots := -b / a
  sum_of_roots = 21 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2225_222543


namespace NUMINAMATH_CALUDE_load_capacity_calculation_l2225_222535

theorem load_capacity_calculation (T H : ℝ) (L : ℝ) : 
  T = 3 → H = 9 → L = (35 * T^3) / H^3 → L = 35 / 27 := by
  sorry

end NUMINAMATH_CALUDE_load_capacity_calculation_l2225_222535


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_eighth_l2225_222576

theorem tan_alpha_minus_pi_eighth (α : Real) 
  (h : 2 * Real.sin α = Real.sin (α - π/4)) : 
  Real.tan (α - π/8) = 3 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_eighth_l2225_222576


namespace NUMINAMATH_CALUDE_function_properties_l2225_222544

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + 2*a*x

def g (a b : ℝ) (x : ℝ) : ℝ := 3*a^2 * Real.log x + b

-- State the theorem
theorem function_properties (a b : ℝ) :
  (a > 0) →
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    f a x₀ = g a b x₀ ∧ 
    (deriv (f a)) x₀ = (deriv (g a b)) x₀ ∧
    a = Real.exp 1) →
  (b = -(Real.exp 1)^2 / 2) ∧
  (∀ x > 0, f a x ≥ g a b x - b) →
  (0 < a ∧ a ≤ Real.exp ((5:ℝ)/6)) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2225_222544


namespace NUMINAMATH_CALUDE_transform_f_to_g_l2225_222597

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := (x + 1)^2 + 3

/-- The function after transformation -/
def g (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Right shift transformation -/
def shift_right (h : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := fun x ↦ h (x - a)

/-- Down shift transformation -/
def shift_down (h : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := fun x ↦ h x - b

/-- Theorem stating that the transformation of f results in g -/
theorem transform_f_to_g : shift_down (shift_right f 2) 1 = g := by
  sorry

end NUMINAMATH_CALUDE_transform_f_to_g_l2225_222597


namespace NUMINAMATH_CALUDE_root_difference_squared_l2225_222507

theorem root_difference_squared (f g : ℝ) : 
  (6 * f^2 + 13 * f - 28 = 0) → 
  (6 * g^2 + 13 * g - 28 = 0) → 
  (f - g)^2 = 169 / 9 := by
sorry

end NUMINAMATH_CALUDE_root_difference_squared_l2225_222507


namespace NUMINAMATH_CALUDE_probability_all_visible_faces_same_color_l2225_222533

/-- Represents the three possible colors for painting the cube faces -/
inductive Color
| Red
| Blue
| Green

/-- A cube with 6 faces, each painted with a color -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- The probability of a specific color arrangement on the cube -/
def colorArrangementProbability : ℚ := (1 : ℚ) / 729

/-- Predicate to check if a cube can be placed with all visible vertical faces the same color -/
def hasAllVisibleFacesSameColor (c : Cube) : Prop := sorry

/-- The number of color arrangements where all visible vertical faces can be the same color -/
def numValidArrangements : ℕ := 57

/-- Theorem stating the probability of a cube having all visible vertical faces the same color -/
theorem probability_all_visible_faces_same_color :
  (numValidArrangements : ℚ) * colorArrangementProbability = 57 / 729 := by sorry

end NUMINAMATH_CALUDE_probability_all_visible_faces_same_color_l2225_222533


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2225_222556

theorem quadratic_roots_relation (q : ℝ) : 
  let eq1 := fun x : ℂ => x^2 + 2*x + q
  let eq2 := fun x : ℂ => (1+q)*(x^2 + 2*x + q) - 2*(q-1)*(x^2 + 1)
  (∃ x y : ℝ, x ≠ y ∧ eq1 x = 0 ∧ eq1 y = 0) ↔ 
  (∀ z : ℂ, eq2 z = 0 → z.im ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2225_222556


namespace NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_2_mod_17_l2225_222565

theorem largest_four_digit_negative_congruent_to_2_mod_17 :
  ∀ x : ℤ, -9999 ≤ x ∧ x < -999 ∧ x ≡ 2 [ZMOD 17] → x ≤ -1001 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_2_mod_17_l2225_222565


namespace NUMINAMATH_CALUDE_percentage_to_pass_l2225_222584

/-- Given a test with maximum marks and a student's performance, 
    calculate the percentage needed to pass the test. -/
theorem percentage_to_pass (max_marks : ℕ) (student_marks : ℕ) (fail_margin : ℕ) :
  max_marks = 300 →
  student_marks = 80 →
  fail_margin = 10 →
  (((student_marks + fail_margin : ℝ) / max_marks) * 100 : ℝ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l2225_222584


namespace NUMINAMATH_CALUDE_eight_player_tournament_l2225_222539

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 8 players, the total number of matches is 28. -/
theorem eight_player_tournament : num_matches 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_eight_player_tournament_l2225_222539


namespace NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l2225_222587

/-- Sum of the first n terms of a geometric sequence -/
def geometric_sum (a₀ r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of a geometric sequence
    with first term 1/3 and common ratio 1/3 is 6560/19683 -/
theorem geometric_sum_first_8_terms :
  geometric_sum (1/3) (1/3) 8 = 6560/19683 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l2225_222587


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2225_222577

theorem inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 1 2 = {x | x^2 - a*x + b < 0}) :
  {x : ℝ | 1/x < b/a} = Set.union (Set.Iio 0) (Set.Ioi (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2225_222577


namespace NUMINAMATH_CALUDE_carol_fraction_l2225_222578

/-- Represents the money each person has -/
structure Money where
  alice : ℚ
  bob : ℚ
  carol : ℚ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.carol = 0 ∧ 
  m.alice > 0 ∧ 
  m.bob > 0 ∧ 
  m.alice / 6 = m.bob / 3 ∧
  m.alice / 6 > 0

/-- The final state after Alice and Bob give money to Carol -/
def final_state (m : Money) : Money :=
  { alice := m.alice * (5/6),
    bob := m.bob * (2/3),
    carol := m.alice / 6 + m.bob / 3 }

/-- The theorem to be proved -/
theorem carol_fraction (m : Money) 
  (h : problem_conditions m) : 
  (final_state m).carol / ((final_state m).alice + (final_state m).bob + (final_state m).carol) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_carol_fraction_l2225_222578


namespace NUMINAMATH_CALUDE_tank_cart_friction_l2225_222593

/-- The frictional force acting on a tank resting on an accelerating cart --/
theorem tank_cart_friction (m₁ m₂ a μ g : ℝ) (h₁ : m₁ = 3) (h₂ : m₂ = 15) (h₃ : a = 4) (h₄ : μ = 0.6) (h₅ : g = 9.8) :
  let F_friction := m₁ * a
  let F_max_static := μ * m₁ * g
  F_friction ≤ F_max_static ∧ F_friction = 12 := by
  sorry

end NUMINAMATH_CALUDE_tank_cart_friction_l2225_222593


namespace NUMINAMATH_CALUDE_pentagon_sum_l2225_222591

/-- Definition of a pentagon -/
structure Pentagon where
  sides : ℕ
  vertices : ℕ
  is_pentagon : sides = 5 ∧ vertices = 5

/-- Theorem: The sum of sides and vertices of a pentagon is 10 -/
theorem pentagon_sum (p : Pentagon) : p.sides + p.vertices = 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_sum_l2225_222591


namespace NUMINAMATH_CALUDE_equation_has_two_real_roots_l2225_222517

theorem equation_has_two_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (x₁ - Real.sqrt (2 * x₁ + 6) = 2) ∧
  (x₂ - Real.sqrt (2 * x₂ + 6) = 2) ∧
  (∀ x : ℝ, x - Real.sqrt (2 * x + 6) = 2 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_has_two_real_roots_l2225_222517


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l2225_222573

/-- A ball rolling on a half circular track and bouncing on the floor -/
theorem ball_bounce_distance 
  (R : ℝ) -- radius of the half circular track
  (v : ℝ) -- velocity of the ball when leaving the track
  (g : ℝ) -- acceleration due to gravity
  (h : R > 0) -- radius is positive
  (hv : v > 0) -- velocity is positive
  (hg : g > 0) -- gravity is positive
  : ∃ (d : ℝ), d = 2 * R - (2 * v / 3) * Real.sqrt (R / g) :=
by sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l2225_222573


namespace NUMINAMATH_CALUDE_harmonic_sum_divisibility_l2225_222527

theorem harmonic_sum_divisibility (p : ℕ) (m n : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2) 
  (h_sum : (m : ℚ) / n = (Finset.range (p - 1)).sum (λ i => 1 / (i + 1 : ℚ))) :
  p ∣ m := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_divisibility_l2225_222527


namespace NUMINAMATH_CALUDE_animal_count_l2225_222550

theorem animal_count (dogs cats frogs : ℕ) : 
  cats = (80 * dogs) / 100 →
  frogs = 2 * dogs →
  frogs = 160 →
  dogs + cats + frogs = 304 := by
sorry

end NUMINAMATH_CALUDE_animal_count_l2225_222550


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2225_222545

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 2 * x₁ - 15 = 0) →
  (3 * x₂^2 - 2 * x₂ - 15 = 0) →
  (x₁^2 + x₂^2 = 94/9) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2225_222545


namespace NUMINAMATH_CALUDE_library_books_theorem_l2225_222515

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being a new arrival
variable (is_new_arrival : Book → Prop)

-- Define the theorem
theorem library_books_theorem (h : ¬ (∀ b : Book, is_new_arrival b)) :
  (∃ b : Book, ¬ is_new_arrival b) ∧ (¬ ∀ b : Book, is_new_arrival b) := by
  sorry

end NUMINAMATH_CALUDE_library_books_theorem_l2225_222515


namespace NUMINAMATH_CALUDE_set_operations_l2225_222582

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {1, 2, 5}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {2, 5}) ∧ (A ∪ (U \ B) = {2, 3, 4, 5, 6}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l2225_222582


namespace NUMINAMATH_CALUDE_soccer_team_average_age_l2225_222585

def ages : List ℕ := [13, 14, 15, 16, 17, 18]
def players : List ℕ := [2, 6, 8, 3, 2, 1]

theorem soccer_team_average_age :
  (List.sum (List.zipWith (· * ·) ages players)) / (List.sum players) = 15 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_average_age_l2225_222585


namespace NUMINAMATH_CALUDE_cross_in_square_l2225_222518

/-- Given a square with side length s containing a cross made up of two squares
    with side length s/2 and two squares with side length s/4, if the total area
    of the cross is 810 cm², then s = 36 cm. -/
theorem cross_in_square (s : ℝ) :
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → s = 36 := by sorry

end NUMINAMATH_CALUDE_cross_in_square_l2225_222518


namespace NUMINAMATH_CALUDE_circle_land_diagram_value_l2225_222501

/-- Represents a digit with circles in Circle Land -/
structure CircleDigit where
  digit : Nat
  circles : Nat

/-- Calculates the value of a CircleDigit -/
def circleValue (cd : CircleDigit) : Nat :=
  cd.digit * (10 ^ cd.circles)

/-- Represents a number in Circle Land -/
def CircleLandNumber (cds : List CircleDigit) : Nat :=
  cds.map circleValue |>.sum

/-- The specific diagram given in the problem -/
def problemDiagram : List CircleDigit :=
  [⟨3, 4⟩, ⟨1, 2⟩, ⟨5, 0⟩]

theorem circle_land_diagram_value :
  CircleLandNumber problemDiagram = 30105 := by
  sorry

end NUMINAMATH_CALUDE_circle_land_diagram_value_l2225_222501


namespace NUMINAMATH_CALUDE_ten_thousand_scientific_notation_l2225_222553

/-- Scientific notation representation of 10,000 -/
def scientific_notation_10000 : ℝ := 1 * (10 ^ 4)

/-- Theorem stating that 10,000 is equal to its scientific notation representation -/
theorem ten_thousand_scientific_notation : 
  (10000 : ℝ) = scientific_notation_10000 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_scientific_notation_l2225_222553


namespace NUMINAMATH_CALUDE_unruly_max_sum_squares_l2225_222523

/-- A quadratic polynomial q(x) with real coefficients a and b -/
def q (a b x : ℝ) : ℝ := x^2 - (a+b)*x + a*b - 1

/-- The condition for q to be unruly -/
def is_unruly (a b : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    ∀ w, q a b (q a b w) = 0 ↔ w = x ∨ w = y ∨ w = z

/-- The sum of squares of roots of q(x) -/
def sum_of_squares (a b : ℝ) : ℝ := (a+b)^2 + 2*(a*b - 1)

/-- Theorem stating that the unruly polynomial maximizing the sum of squares of its roots satisfies q(1) = -3 -/
theorem unruly_max_sum_squares :
  ∃ (a b : ℝ), is_unruly a b ∧
    (∀ (c d : ℝ), is_unruly c d → sum_of_squares c d ≤ sum_of_squares a b) ∧
    q a b 1 = -3 :=
sorry

end NUMINAMATH_CALUDE_unruly_max_sum_squares_l2225_222523


namespace NUMINAMATH_CALUDE_dragon_legs_count_l2225_222574

/-- Represents the number of legs per centipede -/
def centipede_legs : ℕ := 40

/-- Represents the number of heads per dragon -/
def dragon_heads : ℕ := 9

/-- Represents the total number of heads in the cage -/
def total_heads : ℕ := 50

/-- Represents the total number of legs in the cage -/
def total_legs : ℕ := 220

/-- Represents the number of centipedes in the cage -/
def num_centipedes : ℕ := 40

/-- Represents the number of dragons in the cage -/
def num_dragons : ℕ := total_heads - num_centipedes

/-- Theorem stating that each dragon has 4 legs -/
theorem dragon_legs_count : 
  ∃ (dragon_legs : ℕ), 
    dragon_legs = 4 ∧ 
    num_centipedes * centipede_legs + num_dragons * dragon_legs = total_legs :=
sorry

end NUMINAMATH_CALUDE_dragon_legs_count_l2225_222574


namespace NUMINAMATH_CALUDE_quadratic_equations_properties_l2225_222564

theorem quadratic_equations_properties (b c : ℤ) 
  (x₁ x₂ x₁' x₂' : ℤ) :
  (x₁^2 + b*x₁ + c = 0) →
  (x₂^2 + b*x₂ + c = 0) →
  (x₁'^2 + c*x₁' + b = 0) →
  (x₂'^2 + c*x₂' + b = 0) →
  (x₁ * x₂ > 0) →
  (x₁' * x₂' > 0) →
  (
    (x₁ < 0 ∧ x₂ < 0 ∧ x₁' < 0 ∧ x₂' < 0) ∧
    (b - 1 ≤ c ∧ c ≤ b + 1) ∧
    ((b = 4 ∧ c = 4) ∨ (b = 5 ∧ c = 6) ∨ (b = 6 ∧ c = 5))
  ) := by sorry

end NUMINAMATH_CALUDE_quadratic_equations_properties_l2225_222564


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_angle_specific_line_equation_l2225_222506

/-- The equation of a line passing through a given point with a given inclination angle -/
theorem line_equation_through_point_with_angle (x₀ y₀ : ℝ) (θ : ℝ) :
  (x₀ = Real.sqrt 3) →
  (y₀ = -2 * Real.sqrt 3) →
  (θ = 135 * π / 180) →
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧
                 ∀ (x y : ℝ), a * x + b * y + c = 0 ↔
                               y - y₀ = Real.tan θ * (x - x₀) :=
by sorry

/-- The specific equation of the line in the problem -/
theorem specific_line_equation :
  ∃ (x y : ℝ), x + y + Real.sqrt 3 = 0 ↔
                y - (-2 * Real.sqrt 3) = Real.tan (135 * π / 180) * (x - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_angle_specific_line_equation_l2225_222506


namespace NUMINAMATH_CALUDE_college_students_fraction_l2225_222532

theorem college_students_fraction (total : ℕ) (h_total : total > 0) :
  let third_year := (total : ℚ) / 2
  let not_second_year := (total : ℚ) * 7 / 10
  let second_year := total - not_second_year
  let not_third_year := total - third_year
  second_year / not_third_year = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_college_students_fraction_l2225_222532


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2225_222560

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2225_222560


namespace NUMINAMATH_CALUDE_pencil_count_l2225_222547

theorem pencil_count (rows : ℕ) (pencils_per_row : ℕ) (h1 : rows = 2) (h2 : pencils_per_row = 3) :
  rows * pencils_per_row = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2225_222547


namespace NUMINAMATH_CALUDE_train_length_l2225_222566

theorem train_length (speed : ℝ) (train_length : ℝ) : 
  (train_length + 130) / 15 = speed ∧ 
  (train_length + 250) / 20 = speed → 
  train_length = 230 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l2225_222566


namespace NUMINAMATH_CALUDE_triangle_special_angle_l2225_222505

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a^2 + b^2 - √2ab = c^2, then the measure of angle C is π/4 -/
theorem triangle_special_angle (a b c : ℝ) (h : a^2 + b^2 - Real.sqrt 2 * a * b = c^2) :
  let angle_C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  angle_C = π / 4 := by
sorry


end NUMINAMATH_CALUDE_triangle_special_angle_l2225_222505


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l2225_222554

theorem quadratic_equation_condition (m : ℝ) : 
  (∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, (m - 3) * x^2 + m * x + (-2 * m - 2) = a * x^2 + b * x + c) ↔ 
  m = -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l2225_222554


namespace NUMINAMATH_CALUDE_sequence_growth_l2225_222542

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, i ≥ 1 → Nat.gcd (a i) (a (i + 1)) > a (i - 1)

theorem sequence_growth (a : ℕ → ℕ) (h : sequence_property a) :
  ∀ n : ℕ, a n ≥ 2^n :=
by sorry

end NUMINAMATH_CALUDE_sequence_growth_l2225_222542


namespace NUMINAMATH_CALUDE_julieta_total_spent_l2225_222579

-- Define the original prices and price changes
def original_backpack_price : ℕ := 50
def original_binder_price : ℕ := 20
def backpack_price_increase : ℕ := 5
def binder_price_reduction : ℕ := 2
def number_of_binders : ℕ := 3

-- Define the theorem
theorem julieta_total_spent :
  let new_backpack_price := original_backpack_price + backpack_price_increase
  let new_binder_price := original_binder_price - binder_price_reduction
  let total_spent := new_backpack_price + number_of_binders * new_binder_price
  total_spent = 109 := by sorry

end NUMINAMATH_CALUDE_julieta_total_spent_l2225_222579


namespace NUMINAMATH_CALUDE_complex_expression_equality_logarithmic_expression_equality_l2225_222516

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

theorem complex_expression_equality : 
  (1) * (2^(1/3) * 3^(1/2))^6 + (2 * 2^(1/2))^(4/3) - 4 * (16/49)^(-1/2) - 2^(1/4) * 8^0.25 - (-2009)^0 = 100 :=
sorry

theorem logarithmic_expression_equality :
  2 * (lg (2^(1/2)))^2 + lg (2^(1/2)) + lg 5 + ((lg (2^(1/2)))^2 - lg 2 + 1)^(1/2) = 1 :=
sorry

end NUMINAMATH_CALUDE_complex_expression_equality_logarithmic_expression_equality_l2225_222516


namespace NUMINAMATH_CALUDE_x_values_l2225_222500

def U : Set ℕ := Set.univ

def A (x : ℕ) : Set ℕ := {1, 4, x}

def B (x : ℕ) : Set ℕ := {1, x^2}

theorem x_values (x : ℕ) : (Set.compl (A x) ⊂ Set.compl (B x)) → (x = 0 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_x_values_l2225_222500


namespace NUMINAMATH_CALUDE_sin_arccos_three_fifths_l2225_222589

theorem sin_arccos_three_fifths : Real.sin (Real.arccos (3/5)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_three_fifths_l2225_222589


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2225_222583

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) →  -- a_n is a geometric sequence
  a 7 * a 11 = 6 →                           -- a_7 * a_11 = 6
  a 4 + a 14 = 5 →                           -- a_4 + a_14 = 5
  (a 20 / a 10 = 3/2 ∨ a 20 / a 10 = 2/3) :=  -- a_20 / a_10 is either 3/2 or 2/3
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2225_222583


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2225_222596

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y + 20) :
  x + y = 12 + 2 * Real.sqrt 6 ∨ x + y = 12 - 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2225_222596


namespace NUMINAMATH_CALUDE_base5_divisible_by_31_l2225_222504

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (a b c d : ℕ) : ℕ := a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

/-- Checks if a number is divisible by 31 --/
def isDivisibleBy31 (n : ℕ) : Prop := ∃ k : ℕ, n = 31 * k

/-- The main theorem --/
theorem base5_divisible_by_31 (x : ℕ) : 
  x < 5 → (isDivisibleBy31 (base5ToBase10 3 4 x 1) ↔ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_base5_divisible_by_31_l2225_222504


namespace NUMINAMATH_CALUDE_association_membership_l2225_222569

theorem association_membership (M : ℕ) : 
  (525 : ℕ) ≤ M ∧ 
  (315 : ℕ) = (525 * 60 : ℕ) / 100 ∧ 
  (315 : ℝ) = (M : ℝ) * 19.6875 / 100 →
  M = 1600 := by
sorry

end NUMINAMATH_CALUDE_association_membership_l2225_222569


namespace NUMINAMATH_CALUDE_three_digit_numbers_count_l2225_222567

-- Define the set of numbers
def S : Finset ℕ := {1, 2, 3, 4}

-- Define the number of elements to choose
def r : ℕ := 3

-- Theorem statement
theorem three_digit_numbers_count : Nat.descFactorial (Finset.card S) r = 24 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_count_l2225_222567


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2225_222537

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2225_222537


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l2225_222511

theorem imaginary_power_sum : Complex.I ^ 21 + Complex.I ^ 103 + Complex.I ^ 50 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l2225_222511


namespace NUMINAMATH_CALUDE_souvenir_shop_properties_l2225_222541

/-- Represents the cost and profit structure of souvenirs --/
structure SouvenirShop where
  costA : ℕ → ℕ  -- Cost function for type A
  costB : ℕ → ℕ  -- Cost function for type B
  profitA : ℕ    -- Profit per piece of type A
  profitB : ℕ    -- Profit per piece of type B

/-- Theorem stating the properties of the souvenir shop problem --/
theorem souvenir_shop_properties (shop : SouvenirShop) :
  (shop.costA 7 + shop.costB 4 = 760) ∧
  (shop.costA 5 + shop.costB 8 = 800) ∧
  (shop.profitA = 30) ∧
  (shop.profitB = 20) →
  (∃ (x y : ℕ), 
    (∀ n : ℕ, shop.costA n = n * x) ∧
    (∀ n : ℕ, shop.costB n = n * y) ∧
    x = 80 ∧ 
    y = 50) ∧
  (∃ (plans : List ℕ),
    plans.length = 7 ∧
    (∀ a ∈ plans, 
      80 * a + 50 * (100 - a) ≥ 7000 ∧
      80 * a + 50 * (100 - a) ≤ 7200)) ∧
  (∃ (maxA : ℕ) (maxB : ℕ),
    maxA + maxB = 100 ∧
    maxA = 73 ∧
    maxB = 27 ∧
    ∀ a b : ℕ, 
      a + b = 100 →
      shop.profitA * a + shop.profitB * b ≤ shop.profitA * maxA + shop.profitB * maxB) :=
by sorry


end NUMINAMATH_CALUDE_souvenir_shop_properties_l2225_222541


namespace NUMINAMATH_CALUDE_contest_order_l2225_222558

/-- Represents the scores of contestants in a mathematics competition. -/
structure ContestScores where
  adam : ℝ
  bob : ℝ
  charles : ℝ
  david : ℝ
  nonnegative : adam ≥ 0 ∧ bob ≥ 0 ∧ charles ≥ 0 ∧ david ≥ 0
  sum_equality : adam + bob = charles + david
  interchange_inequality : charles + adam > bob + david
  charles_exceeds_sum : charles > adam + bob

/-- Proves that given the contest conditions, the order of scores from highest to lowest is Charles, Adam, Bob, David. -/
theorem contest_order (scores : ContestScores) : 
  scores.charles > scores.adam ∧ 
  scores.adam > scores.bob ∧ 
  scores.bob > scores.david := by
  sorry


end NUMINAMATH_CALUDE_contest_order_l2225_222558


namespace NUMINAMATH_CALUDE_walking_time_calculation_l2225_222555

/-- Proves that given a distance that takes 40 minutes to cover at a speed of 16.5 kmph,
    it will take 165 minutes to cover the same distance at a speed of 4 kmph. -/
theorem walking_time_calculation (distance : ℝ) : 
  distance = 16.5 * (40 / 60) → distance / 4 * 60 = 165 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_calculation_l2225_222555


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l2225_222508

theorem cubic_sum_over_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 18)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 21 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l2225_222508


namespace NUMINAMATH_CALUDE_cube_expansion_coefficient_sum_l2225_222561

theorem cube_expansion_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 3 * x - 1)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_cube_expansion_coefficient_sum_l2225_222561


namespace NUMINAMATH_CALUDE_valid_speaking_orders_eq_600_l2225_222502

/-- The number of students in the class --/
def total_students : ℕ := 7

/-- The number of students to be selected for speaking --/
def selected_speakers : ℕ := 4

/-- The number of special students (A and B) --/
def special_students : ℕ := 2

/-- Function to calculate the number of valid speaking orders --/
def valid_speaking_orders : ℕ :=
  let one_special := special_students * (total_students - special_students).choose (selected_speakers - 1) * (selected_speakers).factorial
  let both_special := special_students.choose 2 * (total_students - special_students).choose (selected_speakers - 2) * (selected_speakers).factorial
  let adjacent := special_students.choose 2 * (total_students - special_students).choose (selected_speakers - 2) * (selected_speakers - 1).factorial * 2
  one_special + both_special - adjacent

/-- Theorem stating that the number of valid speaking orders is 600 --/
theorem valid_speaking_orders_eq_600 : valid_speaking_orders = 600 := by
  sorry

end NUMINAMATH_CALUDE_valid_speaking_orders_eq_600_l2225_222502


namespace NUMINAMATH_CALUDE_quadratic_shift_theorem_l2225_222557

/-- Represents a quadratic function of the form y = a(x-h)² + k -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a quadratic function vertically -/
def verticalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h, k := f.k + shift }

/-- Shifts a quadratic function horizontally -/
def horizontalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h - shift, k := f.k }

/-- The theorem stating that shifting y = 5(x-1)² + 1 down by 3 and left by 2 results in y = 5(x+1)² - 2 -/
theorem quadratic_shift_theorem :
  let f : QuadraticFunction := { a := 5, h := 1, k := 1 }
  let g := horizontalShift (verticalShift f (-3)) 2
  g = { a := 5, h := -1, k := -2 } := by sorry

end NUMINAMATH_CALUDE_quadratic_shift_theorem_l2225_222557


namespace NUMINAMATH_CALUDE_volunteers_distribution_count_l2225_222548

def number_of_volunteers : ℕ := 5
def number_of_schools : ℕ := 3

/-- The number of ways to distribute volunteers to schools -/
def distribute_volunteers : ℕ := sorry

theorem volunteers_distribution_count :
  distribute_volunteers = 150 := by sorry

end NUMINAMATH_CALUDE_volunteers_distribution_count_l2225_222548


namespace NUMINAMATH_CALUDE_simplest_common_denominator_l2225_222559

-- Define the fractions
def fraction1 (x y : ℚ) : ℚ := 1 / (2 * x^2 * y)
def fraction2 (x y : ℚ) : ℚ := 1 / (6 * x * y^3)

-- Define the common denominator
def common_denominator (x y : ℚ) : ℚ := 6 * x^2 * y^3

-- Theorem statement
theorem simplest_common_denominator (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (a b : ℚ), 
    fraction1 x y = a / common_denominator x y ∧
    fraction2 x y = b / common_denominator x y ∧
    (∀ (c : ℚ), c > 0 → 
      (∃ (d e : ℚ), fraction1 x y = d / c ∧ fraction2 x y = e / c) →
      c ≥ common_denominator x y) :=
sorry

end NUMINAMATH_CALUDE_simplest_common_denominator_l2225_222559


namespace NUMINAMATH_CALUDE_paige_to_remainder_ratio_l2225_222581

/-- Represents the number of pieces in a chocolate bar -/
def total_pieces : ℕ := 60

/-- Represents the number of pieces Michael takes -/
def michael_pieces : ℕ := total_pieces / 2

/-- Represents the number of pieces Mandy gets -/
def mandy_pieces : ℕ := 15

/-- Represents the number of pieces Paige takes -/
def paige_pieces : ℕ := total_pieces - michael_pieces - mandy_pieces

/-- Theorem stating the ratio of Paige's pieces to pieces left after Michael's share -/
theorem paige_to_remainder_ratio :
  (paige_pieces : ℚ) / (total_pieces - michael_pieces : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_paige_to_remainder_ratio_l2225_222581


namespace NUMINAMATH_CALUDE_power_product_equality_l2225_222571

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 11 = 39600 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2225_222571


namespace NUMINAMATH_CALUDE_sum_of_bases_l2225_222531

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The value of digit C in base 13 -/
def C : Nat := 12

theorem sum_of_bases :
  to_base_10 [7, 5, 3] 9 + to_base_10 [2, C, 4] 13 = 1129 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_bases_l2225_222531


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2225_222586

theorem trigonometric_identity : 
  2 * Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2225_222586


namespace NUMINAMATH_CALUDE_inequality_range_l2225_222520

theorem inequality_range (m : ℝ) :
  (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ m ∈ Set.Iio (-4) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2225_222520


namespace NUMINAMATH_CALUDE_completing_square_result_l2225_222528

theorem completing_square_result (x : ℝ) : 
  x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l2225_222528


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2225_222572

theorem unique_positive_solution (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  x^3 + 2*y^2 + 1/(4*z) = 1 →
  y^3 + 2*z^2 + 1/(4*x) = 1 →
  z^3 + 2*x^2 + 1/(4*y) = 1 →
  x = (-1 + Real.sqrt 3) / 2 ∧
  y = (-1 + Real.sqrt 3) / 2 ∧
  z = (-1 + Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2225_222572


namespace NUMINAMATH_CALUDE_five_digit_automorphic_number_l2225_222588

theorem five_digit_automorphic_number :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n^2 % 100000 = n :=
sorry

end NUMINAMATH_CALUDE_five_digit_automorphic_number_l2225_222588


namespace NUMINAMATH_CALUDE_locus_is_straight_line_l2225_222525

-- Define the fixed point A
def A : ℝ × ℝ := (1, 1)

-- Define the line l
def l (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the locus of points equidistant from A and l
def locus (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x - A.1)^2 + (y - A.2)^2 = (x + y - 2)^2 / 2

-- Theorem statement
theorem locus_is_straight_line :
  ∃ (a b c : ℝ), ∀ (x y : ℝ), locus (x, y) ↔ a*x + b*y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_is_straight_line_l2225_222525


namespace NUMINAMATH_CALUDE_choose_four_captains_from_twelve_l2225_222503

theorem choose_four_captains_from_twelve (n : ℕ) (k : ℕ) : n = 12 ∧ k = 4 → Nat.choose n k = 990 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_captains_from_twelve_l2225_222503


namespace NUMINAMATH_CALUDE_fuel_cost_savings_l2225_222521

theorem fuel_cost_savings 
  (old_efficiency : ℝ) 
  (old_fuel_cost : ℝ) 
  (trip_distance : ℝ) 
  (efficiency_improvement : ℝ) 
  (fuel_cost_increase : ℝ) 
  (h1 : old_efficiency > 0)
  (h2 : old_fuel_cost > 0)
  (h3 : trip_distance = 1000)
  (h4 : efficiency_improvement = 0.6)
  (h5 : fuel_cost_increase = 0.25) :
  let new_efficiency := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost := old_fuel_cost * (1 + fuel_cost_increase)
  let old_trip_cost := (trip_distance / old_efficiency) * old_fuel_cost
  let new_trip_cost := (trip_distance / new_efficiency) * new_fuel_cost
  let savings_percentage := (old_trip_cost - new_trip_cost) / old_trip_cost * 100
  savings_percentage = 21.875 := by
sorry

end NUMINAMATH_CALUDE_fuel_cost_savings_l2225_222521


namespace NUMINAMATH_CALUDE_least_divisible_by_second_primes_l2225_222526

/-- The second set of four consecutive prime numbers -/
def second_consecutive_primes : Finset Nat := {11, 13, 17, 19}

/-- The product of the second set of four consecutive prime numbers -/
def product_of_primes : Nat := 46219

/-- Theorem stating that the product of the second set of four consecutive primes
    is the least positive whole number divisible by all of them -/
theorem least_divisible_by_second_primes :
  (∀ p ∈ second_consecutive_primes, product_of_primes % p = 0) ∧
  (∀ n : Nat, 0 < n ∧ n < product_of_primes →
    ∃ p ∈ second_consecutive_primes, n % p ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_least_divisible_by_second_primes_l2225_222526


namespace NUMINAMATH_CALUDE_lottery_winning_probability_l2225_222512

theorem lottery_winning_probability :
  let megaball_count : ℕ := 30
  let winnerball_count : ℕ := 50
  let chosen_winnerball_count : ℕ := 6

  let megaball_prob : ℚ := 1 / megaball_count
  let winnerball_prob : ℚ := 1 / (winnerball_count.choose chosen_winnerball_count)

  megaball_prob * winnerball_prob = 1 / 477621000 := by
  sorry

end NUMINAMATH_CALUDE_lottery_winning_probability_l2225_222512


namespace NUMINAMATH_CALUDE_luke_stickers_l2225_222513

theorem luke_stickers (initial bought gift given_away used remaining : ℕ) : 
  bought = 12 →
  gift = 20 →
  given_away = 5 →
  used = 8 →
  remaining = 39 →
  initial + bought + gift - given_away - used = remaining →
  initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_luke_stickers_l2225_222513


namespace NUMINAMATH_CALUDE_girls_entered_classroom_l2225_222580

theorem girls_entered_classroom (initial_boys initial_girls boys_left final_total : ℕ) :
  initial_boys = 5 →
  initial_girls = 4 →
  boys_left = 3 →
  final_total = 8 →
  ∃ girls_entered : ℕ, girls_entered = 2 ∧
    final_total = (initial_boys - boys_left) + (initial_girls + girls_entered) :=
by sorry

end NUMINAMATH_CALUDE_girls_entered_classroom_l2225_222580


namespace NUMINAMATH_CALUDE_division_problem_l2225_222510

theorem division_problem (dividend quotient remainder : ℕ) (h1 : dividend = 2944) (h2 : quotient = 40) (h3 : remainder = 64) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 72 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l2225_222510


namespace NUMINAMATH_CALUDE_carlas_apples_l2225_222529

/-- The number of apples Carla put in her backpack in the morning. -/
def initial_apples : ℕ := sorry

/-- The number of apples stolen by Buffy. -/
def stolen_apples : ℕ := 45

/-- The number of apples that fell out of the backpack. -/
def fallen_apples : ℕ := 26

/-- The number of apples remaining at lunchtime. -/
def remaining_apples : ℕ := 8

theorem carlas_apples : initial_apples = stolen_apples + fallen_apples + remaining_apples := by
  sorry

end NUMINAMATH_CALUDE_carlas_apples_l2225_222529


namespace NUMINAMATH_CALUDE_channel_probabilities_l2225_222592

/-- Represents a binary communication channel with error probabilities -/
structure Channel where
  α : Real
  β : Real
  h_α_pos : 0 < α
  h_α_lt_one : α < 1
  h_β_pos : 0 < β
  h_β_lt_one : β < 1

/-- Probability of receiving 1,0,1 when sending 1,0,1 in single transmission -/
def singleTransProb (c : Channel) : Real :=
  (1 - c.α) * (1 - c.β)^2

/-- Probability of receiving 1,0,1 when sending 1 in triple transmission -/
def tripleTransProb (c : Channel) : Real :=
  c.β * (1 - c.β)^2

/-- Probability of decoding as 1 when sending 1 in triple transmission -/
def tripleTransDecodeProb (c : Channel) : Real :=
  c.β * (1 - c.β)^2 + (1 - c.β)^3

/-- Probability of decoding as 0 when sending 0 in single transmission -/
def singleTransDecodeZeroProb (c : Channel) : Real :=
  1 - c.α

/-- Probability of decoding as 0 when sending 0 in triple transmission -/
def tripleTransDecodeZeroProb (c : Channel) : Real :=
  3 * c.α * (1 - c.α)^2 + (1 - c.α)^3

theorem channel_probabilities (c : Channel) :
  (singleTransProb c = (1 - c.α) * (1 - c.β)^2) ∧
  (tripleTransProb c = c.β * (1 - c.β)^2) ∧
  (tripleTransDecodeProb c = c.β * (1 - c.β)^2 + (1 - c.β)^3) ∧
  (∀ h : 0 < c.α ∧ c.α < 0.5,
    tripleTransDecodeZeroProb c > singleTransDecodeZeroProb c) :=
by sorry

end NUMINAMATH_CALUDE_channel_probabilities_l2225_222592


namespace NUMINAMATH_CALUDE_max_xy_value_l2225_222538

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 18) :
  ∀ z w : ℝ, z > 0 → w > 0 → z + w = 18 → x * y ≥ z * w ∧ x * y ≤ 81 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l2225_222538


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l2225_222552

-- Define the displacement function
def s (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 6 * t^2

-- Theorem statement
theorem instantaneous_velocity_at_2 :
  v 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l2225_222552


namespace NUMINAMATH_CALUDE_total_players_l2225_222570

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) : 
  kabadi = 10 → kho_kho_only = 25 → both = 5 → 
  kabadi + kho_kho_only - both = 30 := by
sorry

end NUMINAMATH_CALUDE_total_players_l2225_222570


namespace NUMINAMATH_CALUDE_praveen_age_multiplier_l2225_222599

def present_age : ℕ := 20

def age_3_years_back : ℕ := present_age - 3

def age_after_10_years : ℕ := present_age + 10

theorem praveen_age_multiplier :
  (age_after_10_years : ℚ) / age_3_years_back = 30 / 17 := by sorry

end NUMINAMATH_CALUDE_praveen_age_multiplier_l2225_222599


namespace NUMINAMATH_CALUDE_x_squared_greater_than_x_l2225_222514

theorem x_squared_greater_than_x (x : ℝ) :
  (x > 1 → x^2 > x) ∧ ¬(x^2 > x → x > 1) := by sorry

end NUMINAMATH_CALUDE_x_squared_greater_than_x_l2225_222514


namespace NUMINAMATH_CALUDE_units_digit_of_special_two_digit_number_l2225_222534

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def units_digit (n : ℕ) : ℕ := n % 10

def digit_product (n : ℕ) : ℕ := tens_digit n * units_digit n

def digit_sum (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem units_digit_of_special_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ N = digit_product N * digit_sum N ∧ units_digit N = 8 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_special_two_digit_number_l2225_222534


namespace NUMINAMATH_CALUDE_cylinder_height_difference_l2225_222509

theorem cylinder_height_difference (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_difference_l2225_222509


namespace NUMINAMATH_CALUDE_addition_verification_l2225_222598

theorem addition_verification (a b s : ℝ) (h : s = a + b) : 
  (s - a = b) ∧ (s - b = a) := by
sorry

end NUMINAMATH_CALUDE_addition_verification_l2225_222598


namespace NUMINAMATH_CALUDE_cos_2x_minus_pi_4_graph_translation_l2225_222522

open Real

theorem cos_2x_minus_pi_4_graph_translation (x : ℝ) : 
  cos (2*x - π/4) = sin (2*(x + π/8)) := by sorry

end NUMINAMATH_CALUDE_cos_2x_minus_pi_4_graph_translation_l2225_222522


namespace NUMINAMATH_CALUDE_sqrt_19_minus_1_between_3_and_4_l2225_222540

theorem sqrt_19_minus_1_between_3_and_4 :
  let a := Real.sqrt 19 - 1
  3 < a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_19_minus_1_between_3_and_4_l2225_222540


namespace NUMINAMATH_CALUDE_solution_set_f_geq_zero_range_m_three_zero_points_l2225_222563

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + 2| - |x - 2| + m

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := f m x - x

-- Theorem 1: Solution set of f(x) ≥ 0 when m = 1
theorem solution_set_f_geq_zero (x : ℝ) :
  f 1 x ≥ 0 ↔ x ≥ -1/2 :=
sorry

-- Theorem 2: Range of m when g(x) has three zero points
theorem range_m_three_zero_points :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g m x = 0 ∧ g m y = 0 ∧ g m z = 0) →
  -2 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_zero_range_m_three_zero_points_l2225_222563
