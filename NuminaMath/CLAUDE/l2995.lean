import Mathlib

namespace NUMINAMATH_CALUDE_orthogonal_vectors_l2995_299557

/-- Two vectors in ℝ³ -/
def v1 : Fin 3 → ℝ := ![3, -1, 4]
def v2 (x : ℝ) : Fin 3 → ℝ := ![x, 4, -2]

/-- Dot product of two vectors in ℝ³ -/
def dot_product (u v : Fin 3 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1) + (u 2) * (v 2)

/-- The theorem stating that x = 4 makes v1 and v2 orthogonal -/
theorem orthogonal_vectors :
  ∃ x : ℝ, dot_product v1 (v2 x) = 0 ∧ x = 4 := by
  sorry

#check orthogonal_vectors

end NUMINAMATH_CALUDE_orthogonal_vectors_l2995_299557


namespace NUMINAMATH_CALUDE_union_cardinality_lower_bound_equality_holds_l2995_299545

theorem union_cardinality_lower_bound 
  (A B C : Finset ℕ) 
  (h : A ∩ B ∩ C = ∅) : 
  (A ∪ B ∪ C).card ≥ (A.card + B.card + C.card) / 2 := by
  sorry

def equality_example : Finset ℕ × Finset ℕ × Finset ℕ :=
  ({1, 2}, {2, 3}, {3, 1})

theorem equality_holds (A B C : Finset ℕ) 
  (h : (A, B, C) = equality_example) :
  (A ∪ B ∪ C).card = (A.card + B.card + C.card) / 2 := by
  sorry

end NUMINAMATH_CALUDE_union_cardinality_lower_bound_equality_holds_l2995_299545


namespace NUMINAMATH_CALUDE_f_properties_l2995_299527

noncomputable def f (x : ℝ) : ℝ := 2^(Real.sin x) + 2^(-Real.sin x)

theorem f_properties :
  -- f is an even function
  (∀ x, f (-x) = f x) ∧
  -- π is a period of f
  (∀ x, f (x + Real.pi) = f x) ∧
  -- π is a local minimum of f
  (∃ ε > 0, ∀ x, x ∈ Set.Ioo (Real.pi - ε) (Real.pi + ε) → f Real.pi ≤ f x) ∧
  -- f is strictly increasing on (0, π/2)
  (∀ x y, x ∈ Set.Ioo 0 (Real.pi / 2) → y ∈ Set.Ioo 0 (Real.pi / 2) → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2995_299527


namespace NUMINAMATH_CALUDE_angle_A_measure_l2995_299555

-- Define the triangle and its properties
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)

-- Define the configuration
def geometric_configuration (t : Triangle) (x y : ℝ) : Prop :=
  t.B = 120 ∧ 
  x = 50 ∧
  y = 130 ∧
  x + (180 - y) + t.C = 180

-- Theorem statement
theorem angle_A_measure (t : Triangle) (x y : ℝ) 
  (h : geometric_configuration t x y) : t.A = 120 :=
sorry

end NUMINAMATH_CALUDE_angle_A_measure_l2995_299555


namespace NUMINAMATH_CALUDE_vector_sum_l2995_299502

theorem vector_sum : 
  let v1 : Fin 3 → ℝ := ![4, -8, 10]
  let v2 : Fin 3 → ℝ := ![-7, 12, -15]
  v1 + v2 = ![-3, 4, -5] := by sorry

end NUMINAMATH_CALUDE_vector_sum_l2995_299502


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2995_299558

theorem complex_equation_solution :
  ∃ (x y : ℝ), (-5 + 2 * Complex.I) * x - (3 - 4 * Complex.I) * y = 2 - Complex.I ∧
  x = -5/14 ∧ y = -1/14 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2995_299558


namespace NUMINAMATH_CALUDE_set_operations_l2995_299566

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x^2 - 4*x ≤ 0}

-- Define the theorem
theorem set_operations :
  (A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 4}) ∧
  (A ∩ (Bᶜ) = {x : ℝ | -1 ≤ x ∧ x < 0}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2995_299566


namespace NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l2995_299511

theorem sum_of_two_squares_equivalence (n : ℤ) : 
  (∃ (a b : ℤ), n = a^2 + b^2) ↔ (∃ (u v : ℤ), 2*n = u^2 + v^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l2995_299511


namespace NUMINAMATH_CALUDE_circle_area_l2995_299519

theorem circle_area (r : ℝ) (h : 6 / (2 * Real.pi * r) = 2 * r) : 
  Real.pi * r^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l2995_299519


namespace NUMINAMATH_CALUDE_matrix_cube_l2995_299500

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_l2995_299500


namespace NUMINAMATH_CALUDE_sponge_city_philosophy_l2995_299585

/-- Represents a sponge city -/
structure SpongeCity where
  resilience : Bool
  waterManagement : Bool
  pilotProject : Bool

/-- Philosophical perspectives on sponge cities -/
inductive PhilosophicalPerspective
  | overall_function_greater
  | integrated_thinking
  | new_connections
  | internal_structure_optimization

/-- Checks if a given philosophical perspective applies to sponge cities -/
def applies_to_sponge_cities (sc : SpongeCity) (pp : PhilosophicalPerspective) : Prop :=
  match pp with
  | PhilosophicalPerspective.overall_function_greater => true
  | PhilosophicalPerspective.integrated_thinking => true
  | _ => false

/-- Theorem: Sponge cities reflect specific philosophical perspectives -/
theorem sponge_city_philosophy (sc : SpongeCity) 
  (h1 : sc.resilience = true) 
  (h2 : sc.waterManagement = true) 
  (h3 : sc.pilotProject = true) :
  (applies_to_sponge_cities sc PhilosophicalPerspective.overall_function_greater) ∧
  (applies_to_sponge_cities sc PhilosophicalPerspective.integrated_thinking) :=
by
  sorry

end NUMINAMATH_CALUDE_sponge_city_philosophy_l2995_299585


namespace NUMINAMATH_CALUDE_expression_value_l2995_299594

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : abs m = 2)  -- |m| = 2
  : (3 * c * d) / (4 * m) + m^2 - 5 * (a + b) = 35/8 ∨ 
    (3 * c * d) / (4 * m) + m^2 - 5 * (a + b) = 29/8 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2995_299594


namespace NUMINAMATH_CALUDE_incorrect_yeast_experiment_method_l2995_299506

/-- Represents an experiment exploring dynamic changes of yeast cell numbers --/
structure YeastExperiment where
  /-- Whether the experiment requires repeated trials --/
  requires_repeated_trials : Bool
  /-- Whether the experiment needs a control group --/
  needs_control_group : Bool

/-- Theorem stating that the incorrect method for yeast cell number experiments 
    is the one claiming no need for repeated trials or control group --/
theorem incorrect_yeast_experiment_method :
  ∀ (e : YeastExperiment), 
    (e.requires_repeated_trials = true) → 
    ¬(e.requires_repeated_trials = false ∧ e.needs_control_group = false) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_yeast_experiment_method_l2995_299506


namespace NUMINAMATH_CALUDE_product_of_square_roots_l2995_299564

theorem product_of_square_roots (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 126 * q * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l2995_299564


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2995_299508

theorem quadratic_one_solution_sum (a : ℝ) : 
  (∃! x, 3 * x^2 + a * x + 6 * x + 7 = 0) ↔ 
  (a = -6 + 2 * Real.sqrt 21 ∨ a = -6 - 2 * Real.sqrt 21) ∧
  (-6 + 2 * Real.sqrt 21) + (-6 - 2 * Real.sqrt 21) = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2995_299508


namespace NUMINAMATH_CALUDE_student_average_mark_l2995_299520

/-- Given a student's marks in 5 subjects, prove that the average mark in 4 subjects
    (excluding physics) is 70, when the total marks are 280 more than the physics marks. -/
theorem student_average_mark (physics chemistry maths biology english : ℕ) :
  physics + chemistry + maths + biology + english = physics + 280 →
  (chemistry + maths + biology + english) / 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_student_average_mark_l2995_299520


namespace NUMINAMATH_CALUDE_f_composition_equals_negative_262144_l2995_299582

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if ¬(z.re = 0 ∧ z.im = 0) then z^2
  else if 0 < z.re then -z^2
  else z^3

-- State the theorem
theorem f_composition_equals_negative_262144 :
  f (f (f (f (1 + I)))) = -262144 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_negative_262144_l2995_299582


namespace NUMINAMATH_CALUDE_shaded_area_is_14_l2995_299598

/-- Represents the grid dimensions --/
structure GridDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle --/
def rectangleArea (w h : ℕ) : ℕ := w * h

/-- Calculates the area of a right-angled triangle --/
def triangleArea (base height : ℕ) : ℕ := base * height / 2

/-- Theorem stating that the shaded area in the grid is 14 square units --/
theorem shaded_area_is_14 (grid : GridDimensions) 
    (h1 : grid.width = 12)
    (h2 : grid.height = 4) : 
  rectangleArea grid.width grid.height - triangleArea grid.width grid.height = 14 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_14_l2995_299598


namespace NUMINAMATH_CALUDE_f_definition_f_of_five_l2995_299507

noncomputable def f : ℝ → ℝ := λ u => (u^3 + 6*u^2 + 21*u + 40) / 27

theorem f_definition (x : ℝ) : f (3*x - 1) = x^3 + x^2 + x + 1 := by sorry

theorem f_of_five : f 5 = 140 / 9 := by sorry

end NUMINAMATH_CALUDE_f_definition_f_of_five_l2995_299507


namespace NUMINAMATH_CALUDE_plot_length_is_56_l2995_299550

/-- Proves that the length of a rectangular plot is 56 meters given the specified conditions -/
theorem plot_length_is_56 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 12 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.5 →
  total_cost = 5300 →
  total_cost = cost_per_meter * perimeter →
  length = 56 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_56_l2995_299550


namespace NUMINAMATH_CALUDE_ceiling_2023_ceiling_quadratic_inequality_ceiling_equality_distance_l2995_299515

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- Theorem 1
theorem ceiling_2023 (x : ℝ) :
  ceiling x = 2023 → x ∈ Set.Ioo 2022 2023 := by sorry

-- Theorem 2
theorem ceiling_quadratic_inequality (x : ℝ) :
  (ceiling x)^2 - 5*(ceiling x) + 6 ≤ 0 → x ∈ Set.Ioo 1 3 := by sorry

-- Theorem 3
theorem ceiling_equality_distance (x y : ℝ) :
  ceiling x = ceiling y → |x - y| < 1 := by sorry

end NUMINAMATH_CALUDE_ceiling_2023_ceiling_quadratic_inequality_ceiling_equality_distance_l2995_299515


namespace NUMINAMATH_CALUDE_nathaniel_tickets_l2995_299523

/-- Given a person with initial tickets who gives a fixed number of tickets to each of their friends,
    calculate the number of remaining tickets. -/
def remaining_tickets (initial : ℕ) (given_per_friend : ℕ) (num_friends : ℕ) : ℕ :=
  initial - given_per_friend * num_friends

/-- Theorem stating that given 11 initial tickets, giving 2 tickets to each of 4 friends
    results in 3 remaining tickets. -/
theorem nathaniel_tickets : remaining_tickets 11 2 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_nathaniel_tickets_l2995_299523


namespace NUMINAMATH_CALUDE_cinema_ticket_pricing_l2995_299597

theorem cinema_ticket_pricing (adult_price : ℚ) : 
  (10 * adult_price + 6 * (adult_price / 2) = 35) →
  ((12 * adult_price + 8 * (adult_price / 2)) * (9 / 10) = 504 / 13) := by
  sorry

end NUMINAMATH_CALUDE_cinema_ticket_pricing_l2995_299597


namespace NUMINAMATH_CALUDE_existence_equivalence_l2995_299533

theorem existence_equivalence (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ 2^x * (3*x + a) < 1) ↔ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_existence_equivalence_l2995_299533


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l2995_299539

def num_chickens : ℕ := 5
def num_dogs : ℕ := 3
def num_cats : ℕ := 6
def total_animals : ℕ := num_chickens + num_dogs + num_cats

def group_arrangements : ℕ := 3

theorem animal_arrangement_count :
  (group_arrangements * num_chickens.factorial * num_dogs.factorial * num_cats.factorial : ℕ) = 1555200 :=
by sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l2995_299539


namespace NUMINAMATH_CALUDE_f_inequality_l2995_299589

/-- A quadratic function with positive leading coefficient and axis of symmetry at x=1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating that f(3^x) > f(2^x) for x > 0 -/
theorem f_inequality (a b c : ℝ) (h_a : a > 0) (h_sym : ∀ x, f a b c (2 - x) = f a b c x) :
  ∀ x > 0, f a b c (3^x) > f a b c (2^x) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2995_299589


namespace NUMINAMATH_CALUDE_special_arithmetic_l2995_299580

/-- In a country with non-standard arithmetic, prove that if 1/5 of 8 equals 4,
    and 1/4 of a number X equals 10, then X must be 16. -/
theorem special_arithmetic (country_fifth : ℚ → ℚ) (X : ℚ) :
  country_fifth 8 = 4 →
  country_fifth X = 10 →
  X = 16 :=
by
  sorry

#check special_arithmetic

end NUMINAMATH_CALUDE_special_arithmetic_l2995_299580


namespace NUMINAMATH_CALUDE_smallest_possible_b_l2995_299581

-- Define the conditions
def no_triangle_2ab (a b : ℝ) : Prop := 2 + a ≤ b
def no_triangle_inverse (a b : ℝ) : Prop := 1/b + 1/a ≤ 2

-- State the theorem
theorem smallest_possible_b :
  ∀ a b : ℝ, 2 < a → a < b → no_triangle_2ab a b → no_triangle_inverse a b →
  b ≥ (5 + Real.sqrt 17) / 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l2995_299581


namespace NUMINAMATH_CALUDE_paint_remaining_l2995_299556

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint = 1 → 
  (initial_paint - initial_paint / 4) / 2 = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_paint_remaining_l2995_299556


namespace NUMINAMATH_CALUDE_decompose_375_l2995_299535

theorem decompose_375 : 
  375 = 3 * 100 + 7 * 10 + 5 * 1 := by sorry

end NUMINAMATH_CALUDE_decompose_375_l2995_299535


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l2995_299552

/-- 
Given a quadratic equation kx^2 - 2x - 1 = 0 with two distinct real roots,
prove that the range of values for k is k > -1 and k ≠ 0.
-/
theorem quadratic_equation_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0) →
  (k > -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l2995_299552


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2995_299549

theorem arithmetic_sequence_ratio : 
  let n1 := (60 - 4) / 4 + 1
  let n2 := (75 - 5) / 5 + 1
  let sum1 := n1 * (4 + 60) / 2
  let sum2 := n2 * (5 + 75) / 2
  sum1 / sum2 = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2995_299549


namespace NUMINAMATH_CALUDE_response_rate_is_sixty_percent_l2995_299544

/-- The response rate percentage for a mail questionnaire -/
def response_rate (responses_needed : ℕ) (questionnaires_mailed : ℕ) : ℚ :=
  (responses_needed : ℚ) / (questionnaires_mailed : ℚ) * 100

/-- Theorem: The response rate is 60% when 750 responses are needed and 1250 questionnaires are mailed -/
theorem response_rate_is_sixty_percent :
  response_rate 750 1250 = 60 := by
  sorry

#eval response_rate 750 1250

end NUMINAMATH_CALUDE_response_rate_is_sixty_percent_l2995_299544


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2995_299569

-- Define the line and hyperbola
def line (a x y : ℝ) : Prop := 2 * a * x - y + 2 * a^2 = 0
def hyperbola (a x y : ℝ) : Prop := x^2 / a^2 - y^2 / 4 = 1

-- Define the condition for no focus
def no_focus (a : ℝ) : Prop := ∀ x y : ℝ, line a x y → hyperbola a x y → False

-- State the theorem
theorem sufficient_not_necessary_condition (a : ℝ) (h : a > 0) :
  (a ≥ 2 → no_focus a) ∧ ¬(no_focus a → a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2995_299569


namespace NUMINAMATH_CALUDE_delta_value_l2995_299565

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ - 3 → Δ = -9 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l2995_299565


namespace NUMINAMATH_CALUDE_correct_assignment_properties_l2995_299536

-- Define the properties of assignment statements
inductive AssignmentProperty : Type
  | InitialValue : AssignmentProperty
  | AssignExpression : AssignmentProperty
  | MultipleAssignments : AssignmentProperty
  | NoMultipleAssignments : AssignmentProperty

-- Define a function to check if a property is correct
def isCorrectProperty (prop : AssignmentProperty) : Prop :=
  match prop with
  | AssignmentProperty.InitialValue => True
  | AssignmentProperty.AssignExpression => True
  | AssignmentProperty.MultipleAssignments => True
  | AssignmentProperty.NoMultipleAssignments => False

-- Theorem stating the correct properties of assignment statements
theorem correct_assignment_properties :
  ∀ (prop : AssignmentProperty),
    isCorrectProperty prop ↔
      (prop = AssignmentProperty.InitialValue ∨
       prop = AssignmentProperty.AssignExpression ∨
       prop = AssignmentProperty.MultipleAssignments) :=
by sorry

end NUMINAMATH_CALUDE_correct_assignment_properties_l2995_299536


namespace NUMINAMATH_CALUDE_max_photo_area_l2995_299529

/-- Given a rectangular frame with area 59.6 square centimeters,
    prove that the maximum area of each of four equal-sized,
    non-overlapping photos within the frame is 14.9 square centimeters. -/
theorem max_photo_area (frame_area : ℝ) (num_photos : ℕ) :
  frame_area = 59.6 ∧ num_photos = 4 →
  (frame_area / num_photos : ℝ) = 14.9 := by
  sorry

end NUMINAMATH_CALUDE_max_photo_area_l2995_299529


namespace NUMINAMATH_CALUDE_pyramid_properties_l2995_299586

-- Define the cone and pyramid
structure Cone where
  height : ℝ
  slantHeight : ℝ

structure Pyramid where
  cone : Cone
  OB : ℝ

-- Define the properties of the cone and pyramid
def isValidCone (c : Cone) : Prop :=
  c.height = 4 ∧ c.slantHeight = 5

def isValidPyramid (p : Pyramid) : Prop :=
  isValidCone p.cone ∧ p.OB = 3

-- Define the properties to be proved
def pyramidVolume (p : Pyramid) : ℝ := sorry

def dihedralAngleAB (p : Pyramid) : ℝ := sorry

def circumscribedSphereRadius (p : Pyramid) : ℝ := sorry

-- Main theorem
theorem pyramid_properties (p : Pyramid) 
  (h : isValidPyramid p) : 
  ∃ (v d r : ℝ),
    pyramidVolume p = v ∧
    dihedralAngleAB p = d ∧
    circumscribedSphereRadius p = r :=
  sorry

end NUMINAMATH_CALUDE_pyramid_properties_l2995_299586


namespace NUMINAMATH_CALUDE_equation_solution_l2995_299525

theorem equation_solution : ∃ (a b c d : ℕ+), 
  2014 = (a.val ^ 2 + b.val ^ 2) * (c.val ^ 3 - d.val ^ 3) ∧ 
  a.val = 5 ∧ b.val = 9 ∧ c.val = 3 ∧ d.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2995_299525


namespace NUMINAMATH_CALUDE_second_discount_percentage_l2995_299521

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 10000 →
  first_discount = 20 →
  final_price = 6840 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l2995_299521


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l2995_299579

/-- The area of the shaded region inside a square with side length 16 cm but outside
    four quarter circles with radius 6 cm at each corner is 256 - 36π cm². -/
theorem shaded_area_square_with_quarter_circles (π : ℝ) :
  let square_side : ℝ := 16
  let circle_radius : ℝ := 6
  let square_area : ℝ := square_side ^ 2
  let quarter_circle_area : ℝ := π * circle_radius ^ 2 / 4
  let total_quarter_circles_area : ℝ := 4 * quarter_circle_area
  let shaded_area : ℝ := square_area - total_quarter_circles_area
  shaded_area = 256 - 36 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l2995_299579


namespace NUMINAMATH_CALUDE_cos_sum_fifteenth_l2995_299588

theorem cos_sum_fifteenth : Real.cos (2 * Real.pi / 15) + Real.cos (4 * Real.pi / 15) + Real.cos (8 * Real.pi / 15) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_fifteenth_l2995_299588


namespace NUMINAMATH_CALUDE_baron_munchausen_contradiction_l2995_299505

theorem baron_munchausen_contradiction (d : ℝ) (T : ℝ) (h1 : d > 0) (h2 : T > 0) : 
  ¬(d / 2 = 5 * (d / (2 * 5)) ∧ d / 2 = 6 * (T / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_baron_munchausen_contradiction_l2995_299505


namespace NUMINAMATH_CALUDE_intersection_point_l2995_299596

def P : ℝ × ℝ × ℝ := (10, -1, 3)
def Q : ℝ × ℝ × ℝ := (20, -11, 8)
def R : ℝ × ℝ × ℝ := (3, 8, -9)
def S : ℝ × ℝ × ℝ := (5, 0, 6)

def line_PQ (t : ℝ) : ℝ × ℝ × ℝ :=
  (P.1 + t * (Q.1 - P.1), P.2.1 + t * (Q.2.1 - P.2.1), P.2.2 + t * (Q.2.2 - P.2.2))

def line_RS (s : ℝ) : ℝ × ℝ × ℝ :=
  (R.1 + s * (S.1 - R.1), R.2.1 + s * (S.2.1 - R.2.1), R.2.2 + s * (S.2.2 - R.2.2))

theorem intersection_point :
  ∃ t s : ℝ, line_PQ t = line_RS s ∧ line_PQ t = (11, -2, 3.5) := by sorry

end NUMINAMATH_CALUDE_intersection_point_l2995_299596


namespace NUMINAMATH_CALUDE_smallest_marble_count_l2995_299516

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the total number of marbles in the urn -/
def total_marbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green

/-- Calculates the number of ways to select marbles according to the given events -/
def event_probability (mc : MarbleCount) (event : Fin 4) : ℕ :=
  match event with
  | 0 => mc.blue.choose 4
  | 1 => (mc.red.choose 2) * (mc.white.choose 2)
  | 2 => (mc.red.choose 2) * (mc.white.choose 1) * (mc.blue.choose 1)
  | 3 => mc.red * mc.white * mc.blue * mc.green

/-- Checks if all events have equal probability -/
def events_equally_likely (mc : MarbleCount) : Prop :=
  ∀ i j : Fin 4, event_probability mc i = event_probability mc j

/-- The main theorem stating the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : 
  ∃ (mc : MarbleCount), 
    events_equally_likely mc ∧ 
    total_marbles mc = 13 ∧ 
    (∀ (mc' : MarbleCount), events_equally_likely mc' → total_marbles mc' ≥ 13) :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l2995_299516


namespace NUMINAMATH_CALUDE_valid_distributions_count_l2995_299542

/-- Represents a triangular array of squares with 11 rows -/
def TriangularArray := Fin 11 → Fin 11 → ℕ

/-- Represents the bottom row of the triangular array -/
def BottomRow := Fin 11 → Fin 2

/-- Calculates the value of a square in the array based on the two squares below it -/
def calculateSquare (array : TriangularArray) (row : Fin 11) (col : Fin 11) : ℕ :=
  if row = 10 then array row col
  else array (row + 1) col + array (row + 1) (col + 1)

/-- Fills the triangular array based on the bottom row -/
def fillArray (bottomRow : BottomRow) : TriangularArray :=
  sorry

/-- Checks if the top square of the array is a multiple of 3 -/
def isTopMultipleOfThree (array : TriangularArray) : Bool :=
  array 0 0 % 3 = 0

/-- Counts the number of valid bottom row distributions -/
def countValidDistributions : ℕ :=
  sorry

theorem valid_distributions_count :
  countValidDistributions = 640 := by sorry

end NUMINAMATH_CALUDE_valid_distributions_count_l2995_299542


namespace NUMINAMATH_CALUDE_fraction_of_number_l2995_299561

theorem fraction_of_number : (3 / 4 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * 5020 = 753 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_number_l2995_299561


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2995_299540

theorem fraction_to_decimal : (58 : ℚ) / 125 = (464 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2995_299540


namespace NUMINAMATH_CALUDE_meat_for_hamburgers_l2995_299567

/-- Given that 3 pounds of meat make 8 hamburgers, prove that 9 pounds of meat are needed for 24 hamburgers -/
theorem meat_for_hamburgers (meat_per_8 : ℝ) (hamburgers : ℝ) 
  (h1 : meat_per_8 = 3) 
  (h2 : hamburgers = 24) : 
  (meat_per_8 / 8) * hamburgers = 9 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_hamburgers_l2995_299567


namespace NUMINAMATH_CALUDE_decreasing_function_l2995_299584

-- Define the four functions
def f1 (x : ℝ) : ℝ := x^2 + 1
def f2 (x : ℝ) : ℝ := -x^2 + 1
def f3 (x : ℝ) : ℝ := 2*x + 1
def f4 (x : ℝ) : ℝ := -2*x + 1

-- Theorem statement
theorem decreasing_function : 
  (∀ x : ℝ, HasDerivAt f4 (-2) x) ∧ 
  (∀ x : ℝ, (HasDerivAt f1 (2*x) x) ∨ (HasDerivAt f2 (-2*x) x) ∨ (HasDerivAt f3 2 x)) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_l2995_299584


namespace NUMINAMATH_CALUDE_equilateral_triangle_condition_l2995_299518

/-- A function that checks if a natural number n satisfies the conditions for forming an equilateral triangle with sticks of lengths 1 to n. -/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- The sum of the first n natural numbers. -/
def sum_of_first_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Theorem stating the necessary and sufficient conditions for forming an equilateral triangle with sticks of lengths 1 to n. -/
theorem equilateral_triangle_condition (n : ℕ) :
  (sum_of_first_n n % 3 = 0) ↔ can_form_equilateral_triangle n :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_condition_l2995_299518


namespace NUMINAMATH_CALUDE_local_tax_deduction_l2995_299509

-- Define Carl's hourly wage in dollars
def carlHourlyWage : ℝ := 25

-- Define the local tax rate as a percentage
def localTaxRate : ℝ := 2.0

-- Define the conversion rate from dollars to cents
def dollarsToCents : ℝ := 100

-- Theorem to prove
theorem local_tax_deduction :
  (carlHourlyWage * dollarsToCents * (localTaxRate / 100)) = 50 := by
  sorry


end NUMINAMATH_CALUDE_local_tax_deduction_l2995_299509


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2995_299595

/-- Given that the solution set of ax^2 + 5x + b > 0 is {x | 2 < x < 3},
    prove that the solution set of bx^2 - 5x + a > 0 is (-1/2, -1/3) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x + b > 0 ↔ 2 < x ∧ x < 3) →
  (∀ x : ℝ, b*x^2 - 5*x + a > 0 ↔ -1/2 < x ∧ x < -1/3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2995_299595


namespace NUMINAMATH_CALUDE_incorrect_simplification_l2995_299532

theorem incorrect_simplification : 
  -(1 + 1/2) ≠ 1 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_simplification_l2995_299532


namespace NUMINAMATH_CALUDE_triangle_quadratic_no_solution_l2995_299577

theorem triangle_quadratic_no_solution (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b^2 + c^2 - a^2)^2 - 4*(b^2)*(c^2) < 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_quadratic_no_solution_l2995_299577


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_of_1000800000_l2995_299531

def n : ℕ := 1000800000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a > b ∧ b > c ∧ c > d ∧ d > e ∧
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    ∀ (x : ℕ), x ∣ n → x ≤ e ∨ x = d ∨ x = c ∨ x = b ∨ x = a ∨ x = n)

theorem fifth_largest_divisor_of_1000800000 :
  is_fifth_largest_divisor 62550000 :=
sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_of_1000800000_l2995_299531


namespace NUMINAMATH_CALUDE_logs_per_tree_l2995_299517

/-- The number of pieces of firewood produced from one log -/
def pieces_per_log : ℕ := 5

/-- The total number of pieces of firewood chopped -/
def total_pieces : ℕ := 500

/-- The number of trees chopped down -/
def trees_chopped : ℕ := 25

/-- Theorem: Given the conditions, each tree produces 4 logs -/
theorem logs_per_tree : 
  (total_pieces / pieces_per_log) / trees_chopped = 4 := by
  sorry

end NUMINAMATH_CALUDE_logs_per_tree_l2995_299517


namespace NUMINAMATH_CALUDE_distinct_roots_iff_m_lt_half_m_value_when_inverse_sum_neg_two_l2995_299514

/-- Given a quadratic equation x^2 - 2(m-1)x + m^2 = 0 -/
def quadratic_equation (m x : ℝ) : Prop :=
  x^2 - 2*(m-1)*x + m^2 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (-2*(m-1))^2 - 4*m^2

theorem distinct_roots_iff_m_lt_half (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂) ↔ m < 1/2 :=
sorry

theorem m_value_when_inverse_sum_neg_two (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ 1/x₁ + 1/x₂ = -2) →
  m = (-1 - Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_iff_m_lt_half_m_value_when_inverse_sum_neg_two_l2995_299514


namespace NUMINAMATH_CALUDE_college_board_committee_count_l2995_299573

/-- Represents a college board. -/
structure Board :=
  (total_members : ℕ)
  (professors : ℕ)
  (non_professors : ℕ)
  (h_total : total_members = professors + non_professors)

/-- Represents a committee formed from the board. -/
structure Committee :=
  (size : ℕ)
  (min_professors : ℕ)

/-- Calculates the number of valid committees for a given board and committee requirements. -/
def count_valid_committees (board : Board) (committee : Committee) : ℕ :=
  sorry

/-- The specific board in the problem. -/
def college_board : Board :=
  { total_members := 15
  , professors := 7
  , non_professors := 8
  , h_total := by rfl }

/-- The specific committee requirements in the problem. -/
def required_committee : Committee :=
  { size := 5
  , min_professors := 2 }

theorem college_board_committee_count :
  count_valid_committees college_board required_committee = 2457 :=
sorry

end NUMINAMATH_CALUDE_college_board_committee_count_l2995_299573


namespace NUMINAMATH_CALUDE_bill_eric_age_difference_l2995_299593

/-- The age difference between two brothers, given their total age and the older brother's age. -/
def age_difference (total_age : ℕ) (older_brother_age : ℕ) : ℕ :=
  older_brother_age - (total_age - older_brother_age)

/-- Theorem stating the age difference between Bill and Eric -/
theorem bill_eric_age_difference :
  let total_age : ℕ := 28
  let bill_age : ℕ := 16
  age_difference total_age bill_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_bill_eric_age_difference_l2995_299593


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2995_299572

/-- Given a rectangle with perimeter 60, its maximum possible area is 225 -/
theorem rectangle_max_area :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * x + 2 * y = 60 →
  x * y ≤ 225 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2995_299572


namespace NUMINAMATH_CALUDE_algebraic_simplification_l2995_299513

theorem algebraic_simplification (x y : ℝ) :
  (3 * x - 2 * y - 4) * (x + y + 5) - (x + 2 * y + 5) * (3 * x - y - 1) =
  -4 * x * y - 3 * x - 7 * y - 15 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l2995_299513


namespace NUMINAMATH_CALUDE_inequality_range_of_a_l2995_299559

theorem inequality_range_of_a (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |x^2 - a*x| + b < 0) ↔
  ((b ≥ -1 ∧ b < 0 ∧ a ∈ Set.Ioo (1 + b) (2 * Real.sqrt (-b))) ∨
   (b < -1 ∧ a ∈ Set.Ioo (1 + b) (1 - b))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_of_a_l2995_299559


namespace NUMINAMATH_CALUDE_sum_of_products_l2995_299546

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 281) 
  (h2 : a + b + c = 17) : 
  a*b + b*c + c*a = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2995_299546


namespace NUMINAMATH_CALUDE_average_listening_time_is_44_l2995_299551

/-- Represents the distribution of audience members and their listening durations -/
structure AudienceDistribution where
  total_audience : ℕ
  lecture_duration : ℕ
  full_listeners_percent : ℚ
  non_listeners_percent : ℚ
  half_listeners_percent : ℚ

/-- Calculates the average listening time given an audience distribution -/
def average_listening_time (dist : AudienceDistribution) : ℚ :=
  sorry

/-- The theorem stating that the average listening time is 44 minutes -/
theorem average_listening_time_is_44 (dist : AudienceDistribution) : 
  dist.lecture_duration = 90 ∧ 
  dist.full_listeners_percent = 30/100 ∧ 
  dist.non_listeners_percent = 15/100 ∧
  dist.half_listeners_percent = 40/100 * (1 - dist.full_listeners_percent - dist.non_listeners_percent) →
  average_listening_time dist = 44 :=
sorry

end NUMINAMATH_CALUDE_average_listening_time_is_44_l2995_299551


namespace NUMINAMATH_CALUDE_only_integer_solution_l2995_299512

theorem only_integer_solution (x y z : ℝ) (n : ℤ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  2 * x^2 + 3 * y^2 + 6 * z^2 = n →
  3 * x + 4 * y + 5 * z = 23 →
  n = 127 :=
by sorry

end NUMINAMATH_CALUDE_only_integer_solution_l2995_299512


namespace NUMINAMATH_CALUDE_quadratic_roots_same_sign_l2995_299591

theorem quadratic_roots_same_sign (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + 2*x₁ + m = 0 ∧ 
   x₂^2 + 2*x₂ + m = 0 ∧
   (x₁ > 0 ∧ x₂ > 0 ∨ x₁ < 0 ∧ x₂ < 0)) →
  (0 < m ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_same_sign_l2995_299591


namespace NUMINAMATH_CALUDE_data_transmission_time_l2995_299524

/-- Represents the number of blocks to be sent -/
def num_blocks : ℕ := 30

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 1024

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 256

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Proves that the time to send the data is 2 minutes -/
theorem data_transmission_time :
  (num_blocks * chunks_per_block) / transmission_rate / seconds_per_minute = 2 :=
sorry

end NUMINAMATH_CALUDE_data_transmission_time_l2995_299524


namespace NUMINAMATH_CALUDE_car_round_trip_speed_l2995_299538

theorem car_round_trip_speed 
  (distance : ℝ) 
  (speed_there : ℝ) 
  (avg_speed : ℝ) 
  (speed_back : ℝ) : 
  distance = 150 → 
  speed_there = 75 → 
  avg_speed = 50 → 
  (2 * distance) / (distance / speed_there + distance / speed_back) = avg_speed →
  speed_back = 37.5 := by
sorry

end NUMINAMATH_CALUDE_car_round_trip_speed_l2995_299538


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l2995_299543

theorem spencer_walk_distance (house_to_library : ℝ) (library_to_post_office : ℝ) (post_office_to_home : ℝ)
  (h1 : house_to_library = 0.3)
  (h2 : library_to_post_office = 0.1)
  (h3 : post_office_to_home = 0.4) :
  house_to_library + library_to_post_office + post_office_to_home = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_spencer_walk_distance_l2995_299543


namespace NUMINAMATH_CALUDE_no_fraction_satisfies_conditions_l2995_299526

theorem no_fraction_satisfies_conditions : ¬∃ (a b n : ℕ), 
  (a < b) ∧ 
  (n < a) ∧ 
  (n < b) ∧ 
  ((a + n : ℚ) / (b + n)) > (3 / 2) * (a / b) ∧
  ((a - n : ℚ) / (b - n)) > (1 / 2) * (a / b) := by
  sorry

end NUMINAMATH_CALUDE_no_fraction_satisfies_conditions_l2995_299526


namespace NUMINAMATH_CALUDE_expression_evaluation_l2995_299548

theorem expression_evaluation :
  11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2995_299548


namespace NUMINAMATH_CALUDE_average_age_of_new_men_l2995_299504

theorem average_age_of_new_men (n : ℕ) (initial_average : ℝ) 
  (replaced_ages : List ℝ) (age_increase : ℝ) :
  n = 20 ∧ 
  replaced_ages = [21, 23, 25, 27] ∧ 
  age_increase = 2 →
  (n * (initial_average + age_increase) - n * initial_average + replaced_ages.sum) / replaced_ages.length = 34 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_new_men_l2995_299504


namespace NUMINAMATH_CALUDE_isabella_hair_length_l2995_299587

/-- The length of Isabella's hair before the haircut -/
def hair_length_before : ℕ := sorry

/-- The length of Isabella's hair after the haircut -/
def hair_length_after : ℕ := 9

/-- The length of hair that was cut off -/
def hair_length_cut : ℕ := 9

/-- Theorem stating that the length of Isabella's hair before the haircut
    is equal to the sum of the length after the haircut and the length cut off -/
theorem isabella_hair_length : hair_length_before = hair_length_after + hair_length_cut := by
  sorry

end NUMINAMATH_CALUDE_isabella_hair_length_l2995_299587


namespace NUMINAMATH_CALUDE_average_of_data_set_l2995_299583

def data_set : List ℤ := [3, -2, 4, 1, 4]

theorem average_of_data_set :
  (data_set.sum : ℚ) / data_set.length = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_set_l2995_299583


namespace NUMINAMATH_CALUDE_class_size_proof_l2995_299578

theorem class_size_proof (avg_age : ℝ) (avg_age_5 : ℝ) (avg_age_9 : ℝ) (age_15th : ℕ) : 
  avg_age = 15 → 
  avg_age_5 = 14 → 
  avg_age_9 = 16 → 
  age_15th = 11 → 
  ∃ (N : ℕ), N = 15 ∧ N * avg_age = 5 * avg_age_5 + 9 * avg_age_9 + age_15th :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l2995_299578


namespace NUMINAMATH_CALUDE_vector_decomposition_l2995_299570

theorem vector_decomposition (e₁ e₂ a : ℝ × ℝ) :
  e₁ = (1, 2) →
  e₂ = (-2, 3) →
  a = (-1, 2) →
  a = (1/7 : ℝ) • e₁ + (4/7 : ℝ) • e₂ := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l2995_299570


namespace NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l2995_299503

/-- Given an equilateral triangle with two vertices at (3,7) and (13,7),
    prove that the y-coordinate of the third vertex in the first quadrant is 7 + 5√3 -/
theorem equilateral_triangle_third_vertex_y_coord :
  ∀ (x y : ℝ),
  let A : ℝ × ℝ := (3, 7)
  let B : ℝ × ℝ := (13, 7)
  let C : ℝ × ℝ := (x, y)
  (x > 0 ∧ y > 0) →  -- C is in the first quadrant
  (dist A B = dist B C ∧ dist B C = dist C A) →  -- Triangle is equilateral
  y = 7 + 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l2995_299503


namespace NUMINAMATH_CALUDE_three_pipes_used_l2995_299522

def tank_filling_problem (rate_a rate_b rate_c : ℝ) : Prop :=
  let total_rate := rate_a + rate_b + rate_c
  rate_c = 2 * rate_b ∧
  rate_b = 2 * rate_a ∧
  rate_a = 1 / 70 ∧
  total_rate = 1 / 10

theorem three_pipes_used (rate_a rate_b rate_c : ℝ) 
  (h : tank_filling_problem rate_a rate_b rate_c) : 
  ∃ (n : ℕ), n = 3 ∧ n > 0 := by
  sorry

#check three_pipes_used

end NUMINAMATH_CALUDE_three_pipes_used_l2995_299522


namespace NUMINAMATH_CALUDE_geometric_progression_min_sum_l2995_299547

/-- A geometric progression with positive terms -/
def GeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_progression_min_sum (a : ℕ → ℝ) (h : GeometricProgression a) 
    (h_prod : a 2 * a 10 = 9) : 
  a 5 + a 7 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_min_sum_l2995_299547


namespace NUMINAMATH_CALUDE_range_of_t_below_line_l2995_299562

/-- A point (x, y) is below a line ax + by + c = 0 if ax + by + c > 0 -/
def IsBelowLine (x y a b c : ℝ) : Prop := a * x + b * y + c > 0

/-- The theorem stating the range of t given the conditions -/
theorem range_of_t_below_line :
  ∀ t : ℝ, IsBelowLine 2 (3 * t) 2 (-1) 6 → t < 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_below_line_l2995_299562


namespace NUMINAMATH_CALUDE_rectangle_area_l2995_299574

/-- A rectangle with length twice its width and perimeter 84 cm has an area of 392 cm² -/
theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  length = 2 * width →
  perimeter = 84 →
  perimeter = 2 * (length + width) →
  area = length * width →
  area = 392 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2995_299574


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l2995_299528

theorem power_of_negative_cube (a : ℝ) : (-a^3)^4 = a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l2995_299528


namespace NUMINAMATH_CALUDE_regression_analysis_l2995_299576

-- Define the data points
def data : List (ℝ × ℝ) := [(5, 17), (6, 20), (8, 25), (9, 28), (12, 35)]

-- Define the regression equation
def regression_equation (x : ℝ) (a : ℝ) : ℝ := 2.6 * x + a

-- Theorem statement
theorem regression_analysis :
  -- 1. Center point
  (let x_mean := (data.map Prod.fst).sum / data.length
   let y_mean := (data.map Prod.snd).sum / data.length
   (x_mean, y_mean) = (8, 25)) ∧
  -- 2. Y-intercept
  (∃ a : ℝ, a = 4.2 ∧
    regression_equation 8 a = 25) ∧
  -- 3. Residual when x = 5
  (let a := 4.2
   let y_pred := regression_equation 5 a
   let y_actual := 17
   y_actual - y_pred = -0.2) := by
  sorry


end NUMINAMATH_CALUDE_regression_analysis_l2995_299576


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l2995_299568

/-- Given a cube of side length n, painted blue on all faces and split into unit cubes,
    if exactly one-third of the total faces of the unit cubes are blue, then n = 3 -/
theorem painted_cube_theorem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l2995_299568


namespace NUMINAMATH_CALUDE_circle_equation_l2995_299530

/-- Theorem: Equation of a circle with specific properties -/
theorem circle_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ y : ℝ, (0 - a)^2 + (y - b)^2 = (a^2 + b^2) → |y| ≤ 1) →
  (∀ x : ℝ, (x - a)^2 + (0 - b)^2 = (a^2 + b^2) → |x| ≤ 2) →
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ 
    (x₁ - a)^2 + (0 - b)^2 = (a^2 + b^2) ∧
    (x₂ - a)^2 + (0 - b)^2 = (a^2 + b^2) ∧
    (x₂ - x₁) / (4 - (x₂ - x₁)) = 3) →
  a = Real.sqrt 7 ∧ b = 2 ∧ a^2 + b^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l2995_299530


namespace NUMINAMATH_CALUDE_modulus_of_complex_power_l2995_299541

theorem modulus_of_complex_power (z : ℂ) :
  z = 2 - 3 * Real.sqrt 2 * Complex.I →
  Complex.abs (z^4) = 484 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_power_l2995_299541


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l2995_299537

theorem negation_of_existential_proposition :
  ¬(∃ x : ℝ, x < 0 ∧ x^2 - 2*x > 0) ↔ (∀ x : ℝ, x < 0 → x^2 - 2*x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l2995_299537


namespace NUMINAMATH_CALUDE_no_triple_perfect_squares_l2995_299590

theorem no_triple_perfect_squares : 
  ¬ ∃ (a b c : ℕ+), 
    (∃ (x y z : ℕ), (a^2 * b * c + 2 : ℕ) = x^2 ∧ 
                    (b^2 * c * a + 2 : ℕ) = y^2 ∧ 
                    (c^2 * a * b + 2 : ℕ) = z^2) :=
by sorry

end NUMINAMATH_CALUDE_no_triple_perfect_squares_l2995_299590


namespace NUMINAMATH_CALUDE_square_rectangle_intersection_l2995_299592

theorem square_rectangle_intersection (EFGH_side_length MO LO shaded_area : ℝ) :
  EFGH_side_length = 8 →
  MO = 12 →
  LO = 8 →
  shaded_area = (MO * LO) / 2 →
  shaded_area = EFGH_side_length * (EFGH_side_length - EM) →
  EM = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_rectangle_intersection_l2995_299592


namespace NUMINAMATH_CALUDE_cross_shaped_graph_paper_rectangles_l2995_299571

/-- Calculates the number of rectangles in a grid --/
def rectangleCount (m n : ℕ) : ℕ :=
  (m * (m + 1) * n * (n + 1)) / 4

/-- Calculates the sum of squares from 1 to n --/
def sumOfSquares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

/-- The side length of the original square graph paper in mm --/
def originalSideLength : ℕ := 30

/-- The side length of the cut-away corner squares in mm --/
def cornerSideLength : ℕ := 10

/-- The total number of smallest squares in the original graph paper --/
def totalSmallestSquares : ℕ := 900

theorem cross_shaped_graph_paper_rectangles :
  let totalRectangles := rectangleCount originalSideLength originalSideLength
  let cornerRectangles := 4 * rectangleCount cornerSideLength originalSideLength
  let remainingSquares := 2 * sumOfSquares originalSideLength - sumOfSquares (originalSideLength - 2 * cornerSideLength)
  totalRectangles - cornerRectangles - remainingSquares = 144130 := by
  sorry

end NUMINAMATH_CALUDE_cross_shaped_graph_paper_rectangles_l2995_299571


namespace NUMINAMATH_CALUDE_stock_price_increase_percentage_l2995_299575

theorem stock_price_increase_percentage (total_stocks : ℕ) (higher_price_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : higher_price_stocks = 1080)
  (h3 : higher_price_stocks > total_stocks - higher_price_stocks) :
  let lower_price_stocks := total_stocks - higher_price_stocks
  (higher_price_stocks - lower_price_stocks) / lower_price_stocks * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_percentage_l2995_299575


namespace NUMINAMATH_CALUDE_fermat_like_equation_l2995_299510

theorem fermat_like_equation (a b c : ℕ) (h1 : Even c) (h2 : a^5 + 4*b^5 = c^5) : b = 0 := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_equation_l2995_299510


namespace NUMINAMATH_CALUDE_interchange_difference_for_62_l2995_299560

def is_two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def interchange_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem interchange_difference_for_62 :
  is_two_digit_number 62 ∧ digit_sum 62 = 8 →
  62 - interchange_digits 62 = 36 := by
  sorry

end NUMINAMATH_CALUDE_interchange_difference_for_62_l2995_299560


namespace NUMINAMATH_CALUDE_circle_hexagon_area_difference_l2995_299553

theorem circle_hexagon_area_difference (r : ℝ) (s : ℝ) : 
  r = (Real.sqrt 2) / 2 →
  s = 1 →
  (π * r^2) - (3 * Real.sqrt 3 / 2 * s^2) = π / 2 - 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_hexagon_area_difference_l2995_299553


namespace NUMINAMATH_CALUDE_solve_average_weight_l2995_299554

def average_weight_problem (num_boys num_girls : ℕ) (avg_weight_boys avg_weight_girls : ℚ) : Prop :=
  let total_children := num_boys + num_girls
  let total_weight := (num_boys : ℚ) * avg_weight_boys + (num_girls : ℚ) * avg_weight_girls
  let avg_weight_all := total_weight / total_children
  (↑(round avg_weight_all) : ℚ) = 141

theorem solve_average_weight :
  average_weight_problem 8 5 160 110 := by
  sorry

end NUMINAMATH_CALUDE_solve_average_weight_l2995_299554


namespace NUMINAMATH_CALUDE_field_fully_fenced_l2995_299501

/-- Proves that a square field can be completely fenced given the specified conditions -/
theorem field_fully_fenced (field_area : ℝ) (wire_cost : ℝ) (budget : ℝ) : 
  field_area = 5000 → 
  wire_cost = 30 → 
  budget = 120000 → 
  ∃ (wire_length : ℝ), wire_length = budget / wire_cost ∧ 
    wire_length ≥ 4 * Real.sqrt field_area := by
  sorry

end NUMINAMATH_CALUDE_field_fully_fenced_l2995_299501


namespace NUMINAMATH_CALUDE_trig_ratio_problem_l2995_299563

theorem trig_ratio_problem (a : ℝ) (h : 2 * Real.sin a = 3 * Real.cos a) :
  (4 * Real.sin a + Real.cos a) / (5 * Real.sin a - 2 * Real.cos a) = 14 / 11 := by
  sorry

end NUMINAMATH_CALUDE_trig_ratio_problem_l2995_299563


namespace NUMINAMATH_CALUDE_mass_CO2_from_CO_combustion_l2995_299599

/-- The mass of CO2 produced from the complete combustion of CO -/
def mass_CO2_produced (initial_moles_CO : ℝ) (molar_mass_CO2 : ℝ) : ℝ :=
  initial_moles_CO * molar_mass_CO2

/-- The balanced chemical reaction coefficient for CO2 -/
def CO2_coefficient : ℚ := 2

/-- The balanced chemical reaction coefficient for CO -/
def CO_coefficient : ℚ := 2

theorem mass_CO2_from_CO_combustion 
  (initial_moles_CO : ℝ)
  (molar_mass_CO2 : ℝ)
  (h1 : initial_moles_CO = 3)
  (h2 : molar_mass_CO2 = 44.01) :
  mass_CO2_produced initial_moles_CO molar_mass_CO2 = 132.03 := by
  sorry

end NUMINAMATH_CALUDE_mass_CO2_from_CO_combustion_l2995_299599


namespace NUMINAMATH_CALUDE_betty_age_l2995_299534

theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 10) :
  betty = 5 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l2995_299534
