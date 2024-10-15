import Mathlib

namespace NUMINAMATH_CALUDE_f_not_monotonic_exists_even_f_exists_three_zeros_l2594_259459

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - abs (x + a)

-- Theorem 1: f is not monotonic for any a
theorem f_not_monotonic : ∀ a : ℝ, ¬(Monotone (f a)) := by sorry

-- Theorem 2: There exists an 'a' for which f is even
theorem exists_even_f : ∃ a : ℝ, ∀ x : ℝ, f a x = f a (-x) := by sorry

-- Theorem 3: There exists a negative 'a' for which f has three zeros
theorem exists_three_zeros : ∃ a : ℝ, a < 0 ∧ (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) := by sorry

end NUMINAMATH_CALUDE_f_not_monotonic_exists_even_f_exists_three_zeros_l2594_259459


namespace NUMINAMATH_CALUDE_purple_probability_ten_sided_die_l2594_259405

/-- Represents a die with a specific number of sides and purple faces -/
structure Die :=
  (sides : ℕ)
  (purpleFaces : ℕ)
  (hPurple : purpleFaces ≤ sides)

/-- Calculates the probability of rolling a purple face on a given die -/
def probabilityPurple (d : Die) : ℚ :=
  d.purpleFaces / d.sides

/-- Theorem stating that for a 10-sided die with 2 purple faces, 
    the probability of rolling a purple face is 1/5 -/
theorem purple_probability_ten_sided_die :
  ∀ d : Die, d.sides = 10 → d.purpleFaces = 2 → probabilityPurple d = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_purple_probability_ten_sided_die_l2594_259405


namespace NUMINAMATH_CALUDE_inverse_proposition_l2594_259403

theorem inverse_proposition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x^3 > y^3) →
  (a^3 > b^3 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l2594_259403


namespace NUMINAMATH_CALUDE_real_number_classification_l2594_259420

theorem real_number_classification : 
  ∀ x : ℝ, x < 0 ∨ x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_real_number_classification_l2594_259420


namespace NUMINAMATH_CALUDE_problem_1_l2594_259446

theorem problem_1 : Real.sqrt 12 + (-2024)^(0 : ℕ) - 4 * Real.sin (π / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2594_259446


namespace NUMINAMATH_CALUDE_total_oranges_proof_l2594_259408

def initial_purchase : ℕ := 10
def additional_purchase : ℕ := 5
def weeks : ℕ := 3

def total_oranges : ℕ :=
  let week1_purchase := initial_purchase + additional_purchase
  let subsequent_weeks_purchase := 2 * week1_purchase
  week1_purchase + (weeks - 1) * subsequent_weeks_purchase

theorem total_oranges_proof :
  total_oranges = 75 :=
by sorry

end NUMINAMATH_CALUDE_total_oranges_proof_l2594_259408


namespace NUMINAMATH_CALUDE_intersection_point_l2594_259426

/-- The first curve -/
def curve1 (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 2

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := 2 * x^3 + x^2 + 7

/-- Theorem stating that (-1, -1) is the only intersection point of the two curves -/
theorem intersection_point : 
  ∃! p : ℝ × ℝ, p.1 = -1 ∧ p.2 = -1 ∧ curve1 p.1 = curve2 p.1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2594_259426


namespace NUMINAMATH_CALUDE_slices_left_over_l2594_259466

-- Define the number of slices for each pizza size
def small_pizza_slices : ℕ := 4
def large_pizza_slices : ℕ := 8

-- Define the number of pizzas purchased
def small_pizzas_bought : ℕ := 3
def large_pizzas_bought : ℕ := 2

-- Define the number of slices each person eats
def george_slices : ℕ := 3
def bob_slices : ℕ := george_slices + 1
def susie_slices : ℕ := bob_slices / 2
def bill_slices : ℕ := 3
def fred_slices : ℕ := 3
def mark_slices : ℕ := 3

-- Calculate total slices and slices eaten
def total_slices : ℕ := small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought
def total_slices_eaten : ℕ := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

-- Theorem to prove
theorem slices_left_over : total_slices - total_slices_eaten = 10 := by
  sorry

end NUMINAMATH_CALUDE_slices_left_over_l2594_259466


namespace NUMINAMATH_CALUDE_water_usage_calculation_l2594_259448

/-- Water pricing policy and usage calculation -/
theorem water_usage_calculation (m : ℝ) (usage : ℝ) (payment : ℝ) : 
  (m > 0) →
  (usage > 0) →
  (payment = if usage ≤ 10 then m * usage else 10 * m + 2 * m * (usage - 10)) →
  (payment = 16 * m) →
  (usage = 13) :=
by sorry

end NUMINAMATH_CALUDE_water_usage_calculation_l2594_259448


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l2594_259484

theorem student_multiplication_problem (initial_number : ℕ) (final_result : ℕ) : 
  initial_number = 48 → final_result = 102 → ∃ (x : ℕ), initial_number * x - 138 = final_result ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l2594_259484


namespace NUMINAMATH_CALUDE_graveyard_bones_count_l2594_259461

/-- Represents the number of bones in a skeleton based on its type -/
def bonesInSkeleton (type : String) : ℕ :=
  match type with
  | "woman" => 20
  | "man" => 25
  | "child" => 10
  | _ => 0

/-- Calculates the total number of bones in the graveyard -/
def totalBonesInGraveyard : ℕ :=
  let totalSkeletons : ℕ := 20
  let womenSkeletons : ℕ := totalSkeletons / 2
  let menSkeletons : ℕ := (totalSkeletons - womenSkeletons) / 2
  let childrenSkeletons : ℕ := totalSkeletons - womenSkeletons - menSkeletons
  
  womenSkeletons * bonesInSkeleton "woman" +
  menSkeletons * bonesInSkeleton "man" +
  childrenSkeletons * bonesInSkeleton "child"

theorem graveyard_bones_count :
  totalBonesInGraveyard = 375 := by
  sorry

#eval totalBonesInGraveyard

end NUMINAMATH_CALUDE_graveyard_bones_count_l2594_259461


namespace NUMINAMATH_CALUDE_equal_sum_sequence_definition_l2594_259430

/-- Definition of an equal sum sequence -/
def is_equal_sum_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = a (n + 1) + a (n + 2)

/-- Theorem stating the definition of an equal sum sequence -/
theorem equal_sum_sequence_definition (a : ℕ → ℝ) :
  is_equal_sum_sequence a ↔
    ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = a (n + 1) + a (n + 2) :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_definition_l2594_259430


namespace NUMINAMATH_CALUDE_parabola_line_intersection_theorem_l2594_259453

-- Define the parabola C: y^2 = 3x
def C (x y : ℝ) : Prop := y^2 = 3*x

-- Define the line l: y = (3/2)x + b
def l (x y b : ℝ) : Prop := y = (3/2)*x + b

-- Define the intersection points E and F
def E (x y : ℝ) : Prop := C x y ∧ ∃ b, l x y b
def F (x y : ℝ) : Prop := C x y ∧ ∃ b, l x y b

-- Define point H on x-axis
def H (x : ℝ) : Prop := ∃ b, l x 0 b

-- Define the vector relationship
def vector_relationship (e_x e_y f_x f_y h_x k : ℝ) : Prop :=
  (h_x - e_x, -e_y) = k • (f_x - h_x, f_y)

-- Theorem statement
theorem parabola_line_intersection_theorem 
  (e_x e_y f_x f_y h_x : ℝ) (k : ℝ) :
  C e_x e_y → C f_x f_y →
  (∃ b, l e_x e_y b ∧ l f_x f_y b) →
  H h_x →
  vector_relationship e_x e_y f_x f_y h_x k →
  k > 1 →
  (f_x - e_x)^2 + (f_y - e_y)^2 = (4*Real.sqrt 13 / 3)^2 →
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_theorem_l2594_259453


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l2594_259457

theorem imaginary_unit_sum (i : ℂ) : i^2 = -1 → i + i^2 + i^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l2594_259457


namespace NUMINAMATH_CALUDE_concentric_circles_properties_l2594_259404

/-- Given two concentric circles where a chord is tangent to the smaller circle -/
structure ConcentricCircles where
  /-- Radius of the smaller circle -/
  r₁ : ℝ
  /-- Length of the chord tangent to the smaller circle -/
  chord_length : ℝ
  /-- The chord is tangent to the smaller circle -/
  tangent_chord : True

/-- Theorem about the radius of the larger circle and the area between the circles -/
theorem concentric_circles_properties (c : ConcentricCircles) 
  (h₁ : c.r₁ = 30)
  (h₂ : c.chord_length = 120) :
  ∃ (r₂ : ℝ) (area : ℝ),
    r₂ = 30 * Real.sqrt 5 ∧ 
    area = 3600 * Real.pi ∧
    r₂ > c.r₁ ∧
    area = Real.pi * (r₂^2 - c.r₁^2) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_properties_l2594_259404


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_80_degree_angle_l2594_259447

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We represent angles in degrees as natural numbers
  angle1 : ℕ
  angle2 : ℕ
  angle3 : ℕ
  -- Sum of angles is 180°
  sum_180 : angle1 + angle2 + angle3 = 180
  -- Two angles are equal (property of isosceles triangle)
  two_equal : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3)

-- Theorem statement
theorem isosceles_triangle_with_80_degree_angle 
  (t : IsoscelesTriangle) 
  (h : t.angle1 = 80 ∨ t.angle2 = 80 ∨ t.angle3 = 80) :
  (t.angle1 = 80 ∧ t.angle2 = 80 ∧ t.angle3 = 20) ∨
  (t.angle1 = 80 ∧ t.angle2 = 20 ∧ t.angle3 = 80) ∨
  (t.angle1 = 20 ∧ t.angle2 = 80 ∧ t.angle3 = 80) ∨
  (t.angle1 = 50 ∧ t.angle2 = 50 ∧ t.angle3 = 80) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_with_80_degree_angle_l2594_259447


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2594_259400

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), r > 0 → 4 * π * r^2 = 400 * π → (4 / 3) * π * r^3 = (4000 / 3) * π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2594_259400


namespace NUMINAMATH_CALUDE_arithmetic_mean_4_16_l2594_259490

theorem arithmetic_mean_4_16 (x : ℝ) : x = (4 + 16) / 2 → x = 10 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_4_16_l2594_259490


namespace NUMINAMATH_CALUDE_weighted_average_markup_percentage_l2594_259488

-- Define the fruit types
inductive Fruit
| Apple
| Orange
| Banana

-- Define the properties for each fruit
def cost (f : Fruit) : ℝ :=
  match f with
  | Fruit.Apple => 30
  | Fruit.Orange => 40
  | Fruit.Banana => 50

def markup_percentage (f : Fruit) : ℝ :=
  match f with
  | Fruit.Apple => 0.10
  | Fruit.Orange => 0.15
  | Fruit.Banana => 0.20

def quantity (f : Fruit) : ℕ :=
  match f with
  | Fruit.Apple => 25
  | Fruit.Orange => 20
  | Fruit.Banana => 15

-- Calculate the markup amount for a fruit
def markup_amount (f : Fruit) : ℝ :=
  cost f * markup_percentage f

-- Calculate the selling price for a fruit
def selling_price (f : Fruit) : ℝ :=
  cost f + markup_amount f

-- Calculate the total selling price for all fruits
def total_selling_price : ℝ :=
  selling_price Fruit.Apple + selling_price Fruit.Orange + selling_price Fruit.Banana

-- Calculate the total cost for all fruits
def total_cost : ℝ :=
  cost Fruit.Apple + cost Fruit.Orange + cost Fruit.Banana

-- Calculate the total markup for all fruits
def total_markup : ℝ :=
  markup_amount Fruit.Apple + markup_amount Fruit.Orange + markup_amount Fruit.Banana

-- Theorem: The weighted average markup percentage is 15.83%
theorem weighted_average_markup_percentage :
  (total_markup / total_cost) * 100 = 15.83 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_markup_percentage_l2594_259488


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l2594_259470

/-- Proves the factorization of x²y - 4xy + 4y -/
theorem factorization_1 (x y : ℝ) : x^2*y - 4*x*y + 4*y = y*(x-2)^2 := by
  sorry

/-- Proves the factorization of x² - 4y² -/
theorem factorization_2 (x y : ℝ) : x^2 - 4*y^2 = (x+2*y)*(x-2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l2594_259470


namespace NUMINAMATH_CALUDE_expression_equals_one_l2594_259496

theorem expression_equals_one :
  (150^2 - 13^2) / (90^2 - 17^2) * ((90 - 17) * (90 + 17)) / ((150 - 13) * (150 + 13)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2594_259496


namespace NUMINAMATH_CALUDE_household_survey_total_l2594_259429

theorem household_survey_total (total : ℕ) (neither : ℕ) (only_a : ℕ) (both : ℕ) : 
  total = 180 ∧ 
  neither = 80 ∧ 
  only_a = 60 ∧ 
  both = 10 ∧ 
  (∃ (only_b : ℕ), only_b = 3 * both) →
  total = neither + only_a + both + (3 * both) :=
by sorry

end NUMINAMATH_CALUDE_household_survey_total_l2594_259429


namespace NUMINAMATH_CALUDE_x_interval_equivalence_l2594_259463

theorem x_interval_equivalence (x : ℝ) : 
  (2/3 < x ∧ x < 3/4) ↔ (2 < 3*x ∧ 3*x < 3) ∧ (2 < 4*x ∧ 4*x < 3) := by
sorry

end NUMINAMATH_CALUDE_x_interval_equivalence_l2594_259463


namespace NUMINAMATH_CALUDE_number_of_groups_l2594_259422

def lunch_times : List ℕ := [10, 12, 15, 8, 16, 18, 19, 18, 20, 18, 18, 20, 28, 22, 25, 20, 15, 16, 21, 16]

def class_interval : ℕ := 4

theorem number_of_groups : 
  let min_time := lunch_times.minimum?
  let max_time := lunch_times.maximum?
  match min_time, max_time with
  | some min, some max => 
    (max - min) / class_interval + 1 = 6
  | _, _ => False
  := by sorry

end NUMINAMATH_CALUDE_number_of_groups_l2594_259422


namespace NUMINAMATH_CALUDE_exists_non_illuminating_rotation_l2594_259407

/-- Represents a three-dimensional cube --/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a projector that illuminates an octant --/
structure Projector where
  position : ℝ × ℝ × ℝ
  illumination : Set (ℝ × ℝ × ℝ)

/-- Represents a rotation in three-dimensional space --/
structure Rotation where
  matrix : Matrix (Fin 3) (Fin 3) ℝ

/-- Function to check if a point is illuminated by the projector --/
def is_illuminated (p : Projector) (point : ℝ × ℝ × ℝ) : Prop :=
  point ∈ p.illumination

/-- Function to apply a rotation to a projector --/
def rotate_projector (r : Rotation) (p : Projector) : Projector :=
  sorry

/-- Theorem stating that there exists a rotation such that no vertices are illuminated --/
theorem exists_non_illuminating_rotation (c : Cube) (p : Projector) :
  p.position = (0, 0, 0) →  -- Projector is at the center of the cube
  ∃ (r : Rotation), ∀ (v : Fin 8), ¬is_illuminated (rotate_projector r p) (c.vertices v) :=
sorry

end NUMINAMATH_CALUDE_exists_non_illuminating_rotation_l2594_259407


namespace NUMINAMATH_CALUDE_suji_age_l2594_259487

theorem suji_age (abi_age suji_age : ℕ) : 
  (abi_age : ℚ) / suji_age = 5 / 4 →
  ((abi_age + 3) : ℚ) / (suji_age + 3) = 11 / 9 →
  suji_age = 24 := by
sorry

end NUMINAMATH_CALUDE_suji_age_l2594_259487


namespace NUMINAMATH_CALUDE_difference_of_squares_l2594_259492

theorem difference_of_squares (x : ℝ) : x^2 - 25 = (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2594_259492


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l2594_259468

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area := (1 / 2) * a * b
  ∀ c : ℝ, (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → area ≤ (1 / 2) * a * c ∧ area ≤ (1 / 2) * b * c :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l2594_259468


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2594_259455

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 2) = 15 → ∃ y : ℝ, (y + 3) * (y - 2) = 15 ∧ x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2594_259455


namespace NUMINAMATH_CALUDE_chloe_first_round_points_l2594_259411

/-- Represents the points scored in a trivia game. -/
structure TriviaProblem where
  first_round : ℤ
  second_round : ℤ
  last_round : ℤ
  total_points : ℤ

/-- The solution to Chloe's trivia game problem. -/
theorem chloe_first_round_points (game : TriviaProblem) 
  (h1 : game.second_round = 50)
  (h2 : game.last_round = -4)
  (h3 : game.total_points = 86)
  (h4 : game.first_round + game.second_round + game.last_round = game.total_points) :
  game.first_round = 40 := by
  sorry

end NUMINAMATH_CALUDE_chloe_first_round_points_l2594_259411


namespace NUMINAMATH_CALUDE_log_sqrt_45_l2594_259454

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_sqrt_45 (a b : ℝ) (h1 : log10 2 = a) (h2 : log10 3 = b) :
  log10 (Real.sqrt 45) = -a/2 + b + 1/2 := by sorry

end NUMINAMATH_CALUDE_log_sqrt_45_l2594_259454


namespace NUMINAMATH_CALUDE_unique_sum_of_squares_l2594_259493

def is_sum_of_squares (n : ℕ) (k : ℕ) : Prop :=
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℕ), n = a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 ∧ (a₁ ≠ 0 → k ≥ 1) ∧
    (a₂ ≠ 0 → k ≥ 2) ∧ (a₃ ≠ 0 → k ≥ 3) ∧ (a₄ ≠ 0 → k ≥ 4) ∧ (a₅ ≠ 0 → k = 5)

def has_unique_representation (n : ℕ) : Prop :=
  ∃! (k : ℕ) (a₁ a₂ a₃ a₄ a₅ : ℕ), k ≤ 5 ∧ is_sum_of_squares n k ∧
    n = a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2

theorem unique_sum_of_squares :
  {n : ℕ | has_unique_representation n} = {1, 2, 3, 6, 7, 15} := by sorry

end NUMINAMATH_CALUDE_unique_sum_of_squares_l2594_259493


namespace NUMINAMATH_CALUDE_consecutive_squares_not_perfect_square_l2594_259438

theorem consecutive_squares_not_perfect_square (n : ℕ) : 
  ∃ k : ℕ, (n - 1)^2 + n^2 + (n + 1)^2 ≠ k^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_not_perfect_square_l2594_259438


namespace NUMINAMATH_CALUDE_greatest_n_value_l2594_259483

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 ∧ ∃ m : ℤ, m = 10 ∧ 101 * m^2 ≤ 12100 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_l2594_259483


namespace NUMINAMATH_CALUDE_existence_of_x_and_y_l2594_259433

theorem existence_of_x_and_y (f : ℝ → ℝ) : ∃ x y : ℝ, f (x - f y) > y * f x + x := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_and_y_l2594_259433


namespace NUMINAMATH_CALUDE_sin_eq_tan_sin_unique_solution_l2594_259406

theorem sin_eq_tan_sin_unique_solution :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ Real.arcsin (1/2) ∧ Real.sin x = Real.tan (Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_sin_eq_tan_sin_unique_solution_l2594_259406


namespace NUMINAMATH_CALUDE_plane_equation_from_point_and_normal_specific_plane_equation_l2594_259497

/-- Given a point M and a normal vector N, this theorem states that
    the equation Ax + By + Cz + D = 0 represents a plane passing through M
    and perpendicular to N, where (A, B, C) are the components of N. -/
theorem plane_equation_from_point_and_normal (M : ℝ × ℝ × ℝ) (N : ℝ × ℝ × ℝ) :
  let (x₀, y₀, z₀) := M
  let (A, B, C) := N
  let D := -(A * x₀ + B * y₀ + C * z₀)
  ∀ (x y z : ℝ), A * x + B * y + C * z + D = 0 ↔
    ((x - x₀) * A + (y - y₀) * B + (z - z₀) * C = 0 ∧
     ∃ (t : ℝ), x - x₀ = t * A ∧ y - y₀ = t * B ∧ z - z₀ = t * C) :=
by sorry

/-- The equation 4x + 3y + 2z - 27 = 0 represents a plane that passes through
    the point (2, 3, 5) and is perpendicular to the vector (4, 3, 2). -/
theorem specific_plane_equation :
  let M : ℝ × ℝ × ℝ := (2, 3, 5)
  let N : ℝ × ℝ × ℝ := (4, 3, 2)
  ∀ (x y z : ℝ), 4 * x + 3 * y + 2 * z - 27 = 0 ↔
    ((x - 2) * 4 + (y - 3) * 3 + (z - 5) * 2 = 0 ∧
     ∃ (t : ℝ), x - 2 = t * 4 ∧ y - 3 = t * 3 ∧ z - 5 = t * 2) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_from_point_and_normal_specific_plane_equation_l2594_259497


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l2594_259440

theorem sum_of_absolute_values_zero (a b : ℝ) : 
  |a + 3| + |2*b - 4| = 0 → a + b = -1 := by sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l2594_259440


namespace NUMINAMATH_CALUDE_cloves_discrepancy_l2594_259410

/-- Represents the number of creatures that can be repelled by 3 cloves of garlic -/
structure RepelRatio :=
  (vampires : ℚ)
  (wights : ℚ)
  (vampire_bats : ℚ)

/-- Represents the number of creatures to be repelled -/
structure CreaturesToRepel :=
  (vampires : ℕ)
  (wights : ℕ)
  (vampire_bats : ℕ)

/-- Calculates the number of cloves needed based on the repel ratio and creatures to repel -/
def cloves_needed (ratio : RepelRatio) (creatures : CreaturesToRepel) : ℚ :=
  3 * (creatures.vampires / ratio.vampires + 
       creatures.wights / ratio.wights + 
       creatures.vampire_bats / ratio.vampire_bats)

/-- The main theorem stating that the calculated cloves needed is not equal to 72 -/
theorem cloves_discrepancy (ratio : RepelRatio) (creatures : CreaturesToRepel) :
  ratio.vampires = 1 →
  ratio.wights = 3 →
  ratio.vampire_bats = 8 →
  creatures.vampires = 30 →
  creatures.wights = 12 →
  creatures.vampire_bats = 40 →
  cloves_needed ratio creatures ≠ 72 := by
  sorry


end NUMINAMATH_CALUDE_cloves_discrepancy_l2594_259410


namespace NUMINAMATH_CALUDE_log_equation_solution_l2594_259427

theorem log_equation_solution (t : ℝ) (h : t > 0) :
  4 * (Real.log t / Real.log 3) = Real.log (4 * t) / Real.log 3 → t = (4 : ℝ) ^ (1 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2594_259427


namespace NUMINAMATH_CALUDE_percentage_calculation_l2594_259456

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 70 → (P / 100) * N - 10 = 25 → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2594_259456


namespace NUMINAMATH_CALUDE_wage_productivity_relationship_l2594_259499

/-- Represents the regression line equation for worker's wage and labor productivity -/
def regression_line (x : ℝ) : ℝ := 50 + 80 * x

/-- Theorem stating the relationship between changes in labor productivity and worker's wage -/
theorem wage_productivity_relationship :
  ∀ x : ℝ, regression_line (x + 1) - regression_line x = 80 := by
  sorry

end NUMINAMATH_CALUDE_wage_productivity_relationship_l2594_259499


namespace NUMINAMATH_CALUDE_f_digit_sum_properties_l2594_259480

/-- The function f(n) = 3n^2 + n + 1 -/
def f (n : ℕ+) : ℕ := 3 * n.val^2 + n.val + 1

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the smallest sum of digits and existence of 1999 sum -/
theorem f_digit_sum_properties :
  (∃ (n : ℕ+), sum_of_digits (f n) = 3) ∧ 
  (∀ (n : ℕ+), sum_of_digits (f n) ≥ 3) ∧
  (∃ (n : ℕ+), sum_of_digits (f n) = 1999) :=
sorry

end NUMINAMATH_CALUDE_f_digit_sum_properties_l2594_259480


namespace NUMINAMATH_CALUDE_shaded_area_regular_octagon_l2594_259450

/-- The area of the shaded region in a regular octagon with side length 12 cm, 
    formed by connecting every other vertex (creating two squares) -/
theorem shaded_area_regular_octagon (side_length : ℝ) (h : side_length = 12) : 
  let octagon_area := 8 * (1/2 * side_length * (side_length / 2))
  octagon_area = 288 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_regular_octagon_l2594_259450


namespace NUMINAMATH_CALUDE_initial_donuts_l2594_259474

theorem initial_donuts (remaining : ℕ) (missing_percent : ℚ) : 
  remaining = 9 → missing_percent = 70/100 → 
  (1 - missing_percent) * 30 = remaining :=
by sorry

end NUMINAMATH_CALUDE_initial_donuts_l2594_259474


namespace NUMINAMATH_CALUDE_gas_price_difference_l2594_259469

/-- The difference between actual and expected gas prices -/
theorem gas_price_difference (actual_gallons : ℕ) (actual_price : ℕ) (expected_gallons : ℕ) :
  actual_gallons = 10 →
  actual_price = 150 →
  expected_gallons = 12 →
  actual_price - (actual_gallons * actual_price / expected_gallons) = 25 := by
sorry

end NUMINAMATH_CALUDE_gas_price_difference_l2594_259469


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2594_259486

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  focal_distance : ℝ
  asymptote_slope : ℝ

-- Define the standard equation of the hyperbola
def standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, (y^2 / (8/5)) - (x^2 / (72/5)) = 1

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) 
  (h_foci : h.focal_distance = 8)
  (h_asymptote : h.asymptote_slope = 1/3) :
  standard_equation h :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2594_259486


namespace NUMINAMATH_CALUDE_kerosene_cost_is_44_cents_l2594_259442

/-- The cost of a liter of kerosene in cents -/
def kerosene_cost_cents (rice_cost_dollars : ℚ) : ℚ :=
  let egg_dozen_cost := rice_cost_dollars
  let egg_cost := egg_dozen_cost / 12
  let half_liter_kerosene_cost := 8 * egg_cost
  let liter_kerosene_cost_dollars := 2 * half_liter_kerosene_cost
  100 * liter_kerosene_cost_dollars

/-- Theorem stating that the cost of a liter of kerosene is 44 cents -/
theorem kerosene_cost_is_44_cents : 
  kerosene_cost_cents (33/100) = 44 := by
sorry

#eval kerosene_cost_cents (33/100)

end NUMINAMATH_CALUDE_kerosene_cost_is_44_cents_l2594_259442


namespace NUMINAMATH_CALUDE_bookshelf_problem_l2594_259423

theorem bookshelf_problem (initial_books : ℕ) 
  (day1_borrow day1_return day2_borrow day2_return : ℤ) : 
  initial_books = 20 →
  day1_borrow = -3 →
  day1_return = 1 →
  day2_borrow = -1 →
  day2_return = 2 →
  (initial_books : ℤ) + day1_borrow + day1_return + day2_borrow + day2_return = 19 :=
by sorry

end NUMINAMATH_CALUDE_bookshelf_problem_l2594_259423


namespace NUMINAMATH_CALUDE_chord_relations_l2594_259472

theorem chord_relations (d s : ℝ) : 
  0 < s ∧ s < d ∧ d < 2 →  -- Conditions for chords in a unit circle
  (d - s = 1 ∧ d * s = 1 ∧ d^2 - s^2 = Real.sqrt 5) ↔
  (d = (1 + Real.sqrt 5) / 2 ∧ s = (-1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_chord_relations_l2594_259472


namespace NUMINAMATH_CALUDE_cube_solid_surface_area_l2594_259444

/-- A solid composed of 7 identical cubes -/
structure CubeSolid where
  -- The volume of each individual cube
  cube_volume : ℝ
  -- The side length of each cube
  cube_side : ℝ
  -- The total volume of the solid
  total_volume : ℝ
  -- The surface area of the solid
  surface_area : ℝ
  -- Conditions
  cube_volume_def : cube_volume = total_volume / 7
  cube_side_def : cube_side ^ 3 = cube_volume
  surface_area_def : surface_area = 30 * (cube_side ^ 2)

/-- Theorem: If the total volume of the CubeSolid is 875 cm³, then its surface area is 750 cm² -/
theorem cube_solid_surface_area (s : CubeSolid) (h : s.total_volume = 875) : 
  s.surface_area = 750 := by
  sorry

#check cube_solid_surface_area

end NUMINAMATH_CALUDE_cube_solid_surface_area_l2594_259444


namespace NUMINAMATH_CALUDE_face_value_in_product_l2594_259451

/-- Given a number with specific local values for its digits and their product, 
    prove the face value of a digit with a given local value in the product. -/
theorem face_value_in_product (n : ℕ) (product : ℕ) 
  (local_value_6 : ℕ) (local_value_8 : ℕ) :
  n = 7098060 →
  local_value_6 = 6000 →
  local_value_8 = 80 →
  product = local_value_6 * local_value_8 →
  (product / 1000) % 10 = 6 →
  (product / 1000) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_face_value_in_product_l2594_259451


namespace NUMINAMATH_CALUDE_generatrix_angle_is_60_degrees_l2594_259445

/-- A cone whose lateral surface unfolds into a semicircle -/
structure SemiCircleCone where
  /-- The radius of the semicircle (equal to the generatrix of the cone) -/
  radius : ℝ
  /-- Assumption that the lateral surface unfolds into a semicircle -/
  lateral_surface_is_semicircle : True

/-- The angle between the two generatrices in the axial section of a cone
    whose lateral surface unfolds into a semicircle is 60 degrees -/
theorem generatrix_angle_is_60_degrees (cone : SemiCircleCone) :
  let angle_rad := Real.pi / 3
  angle_rad = Real.arccos (1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_generatrix_angle_is_60_degrees_l2594_259445


namespace NUMINAMATH_CALUDE_probability_theorem_l2594_259413

/-- The probability of having a child with younger brother, older brother, younger sister, and older sister
    given n > 4 children and equal probability of male and female births -/
def probability (n : ℕ) : ℚ :=
  1 - (n - 2 : ℚ) / 2^(n - 3)

/-- Theorem stating the probability for the given conditions -/
theorem probability_theorem (n : ℕ) (h : n > 4) :
  probability n = 1 - (n - 2 : ℚ) / 2^(n - 3) :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2594_259413


namespace NUMINAMATH_CALUDE_greatest_consecutive_sum_48_l2594_259414

/-- The sum of N consecutive integers starting from a -/
def sum_consecutive (a : ℤ) (N : ℕ) : ℤ := N * (2 * a + N - 1) / 2

/-- The proposition that 96 is the greatest number of consecutive integers whose sum is 48 -/
theorem greatest_consecutive_sum_48 :
  ∀ N : ℕ, (∃ a : ℤ, sum_consecutive a N = 48) → N ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_sum_48_l2594_259414


namespace NUMINAMATH_CALUDE_joel_contributed_22_toys_l2594_259439

/-- The number of toys Joel contributed to the donation -/
def joels_toys (toys_from_friends : ℕ) (total_toys : ℕ) : ℕ :=
  2 * ((total_toys - toys_from_friends) / 3)

/-- Theorem stating that Joel contributed 22 toys -/
theorem joel_contributed_22_toys : 
  joels_toys 75 108 = 22 := by
  sorry

end NUMINAMATH_CALUDE_joel_contributed_22_toys_l2594_259439


namespace NUMINAMATH_CALUDE_tomatoes_sold_on_saturday_l2594_259478

theorem tomatoes_sold_on_saturday (initial_shipment : ℕ) (rotted_amount : ℕ) (final_amount : ℕ) :
  initial_shipment = 1000 →
  rotted_amount = 200 →
  final_amount = 2500 →
  ∃ (sold_amount : ℕ),
    sold_amount = 300 ∧
    final_amount = initial_shipment - sold_amount - rotted_amount + 2 * initial_shipment :=
by sorry

end NUMINAMATH_CALUDE_tomatoes_sold_on_saturday_l2594_259478


namespace NUMINAMATH_CALUDE_basketball_competition_probabilities_l2594_259441

/-- Represents a team in the basketball competition -/
inductive Team : Type
| A
| B
| C

/-- The probability of one team winning against another -/
def win_probability (winner loser : Team) : ℚ :=
  match winner, loser with
  | Team.A, Team.B => 2/3
  | Team.A, Team.C => 2/3
  | Team.B, Team.C => 1/2
  | Team.B, Team.A => 1/3
  | Team.C, Team.A => 1/3
  | Team.C, Team.B => 1/2
  | _, _ => 0

/-- Team A gets a bye in the first match -/
def first_match_bye : Team := Team.A

/-- The probability that Team B is eliminated after the first three matches -/
def prob_b_eliminated_three_matches : ℚ := 11/36

/-- The probability that Team A wins the championship in only four matches -/
def prob_a_wins_four_matches : ℚ := 8/27

/-- The probability that a fifth match is needed -/
def prob_fifth_match_needed : ℚ := 35/54

theorem basketball_competition_probabilities :
  (prob_b_eliminated_three_matches = 11/36) ∧
  (prob_a_wins_four_matches = 8/27) ∧
  (prob_fifth_match_needed = 35/54) :=
by sorry

end NUMINAMATH_CALUDE_basketball_competition_probabilities_l2594_259441


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l2594_259477

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(abs (x + 2) + abs (x - 1) < a)) → a ∈ Set.Iic 3 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l2594_259477


namespace NUMINAMATH_CALUDE_maximum_value_implies_ratio_l2594_259418

/-- The function f(x) = x³ + ax² + bx - a² - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem maximum_value_implies_ratio (a b : ℝ) :
  (∀ x, f a b x ≤ f a b 1) ∧  -- f(x) reaches maximum at x = 1
  (f a b 1 = 10) ∧            -- The maximum value is 10
  (f_deriv a b 1 = 0)         -- Derivative is zero at x = 1
  → a / b = -2 / 3 := by sorry

end NUMINAMATH_CALUDE_maximum_value_implies_ratio_l2594_259418


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2594_259428

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a = 1 → a + b = 1) ↔ (∃ (a b : ℝ), a = 1 ∧ a + b ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2594_259428


namespace NUMINAMATH_CALUDE_christmas_tree_ornaments_l2594_259401

/-- The number of ornaments Pilyulkin hung on the tree -/
def pilyulkin_ornaments : ℕ := 3

/-- The number of ornaments Guslya hung on the tree -/
def guslya_ornaments : ℕ := 2 * pilyulkin_ornaments

/-- The number of ornaments Toropyzhka hung on the tree -/
def toropyzhka_ornaments : ℕ := pilyulkin_ornaments + 15

theorem christmas_tree_ornaments :
  guslya_ornaments = 2 * pilyulkin_ornaments ∧
  toropyzhka_ornaments = pilyulkin_ornaments + 15 ∧
  toropyzhka_ornaments = 2 * (guslya_ornaments + pilyulkin_ornaments) ∧
  pilyulkin_ornaments + guslya_ornaments + toropyzhka_ornaments = 27 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_ornaments_l2594_259401


namespace NUMINAMATH_CALUDE_or_propagation_l2594_259465

theorem or_propagation (p q r : Prop) (h1 : p ∨ q) (h2 : ¬p ∨ r) : q ∨ r := by
  sorry

end NUMINAMATH_CALUDE_or_propagation_l2594_259465


namespace NUMINAMATH_CALUDE_number_of_zeros_equal_l2594_259436

/-- f(n) denotes the number of 0's in the binary representation of a positive integer n -/
def f (n : ℕ+) : ℕ := sorry

/-- Theorem stating that the number of 0's in the binary representation of 8n + 7 
    is equal to the number of 0's in the binary representation of 4n + 3 -/
theorem number_of_zeros_equal (n : ℕ+) : f (8 * n + 7) = f (4 * n + 3) := by
  sorry

end NUMINAMATH_CALUDE_number_of_zeros_equal_l2594_259436


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l2594_259491

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l2594_259491


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l2594_259431

theorem sqrt_difference_equality : 
  Real.sqrt (49 + 81) - Real.sqrt (36 - 25) = Real.sqrt 130 - Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l2594_259431


namespace NUMINAMATH_CALUDE_simplify_fraction_l2594_259495

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) : (x^2 / (x - 2)) - (2*x / (x - 2)) = x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2594_259495


namespace NUMINAMATH_CALUDE_kids_difference_l2594_259409

theorem kids_difference (camp_kids home_kids : ℕ) 
  (h1 : camp_kids = 202958)
  (h2 : home_kids = 777622) :
  home_kids - camp_kids = 574664 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l2594_259409


namespace NUMINAMATH_CALUDE_cubic_expression_value_l2594_259460

theorem cubic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : 
  m^3 + 2*m^2 - 2001 = -2000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l2594_259460


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_2_range_of_m_for_all_real_solution_l2594_259479

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-1)*x - m

-- Part I
theorem solution_set_when_m_is_2 :
  ∀ x : ℝ, f 2 x < 0 ↔ -2 < x ∧ x < 1 := by sorry

-- Part II
theorem range_of_m_for_all_real_solution :
  ∀ m : ℝ, (∀ x : ℝ, f m x ≥ -1) ↔ -3 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_2_range_of_m_for_all_real_solution_l2594_259479


namespace NUMINAMATH_CALUDE_cupcakes_sold_l2594_259402

/-- Proves that Carol sold 9 cupcakes given the initial and final conditions -/
theorem cupcakes_sold (initial : ℕ) (made_after : ℕ) (final : ℕ) : 
  initial = 30 → made_after = 28 → final = 49 → 
  ∃ (sold : ℕ), sold = 9 ∧ initial - sold + made_after = final := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_sold_l2594_259402


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2594_259412

-- Problem 1
theorem problem_1 : Real.sqrt 8 - 2 * Real.sin (π / 4) + |1 - Real.sqrt 2| + (1 / 2)⁻¹ = 2 * Real.sqrt 2 + 1 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 1 ∧ ∀ x : ℝ, x^2 + 4*x - 5 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2594_259412


namespace NUMINAMATH_CALUDE_complex_coordinate_i_times_2_minus_i_l2594_259464

theorem complex_coordinate_i_times_2_minus_i : 
  (Complex.I * (2 - Complex.I)).re = 1 ∧ (Complex.I * (2 - Complex.I)).im = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinate_i_times_2_minus_i_l2594_259464


namespace NUMINAMATH_CALUDE_yearly_dumpling_production_l2594_259432

/-- The monthly production of dumplings in kilograms -/
def monthly_production : ℝ := 182.88

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The yearly production of dumplings in kilograms -/
def yearly_production : ℝ := monthly_production * months_in_year

/-- Theorem stating that the yearly production of dumplings is 2194.56 kg -/
theorem yearly_dumpling_production :
  yearly_production = 2194.56 := by
  sorry

end NUMINAMATH_CALUDE_yearly_dumpling_production_l2594_259432


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2594_259481

theorem rectangular_prism_volume (l w h : ℝ) 
  (face1 : l * w = 10)
  (face2 : w * h = 14)
  (face3 : l * h = 35) :
  l * w * h = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2594_259481


namespace NUMINAMATH_CALUDE_soda_lasts_40_days_l2594_259498

/-- The number of days soda bottles last given the initial quantity and daily consumption rate -/
def soda_duration (total_bottles : ℕ) (daily_consumption : ℕ) : ℕ :=
  total_bottles / daily_consumption

theorem soda_lasts_40_days :
  soda_duration 360 9 = 40 := by
  sorry

end NUMINAMATH_CALUDE_soda_lasts_40_days_l2594_259498


namespace NUMINAMATH_CALUDE_distance_inequality_l2594_259424

theorem distance_inequality (x y : ℝ) :
  Real.sqrt ((x + 4)^2 + (y + 2)^2) + Real.sqrt ((x - 5)^2 + (y + 4)^2) ≤
  Real.sqrt ((x - 2)^2 + (y - 6)^2) + Real.sqrt ((x - 5)^2 + (y - 6)^2) + 20 := by
  sorry

end NUMINAMATH_CALUDE_distance_inequality_l2594_259424


namespace NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l2594_259434

/-- Given x = (3 + √8)^1000, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 1 -/
theorem x_times_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 8) ^ 1000
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l2594_259434


namespace NUMINAMATH_CALUDE_equation_solution_l2594_259449

theorem equation_solution :
  ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2594_259449


namespace NUMINAMATH_CALUDE_mechanic_work_hours_l2594_259494

/-- Proves that a mechanic works 8 hours a day given the specified conditions -/
theorem mechanic_work_hours 
  (hourly_rate : ℕ) 
  (days_worked : ℕ) 
  (parts_cost : ℕ) 
  (total_paid : ℕ) 
  (h : hourly_rate = 60)
  (d : days_worked = 14)
  (p : parts_cost = 2500)
  (t : total_paid = 9220) :
  ∃ (hours_per_day : ℕ), 
    hours_per_day = 8 ∧ 
    hourly_rate * hours_per_day * days_worked + parts_cost = total_paid :=
by sorry

end NUMINAMATH_CALUDE_mechanic_work_hours_l2594_259494


namespace NUMINAMATH_CALUDE_wise_men_hat_guesses_l2594_259419

/-- Represents the maximum number of guaranteed correct hat color guesses -/
def max_guaranteed_correct_guesses (n k : ℕ) : ℕ :=
  n - k - 1

/-- Theorem stating the maximum number of guaranteed correct hat color guesses -/
theorem wise_men_hat_guesses (n k : ℕ) (h1 : k < n) :
  max_guaranteed_correct_guesses n k = n - k - 1 :=
by sorry

end NUMINAMATH_CALUDE_wise_men_hat_guesses_l2594_259419


namespace NUMINAMATH_CALUDE_apples_picked_l2594_259417

theorem apples_picked (benny_apples dan_apples : ℕ) 
  (h1 : benny_apples = 2) 
  (h2 : dan_apples = 9) : 
  benny_apples + dan_apples = 11 := by
sorry

end NUMINAMATH_CALUDE_apples_picked_l2594_259417


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l2594_259443

/-- Given a person who walks 30 km at a slower speed and could have walked 45 km at 15 km/hr 
    in the same amount of time, prove that the slower speed is 10 km/hr. -/
theorem slower_speed_calculation (x : ℝ) (h1 : x > 0) : 
  (30 / x = 45 / 15) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l2594_259443


namespace NUMINAMATH_CALUDE_number_of_friends_who_received_pebbles_l2594_259452

-- Define the given quantities
def total_weight_kg : ℕ := 36
def pebble_weight_g : ℕ := 250
def pebbles_per_friend : ℕ := 4

-- Define the conversion factor from kg to g
def kg_to_g : ℕ := 1000

-- Theorem to prove
theorem number_of_friends_who_received_pebbles :
  (total_weight_kg * kg_to_g) / (pebble_weight_g * pebbles_per_friend) = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_of_friends_who_received_pebbles_l2594_259452


namespace NUMINAMATH_CALUDE_complement_of_37_12_l2594_259476

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_37_12 :
  let a : Angle := ⟨37, 12⟩
  complement a = ⟨52, 48⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_37_12_l2594_259476


namespace NUMINAMATH_CALUDE_train_distance_l2594_259421

/-- Proves that a train traveling at 10 m/s for 8 seconds covers 80 meters. -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 10)
  (h2 : time = 8)
  (h3 : distance = speed * time) : 
  distance = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l2594_259421


namespace NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l2594_259475

theorem max_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y/2 + 1/x + 8/y = 10) : 
  ∃ (z : ℝ), z = 2*x + y ∧ ∀ (w : ℝ), (∃ (a b : ℝ) (ha : a > 0) (hb : b > 0), 
    w = 2*a + b ∧ a + b/2 + 1/a + 8/b = 10) → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l2594_259475


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2594_259462

/-- Given a line ax + by + c = 0 where ac < 0 and bc < 0, prove that the line does not pass through the third quadrant. -/
theorem line_not_in_third_quadrant (a b c : ℝ) (h1 : a * c < 0) (h2 : b * c < 0) :
  ¬∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2594_259462


namespace NUMINAMATH_CALUDE_simplest_fraction_of_0375_l2594_259437

theorem simplest_fraction_of_0375 (c d : ℕ+) : 
  (c : ℚ) / (d : ℚ) = 0.375 ∧ 
  (∀ (m n : ℕ+), (m : ℚ) / (n : ℚ) = 0.375 → c ≤ m ∧ d ≤ n) →
  c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_simplest_fraction_of_0375_l2594_259437


namespace NUMINAMATH_CALUDE_union_complement_theorem_l2594_259416

def I : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {2}

theorem union_complement_theorem :
  B ∪ (I \ A) = {2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_union_complement_theorem_l2594_259416


namespace NUMINAMATH_CALUDE_largest_number_with_sum_19_l2594_259435

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 10) ((m % 10) :: acc)
  aux n []

def sum_digits (n : ℕ) : ℕ :=
  (digits n).sum

def all_digits_different (n : ℕ) : Prop :=
  (digits n).Nodup

theorem largest_number_with_sum_19 :
  ∀ n : ℕ, 
    sum_digits n = 19 → 
    all_digits_different n → 
    n ≤ 65431 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_sum_19_l2594_259435


namespace NUMINAMATH_CALUDE_incircle_iff_reciprocal_heights_sum_l2594_259485

/-- A quadrilateral with heights h₁, h₂, h₃, h₄ -/
structure Quadrilateral where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ
  h₁_pos : 0 < h₁
  h₂_pos : 0 < h₂
  h₃_pos : 0 < h₃
  h₄_pos : 0 < h₄

/-- The property of having an incircle -/
def has_incircle (q : Quadrilateral) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∃ (center : ℝ × ℝ), True  -- We don't specify the exact conditions for an incircle

/-- The main theorem: a quadrilateral has an incircle iff the sum of reciprocals of opposite heights are equal -/
theorem incircle_iff_reciprocal_heights_sum (q : Quadrilateral) :
  has_incircle q ↔ 1 / q.h₁ + 1 / q.h₃ = 1 / q.h₂ + 1 / q.h₄ := by
  sorry

end NUMINAMATH_CALUDE_incircle_iff_reciprocal_heights_sum_l2594_259485


namespace NUMINAMATH_CALUDE_orange_problem_l2594_259473

theorem orange_problem (total : ℕ) (ripe_fraction : ℚ) (eaten_ripe_fraction : ℚ) (eaten_unripe_fraction : ℚ) :
  total = 96 →
  ripe_fraction = 1/2 →
  eaten_ripe_fraction = 1/4 →
  eaten_unripe_fraction = 1/8 →
  (total : ℚ) * ripe_fraction * (1 - eaten_ripe_fraction) +
  (total : ℚ) * (1 - ripe_fraction) * (1 - eaten_unripe_fraction) = 78 :=
by sorry

end NUMINAMATH_CALUDE_orange_problem_l2594_259473


namespace NUMINAMATH_CALUDE_circle_circumference_increase_l2594_259467

theorem circle_circumference_increase (d : ℝ) : 
  let original_circumference := π * d
  let new_circumference := π * (d + 2 * π)
  new_circumference - original_circumference = 2 * π^2 := by
sorry

end NUMINAMATH_CALUDE_circle_circumference_increase_l2594_259467


namespace NUMINAMATH_CALUDE_calculation_proof_l2594_259425

theorem calculation_proof : (-1)^2023 + 6 * Real.cos (π / 3) + (Real.pi - 3.14)^0 - Real.sqrt 16 = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2594_259425


namespace NUMINAMATH_CALUDE_jackie_daily_distance_l2594_259471

/-- Prove that Jackie walks 2 miles per day -/
theorem jackie_daily_distance (jessie_daily : Real) (days : Nat) (extra_distance : Real) :
  jessie_daily = 1.5 →
  days = 6 →
  extra_distance = 3 →
  ∃ (jackie_daily : Real),
    jackie_daily * days = jessie_daily * days + extra_distance ∧
    jackie_daily = 2 := by
  sorry

end NUMINAMATH_CALUDE_jackie_daily_distance_l2594_259471


namespace NUMINAMATH_CALUDE_ships_converge_l2594_259458

/-- Represents a ship with a given round trip duration -/
structure Ship where
  roundTripDays : ℕ

/-- Represents the fleet of ships -/
def Fleet : List Ship := [
  { roundTripDays := 2 },
  { roundTripDays := 3 },
  { roundTripDays := 5 }
]

/-- The number of days after which all ships converge -/
def convergenceDays : ℕ := 30

/-- Theorem stating that the ships converge after the specified number of days -/
theorem ships_converge :
  ∀ (ship : Ship), ship ∈ Fleet → convergenceDays % ship.roundTripDays = 0 := by
  sorry

#check ships_converge

end NUMINAMATH_CALUDE_ships_converge_l2594_259458


namespace NUMINAMATH_CALUDE_second_class_average_marks_l2594_259489

theorem second_class_average_marks (n1 n2 : ℕ) (avg1 avg_total : ℚ) :
  n1 = 35 →
  n2 = 45 →
  avg1 = 40 →
  avg_total = 51.25 →
  (n1 * avg1 + n2 * (n1 * avg1 + n2 * avg_total - n1 * avg1) / n2) / (n1 + n2) = avg_total →
  (n1 * avg1 + n2 * avg_total - n1 * avg1) / n2 = 60 :=
by sorry

end NUMINAMATH_CALUDE_second_class_average_marks_l2594_259489


namespace NUMINAMATH_CALUDE_faster_train_speed_faster_train_speed_is_10_l2594_259482

theorem faster_train_speed 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (speed_ratio : ℝ) : ℝ :=
  let slower_speed := (2 * train_length) / (crossing_time * (1 + speed_ratio))
  let faster_speed := speed_ratio * slower_speed
  faster_speed

theorem faster_train_speed_is_10 :
  faster_train_speed 200 30 3 = 10 := by sorry

end NUMINAMATH_CALUDE_faster_train_speed_faster_train_speed_is_10_l2594_259482


namespace NUMINAMATH_CALUDE_solutions_for_15_l2594_259415

/-- The number of different integer solutions for |x| + |y| = n -/
def numSolutions (n : ℕ) : ℕ :=
  4 * n

theorem solutions_for_15 : numSolutions 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_solutions_for_15_l2594_259415
