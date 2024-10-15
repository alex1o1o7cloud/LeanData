import Mathlib

namespace NUMINAMATH_CALUDE_last_three_average_l572_57235

theorem last_three_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 65 →
  (list.take 4).sum / 4 = 60 →
  (list.drop 4).sum / 3 = 215 / 3 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l572_57235


namespace NUMINAMATH_CALUDE_imaginary_unit_multiplication_l572_57256

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_multiplication :
  i * (1 + i) = -1 + i := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_multiplication_l572_57256


namespace NUMINAMATH_CALUDE_used_car_clients_l572_57253

theorem used_car_clients (num_cars : ℕ) (selections_per_car : ℕ) (cars_per_client : ℕ) : 
  num_cars = 16 → 
  selections_per_car = 3 → 
  cars_per_client = 2 → 
  (num_cars * selections_per_car) / cars_per_client = 24 := by
sorry

end NUMINAMATH_CALUDE_used_car_clients_l572_57253


namespace NUMINAMATH_CALUDE_functional_equation_implies_additive_l572_57293

/-- A function satisfying the given functional equation is additive. -/
theorem functional_equation_implies_additive (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y + x * y) = f x + f y + f (x * y)) :
  ∀ x y : ℝ, f (x + y) = f x + f y := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_additive_l572_57293


namespace NUMINAMATH_CALUDE_plot_length_is_61_l572_57223

def rectangular_plot_length (breadth : ℝ) (length_difference : ℝ) (fencing_cost_per_meter : ℝ) (total_fencing_cost : ℝ) : ℝ :=
  breadth + length_difference

theorem plot_length_is_61 (breadth : ℝ) :
  let length_difference : ℝ := 22
  let fencing_cost_per_meter : ℝ := 26.50
  let total_fencing_cost : ℝ := 5300
  let length := rectangular_plot_length breadth length_difference fencing_cost_per_meter total_fencing_cost
  let perimeter := 2 * (length + breadth)
  fencing_cost_per_meter * perimeter = total_fencing_cost →
  length = 61 := by
sorry

end NUMINAMATH_CALUDE_plot_length_is_61_l572_57223


namespace NUMINAMATH_CALUDE_mirror_area_l572_57258

/-- Given a rectangular frame with outer dimensions 100 cm by 140 cm and a uniform frame width of 12 cm,
    the area of the rectangular mirror that fits exactly inside the frame is 8816 cm². -/
theorem mirror_area (frame_width frame_height frame_thickness : ℕ) 
  (hw : frame_width = 100)
  (hh : frame_height = 140)
  (ht : frame_thickness = 12) :
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 8816 :=
by sorry

end NUMINAMATH_CALUDE_mirror_area_l572_57258


namespace NUMINAMATH_CALUDE_system_solution_l572_57268

theorem system_solution :
  let f (x y z : ℚ) := (x * y = x + 2 * y) ∧ (y * z = y + 3 * z) ∧ (z * x = z + 4 * x)
  ∀ x y z : ℚ, f x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 25/9 ∧ y = 25/7 ∧ z = 25/4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l572_57268


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l572_57246

theorem simplify_and_evaluate (m : ℤ) (h : m = -1) :
  -(m^2 - 3*m) + 2*(m^2 - m - 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l572_57246


namespace NUMINAMATH_CALUDE_reciprocal_of_recurring_decimal_l572_57211

/-- The decimal representation of the recurring decimal 0.363636... -/
def recurring_decimal : ℚ := 36 / 99

/-- The reciprocal of the common fraction form of 0.363636... -/
def reciprocal : ℚ := 11 / 4

theorem reciprocal_of_recurring_decimal : 
  (recurring_decimal)⁻¹ = reciprocal := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_recurring_decimal_l572_57211


namespace NUMINAMATH_CALUDE_chefs_wage_difference_l572_57271

theorem chefs_wage_difference (dishwasher1_wage dishwasher2_wage dishwasher3_wage : ℚ)
  (chef1_percentage chef2_percentage chef3_percentage : ℚ)
  (manager_wage : ℚ) :
  dishwasher1_wage = 6 →
  dishwasher2_wage = 7 →
  dishwasher3_wage = 8 →
  chef1_percentage = 1.2 →
  chef2_percentage = 1.25 →
  chef3_percentage = 1.3 →
  manager_wage = 12.5 →
  manager_wage - (dishwasher1_wage * chef1_percentage + 
                  dishwasher2_wage * chef2_percentage + 
                  dishwasher3_wage * chef3_percentage) = 13.85 := by
  sorry

end NUMINAMATH_CALUDE_chefs_wage_difference_l572_57271


namespace NUMINAMATH_CALUDE_airline_passengers_l572_57269

/-- Given an airline where each passenger can take 8 pieces of luggage,
    and a total of 32 bags, prove that 4 people were flying. -/
theorem airline_passengers (bags_per_person : ℕ) (total_bags : ℕ) (num_people : ℕ) : 
  bags_per_person = 8 →
  total_bags = 32 →
  num_people * bags_per_person = total_bags →
  num_people = 4 := by
sorry

end NUMINAMATH_CALUDE_airline_passengers_l572_57269


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l572_57249

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 3 = 0 → 
  x₂^2 - 4*x₂ - 3 = 0 → 
  x₁ + x₂ = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l572_57249


namespace NUMINAMATH_CALUDE_brick_wall_theorem_l572_57287

/-- Calculates the total number of bricks in a wall with a given number of rows,
    where each row has one less brick than the row below it. -/
def total_bricks (rows : ℕ) (bottom_row_bricks : ℕ) : ℕ :=
  (2 * bottom_row_bricks - rows + 1) * rows / 2

/-- Theorem stating that a wall with 5 rows, 18 bricks in the bottom row,
    and each row having one less brick than the row below it,
    has a total of 80 bricks. -/
theorem brick_wall_theorem :
  total_bricks 5 18 = 80 := by
  sorry

end NUMINAMATH_CALUDE_brick_wall_theorem_l572_57287


namespace NUMINAMATH_CALUDE_sum_of_solutions_l572_57275

theorem sum_of_solutions (x : ℝ) : 
  (18 * x^2 - 45 * x - 70 = 0) → 
  (∃ y : ℝ, 18 * y^2 - 45 * y - 70 = 0 ∧ x + y = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l572_57275


namespace NUMINAMATH_CALUDE_perpendicular_and_parallel_lines_planes_l572_57276

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def perp (l1 l2 : Line) : Prop := sorry
def para (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_and_parallel_lines_planes 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular m α) 
  (h2 : contained_in n β) :
  (parallel α β → perp m n) ∧ (para m n → perpendicular α β) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_and_parallel_lines_planes_l572_57276


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l572_57295

theorem hyperbola_focal_distance (P F₁ F₂ : ℝ × ℝ) :
  (∃ x y : ℝ, P = (x, y) ∧ x^2 / 64 - y^2 / 36 = 1) →  -- P is on the hyperbola
  (∃ c : ℝ, c > 0 ∧ F₁ = (-c, 0) ∧ F₂ = (c, 0)) →  -- F₁ and F₂ are foci
  ‖P - F₁‖ = 15 →  -- |PF₁| = 15
  ‖P - F₂‖ = 31 :=  -- |PF₂| = 31
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l572_57295


namespace NUMINAMATH_CALUDE_circle_equation_from_center_and_chord_l572_57216

/-- The equation of a circle given its center and a chord on a line. -/
theorem circle_equation_from_center_and_chord (x y : ℝ) :
  let center : ℝ × ℝ := (4, 7)
  let chord_length : ℝ := 8
  let line_eq : ℝ → ℝ → ℝ := fun x y => 3 * x - 4 * y + 1
  (∃ (a b : ℝ), (a - 4)^2 + (b - 7)^2 = 25 ∧ 
                line_eq a b = 0 ∧ 
                (a - center.1)^2 + (b - center.2)^2 = (chord_length / 2)^2) →
  (x - 4)^2 + (y - 7)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_from_center_and_chord_l572_57216


namespace NUMINAMATH_CALUDE_real_number_inequalities_l572_57266

-- Define the propositions
theorem real_number_inequalities (a b c : ℝ) : 
  -- Proposition A
  ((a * c^2 > b * c^2) → (a > b)) ∧ 
  -- Proposition B (negation)
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) ∧ 
  -- Proposition C (negation)
  (∃ a b : ℝ, a > b ∧ 1/a ≥ 1/b) ∧ 
  -- Proposition D
  ((a > b ∧ b > 0) → (a^2 > a*b ∧ a*b > b^2)) :=
by sorry

end NUMINAMATH_CALUDE_real_number_inequalities_l572_57266


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l572_57262

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2*I) / (Complex.mk a b) = 1 + I → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l572_57262


namespace NUMINAMATH_CALUDE_event_probability_l572_57213

noncomputable def probability_event (a b : Real) : Real :=
  (min b (3/2) - max a 0) / (b - a)

theorem event_probability : probability_event 0 2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_event_probability_l572_57213


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l572_57222

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

-- Define the property of a_1 and a_2 being roots of the equation
def roots_property (a : ℕ → ℝ) : Prop :=
  (a 1)^2 - (a 3) * (a 1) + (a 4) = 0 ∧
  (a 2)^2 - (a 3) * (a 2) + (a 4) = 0

-- Theorem statement
theorem arithmetic_sequence_theorem (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d → roots_property a → ∀ n : ℕ, a n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l572_57222


namespace NUMINAMATH_CALUDE_square_area_is_56_l572_57241

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 24 * x + 8 * y + 36

-- Define the property that the circle is inscribed in a square with sides parallel to axes
def inscribed_in_square (center_x center_y radius : ℝ) : Prop :=
  ∃ (side_length : ℝ), side_length = 2 * radius

-- Theorem statement
theorem square_area_is_56 :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    inscribed_in_square center_x center_y radius ∧
    4 * radius^2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_56_l572_57241


namespace NUMINAMATH_CALUDE_sufficient_unnecessary_condition_for_hyperbola_l572_57238

/-- The equation of a conic section -/
def conic_equation (k x y : ℝ) : Prop :=
  x^2 / (k - 2) + y^2 / (5 - k) = 1

/-- Condition for the equation to represent a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  k < 2 ∨ k > 5

/-- Statement that k < 1 is a sufficient and unnecessary condition for a hyperbola -/
theorem sufficient_unnecessary_condition_for_hyperbola :
  (∀ k, k < 1 → is_hyperbola k) ∧
  ∃ k, is_hyperbola k ∧ ¬(k < 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_unnecessary_condition_for_hyperbola_l572_57238


namespace NUMINAMATH_CALUDE_garden_area_increase_l572_57251

/-- Proves that adding 60 feet of fence to a rectangular garden of 80x20 feet
    to make it square increases the area by 2625 square feet. -/
theorem garden_area_increase : 
  ∀ (original_length original_width added_fence : ℕ),
    original_length = 80 →
    original_width = 20 →
    added_fence = 60 →
    let original_perimeter := 2 * (original_length + original_width)
    let new_perimeter := original_perimeter + added_fence
    let new_side := new_perimeter / 4
    let original_area := original_length * original_width
    let new_area := new_side * new_side
    new_area - original_area = 2625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_increase_l572_57251


namespace NUMINAMATH_CALUDE_inequality_solution_set_l572_57228

theorem inequality_solution_set (x : ℝ) : 
  (1 + x) * (2 - x) * (3 + x^2) > 0 ↔ -1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l572_57228


namespace NUMINAMATH_CALUDE_simplify_expression_l572_57208

theorem simplify_expression (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(1 + 3*x) - 5*(1 - x^2) = -5*x^2 + 17*x - 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l572_57208


namespace NUMINAMATH_CALUDE_perimeter_of_special_region_l572_57218

/-- The perimeter of a region bounded by four arcs, each being three-quarters of a circle
    constructed on the sides of a unit square, is equal to 3π. -/
theorem perimeter_of_special_region : Real := by
  -- Define the side length of the square
  let square_side : Real := 1

  -- Define the radius of each circle (half the side length)
  let circle_radius : Real := square_side / 2

  -- Define the length of a full circle with this radius
  let full_circle_length : Real := 2 * Real.pi * circle_radius

  -- Define the length of three-quarters of this circle
  let arc_length : Real := (3 / 4) * full_circle_length

  -- Define the perimeter as four times the arc length
  let perimeter : Real := 4 * arc_length

  -- Prove that this perimeter equals 3π
  sorry

end NUMINAMATH_CALUDE_perimeter_of_special_region_l572_57218


namespace NUMINAMATH_CALUDE_fraction_undefined_values_l572_57236

def undefined_values (a : ℝ) : Prop :=
  a^3 - 4*a = 0

theorem fraction_undefined_values :
  {a : ℝ | undefined_values a} = {-2, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_fraction_undefined_values_l572_57236


namespace NUMINAMATH_CALUDE_triangle_is_right_angle_l572_57283

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (angle_sum : A + B + C = π)

-- State the theorem
theorem triangle_is_right_angle (t : Triangle) 
  (h : (Real.cos (t.A / 2))^2 = (t.b + t.c) / (2 * t.c)) : 
  t.c^2 = t.a^2 + t.b^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angle_l572_57283


namespace NUMINAMATH_CALUDE_probability_at_least_one_diamond_or_ace_l572_57292

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of cards that are either diamonds or aces -/
def targetCards : ℕ := 16

/-- The probability of drawing a card that is neither a diamond nor an ace -/
def probNonTarget : ℚ := (deckSize - targetCards) / deckSize

/-- The number of draws -/
def numDraws : ℕ := 3

theorem probability_at_least_one_diamond_or_ace :
  1 - probNonTarget ^ numDraws = 1468 / 2197 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_diamond_or_ace_l572_57292


namespace NUMINAMATH_CALUDE_relay_arrangements_count_l572_57210

/-- Represents the number of people in the class -/
def class_size : ℕ := 5

/-- Represents the number of people needed for the relay -/
def relay_size : ℕ := 4

/-- Represents the number of options for the first runner -/
def first_runner_options : ℕ := 3

/-- Represents the number of options for the last runner -/
def last_runner_options : ℕ := 2

/-- Calculates the number of relay arrangements given the constraints -/
def relay_arrangements : ℕ := 24

/-- Theorem stating that the number of relay arrangements is 24 -/
theorem relay_arrangements_count : 
  relay_arrangements = 24 := by sorry

end NUMINAMATH_CALUDE_relay_arrangements_count_l572_57210


namespace NUMINAMATH_CALUDE_smallest_square_area_for_two_rectangles_l572_57231

/-- The smallest square area that can contain two non-overlapping rectangles -/
theorem smallest_square_area_for_two_rectangles :
  ∀ (w₁ h₁ w₂ h₂ : ℕ),
    w₁ = 2 ∧ h₁ = 4 ∧ w₂ = 3 ∧ h₂ = 5 →
    ∃ (s : ℕ),
      s^2 = 81 ∧
      ∀ (a : ℕ),
        (a ≥ w₁ ∧ a ≥ h₁ ∧ a ≥ w₂ ∧ a ≥ h₂ ∧ a ≥ w₁ + w₂ ∧ a ≥ h₁ + h₂) →
        a^2 ≥ s^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_area_for_two_rectangles_l572_57231


namespace NUMINAMATH_CALUDE_equation_solution_l572_57298

theorem equation_solution :
  ∃ x : ℝ, (x^2 + 3*x + 2) / (x^2 + 1) = x - 2 ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l572_57298


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_smallest_positive_integer_l572_57280

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 648 ∣ (450 * x) → x ≥ 36 := by
  sorry

theorem thirty_six_satisfies : 648 ∣ (450 * 36) := by
  sorry

theorem smallest_positive_integer : 
  ∃ (x : ℕ), x > 0 ∧ 648 ∣ (450 * x) ∧ ∀ (y : ℕ), y > 0 ∧ 648 ∣ (450 * y) → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_smallest_positive_integer_l572_57280


namespace NUMINAMATH_CALUDE_cuboid_volumes_sum_l572_57274

theorem cuboid_volumes_sum (length width height1 height2 : ℝ) 
  (h1 : length = 44)
  (h2 : width = 35)
  (h3 : height1 = 7)
  (h4 : height2 = 3) :
  length * width * height1 + length * width * height2 = 15400 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volumes_sum_l572_57274


namespace NUMINAMATH_CALUDE_smallest_with_20_divisors_l572_57289

def num_divisors (n : ℕ+) : ℕ := (Nat.divisors n.val).card

theorem smallest_with_20_divisors : 
  ∃ (n : ℕ+), num_divisors n = 20 ∧ ∀ (m : ℕ+), m < n → num_divisors m ≠ 20 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_with_20_divisors_l572_57289


namespace NUMINAMATH_CALUDE_probability_A_makes_basket_on_kth_shot_l572_57217

/-- The probability that player A takes k shots to make the basket -/
def P (k : ℕ) : ℝ :=
  (0.24 ^ (k - 1)) * 0.4

/-- Theorem stating the probability formula for player A making a basket on the k-th shot -/
theorem probability_A_makes_basket_on_kth_shot (k : ℕ) :
  P k = (0.24 ^ (k - 1)) * 0.4 :=
by
  sorry

#check probability_A_makes_basket_on_kth_shot

end NUMINAMATH_CALUDE_probability_A_makes_basket_on_kth_shot_l572_57217


namespace NUMINAMATH_CALUDE_second_die_has_seven_sides_l572_57277

/-- The number of sides on the first die -/
def first_die_sides : ℕ := 6

/-- The probability of rolling a sum of 13 with both dice -/
def prob_sum_13 : ℚ := 23809523809523808 / 1000000000000000000

/-- The number of sides on the second die -/
def second_die_sides : ℕ := sorry

theorem second_die_has_seven_sides :
  (1 : ℚ) / (first_die_sides * second_die_sides) = prob_sum_13 ∧ 
  second_die_sides ≥ 7 →
  second_die_sides = 7 := by sorry

end NUMINAMATH_CALUDE_second_die_has_seven_sides_l572_57277


namespace NUMINAMATH_CALUDE_rational_solutions_count_l572_57294

theorem rational_solutions_count (p : ℕ) (hp : Prime p) :
  let f : ℚ → ℚ := λ x => x^4 + (2 - p : ℚ)*x^3 + (2 - 2*p : ℚ)*x^2 + (1 - 2*p : ℚ)*x - p
  (∃ (s : Finset ℚ), s.card = 2 ∧ (∀ x ∈ s, f x = 0) ∧ (∀ x, f x = 0 → x ∈ s)) := by
  sorry

end NUMINAMATH_CALUDE_rational_solutions_count_l572_57294


namespace NUMINAMATH_CALUDE_part_one_part_two_l572_57206

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

-- Part (1)
theorem part_one (m n : ℝ) :
  (∀ x, f m x < 0 ↔ -2 < x ∧ x < n) →
  m = 3/2 ∧ n = 1/2 := by sorry

-- Part (2)
theorem part_two (m : ℝ) :
  (∀ x ∈ Set.Icc m (m+1), f m x < 0) →
  m > -Real.sqrt 2 / 2 ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l572_57206


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l572_57278

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 2 →
  a 3 * a 5 = 4 * (a 6)^2 →
  a 3 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l572_57278


namespace NUMINAMATH_CALUDE_joan_sold_26_books_l572_57254

/-- The number of books Joan sold in the yard sale -/
def books_sold (initial_books remaining_books : ℕ) : ℕ :=
  initial_books - remaining_books

/-- Theorem: Joan sold 26 books in the yard sale -/
theorem joan_sold_26_books :
  books_sold 33 7 = 26 := by
  sorry

end NUMINAMATH_CALUDE_joan_sold_26_books_l572_57254


namespace NUMINAMATH_CALUDE_vectors_parallel_iff_y_eq_neg_one_l572_57272

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

/-- Vector a -/
def a : ℝ × ℝ := (1, 2)

/-- Vector b parameterized by y -/
def b (y : ℝ) : ℝ × ℝ := (1, -2*y)

/-- Theorem: Vectors a and b are parallel if and only if y = -1 -/
theorem vectors_parallel_iff_y_eq_neg_one :
  ∀ y : ℝ, are_parallel a (b y) ↔ y = -1 := by sorry

end NUMINAMATH_CALUDE_vectors_parallel_iff_y_eq_neg_one_l572_57272


namespace NUMINAMATH_CALUDE_triple_composition_even_l572_57297

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem triple_composition_even (g : ℝ → ℝ) (h : IsEven g) : IsEven (fun x ↦ g (g (g x))) := by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l572_57297


namespace NUMINAMATH_CALUDE_f_minimum_value_l572_57201

open Real

noncomputable def f (x : ℝ) : ℝ := (3 * sin x - 4 * cos x - 10) * (3 * sin x + 4 * cos x - 10)

theorem f_minimum_value :
  ∃ (min : ℝ), (∀ (x : ℝ), f x ≥ min) ∧ (min = 25 / 9 - 10 - 80 * Real.sqrt 2 / 3 - 116) := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l572_57201


namespace NUMINAMATH_CALUDE_problem_polygon_area_l572_57230

/-- A point in a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- A polygon defined by a list of grid points --/
def Polygon := List GridPoint

/-- The polygon described in the problem --/
def problemPolygon : Polygon := [
  ⟨0, 0⟩, ⟨10, 0⟩, ⟨20, 0⟩, ⟨30, 10⟩, ⟨20, 30⟩,
  ⟨10, 30⟩, ⟨0, 30⟩, ⟨0, 20⟩, ⟨10, 20⟩, ⟨10, 10⟩
]

/-- Calculate the area of a polygon given its vertices --/
def calculatePolygonArea (p : Polygon) : Int :=
  sorry

theorem problem_polygon_area :
  calculatePolygonArea problemPolygon = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l572_57230


namespace NUMINAMATH_CALUDE_complex_distance_range_l572_57252

theorem complex_distance_range (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min max : ℝ), min = 3 ∧ max = 5 ∧
  (∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 →
    min ≤ Complex.abs (w - 2 - 2*I) ∧ Complex.abs (w - 2 - 2*I) ≤ max) :=
by sorry

end NUMINAMATH_CALUDE_complex_distance_range_l572_57252


namespace NUMINAMATH_CALUDE_det_equation_solution_l572_57264

/-- Definition of 2nd order determinant -/
def det2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem: If |x+1 1-x; 1-x x+1| = 8, then x = 2 -/
theorem det_equation_solution (x : ℝ) : 
  det2 (x + 1) (1 - x) (1 - x) (x + 1) = 8 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_equation_solution_l572_57264


namespace NUMINAMATH_CALUDE_exists_consecutive_numbers_with_properties_l572_57209

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of two consecutive numbers with given properties -/
theorem exists_consecutive_numbers_with_properties :
  ∃ n : ℕ, sum_of_digits n = 8 ∧ (n + 1) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_consecutive_numbers_with_properties_l572_57209


namespace NUMINAMATH_CALUDE_shaded_area_concentric_circles_l572_57219

theorem shaded_area_concentric_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 6) (h₂ : r₂ = 3) :
  let area_triangles := 4 * (1/2 * r₂ * r₂)
  let area_small_sectors := 4 * (1/4 * Real.pi * r₂^2)
  area_triangles + area_small_sectors = 18 + 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_concentric_circles_l572_57219


namespace NUMINAMATH_CALUDE_pond_animals_l572_57285

/-- Given a pond with snails and frogs, calculate the total number of animals -/
theorem pond_animals (num_snails num_frogs : ℕ) : num_snails = 5 → num_frogs = 2 → num_snails + num_frogs = 7 := by
  sorry

end NUMINAMATH_CALUDE_pond_animals_l572_57285


namespace NUMINAMATH_CALUDE_area_trapezoid_EFBA_l572_57299

/-- Rectangle ABCD with points E and F on side DC -/
structure RectangleWithPoints where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Length of segment DE -/
  DE : ℝ
  /-- Length of segment FC -/
  FC : ℝ
  /-- Area of rectangle ABCD -/
  area_ABCD : ℝ
  /-- AB is positive -/
  AB_pos : AB > 0
  /-- BC is positive -/
  BC_pos : BC > 0
  /-- DE is positive -/
  DE_pos : DE > 0
  /-- FC is positive -/
  FC_pos : FC > 0
  /-- Area of ABCD is product of AB and BC -/
  area_eq : area_ABCD = AB * BC
  /-- DE + EF + FC = DC = AB -/
  side_sum : DE + (AB - DE - FC) + FC = AB

/-- The area of trapezoid EFBA is 14 square units -/
theorem area_trapezoid_EFBA (r : RectangleWithPoints) (h1 : r.AB = 10) (h2 : r.BC = 2) 
    (h3 : r.DE = 2) (h4 : r.FC = 4) (h5 : r.area_ABCD = 20) : 
    r.AB * r.BC - r.DE * r.BC / 2 - r.FC * r.BC / 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_area_trapezoid_EFBA_l572_57299


namespace NUMINAMATH_CALUDE_solve_equation_l572_57270

theorem solve_equation : ∃ x : ℚ, (3/4 : ℚ) - (1/2 : ℚ) = 1/x ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l572_57270


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l572_57225

theorem simplify_trig_expression :
  Real.sqrt (1 - 2 * Real.sin (Real.pi + 4) * Real.cos (Real.pi + 4)) = Real.cos 4 - Real.sin 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l572_57225


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l572_57273

theorem min_distance_between_curves (a b c d : ℝ) 
  (h1 : (a + 3 * Real.log a) / b = 1)
  (h2 : (d - 3) / (2 * c) = 1) :
  (∀ x y z w : ℝ, (x + 3 * Real.log x) / y = 1 → (w - 3) / (2 * z) = 1 → 
    (a - c)^2 + (b - d)^2 ≤ (x - z)^2 + (y - w)^2) ∧
  (a - c)^2 + (b - d)^2 = 9/5 * Real.log (9/Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l572_57273


namespace NUMINAMATH_CALUDE_second_smallest_five_digit_pascal_correct_l572_57244

/-- Binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Predicate to check if a number is five digits -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

/-- Predicate to check if a number appears in Pascal's triangle -/
def in_pascal_triangle (n : ℕ) : Prop := ∃ (row col : ℕ), binomial row col = n

/-- The second smallest five-digit number in Pascal's triangle -/
def second_smallest_five_digit_pascal : ℕ := 31465

theorem second_smallest_five_digit_pascal_correct :
  is_five_digit second_smallest_five_digit_pascal ∧
  in_pascal_triangle second_smallest_five_digit_pascal ∧
  ∃ (m : ℕ), is_five_digit m ∧ 
             in_pascal_triangle m ∧ 
             m < second_smallest_five_digit_pascal ∧
             ∀ (k : ℕ), is_five_digit k ∧ 
                        in_pascal_triangle k ∧ 
                        k ≠ m → 
                        second_smallest_five_digit_pascal ≤ k :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_five_digit_pascal_correct_l572_57244


namespace NUMINAMATH_CALUDE_pure_imaginary_square_l572_57226

theorem pure_imaginary_square (a : ℝ) : 
  (∃ b : ℝ, (1 + a * Complex.I)^2 = b * Complex.I) → (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_l572_57226


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l572_57221

theorem power_zero_eq_one (a : ℝ) (h : a ≠ 0) : a ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l572_57221


namespace NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l572_57203

/-- Given an equilateral triangle with two vertices at (0,3) and (10,3),
    prove that the y-coordinate of the third vertex in the first quadrant is 3 + 5√3. -/
theorem equilateral_triangle_third_vertex_y_coord :
  let v1 : ℝ × ℝ := (0, 3)
  let v2 : ℝ × ℝ := (10, 3)
  let side_length : ℝ := 10
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 3 ∧
    (x - 0)^2 + (y - 3)^2 = side_length^2 ∧
    (x - 10)^2 + (y - 3)^2 = side_length^2 ∧
    y = 3 + 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l572_57203


namespace NUMINAMATH_CALUDE_audrey_needs_eight_limes_l572_57233

/-- The number of tablespoons in a cup -/
def tablespoons_per_cup : ℚ := 16

/-- The amount of key lime juice in the original recipe, in cups -/
def original_recipe_juice : ℚ := 1/4

/-- The factor by which Audrey increases the amount of juice -/
def juice_increase_factor : ℚ := 2

/-- The amount of juice one key lime yields, in tablespoons -/
def juice_per_lime : ℚ := 1

/-- Calculates the number of key limes Audrey needs for her pie -/
def key_limes_needed : ℚ :=
  (original_recipe_juice * juice_increase_factor * tablespoons_per_cup) / juice_per_lime

/-- Theorem stating that Audrey needs 8 key limes for her pie -/
theorem audrey_needs_eight_limes : key_limes_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_audrey_needs_eight_limes_l572_57233


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l572_57229

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_uniqueness 
  (f : ℝ → ℝ) 
  (h_quad : is_quadratic f)
  (h_sol : ∀ x : ℝ, f x > 0 ↔ 0 < x ∧ x < 4)
  (h_max : ∀ x : ℝ, x ∈ Set.Icc (-1) 5 → f x ≤ 12)
  (h_attain : ∃ x : ℝ, x ∈ Set.Icc (-1) 5 ∧ f x = 12) :
  ∀ x : ℝ, f x = -3 * x^2 + 12 * x :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l572_57229


namespace NUMINAMATH_CALUDE_sqrt_two_inequality_l572_57250

theorem sqrt_two_inequality (m n : ℕ) (h : (m : ℝ) / n < Real.sqrt 2) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * n^2)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_inequality_l572_57250


namespace NUMINAMATH_CALUDE_correct_purchase_and_savings_l572_57207

/-- Represents the purchase of notebooks by a school -/
structure NotebookPurchase where
  type1 : ℕ  -- number of notebooks of first type
  type2 : ℕ  -- number of notebooks of second type

/-- Calculates the total cost of notebooks without discount -/
def totalCost (purchase : NotebookPurchase) : ℕ :=
  3 * purchase.type1 + 2 * purchase.type2

/-- Calculates the discounted cost of notebooks -/
def discountedCost (purchase : NotebookPurchase) : ℚ :=
  3 * purchase.type1 * (8/10) + 2 * purchase.type2 * (9/10)

/-- Theorem stating the correct purchase and savings -/
theorem correct_purchase_and_savings :
  ∃ (purchase : NotebookPurchase),
    totalCost purchase = 460 ∧
    purchase.type1 = 2 * purchase.type2 + 20 ∧
    purchase.type1 = 120 ∧
    purchase.type2 = 50 ∧
    460 - discountedCost purchase = 82 := by
  sorry


end NUMINAMATH_CALUDE_correct_purchase_and_savings_l572_57207


namespace NUMINAMATH_CALUDE_expression_equality_l572_57257

theorem expression_equality : 
  |1 - Real.sqrt 3| + 3 * Real.tan (30 * π / 180) - (1/2)⁻¹ + (3 - π)^0 = 3.732 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l572_57257


namespace NUMINAMATH_CALUDE_bacteria_growth_proof_l572_57281

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The doubling time of the bacteria population in seconds -/
def doubling_time : ℕ := 30

/-- The total time of the experiment in minutes -/
def total_time : ℕ := 4

/-- The final number of bacteria after the experiment -/
def final_bacteria_count : ℕ := 524288

/-- The initial number of bacteria -/
def initial_bacteria_count : ℕ := 2048

theorem bacteria_growth_proof :
  initial_bacteria_count * 2^(total_time * seconds_per_minute / doubling_time) = final_bacteria_count :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_proof_l572_57281


namespace NUMINAMATH_CALUDE_semicircle_square_properties_l572_57288

-- Define the semicircle and inscribed square
def semicircle_with_square (a b : ℝ) : Prop :=
  ∃ (A B C D E F : ℝ × ℝ),
    -- A and B are endpoints of the diameter
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * ((C.1 - D.1)^2 + (C.2 - D.2)^2) ∧
    -- CDEF is a square with side length 1
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = 1 ∧
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = 1 ∧
    (E.1 - F.1)^2 + (E.2 - F.2)^2 = 1 ∧
    (F.1 - C.1)^2 + (F.2 - C.2)^2 = 1 ∧
    -- AC = a and BC = b
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = a^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = b^2

-- State the theorem
theorem semicircle_square_properties (a b : ℝ) (h : semicircle_with_square a b) :
  a - b = 1 ∧ a * b = 1 ∧ a + b = Real.sqrt 5 ∧ a^2 + b^2 ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_square_properties_l572_57288


namespace NUMINAMATH_CALUDE_probability_sum_14_correct_l572_57232

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Nat := 52

/-- Represents the number of cards with values 2 through 10 in a standard deck -/
def NumberCards : Nat := 36

/-- Represents the number of pairs of number cards that sum to 14 -/
def PairsSummingTo14 : Nat := 76

/-- The probability of selecting two number cards that sum to 14 from a standard deck -/
def probability_sum_14 : ℚ := 19 / 663

theorem probability_sum_14_correct : 
  (PairsSummingTo14 : ℚ) / (StandardDeck * (StandardDeck - 1)) = probability_sum_14 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_14_correct_l572_57232


namespace NUMINAMATH_CALUDE_max_vector_sum_on_unit_circle_l572_57284

theorem max_vector_sum_on_unit_circle :
  let A : ℝ × ℝ := (Real.sqrt 3, 1)
  let O : ℝ × ℝ := (0, 0)
  ∃ (max : ℝ), max = 3 ∧ 
    ∀ (B : ℝ × ℝ), (B.1 - O.1)^2 + (B.2 - O.2)^2 = 1 →
      Real.sqrt ((A.1 - O.1 + B.1 - O.1)^2 + (A.2 - O.2 + B.2 - O.2)^2) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_vector_sum_on_unit_circle_l572_57284


namespace NUMINAMATH_CALUDE_probability_of_b_in_rabbit_l572_57282

def word : String := "rabbit"

def count_letter (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem probability_of_b_in_rabbit :
  (count_letter word 'b' : ℚ) / word.length = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_b_in_rabbit_l572_57282


namespace NUMINAMATH_CALUDE_sample_xy_product_l572_57290

theorem sample_xy_product (x y : ℝ) : 
  (9 + 10 + 11 + x + y) / 5 = 10 →
  ((9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2) / 5 = 2 →
  x * y = 96 := by
sorry

end NUMINAMATH_CALUDE_sample_xy_product_l572_57290


namespace NUMINAMATH_CALUDE_sum_square_units_digits_2023_l572_57227

def first_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def square_units_digit (n : ℕ) : ℕ :=
  (n ^ 2) % 10

def sum_square_units_digits (n : ℕ) : ℕ :=
  (first_odd_integers n).map square_units_digit |> List.sum

theorem sum_square_units_digits_2023 :
  sum_square_units_digits 2023 % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_square_units_digits_2023_l572_57227


namespace NUMINAMATH_CALUDE_correct_total_carrots_l572_57259

/-- The total number of carrots Bianca has after picking, throwing out, and picking again -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

/-- Theorem stating that the total number of carrots is correct -/
theorem correct_total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ)
  (h1 : initial ≥ thrown_out) :
  total_carrots initial thrown_out picked_next_day = initial - thrown_out + picked_next_day :=
by
  sorry

#eval total_carrots 23 10 47  -- Should evaluate to 60

end NUMINAMATH_CALUDE_correct_total_carrots_l572_57259


namespace NUMINAMATH_CALUDE_only_group_d_forms_triangle_l572_57234

/-- A group of three sticks --/
structure StickGroup where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a group of sticks can form a triangle --/
def canFormTriangle (g : StickGroup) : Prop :=
  g.a + g.b > g.c ∧ g.b + g.c > g.a ∧ g.c + g.a > g.b

/-- The given groups of sticks --/
def groupA : StickGroup := ⟨1, 2, 6⟩
def groupB : StickGroup := ⟨2, 2, 4⟩
def groupC : StickGroup := ⟨1, 2, 3⟩
def groupD : StickGroup := ⟨2, 3, 4⟩

/-- Theorem: Only group D can form a triangle --/
theorem only_group_d_forms_triangle :
  ¬(canFormTriangle groupA) ∧
  ¬(canFormTriangle groupB) ∧
  ¬(canFormTriangle groupC) ∧
  canFormTriangle groupD :=
sorry

end NUMINAMATH_CALUDE_only_group_d_forms_triangle_l572_57234


namespace NUMINAMATH_CALUDE_parabola_directrix_l572_57296

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix_equation (y : ℝ) : Prop :=
  y = -1/2

/-- Theorem: The directrix of the given parabola is y = -1/2 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_directrix : ℝ, directrix_equation y_directrix :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l572_57296


namespace NUMINAMATH_CALUDE_total_amount_proof_l572_57214

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def totalAmount (n50 : ℕ) (n500 : ℕ) : ℕ := n50 * 50 + n500 * 500

/-- Proves that the total amount of money is 10350 rupees given the specified conditions -/
theorem total_amount_proof :
  let total_notes : ℕ := 108
  let n50 : ℕ := 97
  let n500 : ℕ := total_notes - n50
  totalAmount n50 n500 = 10350 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_proof_l572_57214


namespace NUMINAMATH_CALUDE_puzzle_border_pieces_l572_57279

theorem puzzle_border_pieces (total_pieces : ℕ) (trevor_pieces : ℕ) (missing_pieces : ℕ) : 
  total_pieces = 500 → 
  trevor_pieces = 105 → 
  missing_pieces = 5 → 
  (total_pieces - missing_pieces - trevor_pieces - 3 * trevor_pieces) = 75 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_border_pieces_l572_57279


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l572_57286

theorem geometric_sequence_solution (x : ℝ) :
  (1 < x ∧ x < 9 ∧ x^2 = 9) ↔ (x = 3 ∨ x = -3) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l572_57286


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l572_57255

theorem smallest_integer_with_remainders : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > 0 ∧ m % 6 = 2 ∧ m % 8 = 3 → n ≤ m) ∧
  n > 0 ∧ n % 6 = 2 ∧ n % 8 = 3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l572_57255


namespace NUMINAMATH_CALUDE_ashley_champagne_bottles_l572_57247

/-- The number of bottles of champagne needed for a wedding toast --/
def bottles_needed (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) : ℕ :=
  (guests * glasses_per_guest) / servings_per_bottle

/-- Theorem: Ashley needs 40 bottles of champagne for her wedding toast --/
theorem ashley_champagne_bottles : 
  bottles_needed 120 2 6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ashley_champagne_bottles_l572_57247


namespace NUMINAMATH_CALUDE_special_function_properties_l572_57267

/-- A function satisfying specific properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, ∀ y > 0, f (x * y) = f x + f y) ∧
  (∀ x > 1, f x < 0) ∧
  (f 3 = -1)

theorem special_function_properties
  (f : ℝ → ℝ)
  (hf : SpecialFunction f) :
  f 1 = 0 ∧
  f (1/9) = 2 ∧
  (∀ x y, x > 0 → y > 0 → x < y → f y < f x) ∧
  (∀ x, f x + f (2 - x) < 2 ↔ 1 - 2 * Real.sqrt 2 / 3 < x ∧ x < 1 + 2 * Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l572_57267


namespace NUMINAMATH_CALUDE_prime_divides_29_power_plus_one_l572_57215

theorem prime_divides_29_power_plus_one (p : ℕ) : 
  Nat.Prime p ∧ p ∣ 29^p + 1 ↔ p = 2 ∨ p = 3 ∨ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_29_power_plus_one_l572_57215


namespace NUMINAMATH_CALUDE_sum_of_vertices_l572_57265

/-- A configuration of numbers on a triangle -/
structure TriangleConfig where
  vertices : Fin 3 → ℕ
  sides : Fin 3 → ℕ
  sum_property : ∀ i : Fin 3, vertices i + sides i + vertices (i + 1) = 17

/-- The set of numbers to be used in the triangle -/
def triangle_numbers : Finset ℕ := {1, 3, 5, 7, 9, 11}

/-- The theorem stating the sum of numbers at the vertices -/
theorem sum_of_vertices (config : TriangleConfig) 
  (h : ∀ n, n ∈ (Finset.image config.vertices Finset.univ ∪ Finset.image config.sides Finset.univ) → n ∈ triangle_numbers) :
  config.vertices 0 + config.vertices 1 + config.vertices 2 = 15 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_vertices_l572_57265


namespace NUMINAMATH_CALUDE_car_speed_problem_l572_57263

/-- Proves that given a car traveling for two hours with an average speed of 40 km/h,
    and a speed of 60 km/h in the second hour, the speed in the first hour must be 20 km/h. -/
theorem car_speed_problem (speed_first_hour : ℝ) (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 60 →
  average_speed = 40 →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_first_hour = 20 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l572_57263


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l572_57245

-- Define the inverse variation relationship
def inverse_variation (y z : ℝ) : Prop := ∃ k : ℝ, y^2 * Real.sqrt z = k

-- Define the theorem
theorem inverse_variation_problem (y₁ y₂ z₁ z₂ : ℝ) 
  (h1 : inverse_variation y₁ z₁)
  (h2 : y₁ = 3)
  (h3 : z₁ = 4)
  (h4 : y₂ = 6) :
  z₂ = 1/4 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l572_57245


namespace NUMINAMATH_CALUDE_polynomial_roots_interlace_l572_57212

theorem polynomial_roots_interlace (p₁ p₂ q₁ q₂ : ℝ) 
  (h : (q₁ - q₂)^2 + (p₁ - p₂)*(p₁*q₂ - p₂*q₁) < 0) :
  let f := fun x : ℝ => x^2 + p₁*x + q₁
  let g := fun x : ℝ => x^2 + p₂*x + q₂
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ < y₂ ∧ g y₁ = 0 ∧ g y₂ = 0) ∧
  (∃ x y : ℝ, (f x = 0 ∧ y₁ < x ∧ x < y₂) ∧ (g y = 0 ∧ x₁ < y ∧ y < x₂)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_interlace_l572_57212


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l572_57242

/-- Given a circle with equation x^2 + y^2 - 2x + 4y + 3 = 0, 
    its center is at (1, -2) and its radius is √2 -/
theorem circle_center_and_radius :
  let f : ℝ × ℝ → ℝ := λ (x, y) => x^2 + y^2 - 2*x + 4*y + 3
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧ 
    radius = Real.sqrt 2 ∧
    ∀ (p : ℝ × ℝ), f p = 0 ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l572_57242


namespace NUMINAMATH_CALUDE_triangle_area_from_circle_and_chord_data_l572_57248

/-- Given a circle and a triangle circumscribed around it, this theorem proves
    the area of the triangle based on given measurements. -/
theorem triangle_area_from_circle_and_chord_data (R : ℝ) (chord_length : ℝ) (center_to_chord : ℝ) (perimeter : ℝ)
  (h1 : chord_length = 16)
  (h2 : center_to_chord = 15)
  (h3 : perimeter = 200)
  (h4 : R^2 = center_to_chord^2 + (chord_length/2)^2) :
  R * (perimeter / 2) = 1700 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_circle_and_chord_data_l572_57248


namespace NUMINAMATH_CALUDE_exponent_calculation_l572_57220

theorem exponent_calculation : ((15^15 / 15^14)^3 * 3^3) / 3^3 = 3375 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l572_57220


namespace NUMINAMATH_CALUDE_units_digit_of_special_three_digit_number_l572_57205

/-- The product of digits of a three-digit number -/
def P (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

/-- The sum of digits of a three-digit number -/
def S (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- A three-digit number is between 100 and 999 -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem units_digit_of_special_three_digit_number (N : ℕ) 
  (h1 : is_three_digit N) 
  (h2 : N = P N + S N) : 
  N % 10 = 9 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_special_three_digit_number_l572_57205


namespace NUMINAMATH_CALUDE_alyssa_toy_cost_l572_57200

/-- Calculates the total cost of toys with various discounts and special offers -/
def total_cost (football_price marbles_price puzzle_price toy_car_price board_game_price 
                stuffed_animal_price action_figure_price : ℝ) : ℝ :=
  let marbles_discounted := marbles_price * (1 - 0.05)
  let puzzle_discounted := puzzle_price * (1 - 0.10)
  let toy_car_discounted := toy_car_price * (1 - 0.15)
  let stuffed_animals_total := stuffed_animal_price * 1.5
  let action_figures_total := action_figure_price * (1 + 0.4)
  football_price + marbles_discounted + puzzle_discounted + toy_car_discounted + 
  board_game_price + stuffed_animals_total + action_figures_total

/-- Theorem stating the total cost of Alyssa's toys -/
theorem alyssa_toy_cost : 
  total_cost 5.71 6.59 4.25 3.95 10.49 8.99 12.39 = 60.468 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_toy_cost_l572_57200


namespace NUMINAMATH_CALUDE_find_c_and_d_l572_57202

/-- Definition of the polynomial g(x) -/
def g (c d x : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

/-- Theorem stating the conditions and the result to be proved -/
theorem find_c_and_d :
  ∀ c d : ℝ,
  g c d 2 = -7 →
  g c d (-1) = -25 →
  c = 2 ∧ d = 8 := by
sorry

end NUMINAMATH_CALUDE_find_c_and_d_l572_57202


namespace NUMINAMATH_CALUDE_complex_modulus_l572_57260

theorem complex_modulus (z : ℂ) : z = (1 + Complex.I) / (2 - Complex.I) → Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l572_57260


namespace NUMINAMATH_CALUDE_basketball_weight_l572_57240

/-- Given that eight identical basketballs weigh the same as four identical watermelons,
    and one watermelon weighs 32 pounds, prove that one basketball weighs 16 pounds. -/
theorem basketball_weight (watermelon_weight : ℝ) (basketball_weight : ℝ) : 
  watermelon_weight = 32 →
  8 * basketball_weight = 4 * watermelon_weight →
  basketball_weight = 16 := by
sorry

end NUMINAMATH_CALUDE_basketball_weight_l572_57240


namespace NUMINAMATH_CALUDE_book_sorting_terminates_and_sorts_width_l572_57204

/-- Represents a book with height and width -/
structure Book where
  height : ℕ
  width : ℕ

/-- The state of the bookshelf -/
structure BookshelfState where
  books : List Book
  n : ℕ

/-- Predicate to check if books are sorted by increasing width -/
def sortedByWidth (state : BookshelfState) : Prop :=
  ∀ i j, i < j → i < state.n → j < state.n →
    (state.books.get ⟨i, by sorry⟩).width < (state.books.get ⟨j, by sorry⟩).width

/-- Predicate to check if a swap is valid -/
def canSwap (state : BookshelfState) (i : ℕ) : Prop :=
  i + 1 < state.n ∧
  (state.books.get ⟨i, by sorry⟩).width > (state.books.get ⟨i + 1, by sorry⟩).width ∧
  (state.books.get ⟨i, by sorry⟩).height < (state.books.get ⟨i + 1, by sorry⟩).height

/-- The main theorem -/
theorem book_sorting_terminates_and_sorts_width
  (initial : BookshelfState)
  (h_n : initial.n ≥ 2)
  (h_unique : ∀ i j, i ≠ j → i < initial.n → j < initial.n →
    (initial.books.get ⟨i, by sorry⟩).height ≠ (initial.books.get ⟨j, by sorry⟩).height ∧
    (initial.books.get ⟨i, by sorry⟩).width ≠ (initial.books.get ⟨j, by sorry⟩).width)
  (h_initial_height : ∀ i j, i < j → i < initial.n → j < initial.n →
    (initial.books.get ⟨i, by sorry⟩).height < (initial.books.get ⟨j, by sorry⟩).height) :
  ∃ (final : BookshelfState),
    (∀ i, ¬canSwap final i) ∧
    sortedByWidth final :=
by sorry

end NUMINAMATH_CALUDE_book_sorting_terminates_and_sorts_width_l572_57204


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l572_57237

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x | x^2 - 5*x + 4 < 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l572_57237


namespace NUMINAMATH_CALUDE_even_increasing_function_inequality_l572_57243

-- Define an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define an increasing function on [0,+∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem even_increasing_function_inequality (f : ℝ → ℝ) (k : ℝ) 
  (h_even : even_function f) 
  (h_increasing : increasing_on_nonneg f) 
  (h_inequality : f k > f 2) : 
  k > 2 ∨ k < -2 :=
sorry

end NUMINAMATH_CALUDE_even_increasing_function_inequality_l572_57243


namespace NUMINAMATH_CALUDE_quadratic_minimum_l572_57224

theorem quadratic_minimum (x : ℝ) :
  let y := 4 * x^2 + 8 * x + 16
  ∀ x', 4 * x'^2 + 8 * x' + 16 ≥ 12 ∧ (4 * (-1)^2 + 8 * (-1) + 16 = 12) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l572_57224


namespace NUMINAMATH_CALUDE_find_first_fraction_l572_57261

def compound_ratio : ℚ := 0.07142857142857142
def second_fraction : ℚ := 1/3
def third_fraction : ℚ := 3/8

theorem find_first_fraction :
  ∃ (first_fraction : ℚ), first_fraction * second_fraction * third_fraction = compound_ratio :=
sorry

end NUMINAMATH_CALUDE_find_first_fraction_l572_57261


namespace NUMINAMATH_CALUDE_correct_bird_count_l572_57291

/-- Given a number of feet on tree branches and the number of feet per bird,
    calculate the number of birds on the tree. -/
def birds_on_tree (total_feet : ℕ) (feet_per_bird : ℕ) : ℕ :=
  total_feet / feet_per_bird

theorem correct_bird_count : birds_on_tree 92 2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_correct_bird_count_l572_57291


namespace NUMINAMATH_CALUDE_dog_drying_ratio_l572_57239

/-- The time (in minutes) it takes to dry a short-haired dog -/
def short_hair_time : ℕ := 10

/-- The number of short-haired dogs -/
def num_short_hair : ℕ := 6

/-- The number of full-haired dogs -/
def num_full_hair : ℕ := 9

/-- The total time (in minutes) it takes to dry all dogs -/
def total_time : ℕ := 240

/-- The ratio of time to dry a full-haired dog to a short-haired dog -/
def drying_ratio : ℚ := 2

theorem dog_drying_ratio :
  ∃ (full_hair_time : ℕ),
    full_hair_time = short_hair_time * (drying_ratio.num / drying_ratio.den) ∧
    num_short_hair * short_hair_time + num_full_hair * full_hair_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_dog_drying_ratio_l572_57239
