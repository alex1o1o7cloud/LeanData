import Mathlib

namespace NUMINAMATH_CALUDE_base4_arithmetic_theorem_l3404_340472

/-- Converts a number from base 4 to base 10 -/
def base4To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10To4 (n : ℕ) : ℕ := sorry

/-- Performs arithmetic operations in base 4 -/
def base4Arithmetic (a b c d : ℕ) : ℕ := 
  let a10 := base4To10 a
  let b10 := base4To10 b
  let c10 := base4To10 c
  let d10 := base4To10 d
  base10To4 (a10 + b10 * c10 / d10)

theorem base4_arithmetic_theorem : 
  base4Arithmetic 231 21 12 3 = 333 := by sorry

end NUMINAMATH_CALUDE_base4_arithmetic_theorem_l3404_340472


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3404_340421

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - I) / (1 - 3*I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3404_340421


namespace NUMINAMATH_CALUDE_periodic_even_symmetric_function_l3404_340488

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is symmetric about the line x = a if f(a - x) = f(a + x) for all x -/
def IsSymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

/-- A function f is periodic with period p if f(x + p) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem periodic_even_symmetric_function (f : ℝ → ℝ) 
  (h_nonconstant : ∃ x y, f x ≠ f y)
  (h_even : IsEven f)
  (h_symmetric : IsSymmetricAbout f (Real.sqrt 2 / 2)) :
  IsPeriodic f (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_periodic_even_symmetric_function_l3404_340488


namespace NUMINAMATH_CALUDE_combined_fuel_efficiency_l3404_340456

/-- Calculates the combined fuel efficiency of two cars -/
theorem combined_fuel_efficiency
  (efficiency1 : ℝ) -- Fuel efficiency of the first car in miles per gallon
  (efficiency2 : ℝ) -- Fuel efficiency of the second car in miles per gallon
  (h1 : efficiency1 = 40) -- Given: Ray's car averages 40 miles per gallon
  (h2 : efficiency2 = 10) -- Given: Tom's car averages 10 miles per gallon
  (distance : ℝ) -- Distance driven by each car
  (h3 : distance > 0) -- Assumption: Distance driven is positive
  : (2 * distance) / ((distance / efficiency1) + (distance / efficiency2)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_combined_fuel_efficiency_l3404_340456


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l3404_340494

/-- A parabola y = ax^2 + 4x + 3 is tangent to the line y = 2x + 1 if and only if a = 1/2 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃ x : ℝ, ax^2 + 4*x + 3 = 2*x + 1 ∧ 
   ∀ y : ℝ, y ≠ x → ax^2 + 4*x + 3 ≠ 2*y + 1) ↔ 
  a = (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l3404_340494


namespace NUMINAMATH_CALUDE_bush_current_age_l3404_340410

def matt_future_age : ℕ := 25
def years_to_future : ℕ := 10
def age_difference : ℕ := 3

theorem bush_current_age : 
  matt_future_age - years_to_future - age_difference = 12 := by
  sorry

end NUMINAMATH_CALUDE_bush_current_age_l3404_340410


namespace NUMINAMATH_CALUDE_rectangle_height_double_area_square_side_double_area_cube_side_double_volume_rectangle_half_width_triple_height_rectangle_double_length_triple_width_geometric_transformations_l3404_340437

-- Define geometric shapes
def Rectangle (w h : ℝ) := w * h
def Square (s : ℝ) := s * s
def Cube (s : ℝ) := s * s * s

-- Theorem for statement (A)
theorem rectangle_height_double_area (w h : ℝ) :
  Rectangle w (2 * h) = 2 * Rectangle w h := by sorry

-- Theorem for statement (B)
theorem square_side_double_area (s : ℝ) :
  Square (2 * s) = 4 * Square s := by sorry

-- Theorem for statement (C)
theorem cube_side_double_volume (s : ℝ) :
  Cube (2 * s) = 8 * Cube s := by sorry

-- Theorem for statement (D)
theorem rectangle_half_width_triple_height (w h : ℝ) :
  Rectangle (w / 2) (3 * h) = (3 / 2) * Rectangle w h := by sorry

-- Theorem for statement (E)
theorem rectangle_double_length_triple_width (l w : ℝ) :
  Rectangle (2 * l) (3 * w) = 6 * Rectangle l w := by sorry

-- Main theorem proving (A) is false and others are true
theorem geometric_transformations :
  (∃ w h : ℝ, Rectangle w (2 * h) ≠ 3 * Rectangle w h) ∧
  (∀ s : ℝ, Square (2 * s) = 4 * Square s) ∧
  (∀ s : ℝ, Cube (2 * s) = 8 * Cube s) ∧
  (∀ w h : ℝ, Rectangle (w / 2) (3 * h) = (3 / 2) * Rectangle w h) ∧
  (∀ l w : ℝ, Rectangle (2 * l) (3 * w) = 6 * Rectangle l w) := by sorry

end NUMINAMATH_CALUDE_rectangle_height_double_area_square_side_double_area_cube_side_double_volume_rectangle_half_width_triple_height_rectangle_double_length_triple_width_geometric_transformations_l3404_340437


namespace NUMINAMATH_CALUDE_arithmetic_progression_theorem_l3404_340486

/-- An arithmetic progression with n terms, first term a, and common difference d. -/
structure ArithmeticProgression where
  n : ℕ
  a : ℚ
  d : ℚ

/-- Sum of the first k terms of an arithmetic progression -/
def sum_first_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  k / 2 * (2 * ap.a + (k - 1) * ap.d)

/-- Sum of the last k terms of an arithmetic progression -/
def sum_last_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  k * (2 * ap.a + (ap.n - k + 1 + ap.n - 1) * ap.d / 2)

/-- Sum of all terms except the first k terms -/
def sum_without_first_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  (ap.n - k) / 2 * (2 * ap.a + (2 * k - 1 + ap.n - 1) * ap.d)

/-- Sum of all terms except the last k terms -/
def sum_without_last_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  (ap.n - k) / 2 * (2 * ap.a + (ap.n - k - 1) * ap.d)

/-- Theorem: If the sum of the first 13 terms is 50% of the sum of the last 13 terms,
    and the sum of all terms without the first 3 terms is 3/2 times the sum of all terms
    without the last 3 terms, then the number of terms in the progression is 18. -/
theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  sum_first_k ap 13 = (1/2) * sum_last_k ap 13 ∧
  sum_without_first_k ap 3 = (3/2) * sum_without_last_k ap 3 →
  ap.n = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_theorem_l3404_340486


namespace NUMINAMATH_CALUDE_kennel_problem_l3404_340441

/-- Represents the number of dogs with various accessories in a kennel -/
structure KennelData where
  total : ℕ
  tags : ℕ
  flea_collars : ℕ
  harnesses : ℕ
  tags_and_flea : ℕ
  tags_and_harnesses : ℕ
  flea_and_harnesses : ℕ
  all_three : ℕ

/-- Calculates the number of dogs with no accessories given kennel data -/
def dogs_with_no_accessories (data : KennelData) : ℕ :=
  data.total - (data.tags + data.flea_collars + data.harnesses - 
    data.tags_and_flea - data.tags_and_harnesses - data.flea_and_harnesses + data.all_three)

/-- Theorem stating that given the specific kennel data, 25 dogs have no accessories -/
theorem kennel_problem (data : KennelData) 
    (h1 : data.total = 120)
    (h2 : data.tags = 60)
    (h3 : data.flea_collars = 50)
    (h4 : data.harnesses = 30)
    (h5 : data.tags_and_flea = 20)
    (h6 : data.tags_and_harnesses = 15)
    (h7 : data.flea_and_harnesses = 10)
    (h8 : data.all_three = 5) :
  dogs_with_no_accessories data = 25 := by
  sorry

end NUMINAMATH_CALUDE_kennel_problem_l3404_340441


namespace NUMINAMATH_CALUDE_max_groups_equals_gcd_l3404_340419

theorem max_groups_equals_gcd (boys girls : ℕ) (h1 : boys = 120) (h2 : girls = 140) :
  let max_groups := Nat.gcd boys girls
  ∀ k : ℕ, k ∣ boys ∧ k ∣ girls → k ≤ max_groups :=
by sorry

end NUMINAMATH_CALUDE_max_groups_equals_gcd_l3404_340419


namespace NUMINAMATH_CALUDE_division_remainder_l3404_340493

theorem division_remainder (k : ℕ) : 
  k > 0 ∧ k < 38 ∧ 
  k % 5 = 2 ∧ 
  (∃ n : ℕ, n > 5 ∧ k % n = 5) →
  k % 7 = 5 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_l3404_340493


namespace NUMINAMATH_CALUDE_possible_values_of_p_l3404_340406

theorem possible_values_of_p (a b c p : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq : a + 1/b = p ∧ b + 1/c = p ∧ c + 1/a = p) :
  p = 1 ∨ p = -1 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_p_l3404_340406


namespace NUMINAMATH_CALUDE_optimal_prevention_plan_l3404_340412

/-- Represents a preventive measure -/
structure PreventiveMeasure where
  cost : ℝ
  effectiveness : ℝ

/-- Calculates the total cost given a set of preventive measures -/
def totalCost (baseProbability : ℝ) (baseLoss : ℝ) (measures : List PreventiveMeasure) : ℝ :=
  let preventionCost := measures.foldl (fun acc m => acc + m.cost) 0
  let incidentProbability := measures.foldl (fun acc m => acc * (1 - m.effectiveness)) baseProbability
  preventionCost + incidentProbability * baseLoss

theorem optimal_prevention_plan 
  (baseProbability : ℝ)
  (baseLoss : ℝ)
  (measureA : PreventiveMeasure)
  (measureB : PreventiveMeasure)
  (h1 : baseProbability = 0.3)
  (h2 : baseLoss = 400)
  (h3 : measureA.cost = 45)
  (h4 : measureA.effectiveness = 0.9)
  (h5 : measureB.cost = 30)
  (h6 : measureB.effectiveness = 0.85) :
  totalCost baseProbability baseLoss [measureA, measureB] < 
  min 
    (totalCost baseProbability baseLoss [])
    (min 
      (totalCost baseProbability baseLoss [measureA])
      (totalCost baseProbability baseLoss [measureB])) := by
  sorry

#check optimal_prevention_plan

end NUMINAMATH_CALUDE_optimal_prevention_plan_l3404_340412


namespace NUMINAMATH_CALUDE_right_triangle_area_l3404_340483

theorem right_triangle_area (a b c : ℝ) (h1 : a = 40) (h2 : c = 41) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3404_340483


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3404_340462

theorem max_sum_of_factors (X Y Z : ℕ) : 
  X > 0 ∧ Y > 0 ∧ Z > 0 →  -- Positive integers
  X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z →  -- Distinct integers
  X * Y * Z = 399 →        -- Product constraint
  X + Y + Z ≤ 29           -- Maximum sum
  := by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3404_340462


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l3404_340443

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the intersecting line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the midpoint condition
def midpoint_condition (k : ℝ) : Prop := (2*k + 4) / (k^2) = 2

-- Main theorem
theorem parabola_intersection_length :
  ∀ (k : ℝ),
  parabola 2 4 →
  k > -1 →
  k ≠ 0 →
  midpoint_condition k →
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line k x₁ y₁ ∧ line k x₂ y₂ ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 60 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l3404_340443


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l3404_340459

-- Define the binary number
def binary_num : ℕ := 0b101101

-- Define the octal number
def octal_num : ℕ := 0o55

-- Theorem statement
theorem binary_to_octal_conversion :
  binary_num = octal_num := by sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l3404_340459


namespace NUMINAMATH_CALUDE_photo_arrangement_probability_l3404_340402

/-- The number of boys -/
def num_boys : ℕ := 2

/-- The number of girls -/
def num_girls : ℕ := 5

/-- The total number of people -/
def total_people : ℕ := num_boys + num_girls

/-- The number of girls between the boys -/
def girls_between : ℕ := 3

/-- The probability of the specific arrangement -/
def probability : ℚ := 1 / 7

theorem photo_arrangement_probability :
  (num_boys = 2) →
  (num_girls = 5) →
  (girls_between = 3) →
  (probability = 1 / 7) :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangement_probability_l3404_340402


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3404_340454

theorem quadratic_equation_solution : 
  ∀ x : ℝ, (2 * x^2 + 10 * x + 12 = -(x + 4) * (x + 6)) ↔ (x = -4 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3404_340454


namespace NUMINAMATH_CALUDE_exists_valid_point_distribution_l3404_340425

/-- Represents a convex pentagon --/
structure ConvexPentagon where
  -- Add necessary fields

/-- Represents a point inside the pentagon --/
structure Point where
  -- Add necessary fields

/-- Represents a triangle formed by the vertices of the pentagon --/
structure Triangle where
  -- Add necessary fields

/-- Function to check if a point is inside a triangle --/
def pointInTriangle (p : Point) (t : Triangle) : Bool :=
  sorry

/-- Function to count points inside a triangle --/
def countPointsInTriangle (points : List Point) (t : Triangle) : Nat :=
  sorry

/-- Theorem stating the existence of a valid point distribution --/
theorem exists_valid_point_distribution (pentagon : ConvexPentagon) :
  ∃ (points : List Point),
    points.length = 18 ∧
    ∀ (t1 t2 : Triangle),
      countPointsInTriangle points t1 = countPointsInTriangle points t2 :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_point_distribution_l3404_340425


namespace NUMINAMATH_CALUDE_magic_trick_minimum_cards_l3404_340405

/-- The number of possible colors for the cards -/
def num_colors : ℕ := 2017

/-- The strategy function type for the assistant -/
def Strategy : Type := Fin num_colors → Fin num_colors

/-- The minimum number of cards needed for the trick -/
def min_cards : ℕ := 2018

theorem magic_trick_minimum_cards :
  ∀ (n : ℕ), n < min_cards →
    ¬∃ (s : Strategy),
      ∀ (colors : Fin n → Fin num_colors),
        ∃ (i : Fin n),
          ∀ (j : Fin n),
            j ≠ i →
              s (colors j) = colors i := by sorry

end NUMINAMATH_CALUDE_magic_trick_minimum_cards_l3404_340405


namespace NUMINAMATH_CALUDE_potato_bag_fraction_l3404_340471

theorem potato_bag_fraction (weight : ℝ) (x : ℝ) : 
  weight = 12 → weight / x = 12 → x = 1 := by sorry

end NUMINAMATH_CALUDE_potato_bag_fraction_l3404_340471


namespace NUMINAMATH_CALUDE_max_gumdrops_l3404_340400

/-- Represents the candy purchasing problem with given constraints --/
def CandyProblem (total_budget : ℕ) (bulk_cost gummy_cost gumdrop_cost : ℕ) 
                 (min_bulk min_gummy : ℕ) : Prop :=
  let remaining_budget := total_budget - (min_bulk * bulk_cost + min_gummy * gummy_cost)
  remaining_budget / gumdrop_cost = 28

/-- Theorem stating the maximum number of gumdrops that can be purchased --/
theorem max_gumdrops : 
  CandyProblem 224 8 6 4 10 5 := by
  sorry

#check max_gumdrops

end NUMINAMATH_CALUDE_max_gumdrops_l3404_340400


namespace NUMINAMATH_CALUDE_hardcover_count_l3404_340403

/-- Represents the purchase of a book series -/
structure BookPurchase where
  total_volumes : ℕ
  paperback_price : ℕ
  hardcover_price : ℕ
  total_cost : ℕ

/-- Theorem stating that under given conditions, the number of hardcover books is 6 -/
theorem hardcover_count (purchase : BookPurchase)
  (h_total : purchase.total_volumes = 8)
  (h_paperback : purchase.paperback_price = 10)
  (h_hardcover : purchase.hardcover_price = 20)
  (h_cost : purchase.total_cost = 140) :
  ∃ (h : ℕ), h = 6 ∧ 
    h * purchase.hardcover_price + (purchase.total_volumes - h) * purchase.paperback_price = purchase.total_cost :=
by sorry

end NUMINAMATH_CALUDE_hardcover_count_l3404_340403


namespace NUMINAMATH_CALUDE_square_dancing_problem_l3404_340446

/-- The number of female students in the first class that satisfies the square dancing conditions --/
def female_students_in_first_class : ℕ := by sorry

theorem square_dancing_problem :
  let males_class1 : ℕ := 17
  let males_class2 : ℕ := 14
  let females_class2 : ℕ := 18
  let males_class3 : ℕ := 15
  let females_class3 : ℕ := 17
  let total_males : ℕ := males_class1 + males_class2 + males_class3
  let total_females : ℕ := female_students_in_first_class + females_class2 + females_class3
  let unpartnered_students : ℕ := 2

  female_students_in_first_class = 9 ∧
  total_males = total_females + unpartnered_students := by sorry

end NUMINAMATH_CALUDE_square_dancing_problem_l3404_340446


namespace NUMINAMATH_CALUDE_number_problem_l3404_340423

theorem number_problem : ∃ x : ℝ, (x - 5) / 3 = 4 ∧ x = 17 := by sorry

end NUMINAMATH_CALUDE_number_problem_l3404_340423


namespace NUMINAMATH_CALUDE_max_value_f_on_interval_range_of_a_for_inequality_l3404_340487

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a

-- Part 1: Maximum value of f(x) when a = 2 on [-1, 1]
theorem max_value_f_on_interval :
  ∃ (M : ℝ), M = 5 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f 2 x ≤ M :=
sorry

-- Part 2: Range of a for f(x)/x ≥ 2 when x ∈ [1, 2]
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (1 : ℝ) 2, f a x / x ≥ 2) ↔ a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_on_interval_range_of_a_for_inequality_l3404_340487


namespace NUMINAMATH_CALUDE_product_lower_bound_l3404_340478

theorem product_lower_bound (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≥ 0) (h₂ : x₂ ≥ 0) (h₃ : x₃ ≥ 0) 
  (h₄ : x₁ + x₂ + x₃ ≤ 1/2) : 
  (1 - x₁) * (1 - x₂) * (1 - x₃) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_product_lower_bound_l3404_340478


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l3404_340428

theorem set_equality_implies_sum (a b : ℝ) (ha : a ≠ 0) :
  ({a, b / a, 1} : Set ℝ) = {a^2, a + b, 0} →
  a^2015 + b^2016 = -1 := by sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l3404_340428


namespace NUMINAMATH_CALUDE_factorization_a4_2a3_1_l3404_340431

theorem factorization_a4_2a3_1 (a : ℝ) : 
  a^4 + 2*a^3 + 1 = (a + 1) * (a^3 + a^2 - a + 1) := by sorry

end NUMINAMATH_CALUDE_factorization_a4_2a3_1_l3404_340431


namespace NUMINAMATH_CALUDE_sum_of_cyclic_equations_l3404_340479

theorem sum_of_cyclic_equations (x y z : ℝ) 
  (h1 : x + y = 1) 
  (h2 : y + z = 1) 
  (h3 : z + x = 1) : 
  x + y + z = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_equations_l3404_340479


namespace NUMINAMATH_CALUDE_two_green_then_red_probability_l3404_340458

/-- The number of traffic checkpoints -/
def num_checkpoints : ℕ := 6

/-- The probability of encountering a red light at each checkpoint -/
def red_light_prob : ℚ := 1/3

/-- The probability of passing exactly two checkpoints before encountering a red light -/
def prob_two_green_then_red : ℚ := 4/27

theorem two_green_then_red_probability :
  (1 - red_light_prob)^2 * red_light_prob = prob_two_green_then_red :=
sorry

end NUMINAMATH_CALUDE_two_green_then_red_probability_l3404_340458


namespace NUMINAMATH_CALUDE_symmetric_function_intersection_l3404_340484

/-- Definition of a symmetric function -/
def symmetricFunction (m n : ℝ) : ℝ → ℝ := λ x ↦ n * x + m

/-- The given function -/
def givenFunction : ℝ → ℝ := λ x ↦ -6 * x + 4

/-- Theorem: The intersection point of the symmetric function of y=-6x+4 with the y-axis is (0, -6) -/
theorem symmetric_function_intersection :
  let f := symmetricFunction (-6) 4
  (0, f 0) = (0, -6) := by sorry

end NUMINAMATH_CALUDE_symmetric_function_intersection_l3404_340484


namespace NUMINAMATH_CALUDE_vector_perpendicular_to_line_l3404_340451

/-- Given a vector a and a line l, prove that they are perpendicular -/
theorem vector_perpendicular_to_line (a : ℝ × ℝ) (l : ℝ → ℝ → Prop) : 
  a = (2, 3) → 
  (∀ x y, l x y ↔ 2 * x + 3 * y - 1 = 0) → 
  ∃ k, k * a.1 + a.2 = 0 ∧ k * 2 - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_vector_perpendicular_to_line_l3404_340451


namespace NUMINAMATH_CALUDE_tan_x_minus_pi_fourth_l3404_340495

theorem tan_x_minus_pi_fourth (x : ℝ) 
  (h1 : x ∈ Set.Ioo 0 π) 
  (h2 : Real.cos (2 * x - π / 2) = Real.sin x ^ 2) : 
  Real.tan (x - π / 4) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_minus_pi_fourth_l3404_340495


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_65_l3404_340463

theorem complex_expression_equals_negative_65 :
  -2^3 * (-3)^2 / (9/8) - |1/2 - 3/2| = -65 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_65_l3404_340463


namespace NUMINAMATH_CALUDE_xiao_gao_score_l3404_340424

/-- Represents a test score system with a standard score and a recorded score. -/
structure TestScore where
  standard : ℕ
  recorded : ℤ

/-- Calculates the actual score given a TestScore. -/
def actualScore (ts : TestScore) : ℕ :=
  ts.standard + ts.recorded.toNat

/-- Theorem stating that for a standard score of 80 and a recorded score of 12,
    the actual score is 92. -/
theorem xiao_gao_score :
  let ts : TestScore := { standard := 80, recorded := 12 }
  actualScore ts = 92 := by
  sorry

end NUMINAMATH_CALUDE_xiao_gao_score_l3404_340424


namespace NUMINAMATH_CALUDE_simplify_expressions_l3404_340439

theorem simplify_expressions (a x : ℝ) :
  (-a^3 + (-4*a^2)*a = -5*a^3) ∧
  (-x^2 * (-x)^2 * (-x^2)^3 - 2*x^10 = -x^10) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3404_340439


namespace NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l3404_340485

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Type := Unit

/-- The number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1s in the first n rows of Pascal's Triangle -/
def numberOfOnes (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ := (numberOfOnes n : ℚ) / (totalElements n : ℚ)

theorem probability_of_one_in_20_rows :
  probabilityOfOne 20 = 13 / 70 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l3404_340485


namespace NUMINAMATH_CALUDE_zach_ticket_purchase_l3404_340452

/-- The number of tickets Zach needs to buy for both rides -/
def tickets_needed (ferris_wheel_cost roller_coaster_cost multiple_ride_discount coupon : ℝ) : ℝ :=
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - coupon

/-- Theorem stating the number of tickets Zach needs to buy -/
theorem zach_ticket_purchase :
  tickets_needed 2.0 7.0 1.0 1.0 = 7.0 := by
  sorry

#eval tickets_needed 2.0 7.0 1.0 1.0

end NUMINAMATH_CALUDE_zach_ticket_purchase_l3404_340452


namespace NUMINAMATH_CALUDE_coin_toss_probability_l3404_340449

/-- The probability of getting exactly k heads in n tosses of a coin with probability r of landing heads -/
def binomial_probability (n k : ℕ) (r : ℚ) : ℚ :=
  (n.choose k : ℚ) * r^k * (1 - r)^(n - k)

/-- The main theorem -/
theorem coin_toss_probability : ∀ r : ℚ,
  0 < r →
  r < 1 →
  binomial_probability 5 1 r = binomial_probability 5 2 r →
  binomial_probability 5 3 r = 40 / 243 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l3404_340449


namespace NUMINAMATH_CALUDE_solution_comparison_l3404_340426

theorem solution_comparison (c d e f : ℝ) (hc : c ≠ 0) (he : e ≠ 0) :
  (-d / c > -f / e) ↔ (f / e > d / c) :=
sorry

end NUMINAMATH_CALUDE_solution_comparison_l3404_340426


namespace NUMINAMATH_CALUDE_negative_subtraction_l3404_340404

theorem negative_subtraction (a b : ℤ) : -5 - (-2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_subtraction_l3404_340404


namespace NUMINAMATH_CALUDE_smallest_four_digit_in_pascals_triangle_l3404_340416

def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (k m : ℕ), n = Nat.choose m k

theorem smallest_four_digit_in_pascals_triangle :
  (∀ n : ℕ, n < 1000 → ¬(is_in_pascals_triangle n ∧ n ≥ 1000)) ∧
  is_in_pascals_triangle 1000 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_in_pascals_triangle_l3404_340416


namespace NUMINAMATH_CALUDE_only_prime_of_form_l3404_340498

theorem only_prime_of_form (p : ℕ) : 
  (∃ x : ℤ, p = 4 * x^4 + 1) ∧ Nat.Prime p ↔ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_only_prime_of_form_l3404_340498


namespace NUMINAMATH_CALUDE_julian_needs_1100_more_legos_l3404_340429

/-- The number of legos Julian has -/
def julian_legos : ℕ := 400

/-- The number of airplane models Julian wants to make -/
def num_models : ℕ := 4

/-- The number of legos required for each airplane model -/
def legos_per_model : ℕ := 375

/-- The number of additional legos Julian needs -/
def additional_legos_needed : ℕ := 1100

/-- Theorem stating that Julian needs 1100 more legos to make 4 identical airplane models -/
theorem julian_needs_1100_more_legos :
  (num_models * legos_per_model) - julian_legos = additional_legos_needed := by
  sorry

end NUMINAMATH_CALUDE_julian_needs_1100_more_legos_l3404_340429


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3404_340413

theorem quadratic_roots_sum_of_squares (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^2 + q^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3404_340413


namespace NUMINAMATH_CALUDE_at_least_one_negative_l3404_340440

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l3404_340440


namespace NUMINAMATH_CALUDE_unique_phone_number_l3404_340418

def is_valid_phone_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def first_upgrade (n : ℕ) : ℕ :=
  let d := n.div 100000
  let r := n.mod 100000
  d * 1000000 + 8 * 100000 + r

def second_upgrade (n : ℕ) : ℕ :=
  2000000000 + n

theorem unique_phone_number :
  ∃! n : ℕ, is_valid_phone_number n ∧ 
    second_upgrade (first_upgrade n) = 81 * n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_phone_number_l3404_340418


namespace NUMINAMATH_CALUDE_equation_solution_l3404_340427

theorem equation_solution (a b : ℝ) :
  (∀ x : ℝ, (a*x^2 + b*x - 5)*(a*x^2 + b*x + 25) + c = (a*x^2 + b*x + 10)^2) →
  c = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3404_340427


namespace NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_three_l3404_340469

theorem fraction_equality_implies_x_equals_three (x : ℝ) :
  (5 / (2 * x - 1) = 3 / x) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_three_l3404_340469


namespace NUMINAMATH_CALUDE_seven_swimmer_race_outcomes_l3404_340447

/-- The number of different possible outcomes for 1st-2nd-3rd place in a race with n swimmers and no ties -/
def race_outcomes (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- Theorem: The number of different possible outcomes for 1st-2nd-3rd place in a race with 7 swimmers and no ties is 210 -/
theorem seven_swimmer_race_outcomes : race_outcomes 7 = 210 := by
  sorry

end NUMINAMATH_CALUDE_seven_swimmer_race_outcomes_l3404_340447


namespace NUMINAMATH_CALUDE_integral_quarter_circle_area_l3404_340475

theorem integral_quarter_circle_area (r : ℝ) (h : r > 0) :
  ∫ x in (0)..(r), Real.sqrt (r^2 - x^2) = (π * r^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_quarter_circle_area_l3404_340475


namespace NUMINAMATH_CALUDE_clock_equivalent_hours_l3404_340415

theorem clock_equivalent_hours : ∃ h : ℕ, h > 6 ∧ h ≡ h^2 [ZMOD 24] ∧ ∀ k : ℕ, k > 6 ∧ k < h → ¬(k ≡ k^2 [ZMOD 24]) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_clock_equivalent_hours_l3404_340415


namespace NUMINAMATH_CALUDE_existence_of_solution_l3404_340480

theorem existence_of_solution : ∃ (x y : ℤ), 2 * x^2 + 8 * y = 26 ∧ x - y = 26 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l3404_340480


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l3404_340491

/-- The perimeter of a large rectangle composed of nine identical smaller rectangles -/
theorem large_rectangle_perimeter (small_length : ℝ) (h1 : small_length = 10) :
  let large_length := 2 * small_length
  let large_height := 4 * small_length / 5
  let perimeter := 2 * (large_length + large_height)
  perimeter = 76 := by sorry

end NUMINAMATH_CALUDE_large_rectangle_perimeter_l3404_340491


namespace NUMINAMATH_CALUDE_biology_marks_calculation_l3404_340457

def english_marks : ℕ := 96
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def average_marks : ℕ := 79
def total_subjects : ℕ := 5

theorem biology_marks_calculation :
  ∃ (biology_marks : ℕ),
    biology_marks = average_marks * total_subjects - (english_marks + math_marks + physics_marks + chemistry_marks) ∧
    biology_marks = 85 := by
  sorry

end NUMINAMATH_CALUDE_biology_marks_calculation_l3404_340457


namespace NUMINAMATH_CALUDE_equation_solution_l3404_340492

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ 2*x ≠ 2 ∧ x / (x - 1) = 3 / (2*x - 2) - 2 ∧ x = 7/6 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3404_340492


namespace NUMINAMATH_CALUDE_contractor_fine_calculation_l3404_340499

/-- Proves that the fine for each day of absence is 7.5 given the contract conditions --/
theorem contractor_fine_calculation (total_days : ℕ) (work_pay : ℝ) (total_earnings : ℝ) (absent_days : ℕ) :
  total_days = 30 →
  work_pay = 25 →
  total_earnings = 360 →
  absent_days = 12 →
  ∃ (fine : ℝ), fine = 7.5 ∧ 
    work_pay * (total_days - absent_days) - fine * absent_days = total_earnings :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_fine_calculation_l3404_340499


namespace NUMINAMATH_CALUDE_range_of_a_l3404_340434

def S (a : ℝ) : Set ℝ := {x | 2 * a * x^2 - x ≤ 0}

def T (a : ℝ) : Set ℝ := {x | 4 * a * x^2 - 4 * a * (1 - 2 * a) * x + 1 ≥ 0}

theorem range_of_a (a : ℝ) (h : S a ∪ T a = Set.univ) : 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3404_340434


namespace NUMINAMATH_CALUDE_zoltan_incorrect_answers_l3404_340436

theorem zoltan_incorrect_answers 
  (total_questions : Nat)
  (answered_questions : Nat)
  (total_score : Int)
  (correct_points : Int)
  (incorrect_points : Int)
  (unanswered_points : Int)
  (h1 : total_questions = 50)
  (h2 : answered_questions = 45)
  (h3 : total_score = 135)
  (h4 : correct_points = 4)
  (h5 : incorrect_points = -1)
  (h6 : unanswered_points = 0) :
  ∃ (incorrect : Nat),
    incorrect = 9 ∧
    (answered_questions - incorrect) * correct_points + 
    incorrect * incorrect_points + 
    (total_questions - answered_questions) * unanswered_points = total_score :=
by sorry

end NUMINAMATH_CALUDE_zoltan_incorrect_answers_l3404_340436


namespace NUMINAMATH_CALUDE_count_numbers_with_three_is_180_l3404_340461

/-- The count of natural numbers from 1 to 1000 that contain the digit 3 at least once -/
def count_numbers_with_three : ℕ :=
  let total_numbers := 1000
  let numbers_without_three := 820
  total_numbers - numbers_without_three

/-- Theorem stating that the count of natural numbers from 1 to 1000 
    containing the digit 3 at least once is equal to 180 -/
theorem count_numbers_with_three_is_180 :
  count_numbers_with_three = 180 := by
  sorry

#eval count_numbers_with_three

end NUMINAMATH_CALUDE_count_numbers_with_three_is_180_l3404_340461


namespace NUMINAMATH_CALUDE_quarters_addition_theorem_l3404_340442

/-- The number of quarters initially in the jar -/
def initial_quarters : ℕ := 267

/-- The value of one quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The target total value in dollars -/
def target_value : ℚ := 100

/-- The number of quarters to be added -/
def quarters_to_add : ℕ := 133

theorem quarters_addition_theorem :
  (initial_quarters + quarters_to_add : ℚ) * quarter_value = target_value := by
  sorry

end NUMINAMATH_CALUDE_quarters_addition_theorem_l3404_340442


namespace NUMINAMATH_CALUDE_camilo_kenny_difference_l3404_340497

def paint_house_problem (judson_contribution kenny_contribution camilo_contribution total_cost : ℕ) : Prop :=
  judson_contribution = 500 ∧
  kenny_contribution = judson_contribution + judson_contribution / 5 ∧
  camilo_contribution > kenny_contribution ∧
  total_cost = 1900 ∧
  judson_contribution + kenny_contribution + camilo_contribution = total_cost

theorem camilo_kenny_difference :
  ∀ judson_contribution kenny_contribution camilo_contribution total_cost,
    paint_house_problem judson_contribution kenny_contribution camilo_contribution total_cost →
    camilo_contribution - kenny_contribution = 200 := by
  sorry

end NUMINAMATH_CALUDE_camilo_kenny_difference_l3404_340497


namespace NUMINAMATH_CALUDE_sheet_length_l3404_340467

theorem sheet_length (width : ℝ) (side_margin : ℝ) (top_bottom_margin : ℝ) (typing_percentage : ℝ) :
  width = 20 →
  side_margin = 2 →
  top_bottom_margin = 3 →
  typing_percentage = 0.64 →
  ∃ length : ℝ,
    length = 30 ∧
    (width - 2 * side_margin) * (length - 2 * top_bottom_margin) = typing_percentage * width * length :=
by sorry

end NUMINAMATH_CALUDE_sheet_length_l3404_340467


namespace NUMINAMATH_CALUDE_certain_number_value_l3404_340433

theorem certain_number_value (p q : ℕ) (x : ℚ) 
  (hp : p > 1) 
  (hq : q > 1) 
  (hx : x * (p + 1) = 28 * (q + 1)) 
  (hpq_min : ∀ (p' q' : ℕ), p' > 1 → q' > 1 → p' + q' < p + q → ¬∃ (x' : ℚ), x' * (p' + 1) = 28 * (q' + 1)) :
  x = 392 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_l3404_340433


namespace NUMINAMATH_CALUDE_odd_sequence_concat_theorem_l3404_340466

def odd_sequence (n : ℕ) : List ℕ :=
  List.filter (λ x => x % 2 = 1) (List.range (n + 1))

def concat_digits (lst : List ℕ) : ℕ := sorry

def digit_sum (n : ℕ) : ℕ := sorry

theorem odd_sequence_concat_theorem :
  let seq := odd_sequence 103
  let A := concat_digits seq
  (Nat.digits 10 A).length = 101 ∧ A % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_sequence_concat_theorem_l3404_340466


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l3404_340496

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ x : ℤ, (x + 6 < 2 + 3*x ∧ (a + x) / 4 > x) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
  (15 < a ∧ a ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l3404_340496


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l3404_340460

theorem circle_tangent_to_line (x y : ℝ) :
  (∀ a b : ℝ, a^2 + b^2 = 2 → (b ≠ 2 - a ∨ (a - 0)^2 + (b - 0)^2 = 2)) ∧
  (∃ c d : ℝ, c^2 + d^2 = 2 ∧ d = 2 - c) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l3404_340460


namespace NUMINAMATH_CALUDE_kimberly_skittles_l3404_340411

def skittles_problem (initial_skittles : ℚ) 
                     (eaten_skittles : ℚ) 
                     (given_skittles : ℚ) 
                     (promotion_skittles : ℚ) 
                     (exchange_skittles : ℚ) : Prop :=
  let remaining_after_eating := initial_skittles - eaten_skittles
  let remaining_after_giving := remaining_after_eating - given_skittles
  let after_promotion := remaining_after_giving + promotion_skittles
  let final_skittles := after_promotion + exchange_skittles
  final_skittles = 18

theorem kimberly_skittles : 
  skittles_problem 7.5 2.25 1.5 3.75 10.5 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_skittles_l3404_340411


namespace NUMINAMATH_CALUDE_williams_riding_time_l3404_340438

def max_riding_time : ℝ := 6

theorem williams_riding_time (x : ℝ) : 
  (2 * max_riding_time) + (2 * x) + (2 * (max_riding_time / 2)) = 21 → x = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_williams_riding_time_l3404_340438


namespace NUMINAMATH_CALUDE_probability_theorem_l3404_340432

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Define the condition x^2 < 1
def condition (x : ℝ) : Prop := x^2 < 1

-- Define the measure of the interval [-2, 2]
def totalMeasure : ℝ := 4

-- Define the measure of the solution set (-1, 1)
def solutionMeasure : ℝ := 2

-- State the theorem
theorem probability_theorem :
  (solutionMeasure / totalMeasure) = (1 / 2) := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l3404_340432


namespace NUMINAMATH_CALUDE_orange_probability_l3404_340473

theorem orange_probability (total : ℕ) (large : ℕ) (small : ℕ) (choose : ℕ) :
  total = 8 →
  large = 5 →
  small = 3 →
  choose = 3 →
  (Nat.choose small choose : ℚ) / (Nat.choose total choose : ℚ) = 1 / 56 :=
by sorry

end NUMINAMATH_CALUDE_orange_probability_l3404_340473


namespace NUMINAMATH_CALUDE_books_left_l3404_340407

/-- Given that Paul had 242 books initially and sold 137 books, prove that he has 105 books left. -/
theorem books_left (initial_books : ℕ) (sold_books : ℕ) (h1 : initial_books = 242) (h2 : sold_books = 137) :
  initial_books - sold_books = 105 := by
  sorry

end NUMINAMATH_CALUDE_books_left_l3404_340407


namespace NUMINAMATH_CALUDE_two_digit_subtraction_equality_l3404_340489

theorem two_digit_subtraction_equality (a b : Nat) : 
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a ≠ b → (70 * a - 7 * a) - (70 * b - 7 * b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_subtraction_equality_l3404_340489


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_7_l3404_340455

theorem largest_integer_less_than_100_remainder_5_mod_7 : 
  ∀ n : ℤ, n < 100 ∧ n % 7 = 5 → n ≤ 96 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_7_l3404_340455


namespace NUMINAMATH_CALUDE_equation_solution_l3404_340476

theorem equation_solution :
  ∃ x : ℚ, (x - 30) / 3 = (5 - 3 * x) / 4 ∧ x = 135 / 13 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3404_340476


namespace NUMINAMATH_CALUDE_copper_zinc_ratio_l3404_340422

/-- Given a mixture of copper and zinc, prove that the ratio of copper to zinc is 77:63 -/
theorem copper_zinc_ratio (total_weight zinc_weight : ℝ)
  (h_total : total_weight = 70)
  (h_zinc : zinc_weight = 31.5)
  : ∃ (a b : ℕ), a = 77 ∧ b = 63 ∧ (total_weight - zinc_weight) / zinc_weight = a / b := by
  sorry

end NUMINAMATH_CALUDE_copper_zinc_ratio_l3404_340422


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3404_340481

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 12 * x - 18

/-- The point of tangency -/
def p : ℝ × ℝ := (-2, 3)

/-- The slope of the tangent line at the point of tangency -/
def m : ℝ := f' p.1

theorem tangent_line_equation :
  ∀ x y : ℝ, y = f p.1 → (y - p.2 = m * (x - p.1)) ↔ (30 * x - y + 63 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3404_340481


namespace NUMINAMATH_CALUDE_magnitude_of_b_l3404_340417

def a : ℝ × ℝ := (2, 3)

theorem magnitude_of_b (b : ℝ × ℝ) 
  (h : (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0) : 
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_b_l3404_340417


namespace NUMINAMATH_CALUDE_taxi_charge_proof_l3404_340470

/-- Calculates the total charge for a taxi trip -/
def totalCharge (initialFee : ℚ) (additionalChargePerIncrement : ℚ) (incrementDistance : ℚ) (tripDistance : ℚ) : ℚ :=
  initialFee + (tripDistance / incrementDistance).floor * additionalChargePerIncrement

/-- Proves that the total charge for a 3.6-mile trip is $4.50 -/
theorem taxi_charge_proof :
  let initialFee : ℚ := 9/4  -- $2.25
  let additionalChargePerIncrement : ℚ := 1/4  -- $0.25
  let incrementDistance : ℚ := 2/5  -- 2/5 mile
  let tripDistance : ℚ := 18/5  -- 3.6 miles
  totalCharge initialFee additionalChargePerIncrement incrementDistance tripDistance = 9/2  -- $4.50
:= by sorry

end NUMINAMATH_CALUDE_taxi_charge_proof_l3404_340470


namespace NUMINAMATH_CALUDE_difference_before_l3404_340465

/-- The number of battle cards Sang-cheol had originally -/
def S : ℕ := sorry

/-- The number of battle cards Byeong-ji had originally -/
def B : ℕ := sorry

/-- Sang-cheol gave Byeong-ji 2 battle cards -/
axiom exchange : S ≥ 2

/-- After the exchange, the difference between Byeong-ji and Sang-cheol was 6 -/
axiom difference_after : B + 2 - (S - 2) = 6

/-- Byeong-ji has more cards than Sang-cheol -/
axiom byeongji_has_more : B > S

/-- The difference between Byeong-ji and Sang-cheol before the exchange was 2 -/
theorem difference_before : B - S = 2 := by sorry

end NUMINAMATH_CALUDE_difference_before_l3404_340465


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3404_340420

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3404_340420


namespace NUMINAMATH_CALUDE_hash_composition_l3404_340444

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.75 * N + 2

-- State the theorem
theorem hash_composition : hash (hash (hash 72)) = 35 := by sorry

end NUMINAMATH_CALUDE_hash_composition_l3404_340444


namespace NUMINAMATH_CALUDE_find_a_solve_inequality_l3404_340464

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 6

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | x < 1 ∨ x > 2}

-- Theorem 1: Given the solution set, prove a = 1
theorem find_a : ∀ a : ℝ, (∀ x : ℝ, f a x > 4 ↔ x ∈ solution_set a) → a = 1 := by sorry

-- Define the linear function for the second inequality
def g (c : ℝ) (x : ℝ) : ℝ := (c - x) * (x + 2)

-- Theorem 2: Solve the inequality (c-x)(x+2) > 0
theorem solve_inequality :
  ∀ c : ℝ,
  (c = -2 → {x : ℝ | g c x > 0} = ∅) ∧
  (c > -2 → {x : ℝ | g c x > 0} = Set.Ioo (-2) c) ∧
  (c < -2 → {x : ℝ | g c x > 0} = Set.Ioo c (-2)) := by sorry

end NUMINAMATH_CALUDE_find_a_solve_inequality_l3404_340464


namespace NUMINAMATH_CALUDE_calculate_upstream_speed_l3404_340482

/-- Represents the speed of a man rowing in different water conditions -/
structure RowingSpeed where
  still : ℝ  -- Speed in still water
  downstream : ℝ  -- Speed downstream
  upstream : ℝ  -- Speed upstream

/-- Theorem: Given a man's speed in still water and downstream, calculate his upstream speed -/
theorem calculate_upstream_speed (speed : RowingSpeed) 
  (h1 : speed.still = 35)
  (h2 : speed.downstream = 40) : 
  speed.upstream = 30 := by
  sorry

#check calculate_upstream_speed

end NUMINAMATH_CALUDE_calculate_upstream_speed_l3404_340482


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3404_340474

def P (x : ℝ) : ℝ := x^4 + 2*x^3 - 23*x^2 + 12*x + 36

theorem polynomial_factorization :
  ∃ (a b c : ℝ),
    (∀ x, P x = (x^2 + a*x + c) * (x^2 + b*x + c)) ∧
    a + b = 2 ∧
    a * b = -35 ∧
    c = 6 ∧
    (∀ x, P x = 0 ↔ x = 2 ∨ x = 3 ∨ x = -1 ∨ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3404_340474


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3404_340445

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3404_340445


namespace NUMINAMATH_CALUDE_power_sum_equality_l3404_340448

theorem power_sum_equality : (3^2)^3 + (2^3)^2 = 793 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3404_340448


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l3404_340477

/-- Time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (total_length : ℝ) 
  (h1 : train_length = 130) 
  (h2 : train_speed_kmh = 45) 
  (h3 : total_length = 245) : 
  (total_length / (train_speed_kmh * 1000 / 3600)) = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l3404_340477


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3404_340450

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ (Real.sqrt 244 - 7)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3404_340450


namespace NUMINAMATH_CALUDE_square_plus_integer_l3404_340408

theorem square_plus_integer (y : ℝ) : y^2 + 14*y + 48 = (y+7)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_integer_l3404_340408


namespace NUMINAMATH_CALUDE_painted_faces_count_l3404_340401

/-- Represents a cube with a given side length -/
structure Cube :=
  (side_length : ℕ)

/-- Represents a painted cube with three adjacent painted faces -/
structure PaintedCube extends Cube :=
  (painted_faces : Fin 3)

/-- Counts the number of unit cubes with at least two painted faces when a painted cube is cut into unit cubes -/
def count_multi_painted_faces (c : PaintedCube) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 cube painted on three adjacent faces, when cut into unit cubes, has 14 cubes with at least two painted faces -/
theorem painted_faces_count (c : PaintedCube) (h : c.side_length = 4) : 
  count_multi_painted_faces c = 14 :=
sorry

end NUMINAMATH_CALUDE_painted_faces_count_l3404_340401


namespace NUMINAMATH_CALUDE_ball_count_problem_l3404_340409

/-- Proves that given the initial ratio of green to yellow balls is 3:7, 
    and after removing 9 balls of each color the new ratio becomes 1:3, 
    the original number of balls in the bag was 90. -/
theorem ball_count_problem (g y : ℕ) : 
  g * 7 = y * 3 →  -- initial ratio is 3:7
  (g - 9) * 3 = (y - 9) * 1 →  -- new ratio is 1:3 after removing 9 of each
  g + y = 90 := by  -- total number of balls is 90
sorry

end NUMINAMATH_CALUDE_ball_count_problem_l3404_340409


namespace NUMINAMATH_CALUDE_sequence_property_l3404_340435

/-- Given a sequence {a_n} with sum of first n terms S_n = 2a_n - a_1,
    and a_1, a_2+1, a_3 form an arithmetic sequence, prove a_n = 2^n -/
theorem sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = 2 * a n - a 1) → 
  (2 * (a 2 + 1) = a 3 + a 1) →
  ∀ n, a n = 2^n := by sorry

end NUMINAMATH_CALUDE_sequence_property_l3404_340435


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l3404_340453

theorem quadratic_root_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 14) →
  p + q = 69 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l3404_340453


namespace NUMINAMATH_CALUDE_quarter_circles_limit_l3404_340468

/-- The limit of the sum of quarter-circle lengths approaches the original circumference -/
theorem quarter_circles_limit (C : ℝ) (h : C > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |2 * n * (C / (2 * n)) - C| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circles_limit_l3404_340468


namespace NUMINAMATH_CALUDE_seating_arrangements_l3404_340414

def n : ℕ := 8

def numArrangements : ℕ := n.factorial - (n-1).factorial * 2

theorem seating_arrangements (n : ℕ) (h : n = 8) : 
  numArrangements = 30240 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3404_340414


namespace NUMINAMATH_CALUDE_final_price_calculation_l3404_340490

def original_price : ℝ := 10.00
def increase_percent : ℝ := 0.40
def decrease_percent : ℝ := 0.30

def price_after_increase (p : ℝ) (i : ℝ) : ℝ := p * (1 + i)
def price_after_decrease (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)

theorem final_price_calculation : 
  price_after_decrease (price_after_increase original_price increase_percent) decrease_percent = 9.80 := by
  sorry

end NUMINAMATH_CALUDE_final_price_calculation_l3404_340490


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l3404_340430

theorem product_of_sum_of_squares (a b c d : ℤ) : ∃ x y : ℤ, (a^2 + b^2) * (c^2 + d^2) = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l3404_340430
