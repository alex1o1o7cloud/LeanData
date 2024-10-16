import Mathlib

namespace NUMINAMATH_CALUDE_mary_max_earnings_l273_27338

/-- Mary's maximum weekly earnings at the restaurant --/
theorem mary_max_earnings (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℚ) 
  (overtime_rate_increase : ℚ) (bonus_hours : ℕ) (bonus_amount : ℚ) :
  max_hours = 80 →
  regular_hours = 20 →
  regular_rate = 8 →
  overtime_rate_increase = 1/4 →
  bonus_hours = 5 →
  bonus_amount = 20 →
  let overtime_hours := max_hours - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  let regular_earnings := regular_hours * regular_rate
  let overtime_earnings := overtime_hours * overtime_rate
  let bonus_count := overtime_hours / bonus_hours
  let total_bonus := bonus_count * bonus_amount
  regular_earnings + overtime_earnings + total_bonus = 1000 := by
sorry

end NUMINAMATH_CALUDE_mary_max_earnings_l273_27338


namespace NUMINAMATH_CALUDE_parabola_properties_l273_27304

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the parabola at a given x -/
def Parabola.eval (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- A parabola that passes through the given points -/
def specificParabola : Parabola :=
  { a := sorry
    b := sorry
    c := sorry }

theorem parabola_properties :
  let p := specificParabola
  -- The parabola passes through the given points
  (p.eval (-2) = 0) ∧
  (p.eval (-1) = 4) ∧
  (p.eval 0 = 6) ∧
  (p.eval 1 = 6) →
  -- 1. The parabola opens downwards
  (p.a < 0) ∧
  -- 2. The axis of symmetry is x = 1/2
  (- p.b / (2 * p.a) = 1/2) ∧
  -- 3. The maximum value of the function is 25/4
  (p.c - p.b^2 / (4 * p.a) = 25/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l273_27304


namespace NUMINAMATH_CALUDE_pizza_fraction_l273_27319

theorem pizza_fraction (total_slices : ℕ) (whole_slice : ℚ) (shared_slice : ℚ) :
  total_slices = 16 →
  whole_slice = 1 / total_slices →
  shared_slice = 1 / (3 * total_slices) →
  whole_slice + shared_slice = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_l273_27319


namespace NUMINAMATH_CALUDE_parabola_directrix_l273_27350

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop :=
  y = (x^2 - 6*x + 5) / 12

/-- The directrix equation -/
def directrix (y : ℝ) : Prop :=
  y = -10/3

/-- Theorem stating that the given directrix is correct for the parabola -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    ∃ f : ℝ × ℝ, ∃ q : ℝ × ℝ, 
      q.2 = d ∧ 
      (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l273_27350


namespace NUMINAMATH_CALUDE_final_cell_count_l273_27361

/-- Calculates the number of cells after a given number of days, 
    where cells double every 3 days starting from an initial population. -/
def cell_count (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * 2^(days / 3)

/-- Theorem stating that given 4 initial cells and 9 days, 
    the final cell count is 32. -/
theorem final_cell_count : cell_count 4 9 = 32 := by
  sorry

end NUMINAMATH_CALUDE_final_cell_count_l273_27361


namespace NUMINAMATH_CALUDE_weight_difference_l273_27324

def bridget_weight : ℕ := 39
def martha_weight : ℕ := 2

theorem weight_difference : bridget_weight - martha_weight = 37 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l273_27324


namespace NUMINAMATH_CALUDE_min_value_of_z_l273_27354

theorem min_value_of_z (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x + 1/x) * (y + 1/y) ≥ 25/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_z_l273_27354


namespace NUMINAMATH_CALUDE_simplify_expression_l273_27326

theorem simplify_expression (x : ℝ) : x^2 * x^4 + x * x^2 * x^3 = 2 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l273_27326


namespace NUMINAMATH_CALUDE_unit_price_ratio_of_quantity_and_price_difference_l273_27314

/-- Given two products A and B, where A offers 30% more quantity and costs 15% less than B,
    this theorem proves that the ratio of unit prices (A to B) is 17/26. -/
theorem unit_price_ratio_of_quantity_and_price_difference 
  (quantity_A quantity_B : ℝ) 
  (price_A price_B : ℝ) 
  (h_quantity : quantity_A = 1.3 * quantity_B)
  (h_price : price_A = 0.85 * price_B)
  (h_positive_quantity : quantity_B > 0)
  (h_positive_price : price_B > 0) :
  (price_A / quantity_A) / (price_B / quantity_B) = 17 / 26 := by
  sorry

end NUMINAMATH_CALUDE_unit_price_ratio_of_quantity_and_price_difference_l273_27314


namespace NUMINAMATH_CALUDE_weighted_average_is_38_5_l273_27320

/-- Represents the marks in different subjects -/
structure Marks where
  mathematics : ℝ
  physics : ℝ
  chemistry : ℝ
  biology : ℝ

/-- Calculates the weighted average of Mathematics, Chemistry, and Biology marks -/
def weightedAverage (m : Marks) : ℝ :=
  0.4 * m.mathematics + 0.3 * m.chemistry + 0.3 * m.biology

/-- Theorem stating that under given conditions, the weighted average is 38.5 -/
theorem weighted_average_is_38_5 (m : Marks) :
  m.mathematics + m.physics + m.biology = 90 ∧
  m.chemistry = m.physics + 10 ∧
  m.biology = m.chemistry - 5 →
  weightedAverage m = 38.5 := by
  sorry

#eval weightedAverage { mathematics := 85, physics := 0, chemistry := 10, biology := 5 }

end NUMINAMATH_CALUDE_weighted_average_is_38_5_l273_27320


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l273_27378

/-- Given a geometric sequence with four terms and common ratio 2,
    prove that (2a₁ + a₂) / (2a₃ + a₄) = 1/4 -/
theorem geometric_sequence_ratio (a₁ a₂ a₃ a₄ : ℝ) :
  a₂ = 2 * a₁ → a₃ = 2 * a₂ → a₄ = 2 * a₃ →
  (2 * a₁ + a₂) / (2 * a₃ + a₄) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l273_27378


namespace NUMINAMATH_CALUDE_haley_initial_trees_l273_27347

/-- The number of trees that died during the typhoon -/
def trees_died : ℕ := 2

/-- The number of trees left after the typhoon -/
def trees_left : ℕ := 10

/-- The initial number of trees before the typhoon -/
def initial_trees : ℕ := trees_left + trees_died

theorem haley_initial_trees : initial_trees = 12 := by
  sorry

end NUMINAMATH_CALUDE_haley_initial_trees_l273_27347


namespace NUMINAMATH_CALUDE_overlap_area_is_two_l273_27367

-- Define the 3x3 grid
def Grid := Fin 3 × Fin 3

-- Define the two quadrilaterals
def quad1 : List Grid := [(0,1), (1,2), (2,1), (1,0)]
def quad2 : List Grid := [(0,0), (2,2), (2,0), (0,2)]

-- Define the function to calculate the area of overlap
def overlapArea (q1 q2 : List Grid) : ℝ :=
  sorry  -- The actual calculation would go here

-- State the theorem
theorem overlap_area_is_two :
  overlapArea quad1 quad2 = 2 :=
sorry

end NUMINAMATH_CALUDE_overlap_area_is_two_l273_27367


namespace NUMINAMATH_CALUDE_diophantus_problem_l273_27395

theorem diophantus_problem (x y z t : ℤ) : 
  x = 11 ∧ y = 4 ∧ z = 7 ∧ t = 9 →
  x + y + z = 22 ∧
  x + y + t = 24 ∧
  x + z + t = 27 ∧
  y + z + t = 20 := by
sorry

end NUMINAMATH_CALUDE_diophantus_problem_l273_27395


namespace NUMINAMATH_CALUDE_average_customers_per_table_l273_27337

/-- Given a restaurant scenario with tables, women, and men, calculate the average number of customers per table. -/
theorem average_customers_per_table 
  (tables : ℝ) 
  (women : ℝ) 
  (men : ℝ) 
  (h_tables : tables = 9.0) 
  (h_women : women = 7.0) 
  (h_men : men = 3.0) : 
  (women + men) / tables = 10.0 / 9.0 := by
  sorry

end NUMINAMATH_CALUDE_average_customers_per_table_l273_27337


namespace NUMINAMATH_CALUDE_triangle_less_than_answer_l273_27387

theorem triangle_less_than_answer (triangle : ℝ) (answer : ℝ) 
  (h : 8.5 + triangle = 5.6 + answer) : triangle < answer := by
  sorry

end NUMINAMATH_CALUDE_triangle_less_than_answer_l273_27387


namespace NUMINAMATH_CALUDE_sin_2023pi_over_6_l273_27359

theorem sin_2023pi_over_6 : Real.sin (2023 * Real.pi / 6) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_2023pi_over_6_l273_27359


namespace NUMINAMATH_CALUDE_reinforcement_size_l273_27330

/-- Calculates the size of a reinforcement given initial garrison size, initial provisions duration,
    time passed before reinforcement, and remaining provisions duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                            (time_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions_left := initial_garrison * (initial_duration - time_passed)
  let total_men := provisions_left / remaining_duration
  total_men - initial_garrison

/-- Proves that given the specified conditions, the reinforcement size is 300 men. -/
theorem reinforcement_size :
  calculate_reinforcement 150 31 16 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l273_27330


namespace NUMINAMATH_CALUDE_inequality_proof_l273_27356

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^2 + x*y + y^2 ≤ 3*(x - Real.sqrt (x*y) + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l273_27356


namespace NUMINAMATH_CALUDE_marbles_problem_l273_27368

theorem marbles_problem (total : ℕ) (given_to_sister : ℕ) : 
  (given_to_sister = total / 6) →
  (given_to_sister = 9) →
  (total - (total / 2 + total / 6) = 18) :=
by sorry

end NUMINAMATH_CALUDE_marbles_problem_l273_27368


namespace NUMINAMATH_CALUDE_zach_needs_six_more_l273_27392

/-- Calculates how much more money Zach needs to buy a bike --/
def money_needed_for_bike (bike_cost allowance lawn_pay babysit_rate current_savings babysit_hours : ℕ) : ℕ :=
  let total_earnings := allowance + lawn_pay + babysit_rate * babysit_hours
  let total_savings := current_savings + total_earnings
  if total_savings ≥ bike_cost then 0
  else bike_cost - total_savings

/-- Proves that Zach needs $6 more to buy the bike --/
theorem zach_needs_six_more : 
  money_needed_for_bike 100 5 10 7 65 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_zach_needs_six_more_l273_27392


namespace NUMINAMATH_CALUDE_calculator_operations_l273_27334

def A (n : ℕ) : ℕ := 2 * n
def B (n : ℕ) : ℕ := n + 1

def applyKeys (n : ℕ) (keys : List (ℕ → ℕ)) : ℕ :=
  keys.foldl (fun acc f => f acc) n

theorem calculator_operations :
  (∃ keys : List (ℕ → ℕ), keys.length = 4 ∧ applyKeys 1 keys = 10) ∧
  (∃ keys : List (ℕ → ℕ), keys.length = 6 ∧ applyKeys 1 keys = 15) ∧
  (∃ keys : List (ℕ → ℕ), keys.length = 8 ∧ applyKeys 1 keys = 100) := by
  sorry

end NUMINAMATH_CALUDE_calculator_operations_l273_27334


namespace NUMINAMATH_CALUDE_town_neighborhoods_count_l273_27372

/-- Represents a town with neighborhoods and street lights -/
structure Town where
  total_lights : ℕ
  lights_per_road : ℕ
  roads_per_neighborhood : ℕ

/-- Calculates the number of neighborhoods in the town -/
def number_of_neighborhoods (t : Town) : ℕ :=
  (t.total_lights / t.lights_per_road) / t.roads_per_neighborhood

/-- Theorem: The number of neighborhoods in the given town is 10 -/
theorem town_neighborhoods_count :
  let t : Town := {
    total_lights := 20000,
    lights_per_road := 500,
    roads_per_neighborhood := 4
  }
  number_of_neighborhoods t = 10 := by
  sorry

end NUMINAMATH_CALUDE_town_neighborhoods_count_l273_27372


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l273_27348

/-- A hyperbola with equation x²/m - y²/4 = 1 and focal distance 6 has m = 5 -/
theorem hyperbola_focal_distance (m : ℝ) : 
  (∃ (x y : ℝ), x^2/m - y^2/4 = 1) →  -- Hyperbola equation
  (∃ (c : ℝ), c = 3) →                -- Focal distance is 6, so c = 3
  m = 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l273_27348


namespace NUMINAMATH_CALUDE_tangent_line_slope_l273_27346

theorem tangent_line_slope (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^4 + m*x
  let f' : ℝ → ℝ := λ x ↦ 4*x^3 + m
  let tangent_slope : ℝ := f' (-1)
  (2 * (-1) + f (-1) + 3 = 0) ∧ (tangent_slope = -2) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l273_27346


namespace NUMINAMATH_CALUDE_white_mailbox_houses_l273_27363

theorem white_mailbox_houses (total_mail : ℕ) (total_houses : ℕ) (red_houses : ℕ) (mail_per_house : ℕ)
  (h1 : total_mail = 48)
  (h2 : total_houses = 8)
  (h3 : red_houses = 3)
  (h4 : mail_per_house = 6) :
  total_houses - red_houses = 5 := by
sorry

end NUMINAMATH_CALUDE_white_mailbox_houses_l273_27363


namespace NUMINAMATH_CALUDE_ratio_proof_l273_27317

theorem ratio_proof (a b c d : ℝ) 
  (h1 : b / c = 13 / 9)
  (h2 : c / d = 5 / 13)
  (h3 : a / d = 1 / 7.2) :
  a / b = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_proof_l273_27317


namespace NUMINAMATH_CALUDE_sad_girls_count_l273_27390

theorem sad_girls_count (total_children happy_children sad_children neutral_children
                         boys girls happy_boys neutral_boys : ℕ) : 
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 19 →
  girls = 41 →
  happy_boys = 6 →
  neutral_boys = 7 →
  total_children = happy_children + sad_children + neutral_children →
  total_children = boys + girls →
  ∃ (sad_girls : ℕ), sad_girls = 4 ∧ sad_children = sad_girls + (boys - happy_boys - neutral_boys) :=
by sorry

end NUMINAMATH_CALUDE_sad_girls_count_l273_27390


namespace NUMINAMATH_CALUDE_preston_high_teachers_l273_27333

/-- Represents the number of students in Preston High School -/
def total_students : ℕ := 1500

/-- Represents the number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- Represents the number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- Represents the number of students in each class -/
def students_per_class : ℕ := 30

/-- Calculates the number of teachers at Preston High School -/
def number_of_teachers : ℕ :=
  (total_students * classes_per_student) / (students_per_class * classes_per_teacher)

/-- Theorem stating that the number of teachers at Preston High School is 60 -/
theorem preston_high_teachers :
  number_of_teachers = 60 := by sorry

end NUMINAMATH_CALUDE_preston_high_teachers_l273_27333


namespace NUMINAMATH_CALUDE_inequality_proof_l273_27374

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * (a + 1)) / (b + 1) + (b * (b + 1)) / (a + 1) ≥ a + b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l273_27374


namespace NUMINAMATH_CALUDE_unshaded_area_between_circles_l273_27352

/-- The area of the unshaded region between two concentric circles -/
theorem unshaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 7) :
  π * r₂^2 - π * r₁^2 = 33 * π :=
by sorry

end NUMINAMATH_CALUDE_unshaded_area_between_circles_l273_27352


namespace NUMINAMATH_CALUDE_two_derived_point_of_neg_two_three_original_point_from_three_derived_k_range_for_distance_condition_l273_27302

/-- Definition of k-derived point -/
def k_derived_point (k : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 + k * P.2, k * P.1 + P.2)

/-- Theorem 1: The 2-derived point of (-2,3) is (4, -1) -/
theorem two_derived_point_of_neg_two_three :
  k_derived_point 2 (-2, 3) = (4, -1) := by sorry

/-- Theorem 2: If the 3-derived point of P is (9,11), then P is (3,2) -/
theorem original_point_from_three_derived :
  ∀ P : ℝ × ℝ, k_derived_point 3 P = (9, 11) → P = (3, 2) := by sorry

/-- Theorem 3: For a point P(0,b) on the positive y-axis, its k-derived point P'(kb,b) 
    has |kb| ≥ 5b if and only if k ≥ 5 or k ≤ -5 -/
theorem k_range_for_distance_condition :
  ∀ k b : ℝ, b > 0 → (|k * b| ≥ 5 * b ↔ k ≥ 5 ∨ k ≤ -5) := by sorry

end NUMINAMATH_CALUDE_two_derived_point_of_neg_two_three_original_point_from_three_derived_k_range_for_distance_condition_l273_27302


namespace NUMINAMATH_CALUDE_eggs_taken_l273_27351

theorem eggs_taken (initial : ℕ) (remaining : ℕ) (taken : ℕ) : 
  initial = 47 → remaining = 42 → taken = initial - remaining → taken = 5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_taken_l273_27351


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l273_27325

/-- 
Given a pentagon where:
- Angles increase sequentially by 10 degrees
- The sum of all angles is 540 degrees
Prove that the largest angle is 128 degrees
-/
theorem pentagon_largest_angle : 
  ∀ (a₁ a₂ a₃ a₄ a₅ : ℝ),
  a₂ = a₁ + 10 →
  a₃ = a₁ + 20 →
  a₄ = a₁ + 30 →
  a₅ = a₁ + 40 →
  a₁ + a₂ + a₃ + a₄ + a₅ = 540 →
  a₅ = 128 := by
sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l273_27325


namespace NUMINAMATH_CALUDE_b_value_l273_27365

def p (x : ℝ) : ℝ := 3 * x - 8

def q (x b : ℝ) : ℝ := 4 * x - b

theorem b_value (b : ℝ) : p (q 3 b) = 10 → b = 6 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l273_27365


namespace NUMINAMATH_CALUDE_rhombus_common_area_l273_27341

/-- Represents a rhombus with given diagonal lengths -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculates the area of the common part of two rhombuses -/
def commonArea (r : Rhombus) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem: The area of the common part of two rhombuses is 9.6 cm² -/
theorem rhombus_common_area :
  let r := Rhombus.mk 4 6
  commonArea r = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_common_area_l273_27341


namespace NUMINAMATH_CALUDE_mingis_test_pages_l273_27344

/-- The number of pages in Mingi's math test -/
def pages_in_test (first_page last_page : ℕ) : ℕ :=
  last_page - first_page + 1

/-- Theorem stating the number of pages in Mingi's math test -/
theorem mingis_test_pages : pages_in_test 8 21 = 14 := by
  sorry

end NUMINAMATH_CALUDE_mingis_test_pages_l273_27344


namespace NUMINAMATH_CALUDE_prob_at_least_two_diff_fruits_l273_27389

/-- Represents the types of fruit Joe can choose from -/
inductive Fruit
  | apple
  | orange
  | banana
  | grape

/-- The probability of choosing a specific fruit -/
def fruit_prob (f : Fruit) : ℝ :=
  match f with
  | Fruit.apple => 0.4
  | Fruit.orange => 0.3
  | Fruit.banana => 0.2
  | Fruit.grape => 0.1

/-- The probability of choosing the same fruit for all three meals -/
def same_fruit_prob : ℝ :=
  (fruit_prob Fruit.apple) ^ 3 +
  (fruit_prob Fruit.orange) ^ 3 +
  (fruit_prob Fruit.banana) ^ 3 +
  (fruit_prob Fruit.grape) ^ 3

/-- Theorem: The probability of eating at least two different kinds of fruit in a day is 0.9 -/
theorem prob_at_least_two_diff_fruits :
  1 - same_fruit_prob = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_diff_fruits_l273_27389


namespace NUMINAMATH_CALUDE_equation_represents_statement_l273_27373

/-- Represents an unknown number -/
def n : ℤ := sorry

/-- The statement "a number increased by five equals 15" -/
def statement : Prop := n + 5 = 15

/-- Theorem stating that the equation correctly represents the given statement -/
theorem equation_represents_statement : statement ↔ n + 5 = 15 := by sorry

end NUMINAMATH_CALUDE_equation_represents_statement_l273_27373


namespace NUMINAMATH_CALUDE_derivative_of_f_l273_27371

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2

-- State the theorem
theorem derivative_of_f :
  deriv f = fun x ↦ 2 * x + 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l273_27371


namespace NUMINAMATH_CALUDE_tetrahedron_edge_ratio_l273_27386

-- Define a tetrahedron type
structure Tetrahedron where
  -- We don't need to define the specific properties here
  mk ::

-- Define a property for similar right-angled triangular faces
def has_similar_right_angled_faces (t : Tetrahedron) : Prop :=
  sorry

-- Define the ratio of longest to shortest edge
def longest_to_shortest_ratio (t : Tetrahedron) : ℝ :=
  sorry

-- Theorem statement
theorem tetrahedron_edge_ratio 
  (t : Tetrahedron) 
  (h : has_similar_right_angled_faces t) : 
  longest_to_shortest_ratio t = Real.sqrt ((1 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_ratio_l273_27386


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l273_27322

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l273_27322


namespace NUMINAMATH_CALUDE_netGoalsForTimesMiddleSchool_l273_27316

/-- Calculates the net goals for a single match -/
def netGoals (goalsFor goalsAgainst : ℤ) : ℤ := goalsFor - goalsAgainst

/-- Represents the scores of three soccer matches -/
structure ThreeMatches where
  match1 : (ℤ × ℤ)
  match2 : (ℤ × ℤ)
  match3 : (ℤ × ℤ)

/-- The specific scores for the Times Middle School soccer team -/
def timesMiddleSchoolScores : ThreeMatches := {
  match1 := (5, 3)
  match2 := (2, 6)
  match3 := (2, 2)
}

/-- Theorem stating that the net number of goals for the given scores is -2 -/
theorem netGoalsForTimesMiddleSchool :
  (netGoals timesMiddleSchoolScores.match1.1 timesMiddleSchoolScores.match1.2) +
  (netGoals timesMiddleSchoolScores.match2.1 timesMiddleSchoolScores.match2.2) +
  (netGoals timesMiddleSchoolScores.match3.1 timesMiddleSchoolScores.match3.2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_netGoalsForTimesMiddleSchool_l273_27316


namespace NUMINAMATH_CALUDE_book_selection_combinations_l273_27345

/-- The number of ways to choose one book from each of three genres -/
def book_combinations (mystery_count : ℕ) (fantasy_count : ℕ) (biography_count : ℕ) : ℕ :=
  mystery_count * fantasy_count * biography_count

/-- Theorem: Given 4 mystery novels, 3 fantasy novels, and 3 biographies,
    the number of ways to choose one book from each genre is 36 -/
theorem book_selection_combinations :
  book_combinations 4 3 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_combinations_l273_27345


namespace NUMINAMATH_CALUDE_arcsin_sum_equals_pi_over_four_l273_27305

theorem arcsin_sum_equals_pi_over_four :
  Real.arcsin (1 / Real.sqrt 10) + Real.arcsin (1 / Real.sqrt 26) + 
  Real.arcsin (1 / Real.sqrt 50) + Real.arcsin (1 / Real.sqrt 65) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sum_equals_pi_over_four_l273_27305


namespace NUMINAMATH_CALUDE_consecutive_integers_product_336_sum_21_l273_27307

theorem consecutive_integers_product_336_sum_21 :
  ∃ (n : ℤ), (n - 1) * n * (n + 1) = 336 → (n - 1) + n + (n + 1) = 21 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_336_sum_21_l273_27307


namespace NUMINAMATH_CALUDE_count_special_numbers_is_792_l273_27366

/-- A function that counts the number of 5-digit numbers beginning with 2 
    and having exactly three identical digits -/
def count_special_numbers : ℕ :=
  let count_with_three_twos := 6 * 9 * 8
  let count_with_three_non_twos := 5 * 8 * 9
  count_with_three_twos + count_with_three_non_twos

/-- Theorem stating that the count of special numbers is 792 -/
theorem count_special_numbers_is_792 : count_special_numbers = 792 := by
  sorry

#eval count_special_numbers

end NUMINAMATH_CALUDE_count_special_numbers_is_792_l273_27366


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tan_a7_l273_27355

theorem arithmetic_sequence_tan_a7 (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 9 = 8 * Real.pi / 3 →                         -- given condition
  Real.tan (a 7) = Real.sqrt 3 :=                       -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tan_a7_l273_27355


namespace NUMINAMATH_CALUDE_find_divisor_l273_27357

theorem find_divisor (dividend : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 76)
  (h2 : remainder = 8)
  (h3 : quotient = 4)
  : ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 17 :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l273_27357


namespace NUMINAMATH_CALUDE_negative_integer_sum_and_square_is_fifteen_l273_27376

theorem negative_integer_sum_and_square_is_fifteen (N : ℤ) : 
  N < 0 → N^2 + N = 15 → N = -5 := by sorry

end NUMINAMATH_CALUDE_negative_integer_sum_and_square_is_fifteen_l273_27376


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l273_27328

theorem tennis_tournament_matches (total_players : Nat) (seeded_players : Nat) : 
  total_players = 128 → seeded_players = 32 → total_players - 1 = 127 :=
by
  sorry

#check tennis_tournament_matches

end NUMINAMATH_CALUDE_tennis_tournament_matches_l273_27328


namespace NUMINAMATH_CALUDE_remove_four_for_target_average_l273_27391

def original_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
def target_average : ℚ := 63/10

theorem remove_four_for_target_average :
  let remaining_list := original_list.filter (· ≠ 4)
  (remaining_list.sum : ℚ) / remaining_list.length = target_average := by
  sorry

end NUMINAMATH_CALUDE_remove_four_for_target_average_l273_27391


namespace NUMINAMATH_CALUDE_dividend_calculation_l273_27311

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 10) :
  divisor * quotient + remainder = 163 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l273_27311


namespace NUMINAMATH_CALUDE_parallel_vectors_m_l273_27343

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m (a b : ℝ × ℝ) (h : vector_parallel a b) :
  a = (2, -1) → b = (-1, 1/2) := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_l273_27343


namespace NUMINAMATH_CALUDE_candy_problem_l273_27300

/-- The number of candy pieces remaining in a bowl after some are taken. -/
def remaining_candy (initial : ℕ) (taken_by_talitha : ℕ) (taken_by_solomon : ℕ) : ℕ :=
  initial - (taken_by_talitha + taken_by_solomon)

/-- Theorem stating that given the initial amount and amounts taken by Talitha and Solomon,
    the remaining candy pieces are 88. -/
theorem candy_problem :
  remaining_candy 349 108 153 = 88 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l273_27300


namespace NUMINAMATH_CALUDE_shaded_area_is_nine_l273_27303

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the spinner shape -/
structure Spinner where
  center : Point
  armLength : ℕ

/-- Represents the entire shaded shape -/
structure ShadedShape where
  spinner : Spinner
  cornerSquares : List Point

/-- Calculates the area of the shaded shape -/
def shadedArea (shape : ShadedShape) : ℕ :=
  let spinnerArea := 2 * shape.spinner.armLength * 2 + 1
  let cornerSquaresArea := shape.cornerSquares.length
  spinnerArea + cornerSquaresArea

/-- The theorem to be proved -/
theorem shaded_area_is_nine :
  ∀ (shape : ShadedShape),
    shape.spinner.center = ⟨3, 3⟩ →
    shape.spinner.armLength = 1 →
    shape.cornerSquares.length = 4 →
    shadedArea shape = 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_nine_l273_27303


namespace NUMINAMATH_CALUDE_min_value_expression_l273_27327

theorem min_value_expression (x y : ℝ) : 
  ∃ (a b : ℝ), (x * y + 1)^2 + (x + y + 1)^2 ≥ 0 ∧ (a * b + 1)^2 + (a + b + 1)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l273_27327


namespace NUMINAMATH_CALUDE_restaurant_budget_theorem_l273_27399

theorem restaurant_budget_theorem (budget : ℝ) (budget_positive : budget > 0) :
  let rent := (1 / 4) * budget
  let remaining := budget - rent
  let food_and_beverages := (1 / 4) * remaining
  (food_and_beverages / budget) * 100 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_budget_theorem_l273_27399


namespace NUMINAMATH_CALUDE_unique_number_with_equal_sums_l273_27313

theorem unique_number_with_equal_sums : 
  ∃! n : ℕ, 
    (n ≥ 10000) ∧ 
    (n % 10000 = 9876) ∧ 
    (n / 10000 + 9876 = n / 1000 + 876) ∧
    (n = 9999876) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_equal_sums_l273_27313


namespace NUMINAMATH_CALUDE_stability_ratio_calculation_l273_27388

theorem stability_ratio_calculation (T H L : ℚ) : 
  T = 3 → H = 9 → L = (30 * T^3) / H^3 → L = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_stability_ratio_calculation_l273_27388


namespace NUMINAMATH_CALUDE_democrat_ratio_l273_27318

theorem democrat_ratio (total_participants male_participants female_participants male_democrats female_democrats : ℕ) : 
  total_participants = 990 →
  female_participants = 330 →
  male_participants = 660 →
  female_democrats = 165 →
  male_democrats + female_democrats = total_participants / 3 →
  male_democrats = male_participants / 4 :=
by sorry

end NUMINAMATH_CALUDE_democrat_ratio_l273_27318


namespace NUMINAMATH_CALUDE_ellipse_to_hyperbola_l273_27397

theorem ellipse_to_hyperbola (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : b = Real.sqrt 3 * c) 
  (h4 : a + c = 3 * Real.sqrt 3) 
  (h5 : a^2 = b^2 + c^2) :
  ∃ (x y : ℝ), y^2 / 12 - x^2 / 9 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_to_hyperbola_l273_27397


namespace NUMINAMATH_CALUDE_series_sum_equals_half_l273_27301

/-- The sum of the series Σ(2^n / (3^(2^n) + 1)) from n = 0 to infinity is equal to 1/2 -/
theorem series_sum_equals_half :
  ∑' n : ℕ, (2 : ℝ)^n / (3^(2^n) + 1) = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_half_l273_27301


namespace NUMINAMATH_CALUDE_particle_movement_l273_27332

/-- Represents a particle in a 2D grid -/
structure Particle where
  x : ℚ
  y : ℚ

/-- Represents the probabilities of movement for Particle A -/
structure ProbA where
  left : ℚ
  right : ℚ
  up : ℚ
  down : ℚ

/-- Represents the probability of movement for Particle B -/
def ProbB : ℚ → Prop := λ y ↦ ∀ (direction : Fin 4), y = 1/4

/-- The theorem statement -/
theorem particle_movement 
  (A : Particle) 
  (B : Particle) 
  (probA : ProbA) 
  (probB : ℚ → Prop) :
  A.x = 0 ∧ A.y = 0 ∧
  B.x = 1 ∧ B.y = 1 ∧
  probA.left = 1/4 ∧ probA.right = 1/4 ∧ probA.up = 1/3 ∧
  ProbB probA.down ∧
  (∃ (x : ℚ), probA.down = x ∧ x + 1/4 + 1/4 + 1/3 = 1) →
  probA.down = 1/6 ∧
  ProbB (1/4) ∧
  (∃ (t : ℕ), t = 3 ∧ 
    (∀ (t' : ℕ), (∃ (A' B' : Particle), A'.x = 2 ∧ A'.y = 1 ∧ B'.x = 2 ∧ B'.y = 1) → t' ≥ t)) ∧
  (9 : ℚ)/1024 = 
    (3 * (1/4)^2 * 1/3) * -- Probability for A
    (1/4 * 3 * (1/4)^2)   -- Probability for B
  := by sorry

end NUMINAMATH_CALUDE_particle_movement_l273_27332


namespace NUMINAMATH_CALUDE_inequality_proof_l273_27312

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xz : x * z = 1)
  (h_x_1z : x * (1 + z) > 1)
  (h_y_1x : y * (1 + x) > 1)
  (h_z_1y : z * (1 + y) > 1) :
  2 * (x + y + z) ≥ -1/x + 1/y + 1/z + 3 := by
sorry


end NUMINAMATH_CALUDE_inequality_proof_l273_27312


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l273_27394

theorem polynomial_division_quotient (z : ℝ) : 
  4 * z^5 - 3 * z^4 + 2 * z^3 - 5 * z^2 + 7 * z - 3 = 
  (z + 2) * (4 * z^4 - 11 * z^3 + 24 * z^2 - 53 * z + 113) + (-229) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l273_27394


namespace NUMINAMATH_CALUDE_smallest_integer_in_ratio_l273_27369

theorem smallest_integer_in_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 60 →
  2 * a = 3 * b →
  3 * b = 5 * c →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_in_ratio_l273_27369


namespace NUMINAMATH_CALUDE_equality_condition_l273_27382

theorem equality_condition (a b c : ℝ) : 
  a + 2*b*c = (a + 2*b)*(a + 2*c) ↔ a + 2*b + 2*c = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l273_27382


namespace NUMINAMATH_CALUDE_absolute_value_calculation_l273_27353

theorem absolute_value_calculation : |(-6)| - (-4) + (-7) = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_calculation_l273_27353


namespace NUMINAMATH_CALUDE_division_simplification_l273_27398

theorem division_simplification (x : ℝ) (hx : x ≠ 0) :
  (1 + 1/x) / ((x^2 + x)/x) = 1/x := by sorry

end NUMINAMATH_CALUDE_division_simplification_l273_27398


namespace NUMINAMATH_CALUDE_charles_chocolate_milk_l273_27364

/-- The amount of chocolate milk Charles can drink given his supplies -/
def chocolate_milk_total (milk_per_glass : ℚ) (syrup_per_glass : ℚ) 
  (total_milk : ℚ) (total_syrup : ℚ) : ℚ :=
  let glasses_from_milk := total_milk / milk_per_glass
  let glasses_from_syrup := total_syrup / syrup_per_glass
  let glasses := min glasses_from_milk glasses_from_syrup
  glasses * (milk_per_glass + syrup_per_glass)

/-- Theorem stating that Charles will drink 160 ounces of chocolate milk -/
theorem charles_chocolate_milk :
  chocolate_milk_total (6.5) (1.5) (130) (60) = 160 := by
  sorry

end NUMINAMATH_CALUDE_charles_chocolate_milk_l273_27364


namespace NUMINAMATH_CALUDE_divisibility_of_difference_quotient_l273_27383

theorem divisibility_of_difference_quotient (a b n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0) 
  (h_div : ∃ k : ℤ, a^n - b^n = n * k) :
  ∃ m : ℤ, (a^n - b^n) / (a - b) = n * m := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_difference_quotient_l273_27383


namespace NUMINAMATH_CALUDE_greatest_difference_units_digit_l273_27342

theorem greatest_difference_units_digit (x : ℕ) :
  x < 10 →
  (840 + x) % 3 = 0 →
  ∃ y, y < 10 ∧ (840 + y) % 3 = 0 ∧ 
  ∀ z, z < 10 → (840 + z) % 3 = 0 → (max x y - min x y) ≥ (max x z - min x z) :=
by sorry

end NUMINAMATH_CALUDE_greatest_difference_units_digit_l273_27342


namespace NUMINAMATH_CALUDE_subset_condition_empty_intersection_condition_l273_27349

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 2*a + 3}

-- Theorem for the subset condition
theorem subset_condition (a : ℝ) : 
  A ⊆ B a ↔ a ∈ Set.Icc (-1/2) 0 :=
sorry

-- Theorem for the empty intersection condition
theorem empty_intersection_condition (a : ℝ) :
  A ∩ B a = ∅ ↔ a ∈ Set.Iic (-2) ∪ Set.Ici (3/2) :=
sorry

end NUMINAMATH_CALUDE_subset_condition_empty_intersection_condition_l273_27349


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l273_27308

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  -- Right triangle condition
  a^2 + b^2 = c^2 →
  -- Perimeter condition
  a + b + c = 32 →
  -- Area condition
  (1/2) * a * b = 20 →
  -- Conclusion: hypotenuse length
  c = 59/4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l273_27308


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l273_27370

theorem system_of_equations_solution : ∀ x y : ℚ,
  (6 * x - 48 * y = 2) ∧ (3 * y - x = 4) →
  x^2 + y^2 = 442 / 25 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l273_27370


namespace NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l273_27336

/-- Returns true if the given natural number contains the digit 7 -/
def containsSeven (n : ℕ) : Prop :=
  ∃ k : ℕ, n / (10 ^ k) % 10 = 7

theorem smallest_n_with_seven_in_squares : 
  ∀ n : ℕ, n < 26 → ¬(containsSeven (n^2) ∧ containsSeven ((n+1)^2)) ∧
  (containsSeven (26^2) ∧ containsSeven (27^2)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l273_27336


namespace NUMINAMATH_CALUDE_book_club_task_distribution_l273_27310

theorem book_club_task_distribution (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_club_task_distribution_l273_27310


namespace NUMINAMATH_CALUDE_parabola_point_l273_27329

/-- Given a point P (x₀, y₀) on the parabola y = 3x² with derivative 6 at x₀, prove P = (1, 3) -/
theorem parabola_point (x₀ y₀ : ℝ) : 
  y₀ = 3 * x₀^2 →                   -- Point P lies on the parabola
  (6 : ℝ) = 6 * x₀ →                -- Derivative at x₀ is 6
  (x₀, y₀) = (1, 3) :=              -- Conclusion: P = (1, 3)
by sorry

end NUMINAMATH_CALUDE_parabola_point_l273_27329


namespace NUMINAMATH_CALUDE_definite_integral_equals_26_over_3_l273_27335

theorem definite_integral_equals_26_over_3 : 
  ∫ x in (0)..(2 * Real.arctan (1/2)), (1 + Real.sin x) / ((1 - Real.sin x)^2) = 26/3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_equals_26_over_3_l273_27335


namespace NUMINAMATH_CALUDE_power_of_six_l273_27393

theorem power_of_six : (6 : ℕ) ^ ((6 : ℕ) / 2) = 216 := by sorry

end NUMINAMATH_CALUDE_power_of_six_l273_27393


namespace NUMINAMATH_CALUDE_min_rental_cost_l273_27339

/-- Represents the number of buses of type A -/
def x : ℕ := sorry

/-- Represents the number of buses of type B -/
def y : ℕ := sorry

/-- The total number of passengers -/
def total_passengers : ℕ := 900

/-- The capacity of a type A bus -/
def capacity_A : ℕ := 36

/-- The capacity of a type B bus -/
def capacity_B : ℕ := 60

/-- The rental cost of a type A bus -/
def cost_A : ℕ := 1600

/-- The rental cost of a type B bus -/
def cost_B : ℕ := 2400

/-- The maximum total number of buses allowed -/
def max_buses : ℕ := 21

theorem min_rental_cost :
  (∃ x y : ℕ,
    x * capacity_A + y * capacity_B ≥ total_passengers ∧
    x + y ≤ max_buses ∧
    y ≤ x + 7 ∧
    ∀ a b : ℕ,
      (a * capacity_A + b * capacity_B ≥ total_passengers ∧
       a + b ≤ max_buses ∧
       b ≤ a + 7) →
      x * cost_A + y * cost_B ≤ a * cost_A + b * cost_B) ∧
  (∀ x y : ℕ,
    x * capacity_A + y * capacity_B ≥ total_passengers ∧
    x + y ≤ max_buses ∧
    y ≤ x + 7 →
    x * cost_A + y * cost_B ≥ 36800) :=
by sorry

end NUMINAMATH_CALUDE_min_rental_cost_l273_27339


namespace NUMINAMATH_CALUDE_angle_complement_relation_l273_27377

theorem angle_complement_relation (x : ℝ) : x = 70 → x = 2 * (90 - x) + 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_relation_l273_27377


namespace NUMINAMATH_CALUDE_enfeoffment_probability_l273_27381

/-- The number of nobility levels in ancient China --/
def nobility_levels : ℕ := 5

/-- The probability that two people are not enfeoffed at the same level --/
def prob_different_levels : ℚ := 4/5

/-- Theorem stating the probability of two people being enfeoffed at different levels --/
theorem enfeoffment_probability :
  (1 : ℚ) - (nobility_levels : ℚ) / (nobility_levels^2 : ℚ) = prob_different_levels :=
by sorry

end NUMINAMATH_CALUDE_enfeoffment_probability_l273_27381


namespace NUMINAMATH_CALUDE_farmer_animals_count_l273_27379

/-- Represents the number of animals a farmer has -/
structure FarmAnimals where
  goats : ℕ
  cows : ℕ
  pigs : ℕ

/-- Calculates the total number of animals -/
def totalAnimals (animals : FarmAnimals) : ℕ :=
  animals.goats + animals.cows + animals.pigs

/-- Theorem stating the total number of animals given the conditions -/
theorem farmer_animals_count :
  ∀ (animals : FarmAnimals),
    animals.goats = 11 →
    animals.cows = animals.goats + 4 →
    animals.pigs = 2 * animals.cows →
    totalAnimals animals = 56 := by
  sorry

end NUMINAMATH_CALUDE_farmer_animals_count_l273_27379


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_expression_l273_27384

/-- Given vectors a and b that are parallel, prove that the given expression equals 3√2 -/
theorem parallel_vectors_trig_expression (x : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (ha : a = (Real.sin x, 2)) 
  (hb : b = (Real.cos x, 1)) 
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  (2 * Real.sin (x + π/4)) / (Real.sin x - Real.cos x) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_expression_l273_27384


namespace NUMINAMATH_CALUDE_square_plus_double_sqrt2_minus_1_l273_27331

theorem square_plus_double_sqrt2_minus_1 :
  let x : ℝ := Real.sqrt 2 - 1
  x^2 + 2*x = 1 := by sorry

end NUMINAMATH_CALUDE_square_plus_double_sqrt2_minus_1_l273_27331


namespace NUMINAMATH_CALUDE_novel_pages_calculation_l273_27375

/-- Calculates the total number of pages in a novel based on a specific reading pattern -/
theorem novel_pages_calculation : 
  let first_four_days := 4
  let next_two_days := 2
  let last_day := 1
  let pages_per_day_first_four := 42
  let pages_per_day_next_two := 50
  let pages_last_day := 30
  
  (first_four_days * pages_per_day_first_four) + 
  (next_two_days * pages_per_day_next_two) + 
  pages_last_day = 298 := by
  sorry

end NUMINAMATH_CALUDE_novel_pages_calculation_l273_27375


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l273_27309

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, mx^2 - mx + 2 = 0 ∧ (∀ y : ℝ, my^2 - my + 2 = 0 → y = x)) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l273_27309


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l273_27306

/-- An arithmetic sequence with specific conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a4 : a 4 = 7)
  (h_sum : a 3 + a 6 = 16)
  (h_an : ∃ n : ℕ, a n = 31) :
  ∃ n : ℕ, a n = 31 ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l273_27306


namespace NUMINAMATH_CALUDE_leftover_coin_value_l273_27321

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The number of quarters in a complete roll -/
def quarters_per_roll : ℕ := 35

/-- The number of dimes in a complete roll -/
def dimes_per_roll : ℕ := 55

/-- James' quarters -/
def james_quarters : ℕ := 97

/-- James' dimes -/
def james_dimes : ℕ := 173

/-- Lindsay's quarters -/
def lindsay_quarters : ℕ := 141

/-- Lindsay's dimes -/
def lindsay_dimes : ℕ := 289

/-- The total number of quarters -/
def total_quarters : ℕ := james_quarters + lindsay_quarters

/-- The total number of dimes -/
def total_dimes : ℕ := james_dimes + lindsay_dimes

theorem leftover_coin_value :
  (total_quarters % quarters_per_roll : ℚ) * quarter_value +
  (total_dimes % dimes_per_roll : ℚ) * dime_value = 92 / 10 := by
  sorry

end NUMINAMATH_CALUDE_leftover_coin_value_l273_27321


namespace NUMINAMATH_CALUDE_product_units_digit_in_base_6_l273_27323

/-- The units digit of the product of two numbers in a given base -/
def unitsDigitInBase (a b : ℕ) (base : ℕ) : ℕ :=
  (a * b) % base

/-- 314 in base 10 -/
def num1 : ℕ := 314

/-- 59 in base 10 -/
def num2 : ℕ := 59

/-- The base we're converting to -/
def targetBase : ℕ := 6

theorem product_units_digit_in_base_6 :
  unitsDigitInBase num1 num2 targetBase = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_units_digit_in_base_6_l273_27323


namespace NUMINAMATH_CALUDE_function_has_period_two_l273_27340

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_unit_shift (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = f (x - 1)

def matches_exp_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = Real.exp (Real.log 2 * x)

-- State the theorem
theorem function_has_period_two (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : has_unit_shift f) 
  (h3 : matches_exp_on_unit_interval f) : 
  ∀ x, f (x + 2) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_has_period_two_l273_27340


namespace NUMINAMATH_CALUDE_smallest_possible_value_l273_27396

theorem smallest_possible_value (x : ℕ+) (a b : ℕ+) : 
  (Nat.gcd a b = x + 2) →
  (Nat.lcm a b = x * (x + 2)) →
  (a = 24) →
  (∀ c : ℕ+, c < b → ¬(Nat.gcd 24 c = x + 2 ∧ Nat.lcm 24 c = x * (x + 2))) →
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_possible_value_l273_27396


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l273_27380

theorem quadratic_form_k_value :
  ∀ (a h k : ℝ), (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l273_27380


namespace NUMINAMATH_CALUDE_fraction_meaningful_l273_27362

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l273_27362


namespace NUMINAMATH_CALUDE_octal_perfect_square_last_digit_l273_27385

/-- A perfect square in octal form (abc)₈ where a ≠ 0 always has c = 1 -/
theorem octal_perfect_square_last_digit (a b c : Nat) (h1 : a ≠ 0) 
  (h2 : ∃ (n : Nat), n^2 = a * 8^2 + b * 8 + c) : c = 1 := by
  sorry

end NUMINAMATH_CALUDE_octal_perfect_square_last_digit_l273_27385


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l273_27360

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
    a < b →
    (Real.sqrt (1 + Real.sqrt (25 + 20 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b) ↔
    (a = 1 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l273_27360


namespace NUMINAMATH_CALUDE_range_of_a_characterize_solution_set_l273_27315

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 + x - a

-- Part 1
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f a x > x^2 + a*x - 1 - a) → a ∈ Set.Icc 1 5 := by sorry

-- Part 2
-- We define a function that characterizes the solution set based on 'a'
noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a > 0 then {x | x < -(a+1)/a ∨ x > 1}
  else if a = 0 then {x | x > 1}
  else if -1/2 < a ∧ a < 0 then {x | 1 < x ∧ x < -(a+1)/a}
  else if a = -1/2 then ∅
  else {x | -(a+1)/a < x ∧ x < 1}

theorem characterize_solution_set (a : ℝ) :
  {x : ℝ | f a x > 1} = solution_set a := by sorry

end NUMINAMATH_CALUDE_range_of_a_characterize_solution_set_l273_27315


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l273_27358

theorem arithmetic_sequence_squares (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_arithmetic : ∃ d : ℝ, b / (c + a) - a / (b + c) = d ∧ c / (a + b) - b / (c + a) = d) :
  ∃ d' : ℝ, b^2 - a^2 = d' ∧ c^2 - b^2 = d' :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l273_27358
