import Mathlib

namespace NUMINAMATH_CALUDE_distribute_students_count_l1425_142523

/-- The number of ways to distribute 5 students into 3 groups -/
def distribute_students : ℕ :=
  let n : ℕ := 5  -- Total number of students
  let k : ℕ := 3  -- Number of groups
  let min_a : ℕ := 2  -- Minimum number of students in Group A
  let min_bc : ℕ := 1  -- Minimum number of students in Groups B and C
  sorry

/-- Theorem stating that the number of distribution schemes is 80 -/
theorem distribute_students_count : distribute_students = 80 := by
  sorry

end NUMINAMATH_CALUDE_distribute_students_count_l1425_142523


namespace NUMINAMATH_CALUDE_extreme_point_of_f_l1425_142547

/-- The function f(x) = 3/2 * x^2 - ln(x) for x > 0 has an extreme point at x = √3/3 -/
theorem extreme_point_of_f (x : ℝ) (h : x > 0) : 
  let f := fun (x : ℝ) => 3/2 * x^2 - Real.log x
  ∃ (c : ℝ), c = Real.sqrt 3 / 3 ∧ 
    (∀ y > 0, f y ≥ f c) ∨ (∀ y > 0, f y ≤ f c) := by
  sorry


end NUMINAMATH_CALUDE_extreme_point_of_f_l1425_142547


namespace NUMINAMATH_CALUDE_domain_intersection_complement_l1425_142572

-- Define the universal set as real numbers
def U : Type := ℝ

-- Define the function f(x) = ln(1-x)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x)

-- Define the domain M of f
def M : Set ℝ := {x | x < 1}

-- Define the set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem domain_intersection_complement :
  M ∩ (Set.univ \ N) = Set.Iic 0 :=
sorry

end NUMINAMATH_CALUDE_domain_intersection_complement_l1425_142572


namespace NUMINAMATH_CALUDE_eleven_million_nine_hundred_thousand_scientific_notation_l1425_142540

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem eleven_million_nine_hundred_thousand_scientific_notation :
  toScientificNotation 11090000 = ScientificNotation.mk 1.109 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_eleven_million_nine_hundred_thousand_scientific_notation_l1425_142540


namespace NUMINAMATH_CALUDE_inequality_solution_l1425_142586

theorem inequality_solution (x : ℝ) : (x - 5) / ((x - 2) * (x^2 - 1)) < 0 ↔ x < -1 ∨ (1 < x ∧ x < 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1425_142586


namespace NUMINAMATH_CALUDE_function_properties_l1425_142596

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x
noncomputable def g (x : ℝ) : ℝ := x^3

theorem function_properties :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≤ f (Real.exp 1)) ∧
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f 1 ≤ f x) ∧
  (∀ x ≥ 1, f x ≤ g x) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1425_142596


namespace NUMINAMATH_CALUDE_empire_state_building_race_l1425_142510

-- Define the total number of steps
def total_steps : ℕ := 1576

-- Define the total time in seconds
def total_time_seconds : ℕ := 11 * 60 + 57

-- Define the function to calculate steps per minute
def steps_per_minute (steps : ℕ) (time_seconds : ℕ) : ℚ :=
  (steps : ℚ) / ((time_seconds : ℚ) / 60)

-- Theorem statement
theorem empire_state_building_race :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |steps_per_minute total_steps total_time_seconds - 130| < ε :=
sorry

end NUMINAMATH_CALUDE_empire_state_building_race_l1425_142510


namespace NUMINAMATH_CALUDE_negative_quadratic_inequality_l1425_142513

/-- A quadratic polynomial ax^2 + bx + c that is negative for all real x -/
structure NegativeQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  is_negative : ∀ x : ℝ, a * x^2 + b * x + c < 0

/-- Theorem: For a negative quadratic polynomial, b/a < c/a + 1 -/
theorem negative_quadratic_inequality (q : NegativeQuadratic) : q.b / q.a < q.c / q.a + 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_quadratic_inequality_l1425_142513


namespace NUMINAMATH_CALUDE_sequence_difference_theorem_l1425_142562

def is_valid_sequence (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧ 
  (∀ n, x n < x (n + 1)) ∧
  (∀ n, x (2 * n + 1) ≤ 2 * n)

theorem sequence_difference_theorem (x : ℕ → ℕ) (h : is_valid_sequence x) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
sorry

end NUMINAMATH_CALUDE_sequence_difference_theorem_l1425_142562


namespace NUMINAMATH_CALUDE_books_taken_out_on_tuesday_l1425_142579

/-- Prove that the number of books taken out on Tuesday is 120, given the initial and final number of books in the library and the changes on Wednesday and Thursday. -/
theorem books_taken_out_on_tuesday (initial_books : ℕ) (final_books : ℕ) (returned_wednesday : ℕ) (withdrawn_thursday : ℕ) 
  (h_initial : initial_books = 250)
  (h_final : final_books = 150)
  (h_wednesday : returned_wednesday = 35)
  (h_thursday : withdrawn_thursday = 15) :
  initial_books - final_books + returned_wednesday - withdrawn_thursday = 120 := by
  sorry

end NUMINAMATH_CALUDE_books_taken_out_on_tuesday_l1425_142579


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l1425_142505

theorem subtraction_of_fractions : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l1425_142505


namespace NUMINAMATH_CALUDE_equal_area_if_equal_midpoints_l1425_142593

/-- A polygon with an even number of sides -/
structure EvenPolygon where
  vertices : List (ℝ × ℝ)
  even_sides : Even vertices.length

/-- The midpoints of the sides of a polygon -/
def midpoints (p : EvenPolygon) : List (ℝ × ℝ) :=
  sorry

/-- The area of a polygon -/
def area (p : EvenPolygon) : ℝ :=
  sorry

/-- Theorem: If two even-sided polygons have the same midpoints, their areas are equal -/
theorem equal_area_if_equal_midpoints (p q : EvenPolygon) 
  (h : midpoints p = midpoints q) : area p = area q :=
  sorry

end NUMINAMATH_CALUDE_equal_area_if_equal_midpoints_l1425_142593


namespace NUMINAMATH_CALUDE_school_transfer_percentage_l1425_142518

theorem school_transfer_percentage :
  ∀ (total_students : ℝ) (school_A_percentage : ℝ) (school_B_percentage : ℝ)
    (transfer_A_to_C_percentage : ℝ) (transfer_B_to_C_percentage : ℝ),
  school_A_percentage = 60 →
  school_B_percentage = 100 - school_A_percentage →
  transfer_A_to_C_percentage = 30 →
  transfer_B_to_C_percentage = 40 →
  let students_A := total_students * (school_A_percentage / 100)
  let students_B := total_students * (school_B_percentage / 100)
  let students_C := (students_A * (transfer_A_to_C_percentage / 100)) +
                    (students_B * (transfer_B_to_C_percentage / 100))
  (students_C / total_students) * 100 = 34 :=
by sorry

end NUMINAMATH_CALUDE_school_transfer_percentage_l1425_142518


namespace NUMINAMATH_CALUDE_volunteer_allocation_schemes_l1425_142534

/-- The number of ways to allocate volunteers to projects -/
def allocate_volunteers (n_volunteers : ℕ) (n_projects : ℕ) : ℕ :=
  (n_volunteers.choose 2) * (n_projects.factorial)

/-- Theorem stating that allocating 5 volunteers to 4 projects results in 240 schemes -/
theorem volunteer_allocation_schemes :
  allocate_volunteers 5 4 = 240 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_allocation_schemes_l1425_142534


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l1425_142511

-- Define the functions
def p (x : ℝ) : ℝ := x^2 - 4*x + 3
def q (x : ℝ) : ℝ := -p x + 2
def r (x : ℝ) : ℝ := p (-x)

-- Define the number of intersection points
def c : ℕ := 2  -- Number of intersections between p and q
def d : ℕ := 1  -- Number of intersections between p and r

-- Theorem statement
theorem intersection_points_theorem :
  (∀ x : ℝ, p x = q x → x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) ∧
  (∀ x : ℝ, p x = r x → x = 0) ∧
  (10 * c + d = 21) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l1425_142511


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1425_142585

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, 1 < x ∧ x < 2 → Real.log x < 1) ∧
  (∃ x : ℝ, Real.log x < 1 ∧ ¬(1 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1425_142585


namespace NUMINAMATH_CALUDE_circle_equation_perpendicular_chord_values_l1425_142533

-- Define the circle
def circle_center : ℝ × ℝ := (2, 0)
def circle_radius : ℝ := 5

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4 * x + 3 * y - 33 = 0

-- Define the intersecting line
def intersecting_line (a x y : ℝ) : Prop := a * x - y - 7 = 0

-- Theorem for the circle equation
theorem circle_equation : 
  ∀ x y : ℝ, (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 ↔ 
  (x - 2)^2 + y^2 = 25 := by sorry

-- Theorem for the values of a
theorem perpendicular_chord_values (a : ℝ) :
  (∃ A B : ℝ × ℝ, 
    (A.1 - circle_center.1)^2 + (A.2 - circle_center.2)^2 = circle_radius^2 ∧
    (B.1 - circle_center.1)^2 + (B.2 - circle_center.2)^2 = circle_radius^2 ∧
    intersecting_line a A.1 A.2 ∧
    intersecting_line a B.1 B.2 ∧
    ((A.1 - circle_center.1) * (B.1 - circle_center.1) + (A.2 - circle_center.2) * (B.2 - circle_center.2) = 0)) →
  (a = 1 ∨ a = -73/17) := by sorry

end NUMINAMATH_CALUDE_circle_equation_perpendicular_chord_values_l1425_142533


namespace NUMINAMATH_CALUDE_solution_set_of_system_l1425_142580

theorem solution_set_of_system (x y : ℝ) :
  x - 2 * y = 1 →
  x^3 - 6 * x * y - 8 * y^3 = 1 →
  y = (x - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_system_l1425_142580


namespace NUMINAMATH_CALUDE_bus_rental_combinations_l1425_142527

theorem bus_rental_combinations :
  let total_people : ℕ := 482
  let large_bus_capacity : ℕ := 42
  let medium_bus_capacity : ℕ := 20
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ =>
    p.1 * large_bus_capacity + p.2 * medium_bus_capacity = total_people
  ) (Finset.product (Finset.range (total_people + 1)) (Finset.range (total_people + 1)))).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_bus_rental_combinations_l1425_142527


namespace NUMINAMATH_CALUDE_expression_simplification_l1425_142595

theorem expression_simplification (x : ℝ) : 
  3 * x + 10 * x^2 - 7 - (1 + 5 * x - 10 * x^2) = 20 * x^2 - 2 * x - 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1425_142595


namespace NUMINAMATH_CALUDE_complex_multiplication_l1425_142563

theorem complex_multiplication (z : ℂ) (h : z = 1 + I) : (1 + z) * z = 1 + 3*I := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1425_142563


namespace NUMINAMATH_CALUDE_smallest_a_unique_b_l1425_142514

def is_all_real_roots (a b : ℝ) : Prop :=
  ∀ x : ℂ, x^4 - a*x^3 + b*x^2 - a*x + 1 = 0 → x.im = 0

theorem smallest_a_unique_b :
  ∃! (a : ℝ), a > 0 ∧
    (∃ (b : ℝ), b > 0 ∧ is_all_real_roots a b) ∧
    (∀ (a' : ℝ), 0 < a' ∧ a' < a →
      ¬∃ (b : ℝ), b > 0 ∧ is_all_real_roots a' b) ∧
    (∃! (b : ℝ), b > 0 ∧ is_all_real_roots a b) ∧
    a = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_unique_b_l1425_142514


namespace NUMINAMATH_CALUDE_pizza_order_theorem_l1425_142556

/-- The number of pizzas needed for a group of people -/
def pizzas_needed (num_people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  (num_people * slices_per_person + slices_per_pizza - 1) / slices_per_pizza

/-- Theorem: The number of pizzas needed for 18 people, where each person gets 3 slices
    and each pizza has 9 slices, is equal to 6 -/
theorem pizza_order_theorem :
  pizzas_needed 18 3 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_theorem_l1425_142556


namespace NUMINAMATH_CALUDE_complex_symmetric_product_l1425_142566

theorem complex_symmetric_product (z₁ z₂ : ℂ) :
  z₁.im = -z₂.im → z₁.re = z₂.re → z₁ = 2 - I → z₁ * z₂ = 5 := by sorry

end NUMINAMATH_CALUDE_complex_symmetric_product_l1425_142566


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1425_142568

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 36)
  (sum2 : b + c = 55)
  (sum3 : c + a = 60) : 
  a + b + c = 75.5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1425_142568


namespace NUMINAMATH_CALUDE_xe_pow_x_strictly_increasing_l1425_142598

/-- The function f(x) = xe^x is strictly increasing on the interval (-1, +∞) -/
theorem xe_pow_x_strictly_increasing :
  ∀ x₁ x₂, -1 < x₁ → x₁ < x₂ → x₁ * Real.exp x₁ < x₂ * Real.exp x₂ := by
  sorry

end NUMINAMATH_CALUDE_xe_pow_x_strictly_increasing_l1425_142598


namespace NUMINAMATH_CALUDE_ticket_cost_after_30_years_l1425_142507

/-- The cost of a ticket to Mars after a given number of years, given an initial cost and a halving period --/
def ticket_cost (initial_cost : ℕ) (halving_period : ℕ) (years : ℕ) : ℕ :=
  initial_cost / (2 ^ (years / halving_period))

/-- Theorem stating that the cost of a ticket to Mars after 30 years is $125,000 --/
theorem ticket_cost_after_30_years :
  ticket_cost 1000000 10 30 = 125000 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_after_30_years_l1425_142507


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1425_142576

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 + a*x + b < 0}) :
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/3) ∪ Set.Ioi (1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1425_142576


namespace NUMINAMATH_CALUDE_midline_tetrahedra_volume_ratio_l1425_142564

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- A tetrahedron formed by connecting midpoints of midlines to vertices -/
structure MidlineTetrahedron where
  volume : ℝ

/-- The common part of three MidlineTetrahedra -/
structure CommonTetrahedron where
  volume : ℝ

/-- Given a regular tetrahedron, construct three MidlineTetrahedra -/
def construct_midline_tetrahedra (t : RegularTetrahedron) : 
  (MidlineTetrahedron × MidlineTetrahedron × MidlineTetrahedron) :=
  sorry

/-- Find the common part of three MidlineTetrahedra -/
def find_common_part (t1 t2 t3 : MidlineTetrahedron) : CommonTetrahedron :=
  sorry

/-- The theorem to be proved -/
theorem midline_tetrahedra_volume_ratio 
  (t : RegularTetrahedron) 
  (t1 t2 t3 : MidlineTetrahedron) 
  (c : CommonTetrahedron) :
  t1 = (construct_midline_tetrahedra t).1 ∧
  t2 = (construct_midline_tetrahedra t).2.1 ∧
  t3 = (construct_midline_tetrahedra t).2.2 ∧
  c = find_common_part t1 t2 t3 →
  c.volume = t.volume / 10 :=
by sorry

end NUMINAMATH_CALUDE_midline_tetrahedra_volume_ratio_l1425_142564


namespace NUMINAMATH_CALUDE_linear_function_x_axis_intersection_l1425_142528

/-- A linear function f(x) = -x + 2 -/
def f (x : ℝ) : ℝ := -x + 2

/-- The x-coordinate of the intersection point with the x-axis -/
def x_intersection : ℝ := 2

theorem linear_function_x_axis_intersection :
  f x_intersection = 0 ∧ x_intersection = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_x_axis_intersection_l1425_142528


namespace NUMINAMATH_CALUDE_max_value_of_operation_achievable_max_value_l1425_142545

theorem max_value_of_operation (n : ℕ) : 
  (10 ≤ n ∧ n ≤ 99) → 2 * (200 - n) ≤ 380 :=
by
  sorry

theorem achievable_max_value : 
  ∃ (n : ℕ), (10 ≤ n ∧ n ≤ 99) ∧ 2 * (200 - n) = 380 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_operation_achievable_max_value_l1425_142545


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1425_142531

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the first inequality
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x > 0}

-- State the theorem
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) (h : solution_set a b c = {x : ℝ | 2 < x ∧ x < 3}) :
  {x : ℝ | c * x^2 - b * x + a > 0} = {x : ℝ | -1/2 < x ∧ x < -1/3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1425_142531


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_divisibility_l1425_142575

theorem gcd_lcm_sum_divisibility (a b : ℕ) (h : a > 0 ∧ b > 0) :
  Nat.gcd a b + Nat.lcm a b = a + b → a ∣ b ∨ b ∣ a := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_divisibility_l1425_142575


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l1425_142522

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 5)
  (h4 : avg_age_group1 = 12)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((group1_size : ℝ) * avg_age_group1 + (group2_size : ℝ) * avg_age_group2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l1425_142522


namespace NUMINAMATH_CALUDE_mike_siblings_l1425_142509

-- Define the characteristics
inductive EyeColor
| Blue
| Green

inductive HairColor
| Black
| Blonde

inductive Sport
| Soccer
| Basketball

-- Define a child's characteristics
structure ChildCharacteristics where
  eyeColor : EyeColor
  hairColor : HairColor
  favoriteSport : Sport

-- Define the children
def Lily : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Black, Sport.Soccer⟩
def Mike : ChildCharacteristics := ⟨EyeColor.Blue, HairColor.Blonde, Sport.Basketball⟩
def Oliver : ChildCharacteristics := ⟨EyeColor.Blue, HairColor.Black, Sport.Soccer⟩
def Emma : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Blonde, Sport.Basketball⟩
def Jacob : ChildCharacteristics := ⟨EyeColor.Blue, HairColor.Blonde, Sport.Soccer⟩
def Sophia : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Blonde, Sport.Soccer⟩

-- Define a function to check if two children share at least one characteristic
def shareCharacteristic (child1 child2 : ChildCharacteristics) : Prop :=
  child1.eyeColor = child2.eyeColor ∨ 
  child1.hairColor = child2.hairColor ∨ 
  child1.favoriteSport = child2.favoriteSport

-- Define the theorem
theorem mike_siblings : 
  shareCharacteristic Mike Emma ∧ 
  shareCharacteristic Mike Jacob ∧ 
  shareCharacteristic Emma Jacob ∧
  ¬(shareCharacteristic Mike Lily ∧ shareCharacteristic Mike Oliver ∧ shareCharacteristic Mike Sophia) :=
by sorry

end NUMINAMATH_CALUDE_mike_siblings_l1425_142509


namespace NUMINAMATH_CALUDE_dentist_age_l1425_142535

/-- The dentist's current age satisfies the given condition and is equal to 32. -/
theorem dentist_age : ∃ (x : ℕ), (x - 8) / 6 = (x + 8) / 10 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_dentist_age_l1425_142535


namespace NUMINAMATH_CALUDE_revenue_difference_is_164_5_l1425_142544

/-- Represents the types of fruits sold by Kevin --/
inductive Fruit
  | Grapes
  | Mangoes
  | PassionFruits

/-- Represents the pricing and quantity information for each fruit --/
structure FruitInfo where
  price : ℕ
  quantity : ℕ
  discountThreshold : ℕ
  discountRate : ℚ

/-- Calculates the revenue for a given fruit with or without discount --/
def calculateRevenue (info : FruitInfo) (applyDiscount : Bool) : ℚ :=
  let price := if applyDiscount && info.quantity > info.discountThreshold
    then info.price * (1 - info.discountRate)
    else info.price
  price * info.quantity

/-- Theorem: The difference between total revenue without and with discounts is $164.5 --/
theorem revenue_difference_is_164_5 (fruitData : Fruit → FruitInfo) 
    (h1 : fruitData Fruit.Grapes = { price := 15, quantity := 13, discountThreshold := 10, discountRate := 0.1 })
    (h2 : fruitData Fruit.Mangoes = { price := 20, quantity := 20, discountThreshold := 15, discountRate := 0.15 })
    (h3 : fruitData Fruit.PassionFruits = { price := 25, quantity := 17, discountThreshold := 5, discountRate := 0.2 })
    (h4 : (fruitData Fruit.Grapes).quantity + (fruitData Fruit.Mangoes).quantity + (fruitData Fruit.PassionFruits).quantity = 50) :
    (calculateRevenue (fruitData Fruit.Grapes) false +
     calculateRevenue (fruitData Fruit.Mangoes) false +
     calculateRevenue (fruitData Fruit.PassionFruits) false) -
    (calculateRevenue (fruitData Fruit.Grapes) true +
     calculateRevenue (fruitData Fruit.Mangoes) true +
     calculateRevenue (fruitData Fruit.PassionFruits) true) = 164.5 := by
  sorry


end NUMINAMATH_CALUDE_revenue_difference_is_164_5_l1425_142544


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1425_142551

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1425_142551


namespace NUMINAMATH_CALUDE_parallelogram_reflection_theorem_l1425_142524

-- Define the type for 2D points
def Point := ℝ × ℝ

-- Define the reflection across x-axis
def reflectX (p : Point) : Point :=
  (p.1, -p.2)

-- Define the reflection across y = x - 2
def reflectYXMinus2 (p : Point) : Point :=
  let p' := (p.1, p.2 + 2)  -- Translate up by 2
  let p'' := (p'.2, p'.1)   -- Reflect across y = x
  (p''.1, p''.2 - 2)        -- Translate back down by 2

-- Define the theorem
theorem parallelogram_reflection_theorem (A B C D : Point)
  (hA : A = (3, 7))
  (hB : B = (5, 11))
  (hC : C = (7, 7))
  (hD : D = (5, 3))
  : reflectYXMinus2 (reflectX D) = (-1, 3) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_reflection_theorem_l1425_142524


namespace NUMINAMATH_CALUDE_common_root_of_polynomials_l1425_142582

theorem common_root_of_polynomials :
  let p₁ (x : ℚ) := 3*x^4 + 13*x^3 + 20*x^2 + 17*x + 7
  let p₂ (x : ℚ) := 3*x^4 + x^3 - 8*x^2 + 11*x - 7
  p₁ (-7/3) = 0 ∧ p₂ (-7/3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_root_of_polynomials_l1425_142582


namespace NUMINAMATH_CALUDE_damien_tall_cupboard_glasses_l1425_142529

/-- Represents the number of glasses in different cupboards --/
structure Cupboards where
  tall : ℕ
  wide : ℕ
  narrow : ℕ

/-- The setup of Damien's glass collection --/
def damien_cupboards : Cupboards where
  tall := 5
  wide := 10
  narrow := 10

/-- Theorem stating the number of glasses in Damien's tall cupboard --/
theorem damien_tall_cupboard_glasses :
  ∃ (c : Cupboards), 
    c.wide = 2 * c.tall ∧ 
    c.narrow = 10 ∧ 
    15 % 3 = 0 ∧ 
    c = damien_cupboards :=
by
  sorry

#check damien_tall_cupboard_glasses

end NUMINAMATH_CALUDE_damien_tall_cupboard_glasses_l1425_142529


namespace NUMINAMATH_CALUDE_regular_octagon_angles_l1425_142578

theorem regular_octagon_angles :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
    n = 8 →
    interior_angle = (180 * (n - 2 : ℝ)) / n →
    exterior_angle = 180 - interior_angle →
    interior_angle = 135 ∧ exterior_angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_angles_l1425_142578


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l1425_142581

def f (x : ℝ) : ℝ := |x| + 1

theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l1425_142581


namespace NUMINAMATH_CALUDE_power_difference_equals_l1425_142508

theorem power_difference_equals (a b c : ℕ) :
  3^456 - 9^5 / 9^3 = 3^456 - 81 := by sorry

end NUMINAMATH_CALUDE_power_difference_equals_l1425_142508


namespace NUMINAMATH_CALUDE_gwen_money_left_l1425_142526

/-- The amount of money Gwen has left after spending some of her birthday money -/
def money_left (received : ℕ) (spent : ℕ) : ℕ :=
  received - spent

/-- Theorem stating that Gwen has 2 dollars left -/
theorem gwen_money_left :
  money_left 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gwen_money_left_l1425_142526


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1425_142546

/-- The circle x^2 + y^2 = m^2 is tangent to the line x - y = m if and only if m = 0 -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = m^2 ∧ x - y = m ∧ 
    (∀ (x' y' : ℝ), x'^2 + y'^2 = m^2 → x' - y' = m → (x', y') = (x, y))) ↔ 
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1425_142546


namespace NUMINAMATH_CALUDE_quadratic_sum_equations_l1425_142584

/-- Given two quadratic equations and their roots, prove the equations for the sums of roots -/
theorem quadratic_sum_equations 
  (a b c α β γ : ℝ) 
  (h1 : a * α ≠ 0) 
  (h2 : b^2 - 4*a*c ≥ 0) 
  (h3 : β^2 - 4*α*γ ≥ 0) 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : x₁ ≤ x₂) 
  (hy : y₁ ≤ y₂) 
  (hx_roots : a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) 
  (hy_roots : α * y₁^2 + β * y₁ + γ = 0 ∧ α * y₂^2 + β * y₂ + γ = 0) :
  ∃ (d δ : ℝ), 
    d^2 = b^2 - 4*a*c ∧ 
    δ^2 = β^2 - 4*α*γ ∧
    (∀ z, 2*a*α*z^2 + 2*(a*β + α*b)*z + (2*a*γ + 2*α*c + b*β - d*δ) = 0 ↔ 
      (z = x₁ + y₁ ∨ z = x₂ + y₂)) ∧
    (∀ u, 2*a*α*u^2 + 2*(a*β + α*b)*u + (2*a*γ + 2*α*c + b*β + d*δ) = 0 ↔ 
      (u = x₁ + y₂ ∨ u = x₂ + y₁)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sum_equations_l1425_142584


namespace NUMINAMATH_CALUDE_average_difference_l1425_142550

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 70 + x) / 3) + 8 → x = 16 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l1425_142550


namespace NUMINAMATH_CALUDE_intersection_points_count_l1425_142519

/-- Definition of the three lines -/
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := 5 * x + y = 1
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2

/-- A point lies on at least two of the three lines -/
def point_on_two_lines (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

/-- The main theorem to prove -/
theorem intersection_points_count :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
    point_on_two_lines p1.1 p1.2 ∧
    point_on_two_lines p2.1 p2.2 ∧
    ∀ (p : ℝ × ℝ), point_on_two_lines p.1 p.2 → p = p1 ∨ p = p2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_points_count_l1425_142519


namespace NUMINAMATH_CALUDE_product_sum_division_l1425_142571

theorem product_sum_division : (10 * 19 * 20 * 53 * 100 + 601) / 13 = 1549277 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_division_l1425_142571


namespace NUMINAMATH_CALUDE_triangle_vertices_l1425_142555

structure Triangle where
  a : ℝ
  m_a : ℝ
  s_a : ℝ

def is_valid_vertex (t : Triangle) (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = t.s_a^2 ∧ 
  |y| = t.m_a

theorem triangle_vertices (t : Triangle) 
  (h1 : t.a = 10) 
  (h2 : t.m_a = 4) 
  (h3 : t.s_a = 5) : 
  (is_valid_vertex t 8 4 ∧ 
   is_valid_vertex t 8 (-4) ∧ 
   is_valid_vertex t 2 4 ∧ 
   is_valid_vertex t 2 (-4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_vertices_l1425_142555


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l1425_142567

/-- Given the conditions of a class height measurement error, prove the number of boys in the class. -/
theorem number_of_boys_in_class 
  (n : ℕ) -- number of boys
  (initial_average : ℝ) -- initial average height
  (wrong_height : ℝ) -- wrongly recorded height
  (correct_height : ℝ) -- correct height of the boy
  (actual_average : ℝ) -- actual average height
  (h1 : initial_average = 182)
  (h2 : wrong_height = 166)
  (h3 : correct_height = 106)
  (h4 : actual_average = 180)
  (h5 : n * initial_average - wrong_height + correct_height = n * actual_average) :
  n = 30 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l1425_142567


namespace NUMINAMATH_CALUDE_min_value_of_f_min_value_of_f_equality_inequality_holds_iff_m_in_range_l1425_142573

-- Part 1
theorem min_value_of_f (a : ℝ) (ha : a > 0) :
  a^2 + 2/a ≥ 3 :=
sorry

theorem min_value_of_f_equality (a : ℝ) (ha : a > 0) :
  a^2 + 2/a = 3 ↔ a = 1 :=
sorry

-- Part 2
def m_range (m : ℝ) : Prop :=
  m ≤ -3 ∨ m ≥ -1

theorem inequality_holds_iff_m_in_range (m : ℝ) :
  (∀ a : ℝ, a > 0 → a^3 + 2 ≥ 3*a*(|m - 1| - |2*m + 3|)) ↔ m_range m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_min_value_of_f_equality_inequality_holds_iff_m_in_range_l1425_142573


namespace NUMINAMATH_CALUDE_settlement_area_theorem_l1425_142506

/-- Represents the lengths of the sides of the fields and forest -/
structure SettlementGeometry where
  r : ℝ  -- Length of the side of the square field
  p : ℝ  -- Length of the shorter side of the rectangular field
  q : ℝ  -- Length of the longer side of the rectangular forest

/-- The total area of the forest and fields given the geometry -/
def totalArea (g : SettlementGeometry) : ℝ :=
  g.r^2 + 4*g.p^2 + 12*g.q

/-- The conditions given in the problem -/
def satisfiesConditions (g : SettlementGeometry) : Prop :=
  12*g.q = g.r^2 + 4*g.p^2 + 45 ∧
  g.r > 0 ∧ g.p > 0 ∧ g.q > 0

theorem settlement_area_theorem (g : SettlementGeometry) 
  (h : satisfiesConditions g) : totalArea g = 135 := by
  sorry

#check settlement_area_theorem

end NUMINAMATH_CALUDE_settlement_area_theorem_l1425_142506


namespace NUMINAMATH_CALUDE_dodecahedron_faces_l1425_142557

/-- A regular dodecahedron is a Platonic solid with 12 faces. -/
def RegularDodecahedron : Type := Unit

/-- The number of faces in a regular dodecahedron. -/
def num_faces (d : RegularDodecahedron) : ℕ := 12

/-- Theorem: A regular dodecahedron has 12 faces. -/
theorem dodecahedron_faces :
  ∀ (d : RegularDodecahedron), num_faces d = 12 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_faces_l1425_142557


namespace NUMINAMATH_CALUDE_kerrys_age_l1425_142521

/-- Proves Kerry's age given the conditions of the birthday candle problem -/
theorem kerrys_age (num_cakes : ℕ) (candles_per_box : ℕ) (cost_per_box : ℚ) (total_cost : ℚ) :
  num_cakes = 3 →
  candles_per_box = 12 →
  cost_per_box = 5/2 →
  total_cost = 5 →
  (total_cost / cost_per_box * candles_per_box) / num_cakes = 8 := by
sorry

end NUMINAMATH_CALUDE_kerrys_age_l1425_142521


namespace NUMINAMATH_CALUDE_product_of_middle_terms_l1425_142561

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_middle_terms 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 7 * a 12 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
sorry

end NUMINAMATH_CALUDE_product_of_middle_terms_l1425_142561


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l1425_142589

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem tenth_term_of_arithmetic_sequence 
  (a₁ d : ℝ) 
  (h₁ : arithmetic_sequence a₁ d 3 = 10) 
  (h₂ : arithmetic_sequence a₁ d 8 = 30) : 
  arithmetic_sequence a₁ d 10 = 38 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l1425_142589


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1425_142503

theorem decimal_to_fraction : (2.75 : ℚ) = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1425_142503


namespace NUMINAMATH_CALUDE_calculate_expression_l1425_142515

theorem calculate_expression (a : ℝ) : (-2 * a^2)^3 / a^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1425_142515


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1425_142504

/-- For a square with perimeter 48 meters, its area is 144 square meters. -/
theorem square_area_from_perimeter : 
  ∀ (s : Real), 
    (4 * s = 48) →  -- perimeter = 4 * side length = 48
    (s * s = 144)   -- area = side length * side length = 144
:= by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l1425_142504


namespace NUMINAMATH_CALUDE_sum_even_integers_40_to_60_l1425_142560

def evenIntegersFrom40To60 : List ℕ := [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]

def x : ℕ := evenIntegersFrom40To60.sum

def y : ℕ := evenIntegersFrom40To60.length

theorem sum_even_integers_40_to_60 : x + y = 561 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_integers_40_to_60_l1425_142560


namespace NUMINAMATH_CALUDE_exists_n_fractional_part_greater_than_bound_l1425_142502

theorem exists_n_fractional_part_greater_than_bound : 
  ∃ n : ℕ+, (2 + Real.sqrt 2)^(n : ℝ) - ⌊(2 + Real.sqrt 2)^(n : ℝ)⌋ > 0.999999 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_fractional_part_greater_than_bound_l1425_142502


namespace NUMINAMATH_CALUDE_volunteer_distribution_theorem_l1425_142590

/-- The number of ways to distribute volunteers among exits -/
def distribute_volunteers (num_volunteers : ℕ) (num_exits : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements -/
theorem volunteer_distribution_theorem :
  distribute_volunteers 5 4 = 240 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_theorem_l1425_142590


namespace NUMINAMATH_CALUDE_max_profit_140000_l1425_142543

structure ProductionPlan where
  productA : ℕ
  productB : ℕ

def componentAUsage (plan : ProductionPlan) : ℕ := 4 * plan.productA
def componentBUsage (plan : ProductionPlan) : ℕ := 4 * plan.productB
def totalHours (plan : ProductionPlan) : ℕ := plan.productA + 2 * plan.productB
def profit (plan : ProductionPlan) : ℕ := 20000 * plan.productA + 30000 * plan.productB

def isValidPlan (plan : ProductionPlan) : Prop :=
  componentAUsage plan ≤ 16 ∧
  componentBUsage plan ≤ 12 ∧
  totalHours plan ≤ 8

theorem max_profit_140000 :
  ∃ (optimalPlan : ProductionPlan),
    isValidPlan optimalPlan ∧
    profit optimalPlan = 140000 ∧
    ∀ (plan : ProductionPlan), isValidPlan plan → profit plan ≤ profit optimalPlan :=
sorry

end NUMINAMATH_CALUDE_max_profit_140000_l1425_142543


namespace NUMINAMATH_CALUDE_nina_has_24_dollars_l1425_142569

/-- The amount of money Nina has -/
def nina_money : ℝ := 24

/-- The original price of a widget -/
def original_price : ℝ := 4

/-- Nina can purchase exactly 6 widgets at the original price -/
axiom nina_purchase_original : nina_money = 6 * original_price

/-- If each widget's price is reduced by $1, Nina can purchase exactly 8 widgets -/
axiom nina_purchase_reduced : nina_money = 8 * (original_price - 1)

/-- Proof that Nina has $24 -/
theorem nina_has_24_dollars : nina_money = 24 := by sorry

end NUMINAMATH_CALUDE_nina_has_24_dollars_l1425_142569


namespace NUMINAMATH_CALUDE_total_cost_matches_expected_l1425_142542

/-- Calculate the total cost of an order with given conditions --/
def calculate_total_cost (burger_price : ℚ) (soda_price : ℚ) (chicken_sandwich_price : ℚ) 
  (happy_hour_discount : ℚ) (coupon_discount : ℚ) (sales_tax : ℚ) 
  (paulo_burgers : ℕ) (paulo_sodas : ℕ) (jeremy_burgers : ℕ) (jeremy_sodas : ℕ) 
  (stephanie_burgers : ℕ) (stephanie_sodas : ℕ) (stephanie_chicken : ℕ) : ℚ :=
  let total_burgers := paulo_burgers + jeremy_burgers + stephanie_burgers
  let total_sodas := paulo_sodas + jeremy_sodas + stephanie_sodas
  let subtotal := burger_price * total_burgers + soda_price * total_sodas + 
                  chicken_sandwich_price * stephanie_chicken
  let tax_amount := sales_tax * subtotal
  let total_with_tax := subtotal + tax_amount
  let coupon_applied := if total_with_tax > 25 then total_with_tax - coupon_discount else total_with_tax
  let happy_hour_discount_amount := if total_burgers > 2 then happy_hour_discount * (burger_price * total_burgers) else 0
  coupon_applied - happy_hour_discount_amount

/-- Theorem stating that the total cost matches the expected result --/
theorem total_cost_matches_expected : 
  calculate_total_cost 6 2 7.5 0.1 5 0.05 1 1 2 2 3 1 1 = 45.48 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_matches_expected_l1425_142542


namespace NUMINAMATH_CALUDE_intersection_distance_l1425_142587

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x = y^2 / 10 + 2.5

-- Define the shared focus
def shared_focus : ℝ × ℝ := (5, 0)

-- Define the directrix of the parabola
def parabola_directrix : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}

-- Theorem statement
theorem intersection_distance :
  ∃ p1 p2 : ℝ × ℝ,
    hyperbola p1.1 p1.2 ∧
    hyperbola p2.1 p2.2 ∧
    parabola p1.1 p1.2 ∧
    parabola p2.1 p2.2 ∧
    p1 ≠ p2 ∧
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 4 * Real.sqrt 218 / 15 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l1425_142587


namespace NUMINAMATH_CALUDE_decreasing_cubic_function_parameter_bound_l1425_142537

/-- Given a function f(x) = ax³ - x that is decreasing on ℝ, prove that a ≤ 0 -/
theorem decreasing_cubic_function_parameter_bound (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => a * x^3 - x) (3 * a * x^2 - 1) x) →
  (∀ x y : ℝ, x < y → (a * x^3 - x) > (a * y^3 - y)) →
  a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_cubic_function_parameter_bound_l1425_142537


namespace NUMINAMATH_CALUDE_robot_position_l1425_142597

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the Cartesian plane defined by y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The robot's path is defined as the set of points equidistant from two given points -/
def RobotPath (A B : Point) : Set Point :=
  {P : Point | (P.x - A.x)^2 + (P.y - A.y)^2 = (P.x - B.x)^2 + (P.y - B.y)^2}

/-- Check if a point is on a line -/
def isOnLine (P : Point) (L : Line) : Prop :=
  P.y = L.m * P.x + L.b

theorem robot_position (a : ℝ) : 
  let A : Point := ⟨a, 0⟩
  let B : Point := ⟨0, 1⟩
  let L : Line := ⟨1, 1⟩  -- y = x + 1
  (∀ P ∈ RobotPath A B, ¬isOnLine P L) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_robot_position_l1425_142597


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_18_factorial_l1425_142583

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_power_of_three_dividing (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc x => acc + (if x % 3 = 0 then 1 else 0) + (if x % 9 = 0 then 1 else 0)) 0

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_three_dividing_18_factorial :
  ones_digit (3^(largest_power_of_three_dividing (factorial 18))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_18_factorial_l1425_142583


namespace NUMINAMATH_CALUDE_trig_equation_result_l1425_142536

theorem trig_equation_result (x : Real) : 
  2 * Real.cos x - 5 * Real.sin x = 3 → 
  Real.sin x + 2 * Real.cos x = 1/2 ∨ Real.sin x + 2 * Real.cos x = 83/29 := by
sorry

end NUMINAMATH_CALUDE_trig_equation_result_l1425_142536


namespace NUMINAMATH_CALUDE_infinitely_many_integers_with_zero_padic_valuation_mod_d_l1425_142517

/-- The p-adic valuation of n! -/
def ν (p : Nat) (n : Nat) : Nat := sorry

theorem infinitely_many_integers_with_zero_padic_valuation_mod_d 
  (d : Nat) (primes : Finset Nat) (h_d : d > 0) (h_primes : ∀ p ∈ primes, Nat.Prime p) :
  ∃ (S : Set Nat), Set.Infinite S ∧ 
    ∀ n ∈ S, ∀ p ∈ primes, (ν p n) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_integers_with_zero_padic_valuation_mod_d_l1425_142517


namespace NUMINAMATH_CALUDE_statement_A_statement_B_l1425_142548

-- Define the parabola E
def E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of E
def F : ℝ × ℝ := (1, 0)

-- Define the circle F
def circle_F (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = r^2

-- Define the line l_0
def l_0 (t : ℝ) (x y : ℝ) : Prop := x = t*y + 1

-- Define the intersection points A and B
def A (t : ℝ) : ℝ × ℝ := (t^2 + 1, 2*t)
def B (t : ℝ) : ℝ × ℝ := (t^2 + 1, -2*t)

-- Define the midpoint M
def M (t : ℝ) : ℝ × ℝ := (2*t^2 + 1, 2*t)

-- Define point T
def T : ℝ × ℝ := (0, 1)

-- Theorem for statement A
theorem statement_A (t : ℝ) : 
  let y_1 := (A t).2
  let y_2 := (B t).2
  let y_3 := -1/t
  1/y_1 + 1/y_2 = 1/y_3 :=
sorry

-- Theorem for statement B
theorem statement_B : 
  ∃ a b c : ℝ, ∀ t : ℝ, 
    let (x, y) := M t
    y^2 = a*x + b*y + c :=
sorry

end NUMINAMATH_CALUDE_statement_A_statement_B_l1425_142548


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1425_142539

/-- Given two incorrect solutions to a quadratic inequality, prove the correct solution -/
theorem quadratic_inequality_solution 
  (b c : ℝ) 
  (h1 : ∀ x, x^2 + b*x + c < 0 ↔ -6 < x ∧ x < 2)
  (h2 : ∃ c', ∀ x, x^2 + b*x + c' < 0 ↔ -3 < x ∧ x < 2) :
  ∀ x, x^2 + b*x + c < 0 ↔ -4 < x ∧ x < 3 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1425_142539


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l1425_142525

theorem square_area_error_percentage (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let area_error_percentage := (area_error / actual_area) * 100
  area_error_percentage = 4.04 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_percentage_l1425_142525


namespace NUMINAMATH_CALUDE_octal_4652_to_decimal_l1425_142553

def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

theorem octal_4652_to_decimal :
  octal_to_decimal [2, 5, 6, 4] = 2474 := by
  sorry

end NUMINAMATH_CALUDE_octal_4652_to_decimal_l1425_142553


namespace NUMINAMATH_CALUDE_range_of_a_l1425_142588

-- Define the functions f and g
def f (x : ℝ) := 3 * abs (x - 1) + abs (3 * x + 1)
def g (a : ℝ) (x : ℝ) := abs (x + 2) + abs (x - a)

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, f x = y}
def B (a : ℝ) : Set ℝ := {y | ∃ x, g a x = y}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (A ∪ B a = B a) → (a ∈ Set.Icc (-6) 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1425_142588


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1425_142516

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = -1/2 + Real.sqrt 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1425_142516


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_l1425_142591

theorem fourth_term_coefficient (a : ℝ) : 
  (Nat.choose 6 3) * a^3 * (-1)^3 = 160 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_l1425_142591


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_l1425_142512

/-- The function f(x) defined in the problem -/
def f (a x : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

/-- Theorem stating the condition for f(x) to be non-negative for all real x -/
theorem f_nonnegative_iff (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_l1425_142512


namespace NUMINAMATH_CALUDE_sara_picked_24_more_peaches_l1425_142530

/-- The number of additional peaches Sara picked at the orchard -/
def additional_peaches (initial_peaches total_peaches : ℝ) : ℝ :=
  total_peaches - initial_peaches

/-- Theorem: Sara picked 24 additional peaches at the orchard -/
theorem sara_picked_24_more_peaches (initial_peaches total_peaches : ℝ)
  (h1 : initial_peaches = 61.0)
  (h2 : total_peaches = 85.0) :
  additional_peaches initial_peaches total_peaches = 24 := by
  sorry

end NUMINAMATH_CALUDE_sara_picked_24_more_peaches_l1425_142530


namespace NUMINAMATH_CALUDE_tangent_line_properties_l1425_142541

-- Define the function
def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 + 2 * x + 1

-- Define the derivative of the function
def f' (m : ℝ) (x : ℝ) : ℝ := 3 * m * x^2 + 2

theorem tangent_line_properties (m : ℝ) :
  -- Part 1: Parallel to y = 3x
  (f' m 1 = 3 → m = 1/3) ∧
  -- Part 2: Perpendicular to y = -1/2x
  (f' m 1 = 2 → ∃ b : ℝ, ∀ x y : ℝ, y = 2 * x + b ↔ y - f m 1 = f' m 1 * (x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l1425_142541


namespace NUMINAMATH_CALUDE_book_words_per_page_l1425_142549

theorem book_words_per_page (total_pages : Nat) (max_words_per_page : Nat) (remainder : Nat) :
  total_pages = 150 →
  max_words_per_page = 100 →
  remainder = 198 →
  ∃ p : Nat,
    p ≤ max_words_per_page ∧
    (total_pages * p) % 221 = remainder ∧
    p = 93 :=
by sorry

end NUMINAMATH_CALUDE_book_words_per_page_l1425_142549


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1425_142592

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = 3 * x - 10) → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1425_142592


namespace NUMINAMATH_CALUDE_not_always_possible_to_equalize_l1425_142574

/-- Represents a board with integers -/
def Board := Matrix (Fin 2018) (Fin 2019) Int

/-- Checks if two positions are neighbors on the board -/
def is_neighbor (i j i' j' : Nat) : Prop :=
  (i = i' ∧ (j = j' + 1 ∨ j + 1 = j')) ∨ 
  (j = j' ∧ (i = i' + 1 ∨ i + 1 = i'))

/-- Represents a single turn of the averaging operation -/
def average_turn (b : Board) (chosen : Set (Fin 2018 × Fin 2019)) : Board :=
  sorry

/-- Represents a sequence of turns -/
def sequence_of_turns (b : Board) (turns : Nat) : Board :=
  sorry

/-- Checks if all numbers on the board are the same -/
def all_same (b : Board) : Prop :=
  ∀ i j i' j', b i j = b i' j'

theorem not_always_possible_to_equalize : ∃ (initial : Board), 
  ∀ (turns : Nat), ¬(all_same (sequence_of_turns initial turns)) :=
sorry

end NUMINAMATH_CALUDE_not_always_possible_to_equalize_l1425_142574


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l1425_142577

/-- A parabola with equation y = dx^2 + ex + f, vertex (-3, 2), and passing through (-5, 10) -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_condition : 2 = d * (-3)^2 + e * (-3) + f
  point_condition : 10 = d * (-5)^2 + e * (-5) + f

/-- The sum of coefficients d, e, and f equals 10 -/
theorem parabola_coefficient_sum (p : Parabola) : p.d + p.e + p.f = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l1425_142577


namespace NUMINAMATH_CALUDE_green_hats_count_l1425_142559

/-- The number of green hard hats initially in the truck -/
def initial_green_hats : ℕ := sorry

/-- The number of pink hard hats initially in the truck -/
def initial_pink_hats : ℕ := 26

/-- The number of yellow hard hats in the truck -/
def yellow_hats : ℕ := 24

/-- The number of pink hard hats Carl takes away -/
def carl_pink_hats : ℕ := 4

/-- The number of pink hard hats John takes away -/
def john_pink_hats : ℕ := 6

/-- The number of green hard hats John takes away -/
def john_green_hats : ℕ := 2 * john_pink_hats

/-- The total number of hard hats remaining in the truck -/
def remaining_hats : ℕ := 43

theorem green_hats_count : initial_green_hats = 15 :=
  by sorry

end NUMINAMATH_CALUDE_green_hats_count_l1425_142559


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l1425_142599

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop :=
  ∃ (t : ℝ), x = 2 + t ∧ y = t

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_through_focus x₁ y₁ ∧ line_through_focus x₂ y₂

-- Theorem statement
theorem parabola_intersection_length
  (x₁ y₁ x₂ y₂ : ℝ)
  (h_intersection : intersection_points x₁ y₁ x₂ y₂)
  (h_sum : x₁ + x₂ = 6) :
  ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2 : ℝ) = 10 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l1425_142599


namespace NUMINAMATH_CALUDE_distance_to_sea_world_l1425_142520

/-- Calculates the distance to Sea World based on given conditions --/
theorem distance_to_sea_world 
  (savings : ℕ) 
  (parking_cost : ℕ) 
  (entrance_cost : ℕ) 
  (meal_pass_cost : ℕ) 
  (car_efficiency : ℕ) 
  (gas_price : ℕ) 
  (additional_savings_needed : ℕ) 
  (h1 : savings = 28)
  (h2 : parking_cost = 10)
  (h3 : entrance_cost = 55)
  (h4 : meal_pass_cost = 25)
  (h5 : car_efficiency = 30)
  (h6 : gas_price = 3)
  (h7 : additional_savings_needed = 95)
  : ℕ := by
  sorry

#check distance_to_sea_world

end NUMINAMATH_CALUDE_distance_to_sea_world_l1425_142520


namespace NUMINAMATH_CALUDE_infant_weight_at_four_months_l1425_142500

/-- Represents the weight of an infant in grams at a given age in months. -/
def infantWeight (birthWeight : ℝ) (ageMonths : ℝ) : ℝ :=
  birthWeight + 700 * ageMonths

/-- Theorem stating that an infant with a birth weight of 3000 grams will weigh 5800 grams at 4 months. -/
theorem infant_weight_at_four_months :
  infantWeight 3000 4 = 5800 := by
  sorry

end NUMINAMATH_CALUDE_infant_weight_at_four_months_l1425_142500


namespace NUMINAMATH_CALUDE_part1_part2_part3_l1425_142538

-- Define the system of linear equations
def system (x y : ℝ) : Prop :=
  2 * x + 3 * y = 6 ∧ 3 * x + 2 * y = 4

-- Define the new operation *
def star (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c

-- Theorem for part 1
theorem part1 (x y : ℝ) (h : system x y) : x + y = 2 ∧ x - y = -2 := by
  sorry

-- Theorem for part 2
theorem part2 : ∃ x y : ℝ, 2024 * x + 2025 * y = 2023 ∧ 2022 * x + 2023 * y = 2021 ∧ x = 2 ∧ y = -1 := by
  sorry

-- Theorem for part 3
theorem part3 (a b c : ℝ) (h1 : star a b c 2 4 = 15) (h2 : star a b c 3 7 = 27) : 
  star a b c 1 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l1425_142538


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1425_142570

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + 
    a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1425_142570


namespace NUMINAMATH_CALUDE_equation_solution_l1425_142532

theorem equation_solution (x y : ℝ) : 
  x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ↔ x = 1/3 ∧ y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1425_142532


namespace NUMINAMATH_CALUDE_red_tint_percentage_l1425_142565

/-- Given a paint mixture, calculate the percentage of red tint after adding more red tint -/
theorem red_tint_percentage (original_volume : ℝ) (original_red_percent : ℝ) (added_red_volume : ℝ) :
  original_volume = 40 →
  original_red_percent = 20 →
  added_red_volume = 10 →
  let original_red_volume := original_red_percent / 100 * original_volume
  let new_red_volume := original_red_volume + added_red_volume
  let new_total_volume := original_volume + added_red_volume
  (new_red_volume / new_total_volume) * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_red_tint_percentage_l1425_142565


namespace NUMINAMATH_CALUDE_abs_geq_one_necessary_not_sufficient_for_x_gt_two_l1425_142554

theorem abs_geq_one_necessary_not_sufficient_for_x_gt_two :
  (∀ x : ℝ, x > 2 → |x| ≥ 1) ∧
  (∃ x : ℝ, |x| ≥ 1 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_abs_geq_one_necessary_not_sufficient_for_x_gt_two_l1425_142554


namespace NUMINAMATH_CALUDE_equation_solutions_l1425_142501

theorem equation_solutions :
  let f (x : ℝ) := 4 * (3 * x)^2 + 3 * x + 6 - (3 * (9 * x^2 + 3 * x + 3))
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1425_142501


namespace NUMINAMATH_CALUDE_tan_addition_result_l1425_142558

theorem tan_addition_result (x : Real) (h : Real.tan x = 3) :
  Real.tan (x + π / 3) = -(6 + 5 * Real.sqrt 3) / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_addition_result_l1425_142558


namespace NUMINAMATH_CALUDE_students_walking_home_l1425_142594

theorem students_walking_home (bus auto bike scooter : ℚ) 
  (h_bus : bus = 1/3)
  (h_auto : auto = 1/5)
  (h_bike : bike = 1/6)
  (h_scooter : scooter = 1/10)
  (h_total : bus + auto + bike + scooter < 1) :
  1 - (bus + auto + bike + scooter) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_l1425_142594


namespace NUMINAMATH_CALUDE_carols_mother_carrots_l1425_142552

theorem carols_mother_carrots : 
  ∀ (carol_carrots good_carrots bad_carrots total_carrots mother_carrots : ℕ),
    carol_carrots = 29 →
    good_carrots = 38 →
    bad_carrots = 7 →
    total_carrots = good_carrots + bad_carrots →
    mother_carrots = total_carrots - carol_carrots →
    mother_carrots = 16 := by
  sorry

end NUMINAMATH_CALUDE_carols_mother_carrots_l1425_142552
