import Mathlib

namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_l2673_267306

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 42

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 70

theorem fgh_supermarkets_count :
  (us_supermarkets + canada_supermarkets = total_supermarkets) ∧
  (us_supermarkets = canada_supermarkets + 14) :=
by sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_l2673_267306


namespace NUMINAMATH_CALUDE_right_triangle_side_values_l2673_267328

theorem right_triangle_side_values (a b x : ℝ) : 
  a = 6 → b = 8 → (x^2 = a^2 + b^2 ∨ b^2 = a^2 + x^2) → (x = 10 ∨ x = 2 * Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_values_l2673_267328


namespace NUMINAMATH_CALUDE_x_range_l2673_267316

theorem x_range (x : ℝ) (h : |2*x + 1| + |2*x - 5| = 6) : 
  x ∈ Set.Icc (-1/2 : ℝ) (5/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_x_range_l2673_267316


namespace NUMINAMATH_CALUDE_problem_solution_l2673_267315

theorem problem_solution (x y : ℝ) (h1 : x / y = 15 / 5) (h2 : y = 25) : x = 75 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2673_267315


namespace NUMINAMATH_CALUDE_number_of_paths_l2673_267323

theorem number_of_paths (paths_A_to_B paths_B_to_D paths_D_to_C : ℕ) 
  (direct_path_A_to_C : ℕ) :
  paths_A_to_B = 2 →
  paths_B_to_D = 3 →
  paths_D_to_C = 3 →
  direct_path_A_to_C = 1 →
  paths_A_to_B * paths_B_to_D * paths_D_to_C + direct_path_A_to_C = 19 :=
by sorry

end NUMINAMATH_CALUDE_number_of_paths_l2673_267323


namespace NUMINAMATH_CALUDE_expression_evaluation_l2673_267375

theorem expression_evaluation :
  let a : ℝ := 40
  let c : ℝ := 4
  1891 - (1600 / a + 8040 / a) * c = 927 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2673_267375


namespace NUMINAMATH_CALUDE_smallest_a_value_l2673_267376

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℝ, Real.sin (a * x + b) = Real.sin (15 * x)) : 
  a ≥ 15 ∧ ∃ (a₀ b₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ (∀ x : ℝ, Real.sin (a₀ * x + b₀) = Real.sin (15 * x)) ∧ a₀ = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2673_267376


namespace NUMINAMATH_CALUDE_jack_cell_phone_cost_l2673_267368

/- Define the cell phone plan parameters -/
def base_cost : ℝ := 25
def text_cost : ℝ := 0.08
def extra_minute_cost : ℝ := 0.10
def free_hours : ℝ := 25

/- Define Jack's usage -/
def texts_sent : ℕ := 150
def hours_talked : ℝ := 26

/- Calculate the total cost -/
def total_cost : ℝ :=
  base_cost +
  (↑texts_sent * text_cost) +
  ((hours_talked - free_hours) * 60 * extra_minute_cost)

/- Theorem to prove -/
theorem jack_cell_phone_cost : total_cost = 43 := by
  sorry

end NUMINAMATH_CALUDE_jack_cell_phone_cost_l2673_267368


namespace NUMINAMATH_CALUDE_zoo_count_l2673_267304

theorem zoo_count (zebras camels monkeys giraffes : ℕ) : 
  camels = zebras / 2 →
  monkeys = 4 * camels →
  giraffes = 2 →
  monkeys = giraffes + 22 →
  zebras = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_count_l2673_267304


namespace NUMINAMATH_CALUDE_ladder_length_proof_l2673_267359

theorem ladder_length_proof (ladder_length wall_height : ℝ) : 
  wall_height = ladder_length + 8/3 →
  ∃ (ladder_base ladder_top : ℝ),
    ladder_base = 3/5 * ladder_length ∧
    ladder_top = 2/5 * wall_height ∧
    ladder_length^2 = ladder_base^2 + ladder_top^2 →
  ladder_length = 8/3 := by
sorry

end NUMINAMATH_CALUDE_ladder_length_proof_l2673_267359


namespace NUMINAMATH_CALUDE_new_drive_size_l2673_267373

/-- Calculates the size of a new external drive based on initial drive conditions and file operations -/
theorem new_drive_size
  (initial_free : ℝ)
  (initial_used : ℝ)
  (deleted_size : ℝ)
  (new_files_size : ℝ)
  (new_free_space : ℝ)
  (h1 : initial_free = 2.4)
  (h2 : initial_used = 12.6)
  (h3 : deleted_size = 4.6)
  (h4 : new_files_size = 2)
  (h5 : new_free_space = 10) :
  initial_used - deleted_size + new_files_size + new_free_space = 20 := by
  sorry

#check new_drive_size

end NUMINAMATH_CALUDE_new_drive_size_l2673_267373


namespace NUMINAMATH_CALUDE_lisas_marbles_problem_l2673_267383

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisas_marbles_problem (num_friends : ℕ) (initial_marbles : ℕ) 
    (h1 : num_friends = 12) (h2 : initial_marbles = 34) : 
    min_additional_marbles num_friends initial_marbles = 44 := by
  sorry

#eval min_additional_marbles 12 34

end NUMINAMATH_CALUDE_lisas_marbles_problem_l2673_267383


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2673_267369

def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2673_267369


namespace NUMINAMATH_CALUDE_line_through_points_l2673_267357

/-- Given a line passing through points (-3, 1) and (1, 5) with equation y = mx + b, prove that m + b = 5 -/
theorem line_through_points (m b : ℝ) : 
  (1 = m * (-3) + b) → (5 = m * 1 + b) → m + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2673_267357


namespace NUMINAMATH_CALUDE_min_buses_second_group_l2673_267364

theorem min_buses_second_group 
  (total_students : ℕ) 
  (bus_capacity : ℕ) 
  (max_buses_first_group : ℕ) 
  (min_buses_second_group : ℕ) : 
  total_students = 550 → 
  bus_capacity = 45 → 
  max_buses_first_group = 8 → 
  min_buses_second_group = 5 → 
  (max_buses_first_group * bus_capacity + min_buses_second_group * bus_capacity ≥ total_students) ∧
  ((min_buses_second_group - 1) * bus_capacity < total_students - max_buses_first_group * bus_capacity) :=
by
  sorry

#check min_buses_second_group

end NUMINAMATH_CALUDE_min_buses_second_group_l2673_267364


namespace NUMINAMATH_CALUDE_rectangular_plot_minus_circular_garden_l2673_267349

/-- The area of a rectangular plot minus a circular garden --/
theorem rectangular_plot_minus_circular_garden :
  let rectangle_length : ℝ := 8
  let rectangle_width : ℝ := 12
  let circle_radius : ℝ := 3
  let rectangle_area := rectangle_length * rectangle_width
  let circle_area := π * circle_radius ^ 2
  rectangle_area - circle_area = 96 - 9 * π := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_minus_circular_garden_l2673_267349


namespace NUMINAMATH_CALUDE_spherical_coords_negated_y_theorem_l2673_267394

/-- Given a point with rectangular coordinates (x, y, z) and spherical coordinates (ρ, θ, φ),
    this function returns the spherical coordinates of the point (x, -y, z) -/
def spherical_coords_negated_y (x y z ρ θ φ : Real) : Real × Real × Real :=
  sorry

/-- Theorem stating that if a point has rectangular coordinates (x, y, z) and 
    spherical coordinates (3, 5π/6, 5π/12), then the point with rectangular 
    coordinates (x, -y, z) has spherical coordinates (3, π/6, 5π/12) -/
theorem spherical_coords_negated_y_theorem (x y z : Real) :
  let (ρ, θ, φ) := (3, 5*π/6, 5*π/12)
  (x = ρ * Real.sin φ * Real.cos θ) →
  (y = ρ * Real.sin φ * Real.sin θ) →
  (z = ρ * Real.cos φ) →
  spherical_coords_negated_y x y z ρ θ φ = (3, π/6, 5*π/12) :=
by
  sorry

end NUMINAMATH_CALUDE_spherical_coords_negated_y_theorem_l2673_267394


namespace NUMINAMATH_CALUDE_triangle_side_sum_max_l2673_267319

theorem triangle_side_sum_max (a b c : ℝ) (C : ℝ) :
  C = π / 3 →
  c = Real.sqrt 3 →
  a > 0 →
  b > 0 →
  c > 0 →
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
  a + b ≤ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_max_l2673_267319


namespace NUMINAMATH_CALUDE_always_has_real_roots_unique_integer_m_for_distinct_positive_integer_roots_l2673_267395

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := m * x^2 - (m + 2) * x + 2

-- Part I: The equation always has real roots
theorem always_has_real_roots :
  ∀ m : ℝ, ∃ x : ℝ, quadratic_equation m x = 0 :=
sorry

-- Part II: Only m = 1 gives two distinct positive integer roots
theorem unique_integer_m_for_distinct_positive_integer_roots :
  ∀ m : ℤ, (∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x ≠ y ∧
    quadratic_equation (m : ℝ) (x : ℝ) = 0 ∧
    quadratic_equation (m : ℝ) (y : ℝ) = 0) ↔ m = 1 :=
sorry

end NUMINAMATH_CALUDE_always_has_real_roots_unique_integer_m_for_distinct_positive_integer_roots_l2673_267395


namespace NUMINAMATH_CALUDE_jellybean_count_jellybean_count_proof_l2673_267333

theorem jellybean_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun black green orange total =>
    (black = 8) →
    (green = black + 2) →
    (orange = green - 1) →
    (total = black + green + orange) →
    (total = 27)

-- The proof is omitted
theorem jellybean_count_proof : jellybean_count 8 10 9 27 := by sorry

end NUMINAMATH_CALUDE_jellybean_count_jellybean_count_proof_l2673_267333


namespace NUMINAMATH_CALUDE_triangle_isosceles_l2673_267302

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angle : 0 < A ∧ A < π

-- Define the theorem
theorem triangle_isosceles (t : Triangle) 
  (h : Real.log (t.a^2) = Real.log (t.b^2) + Real.log (t.c^2) - Real.log (2 * t.b * t.c * Real.cos t.A)) :
  t.a = t.b ∨ t.a = t.c := by
  sorry


end NUMINAMATH_CALUDE_triangle_isosceles_l2673_267302


namespace NUMINAMATH_CALUDE_distance_from_origin_l2673_267381

theorem distance_from_origin (x : ℝ) : |x| > 2 ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_l2673_267381


namespace NUMINAMATH_CALUDE_magnitude_a_minus_2b_l2673_267372

def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (-1, 4)  -- Derived from a + b = (0, 2)

theorem magnitude_a_minus_2b :
  let a : ℝ × ℝ := vector_a
  let b : ℝ × ℝ := vector_b
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 109 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_minus_2b_l2673_267372


namespace NUMINAMATH_CALUDE_meat_for_hamburgers_l2673_267318

/-- Given that 3 pounds of meat make 8 hamburgers, prove that 9 pounds of meat are needed for 24 hamburgers -/
theorem meat_for_hamburgers (meat_per_8 : ℝ) (hamburgers : ℝ) 
  (h1 : meat_per_8 = 3) 
  (h2 : hamburgers = 24) : 
  (meat_per_8 / 8) * hamburgers = 9 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_hamburgers_l2673_267318


namespace NUMINAMATH_CALUDE_impossibleSquareConstruction_l2673_267356

/-- Represents a square constructed on a chord of a unit circle -/
structure SquareOnChord where
  sideLength : ℝ
  twoVerticesOnChord : Bool
  twoVerticesOnCircumference : Bool

/-- Represents a chord of a unit circle -/
structure Chord where
  length : ℝ
  inUnitCircle : length > 0 ∧ length ≤ 2

theorem impossibleSquareConstruction (c : Chord) :
  ¬∃ (s1 s2 : SquareOnChord),
    s1.twoVerticesOnChord ∧
    s1.twoVerticesOnCircumference ∧
    s2.twoVerticesOnChord ∧
    s2.twoVerticesOnCircumference ∧
    s1.sideLength - s2.sideLength = 1 ∧
    s1.sideLength = c.length / Real.sqrt 2 ∧
    s2.sideLength = (c.length - Real.sqrt 2) / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_impossibleSquareConstruction_l2673_267356


namespace NUMINAMATH_CALUDE_most_cost_effective_plan_l2673_267329

/-- Represents the capacity and rental cost of a truck type -/
structure TruckType where
  capacity : ℕ
  rentalCost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

def totalCapacity (a b : TruckType) (plan : RentalPlan) : ℕ :=
  a.capacity * plan.typeA + b.capacity * plan.typeB

def totalCost (a b : TruckType) (plan : RentalPlan) : ℕ :=
  a.rentalCost * plan.typeA + b.rentalCost * plan.typeB

/-- The main theorem stating the most cost-effective rental plan -/
theorem most_cost_effective_plan 
  (typeA typeB : TruckType)
  (h1 : 2 * typeA.capacity + typeB.capacity = 10)
  (h2 : typeA.capacity + 2 * typeB.capacity = 11)
  (h3 : typeA.rentalCost = 100)
  (h4 : typeB.rentalCost = 120) :
  ∃ (plan : RentalPlan),
    totalCapacity typeA typeB plan = 31 ∧
    (∀ (otherPlan : RentalPlan),
      totalCapacity typeA typeB otherPlan = 31 →
      totalCost typeA typeB plan ≤ totalCost typeA typeB otherPlan) ∧
    plan.typeA = 1 ∧
    plan.typeB = 7 ∧
    totalCost typeA typeB plan = 940 :=
  sorry

end NUMINAMATH_CALUDE_most_cost_effective_plan_l2673_267329


namespace NUMINAMATH_CALUDE_david_current_age_l2673_267345

/-- David's current age -/
def david_age : ℕ := sorry

/-- David's daughter's current age -/
def daughter_age : ℕ := 12

/-- Number of years until David's age is twice his daughter's -/
def years_until_double : ℕ := 16

theorem david_current_age :
  david_age = 40 ∧
  david_age + years_until_double = 2 * (daughter_age + years_until_double) :=
by sorry

end NUMINAMATH_CALUDE_david_current_age_l2673_267345


namespace NUMINAMATH_CALUDE_upper_bound_y_l2673_267362

theorem upper_bound_y (x y : ℤ) (h1 : 3 < x) (h2 : x < 6) (h3 : 6 < y) 
  (h4 : ∀ (a b : ℤ), 3 < a → a < 6 → 6 < b → b - a ≤ 6) : y ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_upper_bound_y_l2673_267362


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2673_267380

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometricSequenceTerm (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -1/2
  let r : ℚ := a₂ / a₁
  let a₇ : ℚ := geometricSequenceTerm a₁ r 7
  a₇ = 1/15552 :=
by sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2673_267380


namespace NUMINAMATH_CALUDE_intersection_A_B_l2673_267336

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {x | x > 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2673_267336


namespace NUMINAMATH_CALUDE_number_difference_l2673_267322

theorem number_difference (a b : ℕ) : 
  a + b = 30000 →
  b = 10 * a + 5 →
  b - a = 24548 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2673_267322


namespace NUMINAMATH_CALUDE_max_value_of_function_l2673_267305

theorem max_value_of_function (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∀ x y : ℝ, |x| + |y| ≤ 1 → (∀ x' y' : ℝ, |x'| + |y'| ≤ 1 → a * x + y ≤ a * x' + y') →
  a * x + y = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2673_267305


namespace NUMINAMATH_CALUDE_set_operations_l2673_267317

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x^2 - 4*x ≤ 0}

-- Define the theorem
theorem set_operations :
  (A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 4}) ∧
  (A ∩ (Bᶜ) = {x : ℝ | -1 ≤ x ∧ x < 0}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2673_267317


namespace NUMINAMATH_CALUDE_square_sum_product_inequality_l2673_267382

theorem square_sum_product_inequality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_inequality_l2673_267382


namespace NUMINAMATH_CALUDE_queen_center_probability_queen_center_probability_2004_l2673_267347

/-- Probability of queen being in center after n moves -/
def prob_queen_center (n : ℕ) : ℚ :=
  1/3 + 2/3 * (-1/2)^n

/-- Initial configuration with queen in center -/
def initial_config : List Char := ['R', 'Q', 'R']

/-- Theorem stating the probability of queen being in center after n moves -/
theorem queen_center_probability (n : ℕ) : 
  prob_queen_center n = 1/3 + 2/3 * (-1/2)^n :=
sorry

/-- Corollary for the specific case of 2004 moves -/
theorem queen_center_probability_2004 : 
  prob_queen_center 2004 = 1/3 + 1/(3 * 2^2003) :=
sorry

end NUMINAMATH_CALUDE_queen_center_probability_queen_center_probability_2004_l2673_267347


namespace NUMINAMATH_CALUDE_square_difference_identity_l2673_267361

theorem square_difference_identity : (25 + 15)^2 - (25 - 15)^2 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l2673_267361


namespace NUMINAMATH_CALUDE_total_lunch_cost_l2673_267386

/-- Calculates the total cost of lunch for all students in an elementary school --/
theorem total_lunch_cost (third_grade_classes fourth_grade_classes fifth_grade_classes : ℕ)
  (third_grade_students fourth_grade_students fifth_grade_students : ℕ)
  (hamburger_cost carrot_cost cookie_cost : ℚ) : ℚ :=
  by
  have h1 : third_grade_classes = 5 := by sorry
  have h2 : fourth_grade_classes = 4 := by sorry
  have h3 : fifth_grade_classes = 4 := by sorry
  have h4 : third_grade_students = 30 := by sorry
  have h5 : fourth_grade_students = 28 := by sorry
  have h6 : fifth_grade_students = 27 := by sorry
  have h7 : hamburger_cost = 2.1 := by sorry
  have h8 : carrot_cost = 0.5 := by sorry
  have h9 : cookie_cost = 0.2 := by sorry

  have total_students : ℕ := 
    third_grade_classes * third_grade_students + 
    fourth_grade_classes * fourth_grade_students + 
    fifth_grade_classes * fifth_grade_students

  have lunch_cost_per_student : ℚ := hamburger_cost + carrot_cost + cookie_cost

  have total_cost : ℚ := total_students * lunch_cost_per_student

  exact 1036

end NUMINAMATH_CALUDE_total_lunch_cost_l2673_267386


namespace NUMINAMATH_CALUDE_odd_integer_sequence_sum_l2673_267393

theorem odd_integer_sequence_sum (n : ℕ) : n > 0 → (
  let sum := n / 2 * (5 + (6 * n - 1))
  sum = 597 ↔ n = 13
) := by sorry

end NUMINAMATH_CALUDE_odd_integer_sequence_sum_l2673_267393


namespace NUMINAMATH_CALUDE_license_plate_difference_l2673_267366

/-- The number of possible digits in a license plate -/
def num_digits : ℕ := 10

/-- The number of possible letters in a license plate -/
def num_letters : ℕ := 26

/-- The number of possible license plates for Alpha (LLDDDDLL format) -/
def alpha_plates : ℕ := num_letters^4 * num_digits^4

/-- The number of possible license plates for Beta (LLLDDDD format) -/
def beta_plates : ℕ := num_letters^3 * num_digits^4

/-- The theorem stating the difference in number of license plates between Alpha and Beta -/
theorem license_plate_difference :
  alpha_plates - beta_plates = num_digits^4 * num_letters^3 * 25 := by
  sorry

#eval alpha_plates - beta_plates
#eval num_digits^4 * num_letters^3 * 25

end NUMINAMATH_CALUDE_license_plate_difference_l2673_267366


namespace NUMINAMATH_CALUDE_remaining_quarters_l2673_267350

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.50
def jeans_cost : ℚ := 11.50
def quarter_value : ℚ := 0.25

theorem remaining_quarters : 
  (initial_amount - (pizza_cost + soda_cost + jeans_cost)) / quarter_value = 97 := by
  sorry

end NUMINAMATH_CALUDE_remaining_quarters_l2673_267350


namespace NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l2673_267308

/-- Represents the fraction of orange juice in a mixture -/
def orange_juice_fraction (pitcher1_capacity pitcher2_capacity : ℚ)
  (pitcher1_oj_fraction pitcher2_oj_fraction : ℚ) : ℚ :=
  let total_oj := pitcher1_capacity * pitcher1_oj_fraction + pitcher2_capacity * pitcher2_oj_fraction
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_oj / total_volume

/-- Proves that the fraction of orange juice in the combined mixture is 29167/100000 -/
theorem orange_juice_mixture_fraction :
  orange_juice_fraction 800 800 (1/4) (1/3) = 29167/100000 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l2673_267308


namespace NUMINAMATH_CALUDE_average_weight_of_all_girls_l2673_267378

theorem average_weight_of_all_girls (group1_count : ℕ) (group1_avg : ℝ) 
  (group2_count : ℕ) (group2_avg : ℝ) : 
  group1_count = 16 → 
  group1_avg = 50.25 → 
  group2_count = 8 → 
  group2_avg = 45.15 → 
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  let total_count := group1_count + group2_count
  (total_weight / total_count) = 48.55 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_all_girls_l2673_267378


namespace NUMINAMATH_CALUDE_inequality_proof_l2673_267374

theorem inequality_proof (x : ℝ) (n : ℕ) (hx : x > 1) (hn : n > 1) :
  1 + (x - 1) / (n * x) < x^(1/n) ∧ x^(1/n) < 1 + (x - 1) / n :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2673_267374


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_difference_l2673_267397

-- Define the polynomial
def f (x : ℝ) : ℝ := 20 * x^3 - 40 * x^2 + 18 * x - 1

-- Define the roots
variable (a b c : ℝ)

-- State the theorem
theorem root_sum_reciprocal_difference (ha : f a = 0) (hb : f b = 0) (hc : f c = 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (ha_bounds : 0 < a ∧ a < 1) (hb_bounds : 0 < b ∧ b < 1) (hc_bounds : 0 < c ∧ c < 1) :
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_difference_l2673_267397


namespace NUMINAMATH_CALUDE_spaceDivisions_correct_l2673_267352

/-- The number of parts that n planes can divide space into, given that
    each group of three planes intersects at one point and no group of
    four planes has a common point. -/
def spaceDivisions (n : ℕ) : ℚ :=
  (n^3 + 5*n + 6) / 6

/-- Theorem stating that spaceDivisions correctly calculates the number
    of parts that n planes can divide space into. -/
theorem spaceDivisions_correct (n : ℕ) :
  spaceDivisions n = (n^3 + 5*n + 6) / 6 :=
by sorry

end NUMINAMATH_CALUDE_spaceDivisions_correct_l2673_267352


namespace NUMINAMATH_CALUDE_admissible_set_characterization_l2673_267377

def IsAdmissible (A : Set ℤ) : Prop :=
  ∀ x y k : ℤ, x ∈ A → y ∈ A → (x^2 + k*x*y + y^2) ∈ A

theorem admissible_set_characterization (m n : ℤ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (∀ A : Set ℤ, IsAdmissible A → m ∈ A → n ∈ A → A = Set.univ) ↔ Int.gcd m n = 1 :=
sorry

end NUMINAMATH_CALUDE_admissible_set_characterization_l2673_267377


namespace NUMINAMATH_CALUDE_probability_red_card_equal_suits_l2673_267313

structure Deck :=
  (total_cards : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = (red_suits + black_suits) * cards_per_suit)
  (h_equal_suits : red_suits = black_suits)

def probability_red_card (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit : ℚ) / d.total_cards

theorem probability_red_card_equal_suits (d : Deck) :
  probability_red_card d = 1 :=
sorry

end NUMINAMATH_CALUDE_probability_red_card_equal_suits_l2673_267313


namespace NUMINAMATH_CALUDE_expression_evaluation_l2673_267365

theorem expression_evaluation : 
  let cos_45 : ℝ := Real.sqrt 2 / 2
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (cos_45 - 3)) = (3 * Real.sqrt 3 - 5 * Real.sqrt 2) / 34 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2673_267365


namespace NUMINAMATH_CALUDE_factorization_equality_l2673_267320

theorem factorization_equality (x y : ℝ) : 4 * x^2 - 8 * x * y + 4 * y^2 = 4 * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2673_267320


namespace NUMINAMATH_CALUDE_new_person_weight_l2673_267351

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 5 →
  replaced_weight = 40 →
  avg_increase = 10 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 90 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2673_267351


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2673_267332

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let C : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (c, 0)
  let A : ℝ × ℝ := (x₀, y₀)
  (∃ x₀ y₀, C (x₀, y₀) ∧ (x₀ * b)^2 = (y₀ * a)^2) →  -- A is on the asymptote
  (x₀^2 + y₀^2 = c^2 / 4) →  -- A is on the circle with diameter OF
  (Real.cos (π/6) * c = b) →  -- ∠AFO = π/6
  c / a = 2 :=  -- eccentricity is 2
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2673_267332


namespace NUMINAMATH_CALUDE_sum_after_removal_l2673_267390

theorem sum_after_removal (numbers : List ℝ) (avg : ℝ) (removed : ℝ) :
  numbers.length = 8 →
  numbers.sum / numbers.length = avg →
  avg = 5.2 →
  removed = 4.6 →
  removed ∈ numbers →
  (numbers.erase removed).sum = 37 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_removal_l2673_267390


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2673_267338

/-- A geometric sequence of positive real numbers. -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_prod : a 2 * a 6 = 16)
  (h_sum : a 4 + a 8 = 8) :
  q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2673_267338


namespace NUMINAMATH_CALUDE_cola_cost_l2673_267309

/-- Proves that the cost of each cola bottle is $2 given the conditions of Wilson's purchase. -/
theorem cola_cost (hamburger_price : ℚ) (num_hamburgers : ℕ) (num_cola : ℕ) (discount : ℚ) (total_paid : ℚ) :
  hamburger_price = 5 →
  num_hamburgers = 2 →
  num_cola = 3 →
  discount = 4 →
  total_paid = 12 →
  (total_paid + discount - num_hamburgers * hamburger_price) / num_cola = 2 := by
  sorry

end NUMINAMATH_CALUDE_cola_cost_l2673_267309


namespace NUMINAMATH_CALUDE_cosine_equality_in_range_l2673_267358

theorem cosine_equality_in_range (n : ℤ) :
  100 ≤ n ∧ n ≤ 300 ∧ Real.cos (n * π / 180) = Real.cos (140 * π / 180) → n = 220 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_in_range_l2673_267358


namespace NUMINAMATH_CALUDE_power_equality_l2673_267310

theorem power_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 := by sorry

end NUMINAMATH_CALUDE_power_equality_l2673_267310


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l2673_267399

/-- A quadratic function f(x) = 4x^2 - kx - 8 has monotonicity on the interval (∞, 5] if and only if k ≥ 40 -/
theorem quadratic_monotonicity (k : ℝ) :
  (∀ x > 5, Monotone (fun x => 4 * x^2 - k * x - 8)) ↔ k ≥ 40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l2673_267399


namespace NUMINAMATH_CALUDE_number_equation_solution_l2673_267300

theorem number_equation_solution : 
  ∃ x : ℝ, (45 + 3 * x = 72) ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2673_267300


namespace NUMINAMATH_CALUDE_decimal_point_problem_l2673_267367

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 5 * (1 / x)) : 
  x = Real.sqrt 2 / 20 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l2673_267367


namespace NUMINAMATH_CALUDE_fifth_week_hours_l2673_267343

-- Define the required average hours per week
def required_average : ℝ := 12

-- Define the number of weeks
def num_weeks : ℕ := 5

-- Define the study hours for the first four weeks
def week1_hours : ℝ := 10
def week2_hours : ℝ := 14
def week3_hours : ℝ := 9
def week4_hours : ℝ := 13

-- Define the sum of study hours for the first four weeks
def sum_first_four_weeks : ℝ := week1_hours + week2_hours + week3_hours + week4_hours

-- Theorem to prove
theorem fifth_week_hours : 
  ∃ (x : ℝ), (sum_first_four_weeks + x) / num_weeks = required_average ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_fifth_week_hours_l2673_267343


namespace NUMINAMATH_CALUDE_cube_pyramid_sum_l2673_267307

/-- Represents a three-dimensional shape --/
structure Shape3D where
  edges : ℕ
  corners : ℕ
  faces : ℕ

/-- A cube --/
def cube : Shape3D :=
  { edges := 12, corners := 8, faces := 6 }

/-- A square pyramid --/
def square_pyramid : Shape3D :=
  { edges := 8, corners := 5, faces := 5 }

/-- The shape formed by placing a square pyramid on one face of a cube --/
def cube_with_pyramid : Shape3D :=
  { edges := cube.edges + 4, -- 4 new edges from pyramid apex
    corners := cube.corners + 1, -- 1 new corner (pyramid apex)
    faces := cube.faces + square_pyramid.faces - 1 } -- -1 for shared base

/-- The sum of edges, corners, and faces of the combined shape --/
def combined_sum (s : Shape3D) : ℕ :=
  s.edges + s.corners + s.faces

/-- Theorem stating that the sum of edges, corners, and faces of the combined shape is 34 --/
theorem cube_pyramid_sum :
  combined_sum cube_with_pyramid = 34 := by
  sorry


end NUMINAMATH_CALUDE_cube_pyramid_sum_l2673_267307


namespace NUMINAMATH_CALUDE_same_remainder_mod_27_l2673_267398

/-- Given a six-digit number X, Y is formed by moving the first three digits of X after the last three digits -/
def form_Y (X : ℕ) : ℕ :=
  let a := X / 1000
  let b := X % 1000
  1000 * b + a

/-- Theorem: For any six-digit number X, X and Y (formed from X) have the same remainder when divided by 27 -/
theorem same_remainder_mod_27 (X : ℕ) (h : 100000 ≤ X ∧ X < 1000000) :
  X % 27 = form_Y X % 27 := by
  sorry


end NUMINAMATH_CALUDE_same_remainder_mod_27_l2673_267398


namespace NUMINAMATH_CALUDE_arrangement_count_is_180_l2673_267389

/-- The number of ways to select 4 students from 5 and assign them to 3 subjects --/
def arrangement_count : ℕ := 180

/-- The total number of students --/
def total_students : ℕ := 5

/-- The number of students to be selected --/
def selected_students : ℕ := 4

/-- The number of subjects --/
def subject_count : ℕ := 3

/-- Theorem stating that the number of arrangements is 180 --/
theorem arrangement_count_is_180 :
  arrangement_count = 
    subject_count * 
    (Nat.choose total_students 2) * 
    (Nat.choose (total_students - 2) 1) * 
    (Nat.choose (total_students - 3) 1) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_180_l2673_267389


namespace NUMINAMATH_CALUDE_min_tiles_for_l_shape_min_tiles_for_specific_l_shape_l2673_267346

/-- Represents the dimensions of a rectangle in inches -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle in square inches -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the number of tiles needed to cover a rectangle -/
def tilesNeeded (r : Rectangle) (tileArea : ℕ) : ℕ := 
  (area r + tileArea - 1) / tileArea

theorem min_tiles_for_l_shape (tile : Rectangle) 
  (large : Rectangle) (small : Rectangle) : ℕ :=
  let tileArea := area tile
  let largeRect := Rectangle.mk (feetToInches large.length) (feetToInches large.width)
  let smallRect := Rectangle.mk (feetToInches small.length) (feetToInches small.width)
  tilesNeeded largeRect tileArea + tilesNeeded smallRect tileArea

theorem min_tiles_for_specific_l_shape : 
  min_tiles_for_l_shape (Rectangle.mk 2 6) (Rectangle.mk 3 4) (Rectangle.mk 2 1) = 168 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_l_shape_min_tiles_for_specific_l_shape_l2673_267346


namespace NUMINAMATH_CALUDE_sum_remainder_mod_13_l2673_267354

theorem sum_remainder_mod_13 : (9001 + 9002 + 9003 + 9004) % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_13_l2673_267354


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2673_267379

def A : Set Int := {-1, 1, 2}
def B : Set Int := {-2, -1, 0}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2673_267379


namespace NUMINAMATH_CALUDE_optimal_price_increase_maximizes_profit_l2673_267325

/-- Represents the daily profit function for a meal set -/
structure MealSet where
  baseProfit : ℝ
  baseSales : ℝ
  salesDecreaseRate : ℝ

/-- Calculate the daily profit for a meal set given a price increase -/
def dailyProfit (set : MealSet) (priceIncrease : ℝ) : ℝ :=
  (set.baseProfit + priceIncrease) * (set.baseSales - set.salesDecreaseRate * priceIncrease)

/-- The optimal price increase for meal set A maximizes the total profit -/
theorem optimal_price_increase_maximizes_profit 
  (setA setB : MealSet)
  (totalPriceIncrease : ℝ)
  (hA : setA = { baseProfit := 8, baseSales := 90, salesDecreaseRate := 4 })
  (hB : setB = { baseProfit := 10, baseSales := 70, salesDecreaseRate := 2 })
  (hTotal : totalPriceIncrease = 10) :
  ∃ (x : ℝ), x = 4 ∧ 
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ totalPriceIncrease →
      dailyProfit setA x + dailyProfit setB (totalPriceIncrease - x) ≥
      dailyProfit setA y + dailyProfit setB (totalPriceIncrease - y) :=
by sorry


end NUMINAMATH_CALUDE_optimal_price_increase_maximizes_profit_l2673_267325


namespace NUMINAMATH_CALUDE_scaling_transformation_result_l2673_267330

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y = 3 * Real.sin (2 * x)

-- Define the scaling transformation
def scaling_transformation (x y x' y' : ℝ) : Prop := x' = 2 * x ∧ y' = 3 * y

-- State the theorem
theorem scaling_transformation_result :
  ∀ (x y x' y' : ℝ),
  original_curve x y →
  scaling_transformation x y x' y' →
  y' = 9 * Real.sin x' := by sorry

end NUMINAMATH_CALUDE_scaling_transformation_result_l2673_267330


namespace NUMINAMATH_CALUDE_expression_evaluation_l2673_267301

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2673_267301


namespace NUMINAMATH_CALUDE_math_exam_problem_l2673_267311

theorem math_exam_problem (total : ℕ) (correct : ℕ) (incorrect : ℕ) :
  total = 120 →
  incorrect = 3 * correct →
  total = correct + incorrect →
  correct = 30 := by
sorry

end NUMINAMATH_CALUDE_math_exam_problem_l2673_267311


namespace NUMINAMATH_CALUDE_circle_equation_l2673_267371

theorem circle_equation (x y : ℝ) : 
  (∀ (x₀ y₀ : ℝ), (x₀ = 0 ∧ y₀ = 0) ∨ (x₀ = 4 ∧ y₀ = 0) ∨ (x₀ = -1 ∧ y₀ = 1) → 
    x₀^2 + y₀^2 - 4*x₀ - 6*y₀ = 0) ↔
  x^2 + y^2 - 4*x - 6*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2673_267371


namespace NUMINAMATH_CALUDE_quadratic_extremum_l2673_267363

/-- Given a quadratic function f(x) = ax^2 + bx + c where c = -b^2 / (3a),
    prove that the graph of y = f(x) has a maximum if a < 0 and a minimum if a > 0 -/
theorem quadratic_extremum (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x - b^2 / (3 * a)
  (a < 0 → ∃ x₀, ∀ x, f x ≤ f x₀) ∧
  (a > 0 → ∃ x₀, ∀ x, f x ≥ f x₀) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_extremum_l2673_267363


namespace NUMINAMATH_CALUDE_plane_perpendicular_theorem_l2673_267321

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_theorem 
  (α β : Plane) (n : Line) 
  (h1 : contains β n) 
  (h2 : perpendicular n α) :
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_theorem_l2673_267321


namespace NUMINAMATH_CALUDE_rhombus_area_fraction_l2673_267337

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rhombus defined by four vertices -/
structure Rhombus where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- The grid size -/
def gridSize : ℕ := 6

/-- The rhombus in question -/
def specialRhombus : Rhombus := {
  v1 := ⟨2, 2⟩,
  v2 := ⟨4, 2⟩,
  v3 := ⟨3, 3⟩,
  v4 := ⟨3, 1⟩
}

/-- Calculate the area of a rhombus -/
def rhombusArea (r : Rhombus) : ℝ := sorry

/-- Calculate the area of the grid -/
def gridArea : ℝ := gridSize ^ 2

/-- The main theorem to prove -/
theorem rhombus_area_fraction :
  rhombusArea specialRhombus / gridArea = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_fraction_l2673_267337


namespace NUMINAMATH_CALUDE_abc_divisible_by_four_l2673_267355

theorem abc_divisible_by_four (a b c d : ℤ) (h : a^2 + b^2 + c^2 = d^2) : 
  4 ∣ (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_abc_divisible_by_four_l2673_267355


namespace NUMINAMATH_CALUDE_cone_max_volume_l2673_267391

/-- A cone with slant height 20 cm has maximum volume when its height is (20√3)/3 cm. -/
theorem cone_max_volume (h : ℝ) (h_pos : 0 < h) (h_bound : h < 20) :
  let r := Real.sqrt (400 - h^2)
  let v := (1/3) * Real.pi * h * r^2
  (∀ h' : ℝ, 0 < h' → h' < 20 → 
    (1/3) * Real.pi * h' * (Real.sqrt (400 - h'^2))^2 ≤ v) →
  h = 20 * Real.sqrt 3 / 3 := by
sorry


end NUMINAMATH_CALUDE_cone_max_volume_l2673_267391


namespace NUMINAMATH_CALUDE_distribute_subtraction_l2673_267327

theorem distribute_subtraction (a b c : ℝ) : 5*a - (b + 2*c) = 5*a - b - 2*c := by
  sorry

end NUMINAMATH_CALUDE_distribute_subtraction_l2673_267327


namespace NUMINAMATH_CALUDE_complex_exponential_form_angle_l2673_267370

theorem complex_exponential_form_angle (z : ℂ) : 
  z = 2 - 2 * Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (4 * Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_form_angle_l2673_267370


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_one_l2673_267344

theorem sum_reciprocals_equals_one (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x + 1 / y = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_one_l2673_267344


namespace NUMINAMATH_CALUDE_k_value_theorem_l2673_267353

theorem k_value_theorem (a b k : ℕ+) :
  (a^2 + a * b + b^2 : ℚ) / (a * b - 1 : ℚ) = k →
  k = 4 ∨ k = 7 :=
by sorry

end NUMINAMATH_CALUDE_k_value_theorem_l2673_267353


namespace NUMINAMATH_CALUDE_difference_of_squares_l2673_267326

theorem difference_of_squares (x : ℝ) : x^2 - 36 = (x + 6) * (x - 6) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2673_267326


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l2673_267341

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 10*a*b) :
  |((a + b) / (a - b))| = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l2673_267341


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_of_squares_l2673_267314

theorem arithmetic_geometric_harmonic_mean_sum_of_squares
  (x y z : ℝ)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_of_squares_l2673_267314


namespace NUMINAMATH_CALUDE_parabola_perpendicular_chords_locus_l2673_267324

/-- Given a parabola y^2 = 4px where p > 0, with two perpendicular chords OA and OB
    drawn from the vertex O(0,0), the locus of the projection of O onto AB
    is a circle with equation (x - 2p)^2 + y^2 = 4p^2 -/
theorem parabola_perpendicular_chords_locus (p : ℝ) (h : p > 0) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*p*x}
  let O := (0 : ℝ × ℝ)
  let perpendicular_chords := {(OA, OB) : (ℝ × ℝ) × (ℝ × ℝ) |
    O.1 = 0 ∧ O.2 = 0 ∧
    OA ∈ parabola ∧ OB ∈ parabola ∧
    (OA.2 - O.2) * (OB.2 - O.2) = -(OA.1 - O.1) * (OB.1 - O.1)}
  let projection := {M : ℝ × ℝ | ∃ (OA OB : ℝ × ℝ), (OA, OB) ∈ perpendicular_chords ∧
    (M.2 - O.2) * (OA.1 - OB.1) = (M.1 - O.1) * (OA.2 - OB.2)}
  projection = {(x, y) : ℝ × ℝ | (x - 2*p)^2 + y^2 = 4*p^2} :=
by sorry


end NUMINAMATH_CALUDE_parabola_perpendicular_chords_locus_l2673_267324


namespace NUMINAMATH_CALUDE_arcsin_neg_half_equals_neg_pi_sixth_l2673_267348

theorem arcsin_neg_half_equals_neg_pi_sixth : 
  Real.arcsin (-0.5) = -π/6 := by sorry

end NUMINAMATH_CALUDE_arcsin_neg_half_equals_neg_pi_sixth_l2673_267348


namespace NUMINAMATH_CALUDE_five_students_two_groups_l2673_267340

/-- The number of ways to assign students to groups -/
def assignment_count (num_students : ℕ) (num_groups : ℕ) : ℕ :=
  num_groups ^ num_students

/-- Theorem: There are 32 ways to assign 5 students to 2 groups -/
theorem five_students_two_groups :
  assignment_count 5 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_five_students_two_groups_l2673_267340


namespace NUMINAMATH_CALUDE_three_digit_equation_solution_l2673_267312

/-- Given that 3 + 6AB = 691 and 6AB is a three-digit number, prove that A = 8 -/
theorem three_digit_equation_solution (A B : ℕ) : 
  (3 + 6 * A * 10 + B = 691) → 
  (100 ≤ 6 * A * 10 + B) →
  (6 * A * 10 + B < 1000) →
  A = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_equation_solution_l2673_267312


namespace NUMINAMATH_CALUDE_quinary_decimal_binary_conversion_l2673_267388

/-- Converts a quinary (base-5) number to decimal (base-10) --/
def quinary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal (base-10) number to binary (base-2) --/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec to_binary_aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else to_binary_aux (m / 2) ((m % 2) :: acc)
    to_binary_aux n []

theorem quinary_decimal_binary_conversion :
  let quinary := [4, 3, 2]  -- 234₅ in reverse order
  let decimal := 69
  let binary := [1, 0, 0, 0, 1, 0, 1]  -- 1000101₂
  (quinary_to_decimal quinary = decimal) ∧
  (decimal_to_binary decimal = binary) := by
  sorry

end NUMINAMATH_CALUDE_quinary_decimal_binary_conversion_l2673_267388


namespace NUMINAMATH_CALUDE_boys_ratio_in_class_l2673_267339

theorem boys_ratio_in_class (n_boys n_girls : ℕ) (h_prob : n_boys / (n_boys + n_girls) = 2/3 * (n_girls / (n_boys + n_girls))) :
  n_boys / (n_boys + n_girls) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_boys_ratio_in_class_l2673_267339


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l2673_267385

theorem min_value_sum_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 4) 
  (h2 : e * f * g * h = 9) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l2673_267385


namespace NUMINAMATH_CALUDE_factors_of_96_with_square_sum_208_l2673_267335

theorem factors_of_96_with_square_sum_208 :
  ∀ a b : ℕ+,
    a * b = 96 ∧ 
    a^2 + b^2 = 208 →
    (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) := by
  sorry

end NUMINAMATH_CALUDE_factors_of_96_with_square_sum_208_l2673_267335


namespace NUMINAMATH_CALUDE_f_sum_zero_three_l2673_267387

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_sum_zero_three (f : ℝ → ℝ) 
  (h1 : isOddFunction f) 
  (h2 : ∀ x, x < 0 → f x = x + 2) : 
  f 0 + f 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_f_sum_zero_three_l2673_267387


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2673_267392

theorem sum_of_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2673_267392


namespace NUMINAMATH_CALUDE_sum_first_100_odd_integers_l2673_267331

/-- The sum of the first n positive odd integers -/
def sumFirstNOddIntegers (n : ℕ) : ℕ :=
  n * n

theorem sum_first_100_odd_integers :
  sumFirstNOddIntegers 100 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_100_odd_integers_l2673_267331


namespace NUMINAMATH_CALUDE_winter_sales_calculation_l2673_267303

/-- Represents the sales of pizzas in millions for each season -/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- Calculates the total annual sales given the seasonal sales -/
def totalAnnualSales (sales : SeasonalSales) : ℝ :=
  sales.spring + sales.summer + sales.fall + sales.winter

/-- Theorem: Given the conditions, the number of pizzas sold in winter is 6.6 million -/
theorem winter_sales_calculation (sales : SeasonalSales)
    (h1 : sales.fall = 0.2 * totalAnnualSales sales)
    (h2 : sales.winter = 1.1 * sales.summer)
    (h3 : sales.spring = 5)
    (h4 : sales.summer = 6) :
    sales.winter = 6.6 := by
  sorry

#check winter_sales_calculation

end NUMINAMATH_CALUDE_winter_sales_calculation_l2673_267303


namespace NUMINAMATH_CALUDE_bulls_win_probability_l2673_267384

-- Define the probability of Heat winning a single game
def heat_win_prob : ℚ := 3/4

-- Define the probability of Bulls winning a single game
def bulls_win_prob : ℚ := 1 - heat_win_prob

-- Define the number of games needed to win the series
def games_to_win : ℕ := 4

-- Define the total number of games in a full series
def total_games : ℕ := 7

-- Define the function to calculate the probability of Bulls winning in 7 games
def bulls_win_in_seven : ℚ :=
  -- Probability of 3-3 tie after 6 games
  (Nat.choose 6 3 : ℚ) * bulls_win_prob^3 * heat_win_prob^3 *
  -- Probability of Bulls winning the 7th game
  bulls_win_prob

-- Theorem statement
theorem bulls_win_probability :
  bulls_win_in_seven = 540 / 16384 := by sorry

end NUMINAMATH_CALUDE_bulls_win_probability_l2673_267384


namespace NUMINAMATH_CALUDE_clara_cookie_sales_l2673_267396

/-- Represents the number of cookies in a box for each type -/
def cookies_per_box : Fin 3 → ℕ
  | 0 => 12
  | 1 => 20
  | 2 => 16

/-- Represents the number of boxes sold for each type -/
def boxes_sold : Fin 3 → ℕ
  | 0 => 50
  | 1 => 80
  | 2 => 70

/-- Calculates the total number of cookies sold -/
def total_cookies_sold : ℕ :=
  (cookies_per_box 0 * boxes_sold 0) +
  (cookies_per_box 1 * boxes_sold 1) +
  (cookies_per_box 2 * boxes_sold 2)

theorem clara_cookie_sales :
  total_cookies_sold = 3320 := by
  sorry

end NUMINAMATH_CALUDE_clara_cookie_sales_l2673_267396


namespace NUMINAMATH_CALUDE_investment_rate_proof_l2673_267334

/-- Proves that the remaining investment rate is 7% given the specified conditions --/
theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ) :
  total_investment = 12000 →
  first_investment = 5000 →
  second_investment = 4000 →
  first_rate = 0.05 →
  second_rate = 0.035 →
  desired_income = 600 →
  let remaining_investment := total_investment - first_investment - second_investment
  let first_income := first_investment * first_rate
  let second_income := second_investment * second_rate
  let remaining_income := desired_income - first_income - second_income
  remaining_income / remaining_investment = 0.07 :=
by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l2673_267334


namespace NUMINAMATH_CALUDE_total_amount_calculation_l2673_267342

theorem total_amount_calculation (r p q : ℝ) 
  (h1 : r = 2000.0000000000002) 
  (h2 : r = (2/3) * (p + q + r)) : 
  p + q + r = 3000.0000000000003 := by
sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l2673_267342


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2673_267360

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 1}
def N : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 5}

-- State the theorem
theorem intersection_complement_theorem :
  N ∩ (Mᶜ) = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2673_267360
