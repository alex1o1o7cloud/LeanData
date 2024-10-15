import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l973_97342

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_a1 : a 1 = 2) (h_a4 : a 4 = 4) : a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l973_97342


namespace NUMINAMATH_CALUDE_hybrid_car_journey_length_l973_97372

theorem hybrid_car_journey_length :
  ∀ (d : ℝ),
  d > 60 →
  (60 : ℝ) / d + (d - 60) / (0.04 * (d - 60)) = 50 →
  d = 120 := by
  sorry

end NUMINAMATH_CALUDE_hybrid_car_journey_length_l973_97372


namespace NUMINAMATH_CALUDE_factorization_of_5x_squared_minus_5_l973_97384

theorem factorization_of_5x_squared_minus_5 (x : ℝ) : 5 * x^2 - 5 = 5 * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_5x_squared_minus_5_l973_97384


namespace NUMINAMATH_CALUDE_sqrt_72_div_sqrt_8_minus_abs_neg_2_equals_1_l973_97356

theorem sqrt_72_div_sqrt_8_minus_abs_neg_2_equals_1 :
  Real.sqrt 72 / Real.sqrt 8 - |(-2)| = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_72_div_sqrt_8_minus_abs_neg_2_equals_1_l973_97356


namespace NUMINAMATH_CALUDE_product_234_75_in_base5_l973_97318

/-- Converts a decimal number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of base 5 digits to a decimal number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

/-- Multiplies two numbers in base 5 representation -/
def multiplyBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem product_234_75_in_base5 :
  let a := toBase5 234
  let b := toBase5 75
  multiplyBase5 a b = [4, 5, 0, 6, 2, 0] :=
sorry

end NUMINAMATH_CALUDE_product_234_75_in_base5_l973_97318


namespace NUMINAMATH_CALUDE_increasing_decreasing_behavior_l973_97330

theorem increasing_decreasing_behavior 
  (f : ℝ → ℝ) (a : ℝ) (n : ℕ) 
  (h_f : ∀ x, f x = a * x ^ n) 
  (h_a : a ≠ 0) :
  (n % 2 = 0 ∧ a > 0 → ∀ x ≠ 0, deriv f x > 0) ∧
  (n % 2 = 0 ∧ a < 0 → ∀ x ≠ 0, deriv f x < 0) ∧
  (n % 2 = 1 ∧ a > 0 → (∀ x > 0, deriv f x > 0) ∧ (∀ x < 0, deriv f x < 0)) ∧
  (n % 2 = 1 ∧ a < 0 → (∀ x > 0, deriv f x < 0) ∧ (∀ x < 0, deriv f x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_decreasing_behavior_l973_97330


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l973_97336

theorem complex_fraction_equality : (1 + 3*Complex.I) / (Complex.I - 1) = 1 - 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l973_97336


namespace NUMINAMATH_CALUDE_specific_classroom_seats_l973_97309

/-- Represents a tiered classroom with increasing seats per row -/
structure TieredClassroom where
  rows : ℕ
  firstRowSeats : ℕ
  seatIncrease : ℕ

/-- Calculates the number of seats in the nth row -/
def seatsInRow (c : TieredClassroom) (n : ℕ) : ℕ :=
  c.firstRowSeats + (n - 1) * c.seatIncrease

/-- Calculates the total number of seats in the classroom -/
def totalSeats (c : TieredClassroom) : ℕ :=
  (c.firstRowSeats + seatsInRow c c.rows) * c.rows / 2

/-- Theorem stating the total number of seats in the specific classroom configuration -/
theorem specific_classroom_seats :
  let c : TieredClassroom := { rows := 22, firstRowSeats := 22, seatIncrease := 2 }
  totalSeats c = 946 := by sorry

end NUMINAMATH_CALUDE_specific_classroom_seats_l973_97309


namespace NUMINAMATH_CALUDE_range_of_a_l973_97344

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → a ∈ Set.Ioo (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l973_97344


namespace NUMINAMATH_CALUDE_adjacent_edge_angle_is_45_degrees_l973_97369

/-- A regular tetrahedron with coinciding centers of inscribed and circumscribed spheres -/
structure RegularTetrahedron where
  -- The tetrahedron is regular
  is_regular : Bool
  -- The center of the circumscribed sphere coincides with the center of the inscribed sphere
  centers_coincide : Bool

/-- The angle between two adjacent edges of a regular tetrahedron -/
def adjacent_edge_angle (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem: The angle between two adjacent edges of a regular tetrahedron 
    with coinciding sphere centers is 45 degrees -/
theorem adjacent_edge_angle_is_45_degrees (t : RegularTetrahedron) 
  (h1 : t.is_regular = true) 
  (h2 : t.centers_coincide = true) : 
  adjacent_edge_angle t = 45 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_adjacent_edge_angle_is_45_degrees_l973_97369


namespace NUMINAMATH_CALUDE_fraction_simplification_l973_97366

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l973_97366


namespace NUMINAMATH_CALUDE_beijing_to_lanzhou_distance_l973_97341

/-- The distance from Beijing to Lanzhou, given the distances from Beijing to Lhasa (via Lanzhou) and from Lanzhou to Lhasa. -/
theorem beijing_to_lanzhou_distance 
  (beijing_to_lhasa : ℕ) 
  (lanzhou_to_lhasa : ℕ) 
  (h1 : beijing_to_lhasa = 3985)
  (h2 : lanzhou_to_lhasa = 2054) :
  beijing_to_lhasa - lanzhou_to_lhasa = 1931 :=
by sorry

end NUMINAMATH_CALUDE_beijing_to_lanzhou_distance_l973_97341


namespace NUMINAMATH_CALUDE_equal_squares_sum_l973_97326

theorem equal_squares_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 0 → a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_squares_sum_l973_97326


namespace NUMINAMATH_CALUDE_additional_bags_needed_l973_97361

/-- The number of people guaranteed to show up -/
def guaranteed_visitors : ℕ := 50

/-- The number of additional people who might show up -/
def potential_visitors : ℕ := 40

/-- The number of extravagant gift bags already made -/
def extravagant_bags : ℕ := 10

/-- The number of average gift bags already made -/
def average_bags : ℕ := 20

/-- The total number of visitors Carl is preparing for -/
def total_visitors : ℕ := guaranteed_visitors + potential_visitors

/-- The total number of gift bags already made -/
def existing_bags : ℕ := extravagant_bags + average_bags

/-- Theorem stating the number of additional bags Carl needs to make -/
theorem additional_bags_needed : total_visitors - existing_bags = 60 := by
  sorry

end NUMINAMATH_CALUDE_additional_bags_needed_l973_97361


namespace NUMINAMATH_CALUDE_common_number_in_list_l973_97371

theorem common_number_in_list (list : List ℝ) : 
  list.length = 9 →
  (list.take 5).sum / 5 = 7 →
  (list.drop 4).sum / 5 = 9 →
  list.sum / 9 = 73 / 9 →
  ∃ x ∈ list.take 5 ∩ list.drop 4, x = 7 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_list_l973_97371


namespace NUMINAMATH_CALUDE_unique_cube_prime_factor_l973_97390

def greatest_prime_factor (n : ℕ) : ℕ := sorry

theorem unique_cube_prime_factor : 
  ∃! n : ℕ, n > 1 ∧ 
    (greatest_prime_factor n = n^(1/3)) ∧ 
    (greatest_prime_factor (n + 200) = (n + 200)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_prime_factor_l973_97390


namespace NUMINAMATH_CALUDE_wage_decrease_percentage_l973_97355

theorem wage_decrease_percentage (W : ℝ) (P : ℝ) : 
  W > 0 →  -- Wages are positive
  0.20 * (W - (P / 100) * W) = 0.50 * (0.30 * W) → 
  P = 25 :=
by sorry

end NUMINAMATH_CALUDE_wage_decrease_percentage_l973_97355


namespace NUMINAMATH_CALUDE_wooden_planks_weight_l973_97350

theorem wooden_planks_weight
  (crate_capacity : ℕ)
  (num_crates : ℕ)
  (num_nail_bags : ℕ)
  (nail_bag_weight : ℕ)
  (num_hammer_bags : ℕ)
  (hammer_bag_weight : ℕ)
  (num_plank_bags : ℕ)
  (weight_to_leave_out : ℕ)
  (h1 : crate_capacity = 20)
  (h2 : num_crates = 15)
  (h3 : num_nail_bags = 4)
  (h4 : nail_bag_weight = 5)
  (h5 : num_hammer_bags = 12)
  (h6 : hammer_bag_weight = 5)
  (h7 : num_plank_bags = 10)
  (h8 : weight_to_leave_out = 80) :
  (num_crates * crate_capacity - weight_to_leave_out
    - (num_nail_bags * nail_bag_weight + num_hammer_bags * hammer_bag_weight))
  / num_plank_bags = 14 := by
sorry

end NUMINAMATH_CALUDE_wooden_planks_weight_l973_97350


namespace NUMINAMATH_CALUDE_line_parallel_to_countless_lines_l973_97357

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_to_plane : Line → Plane → Prop)

-- Define the containment relation of a line in a plane
variable (contained_in : Line → Plane → Prop)

-- Define a property for a line being parallel to countless lines in a plane
variable (parallel_to_countless_lines : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_countless_lines 
  (a b : Line) (α : Plane) :
  parallel a b → contained_in b α → 
  parallel_to_countless_lines a α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_countless_lines_l973_97357


namespace NUMINAMATH_CALUDE_polynomial_root_product_l973_97358

theorem polynomial_root_product (b c : ℤ) : 
  (∀ r : ℝ, r^2 - r - 2 = 0 → r^4 - b*r - c = 0) → b*c = 30 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l973_97358


namespace NUMINAMATH_CALUDE_area_triangle_AOC_l973_97334

/-- Circle C with equation x^2 + y^2 - 4x - 6y + 12 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 12 = 0

/-- Point A with coordinates (3, 5) -/
def point_A : ℝ × ℝ := (3, 5)

/-- Origin O -/
def point_O : ℝ × ℝ := (0, 0)

/-- Point C is the center of the circle -/
def point_C : ℝ × ℝ := (2, 3)

/-- The area of triangle AOC is 1/2 -/
theorem area_triangle_AOC :
  let A := point_A
  let O := point_O
  let C := point_C
  (1/2 : ℝ) * ‖(A.1 - O.1, A.2 - O.2)‖ * ‖(C.1 - O.1, C.2 - O.2)‖ * 
    Real.sin (Real.arccos ((A.1 - O.1) * (C.1 - O.1) + (A.2 - O.2) * (C.2 - O.2)) / 
    (‖(A.1 - O.1, A.2 - O.2)‖ * ‖(C.1 - O.1, C.2 - O.2)‖)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_area_triangle_AOC_l973_97334


namespace NUMINAMATH_CALUDE_triangle_side_equality_l973_97377

theorem triangle_side_equality (A B C : Real) (a b c : Real) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →  -- angles are positive and less than π
  (A + B + C = π) →  -- sum of angles in a triangle
  (a > 0) ∧ (b > 0) ∧ (c > 0) →  -- sides are positive
  (a / Real.sin A = b / Real.sin B) →  -- Law of Sines
  (a / Real.sin A = c / Real.sin C) →  -- Law of Sines
  (3 * b * Real.cos C + 3 * c * Real.cos B = a^2) →  -- given condition
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_equality_l973_97377


namespace NUMINAMATH_CALUDE_min_quotient_base12_number_l973_97306

/-- Represents a digit in base 12, ranging from 1 to 10 (in base 10) -/
def Digit12 := {d : ℕ // 1 ≤ d ∧ d ≤ 10}

/-- Converts a base 12 number to base 10 -/
def toBase10 (a b c : Digit12) : ℕ :=
  144 * a.val + 12 * b.val + c.val

/-- Calculates the sum of digits in base 10 -/
def digitSum (a b c : Digit12) : ℕ :=
  a.val + b.val + c.val

/-- The main theorem stating the minimum quotient -/
theorem min_quotient_base12_number :
  ∀ a b c : Digit12,
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (toBase10 a b c : ℚ) / (digitSum a b c) ≥ 24.5 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_base12_number_l973_97306


namespace NUMINAMATH_CALUDE_union_equals_A_A_subset_complement_B_l973_97346

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 2}

-- Theorem 1
theorem union_equals_A (m : ℝ) : A ∪ B m = A → m = 1 := by
  sorry

-- Theorem 2
theorem A_subset_complement_B (m : ℝ) : A ⊆ (B m)ᶜ → m > 5 ∨ m < -3 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_A_subset_complement_B_l973_97346


namespace NUMINAMATH_CALUDE_min_value_expression_l973_97343

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((x^2 + y^2) * (3 * x^2 + y^2))) / (x * y) ≥ 1 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l973_97343


namespace NUMINAMATH_CALUDE_f_negative_a_eq_zero_l973_97391

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_negative_a_eq_zero (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_eq_zero_l973_97391


namespace NUMINAMATH_CALUDE_evelyns_marbles_l973_97314

theorem evelyns_marbles (initial_marbles : ℕ) : 
  initial_marbles + 9 = 104 → initial_marbles = 95 := by
  sorry

end NUMINAMATH_CALUDE_evelyns_marbles_l973_97314


namespace NUMINAMATH_CALUDE_derivative_odd_implies_a_eq_neg_one_l973_97360

/-- Given a real number a and a function f(x) = e^x - ae^(-x), 
    if the derivative of f is an odd function, then a = -1. -/
theorem derivative_odd_implies_a_eq_neg_one (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.exp x - a * Real.exp (-x)
  let f' : ℝ → ℝ := λ x ↦ Real.exp x + a * Real.exp (-x)
  (∀ x, f' x = -f' (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_odd_implies_a_eq_neg_one_l973_97360


namespace NUMINAMATH_CALUDE_office_episodes_l973_97310

theorem office_episodes (total_episodes : ℕ) (weeks : ℕ) (wednesday_episodes : ℕ) 
  (h1 : total_episodes = 201)
  (h2 : weeks = 67)
  (h3 : wednesday_episodes = 2) :
  ∃ monday_episodes : ℕ, 
    weeks * (monday_episodes + wednesday_episodes) = total_episodes ∧ 
    monday_episodes = 1 := by
  sorry

end NUMINAMATH_CALUDE_office_episodes_l973_97310


namespace NUMINAMATH_CALUDE_inequality_proof_l973_97364

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum : x + 2*y + 3*z = 11/12) :
  6*(3*x*y + 4*x*z + 2*y*z) + 6*x + 3*y + 4*z + 72*x*y*z ≤ 107/18 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l973_97364


namespace NUMINAMATH_CALUDE_combined_salaries_BCDE_l973_97396

def salary_A : ℕ := 9000
def average_salary : ℕ := 8200
def num_employees : ℕ := 5

theorem combined_salaries_BCDE :
  salary_A + (num_employees - 1) * (average_salary * num_employees - salary_A) / (num_employees - 1) = average_salary * num_employees :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_BCDE_l973_97396


namespace NUMINAMATH_CALUDE_chess_game_probability_l973_97395

theorem chess_game_probability (draw_prob win_b_prob : ℚ) :
  draw_prob = 1/2 →
  win_b_prob = 1/3 →
  1 - win_b_prob = 2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l973_97395


namespace NUMINAMATH_CALUDE_sin_equality_solution_l973_97305

theorem sin_equality_solution (x : Real) (h1 : x ∈ Set.Icc 0 (2 * Real.pi)) 
  (h2 : Real.sin x = Real.sin (Real.arcsin (2/3) - Real.arcsin (-1/3))) : 
  x = Real.arcsin ((4 * Real.sqrt 2 + Real.sqrt 5) / 9) ∨ 
  x = Real.pi - Real.arcsin ((4 * Real.sqrt 2 + Real.sqrt 5) / 9) := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_solution_l973_97305


namespace NUMINAMATH_CALUDE_rectangle_side_length_l973_97387

/-- Given two rectangles A and B, with sides (a, b) and (c, d) respectively,
    where the ratio of corresponding sides is 3/4 and rectangle B has sides 4 and 8,
    prove that the side a of rectangle A is 3. -/
theorem rectangle_side_length (a b c d : ℝ) : 
  a / c = 3 / 4 →
  b / d = 3 / 4 →
  c = 4 →
  d = 8 →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l973_97387


namespace NUMINAMATH_CALUDE_complex_number_problem_l973_97365

/-- Given a complex number z where z + 2i and z / (2 - i) are real numbers, 
    z = 4 - 2i and (z + ai)² is in the first quadrant when 2 < a < 6 -/
theorem complex_number_problem (z : ℂ) 
  (h1 : (z + 2*Complex.I).im = 0)
  (h2 : (z / (2 - Complex.I)).im = 0) :
  z = 4 - 2*Complex.I ∧ 
  ∀ a : ℝ, (z + a*Complex.I)^2 ∈ {w : ℂ | w.re > 0 ∧ w.im > 0} ↔ 2 < a ∧ a < 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l973_97365


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l973_97320

-- Define the foci of the ellipse
def F1 : ℝ × ℝ := (3, 15)
def F2 : ℝ × ℝ := (28, 45)

-- Define the reflection of F1 over the y-axis
def F1_reflected : ℝ × ℝ := (-3, 15)

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) (k : ℝ) : Prop :=
  dist P F1 + dist P F2 = k

-- Define the tangency condition
def is_tangent_to_y_axis (k : ℝ) : Prop :=
  ∃ y : ℝ, is_on_ellipse (0, y) k ∧
    ∀ y' : ℝ, is_on_ellipse (0, y') k → y = y'

-- State the theorem
theorem ellipse_major_axis_length :
  ∃ k : ℝ, is_tangent_to_y_axis k ∧ k = dist F1_reflected F2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l973_97320


namespace NUMINAMATH_CALUDE_toms_age_difference_l973_97363

theorem toms_age_difference (sister_age : ℕ) : 
  sister_age + 9 = 14 →
  2 * sister_age - 9 = 1 := by
sorry

end NUMINAMATH_CALUDE_toms_age_difference_l973_97363


namespace NUMINAMATH_CALUDE_parentheses_equivalence_l973_97337

theorem parentheses_equivalence (a b c : ℝ) : a + 2*b - 3*c = a + (2*b - 3*c) := by
  sorry

end NUMINAMATH_CALUDE_parentheses_equivalence_l973_97337


namespace NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l973_97329

theorem smallest_x_multiple_of_53 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ ((3*y)^2 + 3*41*(3*y) + 41^2))) ∧
  (53 ∣ ((3*x)^2 + 3*41*(3*x) + 41^2)) ∧
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l973_97329


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l973_97340

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.3 : ℝ)⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l973_97340


namespace NUMINAMATH_CALUDE_total_dangerous_animals_l973_97308

def crocodiles : ℕ := 22
def alligators : ℕ := 23
def vipers : ℕ := 5

theorem total_dangerous_animals : crocodiles + alligators + vipers = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_dangerous_animals_l973_97308


namespace NUMINAMATH_CALUDE_extra_domino_possible_l973_97338

/-- Represents a 6x6 chessboard -/
def Chessboard := Fin 6 → Fin 6 → Bool

/-- A domino is a pair of adjacent squares on the chessboard -/
def Domino := (Fin 6 × Fin 6) × (Fin 6 × Fin 6)

/-- Checks if two squares are adjacent -/
def adjacent (s1 s2 : Fin 6 × Fin 6) : Prop :=
  (s1.1 = s2.1 ∧ s1.2.succ = s2.2) ∨
  (s1.1 = s2.1 ∧ s1.2 = s2.2.succ) ∨
  (s1.1.succ = s2.1 ∧ s1.2 = s2.2) ∨
  (s1.1 = s2.1.succ ∧ s1.2 = s2.2)

/-- Checks if a domino is valid (covers two adjacent squares) -/
def validDomino (d : Domino) : Prop :=
  adjacent d.1 d.2

/-- Checks if two dominoes overlap -/
def overlap (d1 d2 : Domino) : Prop :=
  d1.1 = d2.1 ∨ d1.1 = d2.2 ∨ d1.2 = d2.1 ∨ d1.2 = d2.2

/-- Represents a configuration of 11 dominoes on the chessboard -/
def Configuration := Fin 11 → Domino

/-- Checks if a configuration is valid (no overlaps) -/
def validConfiguration (config : Configuration) : Prop :=
  ∀ i j : Fin 11, i ≠ j → ¬(overlap (config i) (config j))

/-- Theorem: Given a valid configuration of 11 dominoes on a 6x6 chessboard,
    there always exists at least two adjacent empty squares -/
theorem extra_domino_possible (config : Configuration) 
  (h_valid : validConfiguration config) :
  ∃ s1 s2 : Fin 6 × Fin 6, adjacent s1 s2 ∧
    (∀ i : Fin 11, s1 ≠ (config i).1 ∧ s1 ≠ (config i).2 ∧
                   s2 ≠ (config i).1 ∧ s2 ≠ (config i).2) :=
  sorry


end NUMINAMATH_CALUDE_extra_domino_possible_l973_97338


namespace NUMINAMATH_CALUDE_c_value_is_one_l973_97335

/-- The quadratic function f(x) = -x^2 + cx + 12 is positive only on (-∞, -3) ∪ (4, ∞) -/
def is_positive_on_intervals (c : ℝ) : Prop :=
  ∀ x : ℝ, (-x^2 + c*x + 12 > 0) ↔ (x < -3 ∨ x > 4)

/-- The value of c for which f(x) = -x^2 + cx + 12 is positive only on (-∞, -3) ∪ (4, ∞) is 1 -/
theorem c_value_is_one :
  ∃! c : ℝ, is_positive_on_intervals c ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_c_value_is_one_l973_97335


namespace NUMINAMATH_CALUDE_pure_imaginary_product_imaginary_part_quotient_l973_97354

-- Define complex numbers z₁ and z₂
def z₁ (m : ℝ) : ℂ := m + Complex.I
def z₂ (m : ℝ) : ℂ := 2 + m * Complex.I

-- Part 1
theorem pure_imaginary_product (m : ℝ) :
  (z₁ m * z₂ m).re = 0 → m = 0 :=
sorry

-- Part 2
theorem imaginary_part_quotient (m : ℝ) :
  z₁ m ^ 2 - 2 * z₁ m + 2 = 0 →
  (z₂ m / z₁ m).im = -1/2 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_imaginary_part_quotient_l973_97354


namespace NUMINAMATH_CALUDE_max_value_x_plus_2cos_x_l973_97333

open Real

theorem max_value_x_plus_2cos_x (x : ℝ) :
  let f : ℝ → ℝ := λ x => x + 2 * cos x
  (∀ y ∈ Set.Icc 0 (π / 2), f (π / 6) ≥ f y) ∧
  (π / 6 ∈ Set.Icc 0 (π / 2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2cos_x_l973_97333


namespace NUMINAMATH_CALUDE_range_of_m_l973_97394

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ (a b : ℝ), a > 0 → b > 0 → (1/a + 1/b) * Real.sqrt (a^2 + b^2) ≥ 2*m - 4) : 
  m ≤ 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l973_97394


namespace NUMINAMATH_CALUDE_power_sum_equality_l973_97303

theorem power_sum_equality : (3^2)^2 + (2^3)^3 = 593 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l973_97303


namespace NUMINAMATH_CALUDE_distinct_configurations_eq_seven_l973_97322

/-- The group of 2D rotations and flips for a 2x3 rectangle -/
inductive SymmetryGroup
| identity
| rotation180
| flipVertical
| flipHorizontal

/-- A configuration of red and yellow cubes in a 2x3 rectangle -/
def Configuration := Fin 6 → Bool

/-- The number of elements in the symmetry group -/
def symmetryGroupSize : ℕ := 4

/-- The total number of configurations -/
def totalConfigurations : ℕ := Nat.choose 6 3

/-- Function to count fixed points for each symmetry operation -/
noncomputable def fixedPoints (g : SymmetryGroup) : ℕ :=
  match g with
  | SymmetryGroup.identity => totalConfigurations
  | _ => 3  -- For rotation180, flipVertical, and flipHorizontal

/-- The sum of fixed points for all symmetry operations -/
noncomputable def totalFixedPoints : ℕ :=
  (fixedPoints SymmetryGroup.identity) +
  (fixedPoints SymmetryGroup.rotation180) +
  (fixedPoints SymmetryGroup.flipVertical) +
  (fixedPoints SymmetryGroup.flipHorizontal)

/-- The number of distinct configurations -/
noncomputable def distinctConfigurations : ℕ :=
  totalFixedPoints / symmetryGroupSize

theorem distinct_configurations_eq_seven :
  distinctConfigurations = 7 := by sorry

end NUMINAMATH_CALUDE_distinct_configurations_eq_seven_l973_97322


namespace NUMINAMATH_CALUDE_solve_equation_l973_97307

theorem solve_equation : 
  let x := 70 / (8 - 3/4)
  x = 280/29 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l973_97307


namespace NUMINAMATH_CALUDE_sports_store_sales_l973_97389

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 10

/-- The number of customers in each car -/
def customers_per_car : ℕ := 5

/-- The number of sales made by the music store -/
def music_store_sales : ℕ := 30

/-- The total number of customers in the parking lot -/
def total_customers : ℕ := num_cars * customers_per_car

theorem sports_store_sales :
  total_customers - music_store_sales = 20 := by
  sorry

#check sports_store_sales

end NUMINAMATH_CALUDE_sports_store_sales_l973_97389


namespace NUMINAMATH_CALUDE_intersection_of_sets_l973_97397

theorem intersection_of_sets :
  let A : Set ℤ := {-1, 2, 4}
  let B : Set ℤ := {0, 2, 6}
  A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l973_97397


namespace NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l973_97349

theorem smallest_base_for_fourth_power (b : ℕ) : 
  b > 0 ∧ 
  (∃ (x : ℕ), 7 * b^2 + 7 * b + 7 = x^4) ∧
  (∀ (c : ℕ), 0 < c ∧ c < b → ¬∃ (y : ℕ), 7 * c^2 + 7 * c + 7 = y^4) → 
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l973_97349


namespace NUMINAMATH_CALUDE_point_on_y_axis_l973_97347

theorem point_on_y_axis (x : ℝ) :
  (x^2 - 1 = 0) → 
  ((x^2 - 1, 2*x + 4) = (0, 6) ∨ (x^2 - 1, 2*x + 4) = (0, 2)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l973_97347


namespace NUMINAMATH_CALUDE_range_of_a_l973_97370

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) a, f x ∈ Set.Icc (-5 : ℝ) 4) ∧
  (∃ x₁ ∈ Set.Icc (-2 : ℝ) a, f x₁ = -5) ∧
  (∃ x₂ ∈ Set.Icc (-2 : ℝ) a, f x₂ = 4) →
  a ∈ Set.Icc (1 : ℝ) 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l973_97370


namespace NUMINAMATH_CALUDE_books_per_bookshelf_l973_97315

theorem books_per_bookshelf (num_bookshelves : ℕ) (magazines_per_bookshelf : ℕ) (total_items : ℕ) : 
  num_bookshelves = 29 →
  magazines_per_bookshelf = 61 →
  total_items = 2436 →
  (total_items - num_bookshelves * magazines_per_bookshelf) / num_bookshelves = 23 := by
sorry

end NUMINAMATH_CALUDE_books_per_bookshelf_l973_97315


namespace NUMINAMATH_CALUDE_faculty_reduction_percentage_l973_97368

theorem faculty_reduction_percentage (original : ℕ) (reduced : ℕ) : 
  original = 260 → reduced = 195 → 
  (original - reduced : ℚ) / original * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_faculty_reduction_percentage_l973_97368


namespace NUMINAMATH_CALUDE_complex_number_location_l973_97325

theorem complex_number_location :
  ∀ (z : ℂ), (z * Complex.I = 1 - 2 * Complex.I) →
  (z = -2 - Complex.I ∧ z.re < 0 ∧ z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l973_97325


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l973_97304

/-- Represents a 5x5x5 cube with painted faces -/
structure PaintedCube where
  size : Nat
  painted_squares_per_face : Nat
  total_cubes : Nat
  painted_pattern_size : Nat

/-- Calculates the number of unpainted cubes in the PaintedCube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_cubes - (cube.painted_squares_per_face * 6 - (cube.painted_pattern_size - 1) * 4 * 3)

/-- Theorem stating that the number of unpainted cubes is 83 -/
theorem unpainted_cubes_count (cube : PaintedCube) 
  (h1 : cube.size = 5)
  (h2 : cube.painted_squares_per_face = 9)
  (h3 : cube.total_cubes = 125)
  (h4 : cube.painted_pattern_size = 3) : 
  unpainted_cubes cube = 83 := by
  sorry

#eval unpainted_cubes { size := 5, painted_squares_per_face := 9, total_cubes := 125, painted_pattern_size := 3 }

end NUMINAMATH_CALUDE_unpainted_cubes_count_l973_97304


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l973_97345

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) : 
  z.im = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l973_97345


namespace NUMINAMATH_CALUDE_concert_ticket_price_l973_97362

theorem concert_ticket_price (student_price : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (student_tickets : ℕ) :
  student_price = 9 →
  total_tickets = 2000 →
  total_revenue = 20960 →
  student_tickets = 520 →
  ∃ (non_student_price : ℕ),
    non_student_price * (total_tickets - student_tickets) + student_price * student_tickets = total_revenue ∧
    non_student_price = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l973_97362


namespace NUMINAMATH_CALUDE_smallest_n_for_perfect_square_product_l973_97378

/-- The set of integers from 70 to 70 + n, inclusive -/
def numberSet (n : ℕ) : Set ℤ :=
  {x | 70 ≤ x ∧ x ≤ 70 + n}

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (x : ℤ) : Prop :=
  ∃ y : ℤ, x = y * y

/-- Predicate to check if there exist two different numbers in the set whose product is a perfect square -/
def hasPerfectSquareProduct (n : ℕ) : Prop :=
  ∃ a b : ℤ, a ∈ numberSet n ∧ b ∈ numberSet n ∧ a ≠ b ∧ isPerfectSquare (a * b)

theorem smallest_n_for_perfect_square_product : 
  (∀ m : ℕ, m < 28 → ¬hasPerfectSquareProduct m) ∧ hasPerfectSquareProduct 28 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_perfect_square_product_l973_97378


namespace NUMINAMATH_CALUDE_strawberry_jelly_amount_l973_97388

theorem strawberry_jelly_amount (total_jelly : ℕ) (blueberry_jelly : ℕ) 
  (h1 : total_jelly = 6310)
  (h2 : blueberry_jelly = 4518) :
  total_jelly - blueberry_jelly = 1792 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_jelly_amount_l973_97388


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l973_97380

theorem sum_of_reciprocals_of_roots (p q : ℝ) : 
  p^2 - 17*p + 8 = 0 → q^2 - 17*q + 8 = 0 → 1/p + 1/q = 17/8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l973_97380


namespace NUMINAMATH_CALUDE_sum_of_max_min_values_l973_97317

/-- Given real numbers a and b satisfying the condition,
    prove that the sum of max and min values of a^2 + 2b^2 is 16/7 -/
theorem sum_of_max_min_values (a b : ℝ) 
  (h : (a - b/2)^2 = 1 - (7/4)*b^2) : 
  ∃ (t_max t_min : ℝ), 
    (∀ t, t = a^2 + 2*b^2 → t ≤ t_max ∧ t ≥ t_min) ∧
    t_max + t_min = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_values_l973_97317


namespace NUMINAMATH_CALUDE_g_increasing_on_neg_l973_97383

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions on f
variable (h1 : ∀ x y, x < y → f x < f y)  -- f is increasing
variable (h2 : ∀ x, f x < 0)  -- f(x) < 0 for all x

-- Define the function g
def g (x : ℝ) : ℝ := x^2 * f x

-- State the theorem
theorem g_increasing_on_neg : 
  ∀ x y, x < y ∧ y < 0 → g f x < g f y :=
sorry

end NUMINAMATH_CALUDE_g_increasing_on_neg_l973_97383


namespace NUMINAMATH_CALUDE_door_height_calculation_l973_97393

/-- Calculates the height of a door in a room given the room dimensions, door width, window dimensions, number of windows, cost of white washing per square foot, and total cost of white washing. -/
theorem door_height_calculation (room_length room_width room_height : ℝ)
                                (door_width : ℝ)
                                (window_length window_width : ℝ)
                                (num_windows : ℕ)
                                (cost_per_sqft : ℝ)
                                (total_cost : ℝ) :
  room_length = 25 ∧ room_width = 15 ∧ room_height = 12 ∧
  door_width = 3 ∧
  window_length = 4 ∧ window_width = 3 ∧
  num_windows = 3 ∧
  cost_per_sqft = 3 ∧
  total_cost = 2718 →
  ∃ (door_height : ℝ),
    door_height = 6 ∧
    total_cost = (2 * (room_length * room_height + room_width * room_height) -
                  (door_height * door_width + ↑num_windows * window_length * window_width)) * cost_per_sqft :=
by sorry

end NUMINAMATH_CALUDE_door_height_calculation_l973_97393


namespace NUMINAMATH_CALUDE_power_sum_zero_l973_97339

theorem power_sum_zero : (-2 : ℤ)^(3^2) + 2^(3^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_zero_l973_97339


namespace NUMINAMATH_CALUDE_log_product_equality_l973_97385

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.log x^2 / Real.log y^5 * 
  Real.log y^3 / Real.log x^4 * 
  Real.log x^4 / Real.log y^3 * 
  Real.log y^5 / Real.log x^3 * 
  Real.log x^3 / Real.log y^4 = 
  (1 / 6) * (Real.log x / Real.log y) :=
sorry

end NUMINAMATH_CALUDE_log_product_equality_l973_97385


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l973_97324

/-- Proof that for a rectangle with length to width ratio of 5:2 and perimeter 42 cm, 
    the area A can be expressed as (10/29)d^2, where d is the diagonal of the rectangle. -/
theorem rectangle_area_diagonal (length width : ℝ) (d : ℝ) : 
  length / width = 5 / 2 →
  2 * (length + width) = 42 →
  d^2 = length^2 + width^2 →
  length * width = (10/29) * d^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l973_97324


namespace NUMINAMATH_CALUDE_tips_fraction_is_three_sevenths_l973_97399

/-- Represents the waiter's income structure -/
structure WaiterIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the fraction of income from tips -/
def fractionFromTips (income : WaiterIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: The fraction of income from tips is 3/7 when tips are 3/4 of the salary -/
theorem tips_fraction_is_three_sevenths (income : WaiterIncome) 
    (h : income.tips = 3/4 * income.salary) : 
    fractionFromTips income = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_tips_fraction_is_three_sevenths_l973_97399


namespace NUMINAMATH_CALUDE_restaurant_bill_l973_97353

/-- Given a total spent of $23 on an entree and a dessert, where the entree costs $5 more than the dessert, prove that the cost of the entree is $14. -/
theorem restaurant_bill (total : ℝ) (entree_cost : ℝ) (dessert_cost : ℝ) 
  (h1 : total = 23)
  (h2 : entree_cost = dessert_cost + 5)
  (h3 : total = entree_cost + dessert_cost) :
  entree_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_l973_97353


namespace NUMINAMATH_CALUDE_number_count_l973_97398

theorem number_count (average : ℝ) (sum_of_three : ℝ) (average_of_two : ℝ) (n : ℕ) : 
  average = 20 →
  sum_of_three = 48 →
  average_of_two = 26 →
  (average * n : ℝ) = sum_of_three + 2 * average_of_two →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_number_count_l973_97398


namespace NUMINAMATH_CALUDE_expression_evaluation_l973_97316

theorem expression_evaluation : 
  let x : ℤ := -3
  7 * x^2 - 3 * (2 * x^2 - 1) - 4 = 8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l973_97316


namespace NUMINAMATH_CALUDE_average_of_six_integers_l973_97348

theorem average_of_six_integers (a b c d e f : ℤ) :
  a = 22 ∧ b = 23 ∧ c = 23 ∧ d = 25 ∧ e = 26 ∧ f = 31 →
  (a + b + c + d + e + f) / 6 = 25 := by
sorry

end NUMINAMATH_CALUDE_average_of_six_integers_l973_97348


namespace NUMINAMATH_CALUDE_binomial_expansion_property_l973_97367

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the condition for the sum of the first three binomial coefficients
def first_three_sum_condition (n : ℕ) : Prop :=
  binomial n 0 + binomial n 1 + binomial n 2 = 79

-- Define the coefficient of the k-th term in the expansion
def coefficient (n k : ℕ) : ℚ := sorry

-- Define the property of having maximum coefficient
def has_max_coefficient (n k : ℕ) : Prop :=
  ∀ j, j ≠ k → coefficient n k ≥ coefficient n j

theorem binomial_expansion_property (n : ℕ) 
  (h : n > 0) 
  (h_sum : first_three_sum_condition n) :
  n = 12 ∧ has_max_coefficient n 10 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_property_l973_97367


namespace NUMINAMATH_CALUDE_odd_increasing_nonneg_implies_increasing_neg_l973_97302

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is increasing on a set S if f(x) ≤ f(y) for all x, y ∈ S with x ≤ y -/
def IsIncreasingOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

theorem odd_increasing_nonneg_implies_increasing_neg
  (f : ℝ → ℝ) (h_odd : IsOdd f) (h_incr_nonneg : IsIncreasingOn f (Set.Ici 0)) :
  IsIncreasingOn f (Set.Iic 0) :=
sorry

end NUMINAMATH_CALUDE_odd_increasing_nonneg_implies_increasing_neg_l973_97302


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l973_97328

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l973_97328


namespace NUMINAMATH_CALUDE_min_sum_of_product_1806_l973_97331

theorem min_sum_of_product_1806 (x y z : ℕ+) (h : x * y * z = 1806) :
  ∃ (a b c : ℕ+), a * b * c = 1806 ∧ a + b + c ≤ x + y + z ∧ a + b + c = 72 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1806_l973_97331


namespace NUMINAMATH_CALUDE_tom_bought_six_oranges_l973_97381

/-- Represents the number of oranges Tom bought -/
def num_oranges : ℕ := 6

/-- Represents the number of apples Tom bought -/
def num_apples : ℕ := 7 - num_oranges

/-- The cost of an orange in cents -/
def orange_cost : ℕ := 90

/-- The cost of an apple in cents -/
def apple_cost : ℕ := 60

/-- The total number of fruits bought -/
def total_fruits : ℕ := 7

/-- The total cost in cents -/
def total_cost : ℕ := orange_cost * num_oranges + apple_cost * num_apples

theorem tom_bought_six_oranges :
  num_oranges + num_apples = total_fruits ∧
  total_cost % 100 = 0 ∧
  num_oranges = 6 := by
  sorry

end NUMINAMATH_CALUDE_tom_bought_six_oranges_l973_97381


namespace NUMINAMATH_CALUDE_rotation_90_ccw_coordinates_l973_97312

def rotate90CCW (x y : ℝ) : ℝ × ℝ := (-y, x)

theorem rotation_90_ccw_coordinates :
  let A : ℝ × ℝ := (3, 5)
  let A' : ℝ × ℝ := rotate90CCW A.1 A.2
  A' = (5, -3) := by sorry

end NUMINAMATH_CALUDE_rotation_90_ccw_coordinates_l973_97312


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_reverse_sum_l973_97352

def reverse (n : ℕ) : ℕ :=
  let rec rev_aux (n acc : ℕ) : ℕ :=
    if n = 0 then acc
    else rev_aux (n / 10) (acc * 10 + n % 10)
  rev_aux n 0

theorem two_numbers_sum_and_reverse_sum :
  ∃ a b : ℕ,
    a + b = 2017 ∧
    reverse a + reverse b = 8947 ∧
    a = 1408 ∧
    b = 609 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_reverse_sum_l973_97352


namespace NUMINAMATH_CALUDE_unique_positive_number_l973_97313

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l973_97313


namespace NUMINAMATH_CALUDE_sin_symmetry_condition_l973_97311

/-- A function f: ℝ → ℝ is symmetric about x = a if f(a + x) = f(a - x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem sin_symmetry_condition (φ : ℝ) :
  let f := fun x => Real.sin (x + φ)
  (f 0 = f π) ↔ SymmetricAbout f (π / 2) := by sorry

end NUMINAMATH_CALUDE_sin_symmetry_condition_l973_97311


namespace NUMINAMATH_CALUDE_pineapple_sweets_count_l973_97319

/-- Proves the number of initial pineapple-flavored sweets in a candy packet --/
theorem pineapple_sweets_count (cherry : ℕ) (strawberry : ℕ) (remaining : ℕ) : 
  cherry = 30 → 
  strawberry = 40 → 
  remaining = 55 → 
  ∃ (pineapple : ℕ), 
    pineapple + cherry + strawberry = 
    2 * remaining + 5 + (cherry / 2) + (strawberry / 2) ∧ 
    pineapple = 50 := by
  sorry

#check pineapple_sweets_count

end NUMINAMATH_CALUDE_pineapple_sweets_count_l973_97319


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l973_97392

theorem complex_magnitude_problem (a : ℝ) (z : ℂ) : 
  z = (a * Complex.I) / (4 - 3 * Complex.I) → 
  Complex.abs z = 5 → 
  a = 25 ∨ a = -25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l973_97392


namespace NUMINAMATH_CALUDE_distance_between_points_l973_97373

theorem distance_between_points (speed_A speed_B speed_C : ℝ) (extra_time : ℝ) : 
  speed_A = 100 →
  speed_B = 90 →
  speed_C = 75 →
  extra_time = 3 →
  ∃ (distance : ℝ), 
    distance / (speed_A + speed_B) + extra_time = distance / speed_C ∧
    distance = 650 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l973_97373


namespace NUMINAMATH_CALUDE_triangle_count_is_twenty_l973_97359

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square with diagonals and midpoint segments -/
structure SquareWithDiagonalsAndMidpoints :=
  (vertices : Fin 4 → Point)
  (diagonals : Fin 2 → Point × Point)
  (midpoints : Fin 4 → Point)
  (cross : Point × Point)

/-- Counts the number of triangles in the figure -/
def countTriangles (square : SquareWithDiagonalsAndMidpoints) : ℕ :=
  sorry

/-- Theorem stating that the number of triangles in the figure is 20 -/
theorem triangle_count_is_twenty (square : SquareWithDiagonalsAndMidpoints) :
  countTriangles square = 20 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_is_twenty_l973_97359


namespace NUMINAMATH_CALUDE_predictor_accuracy_two_thirds_l973_97323

/-- Represents a match between two teams -/
structure Match where
  team_a_win_prob : ℝ
  team_b_win_prob : ℝ
  (prob_sum_one : team_a_win_prob + team_b_win_prob = 1)

/-- Represents a predictor who chooses winners with the same probability as the team's chance of winning -/
def predictor_correct_prob (m : Match) : ℝ :=
  m.team_a_win_prob * m.team_a_win_prob + m.team_b_win_prob * m.team_b_win_prob

/-- Theorem stating that for a match where one team has 2/3 probability of winning,
    the probability of the predictor correctly choosing the winner is 5/9 -/
theorem predictor_accuracy_two_thirds :
  ∀ m : Match, m.team_a_win_prob = 2/3 → predictor_correct_prob m = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_predictor_accuracy_two_thirds_l973_97323


namespace NUMINAMATH_CALUDE_find_some_number_l973_97386

theorem find_some_number (x : ℝ) (some_number : ℝ) 
  (eq1 : x + some_number = 4) (eq2 : x = 3) : some_number = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_some_number_l973_97386


namespace NUMINAMATH_CALUDE_entrance_exam_correct_answers_l973_97301

theorem entrance_exam_correct_answers 
  (total_questions : ℕ) 
  (correct_marks : ℤ) 
  (wrong_marks : ℤ) 
  (total_score : ℤ) : 
  total_questions = 70 → 
  correct_marks = 3 → 
  wrong_marks = -1 → 
  total_score = 38 → 
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_marks + (total_questions - correct_answers) * wrong_marks = total_score ∧ 
    correct_answers = 27 := by
  sorry

end NUMINAMATH_CALUDE_entrance_exam_correct_answers_l973_97301


namespace NUMINAMATH_CALUDE_hexagonal_prism_edge_sum_specific_l973_97300

/-- Calculates the sum of lengths of all edges of a regular hexagonal prism -/
def hexagonal_prism_edge_sum (base_side_length : ℝ) (height : ℝ) : ℝ :=
  2 * (6 * base_side_length) + 6 * height

theorem hexagonal_prism_edge_sum_specific : 
  hexagonal_prism_edge_sum 6 11 = 138 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_edge_sum_specific_l973_97300


namespace NUMINAMATH_CALUDE_farm_acreage_difference_l973_97375

theorem farm_acreage_difference (total_acres flax_acres : ℕ) 
  (h1 : total_acres = 240)
  (h2 : flax_acres = 80)
  (h3 : flax_acres < total_acres - flax_acres) : 
  total_acres - flax_acres - flax_acres = 80 := by
sorry

end NUMINAMATH_CALUDE_farm_acreage_difference_l973_97375


namespace NUMINAMATH_CALUDE_coefficient_x5_in_binomial_expansion_l973_97351

theorem coefficient_x5_in_binomial_expansion :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k) * (1 ^ (8 - k)) * (1 ^ k)) = 256 ∧
  (Finset.range 9).sum (fun k => if k = 3 then (Nat.choose 8 k) else 0) = 56 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_binomial_expansion_l973_97351


namespace NUMINAMATH_CALUDE_hidden_cannonball_label_l973_97332

structure CannonballPyramid where
  total_cannonballs : Nat
  labels : Finset Char
  label_count : Char → Nat
  visible_count : Char → Nat

def is_valid_pyramid (p : CannonballPyramid) : Prop :=
  p.total_cannonballs = 20 ∧
  p.labels = {'A', 'B', 'C', 'D', 'E'} ∧
  ∀ l ∈ p.labels, p.label_count l = 4 ∧
  ∀ l ∈ p.labels, p.visible_count l ≤ p.label_count l

theorem hidden_cannonball_label (p : CannonballPyramid) 
  (h_valid : is_valid_pyramid p)
  (h_visible : ∀ l ∈ p.labels, l ≠ 'D' → p.visible_count l = 4)
  (h_d_visible : p.visible_count 'D' = 3) :
  p.label_count 'D' - p.visible_count 'D' = 1 := by
sorry

end NUMINAMATH_CALUDE_hidden_cannonball_label_l973_97332


namespace NUMINAMATH_CALUDE_limit_of_a_l973_97374

def a (n : ℕ) : ℚ := (3 * n - 1) / (5 * n + 1)

theorem limit_of_a : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/5| < ε := by sorry

end NUMINAMATH_CALUDE_limit_of_a_l973_97374


namespace NUMINAMATH_CALUDE_four_digit_kabulek_numbers_l973_97376

def is_kabulek (n : ℕ) : Prop :=
  ∃ x y : ℕ,
    n = 100 * x + y ∧
    x < 100 ∧
    y < 100 ∧
    (x + y) ^ 2 = n

theorem four_digit_kabulek_numbers :
  ∀ n : ℕ,
    1000 ≤ n ∧ n < 10000 →
    is_kabulek n ↔ n = 2025 ∨ n = 3025 ∨ n = 9801 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_kabulek_numbers_l973_97376


namespace NUMINAMATH_CALUDE_finish_book_in_three_days_l973_97321

def pages_to_read_on_third_day (total_pages : ℕ) (pages_day1 : ℕ) (fewer_pages_day2 : ℕ) : ℕ :=
  total_pages - (pages_day1 + (pages_day1 - fewer_pages_day2))

theorem finish_book_in_three_days (total_pages : ℕ) (pages_day1 : ℕ) (fewer_pages_day2 : ℕ)
  (h1 : total_pages = 100)
  (h2 : pages_day1 = 35)
  (h3 : fewer_pages_day2 = 5) :
  pages_to_read_on_third_day total_pages pages_day1 fewer_pages_day2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_finish_book_in_three_days_l973_97321


namespace NUMINAMATH_CALUDE_log_inequality_l973_97327

theorem log_inequality : ∃ (a b : ℝ), 
  a = Real.log 0.8 / Real.log 0.7 ∧ 
  b = Real.log 0.9 / Real.log 1.1 ∧ 
  a > 0 ∧ 0 > b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l973_97327


namespace NUMINAMATH_CALUDE_max_distance_point_to_circle_l973_97379

/-- The maximum distance from a point to a circle -/
theorem max_distance_point_to_circle :
  let circle := {(x, y) : ℝ × ℝ | (x - 3)^2 + (y - 4)^2 = 25}
  let point := (2, 3)
  (⨆ p ∈ circle, Real.sqrt ((point.1 - p.1)^2 + (point.2 - p.2)^2)) = Real.sqrt 2 + 5 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_point_to_circle_l973_97379


namespace NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_90_l973_97382

/-- Calculates the average speed on the return trip given the conditions of Cole's journey. -/
theorem coles_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : ℝ :=
  let distance_to_work := speed_to_work * (time_to_work / 60)
  let time_to_return := total_time - (time_to_work / 60)
  distance_to_work / time_to_return

/-- Proves that Cole's average speed on the return trip is 90 km/h given the problem conditions. -/
theorem coles_return_speed_is_90 :
  coles_return_speed 30 2 90 = 90 := by
  sorry

end NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_90_l973_97382
