import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_minimum_value_exists_minimum_l1327_132748

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f (-x) + f (x + 5)

-- Theorem for part (I)
theorem inequality_solution (x : ℝ) : f x > 2 ↔ x > 3 ∨ x < -1 := by sorry

-- Theorem for part (II)
theorem minimum_value : ∀ x, g x ≥ 3 := by sorry

-- Theorem to show that 3 is indeed the minimum value
theorem exists_minimum : ∃ x, g x = 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_minimum_value_exists_minimum_l1327_132748


namespace NUMINAMATH_CALUDE_faye_crayons_l1327_132712

/-- The number of rows of crayons and pencils -/
def num_rows : ℕ := 16

/-- The number of crayons in each row -/
def crayons_per_row : ℕ := 6

/-- The total number of crayons -/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem faye_crayons : total_crayons = 96 := by
  sorry

end NUMINAMATH_CALUDE_faye_crayons_l1327_132712


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1327_132781

theorem sandwich_combinations (n_meat : Nat) (n_cheese : Nat) : 
  n_meat = 12 → n_cheese = 11 → (n_meat.choose 1) * (n_cheese.choose 3) = 1980 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1327_132781


namespace NUMINAMATH_CALUDE_coffee_maker_capacity_l1327_132710

/-- Represents a cylindrical coffee maker -/
structure CoffeeMaker where
  capacity : ℝ
  remaining : ℝ
  emptyPercentage : ℝ

/-- Theorem: A coffee maker with 30 cups remaining when 75% empty has a total capacity of 120 cups -/
theorem coffee_maker_capacity (cm : CoffeeMaker) 
  (h1 : cm.remaining = 30)
  (h2 : cm.emptyPercentage = 0.75)
  : cm.capacity = 120 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_capacity_l1327_132710


namespace NUMINAMATH_CALUDE_percentage_increase_l1327_132765

theorem percentage_increase (original : ℝ) (new : ℝ) (increase_percent : ℝ) 
  (h1 : original = 50)
  (h2 : new = 80)
  (h3 : increase_percent = 60) :
  (new - original) / original * 100 = increase_percent :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_l1327_132765


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l1327_132757

/-- Given two points (x₁, y₁) and (x₂, y₂) on opposite sides of the line 3x - 2y + a = 0,
    prove that the range of values for 'a' is -4 < a < 9 -/
theorem opposite_sides_line_range (x₁ y₁ x₂ y₂ : ℝ) (h : (3*x₁ - 2*y₁ + a) * (3*x₂ - 2*y₂ + a) < 0) :
  -4 < a ∧ a < 9 :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l1327_132757


namespace NUMINAMATH_CALUDE_expression_decrease_l1327_132779

theorem expression_decrease (k x y : ℝ) (hk : k ≠ 0) :
  let x' := 0.75 * x
  let y' := 0.65 * y
  k * x' * y'^2 = (507/1600) * (k * x * y^2) := by
sorry

end NUMINAMATH_CALUDE_expression_decrease_l1327_132779


namespace NUMINAMATH_CALUDE_trajectory_of_point_M_l1327_132730

/-- The trajectory of point M given the specified conditions -/
theorem trajectory_of_point_M :
  ∀ (x y : ℝ),
    x ≠ -1 →
    x ≠ 1 →
    y ≠ 0 →
    (y / (x + 1)) / (y / (x - 1)) = 2 →
    x = -3 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_M_l1327_132730


namespace NUMINAMATH_CALUDE_unique_sums_count_l1327_132778

def set_A : Finset ℕ := {2, 3, 5, 8}
def set_B : Finset ℕ := {1, 4, 6, 7}

theorem unique_sums_count : 
  Finset.card ((set_A.product set_B).image (fun p => p.1 + p.2)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l1327_132778


namespace NUMINAMATH_CALUDE_arrangement_counts_l1327_132711

/-- The number of boys in the row -/
def num_boys : ℕ := 4

/-- The number of girls in the row -/
def num_girls : ℕ := 2

/-- The total number of students in the row -/
def total_students : ℕ := num_boys + num_girls

/-- The number of arrangements where Boy A does not stand at the head or the tail of the row -/
def arrangements_A_not_ends : ℕ := 480

/-- The number of arrangements where the two girls must stand next to each other -/
def arrangements_girls_together : ℕ := 240

/-- The number of arrangements where Students A, B, and C are not next to each other -/
def arrangements_ABC_not_adjacent : ℕ := 144

/-- The number of arrangements where A does not stand at the head, and B does not stand at the tail -/
def arrangements_A_not_head_B_not_tail : ℕ := 504

theorem arrangement_counts :
  arrangements_A_not_ends = 480 ∧
  arrangements_girls_together = 240 ∧
  arrangements_ABC_not_adjacent = 144 ∧
  arrangements_A_not_head_B_not_tail = 504 := by sorry

end NUMINAMATH_CALUDE_arrangement_counts_l1327_132711


namespace NUMINAMATH_CALUDE_equation_solutions_l1327_132706

-- Define the function representing the left side of the equation
def f (x : ℝ) : ℝ := (18 * x - 1) ^ (1/3) - (10 * x + 1) ^ (1/3) - 3 * x ^ (1/3)

-- Define the set of solutions
def solutions : Set ℝ := {0, -5/8317, -60/1614}

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℝ, f x = 0 ↔ x ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1327_132706


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1327_132772

def A (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A a ∪ B = B) ↔ (a = -1/2 ∨ a = 0 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1327_132772


namespace NUMINAMATH_CALUDE_current_speed_is_correct_l1327_132783

/-- Represents the speed of a swimmer in still water -/
def swimmer_speed : ℝ := 6.5

/-- Represents the speed of the current -/
def current_speed : ℝ := 4.5

/-- Represents the distance traveled downstream -/
def downstream_distance : ℝ := 55

/-- Represents the distance traveled upstream -/
def upstream_distance : ℝ := 10

/-- Represents the time taken for both downstream and upstream journeys -/
def travel_time : ℝ := 5

/-- Theorem stating that given the conditions, the speed of the current is 4.5 km/h -/
theorem current_speed_is_correct : 
  downstream_distance / travel_time = swimmer_speed + current_speed ∧
  upstream_distance / travel_time = swimmer_speed - current_speed →
  current_speed = 4.5 := by
  sorry

#check current_speed_is_correct

end NUMINAMATH_CALUDE_current_speed_is_correct_l1327_132783


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l1327_132735

theorem solution_implies_k_value (k : ℝ) :
  (∃ x y : ℝ, k * x + y = 5 ∧ x = 2 ∧ y = 1) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l1327_132735


namespace NUMINAMATH_CALUDE_inequality_range_l1327_132797

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| ≥ a*x) ↔ a ∈ Set.Iic 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1327_132797


namespace NUMINAMATH_CALUDE_result_not_divisible_by_1998_l1327_132701

/-- The operation of multiplying by 2 and adding 1 -/
def operation (n : ℕ) : ℕ := 2 * n + 1

/-- The result of applying the operation k times to n -/
def iterate_operation (n k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => operation (iterate_operation n k)

theorem result_not_divisible_by_1998 (n k : ℕ) :
  ¬(1998 ∣ iterate_operation n k) := by
  sorry

#check result_not_divisible_by_1998

end NUMINAMATH_CALUDE_result_not_divisible_by_1998_l1327_132701


namespace NUMINAMATH_CALUDE_range_of_a_l1327_132741

theorem range_of_a (a : ℝ) : 
  Real.sqrt (a^3 + 2*a^2) = -a * Real.sqrt (a + 2) → 
  -2 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1327_132741


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1327_132775

theorem quadratic_equation_solution (a b m : ℤ) : 
  (∀ x, a * x^2 + 24 * x + b = (m * x - 3)^2) → 
  (a = 16 ∧ b = 9 ∧ m = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1327_132775


namespace NUMINAMATH_CALUDE_max_d_value_in_multiple_of_13_l1327_132756

theorem max_d_value_in_multiple_of_13 :
  let is_valid : (ℕ → ℕ → Bool) :=
    fun d e => (520000 + 10000 * d + 550 + 10 * e) % 13 = 0 ∧ 
               d < 10 ∧ e < 10
  ∃ d e, is_valid d e ∧ d = 6 ∧ ∀ d' e', is_valid d' e' → d' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_in_multiple_of_13_l1327_132756


namespace NUMINAMATH_CALUDE_sue_made_22_buttons_l1327_132728

def mari_buttons : ℕ := 8

def kendra_buttons (m : ℕ) : ℕ := 5 * m + 4

def sue_buttons (k : ℕ) : ℕ := k / 2

theorem sue_made_22_buttons : 
  sue_buttons (kendra_buttons mari_buttons) = 22 := by
  sorry

end NUMINAMATH_CALUDE_sue_made_22_buttons_l1327_132728


namespace NUMINAMATH_CALUDE_greatest_common_piece_length_l1327_132713

theorem greatest_common_piece_length : Nat.gcd 28 (Nat.gcd 42 70) = 14 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_piece_length_l1327_132713


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l1327_132737

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + a + 3 = 0}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) (h : B a ⊆ A) : -2 ≤ a ∧ a < 6 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l1327_132737


namespace NUMINAMATH_CALUDE_concentric_circles_no_common_tangents_l1327_132700

-- Define a circle in a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define concentric circles
def concentric (c1 c2 : Circle) : Prop :=
  c1.center = c2.center ∧ c1.radius ≠ c2.radius

-- Define a tangent line to a circle
def is_tangent_to (line : ℝ × ℝ → ℝ) (c : Circle) : Prop :=
  ∃ (point : ℝ × ℝ), line point = 0 ∧ 
    (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

-- Theorem: Two concentric circles have 0 common tangents
theorem concentric_circles_no_common_tangents (c1 c2 : Circle) 
  (h : concentric c1 c2) : 
  ¬∃ (line : ℝ × ℝ → ℝ), is_tangent_to line c1 ∧ is_tangent_to line c2 :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_no_common_tangents_l1327_132700


namespace NUMINAMATH_CALUDE_heptagon_triangulation_l1327_132720

-- Define a type for polygons
structure Polygon where
  sides : Nat
  isRegular : Bool

-- Define a triangulation
structure Triangulation where
  polygon : Polygon
  numTriangles : Nat
  usesDiagonals : Bool
  verticesFromPolygon : Bool

-- Define a function to count unique triangulations
def countUniqueTriangulations (p : Polygon) (t : Triangulation) : Nat :=
  sorry

-- Theorem statement
theorem heptagon_triangulation :
  let heptagon : Polygon := { sides := 7, isRegular := true }
  let triangulation : Triangulation := {
    polygon := heptagon,
    numTriangles := 5,
    usesDiagonals := true,
    verticesFromPolygon := true
  }
  countUniqueTriangulations heptagon triangulation = 4 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_triangulation_l1327_132720


namespace NUMINAMATH_CALUDE_fencing_requirement_l1327_132786

theorem fencing_requirement (area : ℝ) (uncovered_side : ℝ) : 
  area = 680 → uncovered_side = 20 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    uncovered_side + 2 * width = 88 := by
  sorry

end NUMINAMATH_CALUDE_fencing_requirement_l1327_132786


namespace NUMINAMATH_CALUDE_alex_and_father_ages_l1327_132704

/-- Proves that given the conditions about Alex and his father's ages, 
    Alex is 9 years old and his father is 23 years old. -/
theorem alex_and_father_ages :
  ∀ (alex_age father_age : ℕ),
    father_age = 2 * alex_age + 5 →
    alex_age - 6 = alex_age / 3 →
    alex_age = 9 ∧ father_age = 23 := by
  sorry

end NUMINAMATH_CALUDE_alex_and_father_ages_l1327_132704


namespace NUMINAMATH_CALUDE_lcm_14_21_l1327_132760

theorem lcm_14_21 : Nat.lcm 14 21 = 42 := by
  sorry

end NUMINAMATH_CALUDE_lcm_14_21_l1327_132760


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_l1327_132747

theorem smallest_n_for_candy (n : ℕ) : 
  (∃ m : ℕ, m > 0 ∧ 25 * m % 10 = 0 ∧ 25 * m % 16 = 0 ∧ 25 * m % 18 = 0) →
  (25 * n % 10 = 0 ∧ 25 * n % 16 = 0 ∧ 25 * n % 18 = 0) →
  n ≥ 29 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_l1327_132747


namespace NUMINAMATH_CALUDE_square_side_length_l1327_132768

theorem square_side_length (diagonal : ℝ) (h : diagonal = Real.sqrt 2) : 
  ∃ (side : ℝ), side * side + side * side = diagonal * diagonal ∧ side = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1327_132768


namespace NUMINAMATH_CALUDE_sector_central_angle_l1327_132761

/-- Given a circular sector with radius 8 cm and area 4 cm², 
    prove that its central angle measures 1/8 radians. -/
theorem sector_central_angle (radius : ℝ) (area : ℝ) (angle : ℝ) :
  radius = 8 →
  area = 4 →
  area = 1/2 * angle * radius^2 →
  angle = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1327_132761


namespace NUMINAMATH_CALUDE_abs_eq_sum_l1327_132717

theorem abs_eq_sum (x : ℝ) : (|x - 5| = 23) → (∃ y : ℝ, |y - 5| = 23 ∧ x + y = 10) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sum_l1327_132717


namespace NUMINAMATH_CALUDE_f_shift_three_l1327_132792

/-- Given a function f(x) = x(x-1)/2, prove that f(x+3) = f(x) + 3x + 3 for all real x. -/
theorem f_shift_three (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x * (x - 1) / 2
  f (x + 3) = f x + 3 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_f_shift_three_l1327_132792


namespace NUMINAMATH_CALUDE_monotonic_decrease_interval_l1327_132759

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * |2 * x - 4|

-- State the theorem
theorem monotonic_decrease_interval
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : f a 1 = 9) :
  ∃ (I : Set ℝ), StrictMonoOn (f a) (Set.Iic 2) ∧ I = Set.Iic 2 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decrease_interval_l1327_132759


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l1327_132734

theorem product_of_solutions_abs_y_eq_3_abs_y_minus_2 :
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3*(|y₁| - 2)) ∧ (|y₂| = 3*(|y₂| - 2)) ∧ (y₁ ≠ y₂) ∧ (y₁ * y₂ = -9) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l1327_132734


namespace NUMINAMATH_CALUDE_eighteenth_replacement_is_march_l1327_132788

def months : List String := ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def replacement_interval : Nat := 5

def first_replacement_month : String := "February"

def nth_replacement (n : Nat) : Nat :=
  replacement_interval * (n - 1)

theorem eighteenth_replacement_is_march :
  let months_after_february := nth_replacement 18
  let month_index := months_after_february % 12
  let replacement_month := months[(months.indexOf first_replacement_month + month_index) % 12]
  replacement_month = "March" := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_replacement_is_march_l1327_132788


namespace NUMINAMATH_CALUDE_males_not_listening_l1327_132798

/-- Represents the survey results -/
structure SurveyResults where
  total_listeners : ℕ
  total_non_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ

/-- Theorem stating that the number of males who don't listen is 85 -/
theorem males_not_listening (survey : SurveyResults)
  (h1 : survey.total_listeners = 160)
  (h2 : survey.total_non_listeners = 180)
  (h3 : survey.female_listeners = 75)
  (h4 : survey.male_non_listeners = 85) :
  survey.male_non_listeners = 85 := by
  sorry

#check males_not_listening

end NUMINAMATH_CALUDE_males_not_listening_l1327_132798


namespace NUMINAMATH_CALUDE_factor_expression_l1327_132766

theorem factor_expression (y : ℝ) : 84 * y^13 + 210 * y^26 = 42 * y^13 * (2 + 5 * y^13) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1327_132766


namespace NUMINAMATH_CALUDE_fraction_simplification_l1327_132799

theorem fraction_simplification (d : ℝ) : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1327_132799


namespace NUMINAMATH_CALUDE_complex_number_equality_l1327_132794

theorem complex_number_equality : Complex.abs ((1 - Complex.I) / (1 + Complex.I)) + 2 * Complex.I = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1327_132794


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l1327_132758

theorem parabola_x_intercepts :
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 4 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l1327_132758


namespace NUMINAMATH_CALUDE_equation_proof_l1327_132714

theorem equation_proof : (12 : ℕ)^2 * 6^4 / 432 = 432 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1327_132714


namespace NUMINAMATH_CALUDE_box_values_equality_l1327_132774

theorem box_values_equality : 40506000 = 4 * 10000000 + 5 * 100000 + 6 * 1000 := by
  sorry

end NUMINAMATH_CALUDE_box_values_equality_l1327_132774


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1327_132727

theorem imaginary_part_of_z (z : ℂ) (h : (z - Complex.I) * (1 + 2 * Complex.I) = Complex.I ^ 3) :
  z.im = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1327_132727


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1327_132726

theorem complex_equation_sum (a b : ℝ) : 
  (a - Complex.I) * Complex.I = -b + 2 * Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1327_132726


namespace NUMINAMATH_CALUDE_problem_solution_l1327_132773

theorem problem_solution (a b c : ℝ) 
  (h1 : |a - 4| + |b + 5| = 0) 
  (h2 : a + c = 0) : 
  3*a + 2*b - 4*c = 18 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1327_132773


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1327_132796

def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

theorem intersection_of_A_and_B : A ∩ B = {8, 10} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1327_132796


namespace NUMINAMATH_CALUDE_log_equation_solution_l1327_132752

theorem log_equation_solution :
  ∀ x : ℝ, (x + 5 > 0) ∧ (2*x - 1 > 0) ∧ (3*x^2 - 11*x + 5 > 0) →
  (Real.log (x + 5) + Real.log (2*x - 1) = Real.log (3*x^2 - 11*x + 5)) ↔
  (x = 10 + 3 * Real.sqrt 10 ∨ x = 10 - 3 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1327_132752


namespace NUMINAMATH_CALUDE_not_all_angles_greater_than_60_l1327_132708

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : a + b + c = 180

-- Theorem statement
theorem not_all_angles_greater_than_60 (t : Triangle) : 
  ¬(t.a > 60 ∧ t.b > 60 ∧ t.c > 60) :=
by sorry

end NUMINAMATH_CALUDE_not_all_angles_greater_than_60_l1327_132708


namespace NUMINAMATH_CALUDE_initial_guinea_fowls_eq_80_l1327_132731

/-- Represents the initial state and daily losses of birds in a poultry farm --/
structure PoultryFarm :=
  (initial_chickens : ℕ)
  (initial_turkeys : ℕ)
  (daily_chicken_loss : ℕ)
  (daily_turkey_loss : ℕ)
  (daily_guinea_fowl_loss : ℕ)
  (disease_duration : ℕ)
  (total_birds_after_disease : ℕ)

/-- Calculates the initial number of guinea fowls in the farm --/
def initial_guinea_fowls (farm : PoultryFarm) : ℕ :=
  let remaining_chickens := farm.initial_chickens - farm.daily_chicken_loss * farm.disease_duration
  let remaining_turkeys := farm.initial_turkeys - farm.daily_turkey_loss * farm.disease_duration
  let remaining_guinea_fowls := farm.total_birds_after_disease - remaining_chickens - remaining_turkeys
  remaining_guinea_fowls + farm.daily_guinea_fowl_loss * farm.disease_duration

/-- Theorem stating that the initial number of guinea fowls is 80 --/
theorem initial_guinea_fowls_eq_80 (farm : PoultryFarm) 
  (h1 : farm.initial_chickens = 300)
  (h2 : farm.initial_turkeys = 200)
  (h3 : farm.daily_chicken_loss = 20)
  (h4 : farm.daily_turkey_loss = 8)
  (h5 : farm.daily_guinea_fowl_loss = 5)
  (h6 : farm.disease_duration = 7)
  (h7 : farm.total_birds_after_disease = 349) :
  initial_guinea_fowls farm = 80 := by
  sorry

#eval initial_guinea_fowls {
  initial_chickens := 300,
  initial_turkeys := 200,
  daily_chicken_loss := 20,
  daily_turkey_loss := 8,
  daily_guinea_fowl_loss := 5,
  disease_duration := 7,
  total_birds_after_disease := 349
}

end NUMINAMATH_CALUDE_initial_guinea_fowls_eq_80_l1327_132731


namespace NUMINAMATH_CALUDE_angle_value_proof_l1327_132754

theorem angle_value_proof (ABC DBC ABD : ℝ) (y : ℝ) : 
  ABC = 90 →
  ABD = 3 * y →
  DBC = 2 * y →
  ABD + DBC = 90 →
  y = 18 := by sorry

end NUMINAMATH_CALUDE_angle_value_proof_l1327_132754


namespace NUMINAMATH_CALUDE_expression_value_l1327_132770

theorem expression_value : (fun x : ℝ => x^2 + 3*x - 4) 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1327_132770


namespace NUMINAMATH_CALUDE_felicity_gas_usage_l1327_132782

/-- Proves that Felicity used 23 gallons of gas given the problem conditions -/
theorem felicity_gas_usage (adhira : ℝ) : 
  (adhira + (4 * adhira - 5) = 30) → (4 * adhira - 5 = 23) :=
by
  sorry

end NUMINAMATH_CALUDE_felicity_gas_usage_l1327_132782


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1327_132744

theorem fraction_equation_solution : 
  ∃ x : ℝ, (1 / (x - 1) = 2 / (1 - x) + 1) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1327_132744


namespace NUMINAMATH_CALUDE_x_needs_seven_days_l1327_132707

/-- The number of days x needs to finish the remaining work -/
def remaining_days_for_x (x_days y_days y_worked_days : ℕ) : ℚ :=
  (y_days - y_worked_days) * x_days / y_days

/-- Theorem stating that x needs 7 days to finish the remaining work -/
theorem x_needs_seven_days (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 21)
  (hy : y_days = 15)
  (hw : y_worked_days = 10) :
  remaining_days_for_x x_days y_days y_worked_days = 7 := by
  sorry

#eval remaining_days_for_x 21 15 10

end NUMINAMATH_CALUDE_x_needs_seven_days_l1327_132707


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2017_l1327_132738

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * a 1 + (n * (n - 1) : ℤ) * (a 2 - a 1) / 2

theorem arithmetic_sequence_sum_2017 
  (a : ℕ → ℤ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_first_term : a 1 = -2015)
  (h_sum_condition : sum_arithmetic_sequence a 6 - 2 * sum_arithmetic_sequence a 3 = 18) :
  sum_arithmetic_sequence a 2017 = 2017 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2017_l1327_132738


namespace NUMINAMATH_CALUDE_james_money_l1327_132703

/-- Calculates the total money James has after finding additional bills -/
def total_money (bills_found : ℕ) (bill_value : ℕ) (existing_money : ℕ) : ℕ :=
  bills_found * bill_value + existing_money

/-- Proves that James has $135 after finding 3 $20 bills when he already had $75 -/
theorem james_money :
  total_money 3 20 75 = 135 := by
  sorry

end NUMINAMATH_CALUDE_james_money_l1327_132703


namespace NUMINAMATH_CALUDE_smallest_third_number_lcm_l1327_132771

/-- The lowest common multiple of a list of natural numbers -/
def lcm_list (l : List Nat) : Nat :=
  l.foldl Nat.lcm 1

/-- The theorem states that 10 is the smallest positive integer x
    such that the LCM of 24, 30, and x is 120 -/
theorem smallest_third_number_lcm :
  (∀ x : Nat, x > 0 → x < 10 → lcm_list [24, 30, x] ≠ 120) ∧
  lcm_list [24, 30, 10] = 120 := by
  sorry

end NUMINAMATH_CALUDE_smallest_third_number_lcm_l1327_132771


namespace NUMINAMATH_CALUDE_delta_max_ratio_l1327_132743

def charlie_day1_score : ℕ := 200
def charlie_day1_attempted : ℕ := 400
def charlie_day2_score : ℕ := 160
def charlie_day2_attempted : ℕ := 200
def total_points_attempted : ℕ := 600

def charlie_day1_ratio : ℚ := charlie_day1_score / charlie_day1_attempted
def charlie_day2_ratio : ℚ := charlie_day2_score / charlie_day2_attempted
def charlie_total_ratio : ℚ := (charlie_day1_score + charlie_day2_score) / total_points_attempted

theorem delta_max_ratio (delta_day1_score delta_day1_attempted delta_day2_score delta_day2_attempted : ℕ) :
  delta_day1_attempted + delta_day2_attempted = total_points_attempted →
  delta_day1_attempted ≠ charlie_day1_attempted →
  delta_day1_score > 0 →
  delta_day2_score > 0 →
  (delta_day1_score : ℚ) / delta_day1_attempted < charlie_day1_ratio →
  (delta_day2_score : ℚ) / delta_day2_attempted < charlie_day2_ratio →
  (delta_day1_score + delta_day2_score : ℚ) / total_points_attempted ≤ 479 / 600 :=
by sorry

end NUMINAMATH_CALUDE_delta_max_ratio_l1327_132743


namespace NUMINAMATH_CALUDE_list_mean_mode_relation_l1327_132719

theorem list_mean_mode_relation (x : ℕ) (h1 : x ≤ 200) :
  let L := [30, 60, 70, 150, x, x]
  (L.sum / L.length : ℚ) = 2 * x →
  x = 31 := by
sorry

end NUMINAMATH_CALUDE_list_mean_mode_relation_l1327_132719


namespace NUMINAMATH_CALUDE_parabola_directrix_l1327_132722

/-- The directrix of the parabola y² = 4x is the line x = -1 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y^2 = 4*x → (∃ (a : ℝ), a = -1 ∧ x = a) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1327_132722


namespace NUMINAMATH_CALUDE_marco_run_time_l1327_132750

-- Define the track and run parameters
def total_laps : ℕ := 6
def track_length : ℝ := 450
def first_segment : ℝ := 150
def second_segment : ℝ := 300
def speed_first : ℝ := 5
def speed_second : ℝ := 4

-- Define the theorem
theorem marco_run_time :
  let time_first := first_segment / speed_first
  let time_second := second_segment / speed_second
  let time_per_lap := time_first + time_second
  let total_time := total_laps * time_per_lap
  total_time = 630 := by sorry

end NUMINAMATH_CALUDE_marco_run_time_l1327_132750


namespace NUMINAMATH_CALUDE_dot_product_max_value_l1327_132793

theorem dot_product_max_value (x y z : ℝ) :
  let a : Fin 3 → ℝ := ![1, 1, -2]
  let b : Fin 3 → ℝ := ![x, y, z]
  x^2 + y^2 + z^2 = 16 →
  (∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 16 → 
    (a 0) * x' + (a 1) * y' + (a 2) * z' ≤ (a 0) * x + (a 1) * y + (a 2) * z) →
  (a 0) * x + (a 1) * y + (a 2) * z = 4 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_max_value_l1327_132793


namespace NUMINAMATH_CALUDE_satellite_upgraded_fraction_l1327_132780

/-- Represents a satellite with modular units and sensors -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (upgraded_total : ℕ)

/-- The fraction of upgraded sensors on the satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.upgraded_total / (s.units * s.non_upgraded_per_unit + s.upgraded_total)

theorem satellite_upgraded_fraction
  (s : Satellite)
  (h1 : s.units = 24)
  (h2 : s.non_upgraded_per_unit = s.upgraded_total / 6) :
  upgraded_fraction s = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_satellite_upgraded_fraction_l1327_132780


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1327_132729

theorem simplify_and_evaluate (a : ℤ) (h : a = -1) : 
  (a + 3)^2 + (3 + a) * (3 - a) = 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1327_132729


namespace NUMINAMATH_CALUDE_gcd_with_30_is_6_l1327_132724

theorem gcd_with_30_is_6 : ∃ n : ℕ, 70 < n ∧ n < 80 ∧ Nat.gcd n 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_with_30_is_6_l1327_132724


namespace NUMINAMATH_CALUDE_decimal_93_to_binary_binary_to_decimal_93_l1327_132789

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_93_to_binary :
  to_binary 93 = [true, false, true, true, true, false, true] :=
sorry

theorem binary_to_decimal_93 :
  from_binary [true, false, true, true, true, false, true] = 93 :=
sorry

end NUMINAMATH_CALUDE_decimal_93_to_binary_binary_to_decimal_93_l1327_132789


namespace NUMINAMATH_CALUDE_ned_trays_theorem_l1327_132776

/-- The number of trays Ned can carry at a time -/
def trays_per_trip : ℕ := 8

/-- The number of trips Ned made -/
def num_trips : ℕ := 4

/-- The number of trays Ned picked up from the second table -/
def trays_from_second_table : ℕ := 5

/-- The number of trays Ned picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * num_trips - trays_from_second_table

theorem ned_trays_theorem : trays_from_first_table = 27 := by
  sorry

end NUMINAMATH_CALUDE_ned_trays_theorem_l1327_132776


namespace NUMINAMATH_CALUDE_haley_video_files_l1327_132753

/-- Given the initial number of music files, the number of deleted files,
    and the remaining number of files, calculate the initial number of video files. -/
def initialVideoFiles (initialMusicFiles deletedFiles remainingFiles : ℕ) : ℕ :=
  remainingFiles + deletedFiles - initialMusicFiles

theorem haley_video_files :
  initialVideoFiles 27 11 58 = 42 := by
  sorry

end NUMINAMATH_CALUDE_haley_video_files_l1327_132753


namespace NUMINAMATH_CALUDE_no_triangle_with_special_side_ratios_l1327_132777

theorem no_triangle_with_special_side_ratios :
  ¬ ∃ (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (a + b > c ∧ b + c > a ∧ a + c > b) ∧
    ((a = b / 2 ∧ a = c / 3) ∨ 
     (b = a / 2 ∧ b = c / 3) ∨ 
     (c = a / 2 ∧ c = b / 3)) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_with_special_side_ratios_l1327_132777


namespace NUMINAMATH_CALUDE_oil_depth_theorem_l1327_132769

/-- Represents a horizontal cylindrical oil tank -/
structure OilTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the depth of oil in the tank given the surface area -/
def oilDepth (tank : OilTank) (surfaceArea : ℝ) : Set ℝ :=
  { h : ℝ | ∃ (c : ℝ), 
    c = surfaceArea / tank.length ∧
    c = 2 * Real.sqrt (tank.diameter * h - h^2) ∧
    0 < h ∧ h < tank.diameter }

/-- The main theorem about oil depth in a cylindrical tank -/
theorem oil_depth_theorem (tank : OilTank) (surfaceArea : ℝ) :
  tank.length = 12 →
  tank.diameter = 8 →
  surfaceArea = 60 →
  oilDepth tank surfaceArea = {4 - Real.sqrt 39 / 2, 4 + Real.sqrt 39 / 2} := by
  sorry

end NUMINAMATH_CALUDE_oil_depth_theorem_l1327_132769


namespace NUMINAMATH_CALUDE_circle_radius_through_ROV_l1327_132739

-- Define the pentagon LOVER
structure Pentagon :=
  (L O V E R : ℝ × ℝ)

-- Define properties of the pentagon
def is_convex (p : Pentagon) : Prop := sorry

def is_rectangle (A B C D : ℝ × ℝ) : Prop := sorry

def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem circle_radius_through_ROV (LOVER : Pentagon) :
  is_convex LOVER →
  is_rectangle LOVER.L LOVER.O LOVER.V LOVER.E →
  distance LOVER.O LOVER.V = 20 →
  distance LOVER.L LOVER.O = 23 →
  distance LOVER.V LOVER.E = 23 →
  distance LOVER.R LOVER.E = 23 →
  distance LOVER.R LOVER.L = 23 →
  ∃ (center : ℝ × ℝ), 
    distance center LOVER.R = 23 ∧
    distance center LOVER.O = 23 ∧
    distance center LOVER.V = 23 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_radius_through_ROV_l1327_132739


namespace NUMINAMATH_CALUDE_circle_area_special_condition_l1327_132762

/-- For a circle where three times the reciprocal of its circumference equals half its diameter, 
    the area of the circle is 3/2. -/
theorem circle_area_special_condition (r : ℝ) (h : 3 * (1 / (2 * π * r)) = 1/2 * (2 * r)) : 
  π * r^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_special_condition_l1327_132762


namespace NUMINAMATH_CALUDE_munchausen_theorem_l1327_132745

/-- Represents a polynomial of degree n with n natural roots -/
structure PolynomialWithNaturalRoots (n : ℕ) where
  coeff_a : ℕ
  coeff_b : ℕ
  has_n_natural_roots : Bool

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_intersections : ℕ

/-- 
Given a polynomial of degree n with n natural roots, 
there exists a configuration of lines in the plane 
where the number of lines equals the coefficient of x^(n-1) 
and the number of intersections equals the coefficient of x^(n-2)
-/
theorem munchausen_theorem {n : ℕ} (p : PolynomialWithNaturalRoots n) 
  (h : p.has_n_natural_roots = true) :
  ∃ (lc : LineConfiguration), lc.num_lines = p.coeff_a ∧ lc.num_intersections = p.coeff_b :=
sorry

end NUMINAMATH_CALUDE_munchausen_theorem_l1327_132745


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l1327_132785

theorem quadratic_roots_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 4*h*x = 8 ∧ y^2 + 4*h*y = 8 ∧ x^2 + y^2 = 20) →
  |h| = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l1327_132785


namespace NUMINAMATH_CALUDE_work_completion_time_l1327_132795

/-- The time taken for Ganesh, Ram, and Sohan to complete a work together, given their individual work rates. -/
theorem work_completion_time 
  (ganesh_ram_rate : ℚ) -- Combined work rate of Ganesh and Ram
  (sohan_rate : ℚ)       -- Work rate of Sohan
  (h1 : ganesh_ram_rate = 1 / 24) -- Ganesh and Ram can complete the work in 24 days
  (h2 : sohan_rate = 1 / 48)      -- Sohan can complete the work in 48 days
  : (1 : ℚ) / (ganesh_ram_rate + sohan_rate) = 16 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1327_132795


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l1327_132751

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 220 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes,
    with each box containing at least one ball. -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 220 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l1327_132751


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l1327_132746

/-- A circle with center on the y-axis, radius 1, and tangent to y = 2 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_y_axis : center.1 = 0
  radius_is_one : radius = 1
  tangent_to_y_2 : ∃ (x : ℝ), (center.1 - x)^2 + (center.2 - 2)^2 = radius^2

/-- The equation of a TangentCircle is x^2 + (y-3)^2 = 1 or x^2 + (y-1)^2 = 1 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∃ (y₀ : ℝ), y₀ = 1 ∨ y₀ = 3 ∧ ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2} ↔ x^2 + (y - y₀)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l1327_132746


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1327_132718

theorem polynomial_identity_sum_of_squares : 
  ∀ (p q r s t u : ℤ), 
  (∀ x : ℝ, 1728 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 23456 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1327_132718


namespace NUMINAMATH_CALUDE_six_hours_prep_score_l1327_132732

/-- Represents the relationship between study time and test score -/
structure TestPreparation where
  actualHours : ℝ
  score : ℝ

/-- Calculates effective hours from actual hours -/
def effectiveHours (ah : ℝ) : ℝ := 0.8 * ah

/-- Theorem: Given the conditions, 6 actual hours of preparation results in a score of 96 points -/
theorem six_hours_prep_score :
  ∀ (test1 test2 : TestPreparation),
  test1.actualHours = 5 ∧
  test1.score = 80 ∧
  test2.actualHours = 6 →
  test2.score = 96 := by sorry

end NUMINAMATH_CALUDE_six_hours_prep_score_l1327_132732


namespace NUMINAMATH_CALUDE_constant_product_of_reciprocal_inputs_l1327_132736

theorem constant_product_of_reciprocal_inputs (a b : ℝ) (h : a * b ≠ 2) :
  let f : ℝ → ℝ := λ x => (b * x + 1) / (2 * x + a)
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x * f (1 / x) = k → k = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_constant_product_of_reciprocal_inputs_l1327_132736


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l1327_132733

theorem distance_between_complex_points :
  let z₁ : ℂ := 3 - 4*I
  let z₂ : ℂ := -2 - 3*I
  Complex.abs (z₁ - z₂) = Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l1327_132733


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l1327_132725

theorem pure_imaginary_complex_fraction (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (∃ b : ℝ, (a + Complex.I) / (1 + 2 * Complex.I) = b * Complex.I) →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l1327_132725


namespace NUMINAMATH_CALUDE_sum_of_five_cubes_l1327_132749

theorem sum_of_five_cubes (n : ℤ) : ∃ a b c d e : ℤ, n = a^3 + b^3 + c^3 + d^3 + e^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_cubes_l1327_132749


namespace NUMINAMATH_CALUDE_football_team_progress_l1327_132705

/-- 
Given a football team's yard changes, calculate their net progress.
-/
theorem football_team_progress 
  (loss : ℤ) 
  (gain : ℤ) 
  (h1 : loss = -5)
  (h2 : gain = 9) : 
  loss + gain = 4 := by
sorry

end NUMINAMATH_CALUDE_football_team_progress_l1327_132705


namespace NUMINAMATH_CALUDE_fourth_number_is_eight_l1327_132723

/-- Given four numbers with an arithmetic mean of 20, where three of the numbers are 12, 24, and 36,
    and the fourth number is the square of another number, prove that the fourth number is 8. -/
theorem fourth_number_is_eight (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = 24 →
  c = 36 →
  ∃ x, d = x^2 →
  d = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_is_eight_l1327_132723


namespace NUMINAMATH_CALUDE_ball_speed_equality_time_ball_speed_equality_time_specific_l1327_132715

/-- The time when a ball's average speed equals its instantaneous speed after being dropped from a height and experiencing a perfectly elastic collision. -/
theorem ball_speed_equality_time
  (h : ℝ)  -- Initial height
  (g : ℝ)  -- Acceleration due to gravity
  (h_pos : h > 0)
  (g_pos : g > 0)
  : ∃ (t : ℝ), t > 0 ∧ t = Real.sqrt (2 * h / g + 8 * h / g) :=
by
  sorry

/-- The specific case where h = 45 m and g = 10 m/s² -/
theorem ball_speed_equality_time_specific :
  ∃ (t : ℝ), t > 0 ∧ t = Real.sqrt 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_speed_equality_time_ball_speed_equality_time_specific_l1327_132715


namespace NUMINAMATH_CALUDE_range_of_a_when_B_subset_A_l1327_132740

/-- The set A -/
def A : Set ℝ := {x | x^2 + 4*x = 0}

/-- The set B parameterized by a -/
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

/-- The range of a -/
def range_a : Set ℝ := {a | a = 1 ∨ a ≤ -1}

/-- Theorem stating the range of a when B is a subset of A -/
theorem range_of_a_when_B_subset_A :
  ∀ a : ℝ, B a ⊆ A → a ∈ range_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_B_subset_A_l1327_132740


namespace NUMINAMATH_CALUDE_car_distance_theorem_l1327_132764

/-- Represents the car's driving characteristics and total driving time -/
structure CarDriving where
  speed : ℕ              -- Speed in miles per hour
  drive_time : ℕ         -- Continuous driving time in hours
  cool_time : ℕ          -- Cooling time in hours
  total_time : ℕ         -- Total available time in hours

/-- Calculates the total distance a car can travel given its driving characteristics -/
def total_distance (car : CarDriving) : ℕ :=
  sorry

/-- Theorem stating that a car with given characteristics can travel 88 miles in 13 hours -/
theorem car_distance_theorem :
  let car := CarDriving.mk 8 5 1 13
  total_distance car = 88 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l1327_132764


namespace NUMINAMATH_CALUDE_product_b_sample_size_l1327_132721

/-- Represents the number of items drawn from a specific group in stratified sampling -/
def stratifiedSampleSize (totalItems : ℕ) (sampleSize : ℕ) (groupRatio : ℕ) (totalRatio : ℕ) : ℕ :=
  (sampleSize * groupRatio) / totalRatio

/-- Proves that the number of items drawn from product B in the given stratified sampling scenario is 10 -/
theorem product_b_sample_size :
  let totalItems : ℕ := 1200
  let sampleSize : ℕ := 60
  let ratioA : ℕ := 1
  let ratioB : ℕ := 2
  let ratioC : ℕ := 4
  let ratioD : ℕ := 5
  let totalRatio : ℕ := ratioA + ratioB + ratioC + ratioD
  stratifiedSampleSize totalItems sampleSize ratioB totalRatio = 10 := by
  sorry


end NUMINAMATH_CALUDE_product_b_sample_size_l1327_132721


namespace NUMINAMATH_CALUDE_det_A_eq_121_l1327_132784

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 4, 5, -3; 6, 2, 7]

theorem det_A_eq_121 : A.det = 121 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_121_l1327_132784


namespace NUMINAMATH_CALUDE_marble_remainder_l1327_132790

theorem marble_remainder (r p : ℕ) : 
  r % 8 = 5 → p % 8 = 6 → (r + p) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_remainder_l1327_132790


namespace NUMINAMATH_CALUDE_rebus_solution_exists_l1327_132702

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def rebus_equation (h1 h2 h3 ch1 ch2 : ℕ) : Prop :=
  10 * h1 + h2 + 4000 + 400 * h3 + ch1 * 10 + ch2 = 4000 + 100 * h1 + 10 * h2 + h3

theorem rebus_solution_exists :
  ∃ (h1 h2 h3 ch1 ch2 : ℕ),
    is_odd h1 ∧ is_odd h2 ∧ is_odd h3 ∧
    is_even ch1 ∧ is_even ch2 ∧
    rebus_equation h1 h2 h3 ch1 ch2 ∧
    h1 = 5 ∧ h2 = 5 ∧ h3 = 5 :=
by sorry

end NUMINAMATH_CALUDE_rebus_solution_exists_l1327_132702


namespace NUMINAMATH_CALUDE_circle_equation_correct_circle_properties_l1327_132791

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def lies_on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The specific circle we're considering -/
def our_circle : Circle :=
  { center := (0, 3)
    radius := 1 }

theorem circle_equation_correct :
  ∀ x y : ℝ, x^2 + (y - 3)^2 = 1 ↔ lies_on_circle our_circle (x, y) :=
sorry

theorem circle_properties :
  our_circle.center.1 = 0 ∧
  our_circle.radius = 1 ∧
  lies_on_circle our_circle (1, 3) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_correct_circle_properties_l1327_132791


namespace NUMINAMATH_CALUDE_circle_and_triangle_properties_l1327_132742

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  (k - 1) * x - 2 * y + 5 - 3 * k = 0

-- Define the fixed point P
def point_P : ℝ × ℝ := (3, 1)

-- Define point A
def point_A : ℝ × ℝ := (4, 0)

-- Define the line on which the center of circle C lies
def center_line (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

-- Define circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 14*x - 8*y + 40 = 0

-- Define point Q
def point_Q : ℝ × ℝ := (11, 7)

theorem circle_and_triangle_properties :
  ∀ k : ℝ,
  (∀ x y : ℝ, line_l k x y → (x = point_P.1 ∧ y = point_P.2)) →
  circle_C point_A.1 point_A.2 →
  circle_C point_P.1 point_P.2 →
  (∃ x y : ℝ, center_line x y ∧ (x - point_P.1)^2 + (y - point_P.2)^2 = (x - point_A.1)^2 + (y - point_A.2)^2) →
  (point_Q.1 - point_P.1)^2 + (point_Q.2 - point_P.2)^2 = 4 * ((point_P.1 - 7)^2 + (point_P.2 - 4)^2) →
  ∃ m : ℝ, (m = 5 ∨ m = 65/3) ∧
    ((point_P.1 - 0)^2 + (point_P.2 - m)^2 + (point_Q.1 - 0)^2 + (point_Q.2 - m)^2 =
     (point_Q.1 - point_P.1)^2 + (point_Q.2 - point_P.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_triangle_properties_l1327_132742


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l1327_132716

/-- Given a circle of radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle is three times the area of the circle,
    prove that the length of the longer side of the rectangle is 4.5π cm. -/
theorem rectangle_longer_side_length (r : ℝ) (circle_area rectangle_area : ℝ) 
  (h1 : r = 6)
  (h2 : circle_area = Real.pi * r^2)
  (h3 : rectangle_area = 3 * circle_area)
  (h4 : ∃ (shorter_side longer_side : ℝ), 
        shorter_side = 2 * (2 * r) ∧ 
        rectangle_area = shorter_side * longer_side) :
  ∃ (longer_side : ℝ), longer_side = 4.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l1327_132716


namespace NUMINAMATH_CALUDE_slope_of_line_from_equation_l1327_132787

-- Define the equation
def satisfies_equation (x y : ℝ) : Prop := 3 / x + 4 / y = 0

-- Theorem statement
theorem slope_of_line_from_equation :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ →
    satisfies_equation x₁ y₁ →
    satisfies_equation x₂ y₂ →
    (y₂ - y₁) / (x₂ - x₁) = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_from_equation_l1327_132787


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1327_132767

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1327_132767


namespace NUMINAMATH_CALUDE_sweater_wool_correct_l1327_132763

/-- The number of balls of wool used for a sweater -/
def sweater_wool : ℕ := 4

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for a scarf -/
def scarf_wool : ℕ := 3

/-- The total number of balls of wool used -/
def total_wool : ℕ := 82

theorem sweater_wool_correct : 
  aaron_scarves * scarf_wool + (aaron_sweaters + enid_sweaters) * sweater_wool = total_wool := by
  sorry

end NUMINAMATH_CALUDE_sweater_wool_correct_l1327_132763


namespace NUMINAMATH_CALUDE_parabola_transformation_l1327_132755

/-- A parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shift a parabola horizontally and vertically -/
def shift (p : Parabola) (h v : ℝ) : Parabola where
  f x := p.f (x - h) + v

/-- The original parabola y = 2x^2 -/
def original_parabola : Parabola where
  f x := 2 * x^2

/-- The transformed parabola -/
def transformed_parabola : Parabola :=
  shift (shift original_parabola 3 0) 0 (-4)

theorem parabola_transformation :
  ∀ x, transformed_parabola.f x = 2 * (x + 3)^2 - 4 := by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l1327_132755


namespace NUMINAMATH_CALUDE_exponent_equality_l1327_132709

theorem exponent_equality (y x : ℕ) (h1 : 16 ^ y = 4 ^ x) (h2 : y = 7) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l1327_132709
