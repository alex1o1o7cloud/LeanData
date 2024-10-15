import Mathlib

namespace NUMINAMATH_CALUDE_greatest_divisor_3815_4521_l1528_152851

def is_greatest_divisor (d n1 n2 r1 r2 : ℕ) : Prop :=
  d > 0 ∧
  n1 % d = r1 ∧
  n2 % d = r2 ∧
  ∀ k : ℕ, k > d → (n1 % k ≠ r1 ∨ n2 % k ≠ r2)

theorem greatest_divisor_3815_4521 :
  is_greatest_divisor 64 3815 4521 31 33 := by sorry

end NUMINAMATH_CALUDE_greatest_divisor_3815_4521_l1528_152851


namespace NUMINAMATH_CALUDE_triangle_area_l1528_152862

/-- The area of a triangle with base 15 cm and height 20 cm is 150 cm². -/
theorem triangle_area : 
  let base : ℝ := 15
  let height : ℝ := 20
  let area : ℝ := (base * height) / 2
  area = 150 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1528_152862


namespace NUMINAMATH_CALUDE_first_day_of_month_l1528_152854

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

theorem first_day_of_month (d : DayOfWeek) :
  advanceDay d 29 = DayOfWeek.Monday → d = DayOfWeek.Sunday :=
by sorry


end NUMINAMATH_CALUDE_first_day_of_month_l1528_152854


namespace NUMINAMATH_CALUDE_divisible_by_33_pairs_count_l1528_152811

theorem divisible_by_33_pairs_count : 
  (Finset.filter (fun p : ℕ × ℕ => 
    1 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ 40 ∧ (p.1 * p.2) % 33 = 0) 
    (Finset.product (Finset.range 40) (Finset.range 41))).card = 64 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_33_pairs_count_l1528_152811


namespace NUMINAMATH_CALUDE_cylinder_no_triangular_cross_section_l1528_152875

-- Define the type for geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | TriangularPrism
  | Cube

-- Define a function that determines if a solid can have a triangular cross-section
def canHaveTriangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => False
  | _ => True

-- Theorem stating that only the Cylinder cannot have a triangular cross-section
theorem cylinder_no_triangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveTriangularCrossSection solid) ↔ solid = GeometricSolid.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_cylinder_no_triangular_cross_section_l1528_152875


namespace NUMINAMATH_CALUDE_smallest_m_partition_property_l1528_152896

def S (m : ℕ) : Set ℕ := {n : ℕ | 2 ≤ n ∧ n ≤ m}

def satisfies_condition (A : Set ℕ) : Prop :=
  ∃ a b : ℕ, a ∈ A ∧ b ∈ A ∧ a^b = b^a

theorem smallest_m_partition_property :
  ∀ m : ℕ, m ≥ 2 →
    (∀ A B : Set ℕ, A ∪ B = S m ∧ A ∩ B = ∅ →
      satisfies_condition A ∨ satisfies_condition B) ↔ m ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_partition_property_l1528_152896


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1528_152825

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1528_152825


namespace NUMINAMATH_CALUDE_series_sum_equals_n_l1528_152835

/-- The floor function, denoted as ⌊x⌋, returns the greatest integer less than or equal to x. -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The sum of the series for a given positive integer n -/
noncomputable def series_sum (n : ℕ+) : ℝ :=
  ∑' k : ℕ, (floor ((n : ℝ) + 2^k) / 2^(k+1))

/-- Theorem stating that the sum of the series equals n for every positive integer n -/
theorem series_sum_equals_n (n : ℕ+) : series_sum n = n :=
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_n_l1528_152835


namespace NUMINAMATH_CALUDE_sin_minus_cos_value_l1528_152878

theorem sin_minus_cos_value (α : Real) 
  (h : ∃ (r : Real), r * (Real.cos (α - π/4)) = -1 ∧ r * (Real.sin (α - π/4)) = Real.sqrt 2) : 
  Real.sin α - Real.cos α = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_minus_cos_value_l1528_152878


namespace NUMINAMATH_CALUDE_square_diagonal_length_l1528_152824

theorem square_diagonal_length (perimeter : ℝ) (diagonal : ℝ) : 
  perimeter = 40 → diagonal = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_length_l1528_152824


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l1528_152852

/-- The original quadratic function -/
def original_quadratic (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The converted quadratic function -/
def converted_quadratic (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating the equivalence of the two quadratic functions -/
theorem quadratic_equivalence :
  ∀ x : ℝ, original_quadratic x = converted_quadratic x :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l1528_152852


namespace NUMINAMATH_CALUDE_set_star_A_B_l1528_152841

-- Define the sets A and B
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define the set difference operation
def set_difference (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

-- Define the * operation
def set_star (X Y : Set ℝ) : Set ℝ := (set_difference X Y) ∪ (set_difference Y X)

-- State the theorem
theorem set_star_A_B :
  set_star A B = {x | -3 < x ∧ x < 0} ∪ {x | x > 3} := by
  sorry

end NUMINAMATH_CALUDE_set_star_A_B_l1528_152841


namespace NUMINAMATH_CALUDE_sector_central_angle_l1528_152802

theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = (2/5) * Real.pi) :
  (2 * area) / (r^2) = Real.pi / 5 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1528_152802


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1528_152876

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) (d : ℚ) 
  (h1 : arithmetic_sequence a d)
  (h2 : a 7 * a 11 = 6)
  (h3 : a 4 + a 14 = 5) :
  d = 1/4 ∨ d = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1528_152876


namespace NUMINAMATH_CALUDE_heath_age_l1528_152843

theorem heath_age (heath_age jude_age : ℕ) : 
  jude_age = 2 →
  heath_age + 5 = 3 * (jude_age + 5) →
  heath_age = 16 := by
sorry

end NUMINAMATH_CALUDE_heath_age_l1528_152843


namespace NUMINAMATH_CALUDE_convoy_vehicles_l1528_152848

theorem convoy_vehicles (bridge_length : ℝ) (convoy_speed : ℝ) (crossing_time : ℝ)
                        (vehicle_length : ℝ) (gap_length : ℝ) :
  bridge_length = 298 →
  convoy_speed = 4 →
  crossing_time = 115 →
  vehicle_length = 6 →
  gap_length = 20 →
  ∃ (n : ℕ), n * vehicle_length + (n - 1) * gap_length = convoy_speed * crossing_time - bridge_length ∧
             n = 7 :=
by sorry

end NUMINAMATH_CALUDE_convoy_vehicles_l1528_152848


namespace NUMINAMATH_CALUDE_binomial_9_choose_5_l1528_152800

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_choose_5_l1528_152800


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1528_152837

/-- Given a line L1: 2x + 3y = 9, prove that a line L2 perpendicular to L1 with y-intercept 5 has x-intercept -10/3 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := fun x y ↦ 2 * x + 3 * y = 9
  let m1 : ℝ := -2 / 3  -- slope of L1
  let m2 : ℝ := -1 / m1  -- slope of perpendicular line
  let L2 : ℝ → ℝ → Prop := fun x y ↦ y = m2 * x + 5  -- equation of perpendicular line
  let x_intercept : ℝ := -10 / 3
  (∀ x y, L2 x y → y = 0 → x = x_intercept) :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1528_152837


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_98_and_n_l1528_152842

theorem greatest_common_divisor_of_98_and_n (n : ℕ) : 
  (∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
   {d : ℕ | d ∣ 98 ∧ d ∣ n} = {d1, d2, d3}) → 
  Nat.gcd 98 n = 49 := by
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_98_and_n_l1528_152842


namespace NUMINAMATH_CALUDE_marks_initial_money_l1528_152867

theorem marks_initial_money (x : ℝ) : 
  x / 2 + 14 + x / 3 + 16 = x → x = 180 :=
by sorry

end NUMINAMATH_CALUDE_marks_initial_money_l1528_152867


namespace NUMINAMATH_CALUDE_square_root_of_25_l1528_152866

-- Define the concept of square root
def is_square_root (x y : ℝ) : Prop := y^2 = x

-- Theorem statement
theorem square_root_of_25 : 
  ∃ (a b : ℝ), a ≠ b ∧ is_square_root 25 a ∧ is_square_root 25 b :=
sorry

end NUMINAMATH_CALUDE_square_root_of_25_l1528_152866


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1528_152886

def equation (x : ℝ) : Prop :=
  (3 * x) / (x - 3) + (3 * x^2 - 45) / (x + 3) = 14

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ equation x ∧ ∀ (y : ℝ), y > 0 ∧ equation y → x ≤ y :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1528_152886


namespace NUMINAMATH_CALUDE_triangle_area_l1528_152839

/-- The area of a triangle with base 10 cm and height 3 cm is 15 cm² -/
theorem triangle_area : 
  let base : ℝ := 10
  let height : ℝ := 3
  let area : ℝ := (1/2) * base * height
  area = 15 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1528_152839


namespace NUMINAMATH_CALUDE_power_of_power_three_l1528_152877

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l1528_152877


namespace NUMINAMATH_CALUDE_lollipops_eaten_by_children_l1528_152868

/-- The number of lollipops Sushi's father bought -/
def initial_lollipops : ℕ := 12

/-- The number of lollipops left -/
def remaining_lollipops : ℕ := 7

/-- The number of lollipops eaten by the children -/
def eaten_lollipops : ℕ := initial_lollipops - remaining_lollipops

theorem lollipops_eaten_by_children : eaten_lollipops = 5 := by
  sorry

end NUMINAMATH_CALUDE_lollipops_eaten_by_children_l1528_152868


namespace NUMINAMATH_CALUDE_average_weight_of_children_l1528_152822

def regression_equation (x : ℝ) : ℝ := 2 * x + 7

def children_ages : List ℝ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

theorem average_weight_of_children :
  let weights := children_ages.map regression_equation
  (weights.sum / weights.length) = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_children_l1528_152822


namespace NUMINAMATH_CALUDE_cars_added_during_play_l1528_152853

/-- The number of cars added during a play, given initial car counts and final total. -/
def cars_added (front_initial : ℕ) (back_multiplier : ℕ) (total_final : ℕ) : ℕ :=
  total_final - (front_initial + back_multiplier * front_initial)

/-- Theorem stating that 400 cars were added during the play. -/
theorem cars_added_during_play :
  cars_added 100 2 700 = 400 := by sorry

end NUMINAMATH_CALUDE_cars_added_during_play_l1528_152853


namespace NUMINAMATH_CALUDE_construction_material_total_l1528_152809

theorem construction_material_total (gravel sand : ℝ) 
  (h1 : gravel = 5.91) (h2 : sand = 8.11) : 
  gravel + sand = 14.02 := by
  sorry

end NUMINAMATH_CALUDE_construction_material_total_l1528_152809


namespace NUMINAMATH_CALUDE_sphere_division_l1528_152844

theorem sphere_division (π : ℝ) (h_π : π > 0) : 
  ∃ (R : ℝ), R > 0 ∧ 
  (4 / 3 * π * R^3 = 125 * (4 / 3 * π * 1^3)) ∧ 
  R = 5 := by
sorry

end NUMINAMATH_CALUDE_sphere_division_l1528_152844


namespace NUMINAMATH_CALUDE_quadrant_I_solution_l1528_152879

theorem quadrant_I_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 5 ∧ c * x + y = 7 ∧ x > 3 ∧ y > 1) ↔ c < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_l1528_152879


namespace NUMINAMATH_CALUDE_josh_pencils_l1528_152847

theorem josh_pencils (initial : ℕ) (given_away : ℕ) (left : ℕ) : 
  given_away = 31 → left = 111 → initial = given_away + left →
  initial = 142 := by sorry

end NUMINAMATH_CALUDE_josh_pencils_l1528_152847


namespace NUMINAMATH_CALUDE_modulus_of_complex_quotient_l1528_152874

/-- The modulus of the complex number (4+3i)/(1-2i) is √5 -/
theorem modulus_of_complex_quotient :
  Complex.abs ((4 : ℂ) + 3 * Complex.I) / ((1 : ℂ) - 2 * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_quotient_l1528_152874


namespace NUMINAMATH_CALUDE_f_properties_l1528_152832

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.log x - a

theorem f_properties (a : ℝ) (h : a > 0) :
  -- Part 1
  (∃ (x : ℝ), x > 0 ∧ f a x = 0 ∧ ∀ (y : ℝ), y > 0 → f a y ≥ f a x) ∧
  (¬∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f a y ≤ f a x) ∧
  -- Part 2
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 →
    1 / a < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < a) ∧
  -- Part 3
  (∀ (x : ℝ), x > 0 → Real.exp (2 * x - 2) - Real.exp (x - 1) * Real.log x - x ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l1528_152832


namespace NUMINAMATH_CALUDE_target_hit_probability_l1528_152856

theorem target_hit_probability (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.6) (h_B : prob_B = 0.5) : 
  1 - (1 - prob_A) * (1 - prob_B) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1528_152856


namespace NUMINAMATH_CALUDE_trapezoid_median_equilateral_triangles_l1528_152888

/-- The median of a trapezoid formed by sides of two equilateral triangles -/
theorem trapezoid_median_equilateral_triangles 
  (large_side : ℝ) 
  (small_side : ℝ) 
  (h1 : large_side = 4) 
  (h2 : small_side = large_side / 2) : 
  (large_side + small_side) / 2 = 3 := by
  sorry

#check trapezoid_median_equilateral_triangles

end NUMINAMATH_CALUDE_trapezoid_median_equilateral_triangles_l1528_152888


namespace NUMINAMATH_CALUDE_data_set_average_l1528_152810

theorem data_set_average (x : ℝ) : 
  (2 + 3 + 4 + x + 6) / 5 = 4 → x = 5 := by
sorry

end NUMINAMATH_CALUDE_data_set_average_l1528_152810


namespace NUMINAMATH_CALUDE_min_sum_of_product_2310_l1528_152869

theorem min_sum_of_product_2310 (a b c : ℕ+) (h : a * b * c = 2310) :
  ∃ (x y z : ℕ+), x * y * z = 2310 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 42 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2310_l1528_152869


namespace NUMINAMATH_CALUDE_work_completion_time_l1528_152857

/-- Given that:
  * p can complete the work in 20 days
  * p and q work together for 2 days
  * After 2 days of working together, 0.7 of the work is left
  Prove that q can complete the work alone in 10 days -/
theorem work_completion_time (p_time q_time : ℝ) (h1 : p_time = 20) 
  (h2 : 2 * (1 / p_time + 1 / q_time) = 0.3) : q_time = 10 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l1528_152857


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_line_l1528_152859

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations
def contained_in (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry

-- State the theorem
theorem perpendicular_planes_from_perpendicular_line 
  (α β : Plane) (l : Line) :
  contained_in l β → perpendicular_line_plane l α → perpendicular_plane_plane α β := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_line_l1528_152859


namespace NUMINAMATH_CALUDE_power_function_exponent_l1528_152860

/-- A power function passing through (1/4, 1/2) has exponent 1/2 -/
theorem power_function_exponent (m : ℝ) (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = m * x^a) ∧ f (1/4) = 1/2) →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_power_function_exponent_l1528_152860


namespace NUMINAMATH_CALUDE_magic_potion_cooking_time_l1528_152884

/-- Represents a time on a 24-hour digital clock --/
structure DigitalTime where
  hours : Fin 24
  minutes : Fin 60

/-- Checks if a given time is a magic moment --/
def isMagicMoment (t : DigitalTime) : Prop :=
  t.hours = t.minutes

/-- Calculates the time difference between two DigitalTimes in minutes --/
def timeDifference (start finish : DigitalTime) : ℕ :=
  sorry

/-- Theorem stating the existence of a valid cooking time for the magic potion --/
theorem magic_potion_cooking_time :
  ∃ (start finish : DigitalTime),
    isMagicMoment start ∧
    isMagicMoment finish ∧
    90 ≤ timeDifference start finish ∧
    timeDifference start finish ≤ 120 ∧
    timeDifference start finish = 98 :=
  sorry

end NUMINAMATH_CALUDE_magic_potion_cooking_time_l1528_152884


namespace NUMINAMATH_CALUDE_range_of_a_l1528_152870

/-- Proposition p: The equation a²x² + ax - 2 = 0 has a solution in the interval [-1, 1] -/
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0

/-- Proposition q: There is only one real number x that satisfies x² + 2ax + 2a ≤ 0 -/
def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

/-- If both p and q are false, then -1 < a < 0 or 0 < a < 1 -/
theorem range_of_a (a : ℝ) : ¬(p a) ∧ ¬(q a) → (-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1528_152870


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l1528_152818

theorem intersection_x_coordinate : 
  let line1 : ℝ → ℝ := λ x => 3 * x + 5
  let line2 : ℝ → ℝ := λ x => 35 - 5 * x
  ∃ x : ℝ, x = 15 / 4 ∧ line1 x = line2 x :=
by sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l1528_152818


namespace NUMINAMATH_CALUDE_smallest_matching_set_size_l1528_152863

theorem smallest_matching_set_size : ∃ (N₁ N₂ : Nat), 
  (10000 ≤ N₁ ∧ N₁ < 100000) ∧ 
  (10000 ≤ N₂ ∧ N₂ < 100000) ∧ 
  ∀ (A : Nat), 
    (10000 ≤ A ∧ A < 100000) → 
    (∀ (i j : Fin 5), i ≤ j → (A / 10^(4 - i.val) % 10) ≤ (A / 10^(4 - j.val) % 10)) →
    ∃ (k : Fin 5), 
      ((N₁ / 10^(4 - k.val)) % 10 = (A / 10^(4 - k.val)) % 10) ∨ 
      ((N₂ / 10^(4 - k.val)) % 10 = (A / 10^(4 - k.val)) % 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_matching_set_size_l1528_152863


namespace NUMINAMATH_CALUDE_equations_solutions_l1528_152897

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x - 1 = 0
def equation2 (x : ℝ) : Prop := (x + 3)^2 = x + 3

-- Define the solution sets
def solutions1 : Set ℝ := {2 + Real.sqrt 5, 2 - Real.sqrt 5}
def solutions2 : Set ℝ := {-3, -2}

-- Theorem statement
theorem equations_solutions :
  (∀ x : ℝ, equation1 x ↔ x ∈ solutions1) ∧
  (∀ x : ℝ, equation2 x ↔ x ∈ solutions2) := by
  sorry

end NUMINAMATH_CALUDE_equations_solutions_l1528_152897


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1528_152806

theorem sum_of_three_numbers : ∀ (a b c : ℕ),
  b = 72 →
  a = 2 * b →
  c = a / 3 →
  a + b + c = 264 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1528_152806


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l1528_152819

/-- A batch of products -/
structure Batch where
  good : ℕ
  defective : ℕ
  h_good : good > 2
  h_defective : defective > 2

/-- A sample of two items from a batch -/
structure Sample (b : Batch) where
  good : Fin 3
  defective : Fin 3
  h_sum : good.val + defective.val = 2

/-- Event: At least one defective product in the sample -/
def at_least_one_defective (s : Sample b) : Prop :=
  s.defective.val ≥ 1

/-- Event: All products in the sample are good -/
def all_good (s : Sample b) : Prop :=
  s.good.val = 2

/-- The main theorem: "At least one defective" and "All good" are mutually exclusive -/
theorem mutually_exclusive_events (b : Batch) :
  ∀ (s : Sample b), ¬(at_least_one_defective s ∧ all_good s) :=
by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l1528_152819


namespace NUMINAMATH_CALUDE_seven_unit_disks_cover_radius_two_disk_l1528_152865

-- Define a disk as a pair (center, radius)
def Disk := ℝ × ℝ × ℝ

-- Define a function to check if a point is covered by a disk
def is_covered (point : ℝ × ℝ) (disk : Disk) : Prop :=
  let (cx, cy, r) := disk
  (point.1 - cx)^2 + (point.2 - cy)^2 ≤ r^2

-- Define a function to check if a point is covered by any disk in a list
def is_covered_by_any (point : ℝ × ℝ) (disks : List Disk) : Prop :=
  ∃ d ∈ disks, is_covered point d

-- Define the main theorem
theorem seven_unit_disks_cover_radius_two_disk :
  ∃ (arrangement : List Disk),
    (arrangement.length = 7) ∧
    (∀ d ∈ arrangement, d.2.2 = 1) ∧
    (∀ point : ℝ × ℝ, point.1^2 + point.2^2 ≤ 4 → is_covered_by_any point arrangement) :=
sorry

end NUMINAMATH_CALUDE_seven_unit_disks_cover_radius_two_disk_l1528_152865


namespace NUMINAMATH_CALUDE_spinner_probability_l1528_152890

theorem spinner_probability (p_largest p_next_largest p_smallest : ℝ) : 
  p_largest = (1 : ℝ) / 2 →
  p_next_largest = (1 : ℝ) / 3 →
  p_largest + p_next_largest + p_smallest = 1 →
  p_smallest = (1 : ℝ) / 6 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l1528_152890


namespace NUMINAMATH_CALUDE_truck_wheels_count_l1528_152831

/-- The toll formula for a truck crossing a bridge -/
def toll_formula (x : ℕ) : ℚ :=
  1.50 + 1.50 * (x - 2)

/-- The number of wheels on the front axle of the truck -/
def front_axle_wheels : ℕ := 2

/-- The number of wheels on each of the other axles of the truck -/
def other_axle_wheels : ℕ := 4

/-- Theorem stating that a truck with the given wheel configuration has 18 wheels in total -/
theorem truck_wheels_count :
  ∀ (x : ℕ), 
  x > 0 →
  toll_formula x = 6 →
  front_axle_wheels + (x - 1) * other_axle_wheels = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_truck_wheels_count_l1528_152831


namespace NUMINAMATH_CALUDE_quadratic_rewrite_product_l1528_152827

theorem quadratic_rewrite_product (a b c : ℤ) : 
  (∀ x : ℝ, 16 * x^2 - 40 * x - 72 = (a * x + b)^2 + c) → a * b = -20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_product_l1528_152827


namespace NUMINAMATH_CALUDE_min_height_box_l1528_152836

theorem min_height_box (x : ℝ) (h : x > 0) : 
  (2*x^2 + 4*x*(x + 4) ≥ 120) → (x + 4 ≥ 8) :=
by
  sorry

#check min_height_box

end NUMINAMATH_CALUDE_min_height_box_l1528_152836


namespace NUMINAMATH_CALUDE_power_of_product_l1528_152805

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1528_152805


namespace NUMINAMATH_CALUDE_race_difference_l1528_152894

/-- Given a race where A and B run 110 meters, with A finishing in 20 seconds
    and B finishing in 25 seconds, prove that A beats B by 22 meters. -/
theorem race_difference (race_distance : ℝ) (a_time b_time : ℝ) 
  (h_distance : race_distance = 110)
  (h_a_time : a_time = 20)
  (h_b_time : b_time = 25) :
  race_distance - (race_distance / b_time) * a_time = 22 :=
by sorry

end NUMINAMATH_CALUDE_race_difference_l1528_152894


namespace NUMINAMATH_CALUDE_number_equality_l1528_152830

theorem number_equality (x : ℚ) : 
  (35 / 100 : ℚ) * x = (20 / 100 : ℚ) * 50 → x = 200 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l1528_152830


namespace NUMINAMATH_CALUDE_bill_apples_left_l1528_152873

/-- The number of apples Bill has left after distributing and baking -/
def apples_left (initial_apples : ℕ) (children : ℕ) (apples_per_teacher : ℕ) 
  (teachers_per_child : ℕ) (pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (children * apples_per_teacher * teachers_per_child) - (pies * apples_per_pie)

/-- Theorem stating that Bill has 18 apples left -/
theorem bill_apples_left : 
  apples_left 50 2 3 2 2 10 = 18 := by sorry

end NUMINAMATH_CALUDE_bill_apples_left_l1528_152873


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l1528_152849

/-- Represents a systematic sampling method -/
def systematicSample (totalSize : Nat) (sampleSize : Nat) (startingNumber : Nat) : List Nat :=
  List.range sampleSize |>.map (fun i => startingNumber + i * (totalSize / sampleSize))

/-- The problem statement -/
theorem systematic_sampling_problem (totalSize sampleSize startingNumber : Nat) 
  (h1 : totalSize = 60)
  (h2 : sampleSize = 6)
  (h3 : startingNumber = 7) :
  systematicSample totalSize sampleSize startingNumber = [7, 17, 27, 37, 47, 57] := by
  sorry

#eval systematicSample 60 6 7

end NUMINAMATH_CALUDE_systematic_sampling_problem_l1528_152849


namespace NUMINAMATH_CALUDE_remaining_cards_l1528_152808

-- Define the initial number of baseball cards Mike has
def initial_cards : ℕ := 87

-- Define the number of cards Sam bought
def bought_cards : ℕ := 13

-- Theorem stating that Mike's remaining cards is the difference between initial and bought
theorem remaining_cards : initial_cards - bought_cards = 74 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cards_l1528_152808


namespace NUMINAMATH_CALUDE_outbound_speed_l1528_152880

/-- Proves that given a round trip of 2 hours, with an outbound journey of 70 minutes
    and a return journey at 105 km/h, the outbound journey speed is 75 km/h -/
theorem outbound_speed (total_time : Real) (outbound_time : Real) (return_speed : Real) :
  total_time = 2 →
  outbound_time = 70 / 60 →
  return_speed = 105 →
  (total_time - outbound_time) * return_speed = outbound_time * 75 := by
  sorry

#check outbound_speed

end NUMINAMATH_CALUDE_outbound_speed_l1528_152880


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l1528_152821

theorem factor_difference_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4*y) * (5 + 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l1528_152821


namespace NUMINAMATH_CALUDE_prank_combinations_l1528_152816

/-- The number of people available for the prank on each day of the week -/
def available_people : Fin 5 → ℕ
  | 0 => 2  -- Monday
  | 1 => 3  -- Tuesday
  | 2 => 6  -- Wednesday
  | 3 => 4  -- Thursday
  | 4 => 3  -- Friday

/-- The total number of different combinations of people for the prank across the week -/
def total_combinations : ℕ := (List.range 5).map available_people |>.prod

theorem prank_combinations :
  total_combinations = 432 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l1528_152816


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l1528_152881

theorem average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age : ℕ) (final_people : ℕ) (final_average : ℚ) : 
  initial_people = 7 →
  initial_average = 28 →
  leaving_age = 20 →
  final_people = 6 →
  final_average = 29 →
  (initial_people : ℚ) * initial_average - leaving_age = final_people * final_average := by
  sorry

#check average_age_after_leaving

end NUMINAMATH_CALUDE_average_age_after_leaving_l1528_152881


namespace NUMINAMATH_CALUDE_divisible_by_nine_l1528_152899

theorem divisible_by_nine (n : ℕ) : ∃ k : ℤ, 2^(2*n - 1) + 3*n + 4 = 9*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l1528_152899


namespace NUMINAMATH_CALUDE_negation_of_universal_is_existential_l1528_152823

def A : Set ℤ := {x | ∃ k, x = 2*k + 1}
def B : Set ℤ := {x | ∃ k, x = 2*k}

theorem negation_of_universal_is_existential :
  ¬(∀ x ∈ A, 2*x ∈ B) ↔ ∃ x ∈ A, 2*x ∉ B :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_is_existential_l1528_152823


namespace NUMINAMATH_CALUDE_length_BC_l1528_152804

/-- Right triangles ABC and ABD with specific properties -/
structure RightTriangles where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- D is on the x-axis
  h_D_on_x : D.2 = 0
  -- C is directly below A on the x-axis
  h_C_below_A : C.1 = A.1
  -- Distances
  h_AD : dist A D = 26
  h_BD : dist B D = 10
  h_AC : dist A C = 24
  -- ABC is a right triangle
  h_ABC_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- ABD is a right triangle
  h_ABD_right : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0

/-- The length of BC in the given configuration of right triangles -/
theorem length_BC (t : RightTriangles) : dist t.B t.C = 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_length_BC_l1528_152804


namespace NUMINAMATH_CALUDE_continuous_at_8_l1528_152898

def f (x : ℝ) : ℝ := 5 * x^2 + 5

theorem continuous_at_8 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 8| < δ → |f x - f 8| < ε := by
sorry

end NUMINAMATH_CALUDE_continuous_at_8_l1528_152898


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l1528_152817

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number -/
def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l1528_152817


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_a_l1528_152828

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x ≥ a^2 - a - 2} = {a : ℝ | -1 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_a_l1528_152828


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1528_152861

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 15*a^2 + 22*a - 8 = 0) →
  (b^3 - 15*b^2 + 22*b - 8 = 0) →
  (c^3 - 15*c^2 + 22*c - 8 = 0) →
  (a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 181/9) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1528_152861


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l1528_152815

/-- Given a rectangle ABCD with vertices A(0,0), B(0,2), C(3,2), and D(3,0),
    point E as the midpoint of diagonal BD, and point F on DA such that DF = 1/4 DA,
    prove that the ratio of the area of triangle DFE to the area of quadrilateral ABEF is 3/17. -/
theorem rectangle_area_ratio :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ × ℝ := (3, 2)
  let D : ℝ × ℝ := (3, 0)
  let E : ℝ × ℝ := ((D.1 + B.1) / 2, (D.2 + B.2) / 2)
  let F : ℝ × ℝ := (D.1 - (D.1 - A.1) / 4, A.2)
  let area_DFE := abs ((D.1 - F.1) * E.2) / 2
  let area_ABE := abs (B.1 * E.2 - E.1 * B.2) / 2
  let area_AEF := abs ((F.1 - A.1) * E.2) / 2
  let area_ABEF := area_ABE + area_AEF
  area_DFE / area_ABEF = 3 / 17 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l1528_152815


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l1528_152892

-- Define the equation
def equation (θ : Real) : Prop :=
  Real.cos (15 * Real.pi / 180) = Real.sin (35 * Real.pi / 180) + Real.sin (θ * Real.pi / 180)

-- State the theorem
theorem least_positive_angle_theorem :
  ∃ (θ : Real), θ > 0 ∧ equation θ ∧ ∀ (φ : Real), φ > 0 ∧ equation φ → θ ≤ φ ∧ θ = 35 :=
sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l1528_152892


namespace NUMINAMATH_CALUDE_proportional_set_l1528_152889

/-- A set of four positive real numbers is proportional if the product of the outer terms equals the product of the inner terms. -/
def is_proportional (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * d = b * c

/-- The set {3, 6, 9, 18} is proportional. -/
theorem proportional_set : is_proportional 3 6 9 18 := by
  sorry

end NUMINAMATH_CALUDE_proportional_set_l1528_152889


namespace NUMINAMATH_CALUDE_pastries_cakes_difference_l1528_152801

/-- The number of cakes made by the baker -/
def cakes_made : ℕ := 105

/-- The number of pastries made by the baker -/
def pastries_made : ℕ := 275

/-- The number of pastries sold by the baker -/
def pastries_sold : ℕ := 214

/-- The number of cakes sold by the baker -/
def cakes_sold : ℕ := 163

/-- Theorem stating the difference between pastries and cakes sold -/
theorem pastries_cakes_difference :
  pastries_sold - cakes_sold = 51 := by sorry

end NUMINAMATH_CALUDE_pastries_cakes_difference_l1528_152801


namespace NUMINAMATH_CALUDE_triangle_transformation_l1528_152826

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def transform (p : ℝ × ℝ) : ℝ × ℝ := 
  reflect_y (rotate_180 (reflect_x p))

theorem triangle_transformation :
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (4, 1)
  let C : ℝ × ℝ := (2, 3)
  (transform A = A) ∧ (transform B = B) ∧ (transform C = C) :=
sorry

end NUMINAMATH_CALUDE_triangle_transformation_l1528_152826


namespace NUMINAMATH_CALUDE_purchase_with_discounts_l1528_152882

/-- Calculates the final cost of a purchase with specific discounts -/
theorem purchase_with_discounts 
  (initial_total : ℝ) 
  (discounted_item_price : ℝ) 
  (item_discount_rate : ℝ) 
  (total_discount_rate : ℝ) 
  (h1 : initial_total = 54)
  (h2 : discounted_item_price = 20)
  (h3 : item_discount_rate = 0.2)
  (h4 : total_discount_rate = 0.1) :
  initial_total - 
  (discounted_item_price * item_discount_rate) - 
  ((initial_total - (discounted_item_price * item_discount_rate)) * total_discount_rate) = 45 := by
  sorry

end NUMINAMATH_CALUDE_purchase_with_discounts_l1528_152882


namespace NUMINAMATH_CALUDE_max_y_proof_unique_x_exists_no_greater_y_l1528_152807

/-- The maximum value of y such that there exists a unique x satisfying the given inequality -/
def max_y : ℕ := 112

theorem max_y_proof :
  ∀ y : ℕ, y > max_y →
    ¬(∃! x : ℕ, (9:ℚ)/17 < (x:ℚ)/(x + y) ∧ (x:ℚ)/(x + y) < 8/15) :=
by sorry

theorem unique_x_exists :
  ∃! x : ℕ, (9:ℚ)/17 < (x:ℚ)/(x + max_y) ∧ (x:ℚ)/(x + max_y) < 8/15 :=
by sorry

theorem no_greater_y :
  ∀ y : ℕ, y > max_y →
    ¬(∃! x : ℕ, (9:ℚ)/17 < (x:ℚ)/(x + y) ∧ (x:ℚ)/(x + y) < 8/15) :=
by sorry

end NUMINAMATH_CALUDE_max_y_proof_unique_x_exists_no_greater_y_l1528_152807


namespace NUMINAMATH_CALUDE_quadratic_roots_quadratic_function_l1528_152834

-- Part 1
theorem quadratic_roots (a b c : ℝ) 
  (h : Real.sqrt (a - 2) + abs (b + 1) + (c + 2)^2 = 0) :
  let f := fun x => a * x^2 + b * x + c
  ∃ x1 x2 : ℝ, x1 = (1 + Real.sqrt 17) / 4 ∧ 
              x2 = (1 - Real.sqrt 17) / 4 ∧
              f x1 = 0 ∧ f x2 = 0 :=
sorry

-- Part 2
theorem quadratic_function (a b c : ℝ) 
  (h1 : a * (-1)^2 + b * (-1) + c = 0)
  (h2 : a * 0^2 + b * 0 + c = -3)
  (h3 : a * 3^2 + b * 3 + c = 0) :
  ∀ x : ℝ, a * x^2 + b * x + c = x^2 - 2*x - 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_quadratic_function_l1528_152834


namespace NUMINAMATH_CALUDE_y_value_at_8_l1528_152887

-- Define the function y = k * x^(1/3)
def y (k x : ℝ) : ℝ := k * x^(1/3)

-- State the theorem
theorem y_value_at_8 (k : ℝ) :
  y k 64 = 4 * Real.sqrt 3 → y k 8 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_at_8_l1528_152887


namespace NUMINAMATH_CALUDE_smaller_angle_measure_l1528_152814

/-- A parallelogram with one angle exceeding the other by 70 degrees -/
structure SpecialParallelogram where
  /-- The measure of the smaller angle in degrees -/
  smaller_angle : ℝ
  /-- The measure of the larger angle in degrees -/
  larger_angle : ℝ
  /-- The larger angle exceeds the smaller angle by 70 degrees -/
  angle_difference : larger_angle = smaller_angle + 70
  /-- The sum of adjacent angles is 180 degrees -/
  angle_sum : smaller_angle + larger_angle = 180

/-- The measure of the smaller angle in a special parallelogram is 55 degrees -/
theorem smaller_angle_measure (p : SpecialParallelogram) : p.smaller_angle = 55 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_measure_l1528_152814


namespace NUMINAMATH_CALUDE_divisor_count_problem_l1528_152845

theorem divisor_count_problem (n : ℕ+) :
  (∃ (d : ℕ → ℕ), d (110 * n ^ 3) = 110) →
  (∃ (d : ℕ → ℕ), d (81 * n ^ 4) = 325) :=
by sorry

end NUMINAMATH_CALUDE_divisor_count_problem_l1528_152845


namespace NUMINAMATH_CALUDE_existence_of_special_sequences_l1528_152829

theorem existence_of_special_sequences : ∃ (a b : ℕ → ℕ), 
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, b n < b (n + 1)) ∧ 
  (a 1 = 25) ∧ 
  (b 1 = 57) ∧ 
  (∀ n, (b n)^2 + 1 ≡ 0 [MOD (a n) * ((a n) + 1)]) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_sequences_l1528_152829


namespace NUMINAMATH_CALUDE_solve_for_a_l1528_152820

theorem solve_for_a : ∃ a : ℝ, (2 * (3 - 1) - a = 0) ∧ a = 4 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l1528_152820


namespace NUMINAMATH_CALUDE_well_depth_and_rope_length_l1528_152893

theorem well_depth_and_rope_length :
  ∃! (x y : ℝ),
    x / 4 - 3 = y ∧
    x / 5 + 1 = y ∧
    x = 80 ∧
    y = 17 := by
  sorry

end NUMINAMATH_CALUDE_well_depth_and_rope_length_l1528_152893


namespace NUMINAMATH_CALUDE_triangle_inequality_arithmetic_sequence_l1528_152891

/-- An arithmetic sequence with positive terms and positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ d > 0 ∧ ∀ n, a (n + 1) = a n + d

theorem triangle_inequality_arithmetic_sequence 
  (a : ℕ → ℝ) (d : ℝ) (h : ArithmeticSequence a d) :
  a 2 + a 3 > a 4 ∧ a 2 + a 4 > a 3 ∧ a 3 + a 4 > a 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_arithmetic_sequence_l1528_152891


namespace NUMINAMATH_CALUDE_sequence_general_term_l1528_152895

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = n^2 + 3n,
    prove that the general term a_n = 2n + 2 for all positive integers n. -/
theorem sequence_general_term (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
    (h_S : ∀ n : ℕ+, S n = n^2 + 3*n) :
    ∀ n : ℕ+, a n = 2*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1528_152895


namespace NUMINAMATH_CALUDE_dice_probability_l1528_152858

theorem dice_probability : 
  let n : ℕ := 5  -- number of dice
  let s : ℕ := 12  -- number of sides on each die
  let p_one_digit : ℚ := 3 / 4  -- probability of rolling a one-digit number
  let p_two_digit : ℚ := 1 / 4  -- probability of rolling a two-digit number
  Nat.choose n (n / 2) * p_two_digit ^ (n / 2) * p_one_digit ^ (n - n / 2) = 135 / 512 :=
by sorry

end NUMINAMATH_CALUDE_dice_probability_l1528_152858


namespace NUMINAMATH_CALUDE_sqrt_two_div_sqrt_eighteen_equals_one_third_l1528_152813

theorem sqrt_two_div_sqrt_eighteen_equals_one_third :
  Real.sqrt 2 / Real.sqrt 18 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_div_sqrt_eighteen_equals_one_third_l1528_152813


namespace NUMINAMATH_CALUDE_merry_apples_sold_l1528_152885

/-- The number of apples Merry sold on Saturday and Sunday -/
def apples_sold (saturday_boxes : ℕ) (sunday_boxes : ℕ) (apples_per_box : ℕ) (boxes_left : ℕ) : ℕ :=
  (saturday_boxes - sunday_boxes + sunday_boxes - boxes_left) * apples_per_box

/-- Theorem stating that Merry sold 470 apples on Saturday and Sunday -/
theorem merry_apples_sold :
  apples_sold 50 25 10 3 = 470 := by
  sorry

end NUMINAMATH_CALUDE_merry_apples_sold_l1528_152885


namespace NUMINAMATH_CALUDE_multiply_eight_negative_half_l1528_152803

theorem multiply_eight_negative_half : 8 * (-1/2 : ℚ) = -4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_eight_negative_half_l1528_152803


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_property_l1528_152883

/-- A geometric sequence with its sum function -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) = a n * (a 2 / a 1)
  sum_property : ∀ n : ℕ, n > 0 → S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

/-- The main theorem -/
theorem geometric_sequence_sum_property 
  (seq : GeometricSequence) 
  (h1 : seq.S 3 = 8) 
  (h2 : seq.S 6 = 7) : 
  seq.a 7 + seq.a 8 + seq.a 9 = 1/8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_property_l1528_152883


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1528_152840

/-- A quadratic expression x^2 + bx + c is a perfect square trinomial if there exists a real number k such that x^2 + bx + c = (x + k)^2 for all x. -/
def IsPerfectSquareTrinomial (b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x^2 + b*x + c = (x + k)^2

/-- If x^2 - 8x + a is a perfect square trinomial, then a = 16. -/
theorem perfect_square_trinomial_condition (a : ℝ) :
  IsPerfectSquareTrinomial (-8) a → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1528_152840


namespace NUMINAMATH_CALUDE_yadav_expenditure_l1528_152850

theorem yadav_expenditure (monthly_salary : ℝ) : 
  monthly_salary > 0 →
  (0.6 * monthly_salary) + (0.5 * (0.4 * monthly_salary)) + (0.2 * monthly_salary) = monthly_salary →
  (0.2 * monthly_salary) * 12 = 24624 →
  0.5 * (0.4 * monthly_salary) = 2052 := by
sorry

end NUMINAMATH_CALUDE_yadav_expenditure_l1528_152850


namespace NUMINAMATH_CALUDE_closest_to_99_times_9_l1528_152872

def options : List ℤ := [10000, 100, 100000, 1000, 10]

theorem closest_to_99_times_9 :
  ∀ x ∈ options, |99 * 9 - 1000| ≤ |99 * 9 - x| :=
sorry

end NUMINAMATH_CALUDE_closest_to_99_times_9_l1528_152872


namespace NUMINAMATH_CALUDE_f_positive_at_one_f_solution_set_l1528_152812

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

-- Theorem 1
theorem f_positive_at_one (a : ℝ) :
  f a 1 > 0 ↔ a ∈ Set.Ioo (3 - 2 * Real.sqrt 3) (3 + 2 * Real.sqrt 3) :=
sorry

-- Theorem 2
theorem f_solution_set (a b : ℝ) :
  (∀ x, f a x > b ↔ x ∈ Set.Ioo (-1) 3) ↔
  ((a = 3 - Real.sqrt 3 ∨ a = 3 + Real.sqrt 3) ∧ b = -3) :=
sorry

end NUMINAMATH_CALUDE_f_positive_at_one_f_solution_set_l1528_152812


namespace NUMINAMATH_CALUDE_sum_20_is_850_l1528_152855

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums of the sequence
  sum_5 : S 5 = 10
  sum_10 : S 10 = 50

/-- The sum of the first 20 terms of the geometric sequence is 850 -/
theorem sum_20_is_850 (seq : GeometricSequence) : seq.S 20 = 850 := by
  sorry

end NUMINAMATH_CALUDE_sum_20_is_850_l1528_152855


namespace NUMINAMATH_CALUDE_solve_for_i_l1528_152838

-- Define the equation as a function of x and i
def equation (x i : ℝ) : Prop :=
  (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / i

-- State the theorem
theorem solve_for_i :
  ∃ i : ℝ, equation 0.3 i ∧ abs (i - 2.9993) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_i_l1528_152838


namespace NUMINAMATH_CALUDE_reflection_line_equation_l1528_152864

-- Define the points
def P : ℝ × ℝ := (3, 4)
def Q : ℝ × ℝ := (8, 9)
def R : ℝ × ℝ := (-5, 7)
def P' : ℝ × ℝ := (3, -6)
def Q' : ℝ × ℝ := (8, -11)
def R' : ℝ × ℝ := (-5, -9)

-- Define the line of reflection
def M : ℝ → ℝ := λ x => -1

-- Theorem statement
theorem reflection_line_equation :
  (∀ x y, M x = y ↔ y = -1) ∧
  (P.1 = P'.1 ∧ P.2 + P'.2 = 2 * (M P.1)) ∧
  (Q.1 = Q'.1 ∧ Q.2 + Q'.2 = 2 * (M Q.1)) ∧
  (R.1 = R'.1 ∧ R.2 + R'.2 = 2 * (M R.1)) :=
by sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l1528_152864


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_geq_3_l1528_152871

/-- A quadratic function of the form y = (x - m)^2 - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := (x - m)^2 - 1

/-- The function decreases as x increases when x ≤ 3 -/
def decreasing_for_x_leq_3 (m : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≤ x₂ → x₂ ≤ 3 → f m x₁ ≥ f m x₂

/-- If the quadratic function y = (x - m)^2 - 1 decreases as x increases when x ≤ 3,
    then m ≥ 3 -/
theorem quadratic_decreasing_implies_m_geq_3 (m : ℝ) :
  decreasing_for_x_leq_3 m → m ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_geq_3_l1528_152871


namespace NUMINAMATH_CALUDE_vector_operations_l1528_152846

/-- Given vectors a and b in ℝ², prove their sum and dot product -/
theorem vector_operations (a b : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (3, 1)) : 
  (a.1 + b.1, a.2 + b.2) = (4, 3) ∧ a.1 * b.1 + a.2 * b.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l1528_152846


namespace NUMINAMATH_CALUDE_sum_of_hundredth_powers_divisibility_l1528_152833

/-- QuadraticTrinomial represents a quadratic trinomial with integer coefficients -/
structure QuadraticTrinomial where
  p : ℤ
  q : ℤ

/-- Condition for the discriminant to be positive -/
def has_positive_discriminant (t : QuadraticTrinomial) : Prop :=
  t.p^2 - 4*t.q > 0

/-- Condition for coefficients to be divisible by 5 -/
def coeffs_divisible_by_5 (t : QuadraticTrinomial) : Prop :=
  5 ∣ t.p ∧ 5 ∣ t.q

/-- The sum of the hundredth powers of the roots -/
noncomputable def sum_of_hundredth_powers (t : QuadraticTrinomial) : ℝ :=
  let α := (-t.p + Real.sqrt (t.p^2 - 4*t.q)) / 2
  let β := (-t.p - Real.sqrt (t.p^2 - 4*t.q)) / 2
  α^100 + β^100

/-- The main theorem -/
theorem sum_of_hundredth_powers_divisibility
  (t : QuadraticTrinomial)
  (h_pos : has_positive_discriminant t)
  (h_div : coeffs_divisible_by_5 t) :
  ∃ (k : ℤ), sum_of_hundredth_powers t = k * (5^50 : ℝ) ∧
  ∀ (n : ℕ), n > 50 → ¬∃ (k : ℤ), sum_of_hundredth_powers t = k * (5^n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_sum_of_hundredth_powers_divisibility_l1528_152833
