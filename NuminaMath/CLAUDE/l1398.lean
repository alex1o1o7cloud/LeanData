import Mathlib

namespace NUMINAMATH_CALUDE_average_theorem_l1398_139892

theorem average_theorem (a b c : ℝ) :
  (a + b + c) / 3 = 12 →
  ((2*a + 1) + (2*b + 2) + (2*c + 3) + 2) / 4 = 20 := by
sorry

end NUMINAMATH_CALUDE_average_theorem_l1398_139892


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1398_139868

theorem function_passes_through_point (a : ℝ) (ha : 0 < a ∧ a < 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x - 1)
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1398_139868


namespace NUMINAMATH_CALUDE_cube_shadow_problem_l1398_139805

theorem cube_shadow_problem (x : ℝ) : 
  let cube_edge : ℝ := 2
  let shadow_area : ℝ := 300
  let total_shadow_area : ℝ := shadow_area + cube_edge^2
  let shadow_side : ℝ := Real.sqrt total_shadow_area
  x = (cube_edge / (shadow_side - cube_edge)) →
  ⌊1000 * x⌋ = 706 := by
sorry

end NUMINAMATH_CALUDE_cube_shadow_problem_l1398_139805


namespace NUMINAMATH_CALUDE_geometric_sequence_iff_a_eq_plus_minus_six_l1398_139894

/-- A sequence of three real numbers is geometric if the ratio between consecutive terms is constant. -/
def IsGeometricSequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

/-- The main theorem stating that the sequence 4, a, 9 is geometric if and only if a = 6 or a = -6 -/
theorem geometric_sequence_iff_a_eq_plus_minus_six :
  ∀ a : ℝ, IsGeometricSequence 4 a 9 ↔ (a = 6 ∨ a = -6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_iff_a_eq_plus_minus_six_l1398_139894


namespace NUMINAMATH_CALUDE_volleyball_tournament_equation_l1398_139800

/-- Represents a volleyball tournament. -/
structure VolleyballTournament where
  /-- The number of teams in the tournament. -/
  num_teams : ℕ
  /-- The total number of matches played. -/
  total_matches : ℕ
  /-- Each pair of teams plays against each other once. -/
  each_pair_plays_once : True

/-- Theorem stating the correct equation for the volleyball tournament. -/
theorem volleyball_tournament_equation (t : VolleyballTournament) 
  (h : t.total_matches = 28) : 
  (t.num_teams * (t.num_teams - 1)) / 2 = t.total_matches := by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_equation_l1398_139800


namespace NUMINAMATH_CALUDE_adjacent_book_left_of_middle_adjacent_book_not_right_of_middle_l1398_139827

/-- Represents the price of a book at a given position. -/
def book_price (c : ℕ) (n : ℕ) : ℕ := c + 2 * (n - 1)

/-- The theorem stating that the adjacent book is to the left of the middle book. -/
theorem adjacent_book_left_of_middle (c : ℕ) : 
  book_price c 31 = book_price c 16 + book_price c 15 :=
sorry

/-- The theorem stating that the adjacent book cannot be to the right of the middle book. -/
theorem adjacent_book_not_right_of_middle (c : ℕ) : 
  book_price c 31 ≠ book_price c 16 + book_price c 17 :=
sorry

end NUMINAMATH_CALUDE_adjacent_book_left_of_middle_adjacent_book_not_right_of_middle_l1398_139827


namespace NUMINAMATH_CALUDE_polynomial_without_cubic_and_linear_terms_l1398_139873

theorem polynomial_without_cubic_and_linear_terms 
  (a b : ℝ) 
  (h1 : a - 3 = 0)  -- Coefficient of x^3 is zero
  (h2 : 4 - b = 0)  -- Coefficient of x is zero
  : (a - b) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_without_cubic_and_linear_terms_l1398_139873


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l1398_139834

theorem hyperbola_midpoint_existence :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 - y₁^2/9 = 1) ∧
    (x₂^2 - y₂^2/9 = 1) ∧
    ((x₁ + x₂)/2 = -1) ∧
    ((y₁ + y₂)/2 = -4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l1398_139834


namespace NUMINAMATH_CALUDE_only_rational_root_l1398_139856

def f (x : ℚ) : ℚ := 3 * x^5 - 2 * x^4 + 5 * x^3 - x^2 - 7 * x + 2

theorem only_rational_root :
  ∀ (x : ℚ), f x = 0 ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_only_rational_root_l1398_139856


namespace NUMINAMATH_CALUDE_acceptable_outfits_l1398_139816

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 8

/-- The number of colors available for each item -/
def num_colors : ℕ := 8

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_items^3

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- The number of outfits where shirt and pants are the same color but hat is different -/
def shirt_pants_same : ℕ := num_colors * (num_colors - 1)

/-- Theorem stating the number of acceptable outfit combinations -/
theorem acceptable_outfits : 
  total_combinations - same_color_outfits - shirt_pants_same = 448 := by
  sorry

end NUMINAMATH_CALUDE_acceptable_outfits_l1398_139816


namespace NUMINAMATH_CALUDE_f_extrema_and_monotonicity_l1398_139850

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + a)

theorem f_extrema_and_monotonicity :
  (∃ (x_max x_min : ℝ), f (-3) x_max = 6 * Real.exp (-3) ∧
                        f (-3) x_min = -2 * Real.exp 1 ∧
                        ∀ x, f (-3) x ≤ f (-3) x_max ∧
                              f (-3) x ≥ f (-3) x_min) ∧
  (∀ a, (∀ x y, x < y → f a x < f a y) → a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_and_monotonicity_l1398_139850


namespace NUMINAMATH_CALUDE_sphere_surface_area_doubling_l1398_139807

/-- Given a sphere whose surface area doubles when its radius is doubled,
    prove that if the new surface area is 9856 cm², 
    then the original surface area is 2464 cm². -/
theorem sphere_surface_area_doubling (r : ℝ) :
  (4 * Real.pi * (2 * r)^2 = 9856) → (4 * Real.pi * r^2 = 2464) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_doubling_l1398_139807


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l1398_139809

/-- Given two speeds and an additional distance, proves that the actual distance traveled is 50 km -/
theorem actual_distance_traveled (speed1 speed2 additional_distance : ℝ) 
  (h1 : speed1 = 10)
  (h2 : speed2 = 14)
  (h3 : additional_distance = 20)
  (h4 : ∀ D : ℝ, D / speed1 = (D + additional_distance) / speed2) :
  ∃ D : ℝ, D = 50 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l1398_139809


namespace NUMINAMATH_CALUDE_triangle_abc_solutions_l1398_139852

theorem triangle_abc_solutions (b c : ℝ) (angle_B : ℝ) :
  b = 3 → c = 3 * Real.sqrt 3 → angle_B = π / 6 →
  ∃ (a angle_A angle_C : ℝ),
    ((angle_A = π / 2 ∧ angle_C = π / 3 ∧ a = Real.sqrt 21) ∨
     (angle_A = π / 6 ∧ angle_C = 2 * π / 3 ∧ a = 3)) ∧
    angle_A + angle_B + angle_C = π ∧
    a / (Real.sin angle_A) = b / (Real.sin angle_B) ∧
    b / (Real.sin angle_B) = c / (Real.sin angle_C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_solutions_l1398_139852


namespace NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l1398_139888

theorem tangent_line_to_x_ln_x (m : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (∀ x : ℝ, 2 * x + m = x₀ * Real.log x₀ + (Real.log x₀ + 1) * (x - x₀)) ∧
    (∀ x : ℝ, x > 0 → 2 * x + m ≥ x * Real.log x)) →
  m = -Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l1398_139888


namespace NUMINAMATH_CALUDE_gcd_6Tn_nplus1_eq_one_l1398_139829

/-- The nth triangular number -/
def T (n : ℕ+) : ℕ := (n * (n + 1)) / 2

/-- Theorem: The GCD of 6T_n and n+1 is always 1 for positive integers n -/
theorem gcd_6Tn_nplus1_eq_one (n : ℕ+) : Nat.gcd (6 * T n) (n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_6Tn_nplus1_eq_one_l1398_139829


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l1398_139858

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_and_line_in_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contains α n) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l1398_139858


namespace NUMINAMATH_CALUDE_minimum_commission_rate_l1398_139899

/-- The minimum commission rate problem -/
theorem minimum_commission_rate 
  (old_salary : ℝ) 
  (new_base_salary : ℝ) 
  (sale_value : ℝ) 
  (min_sales : ℝ) 
  (h1 : old_salary = 75000)
  (h2 : new_base_salary = 45000)
  (h3 : sale_value = 750)
  (h4 : min_sales = 266.67)
  : ∃ (commission_rate : ℝ), 
    commission_rate ≥ (old_salary - new_base_salary) / min_sales ∧ 
    commission_rate ≥ 112.50 :=
sorry

end NUMINAMATH_CALUDE_minimum_commission_rate_l1398_139899


namespace NUMINAMATH_CALUDE_problem_solution_l1398_139861

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ab ≤ 4 ∧ Real.sqrt a + Real.sqrt b ≤ 2 * Real.sqrt 2 ∧ a^2 + b^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1398_139861


namespace NUMINAMATH_CALUDE_aron_cleaning_time_l1398_139843

/-- Calculates the total cleaning time per week for Aron -/
def total_cleaning_time (vacuum_time : ℕ) (vacuum_freq : ℕ) (dust_time : ℕ) (dust_freq : ℕ) : ℕ :=
  vacuum_time * vacuum_freq + dust_time * dust_freq

/-- Proves that Aron spends 130 minutes per week cleaning -/
theorem aron_cleaning_time :
  total_cleaning_time 30 3 20 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_aron_cleaning_time_l1398_139843


namespace NUMINAMATH_CALUDE_investment_plans_count_l1398_139818

/-- The number of ways to distribute projects among cities --/
def distribute_projects (n_projects : ℕ) (n_cities : ℕ) (max_per_city : ℕ) : ℕ := sorry

/-- Theorem statement --/
theorem investment_plans_count :
  distribute_projects 3 4 2 = 60 := by sorry

end NUMINAMATH_CALUDE_investment_plans_count_l1398_139818


namespace NUMINAMATH_CALUDE_cube_skew_lines_theorem_l1398_139835

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D
  edge_length : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point1 : Point3D
  point2 : Point3D

/-- Calculate the distance between two skew lines in 3D space -/
def distance_between_skew_lines (l1 l2 : Line3D) : ℝ := sorry

/-- Check if a line segment is perpendicular to two other lines -/
def is_perpendicular_to_lines (segment : Line3D) (l1 l2 : Line3D) : Prop := sorry

/-- Main theorem about the distance between skew lines in a cube and their common perpendicular -/
theorem cube_skew_lines_theorem (cube : Cube) : 
  let A₁D := Line3D.mk cube.A₁ cube.D
  let D₁C := Line3D.mk cube.D₁ cube.C
  let X := Point3D.mk ((2 * cube.D.x + cube.A₁.x) / 3) ((2 * cube.D.y + cube.A₁.y) / 3) ((2 * cube.D.z + cube.A₁.z) / 3)
  let Y := Point3D.mk ((2 * cube.D₁.x + cube.C.x) / 3) ((2 * cube.D₁.y + cube.C.y) / 3) ((2 * cube.D₁.z + cube.C.z) / 3)
  let XY := Line3D.mk X Y
  distance_between_skew_lines A₁D D₁C = cube.edge_length * Real.sqrt 3 / 3 ∧
  is_perpendicular_to_lines XY A₁D D₁C := by
  sorry

end NUMINAMATH_CALUDE_cube_skew_lines_theorem_l1398_139835


namespace NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l1398_139825

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (π / 2 - α) = Real.sqrt 2 / 3) :
  Real.cos (π - 2 * α) = -5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l1398_139825


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_three_l1398_139864

theorem fraction_meaningful_iff_not_three (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_three_l1398_139864


namespace NUMINAMATH_CALUDE_cone_surface_area_ratio_l1398_139814

/-- For a cone whose lateral surface unfolds into a sector with a central angle of 90°,
    the ratio of the lateral surface area to the base area is 4. -/
theorem cone_surface_area_ratio (r : ℝ) (h : r > 0) : 
  let R := 4 * r
  let base_area := π * r^2
  let lateral_area := (1/4) * π * R^2
  lateral_area / base_area = 4 := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_ratio_l1398_139814


namespace NUMINAMATH_CALUDE_sales_tax_difference_l1398_139897

-- Define the item price
def item_price : ℝ := 20

-- Define the two tax rates
def tax_rate_1 : ℝ := 0.065
def tax_rate_2 : ℝ := 0.06

-- State the theorem
theorem sales_tax_difference : 
  (tax_rate_1 - tax_rate_2) * item_price = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l1398_139897


namespace NUMINAMATH_CALUDE_select_students_count_l1398_139853

/-- The number of ways to select 3 students from 5 boys and 3 girls, including both genders -/
def select_students : ℕ :=
  Nat.choose 3 1 * Nat.choose 5 2 + Nat.choose 3 2 * Nat.choose 5 1

/-- Theorem stating that the number of ways to select the students is 45 -/
theorem select_students_count : select_students = 45 := by
  sorry

#eval select_students

end NUMINAMATH_CALUDE_select_students_count_l1398_139853


namespace NUMINAMATH_CALUDE_bus_passengers_problem_l1398_139847

/-- Given a bus with an initial number of passengers and a number of passengers who got off,
    calculate the number of passengers remaining on the bus. -/
def passengers_remaining (initial : ℕ) (got_off : ℕ) : ℕ :=
  initial - got_off

/-- Theorem stating that given 90 initial passengers and 47 passengers who got off,
    the number of remaining passengers is 43. -/
theorem bus_passengers_problem :
  passengers_remaining 90 47 = 43 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_problem_l1398_139847


namespace NUMINAMATH_CALUDE_chiquita_height_l1398_139895

theorem chiquita_height :
  ∀ (chiquita_height martinez_height : ℝ),
    martinez_height = chiquita_height + 2 →
    chiquita_height + martinez_height = 12 →
    chiquita_height = 5 := by
  sorry

end NUMINAMATH_CALUDE_chiquita_height_l1398_139895


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1398_139806

theorem arithmetic_calculation : 8 / 2 - 3 - 9 + 3 * 9 - 3^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1398_139806


namespace NUMINAMATH_CALUDE_scalene_triangle_distinct_lines_l1398_139841

/-- A scalene triangle is a triangle where all sides and angles are different -/
structure ScaleneTriangle where
  -- We don't need to define the specific properties here, just the existence of the triangle
  exists_triangle : True

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side -/
def altitude (t : ScaleneTriangle) : ℕ := 3

/-- A median of a triangle is a line segment from a vertex to the midpoint of the opposite side -/
def median (t : ScaleneTriangle) : ℕ := 3

/-- An angle bisector of a triangle is a line that divides an angle into two equal parts -/
def angle_bisector (t : ScaleneTriangle) : ℕ := 3

/-- The total number of distinct lines in a scalene triangle -/
def total_distinct_lines (t : ScaleneTriangle) : ℕ :=
  altitude t + median t + angle_bisector t

theorem scalene_triangle_distinct_lines (t : ScaleneTriangle) :
  total_distinct_lines t = 9 := by
  sorry

end NUMINAMATH_CALUDE_scalene_triangle_distinct_lines_l1398_139841


namespace NUMINAMATH_CALUDE_taylors_pets_l1398_139882

theorem taylors_pets (taylor_pets : ℕ) (total_pets : ℕ) : 
  (3 * (2 * taylor_pets) + 2 * 2 + taylor_pets = total_pets) →
  (total_pets = 32) →
  (taylor_pets = 4) := by
sorry

end NUMINAMATH_CALUDE_taylors_pets_l1398_139882


namespace NUMINAMATH_CALUDE_parabola_equation_fixed_point_property_l1398_139824

-- Define the ellipse E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 8 = 1}

-- Define the parabola C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the right focus of the ellipse E
def right_focus_E : ℝ × ℝ := (1, 0)

-- Define the directrix of parabola C
def directrix_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Theorem for part (I)
theorem parabola_equation : 
  C = {p : ℝ × ℝ | p.2^2 = 4 * p.1} := by sorry

-- Theorem for part (II)
theorem fixed_point_property (P Q : ℝ × ℝ) 
  (hP : P ∈ C) (hQ : Q ∈ C) (hO : P ≠ (0, 0) ∧ Q ≠ (0, 0)) 
  (hPerp : (P.1 * Q.1 + P.2 * Q.2 = 0)) :
  ∃ (m n : ℝ), m * P.2 = P.1 + n ∧ m * Q.2 = Q.1 + n ∧ n = -4 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_fixed_point_property_l1398_139824


namespace NUMINAMATH_CALUDE_correct_sum_after_misreading_l1398_139828

/-- Given a three-digit number ABC where C was misread as 6 instead of 9,
    and the sum of AB6 and 57 is 823, prove that the correct sum of ABC and 57 is 826 -/
theorem correct_sum_after_misreading (A B : Nat) : 
  (100 * A + 10 * B + 6 + 57 = 823) → 
  (100 * A + 10 * B + 9 + 57 = 826) :=
by sorry

end NUMINAMATH_CALUDE_correct_sum_after_misreading_l1398_139828


namespace NUMINAMATH_CALUDE_factorial_division_l1398_139876

theorem factorial_division :
  (9 : ℕ).factorial / (4 : ℕ).factorial = 15120 :=
by
  have h1 : (9 : ℕ).factorial = 362880 := by sorry
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1398_139876


namespace NUMINAMATH_CALUDE_males_not_interested_count_l1398_139801

/-- Represents the survey results for a yoga class -/
structure YogaSurvey where
  total_not_interested : ℕ
  females_not_interested : ℕ

/-- Calculates the number of males not interested in the yoga class -/
def males_not_interested (survey : YogaSurvey) : ℕ :=
  survey.total_not_interested - survey.females_not_interested

/-- Theorem stating that the number of males not interested is 110 -/
theorem males_not_interested_count (survey : YogaSurvey) 
  (h1 : survey.total_not_interested = 200)
  (h2 : survey.females_not_interested = 90) : 
  males_not_interested survey = 110 := by
  sorry

#eval males_not_interested ⟨200, 90⟩

end NUMINAMATH_CALUDE_males_not_interested_count_l1398_139801


namespace NUMINAMATH_CALUDE_autumn_pencils_l1398_139855

theorem autumn_pencils (initial : ℕ) 
  (misplaced : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) (final : ℕ) : 
  misplaced = 7 → broken = 3 → found = 4 → bought = 2 → final = 16 →
  initial - misplaced - broken + found + bought = final →
  initial = 22 := by
sorry

end NUMINAMATH_CALUDE_autumn_pencils_l1398_139855


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l1398_139840

/-- 
Given a rectangular prism with edges in the ratio 2:1:1.5 and a total edge length of 72 cm,
prove that its volume is 192 cubic centimeters.
-/
theorem rectangular_prism_volume (x : ℝ) 
  (h1 : x > 0)
  (h2 : 4 * (2*x) + 4 * x + 4 * (1.5*x) = 72) : 
  (2*x) * x * (1.5*x) = 192 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l1398_139840


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l1398_139842

theorem arithmetic_series_sum (k : ℕ) : 
  let a₁ : ℤ := k^2 - 1
  let d : ℤ := 1
  let n : ℕ := 2 * k
  let S := (n : ℤ) * (2 * a₁ + (n - 1) * d) / 2
  S = 2 * k^3 + 2 * k^2 - 3 * k :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l1398_139842


namespace NUMINAMATH_CALUDE_non_sibling_probability_l1398_139849

/-- Represents a person in the room -/
structure Person where
  siblings : Nat

/-- Represents the room with people -/
def Room : Type := List Person

/-- The number of people in the room -/
def room_size : Nat := 7

/-- The condition that 4 people have exactly 1 sibling -/
def one_sibling_count (room : Room) : Prop :=
  (room.filter (fun p => p.siblings = 1)).length = 4

/-- The condition that 3 people have exactly 2 siblings -/
def two_siblings_count (room : Room) : Prop :=
  (room.filter (fun p => p.siblings = 2)).length = 3

/-- The probability of selecting two non-siblings -/
def prob_non_siblings (room : Room) : ℚ :=
  16 / 21

/-- The main theorem -/
theorem non_sibling_probability (room : Room) :
  room.length = room_size →
  one_sibling_count room →
  two_siblings_count room →
  prob_non_siblings room = 16 / 21 := by
  sorry


end NUMINAMATH_CALUDE_non_sibling_probability_l1398_139849


namespace NUMINAMATH_CALUDE_parents_age_when_mark_born_l1398_139848

/-- Given the ages of Mark and John, and their parents' current age relative to John's,
    prove the age of the parents when Mark was born. -/
theorem parents_age_when_mark_born
  (mark_age : ℕ)
  (john_age_diff : ℕ)
  (parents_age_factor : ℕ)
  (h1 : mark_age = 18)
  (h2 : john_age_diff = 10)
  (h3 : parents_age_factor = 5) :
  mark_age - (parents_age_factor * (mark_age - john_age_diff)) = 22 :=
by sorry

end NUMINAMATH_CALUDE_parents_age_when_mark_born_l1398_139848


namespace NUMINAMATH_CALUDE_final_price_percentage_l1398_139817

/-- Calculates the final price after applying multiple discounts -/
def final_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ price discount => price * (1 - discount)) original_price

/-- Theorem stating that after applying the given discounts, the final price is 58.14% of the original -/
theorem final_price_percentage (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let discounts := [0.2, 0.1, 0.05, 0.15]
  final_price original_price discounts / original_price = 0.5814 := by
sorry

#eval (final_price 100 [0.2, 0.1, 0.05, 0.15])

end NUMINAMATH_CALUDE_final_price_percentage_l1398_139817


namespace NUMINAMATH_CALUDE_special_polygon_area_l1398_139884

/-- A polygon with 32 congruent sides, where each side is perpendicular to its adjacent sides -/
structure SpecialPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  sides_eq : sides = 32
  perimeter_eq : perimeter = 64
  perimeter_calc : perimeter = sides * side_length

/-- The area of the special polygon -/
def polygon_area (p : SpecialPolygon) : ℝ :=
  36 * p.side_length ^ 2

theorem special_polygon_area (p : SpecialPolygon) : polygon_area p = 144 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_area_l1398_139884


namespace NUMINAMATH_CALUDE_fraction_numerator_is_twelve_l1398_139821

theorem fraction_numerator_is_twelve :
  ∀ (numerator : ℚ),
    (∃ (denominator : ℚ),
      denominator = 2 * numerator + 4 ∧
      numerator / denominator = 3 / 7) →
    numerator = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_is_twelve_l1398_139821


namespace NUMINAMATH_CALUDE_trigonometric_identity_30_degrees_l1398_139819

theorem trigonometric_identity_30_degrees : 
  (Real.tan (30 * π / 180))^2 - (Real.sin (30 * π / 180))^2 = 
  (Real.tan (30 * π / 180))^2 * (Real.sin (30 * π / 180))^2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_30_degrees_l1398_139819


namespace NUMINAMATH_CALUDE_fiftieth_ring_squares_l1398_139886

/-- The number of squares in the nth ring around a 2x3 rectangular center block -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 6

/-- The 50th ring contains 406 squares -/
theorem fiftieth_ring_squares : ring_squares 50 = 406 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_ring_squares_l1398_139886


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1398_139874

/-- The sum of an arithmetic sequence with first term 2, common difference 4, and 15 terms -/
def arithmetic_sum : ℕ → ℕ
  | 0 => 0
  | n + 1 => (2 + 4 * n) + arithmetic_sum n

theorem arithmetic_sequence_sum : arithmetic_sum 15 = 450 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1398_139874


namespace NUMINAMATH_CALUDE_eulers_conjecture_counterexample_l1398_139896

theorem eulers_conjecture_counterexample : 133^5 + 110^5 + 84^5 + 27^5 = 144^5 := by
  sorry

end NUMINAMATH_CALUDE_eulers_conjecture_counterexample_l1398_139896


namespace NUMINAMATH_CALUDE_triangle_isosceles_theorem_l1398_139898

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A triangle is isosceles if at least two of its sides are equal. -/
def isIsosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ a = c

theorem triangle_isosceles_theorem (a b c : ℕ) 
  (ha : isPrime a) (hb : isPrime b) (hc : isPrime c) 
  (hsum : a + b + c = 16) : 
  isIsosceles a b c := by
  sorry

#check triangle_isosceles_theorem

end NUMINAMATH_CALUDE_triangle_isosceles_theorem_l1398_139898


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l1398_139810

theorem chicken_wings_distribution (total_friends : ℕ) (initial_wings : ℕ) (cooked_wings : ℕ) (non_eating_friends : ℕ) :
  total_friends = 15 →
  initial_wings = 7 →
  cooked_wings = 45 →
  non_eating_friends = 2 →
  (initial_wings + cooked_wings) / (total_friends - non_eating_friends) = 4 :=
by sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l1398_139810


namespace NUMINAMATH_CALUDE_headphones_to_case_ratio_l1398_139803

def phone_cost : ℚ := 1000
def contract_cost_per_month : ℚ := 200
def case_cost_percentage : ℚ := 20 / 100
def total_first_year_cost : ℚ := 3700

def case_cost : ℚ := phone_cost * case_cost_percentage
def contract_cost_year : ℚ := contract_cost_per_month * 12
def headphones_cost : ℚ := total_first_year_cost - (phone_cost + case_cost + contract_cost_year)

theorem headphones_to_case_ratio :
  headphones_cost / case_cost = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_headphones_to_case_ratio_l1398_139803


namespace NUMINAMATH_CALUDE_min_ratio_cone_cylinder_volumes_l1398_139802

/-- The minimum ratio of the volume of a cone to the volume of a cylinder, 
    both circumscribed around the same sphere, is 4/3. -/
theorem min_ratio_cone_cylinder_volumes : ℝ := by
  -- Let r be the radius of the sphere
  -- Let h be the height of the cone
  -- Let a be the radius of the base of the cone
  -- The cylinder has height 2r and radius r
  -- The ratio of volumes is (π * a^2 * h / 3) / (π * r^2 * 2r)
  -- We need to prove that the minimum value of this ratio is 4/3
  sorry

end NUMINAMATH_CALUDE_min_ratio_cone_cylinder_volumes_l1398_139802


namespace NUMINAMATH_CALUDE_unique_age_pair_l1398_139872

/-- The set of possible ages for X's sons -/
def AgeSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

/-- Predicate for pairs of ages that satisfy the product condition -/
def ProductCondition (a b : ℕ) : Prop :=
  a ∈ AgeSet ∧ b ∈ AgeSet ∧ ∃ (c d : ℕ), c ≠ a ∧ d ≠ b ∧ c * d = a * b ∧ c ∈ AgeSet ∧ d ∈ AgeSet

/-- Predicate for pairs of ages that satisfy the ratio condition -/
def RatioCondition (a b : ℕ) : Prop :=
  a ∈ AgeSet ∧ b ∈ AgeSet ∧ ∃ (c d : ℕ), c ≠ a ∧ d ≠ b ∧ c * b = a * d ∧ c ∈ AgeSet ∧ d ∈ AgeSet

/-- Predicate for pairs of ages that satisfy the difference condition -/
def DifferenceCondition (a b : ℕ) : Prop :=
  a ∈ AgeSet ∧ b ∈ AgeSet ∧ ∀ (c d : ℕ), c ∈ AgeSet → d ∈ AgeSet → c - d = a - b → (c = a ∧ d = b) ∨ (c = b ∧ d = a)

/-- Theorem stating that (8, 2) is the only pair satisfying all conditions -/
theorem unique_age_pair : ∀ (a b : ℕ), 
  ProductCondition a b ∧ RatioCondition a b ∧ DifferenceCondition a b ↔ (a = 8 ∧ b = 2) ∨ (a = 2 ∧ b = 8) :=
sorry

end NUMINAMATH_CALUDE_unique_age_pair_l1398_139872


namespace NUMINAMATH_CALUDE_badminton_tournament_matches_l1398_139846

theorem badminton_tournament_matches (n : ℕ) (h : n = 6) : 
  (n * (n - 1)) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_badminton_tournament_matches_l1398_139846


namespace NUMINAMATH_CALUDE_natalie_bushes_needed_l1398_139833

/-- Represents the number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers needed to trade for one cabbage -/
def containers_per_cabbage : ℕ := 4

/-- Represents the number of cabbages Natalie wants to obtain -/
def target_cabbages : ℕ := 20

/-- Calculates the number of bushes needed to obtain a given number of cabbages -/
def bushes_needed (cabbages : ℕ) : ℕ :=
  (cabbages * containers_per_cabbage) / containers_per_bush

theorem natalie_bushes_needed : bushes_needed target_cabbages = 8 := by
  sorry

end NUMINAMATH_CALUDE_natalie_bushes_needed_l1398_139833


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l1398_139859

/-- Represents a square quilt -/
structure Quilt :=
  (size : ℕ)
  (shaded_diagonal_squares : ℕ)
  (shaded_full_squares : ℕ)

/-- Calculates the fraction of the quilt that is shaded -/
def shaded_fraction (q : Quilt) : ℚ :=
  (q.shaded_diagonal_squares / 2 + q.shaded_full_squares : ℚ) / (q.size * q.size)

/-- Theorem stating the fraction of the quilt that is shaded -/
theorem quilt_shaded_fraction :
  ∀ q : Quilt, 
    q.size = 4 → 
    q.shaded_diagonal_squares = 4 → 
    q.shaded_full_squares = 1 → 
    shaded_fraction q = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l1398_139859


namespace NUMINAMATH_CALUDE_circle_and_intersection_conditions_l1398_139880

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_and_intersection_conditions (m : ℝ) :
  (∀ x y, circle_equation x y m → m < 5) ∧
  (∃ x1 y1 x2 y2, 
    circle_equation x1 y1 m ∧
    circle_equation x2 y2 m ∧
    line_equation x1 y1 ∧
    line_equation x2 y2 ∧
    perpendicular x1 y1 x2 y2 →
    m = 8/5) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_intersection_conditions_l1398_139880


namespace NUMINAMATH_CALUDE_ticket_sales_total_l1398_139871

/-- Calculates the total sales from ticket sales given the number of tickets sold and prices. -/
theorem ticket_sales_total (total_tickets : ℕ) (child_tickets : ℕ) (adult_price : ℕ) (child_price : ℕ)
  (h1 : total_tickets = 42)
  (h2 : child_tickets = 16)
  (h3 : adult_price = 5)
  (h4 : child_price = 3) :
  (total_tickets - child_tickets) * adult_price + child_tickets * child_price = 178 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_total_l1398_139871


namespace NUMINAMATH_CALUDE_count_different_numerators_l1398_139860

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a recurring decimal in the form 0.ẋyż -/
structure RecurringDecimal where
  x : Digit
  y : Digit
  z : Digit

/-- Converts a RecurringDecimal to a rational number -/
def toRational (d : RecurringDecimal) : ℚ :=
  (d.x.val * 100 + d.y.val * 10 + d.z.val : ℕ) / 999

/-- The set of all possible RecurringDecimals -/
def allRecurringDecimals : Finset RecurringDecimal :=
  sorry

/-- The set of all possible numerators when converting RecurringDecimals to lowest terms -/
def allNumerators : Finset ℕ :=
  sorry

theorem count_different_numerators :
  Finset.card allNumerators = 660 :=
sorry

end NUMINAMATH_CALUDE_count_different_numerators_l1398_139860


namespace NUMINAMATH_CALUDE_morning_snowfall_l1398_139845

theorem morning_snowfall (total : ℝ) (afternoon : ℝ) (h1 : total = 0.63) (h2 : afternoon = 0.5) :
  total - afternoon = 0.13 := by
  sorry

end NUMINAMATH_CALUDE_morning_snowfall_l1398_139845


namespace NUMINAMATH_CALUDE_constant_value_l1398_139870

/-- The function f(x) = x + 4 -/
def f (x : ℝ) : ℝ := x + 4

/-- The theorem stating that if the equation has a solution x = 0.4, then c = 1 -/
theorem constant_value (c : ℝ) :
  (∃ x : ℝ, x = 0.4 ∧ (3 * f (x - 2)) / f 0 + 4 = f (2 * x + c)) → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l1398_139870


namespace NUMINAMATH_CALUDE_interest_credited_proof_l1398_139862

/-- The interest rate per annum as a decimal -/
def interest_rate : ℚ := 5 / 100

/-- The time period in years -/
def time_period : ℚ := 2 / 12

/-- The total amount after interest -/
def total_amount : ℚ := 255.31

/-- The simple interest formula -/
def simple_interest (principal : ℚ) : ℚ :=
  principal * (1 + interest_rate * time_period)

theorem interest_credited_proof :
  ∃ (principal : ℚ),
    simple_interest principal = total_amount ∧
    (total_amount - principal) * 100 = 210 := by
  sorry

end NUMINAMATH_CALUDE_interest_credited_proof_l1398_139862


namespace NUMINAMATH_CALUDE_book_price_calculation_l1398_139893

/-- Given a book with a suggested retail price, this theorem proves that
    if the marked price is 60% of the suggested retail price, and a customer
    pays 60% of the marked price, then the customer pays 36% of the
    suggested retail price. -/
theorem book_price_calculation (suggested_retail_price : ℝ) :
  let marked_price := 0.6 * suggested_retail_price
  let customer_price := 0.6 * marked_price
  customer_price = 0.36 * suggested_retail_price := by
  sorry

#check book_price_calculation

end NUMINAMATH_CALUDE_book_price_calculation_l1398_139893


namespace NUMINAMATH_CALUDE_prob_divisible_by_4_5_or_7_l1398_139822

/-- The probability of selecting a number from 1 to 200 that is divisible by 4, 5, or 7 -/
theorem prob_divisible_by_4_5_or_7 : 
  let S := Finset.range 200
  let divisible_by_4_5_or_7 := fun n => n % 4 = 0 ∨ n % 5 = 0 ∨ n % 7 = 0
  (S.filter divisible_by_4_5_or_7).card / S.card = 97 / 200 := by
  sorry

end NUMINAMATH_CALUDE_prob_divisible_by_4_5_or_7_l1398_139822


namespace NUMINAMATH_CALUDE_probability_six_heads_twelve_flips_l1398_139831

/-- The probability of getting exactly 6 heads when flipping a fair coin 12 times -/
theorem probability_six_heads_twelve_flips : 
  (Nat.choose 12 6 : ℚ) / 2^12 = 231 / 1024 := by sorry

end NUMINAMATH_CALUDE_probability_six_heads_twelve_flips_l1398_139831


namespace NUMINAMATH_CALUDE_integer_roots_fifth_degree_polynomial_l1398_139889

-- Define the set of possible values for m
def PossibleM : Set ℕ := {0, 1, 2, 3, 5}

-- Define a fifth-degree polynomial with integer coefficients
def FifthDegreePolynomial (a b c d e : ℤ) (x : ℤ) : ℤ :=
  x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

-- Define the number of integer roots (counting multiplicity)
def NumberOfIntegerRoots (p : ℤ → ℤ) : ℕ :=
  -- This is a placeholder definition. In reality, this would be more complex.
  0

-- The main theorem
theorem integer_roots_fifth_degree_polynomial 
  (a b c d e : ℤ) : 
  NumberOfIntegerRoots (FifthDegreePolynomial a b c d e) ∈ PossibleM :=
sorry

end NUMINAMATH_CALUDE_integer_roots_fifth_degree_polynomial_l1398_139889


namespace NUMINAMATH_CALUDE_smallest_number_with_five_remainders_l1398_139815

theorem smallest_number_with_five_remainders (n : ℕ) : 
  (∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ n ∧
    a % 11 = 3 ∧ b % 11 = 3 ∧ c % 11 = 3 ∧ d % 11 = 3 ∧ e % 11 = 3 ∧
    ∀ (x : ℕ), x ≤ n → x % 11 = 3 → (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e)) ↔
  n = 47 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_five_remainders_l1398_139815


namespace NUMINAMATH_CALUDE_problem_distribution_l1398_139867

theorem problem_distribution (n m : ℕ) (hn : n = 10) (hm : m = 7) :
  (Nat.choose n m * Nat.factorial m * n^(n - m) : ℕ) = 712800000 :=
by sorry

end NUMINAMATH_CALUDE_problem_distribution_l1398_139867


namespace NUMINAMATH_CALUDE_difference_of_squares_times_three_l1398_139875

theorem difference_of_squares_times_three :
  (650^2 - 350^2) * 3 = 900000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_times_three_l1398_139875


namespace NUMINAMATH_CALUDE_pen_cost_theorem_l1398_139885

/-- The average cost per pen in cents, rounded to the nearest whole number,
    given the number of pens, cost of pens, and shipping cost. -/
def average_cost_per_pen (num_pens : ℕ) (pen_cost shipping_cost : ℚ) : ℕ :=
  let total_cost_cents := (pen_cost + shipping_cost) * 100
  let average_cost := total_cost_cents / num_pens
  (average_cost + 1/2).floor.toNat

/-- Theorem stating that for 300 pens costing $29.85 with $8.10 shipping,
    the average cost per pen is 13 cents when rounded to the nearest whole number. -/
theorem pen_cost_theorem :
  average_cost_per_pen 300 (29.85) (8.10) = 13 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_theorem_l1398_139885


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l1398_139863

theorem quadratic_roots_expression (x₁ x₂ : ℝ) : 
  (-2 * x₁^2 + x₁ + 5 = 0) → 
  (-2 * x₂^2 + x₂ + 5 = 0) → 
  x₁^2 * x₂ + x₁ * x₂^2 = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l1398_139863


namespace NUMINAMATH_CALUDE_salary_change_l1398_139838

theorem salary_change (original_salary : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) : 
  increase_rate = 0.25 ∧ decrease_rate = 0.25 →
  (1 - decrease_rate) * (1 + increase_rate) * original_salary - original_salary = -0.0625 * original_salary := by
sorry

end NUMINAMATH_CALUDE_salary_change_l1398_139838


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1398_139808

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > y ∧ y > 0 → x^2 > y^2) ∧
  ∃ x y : ℝ, x^2 > y^2 ∧ ¬(x > y ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1398_139808


namespace NUMINAMATH_CALUDE_min_square_edge_for_circle_l1398_139891

-- Define the circumference of the circle
def circle_circumference : ℝ := 31.4

-- Define π as an approximation
def π : ℝ := 3.14

-- Define the theorem
theorem min_square_edge_for_circle :
  ∃ (edge_length : ℝ), 
    edge_length = circle_circumference / π ∧ 
    edge_length = 10 := by sorry

end NUMINAMATH_CALUDE_min_square_edge_for_circle_l1398_139891


namespace NUMINAMATH_CALUDE_floor_fraction_difference_l1398_139865

theorem floor_fraction_difference (n : ℕ) (hn : n = 2009) : 
  ⌊((n + 1)^2 : ℝ) / ((n - 1) * n) - (n - 1)^2 / (n * (n + 1))⌋ = 6 := by
  sorry

end NUMINAMATH_CALUDE_floor_fraction_difference_l1398_139865


namespace NUMINAMATH_CALUDE_student_number_factor_l1398_139857

theorem student_number_factor (f : ℝ) : 120 * f - 138 = 102 → f = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_number_factor_l1398_139857


namespace NUMINAMATH_CALUDE_five_pairs_l1398_139869

/-- The number of ordered pairs (b,c) of positive integers satisfying the given conditions -/
def count_pairs : ℕ := 
  (Finset.filter (fun p : ℕ × ℕ => 
    let b := p.1
    let c := p.2
    b > 0 ∧ c > 0 ∧ b^2 ≤ 9*c ∧ c^2 ≤ 9*b) 
  (Finset.product (Finset.range 4) (Finset.range 4))).card

/-- The theorem stating that there are exactly 5 such pairs -/
theorem five_pairs : count_pairs = 5 := by sorry

end NUMINAMATH_CALUDE_five_pairs_l1398_139869


namespace NUMINAMATH_CALUDE_inverse_proportion_point_ordering_l1398_139851

/-- Given an inverse proportion function and three points on its graph, 
    prove the ordering of their x-coordinates. -/
theorem inverse_proportion_point_ordering (k : ℝ) (a b c : ℝ) : 
  (∃ (k : ℝ), -3 = -((k^2 + 1) / a) ∧ 
               -2 = -((k^2 + 1) / b) ∧ 
                1 = -((k^2 + 1) / c)) →
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_ordering_l1398_139851


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1398_139811

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) = 
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧ B = 7 ∧ C = 9 ∧ D = 13 ∧ E = 5 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1398_139811


namespace NUMINAMATH_CALUDE_sixth_number_is_eight_l1398_139866

/-- A structure representing an increasing list of consecutive integers -/
structure ConsecutiveIntegerList where
  start : ℤ
  length : ℕ
  increasing : 0 < length

/-- The nth number in the list -/
def ConsecutiveIntegerList.nthNumber (list : ConsecutiveIntegerList) (n : ℕ) : ℤ :=
  list.start + n - 1

/-- The property that the sum of the 3rd and 4th numbers is 11 -/
def sumProperty (list : ConsecutiveIntegerList) : Prop :=
  list.nthNumber 3 + list.nthNumber 4 = 11

theorem sixth_number_is_eight (list : ConsecutiveIntegerList) 
    (h : sumProperty list) : list.nthNumber 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sixth_number_is_eight_l1398_139866


namespace NUMINAMATH_CALUDE_quadrilateral_area_quadrilateral_area_with_given_diagonal_and_offsets_l1398_139820

/-- The area of a quadrilateral with a diagonal of length d and offsets h₁ and h₂ is (d * h₁ + d * h₂) / 2 -/
theorem quadrilateral_area (d h₁ h₂ : ℝ) (h_d_pos : d > 0) (h_h₁_pos : h₁ > 0) (h_h₂_pos : h₂ > 0) :
  (d * h₁ + d * h₂) / 2 = d * (h₁ + h₂) / 2 :=
by sorry

theorem quadrilateral_area_with_given_diagonal_and_offsets :
  let diagonal : ℝ := 20
  let offset1 : ℝ := 5
  let offset2 : ℝ := 4
  (diagonal * offset1 + diagonal * offset2) / 2 = 90 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_quadrilateral_area_with_given_diagonal_and_offsets_l1398_139820


namespace NUMINAMATH_CALUDE_unique_rectangles_l1398_139878

/-- A rectangle with integer dimensions satisfying area and perimeter conditions -/
structure Rectangle where
  w : ℕ+  -- width
  l : ℕ+  -- length
  area_eq : w * l = 18
  perimeter_eq : 2 * w + 2 * l = 18

/-- The theorem stating that only two rectangles satisfy the conditions -/
theorem unique_rectangles : 
  ∀ r : Rectangle, (r.w = 3 ∧ r.l = 6) ∨ (r.w = 6 ∧ r.l = 3) :=
sorry

end NUMINAMATH_CALUDE_unique_rectangles_l1398_139878


namespace NUMINAMATH_CALUDE_clarence_initial_oranges_l1398_139837

/-- Proves that Clarence's initial number of oranges is 5 -/
theorem clarence_initial_oranges :
  ∀ (initial total from_joyce : ℕ),
    initial + from_joyce = total →
    from_joyce = 3 →
    total = 8 →
    initial = 5 := by
  sorry

end NUMINAMATH_CALUDE_clarence_initial_oranges_l1398_139837


namespace NUMINAMATH_CALUDE_simplify_expression_l1398_139844

theorem simplify_expression (x : ℝ) 
  (h1 : x ≠ 1) 
  (h2 : x ≠ -1) 
  (h3 : x ≠ (-1 + Real.sqrt 5) / 2) 
  (h4 : x ≠ (-1 - Real.sqrt 5) / 2) : 
  1 - (1 / (1 + x / (x^2 - 1))) = x / (x^2 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1398_139844


namespace NUMINAMATH_CALUDE_construct_triangle_from_excenters_l1398_139854

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the concept of an excenter
def is_excenter (P : Point) (T : Triangle) : Prop :=
  sorry -- Definition of excenter

-- Define the concept of altitude foot
def is_altitude_foot (P : Point) (T : Triangle) : Prop :=
  sorry -- Definition of altitude foot

-- Theorem statement
theorem construct_triangle_from_excenters 
  (A₁ B₁ C₁ : Point) 
  (h_excenters : is_excenter A₁ T ∧ is_excenter B₁ T ∧ is_excenter C₁ T) :
  ∃ (T : Triangle),
    is_altitude_foot T.A (Triangle.mk A₁ B₁ C₁) ∧
    is_altitude_foot T.B (Triangle.mk A₁ B₁ C₁) ∧
    is_altitude_foot T.C (Triangle.mk A₁ B₁ C₁) :=
by
  sorry

end NUMINAMATH_CALUDE_construct_triangle_from_excenters_l1398_139854


namespace NUMINAMATH_CALUDE_cost_price_of_article_l1398_139823

/-- The cost price of an article satisfying certain selling price conditions -/
theorem cost_price_of_article : ∃ C : ℝ, 
  (C = 400) ∧ 
  (0.8 * C = C - 0.2 * C) ∧ 
  (1.05 * C = C + 0.05 * C) ∧ 
  (1.05 * C - 0.8 * C = 100) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_article_l1398_139823


namespace NUMINAMATH_CALUDE_parabola_intersection_value_l1398_139881

theorem parabola_intersection_value (m : ℝ) : m^2 - m - 1 = 0 → m^2 - m + 2017 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_value_l1398_139881


namespace NUMINAMATH_CALUDE_root_sum_squares_l1398_139832

theorem root_sum_squares (p q r : ℝ) : 
  (p + q + r = 15) → (p * q + q * r + r * p = 25) → 
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 400 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l1398_139832


namespace NUMINAMATH_CALUDE_expression_zero_iff_x_one_or_three_l1398_139887

theorem expression_zero_iff_x_one_or_three (x : ℝ) :
  x ≠ 0 →
  (x^2 - 4*x + 3) / (5*x) = 0 ↔ x = 1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_zero_iff_x_one_or_three_l1398_139887


namespace NUMINAMATH_CALUDE_nell_baseball_cards_l1398_139890

/-- Nell's baseball card collection problem -/
theorem nell_baseball_cards :
  ∀ (initial_cards given_cards remaining_cards : ℕ),
  given_cards = 28 →
  remaining_cards = 276 →
  initial_cards = given_cards + remaining_cards →
  initial_cards = 304 :=
by
  sorry

end NUMINAMATH_CALUDE_nell_baseball_cards_l1398_139890


namespace NUMINAMATH_CALUDE_product_of_roots_l1398_139836

theorem product_of_roots (x : ℝ) : (x - 1) * (x + 4) = 22 → ∃ y : ℝ, (x - 1) * (x + 4) = 22 ∧ (y - 1) * (y + 4) = 22 ∧ x * y = -26 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1398_139836


namespace NUMINAMATH_CALUDE_track_walking_speed_l1398_139830

theorem track_walking_speed 
  (track_width : ℝ) 
  (time_difference : ℝ) 
  (inner_length : ℝ → ℝ → ℝ) 
  (outer_length : ℝ → ℝ → ℝ) :
  track_width = 6 →
  time_difference = 48 →
  (∀ a b, inner_length a b = 2 * a + 2 * π * b) →
  (∀ a b, outer_length a b = 2 * a + 2 * π * (b + track_width)) →
  ∃ s a b, 
    outer_length a b / s = inner_length a b / s + time_difference ∧
    s = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_track_walking_speed_l1398_139830


namespace NUMINAMATH_CALUDE_mod_23_equivalence_l1398_139877

theorem mod_23_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 39548 ≡ n [ZMOD 23] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_mod_23_equivalence_l1398_139877


namespace NUMINAMATH_CALUDE_checkerboard_valid_squares_l1398_139879

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  position : Nat × Nat

/-- The checkerboard -/
def Checkerboard : Type := Array (Array Bool)

/-- Creates a 10x10 checkerboard with alternating black and white squares -/
def create_checkerboard : Checkerboard := sorry

/-- Checks if a square contains at least 8 black squares -/
def has_at_least_8_black (board : Checkerboard) (square : Square) : Bool := sorry

/-- Counts the number of valid squares on the board -/
def count_valid_squares (board : Checkerboard) : Nat := sorry

theorem checkerboard_valid_squares :
  let board := create_checkerboard
  count_valid_squares board = 140 := by sorry

end NUMINAMATH_CALUDE_checkerboard_valid_squares_l1398_139879


namespace NUMINAMATH_CALUDE_parabola_line_theorem_l1398_139826

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Checks if a point lies on a given parabola -/
def isOnParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is the centroid of a triangle -/
def isCentroid (centroid : Point) (p1 p2 p3 : Point) : Prop :=
  centroid.x = (p1.x + p2.x + p3.x) / 3 ∧
  centroid.y = (p1.y + p2.y + p3.y) / 3

theorem parabola_line_theorem (parabola : Parabola) 
    (A B C F : Point) : 
    isOnParabola A parabola → 
    isOnParabola B parabola → 
    isOnParabola C parabola → 
    A.x = 1 → 
    A.y = 2 → 
    F.x = parabola.p → 
    F.y = 0 → 
    isCentroid F A B C → 
    ∃ (line : Line), 
      line.a = 2 ∧ 
      line.b = 1 ∧ 
      line.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_theorem_l1398_139826


namespace NUMINAMATH_CALUDE_pond_volume_calculation_l1398_139813

/-- The volume of a rectangular pond -/
def pond_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of a rectangular pond with dimensions 20 m x 10 m x 5 m is 1000 cubic meters -/
theorem pond_volume_calculation : pond_volume 20 10 5 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_calculation_l1398_139813


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l1398_139812

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  sampleSize : ℕ
  sampleSize_le_population : sampleSize ≤ population

/-- The probability of an individual being selected in a systematic sampling -/
def selectionProbability (s : SystematicSampling) : ℚ :=
  s.sampleSize / s.population

theorem systematic_sampling_probability 
  (s : SystematicSampling) 
  (h1 : s.population = 121) 
  (h2 : s.sampleSize = 12) : 
  selectionProbability s = 12 / 121 := by
  sorry

#check systematic_sampling_probability

end NUMINAMATH_CALUDE_systematic_sampling_probability_l1398_139812


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_of_two_unbounded_l1398_139839

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: For any positive real number M, there exists a positive integer n 
    such that the sum of the digits of 2^n is greater than M -/
theorem sum_of_digits_of_power_of_two_unbounded :
  ∀ M : ℝ, M > 0 → ∃ n : ℕ, (sumOfDigits (2^n : ℕ)) > M := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_of_two_unbounded_l1398_139839


namespace NUMINAMATH_CALUDE_n_equals_t_plus_2_l1398_139804

theorem n_equals_t_plus_2 (t : ℝ) (h : t ≠ 3) :
  let n := (4*t^2 - 10*t - 2 - 3*(t^2 - t + 3) + t^2 + 5*t - 1) / ((t + 7) + (t - 13))
  n = t + 2 := by sorry

end NUMINAMATH_CALUDE_n_equals_t_plus_2_l1398_139804


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l1398_139883

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 / 4 = 1

-- Define the line
def line (m x y : ℝ) : Prop := m * x + y + m - 1 = 0

-- Theorem statement
theorem line_intersects_ellipse (m : ℝ) : 
  ∃ (x y : ℝ), ellipse x y ∧ line m x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l1398_139883
