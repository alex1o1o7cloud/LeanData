import Mathlib

namespace advertising_department_size_l2563_256317

theorem advertising_department_size 
  (total_employees : ℕ) 
  (sample_size : ℕ) 
  (selected_from_ad : ℕ) 
  (h1 : total_employees = 1000)
  (h2 : sample_size = 80)
  (h3 : selected_from_ad = 4) :
  (selected_from_ad : ℚ) / (sample_size : ℚ) = (50 : ℚ) / (total_employees : ℚ) :=
by
  sorry

#check advertising_department_size

end advertising_department_size_l2563_256317


namespace existence_of_twin_primes_l2563_256302

theorem existence_of_twin_primes : ∃ n : ℕ, Prime n ∧ Prime (n + 2) := by
  sorry

end existence_of_twin_primes_l2563_256302


namespace number_equals_sixteen_l2563_256372

theorem number_equals_sixteen : ∃ x : ℝ, 0.0025 * x = 0.04 ∧ x = 16 := by
  sorry

end number_equals_sixteen_l2563_256372


namespace min_value_x_l2563_256390

theorem min_value_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ 2 * Real.log 3 - (1/3) * Real.log x) : x ≥ 27 := by
  sorry

end min_value_x_l2563_256390


namespace unique_solution_cubic_l2563_256399

theorem unique_solution_cubic (b : ℝ) : 
  (∃! x : ℝ, x^3 - b*x^2 - 3*b*x + b^2 - 2 = 0) ↔ b = 7/4 := by
sorry

end unique_solution_cubic_l2563_256399


namespace min_box_height_l2563_256301

/-- The minimum height of a box with a square base, where the height is 5 units more
    than the side length of the base, and the surface area is at least 120 square units. -/
theorem min_box_height (x : ℝ) (h1 : x > 0) : 
  let height := x + 5
  let surface_area := 2 * x^2 + 4 * x * height
  surface_area ≥ 120 → height ≥ 25/3 := by
  sorry

end min_box_height_l2563_256301


namespace constant_s_value_l2563_256378

theorem constant_s_value : ∃ (s : ℝ), ∀ (x : ℝ),
  (3 * x^3 - 2 * x^2 + x + 6) * (2 * x^3 + s * x^2 + 3 * x + 5) =
  6 * x^6 + s * x^5 + 5 * x^4 + 17 * x^3 + 10 * x^2 + 33 * x + 30 ∧ s = 4 := by
  sorry

end constant_s_value_l2563_256378


namespace perpendicular_line_through_circle_center_l2563_256319

/-- Given a circle with equation x^2 + 2x + y^2 = 0 and a line x + y = 0,
    prove that x - y + 1 = 0 is the equation of the line passing through
    the center of the circle and perpendicular to the given line. -/
theorem perpendicular_line_through_circle_center :
  let circle : ℝ × ℝ → Prop := λ p => p.1^2 + 2*p.1 + p.2^2 = 0
  let given_line : ℝ × ℝ → Prop := λ p => p.1 + p.2 = 0
  let perpendicular_line : ℝ × ℝ → Prop := λ p => p.1 - p.2 + 1 = 0
  let center : ℝ × ℝ := (-1, 0)
  (∀ p, circle p ↔ (p.1 + 1)^2 + p.2^2 = 1) →
  perpendicular_line center ∧
  (∀ p q : ℝ × ℝ, p ≠ q →
    given_line p ∧ given_line q →
    perpendicular_line p ∧ perpendicular_line q →
    (p.1 - q.1) * (p.1 - q.1 + q.2 - p.2) = 0) :=
by sorry

end perpendicular_line_through_circle_center_l2563_256319


namespace unique_m_satisfying_conditions_l2563_256353

theorem unique_m_satisfying_conditions : ∃! m : ℤ,
  (∃ x : ℤ, (m * x - 1) / (x - 1) = 2 + 1 / (1 - x)) ∧
  (4 - 2 * (m - 1) * (1 / 2) ≥ 0) ∧
  m ≠ 1 := by
  sorry

end unique_m_satisfying_conditions_l2563_256353


namespace consecutive_cubes_divisibility_l2563_256337

theorem consecutive_cubes_divisibility (a : ℤ) : 
  ∃ (k₁ k₂ : ℤ), 3 * a * (a^2 + 2) = 3 * a * k₁ ∧ 3 * a * (a^2 + 2) = 9 * k₂ := by
  sorry

end consecutive_cubes_divisibility_l2563_256337


namespace set_union_problem_l2563_256351

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem set_union_problem (x : ℝ) :
  (A x ∩ B x = {9}) → (A x ∪ B x = {-8, -4, 4, -7, 9}) :=
by sorry

end set_union_problem_l2563_256351


namespace exponential_inequality_l2563_256316

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a < b := by
  sorry

end exponential_inequality_l2563_256316


namespace cylinder_sphere_min_volume_l2563_256366

/-- Given a cylinder with lateral surface area 4π and an external tangent sphere,
    prove that the total surface area of the cylinder is 6π when the volume of the sphere is minimum -/
theorem cylinder_sphere_min_volume (r h : ℝ) : 
  r > 0 → h > 0 →
  2 * Real.pi * r * h = 4 * Real.pi →
  (∀ R : ℝ, R > 0 → R^2 ≥ r^2 + (h/2)^2) →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 6 * Real.pi :=
by sorry

end cylinder_sphere_min_volume_l2563_256366


namespace equation_solution_l2563_256376

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 1, 0), (1, 2, 0), (3, 4, 2), (4, 3, 2), (1, 0, 2), (0, 1, 2), (2, 4, 3), (4, 2, 3)}

theorem equation_solution :
  {(a, b, c) : ℕ × ℕ × ℕ | (c - 1) * (a * b - b - a) = a + b - 2} = solution_set :=
by sorry

end equation_solution_l2563_256376


namespace triangle_theorem_l2563_256318

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * sin t.C = t.c * cos t.B + t.c) :
  t.B = π / 3 ∧ 
  (t.b ^ 2 = t.a * t.c → 1 / tan t.A + 1 / tan t.C = 2 * Real.sqrt 3 / 3) :=
sorry

end triangle_theorem_l2563_256318


namespace dogs_barking_l2563_256346

theorem dogs_barking (initial_dogs : ℕ) (additional_dogs : ℕ) :
  initial_dogs = 30 →
  additional_dogs = 10 →
  initial_dogs + additional_dogs = 40 := by
sorry

end dogs_barking_l2563_256346


namespace max_radius_third_jar_l2563_256328

theorem max_radius_third_jar (pot_diameter : ℝ) (jar1_radius : ℝ) (jar2_radius : ℝ) :
  pot_diameter = 36 →
  jar1_radius = 6 →
  jar2_radius = 12 →
  ∃ (max_radius : ℝ),
    max_radius = 36 / 7 ∧
    ∀ (r : ℝ), r > max_radius →
      ¬ (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
        (x1^2 + y1^2 ≤ (pot_diameter/2)^2) ∧
        (x2^2 + y2^2 ≤ (pot_diameter/2)^2) ∧
        (x3^2 + y3^2 ≤ (pot_diameter/2)^2) ∧
        ((x1 - x2)^2 + (y1 - y2)^2 ≥ (jar1_radius + jar2_radius)^2) ∧
        ((x1 - x3)^2 + (y1 - y3)^2 ≥ (jar1_radius + r)^2) ∧
        ((x2 - x3)^2 + (y2 - y3)^2 ≥ (jar2_radius + r)^2)) :=
by
  sorry


end max_radius_third_jar_l2563_256328


namespace acute_angle_cosine_difference_l2563_256341

theorem acute_angle_cosine_difference (α : Real) : 
  0 < α → α < π / 2 →  -- acute angle condition
  3 * Real.sin α = Real.tan α →  -- given equation
  Real.cos (α - π / 4) = (4 + Real.sqrt 2) / 6 := by
  sorry

end acute_angle_cosine_difference_l2563_256341


namespace group_size_problem_l2563_256338

theorem group_size_problem (T : ℕ) (L : ℕ) : 
  T > 90 →  -- Total number of people is greater than 90
  L = T - 90 →  -- Number of people under 20 is the total minus 90
  (L : ℚ) / T = 2/5 →  -- Probability of selecting someone under 20 is 0.4
  T = 150 :=
by
  sorry

end group_size_problem_l2563_256338


namespace least_four_digit_multiple_l2563_256342

theorem least_four_digit_multiple : ∀ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) → -- four-digit positive integer
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) → -- divisible by 3, 5, and 7
  1050 ≤ n := by
  sorry

#check least_four_digit_multiple

end least_four_digit_multiple_l2563_256342


namespace second_train_speed_l2563_256360

/-- Calculates the speed of the second train given the parameters of two trains meeting. -/
theorem second_train_speed
  (length1 : ℝ) (length2 : ℝ) (speed1 : ℝ) (clear_time : ℝ)
  (h1 : length1 = 120) -- Length of first train in meters
  (h2 : length2 = 280) -- Length of second train in meters
  (h3 : speed1 = 42) -- Speed of first train in kmph
  (h4 : clear_time = 20 / 3600) -- Time to clear in hours
  : ∃ (speed2 : ℝ), speed2 = 30 := by
  sorry

end second_train_speed_l2563_256360


namespace ellipse_m_value_l2563_256395

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop := x^2 / m + y^2 / 16 = 1

-- Define the distances from a point to the foci
def distance_to_foci (d1 d2 : ℝ) : Prop := d1 = 3 ∧ d2 = 7

-- Theorem statement
theorem ellipse_m_value (x y m : ℝ) :
  ellipse_equation x y m →
  ∃ (d1 d2 : ℝ), distance_to_foci d1 d2 →
  m = 25 := by
sorry

end ellipse_m_value_l2563_256395


namespace exercise_books_count_l2563_256371

/-- Given a shop with pencils, pens, and exercise books in the ratio 14 : 4 : 3,
    and 140 pencils, prove that there are 30 exercise books. -/
theorem exercise_books_count (pencils : ℕ) (pens : ℕ) (books : ℕ) : 
  pencils = 140 →
  pencils / 14 = pens / 4 →
  pencils / 14 = books / 3 →
  books = 30 := by
sorry

end exercise_books_count_l2563_256371


namespace positive_solution_of_equation_l2563_256384

theorem positive_solution_of_equation : ∃ (x : ℝ), 
  x > 0 ∧ 
  (1/3) * (4*x^2 - 1) = (x^2 - 60*x - 12) * (x^2 + 30*x + 6) ∧ 
  x = 30 + 2 * Real.sqrt 231 := by
  sorry

end positive_solution_of_equation_l2563_256384


namespace remainder_problem_l2563_256308

theorem remainder_problem (N : ℤ) : 
  ∃ k : ℤ, N = 35 * k + 25 → ∃ m : ℤ, N = 15 * m + 10 := by
sorry

end remainder_problem_l2563_256308


namespace power_of_fraction_five_sixths_fourth_l2563_256388

theorem power_of_fraction_five_sixths_fourth : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by
  sorry

end power_of_fraction_five_sixths_fourth_l2563_256388


namespace min_room_dimensions_l2563_256321

/-- The minimum dimensions of a rectangular room that can accommodate a 9' × 12' table --/
theorem min_room_dimensions (table_width : ℝ) (table_length : ℝ) 
  (hw : table_width = 9) (hl : table_length = 12) :
  ∃ (S T : ℝ), 
    S > T ∧ 
    S ≥ Real.sqrt (table_width^2 + table_length^2) ∧
    T ≥ max table_width table_length ∧
    ∀ (S' T' : ℝ), (S' > T' ∧ 
                    S' ≥ Real.sqrt (table_width^2 + table_length^2) ∧ 
                    T' ≥ max table_width table_length) → 
                    (S ≤ S' ∧ T ≤ T') ∧
    S = 15 ∧ T = 12 :=
by sorry

end min_room_dimensions_l2563_256321


namespace pythagorean_cube_equation_solutions_l2563_256312

theorem pythagorean_cube_equation_solutions :
  ∀ a b c : ℕ+,
    a^2 + b^2 = c^2 ∧ a^3 + b^3 + 1 = (c - 1)^3 →
    ((a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10)) :=
by sorry

end pythagorean_cube_equation_solutions_l2563_256312


namespace logo_shaded_area_l2563_256307

/-- Represents a logo with specific geometric properties. -/
structure Logo where
  /-- Length of vertical straight edges and diameters of small semicircles -/
  edge_length : ℝ
  /-- Rotational symmetry property -/
  has_rotational_symmetry : Prop

/-- Calculates the shaded area of the logo -/
def shaded_area (logo : Logo) : ℝ :=
  sorry

/-- Theorem stating that the shaded area of a logo with specific properties is 4 + π -/
theorem logo_shaded_area (logo : Logo) 
  (h1 : logo.edge_length = 2)
  (h2 : logo.has_rotational_symmetry) : 
  shaded_area logo = 4 + π := by
  sorry

end logo_shaded_area_l2563_256307


namespace polynomial_simplification_l2563_256391

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 5 * x + 9) * (x + 2) - (x + 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x + 2) * (x + 4) =
  6 * x^3 - 28 * x^2 - 59 * x + 42 := by
  sorry

end polynomial_simplification_l2563_256391


namespace max_sum_of_logs_l2563_256363

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem max_sum_of_logs (x y : ℝ) (h1 : x + y = 4) (h2 : x > 0) (h3 : y > 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 → lg a + lg b ≤ lg 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 4 ∧ lg a + lg b = lg 4) :=
sorry

end max_sum_of_logs_l2563_256363


namespace partial_fraction_decomposition_l2563_256358

theorem partial_fraction_decomposition :
  ∃ (P Q : ℚ), P = 22/9 ∧ Q = -4/9 ∧
  ∀ (x : ℚ), x ≠ 7 ∧ x ≠ -2 →
    (2*x + 8) / (x^2 - 5*x - 14) = P / (x - 7) + Q / (x + 2) := by
  sorry

end partial_fraction_decomposition_l2563_256358


namespace residue_mod_14_l2563_256393

theorem residue_mod_14 : (320 * 16 - 28 * 5 + 7) % 14 = 3 := by
  sorry

end residue_mod_14_l2563_256393


namespace fraction_equality_l2563_256357

theorem fraction_equality (x y : ℝ) (h : x / 2 = y / 5) : x / y = 2 / 5 := by
  sorry

end fraction_equality_l2563_256357


namespace income_percentage_difference_l2563_256383

/-- Given the monthly incomes of A, B, and C, prove that B's income is 12% more than C's. -/
theorem income_percentage_difference :
  ∀ (A_annual B_monthly C_monthly : ℝ),
  A_annual = 436800.0000000001 →
  C_monthly = 13000 →
  A_annual / 12 / B_monthly = 5 / 2 →
  (B_monthly - C_monthly) / C_monthly = 0.12 :=
by
  sorry

end income_percentage_difference_l2563_256383


namespace pond_length_l2563_256306

/-- Given a rectangular field with length 20 m and width 10 m, containing a square pond
    whose area is 1/8 of the field's area, the length of the pond is 5 m. -/
theorem pond_length (field_length field_width pond_area : ℝ) : 
  field_length = 20 →
  field_width = 10 →
  field_length = 2 * field_width →
  pond_area = (1 / 8) * (field_length * field_width) →
  Real.sqrt pond_area = 5 := by
  sorry


end pond_length_l2563_256306


namespace triangle_area_theorem_l2563_256333

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  angleB : ℝ

-- Define the theorem
theorem triangle_area_theorem (t : Triangle) (h1 : t.a = Real.sqrt 3) (h2 : t.b = 1) (h3 : t.angleB = π / 6) :
  ∃ (S : ℝ), (S = Real.sqrt 3 / 2 ∨ S = Real.sqrt 3 / 4) ∧ 
  (∃ (angleA angleC : ℝ), 
    angleA + t.angleB + angleC = π ∧
    S = 1/2 * t.a * t.b * Real.sin angleC) :=
sorry

end triangle_area_theorem_l2563_256333


namespace green_beans_count_l2563_256350

def total_beans : ℕ := 572

def red_beans : ℕ := total_beans / 4

def remaining_after_red : ℕ := total_beans - red_beans

def white_beans : ℕ := remaining_after_red / 3

def remaining_after_white : ℕ := remaining_after_red - white_beans

def blue_beans : ℕ := remaining_after_white / 5

def remaining_after_blue : ℕ := remaining_after_white - blue_beans

def yellow_beans : ℕ := remaining_after_blue / 6

def remaining_after_yellow : ℕ := remaining_after_blue - yellow_beans

def green_beans : ℕ := remaining_after_yellow / 2

theorem green_beans_count : green_beans = 95 := by
  sorry

end green_beans_count_l2563_256350


namespace choose_and_assign_roles_l2563_256347

/-- The number of members in the group -/
def group_size : ℕ := 4

/-- The number of roles to be assigned -/
def roles_count : ℕ := 3

/-- The number of ways to choose and assign roles -/
def ways_to_choose_and_assign : ℕ := group_size * (group_size - 1) * (group_size - 2)

theorem choose_and_assign_roles :
  ways_to_choose_and_assign = 24 :=
sorry

end choose_and_assign_roles_l2563_256347


namespace equal_selection_probability_all_students_equal_probability_l2563_256397

/-- Represents the probability of a student being selected -/
def selection_probability (total_students : ℕ) (eliminated : ℕ) (selected : ℕ) : ℚ :=
  selected / (total_students - eliminated)

/-- The selection method results in equal probability for all students -/
theorem equal_selection_probability 
  (total_students : ℕ) 
  (eliminated : ℕ) 
  (selected : ℕ) 
  (h1 : total_students = 2004) 
  (h2 : eliminated = 4) 
  (h3 : selected = 50) :
  selection_probability total_students eliminated selected = 1 / 40 :=
sorry

/-- The probability of selection is the same for all students -/
theorem all_students_equal_probability 
  (student1 student2 : ℕ) 
  (h_student1 : student1 ≤ 2004) 
  (h_student2 : student2 ≤ 2004) :
  selection_probability 2004 4 50 = selection_probability 2004 4 50 :=
sorry

end equal_selection_probability_all_students_equal_probability_l2563_256397


namespace determinant_problems_l2563_256324

def matrix1 : Matrix (Fin 3) (Fin 3) ℤ := !![3, 2, 1; 2, 5, 3; 3, 4, 3]

def matrix2 (a b c : ℤ) : Matrix (Fin 3) (Fin 3) ℤ := !![a, b, c; b, c, a; c, a, b]

theorem determinant_problems :
  (Matrix.det matrix1 = 8) ∧
  (∀ a b c : ℤ, Matrix.det (matrix2 a b c) = 3 * a * b * c - a^3 - b^3 - c^3) := by
  sorry

end determinant_problems_l2563_256324


namespace one_root_of_sum_equation_l2563_256323

/-- A reduced quadratic trinomial with two distinct roots -/
structure ReducedQuadraticTrinomial where
  b : ℝ
  c : ℝ
  has_distinct_roots : b^2 - 4*c > 0

/-- The discriminant of a quadratic trinomial -/
def discriminant (f : ReducedQuadraticTrinomial) : ℝ := f.b^2 - 4*f.c

/-- The quadratic function corresponding to a ReducedQuadraticTrinomial -/
def quad_function (f : ReducedQuadraticTrinomial) (x : ℝ) : ℝ := x^2 + f.b * x + f.c

/-- The theorem stating that f(x) + f(x - √D) = 0 has exactly one root -/
theorem one_root_of_sum_equation (f : ReducedQuadraticTrinomial) :
  ∃! x : ℝ, quad_function f x + quad_function f (x - Real.sqrt (discriminant f)) = 0 :=
sorry

end one_root_of_sum_equation_l2563_256323


namespace line_classification_l2563_256331

-- Define the coordinate plane
def CoordinatePlane : Type := ℝ × ℝ

-- Define an integer point
def IntegerPoint (p : CoordinatePlane) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

-- Define a line on the coordinate plane
def Line : Type := CoordinatePlane → Prop

-- Define set I as the set of all lines
def I : Set Line := Set.univ

-- Define set M as the set of lines passing through exactly one integer point
def M : Set Line :=
  {l : Line | ∃! (p : CoordinatePlane), IntegerPoint p ∧ l p}

-- Define set N as the set of lines passing through no integer points
def N : Set Line :=
  {l : Line | ∀ (p : CoordinatePlane), l p → ¬IntegerPoint p}

-- Define set P as the set of lines passing through infinitely many integer points
def P : Set Line :=
  {l : Line | ∀ (n : ℕ), ∃ (S : Finset CoordinatePlane),
    Finset.card S = n ∧ (∀ (p : CoordinatePlane), p ∈ S → IntegerPoint p ∧ l p)}

theorem line_classification :
  (M ∪ N ∪ P = I) ∧ (N ≠ ∅) ∧ (M ≠ ∅) ∧ (P ≠ ∅) := by sorry

end line_classification_l2563_256331


namespace problem_solution_l2563_256303

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}

theorem problem_solution :
  (A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8}) ∧
  (∀ a : ℝ, A ∩ C a = ∅ ↔ a ≥ 8) := by
  sorry

end problem_solution_l2563_256303


namespace soccer_team_physics_count_l2563_256356

theorem soccer_team_physics_count (total : ℕ) (math : ℕ) (both : ℕ) (physics : ℕ) : 
  total = 15 → 
  math = 10 → 
  both = 4 → 
  math + physics - both = total → 
  physics = 9 := by
sorry

end soccer_team_physics_count_l2563_256356


namespace work_time_relation_l2563_256396

/-- Given three workers A, B, and C, where:
    - A takes m times as long to do a piece of work as B and C together
    - B takes n times as long as C and A together
    - C takes x times as long as A and B together
    This theorem proves that x = (m + n + 2) / (mn - 1) -/
theorem work_time_relation (m n x : ℝ) (hm : m > 0) (hn : n > 0) (hx : x > 0)
  (hA : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a = m * (1/(b+c)))
  (hB : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/b = n * (1/(a+c)))
  (hC : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/c = x * (1/(a+b))) :
  x = (m + n + 2) / (m * n - 1) :=
by sorry

end work_time_relation_l2563_256396


namespace evaluate_expression_l2563_256309

theorem evaluate_expression : 
  3999^3 - 2 * 3998 * 3999^2 - 2 * 3998^2 * 3999 + 3997^3 = 95806315 := by
  sorry

end evaluate_expression_l2563_256309


namespace janet_weekly_income_l2563_256373

/-- Calculates the total income for Janet's exterminator work and sculpture sales --/
def janet_total_income (small_res_rate : ℕ) (large_res_rate : ℕ) (commercial_rate : ℕ)
                       (small_res_hours : ℕ) (large_res_hours : ℕ) (commercial_hours : ℕ)
                       (small_sculpture_price : ℕ) (medium_sculpture_price : ℕ) (large_sculpture_price : ℕ)
                       (small_sculpture_weight : ℕ) (small_sculpture_count : ℕ)
                       (medium_sculpture_weight : ℕ) (medium_sculpture_count : ℕ)
                       (large_sculpture_weight : ℕ) (large_sculpture_count : ℕ) : ℕ :=
  let exterminator_income := small_res_rate * small_res_hours + 
                             large_res_rate * large_res_hours + 
                             commercial_rate * commercial_hours
  let sculpture_income := small_sculpture_price * small_sculpture_weight * small_sculpture_count +
                          medium_sculpture_price * medium_sculpture_weight * medium_sculpture_count +
                          large_sculpture_price * large_sculpture_weight * large_sculpture_count
  exterminator_income + sculpture_income

/-- Theorem stating that Janet's total income for the week is $2320 --/
theorem janet_weekly_income : 
  janet_total_income 70 85 100 10 5 5 20 25 30 4 2 7 1 12 1 = 2320 := by
  sorry

end janet_weekly_income_l2563_256373


namespace right_triangle_hypotenuse_l2563_256359

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 ∧ b = 36 ∧ c^2 = a^2 + b^2 → c = 39 :=
by sorry

end right_triangle_hypotenuse_l2563_256359


namespace impossibleTransformation_l2563_256300

-- Define the button colors
inductive Color
| A
| B
| C

-- Define the configuration as a list of colors
def Configuration := List Color

-- Define the card values
inductive CardValue
| One
| NegOne
| Zero

-- Function to calculate the card value between two adjacent colors
def getCardValue (c1 c2 : Color) : CardValue :=
  match c1, c2 with
  | Color.B, Color.A => CardValue.One
  | Color.A, Color.C => CardValue.One
  | Color.A, Color.B => CardValue.NegOne
  | Color.C, Color.A => CardValue.NegOne
  | _, _ => CardValue.Zero

-- Function to calculate the sum of card values for a configuration
def sumCardValues (config : Configuration) : Int :=
  let pairs := List.zip config (config.rotateLeft 1)
  let cardValues := pairs.map (fun (c1, c2) => getCardValue c1 c2)
  cardValues.foldl (fun sum cv => 
    sum + match cv with
    | CardValue.One => 1
    | CardValue.NegOne => -1
    | CardValue.Zero => 0
  ) 0

-- Define the initial and final configurations
def initialConfig : Configuration := [Color.A, Color.C, Color.B, Color.C, Color.B]
def finalConfig : Configuration := [Color.A, Color.B, Color.C, Color.B, Color.C]

-- Theorem: It's impossible to transform the initial configuration to the final configuration
theorem impossibleTransformation : 
  ∀ (swapSequence : List (Configuration → Configuration)),
  (∀ (config : Configuration), sumCardValues config = sumCardValues (swapSequence.foldl (fun c f => f c) config)) →
  swapSequence.foldl (fun c f => f c) initialConfig ≠ finalConfig :=
sorry

end impossibleTransformation_l2563_256300


namespace beth_sold_coins_l2563_256367

theorem beth_sold_coins (initial_coins carl_gift kept_coins : ℕ) 
  (h1 : initial_coins = 250)
  (h2 : carl_gift = 75)
  (h3 : kept_coins = 135) :
  initial_coins + carl_gift - kept_coins = 190 :=
by
  sorry

end beth_sold_coins_l2563_256367


namespace unique_natural_number_with_special_division_property_l2563_256327

theorem unique_natural_number_with_special_division_property :
  ∃! (n : ℕ), ∃ (a b : ℕ),
    n = 12 * b + a ∧
    n = 10 * a + b ∧
    a ≤ 11 ∧
    b ≤ 9 ∧
    n = 119 := by
  sorry

end unique_natural_number_with_special_division_property_l2563_256327


namespace output_for_three_l2563_256329

def f (a : ℤ) : ℤ :=
  if a < 10 then 2 * a else a + 1

theorem output_for_three :
  f 3 = 6 :=
by sorry

end output_for_three_l2563_256329


namespace football_lineup_combinations_l2563_256332

theorem football_lineup_combinations (total_players : ℕ) 
  (offensive_linemen : ℕ) (running_backs : ℕ) : 
  total_players = 12 → offensive_linemen = 3 → running_backs = 4 →
  (offensive_linemen * running_backs * (total_players - 2) * (total_players - 3) = 1080) := by
  sorry

end football_lineup_combinations_l2563_256332


namespace function_properties_l2563_256345

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem function_properties (a : ℝ) :
  (∀ x, x < -1 ∨ x > 3 → ∀ y, y > x → f y a < f x a) ∧
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y a ≤ f x a) ∧
  (f 2 a = 20) →
  a = -2 ∧
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x a ≤ f y a ∧ f x a = -7) :=
by sorry

#check function_properties

end function_properties_l2563_256345


namespace parallelogram_base_length_l2563_256349

/-- Given a parallelogram with height 6 meters and area 72 square meters, its base is 12 meters. -/
theorem parallelogram_base_length (height : ℝ) (area : ℝ) (base : ℝ) : 
  height = 6 → area = 72 → area = base * height → base = 12 := by sorry

end parallelogram_base_length_l2563_256349


namespace max_food_per_guest_l2563_256380

theorem max_food_per_guest (total_food : ℝ) (min_guests : ℕ) 
  (h1 : total_food = 411)
  (h2 : min_guests = 165) :
  total_food / min_guests = 411 / 165 := by
  sorry

end max_food_per_guest_l2563_256380


namespace problem_trip_mpg_l2563_256377

/-- Represents a car trip with odometer readings and gas fill amounts -/
structure CarTrip where
  initial_odometer : ℕ
  final_odometer : ℕ
  gas_fills : List ℕ

/-- Calculates the average miles per gallon for a car trip -/
def averageMPG (trip : CarTrip) : ℚ :=
  let total_distance := trip.final_odometer - trip.initial_odometer
  let total_gas := trip.gas_fills.sum
  (total_distance : ℚ) / total_gas

/-- The specific car trip from the problem -/
def problemTrip : CarTrip := {
  initial_odometer := 68300
  final_odometer := 69600
  gas_fills := [15, 25]
}

/-- Theorem stating that the average MPG for the problem trip is 32.5 -/
theorem problem_trip_mpg : averageMPG problemTrip = 32.5 := by
  sorry


end problem_trip_mpg_l2563_256377


namespace remainder_problem_l2563_256370

theorem remainder_problem (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 39 = 15 := by
  sorry

end remainder_problem_l2563_256370


namespace tangent_equation_solution_l2563_256322

theorem tangent_equation_solution :
  ∃! x : Real, 0 ≤ x ∧ x ≤ 180 ∧
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 30 := by
sorry

end tangent_equation_solution_l2563_256322


namespace kyle_age_l2563_256398

def age_problem (casey shelley kyle julian frederick tyson : ℕ) : Prop :=
  (shelley + 3 = kyle) ∧
  (shelley = julian + 4) ∧
  (julian + 20 = frederick) ∧
  (frederick = 2 * tyson) ∧
  (tyson = 2 * casey) ∧
  (casey = 15)

theorem kyle_age :
  ∀ casey shelley kyle julian frederick tyson : ℕ,
  age_problem casey shelley kyle julian frederick tyson →
  kyle = 47 :=
by
  sorry

end kyle_age_l2563_256398


namespace fish_lost_calculation_l2563_256379

/-- The number of fish Alex lost back to the lake in the fishing tournament -/
def fish_lost : ℕ := by sorry

theorem fish_lost_calculation (jacob_initial : ℕ) (alex_multiplier : ℕ) (jacob_additional : ℕ) :
  jacob_initial = 8 →
  alex_multiplier = 7 →
  jacob_additional = 26 →
  fish_lost = alex_multiplier * jacob_initial - (jacob_initial + jacob_additional + 1) := by sorry

end fish_lost_calculation_l2563_256379


namespace conic_intersection_lines_concurrent_l2563_256335

-- Define the type for a conic
def Conic := Type

-- Define the type for a point
def Point := Type

-- Define the type for a line
def Line := Type

-- Define a function to check if a point is on a conic
def point_on_conic (p : Point) (c : Conic) : Prop := sorry

-- Define a function to create a line from two points
def line_through_points (p q : Point) : Line := sorry

-- Define a function to check if three lines are concurrent
def are_concurrent (l₁ l₂ l₃ : Line) : Prop := sorry

-- Define the theorem
theorem conic_intersection_lines_concurrent 
  (𝓔₁ 𝓔₂ 𝓔₃ : Conic) 
  (A B : Point) 
  (h_common : point_on_conic A 𝓔₁ ∧ point_on_conic A 𝓔₂ ∧ point_on_conic A 𝓔₃ ∧
              point_on_conic B 𝓔₁ ∧ point_on_conic B 𝓔₂ ∧ point_on_conic B 𝓔₃)
  (C D E F G H : Point)
  (h_intersections : point_on_conic C 𝓔₁ ∧ point_on_conic C 𝓔₂ ∧
                     point_on_conic D 𝓔₁ ∧ point_on_conic D 𝓔₂ ∧
                     point_on_conic E 𝓔₁ ∧ point_on_conic E 𝓔₃ ∧
                     point_on_conic F 𝓔₁ ∧ point_on_conic F 𝓔₃ ∧
                     point_on_conic G 𝓔₂ ∧ point_on_conic G 𝓔₃ ∧
                     point_on_conic H 𝓔₂ ∧ point_on_conic H 𝓔₃)
  (ℓ₁₂ := line_through_points C D)
  (ℓ₁₃ := line_through_points E F)
  (ℓ₂₃ := line_through_points G H) :
  are_concurrent ℓ₁₂ ℓ₁₃ ℓ₂₃ := by
  sorry

end conic_intersection_lines_concurrent_l2563_256335


namespace married_couple_survival_probability_l2563_256365

/-- The probability problem for a married couple's survival over 10 years -/
theorem married_couple_survival_probability 
  (p_man : ℝ) 
  (p_neither : ℝ) 
  (h_man : p_man = 1/4) 
  (h_neither : p_neither = 1/2) : 
  ∃ p_wife : ℝ, 
    p_wife = 1/3 ∧ 
    p_neither = 1 - (p_man + p_wife - p_man * p_wife) := by
  sorry

end married_couple_survival_probability_l2563_256365


namespace max_n_for_300_triangles_max_n_is_102_l2563_256381

/-- Represents a convex polygon with interior points -/
structure ConvexPolygon where
  n : ℕ  -- number of vertices in the polygon
  interior_points : ℕ -- number of interior points
  no_collinear : Prop -- property that no three points are collinear

/-- The number of triangles formed in a convex polygon with interior points -/
def num_triangles (p : ConvexPolygon) : ℕ :=
  p.n + p.interior_points + 198

/-- Theorem stating the maximum value of n for which no more than 300 triangles can be formed -/
theorem max_n_for_300_triangles (p : ConvexPolygon) 
  (h1 : p.interior_points = 100) 
  (h2 : num_triangles p ≤ 300) : 
  p.n ≤ 102 := by
  sorry

/-- The maximum value of n is indeed 102 -/
theorem max_n_is_102 (p : ConvexPolygon) 
  (h1 : p.interior_points = 100) 
  (h2 : num_triangles p ≤ 300) : 
  ∃ (q : ConvexPolygon), q.n = 102 ∧ q.interior_points = 100 ∧ num_triangles q = 300 := by
  sorry

end max_n_for_300_triangles_max_n_is_102_l2563_256381


namespace brett_marbles_difference_l2563_256382

/-- The number of red marbles Brett has -/
def red_marbles : ℕ := 6

/-- The number of blue marbles Brett has -/
def blue_marbles : ℕ := 5 * red_marbles

/-- The difference between blue and red marbles -/
def marble_difference : ℕ := blue_marbles - red_marbles

theorem brett_marbles_difference : marble_difference = 24 := by
  sorry

end brett_marbles_difference_l2563_256382


namespace arithmetic_geometric_mean_inequality_l2563_256364

theorem arithmetic_geometric_mean_inequality 
  (a b c d x y : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_arithmetic : b - a = c - b ∧ c - b = d - c) 
  (h_x : x = (a + d) / 2) 
  (h_y : y = Real.sqrt (b * c)) : 
  x ≥ y := by
sorry

end arithmetic_geometric_mean_inequality_l2563_256364


namespace gcd_count_l2563_256326

def count_gcd_values (a b : ℕ) : Prop :=
  (Nat.gcd a b * Nat.lcm a b = 360) →
  (∃ (S : Finset ℕ), (∀ x ∈ S, ∃ (c d : ℕ), Nat.gcd c d = x ∧ Nat.gcd c d * Nat.lcm c d = 360) ∧
                     (∀ y, (∃ (e f : ℕ), Nat.gcd e f = y ∧ Nat.gcd e f * Nat.lcm e f = 360) → y ∈ S) ∧
                     S.card = 14)

theorem gcd_count : ∀ a b : ℕ, count_gcd_values a b :=
sorry

end gcd_count_l2563_256326


namespace simple_interest_problem_l2563_256348

theorem simple_interest_problem (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2000 → P = 2500 := by
sorry

end simple_interest_problem_l2563_256348


namespace intersection_union_sets_l2563_256320

theorem intersection_union_sets : 
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {1, 2, 3, 4}
  let P : Set ℕ := {2, 3, 4, 5}
  (M ∩ N) ∪ P = {1, 2, 3, 4, 5} := by
  sorry

end intersection_union_sets_l2563_256320


namespace candy_jar_problem_l2563_256315

theorem candy_jar_problem (total : ℕ) (blue : ℕ) (red : ℕ) : 
  total = 3409 → 
  blue = 3264 → 
  total = red + blue → 
  red = 145 := by
sorry

end candy_jar_problem_l2563_256315


namespace inequality_theorem_largest_constant_equality_condition_l2563_256339

theorem inequality_theorem (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) :=
by sorry

theorem largest_constant :
  ∀ C : ℝ, (∀ x₁ x₂ x₃ x₄ x₅ x₆ : ℝ, (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ C * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂))) → C ≤ 3 :=
by sorry

theorem equality_condition (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 = 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) ↔
  x₁ + x₄ = x₂ + x₅ ∧ x₂ + x₅ = x₃ + x₆ :=
by sorry

end inequality_theorem_largest_constant_equality_condition_l2563_256339


namespace pizza_payment_difference_l2563_256313

theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let plain_pizza_cost : ℚ := 12
  let mushroom_cost : ℚ := 3
  let bob_mushroom_slices : ℕ := total_slices / 3
  let bob_plain_slices : ℕ := 3
  let alice_slices : ℕ := total_slices - (bob_mushroom_slices + bob_plain_slices)
  let total_cost : ℚ := plain_pizza_cost + mushroom_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let bob_payment : ℚ := (bob_mushroom_slices + bob_plain_slices) * cost_per_slice
  let alice_payment : ℚ := alice_slices * (plain_pizza_cost / total_slices)
  bob_payment - alice_payment = 3.75 := by sorry

end pizza_payment_difference_l2563_256313


namespace larry_channels_l2563_256374

/-- The number of channels Larry has after all changes --/
def final_channels (initial : ℕ) (removed1 removed2 added1 added2 added3 : ℕ) : ℕ :=
  initial - removed1 + added1 - removed2 + added2 + added3

/-- Theorem stating that Larry's final number of channels is 147 --/
theorem larry_channels : 
  final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end larry_channels_l2563_256374


namespace cube_root_sum_l2563_256352

theorem cube_root_sum (a : ℝ) (h : a^3 = 7) :
  (0.007 : ℝ)^(1/3) + 7000^(1/3) = 10.1 * a := by sorry

end cube_root_sum_l2563_256352


namespace sin_EAF_value_l2563_256343

/-- A rectangle ABCD with E and F trisecting CD -/
structure RectangleWithTrisection where
  /-- Point A of the rectangle -/
  A : ℝ × ℝ
  /-- Point B of the rectangle -/
  B : ℝ × ℝ
  /-- Point C of the rectangle -/
  C : ℝ × ℝ
  /-- Point D of the rectangle -/
  D : ℝ × ℝ
  /-- Point E trisecting CD -/
  E : ℝ × ℝ
  /-- Point F trisecting CD -/
  F : ℝ × ℝ
  /-- ABCD is a rectangle -/
  is_rectangle : (A.1 = D.1) ∧ (B.1 = C.1) ∧ (A.2 = B.2) ∧ (C.2 = D.2)
  /-- AB = 8 -/
  AB_length : (B.1 - A.1) = 8
  /-- BC = 6 -/
  BC_length : (B.2 - C.2) = 6
  /-- E and F trisect CD -/
  trisection : (E.1 - C.1) = (2/3) * (D.1 - C.1) ∧ (F.1 - C.1) = (1/3) * (D.1 - C.1)

/-- The sine of angle EAF in the given rectangle with trisection -/
def sin_EAF (r : RectangleWithTrisection) : ℝ :=
  sorry

/-- Theorem stating that sin ∠EAF = 12√13 / 194 -/
theorem sin_EAF_value (r : RectangleWithTrisection) : 
  sin_EAF r = 12 * Real.sqrt 13 / 194 :=
sorry

end sin_EAF_value_l2563_256343


namespace euler_formula_third_quadrant_l2563_256355

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- Define the third quadrant
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- State the theorem
theorem euler_formula_third_quadrant (θ : ℝ) (k : ℤ) :
  (2 * k * Real.pi + Real.pi / 2 < θ) ∧ (θ ≤ 2 * k * Real.pi + 2 * Real.pi / 3) →
  third_quadrant (cexp (2 * θ * Complex.I)) :=
sorry

end euler_formula_third_quadrant_l2563_256355


namespace average_weight_decrease_l2563_256354

/-- Calculates the decrease in average weight when a new person is added to a group --/
theorem average_weight_decrease (initial_count : ℕ) (initial_average : ℝ) (new_weight : ℝ) :
  initial_count = 20 →
  initial_average = 60 →
  new_weight = 45 →
  let total_weight := initial_count * initial_average
  let new_total_weight := total_weight + new_weight
  let new_count := initial_count + 1
  let new_average := new_total_weight / new_count
  abs (initial_average - new_average - 0.71) < 0.01 := by
  sorry

end average_weight_decrease_l2563_256354


namespace writer_average_speed_l2563_256394

/-- Calculates the average writing speed given the words and hours for two writing sessions -/
def average_writing_speed (words1 : ℕ) (hours1 : ℕ) (words2 : ℕ) (hours2 : ℕ) : ℚ :=
  (words1 + words2 : ℚ) / (hours1 + hours2 : ℚ)

/-- Theorem stating that the average writing speed for the given sessions is 500 words per hour -/
theorem writer_average_speed :
  average_writing_speed 30000 60 50000 100 = 500 := by
  sorry

end writer_average_speed_l2563_256394


namespace simplify_expression_l2563_256387

theorem simplify_expression : 
  2 - (2 / (1 + Real.sqrt 2)) - (2 / (1 - Real.sqrt 2)) = -2 := by
  sorry

end simplify_expression_l2563_256387


namespace min_average_books_borrowed_l2563_256334

theorem min_average_books_borrowed (total_students : ℕ) 
  (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 25)
  (h2 : zero_books = 2)
  (h3 : one_book = 12)
  (h4 : two_books = 4)
  (h5 : zero_books + one_book + two_books < total_students) :
  let remaining_students := total_students - (zero_books + one_book + two_books)
  let min_total_books := one_book * 1 + two_books * 2 + remaining_students * 3
  (min_total_books : ℚ) / total_students ≥ 1.64 := by
  sorry

end min_average_books_borrowed_l2563_256334


namespace cube_volume_from_surface_area_l2563_256392

/-- Given a cube with surface area 150 square units, its volume is 125 cubic units. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ, s > 0 → 6 * s^2 = 150 → s^3 = 125 := by
  sorry

end cube_volume_from_surface_area_l2563_256392


namespace limit_alternating_log_infinity_l2563_256304

/-- The limit of (-1)^n * log(n) as n approaches infinity is infinity. -/
theorem limit_alternating_log_infinity :
  ∀ M : ℝ, M > 0 → ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |(-1:ℝ)^n * Real.log n| > M :=
sorry

end limit_alternating_log_infinity_l2563_256304


namespace carpet_dimensions_l2563_256375

/-- Represents a rectangular carpet with integral side lengths -/
structure Carpet where
  width : ℕ
  length : ℕ

/-- Represents a rectangular room -/
structure Room where
  width : ℕ
  length : ℕ

/-- Checks if a carpet fits perfectly in a room (diagonally) -/
def fitsInRoom (c : Carpet) (r : Room) : Prop :=
  c.width ^ 2 + c.length ^ 2 = r.width ^ 2 + r.length ^ 2

theorem carpet_dimensions :
  ∀ (c : Carpet) (r1 r2 : Room),
    r1.width = 38 →
    r2.width = 50 →
    r1.length = r2.length →
    fitsInRoom c r1 →
    fitsInRoom c r2 →
    c.width = 25 ∧ c.length = 50 := by
  sorry


end carpet_dimensions_l2563_256375


namespace alice_and_bob_money_l2563_256314

theorem alice_and_bob_money : (5 : ℚ) / 8 + (3 : ℚ) / 5 = 1.225 := by sorry

end alice_and_bob_money_l2563_256314


namespace matrix_power_four_l2563_256310

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; -1, 0]

theorem matrix_power_four : A^4 = !![(-1 : ℤ), (-1 : ℤ); (1 : ℤ), (0 : ℤ)] := by sorry

end matrix_power_four_l2563_256310


namespace desks_per_row_l2563_256368

theorem desks_per_row (total_students : ℕ) (restroom_students : ℕ) (rows : ℕ) :
  total_students = 23 →
  restroom_students = 2 →
  rows = 4 →
  let absent_students := 3 * restroom_students - 1
  let present_students := total_students - restroom_students - absent_students
  let total_desks := (3 * present_students) / 2
  total_desks / rows = 6 :=
by sorry

end desks_per_row_l2563_256368


namespace parabola_vertex_and_focus_l2563_256336

/-- A parabola is defined by the equation x = (1/8) * y^2 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = (1/8) * p.2^2}

/-- The vertex of a parabola is the point where it turns -/
def Vertex (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The focus of a parabola is a fixed point used in its geometric definition -/
def Focus (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem parabola_vertex_and_focus :
  Vertex Parabola = (0, 0) ∧ Focus Parabola = (1/2, 0) := by
  sorry

end parabola_vertex_and_focus_l2563_256336


namespace max_value_expression_l2563_256362

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 2) :
  a * b * Real.sqrt 3 + 3 * b * c ≤ 2 ∧ ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 = 2 ∧ a * b * Real.sqrt 3 + 3 * b * c = 2 :=
by sorry

end max_value_expression_l2563_256362


namespace probability_two_black_marbles_l2563_256305

/-- The probability of drawing two black marbles without replacement from a jar -/
theorem probability_two_black_marbles (blue yellow black : ℕ) 
  (h_blue : blue = 4)
  (h_yellow : yellow = 5)
  (h_black : black = 12) : 
  (black / (blue + yellow + black)) * ((black - 1) / (blue + yellow + black - 1)) = 11 / 35 := by
  sorry

end probability_two_black_marbles_l2563_256305


namespace min_value_problem_l2563_256389

theorem min_value_problem (a : ℝ) (h_a : a > 0) :
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ y ≥ a * (x - 3) ∧
    (∀ x' y' : ℝ, x' ≥ 1 → x' + y' ≤ 3 → y' ≥ a * (x' - 3) → 2 * x' + y' ≥ 2 * x + y) ∧
    2 * x + y = 1) →
  a = 1 / 2 := by
sorry

end min_value_problem_l2563_256389


namespace pascal_triangle_30_rows_count_l2563_256330

/-- Number of elements in a row of Pascal's Triangle -/
def pascal_row_count (n : ℕ) : ℕ := n + 1

/-- Sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of elements in the first 30 rows of Pascal's Triangle is 465 -/
theorem pascal_triangle_30_rows_count : sum_first_n 30 = 465 := by sorry

end pascal_triangle_30_rows_count_l2563_256330


namespace max_additional_license_plates_l2563_256311

def initial_first_set : Finset Char := {'C', 'H', 'L', 'P', 'R'}
def initial_second_set : Finset Char := {'A', 'I', 'O'}
def initial_third_set : Finset Char := {'D', 'M', 'N', 'T'}

def initial_combinations : ℕ := initial_first_set.card * initial_second_set.card * initial_third_set.card

def max_additional_combinations : ℕ := 
  (initial_first_set.card * (initial_second_set.card + 2) * initial_third_set.card) - initial_combinations

theorem max_additional_license_plates : max_additional_combinations = 40 := by
  sorry

end max_additional_license_plates_l2563_256311


namespace R_C_S_collinear_l2563_256344

-- Define the ellipse Γ
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 16 + p.2^2 / 9 = 1}

-- Define points A and B
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (4, 0)

-- Define point C outside the ellipse
variable (C : ℝ × ℝ)
axiom C_outside : C ∉ Γ

-- Define points P and Q as intersections of CA and CB with Γ
variable (P Q : ℝ × ℝ)
axiom P_on_Γ : P ∈ Γ
axiom Q_on_Γ : Q ∈ Γ
axiom P_on_CA : ∃ t : ℝ, P = (1 - t) • A + t • C
axiom Q_on_CB : ∃ t : ℝ, Q = (1 - t) • B + t • C

-- Define points R and S as intersections of tangents with Γ
variable (R S : ℝ × ℝ)
axiom R_on_Γ : R ∈ Γ
axiom S_on_Γ : S ∈ Γ

-- Define the tangent condition
def is_tangent (p q : ℝ × ℝ) : Prop := sorry

axiom AQ_tangent : is_tangent A Q
axiom BP_tangent : is_tangent B P

-- Theorem to prove
theorem R_C_S_collinear : ∃ (m b : ℝ), R.2 = m * R.1 + b ∧ C.2 = m * C.1 + b ∧ S.2 = m * S.1 + b :=
sorry

end R_C_S_collinear_l2563_256344


namespace gcd_n_pow_13_minus_n_l2563_256361

theorem gcd_n_pow_13_minus_n : ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℤ), (d : ℤ) ∣ (n^13 - n)) ∧ 
  (∀ (k : ℕ), k > 0 → (∀ (n : ℤ), (k : ℤ) ∣ (n^13 - n)) → k ∣ d) ∧
  d = 2730 := by
sorry

end gcd_n_pow_13_minus_n_l2563_256361


namespace function_property_l2563_256385

theorem function_property (f : ℝ → ℝ) (h : ∀ x, f (Real.sin x) = Real.sin (2011 * x)) :
  ∀ x, f (Real.cos x) = Real.cos (2011 * x) := by
  sorry

end function_property_l2563_256385


namespace net_gain_calculation_l2563_256386

def initial_value : ℝ := 500000

def first_sale_profit : ℝ := 0.15
def first_buyback_loss : ℝ := 0.05
def second_sale_profit : ℝ := 0.10
def final_buyback_loss : ℝ := 0.10

def first_sale (value : ℝ) : ℝ := value * (1 + first_sale_profit)
def first_buyback (value : ℝ) : ℝ := value * (1 - first_buyback_loss)
def second_sale (value : ℝ) : ℝ := value * (1 + second_sale_profit)
def final_buyback (value : ℝ) : ℝ := value * (1 - final_buyback_loss)

def total_sales (v : ℝ) : ℝ := first_sale v + second_sale (first_buyback (first_sale v))
def total_purchases (v : ℝ) : ℝ := first_buyback (first_sale v) + final_buyback (second_sale (first_buyback (first_sale v)))

theorem net_gain_calculation (v : ℝ) : 
  total_sales v - total_purchases v = 88837.50 :=
by sorry

end net_gain_calculation_l2563_256386


namespace metallic_sheet_length_proof_l2563_256325

/-- The length of a rectangular metallic sheet that forms a box of volume 24000 m³ when 10 m squares are cut from each corner. -/
def metallic_sheet_length : ℝ := 820

/-- The width of the rectangular metallic sheet. -/
def sheet_width : ℝ := 50

/-- The side length of the square cut from each corner. -/
def corner_cut : ℝ := 10

/-- The volume of the resulting box. -/
def box_volume : ℝ := 24000

theorem metallic_sheet_length_proof :
  (metallic_sheet_length - 2 * corner_cut) * (sheet_width - 2 * corner_cut) * corner_cut = box_volume :=
by sorry

end metallic_sheet_length_proof_l2563_256325


namespace ellipse_intersection_l2563_256340

/-- Definition of an ellipse with given foci and a point on it -/
def is_ellipse (f₁ f₂ p : ℝ × ℝ) : Prop :=
  Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
  Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) =
  Real.sqrt ((0 - f₁.1)^2 + (0 - f₁.2)^2) +
  Real.sqrt ((0 - f₂.1)^2 + (0 - f₂.2)^2)

theorem ellipse_intersection :
  let f₁ : ℝ × ℝ := (0, 5)
  let f₂ : ℝ × ℝ := (4, 0)
  let p : ℝ × ℝ := (28/9, 0)
  is_ellipse f₁ f₂ (0, 0) → is_ellipse f₁ f₂ p :=
by sorry

end ellipse_intersection_l2563_256340


namespace clothing_purchase_properties_l2563_256369

/-- Represents the clothing purchase problem for a recitation competition. -/
structure ClothingPurchase where
  total_students : Nat
  combined_cost : Nat
  cost_ratio : Nat → Nat → Prop
  boy_girl_ratio : Nat → Nat → Prop
  max_total_cost : Nat

/-- Calculates the unit prices of men's and women's clothing. -/
def calculate_unit_prices (cp : ClothingPurchase) : Nat × Nat :=
  sorry

/-- Counts the number of valid purchasing plans. -/
def count_valid_plans (cp : ClothingPurchase) : Nat :=
  sorry

/-- Calculates the minimum cost of clothing purchase. -/
def minimum_cost (cp : ClothingPurchase) : Nat :=
  sorry

/-- Main theorem proving the properties of the clothing purchase problem. -/
theorem clothing_purchase_properties (cp : ClothingPurchase) 
  (h1 : cp.total_students = 150)
  (h2 : cp.combined_cost = 220)
  (h3 : cp.cost_ratio = λ m w => 6 * m = 5 * w)
  (h4 : cp.boy_girl_ratio = λ b g => b ≤ 2 * g / 3)
  (h5 : cp.max_total_cost = 17000) :
  let (men_price, women_price) := calculate_unit_prices cp
  men_price = 100 ∧ 
  women_price = 120 ∧ 
  count_valid_plans cp = 11 ∧ 
  minimum_cost cp = 16800 :=
sorry

end clothing_purchase_properties_l2563_256369
