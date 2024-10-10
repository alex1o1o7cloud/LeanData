import Mathlib

namespace lollipop_distribution_theorem_l920_92030

/-- Given a number of lollipops and kids, calculate the minimum number of additional
    lollipops needed for equal distribution -/
def min_additional_lollipops (total_lollipops : ℕ) (num_kids : ℕ) : ℕ :=
  let lollipops_per_kid := (total_lollipops + num_kids - 1) / num_kids
  lollipops_per_kid * num_kids - total_lollipops

/-- Theorem stating that for 650 lollipops and 42 kids, 
    the minimum number of additional lollipops needed is 22 -/
theorem lollipop_distribution_theorem :
  min_additional_lollipops 650 42 = 22 := by
  sorry


end lollipop_distribution_theorem_l920_92030


namespace geometric_sequence_fourth_term_l920_92042

theorem geometric_sequence_fourth_term 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h1 : a₁ = 2^(1/4)) 
  (h2 : a₂ = 2^(1/5)) 
  (h3 : a₃ = 2^(1/10)) 
  (h_geometric : ∃ r : ℝ, r ≠ 0 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) : 
  a₄ = 2^(1/10) := by
sorry

end geometric_sequence_fourth_term_l920_92042


namespace cistern_filling_time_l920_92020

theorem cistern_filling_time (p q : ℝ) (h1 : q = 15) (h2 : 2/p + 2/q + 10.5/q = 1) : p = 12 := by
  sorry

end cistern_filling_time_l920_92020


namespace rectangles_on_4x4_grid_l920_92012

/-- A 4x4 grid of points separated by unit distances -/
def Grid := Fin 5 × Fin 5

/-- A rectangle on the grid is defined by two vertical lines and two horizontal lines -/
def Rectangle := (Fin 5 × Fin 5) × (Fin 5 × Fin 5)

/-- The number of rectangles on a 4x4 grid -/
def num_rectangles : ℕ := sorry

theorem rectangles_on_4x4_grid : num_rectangles = 100 := by sorry

end rectangles_on_4x4_grid_l920_92012


namespace rectangle_division_exists_l920_92002

/-- A rectangle in a 2D plane --/
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

/-- Predicate to check if a set of points forms a rectangle --/
def IsRectangle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- A division of a rectangle into smaller rectangles --/
def RectangleDivision (r : Rectangle) (divisions : List Rectangle) : Prop := sorry

/-- Check if the union of two rectangles forms a rectangle --/
def UnionIsRectangle (r1 r2 : Rectangle) : Prop := sorry

/-- Main theorem: There exists a division of a rectangle into 5 smaller rectangles
    such that the union of any two of them is not a rectangle --/
theorem rectangle_division_exists :
  ∃ (r : Rectangle) (divisions : List Rectangle),
    RectangleDivision r divisions ∧
    divisions.length = 5 ∧
    ∀ (r1 r2 : Rectangle), r1 ∈ divisions → r2 ∈ divisions → r1 ≠ r2 →
      ¬UnionIsRectangle r1 r2 := by
  sorry

end rectangle_division_exists_l920_92002


namespace solve_cubic_equation_l920_92094

theorem solve_cubic_equation (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) :
  (x + 1)^3 = x^3 → x = 0 := by
  sorry

end solve_cubic_equation_l920_92094


namespace max_books_borrowed_l920_92080

theorem max_books_borrowed (total_students : ℕ) (zero_book_students : ℕ) (one_book_students : ℕ) (two_book_students : ℕ) (avg_books : ℕ) :
  total_students = 40 →
  zero_book_students = 2 →
  one_book_students = 12 →
  two_book_students = 12 →
  avg_books = 2 →
  ∃ (max_books : ℕ),
    max_books = 5 ∧
    ∀ (student_books : ℕ),
      student_books ≤ max_books ∧
      (total_students * avg_books =
        0 * zero_book_students +
        1 * one_book_students +
        2 * two_book_students +
        (total_students - zero_book_students - one_book_students - two_book_students) * 3 +
        (max_books - 3)) :=
by sorry

end max_books_borrowed_l920_92080


namespace slope_angle_45_implies_a_equals_3_l920_92082

theorem slope_angle_45_implies_a_equals_3 (a : ℝ) :
  (∃ (x y : ℝ), (a - 2) * x - y + 3 = 0 ∧ 
   Real.arctan ((a - 2) : ℝ) = π / 4) →
  a = 3 := by sorry

end slope_angle_45_implies_a_equals_3_l920_92082


namespace geometric_sequence_sum_l920_92000

/-- Given a geometric sequence with common ratio 2 and sum of first 4 terms equal to 1,
    prove that the sum of the first 8 terms is 17. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  (a 0 + a 1 + a 2 + a 3 = 1) →  -- sum of first 4 terms is 1
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 17) :=
by sorry

end geometric_sequence_sum_l920_92000


namespace library_reorganization_l920_92037

theorem library_reorganization (total_books : ℕ) (books_per_section : ℕ) (remainder : ℕ) : 
  total_books = 1521 * 41 →
  books_per_section = 45 →
  remainder = total_books % books_per_section →
  remainder = 36 := by
sorry

end library_reorganization_l920_92037


namespace carbon_mass_percentage_l920_92038

/-- The mass percentage of an element in a compound -/
def mass_percentage (element : String) (compound : String) : ℝ := sorry

/-- The given mass percentage of C in the compound -/
def given_percentage : ℝ := 54.55

theorem carbon_mass_percentage (compound : String) :
  mass_percentage "C" compound = given_percentage := by sorry

end carbon_mass_percentage_l920_92038


namespace students_not_picked_l920_92047

theorem students_not_picked (total : ℕ) (groups : ℕ) (per_group : ℕ) (h1 : total = 64) (h2 : groups = 4) (h3 : per_group = 7) : 
  total - (groups * per_group) = 36 := by
sorry

end students_not_picked_l920_92047


namespace count_divisors_of_twenty_divisible_by_five_l920_92017

theorem count_divisors_of_twenty_divisible_by_five : 
  let a : ℕ → Prop := λ n => 
    n > 0 ∧ 5 ∣ n ∧ n ∣ 20
  (Finset.filter a (Finset.range 21)).card = 3 := by
  sorry

end count_divisors_of_twenty_divisible_by_five_l920_92017


namespace inequality_proof_l920_92071

theorem inequality_proof (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x * y * z = 1) : 
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + x) * (1 + z))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end inequality_proof_l920_92071


namespace quadratic_inequality_empty_solution_set_l920_92079

theorem quadratic_inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end quadratic_inequality_empty_solution_set_l920_92079


namespace division_value_problem_l920_92035

theorem division_value_problem (x : ℝ) : 
  (740 / x) - 175 = 10 → x = 4 := by
  sorry

end division_value_problem_l920_92035


namespace read_time_is_two_hours_l920_92098

/-- Calculates the time taken to read a given number of pages at an increased reading speed. -/
def time_to_read (normal_speed : ℕ) (speed_increase : ℕ) (total_pages : ℕ) : ℚ :=
  total_pages / (normal_speed * speed_increase)

/-- Theorem stating that given the conditions from the problem, the time taken to read is 2 hours. -/
theorem read_time_is_two_hours (normal_speed : ℕ) (speed_increase : ℕ) (total_pages : ℕ)
  (h1 : normal_speed = 12)
  (h2 : speed_increase = 3)
  (h3 : total_pages = 72) :
  time_to_read normal_speed speed_increase total_pages = 2 := by
  sorry

end read_time_is_two_hours_l920_92098


namespace transform_F_coordinates_l920_92086

/-- Reflects a point over the x-axis -/
def reflect_over_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Rotates a point 90 degrees counterclockwise around the origin -/
def rotate_90_ccw (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

/-- The initial coordinates of point F -/
def F : ℝ × ℝ := (3, -1)

theorem transform_F_coordinates :
  (rotate_90_ccw (reflect_over_x F)) = (-1, 3) := by
  sorry

end transform_F_coordinates_l920_92086


namespace blackboard_numbers_l920_92026

theorem blackboard_numbers (n : ℕ) (h : n = 1987) :
  let S := n * (n + 1) / 2
  let remaining_sum := S % 7
  ∃ x, x ≤ 6 ∧ (x + 987) % 7 = remaining_sum :=
by
  sorry

end blackboard_numbers_l920_92026


namespace positive_X_value_l920_92008

-- Define the ⊠ operation
def boxtimes (X Y : ℤ) : ℤ := X^2 - 2*X + Y^2

-- Theorem statement
theorem positive_X_value :
  ∃ X : ℤ, (boxtimes X 7 = 164) ∧ (X > 0) ∧ (∀ Y : ℤ, (boxtimes Y 7 = 164) ∧ (Y > 0) → Y = X) :=
by sorry

end positive_X_value_l920_92008


namespace largest_of_three_consecutive_odds_l920_92044

theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  Odd a ∧ Odd b ∧ Odd c ∧  -- a, b, c are odd
  b = a + 2 ∧ c = b + 2 ∧   -- consecutive with difference 2
  a + b + c = 75            -- sum is 75
  → c = 27 := by sorry

end largest_of_three_consecutive_odds_l920_92044


namespace b_range_l920_92045

noncomputable section

def y (a x : ℝ) : ℝ := a^x

def f (a b x : ℝ) : ℝ := a^x + (Real.log x) / (Real.log a) + b

theorem b_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 1 2, y a x ≤ 6 - y a x) →
  (∃ x ∈ Set.Ioo 1 2, f a b x = 0) →
  -5 < b ∧ b < -2 :=
sorry

end b_range_l920_92045


namespace raccoon_nuts_problem_l920_92009

theorem raccoon_nuts_problem (raccoon_holes possum_holes : ℕ) : 
  raccoon_holes + possum_holes = 25 →
  possum_holes = raccoon_holes - 3 →
  5 * raccoon_holes = 6 * possum_holes →
  5 * raccoon_holes = 70 := by
  sorry

end raccoon_nuts_problem_l920_92009


namespace prime_power_equation_l920_92051

theorem prime_power_equation (p q s : Nat) (y : Nat) (hp : Prime p) (hq : Prime q) (hs : Prime s) (hy : y > 1) 
  (h : 2^s * q = p^y - 1) : p = 3 ∨ p = 5 := by
  sorry

end prime_power_equation_l920_92051


namespace special_sequence_eventually_periodic_l920_92054

/-- A sequence of positive integers satisfying the given property -/
def SpecialSequence (a : ℕ → ℕ+) : Prop :=
  ∀ n : ℕ, (a n : ℕ) * (a (n + 1) : ℕ) = (a (n + 2) : ℕ) * (a (n + 3) : ℕ)

/-- Definition of eventual periodicity for a sequence -/
def EventuallyPeriodic (a : ℕ → ℕ+) : Prop :=
  ∃ N k : ℕ, k > 0 ∧ ∀ n ≥ N, a n = a (n + k)

/-- Theorem stating that a special sequence is eventually periodic -/
theorem special_sequence_eventually_periodic (a : ℕ → ℕ+) 
  (h : SpecialSequence a) : EventuallyPeriodic a :=
sorry

end special_sequence_eventually_periodic_l920_92054


namespace inequality_subset_l920_92078

/-- The solution set of the system of inequalities is a subset of 2x^2 - 9x + a < 0 iff a ≤ 9 -/
theorem inequality_subset (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0 → 2*x^2 - 9*x + a < 0) ↔ a ≤ 9 := by
  sorry

end inequality_subset_l920_92078


namespace charles_skittles_l920_92075

/-- The number of Skittles Charles has left after Diana takes some away. -/
def skittles_left (initial : ℕ) (taken : ℕ) : ℕ := initial - taken

/-- Theorem: If Charles has 25 Skittles initially and Diana takes 7 Skittles away,
    then Charles will have 18 Skittles left. -/
theorem charles_skittles : skittles_left 25 7 = 18 := by
  sorry

end charles_skittles_l920_92075


namespace calculate_expression_l920_92022

theorem calculate_expression : 
  (Real.pi - 3.14) ^ 0 + |-Real.sqrt 3| - (1/2)⁻¹ - Real.sin (60 * π / 180) = -1 + Real.sqrt 3 / 2 := by
  sorry

end calculate_expression_l920_92022


namespace min_value_of_f_l920_92084

theorem min_value_of_f (x : ℝ) : 
  ∃ (m : ℝ), (∀ y : ℝ, 2 * (Real.cos y)^2 + Real.sin y ≥ m) ∧ 
  (∃ z : ℝ, 2 * (Real.cos z)^2 + Real.sin z = m) ∧ 
  m = -1 :=
sorry

end min_value_of_f_l920_92084


namespace jake_has_one_more_balloon_l920_92014

/-- The number of balloons Allan initially brought to the park -/
def allan_initial : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 6

/-- The number of additional balloons Allan bought at the park -/
def allan_bought : ℕ := 3

/-- The total number of balloons Allan had in the park -/
def allan_total : ℕ := allan_initial + allan_bought

/-- The difference between Jake's balloons and Allan's total balloons -/
def balloon_difference : ℕ := jake_balloons - allan_total

theorem jake_has_one_more_balloon : balloon_difference = 1 := by
  sorry

end jake_has_one_more_balloon_l920_92014


namespace absolute_value_equation_solution_l920_92043

theorem absolute_value_equation_solution :
  ∃ x : ℝ, (|x - 25| + |x - 21| = |3*x - 75|) ∧ (x = 71/3) := by
  sorry

end absolute_value_equation_solution_l920_92043


namespace vectors_collinear_necessary_not_sufficient_l920_92074

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a vector in 3D space
def Vector3D (A B : Point3D) : Point3D :=
  ⟨B.x - A.x, B.y - A.y, B.z - A.z⟩

-- Define collinearity for vectors
def vectorsCollinear (v1 v2 : Point3D) : Prop :=
  ∃ k : ℝ, v1 = ⟨k * v2.x, k * v2.y, k * v2.z⟩

-- Define collinearity for points
def pointsCollinear (A B C D : Point3D) : Prop :=
  ∃ t u v : ℝ, Vector3D A B = ⟨t * (C.x - A.x), t * (C.y - A.y), t * (C.z - A.z)⟩ ∧
               Vector3D A C = ⟨u * (D.x - A.x), u * (D.y - A.y), u * (D.z - A.z)⟩ ∧
               Vector3D A D = ⟨v * (B.x - A.x), v * (B.y - A.y), v * (B.z - A.z)⟩

theorem vectors_collinear_necessary_not_sufficient (A B C D : Point3D) :
  (pointsCollinear A B C D → vectorsCollinear (Vector3D A B) (Vector3D C D)) ∧
  ¬(vectorsCollinear (Vector3D A B) (Vector3D C D) → pointsCollinear A B C D) :=
by sorry

end vectors_collinear_necessary_not_sufficient_l920_92074


namespace printer_problem_l920_92041

/-- Calculates the time needed to print a given number of pages at a specific rate -/
def print_time (pages : ℕ) (rate : ℕ) : ℚ :=
  (pages : ℚ) / (rate : ℚ)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem printer_problem : 
  let pages : ℕ := 300
  let rate : ℕ := 20
  round_to_nearest (print_time pages rate) = 15 := by
  sorry

end printer_problem_l920_92041


namespace juvy_garden_rosemary_rows_l920_92064

/-- Represents a garden with rows of plants -/
structure Garden where
  total_rows : ℕ
  plants_per_row : ℕ
  parsley_rows : ℕ
  chive_plants : ℕ

/-- Calculates the number of rows planted with rosemary -/
def rosemary_rows (g : Garden) : ℕ :=
  g.total_rows - g.parsley_rows - (g.chive_plants / g.plants_per_row)

/-- Theorem stating that Juvy's garden has 2 rows of rosemary -/
theorem juvy_garden_rosemary_rows :
  let g : Garden := {
    total_rows := 20,
    plants_per_row := 10,
    parsley_rows := 3,
    chive_plants := 150
  }
  rosemary_rows g = 2 := by sorry

end juvy_garden_rosemary_rows_l920_92064


namespace average_marks_chemistry_mathematics_l920_92088

/-- Given that the total marks in physics, chemistry, and mathematics is 180 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 90. -/
theorem average_marks_chemistry_mathematics (P C M : ℕ) : 
  P + C + M = P + 180 → (C + M) / 2 = 90 := by
  sorry

end average_marks_chemistry_mathematics_l920_92088


namespace ellipse_foci_on_y_axis_iff_l920_92095

/-- Represents an ellipse equation of the form mx^2 + ny^2 = 1 --/
structure EllipseEquation (m n : ℝ) where
  eq : ∀ x y : ℝ, m * x^2 + n * y^2 = 1

/-- Predicate to check if an ellipse has foci on the y-axis --/
def hasFociOnYAxis (m n : ℝ) : Prop :=
  m > n ∧ n > 0

/-- Theorem stating that m > n > 0 is necessary and sufficient for 
    mx^2 + ny^2 = 1 to represent an ellipse with foci on the y-axis --/
theorem ellipse_foci_on_y_axis_iff (m n : ℝ) :
  hasFociOnYAxis m n ↔ ∃ (e : EllipseEquation m n), True :=
sorry

end ellipse_foci_on_y_axis_iff_l920_92095


namespace complex_power_simplification_l920_92053

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The main theorem -/
theorem complex_power_simplification :
  ((1 + i) / (1 - i)) ^ 1002 = -1 := by sorry

end complex_power_simplification_l920_92053


namespace pet_store_cages_l920_92029

/-- The number of bird cages in a pet store --/
def num_cages (num_parrots : Float) (num_parakeets : Float) (avg_birds_per_cage : Float) : Float :=
  (num_parrots + num_parakeets) / avg_birds_per_cage

/-- Theorem: The pet store has approximately 6 bird cages --/
theorem pet_store_cages :
  let num_parrots : Float := 6.0
  let num_parakeets : Float := 2.0
  let avg_birds_per_cage : Float := 1.333333333
  (num_cages num_parrots num_parakeets avg_birds_per_cage).round = 6 := by
  sorry

end pet_store_cages_l920_92029


namespace stream_speed_l920_92027

/-- Given a boat's speed in still water and its downstream travel time and distance,
    calculate the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (downstream_time : ℝ) (downstream_distance : ℝ) :
  boat_speed = 24 →
  downstream_time = 7 →
  downstream_distance = 196 →
  ∃ stream_speed : ℝ,
    stream_speed = 4 ∧
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry

#check stream_speed

end stream_speed_l920_92027


namespace curve_and_tangent_properties_l920_92091

/-- The function f(x) -/
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + n

/-- The tangent line of f at x = 0 -/
def tangent_line (x : ℝ) : ℝ := 4*x + 3

/-- The inequality condition for x > 1 -/
def inequality_condition (m n : ℝ) (k : ℤ) (x : ℝ) : Prop :=
  x > 1 → f m n (x + x * Real.log x) > f m n (↑k * (x - 1))

theorem curve_and_tangent_properties :
  ∃ (m n : ℝ) (k : ℤ),
    (∀ x, f m n x = tangent_line x) ∧
    (∀ x, inequality_condition m n k x) ∧
    m = 4 ∧ n = 3 ∧ k = 3 ∧
    (∀ k' : ℤ, (∀ x, inequality_condition m n k' x) → k' ≤ k) := by
  sorry

end curve_and_tangent_properties_l920_92091


namespace range_of_b_monotonicity_condition_comparison_inequality_l920_92063

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - x * Real.log x

-- Statement 1
theorem range_of_b (a : ℝ) (b : ℝ) (h1 : a > 0) (h2 : f a 1 = 2) 
  (h3 : ∀ x > 0, f a x ≥ b * x^2 + 2 * x) : b ≤ 0 := sorry

-- Statement 2
theorem monotonicity_condition (a : ℝ) (h : a > 0) : 
  (∀ x > 0, Monotone (f a)) ↔ a ≥ 1 / (2 * Real.exp 1) := sorry

-- Statement 3
theorem comparison_inequality (x y : ℝ) (h1 : 1 / Real.exp 1 < x) (h2 : x < y) (h3 : y < 1) :
  y / x < (1 + Real.log y) / (1 + Real.log x) := sorry

end range_of_b_monotonicity_condition_comparison_inequality_l920_92063


namespace all_four_digit_palindromes_divisible_by_11_l920_92034

/-- A four-digit palindrome is a number of the form abba where a and b are digits and a ≠ 0 -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ (a b : ℕ), 0 < a ∧ a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem all_four_digit_palindromes_divisible_by_11 :
  ∀ n : ℕ, is_four_digit_palindrome n → n % 11 = 0 :=
by sorry

end all_four_digit_palindromes_divisible_by_11_l920_92034


namespace area_inside_rectangle_outside_circles_l920_92031

/-- The area inside a rectangle but outside three quarter circles --/
theorem area_inside_rectangle_outside_circles (CD DA : ℝ) (r₁ r₂ r₃ : ℝ) :
  CD = 3 →
  DA = 5 →
  r₁ = 1 →
  r₂ = 2 →
  r₃ = 3 →
  (CD * DA) - ((π * r₁^2) / 4 + (π * r₂^2) / 4 + (π * r₃^2) / 4) = 15 - (7 * π / 2) :=
by sorry

end area_inside_rectangle_outside_circles_l920_92031


namespace other_root_of_quadratic_l920_92072

theorem other_root_of_quadratic (c : ℝ) : 
  (3^2 - 5*3 + c = 0) → 
  (∃ x : ℝ, x ≠ 3 ∧ x^2 - 5*x + c = 0 ∧ x = 2) := by
  sorry

end other_root_of_quadratic_l920_92072


namespace morning_sodas_count_l920_92087

theorem morning_sodas_count (afternoon_sodas : ℕ) (total_sodas : ℕ) 
  (h1 : afternoon_sodas = 19)
  (h2 : total_sodas = 96) :
  total_sodas - afternoon_sodas = 77 := by
  sorry

end morning_sodas_count_l920_92087


namespace symmetry_implies_sum_l920_92039

theorem symmetry_implies_sum (a b : ℝ) :
  (∀ x y : ℝ, y = a * x + 8 ↔ x = -1/2 * y + b) →
  a + b = 2 := by
  sorry

end symmetry_implies_sum_l920_92039


namespace complement_of_M_l920_92010

def M : Set ℝ := {a : ℝ | a^2 - 2*a > 0}

theorem complement_of_M : 
  {a : ℝ | a ∉ M} = Set.Icc 0 2 := by sorry

end complement_of_M_l920_92010


namespace commute_speed_theorem_l920_92021

theorem commute_speed_theorem (d : ℝ) (t : ℝ) (h1 : d = 50 * (t + 1/15)) (h2 : d = 70 * (t - 1/15)) :
  d / t = 58 := by sorry

end commute_speed_theorem_l920_92021


namespace points_on_line_relationship_l920_92067

/-- Given two points A(-2, y₁) and B(1, y₂) on the line y = -2x + 3, prove that y₁ > y₂ -/
theorem points_on_line_relationship (y₁ y₂ : ℝ) : 
  ((-2 : ℝ), y₁) ∈ {(x, y) | y = -2*x + 3} → 
  ((1 : ℝ), y₂) ∈ {(x, y) | y = -2*x + 3} → 
  y₁ > y₂ := by
  sorry

end points_on_line_relationship_l920_92067


namespace tank_capacity_l920_92070

/-- Represents the tank system with its properties -/
structure TankSystem where
  capacity : ℝ
  outletA_time : ℝ
  outletB_time : ℝ
  inlet_rate : ℝ
  combined_extra_time : ℝ

/-- The tank system satisfies the given conditions -/
def satisfies_conditions (ts : TankSystem) : Prop :=
  ts.outletA_time = 5 ∧
  ts.outletB_time = 8 ∧
  ts.inlet_rate = 4 * 60 ∧
  ts.combined_extra_time = 3

/-- The theorem stating that the tank capacity is 1200 litres -/
theorem tank_capacity (ts : TankSystem) 
  (h : satisfies_conditions ts) : ts.capacity = 1200 := by
  sorry

#check tank_capacity

end tank_capacity_l920_92070


namespace abs_equation_solution_l920_92050

theorem abs_equation_solution : ∃! x : ℚ, |x - 3| = |x - 4| := by
  sorry

end abs_equation_solution_l920_92050


namespace jellybeans_remaining_l920_92032

/-- Given a jar of jellybeans and a class of students, calculate the remaining jellybeans after some students eat them. -/
theorem jellybeans_remaining (total_jellybeans : ℕ) (total_students : ℕ) (absent_students : ℕ) (jellybeans_per_student : ℕ)
  (h1 : total_jellybeans = 100)
  (h2 : total_students = 24)
  (h3 : absent_students = 2)
  (h4 : jellybeans_per_student = 3) :
  total_jellybeans - (total_students - absent_students) * jellybeans_per_student = 34 := by
  sorry

end jellybeans_remaining_l920_92032


namespace dihedral_angle_cosine_value_l920_92096

/-- Regular triangular pyramid with inscribed sphere -/
structure RegularTriangularPyramid where
  /-- Side length of the base triangle -/
  base_side : ℝ
  /-- Radius of the inscribed sphere -/
  sphere_radius : ℝ
  /-- The sphere radius is one-fourth of the base side length -/
  radius_relation : sphere_radius = base_side / 4

/-- Dihedral angle at the apex of a regular triangular pyramid -/
def dihedral_angle_cosine (pyramid : RegularTriangularPyramid) : ℝ :=
  -- Definition of the dihedral angle cosine
  sorry

/-- Theorem: The cosine of the dihedral angle at the apex is 23/26 -/
theorem dihedral_angle_cosine_value (pyramid : RegularTriangularPyramid) :
  dihedral_angle_cosine pyramid = 23 / 26 :=
by
  sorry

end dihedral_angle_cosine_value_l920_92096


namespace carols_piggy_bank_l920_92024

/-- Represents the contents of Carol's piggy bank -/
structure PiggyBank where
  nickels : ℕ
  dimes : ℕ

/-- The value of the piggy bank in cents -/
def bankValue (bank : PiggyBank) : ℕ :=
  5 * bank.nickels + 10 * bank.dimes

theorem carols_piggy_bank :
  ∃ (bank : PiggyBank),
    bankValue bank = 455 ∧
    bank.nickels = bank.dimes + 7 ∧
    bank.nickels = 35 := by
  sorry

end carols_piggy_bank_l920_92024


namespace carlos_blocks_given_l920_92055

/-- The number of blocks Carlos gave to Rachel -/
def blocks_given (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem carlos_blocks_given :
  blocks_given 58 37 = 21 := by
  sorry

end carlos_blocks_given_l920_92055


namespace trig_identity_l920_92052

theorem trig_identity (α : Real) (h : π < α ∧ α < 3*π/2) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2*α))) = Real.sin (α/2) := by
  sorry

end trig_identity_l920_92052


namespace diagonals_not_parallel_in_32gon_l920_92023

/-- The number of sides in the regular polygon -/
def n : ℕ := 32

/-- The total number of diagonals in an n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of pairs of parallel sides in an n-sided polygon -/
def parallel_side_pairs (n : ℕ) : ℕ := n / 2

/-- The number of diagonals parallel to one pair of sides -/
def diagonals_per_parallel_pair (n : ℕ) : ℕ := (n - 4) / 2

/-- The total number of parallel diagonals -/
def total_parallel_diagonals (n : ℕ) : ℕ :=
  parallel_side_pairs n * diagonals_per_parallel_pair n

/-- The number of diagonals not parallel to any side in a regular 32-gon -/
theorem diagonals_not_parallel_in_32gon :
  total_diagonals n - total_parallel_diagonals n = 240 := by
  sorry

end diagonals_not_parallel_in_32gon_l920_92023


namespace lily_pad_coverage_time_l920_92006

def days_to_half_coverage : ℕ := 57

theorem lily_pad_coverage_time :
  ∀ (total_coverage : ℝ) (daily_growth_factor : ℝ),
    total_coverage > 0 →
    daily_growth_factor = 2 →
    (daily_growth_factor ^ days_to_half_coverage : ℝ) * (1 / 2) = 1 →
    (daily_growth_factor ^ (days_to_half_coverage + 1) : ℝ) = total_coverage :=
by sorry

end lily_pad_coverage_time_l920_92006


namespace attendees_with_all_items_l920_92068

def venue_capacity : ℕ := 5400
def tshirt_interval : ℕ := 90
def cap_interval : ℕ := 45
def wristband_interval : ℕ := 60

theorem attendees_with_all_items :
  (venue_capacity / (Nat.lcm tshirt_interval (Nat.lcm cap_interval wristband_interval))) = 30 := by
  sorry

end attendees_with_all_items_l920_92068


namespace soccer_ball_cost_l920_92019

theorem soccer_ball_cost (ball_cost shirt_cost : ℝ) : 
  ball_cost + shirt_cost = 100 →
  2 * ball_cost + 3 * shirt_cost = 262 →
  ball_cost = 38 := by
sorry

end soccer_ball_cost_l920_92019


namespace polynomial_evaluation_l920_92076

theorem polynomial_evaluation :
  let f (x : ℝ) := 2 * x^4 + 3 * x^3 + 5 * x^2 + x + 4
  f (-2) = 30 := by sorry

end polynomial_evaluation_l920_92076


namespace marathon_debate_duration_in_minutes_l920_92048

/-- Converts hours, minutes, and seconds to total minutes and rounds to the nearest whole number -/
def totalMinutesRounded (hours minutes seconds : ℕ) : ℕ :=
  let totalMinutes : ℚ := hours * 60 + minutes + seconds / 60
  (totalMinutes + 1/2).floor.toNat

/-- The marathon debate duration -/
def marathonDebateDuration : ℕ × ℕ × ℕ := (12, 15, 30)

theorem marathon_debate_duration_in_minutes :
  totalMinutesRounded marathonDebateDuration.1 marathonDebateDuration.2.1 marathonDebateDuration.2.2 = 736 := by
  sorry

end marathon_debate_duration_in_minutes_l920_92048


namespace garden_area_increase_l920_92011

theorem garden_area_increase : 
  let original_length : ℝ := 40
  let original_width : ℝ := 10
  let original_perimeter : ℝ := 2 * (original_length + original_width)
  let new_side_length : ℝ := original_perimeter / 4
  let original_area : ℝ := original_length * original_width
  let new_area : ℝ := new_side_length ^ 2
  new_area - original_area = 225 := by sorry

end garden_area_increase_l920_92011


namespace parallel_lines_m_equals_one_l920_92025

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1 ∧ a1 * c2 ≠ a2 * c1

/-- The theorem statement -/
theorem parallel_lines_m_equals_one (m : ℝ) :
  parallel_lines 1 (1 + m) (m - 2) (2 * m) 4 16 → m = 1 := by
  sorry


end parallel_lines_m_equals_one_l920_92025


namespace initial_solution_volume_l920_92090

theorem initial_solution_volume 
  (V : ℝ)  -- Initial volume in liters
  (h1 : 0.20 * V + 3.6 = 0.50 * (V + 3.6))  -- Equation representing the alcohol balance
  : V = 6 := by
  sorry

end initial_solution_volume_l920_92090


namespace math_reading_homework_difference_l920_92018

theorem math_reading_homework_difference :
  let math_pages : ℕ := 5
  let reading_pages : ℕ := 2
  math_pages - reading_pages = 3 :=
by sorry

end math_reading_homework_difference_l920_92018


namespace triangle_conversion_cost_l920_92058

theorem triangle_conversion_cost 
  (side1 : ℝ) (side2 : ℝ) (angle : ℝ) (cost_per_sqm : ℝ) :
  side1 = 32 →
  side2 = 68 →
  angle = 30 * π / 180 →
  cost_per_sqm = 50 →
  (1/2 * side1 * side2 * Real.sin angle) * cost_per_sqm = 54400 :=
by sorry

end triangle_conversion_cost_l920_92058


namespace notebook_cost_l920_92069

theorem notebook_cost (total_spent : ℝ) (backpack_cost : ℝ) (pens_cost : ℝ) (pencils_cost : ℝ) (num_notebooks : ℕ) :
  total_spent = 32 →
  backpack_cost = 15 →
  pens_cost = 1 →
  pencils_cost = 1 →
  num_notebooks = 5 →
  (total_spent - (backpack_cost + pens_cost + pencils_cost)) / num_notebooks = 3 := by
  sorry

end notebook_cost_l920_92069


namespace no_even_three_digit_sum_27_l920_92015

/-- A function that returns the digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is even -/
def isEven (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has exactly 3 digits -/
def isThreeDigit (n : ℕ) : Prop := sorry

theorem no_even_three_digit_sum_27 :
  ¬∃ n : ℕ, isThreeDigit n ∧ digitSum n = 27 ∧ isEven n :=
sorry

end no_even_three_digit_sum_27_l920_92015


namespace vanya_correct_answers_l920_92089

/-- The number of questions Sasha asked Vanya -/
def total_questions : ℕ := 50

/-- The number of candies Vanya receives for a correct answer -/
def correct_reward : ℕ := 7

/-- The number of candies Vanya gives for an incorrect answer -/
def incorrect_penalty : ℕ := 3

/-- The number of questions Vanya answered correctly -/
def correct_answers : ℕ := 15

theorem vanya_correct_answers :
  correct_answers * correct_reward = (total_questions - correct_answers) * incorrect_penalty :=
by sorry

end vanya_correct_answers_l920_92089


namespace diophantine_equation_solutions_l920_92003

theorem diophantine_equation_solutions :
  ∀ n k m : ℕ, 5^n - 3^k = m^2 →
    ((n = 0 ∧ k = 0 ∧ m = 0) ∨ (n = 2 ∧ k = 2 ∧ m = 4)) :=
by sorry

end diophantine_equation_solutions_l920_92003


namespace sports_equipment_choices_l920_92083

theorem sports_equipment_choices (basketballs volleyballs : ℕ) 
  (hb : basketballs = 5) (hv : volleyballs = 4) : 
  basketballs * volleyballs = 20 := by
  sorry

end sports_equipment_choices_l920_92083


namespace proposition_p_and_not_q_l920_92062

open Real

theorem proposition_p_and_not_q : 
  (∃ x : ℝ, x^2 + 2*x + 5 ≤ 4) ∧ 
  (∀ x ∈ Set.Ioo 0 (π/2), sin x + 4/(sin x) > 4) := by
  sorry

end proposition_p_and_not_q_l920_92062


namespace student_number_problem_l920_92040

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 106 → x = 122 := by
  sorry

end student_number_problem_l920_92040


namespace sea_glass_collection_l920_92060

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_red rose_blue : ℕ) 
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_red = 9)
  (h4 : rose_blue = 11)
  (dorothy_red : ℕ)
  (h5 : dorothy_red = 2 * (blanche_red + rose_red))
  (dorothy_blue : ℕ)
  (h6 : dorothy_blue = 3 * rose_blue) :
  dorothy_red + dorothy_blue = 57 := by
sorry

end sea_glass_collection_l920_92060


namespace simplified_tax_system_is_most_suitable_l920_92061

-- Define the business characteristics
structure BusinessCharacteristics where
  isFlowerSelling : Bool
  hasNoExperience : Bool
  hasSingleOutlet : Bool
  isSelfOperated : Bool

-- Define the tax regimes
inductive TaxRegime
  | UnifiedAgricultural
  | Simplified
  | General
  | Patent

-- Define a function to determine the most suitable tax regime
def mostSuitableTaxRegime (business : BusinessCharacteristics) : TaxRegime :=
  sorry

-- Theorem statement
theorem simplified_tax_system_is_most_suitable 
  (leonidBusiness : BusinessCharacteristics)
  (h1 : leonidBusiness.isFlowerSelling = true)
  (h2 : leonidBusiness.hasNoExperience = true)
  (h3 : leonidBusiness.hasSingleOutlet = true)
  (h4 : leonidBusiness.isSelfOperated = true) :
  mostSuitableTaxRegime leonidBusiness = TaxRegime.Simplified :=
sorry

end simplified_tax_system_is_most_suitable_l920_92061


namespace sufficient_condition_exclusive_or_condition_l920_92081

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0

-- Part 1: p is a sufficient condition for q
theorem sufficient_condition (m : ℝ) :
  (∀ x, p x → q x m) → m ∈ Set.Ici 4 :=
sorry

-- Part 2: m = 5, "p or q" is true, "p and q" is false
theorem exclusive_or_condition (x : ℝ) :
  (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) → x ∈ Set.Icc (-4) (-1) ∪ Set.Ioc 5 6 :=
sorry

end sufficient_condition_exclusive_or_condition_l920_92081


namespace book_reading_permutations_l920_92099

theorem book_reading_permutations :
  let n : ℕ := 5  -- total number of books
  let r : ℕ := 3  -- number of books to read
  Nat.factorial n / Nat.factorial (n - r) = 60 := by
  sorry

end book_reading_permutations_l920_92099


namespace literature_tech_cost_difference_l920_92097

theorem literature_tech_cost_difference :
  let num_books : ℕ := 45
  let lit_cost : ℕ := 7
  let tech_cost : ℕ := 5
  (num_books * lit_cost) - (num_books * tech_cost) = 90 := by
sorry

end literature_tech_cost_difference_l920_92097


namespace value_of_b_l920_92016

theorem value_of_b : ∀ b : ℕ, (5 ^ 5 * b = 3 * 15 ^ 5) ∧ (b = 9 ^ 3) → b = 729 := by
  sorry

end value_of_b_l920_92016


namespace chloe_score_l920_92046

/-- Calculates the total score in Chloe's video game. -/
def total_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ) : ℕ :=
  points_per_treasure * (treasures_level1 + treasures_level2)

/-- Proves that Chloe's total score is 81 points given the specified conditions. -/
theorem chloe_score :
  total_score 9 6 3 = 81 := by
  sorry

end chloe_score_l920_92046


namespace roots_sum_bound_l920_92033

theorem roots_sum_bound (v w : ℂ) : 
  v ≠ w → 
  v^2021 = 1 → 
  w^2021 = 1 → 
  Complex.abs (v + w) < Real.sqrt (2 + Real.sqrt 5) := by
sorry

end roots_sum_bound_l920_92033


namespace carl_weight_l920_92005

theorem carl_weight (billy brad carl : ℕ) 
  (h1 : billy = brad + 9)
  (h2 : brad = carl + 5)
  (h3 : billy = 159) : 
  carl = 145 := by sorry

end carl_weight_l920_92005


namespace max_attempts_for_ten_rooms_l920_92057

/-- The maximum number of attempts needed to match n keys to n rooms -/
def maxAttempts (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- The number of rooms and keys -/
def numRooms : ℕ := 10

theorem max_attempts_for_ten_rooms :
  maxAttempts numRooms = 45 :=
by sorry

end max_attempts_for_ten_rooms_l920_92057


namespace third_angle_is_90_l920_92065

-- Define a triangle with two known angles
def Triangle (angle1 angle2 : ℝ) :=
  { angle3 : ℝ // angle1 + angle2 + angle3 = 180 }

-- Theorem: In a triangle with angles of 30 and 60 degrees, the third angle is 90 degrees
theorem third_angle_is_90 :
  ∀ (t : Triangle 30 60), t.val = 90 := by
  sorry

end third_angle_is_90_l920_92065


namespace triangle_side_value_l920_92036

theorem triangle_side_value (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + 10 * c = 2 * (Real.sin A + Real.sin B + 10 * Real.sin C) ∧
  A = π / 3 →
  a = Real.sqrt 3 := by
sorry

end triangle_side_value_l920_92036


namespace g_at_7_equals_neg_20_l920_92077

/-- A polynomial function g(x) of degree 7 -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 4

/-- Theorem stating that g(7) = -20 given g(-7) = 12 -/
theorem g_at_7_equals_neg_20 (a b c : ℝ) : g a b c (-7) = 12 → g a b c 7 = -20 := by
  sorry

end g_at_7_equals_neg_20_l920_92077


namespace log_stack_sum_l920_92028

theorem log_stack_sum : ∀ (a l n : ℕ), 
  a = 15 → l = 4 → n = 12 → 
  (n * (a + l)) / 2 = 114 := by
sorry

end log_stack_sum_l920_92028


namespace f_range_l920_92085

/-- The function f(x) = |x+10| - |3x-1| -/
def f (x : ℝ) : ℝ := |x + 10| - |3*x - 1|

/-- The range of f is (-∞, 31] -/
theorem f_range :
  Set.range f = Set.Iic 31 := by sorry

end f_range_l920_92085


namespace stone_145_is_5_l920_92056

def stone_number (n : ℕ) : ℕ := 
  let cycle := 28
  n % cycle

theorem stone_145_is_5 : stone_number 145 = stone_number 5 := by
  sorry

end stone_145_is_5_l920_92056


namespace arithmetic_sequence_sum_l920_92013

/-- Given an arithmetic sequence {a_n} where a_3 + a_4 + a_5 = 12, 
    the sum of the first seven terms is 28. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 3 + a 4 + a 5 = 12 →                    -- given condition
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := by
sorry


end arithmetic_sequence_sum_l920_92013


namespace rectangle_perimeter_l920_92092

/-- Given a rectangle with length thrice its breadth and area 507 m², 
    prove that its perimeter is 104 m. -/
theorem rectangle_perimeter (breadth length : ℝ) : 
  length = 3 * breadth → 
  breadth * length = 507 → 
  2 * (length + breadth) = 104 := by
  sorry

end rectangle_perimeter_l920_92092


namespace greatest_prime_factor_of_sum_l920_92066

def product_of_evens (x : ℕ) : ℕ :=
  if x % 2 = 0 then
    Finset.prod (Finset.range ((x / 2) + 1)) (fun i => 2 * i)
  else
    Finset.prod (Finset.range (x / 2)) (fun i => 2 * i)

def greatest_prime_factor (n : ℕ) : ℕ := sorry

theorem greatest_prime_factor_of_sum : 
  greatest_prime_factor (product_of_evens 26 + product_of_evens 24) = 23 := by sorry

end greatest_prime_factor_of_sum_l920_92066


namespace sum_gcf_lcm_l920_92049

/-- The greatest common factor of 15, 20, and 30 -/
def A : ℕ := Nat.gcd 15 (Nat.gcd 20 30)

/-- The least common multiple of 15, 20, and 30 -/
def B : ℕ := Nat.lcm 15 (Nat.lcm 20 30)

/-- The sum of the greatest common factor and least common multiple of 15, 20, and 30 is 65 -/
theorem sum_gcf_lcm : A + B = 65 := by sorry

end sum_gcf_lcm_l920_92049


namespace min_members_in_association_l920_92004

/-- Represents an association with men and women members -/
structure Association where
  men : ℕ
  women : ℕ

/-- Calculates the total number of members in the association -/
def Association.totalMembers (a : Association) : ℕ := a.men + a.women

/-- Calculates the number of homeowners in the association -/
def Association.homeowners (a : Association) : ℚ := 0.1 * a.men + 0.2 * a.women

/-- Theorem stating the minimum number of members in the association -/
theorem min_members_in_association :
  ∃ (a : Association), a.homeowners ≥ 18 ∧
  (∀ (b : Association), b.homeowners ≥ 18 → a.totalMembers ≤ b.totalMembers) ∧
  a.totalMembers = 91 := by
  sorry

end min_members_in_association_l920_92004


namespace acme_vowel_soup_strings_l920_92073

/-- Represents the number of times each vowel appears in the soup -/
def vowel_count : Fin 5 → ℕ
  | 0 => 6  -- A
  | 1 => 6  -- E
  | 2 => 6  -- I
  | 3 => 6  -- O
  | 4 => 3  -- U

/-- The length of the strings to be formed -/
def string_length : ℕ := 6

/-- Calculates the number of possible strings -/
def count_strings : ℕ :=
  (Finset.range 4).sum (λ k =>
    (Nat.choose string_length k) * (4 * vowel_count 0) ^ (string_length - k))

theorem acme_vowel_soup_strings :
  count_strings = 117072 :=
sorry

end acme_vowel_soup_strings_l920_92073


namespace stating_correct_deposit_equation_l920_92001

/-- Represents the annual interest rate as a decimal -/
def annual_rate : ℝ := 0.0369

/-- Represents the number of years for the fixed deposit -/
def years : ℕ := 3

/-- Represents the tax rate on interest as a decimal -/
def tax_rate : ℝ := 0.2

/-- Represents the final withdrawal amount in yuan -/
def final_amount : ℝ := 5442.8

/-- 
Theorem stating the correct equation for calculating the initial deposit amount,
given the annual interest rate, number of years, tax rate, and final withdrawal amount.
-/
theorem correct_deposit_equation (x : ℝ) :
  x + x * annual_rate * (years : ℝ) * (1 - tax_rate) = final_amount :=
sorry

end stating_correct_deposit_equation_l920_92001


namespace program_output_l920_92059

theorem program_output : 
  let a₀ := 10
  let b := a₀ - 8
  let a₁ := a₀ - b
  a₁ = 8 := by sorry

end program_output_l920_92059


namespace N_is_composite_l920_92093

def N (y : ℕ) : ℚ := (y^125 - 1) / (3^22 - 1)

theorem N_is_composite : ∃ (y : ℕ) (a b : ℕ), a > 1 ∧ b > 1 ∧ N y = (a * b : ℚ) := by
  sorry

end N_is_composite_l920_92093


namespace subtract_and_multiply_l920_92007

theorem subtract_and_multiply (N V : ℝ) : N = 12 → (4 * N - 3 = 9 * (N - V)) → V = 7 := by
  sorry

end subtract_and_multiply_l920_92007
