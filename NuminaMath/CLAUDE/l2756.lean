import Mathlib

namespace distinct_arrangements_count_l2756_275699

/-- Represents a configuration of lit buttons on a 3 × 2 grid -/
def ButtonGrid := Fin 3 → Fin 2 → Bool

/-- Returns true if at least one button in the grid is lit -/
def atLeastOneLit (grid : ButtonGrid) : Prop :=
  ∃ i j, grid i j = true

/-- Two grids are equivalent if one can be obtained from the other by translation -/
def equivalentGrids (grid1 grid2 : ButtonGrid) : Prop :=
  sorry

/-- The number of distinct observable arrangements -/
def distinctArrangements : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct arrangements is 44 -/
theorem distinct_arrangements_count :
  distinctArrangements = 44 :=
sorry

end distinct_arrangements_count_l2756_275699


namespace spherical_cap_area_ratio_l2756_275664

/-- Given two concentric spheres and a spherical cap area on the smaller sphere,
    calculate the corresponding spherical cap area on the larger sphere. -/
theorem spherical_cap_area_ratio (R₁ R₂ A₁ : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) (h₃ : A₁ > 0) :
  let A₂ := A₁ * (R₂ / R₁)^2
  R₁ = 4 → R₂ = 6 → A₁ = 17 → A₂ = 38.25 := by
  sorry

end spherical_cap_area_ratio_l2756_275664


namespace probability_of_selecting_specific_pair_l2756_275687

/-- Given a box of shoes with the following properties:
    - There are 20 pairs of shoes (40 shoes in total)
    - Each pair has a unique design
    Prove that the probability of randomly selecting both shoes
    from a specific pair (pair A) is 1/780. -/
theorem probability_of_selecting_specific_pair (total_shoes : Nat) (total_pairs : Nat)
    (h1 : total_shoes = 40)
    (h2 : total_pairs = 20)
    (h3 : total_shoes = 2 * total_pairs) :
  (1 : ℚ) / total_shoes * (1 : ℚ) / (total_shoes - 1) = 1 / 780 := by
  sorry

#check probability_of_selecting_specific_pair

end probability_of_selecting_specific_pair_l2756_275687


namespace women_in_room_l2756_275602

theorem women_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 7 / 9 →
  initial_men + 8 = 28 →
  2 * (initial_women - 2) = 48 :=
by sorry

end women_in_room_l2756_275602


namespace blue_segments_count_l2756_275653

/-- Set A of points (x, y) where x and y are natural numbers between 1 and 20 inclusive -/
def A : Set (ℕ × ℕ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 20 ∧ 1 ≤ p.2 ∧ p.2 ≤ 20}

/-- Set B of points (x, y) where x and y are natural numbers between 2 and 19 inclusive -/
def B : Set (ℕ × ℕ) := {p | 2 ≤ p.1 ∧ p.1 ≤ 19 ∧ 2 ≤ p.2 ∧ p.2 ≤ 19}

/-- Color of a point in A -/
inductive Color
| Red
| Blue

/-- Coloring function for points in A -/
def coloring : A → Color := sorry

/-- Total number of red points in A -/
def total_red_points : ℕ := 219

/-- Number of red points in B -/
def red_points_in_B : ℕ := 180

/-- Corner points are blue -/
axiom corner_points_blue :
  coloring ⟨(1, 1), sorry⟩ = Color.Blue ∧
  coloring ⟨(1, 20), sorry⟩ = Color.Blue ∧
  coloring ⟨(20, 1), sorry⟩ = Color.Blue ∧
  coloring ⟨(20, 20), sorry⟩ = Color.Blue

/-- Number of black line segments of length 1 -/
def black_segments : ℕ := 237

/-- Theorem: The number of blue line segments of length 1 is 233 -/
theorem blue_segments_count : ℕ := by
  sorry

end blue_segments_count_l2756_275653


namespace alex_age_l2756_275626

/-- Given the ages of Alex, Bella, and Carlos, prove that Alex is 20 years old. -/
theorem alex_age (bella_age carlos_age alex_age : ℕ) : 
  bella_age = 21 →
  carlos_age = bella_age + 5 →
  alex_age = carlos_age - 6 →
  alex_age = 20 := by
sorry

end alex_age_l2756_275626


namespace inverse_of_A_l2756_275615

def A : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; -2, 9]

theorem inverse_of_A :
  A⁻¹ = !![9/35, -4/35; 2/35, 3/35] := by sorry

end inverse_of_A_l2756_275615


namespace intersection_of_M_and_N_l2756_275623

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x > 1}
def N : Set ℝ := {x | ∃ y, y = Real.log (3/2 - x)}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 3/2} := by sorry

end intersection_of_M_and_N_l2756_275623


namespace circle_equation_symmetric_center_l2756_275671

/-- A circle C in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The standard equation of a circle. -/
def standardEquation (c : Circle) : ℝ → ℝ → Prop :=
  λ x y => (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Symmetry of two points about the line y = x. -/
def symmetricAboutDiagonal (p q : ℝ × ℝ) : Prop :=
  p.1 + q.2 = p.2 + q.1 ∧ p.1 + p.2 = q.1 + q.2

theorem circle_equation_symmetric_center (c : Circle) :
  c.radius = 1 →
  symmetricAboutDiagonal c.center (1, 0) →
  standardEquation c = λ x y => x^2 + (y - 1)^2 = 1 :=
sorry

end circle_equation_symmetric_center_l2756_275671


namespace calculation_proof_l2756_275692

theorem calculation_proof : 
  Real.sqrt 5 * (-Real.sqrt 10) - (1/7)⁻¹ + |-(2^3)| = -5 * Real.sqrt 2 + 1 := by
sorry

end calculation_proof_l2756_275692


namespace sin_product_equality_l2756_275628

theorem sin_product_equality : 
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 
  (Real.cos (20 * π / 180) - 1/2) / 8 := by sorry

end sin_product_equality_l2756_275628


namespace set_intersection_theorem_l2756_275682

-- Define set A
def A : Set ℝ := {x | |x| > 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem set_intersection_theorem :
  (Set.univ \ A) ∩ B = {x | 0 ≤ x ∧ x ≤ 1} :=
sorry

end set_intersection_theorem_l2756_275682


namespace right_triangle_set_l2756_275680

theorem right_triangle_set : ∃! (a b c : ℕ), 
  ((a = 7 ∧ b = 24 ∧ c = 25) ∨ 
   (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
   (a = 4 ∧ b = 5 ∧ c = 6) ∨ 
   (a = 8 ∧ b = 15 ∧ c = 18)) ∧ 
  a^2 + b^2 = c^2 := by
  sorry

end right_triangle_set_l2756_275680


namespace solution_set_theorem_m_value_theorem_l2756_275693

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 3|

-- Define the function g
def g (x m : ℝ) : ℝ := f (x + m) + f (x - m)

-- Theorem for the solution set of the inequality
theorem solution_set_theorem :
  {x : ℝ | f x > 5 - |x + 2|} = {x : ℝ | x < 0 ∨ x > 2} :=
sorry

-- Theorem for the value of m
theorem m_value_theorem (m : ℝ) :
  (∀ x, g x m ≥ 4) ∧ (∃ x, g x m = 4) → m = 1 ∨ m = -1 :=
sorry

end solution_set_theorem_m_value_theorem_l2756_275693


namespace haley_weight_l2756_275613

/-- Given the weights of Verna, Haley, and Sherry, prove Haley's weight -/
theorem haley_weight (V H S : ℝ) 
  (verna_haley : V = H + 17)
  (verna_sherry : V = S / 2)
  (total_weight : V + S = 360) :
  H = 103 := by
  sorry

end haley_weight_l2756_275613


namespace min_value_theorem_l2756_275698

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) (hab : a * b = 1/4) :
  ∃ (min_val : ℝ), min_val = 4 + 4 * Real.sqrt 2 / 3 ∧
  ∀ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x * y = 1/4 →
  1 / (1 - x) + 2 / (1 - y) ≥ min_val :=
by sorry

end min_value_theorem_l2756_275698


namespace trapezoid_median_length_l2756_275694

/-- Given a triangle and a trapezoid with equal areas and the same altitude,
    if the base of the triangle is 24 inches and the base of the trapezoid
    is half the length of the triangle's base, then the median of the
    trapezoid is 12 inches. -/
theorem trapezoid_median_length
  (triangle_area trapezoid_area : ℝ)
  (altitude : ℝ)
  (triangle_base trapezoid_base : ℝ)
  (trapezoid_median : ℝ) :
  triangle_area = trapezoid_area →
  triangle_base = 24 →
  trapezoid_base = triangle_base / 2 →
  triangle_area = (1 / 2) * triangle_base * altitude →
  trapezoid_area = trapezoid_median * altitude →
  trapezoid_median = (trapezoid_base + trapezoid_base) / 2 →
  trapezoid_median = 12 := by
  sorry


end trapezoid_median_length_l2756_275694


namespace unique_integer_pair_solution_l2756_275630

theorem unique_integer_pair_solution : 
  ∃! (x y : ℤ), Real.sqrt (x - Real.sqrt (x + 23)) = 2 * Real.sqrt 2 - y := by
  sorry

end unique_integer_pair_solution_l2756_275630


namespace expression_value_l2756_275659

theorem expression_value (x : ℝ) : 
  let a : ℝ := 2005 * x + 2009
  let b : ℝ := 2005 * x + 2010
  let c : ℝ := 2005 * x + 2011
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 3 := by
sorry

end expression_value_l2756_275659


namespace density_of_M_l2756_275620

def M : Set ℝ :=
  {r : ℝ | ∃ (m n : ℕ+), r = (m + n) / Real.sqrt (m^2 + n^2)}

theorem density_of_M : ∀ (x y : ℝ), x ∈ M → y ∈ M → x < y →
  ∃ (z : ℝ), z ∈ M ∧ x < z ∧ z < y :=
by sorry

end density_of_M_l2756_275620


namespace max_value_on_curve_l2756_275638

/-- Given a point (a,b) on the curve y = e^2 / x where a > 1 and b > 1,
    the maximum value of a^(ln b) is e. -/
theorem max_value_on_curve (a b : ℝ) : 
  a > 1 → b > 1 → b = Real.exp 2 / a → (Real.exp 1 : ℝ) ≥ a^(Real.log b) := by
  sorry

end max_value_on_curve_l2756_275638


namespace smallest_sum_solution_l2756_275677

theorem smallest_sum_solution : ∃ (a b c : ℕ), 
  (a * c + 2 * b * c + a + 2 * b = c^2 + c + 6) ∧ 
  (∀ (x y z : ℕ), (x * z + 2 * y * z + x + 2 * y = z^2 + z + 6) → 
    (a + b + c ≤ x + y + z)) ∧
  (a = 2 ∧ b = 1 ∧ c = 1) := by
  sorry

end smallest_sum_solution_l2756_275677


namespace ten_people_prob_l2756_275608

/-- Represents the number of valid arrangements where no two adjacent people are standing
    for n people around a circular table. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The probability of no two adjacent people standing when n people
    each flip a fair coin around a circular table. -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n)

/-- The main theorem stating the probability for 10 people. -/
theorem ten_people_prob : noAdjacentStandingProb 10 = 123 / 1024 := by
  sorry


end ten_people_prob_l2756_275608


namespace inequality_proof_l2756_275686

theorem inequality_proof (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  ¬(abs a + abs b > abs (a + b)) := by
  sorry

end inequality_proof_l2756_275686


namespace product_evaluation_l2756_275681

theorem product_evaluation : (3 + 1) * (3^3 + 1^3) * (3^9 + 1^9) = 2878848 := by
  sorry

end product_evaluation_l2756_275681


namespace chocolate_mixture_proof_l2756_275621

theorem chocolate_mixture_proof (initial_weight : ℝ) (initial_percentage : ℝ) 
  (final_weight : ℝ) (final_percentage : ℝ) (added_pure_chocolate : ℝ) : 
  initial_weight = 620 →
  initial_percentage = 0.1 →
  final_weight = 1000 →
  final_percentage = 0.7 →
  added_pure_chocolate = 638 →
  (initial_weight * initial_percentage + added_pure_chocolate) / final_weight = final_percentage :=
by
  sorry

#check chocolate_mixture_proof

end chocolate_mixture_proof_l2756_275621


namespace vacation_cost_difference_l2756_275673

/-- Proves that the difference between Tom's and Dorothy's payments to equalize costs is 20 --/
theorem vacation_cost_difference (tom_paid dorothy_paid sammy_paid : ℕ) 
  (h1 : tom_paid = 105)
  (h2 : dorothy_paid = 125)
  (h3 : sammy_paid = 175) : 
  (((tom_paid + dorothy_paid + sammy_paid) / 3 - tom_paid) - 
   ((tom_paid + dorothy_paid + sammy_paid) / 3 - dorothy_paid)) = 20 := by
  sorry

#eval ((105 + 125 + 175) / 3 - 105) - ((105 + 125 + 175) / 3 - 125)

end vacation_cost_difference_l2756_275673


namespace real_roots_of_polynomial_l2756_275678

theorem real_roots_of_polynomial (x : ℝ) :
  x^4 + 2*x^3 - x - 2 = 0 ↔ x = 1 ∨ x = -2 := by
  sorry

end real_roots_of_polynomial_l2756_275678


namespace family_adults_count_l2756_275636

/-- Represents the number of adults in a family visiting an amusement park. -/
def adults : ℕ := sorry

/-- The cost of an adult ticket in dollars. -/
def adult_ticket_cost : ℕ := 22

/-- The cost of a child ticket in dollars. -/
def child_ticket_cost : ℕ := 7

/-- The number of children in the family. -/
def num_children : ℕ := 2

/-- The total cost for the family's admission in dollars. -/
def total_cost : ℕ := 58

/-- Theorem stating that the number of adults in the family is 2. -/
theorem family_adults_count : adults = 2 := by
  sorry

end family_adults_count_l2756_275636


namespace missing_number_equation_l2756_275658

theorem missing_number_equation : ∃! x : ℝ, x + 3699 + 1985 - 2047 = 31111 := by
  sorry

end missing_number_equation_l2756_275658


namespace train_speed_l2756_275619

/-- Proves that a train with given length, crossing a bridge of given length in a specific time, has a specific speed in km/hr -/
theorem train_speed (train_length bridge_length : Real) (crossing_time : Real) :
  train_length = 110 →
  bridge_length = 132 →
  crossing_time = 24.198064154867613 →
  (train_length + bridge_length) / crossing_time * 3.6 = 36 := by
  sorry

end train_speed_l2756_275619


namespace price_reduction_percentage_l2756_275611

theorem price_reduction_percentage (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500)
  (h2 : reduction_amount = 400) :
  (reduction_amount / original_price) * 100 = 80 := by
sorry

end price_reduction_percentage_l2756_275611


namespace lcm_gcf_ratio_150_500_l2756_275690

theorem lcm_gcf_ratio_150_500 : Nat.lcm 150 500 / Nat.gcd 150 500 = 30 := by
  sorry

end lcm_gcf_ratio_150_500_l2756_275690


namespace pages_per_day_l2756_275635

/-- Given a book with 144 pages, prove that reading two-thirds of it in 12 days results in 8 pages read per day. -/
theorem pages_per_day (total_pages : ℕ) (days_read : ℕ) (fraction_read : ℚ) : 
  total_pages = 144 → 
  days_read = 12 → 
  fraction_read = 2/3 →
  (fraction_read * total_pages) / days_read = 8 := by
sorry

end pages_per_day_l2756_275635


namespace opposite_of_2023_l2756_275655

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end opposite_of_2023_l2756_275655


namespace arithmetic_calculation_l2756_275614

theorem arithmetic_calculation : 3 + (12 / 3 - 1)^2 = 12 := by
  sorry

end arithmetic_calculation_l2756_275614


namespace trigonometric_inequality_l2756_275670

theorem trigonometric_inequality : 
  let a := (1/2) * Real.cos (6 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * π / 180)
  let b := 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  a < c ∧ c < b := by sorry

end trigonometric_inequality_l2756_275670


namespace same_speed_problem_l2756_275696

theorem same_speed_problem (x : ℝ) :
  let jack_speed := x^2 - 9*x - 18
  let jill_distance := x^2 - 5*x - 66
  let jill_time := x + 6
  let jill_speed := jill_distance / jill_time
  (x ≠ -6) →
  (jack_speed = jill_speed) →
  jack_speed = -4 :=
by sorry

end same_speed_problem_l2756_275696


namespace horner_rule_v₃_l2756_275675

/-- Horner's Rule for a specific polynomial -/
def horner_polynomial (x : ℤ) : ℤ := (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 3) * x + 2

/-- The third intermediate value in Horner's Rule calculation -/
def v₃ (x : ℤ) : ℤ :=
  let v₀ := 1
  let v₁ := x - 5 * v₀
  let v₂ := x * v₁ + 6
  x * v₂ + 0

theorem horner_rule_v₃ :
  v₃ (-2) = -40 :=
by sorry

end horner_rule_v₃_l2756_275675


namespace volume_surface_area_ratio_eight_cubes_l2756_275627

/-- A shape created by joining unit cubes in a line -/
structure LineCubes where
  num_cubes : ℕ

/-- Calculate the volume of the shape -/
def volume (shape : LineCubes) : ℕ :=
  shape.num_cubes

/-- Calculate the surface area of the shape -/
def surface_area (shape : LineCubes) : ℕ :=
  2 * shape.num_cubes + 2 * 4

/-- The ratio of volume to surface area for a shape with 8 unit cubes -/
theorem volume_surface_area_ratio_eight_cubes :
  let shape : LineCubes := { num_cubes := 8 }
  (volume shape : ℚ) / (surface_area shape : ℚ) = 4 / 9 := by
  sorry


end volume_surface_area_ratio_eight_cubes_l2756_275627


namespace dogsled_speed_difference_l2756_275651

/-- Calculates the difference in average speed between two dogsled teams -/
theorem dogsled_speed_difference 
  (course_distance : ℝ) 
  (team_e_speed : ℝ) 
  (time_difference : ℝ) : 
  course_distance = 300 →
  team_e_speed = 20 →
  time_difference = 3 →
  (course_distance / (course_distance / team_e_speed - time_difference)) - team_e_speed = 5 := by
  sorry


end dogsled_speed_difference_l2756_275651


namespace student_grade_proof_l2756_275618

def courses_last_year : ℕ := 6
def avg_grade_last_year : ℝ := 100
def courses_year_before : ℕ := 5
def avg_grade_two_years : ℝ := 77
def total_courses : ℕ := courses_last_year + courses_year_before

theorem student_grade_proof :
  ∃ (avg_grade_year_before : ℝ),
    avg_grade_year_before * courses_year_before + avg_grade_last_year * courses_last_year =
    avg_grade_two_years * total_courses ∧
    avg_grade_year_before = 49.4 := by
  sorry

end student_grade_proof_l2756_275618


namespace min_a_for_inequality_l2756_275632

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^3 - 3*x + 3) - a * Real.exp x - x

theorem min_a_for_inequality :
  ∃ (a : ℝ), a = 1 - 1 / Real.exp 1 ∧
  (∀ (x : ℝ), x ≥ -2 → f a x ≤ 0) ∧
  (∀ (b : ℝ), (∀ (x : ℝ), x ≥ -2 → f b x ≤ 0) → b ≥ a) :=
sorry

end min_a_for_inequality_l2756_275632


namespace hall_width_proof_l2756_275648

/-- Given a rectangular hall with specified dimensions and cost constraints, 
    prove that the width of the hall is 25 meters. -/
theorem hall_width_proof (length height : ℝ) (cost_per_sqm total_cost : ℝ) 
    (h1 : length = 20)
    (h2 : height = 5)
    (h3 : cost_per_sqm = 20)
    (h4 : total_cost = 19000) :
  ∃ (width : ℝ), 
    total_cost = cost_per_sqm * (length * width + 2 * length * height + 2 * width * height) ∧ 
    width = 25 := by
  sorry

end hall_width_proof_l2756_275648


namespace prob_same_length_is_17_35_l2756_275697

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Set (ℕ × ℕ) := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of elements in set T -/
def total_elements : ℕ := num_sides + num_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ :=
  (num_sides * (num_sides - 1) + num_diagonals * (num_diagonals - 1)) /
  (total_elements * (total_elements - 1))

theorem prob_same_length_is_17_35 : prob_same_length = 17 / 35 := by
  sorry

end prob_same_length_is_17_35_l2756_275697


namespace meeting_point_distance_l2756_275684

/-- 
Given two people starting at opposite ends of a path, prove that they meet
when the slower person has traveled a specific distance.
-/
theorem meeting_point_distance 
  (total_distance : ℝ) 
  (speed_slow : ℝ) 
  (speed_fast : ℝ) 
  (h1 : total_distance = 36)
  (h2 : speed_slow = 3)
  (h3 : speed_fast = 6)
  (h4 : speed_slow > 0)
  (h5 : speed_fast > speed_slow) :
  ∃ (meeting_distance : ℝ), 
    meeting_distance = total_distance * speed_slow / (speed_slow + speed_fast) ∧ 
    meeting_distance = 12 := by
sorry


end meeting_point_distance_l2756_275684


namespace smallest_perimeter_is_108_l2756_275654

/-- Represents a triangle with sides a, b, c and incenter radius r -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  r : ℕ+
  isIsosceles : a = b
  incenterRadius : r = 8

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℕ := t.a.val + t.b.val + t.c.val

/-- Theorem: The smallest possible perimeter of a triangle satisfying the given conditions is 108 -/
theorem smallest_perimeter_is_108 :
  ∀ t : Triangle, perimeter t ≥ 108 :=
sorry

end smallest_perimeter_is_108_l2756_275654


namespace team_transfer_equation_l2756_275625

theorem team_transfer_equation (x : ℤ) : 
  let team_a_initial : ℤ := 37
  let team_b_initial : ℤ := 23
  let team_a_final : ℤ := team_a_initial + x
  let team_b_final : ℤ := team_b_initial - x
  team_a_final = 2 * team_b_final →
  37 + x = 2 * (23 - x) :=
by
  sorry

end team_transfer_equation_l2756_275625


namespace base_conversion_theorem_l2756_275603

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (a b c : ℕ) : ℕ := a * 7^2 + b * 7 + c

/-- Theorem: If 451 in base 7 equals xy in base 10 (where x and y are single digits),
    then (x * y) / 10 = 0.6 -/
theorem base_conversion_theorem (x y : ℕ) (h1 : x < 10) (h2 : y < 10) 
    (h3 : base7ToBase10 4 5 1 = 10 * x + y) : 
    (x * y : ℚ) / 10 = 6 / 10 := by
  sorry

end base_conversion_theorem_l2756_275603


namespace parabola_equation_l2756_275657

/-- A parabola with its focus and a line passing through it -/
structure ParabolaWithLine where
  p : ℝ
  focus : ℝ × ℝ
  line : Set (ℝ × ℝ)
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The conditions of the problem -/
def problem_conditions (P : ParabolaWithLine) : Prop :=
  P.p > 0 ∧
  P.focus = (P.p / 2, 0) ∧
  P.focus ∈ P.line ∧
  P.A ∈ P.line ∧ P.B ∈ P.line ∧
  P.A.1 ^ 2 = 2 * P.p * P.A.2 ∧
  P.B.1 ^ 2 = 2 * P.p * P.B.2 ∧
  ((P.A.1 + P.B.1) / 2, (P.A.2 + P.B.2) / 2) = (3, 2)

/-- The theorem statement -/
theorem parabola_equation (P : ParabolaWithLine) :
  problem_conditions P →
  P.p = 2 ∨ P.p = 4 :=
sorry

end parabola_equation_l2756_275657


namespace existence_equivalence_l2756_275637

/-- Proves the equivalence between the existence of x in [1, 2] satisfying 
    2x^2 - ax + 2 > 0 and a < 4 for any real number a -/
theorem existence_equivalence (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x^2 - a * x + 2 > 0) ↔ a < 4 := by
  sorry

#check existence_equivalence

end existence_equivalence_l2756_275637


namespace gold_coin_distribution_l2756_275639

theorem gold_coin_distribution (x y : ℕ) (k : ℕ) 
  (h1 : x + y = 16) 
  (h2 : x > y) 
  (h3 : x^2 - y^2 = k * (x - y)) : 
  k = 16 :=
sorry

end gold_coin_distribution_l2756_275639


namespace expression_simplification_l2756_275666

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 + 2) 
  (hb : b = Real.sqrt 3 - 2) : 
  (a^2 / (a^2 + 2*a*b + b^2) - a / (a + b)) / (a^2 / (a^2 - b^2) - b / (a - b) - 1) = 2 * Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l2756_275666


namespace gardens_area_difference_l2756_275641

/-- Represents a rectangular garden with length and width -/
structure Garden where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Calculates the usable area of a garden with a path around the perimeter -/
def Garden.usableArea (g : Garden) (pathWidth : ℝ) : ℝ :=
  (g.length - 2 * pathWidth) * (g.width - 2 * pathWidth)

theorem gardens_area_difference : 
  let karlGarden : Garden := { length := 22, width := 50 }
  let makennaGarden : Garden := { length := 30, width := 46 }
  let pathWidth : ℝ := 1
  makennaGarden.usableArea pathWidth - karlGarden.area = 132 := by sorry

end gardens_area_difference_l2756_275641


namespace units_digit_of_product_l2756_275656

theorem units_digit_of_product (a b c : ℕ) : 
  (2^104 * 5^205 * 11^302) % 10 = 0 :=
by sorry

end units_digit_of_product_l2756_275656


namespace square_sum_from_product_and_sum_l2756_275660

theorem square_sum_from_product_and_sum (x y : ℝ) 
  (h1 : x * y = 16) 
  (h2 : x + y = 8) : 
  x^2 + y^2 = 32 := by
sorry

end square_sum_from_product_and_sum_l2756_275660


namespace largest_divisible_digit_l2756_275646

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

def number_with_digit (d : ℕ) : ℕ := 78120 + d

theorem largest_divisible_digit : 
  (∀ d : ℕ, d ≤ 9 → is_divisible_by_6 (number_with_digit d) → d ≤ 6) ∧ 
  is_divisible_by_6 (number_with_digit 6) :=
sorry

end largest_divisible_digit_l2756_275646


namespace min_value_of_f_l2756_275616

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10

/-- The theorem stating the minimum value of the function -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = 13/5 ∧ ∀ (x y : ℝ), f x y ≥ min :=
sorry

end min_value_of_f_l2756_275616


namespace divide_seven_friends_four_teams_l2756_275674

/-- The number of ways to divide n friends among k teams -/
def divideFriends (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: Dividing 7 friends among 4 teams results in 16384 ways -/
theorem divide_seven_friends_four_teams : 
  divideFriends 7 4 = 16384 := by
  sorry

end divide_seven_friends_four_teams_l2756_275674


namespace tenth_term_of_sequence_l2756_275606

theorem tenth_term_of_sequence (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h : ∀ k, S k = k^2 + 2*k) : 
  a 10 = 21 := by
  sorry

end tenth_term_of_sequence_l2756_275606


namespace three_additional_trams_needed_l2756_275644

/-- The number of trams needed to reduce intervals by one-fifth -/
def additional_trams (initial_trams : ℕ) : ℕ :=
  let total_distance := 60
  let initial_interval := total_distance / initial_trams
  let new_interval := initial_interval * 4 / 5
  let new_total_trams := total_distance / new_interval
  new_total_trams - initial_trams

/-- Theorem stating that 3 additional trams are needed -/
theorem three_additional_trams_needed :
  additional_trams 12 = 3 := by
  sorry

#eval additional_trams 12

end three_additional_trams_needed_l2756_275644


namespace marble_selection_probability_l2756_275649

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 2

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 2

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 2

/-- The number of yellow marbles in the bag -/
def yellow_marbles : ℕ := 1

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles + yellow_marbles

/-- The number of marbles to be selected -/
def selected_marbles : ℕ := 3

/-- The probability of selecting one red, one blue, and one green marble -/
def probability_red_blue_green : ℚ := 8 / 35

theorem marble_selection_probability :
  probability_red_blue_green = (red_marbles * blue_marbles * green_marbles : ℚ) / (total_marbles.choose selected_marbles) :=
by sorry

end marble_selection_probability_l2756_275649


namespace square_sum_from_difference_and_product_l2756_275622

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) (h2 : x * y = 16) : x^2 + y^2 = 432 := by
  sorry

end square_sum_from_difference_and_product_l2756_275622


namespace sum_of_divisors_30_l2756_275672

theorem sum_of_divisors_30 : (Finset.filter (· ∣ 30) (Finset.range 31)).sum id = 72 := by
  sorry

end sum_of_divisors_30_l2756_275672


namespace sum_smallest_largest_prime_1_to_50_l2756_275668

theorem sum_smallest_largest_prime_1_to_50 : ∃ (p q : Nat), 
  (p.Prime ∧ q.Prime) ∧ 
  (∀ r, r.Prime → 1 < r ∧ r ≤ 50 → p ≤ r ∧ r ≤ q) ∧ 
  p + q = 49 := by
sorry

end sum_smallest_largest_prime_1_to_50_l2756_275668


namespace x_plus_y_values_l2756_275662

theorem x_plus_y_values (x y : ℝ) : 
  (|x| = 3) → (|y| = 2) → (|x - y| = y - x) → 
  (x + y = -1 ∨ x + y = -5) := by
sorry

end x_plus_y_values_l2756_275662


namespace paper_pallet_ratio_l2756_275679

theorem paper_pallet_ratio (total : ℕ) (towels tissues cups plates : ℕ) : 
  total = 20 → 
  towels = total / 2 → 
  tissues = total / 4 → 
  cups = 1 → 
  plates = total - (towels + tissues + cups) → 
  (plates : ℚ) / total = 1 / 5 := by sorry

end paper_pallet_ratio_l2756_275679


namespace jasmine_swimming_totals_l2756_275605

/-- Jasmine's weekly swimming routine -/
structure SwimmingRoutine where
  monday_laps : ℕ
  tuesday_laps : ℕ
  tuesday_aerobics : ℕ
  wednesday_laps : ℕ
  wednesday_time_per_lap : ℕ
  thursday_laps : ℕ
  friday_laps : ℕ

/-- Calculate total laps and partial time for a given number of weeks -/
def calculate_totals (routine : SwimmingRoutine) (weeks : ℕ) :
  (ℕ × ℕ) :=
  let weekly_laps := routine.monday_laps + routine.tuesday_laps +
                     routine.wednesday_laps + routine.thursday_laps +
                     routine.friday_laps
  let weekly_partial_time := routine.tuesday_aerobics +
                             (routine.wednesday_laps * routine.wednesday_time_per_lap)
  (weekly_laps * weeks, weekly_partial_time * weeks)

theorem jasmine_swimming_totals :
  let routine := SwimmingRoutine.mk 10 15 20 12 2 18 20
  let (total_laps, partial_time) := calculate_totals routine 5
  total_laps = 375 ∧ partial_time = 220 := by sorry

end jasmine_swimming_totals_l2756_275605


namespace inequality_preserved_subtraction_l2756_275685

theorem inequality_preserved_subtraction (a b : ℝ) (h : a < b) : a - 1 < b - 1 := by
  sorry

end inequality_preserved_subtraction_l2756_275685


namespace dave_has_more_cats_l2756_275691

/-- The number of pets owned by Teddy, Ben, and Dave -/
structure PetOwnership where
  teddy_dogs : ℕ
  teddy_cats : ℕ
  ben_dogs : ℕ
  dave_dogs : ℕ
  dave_cats : ℕ

/-- The conditions of the pet ownership problem -/
def pet_problem (p : PetOwnership) : Prop :=
  p.teddy_dogs = 7 ∧
  p.teddy_cats = 8 ∧
  p.ben_dogs = p.teddy_dogs + 9 ∧
  p.dave_dogs = p.teddy_dogs - 5 ∧
  p.teddy_dogs + p.teddy_cats + p.ben_dogs + p.dave_dogs + p.dave_cats = 54

/-- The theorem stating that Dave has 13 more cats than Teddy -/
theorem dave_has_more_cats (p : PetOwnership) (h : pet_problem p) :
  p.dave_cats = p.teddy_cats + 13 := by
  sorry

end dave_has_more_cats_l2756_275691


namespace correct_average_weight_l2756_275683

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) :
  n = 20 ∧ 
  initial_average = 58.4 ∧ 
  misread_weight = 56 ∧ 
  correct_weight = 68 →
  (n : ℝ) * initial_average + (correct_weight - misread_weight) = n * 59 :=
by sorry

end correct_average_weight_l2756_275683


namespace janes_numbers_l2756_275642

def is_between (n : ℕ) (a b : ℕ) : Prop := a ≤ n ∧ n ≤ b

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def satisfies_conditions (n : ℕ) : Prop :=
  is_between n 100 150 ∧
  n % 7 = 0 ∧
  n % 3 ≠ 0 ∧
  sum_of_digits n % 4 = 0

theorem janes_numbers : 
  {n : ℕ | satisfies_conditions n} = {112, 147} := by sorry

end janes_numbers_l2756_275642


namespace expansion_term_count_l2756_275604

/-- The number of terms in the expansion of a product of sums of distinct variables -/
def expansion_terms (x y z : ℕ) : ℕ := x * y * z

/-- The first factor (a+b+c) has 3 terms -/
def factor1_terms : ℕ := 3

/-- The second factor (d+e+f+g) has 4 terms -/
def factor2_terms : ℕ := 4

/-- The third factor (h+i) has 2 terms -/
def factor3_terms : ℕ := 2

theorem expansion_term_count : 
  expansion_terms factor1_terms factor2_terms factor3_terms = 24 := by
  sorry

end expansion_term_count_l2756_275604


namespace smallest_four_digit_number_congruence_l2756_275624

theorem smallest_four_digit_number_congruence (x : ℕ) : 
  (x ≥ 1000 ∧ x < 10000) →
  (3 * x ≡ 9 [ZMOD 18]) →
  (5 * x + 20 ≡ 30 [ZMOD 15]) →
  (3 * x - 4 ≡ 2 * x [ZMOD 35]) →
  x ≥ 1004 :=
by sorry

end smallest_four_digit_number_congruence_l2756_275624


namespace complex_equation_solution_l2756_275607

theorem complex_equation_solution (i : ℂ) (z : ℂ) (h1 : i * i = -1) (h2 : i * z = 1) :
  z = -i := by
  sorry

end complex_equation_solution_l2756_275607


namespace blue_yellow_probability_l2756_275601

/-- The probability of drawing a blue chip first and then a yellow chip without replacement -/
def draw_blue_then_yellow (blue : ℕ) (yellow : ℕ) : ℚ :=
  (blue : ℚ) / (blue + yellow) * yellow / (blue + yellow - 1)

/-- Theorem stating the probability of drawing a blue chip first and then a yellow chip
    without replacement from a bag containing 10 blue chips and 5 yellow chips -/
theorem blue_yellow_probability :
  draw_blue_then_yellow 10 5 = 5 / 21 := by
  sorry

#eval draw_blue_then_yellow 10 5

end blue_yellow_probability_l2756_275601


namespace absolute_value_inequality_l2756_275631

theorem absolute_value_inequality (x : ℝ) : 
  2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ↔ ((-2 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 8)) :=
by sorry

end absolute_value_inequality_l2756_275631


namespace train_speed_calculation_l2756_275695

theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  bridge_length = 265 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_calculation_l2756_275695


namespace factor_theorem_l2756_275610

theorem factor_theorem (p q : ℝ) : 
  (∀ x : ℝ, (x - 3) * (x + 5) = x^2 + p*x + q) → p = 2 := by
  sorry

end factor_theorem_l2756_275610


namespace three_digit_sum_property_l2756_275600

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

def is_triple_digit (n : ℕ) : Prop :=
  ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 100 * a + 10 * a + a

theorem three_digit_sum_property :
  {n : ℕ | is_three_digit n ∧ is_triple_digit (n + sum_of_digits n)} =
  {105, 324, 429, 543, 648, 762, 867, 981} := by sorry

end three_digit_sum_property_l2756_275600


namespace fraction_product_equality_l2756_275633

theorem fraction_product_equality : (2/3)^4 * (1/5) * (3/4) = 4/135 := by
  sorry

end fraction_product_equality_l2756_275633


namespace trip_distance_l2756_275612

theorem trip_distance (total_time hiking_speed canoe_speed hiking_distance : ℝ) 
  (h1 : total_time = 5.5)
  (h2 : hiking_speed = 5)
  (h3 : canoe_speed = 12)
  (h4 : hiking_distance = 27) :
  hiking_distance + (total_time - hiking_distance / hiking_speed) * canoe_speed = 28.2 := by
  sorry

#check trip_distance

end trip_distance_l2756_275612


namespace a_10_ends_with_1000_nines_l2756_275643

def a : ℕ → ℕ
  | 0 => 9
  | (n + 1) => 3 * (a n)^4 + 4 * (a n)^3

def ends_with_nines (n : ℕ) (k : ℕ) : Prop :=
  ∃ m : ℕ, n = m * 10^k + (10^k - 1)

theorem a_10_ends_with_1000_nines : ends_with_nines (a 10) 1000 := by
  sorry

end a_10_ends_with_1000_nines_l2756_275643


namespace p_is_true_q_is_false_p_or_q_is_true_p_and_not_q_is_true_l2756_275647

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem stating that p is true
theorem p_is_true : p := sorry

-- Theorem stating that q is false
theorem q_is_false : ¬q := sorry

-- Theorem stating that the disjunction of p and q is true
theorem p_or_q_is_true : p ∨ q := sorry

-- Theorem stating that the conjunction of p and not q is true
theorem p_and_not_q_is_true : p ∧ ¬q := sorry

end p_is_true_q_is_false_p_or_q_is_true_p_and_not_q_is_true_l2756_275647


namespace sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2756_275617

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
sorry

theorem sum_of_roots_specific_quadratic :
  let f : ℝ → ℝ := λ x => x^2 - 7*x + 2 - 16
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 7) :=
sorry

end sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2756_275617


namespace max_rectangle_area_l2756_275640

/-- The perimeter of the rectangle in feet -/
def perimeter : ℕ := 190

/-- The maximum area of the rectangle in square feet -/
def max_area : ℕ := 2256

/-- A function to calculate the area of a rectangle given one side length -/
def area (x : ℕ) : ℕ := x * (perimeter / 2 - x)

/-- Theorem stating that the maximum area of a rectangle with the given perimeter and integer side lengths is 2256 square feet -/
theorem max_rectangle_area :
  ∀ x : ℕ, x > 0 ∧ x < perimeter / 2 → area x ≤ max_area :=
sorry

end max_rectangle_area_l2756_275640


namespace conors_work_week_l2756_275676

/-- Conor's vegetable chopping problem -/
theorem conors_work_week (eggplants carrots potatoes total : ℕ) 
  (h1 : eggplants = 12)
  (h2 : carrots = 9)
  (h3 : potatoes = 8)
  (h4 : total = 116) : 
  total / (eggplants + carrots + potatoes) = 4 := by
  sorry

#check conors_work_week

end conors_work_week_l2756_275676


namespace min_distance_A_D_l2756_275661

/-- Given points A, B, C, D, E in a metric space, prove that the minimum distance between A and D is 2 units -/
theorem min_distance_A_D (X : Type*) [MetricSpace X] (A B C D E : X) :
  dist A B = 12 →
  dist B C = 7 →
  dist C E = 2 →
  dist E D = 5 →
  ∃ (d : ℝ), d ≥ 2 ∧ ∀ (d' : ℝ), dist A D ≥ d' → d ≤ d' :=
by sorry

end min_distance_A_D_l2756_275661


namespace solution_set_f_greater_than_2_range_of_t_l2756_275609

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x > 2/3 ∨ x < -6} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (7/2)*t} = {t : ℝ | 3/2 ≤ t ∧ t ≤ 2} := by sorry

end solution_set_f_greater_than_2_range_of_t_l2756_275609


namespace multiplication_of_powers_of_ten_l2756_275650

theorem multiplication_of_powers_of_ten : (2 * 10^3) * (8 * 10^3) = 1.6 * 10^7 := by
  sorry

end multiplication_of_powers_of_ten_l2756_275650


namespace route_time_difference_l2756_275689

theorem route_time_difference (x : ℝ) (h : x > 0) : 
  10 / x - 7 / ((1 + 0.4) * x) = 10 / 60 :=
by
  sorry

#check route_time_difference

end route_time_difference_l2756_275689


namespace specific_arithmetic_sequence_common_difference_l2756_275669

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  first_term : ℚ
  last_term : ℚ
  sum : ℚ
  is_arithmetic : Bool

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℚ :=
  sorry

/-- Theorem stating the common difference of the specific arithmetic sequence -/
theorem specific_arithmetic_sequence_common_difference :
  let seq := ArithmeticSequence.mk 3 28 186 true
  common_difference seq = 25 / 11 := by
  sorry

end specific_arithmetic_sequence_common_difference_l2756_275669


namespace bakery_rolls_combinations_l2756_275665

theorem bakery_rolls_combinations :
  let n : ℕ := 8  -- total number of rolls
  let k : ℕ := 4  -- number of roll types
  let remaining : ℕ := n - k  -- remaining rolls after putting one in each category
  (Nat.choose (remaining + k - 1) (k - 1)) = 35 := by
  sorry

end bakery_rolls_combinations_l2756_275665


namespace sum_of_common_x_coords_l2756_275652

/-- Given two congruences modulo 16, find the sum of x-coordinates of common points -/
theorem sum_of_common_x_coords : ∃ (S : Finset ℕ),
  (∀ x ∈ S, ∃ y : ℕ, (y ≡ 5 * x + 2 [ZMOD 16] ∧ y ≡ 11 * x + 12 [ZMOD 16])) ∧
  (∀ x : ℕ, (∃ y : ℕ, y ≡ 5 * x + 2 [ZMOD 16] ∧ y ≡ 11 * x + 12 [ZMOD 16]) → x ∈ S) ∧
  (Finset.sum S id = 10) :=
by sorry

end sum_of_common_x_coords_l2756_275652


namespace calculator_minimum_operations_l2756_275634

/-- Represents the possible operations on the calculator --/
inductive Operation
  | AddOne
  | TimesTwo

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.TimesTwo => n * 2

/-- Checks if a sequence of operations transforms 1 into the target --/
def isValidSequence (ops : List Operation) (target : ℕ) : Prop :=
  ops.foldl applyOperation 1 = target

/-- The theorem to be proved --/
theorem calculator_minimum_operations :
  ∃ (ops : List Operation),
    isValidSequence ops 400 ∧
    ops.length = 10 ∧
    (∀ (other_ops : List Operation),
      isValidSequence other_ops 400 → other_ops.length ≥ 10) := by
  sorry


end calculator_minimum_operations_l2756_275634


namespace polar_cartesian_equivalence_l2756_275629

/-- The curve C in polar coordinates -/
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ - 4 * Real.sin θ

/-- The curve C in Cartesian coordinates -/
def cartesian_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0

/-- Theorem stating the equivalence of polar and Cartesian equations for curve C -/
theorem polar_cartesian_equivalence :
  ∀ (x y ρ θ : ℝ), 
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (polar_equation ρ θ ↔ cartesian_equation x y) := by
  sorry

end polar_cartesian_equivalence_l2756_275629


namespace rosa_phone_calls_l2756_275663

theorem rosa_phone_calls (total_pages : ℝ) (pages_this_week : ℝ) 
  (h1 : total_pages = 18.8) 
  (h2 : pages_this_week = 8.6) : 
  total_pages - pages_this_week = 10.2 := by
sorry

end rosa_phone_calls_l2756_275663


namespace thirtieth_triangular_number_l2756_275667

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number :
  triangular_number 30 = 465 := by
  sorry

end thirtieth_triangular_number_l2756_275667


namespace bicyclist_average_speed_l2756_275688

/-- The average speed of a bicyclist's trip -/
theorem bicyclist_average_speed :
  let total_distance : ℝ := 250
  let first_part_distance : ℝ := 100
  let first_part_speed : ℝ := 20
  let second_part_distance : ℝ := total_distance - first_part_distance
  let second_part_speed : ℝ := 15
  let average_speed : ℝ := total_distance / (first_part_distance / first_part_speed + second_part_distance / second_part_speed)
  average_speed = 250 / (100 / 20 + 150 / 15) :=
by
  sorry

#eval (250 : Float) / ((100 : Float) / 20 + (150 : Float) / 15)

end bicyclist_average_speed_l2756_275688


namespace used_car_percentage_l2756_275645

theorem used_car_percentage (used_price original_price : ℝ) 
  (h1 : used_price = 15000)
  (h2 : original_price = 37500) :
  (used_price / original_price) * 100 = 40 := by
  sorry

end used_car_percentage_l2756_275645
