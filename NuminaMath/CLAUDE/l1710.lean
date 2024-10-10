import Mathlib

namespace quadrilateral_fourth_angle_l1710_171099

theorem quadrilateral_fourth_angle
  (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 = 100)
  (h2 : angle2 = 60)
  (h3 : angle3 = 90)
  (h_sum : angle1 + angle2 + angle3 + angle4 = 360) :
  angle4 = 110 := by
sorry

end quadrilateral_fourth_angle_l1710_171099


namespace largest_unreachable_proof_l1710_171096

/-- The largest integer that cannot be expressed as a non-negative linear combination of 17 and 11 -/
def largest_unreachable : ℕ := 159

/-- The width of the paper in half-inches -/
def paper_width : ℕ := 17

/-- The length of the paper in inches -/
def paper_length : ℕ := 11

/-- A predicate that checks if a natural number can be expressed as a non-negative linear combination of paper_width and paper_length -/
def is_reachable (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * paper_width + b * paper_length

theorem largest_unreachable_proof :
  (∀ n > largest_unreachable, is_reachable n) ∧
  ¬is_reachable largest_unreachable :=
sorry

end largest_unreachable_proof_l1710_171096


namespace area_of_smaller_triangle_l1710_171019

/-- Given an outer equilateral triangle with area 36 square units and an inner equilateral triangle
    with area 4 square units, if the space between these triangles is divided into four congruent
    triangles, then the area of each of these smaller triangles is 8 square units. -/
theorem area_of_smaller_triangle (outer_area inner_area : ℝ) (h1 : outer_area = 36)
    (h2 : inner_area = 4) (h3 : outer_area > inner_area) :
  (outer_area - inner_area) / 4 = 8 :=
by sorry

end area_of_smaller_triangle_l1710_171019


namespace equation_represents_hyperbola_l1710_171067

/-- Given an equation mx^2 - my^2 = n where m and n are real numbers and mn < 0,
    the curve represented by this equation is a hyperbola with foci on the y-axis. -/
theorem equation_represents_hyperbola (m n : ℝ) (h : m * n < 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), m * x^2 - m * y^2 = n ↔ y^2 / a^2 - x^2 / b^2 = 1) :=
sorry

end equation_represents_hyperbola_l1710_171067


namespace quadratic_root_range_l1710_171015

theorem quadratic_root_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    a * x₁^2 - 2*(a+1)*x₁ + (a-1) = 0 ∧
    a * x₂^2 - 2*(a+1)*x₂ + (a-1) = 0 ∧
    x₁ > 2 ∧ x₂ < 2) →
  (0 < a ∧ a < 5) :=
by sorry

end quadratic_root_range_l1710_171015


namespace morio_age_at_michiko_birth_l1710_171028

/-- Proves that Morio's age when Michiko was born is 38 years old -/
theorem morio_age_at_michiko_birth 
  (teresa_current_age : ℕ) 
  (morio_current_age : ℕ) 
  (teresa_age_at_birth : ℕ) 
  (h1 : teresa_current_age = 59) 
  (h2 : morio_current_age = 71) 
  (h3 : teresa_age_at_birth = 26) :
  morio_current_age - (teresa_current_age - teresa_age_at_birth) = 38 :=
by
  sorry


end morio_age_at_michiko_birth_l1710_171028


namespace line_symmetry_l1710_171018

/-- The original line -/
def original_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- The line of symmetry -/
def symmetry_line (x : ℝ) : Prop := x = 1

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to the symmetry_line -/
theorem line_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  original_line x₁ y₁ →
  symmetric_line x₂ y₂ →
  ∃ (x_sym : ℝ),
    symmetry_line x_sym ∧
    x_sym - x₁ = x₂ - x_sym ∧
    y₁ = y₂ :=
sorry

end line_symmetry_l1710_171018


namespace two_face_cards_probability_l1710_171071

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of face cards in a standard deck
def face_cards : ℕ := 12

-- Define the probability of selecting two face cards
def prob_two_face_cards : ℚ := 22 / 442

-- Theorem statement
theorem two_face_cards_probability :
  (face_cards / total_cards) * ((face_cards - 1) / (total_cards - 1)) = prob_two_face_cards := by
  sorry

end two_face_cards_probability_l1710_171071


namespace workshop_production_balance_l1710_171051

/-- Represents the production balance in a workshop --/
theorem workshop_production_balance 
  (total_workers : ℕ) 
  (bolts_per_worker : ℕ) 
  (nuts_per_worker : ℕ) 
  (nuts_per_bolt : ℕ) 
  (x : ℕ) : 
  total_workers = 16 → 
  bolts_per_worker = 1200 → 
  nuts_per_worker = 2000 → 
  nuts_per_bolt = 2 → 
  x ≤ total_workers →
  2 * bolts_per_worker * x = nuts_per_worker * (total_workers - x) := by
  sorry

#check workshop_production_balance

end workshop_production_balance_l1710_171051


namespace simplify_expression_l1710_171046

theorem simplify_expression : (5 + 4 + 6) / 3 - 2 / 3 = 13 / 3 := by
  sorry

end simplify_expression_l1710_171046


namespace tetrahedron_volume_l1710_171033

/-- The volume of a tetrahedron OABC where:
  - Triangle ABC has sides of length 7, 8, and 9
  - A is on the positive x-axis, B on the positive y-axis, and C on the positive z-axis
  - O is the origin (0, 0, 0)
-/
theorem tetrahedron_volume : ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a^2 + b^2 : ℝ) = 49 ∧
  (b^2 + c^2 : ℝ) = 64 ∧
  (c^2 + a^2 : ℝ) = 81 ∧
  (1/6 : ℝ) * a * b * c = 8 * Real.sqrt 11 := by
  sorry

end tetrahedron_volume_l1710_171033


namespace existence_of_a_value_of_a_l1710_171037

-- Define the sets A, B, and C as functions of real numbers
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + 4*a^2 - 3 = 0}
def B : Set ℝ := {x | x^2 - x - 2 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem 1
theorem existence_of_a : ∃ a : ℝ, A a = B ∧ a = 1/2 := by sorry

-- Theorem 2
theorem value_of_a (a : ℝ) : (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) → a = -1 := by sorry

end existence_of_a_value_of_a_l1710_171037


namespace years_between_second_and_third_car_l1710_171003

def year_first_car : ℕ := 1970
def years_between_first_and_second : ℕ := 10
def year_third_car : ℕ := 2000

theorem years_between_second_and_third_car : 
  year_third_car - (year_first_car + years_between_first_and_second) = 20 := by
  sorry

end years_between_second_and_third_car_l1710_171003


namespace sqrt_fourth_power_eq_256_l1710_171088

theorem sqrt_fourth_power_eq_256 (y : ℝ) : (Real.sqrt y) ^ 4 = 256 → y = 16 := by
  sorry

end sqrt_fourth_power_eq_256_l1710_171088


namespace irreducible_polynomial_l1710_171082

/-- A polynomial of the form x^n + 5x^(n-1) + 3 is irreducible over ℤ[X] for any integer n > 1 -/
theorem irreducible_polynomial (n : ℕ) (hn : n > 1) :
  Irreducible (Polynomial.monomial n 1 + Polynomial.monomial (n-1) 5 + Polynomial.monomial 0 3 : Polynomial ℤ) := by
  sorry

end irreducible_polynomial_l1710_171082


namespace gcd_problem_l1710_171023

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) :
  Nat.gcd (Int.natAbs (4 * b^2 + 63 * b + 144)) (Int.natAbs (2 * b + 7)) = 1 := by
  sorry

end gcd_problem_l1710_171023


namespace unique_function_theorem_l1710_171044

-- Define the property that the function must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y + 1) = x + y + 1

-- State the theorem
theorem unique_function_theorem :
  ∃! f : ℝ → ℝ, SatisfiesProperty f ∧ ∀ x : ℝ, f x = x :=
by
  sorry

end unique_function_theorem_l1710_171044


namespace power_inequality_l1710_171025

theorem power_inequality (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  a^m + a^(-m) > a^n + a^(-n) := by
  sorry

end power_inequality_l1710_171025


namespace hawk_breeding_theorem_l1710_171017

/-- Given information about hawk breeding --/
structure HawkBreeding where
  num_kettles : ℕ
  pregnancies_per_kettle : ℕ
  survival_rate : ℚ
  expected_babies : ℕ

/-- Calculate the number of babies yielded per batch before loss --/
def babies_per_batch (h : HawkBreeding) : ℚ :=
  (h.expected_babies : ℚ) / h.survival_rate / ((h.num_kettles * h.pregnancies_per_kettle) : ℚ)

/-- Theorem stating the number of babies yielded per batch --/
theorem hawk_breeding_theorem (h : HawkBreeding) 
  (h_kettles : h.num_kettles = 6)
  (h_pregnancies : h.pregnancies_per_kettle = 15)
  (h_survival : h.survival_rate = 3/4)
  (h_expected : h.expected_babies = 270) :
  babies_per_batch h = 4 := by
  sorry


end hawk_breeding_theorem_l1710_171017


namespace midline_characterization_l1710_171042

/-- Triangle type -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Function to calculate the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Function to check if a point is inside a triangle -/
def is_inside (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Function to check if a point is on the midline of a triangle -/
def on_midline (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem midline_characterization (t : Triangle) (M : ℝ × ℝ) :
  is_inside M t →
  (on_midline M t ↔ area ⟨M, t.A, t.B⟩ = area ⟨M, t.B, t.C⟩ + area ⟨M, t.C, t.A⟩) :=
by sorry

end midline_characterization_l1710_171042


namespace hyperbola_standard_equation_l1710_171032

/-- Given a hyperbola with asymptotes y = ±(2/3)x and real axis length 12,
    its standard equation is either (x²/36) - (y²/16) = 1 or (y²/36) - (x²/16) = 1 -/
theorem hyperbola_standard_equation
  (asymptote_slope : ℝ)
  (real_axis_length : ℝ)
  (h1 : asymptote_slope = 2/3)
  (h2 : real_axis_length = 12) :
  (∃ (x y : ℝ), x^2/36 - y^2/16 = 1) ∨
  (∃ (x y : ℝ), y^2/36 - x^2/16 = 1) :=
sorry

end hyperbola_standard_equation_l1710_171032


namespace six_grades_assignments_l1710_171009

/-- The number of ways to assign n grades, where grades are 2, 3, or 4, and no two consecutive 2s are allowed. -/
def gradeAssignments (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | n + 2 => 2 * gradeAssignments (n + 1) + 2 * gradeAssignments n

/-- The theorem stating that there are 448 ways to assign 6 grades under the given conditions. -/
theorem six_grades_assignments : gradeAssignments 6 = 448 := by
  sorry

end six_grades_assignments_l1710_171009


namespace diophantine_equation_solutions_l1710_171016

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    (x = -7 ∧ y = -99) ∨ (x = -1 ∧ y = -9) ∨ (x = 1 ∧ y = 5) ∨ (x = 7 ∧ y = -97) :=
by sorry

end diophantine_equation_solutions_l1710_171016


namespace complex_modulus_equality_l1710_171089

theorem complex_modulus_equality (t : ℝ) :
  t > 0 → (Complex.abs (8 + 2 * t * Complex.I) = 14 ↔ t = Real.sqrt 33) := by
  sorry

end complex_modulus_equality_l1710_171089


namespace distance_center_to_line_l1710_171045

/-- The distance from the center of the unit circle to a line ax + by + c = 0, 
    where a^2 + b^2 ≠ 4c^2 and c ≠ 0, is 1/2. -/
theorem distance_center_to_line (a b c : ℝ) 
  (h1 : a^2 + b^2 ≠ 4 * c^2) (h2 : c ≠ 0) : 
  let d := |c| / Real.sqrt (a^2 + b^2)
  d = 1/2 := by sorry

end distance_center_to_line_l1710_171045


namespace pythagorean_equivalent_l1710_171086

theorem pythagorean_equivalent (t : ℝ) : 
  (∃ (a b : ℚ), (2 * t) / (1 + t^2) = a ∧ (1 - t^2) / (1 + t^2) = b) → 
  ∃ (q : ℚ), (t : ℝ) = q :=
by sorry

end pythagorean_equivalent_l1710_171086


namespace total_travel_time_l1710_171001

theorem total_travel_time (total_distance : ℝ) (initial_time : ℝ) (lunch_time : ℝ) 
  (h1 : total_distance = 200)
  (h2 : initial_time = 1)
  (h3 : lunch_time = 1)
  (h4 : initial_time * 4 * total_distance / 4 = total_distance) :
  initial_time + lunch_time + (total_distance - total_distance / 4) / (total_distance / 4 / initial_time) = 5 := by
  sorry

end total_travel_time_l1710_171001


namespace right_triangle_vector_k_l1710_171062

-- Define a right-angled triangle ABC
structure RightTriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Define the theorem
theorem right_triangle_vector_k (k : ℝ) (triangle : RightTriangleABC) 
  (hBA : triangle.B.1 - triangle.A.1 = k ∧ triangle.B.2 - triangle.A.2 = 1)
  (hBC : triangle.B.1 - triangle.C.1 = 2 ∧ triangle.B.2 - triangle.C.2 = 3) :
  k = 5 := by
  sorry

end right_triangle_vector_k_l1710_171062


namespace nonagon_diagonal_intersection_probability_l1710_171083

/-- A regular nonagon is a 9-sided polygon with all sides and angles equal -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- The probability that two randomly chosen diagonals of a regular nonagon intersect inside the nonagon -/
def intersectionProbability (n : RegularNonagon) : ℚ :=
  6 / 13

/-- Theorem: The probability that two randomly chosen diagonals of a regular nonagon 
    intersect inside the nonagon is 6/13 -/
theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) : 
  intersectionProbability n = 6 / 13 := by
  sorry


end nonagon_diagonal_intersection_probability_l1710_171083


namespace monomial_coefficient_and_degree_l1710_171005

/-- Represents a monomial with coefficient and variables -/
structure Monomial where
  coeff : ℚ
  vars : List (Char × ℕ)

/-- Calculate the degree of a monomial -/
def monomialDegree (m : Monomial) : ℕ :=
  m.vars.foldl (fun acc (_, exp) => acc + exp) 0

/-- The monomial -2/3 * a * b^2 -/
def mono : Monomial :=
  { coeff := -2/3
  , vars := [('a', 1), ('b', 2)] }

theorem monomial_coefficient_and_degree :
  mono.coeff = -2/3 ∧ monomialDegree mono = 3 := by
  sorry

end monomial_coefficient_and_degree_l1710_171005


namespace geometric_sequence_minimum_value_l1710_171007

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_relation : a 7 = a 6 + 2 * a 5)
  (h_exist : ∃ m n : ℕ, Real.sqrt (a m * a n) = 2 * Real.sqrt 2 * a 1) :
  (∃ m n : ℕ, 1 / m + 4 / n = 11 / 6) ∧
  (∀ m n : ℕ, 1 / m + 4 / n ≥ 11 / 6) :=
sorry

end geometric_sequence_minimum_value_l1710_171007


namespace apples_to_friends_l1710_171084

theorem apples_to_friends (initial_apples : ℕ) (apples_left : ℕ) (apples_to_teachers : ℕ) (apples_eaten : ℕ) :
  initial_apples = 25 →
  apples_left = 3 →
  apples_to_teachers = 16 →
  apples_eaten = 1 →
  initial_apples - apples_left - apples_to_teachers - apples_eaten = 5 :=
by sorry

end apples_to_friends_l1710_171084


namespace cheerful_team_tasks_l1710_171090

theorem cheerful_team_tasks (correct_points : ℕ) (incorrect_points : ℕ) (total_points : ℤ) (max_tasks : ℕ) :
  correct_points = 9 →
  incorrect_points = 5 →
  total_points = 57 →
  max_tasks = 15 →
  ∃ (x y : ℕ),
    x + y ≤ max_tasks ∧
    (x : ℤ) * correct_points - (y : ℤ) * incorrect_points = total_points ∧
    x = 8 :=
by sorry

end cheerful_team_tasks_l1710_171090


namespace least_number_of_tiles_l1710_171075

def room_length : ℕ := 672
def room_width : ℕ := 432

theorem least_number_of_tiles (length : ℕ) (width : ℕ) 
  (h1 : length = room_length) (h2 : width = room_width) : 
  ∃ (tile_size : ℕ), tile_size > 0 ∧ 
  length % tile_size = 0 ∧ 
  width % tile_size = 0 ∧
  (length / tile_size) * (width / tile_size) = 126 := by
  sorry

end least_number_of_tiles_l1710_171075


namespace water_volume_for_four_balls_l1710_171058

/-- The volume of water needed to cover four touching balls in a cylinder -/
theorem water_volume_for_four_balls (r ball_radius container_radius : ℝ) 
  (h_ball_radius : ball_radius = 0.5)
  (h_container_radius : container_radius = 1) :
  let water_height := container_radius + ball_radius
  let cylinder_volume := π * container_radius^2 * water_height
  let ball_volume := (4/3) * π * ball_radius^3
  cylinder_volume - 4 * ball_volume = (2/3) * π := by sorry

end water_volume_for_four_balls_l1710_171058


namespace pulled_pork_sandwiches_l1710_171094

def total_sauce : ℚ := 5
def burger_sauce : ℚ := 1/4
def sandwich_sauce : ℚ := 1/6
def num_burgers : ℕ := 8

theorem pulled_pork_sandwiches :
  ∃ (n : ℕ), n * sandwich_sauce + num_burgers * burger_sauce = total_sauce ∧ n = 18 :=
by sorry

end pulled_pork_sandwiches_l1710_171094


namespace roots_equation_result_l1710_171061

theorem roots_equation_result (γ δ : ℝ) : 
  γ^2 - 3*γ - 2 = 0 → δ^2 - 3*δ - 2 = 0 → 7*γ^4 + 10*δ^3 = 1363 := by
  sorry

end roots_equation_result_l1710_171061


namespace line_through_circle_centers_l1710_171073

/-- Given two circles that pass through (1, 1), prove the equation of the line through their centers -/
theorem line_through_circle_centers (D₁ E₁ D₂ E₂ : ℝ) : 
  (1^2 + 1^2 + D₁*1 + E₁*1 + 3 = 0) →
  (1^2 + 1^2 + D₂*1 + E₂*1 + 3 = 0) →
  ∃ (k : ℝ), ∀ (x y : ℝ), (x = D₁ ∧ y = E₁) ∨ (x = D₂ ∧ y = E₂) → x + y + 5 = k := by
  sorry

#check line_through_circle_centers

end line_through_circle_centers_l1710_171073


namespace digit_before_y_l1710_171004

/-- Given a number of the form xy86038 where x and y are single digits,
    if y = 3 and the number is divisible by 11,
    then x = 6 -/
theorem digit_before_y (x y : ℕ) : 
  y = 3 →
  x < 10 →
  y < 10 →
  (x * 1000000 + y * 100000 + 86038) % 11 = 0 →
  (∀ z < y, (x * 1000000 + z * 100000 + 86038) % 11 ≠ 0) →
  x = 6 := by
sorry

end digit_before_y_l1710_171004


namespace increasing_sequences_with_divisibility_property_l1710_171043

theorem increasing_sequences_with_divisibility_property :
  ∃ (a b : ℕ → ℕ), 
    (∀ n : ℕ, a n < a (n + 1)) ∧ 
    (∀ n : ℕ, b n < b (n + 1)) ∧
    (∀ n : ℕ, (a n * (a n + 1)) ∣ (b n ^ 2 + 1)) :=
by
  let a : ℕ → ℕ := λ n => (2^(2*n) + 1)^2
  let b : ℕ → ℕ := λ n => 2^(n*(2^(2*n) + 1)) + (2^(2*n) + 1)^2 * (2^(n*(2^(2*n)+1)) - (2^(2*n) + 1))
  sorry

end increasing_sequences_with_divisibility_property_l1710_171043


namespace census_survey_is_D_census_suitability_criterion_l1710_171065

-- Define the survey options
inductive SurveyOption
| A : SurveyOption  -- West Lake Longjing tea quality
| B : SurveyOption  -- Xiaoshan TV station viewership
| C : SurveyOption  -- Xiaoshan people's happiness index
| D : SurveyOption  -- Classmates' health status

-- Define the property of being suitable for a census
def SuitableForCensus (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.D => True
  | _ => False

-- Define the property of having a small quantity of subjects
def HasSmallQuantity (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.D => True
  | _ => False

-- Theorem stating that the survey suitable for a census is option D
theorem census_survey_is_D :
  ∀ (option : SurveyOption),
    SuitableForCensus option ↔ option = SurveyOption.D :=
  sorry

-- Theorem stating that a survey is suitable for a census if and only if it has a small quantity of subjects
theorem census_suitability_criterion :
  ∀ (option : SurveyOption),
    SuitableForCensus option ↔ HasSmallQuantity option :=
  sorry

end census_survey_is_D_census_suitability_criterion_l1710_171065


namespace heather_walking_distance_l1710_171012

/-- The total distance Heather walked at the county fair -/
theorem heather_walking_distance :
  let car_to_entrance : ℚ := 0.3333333333333333
  let entrance_to_rides : ℚ := 0.3333333333333333
  let rides_to_car : ℚ := 0.08333333333333333
  car_to_entrance + entrance_to_rides + rides_to_car = 0.75
:= by sorry

end heather_walking_distance_l1710_171012


namespace cherry_price_level_6_l1710_171091

noncomputable def cherryPrice (a b x : ℝ) : ℝ := Real.exp (a * x + b)

theorem cherry_price_level_6 (a b : ℝ) :
  (cherryPrice a b 1 / cherryPrice a b 5 = 3) →
  (cherryPrice a b 3 = 60) →
  ∃ ε > 0, |cherryPrice a b 6 - 170| < ε := by
sorry

end cherry_price_level_6_l1710_171091


namespace revenue_maximized_at_13_l1710_171052

/-- Revenue function for book sales -/
def R (p : ℝ) : ℝ := p * (130 - 5 * p)

/-- Theorem stating that the revenue is maximized at p = 13 -/
theorem revenue_maximized_at_13 :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 26 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 26 → R p ≥ R q ∧
  p = 13 :=
sorry

end revenue_maximized_at_13_l1710_171052


namespace function_inequality_l1710_171040

/-- Given a function f(x) = x - 1 - ln x, if f(x) ≥ kx - 2 for all x > 0, 
    then k ≤ 1 - 1/e² -/
theorem function_inequality (k : ℝ) : 
  (∀ x > 0, x - 1 - Real.log x ≥ k * x - 2) → k ≤ 1 - 1 / Real.exp 2 := by
  sorry

end function_inequality_l1710_171040


namespace integer_ratio_problem_l1710_171038

theorem integer_ratio_problem (a b : ℤ) : 
  1996 * a + b / 96 = a + b → b / a = 2016 ∨ a / b = 1 / 2016 := by
  sorry

end integer_ratio_problem_l1710_171038


namespace not_always_equal_to_self_l1710_171093

-- Define the ❤ operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Theorem stating that the statement "x ❤ 0 = x for all x" is false
theorem not_always_equal_to_self : ¬ ∀ x : ℝ, heartsuit x 0 = x := by
  sorry

end not_always_equal_to_self_l1710_171093


namespace max_common_chord_length_l1710_171060

-- Define the circles
def circle1 (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x + 2*a*y + 2*a^2 - 1 = 0

def circle2 (b : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*b*x + 2*b*y + 2*b^2 - 2 = 0

-- Define the common chord
def commonChord (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | circle1 a p.1 p.2 ∧ circle2 b p.1 p.2}

-- Theorem statement
theorem max_common_chord_length (a b : ℝ) :
  ∃ (l : ℝ), l = 2 ∧ ∀ (p q : ℝ × ℝ), p ∈ commonChord a b → q ∈ commonChord a b →
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ l :=
by sorry

end max_common_chord_length_l1710_171060


namespace complement_of_A_in_U_l1710_171087

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≤ 4}

-- Define set A
def A : Set ℝ := {x | |x + 1| ≤ 1}

-- State the theorem
theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end complement_of_A_in_U_l1710_171087


namespace newspaper_expense_difference_is_142_l1710_171077

/-- Calculates the difference in annual newspaper expenses between Juanita and Grant -/
def newspaper_expense_difference : ℝ :=
  let grant_base_cost : ℝ := 200
  let grant_discount_rate : ℝ := 0.1
  let juanita_mon_wed_price : ℝ := 0.5
  let juanita_thu_sat_price : ℝ := 0.75
  let juanita_sun_price : ℝ := 2.5
  let juanita_sun_coupon : ℝ := 0.25
  let weeks_per_year : ℕ := 52
  let months_per_year : ℕ := 12

  let grant_annual_cost : ℝ := grant_base_cost * (1 - grant_discount_rate)
  let juanita_mon_wed_annual : ℝ := 3 * juanita_mon_wed_price * weeks_per_year
  let juanita_thu_sat_annual : ℝ := 3 * juanita_thu_sat_price * weeks_per_year
  let juanita_sun_annual : ℝ := juanita_sun_price * weeks_per_year - juanita_sun_coupon * months_per_year
  let juanita_annual_cost : ℝ := juanita_mon_wed_annual + juanita_thu_sat_annual + juanita_sun_annual

  juanita_annual_cost - grant_annual_cost

theorem newspaper_expense_difference_is_142 :
  newspaper_expense_difference = 142 := by
  sorry

end newspaper_expense_difference_is_142_l1710_171077


namespace simple_interest_calculation_l1710_171011

/-- Calculate simple interest -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Proof of simple interest calculation -/
theorem simple_interest_calculation :
  let principal : ℚ := 15000
  let rate : ℚ := 6
  let time : ℚ := 3
  simple_interest principal rate time = 2700 := by
  sorry

end simple_interest_calculation_l1710_171011


namespace solution_set_of_f_less_than_two_range_of_m_when_f_geq_m_squared_l1710_171050

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| + |x|

-- Theorem for part (1)
theorem solution_set_of_f_less_than_two (m : ℝ) (h : f 1 m = 1) :
  {x : ℝ | f x m < 2} = Set.Ioo (-1/2) (3/2) := by sorry

-- Theorem for part (2)
theorem range_of_m_when_f_geq_m_squared :
  {m : ℝ | ∀ x, f x m ≥ m^2} = Set.Icc (-1) 1 := by sorry

end solution_set_of_f_less_than_two_range_of_m_when_f_geq_m_squared_l1710_171050


namespace evaluate_expression_l1710_171076

theorem evaluate_expression : (-2 : ℤ) ^ (4^2) + 2^(4^2) = 2^17 := by
  sorry

end evaluate_expression_l1710_171076


namespace penalty_kicks_count_l1710_171010

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) : 
  total_players = 25 → goalies = 4 → total_players * goalies - goalies^2 = 96 :=
by
  sorry

end penalty_kicks_count_l1710_171010


namespace max_value_theorem_l1710_171034

/-- A function that checks if three numbers can form a triangle with non-zero area -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if a set of eleven consecutive integers contains a triangle-forming trio -/
def has_triangle_trio (start : ℕ) : Prop :=
  ∃ i j k, i < j ∧ j < k ∧ k < start + 11 ∧ can_form_triangle (start + i) (start + j) (start + k)

/-- The theorem stating that 499 is the maximum value satisfying the condition -/
theorem max_value_theorem : 
  (∀ start : ℕ, 5 ≤ start ∧ start ≤ 489 → has_triangle_trio start) ∧
  ¬(∀ start : ℕ, 5 ≤ start ∧ start ≤ 490 → has_triangle_trio start) :=
sorry

end max_value_theorem_l1710_171034


namespace function_composition_fraction_l1710_171054

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_fraction :
  f (g (f 3)) / g (f (g 3)) = 59 / 19 := by sorry

end function_composition_fraction_l1710_171054


namespace line_intercepts_sum_l1710_171047

/-- Proves that for the line 3x + 5y + d = 0, if the sum of its x-intercept and y-intercept is 16, then d = -30 -/
theorem line_intercepts_sum (d : ℝ) : 
  (∃ (x y : ℝ), 3 * x + 5 * y + d = 0 ∧ x + y = 16) → d = -30 := by
sorry

end line_intercepts_sum_l1710_171047


namespace dog_fur_objects_l1710_171053

theorem dog_fur_objects (burrs : ℕ) (ticks : ℕ) : 
  burrs = 12 → ticks = 6 * burrs → burrs + ticks = 84 := by
  sorry

end dog_fur_objects_l1710_171053


namespace sector_area_l1710_171031

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (area : ℝ) : 
  arc_length = 3 → central_angle = 1 → area = (arc_length * arc_length) / (2 * central_angle) → area = 9/2 := by
  sorry

end sector_area_l1710_171031


namespace always_uninfected_cell_l1710_171048

/-- Represents a square grid of cells -/
structure Grid (n : ℕ) where
  side_length : ℕ
  cells : Fin n → Fin n → Bool

/-- Represents the state of infection in the grid -/
structure InfectionState (n : ℕ) where
  grid : Grid n
  infected_cells : Set (Fin n × Fin n)

/-- A function to determine if a cell can be infected based on its neighbors -/
def can_be_infected (state : InfectionState n) (cell : Fin n × Fin n) : Bool :=
  sorry

/-- The perimeter of the infected region -/
def infected_perimeter (state : InfectionState n) : ℕ :=
  sorry

/-- Theorem stating that there will always be at least one uninfected cell -/
theorem always_uninfected_cell (n : ℕ) (initial_state : InfectionState n) :
  ∃ (cell : Fin n × Fin n), cell ∉ initial_state.infected_cells ∧
    ∀ (final_state : InfectionState n),
      (∀ (c : Fin n × Fin n), c ∉ initial_state.infected_cells →
        (c ∈ final_state.infected_cells → can_be_infected initial_state c)) →
      cell ∉ final_state.infected_cells :=
  sorry

end always_uninfected_cell_l1710_171048


namespace semicircle_area_comparison_l1710_171030

theorem semicircle_area_comparison : ∀ (short_side long_side : ℝ),
  short_side = 8 →
  long_side = 12 →
  let large_semicircle_area := π * (long_side / 2)^2
  let small_semicircle_area := π * (short_side / 2)^2
  large_semicircle_area = small_semicircle_area * 2.25 :=
by sorry

end semicircle_area_comparison_l1710_171030


namespace max_value_of_expression_l1710_171066

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  ab + bc + 2*c*a ≤ 9/2 ∧ ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧ ab + bc + 2*c*a = 9/2 :=
sorry

end max_value_of_expression_l1710_171066


namespace javier_has_four_children_l1710_171002

/-- The number of children Javier has -/
def num_children : ℕ :=
  let total_legs : ℕ := 22
  let num_dogs : ℕ := 2
  let num_cats : ℕ := 1
  let javier_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let cat_legs : ℕ := 4
  (total_legs - (num_dogs * dog_legs + num_cats * cat_legs + javier_legs)) / 2

theorem javier_has_four_children : num_children = 4 := by
  sorry

end javier_has_four_children_l1710_171002


namespace count_integer_pairs_verify_count_l1710_171085

/-- The number of positive integer pairs (b,s) satisfying log₄(b²⁰s¹⁹⁰) = 4012 -/
theorem count_integer_pairs : Nat := by sorry

/-- Verifies that the count is correct -/
theorem verify_count : count_integer_pairs = 210 := by sorry

end count_integer_pairs_verify_count_l1710_171085


namespace rectangle_length_is_three_times_width_l1710_171029

/-- Represents the construction of a large square from six identical smaller squares and a rectangle -/
structure SquareConstruction where
  /-- Side length of each small square -/
  s : ℝ
  /-- Assertion that s is positive -/
  s_pos : 0 < s

/-- The length of the rectangle in the construction -/
def rectangleLength (c : SquareConstruction) : ℝ := 3 * c.s

/-- The width of the rectangle in the construction -/
def rectangleWidth (c : SquareConstruction) : ℝ := c.s

/-- The theorem stating that the length of the rectangle is 3 times its width -/
theorem rectangle_length_is_three_times_width (c : SquareConstruction) :
  rectangleLength c = 3 * rectangleWidth c := by
  sorry

end rectangle_length_is_three_times_width_l1710_171029


namespace sum_of_distinct_prime_factors_l1710_171095

def n : ℕ := 245700

theorem sum_of_distinct_prime_factors :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id : ℕ)
    = 30 := by sorry

end sum_of_distinct_prime_factors_l1710_171095


namespace son_work_time_l1710_171056

-- Define the work rates
def man_rate : ℚ := 1 / 10
def combined_rate : ℚ := 1 / 5

-- Define the son's work rate
def son_rate : ℚ := combined_rate - man_rate

-- Theorem to prove
theorem son_work_time :
  son_rate = 1 / 10 ∧ (1 / son_rate : ℚ) = 10 :=
by sorry

end son_work_time_l1710_171056


namespace decimal_to_fraction_l1710_171072

def repeating_decimal (a b : ℕ) : ℚ := (a : ℚ) / (99 : ℚ) + (b : ℚ) / (100 : ℚ)

theorem decimal_to_fraction :
  ∃ (n d : ℕ), d ≠ 0 ∧ 
  repeating_decimal 36 0 = (n : ℚ) / (d : ℚ) ∧
  ∀ (n' d' : ℕ), d' ≠ 0 → repeating_decimal 36 0 = (n' : ℚ) / (d' : ℚ) → d ≤ d' ∧
  d = 11 :=
sorry

end decimal_to_fraction_l1710_171072


namespace twenty_nine_impossible_l1710_171063

/-- Represents the score for a test with 10 questions. -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  sum_is_ten : correct + unanswered + incorrect = 10

/-- Calculates the total score for a given TestScore. -/
def totalScore (ts : TestScore) : Nat :=
  3 * ts.correct + ts.unanswered

/-- Theorem stating that 29 is not a possible total score. -/
theorem twenty_nine_impossible : ¬∃ (ts : TestScore), totalScore ts = 29 := by
  sorry

end twenty_nine_impossible_l1710_171063


namespace manuscript_cost_calculation_l1710_171020

def manuscript_typing_cost (first_typing_rate : ℕ) (revision_rate : ℕ) 
  (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) : ℕ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let first_typing_cost := total_pages * first_typing_rate
  let revision_cost_once := pages_revised_once * revision_rate
  let revision_cost_twice := pages_revised_twice * (2 * revision_rate)
  first_typing_cost + revision_cost_once + revision_cost_twice

theorem manuscript_cost_calculation :
  manuscript_typing_cost 5 4 100 30 20 = 780 := by
  sorry

end manuscript_cost_calculation_l1710_171020


namespace function_upper_bound_l1710_171069

theorem function_upper_bound 
  (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, f x ≥ 0)
  (h2 : f 1 = 1)
  (h3 : ∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≤ f x₁ + f x₂) :
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
by
  sorry


end function_upper_bound_l1710_171069


namespace pizza_expense_proof_l1710_171006

/-- Proves that given a total expense of $465 on pizzas in May (31 days),
    and assuming equal daily consumption, the daily expense on pizzas is $15. -/
theorem pizza_expense_proof (total_expense : ℕ) (days_in_may : ℕ) (daily_expense : ℕ) :
  total_expense = 465 →
  days_in_may = 31 →
  daily_expense * days_in_may = total_expense →
  daily_expense = 15 := by
sorry

end pizza_expense_proof_l1710_171006


namespace max_value_cos_sin_l1710_171097

theorem max_value_cos_sin : 
  ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 ∧ 
  ∃ y : ℝ, 3 * Real.cos y + 4 * Real.sin y = 5 := by
  sorry

end max_value_cos_sin_l1710_171097


namespace tom_seashells_l1710_171074

/-- The number of broken seashells Tom found -/
def broken_seashells : ℕ := 4

/-- The number of unbroken seashells Tom found -/
def unbroken_seashells : ℕ := 3

/-- The total number of seashells Tom found -/
def total_seashells : ℕ := broken_seashells + unbroken_seashells

theorem tom_seashells : total_seashells = 7 := by
  sorry

end tom_seashells_l1710_171074


namespace train_length_calculation_l1710_171055

theorem train_length_calculation (pole_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  pole_time = 50 →
  platform_length = 500 →
  platform_time = 100 →
  ∃ (train_length : ℝ) (train_speed : ℝ),
    train_length = train_speed * pole_time ∧
    train_length + platform_length = train_speed * platform_time ∧
    train_length = 500 :=
by sorry

end train_length_calculation_l1710_171055


namespace upstream_downstream_time_ratio_l1710_171041

/-- The ratio of upstream to downstream swimming time -/
theorem upstream_downstream_time_ratio 
  (swim_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : swim_speed = 9) 
  (h2 : stream_speed = 3) : 
  (swim_speed - stream_speed)⁻¹ / (swim_speed + stream_speed)⁻¹ = 2 := by
  sorry

end upstream_downstream_time_ratio_l1710_171041


namespace stage_25_l1710_171026

/-- Represents the number of toothpicks in a stage of the triangle pattern. -/
def toothpicks (n : ℕ) : ℕ := 3 * n

/-- The triangle pattern starts with 1 toothpick per side in Stage 1. -/
axiom stage_one : toothpicks 1 = 3

/-- Each stage adds one toothpick to each side of the triangle. -/
axiom stage_increase (n : ℕ) : toothpicks (n + 1) = toothpicks n + 3

/-- The number of toothpicks in the 25th stage is 75. -/
theorem stage_25 : toothpicks 25 = 75 := by sorry

end stage_25_l1710_171026


namespace basketball_scores_l1710_171098

/-- The number of different total point scores for a basketball player who made 7 baskets,
    each worth either 2 or 3 points. -/
def differentScores : ℕ := by sorry

theorem basketball_scores :
  let totalBaskets : ℕ := 7
  let twoPointValue : ℕ := 2
  let threePointValue : ℕ := 3
  differentScores = 8 := by sorry

end basketball_scores_l1710_171098


namespace S_minimum_at_n_min_l1710_171027

/-- The sequence a_n with general term 2n - 49 -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

/-- The value of n for which S_n reaches its minimum -/
def n_min : ℕ := 24

theorem S_minimum_at_n_min :
  ∀ k : ℕ, k ≠ 0 → S n_min ≤ S k :=
sorry

end S_minimum_at_n_min_l1710_171027


namespace absolute_value_inequality_solution_set_l1710_171078

theorem absolute_value_inequality_solution_set :
  {x : ℝ | 3 ≤ |5 - 2*x| ∧ |5 - 2*x| < 9} = 
  {x : ℝ | -2 < x ∧ x ≤ 1} ∪ {x : ℝ | 4 ≤ x ∧ x < 7} := by
  sorry

end absolute_value_inequality_solution_set_l1710_171078


namespace complement_of_A_l1710_171014

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem complement_of_A (A B : Set ℕ) 
  (h1 : A ∪ B = {1, 2, 3, 4, 5})
  (h2 : A ∩ B = {3, 4, 5}) :
  Aᶜ = {6} := by
  sorry

end complement_of_A_l1710_171014


namespace triangle_property_l1710_171057

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  2 * b * Real.cos A = c * Real.cos A + a * Real.cos C ∧
  a = 4 →
  A = π / 3 ∧ 
  (∀ (b' c' : ℝ), b' > 0 → c' > 0 → 
    4 * 4 = b' * b' + c' * c' - b' * c' → 
    1/2 * b' * c' * Real.sin A ≤ 4 * Real.sqrt 3) :=
by sorry

end triangle_property_l1710_171057


namespace symmetry_line_is_correct_l1710_171079

/-- The line of symmetry between two circles -/
def line_of_symmetry (c1 c2 : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  fun p => ∃ (q : ℝ × ℝ), c1 q ∧ c2 (2 * p.1 - q.1, 2 * p.2 - q.2)

/-- First circle: x^2 + y^2 = 9 -/
def circle1 : ℝ × ℝ → Prop :=
  fun p => p.1^2 + p.2^2 = 9

/-- Second circle: x^2 + y^2 - 4x + 4y - 1 = 0 -/
def circle2 : ℝ × ℝ → Prop :=
  fun p => p.1^2 + p.2^2 - 4*p.1 + 4*p.2 - 1 = 0

/-- The equation of the line of symmetry: x - y - 2 = 0 -/
def symmetry_line : ℝ × ℝ → Prop :=
  fun p => p.1 - p.2 - 2 = 0

theorem symmetry_line_is_correct : 
  line_of_symmetry circle1 circle2 = symmetry_line :=
sorry

end symmetry_line_is_correct_l1710_171079


namespace quadratic_shift_l1710_171092

/-- The original quadratic function -/
def f (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 - b * x + 3

/-- The mistakenly drawn function -/
def g (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + 3

/-- The shifted original function -/
def f_shifted (b : ℝ) (x : ℝ) : ℝ := f b (x + 6)

theorem quadratic_shift (b : ℝ) : 
  (∀ x, g b x = f_shifted b x) → b = 12 := by
  sorry

end quadratic_shift_l1710_171092


namespace inequality_proof_l1710_171035

theorem inequality_proof (x y : ℝ) 
  (h : x^2 + x*y + y^2 = (x + y)^2 - x*y ∧ 
       x^2 + x*y + y^2 = (x + y - Real.sqrt (x*y)) * (x + y + Real.sqrt (x*y))) : 
  x + y + Real.sqrt (x*y) ≤ 3*(x + y - Real.sqrt (x*y)) := by
sorry

end inequality_proof_l1710_171035


namespace congruence_remainders_l1710_171049

theorem congruence_remainders (x : ℤ) 
  (h1 : x ≡ 25 [ZMOD 35])
  (h2 : x ≡ 31 [ZMOD 42]) :
  (x ≡ 10 [ZMOD 15]) ∧ (x ≡ 13 [ZMOD 18]) := by
  sorry

end congruence_remainders_l1710_171049


namespace intersection_A_complement_B_l1710_171000

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x > -1}

-- Define set B
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x ≤ 2} := by sorry

end intersection_A_complement_B_l1710_171000


namespace ratio_chain_l1710_171008

theorem ratio_chain (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 2)
  : e / a = 1 / 17.5 := by
  sorry

end ratio_chain_l1710_171008


namespace simplify_expression_constant_sum_l1710_171022

/-- Given expressions for A and B in terms of a and b -/
def A (a b : ℝ) : ℝ := 2 * a^2 + a * b - 2 * b - 1

/-- Given expressions for A and B in terms of a and b -/
def B (a b : ℝ) : ℝ := -a^2 + a * b - 2

/-- Theorem 1: Simplification of 3A - (2A - 2B) -/
theorem simplify_expression (a b : ℝ) :
  3 * A a b - (2 * A a b - 2 * B a b) = 3 * a * b - 2 * b - 5 := by sorry

/-- Theorem 2: Value of a when A + 2B is constant for any b -/
theorem constant_sum (a : ℝ) :
  (∀ b : ℝ, ∃ k : ℝ, A a b + 2 * B a b = k) → a = 2 / 3 := by sorry

end simplify_expression_constant_sum_l1710_171022


namespace solve_equation_l1710_171021

theorem solve_equation (Q : ℝ) : (Q^3)^(1/2) = 9 * 729^(1/6) → Q = 9 := by
  sorry

end solve_equation_l1710_171021


namespace triangle_problem_l1710_171068

/-- Given an acute triangle ABC with collinear vectors m and n, prove B = π/6 and a + c = 2 + √3 -/
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) : 
  0 < B → B < π/2 →  -- B is acute
  (2 * Real.sin (A + C)) * (2 * Real.cos (B/2)^2 - 1) = Real.sqrt 3 * Real.cos (2*B) →  -- m and n are collinear
  b = 1 →  -- given condition
  a * c * Real.sin B / 2 = Real.sqrt 3 / 2 →  -- area condition
  (B = π/6 ∧ a + c = 2 + Real.sqrt 3) := by
  sorry

end triangle_problem_l1710_171068


namespace three_layer_rug_area_l1710_171080

/-- Given three rugs with a total area, floor area covered when overlapped, and area covered by two layers,
    calculate the area covered by three layers. -/
theorem three_layer_rug_area (total_area floor_area two_layer_area : ℝ) 
    (h1 : total_area = 204)
    (h2 : floor_area = 140)
    (h3 : two_layer_area = 24) :
  total_area - floor_area = 64 := by
  sorry

#check three_layer_rug_area

end three_layer_rug_area_l1710_171080


namespace complex_modulus_sqrt_5_l1710_171013

theorem complex_modulus_sqrt_5 (z : ℂ) (x y : ℝ) :
  z = x + y * I →
  x / (1 - I) = 1 + y * I →
  Complex.abs z = Real.sqrt 5 := by
sorry

end complex_modulus_sqrt_5_l1710_171013


namespace root_sum_reciprocal_l1710_171039

theorem root_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → x₂^2 - 3*x₂ - 1 = 0 → x₁ ≠ x₂ → 
  (1/x₁ + 1/x₂ : ℝ) = -3 := by
  sorry

end root_sum_reciprocal_l1710_171039


namespace final_price_is_66_percent_l1710_171070

/-- The percentage of the suggested retail price paid after discounts and tax -/
def final_price_percentage (initial_discount : ℝ) (clearance_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  (1 - initial_discount) * (1 - clearance_discount) * (1 + sales_tax)

/-- Theorem stating that the final price is 66% of the suggested retail price -/
theorem final_price_is_66_percent :
  final_price_percentage 0.2 0.25 0.1 = 0.66 := by
  sorry

#eval final_price_percentage 0.2 0.25 0.1

end final_price_is_66_percent_l1710_171070


namespace sqrt_meaningful_l1710_171064

theorem sqrt_meaningful (x : ℝ) : ∃ y : ℝ, y ^ 2 = x - 3 ↔ x ≥ 3 := by sorry

end sqrt_meaningful_l1710_171064


namespace ellipse_eccentricity_range_l1710_171036

theorem ellipse_eccentricity_range (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (x y : ℝ), 
    (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
    ((x + c)^2 + y^2) * ((x - c)^2 + y^2) = (2*c^2)^2 →
    (1/2 : ℝ) ≤ c/a ∧ c/a ≤ (Real.sqrt 3)/3 := by
  sorry

end ellipse_eccentricity_range_l1710_171036


namespace straight_line_angle_sum_l1710_171059

-- Define the theorem
theorem straight_line_angle_sum 
  (x y : ℝ) 
  (h1 : x + y = 76)  -- Given condition
  (h2 : 3 * x + 2 * y = 180)  -- Straight line segment condition
  : x = 28 := by
  sorry


end straight_line_angle_sum_l1710_171059


namespace monotone_increasing_interval_l1710_171024

/-- The function f(x) = ax / (x^2 + 1) is monotonically increasing on (-1, 1) for a > 0 -/
theorem monotone_increasing_interval (a : ℝ) (h : a > 0) :
  StrictMonoOn (fun x => a * x / (x^2 + 1)) (Set.Ioo (-1) 1) := by
  sorry

end monotone_increasing_interval_l1710_171024


namespace power_of_power_l1710_171081

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l1710_171081
