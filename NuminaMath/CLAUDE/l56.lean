import Mathlib

namespace distance_to_left_focus_l56_5644

/-- Given an ellipse and a hyperbola, prove that the distance from their intersection point
    in the first quadrant to the left focus of the ellipse is 4. -/
theorem distance_to_left_focus (x y : ℝ) : 
  x > 0 → y > 0 →  -- P is in the first quadrant
  x^2 / 9 + y^2 / 5 = 1 →  -- Ellipse equation
  x^2 - y^2 / 3 = 1 →  -- Hyperbola equation
  ∃ (f₁ : ℝ × ℝ), -- Left focus of the ellipse
    Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) = 4 := by
  sorry

end distance_to_left_focus_l56_5644


namespace quadratic_equal_roots_l56_5647

theorem quadratic_equal_roots : ∃ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 ∧
  ∀ y : ℝ, 4 * y^2 - 4 * y + 1 = 0 → y = x := by
  sorry

end quadratic_equal_roots_l56_5647


namespace sum_and_difference_problem_l56_5651

theorem sum_and_difference_problem (a b : ℤ) : 
  a + b = 56 → 
  a = b + 12 → 
  a = 22 → 
  b = 10 := by
sorry

end sum_and_difference_problem_l56_5651


namespace nursery_school_fraction_l56_5627

theorem nursery_school_fraction (total_students : ℕ) 
  (under_three : ℕ) (not_between_three_and_four : ℕ) :
  total_students = 50 →
  under_three = 20 →
  not_between_three_and_four = 25 →
  (total_students - not_between_three_and_four : ℚ) / total_students = 9 / 10 :=
by
  sorry

end nursery_school_fraction_l56_5627


namespace square_root_equation_implies_product_l56_5675

theorem square_root_equation_implies_product (x : ℝ) :
  Real.sqrt (8 + x) + Real.sqrt (25 - x^2) = 9 →
  (8 + x) * (25 - x^2) = 576 := by
sorry

end square_root_equation_implies_product_l56_5675


namespace class_test_theorem_l56_5645

/-- A theorem about a class test where some students didn't take the test -/
theorem class_test_theorem 
  (total_students : ℕ) 
  (answered_q2 : ℕ) 
  (did_not_take : ℕ) 
  (answered_both : ℕ) 
  (h1 : total_students = 30)
  (h2 : answered_q2 = 22)
  (h3 : did_not_take = 5)
  (h4 : answered_both = 22)
  (h5 : answered_both ≤ answered_q2)
  (h6 : did_not_take + answered_q2 ≤ total_students) :
  ∃ (answered_q1 : ℕ), answered_q1 = answered_both ∧ 
    answered_q1 + (answered_q2 - answered_both) + did_not_take ≤ total_students :=
by
  sorry

end class_test_theorem_l56_5645


namespace geometric_sequence_fourth_term_l56_5606

/-- Given a geometric sequence with first term 1000 and sixth term 125, 
    the fourth term is 125. -/
theorem geometric_sequence_fourth_term :
  ∀ (a : ℝ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * (a 1)⁻¹ * a 0) →  -- Geometric sequence definition
  a 0 = 1000 →                                 -- First term is 1000
  a 5 = 125 →                                  -- Sixth term is 125
  a 3 = 125 := by
sorry

end geometric_sequence_fourth_term_l56_5606


namespace union_of_sets_l56_5630

theorem union_of_sets : 
  let M : Set ℕ := {2, 3, 5}
  let N : Set ℕ := {3, 4, 5}
  M ∪ N = {2, 3, 4, 5} := by
  sorry

end union_of_sets_l56_5630


namespace set_equality_implies_m_equals_negative_one_l56_5641

theorem set_equality_implies_m_equals_negative_one (m : ℝ) :
  let A : Set ℝ := {m, 2}
  let B : Set ℝ := {m^2 - 2, 2}
  A = B → m = -1 :=
by
  sorry

end set_equality_implies_m_equals_negative_one_l56_5641


namespace committee_problem_l56_5601

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_problem :
  let total_students : ℕ := 10
  let committee_size : ℕ := 5
  let shared_members : ℕ := 3

  -- Number of different 5-student committees from 10 students
  (choose total_students committee_size = 252) ∧ 
  
  -- Number of ways to choose two 5-student committees with exactly 3 overlapping members
  ((choose total_students committee_size * 
    choose committee_size shared_members * 
    choose (total_students - committee_size) (committee_size - shared_members)) / 2 = 12600) :=
by sorry

end committee_problem_l56_5601


namespace skyscraper_arrangement_impossible_l56_5673

/-- The number of cyclic permutations of n elements -/
def cyclic_permutations (n : ℕ) : ℕ := (n - 1).factorial

/-- The maximum number of regions that n lines can divide a plane into -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of lines connecting n points -/
def connecting_lines (n : ℕ) : ℕ := n.choose 2

theorem skyscraper_arrangement_impossible :
  let n := 7
  let permutations := cyclic_permutations n
  let lines := connecting_lines n
  let regions := max_regions lines
  regions < permutations := by sorry

end skyscraper_arrangement_impossible_l56_5673


namespace min_sum_abs_values_l56_5656

theorem min_sum_abs_values (x : ℝ) :
  ∃ (m : ℝ), (∀ (y : ℝ), |y + 1| + |y + 2| + |y + 6| ≥ m) ∧
             (∃ (z : ℝ), |z + 1| + |z + 2| + |z + 6| = m) ∧
             (m = 5) :=
by sorry

end min_sum_abs_values_l56_5656


namespace xiaoming_class_ratio_l56_5678

theorem xiaoming_class_ratio (n : ℕ) (h1 : 30 < n) (h2 : n < 40) : ¬ ∃ k : ℕ, n = 10 * k := by
  sorry

end xiaoming_class_ratio_l56_5678


namespace arithmetic_sequence_property_l56_5616

/-- An arithmetic sequence with the given property -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) :
  2 * a 9 - a 10 = 24 := by
  sorry

end arithmetic_sequence_property_l56_5616


namespace complex_first_quadrant_a_range_l56_5604

theorem complex_first_quadrant_a_range (a : ℝ) :
  let z : ℂ := Complex.mk a (1 - a)
  (0 < z.re ∧ 0 < z.im) → (0 < a ∧ a < 1) := by
  sorry

end complex_first_quadrant_a_range_l56_5604


namespace shopping_cost_l56_5666

def toilet_paper_quantity : ℕ := 10
def paper_towel_quantity : ℕ := 7
def tissue_quantity : ℕ := 3

def toilet_paper_price : ℚ := 3/2
def paper_towel_price : ℚ := 2
def tissue_price : ℚ := 2

def total_cost : ℚ := 
  toilet_paper_quantity * toilet_paper_price + 
  paper_towel_quantity * paper_towel_price + 
  tissue_quantity * tissue_price

theorem shopping_cost : total_cost = 35 := by
  sorry

end shopping_cost_l56_5666


namespace zodiac_pigeonhole_l56_5663

/-- The number of Greek Zodiac signs -/
def greek_zodiac_count : ℕ := 12

/-- The number of Chinese Zodiac signs -/
def chinese_zodiac_count : ℕ := 12

/-- The minimum number of people required to ensure at least 3 people have the same Greek Zodiac sign -/
def min_people_same_greek_sign : ℕ := greek_zodiac_count * 2 + 1

/-- The minimum number of people required to ensure at least 2 people have the same combination of Greek and Chinese Zodiac signs -/
def min_people_same_combined_signs : ℕ := greek_zodiac_count * chinese_zodiac_count + 1

theorem zodiac_pigeonhole :
  (min_people_same_greek_sign = 25) ∧
  (min_people_same_combined_signs = 145) := by
  sorry

end zodiac_pigeonhole_l56_5663


namespace prime_fraction_characterization_l56_5602

theorem prime_fraction_characterization (k x y : ℕ+) :
  (∃ p : ℕ, Nat.Prime p ∧ (x : ℝ)^(k : ℕ) * y / ((x : ℝ)^2 + (y : ℝ)^2) = p) ↔ k = 2 ∨ k = 3 := by
  sorry

end prime_fraction_characterization_l56_5602


namespace no_solution_equation1_unique_solution_equation2_l56_5690

-- Define the first equation
def equation1 (x : ℝ) : Prop :=
  x ≠ 2 ∧ 3*x ≠ 6 ∧ (5*x - 4) / (x - 2) = (4*x + 10) / (3*x - 6) - 1

-- Define the second equation
def equation2 (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -2 ∧ 1 - (x - 2) / (2 + x) = 16 / (x^2 - 4)

-- Theorem for the first equation
theorem no_solution_equation1 : ¬∃ x, equation1 x :=
  sorry

-- Theorem for the second equation
theorem unique_solution_equation2 : ∃! x, equation2 x ∧ x = 6 :=
  sorry

end no_solution_equation1_unique_solution_equation2_l56_5690


namespace max_unused_cubes_l56_5623

/-- The side length of the original cube in small cube units -/
def original_side_length : ℕ := 10

/-- The total number of small cubes in the original cube -/
def total_cubes : ℕ := original_side_length ^ 3

/-- The function that calculates the number of small cubes used in a hollow cube of side length x -/
def cubes_used (x : ℕ) : ℕ := 6 * (x - 1) ^ 2 + 2

/-- The side length of the largest possible hollow cube -/
def largest_hollow_side : ℕ := 13

theorem max_unused_cubes :
  ∃ (unused : ℕ), unused = total_cubes - cubes_used largest_hollow_side ∧
  unused = 134 ∧
  ∀ (x : ℕ), x > largest_hollow_side → cubes_used x > total_cubes :=
sorry

end max_unused_cubes_l56_5623


namespace find_b_l56_5638

theorem find_b (p q : ℝ → ℝ) (b : ℝ) 
  (h1 : ∀ x, p x = 2 * x - 7)
  (h2 : ∀ x, q x = 3 * x - b)
  (h3 : p (q 4) = 7) : 
  b = 5 := by sorry

end find_b_l56_5638


namespace jane_started_with_87_crayons_l56_5655

/-- The number of crayons Jane started with -/
def initial_crayons : ℕ := sorry

/-- The number of crayons eaten by the hippopotamus -/
def eaten_crayons : ℕ := 7

/-- The number of crayons Jane ended up with -/
def remaining_crayons : ℕ := 80

/-- Theorem stating that Jane started with 87 crayons -/
theorem jane_started_with_87_crayons :
  initial_crayons = eaten_crayons + remaining_crayons :=
by sorry

end jane_started_with_87_crayons_l56_5655


namespace polynomial_factor_implies_b_value_l56_5680

theorem polynomial_factor_implies_b_value (a b : ℤ) :
  (∃ (c : ℤ), (X^2 - X - 1) * (a*X - c) = a*X^3 + b*X^2 - X + 1) →
  b = -1 :=
by sorry

end polynomial_factor_implies_b_value_l56_5680


namespace angle_of_inclination_range_l56_5607

theorem angle_of_inclination_range (θ : ℝ) (α : ℝ) :
  (∃ x y : ℝ, Real.sqrt 3 * x + y * Real.cos θ - 1 = 0) →
  (α = Real.arctan (Real.sqrt 3 / (-Real.cos θ))) →
  π / 3 ≤ α ∧ α ≤ 2 * π / 3 := by
  sorry

end angle_of_inclination_range_l56_5607


namespace share_multiple_is_four_l56_5653

/-- Represents the shares of three people in a division problem. -/
structure Shares where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Proves that the multiple of a's share is 4 under given conditions. -/
theorem share_multiple_is_four 
  (total : ℝ) 
  (shares : Shares) 
  (h_total : total = 880)
  (h_c_share : shares.c = 160)
  (h_sum : shares.a + shares.b + shares.c = total)
  (h_equal : ∃ x : ℝ, x * shares.a = 5 * shares.b ∧ x * shares.a = 10 * shares.c) :
  ∃ x : ℝ, x = 4 ∧ x * shares.a = 5 * shares.b ∧ x * shares.a = 10 * shares.c := by
  sorry

end share_multiple_is_four_l56_5653


namespace valid_cube_assignment_exists_l56_5611

/-- A cube is represented as a set of 8 vertices -/
def Cube := Fin 8

/-- An edge in the cube is a pair of vertices -/
def Edge := Prod Cube Cube

/-- The set of edges in a cube -/
def cubeEdges : Set Edge := sorry

/-- An assignment of natural numbers to the vertices of a cube -/
def Assignment := Cube → ℕ+

/-- Checks if one number divides another -/
def divides (a b : ℕ+) : Prop := ∃ k : ℕ+, b = a * k

/-- The main theorem stating the existence of a valid assignment -/
theorem valid_cube_assignment_exists : ∃ (f : Assignment),
  (∀ (e : Edge), e ∈ cubeEdges →
    (divides (f e.1) (f e.2) ∨ divides (f e.2) (f e.1))) ∧
  (∀ (v w : Cube), (Prod.mk v w) ∉ cubeEdges →
    ¬(divides (f v) (f w)) ∧ ¬(divides (f w) (f v))) := by
  sorry

end valid_cube_assignment_exists_l56_5611


namespace smallest_y_value_l56_5654

theorem smallest_y_value (y : ℝ) : 
  (3 * y^2 + 33 * y - 90 = y * (y + 18)) → y ≥ -9 :=
by sorry

end smallest_y_value_l56_5654


namespace dave_spent_22_tickets_l56_5677

def tickets_spent_on_beanie (initial_tickets : ℕ) (additional_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  initial_tickets + additional_tickets - remaining_tickets

theorem dave_spent_22_tickets : 
  tickets_spent_on_beanie 25 15 18 = 22 := by
  sorry

end dave_spent_22_tickets_l56_5677


namespace circle_arrangement_impossibility_l56_5614

theorem circle_arrangement_impossibility :
  ¬ ∃ (arrangement : Fin 2017 → ℕ),
    (∀ i, arrangement i ∈ Finset.range 2017 ∧ arrangement i ≠ 0) ∧
    (∀ i j, i ≠ j → arrangement i ≠ arrangement j) ∧
    (∀ i, Even ((arrangement i) + (arrangement ((i + 1) % 2017)) + (arrangement ((i + 2) % 2017)))) :=
by sorry

end circle_arrangement_impossibility_l56_5614


namespace square_difference_of_integers_l56_5600

theorem square_difference_of_integers (a b : ℕ) 
  (h1 : a + b = 60) 
  (h2 : a - b = 16) : 
  a^2 - b^2 = 960 := by
sorry

end square_difference_of_integers_l56_5600


namespace initial_owls_count_l56_5639

theorem initial_owls_count (initial_owls final_owls joined_owls : ℕ) : 
  initial_owls + joined_owls = final_owls →
  joined_owls = 2 →
  final_owls = 5 →
  initial_owls = 3 := by
  sorry

end initial_owls_count_l56_5639


namespace division_instead_of_multiplication_error_l56_5650

theorem division_instead_of_multiplication_error (y : ℝ) (h : y > 0) :
  (|8 * y - y / 8| / (8 * y)) * 100 = 98 := by
  sorry

end division_instead_of_multiplication_error_l56_5650


namespace absolute_value_inequality_l56_5628

theorem absolute_value_inequality (x : ℝ) :
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ ((1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5)) :=
by sorry

end absolute_value_inequality_l56_5628


namespace jump_rope_competition_theorem_l56_5669

/-- Represents a jump rope competition for a class of students. -/
structure JumpRopeCompetition where
  totalStudents : ℕ
  initialParticipants : ℕ
  initialAverage : ℕ
  lateStudentScores : List ℕ

/-- Calculates the new average score for the entire class after late students participate. -/
def newAverageScore (comp : JumpRopeCompetition) : ℚ :=
  let initialTotal := comp.initialParticipants * comp.initialAverage
  let lateTotal := comp.lateStudentScores.sum
  let totalJumps := initialTotal + lateTotal
  totalJumps / comp.totalStudents

/-- The main theorem stating that for the given competition parameters, 
    the new average score is 21. -/
theorem jump_rope_competition_theorem (comp : JumpRopeCompetition) 
  (h1 : comp.totalStudents = 30)
  (h2 : comp.initialParticipants = 26)
  (h3 : comp.initialAverage = 20)
  (h4 : comp.lateStudentScores = [26, 27, 28, 29]) :
  newAverageScore comp = 21 := by
  sorry

#eval newAverageScore {
  totalStudents := 30,
  initialParticipants := 26,
  initialAverage := 20,
  lateStudentScores := [26, 27, 28, 29]
}

end jump_rope_competition_theorem_l56_5669


namespace intersection_A_complement_B_l56_5615

def A : Set ℝ := {2, 3, 4, 5, 6}
def B : Set ℝ := {x : ℝ | x^2 - 8*x + 12 ≥ 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {3, 4, 5} := by
  sorry

end intersection_A_complement_B_l56_5615


namespace pythagorean_field_planting_l56_5699

theorem pythagorean_field_planting (a b : ℝ) (h1 : a = 5) (h2 : b = 12) : 
  let c := Real.sqrt (a^2 + b^2)
  let x := (a * b) / c
  let triangle_area := (a * b) / 2
  let square_area := x^2
  let planted_area := triangle_area - square_area
  let shortest_distance := (2 * square_area) / c
  shortest_distance = 3 → planted_area / triangle_area = 792 / 845 := by
sorry


end pythagorean_field_planting_l56_5699


namespace no_intersection_l56_5619

theorem no_intersection :
  ¬∃ (x y : ℝ), (y = |3*x + 4| ∧ y = -|2*x + 1|) := by
  sorry

end no_intersection_l56_5619


namespace line_AB_equation_l56_5637

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (3, 2)

-- Define that P is the midpoint of AB
def is_midpoint (A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem line_AB_equation (A B : ℝ × ℝ) :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ is_midpoint A B →
  ∀ x y : ℝ, (y = x - 1) ↔ (y - A.2 = A.2 - A.1 * (x - A.1)) :=
by sorry

end line_AB_equation_l56_5637


namespace total_golf_balls_l56_5671

/-- Represents the number of golf balls in one dozen -/
def dozen : ℕ := 12

/-- Represents the number of dozens Dan buys -/
def dan_dozens : ℕ := 5

/-- Represents the number of dozens Gus buys -/
def gus_dozens : ℕ := 3

/-- Represents the number of dozens Chris buys -/
def chris_dozens : ℕ := 4

/-- Represents the additional golf balls Chris buys -/
def chris_extra : ℕ := 6

/-- Represents the number of dozens Emily buys -/
def emily_dozens : ℕ := 2

/-- Represents the number of dozens Fred buys -/
def fred_dozens : ℕ := 1

/-- Theorem stating the total number of golf balls bought by the friends -/
theorem total_golf_balls :
  (dan_dozens + gus_dozens + chris_dozens + emily_dozens + fred_dozens) * dozen + chris_extra = 186 := by
  sorry

end total_golf_balls_l56_5671


namespace vector_operations_and_parallel_condition_l56_5665

def a : Fin 2 → ℝ := ![2, 0]
def b : Fin 2 → ℝ := ![1, 4]

theorem vector_operations_and_parallel_condition :
  (2 • a + 3 • b = ![7, 12]) ∧
  (a - 2 • b = ![0, -8]) ∧
  (∃ (k : ℝ), ∃ (t : ℝ), k • a + b = t • (a + 2 • b) → k = 1/2) := by sorry

end vector_operations_and_parallel_condition_l56_5665


namespace coefficient_equals_k_squared_minus_one_l56_5624

theorem coefficient_equals_k_squared_minus_one (k : ℝ) (h1 : k > 0) :
  (∃ b : ℝ, (k * b^2 - b)^2 = k^2 * b^4 - 2 * k * b^3 + k^2 * b^2 - b^2) →
  k = Real.sqrt 2 :=
sorry

end coefficient_equals_k_squared_minus_one_l56_5624


namespace measuring_rod_with_rope_l56_5696

theorem measuring_rod_with_rope (x y : ℝ) 
  (h1 : x - y = 5)
  (h2 : y - (1/2) * x = 5) : 
  x - y = 5 ∧ y - (1/2) * x = 5 := by
  sorry

end measuring_rod_with_rope_l56_5696


namespace triangle_circumcircle_radius_l56_5646

theorem triangle_circumcircle_radius 
  (a : ℝ) 
  (A : ℝ) 
  (h1 : a = 2) 
  (h2 : A = 2 * π / 3) : 
  ∃ R : ℝ, R = (2 * Real.sqrt 3) / 3 ∧ 
  R = a / (2 * Real.sin A) := by
  sorry

end triangle_circumcircle_radius_l56_5646


namespace bucket_sand_problem_l56_5689

theorem bucket_sand_problem (capacity_A : ℝ) (initial_sand_A : ℝ) :
  capacity_A > 0 →
  initial_sand_A ≥ 0 →
  initial_sand_A ≤ capacity_A →
  let capacity_B := capacity_A / 2
  let sand_B := 3 / 8 * capacity_B
  let total_sand := initial_sand_A + sand_B
  total_sand = 0.4375 * capacity_A →
  initial_sand_A = 1 / 4 * capacity_A :=
by sorry

end bucket_sand_problem_l56_5689


namespace integral_sqrt_x_2_minus_x_l56_5631

theorem integral_sqrt_x_2_minus_x (x : ℝ) : ∫ x in (0:ℝ)..1, Real.sqrt (x * (2 - x)) = π / 4 := by
  sorry

end integral_sqrt_x_2_minus_x_l56_5631


namespace distinct_collections_l56_5698

/-- Represents the letter counts in CALCULATOR --/
structure LetterCounts where
  a : Nat
  c : Nat
  l : Nat
  other_vowels : Nat
  other_consonants : Nat

/-- Represents a selection of letters --/
structure Selection where
  a : Nat
  c : Nat
  l : Nat
  other_vowels : Nat
  other_consonants : Nat

/-- Checks if a selection is valid --/
def is_valid_selection (s : Selection) : Prop :=
  s.a + s.other_vowels = 3 ∧ 
  s.c + s.l + s.other_consonants = 6

/-- Counts distinct vowel selections --/
def count_vowel_selections (total : LetterCounts) : Nat :=
  3 -- This is a simplification based on the problem's specifics

/-- Counts distinct consonant selections --/
noncomputable def count_consonant_selections (total : LetterCounts) : Nat :=
  sorry -- This would be calculated based on the combinations in the solution

/-- The main theorem --/
theorem distinct_collections (total : LetterCounts) 
  (h1 : total.a = 2)
  (h2 : total.c = 2)
  (h3 : total.l = 2)
  (h4 : total.other_vowels = 2)
  (h5 : total.other_consonants = 2) :
  (count_vowel_selections total) * (count_consonant_selections total) = 
  3 * (count_consonant_selections total) := by
  sorry

#check distinct_collections

end distinct_collections_l56_5698


namespace min_expense_is_2200_l56_5672

/-- Represents the types of trucks available --/
inductive TruckType
| A
| B

/-- Represents the characteristics of a truck type --/
structure TruckInfo where
  cost : ℕ
  capacity : ℕ

/-- The problem setup --/
def problem_setup : (TruckType → TruckInfo) × ℕ × ℕ × ℕ :=
  (λ t => match t with
    | TruckType.A => ⟨400, 20⟩
    | TruckType.B => ⟨300, 10⟩,
   4,  -- number of Type A trucks
   8,  -- number of Type B trucks
   100) -- total air conditioners to transport

/-- Calculate the minimum transportation expense --/
def min_transportation_expense (setup : (TruckType → TruckInfo) × ℕ × ℕ × ℕ) : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem min_expense_is_2200 :
  min_transportation_expense problem_setup = 2200 :=
sorry

end min_expense_is_2200_l56_5672


namespace tricycle_wheels_l56_5608

theorem tricycle_wheels (num_bicycles num_tricycles bicycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 24)
  (h2 : num_tricycles = 14)
  (h3 : bicycle_wheels = 2)
  (h4 : total_wheels = 90)
  : (total_wheels - num_bicycles * bicycle_wheels) / num_tricycles = 3 := by
  sorry

end tricycle_wheels_l56_5608


namespace intersection_sum_l56_5626

/-- Two lines intersect at a point if the point satisfies both line equations -/
def intersect_at (x y a b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 = (1/3) * p.2 + a ∧ p.2 = (1/3) * p.1 + b

/-- The problem statement -/
theorem intersection_sum (a b : ℝ) :
  intersect_at 3 1 a b (3, 1) → a + b = 8/3 := by
  sorry

end intersection_sum_l56_5626


namespace factorial_simplification_l56_5629

theorem factorial_simplification (N : ℕ) :
  (Nat.factorial (N + 2)) / (Nat.factorial N * (N + 3)) = ((N + 2) * (N + 1)) / (N + 3) := by
  sorry

end factorial_simplification_l56_5629


namespace odd_function_sum_l56_5687

/-- A function f is odd on an interval [a, b] -/
def IsOddOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Set.Icc a b, f (-x) = -f x) ∧ a + b = 0

/-- The main theorem -/
theorem odd_function_sum (a b c : ℝ) :
  IsOddOn (fun x ↦ a * x^3 + x + c) a b →
  a + b + c + 2 = 2 := by
  sorry


end odd_function_sum_l56_5687


namespace coin_toss_and_die_roll_probability_l56_5674

/-- The probability of getting exactly three heads and one tail when tossing four coins -/
def prob_three_heads_one_tail : ℚ := 1 / 4

/-- The probability of rolling a number greater than 4 on a six-sided die -/
def prob_die_greater_than_four : ℚ := 1 / 3

/-- The number of coins tossed -/
def num_coins : ℕ := 4

/-- The number of sides on the die -/
def num_die_sides : ℕ := 6

theorem coin_toss_and_die_roll_probability :
  prob_three_heads_one_tail * prob_die_greater_than_four = 1 / 12 :=
sorry

end coin_toss_and_die_roll_probability_l56_5674


namespace square_plus_double_equals_one_implies_double_square_plus_quadruple_plus_one_equals_three_l56_5697

theorem square_plus_double_equals_one_implies_double_square_plus_quadruple_plus_one_equals_three
  (a : ℝ) (h : a^2 + 2*a = 1) : 2*a^2 + 4*a + 1 = 3 := by
  sorry

end square_plus_double_equals_one_implies_double_square_plus_quadruple_plus_one_equals_three_l56_5697


namespace initial_shoe_collection_l56_5652

theorem initial_shoe_collection (initial_collection : ℕ) : 
  (initial_collection : ℝ) * 0.7 + 6 = 62 → initial_collection = 80 :=
by sorry

end initial_shoe_collection_l56_5652


namespace quadratic_completion_of_square_l56_5635

/-- Given a quadratic expression 3x^2 + 9x + 20, prove that when expressed in the form a(x - h)^2 + k, the value of h is -3/2. -/
theorem quadratic_completion_of_square (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
by sorry

end quadratic_completion_of_square_l56_5635


namespace pen_purchasing_plans_l56_5649

theorem pen_purchasing_plans :
  ∃! (solutions : List (ℕ × ℕ)), 
    solutions.length = 3 ∧
    (∀ (x y : ℕ), (x, y) ∈ solutions ↔ 
      x > 0 ∧ y > 0 ∧ 15 * x + 10 * y = 105) :=
by sorry

end pen_purchasing_plans_l56_5649


namespace shelly_thread_calculation_l56_5661

/-- The number of friends Shelly made in classes -/
def class_friends : ℕ := 10

/-- The number of friends Shelly made from after-school clubs -/
def club_friends : ℕ := 2 * class_friends

/-- The amount of thread needed for each keychain for class friends (in inches) -/
def class_thread_per_keychain : ℕ := 16

/-- The amount of thread needed for each keychain for after-school club friends (in inches) -/
def club_thread_per_keychain : ℕ := 20

/-- The total amount of thread Shelly needs (in inches) -/
def total_thread_needed : ℕ := class_friends * class_thread_per_keychain + club_friends * club_thread_per_keychain

theorem shelly_thread_calculation :
  total_thread_needed = 560 := by
  sorry

end shelly_thread_calculation_l56_5661


namespace power_of_two_equals_quadratic_plus_linear_plus_one_l56_5679

theorem power_of_two_equals_quadratic_plus_linear_plus_one
  (x y : ℕ) (h : 2^x = y^2 + y + 1) : x = 0 ∧ y = 0 := by
  sorry

end power_of_two_equals_quadratic_plus_linear_plus_one_l56_5679


namespace coeff_x_squared_is_thirteen_l56_5682

/-- The coefficient of x^2 in the expansion of (1-x)^3(2x^2+1)^5 -/
def coeff_x_squared : ℕ :=
  (Nat.choose 5 4) * 2 + 3 * (Nat.choose 5 5)

/-- Theorem stating that the coefficient of x^2 in the expansion of (1-x)^3(2x^2+1)^5 is 13 -/
theorem coeff_x_squared_is_thirteen : coeff_x_squared = 13 := by
  sorry

end coeff_x_squared_is_thirteen_l56_5682


namespace bicycle_costs_l56_5695

theorem bicycle_costs (B H L : ℝ) 
  (total_cost : B + H + L = 480)
  (bicycle_helmet_ratio : B = 5 * H)
  (lock_helmet_ratio : L = 0.5 * H)
  (lock_total_ratio : L = 0.1 * 480) : 
  B = 360 ∧ H = 72 ∧ L = 48 := by
  sorry

end bicycle_costs_l56_5695


namespace doughnuts_distribution_l56_5609

theorem doughnuts_distribution (total_doughnuts : ℕ) (total_boxes : ℕ) (first_two_boxes : ℕ) (doughnuts_per_first_two : ℕ) :
  total_doughnuts = 72 →
  total_boxes = 6 →
  first_two_boxes = 2 →
  doughnuts_per_first_two = 12 →
  (total_doughnuts - first_two_boxes * doughnuts_per_first_two) % (total_boxes - first_two_boxes) = 0 →
  (total_doughnuts - first_two_boxes * doughnuts_per_first_two) / (total_boxes - first_two_boxes) = 12 :=
by sorry

end doughnuts_distribution_l56_5609


namespace xiaos_speed_correct_l56_5620

/-- Xiao Hu Ma's speed in meters per minute -/
def xiaos_speed : ℝ := 80

/-- Distance between Xiao Hu Ma's house and school in meters -/
def total_distance : ℝ := 1800

/-- Distance from the meeting point to school in meters -/
def remaining_distance : ℝ := 200

/-- Time difference between Xiao Hu Ma and his father starting in minutes -/
def time_difference : ℝ := 10

theorem xiaos_speed_correct :
  xiaos_speed * (total_distance - remaining_distance) / xiaos_speed -
  (total_distance - remaining_distance) / (2 * xiaos_speed) = time_difference := by
  sorry

end xiaos_speed_correct_l56_5620


namespace alternating_coloring_uniform_rows_l56_5633

/-- Represents a color in the pattern -/
inductive Color
| A
| B

/-- Represents the grid of the bracelet -/
def BraceletGrid := Fin 10 → Fin 2 → Color

/-- A coloring function that alternates colors in each column -/
def alternatingColoring : BraceletGrid :=
  fun i j => if j = 0 then Color.A else Color.B

/-- Theorem stating that the alternating coloring results in uniform rows -/
theorem alternating_coloring_uniform_rows :
  (∀ i : Fin 10, alternatingColoring i 0 = Color.A) ∧
  (∀ i : Fin 10, alternatingColoring i 1 = Color.B) := by
  sorry


end alternating_coloring_uniform_rows_l56_5633


namespace quadratic_real_roots_l56_5632

/-- The quadratic equation (m-3)x^2 - 2x + 1 = 0 has real roots if and only if m ≤ 4 and m ≠ 3. -/
theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by sorry

end quadratic_real_roots_l56_5632


namespace ab_product_l56_5658

theorem ab_product (a b : ℚ) (h : 6 * a = 20 ∧ 7 * b = 20) : 84 * a * b = 800 := by
  sorry

end ab_product_l56_5658


namespace parabola_transformation_l56_5684

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = -2x^2 + 1 -/
def original_parabola : Parabola := ⟨-2, 0, 1⟩

/-- Moves a parabola horizontally by h units -/
def move_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  ⟨p.a, -2 * p.a * h + p.b, p.a * h^2 - p.b * h + p.c⟩

/-- Moves a parabola vertically by k units -/
def move_vertical (p : Parabola) (k : ℝ) : Parabola :=
  ⟨p.a, p.b, p.c + k⟩

/-- The final parabola after moving right by 1 and up by 1 -/
def final_parabola : Parabola :=
  move_vertical (move_horizontal original_parabola 1) 1

theorem parabola_transformation :
  final_parabola = ⟨-2, 4, 2⟩ := by sorry

end parabola_transformation_l56_5684


namespace curve_expression_bound_l56_5670

theorem curve_expression_bound :
  ∀ x y : ℝ, x^2 + (y^2)/4 = 4 → 
  ∃ t : ℝ, x = 2*Real.cos t ∧ y = 4*Real.sin t ∧ 
  -4 ≤ Real.sqrt 3 * x + (1/2) * y ∧ Real.sqrt 3 * x + (1/2) * y ≤ 4 :=
by sorry

end curve_expression_bound_l56_5670


namespace four_machines_copies_l56_5621

/-- Represents a copying machine with a specific rate --/
structure Machine where
  copies : ℕ
  minutes : ℕ

/-- Calculates the total number of copies produced by multiple machines in a given time --/
def totalCopies (machines : List Machine) (workTime : ℕ) : ℕ :=
  machines.foldl (fun acc m => acc + workTime * m.copies / m.minutes) 0

/-- Theorem stating the total number of copies produced by four specific machines in 40 minutes --/
theorem four_machines_copies : 
  let machineA : Machine := ⟨100, 8⟩
  let machineB : Machine := ⟨150, 10⟩
  let machineC : Machine := ⟨200, 12⟩
  let machineD : Machine := ⟨250, 15⟩
  let machines : List Machine := [machineA, machineB, machineC, machineD]
  totalCopies machines 40 = 2434 := by
  sorry

end four_machines_copies_l56_5621


namespace min_value_inequality_l56_5636

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  x / y + 1 / x ≥ 3 := by
  sorry

end min_value_inequality_l56_5636


namespace lcm_gcd_problem_l56_5659

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 55 → 
  a = 210 → 
  b = 605 := by
sorry

end lcm_gcd_problem_l56_5659


namespace regular_tetrahedron_properties_regular_tetrahedron_all_properties_l56_5618

/-- Definition of a regular tetrahedron -/
structure RegularTetrahedron where
  /-- All edges of the tetrahedron are equal -/
  edges_equal : Bool
  /-- All faces of the tetrahedron are congruent equilateral triangles -/
  faces_congruent : Bool
  /-- The angle between any two edges at the same vertex is equal -/
  vertex_angles_equal : Bool
  /-- The dihedral angle between any two adjacent faces is equal -/
  dihedral_angles_equal : Bool

/-- Theorem: Properties of a regular tetrahedron -/
theorem regular_tetrahedron_properties (t : RegularTetrahedron) : 
  t.edges_equal ∧ 
  t.faces_congruent ∧ 
  t.vertex_angles_equal ∧ 
  t.dihedral_angles_equal := by
  sorry

/-- Corollary: All three properties mentioned in the problem are true for a regular tetrahedron -/
theorem regular_tetrahedron_all_properties (t : RegularTetrahedron) :
  (t.edges_equal ∧ t.vertex_angles_equal) ∧
  (t.faces_congruent ∧ t.dihedral_angles_equal) ∧
  (t.faces_congruent ∧ t.vertex_angles_equal) := by
  sorry

end regular_tetrahedron_properties_regular_tetrahedron_all_properties_l56_5618


namespace m_range_l56_5693

/-- The range of m given the specified conditions -/
theorem m_range (h1 : ∀ x : ℝ, 2 * x > m * (x^2 + 1)) 
                (h2 : ∃ x₀ : ℝ, x₀^2 + 2*x₀ - m - 1 = 0) : 
  -2 ≤ m ∧ m < -1 := by
  sorry

end m_range_l56_5693


namespace range_of_m_l56_5625

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 2 / x + 1 / y = 1) (h2 : ∀ x y, x > 0 → y > 0 → 2 / x + 1 / y = 1 → x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
sorry

end range_of_m_l56_5625


namespace root_equation_value_l56_5688

theorem root_equation_value (m : ℝ) : 
  m^2 - m - 2 = 0 → m^2 - m + 2023 = 2025 := by
  sorry

end root_equation_value_l56_5688


namespace x_minus_y_equals_fourteen_l56_5676

theorem x_minus_y_equals_fourteen (x y : ℝ) (h : x^2 + y^2 = 16*x - 12*y + 100) : x - y = 14 := by
  sorry

end x_minus_y_equals_fourteen_l56_5676


namespace solution_implies_range_l56_5610

/-- The function f(x) = x^2 - 4x - 2 -/
def f (x : ℝ) := x^2 - 4*x - 2

/-- The theorem stating that if x^2 - 4x - 2 - a > 0 has solutions in (1,4), then a < -2 -/
theorem solution_implies_range (a : ℝ) : 
  (∃ x ∈ Set.Ioo 1 4, f x > a) → a < -2 := by
  sorry

end solution_implies_range_l56_5610


namespace arithmetic_sequence_problem_l56_5617

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that in an arithmetic sequence {aₙ} where
    a₅ + a₆ = 16 and a₈ = 12, the third term a₃ equals 4. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 5 + a 6 = 16)
    (h_eighth : a 8 = 12) : 
  a 3 = 4 := by
  sorry

end arithmetic_sequence_problem_l56_5617


namespace wood_square_weight_relation_second_wood_square_weight_l56_5642

/-- Represents the properties of a square piece of wood -/
structure WoodSquare where
  side_length : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two square pieces of wood with uniform density and thickness -/
theorem wood_square_weight_relation (w1 w2 : WoodSquare) 
  (h1 : w1.side_length = 3)
  (h2 : w1.weight = 12)
  (h3 : w2.side_length = 6) :
  w2.weight = 48 := by
  sorry

/-- Main theorem proving the weight of the second piece of wood -/
theorem second_wood_square_weight :
  ∃ (w1 w2 : WoodSquare), 
    w1.side_length = 3 ∧ 
    w1.weight = 12 ∧ 
    w2.side_length = 6 ∧ 
    w2.weight = 48 := by
  sorry

end wood_square_weight_relation_second_wood_square_weight_l56_5642


namespace polynomial_factorization_l56_5685

theorem polynomial_factorization (x y z : ℝ) :
  x * (y - z)^3 + y * (z - x)^3 + z * (x - y)^3 = (x - y) * (y - z) * (z - x) * (x + y + z) := by
  sorry

end polynomial_factorization_l56_5685


namespace shaded_area_is_32_5_l56_5643

/-- Represents a rectangular grid -/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Represents a right-angled triangle -/
structure RightTriangle where
  base : ℕ
  height : ℕ

/-- Calculates the area of a shaded region in a grid, excluding a right-angled triangle -/
def shadedArea (g : Grid) (t : RightTriangle) : ℚ :=
  (g.rows * g.cols : ℚ) - (t.base * t.height : ℚ) / 2

/-- Theorem stating that the shaded area in the given problem is 32.5 square units -/
theorem shaded_area_is_32_5 :
  let g : Grid := ⟨4, 13⟩
  let t : RightTriangle := ⟨13, 3⟩
  shadedArea g t = 32.5 := by
  sorry

end shaded_area_is_32_5_l56_5643


namespace boy_and_bus_speeds_l56_5683

/-- Represents the problem of finding the speeds of a boy and a bus given certain conditions. -/
theorem boy_and_bus_speeds
  (total_distance : ℝ)
  (first_meeting_time : ℝ)
  (boy_additional_distance : ℝ)
  (stop_time : ℝ) :
  total_distance = 4.5 ∧
  first_meeting_time = 0.25 ∧
  boy_additional_distance = 9 / 28 ∧
  stop_time = 4 / 60 →
  ∃ (boy_speed bus_speed : ℝ),
    boy_speed = 3 ∧
    bus_speed = 45 ∧
    boy_speed > 0 ∧
    bus_speed > 0 ∧
    boy_speed * first_meeting_time + boy_additional_distance =
      bus_speed * first_meeting_time - total_distance ∧
    bus_speed * (first_meeting_time + 2 * stop_time) = 2 * total_distance :=
by sorry

end boy_and_bus_speeds_l56_5683


namespace quadratic_function_property_l56_5603

/-- Given a quadratic function f(x) = x^2 + ax + b where a and b are distinct real numbers,
    if f(a) = f(b), then f(2) = 4 -/
theorem quadratic_function_property (a b : ℝ) (h_distinct : a ≠ b) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (f a = f b) → f 2 = 4 := by
  sorry

end quadratic_function_property_l56_5603


namespace rahim_average_book_price_l56_5657

/-- The average price of books bought by Rahim -/
def average_price (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating the average price of books bought by Rahim -/
theorem rahim_average_book_price :
  let books1 := 65
  let books2 := 50
  let price1 := 1160
  let price2 := 920
  abs (average_price books1 books2 price1 price2 - 18.09) < 0.01 := by
  sorry

end rahim_average_book_price_l56_5657


namespace playground_children_count_l56_5622

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 40) 
  (h2 : girls = 77) : 
  boys + girls = 117 := by
sorry

end playground_children_count_l56_5622


namespace equation_solutions_l56_5692

theorem equation_solutions :
  (∃ x₁ x₂, (3 * x₁ + 2)^2 = 16 ∧ (3 * x₂ + 2)^2 = 16 ∧ x₁ = 2/3 ∧ x₂ = -2) ∧
  (∃ x, (1/2) * (2 * x - 1)^3 = -4 ∧ x = -1/2) := by
  sorry

end equation_solutions_l56_5692


namespace not_divides_two_pow_minus_one_l56_5640

theorem not_divides_two_pow_minus_one (n : ℕ) (hn : n > 1) : ¬(n ∣ 2^n - 1) := by
  sorry

end not_divides_two_pow_minus_one_l56_5640


namespace intersection_of_M_and_N_l56_5686

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {y | ∃ x, y = -x^2 + 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end intersection_of_M_and_N_l56_5686


namespace line_contains_point_l56_5660

theorem line_contains_point (k : ℝ) : 
  (2 + 3 * k * (-1/3) = -4 * 1) → k = 6 := by
  sorry

end line_contains_point_l56_5660


namespace quadratic_equation_roots_range_l56_5694

/-- The range of m for which the quadratic equation (m-1)x^2 + 2x + 1 = 0 has two real roots -/
theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (m - 1) * x₁^2 + 2 * x₁ + 1 = 0 ∧ 
    (m - 1) * x₂^2 + 2 * x₂ + 1 = 0) ↔ 
  (m ≤ 2 ∧ m ≠ 1) :=
by sorry

end quadratic_equation_roots_range_l56_5694


namespace sufficient_but_not_necessary_l56_5691

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- State the theorem
theorem sufficient_but_not_necessary : 
  (p 2) ∧ (∃ a : ℝ, a ≠ 2 ∧ p a) :=
sorry

end sufficient_but_not_necessary_l56_5691


namespace problem_solution_l56_5664

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t, Real.sqrt 2 / 2 + Real.sqrt 3 * t)

-- Define the curve C in polar form
def curve_C (θ : ℝ) : ℝ := 2 * Real.cos (θ - Real.pi / 4)

-- Define point P
def point_P : ℝ × ℝ := (0, Real.sqrt 2 / 2)

-- Theorem statement
theorem problem_solution :
  -- 1. The slope angle of line l is π/3
  (let slope := (Real.sqrt 3);
   Real.arctan slope = Real.pi / 3) ∧
  -- 2. The rectangular equation of curve C
  (∀ x y : ℝ, (x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 1 ↔
    ∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ∧
  -- 3. If line l intersects curve C at points A and B, then |PA| + |PB| = √10/2
  (∃ A B : ℝ × ℝ,
    (∃ t : ℝ, line_l t = A) ∧
    (∃ t : ℝ, line_l t = B) ∧
    (∃ θ : ℝ, A.1 = curve_C θ * Real.cos θ ∧ A.2 = curve_C θ * Real.sin θ) ∧
    (∃ θ : ℝ, B.1 = curve_C θ * Real.cos θ ∧ B.2 = curve_C θ * Real.sin θ) ∧
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
    Real.sqrt 10 / 2) :=
sorry

end problem_solution_l56_5664


namespace rectangle_length_l56_5662

/-- Given a rectangle with area 28 square centimeters and width 4 centimeters, its length is 7 centimeters. -/
theorem rectangle_length (area width : ℝ) (h_area : area = 28) (h_width : width = 4) :
  area / width = 7 := by
  sorry

end rectangle_length_l56_5662


namespace waitress_average_orders_per_hour_l56_5634

theorem waitress_average_orders_per_hour
  (hourly_wage : ℝ)
  (tip_rate : ℝ)
  (num_shifts : ℕ)
  (hours_per_shift : ℕ)
  (total_earnings : ℝ)
  (h1 : hourly_wage = 4)
  (h2 : tip_rate = 0.15)
  (h3 : num_shifts = 3)
  (h4 : hours_per_shift = 8)
  (h5 : total_earnings = 240) :
  let total_hours : ℕ := num_shifts * hours_per_shift
  let wage_earnings : ℝ := hourly_wage * total_hours
  let tip_earnings : ℝ := total_earnings - wage_earnings
  let total_orders : ℝ := tip_earnings / tip_rate
  let avg_orders_per_hour : ℝ := total_orders / total_hours
  avg_orders_per_hour = 40 := by
sorry

end waitress_average_orders_per_hour_l56_5634


namespace cryptarithm_solutions_l56_5667

def is_valid_solution (tuk : ℕ) (ctuk : ℕ) : Prop :=
  tuk ≥ 100 ∧ tuk < 1000 ∧ ctuk ≥ 1000 ∧ ctuk < 10000 ∧
  5 * tuk = ctuk ∧
  (tuk.digits 10).card = 3 ∧ (ctuk.digits 10).card = 4

theorem cryptarithm_solutions :
  (∀ tuk ctuk : ℕ, is_valid_solution tuk ctuk → (tuk = 250 ∧ ctuk = 1250) ∨ (tuk = 750 ∧ ctuk = 3750)) ∧
  is_valid_solution 250 1250 ∧
  is_valid_solution 750 3750 := by
  sorry

end cryptarithm_solutions_l56_5667


namespace qrs_company_profit_change_l56_5681

theorem qrs_company_profit_change (march_profit : ℝ) : 
  let april_profit := 1.10 * march_profit
  let may_profit := april_profit * (1 - x / 100)
  let june_profit := may_profit * 1.50
  june_profit = 1.3200000000000003 * march_profit →
  x = 20 :=
by
  sorry

end qrs_company_profit_change_l56_5681


namespace ice_cream_cost_l56_5605

theorem ice_cream_cost (ice_cream_cartons yoghurt_cartons : ℕ) 
  (yoghurt_cost : ℚ) (cost_difference : ℚ) :
  ice_cream_cartons = 19 →
  yoghurt_cartons = 4 →
  yoghurt_cost = 1 →
  cost_difference = 129 →
  ∃ (ice_cream_cost : ℚ), 
    ice_cream_cost * ice_cream_cartons = yoghurt_cost * yoghurt_cartons + cost_difference ∧
    ice_cream_cost = 7 := by
  sorry

end ice_cream_cost_l56_5605


namespace inheritance_division_l56_5613

theorem inheritance_division (A B : ℝ) : 
  A + B = 100 ∧ 
  (1/4 : ℝ) * B - (1/3 : ℝ) * A = 11 →
  A = 24 ∧ B = 76 := by
sorry

end inheritance_division_l56_5613


namespace unique_integer_solution_l56_5612

theorem unique_integer_solution :
  ∃! (a b c : ℤ), a^2 + b^2 + c^2 + 3 < a*b + 3*b + 2*c ∧ a = 1 ∧ b = 2 ∧ c = 1 :=
by sorry

end unique_integer_solution_l56_5612


namespace equation_solution_exists_l56_5648

def f (x : ℝ) := x^3 - x - 1

theorem equation_solution_exists :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 1.5 ∧ |f x| ≤ 0.001 :=
by
  sorry

end equation_solution_exists_l56_5648


namespace shop_owner_profit_l56_5668

/-- Calculates the percentage profit of a shop owner who cheats with weights -/
theorem shop_owner_profit (buying_cheat : ℝ) (selling_cheat : ℝ) : 
  buying_cheat = 0.14 →
  selling_cheat = 0.20 →
  (((1 + buying_cheat) / (1 - selling_cheat)) - 1) * 100 = 42.5 := by
  sorry

end shop_owner_profit_l56_5668
