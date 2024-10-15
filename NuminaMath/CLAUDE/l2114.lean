import Mathlib

namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_not_p_sufficient_not_necessary_l2114_211425

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1: Range of x when a = 1 and p ∧ q is true
theorem range_of_x_when_a_is_one (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

-- Part 2: Range of a for which ¬p is sufficient but not necessary for ¬q
theorem range_of_a_for_not_p_sufficient_not_necessary (a : ℝ) :
  (∀ x, ¬(p x a) → ¬(q x)) ∧ (∃ x, ¬(q x) ∧ p x a) ↔ 1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_not_p_sufficient_not_necessary_l2114_211425


namespace NUMINAMATH_CALUDE_min_boxes_equal_candies_l2114_211448

/-- The number of candies in a box of "Sweet Mathematics" -/
def SM_box_size : ℕ := 12

/-- The number of candies in a box of "Geometry with Nuts" -/
def GN_box_size : ℕ := 15

/-- The minimum number of boxes of "Sweet Mathematics" needed -/
def min_SM_boxes : ℕ := 5

/-- The minimum number of boxes of "Geometry with Nuts" needed -/
def min_GN_boxes : ℕ := 4

theorem min_boxes_equal_candies :
  min_SM_boxes * SM_box_size = min_GN_boxes * GN_box_size ∧
  ∀ (sm gn : ℕ), sm * SM_box_size = gn * GN_box_size →
    sm ≥ min_SM_boxes ∧ gn ≥ min_GN_boxes :=
by sorry

end NUMINAMATH_CALUDE_min_boxes_equal_candies_l2114_211448


namespace NUMINAMATH_CALUDE_lucy_cookie_sales_l2114_211422

theorem lucy_cookie_sales : ∀ (first_round second_round total : ℕ),
  first_round = 34 →
  second_round = 27 →
  total = first_round + second_round →
  total = 61 := by
  sorry

end NUMINAMATH_CALUDE_lucy_cookie_sales_l2114_211422


namespace NUMINAMATH_CALUDE_similar_squares_side_length_l2114_211405

theorem similar_squares_side_length (small_side : ℝ) (area_ratio : ℝ) :
  small_side = 4 →
  area_ratio = 9 →
  ∃ large_side : ℝ,
    large_side = small_side * Real.sqrt area_ratio ∧
    large_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_similar_squares_side_length_l2114_211405


namespace NUMINAMATH_CALUDE_angle_325_same_terminal_side_as_neg_35_l2114_211436

/-- 
Given an angle θ in degrees, this function returns true if θ has the same terminal side as -35°.
-/
def hasSameTerminalSideAs (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = k * 360 + (-35)

/-- 
This theorem states that 325° has the same terminal side as -35° and is between 0° and 360°.
-/
theorem angle_325_same_terminal_side_as_neg_35 :
  hasSameTerminalSideAs 325 ∧ 0 ≤ 325 ∧ 325 < 360 := by
  sorry

end NUMINAMATH_CALUDE_angle_325_same_terminal_side_as_neg_35_l2114_211436


namespace NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l2114_211437

/-- Given a parabola y = ax^2 + bx + c passing through points (-2, 9), (4, 9), and (5, 13),
    the x-coordinate of its vertex is 1. -/
theorem parabola_vertex_x_coordinate
  (a b c : ℝ)
  (h1 : a * (-2)^2 + b * (-2) + c = 9)
  (h2 : a * 4^2 + b * 4 + c = 9)
  (h3 : a * 5^2 + b * 5 + c = 13) :
  (∃ y : ℝ, a * 1^2 + b * 1 + c = y ∧
    ∀ x : ℝ, a * x^2 + b * x + c ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l2114_211437


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l2114_211440

-- Define the circles M₁ and M₂
def M₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def M₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the trajectory of the center of the moving circle M
def trajectory (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1 ∧ x ≠ -2

-- State the theorem
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
    (∃ (R : ℝ), 
      (∀ (x₁ y₁ : ℝ), M₁ x₁ y₁ → (x - x₁)^2 + (y - y₁)^2 = (1 + R)^2) ∧
      (∀ (x₂ y₂ : ℝ), M₂ x₂ y₂ → (x - x₂)^2 + (y - y₂)^2 = (3 - R)^2)) →
    trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l2114_211440


namespace NUMINAMATH_CALUDE_lucas_100_mod_5_l2114_211415

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | (n + 2) => lucas (n + 1) + lucas n

theorem lucas_100_mod_5 : lucas 99 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lucas_100_mod_5_l2114_211415


namespace NUMINAMATH_CALUDE_round_trip_average_speed_river_boat_average_speed_l2114_211447

/-- The average speed for a round trip given upstream and downstream speeds -/
theorem round_trip_average_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed > 0)
  (h2 : downstream_speed > 0)
  (h3 : upstream_speed ≠ downstream_speed) :
  (2 * upstream_speed * downstream_speed) / (upstream_speed + downstream_speed) = 
    (2 * 3 * 7) / (3 + 7) := by
  sorry

/-- The specific case for the river boat problem -/
theorem river_boat_average_speed :
  (2 * 3 * 7) / (3 + 7) = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_river_boat_average_speed_l2114_211447


namespace NUMINAMATH_CALUDE_gmat_test_problem_l2114_211401

theorem gmat_test_problem (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ)
  (h1 : p_first = 0.85)
  (h2 : p_second = 0.65)
  (h3 : p_neither = 0.05) :
  p_first + p_second - (1 - p_neither) = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_gmat_test_problem_l2114_211401


namespace NUMINAMATH_CALUDE_star_op_two_neg_four_l2114_211432

-- Define the * operation for rational numbers
def star_op (x y : ℚ) : ℚ := (x * y) / (x + y)

-- Theorem statement
theorem star_op_two_neg_four : star_op 2 (-4) = 4 := by sorry

end NUMINAMATH_CALUDE_star_op_two_neg_four_l2114_211432


namespace NUMINAMATH_CALUDE_election_vote_ratio_l2114_211463

theorem election_vote_ratio :
  let joey_votes : ℕ := 8
  let barry_votes : ℕ := 2 * (joey_votes + 3)
  let marcy_votes : ℕ := 66
  (marcy_votes : ℚ) / barry_votes = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_election_vote_ratio_l2114_211463


namespace NUMINAMATH_CALUDE_point_one_and_ten_are_reciprocals_l2114_211458

/-- Two numbers are reciprocals if their product is 1 -/
def are_reciprocals (a b : ℝ) : Prop := a * b = 1

/-- 0.1 and 10 are reciprocals of each other -/
theorem point_one_and_ten_are_reciprocals : are_reciprocals 0.1 10 := by
  sorry

end NUMINAMATH_CALUDE_point_one_and_ten_are_reciprocals_l2114_211458


namespace NUMINAMATH_CALUDE_geometric_sequence_theorem_l2114_211412

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1

/-- Given conditions for the geometric sequence -/
def satisfies_conditions (seq : GeometricSequence) : Prop :=
  seq.a 3 + seq.a 6 = 36 ∧ seq.a 4 + seq.a 7 = 18

theorem geometric_sequence_theorem (seq : GeometricSequence) 
  (h : satisfies_conditions seq) : 
  ∃ n : ℕ, seq.a n = 1/2 ∧ n = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_theorem_l2114_211412


namespace NUMINAMATH_CALUDE_meaningful_square_root_range_l2114_211408

theorem meaningful_square_root_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 1 / (x - 1)) → x > 1 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_square_root_range_l2114_211408


namespace NUMINAMATH_CALUDE_li_zhi_assignment_l2114_211456

-- Define the universities
inductive University
| Tongji
| ShanghaiJiaoTong
| ShanghaiNormal

-- Define the volunteer roles
inductive VolunteerRole
| Translator
| City
| Social

-- Define the students
inductive Student
| LiZhi
| WenWen
| LiuBing

-- Define the assignment function
def assignment (s : Student) : University × VolunteerRole :=
  sorry

-- State the theorem
theorem li_zhi_assignment :
  (∀ s, s = Student.LiZhi → (assignment s).1 ≠ University.Tongji) →
  (∀ s, s = Student.WenWen → (assignment s).1 ≠ University.ShanghaiJiaoTong) →
  (∀ s, (assignment s).1 = University.Tongji → (assignment s).2 ≠ VolunteerRole.Translator) →
  (∀ s, (assignment s).1 = University.ShanghaiJiaoTong → (assignment s).2 = VolunteerRole.City) →
  (∀ s, s = Student.WenWen → (assignment s).2 ≠ VolunteerRole.Social) →
  (assignment Student.LiZhi).1 = University.ShanghaiJiaoTong ∧ 
  (assignment Student.LiZhi).2 = VolunteerRole.City :=
by
  sorry

end NUMINAMATH_CALUDE_li_zhi_assignment_l2114_211456


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2114_211400

theorem complex_expression_simplification (i : ℂ) (h : i^2 = -1) :
  i * (1 - i) - 1 = i := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2114_211400


namespace NUMINAMATH_CALUDE_rachel_tips_l2114_211499

theorem rachel_tips (hourly_wage : ℚ) (people_served : ℕ) (total_made : ℚ) 
  (hw : hourly_wage = 12)
  (ps : people_served = 20)
  (tm : total_made = 37) :
  (total_made - hourly_wage) / people_served = 25 / 20 := by
  sorry

#eval (37 : ℚ) - 12
#eval (25 : ℚ) / 20

end NUMINAMATH_CALUDE_rachel_tips_l2114_211499


namespace NUMINAMATH_CALUDE_four_students_three_teams_l2114_211411

/-- The number of ways students can sign up for sports teams -/
def signup_ways (num_students : ℕ) (num_teams : ℕ) : ℕ :=
  num_teams ^ num_students

/-- Theorem: 4 students signing up for 3 teams results in 3^4 ways -/
theorem four_students_three_teams :
  signup_ways 4 3 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_four_students_three_teams_l2114_211411


namespace NUMINAMATH_CALUDE_sequence_properties_l2114_211481

def a (n : ℕ) : ℚ := (2 * n) / (3 * n + 2)

theorem sequence_properties : 
  (a 3 = 6 / 11) ∧ 
  (∀ n : ℕ, a (n - 1) = (2 * n - 2) / (3 * n - 1)) ∧ 
  (a 8 = 8 / 13) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2114_211481


namespace NUMINAMATH_CALUDE_infinitely_many_silesian_infinitely_many_non_silesian_l2114_211472

/-- An integer n is Silesian if there exist positive integers a, b, c such that
    n = (a² + b² + c²) / (ab + bc + ca) -/
def is_silesian (n : ℤ) : Prop :=
  ∃ (a b c : ℕ+), n = (a.val^2 + b.val^2 + c.val^2) / (a.val * b.val + b.val * c.val + c.val * a.val)

/-- There are infinitely many Silesian integers -/
theorem infinitely_many_silesian : ∀ N : ℕ, ∃ n : ℤ, n > N ∧ is_silesian n :=
sorry

/-- There are infinitely many positive integers that are not Silesian -/
theorem infinitely_many_non_silesian : ∀ N : ℕ, ∃ k : ℕ, k > N ∧ ¬is_silesian (3 * k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_silesian_infinitely_many_non_silesian_l2114_211472


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2114_211410

/-- The line equation kx - y - 2k + 3 = 0 is tangent to the circle x^2 + (y + 1)^2 = 4 if and only if k = 3/4 -/
theorem line_tangent_to_circle (k : ℝ) : 
  (∀ x y : ℝ, k * x - y - 2 * k + 3 = 0 → x^2 + (y + 1)^2 = 4) ↔ k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2114_211410


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l2114_211483

-- Define a random variable following normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def probability (ξ : normal_distribution 1 σ) (a b : ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_symmetry 
  (σ : ℝ) 
  (ξ : normal_distribution 1 σ) 
  (h : probability ξ 0 1 = 0.4) : 
  probability ξ 0 2 = 0.8 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l2114_211483


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2114_211404

theorem matrix_equation_solution :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  M^3 - 3 • M^2 + 2 • M = !![8, 16; 4, 8] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2114_211404


namespace NUMINAMATH_CALUDE_pam_has_ten_bags_l2114_211467

/-- Represents the number of apples in one of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- Represents the total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- Represents the number of Gerald's bags equivalent to one of Pam's bags -/
def bags_ratio : ℕ := 3

/-- Calculates the number of bags Pam has -/
def pams_bag_count : ℕ := pams_total_apples / (bags_ratio * geralds_bag_count)

/-- Theorem stating that Pam has 10 bags of apples -/
theorem pam_has_ten_bags : pams_bag_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_pam_has_ten_bags_l2114_211467


namespace NUMINAMATH_CALUDE_even_product_probability_l2114_211471

-- Define the spinners
def spinner1 : List ℕ := [0, 2]
def spinner2 : List ℕ := [1, 3, 5]

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := n % 2 = 0

-- Define a function to calculate the probability of an even product
def probEvenProduct (s1 s2 : List ℕ) : ℚ :=
  let totalOutcomes := s1.length * s2.length
  let evenOutcomes := (s1.filter isEven).length * s2.length
  evenOutcomes / totalOutcomes

-- Theorem statement
theorem even_product_probability :
  probEvenProduct spinner1 spinner2 = 1 := by sorry

end NUMINAMATH_CALUDE_even_product_probability_l2114_211471


namespace NUMINAMATH_CALUDE_number_count_l2114_211451

theorem number_count (n : ℕ) (S : ℝ) : 
  S / n = 60 →                  -- average of all numbers is 60
  (58 * 6 : ℝ) = S / n * 6 →    -- average of first 6 numbers is 58
  (65 * 6 : ℝ) = S / n * 6 →    -- average of last 6 numbers is 65
  78 = S / n →                  -- 6th number is 78
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_number_count_l2114_211451


namespace NUMINAMATH_CALUDE_power_multiplication_l2114_211430

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2114_211430


namespace NUMINAMATH_CALUDE_apple_orange_ratio_l2114_211477

theorem apple_orange_ratio (num_oranges : ℕ) : 
  (15 : ℚ) + num_oranges = 50 * (3/2) → 
  (15 : ℚ) / num_oranges = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_apple_orange_ratio_l2114_211477


namespace NUMINAMATH_CALUDE_zebra_sleeps_longer_l2114_211409

/-- Proves that a zebra sleeps 2 hours more per night than a cougar, given the conditions -/
theorem zebra_sleeps_longer (cougar_sleep : ℕ) (total_sleep : ℕ) : 
  cougar_sleep = 4 →
  total_sleep = 70 →
  (total_sleep - 7 * cougar_sleep) / 7 - cougar_sleep = 2 := by
sorry

end NUMINAMATH_CALUDE_zebra_sleeps_longer_l2114_211409


namespace NUMINAMATH_CALUDE_quadratic_necessary_not_sufficient_l2114_211461

theorem quadratic_necessary_not_sufficient :
  (∀ x : ℝ, x > 2 → x^2 + 5*x - 6 > 0) ∧
  (∃ x : ℝ, x^2 + 5*x - 6 > 0 ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_necessary_not_sufficient_l2114_211461


namespace NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l2114_211421

theorem dogwood_trees_planted_tomorrow 
  (initial_trees : ℕ) 
  (planted_today : ℕ) 
  (final_total : ℕ) :
  initial_trees = 7 →
  planted_today = 3 →
  final_total = 12 →
  final_total - (initial_trees + planted_today) = 2 :=
by sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l2114_211421


namespace NUMINAMATH_CALUDE_circle_C_equation_chord_AB_length_line_MN_equation_l2114_211435

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define the line l₁
def line_l1 : Set (ℝ × ℝ) := {p | p.1 - p.2 - 2 * Real.sqrt 2 = 0}

-- Define the line l₂
def line_l2 : Set (ℝ × ℝ) := {p | 4 * p.1 - 3 * p.2 + 5 = 0}

-- Define point G
def point_G : ℝ × ℝ := (1, 3)

-- Statement 1: Equation of circle C
theorem circle_C_equation : circle_C = {p : ℝ × ℝ | p.1^2 + p.2^2 = 4} := by sorry

-- Statement 2: Length of chord AB
theorem chord_AB_length : 
  let chord_AB := circle_C ∩ line_l2
  (Set.ncard chord_AB = 2) → 
  ∃ a b : ℝ × ℝ, a ∈ chord_AB ∧ b ∈ chord_AB ∧ a ≠ b ∧ 
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 2 * Real.sqrt 3 := by sorry

-- Statement 3: Equation of line MN
theorem line_MN_equation : 
  ∃ M N : ℝ × ℝ, 
    M ∈ circle_C ∧ N ∈ circle_C ∧ M ≠ N ∧
    ((point_G.1 - M.1)^2 + (point_G.2 - M.2)^2) * 4 = ((M.1)^2 + (M.2)^2) * ((point_G.1)^2 + (point_G.2)^2) ∧
    ((point_G.1 - N.1)^2 + (point_G.2 - N.2)^2) * 4 = ((N.1)^2 + (N.2)^2) * ((point_G.1)^2 + (point_G.2)^2) ∧
    (∀ p : ℝ × ℝ, p ∈ {q | q.1 + 3 * q.2 - 4 = 0} ↔ (p.1 - M.1) * (N.2 - M.2) = (p.2 - M.2) * (N.1 - M.1)) := by sorry

end NUMINAMATH_CALUDE_circle_C_equation_chord_AB_length_line_MN_equation_l2114_211435


namespace NUMINAMATH_CALUDE_calculation_proofs_l2114_211407

theorem calculation_proofs :
  (7 - (-1/2) + 3/2 = 9) ∧
  ((-1)^99 + (1-5)^2 * (3/8) = 5) ∧
  (-2^3 * (5/8) / (-1/3) - 6 * (2/3 - 1/2) = 14) := by
sorry

end NUMINAMATH_CALUDE_calculation_proofs_l2114_211407


namespace NUMINAMATH_CALUDE_problem_solution_l2114_211470

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sqrt 3 * Real.sin x + Real.cos x) - 1

theorem problem_solution :
  (∀ x ∈ Set.Icc 0 (π/4), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc 0 (π/4), f x ≥ 1) ∧
  (∀ x₀ ∈ Set.Icc (π/4) (π/2), f x₀ = 6/5 → Real.cos (2*x₀) = (3 - 4*Real.sqrt 3)/10) ∧
  (∀ ω > 0, (∀ x ∈ Set.Ioo (π/3) (2*π/3), StrictMono (λ x => f (ω*x))) → ω ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2114_211470


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l2114_211468

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l2114_211468


namespace NUMINAMATH_CALUDE_zeros_in_Q_l2114_211428

/-- R_k represents an integer whose base-ten representation consists of k consecutive ones -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- Q is the quotient of R_30 and R_6 -/
def Q : ℕ := R 30 / R 6

/-- count_zeros counts the number of zeros in the base-ten representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 25 := by sorry

end NUMINAMATH_CALUDE_zeros_in_Q_l2114_211428


namespace NUMINAMATH_CALUDE_initial_water_ratio_l2114_211434

/-- Represents the tank and its properties --/
structure Tank where
  capacity : ℝ
  inflow_rate : ℝ
  outflow_rate1 : ℝ
  outflow_rate2 : ℝ
  fill_time : ℝ

/-- Calculates the net flow rate into the tank --/
def net_flow_rate (t : Tank) : ℝ :=
  t.inflow_rate - (t.outflow_rate1 + t.outflow_rate2)

/-- Calculates the initial amount of water in the tank --/
def initial_water (t : Tank) : ℝ :=
  t.capacity - (net_flow_rate t * t.fill_time)

/-- Theorem: The ratio of initial water to total capacity is 1:2 --/
theorem initial_water_ratio (t : Tank) 
  (h1 : t.capacity = 2)
  (h2 : t.inflow_rate = 0.5)
  (h3 : t.outflow_rate1 = 0.25)
  (h4 : t.outflow_rate2 = 1/6)
  (h5 : t.fill_time = 12) :
  initial_water t / t.capacity = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_ratio_l2114_211434


namespace NUMINAMATH_CALUDE_product_of_specific_numbers_l2114_211452

theorem product_of_specific_numbers : 469160 * 9999 = 4690696840 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_numbers_l2114_211452


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l2114_211459

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The atomic weight of Deuterium in g/mol -/
def atomic_weight_D : ℝ := 2.01

/-- The number of Barium atoms in the compound -/
def num_Ba : ℕ := 2

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The number of regular Hydrogen atoms in the compound -/
def num_H : ℕ := 4

/-- The number of Deuterium atoms in the compound -/
def num_D : ℕ := 1

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ :=
  (num_Ba : ℝ) * atomic_weight_Ba +
  (num_O : ℝ) * atomic_weight_O +
  (num_H : ℝ) * atomic_weight_H +
  (num_D : ℝ) * atomic_weight_D

theorem molecular_weight_calculation :
  molecular_weight = 328.71 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l2114_211459


namespace NUMINAMATH_CALUDE_sum_reciprocals_inequality_l2114_211482

theorem sum_reciprocals_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_inequality_l2114_211482


namespace NUMINAMATH_CALUDE_circle_area_solution_l2114_211473

theorem circle_area_solution :
  ∃! (x y z : ℕ), 6 * x + 15 * y + 83 * z = 220 ∧ x = 4 ∧ y = 2 ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_circle_area_solution_l2114_211473


namespace NUMINAMATH_CALUDE_evaluate_expression_l2114_211493

theorem evaluate_expression : 
  let sixteen : ℝ := 2^4
  let eight : ℝ := 2^3
  ∀ ε > 0, |Real.sqrt ((sixteen^15 + eight^20) / (sixteen^7 + eight^21)) - (1/2)| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2114_211493


namespace NUMINAMATH_CALUDE_perfect_square_property_l2114_211478

theorem perfect_square_property : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 8 * n + 1 = k * k) ∧ 
  (∃ (m : ℕ), n = 2 * m) ∧
  (∃ (p : ℕ), n = p * p) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_property_l2114_211478


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2114_211427

/-- Given f(x) = (2x-1)e^x - a(x^2+x) and g(x) = -ax^2 - a, where a ∈ ℝ,
    if f(x) ≥ g(x) for all x ∈ ℝ, then 1 ≤ a ≤ 4e^(3/2) -/
theorem function_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, (2*x - 1) * Real.exp x - a*(x^2 + x) ≥ -a*x^2 - a) →
  1 ≤ a ∧ a ≤ 4 * Real.exp (3/2) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2114_211427


namespace NUMINAMATH_CALUDE_hallies_art_earnings_l2114_211450

/-- Calculates the total money Hallie makes from her art -/
def total_money (prize : ℕ) (num_paintings : ℕ) (price_per_painting : ℕ) : ℕ :=
  prize + num_paintings * price_per_painting

/-- Proves that Hallie's total earnings from her art is $300 -/
theorem hallies_art_earnings : total_money 150 3 50 = 300 := by
  sorry

end NUMINAMATH_CALUDE_hallies_art_earnings_l2114_211450


namespace NUMINAMATH_CALUDE_max_expression_value_l2114_211484

def expression (x y z w : ℕ) : ℕ := x * y^z - w

theorem max_expression_value :
  ∃ (x y z w : ℕ),
    x ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    y ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    z ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    w ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
    expression x y z w = 24 ∧
    ∀ (a b c d : ℕ),
      a ∈ ({0, 1, 2, 3} : Set ℕ) →
      b ∈ ({0, 1, 2, 3} : Set ℕ) →
      c ∈ ({0, 1, 2, 3} : Set ℕ) →
      d ∈ ({0, 1, 2, 3} : Set ℕ) →
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
      expression a b c d ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_max_expression_value_l2114_211484


namespace NUMINAMATH_CALUDE_problem_statement_l2114_211418

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem problem_statement (a : ℝ) :
  (p a ↔ a ≤ 1) ∧
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a > 1 ∨ (-2 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l2114_211418


namespace NUMINAMATH_CALUDE_sequence_divisibility_l2114_211416

theorem sequence_divisibility (n : ℤ) : 
  (∃ k : ℤ, 7 * n - 3 = 5 * k) ∧ 
  (∀ m : ℤ, 7 * n - 3 ≠ 3 * m) ↔ 
  ∃ t : ℕ, n = 5 * t - 1 ∧ ∀ m : ℕ, t ≠ 3 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l2114_211416


namespace NUMINAMATH_CALUDE_max_value_4x_3y_l2114_211498

theorem max_value_4x_3y (x y : ℝ) : 
  x^2 + y^2 = 16*x + 8*y + 10 → 
  (4*x + 3*y ≤ (82.47 : ℝ) / 18) ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 16*x₀ + 8*y₀ + 10 ∧ 4*x₀ + 3*y₀ = (82.47 : ℝ) / 18 :=
by sorry

end NUMINAMATH_CALUDE_max_value_4x_3y_l2114_211498


namespace NUMINAMATH_CALUDE_sickness_temp_increase_l2114_211466

def normal_temp : ℝ := 95
def fever_threshold : ℝ := 100
def above_threshold : ℝ := 5

theorem sickness_temp_increase : 
  let current_temp := fever_threshold + above_threshold
  current_temp - normal_temp = 10 := by sorry

end NUMINAMATH_CALUDE_sickness_temp_increase_l2114_211466


namespace NUMINAMATH_CALUDE_D_180_l2114_211480

/-- 
D(n) represents the number of ways to express a positive integer n 
as a product of integers greater than 1, where the order matters.
-/
def D (n : ℕ+) : ℕ := sorry

/-- The prime factorization of 180 -/
def prime_factorization_180 : List ℕ+ := [2, 2, 3, 3, 5]

/-- Theorem stating that D(180) = 43 -/
theorem D_180 : D 180 = 43 := by sorry

end NUMINAMATH_CALUDE_D_180_l2114_211480


namespace NUMINAMATH_CALUDE_additional_chicken_wings_l2114_211441

theorem additional_chicken_wings 
  (num_friends : ℕ) 
  (pre_cooked_wings : ℕ) 
  (wings_per_friend : ℕ) : 
  num_friends = 4 → 
  pre_cooked_wings = 9 → 
  wings_per_friend = 4 → 
  num_friends * wings_per_friend - pre_cooked_wings = 7 := by
  sorry

end NUMINAMATH_CALUDE_additional_chicken_wings_l2114_211441


namespace NUMINAMATH_CALUDE_drama_club_adult_ticket_price_l2114_211442

/-- Calculates the adult ticket price for a drama club performance --/
theorem drama_club_adult_ticket_price 
  (total_tickets : ℕ) 
  (student_price : ℕ) 
  (total_amount : ℕ) 
  (student_count : ℕ) 
  (h1 : total_tickets = 1500)
  (h2 : student_price = 6)
  (h3 : total_amount = 16200)
  (h4 : student_count = 300) :
  ∃ (adult_price : ℕ), 
    (total_tickets - student_count) * adult_price + student_count * student_price = total_amount ∧ 
    adult_price = 12 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_adult_ticket_price_l2114_211442


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_l2114_211479

/-- Given a parabola y = x^2 - 3mx + m + n, prove that for the parabola to intersect
    the x-axis for all real numbers m, n must satisfy n ≤ -1/9 -/
theorem parabola_intersects_x_axis (n : ℝ) :
  (∀ m : ℝ, ∃ x : ℝ, x^2 - 3*m*x + m + n = 0) ↔ n ≤ -1/9 := by sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_l2114_211479


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2114_211460

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (A a) ∩ (B a) = {9} → a = -3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2114_211460


namespace NUMINAMATH_CALUDE_gym_purchase_theorem_l2114_211424

/-- Cost calculation for Option 1 -/
def costOption1 (x : ℕ) : ℚ :=
  1500 + 15 * (x - 20)

/-- Cost calculation for Option 2 -/
def costOption2 (x : ℕ) : ℚ :=
  (1500 + 15 * x) * (9/10)

/-- Cost calculation for the most cost-effective option -/
def costEffectiveOption (x : ℕ) : ℚ :=
  1500 + (x - 20) * 15 * (9/10)

theorem gym_purchase_theorem (x : ℕ) (h : x > 20) :
  (costOption1 40 < costOption2 40) ∧
  (costOption1 100 = costOption2 100) ∧
  (costEffectiveOption 40 < min (costOption1 40) (costOption2 40)) :=
by sorry

end NUMINAMATH_CALUDE_gym_purchase_theorem_l2114_211424


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l2114_211438

theorem partial_fraction_decomposition_product (x : ℝ) 
  (A B C : ℝ) : 
  (x^2 - 25) / (x^3 - 4*x^2 + x + 6) = 
  A / (x - 3) + B / (x + 1) + C / (x - 2) →
  A * B * C = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l2114_211438


namespace NUMINAMATH_CALUDE_sandy_shirt_cost_l2114_211495

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℝ := 13.99

/-- The amount Sandy received for returning a jacket -/
def jacket_return : ℝ := 7.43

/-- The net amount Sandy spent on clothes -/
def net_spend : ℝ := 18.7

/-- The amount Sandy spent on the shirt -/
def shirt_cost : ℝ := net_spend + jacket_return - shorts_cost

theorem sandy_shirt_cost : shirt_cost = 12.14 := by sorry

end NUMINAMATH_CALUDE_sandy_shirt_cost_l2114_211495


namespace NUMINAMATH_CALUDE_quadratic_equations_intersection_l2114_211426

theorem quadratic_equations_intersection (p q : ℝ) : 
  (∃ M N : Set ℝ, 
    (∀ x : ℝ, x ∈ M ↔ x^2 - p*x + 6 = 0) ∧
    (∀ x : ℝ, x ∈ N ↔ x^2 + 6*x - q = 0) ∧
    (M ∩ N = {2})) →
  p + q = 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_intersection_l2114_211426


namespace NUMINAMATH_CALUDE_factors_180_l2114_211496

/-- The number of positive factors of 180 -/
def num_factors_180 : ℕ :=
  (Finset.filter (· ∣ 180) (Finset.range 181)).card

/-- Theorem stating that the number of positive factors of 180 is 18 -/
theorem factors_180 : num_factors_180 = 18 := by
  sorry

end NUMINAMATH_CALUDE_factors_180_l2114_211496


namespace NUMINAMATH_CALUDE_belt_and_road_population_l2114_211406

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem belt_and_road_population : 
  toScientificNotation 4400000000 = ScientificNotation.mk 4.4 9 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_belt_and_road_population_l2114_211406


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2114_211449

/-- Given a geometric sequence where the first term is 1000 and the eighth term is 125,
    prove that the sixth term is 31.25. -/
theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)  -- The sequence
  (h1 : a 1 = 1000)  -- First term is 1000
  (h2 : a 8 = 125)   -- Eighth term is 125
  (h_geom : ∀ n, a (n + 1) = a n * (a 2 / a 1))  -- Geometric sequence property
  : a 6 = 31.25 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2114_211449


namespace NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2114_211494

theorem remainder_sum_mod_seven : (9^7 + 6^9 + 5^11) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2114_211494


namespace NUMINAMATH_CALUDE_cos_negative_300_degrees_l2114_211492

theorem cos_negative_300_degrees : Real.cos (-(300 * π / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_300_degrees_l2114_211492


namespace NUMINAMATH_CALUDE_figure_area_l2114_211443

theorem figure_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) 
  (h1 : rect1_width = 7 ∧ rect1_height = 7)
  (h2 : rect2_width = 3 ∧ rect2_height = 2)
  (h3 : rect3_width = 4 ∧ rect3_height = 4) :
  rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height = 71 := by
sorry

end NUMINAMATH_CALUDE_figure_area_l2114_211443


namespace NUMINAMATH_CALUDE_math_class_grade_distribution_l2114_211414

theorem math_class_grade_distribution (total_students : ℕ) 
  (prob_A : ℚ) (prob_B : ℚ) (prob_C : ℚ) : 
  total_students = 40 →
  prob_A = 0.8 * prob_B →
  prob_C = 1.2 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  ∃ (num_B : ℕ), num_B = 13 ∧ 
    (↑num_B : ℚ) * prob_B = (total_students : ℚ) * prob_B := by
  sorry

end NUMINAMATH_CALUDE_math_class_grade_distribution_l2114_211414


namespace NUMINAMATH_CALUDE_count_valid_integers_l2114_211444

/-- A function that returns true if a natural number is a four-digit positive integer -/
def isFourDigitPositive (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- A function that returns true if a natural number is divisible by 25 -/
def isDivisibleBy25 (n : ℕ) : Prop :=
  n % 25 = 0

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- A function that returns true if the sum of digits of a natural number is divisible by 3 -/
def sumOfDigitsDivisibleBy3 (n : ℕ) : Prop :=
  (sumOfDigits n) % 3 = 0

/-- The count of positive four-digit integers divisible by 25 with sum of digits divisible by 3 -/
def countValidIntegers : ℕ :=
  sorry

/-- Theorem stating that the count of valid integers satisfies all conditions -/
theorem count_valid_integers :
  ∃ (n : ℕ), n = countValidIntegers ∧
  ∀ (m : ℕ), (isFourDigitPositive m ∧ isDivisibleBy25 m ∧ sumOfDigitsDivisibleBy3 m) →
  (m ∈ Finset.range n) :=
  sorry

end NUMINAMATH_CALUDE_count_valid_integers_l2114_211444


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2114_211474

-- Problem 1
theorem problem_1 : 42.67 - (12.95 - 7.33) = 37.05 := by sorry

-- Problem 2
theorem problem_2 : (8.4 - 8.4 * (3.12 - 3.7)) / 0.42 = 31.6 := by sorry

-- Problem 3
theorem problem_3 : 5.13 * 0.23 + 8.7 * 0.513 - 5.13 = 0.513 := by sorry

-- Problem 4
theorem problem_4 : 6.66 * 222 + 3.33 * 556 = 3330 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2114_211474


namespace NUMINAMATH_CALUDE_evaluate_expression_l2114_211476

theorem evaluate_expression : (2^3001 * 3^3003) / 6^3002 = 3/2 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2114_211476


namespace NUMINAMATH_CALUDE_fraction_simplification_l2114_211420

theorem fraction_simplification (x y : ℚ) (hx : x = 2) (hy : y = 3) :
  (1 / y) / (1 / x) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2114_211420


namespace NUMINAMATH_CALUDE_original_banana_count_l2114_211439

/-- The number of bananas Willie and Charles originally had together -/
def total_bananas (willie_bananas : ℝ) (charles_bananas : ℝ) : ℝ :=
  willie_bananas + charles_bananas

/-- Theorem stating that Willie and Charles originally had 83.0 bananas together -/
theorem original_banana_count : total_bananas 48.0 35.0 = 83.0 := by
  sorry

end NUMINAMATH_CALUDE_original_banana_count_l2114_211439


namespace NUMINAMATH_CALUDE_hundreds_digit_of_expression_l2114_211475

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to get the hundreds digit
def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

-- State the theorem
theorem hundreds_digit_of_expression :
  hundreds_digit ((factorial 17 / 5) - (factorial 10 / 2)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_hundreds_digit_of_expression_l2114_211475


namespace NUMINAMATH_CALUDE_line_properties_l2114_211491

/-- Two lines in the plane, parameterized by a -/
def Line1 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a * x - y + 1 = 0

def Line2 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x + a * y + 1 = 0

/-- The theorem stating the properties of the two lines -/
theorem line_properties :
  ∀ a : ℝ,
    (∀ x y : ℝ, Line1 a x y → Line2 a x y → (a * 1 - 1 * a = 0)) ∧ 
    (Line1 a 0 1) ∧
    (Line2 a (-1) 0) ∧
    (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → Line1 a x y → Line2 a x y → x^2 + x + y^2 - y = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l2114_211491


namespace NUMINAMATH_CALUDE_probability_higher_first_lower_second_l2114_211445

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def favorable_outcomes (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s |>.filter (fun (a, b) => a > b)

theorem probability_higher_first_lower_second :
  (favorable_outcomes card_set).card / (card_set.card * card_set.card : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_higher_first_lower_second_l2114_211445


namespace NUMINAMATH_CALUDE_optimal_solution_l2114_211464

/-- Represents the prices and quantities of agricultural products A and B --/
structure AgriProducts where
  price_A : ℝ
  price_B : ℝ
  quantity_A : ℝ
  quantity_B : ℝ

/-- Defines the conditions given in the problem --/
def satisfies_conditions (p : AgriProducts) : Prop :=
  2 * p.price_A + 3 * p.price_B = 690 ∧
  p.price_A + 4 * p.price_B = 720 ∧
  p.quantity_A + p.quantity_B = 40 ∧
  p.price_A * p.quantity_A + p.price_B * p.quantity_B ≤ 5400 ∧
  p.quantity_A ≤ 3 * p.quantity_B

/-- Calculates the profit given the prices and quantities --/
def profit (p : AgriProducts) : ℝ :=
  (160 - p.price_A) * p.quantity_A + (200 - p.price_B) * p.quantity_B

/-- Theorem stating the optimal solution --/
theorem optimal_solution :
  ∃ (p : AgriProducts),
    satisfies_conditions p ∧
    p.price_A = 120 ∧
    p.price_B = 150 ∧
    p.quantity_A = 20 ∧
    p.quantity_B = 20 ∧
    ∀ (q : AgriProducts), satisfies_conditions q → profit q ≤ profit p :=
  sorry


end NUMINAMATH_CALUDE_optimal_solution_l2114_211464


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l2114_211486

-- Define the points A and B
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the condition for point C
def condition (C : ℝ × ℝ) : Prop :=
  let (x, y) := C
  (x - 3) * (x + 1) + y * y = 5

-- Define the line l
def line_l (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x - y + 3 = 0

-- State the theorem
theorem trajectory_and_intersection :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ C, condition C ↔ (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2) ∧
    (center = (1, 0) ∧ radius = 3) ∧
    (∃ M N : ℝ × ℝ,
      M ≠ N ∧
      condition M ∧ condition N ∧
      line_l M ∧ line_l N ∧
      ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l2114_211486


namespace NUMINAMATH_CALUDE_cubic_roots_properties_l2114_211462

theorem cubic_roots_properties (x₁ x₂ x₃ : ℝ) :
  (x₁^3 - 17*x₁ - 18 = 0) →
  (x₂^3 - 17*x₂ - 18 = 0) →
  (x₃^3 - 17*x₃ - 18 = 0) →
  (-4 < x₁) → (x₁ < -3) →
  (4 < x₃) → (x₃ < 5) →
  (⌊x₂⌋ = -2) ∧ (Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = -π/4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_properties_l2114_211462


namespace NUMINAMATH_CALUDE_carrie_highlighters_l2114_211497

/-- The total number of highlighters in Carrie's desk drawer -/
def total_highlighters (y p b o g : ℕ) : ℕ := y + p + b + o + g

/-- Theorem stating the total number of highlighters in Carrie's desk drawer -/
theorem carrie_highlighters : ∃ (y p b o g : ℕ),
  y = 7 ∧
  p = y + 7 ∧
  b = p + 5 ∧
  o + g = 21 ∧
  o * 7 = g * 3 ∧
  total_highlighters y p b o g = 61 :=
by sorry

end NUMINAMATH_CALUDE_carrie_highlighters_l2114_211497


namespace NUMINAMATH_CALUDE_minimum_value_of_polynomial_l2114_211487

theorem minimum_value_of_polynomial (a b : ℝ) : 
  2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a + 4 * b + 1999 ≥ 1947 ∧ 
  ∃ (a b : ℝ), 2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a + 4 * b + 1999 = 1947 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_polynomial_l2114_211487


namespace NUMINAMATH_CALUDE_missing_number_proof_l2114_211403

theorem missing_number_proof (numbers : List ℕ) (missing : ℕ) : 
  numbers = [744, 745, 747, 748, 749, 753, 755, 755] →
  (numbers.sum + missing) / 9 = 750 →
  missing = 804 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2114_211403


namespace NUMINAMATH_CALUDE_roses_given_to_friends_l2114_211465

def total_money : ℕ := 300
def rose_price : ℕ := 2
def jenna_fraction : ℚ := 1/3
def imma_fraction : ℚ := 1/2

theorem roses_given_to_friends :
  let total_roses := total_money / rose_price
  let jenna_roses := (jenna_fraction * total_roses).floor
  let imma_roses := (imma_fraction * total_roses).floor
  jenna_roses + imma_roses = 125 := by sorry

end NUMINAMATH_CALUDE_roses_given_to_friends_l2114_211465


namespace NUMINAMATH_CALUDE_equation_solution_l2114_211485

theorem equation_solution : ∃ x : ℕ, 9^12 + 9^12 + 9^12 = 3^x ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2114_211485


namespace NUMINAMATH_CALUDE_not_perfect_square_2005_l2114_211402

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluates a polynomial at a given point -/
def eval (P : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  sorry

theorem not_perfect_square_2005 (P : IntPolynomial) :
  eval P 5 = 2005 → ¬(is_perfect_square (eval P 2005)) :=
sorry

end NUMINAMATH_CALUDE_not_perfect_square_2005_l2114_211402


namespace NUMINAMATH_CALUDE_sequence_bound_l2114_211431

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i, 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j, i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bound_l2114_211431


namespace NUMINAMATH_CALUDE_rotation_implies_equilateral_l2114_211417

-- Define the triangle
variable (A₁ A₂ A₃ : ℝ × ℝ)

-- Define the rotation function
def rotate (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the sequence of rotations
def rotate_sequence (n : ℕ) (P₀ : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define equilateral triangle
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  sorry

theorem rotation_implies_equilateral 
  (P₀ : ℝ × ℝ) 
  (h : rotate_sequence 1986 P₀ = P₀) : 
  is_equilateral A₁ A₂ A₃ :=
sorry

end NUMINAMATH_CALUDE_rotation_implies_equilateral_l2114_211417


namespace NUMINAMATH_CALUDE_raja_income_distribution_l2114_211429

theorem raja_income_distribution (monthly_income : ℝ) 
  (household_percentage : ℝ) (medicine_percentage : ℝ) (savings : ℝ) :
  monthly_income = 37500 →
  household_percentage = 35 →
  medicine_percentage = 5 →
  savings = 15000 →
  ∃ (clothes_percentage : ℝ),
    clothes_percentage = 20 ∧
    (household_percentage / 100 + medicine_percentage / 100 + clothes_percentage / 100) * monthly_income + savings = monthly_income :=
by sorry

end NUMINAMATH_CALUDE_raja_income_distribution_l2114_211429


namespace NUMINAMATH_CALUDE_kamal_math_marks_l2114_211433

/-- Calculates Kamal's marks in Mathematics given his marks in other subjects and his average -/
theorem kamal_math_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) 
  (h1 : english = 76)
  (h2 : physics = 82)
  (h3 : chemistry = 67)
  (h4 : biology = 85)
  (h5 : average = 74) :
  let total := average * 5
  let math := total - (english + physics + chemistry + biology)
  math = 60 := by sorry

end NUMINAMATH_CALUDE_kamal_math_marks_l2114_211433


namespace NUMINAMATH_CALUDE_cube_root_inequality_l2114_211446

theorem cube_root_inequality (x : ℝ) : 
  (x ^ (1/3) : ℝ) - 3 / ((x ^ (1/3) : ℝ) + 4) ≤ 0 ↔ -27 < x ∧ x < -1 := by sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l2114_211446


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2114_211419

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h1 : a 5 * a 14 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2114_211419


namespace NUMINAMATH_CALUDE_triathlon_running_speed_l2114_211423

/-- Calculates the running speed given swimming speed and average speed -/
def calculate_running_speed (swimming_speed : ℝ) (average_speed : ℝ) : ℝ :=
  2 * average_speed - swimming_speed

/-- Proves that given a swimming speed of 1 mph and an average speed of 4 mph,
    the running speed is 7 mph -/
theorem triathlon_running_speed :
  let swimming_speed : ℝ := 1
  let average_speed : ℝ := 4
  calculate_running_speed swimming_speed average_speed = 7 := by
sorry

#eval calculate_running_speed 1 4

end NUMINAMATH_CALUDE_triathlon_running_speed_l2114_211423


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2114_211457

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + 20 + 3*x + 15 + (3*x + 6)) / 5 = 30 → x = 99 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2114_211457


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l2114_211453

theorem unique_solution_logarithmic_equation :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.log (x^3 + (1/3) * y^3 + 1/9) = Real.log x + Real.log y := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l2114_211453


namespace NUMINAMATH_CALUDE_find_n_l2114_211455

theorem find_n (n : ℕ) : 
  (Nat.lcm n 14 = 56) → (Nat.gcd n 14 = 12) → n = 48 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2114_211455


namespace NUMINAMATH_CALUDE_emily_candy_distribution_l2114_211454

/-- Given that Emily has 34 pieces of candy and 5 friends, prove that she needs to remove 4 pieces
    to distribute the remaining candies equally among her friends. -/
theorem emily_candy_distribution (total_candy : Nat) (num_friends : Nat) 
    (h1 : total_candy = 34) (h2 : num_friends = 5) :
    ∃ (removed : Nat) (distributed : Nat),
      removed = 4 ∧
      distributed * num_friends = total_candy - removed ∧
      ∀ r, r < removed → ¬∃ d, d * num_friends = total_candy - r :=
by sorry

end NUMINAMATH_CALUDE_emily_candy_distribution_l2114_211454


namespace NUMINAMATH_CALUDE_dave_initial_apps_l2114_211413

/-- The number of apps Dave initially had on his phone -/
def initial_apps : ℕ := sorry

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 18

/-- The number of apps remaining after deletion -/
def remaining_apps : ℕ := 5

/-- Theorem stating that Dave initially had 23 apps -/
theorem dave_initial_apps : initial_apps = 23 := by
  sorry

end NUMINAMATH_CALUDE_dave_initial_apps_l2114_211413


namespace NUMINAMATH_CALUDE_inequality_solution_l2114_211469

theorem inequality_solution (x : ℝ) : 3 * x - 6 > 5 * (x - 2) → x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2114_211469


namespace NUMINAMATH_CALUDE_executive_board_count_l2114_211490

/-- The number of ways to choose an executive board from a club -/
def choose_executive_board (total_members : ℕ) (board_size : ℕ) (specific_roles : ℕ) : ℕ :=
  Nat.choose total_members board_size * (board_size * (board_size - 1))

/-- Theorem stating the number of ways to choose the executive board -/
theorem executive_board_count :
  choose_executive_board 40 6 2 = 115151400 := by
  sorry

end NUMINAMATH_CALUDE_executive_board_count_l2114_211490


namespace NUMINAMATH_CALUDE_f_1000_value_l2114_211488

def is_multiplicative_to_additive (f : ℕ+ → ℕ) : Prop :=
  ∀ x y : ℕ+, f (x * y) = f x + f y

theorem f_1000_value
  (f : ℕ+ → ℕ)
  (h_mult_add : is_multiplicative_to_additive f)
  (h_10 : f 10 = 16)
  (h_40 : f 40 = 22) :
  f 1000 = 48 :=
sorry

end NUMINAMATH_CALUDE_f_1000_value_l2114_211488


namespace NUMINAMATH_CALUDE_candy_probability_l2114_211489

def yellow_candies : ℕ := 2
def red_candies : ℕ := 4

def total_candies : ℕ := yellow_candies + red_candies

def favorable_arrangements : ℕ := 1

def total_arrangements : ℕ := Nat.choose total_candies yellow_candies

def probability : ℚ := favorable_arrangements / total_arrangements

theorem candy_probability : probability = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_candy_probability_l2114_211489
