import Mathlib

namespace prob_no_adjacent_birch_is_2_55_l118_11845

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of trees -/
def total_trees : ℕ := 15

/-- The number of birch trees -/
def birch_trees : ℕ := 6

/-- The number of non-birch trees -/
def non_birch_trees : ℕ := total_trees - birch_trees

/-- The probability of no two birch trees being adjacent when arranged randomly -/
def prob_no_adjacent_birch : ℚ := 
  choose (non_birch_trees + 1) birch_trees / choose total_trees birch_trees

theorem prob_no_adjacent_birch_is_2_55 : 
  prob_no_adjacent_birch = 2 / 55 := by sorry

end prob_no_adjacent_birch_is_2_55_l118_11845


namespace jump_rope_record_rate_l118_11814

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The record number of consecutive ropes jumped -/
def record_jumps : ℕ := 54000

/-- The time limit in hours -/
def time_limit : ℕ := 5

/-- The required rate of jumps per second -/
def required_rate : ℚ := 3

theorem jump_rope_record_rate :
  (record_jumps : ℚ) / ((time_limit * seconds_per_hour) : ℚ) = required_rate :=
sorry

end jump_rope_record_rate_l118_11814


namespace derivative_sin_minus_cos_at_pi_l118_11885

open Real

theorem derivative_sin_minus_cos_at_pi :
  let f : ℝ → ℝ := fun x ↦ sin x - cos x
  deriv f π = -1 := by
  sorry

end derivative_sin_minus_cos_at_pi_l118_11885


namespace female_democrats_count_l118_11874

theorem female_democrats_count 
  (total : ℕ) 
  (female : ℕ) 
  (male : ℕ) 
  (h1 : total = 750)
  (h2 : female + male = total)
  (h3 : (female / 2 + male / 4 : ℚ) = total / 3) :
  female / 2 = 125 := by
sorry

end female_democrats_count_l118_11874


namespace union_when_m_is_neg_one_subset_iff_m_leq_neg_two_disjoint_iff_m_geq_zero_l118_11806

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Statement 1
theorem union_when_m_is_neg_one : 
  A ∪ B (-1) = {x : ℝ | -2 < x ∧ x < 3} := by sorry

-- Statement 2
theorem subset_iff_m_leq_neg_two :
  ∀ m : ℝ, A ⊆ B m ↔ m ≤ -2 := by sorry

-- Statement 3
theorem disjoint_iff_m_geq_zero :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ≥ 0 := by sorry

end union_when_m_is_neg_one_subset_iff_m_leq_neg_two_disjoint_iff_m_geq_zero_l118_11806


namespace first_equation_is_double_root_second_equation_coefficients_l118_11801

-- Definition of a double root equation
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 2 * x

-- Theorem 1
theorem first_equation_is_double_root :
  is_double_root_equation 1 (-3) 2 :=
sorry

-- Theorem 2
theorem second_equation_coefficients (a b : ℝ) :
  is_double_root_equation a b (-6) ∧ (a * 2^2 + b * 2 - 6 = 0) →
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) :=
sorry

end first_equation_is_double_root_second_equation_coefficients_l118_11801


namespace library_capacity_is_400_l118_11879

/-- The capacity of Karson's home library -/
def library_capacity : ℕ := sorry

/-- The number of books Karson currently has -/
def current_books : ℕ := 120

/-- The number of additional books Karson needs to buy -/
def additional_books : ℕ := 240

/-- The percentage of the library that will be full after buying additional books -/
def full_percentage : ℚ := 9/10

theorem library_capacity_is_400 : 
  library_capacity = 400 :=
by
  have h1 : current_books + additional_books = (library_capacity : ℚ) * full_percentage :=
    sorry
  sorry

end library_capacity_is_400_l118_11879


namespace cube_distance_theorem_l118_11818

/-- Represents a cube with a specific configuration above a plane -/
structure CubeAbovePlane where
  side_length : ℝ
  adjacent_heights : Fin 3 → ℝ
  distance_numerator : ℕ
  distance_denominator : ℕ

/-- The specific cube configuration given in the problem -/
def problem_cube : CubeAbovePlane :=
  { side_length := 12
  , adjacent_heights := ![13, 14, 16]
  , distance_numerator := 9
  , distance_denominator := 1 }

/-- Theorem stating the properties of the cube's distance from the plane -/
theorem cube_distance_theorem (cube : CubeAbovePlane) 
  (h_side : cube.side_length = 12)
  (h_heights : cube.adjacent_heights = ![13, 14, 16])
  (h_distance : ∃ (p q u : ℕ), p + q + u < 1200 ∧ 
    (cube.distance_numerator : ℝ) / cube.distance_denominator = p - Real.sqrt q) :
  cube.distance_numerator = 9 ∧ cube.distance_denominator = 1 := by
  sorry

end cube_distance_theorem_l118_11818


namespace test_score_ratio_l118_11868

theorem test_score_ratio (total_questions : ℕ) (score : ℕ) (correct_answers : ℕ)
  (h1 : total_questions = 100)
  (h2 : score = 79)
  (h3 : correct_answers = 93)
  (h4 : correct_answers ≤ total_questions) :
  (total_questions - correct_answers) / correct_answers = 7 / 93 := by
sorry

end test_score_ratio_l118_11868


namespace no_positive_solutions_iff_p_in_range_l118_11847

/-- The set A of real solutions to the quadratic equation x^2 + (p + 2)x + 1 = 0 -/
def A (p : ℝ) : Set ℝ :=
  {x : ℝ | x^2 + (p + 2)*x + 1 = 0}

/-- The theorem stating the equivalence between A having no positive real solutions
    and p belonging to the specified range -/
theorem no_positive_solutions_iff_p_in_range (p : ℝ) :
  (A p ∩ Set.Ici 0 = ∅) ↔ p ∈ Set.Ioo (-4) 0 ∪ Set.Ici 0 := by
  sorry

end no_positive_solutions_iff_p_in_range_l118_11847


namespace arithmetic_sequence_property_l118_11804

/-- For an arithmetic sequence with general term a_n = 3n - 4, 
    the difference between the first term and the common difference is -4. -/
theorem arithmetic_sequence_property : 
  ∀ (a : ℕ → ℤ), 
  (∀ n, a n = 3*n - 4) → 
  (a 1 - (a 2 - a 1) = -4) :=
by sorry

end arithmetic_sequence_property_l118_11804


namespace intersection_point_satisfies_equations_intersection_point_unique_l118_11830

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (69/29, 43/29)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 5*x - 6*y = 3

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 8*x + 2*y = 22

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique :
  ∀ x y : ℚ, line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end intersection_point_satisfies_equations_intersection_point_unique_l118_11830


namespace cone_lateral_area_l118_11842

/-- The lateral area of a cone with base radius 3 and height 4 is 15π -/
theorem cone_lateral_area : 
  let r : ℝ := 3
  let h : ℝ := 4
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let S : ℝ := π * r * l
  S = 15 * π := by sorry

end cone_lateral_area_l118_11842


namespace divisibility_of_repeated_eight_l118_11870

theorem divisibility_of_repeated_eight : ∃ k : ℕ, 8 * (10^1974 - 1) / 9 = 13 * k := by
  sorry

end divisibility_of_repeated_eight_l118_11870


namespace choir_average_age_l118_11813

theorem choir_average_age (num_females : ℕ) (num_males : ℕ) 
  (avg_age_females : ℚ) (avg_age_males : ℚ) :
  num_females = 12 →
  num_males = 18 →
  avg_age_females = 28 →
  avg_age_males = 38 →
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 34 := by
  sorry

end choir_average_age_l118_11813


namespace student_sample_size_l118_11876

/-- Represents the frequency distribution of student weights --/
structure WeightDistribution where
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ
  remaining : ℕ

/-- The total number of students in the sample --/
def total_students (w : WeightDistribution) : ℕ :=
  w.group1 + w.group2 + w.group3 + w.remaining

/-- The given conditions for the weight distribution --/
def weight_distribution_conditions (w : WeightDistribution) : Prop :=
  w.group1 + w.group2 + w.group3 > 0 ∧
  w.group2 = 12 ∧
  w.group2 = 2 * w.group1 ∧
  w.group3 = 3 * w.group1

theorem student_sample_size :
  ∃ w : WeightDistribution, weight_distribution_conditions w ∧ total_students w = 48 :=
sorry

end student_sample_size_l118_11876


namespace correct_operation_l118_11827

theorem correct_operation (a b : ℝ) : -a^2*b + 2*a^2*b = a^2*b := by
  sorry

end correct_operation_l118_11827


namespace double_y_plus_8_not_less_than_negative_3_l118_11831

theorem double_y_plus_8_not_less_than_negative_3 :
  ∀ y : ℝ, (2 * y + 8 ≥ -3) ↔ (∃ z : ℝ, z = 2 * y ∧ z + 8 ≥ -3) :=
by sorry

end double_y_plus_8_not_less_than_negative_3_l118_11831


namespace tan_sqrt_three_iff_periodic_l118_11851

theorem tan_sqrt_three_iff_periodic (x : ℝ) : 
  Real.tan x = Real.sqrt 3 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 3 := by
  sorry

end tan_sqrt_three_iff_periodic_l118_11851


namespace line_symmetry_l118_11854

-- Define the lines
def line_l (x y : ℝ) : Prop := 3 * x - y + 3 = 0
def line_1 (x y : ℝ) : Prop := x - y - 2 = 0
def line_2 (x y : ℝ) : Prop := 7 * x + y + 22 = 0

-- Define symmetry with respect to line_l
def symmetric_wrt_l (x y x' y' : ℝ) : Prop :=
  -- The product of the slopes of PP' and line_l is -1
  ((y' - y) / (x' - x)) * 3 = -1 ∧
  -- The midpoint of PP' lies on line_l
  3 * ((x + x') / 2) - ((y + y') / 2) + 3 = 0

-- Theorem statement
theorem line_symmetry :
  ∀ x y x' y' : ℝ,
    line_1 x y ∧ line_2 x' y' →
    symmetric_wrt_l x y x' y' :=
sorry

end line_symmetry_l118_11854


namespace final_sum_after_transformations_l118_11820

/-- Given two numbers x and y with sum T, prove that after transformations, 
    the sum of the resulting numbers is 4T + 22 -/
theorem final_sum_after_transformations (x y T : ℝ) (h : x + y = T) :
  3 * (x + 4) + 2 * (y + 5) = 4 * T + 22 := by
  sorry

end final_sum_after_transformations_l118_11820


namespace martha_apples_l118_11899

/-- Given Martha's initial apples and the distribution to her friends, 
    prove the number of additional apples she needs to give away to be left with 4. -/
theorem martha_apples (initial_apples : ℕ) (jane_apples : ℕ) (james_extra : ℕ) :
  initial_apples = 20 →
  jane_apples = 5 →
  james_extra = 2 →
  initial_apples - jane_apples - (jane_apples + james_extra) - 4 = 4 :=
by sorry

end martha_apples_l118_11899


namespace sum_of_squares_of_roots_l118_11828

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 15*x + 6 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 15 ∧ r₁ * r₂ = 6 ∧ r₁^2 + r₂^2 = 213 := by
  sorry

end sum_of_squares_of_roots_l118_11828


namespace tan_greater_than_cubic_l118_11884

theorem tan_greater_than_cubic (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  Real.tan x > x + (1 / 3) * x^3 := by
  sorry

end tan_greater_than_cubic_l118_11884


namespace leaf_fall_problem_l118_11832

/-- The rate of leaves falling per hour in the second and third hour -/
def leaf_fall_rate (first_hour : ℕ) (average : ℚ) : ℚ :=
  (3 * average - first_hour) / 2

theorem leaf_fall_problem (first_hour : ℕ) (average : ℚ) :
  first_hour = 7 →
  average = 5 →
  leaf_fall_rate first_hour average = 4 := by
sorry

end leaf_fall_problem_l118_11832


namespace ladder_angle_elevation_l118_11810

def ladder_foot_distance : ℝ := 4.6
def ladder_length : ℝ := 9.2

theorem ladder_angle_elevation :
  let cos_angle := ladder_foot_distance / ladder_length
  let angle := Real.arccos cos_angle
  ∃ ε > 0, abs (angle - Real.pi / 3) < ε :=
by
  sorry

end ladder_angle_elevation_l118_11810


namespace prob_king_hearts_or_spade_l118_11833

-- Define the total number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of spades in the deck
def num_spades : ℕ := 13

-- Define the probability of drawing the King of Hearts
def prob_king_hearts : ℚ := 1 / total_cards

-- Define the probability of drawing a Spade
def prob_spade : ℚ := num_spades / total_cards

-- Theorem to prove
theorem prob_king_hearts_or_spade :
  prob_king_hearts + prob_spade = 7 / 26 := by
  sorry

end prob_king_hearts_or_spade_l118_11833


namespace imaginary_part_of_z_l118_11825

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2 - Complex.I) :
  z.im = -3/2 := by
  sorry

end imaginary_part_of_z_l118_11825


namespace newspaper_cost_theorem_l118_11863

/-- The cost of a weekday newspaper -/
def weekday_cost : ℚ := 1/2

/-- The cost of a Sunday newspaper -/
def sunday_cost : ℚ := 2

/-- The number of weekday newspapers bought per week -/
def weekday_papers_per_week : ℕ := 3

/-- The number of weeks -/
def num_weeks : ℕ := 8

/-- The total cost of newspapers over the given number of weeks -/
def total_cost : ℚ := num_weeks * (weekday_papers_per_week * weekday_cost + sunday_cost)

theorem newspaper_cost_theorem : total_cost = 28 := by
  sorry

end newspaper_cost_theorem_l118_11863


namespace percentage_equivalence_l118_11809

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 := by
  sorry

end percentage_equivalence_l118_11809


namespace rectangle_ratio_l118_11890

/-- Given a rectangular plot with area 363 sq m and breadth 11 m, 
    prove that the ratio of length to breadth is 3:1 -/
theorem rectangle_ratio : ∀ (length breadth : ℝ),
  breadth = 11 →
  length * breadth = 363 →
  length / breadth = 3 := by
  sorry

end rectangle_ratio_l118_11890


namespace triangle_radius_equations_l118_11836

/-- Given a triangle ABC with angles 2α, 2β, and 2γ, prove two equations involving inradius, exradii, and side lengths. -/
theorem triangle_radius_equations (R α β γ : ℝ) (r r_a r_b r_c a b c : ℝ) 
  (h_r : r = 4 * R * Real.sin α * Real.sin β * Real.sin γ)
  (h_ra : r_a = 4 * R * Real.sin α * Real.cos β * Real.cos γ)
  (h_rb : r_b = 4 * R * Real.cos α * Real.sin β * Real.cos γ)
  (h_rc : r_c = 4 * R * Real.cos α * Real.cos β * Real.sin γ)
  (h_a : a = 4 * R * Real.sin α * Real.cos α)
  (h_bc : b + c = 4 * R * Real.sin (β + γ) * Real.cos (β - γ)) :
  (a * (b + c) = (r + r_a) * (4 * R + r - r_a)) ∧ 
  (a * (b - c) = (r_b - r_c) * (4 * R - r_b - r_c)) := by
  sorry

end triangle_radius_equations_l118_11836


namespace ribbon_solution_l118_11886

def ribbon_problem (total : ℝ) : Prop :=
  let remaining_after_first := total / 2
  let remaining_after_second := remaining_after_first * 2 / 3
  let remaining_after_third := remaining_after_second / 2
  remaining_after_third = 250

theorem ribbon_solution :
  ribbon_problem 1500 := by sorry

end ribbon_solution_l118_11886


namespace quadratic_equation_sum_l118_11824

theorem quadratic_equation_sum (x p q : ℝ) : 
  (5 * x^2 - 30 * x - 45 = 0) → 
  ((x + p)^2 = q) → 
  (p + q = 15) := by
sorry

end quadratic_equation_sum_l118_11824


namespace min_both_beethoven_vivaldi_l118_11871

/-- The minimum number of people who like both Beethoven and Vivaldi in a group of 120 people,
    where 95 like Beethoven and 80 like Vivaldi. -/
theorem min_both_beethoven_vivaldi (total : ℕ) (beethoven : ℕ) (vivaldi : ℕ)
    (h_total : total = 120)
    (h_beethoven : beethoven = 95)
    (h_vivaldi : vivaldi = 80) :
    beethoven + vivaldi - total ≥ 55 := by
  sorry

end min_both_beethoven_vivaldi_l118_11871


namespace consecutive_even_sum_squares_l118_11898

theorem consecutive_even_sum_squares (a b c d : ℕ) : 
  (∃ n : ℕ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) →
  (a + b + c + d = 36) →
  (a^2 + b^2 + c^2 + d^2 = 344) :=
by sorry

end consecutive_even_sum_squares_l118_11898


namespace solution_value_l118_11811

theorem solution_value (m : ℝ) : (3 * m - 2 * 3 = 6) → m = 4 := by
  sorry

end solution_value_l118_11811


namespace susie_bob_ratio_l118_11856

-- Define the number of slices for each pizza size
def small_pizza_slices : ℕ := 4
def large_pizza_slices : ℕ := 8

-- Define the number of pizzas George purchased
def small_pizzas_bought : ℕ := 3
def large_pizzas_bought : ℕ := 2

-- Define the number of pieces eaten by each person
def george_pieces : ℕ := 3
def bob_pieces : ℕ := george_pieces + 1
def bill_fred_mark_pieces : ℕ := 3 * 3

-- Define the number of slices left over
def leftover_slices : ℕ := 10

-- Calculate the total number of slices
def total_slices : ℕ := small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought

-- Define Susie's pieces as a function of the other variables
def susie_pieces : ℕ := total_slices - leftover_slices - (george_pieces + bob_pieces + bill_fred_mark_pieces)

-- Theorem to prove
theorem susie_bob_ratio :
  susie_pieces * 2 = bob_pieces := by sorry

end susie_bob_ratio_l118_11856


namespace employee_pay_calculation_l118_11823

def total_pay : ℝ := 570
def x_pay_ratio : ℝ := 1.2

theorem employee_pay_calculation (x y : ℝ) 
  (h1 : x + y = total_pay) 
  (h2 : x = x_pay_ratio * y) : 
  y = 259.09 := by sorry

end employee_pay_calculation_l118_11823


namespace box_depth_l118_11802

theorem box_depth (length width : ℕ) (num_cubes : ℕ) (depth : ℕ) : 
  length = 35 → 
  width = 20 → 
  num_cubes = 56 →
  (∃ (cube_edge : ℕ), 
    cube_edge ∣ length ∧ 
    cube_edge ∣ width ∧ 
    cube_edge ∣ depth ∧
    cube_edge ^ 3 * num_cubes = length * width * depth) →
  depth = 10 := by
  sorry


end box_depth_l118_11802


namespace exam_time_allocation_l118_11892

theorem exam_time_allocation (total_time : ℝ) (total_questions : ℕ) (type_a_count : ℕ) :
  total_time = 180 ∧
  total_questions = 200 ∧
  type_a_count = 10 →
  let type_b_count : ℕ := total_questions - type_a_count
  let time_ratio : ℝ := 2
  let type_b_time : ℝ := total_time / (type_b_count + time_ratio * type_a_count)
  let type_a_time : ℝ := time_ratio * type_b_time
  type_a_count * type_a_time = 120 / 7 :=
by sorry

end exam_time_allocation_l118_11892


namespace intersection_of_P_and_Q_l118_11877

def P : Set ℝ := {x | x^2 - 9 < 0}
def Q : Set ℝ := {y | ∃ x : ℤ, y = 2 * x}

theorem intersection_of_P_and_Q : P ∩ Q = {-2, 0, 2} := by sorry

end intersection_of_P_and_Q_l118_11877


namespace spurs_basketballs_l118_11843

/-- The total number of basketballs for a team -/
def total_basketballs (num_players : ℕ) (balls_per_player : ℕ) : ℕ :=
  num_players * balls_per_player

/-- Theorem: A team of 22 players, each with 11 basketballs, has 242 basketballs in total -/
theorem spurs_basketballs : total_basketballs 22 11 = 242 := by
  sorry

end spurs_basketballs_l118_11843


namespace average_speed_on_time_l118_11837

/-- The average speed needed to reach the destination on time given the conditions -/
theorem average_speed_on_time (total_distance : ℝ) (late_speed : ℝ) (late_time : ℝ) :
  total_distance = 70 →
  late_speed = 35 →
  late_time = 0.25 →
  (total_distance / late_speed) - late_time = total_distance / 40 :=
by sorry

end average_speed_on_time_l118_11837


namespace factorization_equality_l118_11840

theorem factorization_equality (a b : ℝ) : a * b^2 - a = a * (b + 1) * (b - 1) := by
  sorry

end factorization_equality_l118_11840


namespace skylar_age_l118_11819

/-- Represents the age when Skylar started donating -/
def starting_age : ℕ := 17

/-- Represents the annual donation amount in thousands -/
def annual_donation : ℕ := 8

/-- Represents the total amount donated in thousands -/
def total_donated : ℕ := 440

/-- Calculates Skylar's current age -/
def current_age : ℕ := starting_age + (total_donated / annual_donation)

/-- Proves that Skylar's current age is 72 years -/
theorem skylar_age : current_age = 72 := by
  sorry

end skylar_age_l118_11819


namespace quadratic_root_implies_coefficient_l118_11860

theorem quadratic_root_implies_coefficient (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 6 = 0 ∧ x = 2) → a = 5 := by
  sorry

end quadratic_root_implies_coefficient_l118_11860


namespace max_value_on_interval_max_value_at_one_l118_11875

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

-- Define the property of f being monotonically decreasing on (-∞, 2]
def is_monotone_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 2 → f a x ≥ f a y

-- Theorem 1: Maximum value of f(x) on [3, 5] is 2
theorem max_value_on_interval (a : ℝ) 
  (h : is_monotone_decreasing_on_interval a) : 
  (∀ x, x ∈ Set.Icc 3 5 → f a x ≤ 2) ∧ (∃ x, x ∈ Set.Icc 3 5 ∧ f a x = 2) :=
sorry

-- Theorem 2: Maximum value of f(1) is -6
theorem max_value_at_one (a : ℝ) 
  (h : is_monotone_decreasing_on_interval a) : 
  f a 1 ≤ -6 :=
sorry

end max_value_on_interval_max_value_at_one_l118_11875


namespace sum_of_ages_is_41_l118_11852

/-- The sum of Henry and Jill's present ages -/
def sumOfAges (henryAge : ℕ) (jillAge : ℕ) : ℕ :=
  henryAge + jillAge

/-- Theorem stating that the sum of Henry and Jill's present ages is 41 -/
theorem sum_of_ages_is_41 (henryAge : ℕ) (jillAge : ℕ) 
  (h1 : henryAge = 25) 
  (h2 : jillAge = 16) : 
  sumOfAges henryAge jillAge = 41 := by
  sorry

#check sum_of_ages_is_41

end sum_of_ages_is_41_l118_11852


namespace last_equal_sum_date_l118_11896

def is_valid_date (year month day : ℕ) : Prop :=
  year = 2008 ∧ month ≥ 1 ∧ month ≤ 12 ∧ day ≥ 1 ∧ day ≤ 31

def sum_first_four (year month day : ℕ) : ℕ :=
  (day / 10) + (day % 10) + (month / 10) + (month % 10)

def sum_last_four (year : ℕ) : ℕ :=
  (year / 1000) + ((year / 100) % 10) + ((year / 10) % 10) + (year % 10)

def has_equal_sum (year month day : ℕ) : Prop :=
  sum_first_four year month day = sum_last_four year

def is_after (year1 month1 day1 year2 month2 day2 : ℕ) : Prop :=
  year1 > year2 ∨ (year1 = year2 ∧ (month1 > month2 ∨ (month1 = month2 ∧ day1 > day2)))

theorem last_equal_sum_date :
  ∀ (year month day : ℕ),
    is_valid_date year month day →
    has_equal_sum year month day →
    ¬(is_after year month day 2008 12 25) →
    year = 2008 ∧ month = 12 ∧ day = 25 :=
sorry

end last_equal_sum_date_l118_11896


namespace age_ratio_equation_exists_l118_11873

/-- Represents the ages of three people in terms of a common multiplier -/
structure AgeRatio :=
  (x : ℝ)  -- Common multiplier
  (y : ℝ)  -- Number of years ago

/-- The equation relating the ages and the sum from y years ago -/
def ageEquation (r : AgeRatio) : Prop :=
  20 * r.x - 3 * r.y = 76

theorem age_ratio_equation_exists :
  ∃ r : AgeRatio, ageEquation r :=
sorry

end age_ratio_equation_exists_l118_11873


namespace robert_ate_ten_chocolates_l118_11844

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The number of additional chocolates Robert ate compared to Nickel -/
def robert_additional_chocolates : ℕ := 5

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := nickel_chocolates + robert_additional_chocolates

theorem robert_ate_ten_chocolates : robert_chocolates = 10 := by
  sorry

end robert_ate_ten_chocolates_l118_11844


namespace pharmacy_purchase_cost_bob_pharmacy_purchase_cost_l118_11826

/-- Calculates the total cost of a pharmacy purchase including sales tax -/
theorem pharmacy_purchase_cost (nose_spray_cost : ℚ) (nose_spray_count : ℕ) 
  (nose_spray_discount : ℚ) (cough_syrup_cost : ℚ) (cough_syrup_count : ℕ) 
  (cough_syrup_discount : ℚ) (ibuprofen_cost : ℚ) (ibuprofen_count : ℕ) 
  (sales_tax_rate : ℚ) : ℚ :=
  let nose_spray_total := (nose_spray_cost * ↑(nose_spray_count / 2)) * (1 - nose_spray_discount)
  let cough_syrup_total := (cough_syrup_cost * ↑cough_syrup_count) * (1 - cough_syrup_discount)
  let ibuprofen_total := ibuprofen_cost * ↑ibuprofen_count
  let subtotal := nose_spray_total + cough_syrup_total + ibuprofen_total
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax

/-- The total cost of Bob's pharmacy purchase, rounded to the nearest cent, is $56.38 -/
theorem bob_pharmacy_purchase_cost : 
  ⌊pharmacy_purchase_cost 3 10 (1/5) 7 4 (1/10) 5 3 (2/25) * 100⌋ / 100 = 56381 / 1000 := by
  sorry

end pharmacy_purchase_cost_bob_pharmacy_purchase_cost_l118_11826


namespace distance_traveled_is_9_miles_l118_11878

/-- The total distance traveled when biking and jogging for a given time and rate -/
def total_distance (bike_time : ℚ) (bike_rate : ℚ) (jog_time : ℚ) (jog_rate : ℚ) : ℚ :=
  (bike_time * bike_rate) + (jog_time * jog_rate)

/-- Theorem stating that the total distance traveled is 9 miles -/
theorem distance_traveled_is_9_miles :
  let bike_time : ℚ := 1/2  -- 30 minutes in hours
  let bike_rate : ℚ := 6
  let jog_time : ℚ := 3/4   -- 45 minutes in hours
  let jog_rate : ℚ := 8
  total_distance bike_time bike_rate jog_time jog_rate = 9 := by
  sorry

#eval total_distance (1/2) 6 (3/4) 8

end distance_traveled_is_9_miles_l118_11878


namespace exponent_division_l118_11835

theorem exponent_division (x : ℝ) (hx : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end exponent_division_l118_11835


namespace union_of_A_and_B_l118_11888

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {3, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} := by
  sorry

end union_of_A_and_B_l118_11888


namespace pancake_max_pieces_l118_11880

/-- The maximum number of pieces a circle can be divided into with n straight cuts -/
def maxPieces (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- A round pancake can be divided into at most 7 pieces with three straight cuts -/
theorem pancake_max_pieces :
  maxPieces 3 = 7 :=
sorry

end pancake_max_pieces_l118_11880


namespace base6_addition_sum_l118_11882

/-- Represents a single digit in base 6 -/
def Base6Digit := Fin 6

/-- Converts a Base6Digit to its natural number representation -/
def to_nat (d : Base6Digit) : Nat := d.val

/-- Represents the base-6 addition problem 5CD₆ + 32₆ = 61C₆ -/
def base6_addition_problem (C D : Base6Digit) : Prop :=
  (5 * 6 * 6 + to_nat C * 6 + to_nat D) + (3 * 6 + 2) = 
  (6 * 6 + 1 * 6 + to_nat C)

theorem base6_addition_sum (C D : Base6Digit) :
  base6_addition_problem C D → to_nat C + to_nat D = 6 := by
  sorry

end base6_addition_sum_l118_11882


namespace inequality_proof_l118_11881

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a - b / a > b - a / b := by
  sorry

end inequality_proof_l118_11881


namespace price_decrease_percentage_l118_11838

def original_price : ℝ := 77.95
def sale_price : ℝ := 59.95

theorem price_decrease_percentage :
  let difference := original_price - sale_price
  let percentage_decrease := (difference / original_price) * 100
  ∃ ε > 0, abs (percentage_decrease - 23.08) < ε :=
sorry

end price_decrease_percentage_l118_11838


namespace c_7_equals_448_l118_11805

/-- Sequence definition -/
def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

/-- Theorem stating that c_7 equals 448 -/
theorem c_7_equals_448 : c 7 = 448 := by
  sorry

end c_7_equals_448_l118_11805


namespace soda_discount_percentage_l118_11897

/-- Given the regular price per can and the discounted price for 72 cans,
    calculate the discount percentage. -/
theorem soda_discount_percentage
  (regular_price : ℝ)
  (discounted_price : ℝ)
  (h_regular_price : regular_price = 0.60)
  (h_discounted_price : discounted_price = 34.56) :
  (1 - discounted_price / (72 * regular_price)) * 100 = 20 := by
sorry

end soda_discount_percentage_l118_11897


namespace vector_equality_l118_11834

variable {V : Type*} [AddCommGroup V]

/-- For any four points A, B, C, and D in a vector space,
    DA + CD - CB = BA -/
theorem vector_equality (A B C D : V) : D - A + (C - D) - (C - B) = B - A := by
  sorry

end vector_equality_l118_11834


namespace temperature_increase_l118_11894

theorem temperature_increase (morning_temp afternoon_temp : ℤ) : 
  morning_temp = -3 → afternoon_temp = 5 → afternoon_temp - morning_temp = 8 :=
by
  sorry

end temperature_increase_l118_11894


namespace log_ride_cost_l118_11822

/-- The cost of a single log ride, given the following conditions:
  * Dolly wants to ride the Ferris wheel twice
  * Dolly wants to ride the roller coaster three times
  * Dolly wants to ride the log ride seven times
  * The Ferris wheel costs 2 tickets per ride
  * The roller coaster costs 5 tickets per ride
  * Dolly has 20 tickets
  * Dolly needs to buy 6 more tickets
-/
theorem log_ride_cost : ℕ := by
  sorry

#check log_ride_cost

end log_ride_cost_l118_11822


namespace complex_number_quadrant_l118_11895

theorem complex_number_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (i / (1 + i) : ℂ) = a + b * I :=
  sorry

end complex_number_quadrant_l118_11895


namespace equation_solution_l118_11869

theorem equation_solution : ∃! x : ℝ, 0.05 * x + 0.12 * (30 + x) = 15.84 ∧ x = 72 := by
  sorry

end equation_solution_l118_11869


namespace man_brother_age_difference_l118_11807

/-- Represents the age difference between a man and his brother -/
def ageDifference (manAge brotherAge : ℕ) : ℕ := manAge - brotherAge

/-- The problem statement -/
theorem man_brother_age_difference :
  ∀ (manAge brotherAge : ℕ),
    brotherAge = 10 →
    manAge > brotherAge →
    manAge + 2 = 2 * (brotherAge + 2) →
    ageDifference manAge brotherAge = 12 := by
  sorry

end man_brother_age_difference_l118_11807


namespace triangle_abc_properties_l118_11893

/-- Triangle ABC with vertices A(-2, 3), B(5, 3), and C(5, -2) is a right triangle with perimeter 12 + √74 -/
theorem triangle_abc_properties :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (5, 3)
  let C : ℝ × ℝ := (5, -2)
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  -- Triangle ABC is a right triangle with right angle at B
  AB^2 + BC^2 = AC^2 ∧
  -- The perimeter of triangle ABC is 12 + √74
  AB + BC + AC = 12 + Real.sqrt 74 :=
by sorry

end triangle_abc_properties_l118_11893


namespace extrema_of_quadratic_form_l118_11821

theorem extrema_of_quadratic_form (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  1 ≤ x^2 + 2*y^2 + 3*z^2 ∧ x^2 + 2*y^2 + 3*z^2 ≤ 3 := by
  sorry

end extrema_of_quadratic_form_l118_11821


namespace cube_edge_ratio_l118_11846

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^3 / b^3 = 64 → a / b = 4 := by
sorry

end cube_edge_ratio_l118_11846


namespace saras_sister_notebooks_l118_11812

/-- Calculates the final number of notebooks given initial, ordered, and lost quantities. -/
def final_notebooks (initial ordered lost : ℕ) : ℕ :=
  initial + ordered - lost

/-- Theorem stating that Sara's sister's final number of notebooks is 8. -/
theorem saras_sister_notebooks : final_notebooks 4 6 2 = 8 := by
  sorry

end saras_sister_notebooks_l118_11812


namespace cubic_difference_l118_11862

theorem cubic_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : 
  a^3 - b^3 = 108 := by
sorry

end cubic_difference_l118_11862


namespace broken_line_length_lower_bound_l118_11803

/-- A broken line in a square -/
structure BrokenLine where
  -- The square containing the broken line
  square : Set (ℝ × ℝ)
  -- The broken line itself
  line : Set (ℝ × ℝ)
  -- The square has side length 50
  square_side : ∀ (x y : ℝ), (x, y) ∈ square → 0 ≤ x ∧ x ≤ 50 ∧ 0 ≤ y ∧ y ≤ 50
  -- The broken line is contained within the square
  line_in_square : line ⊆ square
  -- For any point in the square, there's a point on the line within distance 1
  close_point : ∀ (p : ℝ × ℝ), p ∈ square → ∃ (q : ℝ × ℝ), q ∈ line ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ 1

/-- The length of a broken line -/
noncomputable def length (bl : BrokenLine) : ℝ := sorry

/-- Theorem: The length of the broken line is greater than 1248 -/
theorem broken_line_length_lower_bound (bl : BrokenLine) : length bl > 1248 := by
  sorry

end broken_line_length_lower_bound_l118_11803


namespace mittens_count_l118_11853

theorem mittens_count (original_plugs current_plugs mittens : ℕ) : 
  mittens = original_plugs - 20 →
  current_plugs = original_plugs + 30 →
  400 = 2 * current_plugs →
  mittens = 150 := by
  sorry

end mittens_count_l118_11853


namespace f_zero_gt_f_four_l118_11855

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- State that f is differentiable on ℝ
variable (hf : Differentiable ℝ f)

-- Define the condition that f(x) = x² + 2f''(2)x - 3
variable (hf_eq : ∀ x, f x = x^2 + 2 * (deriv^[2] f 2) * x - 3)

-- Theorem to prove
theorem f_zero_gt_f_four : f 0 > f 4 := by
  sorry

end f_zero_gt_f_four_l118_11855


namespace train_platform_problem_l118_11887

/-- Given a train and two platforms, calculate the length of the second platform -/
theorem train_platform_problem (train_length platform1_length : ℝ)
  (time1 time2 : ℝ) :
  train_length = 100 →
  platform1_length = 350 →
  time1 = 15 →
  time2 = 20 →
  (train_length + platform1_length) / time1 = (train_length + 500) / time2 :=
by sorry

end train_platform_problem_l118_11887


namespace lilacs_sold_l118_11861

/-- Represents the number of lilacs sold -/
def lilacs : ℕ := sorry

/-- Represents the number of roses sold -/
def roses : ℕ := 3 * lilacs

/-- Represents the number of gardenias sold -/
def gardenias : ℕ := lilacs / 2

/-- The total number of flowers sold -/
def total_flowers : ℕ := 45

/-- Theorem stating that the number of lilacs sold is 10 -/
theorem lilacs_sold : lilacs = 10 := by
  sorry

end lilacs_sold_l118_11861


namespace max_marks_proof_l118_11859

/-- Given a maximum mark M, calculate the passing mark as 60% of M -/
def passing_mark (M : ℝ) : ℝ := 0.6 * M

/-- The maximum marks for an exam -/
def M : ℝ := 300

/-- The marks obtained by the student -/
def obtained_marks : ℝ := 160

/-- The number of marks by which the student failed -/
def failed_by : ℝ := 20

theorem max_marks_proof :
  passing_mark M = obtained_marks + failed_by :=
sorry

end max_marks_proof_l118_11859


namespace friend_initial_savings_l118_11858

/-- Proves that given the conditions of the savings problem, the friend's initial amount is $210 --/
theorem friend_initial_savings (your_initial : ℕ) (your_weekly : ℕ) (friend_weekly : ℕ) (weeks : ℕ) 
  (h1 : your_initial = 160)
  (h2 : your_weekly = 7)
  (h3 : friend_weekly = 5)
  (h4 : weeks = 25)
  (h5 : your_initial + your_weekly * weeks = friend_initial + friend_weekly * weeks) :
  friend_initial = 210 := by
  sorry

#check friend_initial_savings

end friend_initial_savings_l118_11858


namespace q_satisfies_conditions_q_unique_l118_11849

/-- The cubic polynomial q(x) that satisfies the given conditions -/
def q (x : ℝ) : ℝ := -4 * x^3 + 24 * x^2 - 44 * x + 24

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q 1 = 0 ∧ q 2 = 0 ∧ q 3 = 0 ∧ q 4 = -24 := by
  sorry

/-- Theorem stating that q(x) is the unique cubic polynomial satisfying the conditions -/
theorem q_unique (p : ℝ → ℝ) (h_cubic : ∃ a b c d, ∀ x, p x = a * x^3 + b * x^2 + c * x + d) 
  (h_cond : p 1 = 0 ∧ p 2 = 0 ∧ p 3 = 0 ∧ p 4 = -24) :
  ∀ x, p x = q x := by
  sorry

end q_satisfies_conditions_q_unique_l118_11849


namespace xy_value_l118_11800

theorem xy_value (x y : ℝ) 
  (h1 : x + y = 2) 
  (h2 : x^2 * y^3 + y^2 * x^3 = 32) : 
  x * y = -8 := by
sorry

end xy_value_l118_11800


namespace oblique_triangular_prism_surface_area_l118_11883

/-- The total surface area of an oblique triangular prism -/
theorem oblique_triangular_prism_surface_area
  (a l : ℝ)
  (h_a_pos : 0 < a)
  (h_l_pos : 0 < l) :
  let lateral_surface_area := 3 * a * l
  let base_area := a^2 * Real.sqrt 3 / 2
  let total_surface_area := lateral_surface_area + 2 * base_area
  total_surface_area = 3 * a * l + a^2 * Real.sqrt 3 :=
by sorry


end oblique_triangular_prism_surface_area_l118_11883


namespace correct_difference_is_1552_l118_11841

/-- Calculates the correct difference given the erroneous calculation and mistakes made --/
def correct_difference (erroneous_difference : ℕ) 
  (units_mistake : ℕ) (tens_mistake : ℕ) (hundreds_mistake : ℕ) : ℕ :=
  erroneous_difference - hundreds_mistake + tens_mistake - units_mistake

/-- Proves that the correct difference is 1552 given the specific mistakes in the problem --/
theorem correct_difference_is_1552 : 
  correct_difference 1994 2 60 500 = 1552 := by sorry

end correct_difference_is_1552_l118_11841


namespace race_probability_l118_11866

theorem race_probability (total_cars : ℕ) (prob_x prob_y prob_z : ℚ) 
  (h_total : total_cars = 10)
  (h_x : prob_x = 1/7)
  (h_y : prob_y = 1/3)
  (h_z : prob_z = 1/5)
  (h_no_tie : ∀ a b : ℕ, a ≠ b → a ≤ total_cars → b ≤ total_cars → 
    (prob_x + prob_y + prob_z ≤ 1)) :
  prob_x + prob_y + prob_z = 71/105 := by
sorry

end race_probability_l118_11866


namespace chinese_chess_tournament_l118_11872

-- Define the winning relation
def Wins (n : ℕ) : (ℕ → ℕ → Prop) := sorry

-- Main theorem
theorem chinese_chess_tournament (n : ℕ) (h : n ≥ 2) :
  ∃ (P : ℕ → ℕ → ℕ),
    (∀ i j i' j', i ≤ n ∧ j ≤ n ∧ i' ≤ n ∧ j' ≤ n ∧ (i, j) ≠ (i', j') → P i j ≠ P i' j') ∧ 
    (∀ i j, i ≤ n ∧ j ≤ n → P i j ≤ 2*n^2) ∧
    (∀ i j i' j', i < i' ∧ i ≤ n ∧ j ≤ n ∧ i' ≤ n ∧ j' ≤ n → Wins n (P i j) (P i' j')) :=
by
  sorry

-- Transitive property of winning
axiom wins_trans (n : ℕ) : ∀ a b c, Wins n a b → Wins n b c → Wins n a c

-- Maximum number of draws
axiom max_draws (n : ℕ) : ∃ (draw_count : ℕ), draw_count ≤ n^3/16 ∧ 
  ∀ a b, a ≤ 2*n^2 ∧ b ≤ 2*n^2 ∧ a ≠ b → (Wins n a b ∨ Wins n b a ∨ (¬Wins n a b ∧ ¬Wins n b a))

end chinese_chess_tournament_l118_11872


namespace triangle_formation_l118_11850

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given sticks of length 6 and 12, proves which of the given lengths can form a triangle -/
theorem triangle_formation (l : ℝ) : 
  (l = 5 ∨ l = 6 ∨ l = 11 ∨ l = 20) → 
  (can_form_triangle 6 12 l ↔ l = 11) :=
by sorry

end triangle_formation_l118_11850


namespace percentage_increase_l118_11817

theorem percentage_increase (N : ℝ) (P : ℝ) : 
  N = 80 →
  N + (P / 100) * N - (N - (25 / 100) * N) = 30 →
  P = 12.5 := by
  sorry

end percentage_increase_l118_11817


namespace translation_theorem_l118_11857

/-- The original function f(x) = 2x^2 - 2x -/
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x

/-- The transformed function g(x) = 2x^2 - 10x - 9 -/
def g (x : ℝ) : ℝ := 2 * x^2 - 10 * x - 9

/-- Theorem stating that g is the result of translating f 2 units right and 3 units down -/
theorem translation_theorem : ∀ x : ℝ, g x = f (x - 2) - 3 := by sorry

end translation_theorem_l118_11857


namespace equation_solution_l118_11816

theorem equation_solution :
  let x : ℝ := (173 * 240) / 120
  ∃ ε > 0, ε < 0.005 ∧ |x - 345.33| < ε :=
by
  sorry

end equation_solution_l118_11816


namespace intersection_complement_equals_l118_11864

def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {1, 3, 6}
def B : Set ℤ := {1, 4, 5}

theorem intersection_complement_equals : A ∩ (U \ B) = {3, 6} := by sorry

end intersection_complement_equals_l118_11864


namespace nested_average_equality_l118_11808

def avg_pair (a b : ℚ) : ℚ := (a + b) / 2

def avg_quad (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem nested_average_equality : 
  avg_quad 
    (avg_quad (avg_pair 2 4) (avg_pair 1 3) (avg_pair 0 2) (avg_pair 1 1))
    (avg_pair 3 3)
    (avg_pair 2 2)
    (avg_pair 4 0) = 35 / 16 := by
  sorry

end nested_average_equality_l118_11808


namespace townspeople_win_probability_l118_11865

/-- The probability that the townspeople win in a game with 2 townspeople and 1 goon -/
theorem townspeople_win_probability :
  let total_participants : ℕ := 2 + 1
  let num_goons : ℕ := 1
  let townspeople_win_condition := (num_goons / total_participants : ℚ)
  townspeople_win_condition = 1 / 3 := by
  sorry

end townspeople_win_probability_l118_11865


namespace racers_meeting_time_l118_11839

/-- The time in seconds for the Racing Magic to complete one lap -/
def racing_magic_lap_time : ℕ := 60

/-- The number of laps the Charging Bull completes in one hour -/
def charging_bull_laps_per_hour : ℕ := 40

/-- The time in seconds for the Charging Bull to complete one lap -/
def charging_bull_lap_time : ℕ := 3600 / charging_bull_laps_per_hour

/-- The least common multiple of the two lap times -/
def lcm_lap_times : ℕ := Nat.lcm racing_magic_lap_time charging_bull_lap_time

/-- The time in minutes for the racers to meet at the starting point for the second time -/
def meeting_time_minutes : ℕ := lcm_lap_times / 60

theorem racers_meeting_time :
  meeting_time_minutes = 3 := by sorry

end racers_meeting_time_l118_11839


namespace initial_mean_calculation_l118_11891

theorem initial_mean_calculation (n : ℕ) (incorrect_value correct_value : ℝ) (correct_mean : ℝ) :
  n = 30 ∧ 
  incorrect_value = 135 ∧ 
  correct_value = 165 ∧ 
  correct_mean = 151 →
  ∃ (initial_mean : ℝ),
    n * initial_mean + (correct_value - incorrect_value) = n * correct_mean ∧
    initial_mean = 150 :=
by sorry

end initial_mean_calculation_l118_11891


namespace perfect_square_addition_l118_11829

theorem perfect_square_addition : ∃ x : ℤ,
  (∃ a : ℤ, 100 + x = a^2) ∧
  (∃ b : ℤ, 164 + x = b^2) ∧
  x = 125 := by
  sorry

end perfect_square_addition_l118_11829


namespace equality_sum_l118_11889

theorem equality_sum (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + N = 183 := by
  sorry

end equality_sum_l118_11889


namespace cody_age_l118_11867

theorem cody_age (grandmother_age : ℕ) (age_ratio : ℕ) (cody_age : ℕ) : 
  grandmother_age = 84 →
  grandmother_age = age_ratio * cody_age →
  age_ratio = 6 →
  cody_age = 14 := by
sorry

end cody_age_l118_11867


namespace sum_of_arithmetic_sequence_l118_11815

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- State the theorem
theorem sum_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a5 : a 5 = 3) 
  (h_a6 : a 6 = -2) : 
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3) :=
sorry

end sum_of_arithmetic_sequence_l118_11815


namespace translate_line_upward_5_units_l118_11848

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a linear function vertically by a given amount -/
def translateVertically (f : LinearFunction) (amount : ℝ) : LinearFunction :=
  { slope := f.slope, yIntercept := f.yIntercept + amount }

theorem translate_line_upward_5_units :
  let original : LinearFunction := { slope := 2, yIntercept := -4 }
  let translated := translateVertically original 5
  translated = { slope := 2, yIntercept := 1 } := by
  sorry

end translate_line_upward_5_units_l118_11848
