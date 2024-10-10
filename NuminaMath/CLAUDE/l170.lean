import Mathlib

namespace common_external_tangent_y_intercept_value_l170_17018

/-- The y-intercept of the common external tangent to two circles --/
def common_external_tangent_y_intercept : ℝ := sorry

/-- First circle center --/
def center1 : ℝ × ℝ := (1, 3)

/-- Second circle center --/
def center2 : ℝ × ℝ := (13, 6)

/-- First circle radius --/
def radius1 : ℝ := 3

/-- Second circle radius --/
def radius2 : ℝ := 6

theorem common_external_tangent_y_intercept_value :
  ∃ (m : ℝ), m > 0 ∧ 
  ∀ (x y : ℝ), y = m * x + common_external_tangent_y_intercept →
  ((x - center1.1)^2 + (y - center1.2)^2 = radius1^2 ∨
   (x - center2.1)^2 + (y - center2.2)^2 = radius2^2) →
  ∀ (x' y' : ℝ), (x' - center1.1)^2 + (y' - center1.2)^2 < radius1^2 →
                 (x' - center2.1)^2 + (y' - center2.2)^2 < radius2^2 →
                 y' ≠ m * x' + common_external_tangent_y_intercept := by
  sorry

#check common_external_tangent_y_intercept_value

end common_external_tangent_y_intercept_value_l170_17018


namespace exists_x_for_all_m_greater_than_one_l170_17059

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| + |x - 2|

-- State the theorem
theorem exists_x_for_all_m_greater_than_one :
  ∀ m : ℝ, m > 1 → ∃ x : ℝ, f x = 4 / (m - 1) + m :=
by sorry

end exists_x_for_all_m_greater_than_one_l170_17059


namespace complex_number_range_l170_17034

theorem complex_number_range (z₁ z₂ : ℂ) (a : ℝ) : 
  z₁ = ((-1 + 3*I) * (1 - I) - (1 + 3*I)) / I →
  z₂ = z₁ + a * I →
  Complex.abs z₂ ≤ 2 →
  a ∈ Set.Icc (1 - Real.sqrt 3) (1 + Real.sqrt 3) :=
by sorry

end complex_number_range_l170_17034


namespace quadratic_positivity_condition_l170_17053

theorem quadratic_positivity_condition (m : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 0 ∧ 
  ∃ m₀ : ℝ, m₀ > 0 ∧ ¬(∀ x : ℝ, x^2 + 2*x + m₀ > 0) :=
by sorry

end quadratic_positivity_condition_l170_17053


namespace power_of_seven_mod_ten_l170_17070

theorem power_of_seven_mod_ten : 7^150 % 10 = 9 := by
  sorry

end power_of_seven_mod_ten_l170_17070


namespace rogers_coin_donation_l170_17049

theorem rogers_coin_donation (pennies nickels dimes coins_left : ℕ) :
  pennies = 42 →
  nickels = 36 →
  dimes = 15 →
  coins_left = 27 →
  pennies + nickels + dimes - coins_left = 66 := by
  sorry

end rogers_coin_donation_l170_17049


namespace lattice_points_on_segment_l170_17013

/-- The number of lattice points on a line segment with given endpoints -/
def latticePointCount (x1 y1 x2 y2 : Int) : Nat :=
  sorry

/-- Theorem stating that the number of lattice points on the given line segment is 6 -/
theorem lattice_points_on_segment : latticePointCount 5 26 40 146 = 6 := by
  sorry

end lattice_points_on_segment_l170_17013


namespace median_in_70_79_interval_l170_17014

/-- Represents a score interval with its lower bound and number of students -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (count : ℕ)

/-- The distribution of scores for 100 students -/
def score_distribution : List ScoreInterval :=
  [⟨90, 22⟩, ⟨80, 18⟩, ⟨70, 20⟩, ⟨60, 15⟩, ⟨50, 25⟩]

/-- The total number of students -/
def total_students : ℕ := 100

/-- The position of the median in the sorted list of scores -/
def median_position : ℕ := total_students / 2

/-- Finds the interval containing the median score -/
def find_median_interval (distribution : List ScoreInterval) (total : ℕ) (median_pos : ℕ) : ScoreInterval :=
  sorry

/-- Theorem stating that the interval 70-79 contains the median score -/
theorem median_in_70_79_interval :
  find_median_interval score_distribution total_students median_position = ⟨70, 20⟩ :=
sorry

end median_in_70_79_interval_l170_17014


namespace bottles_drank_l170_17037

def initial_bottles : ℕ := 17
def remaining_bottles : ℕ := 14

theorem bottles_drank : initial_bottles - remaining_bottles = 3 := by
  sorry

end bottles_drank_l170_17037


namespace specific_pairs_probability_l170_17083

/-- The probability of two specific pairs forming in a random pairing of students -/
theorem specific_pairs_probability (n : ℕ) (h : n = 32) : 
  (1 : ℚ) / (n - 1) * (1 : ℚ) / (n - 2) = 1 / 930 :=
by sorry

end specific_pairs_probability_l170_17083


namespace cookie_count_l170_17069

theorem cookie_count (paul_cookies : ℕ) (paula_difference : ℕ) : 
  paul_cookies = 45 →
  paula_difference = 3 →
  paul_cookies + (paul_cookies - paula_difference) = 87 :=
by
  sorry

end cookie_count_l170_17069


namespace floor_ceiling_sum_l170_17026

theorem floor_ceiling_sum : ⌊(-3.75 : ℝ)⌋ + ⌈(34.25 : ℝ)⌉ = 31 := by
  sorry

end floor_ceiling_sum_l170_17026


namespace triangle_side_simplification_l170_17085

theorem triangle_side_simplification (k : ℝ) (h1 : 3 < k) (h2 : k < 5) :
  |2*k - 5| - Real.sqrt (k^2 - 12*k + 36) = 3*k - 11 := by
  sorry

end triangle_side_simplification_l170_17085


namespace regular_polygon_interior_angle_l170_17086

theorem regular_polygon_interior_angle (n : ℕ) (n_ge_3 : n ≥ 3) :
  (((n - 2) * 180) / n = 150) → n = 12 := by
  sorry

end regular_polygon_interior_angle_l170_17086


namespace equation_solution_l170_17006

theorem equation_solution (a : ℚ) : 
  (∀ x : ℚ, (2*a*x + 3) / (a - x) = 3/4 ↔ x = 1) → a = -3 := by
sorry

end equation_solution_l170_17006


namespace valid_last_score_l170_17031

def scores : List Nat := [65, 72, 75, 79, 82, 86, 90, 98]

def isIntegerAverage (sublist : List Nat) : Prop :=
  (sublist.sum * 100) % sublist.length = 0

def isValidLastScore (last : Nat) : Prop :=
  ∀ n : Nat, n ≥ 1 → n < 8 → 
    isIntegerAverage (scores.take n ++ [last])

theorem valid_last_score : 
  isValidLastScore 79 := by sorry

end valid_last_score_l170_17031


namespace symmetric_point_xoy_plane_l170_17044

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoy plane in 3D space -/
def xoy_plane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xoy plane -/
def symmetric_wrt_xoy (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = q.y ∧ p.z = -q.z

theorem symmetric_point_xoy_plane :
  let M : Point3D := ⟨2, 5, 8⟩
  let N : Point3D := ⟨2, 5, -8⟩
  symmetric_wrt_xoy M N := by sorry

end symmetric_point_xoy_plane_l170_17044


namespace sum_division_l170_17028

/-- The problem of dividing a sum among four people with specific ratios -/
theorem sum_division (w x y z : ℝ) (total : ℝ) : 
  w > 0 ∧ 
  x = 0.8 * w ∧ 
  y = 0.65 * w ∧ 
  z = 0.45 * w ∧
  y = 78 →
  total = w + x + y + z ∧ total = 348 := by
  sorry

end sum_division_l170_17028


namespace quadratic_point_relation_l170_17008

/-- A quadratic function y = x^2 - 4x + n, where n is a constant -/
def quadratic (n : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + n

theorem quadratic_point_relation (n : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  quadratic n x₁ = y₁ →
  quadratic n x₂ = y₂ →
  y₁ > y₂ →
  |x₁ - 2| > |x₂ - 2| :=
sorry

end quadratic_point_relation_l170_17008


namespace product_of_sums_l170_17012

theorem product_of_sums (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 35)
  (h_sum_prod : a * b + b * c + c * a = 320)
  (h_prod : a * b * c = 600) :
  (a + b) * (b + c) * (c + a) = 10600 := by
sorry

end product_of_sums_l170_17012


namespace prob_one_head_in_three_flips_l170_17081

/-- The probability of getting exactly one head in three flips of a fair coin -/
theorem prob_one_head_in_three_flips :
  let n : ℕ := 3  -- number of flips
  let k : ℕ := 1  -- number of desired heads
  let p : ℚ := 1/2  -- probability of getting heads on a single flip
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/8 := by
  sorry

end prob_one_head_in_three_flips_l170_17081


namespace binomial_square_coefficient_l170_17058

theorem binomial_square_coefficient (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 18 * x + 16 = (r * x + s)^2) → a = 81/16 := by
  sorry

end binomial_square_coefficient_l170_17058


namespace division_problem_l170_17094

theorem division_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 144 ∧ divisor = 11 ∧ remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 13 := by
sorry

end division_problem_l170_17094


namespace find_y_l170_17078

theorem find_y (x y : ℝ) (h1 : x^(2*y) = 16) (h2 : x = 2) : y = 2 := by
  sorry

end find_y_l170_17078


namespace power_of_two_with_nines_l170_17033

theorem power_of_two_with_nines (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, ∃ m : ℕ,
    (2^n % 10^k = m) ∧ 
    (∃ count : ℕ, count ≥ k/2 ∧ 
      (∀ i : ℕ, i < k → 
        ((m / 10^i) % 10 = 9 → count > 0) ∧
        ((m / 10^i) % 10 ≠ 9 → count = count))) :=
sorry

end power_of_two_with_nines_l170_17033


namespace g_of_3_eq_6_l170_17092

/-- The function g(x) = x^3 - 3x^2 + 2x -/
def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- Theorem: g(3) = 6 -/
theorem g_of_3_eq_6 : g 3 = 6 := by sorry

end g_of_3_eq_6_l170_17092


namespace train_length_l170_17041

/-- The length of a train given its crossing times over a platform and a signal pole -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 200)
  (h2 : platform_time = 30)
  (h3 : pole_time = 18) :
  ∃ (train_length : ℝ), 
    train_length + platform_length = (train_length / pole_time) * platform_time ∧ 
    train_length = 300 := by
  sorry

end train_length_l170_17041


namespace sum_divisible_by_three_l170_17084

theorem sum_divisible_by_three (a : ℤ) : ∃ k : ℤ, a^3 + 2*a = 3*k := by
  sorry

end sum_divisible_by_three_l170_17084


namespace prove_correct_statements_l170_17038

-- Define the types of relationships
inductive Relationship
| Functional
| Correlation

-- Define the properties of relationships
def isDeterministic (r : Relationship) : Prop :=
  match r with
  | Relationship.Functional => True
  | Relationship.Correlation => False

-- Define regression analysis
def regressionAnalysis (r : Relationship) : Prop :=
  match r with
  | Relationship.Functional => False
  | Relationship.Correlation => True

-- Define the set of correct statements
def correctStatements : Set Nat :=
  {1, 2, 4}

-- Theorem to prove
theorem prove_correct_statements :
  (isDeterministic Relationship.Functional) ∧
  (¬isDeterministic Relationship.Correlation) ∧
  (regressionAnalysis Relationship.Correlation) →
  correctStatements = {1, 2, 4} := by
  sorry

end prove_correct_statements_l170_17038


namespace wrong_height_calculation_l170_17032

/-- Proves that the wrongly written height of a boy is 176 cm given the following conditions:
  * There are 35 boys in a class
  * The initially calculated average height was 182 cm
  * One boy's height was recorded incorrectly
  * The boy's actual height is 106 cm
  * The correct average height is 180 cm
-/
theorem wrong_height_calculation (n : ℕ) (initial_avg correct_avg actual_height : ℝ) :
  n = 35 →
  initial_avg = 182 →
  correct_avg = 180 →
  actual_height = 106 →
  (n : ℝ) * initial_avg - (n : ℝ) * correct_avg + actual_height = 176 := by
  sorry

end wrong_height_calculation_l170_17032


namespace total_elephants_l170_17072

theorem total_elephants (we_preserve : ℕ) (gestures : ℕ) (natures_last : ℕ) : 
  we_preserve = 70 →
  gestures = 3 * we_preserve →
  natures_last = 5 * gestures →
  we_preserve + gestures + natures_last = 1330 := by
  sorry

#check total_elephants

end total_elephants_l170_17072


namespace interior_edges_sum_is_seven_l170_17039

/-- A rectangular picture frame with specific properties -/
structure PictureFrame where
  /-- Width of the wood pieces used in the frame -/
  woodWidth : ℝ
  /-- Length of one outer edge of the frame -/
  outerEdgeLength : ℝ
  /-- Exposed area of the frame (excluding the picture) -/
  exposedArea : ℝ

/-- Calculates the sum of the lengths of the four interior edges of the frame -/
def interiorEdgesSum (frame : PictureFrame) : ℝ :=
  sorry

/-- Theorem stating that for a frame with given properties, the sum of interior edges is 7 inches -/
theorem interior_edges_sum_is_seven 
  (frame : PictureFrame)
  (h1 : frame.woodWidth = 2)
  (h2 : frame.outerEdgeLength = 6)
  (h3 : frame.exposedArea = 30) :
  interiorEdgesSum frame = 7 :=
sorry

end interior_edges_sum_is_seven_l170_17039


namespace distance_to_axis_of_symmetry_l170_17099

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ parabola_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define the axis of symmetry
def axis_of_symmetry : ℝ := -3

-- Theorem statement
theorem distance_to_axis_of_symmetry (A B : ℝ × ℝ) :
  intersection_points A B →
  let midpoint_x := (A.1 + B.1) / 2
  |midpoint_x - axis_of_symmetry| = 11 := by
  sorry

end distance_to_axis_of_symmetry_l170_17099


namespace logarithm_proportionality_l170_17000

theorem logarithm_proportionality (P K a b : ℝ) 
  (hP : P > 0) (hK : K > 0) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) : 
  (Real.log P / Real.log a) / (Real.log P / Real.log b) = 
  (Real.log K / Real.log a) / (Real.log K / Real.log b) := by
  sorry

end logarithm_proportionality_l170_17000


namespace highest_throw_is_37_l170_17035

def highest_throw (christine_first : ℕ) (janice_first_diff : ℕ) 
                  (christine_second_diff : ℕ) (christine_third_diff : ℕ) 
                  (janice_third_diff : ℕ) : ℕ :=
  let christine_first := christine_first
  let janice_first := christine_first - janice_first_diff
  let christine_second := christine_first + christine_second_diff
  let janice_second := janice_first * 2
  let christine_third := christine_second + christine_third_diff
  let janice_third := christine_first + janice_third_diff
  max christine_first (max christine_second (max christine_third 
    (max janice_first (max janice_second janice_third))))

theorem highest_throw_is_37 : 
  highest_throw 20 4 10 4 17 = 37 := by
  sorry

end highest_throw_is_37_l170_17035


namespace greatest_prime_factor_of_169_l170_17011

theorem greatest_prime_factor_of_169 : ∃ p : ℕ, p.Prime ∧ p ∣ 169 ∧ ∀ q : ℕ, q.Prime → q ∣ 169 → q ≤ p :=
  sorry

end greatest_prime_factor_of_169_l170_17011


namespace kelly_egg_income_l170_17046

/-- Calculates the money made from selling eggs over a given period. -/
def money_from_eggs (num_chickens : ℕ) (eggs_per_day : ℕ) (price_per_dozen : ℕ) (num_weeks : ℕ) : ℕ :=
  let eggs_per_week := num_chickens * eggs_per_day * 7
  let total_eggs := eggs_per_week * num_weeks
  let dozens := total_eggs / 12
  dozens * price_per_dozen

/-- Proves that Kelly makes $280 in 4 weeks from selling eggs. -/
theorem kelly_egg_income : money_from_eggs 8 3 5 4 = 280 := by
  sorry

end kelly_egg_income_l170_17046


namespace probability_in_tournament_of_26_l170_17074

/-- The probability of two specific participants playing against each other in a tournament. -/
def probability_of_match (n : ℕ) : ℚ :=
  (n - 1) / (n * (n - 1) / 2)

/-- Theorem: In a tournament with 26 participants, the probability of two specific participants
    playing against each other is 1/13. -/
theorem probability_in_tournament_of_26 :
  probability_of_match 26 = 1 / 13 := by
  sorry

#eval probability_of_match 26  -- To check the result

end probability_in_tournament_of_26_l170_17074


namespace cupcakes_left_after_distribution_l170_17064

/-- Theorem: Cupcakes Left After Distribution

Given:
- Dani brings two and half dozen cupcakes
- There are 27 students (including Dani)
- There is 1 teacher
- There is 1 teacher's aid
- 3 students called in sick

Prove that the number of cupcakes left after Dani gives one to everyone in the class is 4.
-/
theorem cupcakes_left_after_distribution 
  (cupcakes_per_dozen : ℕ)
  (total_students : ℕ)
  (teacher_count : ℕ)
  (teacher_aid_count : ℕ)
  (sick_students : ℕ)
  (h1 : cupcakes_per_dozen = 12)
  (h2 : total_students = 27)
  (h3 : teacher_count = 1)
  (h4 : teacher_aid_count = 1)
  (h5 : sick_students = 3) :
  2 * cupcakes_per_dozen + cupcakes_per_dozen / 2 - 
  (total_students - sick_students + teacher_count + teacher_aid_count) = 4 := by
  sorry

end cupcakes_left_after_distribution_l170_17064


namespace soda_cost_theorem_l170_17042

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℝ := 2

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℝ := 3

/-- The cost of Bob's fruit drink in dollars -/
def fruit_drink_cost : ℝ := 2

/-- The cost of Andy's soda in dollars -/
def soda_cost : ℝ := 4

/-- Andy's total spending in dollars -/
def andy_spending : ℝ := 2 * hamburger_cost + soda_cost

/-- Bob's total spending in dollars -/
def bob_spending : ℝ := 2 * sandwich_cost + fruit_drink_cost

theorem soda_cost_theorem : 
  andy_spending = bob_spending → soda_cost = 4 := by sorry

end soda_cost_theorem_l170_17042


namespace no_real_solutions_exponential_equation_l170_17005

theorem no_real_solutions_exponential_equation :
  ∀ x : ℝ, (2 : ℝ)^(5*x+2) * (4 : ℝ)^(2*x+4) ≠ (8 : ℝ)^(3*x+7) := by
sorry

end no_real_solutions_exponential_equation_l170_17005


namespace geometric_sequence_seventh_term_l170_17067

/-- Given a geometric sequence {a_n} where a_1 + a_2 = 3 and a_2 + a_3 = 6, prove that a_7 = 64 -/
theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 3)
  (h2 : a 2 + a 3 = 6)
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = a 2 / a 1) :
  a 7 = 64 := by
sorry

end geometric_sequence_seventh_term_l170_17067


namespace f_monotone_increasing_a_value_for_odd_function_l170_17027

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem f_monotone_increasing (a : ℝ) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
sorry

theorem a_value_for_odd_function :
  (∀ x : ℝ, f a x = -f a (-x)) → a = 1/2 :=
sorry

end f_monotone_increasing_a_value_for_odd_function_l170_17027


namespace infinite_solutions_cube_equation_l170_17073

theorem infinite_solutions_cube_equation :
  ∀ n : ℕ, ∃ x y z : ℤ, 
    x^2 + y^2 + z^2 = x^3 + y^3 + z^3 ∧
    (∀ m : ℕ, m < n → 
      ∃ x' y' z' : ℤ, 
        x'^2 + y'^2 + z'^2 = x'^3 + y'^3 + z'^3 ∧
        (x', y', z') ≠ (x, y, z)) :=
by
  sorry

end infinite_solutions_cube_equation_l170_17073


namespace remainder_sum_mod_five_l170_17063

theorem remainder_sum_mod_five (f y : ℤ) 
  (hf : f % 5 = 3) 
  (hy : y % 5 = 4) : 
  (f + y) % 5 = 2 :=
by sorry

end remainder_sum_mod_five_l170_17063


namespace two_volunteers_same_project_l170_17052

/-- The number of volunteers -/
def num_volunteers : ℕ := 3

/-- The number of projects -/
def num_projects : ℕ := 7

/-- The probability that exactly two volunteers are assigned to the same project -/
def probability_two_same_project : ℚ := 18/49

theorem two_volunteers_same_project :
  (num_volunteers = 3) →
  (num_projects = 7) →
  (∀ volunteer, volunteer ≤ num_volunteers → ∃! project, project ≤ num_projects) →
  probability_two_same_project = 18/49 := by
  sorry

end two_volunteers_same_project_l170_17052


namespace triangle_area_rational_l170_17048

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a triangle with three IntPoints
structure IntTriangle where
  p1 : IntPoint
  p2 : IntPoint
  p3 : IntPoint

-- Function to calculate the area of a triangle given its vertices
def triangleArea (t : IntTriangle) : ℚ :=
  let x1 := t.p1.x
  let y1 := t.p1.y
  let x2 := t.p2.x
  let y2 := t.p2.y
  let x3 := t.p3.x
  let y3 := t.p3.y
  (1 / 2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Theorem stating that the area of the triangle is rational
theorem triangle_area_rational (t : IntTriangle) 
  (h1 : t.p1.x - t.p1.y = 1)
  (h2 : t.p2.x - t.p2.y = 1)
  (h3 : t.p3.x - t.p3.y = 1) :
  ∃ (q : ℚ), triangleArea t = q :=
by
  sorry


end triangle_area_rational_l170_17048


namespace tangent_line_parabola_l170_17065

/-- The equation of the tangent line to the parabola y = x^2 at the point (-1, 1) -/
theorem tangent_line_parabola :
  let f (x : ℝ) := x^2
  let p : ℝ × ℝ := (-1, 1)
  let tangent_line (x y : ℝ) := 2*x + y + 1 = 0
  (∀ x, (f x, x) ∈ Set.range (λ t => (t, f t))) →
  (p.1, p.2) ∈ Set.range (λ t => (t, f t)) →
  ∃ m b, (∀ x, tangent_line x (m*x + b)) ∧
         (tangent_line p.1 p.2) ∧
         (∀ ε > 0, ∃ δ > 0, ∀ x, |x - p.1| < δ → |f x - (m*x + b)| < ε * |x - p.1|) := by
  sorry

end tangent_line_parabola_l170_17065


namespace painted_cube_theorem_l170_17025

theorem painted_cube_theorem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 ↔ n = 3 :=
by sorry

end painted_cube_theorem_l170_17025


namespace complex_multiplication_l170_17088

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : (1 - i)^2 * i = 2 := by
  sorry

end complex_multiplication_l170_17088


namespace base_conversion_problem_l170_17054

theorem base_conversion_problem :
  ∀ (a b : ℕ),
    a < 10 →
    b < 10 →
    235 = 1 * 7^2 + a * 7^1 + b * 7^0 →
    (a + b : ℚ) / 7 = 6 / 7 := by
  sorry

end base_conversion_problem_l170_17054


namespace meaningful_range_l170_17075

def is_meaningful (x : ℝ) : Prop :=
  3 - x ≥ 0 ∧ x - 1 > 0 ∧ x ≠ 2

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ (1 < x ∧ x ≤ 3 ∧ x ≠ 2) :=
by sorry

end meaningful_range_l170_17075


namespace squarefree_term_existence_l170_17051

/-- A positive integer is squarefree if it's not divisible by any square number greater than 1 -/
def IsSquarefree (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 1 → d * d ∣ n → d = 1

/-- An arithmetic sequence of positive integers -/
def IsArithmeticSeq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem squarefree_term_existence :
  ∃ C : ℝ, C > 0 ∧
    ∀ a : ℕ → ℕ, IsArithmeticSeq a →
      IsSquarefree (Nat.gcd (a 1) (a 2)) →
        ∃ m : ℕ, m > 0 ∧ m ≤ ⌊C * (a 2)^2⌋ ∧ IsSquarefree (a m) :=
sorry

end squarefree_term_existence_l170_17051


namespace inverse_proportion_l170_17061

/-- Given that p and q are inversely proportional, prove that if p = 20 when q = 8, then p = 16 when q = 10. -/
theorem inverse_proportion (p q : ℝ) (h : p * q = 20 * 8) : 
  p * 10 = 16 * 10 := by
  sorry

end inverse_proportion_l170_17061


namespace coin_drop_probability_l170_17090

theorem coin_drop_probability : 
  let square_side : ℝ := 10
  let black_square_side : ℝ := 1
  let coin_diameter : ℝ := 2
  let coin_radius : ℝ := coin_diameter / 2
  let drop_area_side : ℝ := square_side - coin_diameter
  let drop_area : ℝ := drop_area_side ^ 2
  let extended_black_square_side : ℝ := black_square_side + coin_diameter
  let extended_black_area : ℝ := 4 * (extended_black_square_side ^ 2)
  extended_black_area / drop_area = 9 / 16 := by sorry

end coin_drop_probability_l170_17090


namespace roxy_initial_flowering_plants_l170_17091

/-- The initial number of flowering plants in Roxy's garden -/
def initial_flowering_plants : ℕ := 7

/-- The initial number of fruiting plants in Roxy's garden -/
def initial_fruiting_plants : ℕ := 2 * initial_flowering_plants

/-- The number of flowering plants bought on Saturday -/
def flowering_plants_bought : ℕ := 3

/-- The number of fruiting plants bought on Saturday -/
def fruiting_plants_bought : ℕ := 2

/-- The number of flowering plants given away on Sunday -/
def flowering_plants_given : ℕ := 1

/-- The number of fruiting plants given away on Sunday -/
def fruiting_plants_given : ℕ := 4

/-- The total number of plants remaining after all transactions -/
def total_plants_remaining : ℕ := 21

theorem roxy_initial_flowering_plants :
  (initial_flowering_plants + flowering_plants_bought - flowering_plants_given) +
  (initial_fruiting_plants + fruiting_plants_bought - fruiting_plants_given) =
  total_plants_remaining :=
by sorry

end roxy_initial_flowering_plants_l170_17091


namespace sprinkler_system_water_usage_l170_17040

theorem sprinkler_system_water_usage 
  (morning_usage : ℝ) 
  (evening_usage : ℝ) 
  (total_water : ℝ) 
  (h1 : morning_usage = 4)
  (h2 : evening_usage = 6)
  (h3 : total_water = 50) :
  (total_water / (morning_usage + evening_usage) = 5) :=
by sorry

end sprinkler_system_water_usage_l170_17040


namespace baseball_groups_l170_17024

/-- The number of groups formed from baseball players -/
def number_of_groups (new_players returning_players players_per_group : ℕ) : ℕ :=
  (new_players + returning_players) / players_per_group

/-- Theorem: The number of groups formed is 9 -/
theorem baseball_groups :
  number_of_groups 48 6 6 = 9 := by
  sorry

end baseball_groups_l170_17024


namespace least_product_of_two_primes_above_50_l170_17017

theorem least_product_of_two_primes_above_50 (p q : ℕ) : 
  p.Prime → q.Prime → p > 50 → q > 50 → p ≠ q → 
  ∃ (min_product : ℕ), min_product = 3127 ∧ 
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 50 → s > 50 → r ≠ s → 
      p * q ≤ r * s := by
  sorry

end least_product_of_two_primes_above_50_l170_17017


namespace total_wattage_calculation_l170_17095

def light_A_initial : ℝ := 60
def light_B_initial : ℝ := 40
def light_C_initial : ℝ := 50

def light_A_increase : ℝ := 0.12
def light_B_increase : ℝ := 0.20
def light_C_increase : ℝ := 0.15

def total_new_wattage : ℝ :=
  light_A_initial * (1 + light_A_increase) +
  light_B_initial * (1 + light_B_increase) +
  light_C_initial * (1 + light_C_increase)

theorem total_wattage_calculation :
  total_new_wattage = 172.7 := by sorry

end total_wattage_calculation_l170_17095


namespace vectors_are_parallel_l170_17080

def a : ℝ × ℝ × ℝ := (1, 2, -2)
def b : ℝ × ℝ × ℝ := (-2, -4, 4)

theorem vectors_are_parallel : ∃ k : ℝ, b = k • a := by
  sorry

end vectors_are_parallel_l170_17080


namespace rotation_composition_implies_triangle_angles_l170_17021

/-- Represents a rotation in 2D space -/
structure Rotation2D where
  angle : ℝ
  center : ℝ × ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Composition of rotations -/
def compose_rotations (r1 r2 r3 : Rotation2D) : Rotation2D :=
  sorry

/-- Check if a rotation is the identity transformation -/
def is_identity (r : Rotation2D) : Prop :=
  sorry

/-- Get the angle at a vertex of a triangle -/
def angle_at_vertex (t : Triangle) (v : ℝ × ℝ) : ℝ :=
  sorry

theorem rotation_composition_implies_triangle_angles 
  (α β γ : ℝ) (t : Triangle) (r_A r_B r_C : Rotation2D) :
  0 < α ∧ α < π →
  0 < β ∧ β < π →
  0 < γ ∧ γ < π →
  α + β + γ = π →
  r_A.angle = 2 * α →
  r_B.angle = 2 * β →
  r_C.angle = 2 * γ →
  r_A.center = t.A →
  r_B.center = t.B →
  r_C.center = t.C →
  is_identity (compose_rotations r_C r_B r_A) →
  angle_at_vertex t t.A = α ∧
  angle_at_vertex t t.B = β ∧
  angle_at_vertex t t.C = γ :=
by sorry

end rotation_composition_implies_triangle_angles_l170_17021


namespace alien_trees_conversion_l170_17022

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (hundreds tens units : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + units * 7^0

/-- The problem statement --/
theorem alien_trees_conversion :
  base7ToBase10 2 5 3 = 136 := by
  sorry

end alien_trees_conversion_l170_17022


namespace cos_seven_expansion_sum_of_squares_l170_17089

theorem cos_seven_expansion_sum_of_squares : 
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ), 
    (∀ θ : ℝ, Real.cos θ ^ 7 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + 
      b₃ * Real.cos (3 * θ) + b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + 
      b₆ * Real.cos (6 * θ) + b₇ * Real.cos (7 * θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 429 / 1024 := by
  sorry

end cos_seven_expansion_sum_of_squares_l170_17089


namespace base7_perfect_square_last_digit_l170_17066

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Represents a number in base 7 as ab2c -/
structure Base7Rep where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a ≠ 0

/-- Converts a Base7Rep to its decimal equivalent -/
def toDecimal (rep : Base7Rep) : ℕ :=
  rep.a * 7^3 + rep.b * 7^2 + 2 * 7 + rep.c

theorem base7_perfect_square_last_digit (n : ℕ) (rep : Base7Rep) :
  isPerfectSquare n ∧ n = toDecimal rep → rep.c = 2 ∨ rep.c = 3 ∨ rep.c = 6 := by
  sorry

end base7_perfect_square_last_digit_l170_17066


namespace simplify_trig_expression_l170_17079

theorem simplify_trig_expression :
  Real.sqrt (2 - Real.sin 1 ^ 2 + Real.cos 2) = Real.sqrt 3 * Real.cos 1 := by
  sorry

end simplify_trig_expression_l170_17079


namespace geometric_series_common_ratio_l170_17002

/-- The first term of the geometric series -/
def a₁ : ℚ := 7/8

/-- The second term of the geometric series -/
def a₂ : ℚ := -21/32

/-- The third term of the geometric series -/
def a₃ : ℚ := 63/128

/-- The common ratio of the geometric series -/
def r : ℚ := -3/4

theorem geometric_series_common_ratio :
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by
  sorry

end geometric_series_common_ratio_l170_17002


namespace abc_remainder_l170_17016

theorem abc_remainder (a b c : ℕ) : 
  a < 9 → b < 9 → c < 9 →
  (a + 3*b + 2*c) % 9 = 3 →
  (2*a + 2*b + 3*c) % 9 = 6 →
  (3*a + b + 2*c) % 9 = 1 →
  (a*b*c) % 9 = 4 := by
sorry

end abc_remainder_l170_17016


namespace total_path_is_2125_feet_l170_17001

/-- Represents the scale of the plan in feet per inch -/
def scale : ℝ := 500

/-- Represents the initial path length on the plan in inches -/
def initial_path : ℝ := 3

/-- Represents the path extension on the plan in inches -/
def path_extension : ℝ := 1.25

/-- Calculates the total path length in feet -/
def total_path_length : ℝ := (initial_path + path_extension) * scale

/-- Theorem stating that the total path length is 2125 feet -/
theorem total_path_is_2125_feet : total_path_length = 2125 := by sorry

end total_path_is_2125_feet_l170_17001


namespace simplify_expression_l170_17098

theorem simplify_expression (b : ℝ) :
  3 * b * (3 * b^2 - 2 * b + 1) + 2 * b^2 = 9 * b^3 - 4 * b^2 + 3 * b := by
  sorry

end simplify_expression_l170_17098


namespace inequality_solution_l170_17045

-- Define the given inequality and its solution set
def given_inequality (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3*x + 2 > 0
def solution_set (b : ℝ) (x : ℝ) : Prop := x < 1 ∨ x > b

-- Define the values to be proven
def a_value : ℝ := 1
def b_value : ℝ := 2

-- Define the new inequality
def new_inequality (m : ℝ) (x : ℝ) : Prop := m * x^2 - (2*m + 1)*x + 2 < 0

-- Define the solution sets for different m values
def solution_set_m_zero (x : ℝ) : Prop := x > 2
def solution_set_m_gt_half (m : ℝ) (x : ℝ) : Prop := 1/m < x ∧ x < 2
def solution_set_m_half : Set ℝ := ∅
def solution_set_m_between_zero_half (m : ℝ) (x : ℝ) : Prop := 2 < x ∧ x < 1/m
def solution_set_m_neg (m : ℝ) (x : ℝ) : Prop := x < 1/m ∨ x > 2

-- State the theorem
theorem inequality_solution :
  (∀ x, given_inequality a_value x ↔ solution_set b_value x) ∧
  (∀ m x, new_inequality m x ↔
    (m = 0 ∧ solution_set_m_zero x) ∨
    (m > 1/2 ∧ solution_set_m_gt_half m x) ∨
    (m = 1/2 ∧ x ∈ solution_set_m_half) ∨
    (0 < m ∧ m < 1/2 ∧ solution_set_m_between_zero_half m x) ∨
    (m < 0 ∧ solution_set_m_neg m x)) :=
sorry

end inequality_solution_l170_17045


namespace cars_meeting_halfway_l170_17097

/-- Two cars meeting halfway between two points --/
theorem cars_meeting_halfway 
  (total_distance : ℝ) 
  (speed_car1 : ℝ) 
  (start_time_car1 start_time_car2 : ℕ) 
  (speed_car2 : ℝ) :
  total_distance = 600 →
  speed_car1 = 50 →
  start_time_car1 = 7 →
  start_time_car2 = 8 →
  (total_distance / 2) / speed_car1 + start_time_car1 = 
    (total_distance / 2) / speed_car2 + start_time_car2 →
  speed_car2 = 60 := by
sorry

end cars_meeting_halfway_l170_17097


namespace arithmetic_sequence_sum_maximum_l170_17060

theorem arithmetic_sequence_sum_maximum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence
  (a 11 / a 10 < -1) →  -- given condition
  (∃ k, ∀ n, S n ≤ S k) →  -- sum has a maximum value
  (∀ n > 19, S n ≤ 0) ∧ (S 19 > 0) :=
by sorry


end arithmetic_sequence_sum_maximum_l170_17060


namespace intersection_distance_theorem_l170_17062

/-- A linear function f(x) = ax + b -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- The distance between intersection points of two functions -/
def intersectionDistance (f g : ℝ → ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem intersection_distance_theorem (f : LinearFunction) :
  intersectionDistance (fun x => x^2 - 1) (fun x => f.a * x + f.b + 1) = 3 * Real.sqrt 10 →
  intersectionDistance (fun x => x^2) (fun x => f.a * x + f.b + 3) = 3 * Real.sqrt 14 →
  intersectionDistance (fun x => x^2) (fun x => f.a * x + f.b) = 3 * Real.sqrt 2 := by
  sorry

end intersection_distance_theorem_l170_17062


namespace f_decreasing_on_positive_reals_l170_17010

/-- The function f(x) = -x^2 + 3 is decreasing on the interval (0, +∞) -/
theorem f_decreasing_on_positive_reals :
  ∀ (x₁ x₂ : ℝ), 0 < x₁ → x₁ < x₂ → (-x₁^2 + 3) > (-x₂^2 + 3) := by
  sorry

end f_decreasing_on_positive_reals_l170_17010


namespace grade_assignment_count_l170_17082

/-- The number of possible grades a professor can assign to each student. -/
def num_grades : ℕ := 4

/-- The number of students in the class. -/
def num_students : ℕ := 12

/-- The number of ways to assign grades to all students. -/
def num_ways : ℕ := num_grades ^ num_students

/-- Theorem stating that the number of ways to assign grades is 16,777,216. -/
theorem grade_assignment_count : num_ways = 16777216 := by
  sorry

end grade_assignment_count_l170_17082


namespace julia_running_time_difference_l170_17023

/-- Julia's running times with different shoes -/
theorem julia_running_time_difference (x : ℝ) : 
  let old_pace : ℝ := 10  -- minutes per mile in old shoes
  let new_pace : ℝ := 13  -- minutes per mile in new shoes
  let miles_for_known_difference : ℝ := 5
  let known_time_difference : ℝ := 15  -- minutes difference for 5 miles
  -- Prove that the time difference for x miles is 3x minutes
  (new_pace - old_pace) * x = 3 * x ∧
  -- Also prove that this is consistent with the given information for 5 miles
  (new_pace - old_pace) * miles_for_known_difference = known_time_difference
  := by sorry

end julia_running_time_difference_l170_17023


namespace complex_equation_solution_l170_17019

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 + a * Complex.I) / (1 - Complex.I) = -2 - Complex.I →
  a = -3 := by sorry

end complex_equation_solution_l170_17019


namespace decimal_127_to_octal_has_three_consecutive_digits_l170_17015

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

def is_consecutive_digits (digits : List ℕ) : Bool :=
  match digits with
  | [] => true
  | [_] => true
  | x :: y :: rest => (y = x + 1) && is_consecutive_digits (y :: rest)

theorem decimal_127_to_octal_has_three_consecutive_digits :
  let octal_digits := decimal_to_octal 127
  octal_digits.length = 3 ∧ is_consecutive_digits octal_digits = true :=
sorry

end decimal_127_to_octal_has_three_consecutive_digits_l170_17015


namespace a_investment_value_l170_17003

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  total_profit : ℝ
  c_profit_share : ℝ

/-- Theorem stating that given the conditions of the problem, a's investment is 30000. -/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 45000)
  (hc : p.c_investment = 50000)
  (htotal : p.total_profit = 90000)
  (hc_share : p.c_profit_share = 36000) :
  p.a_investment = 30000 := by
  sorry

#check a_investment_value

end a_investment_value_l170_17003


namespace largest_number_of_three_l170_17093

theorem largest_number_of_three (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p * q + p * r + q * r = -8)
  (prod_eq : p * q * r = -20) :
  max p (max q r) = (-1 + Real.sqrt 21) / 2 := by
  sorry

end largest_number_of_three_l170_17093


namespace roots_product_l170_17030

def Q (d e f : ℝ) (x : ℝ) : ℝ := x^3 + d*x^2 + e*x + f

theorem roots_product (d e f : ℝ) :
  (∀ x, Q d e f x = 0 ↔ x = Real.cos (2*π/9) ∨ x = Real.cos (4*π/9) ∨ x = Real.cos (8*π/9)) →
  d * e * f = 1 / 27 :=
sorry

end roots_product_l170_17030


namespace min_value_h_negative_reals_l170_17009

-- Define odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the theorem
theorem min_value_h_negative_reals 
  (f g : ℝ → ℝ) 
  (hf : IsOdd f) 
  (hg : IsOdd g) 
  (h : ℝ → ℝ) 
  (h_def : ∀ x, h x = f x + g x - 2) 
  (h_max : ∃ M, M = 6 ∧ ∀ x > 0, h x ≤ M) :
  ∃ m, m = -10 ∧ ∀ x < 0, h x ≥ m := by
sorry

end min_value_h_negative_reals_l170_17009


namespace greatest_integer_satisfying_inequality_l170_17068

theorem greatest_integer_satisfying_inequality :
  ∃ (n : ℤ), n^2 - 11*n + 24 ≤ 0 ∧
  n = 8 ∧
  ∀ (m : ℤ), m^2 - 11*m + 24 ≤ 0 → m ≤ n :=
by sorry

end greatest_integer_satisfying_inequality_l170_17068


namespace tea_bags_in_box_l170_17057

theorem tea_bags_in_box (cups_per_bag_min cups_per_bag_max : ℕ) 
                        (natasha_cups inna_cups : ℕ) : 
  cups_per_bag_min = 2 →
  cups_per_bag_max = 3 →
  natasha_cups = 41 →
  inna_cups = 58 →
  ∃ n : ℕ, 
    n * cups_per_bag_min ≤ natasha_cups ∧ 
    natasha_cups ≤ n * cups_per_bag_max ∧
    n * cups_per_bag_min ≤ inna_cups ∧ 
    inna_cups ≤ n * cups_per_bag_max ∧
    n = 20 := by
  sorry

end tea_bags_in_box_l170_17057


namespace total_difference_of_sequences_l170_17076

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem total_difference_of_sequences : 
  let n : ℕ := 72
  let d : ℕ := 3
  let a₁ : ℕ := 2001
  let b₁ : ℕ := 501
  arithmetic_sequence_sum a₁ d n - arithmetic_sequence_sum b₁ d n = 108000 := by
sorry

end total_difference_of_sequences_l170_17076


namespace quadratic_sufficient_not_necessary_l170_17096

theorem quadratic_sufficient_not_necessary :
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) ∧
  ¬(∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
by sorry

end quadratic_sufficient_not_necessary_l170_17096


namespace abs_sin_integral_over_2pi_l170_17055

theorem abs_sin_integral_over_2pi (f : ℝ → ℝ) : 
  (∫ x in (0)..(2 * Real.pi), |Real.sin x|) = 4 := by
  sorry

end abs_sin_integral_over_2pi_l170_17055


namespace audrey_balls_l170_17047

theorem audrey_balls (jake_balls : ℕ) (difference : ℕ) : 
  jake_balls = 7 → difference = 34 → jake_balls + difference = 41 :=
by
  sorry

end audrey_balls_l170_17047


namespace max_visitable_halls_is_91_l170_17029

/-- Represents a triangular castle divided into smaller triangular halls. -/
structure TriangularCastle where
  total_halls : ℕ
  side_length : ℝ
  hall_side_length : ℝ

/-- Represents a path through the castle halls. -/
def VisitPath (castle : TriangularCastle) := List ℕ

/-- Checks if a path is valid (no repeated visits). -/
def is_valid_path (castle : TriangularCastle) (path : VisitPath castle) : Prop :=
  path.length ≤ castle.total_halls ∧ path.Nodup

/-- The maximum number of halls that can be visited. -/
def max_visitable_halls (castle : TriangularCastle) : ℕ :=
  91

/-- Theorem stating that the maximum number of visitable halls is 91. -/
theorem max_visitable_halls_is_91 (castle : TriangularCastle) 
  (h1 : castle.total_halls = 100)
  (h2 : castle.side_length = 100)
  (h3 : castle.hall_side_length = 10) :
  ∀ (path : VisitPath castle), is_valid_path castle path → path.length ≤ max_visitable_halls castle :=
by sorry

end max_visitable_halls_is_91_l170_17029


namespace transform_standard_deviation_l170_17043

def standardDeviation (sample : Fin 10 → ℝ) : ℝ := sorry

theorem transform_standard_deviation 
  (x : Fin 10 → ℝ) 
  (h : standardDeviation x = 8) : 
  standardDeviation (fun i => 2 * x i - 1) = 16 := by sorry

end transform_standard_deviation_l170_17043


namespace perfect_cube_in_range_l170_17077

theorem perfect_cube_in_range : 
  ∃! (K : ℤ), 
    K > 1 ∧ 
    ∃ (Z : ℤ), 3000 < Z ∧ Z < 4000 ∧ Z = K^4 ∧ 
    ∃ (n : ℤ), Z = n^3 ∧
    K = 7 := by
  sorry

end perfect_cube_in_range_l170_17077


namespace department_store_problem_l170_17007

/-- The cost price per item in yuan -/
def cost_price : ℝ := 120

/-- The relationship between selling price and daily sales volume -/
def price_volume_relation (x y : ℝ) : Prop := x + y = 200

/-- The daily profit function -/
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * (200 - x)

theorem department_store_problem :
  (∃ a : ℝ, price_volume_relation 180 a ∧ a = 20) ∧
  (∃ x : ℝ, daily_profit x = 1600 ∧ x = 160) ∧
  (∀ m n : ℝ, m ≠ n → daily_profit (200 - m) = daily_profit (200 - n) → m + n = 80) :=
sorry

end department_store_problem_l170_17007


namespace systematic_sampling_probability_l170_17071

/-- Represents a systematic sampling process -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  removed_size : ℕ
  h_pop_size : population_size = 1002
  h_sample_size : sample_size = 50
  h_removed_size : removed_size = 2

/-- The probability of an individual being selected in the systematic sampling process -/
def selection_probability (s : SystematicSampling) : ℚ :=
  s.sample_size / s.population_size

theorem systematic_sampling_probability (s : SystematicSampling) :
  selection_probability s = 50 / 1002 := by
  sorry

#eval (50 : ℚ) / 1002

end systematic_sampling_probability_l170_17071


namespace stratified_sampling_middle_batch_l170_17087

/-- Represents the number of units drawn from each batch in a stratified sampling -/
structure BatchSampling where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given BatchSampling forms an arithmetic sequence -/
def is_arithmetic_sequence (s : BatchSampling) : Prop :=
  s.c - s.b = s.b - s.a

/-- The theorem stating that in a stratified sampling of 60 units from three batches
    forming an arithmetic sequence, the number of units drawn from the middle batch is 20 -/
theorem stratified_sampling_middle_batch :
  ∀ s : BatchSampling,
    is_arithmetic_sequence s →
    s.a + s.b + s.c = 60 →
    s.b = 20 := by
  sorry

end stratified_sampling_middle_batch_l170_17087


namespace largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l170_17020

theorem largest_integer_for_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 28 < 0 → n ≤ 6 :=
by
  sorry

theorem six_satisfies_inequality :
  (6 : ℤ)^2 - 11*6 + 28 < 0 :=
by
  sorry

theorem seven_does_not_satisfy_inequality :
  (7 : ℤ)^2 - 11*7 + 28 ≥ 0 :=
by
  sorry

end largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l170_17020


namespace min_value_of_sum_of_powers_l170_17004

theorem min_value_of_sum_of_powers (a b : ℝ) (h : a + b = 2) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x y : ℝ), x + y = 2 → 3^x + 3^y ≥ m := by
  sorry

end min_value_of_sum_of_powers_l170_17004


namespace simplify_expression_l170_17050

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 64) - Real.sqrt (9 + 1/4))^2 = 69/4 - 2 * Real.sqrt 74 := by
  sorry

end simplify_expression_l170_17050


namespace georges_trivia_score_l170_17036

/-- George's trivia game score calculation -/
theorem georges_trivia_score :
  ∀ (first_half_correct second_half_correct points_per_question : ℕ),
    first_half_correct = 6 →
    second_half_correct = 4 →
    points_per_question = 3 →
    (first_half_correct + second_half_correct) * points_per_question = 30 :=
by
  sorry

end georges_trivia_score_l170_17036


namespace matrix_power_equality_l170_17056

def A (a : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![1, 3, a; 0, 1, 5; 0, 0, 1]

theorem matrix_power_equality (a : ℝ) (n : ℕ) :
  (A a) ^ n = !![1, 27, 3000; 0, 1, 45; 0, 0, 1] →
  a + n = 278 := by
  sorry

end matrix_power_equality_l170_17056
