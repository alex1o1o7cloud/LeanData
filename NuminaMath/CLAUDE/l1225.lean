import Mathlib

namespace NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l1225_122521

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 9*x^2 + 18*x + 38

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! p : ℝ × ℝ, 
    (p.1 = g p.2 ∧ p.2 = g p.1) ∧ 
    p = (-2, -2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l1225_122521


namespace NUMINAMATH_CALUDE_sara_joe_height_difference_l1225_122517

/-- Given the heights of Sara, Joe, and Roy, prove that Sara is 6 inches taller than Joe. -/
theorem sara_joe_height_difference :
  ∀ (sara_height joe_height roy_height : ℕ),
    sara_height = 45 →
    joe_height = roy_height + 3 →
    roy_height = 36 →
    sara_height - joe_height = 6 := by
sorry

end NUMINAMATH_CALUDE_sara_joe_height_difference_l1225_122517


namespace NUMINAMATH_CALUDE_type_B_first_is_better_l1225_122574

/-- Represents the score distribution for a two-question quiz -/
structure ScoreDistribution where
  p0 : ℝ  -- Probability of scoring 0
  p1 : ℝ  -- Probability of scoring the first question's points
  p2 : ℝ  -- Probability of scoring both questions' points
  sum_to_one : p0 + p1 + p2 = 1

/-- Calculates the expected score given a score distribution and point values -/
def expectedScore (d : ScoreDistribution) (points1 points2 : ℝ) : ℝ :=
  d.p1 * points1 + d.p2 * (points1 + points2)

/-- Represents the quiz setup -/
structure QuizSetup where
  probA : ℝ  -- Probability of correctly answering type A
  probB : ℝ  -- Probability of correctly answering type B
  pointsA : ℝ  -- Points for correct answer in type A
  pointsB : ℝ  -- Points for correct answer in type B
  probA_bounds : 0 ≤ probA ∧ probA ≤ 1
  probB_bounds : 0 ≤ probB ∧ probB ≤ 1
  positive_points : pointsA > 0 ∧ pointsB > 0

/-- Theorem: Starting with type B questions yields a higher expected score -/
theorem type_B_first_is_better (q : QuizSetup) : 
  let distA : ScoreDistribution := {
    p0 := 1 - q.probA,
    p1 := q.probA * (1 - q.probB),
    p2 := q.probA * q.probB,
    sum_to_one := by sorry
  }
  let distB : ScoreDistribution := {
    p0 := 1 - q.probB,
    p1 := q.probB * (1 - q.probA),
    p2 := q.probB * q.probA,
    sum_to_one := by sorry
  }
  expectedScore distB q.pointsB q.pointsA > expectedScore distA q.pointsA q.pointsB :=
by sorry

end NUMINAMATH_CALUDE_type_B_first_is_better_l1225_122574


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1225_122537

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem seventh_term_of_geometric_sequence 
  (a r : ℝ) 
  (h_positive : ∀ n, geometric_sequence a r n > 0)
  (h_fifth : geometric_sequence a r 5 = 16)
  (h_ninth : geometric_sequence a r 9 = 2) :
  geometric_sequence a r 7 = 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1225_122537


namespace NUMINAMATH_CALUDE_problem_solution_l1225_122506

theorem problem_solution : 
  (Real.sqrt 27 + Real.sqrt 2 * Real.sqrt 6 + Real.sqrt 20 - 5 * Real.sqrt (1/5) = 5 * Real.sqrt 3 + Real.sqrt 5) ∧
  ((Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) + (Real.sqrt 3 - 2) = Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1225_122506


namespace NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_l1225_122560

/-- Represents a rectangular grid with specific properties -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)
  (qr_length : ℕ)
  (st_length : ℕ)
  (rstu_height : ℕ)

/-- Calculates the area of a right triangle given its base and height -/
def triangle_area (base height : ℕ) : ℚ :=
  (base * height : ℚ) / 2

/-- Calculates the area of a rectangle given its width and height -/
def rectangle_area (width height : ℕ) : ℕ :=
  width * height

/-- Calculates the shaded area of the grid -/
def shaded_area (g : Grid) : ℚ :=
  triangle_area g.qr_length g.height +
  triangle_area g.st_length (g.height - g.rstu_height) +
  rectangle_area (g.st_length) g.rstu_height

/-- Calculates the total area of the grid -/
def total_area (g : Grid) : ℕ :=
  rectangle_area g.width g.height

/-- Calculates the unshaded area of the grid -/
def unshaded_area (g : Grid) : ℚ :=
  (total_area g : ℚ) - shaded_area g

/-- Theorem stating the ratio of shaded to unshaded area -/
theorem shaded_to_unshaded_ratio (g : Grid) (h1 : g.width = 9) (h2 : g.height = 4)
    (h3 : g.qr_length = 3) (h4 : g.st_length = 4) (h5 : g.rstu_height = 2) :
    (shaded_area g) / (unshaded_area g) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_l1225_122560


namespace NUMINAMATH_CALUDE_inequality_proof_l1225_122593

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 1) : a * Real.exp b < b * Real.exp a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1225_122593


namespace NUMINAMATH_CALUDE_childs_running_speed_l1225_122520

/-- Proves that the child's running speed on a still sidewalk is 74 m/min given the problem conditions -/
theorem childs_running_speed 
  (speed_still : ℝ) 
  (sidewalk_speed : ℝ) 
  (distance_against : ℝ) 
  (time_against : ℝ) 
  (h1 : speed_still = 74) 
  (h2 : distance_against = 165) 
  (h3 : time_against = 3) 
  (h4 : (speed_still - sidewalk_speed) * time_against = distance_against) : 
  speed_still = 74 := by
sorry

end NUMINAMATH_CALUDE_childs_running_speed_l1225_122520


namespace NUMINAMATH_CALUDE_percentage_relation_l1225_122554

/-- Given three real numbers A, B, and C, where A is 8% of C and 50% of B,
    prove that B is 16% of C. -/
theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.08 * C) 
  (h2 : A = 0.5 * B) : 
  B = 0.16 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l1225_122554


namespace NUMINAMATH_CALUDE_sine_cosine_zero_points_l1225_122541

theorem sine_cosine_zero_points (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)
  (∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, 0 < x ∧ x < 4 * Real.pi ∧ f x = 0) →
  7 / 6 < ω ∧ ω ≤ 17 / 12 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_zero_points_l1225_122541


namespace NUMINAMATH_CALUDE_geometry_book_pages_multiple_l1225_122509

/-- Given that:
    - The old edition of a Geometry book has 340 pages
    - The new edition has 450 pages
    - The new edition has 230 pages less than m times the old edition's pages
    Prove that m = 2 -/
theorem geometry_book_pages_multiple (old_pages new_pages less_pages : ℕ) 
    (h1 : old_pages = 340)
    (h2 : new_pages = 450)
    (h3 : less_pages = 230) :
    ∃ m : ℚ, old_pages * m - less_pages = new_pages ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometry_book_pages_multiple_l1225_122509


namespace NUMINAMATH_CALUDE_test_questions_count_l1225_122528

theorem test_questions_count : ∀ (total_questions : ℕ),
  (total_questions % 4 = 0) →
  (20 : ℚ) / total_questions > (60 : ℚ) / 100 →
  (20 : ℚ) / total_questions < (70 : ℚ) / 100 →
  total_questions = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l1225_122528


namespace NUMINAMATH_CALUDE_fourth_largest_divisor_l1225_122596

def n : ℕ := 1234560000

-- Define a function to get the list of divisors
def divisors (m : ℕ) : List ℕ := sorry

-- Define a function to get the nth largest element from a list
def nthLargest (l : List ℕ) (k : ℕ) : ℕ := sorry

theorem fourth_largest_divisor :
  nthLargest (divisors n) 4 = 154320000 := by sorry

end NUMINAMATH_CALUDE_fourth_largest_divisor_l1225_122596


namespace NUMINAMATH_CALUDE_motorcycles_parked_count_l1225_122585

/-- The number of motorcycles parked between cars on a road -/
def motorcycles_parked (foreign_cars : ℕ) (domestic_cars_between : ℕ) : ℕ :=
  let total_cars := foreign_cars + (foreign_cars - 1) * domestic_cars_between
  total_cars - 1

/-- Theorem stating that given 5 foreign cars and 2 domestic cars between each pair,
    the number of motorcycles parked between all adjacent cars is 12 -/
theorem motorcycles_parked_count :
  motorcycles_parked 5 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_motorcycles_parked_count_l1225_122585


namespace NUMINAMATH_CALUDE_two_consecutive_late_charges_l1225_122568

theorem two_consecutive_late_charges (original_bill : ℝ) (late_charge_rate : ℝ) : 
  original_bill = 500 →
  late_charge_rate = 0.02 →
  (original_bill * (1 + late_charge_rate) * (1 + late_charge_rate)) = 520.20 := by
  sorry

end NUMINAMATH_CALUDE_two_consecutive_late_charges_l1225_122568


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_simplify_sqrt_expression_l1225_122569

-- Part 1
theorem simplify_sqrt_fraction :
  (Real.sqrt 5 + 1) / (Real.sqrt 5 - 1) = (3 + Real.sqrt 5) / 2 := by sorry

-- Part 2
theorem simplify_sqrt_expression :
  Real.sqrt 12 * Real.sqrt 2 / Real.sqrt ((-3)^2) = 2 * Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_simplify_sqrt_expression_l1225_122569


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l1225_122544

def Q (x : ℝ) (Q0 Q1 Q2 : ℝ) : ℝ := Q0 + Q1 * x + Q2 * x^2

theorem polynomial_uniqueness (Q0 Q1 Q2 : ℝ) :
  Q (-1) Q0 Q1 Q2 = -3 →
  Q 3 Q0 Q1 Q2 = 5 →
  ∀ x, Q x Q0 Q1 Q2 = 3 * x^2 + 7 * x - 5 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l1225_122544


namespace NUMINAMATH_CALUDE_subtraction_of_negative_two_minus_negative_four_equals_six_l1225_122582

theorem subtraction_of_negative (a b : ℤ) : a - (-b) = a + b := by sorry

theorem two_minus_negative_four_equals_six : 2 - (-4) = 6 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_two_minus_negative_four_equals_six_l1225_122582


namespace NUMINAMATH_CALUDE_a_range_l1225_122551

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a^2 / (4 * x)

noncomputable def g (x : ℝ) : ℝ := x - log x

theorem a_range (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ (x₁ x₂ : ℝ), 1 ≤ x₁ ∧ x₁ ≤ Real.exp 1 ∧ 1 ≤ x₂ ∧ x₂ ≤ Real.exp 1 → f a x₁ ≥ g x₂) : 
  a ≥ 2 * sqrt (Real.exp 1 - 2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l1225_122551


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1225_122511

/-- Represents a sampling method -/
inductive SamplingMethod
  | Simple
  | Stratified
  | Systematic

/-- Represents a population with subgroups -/
structure Population where
  subgroups : List (Set α)
  significant_differences : Bool

/-- Determines the most appropriate sampling method for a given population -/
def most_appropriate_sampling_method (pop : Population) : SamplingMethod :=
  if pop.significant_differences then
    SamplingMethod.Stratified
  else
    SamplingMethod.Simple

/-- Theorem stating that stratified sampling is most appropriate for populations with significant differences between subgroups -/
theorem stratified_sampling_most_appropriate
  (pop : Population)
  (h : pop.significant_differences = true) :
  most_appropriate_sampling_method pop = SamplingMethod.Stratified :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1225_122511


namespace NUMINAMATH_CALUDE_function_value_at_two_l1225_122533

/-- Given a function f: ℝ → ℝ such that f(x) = ax^5 + bx^3 + cx + 8 for some real constants a, b, and c, 
    and f(-2) = 10, prove that f(2) = 6. -/
theorem function_value_at_two (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 + b * x^3 + c * x + 8)
    (h2 : f (-2) = 10) : 
  f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l1225_122533


namespace NUMINAMATH_CALUDE_alexander_pencil_difference_alexander_pencil_difference_proof_l1225_122584

/-- Proves that Alexander has 60 more pencils than Asaf given the problem conditions -/
theorem alexander_pencil_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun asaf_age alexander_age asaf_pencils alexander_pencils =>
    asaf_age = 50 ∧
    asaf_age + alexander_age = 140 ∧
    alexander_age - asaf_age = asaf_pencils / 2 ∧
    asaf_pencils + alexander_pencils = 220 →
    alexander_pencils - asaf_pencils = 60

/-- Proof of the theorem -/
theorem alexander_pencil_difference_proof :
  ∃ (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ),
    alexander_pencil_difference asaf_age alexander_age asaf_pencils alexander_pencils :=
by
  sorry

end NUMINAMATH_CALUDE_alexander_pencil_difference_alexander_pencil_difference_proof_l1225_122584


namespace NUMINAMATH_CALUDE_last_two_digits_of_product_squared_l1225_122508

theorem last_two_digits_of_product_squared : 
  (301 * 402 * 503 * 604 * 646 * 547 * 448 * 349)^2 % 100 = 76 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_product_squared_l1225_122508


namespace NUMINAMATH_CALUDE_f_properties_l1225_122549

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sin x * Real.cos x

theorem f_properties :
  (f (π / 8) = Real.sqrt 2 + 1) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧
  (∀ x, f x ≥ 1 - Real.sqrt 2) ∧
  (∃ x, f x = 1 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l1225_122549


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l1225_122572

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 132 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l1225_122572


namespace NUMINAMATH_CALUDE_election_results_l1225_122510

theorem election_results (total_votes : ℕ) (invalid_percentage : ℚ) 
  (candidate_A_percentage : ℚ) (candidate_B_percentage : ℚ) :
  total_votes = 1250000 →
  invalid_percentage = 1/5 →
  candidate_A_percentage = 9/20 →
  candidate_B_percentage = 7/20 →
  ∃ (valid_votes : ℕ) (votes_A votes_B votes_C : ℕ),
    valid_votes = total_votes * (1 - invalid_percentage) ∧
    votes_A = valid_votes * candidate_A_percentage ∧
    votes_B = valid_votes * candidate_B_percentage ∧
    votes_C = valid_votes * (1 - candidate_A_percentage - candidate_B_percentage) ∧
    votes_A = 450000 ∧
    votes_B = 350000 ∧
    votes_C = 200000 :=
by sorry

end NUMINAMATH_CALUDE_election_results_l1225_122510


namespace NUMINAMATH_CALUDE_polynomial_roots_l1225_122518

theorem polynomial_roots : ∃ (p : ℝ → ℝ), 
  (∀ x, p x = 8*x^4 + 14*x^3 - 66*x^2 + 40*x) ∧ 
  (p 0 = 0) ∧ (p (1/2) = 0) ∧ (p 2 = 0) ∧ (p (-5) = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1225_122518


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1225_122524

theorem perpendicular_lines_a_values 
  (a : ℝ) 
  (h_perp : (a * (2*a - 1)) + (-1 * a) = 0) : 
  a = 1 ∨ a = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1225_122524


namespace NUMINAMATH_CALUDE_disjoint_sets_range_l1225_122557

def set_A : Set (ℝ × ℝ) := {p | p.2 = -|p.1| - 2}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + p.2^2 = a^2}

theorem disjoint_sets_range (a : ℝ) :
  set_A ∩ set_B a = ∅ ↔ -2 * Real.sqrt 2 - 2 < a ∧ a < 2 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_disjoint_sets_range_l1225_122557


namespace NUMINAMATH_CALUDE_unique_multiplication_problem_l1225_122513

theorem unique_multiplication_problem :
  ∃! (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    100 ≤ a * b ∧ a * b < 1000 ∧
    (a * b) % 100 / 10 = 1 ∧
    (a * b) % 10 = 2 ∧
    (b * (a % 10)) % 100 = 0 ∧
    (a % 10 + b % 10) = 6 ∧
    a * b = 612 :=
by sorry

end NUMINAMATH_CALUDE_unique_multiplication_problem_l1225_122513


namespace NUMINAMATH_CALUDE_cookie_bags_count_l1225_122512

/-- Given a total number of cookies and the fact that each bag contains an equal number of cookies,
    prove that the number of bags is 14. -/
theorem cookie_bags_count (total_cookies : ℕ) (cookies_per_bag : ℕ) (total_candies : ℕ) :
  total_cookies = 28 →
  cookies_per_bag > 0 →
  total_cookies = 14 * cookies_per_bag →
  (∃ (num_bags : ℕ), num_bags = 14 ∧ num_bags * cookies_per_bag = total_cookies) :=
by sorry

end NUMINAMATH_CALUDE_cookie_bags_count_l1225_122512


namespace NUMINAMATH_CALUDE_largest_number_l1225_122536

theorem largest_number : 
  let numbers : List ℝ := [0.978, 0.9719, 0.9781, 0.917, 0.9189]
  ∀ x ∈ numbers, x ≤ 0.9781 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1225_122536


namespace NUMINAMATH_CALUDE_fraction_simplification_l1225_122576

theorem fraction_simplification (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a / b = (100 * a + a) / (100 * b + b)) 
  (h4 : a / b = (10000 * a + 100 * a + a) / (10000 * b + 100 * b + b)) :
  ∀ (d : ℕ), d > 1 → d ∣ a → d ∣ b → False :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1225_122576


namespace NUMINAMATH_CALUDE_sum_of_ages_l1225_122558

theorem sum_of_ages (henry_age jill_age : ℕ) : 
  henry_age = 23 →
  jill_age = 17 →
  henry_age - 11 = 2 * (jill_age - 11) →
  henry_age + jill_age = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1225_122558


namespace NUMINAMATH_CALUDE_six_balls_two_boxes_at_least_two_l1225_122580

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes -/
def totalArrangements (n : ℕ) : ℕ := 2^n

/-- The number of ways to choose k balls from n distinguishable balls -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes
    where one box must contain at least m balls -/
def validArrangements (n m : ℕ) : ℕ :=
  totalArrangements n - (choose n 0 + choose n 1)

theorem six_balls_two_boxes_at_least_two :
  validArrangements 6 2 = 57 := by sorry

end NUMINAMATH_CALUDE_six_balls_two_boxes_at_least_two_l1225_122580


namespace NUMINAMATH_CALUDE_f_10_equals_756_l1225_122532

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

theorem f_10_equals_756 : f 10 = 756 := by
  sorry

end NUMINAMATH_CALUDE_f_10_equals_756_l1225_122532


namespace NUMINAMATH_CALUDE_equal_money_after_11_weeks_l1225_122571

/-- Carol's initial amount in dollars -/
def carol_initial : ℕ := 40

/-- Carol's weekly savings in dollars -/
def carol_savings : ℕ := 12

/-- Mike's initial amount in dollars -/
def mike_initial : ℕ := 150

/-- Mike's weekly savings in dollars -/
def mike_savings : ℕ := 2

/-- The number of weeks it takes for Carol and Mike to have the same amount of money -/
def weeks_to_equal_money : ℕ := 11

theorem equal_money_after_11_weeks :
  carol_initial + carol_savings * weeks_to_equal_money =
  mike_initial + mike_savings * weeks_to_equal_money :=
by sorry

end NUMINAMATH_CALUDE_equal_money_after_11_weeks_l1225_122571


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l1225_122504

/-- Solution set A for x^2 - 3x + 2 > 0 -/
def A : Set ℝ := {x | x^2 - 3*x + 2 > 0}

/-- Solution set B for mx^2 - (m+2)x + 2 < 0, where m ∈ ℝ -/
def B (m : ℝ) : Set ℝ := {x | m*x^2 - (m+2)*x + 2 < 0}

/-- Complement of A in ℝ -/
def complement_A : Set ℝ := {x | ¬(x ∈ A)}

theorem quadratic_inequality_solutions (m : ℝ) :
  (B m ⊆ complement_A ↔ 1 ≤ m ∧ m ≤ 2) ∧
  ((A ∩ B m).Nonempty ↔ m < 1 ∨ m > 2) ∧
  (A ∪ B m = A ↔ m ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l1225_122504


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l1225_122592

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) 
  (h1 : x = -1) 
  (h2 : y = 1) 
  (h3 : ρ > 0) 
  (h4 : 0 < θ ∧ θ < π) 
  (h5 : ρ = Real.sqrt (x^2 + y^2)) 
  (h6 : θ = Real.arctan (y / x) + π) : 
  (ρ, θ) = (Real.sqrt 2, 3 * π / 4) := by
sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l1225_122592


namespace NUMINAMATH_CALUDE_total_games_in_season_l1225_122530

theorem total_games_in_season (total_teams : ℕ) (num_divisions : ℕ) (teams_per_division : ℕ)
  (intra_division_games : ℕ) (inter_division_games : ℕ)
  (h1 : total_teams = 24)
  (h2 : num_divisions = 3)
  (h3 : teams_per_division = 8)
  (h4 : total_teams = num_divisions * teams_per_division)
  (h5 : intra_division_games = 3)
  (h6 : inter_division_games = 2) :
  (total_teams * (((teams_per_division - 1) * intra_division_games) +
   ((total_teams - teams_per_division) * inter_division_games))) / 2 = 636 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_season_l1225_122530


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l1225_122550

-- Define the propositions P and Q
def P (x : ℝ) : Prop := |2*x - 3| < 1
def Q (x : ℝ) : Prop := x*(x - 3) < 0

-- Theorem stating that P is a sufficient but not necessary condition for Q
theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ 
  (∃ x : ℝ, Q x ∧ ¬(P x)) := by
  sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l1225_122550


namespace NUMINAMATH_CALUDE_first_discount_percentage_l1225_122503

theorem first_discount_percentage (original_price final_price : ℝ) 
  (h1 : original_price = 10000)
  (h2 : final_price = 6840) : ∃ x : ℝ,
  final_price = original_price * (100 - x) / 100 * 90 / 100 * 95 / 100 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l1225_122503


namespace NUMINAMATH_CALUDE_modulus_z_l1225_122523

theorem modulus_z (r k : ℝ) (z : ℂ) 
  (hr : |r| < 2) 
  (hk : |k| < 3) 
  (hz : z + k * z⁻¹ = r) : 
  Complex.abs z = Real.sqrt ((r^2 - 2*k) / 2) := by
  sorry

end NUMINAMATH_CALUDE_modulus_z_l1225_122523


namespace NUMINAMATH_CALUDE_unique_g_3_l1225_122590

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the conditions
axiom g_1 : g 1 = -1
axiom g_property : ∀ x y : ℝ, g (x^2 - y^2) = (x - y) * (g x - g y)

-- Define m as the number of possible values of g(3)
def m : ℕ := sorry

-- Define t as the sum of all possible values of g(3)
def t : ℝ := sorry

-- Theorem statement
theorem unique_g_3 : m = 1 ∧ t = -3 := by sorry

end NUMINAMATH_CALUDE_unique_g_3_l1225_122590


namespace NUMINAMATH_CALUDE_united_additional_charge_value_l1225_122545

/-- Represents the additional charge per minute for United Telephone -/
def united_additional_charge : ℝ := sorry

/-- The base rate for United Telephone -/
def united_base_rate : ℝ := 11

/-- The base rate for Atlantic Call -/
def atlantic_base_rate : ℝ := 12

/-- The additional charge per minute for Atlantic Call -/
def atlantic_additional_charge : ℝ := 0.2

/-- The number of minutes for which the bills are equal -/
def equal_bill_minutes : ℝ := 20

theorem united_additional_charge_value : 
  (united_base_rate + equal_bill_minutes * united_additional_charge = 
   atlantic_base_rate + equal_bill_minutes * atlantic_additional_charge) → 
  united_additional_charge = 0.25 := by sorry

end NUMINAMATH_CALUDE_united_additional_charge_value_l1225_122545


namespace NUMINAMATH_CALUDE_storage_unit_blocks_l1225_122548

/-- Represents the dimensions of a rectangular storage unit -/
structure StorageUnit where
  length : ℝ
  width : ℝ
  height : ℝ
  wallThickness : ℝ

/-- Calculates the number of blocks needed for a storage unit -/
def blocksNeeded (unit : StorageUnit) : ℝ :=
  let totalVolume := unit.length * unit.width * unit.height
  let interiorLength := unit.length - 2 * unit.wallThickness
  let interiorWidth := unit.width - 2 * unit.wallThickness
  let interiorHeight := unit.height - unit.wallThickness
  let interiorVolume := interiorLength * interiorWidth * interiorHeight
  totalVolume - interiorVolume

/-- Theorem stating that the storage unit with given dimensions requires 738 blocks -/
theorem storage_unit_blocks :
  let unit : StorageUnit := {
    length := 15,
    width := 12,
    height := 8,
    wallThickness := 1.5
  }
  blocksNeeded unit = 738 := by sorry

end NUMINAMATH_CALUDE_storage_unit_blocks_l1225_122548


namespace NUMINAMATH_CALUDE_min_sum_squares_l1225_122566

theorem min_sum_squares (a b c d e f g h : ℤ) : 
  a ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  b ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  c ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  d ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  e ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  f ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  g ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  h ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → a ≠ g → a ≠ h →
  b ≠ c → b ≠ d → b ≠ e → b ≠ f → b ≠ g → b ≠ h →
  c ≠ d → c ≠ e → c ≠ f → c ≠ g → c ≠ h →
  d ≠ e → d ≠ f → d ≠ g → d ≠ h →
  e ≠ f → e ≠ g → e ≠ h →
  f ≠ g → f ≠ h →
  g ≠ h →
  34 ≤ (a + b + c + d)^2 + (e + f + g + h)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1225_122566


namespace NUMINAMATH_CALUDE_ted_eats_four_cookies_l1225_122543

-- Define the problem parameters
def days : ℕ := 6
def trays_per_day : ℕ := 2
def cookies_per_tray : ℕ := 12
def frank_daily_consumption : ℕ := 1
def cookies_left : ℕ := 134

-- Define the function to calculate Ted's consumption
def ted_consumption : ℕ :=
  days * trays_per_day * cookies_per_tray - 
  days * frank_daily_consumption - 
  cookies_left

-- Theorem statement
theorem ted_eats_four_cookies : ted_consumption = 4 := by
  sorry

end NUMINAMATH_CALUDE_ted_eats_four_cookies_l1225_122543


namespace NUMINAMATH_CALUDE_expected_value_decahedral_die_l1225_122579

/-- A fair decahedral die with faces numbered 1 to 10 -/
def DecahedralDie : Finset ℕ := Finset.range 10

/-- The probability of each outcome for a fair die -/
def prob (n : ℕ) : ℚ := 1 / 10

/-- The expected value of rolling the decahedral die -/
def expected_value : ℚ := (DecahedralDie.sum (fun i => prob i * (i + 1)))

/-- Theorem: The expected value of rolling a fair decahedral die with faces numbered 1 to 10 is 5.5 -/
theorem expected_value_decahedral_die : expected_value = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_decahedral_die_l1225_122579


namespace NUMINAMATH_CALUDE_select_one_each_select_at_least_two_surgical_l1225_122526

/-- The number of nursing experts -/
def num_nursing : ℕ := 3

/-- The number of surgical experts -/
def num_surgical : ℕ := 5

/-- The number of psychological therapy experts -/
def num_psych : ℕ := 2

/-- The total number of experts to be selected -/
def num_selected : ℕ := 4

/-- Function to calculate the number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem for part 1 -/
theorem select_one_each : 
  choose num_surgical 1 * choose num_psych 1 * choose num_nursing 2 = 30 := by sorry

/-- Theorem for part 2 -/
theorem select_at_least_two_surgical :
  (choose 4 1 * choose 4 2 + choose 4 2 * choose 4 1 + choose 4 3) +
  (choose 4 2 * choose 5 2 + choose 4 3 * choose 5 1 + choose 4 4) = 133 := by sorry

end NUMINAMATH_CALUDE_select_one_each_select_at_least_two_surgical_l1225_122526


namespace NUMINAMATH_CALUDE_missing_mark_calculation_l1225_122535

def calculate_missing_mark (english math chemistry biology average : ℕ) : ℕ :=
  5 * average - (english + math + chemistry + biology)

theorem missing_mark_calculation (english math chemistry biology average : ℕ) :
  calculate_missing_mark english math chemistry biology average =
  5 * average - (english + math + chemistry + biology) :=
by sorry

end NUMINAMATH_CALUDE_missing_mark_calculation_l1225_122535


namespace NUMINAMATH_CALUDE_tank_water_problem_l1225_122561

theorem tank_water_problem (added_saline : ℝ) (salt_concentration_added : ℝ) 
  (salt_concentration_final : ℝ) (initial_water : ℝ) : 
  added_saline = 66.67 →
  salt_concentration_added = 0.25 →
  salt_concentration_final = 0.10 →
  initial_water = 100 →
  salt_concentration_added * added_saline = 
    salt_concentration_final * (initial_water + added_saline) :=
by sorry

end NUMINAMATH_CALUDE_tank_water_problem_l1225_122561


namespace NUMINAMATH_CALUDE_max_gcd_sum_1729_l1225_122534

theorem max_gcd_sum_1729 (a b : ℕ+) (h : a + b = 1729) : 
  ∃ (x y : ℕ+), x + y = 1729 ∧ Nat.gcd x y = 247 ∧ 
  ∀ (c d : ℕ+), c + d = 1729 → Nat.gcd c d ≤ 247 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1729_l1225_122534


namespace NUMINAMATH_CALUDE_xiaoming_calculation_correction_l1225_122583

theorem xiaoming_calculation_correction 
  (A a b c : ℝ) 
  (h : A + 2 * (a * b + 2 * b * c - 4 * a * c) = 3 * a * b - 2 * a * c + 5 * b * c) : 
  A - 2 * (a * b + 2 * b * c - 4 * a * c) = -a * b + 14 * a * c - 3 * b * c := by
sorry

end NUMINAMATH_CALUDE_xiaoming_calculation_correction_l1225_122583


namespace NUMINAMATH_CALUDE_simultaneous_congruences_l1225_122556

theorem simultaneous_congruences (x : ℤ) :
  x % 2 = 1 ∧ x % 3 = 2 ∧ x % 5 = 3 ∧ x % 7 = 4 → x % 210 = 53 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_congruences_l1225_122556


namespace NUMINAMATH_CALUDE_mans_speed_with_current_is_25_l1225_122540

/-- Given a man's speed against a current and the current's speed, 
    calculate the man's speed with the current. -/
def mans_speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed with the current is 25 km/hr. -/
theorem mans_speed_with_current_is_25 :
  mans_speed_with_current 20 2.5 = 25 := by
  sorry

#eval mans_speed_with_current 20 2.5

end NUMINAMATH_CALUDE_mans_speed_with_current_is_25_l1225_122540


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1225_122538

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ 0 < a ∧ a = b ∧ b < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1225_122538


namespace NUMINAMATH_CALUDE_max_sum_of_square_roots_l1225_122546

theorem max_sum_of_square_roots (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  (Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2)) ≤ 9 ∧
  ∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 7 ∧
    Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_square_roots_l1225_122546


namespace NUMINAMATH_CALUDE_no_real_roots_of_quadratic_l1225_122562

theorem no_real_roots_of_quadratic (x : ℝ) : ¬∃x, x^2 - 4*x + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_of_quadratic_l1225_122562


namespace NUMINAMATH_CALUDE_savings_difference_l1225_122565

def initial_order : ℝ := 15000

def option1_discounts : List ℝ := [0.10, 0.25, 0.15]
def option2_discounts : List ℝ := [0.30, 0.10, 0.05]

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem savings_difference :
  apply_discounts initial_order option2_discounts - 
  apply_discounts initial_order option1_discounts = 371.25 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l1225_122565


namespace NUMINAMATH_CALUDE_trees_on_promenade_l1225_122591

/-- The number of trees planted along a circular promenade -/
def number_of_trees (promenade_length : ℕ) (tree_interval : ℕ) : ℕ :=
  promenade_length / tree_interval

/-- Theorem: The number of trees planted along a circular promenade of length 1200 meters, 
    with trees planted at intervals of 30 meters, is equal to 40. -/
theorem trees_on_promenade : number_of_trees 1200 30 = 40 := by
  sorry

end NUMINAMATH_CALUDE_trees_on_promenade_l1225_122591


namespace NUMINAMATH_CALUDE_three_hundredth_term_omit_squares_l1225_122581

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def omit_squares_sequence (n : ℕ) : ℕ :=
  n + (Nat.sqrt n)

theorem three_hundredth_term_omit_squares : omit_squares_sequence 300 = 317 := by
  sorry

end NUMINAMATH_CALUDE_three_hundredth_term_omit_squares_l1225_122581


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1225_122516

/-- The set of points (x, y) satisfying y(x+1) = x^2 - 1 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 * (p.1 + 1) = p.1^2 - 1}

/-- The vertical line x = -1 -/
def L1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -1}

/-- The line y = x - 1 -/
def L2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 1}

/-- Theorem stating that S is equivalent to the union of L1 and L2 -/
theorem solution_set_equivalence : S = L1 ∪ L2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1225_122516


namespace NUMINAMATH_CALUDE_train_length_l1225_122599

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) : 
  speed_kmph = 18 → crossing_time = 5 → 
  (speed_kmph * 1000 / 3600) * crossing_time = 25 := by sorry

end NUMINAMATH_CALUDE_train_length_l1225_122599


namespace NUMINAMATH_CALUDE_square_root_representation_l1225_122559

theorem square_root_representation (x : ℝ) (h : x = 0.25) :
  ∃ y : ℝ, y > 0 ∧ y^2 = x ∧ (∀ z : ℝ, z^2 = x → z = y ∨ z = -y) :=
by sorry

end NUMINAMATH_CALUDE_square_root_representation_l1225_122559


namespace NUMINAMATH_CALUDE_defective_pens_l1225_122522

theorem defective_pens (total_pens : ℕ) (prob_non_defective : ℚ) (defective_pens : ℕ) : 
  total_pens = 9 →
  prob_non_defective = 5 / 12 →
  (total_pens - defective_pens : ℚ) / total_pens * ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = prob_non_defective →
  defective_pens = 3 := by
sorry

end NUMINAMATH_CALUDE_defective_pens_l1225_122522


namespace NUMINAMATH_CALUDE_original_price_proof_l1225_122589

/-- The original price of a part before discount -/
def original_price : ℝ := 62.71

/-- The number of parts Clark bought -/
def num_parts : ℕ := 7

/-- The total amount Clark paid after discount -/
def total_paid : ℝ := 439

/-- Theorem stating that the original price multiplied by the number of parts equals the total amount paid -/
theorem original_price_proof : original_price * num_parts = total_paid := by sorry

end NUMINAMATH_CALUDE_original_price_proof_l1225_122589


namespace NUMINAMATH_CALUDE_exp_gt_one_plus_x_l1225_122527

theorem exp_gt_one_plus_x (x : ℝ) (h : x ≠ 0) : Real.exp x > 1 + x := by
  sorry

end NUMINAMATH_CALUDE_exp_gt_one_plus_x_l1225_122527


namespace NUMINAMATH_CALUDE_divisors_of_power_difference_l1225_122598

theorem divisors_of_power_difference (n : ℕ) :
  n = 11^60 - 17^24 →
  ∃ (d : ℕ), d ≥ 120 ∧ (∀ (x : ℕ), x ∣ n → x > 0 → x ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_power_difference_l1225_122598


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l1225_122586

/-- A particle moving in a straight line with distance-time relationship s(t) = 4t^2 - 3 -/
def s (t : ℝ) : ℝ := 4 * t^2 - 3

/-- The instantaneous velocity function v(t) -/
def v (t : ℝ) : ℝ := 8 * t

theorem instantaneous_velocity_at_5 : v 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l1225_122586


namespace NUMINAMATH_CALUDE_probability_of_speaking_hindi_l1225_122514

/-- The probability of speaking Hindi in a village -/
theorem probability_of_speaking_hindi 
  (total_population : ℕ) 
  (tamil_speakers : ℕ) 
  (english_speakers : ℕ) 
  (both_speakers : ℕ) 
  (h_total : total_population = 1024)
  (h_tamil : tamil_speakers = 720)
  (h_english : english_speakers = 562)
  (h_both : both_speakers = 346)
  (h_non_negative : total_population ≥ tamil_speakers + english_speakers - both_speakers) :
  (total_population - (tamil_speakers + english_speakers - both_speakers)) / total_population = 
  (1024 - (720 + 562 - 346)) / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_speaking_hindi_l1225_122514


namespace NUMINAMATH_CALUDE_superhero_movie_count_l1225_122529

theorem superhero_movie_count (total_movies : ℕ) (dalton_movies : ℕ) (alex_movies : ℕ) (shared_movies : ℕ) :
  total_movies = 30 →
  dalton_movies = 7 →
  alex_movies = 15 →
  shared_movies = 2 →
  ∃ (hunter_movies : ℕ), hunter_movies = total_movies - dalton_movies - alex_movies + shared_movies :=
by
  sorry

end NUMINAMATH_CALUDE_superhero_movie_count_l1225_122529


namespace NUMINAMATH_CALUDE_sum_m_n_equals_three_l1225_122570

theorem sum_m_n_equals_three (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m + 5 < n)
  (h4 : (m + (m + 3) + (m + 5) + n + (n + 2) + (2 * n - 1)) / 6 = n + 1)
  (h5 : ((m + 5) + n) / 2 = n + 1) : m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_three_l1225_122570


namespace NUMINAMATH_CALUDE_line_equations_l1225_122553

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Define the property of line l1
def line_l1_property (l : ℝ → ℝ → Prop) : Prop :=
  l point_P.1 point_P.2 ∧
  ∃ a b : ℝ, ∀ x y : ℝ, l x y ↔ (x = a ∨ b*x - y = 0) ∧
  ∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧ l x1 y1 ∧ l x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 32

-- Define the property of line l2
def line_l2_property (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y z w : ℝ, l x y ∧ l z w → y - x = w - z) ∧
  ∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧ l x1 y1 ∧ l x2 y2 ∧
    x1*x2 + y1*y2 = 0

-- Theorem statement
theorem line_equations :
  ∃ l1 l2 : ℝ → ℝ → Prop,
    line_l1_property l1 ∧ line_l2_property l2 ∧
    (∀ x y : ℝ, l1 x y ↔ (x = 2 ∨ 3*x - 4*y - 6 = 0)) ∧
    (∀ x y : ℝ, l2 x y ↔ (x - y - 4 = 0 ∨ x - y + 1 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l1225_122553


namespace NUMINAMATH_CALUDE_least_divisible_by_240_cubed_l1225_122588

theorem least_divisible_by_240_cubed (a : ℕ) : 
  (∀ n : ℕ, n < 60 → ¬(240 ∣ n^3)) ∧ (240 ∣ 60^3) := by
sorry

end NUMINAMATH_CALUDE_least_divisible_by_240_cubed_l1225_122588


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1225_122587

theorem inequality_solution_set :
  {x : ℝ | |x - 1| + 2*x > 4} = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1225_122587


namespace NUMINAMATH_CALUDE_max_product_with_linear_constraint_max_product_achieved_l1225_122542

theorem max_product_with_linear_constraint (a b : ℝ) :
  a > 0 → b > 0 → 6 * a + 5 * b = 75 → a * b ≤ 46.875 := by
  sorry

theorem max_product_achieved (a b : ℝ) :
  a > 0 → b > 0 → 6 * a + 5 * b = 75 → a * b = 46.875 → a = 75 / 11 ∧ b = 90 / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_product_with_linear_constraint_max_product_achieved_l1225_122542


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1225_122539

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : 
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1225_122539


namespace NUMINAMATH_CALUDE_talent_show_proof_l1225_122597

theorem talent_show_proof (total : ℕ) (cant_sing cant_dance cant_act : ℕ) : 
  total = 120 →
  cant_sing = 50 →
  cant_dance = 75 →
  cant_act = 35 →
  let can_sing := total - cant_sing
  let can_dance := total - cant_dance
  let can_act := total - cant_act
  let two_talents := can_sing + can_dance + can_act - total
  two_talents = 80 := by
sorry

end NUMINAMATH_CALUDE_talent_show_proof_l1225_122597


namespace NUMINAMATH_CALUDE_deer_per_hunting_wolf_l1225_122555

theorem deer_per_hunting_wolf (hunting_wolves : ℕ) (additional_wolves : ℕ) 
  (meat_per_wolf_per_day : ℕ) (days_between_hunts : ℕ) (meat_per_deer : ℕ) : 
  hunting_wolves = 4 →
  additional_wolves = 16 →
  meat_per_wolf_per_day = 8 →
  days_between_hunts = 5 →
  meat_per_deer = 200 →
  (hunting_wolves + additional_wolves) * meat_per_wolf_per_day * days_between_hunts / 
  (meat_per_deer * hunting_wolves) = 1 := by
sorry

end NUMINAMATH_CALUDE_deer_per_hunting_wolf_l1225_122555


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l1225_122501

theorem abs_sum_minimum (x : ℝ) : 
  |x + 3| + |x + 5| + |x + 6| ≥ 1 ∧ ∃ y : ℝ, |y + 3| + |y + 5| + |y + 6| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l1225_122501


namespace NUMINAMATH_CALUDE_tyler_cake_servings_l1225_122525

/-- The number of people the original recipe serves -/
def original_recipe_servings : ℕ := 4

/-- The number of eggs required for the original recipe -/
def original_recipe_eggs : ℕ := 2

/-- The total number of eggs Tyler needs for his cake -/
def tylers_eggs : ℕ := 4

/-- The number of people Tyler wants to make the cake for -/
def tylers_servings : ℕ := 8

theorem tyler_cake_servings :
  tylers_servings = original_recipe_servings * (tylers_eggs / original_recipe_eggs) :=
by sorry

end NUMINAMATH_CALUDE_tyler_cake_servings_l1225_122525


namespace NUMINAMATH_CALUDE_limit_sin_difference_l1225_122505

theorem limit_sin_difference (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
    |(1 / (4 * Real.sin x ^ 2) - 1 / Real.sin (2 * x) ^ 2) - (-1/4)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_sin_difference_l1225_122505


namespace NUMINAMATH_CALUDE_similar_triangles_segment_length_l1225_122594

/-- Triangle similarity is a relation between two triangles -/
def TriangleSimilar (t1 t2 : Type) : Prop := sorry

/-- Length of a segment -/
def SegmentLength (s : Type) : ℝ := sorry

theorem similar_triangles_segment_length 
  (PQR XYZ GHI : Type) 
  (h1 : TriangleSimilar PQR XYZ) 
  (h2 : TriangleSimilar XYZ GHI) 
  (h3 : SegmentLength PQ = 5) 
  (h4 : SegmentLength QR = 15) 
  (h5 : SegmentLength HI = 30) : 
  SegmentLength XY = 2.5 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_segment_length_l1225_122594


namespace NUMINAMATH_CALUDE_tangent_line_at_one_unique_zero_implies_a_one_max_value_of_g_l1225_122515

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + a * x^2 + 2
def g (a : ℝ) (x : ℝ) : ℝ := f a x - x - 2

-- Part I
theorem tangent_line_at_one (a : ℝ) (h : a = -1) :
  ∃ (m b : ℝ), m = 3 ∧ b = -4 ∧ ∀ x y, y = f a x → y = m * (x - 1) + f a 1 :=
sorry

-- Part II
theorem unique_zero_implies_a_one (a : ℝ) (h : a > 0) :
  (∃! x, g a x = 0) → a = 1 :=
sorry

-- Part III
theorem max_value_of_g (x : ℝ) (h1 : Real.exp (-2) < x) (h2 : x < Real.exp 1) :
  g 1 x ≤ 2 * Real.exp 2 - 3 * Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_unique_zero_implies_a_one_max_value_of_g_l1225_122515


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1225_122547

theorem regular_polygon_interior_angle (C : ℕ) : 
  C > 2 → (288 : ℝ) = (C - 2 : ℝ) * 180 / C → C = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1225_122547


namespace NUMINAMATH_CALUDE_favorite_fruit_strawberries_l1225_122519

theorem favorite_fruit_strawberries (total : ℕ) (oranges pears apples : ℕ)
  (h1 : total = 450)
  (h2 : oranges = 70)
  (h3 : pears = 120)
  (h4 : apples = 147) :
  total - (oranges + pears + apples) = 113 :=
by sorry

end NUMINAMATH_CALUDE_favorite_fruit_strawberries_l1225_122519


namespace NUMINAMATH_CALUDE_sheet_to_box_volume_l1225_122578

/-- Represents the dimensions of a rectangular sheet. -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Represents the sizes of squares cut from corners. -/
structure CornerCuts where
  cut1 : ℝ
  cut2 : ℝ
  cut3 : ℝ
  cut4 : ℝ

/-- Represents the dimensions of the resulting box. -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions. -/
def boxVolume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

/-- Theorem stating the relationship between the original sheet, corner cuts, and resulting box. -/
theorem sheet_to_box_volume 
  (sheet : SheetDimensions) 
  (cuts : CornerCuts) 
  (box : BoxDimensions) : 
  sheet.length = 48 ∧ 
  sheet.width = 36 ∧
  cuts.cut1 = 7 ∧ 
  cuts.cut2 = 5 ∧ 
  cuts.cut3 = 6 ∧ 
  cuts.cut4 = 4 ∧
  box.length = sheet.length - (cuts.cut1 + cuts.cut4) ∧
  box.width = sheet.width - (cuts.cut2 + cuts.cut3) ∧
  box.height = min cuts.cut1 (min cuts.cut2 (min cuts.cut3 cuts.cut4)) →
  boxVolume box = 3700 ∧ 
  box.length = 37 ∧ 
  box.width = 25 := by
  sorry

end NUMINAMATH_CALUDE_sheet_to_box_volume_l1225_122578


namespace NUMINAMATH_CALUDE_log_sum_equals_zero_l1225_122575

theorem log_sum_equals_zero (a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (h_log : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_log_sum_equals_zero_l1225_122575


namespace NUMINAMATH_CALUDE_second_longest_piece_length_l1225_122502

/-- The length of the second longest piece of rope when a 142.75-inch rope is cut into five pieces
    in the ratio (√2):6:(4/3):(3^2):(1/2) is approximately 46.938 inches. -/
theorem second_longest_piece_length (total_length : ℝ) (piece1 piece2 piece3 piece4 piece5 : ℝ)
  (h1 : total_length = 142.75)
  (h2 : piece1 / (Real.sqrt 2) = piece2 / 6)
  (h3 : piece2 / 6 = piece3 / (4/3))
  (h4 : piece3 / (4/3) = piece4 / 9)
  (h5 : piece4 / 9 = piece5 / (1/2))
  (h6 : piece1 + piece2 + piece3 + piece4 + piece5 = total_length) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |piece2 - 46.938| < ε ∧
  (piece2 > piece1 ∨ piece2 > piece3 ∨ piece2 > piece5) ∧
  (piece4 > piece2 ∨ piece4 = piece2) :=
by sorry

end NUMINAMATH_CALUDE_second_longest_piece_length_l1225_122502


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l1225_122552

/-- The range of m for which the line 2kx-y+1=0 always intersects the ellipse x²/9 + y²/m = 1 -/
theorem line_ellipse_intersection_range :
  ∀ (k : ℝ), (∀ (x y : ℝ), 2 * k * x - y + 1 = 0 → x^2 / 9 + y^2 / m = 1) →
  m ∈ Set.Icc 1 9 ∪ Set.Ioi 9 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l1225_122552


namespace NUMINAMATH_CALUDE_lateral_edge_length_l1225_122577

/-- A regular pyramid with a square base -/
structure RegularPyramid where
  -- The side length of the square base
  base_side : ℝ
  -- The volume of the pyramid
  volume : ℝ
  -- The length of a lateral edge
  lateral_edge : ℝ

/-- Theorem: In a regular pyramid with square base, if the volume is 4/3 and the base side length is 2, 
    then the lateral edge length is √3 -/
theorem lateral_edge_length (p : RegularPyramid) 
  (h1 : p.volume = 4/3) 
  (h2 : p.base_side = 2) : 
  p.lateral_edge = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_lateral_edge_length_l1225_122577


namespace NUMINAMATH_CALUDE_reflection_sum_l1225_122531

/-- Given a point C with coordinates (3, y) that is reflected over the line y = x to point D,
    the sum of all coordinate values of C and D is equal to 2y + 6. -/
theorem reflection_sum (y : ℝ) : 
  let C := (3, y)
  let D := (y, 3)
  (C.1 + C.2 + D.1 + D.2) = 2 * y + 6 := by
  sorry

end NUMINAMATH_CALUDE_reflection_sum_l1225_122531


namespace NUMINAMATH_CALUDE_water_difference_l1225_122567

theorem water_difference (s h : ℝ) 
  (h1 : s > h) 
  (h2 : (s - 0.43) - (h + 0.43) = 0.88) : 
  s - h = 1.74 := by
  sorry

end NUMINAMATH_CALUDE_water_difference_l1225_122567


namespace NUMINAMATH_CALUDE_largest_common_divisor_342_285_l1225_122573

theorem largest_common_divisor_342_285 : ∃ (n : ℕ), n > 0 ∧ n ∣ 342 ∧ n ∣ 285 ∧ ∀ (m : ℕ), m > n → (m ∣ 342 ∧ m ∣ 285 → False) :=
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_342_285_l1225_122573


namespace NUMINAMATH_CALUDE_equation_solution_l1225_122507

theorem equation_solution : 
  ∃! x : ℚ, (x - 15) / 3 = (3 * x + 10) / 8 :=
by
  use (-150)
  constructor
  · -- Prove that x = -150 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_equation_solution_l1225_122507


namespace NUMINAMATH_CALUDE_sun_valley_combined_population_sun_valley_combined_population_proof_l1225_122563

/-- Proves that the combined population of Sun City and Valley City is 41550 given the conditions in the problem. -/
theorem sun_valley_combined_population : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun willowdale roseville sun x valley =>
    willowdale = 2000 ∧
    roseville = 3 * willowdale - 500 ∧
    sun = 2 * roseville + 1000 ∧
    x = (6 * sun) / 10 ∧
    valley = 4 * x + 750 →
    sun + valley = 41550

/-- Proof of the theorem -/
theorem sun_valley_combined_population_proof : 
  ∃ (willowdale roseville sun x valley : ℕ), 
    sun_valley_combined_population willowdale roseville sun x valley :=
by
  sorry

#check sun_valley_combined_population
#check sun_valley_combined_population_proof

end NUMINAMATH_CALUDE_sun_valley_combined_population_sun_valley_combined_population_proof_l1225_122563


namespace NUMINAMATH_CALUDE_solution_set_eq_l1225_122564

-- Define a decreasing function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_decreasing : ∀ x y, x < y → f x > f y
axiom f_0 : f 0 = -2
axiom f_neg_3 : f (-3) = 2

-- Define the solution set
def solution_set : Set ℝ := {x | |f (x - 2)| > 2}

-- State the theorem
theorem solution_set_eq : solution_set = Set.Iic (-1) ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_eq_l1225_122564


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l1225_122500

theorem max_product_sum_2000 : 
  ∀ x y : ℤ, x + y = 2000 → x * y ≤ 1000000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l1225_122500


namespace NUMINAMATH_CALUDE_cosine_product_identity_l1225_122595

theorem cosine_product_identity (n : ℕ) (hn : n = 7 ∨ n = 9) : 
  Real.cos (2 * Real.pi / n) * Real.cos (4 * Real.pi / n) * Real.cos (8 * Real.pi / n) = 
  (-1 : ℝ) ^ ((n - 1) / 2) * (1 / 8) :=
by sorry

end NUMINAMATH_CALUDE_cosine_product_identity_l1225_122595
