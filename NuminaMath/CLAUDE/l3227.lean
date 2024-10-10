import Mathlib

namespace base_seven_digits_of_1234_digits_in_base_seven_1234_l3227_322712

theorem base_seven_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n :=
by
  -- The proof would go here
  sorry

theorem digits_in_base_seven_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n ∧ n = 4 :=
by
  -- The proof would go here
  sorry

end base_seven_digits_of_1234_digits_in_base_seven_1234_l3227_322712


namespace proportion_theorem_l3227_322799

theorem proportion_theorem (y : ℝ) : 
  (0.75 : ℝ) / 0.9 = y / 6 → y = 5 := by
  sorry

end proportion_theorem_l3227_322799


namespace amount_problem_l3227_322718

theorem amount_problem (a b : ℝ) 
  (h1 : a + b = 1210)
  (h2 : (4/5) * a = (2/3) * b) :
  b = 453.75 := by
sorry

end amount_problem_l3227_322718


namespace right_triangle_tan_l3227_322751

theorem right_triangle_tan (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 40) (h3 : c = 41) :
  b / a = 9 / 40 := by
  sorry

end right_triangle_tan_l3227_322751


namespace joan_balloons_l3227_322740

/-- Given that Joan initially has 9 blue balloons and loses 2 balloons,
    prove that she has 7 blue balloons remaining. -/
theorem joan_balloons : 
  let initial_balloons : ℕ := 9
  let lost_balloons : ℕ := 2
  initial_balloons - lost_balloons = 7 := by
sorry

end joan_balloons_l3227_322740


namespace rectangle_length_equals_eight_l3227_322763

theorem rectangle_length_equals_eight
  (square_perimeter : ℝ)
  (rectangle_width : ℝ)
  (triangle_height : ℝ)
  (h1 : square_perimeter = 64)
  (h2 : rectangle_width = 8)
  (h3 : triangle_height = 64)
  (h4 : (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * rectangle_length) :
  rectangle_length = 8 :=
by
  sorry

end rectangle_length_equals_eight_l3227_322763


namespace tangent_line_to_circle_l3227_322762

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the point P
def P : ℝ × ℝ := (2, 4)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x + 2*y - 10 = 0

theorem tangent_line_to_circle :
  (∀ x y, C x y → ¬(tangent_line x y)) ∧
  tangent_line P.1 P.2 ∧
  ∃! p : ℝ × ℝ, C p.1 p.2 ∧ tangent_line p.1 p.2 ∧ p ≠ P :=
sorry

end tangent_line_to_circle_l3227_322762


namespace sqrt_product_is_eight_l3227_322705

theorem sqrt_product_is_eight :
  Real.sqrt (9 - Real.sqrt 77) * Real.sqrt 2 * (Real.sqrt 11 - Real.sqrt 7) * (9 + Real.sqrt 77) = 8 := by
  sorry

end sqrt_product_is_eight_l3227_322705


namespace product_price_l3227_322781

/-- Given that m kilograms of a product costs 9 yuan, 
    prove that n kilograms of the same product costs (9n/m) yuan. -/
theorem product_price (m n : ℝ) (hm : m > 0) : 
  (9 : ℝ) / m * n = 9 * n / m := by sorry

end product_price_l3227_322781


namespace exponential_function_passes_through_point_zero_one_l3227_322797

theorem exponential_function_passes_through_point_zero_one
  (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x
  f 0 = 1 := by
sorry

end exponential_function_passes_through_point_zero_one_l3227_322797


namespace scientific_notation_correct_l3227_322796

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be converted -/
def original_number : ℕ := 2270000

/-- Converts a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation := sorry

theorem scientific_notation_correct :
  let sn := to_scientific_notation original_number
  sn.coefficient = 2.27 ∧ sn.exponent = 6 ∧ original_number = sn.coefficient * (10 ^ sn.exponent) := by
  sorry

end scientific_notation_correct_l3227_322796


namespace length_PS_specific_quadrilateral_l3227_322701

/-- A quadrilateral with two right angles and specified side lengths -/
structure RightQuadrilateral where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  angle_Q_is_right : Bool
  angle_R_is_right : Bool

/-- The length of PS in a right quadrilateral PQRS -/
def length_PS (quad : RightQuadrilateral) : ℝ :=
  sorry

/-- Theorem: In a right quadrilateral PQRS where PQ = 7, QR = 10, RS = 25, 
    and angles Q and R are right angles, the length of PS is 2√106 -/
theorem length_PS_specific_quadrilateral :
  let quad : RightQuadrilateral := {
    PQ := 7,
    QR := 10,
    RS := 25,
    angle_Q_is_right := true,
    angle_R_is_right := true
  }
  length_PS quad = 2 * Real.sqrt 106 := by
  sorry

end length_PS_specific_quadrilateral_l3227_322701


namespace delegates_with_female_count_l3227_322704

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose delegates with at least one female student. -/
def delegates_with_female (male_count female_count delegate_count : ℕ) : ℕ :=
  (choose female_count 1 * choose male_count (delegate_count - 1)) +
  (choose female_count 2 * choose male_count (delegate_count - 2)) +
  (choose female_count 3 * choose male_count (delegate_count - 3))

theorem delegates_with_female_count :
  delegates_with_female 4 3 3 = 31 := by sorry

end delegates_with_female_count_l3227_322704


namespace greatest_abcba_divisible_by_13_l3227_322707

def is_valid_abcba (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_abcba_divisible_by_13 :
  ∀ n : ℕ, is_valid_abcba n → n ≤ 96769 ∧ 96769 % 13 = 0 ∧ is_valid_abcba 96769 :=
sorry

end greatest_abcba_divisible_by_13_l3227_322707


namespace harry_owns_three_geckos_l3227_322765

/-- Represents the number of geckos Harry owns -/
def num_geckos : ℕ := 3

/-- Represents the number of iguanas Harry owns -/
def num_iguanas : ℕ := 2

/-- Represents the number of snakes Harry owns -/
def num_snakes : ℕ := 4

/-- Represents the monthly feeding cost per snake in dollars -/
def snake_cost : ℕ := 10

/-- Represents the monthly feeding cost per iguana in dollars -/
def iguana_cost : ℕ := 5

/-- Represents the monthly feeding cost per gecko in dollars -/
def gecko_cost : ℕ := 15

/-- Represents the total annual feeding cost for all pets in dollars -/
def total_annual_cost : ℕ := 1140

/-- Theorem stating that the number of geckos Harry owns is 3 -/
theorem harry_owns_three_geckos :
  num_geckos = 3 ∧
  num_geckos * gecko_cost * 12 + num_iguanas * iguana_cost * 12 + num_snakes * snake_cost * 12 = total_annual_cost :=
by sorry

end harry_owns_three_geckos_l3227_322765


namespace sphere_volume_circumscribing_rectangular_solid_l3227_322739

/-- The volume of a sphere that circumscribes a rectangular solid with dimensions 3, 2, and 1 -/
theorem sphere_volume_circumscribing_rectangular_solid :
  let length : ℝ := 3
  let width : ℝ := 2
  let height : ℝ := 1
  let radius : ℝ := Real.sqrt (length^2 + width^2 + height^2) / 2
  let volume : ℝ := (4 / 3) * Real.pi * radius^3
  volume = (7 * Real.sqrt 14 * Real.pi) / 3 := by
  sorry

#check sphere_volume_circumscribing_rectangular_solid

end sphere_volume_circumscribing_rectangular_solid_l3227_322739


namespace quadratic_real_roots_range_l3227_322775

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + k = 0) ↔ k ≤ 9/4 := by
  sorry

end quadratic_real_roots_range_l3227_322775


namespace zhang_ning_match_results_l3227_322738

/-- Represents the outcome of a badminton match --/
inductive MatchOutcome
  | WinTwoZero
  | WinTwoOne
  | LoseTwoOne
  | LoseTwoZero

/-- Probability of Xie Xingfang winning a single set in the first two sets --/
def p_xie : ℝ := 0.6

/-- Probability of Zhang Ning winning the third set if the score reaches 1:1 --/
def p_zhang_third : ℝ := 0.6

/-- Calculates the probability of Zhang Ning winning with a score of 2:1 --/
def prob_zhang_win_two_one : ℝ :=
  2 * (1 - p_xie) * p_xie * p_zhang_third

/-- Calculates the expected value of Zhang Ning's net winning sets --/
def expected_net_wins : ℝ :=
  -2 * (p_xie * p_xie) +
  -1 * (2 * (1 - p_xie) * p_xie * (1 - p_zhang_third)) +
  1 * prob_zhang_win_two_one +
  2 * ((1 - p_xie) * (1 - p_xie))

/-- Theorem stating the probability of Zhang Ning winning 2:1 and her expected net winning sets --/
theorem zhang_ning_match_results :
  prob_zhang_win_two_one = 0.288 ∧ expected_net_wins = 0.496 := by
  sorry


end zhang_ning_match_results_l3227_322738


namespace product_of_differences_l3227_322743

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2006) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2007)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2006) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2007)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2006) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2007) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/2006 := by
  sorry

end product_of_differences_l3227_322743


namespace remainder_problem_l3227_322747

theorem remainder_problem (n : ℤ) : n % 8 = 3 → (4 * n - 9) % 8 = 3 := by
  sorry

end remainder_problem_l3227_322747


namespace arithmetic_sequence_sum_l3227_322770

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 - a 9 + a 17 = 7 →
  a 3 + a 15 = 14 := by
sorry

end arithmetic_sequence_sum_l3227_322770


namespace expression_equals_503_l3227_322717

theorem expression_equals_503 : 2015 * (1999/2015) * (1/4) - 2011/2015 = 503 := by
  sorry

end expression_equals_503_l3227_322717


namespace segment_distinctness_l3227_322728

theorem segment_distinctness (n : ℕ) (h : n ≥ 4) :
  ¬ ∀ (points : Fin (n + 1) → ℕ),
    (points 0 = 0 ∧ points (Fin.last n) = (n^2 + n) / 2) →
    (∀ i j : Fin (n + 1), i < j → points i < points j) →
    (∀ i j k l : Fin (n + 1), i < j ∧ k < l → 
      (points j - points i ≠ points l - points k ∨ (i = k ∧ j = l))) :=
by sorry

end segment_distinctness_l3227_322728


namespace triangle_area_is_two_l3227_322766

/-- A line in 2D space --/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The area of a triangle given by three points --/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- The intersection point of two lines --/
def lineIntersection (l1 l2 : Line) : ℝ × ℝ := sorry

/-- The main theorem --/
theorem triangle_area_is_two :
  let line1 : Line := { slope := 3/4, point := (3, 3) }
  let line2 : Line := { slope := -1, point := (3, 3) }
  let line3 : Line := { slope := -1, point := (0, 14) }
  let p1 := (3, 3)
  let p2 := lineIntersection line1 line3
  let p3 := lineIntersection line2 line3
  triangleArea p1 p2 p3 = 2 := by sorry

end triangle_area_is_two_l3227_322766


namespace trig_identity_special_case_l3227_322723

theorem trig_identity_special_case : 
  Real.cos (60 * π / 180 + 30 * π / 180) * Real.cos (60 * π / 180 - 30 * π / 180) + 
  Real.sin (60 * π / 180 + 30 * π / 180) * Real.sin (60 * π / 180 - 30 * π / 180) = 1/2 := by
  sorry

end trig_identity_special_case_l3227_322723


namespace largest_solution_floor_equation_l3227_322710

theorem largest_solution_floor_equation :
  let floor_eq (x : ℝ) := ⌊x⌋ = 7 + 50 * (x - ⌊x⌋)
  ∃ (max_sol : ℝ), floor_eq max_sol ∧
    ∀ (y : ℝ), floor_eq y → y ≤ max_sol ∧
    max_sol = 2849 / 50
  := by sorry

end largest_solution_floor_equation_l3227_322710


namespace dave_has_least_money_l3227_322706

-- Define the set of people
inductive Person : Type
  | Alice : Person
  | Ben : Person
  | Carol : Person
  | Dave : Person
  | Ethan : Person

-- Define a function to represent the amount of money each person has
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q
axiom ethan_less_than_alice : money Person.Ethan < money Person.Alice
axiom ben_more_than_dave : money Person.Dave < money Person.Ben
axiom carol_more_than_dave : money Person.Dave < money Person.Carol
axiom alice_between_dave_and_ben : money Person.Dave < money Person.Alice ∧ money Person.Alice < money Person.Ben
axiom carol_between_ethan_and_alice : money Person.Ethan < money Person.Carol ∧ money Person.Carol < money Person.Alice

-- Theorem to prove
theorem dave_has_least_money :
  ∀ (p : Person), p ≠ Person.Dave → money Person.Dave < money p :=
sorry

end dave_has_least_money_l3227_322706


namespace stamp_difference_l3227_322727

theorem stamp_difference (k a : ℕ) (h1 : k * 3 = a * 5) 
  (h2 : (k - 12) * 6 = (a + 12) * 8) : k - 12 = (a + 12) + 32 := by
  sorry

end stamp_difference_l3227_322727


namespace max_intersections_count_l3227_322714

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 15

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 10

/-- The total number of segments connecting points on x-axis to points on y-axis -/
def num_segments : ℕ := num_x_points * num_y_points

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := (num_x_points.choose 2) * (num_y_points.choose 2)

theorem max_intersections_count :
  max_intersections = 4725 :=
sorry

end max_intersections_count_l3227_322714


namespace quadratic_equation_k_value_l3227_322733

theorem quadratic_equation_k_value (x1 x2 k : ℝ) : 
  x1^2 - 6*x1 + k = 0 →
  x2^2 - 6*x2 + k = 0 →
  1/x1 + 1/x2 = 3 →
  k = 2 := by
sorry

end quadratic_equation_k_value_l3227_322733


namespace leila_spending_l3227_322756

/-- The amount Leila spent at the supermarket -/
def supermarket_cost : ℝ := 100

/-- The cost of fixing Leila's automobile -/
def automobile_cost : ℝ := 350

/-- The total amount Leila spent -/
def total_cost : ℝ := supermarket_cost + automobile_cost

theorem leila_spending :
  (automobile_cost = 3 * supermarket_cost + 50) →
  total_cost = 450 := by
  sorry

end leila_spending_l3227_322756


namespace probability_sum_three_two_dice_l3227_322789

theorem probability_sum_three_two_dice : 
  let total_outcomes : ℕ := 6 * 6
  let favorable_outcomes : ℕ := 2
  favorable_outcomes / total_outcomes = (1 : ℚ) / 18 := by
  sorry

end probability_sum_three_two_dice_l3227_322789


namespace logarithm_simplification_l3227_322731

theorem logarithm_simplification
  (p q r s t u : ℝ)
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0) :
  Real.log (p / q) - Real.log (q / r) + Real.log (r / s) + Real.log ((s * t) / (p * u)) = Real.log (t / u) :=
by sorry

end logarithm_simplification_l3227_322731


namespace chemical_mixture_problem_l3227_322726

/-- Given two solutions x and y, where:
  - x has A% of chemical a and 90% of chemical b
  - y has 20% of chemical a and 80% of chemical b
  - A mixture of x and y is 12% chemical a
  - The mixture is 80% solution x and 20% solution y
  Prove that A = 10 -/
theorem chemical_mixture_problem (A : ℝ) : 
  A + 90 = 100 →
  0.8 * A + 0.2 * 20 = 12 →
  A = 10 := by sorry

end chemical_mixture_problem_l3227_322726


namespace copy_paper_purchase_solution_l3227_322790

/-- Represents the purchase of copy papers -/
structure CopyPaperPurchase where
  white : ℕ
  colored : ℕ

/-- The total cost of the purchase in yuan -/
def total_cost (p : CopyPaperPurchase) : ℕ := 80 * p.white + 180 * p.colored

/-- The relationship between white and colored paper quantities -/
def quantity_relation (p : CopyPaperPurchase) : Prop :=
  p.white = 5 * p.colored - 3

/-- The main theorem stating the solution to the problem -/
theorem copy_paper_purchase_solution :
  ∃ (p : CopyPaperPurchase),
    total_cost p = 2660 ∧
    quantity_relation p ∧
    p.white = 22 ∧
    p.colored = 5 := by
  sorry

end copy_paper_purchase_solution_l3227_322790


namespace batsman_average_increase_l3227_322702

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  totalRuns : ℕ
  innings : ℕ
  average : ℚ

/-- Calculates the increase in average after a new inning -/
def averageIncrease (initialStats : BatsmanStats) (newInningRuns : ℕ) : ℚ :=
  let newStats : BatsmanStats := {
    totalRuns := initialStats.totalRuns + newInningRuns,
    innings := initialStats.innings + 1,
    average := (initialStats.totalRuns + newInningRuns : ℚ) / (initialStats.innings + 1)
  }
  newStats.average - initialStats.average

/-- Theorem: The batsman's average increased by 3 runs per inning -/
theorem batsman_average_increase :
  ∀ (initialStats : BatsmanStats),
    initialStats.innings = 16 →
    averageIncrease initialStats 88 = 40 →
    averageIncrease initialStats 88 = 3 := by
  sorry

end batsman_average_increase_l3227_322702


namespace perpendicular_lines_parallel_l3227_322746

-- Define the plane and lines
variable (α : Set (Real × Real × Real))
variable (m n : Set (Real × Real × Real))

-- Define the perpendicular and parallel relations
def perpendicular (l : Set (Real × Real × Real)) (p : Set (Real × Real × Real)) : Prop := sorry
def parallel (l1 l2 : Set (Real × Real × Real)) : Prop := sorry

-- State the theorem
theorem perpendicular_lines_parallel (h1 : perpendicular m α) (h2 : perpendicular n α) :
  parallel m n := by sorry

end perpendicular_lines_parallel_l3227_322746


namespace student_group_aging_l3227_322780

/-- Represents a group of students with their average age and age variance -/
structure StudentGroup where
  averageAge : ℝ
  ageVariance : ℝ

/-- Function to calculate the new state of a StudentGroup after a given time -/
def ageStudentGroup (group : StudentGroup) (years : ℝ) : StudentGroup :=
  { averageAge := group.averageAge + years
    ageVariance := group.ageVariance }

theorem student_group_aging :
  let initialGroup : StudentGroup := { averageAge := 13, ageVariance := 3 }
  let yearsLater : ℝ := 2
  let finalGroup := ageStudentGroup initialGroup yearsLater
  finalGroup.averageAge = 15 ∧ finalGroup.ageVariance = 3 := by
  sorry


end student_group_aging_l3227_322780


namespace systematic_sampling_l3227_322778

/-- Systematic sampling problem -/
theorem systematic_sampling
  (total_employees : ℕ)
  (sample_size : ℕ)
  (fifth_group_number : ℕ)
  (h1 : total_employees = 200)
  (h2 : sample_size = 40)
  (h3 : fifth_group_number = 22) :
  let first_group_number := 2
  let group_difference := (fifth_group_number - first_group_number) / 4
  (9 * group_difference + first_group_number) = 47 := by
  sorry

end systematic_sampling_l3227_322778


namespace power_product_equals_l3227_322724

theorem power_product_equals : 3^5 * 4^5 = 248832 := by
  sorry

end power_product_equals_l3227_322724


namespace percentage_of_150_to_60_prove_percentage_l3227_322734

theorem percentage_of_150_to_60 : Real → Prop :=
  fun x => (150 / 60) * 100 = x

theorem prove_percentage :
  ∃ x, percentage_of_150_to_60 x ∧ x = 250 :=
by
  sorry

end percentage_of_150_to_60_prove_percentage_l3227_322734


namespace arithmetic_sequence_triangle_cos_identity_l3227_322786

theorem arithmetic_sequence_triangle_cos_identity (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- Arithmetic sequence condition
  2 * b = a + c ∧
  -- Side-angle relationships (law of sines)
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) →
  -- Theorem to prove
  5 * (Real.cos A) - 4 * (Real.cos A) * (Real.cos C) + 5 * (Real.cos C) = 4 := by
  sorry

end arithmetic_sequence_triangle_cos_identity_l3227_322786


namespace shorter_can_radius_l3227_322749

/-- Given two cylindrical cans with equal volume, where one can's height is twice 
    the other's and the taller can's radius is 10 units, the radius of the shorter 
    can is 10√2 units. -/
theorem shorter_can_radius (h : ℝ) (r : ℝ) : 
  h > 0 → -- height is positive
  π * (10^2) * (2*h) = π * r^2 * h → -- volumes are equal
  r = 10 * Real.sqrt 2 := by
sorry

end shorter_can_radius_l3227_322749


namespace least_sum_of_four_primes_l3227_322759

def is_sum_of_four_primes (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ : ℕ, 
    p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₄.Prime ∧
    p₁ > 10 ∧ p₂ > 10 ∧ p₃ > 10 ∧ p₄ > 10 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n = p₁ + p₂ + p₃ + p₄

theorem least_sum_of_four_primes : 
  (is_sum_of_four_primes 60) ∧ (∀ m < 60, ¬(is_sum_of_four_primes m)) :=
by sorry

end least_sum_of_four_primes_l3227_322759


namespace pizza_toppings_combinations_l3227_322755

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end pizza_toppings_combinations_l3227_322755


namespace sum_odd_numbers_l3227_322788

/-- Sum of first n natural numbers -/
def sum_n (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of first n odd numbers -/
def sum_odd (n : ℕ) : ℕ := n^2

/-- The 35th odd number -/
def last_odd : ℕ := 69

/-- Number of odd numbers up to 69 -/
def num_odds : ℕ := (last_odd + 1) / 2

theorem sum_odd_numbers :
  3 * (sum_odd num_odds) = 3675 :=
sorry

end sum_odd_numbers_l3227_322788


namespace distributive_property_implies_fraction_additivity_l3227_322757

theorem distributive_property_implies_fraction_additivity 
  {a b c : ℝ} (h1 : c ≠ 0) (h2 : (a + b) * c = a * c + b * c) :
  (a + b) / c = a / c + b / c :=
sorry

end distributive_property_implies_fraction_additivity_l3227_322757


namespace proportion_problem_l3227_322736

theorem proportion_problem (y : ℝ) : 
  (0.25 : ℝ) / 0.75 = y / 6 → y = 2 := by
  sorry

end proportion_problem_l3227_322736


namespace volume_of_rotated_composite_region_l3227_322720

/-- The volume of a solid formed by rotating a composite region about the y-axis -/
theorem volume_of_rotated_composite_region :
  let square_side : ℝ := 4
  let rectangle_width : ℝ := 5
  let rectangle_height : ℝ := 3
  let volume_square : ℝ := π * (square_side / 2)^2 * square_side
  let volume_rectangle : ℝ := π * (rectangle_height / 2)^2 * rectangle_width
  let total_volume : ℝ := volume_square + volume_rectangle
  total_volume = (109 * π) / 4 := by
  sorry

end volume_of_rotated_composite_region_l3227_322720


namespace extreme_values_cubic_l3227_322768

/-- Given a cubic function with extreme values at x=1 and x=2, prove that b=4 -/
theorem extreme_values_cubic (a b : ℝ) : 
  let f := fun x : ℝ => 2 * x^3 + 3 * a * x^2 + 3 * b * x
  let f' := fun x : ℝ => 6 * x^2 + 6 * a * x + 3 * b
  (f' 1 = 0 ∧ f' 2 = 0) → b = 4 := by
sorry

end extreme_values_cubic_l3227_322768


namespace total_amount_192_rupees_l3227_322787

/-- Represents the denominations of rupee notes -/
inductive Denomination
  | One
  | Five
  | Ten

/-- Calculates the value of a single note of a given denomination -/
def noteValue (d : Denomination) : Nat :=
  match d with
  | Denomination.One => 1
  | Denomination.Five => 5
  | Denomination.Ten => 10

/-- Represents the collection of notes -/
structure NoteCollection where
  totalNotes : Nat
  denominations : List Denomination
  equalDenominations : List.length denominations = 3
  equalDistribution : totalNotes % (List.length denominations) = 0

/-- Theorem stating that a collection of 36 notes equally distributed among
    one-rupee, five-rupee, and ten-rupee denominations totals 192 rupees -/
theorem total_amount_192_rupees (nc : NoteCollection)
    (h1 : nc.totalNotes = 36)
    (h2 : nc.denominations = [Denomination.One, Denomination.Five, Denomination.Ten]) :
    (nc.totalNotes / 3) * (noteValue Denomination.One +
                           noteValue Denomination.Five +
                           noteValue Denomination.Ten) = 192 := by
  sorry

end total_amount_192_rupees_l3227_322787


namespace jelly_bean_distribution_l3227_322713

theorem jelly_bean_distribution (total_jelly_beans : ℕ) (leftover_jelly_beans : ℕ) : 
  total_jelly_beans = 726 →
  leftover_jelly_beans = 4 →
  ∃ (girls : ℕ),
    let boys := girls + 3
    let students := girls + boys
    let distributed_jelly_beans := boys * boys + girls * (2 * girls + 1)
    distributed_jelly_beans = total_jelly_beans - leftover_jelly_beans →
    students = 31 := by
  sorry

end jelly_bean_distribution_l3227_322713


namespace athlete_B_most_stable_l3227_322732

-- Define the athletes
inductive Athlete : Type
  | A : Athlete
  | B : Athlete
  | C : Athlete

-- Define the variance for each athlete
def variance (a : Athlete) : ℝ :=
  match a with
  | Athlete.A => 0.78
  | Athlete.B => 0.2
  | Athlete.C => 1.28

-- Define the concept of most stable performance
def most_stable (a : Athlete) : Prop :=
  ∀ b : Athlete, variance a ≤ variance b

-- Theorem statement
theorem athlete_B_most_stable :
  most_stable Athlete.B :=
sorry

end athlete_B_most_stable_l3227_322732


namespace inverse_f_sum_l3227_322741

-- Define the function f
def f (x : ℝ) : ℝ := x^2 * abs x

-- State the theorem
theorem inverse_f_sum : (∃ y₁ y₂ : ℝ, f y₁ = 8 ∧ f y₂ = -27 ∧ y₁ + y₂ = -1) := by
  sorry

end inverse_f_sum_l3227_322741


namespace complement_intersection_equals_set_l3227_322752

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem complement_intersection_equals_set :
  (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end complement_intersection_equals_set_l3227_322752


namespace inverse_of_B_cubed_l3227_322774

def B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; -2, -3]

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = B_inv) : 
  (B^3)⁻¹ = B⁻¹ := by
  sorry

end inverse_of_B_cubed_l3227_322774


namespace perpendicular_vectors_tan_theta_l3227_322754

theorem perpendicular_vectors_tan_theta :
  ∀ θ : ℝ,
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (Real.sqrt 3, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  Real.tan θ = -Real.sqrt 3 := by
sorry

end perpendicular_vectors_tan_theta_l3227_322754


namespace train_speed_l3227_322793

/-- The speed of a train given its length and time to cross an electric pole. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 3500) (h2 : time = 80) :
  length / time = 43.75 := by
  sorry

end train_speed_l3227_322793


namespace quadratic_max_abs_value_bound_l3227_322750

/-- For any quadratic function f(x) = x^2 + px + q, 
    the maximum absolute value of f(1), f(2), and f(3) 
    is greater than or equal to 1/2. -/
theorem quadratic_max_abs_value_bound (p q : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + p*x + q
  ∃ i : Fin 3, |f (i.val + 1)| ≥ 1/2 := by
  sorry

end quadratic_max_abs_value_bound_l3227_322750


namespace total_spent_is_124_l3227_322771

/-- The total amount spent on entertainment and additional expenses -/
def total_spent (computer_game_cost movie_ticket_cost num_tickets snack_cost transportation_cost num_trips : ℕ) : ℕ :=
  computer_game_cost + 
  movie_ticket_cost * num_tickets + 
  snack_cost + 
  transportation_cost * num_trips

/-- Theorem stating that the total amount spent is $124 given the specific costs -/
theorem total_spent_is_124 :
  total_spent 66 12 3 7 5 3 = 124 := by
  sorry

end total_spent_is_124_l3227_322771


namespace figure_to_square_possible_l3227_322737

/-- A figure composed of unit squares -/
structure UnitSquareFigure where
  area : ℕ

/-- A part of a figure after cutting -/
structure FigurePart where
  area : ℕ

/-- Represents a square -/
structure Square where
  side_length : ℕ

/-- Theorem stating that a UnitSquareFigure can be cut into three parts to form a square -/
theorem figure_to_square_possible (fig : UnitSquareFigure) 
  (h : ∃ n : ℕ, n * n = fig.area) : 
  ∃ (part1 part2 part3 : FigurePart) (sq : Square),
    part1.area + part2.area + part3.area = fig.area ∧
    sq.side_length * sq.side_length = fig.area :=
sorry

end figure_to_square_possible_l3227_322737


namespace cubic_extrema_l3227_322729

/-- A cubic function f(x) = ax³ + bx² where a > 0 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2

theorem cubic_extrema (a b : ℝ) (h₁ : a > 0) :
  (∀ x, f a b x ≤ f a b 0) ∧  -- maximum at x = 0
  (∀ x, f a b x ≥ f a b (1/3)) -- minimum at x = 1/3
  → a + 2*b = 0 := by sorry

end cubic_extrema_l3227_322729


namespace article_cost_l3227_322742

/-- Proves that the cost of an article is 120, given the selling prices and gain difference --/
theorem article_cost (sp1 sp2 : ℕ) (gain_diff : ℚ) :
  sp1 = 380 →
  sp2 = 420 →
  gain_diff = 8 / 100 →
  sp2 - (sp1 - (sp2 - sp1)) = 120 := by
sorry

end article_cost_l3227_322742


namespace gcf_of_270_and_180_l3227_322779

theorem gcf_of_270_and_180 : Nat.gcd 270 180 = 90 := by
  sorry

end gcf_of_270_and_180_l3227_322779


namespace sqrt_sum_squares_eq_product_l3227_322748

theorem sqrt_sum_squares_eq_product (a b c : ℝ) : 
  (Real.sqrt (a^2 + b^2) = a * b) ∧ (a + b + c = 0) → (a = 0 ∧ b = 0 ∧ c = 0) := by
  sorry

end sqrt_sum_squares_eq_product_l3227_322748


namespace solutions_of_quadratic_equation_l3227_322761

theorem solutions_of_quadratic_equation :
  ∀ x : ℝ, x * (2 * x + 1) = 0 ↔ x = 0 ∨ x = -1/2 := by sorry

end solutions_of_quadratic_equation_l3227_322761


namespace rick_ironing_rate_l3227_322719

/-- The number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := sorry

/-- The number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := 3

/-- The number of hours Rick spent ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- The number of hours Rick spent ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- The total number of pieces of clothing Rick ironed -/
def total_pieces : ℕ := 27

theorem rick_ironing_rate : shirts_per_hour = 4 := by
  sorry

end rick_ironing_rate_l3227_322719


namespace solution_set_for_a_3_f_geq_1_iff_a_leq_1_or_geq_3_l3227_322769

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := |x| + 2*|x + 2 - a|

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := g a (x - 2)

-- Theorem for part (1)
theorem solution_set_for_a_3 :
  {x : ℝ | g 3 x ≤ 4} = Set.Icc (-2/3) 2 := by sorry

-- Theorem for part (2)
theorem f_geq_1_iff_a_leq_1_or_geq_3 :
  (∀ x, f a x ≥ 1) ↔ (a ≤ 1 ∨ a ≥ 3) := by sorry

end solution_set_for_a_3_f_geq_1_iff_a_leq_1_or_geq_3_l3227_322769


namespace cosine_C_value_l3227_322777

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- State the theorem
theorem cosine_C_value (t : Triangle) 
  (h1 : t.c = 2 * t.a)  -- Given condition: c = 2a
  (h2 : Real.sin t.A / Real.sin t.B = 2/3)  -- Given condition: sin A / sin B = 2/3
  : Real.cos t.C = -1/4 := by
  sorry

end cosine_C_value_l3227_322777


namespace component_unqualified_l3227_322776

/-- A component is qualified if its diameter is within the specified range. -/
def IsQualified (measured : ℝ) (nominal : ℝ) (tolerance : ℝ) : Prop :=
  nominal - tolerance ≤ measured ∧ measured ≤ nominal + tolerance

/-- The component is unqualified if it's not qualified. -/
def IsUnqualified (measured : ℝ) (nominal : ℝ) (tolerance : ℝ) : Prop :=
  ¬(IsQualified measured nominal tolerance)

theorem component_unqualified (measured : ℝ) (h : measured = 19.9) :
  IsUnqualified measured 20 0.02 := by
  sorry

#check component_unqualified

end component_unqualified_l3227_322776


namespace work_completion_time_l3227_322767

theorem work_completion_time 
  (x : ℝ) 
  (hx : x > 0) 
  (h_combined : 1/x + 1/8 = 3/16) : x = 16 := by
  sorry

end work_completion_time_l3227_322767


namespace license_plate_theorem_l3227_322744

def license_plate_combinations : ℕ :=
  let alphabet_size : ℕ := 26
  let plate_length : ℕ := 5
  let repeated_letters : ℕ := 2
  let non_zero_digits : ℕ := 9

  let choose_repeated_letters := Nat.choose alphabet_size repeated_letters
  let assign_first_repeat := Nat.choose plate_length repeated_letters
  let assign_second_repeat := Nat.choose (plate_length - repeated_letters) repeated_letters
  let remaining_letter_choices := alphabet_size - repeated_letters
  
  choose_repeated_letters * assign_first_repeat * assign_second_repeat * remaining_letter_choices * non_zero_digits

theorem license_plate_theorem : license_plate_combinations = 210600 := by
  sorry

end license_plate_theorem_l3227_322744


namespace function_divisibility_condition_l3227_322785

theorem function_divisibility_condition (f : ℕ+ → ℕ+) :
  (∀ n m : ℕ+, (n + f m) ∣ (f n + n * f m)) →
  (∀ n : ℕ+, f n = n ^ 2 ∨ f n = 1) :=
by sorry

end function_divisibility_condition_l3227_322785


namespace max_profit_at_max_price_l3227_322782

-- Define the problem parameters
def raw_material_price : ℝ := 30
def min_selling_price : ℝ := 30
def max_selling_price : ℝ := 60
def additional_cost : ℝ := 450

-- Define the sales volume function
def sales_volume (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - raw_material_price) * sales_volume x - additional_cost

-- State the theorem
theorem max_profit_at_max_price :
  ∀ x ∈ Set.Icc min_selling_price max_selling_price,
    profit x ≤ profit max_selling_price ∧
    profit max_selling_price = 1950 :=
by sorry

end max_profit_at_max_price_l3227_322782


namespace unique_line_through_5_2_l3227_322784

/-- A line in the xy-plane is represented by its x and y intercepts -/
structure Line where
  x_intercept : ℕ
  y_intercept : ℕ

/-- Check if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Check if a number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- Check if a line passes through the point (5,2) -/
def passes_through_5_2 (l : Line) : Prop :=
  5 / l.x_intercept + 2 / l.y_intercept = 1

/-- The main theorem to be proved -/
theorem unique_line_through_5_2 : 
  ∃! l : Line, 
    is_prime l.x_intercept ∧ 
    is_power_of_two l.y_intercept ∧ 
    passes_through_5_2 l :=
sorry

end unique_line_through_5_2_l3227_322784


namespace stock_price_increase_l3227_322773

/-- Given a stock price that decreased by 8% in the first year and had a net percentage change of 1.20% over two years, the percentage increase in the second year was 10%. -/
theorem stock_price_increase (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_decrease := initial_price * (1 - 0.08)
  let final_price := initial_price * (1 + 0.012)
  let increase_percentage := (final_price / price_after_decrease - 1) * 100
  increase_percentage = 10 := by sorry

end stock_price_increase_l3227_322773


namespace train_length_l3227_322772

/-- Calculate the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (time : ℝ) : 
  speed = 72 → platform_length = 230 → time = 26 → 
  (speed * 1000 / 3600) * time - platform_length = 290 := by
  sorry

end train_length_l3227_322772


namespace mike_baseball_cards_l3227_322700

/-- 
Given that Mike initially has 87 baseball cards and Sam buys 13 of them,
prove that Mike will have 74 baseball cards remaining.
-/
theorem mike_baseball_cards (initial_cards : ℕ) (bought_cards : ℕ) (remaining_cards : ℕ) :
  initial_cards = 87 →
  bought_cards = 13 →
  remaining_cards = initial_cards - bought_cards →
  remaining_cards = 74 := by
sorry

end mike_baseball_cards_l3227_322700


namespace additive_multiplicative_inverses_l3227_322760

theorem additive_multiplicative_inverses 
  (x y p q : ℝ) 
  (h1 : x + y = 0)  -- x and y are additive inverses
  (h2 : p * q = 1)  -- p and q are multiplicative inverses
  : (x + y) - 2 * p * q = -2 := by
sorry

end additive_multiplicative_inverses_l3227_322760


namespace ten_people_round_table_with_pair_l3227_322716

/-- The number of ways to arrange n people around a round table -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange n people around a round table
    when two specific people must sit next to each other -/
def roundTableArrangementsWithPair (n : ℕ) : ℕ :=
  2 * roundTableArrangements (n - 1)

/-- Theorem: There are 80,640 ways to arrange 10 people around a round table
    when two specific people must sit next to each other -/
theorem ten_people_round_table_with_pair :
  roundTableArrangementsWithPair 10 = 80640 := by
  sorry

end ten_people_round_table_with_pair_l3227_322716


namespace jerrys_age_l3227_322735

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 18 →
  mickey_age = 2 * jerry_age - 6 →
  jerry_age = 12 :=
by sorry

end jerrys_age_l3227_322735


namespace log_difference_equals_six_l3227_322725

theorem log_difference_equals_six : 
  ∀ (log₄ : ℝ → ℝ),
  (log₄ 256 = 4) →
  (log₄ (1/16) = -2) →
  (log₄ 256 - log₄ (1/16) = 6) :=
by
  sorry

end log_difference_equals_six_l3227_322725


namespace simple_interest_calculation_l3227_322745

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 4016.25)
  (h2 : rate = 12)
  (h3 : time = 5) :
  ∃ principal : ℝ,
    interest = principal * rate * time / 100 ∧
    principal = 6693.75 := by
  sorry

end simple_interest_calculation_l3227_322745


namespace number_of_classes_for_histogram_l3227_322708

theorem number_of_classes_for_histogram (tallest_height shortest_height class_interval : ℝ)
  (h1 : tallest_height = 186)
  (h2 : shortest_height = 154)
  (h3 : class_interval = 5)
  : Int.ceil ((tallest_height - shortest_height) / class_interval) = 7 := by
  sorry

end number_of_classes_for_histogram_l3227_322708


namespace trace_bag_weight_l3227_322783

/-- Given:
  - Trace has 5 shopping bags
  - Trace's 5 bags weigh the same as Gordon's 2 bags
  - One of Gordon's bags weighs 3 pounds
  - The other of Gordon's bags weighs 7 pounds
  - All of Trace's bags weigh the same amount
Prove that one of Trace's bags weighs 2 pounds -/
theorem trace_bag_weight :
  ∀ (trace_bag_count : ℕ) 
    (gordon_bag1_weight gordon_bag2_weight : ℕ)
    (trace_total_weight : ℕ),
  trace_bag_count = 5 →
  gordon_bag1_weight = 3 →
  gordon_bag2_weight = 7 →
  trace_total_weight = gordon_bag1_weight + gordon_bag2_weight →
  ∃ (trace_single_bag_weight : ℕ),
    trace_single_bag_weight * trace_bag_count = trace_total_weight ∧
    trace_single_bag_weight = 2 :=
by sorry

end trace_bag_weight_l3227_322783


namespace correct_sunset_time_l3227_322795

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds a duration to a time -/
def addDuration (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes
  let additionalHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  let newHours := (t.hours + d.hours + additionalHours) % 24
  { hours := newHours, minutes := newMinutes }

theorem correct_sunset_time :
  let sunrise : Time := { hours := 6, minutes := 32 }
  let daylight : Duration := { hours := 11, minutes := 35 }
  let sunset := addDuration sunrise daylight
  sunset = { hours := 18, minutes := 7 } :=
by sorry

end correct_sunset_time_l3227_322795


namespace double_length_isosceles_triangle_base_length_l3227_322794

/-- A triangle with one side length being twice the length of another side is called a "double-length triangle". -/
def is_double_length_triangle (a b c : ℝ) : Prop :=
  a = 2*b ∨ a = 2*c ∨ b = 2*a ∨ b = 2*c ∨ c = 2*a ∨ c = 2*b

/-- An isosceles triangle is a triangle with at least two equal sides. -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ a = c ∨ b = c

theorem double_length_isosceles_triangle_base_length 
  (a b c : ℝ) 
  (h_isosceles : is_isosceles_triangle a b c) 
  (h_double_length : is_double_length_triangle a b c) 
  (h_side_length : a = 6) : 
  (a = b ∧ a = 2*c ∧ c = 3) ∨ (a = c ∧ a = 2*b ∧ b = 3) :=
sorry

end double_length_isosceles_triangle_base_length_l3227_322794


namespace complement_of_A_in_U_l3227_322798

def A : Set ℤ := {0, 1, 2}

def U : Set ℤ := {z | ∃ x y, x ∈ A ∧ y ∈ A ∧ z = x - y}

theorem complement_of_A_in_U :
  (A : Set ℤ)ᶜ ∩ U = {-2, -1} := by sorry

end complement_of_A_in_U_l3227_322798


namespace sequence_sum_exp_l3227_322764

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = ln(1 + 1/n),
    prove that e^(a_7 + a_8 + a_9) = 20/21 -/
theorem sequence_sum_exp (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = Real.log (1 + 1 / n)) :
  Real.exp (a 7 + a 8 + a 9) = 20 / 21 := by
  sorry

end sequence_sum_exp_l3227_322764


namespace sum_of_absolute_roots_l3227_322753

theorem sum_of_absolute_roots (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℝ, x^3 - 2011*x + m = (x - a) * (x - b) * (x - c)) →
  |a| + |b| + |c| = 98 := by
  sorry

end sum_of_absolute_roots_l3227_322753


namespace floor_ceil_sum_l3227_322703

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.1 : ℝ)⌉ = 31 := by sorry

end floor_ceil_sum_l3227_322703


namespace triangle_median_similarity_exists_l3227_322792

/-- 
Given a triangle with sides a, b, c (where a < b < c), we define the following:
1) The triangle formed by the medians is similar to the original triangle.
2) The relationship between sides and medians is given by:
   4sa² = -a² + 2b² + 2c²
   4sb² = 2a² - b² + 2c²
   4sc² = 2a² + 2b² - c²
   where sa, sb, sc are the medians opposite to sides a, b, c respectively.
3) The sides satisfy the equation: b² = (a² + c²) / 2

This theorem states that there exists a triplet of natural numbers (a, b, c) 
that satisfies all these conditions, with a < b < c.
-/
theorem triangle_median_similarity_exists : 
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ 
  (b * b : ℚ) = (a * a + c * c) / 2 ∧
  (∃ (sa sb sc : ℚ), 
    4 * sa * sa = -a * a + 2 * b * b + 2 * c * c ∧
    4 * sb * sb = 2 * a * a - b * b + 2 * c * c ∧
    4 * sc * sc = 2 * a * a + 2 * b * b - c * c ∧
    (a : ℚ) / sc = (b : ℚ) / sb ∧ (b : ℚ) / sb = (c : ℚ) / sa) :=
by sorry

end triangle_median_similarity_exists_l3227_322792


namespace two_digit_number_property_l3227_322711

theorem two_digit_number_property (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 →
  x^2 + y^2 = 10*x + x*y →
  10*x + y - 36 = 10*y + x →
  10*x + y = 48 ∨ 10*x + y = 37 := by
sorry

end two_digit_number_property_l3227_322711


namespace grapes_purchased_l3227_322758

theorem grapes_purchased (grape_price mango_price mango_weight total_paid : ℕ) 
  (h1 : grape_price = 80)
  (h2 : mango_price = 55)
  (h3 : mango_weight = 9)
  (h4 : total_paid = 1135)
  : ∃ (grape_weight : ℕ), grape_weight * grape_price + mango_weight * mango_price = total_paid ∧ grape_weight = 8 := by
  sorry

end grapes_purchased_l3227_322758


namespace list_number_fraction_l3227_322715

theorem list_number_fraction (list : List ℝ) (n : ℝ) :
  list.length = 21 →
  n ∈ list →
  list.Pairwise (·≠·) →
  n = 4 * ((list.sum - n) / 20) →
  n = (1 / 6) * list.sum :=
by sorry

end list_number_fraction_l3227_322715


namespace blueberry_lake_fish_count_l3227_322791

/-- The number of fish associated with each white duck -/
def white_duck_fish : ℕ := 8

/-- The number of fish associated with each black duck -/
def black_duck_fish : ℕ := 15

/-- The number of fish associated with each multicolor duck -/
def multicolor_duck_fish : ℕ := 20

/-- The number of fish associated with each golden duck -/
def golden_duck_fish : ℕ := 25

/-- The number of fish associated with each teal duck -/
def teal_duck_fish : ℕ := 30

/-- The number of white ducks in Blueberry Lake -/
def white_ducks : ℕ := 10

/-- The number of black ducks in Blueberry Lake -/
def black_ducks : ℕ := 12

/-- The number of multicolor ducks in Blueberry Lake -/
def multicolor_ducks : ℕ := 8

/-- The number of golden ducks in Blueberry Lake -/
def golden_ducks : ℕ := 6

/-- The number of teal ducks in Blueberry Lake -/
def teal_ducks : ℕ := 14

/-- The total number of fish in Blueberry Lake -/
def total_fish : ℕ := white_duck_fish * white_ducks + 
                      black_duck_fish * black_ducks + 
                      multicolor_duck_fish * multicolor_ducks + 
                      golden_duck_fish * golden_ducks + 
                      teal_duck_fish * teal_ducks

theorem blueberry_lake_fish_count : total_fish = 990 := by
  sorry

end blueberry_lake_fish_count_l3227_322791


namespace cheolsu_weight_l3227_322730

/-- Proves that Cheolsu's weight is 36 kg given the conditions stated in the problem -/
theorem cheolsu_weight :
  ∀ (cheolsu_weight mother_weight : ℝ),
    cheolsu_weight = (2 / 3) * mother_weight →
    cheolsu_weight + 72 = 2 * mother_weight →
    cheolsu_weight = 36 := by
  sorry

end cheolsu_weight_l3227_322730


namespace antonios_meatballs_l3227_322722

/-- Antonio's meatball problem -/
theorem antonios_meatballs (recipe_amount : ℚ) (family_members : ℕ) (total_hamburger : ℚ) : 
  recipe_amount = 1/8 →
  family_members = 8 →
  total_hamburger = 4 →
  (total_hamburger / recipe_amount) / family_members = 4 :=
by sorry

end antonios_meatballs_l3227_322722


namespace orange_harvest_theorem_l3227_322721

-- Define the number of days for the harvest
def harvest_days : ℕ := 4

-- Define the total number of sacks harvested
def total_sacks : ℕ := 56

-- Define the function to calculate sacks per day
def sacks_per_day (total : ℕ) (days : ℕ) : ℕ := total / days

-- Theorem statement
theorem orange_harvest_theorem : 
  sacks_per_day total_sacks harvest_days = 14 := by
  sorry

end orange_harvest_theorem_l3227_322721


namespace basketball_shot_expectation_l3227_322709

theorem basketball_shot_expectation (a b : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (h_exp : 3 * a + 2 * b = 2) :
  (∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 3 * x + 2 * y = 2 → 
    2 / a + 1 / (3 * b) ≤ 2 / x + 1 / (3 * y)) ∧
  2 / a + 1 / (3 * b) = 16 / 3 := by
  sorry

end basketball_shot_expectation_l3227_322709
