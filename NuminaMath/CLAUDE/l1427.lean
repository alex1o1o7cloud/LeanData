import Mathlib

namespace factorization_quadratic_l1427_142716

theorem factorization_quadratic (a x y : ℝ) : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) := by
  sorry

end factorization_quadratic_l1427_142716


namespace ultra_marathon_average_time_l1427_142711

/-- Calculates the average time per mile given the total distance and time -/
def averageTimePerMile (distance : ℕ) (hours : ℕ) (minutes : ℕ) : ℚ :=
  let totalMinutes : ℕ := hours * 60 + minutes
  (totalMinutes : ℚ) / distance

theorem ultra_marathon_average_time :
  averageTimePerMile 32 4 52 = 9.125 := by
  sorry

end ultra_marathon_average_time_l1427_142711


namespace necessary_not_sufficient_condition_l1427_142778

theorem necessary_not_sufficient_condition (a b : ℝ) (h : a > b) :
  (∃ c : ℝ, c ≥ 0 ∧ ¬(a * c > b * c)) ∧
  (∀ c : ℝ, a * c > b * c → c ≥ 0) := by
  sorry

end necessary_not_sufficient_condition_l1427_142778


namespace sports_club_total_members_l1427_142781

/-- The number of members in a sports club -/
def sports_club_members (badminton tennis both neither : ℕ) : ℕ :=
  badminton + tennis - both + neither

/-- Theorem: The sports club has 42 members -/
theorem sports_club_total_members :
  sports_club_members 20 23 7 6 = 42 := by
  sorry

end sports_club_total_members_l1427_142781


namespace expected_score_is_correct_l1427_142740

/-- The expected score for a round in the basketball shooting game. -/
def expected_score : ℝ := 6

/-- The probability of making a shot. -/
def shot_probability : ℝ := 0.5

/-- The score for making the first shot. -/
def first_shot_score : ℕ := 8

/-- The score for making the second shot (after missing the first). -/
def second_shot_score : ℕ := 6

/-- The score for making the third shot (after missing the first two). -/
def third_shot_score : ℕ := 4

/-- The score for missing all three shots. -/
def miss_all_score : ℕ := 0

/-- Theorem stating that the expected score is correct given the game rules. -/
theorem expected_score_is_correct :
  expected_score = 
    shot_probability * first_shot_score +
    (1 - shot_probability) * shot_probability * second_shot_score +
    (1 - shot_probability) * (1 - shot_probability) * shot_probability * third_shot_score +
    (1 - shot_probability) * (1 - shot_probability) * (1 - shot_probability) * miss_all_score :=
by sorry

end expected_score_is_correct_l1427_142740


namespace stock_price_change_l1427_142706

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let day1_price := initial_price * (1 - 0.15)
  let day2_price := day1_price * (1 + 0.25)
  (day2_price - initial_price) / initial_price = 0.0625 := by
sorry

end stock_price_change_l1427_142706


namespace train_speed_fraction_l1427_142761

theorem train_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 2 → delay = 1/3 → 
  (2 : ℝ) / (2 + delay) = 6/7 := by sorry

end train_speed_fraction_l1427_142761


namespace league_games_l1427_142700

theorem league_games (n : ℕ) (h1 : n = 8) : (n.choose 2) = 28 := by
  sorry

end league_games_l1427_142700


namespace sqrt_equation_solution_l1427_142760

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (4 * x + 7) / Real.sqrt (8 * x + 10) = Real.sqrt 7 / 4) → x = -21/4 := by
  sorry

end sqrt_equation_solution_l1427_142760


namespace inequality_system_solution_range_l1427_142703

theorem inequality_system_solution_range (k : ℝ) : 
  (∀ x : ℤ, (x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+5)*x + 5*k < 0) ↔ x = -2) →
  -3 ≤ k ∧ k < 2 := by
sorry

end inequality_system_solution_range_l1427_142703


namespace hyperbola_eccentricity_range_l1427_142720

-- Define the hyperbola
def Hyperbola (a b : ℝ) := {(x, y) : ℝ × ℝ | y^2 / a^2 - x^2 / b^2 = 1}

-- Define the foci
def Foci (a b : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

-- Define a point on the hyperbola
def PointOnHyperbola (a b : ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def Distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the eccentricity
def Eccentricity (a b : ℝ) : ℝ := sorry

theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let (f1x, f1y, f2x, f2y) := Foci a b
  let p := PointOnHyperbola a b
  let d1 := Distance p (f1x, f1y)
  let d2 := Distance p (f2x, f2y)
  d1 = 3 * d2 →
  let e := Eccentricity a b
  1 < e ∧ e ≤ 2 := by sorry

end hyperbola_eccentricity_range_l1427_142720


namespace unselected_probability_l1427_142743

/-- The type representing a selection of five consecutive integers from a circle of 10 numbers -/
def Selection := Fin 10

/-- The type representing the choices of four people -/
def Choices := Fin 4 → Selection

/-- The probability that there exists a number not selected by any of the four people -/
def probability_unselected (choices : Choices) : ℚ :=
  sorry

/-- The main theorem stating the probability of an unselected number -/
theorem unselected_probability :
  ∃ (p : ℚ), (∀ (choices : Choices), probability_unselected choices = p) ∧ 10000 * p = 3690 :=
sorry

end unselected_probability_l1427_142743


namespace min_value_m_exists_l1427_142768

theorem min_value_m_exists (m : ℝ) : 
  (∃ x₀ : ℝ, |x₀ + 1| + |x₀ - 1| ≤ m) ↔ m ≥ 2 :=
by sorry

end min_value_m_exists_l1427_142768


namespace min_subsets_to_guess_l1427_142726

/-- The set of possible choices for player A -/
def S : Set Nat := Finset.range 1001

/-- The condition that ensures B can always guess correctly -/
def can_guess (k₁ k₂ k₃ : Nat) : Prop :=
  (k₁ + 1) * (k₂ + 1) * (k₃ + 1) ≥ 1001

/-- The sum of subsets chosen by B -/
def total_subsets (k₁ k₂ k₃ : Nat) : Nat :=
  k₁ + k₂ + k₃

/-- The theorem stating that 28 is the minimum value -/
theorem min_subsets_to_guess :
  ∃ k₁ k₂ k₃ : Nat,
    can_guess k₁ k₂ k₃ ∧
    total_subsets k₁ k₂ k₃ = 28 ∧
    ∀ k₁' k₂' k₃' : Nat,
      can_guess k₁' k₂' k₃' →
      total_subsets k₁' k₂' k₃' ≥ 28 :=
sorry

end min_subsets_to_guess_l1427_142726


namespace expression_value_l1427_142788

theorem expression_value (x : ℝ) (h : x^2 + 3*x - 1 = 0) : 
  (x - 3)^2 - (2*x + 1)*(2*x - 1) - 3*x = 7 := by
  sorry

end expression_value_l1427_142788


namespace perimeter_of_specific_cut_pentagon_l1427_142728

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents a pentagon formed by cutting a smaller equilateral triangle from a larger one -/
structure CutPentagon where
  originalTriangle : EquilateralTriangle
  cutTriangle : EquilateralTriangle

/-- Calculates the perimeter of the pentagon formed by cutting a smaller equilateral triangle
    from a corner of a larger equilateral triangle -/
def perimeterOfCutPentagon (p : CutPentagon) : ℝ :=
  p.originalTriangle.sideLength + p.originalTriangle.sideLength +
  (p.originalTriangle.sideLength - p.cutTriangle.sideLength) +
  p.cutTriangle.sideLength + p.cutTriangle.sideLength

/-- Theorem stating that the perimeter of the specific cut pentagon is 14 units -/
theorem perimeter_of_specific_cut_pentagon :
  let largeTriangle : EquilateralTriangle := { sideLength := 5 }
  let smallTriangle : EquilateralTriangle := { sideLength := 2 }
  let cutPentagon : CutPentagon := { originalTriangle := largeTriangle, cutTriangle := smallTriangle }
  perimeterOfCutPentagon cutPentagon = 14 := by
  sorry

end perimeter_of_specific_cut_pentagon_l1427_142728


namespace set_union_problem_l1427_142730

theorem set_union_problem (a b : ℝ) :
  let M : Set ℝ := {a, b}
  let N : Set ℝ := {a + 1, 3}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
sorry

end set_union_problem_l1427_142730


namespace function_composition_l1427_142774

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_composition (h : ∀ x, f (3*x + 2) = 9*x + 8) : 
  ∀ x, f x = 3*x + 2 := by
  sorry

end function_composition_l1427_142774


namespace principal_square_root_nine_sixteenths_l1427_142755

theorem principal_square_root_nine_sixteenths (x : ℝ) : x = Real.sqrt (9 / 16) → x = 3 / 4 := by
  sorry

end principal_square_root_nine_sixteenths_l1427_142755


namespace fraction_subtraction_l1427_142775

theorem fraction_subtraction : 
  (2 + 6 + 8) / (1 + 2 + 3) - (1 + 2 + 3) / (2 + 6 + 8) = 55 / 24 := by
  sorry

end fraction_subtraction_l1427_142775


namespace hyperbola_foci_distance_l1427_142793

/-- The distance between the foci of the hyperbola x^2 - 6x - 4y^2 - 8y = 27 is 4√10 -/
theorem hyperbola_foci_distance :
  ∃ (a b c : ℝ),
    (∀ x y : ℝ, x^2 - 6*x - 4*y^2 - 8*y = 27 ↔ (x - 3)^2 / a^2 - (y + 1)^2 / b^2 = 1) ∧
    c^2 = a^2 + b^2 ∧
    2*c = 4 * Real.sqrt 10 :=
by sorry

end hyperbola_foci_distance_l1427_142793


namespace merchant_articles_count_l1427_142701

theorem merchant_articles_count (N : ℕ) (CP SP : ℝ) : 
  N > 0 → 
  CP > 0 →
  N * CP = 15 * SP → 
  SP = CP * (1 + 33.33 / 100) → 
  N = 20 := by
sorry

end merchant_articles_count_l1427_142701


namespace five_digit_divisible_by_165_l1427_142767

/-- Represents a 5-digit number in the form XX4XY -/
structure FiveDigitNumber where
  x : ℕ
  y : ℕ
  is_valid : x < 10 ∧ y < 10

/-- The 5-digit number as an integer -/
def FiveDigitNumber.to_int (n : FiveDigitNumber) : ℤ :=
  ↑(n.x * 10000 + n.x * 1000 + 400 + n.x * 10 + n.y)

theorem five_digit_divisible_by_165 (n : FiveDigitNumber) :
  n.to_int % 165 = 0 → n.x + n.y = 14 := by
  sorry


end five_digit_divisible_by_165_l1427_142767


namespace increase_decrease_theorem_l1427_142732

theorem increase_decrease_theorem (k r s N : ℝ) 
  (hk : k > 0) (hr : r > 0) (hs : s > 0) (hN : N > 0) (hr_bound : r < 80) :
  N * (1 + k / 100) * (1 - r / 100) + 10 * s > N ↔ k > 100 * r / (100 - r) := by
sorry

end increase_decrease_theorem_l1427_142732


namespace card_difference_l1427_142765

/-- Given a total of 500 cards divided in the ratio of 11:9, prove that the difference between the larger share and the smaller share is 50 cards. -/
theorem card_difference (total : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (h1 : total = 500) (h2 : ratio_a = 11) (h3 : ratio_b = 9) : 
  (total * ratio_a) / (ratio_a + ratio_b) - (total * ratio_b) / (ratio_a + ratio_b) = 50 := by
sorry

end card_difference_l1427_142765


namespace fifth_term_is_14_l1427_142724

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_second_third : a 2 + a 3 = 13

/-- The fifth term of the arithmetic sequence equals 14 -/
theorem fifth_term_is_14 (seq : ArithmeticSequence) : seq.a 5 = 14 := by
  sorry

end fifth_term_is_14_l1427_142724


namespace distance_between_points_l1427_142756

theorem distance_between_points (x : ℝ) :
  (x - 2)^2 + (5 - 5)^2 = 5^2 → x = -3 ∨ x = 7 := by
  sorry

end distance_between_points_l1427_142756


namespace sum_of_ages_l1427_142753

def father_age : ℕ := 48
def son_age : ℕ := 27

theorem sum_of_ages : father_age + son_age = 75 := by
  sorry

end sum_of_ages_l1427_142753


namespace cubic_difference_division_l1427_142710

theorem cubic_difference_division (a b : ℝ) (ha : a = 6) (hb : b = 3) :
  (a^3 - b^3) / (a^2 + a*b + b^2) = 3 := by
  sorry

end cubic_difference_division_l1427_142710


namespace metal_disc_weight_expectation_l1427_142702

/-- The nominal radius of a metal disc in meters -/
def nominal_radius : ℝ := 0.5

/-- The standard deviation of the radius in meters -/
def radius_std_dev : ℝ := 0.01

/-- The weight of a disc with exactly 1 m diameter in kilograms -/
def nominal_weight : ℝ := 100

/-- The number of discs in the stack -/
def num_discs : ℕ := 100

/-- The expected weight of the stack of discs in kilograms -/
def expected_stack_weight : ℝ := 10004

theorem metal_disc_weight_expectation :
  let expected_area := π * (nominal_radius^2 + radius_std_dev^2)
  let expected_single_weight := nominal_weight * expected_area / (π * nominal_radius^2)
  expected_single_weight * num_discs = expected_stack_weight :=
sorry

end metal_disc_weight_expectation_l1427_142702


namespace smaller_number_problem_l1427_142758

theorem smaller_number_problem (x y : ℕ) : 
  x * y = 56 → x + y = 15 → x ≤ y → x ∣ 28 → x = 7 := by
  sorry

end smaller_number_problem_l1427_142758


namespace sector_area_l1427_142782

theorem sector_area (α : Real) (perimeter : Real) (h1 : α = 1/3) (h2 : perimeter = 7) :
  let r := perimeter / (2 + α)
  (1/2) * α * r^2 = 3/2 := by sorry

end sector_area_l1427_142782


namespace unique_solution_iff_b_eq_two_or_six_l1427_142731

/-- The function g(x) = x^2 + bx + 2b -/
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 2*b

/-- The statement that |g(x)| ≤ 3 has exactly one solution -/
def has_unique_solution (b : ℝ) : Prop :=
  ∃! x, |g b x| ≤ 3

/-- Theorem: The inequality |x^2 + bx + 2b| ≤ 3 has exactly one solution
    if and only if b = 2 or b = 6 -/
theorem unique_solution_iff_b_eq_two_or_six :
  ∀ b : ℝ, has_unique_solution b ↔ (b = 2 ∨ b = 6) := by sorry

end unique_solution_iff_b_eq_two_or_six_l1427_142731


namespace legs_walking_on_ground_l1427_142776

theorem legs_walking_on_ground (num_horses : ℕ) (num_men : ℕ) : 
  num_horses = 12 →
  num_men = num_horses →
  num_horses * 4 + (num_men / 2) * 2 = 60 :=
by sorry

end legs_walking_on_ground_l1427_142776


namespace remainder_7n_mod_5_l1427_142787

theorem remainder_7n_mod_5 (n : ℤ) (h : n % 5 = 3) : (7 * n) % 5 = 1 := by
  sorry

end remainder_7n_mod_5_l1427_142787


namespace novel_pages_prove_novel_pages_l1427_142709

theorem novel_pages : ℕ → Prop :=
  fun total_pages =>
    let day1_read := total_pages / 6 + 10
    let day1_remaining := total_pages - day1_read
    let day2_read := day1_remaining / 5 + 20
    let day2_remaining := day1_remaining - day2_read
    let day3_read := day2_remaining / 4 + 25
    let day3_remaining := day2_remaining - day3_read
    day3_remaining = 130 → total_pages = 352

theorem prove_novel_pages : novel_pages 352 := by
  sorry

end novel_pages_prove_novel_pages_l1427_142709


namespace normal_distribution_symmetry_l1427_142722

/-- A random variable following a normal distribution with mean μ and standard deviation σ -/
structure NormalRV (μ σ : ℝ) where
  X : ℝ → ℝ  -- The random variable as a function

/-- The probability that a random variable X is greater than a given value -/
noncomputable def prob_gt (X : ℝ → ℝ) (x : ℝ) : ℝ := sorry

/-- The probability that a random variable X is less than a given value -/
noncomputable def prob_lt (X : ℝ → ℝ) (x : ℝ) : ℝ := sorry

/-- Theorem: For a normal distribution N(3,1), if P(X > 2c-1) = P(X < c+3), then c = 4/3 -/
theorem normal_distribution_symmetry (c : ℝ) (X : NormalRV 3 1) :
  prob_gt X.X (2*c - 1) = prob_lt X.X (c + 3) → c = 4/3 := by
  sorry

end normal_distribution_symmetry_l1427_142722


namespace opposite_of_2023_l1427_142735

-- Define the concept of opposite (additive inverse)
def opposite (a : ℤ) : ℤ := -a

-- State the theorem
theorem opposite_of_2023 : opposite 2023 = -2023 := by sorry

end opposite_of_2023_l1427_142735


namespace least_whole_number_ratio_l1427_142738

theorem least_whole_number_ratio (x : ℕ) : x ≥ 3 ↔ (6 - x : ℚ) / (7 - x) < 16 / 21 :=
sorry

end least_whole_number_ratio_l1427_142738


namespace largest_B_181_l1427_142792

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The sequence B_k as defined in the problem -/
def B (k : ℕ) : ℝ := (binomial 2000 k : ℝ) * (0.1 ^ k)

/-- Theorem stating that B_181 is the largest among all B_k -/
theorem largest_B_181 : ∀ k : ℕ, k ≤ 2000 → B 181 ≥ B k := by sorry

end largest_B_181_l1427_142792


namespace company_j_salary_difference_l1427_142754

/-- Represents the company J with its payroll information -/
structure CompanyJ where
  factory_workers : ℕ
  office_workers : ℕ
  factory_payroll : ℕ
  office_payroll : ℕ

/-- Calculates the difference between average monthly salaries of office and factory workers -/
def salary_difference (company : CompanyJ) : ℚ :=
  (company.office_payroll / company.office_workers) - (company.factory_payroll / company.factory_workers)

/-- Theorem stating the salary difference in Company J -/
theorem company_j_salary_difference :
  ∃ (company : CompanyJ),
    company.factory_workers = 15 ∧
    company.office_workers = 30 ∧
    company.factory_payroll = 30000 ∧
    company.office_payroll = 75000 ∧
    salary_difference company = 500 := by
  sorry

end company_j_salary_difference_l1427_142754


namespace average_age_of_fourteen_students_l1427_142773

theorem average_age_of_fourteen_students
  (total_students : Nat)
  (total_average_age : ℚ)
  (ten_students_average : ℚ)
  (twenty_fifth_student_age : ℚ)
  (h1 : total_students = 25)
  (h2 : total_average_age = 25)
  (h3 : ten_students_average = 22)
  (h4 : twenty_fifth_student_age = 13) :
  (total_students * total_average_age - 10 * ten_students_average - twenty_fifth_student_age) / 14 = 28 := by
  sorry

end average_age_of_fourteen_students_l1427_142773


namespace simplify_and_ratio_l1427_142780

theorem simplify_and_ratio (k : ℚ) : ∃ (a b : ℚ), 
  (6 * k + 12) / 3 = a * k + b ∧ a / b = 1 / 2 := by
  sorry

end simplify_and_ratio_l1427_142780


namespace angle_b_measure_l1427_142708

theorem angle_b_measure (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : C = 3 * A) (h3 : B = 2 * A) : B = 60 := by
  sorry

end angle_b_measure_l1427_142708


namespace circle_inequality_l1427_142746

theorem circle_inequality (c : ℝ) : 
  (∀ x y : ℝ, x^2 + (y-1)^2 = 1 → x + y + c ≥ 0) → 
  c ≥ Real.sqrt 2 - 1 := by
sorry

end circle_inequality_l1427_142746


namespace age_problem_l1427_142737

theorem age_problem (A B C : ℕ) : 
  A = B + 2 →
  B = 2 * C →
  A + B + C = 37 →
  B = 14 :=
by
  sorry

end age_problem_l1427_142737


namespace divisor_calculation_l1427_142771

theorem divisor_calculation (dividend quotient remainder : ℕ) (h1 : dividend = 76) (h2 : quotient = 4) (h3 : remainder = 8) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 17 := by
  sorry

end divisor_calculation_l1427_142771


namespace complex_subtraction_simplification_l1427_142721

theorem complex_subtraction_simplification :
  (-5 + 3 * Complex.I) - (2 - 7 * Complex.I) = -7 + 10 * Complex.I := by
  sorry

end complex_subtraction_simplification_l1427_142721


namespace dans_work_time_l1427_142794

/-- Dan's work rate in job completion per hour -/
def dans_rate : ℚ := 1 / 15

/-- Annie's work rate in job completion per hour -/
def annies_rate : ℚ := 1 / 10

/-- The time Annie works to complete the job after Dan stops -/
def annies_time : ℚ := 6

theorem dans_work_time (x : ℚ) : 
  x * dans_rate + annies_time * annies_rate = 1 → x = 6 := by
  sorry

end dans_work_time_l1427_142794


namespace inequality_solution_l1427_142704

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 5 / (x + 4) ≥ 1) ↔ (x ∈ Set.Icc (-4 : ℝ) (-2) ∨ x ∈ Set.Ico (-2 : ℝ) 5) :=
sorry

end inequality_solution_l1427_142704


namespace product_in_geometric_sequence_l1427_142705

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_in_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 18 = -15) →
  (a 2 * a 18 = 16) →
  a 3 * a 10 * a 17 = -64 := by
  sorry

end product_in_geometric_sequence_l1427_142705


namespace triangle_area_is_two_l1427_142725

/-- The area of the triangle formed by the line x + y - 2 = 0 and the coordinate axes -/
def triangle_area : ℝ := 2

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

theorem triangle_area_is_two :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ ∧
    x₁ = 0 ∧ y₂ = 0 ∧
    (1/2 : ℝ) * x₂ * y₁ = triangle_area :=
by sorry

end triangle_area_is_two_l1427_142725


namespace parallel_line_y_intercept_l1427_142733

/-- A line parallel to y = -3x + 6 passing through (3, -2) has y-intercept 7 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b y = -3 * x + b 0) →  -- b is a line with slope -3
  b (-2) = 3 * (-3) + b 0 →     -- b passes through (3, -2)
  b 0 = 7 :=                    -- y-intercept of b is 7
by
  sorry

end parallel_line_y_intercept_l1427_142733


namespace cosine_of_angle_between_vectors_l1427_142717

/-- Given two planar vectors a and b, prove that the cosine of the angle between them is -3/5. -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) : 
  a = (2, 4) → a - 2 • b = (0, 8) → 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -3/5 := by
  sorry

#check cosine_of_angle_between_vectors

end cosine_of_angle_between_vectors_l1427_142717


namespace puppy_adoption_cost_puppy_adoption_cost_proof_l1427_142749

/-- The cost to get each puppy ready for adoption, given:
  * Cost for cats is $50 per cat
  * Cost for adult dogs is $100 per dog
  * 2 cats, 3 adult dogs, and 2 puppies were adopted
  * Total cost for all animals is $700
-/
theorem puppy_adoption_cost : ℝ :=
  let cat_cost : ℝ := 50
  let dog_cost : ℝ := 100
  let num_cats : ℕ := 2
  let num_dogs : ℕ := 3
  let num_puppies : ℕ := 2
  let total_cost : ℝ := 700
  150

theorem puppy_adoption_cost_proof (cat_cost dog_cost total_cost : ℝ) (num_cats num_dogs num_puppies : ℕ) 
  (h_cat_cost : cat_cost = 50)
  (h_dog_cost : dog_cost = 100)
  (h_num_cats : num_cats = 2)
  (h_num_dogs : num_dogs = 3)
  (h_num_puppies : num_puppies = 2)
  (h_total_cost : total_cost = 700)
  : puppy_adoption_cost = (total_cost - (↑num_cats * cat_cost + ↑num_dogs * dog_cost)) / ↑num_puppies :=
by sorry

end puppy_adoption_cost_puppy_adoption_cost_proof_l1427_142749


namespace pauls_homework_average_l1427_142789

/-- Represents the homework schedule for Paul --/
structure HomeworkSchedule where
  weeknight_hours : ℕ
  weekend_hours : ℕ
  practice_nights : ℕ
  total_nights : ℕ

/-- Calculates the average homework hours per available night --/
def average_homework_hours (schedule : HomeworkSchedule) : ℚ :=
  let total_homework := schedule.weeknight_hours * (schedule.total_nights - 2) + schedule.weekend_hours
  let available_nights := schedule.total_nights - schedule.practice_nights
  (total_homework : ℚ) / available_nights

/-- Theorem stating that Paul's average homework hours per available night is 3 --/
theorem pauls_homework_average (pauls_schedule : HomeworkSchedule) 
  (h1 : pauls_schedule.weeknight_hours = 2)
  (h2 : pauls_schedule.weekend_hours = 5)
  (h3 : pauls_schedule.practice_nights = 2)
  (h4 : pauls_schedule.total_nights = 7) :
  average_homework_hours pauls_schedule = 3 := by
  sorry


end pauls_homework_average_l1427_142789


namespace closure_union_M_N_l1427_142729

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x : ℝ | x ≤ -3}

-- State the theorem
theorem closure_union_M_N :
  closure (M ∪ N) = {x : ℝ | x ≥ 1} := by sorry

end closure_union_M_N_l1427_142729


namespace parallel_line_length_l1427_142713

/-- Given a triangle with base 20 inches and a parallel line dividing it into two parts
    where the upper part has 3/4 of the total area, the length of this parallel line is 10 inches. -/
theorem parallel_line_length (base : ℝ) (parallel_line : ℝ) : 
  base = 20 →
  (parallel_line / base) ^ 2 = 1 / 4 →
  parallel_line = 10 :=
by sorry

end parallel_line_length_l1427_142713


namespace system_solution_l1427_142764

theorem system_solution :
  let S : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 
    x + y - z = 4 ∧
    x^2 - y^2 + z^2 = -4 ∧
    x * y * z = 6}
  S = {(2, 3, 1), (-1, 3, -2)} := by sorry

end system_solution_l1427_142764


namespace ellipse_foci_distance_l1427_142712

-- Define the ellipse type
structure Ellipse where
  endpoints : List (ℝ × ℝ)
  axes_perpendicular : Bool

-- Define the function to calculate the distance between foci
noncomputable def distance_between_foci (e : Ellipse) : ℝ :=
  sorry

-- Theorem statement
theorem ellipse_foci_distance 
  (e : Ellipse) 
  (h1 : e.endpoints = [(1, 3), (7, -5), (1, -5)])
  (h2 : e.axes_perpendicular = true) : 
  distance_between_foci e = 12 := by
  sorry

end ellipse_foci_distance_l1427_142712


namespace no_solutions_for_equation_l1427_142751

theorem no_solutions_for_equation : ¬∃ (x y : ℕ+), x^12 = 26*y^3 + 2023 := by
  sorry

end no_solutions_for_equation_l1427_142751


namespace parallel_line_through_point_l1427_142798

/-- Given two lines in a plane, this theorem states that if one line passes through 
    a specific point and is parallel to the other line, then it has a specific equation. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (x - 2*y + 3 = 0) →  -- Given line
  (2 = x ∧ 0 = y) →    -- Point (2, 0)
  (2*y - x + 2 = 0) →  -- Equation to prove
  ∃ (m b : ℝ), (y = m*x + b ∧ 2*y - x + 2 = 0) ∧ 
               (∃ (c : ℝ), x - 2*y + c = 0) :=
by sorry

end parallel_line_through_point_l1427_142798


namespace scientific_notation_of_error_l1427_142715

theorem scientific_notation_of_error : ∃ (a : ℝ) (n : ℤ), 
  0.0000003 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3 ∧ n = -7 := by
  sorry

end scientific_notation_of_error_l1427_142715


namespace identify_counterfeit_coins_l1427_142770

/-- Represents a coin which can be either real or counterfeit -/
inductive Coin
| Real
| CounterfeitLight
| CounterfeitHeavy

/-- Represents the result of a weighing -/
inductive WeighingResult
| LeftHeavier
| RightHeavier
| Equal

/-- Represents a set of five coins -/
def CoinSet := Fin 5 → Coin

/-- Represents a weighing operation on the balance scale -/
def Weighing := List Nat → List Nat → WeighingResult

/-- The main theorem stating that it's possible to identify counterfeit coins in three weighings -/
theorem identify_counterfeit_coins 
  (coins : CoinSet) 
  (h1 : ∃ (i j : Fin 5), i ≠ j ∧ coins i = Coin.CounterfeitLight ∧ coins j = Coin.CounterfeitHeavy) 
  (h2 : ∀ (i : Fin 5), coins i ≠ Coin.CounterfeitLight → coins i ≠ Coin.CounterfeitHeavy → coins i = Coin.Real) :
  ∃ (w1 w2 w3 : Weighing), 
    ∀ (i j : Fin 5), 
      (coins i = Coin.CounterfeitLight ∧ coins j = Coin.CounterfeitHeavy) → 
      ∃ (f : Weighing → Weighing → Weighing → Fin 5 × Fin 5), 
        f w1 w2 w3 = (i, j) :=
sorry

end identify_counterfeit_coins_l1427_142770


namespace sugar_amount_in_new_recipe_l1427_142727

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio :=
  (flour : ℕ)
  (water : ℕ)
  (sugar : ℕ)

/-- Calculates the new ratio based on the original ratio -/
def new_ratio (original : RecipeRatio) : RecipeRatio :=
  { flour := original.flour,
    water := original.flour / 2,
    sugar := original.sugar * 2 }

/-- Calculates the amount of an ingredient based on the ratio and a known amount -/
def calculate_amount (ratio : RecipeRatio) (known_part : ℕ) (known_amount : ℚ) (target_part : ℕ) : ℚ :=
  (known_amount * target_part) / known_part

theorem sugar_amount_in_new_recipe : 
  let original_ratio := RecipeRatio.mk 8 4 3
  let new_ratio := new_ratio original_ratio
  let water_amount : ℚ := 2
  calculate_amount new_ratio new_ratio.water water_amount new_ratio.sugar = 3 := by
  sorry

end sugar_amount_in_new_recipe_l1427_142727


namespace segment_point_relation_l1427_142777

/-- Represents a line segment with a midpoint -/
structure Segment where
  length : ℝ
  midpoint : ℝ

/-- Represents a point on a segment -/
structure PointOnSegment where
  segment : Segment
  distanceFromMidpoint : ℝ

/-- The theorem statement -/
theorem segment_point_relation 
  (ab : Segment)
  (a_prime_b_prime : Segment)
  (p : PointOnSegment)
  (p_prime : PointOnSegment)
  (h1 : ab.length = 10)
  (h2 : a_prime_b_prime.length = 18)
  (h3 : p.segment = ab)
  (h4 : p_prime.segment = a_prime_b_prime)
  (h5 : 3 * p.distanceFromMidpoint - 2 * p_prime.distanceFromMidpoint = 6)
  : p.distanceFromMidpoint + p_prime.distanceFromMidpoint = 12 := by
  sorry

end segment_point_relation_l1427_142777


namespace common_terms_is_geometric_l1427_142784

/-- Arithmetic sequence with sum of first n terms S_n = (3n^2 + 5n) / 2 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  3 * n + 1

/-- Geometric sequence with b_3 = 4 and b_6 = 32 -/
def geometric_sequence (n : ℕ) : ℚ :=
  2^(n - 1)

/-- Sequence of common terms between arithmetic_sequence and geometric_sequence -/
def common_terms (n : ℕ) : ℚ :=
  4^n

theorem common_terms_is_geometric :
  ∀ n : ℕ, n > 0 → ∃ k : ℕ, k > 0 ∧ 
    arithmetic_sequence k = geometric_sequence k ∧
    common_terms n = arithmetic_sequence k := by
  sorry

end common_terms_is_geometric_l1427_142784


namespace total_amount_is_200_l1427_142791

/-- Represents the distribution of money among four individuals -/
structure MoneyDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total amount of money distributed -/
def total_amount (d : MoneyDistribution) : ℝ :=
  d.w + d.x + d.y + d.z

/-- Theorem stating the total amount given the conditions -/
theorem total_amount_is_200 (d : MoneyDistribution) 
  (h1 : d.x = 0.75 * d.w)
  (h2 : d.y = 0.45 * d.w)
  (h3 : d.z = 0.30 * d.w)
  (h4 : d.y = 36) :
  total_amount d = 200 := by
  sorry

#check total_amount_is_200

end total_amount_is_200_l1427_142791


namespace school_seat_cost_l1427_142772

/-- Calculates the total cost of seats with discounts applied --/
def totalCostWithDiscounts (
  rows1 : ℕ) (seats1 : ℕ) (price1 : ℕ) (discount1 : ℚ)
  (rows2 : ℕ) (seats2 : ℕ) (price2 : ℕ) (discount2 : ℚ) (extraDiscount2 : ℚ)
  (rows3 : ℕ) (seats3 : ℕ) (price3 : ℕ) (discount3 : ℚ) : ℚ :=
  let totalSeats1 := rows1 * seats1
  let totalSeats2 := rows2 * seats2
  let totalSeats3 := rows3 * seats3
  let cost1 := totalSeats1 * price1
  let cost2 := totalSeats2 * price2
  let cost3 := totalSeats3 * price3
  let discountedCost1 := cost1 * (1 - discount1 * (totalSeats1 / seats1))
  let discountedCost2 := 
    if totalSeats2 ≥ 30 then
      cost2 * (1 - discount2 * (totalSeats2 / seats2)) * (1 - extraDiscount2)
    else
      cost2 * (1 - discount2 * (totalSeats2 / seats2))
  let discountedCost3 := cost3 * (1 - discount3 * (totalSeats3 / seats3))
  discountedCost1 + discountedCost2 + discountedCost3

/-- Theorem stating the total cost for the school --/
theorem school_seat_cost : 
  totalCostWithDiscounts 10 20 60 (12/100)
                         10 15 50 (10/100) (3/100)
                         5 10 40 (8/100) = 18947.50 := by
  sorry

end school_seat_cost_l1427_142772


namespace chipmunk_families_left_l1427_142795

theorem chipmunk_families_left (original : ℕ) (went_away : ℕ) (h1 : original = 86) (h2 : went_away = 65) :
  original - went_away = 21 := by
  sorry

end chipmunk_families_left_l1427_142795


namespace line_problem_l1427_142785

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_problem (m n : ℝ) :
  let l1 : Line := ⟨2, 2, -1⟩
  let l2 : Line := ⟨4, n, 3⟩
  let l3 : Line := ⟨m, 6, 1⟩
  are_parallel l1 l2 ∧ are_perpendicular l1 l3 → m + n = -2 := by
  sorry

end line_problem_l1427_142785


namespace triangle_problem_l1427_142759

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C opposite to them respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement for the given triangle problem -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 3)
  (h2 : t.b - t.c = 2)
  (h3 : Real.cos t.B = -1/2) :
  t.b = 7 ∧ t.c = 5 ∧ Real.sin (t.B - t.C) = (4 * Real.sqrt 3) / 7 := by
  sorry


end triangle_problem_l1427_142759


namespace imaginary_part_of_z_l1427_142736

-- Define the complex number z
def z : ℂ := Complex.I * (2 + Complex.I)

-- Theorem stating that the imaginary part of z is 2
theorem imaginary_part_of_z : z.im = 2 := by
  sorry

end imaginary_part_of_z_l1427_142736


namespace max_round_value_l1427_142744

/-- Represents a digit assignment for the letter puzzle --/
structure DigitAssignment where
  H : Fin 10
  M : Fin 10
  T : Fin 10
  G : Fin 10
  U : Fin 10
  S : Fin 10
  R : Fin 10
  O : Fin 10
  N : Fin 10
  D : Fin 10

/-- Checks if all digits in the assignment are distinct --/
def allDistinct (a : DigitAssignment) : Prop :=
  a.H ≠ a.M ∧ a.H ≠ a.T ∧ a.H ≠ a.G ∧ a.H ≠ a.U ∧ a.H ≠ a.S ∧ a.H ≠ a.R ∧ a.H ≠ a.O ∧ a.H ≠ a.N ∧ a.H ≠ a.D ∧
  a.M ≠ a.T ∧ a.M ≠ a.G ∧ a.M ≠ a.U ∧ a.M ≠ a.S ∧ a.M ≠ a.R ∧ a.M ≠ a.O ∧ a.M ≠ a.N ∧ a.M ≠ a.D ∧
  a.T ≠ a.G ∧ a.T ≠ a.U ∧ a.T ≠ a.S ∧ a.T ≠ a.R ∧ a.T ≠ a.O ∧ a.T ≠ a.N ∧ a.T ≠ a.D ∧
  a.G ≠ a.U ∧ a.G ≠ a.S ∧ a.G ≠ a.R ∧ a.G ≠ a.O ∧ a.G ≠ a.N ∧ a.G ≠ a.D ∧
  a.U ≠ a.S ∧ a.U ≠ a.R ∧ a.U ≠ a.O ∧ a.U ≠ a.N ∧ a.U ≠ a.D ∧
  a.S ≠ a.R ∧ a.S ≠ a.O ∧ a.S ≠ a.N ∧ a.S ≠ a.D ∧
  a.R ≠ a.O ∧ a.R ≠ a.N ∧ a.R ≠ a.D ∧
  a.O ≠ a.N ∧ a.O ≠ a.D ∧
  a.N ≠ a.D

/-- Checks if the equation HMMT + GUTS = ROUND is satisfied --/
def equationSatisfied (a : DigitAssignment) : Prop :=
  1000 * a.H.val + 100 * a.M.val + 10 * a.M.val + a.T.val +
  1000 * a.G.val + 100 * a.U.val + 10 * a.T.val + a.S.val =
  10000 * a.R.val + 1000 * a.O.val + 100 * a.U.val + 10 * a.N.val + a.D.val

/-- Checks if there are no leading zeroes --/
def noLeadingZeroes (a : DigitAssignment) : Prop :=
  a.H ≠ 0 ∧ a.G ≠ 0 ∧ a.R ≠ 0

/-- The value of ROUND for a given digit assignment --/
def roundValue (a : DigitAssignment) : ℕ :=
  10000 * a.R.val + 1000 * a.O.val + 100 * a.U.val + 10 * a.N.val + a.D.val

/-- The main theorem statement --/
theorem max_round_value :
  ∀ a : DigitAssignment,
    allDistinct a →
    equationSatisfied a →
    noLeadingZeroes a →
    roundValue a ≤ 16352 :=
sorry

end max_round_value_l1427_142744


namespace greatest_two_digit_multiple_of_17_l1427_142796

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n = 85 ∧ 
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ 17 ∣ m → m ≤ n) ∧ 
  10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n :=
by sorry

end greatest_two_digit_multiple_of_17_l1427_142796


namespace quadratic_discriminant_condition_l1427_142783

theorem quadratic_discriminant_condition (a b c : ℝ) :
  (2 * a ≠ 0) →
  (ac = (9 * b^2 - 25) / 32) ↔ ((3 * b)^2 - 4 * (2 * a) * (4 * c) = 25) :=
by sorry

end quadratic_discriminant_condition_l1427_142783


namespace equal_spending_dolls_l1427_142766

/-- The number of sisters Tonya is buying gifts for -/
def num_sisters : ℕ := 2

/-- The cost of each doll in dollars -/
def doll_cost : ℕ := 15

/-- The cost of each lego set in dollars -/
def lego_cost : ℕ := 20

/-- The number of lego sets bought for the older sister -/
def num_lego_sets : ℕ := 3

/-- The total amount spent on the older sister in dollars -/
def older_sister_cost : ℕ := num_lego_sets * lego_cost

/-- The number of dolls bought for the younger sister -/
def num_dolls : ℕ := older_sister_cost / doll_cost

theorem equal_spending_dolls : num_dolls = 4 := by
  sorry

end equal_spending_dolls_l1427_142766


namespace intersection_of_A_and_B_l1427_142748

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l1427_142748


namespace mike_seeds_left_l1427_142799

/-- The number of seeds Mike has left after feeding the birds -/
def seeds_left (total : ℕ) (left : ℕ) (right_multiplier : ℕ) (late : ℕ) : ℕ :=
  total - (left + right_multiplier * left + late)

/-- Theorem stating that Mike has 30 seeds left -/
theorem mike_seeds_left :
  seeds_left 120 20 2 30 = 30 := by
  sorry

end mike_seeds_left_l1427_142799


namespace cubic_polynomial_uniqueness_l1427_142742

-- Define the cubic polynomial whose roots are a, b, c
def cubic (x : ℝ) := x^3 + 4*x^2 + 6*x + 8

-- Define the properties of P
def P_properties (P : ℝ → ℝ) (a b c : ℝ) : Prop :=
  cubic a = 0 ∧ cubic b = 0 ∧ cubic c = 0 ∧
  P a = b + c ∧ P b = a + c ∧ P c = a + b ∧
  P (a + b + c) = -20

-- Define the specific polynomial we want to prove is equal to P
def target_poly (x : ℝ) := 2*x^3 + 7*x^2 + 11*x + 12

-- The main theorem
theorem cubic_polynomial_uniqueness :
  ∀ (P : ℝ → ℝ) (a b c : ℝ),
  P_properties P a b c →
  (∀ x, P x = target_poly x) :=
sorry

end cubic_polynomial_uniqueness_l1427_142742


namespace train_speed_l1427_142762

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 160) (h2 : time = 6) :
  length / time = 26.67 := by
  sorry

end train_speed_l1427_142762


namespace two_numbers_puzzle_l1427_142763

def is_two_digit_same_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (n / 10 = n % 10)

def is_three_digit_same_digits (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = (n / 10) % 10) ∧ (n / 100 = n % 10)

theorem two_numbers_puzzle :
  ∀ a b : ℕ,
    a > 0 ∧ b > 0 →
    is_two_digit_same_digits (a + b) →
    is_three_digit_same_digits (a * b) →
    ((a = 37 ∧ b = 18) ∨ (a = 18 ∧ b = 37) ∨ (a = 74 ∧ b = 3) ∨ (a = 3 ∧ b = 74)) :=
by sorry

end two_numbers_puzzle_l1427_142763


namespace smallest_a_value_l1427_142779

-- Define the polynomial
def polynomial (a b x : ℤ) : ℤ := x^3 - a*x^2 + b*x - 2310

-- Define the property of having three positive integer roots
def has_three_positive_integer_roots (a b : ℤ) : Prop :=
  ∃ (x y z : ℤ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    polynomial a b x = 0 ∧ polynomial a b y = 0 ∧ polynomial a b z = 0

-- State the theorem
theorem smallest_a_value :
  ∀ a b : ℤ, has_three_positive_integer_roots a b →
    (∀ a' b' : ℤ, has_three_positive_integer_roots a' b' → a ≤ a') →
    a = 88 :=
sorry

end smallest_a_value_l1427_142779


namespace conference_handshakes_l1427_142741

theorem conference_handshakes (total : ℕ) (group_a : ℕ) (group_b : ℕ) :
  total = 50 →
  group_a = 30 →
  group_b = 20 →
  group_a + group_b = total →
  (group_a * group_b) + (group_b * (group_b - 1) / 2) = 790 :=
by sorry

end conference_handshakes_l1427_142741


namespace martha_collected_90_cans_l1427_142757

/-- The number of cans Martha collected -/
def martha_cans : ℕ := sorry

/-- The number of cans Diego collected -/
def diego_cans (m : ℕ) : ℕ := m / 2 + 10

/-- The total number of cans collected -/
def total_cans : ℕ := 145

theorem martha_collected_90_cans :
  martha_cans = 90 ∧ 
  diego_cans martha_cans = martha_cans / 2 + 10 ∧
  martha_cans + diego_cans martha_cans = total_cans :=
sorry

end martha_collected_90_cans_l1427_142757


namespace smallest_n_with_conditions_n_9000_satisfies_conditions_l1427_142750

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def contains_digit_9 (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ n.digits 10 ∧ d = 9

theorem smallest_n_with_conditions :
  ∀ n : ℕ, n > 0 →
    (is_terminating_decimal n ∧ contains_digit_9 n ∧ n % 3 = 0) →
    n ≥ 9000 :=
by sorry

theorem n_9000_satisfies_conditions :
  is_terminating_decimal 9000 ∧ contains_digit_9 9000 ∧ 9000 % 3 = 0 :=
by sorry

end smallest_n_with_conditions_n_9000_satisfies_conditions_l1427_142750


namespace unique_remainder_mod_11_l1427_142797

theorem unique_remainder_mod_11 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end unique_remainder_mod_11_l1427_142797


namespace smallest_common_multiple_of_7_and_4_l1427_142718

theorem smallest_common_multiple_of_7_and_4 : ∃ (n : ℕ), n > 0 ∧ n % 7 = 0 ∧ n % 4 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 7 = 0 ∧ m % 4 = 0 → n ≤ m :=
by sorry

end smallest_common_multiple_of_7_and_4_l1427_142718


namespace max_rental_income_l1427_142719

/-- Represents the daily rental income for the construction company. -/
def daily_rental_income (x : ℕ) : ℝ :=
  -200 * x + 80000

/-- The problem statement and proof objective. -/
theorem max_rental_income :
  let total_vehicles : ℕ := 50
  let type_a_vehicles : ℕ := 20
  let type_b_vehicles : ℕ := 30
  let site_a_vehicles : ℕ := 30
  let site_b_vehicles : ℕ := 20
  let site_a_type_a_price : ℝ := 1800
  let site_a_type_b_price : ℝ := 1600
  let site_b_type_a_price : ℝ := 1600
  let site_b_type_b_price : ℝ := 1200
  ∀ x : ℕ, x ≤ type_a_vehicles →
    daily_rental_income x ≤ 80000 ∧
    (∃ x₀ : ℕ, x₀ ≤ type_a_vehicles ∧ daily_rental_income x₀ = 80000) :=
by sorry

#check max_rental_income

end max_rental_income_l1427_142719


namespace power_of_54_l1427_142734

theorem power_of_54 (a b : ℕ+) (h : (54 : ℕ) ^ a.val = a.val ^ b.val) :
  ∃ k : ℕ, a.val = (54 : ℕ) ^ k := by
sorry

end power_of_54_l1427_142734


namespace equation_proof_l1427_142786

theorem equation_proof : (8/3 + 3/2) / (15/4) - 0.4 = 32/45 := by
  sorry

end equation_proof_l1427_142786


namespace one_more_green_than_red_peaches_l1427_142707

/-- Given a basket of peaches with specified quantities of red, yellow, and green peaches,
    prove that there is one more green peach than red peaches. -/
theorem one_more_green_than_red_peaches 
  (red : ℕ) (yellow : ℕ) (green : ℕ)
  (h_red : red = 7)
  (h_yellow : yellow = 71)
  (h_green : green = 8) :
  green - red = 1 := by
  sorry

end one_more_green_than_red_peaches_l1427_142707


namespace probability_point_in_circle_l1427_142723

/-- The probability of a point randomly selected from a square with side length 4
    being within a circle of radius 2 centered at the origin is π/4. -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) :
  square_side = 4 →
  circle_radius = 2 →
  (π * circle_radius^2) / (square_side^2) = π / 4 := by
  sorry

end probability_point_in_circle_l1427_142723


namespace go_and_chess_problem_l1427_142714

theorem go_and_chess_problem (x y z : ℝ) : 
  (3 * x + 5 * y = 98) →
  (8 * x + 3 * y = 158) →
  (z + (40 - z) = 40) →
  (16 * z + 10 * (40 - z) ≤ 550) →
  (x = 16 ∧ y = 10 ∧ z ≤ 25) := by
  sorry

end go_and_chess_problem_l1427_142714


namespace unique_three_digit_number_l1427_142745

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_nonzero_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ 0 ∧ tens ≠ 0 ∧ ones ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem unique_three_digit_number :
  ∃! n : ℕ, is_three_digit n ∧ has_nonzero_digits n ∧ 222 * (sum_of_digits n) - n = 1990 :=
sorry

end unique_three_digit_number_l1427_142745


namespace polyhedron_sum_l1427_142739

/-- Represents a convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ -- number of vertices
  E : ℕ -- number of edges
  F : ℕ -- number of faces
  q : ℕ -- number of quadrilateral faces
  h : ℕ -- number of hexagonal faces
  Q : ℕ -- number of quadrilateral faces meeting at each vertex
  H : ℕ -- number of hexagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 24
  face_types : q + h = F
  edge_count : E = 2*q + 3*h
  vertex_degree : Q = 1 ∧ H = 1

/-- The main theorem to be proved -/
theorem polyhedron_sum (p : ConvexPolyhedron) : 100 * p.H + 10 * p.Q + p.V = 136 := by
  sorry

end polyhedron_sum_l1427_142739


namespace negation_equivalence_l1427_142790

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x > 0 ∧ -x^2 + 2*x - 1 > 0) ↔ 
  (∀ x : ℝ, x > 0 → -x^2 + 2*x - 1 ≤ 0) :=
by sorry

end negation_equivalence_l1427_142790


namespace constant_d_value_l1427_142752

variables (a d : ℝ)

theorem constant_d_value (h : ∀ x : ℝ, (x + 3) * (x + a) = x^2 + d*x + 12) : d = 7 := by
  sorry

end constant_d_value_l1427_142752


namespace P_greater_than_Q_l1427_142769

theorem P_greater_than_Q : 
  let P : ℝ := Real.sqrt 7 - 1
  let Q : ℝ := Real.sqrt 11 - Real.sqrt 5
  P > Q := by sorry

end P_greater_than_Q_l1427_142769


namespace complement_P_intersect_Q_l1427_142747

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q :
  (Set.univ \ P) ∩ Q = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end complement_P_intersect_Q_l1427_142747
