import Mathlib

namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2316_231662

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {0, 2, 4}

-- Define set B
def B : Set Nat := {0, 5}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2316_231662


namespace NUMINAMATH_CALUDE_sqrt_square_789256_l2316_231688

theorem sqrt_square_789256 : (Real.sqrt 789256)^2 = 789256 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_789256_l2316_231688


namespace NUMINAMATH_CALUDE_abs_inequality_solution_l2316_231679

theorem abs_inequality_solution (x : ℝ) : 
  |x + 3| - |2*x - 1| < x/2 + 1 ↔ x < -2/5 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_l2316_231679


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l2316_231659

theorem trigonometric_expression_equality : 
  (Real.cos (190 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180))) / 
  (Real.sin (290 * π / 180) * Real.sqrt (1 - Real.cos (40 * π / 180))) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l2316_231659


namespace NUMINAMATH_CALUDE_trig_simplification_l2316_231632

theorem trig_simplification (α : ℝ) : 
  (2 * Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2316_231632


namespace NUMINAMATH_CALUDE_num_adults_on_trip_l2316_231614

def total_eggs : ℕ := 36
def eggs_per_adult : ℕ := 3
def num_girls : ℕ := 7
def num_boys : ℕ := 10
def eggs_per_girl : ℕ := 1
def eggs_per_boy : ℕ := eggs_per_girl + 1

theorem num_adults_on_trip : 
  total_eggs - (num_girls * eggs_per_girl + num_boys * eggs_per_boy) = 3 * eggs_per_adult := by
  sorry

end NUMINAMATH_CALUDE_num_adults_on_trip_l2316_231614


namespace NUMINAMATH_CALUDE_money_distribution_l2316_231650

theorem money_distribution (a b c total : ℕ) : 
  a + b + c = 9 →
  b = 3 →
  1200 * 3 = total →
  total = 3600 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l2316_231650


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l2316_231672

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (80 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (40 * π / 180) = 
  (4 + 2 * (1 / Real.cos (40 * π / 180))) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l2316_231672


namespace NUMINAMATH_CALUDE_bacon_tomatoes_difference_l2316_231658

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 228

/-- The number of students who suggested bacon -/
def bacon : ℕ := 337

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 23

/-- The difference between the number of students who suggested bacon and tomatoes -/
theorem bacon_tomatoes_difference : bacon - tomatoes = 314 := by
  sorry

end NUMINAMATH_CALUDE_bacon_tomatoes_difference_l2316_231658


namespace NUMINAMATH_CALUDE_height_difference_after_growth_spurt_l2316_231677

theorem height_difference_after_growth_spurt 
  (uncle_height : ℝ) 
  (james_initial_ratio : ℝ) 
  (sarah_initial_ratio : ℝ) 
  (james_growth : ℝ) 
  (sarah_growth : ℝ) 
  (h1 : uncle_height = 72) 
  (h2 : james_initial_ratio = 2/3) 
  (h3 : sarah_initial_ratio = 3/4) 
  (h4 : james_growth = 10) 
  (h5 : sarah_growth = 12) : 
  (james_initial_ratio * uncle_height + james_growth + 
   sarah_initial_ratio * james_initial_ratio * uncle_height + sarah_growth) - uncle_height = 34 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_after_growth_spurt_l2316_231677


namespace NUMINAMATH_CALUDE_no_real_solutions_for_x_l2316_231606

theorem no_real_solutions_for_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x + 1/y = 8) (eq2 : y + 1/x = 7/20) : False :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_x_l2316_231606


namespace NUMINAMATH_CALUDE_problem_solution_l2316_231633

theorem problem_solution (m n : ℕ) 
  (h1 : m + 10 < n + 1) 
  (h2 : (m + (m + 4) + (m + 10) + (n + 1) + (n + 2) + 2*n) / 6 = n) 
  (h3 : ((m + 10) + (n + 1)) / 2 = n) : 
  m + n = 21 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2316_231633


namespace NUMINAMATH_CALUDE_product_of_two_digit_numbers_l2316_231653

theorem product_of_two_digit_numbers (a b c d : ℕ) : 
  a < 10 → b < 10 → c < 10 → d < 10 →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (10 * a + b) * (10 * c + b) = 111 * d →
  a + b + c + d = 21 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_digit_numbers_l2316_231653


namespace NUMINAMATH_CALUDE_pentagon_square_side_ratio_l2316_231699

/-- The ratio of the side length of a regular pentagon to the side length of a square 
    with the same perimeter -/
theorem pentagon_square_side_ratio : 
  ∀ (pentagon_side square_side : ℝ),
  pentagon_side > 0 → square_side > 0 →
  5 * pentagon_side = 20 →
  4 * square_side = 20 →
  pentagon_side / square_side = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_square_side_ratio_l2316_231699


namespace NUMINAMATH_CALUDE_sin_difference_product_l2316_231646

theorem sin_difference_product (a b : ℝ) :
  Real.sin (2 * a + b) - Real.sin b = 2 * Real.cos (a + b) * Real.sin a := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_product_l2316_231646


namespace NUMINAMATH_CALUDE_no_valid_box_dimensions_l2316_231636

theorem no_valid_box_dimensions : 
  ¬∃ (a b c : ℕ), 
    (1 ≤ a) ∧ (a ≤ b) ∧ (b ≤ c) ∧ 
    (a * b * c = 3 * (2 * a * b + 2 * b * c + 2 * c * a)) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_box_dimensions_l2316_231636


namespace NUMINAMATH_CALUDE_knight_statements_count_l2316_231649

/-- Represents the type of islanders -/
inductive IslanderType
| Knight
| Liar

/-- The total number of islanders -/
def total_islanders : ℕ := 28

/-- The number of times "You are a liar!" was said -/
def liar_statements : ℕ := 230

/-- Function to calculate the number of "You are a knight!" statements -/
def knight_statements (knights : ℕ) (liars : ℕ) : ℕ :=
  knights * (knights - 1) / 2 + liars * (liars - 1) / 2

theorem knight_statements_count :
  ∃ (knights liars : ℕ),
    knights ≥ 2 ∧
    liars ≥ 2 ∧
    knights + liars = total_islanders ∧
    knights * liars = liar_statements / 2 ∧
    knight_statements knights liars + liar_statements = total_islanders * (total_islanders - 1) ∧
    knight_statements knights liars = 526 :=
by
  sorry

end NUMINAMATH_CALUDE_knight_statements_count_l2316_231649


namespace NUMINAMATH_CALUDE_intersection_M_N_l2316_231674

def M : Set ℝ := {x | x^2 - 1 < 0}

def N : Set ℝ := {y | ∃ x ∈ M, y = Real.log (x + 2)}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2316_231674


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2316_231620

/-- Given a point A(2,1) and a line 2x-y+3=0, prove that 2x-y-3=0 is the equation of the line passing through A and parallel to the given line. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (2 * x - y + 3 = 0) →  -- Given line equation
  (2 * 2 - 1 + 3 = 0) →  -- Point A(2,1) satisfies the equation of the parallel line
  (2 * x - y - 3 = 0) →  -- Equation of the line to be proved
  (∃ k : ℝ, k ≠ 0 ∧ (2 : ℝ) / 1 = (2 : ℝ) / 1) ∧  -- Slopes are equal (parallel condition)
  (2 * 2 - 1 - 3 = 0)  -- Point A(2,1) satisfies the equation of the line to be proved
  := by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2316_231620


namespace NUMINAMATH_CALUDE_jinyoung_fewest_marbles_l2316_231640

-- Define the number of marbles for each person
def minjeong_marbles : ℕ := 6
def joohwan_marbles : ℕ := 7
def sunho_marbles : ℕ := minjeong_marbles - 1
def jinyoung_marbles : ℕ := joohwan_marbles - 3

-- Define a function to get the number of marbles for each person
def marbles (person : String) : ℕ :=
  match person with
  | "Minjeong" => minjeong_marbles
  | "Joohwan" => joohwan_marbles
  | "Sunho" => sunho_marbles
  | "Jinyoung" => jinyoung_marbles
  | _ => 0

-- Theorem: Jinyoung has the fewest marbles
theorem jinyoung_fewest_marbles :
  ∀ person, person ≠ "Jinyoung" → marbles "Jinyoung" ≤ marbles person :=
by sorry

end NUMINAMATH_CALUDE_jinyoung_fewest_marbles_l2316_231640


namespace NUMINAMATH_CALUDE_smallest_q_for_five_in_range_l2316_231689

/-- The function g(x) defined as x^2 - 4x + q -/
def g (q : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + q

/-- 5 is within the range of g(x) -/
def in_range (q : ℝ) : Prop := ∃ x, g q x = 5

/-- The smallest value of q such that 5 is within the range of g(x) is 9 -/
theorem smallest_q_for_five_in_range : 
  (∃ q₀, in_range q₀ ∧ ∀ q, in_range q → q₀ ≤ q) ∧ 
  (∀ q, in_range q ↔ 9 ≤ q) :=
sorry

end NUMINAMATH_CALUDE_smallest_q_for_five_in_range_l2316_231689


namespace NUMINAMATH_CALUDE_quartic_root_ratio_l2316_231671

theorem quartic_root_ratio (a b c d e : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) →
  d / e = -25 / 12 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_ratio_l2316_231671


namespace NUMINAMATH_CALUDE_product_mod_25_l2316_231668

theorem product_mod_25 (m : ℕ) : 
  95 * 115 * 135 ≡ m [MOD 25] → 0 ≤ m → m < 25 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_25_l2316_231668


namespace NUMINAMATH_CALUDE_cos_72_minus_cos_144_l2316_231629

/-- Proves that the difference between cosine of 72 degrees and cosine of 144 degrees is 1/2 -/
theorem cos_72_minus_cos_144 : Real.cos (72 * π / 180) - Real.cos (144 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_72_minus_cos_144_l2316_231629


namespace NUMINAMATH_CALUDE_dads_real_age_l2316_231605

theorem dads_real_age (reported_age : ℕ) (h : reported_age = 35) : 
  ∃ (real_age : ℕ), (5 : ℚ) / 7 * real_age = reported_age ∧ real_age = 49 := by
  sorry

end NUMINAMATH_CALUDE_dads_real_age_l2316_231605


namespace NUMINAMATH_CALUDE_house_number_unit_digit_l2316_231690

def is_divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

def hundred_digit (n : ℕ) : ℕ := (n / 100) % 10

def unit_digit (n : ℕ) : ℕ := n % 10

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem house_number_unit_digit (n : ℕ) 
  (three_digit : 100 ≤ n ∧ n < 1000)
  (exactly_three_true : ∃ (s1 s2 s3 s4 s5 : Prop), 
    (s1 = is_divisible_by n 9) ∧
    (s2 = is_even n) ∧
    (s3 = (hundred_digit n = 3)) ∧
    (s4 = is_odd (unit_digit n)) ∧
    (s5 = is_divisible_by n 5) ∧
    ((s1 ∧ s2 ∧ s3 ∧ ¬s4 ∧ ¬s5) ∨
     (s1 ∧ s2 ∧ s3 ∧ ¬s4 ∧ s5) ∨
     (s1 ∧ s2 ∧ ¬s3 ∧ ¬s4 ∧ s5) ∨
     (s1 ∧ ¬s2 ∧ s3 ∧ ¬s4 ∧ s5) ∨
     (¬s1 ∧ s2 ∧ s3 ∧ ¬s4 ∧ s5))) :
  unit_digit n = 0 := by sorry

end NUMINAMATH_CALUDE_house_number_unit_digit_l2316_231690


namespace NUMINAMATH_CALUDE_free_throw_probabilities_l2316_231630

/-- Free throw success rates for players A and B -/
structure FreeThrowRates where
  player_a : ℚ
  player_b : ℚ

/-- Calculates the probability of exactly one successful shot when each player takes one free throw -/
def prob_one_success (rates : FreeThrowRates) : ℚ :=
  rates.player_a * (1 - rates.player_b) + rates.player_b * (1 - rates.player_a)

/-- Calculates the probability of at least one successful shot when each player takes two free throws -/
def prob_at_least_one_success (rates : FreeThrowRates) : ℚ :=
  1 - (1 - rates.player_a)^2 * (1 - rates.player_b)^2

/-- Theorem stating the probabilities for the given free throw rates -/
theorem free_throw_probabilities (rates : FreeThrowRates) 
  (h1 : rates.player_a = 1/2) (h2 : rates.player_b = 2/5) : 
  prob_one_success rates = 1/2 ∧ prob_at_least_one_success rates = 91/100 := by
  sorry

#eval prob_one_success ⟨1/2, 2/5⟩
#eval prob_at_least_one_success ⟨1/2, 2/5⟩

end NUMINAMATH_CALUDE_free_throw_probabilities_l2316_231630


namespace NUMINAMATH_CALUDE_clue_represents_8671_l2316_231615

/-- Represents a mapping from characters to digits -/
def CharToDigitMap := Char → Nat

/-- Creates a mapping from the string "BEST OF LUCK" to digits 0-9 in order -/
def createBestOfLuckMap : CharToDigitMap :=
  fun c => match c with
    | 'B' => 0
    | 'E' => 1
    | 'S' => 2
    | 'T' => 3
    | 'O' => 4
    | 'F' => 5
    | 'L' => 6
    | 'U' => 7
    | 'C' => 8
    | 'K' => 9
    | _ => 0  -- Default case, should not be reached for valid inputs

/-- Converts a string to a number using the given character-to-digit mapping -/
def stringToNumber (map : CharToDigitMap) (s : String) : Nat :=
  s.foldl (fun acc c => 10 * acc + map c) 0

/-- Theorem: The code word "CLUE" represents the number 8671 -/
theorem clue_represents_8671 :
  stringToNumber createBestOfLuckMap "CLUE" = 8671 := by
  sorry

#eval stringToNumber createBestOfLuckMap "CLUE"

end NUMINAMATH_CALUDE_clue_represents_8671_l2316_231615


namespace NUMINAMATH_CALUDE_john_saturday_earnings_l2316_231607

/-- The amount of money John earned on Saturday -/
def saturday_earnings : ℝ := sorry

/-- The amount of money John earned on Sunday -/
def sunday_earnings : ℝ := sorry

/-- The amount of money John earned the previous weekend -/
def previous_weekend_earnings : ℝ := 20

/-- The cost of the pogo stick -/
def pogo_stick_cost : ℝ := 60

/-- The additional amount John needs to buy the pogo stick -/
def additional_needed : ℝ := 13

theorem john_saturday_earnings :
  saturday_earnings = 18 ∧
  sunday_earnings = saturday_earnings / 2 ∧
  previous_weekend_earnings + saturday_earnings + sunday_earnings = pogo_stick_cost - additional_needed :=
by sorry

end NUMINAMATH_CALUDE_john_saturday_earnings_l2316_231607


namespace NUMINAMATH_CALUDE_average_salary_l2316_231691

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 15000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

theorem average_salary : total_salary / num_people = 9000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_l2316_231691


namespace NUMINAMATH_CALUDE_complex_number_calculation_l2316_231694

theorem complex_number_calculation (z : ℂ) : z = 1 + I → (2 / z) + z^2 = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_calculation_l2316_231694


namespace NUMINAMATH_CALUDE_function_properties_l2316_231611

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h₁ : ∃ x, f x ≠ 0)
variable (h₂ : ∀ x, f (x + 3) = -f (3 - x))
variable (h₃ : ∀ x, f (x + 4) = -f (4 - x))

-- Theorem statement
theorem function_properties :
  (∀ x, f (-x) = -f x) ∧ (∃ p > 0, ∀ x, f (x + p) = f x) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2316_231611


namespace NUMINAMATH_CALUDE_sugar_sacks_weight_l2316_231695

theorem sugar_sacks_weight (x y : ℝ) 
  (h1 : y - x = 8)
  (h2 : x - 1 = 0.6 * (y + 1)) : 
  x + y = 40 := by
sorry

end NUMINAMATH_CALUDE_sugar_sacks_weight_l2316_231695


namespace NUMINAMATH_CALUDE_logo_scaling_l2316_231681

theorem logo_scaling (w h W : ℝ) (hw : w > 0) (hh : h > 0) (hW : W > 0) :
  let scale := W / w
  let H := scale * h
  (W / w = H / h) ∧ (H = (W / w) * h) := by sorry

end NUMINAMATH_CALUDE_logo_scaling_l2316_231681


namespace NUMINAMATH_CALUDE_wage_decrease_percentage_l2316_231626

theorem wage_decrease_percentage (wages_last_week : ℝ) (x : ℝ) : 
  wages_last_week > 0 →
  (0.2 * wages_last_week * (1 - x / 100) = 0.6999999999999999 * (0.2 * wages_last_week)) →
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_wage_decrease_percentage_l2316_231626


namespace NUMINAMATH_CALUDE_equal_share_theorem_l2316_231645

/-- Represents the number of candies each person has initially -/
structure CandyDistribution :=
  (mark : ℕ)
  (peter : ℕ)
  (john : ℕ)

/-- Calculates the number of candies each person gets after sharing equally -/
def share_candies (dist : CandyDistribution) : ℕ :=
  (dist.mark + dist.peter + dist.john) / 3

/-- Theorem: Given the initial candy distribution, prove that each person gets 30 candies after sharing -/
theorem equal_share_theorem (dist : CandyDistribution) 
  (h1 : dist.mark = 30)
  (h2 : dist.peter = 25)
  (h3 : dist.john = 35) :
  share_candies dist = 30 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_theorem_l2316_231645


namespace NUMINAMATH_CALUDE_B_subset_A_l2316_231654

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem B_subset_A : B ⊆ A := by sorry

end NUMINAMATH_CALUDE_B_subset_A_l2316_231654


namespace NUMINAMATH_CALUDE_john_test_scores_l2316_231619

theorem john_test_scores (total_tests : ℕ) (target_percentage : ℚ) 
  (tests_taken : ℕ) (tests_at_target : ℕ) : 
  total_tests = 60 →
  target_percentage = 85 / 100 →
  tests_taken = 40 →
  tests_at_target = 28 →
  (total_tests - tests_taken : ℕ) - 
    (↑total_tests * target_percentage - tests_at_target : ℚ).floor = 0 :=
by sorry

end NUMINAMATH_CALUDE_john_test_scores_l2316_231619


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2316_231609

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (80 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 16 / (80 - c) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2316_231609


namespace NUMINAMATH_CALUDE_shortest_altitude_of_special_triangle_l2316_231600

theorem shortest_altitude_of_special_triangle :
  ∀ (a b c h : ℝ),
  a = 9 ∧ b = 12 ∧ c = 15 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = (1/2) * c * h →
  h = 7.2 :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_special_triangle_l2316_231600


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_rational_square_minus_two_l2316_231692

theorem negation_of_existence (p : ℚ → Prop) : 
  (¬ ∃ x : ℚ, p x) ↔ (∀ x : ℚ, ¬ p x) := by sorry

theorem negation_of_rational_square_minus_two :
  (¬ ∃ x : ℚ, x^2 - 2 = 0) ↔ (∀ x : ℚ, x^2 - 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_rational_square_minus_two_l2316_231692


namespace NUMINAMATH_CALUDE_quadratic_sum_l2316_231675

/-- Given a quadratic function f(x) = ax^2 + bx + c where a = 2, b = -3, c = 4,
    and f(1) = 3, prove that 2a - b + c = 11 -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℤ) :
  a = 2 ∧ b = -3 ∧ c = 4 ∧
  (∀ x, f x = a * x^2 + b * x + c) ∧
  f 1 = 3 →
  2 * a - b + c = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2316_231675


namespace NUMINAMATH_CALUDE_equation_solutions_count_l2316_231604

theorem equation_solutions_count :
  ∃! (solutions : Finset ℝ),
    Finset.card solutions = 8 ∧
    ∀ θ ∈ solutions,
      0 < θ ∧ θ ≤ 2 * Real.pi ∧
      2 - 4 * Real.sin θ + 6 * Real.cos (2 * θ) + Real.sin (3 * θ) = 0 ∧
    ∀ θ : ℝ,
      0 < θ ∧ θ ≤ 2 * Real.pi ∧
      2 - 4 * Real.sin θ + 6 * Real.cos (2 * θ) + Real.sin (3 * θ) = 0 →
      θ ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l2316_231604


namespace NUMINAMATH_CALUDE_finley_tickets_l2316_231618

theorem finley_tickets (total_tickets : ℕ) (ratio_jensen : ℕ) (ratio_finley : ℕ) : 
  total_tickets = 400 →
  ratio_jensen = 4 →
  ratio_finley = 11 →
  (3 * total_tickets / 4) * ratio_finley / (ratio_jensen + ratio_finley) = 220 := by
  sorry

end NUMINAMATH_CALUDE_finley_tickets_l2316_231618


namespace NUMINAMATH_CALUDE_parabola_properties_l2316_231639

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem parabola_properties :
  (∃ (x y : ℝ), IsLocalMin f x ∧ f x = y ∧ x = -1 ∧ y = -4) ∧
  (∀ x : ℝ, x ≥ 2 → f x ≥ 5) ∧
  (∃ x : ℝ, x ≥ 2 ∧ f x = 5) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2316_231639


namespace NUMINAMATH_CALUDE_committee_formation_count_l2316_231693

theorem committee_formation_count : Nat.choose 15 6 = 5005 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l2316_231693


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2316_231624

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2316_231624


namespace NUMINAMATH_CALUDE_f_derivative_at_pi_half_l2316_231642

noncomputable def f (x : ℝ) : ℝ := x / Real.sin x

theorem f_derivative_at_pi_half :
  deriv f (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_pi_half_l2316_231642


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2316_231610

theorem sufficient_but_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → 1/a < 1/2) ∧ 
  (∃ a, 1/a < 1/2 ∧ ¬(a > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2316_231610


namespace NUMINAMATH_CALUDE_equilibrium_concentration_Ca_OH_2_l2316_231651

-- Define the reaction components
inductive Species
| CaO
| H2O
| Ca_OH_2

-- Define the reaction
def reaction : List Species := [Species.CaO, Species.H2O, Species.Ca_OH_2]

-- Define the equilibrium constant
def Kp : ℝ := 0.02

-- Define the equilibrium concentration function
noncomputable def equilibrium_concentration (s : Species) : ℝ :=
  match s with
  | Species.CaO => 0     -- Not applicable (solid)
  | Species.H2O => 0     -- Not applicable (liquid)
  | Species.Ca_OH_2 => Kp -- Equilibrium concentration equals Kp

-- Theorem statement
theorem equilibrium_concentration_Ca_OH_2 :
  equilibrium_concentration Species.Ca_OH_2 = Kp := by sorry

end NUMINAMATH_CALUDE_equilibrium_concentration_Ca_OH_2_l2316_231651


namespace NUMINAMATH_CALUDE_school_parade_l2316_231697

theorem school_parade (a b : ℕ+) : 
  ∃ k : ℕ, a.val * b.val * (a.val^2 - b.val^2) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_school_parade_l2316_231697


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l2316_231666

theorem square_sum_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 20) 
  (h2 : p + q = 10) : 
  p^2 + q^2 = 60 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l2316_231666


namespace NUMINAMATH_CALUDE_wiper_generates_sector_l2316_231698

/-- Represents a car wiper -/
structure CarWiper :=
  (length : ℝ)

/-- Represents a windshield -/
structure Windshield :=
  (width : ℝ)
  (height : ℝ)

/-- Represents a sector on a windshield -/
structure Sector :=
  (angle : ℝ)
  (radius : ℝ)

/-- The action of a car wiper on a windshield -/
def wiper_action (w : CarWiper) (s : Windshield) : Sector :=
  sorry

/-- States that a line (represented by a car wiper) generates a surface (represented by a sector) -/
theorem wiper_generates_sector (w : CarWiper) (s : Windshield) :
  ∃ (sector : Sector), wiper_action w s = sector :=
sorry

end NUMINAMATH_CALUDE_wiper_generates_sector_l2316_231698


namespace NUMINAMATH_CALUDE_sequence_non_positive_l2316_231625

theorem sequence_non_positive
  (n : ℕ)
  (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k : ℕ, k < n → a k.pred - 2 * a k + a k.succ ≥ 0) :
  ∀ k : ℕ, k ≤ n → a k ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l2316_231625


namespace NUMINAMATH_CALUDE_trapezoid_area_division_l2316_231627

/-- Given a trapezoid with specific properties, prove that the largest integer less than x^2/50 is 72 -/
theorem trapezoid_area_division (b : ℝ) (h : ℝ) (x : ℝ) : 
  b > 0 ∧ h > 0 ∧
  (b + 12.5) / (b + 37.5) = 3 / 5 ∧
  x > 0 ∧
  (25 + x) * ((x - 25) / 50) = 50 ∧
  x^2 - 75*x + 3125 = 0 →
  ⌊x^2 / 50⌋ = 72 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_division_l2316_231627


namespace NUMINAMATH_CALUDE_balls_satisfy_conditions_l2316_231647

/-- Represents a word in the Russian language -/
structure RussianWord where
  word : String

/-- Represents a festive dance event -/
structure FestiveDanceEvent where
  name : String

/-- Represents a sporting event -/
inductive SportingEvent
| FigureSkating
| RhythmicGymnastics
| Other

/-- Represents the Russian pension system -/
structure RussianPensionSystem where
  calculationMethod : String
  yearIntroduced : Nat

/-- Checks if a word sounds similar to a festive dance event -/
def soundsSimilarTo (w : RussianWord) (e : FestiveDanceEvent) : Prop :=
  sorry

/-- Checks if a word is used in a sporting event -/
def usedInSportingEvent (w : RussianWord) (e : SportingEvent) : Prop :=
  sorry

/-- Checks if a word is used in the Russian pension system -/
def usedInPensionSystem (w : RussianWord) (p : RussianPensionSystem) : Prop :=
  sorry

/-- The main theorem stating that "баллы" satisfies all conditions -/
theorem balls_satisfy_conditions :
  ∃ (w : RussianWord) (e : FestiveDanceEvent) (p : RussianPensionSystem),
    w.word = "баллы" ∧
    soundsSimilarTo w e ∧
    usedInSportingEvent w SportingEvent.FigureSkating ∧
    usedInSportingEvent w SportingEvent.RhythmicGymnastics ∧
    usedInPensionSystem w p ∧
    p.yearIntroduced = 2015 :=
  sorry


end NUMINAMATH_CALUDE_balls_satisfy_conditions_l2316_231647


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_range_of_m_l2316_231617

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def C (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 3 * m}

-- State the theorems
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < 5} := by sorry

theorem union_complement_A_B : (Set.univ \ A) ∪ B = {x | -2 < x ∧ x < 5} := by sorry

theorem range_of_m (m : ℝ) : B ∩ C m = C m → m < -1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_range_of_m_l2316_231617


namespace NUMINAMATH_CALUDE_geometric_mean_of_1_and_9_l2316_231682

theorem geometric_mean_of_1_and_9 : 
  ∃ (x : ℝ), x^2 = 1 * 9 ∧ (x = 3 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_1_and_9_l2316_231682


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_l2316_231656

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (notParallel : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicular (l m : Line) (α : Plane) :
  perpendicular l α → notParallel l m → perpendicular m α := by
  sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_l2316_231656


namespace NUMINAMATH_CALUDE_units_digit_of_p_plus_five_l2316_231613

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

def has_positive_units_digit (n : ℕ) : Prop := n % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_p_plus_five (p : ℕ) 
  (h_even : is_positive_even p)
  (h_pos_digit : has_positive_units_digit p)
  (h_cube_square : units_digit (p^3) = units_digit (p^2)) :
  units_digit (p + 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_p_plus_five_l2316_231613


namespace NUMINAMATH_CALUDE_parallelogram_sum_l2316_231635

/-- A parallelogram with sides 12, 4z + 2, 3x - 1, and 7y + 3 -/
structure Parallelogram (x y z : ℚ) where
  side1 : ℚ := 12
  side2 : ℚ := 4 * z + 2
  side3 : ℚ := 3 * x - 1
  side4 : ℚ := 7 * y + 3
  opposite_sides_equal1 : side1 = side3
  opposite_sides_equal2 : side2 = side4

/-- The sum of x, y, and z in the parallelogram equals 121/21 -/
theorem parallelogram_sum (x y z : ℚ) (p : Parallelogram x y z) : x + y + z = 121/21 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_sum_l2316_231635


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2316_231628

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :
  Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2316_231628


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2316_231684

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x| ≥ a * x) → |a| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2316_231684


namespace NUMINAMATH_CALUDE_kho_kho_only_count_l2316_231644

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 30

/-- The total number of players -/
def total_players : ℕ := 40

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 10

/-- The number of people who play both kabadi and kho kho -/
def both_players : ℕ := 5

/-- Theorem stating that the number of people who play kho kho only is 30 -/
theorem kho_kho_only_count :
  kho_kho_only = total_players - kabadi_players + both_players :=
by sorry

end NUMINAMATH_CALUDE_kho_kho_only_count_l2316_231644


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2316_231631

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a - b)^2 = 4*(a*b)^3) :
  (1/a + 1/b) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2316_231631


namespace NUMINAMATH_CALUDE_derivative_of_f_l2316_231661

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = (1 - log x) / (x^2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l2316_231661


namespace NUMINAMATH_CALUDE_rectangle_perimeter_product_l2316_231637

theorem rectangle_perimeter_product (a b c d : ℝ) : 
  (a + b = 11 ∧ a + b + c = 19.5 ∧ c = d) ∨
  (a + c = 11 ∧ a + b + c = 19.5 ∧ b = d) ∨
  (b + c = 11 ∧ a + b + c = 19.5 ∧ a = d) →
  (2 * (a + b)) * (2 * (a + c)) * (2 * (b + c)) = 15400 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_product_l2316_231637


namespace NUMINAMATH_CALUDE_equation_solution_l2316_231687

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ x ≠ -6 ∧
  (3*x - 6) / (x^2 + 5*x - 6) = (x + 3) / (x - 1) ∧
  x = 9/2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2316_231687


namespace NUMINAMATH_CALUDE_max_single_painted_face_theorem_l2316_231652

/-- Represents a large cube composed of smaller cubes -/
structure LargeCube where
  size : Nat
  painted_faces : Nat

/-- Calculates the maximum number of smaller cubes with exactly one face painted -/
def max_single_painted_face (cube : LargeCube) : Nat :=
  if cube.size = 4 ∧ cube.painted_faces = 3 then 32 else 0

/-- Theorem stating the maximum number of smaller cubes with exactly one face painted -/
theorem max_single_painted_face_theorem (cube : LargeCube) :
  cube.size = 4 ∧ cube.painted_faces = 3 →
  max_single_painted_face cube = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_single_painted_face_theorem_l2316_231652


namespace NUMINAMATH_CALUDE_prime_even_intersection_l2316_231634

def P : Set ℕ := {n : ℕ | Nat.Prime n}
def Q : Set ℕ := {n : ℕ | Even n}

theorem prime_even_intersection : P ∩ Q = {2} := by
  sorry

end NUMINAMATH_CALUDE_prime_even_intersection_l2316_231634


namespace NUMINAMATH_CALUDE_different_color_number_probability_l2316_231685

/-- Represents the total number of balls -/
def total_balls : ℕ := 9

/-- Represents the number of balls to be drawn -/
def drawn_balls : ℕ := 3

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the number of balls per color -/
def balls_per_color : ℕ := 3

/-- Represents the number of possible numbers on each ball -/
def num_numbers : ℕ := 3

/-- The probability of drawing 3 balls with different colors and numbers -/
def probability_different : ℚ := 1 / 14

theorem different_color_number_probability :
  (Nat.factorial num_colors) / (Nat.choose total_balls drawn_balls) = probability_different :=
sorry

end NUMINAMATH_CALUDE_different_color_number_probability_l2316_231685


namespace NUMINAMATH_CALUDE_equation_solution_l2316_231680

theorem equation_solution : ∃! x : ℚ, 3 * x - 5 = |-20 + 6| := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2316_231680


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l2316_231616

/-- Given two lines l₁ and l₂ in the xy-plane:
    l₁: mx + y - 1 = 0
    l₂: x - 2y + 5 = 0
    If l₁ is perpendicular to l₂, then m = 2. -/
theorem perpendicular_lines_slope (m : ℝ) : 
  (∀ x y, mx + y - 1 = 0 → x - 2*y + 5 = 0 → (mx + y - 1 = 0 ∧ x - 2*y + 5 = 0) → m = 2) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l2316_231616


namespace NUMINAMATH_CALUDE_lily_total_books_l2316_231621

def mike_books_tuesday : ℕ := 45
def corey_books_tuesday : ℕ := 2 * mike_books_tuesday
def mike_gave_to_lily : ℕ := 10
def corey_gave_to_lily : ℕ := mike_gave_to_lily + 15

theorem lily_total_books : mike_gave_to_lily + corey_gave_to_lily = 35 :=
by sorry

end NUMINAMATH_CALUDE_lily_total_books_l2316_231621


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_min_l2316_231663

/-- Given a parabola y = ax^2 + bx + c with positive integer coefficients that intersects
    the x-axis at two distinct points within distance 1 of the origin, 
    the sum of its coefficients is at least 11. -/
theorem parabola_coefficient_sum_min (a b c : ℕ+) 
  (h_distinct : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0)
  (h_distance : ∀ x : ℝ, a * x^2 + b * x + c = 0 → |x| < 1) :
  a + b + c ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_min_l2316_231663


namespace NUMINAMATH_CALUDE_triangle_satisfies_conditions_l2316_231683

/-- Triangle ABC with given properties --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  euler_line : ℝ → ℝ → Prop

/-- The specific triangle we're considering --/
def our_triangle : Triangle where
  A := (-4, 0)
  B := (0, 4)
  C := (0, -2)
  euler_line := fun x y => x - y + 2 = 0

/-- Theorem stating that the given triangle satisfies the conditions --/
theorem triangle_satisfies_conditions (t : Triangle) : 
  t.A = (-4, 0) ∧ 
  t.B = (0, 4) ∧ 
  t.C = (0, -2) ∧ 
  (∀ x y, t.euler_line x y ↔ x - y + 2 = 0) →
  t = our_triangle :=
sorry

end NUMINAMATH_CALUDE_triangle_satisfies_conditions_l2316_231683


namespace NUMINAMATH_CALUDE_congruence_solution_l2316_231641

theorem congruence_solution (x : ℤ) 
  (h1 : (2 + x) % (5^3) = 2^2 % (5^3))
  (h2 : (3 + x) % (7^3) = 3^2 % (7^3))
  (h3 : (4 + x) % (11^3) = 5^2 % (11^3)) :
  x % 385 = 307 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2316_231641


namespace NUMINAMATH_CALUDE_select_four_with_girl_l2316_231665

/-- The number of ways to select 4 people from 4 boys and 2 girls with at least one girl -/
def select_with_girl (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose boys to_select

theorem select_four_with_girl :
  select_with_girl 6 4 2 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_select_four_with_girl_l2316_231665


namespace NUMINAMATH_CALUDE_pi_is_irrational_l2316_231623

-- Define the property of being an infinite non-repeating decimal
def is_infinite_non_repeating_decimal (x : ℝ) : Prop := sorry

-- Define the property of being an irrational number
def is_irrational (x : ℝ) : Prop := sorry

-- State the theorem
theorem pi_is_irrational :
  is_infinite_non_repeating_decimal π →
  (∀ x : ℝ, is_infinite_non_repeating_decimal x → is_irrational x) →
  is_irrational π :=
by sorry

end NUMINAMATH_CALUDE_pi_is_irrational_l2316_231623


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2316_231601

-- Define the equations
def equation1 (x : ℝ) : Prop := (2*x - 5)/6 - (3*x + 1)/2 = 1
def equation2 (x : ℝ) : Prop := 3*x - 7*(x - 1) = 3 - 2*(x + 3)

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = -2 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2316_231601


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2316_231602

theorem min_value_quadratic :
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2316_231602


namespace NUMINAMATH_CALUDE_robin_water_bottles_l2316_231670

theorem robin_water_bottles (morning_bottles : ℕ) (afternoon_bottles : ℕ) 
  (h1 : morning_bottles = 7) 
  (h2 : afternoon_bottles = 7) : 
  morning_bottles + afternoon_bottles = 14 := by
  sorry

end NUMINAMATH_CALUDE_robin_water_bottles_l2316_231670


namespace NUMINAMATH_CALUDE_area_between_circles_l2316_231667

/-- Two circles with given properties -/
structure TwoCircles where
  r₁ : ℝ
  r₂ : ℝ
  radius_relation : r₁ = 3 * r₂
  chord_length : ℝ
  chord_length_eq : chord_length = 20

/-- The area between two circles with the given properties is 160π -/
theorem area_between_circles (c : TwoCircles) : 
  π * c.r₁^2 - π * c.r₂^2 = 160 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_l2316_231667


namespace NUMINAMATH_CALUDE_negation_of_implication_l2316_231655

theorem negation_of_implication (x : ℝ) :
  ¬(x > 0 → x^2 > 0) ↔ (x ≤ 0 → x^2 ≤ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2316_231655


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2316_231686

theorem gcd_of_three_numbers : Nat.gcd 13680 (Nat.gcd 20400 47600) = 80 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2316_231686


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2316_231638

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (3 * a^3 - 5 * a^2 + 170 * a - 7 = 0) →
  (3 * b^3 - 5 * b^2 + 170 * b - 7 = 0) →
  (3 * c^3 - 5 * c^2 + 170 * c - 7 = 0) →
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 
  (11/3 - c)^3 + (11/3 - a)^3 + (11/3 - b)^3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2316_231638


namespace NUMINAMATH_CALUDE_system_solution_l2316_231648

theorem system_solution :
  ∃ (x y : ℝ),
    (10 * x^2 + 5 * y^2 - 2 * x * y - 38 * x - 6 * y + 41 = 0) ∧
    (3 * x^2 - 2 * y^2 + 5 * x * y - 17 * x - 6 * y + 20 = 0) ∧
    (x = 2) ∧ (y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2316_231648


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l2316_231657

theorem opposite_of_negative_fraction :
  -(-(1 / 2023)) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l2316_231657


namespace NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l2316_231603

theorem tangent_line_to_x_ln_x (m : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (2 * x₀ + m = x₀ * Real.log x₀) ∧ 
    (2 = Real.log x₀ + 1)) → 
  m = -Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l2316_231603


namespace NUMINAMATH_CALUDE_divisor_cube_eq_four_n_l2316_231696

/-- The number of positive divisors of a positive integer -/
def d (n : ℕ+) : ℕ := sorry

/-- The theorem stating that the only positive integers n that satisfy d(n)^3 = 4n are 2, 128, and 2000 -/
theorem divisor_cube_eq_four_n : 
  ∀ n : ℕ+, d n ^ 3 = 4 * n ↔ n = 2 ∨ n = 128 ∨ n = 2000 := by sorry

end NUMINAMATH_CALUDE_divisor_cube_eq_four_n_l2316_231696


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2316_231676

noncomputable def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8| + 1

theorem sum_of_max_min_g : 
  ∃ (max_g min_g : ℝ), 
    (∀ x ∈ Set.Icc 3 7, g x ≤ max_g) ∧
    (∃ x ∈ Set.Icc 3 7, g x = max_g) ∧
    (∀ x ∈ Set.Icc 3 7, min_g ≤ g x) ∧
    (∃ x ∈ Set.Icc 3 7, g x = min_g) ∧
    max_g + min_g = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2316_231676


namespace NUMINAMATH_CALUDE_midpoint_property_l2316_231608

/-- Given two points D and E in the plane, if F is their midpoint, 
    then 3 times the x-coordinate of F minus 5 times the y-coordinate of F equals 4. -/
theorem midpoint_property (D E F : ℝ × ℝ) : 
  D = (15, 3) → 
  E = (6, 8) → 
  F.1 = (D.1 + E.1) / 2 →
  F.2 = (D.2 + E.2) / 2 →
  3 * F.1 - 5 * F.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_property_l2316_231608


namespace NUMINAMATH_CALUDE_unique_prime_square_sum_l2316_231669

theorem unique_prime_square_sum (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → ∃ (n : ℕ), p^(q+1) + q^(p+1) = n^2 → p = 2 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_square_sum_l2316_231669


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2316_231673

theorem binomial_expansion_coefficient (x : ℝ) :
  ∃ m : ℝ, (1 + 2*x)^3 = 1 + 6*x + m*x^2 + 8*x^3 ∧ m = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2316_231673


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2316_231612

def A : Set ℝ := {x | |x - 1| ≤ 1}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2316_231612


namespace NUMINAMATH_CALUDE_not_right_triangle_l2316_231664

theorem not_right_triangle (a b c : ℝ) : 
  (a = 3 ∧ b = 4 ∧ c = 5) ∨ 
  (a = 1 ∧ b = Real.sqrt 3 ∧ c = 2) ∨ 
  (a = Real.sqrt 11 ∧ b = 2 ∧ c = 4) ∨ 
  (a^2 = (c+b)*(c-b)) →
  (¬(a^2 + b^2 = c^2) ↔ a = Real.sqrt 11 ∧ b = 2 ∧ c = 4) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l2316_231664


namespace NUMINAMATH_CALUDE_linear_function_property_l2316_231643

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- Given a linear function g where g(8) - g(3) = 15, prove that g(20) - g(3) = 51. -/
theorem linear_function_property (g : ℝ → ℝ) (h1 : LinearFunction g) (h2 : g 8 - g 3 = 15) :
  g 20 - g 3 = 51 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l2316_231643


namespace NUMINAMATH_CALUDE_brianna_cd_purchase_l2316_231660

/-- Represents the fraction of Brianna's money used to buy half of the CDs -/
def money_fraction_for_half_cds : ℚ := 1/4

/-- Represents the fraction of CDs bought with one quarter of Brianna's money -/
def cd_fraction_bought : ℚ := 1/2

/-- Represents the fraction of money left after buying all CDs -/
def money_fraction_left : ℚ := 1/2

theorem brianna_cd_purchase :
  money_fraction_for_half_cds * (1 / cd_fraction_bought) = 1 - money_fraction_left := by
  sorry

end NUMINAMATH_CALUDE_brianna_cd_purchase_l2316_231660


namespace NUMINAMATH_CALUDE_simplify_expression_l2316_231678

theorem simplify_expression (z : ℝ) : (3 - 5 * z^2) - (5 + 7 * z^2) = -2 - 12 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2316_231678


namespace NUMINAMATH_CALUDE_tony_preparation_time_l2316_231622

/-- The total time Tony spent preparing to be an astronaut -/
def total_preparation_time (
  science_degree_time : ℝ
  ) (num_other_degrees : ℕ
  ) (physics_grad_time : ℝ
  ) (scientist_work_time : ℝ
  ) (num_internships : ℕ
  ) (internship_duration : ℝ
  ) : ℝ :=
  science_degree_time +
  num_other_degrees * science_degree_time +
  physics_grad_time +
  scientist_work_time +
  num_internships * internship_duration

/-- Theorem stating that Tony's total preparation time is 18.5 years -/
theorem tony_preparation_time :
  total_preparation_time 4 2 2 3 3 0.5 = 18.5 := by
  sorry


end NUMINAMATH_CALUDE_tony_preparation_time_l2316_231622
