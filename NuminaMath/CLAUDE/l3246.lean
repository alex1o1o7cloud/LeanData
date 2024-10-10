import Mathlib

namespace zero_vector_magnitude_is_zero_l3246_324606

/-- The magnitude of the zero vector in a 2D plane is 0. -/
theorem zero_vector_magnitude_is_zero :
  ∀ (v : ℝ × ℝ), v = (0, 0) → ‖v‖ = 0 := by
  sorry

end zero_vector_magnitude_is_zero_l3246_324606


namespace arithmetic_sequence_sum_l3246_324667

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 = 4 → a 2 = 6 →
  (a 1 + a 2 + a 3 + a 4 = 28) :=
by
  sorry

end arithmetic_sequence_sum_l3246_324667


namespace divisors_of_6440_l3246_324651

theorem divisors_of_6440 : 
  let n : ℕ := 6440
  let prime_factorization : List (ℕ × ℕ) := [(2, 3), (5, 1), (7, 1), (23, 1)]
  ∀ (is_valid_factorization : n = (List.foldl (λ acc (p, e) => acc * p^e) 1 prime_factorization)),
  (List.foldl (λ acc (_, e) => acc * (e + 1)) 1 prime_factorization) = 32 := by
sorry

end divisors_of_6440_l3246_324651


namespace escalator_steps_count_l3246_324678

/-- Represents the number of steps a person climbs on the escalator -/
structure ClimbingSteps where
  steps : ℕ

/-- Represents the speed at which a person climbs the escalator -/
structure ClimbingSpeed where
  speed : ℕ

/-- Represents a person climbing the escalator -/
structure Person where
  climbingSteps : ClimbingSteps
  climbingSpeed : ClimbingSpeed

/-- Calculates the total number of steps in the escalator -/
def escalatorSteps (personA personB : Person) : ℕ :=
  sorry

theorem escalator_steps_count
  (personA personB : Person)
  (hA : personA.climbingSteps.steps = 55)
  (hB : personB.climbingSteps.steps = 60)
  (hSpeed : personB.climbingSpeed.speed = 2 * personA.climbingSpeed.speed) :
  escalatorSteps personA personB = 66 :=
sorry

end escalator_steps_count_l3246_324678


namespace share_of_a_l3246_324604

def total : ℕ := 366

def shares (a b c : ℕ) : Prop :=
  a + b + c = total ∧
  a = (b + c) / 2 ∧
  b = (a + c) * 2 / 3

theorem share_of_a : ∃ a b c : ℕ, shares a b c ∧ a = 122 := by sorry

end share_of_a_l3246_324604


namespace samantha_route_count_l3246_324679

/-- Represents the number of ways to arrange k items out of n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of routes from Samantha's house to the southwest corner of City Park -/
def routes_to_park : ℕ := binomial 4 1

/-- The number of routes through City Park -/
def routes_through_park : ℕ := 1

/-- The number of routes from the northeast corner of City Park to school -/
def routes_to_school : ℕ := binomial 6 3

/-- The total number of possible routes Samantha can take -/
def total_routes : ℕ := routes_to_park * routes_through_park * routes_to_school

theorem samantha_route_count : total_routes = 80 := by sorry

end samantha_route_count_l3246_324679


namespace sandwich_bread_packs_l3246_324690

theorem sandwich_bread_packs (total_sandwiches : ℕ) (slices_per_sandwich : ℕ) (packs_bought : ℕ) :
  total_sandwiches = 8 →
  slices_per_sandwich = 2 →
  packs_bought = 4 →
  (total_sandwiches * slices_per_sandwich) / packs_bought = 4 :=
by
  sorry

end sandwich_bread_packs_l3246_324690


namespace biased_coin_probability_l3246_324628

def probability_of_heads (p : ℝ) (k : ℕ) (n : ℕ) : ℝ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem biased_coin_probability : 
  ∀ p : ℝ, 
  0 < p → p < 1 →
  probability_of_heads p 1 7 = probability_of_heads p 2 7 →
  probability_of_heads p 1 7 ≠ 0 →
  probability_of_heads p 4 7 = 945 / 16384 := by
sorry

end biased_coin_probability_l3246_324628


namespace mark_cans_proof_l3246_324653

/-- The number of cans Mark bought -/
def mark_cans : ℕ := 27

/-- The number of cans Jennifer initially bought -/
def jennifer_initial : ℕ := 40

/-- The total number of cans Jennifer brought home -/
def jennifer_total : ℕ := 100

/-- For every 5 cans Mark bought, Jennifer bought 11 cans -/
def jennifer_to_mark_ratio : ℚ := 11 / 5

theorem mark_cans_proof :
  (jennifer_total - jennifer_initial : ℚ) / jennifer_to_mark_ratio = mark_cans := by
  sorry

end mark_cans_proof_l3246_324653


namespace intersection_A_B_zero_union_A_B_equals_A_l3246_324635

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x < a + 5}

-- Theorem 1: Intersection of A and B when a = 0
theorem intersection_A_B_zero : A ∩ B 0 = {x | -1 < x ∧ x < 5} := by sorry

-- Theorem 2: Range of a for which A ∪ B = A
theorem union_A_B_equals_A (a : ℝ) : A ∪ B a = A ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ici 6 := by sorry

end intersection_A_B_zero_union_A_B_equals_A_l3246_324635


namespace evaluate_expression_l3246_324630

theorem evaluate_expression (x y z : ℚ) 
  (hx : x = 1/4) 
  (hy : y = 3/4) 
  (hz : z = -2) : 
  x^3 * y^2 * z^2 = 9/16 := by
sorry

end evaluate_expression_l3246_324630


namespace division_problem_l3246_324613

theorem division_problem (dividend : Nat) (divisor : Nat) (remainder : Nat) (quotient : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 34 →
  divisor = 7 →
  remainder = 6 →
  quotient = 4 := by
  sorry

end division_problem_l3246_324613


namespace project_completion_proof_l3246_324640

/-- The number of days Person A takes to complete the project alone -/
def person_a_days : ℕ := 20

/-- The number of days Person B takes to complete the project alone -/
def person_b_days : ℕ := 10

/-- The total number of days taken to complete the project -/
def total_days : ℕ := 12

/-- The number of days Person B worked alone -/
def person_b_worked_days : ℕ := 8

theorem project_completion_proof :
  (1 : ℚ) = (total_days - person_b_worked_days : ℚ) / person_a_days + 
            (person_b_worked_days : ℚ) / person_b_days :=
by sorry

end project_completion_proof_l3246_324640


namespace exam_score_problem_l3246_324648

theorem exam_score_problem (total_questions : ℕ) 
  (correct_score wrong_score total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 120 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_answers * correct_score + (total_questions - correct_answers) * wrong_score = total_score ∧
    correct_answers = 36 :=
by sorry

end exam_score_problem_l3246_324648


namespace confidence_level_error_probability_l3246_324637

/-- Represents the confidence level as a real number between 0 and 1 -/
def ConfidenceLevel : Type := { r : ℝ // 0 < r ∧ r < 1 }

/-- Represents the probability of making an incorrect inference -/
def ErrorProbability : Type := { r : ℝ // 0 ≤ r ∧ r ≤ 1 }

/-- Given a confidence level, calculates the probability of making an incorrect inference -/
def calculateErrorProbability (cl : ConfidenceLevel) : ErrorProbability :=
  sorry

theorem confidence_level_error_probability 
  (cl : ConfidenceLevel) 
  (hp : cl.val = 0.95) :
  (calculateErrorProbability cl).val = 0.05 := by
  sorry

end confidence_level_error_probability_l3246_324637


namespace orange_distribution_l3246_324676

-- Define the total number of oranges
def total_oranges : ℕ := 30

-- Define the number of people
def num_people : ℕ := 3

-- Define the minimum number of oranges each person must receive
def min_oranges : ℕ := 3

-- Define the function to calculate the number of ways to distribute oranges
def ways_to_distribute (total : ℕ) (people : ℕ) (min : ℕ) : ℕ :=
  Nat.choose (total - people * min + people - 1) (people - 1)

-- Theorem statement
theorem orange_distribution :
  ways_to_distribute total_oranges num_people min_oranges = 253 := by
  sorry

end orange_distribution_l3246_324676


namespace sum_20_terms_l3246_324609

/-- An arithmetic progression with the sum of its 4th and 12th terms equal to 20 -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_4_12 : a + 3*d + a + 11*d = 20  -- Sum of 4th and 12th terms is 20

/-- Theorem about the sum of first 20 terms of the arithmetic progression -/
theorem sum_20_terms (ap : ArithmeticProgression) :
  ∃ k : ℝ, k = 200 + 120 * ap.d ∧ 
  (∀ n : ℕ, n ≤ 20 → (n : ℝ) / 2 * (2 * ap.a + (n - 1) * ap.d) ≤ k) ∧
  (∀ ε > 0, ∃ n : ℕ, n ≤ 20 ∧ k - (n : ℝ) / 2 * (2 * ap.a + (n - 1) * ap.d) < ε) :=
by sorry


end sum_20_terms_l3246_324609


namespace reverse_two_digit_number_l3246_324614

/-- For a two-digit number with tens digit x and units digit y,
    the number formed by reversing its digits is 10y + x. -/
theorem reverse_two_digit_number (x y : ℕ) 
  (h1 : x ≥ 1 ∧ x ≤ 9) (h2 : y ≥ 0 ∧ y ≤ 9) : 
  (10 * y + x) = (10 * y + x) := by
  sorry

#check reverse_two_digit_number

end reverse_two_digit_number_l3246_324614


namespace triangle_properties_l3246_324687

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (3 * Real.cos B * Real.cos C + 1 = 3 * Real.sin B * Real.sin C + Real.cos (2 * A)) →
  (A = π / 3) ∧
  (a = 2 * Real.sqrt 3 → ∃ (max_value : ℝ), max_value = 4 * Real.sqrt 7 ∧
    ∀ (b' c' : ℝ), b' + 2 * c' ≤ max_value) :=
by sorry

end triangle_properties_l3246_324687


namespace problem_1_problem_2_problem_3_l3246_324699

-- Problem 1
theorem problem_1 (x : ℝ) (h : x^2 + x - 2 = 0) :
  x^2 + x + 2023 = 2025 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) (h : a + b = 5) :
  2*(a + b) - 4*a - 4*b + 21 = 11 := by sorry

-- Problem 3
theorem problem_3 (a b : ℝ) (h1 : a^2 + 3*a*b = 20) (h2 : b^2 + 5*a*b = 8) :
  2*a^2 - b^2 + a*b = 32 := by sorry

end problem_1_problem_2_problem_3_l3246_324699


namespace sin_cos_fourth_power_nonnegative_with_zero_l3246_324698

theorem sin_cos_fourth_power_nonnegative_with_zero (x : ℝ) :
  (∀ x, (Real.sin x + Real.cos x)^4 ≥ 0) ∧
  (∃ x, (Real.sin x + Real.cos x)^4 = 0) := by
sorry

end sin_cos_fourth_power_nonnegative_with_zero_l3246_324698


namespace marsh_birds_count_l3246_324662

theorem marsh_birds_count (geese ducks : ℕ) (h1 : geese = 58) (h2 : ducks = 37) :
  geese + ducks = 95 := by
  sorry

end marsh_birds_count_l3246_324662


namespace sphere_cylinder_volume_difference_l3246_324617

/-- The volume difference between a sphere and an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere : ℝ) (r_cylinder : ℝ) 
  (h_sphere : r_sphere = 7)
  (h_cylinder : r_cylinder = 4) :
  (4 / 3 * π * r_sphere^3) - (π * r_cylinder^2 * Real.sqrt (4 * r_sphere^2 - 4 * r_cylinder^2)) = 
  1372 * π / 3 - 32 * π * Real.sqrt 33 := by
  sorry

end sphere_cylinder_volume_difference_l3246_324617


namespace machine_ok_l3246_324643

/-- The nominal portion weight -/
def nominal_weight : ℝ := 390

/-- The greatest deviation from the mean among preserved measurements -/
def max_deviation : ℝ := 39

/-- Condition: The greatest deviation doesn't exceed 10% of the nominal weight -/
axiom deviation_within_limit : max_deviation ≤ 0.1 * nominal_weight

/-- Condition: Deviations of unreadable measurements are less than the max deviation -/
axiom unreadable_deviations_less : ∀ x : ℝ, x < max_deviation → x < nominal_weight - 380

/-- Definition: A machine requires repair if the standard deviation exceeds max_deviation -/
def requires_repair (std_dev : ℝ) : Prop := std_dev > max_deviation

/-- Theorem: The machine does not require repair -/
theorem machine_ok : ∃ std_dev : ℝ, std_dev ≤ max_deviation ∧ ¬(requires_repair std_dev) := by
  sorry


end machine_ok_l3246_324643


namespace power_of_difference_squared_l3246_324636

theorem power_of_difference_squared : (3^2 - 3)^2 = 36 := by
  sorry

end power_of_difference_squared_l3246_324636


namespace no_120_cents_combination_l3246_324697

/-- Represents the types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A selection of coins --/
def CoinSelection := List Coin

/-- Calculates the total value of a coin selection in cents --/
def totalValue (selection : CoinSelection) : ℕ :=
  selection.map coinValue |>.sum

/-- Theorem: It's impossible to select 6 coins with a total value of 120 cents --/
theorem no_120_cents_combination :
  ¬ ∃ (selection : CoinSelection), selection.length = 6 ∧ totalValue selection = 120 := by
  sorry

end no_120_cents_combination_l3246_324697


namespace cube_remainder_mod_nine_l3246_324692

theorem cube_remainder_mod_nine (n : ℤ) :
  (n % 9 = 1 ∨ n % 9 = 4 ∨ n % 9 = 7) → n^3 % 9 = 1 := by
  sorry

end cube_remainder_mod_nine_l3246_324692


namespace partnership_investment_timing_l3246_324611

/-- A partnership problem where three partners invest at different times --/
theorem partnership_investment_timing 
  (x : ℝ) -- A's investment
  (annual_gain : ℝ) -- Total annual gain
  (a_share : ℝ) -- A's share of the gain
  (h1 : annual_gain = 12000) -- Given annual gain
  (h2 : a_share = 4000) -- Given A's share
  (h3 : a_share / annual_gain = 1/3) -- A's share ratio
  : ∃ (m : ℝ), -- The number of months after which C invests
    (x * 12) / (x * 12 + 2*x * 6 + 3*x * (12 - m)) = 1/3 ∧ 
    m = 8 := by
  sorry

end partnership_investment_timing_l3246_324611


namespace point_placement_l3246_324641

theorem point_placement (x : ℕ) : 
  (9 * x - 8 = 82) → x = 10 := by
  sorry

end point_placement_l3246_324641


namespace jeans_pricing_l3246_324612

theorem jeans_pricing (manufacturing_cost : ℝ) (manufacturing_cost_pos : manufacturing_cost > 0) :
  let retail_price := manufacturing_cost * (1 + 0.4)
  let customer_price := retail_price * (1 + 0.1)
  (customer_price - manufacturing_cost) / manufacturing_cost = 0.54 := by
sorry

end jeans_pricing_l3246_324612


namespace variety_promotion_criterion_variety_B_more_suitable_l3246_324608

/-- Represents a rice variety with its yield statistics -/
structure RiceVariety where
  mean_yield : ℝ
  variance : ℝ

/-- Determines if a rice variety is more suitable for promotion based on yield stability -/
def more_suitable_for_promotion (a b : RiceVariety) : Prop :=
  a.mean_yield = b.mean_yield ∧ a.variance < b.variance

/-- Theorem stating that given two varieties with equal mean yields, 
    the one with lower variance is more suitable for promotion -/
theorem variety_promotion_criterion 
  (a b : RiceVariety) 
  (h_equal_means : a.mean_yield = b.mean_yield) 
  (h_lower_variance : a.variance < b.variance) : 
  more_suitable_for_promotion b a := by
  sorry

/-- The specific rice varieties from the problem -/
def variety_A : RiceVariety := ⟨1042, 6.5⟩
def variety_B : RiceVariety := ⟨1042, 1.2⟩

/-- Theorem applying the general criterion to the specific varieties -/
theorem variety_B_more_suitable : 
  more_suitable_for_promotion variety_B variety_A := by
  sorry

end variety_promotion_criterion_variety_B_more_suitable_l3246_324608


namespace cos_pi_half_plus_alpha_l3246_324623

theorem cos_pi_half_plus_alpha (α : Real) : 
  (∃ P : Real × Real, P.1 = -4/5 ∧ P.2 = 3/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) → 
  Real.cos (π/2 + α) = -3/5 := by
sorry

end cos_pi_half_plus_alpha_l3246_324623


namespace prob_sum_gt_five_l3246_324657

/-- The probability of rolling two dice and getting a sum greater than five -/
def prob_sum_greater_than_five : ℚ := 2/3

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is less than or equal to five -/
def outcomes_sum_le_five : ℕ := 12

theorem prob_sum_gt_five :
  prob_sum_greater_than_five = 1 - (outcomes_sum_le_five : ℚ) / total_outcomes :=
sorry

end prob_sum_gt_five_l3246_324657


namespace sum_after_decrease_l3246_324620

theorem sum_after_decrease (a b : ℤ) : 
  a + b = 100 → (a - 48) + b = 52 := by
  sorry

end sum_after_decrease_l3246_324620


namespace problem_1_l3246_324601

theorem problem_1 (x y : ℝ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 := by
  sorry

end problem_1_l3246_324601


namespace sams_income_l3246_324603

/-- Represents the income tax calculation for Sam's region -/
noncomputable def income_tax (q : ℝ) (income : ℝ) : ℝ :=
  0.01 * q * 30000 +
  0.01 * (q + 3) * (min 45000 (max 30000 income) - 30000) +
  0.01 * (q + 5) * (max 0 (income - 45000))

/-- Theorem stating Sam's annual income given the tax structure -/
theorem sams_income (q : ℝ) :
  ∃ (income : ℝ),
    income_tax q income = 0.01 * (q + 0.35) * income ∧
    income = 48376 :=
by sorry

end sams_income_l3246_324603


namespace writer_birthday_theorem_l3246_324666

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of leap years in a given range -/
def leapYearsInRange (startYear endYear : Nat) : Nat :=
  sorry

/-- Calculates the day of the week given a number of days before Friday -/
def dayBeforeFriday (days : Nat) : DayOfWeek :=
  sorry

theorem writer_birthday_theorem :
  let startYear := 1780
  let endYear := 2020
  let yearsDiff := endYear - startYear
  let leapYears := leapYearsInRange startYear endYear
  let regularYears := yearsDiff - leapYears
  let totalDaysBackward := regularYears + 2 * leapYears
  dayBeforeFriday (totalDaysBackward % 7) = DayOfWeek.Sunday :=
by sorry

end writer_birthday_theorem_l3246_324666


namespace intersection_when_m_is_two_subset_condition_l3246_324650

-- Define set A
def A : Set ℝ := {y | ∃ x, -13/2 ≤ x ∧ x ≤ 3/2 ∧ y = Real.sqrt (3 - 2*x)}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ m + 1}

-- Theorem 1: When m = 2, A ∩ B = [0, 3]
theorem intersection_when_m_is_two : 
  A ∩ B 2 = Set.Icc 0 3 := by sorry

-- Theorem 2: B ⊆ A if and only if m ≤ 1
theorem subset_condition : 
  ∀ m : ℝ, B m ⊆ A ↔ m ≤ 1 := by sorry

end intersection_when_m_is_two_subset_condition_l3246_324650


namespace probability_two_boys_three_girls_l3246_324616

def probability_boy_or_girl : ℝ := 0.5

def number_of_children : ℕ := 5

def number_of_boys : ℕ := 2

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_two_boys_three_girls :
  (binomial_coefficient number_of_children number_of_boys : ℝ) *
  probability_boy_or_girl ^ number_of_boys *
  probability_boy_or_girl ^ (number_of_children - number_of_boys) =
  0.3125 := by
  sorry

end probability_two_boys_three_girls_l3246_324616


namespace probability_less_than_20_l3246_324672

theorem probability_less_than_20 (total : ℕ) (more_than_30 : ℕ) (h1 : total = 160) (h2 : more_than_30 = 90) :
  let less_than_20 := total - more_than_30
  (less_than_20 : ℚ) / total = 7 / 16 := by
  sorry

end probability_less_than_20_l3246_324672


namespace rectangle_perimeter_l3246_324660

/-- A rectangular field with given area and width has a specific perimeter -/
theorem rectangle_perimeter (area width : ℝ) (h_area : area = 750) (h_width : width = 25) :
  2 * (area / width + width) = 110 := by
  sorry

end rectangle_perimeter_l3246_324660


namespace rectangle_longer_side_length_l3246_324688

/-- Given a circle of radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle is three times the area of the circle,
    prove that the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side_length (r : ℝ) (circle_area rectangle_area : ℝ) :
  r = 6 →
  circle_area = π * r^2 →
  rectangle_area = 3 * circle_area →
  ∃ (shorter_side longer_side : ℝ),
    shorter_side = 2 * r ∧
    rectangle_area = shorter_side * longer_side ∧
    longer_side = 9 * π :=
by sorry

end rectangle_longer_side_length_l3246_324688


namespace max_value_fraction_sum_l3246_324645

theorem max_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a / (a + 1) + b / (b + 1)) ≤ 2/3 ∧
  (a / (a + 1) + b / (b + 1) = 2/3 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end max_value_fraction_sum_l3246_324645


namespace carrots_picked_first_day_l3246_324607

theorem carrots_picked_first_day (carrots_thrown_out carrots_second_day total_carrots : ℕ) 
  (h1 : carrots_thrown_out = 4)
  (h2 : carrots_second_day = 46)
  (h3 : total_carrots = 61) :
  ∃ carrots_first_day : ℕ, 
    carrots_first_day + carrots_second_day - carrots_thrown_out = total_carrots ∧ 
    carrots_first_day = 19 := by
  sorry

end carrots_picked_first_day_l3246_324607


namespace equal_distance_to_axes_l3246_324624

theorem equal_distance_to_axes (m : ℝ) : 
  let M : ℝ × ℝ := (-3*m - 1, -2*m)
  (|M.1| = |M.2|) ↔ (m = -1/5 ∨ m = -1) := by
sorry

end equal_distance_to_axes_l3246_324624


namespace remainder_divisibility_l3246_324691

theorem remainder_divisibility (N : ℤ) : N % 17 = 2 → N % 357 = 2 := by
  sorry

end remainder_divisibility_l3246_324691


namespace fraction_division_problem_expression_evaluation_problem_l3246_324671

-- Problem 1
theorem fraction_division_problem :
  (3/4 - 7/8) / (-7/8) = 1 + 1/7 := by sorry

-- Problem 2
theorem expression_evaluation_problem :
  2^1 - |0 - 4| + (1/3) * (-3^2) = -5 := by sorry

end fraction_division_problem_expression_evaluation_problem_l3246_324671


namespace sqrt_problem_1_sqrt_problem_2_sqrt_problem_3_sqrt_problem_4_l3246_324665

-- 1. Prove that √18 - √32 + √2 = 0
theorem sqrt_problem_1 : Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 := by sorry

-- 2. Prove that (√27 - √12) / √3 = 1
theorem sqrt_problem_2 : (Real.sqrt 27 - Real.sqrt 12) / Real.sqrt 3 = 1 := by sorry

-- 3. Prove that √(1/6) + √24 - √600 = -43/6 * √6
theorem sqrt_problem_3 : Real.sqrt (1/6) + Real.sqrt 24 - Real.sqrt 600 = -43/6 * Real.sqrt 6 := by sorry

-- 4. Prove that (√3 + 1)(√3 - 1) = 2
theorem sqrt_problem_4 : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 2 := by sorry

end sqrt_problem_1_sqrt_problem_2_sqrt_problem_3_sqrt_problem_4_l3246_324665


namespace qin_jiushao_v3_l3246_324652

def qin_jiushao_algorithm (coeffs : List ℤ) (x : ℤ) : List ℤ :=
  let f := λ acc coeff => acc * x + coeff
  List.scanl f 0 coeffs.reverse

def polynomial : List ℤ := [64, -192, 240, -160, 60, -12, 1]

theorem qin_jiushao_v3 :
  (qin_jiushao_algorithm polynomial 2).get! 3 = -80 := by sorry

end qin_jiushao_v3_l3246_324652


namespace circle_family_properties_l3246_324625

-- Define the family of circles
def circle_family (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*a - 20 = 0

-- Define the fixed circle
def fixed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem circle_family_properties :
  (∀ a : ℝ, circle_family a 4 (-2)) ∧ 
  (circle_family (1 + Real.sqrt 5 / 5) = fixed_circle) ∧
  (circle_family (1 - Real.sqrt 5 / 5) = fixed_circle) :=
sorry

end circle_family_properties_l3246_324625


namespace allocation_ways_l3246_324634

theorem allocation_ways (n : ℕ) (k : ℕ) (h : n = 8 ∧ k = 4) : k^n = 65536 := by
  sorry

end allocation_ways_l3246_324634


namespace trigonometric_identity_l3246_324629

theorem trigonometric_identity (x : ℝ) :
  (1 / Real.cos (2022 * x) + Real.tan (2022 * x) = 1 / 2022) →
  (1 / Real.cos (2022 * x) - Real.tan (2022 * x) = 2022) := by
  sorry

end trigonometric_identity_l3246_324629


namespace horner_method_for_f_l3246_324656

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def f (x : ℝ) : ℝ := 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem horner_method_for_f :
  f 2 = horner_polynomial [1, 2, 0, 3, 0, 0, 5] 2 ∧ 
  horner_polynomial [1, 2, 0, 3, 0, 0, 5] 2 = 373 := by
  sorry

end horner_method_for_f_l3246_324656


namespace investment_proof_l3246_324682

/-- Represents the monthly interest rate as a decimal -/
def monthly_interest_rate : ℝ := 0.10

/-- Calculates the total amount after n months given an initial investment -/
def total_after_n_months (initial_investment : ℝ) (n : ℕ) : ℝ :=
  initial_investment * (1 + monthly_interest_rate) ^ n

/-- Theorem stating that an initial investment of $300 results in $363 after 2 months -/
theorem investment_proof :
  ∃ (initial_investment : ℝ),
    total_after_n_months initial_investment 2 = 363 ∧
    initial_investment = 300 :=
by
  sorry


end investment_proof_l3246_324682


namespace subset_implies_m_leq_3_l3246_324695

/-- Set A definition -/
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

/-- Set B definition -/
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

/-- Theorem stating that if B is a subset of A, then m ≤ 3 -/
theorem subset_implies_m_leq_3 (m : ℝ) (h : B m ⊆ A) : m ≤ 3 := by
  sorry

end subset_implies_m_leq_3_l3246_324695


namespace carbon_emissions_solution_l3246_324605

theorem carbon_emissions_solution :
  ∃! (x y : ℝ), x + y = 70 ∧ x = 5 * y - 8 ∧ x = 57 ∧ y = 13 := by
  sorry

end carbon_emissions_solution_l3246_324605


namespace complex_sum_magnitude_l3246_324683

theorem complex_sum_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs (z₁ - z₂) = Real.sqrt 3) : 
  Complex.abs (z₁ + z₂) = 1 := by sorry

end complex_sum_magnitude_l3246_324683


namespace extra_legs_count_l3246_324694

/-- Represents the number of legs for a cow -/
def cow_legs : ℕ := 4

/-- Represents the number of legs for a chicken -/
def chicken_legs : ℕ := 2

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 9

theorem extra_legs_count (num_chickens : ℕ) : 
  cow_legs * num_cows + chicken_legs * num_chickens = 
  2 * (num_cows + num_chickens) + 18 := by
  sorry

end extra_legs_count_l3246_324694


namespace lineup_count_l3246_324675

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_members : ℕ) (offensive_linemen : ℕ) (quarterbacks : ℕ) : ℕ :=
  let remaining := total_members - offensive_linemen - quarterbacks
  offensive_linemen * quarterbacks * remaining * (remaining - 1) * (remaining - 2)

/-- Theorem stating that the number of ways to choose the starting lineup is 5760 -/
theorem lineup_count :
  choose_lineup 12 4 2 = 5760 :=
by sorry

end lineup_count_l3246_324675


namespace expression_evaluation_l3246_324658

theorem expression_evaluation (a b c : ℚ) : 
  a = 5 → 
  b = a + 4 → 
  c = b - 12 → 
  a + 2 ≠ 0 → 
  b - 3 ≠ 0 → 
  c + 7 ≠ 0 → 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 7 / 3 :=
by sorry

end expression_evaluation_l3246_324658


namespace correct_horses_b_l3246_324627

/-- Represents the number of horses put in the pasture by party b -/
def horses_b : ℕ := 6

/-- Represents the total cost of the pasture -/
def total_cost : ℕ := 870

/-- Represents the amount b should pay -/
def b_payment : ℕ := 360

/-- Represents the number of horses put in by party a -/
def horses_a : ℕ := 12

/-- Represents the number of months horses from party a stayed -/
def months_a : ℕ := 8

/-- Represents the number of months horses from party b stayed -/
def months_b : ℕ := 9

/-- Represents the number of horses put in by party c -/
def horses_c : ℕ := 18

/-- Represents the number of months horses from party c stayed -/
def months_c : ℕ := 6

theorem correct_horses_b :
  (horses_b * months_b : ℚ) / (horses_a * months_a + horses_b * months_b + horses_c * months_c) * total_cost = b_payment :=
by sorry

end correct_horses_b_l3246_324627


namespace square_plus_reciprocal_square_l3246_324644

theorem square_plus_reciprocal_square (a : ℝ) (h : (a + 1/a)^4 = 5) :
  a^2 + 1/a^2 = Real.sqrt 5 - 2 := by
  sorry

end square_plus_reciprocal_square_l3246_324644


namespace initial_books_on_shelf_l3246_324642

theorem initial_books_on_shelf (books_taken : ℕ) (books_left : ℕ) : 
  books_taken = 10 → books_left = 28 → books_taken + books_left = 38 := by
  sorry

end initial_books_on_shelf_l3246_324642


namespace functions_properties_l3246_324649

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (ω * x + φ)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * Real.cos (ω * x)

theorem functions_properties (ω φ : ℝ) :
  ω > 0 ∧
  0 ≤ φ ∧ φ < π ∧
  (∀ x : ℝ, f ω φ (x + π / ω) = f ω φ x) ∧
  (∀ x : ℝ, g ω (x + π / ω) = g ω x) ∧
  f ω φ (-π/6) + g ω (-π/6) = 0 →
  ω = 2 ∧
  φ = π/6 ∧
  ∀ x : ℝ, f ω φ x + g ω x = Real.sqrt 6 * Real.sin (2 * x + π/3) := by
sorry

end functions_properties_l3246_324649


namespace number_of_skirts_l3246_324619

theorem number_of_skirts (total_ways : ℕ) (num_pants : ℕ) : 
  total_ways = 7 → num_pants = 4 → total_ways - num_pants = 3 := by
  sorry

end number_of_skirts_l3246_324619


namespace faster_increase_l3246_324668

-- Define the functions
def y₁ (x : ℝ) : ℝ := 100 * x
def y₂ (x : ℝ) : ℝ := 1000 + 100 * x
def y₃ (x : ℝ) : ℝ := 10000 + 99 * x

-- State the theorem
theorem faster_increase : 
  (∀ x : ℝ, (deriv y₁ x) = (deriv y₂ x)) ∧ 
  (∀ x : ℝ, (deriv y₁ x) > (deriv y₃ x)) := by
  sorry

end faster_increase_l3246_324668


namespace range_of_k_for_quadratic_inequality_l3246_324684

theorem range_of_k_for_quadratic_inequality :
  {k : ℝ | ∀ x : ℝ, k * x^2 - k * x - 1 < 0} = {k : ℝ | -4 < k ∧ k ≤ 0} :=
by sorry

end range_of_k_for_quadratic_inequality_l3246_324684


namespace f_not_in_second_quadrant_l3246_324680

/-- The function f(x) = x - 2 -/
def f (x : ℝ) : ℝ := x - 2

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem f_not_in_second_quadrant :
  ∀ x : ℝ, ¬(in_second_quadrant x (f x)) :=
by
  sorry

end f_not_in_second_quadrant_l3246_324680


namespace complex_number_quadrant_l3246_324615

theorem complex_number_quadrant (z : ℂ) (h : z * (1 + Complex.I) = 1 + 2 * Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end complex_number_quadrant_l3246_324615


namespace sin_2phi_value_l3246_324677

theorem sin_2phi_value (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := by
sorry

end sin_2phi_value_l3246_324677


namespace equation_solution_l3246_324685

theorem equation_solution (m n : ℚ) : 
  (m * 1 + n * 1 = 6) → 
  (m * 2 + n * (-1) = 6) → 
  (m = 4 ∧ n = 2) := by
sorry

end equation_solution_l3246_324685


namespace opposite_points_probability_l3246_324674

theorem opposite_points_probability (n : ℕ) (h : n = 12) : 
  (n / 2) / (n.choose 2) = 1 / 11 := by
  sorry

end opposite_points_probability_l3246_324674


namespace min_ab_value_l3246_324633

-- Define the triangle ABC
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Define the condition 2c * cos B = 2a + b
def condition1 (a b c : ℝ) (B : ℝ) : Prop :=
  2 * c * Real.cos B = 2 * a + b

-- Define the area condition S = (√3/2) * c
def condition2 (c : ℝ) (S : ℝ) : Prop :=
  S = (Real.sqrt 3 / 2) * c

-- Theorem statement
theorem min_ab_value (a b c : ℝ) (B : ℝ) (S : ℝ) :
  triangle a b c →
  condition1 a b c B →
  condition2 c S →
  a * b ≥ 12 :=
sorry

end min_ab_value_l3246_324633


namespace dog_pickup_duration_l3246_324663

-- Define the time in minutes for each activity
def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_cleaning_time : ℕ := 10
def cooking_time : ℕ := 90

-- Define the total time from work end to dinner
def total_time : ℕ := 180

-- Define the time to pick up the dog (unknown)
def dog_pickup_time : ℕ := total_time - (commute_time + grocery_time + dry_cleaning_time + cooking_time)

-- Theorem to prove
theorem dog_pickup_duration : dog_pickup_time = 20 := by
  sorry

end dog_pickup_duration_l3246_324663


namespace trajectory_classification_l3246_324600

/-- The trajectory of a point P(x,y) satisfying |PF₁| + |PF₂| = 2a, where F₁(-5,0) and F₂(5,0) are fixed points -/
def trajectory (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p (-5, 0) + dist p (5, 0) = 2 * a}

/-- The distance between F₁ and F₂ -/
def f₁f₂_distance : ℝ := 10

theorem trajectory_classification (a : ℝ) (h : a > 0) :
  (a = f₁f₂_distance → trajectory a = {p : ℝ × ℝ | p.1 ∈ Set.Icc (-5 : ℝ) 5 ∧ p.2 = 0}) ∧
  (a > f₁f₂_distance → ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ trajectory a = {p : ℝ × ℝ | (p.1 / c)^2 + (p.2 / d)^2 = 1}) ∧
  (a < f₁f₂_distance → trajectory a = ∅) :=
sorry

end trajectory_classification_l3246_324600


namespace original_sales_tax_percentage_l3246_324654

/-- Proves that the original sales tax percentage was 5% given the conditions of the problem -/
theorem original_sales_tax_percentage
  (item_price : ℝ)
  (reduced_tax_rate : ℝ)
  (tax_difference : ℝ)
  (h1 : item_price = 1000)
  (h2 : reduced_tax_rate = 0.04)
  (h3 : tax_difference = 10)
  (h4 : item_price * reduced_tax_rate + tax_difference = item_price * (original_tax_rate / 100)) :
  original_tax_rate = 5 := by
  sorry

end original_sales_tax_percentage_l3246_324654


namespace cupboard_cost_price_l3246_324646

theorem cupboard_cost_price (selling_price selling_price_increased : ℝ) 
  (h1 : selling_price = 0.84 * 3750)
  (h2 : selling_price_increased = 1.16 * 3750)
  (h3 : selling_price_increased = selling_price + 1200) : 
  ∃ (cost_price : ℝ), cost_price = 3750 := by
sorry

end cupboard_cost_price_l3246_324646


namespace linear_equation_condition_l3246_324647

/-- If mx^(m+2) + m - 2 = 0 is a linear equation with respect to x, then m = -1 -/
theorem linear_equation_condition (m : ℝ) : 
  (∃ a b, ∀ x, m * x^(m + 2) + m - 2 = a * x + b) → m = -1 :=
by sorry

end linear_equation_condition_l3246_324647


namespace rectangle_count_in_grid_l3246_324659

/-- The number of dots in each row and column of the square array -/
def gridSize : Nat := 5

/-- The number of different rectangles that can be formed in the grid -/
def numRectangles : Nat := (gridSize.choose 2) * (gridSize.choose 2)

theorem rectangle_count_in_grid : numRectangles = 100 := by sorry

end rectangle_count_in_grid_l3246_324659


namespace power_equation_l3246_324621

/-- Given a real number a and integers m and n such that a^m = 2 and a^n = 5,
    prove that a^(3m+2n) = 200 -/
theorem power_equation (a : ℝ) (m n : ℤ) (h1 : a^m = 2) (h2 : a^n = 5) :
  a^(3*m + 2*n) = 200 := by
  sorry

end power_equation_l3246_324621


namespace relationship_abc_l3246_324670

theorem relationship_abc : 
  let a : ℝ := (1/2)^2
  let b : ℝ := 2^(1/2)
  let c : ℝ := Real.log 2 / Real.log (1/2)
  c < a ∧ a < b := by sorry

end relationship_abc_l3246_324670


namespace track_width_l3246_324610

theorem track_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) : 
  r₁ - r₂ = 10 := by
sorry

end track_width_l3246_324610


namespace arithmetic_sequence_common_difference_l3246_324673

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_diff : a 7 - 2 * a 4 = 6) 
  (h_third : a 3 = 2) : 
  ∃ d : ℝ, d = 4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l3246_324673


namespace max_value_abc_l3246_324696

theorem max_value_abc (a b c : ℝ) (h : a + 3*b + c = 6) :
  ∃ m : ℝ, m = 8 ∧ ∀ x y z : ℝ, x + 3*y + z = 6 → x*y + x*z + y*z ≤ m :=
sorry

end max_value_abc_l3246_324696


namespace birds_on_fence_l3246_324661

theorem birds_on_fence (initial_birds : ℕ) (initial_storks : ℕ) (additional_birds : ℕ) : 
  initial_birds = 2 →
  initial_storks = 6 →
  initial_storks = (initial_birds + additional_birds + 1) →
  additional_birds = 3 := by
sorry

end birds_on_fence_l3246_324661


namespace trigonometric_equation_solution_l3246_324664

theorem trigonometric_equation_solution (t : ℝ) : 
  (2 * (Real.sin (2 * t))^5 - (Real.sin (2 * t))^3 - 6 * (Real.sin (2 * t))^2 + 3 = 0) ↔ 
  (∃ k : ℤ, t = (π / 8) * (2 * ↑k + 1)) :=
by sorry

end trigonometric_equation_solution_l3246_324664


namespace tetrahedron_triangles_l3246_324669

/-- The number of vertices in a regular tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a regular tetrahedron -/
def distinct_triangles : ℕ := Nat.choose tetrahedron_vertices triangle_vertices

theorem tetrahedron_triangles :
  distinct_triangles = 4 :=
sorry

end tetrahedron_triangles_l3246_324669


namespace new_manufacturing_cost_l3246_324632

/-- Given a constant selling price, an initial manufacturing cost, and profit percentages,
    calculate the new manufacturing cost after a change in profit percentage. -/
theorem new_manufacturing_cost
  (P : ℝ)  -- Selling price
  (initial_cost : ℝ)  -- Initial manufacturing cost
  (initial_profit_percent : ℝ)  -- Initial profit as a percentage of selling price
  (new_profit_percent : ℝ)  -- New profit as a percentage of selling price
  (h1 : initial_cost = 80)  -- Initial cost is $80
  (h2 : initial_profit_percent = 0.20)  -- Initial profit is 20%
  (h3 : new_profit_percent = 0.50)  -- New profit is 50%
  (h4 : P - initial_cost = initial_profit_percent * P)  -- Initial profit equation
  : P - new_profit_percent * P = 50 := by
  sorry


end new_manufacturing_cost_l3246_324632


namespace min_value_expression_l3246_324631

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y)
  (h_sum : x + y + 1/x + 1/y = 2022) :
  (x + 1/y) * (x + 1/y - 2016) + (y + 1/x) * (y + 1/x - 2016) ≥ -2032188 :=
by sorry

end min_value_expression_l3246_324631


namespace scientific_notation_correct_l3246_324602

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number 188 million -/
def number : ℝ := 188000000

/-- The scientific notation representation of 188 million -/
def scientificForm : ScientificNotation :=
  { coefficient := 1.88
    exponent := 8
    coeff_range := by sorry }

theorem scientific_notation_correct :
  number = scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent := by sorry

end scientific_notation_correct_l3246_324602


namespace collinear_points_b_value_l3246_324622

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_b_value :
  ∀ b : ℝ,
  let A : Point := ⟨3, 1⟩
  let B : Point := ⟨-2, b⟩
  let C : Point := ⟨8, 11⟩
  collinear A B C → b = -9 := by
  sorry

end collinear_points_b_value_l3246_324622


namespace factorization_proof_l3246_324638

theorem factorization_proof (x y m n : ℝ) : 
  x^2 * (m - n) + y^2 * (n - m) = (m - n) * (x + y) * (x - y) := by
sorry

end factorization_proof_l3246_324638


namespace tourist_distribution_count_l3246_324686

def num_guides : ℕ := 3
def num_tourists : ℕ := 8

theorem tourist_distribution_count :
  (3^8 : ℕ) - num_guides * (2^8 : ℕ) + (num_guides.choose 2) * (1^8 : ℕ) = 5796 :=
sorry

end tourist_distribution_count_l3246_324686


namespace wire_cut_ratio_l3246_324693

theorem wire_cut_ratio (x y : ℝ) : 
  x > 0 → y > 0 → -- Ensure positive lengths
  (4 * (x / 4) = 5 * (y / 5)) → -- Equal perimeters condition
  x / y = 1 := by
sorry

end wire_cut_ratio_l3246_324693


namespace perpendicular_lines_a_value_l3246_324655

-- Define the coefficients of the lines
def l1_coeff (a : ℝ) : ℝ × ℝ := (a + 2, 1 - a)
def l2_coeff (a : ℝ) : ℝ × ℝ := (a - 1, 2*a + 3)

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop :=
  (l1_coeff a).1 * (l2_coeff a).1 + (l1_coeff a).2 * (l2_coeff a).2 = 0

-- Theorem statement
theorem perpendicular_lines_a_value (a : ℝ) :
  perpendicular a → a = 1 ∨ a = -1 := by
  sorry

end perpendicular_lines_a_value_l3246_324655


namespace arithmetic_sequence_common_difference_l3246_324681

/-- Given an arithmetic sequence {aₙ} with a₃ = 0 and a₁ = 4, 
    the common difference d is -2. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : a 3 = 0) 
  (h2 : a 1 = 4) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) :
  a 2 - a 1 = -2 := by
  sorry

end arithmetic_sequence_common_difference_l3246_324681


namespace sandwiches_problem_l3246_324618

def sandwiches_left (initial : ℕ) (ruth_ate : ℕ) (brother_given : ℕ) (first_cousin_ate : ℕ) (other_cousins_ate : ℕ) : ℕ :=
  initial - ruth_ate - brother_given - first_cousin_ate - other_cousins_ate

theorem sandwiches_problem :
  sandwiches_left 10 1 2 2 2 = 3 := by
  sorry

end sandwiches_problem_l3246_324618


namespace complex_square_simplification_l3246_324689

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 25 - 24 * i :=
by sorry

end complex_square_simplification_l3246_324689


namespace line_parameterization_l3246_324639

/-- Given a line y = 2x - 40 parameterized by (x, y) = (g(t), 20t - 14),
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t x y : ℝ, y = 2*x - 40 ∧ x = g t ∧ y = 20*t - 14) →
  (∀ t : ℝ, g t = 10*t + 13) :=
by sorry

end line_parameterization_l3246_324639


namespace square_side_length_average_l3246_324626

theorem square_side_length_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end square_side_length_average_l3246_324626
