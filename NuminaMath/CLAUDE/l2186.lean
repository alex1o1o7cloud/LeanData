import Mathlib

namespace NUMINAMATH_CALUDE_meeting_impossible_l2186_218691

-- Define the type for people in the meeting
def Person : Type := ℕ

-- Define the relationship of knowing each other
def knows (p q : Person) : Prop := sorry

-- Define the number of people in the meeting
def num_people : ℕ := 65

-- State the conditions of the problem
axiom condition1 : ∀ p : Person, ∃ S : Finset Person, S.card ≥ 56 ∧ ∀ q ∈ S, ¬knows p q

axiom condition2 : ∀ p q : Person, p ≠ q → ∃ r : Person, r ≠ p ∧ r ≠ q ∧ knows r p ∧ knows r q

-- The theorem to be proved
theorem meeting_impossible : False := sorry

end NUMINAMATH_CALUDE_meeting_impossible_l2186_218691


namespace NUMINAMATH_CALUDE_decimal_representation_symmetry_l2186_218665

/-- The main period of the decimal representation of 1/p -/
def decimal_period (p : ℕ) : List ℕ :=
  sorry

/-- Count occurrences of a digit in a list -/
def count_occurrences (digit : ℕ) (l : List ℕ) : ℕ :=
  sorry

theorem decimal_representation_symmetry (p n : ℕ) (h1 : Nat.Prime p) (h2 : p ∣ 10^n + 1) :
  ∀ i ∈ Finset.range 10,
    count_occurrences i (decimal_period p) = count_occurrences (9 - i) (decimal_period p) :=
by sorry

end NUMINAMATH_CALUDE_decimal_representation_symmetry_l2186_218665


namespace NUMINAMATH_CALUDE_brandon_application_theorem_l2186_218614

/-- The number of businesses Brandon can still apply to -/
def businesses_can_apply (total : ℕ) (fired : ℕ) (quit : ℕ) (x : ℕ) (y : ℕ) : ℕ :=
  total - (fired + quit - x) + y

theorem brandon_application_theorem (x y : ℕ) :
  businesses_can_apply 72 36 24 x y = 12 + x + y := by
  sorry

end NUMINAMATH_CALUDE_brandon_application_theorem_l2186_218614


namespace NUMINAMATH_CALUDE_jessie_initial_weight_l2186_218686

/-- Represents Jessie's weight change after jogging --/
structure WeightChange where
  lost : ℕ      -- Weight lost in kilograms
  current : ℕ   -- Current weight in kilograms

/-- Calculates the initial weight before jogging --/
def initial_weight (w : WeightChange) : ℕ :=
  w.lost + w.current

/-- Theorem stating Jessie's initial weight was 192 kg --/
theorem jessie_initial_weight :
  let w : WeightChange := { lost := 126, current := 66 }
  initial_weight w = 192 := by
  sorry

end NUMINAMATH_CALUDE_jessie_initial_weight_l2186_218686


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_six_l2186_218671

-- Define the polynomials
def p (x : ℝ) : ℝ := -3*x^3 - 8*x^2 + 3*x + 2
def q (x : ℝ) : ℝ := -2*x^2 - 7*x - 4

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p x * q x

-- Theorem statement
theorem coefficient_of_x_cubed_is_six :
  ∃ (a b c d : ℝ), product = fun x ↦ 6*x^3 + a*x^2 + b*x + c + d*x^4 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_six_l2186_218671


namespace NUMINAMATH_CALUDE_mike_score_is_99_percent_l2186_218660

/-- Represents the exam scores of four students -/
structure ExamScores where
  gibi : ℝ
  jigi : ℝ
  mike : ℝ
  lizzy : ℝ

/-- Theorem stating that Mike's score is 99% given the conditions -/
theorem mike_score_is_99_percent 
  (scores : ExamScores)
  (h_gibi : scores.gibi = 59)
  (h_jigi : scores.jigi = 55)
  (h_lizzy : scores.lizzy = 67)
  (h_max_score : ℝ := 700)
  (h_average : (scores.gibi + scores.jigi + scores.mike + scores.lizzy) / 4 * h_max_score / 100 = 490) :
  scores.mike = 99 := by
sorry

end NUMINAMATH_CALUDE_mike_score_is_99_percent_l2186_218660


namespace NUMINAMATH_CALUDE_sugar_in_house_l2186_218616

/-- Given the total sugar needed and additional sugar needed, prove the amount of sugar stored in the house. -/
theorem sugar_in_house (total_sugar : ℕ) (additional_sugar : ℕ) 
  (h1 : total_sugar = 450)
  (h2 : additional_sugar = 163) :
  total_sugar - additional_sugar = 287 := by
  sorry

end NUMINAMATH_CALUDE_sugar_in_house_l2186_218616


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2186_218668

theorem expand_and_simplify (x : ℝ) : (x + 5) * (4 * x - 9 - 3) = 4 * x^2 + 8 * x - 60 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2186_218668


namespace NUMINAMATH_CALUDE_sum_of_squares_l2186_218693

theorem sum_of_squares (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (eq_a : a + a^2 = 1) (eq_b : b^2 + b^4 = 1) : a^2 + b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2186_218693


namespace NUMINAMATH_CALUDE_total_production_cost_l2186_218651

def initial_cost_per_episode : ℕ := 100000
def cost_increase_rate : ℚ := 1.2
def initial_episodes : ℕ := 12
def season_2_increase : ℚ := 1.3
def subsequent_seasons_increase : ℚ := 1.1
def final_season_decrease : ℚ := 0.85
def total_seasons : ℕ := 7

def calculate_total_cost : ℕ := sorry

theorem total_production_cost :
  calculate_total_cost = 25673856 := by sorry

end NUMINAMATH_CALUDE_total_production_cost_l2186_218651


namespace NUMINAMATH_CALUDE_problem_statements_l2186_218622

noncomputable section

variable (k : ℝ)

def f (x : ℝ) : ℝ := Real.log x

def g (x : ℝ) : ℝ := x^2 + k*x

def a (x₁ x₂ : ℝ) : ℝ := (f x₁ - f x₂) / (x₁ - x₂)

def b (x₁ x₂ : ℝ) : ℝ := (g k x₁ - g k x₂) / (x₁ - x₂)

theorem problem_statements :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → a x₁ x₂ > 0) ∧
  (∃ k : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b k x₁ x₂ ≤ 0) ∧
  (∀ k : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b k x₁ x₂ / a x₁ x₂ = 2) ∧
  (∀ k : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b k x₁ x₂ / a x₁ x₂ = -2) → k < -4) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_statements_l2186_218622


namespace NUMINAMATH_CALUDE_largest_angle_cosine_l2186_218666

theorem largest_angle_cosine (A B C : ℝ) (h1 : A = π/6) 
  (h2 : 2 * (B * C * Real.cos A) = 3 * (B^2 + C^2 - 2*B*C*Real.cos A)) :
  Real.cos (max A (max B C)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_cosine_l2186_218666


namespace NUMINAMATH_CALUDE_max_type_a_stationery_l2186_218667

/-- Represents the number of items for each stationery type -/
structure Stationery where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total cost of stationery -/
def totalCost (s : Stationery) : ℕ :=
  3 * s.a + 2 * s.b + s.c

/-- Checks if the stationery purchase satisfies all conditions -/
def isValidPurchase (s : Stationery) : Prop :=
  s.b = s.a - 2 ∧
  3 * s.a ≤ 33 ∧
  totalCost s = 66

/-- Theorem: The maximum number of Type A stationery that can be purchased is 11 -/
theorem max_type_a_stationery :
  ∃ (s : Stationery), isValidPurchase s ∧
  (∀ (t : Stationery), isValidPurchase t → t.a ≤ s.a) ∧
  s.a = 11 := by
  sorry


end NUMINAMATH_CALUDE_max_type_a_stationery_l2186_218667


namespace NUMINAMATH_CALUDE_f_major_premise_incorrect_l2186_218610

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2

-- State that f'(0) = 0
theorem f'_zero : f' 0 = 0 := by sorry

-- Define what it means for a point to be an extremum
def is_extremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≥ f x ∨ f x₀ ≤ f x

-- Theorem stating that the major premise is incorrect
theorem major_premise_incorrect :
  ¬(∀ x₀ : ℝ, f' x₀ = 0 → is_extremum f x₀) := by sorry

end NUMINAMATH_CALUDE_f_major_premise_incorrect_l2186_218610


namespace NUMINAMATH_CALUDE_sum_c_d_eq_nine_l2186_218633

/-- A quadrilateral PQRS with specific vertex coordinates -/
structure Quadrilateral (c d : ℤ) :=
  (c_pos : c > 0)
  (d_pos : d > 0)
  (c_gt_d : c > d)

/-- The area of the quadrilateral PQRS -/
def area (q : Quadrilateral c d) : ℝ := 2 * ((c : ℝ)^2 - (d : ℝ)^2)

theorem sum_c_d_eq_nine {c d : ℤ} (q : Quadrilateral c d) (h : area q = 18) :
  c + d = 9 := by
  sorry

#check sum_c_d_eq_nine

end NUMINAMATH_CALUDE_sum_c_d_eq_nine_l2186_218633


namespace NUMINAMATH_CALUDE_correct_rounded_result_l2186_218683

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

theorem correct_rounded_result : round_to_nearest_hundred (68 + 57) = 100 := by
  sorry

end NUMINAMATH_CALUDE_correct_rounded_result_l2186_218683


namespace NUMINAMATH_CALUDE_uncle_pill_duration_l2186_218658

/-- Represents the duration in days that a bottle of pills lasts -/
def bottle_duration (pills_per_bottle : ℕ) (dose : ℚ) (days_between_doses : ℕ) : ℚ :=
  (pills_per_bottle : ℚ) * (days_between_doses : ℚ) / dose

/-- Converts days to months, assuming 30 days per month -/
def days_to_months (days : ℚ) : ℚ :=
  days / 30

theorem uncle_pill_duration :
  let pills_per_bottle : ℕ := 60
  let dose : ℚ := 3/4
  let days_between_doses : ℕ := 3
  days_to_months (bottle_duration pills_per_bottle dose days_between_doses) = 8 := by
  sorry

#eval days_to_months (bottle_duration 60 (3/4) 3)

end NUMINAMATH_CALUDE_uncle_pill_duration_l2186_218658


namespace NUMINAMATH_CALUDE_initial_sum_is_500_l2186_218618

/-- Prove that the initial sum of money is $500 given the conditions of the problem. -/
theorem initial_sum_is_500 
  (sum_after_2_years : ℝ → ℝ → ℝ → ℝ) -- Function for final amount after 2 years
  (initial_sum : ℝ)  -- Initial sum of money
  (interest_rate : ℝ) -- Original interest rate
  (h1 : sum_after_2_years initial_sum interest_rate 2 = 600) -- First condition
  (h2 : sum_after_2_years initial_sum (interest_rate + 0.1) 2 = 700) -- Second condition
  : initial_sum = 500 := by
  sorry

end NUMINAMATH_CALUDE_initial_sum_is_500_l2186_218618


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l2186_218634

theorem simplify_nested_expression (x : ℝ) : 1 - (1 - (1 + (1 - (1 + (1 - x))))) = 3 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l2186_218634


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2186_218669

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2186_218669


namespace NUMINAMATH_CALUDE_problems_per_page_l2186_218653

/-- Given the total number of homework problems, the number of finished problems,
    and the number of remaining pages, calculate the number of problems per page. -/
theorem problems_per_page
  (total_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_pages : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : remaining_pages = 2)
  (h4 : remaining_pages > 0)
  (h5 : finished_problems ≤ total_problems) :
  (total_problems - finished_problems) / remaining_pages = 7 := by
sorry

end NUMINAMATH_CALUDE_problems_per_page_l2186_218653


namespace NUMINAMATH_CALUDE_prob_heart_then_king_is_one_52_l2186_218664

def standard_deck : ℕ := 52
def hearts_in_deck : ℕ := 13
def kings_in_deck : ℕ := 4

def prob_heart_then_king : ℚ :=
  (hearts_in_deck / standard_deck) * (kings_in_deck / (standard_deck - 1))

theorem prob_heart_then_king_is_one_52 :
  prob_heart_then_king = 1 / 52 := by sorry

end NUMINAMATH_CALUDE_prob_heart_then_king_is_one_52_l2186_218664


namespace NUMINAMATH_CALUDE_park_visitors_l2186_218601

/-- Represents the charging conditions for the park visit -/
structure ParkVisitConditions where
  base_fee : ℕ  -- Base fee per person
  base_limit : ℕ  -- Number of people for base fee
  discount_per_person : ℕ  -- Discount per additional person
  min_fee : ℕ  -- Minimum fee per person
  total_paid : ℕ  -- Total amount paid

/-- Calculates the fee per person based on the number of visitors -/
def fee_per_person (conditions : ParkVisitConditions) (num_visitors : ℕ) : ℕ :=
  max conditions.min_fee (conditions.base_fee - conditions.discount_per_person * (num_visitors - conditions.base_limit))

/-- Theorem: Given the charging conditions, 30 people visited the park -/
theorem park_visitors (conditions : ParkVisitConditions) 
  (h1 : conditions.base_fee = 100)
  (h2 : conditions.base_limit = 25)
  (h3 : conditions.discount_per_person = 2)
  (h4 : conditions.min_fee = 70)
  (h5 : conditions.total_paid = 2700) :
  ∃ (num_visitors : ℕ), 
    num_visitors = 30 ∧ 
    num_visitors * (fee_per_person conditions num_visitors) = conditions.total_paid :=
sorry

end NUMINAMATH_CALUDE_park_visitors_l2186_218601


namespace NUMINAMATH_CALUDE_mixtape_length_example_l2186_218662

/-- The length of a mixtape given the number of songs on each side and the length of each song. -/
def mixtape_length (side1_songs : ℕ) (side2_songs : ℕ) (song_length : ℕ) : ℕ :=
  (side1_songs + side2_songs) * song_length

/-- Theorem stating that a mixtape with 6 songs on the first side, 4 songs on the second side,
    and each song being 4 minutes long has a total length of 40 minutes. -/
theorem mixtape_length_example : mixtape_length 6 4 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_mixtape_length_example_l2186_218662


namespace NUMINAMATH_CALUDE_p_hyperbola_range_p_necessary_not_sufficient_for_q_l2186_218628

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (m - 4) = 1
def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (4 - m) = 1

-- Define what it means for p to represent a hyperbola
def p_is_hyperbola (m : ℝ) : Prop := (m - 1) * (m - 4) < 0

-- Define what it means for q to represent an ellipse
def q_is_ellipse (m : ℝ) : Prop := m - 2 > 0 ∧ 4 - m > 0 ∧ m - 2 ≠ 4 - m

-- Theorem 1: The range of m for which p represents a hyperbola
theorem p_hyperbola_range : 
  ∀ m : ℝ, p_is_hyperbola m ↔ (1 < m ∧ m < 4) :=
sorry

-- Theorem 2: p being true is necessary but not sufficient for q being true
theorem p_necessary_not_sufficient_for_q :
  (∀ m : ℝ, q_is_ellipse m → p_is_hyperbola m) ∧
  (∃ m : ℝ, p_is_hyperbola m ∧ ¬q_is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_p_hyperbola_range_p_necessary_not_sufficient_for_q_l2186_218628


namespace NUMINAMATH_CALUDE_range_of_sqrt_function_l2186_218673

theorem range_of_sqrt_function (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (2 - x)) ↔ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sqrt_function_l2186_218673


namespace NUMINAMATH_CALUDE_lemonade_glasses_count_l2186_218649

/-- The number of glasses of lemonade that can be served from one pitcher -/
def glasses_per_pitcher : ℕ := 5

/-- The number of pitchers of lemonade prepared -/
def number_of_pitchers : ℕ := 6

/-- The total number of glasses of lemonade that can be served -/
def total_glasses : ℕ := glasses_per_pitcher * number_of_pitchers

theorem lemonade_glasses_count : total_glasses = 30 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_glasses_count_l2186_218649


namespace NUMINAMATH_CALUDE_green_to_red_ratio_is_three_to_one_l2186_218617

/-- Represents the contents of a bag of mints -/
structure MintBag where
  green : ℕ
  red : ℕ

/-- The ratio of green mints to red mints -/
def mintRatio (bag : MintBag) : ℚ :=
  bag.green / bag.red

theorem green_to_red_ratio_is_three_to_one 
  (bag : MintBag) 
  (h_total : bag.green + bag.red > 0)
  (h_green_percent : (bag.green : ℚ) / (bag.green + bag.red) = 3/4) :
  mintRatio bag = 3/1 := by
sorry

end NUMINAMATH_CALUDE_green_to_red_ratio_is_three_to_one_l2186_218617


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l2186_218603

theorem imaginary_power_sum : ∃ i : ℂ, i^2 = -1 ∧ i^50 + i^250 = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l2186_218603


namespace NUMINAMATH_CALUDE_unique_right_triangle_l2186_218656

/-- Check if three numbers form a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The given sets of line segments -/
def segment_sets : List (ℕ × ℕ × ℕ) :=
  [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]

/-- Theorem: Only one set in segment_sets satisfies the Pythagorean theorem -/
theorem unique_right_triangle : 
  ∃! (a b c : ℕ), (a, b, c) ∈ segment_sets ∧ is_pythagorean_triple a b c :=
by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_l2186_218656


namespace NUMINAMATH_CALUDE_trigonometric_expression_proof_l2186_218607

theorem trigonometric_expression_proof (sin30 cos30 sin60 cos60 : ℝ) 
  (h1 : sin30 = 1/2)
  (h2 : cos30 = Real.sqrt 3 / 2)
  (h3 : sin60 = Real.sqrt 3 / 2)
  (h4 : cos60 = 1/2) :
  (1 - 1/(sin30^2)) * (1 + 1/(cos60^2)) * (1 - 1/(cos30^2)) * (1 + 1/(sin60^2)) = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_proof_l2186_218607


namespace NUMINAMATH_CALUDE_banana_cost_theorem_l2186_218641

def cost_of_fruit (apple_cost banana_cost orange_cost : ℚ)
                  (apple_count banana_count orange_count : ℕ)
                  (average_cost : ℚ) : Prop :=
  let total_count := apple_count + banana_count + orange_count
  let total_cost := apple_cost * apple_count + banana_cost * banana_count + orange_cost * orange_count
  total_cost = average_cost * total_count

theorem banana_cost_theorem :
  ∀ (banana_cost : ℚ),
    cost_of_fruit 2 banana_cost 3 12 4 4 2 →
    banana_cost = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_cost_theorem_l2186_218641


namespace NUMINAMATH_CALUDE_ben_win_probability_l2186_218635

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 3 / 7) (h2 : ¬ ∃ tie_prob : ℚ, tie_prob ≠ 0) : 
  1 - lose_prob = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_ben_win_probability_l2186_218635


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l2186_218695

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_c_value :
  ∀ c : ℝ, (∀ x y : ℝ, y = 5 * x + 7 ↔ y = (3 * c) * x + 1) → c = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l2186_218695


namespace NUMINAMATH_CALUDE_binomial_18_4_l2186_218613

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_4_l2186_218613


namespace NUMINAMATH_CALUDE_probability_ten_heads_in_twelve_flips_l2186_218637

theorem probability_ten_heads_in_twelve_flips :
  let n : ℕ := 12  -- Total number of coin flips
  let k : ℕ := 10  -- Number of desired heads
  let p : ℚ := 1/2 -- Probability of getting heads on a single flip (fair coin)
  Nat.choose n k * p^k * (1-p)^(n-k) = 66/4096 :=
by sorry

end NUMINAMATH_CALUDE_probability_ten_heads_in_twelve_flips_l2186_218637


namespace NUMINAMATH_CALUDE_team_selection_count_l2186_218608

/-- The number of ways to select a team of 8 members with an equal number of boys and girls
    from a group of 8 boys and 10 girls -/
def select_team (boys girls team_size : ℕ) : ℕ :=
  Nat.choose boys (team_size / 2) * Nat.choose girls (team_size / 2)

/-- Theorem stating the number of ways to select the team -/
theorem team_selection_count :
  select_team 8 10 8 = 14700 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l2186_218608


namespace NUMINAMATH_CALUDE_special_triangle_area_squared_l2186_218606

/-- An equilateral triangle with vertices on the hyperbola xy = 4 and centroid at a vertex of the hyperbola -/
structure SpecialTriangle where
  -- The hyperbola equation
  hyperbola : ℝ → ℝ → Prop
  hyperbola_def : hyperbola = fun x y ↦ x * y = 4

  -- The triangle is equilateral
  is_equilateral : Prop

  -- Vertices lie on the hyperbola
  vertices_on_hyperbola : Prop

  -- Centroid is at a vertex of the hyperbola
  centroid_on_hyperbola : Prop

/-- The square of the area of the special triangle is 3888 -/
theorem special_triangle_area_squared (t : SpecialTriangle) : 
  ∃ (area : ℝ), area^2 = 3888 := by sorry

end NUMINAMATH_CALUDE_special_triangle_area_squared_l2186_218606


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_l2186_218636

theorem cubic_sum_of_roots (m n r s : ℝ) : 
  (r^2 - m*r - n = 0) → (s^2 - m*s - n = 0) → r^3 + s^3 = m^3 + 3*n*m :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_l2186_218636


namespace NUMINAMATH_CALUDE_calculation_proof_l2186_218657

theorem calculation_proof :
  ((-1/12 - 1/36 + 1/6) * (-36) = -2) ∧
  ((-99 - 11/12) * 24 = -2398) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l2186_218657


namespace NUMINAMATH_CALUDE_museum_ticket_price_museum_ticket_price_is_6_l2186_218625

theorem museum_ticket_price (friday_price : ℝ) (saturday_visitors : ℕ) 
  (saturday_visitor_ratio : ℝ) (saturday_revenue_ratio : ℝ) : ℝ :=
let friday_visitors : ℕ := saturday_visitors / 2
let friday_revenue : ℝ := friday_visitors * friday_price
let saturday_revenue : ℝ := friday_revenue * saturday_revenue_ratio
let k : ℝ := saturday_revenue / saturday_visitors
k

theorem museum_ticket_price_is_6 :
  museum_ticket_price 9 200 2 (4/3) = 6 := by
sorry

end NUMINAMATH_CALUDE_museum_ticket_price_museum_ticket_price_is_6_l2186_218625


namespace NUMINAMATH_CALUDE_symmetry_x_axis_coordinates_l2186_218655

/-- Two points are symmetric with respect to the x-axis if they have the same x-coordinate
    and opposite y-coordinates -/
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- Given that P(-2, 3) is symmetric to Q(a, b) with respect to the x-axis,
    prove that a = -2 and b = -3 -/
theorem symmetry_x_axis_coordinates :
  let P : ℝ × ℝ := (-2, 3)
  let Q : ℝ × ℝ := (a, b)
  symmetric_x_axis P Q → a = -2 ∧ b = -3 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_x_axis_coordinates_l2186_218655


namespace NUMINAMATH_CALUDE_school_play_tickets_l2186_218688

/-- Calculates the total number of tickets sold for a school play. -/
def total_tickets (adult_tickets : ℕ) : ℕ :=
  adult_tickets + 2 * adult_tickets

/-- Theorem: Given 122 adult tickets and student tickets being twice the number of adult tickets,
    the total number of tickets sold is 366. -/
theorem school_play_tickets : total_tickets 122 = 366 := by
  sorry

end NUMINAMATH_CALUDE_school_play_tickets_l2186_218688


namespace NUMINAMATH_CALUDE_same_prime_factors_power_of_two_l2186_218661

theorem same_prime_factors_power_of_two (b m n : ℕ) 
  (hb : b ≠ 1) (hmn : m ≠ n) 
  (h_same_factors : ∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) :
  ∃ k : ℕ, b + 1 = 2^k := by sorry

end NUMINAMATH_CALUDE_same_prime_factors_power_of_two_l2186_218661


namespace NUMINAMATH_CALUDE_max_houses_buildable_l2186_218605

def houses_buildable (sinks doors windows toilets : ℕ) : ℕ :=
  min (sinks / 6) (min (doors / 4) (min (windows / 8) (toilets / 3)))

theorem max_houses_buildable :
  houses_buildable 266 424 608 219 = 73 := by
  sorry

end NUMINAMATH_CALUDE_max_houses_buildable_l2186_218605


namespace NUMINAMATH_CALUDE_equation_solvable_for_small_primes_l2186_218675

theorem equation_solvable_for_small_primes :
  ∀ p : ℕ, p.Prime → p ≤ 100 →
  ∃ x y : ℕ, (y^37 : ℤ) ≡ (x^3 + 11 : ℤ) [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_equation_solvable_for_small_primes_l2186_218675


namespace NUMINAMATH_CALUDE_project_completion_time_l2186_218640

/-- The time taken for teams A and D to complete a project given the completion times of other team combinations -/
theorem project_completion_time (t_AB t_BC t_CD : ℝ) (h_AB : t_AB = 20) (h_BC : t_BC = 60) (h_CD : t_CD = 30) :
  1 / (1 / t_AB + 1 / t_CD - 1 / t_BC) = 15 := by
  sorry

#check project_completion_time

end NUMINAMATH_CALUDE_project_completion_time_l2186_218640


namespace NUMINAMATH_CALUDE_intersection_points_l2186_218674

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

def g (x : ℝ) : ℝ := -f x

def h (x : ℝ) : ℝ := f (-x)

theorem intersection_points :
  (∃ a b : ℝ, a ≠ b ∧ f a = g a ∧ f b = g b) ∧
  (∃! c : ℝ, f c = h c) ∧
  (∀ x y : ℝ, x ≠ y → (f x = g x ∧ f y = g y) → (f x = h x ∧ f y = h y) → False) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l2186_218674


namespace NUMINAMATH_CALUDE_impossible_chord_length_l2186_218663

theorem impossible_chord_length (r : ℝ) (chord_length : ℝ) : 
  r = 5 → chord_length = 11 → chord_length > 2 * r := by sorry

end NUMINAMATH_CALUDE_impossible_chord_length_l2186_218663


namespace NUMINAMATH_CALUDE_student_fail_marks_l2186_218652

theorem student_fail_marks (pass_percentage : ℝ) (max_score : ℕ) (obtained_score : ℕ) : 
  pass_percentage = 36 / 100 → 
  max_score = 400 → 
  obtained_score = 130 → 
  ⌈pass_percentage * max_score⌉ - obtained_score = 14 := by
  sorry

end NUMINAMATH_CALUDE_student_fail_marks_l2186_218652


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2186_218647

theorem polynomial_division_theorem (z : ℝ) :
  4 * z^4 - 6 * z^3 + 7 * z^2 - 17 * z + 3 =
  (5 * z + 4) * (z^3 - (26/5) * z^2 + (1/5) * z - 67/25) + 331/25 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2186_218647


namespace NUMINAMATH_CALUDE_inequality_condition_l2186_218654

theorem inequality_condition (x : ℝ) : x * (x + 2) > x * (3 - x) + 1 ↔ x < -1/2 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l2186_218654


namespace NUMINAMATH_CALUDE_f_f_has_four_roots_l2186_218638

def f (x : ℝ) := x^2 - 3*x + 2

theorem f_f_has_four_roots :
  ∃! (s : Finset ℝ), s.card = 4 ∧ (∀ x ∈ s, f (f x) = 0) ∧ (∀ y, f (f y) = 0 → y ∈ s) :=
sorry

end NUMINAMATH_CALUDE_f_f_has_four_roots_l2186_218638


namespace NUMINAMATH_CALUDE_video_game_expenditure_l2186_218611

theorem video_game_expenditure (total : ℝ) (books snacks movies video_games : ℝ) : 
  total = 50 ∧ 
  books = (1/4) * total ∧ 
  snacks = (1/5) * total ∧ 
  movies = (2/5) * total ∧ 
  total = books + snacks + movies + video_games 
  → video_games = 7.5 := by
sorry

end NUMINAMATH_CALUDE_video_game_expenditure_l2186_218611


namespace NUMINAMATH_CALUDE_problem_solution_l2186_218678

theorem problem_solution (a : ℚ) : a + a / 4 = 6 / 2 → a = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2186_218678


namespace NUMINAMATH_CALUDE_ellipse_foci_y_axis_m_range_l2186_218648

/-- Represents an ellipse with the given equation -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (|m| - 1) + y^2 / (2 - m) = 1

/-- Indicates that the ellipse has foci on the y-axis -/
def has_foci_on_y_axis (e : Ellipse m) : Prop :=
  2 - m > |m| - 1 ∧ |m| - 1 > 0

/-- The range of m for which the ellipse has foci on the y-axis -/
def m_range (m : ℝ) : Prop :=
  m < -1 ∨ (1 < m ∧ m < 3/2)

/-- Theorem stating the range of m for an ellipse with foci on the y-axis -/
theorem ellipse_foci_y_axis_m_range (m : ℝ) :
  (∃ e : Ellipse m, has_foci_on_y_axis e) ↔ m_range m :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_y_axis_m_range_l2186_218648


namespace NUMINAMATH_CALUDE_cost_of_pizza_slice_l2186_218646

/-- The cost of a slice of pizza given the conditions of Zoe's purchase -/
theorem cost_of_pizza_slice (num_people : ℕ) (soda_cost : ℚ) (total_spent : ℚ) :
  num_people = 6 →
  soda_cost = 1/2 →
  total_spent = 9 →
  (total_spent - num_people * soda_cost) / num_people = 1 := by
  sorry

#check cost_of_pizza_slice

end NUMINAMATH_CALUDE_cost_of_pizza_slice_l2186_218646


namespace NUMINAMATH_CALUDE_probability_not_rain_l2186_218670

theorem probability_not_rain (p : ℚ) (h : p = 3 / 10) : 1 - p = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_rain_l2186_218670


namespace NUMINAMATH_CALUDE_poll_total_count_l2186_218680

theorem poll_total_count : ∀ (total : ℕ),
  (45 : ℚ) / 100 * total + (8 : ℚ) / 100 * total + (94 : ℕ) = total →
  total = 200 := by
  sorry

end NUMINAMATH_CALUDE_poll_total_count_l2186_218680


namespace NUMINAMATH_CALUDE_triangle_angle_b_is_pi_third_l2186_218631

theorem triangle_angle_b_is_pi_third (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a / Real.sin A = b / Real.sin B ∧  -- Law of sines
  b / Real.sin B = c / Real.sin C ∧  -- Law of sines
  b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2  -- Given condition
  → B = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_b_is_pi_third_l2186_218631


namespace NUMINAMATH_CALUDE_modified_coin_expected_winnings_l2186_218698

/-- A coin with three possible outcomes -/
structure Coin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_edge : ℚ
  winnings_heads : ℚ
  winnings_tails : ℚ
  loss_edge : ℚ

/-- The modified weighted coin as described in the problem -/
def modified_coin : Coin :=
  { prob_heads := 1/3
  , prob_tails := 1/2
  , prob_edge := 1/6
  , winnings_heads := 2
  , winnings_tails := 2
  , loss_edge := 4 }

/-- Expected winnings from flipping the coin -/
def expected_winnings (c : Coin) : ℚ :=
  c.prob_heads * c.winnings_heads + c.prob_tails * c.winnings_tails - c.prob_edge * c.loss_edge

/-- Theorem stating that the expected winnings from flipping the modified coin is 1 -/
theorem modified_coin_expected_winnings :
  expected_winnings modified_coin = 1 := by
  sorry

end NUMINAMATH_CALUDE_modified_coin_expected_winnings_l2186_218698


namespace NUMINAMATH_CALUDE_savings_after_twelve_months_l2186_218689

def savings_sequence (n : ℕ) : ℕ := 2 ^ n

theorem savings_after_twelve_months :
  savings_sequence 12 = 4096 := by sorry

end NUMINAMATH_CALUDE_savings_after_twelve_months_l2186_218689


namespace NUMINAMATH_CALUDE_ten_steps_climb_ways_l2186_218672

/-- The number of ways to climb n steps, where each move is either climbing 1 step or 2 steps -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => climbStairs k + climbStairs (k + 1)

/-- Theorem stating that there are 89 ways to climb 10 steps -/
theorem ten_steps_climb_ways : climbStairs 10 = 89 := by
  sorry

end NUMINAMATH_CALUDE_ten_steps_climb_ways_l2186_218672


namespace NUMINAMATH_CALUDE_females_band_not_orchestra_l2186_218619

/-- Represents the number of students in different groups -/
structure StudentGroups where
  bandFemales : ℕ
  bandMales : ℕ
  orchestraFemales : ℕ
  orchestraMales : ℕ
  bothFemales : ℕ
  totalStudents : ℕ

/-- Theorem stating the number of females in the band but not in the orchestra -/
theorem females_band_not_orchestra (g : StudentGroups)
  (h1 : g.bandFemales = 120)
  (h2 : g.bandMales = 70)
  (h3 : g.orchestraFemales = 70)
  (h4 : g.orchestraMales = 110)
  (h5 : g.bothFemales = 45)
  (h6 : g.totalStudents = 250) :
  g.bandFemales - g.bothFemales = 75 := by
  sorry

#check females_band_not_orchestra

end NUMINAMATH_CALUDE_females_band_not_orchestra_l2186_218619


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l2186_218681

-- Define the displacement function
def displacement (t : ℝ) : ℝ := t^2 - t

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds : 
  velocity 3 = 5 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l2186_218681


namespace NUMINAMATH_CALUDE_game_ends_in_37_rounds_l2186_218609

/-- Represents the state of the game at any point --/
structure GameState where
  a : ℕ  -- tokens of player A
  b : ℕ  -- tokens of player B
  c : ℕ  -- tokens of player C

/-- Represents a single round of the game --/
def playRound (state : GameState) : GameState :=
  if state.a ≥ state.b ∧ state.a ≥ state.c then
    { a := state.a - 3, b := state.b + 1, c := state.c + 1 }
  else if state.b ≥ state.a ∧ state.b ≥ state.c then
    { a := state.a + 1, b := state.b - 3, c := state.c + 1 }
  else
    { a := state.a + 1, b := state.b + 1, c := state.c - 3 }

/-- Checks if the game has ended (any player has 0 tokens) --/
def gameEnded (state : GameState) : Bool :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- Plays the game for a given number of rounds --/
def playGame (initialState : GameState) (rounds : ℕ) : GameState :=
  match rounds with
  | 0 => initialState
  | n + 1 => playRound (playGame initialState n)

/-- The main theorem to prove --/
theorem game_ends_in_37_rounds :
  let initialState : GameState := { a := 15, b := 14, c := 13 }
  let finalState := playGame initialState 37
  gameEnded finalState ∧ ¬gameEnded (playGame initialState 36) := by
  sorry

#check game_ends_in_37_rounds

end NUMINAMATH_CALUDE_game_ends_in_37_rounds_l2186_218609


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2186_218659

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t^2 - 4, t^3 - 6*t + 3)

-- Define the self-intersection point
def intersection_point : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = intersection_point :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2186_218659


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_64_l2186_218650

theorem product_of_fractions_equals_64 :
  (8 / 4) * (10 / 25) * (20 / 10) * (15 / 45) * (40 / 20) * (24 / 8) * (30 / 15) * (35 / 7) = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_64_l2186_218650


namespace NUMINAMATH_CALUDE_prove_age_difference_l2186_218621

-- Define the given information
def wayne_age_2021 : ℕ := 37
def peter_age_diff : ℕ := 3
def julia_birth_year : ℕ := 1979

-- Define the current year
def current_year : ℕ := 2021

-- Define the age difference between Julia and Peter
def julia_peter_age_diff : ℕ := 2

-- Theorem to prove
theorem prove_age_difference :
  (current_year - wayne_age_2021 - peter_age_diff) - julia_birth_year = julia_peter_age_diff :=
by sorry

end NUMINAMATH_CALUDE_prove_age_difference_l2186_218621


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_l2186_218604

theorem triangle_isosceles_or_right 
  (a b c : ℝ) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_lengths_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h : a^2 * c^2 - b^2 * c^2 = a^4 - b^4) : 
  (a = b) ∨ (a^2 + b^2 = c^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_l2186_218604


namespace NUMINAMATH_CALUDE_unique_n_divisibility_l2186_218612

theorem unique_n_divisibility : ∃! n : ℕ, 
  0 < n ∧ n < 1000 ∧ 
  (∃ k₁ k₂ k₃ : ℕ, 
    345564 - n = 13 * k₁ ∧ 
    345564 - n = 17 * k₂ ∧ 
    345564 - n = 19 * k₃) :=
by sorry

end NUMINAMATH_CALUDE_unique_n_divisibility_l2186_218612


namespace NUMINAMATH_CALUDE_count_divisors_252_not_div_by_seven_l2186_218630

def divisors_not_div_by_seven (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ x => x > 0 ∧ n % x = 0 ∧ x % 7 ≠ 0)

theorem count_divisors_252_not_div_by_seven :
  (divisors_not_div_by_seven 252).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_divisors_252_not_div_by_seven_l2186_218630


namespace NUMINAMATH_CALUDE_clea_escalator_time_l2186_218643

/-- Represents the scenario of Clea walking on an escalator -/
structure EscalatorScenario where
  /-- Clea's walking speed (units per second) -/
  walking_speed : ℝ
  /-- Total distance of the escalator (units) -/
  escalator_distance : ℝ
  /-- Speed of the moving escalator (units per second) -/
  escalator_speed : ℝ

/-- Time taken for Clea to walk down the stationary escalator -/
def time_stationary (scenario : EscalatorScenario) : ℝ :=
  80

/-- Time taken for Clea to walk down the moving escalator -/
def time_moving (scenario : EscalatorScenario) : ℝ :=
  32

/-- Theorem stating the time taken for the given scenario -/
theorem clea_escalator_time (scenario : EscalatorScenario) :
  scenario.escalator_speed = 1.5 * scenario.walking_speed →
  (scenario.escalator_distance / scenario.walking_speed / 2) +
  (scenario.escalator_distance / (2 * scenario.escalator_speed)) = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_clea_escalator_time_l2186_218643


namespace NUMINAMATH_CALUDE_largest_n_with_2020_sets_l2186_218645

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define a_n as the number of sets S of positive integers
-- such that the sum of F_k for k in S equals n
def a (n : ℕ) : ℕ := sorry

-- State the theorem
theorem largest_n_with_2020_sets :
  ∃ n : ℕ, a n = 2020 ∧ ∀ m : ℕ, m > n → a m ≠ 2020 ∧ n = fib 2022 - 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_with_2020_sets_l2186_218645


namespace NUMINAMATH_CALUDE_fold_square_crease_length_l2186_218642

-- Define the square ABCD
def square_side : ℝ := 8

-- Define point E on AD
def AE : ℝ := 2
def ED : ℝ := 6

-- Define FD as x
def FD : ℝ → ℝ := λ x => x

-- Define CF and EF
def CF (x : ℝ) : ℝ := square_side - x
def EF (x : ℝ) : ℝ := square_side - x

-- State the theorem
theorem fold_square_crease_length :
  ∃ x : ℝ, FD x = 7/4 ∧ CF x = EF x ∧ CF x^2 = FD x^2 + ED^2 := by
  sorry

end NUMINAMATH_CALUDE_fold_square_crease_length_l2186_218642


namespace NUMINAMATH_CALUDE_brendans_morning_catch_l2186_218600

theorem brendans_morning_catch (total : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ)
  (h1 : total = 23)
  (h2 : thrown_back = 3)
  (h3 : afternoon_catch = 5)
  (h4 : dad_catch = 13) :
  total = (morning_catch - thrown_back + afternoon_catch + dad_catch) →
  morning_catch = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_brendans_morning_catch_l2186_218600


namespace NUMINAMATH_CALUDE_digit_sum_l2186_218624

theorem digit_sum (P Q R S T : Nat) : 
  (P < 10 ∧ Q < 10 ∧ R < 10 ∧ S < 10 ∧ T < 10) → 
  (4 * (P * 10000 + Q * 1000 + R * 100 + S * 10 + T) = 41024) → 
  (P + Q + R + S + T = 14) := by
sorry

end NUMINAMATH_CALUDE_digit_sum_l2186_218624


namespace NUMINAMATH_CALUDE_zuminglish_seven_letter_words_l2186_218632

/-- Represents the ending of a word -/
inductive WordEnding
| CC  -- Two consonants
| CV  -- Consonant followed by vowel
| VC  -- Vowel followed by consonant

/-- Represents the rules of Zuminglish -/
structure Zuminglish where
  -- The number of n-letter words ending in each type
  count : ℕ → WordEnding → ℕ
  -- Initial conditions for 2-letter words
  init_CC : count 2 WordEnding.CC = 4
  init_CV : count 2 WordEnding.CV = 2
  init_VC : count 2 WordEnding.VC = 2
  -- Recursive relations
  rec_CC : ∀ n, count (n+1) WordEnding.CC = 2 * (count n WordEnding.CC + count n WordEnding.VC)
  rec_CV : ∀ n, count (n+1) WordEnding.CV = count n WordEnding.CC
  rec_VC : ∀ n, count (n+1) WordEnding.VC = 2 * count n WordEnding.CV

/-- The main theorem stating the number of valid 7-letter words in Zuminglish -/
theorem zuminglish_seven_letter_words (z : Zuminglish) :
  z.count 7 WordEnding.CC + z.count 7 WordEnding.CV + z.count 7 WordEnding.VC = 912 := by
  sorry


end NUMINAMATH_CALUDE_zuminglish_seven_letter_words_l2186_218632


namespace NUMINAMATH_CALUDE_count_common_divisors_9240_8820_l2186_218677

/-- The number of positive divisors that 9240 and 8820 have in common -/
def common_divisors_count : ℕ := 24

/-- Theorem stating that the number of positive divisors that 9240 and 8820 have in common is 24 -/
theorem count_common_divisors_9240_8820 : 
  (Nat.divisors 9240 ∩ Nat.divisors 8820).card = common_divisors_count := by
  sorry

end NUMINAMATH_CALUDE_count_common_divisors_9240_8820_l2186_218677


namespace NUMINAMATH_CALUDE_M_on_y_axis_coordinates_l2186_218692

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point on the y-axis -/
def on_y_axis (p : Point) : Prop := p.x = 0

/-- The point M with coordinates (m+1, m+3) -/
def M (m : ℝ) : Point :=
  { x := m + 1
    y := m + 3 }

/-- Theorem: If M(m+1, m+3) is on the y-axis, then its coordinates are (0, 2) -/
theorem M_on_y_axis_coordinates :
  ∀ m : ℝ, on_y_axis (M m) → M m = { x := 0, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_M_on_y_axis_coordinates_l2186_218692


namespace NUMINAMATH_CALUDE_sock_ratio_is_7_19_l2186_218626

/-- Represents the ratio of black socks to blue socks -/
structure SockRatio where
  black : ℕ
  blue : ℕ

/-- Represents the order of socks -/
structure SockOrder where
  black : ℕ
  blue : ℕ
  price_ratio : ℚ
  bill_increase : ℚ

/-- Calculates the ratio of black socks to blue socks given a sock order -/
def calculate_sock_ratio (order : SockOrder) : SockRatio :=
  sorry

/-- The specific sock order from the problem -/
def tom_order : SockOrder :=
  { black := 5
  , blue := 0  -- Unknown, to be calculated
  , price_ratio := 3
  , bill_increase := 3/5 }

theorem sock_ratio_is_7_19 : 
  let ratio := calculate_sock_ratio tom_order
  ratio.black = 7 ∧ ratio.blue = 19 := by sorry

end NUMINAMATH_CALUDE_sock_ratio_is_7_19_l2186_218626


namespace NUMINAMATH_CALUDE_percentage_problem_l2186_218644

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 600 = (50 / 100) * 1080 → P = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2186_218644


namespace NUMINAMATH_CALUDE_problem_solution_l2186_218685

theorem problem_solution (x y : ℝ) (h : 0.5 * x = y + 20) : x - 2 * y = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2186_218685


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2186_218676

theorem quadratic_roots_theorem (k : ℝ) (a b : ℝ) : 
  (∀ x, k * (x^2 - x) + x + 2 = 0 ↔ x = a ∨ x = b) →
  (a / b + b / a = 3 / 7) →
  (∃ k₁ k₂ : ℝ, 
    k₁ = (20 + Real.sqrt 988) / 14 ∧
    k₂ = (20 - Real.sqrt 988) / 14 ∧
    (k = k₁ ∨ k = k₂) ∧
    k₁ / k₂ + k₂ / k₁ = -104 / 21) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2186_218676


namespace NUMINAMATH_CALUDE_first_discount_percentage_l2186_218620

/-- Proves that given specific conditions on the original price, final price, and second discount,
    the first discount must be 12%. -/
theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 400 →
  final_price = 334.4 →
  second_discount = 5 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    first_discount = 12 := by
  sorry

#check first_discount_percentage

end NUMINAMATH_CALUDE_first_discount_percentage_l2186_218620


namespace NUMINAMATH_CALUDE_specific_ellipse_area_l2186_218699

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Given the endpoints of the major axis and a point on the ellipse, 
    calculate the ellipse parameters -/
def calculateEllipse (p1 p2 p3 : Point) : Ellipse :=
  sorry

/-- Calculate the area of an ellipse -/
def ellipseArea (e : Ellipse) : ℝ :=
  sorry

/-- The main theorem stating the area of the specific ellipse -/
theorem specific_ellipse_area : 
  let p1 : Point := ⟨-10, 3⟩
  let p2 : Point := ⟨8, 3⟩
  let p3 : Point := ⟨6, 8⟩
  let e : Ellipse := calculateEllipse p1 p2 p3
  ellipseArea e = (405 * Real.pi) / (4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_area_l2186_218699


namespace NUMINAMATH_CALUDE_at_least_two_same_number_l2186_218682

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of at least two dice showing the same number -/
def prob_same_number : ℝ := 1

theorem at_least_two_same_number :
  num_dice > num_sides → prob_same_number = 1 := by sorry

end NUMINAMATH_CALUDE_at_least_two_same_number_l2186_218682


namespace NUMINAMATH_CALUDE_pirate_loot_value_l2186_218627

/-- Converts a number from base 5 to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The loot values in base 5 --/
def silverware : List Nat := [3, 2, 1, 4]
def silkGarments : List Nat := [1, 2, 0, 2]
def rareSpices : List Nat := [1, 3, 2]

theorem pirate_loot_value :
  base5ToBase10 silverware + base5ToBase10 silkGarments + base5ToBase10 rareSpices = 865 := by
  sorry


end NUMINAMATH_CALUDE_pirate_loot_value_l2186_218627


namespace NUMINAMATH_CALUDE_folded_blankets_theorem_l2186_218687

/-- The thickness of a stack of folded blankets -/
def folded_blankets_thickness (initial_thickness : ℕ) (num_blankets : ℕ) (num_folds : ℕ) : ℕ :=
  num_blankets * initial_thickness * (2 ^ num_folds)

/-- Theorem: The thickness of n blankets, each initially 3 inches thick and folded 4 times, is 48n inches -/
theorem folded_blankets_theorem (n : ℕ) :
  folded_blankets_thickness 3 n 4 = 48 * n := by
  sorry

end NUMINAMATH_CALUDE_folded_blankets_theorem_l2186_218687


namespace NUMINAMATH_CALUDE_emilys_earnings_l2186_218615

-- Define the work hours for each day
def monday_hours : Real := 1
def wednesday_start : Real := 14.17  -- 2:10 PM in 24-hour format
def wednesday_end : Real := 16.83    -- 4:50 PM in 24-hour format
def thursday_hours : Real := 0.5
def saturday_hours : Real := 0.5

-- Define the hourly rate
def hourly_rate : Real := 4

-- Define the total earnings
def total_earnings : Real :=
  (monday_hours + (wednesday_end - wednesday_start) + thursday_hours + saturday_hours) * hourly_rate

-- Theorem to prove
theorem emilys_earnings :
  total_earnings = 18.68 := by
  sorry

end NUMINAMATH_CALUDE_emilys_earnings_l2186_218615


namespace NUMINAMATH_CALUDE_waiter_customers_l2186_218690

/-- The initial number of customers before 5 more arrived -/
def initial_customers : ℕ := 3

/-- The number of additional customers that arrived -/
def additional_customers : ℕ := 5

/-- The total number of customers after the additional customers arrived -/
def total_customers : ℕ := 8

theorem waiter_customers : 
  initial_customers + additional_customers = total_customers := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l2186_218690


namespace NUMINAMATH_CALUDE_overlap_area_theorem_l2186_218679

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side : ℝ)
  (rotation : ℝ)

/-- Calculates the area of the polygon formed by overlapping rotated squares -/
def overlappingArea (sheets : List Sheet) : ℝ :=
  sorry

theorem overlap_area_theorem : 
  let sheets : List Sheet := [
    { side := 8, rotation := 0 },
    { side := 8, rotation := 15 },
    { side := 8, rotation := 45 },
    { side := 8, rotation := 75 }
  ]
  overlappingArea sheets = 512 := by sorry

end NUMINAMATH_CALUDE_overlap_area_theorem_l2186_218679


namespace NUMINAMATH_CALUDE_number_of_divisors_5400_l2186_218694

theorem number_of_divisors_5400 : Nat.card (Nat.divisors 5400) = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_5400_l2186_218694


namespace NUMINAMATH_CALUDE_materik_position_l2186_218629

def Alphabet : Finset Char := {'A', 'E', 'I', 'K', 'M', 'R', 'T'}

def Word := List Char

def isValidWord (w : Word) : Prop :=
  w.length = 7 ∧ w.toFinset = Alphabet

def alphabeticalOrder (order : List Char) : Prop :=
  order.length = 7 ∧ order.toFinset = Alphabet

def wordPosition (w : Word) (order : List Char) : ℕ :=
  sorry

theorem materik_position 
  (order : List Char) 
  (h_order : alphabeticalOrder order) 
  (h_metrika : wordPosition ['M', 'E', 'T', 'R', 'I', 'K', 'A'] order = 3634) :
  wordPosition ['M', 'A', 'T', 'E', 'R', 'I', 'K'] order = 3745 :=
sorry

end NUMINAMATH_CALUDE_materik_position_l2186_218629


namespace NUMINAMATH_CALUDE_units_digit_of_7_cubed_l2186_218639

theorem units_digit_of_7_cubed (n : ℕ) : n = 7^3 → n % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_cubed_l2186_218639


namespace NUMINAMATH_CALUDE_length_of_AB_prime_l2186_218602

/-- Given points A, B, and C, where A' and B' are on the line y=x, and AC and BC intersect at C,
    prove that the length of A'B' is 10√2/11 -/
theorem length_of_AB_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 7) →
  A'.1 = A'.2 →
  B'.1 = B'.2 →
  (C.2 - A.2) / (C.1 - A.1) = (A'.2 - A.2) / (A'.1 - A.1) →
  (C.2 - B.2) / (C.1 - B.1) = (B'.2 - B.2) / (B'.1 - B.1) →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 10 * Real.sqrt 2 / 11 := by
  sorry

#check length_of_AB_prime

end NUMINAMATH_CALUDE_length_of_AB_prime_l2186_218602


namespace NUMINAMATH_CALUDE_roberts_ride_time_l2186_218697

/-- The time taken for Robert to ride along a semi-circular path on a highway segment -/
theorem roberts_ride_time 
  (highway_length : ℝ) 
  (highway_width : ℝ) 
  (speed : ℝ) 
  (miles_to_feet : ℝ) 
  (h1 : highway_length = 1) 
  (h2 : highway_width = 40) 
  (h3 : speed = 5) 
  (h4 : miles_to_feet = 5280) : 
  ∃ (time : ℝ), time = π / 10 := by
sorry

end NUMINAMATH_CALUDE_roberts_ride_time_l2186_218697


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2186_218623

-- System (1)
theorem system_one_solution (x y : ℝ) : 
  x + y = 1 ∧ 3 * x + y = 5 → x = 2 ∧ y = -1 := by sorry

-- System (2)
theorem system_two_solution (x y : ℝ) : 
  3 * (x - 1) + 4 * y = 1 ∧ 2 * x + 3 * (y + 1) = 2 → x = 16 ∧ y = -11 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2186_218623


namespace NUMINAMATH_CALUDE_birth_outcome_probabilities_l2186_218696

def num_children : ℕ := 5
def prob_boy : ℚ := 1/2
def prob_girl : ℚ := 1/2

theorem birth_outcome_probabilities :
  let prob_all_boys : ℚ := prob_boy ^ num_children
  let prob_all_girls : ℚ := prob_girl ^ num_children
  let prob_three_girls_two_boys : ℚ := (Nat.choose num_children 3) * (prob_girl ^ 3) * (prob_boy ^ 2)
  let prob_four_one : ℚ := 2 * (Nat.choose num_children 1) * (prob_girl ^ 4) * prob_boy
  prob_three_girls_two_boys = prob_four_one ∧
  prob_three_girls_two_boys > prob_all_boys ∧
  prob_three_girls_two_boys > prob_all_girls :=
by
  sorry

#check birth_outcome_probabilities

end NUMINAMATH_CALUDE_birth_outcome_probabilities_l2186_218696


namespace NUMINAMATH_CALUDE_eliot_account_balance_l2186_218684

theorem eliot_account_balance :
  ∀ (A E : ℝ),
  A > E →
  A - E = (1 / 12) * (A + E) →
  1.1 * A = 1.2 * E + 21 →
  E = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_eliot_account_balance_l2186_218684
