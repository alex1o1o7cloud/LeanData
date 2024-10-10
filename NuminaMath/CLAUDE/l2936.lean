import Mathlib

namespace f_derivative_lower_bound_and_range_l2936_293617

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_derivative_lower_bound_and_range :
  (∀ x : ℝ, (deriv f) x ≥ 2) ∧
  (∀ x : ℝ, x ≥ 0 → f (x^2 - 1) < Real.exp 1 - Real.exp (-1) → 0 ≤ x ∧ x < Real.sqrt 2) :=
by sorry

end f_derivative_lower_bound_and_range_l2936_293617


namespace probability_of_one_in_twenty_rows_l2936_293600

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) := sorry

/-- Counts the number of ones in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def countElements (n : ℕ) : ℕ := sorry

/-- The probability of randomly selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ :=
  (countOnes n : ℚ) / (countElements n : ℚ)

theorem probability_of_one_in_twenty_rows :
  probabilityOfOne 20 = 13 / 70 := by sorry

end probability_of_one_in_twenty_rows_l2936_293600


namespace range_of_a_l2936_293662

/-- Proposition p: There exists x ∈ ℝ such that x^2 - 2x + a^2 = 0 -/
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

/-- Proposition q: For all x ∈ ℝ, ax^2 - ax + 1 > 0 -/
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

/-- The range of a given p ∧ (¬q) is true -/
theorem range_of_a (a : ℝ) (h : p a ∧ ¬(q a)) : -1 ≤ a ∧ a < 0 :=
sorry

end range_of_a_l2936_293662


namespace cyclists_max_daily_distance_l2936_293684

theorem cyclists_max_daily_distance (distance_to_boston distance_to_atlanta : ℕ) 
  (h1 : distance_to_boston = 840) 
  (h2 : distance_to_atlanta = 440) : 
  (Nat.gcd distance_to_boston distance_to_atlanta) = 40 := by
  sorry

end cyclists_max_daily_distance_l2936_293684


namespace quadrant_restriction_l2936_293646

theorem quadrant_restriction (θ : Real) :
  1 + Real.sin θ * Real.sqrt (Real.sin θ * Real.sin θ) + 
  Real.cos θ * Real.sqrt (Real.cos θ * Real.cos θ) = 0 →
  (Real.sin θ > 0 ∧ Real.cos θ > 0) ∨ 
  (Real.sin θ > 0 ∧ Real.cos θ < 0) ∨ 
  (Real.sin θ < 0 ∧ Real.cos θ > 0) → False := by
  sorry

end quadrant_restriction_l2936_293646


namespace school_garden_flowers_l2936_293667

theorem school_garden_flowers (total : ℕ) (yellow : ℕ) : 
  total = 96 → yellow = 12 → ∃ (green : ℕ), 
    green + 3 * green + (total / 2) + yellow = total ∧ green = 9 := by
  sorry

end school_garden_flowers_l2936_293667


namespace quadratic_single_solution_l2936_293683

theorem quadratic_single_solution (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 - 7 * x + m = 0) ↔ m = 49 / 12 := by
  sorry

end quadratic_single_solution_l2936_293683


namespace sum_of_squares_l2936_293616

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 88) : x^2 + y^2 = 400 := by
  sorry

end sum_of_squares_l2936_293616


namespace sum_of_factors_36_l2936_293637

/-- The sum of positive factors of 36 is 91. -/
theorem sum_of_factors_36 : (Finset.filter (· ∣ 36) (Finset.range 37)).sum id = 91 := by
  sorry

end sum_of_factors_36_l2936_293637


namespace coronavirus_size_scientific_notation_l2936_293697

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem coronavirus_size_scientific_notation :
  toScientificNotation 0.0000012 = ScientificNotation.mk 1.2 (-6) sorry := by
  sorry

end coronavirus_size_scientific_notation_l2936_293697


namespace integers_between_cubes_l2936_293629

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(10.1 : ℝ)^3⌋ - ⌈(9.8 : ℝ)^3⌉ + 1) ∧ n = 89 := by sorry

end integers_between_cubes_l2936_293629


namespace largest_root_of_equation_l2936_293644

theorem largest_root_of_equation (x : ℝ) :
  (x - 37)^2 - 169 = 0 → x ≤ 50 ∧ ∃ y, (y - 37)^2 - 169 = 0 ∧ y = 50 := by
  sorry

end largest_root_of_equation_l2936_293644


namespace project_selection_count_l2936_293623

def num_key_projects : ℕ := 4
def num_general_projects : ℕ := 6
def projects_to_select : ℕ := 3

def select_projects (n k : ℕ) : ℕ := Nat.choose n k

theorem project_selection_count : 
  (select_projects (num_general_projects - 1) (projects_to_select - 1) * 
   select_projects (num_key_projects - 1) (projects_to_select - 1)) +
  (select_projects (num_key_projects - 1) 1 * 
   select_projects (num_general_projects - 1) 1) = 45 := by sorry

end project_selection_count_l2936_293623


namespace arithmetic_sequence_properties_l2936_293651

def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def increasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (d : ℝ) (h : d > 0) (h_arith : arithmeticSequence a d) :
  increasingSequence a ∧
  increasingSequence (fun n ↦ a n + 3 * n * d) ∧
  (¬ ∀ d, arithmeticSequence a d → increasingSequence (fun n ↦ n * a n)) ∧
  (¬ ∀ d, arithmeticSequence a d → increasingSequence (fun n ↦ a n / n)) :=
by sorry

end arithmetic_sequence_properties_l2936_293651


namespace negation_of_forall_positive_negation_of_gt_zero_l2936_293633

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_gt_zero :
  (¬∀ x : ℝ, x^2 - 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) :=
by sorry

end negation_of_forall_positive_negation_of_gt_zero_l2936_293633


namespace nell_gave_28_cards_l2936_293677

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff (initial_cards : ℕ) (remaining_cards : ℕ) : ℕ :=
  initial_cards - remaining_cards

/-- Proof that Nell gave 28 cards to Jeff -/
theorem nell_gave_28_cards :
  cards_given_to_jeff 304 276 = 28 := by
  sorry

end nell_gave_28_cards_l2936_293677


namespace train_length_l2936_293626

theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
  (h1 : platform_crossing_time = 39)
  (h2 : pole_crossing_time = 18)
  (h3 : platform_length = 700) :
  ∃ (train_length : ℝ),
    train_length = 600 ∧
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by
  sorry

end train_length_l2936_293626


namespace highDiveVelocity_l2936_293675

/-- The height function for a high-dive swimmer -/
def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

/-- The instantaneous velocity of the high-dive swimmer at t=1s -/
theorem highDiveVelocity : 
  (deriv h) 1 = -3.3 := by sorry

end highDiveVelocity_l2936_293675


namespace remainder_theorem_l2936_293673

def polynomial (x : ℝ) : ℝ := 5 * x^3 - 18 * x^2 + 31 * x - 40
def divisor (x : ℝ) : ℝ := 5 * x - 10

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + (-10) := by
  sorry

end remainder_theorem_l2936_293673


namespace square_x_plus_2y_l2936_293605

theorem square_x_plus_2y (x y : ℝ) 
  (h1 : x * (x + y) = 40) 
  (h2 : y * (x + y) = 90) : 
  (x + 2*y)^2 = 310 + 8100/130 := by
  sorry

end square_x_plus_2y_l2936_293605


namespace sum_in_base_6_l2936_293614

/-- Converts a base 6 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- The sum of 453₆, 436₆, and 42₆ in base 6 is 1415₆ --/
theorem sum_in_base_6 :
  to_base_6 (to_base_10 [3, 5, 4] + to_base_10 [6, 3, 4] + to_base_10 [2, 4]) = [5, 1, 4, 1] :=
sorry

end sum_in_base_6_l2936_293614


namespace curve_C_and_point_Q_existence_l2936_293609

noncomputable section

-- Define the circle O
def circle_O : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the curve C
def curve_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

-- Define the fixed point (0, 1/2)
def fixed_point : ℝ × ℝ := (0, 1/2)

-- Define the point Q
def Q : ℝ × ℝ := (0, 6)

-- State the theorem
theorem curve_C_and_point_Q_existence :
  ∀ (P : ℝ × ℝ),
  (∃ (center : ℝ × ℝ), (center.1 - F.1)^2 + (center.2 - F.2)^2 = (center.1 - P.1)^2 + (center.2 - P.2)^2 ∧
                       ∃ (T : ℝ × ℝ), T ∈ circle_O ∧ (center.1 - T.1)^2 + (center.2 - T.2)^2 = (F.1 - P.1)^2 / 4 + (F.2 - P.2)^2 / 4) →
  P ∈ curve_C ∧
  ∀ (M N : ℝ × ℝ), M ∈ curve_C → N ∈ curve_C →
    (N.2 - M.2) * fixed_point.1 = (N.1 - M.1) * (fixed_point.2 - M.2) + M.1 * (N.2 - M.2) →
    (M.2 - Q.2) / (M.1 - Q.1) + (N.2 - Q.2) / (N.1 - Q.1) = 0 :=
by sorry

end

end curve_C_and_point_Q_existence_l2936_293609


namespace intersection_implies_a_value_l2936_293635

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {9, a-5, 1-a}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {9} → a = -3 :=
by sorry

end intersection_implies_a_value_l2936_293635


namespace meeting_at_163rd_streetlight_l2936_293689

/-- The number of streetlights along the alley -/
def num_streetlights : ℕ := 400

/-- The position where Alla and Boris meet -/
def meeting_point : ℕ := 163

/-- Alla's position when observation is made -/
def alla_observed_pos : ℕ := 55

/-- Boris's position when observation is made -/
def boris_observed_pos : ℕ := 321

/-- The theorem stating that Alla and Boris meet at the 163rd streetlight -/
theorem meeting_at_163rd_streetlight :
  let alla_distance := alla_observed_pos - 1
  let boris_distance := num_streetlights - boris_observed_pos
  let total_observed_distance := alla_distance + boris_distance
  let scaling_factor := (num_streetlights - 1) / total_observed_distance
  (1 : ℚ) + scaling_factor * alla_distance = meeting_point := by
  sorry

end meeting_at_163rd_streetlight_l2936_293689


namespace solution_set_part1_range_of_a_part2_l2936_293625

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x - 2|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f x a ≥ 2*a - 1} = {a : ℝ | a ≤ 1} :=
sorry

end solution_set_part1_range_of_a_part2_l2936_293625


namespace asterisk_replacement_l2936_293666

theorem asterisk_replacement : ∃ x : ℚ, (x / 18) * (36 / 72) = 1 ∧ x = 36 := by
  sorry

end asterisk_replacement_l2936_293666


namespace arithmetic_sequence_20th_term_l2936_293660

/-- Given an arithmetic sequence with first term 2 and common difference 3,
    the 20th term of the sequence is 59. -/
theorem arithmetic_sequence_20th_term : 
  ∀ (a : ℕ → ℤ), 
    (a 1 = 2) →  -- First term is 2
    (∀ n, a (n + 1) - a n = 3) →  -- Common difference is 3
    a 20 = 59 := by
  sorry

end arithmetic_sequence_20th_term_l2936_293660


namespace square_sum_given_product_and_sum_l2936_293631

theorem square_sum_given_product_and_sum (a b : ℝ) 
  (h1 : a * b = 16) 
  (h2 : a + b = 10) : 
  a^2 + b^2 = 68 := by
sorry

end square_sum_given_product_and_sum_l2936_293631


namespace train_length_l2936_293691

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 9 → ∃ (length : ℝ), abs (length - 150.03) < 0.01 := by
  sorry


end train_length_l2936_293691


namespace square_area_difference_l2936_293652

-- Define the sides of the squares
def a : ℕ := 12
def b : ℕ := 9
def c : ℕ := 7
def d : ℕ := 3

-- Define the theorem
theorem square_area_difference : a ^ 2 + c ^ 2 - b ^ 2 - d ^ 2 = 103 := by
  sorry

end square_area_difference_l2936_293652


namespace fourth_number_in_sequence_l2936_293665

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence
  (a : ℕ → ℕ)
  (h_seq : fibonacci_like_sequence a)
  (h_7 : a 7 = 42)
  (h_9 : a 9 = 110) :
  a 4 = 10 := by
  sorry

end fourth_number_in_sequence_l2936_293665


namespace line_through_points_l2936_293610

/-- 
A line in a rectangular coordinate system is defined by the equation x = 5y + 5.
This line passes through two points (m, n) and (m + 2, n + p).
The theorem proves that under these conditions, p must equal 2/5.
-/
theorem line_through_points (m n : ℝ) : 
  (m = 5 * n + 5) → 
  (m + 2 = 5 * (n + p) + 5) → 
  p = 2/5 :=
by
  sorry

end line_through_points_l2936_293610


namespace root_range_implies_k_range_l2936_293645

theorem root_range_implies_k_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
    2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
    |x₁ - 2*n| = k * Real.sqrt x₁ ∧
    |x₂ - 2*n| = k * Real.sqrt x₂) →
  0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1) :=
by sorry

end root_range_implies_k_range_l2936_293645


namespace lottery_theorem_l2936_293612

-- Define the lottery setup
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

-- Define the probability of drawing a red ball first given a white ball second
def prob_red_given_white : ℚ := 5/11

-- Define the probabilities for the distribution of red balls drawn
def prob_zero_red : ℚ := 27/125
def prob_one_red : ℚ := 549/1000
def prob_two_red : ℚ := 47/200

-- Define the expected number of red balls drawn
def expected_red_balls : ℚ := 1019/1000

-- Theorem statement
theorem lottery_theorem :
  (total_balls = red_balls + white_balls) →
  (prob_red_given_white = 5/11) ∧
  (prob_zero_red + prob_one_red + prob_two_red = 1) ∧
  (expected_red_balls = 0 * prob_zero_red + 1 * prob_one_red + 2 * prob_two_red) :=
by sorry

end lottery_theorem_l2936_293612


namespace find_number_from_announcements_l2936_293601

def circle_number_game (numbers : Fin 15 → ℝ) (announcements : Fin 15 → ℝ) : Prop :=
  ∀ i : Fin 15, announcements i = (numbers (i - 1) + numbers (i + 1)) / 2

theorem find_number_from_announcements 
  (numbers : Fin 15 → ℝ) (announcements : Fin 15 → ℝ)
  (h_circle : circle_number_game numbers announcements)
  (h_8th : announcements 7 = 10)
  (h_exists_5 : ∃ j : Fin 15, announcements j = 5) :
  ∃ k : Fin 15, announcements k = 5 ∧ numbers k = 0 := by
sorry

end find_number_from_announcements_l2936_293601


namespace jasmine_purchase_cost_l2936_293630

/-- The cost calculation for Jasmine's purchase of coffee beans and milk. -/
theorem jasmine_purchase_cost :
  let coffee_beans_pounds : ℕ := 4
  let milk_gallons : ℕ := 2
  let coffee_bean_price_per_pound : ℚ := 5/2
  let milk_price_per_gallon : ℚ := 7/2
  let total_cost : ℚ := coffee_beans_pounds * coffee_bean_price_per_pound + milk_gallons * milk_price_per_gallon
  total_cost = 17 := by sorry

end jasmine_purchase_cost_l2936_293630


namespace knicks_knacks_knocks_conversion_l2936_293643

/-- Given the conversion rates between knicks, knacks, and knocks, 
    prove that 36 knocks are equal to 40 knicks. -/
theorem knicks_knacks_knocks_conversion :
  (∀ (knicks knacks knocks : ℚ),
    5 * knicks = 3 * knacks →
    4 * knacks = 6 * knocks →
    36 * knocks = 40 * knicks) :=
by sorry

end knicks_knacks_knocks_conversion_l2936_293643


namespace same_root_value_l2936_293654

theorem same_root_value (a b c d : ℝ) (h : a ≠ c) :
  ∀ α : ℝ, (α^2 + a*α + b = 0 ∧ α^2 + c*α + d = 0) → α = (d - b) / (a - c) := by
  sorry

end same_root_value_l2936_293654


namespace pet_store_puppies_l2936_293676

theorem pet_store_puppies (sold : ℕ) (cages : ℕ) (puppies_per_cage : ℕ)
  (h1 : sold = 30)
  (h2 : cages = 6)
  (h3 : puppies_per_cage = 8) :
  sold + cages * puppies_per_cage = 78 :=
by sorry

end pet_store_puppies_l2936_293676


namespace sequence_ratio_l2936_293639

/-- Given two sequences, one arithmetic and one geometric, prove that (a₂ - a₁) / b₂ = 1/2 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℚ) : 
  ((-1 : ℚ) - a₁ = a₁ - a₂) ∧ 
  (a₁ - a₂ = a₂ - (-4)) ∧ 
  ((-1 : ℚ) * b₁ = b₁ * b₂) ∧ 
  (b₁ * b₂ = b₂ * b₃) ∧ 
  (b₂ * b₃ = b₃ * (-4)) → 
  (a₂ - a₁) / b₂ = 1/2 := by
sorry

end sequence_ratio_l2936_293639


namespace range_of_a_l2936_293664

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (4*x - 3) ≤ 1
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- State the theorem
theorem range_of_a : 
  (∀ x a : ℝ, q x a → p x) ∧ 
  (∃ x : ℝ, p x ∧ ∀ a : ℝ, ¬(q x a)) →
  ∀ a : ℝ, (0 ≤ a ∧ a ≤ 1/2) ↔ (∃ x : ℝ, q x a) :=
sorry

end range_of_a_l2936_293664


namespace cos_105_degrees_l2936_293615

theorem cos_105_degrees :
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_degrees_l2936_293615


namespace shoes_theorem_l2936_293656

/-- The number of pairs of shoes Ellie, Riley, and Jordan have in total -/
def total_shoes (ellie riley jordan : ℕ) : ℕ := ellie + riley + jordan

/-- The theorem stating the total number of shoes given the conditions -/
theorem shoes_theorem (ellie riley jordan : ℕ) 
  (h1 : ellie = 8)
  (h2 : riley = ellie - 3)
  (h3 : jordan = ((ellie + riley) * 3) / 2) :
  total_shoes ellie riley jordan = 32 := by
  sorry

end shoes_theorem_l2936_293656


namespace min_stamps_for_50_cents_l2936_293636

/-- Represents the number of ways to make 50 cents using 5 cent and 7 cent stamps -/
def stamp_combinations : Set (ℕ × ℕ) :=
  {(s, t) | 5 * s + 7 * t = 50 ∧ s ≥ 0 ∧ t ≥ 0}

/-- The total number of stamps used in a combination -/
def total_stamps (combination : ℕ × ℕ) : ℕ :=
  combination.1 + combination.2

theorem min_stamps_for_50_cents :
  ∃ (combination : ℕ × ℕ),
    combination ∈ stamp_combinations ∧
    (∀ other ∈ stamp_combinations, total_stamps combination ≤ total_stamps other) ∧
    total_stamps combination = 8 :=
  sorry

end min_stamps_for_50_cents_l2936_293636


namespace flyer_multiple_l2936_293620

theorem flyer_multiple (maisie_flyers donna_flyers : ℕ) (h1 : maisie_flyers = 33) (h2 : donna_flyers = 71) :
  ∃ x : ℕ, donna_flyers = 5 + x * maisie_flyers ∧ x = 2 := by
  sorry

end flyer_multiple_l2936_293620


namespace perfect_square_divisibility_l2936_293619

theorem perfect_square_divisibility (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) :
  ∃ k : ℕ, a = k^2 := by
sorry

end perfect_square_divisibility_l2936_293619


namespace count_sequences_100_l2936_293694

/-- The number of sequences of length n, where each sequence contains at least one 4 or 5,
    and any two consecutive members differ by no more than 2. -/
def count_sequences (n : ℕ) : ℕ :=
  5^n - 3^n

/-- The theorem stating that the number of valid sequences of length 100 is 5^100 - 3^100. -/
theorem count_sequences_100 :
  count_sequences 100 = 5^100 - 3^100 :=
by sorry

end count_sequences_100_l2936_293694


namespace consecutive_odd_product_square_l2936_293642

theorem consecutive_odd_product_square : 
  ∃ (n : ℤ), (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3) = 9 := by
  sorry

end consecutive_odd_product_square_l2936_293642


namespace compound_interest_rate_equation_l2936_293638

/-- Proves that the given compound interest scenario results in the specified equation for the interest rate. -/
theorem compound_interest_rate_equation (P r : ℝ) 
  (h1 : P * (1 + r)^3 = 310) 
  (h2 : P * (1 + r)^8 = 410) : 
  (1 + r)^5 = 410/310 := by
  sorry

end compound_interest_rate_equation_l2936_293638


namespace eyes_seeing_airplane_l2936_293687

/-- Given 200 students on a field and 3/4 of them looking up at an airplane,
    prove that the number of eyes that saw the airplane is 300. -/
theorem eyes_seeing_airplane (total_students : ℕ) (fraction_looking_up : ℚ) : 
  total_students = 200 →
  fraction_looking_up = 3/4 →
  (total_students : ℚ) * fraction_looking_up * 2 = 300 :=
by
  sorry

end eyes_seeing_airplane_l2936_293687


namespace kimikos_age_l2936_293641

theorem kimikos_age (kayla_age kimiko_age min_driving_age wait_time : ℕ) : 
  kayla_age = kimiko_age / 2 →
  min_driving_age = 18 →
  kayla_age + wait_time = min_driving_age →
  wait_time = 5 →
  kimiko_age = 26 := by
sorry

end kimikos_age_l2936_293641


namespace solve_system_of_equations_no_solution_for_inequalities_l2936_293663

-- Part 1: System of equations
theorem solve_system_of_equations :
  ∃! (x y : ℝ), x - 3 * y = -5 ∧ 2 * x + 2 * y = 6 ∧ x = 1 ∧ y = 2 :=
by sorry

-- Part 2: System of inequalities
theorem no_solution_for_inequalities :
  ¬∃ (x : ℝ), 2 * x < -4 ∧ (1/2) * x - 5 > 1 - (3/2) * x :=
by sorry

end solve_system_of_equations_no_solution_for_inequalities_l2936_293663


namespace pages_difference_l2936_293659

/-- The number of pages Person A reads per day -/
def pages_per_day_A : ℕ := 8

/-- The number of pages Person B reads per day (when not resting) -/
def pages_per_day_B : ℕ := 13

/-- The total number of days -/
def total_days : ℕ := 7

/-- The number of days in Person B's reading cycle -/
def cycle_days : ℕ := 3

/-- The number of days Person B reads in a cycle -/
def reading_days_per_cycle : ℕ := 2

/-- Calculate the number of pages read by Person A -/
def pages_read_A : ℕ := total_days * pages_per_day_A

/-- Calculate the number of full cycles in the total days -/
def full_cycles : ℕ := total_days / cycle_days

/-- Calculate the number of days Person B reads -/
def reading_days_B : ℕ := full_cycles * reading_days_per_cycle + (total_days % cycle_days)

/-- Calculate the number of pages read by Person B -/
def pages_read_B : ℕ := reading_days_B * pages_per_day_B

/-- The theorem to prove -/
theorem pages_difference : pages_read_B - pages_read_A = 9 := by
  sorry

end pages_difference_l2936_293659


namespace psychology_majors_percentage_l2936_293658

/-- Given a college with the following properties:
  * 40% of total students are freshmen
  * 50% of freshmen are enrolled in the school of liberal arts
  * 10% of total students are freshmen psychology majors in the school of liberal arts
  Prove that 50% of freshmen in the school of liberal arts are psychology majors -/
theorem psychology_majors_percentage 
  (total_students : ℕ) 
  (freshmen_percent : ℚ) 
  (liberal_arts_percent : ℚ) 
  (psych_majors_percent : ℚ) 
  (h1 : freshmen_percent = 40 / 100) 
  (h2 : liberal_arts_percent = 50 / 100) 
  (h3 : psych_majors_percent = 10 / 100) : 
  (psych_majors_percent * total_students) / (freshmen_percent * liberal_arts_percent * total_students) = 50 / 100 := by
  sorry

end psychology_majors_percentage_l2936_293658


namespace orangeade_price_day1_l2936_293693

/-- Represents the price of orangeade per glass on a given day -/
structure OrangeadePrice where
  price : ℚ
  day : ℕ

/-- Represents the amount of orangeade made on a given day -/
structure OrangeadeAmount where
  amount : ℚ
  day : ℕ

/-- Represents the revenue from selling orangeade on a given day -/
def revenue (price : OrangeadePrice) (amount : OrangeadeAmount) : ℚ :=
  price.price * amount.amount

theorem orangeade_price_day1 (juice : ℚ) 
  (amount_day1 : OrangeadeAmount) 
  (amount_day2 : OrangeadeAmount)
  (price_day1 : OrangeadePrice)
  (price_day2 : OrangeadePrice) :
  amount_day1.amount = 2 * juice →
  amount_day2.amount = 3 * juice →
  amount_day1.day = 1 →
  amount_day2.day = 2 →
  price_day1.day = 1 →
  price_day2.day = 2 →
  price_day2.price = 2/5 →
  revenue price_day1 amount_day1 = revenue price_day2 amount_day2 →
  price_day1.price = 3/5 := by
  sorry

#eval (3 : ℚ) / 5  -- Should output 0.6

end orangeade_price_day1_l2936_293693


namespace hot_pepper_percentage_is_twenty_percent_l2936_293686

/-- Represents the total number of peppers picked by Joel over 7 days -/
def total_peppers : ℕ := 80

/-- Represents the number of non-hot peppers picked by Joel -/
def non_hot_peppers : ℕ := 64

/-- Calculates the percentage of hot peppers in Joel's garden -/
def hot_pepper_percentage : ℚ :=
  (total_peppers - non_hot_peppers : ℚ) / total_peppers * 100

/-- Proves that the percentage of hot peppers in Joel's garden is 20% -/
theorem hot_pepper_percentage_is_twenty_percent :
  hot_pepper_percentage = 20 := by
  sorry

end hot_pepper_percentage_is_twenty_percent_l2936_293686


namespace regression_not_exact_l2936_293657

-- Define the linear regression model
def linear_regression (x : ℝ) : ℝ := 0.5 * x - 85

-- Define the specific x value we're interested in
def x_value : ℝ := 200

-- Theorem stating that y is not necessarily exactly 15 when x = 200
theorem regression_not_exact : 
  ∃ (ε : ℝ), ε ≠ 0 ∧ linear_regression x_value + ε = 15 := by
  sorry

end regression_not_exact_l2936_293657


namespace jenna_one_way_distance_l2936_293632

/-- Calculates the one-way distance for a truck driver's round trip. -/
def one_way_distance (pay_rate : ℚ) (total_payment : ℚ) : ℚ :=
  (total_payment / pay_rate) / 2

/-- Proves that given a pay rate of $0.40 per mile and a total payment of $320 for a round trip, the one-way distance is 400 miles. -/
theorem jenna_one_way_distance :
  one_way_distance (40 / 100) 320 = 400 := by
  sorry

end jenna_one_way_distance_l2936_293632


namespace inequality_proof_l2936_293611

theorem inequality_proof (a b c d : ℝ) (h1 : a * b = 1) (h2 : a * c + b * d = 2) : c * d ≤ 1 := by
  sorry

end inequality_proof_l2936_293611


namespace bird_families_count_l2936_293627

theorem bird_families_count (africa asia left : ℕ) (h1 : africa = 23) (h2 : asia = 37) (h3 : left = 25) :
  africa + asia + left = 85 := by
  sorry

end bird_families_count_l2936_293627


namespace single_elimination_256_players_l2936_293699

/-- A single-elimination tournament structure -/
structure Tournament :=
  (num_players : ℕ)
  (is_single_elimination : Bool)

/-- The number of games needed to determine a champion in a single-elimination tournament -/
def games_to_champion (t : Tournament) : ℕ :=
  t.num_players - 1

/-- Theorem: In a single-elimination tournament with 256 players, 255 games are needed to determine the champion -/
theorem single_elimination_256_players :
  ∀ t : Tournament, t.num_players = 256 → t.is_single_elimination = true →
  games_to_champion t = 255 :=
by
  sorry

end single_elimination_256_players_l2936_293699


namespace cube_equality_condition_l2936_293698

/-- Represents a cube with edge length n -/
structure Cube (n : ℕ) where
  (edge_length : n > 3)

/-- The number of unit cubes with exactly two faces painted -/
def two_faces_painted (c : Cube n) : ℕ := 12 * (n - 4)

/-- The number of unit cubes with no faces painted -/
def no_faces_painted (c : Cube n) : ℕ := (n - 2)^3

/-- Theorem stating the equality condition for n = 5 -/
theorem cube_equality_condition (n : ℕ) (c : Cube n) :
  two_faces_painted c = no_faces_painted c ↔ n = 5 :=
sorry

end cube_equality_condition_l2936_293698


namespace integer_sum_problem_l2936_293655

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 8 → x * y = 120 → x + y = 4 * Real.sqrt 34 := by
  sorry

end integer_sum_problem_l2936_293655


namespace cow_profit_calculation_l2936_293653

def cow_profit (purchase_price : ℕ) (daily_food_cost : ℕ) (vaccination_cost : ℕ) (days : ℕ) (selling_price : ℕ) : ℕ :=
  selling_price - (purchase_price + daily_food_cost * days + vaccination_cost)

theorem cow_profit_calculation :
  cow_profit 600 20 500 40 2500 = 600 := by
  sorry

end cow_profit_calculation_l2936_293653


namespace probability_two_non_defective_pens_l2936_293634

/-- Given a box of pens, calculate the probability of selecting two non-defective pens. -/
theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (selected_pens : ℕ) 
  (h1 : total_pens = 10) 
  (h2 : defective_pens = 3) 
  (h3 : selected_pens = 2) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 7 / 15 := by
sorry


end probability_two_non_defective_pens_l2936_293634


namespace largest_monochromatic_subgraph_2024_l2936_293681

/-- A 3-coloring of the edges of a complete graph -/
def ThreeColoring (n : ℕ) := Fin 3 → Sym2 (Fin n)

/-- A function that returns the size of the largest monochromatic connected subgraph -/
noncomputable def largestMonochromaticSubgraph (n : ℕ) (coloring : ThreeColoring n) : ℕ := sorry

theorem largest_monochromatic_subgraph_2024 :
  ∀ (coloring : ThreeColoring 2024),
  largestMonochromaticSubgraph 2024 coloring ≥ 1012 := by sorry

end largest_monochromatic_subgraph_2024_l2936_293681


namespace gcd_1037_425_l2936_293647

theorem gcd_1037_425 : Nat.gcd 1037 425 = 17 := by
  sorry

end gcd_1037_425_l2936_293647


namespace element_n3_l2936_293671

/-- Represents a right triangular number array where each column forms an arithmetic sequence
    and each row (starting from the third row) forms a geometric sequence with a constant common ratio. -/
structure TriangularArray where
  -- a[i][j] represents the element in the i-th row and j-th column
  a : Nat → Nat → Rat
  -- Each column forms an arithmetic sequence
  column_arithmetic : ∀ i j k, i ≥ j → k ≥ j → a (i+1) j - a i j = a (k+1) j - a k j
  -- Each row forms a geometric sequence (starting from the third row)
  row_geometric : ∀ i j, i ≥ 3 → j < i → a i (j+1) / a i j = a i (j+2) / a i (j+1)

/-- The element a_{n3} in the n-th row and 3rd column is equal to n/16 -/
theorem element_n3 (arr : TriangularArray) (n : Nat) :
  arr.a n 3 = n / 16 := by
  sorry

end element_n3_l2936_293671


namespace sufficient_not_necessary_l2936_293613

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 6 = 0

-- Define the condition x = 2
def condition (x : ℝ) : Prop := x = 2

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ x : ℝ, quadratic_equation x ↔ (x = 2 ∨ x = 3)) →
  (∀ x : ℝ, condition x → quadratic_equation x) ∧
  ¬(∀ x : ℝ, quadratic_equation x → condition x) :=
by sorry

end sufficient_not_necessary_l2936_293613


namespace max_value_of_z_l2936_293690

theorem max_value_of_z (x y : ℝ) 
  (h1 : |2*x + y + 1| ≤ |x + 2*y + 2|) 
  (h2 : -1 ≤ y) (h3 : y ≤ 1) : 
  (∀ (x' y' : ℝ), |2*x' + y' + 1| ≤ |x' + 2*y' + 2| → -1 ≤ y' → y' ≤ 1 → 2*x' + y' ≤ 2*x + y) →
  2*x + y = 5 := by sorry

end max_value_of_z_l2936_293690


namespace solve_equation_l2936_293679

theorem solve_equation (x : ℝ) : 11 + Real.sqrt (-4 + 6 * x / 3) = 13 → x = 4 := by
  sorry

end solve_equation_l2936_293679


namespace percentage_in_quarters_calculation_l2936_293669

/-- Given a collection of coins, calculate the percentage of the total value that is in quarters. -/
def percentageInQuarters (dimes nickels quarters : ℕ) : ℚ :=
  let dimesValue : ℕ := dimes * 10
  let nickelsValue : ℕ := nickels * 5
  let quartersValue : ℕ := quarters * 25
  let totalValue : ℕ := dimesValue + nickelsValue + quartersValue
  (quartersValue : ℚ) / (totalValue : ℚ) * 100

theorem percentage_in_quarters_calculation :
  percentageInQuarters 70 40 30 = 750 / 1650 * 100 := by
  sorry

end percentage_in_quarters_calculation_l2936_293669


namespace ratio_calculation_l2936_293628

theorem ratio_calculation (A B C : ℚ) (h : A = 2 * B ∧ C = 4 * B) :
  (3 * A + 2 * B) / (4 * C - A) = 4 / 7 := by
  sorry

end ratio_calculation_l2936_293628


namespace probability_under_20_l2936_293674

theorem probability_under_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 150) (h2 : over_30 = 90) :
  let under_20 := total - over_30
  (under_20 : ℚ) / total = 2/5 := by sorry

end probability_under_20_l2936_293674


namespace rose_difference_after_changes_l2936_293622

/-- Calculates the difference in red roses between two people after changes -/
def rose_difference (santiago_initial : ℕ) (garrett_initial : ℕ) (given_away : ℕ) (received : ℕ) : ℕ :=
  (santiago_initial - given_away + received) - (garrett_initial - given_away + received)

theorem rose_difference_after_changes :
  rose_difference 58 24 10 5 = 34 := by
  sorry

end rose_difference_after_changes_l2936_293622


namespace go_game_competition_l2936_293682

/-- Represents the probability of a player winning a single game -/
structure GameProbability where
  player_a : ℝ
  player_b : ℝ
  sum_to_one : player_a + player_b = 1

/-- Represents the state of the game after the first two games -/
structure GameState where
  a_wins : ℕ
  b_wins : ℕ
  total_games : a_wins + b_wins = 2

/-- The probability of the competition ending after 2 more games -/
def probability_end_in_two_more_games (p : GameProbability) : ℝ :=
  p.player_a * p.player_a + p.player_b * p.player_b

/-- The probability of player A winning the competition -/
def probability_a_wins (p : GameProbability) : ℝ :=
  p.player_a * p.player_a + 
  p.player_b * p.player_a * p.player_a + 
  p.player_a * p.player_b * p.player_a

theorem go_game_competition 
  (p : GameProbability) 
  (state : GameState) 
  (h_p : p.player_a = 0.6 ∧ p.player_b = 0.4) 
  (h_state : state.a_wins = 1 ∧ state.b_wins = 1) : 
  probability_end_in_two_more_games p = 0.52 ∧ 
  probability_a_wins p = 0.648 := by
  sorry


end go_game_competition_l2936_293682


namespace expression_value_at_sqrt3_over_2_l2936_293640

theorem expression_value_at_sqrt3_over_2 :
  let x : ℝ := Real.sqrt 3 / 2
  (1 + x) / (1 + Real.sqrt (1 + x)) + (1 - x) / (1 - Real.sqrt (1 - x)) = 1 := by
  sorry

end expression_value_at_sqrt3_over_2_l2936_293640


namespace some_students_not_club_members_l2936_293688

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (ClubMember : U → Prop)
variable (StudiesLate : U → Prop)

-- State the theorem
theorem some_students_not_club_members
  (h1 : ∃ x, Student x ∧ ¬StudiesLate x)
  (h2 : ∀ x, ClubMember x → StudiesLate x) :
  ∃ x, Student x ∧ ¬ClubMember x :=
by
  sorry


end some_students_not_club_members_l2936_293688


namespace two_layer_triangle_structure_l2936_293648

/-- Calculates the number of small triangles in a layer given the number of triangles in the base row -/
def trianglesInLayer (baseTriangles : ℕ) : ℕ :=
  (baseTriangles * (baseTriangles + 1)) / 2

/-- Calculates the total number of toothpicks required for the two-layer structure -/
def totalToothpicks (lowerBaseTriangles upperBaseTriangles : ℕ) : ℕ :=
  let lowerTriangles := trianglesInLayer lowerBaseTriangles
  let upperTriangles := trianglesInLayer upperBaseTriangles
  let totalTriangles := lowerTriangles + upperTriangles
  let totalEdges := 3 * totalTriangles
  let boundaryEdges := 3 * lowerBaseTriangles + 3 * upperBaseTriangles - 3
  (totalEdges - boundaryEdges) / 2 + boundaryEdges

/-- The main theorem stating that the structure with 100 triangles in the lower base
    and 99 in the upper base requires 15596 toothpicks -/
theorem two_layer_triangle_structure :
  totalToothpicks 100 99 = 15596 := by
  sorry


end two_layer_triangle_structure_l2936_293648


namespace basketball_not_table_tennis_l2936_293606

theorem basketball_not_table_tennis 
  (total : ℕ) 
  (basketball : ℕ) 
  (table_tennis : ℕ) 
  (neither : ℕ) 
  (h1 : total = 30) 
  (h2 : basketball = 15) 
  (h3 : table_tennis = 10) 
  (h4 : neither = 8) :
  ∃ (both : ℕ), 
    basketball - both = 12 ∧ 
    total = (basketball - both) + (table_tennis - both) + both + neither :=
by sorry

end basketball_not_table_tennis_l2936_293606


namespace quadrilateral_max_area_and_angles_l2936_293603

/-- A quadrilateral with two sides of length 3 and two sides of length 4 -/
structure Quadrilateral :=
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ)
  (side1_eq_3 : side1 = 3)
  (side2_eq_4 : side2 = 4)
  (side3_eq_3 : side3 = 3)
  (side4_eq_4 : side4 = 4)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- The angles of a quadrilateral -/
def angles (q : Quadrilateral) : Fin 4 → ℝ := sorry

/-- The sum of two opposite angles in a quadrilateral -/
def opposite_angles_sum (q : Quadrilateral) : ℝ := 
  angles q 0 + angles q 2

theorem quadrilateral_max_area_and_angles (q : Quadrilateral) : 
  (∀ q' : Quadrilateral, area q' ≤ area q) → 
  (area q = 12 ∧ opposite_angles_sum q = 180) := by sorry

end quadrilateral_max_area_and_angles_l2936_293603


namespace mod_inverse_sum_equals_26_l2936_293608

theorem mod_inverse_sum_equals_26 : ∃ x y : ℤ, 
  (0 ≤ x ∧ x < 31) ∧ 
  (0 ≤ y ∧ y < 31) ∧ 
  (5 * x ≡ 1 [ZMOD 31]) ∧ 
  (5 * 5 * y ≡ 1 [ZMOD 31]) ∧ 
  ((x + y) % 31 = 26) := by
  sorry

end mod_inverse_sum_equals_26_l2936_293608


namespace red_peaches_count_l2936_293685

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := sorry

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 11

/-- The difference between green and red peaches -/
def difference : ℕ := 6

/-- Theorem stating that the number of red peaches is 5 -/
theorem red_peaches_count : red_peaches = 5 := by
  sorry

/-- The relationship between green and red peaches -/
axiom green_red_relation : green_peaches = red_peaches + difference


end red_peaches_count_l2936_293685


namespace expand_and_subtract_l2936_293692

theorem expand_and_subtract (x : ℝ) : (x + 3) * (2 * x - 5) - (2 * x + 1) = 2 * x^2 - x - 16 := by
  sorry

end expand_and_subtract_l2936_293692


namespace x_value_theorem_l2936_293602

theorem x_value_theorem (x n : ℕ) :
  x = 2^n - 32 ∧
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 3 ∧ q ≠ 3 ∧ 
    (∀ r : ℕ, Prime r → r ∣ x ↔ r = 3 ∨ r = p ∨ r = q)) →
  x = 480 ∨ x = 2016 := by
sorry

end x_value_theorem_l2936_293602


namespace first_fibonacci_exceeding_3000_l2936_293621

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem first_fibonacci_exceeding_3000 :
  (∀ k < 19, fibonacci k ≤ 3000) ∧ fibonacci 19 > 3000 := by sorry

end first_fibonacci_exceeding_3000_l2936_293621


namespace divisors_of_10n_l2936_293649

/-- Given a natural number n where 100n^2 has exactly 55 different natural divisors,
    prove that 10n has exactly 18 natural divisors. -/
theorem divisors_of_10n (n : ℕ) (h : (Nat.divisors (100 * n^2)).card = 55) :
  (Nat.divisors (10 * n)).card = 18 :=
sorry

end divisors_of_10n_l2936_293649


namespace digit_difference_in_base_d_l2936_293661

/-- Given two digits A and B in base d > 6, if ̅AB_d + ̅AA_d = 162_d, then A_d - B_d = 3_d -/
theorem digit_difference_in_base_d (d A B : ℕ) (h1 : d > 6) 
  (h2 : A < d) (h3 : B < d) 
  (h4 : A * d + B + A * d + A = 1 * d^2 + 6 * d + 2) : 
  A - B = 3 := by
sorry

end digit_difference_in_base_d_l2936_293661


namespace inequality_solution_set_l2936_293678

theorem inequality_solution_set (a : ℝ) (h : a > 0) :
  let solution_set := {x : ℝ | a * (x - 1) / (x - 2) > 1}
  (a = 1 → solution_set = {x : ℝ | x > 2}) ∧
  (0 < a ∧ a < 1 → solution_set = {x : ℝ | (a - 2) / (1 - a) < x ∧ x < 2}) ∧
  (a > 1 → solution_set = {x : ℝ | x < (a - 2) / (a - 1) ∨ x > 2}) := by
  sorry

end inequality_solution_set_l2936_293678


namespace larger_number_is_eleven_l2936_293695

theorem larger_number_is_eleven (x y : ℝ) (h1 : y - x = 2) (h2 : x + y = 20) : 
  max x y = 11 := by
sorry

end larger_number_is_eleven_l2936_293695


namespace hyperbola_asymptotic_lines_l2936_293650

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := 3 * x^2 - y^2 = 3

/-- The asymptotic lines equation -/
def asymptotic_lines_eq (x y : ℝ) : Prop := y^2 = 3 * x^2

/-- Theorem: The asymptotic lines of the hyperbola 3x^2 - y^2 = 3 are y = ± √3x -/
theorem hyperbola_asymptotic_lines :
  ∀ x y : ℝ, hyperbola_eq x y → asymptotic_lines_eq x y :=
by
  sorry

end hyperbola_asymptotic_lines_l2936_293650


namespace division_vs_multiplication_error_l2936_293672

theorem division_vs_multiplication_error (x : ℝ) (h : x > 0) :
  ∃ (ε : ℝ), abs (ε - 98) < 1 ∧
  (abs ((8 * x) - (x / 8)) / (8 * x)) * 100 = ε :=
sorry

end division_vs_multiplication_error_l2936_293672


namespace widget_price_reduction_l2936_293624

theorem widget_price_reduction (total_money : ℝ) (original_quantity : ℕ) (reduced_quantity : ℕ) :
  total_money = 27.60 ∧ original_quantity = 6 ∧ reduced_quantity = 8 →
  (total_money / original_quantity) - (total_money / reduced_quantity) = 1.15 := by
  sorry

end widget_price_reduction_l2936_293624


namespace min_matches_25_players_l2936_293668

/-- Represents a chess tournament. -/
structure ChessTournament where
  numPlayers : ℕ
  skillLevels : Fin numPlayers → ℕ
  uniqueSkills : ∀ i j, i ≠ j → skillLevels i ≠ skillLevels j

/-- The minimum number of matches required to determine the two strongest players. -/
def minMatchesForTopTwo (tournament : ChessTournament) : ℕ :=
  -- Definition to be proved
  28

/-- Theorem stating the minimum number of matches for a 25-player tournament. -/
theorem min_matches_25_players (tournament : ChessTournament) 
  (h_players : tournament.numPlayers = 25) :
  minMatchesForTopTwo tournament = 28 := by
  sorry

#check min_matches_25_players

end min_matches_25_players_l2936_293668


namespace inscribed_circle_radius_l2936_293618

/-- Given three non-overlapping circles with radii r₁, r₂, r₃ where r₁ > r₂ and r₁ > r₃,
    the quadrilateral formed by their external common tangents has an inscribed circle
    with radius r = (r₁r₂r₃) / (r₁r₂ - r₁r₃ - r₂r₃) -/
theorem inscribed_circle_radius
  (r₁ r₂ r₃ : ℝ)
  (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₃ > 0)
  (h₄ : r₁ > r₂) (h₅ : r₁ > r₃)
  (h_non_overlap : r₁ < r₂ + r₃) :
  ∃ r : ℝ, r > 0 ∧ r = (r₁ * r₂ * r₃) / (r₁ * r₂ - r₁ * r₃ - r₂ * r₃) :=
by sorry

end inscribed_circle_radius_l2936_293618


namespace exists_bound_factorial_digit_sum_l2936_293680

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number b such that for all natural numbers n > b,
    the sum of the digits of n! is greater than or equal to 10^100 -/
theorem exists_bound_factorial_digit_sum :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (n.factorial) ≥ 10^100 := by sorry

end exists_bound_factorial_digit_sum_l2936_293680


namespace edward_tickets_l2936_293696

theorem edward_tickets (booth_tickets : ℕ) (ride_cost : ℕ) (num_rides : ℕ) : 
  booth_tickets = 23 → ride_cost = 7 → num_rides = 8 →
  ∃ total_tickets : ℕ, total_tickets = booth_tickets + ride_cost * num_rides :=
by
  sorry

end edward_tickets_l2936_293696


namespace initial_flour_amount_l2936_293604

theorem initial_flour_amount (initial : ℕ) : 
  initial + 2 = 10 → initial = 8 := by
  sorry

end initial_flour_amount_l2936_293604


namespace fifth_term_of_arithmetic_sequence_l2936_293607

/-- Given an arithmetic sequence {aₙ}, Sₙ represents the sum of its first n terms -/
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := sorry

/-- aₙ is an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := sorry

theorem fifth_term_of_arithmetic_sequence (a : ℕ → ℝ) :
  is_arithmetic_sequence a → S a 9 = 45 → a 5 = 5 := by sorry

end fifth_term_of_arithmetic_sequence_l2936_293607


namespace parabola_chord_length_l2936_293670

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop := ∃ t : ℝ, x = 1 + t ∧ y = t

-- Define the intersection points
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y
  on_line : line_through_focus x y

-- Theorem statement
theorem parabola_chord_length 
  (A B : IntersectionPoint) 
  (sum_condition : A.x + B.x = 6) : 
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 8 := by
  sorry

end parabola_chord_length_l2936_293670
