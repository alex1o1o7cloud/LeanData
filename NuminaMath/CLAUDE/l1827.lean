import Mathlib

namespace career_preference_theorem_l1827_182771

/-- Represents the ratio of boys to girls in a class -/
def boy_girl_ratio : ℚ := 2 / 3

/-- Represents the fraction of boys who prefer the career -/
def boy_preference : ℚ := 1 / 3

/-- Represents the fraction of girls who prefer the career -/
def girl_preference : ℚ := 2 / 3

/-- Calculates the degrees in a circle graph for a given career preference -/
def career_preference_degrees (ratio : ℚ) (boy_pref : ℚ) (girl_pref : ℚ) : ℚ :=
  360 * ((ratio * boy_pref + girl_pref) / (ratio + 1))

/-- Theorem stating that the career preference degrees is 192 -/
theorem career_preference_theorem :
  career_preference_degrees boy_girl_ratio boy_preference girl_preference = 192 := by
  sorry

#eval career_preference_degrees boy_girl_ratio boy_preference girl_preference

end career_preference_theorem_l1827_182771


namespace b_share_is_1500_l1827_182709

/-- Calculates the share of the second child (B) when distributing money among three children in a given ratio -/
def calculate_b_share (total_money : ℚ) (ratio_a ratio_b ratio_c : ℕ) : ℚ :=
  let total_parts := ratio_a + ratio_b + ratio_c
  let part_value := total_money / total_parts
  ratio_b * part_value

/-- Theorem stating that given $4500 distributed in the ratio 2:3:4, B's share is $1500 -/
theorem b_share_is_1500 :
  calculate_b_share 4500 2 3 4 = 1500 := by
  sorry

#eval calculate_b_share 4500 2 3 4

end b_share_is_1500_l1827_182709


namespace expression_simplification_l1827_182717

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 1) / x / (x - 1 / x) = Real.sqrt 2 - 1 := by
  sorry

end expression_simplification_l1827_182717


namespace quadratic_expression_value_l1827_182739

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3 * x + y = 7) 
  (h2 : x + 3 * y = 8) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 113 := by
  sorry

end quadratic_expression_value_l1827_182739


namespace negation_of_forall_positive_negation_of_proposition_l1827_182724

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∀ x > 0, x^2 + x > 0) ↔ (∃ x > 0, x^2 + x ≤ 0) :=
by sorry

end negation_of_forall_positive_negation_of_proposition_l1827_182724


namespace wheat_packets_in_gunny_bag_l1827_182761

/-- The maximum number of wheat packets that can be accommodated in a gunny bag -/
def max_wheat_packets (bag_capacity : ℝ) (ton_to_kg : ℝ) (kg_to_g : ℝ) 
  (packet_weight_pounds : ℝ) (packet_weight_ounces : ℝ) 
  (pound_to_kg : ℝ) (ounce_to_g : ℝ) : ℕ :=
  sorry

/-- Theorem stating the maximum number of wheat packets in the gunny bag -/
theorem wheat_packets_in_gunny_bag : 
  max_wheat_packets 13 1000 1000 16 4 0.453592 28.3495 = 1763 := by
  sorry

end wheat_packets_in_gunny_bag_l1827_182761


namespace tangent_line_implies_a_value_l1827_182780

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * x + 1

-- State the theorem
theorem tangent_line_implies_a_value (a : ℝ) (h1 : a ≠ 0) :
  (∃ (m : ℝ), (∀ x : ℝ, x + f a x - 2 = m * (x - 1)) ∧ 
               (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → 
                 |(f a x - f a 1) - m * (x - 1)| < ε * |x - 1|)) →
  a = -1 :=
by sorry

end tangent_line_implies_a_value_l1827_182780


namespace function_domain_range_implies_interval_l1827_182751

open Real

theorem function_domain_range_implies_interval (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (-1/2) (1/4)) →
  (∀ x, f x = sin x * sin (x + π/3) - 1/4) →
  m < n →
  π/3 ≤ n - m ∧ n - m ≤ 2*π/3 := by
  sorry

end function_domain_range_implies_interval_l1827_182751


namespace journey_matches_graph_l1827_182756

/-- Represents a segment of a journey --/
inductive JourneySegment
  | SlowAway
  | FastAway
  | Stationary
  | FastTowards
  | SlowTowards

/-- Represents a complete journey --/
def Journey := List JourneySegment

/-- Represents the shape of a graph segment --/
inductive GraphSegment
  | GradualIncline
  | SteepIncline
  | FlatLine
  | SteepDecline
  | GradualDecline

/-- Represents a complete graph --/
def Graph := List GraphSegment

/-- The journey we're analyzing --/
def janesJourney : Journey :=
  [JourneySegment.SlowAway, JourneySegment.FastAway, JourneySegment.Stationary,
   JourneySegment.FastTowards, JourneySegment.SlowTowards]

/-- The correct graph representation --/
def correctGraph : Graph :=
  [GraphSegment.GradualIncline, GraphSegment.SteepIncline, GraphSegment.FlatLine,
   GraphSegment.SteepDecline, GraphSegment.GradualDecline]

/-- Function to convert a journey to its graph representation --/
def journeyToGraph (j : Journey) : Graph :=
  sorry

/-- Theorem stating that the journey converts to the correct graph --/
theorem journey_matches_graph : journeyToGraph janesJourney = correctGraph := by
  sorry

end journey_matches_graph_l1827_182756


namespace evaluate_expression_l1827_182750

theorem evaluate_expression : Real.sqrt (Real.sqrt 81) + Real.sqrt 256 - Real.sqrt 49 = 12 := by
  sorry

end evaluate_expression_l1827_182750


namespace jim_diving_hours_l1827_182769

/-- The number of gold coins Jim finds per hour -/
def gold_coins_per_hour : ℕ := 25

/-- The number of gold coins in the treasure chest -/
def chest_coins : ℕ := 100

/-- The number of smaller bags Jim found -/
def num_smaller_bags : ℕ := 2

/-- The number of gold coins in each smaller bag -/
def coins_per_smaller_bag : ℕ := chest_coins / 2

/-- The total number of gold coins Jim found -/
def total_coins : ℕ := chest_coins + num_smaller_bags * coins_per_smaller_bag

/-- Theorem: Jim spent 8 hours scuba diving -/
theorem jim_diving_hours : total_coins / gold_coins_per_hour = 8 := by
  sorry

end jim_diving_hours_l1827_182769


namespace wage_difference_l1827_182714

/-- Represents the hourly wages at Joe's Steakhouse -/
structure JoesSteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ
  manager_wage : manager = 8.50
  dishwasher_wage : dishwasher = manager / 2
  chef_wage : chef = dishwasher * 1.20

/-- The difference between a manager's hourly wage and a chef's hourly wage is $3.40 -/
theorem wage_difference (w : JoesSteakhouseWages) : w.manager - w.chef = 3.40 := by
  sorry

end wage_difference_l1827_182714


namespace tinks_are_falars_and_gymes_l1827_182747

-- Define the types for our entities
variable (U : Type) -- Universe type
variable (Falar Gyme Halp Tink Isoy : Set U)

-- State the given conditions
variable (h1 : Falar ⊆ Gyme)
variable (h2 : Halp ⊆ Tink)
variable (h3 : Isoy ⊆ Falar)
variable (h4 : Tink ⊆ Isoy)

-- State the theorem to be proved
theorem tinks_are_falars_and_gymes : Tink ⊆ Falar ∧ Tink ⊆ Gyme := by
  sorry

end tinks_are_falars_and_gymes_l1827_182747


namespace complex_equation_solution_l1827_182755

theorem complex_equation_solution :
  let z : ℂ := -3 * I / 4
  2 - 3 * I * z = -4 + 5 * I * z :=
by
  sorry

end complex_equation_solution_l1827_182755


namespace prob_two_eggplants_germination_rate_expected_value_X_l1827_182793

-- Define the number of plots
def num_plots : ℕ := 4

-- Define the probability of planting eggplant in each plot
def prob_eggplant : ℚ := 1/3

-- Define the probability of planting cucumber in each plot
def prob_cucumber : ℚ := 2/3

-- Define the emergence rate of eggplant seeds
def emergence_rate_eggplant : ℚ := 95/100

-- Define the emergence rate of cucumber seeds
def emergence_rate_cucumber : ℚ := 98/100

-- Define the number of rows
def num_rows : ℕ := 2

-- Define the number of columns
def num_columns : ℕ := 2

-- Theorem for the probability of exactly 2 plots planting eggplants
theorem prob_two_eggplants : 
  (Nat.choose num_plots 2 : ℚ) * prob_eggplant^2 * prob_cucumber^2 = 8/27 := by sorry

-- Theorem for the germination rate of seeds for each plot
theorem germination_rate : 
  prob_eggplant * emergence_rate_eggplant + prob_cucumber * emergence_rate_cucumber = 97/100 := by sorry

-- Define the random variable X as the number of rows planting cucumbers
def X : Fin 3 → ℚ
| 0 => 1/25
| 1 => 16/25
| 2 => 8/25

-- Theorem for the expected value of X
theorem expected_value_X : 
  Finset.sum (Finset.range 3) (λ i => (i : ℚ) * X i) = 32/25 := by sorry

end prob_two_eggplants_germination_rate_expected_value_X_l1827_182793


namespace sum_of_squares_l1827_182708

variables {x y z w a b c d : ℝ}

theorem sum_of_squares (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x * w = d)
  (h5 : a ≠ 0) (h6 : b ≠ 0) (h7 : d ≠ 0) :
  x^2 + y^2 + z^2 + w^2 = (a * b + b * d + d * a)^2 / (a * b * d) := by
  sorry

end sum_of_squares_l1827_182708


namespace distance_O_to_B_l1827_182727

/-- Represents a person moving on the road -/
structure Person where
  name : String
  startPosition : ℝ
  speed : ℝ

/-- Represents the road with three locations -/
structure Road where
  a : ℝ
  o : ℝ
  b : ℝ

/-- The main theorem stating the distance between O and B -/
theorem distance_O_to_B 
  (road : Road)
  (jia yi : Person)
  (h1 : road.o = 0)  -- Set O as the origin
  (h2 : road.a = -1360)  -- A is 1360 meters to the left of O
  (h3 : jia.startPosition = road.a)
  (h4 : yi.startPosition = road.o)
  (h5 : jia.speed * 10 + road.a = yi.speed * 10)  -- Equidistant at 10 minutes
  (h6 : jia.speed * 40 + road.a = road.b)  -- Meet at B at 40 minutes
  (h7 : yi.speed * 40 = road.b)  -- Yi also reaches B at 40 minutes
  : road.b = 2040 := by
  sorry


end distance_O_to_B_l1827_182727


namespace exponent_division_l1827_182786

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^15 / x^3 = x^12 := by
  sorry

end exponent_division_l1827_182786


namespace arithmetic_expressions_l1827_182782

theorem arithmetic_expressions :
  let expr1 := (3.6 - 0.8) * (1.8 + 2.05)
  let expr2 := (34.28 / 2) - (16.2 / 4)
  (expr1 = (3.6 - 0.8) * (1.8 + 2.05)) ∧
  (expr2 = (34.28 / 2) - (16.2 / 4)) := by sorry

end arithmetic_expressions_l1827_182782


namespace tom_coins_value_l1827_182715

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The number of quarters Tom found -/
def num_quarters : ℕ := 10

/-- The number of dimes Tom found -/
def num_dimes : ℕ := 3

/-- The number of nickels Tom found -/
def num_nickels : ℕ := 4

/-- The number of pennies Tom found -/
def num_pennies : ℕ := 200

/-- The total value of the coins Tom found in dollars -/
def total_value : ℚ := num_quarters * quarter_value + num_dimes * dime_value + 
                       num_nickels * nickel_value + num_pennies * penny_value

theorem tom_coins_value : total_value = 5 := by
  sorry

end tom_coins_value_l1827_182715


namespace divisor_problem_l1827_182777

theorem divisor_problem (x d : ℤ) (h1 : x % d = 7) (h2 : (x + 11) % 31 = 18) (h3 : d > 7) : d = 31 := by
  sorry

end divisor_problem_l1827_182777


namespace problem_solution_l1827_182753

theorem problem_solution : 
  (0.027 ^ (-1/3 : ℝ)) + (16 ^ 3) ^ (1/4 : ℝ) - 3⁻¹ + ((2 : ℝ).sqrt - 1) ^ (0 : ℝ) = 12 := by
  sorry

end problem_solution_l1827_182753


namespace curve_length_l1827_182700

/-- The length of a curve defined by the intersection of a plane and a sphere --/
theorem curve_length (x y z : ℝ) : 
  x + y + z = 8 → 
  x * y + y * z + x * z = -18 → 
  (∃ (l : ℝ), l = 4 * Real.pi * Real.sqrt (59 / 3) ∧ 
    l = 2 * Real.pi * Real.sqrt (100 - (8 * 8) / 3)) := by
  sorry

end curve_length_l1827_182700


namespace sixth_term_is_one_sixteenth_l1827_182723

/-- A geometric sequence with specific conditions -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  (∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q) ∧
  a 1 + a 3 = 5/2 ∧
  a 2 + a 4 = 5/4

/-- The sixth term of the geometric sequence is 1/16 -/
theorem sixth_term_is_one_sixteenth (a : ℕ → ℚ) (h : GeometricSequence a) :
  a 6 = 1/16 := by
  sorry

end sixth_term_is_one_sixteenth_l1827_182723


namespace horses_oats_meals_per_day_l1827_182730

/-- The number of horses Peter has -/
def num_horses : ℕ := 4

/-- The amount of oats each horse eats per meal (in pounds) -/
def oats_per_meal : ℕ := 4

/-- The amount of grain each horse eats per day (in pounds) -/
def grain_per_day : ℕ := 3

/-- The total amount of food needed for all horses for 3 days (in pounds) -/
def total_food_3days : ℕ := 132

/-- The number of days food is needed for -/
def num_days : ℕ := 3

/-- The number of times horses eat oats per day -/
def oats_meals_per_day : ℕ := 2

theorem horses_oats_meals_per_day : 
  num_days * num_horses * (oats_per_meal * oats_meals_per_day + grain_per_day) = total_food_3days :=
by sorry

end horses_oats_meals_per_day_l1827_182730


namespace odd_even_intersection_empty_l1827_182765

def odd_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def even_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem odd_even_intersection_empty :
  odd_integers ∩ even_integers = ∅ := by sorry

end odd_even_intersection_empty_l1827_182765


namespace first_friend_shells_l1827_182713

/-- Proves the amount of shells added by the first friend given initial conditions --/
theorem first_friend_shells (initial_shells : ℕ) (second_friend_shells : ℕ) (total_shells : ℕ)
  (h1 : initial_shells = 5)
  (h2 : second_friend_shells = 17)
  (h3 : total_shells = 37)
  : total_shells - initial_shells - second_friend_shells = 15 := by
  sorry

end first_friend_shells_l1827_182713


namespace polynomial_factorization_l1827_182760

/-- The polynomial with unknown coefficients a and b -/
def P (x a b : ℝ) : ℝ := 3 * x^4 + a * x^3 + 48 * x^2 + b * x + 12

/-- The given factor of the polynomial -/
def F (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 2

/-- Theorem stating that the polynomial P has the factor F when a = -26.5 and b = -40 -/
theorem polynomial_factorization (x : ℝ) : 
  ∃ (Q : ℝ → ℝ), P x (-26.5) (-40) = F x * Q x := by sorry

end polynomial_factorization_l1827_182760


namespace trigonometric_identities_l1827_182787

theorem trigonometric_identities (α β γ : Real) (h : α + β + γ = Real.pi) :
  let half_sum := (α + β) / 2
  let half_gamma := γ / 2
  (Real.sin half_sum - Real.cos half_gamma = 0) ∧
  (Real.tan half_gamma + Real.tan half_sum - (1 / Real.tan half_sum + 1 / Real.tan half_gamma) = 0) ∧
  (Real.sin half_sum ^ 2 + (1 / Real.tan half_sum) * (1 / Real.tan half_gamma) - Real.cos half_gamma ^ 2 = 1) ∧
  (Real.cos half_sum ^ 2 + Real.tan half_sum * Real.tan half_gamma + Real.cos half_gamma ^ 2 = 2) :=
by sorry

end trigonometric_identities_l1827_182787


namespace factorial_15_base_18_trailing_zeros_l1827_182710

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def countTrailingZerosBase18 (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

theorem factorial_15_base_18_trailing_zeros :
  countTrailingZerosBase18 (factorial 15) = 3 := by
  sorry

end factorial_15_base_18_trailing_zeros_l1827_182710


namespace smallest_dance_class_size_l1827_182733

theorem smallest_dance_class_size :
  ∀ n : ℕ,
  n > 40 →
  (∀ m : ℕ, m > 40 ∧ 5 * m + 2 < 5 * n + 2 → m = n) →
  5 * n + 2 = 207 :=
by
  sorry

end smallest_dance_class_size_l1827_182733


namespace first_number_is_55_l1827_182797

def number_list : List ℕ := [55, 57, 58, 59, 62, 62, 63, 65, 65]

theorem first_number_is_55 (average_is_60 : (number_list.sum / number_list.length : ℚ) = 60) :
  number_list.head? = some 55 := by
  sorry

end first_number_is_55_l1827_182797


namespace price_change_theorem_l1827_182784

theorem price_change_theorem (initial_price : ℝ) (initial_price_positive : 0 < initial_price) :
  let price_after_increase := initial_price * (1 + 0.35)
  let price_after_first_discount := price_after_increase * (1 - 0.10)
  let final_price := price_after_first_discount * (1 - 0.15)
  (final_price - initial_price) / initial_price = 0.03275 := by
sorry

end price_change_theorem_l1827_182784


namespace inscribed_circle_radius_l1827_182775

/-- A symmetric trapezoid with an inscribed and circumscribed circle -/
structure SymmetricTrapezoid where
  -- The lengths of the parallel sides
  a : ℝ
  b : ℝ
  -- The radius of the circumscribed circle
  R : ℝ
  -- The radius of the inscribed circle
  ρ : ℝ
  -- Conditions
  h_symmetric : a ≥ b
  h_R : R = 1
  h_inscribed : ρ > 0
  h_center_bisects : ∃ (K : ℝ × ℝ), K.1^2 + K.2^2 = (R/2)^2

/-- The radius of the inscribed circle in the symmetric trapezoid -/
theorem inscribed_circle_radius (T : SymmetricTrapezoid) : T.ρ = Real.sqrt (9/40) := by
  sorry

end inscribed_circle_radius_l1827_182775


namespace divisibility_of_concatenated_numbers_l1827_182752

theorem divisibility_of_concatenated_numbers (a b : ℕ) : 
  100 ≤ a ∧ a < 1000 →
  100 ≤ b ∧ b < 1000 →
  37 ∣ (a + b) →
  37 ∣ (1000 * a + b) := by
sorry

end divisibility_of_concatenated_numbers_l1827_182752


namespace problem_solution_l1827_182735

def f (x : ℝ) := |2 * x - 1|

def g (x : ℝ) := f x + f (x - 1)

theorem problem_solution :
  (∀ x : ℝ, f x < 4 ↔ -3/2 < x ∧ x < 5/2) ∧
  (∃ a : ℝ, (∀ x : ℝ, g x ≥ a) ∧
    ∀ m n : ℝ, m > 0 → n > 0 → m + n = a →
      2/m + 1/n ≥ 3/2 + Real.sqrt 2) :=
by sorry

end problem_solution_l1827_182735


namespace triangle_neg_three_four_l1827_182788

/-- The triangle operation -/
def triangle (a b : ℤ) : ℤ := a * b - a - b + 1

/-- Theorem stating that (-3) △ 4 = -12 -/
theorem triangle_neg_three_four : triangle (-3) 4 = -12 := by sorry

end triangle_neg_three_four_l1827_182788


namespace prob_gpa_at_least_3_75_l1827_182732

/-- Grade points for each letter grade -/
def gradePoints : Char → ℕ
| 'A' => 4
| 'B' => 3
| 'C' => 2
| 'D' => 1
| _ => 0

/-- Calculate GPA from total points -/
def calculateGPA (totalPoints : ℕ) : ℚ :=
  totalPoints / 4

/-- Probability of getting an A in English -/
def probAEnglish : ℚ := 1 / 3

/-- Probability of getting a B in English -/
def probBEnglish : ℚ := 1 / 2

/-- Probability of getting an A in History -/
def probAHistory : ℚ := 1 / 5

/-- Probability of getting a B in History -/
def probBHistory : ℚ := 1 / 2

theorem prob_gpa_at_least_3_75 :
  let mathPoints := gradePoints 'B'
  let sciencePoints := gradePoints 'B'
  let totalFixedPoints := mathPoints + sciencePoints
  let requiredPoints := 15
  let probBothA := probAEnglish * probAHistory
  probBothA = 1 / 15 ∧
  (∀ (englishGrade historyGrade : Char),
    calculateGPA (totalFixedPoints + gradePoints englishGrade + gradePoints historyGrade) ≥ 15 / 4 →
    (englishGrade = 'A' ∧ historyGrade = 'A')) →
  probBothA = 1 / 15 := by
sorry

end prob_gpa_at_least_3_75_l1827_182732


namespace water_left_in_bathtub_l1827_182740

/-- The amount of water left in a bathtub after a faucet drips and water evaporates --/
theorem water_left_in_bathtub
  (drip_rate : ℝ)
  (evap_rate : ℝ)
  (time : ℝ)
  (dumped : ℝ)
  (h1 : drip_rate = 40)
  (h2 : evap_rate = 200)
  (h3 : time = 9)
  (h4 : dumped = 12000) :
  drip_rate * time * 60 - evap_rate * time - dumped = 7800 :=
by sorry

end water_left_in_bathtub_l1827_182740


namespace eddie_study_games_l1827_182791

/-- Calculates the maximum number of games that can be played in a study block -/
def max_games (study_block_minutes : ℕ) (homework_minutes : ℕ) (game_duration : ℕ) : ℕ :=
  (study_block_minutes - homework_minutes) / game_duration

/-- Theorem stating that given the specific conditions, the maximum number of games is 7 -/
theorem eddie_study_games :
  max_games 60 25 5 = 7 := by
  sorry

end eddie_study_games_l1827_182791


namespace tower_height_calculation_l1827_182720

/-- Given a mountain and a tower, if the angles of depression from the top of the mountain
    to the top and bottom of the tower are as specified, then the height of the tower is 200m. -/
theorem tower_height_calculation (mountain_height : ℝ) (angle_to_top angle_to_bottom : ℝ) :
  mountain_height = 300 →
  angle_to_top = 30 * π / 180 →
  angle_to_bottom = 60 * π / 180 →
  ∃ (tower_height : ℝ), tower_height = 200 :=
by
  sorry

end tower_height_calculation_l1827_182720


namespace largest_n_factorial_sum_perfect_square_l1827_182790

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem largest_n_factorial_sum_perfect_square :
  (∀ n : ℕ, n > 3 → ¬(is_perfect_square (sum_factorials n))) ∧
  (is_perfect_square (sum_factorials 3)) ∧
  (∀ n : ℕ, n > 0 → n < 3 → ¬(is_perfect_square (sum_factorials n))) :=
sorry

end largest_n_factorial_sum_perfect_square_l1827_182790


namespace john_plays_two_periods_l1827_182763

def points_per_4_minutes : ℕ := 2 * 2 + 1 * 3
def minutes_per_period : ℕ := 12
def total_points : ℕ := 42

theorem john_plays_two_periods :
  (total_points / (points_per_4_minutes * (minutes_per_period / 4))) = 2 := by
  sorry

end john_plays_two_periods_l1827_182763


namespace tv_price_increase_l1827_182743

theorem tv_price_increase (initial_price : ℝ) (first_increase : ℝ) : 
  first_increase > 0 →
  (initial_price * (1 + first_increase / 100) * 1.4 = initial_price * 1.82) →
  first_increase = 30 := by
sorry

end tv_price_increase_l1827_182743


namespace fish_food_calculation_l1827_182772

/-- The total amount of food Layla needs to give her fish -/
def total_fish_food (goldfish : ℕ) (goldfish_food : ℚ)
                    (swordtails : ℕ) (swordtails_food : ℚ)
                    (guppies : ℕ) (guppies_food : ℚ)
                    (angelfish : ℕ) (angelfish_food : ℚ)
                    (tetra : ℕ) (tetra_food : ℚ) : ℚ :=
  goldfish * goldfish_food +
  swordtails * swordtails_food +
  guppies * guppies_food +
  angelfish * angelfish_food +
  tetra * tetra_food

theorem fish_food_calculation :
  total_fish_food 4 1 5 2 10 (1/2) 3 (3/2) 6 1 = 59/2 := by
  sorry

end fish_food_calculation_l1827_182772


namespace consecutive_integers_product_sum_l1827_182768

theorem consecutive_integers_product_sum :
  ∀ x y z : ℤ,
  (y = x + 1) →
  (z = y + 1) →
  (x * y * z = 336) →
  (x + y + z = 21) :=
by
  sorry

end consecutive_integers_product_sum_l1827_182768


namespace bridge_length_l1827_182785

theorem bridge_length 
  (left_bank : ℚ) 
  (right_bank : ℚ) 
  (river_width : ℚ) :
  left_bank = 1/4 →
  right_bank = 1/3 →
  river_width = 120 →
  (1 - left_bank - right_bank) * (288 : ℚ) = river_width :=
by sorry

end bridge_length_l1827_182785


namespace sphere_speeds_solution_l1827_182796

/-- Represents the speeds of two spheres moving towards the vertex of a right angle --/
structure SphereSpeeds where
  small : ℝ
  large : ℝ

/-- The problem setup and conditions --/
def sphereProblem (s : SphereSpeeds) : Prop :=
  let r₁ := 2 -- radius of smaller sphere
  let r₂ := 3 -- radius of larger sphere
  let d₁ := 6 -- initial distance of smaller sphere from vertex
  let d₂ := 16 -- initial distance of larger sphere from vertex
  let t₁ := 1 -- time after which distance between centers is measured
  let t₂ := 3 -- time at which spheres collide
  -- Initial positions
  (d₁ - s.small * t₁) ^ 2 + (d₂ - s.large * t₁) ^ 2 = 13 ^ 2 ∧
  -- Collision positions
  (d₁ - s.small * t₂) ^ 2 + (d₂ - s.large * t₂) ^ 2 = (r₁ + r₂) ^ 2

/-- The theorem stating the solution to the sphere problem --/
theorem sphere_speeds_solution :
  ∃ s : SphereSpeeds, sphereProblem s ∧ s.small = 1 ∧ s.large = 4 := by
  sorry

end sphere_speeds_solution_l1827_182796


namespace root_reciprocal_sum_l1827_182783

theorem root_reciprocal_sum (m : ℝ) : 
  (∃ α β : ℝ, α^2 - 2*(m+1)*α + m + 4 = 0 ∧ 
              β^2 - 2*(m+1)*β + m + 4 = 0 ∧ 
              α ≠ β ∧
              1/α + 1/β = 1) → 
  m = 2 := by
sorry

end root_reciprocal_sum_l1827_182783


namespace complex_magnitude_squared_l1827_182721

theorem complex_magnitude_squared (z : ℂ) (h : z + Complex.abs z = 2 + 8*I) : Complex.abs z ^ 2 = 289 := by
  sorry

end complex_magnitude_squared_l1827_182721


namespace parallel_vectors_m_value_l1827_182766

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -3)
  let b : ℝ × ℝ := (-2, m)
  parallel a b → m = 6 := by
  sorry

end parallel_vectors_m_value_l1827_182766


namespace initial_milk_percentage_l1827_182798

/-- Given a mixture of milk and water, prove the initial percentage of milk. -/
theorem initial_milk_percentage
  (total_initial_volume : ℝ)
  (added_water : ℝ)
  (final_milk_percentage : ℝ)
  (h1 : total_initial_volume = 60)
  (h2 : added_water = 40.8)
  (h3 : final_milk_percentage = 50) :
  (total_initial_volume * 84 / 100) / total_initial_volume = 
  (total_initial_volume * final_milk_percentage / 100) / (total_initial_volume + added_water) :=
by sorry

end initial_milk_percentage_l1827_182798


namespace urgent_painting_time_l1827_182734

/-- Represents the time required to paint an office -/
structure PaintingTime where
  painters : ℕ
  days : ℚ

/-- Represents the total work required to paint an office -/
def totalWork (pt : PaintingTime) : ℚ := pt.painters * pt.days

theorem urgent_painting_time 
  (first_office : PaintingTime)
  (second_office_normal : PaintingTime)
  (second_office_urgent : PaintingTime)
  (h1 : first_office.painters = 3)
  (h2 : first_office.days = 2)
  (h3 : second_office_normal.painters = 2)
  (h4 : totalWork first_office = totalWork second_office_normal)
  (h5 : second_office_urgent.painters = second_office_normal.painters)
  (h6 : second_office_urgent.days = 3/4 * second_office_normal.days) :
  second_office_urgent.days = 2.25 := by
  sorry

end urgent_painting_time_l1827_182734


namespace x_equals_negative_x_is_valid_l1827_182707

/-- An assignment statement is valid if it assigns a value to a variable -/
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (val : String), stmt = var ++ " = " ++ val

/-- The statement "x = -x" -/
def statement : String := "x = -x"

/-- Theorem: The statement "x = -x" is a valid assignment statement -/
theorem x_equals_negative_x_is_valid : is_valid_assignment statement := by
  sorry

end x_equals_negative_x_is_valid_l1827_182707


namespace standard_deck_two_card_selections_l1827_182737

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = suits * cards_per_suit)

/-- The number of ways to select two different cards from a deck, where order matters -/
def two_card_selections (d : Deck) : Nat :=
  d.total_cards * (d.total_cards - 1)

/-- Theorem: The number of ways to select two different cards from a standard deck of 52 cards, where order matters, is 2652 -/
theorem standard_deck_two_card_selections :
  ∃ (d : Deck), d.total_cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ two_card_selections d = 2652 := by
  sorry

end standard_deck_two_card_selections_l1827_182737


namespace omega_even_implies_periodic_l1827_182746

/-- Definition of an Ω function -/
def is_omega_function (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f x = T * f (x + T)

/-- Definition of an even function -/
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- Definition of a periodic function -/
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- Theorem: If f is an Ω function and even, then it is periodic -/
theorem omega_even_implies_periodic
  (f : ℝ → ℝ) (h_omega : is_omega_function f) (h_even : is_even_function f) :
  ∃ T : ℝ, T ≠ 0 ∧ is_periodic f (2 * T) :=
by sorry


end omega_even_implies_periodic_l1827_182746


namespace polynomial_factorization_l1827_182778

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 3*x + 2) * (x^2 + 7*x + 12) + (x^2 + 5*x - 6) = (x^2 + 5*x + 2) * (x^2 + 5*x + 9) := by
  sorry

end polynomial_factorization_l1827_182778


namespace choir_members_count_l1827_182745

theorem choir_members_count : ∃! n : ℕ, 
  150 ≤ n ∧ n ≤ 300 ∧ 
  n % 10 = 6 ∧ 
  n % 11 = 6 := by
sorry

end choir_members_count_l1827_182745


namespace ideal_solution_range_l1827_182722

theorem ideal_solution_range (m n q : ℝ) : 
  m + 2*n = 6 →
  2*m + n = 3*q →
  m + n > 1 →
  q > -1 := by
sorry

end ideal_solution_range_l1827_182722


namespace octopus_family_total_l1827_182725

/-- Represents the number of children of each color in the octopus family -/
structure OctopusFamily :=
  (white : ℕ)
  (blue : ℕ)
  (striped : ℕ)

/-- The conditions of the octopus family problem -/
def octopusFamilyConditions (initial final : OctopusFamily) : Prop :=
  -- Initially, there were equal numbers of white, blue, and striped octopus children
  initial.white = initial.blue ∧ initial.blue = initial.striped
  -- Some blue octopus children became striped
  ∧ final.blue < initial.blue
  ∧ final.striped > initial.striped
  -- After the change, the total number of blue and white octopus children was 10
  ∧ final.blue + final.white = 10
  -- After the change, the total number of white and striped octopus children was 18
  ∧ final.white + final.striped = 18
  -- The total number of children remains constant
  ∧ initial.white + initial.blue + initial.striped = final.white + final.blue + final.striped

/-- The theorem stating that under the given conditions, the total number of children is 21 -/
theorem octopus_family_total (initial final : OctopusFamily) :
  octopusFamilyConditions initial final →
  final.white + final.blue + final.striped = 21 :=
by
  sorry


end octopus_family_total_l1827_182725


namespace abs_x_y_sum_l1827_182764

theorem abs_x_y_sum (x y : ℝ) : 
  (|x| = 7 ∧ |y| = 9 ∧ |x + y| = -(x + y)) → (x - y = 16 ∨ x - y = -16) := by
  sorry

end abs_x_y_sum_l1827_182764


namespace second_year_sample_size_l1827_182718

/-- Represents the distribution of students across four years -/
structure StudentDistribution where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the total number of students -/
def total_students (d : StudentDistribution) : ℕ :=
  d.first + d.second + d.third + d.fourth

/-- Calculates the number of students to sample from a specific year -/
def sample_size_for_year (total_population : ℕ) (year_population : ℕ) (sample_size : ℕ) : ℕ :=
  (year_population * sample_size) / total_population

theorem second_year_sample_size 
  (total_population : ℕ) 
  (distribution : StudentDistribution) 
  (sample_size : ℕ) :
  total_population = 5000 →
  distribution = { first := 5, second := 4, third := 3, fourth := 1 } →
  sample_size = 260 →
  sample_size_for_year total_population distribution.second sample_size = 80 := by
  sorry

#check second_year_sample_size

end second_year_sample_size_l1827_182718


namespace inequality_proof_l1827_182762

theorem inequality_proof (y : ℝ) (h : y > 0) :
  2 * y ≥ 3 - 1 / y^2 ∧ (2 * y = 3 - 1 / y^2 ↔ y = 1) := by
  sorry

end inequality_proof_l1827_182762


namespace max_G_ratio_is_six_fifths_l1827_182703

/-- Represents a four-digit number --/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : thousands ≥ 1 ∧ thousands ≤ 9 ∧ 
             hundreds ≥ 0 ∧ hundreds ≤ 9 ∧ 
             tens ≥ 0 ∧ tens ≤ 9 ∧ 
             units ≥ 0 ∧ units ≤ 9

/-- Defines a "difference 2 multiple" --/
def isDifference2Multiple (n : FourDigitNumber) : Prop :=
  n.thousands - n.hundreds = 2 ∧ n.tens - n.units = 4

/-- Defines a "difference 3 multiple" --/
def isDifference3Multiple (n : FourDigitNumber) : Prop :=
  n.thousands - n.hundreds = 3 ∧ n.tens - n.units = 6

/-- Calculates the sum of digits --/
def G (n : FourDigitNumber) : Nat :=
  n.thousands + n.hundreds + n.tens + n.units

/-- Calculates F(p,q) --/
def F (p q : FourDigitNumber) : Int :=
  (1000 * p.thousands + 100 * p.hundreds + 10 * p.tens + p.units -
   (1000 * q.thousands + 100 * q.hundreds + 10 * q.tens + q.units)) / 10

/-- Main theorem --/
theorem max_G_ratio_is_six_fifths 
  (p q : FourDigitNumber) 
  (h1 : isDifference2Multiple p)
  (h2 : isDifference3Multiple q)
  (h3 : p.units = 3)
  (h4 : q.units = 3)
  (h5 : ∃ k : Int, F p q / (G p - G q + 3) = k) :
  ∀ (p' q' : FourDigitNumber), 
    isDifference2Multiple p' → 
    isDifference3Multiple q' → 
    p'.units = 3 → 
    q'.units = 3 → 
    (∃ k : Int, F p' q' / (G p' - G q' + 3) = k) → 
    (G p : ℚ) / (G q) ≥ (G p' : ℚ) / (G q') := by
  sorry

end max_G_ratio_is_six_fifths_l1827_182703


namespace triangle_third_side_length_l1827_182757

theorem triangle_third_side_length 
  (a b : ℝ) 
  (θ : Real) 
  (ha : a = 5) 
  (hb : b = 12) 
  (hθ : θ = 150 * π / 180) : 
  ∃ c : ℝ, c = Real.sqrt (169 + 60 * Real.sqrt 3) ∧ 
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) :=
sorry

end triangle_third_side_length_l1827_182757


namespace expression_simplification_l1827_182799

theorem expression_simplification (x : ℝ) (h : x = 3) : 
  (x - 1 + (2 - 2*x) / (x + 1)) / ((x^2 - x) / (x + 1)) = 2/3 := by
  sorry

end expression_simplification_l1827_182799


namespace problem_statement_l1827_182749

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem problem_statement (x x₁ x₂ : ℝ) :
  (∀ x > 0, f x ≥ 1) ∧
  (∀ x > 1, f x < g x) ∧
  (x₁ > x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ g x₁ = g x₂ → x₁ * x₂ < 1) :=
by sorry

end problem_statement_l1827_182749


namespace calculate_expression_solve_system_of_equations_l1827_182728

-- Part 1
theorem calculate_expression : 
  (6 - 2 * Real.sqrt 3) * Real.sqrt 3 - Real.sqrt ((2 - Real.sqrt 2)^2) + 1 / Real.sqrt 2 = 
  6 * Real.sqrt 3 - 8 + 3 * Real.sqrt 2 / 2 := by sorry

-- Part 2
theorem solve_system_of_equations (x y : ℝ) :
  5 * x - y = -9 ∧ 3 * x + y = 1 → x = -1 ∧ y = 4 := by sorry

end calculate_expression_solve_system_of_equations_l1827_182728


namespace f_strictly_increasing_on_interval_l1827_182779

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

-- State the theorem
theorem f_strictly_increasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x < f y := by
  sorry

end f_strictly_increasing_on_interval_l1827_182779


namespace uphill_divisible_by_nine_count_l1827_182767

/-- An uphill integer is a positive integer where each digit is strictly greater than the previous digit. -/
def UphillInteger (n : ℕ) : Prop := sorry

/-- Check if a natural number ends with 6 -/
def EndsWithSix (n : ℕ) : Prop := sorry

/-- Count the number of uphill integers ending in 6 that are divisible by 9 -/
def CountUphillDivisibleBySix : ℕ := sorry

theorem uphill_divisible_by_nine_count : CountUphillDivisibleBySix = 2 := by sorry

end uphill_divisible_by_nine_count_l1827_182767


namespace unique_solution_3x_4y_5z_l1827_182792

theorem unique_solution_3x_4y_5z : 
  ∀ x y z : ℕ+, 
    (3 : ℕ)^(x : ℕ) + (4 : ℕ)^(y : ℕ) = (5 : ℕ)^(z : ℕ) → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry

#check unique_solution_3x_4y_5z

end unique_solution_3x_4y_5z_l1827_182792


namespace leap_year_hours_l1827_182759

/-- The number of days in a leap year -/
def days_in_leap_year : ℕ := 366

/-- The number of hours in a day -/
def hours_in_day : ℕ := 24

/-- The number of hours in a leap year -/
def hours_in_leap_year : ℕ := days_in_leap_year * hours_in_day

theorem leap_year_hours :
  hours_in_leap_year = 8784 :=
by sorry

end leap_year_hours_l1827_182759


namespace football_player_goal_increase_l1827_182773

/-- The increase in average goals score after the fifth match -/
def goalScoreIncrease (totalGoals : ℕ) (fifthMatchGoals : ℕ) : ℚ :=
  let firstFourAverage := (totalGoals - fifthMatchGoals : ℚ) / 4
  let newAverage := (totalGoals : ℚ) / 5
  newAverage - firstFourAverage

/-- Theorem stating the increase in average goals score -/
theorem football_player_goal_increase :
  goalScoreIncrease 4 2 = 3/10 := by
  sorry

end football_player_goal_increase_l1827_182773


namespace line_intercepts_sum_l1827_182704

/-- Given a line with equation y = -1/2 + 3x, prove that the sum of its x-intercept and y-intercept is -1/3. -/
theorem line_intercepts_sum (x y : ℝ) : 
  y = -1/2 + 3*x → -- Line equation
  ∃ (x_int y_int : ℝ),
    (0 = -1/2 + 3*x_int) ∧  -- x-intercept
    (y_int = -1/2 + 3*0) ∧  -- y-intercept
    (x_int + y_int = -1/3) := by
  sorry


end line_intercepts_sum_l1827_182704


namespace second_store_cars_l1827_182795

-- Define the number of stores
def num_stores : ℕ := 5

-- Define the car counts for known stores
def first_store : ℕ := 30
def third_store : ℕ := 14
def fourth_store : ℕ := 21
def fifth_store : ℕ := 25

-- Define the mean
def mean : ℚ := 20.8

-- Define the theorem
theorem second_store_cars :
  ∃ (second_store : ℕ),
    (first_store + second_store + third_store + fourth_store + fifth_store) / num_stores = mean ∧
    second_store = 14 := by
  sorry

end second_store_cars_l1827_182795


namespace seven_hash_three_l1827_182781

/-- Custom operator # defined for real numbers -/
def hash (a b : ℝ) : ℝ := 4*a + 2*b - 6

/-- Theorem stating that 7 # 3 = 28 -/
theorem seven_hash_three : hash 7 3 = 28 := by
  sorry

end seven_hash_three_l1827_182781


namespace trigonometric_identities_l1827_182719

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) :
  (sin α - 4 * cos α) / (5 * sin α + 2 * cos α) = -1/6 ∧
  4 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = 1 := by
sorry

end trigonometric_identities_l1827_182719


namespace environmental_legislation_support_l1827_182789

theorem environmental_legislation_support (men : ℕ) (women : ℕ) 
  (men_support_rate : ℚ) (women_support_rate : ℚ) :
  men = 200 →
  women = 1200 →
  men_support_rate = 70 / 100 →
  women_support_rate = 75 / 100 →
  let total_surveyed := men + women
  let total_supporters := men * men_support_rate + women * women_support_rate
  let overall_support_rate := total_supporters / total_surveyed
  ‖overall_support_rate - 74 / 100‖ < 1 / 100 :=
by sorry

end environmental_legislation_support_l1827_182789


namespace quadratic_to_linear_inequality_l1827_182716

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := x^2 + a*x + b

-- Define the linear function
def g (a b : ℝ) (x : ℝ) := a*x + b

-- State the theorem
theorem quadratic_to_linear_inequality 
  (a b : ℝ) 
  (h : ∀ x : ℝ, f a b x > 0 ↔ x < -3 ∨ x > 1) :
  ∀ x : ℝ, g a b x < 0 ↔ x < 3/2 :=
sorry

end quadratic_to_linear_inequality_l1827_182716


namespace epsilon_max_ratio_l1827_182701

/-- Represents a contestant's performance in a math contest --/
structure ContestPerformance where
  day1_score : ℕ
  day1_attempted : ℕ
  day2_score : ℕ
  day2_attempted : ℕ
  day3_score : ℕ
  day3_attempted : ℕ

/-- Calculates the total score for a contestant --/
def totalScore (p : ContestPerformance) : ℕ := p.day1_score + p.day2_score + p.day3_score

/-- Calculates the total attempted points for a contestant --/
def totalAttempted (p : ContestPerformance) : ℕ := 
  p.day1_attempted + p.day2_attempted + p.day3_attempted

/-- Calculates the success ratio for a contestant --/
def successRatio (p : ContestPerformance) : ℚ := 
  (totalScore p : ℚ) / (totalAttempted p : ℚ)

/-- Delta's performance in the contest --/
def delta : ContestPerformance := {
  day1_score := 210,
  day1_attempted := 350,
  day2_score := 320, -- Assumed based on total score
  day2_attempted := 450, -- Assumed based on total attempted
  day3_score := 0, -- Placeholder
  day3_attempted := 0 -- Placeholder
}

theorem epsilon_max_ratio :
  ∀ epsilon : ContestPerformance,
  totalAttempted epsilon = 800 →
  totalAttempted delta = 800 →
  successRatio delta = 530 / 800 →
  epsilon.day1_attempted ≠ 350 →
  epsilon.day1_score > 0 →
  epsilon.day2_score > 0 →
  epsilon.day3_score > 0 →
  (epsilon.day1_score : ℚ) / (epsilon.day1_attempted : ℚ) < 210 / 350 →
  (epsilon.day2_score : ℚ) / (epsilon.day2_attempted : ℚ) < (delta.day2_score : ℚ) / (delta.day2_attempted : ℚ) →
  (epsilon.day3_score : ℚ) / (epsilon.day3_attempted : ℚ) < (delta.day3_score : ℚ) / (delta.day3_attempted : ℚ) →
  successRatio epsilon ≤ 789 / 800 :=
by sorry

end epsilon_max_ratio_l1827_182701


namespace fraction_simplification_l1827_182702

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 4) (h2 : a ≠ -4) :
  (2 * a) / (a^2 - 16) - 1 / (a - 4) = 1 / (a + 4) := by
  sorry

end fraction_simplification_l1827_182702


namespace circle_properties_l1827_182758

-- Define the circle P
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def intersects_x_axis (P : Circle) : Prop :=
  P.radius^2 - P.center.2^2 = 2

def intersects_y_axis (P : Circle) : Prop :=
  P.radius^2 - P.center.1^2 = 3

def distance_to_y_eq_x (P : Circle) : Prop :=
  |P.center.2 - P.center.1| = 1

-- Define the theorem
theorem circle_properties (P : Circle) 
  (hx : intersects_x_axis P) 
  (hy : intersects_y_axis P) 
  (hd : distance_to_y_eq_x P) : 
  (∃ a b : ℝ, P.center = (a, b) ∧ b^2 - a^2 = 1) ∧ 
  ((P.center = (0, 1) ∧ P.radius = Real.sqrt 3) ∨ 
   (P.center = (0, -1) ∧ P.radius = Real.sqrt 3)) :=
by sorry

end circle_properties_l1827_182758


namespace cos_180_degrees_l1827_182729

/-- Cosine of 180 degrees is -1 -/
theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end cos_180_degrees_l1827_182729


namespace divisibility_counterexample_l1827_182706

theorem divisibility_counterexample : 
  ∃ (a b c : ℤ), (a ∣ b * c) ∧ ¬(a ∣ b) ∧ ¬(a ∣ c) := by
  sorry

end divisibility_counterexample_l1827_182706


namespace dot_product_FA_AB_is_zero_l1827_182748

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the focus F
def focus : ℝ × ℝ := (4, 0)

-- Define point A on y-axis with |OA| = |OF|
def point_A : ℝ × ℝ := (0, 4)  -- We choose the positive y-coordinate

-- Define point B as intersection of directrix and x-axis
def point_B : ℝ × ℝ := (-4, 0)

-- Define vector FA
def vector_FA : ℝ × ℝ := (point_A.1 - focus.1, point_A.2 - focus.2)

-- Define vector AB
def vector_AB : ℝ × ℝ := (point_B.1 - point_A.1, point_B.2 - point_A.2)

-- Theorem statement
theorem dot_product_FA_AB_is_zero :
  vector_FA.1 * vector_AB.1 + vector_FA.2 * vector_AB.2 = 0 :=
sorry

end dot_product_FA_AB_is_zero_l1827_182748


namespace february_bill_increase_l1827_182726

def january_bill : ℝ := 179.99999999999991

theorem february_bill_increase (february_bill : ℝ) 
  (h1 : february_bill / january_bill = 3 / 2) 
  (h2 : ∃ (increased_bill : ℝ), increased_bill / january_bill = 5 / 3) : 
  ∃ (increased_bill : ℝ), increased_bill - february_bill = 30 :=
sorry

end february_bill_increase_l1827_182726


namespace simplify_expression_l1827_182712

theorem simplify_expression : (((81 : ℚ) / 16) ^ (3 / 4) - (-1) ^ (0 : ℕ)) = 19 / 8 := by
  sorry

end simplify_expression_l1827_182712


namespace geometric_sequence_implies_c_equals_six_l1827_182741

/-- A function f(x) = x^2 + x + c where f(1), f(2), and f(3) form a geometric sequence. -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + x + c

/-- The theorem stating that if f(1), f(2), and f(3) form a geometric sequence, then c = 6. -/
theorem geometric_sequence_implies_c_equals_six (c : ℝ) :
  (∃ r : ℝ, f c 2 = f c 1 * r ∧ f c 3 = f c 2 * r) → c = 6 := by
  sorry

end geometric_sequence_implies_c_equals_six_l1827_182741


namespace sets_equality_and_inclusion_l1827_182742

def A : Set ℝ := {x | x^2 - 4*x - 12 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x - 6*m^2 ≤ 0}

theorem sets_equality_and_inclusion (m : ℝ) (h : m > 0) :
  A = {x : ℝ | -2 ≤ x ∧ x ≤ 6} ∧
  B m = {x : ℝ | -2*m ≤ x ∧ x ≤ 3*m} ∧
  (A ⊆ B m ↔ m ≥ 2) ∧
  (B m ⊆ A ↔ 0 < m ∧ m ≤ 1) :=
by sorry

end sets_equality_and_inclusion_l1827_182742


namespace curve_translation_l1827_182711

-- Define the original curve
def original_curve (x y : ℝ) : Prop :=
  y * Real.sin x - 2 * y + 3 = 0

-- Define the translated curve
def translated_curve (x y : ℝ) : Prop :=
  (1 + y) * Real.cos x - 2 * y + 1 = 0

-- Theorem statement
theorem curve_translation :
  ∀ (x y : ℝ),
  original_curve (x + π/2) (y + 1) ↔ translated_curve x y :=
by sorry

end curve_translation_l1827_182711


namespace odd_numbers_product_equality_l1827_182754

theorem odd_numbers_product_equality (a b c d : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  a < b → b < c → c < d →
  a * d = b * c →
  ∃ k l : ℕ, a + d = 2^k ∧ b + c = 2^l →
  a = 1 := by sorry

end odd_numbers_product_equality_l1827_182754


namespace hash_composition_l1827_182770

-- Define the # operation
def hash (x : ℝ) : ℝ := 8 - x

-- Define the # operation
def hash_prefix (x : ℝ) : ℝ := x - 8

-- Theorem statement
theorem hash_composition : hash_prefix (hash 14) = -14 := by
  sorry

end hash_composition_l1827_182770


namespace system_inequalities_solution_l1827_182738

theorem system_inequalities_solution : 
  {x : ℕ | 4 * (x + 1) ≤ 7 * x + 10 ∧ x - 5 < (x - 8) / 3} = {0, 1, 2, 3} := by
  sorry

end system_inequalities_solution_l1827_182738


namespace log_1560_base_5_rounded_l1827_182731

theorem log_1560_base_5_rounded (ε : ℝ) (h : ε > 0) :
  ∃ (n : ℤ), n = 5 ∧ |Real.log 1560 / Real.log 5 - n| < 1/2 + ε :=
sorry

end log_1560_base_5_rounded_l1827_182731


namespace logarithm_problem_l1827_182774

noncomputable def a : ℝ := Real.log 55 / Real.log 50
noncomputable def b : ℝ := Real.log 20 / Real.log 55

theorem logarithm_problem (a b : ℝ) (h1 : a = Real.log 55 / Real.log 50) (h2 : b = Real.log 20 / Real.log 55) :
  Real.log (2662 * Real.sqrt 10) / Real.log 250 = (18 * a + 11 * a * b - 13) / (10 - 2 * a * b) := by
  sorry

end logarithm_problem_l1827_182774


namespace simplify_square_roots_l1827_182744

theorem simplify_square_roots : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end simplify_square_roots_l1827_182744


namespace julian_facebook_friends_l1827_182794

theorem julian_facebook_friends :
  ∀ (julian_friends : ℕ) (julian_boys julian_girls boyd_boys boyd_girls : ℝ),
    julian_boys = 0.6 * julian_friends →
    julian_girls = 0.4 * julian_friends →
    boyd_girls = 2 * julian_girls →
    boyd_boys + boyd_girls = 100 →
    boyd_boys = 0.36 * 100 →
    julian_friends = 80 :=
by
  sorry

end julian_facebook_friends_l1827_182794


namespace units_digit_27_45_l1827_182736

theorem units_digit_27_45 : (27 ^ 45) % 10 = 7 := by
  sorry

end units_digit_27_45_l1827_182736


namespace sqrt_fraction_equality_l1827_182776

theorem sqrt_fraction_equality : 
  (Real.sqrt ((8:ℝ)^2 + 15^2)) / (Real.sqrt (36 + 64)) = 17 / 10 := by
  sorry

end sqrt_fraction_equality_l1827_182776


namespace line_slope_and_intercept_l1827_182705

/-- Given a line with equation 4y = 6x - 12, prove that its slope is 3/2 and y-intercept is -3. -/
theorem line_slope_and_intercept :
  ∃ (m b : ℚ), m = 3/2 ∧ b = -3 ∧
  ∀ (x y : ℚ), 4*y = 6*x - 12 ↔ y = m*x + b :=
sorry

end line_slope_and_intercept_l1827_182705
