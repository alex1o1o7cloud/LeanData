import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l938_93879

theorem equation_solution : ∃ x : ℝ, 61 + 5 * 12 / (180 / x) = 62 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l938_93879


namespace NUMINAMATH_CALUDE_least_integer_with_conditions_l938_93858

/-- The number of 8's in the solution -/
def num_eights : ℕ := 93

/-- The number of 9's in the solution -/
def num_nines : ℕ := 140

/-- The sum of digits in the solution -/
def digit_sum : ℕ := 2011

/-- Constructs the integer from the given number of 8's and 9's -/
def construct_number (n_eights n_nines : ℕ) : ℕ := sorry

/-- Checks if a number is a power of 6 -/
def is_power_of_six (n : ℕ) : Prop := sorry

/-- Calculates the sum of digits of a number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Calculates the product of digits of a number -/
def product_of_digits (n : ℕ) : ℕ := sorry

theorem least_integer_with_conditions :
  let n := construct_number num_eights num_nines
  ∀ m : ℕ, m < n →
    (sum_of_digits m = digit_sum ∧ is_power_of_six (product_of_digits m)) →
    False :=
by sorry

end NUMINAMATH_CALUDE_least_integer_with_conditions_l938_93858


namespace NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_r_l938_93892

-- Define the propositions
variable (p q r : Prop)

-- Define what it means for a condition to be sufficient but not necessary
def sufficient_but_not_necessary (a b : Prop) : Prop :=
  (a → b) ∧ ¬(b → a)

-- Define what it means for a condition to be necessary but not sufficient
def necessary_but_not_sufficient (a b : Prop) : Prop :=
  (b → a) ∧ ¬(a → b)

-- State the theorem
theorem p_sufficient_but_not_necessary_for_r
  (h1 : sufficient_but_not_necessary p q)
  (h2 : necessary_but_not_sufficient r q) :
  sufficient_but_not_necessary p r :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_r_l938_93892


namespace NUMINAMATH_CALUDE_inequality_solution_set_l938_93895

-- Define the inequality function
def f (x : ℝ) : ℝ := -x^2 - x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | -2 < x ∧ x < 1}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x > 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l938_93895


namespace NUMINAMATH_CALUDE_trig_simplification_l938_93847

theorem trig_simplification :
  1 / Real.sin (70 * π / 180) - Real.sqrt 3 / Real.cos (70 * π / 180) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l938_93847


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l938_93864

/-- The function f(x) = (1/2)^x + m does not pass through the first quadrant if and only if m ≤ -1 -/
theorem function_not_in_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, x > 0 → (1/2)^x + m ≤ 0) ↔ m ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l938_93864


namespace NUMINAMATH_CALUDE_acid_concentration_proof_l938_93827

/-- Represents the composition of an acid-water mixture -/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- The original mixture before any additions -/
def original_mixture : Mixture :=
  { acid := 0,  -- We don't know the initial acid amount
    water := 0 } -- We don't know the initial water amount

theorem acid_concentration_proof :
  -- The total volume of the original mixture is 10 ounces
  original_mixture.acid + original_mixture.water = 10 →
  -- After adding 1 ounce of water, the acid concentration becomes 25%
  original_mixture.acid / (original_mixture.acid + original_mixture.water + 1) = 1/4 →
  -- After adding 1 ounce of acid to the water-added mixture, the concentration becomes 40%
  (original_mixture.acid + 1) / (original_mixture.acid + original_mixture.water + 2) = 2/5 →
  -- Then the original acid concentration was 27.5%
  original_mixture.acid / (original_mixture.acid + original_mixture.water) = 11/40 :=
by sorry

end NUMINAMATH_CALUDE_acid_concentration_proof_l938_93827


namespace NUMINAMATH_CALUDE_seminar_attendees_l938_93888

theorem seminar_attendees (total : ℕ) (a : ℕ) (h1 : total = 185) (h2 : a = 30) : 
  total - (a + 2*a + (a + 10) + ((a + 10) - 5)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_seminar_attendees_l938_93888


namespace NUMINAMATH_CALUDE_janice_pebbles_l938_93862

/-- The number of pebbles each friend received -/
def pebbles_per_friend : ℕ := 4

/-- The number of friends who received pebbles -/
def number_of_friends : ℕ := 9

/-- The total number of pebbles Janice gave away -/
def total_pebbles : ℕ := pebbles_per_friend * number_of_friends

theorem janice_pebbles : total_pebbles = 36 := by
  sorry

end NUMINAMATH_CALUDE_janice_pebbles_l938_93862


namespace NUMINAMATH_CALUDE_systems_solution_l938_93870

theorem systems_solution : ∃ (x y : ℝ), 
  (x - y = 1 ∧ 3*x + y = 11) ∧ 
  (3*x - 2*y = 5 ∧ 2*x + 3*y = 12) ∧
  (x = 3 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_systems_solution_l938_93870


namespace NUMINAMATH_CALUDE_proportional_from_equality_l938_93865

/-- Two real numbers are directly proportional if their ratio is constant -/
def DirectlyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x = k * y

/-- Given x/3 = y/4, prove that x and y are directly proportional -/
theorem proportional_from_equality (x y : ℝ) (h : x / 3 = y / 4) :
  DirectlyProportional x y := by
  sorry

end NUMINAMATH_CALUDE_proportional_from_equality_l938_93865


namespace NUMINAMATH_CALUDE_unique_digit_solution_l938_93885

theorem unique_digit_solution :
  ∃! (a b c d e f g h i j : ℕ),
    (a ∈ Finset.range 10) ∧
    (b ∈ Finset.range 10) ∧
    (c ∈ Finset.range 10) ∧
    (d ∈ Finset.range 10) ∧
    (e ∈ Finset.range 10) ∧
    (f ∈ Finset.range 10) ∧
    (g ∈ Finset.range 10) ∧
    (h ∈ Finset.range 10) ∧
    (i ∈ Finset.range 10) ∧
    (j ∈ Finset.range 10) ∧
    ({a, b, c, d, e, f, g, h, i, j} : Finset ℕ).card = 10 ∧
    20 * (a - 8) = 20 ∧
    b / 2 + 17 = 20 ∧
    c * d - 4 = 20 ∧
    (e + 8) / 12 = f ∧
    4 * g + h = 20 ∧
    20 * (i - j) = 100 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l938_93885


namespace NUMINAMATH_CALUDE_base_five_last_digit_l938_93807

theorem base_five_last_digit (n : ℕ) (h : n = 89) : n % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_five_last_digit_l938_93807


namespace NUMINAMATH_CALUDE_correct_ranking_count_l938_93822

/-- Represents a team in the tournament -/
inductive Team : Type
| E : Team
| F : Team
| G : Team
| H : Team

/-- Represents the outcome of a match -/
inductive MatchOutcome : Type
| Win : Team → MatchOutcome
| Draw : MatchOutcome

/-- Represents the final ranking of teams -/
def Ranking := List Team

/-- The structure of the tournament -/
structure Tournament :=
  (saturdayMatch1 : MatchOutcome)
  (saturdayMatch2 : MatchOutcome)
  (sundayMatch1Winner : Team)
  (sundayMatch2Winner : Team)

/-- Function to calculate the number of possible rankings -/
def countPossibleRankings : ℕ :=
  -- Implementation details omitted
  256

/-- Theorem stating that the number of possible rankings is 256 -/
theorem correct_ranking_count :
  countPossibleRankings = 256 := by sorry


end NUMINAMATH_CALUDE_correct_ranking_count_l938_93822


namespace NUMINAMATH_CALUDE_student_number_problem_l938_93810

theorem student_number_problem (x : ℤ) : x = 48 ↔ 5 * x - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l938_93810


namespace NUMINAMATH_CALUDE_cream_fraction_after_mixing_l938_93855

/-- Represents the contents of a cup -/
structure CupContents where
  coffee : ℚ
  cream : ℚ

/-- Represents the mixing process -/
def mix_and_transfer (cup1 cup2 : CupContents) : (CupContents × CupContents) :=
  sorry

theorem cream_fraction_after_mixing :
  let initial_cup1 : CupContents := { coffee := 4, cream := 0 }
  let initial_cup2 : CupContents := { coffee := 0, cream := 4 }
  let (final_cup1, _) := mix_and_transfer initial_cup1 initial_cup2
  (final_cup1.cream / (final_cup1.coffee + final_cup1.cream)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_cream_fraction_after_mixing_l938_93855


namespace NUMINAMATH_CALUDE_average_difference_l938_93830

-- Define the number of students and teachers
def num_students : ℕ := 120
def num_teachers : ℕ := 6

-- Define the class enrollments
def class_enrollments : List ℕ := [40, 40, 20, 10, 5, 5]

-- Define t (average number of students per teacher)
def t : ℚ := (num_students : ℚ) / num_teachers

-- Define s (average number of students per student)
def s : ℚ := (List.sum (List.map (fun x => x * x) class_enrollments) : ℚ) / num_students

-- Theorem to prove
theorem average_difference : t - s = -11.25 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l938_93830


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l938_93805

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  (a^3 + b^3 + c^3 = 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l938_93805


namespace NUMINAMATH_CALUDE_x_range_l938_93869

theorem x_range (x : Real) 
  (h1 : -Real.pi/2 ≤ x ∧ x ≤ 3*Real.pi/2) 
  (h2 : Real.sqrt (1 + Real.sin (2*x)) = Real.sin x + Real.cos x) : 
  -Real.pi/4 ≤ x ∧ x ≤ 3*Real.pi/4 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l938_93869


namespace NUMINAMATH_CALUDE_two_intersections_l938_93836

/-- A line in a plane represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are identical -/
def identical (l1 l2 : Line) : Prop :=
  parallel l1 l2 ∧ l1.a * l2.c = l1.c * l2.a

/-- Check if two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  ¬(parallel l1 l2) ∨ identical l1 l2

/-- The number of distinct intersection points of at least two lines -/
def num_intersections (lines : List Line) : ℕ :=
  sorry

/-- The three lines from the problem -/
def line1 : Line := ⟨3, 2, 4⟩
def line2 : Line := ⟨-1, 3, 3⟩
def line3 : Line := ⟨6, -4, 8⟩

/-- The main theorem -/
theorem two_intersections :
  num_intersections [line1, line2, line3] = 2 := by sorry

end NUMINAMATH_CALUDE_two_intersections_l938_93836


namespace NUMINAMATH_CALUDE_intersection_M_N_l938_93881

def M : Set ℝ := {x | (x - 1)^2 < 4}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l938_93881


namespace NUMINAMATH_CALUDE_pie_weight_theorem_l938_93882

theorem pie_weight_theorem (total_weight : ℝ) (fridge_weight : ℝ) : 
  (5 / 6 : ℝ) * total_weight = fridge_weight → 
  (1 / 6 : ℝ) * total_weight = 240 :=
by
  sorry

#check pie_weight_theorem 1440 1200

end NUMINAMATH_CALUDE_pie_weight_theorem_l938_93882


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l938_93832

/-- Given vectors a and b that are parallel, prove that their sum has magnitude √5 -/
theorem parallel_vectors_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, -4]
  (∃ (k : ℝ), a = k • b) → 
  ‖a + b‖ = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l938_93832


namespace NUMINAMATH_CALUDE_remaining_amount_l938_93828

def initial_amount : ℚ := 343
def fraction_given : ℚ := 1/7
def num_recipients : ℕ := 2

theorem remaining_amount :
  initial_amount - (fraction_given * initial_amount * num_recipients) = 245 :=
by sorry

end NUMINAMATH_CALUDE_remaining_amount_l938_93828


namespace NUMINAMATH_CALUDE_range_of_a_l938_93852

theorem range_of_a (a : ℝ) : 
  a < 9 * a^3 - 11 * a ∧ 9 * a^3 - 11 * a < |a| → 
  a ∈ Set.Ioo (-2 * Real.sqrt 3 / 3) (-Real.sqrt 10 / 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l938_93852


namespace NUMINAMATH_CALUDE_son_age_l938_93841

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end NUMINAMATH_CALUDE_son_age_l938_93841


namespace NUMINAMATH_CALUDE_geometric_sequence_and_log_function_l938_93857

/-- Given real numbers a, b, c, and d forming a geometric sequence,
    and for the function y = ln x - x, when x = b reaches its maximum value at c,
    then ad = -1 -/
theorem geometric_sequence_and_log_function
  (a b c d : ℝ)
  (h_geometric : b / a = c / b ∧ c / b = d / c)
  (h_max : b > 0 ∧ c > 0 ∧ (∀ x > 0, Real.log x - x ≤ Real.log c - c) ∧ Real.log b - b = Real.log c - c) :
  a * d = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_log_function_l938_93857


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l938_93877

theorem hot_dogs_remainder :
  25197625 % 4 = 1 := by sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l938_93877


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_sum_l938_93837

theorem sqrt_plus_square_zero_implies_sum (x y : ℝ) :
  Real.sqrt (x - 1) + (y + 2)^2 = 0 → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_sum_l938_93837


namespace NUMINAMATH_CALUDE_flax_acres_for_given_farm_l938_93861

/-- Represents a farm with sunflowers and flax -/
structure Farm where
  total_acres : ℕ
  sunflower_excess : ℕ

/-- Calculates the number of acres of flax to be planted -/
def flax_acres (f : Farm) : ℕ :=
  (f.total_acres - f.sunflower_excess) / 2

theorem flax_acres_for_given_farm :
  let farm : Farm := { total_acres := 240, sunflower_excess := 80 }
  flax_acres farm = 80 := by
  sorry

#eval flax_acres { total_acres := 240, sunflower_excess := 80 }

end NUMINAMATH_CALUDE_flax_acres_for_given_farm_l938_93861


namespace NUMINAMATH_CALUDE_james_delivery_l938_93829

/-- Calculates the number of bags delivered by James in a given number of days -/
def bags_delivered (bags_per_trip : ℕ) (trips_per_day : ℕ) (days : ℕ) : ℕ :=
  bags_per_trip * trips_per_day * days

/-- Theorem stating that James delivers 1000 bags in 5 days -/
theorem james_delivery : bags_delivered 10 20 5 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_james_delivery_l938_93829


namespace NUMINAMATH_CALUDE_expression_value_for_2016_l938_93874

theorem expression_value_for_2016 :
  let x : ℤ := 2016
  (x^2 - x) - (x^2 - 2*x + 1) = 2015 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_for_2016_l938_93874


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l938_93859

theorem partial_fraction_decomposition :
  ∃ (A B C : ℝ), A = 16 ∧ B = 4 ∧ C = -16 ∧
  ∀ (x : ℝ), x ≠ 2 → x ≠ 4 →
    8 * x^2 / ((x - 4) * (x - 2)^3) = A / (x - 4) + B / (x - 2) + C / (x - 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l938_93859


namespace NUMINAMATH_CALUDE_possible_no_snorers_in_sample_l938_93821

-- Define the types for our problem
def Person : Type := Unit
def HasHeartDisease (p : Person) : Prop := sorry
def Snores (p : Person) : Prop := sorry

-- Define correlation and confidence
def Correlation (A B : Person → Prop) : Prop := sorry
def ConfidenceLevel : ℝ := sorry

-- State the theorem
theorem possible_no_snorers_in_sample 
  (corr : Correlation HasHeartDisease Snores)
  (conf : ConfidenceLevel > 0.99)
  : ∃ (sample : Finset Person), 
    (Finset.card sample = 100) ∧ 
    (∀ p ∈ sample, HasHeartDisease p) ∧
    (∀ p ∈ sample, ¬Snores p) :=
sorry

end NUMINAMATH_CALUDE_possible_no_snorers_in_sample_l938_93821


namespace NUMINAMATH_CALUDE_fruit_draw_ways_l938_93845

/-- The number of fruits in the basket -/
def num_fruits : ℕ := 5

/-- The number of draws -/
def num_draws : ℕ := 2

/-- The number of ways to draw a fruit twice from a basket of 5 distinct fruits, considering the order -/
def num_ways : ℕ := num_fruits * (num_fruits - 1)

theorem fruit_draw_ways :
  num_ways = 20 :=
by sorry

end NUMINAMATH_CALUDE_fruit_draw_ways_l938_93845


namespace NUMINAMATH_CALUDE_percentage_of_x_minus_y_l938_93804

theorem percentage_of_x_minus_y (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (30 / 100) * (x + y) →
  y = (25 / 100) * x →
  P = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_x_minus_y_l938_93804


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_l938_93887

/-- The volume of a cube with a given space diagonal -/
theorem cube_volume_from_diagonal (d : ℝ) (h : d = 5 * Real.sqrt 3) : 
  let s := d / Real.sqrt 3
  s ^ 3 = 125 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_l938_93887


namespace NUMINAMATH_CALUDE_sin_15_and_tan_75_l938_93808

theorem sin_15_and_tan_75 :
  (Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4) ∧
  (Real.tan (75 * π / 180) = 2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_15_and_tan_75_l938_93808


namespace NUMINAMATH_CALUDE_small_boxes_in_big_box_l938_93883

theorem small_boxes_in_big_box 
  (total_big_boxes : ℕ) 
  (candles_per_small_box : ℕ) 
  (total_candles : ℕ) 
  (h1 : total_big_boxes = 50)
  (h2 : candles_per_small_box = 40)
  (h3 : total_candles = 8000) :
  (total_candles / candles_per_small_box) / total_big_boxes = 4 := by
sorry

end NUMINAMATH_CALUDE_small_boxes_in_big_box_l938_93883


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l938_93889

universe u

def U : Set (Fin 5) := {0, 1, 2, 3, 4}
def A : Set (Fin 5) := {2, 3, 4}
def B : Set (Fin 5) := {0, 1, 4}

theorem complement_A_inter_B :
  (Aᶜ ∩ B : Set (Fin 5)) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l938_93889


namespace NUMINAMATH_CALUDE_problem_solution_l938_93801

theorem problem_solution (x y : ℚ) 
  (sum_eq : x + y = 500)
  (ratio_eq : x / y = 7 / 8) :
  y - x = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l938_93801


namespace NUMINAMATH_CALUDE_complex_addition_l938_93894

theorem complex_addition (z₁ z₂ : ℂ) (h₁ : z₁ = 1 + I) (h₂ : z₂ = 2 + 3*I) :
  z₁ + z₂ = 3 + 4*I := by sorry

end NUMINAMATH_CALUDE_complex_addition_l938_93894


namespace NUMINAMATH_CALUDE_card_distribution_theorem_l938_93868

/-- Represents the state of card distribution among points -/
structure CardState (n : ℕ) where
  cards_at_A : Fin n → ℕ
  cards_at_O : ℕ

/-- Represents a move in the game -/
inductive Move (n : ℕ)
  | outer (i : Fin n) : Move n
  | inner : Move n

/-- Applies a move to a card state -/
def apply_move (n : ℕ) (state : CardState n) (move : Move n) : CardState n :=
  sorry

/-- Checks if a state is valid according to the game rules -/
def is_valid_state (n : ℕ) (state : CardState n) : Prop :=
  sorry

/-- Checks if a state is the goal state (all points have ≥ n+1 cards) -/
def is_goal_state (n : ℕ) (state : CardState n) : Prop :=
  sorry

/-- The main theorem to be proved -/
theorem card_distribution_theorem (n : ℕ) (h_n : n ≥ 3) (T : ℕ) (h_T : T ≥ n^2 + 3*n + 1)
  (initial_state : CardState n) (h_initial : is_valid_state n initial_state) :
  ∃ (moves : List (Move n)), 
    is_goal_state n (moves.foldl (apply_move n) initial_state) :=
  sorry

end NUMINAMATH_CALUDE_card_distribution_theorem_l938_93868


namespace NUMINAMATH_CALUDE_sequence_a_monotonicity_l938_93875

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def sequence_a (x y : V) (n : ℕ) : ℝ := ‖x - n • y‖

theorem sequence_a_monotonicity (x y : V) (hx : x ≠ 0) (hy : y ≠ 0) :
  (∀ n : ℕ, sequence_a V x y n < sequence_a V x y (n + 1)) ↔
  (3 * ‖y‖ > 2 * ‖x‖ * ‖y‖⁻¹ * (inner x y)) ∧
  ¬(∀ n : ℕ, sequence_a V x y (n + 1) < sequence_a V x y n) :=
sorry

end NUMINAMATH_CALUDE_sequence_a_monotonicity_l938_93875


namespace NUMINAMATH_CALUDE_yellow_better_for_fine_gift_l938_93866

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Yellow

/-- Represents the contents of the bag -/
structure Bag :=
  (red : Nat)
  (yellow : Nat)

/-- Calculates the probability of drawing two balls of the same color -/
def probSameColor (b : Bag) : Rat :=
  let total := b.red + b.yellow
  let sameRed := (b.red * (b.red - 1)) / 2
  let sameYellow := (b.yellow * (b.yellow - 1)) / 2
  (sameRed + sameYellow) / ((total * (total - 1)) / 2)

/-- The initial bag configuration -/
def initialBag : Bag := ⟨1, 3⟩

/-- Theorem: Adding a yellow ball gives a higher probability of drawing two balls of the same color -/
theorem yellow_better_for_fine_gift :
  probSameColor ⟨initialBag.red, initialBag.yellow + 1⟩ > 
  probSameColor ⟨initialBag.red + 1, initialBag.yellow⟩ :=
sorry

end NUMINAMATH_CALUDE_yellow_better_for_fine_gift_l938_93866


namespace NUMINAMATH_CALUDE_magenta_opposite_cyan_l938_93811

-- Define the colors
inductive Color
| Yellow
| Orange
| Blue
| Cyan
| Magenta
| Black

-- Define a cube
structure Cube where
  faces : Fin 6 → Color

-- Define the property of opposite faces
def opposite (c : Cube) (f1 f2 : Fin 6) : Prop :=
  (f1.val + f2.val) % 6 = 3

-- Define the given conditions
def cube_conditions (c : Cube) : Prop :=
  ∃ (top front right : Fin 6),
    c.faces top = Color.Cyan ∧
    c.faces right = Color.Blue ∧
    (c.faces front = Color.Yellow ∨ c.faces front = Color.Orange ∨ c.faces front = Color.Black)

-- Theorem statement
theorem magenta_opposite_cyan (c : Cube) :
  cube_conditions c →
  ∃ (magenta_face cyan_face : Fin 6),
    c.faces magenta_face = Color.Magenta ∧
    c.faces cyan_face = Color.Cyan ∧
    opposite c magenta_face cyan_face :=
by sorry

end NUMINAMATH_CALUDE_magenta_opposite_cyan_l938_93811


namespace NUMINAMATH_CALUDE_point_B_coordinate_l938_93856

/-- Given two points A and B on a number line, where A is 3 units to the left of the origin
    and the distance between A and B is 1, prove that the coordinate of B is either -4 or -2. -/
theorem point_B_coordinate (A B : ℝ) : 
  A = -3 → abs (B - A) = 1 → (B = -4 ∨ B = -2) := by sorry

end NUMINAMATH_CALUDE_point_B_coordinate_l938_93856


namespace NUMINAMATH_CALUDE_max_value_sine_function_l938_93806

theorem max_value_sine_function (x : ℝ) (h : x ∈ Set.Icc 0 (π/4)) :
  (∃ (max_y : ℝ), max_y = Real.sqrt 3 ∧
    (∀ y : ℝ, y = Real.sqrt 3 * Real.sin (2*x + π/4) → y ≤ max_y) ∧
    max_y = Real.sqrt 3 * Real.sin (2*(π/8) + π/4)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sine_function_l938_93806


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l938_93854

theorem smallest_five_digit_multiple_of_18 : ∃ (n : ℕ), 
  (n = 10008) ∧ 
  (∃ (k : ℕ), n = 18 * k) ∧ 
  (n ≥ 10000) ∧ 
  (∀ (m : ℕ), (∃ (j : ℕ), m = 18 * j) → m ≥ 10000 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l938_93854


namespace NUMINAMATH_CALUDE_sum_first_5_even_numbers_is_30_l938_93800

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.map (fun i => 2 * i) (List.range n)

theorem sum_first_5_even_numbers_is_30 :
  List.sum (first_n_even_numbers 5) = 30 :=
by
  sorry

#check sum_first_5_even_numbers_is_30

end NUMINAMATH_CALUDE_sum_first_5_even_numbers_is_30_l938_93800


namespace NUMINAMATH_CALUDE_solution_set_part1_value_of_a_part2_l938_93846

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | |x - 2| ≥ 7 - |x - 1|} = {x : ℝ | x ≤ -2 ∨ x ≥ 5} :=
sorry

-- Part 2
theorem value_of_a_part2 (a : ℝ) :
  {x : ℝ | |x - a| ≤ 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} → a = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_value_of_a_part2_l938_93846


namespace NUMINAMATH_CALUDE_sum_reciprocal_squared_plus_one_ge_one_l938_93816

theorem sum_reciprocal_squared_plus_one_ge_one (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squared_plus_one_ge_one_l938_93816


namespace NUMINAMATH_CALUDE_T_not_subset_S_l938_93873

def S : Set ℤ := {x | ∃ n : ℤ, x = 2 * n + 1}
def T : Set ℤ := {y | ∃ k : ℤ, y = 4 * k + 1}

theorem T_not_subset_S : ¬(T ⊆ S) := by
  sorry

end NUMINAMATH_CALUDE_T_not_subset_S_l938_93873


namespace NUMINAMATH_CALUDE_functional_equation_solution_l938_93813

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (k : ℝ) :
  ∀ f : RealFunction, 
    (∀ x y : ℝ, f (f x + f y + k * x * y) = x * f y + y * f x) →
    (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l938_93813


namespace NUMINAMATH_CALUDE_polygon_sides_l938_93825

theorem polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l938_93825


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l938_93824

/-- Calculates the molecular weight of a compound given the number of atoms and atomic weights -/
def molecularWeight (carbonAtoms hydrogenAtoms oxygenAtoms : ℕ) 
  (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  (carbonAtoms : ℝ) * carbonWeight + 
  (hydrogenAtoms : ℝ) * hydrogenWeight + 
  (oxygenAtoms : ℝ) * oxygenWeight

/-- The molecular weight of the given compound is approximately 58.078 g/mol -/
theorem compound_molecular_weight :
  let carbonAtoms : ℕ := 3
  let hydrogenAtoms : ℕ := 6
  let oxygenAtoms : ℕ := 1
  let carbonWeight : ℝ := 12.01
  let hydrogenWeight : ℝ := 1.008
  let oxygenWeight : ℝ := 16.00
  abs (molecularWeight carbonAtoms hydrogenAtoms oxygenAtoms 
    carbonWeight hydrogenWeight oxygenWeight - 58.078) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l938_93824


namespace NUMINAMATH_CALUDE_prime_divides_binomial_l938_93803

theorem prime_divides_binomial (n k : ℕ) (h_prime : Nat.Prime n) (h_k_pos : 0 < k) (h_k_lt_n : k < n) :
  n ∣ Nat.choose n k := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_binomial_l938_93803


namespace NUMINAMATH_CALUDE_camping_problem_l938_93860

theorem camping_problem (p m s : ℝ) 
  (h1 : s + m = p + 20)  -- Peter's indirect route equation
  (h2 : s + p = m + 16)  -- Michael's indirect route equation
  : s = 18 ∧ m = p + 2 := by
  sorry

end NUMINAMATH_CALUDE_camping_problem_l938_93860


namespace NUMINAMATH_CALUDE_bacon_count_l938_93833

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 228

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 23

/-- The difference between students suggesting bacon and tomatoes -/
def bacon_tomato_diff : ℕ := 314

/-- The number of students who suggested bacon -/
def bacon : ℕ := tomatoes + bacon_tomato_diff

theorem bacon_count : bacon = 337 := by
  sorry

end NUMINAMATH_CALUDE_bacon_count_l938_93833


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l938_93812

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l938_93812


namespace NUMINAMATH_CALUDE_power_relations_l938_93834

/-- Given real numbers a, b, c, d satisfying certain conditions, 
    prove statements about their powers. -/
theorem power_relations (a b c d : ℝ) 
    (sum_eq : a + b = c + d) 
    (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
    (a^5 + b^5 = c^5 + d^5) ∧ 
    ¬(∀ (a b c d : ℝ), (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) := by
  sorry


end NUMINAMATH_CALUDE_power_relations_l938_93834


namespace NUMINAMATH_CALUDE_bug_path_tiles_l938_93831

-- Define the garden dimensions
def width : ℕ := 12
def length : ℕ := 18

-- Define the function to calculate the number of tiles visited
def tilesVisited (w l : ℕ) : ℕ := w + l - Nat.gcd w l

-- Theorem statement
theorem bug_path_tiles :
  tilesVisited width length = 24 :=
sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l938_93831


namespace NUMINAMATH_CALUDE_not_always_greater_slope_for_greater_angle_not_always_inclination_equals_arctan_slope_not_different_angles_for_equal_slopes_l938_93826

-- Define a straight line
structure Line where
  slope : ℝ
  inclination_angle : ℝ

-- Statement 1
theorem not_always_greater_slope_for_greater_angle : 
  ¬ ∀ (l1 l2 : Line), l1.inclination_angle > l2.inclination_angle → l1.slope > l2.slope :=
sorry

-- Statement 2
theorem not_always_inclination_equals_arctan_slope :
  ¬ ∀ (l : Line), l.slope = Real.tan l.inclination_angle → l.inclination_angle = Real.arctan l.slope :=
sorry

-- Statement 3
theorem not_different_angles_for_equal_slopes :
  ¬ ∃ (l1 l2 : Line), l1.slope = l2.slope ∧ l1.inclination_angle ≠ l2.inclination_angle :=
sorry

end NUMINAMATH_CALUDE_not_always_greater_slope_for_greater_angle_not_always_inclination_equals_arctan_slope_not_different_angles_for_equal_slopes_l938_93826


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l938_93871

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 1 < 0) ↔ (∃ x : ℝ, x^2 - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l938_93871


namespace NUMINAMATH_CALUDE_monotone_function_implies_increasing_sequence_but_not_converse_l938_93863

theorem monotone_function_implies_increasing_sequence_but_not_converse 
  (f : ℝ → ℝ) (a : ℕ → ℝ) (h : ∀ n, a n = f n) :
  (∀ x y, 1 ≤ x ∧ x ≤ y → f x ≤ f y) →
  (∀ n m, 1 ≤ n ∧ n ≤ m → a n ≤ a m) ∧
  ¬ ((∀ n m, 1 ≤ n ∧ n ≤ m → a n ≤ a m) →
     (∀ x y, 1 ≤ x ∧ x ≤ y → f x ≤ f y)) :=
by sorry

end NUMINAMATH_CALUDE_monotone_function_implies_increasing_sequence_but_not_converse_l938_93863


namespace NUMINAMATH_CALUDE_cosine_largest_angle_bound_l938_93867

/-- Represents a sequence of non-degenerate triangles -/
def TriangleSequence := ℕ → (ℝ × ℝ × ℝ)

/-- Conditions for a valid triangle sequence -/
def IsValidTriangleSequence (seq : TriangleSequence) : Prop :=
  ∀ n : ℕ, let (a, b, c) := seq n
    0 < a ∧ a ≤ b ∧ b ≤ c ∧ a + b > c

/-- Sum of the shortest sides of the triangles -/
noncomputable def SumShortestSides (seq : TriangleSequence) : ℝ :=
  ∑' n, (seq n).1

/-- Sum of the second longest sides of the triangles -/
noncomputable def SumSecondLongestSides (seq : TriangleSequence) : ℝ :=
  ∑' n, (seq n).2.1

/-- Sum of the longest sides of the triangles -/
noncomputable def SumLongestSides (seq : TriangleSequence) : ℝ :=
  ∑' n, (seq n).2.2

/-- Cosine of the largest angle of the resultant triangle -/
noncomputable def CosLargestAngle (seq : TriangleSequence) : ℝ :=
  let A := SumShortestSides seq
  let B := SumSecondLongestSides seq
  let C := SumLongestSides seq
  (A^2 + B^2 - C^2) / (2 * A * B)

/-- The main theorem stating that the cosine of the largest angle is bounded below by 1 - √2 -/
theorem cosine_largest_angle_bound (seq : TriangleSequence) 
  (h : IsValidTriangleSequence seq) : 
  CosLargestAngle seq ≥ 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_largest_angle_bound_l938_93867


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l938_93893

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l938_93893


namespace NUMINAMATH_CALUDE_triangle_nested_calc_l938_93850

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := a^2 - 2*b

-- State the theorem
theorem triangle_nested_calc : triangle (-2) (triangle 3 2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_nested_calc_l938_93850


namespace NUMINAMATH_CALUDE_all_dice_same_number_probability_l938_93878

/-- The probability of a single die showing a specific number -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice being tossed -/
def num_dice : ℕ := 4

/-- The probability of all dice showing the same number -/
def all_same_prob : ℚ := (single_die_prob) ^ (num_dice - 1)

theorem all_dice_same_number_probability :
  all_same_prob = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_all_dice_same_number_probability_l938_93878


namespace NUMINAMATH_CALUDE_sand_art_calculation_l938_93842

/-- The amount of sand needed to fill shapes given their dimensions and sand density. -/
theorem sand_art_calculation (rectangle_length : ℝ) (rectangle_area : ℝ) (square_side : ℝ) (sand_density : ℝ) : 
  rectangle_length = 7 →
  rectangle_area = 42 →
  square_side = 5 →
  sand_density = 3 →
  rectangle_area * sand_density + square_side * square_side * sand_density = 201 := by
  sorry

#check sand_art_calculation

end NUMINAMATH_CALUDE_sand_art_calculation_l938_93842


namespace NUMINAMATH_CALUDE_apples_used_for_pie_l938_93876

theorem apples_used_for_pie (initial_apples : ℕ) (remaining_apples : ℕ) 
  (h1 : initial_apples = 19) 
  (h2 : remaining_apples = 4) : 
  initial_apples - remaining_apples = 15 := by
  sorry

end NUMINAMATH_CALUDE_apples_used_for_pie_l938_93876


namespace NUMINAMATH_CALUDE_soup_problem_solution_l938_93872

/-- Represents the number of people a can of soup can feed -/
structure SoupCanCapacity where
  adults : Nat
  children : Nat

/-- Represents the problem scenario -/
structure SoupProblem where
  capacity : SoupCanCapacity
  totalCans : Nat
  childrenFed : Nat

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdultsFed (problem : SoupProblem) : Nat :=
  let cansForChildren := (problem.childrenFed + problem.capacity.children - 1) / problem.capacity.children
  let remainingCans := problem.totalCans - cansForChildren
  remainingCans * problem.capacity.adults

/-- Theorem stating the problem and its solution -/
theorem soup_problem_solution (problem : SoupProblem)
  (h1 : problem.capacity = ⟨4, 6⟩)
  (h2 : problem.totalCans = 7)
  (h3 : problem.childrenFed = 18) :
  remainingAdultsFed problem = 16 := by
  sorry

end NUMINAMATH_CALUDE_soup_problem_solution_l938_93872


namespace NUMINAMATH_CALUDE_total_cost_is_2250_l938_93896

def apple_quantity : ℕ := 8
def apple_price : ℕ := 70
def mango_quantity : ℕ := 9
def mango_price : ℕ := 55
def orange_quantity : ℕ := 5
def orange_price : ℕ := 40
def banana_quantity : ℕ := 12
def banana_price : ℕ := 30
def grape_quantity : ℕ := 7
def grape_price : ℕ := 45
def cherry_quantity : ℕ := 4
def cherry_price : ℕ := 80

def total_cost : ℕ := 
  apple_quantity * apple_price + 
  mango_quantity * mango_price + 
  orange_quantity * orange_price + 
  banana_quantity * banana_price + 
  grape_quantity * grape_price + 
  cherry_quantity * cherry_price

theorem total_cost_is_2250 : total_cost = 2250 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_2250_l938_93896


namespace NUMINAMATH_CALUDE_line_direction_vector_l938_93849

/-- Given a line passing through points (-5, 0) and (-2, 2), if its direction vector
    is of the form (2, b), then b = 4/3 -/
theorem line_direction_vector (b : ℚ) : 
  let p1 : ℚ × ℚ := (-5, 0)
  let p2 : ℚ × ℚ := (-2, 2)
  let dir : ℚ × ℚ := (2, b)
  (∃ (k : ℚ), k • (p2.1 - p1.1, p2.2 - p1.2) = dir) → b = 4/3 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l938_93849


namespace NUMINAMATH_CALUDE_boy_initial_height_l938_93820

/-- Represents the growth rates and heights of a tree and a boy -/
structure GrowthProblem where
  initialTreeHeight : ℝ
  finalTreeHeight : ℝ
  finalBoyHeight : ℝ
  treeGrowthRate : ℝ
  boyGrowthRate : ℝ

/-- Theorem stating the boy's initial height given the growth problem parameters -/
theorem boy_initial_height (p : GrowthProblem)
  (h1 : p.initialTreeHeight = 16)
  (h2 : p.finalTreeHeight = 40)
  (h3 : p.finalBoyHeight = 36)
  (h4 : p.treeGrowthRate = 2 * p.boyGrowthRate) :
  p.finalBoyHeight - (p.finalTreeHeight - p.initialTreeHeight) / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_boy_initial_height_l938_93820


namespace NUMINAMATH_CALUDE_parking_space_area_l938_93848

/-- A rectangular parking space with three painted sides -/
structure ParkingSpace where
  length : ℝ
  width : ℝ
  painted_sides_sum : ℝ
  unpainted_side : ℝ
  is_rectangular : length > 0 ∧ width > 0
  three_sides_painted : painted_sides_sum = 2 * width + length
  unpainted_is_length : unpainted_side = length

/-- The area of a parking space is equal to its length multiplied by its width -/
def area (p : ParkingSpace) : ℝ := p.length * p.width

/-- Theorem: If a rectangular parking space has an unpainted side of 9 feet
    and the sum of the painted sides is 37 feet, then its area is 126 square feet -/
theorem parking_space_area 
  (p : ParkingSpace) 
  (h1 : p.unpainted_side = 9) 
  (h2 : p.painted_sides_sum = 37) : 
  area p = 126 := by
  sorry

end NUMINAMATH_CALUDE_parking_space_area_l938_93848


namespace NUMINAMATH_CALUDE_tangent_angle_range_l938_93817

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 2

noncomputable def α (x : ℝ) : ℝ := Real.arctan (3 * x^2 - 1)

theorem tangent_angle_range :
  ∀ x : ℝ, α x ∈ Set.Icc 0 (Real.pi / 2) ∪ Set.Icc (3 * Real.pi / 4) Real.pi :=
by sorry

end NUMINAMATH_CALUDE_tangent_angle_range_l938_93817


namespace NUMINAMATH_CALUDE_jenna_max_tanning_time_l938_93809

/-- Represents Jenna's tanning schedule and calculates the maximum tanning time in a month. -/
def jennaTanningSchedule : ℕ :=
  let minutesPerDay : ℕ := 30
  let daysPerWeek : ℕ := 2
  let weeksFirstPeriod : ℕ := 2
  let minutesLastTwoWeeks : ℕ := 80
  
  let minutesFirstTwoWeeks := minutesPerDay * daysPerWeek * weeksFirstPeriod
  minutesFirstTwoWeeks + minutesLastTwoWeeks

/-- Proves that Jenna's maximum tanning time in a month is 200 minutes. -/
theorem jenna_max_tanning_time : jennaTanningSchedule = 200 := by
  sorry

end NUMINAMATH_CALUDE_jenna_max_tanning_time_l938_93809


namespace NUMINAMATH_CALUDE_largest_sphere_in_folded_rectangle_l938_93835

/-- Represents a rectangle ABCD folded into a tetrahedron D-ABC -/
structure FoldedRectangle where
  ab : ℝ
  bc : ℝ
  d_projects_on_ab : Bool

/-- The radius of the largest inscribed sphere in the tetrahedron formed by folding the rectangle -/
def largest_inscribed_sphere_radius (r : FoldedRectangle) : ℝ := 
  sorry

/-- Theorem stating that for a rectangle with AB = 4 and BC = 3, folded into a tetrahedron
    where D projects onto AB, the radius of the largest inscribed sphere is 3/2 -/
theorem largest_sphere_in_folded_rectangle :
  ∀ (r : FoldedRectangle), 
    r.ab = 4 ∧ r.bc = 3 ∧ r.d_projects_on_ab = true →
    largest_inscribed_sphere_radius r = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_largest_sphere_in_folded_rectangle_l938_93835


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l938_93844

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem tenth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a3 : a 3 = 4) : 
  a 10 = 18 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l938_93844


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l938_93898

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 5*x + 1 - 16
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l938_93898


namespace NUMINAMATH_CALUDE_y_derivative_l938_93851

-- Define the function
noncomputable def y (x : ℝ) : ℝ := 
  -(Real.sinh x) / (2 * (Real.cosh x)^2) + (3/2) * Real.arcsin (Real.tanh x)

-- State the theorem
theorem y_derivative (x : ℝ) : 
  deriv y x = Real.cosh (2*x) / (Real.cosh x)^3 := by sorry

end NUMINAMATH_CALUDE_y_derivative_l938_93851


namespace NUMINAMATH_CALUDE_smaller_cuboid_width_l938_93890

/-- Proves that the width of smaller cuboids is 6 meters given the dimensions of the original cuboid,
    the length and height of smaller cuboids, and the number of smaller cuboids. -/
theorem smaller_cuboid_width
  (original_length : ℝ)
  (original_width : ℝ)
  (original_height : ℝ)
  (small_length : ℝ)
  (small_height : ℝ)
  (num_small_cuboids : ℕ)
  (h1 : original_length = 18)
  (h2 : original_width = 15)
  (h3 : original_height = 2)
  (h4 : small_length = 5)
  (h5 : small_height = 3)
  (h6 : num_small_cuboids = 6) :
  ∃ (small_width : ℝ), small_width = 6 ∧
    original_length * original_width * original_height =
    num_small_cuboids * small_length * small_width * small_height :=
by sorry

end NUMINAMATH_CALUDE_smaller_cuboid_width_l938_93890


namespace NUMINAMATH_CALUDE_total_pies_l938_93853

theorem total_pies (percent_with_forks : ℝ) (pies_without_forks : ℕ) : 
  percent_with_forks = 0.68 →
  pies_without_forks = 640 →
  ∃ (total_pies : ℕ), 
    (1 - percent_with_forks) * (total_pies : ℝ) = pies_without_forks ∧
    total_pies = 2000 :=
by sorry

end NUMINAMATH_CALUDE_total_pies_l938_93853


namespace NUMINAMATH_CALUDE_police_catch_thief_time_l938_93815

/-- Proves that the time taken by the police to catch the thief is 2 hours -/
theorem police_catch_thief_time
  (thief_speed : ℝ)
  (police_station_distance : ℝ)
  (police_delay : ℝ)
  (police_speed : ℝ)
  (h1 : thief_speed = 20)
  (h2 : police_station_distance = 60)
  (h3 : police_delay = 1)
  (h4 : police_speed = 40)
  : ℝ :=
by
  sorry

#check police_catch_thief_time

end NUMINAMATH_CALUDE_police_catch_thief_time_l938_93815


namespace NUMINAMATH_CALUDE_girls_in_class_l938_93843

theorem girls_in_class (total_students : ℕ) (girl_ratio boy_ratio : ℕ) (h1 : total_students = 20) (h2 : girl_ratio = 2) (h3 : boy_ratio = 3) : 
  (girl_ratio * total_students) / (girl_ratio + boy_ratio) = 8 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l938_93843


namespace NUMINAMATH_CALUDE_kanul_cash_theorem_l938_93802

/-- Represents the total amount of cash Kanul had -/
def T : ℝ := sorry

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 5000

/-- The amount spent on machinery -/
def machinery : ℝ := 200

/-- The percentage of total cash spent -/
def percentage_spent : ℝ := 0.30

theorem kanul_cash_theorem : 
  T = (raw_materials + machinery) / (1 - percentage_spent) :=
by sorry

end NUMINAMATH_CALUDE_kanul_cash_theorem_l938_93802


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l938_93819

theorem students_in_both_clubs
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (either_club : ℕ)
  (h1 : total_students = 400)
  (h2 : drama_club = 180)
  (h3 : science_club = 230)
  (h4 : either_club = 350) :
  drama_club + science_club - either_club = 60 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l938_93819


namespace NUMINAMATH_CALUDE_M_bounds_l938_93838

/-- Represents the minimum number of black points needed in an n × n square lattice
    so that every square path has at least one black point on it. -/
def M (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds for M(n) in an n × n square lattice. -/
theorem M_bounds (n : ℕ) : (2 : ℝ) / 7 * (n - 1)^2 ≤ (M n : ℝ) ∧ (M n : ℝ) ≤ 2 / 7 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_M_bounds_l938_93838


namespace NUMINAMATH_CALUDE_equation_solution_l938_93818

theorem equation_solution : ∃ x : ℝ, 4 * (4^x) + Real.sqrt (16 * (16^x)) = 32 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l938_93818


namespace NUMINAMATH_CALUDE_unique_divisible_by_11_l938_93886

/-- A number is divisible by 11 if the alternating sum of its digits is divisible by 11 -/
def isDivisibleBy11 (n : ℕ) : Prop :=
  (n / 100 - (n / 10 % 10) + n % 10) % 11 = 0

/-- The set of three-digit numbers with units digit 5 and hundreds digit 6 -/
def validNumbers : Set ℕ :=
  {n : ℕ | 600 ≤ n ∧ n < 700 ∧ n % 10 = 5 ∧ n / 100 = 6}

theorem unique_divisible_by_11 :
  ∃! n : ℕ, n ∈ validNumbers ∧ isDivisibleBy11 n ∧ n = 605 := by
  sorry

#check unique_divisible_by_11

end NUMINAMATH_CALUDE_unique_divisible_by_11_l938_93886


namespace NUMINAMATH_CALUDE_abs_inequality_solution_l938_93814

theorem abs_inequality_solution (x : ℝ) : 
  |x + 3| - |2*x - 1| < x/2 + 1 ↔ x < -2/5 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_l938_93814


namespace NUMINAMATH_CALUDE_simplest_form_expression_l938_93891

theorem simplest_form_expression (x y a : ℝ) (h : x ≠ 2) : 
  (∀ k : ℝ, k ≠ 0 → (1 : ℝ) / (x - 2) ≠ k * (1 : ℝ) / (x - 2)) ∧ 
  (∃ k : ℝ, k ≠ 0 ∧ (x^2 * y) / (2 * x) = k * (x * y) / 2) ∧
  (∃ k : ℝ, k ≠ 0 ∧ (2 * a) / 8 = k * a / 4) :=
sorry

end NUMINAMATH_CALUDE_simplest_form_expression_l938_93891


namespace NUMINAMATH_CALUDE_company_employees_l938_93840

theorem company_employees (december_employees : ℕ) (increase_percentage : ℚ) : 
  december_employees = 470 →
  increase_percentage = 15 / 100 →
  ∃ (january_employees : ℕ), 
    (↑december_employees : ℚ) = (1 + increase_percentage) * january_employees ∧
    january_employees = 409 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_l938_93840


namespace NUMINAMATH_CALUDE_sad_girls_count_l938_93884

theorem sad_girls_count (total_children happy_children sad_children neutral_children
                         boys girls happy_boys neutral_boys : ℕ)
                        (h1 : total_children = 60)
                        (h2 : happy_children = 30)
                        (h3 : sad_children = 10)
                        (h4 : neutral_children = 20)
                        (h5 : boys = 16)
                        (h6 : girls = 44)
                        (h7 : happy_boys = 6)
                        (h8 : neutral_boys = 4)
                        (h9 : total_children = happy_children + sad_children + neutral_children)
                        (h10 : total_children = boys + girls) :
  girls - (happy_children - happy_boys) - (neutral_children - neutral_boys) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sad_girls_count_l938_93884


namespace NUMINAMATH_CALUDE_harold_bought_four_coffees_l938_93897

/-- The cost of items bought on two different days --/
structure PurchaseData where
  doughnut_cost : ℚ
  harold_total : ℚ
  harold_doughnuts : ℕ
  melinda_total : ℚ
  melinda_doughnuts : ℕ
  melinda_coffees : ℕ

/-- Calculate the number of coffees Harold bought --/
def calculate_harold_coffees (data : PurchaseData) : ℕ :=
  sorry

/-- Theorem stating that Harold bought 4 coffees --/
theorem harold_bought_four_coffees (data : PurchaseData) 
  (h1 : data.doughnut_cost = 45/100)
  (h2 : data.harold_total = 491/100)
  (h3 : data.harold_doughnuts = 3)
  (h4 : data.melinda_total = 759/100)
  (h5 : data.melinda_doughnuts = 5)
  (h6 : data.melinda_coffees = 6) :
  calculate_harold_coffees data = 4 := by
    sorry

end NUMINAMATH_CALUDE_harold_bought_four_coffees_l938_93897


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l938_93899

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  S : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  first_term : a 1 = 2
  third_sum : S 3 = 12

/-- The main theorem combining both parts of the problem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n) ∧
  (∃ k : ℕ, k > 0 ∧ (seq.a 3) * (seq.a (k + 1)) = (seq.S k)^2 ∧ k = 2) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l938_93899


namespace NUMINAMATH_CALUDE_exponential_function_point_l938_93839

theorem exponential_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  a^(1 - 1) - 2 = -1 := by sorry

end NUMINAMATH_CALUDE_exponential_function_point_l938_93839


namespace NUMINAMATH_CALUDE_min_value_theorem_l938_93823

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a + c) * (a + b) = 6 - 2 * Real.sqrt 5) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l938_93823


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l938_93880

theorem triangle_sine_inequality (A B C : ℝ) (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π) (h7 : A + B + C = π) :
  Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8 ∧
  (Real.sin A * Real.sin B * Real.sin C = 3 * Real.sqrt 3 / 8 ↔ A = π/3 ∧ B = π/3 ∧ C = π/3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l938_93880
