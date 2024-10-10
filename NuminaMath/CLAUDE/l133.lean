import Mathlib

namespace number_exceeding_percentage_l133_13329

theorem number_exceeding_percentage (x : ℝ) : x = 0.16 * x + 42 → x = 50 := by
  sorry

end number_exceeding_percentage_l133_13329


namespace number_problem_l133_13305

theorem number_problem (x : ℝ) : (x / 6) * 12 = 9 → x = 4.5 := by sorry

end number_problem_l133_13305


namespace bus_average_speed_with_stoppages_l133_13310

/-- Proves that given a bus with an average speed of 60 km/hr excluding stoppages
    and stopping for 45 minutes per hour, the average speed including stoppages is 15 km/hr. -/
theorem bus_average_speed_with_stoppages
  (speed_without_stoppages : ℝ)
  (stopping_time : ℝ)
  (h1 : speed_without_stoppages = 60)
  (h2 : stopping_time = 45) :
  let moving_time : ℝ := 60 - stopping_time
  let distance_covered : ℝ := speed_without_stoppages * (moving_time / 60)
  let speed_with_stoppages : ℝ := distance_covered
  speed_with_stoppages = 15 := by
  sorry

end bus_average_speed_with_stoppages_l133_13310


namespace lock_combination_l133_13383

/-- Represents a digit in the cryptarithmetic problem -/
structure Digit where
  value : Nat
  is_valid : value < 10

/-- Represents the base of the number system -/
structure Base where
  value : Nat
  is_valid : value > 1

/-- Function to convert a number from base b to base 10 -/
def to_decimal (digits : List Digit) (b : Base) : Nat :=
  sorry

/-- The cryptarithmetic equation -/
def cryptarithmetic_equation (T I D E : Digit) (b : Base) : Prop :=
  to_decimal [T, I, D, E] b + to_decimal [E, D, I, T] b + to_decimal [T, I, D, E] b
  = to_decimal [D, I, E, T] b

/-- All digits are distinct -/
def all_distinct (T I D E : Digit) : Prop :=
  T.value ≠ I.value ∧ T.value ≠ D.value ∧ T.value ≠ E.value ∧
  I.value ≠ D.value ∧ I.value ≠ E.value ∧ D.value ≠ E.value

theorem lock_combination :
  ∃ (T I D E : Digit) (b : Base),
    cryptarithmetic_equation T I D E b ∧
    all_distinct T I D E ∧
    to_decimal [T, I, D] (Base.mk 10 sorry) = 984 :=
  sorry

end lock_combination_l133_13383


namespace male_alligators_mating_season_l133_13311

/-- Represents the alligator population on Lagoon Island -/
structure AlligatorPopulation where
  males : ℕ
  adult_females : ℕ
  juvenile_females : ℕ

/-- The ratio of males to adult females to juvenile females -/
def population_ratio : AlligatorPopulation := ⟨2, 3, 5⟩

/-- The number of adult females during non-mating season -/
def non_mating_adult_females : ℕ := 15

/-- Theorem stating the number of male alligators during mating season -/
theorem male_alligators_mating_season :
  ∃ (pop : AlligatorPopulation),
    pop.adult_females = non_mating_adult_females ∧
    pop.males * population_ratio.adult_females = population_ratio.males * pop.adult_females ∧
    pop.males = 10 :=
by sorry

end male_alligators_mating_season_l133_13311


namespace geometric_sequence_second_term_l133_13393

/-- For a geometric sequence with common ratio 2 and sum of first 3 terms 34685, the second term is 9910 -/
theorem geometric_sequence_second_term : ∀ (a : ℕ → ℚ), 
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  (a 1 + a 2 + a 3 = 34685) →   -- sum of first 3 terms is 34685
  a 2 = 9910 := by
sorry

end geometric_sequence_second_term_l133_13393


namespace one_in_A_l133_13313

def A : Set ℕ := {1, 2}

theorem one_in_A : 1 ∈ A := by sorry

end one_in_A_l133_13313


namespace janet_siblings_difference_l133_13346

/-- The number of siblings each person has -/
structure Siblings where
  masud : ℕ
  janet : ℕ
  carlos : ℕ
  stella : ℕ
  lila : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : Siblings) : Prop :=
  s.masud = 45 ∧
  s.janet = 4 * s.masud - 60 ∧
  s.carlos = s.stella + 20 ∧
  s.stella = (5 * s.carlos - 16) / 2 ∧
  s.lila = s.carlos + s.stella + (s.carlos + s.stella) / 3

/-- The theorem to be proved -/
theorem janet_siblings_difference (s : Siblings) 
  (h : problem_conditions s) : 
  s.janet = s.carlos + s.stella + s.lila - 286 := by
  sorry


end janet_siblings_difference_l133_13346


namespace trapezoid_ratio_theorem_l133_13331

/-- Represents a trapezoid with bases and a point inside it -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  isIsosceles : Bool
  EFGreaterGH : EF > GH

/-- Represents the areas of triangles formed by dividing a trapezoid -/
structure TriangleAreas where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- Theorem stating the ratio of bases in a trapezoid given specific triangle areas -/
theorem trapezoid_ratio_theorem (T : Trapezoid) (A : TriangleAreas) :
  T.isIsosceles = true ∧
  A.area1 = 3 ∧ A.area2 = 4 ∧ A.area3 = 6 ∧ A.area4 = 7 →
  T.EF / T.GH = 7 / 3 := by
  sorry

end trapezoid_ratio_theorem_l133_13331


namespace triangle_side_length_l133_13326

theorem triangle_side_length (a : ℝ) : 
  (5 : ℝ) > 0 ∧ (8 : ℝ) > 0 ∧ a > 0 →
  (5 + 8 > a ∧ 5 + a > 8 ∧ 8 + a > 5) ↔ (3 < a ∧ a < 13) :=
by sorry

end triangle_side_length_l133_13326


namespace polygon_sides_greater_than_diagonals_l133_13324

theorem polygon_sides_greater_than_diagonals (n : ℕ) (d : ℕ) : 
  (n ≥ 3 ∧ d = n * (n - 3) / 2) → (n > d ↔ n = 3 ∨ n = 4) := by
  sorry

end polygon_sides_greater_than_diagonals_l133_13324


namespace cashier_money_value_l133_13339

theorem cashier_money_value (total_bills : ℕ) (five_dollar_bills : ℕ) : 
  total_bills = 126 →
  five_dollar_bills = 84 →
  (total_bills - five_dollar_bills) * 10 + five_dollar_bills * 5 = 840 :=
by sorry

end cashier_money_value_l133_13339


namespace circular_board_holes_l133_13304

/-- The number of holes on the circular board -/
def n : ℕ := 91

/-- Proposition: The number of holes on the circular board satisfies all conditions -/
theorem circular_board_holes :
  n < 100 ∧
  ∃ k : ℕ, k > 0 ∧ 2 * k ≡ 1 [ZMOD n] ∧
  ∃ m : ℕ, m > 0 ∧ 4 * m ≡ 2 * k [ZMOD n] ∧
  6 ≡ 0 [ZMOD n] :=
by sorry

end circular_board_holes_l133_13304


namespace digit_sum_l133_13391

/-- Given digits c and d, if 5c * d4 = 1200, then c + d = 2 -/
theorem digit_sum (c d : ℕ) : 
  c < 10 → d < 10 → (50 + c) * (10 * d + 4) = 1200 → c + d = 2 := by
  sorry

end digit_sum_l133_13391


namespace game_results_l133_13365

/-- Represents a strategy for choosing digits -/
def Strategy := Nat → Nat

/-- Represents the result of the game -/
inductive GameResult
| FirstPlayerWins
| SecondPlayerWins

/-- Determines if a list of digits is divisible by 9 -/
def isDivisibleBy9 (digits : List Nat) : Prop :=
  digits.sum % 9 = 0

/-- Simulates the game for a given k and returns the result -/
def playGame (k : Nat) (firstPlayerStrategy : Strategy) (secondPlayerStrategy : Strategy) : GameResult :=
  sorry

/-- Theorem stating the game results for k = 10 and k = 15 -/
theorem game_results :
  (∀ (firstPlayerStrategy : Strategy),
    ∃ (secondPlayerStrategy : Strategy),
      playGame 10 firstPlayerStrategy secondPlayerStrategy = GameResult.SecondPlayerWins) ∧
  (∃ (firstPlayerStrategy : Strategy),
    ∀ (secondPlayerStrategy : Strategy),
      playGame 15 firstPlayerStrategy secondPlayerStrategy = GameResult.FirstPlayerWins) :=
sorry

end game_results_l133_13365


namespace total_items_for_58_slices_l133_13366

/-- Given the number of slices of bread, calculate the total number of items -/
def totalItems (slices : ℕ) : ℕ :=
  let milk := slices - 18
  let cookies := slices + 27
  slices + milk + cookies

theorem total_items_for_58_slices :
  totalItems 58 = 183 := by
  sorry

end total_items_for_58_slices_l133_13366


namespace max_product_l133_13371

def digits : Finset Nat := {1, 3, 5, 8, 9}

def valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def three_digit (a b c : Nat) : Nat := 100 * a + 10 * b + c

def two_digit (d e : Nat) : Nat := 10 * d + e

theorem max_product :
  ∀ a b c d e,
    valid_combination a b c d e →
    (three_digit a b c) * (two_digit d e) ≤ (three_digit 9 3 1) * (two_digit 8 5) :=
by sorry

end max_product_l133_13371


namespace jerrys_age_l133_13372

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 30 →
  mickey_age = 4 * jerry_age + 10 →
  jerry_age = 5 := by
sorry

end jerrys_age_l133_13372


namespace second_train_speed_l133_13396

/-- Given two trains starting from the same station, traveling in the same direction for 10 hours,
    with the first train moving at 10 mph and the distance between them after 10 hours being 250 miles,
    prove that the speed of the second train is 35 mph. -/
theorem second_train_speed (first_train_speed : ℝ) (time : ℝ) (distance_between : ℝ) :
  first_train_speed = 10 →
  time = 10 →
  distance_between = 250 →
  ∃ second_train_speed : ℝ,
    second_train_speed * time - first_train_speed * time = distance_between ∧
    second_train_speed = 35 := by
  sorry

end second_train_speed_l133_13396


namespace emily_egg_collection_l133_13325

/-- The number of baskets Emily used -/
def num_baskets : ℕ := 303

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 28

/-- The total number of eggs Emily collected -/
def total_eggs : ℕ := num_baskets * eggs_per_basket

theorem emily_egg_collection : total_eggs = 8484 := by
  sorry

end emily_egg_collection_l133_13325


namespace right_triangle_hypotenuse_and_perimeter_l133_13309

theorem right_triangle_hypotenuse_and_perimeter : 
  ∀ (a b h : ℝ), 
    a = 24 → 
    b = 25 → 
    h^2 = a^2 + b^2 → 
    h = Real.sqrt 1201 ∧ 
    a + b + h = 49 + Real.sqrt 1201 := by
  sorry

end right_triangle_hypotenuse_and_perimeter_l133_13309


namespace square_perimeter_sum_l133_13307

theorem square_perimeter_sum (a b : ℝ) (h1 : a + b = 130) (h2 : a - b = 42) :
  4 * (Real.sqrt a.toNNReal + Real.sqrt b.toNNReal) = 4 * (Real.sqrt 86 + 2 * Real.sqrt 11) :=
by sorry

end square_perimeter_sum_l133_13307


namespace earliest_time_84_degrees_l133_13367

/-- Temperature function representing the temperature in Austin, TX on a summer day -/
def T (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The earliest positive real solution to the temperature equation when it equals 84 degrees -/
theorem earliest_time_84_degrees :
  ∀ t : ℝ, t > 0 → T t = 84 → t ≥ 22 :=
by sorry

end earliest_time_84_degrees_l133_13367


namespace complement_union_problem_l133_13361

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_problem : (U \ A) ∪ B = {0, 2, 4} := by
  sorry

end complement_union_problem_l133_13361


namespace anna_ham_sandwich_problem_l133_13341

/-- The number of additional ham slices Anna needs to make a certain number of sandwiches -/
def additional_slices (slices_per_sandwich : ℕ) (current_slices : ℕ) (desired_sandwiches : ℕ) : ℕ :=
  slices_per_sandwich * desired_sandwiches - current_slices

theorem anna_ham_sandwich_problem : 
  additional_slices 3 31 50 = 119 := by
  sorry

end anna_ham_sandwich_problem_l133_13341


namespace green_ball_probability_l133_13358

/-- Represents a container of colored balls -/
structure Container where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- The probability of selecting a specific container -/
def containerProb : ℚ := 1 / 3

/-- The containers in the problem -/
def containers : List Container := [
  ⟨10, 5, 3⟩,   -- Container I
  ⟨3, 5, 2⟩,    -- Container II
  ⟨3, 5, 2⟩     -- Container III
]

/-- The probability of selecting a green ball from a given container -/
def greenProb (c : Container) : ℚ :=
  c.green / (c.red + c.green + c.blue)

/-- The total probability of selecting a green ball -/
def totalGreenProb : ℚ :=
  (containers.map (λ c ↦ containerProb * greenProb c)).sum

theorem green_ball_probability : totalGreenProb = 23 / 54 := by
  sorry

end green_ball_probability_l133_13358


namespace train_speed_in_kmh_l133_13376

-- Define the given parameters
def train_length : ℝ := 80
def bridge_length : ℝ := 295
def crossing_time : ℝ := 30

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_in_kmh :
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * conversion_factor
  speed_kmh = 45 := by sorry

end train_speed_in_kmh_l133_13376


namespace sufficient_not_necessary_condition_l133_13327

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x = -2 → x^2 = 4) ∧
  ¬(∀ x : ℝ, x^2 = 4 → x = -2) :=
by sorry

end sufficient_not_necessary_condition_l133_13327


namespace first_black_ace_most_likely_at_first_position_l133_13338

/-- Probability of drawing the first black ace at position k in a shuffled 52-card deck --/
def probability_first_black_ace (k : ℕ) : ℚ :=
  if k ≥ 1 ∧ k ≤ 51 then (52 - k : ℚ) / 1326 else 0

/-- The position where the probability of drawing the first black ace is maximized --/
def max_probability_position : ℕ := 1

/-- Theorem stating that the probability of drawing the first black ace is maximized at position 1 --/
theorem first_black_ace_most_likely_at_first_position :
  ∀ k, k ≥ 1 → k ≤ 51 → probability_first_black_ace max_probability_position ≥ probability_first_black_ace k :=
by
  sorry


end first_black_ace_most_likely_at_first_position_l133_13338


namespace f_properties_l133_13330

def f (x : ℝ) : ℝ := |x + 3| + |x - 2|

theorem f_properties :
  (∀ x, f x > 7 ↔ x < -4 ∨ x > 3) ∧
  (∀ m, m > 1 → ∃ x, f x = 4 / (m - 1) + m) := by sorry

end f_properties_l133_13330


namespace lowest_cost_l133_13300

variable (x y z a b c : ℝ)

/-- The painting areas of the three rooms satisfy x < y < z -/
axiom area_order : x < y ∧ y < z

/-- The painting costs of the three colors satisfy a < b < c -/
axiom cost_order : a < b ∧ b < c

/-- The total cost function for a painting scheme -/
def total_cost (p q r : ℝ) : ℝ := p*x + q*y + r*z

/-- The theorem stating that az + by + cx is the lowest total cost -/
theorem lowest_cost : 
  ∀ (p q r : ℝ), (p = a ∧ q = b ∧ r = c) ∨ (p = a ∧ q = c ∧ r = b) ∨ 
                  (p = b ∧ q = a ∧ r = c) ∨ (p = b ∧ q = c ∧ r = a) ∨ 
                  (p = c ∧ q = a ∧ r = b) ∨ (p = c ∧ q = b ∧ r = a) →
                  total_cost a b c ≤ total_cost p q r :=
by sorry

end lowest_cost_l133_13300


namespace expand_triple_product_l133_13315

theorem expand_triple_product (x y z : ℝ) :
  (x - 5) * (3 * y + 6) * (z + 4) =
  3 * x * y * z + 6 * x * z - 15 * y * z - 30 * z + 12 * x * y + 24 * x - 60 * y - 120 :=
by sorry

end expand_triple_product_l133_13315


namespace inequality_transformation_l133_13332

theorem inequality_transformation (h : (1/4 : ℝ) > (1/8 : ℝ)) : (2 : ℝ) < (3 : ℝ) := by
  sorry

end inequality_transformation_l133_13332


namespace negation_equivalence_l133_13322

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by
  sorry

end negation_equivalence_l133_13322


namespace inequality_constraint_l133_13381

theorem inequality_constraint (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → x^2 + a*x + 9 ≥ 0) ↔ a ≥ -6 :=
by sorry

end inequality_constraint_l133_13381


namespace total_silverware_l133_13306

/-- The number of types of silverware --/
def num_types : ℕ := 4

/-- The initial number of each type for personal use --/
def initial_personal : ℕ := 5

/-- The number of extra pieces of each type for guests --/
def extra_for_guests : ℕ := 10

/-- The reduction in the number of spoons --/
def spoon_reduction : ℕ := 4

/-- The reduction in the number of butter knives --/
def butter_knife_reduction : ℕ := 4

/-- The reduction in the number of steak knives --/
def steak_knife_reduction : ℕ := 5

/-- The reduction in the number of forks --/
def fork_reduction : ℕ := 3

/-- The theorem stating the total number of silverware pieces Stephanie will buy --/
theorem total_silverware : 
  (initial_personal + extra_for_guests - spoon_reduction) +
  (initial_personal + extra_for_guests - butter_knife_reduction) +
  (initial_personal + extra_for_guests - steak_knife_reduction) +
  (initial_personal + extra_for_guests - fork_reduction) = 44 := by
  sorry

end total_silverware_l133_13306


namespace survey_result_l133_13302

/-- Represents the result of a stratified sampling survey -/
structure SurveyResult where
  totalPopulation : ℕ
  sampleSize : ℕ
  physicsInSample : ℕ
  historyInPopulation : ℕ

/-- Checks if the survey result is valid based on the given conditions -/
def isValidSurvey (s : SurveyResult) : Prop :=
  s.totalPopulation = 1500 ∧
  s.sampleSize = 120 ∧
  s.physicsInSample = 80 ∧
  s.sampleSize - s.physicsInSample > 0 ∧
  s.sampleSize < s.totalPopulation

/-- Theorem stating the result of the survey -/
theorem survey_result (s : SurveyResult) (h : isValidSurvey s) :
  s.historyInPopulation = 500 := by
  sorry

#check survey_result

end survey_result_l133_13302


namespace point_translation_l133_13362

def translate_point (x y dx dy : ℝ) : ℝ × ℝ :=
  (x + dx, y - dy)

theorem point_translation :
  let initial_point : ℝ × ℝ := (1, 2)
  let right_translation : ℝ := 1
  let down_translation : ℝ := 3
  translate_point initial_point.1 initial_point.2 right_translation down_translation = (2, -1) := by
sorry

end point_translation_l133_13362


namespace sector_cone_properties_l133_13308

/-- Represents a cone formed from a sector of a circular sheet -/
structure SectorCone where
  sheet_radius : ℝ
  num_sectors : ℕ

/-- Calculate the height of a cone formed from a sector of a circular sheet -/
def cone_height (c : SectorCone) : ℝ :=
  sorry

/-- Calculate the volume of a cone formed from a sector of a circular sheet -/
def cone_volume (c : SectorCone) : ℝ :=
  sorry

theorem sector_cone_properties (c : SectorCone) 
  (h_radius : c.sheet_radius = 12)
  (h_sectors : c.num_sectors = 4) :
  cone_height c = 3 * Real.sqrt 15 ∧ 
  cone_volume c = 9 * Real.pi * Real.sqrt 15 :=
by sorry

end sector_cone_properties_l133_13308


namespace two_valid_M_values_l133_13373

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem two_valid_M_values :
  ∃! (s : Finset ℕ), 
    (∀ M ∈ s, is_two_digit M ∧ 
      let diff := M - reverse_digits M
      diff > 0 ∧ 
      is_perfect_cube diff ∧ 
      27 < diff ∧ 
      diff < 100) ∧
    s.card = 2 := by sorry

end two_valid_M_values_l133_13373


namespace linear_equation_solution_l133_13328

theorem linear_equation_solution (x y : ℝ) : 3 * x + y = 1 → y = -3 * x + 1 := by
  sorry

end linear_equation_solution_l133_13328


namespace math_homework_pages_l133_13320

theorem math_homework_pages 
  (total_pages : ℕ) 
  (math_pages : ℕ) 
  (reading_pages : ℕ) 
  (problems_per_page : ℕ) 
  (total_problems : ℕ) :
  total_pages = math_pages + reading_pages →
  reading_pages = 6 →
  problems_per_page = 4 →
  total_problems = 40 →
  math_pages = 4 := by
sorry

end math_homework_pages_l133_13320


namespace circular_field_diameter_circular_field_diameter_proof_l133_13317

/-- The diameter of a circular field given the cost of fencing per meter and the total cost -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- Proof that the diameter of the circular field is approximately 28 meters -/
theorem circular_field_diameter_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |circular_field_diameter 1.50 131.95 - 28| < ε :=
sorry

end circular_field_diameter_circular_field_diameter_proof_l133_13317


namespace seating_arrangements_l133_13343

/-- The number of ways to arrange n people in a row. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row, 
    where a group of k people must sit consecutively. -/
def arrangementsWithConsecutiveGroup (n k : ℕ) : ℕ := 
  arrangements (n - k + 1) * arrangements k

/-- The number of ways to arrange 10 people in a row, 
    where 4 specific people cannot sit in 4 consecutive seats. -/
theorem seating_arrangements : 
  arrangements 10 - arrangementsWithConsecutiveGroup 10 4 = 3507840 := by
  sorry

#eval arrangements 10 - arrangementsWithConsecutiveGroup 10 4

end seating_arrangements_l133_13343


namespace puzzle_time_relationship_l133_13369

/-- Represents the time needed to complete a puzzle given the gluing rate -/
def puzzle_completion_time (initial_pieces : ℕ) (pieces_per_minute : ℕ) : ℕ :=
  (initial_pieces - 1) / (pieces_per_minute - 1)

/-- Theorem stating the relationship between puzzle completion times
    with different gluing rates -/
theorem puzzle_time_relationship :
  ∀ (initial_pieces : ℕ),
    initial_pieces > 1 →
    puzzle_completion_time initial_pieces 2 = 120 →
    puzzle_completion_time initial_pieces 3 = 60 := by
  sorry

end puzzle_time_relationship_l133_13369


namespace some_number_value_l133_13342

theorem some_number_value (x : ℝ) : 65 + 5 * 12 / (x / 3) = 66 → x = 180 := by
  sorry

end some_number_value_l133_13342


namespace quadratic_increasing_implies_a_bound_l133_13398

/-- A function f is increasing on an interval [a, +∞) if for all x₁, x₂ in the interval with x₁ < x₂, f(x₁) < f(x₂) --/
def IncreasingOn (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ → x₁ < x₂ → f x₁ < f x₂

/-- The quadratic function f(x) = x^2 + 2ax + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem quadratic_increasing_implies_a_bound (a : ℝ) :
  IncreasingOn (f a) 2 → a ≥ -2 := by
  sorry

end quadratic_increasing_implies_a_bound_l133_13398


namespace jogger_faster_speed_l133_13333

/-- Represents the jogger's speed and distance scenario -/
def JoggerScenario (actual_distance : ℝ) (actual_speed : ℝ) (faster_distance : ℝ) (faster_speed : ℝ) : Prop :=
  (actual_distance / actual_speed) = (faster_distance / faster_speed)

/-- Theorem stating the jogger's faster speed given the conditions -/
theorem jogger_faster_speed :
  ∀ (actual_distance actual_speed faster_distance faster_speed : ℝ),
    actual_distance = 30 →
    actual_speed = 12 →
    faster_distance = actual_distance + 10 →
    JoggerScenario actual_distance actual_speed faster_distance faster_speed →
    faster_speed = 16 := by
  sorry


end jogger_faster_speed_l133_13333


namespace susans_roses_l133_13387

theorem susans_roses (D : ℚ) : 
  -- Initial number of roses is 12D
  (12 * D : ℚ) > 0 →
  -- Half given to daughter, half placed in vase
  let vase_roses := 6 * D
  -- One-third of vase flowers wilted
  let unwilted_ratio := 2 / 3
  -- 12 flowers remained after removing wilted ones
  unwilted_ratio * vase_roses = 12 →
  -- Prove that D = 3
  D = 3 := by
sorry

end susans_roses_l133_13387


namespace total_cost_theorem_l133_13336

def cost_of_meat (pork_price chicken_price pork_weight chicken_weight : ℝ) : ℝ :=
  pork_price * pork_weight + chicken_price * chicken_weight

theorem total_cost_theorem (pork_price : ℝ) (h1 : pork_price = 6) :
  let chicken_price := pork_price - 2
  cost_of_meat pork_price chicken_price 1 3 = 18 := by
  sorry

end total_cost_theorem_l133_13336


namespace infinite_solutions_condition_l133_13359

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
sorry

end infinite_solutions_condition_l133_13359


namespace first_number_proof_l133_13384

theorem first_number_proof (x : ℝ) (h : x / 14.5 = 175) : x = 2537.5 := by
  sorry

end first_number_proof_l133_13384


namespace range_of_f_l133_13351

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 2 :=
by sorry

end range_of_f_l133_13351


namespace mississippi_permutations_l133_13388

theorem mississippi_permutations :
  let total_letters : ℕ := 11
  let m_count : ℕ := 1
  let i_count : ℕ := 4
  let s_count : ℕ := 4
  let p_count : ℕ := 2
  (Nat.factorial total_letters) / 
  (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count) = 34650 := by
  sorry

end mississippi_permutations_l133_13388


namespace bruce_goals_l133_13370

theorem bruce_goals (bruce_goals : ℕ) 
  (michael_goals : ℕ)
  (h1 : michael_goals = 3 * bruce_goals)
  (h2 : bruce_goals + michael_goals = 16) : 
  bruce_goals = 4 := by
sorry

end bruce_goals_l133_13370


namespace school_population_l133_13349

theorem school_population (x : ℝ) : 
  (242 = (x / 100) * (50 / 100 * x)) → x = 220 := by
  sorry

end school_population_l133_13349


namespace matrix_inverse_from_eigenvectors_l133_13323

theorem matrix_inverse_from_eigenvectors :
  ∀ (a b c d : ℝ),
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A.mulVec ![1, 1] = (6 : ℝ) • ![1, 1]) →
  (A.mulVec ![3, -2] = (1 : ℝ) • ![3, -2]) →
  A⁻¹ = !![2/3, -1/2; -1/3, 1/2] :=
by sorry

end matrix_inverse_from_eigenvectors_l133_13323


namespace sequence_sum_equals_n_squared_l133_13363

def sequence_sum (n : ℕ) : ℕ :=
  (List.range (n + 1)).sum + (List.range n).sum

theorem sequence_sum_equals_n_squared (n : ℕ) : sequence_sum n = n^2 := by
  sorry

end sequence_sum_equals_n_squared_l133_13363


namespace carlos_singles_percentage_l133_13390

/-- Represents the statistics of Carlos's baseball hits -/
structure BaseballStats where
  total_hits : ℕ
  home_runs : ℕ
  triples : ℕ
  doubles : ℕ

/-- Calculates the percentage of singles in Carlos's hits -/
def percentage_singles (stats : BaseballStats) : ℚ :=
  let non_singles := stats.home_runs + stats.triples + stats.doubles
  let singles := stats.total_hits - non_singles
  (singles : ℚ) / stats.total_hits * 100

/-- Carlos's baseball statistics -/
def carlos_stats : BaseballStats :=
  { total_hits := 50
  , home_runs := 3
  , triples := 2
  , doubles := 8 }

/-- Theorem stating that the percentage of singles in Carlos's hits is 74% -/
theorem carlos_singles_percentage :
  percentage_singles carlos_stats = 74 := by
  sorry


end carlos_singles_percentage_l133_13390


namespace complement_N_subset_M_l133_13360

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x * (x - 3) < 0}
def N : Set ℝ := {x | x < 1 ∨ x ≥ 3}

-- State the theorem
theorem complement_N_subset_M : (𝒰 \ N) ⊆ M := by
  sorry

end complement_N_subset_M_l133_13360


namespace sphere_volume_l133_13301

theorem sphere_volume (r : ℝ) (d V : ℝ) (h : d = (16 / 9 * V) ^ (1 / 3)) (h_r : r = 1 / 3) : V = 1 / 6 := by
  sorry

end sphere_volume_l133_13301


namespace congruence_solution_l133_13368

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end congruence_solution_l133_13368


namespace min_participants_is_100_l133_13374

/-- Represents the number of correct answers for each question in the quiz. -/
structure QuizResults where
  q1 : Nat
  q2 : Nat
  q3 : Nat
  q4 : Nat

/-- Calculates the minimum number of participants given quiz results. -/
def minParticipants (results : QuizResults) : Nat :=
  ((results.q1 + results.q2 + results.q3 + results.q4 + 1) / 2)

/-- Theorem: The minimum number of participants in the quiz is 100. -/
theorem min_participants_is_100 (results : QuizResults) 
  (h1 : results.q1 = 90)
  (h2 : results.q2 = 50)
  (h3 : results.q3 = 40)
  (h4 : results.q4 = 20)
  (h5 : ∀ n : Nat, n ≤ minParticipants results → 
       2 * n ≥ results.q1 + results.q2 + results.q3 + results.q4) :
  minParticipants results = 100 := by
  sorry

#eval minParticipants ⟨90, 50, 40, 20⟩

end min_participants_is_100_l133_13374


namespace nancys_water_intake_l133_13318

/-- Nancy's weight in pounds -/
def nancys_weight : ℝ := 90

/-- The percentage of body weight Nancy drinks in water -/
def water_percentage : ℝ := 0.6

/-- The amount of water Nancy drinks daily in pounds -/
def water_intake : ℝ := nancys_weight * water_percentage

theorem nancys_water_intake : water_intake = 54 := by
  sorry

end nancys_water_intake_l133_13318


namespace rectangle_area_l133_13334

theorem rectangle_area (x y : ℝ) (h_perimeter : x + y = 6) (h_diagonal : x^2 + y^2 = 25) :
  x * y = 5.5 := by
  sorry

end rectangle_area_l133_13334


namespace bridget_apples_l133_13355

theorem bridget_apples (x : ℕ) : 
  (x / 3 : ℚ) + 5 + 2 + 8 = x → x = 30 :=
by sorry

end bridget_apples_l133_13355


namespace expression_value_l133_13316

theorem expression_value (x y : ℚ) (hx : x = -5/4) (hy : y = -3/2) :
  -2 * x - y^2 = 1/4 := by sorry

end expression_value_l133_13316


namespace max_final_number_l133_13375

/-- The game function that takes a list of integers and returns the largest prime divisor of their sum -/
def game (pair : List Nat) : Nat :=
  sorry

/-- Function to perform one round of the game on a list of numbers -/
def gameRound (numbers : List Nat) : List Nat :=
  sorry

/-- Function to play the game until only one number remains -/
def playUntilOne (numbers : List Nat) : Nat :=
  sorry

theorem max_final_number : 
  ∃ (finalPairing : List (List Nat)), 
    (finalPairing.join = List.range 32) ∧ 
    (∀ pair ∈ finalPairing, pair.length = 2) ∧
    (playUntilOne (finalPairing.map game) = 11) ∧
    (∀ otherPairing : List (List Nat), 
      (otherPairing.join = List.range 32) → 
      (∀ pair ∈ otherPairing, pair.length = 2) →
      playUntilOne (otherPairing.map game) ≤ 11) :=
sorry

end max_final_number_l133_13375


namespace problem_statement_l133_13350

theorem problem_statement (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 152) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 154 := by
  sorry

end problem_statement_l133_13350


namespace fraction_value_l133_13389

theorem fraction_value (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : 1/a + 1/b = 3) :
  (a + 2*a*b + b) / (2*a*b - a - b) = -5 := by
  sorry

end fraction_value_l133_13389


namespace prob_win_at_least_once_l133_13395

-- Define the probability of winning a single game
def prob_win_single : ℚ := 1 / 9

-- Define the probability of losing a single game
def prob_lose_single : ℚ := 1 - prob_win_single

-- Define the number of games played
def num_games : ℕ := 3

-- Theorem statement
theorem prob_win_at_least_once :
  1 - prob_lose_single ^ num_games = 217 / 729 := by
  sorry

end prob_win_at_least_once_l133_13395


namespace arccos_one_over_sqrt_two_l133_13377

theorem arccos_one_over_sqrt_two (π : ℝ) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end arccos_one_over_sqrt_two_l133_13377


namespace octahedron_cube_volume_ratio_l133_13344

/-- Given a cube, an octahedron is formed by joining the centers of adjoining faces. -/
structure OctahedronFromCube where
  cube_side : ℝ
  cube_volume : ℝ
  octahedron_side : ℝ
  octahedron_volume : ℝ

/-- The ratio of the volume of the octahedron to the volume of the cube is 1/6. -/
theorem octahedron_cube_volume_ratio (o : OctahedronFromCube) :
  o.octahedron_volume / o.cube_volume = 1 / 6 := by
  sorry

end octahedron_cube_volume_ratio_l133_13344


namespace opposite_reciprocal_absolute_value_l133_13399

theorem opposite_reciprocal_absolute_value (a b c d m : ℝ) : 
  (a = -b) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (abs m = 5) →  -- absolute value of m is 5
  (-a - m * c * d - b = 5 ∨ -a - m * c * d - b = -5) :=
by
  sorry

end opposite_reciprocal_absolute_value_l133_13399


namespace water_evaporation_per_day_l133_13352

def initial_water : ℝ := 10
def evaporation_period : ℕ := 20
def evaporation_percentage : ℝ := 0.12

theorem water_evaporation_per_day :
  let total_evaporated := initial_water * evaporation_percentage
  let daily_evaporation := total_evaporated / evaporation_period
  daily_evaporation = 0.06 := by sorry

end water_evaporation_per_day_l133_13352


namespace special_triangle_rs_distance_l133_13348

/-- Triangle ABC with altitude CH and inscribed circles in ACH and BCH -/
structure SpecialTriangle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  -- CH is an altitude
  altitude : (C.1 - H.1) * (B.1 - A.1) + (C.2 - H.2) * (B.2 - A.2) = 0
  -- R is on CH
  r_on_ch : ∃ t : ℝ, R = (C.1 + t * (H.1 - C.1), C.2 + t * (H.2 - C.2))
  -- S is on CH
  s_on_ch : ∃ t : ℝ, S = (C.1 + t * (H.1 - C.1), C.2 + t * (H.2 - C.2))
  -- Side lengths
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2000
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 1997
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 1998

/-- The distance between R and S is 2001/4000 -/
theorem special_triangle_rs_distance (t : SpecialTriangle) :
  Real.sqrt ((t.R.1 - t.S.1)^2 + (t.R.2 - t.S.2)^2) = 2001 / 4000 := by
  sorry

end special_triangle_rs_distance_l133_13348


namespace fifty_cent_items_count_l133_13386

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents50 : ℕ
  dollars2 : ℕ
  dollars4 : ℕ

/-- The proposition that the given item counts satisfy the problem conditions -/
def satisfiesConditions (counts : ItemCounts) : Prop :=
  counts.cents50 + counts.dollars2 + counts.dollars4 = 50 ∧
  50 * counts.cents50 + 200 * counts.dollars2 + 400 * counts.dollars4 = 5000

/-- The theorem stating that the only solution satisfying the conditions has 36 items at 50 cents -/
theorem fifty_cent_items_count :
  ∀ counts : ItemCounts, satisfiesConditions counts → counts.cents50 = 36 := by
  sorry

end fifty_cent_items_count_l133_13386


namespace pams_apple_bags_l133_13397

theorem pams_apple_bags (gerald_apples_per_bag : ℕ) (pam_total_apples : ℕ) : 
  gerald_apples_per_bag = 40 →
  pam_total_apples = 1200 →
  ∃ (pam_bags : ℕ), pam_bags * (3 * gerald_apples_per_bag) = pam_total_apples ∧ pam_bags = 10 :=
by sorry

end pams_apple_bags_l133_13397


namespace marble_distribution_proof_l133_13364

/-- The number of marbles in the jar -/
def total_marbles : ℕ := 312

/-- The number of people in the group today -/
def group_size : ℕ := 24

/-- The number of additional people joining in the future scenario -/
def additional_people : ℕ := 2

/-- The decrease in marbles per person in the future scenario -/
def marble_decrease : ℕ := 1

theorem marble_distribution_proof :
  (total_marbles / group_size = total_marbles / (group_size + additional_people) + marble_decrease) ∧
  (total_marbles % group_size = 0) :=
sorry

end marble_distribution_proof_l133_13364


namespace max_b_value_l133_13356

theorem max_b_value (x b : ℤ) : 
  x^2 + b*x = -20 → 
  b > 0 → 
  ∃ (y : ℤ), x^2 + y*x = -20 ∧ y > 0 → 
  y ≤ 21 :=
sorry

end max_b_value_l133_13356


namespace rabbit_distribution_theorem_l133_13319

/-- Represents the number of pet stores --/
def num_stores : ℕ := 5

/-- Represents the number of parent rabbits --/
def num_parents : ℕ := 2

/-- Represents the number of offspring rabbits --/
def num_offspring : ℕ := 4

/-- Represents the total number of rabbits --/
def total_rabbits : ℕ := num_parents + num_offspring

/-- 
Represents the number of ways to distribute rabbits to pet stores 
such that no store gets both a parent and a child
--/
def distribution_ways : ℕ := sorry

theorem rabbit_distribution_theorem : 
  distribution_ways = 560 := by sorry

end rabbit_distribution_theorem_l133_13319


namespace universal_transportation_method_l133_13392

-- Define a type for cities
variable {City : Type}

-- Define a relation for connectivity between cities
variable (connected : City → City → Prop)

-- Define air and water connectivity
variable (air_connected : City → City → Prop)
variable (water_connected : City → City → Prop)

-- Axiom: Any two cities are connected by either air or water
axiom connectivity : ∀ (c1 c2 : City), c1 ≠ c2 → air_connected c1 c2 ∨ water_connected c1 c2

-- Define the theorem
theorem universal_transportation_method 
  (h : ∀ (c1 c2 : City), connected c1 c2 ↔ (air_connected c1 c2 ∨ water_connected c1 c2)) :
  (∀ (c1 c2 : City), air_connected c1 c2) ∨ (∀ (c1 c2 : City), water_connected c1 c2) :=
sorry

end universal_transportation_method_l133_13392


namespace sqrt_equation_solution_l133_13379

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (4 - 5 * x) = 10 → x = -19.2 := by
  sorry

end sqrt_equation_solution_l133_13379


namespace walking_speed_problem_l133_13357

/-- Proves that given a circular track of 640 m, two people walking in opposite directions 
    from the same starting point, meeting after 4.8 minutes, with one person walking at 3.8 km/hr, 
    the other person's speed is 4.2 km/hr. -/
theorem walking_speed_problem (track_length : ℝ) (meeting_time : ℝ) (geeta_speed : ℝ) :
  track_length = 640 →
  meeting_time = 4.8 →
  geeta_speed = 3.8 →
  ∃ lata_speed : ℝ,
    lata_speed = 4.2 ∧
    (lata_speed + geeta_speed) * meeting_time / 60 = track_length / 1000 :=
by sorry

end walking_speed_problem_l133_13357


namespace zachary_crunches_pushups_difference_l133_13347

/-- Zachary's push-ups -/
def zachary_pushups : ℕ := 46

/-- Zachary's crunches -/
def zachary_crunches : ℕ := 58

/-- David's push-ups in terms of Zachary's -/
def david_pushups : ℕ := zachary_pushups + 38

/-- David's crunches in terms of Zachary's -/
def david_crunches : ℕ := zachary_crunches - 62

/-- Theorem stating the difference between Zachary's crunches and push-ups -/
theorem zachary_crunches_pushups_difference :
  zachary_crunches - zachary_pushups = 12 := by sorry

end zachary_crunches_pushups_difference_l133_13347


namespace inequality_proof_l133_13385

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) : 
  x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 ≥ 1 := by
  sorry

end inequality_proof_l133_13385


namespace fib_50_mod_5_l133_13340

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fib_50_mod_5 : fibonacci 50 % 5 = 0 := by
  sorry

end fib_50_mod_5_l133_13340


namespace chernomor_max_coins_l133_13335

/-- Represents the problem of distributing coins among bogatyrs --/
structure BogatyrProblem where
  total_bogatyrs : Nat
  total_coins : Nat

/-- Represents a distribution of bogatyrs into groups --/
structure Distribution where
  groups : List Nat
  coins_per_group : List Nat

/-- Calculates the remainder for Chernomor given a distribution --/
def remainder (d : Distribution) : Nat :=
  d.groups.zip d.coins_per_group
    |> List.map (fun (g, c) => c % g)
    |> List.sum

/-- The maximum remainder Chernomor can get with arbitrary distribution --/
def max_remainder_arbitrary (p : BogatyrProblem) : Nat :=
  sorry

/-- The maximum remainder Chernomor can get with equal distribution --/
def max_remainder_equal (p : BogatyrProblem) : Nat :=
  sorry

theorem chernomor_max_coins (p : BogatyrProblem) 
  (h1 : p.total_bogatyrs = 33) (h2 : p.total_coins = 240) : 
  max_remainder_arbitrary p = 31 ∧ max_remainder_equal p = 30 := by
  sorry

end chernomor_max_coins_l133_13335


namespace fourth_altitude_is_six_times_radius_l133_13337

/-- A tetrahedron with an inscribed sphere -/
structure Tetrahedron :=
  (r : ℝ)  -- radius of the inscribed sphere
  (h₁ h₂ h₃ h₄ : ℝ)  -- altitudes of the tetrahedron
  (h₁_eq : h₁ = 3 * r)
  (h₂_eq : h₂ = 4 * r)
  (h₃_eq : h₃ = 4 * r)
  (sum_reciprocals : 1 / h₁ + 1 / h₂ + 1 / h₃ + 1 / h₄ = 1 / r)

/-- The fourth altitude of the tetrahedron is 6 times the radius of its inscribed sphere -/
theorem fourth_altitude_is_six_times_radius (T : Tetrahedron) : T.h₄ = 6 * T.r := by
  sorry

end fourth_altitude_is_six_times_radius_l133_13337


namespace simplify_nested_expression_l133_13382

theorem simplify_nested_expression : -(-(-|(-1)|^2)^3)^4 = -1 := by
  sorry

end simplify_nested_expression_l133_13382


namespace mike_average_weekly_time_l133_13394

/-- Represents Mike's weekly TV and video game schedule --/
structure MikeSchedule where
  mon_wed_fri_tv : ℕ -- Hours of TV on Monday, Wednesday, Friday
  tue_thu_tv : ℕ -- Hours of TV on Tuesday, Thursday
  weekend_tv : ℕ -- Hours of TV on weekends
  vg_days : ℕ -- Number of days Mike plays video games

/-- Calculates the average weekly time Mike spends on TV and video games over 4 weeks --/
def average_weekly_time (s : MikeSchedule) : ℚ :=
  let weekly_tv := s.mon_wed_fri_tv * 3 + s.tue_thu_tv * 2 + s.weekend_tv * 2
  let daily_vg := (weekly_tv / 7 : ℚ) / 2
  let weekly_vg := daily_vg * s.vg_days
  (weekly_tv + weekly_vg) / 7

/-- Theorem stating that Mike's average weekly time spent on TV and video games is 34 hours --/
theorem mike_average_weekly_time :
  let s : MikeSchedule := { mon_wed_fri_tv := 4, tue_thu_tv := 3, weekend_tv := 5, vg_days := 3 }
  average_weekly_time s = 34 := by sorry

end mike_average_weekly_time_l133_13394


namespace product_digit_sum_l133_13354

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

theorem product_digit_sum :
  let product := number1 * number2
  let tens_digit := (product / 10) % 10
  let units_digit := product % 10
  tens_digit + units_digit = 9 := by sorry

end product_digit_sum_l133_13354


namespace max_ski_trips_l133_13378

/-- Proves the maximum number of ski trips in a given time --/
theorem max_ski_trips (lift_time ski_time total_time : ℕ) : 
  lift_time = 15 →
  ski_time = 5 →
  total_time = 120 →
  (total_time / (lift_time + ski_time) : ℕ) = 6 := by
  sorry

#check max_ski_trips

end max_ski_trips_l133_13378


namespace negation_of_proposition_l133_13312

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ ≤ 0) := by
  sorry

end negation_of_proposition_l133_13312


namespace ellipse_fixed_point_intersection_l133_13345

theorem ellipse_fixed_point_intersection :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
  k ≠ 0 →
  A ≠ (2, 0) →
  B ≠ (2, 0) →
  A.1^2 / 4 + A.2^2 / 3 = 1 →
  B.1^2 / 4 + B.2^2 / 3 = 1 →
  A.2 = k * (A.1 - 2/7) →
  B.2 = k * (B.1 - 2/7) →
  (A.1 - 2)^2 + A.2^2 = (B.1 - 2)^2 + B.2^2 →
  (A.1 - 2)^2 + A.2^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 →
  ∃ (m : ℝ), A.2 = k * (A.1 - 2/7) ∧ B.2 = k * (B.1 - 2/7) := by
sorry

end ellipse_fixed_point_intersection_l133_13345


namespace inscribed_box_radius_l133_13303

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  length : ℝ
  width : ℝ
  height : ℝ
  radius : ℝ
  length_eq_twice_height : length = 2 * height
  surface_area_eq_288 : 2 * (length * width + width * height + length * height) = 288
  edge_sum_eq_96 : 4 * (length + width + height) = 96
  inscribed_in_sphere : (2 * radius) ^ 2 = length ^ 2 + width ^ 2 + height ^ 2

/-- The radius of the sphere containing the inscribed box is 4√5 -/
theorem inscribed_box_radius (box : InscribedBox) : box.radius = 4 * Real.sqrt 5 := by
  sorry

end inscribed_box_radius_l133_13303


namespace geometric_sequence_third_term_l133_13314

/-- If 1, 3, and x form a geometric sequence, then x = 9 -/
theorem geometric_sequence_third_term (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ 3 = 1 * r ∧ x = 3 * r) → x = 9 := by
  sorry

end geometric_sequence_third_term_l133_13314


namespace exponentiation_equality_l133_13380

theorem exponentiation_equality : 
  (-2 : ℤ)^3 = -2^3 ∧ 
  (-4 : ℤ)^2 ≠ -4^2 ∧ 
  (-1 : ℤ)^2020 ≠ (-1 : ℤ)^2021 ∧ 
  (2/3 : ℚ)^3 = (2/3 : ℚ)^3 := by sorry

end exponentiation_equality_l133_13380


namespace probability_same_gender_l133_13353

def total_volunteers : ℕ := 5
def male_volunteers : ℕ := 3
def female_volunteers : ℕ := 2
def volunteers_needed : ℕ := 2

def same_gender_combinations : ℕ := (male_volunteers.choose volunteers_needed) + (female_volunteers.choose volunteers_needed)
def total_combinations : ℕ := total_volunteers.choose volunteers_needed

theorem probability_same_gender :
  (same_gender_combinations : ℚ) / total_combinations = 2 / 5 := by sorry

end probability_same_gender_l133_13353


namespace circle_area_from_circumference_l133_13321

theorem circle_area_from_circumference : ∀ (r : ℝ), 
  (2 * π * r = 18 * π) → (π * r^2 = 81 * π) := by
  sorry

end circle_area_from_circumference_l133_13321
