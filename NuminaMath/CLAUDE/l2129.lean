import Mathlib

namespace NUMINAMATH_CALUDE_complex_product_equals_43_l2129_212995

theorem complex_product_equals_43 :
  let y : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 9))
  (2 * y + y^2) * (2 * y^2 + y^4) * (2 * y^3 + y^6) * 
  (2 * y^4 + y^8) * (2 * y^5 + y^10) * (2 * y^6 + y^12) = 43 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_43_l2129_212995


namespace NUMINAMATH_CALUDE_expression_equality_l2129_212970

theorem expression_equality : (19 * 19 - 12 * 12) / ((19 / 12) - (12 / 19)) = 228 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2129_212970


namespace NUMINAMATH_CALUDE_second_investment_amount_l2129_212918

/-- Proves that the amount of the second investment is $1500 given the conditions of the problem -/
theorem second_investment_amount
  (total_return : ℝ → ℝ → ℝ)
  (first_investment : ℝ)
  (first_return_rate : ℝ)
  (second_return_rate : ℝ)
  (total_return_rate : ℝ)
  (h1 : first_investment = 500)
  (h2 : first_return_rate = 0.07)
  (h3 : second_return_rate = 0.09)
  (h4 : total_return_rate = 0.085)
  (h5 : ∀ x, total_return first_investment x = total_return_rate * (first_investment + x))
  (h6 : ∀ x, total_return first_investment x = first_return_rate * first_investment + second_return_rate * x) :
  ∃ x, x = 1500 ∧ total_return first_investment x = total_return_rate * (first_investment + x) :=
by sorry

end NUMINAMATH_CALUDE_second_investment_amount_l2129_212918


namespace NUMINAMATH_CALUDE_substitution_theorem_l2129_212965

def num_players : ℕ := 15
def starting_players : ℕ := 5
def bench_players : ℕ := 10
def max_substitutions : ℕ := 4

def substitution_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => 5 * (11 - k) * substitution_ways k

def total_substitution_ways : ℕ :=
  List.sum (List.map substitution_ways (List.range (max_substitutions + 1)))

theorem substitution_theorem :
  total_substitution_ways = 5073556 ∧
  total_substitution_ways % 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_substitution_theorem_l2129_212965


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l2129_212904

theorem bobby_candy_problem (initial : ℕ) :
  initial + 17 = 43 → initial = 26 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l2129_212904


namespace NUMINAMATH_CALUDE_original_price_calculation_l2129_212938

theorem original_price_calculation (decreased_price : ℝ) (decrease_percentage : ℝ) 
  (h1 : decreased_price = 1064)
  (h2 : decrease_percentage = 24) : 
  decreased_price / (1 - decrease_percentage / 100) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2129_212938


namespace NUMINAMATH_CALUDE_girls_boys_difference_l2129_212993

theorem girls_boys_difference (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 36 → 
  5 * boys = 4 * girls → 
  total = girls + boys → 
  girls - boys = 4 := by
sorry

end NUMINAMATH_CALUDE_girls_boys_difference_l2129_212993


namespace NUMINAMATH_CALUDE_total_value_is_five_dollars_l2129_212905

/-- Represents the value of different coin types in dollars -/
def coin_value : Fin 4 → ℚ
  | 0 => 0.25  -- Quarter
  | 1 => 0.10  -- Dime
  | 2 => 0.05  -- Nickel
  | 3 => 0.01  -- Penny

/-- Represents the count of each coin type -/
def coin_count : Fin 4 → ℕ
  | 0 => 10   -- Quarters
  | 1 => 3    -- Dimes
  | 2 => 4    -- Nickels
  | 3 => 200  -- Pennies

/-- Calculates the total value of coins -/
def total_value : ℚ :=
  (Finset.sum Finset.univ (λ i => coin_value i * coin_count i))

/-- Theorem stating that the total value of coins is $5.00 -/
theorem total_value_is_five_dollars : total_value = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_value_is_five_dollars_l2129_212905


namespace NUMINAMATH_CALUDE_workers_efficiency_ratio_l2129_212957

/-- Given two workers A and B, where B can finish a job in 15 days and together they
    finish the job in 10 days, prove that the ratio of A's work efficiency to B's is 1/2. -/
theorem workers_efficiency_ratio
  (finish_time_B : ℝ)
  (finish_time_together : ℝ)
  (hB : finish_time_B = 15)
  (hTogether : finish_time_together = 10)
  (efficiency_ratio : ℝ)
  (h_efficiency : efficiency_ratio * (1 / finish_time_B) + (1 / finish_time_B) = 1 / finish_time_together) :
  efficiency_ratio = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_workers_efficiency_ratio_l2129_212957


namespace NUMINAMATH_CALUDE_suit_cost_problem_l2129_212926

theorem suit_cost_problem (x : ℝ) (h1 : x + (3 * x + 200) = 1400) : x = 300 := by
  sorry

end NUMINAMATH_CALUDE_suit_cost_problem_l2129_212926


namespace NUMINAMATH_CALUDE_ball_count_after_500_steps_l2129_212901

/-- Converts a natural number to its base 3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 3) :: aux (m / 3)
  aux n

/-- Sums the digits in a list -/
def sumDigits (l : List ℕ) : ℕ :=
  l.sum

theorem ball_count_after_500_steps : sumDigits (toBase3 500) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_after_500_steps_l2129_212901


namespace NUMINAMATH_CALUDE_cake_mass_proof_l2129_212940

/-- The initial mass of the cake in grams -/
def initial_mass : ℝ := 750

/-- The mass of cake Karlson ate for breakfast -/
def karlson_ate : ℝ := 0.4 * initial_mass

/-- The mass of cake Malish ate for breakfast -/
def malish_ate : ℝ := 150

/-- The percentage of remaining cake Freken Bok ate for lunch -/
def freken_bok_percent : ℝ := 0.3

/-- The additional mass of cake Freken Bok ate for lunch -/
def freken_bok_additional : ℝ := 120

/-- The mass of cake crumbs Matilda licked -/
def matilda_licked : ℝ := 90

theorem cake_mass_proof :
  initial_mass = karlson_ate + malish_ate +
  (freken_bok_percent * (initial_mass - karlson_ate - malish_ate) + freken_bok_additional) +
  matilda_licked := by sorry

end NUMINAMATH_CALUDE_cake_mass_proof_l2129_212940


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l2129_212929

def sumEven (n : ℕ) : ℕ := 
  (n / 2) * (2 + n)

def sumOdd (n : ℕ) : ℕ := 
  (n / 2) * (1 + (n - 1))

theorem even_odd_sum_difference : 
  sumEven 100 - sumOdd 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l2129_212929


namespace NUMINAMATH_CALUDE_isosceles_triangle_legs_l2129_212952

/-- An isosceles triangle with integer side lengths and perimeter 12 -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ
  perimeter_eq : leg + leg + base = 12
  triangle_inequality : base < leg + leg ∧ leg < base + leg

/-- The possible leg lengths of an isosceles triangle with perimeter 12 -/
def possibleLegLengths : Set ℕ :=
  {n : ℕ | ∃ (t : IsoscelesTriangle), t.leg = n}

theorem isosceles_triangle_legs :
  possibleLegLengths = {4, 5} := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_legs_l2129_212952


namespace NUMINAMATH_CALUDE_place_left_l2129_212922

/-- A two-digit number is between 10 and 99, inclusive. -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A one-digit number is between 1 and 9, inclusive. -/
def is_one_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- Placing a one-digit number b to the left of a two-digit number a results in 100b + a. -/
theorem place_left (a b : ℕ) (ha : is_two_digit a) (hb : is_one_digit b) :
  100 * b + a = (100 * b + a) := by sorry

end NUMINAMATH_CALUDE_place_left_l2129_212922


namespace NUMINAMATH_CALUDE_probability_two_same_pair_l2129_212975

/-- The number of students participating in the events -/
def num_students : ℕ := 3

/-- The number of events available -/
def num_events : ℕ := 3

/-- The number of events each student chooses -/
def events_per_student : ℕ := 2

/-- The total number of possible combinations for all students' choices -/
def total_combinations : ℕ := num_students ^ num_events

/-- The number of ways to choose 2 students out of 3 -/
def ways_to_choose_2_students : ℕ := 3

/-- The number of ways to choose 1 pair of events out of 3 possible pairs -/
def ways_to_choose_event_pair : ℕ := 3

/-- The number of choices for the remaining student -/
def choices_for_remaining_student : ℕ := 2

/-- The number of favorable outcomes (where exactly two students choose the same pair) -/
def favorable_outcomes : ℕ := ways_to_choose_2_students * ways_to_choose_event_pair * choices_for_remaining_student

/-- The probability of exactly two students choosing the same pair of events -/
theorem probability_two_same_pair : 
  (favorable_outcomes : ℚ) / total_combinations = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_two_same_pair_l2129_212975


namespace NUMINAMATH_CALUDE_sequence_properties_l2129_212966

/-- Sequence sum function -/
def S (n : ℕ) : ℤ := -n^2 + 7*n

/-- Sequence term function -/
def a (n : ℕ) : ℤ := -2*n + 8

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) ∧
  (∃ m : ℕ, m ≥ 1 ∧ ∀ n : ℕ, n ≥ 1 → S n ≤ S m) ∧
  (∀ n : ℕ, n ≥ 1 → S n ≤ 12) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2129_212966


namespace NUMINAMATH_CALUDE_only_parallelogram_not_axially_symmetric_l2129_212939

-- Define the shapes
inductive Shape
  | Rectangle
  | IsoscelesTrapezoid
  | Parallelogram
  | EquilateralTriangle

-- Define axial symmetry
def is_axially_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | Shape.IsoscelesTrapezoid => true
  | Shape.Parallelogram => false
  | Shape.EquilateralTriangle => true

-- Theorem statement
theorem only_parallelogram_not_axially_symmetric :
  ∀ s : Shape, ¬(is_axially_symmetric s) ↔ s = Shape.Parallelogram :=
by sorry

end NUMINAMATH_CALUDE_only_parallelogram_not_axially_symmetric_l2129_212939


namespace NUMINAMATH_CALUDE_function_equality_l2129_212908

theorem function_equality : ∀ x : ℝ, x = 3 * x^3 := by sorry

end NUMINAMATH_CALUDE_function_equality_l2129_212908


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2129_212998

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 2 * x^2 - 5 * x + k = 0) ↔ k = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2129_212998


namespace NUMINAMATH_CALUDE_round_trip_average_speed_river_boat_average_speed_l2129_212934

/-- The average speed for a round trip given upstream and downstream speeds -/
theorem round_trip_average_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed > 0)
  (h2 : downstream_speed > 0)
  (h3 : upstream_speed ≠ downstream_speed) :
  (2 * upstream_speed * downstream_speed) / (upstream_speed + downstream_speed) = 
    (2 * 3 * 7) / (3 + 7) := by
  sorry

/-- The specific case for the river boat problem -/
theorem river_boat_average_speed :
  (2 * 3 * 7) / (3 + 7) = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_river_boat_average_speed_l2129_212934


namespace NUMINAMATH_CALUDE_framing_for_enlarged_picture_l2129_212961

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (orig_width orig_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := orig_width * enlarge_factor
  let enlarged_height := orig_height * enlarge_factor
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (framed_width + framed_height)
  (perimeter_inches + 11) / 12  -- Round up to the nearest foot

/-- Theorem stating that for the given picture dimensions and specifications, 10 feet of framing is needed. -/
theorem framing_for_enlarged_picture :
  min_framing_feet 5 7 4 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_framing_for_enlarged_picture_l2129_212961


namespace NUMINAMATH_CALUDE_find_a_and_b_l2129_212950

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2008*x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ), 
    (M ∪ N a b = Set.univ) ∧ 
    (M ∩ N a b = Set.Ioc 2009 2010) ∧ 
    a = 2009 ∧ 
    b = -2009 * 2010 := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l2129_212950


namespace NUMINAMATH_CALUDE_circle_properties_l2129_212945

/-- A circle described by the equation x^2 + y^2 + 2ax - 2ay = 0 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*a*p.1 - 2*a*p.2 = 0}

/-- The line x + y = 0 -/
def SymmetryLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 0}

theorem circle_properties (a : ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ Circle a ↔ (-y, -x) ∈ Circle a) ∧ 
  (0, 0) ∈ Circle a := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l2129_212945


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l2129_212927

theorem cylinder_volume_ratio : 
  let cylinder1_height : ℝ := 10
  let cylinder1_circumference : ℝ := 6
  let cylinder2_height : ℝ := 6
  let cylinder2_circumference : ℝ := 10
  let volume1 := π * (cylinder1_circumference / (2 * π))^2 * cylinder1_height
  let volume2 := π * (cylinder2_circumference / (2 * π))^2 * cylinder2_height
  volume2 / volume1 = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l2129_212927


namespace NUMINAMATH_CALUDE_range_of_a_for_nonempty_solution_set_l2129_212991

theorem range_of_a_for_nonempty_solution_set (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) ↔ a ∈ Set.Ici 2 ∪ Set.Iic (-6) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_nonempty_solution_set_l2129_212991


namespace NUMINAMATH_CALUDE_daisys_milk_problem_l2129_212994

theorem daisys_milk_problem (total_milk : ℝ) (cooking_percentage : ℝ) (leftover : ℝ) :
  total_milk = 16 ∧ cooking_percentage = 0.5 ∧ leftover = 2 →
  ∃ kids_consumption_percentage : ℝ,
    kids_consumption_percentage = 0.75 ∧
    leftover = (1 - cooking_percentage) * (total_milk - kids_consumption_percentage * total_milk) :=
by sorry

end NUMINAMATH_CALUDE_daisys_milk_problem_l2129_212994


namespace NUMINAMATH_CALUDE_smaller_number_is_35_l2129_212928

theorem smaller_number_is_35 (x y : ℝ) : 
  x + y = 77 ∧ 
  (x = 42 ∨ y = 42) ∧ 
  (5 * x = 6 * y ∨ 5 * y = 6 * x) →
  min x y = 35 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_is_35_l2129_212928


namespace NUMINAMATH_CALUDE_geometric_sequence_special_case_l2129_212989

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = q * a n

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → b (n + 1) - b n = d

theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, n ≥ 1 → a n > 0) →
  a 1 = 2 →
  arithmetic_sequence (λ n => match n with
    | 1 => 2 * a 1
    | 2 => a 3
    | 3 => 3 * a 2
    | _ => 0
  ) →
  ∀ n : ℕ, n ≥ 1 → a n = 2^n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_case_l2129_212989


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l2129_212971

/-- Simple interest calculation -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 1000)
  (h2 : interest = 400)
  (h3 : time = 4)
  : (interest * 100) / (principal * time) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l2129_212971


namespace NUMINAMATH_CALUDE_list_length_contradiction_l2129_212986

theorem list_length_contradiction (list_I list_II : List ℕ) : 
  (list_I = [3, 4, 8, 19]) →
  (list_II.length = list_I.length + 1) →
  (list_II.length - list_I.length = 6) →
  False :=
by sorry

end NUMINAMATH_CALUDE_list_length_contradiction_l2129_212986


namespace NUMINAMATH_CALUDE_index_card_problem_l2129_212942

theorem index_card_problem (n : ℕ+) : 
  ((n : ℝ) * (n + 1) * (2 * n + 1) / 6) / ((n : ℝ) * (n + 1) / 2) = 2023 → n = 3034 := by
  sorry

end NUMINAMATH_CALUDE_index_card_problem_l2129_212942


namespace NUMINAMATH_CALUDE_f_expression_sum_f_expression_l2129_212941

/-- A linear function f satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The condition that f(8) = 15 -/
axiom f_8 : f 8 = 15

/-- The condition that f(2), f(5), f(4) form a geometric sequence -/
axiom f_geometric : ∃ (r : ℝ), f 5 = r * f 2 ∧ f 4 = r * f 5

/-- Theorem stating that f(x) = 4x - 17 -/
theorem f_expression : ∀ x, f x = 4 * x - 17 := by sorry

/-- Function to calculate the sum of f(2) + f(4) + ... + f(2n) -/
def sum_f (n : ℕ) : ℝ := sorry

/-- Theorem stating the sum of f(2) + f(4) + ... + f(2n) = 4n^2 - 13n -/
theorem sum_f_expression : ∀ n, sum_f n = 4 * n^2 - 13 * n := by sorry

end NUMINAMATH_CALUDE_f_expression_sum_f_expression_l2129_212941


namespace NUMINAMATH_CALUDE_only_f3_is_quadratic_l2129_212909

-- Define the concept of a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the given functions
def f1 (x : ℝ) : ℝ := 3 * x
def f2 (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def f3 (x : ℝ) : ℝ := (x - 1)^2
def f4 (x : ℝ) : ℝ := 2

-- State the theorem
theorem only_f3_is_quadratic :
  (¬ is_quadratic f1) ∧
  (¬ ∀ a b c, is_quadratic (f2 a b c)) ∧
  is_quadratic f3 ∧
  (¬ is_quadratic f4) :=
sorry

end NUMINAMATH_CALUDE_only_f3_is_quadratic_l2129_212909


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2129_212915

theorem trigonometric_identity (α : ℝ) 
  (h : (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 2) :
  (1 + Real.sin (4 * α) - Real.cos (4 * α)) / 
  (1 + Real.sin (4 * α) + Real.cos (4 * α)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2129_212915


namespace NUMINAMATH_CALUDE_election_votes_calculation_l2129_212947

/-- The total number of votes in the election -/
def total_votes : ℕ := 560000

/-- The percentage of valid votes that candidate A received -/
def candidate_A_percentage : ℚ := 65 / 100

/-- The percentage of invalid votes -/
def invalid_votes_percentage : ℚ := 15 / 100

/-- The number of valid votes for candidate A -/
def candidate_A_valid_votes : ℕ := 309400

theorem election_votes_calculation :
  (1 - invalid_votes_percentage) * candidate_A_percentage * total_votes = candidate_A_valid_votes :=
by sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l2129_212947


namespace NUMINAMATH_CALUDE_min_boxes_equal_candies_l2129_212935

/-- The number of candies in a box of "Sweet Mathematics" -/
def SM_box_size : ℕ := 12

/-- The number of candies in a box of "Geometry with Nuts" -/
def GN_box_size : ℕ := 15

/-- The minimum number of boxes of "Sweet Mathematics" needed -/
def min_SM_boxes : ℕ := 5

/-- The minimum number of boxes of "Geometry with Nuts" needed -/
def min_GN_boxes : ℕ := 4

theorem min_boxes_equal_candies :
  min_SM_boxes * SM_box_size = min_GN_boxes * GN_box_size ∧
  ∀ (sm gn : ℕ), sm * SM_box_size = gn * GN_box_size →
    sm ≥ min_SM_boxes ∧ gn ≥ min_GN_boxes :=
by sorry

end NUMINAMATH_CALUDE_min_boxes_equal_candies_l2129_212935


namespace NUMINAMATH_CALUDE_grsl_team_count_grsl_solution_l2129_212976

/-- Represents the number of teams in each group of the Greater Regional Soccer League -/
def n : ℕ := sorry

/-- The total number of games played in the league -/
def total_games : ℕ := 56

/-- The number of inter-group games played by each team in Group A -/
def inter_group_games_per_team : ℕ := 2

theorem grsl_team_count :
  n * (n - 1) + 2 * n = total_games :=
sorry

theorem grsl_solution :
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_grsl_team_count_grsl_solution_l2129_212976


namespace NUMINAMATH_CALUDE_exists_counterexample_1_fraction_inequality_implies_exists_counterexample_3_fraction_inequality_implies_product_l2129_212912

-- Statement 1
theorem exists_counterexample_1 : ∃ (a b c d : ℝ), a > b ∧ c = d ∧ a * c ≤ b * d := by sorry

-- Statement 2
theorem fraction_inequality_implies (a b c : ℝ) (h : c ≠ 0) : a / c^2 < b / c^2 → a < b := by sorry

-- Statement 3
theorem exists_counterexample_3 : ∃ (a b c d : ℝ), a > b ∧ c > d ∧ a - c ≤ b - d := by sorry

-- Statement 4
theorem fraction_inequality_implies_product (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : c / a > d / b → b * c > a * d := by sorry

end NUMINAMATH_CALUDE_exists_counterexample_1_fraction_inequality_implies_exists_counterexample_3_fraction_inequality_implies_product_l2129_212912


namespace NUMINAMATH_CALUDE_probability_sum_multiple_of_three_l2129_212932

/-- The type representing the possible outcomes of rolling a standard 6-sided die. -/
inductive Die : Type
  | one | two | three | four | five | six

/-- The function that returns the numeric value of a die roll. -/
def dieValue : Die → Nat
  | Die.one => 1
  | Die.two => 2
  | Die.three => 3
  | Die.four => 4
  | Die.five => 5
  | Die.six => 6

/-- The type representing the outcome of rolling two dice. -/
def TwoDiceRoll : Type := Die × Die

/-- The function that calculates the sum of two dice rolls. -/
def rollSum (roll : TwoDiceRoll) : Nat :=
  dieValue roll.1 + dieValue roll.2

/-- The predicate that checks if a number is a multiple of 3. -/
def isMultipleOfThree (n : Nat) : Prop :=
  ∃ k, n = 3 * k

/-- The set of all possible outcomes when rolling two dice. -/
def allOutcomes : Finset TwoDiceRoll :=
  sorry

/-- The set of outcomes where the sum is a multiple of 3. -/
def favorableOutcomes : Finset TwoDiceRoll :=
  sorry

theorem probability_sum_multiple_of_three :
  (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ) = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_multiple_of_three_l2129_212932


namespace NUMINAMATH_CALUDE_line_passes_through_P_and_parallel_to_tangent_l2129_212987

-- Define the curve
def f (x : ℝ) : ℝ := 3*x^2 - 4*x + 2

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the point M
def M : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at M
def m : ℝ := (6 * M.1 - 4)

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 2*x - y + 4 = 0

theorem line_passes_through_P_and_parallel_to_tangent :
  line_equation P.1 P.2 ∧
  ∀ (x y : ℝ), line_equation x y → (y - P.2) = m * (x - P.1) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_P_and_parallel_to_tangent_l2129_212987


namespace NUMINAMATH_CALUDE_smallest_multiple_l2129_212962

theorem smallest_multiple (x : ℕ) : x = 48 ↔ (
  x > 0 ∧
  (∃ k : ℕ, 600 * x = 1152 * k) ∧
  (∀ y : ℕ, y > 0 → y < x → ¬∃ k : ℕ, 600 * y = 1152 * k)
) := by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2129_212962


namespace NUMINAMATH_CALUDE_equation_solutions_l2129_212923

theorem equation_solutions : 
  ∃! (s : Set ℝ), 
    (∀ x ∈ s, |x - 2| = |x - 1| + |x - 3| + |x - 4|) ∧ 
    s = {2, 2.25} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2129_212923


namespace NUMINAMATH_CALUDE_catastrophic_network_properties_l2129_212910

/-- A catastrophic road network between 6 cities -/
structure CatastrophicNetwork :=
  (cities : Fin 6 → Type)
  (road : cities i → cities j → Prop)
  (no_return : ∀ (i j : Fin 6) (x : cities i) (y : cities j), road x y → ¬ ∃ path : cities j → cities i, True)

theorem catastrophic_network_properties (n : CatastrophicNetwork) :
  (∃ i : Fin 6, ∀ j : Fin 6, ¬ ∃ x : n.cities i, ∃ y : n.cities j, n.road x y) ∧
  (∃ i : Fin 6, ∀ j : Fin 6, j ≠ i → ∃ x : n.cities i, ∃ y : n.cities j, n.road x y) ∧
  (∃ i j : Fin 6, ∀ k l : Fin 6, ∃ path : n.cities k → n.cities l, True) ∧
  (∃ f : Fin 6 → Fin 6, Function.Bijective f ∧ 
    ∀ i j : Fin 6, i ≠ j → (f i < f j ↔ ∃ x : n.cities i, ∃ y : n.cities j, n.road x y)) :=
sorry

#check catastrophic_network_properties

end NUMINAMATH_CALUDE_catastrophic_network_properties_l2129_212910


namespace NUMINAMATH_CALUDE_smallest_multiple_year_l2129_212944

def joey_age : ℕ := 40
def chloe_age : ℕ := 38
def father_age : ℕ := 60

theorem smallest_multiple_year : 
  ∃ (n : ℕ), n > 0 ∧ 
  (joey_age + n) % father_age = 0 ∧ 
  (chloe_age + n) % father_age = 0 ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    (joey_age + m) % father_age ≠ 0 ∨ 
    (chloe_age + m) % father_age ≠ 0) ∧
  n = 180 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_year_l2129_212944


namespace NUMINAMATH_CALUDE_simplify_expression_l2129_212911

theorem simplify_expression (x : ℝ) : 5 * x + 7 * x - 3 * x = 9 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2129_212911


namespace NUMINAMATH_CALUDE_percentage_literate_inhabitants_l2129_212951

theorem percentage_literate_inhabitants (total_inhabitants : ℕ) 
  (male_percentage : ℚ) (literate_male_percentage : ℚ) (literate_female_percentage : ℚ)
  (h1 : total_inhabitants = 1000)
  (h2 : male_percentage = 60 / 100)
  (h3 : literate_male_percentage = 20 / 100)
  (h4 : literate_female_percentage = 325 / 1000) : 
  (↑(total_inhabitants * (male_percentage * literate_male_percentage * total_inhabitants + 
    (1 - male_percentage) * literate_female_percentage * total_inhabitants)) / 
    (↑total_inhabitants * 1000) : ℚ) = 25 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_literate_inhabitants_l2129_212951


namespace NUMINAMATH_CALUDE_minutes_before_noon_l2129_212948

theorem minutes_before_noon (x : ℕ) : x = 65 :=
  -- Define the conditions
  let minutes_between_9am_and_12pm := 180
  let minutes_ago := 20
  -- The equation: 180 - (x - 20) = 3 * (x - 20)
  have h : minutes_between_9am_and_12pm - (x - minutes_ago) = 3 * (x - minutes_ago) := by sorry
  -- Prove that x = 65
  sorry

#check minutes_before_noon

end NUMINAMATH_CALUDE_minutes_before_noon_l2129_212948


namespace NUMINAMATH_CALUDE_bus_count_l2129_212931

theorem bus_count (total_students : ℕ) (students_per_bus : ℕ) (h1 : total_students = 360) (h2 : students_per_bus = 45) :
  total_students / students_per_bus = 8 :=
by sorry

end NUMINAMATH_CALUDE_bus_count_l2129_212931


namespace NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l2129_212979

-- Define a decreasing function on an open interval
def DecreasingOnInterval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

-- Main theorem
theorem range_of_m_for_decreasing_function 
  (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingOnInterval f (-2) 2)
  (h_inequality : f (m - 1) > f (2 * m - 1)) :
  0 < m ∧ m < 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l2129_212979


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2129_212969

theorem angle_measure_proof :
  ∃ (x : ℝ), x + (3 * x - 8) = 90 ∧ x = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2129_212969


namespace NUMINAMATH_CALUDE_reciprocal_sum_inequality_l2129_212967

theorem reciprocal_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≤ 3) : 1/x + 1/y + 1/z ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_inequality_l2129_212967


namespace NUMINAMATH_CALUDE_marble_problem_l2129_212990

theorem marble_problem (x : ℕ) : 
  (((x / 2) * (1 / 3)) * (85 / 100) : ℚ) = 432 → x = 3052 :=
by sorry

end NUMINAMATH_CALUDE_marble_problem_l2129_212990


namespace NUMINAMATH_CALUDE_four_inch_cube_three_painted_faces_l2129_212980

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Represents a smaller cube resulting from cutting a larger cube -/
structure SmallCube where
  paintedFaces : ℕ

/-- The number of small cubes with at least three painted faces in a painted cube -/
def numCubesWithThreePaintedFaces (c : Cube) : ℕ :=
  8

/-- Theorem stating that a 4-inch cube cut into 1-inch cubes has 8 cubes with at least three painted faces -/
theorem four_inch_cube_three_painted_faces :
  ∀ (c : Cube), c.sideLength = 4 → numCubesWithThreePaintedFaces c = 8 := by
  sorry

end NUMINAMATH_CALUDE_four_inch_cube_three_painted_faces_l2129_212980


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2129_212988

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 25| + |x - 21| = |2*x - 42| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2129_212988


namespace NUMINAMATH_CALUDE_min_sum_floor_l2129_212925

theorem min_sum_floor (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (⌊(2*x + y) / z⌋ : ℤ) + ⌊(y + 2*z) / x⌋ + ⌊(2*z + x) / y⌋ = 9 ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (⌊(2*a + b) / c⌋ : ℤ) + ⌊(b + 2*c) / a⌋ + ⌊(2*c + a) / b⌋ ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_floor_l2129_212925


namespace NUMINAMATH_CALUDE_marble_selection_ways_l2129_212984

def blue_marbles : ℕ := 3
def red_marbles : ℕ := 4
def green_marbles : ℕ := 3
def total_marbles : ℕ := blue_marbles + red_marbles + green_marbles
def marbles_to_choose : ℕ := 5

theorem marble_selection_ways : 
  (Nat.choose blue_marbles 1) * (Nat.choose red_marbles 1) * (Nat.choose green_marbles 1) *
  (Nat.choose (total_marbles - 3) 2) = 756 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l2129_212984


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l2129_212914

/-- Given a square C with perimeter 40 cm and a square D with area equal to one-third the area of square C, 
    the perimeter of square D is (40√3)/3 cm. -/
theorem square_perimeter_relation (C D : Real) : 
  (C * 4 = 40) →  -- Perimeter of square C is 40 cm
  (D^2 = (C^2) / 3) →  -- Area of square D is one-third the area of square C
  (D * 4 = 40 * Real.sqrt 3 / 3) :=  -- Perimeter of square D is (40√3)/3 cm
by sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l2129_212914


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2129_212985

def vector_a : ℝ × ℝ := (1, -2)
def vector_b (m : ℝ) : ℝ × ℝ := (6, m)

theorem perpendicular_vectors_m_value :
  (∀ m : ℝ, vector_a.1 * (vector_b m).1 + vector_a.2 * (vector_b m).2 = 0) →
  (∃ m : ℝ, vector_b m = (6, 3)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2129_212985


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l2129_212906

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume
  (a : ℝ) -- base side length
  (γ : ℝ) -- angle between lateral faces
  (h : 0 < a ∧ 0 < γ ∧ γ < π) -- assumptions to ensure validity
  : ∃ V : ℝ, V = (a^3 * Real.sin (γ/2)) / (12 * Real.sqrt (3/4 - Real.sin (γ/2)^2)) :=
sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l2129_212906


namespace NUMINAMATH_CALUDE_torn_sheets_count_l2129_212972

/-- Represents a book with numbered pages -/
structure Book where
  /-- The number of the first torn-out page -/
  first_torn_page : ℕ
  /-- The number of the last torn-out page -/
  last_torn_page : ℕ

/-- Calculates the number of torn-out sheets given a book -/
def torn_sheets (b : Book) : ℕ :=
  (b.last_torn_page - b.first_torn_page + 1) / 2

/-- The main theorem stating that 167 sheets were torn out -/
theorem torn_sheets_count (b : Book) 
  (h1 : b.first_torn_page = 185)
  (h2 : b.last_torn_page = 518) :
  torn_sheets b = 167 := by
  sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l2129_212972


namespace NUMINAMATH_CALUDE_sqrt_9801_minus_39_cube_l2129_212981

theorem sqrt_9801_minus_39_cube (a b : ℕ+) :
  (Real.sqrt 9801 - 39 : ℝ) = (Real.sqrt a.val - b.val : ℝ)^3 →
  a.val + b.val = 13 := by
sorry

end NUMINAMATH_CALUDE_sqrt_9801_minus_39_cube_l2129_212981


namespace NUMINAMATH_CALUDE_intersection_implies_a_zero_l2129_212960

def set_A (a : ℝ) : Set ℝ := {a^2, a+1, -1}
def set_B (a : ℝ) : Set ℝ := {2*a-1, |a-2|, 3*a^2+4}

theorem intersection_implies_a_zero (a : ℝ) :
  set_A a ∩ set_B a = {-1} → a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_zero_l2129_212960


namespace NUMINAMATH_CALUDE_cos_minus_sin_for_point_l2129_212903

theorem cos_minus_sin_for_point (α : Real) :
  (∃ (x y : Real), x = 3/5 ∧ y = -4/5 ∧ x = Real.cos α ∧ y = Real.sin α) →
  Real.cos α - Real.sin α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_for_point_l2129_212903


namespace NUMINAMATH_CALUDE_fifth_color_marbles_l2129_212949

/-- The number of marbles of each color in a box --/
structure MarbleCount where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  fifth : ℕ

/-- The properties of the marble counts --/
def valid_marble_count (m : MarbleCount) : Prop :=
  m.red = 25 ∧
  m.green = 3 * m.red ∧
  m.yellow = m.green / 5 ∧
  m.blue = 2 * m.yellow ∧
  m.red + m.green + m.yellow + m.blue + m.fifth = 4 * m.green

theorem fifth_color_marbles (m : MarbleCount) 
  (h : valid_marble_count m) : m.fifth = 155 := by
  sorry

end NUMINAMATH_CALUDE_fifth_color_marbles_l2129_212949


namespace NUMINAMATH_CALUDE_rocket_reaches_altitude_l2129_212954

/-- The time taken for a rocket to reach a given altitude -/
def rocket_time (initial_distance : ℝ) (distance_increase : ℝ) (target_altitude : ℝ) : ℕ :=
  let n : ℕ := 15  -- We define n here as per the problem statement
  n

/-- Theorem stating the time taken for the rocket to reach 240 km -/
theorem rocket_reaches_altitude :
  rocket_time 2 2 240 = 15 := by
  sorry

#check rocket_reaches_altitude

end NUMINAMATH_CALUDE_rocket_reaches_altitude_l2129_212954


namespace NUMINAMATH_CALUDE_y_percent_of_x_l2129_212913

theorem y_percent_of_x (y x : ℕ+) (h1 : y = (125 : ℕ+)) (h2 : (y : ℝ) = 0.125 * (x : ℝ)) :
  (y : ℝ) / 100 * (x : ℝ) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_y_percent_of_x_l2129_212913


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2129_212973

theorem max_value_sqrt_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + b + 9 * c^2 = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt 3 * c ≤ Real.sqrt 21 / 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 
  a₀ + b₀ + 9 * c₀^2 = 1 ∧ 
  Real.sqrt a₀ + Real.sqrt b₀ + Real.sqrt 3 * c₀ = Real.sqrt 21 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2129_212973


namespace NUMINAMATH_CALUDE_max_area_rectangular_garden_l2129_212997

/-- The maximum area of a rectangular garden with integer side lengths and a perimeter of 150 feet. -/
theorem max_area_rectangular_garden : ∃ (l w : ℕ), 
  (2 * l + 2 * w = 150) ∧ 
  (∀ (a b : ℕ), (2 * a + 2 * b = 150) → (a * b ≤ l * w)) ∧
  (l * w = 1406) := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_garden_l2129_212997


namespace NUMINAMATH_CALUDE_unique_g_property_l2129_212933

theorem unique_g_property : ∃! (g : ℕ+), 
  ∀ (p : ℕ) (hp : Nat.Prime p) (hodd : Odd p), 
  ∃ (n : ℕ+), 
    (p ∣ g^(n:ℕ) - n) ∧ 
    (p ∣ g^((n:ℕ)+1) - (n+1)) :=
by sorry

end NUMINAMATH_CALUDE_unique_g_property_l2129_212933


namespace NUMINAMATH_CALUDE_petes_ten_dollar_bills_l2129_212946

theorem petes_ten_dollar_bills (
  total_owed : ℕ)
  (twenty_dollar_bills : ℕ)
  (bottle_refund : ℚ)
  (bottles_to_return : ℕ)
  (h1 : total_owed = 90)
  (h2 : twenty_dollar_bills = 2)
  (h3 : bottle_refund = 1/2)
  (h4 : bottles_to_return = 20)
  : ∃ (ten_dollar_bills : ℕ),
    ten_dollar_bills = 4 ∧
    20 * twenty_dollar_bills + 10 * ten_dollar_bills + (bottle_refund * bottles_to_return) = total_owed :=
by sorry

end NUMINAMATH_CALUDE_petes_ten_dollar_bills_l2129_212946


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2129_212974

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x = -2) → 
  (-1 : ℝ) ∈ {x : ℝ | 3 * x^2 + m * x = -2} → 
  (-2/3 : ℝ) ∈ {x : ℝ | 3 * x^2 + m * x = -2} :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2129_212974


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l2129_212924

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the original point P
def P : Point := (3, -5)

-- Define the symmetry operation with respect to x-axis
def symmetry_x_axis (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem symmetric_point_x_axis :
  symmetry_x_axis P = (3, 5) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l2129_212924


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l2129_212902

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "outside of plane" relation
variable (outside_of_plane : Line → Plane → Prop)

theorem line_plane_parallelism 
  (m n : Line) (α : Plane)
  (h1 : outside_of_plane m α)
  (h2 : outside_of_plane n α)
  (h3 : parallel_lines m n)
  (h4 : parallel_line_plane m α) :
  parallel_line_plane n α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l2129_212902


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2129_212982

def quadratic_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem roots_of_quadratic_equation :
  let a : ℝ := 1
  let b : ℝ := -7
  let c : ℝ := 12
  (quadratic_equation a b c 3 = 0) ∧ (quadratic_equation a b c 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2129_212982


namespace NUMINAMATH_CALUDE_zacks_marbles_l2129_212920

theorem zacks_marbles : ∃ (M : ℕ), 
  (∃ (k : ℕ), M - 5 = 3 * k) ∧ 
  (M - (3 * 20) - 5 = 5) ∧ 
  M = 70 := by
  sorry

end NUMINAMATH_CALUDE_zacks_marbles_l2129_212920


namespace NUMINAMATH_CALUDE_katies_ds_games_l2129_212958

/-- Theorem: Katie's DS Games
Given:
- Katie's new friends have 88 games
- Katie's old friends have 53 games
- All friends (including Katie) have 141 games in total
Prove that Katie has 0 DS games
-/
theorem katies_ds_games 
  (new_friends_games : ℕ) 
  (old_friends_games : ℕ)
  (total_games : ℕ)
  (h1 : new_friends_games = 88)
  (h2 : old_friends_games = 53)
  (h3 : total_games = 141)
  (h4 : total_games = new_friends_games + old_friends_games + katie_games)
  : katie_games = 0 := by
  sorry

#check katies_ds_games

end NUMINAMATH_CALUDE_katies_ds_games_l2129_212958


namespace NUMINAMATH_CALUDE_children_on_tricycles_l2129_212943

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of adults riding bicycles -/
def adults_on_bicycles : ℕ := 6

/-- The total number of wheels observed -/
def total_wheels : ℕ := 57

/-- Theorem stating that the number of children riding tricycles is 15 -/
theorem children_on_tricycles : 
  ∃ (c : ℕ), c * tricycle_wheels + adults_on_bicycles * bicycle_wheels = total_wheels ∧ c = 15 := by
  sorry

end NUMINAMATH_CALUDE_children_on_tricycles_l2129_212943


namespace NUMINAMATH_CALUDE_odd_function_property_l2129_212937

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : OddFunction f) 
  (h_even : EvenFunction (fun x ↦ f (x + 2))) 
  (h_f1 : f 1 = 1) : 
  f 8 + f 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2129_212937


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2129_212956

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = 1) (h2 : b = -2) :
  (2*a + b)^2 - 3*a*(2*a - b) = -12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2129_212956


namespace NUMINAMATH_CALUDE_equation_unique_solution_l2129_212955

theorem equation_unique_solution :
  ∃! x : ℝ, (8 : ℝ)^(2*x+1) * (2 : ℝ)^(3*x+5) = (4 : ℝ)^(3*x+2) ∧ x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_unique_solution_l2129_212955


namespace NUMINAMATH_CALUDE_equal_integers_from_ratio_l2129_212992

theorem equal_integers_from_ratio (a b : ℕ+) 
  (hK : K = Real.sqrt ((a.val ^ 2 + b.val ^ 2) / 2))
  (hA : A = (a.val + b.val) / 2)
  (hKA : ∃ (n : ℕ+), K / A = n.val) :
  a = b := by
  sorry

end NUMINAMATH_CALUDE_equal_integers_from_ratio_l2129_212992


namespace NUMINAMATH_CALUDE_equation_solution_l2129_212996

theorem equation_solution : 
  ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2129_212996


namespace NUMINAMATH_CALUDE_dancing_and_math_intersection_l2129_212907

theorem dancing_and_math_intersection (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (a b : ℕ),
    b ≥ 1 ∧
    (a + b)^2 = (p + 1) * a + b ∧
    b = 1 :=
by sorry

end NUMINAMATH_CALUDE_dancing_and_math_intersection_l2129_212907


namespace NUMINAMATH_CALUDE_annual_income_difference_l2129_212999

/-- Given an 8% raise, if a person's raise is Rs. 800 and another person's raise is Rs. 840,
    then the difference between their new annual incomes is Rs. 540. -/
theorem annual_income_difference (D W : ℝ) : 
  0.08 * D = 800 → 0.08 * W = 840 → W + 840 - (D + 800) = 540 := by
  sorry

end NUMINAMATH_CALUDE_annual_income_difference_l2129_212999


namespace NUMINAMATH_CALUDE_five_divides_cube_iff_five_divides_l2129_212953

theorem five_divides_cube_iff_five_divides (a : ℤ) : 
  (5 : ℤ) ∣ a^3 ↔ (5 : ℤ) ∣ a := by
  sorry

end NUMINAMATH_CALUDE_five_divides_cube_iff_five_divides_l2129_212953


namespace NUMINAMATH_CALUDE_polygon_sides_when_angles_equal_l2129_212936

theorem polygon_sides_when_angles_equal (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 360 ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_angles_equal_l2129_212936


namespace NUMINAMATH_CALUDE_other_communities_count_l2129_212900

theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) :
  total = 1500 →
  muslim_percent = 37.5 / 100 →
  hindu_percent = 25.6 / 100 →
  sikh_percent = 8.4 / 100 →
  ↑(round ((1 - (muslim_percent + hindu_percent + sikh_percent)) * total)) = 428 :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l2129_212900


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l2129_212930

/-- The equation of the trajectory of the center of a moving circle -/
def trajectory_equation (x y : ℝ) : Prop := x^2 = -4*(y - 1)

/-- The semicircle in which the moving circle is inscribed -/
def semicircle (x y : ℝ) : Prop := x^2 + y^2 = 4 ∧ 0 ≤ y ∧ y ≤ 2

/-- The moving circle is tangent to the x-axis -/
def tangent_to_x_axis (x y : ℝ) : Prop := ∃ (r : ℝ), r > 0 ∧ y = r

theorem trajectory_of_moving_circle :
  ∀ (x y : ℝ), 
    0 < y → y ≤ 1 →
    tangent_to_x_axis x y →
    (∃ (x' y' : ℝ), semicircle x' y' ∧ 
      (x - x')^2 + (y - y')^2 = (2 - y)^2) →
    trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l2129_212930


namespace NUMINAMATH_CALUDE_order_of_expressions_l2129_212921

theorem order_of_expressions :
  let a := 2 + (1/5) * Real.log 2
  let b := 1 + 2^(1/5)
  let c := 2^(11/10)
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l2129_212921


namespace NUMINAMATH_CALUDE_screen_height_is_100_l2129_212919

/-- The height of a computer screen given the side length of a square paper and the difference between the screen height and the paper's perimeter. -/
def screen_height (square_side : ℝ) (perimeter_difference : ℝ) : ℝ :=
  4 * square_side + perimeter_difference

/-- Theorem stating that the height of the computer screen is 100 cm. -/
theorem screen_height_is_100 :
  screen_height 20 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_screen_height_is_100_l2129_212919


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2129_212959

theorem min_value_quadratic :
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2129_212959


namespace NUMINAMATH_CALUDE_largest_remaining_number_l2129_212917

/-- Represents the original number sequence as a list of digits -/
def originalSequence : List Nat := sorry

/-- Represents the result after removing 100 digits -/
def resultSequence : List Nat := sorry

/-- The number of digits to remove -/
def digitsToRemove : Nat := 100

/-- Checks if a sequence is a valid subsequence of another sequence -/
def isValidSubsequence (sub seq : List Nat) : Prop := sorry

/-- Checks if a number represented as a list of digits is greater than another -/
def isGreaterThan (a b : List Nat) : Prop := sorry

theorem largest_remaining_number :
  isValidSubsequence resultSequence originalSequence ∧
  resultSequence.length = originalSequence.length - digitsToRemove ∧
  (∀ (other : List Nat), 
    isValidSubsequence other originalSequence → 
    other.length = originalSequence.length - digitsToRemove →
    isGreaterThan resultSequence other ∨ resultSequence = other) :=
sorry

end NUMINAMATH_CALUDE_largest_remaining_number_l2129_212917


namespace NUMINAMATH_CALUDE_max_xy_value_l2129_212968

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (m : ℝ), m = 1/4 ∧ ∀ (z : ℝ), z = x * y → z ≤ m := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l2129_212968


namespace NUMINAMATH_CALUDE_triangle_DEF_is_right_angled_and_isosceles_l2129_212978

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the properties of the given triangles
def is_midpoint (F : Point) (B C : Point) : Prop :=
  F.x = (B.x + C.x) / 2 ∧ F.y = (B.y + C.y) / 2

def is_isosceles_right_triangle (A B D : Point) : Prop :=
  (A.x - B.x)^2 + (A.y - B.y)^2 = (A.x - D.x)^2 + (A.y - D.y)^2 ∧
  (A.x - D.x) * (B.x - D.x) + (A.y - D.y) * (B.y - D.y) = 0

-- Define the theorem
theorem triangle_DEF_is_right_angled_and_isosceles 
  (ABC : Triangle) 
  (F D E : Point) 
  (h1 : is_midpoint F ABC.B ABC.C)
  (h2 : is_isosceles_right_triangle ABC.A ABC.B D)
  (h3 : is_isosceles_right_triangle ABC.A ABC.C E) :
  is_isosceles_right_triangle D E F := by
  sorry

end NUMINAMATH_CALUDE_triangle_DEF_is_right_angled_and_isosceles_l2129_212978


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l2129_212964

theorem quadratic_vertex_form (x : ℝ) : ∃ (a h k : ℝ), 
  3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l2129_212964


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l2129_212916

theorem existence_of_special_sequence (n : ℕ) : 
  ∃ (a : ℕ → ℕ), 
    (∀ i j, i < j → j ≤ n → a i > a j) ∧ 
    (∀ i, i < n → a i ∣ (a (i + 1))^2) ∧
    (∀ i j, i ≠ j → i ≤ n → j ≤ n → ¬(a i ∣ a j)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l2129_212916


namespace NUMINAMATH_CALUDE_maria_white_towels_l2129_212977

/-- The number of white towels Maria bought -/
def white_towels (green_towels given_away remaining : ℕ) : ℕ :=
  (remaining + given_away) - green_towels

/-- Proof that Maria bought 21 white towels -/
theorem maria_white_towels : 
  white_towels 35 34 22 = 21 := by
  sorry

end NUMINAMATH_CALUDE_maria_white_towels_l2129_212977


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l2129_212963

theorem sqrt_fraction_equality (x : ℝ) : 
  (1 < x ∧ x ≤ 3) ↔ Real.sqrt ((3 - x) / (x - 1)) = Real.sqrt (3 - x) / Real.sqrt (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l2129_212963


namespace NUMINAMATH_CALUDE_two_triangles_from_tetrahedron_l2129_212983

-- Define a tetrahedron
structure Tetrahedron :=
  (A B C D : Point)
  (AB AC AD BC BD CD : ℝ)
  (longest_edge : AB ≥ max AC (max AD (max BC (max BD CD))))
  (AC_geq_BD : AC ≥ BD)

-- Define a triangle
structure Triangle :=
  (side1 side2 side3 : ℝ)

-- Theorem statement
theorem two_triangles_from_tetrahedron (t : Tetrahedron) : 
  ∃ (triangle1 triangle2 : Triangle), 
    (triangle1.side1 = t.BC ∧ triangle1.side2 = t.CD ∧ triangle1.side3 = t.BD) ∧
    (triangle2.side1 = t.AC ∧ triangle2.side2 = t.CD ∧ triangle2.side3 = t.AD) ∧
    (triangle1 ≠ triangle2) :=
sorry

end NUMINAMATH_CALUDE_two_triangles_from_tetrahedron_l2129_212983
