import Mathlib

namespace total_non_basalt_rocks_l1923_192315

/-- Given two boxes of rocks with some being basalt, calculate the total number of non-basalt rocks --/
theorem total_non_basalt_rocks (total_A total_B basalt_A basalt_B : ℕ) 
  (h1 : total_A = 57)
  (h2 : basalt_A = 25)
  (h3 : total_B = 49)
  (h4 : basalt_B = 19) :
  (total_A - basalt_A) + (total_B - basalt_B) = 62 := by
  sorry

#check total_non_basalt_rocks

end total_non_basalt_rocks_l1923_192315


namespace impossible_to_equalize_l1923_192304

/-- Represents the numbers in the six sectors of the circle -/
structure CircleNumbers where
  sectors : Fin 6 → ℤ

/-- Represents an operation of increasing two adjacent numbers by 1 -/
inductive Operation
  | increase_adjacent : Fin 6 → Operation

/-- Applies an operation to the circle numbers -/
def apply_operation (nums : CircleNumbers) (op : Operation) : CircleNumbers :=
  match op with
  | Operation.increase_adjacent i =>
      let j := (i + 1) % 6
      { sectors := fun k =>
          if k = i || k = j then nums.sectors k + 1 else nums.sectors k }

/-- Checks if all numbers in the circle are equal -/
def all_equal (nums : CircleNumbers) : Prop :=
  ∀ i j : Fin 6, nums.sectors i = nums.sectors j

/-- The main theorem stating that it's impossible to make all numbers equal -/
theorem impossible_to_equalize (initial : CircleNumbers) :
  ¬∃ (ops : List Operation), all_equal (ops.foldl apply_operation initial) :=
sorry

end impossible_to_equalize_l1923_192304


namespace implication_contrapositive_equivalence_l1923_192320

theorem implication_contrapositive_equivalence (R S : Prop) :
  (R → S) ↔ (¬S → ¬R) := by sorry

end implication_contrapositive_equivalence_l1923_192320


namespace pythagorean_diagonal_l1923_192316

theorem pythagorean_diagonal (m : ℕ) (h_m : m ≥ 3) : 
  let width : ℕ := 2 * m
  let diagonal : ℕ := m^2 + 1
  let height : ℕ := diagonal - 2
  (width : ℤ)^2 + height^2 = diagonal^2 := by sorry

end pythagorean_diagonal_l1923_192316


namespace age_problem_l1923_192359

/-- Proves that given the age conditions, Mária is 36 2/3 years old and Anna is 7 1/3 years old -/
theorem age_problem (x y : ℚ) : 
  x + y = 44 → 
  x = 2 * (y - (-1/2 * x + 3/2 * (2/3 * y))) → 
  x = 110/3 ∧ y = 22/3 := by
sorry

end age_problem_l1923_192359


namespace divisor_problem_l1923_192388

theorem divisor_problem (dividend : ℤ) (quotient : ℤ) (remainder : ℤ) (divisor : ℤ) : 
  dividend = 151 ∧ quotient = 11 ∧ remainder = -4 →
  divisor = 14 ∧ dividend = divisor * quotient + remainder :=
by sorry

end divisor_problem_l1923_192388


namespace rational_fraction_implication_l1923_192372

theorem rational_fraction_implication (x : ℝ) :
  (∃ a : ℚ, (x / (x^2 + x + 1) : ℝ) = a) →
  (∃ b : ℚ, (x^2 / (x^4 + x^2 + 1) : ℝ) = b) :=
by sorry

end rational_fraction_implication_l1923_192372


namespace linear_functions_property_l1923_192333

/-- Given two linear functions f and g with specific properties, prove that A + B + 2C equals itself. -/
theorem linear_functions_property (A B C : ℝ) (h1 : A ≠ B) (h2 : A + B ≠ 0) (h3 : C ≠ 0)
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x + B + C)
  (hg : ∀ x, g x = B * x + A - C)
  (h4 : ∀ x, f (g x) - g (f x) = 2 * C) :
  A + B + 2 * C = A + B + 2 * C := by
  sorry

end linear_functions_property_l1923_192333


namespace tangent_slope_point_coordinates_l1923_192306

theorem tangent_slope_point_coordinates :
  ∀ (x y : ℝ), 
    y = 1 / x →  -- The curve equation
    (-1 / x^2) = -4 →  -- The slope of the tangent line
    ((x = 1/2 ∧ y = 2) ∨ (x = -1/2 ∧ y = -2)) := by sorry

end tangent_slope_point_coordinates_l1923_192306


namespace complex_parts_of_3i_times_1_plus_i_l1923_192397

theorem complex_parts_of_3i_times_1_plus_i :
  let z : ℂ := 3 * Complex.I * (1 + Complex.I)
  (z.re = -3) ∧ (z.im = 3) := by sorry

end complex_parts_of_3i_times_1_plus_i_l1923_192397


namespace school_sampling_theorem_l1923_192378

/-- Represents the types of schools --/
inductive SchoolType
  | Primary
  | Middle
  | University

/-- Represents the count of each school type --/
structure SchoolCounts where
  primary : Nat
  middle : Nat
  university : Nat

/-- Represents the result of stratified sampling --/
structure SamplingResult where
  primary : Nat
  middle : Nat
  university : Nat

/-- Calculates the stratified sampling result --/
def stratifiedSample (counts : SchoolCounts) (totalSample : Nat) : SamplingResult :=
  { primary := counts.primary * totalSample / (counts.primary + counts.middle + counts.university),
    middle := counts.middle * totalSample / (counts.primary + counts.middle + counts.university),
    university := counts.university * totalSample / (counts.primary + counts.middle + counts.university) }

/-- Calculates the probability of selecting two primary schools --/
def probabilityTwoPrimary (sample : SamplingResult) : Rat :=
  (sample.primary * (sample.primary - 1)) / (2 * (sample.primary + sample.middle + sample.university) * (sample.primary + sample.middle + sample.university - 1))

theorem school_sampling_theorem (counts : SchoolCounts) (h : counts = { primary := 21, middle := 14, university := 7 }) :
  let sample := stratifiedSample counts 6
  sample = { primary := 3, middle := 2, university := 1 } ∧
  probabilityTwoPrimary sample = 1/5 := by
  sorry

#check school_sampling_theorem

end school_sampling_theorem_l1923_192378


namespace cylinder_cone_volume_relation_l1923_192313

theorem cylinder_cone_volume_relation :
  ∀ (d : ℝ) (h : ℝ),
    d > 0 →
    h = 2 * d →
    π * (d / 2)^2 * h = 81 * π →
    (1 / 3) * π * (d / 2)^2 * h = 27 * π * (6 : ℝ)^(1/3) := by
  sorry

end cylinder_cone_volume_relation_l1923_192313


namespace next_repeated_year_correct_l1923_192380

/-- A year consists of a repeated two-digit number if it can be written as ABAB where A and B are digits -/
def is_repeated_two_digit (year : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ year = a * 1000 + b * 100 + a * 10 + b

/-- The next year after 2020 with a repeated two-digit number -/
def next_repeated_year : ℕ := 2121

theorem next_repeated_year_correct :
  (next_repeated_year > 2020) ∧ 
  (is_repeated_two_digit next_repeated_year) ∧
  (∀ y : ℕ, 2020 < y ∧ y < next_repeated_year → ¬(is_repeated_two_digit y)) ∧
  (next_repeated_year - 2020 = 101) :=
sorry

end next_repeated_year_correct_l1923_192380


namespace ellis_card_difference_l1923_192349

/-- Represents the number of cards each player has -/
structure CardDistribution where
  ellis : ℕ
  orion : ℕ

/-- Calculates the card distribution based on the total cards and ratio -/
def distribute_cards (total : ℕ) (ellis_ratio : ℕ) (orion_ratio : ℕ) : CardDistribution :=
  let part_value := total / (ellis_ratio + orion_ratio)
  { ellis := ellis_ratio * part_value,
    orion := orion_ratio * part_value }

/-- Theorem stating that Ellis has 332 more cards than Orion -/
theorem ellis_card_difference (total : ℕ) (ellis_ratio : ℕ) (orion_ratio : ℕ)
  (h_total : total = 2500)
  (h_ellis_ratio : ellis_ratio = 17)
  (h_orion_ratio : orion_ratio = 13) :
  let distribution := distribute_cards total ellis_ratio orion_ratio
  distribution.ellis - distribution.orion = 332 := by
  sorry


end ellis_card_difference_l1923_192349


namespace root_distance_range_l1923_192389

variables (a b c d : ℝ) (x₁ x₂ : ℝ)

def g (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem root_distance_range (ha : a ≠ 0) 
  (hsum : a + b + c = 0) 
  (hf : f 0 * f 1 > 0) 
  (hx₁ : f x₁ = 0) 
  (hx₂ : f x₂ = 0) 
  (hx_distinct : x₁ ≠ x₂) :
  |x₁ - x₂| ∈ Set.Icc (Real.sqrt 3 / 3) (2 / 3) :=
sorry

end root_distance_range_l1923_192389


namespace saree_price_calculation_l1923_192303

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.15) = 306 → P = 450 := by
  sorry

end saree_price_calculation_l1923_192303


namespace movie_ticket_distribution_l1923_192324

theorem movie_ticket_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (n.descFactorial k) = 720 := by
  sorry

end movie_ticket_distribution_l1923_192324


namespace four_solutions_to_equation_l1923_192368

theorem four_solutions_to_equation :
  ∃! (s : Finset (ℤ × ℤ)), s.card = 4 ∧ ∀ (x y : ℤ), (x, y) ∈ s ↔ x^2020 + y^2 = 2*y :=
sorry

end four_solutions_to_equation_l1923_192368


namespace specific_divisors_of_20_pow_30_l1923_192373

def count_specific_divisors (n : ℕ) : ℕ :=
  let total_divisors := (60 + 1) * (30 + 1)
  let divisors_less_than_sqrt := (total_divisors - 1) / 2
  let divisors_of_sqrt := (30 + 1) * (15 + 1)
  divisors_less_than_sqrt - divisors_of_sqrt + 1

theorem specific_divisors_of_20_pow_30 :
  count_specific_divisors 20 = 450 := by
  sorry

end specific_divisors_of_20_pow_30_l1923_192373


namespace proposition_truth_l1923_192308

-- Define the propositions P and q
def P : Prop := ∀ x y : ℝ, x > y → -x > -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Define the compound propositions
def prop1 : Prop := P ∧ q
def prop2 : Prop := ¬P ∨ ¬q
def prop3 : Prop := P ∧ ¬q
def prop4 : Prop := ¬P ∨ q

-- Theorem statement
theorem proposition_truth : 
  ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 :=
sorry

end proposition_truth_l1923_192308


namespace cosine_sum_squared_l1923_192348

theorem cosine_sum_squared : 
  (Real.cos (42 * π / 180) + Real.cos (102 * π / 180) + 
   Real.cos (114 * π / 180) + Real.cos (174 * π / 180))^2 = 3/4 := by
  sorry

end cosine_sum_squared_l1923_192348


namespace car_wash_earnings_l1923_192374

/-- Proves that a car wash company cleaning 80 cars per day at $5 per car will earn $2000 in 5 days -/
theorem car_wash_earnings 
  (cars_per_day : ℕ) 
  (price_per_car : ℕ) 
  (num_days : ℕ) 
  (h1 : cars_per_day = 80) 
  (h2 : price_per_car = 5) 
  (h3 : num_days = 5) : 
  cars_per_day * price_per_car * num_days = 2000 := by
  sorry

#check car_wash_earnings

end car_wash_earnings_l1923_192374


namespace arithmetic_sequence_first_term_l1923_192337

theorem arithmetic_sequence_first_term 
  (a : ℚ) -- First term of the sequence
  (d : ℚ) -- Common difference of the sequence
  (h1 : (30 : ℚ) / 2 * (2 * a + 29 * d) = 450) -- Sum of first 30 terms
  (h2 : (30 : ℚ) / 2 * (2 * (a + 30 * d) + 29 * d) = 1950) -- Sum of next 30 terms
  : a = -55 / 6 := by
  sorry

end arithmetic_sequence_first_term_l1923_192337


namespace expression_simplification_l1923_192391

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (((x^2 + 1) / (x - 1) - x + 1) / ((x^2) / (1 - x))) = -Real.sqrt 2 := by
  sorry

end expression_simplification_l1923_192391


namespace solution_set_when_a_is_3_range_of_a_when_inequality_holds_l1923_192364

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≥ 2} = {x : ℝ | x ≤ 2/3 ∨ x ≥ 2} :=
by sorry

-- Part 2
theorem range_of_a_when_inequality_holds :
  (∀ x : ℝ, f a x ≥ 5 - x) → a ≥ 6 :=
by sorry

end solution_set_when_a_is_3_range_of_a_when_inequality_holds_l1923_192364


namespace brick_surface_area_is_54_l1923_192369

/-- Represents the surface areas of a brick -/
structure BrickAreas where
  front : ℝ
  side : ℝ
  top : ℝ

/-- The surface areas of the three arrangements -/
def arrangement1 (b : BrickAreas) : ℝ := 4 * b.front + 4 * b.side + 2 * b.top
def arrangement2 (b : BrickAreas) : ℝ := 4 * b.front + 2 * b.side + 4 * b.top
def arrangement3 (b : BrickAreas) : ℝ := 2 * b.front + 4 * b.side + 4 * b.top

/-- The surface area of a single brick -/
def brickSurfaceArea (b : BrickAreas) : ℝ := 2 * (b.front + b.side + b.top)

theorem brick_surface_area_is_54 (b : BrickAreas) 
  (h1 : arrangement1 b = 72)
  (h2 : arrangement2 b = 96)
  (h3 : arrangement3 b = 102) : 
  brickSurfaceArea b = 54 := by
  sorry

end brick_surface_area_is_54_l1923_192369


namespace total_length_of_sticks_l1923_192356

/-- The total length of 5 sticks with specific length relationships -/
theorem total_length_of_sticks : ∀ (stick1 stick2 stick3 stick4 stick5 : ℝ),
  stick1 = 3 →
  stick2 = 2 * stick1 →
  stick3 = stick2 - 1 →
  stick4 = stick3 / 2 →
  stick5 = 4 * stick4 →
  stick1 + stick2 + stick3 + stick4 + stick5 = 26.5 := by
  sorry

end total_length_of_sticks_l1923_192356


namespace turn_on_all_in_four_moves_l1923_192394

/-- Represents a light bulb on a 2D grid -/
structure Bulb where
  x : ℕ
  y : ℕ
  is_on : Bool

/-- Represents a line on a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the state of the grid -/
def GridState := List Bulb

/-- Checks if a bulb is on the specified side of a line -/
def is_on_side (b : Bulb) (l : Line) (positive_side : Bool) : Bool :=
  sorry

/-- Applies a move to the grid state -/
def apply_move (state : GridState) (l : Line) (positive_side : Bool) : GridState :=
  sorry

/-- Checks if all bulbs are on -/
def all_on (state : GridState) : Bool :=
  sorry

/-- Theorem: It's possible to turn on all bulbs in exactly four moves -/
theorem turn_on_all_in_four_moves :
  ∃ (moves : List (Line × Bool)),
    moves.length = 4 ∧
    let initial_state : GridState := [
      {x := 0, y := 0, is_on := false},
      {x := 0, y := 1, is_on := false},
      {x := 1, y := 0, is_on := false},
      {x := 1, y := 1, is_on := false}
    ]
    let final_state := moves.foldl (λ state move => apply_move state move.1 move.2) initial_state
    all_on final_state :=
  sorry

end turn_on_all_in_four_moves_l1923_192394


namespace lowest_unique_score_above_100_l1923_192310

/-- Represents the scoring system and conditions of the math competition. -/
structure MathCompetition where
  total_questions : Nat
  base_score : Nat
  correct_points : Nat
  wrong_points : Nat
  score : Nat

/-- Checks if a given score is valid for the math competition. -/
def is_valid_score (comp : MathCompetition) (correct wrong : Nat) : Prop :=
  correct + wrong ≤ comp.total_questions ∧
  comp.score = comp.base_score + comp.correct_points * correct - comp.wrong_points * wrong

/-- Checks if a score has a unique solution for correct and wrong answers. -/
def has_unique_solution (comp : MathCompetition) : Prop :=
  ∃! (correct wrong : Nat), is_valid_score comp correct wrong

/-- The main theorem stating that 150 is the lowest score above 100 with a unique solution. -/
theorem lowest_unique_score_above_100 : 
  let comp : MathCompetition := {
    total_questions := 50,
    base_score := 50,
    correct_points := 5,
    wrong_points := 2,
    score := 150
  }
  (comp.score > 100) ∧ 
  has_unique_solution comp ∧
  ∀ (s : Nat), 100 < s ∧ s < comp.score → 
    ¬(has_unique_solution {comp with score := s}) := by
  sorry

end lowest_unique_score_above_100_l1923_192310


namespace least_common_multiple_5_to_15_l1923_192314

theorem least_common_multiple_5_to_15 : ∃ n : ℕ, 
  (∀ k : ℕ, 5 ≤ k → k ≤ 15 → k ∣ n) ∧ 
  (∀ m : ℕ, m > 0 → (∀ k : ℕ, 5 ≤ k → k ≤ 15 → k ∣ m) → n ≤ m) ∧
  n = 360360 := by
sorry

end least_common_multiple_5_to_15_l1923_192314


namespace fourth_fifth_sum_arithmetic_l1923_192355

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fourth_fifth_sum_arithmetic (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 8 →
  a 3 = 13 →
  a 6 = 33 →
  a 7 = 38 →
  a 4 + a 5 = 41 := by
sorry

end fourth_fifth_sum_arithmetic_l1923_192355


namespace least_three_digit_multiple_l1923_192301

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → ¬((2 ∣ m) ∧ (3 ∣ m) ∧ (5 ∣ m) ∧ (7 ∣ m))) ∧
  n = 210 :=
by sorry

end least_three_digit_multiple_l1923_192301


namespace rearrangeable_shapes_exist_l1923_192357

/-- Represents a shape that can be divided and rearranged -/
structure Divisible2DShape where
  area : ℝ
  can_form_square : Bool
  can_form_triangle : Bool

/-- Represents a set of shapes that can be rearranged -/
def ShapeSet := List Divisible2DShape

/-- Function to check if a shape set can form a square -/
def can_form_square (shapes : ShapeSet) : Bool :=
  shapes.any (·.can_form_square)

/-- Function to check if a shape set can form a triangle -/
def can_form_triangle (shapes : ShapeSet) : Bool :=
  shapes.any (·.can_form_triangle)

/-- The main theorem statement -/
theorem rearrangeable_shapes_exist (a : ℝ) (h : a > 0) :
  ∃ (shapes : ShapeSet),
    -- The total area of shapes is greater than the initial square
    (shapes.map (·.area)).sum > a^2 ∧
    -- The shape set can form two different squares
    can_form_square shapes ∧
    -- The shape set can form two different triangles
    can_form_triangle shapes :=
  sorry


end rearrangeable_shapes_exist_l1923_192357


namespace max_product_constraint_max_product_achievable_l1923_192312

theorem max_product_constraint (x y : ℕ+) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 := by
  sorry

theorem max_product_achievable : ∃ (x y : ℕ+), 7 * x + 4 * y = 150 ∧ x * y = 200 := by
  sorry

end max_product_constraint_max_product_achievable_l1923_192312


namespace quadratic_inequality_always_negative_l1923_192396

theorem quadratic_inequality_always_negative : ∀ x : ℝ, -9 * x^2 + 6 * x - 8 < 0 := by
  sorry

end quadratic_inequality_always_negative_l1923_192396


namespace star_five_three_l1923_192321

/-- Define the binary operation ⋆ -/
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

/-- Theorem: When a = 5 and b = 3, a ⋆ b = 4 -/
theorem star_five_three : star 5 3 = 4 := by
  sorry

end star_five_three_l1923_192321


namespace simultaneous_hit_probability_l1923_192331

theorem simultaneous_hit_probability 
  (prob_A_hit : ℝ) 
  (prob_B_miss : ℝ) 
  (h1 : prob_A_hit = 0.8) 
  (h2 : prob_B_miss = 0.3) 
  (h3 : 0 ≤ prob_A_hit ∧ prob_A_hit ≤ 1) 
  (h4 : 0 ≤ prob_B_miss ∧ prob_B_miss ≤ 1) :
  prob_A_hit * (1 - prob_B_miss) = 14/25 := by
  sorry

end simultaneous_hit_probability_l1923_192331


namespace cookies_for_thanksgiving_l1923_192351

/-- The number of cookies Helen baked three days ago -/
def cookies_day1 : ℕ := 31

/-- The number of cookies Helen baked two days ago -/
def cookies_day2 : ℕ := 270

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_day3 : ℕ := 419

/-- The number of cookies Beaky ate from the first day's batch -/
def cookies_eaten_by_beaky : ℕ := 5

/-- The percentage of cookies that crumbled from the second day's batch -/
def crumble_percentage : ℚ := 15 / 100

/-- The number of cookies Helen gave away from the third day's batch -/
def cookies_given_away : ℕ := 30

/-- The number of cookies Helen received as a gift from Lucy -/
def cookies_gifted : ℕ := 45

/-- The total number of cookies available at Helen's house for Thanksgiving -/
def total_cookies : ℕ := 690

theorem cookies_for_thanksgiving :
  (cookies_day1 - cookies_eaten_by_beaky) +
  (cookies_day2 - Int.floor (crumble_percentage * cookies_day2)) +
  (cookies_day3 - cookies_given_away) +
  cookies_gifted = total_cookies := by
  sorry

end cookies_for_thanksgiving_l1923_192351


namespace quadratic_root_difference_l1923_192387

theorem quadratic_root_difference (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + 5*x₁ + k = 0 ∧ 
   x₂^2 + 5*x₂ + k = 0 ∧ 
   |x₁ - x₂| = 3) → 
  k = 4 := by
sorry

end quadratic_root_difference_l1923_192387


namespace percentage_difference_l1923_192334

theorem percentage_difference : 
  (0.6 * 50 + 0.45 * 30) - (0.4 * 30 + 0.25 * 20) = 26.5 := by
  sorry

end percentage_difference_l1923_192334


namespace team_cautions_l1923_192300

theorem team_cautions (total_players : ℕ) (red_cards : ℕ) (yellow_per_red : ℕ) :
  total_players = 11 →
  red_cards = 3 →
  yellow_per_red = 2 →
  ∃ (no_caution players_with_yellow : ℕ),
    no_caution + players_with_yellow = total_players ∧
    players_with_yellow = red_cards * yellow_per_red ∧
    no_caution = 5 :=
by sorry

end team_cautions_l1923_192300


namespace quadratic_sum_l1923_192363

/-- A quadratic function with vertex at (2, 5) passing through (3, 2) -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ :=
  fun x ↦ d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  (QuadraticFunction d e f 2 = 5) →
  (QuadraticFunction d e f 3 = 2) →
  d + e + 2*f = -5 := by
    sorry

end quadratic_sum_l1923_192363


namespace min_value_of_function_l1923_192341

theorem min_value_of_function (x : ℝ) (h : x ≥ 0) :
  x + 1 / (x + 1) ≥ 1 ∧ ∃ y : ℝ, y ≥ 0 ∧ y + 1 / (y + 1) = 1 := by
  sorry

end min_value_of_function_l1923_192341


namespace problem_proof_l1923_192345

theorem problem_proof : 
  (14^2 * 5^3) / 568 = 43.13380281690141 := by sorry

end problem_proof_l1923_192345


namespace simplest_quadratic_radical_l1923_192338

/-- A quadratic radical is considered simpler if it cannot be further simplified to a non-radical form or a simpler radical form. -/
def IsSimplestQuadraticRadical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≠ x → (∃ z : ℝ, y = z ^ 2) → ¬(∃ w : ℝ, x = w ^ 2)

/-- The given options for quadratic radicals -/
def QuadraticRadicals (a b : ℝ) : Set ℝ :=
  {Real.sqrt 9, Real.sqrt (a^2 + b^2), Real.sqrt 0.7, Real.sqrt (a^3)}

/-- Theorem stating that √(a² + b²) is the simplest quadratic radical among the given options -/
theorem simplest_quadratic_radical (a b : ℝ) :
  ∀ x ∈ QuadraticRadicals a b, x = Real.sqrt (a^2 + b^2) ∨ ¬(IsSimplestQuadraticRadical x) :=
sorry

end simplest_quadratic_radical_l1923_192338


namespace quadratic_factorization_l1923_192326

theorem quadratic_factorization (y : ℝ) : 9*y^2 - 30*y + 25 = (3*y - 5)^2 := by
  sorry

end quadratic_factorization_l1923_192326


namespace total_candles_l1923_192330

theorem total_candles (bedroom : ℕ) (living_room : ℕ) (donovan : ℕ) : 
  bedroom = 20 →
  bedroom = 2 * living_room →
  donovan = 20 →
  bedroom + living_room + donovan = 50 := by
sorry

end total_candles_l1923_192330


namespace second_hour_billboards_l1923_192384

/-- The number of billboards counted in the second hour -/
def billboards_second_hour (first_hour : ℕ) (third_hour : ℕ) (total_hours : ℕ) (average : ℕ) : ℕ :=
  average * total_hours - first_hour - third_hour

theorem second_hour_billboards :
  billboards_second_hour 17 23 3 20 = 20 := by
  sorry

#eval billboards_second_hour 17 23 3 20

end second_hour_billboards_l1923_192384


namespace a_oxen_count_l1923_192376

/-- Represents the number of oxen and months for each person renting the pasture -/
structure Grazing where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent based on oxen and months -/
def calculateShare (g : Grazing) (totalRent : ℚ) (totalProduct : ℕ) : ℚ :=
  totalRent * (g.oxen * g.months : ℚ) / totalProduct

theorem a_oxen_count (a : Grazing) (b c : Grazing) 
    (h1 : b.oxen = 12 ∧ b.months = 5)
    (h2 : c.oxen = 15 ∧ c.months = 3)
    (h3 : a.months = 7)
    (h4 : calculateShare c 245 (a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) = 63) :
  a.oxen = 17 := by
  sorry

#check a_oxen_count

end a_oxen_count_l1923_192376


namespace sphere_surface_area_of_inscribed_parallelepiped_l1923_192319

/-- The surface area of a sphere that circumscribes a rectangular parallelepiped with edge lengths 3, 4, and 5 is equal to 50π. -/
theorem sphere_surface_area_of_inscribed_parallelepiped (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diameter := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diameter / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 50 * Real.pi := by
  sorry

end sphere_surface_area_of_inscribed_parallelepiped_l1923_192319


namespace center_number_is_five_l1923_192343

-- Define the 2x3 array type
def Array2x3 := Fin 2 → Fin 3 → Nat

-- Define a predicate for consecutive numbers
def Consecutive (a b : Nat) : Prop := a + 1 = b ∨ b + 1 = a

-- Define diagonal adjacency
def DiagonallyAdjacent (i1 j1 i2 j2 : Nat) : Prop :=
  (i1 + 1 = i2 ∧ j1 + 1 = j2) ∨ (i1 + 1 = i2 ∧ j1 = j2 + 1) ∨
  (i1 = i2 + 1 ∧ j1 + 1 = j2) ∨ (i1 = i2 + 1 ∧ j1 = j2 + 1)

-- Define the property of consecutive numbers being diagonally adjacent
def ConsecutiveAreDiagonallyAdjacent (arr : Array2x3) : Prop :=
  ∀ i1 j1 i2 j2, Consecutive (arr i1 j1) (arr i2 j2) → DiagonallyAdjacent i1 j1 i2 j2

-- Define the property that all numbers from 1 to 5 are present
def ContainsAllNumbers (arr : Array2x3) : Prop :=
  ∀ n, n ≥ 1 ∧ n ≤ 5 → ∃ i j, arr i j = n

-- Define the property that corner numbers on one long side sum to 6
def CornersSum6 (arr : Array2x3) : Prop :=
  (arr 0 0 + arr 0 2 = 6) ∨ (arr 1 0 + arr 1 2 = 6)

-- The main theorem
theorem center_number_is_five (arr : Array2x3) 
  (h1 : ConsecutiveAreDiagonallyAdjacent arr)
  (h2 : ContainsAllNumbers arr)
  (h3 : CornersSum6 arr) :
  (arr 0 1 = 5) ∨ (arr 1 1 = 5) :=
sorry

end center_number_is_five_l1923_192343


namespace yoongi_age_l1923_192329

theorem yoongi_age (hoseok_age yoongi_age : ℕ) 
  (h1 : yoongi_age = hoseok_age - 2)
  (h2 : yoongi_age + hoseok_age = 18) :
  yoongi_age = 8 := by
  sorry

end yoongi_age_l1923_192329


namespace sum_of_digits_8_pow_1502_l1923_192347

/-- The sum of the tens digit and the units digit in the decimal representation of 8^1502 is 10 -/
theorem sum_of_digits_8_pow_1502 : ∃ n : ℕ, 8^1502 = 100 * n + 64 := by
  sorry

end sum_of_digits_8_pow_1502_l1923_192347


namespace divisibility_by_27_l1923_192385

theorem divisibility_by_27 (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) :
  27 ∣ (x + y + z) := by
  sorry

end divisibility_by_27_l1923_192385


namespace functional_equation_solution_l1923_192379

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + x*y) = f x * f y + y * f x + x * f (x + y)) :
  (∀ x : ℝ, f x = 1 - x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end functional_equation_solution_l1923_192379


namespace infinite_product_equals_nine_l1923_192392

/-- The series S is defined as the sum 1/2 + 2/4 + 3/8 + 4/16 + ... -/
def S : ℝ := 2

/-- The infinite product P is defined as 3^(1/2) * 9^(1/4) * 27^(1/8) * 81^(1/16) * ... -/
noncomputable def P : ℝ := Real.rpow 3 S

theorem infinite_product_equals_nine : P = 9 := by sorry

end infinite_product_equals_nine_l1923_192392


namespace min_days_for_eleven_groups_l1923_192366

/-- Represents a festival schedule --/
structure FestivalSchedule where
  days : ℕ
  groups : ℕ
  performing : Fin days → Finset (Fin groups)
  watching : Fin days → Finset (Fin groups)

/-- Checks if a festival schedule is valid --/
def isValidSchedule (s : FestivalSchedule) : Prop :=
  (∀ d, s.performing d ∩ s.watching d = ∅) ∧
  (∀ d, s.performing d ∪ s.watching d = Finset.univ) ∧
  (∀ g₁ g₂, g₁ ≠ g₂ → ∃ d, g₁ ∈ s.watching d ∧ g₂ ∈ s.performing d)

/-- The main theorem --/
theorem min_days_for_eleven_groups :
  ∃ (s : FestivalSchedule), s.groups = 11 ∧ s.days = 6 ∧ isValidSchedule s ∧
  ∀ (s' : FestivalSchedule), s'.groups = 11 → isValidSchedule s' → s'.days ≥ 6 := by
  sorry


end min_days_for_eleven_groups_l1923_192366


namespace expression_simplification_l1923_192318

theorem expression_simplification (y : ℝ) : 
  3 * y + 4 * y^2 - 2 - (7 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 9 := by
  sorry

end expression_simplification_l1923_192318


namespace negation_of_inequality_l1923_192386

theorem negation_of_inequality (x : Real) : 
  (¬ ∀ x ∈ Set.Ioo 0 (π/2), x > Real.sin x) ↔ 
  (∃ x ∈ Set.Ioo 0 (π/2), x ≤ Real.sin x) := by
sorry

end negation_of_inequality_l1923_192386


namespace solve_for_B_l1923_192361

theorem solve_for_B : ∃ B : ℝ, (4 * B + 4 - 3 = 29) ∧ (B = 7) := by
  sorry

end solve_for_B_l1923_192361


namespace additional_income_needed_l1923_192395

/-- Calculate the additional annual income needed to reach a target amount after expenses --/
theorem additional_income_needed
  (current_income : ℝ)
  (rent : ℝ)
  (groceries : ℝ)
  (gas : ℝ)
  (target_amount : ℝ)
  (h1 : current_income = 65000)
  (h2 : rent = 20000)
  (h3 : groceries = 5000)
  (h4 : gas = 8000)
  (h5 : target_amount = 42000) :
  current_income + 10000 - (rent + groceries + gas) ≥ target_amount ∧
  ∀ x : ℝ, x < 10000 → current_income + x - (rent + groceries + gas) < target_amount :=
by
  sorry

#check additional_income_needed

end additional_income_needed_l1923_192395


namespace matching_color_probability_l1923_192311

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ

/-- Calculates the total number of jelly beans a person has -/
def total_jelly_beans (jb : JellyBeans) : ℕ := jb.green + jb.red + jb.blue

/-- Abe's jelly beans -/
def abe_jb : JellyBeans := { green := 2, red := 3, blue := 0 }

/-- Bob's jelly beans -/
def bob_jb : JellyBeans := { green := 2, red := 3, blue := 2 }

/-- Calculates the probability of picking a specific color -/
def prob_color (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / (total_jelly_beans jb)

/-- Theorem: The probability of Abe and Bob showing the same color is 13/35 -/
theorem matching_color_probability : 
  (prob_color abe_jb abe_jb.green * prob_color bob_jb bob_jb.green) +
  (prob_color abe_jb abe_jb.red * prob_color bob_jb bob_jb.red) = 13/35 := by
  sorry

end matching_color_probability_l1923_192311


namespace normal_dist_two_std_dev_below_mean_l1923_192365

/-- For a normal distribution with mean μ and standard deviation σ,
    the value that is exactly 2 standard deviations less than the mean
    is equal to μ - 2σ. -/
theorem normal_dist_two_std_dev_below_mean (μ σ : ℝ) :
  let value := μ - 2 * σ
  μ = 16.5 → σ = 1.5 → value = 13.5 := by sorry

end normal_dist_two_std_dev_below_mean_l1923_192365


namespace local_minimum_implies_b_range_l1923_192352

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

-- State the theorem
theorem local_minimum_implies_b_range :
  ∀ b : ℝ, (∃ c ∈ Set.Ioo 0 1, IsLocalMin (f b) c) → 0 < b ∧ b < 1 := by
  sorry

end local_minimum_implies_b_range_l1923_192352


namespace digit_sum_proof_l1923_192336

theorem digit_sum_proof (A B : ℕ) :
  A ≤ 9 ∧ B ≤ 9 ∧ 
  111 * A + 110 * A + B + 100 * A + 11 * B + 111 * B = 1503 →
  A = 2 ∧ B = 7 := by
sorry

end digit_sum_proof_l1923_192336


namespace license_plate_difference_l1923_192382

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible Sunshine license plates -/
def sunshine_plates : ℕ := num_letters^3 * num_digits^3

/-- The number of possible Prairie license plates -/
def prairie_plates : ℕ := num_letters^2 * num_digits^4

/-- The difference in the number of possible license plates between Sunshine and Prairie -/
def plate_difference : ℕ := sunshine_plates - prairie_plates

theorem license_plate_difference :
  plate_difference = 10816000 := by
  sorry

end license_plate_difference_l1923_192382


namespace real_part_of_complex_number_l1923_192317

theorem real_part_of_complex_number (z : ℂ) (a : ℝ) :
  z = (1 : ℂ) / (1 + I) + a * I → z.im = 0 → a = (1 : ℝ) / 2 := by
  sorry

end real_part_of_complex_number_l1923_192317


namespace f_properties_l1923_192323

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + x^2

theorem f_properties (a : ℝ) :
  (∀ x > 1, Monotone (f (-2)))
  ∧ (∀ x ∈ Set.Icc 1 (exp 1), f a x ≥ 
      (if a ≥ -2 then 1
       else if a > -2 * (exp 1)^2 then a/2 * log (-a/2) - a/2
       else a + (exp 1)^2))
  ∧ (∃ x ∈ Set.Icc 1 (exp 1), f a x = 
      (if a ≥ -2 then 1
       else if a > -2 * (exp 1)^2 then a/2 * log (-a/2) - a/2
       else a + (exp 1)^2))
  ∧ (∃ x ∈ Set.Icc 1 (exp 1), f a x = 
      (if a ≥ -2 then f a 1
       else if a > -2 * (exp 1)^2 then f a (sqrt (-a/2))
       else f a (exp 1))) :=
by sorry

end f_properties_l1923_192323


namespace x_equation_implies_y_values_l1923_192307

theorem x_equation_implies_y_values (x : ℝ) :
  x^2 + 9 * (x / (x - 3))^2 = 54 →
  let y := ((x - 3)^2 * (x + 4)) / (2 * x - 4)
  y = 11.25 ∨ y = 10.125 := by
sorry

end x_equation_implies_y_values_l1923_192307


namespace range_of_a_l1923_192353

def linear_function (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x + a - 8

def fractional_equation (a : ℝ) (y : ℝ) : Prop :=
  (y - 5) / (1 - y) + 3 = a / (y - 1)

theorem range_of_a (a : ℝ) :
  (∀ x y, y = linear_function a x → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) →
  (∀ y, fractional_equation a y → y > -3) →
  1 < a ∧ a < 8 ∧ a ≠ 4 :=
sorry

end range_of_a_l1923_192353


namespace central_symmetry_line_symmetry_two_lines_max_distance_l1923_192327

-- Define the curve C
def C (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + a * p.1 * p.2 = 1}

-- Statement 1: C is centrally symmetric about the origin for all a
theorem central_symmetry (a : ℝ) : ∀ p : ℝ × ℝ, p ∈ C a → (-p.1, -p.2) ∈ C a := by sorry

-- Statement 2: C is symmetric about the lines y = x and y = -x for all a
theorem line_symmetry (a : ℝ) : 
  (∀ p : ℝ × ℝ, p ∈ C a → (p.2, p.1) ∈ C a) ∧ 
  (∀ p : ℝ × ℝ, p ∈ C a → (-p.2, -p.1) ∈ C a) := by sorry

-- Statement 3: There exist at least two distinct values of a for which C represents two lines
theorem two_lines : ∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧ 
  (∃ l₁ l₂ m₁ m₂ : ℝ → ℝ, C a₁ = {p : ℝ × ℝ | p.2 = l₁ p.1 ∨ p.2 = l₂ p.1} ∧ 
                          C a₂ = {p : ℝ × ℝ | p.2 = m₁ p.1 ∨ p.2 = m₂ p.1}) := by sorry

-- Statement 4: When a = 1, the maximum distance between any two points on C is 2√2
theorem max_distance : 
  (∀ p q : ℝ × ℝ, p ∈ C 1 → q ∈ C 1 → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ 2 * Real.sqrt 2) ∧
  (∃ p q : ℝ × ℝ, p ∈ C 1 ∧ q ∈ C 1 ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 2) := by sorry

end central_symmetry_line_symmetry_two_lines_max_distance_l1923_192327


namespace simplify_expression_l1923_192340

theorem simplify_expression (x : ℝ) (h : x < 0) :
  (2 * abs x + (x^6)^(1/6) + (x^5)^(1/5)) / x = -2 := by sorry

end simplify_expression_l1923_192340


namespace exists_valid_partition_l1923_192399

/-- A directed graph where each vertex has outdegree 2 -/
structure Graph (V : Type*) :=
  (edges : V → Finset V)
  (outdegree_two : ∀ v : V, (edges v).card = 2)

/-- A partition of vertices into three sets -/
def Partition (V : Type*) := V → Fin 3

/-- The main theorem statement -/
theorem exists_valid_partition {V : Type*} [Fintype V] (G : Graph V) :
  ∃ (p : Partition V), ∀ (v : V),
    ∃ (w : V), w ∈ G.edges v ∧ p w ≠ p v :=
sorry

end exists_valid_partition_l1923_192399


namespace lucas_sticker_redistribution_l1923_192390

theorem lucas_sticker_redistribution
  (n : ℚ)  -- Noah's initial number of stickers
  (h1 : n > 0)  -- Ensure n is positive
  (emma : ℚ)  -- Emma's initial number of stickers
  (h2 : emma = 3 * n)  -- Emma has 3 times as many stickers as Noah
  (lucas : ℚ)  -- Lucas's initial number of stickers
  (h3 : lucas = 4 * emma)  -- Lucas has 4 times as many stickers as Emma
  : (lucas - (lucas + emma + n) / 3) / lucas = 7 / 36 := by
  sorry

end lucas_sticker_redistribution_l1923_192390


namespace count_non_multiples_is_675_l1923_192339

/-- The count of three-digit numbers that are not multiples of 6 or 8 -/
def count_non_multiples : ℕ :=
  let total_three_digit_numbers := 999 - 100 + 1
  let multiples_of_6 := (999 / 6) - (99 / 6)
  let multiples_of_8 := (999 / 8) - (99 / 8)
  let multiples_of_24 := (999 / 24) - (99 / 24)
  total_three_digit_numbers - (multiples_of_6 + multiples_of_8 - multiples_of_24)

theorem count_non_multiples_is_675 : count_non_multiples = 675 := by
  sorry

end count_non_multiples_is_675_l1923_192339


namespace toothpick_20th_stage_l1923_192383

def toothpick_sequence (n : ℕ) : ℕ := 5 + 3 * (n - 1)

theorem toothpick_20th_stage :
  toothpick_sequence 20 = 62 := by
sorry

end toothpick_20th_stage_l1923_192383


namespace geometric_sequence_fourth_term_l1923_192346

/-- A geometric sequence with first term 1024 and sixth term 125 has its fourth term equal to 2000 -/
theorem geometric_sequence_fourth_term : ∀ (a : ℕ → ℝ), 
  (∃ r : ℝ, ∀ n : ℕ, a n = 1024 * r ^ (n - 1)) →  -- Geometric sequence definition
  a 1 = 1024 →                                   -- First term condition
  a 6 = 125 →                                    -- Sixth term condition
  a 4 = 2000 :=                                  -- Fourth term (to prove)
by
  sorry

end geometric_sequence_fourth_term_l1923_192346


namespace bob_net_increase_theorem_l1923_192322

/-- Calculates the net increase in weekly earnings given a raise, work hours, and benefit reduction --/
def netIncreaseInWeeklyEarnings (hourlyRaise : ℚ) (weeklyHours : ℕ) (monthlyBenefitReduction : ℚ) : ℚ :=
  let weeklyRaise := hourlyRaise * weeklyHours
  let weeklyBenefitReduction := monthlyBenefitReduction / 4
  weeklyRaise - weeklyBenefitReduction

/-- Theorem stating that given the specified conditions, the net increase in weekly earnings is $5 --/
theorem bob_net_increase_theorem :
  netIncreaseInWeeklyEarnings (1/2) 40 60 = 5 := by
  sorry

end bob_net_increase_theorem_l1923_192322


namespace positive_real_inequalities_l1923_192360

theorem positive_real_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2) ∧
  (a^3 + b^3 + c^3 + 1/a + 1/b + 1/c ≥ 2 * (a + b + c)) := by
  sorry

end positive_real_inequalities_l1923_192360


namespace positive_integer_sum_greater_than_product_l1923_192358

theorem positive_integer_sum_greater_than_product (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  m + n > m * n ↔ m = 1 ∨ n = 1 := by
sorry

end positive_integer_sum_greater_than_product_l1923_192358


namespace min_exponent_sum_l1923_192370

theorem min_exponent_sum (A : ℕ+) (α β γ : ℕ) 
  (h1 : A = 2^α * 3^β * 5^γ)
  (h2 : ∃ (k : ℕ), A / 2 = k^2)
  (h3 : ∃ (m : ℕ), A / 3 = m^3)
  (h4 : ∃ (n : ℕ), A / 5 = n^5) :
  α + β + γ ≥ 31 :=
sorry

end min_exponent_sum_l1923_192370


namespace parallel_vectors_x_value_l1923_192332

/-- Two planar vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 1)
  let b : ℝ × ℝ := (x, 2)
  are_parallel a b → x = 8 :=
by sorry

end parallel_vectors_x_value_l1923_192332


namespace simplify_fraction_l1923_192342

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end simplify_fraction_l1923_192342


namespace function_transformation_l1923_192398

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_transformation (x : ℝ) : f (x + 1) = 3 * x + 2 → f x = 3 * x - 1 := by
  sorry

end function_transformation_l1923_192398


namespace equal_ratios_sum_l1923_192350

theorem equal_ratios_sum (P Q : ℚ) :
  (4 : ℚ) / 9 = P / 63 ∧ (4 : ℚ) / 9 = 108 / Q → P + Q = 271 := by
  sorry

end equal_ratios_sum_l1923_192350


namespace bob_yogurt_order_l1923_192367

-- Define the problem parameters
def expired_percentage : ℚ := 40 / 100
def pack_cost : ℚ := 12
def total_refund : ℚ := 384

-- State the theorem
theorem bob_yogurt_order :
  ∃ (total_packs : ℚ),
    total_packs * expired_percentage * pack_cost = total_refund ∧
    total_packs = 80 := by
  sorry

end bob_yogurt_order_l1923_192367


namespace smallest_sum_a_b_l1923_192302

theorem smallest_sum_a_b (a b : ℕ+) 
  (h : (1 : ℚ) / a + (1 : ℚ) / (2 * a) + (1 : ℚ) / (3 * a) = (1 : ℚ) / (b^2 - 2*b)) : 
  ∀ (x y : ℕ+), 
    ((1 : ℚ) / x + (1 : ℚ) / (2 * x) + (1 : ℚ) / (3 * x) = (1 : ℚ) / (y^2 - 2*y)) → 
    (x + y : ℕ) ≥ (a + b : ℕ) ∧ (a + b : ℕ) = 50 :=
by sorry

end smallest_sum_a_b_l1923_192302


namespace april_roses_problem_l1923_192377

theorem april_roses_problem (price : ℕ) (leftover : ℕ) (earnings : ℕ) (initial : ℕ) : 
  price = 7 → 
  leftover = 4 → 
  earnings = 35 → 
  price * (initial - leftover) = earnings → 
  initial = 9 := by
sorry

end april_roses_problem_l1923_192377


namespace relation_between_x_and_y_l1923_192381

theorem relation_between_x_and_y (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end relation_between_x_and_y_l1923_192381


namespace dinner_bill_proof_l1923_192325

/-- The number of friends who went to dinner -/
def total_friends : ℕ := 10

/-- The number of friends who paid -/
def paying_friends : ℕ := 9

/-- The extra amount each paying friend contributed -/
def extra_payment : ℚ := 3

/-- The discount rate applied to the bill -/
def discount_rate : ℚ := 1/10

/-- The original bill before discount -/
def original_bill : ℚ := 300

theorem dinner_bill_proof :
  let discounted_bill := original_bill * (1 - discount_rate)
  let individual_share := discounted_bill / total_friends
  paying_friends * (individual_share + extra_payment) = discounted_bill :=
by sorry

end dinner_bill_proof_l1923_192325


namespace greatest_common_measure_l1923_192371

theorem greatest_common_measure (a b c : ℕ) (ha : a = 700) (hb : b = 385) (hc : c = 1295) :
  Nat.gcd a (Nat.gcd b c) = 35 := by
  sorry

end greatest_common_measure_l1923_192371


namespace chris_breath_holding_start_l1923_192354

def breath_holding_progression (start : ℕ) (days : ℕ) : ℕ :=
  start + 10 * (days - 1)

theorem chris_breath_holding_start :
  ∃ (start : ℕ),
    breath_holding_progression start 2 = 20 ∧
    breath_holding_progression start 6 = 90 ∧
    start = 10 := by
  sorry

end chris_breath_holding_start_l1923_192354


namespace max_individual_score_l1923_192335

theorem max_individual_score (n : ℕ) (total_score : ℕ) (min_score : ℕ) 
  (h1 : n = 12)
  (h2 : total_score = 100)
  (h3 : min_score = 7)
  (h4 : ∀ player, player ∈ Finset.range n → player ≥ min_score) :
  (total_score - (n - 1) * min_score) = 23 := by
  sorry

end max_individual_score_l1923_192335


namespace semester_days_l1923_192393

/-- Calculates the number of days given daily distance and total distance -/
def calculate_days (daily_distance : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / daily_distance

/-- Theorem stating that given the specific conditions, the number of days is 160 -/
theorem semester_days : calculate_days 10 1600 = 160 := by
  sorry

end semester_days_l1923_192393


namespace ivan_travel_theorem_l1923_192344

/-- Represents the travel scenario of Ivan Semenovich -/
structure TravelScenario where
  usual_travel_time : ℝ
  usual_arrival_time : ℝ
  late_departure : ℝ
  speed_increase : ℝ
  new_arrival_time : ℝ

/-- The theorem to be proved -/
theorem ivan_travel_theorem (scenario : TravelScenario) 
  (h1 : scenario.usual_arrival_time = 9 * 60)  -- 9:00 AM in minutes
  (h2 : scenario.late_departure = 40)
  (h3 : scenario.speed_increase = 0.6)
  (h4 : scenario.new_arrival_time = 8 * 60 + 35)  -- 8:35 AM in minutes
  : ∃ (optimal_increase : ℝ),
    optimal_increase = 0.3 ∧
    scenario.usual_arrival_time = 
      scenario.usual_travel_time * (1 - scenario.late_departure / scenario.usual_travel_time) / (1 + optimal_increase) + 
      scenario.late_departure :=
by sorry

end ivan_travel_theorem_l1923_192344


namespace alternating_sequences_20_l1923_192328

/-- A function that computes the number of alternating sequences -/
def A : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => A (n + 1) + A n

/-- The number of alternating sequences for n = 20 is 10946 -/
theorem alternating_sequences_20 : A 20 = 10946 := by
  sorry

end alternating_sequences_20_l1923_192328


namespace pigeon_count_l1923_192305

/-- The number of pigeons in the pigeon house -/
def num_pigeons : ℕ := 600

/-- The number of days the feed lasts if 75 pigeons are sold -/
def days_after_selling : ℕ := 20

/-- The number of days the feed lasts if 100 pigeons are bought -/
def days_after_buying : ℕ := 15

/-- The number of pigeons sold -/
def pigeons_sold : ℕ := 75

/-- The number of pigeons bought -/
def pigeons_bought : ℕ := 100

/-- Theorem stating that the number of pigeons in the pigeon house is 600 -/
theorem pigeon_count : 
  (num_pigeons - pigeons_sold) * days_after_selling = (num_pigeons + pigeons_bought) * days_after_buying :=
by sorry

end pigeon_count_l1923_192305


namespace series_sum_l1923_192375

/-- The positive real solution to x³ + (1/4)x - 1 = 0 -/
noncomputable def s : ℝ := sorry

/-- The infinite series s³ + 2s⁷ + 3s¹¹ + 4s¹⁵ + ... -/
noncomputable def T : ℝ := sorry

/-- s is a solution to the equation x³ + (1/4)x - 1 = 0 -/
axiom s_def : s^3 + (1/4) * s - 1 = 0

/-- s is positive -/
axiom s_pos : s > 0

/-- T is equal to the infinite series s³ + 2s⁷ + 3s¹¹ + 4s¹⁵ + ... -/
axiom T_def : T = s^3 + 2*s^7 + 3*s^11 + 4*s^15 + sorry

theorem series_sum : T = 16 * s := by sorry

end series_sum_l1923_192375


namespace interest_calculation_years_l1923_192309

theorem interest_calculation_years (P r : ℝ) (h1 : P = 625) (h2 : r = 0.04) : 
  ∃ n : ℕ, n = 2 ∧ P * ((1 + r)^n - 1) - P * r * n = 1 := by
  sorry

end interest_calculation_years_l1923_192309


namespace prop_1_prop_2_prop_3_l1923_192362

-- Define the function f(x)
def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

-- Theorem for proposition ①
theorem prop_1 (b : ℝ) : 
  ∀ x, f x b 0 = -f (-x) b 0 := by sorry

-- Theorem for proposition ②
theorem prop_2 (c : ℝ) (h : c > 0) :
  ∃! x, f x 0 c = 0 := by sorry

-- Theorem for proposition ③
theorem prop_3 (b c : ℝ) :
  ∀ x, f x b c = f (-x) b c + 2 * c := by sorry

end prop_1_prop_2_prop_3_l1923_192362
