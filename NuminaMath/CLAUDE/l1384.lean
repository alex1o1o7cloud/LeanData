import Mathlib

namespace inequality_solution_set_l1384_138437

def f (x : ℝ) : ℝ := x * abs (x - 2)

theorem inequality_solution_set (x : ℝ) :
  f (Real.sqrt 2 - x) ≤ f 1 ↔ x ≥ -1 := by sorry

end inequality_solution_set_l1384_138437


namespace smallest_and_largest_with_digit_sum_17_l1384_138488

def digit_sum (n : ℕ) : ℕ := sorry

def all_digits_different (n : ℕ) : Prop := sorry

theorem smallest_and_largest_with_digit_sum_17 :
  ∃ (smallest largest : ℕ),
    (∀ n : ℕ, digit_sum n = 17 → all_digits_different n →
      smallest ≤ n ∧ n ≤ largest) ∧
    digit_sum smallest = 17 ∧
    all_digits_different smallest ∧
    digit_sum largest = 17 ∧
    all_digits_different largest ∧
    smallest = 89 ∧
    largest = 743210 :=
sorry

end smallest_and_largest_with_digit_sum_17_l1384_138488


namespace min_value_of_expression_l1384_138442

theorem min_value_of_expression (a b : ℕ) (ha : 0 < a ∧ a < 9) (hb : 0 < b ∧ b < 9) :
  ∃ (m : ℤ), m = -5 ∧ ∀ (x y : ℕ), (0 < x ∧ x < 9) → (0 < y ∧ y < 9) → m ≤ (3 * x^2 - x * y : ℤ) :=
sorry

end min_value_of_expression_l1384_138442


namespace unique_number_l1384_138474

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  digit_sum n = 12 ∧ 
  reverse_digits (n + 36) = n :=
by sorry

end unique_number_l1384_138474


namespace lcm_gcd_product_l1384_138445

theorem lcm_gcd_product : Nat.lcm 6 (Nat.lcm 8 12) * Nat.gcd 6 (Nat.gcd 8 12) = 48 := by
  sorry

end lcm_gcd_product_l1384_138445


namespace largest_number_l1384_138461

theorem largest_number (S : Set ℝ) (hS : S = {1/2, 0, 1, -9}) : 
  ∃ m ∈ S, ∀ x ∈ S, x ≤ m ∧ m = 1 :=
sorry

end largest_number_l1384_138461


namespace johns_drive_speed_l1384_138483

/-- Proves that given the conditions of John's drive, his average speed during the last 40 minutes was 70 mph -/
theorem johns_drive_speed (total_distance : ℝ) (total_time : ℝ) (speed_first_40 : ℝ) (speed_next_40 : ℝ)
  (h1 : total_distance = 120)
  (h2 : total_time = 2)
  (h3 : speed_first_40 = 50)
  (h4 : speed_next_40 = 60) :
  let time_segment := total_time / 3
  let distance_first_40 := speed_first_40 * time_segment
  let distance_next_40 := speed_next_40 * time_segment
  let distance_last_40 := total_distance - (distance_first_40 + distance_next_40)
  distance_last_40 / time_segment = 70 := by
  sorry

end johns_drive_speed_l1384_138483


namespace sally_earnings_l1384_138490

def earnings_per_house : ℕ := 25
def houses_cleaned : ℕ := 96

theorem sally_earnings :
  (earnings_per_house * houses_cleaned) / 12 = 200 := by
  sorry

end sally_earnings_l1384_138490


namespace square_garden_area_l1384_138479

theorem square_garden_area (p : ℝ) (s : ℝ) : 
  p = 28 →                   -- The perimeter is 28 feet
  p = 4 * s →                -- Perimeter of a square is 4 times the side length
  s^2 = p + 21 →             -- Area is equal to perimeter plus 21
  s^2 = 49 :=                -- The area of the garden is 49 square feet
by
  sorry

end square_garden_area_l1384_138479


namespace square_root_of_four_l1384_138481

theorem square_root_of_four :
  {y : ℝ | y^2 = 4} = {2, -2} := by sorry

end square_root_of_four_l1384_138481


namespace hotdog_ratio_l1384_138482

/-- Represents the number of hotdogs for each person -/
structure Hotdogs where
  ella : ℕ
  emma : ℕ
  luke : ℕ
  hunter : ℕ

/-- Given conditions for the hotdog problem -/
def hotdog_problem (h : Hotdogs) : Prop :=
  h.ella = 2 ∧
  h.emma = 2 ∧
  h.luke = 2 * (h.ella + h.emma) ∧
  h.ella + h.emma + h.luke + h.hunter = 14

/-- Theorem stating the ratio of Hunter's hotdogs to his sisters' total hotdogs -/
theorem hotdog_ratio (h : Hotdogs) (hcond : hotdog_problem h) :
  h.hunter / (h.ella + h.emma) = 1 / 2 :=
by sorry

end hotdog_ratio_l1384_138482


namespace ellipse_and_line_properties_l1384_138491

-- Define the ellipse C
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define the line l
structure Line where
  k : ℝ
  m : ℝ

-- Define the properties of the ellipse
def is_valid_ellipse (C : Ellipse) : Prop :=
  C.a = 2 ∧ C.b^2 = 3 ∧ C.a > C.b ∧ C.b > 0

-- Define the intersection of line and ellipse
def line_intersects_ellipse (l : Line) (C : Ellipse) : Prop :=
  ∃ x y, (x^2 / C.a^2) + (y^2 / C.b^2) = 1 ∧ y = l.k * x + l.m

-- Define the condition for the circle passing through origin
def circle_passes_through_origin (l : Line) (C : Ellipse) : Prop :=
  ∃ x₁ y₁ x₂ y₂, 
    line_intersects_ellipse l C ∧
    x₁ * x₂ + y₁ * y₂ = 0 ∧
    y₁ = l.k * x₁ + l.m ∧
    y₂ = l.k * x₂ + l.m

-- Main theorem
theorem ellipse_and_line_properties (C : Ellipse) (l : Line) :
  is_valid_ellipse C →
  circle_passes_through_origin l C →
  (C.a^2 = 4 ∧ C.b^2 = 3) ∧
  (l.m < -2 * Real.sqrt 21 / 7 ∨ l.m > 2 * Real.sqrt 21 / 7) :=
sorry

end ellipse_and_line_properties_l1384_138491


namespace park_area_l1384_138408

/-- Represents a rectangular park with given properties -/
structure RectangularPark where
  length : ℝ
  breadth : ℝ
  ratio : length / breadth = 1 / 3
  perimeter : length * 2 + breadth * 2 = 1600

/-- The area of the rectangular park is 120000 square meters -/
theorem park_area (park : RectangularPark) : park.length * park.breadth = 120000 := by
  sorry

end park_area_l1384_138408


namespace trigonometric_expression_equality_vector_expression_equality_l1384_138454

-- Part 1
theorem trigonometric_expression_equality :
  Real.cos (25 * Real.pi / 3) + Real.tan (-15 * Real.pi / 4) = 3/2 := by sorry

-- Part 2
theorem vector_expression_equality {n : Type*} [NormedAddCommGroup n] :
  ∀ (a b : n), 2 • (a - b) - (2 • a + b) + 3 • b = 0 := by sorry

end trigonometric_expression_equality_vector_expression_equality_l1384_138454


namespace triangle_theorem_l1384_138424

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angles : A > 0 ∧ B > 0 ∧ C > 0
  h_angle_sum : A + B + C = π
  h_area : S = (1/2) * b * c * Real.sin A

-- Define the main theorem
theorem triangle_theorem (t : Triangle) :
  (3 * t.a^2 - 4 * Real.sqrt 3 * t.S = 3 * t.b^2 + 3 * t.c^2) →
  (t.A = 2 * π / 3) ∧
  (t.a = 3 → 6 < t.a + t.b + t.c ∧ t.a + t.b + t.c < 3 + 2 * Real.sqrt 3) :=
by sorry

end triangle_theorem_l1384_138424


namespace rectangle_perimeter_l1384_138444

/-- Given a rectangle with specific properties, prove its perimeter is 92 cm -/
theorem rectangle_perimeter (width length : ℕ) : 
  width = 34 ∧ 
  width % 4 = 2 ∧ 
  (width / 4) * length = 24 → 
  2 * (width + length) = 92 := by
  sorry

end rectangle_perimeter_l1384_138444


namespace range_of_a_l1384_138473

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B (a : ℝ) : Set ℝ := {x | (2*x - a) / (x + 1) > 1}

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (A ⊂ B a ∧ A ≠ B a) → a < 1 :=
by sorry

end range_of_a_l1384_138473


namespace min_value_sum_and_reciprocal_l1384_138407

theorem min_value_sum_and_reciprocal (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (((a^2 + b^2 : ℚ) / (a * b)) + ((a * b : ℚ) / (a^2 + b^2))) ≥ 2 ∧
  ∃ (a' b' : ℤ), a' ≠ 0 ∧ b' ≠ 0 ∧ (((a'^2 + b'^2 : ℚ) / (a' * b')) + ((a' * b' : ℚ) / (a'^2 + b'^2))) = 2 :=
sorry

end min_value_sum_and_reciprocal_l1384_138407


namespace six_couples_handshakes_l1384_138459

/-- The number of handshakes in a gathering of couples -/
def num_handshakes (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 6 couples, where each person shakes hands
    with everyone except their spouse and one other person, 
    the total number of handshakes is 54. -/
theorem six_couples_handshakes :
  num_handshakes 6 = 54 := by
  sorry

#eval num_handshakes 6  -- Should output 54

end six_couples_handshakes_l1384_138459


namespace share_calculation_l1384_138475

theorem share_calculation (total A B C : ℝ) 
  (h1 : total = 1800)
  (h2 : A = (2/5) * (B + C))
  (h3 : B = (1/5) * (A + C))
  (h4 : A + B + C = total) :
  A = 3600/7 := by
  sorry

end share_calculation_l1384_138475


namespace chord_distance_l1384_138422

/-- Given a circle intersected by three equally spaced parallel lines resulting in chords of lengths 38, 38, and 34, the distance between two adjacent parallel chords is 6. -/
theorem chord_distance (r : ℝ) (d : ℝ) : 
  d > 0 ∧ 
  r^2 = d^2 + 19^2 ∧ 
  r^2 = (3*d)^2 + 17^2 →
  2*d = 6 :=
by sorry

end chord_distance_l1384_138422


namespace special_bet_cost_l1384_138452

def lottery_numbers : ℕ := 36
def numbers_per_bet : ℕ := 7
def cost_per_bet : ℕ := 2

def consecutive_numbers_01_to_10 : ℕ := 3
def consecutive_numbers_11_to_20 : ℕ := 2
def single_number_21_to_30 : ℕ := 1
def single_number_31_to_36 : ℕ := 1

def ways_01_to_10 : ℕ := 10 - consecutive_numbers_01_to_10 + 1
def ways_11_to_20 : ℕ := 10 - consecutive_numbers_11_to_20 + 1
def ways_21_to_30 : ℕ := 10
def ways_31_to_36 : ℕ := 6

theorem special_bet_cost (total_combinations : ℕ) (total_cost : ℕ) :
  total_combinations = ways_01_to_10 * ways_11_to_20 * ways_21_to_30 * ways_31_to_36 ∧
  total_cost = total_combinations * cost_per_bet ∧
  total_cost = 8640 := by
  sorry

end special_bet_cost_l1384_138452


namespace angle_equality_l1384_138416

-- Define the problem statement
theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : Real.sqrt 2 * Real.sin (π/6) = Real.cos θ - Real.sin θ) : 
  θ = π/12 := by
  sorry

end angle_equality_l1384_138416


namespace no_real_d_for_two_distinct_roots_l1384_138435

/-- The function g(x) = x^2 + 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that there are no real values of d such that g(g(x)) has exactly 2 distinct real roots -/
theorem no_real_d_for_two_distinct_roots :
  ¬ ∃ d : ℝ, ∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ ∀ x : ℝ, g_comp d x = 0 ↔ x = r₁ ∨ x = r₂ :=
sorry

end no_real_d_for_two_distinct_roots_l1384_138435


namespace factorization_equality_l1384_138409

theorem factorization_equality (a b : ℝ) : 3 * a^2 + 6 * a * b = 3 * a * (a + 2 * b) := by
  sorry

end factorization_equality_l1384_138409


namespace haley_concert_spending_l1384_138498

def ticket_price : ℕ := 4
def tickets_for_self_and_friends : ℕ := 3
def extra_tickets : ℕ := 5

theorem haley_concert_spending :
  (tickets_for_self_and_friends + extra_tickets) * ticket_price = 32 := by
  sorry

end haley_concert_spending_l1384_138498


namespace min_abs_diff_bound_l1384_138449

theorem min_abs_diff_bound (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  min (min (|a - b|) (|b - c|)) (|c - a|) ≤ Real.sqrt 2 / 2 := by
  sorry

end min_abs_diff_bound_l1384_138449


namespace binomial_coefficient_equality_l1384_138400

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 24 x = Nat.choose 24 (3*x - 8)) → (x = 4 ∨ x = 8) := by
  sorry

end binomial_coefficient_equality_l1384_138400


namespace thabo_hardcover_nonfiction_count_l1384_138455

/-- Represents the number of books Thabo owns of each type -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (bc : BookCollection) : Prop :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction = 180 ∧
  bc.paperback_nonfiction = bc.hardcover_nonfiction + 20 ∧
  bc.paperback_fiction = 2 * bc.paperback_nonfiction

theorem thabo_hardcover_nonfiction_count :
  ∀ bc : BookCollection, is_valid_collection bc → bc.hardcover_nonfiction = 30 :=
by
  sorry

end thabo_hardcover_nonfiction_count_l1384_138455


namespace infinite_fibonacci_divisible_l1384_138414

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: For any positive integer N, there are infinitely many Fibonacci numbers divisible by N -/
theorem infinite_fibonacci_divisible (N : ℕ) (hN : N > 0) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ k ∈ S, N ∣ fib k := by
  sorry

end infinite_fibonacci_divisible_l1384_138414


namespace watch_selling_prices_l1384_138438

/-- Calculates the selling price given the cost price and profit percentage -/
def sellingPrice (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  costPrice * (1 + profitPercentage / 100)

theorem watch_selling_prices :
  let watch1CP : ℚ := 1400
  let watch1Profit : ℚ := 5
  let watch2CP : ℚ := 1800
  let watch2Profit : ℚ := 8
  let watch3CP : ℚ := 2500
  let watch3Profit : ℚ := 12
  (sellingPrice watch1CP watch1Profit = 1470) ∧
  (sellingPrice watch2CP watch2Profit = 1944) ∧
  (sellingPrice watch3CP watch3Profit = 2800) :=
by sorry

end watch_selling_prices_l1384_138438


namespace rachel_essay_time_l1384_138426

/-- Calculates the total time spent on an essay in hours -/
def total_essay_time (pages_written : ℕ) (writing_rate : ℚ) (research_time : ℕ) (editing_time : ℕ) : ℚ :=
  let writing_time : ℚ := pages_written * writing_rate
  let total_minutes : ℚ := research_time + writing_time + editing_time
  total_minutes / 60

/-- Theorem: Rachel spends 5 hours completing the essay -/
theorem rachel_essay_time : 
  total_essay_time 6 (30 : ℚ) 45 75 = 5 := by
  sorry

end rachel_essay_time_l1384_138426


namespace max_collisions_l1384_138430

/-- Represents an ant with a position and velocity -/
structure Ant where
  position : ℝ
  velocity : ℝ

/-- The configuration of n ants on a line -/
def AntConfiguration (n : ℕ) := Fin n → Ant

/-- Predicate to check if the total number of collisions is finite -/
def HasFiniteCollisions (config : AntConfiguration n) : Prop := sorry

/-- The number of collisions that occur in a given configuration -/
def NumberOfCollisions (config : AntConfiguration n) : ℕ := sorry

/-- Theorem stating the maximum number of collisions for n ants -/
theorem max_collisions (n : ℕ) (h : n > 0) :
  ∃ (config : AntConfiguration n),
    HasFiniteCollisions config ∧
    NumberOfCollisions config = n * (n - 1) / 2 ∧
    ∀ (other_config : AntConfiguration n),
      HasFiniteCollisions other_config →
      NumberOfCollisions other_config ≤ n * (n - 1) / 2 := by
  sorry

end max_collisions_l1384_138430


namespace custom_deck_probability_l1384_138477

/-- A custom deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (new_ranks : ℕ)

/-- The probability of drawing a specific type of card -/
def draw_probability (d : Deck) (favorable_cards : ℕ) : ℚ :=
  favorable_cards / d.total_cards

/-- Our specific deck configuration -/
def custom_deck : Deck :=
  { total_cards := 60
  , ranks := 15
  , suits := 4
  , cards_per_suit := 15
  , new_ranks := 2 }

theorem custom_deck_probability :
  let d := custom_deck
  let diamond_cards := d.cards_per_suit
  let new_rank_cards := d.new_ranks * d.suits
  let favorable_cards := diamond_cards + new_rank_cards - d.new_ranks
  draw_probability d favorable_cards = 7 / 20 := by
  sorry


end custom_deck_probability_l1384_138477


namespace sin_double_angle_special_case_l1384_138469

/-- Given an angle θ in the Cartesian coordinate system with vertex at the origin,
    initial side on the positive x-axis, and terminal side on the line y = 3x,
    prove that sin 2θ = 3/5 -/
theorem sin_double_angle_special_case (θ : Real) :
  (∃ (x y : Real), y = 3 * x ∧ x > 0 ∧ y > 0 ∧ (θ = Real.arctan (y / x))) →
  Real.sin (2 * θ) = 3 / 5 := by
  sorry

end sin_double_angle_special_case_l1384_138469


namespace a_10_equals_21_l1384_138495

def arithmetic_sequence (b : ℕ+ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ+, b (n + 1) - b n = d

theorem a_10_equals_21
  (a : ℕ+ → ℚ)
  (b : ℕ+ → ℚ)
  (h1 : a 1 = 3)
  (h2 : arithmetic_sequence b)
  (h3 : ∀ n : ℕ+, b n = a (n + 1) - a n)
  (h4 : b 3 = -2)
  (h5 : b 10 = 12) :
  a 10 = 21 := by
sorry

end a_10_equals_21_l1384_138495


namespace arithmetic_sequence_term_number_l1384_138405

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_term_number
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a5 : a 5 = 33)
  (h_a45 : a 45 = 153)
  : (∃ n : ℕ, a n = 201) ∧ (∀ n : ℕ, a n = 201 → n = 61) :=
sorry

end arithmetic_sequence_term_number_l1384_138405


namespace equation_solutions_l1384_138463

theorem equation_solutions (x : ℝ) : 
  (8 / (Real.sqrt (x - 9) - 10) + 2 / (Real.sqrt (x - 9) - 5) + 
   9 / (Real.sqrt (x - 9) + 5) + 15 / (Real.sqrt (x - 9) + 10) = 0) ↔ 
  (x = (70/23)^2 + 9 ∨ x = (25/11)^2 + 9 ∨ x = 575/34 + 9) :=
sorry

end equation_solutions_l1384_138463


namespace square_side_length_l1384_138423

theorem square_side_length (x : ℝ) (h : x > 0) : 
  x^2 = x * 2 / 2 → x = 1 := by
sorry

end square_side_length_l1384_138423


namespace a_sequence_property_l1384_138428

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | (n + 3) => a (n + 1) + 1998 * a n

theorem a_sequence_property (n : ℕ) (h : n > 0) :
  a (2 * n - 1) = 2 * a n * a (n + 1) + 1998 * a (n - 1) ^ 2 := by
  sorry

end a_sequence_property_l1384_138428


namespace fencing_theorem_l1384_138436

/-- Represents a rectangular field with given dimensions -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- Calculates the fencing required for three sides of a rectangular field -/
def fencing_required (field : RectangularField) : ℝ :=
  2 * field.width + field.length

theorem fencing_theorem (field : RectangularField) 
  (h1 : field.area = 600)
  (h2 : field.uncovered_side = 30)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  fencing_required field = 70 := by
  sorry

#check fencing_theorem

end fencing_theorem_l1384_138436


namespace cubic_expression_evaluation_l1384_138499

theorem cubic_expression_evaluation :
  1001^3 - 1000 * 1001^2 - 1000^2 * 1001 + 1000^3 = 2001 := by
  sorry

end cubic_expression_evaluation_l1384_138499


namespace reciprocal_sum_fourths_sixths_l1384_138403

theorem reciprocal_sum_fourths_sixths : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by sorry

end reciprocal_sum_fourths_sixths_l1384_138403


namespace no_two_right_angles_l1384_138457

-- Define a triangle as a structure with three angles
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_is_180 : A + B + C = 180

-- Theorem: A triangle cannot have two right angles
theorem no_two_right_angles (t : Triangle) : ¬(t.A = 90 ∧ t.B = 90 ∨ t.A = 90 ∧ t.C = 90 ∨ t.B = 90 ∧ t.C = 90) := by
  sorry


end no_two_right_angles_l1384_138457


namespace absolute_value_of_z_l1384_138440

theorem absolute_value_of_z (r : ℝ) (z : ℂ) 
  (hr : |r| > 2) 
  (hz : z - 1/z = r) : 
  Complex.abs z = Real.sqrt ((r^2 / 2) + 1) := by
  sorry

end absolute_value_of_z_l1384_138440


namespace cube_root_64_equals_2_power_m_l1384_138465

theorem cube_root_64_equals_2_power_m (m : ℝ) : (64 : ℝ)^(1/3) = 2^m → m = 2 := by
  sorry

end cube_root_64_equals_2_power_m_l1384_138465


namespace derek_dogs_at_six_l1384_138401

theorem derek_dogs_at_six (dogs_at_six cars_at_six : ℕ) 
  (h1 : dogs_at_six = 3 * cars_at_six)
  (h2 : cars_at_six + 210 = 2 * 120)
  : dogs_at_six = 90 := by
  sorry

end derek_dogs_at_six_l1384_138401


namespace min_value_of_z_l1384_138406

theorem min_value_of_z (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x - 2*y + 3 = 0) :
  ∀ z : ℝ, z = y^2 / x → z ≥ 3 :=
sorry

end min_value_of_z_l1384_138406


namespace perpendicular_lines_from_perpendicular_parallel_planes_l1384_138431

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_parallel_planes
  (a b : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : contained_in b β)
  (h3 : parallel α β) :
  line_perpendicular a b :=
sorry

end perpendicular_lines_from_perpendicular_parallel_planes_l1384_138431


namespace c_alone_time_l1384_138412

-- Define the work rates for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
def condition1 : Prop := rA + rB = 1/3
def condition2 : Prop := rB + rC = 1/3
def condition3 : Prop := rA + rC = 2/3

-- Theorem to prove
theorem c_alone_time (h1 : condition1 rA rB) (h2 : condition2 rB rC) (h3 : condition3 rA rC) :
  1 / rC = 3 := by
  sorry


end c_alone_time_l1384_138412


namespace log_sum_equality_l1384_138448

-- Define the problem
theorem log_sum_equality : Real.log 50 + Real.log 20 + Real.log 4 = 3.60206 := by
  sorry

#check log_sum_equality

end log_sum_equality_l1384_138448


namespace triangle_reconstruction_from_nagel_point_vertex_and_altitude_foot_l1384_138453

/- Define the necessary types and structures -/
structure Triangle where
  A : Point
  B : Point
  C : Point

structure Point where
  x : ℝ
  y : ℝ

/- Define the given information -/
def nagel_point (t : Triangle) : Point := sorry
def altitude_foot (t : Triangle) (v : Point) : Point := sorry

/- State the theorem -/
theorem triangle_reconstruction_from_nagel_point_vertex_and_altitude_foot 
  (N : Point) (B : Point) (E : Point) :
  ∃! (t : Triangle), 
    B = t.B ∧ 
    N = nagel_point t ∧ 
    E = altitude_foot t B := by
  sorry

end triangle_reconstruction_from_nagel_point_vertex_and_altitude_foot_l1384_138453


namespace sum_of_squares_zero_implies_sum_l1384_138429

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 2)^2 + (y - 6)^2 + (z - 8)^2 = 0 → 2*x + 2*y + 2*z = 32 := by
  sorry

end sum_of_squares_zero_implies_sum_l1384_138429


namespace derivative_bound_l1384_138470

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem derivative_bound
  (h_cont : ContDiff ℝ 3 f)
  (h_pos : ∀ x, f x > 0 ∧ (deriv f) x > 0 ∧ (deriv^[2] f) x > 0 ∧ (deriv^[3] f) x > 0)
  (h_bound : ∀ x, (deriv^[3] f) x ≤ f x) :
  ∀ x, (deriv f) x < 2 * f x :=
sorry

end derivative_bound_l1384_138470


namespace pure_imaginary_complex_number_l1384_138450

theorem pure_imaginary_complex_number (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 - 1) (x - 1)
  (z.re = 0 ∧ z ≠ 0) → x = -1 := by
  sorry

end pure_imaginary_complex_number_l1384_138450


namespace tangent_intersection_theorem_l1384_138460

/-- The x-coordinate of the point where a line tangent to two circles intersects the x-axis -/
def tangent_intersection_x : ℝ := 4.5

/-- The radius of the first circle -/
def r1 : ℝ := 3

/-- The radius of the second circle -/
def r2 : ℝ := 5

/-- The x-coordinate of the center of the second circle -/
def c2_x : ℝ := 12

theorem tangent_intersection_theorem :
  let x := tangent_intersection_x
  x > 0 ∧ 
  x / (c2_x - x) = r1 / r2 := by
  sorry

end tangent_intersection_theorem_l1384_138460


namespace solution_characterization_l1384_138432

def divides (x y : ℤ) : Prop := ∃ k : ℤ, y = k * x

def is_solution (a b : ℕ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ 
  divides (2 * a + 1) (3 * b - 1) ∧
  divides (2 * b + 1) (3 * a - 1)

theorem solution_characterization :
  ∀ a b : ℕ, is_solution a b ↔ ((a = 2 ∧ b = 2) ∨ (a = 12 ∧ b = 17) ∨ (a = 17 ∧ b = 12)) :=
by sorry

end solution_characterization_l1384_138432


namespace range_of_expression_l1384_138476

theorem range_of_expression (a b c : ℝ) 
  (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) 
  (h4 : -2 < c) (h5 : c < -1) : 
  ∃ (x : ℝ), 0 < x ∧ x < 8 ∧ x = (a - b) * c^2 :=
sorry

end range_of_expression_l1384_138476


namespace quadratic_function_properties_l1384_138410

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Theorem stating the properties of a specific quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction)
  (point1 : f.a * (-1)^2 + f.b * (-1) + f.c = -1)
  (point2 : f.c = 1)
  (condition : f.a * (-2)^2 + f.b * (-2) + f.c > 1) :
  (f.a * f.b * f.c > 0) ∧
  (∃ x y : ℝ, x ≠ y ∧ f.a * x^2 + f.b * x + f.c - 3 = 0 ∧ f.a * y^2 + f.b * y + f.c - 3 = 0) ∧
  (f.a + f.b + f.c > 7) := by
sorry

end quadratic_function_properties_l1384_138410


namespace stacy_height_proof_l1384_138492

/-- Calculates Stacy's current height given her previous height, James' growth, and the difference between their growth. -/
def stacys_current_height (stacy_previous_height james_growth growth_difference : ℕ) : ℕ :=
  stacy_previous_height + james_growth + growth_difference

/-- Proves that Stacy's current height is 57 inches. -/
theorem stacy_height_proof :
  stacys_current_height 50 1 6 = 57 := by
  sorry

end stacy_height_proof_l1384_138492


namespace fabric_area_calculation_l1384_138411

/-- The area of a rectangular piece of fabric -/
def fabric_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The area of a rectangular piece of fabric with width 3 cm and length 8 cm is 24 square cm -/
theorem fabric_area_calculation :
  fabric_area 3 8 = 24 := by
  sorry

end fabric_area_calculation_l1384_138411


namespace min_value_reciprocal_sum_l1384_138425

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 2*y' = 1 → 1/x' + 1/y' ≥ 3 + 2*Real.sqrt 2) ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 1 ∧ 1/x₀ + 1/y₀ = 3 + 2*Real.sqrt 2) :=
by sorry

end min_value_reciprocal_sum_l1384_138425


namespace comic_book_stacking_arrangements_l1384_138467

theorem comic_book_stacking_arrangements :
  let hulk_comics : ℕ := 8
  let ironman_comics : ℕ := 7
  let wolverine_comics : ℕ := 6
  let total_comics : ℕ := hulk_comics + ironman_comics + wolverine_comics
  let arrange_hulk : ℕ := Nat.factorial hulk_comics
  let arrange_ironman : ℕ := Nat.factorial ironman_comics
  let arrange_wolverine : ℕ := Nat.factorial wolverine_comics
  let arrange_within_groups : ℕ := arrange_hulk * arrange_ironman * arrange_wolverine
  let arrange_groups : ℕ := Nat.factorial 3
  arrange_within_groups * arrange_groups = 69657088000 :=
by
  sorry

end comic_book_stacking_arrangements_l1384_138467


namespace smallest_m_for_meaningful_sqrt_l1384_138418

theorem smallest_m_for_meaningful_sqrt (m : ℤ) : 
  (∀ k : ℤ, k < m → ¬(2*k + 1 ≥ 0)) → (2*m + 1 ≥ 0) → m = 0 :=
by sorry

end smallest_m_for_meaningful_sqrt_l1384_138418


namespace x_power_ten_plus_inverse_l1384_138458

theorem x_power_ten_plus_inverse (x : ℝ) (h : x + 1/x = 5) : x^10 + 1/x^10 = 6430223 := by
  sorry

end x_power_ten_plus_inverse_l1384_138458


namespace min_value_expression_l1384_138421

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  b / a^2 + 4 / b + a / 2 ≥ 2 * Real.sqrt 2 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ b₀ / a₀^2 + 4 / b₀ + a₀ / 2 = 2 * Real.sqrt 2 :=
sorry

end min_value_expression_l1384_138421


namespace min_value_expression_min_value_achievable_l1384_138439

theorem min_value_expression (x y : ℝ) : x^2 + 8*x*Real.sin y - 16*(Real.cos y)^2 ≥ -16 := by sorry

theorem min_value_achievable : ∃ x y : ℝ, x^2 + 8*x*Real.sin y - 16*(Real.cos y)^2 = -16 := by sorry

end min_value_expression_min_value_achievable_l1384_138439


namespace marble_distribution_l1384_138462

def jasmine_initial : ℕ := 120
def lola_initial : ℕ := 15
def marbles_given : ℕ := 19

theorem marble_distribution :
  (jasmine_initial - marbles_given) = 3 * (lola_initial + marbles_given) := by
  sorry

end marble_distribution_l1384_138462


namespace ellipse_theorem_l1384_138447

/-- An ellipse with center at the origin, foci on the x-axis, 
    minor axis length 8√2, and eccentricity 1/3 -/
structure Ellipse where
  b : ℝ
  e : ℝ
  minor_axis : b = 4 * Real.sqrt 2
  eccentricity : e = 1/3

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 32 = 1

/-- Theorem stating that the given ellipse satisfies the equation -/
theorem ellipse_theorem (E : Ellipse) (x y : ℝ) :
  ellipse_equation x y := by
  sorry

end ellipse_theorem_l1384_138447


namespace triangle_determines_plane_l1384_138404

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Function to determine a plane from a triangle -/
def planeFromTriangle (t : Triangle3D) : Plane3D := sorry

theorem triangle_determines_plane (t : Triangle3D) : 
  ¬collinear t.a t.b t.c → ∃! p : Plane3D, p = planeFromTriangle t :=
sorry

end triangle_determines_plane_l1384_138404


namespace tv_cost_l1384_138413

theorem tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 600 →
  furniture_fraction = 3/4 →
  tv_cost = savings - (furniture_fraction * savings) →
  tv_cost = 150 := by
sorry

end tv_cost_l1384_138413


namespace smallest_number_of_eggs_l1384_138484

theorem smallest_number_of_eggs (total_containers : ℕ) (deficient_containers : ℕ) : 
  deficient_containers = 3 →
  (15 * total_containers - deficient_containers > 150) →
  (∀ n : ℕ, 15 * n - deficient_containers > 150 → n ≥ total_containers) →
  15 * total_containers - deficient_containers = 162 :=
by sorry

end smallest_number_of_eggs_l1384_138484


namespace gnome_with_shoes_weighs_34_l1384_138456

/-- The weight of a gnome without shoes -/
def gnome_weight : ℝ := sorry

/-- The weight of a gnome with shoes -/
def gnome_with_shoes_weight : ℝ := sorry

/-- The difference in weight between a gnome with shoes and without shoes -/
def shoe_weight_difference : ℝ := 2

/-- The total weight of five gnomes with shoes and five gnomes without shoes -/
def total_weight : ℝ := 330

/-- Theorem stating that a gnome with shoes weighs 34 kg -/
theorem gnome_with_shoes_weighs_34 :
  gnome_with_shoes_weight = 34 :=
by
  sorry

/-- Axiom: A gnome with shoes weighs 2 kg more than a gnome without shoes -/
axiom shoe_weight_relation :
  gnome_with_shoes_weight = gnome_weight + shoe_weight_difference

/-- Axiom: The total weight of five gnomes with shoes and five gnomes without shoes is 330 kg -/
axiom total_weight_relation :
  5 * gnome_with_shoes_weight + 5 * gnome_weight = total_weight

end gnome_with_shoes_weighs_34_l1384_138456


namespace inequality_equivalence_l1384_138466

theorem inequality_equivalence (x : ℝ) : -4 * x - 8 > 0 ↔ x < -2 := by
  sorry

end inequality_equivalence_l1384_138466


namespace five_mondays_in_september_l1384_138480

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Represents a month with its dates -/
structure Month where
  dates : List Date
  numDays : Nat

def August : Month := sorry
def September : Month := sorry

/-- Counts the number of occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat := sorry

/-- Determines the day of the week for the first day of the next month -/
def nextMonthFirstDay (m : Month) : DayOfWeek := sorry

theorem five_mondays_in_september 
  (h1 : August.numDays = 31)
  (h2 : September.numDays = 30)
  (h3 : countDayOccurrences August DayOfWeek.Sunday = 5) :
  countDayOccurrences September DayOfWeek.Monday = 5 := by sorry

end five_mondays_in_september_l1384_138480


namespace system_one_solution_l1384_138471

theorem system_one_solution (x y : ℝ) : 
  x + 3 * y = 3 ∧ x - y = 1 → x = (3 : ℝ) / 2 ∧ y = (1 : ℝ) / 2 := by
  sorry


end system_one_solution_l1384_138471


namespace car_original_price_l1384_138496

/-- Proves the original price of a car given repair cost, selling price, and profit percentage -/
theorem car_original_price (repair_cost selling_price : ℝ) (profit_percentage : ℝ) :
  repair_cost = 12000 →
  selling_price = 80000 →
  profit_percentage = 40.35 →
  ∃ (original_price : ℝ),
    (selling_price - (original_price + repair_cost)) / original_price * 100 = profit_percentage ∧
    abs (original_price - 48425.44) < 0.01 := by
  sorry

end car_original_price_l1384_138496


namespace quadratic_function_equal_values_l1384_138464

theorem quadratic_function_equal_values (a m n : ℝ) (h1 : a ≠ 0) (h2 : m ≠ n) :
  (a * m^2 - 4 * a * m - 3 = a * n^2 - 4 * a * n - 3) → m + n = 4 := by
  sorry

end quadratic_function_equal_values_l1384_138464


namespace principal_calculation_l1384_138417

theorem principal_calculation (P r : ℝ) : 
  P * r * 2 = 10200 →
  P * ((1 + r)^2 - 1) = 11730 →
  P = 17000 := by
sorry

end principal_calculation_l1384_138417


namespace distance_to_origin_l1384_138443

/-- The distance from the point corresponding to the complex number 2i/(1-i) to the origin in the complex plane is √2. -/
theorem distance_to_origin : Complex.abs (2 * Complex.I / (1 - Complex.I)) = Real.sqrt 2 := by
  sorry

end distance_to_origin_l1384_138443


namespace pages_left_to_read_l1384_138478

/-- Calculates the number of pages left to be read in a book --/
theorem pages_left_to_read 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (daily_reading : ℕ) 
  (days : ℕ) 
  (h1 : total_pages = 381) 
  (h2 : pages_read = 149) 
  (h3 : daily_reading = 20) 
  (h4 : days = 7) :
  total_pages - (pages_read + daily_reading * days) = 92 := by
  sorry

end pages_left_to_read_l1384_138478


namespace quadratic_roots_sum_of_squares_l1384_138451

theorem quadratic_roots_sum_of_squares (m n : ℝ) : 
  (m^2 - 2*m - 1 = 0) → (n^2 - 2*n - 1 = 0) → m^2 + n^2 = 6 := by
  sorry

end quadratic_roots_sum_of_squares_l1384_138451


namespace vowel_count_l1384_138434

theorem vowel_count (num_vowels : ℕ) (total_written : ℕ) : 
  num_vowels = 5 → total_written = 15 → (total_written / num_vowels : ℕ) = 3 := by
  sorry

end vowel_count_l1384_138434


namespace two_x_minus_y_value_l1384_138472

theorem two_x_minus_y_value (x y : ℝ) (hx : |x| = 3) (hy : |y| = 2) (hxy : x > y) :
  2 * x - y = 4 ∨ 2 * x - y = 8 := by
sorry

end two_x_minus_y_value_l1384_138472


namespace soccer_balls_count_initial_balls_count_l1384_138494

/-- The initial number of soccer balls in the bag -/
def initial_balls : ℕ := sorry

/-- The number of additional balls added to the bag -/
def added_balls : ℕ := 18

/-- The final number of balls in the bag -/
def final_balls : ℕ := 24

/-- Theorem stating that the initial number of balls plus the added balls equals the final number of balls -/
theorem soccer_balls_count : initial_balls + added_balls = final_balls := by sorry

/-- Theorem proving that the initial number of balls is 6 -/
theorem initial_balls_count : initial_balls = 6 := by sorry

end soccer_balls_count_initial_balls_count_l1384_138494


namespace anns_total_blocks_l1384_138419

/-- Ann's initial number of blocks -/
def initial_blocks : ℕ := 9

/-- Number of blocks Ann finds -/
def found_blocks : ℕ := 44

/-- Theorem: Ann's total number of blocks after finding more -/
theorem anns_total_blocks : initial_blocks + found_blocks = 53 := by
  sorry

end anns_total_blocks_l1384_138419


namespace exists_natural_number_with_seventh_eighth_root_natural_l1384_138441

theorem exists_natural_number_with_seventh_eighth_root_natural :
  ∃ (n : ℕ), n > 1 ∧ ∃ (m : ℕ), n^(7/8) = m := by
  sorry

end exists_natural_number_with_seventh_eighth_root_natural_l1384_138441


namespace circle_radius_in_triangle_l1384_138420

/-- Represents a triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Determines if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop := sorry

/-- Determines if a circle is tangent to two sides of a triangle -/
def is_tangent_to_sides (c : Circle) (t : Triangle) : Prop := sorry

/-- Determines if a circle lies entirely within a triangle -/
def lies_within_triangle (c : Circle) (t : Triangle) : Prop := sorry

/-- Main theorem statement -/
theorem circle_radius_in_triangle (t : Triangle) (r s : Circle) : 
  t.a = 120 → t.b = 120 → t.c = 70 →
  r.radius = 20 →
  is_tangent_to_sides r t →
  are_externally_tangent r s →
  is_tangent_to_sides s t →
  lies_within_triangle s t →
  s.radius = 54 - 8 * Real.sqrt 41 := by
  sorry

end circle_radius_in_triangle_l1384_138420


namespace no_prime_satisfies_equation_l1384_138489

theorem no_prime_satisfies_equation : 
  ¬ ∃ (q : ℕ), Nat.Prime q ∧ 
  (1 * q^3 + 0 * q^2 + 1 * q + 2) + 
  (3 * q^2 + 0 * q + 7) + 
  (1 * q^2 + 1 * q + 4) + 
  (1 * q^2 + 2 * q + 6) + 
  7 = 
  (1 * q^2 + 4 * q + 3) + 
  (2 * q^2 + 7 * q + 2) + 
  (3 * q^2 + 6 * q + 1) :=
by sorry

end no_prime_satisfies_equation_l1384_138489


namespace arithmetic_sequence_formula_l1384_138415

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of an arithmetic sequence with first term a₁ and common difference d. -/
def arithmetic_sequence_term (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1 : ℤ) * d

theorem arithmetic_sequence_formula (a : ℕ → ℤ) :
  is_arithmetic_sequence a → a 1 = 1 → a 3 = -3 →
  ∀ n : ℕ, a n = -2 * n + 3 := by
  sorry

end arithmetic_sequence_formula_l1384_138415


namespace equation_system_solutions_l1384_138446

theorem equation_system_solutions :
  ∀ (x y z : ℝ),
  (x = (2 * z^2) / (1 + z^2)) ∧
  (y = (2 * x^2) / (1 + x^2)) ∧
  (z = (2 * y^2) / (1 + y^2)) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by sorry

end equation_system_solutions_l1384_138446


namespace inequality_always_holds_l1384_138433

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end inequality_always_holds_l1384_138433


namespace prop_values_l1384_138468

theorem prop_values (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(¬p ∨ q)) : 
  p ∧ ¬q := by
  sorry

end prop_values_l1384_138468


namespace investment_percentage_l1384_138493

/-- Proves that given the investment conditions, the unknown percentage is 4% -/
theorem investment_percentage (total_investment : ℝ) (known_rate : ℝ) (unknown_rate : ℝ) 
  (total_interest : ℝ) (amount_at_unknown_rate : ℝ) :
  total_investment = 17000 →
  known_rate = 18 →
  total_interest = 1380 →
  amount_at_unknown_rate = 12000 →
  (amount_at_unknown_rate * unknown_rate / 100 + 
   (total_investment - amount_at_unknown_rate) * known_rate / 100 = total_interest) →
  unknown_rate = 4 := by
sorry


end investment_percentage_l1384_138493


namespace inequality_solution_implies_a_value_l1384_138487

-- Define the inequality
def inequality (x a : ℝ) : Prop := (x + a) / (x^2 + 4*x + 3) > 0

-- Define the solution set
def solution_set (x : ℝ) : Prop := (-3 < x ∧ x < -1) ∨ x > 2

-- Theorem statement
theorem inequality_solution_implies_a_value :
  (∀ x : ℝ, inequality x a ↔ solution_set x) → a = -2 := by
  sorry

end inequality_solution_implies_a_value_l1384_138487


namespace hyperbola_intersection_slopes_product_l1384_138486

/-- Hyperbola C with asymptotic line equation y = ±√3x and point P(2,3) on it -/
structure Hyperbola :=
  (asymptote : ℝ → ℝ)
  (point : ℝ × ℝ)
  (h_asymptote : ∀ x, asymptote x = Real.sqrt 3 * x ∨ asymptote x = -Real.sqrt 3 * x)
  (h_point : point = (2, 3))

/-- Line l: y = kx + m -/
structure Line :=
  (k m : ℝ)

/-- Intersection points A and B of line l with hyperbola C -/
structure Intersection :=
  (A B : ℝ × ℝ)
  (k₁ k₂ : ℝ)

/-- The theorem to be proved -/
theorem hyperbola_intersection_slopes_product
  (C : Hyperbola) (l : Line) (I : Intersection) :
  ∃ (k m : ℝ), l.k = -3/2 ∧ I.k₁ * I.k₂ = -3 := by sorry

end hyperbola_intersection_slopes_product_l1384_138486


namespace smallest_undefined_value_l1384_138402

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y < 1, ∃ z, (y + 2) / (10 * y^2 - 90 * y + 20) = z) ∧ 
  ¬∃ z, (1 + 2) / (10 * 1^2 - 90 * 1 + 20) = z := by
  sorry

end smallest_undefined_value_l1384_138402


namespace line_through_point_l1384_138497

theorem line_through_point (k : ℝ) : (2 * k * 3 - 1 = 5) ↔ (k = 1) := by sorry

end line_through_point_l1384_138497


namespace finite_consecutive_divisible_pairs_infinite_highly_divisible_multiples_l1384_138485

-- Define the number of divisors function
def d (n : ℕ) : ℕ := (Nat.divisors n).card

-- Define highly divisible property
def is_highly_divisible (n : ℕ) : Prop :=
  ∀ m : ℕ, m < n → d m < d n

-- Define consecutive highly divisible property
def consecutive_highly_divisible (m n : ℕ) : Prop :=
  is_highly_divisible m ∧ is_highly_divisible n ∧ m < n ∧
  ∀ s : ℕ, m < s → s < n → ¬is_highly_divisible s

-- Theorem for part (a)
theorem finite_consecutive_divisible_pairs :
  {p : ℕ × ℕ | consecutive_highly_divisible p.1 p.2 ∧ p.1 ∣ p.2}.Finite :=
sorry

-- Theorem for part (b)
theorem infinite_highly_divisible_multiples (p : ℕ) (hp : Nat.Prime p) :
  {r : ℕ | is_highly_divisible r ∧ is_highly_divisible (p * r)}.Infinite :=
sorry

end finite_consecutive_divisible_pairs_infinite_highly_divisible_multiples_l1384_138485


namespace triangle_inequality_range_l1384_138427

theorem triangle_inequality_range (A B C : ℝ) (t : ℝ) : 
  0 < B → B ≤ π/3 → 
  (∀ x : ℝ, (x + 2 + Real.sin (2*B))^2 + (Real.sqrt 2 * t * Real.sin (B + π/4))^2 ≥ 1) →
  t ∈ Set.Ici 1 ∪ Set.Iic (-1) :=
by sorry

end triangle_inequality_range_l1384_138427
