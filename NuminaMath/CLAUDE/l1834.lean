import Mathlib

namespace boat_speed_in_still_water_l1834_183437

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed : ℝ) (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ),
  current_speed = 6 →
  downstream_distance = 5.2 →
  downstream_time = 1/5 →
  (boat_speed + current_speed) * downstream_time = downstream_distance →
  boat_speed = 20 := by
sorry

end boat_speed_in_still_water_l1834_183437


namespace monotonic_decreasing_quadratic_l1834_183488

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem monotonic_decreasing_quadratic (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, x < y → f a x > f a y) →
  a ∈ Set.Ici 2 :=
by sorry

end monotonic_decreasing_quadratic_l1834_183488


namespace correct_food_suggestion_ratio_l1834_183452

/-- The ratio of food suggestions by students -/
def food_suggestion_ratio (sushi mashed_potatoes bacon tomatoes : ℕ) : List ℕ :=
  [sushi, mashed_potatoes, bacon, tomatoes]

/-- Theorem stating the correct ratio of food suggestions -/
theorem correct_food_suggestion_ratio :
  food_suggestion_ratio 297 144 467 79 = [297, 144, 467, 79] := by
  sorry

end correct_food_suggestion_ratio_l1834_183452


namespace chess_piece_arrangements_l1834_183472

def num_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial 2)^k

theorem chess_piece_arrangements :
  num_arrangements 6 3 = 90 := by
  sorry

end chess_piece_arrangements_l1834_183472


namespace max_reciprocal_sum_exists_l1834_183453

/-- Given a quadratic polynomial x^2 - px + q with roots r₁ and r₂ satisfying
    r₁ + r₂ = r₁² + r₂² = r₁⁴ + r₂⁴, there exists a maximum value for 1/r₁⁵ + 1/r₂⁵ -/
theorem max_reciprocal_sum_exists (p q r₁ r₂ : ℝ) : 
  (r₁ * r₁ - p * r₁ + q = 0) →
  (r₂ * r₂ - p * r₂ + q = 0) →
  (r₁ + r₂ = r₁^2 + r₂^2) →
  (r₁ + r₂ = r₁^4 + r₂^4) →
  ∃ (M : ℝ), ∀ (s₁ s₂ : ℝ), 
    (s₁ * s₁ - p * s₁ + q = 0) →
    (s₂ * s₂ - p * s₂ + q = 0) →
    (s₁ + s₂ = s₁^2 + s₂^2) →
    (s₁ + s₂ = s₁^4 + s₂^4) →
    1/s₁^5 + 1/s₂^5 ≤ M :=
by
  sorry


end max_reciprocal_sum_exists_l1834_183453


namespace august_tips_multiple_l1834_183475

/-- 
Proves that if a worker's tips for one month (August) are 0.625 of their total tips for 7 months, 
and the August tips are some multiple of the average tips for the other 6 months, then this multiple is 10.
-/
theorem august_tips_multiple (total_months : ℕ) (other_months : ℕ) (august_ratio : ℝ) (M : ℝ) : 
  total_months = 7 → 
  other_months = 6 → 
  august_ratio = 0.625 →
  M * (1 / other_months : ℝ) * (1 - august_ratio) * total_months = august_ratio →
  M = 10 := by
  sorry

end august_tips_multiple_l1834_183475


namespace sqrt_sum_theorem_l1834_183403

theorem sqrt_sum_theorem (a b : ℝ) : 
  Real.sqrt ((a - b)^2) + (a - b)^(1/5) = 
    if a ≥ b then 2*(a - b) else 0 := by
  sorry

end sqrt_sum_theorem_l1834_183403


namespace second_number_value_l1834_183445

theorem second_number_value (A B : ℝ) : 
  A = 15 → 
  0.4 * A = 0.8 * B + 2 → 
  B = 5 := by
sorry

end second_number_value_l1834_183445


namespace rationalize_sqrt_five_l1834_183427

theorem rationalize_sqrt_five : 
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = 11/4 + (5/4) * Real.sqrt 5 := by
  sorry

end rationalize_sqrt_five_l1834_183427


namespace smallest_block_size_l1834_183451

/-- Given a rectangular block with dimensions l, m, and n,
    where (l-1)(m-1)(n-1) = 210, the smallest possible value of l*m*n is 336. -/
theorem smallest_block_size (l m n : ℕ) (h : (l-1)*(m-1)*(n-1) = 210) :
  l*m*n ≥ 336 ∧ ∃ (l' m' n' : ℕ), (l'-1)*(m'-1)*(n'-1) = 210 ∧ l'*m'*n' = 336 := by
  sorry

end smallest_block_size_l1834_183451


namespace cube_surface_area_l1834_183410

/-- The surface area of a cube with volume 64 cubic cm is 96 square cm. -/
theorem cube_surface_area (cube_volume : ℝ) (h : cube_volume = 64) : 
  6 * (cube_volume ^ (1/3))^2 = 96 := by
  sorry

end cube_surface_area_l1834_183410


namespace polynomial_properties_l1834_183458

/-- Definition of the polynomial -/
def p (x y : ℝ) : ℝ := x * y^3 - x^2 + 7

/-- The degree of the polynomial -/
def degree_p : ℕ := 4

/-- The number of terms in the polynomial -/
def num_terms_p : ℕ := 3

theorem polynomial_properties :
  (degree_p = 4) ∧ (num_terms_p = 3) := by sorry

end polynomial_properties_l1834_183458


namespace sandy_loses_two_marks_l1834_183433

/-- Represents Sandy's math test results -/
structure SandyTest where
  correct_mark : ℕ  -- marks for each correct sum
  total_sums : ℕ    -- total number of sums attempted
  total_marks : ℕ   -- total marks obtained
  correct_sums : ℕ  -- number of correct sums

/-- Calculates the marks lost for each incorrect sum -/
def marks_lost_per_incorrect (test : SandyTest) : ℚ :=
  let correct_marks := test.correct_mark * test.correct_sums
  let incorrect_sums := test.total_sums - test.correct_sums
  let total_marks_lost := correct_marks - test.total_marks
  (total_marks_lost : ℚ) / incorrect_sums

/-- Theorem stating that Sandy loses 2 marks for each incorrect sum -/
theorem sandy_loses_two_marks (test : SandyTest) 
  (h1 : test.correct_mark = 3)
  (h2 : test.total_sums = 30)
  (h3 : test.total_marks = 50)
  (h4 : test.correct_sums = 22) :
  marks_lost_per_incorrect test = 2 := by
  sorry

#eval marks_lost_per_incorrect { correct_mark := 3, total_sums := 30, total_marks := 50, correct_sums := 22 }

end sandy_loses_two_marks_l1834_183433


namespace solution_set_of_composite_function_l1834_183489

/-- Given a function f(x) = 2x - 1, the solution set of f[f(x)] ≥ 1 is {x | x ≥ 1} -/
theorem solution_set_of_composite_function (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x - 1) :
  {x : ℝ | f (f x) ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

end solution_set_of_composite_function_l1834_183489


namespace factorial_ratio_l1834_183435

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end factorial_ratio_l1834_183435


namespace existence_of_special_sequence_l1834_183464

theorem existence_of_special_sequence :
  ∃ (a : ℕ → ℝ) (x y : ℝ),
    (∀ n, a n ≠ 0) ∧
    (∀ n, a (n + 2) = x * a (n + 1) + y * a n) ∧
    (∀ r > 0, ∃ i j : ℕ, |a i| < r ∧ r < |a j|) := by
  sorry

end existence_of_special_sequence_l1834_183464


namespace trapezoid_circumradii_relation_l1834_183450

-- Define a trapezoid
structure Trapezoid :=
  (A₁ A₂ A₃ A₄ : ℝ × ℝ)

-- Define the diagonal lengths
def diagonal₁ (t : Trapezoid) : ℝ := sorry
def diagonal₂ (t : Trapezoid) : ℝ := sorry

-- Define the circumradius of a triangle formed by three points of the trapezoid
def circumradius (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem trapezoid_circumradii_relation (t : Trapezoid) :
  let e := diagonal₁ t
  let f := diagonal₂ t
  let r₁ := circumradius t.A₂ t.A₃ t.A₄
  let r₂ := circumradius t.A₁ t.A₃ t.A₄
  let r₃ := circumradius t.A₁ t.A₂ t.A₄
  let r₄ := circumradius t.A₁ t.A₂ t.A₃
  (r₂ + r₄) / e = (r₁ + r₃) / f := by sorry

end trapezoid_circumradii_relation_l1834_183450


namespace movie_theater_seats_l1834_183434

theorem movie_theater_seats (sections : ℕ) (seats_per_section : ℕ) 
  (h1 : sections = 9)
  (h2 : seats_per_section = 30) :
  sections * seats_per_section = 270 := by
sorry

end movie_theater_seats_l1834_183434


namespace total_pages_to_read_l1834_183474

def pages_read : ℕ := 113
def days_left : ℕ := 5
def pages_per_day : ℕ := 59

theorem total_pages_to_read : pages_read + days_left * pages_per_day = 408 := by
  sorry

end total_pages_to_read_l1834_183474


namespace determine_bal_meaning_l1834_183421

/-- Represents the possible responses from a native --/
inductive Response
| Bal
| Da

/-- Represents the possible meanings of a word --/
inductive Meaning
| Yes
| No

/-- A native person who can respond to questions --/
structure Native where
  response : String → Response

/-- The meaning of the word "bal" --/
def balMeaning (n : Native) : Meaning :=
  match n.response "Are you a human?" with
  | Response.Bal => Meaning.Yes
  | Response.Da => Meaning.No

/-- Theorem stating that it's possible to determine the meaning of "bal" with a single question --/
theorem determine_bal_meaning (n : Native) :
  (∀ q : String, n.response q = Response.Bal ∨ n.response q = Response.Da) →
  (n.response "Are you a human?" = Response.Da → Meaning.Yes = Meaning.Yes) →
  (∀ q : String, n.response q = Response.Bal → Meaning.Yes = balMeaning n) :=
by
  sorry


end determine_bal_meaning_l1834_183421


namespace clock_hands_straight_period_l1834_183446

/-- Represents the number of times clock hands are straight in a given period -/
def straight_hands (period : ℝ) : ℕ := sorry

/-- Represents the number of times clock hands coincide in a given period -/
def coinciding_hands (period : ℝ) : ℕ := sorry

/-- Represents the number of times clock hands are opposite in a given period -/
def opposite_hands (period : ℝ) : ℕ := sorry

theorem clock_hands_straight_period :
  straight_hands 12 = 22 ∧
  (∀ period : ℝ, straight_hands period = coinciding_hands period + opposite_hands period) ∧
  coinciding_hands 12 = 11 ∧
  opposite_hands 12 = 11 :=
by sorry

end clock_hands_straight_period_l1834_183446


namespace triangle_properties_l1834_183422

open Real

theorem triangle_properties (A B C : ℝ) (a b : ℝ) :
  let D := (A + B) / 2
  2 * sin A * cos B + b * sin (2 * A) + 2 * sqrt 3 * a * cos C = 0 →
  2 = 2 →
  sqrt 3 = sqrt 3 →
  C = 2 * π / 3 ∧
  (1/2) * (1/2) * a * 2 * sin C = sqrt 3 := by
  sorry

end triangle_properties_l1834_183422


namespace min_value_of_expression_l1834_183499

/-- The line equation ax + 2by - 2 = 0 -/
def line_equation (a b x y : ℝ) : Prop := a * x + 2 * b * y - 2 = 0

/-- The circle equation x^2 + y^2 - 4x - 2y - 8 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 8 = 0

/-- The line bisects the circumference of the circle -/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a b x y → circle_equation x y

theorem min_value_of_expression (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_bisect : line_bisects_circle a b) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_of_expression_l1834_183499


namespace symmetric_function_value_l1834_183428

/-- A function with a graph symmetric about the origin -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The main theorem -/
theorem symmetric_function_value (f : ℝ → ℝ) 
  (h_sym : SymmetricAboutOrigin f)
  (h_pos : ∀ x > 0, f x = 2^x - 3) : 
  f (-2) = -1 := by
  sorry

end symmetric_function_value_l1834_183428


namespace smallest_k_for_no_real_roots_l1834_183478

theorem smallest_k_for_no_real_roots :
  ∃ (k : ℤ), k = 3 ∧
  (∀ (x : ℝ), 3 * x * (k * x - 5) - 2 * x^2 + 8 ≠ 0) ∧
  (∀ (k' : ℤ), k' < k →
    ∃ (x : ℝ), 3 * x * (k' * x - 5) - 2 * x^2 + 8 = 0) :=
by sorry

end smallest_k_for_no_real_roots_l1834_183478


namespace marco_has_largest_number_l1834_183495

def ellen_final (start : ℕ) : ℕ :=
  ((start - 2) * 3) + 4

def marco_final (start : ℕ) : ℕ :=
  ((start * 3) - 3) + 5

def lucia_final (start : ℕ) : ℕ :=
  ((start - 3) + 5) * 3

theorem marco_has_largest_number :
  let ellen_start := 12
  let marco_start := 15
  let lucia_start := 13
  marco_final marco_start > ellen_final ellen_start ∧
  marco_final marco_start > lucia_final lucia_start :=
by sorry

end marco_has_largest_number_l1834_183495


namespace cannot_reach_54_from_12_l1834_183462

/-- Represents the possible operations that can be performed on the number -/
inductive Operation
  | MultiplyBy2
  | DivideBy2
  | MultiplyBy3
  | DivideBy3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.MultiplyBy2 => n * 2
  | Operation.DivideBy2 => n / 2
  | Operation.MultiplyBy3 => n * 3
  | Operation.DivideBy3 => n / 3

/-- Applies a sequence of operations to a number -/
def applyOperations (initial : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation initial

/-- Theorem stating that it's impossible to reach 54 from 12 after 60 operations -/
theorem cannot_reach_54_from_12 (ops : List Operation) :
  ops.length = 60 → applyOperations 12 ops ≠ 54 := by
  sorry


end cannot_reach_54_from_12_l1834_183462


namespace five_digit_square_theorem_l1834_183470

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def remove_first_digit (n : ℕ) : ℕ := n % 10000
def remove_first_two_digits (n : ℕ) : ℕ := n % 1000
def remove_first_three_digits (n : ℕ) : ℕ := n % 100

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  is_perfect_square n ∧
  is_perfect_square (remove_first_digit n) ∧
  is_perfect_square (remove_first_two_digits n) ∧
  is_perfect_square (remove_first_three_digits n)

theorem five_digit_square_theorem :
  {n : ℕ | is_valid_number n} = {81225, 34225, 27225, 15625, 75625} :=
by sorry

end five_digit_square_theorem_l1834_183470


namespace sin_315_degrees_l1834_183497

/-- Proves that sin 315° = -√2/2 -/
theorem sin_315_degrees : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_315_degrees_l1834_183497


namespace prob_ace_king_is_4_663_l1834_183406

/-- Represents a standard deck of cards. -/
structure Deck :=
  (total_cards : ℕ := 52)
  (num_aces : ℕ := 4)
  (num_kings : ℕ := 4)

/-- Calculates the probability of drawing an Ace first and a King second from a standard deck. -/
def prob_ace_then_king (d : Deck) : ℚ :=
  (d.num_aces : ℚ) / d.total_cards * d.num_kings / (d.total_cards - 1)

/-- Theorem stating the probability of drawing an Ace first and a King second from a standard deck. -/
theorem prob_ace_king_is_4_663 (d : Deck) : prob_ace_then_king d = 4 / 663 := by
  sorry

end prob_ace_king_is_4_663_l1834_183406


namespace arithmetic_mean_problem_l1834_183436

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 20 + 3*x + 17 + (2*x + 6) + (x + 24)) / 6 = 26 → x = 79/7 :=
by sorry

end arithmetic_mean_problem_l1834_183436


namespace find_A_l1834_183442

theorem find_A : ∃ A : ℝ, (∃ B : ℝ, 211.5 = B - A ∧ B = 10 * A) → A = 23.5 := by
  sorry

end find_A_l1834_183442


namespace player5_score_l1834_183454

/-- Represents a basketball player's score breakdown -/
structure PlayerScore where
  twoPointers : Nat
  threePointers : Nat
  freeThrows : Nat

/-- Calculates the total points scored by a player -/
def totalPoints (score : PlayerScore) : Nat :=
  2 * score.twoPointers + 3 * score.threePointers + score.freeThrows

theorem player5_score 
  (teamAScore : Nat)
  (player1 : PlayerScore)
  (player2 : PlayerScore)
  (player3 : PlayerScore)
  (player4 : PlayerScore)
  (h1 : teamAScore = 75)
  (h2 : player1 = ⟨0, 5, 0⟩)
  (h3 : player2 = ⟨5, 0, 5⟩)
  (h4 : player3 = ⟨0, 3, 3⟩)
  (h5 : player4 = ⟨6, 0, 0⟩) :
  teamAScore - (totalPoints player1 + totalPoints player2 + totalPoints player3 + totalPoints player4) = 14 := by
  sorry

#eval totalPoints ⟨0, 5, 0⟩  -- Player 1
#eval totalPoints ⟨5, 0, 5⟩  -- Player 2
#eval totalPoints ⟨0, 3, 3⟩  -- Player 3
#eval totalPoints ⟨6, 0, 0⟩  -- Player 4

end player5_score_l1834_183454


namespace C₂_function_l1834_183455

-- Define the original function f
variable (f : ℝ → ℝ)

-- Define C as the graph of y = f(x)
def C (f : ℝ → ℝ) : Set (ℝ × ℝ) := {(x, y) | y = f x}

-- Define C₁ as symmetric to C with respect to x = 1
def C₁ (f : ℝ → ℝ) : Set (ℝ × ℝ) := {(x, y) | y = f (2 - x)}

-- Define C₂ as C₁ shifted one unit to the left
def C₂ (f : ℝ → ℝ) : Set (ℝ × ℝ) := {(x, y) | ∃ x', x = x' - 1 ∧ (x', y) ∈ C₁ f}

-- Theorem: The function corresponding to C₂ is y = f(1 - x)
theorem C₂_function (f : ℝ → ℝ) : C₂ f = {(x, y) | y = f (1 - x)} := by sorry

end C₂_function_l1834_183455


namespace production_days_l1834_183444

theorem production_days (n : ℕ) 
  (h1 : (n * 50 + 115) / (n + 1) = 55) : n = 12 := by
  sorry

end production_days_l1834_183444


namespace gcd_of_three_numbers_l1834_183498

theorem gcd_of_three_numbers :
  Nat.gcd 13642 (Nat.gcd 19236 34176) = 2 := by
  sorry

end gcd_of_three_numbers_l1834_183498


namespace donut_theorem_l1834_183416

def donut_problem (initial : ℕ) (eaten : ℕ) (taken : ℕ) : ℕ :=
  let remaining_after_eaten := initial - eaten
  let remaining_after_taken := remaining_after_eaten - taken
  remaining_after_taken - remaining_after_taken / 2

theorem donut_theorem : donut_problem 50 2 4 = 22 := by
  sorry

end donut_theorem_l1834_183416


namespace equilateral_triangle_and_regular_pentagon_not_similar_l1834_183483

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a regular pentagon
structure RegularPentagon where
  side : ℝ
  side_positive : side > 0

-- Define similarity between shapes
def similar (shape1 shape2 : Type) : Prop := sorry

-- Theorem statement
theorem equilateral_triangle_and_regular_pentagon_not_similar :
  ∀ (t : EquilateralTriangle) (p : RegularPentagon), ¬(similar EquilateralTriangle RegularPentagon) :=
by
  sorry

end equilateral_triangle_and_regular_pentagon_not_similar_l1834_183483


namespace cyclists_problem_l1834_183459

/-- Two cyclists problem -/
theorem cyclists_problem (v₁ v₂ t : ℝ) :
  v₁ > 0 ∧ v₂ > 0 ∧ t > 0 ∧
  v₁ * t = v₂ * (1.5 : ℝ) ∧
  v₂ * t = v₁ * (2/3 : ℝ) →
  t = 1 ∧ v₁ / v₂ = 3/2 := by
  sorry

end cyclists_problem_l1834_183459


namespace min_k_for_reciprocal_like_l1834_183490

/-- A directed graph representing people liking each other in a group -/
structure LikeGraph where
  n : ℕ  -- number of people
  k : ℕ  -- number of people each person likes
  edges : Fin n → Finset (Fin n)
  outDegree : ∀ v, (edges v).card = k

/-- There exists a pair of people who like each other reciprocally -/
def hasReciprocalLike (g : LikeGraph) : Prop :=
  ∃ i j : Fin g.n, i ≠ j ∧ i ∈ g.edges j ∧ j ∈ g.edges i

/-- The minimum k that guarantees a reciprocal like in a group of 30 people -/
theorem min_k_for_reciprocal_like :
  ∀ k : ℕ, (∀ g : LikeGraph, g.n = 30 ∧ g.k = k → hasReciprocalLike g) ↔ k ≥ 15 :=
sorry

end min_k_for_reciprocal_like_l1834_183490


namespace perpendicular_parallel_implies_parallel_l1834_183491

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_parallel
  (α β : Plane) (m n : Line)
  (h_distinct_planes : α ≠ β)
  (h_distinct_lines : m ≠ n)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_β : perpendicular n β)
  (h_α_parallel_β : parallel α β) :
  parallel_lines m n :=
sorry

end perpendicular_parallel_implies_parallel_l1834_183491


namespace quadratic_point_on_graph_l1834_183430

/-- Given a quadratic function y = -ax² + 2ax + 3 where a > 0,
    if the point (m, 3) lies on the graph and m ≠ 0, then m = 2 -/
theorem quadratic_point_on_graph (a m : ℝ) (ha : a > 0) (hm : m ≠ 0) :
  (3 = -a * m^2 + 2 * a * m + 3) → m = 2 := by
  sorry

end quadratic_point_on_graph_l1834_183430


namespace x_value_when_y_is_two_l1834_183448

theorem x_value_when_y_is_two (x y : ℝ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
sorry

end x_value_when_y_is_two_l1834_183448


namespace min_sum_distances_l1834_183496

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define line l₁
def line_l₁ (x y : ℝ) : Prop := 4*x - 3*y + 6 = 0

-- Define line l₂
def line_l₂ (x : ℝ) : Prop := x = -1

-- Define the distance function from a point to a line
noncomputable def dist_point_to_line (px py : ℝ) (a b c : ℝ) : ℝ :=
  abs (a * px + b * py + c) / Real.sqrt (a^2 + b^2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem min_sum_distances :
  ∃ (d : ℝ), d = 2 ∧
  ∀ (px py : ℝ), parabola px py →
    d ≤ dist_point_to_line px py 4 (-3) 6 + abs (px + 1) :=
by sorry

end min_sum_distances_l1834_183496


namespace longest_side_of_triangle_l1834_183409

theorem longest_side_of_triangle (x : ℝ) : 
  7 + (x + 4) + (2 * x + 1) = 36 → 
  max 7 (max (x + 4) (2 * x + 1)) = 17 := by
sorry

end longest_side_of_triangle_l1834_183409


namespace person_age_l1834_183418

theorem person_age : ∃ x : ℕ, x = 30 ∧ 3 * (x + 5) - 3 * (x - 5) = x := by sorry

end person_age_l1834_183418


namespace arithmetic_calculations_l1834_183441

theorem arithmetic_calculations : 
  ((82 - 15) * (32 + 18) = 3350) ∧ ((25 + 4) * 75 = 2175) := by
  sorry

end arithmetic_calculations_l1834_183441


namespace company_merger_managers_percentage_l1834_183449

/-- Represents the percentage of managers in Company 2 -/
def m : ℝ := sorry

/-- The total number of employees in Company 1 -/
def F : ℝ := sorry

/-- The total number of employees in Company 2 -/
def S : ℝ := sorry

theorem company_merger_managers_percentage :
  (0.1 * F + m * S = 0.25 * (F + S)) ∧
  (F = 0.25 * (F + S)) ∧
  (0 < F) ∧ (0 < S) ∧
  (0 ≤ m) ∧ (m ≤ 1) ∧
  (m + 0.1 + 0.6 ≤ 1) →
  m = 0.225 := by sorry

end company_merger_managers_percentage_l1834_183449


namespace unique_six_digit_number_with_permutation_multiples_l1834_183485

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 6 ∧ digits.toFinset.card = 6

def is_permutation (a b : ℕ) : Prop :=
  (a.digits 10).toFinset = (b.digits 10).toFinset

theorem unique_six_digit_number_with_permutation_multiples :
  ∃! n : ℕ, is_six_digit n ∧ has_distinct_digits n ∧
    (∀ k : Fin 5, is_permutation n ((k + 2) * n)) ∧
    n = 142857 := by sorry

end unique_six_digit_number_with_permutation_multiples_l1834_183485


namespace arithmetic_progression_reciprocals_implies_squares_l1834_183411

theorem arithmetic_progression_reciprocals_implies_squares
  (a b c : ℝ)
  (h : ∃ (k : ℝ), (1 / (a + c)) - (1 / (a + b)) = k ∧ (1 / (b + c)) - (1 / (a + c)) = k) :
  ∃ (r : ℝ), b^2 - a^2 = r ∧ c^2 - b^2 = r :=
by sorry

end arithmetic_progression_reciprocals_implies_squares_l1834_183411


namespace class_average_theorem_l1834_183432

/-- Given a class with three groups of students, where:
    1. 25% of the class averages 80% on a test
    2. 50% of the class averages 65% on the test
    3. The remainder of the class averages 90% on the test
    Prove that the overall class average is 75% -/
theorem class_average_theorem (group1_proportion : Real) (group1_average : Real)
                              (group2_proportion : Real) (group2_average : Real)
                              (group3_proportion : Real) (group3_average : Real) :
  group1_proportion = 0.25 →
  group1_average = 0.80 →
  group2_proportion = 0.50 →
  group2_average = 0.65 →
  group3_proportion = 0.25 →
  group3_average = 0.90 →
  group1_proportion + group2_proportion + group3_proportion = 1 →
  group1_proportion * group1_average +
  group2_proportion * group2_average +
  group3_proportion * group3_average = 0.75 := by
  sorry


end class_average_theorem_l1834_183432


namespace fraction_equality_l1834_183461

theorem fraction_equality (x y : ℚ) (hx : x = 4/7) (hy : y = 5/11) : 
  (7*x + 11*y) / (77*x*y) = 9/20 := by
sorry

end fraction_equality_l1834_183461


namespace incorrect_step_identification_l1834_183417

theorem incorrect_step_identification :
  (2 * Real.sqrt 3 = Real.sqrt (2^2 * 3)) ∧
  (2 * Real.sqrt 3 ≠ -2 * Real.sqrt 3) ∧
  (Real.sqrt ((-2)^2 * 3) ≠ -2 * Real.sqrt 3) :=
by sorry

end incorrect_step_identification_l1834_183417


namespace homothety_containment_l1834_183401

/-- A convex polygon in R^2 -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ (convexHull ℝ vertices)

/-- Homothety transformation in R^2 -/
def homothety (center : ℝ × ℝ) (ratio : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (center.1 + ratio * (point.1 - center.1), center.2 + ratio * (point.2 - center.2))

/-- The image of a set under homothety -/
def homotheticImage (center : ℝ × ℝ) (ratio : ℝ) (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p | ∃ q ∈ s, p = homothety center ratio q}

theorem homothety_containment (P : ConvexPolygon) :
  ∃ O : ℝ × ℝ, homotheticImage O (-1/2) (convexHull ℝ P.vertices) ⊆ convexHull ℝ P.vertices :=
sorry

end homothety_containment_l1834_183401


namespace pencil_pen_cost_l1834_183438

/-- The cost of pencils and pens -/
theorem pencil_pen_cost (x y : ℚ) 
  (h1 : 8 * x + 3 * y = 5.1)
  (h2 : 3 * x + 5 * y = 4.95) :
  4 * x + 4 * y = 4.488 := by
  sorry

end pencil_pen_cost_l1834_183438


namespace average_speed_round_trip_l1834_183469

/-- Given a round trip with outbound speed of 96 mph and return speed of 88 mph,
    prove that the average speed for the entire trip is (2 * 96 * 88) / (96 + 88) mph. -/
theorem average_speed_round_trip (outbound_speed return_speed : ℝ) 
  (h1 : outbound_speed = 96) 
  (h2 : return_speed = 88) : 
  (2 * outbound_speed * return_speed) / (outbound_speed + return_speed) = 
  (2 * 96 * 88) / (96 + 88) :=
by sorry

end average_speed_round_trip_l1834_183469


namespace inequality_proof_l1834_183466

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 := by
  sorry

end inequality_proof_l1834_183466


namespace dish_price_theorem_l1834_183413

/-- The original price of a dish that satisfies the given conditions -/
def original_price : ℝ := 40

/-- John's total payment -/
def john_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * price

/-- Jane's total payment -/
def jane_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * (0.9 * price)

/-- Theorem stating that the original price satisfies the given conditions -/
theorem dish_price_theorem : 
  john_payment original_price = jane_payment original_price + 0.60 := by
  sorry

#eval original_price

end dish_price_theorem_l1834_183413


namespace binary_subtraction_example_l1834_183407

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0 -/
def BinaryNum := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : ℕ) : BinaryNum :=
  sorry

/-- Converts a binary number to its natural number representation -/
def fromBinary (b : BinaryNum) : ℕ :=
  sorry

/-- Performs binary subtraction -/
def binarySubtract (a b : BinaryNum) : BinaryNum :=
  sorry

theorem binary_subtraction_example :
  binarySubtract (toBinary 27) (toBinary 5) = toBinary 22 :=
sorry

end binary_subtraction_example_l1834_183407


namespace angle_measure_proof_l1834_183467

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 4 * x + 7) → x = 173 / 5 := by
  sorry

end angle_measure_proof_l1834_183467


namespace translation_of_segment_l1834_183429

/-- Translation of a point in 2D space -/
def translate (p q : ℝ × ℝ) : ℝ × ℝ := (p.1 + q.1, p.2 + q.2)

theorem translation_of_segment (A B C : ℝ × ℝ) :
  A = (-2, 5) →
  B = (-3, 0) →
  C = (3, 7) →
  translate A (5, 2) = C →
  translate B (5, 2) = (2, 2) := by
  sorry

end translation_of_segment_l1834_183429


namespace highest_percentage_increase_survey_d_l1834_183405

structure Survey where
  customers : ℕ
  responses : ℕ

def response_rate (s : Survey) : ℚ :=
  s.responses / s.customers

def percentage_change (a b : ℚ) : ℚ :=
  (b - a) / a * 100

theorem highest_percentage_increase_survey_d (survey_a survey_b survey_c survey_d : Survey)
  (ha : survey_a = { customers := 100, responses := 15 })
  (hb : survey_b = { customers := 120, responses := 27 })
  (hc : survey_c = { customers := 140, responses := 39 })
  (hd : survey_d = { customers := 160, responses := 56 }) :
  let change_ab := percentage_change (response_rate survey_a) (response_rate survey_b)
  let change_ac := percentage_change (response_rate survey_a) (response_rate survey_c)
  let change_ad := percentage_change (response_rate survey_a) (response_rate survey_d)
  change_ad > change_ab ∧ change_ad > change_ac := by
  sorry

end highest_percentage_increase_survey_d_l1834_183405


namespace total_chairs_l1834_183402

/-- Calculates the total number of chairs at a wedding. -/
theorem total_chairs (initial_rows : ℕ) (chairs_per_row : ℕ) (extra_chairs : ℕ) : 
  initial_rows = 7 → chairs_per_row = 12 → extra_chairs = 11 →
  initial_rows * chairs_per_row + extra_chairs = 95 := by
  sorry

end total_chairs_l1834_183402


namespace ellipse_y_axis_intersection_l1834_183424

/-- Definition of the ellipse with given foci and one intersection point -/
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (-1, 3)
  let F₂ : ℝ × ℝ := (4, 1)
  let P₁ : ℝ × ℝ := (0, 1)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 
  Real.sqrt ((P₁.1 - F₁.1)^2 + (P₁.2 - F₁.2)^2) + 
  Real.sqrt ((P₁.1 - F₂.1)^2 + (P₁.2 - F₂.2)^2)

/-- The theorem stating that (0, -2) is the other intersection point -/
theorem ellipse_y_axis_intersection :
  ∃ (y : ℝ), y ≠ 1 ∧ ellipse (0, y) ∧ y = -2 := by
  sorry

end ellipse_y_axis_intersection_l1834_183424


namespace homework_problem_count_l1834_183468

theorem homework_problem_count (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) : 
  math_pages = 2 → reading_pages = 4 → problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end homework_problem_count_l1834_183468


namespace function_value_at_symmetry_point_l1834_183471

theorem function_value_at_symmetry_point (ω φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.cos (ω * x + φ)
  (∀ x, f (π / 3 + x) = f (π / 3 - x)) →
  f (π / 3) = 3 ∨ f (π / 3) = -3 := by
  sorry

end function_value_at_symmetry_point_l1834_183471


namespace square_difference_l1834_183426

theorem square_difference (n m : ℕ+) (h : n * (4 * n + 1) = m * (5 * m + 1)) :
  ∃ k : ℕ+, n - m = k^2 := by sorry

end square_difference_l1834_183426


namespace parallel_vectors_m_value_l1834_183473

/-- Given two 2D vectors a and b, where a = (2, 1) and b = (m, -1),
    and a is parallel to b, prove that m = -2. -/
theorem parallel_vectors_m_value :
  ∀ (m : ℝ),
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![m, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, a i = k * b i)) →
  m = -2 :=
by sorry

end parallel_vectors_m_value_l1834_183473


namespace least_years_to_double_l1834_183439

def interest_rate : ℝ := 0.13

def more_than_doubled (years : ℕ) : Prop :=
  (1 + interest_rate) ^ years > 2

theorem least_years_to_double :
  (∀ y : ℕ, y < 6 → ¬(more_than_doubled y)) ∧ 
  more_than_doubled 6 :=
sorry

end least_years_to_double_l1834_183439


namespace binomial_coefficient_times_two_l1834_183492

theorem binomial_coefficient_times_two : 2 * (Nat.choose 30 3) = 8120 := by
  sorry

end binomial_coefficient_times_two_l1834_183492


namespace fractal_sequence_2000_and_sum_l1834_183400

/-- The fractal sequence a_n -/
def fractal_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 
    let k := (n + 1).log2
    if n + 1 = 2^k - 1 then k
    else fractal_sequence (n - 2^(k-1) + 1)

/-- Sum of the first n terms of the fractal sequence -/
def fractal_sum (n : ℕ) : ℕ :=
  (List.range n).map fractal_sequence |>.sum

theorem fractal_sequence_2000_and_sum :
  fractal_sequence 1999 = 2 ∧ fractal_sum 2000 = 4004 := by
  sorry

#eval fractal_sequence 1999
#eval fractal_sum 2000

end fractal_sequence_2000_and_sum_l1834_183400


namespace toy_shop_problem_l1834_183457

/-- Toy shop problem -/
theorem toy_shop_problem 
  (total_A : ℝ) (total_B : ℝ) (diff : ℕ) (ratio : ℝ) 
  (sell_A : ℝ) (sell_B : ℝ) (total_toys : ℕ) (min_profit : ℝ) :
  total_A = 1200 →
  total_B = 1500 →
  diff = 20 →
  ratio = 1.5 →
  sell_A = 12 →
  sell_B = 20 →
  total_toys = 75 →
  min_profit = 300 →
  ∃ (cost_A cost_B : ℝ) (max_A : ℕ),
    -- Part 1: Cost of toys
    cost_A = 10 ∧ 
    cost_B = 15 ∧
    total_A / cost_A - total_B / cost_B = diff ∧
    cost_B = ratio * cost_A ∧
    -- Part 2: Maximum number of type A toys
    max_A = 25 ∧
    ∀ m : ℕ, 
      m ≤ total_toys →
      (sell_A - cost_A) * m + (sell_B - cost_B) * (total_toys - m) ≥ min_profit →
      m ≤ max_A := by
  sorry

end toy_shop_problem_l1834_183457


namespace reaction_gibbs_free_energy_change_l1834_183477

/-- The standard Gibbs free energy of formation of NaOH in kJ/mol -/
def ΔG_f_NaOH : ℝ := -381.1

/-- The standard Gibbs free energy of formation of Na₂O in kJ/mol -/
def ΔG_f_Na2O : ℝ := -378

/-- The standard Gibbs free energy of formation of H₂O (liquid) in kJ/mol -/
def ΔG_f_H2O : ℝ := -237

/-- The temperature in Kelvin -/
def T : ℝ := 298

/-- 
The standard Gibbs free energy change (ΔG°₂₉₈) for the reaction Na₂O + H₂O → 2NaOH at 298 K
is equal to -147.2 kJ/mol, given the standard Gibbs free energies of formation for NaOH, Na₂O, and H₂O.
-/
theorem reaction_gibbs_free_energy_change : 
  2 * ΔG_f_NaOH - (ΔG_f_Na2O + ΔG_f_H2O) = -147.2 := by sorry

end reaction_gibbs_free_energy_change_l1834_183477


namespace town_population_problem_l1834_183425

theorem town_population_problem (p : ℕ) : 
  (p + 1500 : ℝ) * 0.8 = p + 1500 + 50 → p = 1750 := by
  sorry

end town_population_problem_l1834_183425


namespace selling_price_for_target_profit_l1834_183443

-- Define the cost price
def cost_price : ℝ := 40

-- Define the function for monthly sales volume based on selling price
def sales_volume (x : ℝ) : ℝ := 1000 - 10 * x

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_volume x

-- Theorem stating the selling prices that result in 8000 yuan profit
theorem selling_price_for_target_profit : 
  ∃ (x : ℝ), (x = 60 ∨ x = 80) ∧ profit x = 8000 := by
  sorry


end selling_price_for_target_profit_l1834_183443


namespace closest_integer_to_cube_root_200_l1834_183447

theorem closest_integer_to_cube_root_200 : 
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |m^3 - 200| ≥ |n^3 - 200| := by
  sorry

end closest_integer_to_cube_root_200_l1834_183447


namespace greatest_number_l1834_183414

def octal_to_decimal (n : ℕ) : ℕ := 3 * 8^1 + 2 * 8^0

def base5_to_decimal (n : ℕ) : ℕ := 1 * 5^2 + 1 * 5^1 + 1 * 5^0

def binary_to_decimal (n : ℕ) : ℕ := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0

def base6_to_decimal (n : ℕ) : ℕ := 5 * 6^1 + 4 * 6^0

theorem greatest_number : 
  binary_to_decimal 101010 > octal_to_decimal 32 ∧
  binary_to_decimal 101010 > base5_to_decimal 111 ∧
  binary_to_decimal 101010 > base6_to_decimal 54 := by
  sorry

end greatest_number_l1834_183414


namespace square_points_sum_l1834_183479

/-- A square with side length 900 and two points on one of its sides. -/
structure SquareWithPoints where
  /-- The side length of the square -/
  side_length : ℝ
  /-- The angle EOF in degrees -/
  angle_EOF : ℝ
  /-- The length of EF -/
  EF_length : ℝ
  /-- The distance BF expressed as p + q√r -/
  BF_distance : ℝ → ℝ → ℝ → ℝ
  /-- Condition that side_length is 900 -/
  h_side_length : side_length = 900
  /-- Condition that angle EOF is 45° -/
  h_angle_EOF : angle_EOF = 45
  /-- Condition that EF length is 400 -/
  h_EF_length : EF_length = 400
  /-- Condition that BF = p + q√r -/
  h_BF_distance : ∀ p q r, BF_distance p q r = p + q * Real.sqrt r

/-- The theorem stating that p + q + r = 307 for the given conditions -/
theorem square_points_sum (s : SquareWithPoints) (p q r : ℕ) 
  (h_positive : p > 0 ∧ q > 0 ∧ r > 0)
  (h_prime : ∀ (k : ℕ), k > 1 → k ^ 2 ∣ r → k.Prime → False) :
  p + q + r = 307 := by
  sorry

end square_points_sum_l1834_183479


namespace silver_car_percentage_l1834_183420

theorem silver_car_percentage (initial_cars : ℕ) (initial_silver_percent : ℚ)
  (new_cars : ℕ) (new_non_silver_percent : ℚ) :
  initial_cars = 40 →
  initial_silver_percent = 1/5 →
  new_cars = 80 →
  new_non_silver_percent = 1/2 →
  let total_cars := initial_cars + new_cars
  let initial_silver := initial_cars * initial_silver_percent
  let new_silver := new_cars * (1 - new_non_silver_percent)
  let total_silver := initial_silver + new_silver
  (total_silver / total_cars) = 2/5 := by
  sorry

end silver_car_percentage_l1834_183420


namespace specific_trapezoid_ratio_l1834_183460

/-- Represents a trapezoid with extended legs -/
structure ExtendedTrapezoid where
  -- Base lengths
  ab : ℝ
  cd : ℝ
  -- Height
  h : ℝ
  -- Condition that it's a valid trapezoid (cd > ab)
  h_valid : cd > ab

/-- The ratio of the area of triangle EAB to the area of trapezoid ABCD -/
def area_ratio (t : ExtendedTrapezoid) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem stating the ratio for the specific trapezoid in the problem -/
theorem specific_trapezoid_ratio :
  let t : ExtendedTrapezoid := ⟨5, 20, 12, by norm_num⟩
  area_ratio t = 1 / 15 := by
  sorry

end specific_trapezoid_ratio_l1834_183460


namespace lcm_gcd_product_24_36_l1834_183415

theorem lcm_gcd_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end lcm_gcd_product_24_36_l1834_183415


namespace exists_max_k_l1834_183431

theorem exists_max_k (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^3 * (x^2/y^2 + y^2/x^2) + k^2 * (x/y + y/x)) :
  ∃ k_max : ℝ, k ≤ k_max ∧
    ∀ k' : ℝ, k' > 0 → 
      (∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ 
        5 = k'^3 * (x'^2/y'^2 + y'^2/x'^2) + k'^2 * (x'/y' + y'/x')) →
      k' ≤ k_max :=
sorry

end exists_max_k_l1834_183431


namespace annual_music_cost_l1834_183404

/-- Calculates the annual cost of music for John given his monthly music consumption, average song length, and price per song. -/
theorem annual_music_cost 
  (monthly_hours : ℕ) 
  (song_length_minutes : ℕ) 
  (price_per_song : ℚ) : 
  monthly_hours = 20 → 
  song_length_minutes = 3 → 
  price_per_song = 1/2 → 
  (monthly_hours * 60 / song_length_minutes) * price_per_song * 12 = 2400 := by
sorry

end annual_music_cost_l1834_183404


namespace min_distance_after_11_hours_l1834_183412

/-- Represents the turtle's movement on a 2D plane -/
structure TurtleMovement where
  speed : ℝ
  duration : ℕ

/-- Calculates the minimum possible distance from the starting point -/
def minDistanceFromStart (movement : TurtleMovement) : ℝ :=
  sorry

/-- Theorem stating the minimum distance for the given conditions -/
theorem min_distance_after_11_hours :
  let movement : TurtleMovement := ⟨5, 11⟩
  minDistanceFromStart movement = 5 := by
  sorry

end min_distance_after_11_hours_l1834_183412


namespace magazine_purchasing_methods_l1834_183487

/-- Represents the number of magazine types priced at 2 yuan -/
def magazines_2yuan : ℕ := 8

/-- Represents the number of magazine types priced at 1 yuan -/
def magazines_1yuan : ℕ := 3

/-- Represents the total amount spent -/
def total_spent : ℕ := 10

/-- Calculates the number of ways to buy magazines -/
def number_of_ways : ℕ := 
  Nat.choose magazines_2yuan 5 + 
  Nat.choose magazines_2yuan 4 * Nat.choose magazines_1yuan 2

theorem magazine_purchasing_methods :
  number_of_ways = 266 := by sorry

end magazine_purchasing_methods_l1834_183487


namespace sequence_length_l1834_183423

theorem sequence_length (m : ℕ+) (a : ℕ → ℝ) 
  (h0 : a 0 = 37)
  (h1 : a 1 = 72)
  (hm : a m = 0)
  (h_rec : ∀ k ∈ Finset.range (m - 1), a (k + 2) = a k - 3 / a (k + 1)) :
  m = 889 := by
  sorry

end sequence_length_l1834_183423


namespace class_gpa_calculation_l1834_183493

/-- Calculates the overall GPA of a class given the number of students and their GPAs in three groups -/
def overall_gpa (total_students : ℕ) (group1_students : ℕ) (group1_gpa : ℚ) 
                (group2_students : ℕ) (group2_gpa : ℚ)
                (group3_students : ℕ) (group3_gpa : ℚ) : ℚ :=
  (group1_students * group1_gpa + group2_students * group2_gpa + group3_students * group3_gpa) / total_students

/-- Theorem stating that the overall GPA of the class is 1030/60 -/
theorem class_gpa_calculation :
  overall_gpa 60 20 15 15 17 25 19 = 1030 / 60 := by
  sorry

#eval overall_gpa 60 20 15 15 17 25 19

end class_gpa_calculation_l1834_183493


namespace van_speed_problem_l1834_183484

theorem van_speed_problem (distance : ℝ) (original_time : ℝ) (time_factor : ℝ) 
  (h1 : distance = 600)
  (h2 : original_time = 5)
  (h3 : time_factor = 3 / 2) :
  distance / (original_time * time_factor) = 80 := by
sorry

end van_speed_problem_l1834_183484


namespace unique_congruence_in_range_l1834_183481

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end unique_congruence_in_range_l1834_183481


namespace vector_sum_problem_l1834_183408

theorem vector_sum_problem (x y : ℝ) : 
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (2, y)
  (a.1 + b.1, a.2 + b.2) = (1, -1) → x + y = -3 := by
  sorry

end vector_sum_problem_l1834_183408


namespace negation_equivalence_l1834_183419

theorem negation_equivalence :
  (¬ (∀ x : ℝ, x^2 + x + 1 > 0)) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end negation_equivalence_l1834_183419


namespace books_sold_l1834_183480

theorem books_sold (initial_books : ℕ) (added_books : ℕ) (final_books : ℕ) : 
  initial_books = 4 → added_books = 10 → final_books = 11 → 
  initial_books - (final_books - added_books) = 3 :=
by
  sorry

end books_sold_l1834_183480


namespace school_books_count_l1834_183482

def total_books : ℕ := 58
def sports_books : ℕ := 39

theorem school_books_count : total_books - sports_books = 19 := by
  sorry

end school_books_count_l1834_183482


namespace tony_remaining_money_l1834_183476

def initial_money : ℕ := 20
def ticket_cost : ℕ := 8
def hotdog_cost : ℕ := 3

theorem tony_remaining_money :
  initial_money - ticket_cost - hotdog_cost = 9 := by
  sorry

end tony_remaining_money_l1834_183476


namespace skew_lines_distance_in_isosceles_triangle_sphere_setup_l1834_183465

/-- Given an isosceles triangle ABC on plane P with two skew lines passing through A and C,
    tangent to a sphere touching P at B, prove the distance between the lines. -/
theorem skew_lines_distance_in_isosceles_triangle_sphere_setup
  (l a r α : ℝ)
  (hl : l > 0)
  (ha : a > 0)
  (hr : r > 0)
  (hα : 0 < α ∧ α < π / 2)
  (h_isosceles : 2 * a ≤ l) :
  ∃ x : ℝ, x = (2 * a * Real.tan α * Real.sqrt (2 * r * l * Real.sin α - (l^2 + r^2) * Real.sin α^2)) /
              Real.sqrt (l^2 - a^2 * Real.cos α^2) :=
by sorry

end skew_lines_distance_in_isosceles_triangle_sphere_setup_l1834_183465


namespace man_half_father_age_l1834_183463

/-- Prove that the number of years it takes for a man to become half his father's age is 5 -/
theorem man_half_father_age (father_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  father_age = 25 →
  man_age = (2 * father_age) / 5 →
  man_age + years = (father_age + years) / 2 →
  years = 5 := by sorry

end man_half_father_age_l1834_183463


namespace circle_equation_l1834_183456

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y + 1 = 0
def y_axis (x : ℝ) : Prop := x = 0

-- State the theorem
theorem circle_equation (C : Circle) : 
  (∃ x y : ℝ, line1 x y ∧ y_axis x ∧ C.center = (x, y)) →  -- Center condition
  (∃ x y : ℝ, line2 x y ∧ (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2) →  -- Tangent condition
  ∀ x y : ℝ, (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2 ↔ x^2 + (y-1)^2 = 2 :=
by sorry


end circle_equation_l1834_183456


namespace green_blue_difference_after_double_border_l1834_183486

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexagonalFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Calculates the number of tiles in a single border layer of a hexagon -/
def border_layer_tiles (layer : ℕ) : ℕ :=
  6 * (2 * layer + 1)

/-- Adds a double border of green tiles to a hexagonal figure -/
def add_double_border (figure : HexagonalFigure) : HexagonalFigure :=
  { blue_tiles := figure.blue_tiles,
    green_tiles := figure.green_tiles + border_layer_tiles 1 + border_layer_tiles 2 }

/-- Theorem: The difference between green and blue tiles after adding a double border is 50 -/
theorem green_blue_difference_after_double_border (initial_figure : HexagonalFigure)
    (h_blue : initial_figure.blue_tiles = 20)
    (h_green : initial_figure.green_tiles = 10) :
    let final_figure := add_double_border initial_figure
    final_figure.green_tiles - final_figure.blue_tiles = 50 := by
  sorry


end green_blue_difference_after_double_border_l1834_183486


namespace sale_price_equals_original_l1834_183440

theorem sale_price_equals_original (x : ℝ) : x > 0 → 0.8 * (1.25 * x) = x := by
  sorry

end sale_price_equals_original_l1834_183440


namespace min_max_sum_l1834_183494

theorem min_max_sum (a b c d e f : ℕ+) (h : a + b + c + d + e + f = 1800) :
  361 ≤ max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) := by
  sorry

end min_max_sum_l1834_183494
