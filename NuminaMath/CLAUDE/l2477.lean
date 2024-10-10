import Mathlib

namespace geometric_sequence_cosine_l2477_247742

open Real

theorem geometric_sequence_cosine (a : ℝ) : 
  0 < a → a < 2 * π → 
  (∃ r : ℝ, cos a * r = cos (2 * a) ∧ cos (2 * a) * r = cos (3 * a)) → 
  a = π := by
sorry

end geometric_sequence_cosine_l2477_247742


namespace exam_total_questions_l2477_247721

/-- Represents an exam with given parameters -/
structure Exam where
  totalTime : ℕ
  answeredQuestions : ℕ
  timeUsed : ℕ
  timeLeftWhenFinished : ℕ

/-- Calculates the total number of questions on the exam -/
def totalQuestions (e : Exam) : ℕ :=
  let remainingTime := e.totalTime - e.timeUsed
  let questionRate := e.answeredQuestions / e.timeUsed
  e.answeredQuestions + questionRate * remainingTime

/-- Theorem stating that the total number of questions on the given exam is 80 -/
theorem exam_total_questions :
  let e : Exam := {
    totalTime := 60,
    answeredQuestions := 16,
    timeUsed := 12,
    timeLeftWhenFinished := 0
  }
  totalQuestions e = 80 := by
  sorry


end exam_total_questions_l2477_247721


namespace smallest_solution_congruence_system_l2477_247719

theorem smallest_solution_congruence_system (x : ℕ) : x = 1309 ↔ 
  (x > 0) ∧
  (3 * x ≡ 9 [MOD 12]) ∧ 
  (5 * x + 4 ≡ 14 [MOD 7]) ∧ 
  (4 * x - 3 ≡ 2 * x + 5 [MOD 17]) ∧ 
  (x ≡ 4 [MOD 11]) ∧
  (∀ y : ℕ, y > 0 → 
    (3 * y ≡ 9 [MOD 12]) → 
    (5 * y + 4 ≡ 14 [MOD 7]) → 
    (4 * y - 3 ≡ 2 * y + 5 [MOD 17]) → 
    (y ≡ 4 [MOD 11]) → 
    y ≥ x) :=
sorry

end smallest_solution_congruence_system_l2477_247719


namespace exists_number_not_exceeding_kr_l2477_247798

/-- The operation that replaces a number with two new numbers -/
def replace_operation (x : ℝ) : ℝ × ℝ :=
  sorry

/-- Perform the operation k^2 - 1 times -/
def perform_operations (r : ℝ) (k : ℕ) : List ℝ :=
  sorry

theorem exists_number_not_exceeding_kr (r : ℝ) (k : ℕ) (h_r : r > 0) :
  ∃ x ∈ perform_operations r k, x ≤ k * r :=
sorry

end exists_number_not_exceeding_kr_l2477_247798


namespace kylie_and_nelly_stamps_l2477_247779

/-- Given that Kylie has 34 stamps and Nelly has 44 more stamps than Kylie,
    prove that they have 112 stamps together. -/
theorem kylie_and_nelly_stamps :
  let kylie_stamps : ℕ := 34
  let nelly_stamps : ℕ := kylie_stamps + 44
  kylie_stamps + nelly_stamps = 112 := by sorry

end kylie_and_nelly_stamps_l2477_247779


namespace georges_socks_l2477_247759

/-- The number of socks George's dad gave him -/
def socks_from_dad (initial_socks bought_socks total_socks : ℝ) : ℝ :=
  total_socks - (initial_socks + bought_socks)

/-- Proof that George's dad gave him 4 socks -/
theorem georges_socks : socks_from_dad 28 36 68 = 4 := by
  sorry

end georges_socks_l2477_247759


namespace cot_thirty_degrees_l2477_247760

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end cot_thirty_degrees_l2477_247760


namespace new_person_weight_l2477_247795

/-- Given a group of 8 people, prove that when one person weighing 65 kg is replaced
    by a new person, and the average weight increases by 2.5 kg,
    the weight of the new person is 85 kg. -/
theorem new_person_weight (initial_average : ℝ) : 
  let num_people : ℕ := 8
  let weight_increase : ℝ := 2.5
  let old_person_weight : ℝ := 65
  let new_average : ℝ := initial_average + weight_increase
  let new_person_weight : ℝ := old_person_weight + (num_people * weight_increase)
  new_person_weight = 85 := by
sorry

end new_person_weight_l2477_247795


namespace largest_multiple_of_9_under_120_l2477_247723

theorem largest_multiple_of_9_under_120 : 
  ∃ n : ℕ, n * 9 = 117 ∧ 117 < 120 ∧ ∀ m : ℕ, m * 9 < 120 → m * 9 ≤ 117 :=
sorry

end largest_multiple_of_9_under_120_l2477_247723


namespace complex_root_implies_positive_triangle_l2477_247744

theorem complex_root_implies_positive_triangle (a b c α β : ℝ) :
  α > 0 →
  β ≠ 0 →
  Complex.I ^ 2 = -1 →
  (α + β * Complex.I) ^ 2 - (a + b + c) * (α + β * Complex.I) + (a * b + b * c + c * a) = 0 →
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  Real.sqrt a + Real.sqrt b > Real.sqrt c ∧
  Real.sqrt b + Real.sqrt c > Real.sqrt a ∧
  Real.sqrt c + Real.sqrt a > Real.sqrt b :=
by sorry

end complex_root_implies_positive_triangle_l2477_247744


namespace bd_squared_equals_25_l2477_247767

theorem bd_squared_equals_25 
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3)
  (h3 : 2*a - 3*b + c + 4*d = 17)
  : (b - d)^2 = 25 := by
  sorry

end bd_squared_equals_25_l2477_247767


namespace problem_statement_l2477_247713

theorem problem_statement (x y : ℝ) (h : |x - 1/2| + Real.sqrt (y^2 - 1) = 0) : 
  |x| + |y| = 3/2 := by
sorry

end problem_statement_l2477_247713


namespace janet_oranges_l2477_247757

theorem janet_oranges (sharon_oranges : ℕ) (total_oranges : ℕ) (h1 : sharon_oranges = 7) (h2 : total_oranges = 16) :
  total_oranges - sharon_oranges = 9 :=
by sorry

end janet_oranges_l2477_247757


namespace percentage_relationship_l2477_247780

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 28.57142857142857 / 100)) :
  y = x * (1 + 28.57142857142857 / 100) :=
by sorry

end percentage_relationship_l2477_247780


namespace painted_square_ratio_l2477_247730

/-- Given a square with side length s and a brush of width w, 
    if the painted area along the midline and one diagonal is one-third of the square's area, 
    then the ratio s/w equals 2√2 + 1 -/
theorem painted_square_ratio (s w : ℝ) (h_positive_s : 0 < s) (h_positive_w : 0 < w) :
  s * w + 2 * (1/2 * ((s * Real.sqrt 2) / 2 - (w * Real.sqrt 2) / 2)^2) = s^2 / 3 →
  s / w = 2 * Real.sqrt 2 + 1 := by
  sorry

end painted_square_ratio_l2477_247730


namespace smallest_n_for_red_vertices_symmetry_l2477_247700

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A set of 5 red vertices in a regular polygon -/
def RedVertices (n : ℕ) := Fin 5 → Fin n

/-- An axis of symmetry of a regular polygon -/
def AxisOfSymmetry (n : ℕ) := ℕ

/-- Checks if a vertex is reflected onto another vertex across an axis -/
def isReflectedOnto (n : ℕ) (p : RegularPolygon n) (v1 v2 : Fin n) (axis : AxisOfSymmetry n) : Prop :=
  sorry

/-- The main theorem -/
theorem smallest_n_for_red_vertices_symmetry :
  (∀ n : ℕ, n ≥ 14 →
    ∀ p : RegularPolygon n,
    ∀ red : RedVertices n,
    ∃ axis : AxisOfSymmetry n,
    ∀ v1 v2 : Fin 5, v1 ≠ v2 → ¬isReflectedOnto n p (red v1) (red v2) axis) ∧
  (∀ n : ℕ, n < 14 →
    ∃ p : RegularPolygon n,
    ∃ red : RedVertices n,
    ∀ axis : AxisOfSymmetry n,
    ∃ v1 v2 : Fin 5, v1 ≠ v2 ∧ isReflectedOnto n p (red v1) (red v2) axis) :=
sorry

end smallest_n_for_red_vertices_symmetry_l2477_247700


namespace years_of_writing_comics_l2477_247755

/-- Represents the number of comics written in a year -/
def comics_per_year : ℕ := 182

/-- Represents the total number of comics written -/
def total_comics : ℕ := 730

/-- Theorem: Given the conditions, the number of years of writing comics is 4 -/
theorem years_of_writing_comics : 
  (total_comics / comics_per_year : ℕ) = 4 := by sorry

end years_of_writing_comics_l2477_247755


namespace inequality_solution_l2477_247702

theorem inequality_solution (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 ↔ x = 14 / 5 := by
  sorry

end inequality_solution_l2477_247702


namespace student_travel_fraction_l2477_247718

theorem student_travel_fraction (total_distance : ℝ) 
  (bus_fraction : ℝ) (car_distance : ℝ) :
  total_distance = 90 ∧ 
  bus_fraction = 2/3 ∧ 
  car_distance = 12 →
  (total_distance - (bus_fraction * total_distance + car_distance)) / total_distance = 1/5 := by
  sorry

end student_travel_fraction_l2477_247718


namespace find_t_l2477_247710

def A (t : ℝ) : Set ℝ := {-4, t^2}
def B (t : ℝ) : Set ℝ := {t-5, 9, 1-t}

theorem find_t : ∀ t : ℝ, 9 ∈ A t ∩ B t → t = -3 := by sorry

end find_t_l2477_247710


namespace library_books_theorem_l2477_247704

/-- The number of books taken by the librarian -/
def books_taken : ℕ := 10

/-- The number of books that can fit on each shelf -/
def books_per_shelf : ℕ := 4

/-- The number of shelves needed for the remaining books -/
def shelves_needed : ℕ := 9

/-- The total number of books to put away -/
def total_books : ℕ := 46

theorem library_books_theorem :
  total_books = books_per_shelf * shelves_needed + books_taken := by
  sorry

end library_books_theorem_l2477_247704


namespace simplify_expression_l2477_247761

theorem simplify_expression (a : ℝ) : a^2 * (-a)^4 = a^6 := by
  sorry

end simplify_expression_l2477_247761


namespace right_triangle_perimeter_l2477_247793

/-- A right triangle with one leg of prime length n and other sides of natural number lengths has perimeter n + n^2 -/
theorem right_triangle_perimeter (n : ℕ) (h_prime : Nat.Prime n) :
  ∃ (x y : ℕ), x^2 + n^2 = y^2 ∧ x + y + n = n + n^2 := by
  sorry

end right_triangle_perimeter_l2477_247793


namespace unique_quadratic_solution_positive_m_for_unique_solution_l2477_247772

theorem unique_quadratic_solution (m : ℝ) :
  (∃! x : ℝ, 16 * x^2 + m * x + 4 = 0) ↔ m = 16 ∨ m = -16 :=
by sorry

theorem positive_m_for_unique_solution :
  ∃ m : ℝ, m > 0 ∧ (∃! x : ℝ, 16 * x^2 + m * x + 4 = 0) ∧ m = 16 :=
by sorry

end unique_quadratic_solution_positive_m_for_unique_solution_l2477_247772


namespace sample_size_l2477_247778

theorem sample_size (n : ℕ) (f₁ f₂ f₃ f₄ f₅ f₆ : ℕ) : 
  f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = n →
  f₁ = 2 * (f₆) →
  f₂ = 3 * (f₆) →
  f₃ = 4 * (f₆) →
  f₄ = 6 * (f₆) →
  f₅ = 4 * (f₆) →
  f₁ + f₂ + f₃ = 27 →
  n = 60 := by
sorry

end sample_size_l2477_247778


namespace room_length_calculation_l2477_247768

theorem room_length_calculation (breadth height pole_length : ℝ) 
  (h1 : breadth = 8)
  (h2 : height = 9)
  (h3 : pole_length = 17) : 
  ∃ length : ℝ, length^2 + breadth^2 + height^2 = pole_length^2 ∧ length = 12 := by
  sorry

end room_length_calculation_l2477_247768


namespace stamp_problem_l2477_247756

theorem stamp_problem (x y : ℕ) : 
  (x + y > 400) →
  (∃ k : ℕ, x - k = (13 : ℚ) / 19 * (y + k)) →
  (∃ k : ℕ, y - k = (11 : ℚ) / 17 * (x + k)) →
  x = 227 ∧ y = 221 :=
by sorry

end stamp_problem_l2477_247756


namespace range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l2477_247750

def proposition_p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0 ∧ a > 0

def proposition_q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, proposition_p x 1 ∧ proposition_q x → 2 < x ∧ x < 3 :=
sorry

theorem range_of_a_when_not_p_implies_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬(proposition_p x a) → ¬(proposition_q x)) →
  1 < a ∧ a ≤ 2 :=
sorry

end range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l2477_247750


namespace game_of_thrones_percentage_l2477_247708

/-- Represents the vote counts for each book --/
structure VoteCounts where
  gameOfThrones : ℕ
  twilight : ℕ
  artOfTheDeal : ℕ

/-- Calculates the altered vote counts after tampering --/
def alteredVotes (original : VoteCounts) : VoteCounts :=
  { gameOfThrones := original.gameOfThrones,
    twilight := original.twilight / 2,
    artOfTheDeal := original.artOfTheDeal / 5 }

/-- Calculates the total number of altered votes --/
def totalAlteredVotes (altered : VoteCounts) : ℕ :=
  altered.gameOfThrones + altered.twilight + altered.artOfTheDeal

/-- Theorem: The percentage of altered votes for Game of Thrones is 50% --/
theorem game_of_thrones_percentage (original : VoteCounts)
  (h1 : original.gameOfThrones = 10)
  (h2 : original.twilight = 12)
  (h3 : original.artOfTheDeal = 20) :
  (alteredVotes original).gameOfThrones * 100 / (totalAlteredVotes (alteredVotes original)) = 50 := by
  sorry


end game_of_thrones_percentage_l2477_247708


namespace negative_one_squared_and_one_are_opposite_l2477_247796

-- Define opposite numbers
def are_opposite (a b : ℤ) : Prop := a + b = 0

-- Theorem statement
theorem negative_one_squared_and_one_are_opposite : 
  are_opposite (-(1^2)) 1 := by sorry

end negative_one_squared_and_one_are_opposite_l2477_247796


namespace hotel_rooms_l2477_247751

theorem hotel_rooms (total_floors : Nat) (unavailable_floors : Nat) (available_rooms : Nat) :
  total_floors = 10 →
  unavailable_floors = 1 →
  available_rooms = 90 →
  (total_floors - unavailable_floors) * (available_rooms / (total_floors - unavailable_floors)) = available_rooms :=
by
  sorry

end hotel_rooms_l2477_247751


namespace arithmetic_sequence_seventh_term_l2477_247762

/-- An arithmetic sequence with the given properties has its 7th term equal to 19 -/
theorem arithmetic_sequence_seventh_term (n : ℕ) (a d : ℚ) 
  (h1 : n > 7)
  (h2 : 5 * a + 10 * d = 34)
  (h3 : 5 * a + 5 * (n - 1) * d = 146)
  (h4 : n * (2 * a + (n - 1) * d) / 2 = 234) :
  a + 6 * d = 19 := by
  sorry

end arithmetic_sequence_seventh_term_l2477_247762


namespace race_distance_l2477_247781

/-- The race problem -/
theorem race_distance (a_time b_time : ℕ) (beat_distance : ℕ) (total_distance : ℕ) : 
  a_time = 36 →
  b_time = 45 →
  beat_distance = 20 →
  (total_distance : ℚ) / a_time * b_time = total_distance + beat_distance →
  total_distance = 80 := by
  sorry

end race_distance_l2477_247781


namespace correct_sunset_time_l2477_247711

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the sunset time given sunrise time and daylight duration -/
def calculateSunset (sunrise : Time) (daylight : Duration) : Time :=
  { hours := (sunrise.hours + daylight.hours + (sunrise.minutes + daylight.minutes) / 60) % 24,
    minutes := (sunrise.minutes + daylight.minutes) % 60 }

theorem correct_sunset_time :
  let sunrise : Time := { hours := 7, minutes := 12 }
  let daylight : Duration := { hours := 9, minutes := 45 }
  let calculated_sunset : Time := calculateSunset sunrise daylight
  calculated_sunset = { hours := 16, minutes := 57 } :=
by sorry

end correct_sunset_time_l2477_247711


namespace apples_in_baskets_l2477_247705

theorem apples_in_baskets (num_baskets : ℕ) (total_apples : ℕ) (apples_per_basket : ℕ) :
  num_baskets = 37 →
  total_apples = 629 →
  num_baskets * apples_per_basket = total_apples →
  apples_per_basket = 17 := by
  sorry

end apples_in_baskets_l2477_247705


namespace ellipse_equation_l2477_247797

/-- Given an ellipse with equation x²/a² + y²/b² = 1 (a > 0, b > 0),
    if the line 2x + y - 2 = 0 passes through its upper vertex and right focus,
    then the equation of the ellipse is x²/5 + y²/4 = 1. -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ 2*x + y = 2 ∧
   ((x = a ∧ y = 0) ∨ (x = 0 ∧ y = b))) →
  a^2 = 5 ∧ b^2 = 4 :=
by sorry

end ellipse_equation_l2477_247797


namespace arithmetic_geometric_sequence_sum_l2477_247747

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℝ

/-- Sum of the first n terms of a sequence -/
def SumOfFirstNTerms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem arithmetic_geometric_sequence_sum (a : ArithmeticGeometricSequence) :
  let S := SumOfFirstNTerms a.a
  S 2 = 3 ∧ S 4 = 15 → S 6 = 63 := by
  sorry

end arithmetic_geometric_sequence_sum_l2477_247747


namespace sequence_bounded_l2477_247763

/-- A sequence of non-negative real numbers satisfying certain conditions is bounded -/
theorem sequence_bounded (c : ℝ) (a : ℕ → ℝ) 
  (hc : c > 2)
  (ha_nonneg : ∀ n, a n ≥ 0)
  (h1 : ∀ m n : ℕ, a (m + n) ≤ 2 * a m + 2 * a n)
  (h2 : ∀ k : ℕ, a (2^k) ≤ 1 / ((k + 1 : ℝ)^c)) :
  ∃ M : ℝ, ∀ n : ℕ, a n ≤ M :=
sorry

end sequence_bounded_l2477_247763


namespace smallest_multiple_of_8_no_repeated_digits_remainder_l2477_247731

/-- A function that checks if a natural number has no repeated digits -/
def hasNoRepeatedDigits (n : ℕ) : Prop := sorry

/-- The smallest multiple of 8 with no repeated digits -/
def M : ℕ := sorry

theorem smallest_multiple_of_8_no_repeated_digits_remainder :
  (M % 1000 = 120) ∧
  (∀ k : ℕ, k < M → (k % 8 = 0 → ¬hasNoRepeatedDigits k)) :=
sorry

end smallest_multiple_of_8_no_repeated_digits_remainder_l2477_247731


namespace item_distribution_l2477_247752

theorem item_distribution (n₁ n₂ n₃ k : ℕ) (h₁ : n₁ = 5) (h₂ : n₂ = 3) (h₃ : n₃ = 2) (h₄ : k = 3) :
  (Nat.choose (n₁ + k - 1) (k - 1)) * (Nat.choose (n₂ + k - 1) (k - 1)) * (Nat.choose (n₃ + k - 1) (k - 1)) = 1260 :=
by sorry

end item_distribution_l2477_247752


namespace solution_set_quadratic_inequality_l2477_247758

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 5*x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

end solution_set_quadratic_inequality_l2477_247758


namespace intercept_ratio_l2477_247777

theorem intercept_ratio (b s t : ℝ) (hb : b ≠ 0) : 
  0 = 8 * s + b ∧ 0 = 4 * t + b → s / t = 1 / 2 := by
sorry

end intercept_ratio_l2477_247777


namespace b_invests_after_six_months_l2477_247794

/-- A partnership model with three partners --/
structure Partnership where
  x : ℝ  -- A's investment
  m : ℝ  -- Months after which B invests
  total_gain : ℝ  -- Total annual gain
  a_share : ℝ  -- A's share of the gain

/-- The investment-time products for each partner --/
def investment_time (p : Partnership) : ℝ × ℝ × ℝ :=
  (p.x * 12, 2 * p.x * (12 - p.m), 3 * p.x * 4)

/-- The total investment-time product --/
def total_investment_time (p : Partnership) : ℝ :=
  let (a, b, c) := investment_time p
  a + b + c

/-- Theorem stating that B invests after 6 months --/
theorem b_invests_after_six_months (p : Partnership) 
  (h1 : p.total_gain = 12000)
  (h2 : p.a_share = 4000)
  (h3 : p.a_share / p.total_gain = 1 / 3)
  (h4 : p.x * 12 = (1 / 3) * total_investment_time p) :
  p.m = 6 := by
  sorry


end b_invests_after_six_months_l2477_247794


namespace arithmetic_geometric_ratio_l2477_247707

/-- Given an arithmetic sequence with a non-zero common difference,
    if its second, third, and sixth terms form a geometric sequence,
    then the common ratio of this geometric sequence is 3. -/
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) -- The arithmetic sequence
  (d : ℝ) -- The common difference of the arithmetic sequence
  (h1 : d ≠ 0) -- The common difference is non-zero
  (h2 : ∀ n, a (n + 1) = a n + d) -- Definition of arithmetic sequence
  (h3 : ∃ r, r ≠ 0 ∧ a 3 = r * a 2 ∧ a 6 = r * a 3) -- Second, third, and sixth terms form a geometric sequence
  : ∃ r, r = 3 ∧ a 3 = r * a 2 ∧ a 6 = r * a 3 := by
  sorry

end arithmetic_geometric_ratio_l2477_247707


namespace sin_180_degrees_l2477_247714

theorem sin_180_degrees : Real.sin (π) = 0 := by
  sorry

end sin_180_degrees_l2477_247714


namespace lindas_tv_cost_l2477_247727

/-- The cost of Linda's TV purchase, given her original savings and the fraction spent on furniture. -/
theorem lindas_tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 1200 → 
  furniture_fraction = 3/4 → 
  tv_cost = savings * (1 - furniture_fraction) → 
  tv_cost = 300 := by
  sorry

end lindas_tv_cost_l2477_247727


namespace painting_theorem_l2477_247753

/-- Represents the portion of a wall painted in a given time -/
def paint_portion (rate : ℚ) (time : ℚ) : ℚ := rate * time

/-- The combined painting rate of two painters -/
def combined_rate (rate1 : ℚ) (rate2 : ℚ) : ℚ := rate1 + rate2

theorem painting_theorem (heidi_rate liam_rate : ℚ) 
  (h1 : heidi_rate = 1 / 60)
  (h2 : liam_rate = 1 / 90)
  (time : ℚ)
  (h3 : time = 15) :
  paint_portion (combined_rate heidi_rate liam_rate) time = 5 / 12 := by
  sorry

end painting_theorem_l2477_247753


namespace house_number_painting_cost_l2477_247735

/-- Calculates the sum of digits for numbers in an arithmetic sequence --/
def sumOfDigits (start : ℕ) (diff : ℕ) (count : ℕ) : ℕ :=
  sorry

/-- Calculates the total cost of painting house numbers --/
def totalCost (southStart southDiff northStart northDiff housesPerSide : ℕ) : ℕ :=
  sorry

theorem house_number_painting_cost :
  totalCost 5 7 7 8 25 = 125 :=
sorry

end house_number_painting_cost_l2477_247735


namespace expression_equals_six_l2477_247706

theorem expression_equals_six :
  2 - Real.sqrt 3 + (2 - Real.sqrt 3)⁻¹ + (Real.sqrt 3 + 2)⁻¹ = 6 := by
  sorry

end expression_equals_six_l2477_247706


namespace psychologist_pricing_l2477_247703

theorem psychologist_pricing (F A : ℝ) 
  (h1 : F + 4 * A = 300)  -- 5 hours of therapy costs $300
  (h2 : F + 2 * A = 188)  -- 3 hours of therapy costs $188
  : F - A = 20 := by
  sorry

end psychologist_pricing_l2477_247703


namespace equal_one_two_digit_prob_l2477_247725

/-- A 20-sided die with numbers from 1 to 20 -/
def twentySidedDie : Finset ℕ := Finset.range 20

/-- The probability of rolling a one-digit number on a 20-sided die -/
def probOneDigit : ℚ := 9 / 20

/-- The probability of rolling a two-digit number on a 20-sided die -/
def probTwoDigit : ℚ := 11 / 20

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The probability of rolling an equal number of one-digit and two-digit numbers on 6 20-sided dice -/
theorem equal_one_two_digit_prob : 
  (Nat.choose numDice (numDice / 2) : ℚ) * probOneDigit ^ (numDice / 2) * probTwoDigit ^ (numDice / 2) = 970701 / 3200000 := by
  sorry

end equal_one_two_digit_prob_l2477_247725


namespace election_win_margin_l2477_247754

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) :
  (winner_votes : ℚ) / total_votes = 62 / 100 →
  winner_votes = 868 →
  winner_votes - (total_votes - winner_votes) = 336 :=
by sorry

end election_win_margin_l2477_247754


namespace triangle_area_rational_l2477_247717

/-- Given a triangle with vertices whose coordinates are integers adjusted by adding 0.5,
    prove that its area is always rational. -/
theorem triangle_area_rational (x₁ x₂ x₃ y₁ y₂ y₃ : ℤ) :
  ∃ (p q : ℤ), q ≠ 0 ∧ 
    (1/2 : ℚ) * |((x₁ + 1/2) * ((y₂ + 1/2) - (y₃ + 1/2)) + 
                  (x₂ + 1/2) * ((y₃ + 1/2) - (y₁ + 1/2)) + 
                  (x₃ + 1/2) * ((y₁ + 1/2) - (y₂ + 1/2)))| = p / q :=
by sorry

end triangle_area_rational_l2477_247717


namespace range_of_x_l2477_247737

theorem range_of_x (a b c x : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) 
  (h2 : a + 2*b + 3*c > |x + 1|) : -7 < x ∧ x < 5 := by
  sorry

end range_of_x_l2477_247737


namespace logarithmic_equation_solution_l2477_247766

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ 
  log_base 3 (x - 1) + log_base (Real.sqrt 3) (x^2 - 1) + log_base (1/3) (x - 1) = 3 ∧
  x = Real.sqrt (1 + 3 * Real.sqrt 3) :=
sorry

end logarithmic_equation_solution_l2477_247766


namespace function_non_negative_implies_a_range_l2477_247726

theorem function_non_negative_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, x^2 - 4*x + a ≥ 0) → a ∈ Set.Ici 3 :=
by sorry

end function_non_negative_implies_a_range_l2477_247726


namespace vector_computation_l2477_247765

theorem vector_computation : 
  5 • !![3, -9] - 4 • !![2, -6] + !![1, 3] = !![8, -18] := by
  sorry

end vector_computation_l2477_247765


namespace min_value_of_function_min_value_achievable_l2477_247790

theorem min_value_of_function (x : ℝ) (h : x > 0) : 2 * x + 3 / x ≥ 2 * Real.sqrt 6 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, x > 0 ∧ 2 * x + 3 / x = 2 * Real.sqrt 6 := by
  sorry

end min_value_of_function_min_value_achievable_l2477_247790


namespace area_enclosed_by_curve_l2477_247749

/-- The area enclosed by a curve composed of 12 congruent circular arcs -/
theorem area_enclosed_by_curve (arc_length : Real) (hexagon_side : Real) : 
  arc_length = 5 * Real.pi / 6 →
  hexagon_side = 4 →
  ∃ (area : Real), 
    area = 48 * Real.sqrt 3 + 125 * Real.pi / 2 ∧
    area = (3 * Real.sqrt 3 / 2 * hexagon_side ^ 2) + 
           (12 * (arc_length / (2 * Real.pi)) * Real.pi * (arc_length / Real.pi) ^ 2) :=
by sorry


end area_enclosed_by_curve_l2477_247749


namespace min_total_cost_l2477_247739

/-- Represents a salon with prices for haircut, facial cleaning, and nails -/
structure Salon where
  name : String
  haircut : ℕ
  facial : ℕ
  nails : ℕ

/-- Calculates the total cost of services at a salon -/
def totalCost (s : Salon) : ℕ := s.haircut + s.facial + s.nails

/-- The list of salons with their prices -/
def salonList : List Salon := [
  { name := "Gustran Salon", haircut := 45, facial := 22, nails := 30 },
  { name := "Barbara's Shop", haircut := 30, facial := 28, nails := 40 },
  { name := "The Fancy Salon", haircut := 34, facial := 30, nails := 20 }
]

/-- Theorem: The minimum total cost among the salons is 84 -/
theorem min_total_cost : 
  (salonList.map totalCost).minimum? = some 84 := by
  sorry

end min_total_cost_l2477_247739


namespace number_divided_by_three_equals_number_minus_five_l2477_247738

theorem number_divided_by_three_equals_number_minus_five : 
  ∃! x : ℝ, x / 3 = x - 5 := by sorry

end number_divided_by_three_equals_number_minus_five_l2477_247738


namespace prism_volume_l2477_247799

-- Define the prism dimensions
variable (a b c : ℝ)

-- Define the conditions
axiom face_area_1 : a * b = 30
axiom face_area_2 : b * c = 72
axiom face_area_3 : c * a = 45

-- State the theorem
theorem prism_volume : a * b * c = 180 * Real.sqrt 3 := by sorry

end prism_volume_l2477_247799


namespace truncated_cone_angle_l2477_247776

theorem truncated_cone_angle (R : ℝ) (h : ℝ) (r : ℝ) : 
  h = R → 
  (12 * r) / Real.sqrt 3 = 3 * R * Real.sqrt 3 → 
  Real.arctan (h / (R - r)) = Real.arctan 4 := by
  sorry

end truncated_cone_angle_l2477_247776


namespace sequence_sum_l2477_247733

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = n² + 1, prove a₁ + a₉ = 19 -/
theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2 + 1) : 
    a 1 + a 9 = 19 := by
  sorry

end sequence_sum_l2477_247733


namespace truck_rental_example_l2477_247785

/-- Calculates the total cost of renting a truck given the daily rate, per-mile rate, number of days, and miles driven. -/
def truck_rental_cost (daily_rate : ℚ) (mile_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mile_rate * miles

/-- Proves that renting a truck for $35 per day and $0.25 per mile for 3 days and 300 miles costs $180 in total. -/
theorem truck_rental_example : truck_rental_cost 35 (1/4) 3 300 = 180 := by
  sorry

end truck_rental_example_l2477_247785


namespace equal_balls_probability_l2477_247712

/-- Represents the urn state -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents a single draw operation -/
inductive Draw
  | Red
  | Blue

/-- Performs a single draw operation on the urn state -/
def drawOperation (state : UrnState) (draw : Draw) : UrnState :=
  match draw with
  | Draw.Red => UrnState.mk (state.red + 1) state.blue
  | Draw.Blue => UrnState.mk state.red (state.blue + 1)

/-- Performs a sequence of draw operations on the urn state -/
def performOperations (initial : UrnState) (draws : List Draw) : UrnState :=
  draws.foldl drawOperation initial

/-- Calculates the probability of a specific sequence of draws -/
def sequenceProbability (draws : List Draw) : ℚ :=
  sorry

/-- Calculates the number of valid sequences that result in 4 red and 4 blue balls -/
def validSequencesCount : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem equal_balls_probability :
  let initialState := UrnState.mk 2 1
  let finalState := UrnState.mk 4 4
  (validSequencesCount * sequenceProbability (List.replicate 5 Draw.Red)) = 8 / 21 := by
  sorry

end equal_balls_probability_l2477_247712


namespace employment_percentage_l2477_247764

theorem employment_percentage (total_population : ℝ) 
  (employed_males_percentage : ℝ) (employed_females_ratio : ℝ) :
  employed_males_percentage = 36 →
  employed_females_ratio = 50 →
  (employed_males_percentage / employed_females_ratio) * 100 = 72 :=
by
  sorry

end employment_percentage_l2477_247764


namespace quadratic_root_arithmetic_sequence_l2477_247732

/-- Given real numbers a, b, c forming an arithmetic sequence with c ≥ b ≥ a ≥ 0,
    the single root of the quadratic cx^2 + bx + a = 0 is -1 - (√3)/3 -/
theorem quadratic_root_arithmetic_sequence (a b c : ℝ) : 
  c ≥ b ∧ b ≥ a ∧ a ≥ 0 →
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →
  (∃! x : ℝ, c*x^2 + b*x + a = 0) →
  (∃ x : ℝ, c*x^2 + b*x + a = 0 ∧ x = -1 - Real.sqrt 3 / 3) :=
by sorry

end quadratic_root_arithmetic_sequence_l2477_247732


namespace polynomial_division_remainder_l2477_247789

def p (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 - 12 * x + 7
def d (x : ℝ) : ℝ := 2 * x + 3
def q (x : ℝ) : ℝ := x^2 - 4 * x + 2
def r (x : ℝ) : ℝ := -4 * x + 1

theorem polynomial_division_remainder :
  ∀ x : ℝ, p x = d x * q x + r x :=
sorry

end polynomial_division_remainder_l2477_247789


namespace find_m_l2477_247734

theorem find_m : ∃ m : ℝ, ∀ x y : ℝ, (2*x + y)*(x - 2*y) = 2*x^2 - m*x*y - 2*y^2 → m = 3 := by
  sorry

end find_m_l2477_247734


namespace sum_of_six_smallest_multiples_of_12_l2477_247728

theorem sum_of_six_smallest_multiples_of_12 : 
  (Finset.range 6).sum (λ i => 12 * (i + 1)) = 252 := by
  sorry

end sum_of_six_smallest_multiples_of_12_l2477_247728


namespace parallel_line_plane_false_l2477_247743

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem parallel_line_plane_false : 
  ¬ (∀ (α : Plane3D) (b : Line3D), 
    parallel_line_plane b α → 
    (∀ (a : Line3D), line_in_plane a α → parallel_lines b a)) := by
  sorry

end parallel_line_plane_false_l2477_247743


namespace probability_three_specific_heads_out_of_five_probability_three_specific_heads_out_of_five_proof_l2477_247740

/-- The probability of getting heads on exactly three specific coins out of five coins -/
theorem probability_three_specific_heads_out_of_five : ℝ :=
  let n_coins : ℕ := 5
  let n_specific_coins : ℕ := 3
  let p_head : ℝ := 1 / 2
  1 / 8

/-- Proof of the theorem -/
theorem probability_three_specific_heads_out_of_five_proof :
  probability_three_specific_heads_out_of_five = 1 / 8 := by
  sorry

end probability_three_specific_heads_out_of_five_probability_three_specific_heads_out_of_five_proof_l2477_247740


namespace elephant_hole_theorem_l2477_247724

/-- A paper represents a rectangular sheet with a given area -/
structure Paper where
  area : ℝ
  area_pos : area > 0

/-- A series of cuts can be represented as a function that transforms a paper -/
def Cut := Paper → Paper

/-- The theorem states that there exists a cut that can create a hole larger than the original paper -/
theorem elephant_hole_theorem (initial_paper : Paper) (k : ℝ) (h_k : k > 1) :
  ∃ (cut : Cut), (cut initial_paper).area > k * initial_paper.area := by
  sorry

end elephant_hole_theorem_l2477_247724


namespace two_digit_multiples_of_6_and_9_l2477_247736

theorem two_digit_multiples_of_6_and_9 : 
  (Finset.filter (fun n => n % 6 = 0 ∧ n % 9 = 0) (Finset.range 90 \ Finset.range 10)).card = 5 := by
  sorry

end two_digit_multiples_of_6_and_9_l2477_247736


namespace inequality_proof_l2477_247769

theorem inequality_proof (x a : ℝ) (h : x < a ∧ a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end inequality_proof_l2477_247769


namespace f_has_max_and_min_l2477_247786

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a + 6)

/-- Theorem stating the condition for f to have both maximum and minimum values -/
theorem f_has_max_and_min (a : ℝ) : 
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) ↔ (a < -3 ∨ a > 6) :=
sorry

end f_has_max_and_min_l2477_247786


namespace binomial_coefficient_congruence_l2477_247783

theorem binomial_coefficient_congruence (n : ℕ+) :
  ∃ σ : Fin (2^(n.val-1)) ≃ Fin (2^(n.val-1)),
    ∀ k : Fin (2^(n.val-1)),
      (Nat.choose (2^n.val - 1) k) ≡ (2 * σ k + 1) [MOD 2^n.val] := by
sorry

end binomial_coefficient_congruence_l2477_247783


namespace square_side_length_l2477_247741

theorem square_side_length (area : ℝ) (side_length : ℝ) :
  area = 4 ∧ area = side_length ^ 2 → side_length = 2 := by
  sorry

end square_side_length_l2477_247741


namespace fixed_point_of_exponential_function_l2477_247774

/-- For all a > 0 and a ≠ 1, the function f(x) = a^(x-3) - 3 passes through the point (3, -2) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) - 3
  f 3 = -2 := by
sorry

end fixed_point_of_exponential_function_l2477_247774


namespace chord_probability_chord_probability_proof_l2477_247722

/-- The probability that a randomly chosen point on a circle's circumference,
    when connected to a fixed point on the circumference, forms a chord with
    length between R and √3R, where R is the radius of the circle. -/
theorem chord_probability (R : ℝ) (R_pos : R > 0) : ℝ :=
  1 / 3

/-- Proof of the chord probability theorem -/
theorem chord_probability_proof (R : ℝ) (R_pos : R > 0) :
  chord_probability R R_pos = 1 / 3 := by
  sorry

end chord_probability_chord_probability_proof_l2477_247722


namespace rectangle_area_breadth_ratio_l2477_247791

/-- Proves that for a rectangular plot with breadth 11 metres and length 10 metres more than its breadth, 
    the area of the plot divided by its breadth equals 21. -/
theorem rectangle_area_breadth_ratio : 
  ∀ (length breadth area : ℝ),
    breadth = 11 →
    length = breadth + 10 →
    area = length * breadth →
    area / breadth = 21 := by
  sorry

end rectangle_area_breadth_ratio_l2477_247791


namespace joan_picked_apples_l2477_247729

/-- The number of apples Joan has now -/
def total_apples : ℕ := 70

/-- The number of apples Melanie gave to Joan -/
def melanie_apples : ℕ := 27

/-- The number of apples Joan picked from the orchard -/
def orchard_apples : ℕ := total_apples - melanie_apples

theorem joan_picked_apples : orchard_apples = 43 := by
  sorry

end joan_picked_apples_l2477_247729


namespace eighth_root_of_3906250000000001_l2477_247748

theorem eighth_root_of_3906250000000001 :
  let n : ℕ := 3906250000000001
  ∃ (m : ℕ), m ^ 8 = n ∧ m = 101 :=
by
  sorry

end eighth_root_of_3906250000000001_l2477_247748


namespace problem_statement_l2477_247715

theorem problem_statement (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xyz : x * y * z = 1)
  (h_x_z : x + 1 / z = 10)
  (h_y_x : y + 1 / x = 5) :
  z + 1 / y = 17 / 49 := by
sorry

end problem_statement_l2477_247715


namespace michelle_needs_three_more_racks_l2477_247709

/-- The number of additional drying racks Michelle needs -/
def additional_racks_needed : ℕ :=
  let total_flour : ℕ := 6 * 12 -- 6 bags * 12 cups per bag
  let flour_per_type : ℕ := total_flour / 2 -- equal amounts for both types
  let pasta_type1 : ℕ := flour_per_type / 3 -- 3 cups per pound for type 1
  let pasta_type2 : ℕ := flour_per_type / 4 -- 4 cups per pound for type 2
  let total_pasta : ℕ := pasta_type1 + pasta_type2
  let total_racks_needed : ℕ := (total_pasta + 4) / 5 -- Ceiling division by 5
  total_racks_needed - 2 -- Subtract the 2 racks she already owns

theorem michelle_needs_three_more_racks :
  additional_racks_needed = 3 := by
  sorry

end michelle_needs_three_more_racks_l2477_247709


namespace fraction_equality_l2477_247773

theorem fraction_equality : 
  (2 + 4 - 8 + 16 + 32 - 64 + 128 - 256) / (4 + 8 - 16 + 32 + 64 - 128 + 256 - 512) = 1 / 2 := by
  sorry

end fraction_equality_l2477_247773


namespace square_difference_squared_l2477_247701

theorem square_difference_squared : (7^2 - 3^2)^2 = 1600 := by
  sorry

end square_difference_squared_l2477_247701


namespace diagonal_length_from_offsets_and_area_l2477_247784

/-- The length of a diagonal of a quadrilateral, given its offsets and area -/
theorem diagonal_length_from_offsets_and_area 
  (offset1 : ℝ) (offset2 : ℝ) (area : ℝ) :
  offset1 = 7 →
  offset2 = 3 →
  area = 50 →
  ∃ (d : ℝ), d = 10 ∧ area = (1/2) * d * (offset1 + offset2) :=
by sorry

end diagonal_length_from_offsets_and_area_l2477_247784


namespace number_division_theorem_l2477_247716

theorem number_division_theorem : ∃! n : ℕ, 
  n / (2615 + 3895) = 3 * (3895 - 2615) ∧ 
  n % (2615 + 3895) = 65 := by
sorry

end number_division_theorem_l2477_247716


namespace special_quadrilateral_is_kite_l2477_247775

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  /-- The diagonals of the quadrilateral bisect each other -/
  diagonals_bisect : Bool
  /-- The diagonals of the quadrilateral are perpendicular -/
  diagonals_perpendicular : Bool
  /-- Two adjacent sides of the quadrilateral are equal -/
  two_adjacent_sides_equal : Bool

/-- Definition of a kite -/
def is_kite (q : SpecialQuadrilateral) : Prop :=
  q.diagonals_bisect ∧ q.diagonals_perpendicular ∧ q.two_adjacent_sides_equal

/-- The main theorem stating that a quadrilateral with the given properties is most likely a kite -/
theorem special_quadrilateral_is_kite (q : SpecialQuadrilateral) 
  (h1 : q.diagonals_bisect = true) 
  (h2 : q.diagonals_perpendicular = true) 
  (h3 : q.two_adjacent_sides_equal = true) : 
  is_kite q :=
sorry

end special_quadrilateral_is_kite_l2477_247775


namespace max_min_x_values_l2477_247770

theorem max_min_x_values (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (inequality : (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 2) :
  (∀ w, w = x → w ≤ 2/3) ∧ 
  (∃ v, v = x ∧ v = 2/3) ∧
  (∀ u, u = x → u ≥ -2/3) ∧
  (∃ t, t = x ∧ t = -2/3) :=
sorry

end max_min_x_values_l2477_247770


namespace sara_marbles_l2477_247788

theorem sara_marbles (initial_marbles additional_marbles : ℝ) 
  (h1 : initial_marbles = 4892.5)
  (h2 : additional_marbles = 2337.8) :
  initial_marbles + additional_marbles = 7230.3 := by
sorry

end sara_marbles_l2477_247788


namespace cookie_shop_problem_l2477_247792

def num_cookie_flavors : ℕ := 7
def num_milk_types : ℕ := 4
def total_products : ℕ := 4

def ways_charlie_buys (k : ℕ) : ℕ := Nat.choose (num_cookie_flavors + num_milk_types) k

def ways_delta_buys_distinct (k : ℕ) : ℕ := Nat.choose num_cookie_flavors k

def ways_delta_buys_with_repeats (k : ℕ) : ℕ :=
  if k = 1 then num_cookie_flavors
  else if k = 2 then ways_delta_buys_distinct 2 + num_cookie_flavors
  else if k = 3 then ways_delta_buys_distinct 3 + num_cookie_flavors * (num_cookie_flavors - 1) + num_cookie_flavors
  else if k = 4 then ways_delta_buys_distinct 4 + num_cookie_flavors * (num_cookie_flavors - 1) + 
                     (num_cookie_flavors * (num_cookie_flavors - 1)) / 2 + num_cookie_flavors
  else 0

def total_ways : ℕ :=
  (ways_charlie_buys 4) +
  (ways_charlie_buys 3 * ways_delta_buys_with_repeats 1) +
  (ways_charlie_buys 2 * ways_delta_buys_with_repeats 2) +
  (ways_charlie_buys 1 * ways_delta_buys_with_repeats 3) +
  (ways_delta_buys_with_repeats 4)

theorem cookie_shop_problem : total_ways = 4054 := by sorry

end cookie_shop_problem_l2477_247792


namespace parallelogram_area_l2477_247787

/-- The area of a parallelogram with one angle of 135 degrees and two consecutive sides of lengths 10 and 17 is equal to 85√2. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 17) (h3 : θ = 135 * π / 180) :
  a * b * Real.sin θ = 85 * Real.sqrt 2 := by
  sorry

end parallelogram_area_l2477_247787


namespace triangle_side_length_l2477_247720

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S_ABC = 1/4(a^2 + b^2 - c^2), b = 1, and a = √2, then c = 1. -/
theorem triangle_side_length (a b c : ℝ) (h_area : (a^2 + b^2 - c^2) / 4 = a * b * Real.sin (π/4) / 2)
  (h_b : b = 1) (h_a : a = Real.sqrt 2) : c = 1 := by
  sorry

end triangle_side_length_l2477_247720


namespace cuboid_volume_example_l2477_247746

/-- The volume of a cuboid with given base area and height -/
def cuboid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  base_area * height

/-- Theorem: The volume of a cuboid with base area 18 m^2 and height 8 m is 144 m^3 -/
theorem cuboid_volume_example : cuboid_volume 18 8 = 144 := by
  sorry

end cuboid_volume_example_l2477_247746


namespace average_movie_length_l2477_247745

def miles_run : ℕ := 15
def minutes_per_mile : ℕ := 12
def number_of_movies : ℕ := 2

theorem average_movie_length :
  (miles_run * minutes_per_mile) / number_of_movies = 90 :=
by sorry

end average_movie_length_l2477_247745


namespace arithmetic_mean_of_special_set_l2477_247782

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set := List.replicate (n - 3) 1 ++ [1 + 1/n, 1 + 1/n, 1 - 1/n]
  (set.sum / n : ℚ) = 1 + 1/n^2 := by
  sorry

end arithmetic_mean_of_special_set_l2477_247782


namespace prob_X_equals_three_l2477_247771

/-- X is a random variable following a binomial distribution B(6, 1/2) -/
def X : Real → Real := sorry

/-- The probability mass function of X -/
def pmf (k : ℕ) : Real := sorry

/-- Theorem: The probability of X = 3 is 5/16 -/
theorem prob_X_equals_three : pmf 3 = 5/16 := by sorry

end prob_X_equals_three_l2477_247771
