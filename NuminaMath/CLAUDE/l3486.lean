import Mathlib

namespace fred_has_five_balloons_l3486_348691

/-- The number of yellow balloons Fred has -/
def fred_balloons (total sam mary : ℕ) : ℕ := total - (sam + mary)

/-- Theorem: Fred has 5 yellow balloons -/
theorem fred_has_five_balloons (total sam mary : ℕ) 
  (h_total : total = 18) 
  (h_sam : sam = 6) 
  (h_mary : mary = 7) : 
  fred_balloons total sam mary = 5 := by
  sorry

end fred_has_five_balloons_l3486_348691


namespace dissimilar_terms_expansion_l3486_348614

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^12 -/
def dissimilar_terms : ℕ := 455

/-- The number of variables in the expansion -/
def num_variables : ℕ := 4

/-- The power to which the sum is raised -/
def power : ℕ := 12

/-- Theorem stating that the number of dissimilar terms in (a + b + c + d)^12 is 455 -/
theorem dissimilar_terms_expansion :
  dissimilar_terms = Nat.choose (power + num_variables - 1) (num_variables - 1) := by
  sorry

end dissimilar_terms_expansion_l3486_348614


namespace modulus_of_z_l3486_348667

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end modulus_of_z_l3486_348667


namespace chord_equation_l3486_348628

theorem chord_equation (m n s t : ℝ) (hm : 0 < m) (hn : 0 < n) (hs : 0 < s) (ht : 0 < t)
  (h1 : m + n = 2) (h2 : m / s + n / t = 9) (h3 : s + t = 4 / 9)
  (h4 : ∃ (x1 y1 x2 y2 : ℝ), 
    x1^2 / 4 + y1^2 / 2 = 1 ∧ 
    x2^2 / 4 + y2^2 / 2 = 1 ∧ 
    (x1 + x2) / 2 = m ∧ 
    (y1 + y2) / 2 = n) :
  ∃ (a b c : ℝ), a * m + b * n + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3 := by
sorry

end chord_equation_l3486_348628


namespace candidate_a_vote_percentage_l3486_348661

/-- Represents the percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.6

/-- Represents the percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 1 - democrat_percentage

/-- Represents the percentage of Republican voters expected to vote for candidate A -/
def republican_vote_percentage : ℝ := 0.2

/-- Represents the total percentage of registered voters expected to vote for candidate A -/
def total_vote_percentage : ℝ := 0.5

/-- Represents the percentage of Democratic voters expected to vote for candidate A -/
def democrat_vote_percentage : ℝ := 0.7

theorem candidate_a_vote_percentage :
  democrat_percentage * democrat_vote_percentage +
  republican_percentage * republican_vote_percentage =
  total_vote_percentage :=
sorry

end candidate_a_vote_percentage_l3486_348661


namespace distribute_books_correct_l3486_348683

/-- The number of ways to distribute 6 different books among three people. -/
def distribute_books : Nat × Nat × Nat × Nat :=
  let n : Nat := 6  -- Total number of books
  let k : Nat := 3  -- Number of people

  -- Scenario 1: One person gets 1 book, another gets 2 books, the last gets 3 books
  let scenario1 : Nat := k.factorial * n.choose 1 * (n - 1).choose 2 * (n - 3).choose 3

  -- Scenario 2: Books are evenly distributed, each person getting 2 books
  let scenario2 : Nat := (n.choose 2 * (n - 2).choose 2 * (n - 4).choose 2) / k.factorial

  -- Scenario 3: One part gets 4 books, other two parts get 1 book each
  let scenario3 : Nat := n.choose 4

  -- Scenario 4: A gets 1 book, B gets 1 book, C gets 4 books
  let scenario4 : Nat := n.choose 4 * (n - 4).choose 1

  (scenario1, scenario2, scenario3, scenario4)

theorem distribute_books_correct :
  distribute_books = (360, 90, 15, 30) := by
  sorry

end distribute_books_correct_l3486_348683


namespace odd_periodic_function_property_l3486_348615

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem odd_periodic_function_property
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : has_period f (π / 2))
  (h_value : f (π / 3) = 1) :
  f (-5 * π / 6) = -1 :=
sorry

end odd_periodic_function_property_l3486_348615


namespace certain_amount_proof_l3486_348688

theorem certain_amount_proof : 
  ∀ (amount : ℝ), 
    (0.25 * 680 = 0.20 * 1000 - amount) → 
    amount = 30 := by
  sorry

end certain_amount_proof_l3486_348688


namespace inequality_proof_l3486_348622

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (h : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end inequality_proof_l3486_348622


namespace symmetric_point_in_third_quadrant_l3486_348660

/-- A point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to x-axis -/
def symmetricXAxis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Definition of third quadrant -/
def isThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The main theorem -/
theorem symmetric_point_in_third_quadrant :
  let P : Point := ⟨-2, 1⟩
  let P' := symmetricXAxis P
  isThirdQuadrant P' :=
by
  sorry


end symmetric_point_in_third_quadrant_l3486_348660


namespace black_ball_count_l3486_348679

/-- Given a bag with white and black balls, prove the number of black balls when the probability of drawing a white ball is known -/
theorem black_ball_count 
  (white_balls : ℕ) 
  (black_balls : ℕ) 
  (total_balls : ℕ) 
  (prob_white : ℚ) 
  (h1 : white_balls = 20)
  (h2 : total_balls = white_balls + black_balls)
  (h3 : prob_white = 2/5)
  (h4 : prob_white = white_balls / total_balls) :
  black_balls = 30 := by
sorry

end black_ball_count_l3486_348679


namespace value_of_S_l3486_348669

theorem value_of_S : let S : ℝ := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 1 / (Real.sqrt 12 - 3)
  S = 7 := by sorry

end value_of_S_l3486_348669


namespace certain_number_problem_l3486_348674

theorem certain_number_problem : ∃ x : ℚ, (1206 / 3 : ℚ) = 3 * x ∧ x = 134 := by
  sorry

end certain_number_problem_l3486_348674


namespace smallest_satisfying_number_l3486_348682

/-- Represents the decimal expansion of a rational number -/
def DecimalExpansion (q : ℚ) : List ℕ := sorry

/-- Checks if a list contains three consecutive identical elements -/
def hasThreeConsecutiveIdentical (l : List ℕ) : Prop := sorry

/-- Checks if a list is entirely composed of identical elements -/
def isEntirelyIdentical (l : List ℕ) : Prop := sorry

/-- Checks if a natural number satisfies the given conditions -/
def satisfiesConditions (n : ℕ) : Prop :=
  let expansion := DecimalExpansion (1 / n)
  hasThreeConsecutiveIdentical expansion ∧ ¬isEntirelyIdentical expansion

theorem smallest_satisfying_number :
  satisfiesConditions 157 ∧ ∀ m < 157, ¬satisfiesConditions m := by sorry

end smallest_satisfying_number_l3486_348682


namespace unique_six_digit_number_l3486_348675

/-- A six-digit number starting with 1 -/
def SixDigitNumber (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 1

/-- Function to move the first digit of a six-digit number to the last position -/
def MoveFirstToLast (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

/-- The main theorem -/
theorem unique_six_digit_number : 
  ∀ n : ℕ, SixDigitNumber n → (MoveFirstToLast n = 3 * n) → n = 142857 :=
by
  sorry

end unique_six_digit_number_l3486_348675


namespace arithmetic_geometric_sequence_l3486_348632

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 := by
  sorry

end arithmetic_geometric_sequence_l3486_348632


namespace max_stamps_with_50_dollars_l3486_348607

/-- The maximum number of stamps that can be purchased with a given amount of money and stamp price. -/
def maxStamps (totalMoney stampPrice : ℕ) : ℕ :=
  totalMoney / stampPrice

/-- Theorem stating that with $50 and stamps costing 25 cents each, the maximum number of stamps that can be purchased is 200. -/
theorem max_stamps_with_50_dollars : 
  let dollarAmount : ℕ := 50
  let stampPriceCents : ℕ := 25
  let totalCents : ℕ := dollarAmount * 100
  maxStamps totalCents stampPriceCents = 200 := by
  sorry

#eval maxStamps (50 * 100) 25

end max_stamps_with_50_dollars_l3486_348607


namespace point_coordinates_l3486_348655

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant of the 2D plane -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (M : Point) 
  (h1 : secondQuadrant M)
  (h2 : distanceToXAxis M = 5)
  (h3 : distanceToYAxis M = 3) :
  M.x = -3 ∧ M.y = 5 := by
  sorry

end point_coordinates_l3486_348655


namespace equation_is_parabola_l3486_348663

/-- Represents a conic section --/
inductive ConicSection
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines the type of conic section for the given equation --/
def determine_conic_section (equation : ℝ → ℝ → Prop) : ConicSection := sorry

/-- The equation |x-3| = √((y+4)² + x²) --/
def equation (x y : ℝ) : Prop :=
  |x - 3| = Real.sqrt ((y + 4)^2 + x^2)

theorem equation_is_parabola :
  determine_conic_section equation = ConicSection.Parabola := by sorry

end equation_is_parabola_l3486_348663


namespace consecutive_numbers_sum_l3486_348652

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 105) → 
  (n + (n + 5) = 35) := by
  sorry

end consecutive_numbers_sum_l3486_348652


namespace cow_spots_problem_l3486_348673

theorem cow_spots_problem (left_spots right_spots total_spots additional_spots : ℕ) :
  left_spots = 16 →
  right_spots = 3 * left_spots + additional_spots →
  total_spots = left_spots + right_spots →
  total_spots = 71 →
  additional_spots = 7 := by
  sorry

end cow_spots_problem_l3486_348673


namespace quadratic_roots_sum_product_l3486_348687

theorem quadratic_roots_sum_product (x₁ x₂ k d : ℝ) : 
  x₁ ≠ x₂ →
  4 * x₁^2 - k * x₁ = d →
  4 * x₂^2 - k * x₂ = d →
  x₁ + x₂ = 2 →
  d = -12 := by
sorry

end quadratic_roots_sum_product_l3486_348687


namespace climbing_five_floors_l3486_348635

/-- The number of ways to climb a building with a given number of floors and staircases per floor -/
def climbingWays (floors : ℕ) (staircasesPerFloor : ℕ) : ℕ :=
  staircasesPerFloor ^ (floors - 1)

/-- Theorem: In a 5-floor building with 2 staircases per floor, there are 16 ways to go from the first to the fifth floor -/
theorem climbing_five_floors :
  climbingWays 5 2 = 16 := by
  sorry

end climbing_five_floors_l3486_348635


namespace sphere_radius_l3486_348611

/-- Given a sphere and a pole under parallel sun rays, where the sphere's shadow extends
    12 meters from the point of contact with the ground, and a 3-meter tall pole casts
    a 4-meter shadow, the radius of the sphere is 9 meters. -/
theorem sphere_radius (shadow_length : ℝ) (pole_height : ℝ) (pole_shadow : ℝ) :
  shadow_length = 12 →
  pole_height = 3 →
  pole_shadow = 4 →
  ∃ (sphere_radius : ℝ), sphere_radius = 9 :=
by sorry

end sphere_radius_l3486_348611


namespace string_length_problem_l3486_348613

theorem string_length_problem (total_strings : ℕ) (avg_length : ℝ) (other_strings : ℕ) (other_avg : ℝ) :
  total_strings = 6 →
  avg_length = 80 →
  other_strings = 4 →
  other_avg = 85 →
  let remaining_strings := total_strings - other_strings
  let total_length := avg_length * total_strings
  let other_length := other_avg * other_strings
  let remaining_length := total_length - other_length
  remaining_length / remaining_strings = 70 := by
sorry

end string_length_problem_l3486_348613


namespace equation_roots_l3486_348643

theorem equation_roots : 
  ∀ x : ℝ, (21 / (x^2 - 9) - 3 / (x - 3) = 1) ↔ (x = 3 ∨ x = -7) :=
by sorry

end equation_roots_l3486_348643


namespace product_mod_eight_l3486_348671

theorem product_mod_eight : (71 * 73) % 8 = 7 := by
  sorry

end product_mod_eight_l3486_348671


namespace R_and_T_largest_area_l3486_348657

/-- Represents a polygon constructed from unit squares and right triangles with legs of length 1 -/
structure Polygon where
  squares : ℕ
  triangles : ℕ

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℚ :=
  p.squares + p.triangles / 2

/-- The five polygons P, Q, R, S, T -/
def P : Polygon := ⟨3, 2⟩
def Q : Polygon := ⟨4, 1⟩
def R : Polygon := ⟨6, 0⟩
def S : Polygon := ⟨2, 4⟩
def T : Polygon := ⟨5, 2⟩

/-- Theorem stating that R and T have the largest area among the five polygons -/
theorem R_and_T_largest_area :
  area R = area T ∧
  area R ≥ area P ∧
  area R ≥ area Q ∧
  area R ≥ area S :=
sorry

end R_and_T_largest_area_l3486_348657


namespace integer_1200_in_column_B_l3486_348654

/-- The column type representing the six columns A, B, C, D, E, F --/
inductive Column
| A | B | C | D | E | F

/-- The function that maps a positive integer to its corresponding column in the zigzag pattern --/
def columnFor (n : ℕ) : Column :=
  match (n - 1) % 10 with
  | 0 => Column.A
  | 1 => Column.B
  | 2 => Column.C
  | 3 => Column.D
  | 4 => Column.E
  | 5 => Column.F
  | 6 => Column.E
  | 7 => Column.D
  | 8 => Column.C
  | _ => Column.B

/-- Theorem stating that the integer 1200 will be placed in column B --/
theorem integer_1200_in_column_B : columnFor 1200 = Column.B := by
  sorry

end integer_1200_in_column_B_l3486_348654


namespace magicians_number_l3486_348624

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a ≤ 9
  h2 : b ≤ 9
  h3 : c ≤ 9
  h4 : 0 < a

/-- Calculates the value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- Calculates the sum of all permutations of a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  (value n) + 
  (100 * n.a + 10 * n.c + n.b) +
  (100 * n.b + 10 * n.c + n.a) +
  (100 * n.b + 10 * n.a + n.c) +
  (100 * n.c + 10 * n.a + n.b) +
  (100 * n.c + 10 * n.b + n.a)

/-- The main theorem to prove -/
theorem magicians_number (n : ThreeDigitNumber) 
  (h : sumOfPermutations n = 4332) : value n = 118 := by
  sorry

end magicians_number_l3486_348624


namespace perpendicular_lines_l3486_348649

def line1 (x y : ℝ) : Prop := 3 * y - 2 * x + 4 = 0
def line2 (x y : ℝ) (b : ℝ) : Prop := 5 * y + b * x - 1 = 0

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines (b : ℝ) : 
  (∀ x y : ℝ, line1 x y → ∃ m1 : ℝ, y = m1 * x + (-4/3)) →
  (∀ x y : ℝ, line2 x y b → ∃ m2 : ℝ, y = m2 * x + (1/5)) →
  perpendicular (2/3) (-b/5) →
  b = 15/2 := by sorry

end perpendicular_lines_l3486_348649


namespace equilateral_triangle_area_l3486_348642

theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = Real.sqrt 8) :
  let side := (4 * Real.sqrt 6) / 3
  let area := (1 / 2) * side * h
  area = 8 * Real.sqrt 3 := by sorry

end equilateral_triangle_area_l3486_348642


namespace benny_birthday_money_l3486_348606

/-- The amount of money Benny spent on baseball gear -/
def money_spent : ℕ := 34

/-- The amount of money Benny had left over -/
def money_left : ℕ := 33

/-- The total amount of money Benny received for his birthday -/
def total_money : ℕ := money_spent + money_left

theorem benny_birthday_money :
  total_money = 67 := by sorry

end benny_birthday_money_l3486_348606


namespace marble_selection_l3486_348678

theorem marble_selection (n m k b : ℕ) (h1 : n = 10) (h2 : m = 2) (h3 : k = 4) (h4 : b = 2) :
  (Nat.choose n k) - (Nat.choose (n - m) k) = 140 := by
  sorry

end marble_selection_l3486_348678


namespace junior_score_junior_score_is_89_l3486_348638

theorem junior_score (n : ℝ) (junior_ratio : ℝ) (senior_ratio : ℝ) 
  (overall_avg : ℝ) (senior_avg : ℝ) : ℝ :=
  let junior_count := junior_ratio * n
  let senior_count := senior_ratio * n
  let total_score := overall_avg * n
  let senior_total := senior_avg * senior_count
  let junior_total := total_score - senior_total
  junior_total / junior_count

theorem junior_score_is_89 :
  junior_score 100 0.2 0.8 85 84 = 89 := by
  sorry

end junior_score_junior_score_is_89_l3486_348638


namespace standard_spherical_coordinates_example_l3486_348617

/-- 
Given a point in spherical coordinates (ρ, θ, φ), this function returns the 
standard representation coordinates (ρ', θ', φ') where:
- ρ' = ρ
- θ' is θ adjusted to be in the range [0, 2π)
- φ' is φ adjusted to be in the range [0, π]
-/
def standardSphericalCoordinates (ρ θ φ : Real) : Real × Real × Real :=
  sorry

theorem standard_spherical_coordinates_example :
  let (ρ, θ, φ) := (5, 3 * Real.pi / 8, 9 * Real.pi / 5)
  let (ρ', θ', φ') := standardSphericalCoordinates ρ θ φ
  ρ' = 5 ∧ θ' = 3 * Real.pi / 8 ∧ φ' = Real.pi / 5 :=
by sorry

end standard_spherical_coordinates_example_l3486_348617


namespace days_for_one_piece_correct_l3486_348612

/-- The number of days Aarti needs to complete one piece of work -/
def days_for_one_piece : ℝ := 6

/-- The number of days Aarti needs to complete three pieces of work -/
def days_for_three_pieces : ℝ := 18

/-- Theorem stating that the number of days for one piece of work is correct -/
theorem days_for_one_piece_correct : 
  days_for_one_piece * 3 = days_for_three_pieces :=
sorry

end days_for_one_piece_correct_l3486_348612


namespace jenny_reading_speed_l3486_348621

/-- Represents Jenny's reading challenge --/
structure ReadingChallenge where
  days : ℕ
  books : ℕ
  book1_words : ℕ
  book2_words : ℕ
  book3_words : ℕ
  reading_minutes_per_day : ℕ

/-- Calculates the reading speed in words per hour --/
def calculate_reading_speed (challenge : ReadingChallenge) : ℕ :=
  let total_words := challenge.book1_words + challenge.book2_words + challenge.book3_words
  let words_per_day := total_words / challenge.days
  let reading_hours_per_day := challenge.reading_minutes_per_day / 60
  words_per_day / reading_hours_per_day

/-- Jenny's specific reading challenge --/
def jenny_challenge : ReadingChallenge :=
  { days := 10
  , books := 3
  , book1_words := 200
  , book2_words := 400
  , book3_words := 300
  , reading_minutes_per_day := 54
  }

/-- Theorem stating that Jenny's reading speed is 100 words per hour --/
theorem jenny_reading_speed :
  calculate_reading_speed jenny_challenge = 100 := by
  sorry

end jenny_reading_speed_l3486_348621


namespace exists_a_divides_a_squared_minus_a_l3486_348610

theorem exists_a_divides_a_squared_minus_a (n k : ℕ) 
  (h1 : n > 1) 
  (h2 : k = (Nat.factors n).card) : 
  ∃ a : ℕ, 1 < a ∧ a < n / k + 1 ∧ n ∣ (a^2 - a) := by
  sorry

end exists_a_divides_a_squared_minus_a_l3486_348610


namespace expression_evaluation_l3486_348664

theorem expression_evaluation :
  let x : ℚ := 2/3
  let y : ℚ := 4/5
  (6*x + 8*y + x^2*y) / (60*x*y^2) = 21/50 := by sorry

end expression_evaluation_l3486_348664


namespace common_internal_tangent_length_l3486_348604

theorem common_internal_tangent_length
  (center_distance : ℝ)
  (radius1 : ℝ)
  (radius2 : ℝ)
  (h1 : center_distance = 50)
  (h2 : radius1 = 7)
  (h3 : radius2 = 10) :
  Real.sqrt (center_distance^2 - (radius1 + radius2)^2) = Real.sqrt 2211 :=
by sorry

end common_internal_tangent_length_l3486_348604


namespace min_sum_of_digits_for_odd_primes_l3486_348692

def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def n (p : ℕ) : ℕ := p^4 - 5*p^2 + 13

theorem min_sum_of_digits_for_odd_primes :
  ∀ p : ℕ, is_odd_prime p → sum_of_digits (n p) ≥ sum_of_digits (n 5) :=
sorry

end min_sum_of_digits_for_odd_primes_l3486_348692


namespace largest_even_digit_multiple_of_9_under_1000_l3486_348666

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_even_digit_multiple_of_9_under_1000 :
  ∃ (n : ℕ), n = 360 ∧
    n < 1000 ∧
    has_only_even_digits n ∧
    n % 9 = 0 ∧
    ∀ (m : ℕ), m < 1000 → has_only_even_digits m → m % 9 = 0 → m ≤ n :=
by sorry

end largest_even_digit_multiple_of_9_under_1000_l3486_348666


namespace unique_positive_solution_l3486_348699

/-- A system of linear equations with integer coefficients -/
structure LinearSystem where
  eq1 : ℤ → ℤ → ℤ → ℤ
  eq2 : ℤ → ℤ → ℤ → ℤ

/-- The specific system of equations from the problem -/
def problemSystem : LinearSystem :=
  { eq1 := λ x y z => 3*x - 4*y + 5*z - 10
    eq2 := λ x y z => 8*x + 7*y - 3*z - 13 }

/-- A solution to the system is valid if both equations equal zero -/
def isValidSolution (s : LinearSystem) (x y z : ℤ) : Prop :=
  s.eq1 x y z = 0 ∧ s.eq2 x y z = 0

/-- A positive integer solution -/
def isPositiveIntegerSolution (x y z : ℤ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0

theorem unique_positive_solution :
  ∀ x y z : ℤ,
    isValidSolution problemSystem x y z ∧ isPositiveIntegerSolution x y z
    ↔ x = 1 ∧ y = 2 ∧ z = 3 := by
  sorry

end unique_positive_solution_l3486_348699


namespace no_solution_iff_m_eq_four_l3486_348645

-- Define the equation
def equation (x m : ℝ) : Prop := 2 / x = m / (2 * x + 1)

-- Theorem stating the condition for no solution
theorem no_solution_iff_m_eq_four :
  (∀ x : ℝ, ¬ equation x m) ↔ m = 4 := by
  sorry

end no_solution_iff_m_eq_four_l3486_348645


namespace complex_norm_squared_l3486_348605

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.normSq z = 5 - 7*I) : Complex.normSq z = (74:ℝ)/10 := by
  sorry

end complex_norm_squared_l3486_348605


namespace triple_percent_40_l3486_348668

/-- The operation % defined on real numbers -/
def percent (M : ℝ) : ℝ := 0.4 * M + 2

/-- Theorem stating that applying the percent operation three times to 40 results in 5.68 -/
theorem triple_percent_40 : percent (percent (percent 40)) = 5.68 := by
  sorry

end triple_percent_40_l3486_348668


namespace original_number_is_two_thirds_l3486_348653

theorem original_number_is_two_thirds :
  ∃ x : ℚ, (1 + 1 / x = 5 / 2) ∧ (x = 2 / 3) := by sorry

end original_number_is_two_thirds_l3486_348653


namespace smallest_n_for_pie_distribution_l3486_348630

theorem smallest_n_for_pie_distribution (N : ℕ) : 
  N > 70 → (21 * N) % 70 = 0 → (∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N) → N = 80 :=
by sorry

end smallest_n_for_pie_distribution_l3486_348630


namespace five_students_three_locations_l3486_348644

/-- The number of ways to assign students to locations -/
def assignment_count (n : ℕ) (k : ℕ) : ℕ :=
  -- n: number of students
  -- k: number of locations
  sorry

/-- Theorem stating the number of assignment plans for 5 students and 3 locations -/
theorem five_students_three_locations :
  assignment_count 5 3 = 150 := by sorry

end five_students_three_locations_l3486_348644


namespace blue_rows_count_l3486_348695

/-- Given a grid with the following properties:
  * 10 rows and 15 squares per row
  * 4 rows of 6 squares in the middle are red
  * 66 squares are green
  * All remaining squares are blue
  * Blue squares cover entire rows
Prove that the number of rows colored blue at the beginning and end of the grid is 4 -/
theorem blue_rows_count (total_rows : Nat) (squares_per_row : Nat) 
  (red_rows : Nat) (red_squares_per_row : Nat) (green_squares : Nat) : 
  total_rows = 10 → 
  squares_per_row = 15 → 
  red_rows = 4 → 
  red_squares_per_row = 6 → 
  green_squares = 66 → 
  (total_rows * squares_per_row - red_rows * red_squares_per_row - green_squares) / squares_per_row = 4 := by
  sorry

end blue_rows_count_l3486_348695


namespace chucks_team_lead_l3486_348601

/-- The lead of Chuck's team over the Yellow Team -/
def lead (chuck_score yellow_score : ℕ) : ℕ := chuck_score - yellow_score

/-- Theorem stating that Chuck's team's lead over the Yellow Team is 17 points -/
theorem chucks_team_lead : lead 72 55 = 17 := by
  sorry

end chucks_team_lead_l3486_348601


namespace fraction_equivalences_l3486_348602

theorem fraction_equivalences : 
  ∃ (n : ℕ) (p : ℕ) (d : ℚ),
    (n : ℚ) / 15 = 4 / 5 ∧
    (4 : ℚ) / 5 = p / 100 ∧
    (4 : ℚ) / 5 = d ∧
    d = 0.8 ∧
    p = 80 :=
sorry

end fraction_equivalences_l3486_348602


namespace total_books_l3486_348626

theorem total_books (tim_books sam_books alex_books : ℕ) 
  (h1 : tim_books = 44)
  (h2 : sam_books = 52)
  (h3 : alex_books = 65) :
  tim_books + sam_books + alex_books = 161 := by
  sorry

end total_books_l3486_348626


namespace total_cards_traded_is_128_l3486_348696

/-- Represents the number of cards of each type --/
structure CardCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents a trade of cards --/
structure Trade where
  fromA : ℕ
  fromB : ℕ
  fromC : ℕ
  toA : ℕ
  toB : ℕ
  toC : ℕ

/-- Calculates the total number of cards traded in a single trade --/
def cardsTraded (trade : Trade) : ℕ :=
  trade.fromA + trade.fromB + trade.fromC + trade.toA + trade.toB + trade.toC

/-- Represents the initial card counts and trades for each round --/
structure RoundData where
  initialPadma : CardCounts
  initialRobert : CardCounts
  padmaTrade : Trade
  robertTrade : Trade

theorem total_cards_traded_is_128 
  (round1 : RoundData)
  (round2 : RoundData)
  (round3 : RoundData)
  (h1 : round1.initialPadma = ⟨50, 45, 30⟩)
  (h2 : round1.padmaTrade = ⟨5, 12, 0, 0, 0, 20⟩)
  (h3 : round2.initialRobert = ⟨60, 50, 40⟩)
  (h4 : round2.robertTrade = ⟨10, 3, 15, 8, 18, 0⟩)
  (h5 : round3.padmaTrade = ⟨0, 15, 10, 12, 0, 0⟩) :
  cardsTraded round1.padmaTrade + cardsTraded round2.robertTrade + cardsTraded round3.padmaTrade = 128 := by
  sorry


end total_cards_traded_is_128_l3486_348696


namespace quadratic_distinct_roots_range_l3486_348627

theorem quadratic_distinct_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0) →
  m < 1 := by
  sorry

end quadratic_distinct_roots_range_l3486_348627


namespace joshua_finish_time_difference_l3486_348662

/-- Race parameters -/
def race_length : ℕ := 15
def uphill_length : ℕ := 5
def flat_length : ℕ := race_length - uphill_length

/-- Runner speeds (in minutes per mile) -/
def malcolm_flat_speed : ℕ := 4
def joshua_flat_speed : ℕ := 6
def malcolm_uphill_additional : ℕ := 2
def joshua_uphill_additional : ℕ := 3

/-- Calculate total race time for a runner -/
def total_race_time (flat_speed uphill_additional : ℕ) : ℕ :=
  flat_speed * flat_length + (flat_speed + uphill_additional) * uphill_length

/-- Theorem: Joshua finishes 35 minutes after Malcolm -/
theorem joshua_finish_time_difference :
  total_race_time joshua_flat_speed joshua_uphill_additional -
  total_race_time malcolm_flat_speed malcolm_uphill_additional = 35 := by
  sorry


end joshua_finish_time_difference_l3486_348662


namespace sum_division_l3486_348637

theorem sum_division (x y z : ℝ) (total : ℝ) (y_share : ℝ) : 
  total = 245 →
  y_share = 63 →
  y = 0.45 * x →
  total = x + y + z →
  z / x = 0.30 := by
  sorry

end sum_division_l3486_348637


namespace k_range_theorem_l3486_348656

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x - 1

-- Define the even function g and odd function h
def g (x : ℝ) : ℝ := x^2 - 1
def h (x : ℝ) : ℝ := x

-- State the theorem
theorem k_range_theorem (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → g (k*x + k/x) < g (x^2 + 1/x^2 + 1)) ↔ 
  (-3/2 < k ∧ k < 3/2) :=
sorry

end k_range_theorem_l3486_348656


namespace ryan_reads_more_pages_l3486_348609

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pages Ryan read -/
def ryan_total_pages : ℕ := 2100

/-- The number of pages Ryan's brother read per day -/
def brother_pages_per_day : ℕ := 200

/-- The difference in average pages read per day between Ryan and his brother -/
def page_difference : ℕ := ryan_total_pages / days_in_week - brother_pages_per_day

theorem ryan_reads_more_pages :
  page_difference = 100 := by sorry

end ryan_reads_more_pages_l3486_348609


namespace bob_winning_strategy_l3486_348620

/-- A polynomial with natural number coefficients -/
def NatPoly := ℕ → ℕ

/-- The evaluation of a polynomial at a given point -/
def eval (P : NatPoly) (x : ℤ) : ℕ :=
  sorry

/-- The degree of a polynomial -/
def degree (P : NatPoly) : ℕ :=
  sorry

/-- Bob's strategy: choose two integers and receive their polynomial values -/
def bob_strategy (P : NatPoly) : (ℤ × ℤ × ℕ × ℕ) :=
  sorry

theorem bob_winning_strategy :
  ∀ (P Q : NatPoly),
    (∀ (x : ℤ), eval P x = eval Q x) →
    let (a, b, Pa, Pb) := bob_strategy P
    eval P a = Pa ∧ eval P b = Pb ∧ eval Q a = Pa ∧ eval Q b = Pb →
    P = Q :=
  sorry

end bob_winning_strategy_l3486_348620


namespace solution_set_l3486_348618

def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}

def B (m : ℝ) : Set ℝ := {x : ℝ | m*x + 1 = 0}

theorem solution_set (m : ℝ) : (A ∪ B m = A) ↔ m ∈ ({0, -1/2, -1/3} : Set ℝ) := by
  sorry

end solution_set_l3486_348618


namespace eccentric_annulus_area_l3486_348633

/-- Eccentric annulus area theorem -/
theorem eccentric_annulus_area 
  (R r d : ℝ) 
  (h1 : R > r) 
  (h2 : d < R) : 
  Real.pi * (R - r - d^2 / (R - r)) = 
    Real.pi * R^2 - Real.pi * r^2 :=
sorry

end eccentric_annulus_area_l3486_348633


namespace carla_water_consumption_l3486_348651

/-- Given the conditions of Carla's liquid consumption, prove that she drank 15 ounces of water. -/
theorem carla_water_consumption (water soda : ℝ) 
  (h1 : soda = 3 * water - 6)
  (h2 : water + soda = 54) :
  water = 15 := by
  sorry

end carla_water_consumption_l3486_348651


namespace f_strictly_increasing_when_a_eq_one_f_increasing_intervals_l3486_348694

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x

-- Theorem for part (I)
theorem f_strictly_increasing_when_a_eq_one :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f 1 x₁ < f 1 x₂ := by sorry

-- Theorem for part (II)
theorem f_increasing_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (0 < a → a < 1 → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < a → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f a x₁ < f a x₂)) ∧
  (a = 1 → ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (1 < a → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂ : ℝ, a < x₁ → x₁ < x₂ → f a x₁ < f a x₂)) := by sorry

end

end f_strictly_increasing_when_a_eq_one_f_increasing_intervals_l3486_348694


namespace min_value_x_plus_y_l3486_348670

theorem min_value_x_plus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x * y = 2 * x + y) :
  x + y ≥ 3 + 2 * Real.sqrt 2 ∧
  (x + y = 3 + 2 * Real.sqrt 2 ↔ x = Real.sqrt 2 + 1) :=
sorry

end min_value_x_plus_y_l3486_348670


namespace expression_simplification_l3486_348680

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (x / (x^2 - 2*x + 1)) / ((x + 1) / (x^2 - 1) + 1) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l3486_348680


namespace complement_M_intersect_N_l3486_348672

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def N : Set ℝ := {x | x - 2 ≤ x ∧ x < 3}

theorem complement_M_intersect_N :
  ∀ x : ℝ, x ∈ (M ∩ N)ᶜ ↔ x < -2 ∨ x ≥ 1 := by
  sorry

end complement_M_intersect_N_l3486_348672


namespace husband_catches_up_l3486_348690

/-- Yolanda's bike speed in miles per hour -/
def yolanda_speed : ℝ := 20

/-- Yolanda's husband's car speed in miles per hour -/
def husband_speed : ℝ := 40

/-- Time difference between Yolanda and her husband's departure in minutes -/
def time_difference : ℝ := 15

/-- The time it takes for Yolanda's husband to catch up to her in minutes -/
def catch_up_time : ℝ := 15

theorem husband_catches_up :
  yolanda_speed * (catch_up_time + time_difference) / 60 = husband_speed * catch_up_time / 60 :=
sorry

end husband_catches_up_l3486_348690


namespace trader_weight_manipulation_l3486_348600

theorem trader_weight_manipulation :
  ∀ (supplier_weight : ℝ) (cost_price : ℝ),
  supplier_weight > 0 → cost_price > 0 →
  let actual_bought_weight := supplier_weight * 1.1
  let claimed_sell_weight := actual_bought_weight
  let actual_sell_weight := claimed_sell_weight / 1.65
  let weight_difference := claimed_sell_weight - actual_sell_weight
  (cost_price * actual_sell_weight) * 1.65 = cost_price * claimed_sell_weight →
  weight_difference / actual_sell_weight = 0.65 := by
  sorry

end trader_weight_manipulation_l3486_348600


namespace lumber_price_increase_l3486_348658

theorem lumber_price_increase 
  (original_lumber_cost : ℝ)
  (original_nails_cost : ℝ)
  (original_fabric_cost : ℝ)
  (total_cost_increase : ℝ)
  (h1 : original_lumber_cost = 450)
  (h2 : original_nails_cost = 30)
  (h3 : original_fabric_cost = 80)
  (h4 : total_cost_increase = 97) :
  let original_total_cost := original_lumber_cost + original_nails_cost + original_fabric_cost
  let new_total_cost := original_total_cost + total_cost_increase
  let new_lumber_cost := new_total_cost - (original_nails_cost + original_fabric_cost)
  let lumber_cost_increase := new_lumber_cost - original_lumber_cost
  let percentage_increase := (lumber_cost_increase / original_lumber_cost) * 100
  percentage_increase = 21.56 := by
  sorry

end lumber_price_increase_l3486_348658


namespace opposite_of_2023_l3486_348689

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by
  sorry

end opposite_of_2023_l3486_348689


namespace brian_watching_time_l3486_348648

/-- The total time Brian spends watching animal videos -/
def total_watching_time (cat_video_length : ℕ) : ℕ :=
  let dog_video_length := 2 * cat_video_length
  let gorilla_video_length := 2 * (cat_video_length + dog_video_length)
  cat_video_length + dog_video_length + gorilla_video_length

/-- Theorem stating that Brian spends 36 minutes watching animal videos -/
theorem brian_watching_time : total_watching_time 4 = 36 := by
  sorry

end brian_watching_time_l3486_348648


namespace ratio_x_to_y_l3486_348603

theorem ratio_x_to_y (x y : ℝ) (h : (12*x - 5*y) / (15*x - 3*y) = 4/7) : 
  x / y = 23/24 := by
sorry

end ratio_x_to_y_l3486_348603


namespace curve_classification_l3486_348650

-- Define the curve equation
def curve_equation (x y m : ℝ) : Prop :=
  x^2 / (5 - m) + y^2 / (2 - m) = 1

-- Define the condition for m
def m_condition (m : ℝ) : Prop := m < 3

-- Define the result for different ranges of m
def curve_type (m : ℝ) : Prop :=
  (m < 2 → ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ∀ x y : ℝ, curve_equation x y m ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  (2 < m → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    ∀ x y : ℝ, curve_equation x y m ↔ (x^2 / a^2 - y^2 / b^2 = 1))

-- Theorem statement
theorem curve_classification (m : ℝ) :
  m_condition m → curve_type m :=
sorry

end curve_classification_l3486_348650


namespace cow_spots_count_l3486_348634

/-- The number of spots on a cow with given left and right side spot counts. -/
def total_spots (left : ℕ) (right : ℕ) : ℕ := left + right

/-- The number of spots on the right side of the cow, given the number on the left side. -/
def right_spots (left : ℕ) : ℕ := 3 * left + 7

theorem cow_spots_count :
  let left := 16
  let right := right_spots left
  total_spots left right = 71 := by
  sorry

end cow_spots_count_l3486_348634


namespace f_divides_m_minus_n_prime_condition_l3486_348608

def f (x : ℚ) : ℕ :=
  let p := x.num.natAbs
  let q := x.den
  p + q

theorem f_divides_m_minus_n (x : ℚ) (m n : ℕ) (h : m > 0) (k : n > 0) (hx : x > 0) :
  f x = f ((m : ℚ) * x / n) → (f x : ℤ) ∣ |m - n| :=
by sorry

theorem prime_condition (n : ℕ) (hn : n > 1) :
  (∀ x : ℚ, x > 0 → f x = f ((2^n : ℚ) * x) → f x = 2^n - 1) ↔ Nat.Prime n :=
by sorry

end f_divides_m_minus_n_prime_condition_l3486_348608


namespace m_fourth_plus_twice_m_cubed_minus_m_plus_2007_l3486_348665

theorem m_fourth_plus_twice_m_cubed_minus_m_plus_2007 (m : ℝ) 
  (h : m^2 + m - 1 = 0) : 
  m^4 + 2*m^3 - m + 2007 = 2007 := by
  sorry

end m_fourth_plus_twice_m_cubed_minus_m_plus_2007_l3486_348665


namespace parabola_axis_of_symmetry_l3486_348631

/-- Given a parabola passing through points (-2,0) and (4,0), 
    its axis of symmetry is the line x = 1 -/
theorem parabola_axis_of_symmetry :
  ∀ (f : ℝ → ℝ),
  (f (-2) = 0) →
  (f 4 = 0) →
  (∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c) →
  (∀ x, f (1 + x) = f (1 - x)) :=
by sorry

end parabola_axis_of_symmetry_l3486_348631


namespace eight_divides_Q_largest_divisor_eight_largest_divisor_l3486_348677

/-- The product of three consecutive positive even integers -/
def Q (n : ℕ) : ℕ := (2*n) * (2*n + 2) * (2*n + 4)

/-- 8 divides Q for all positive n -/
theorem eight_divides_Q (n : ℕ) : (8 : ℕ) ∣ Q n := by sorry

/-- For any d > 8, there exists an n such that d does not divide Q n -/
theorem largest_divisor (d : ℕ) (h : d > 8) : ∃ n : ℕ, ¬(d ∣ Q n) := by sorry

/-- 8 is the largest integer that divides Q for all positive n -/
theorem eight_largest_divisor : ∀ d : ℕ, (∀ n : ℕ, d ∣ Q n) → d ≤ 8 := by sorry

end eight_divides_Q_largest_divisor_eight_largest_divisor_l3486_348677


namespace james_change_calculation_l3486_348647

/-- Calculates the change James receives after purchasing items with discounts. -/
theorem james_change_calculation (candy_packs : ℕ) (chocolate_bars : ℕ) (chip_bags : ℕ)
  (candy_price : ℚ) (chocolate_price : ℚ) (chip_price : ℚ)
  (candy_discount : ℚ) (chip_discount : ℚ) (payment : ℚ) :
  candy_packs = 3 →
  chocolate_bars = 2 →
  chip_bags = 4 →
  candy_price = 12 →
  chocolate_price = 3 →
  chip_price = 2 →
  candy_discount = 15 / 100 →
  chip_discount = 10 / 100 →
  payment = 50 →
  let candy_total := candy_packs * candy_price * (1 - candy_discount)
  let chocolate_total := chocolate_price -- Due to buy-one-get-one-free offer
  let chip_total := chip_bags * chip_price * (1 - chip_discount)
  let total_cost := candy_total + chocolate_total + chip_total
  payment - total_cost = 9.2 := by sorry

end james_change_calculation_l3486_348647


namespace infinite_division_sum_equal_l3486_348697

/-- Represents a shape with an area -/
class HasArea (α : Type*) where
  area : α → ℝ

/-- Represents a shape that can be divided -/
class Divisible (α : Type*) where
  divide : α → ℝ → α

variable (T : Type*) [HasArea T] [Divisible T]
variable (Q : Type*) [HasArea Q] [Divisible Q]

/-- The sum of areas after infinite divisions -/
noncomputable def infiniteDivisionSum (shape : T) (ratio : ℝ) : ℝ := sorry

/-- Theorem stating the equality of infinite division sums -/
theorem infinite_division_sum_equal
  (triangle : T)
  (quad : Q)
  (ratio : ℝ)
  (h : HasArea.area triangle = 1.5 * HasArea.area quad) :
  infiniteDivisionSum T triangle ratio = infiniteDivisionSum Q quad ratio := by
  sorry

end infinite_division_sum_equal_l3486_348697


namespace equation_represents_line_l3486_348659

-- Define the equation
def equation (x y : ℝ) : Prop :=
  ((x^2 + y^2 - 2*x) * Real.sqrt (x + y - 3) = 0)

-- Define what it means for the equation to represent a line
def represents_line (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ ∀ x y : ℝ, f x y ↔ a*x + b*y + c = 0

-- Theorem statement
theorem equation_represents_line :
  represents_line equation :=
sorry

end equation_represents_line_l3486_348659


namespace rational_cosine_terms_l3486_348640

theorem rational_cosine_terms (x : ℝ) 
  (hS : ∃ q : ℚ, (Real.sin (64 * x) + Real.sin (65 * x)) = ↑q)
  (hC : ∃ q : ℚ, (Real.cos (64 * x) + Real.cos (65 * x)) = ↑q) :
  ∃ (q1 q2 : ℚ), Real.cos (64 * x) = ↑q1 ∧ Real.cos (65 * x) = ↑q2 :=
sorry

end rational_cosine_terms_l3486_348640


namespace area_of_triangle_perimeter_of_triangle_l3486_348639

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 3 ∧ t.b = 2 * t.c

-- Part 1
theorem area_of_triangle (t : Triangle) (h : triangle_conditions t) (h_A : t.A = 2 * Real.pi / 3) :
  (1/2) * t.b * t.c * Real.sin t.A = 9 * Real.sqrt 3 / 14 := by sorry

-- Part 2
theorem perimeter_of_triangle (t : Triangle) (h : triangle_conditions t) (h_BC : 2 * Real.sin t.B - Real.sin t.C = 1) :
  t.a + t.b + t.c = 4 * Real.sqrt 2 - Real.sqrt 5 + 3 ∨ 
  t.a + t.b + t.c = 4 * Real.sqrt 2 + Real.sqrt 5 + 3 := by sorry

end area_of_triangle_perimeter_of_triangle_l3486_348639


namespace pool_filling_time_l3486_348686

/-- Proves that filling a 30,000-gallon pool with 5 hoses, each supplying 2.5 gallons per minute, takes 40 hours. -/
theorem pool_filling_time :
  let pool_capacity : ℝ := 30000
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℝ := 2.5
  let minutes_per_hour : ℕ := 60
  let total_flow_rate : ℝ := num_hoses * flow_rate_per_hose * minutes_per_hour
  pool_capacity / total_flow_rate = 40 := by
  sorry

end pool_filling_time_l3486_348686


namespace tank_volume_proof_l3486_348685

def inletRate : ℝ := 5
def outletRate1 : ℝ := 9
def outletRate2 : ℝ := 8
def emptyTime : ℝ := 2880
def inchesPerFoot : ℝ := 12

def tankVolume : ℝ := 20

theorem tank_volume_proof :
  let netEmptyRate := outletRate1 + outletRate2 - inletRate
  let volumeInCubicInches := netEmptyRate * emptyTime
  let cubicInchesPerCubicFoot := inchesPerFoot ^ 3
  volumeInCubicInches / cubicInchesPerCubicFoot = tankVolume := by
  sorry

#check tank_volume_proof

end tank_volume_proof_l3486_348685


namespace arithmetic_parabola_common_point_l3486_348623

/-- Represents a parabola with coefficients forming an arithmetic progression -/
structure ArithmeticParabola where
  a : ℝ
  d : ℝ

/-- The equation of the parabola given by y = ax^2 + bx + c where b = a + d and c = a + 2d -/
def ArithmeticParabola.equation (p : ArithmeticParabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + (p.a + p.d) * x + (p.a + 2 * p.d)

/-- Theorem stating that all arithmetic parabolas pass through the point (-2, 0) -/
theorem arithmetic_parabola_common_point (p : ArithmeticParabola) :
  p.equation (-2) 0 := by sorry

end arithmetic_parabola_common_point_l3486_348623


namespace sum_of_specific_S_l3486_348681

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

theorem sum_of_specific_S : S 17 + S 33 + S 50 = 1 := by
  sorry

end sum_of_specific_S_l3486_348681


namespace problem_solution_l3486_348625

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xyz : x * y * z = 1)
  (h_x_z : x + 1 / z = 4)
  (h_y_x : y + 1 / x = 20) :
  z + 1 / y = 26 / 79 := by
sorry

end problem_solution_l3486_348625


namespace function_equation_solution_l3486_348641

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_equation_solution :
  (∀ x : ℝ, 2 * f (x - 1) - 3 * f (1 - x) = 5 * x) →
  (∀ x : ℝ, f x = x - 5) :=
by sorry

end function_equation_solution_l3486_348641


namespace a_in_second_quadrant_l3486_348629

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the second quadrant of a rectangular coordinate system -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point A with coordinates dependent on x -/
def A (x : ℝ) : Point :=
  { x := 6 - 2*x, y := x - 5 }

/-- Theorem stating the condition for point A to be in the second quadrant -/
theorem a_in_second_quadrant :
  ∀ x : ℝ, SecondQuadrant (A x) ↔ x > 5 := by
  sorry

end a_in_second_quadrant_l3486_348629


namespace inscribed_rhombus_rectangle_perimeter_l3486_348616

/-- A rhombus inscribed in a rectangle -/
structure InscribedRhombus where
  /-- Length of EA -/
  ea : ℝ
  /-- Length of FB -/
  fb : ℝ
  /-- Length of AD (rhombus side) -/
  ad : ℝ
  /-- Length of BC (rhombus side) -/
  bc : ℝ
  /-- EA is positive -/
  ea_pos : 0 < ea
  /-- FB is positive -/
  fb_pos : 0 < fb
  /-- AD is positive -/
  ad_pos : 0 < ad
  /-- BC is positive -/
  bc_pos : 0 < bc

/-- The perimeter of the rectangle containing the inscribed rhombus -/
def rectangle_perimeter (r : InscribedRhombus) : ℝ :=
  2 * (r.ea + r.ad + r.fb + r.bc)

/-- Theorem stating that for the given measurements, the rectangle perimeter is 238 -/
theorem inscribed_rhombus_rectangle_perimeter :
  ∃ r : InscribedRhombus, r.ea = 12 ∧ r.fb = 25 ∧ r.ad = 37 ∧ r.bc = 45 ∧ rectangle_perimeter r = 238 := by
  sorry

end inscribed_rhombus_rectangle_perimeter_l3486_348616


namespace intersection_point_l3486_348646

/-- The linear function y = 2x + 4 -/
def f (x : ℝ) : ℝ := 2 * x + 4

/-- The y-axis is the vertical line with x-coordinate 0 -/
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}

/-- The graph of the linear function f -/
def graph_f : Set (ℝ × ℝ) := {p | p.2 = f p.1}

/-- The intersection point of the graph of f with the y-axis -/
def intersection : ℝ × ℝ := (0, f 0)

theorem intersection_point :
  intersection ∈ y_axis ∧ intersection ∈ graph_f ∧ intersection = (0, 4) := by
  sorry

end intersection_point_l3486_348646


namespace jake_weight_loss_l3486_348684

theorem jake_weight_loss (total_weight sister_weight jake_weight : ℕ) 
  (h1 : total_weight = 212)
  (h2 : jake_weight = 152)
  (h3 : total_weight = jake_weight + sister_weight) :
  jake_weight - (2 * sister_weight) = 32 := by
  sorry

end jake_weight_loss_l3486_348684


namespace factorial_calculation_l3486_348693

theorem factorial_calculation : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end factorial_calculation_l3486_348693


namespace smallest_composite_no_small_factors_l3486_348636

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 323) ∧ 
  (has_no_small_prime_factors 323) ∧ 
  (∀ m : ℕ, m < 323 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end smallest_composite_no_small_factors_l3486_348636


namespace total_tickets_sold_l3486_348698

theorem total_tickets_sold (child_cost adult_cost total_revenue child_count : ℕ) 
  (h1 : child_cost = 6)
  (h2 : adult_cost = 9)
  (h3 : total_revenue = 1875)
  (h4 : child_count = 50) :
  child_count + (total_revenue - child_cost * child_count) / adult_cost = 225 :=
by sorry

end total_tickets_sold_l3486_348698


namespace parabola_equation_l3486_348619

/-- A parabola with vertex at the origin and directrix y = 4 has the standard equation x^2 = -16y -/
theorem parabola_equation (p : ℝ → ℝ → Prop) :
  (∀ x y, p x y ↔ y = -x^2 / 16) →  -- Standard equation of the parabola
  (∀ x, p x 0) →  -- Vertex at the origin
  (∀ x, p x 4 ↔ x = 0) →  -- Directrix equation
  ∀ x y, p x y ↔ x^2 = -16 * y := by sorry

end parabola_equation_l3486_348619


namespace madhav_rank_l3486_348676

theorem madhav_rank (total_students : ℕ) (rank_from_last : ℕ) (rank_from_start : ℕ) : 
  total_students = 31 →
  rank_from_last = 15 →
  rank_from_start = total_students - (rank_from_last - 1) →
  rank_from_start = 17 := by
sorry

end madhav_rank_l3486_348676
