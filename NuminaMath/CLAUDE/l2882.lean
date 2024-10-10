import Mathlib

namespace quadratic_function_bounds_l2882_288287

/-- Given a quadratic function f and its derivative g, 
    prove bounds on c and g(x) when f is bounded on [-1, 1] -/
theorem quadratic_function_bounds 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x^2 + b * x + c) 
  (hg : ∀ x, g x = a * x + b) 
  (hbound : ∀ x ∈ Set.Icc (-1) 1, |f x| ≤ 1) : 
  (|c| ≤ 1) ∧ (∀ x ∈ Set.Icc (-1) 1, |g x| ≤ 2) := by
  sorry

end quadratic_function_bounds_l2882_288287


namespace probability_of_white_ball_l2882_288229

-- Define the number of balls of each color
def white_balls : ℕ := 6
def yellow_balls : ℕ := 5
def red_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := white_balls + yellow_balls + red_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball : prob_white = 2 / 5 := by
  sorry

end probability_of_white_ball_l2882_288229


namespace height_ratio_equals_similarity_ratio_l2882_288289

/-- Two similar triangles with heights h₁ and h₂ and similarity ratio 1:4 -/
structure SimilarTriangles where
  h₁ : ℝ
  h₂ : ℝ
  h₁_pos : h₁ > 0
  h₂_pos : h₂ > 0
  similarity_ratio : h₁ / h₂ = 1 / 4

/-- The ratio of heights of similar triangles with similarity ratio 1:4 is 1:4 -/
theorem height_ratio_equals_similarity_ratio (t : SimilarTriangles) :
  t.h₁ / t.h₂ = 1 / 4 := by
  sorry

end height_ratio_equals_similarity_ratio_l2882_288289


namespace no_ruler_for_quadratic_sum_l2882_288284

-- Define the type of monotonic functions on [0, 10]
def MonotonicOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 10 → f x ≤ f y

-- State the theorem
theorem no_ruler_for_quadratic_sum :
  ¬ ∃ (f g h : ℝ → ℝ),
    (MonotonicOn f) ∧ (MonotonicOn g) ∧ (MonotonicOn h) ∧
    (∀ x y, 0 ≤ x ∧ x ≤ 10 ∧ 0 ≤ y ∧ y ≤ 10 →
      f x + g y = h (x^2 + x*y + y^2)) :=
by sorry

end no_ruler_for_quadratic_sum_l2882_288284


namespace modified_full_house_probability_l2882_288252

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of ranks in a standard deck -/
def NumberOfRanks : ℕ := 13

/-- Represents the number of cards per rank -/
def CardsPerRank : ℕ := 4

/-- Represents the number of cards drawn -/
def CardsDrawn : ℕ := 6

/-- Represents a modified full house -/
structure ModifiedFullHouse :=
  (rank1 : Fin NumberOfRanks)
  (rank2 : Fin NumberOfRanks)
  (rank3 : Fin NumberOfRanks)
  (h1 : rank1 ≠ rank2)
  (h2 : rank1 ≠ rank3)
  (h3 : rank2 ≠ rank3)

/-- The probability of drawing a modified full house -/
def probabilityModifiedFullHouse : ℚ :=
  24 / 2977

theorem modified_full_house_probability :
  probabilityModifiedFullHouse = (NumberOfRanks * (CardsPerRank.choose 3) * (NumberOfRanks - 1) * (CardsPerRank.choose 2) * ((NumberOfRanks - 2) * CardsPerRank)) / (StandardDeck.choose CardsDrawn) :=
by sorry

end modified_full_house_probability_l2882_288252


namespace faster_train_speed_l2882_288233

/-- Calculates the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 100 →
  crossing_time = 10 →
  (2 * train_length) / crossing_time = 40 / 3 :=
by
  sorry

#check faster_train_speed

end faster_train_speed_l2882_288233


namespace least_non_lucky_multiple_of_10_l2882_288280

def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isLucky (n : ℕ) : Prop := 
  n > 0 ∧ n % sumOfDigits n = 0

def isMultipleOf10 (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 10 * k

theorem least_non_lucky_multiple_of_10 : 
  (∀ m : ℕ, m < 110 → isMultipleOf10 m → isLucky m) ∧ 
  isMultipleOf10 110 ∧ 
  ¬isLucky 110 := by
sorry

end least_non_lucky_multiple_of_10_l2882_288280


namespace methane_hydrate_density_scientific_notation_l2882_288285

theorem methane_hydrate_density_scientific_notation :
  0.00092 = 9.2 * 10^(-4) := by sorry

end methane_hydrate_density_scientific_notation_l2882_288285


namespace find_r_l2882_288239

theorem find_r (k : ℝ) (r : ℝ) 
  (h1 : 5 = k * 3^r) 
  (h2 : 45 = k * 9^r) : 
  r = 2 := by
sorry

end find_r_l2882_288239


namespace max_k_for_exp_inequality_l2882_288216

theorem max_k_for_exp_inequality : 
  (∃ k : ℝ, ∀ x : ℝ, Real.exp x ≥ k * x) ∧ 
  (∀ k : ℝ, (∀ x : ℝ, Real.exp x ≥ k * x) → k ≤ Real.exp 1) := by
  sorry

end max_k_for_exp_inequality_l2882_288216


namespace factor_x9_minus_512_l2882_288257

theorem factor_x9_minus_512 (x : ℝ) : x^9 - 512 = (x^3 - 2) * (x^6 + 2*x^3 + 4) := by
  sorry

end factor_x9_minus_512_l2882_288257


namespace cubic_polynomials_common_roots_l2882_288267

theorem cubic_polynomials_common_roots (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    r^3 + a*r^2 + 20*r + 10 = 0 ∧
    r^3 + b*r^2 + 17*r + 12 = 0 ∧
    s^3 + a*s^2 + 20*s + 10 = 0 ∧
    s^3 + b*s^2 + 17*s + 12 = 0) →
  a = 1 ∧ b = 0 :=
by sorry

end cubic_polynomials_common_roots_l2882_288267


namespace round_trip_distance_l2882_288225

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The distance to the star in light-years -/
def star_distance : ℝ := 25

/-- The duration of the round trip in years -/
def trip_duration : ℝ := 50

/-- The total distance traveled by light in a round trip to the star over the given duration -/
def total_distance : ℝ := 2 * star_distance * light_year_distance

theorem round_trip_distance : total_distance = 5.87e14 := by
  sorry

end round_trip_distance_l2882_288225


namespace solve_equation_l2882_288292

/-- Custom operation for pairs of real numbers -/
def pairOp (a b c d : ℝ) : ℝ := a * c + b * d

theorem solve_equation (x : ℝ) (h : pairOp (2 * x) 3 3 (-1) = 3) : x = 1 := by
  sorry

end solve_equation_l2882_288292


namespace root_sum_ratio_l2882_288294

theorem root_sum_ratio (k : ℝ) (a b : ℝ) : 
  (k * (a^2 - 2*a) + 3*a + 7 = 0) →
  (k * (b^2 - 2*b) + 3*b + 7 = 0) →
  (a / b + b / a = 3 / 4) →
  ∃ (k₁ k₂ : ℝ), k₁ / k₂ + k₂ / k₁ = 433.42 := by sorry

end root_sum_ratio_l2882_288294


namespace pond_length_l2882_288209

/-- Given a rectangular pond with width 10 meters, depth 8 meters, and volume 1600 cubic meters,
    prove that the length of the pond is 20 meters. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) (length : ℝ) :
  width = 10 →
  depth = 8 →
  volume = 1600 →
  volume = length * width * depth →
  length = 20 := by
sorry

end pond_length_l2882_288209


namespace rosie_circles_count_l2882_288235

/-- Proves that given a circular track of 1/4 mile length, if person A runs 3 miles
    and person B runs at twice the speed of person A, then person B circles the track 24 times. -/
theorem rosie_circles_count (track_length : ℝ) (lou_distance : ℝ) (speed_ratio : ℝ) : 
  track_length = 1/4 →
  lou_distance = 3 →
  speed_ratio = 2 →
  (lou_distance * speed_ratio) / track_length = 24 := by
sorry

end rosie_circles_count_l2882_288235


namespace simplified_expression_approximation_l2882_288230

theorem simplified_expression_approximation :
  let expr := Real.sqrt 5 * 5^(1/3) + 18 / (2^2) * 3 - 8^(3/2)
  ∃ ε > 0, |expr + 1.8| < ε ∧ ε < 0.1 :=
by sorry

end simplified_expression_approximation_l2882_288230


namespace factorization_problems_l2882_288277

theorem factorization_problems (x y : ℝ) : 
  ((x^2 + y^2)^2 - 4*x^2*y^2 = (x + y)^2 * (x - y)^2) ∧ 
  (3*x^3 - 12*x^2*y + 12*x*y^2 = 3*x*(x - 2*y)^2) := by
  sorry

end factorization_problems_l2882_288277


namespace max_value_of_expression_l2882_288261

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  x^2 + 2*x*y ≤ 25 ∧ ∃ x y : ℝ, x + y = 5 ∧ x^2 + 2*x*y = 25 := by
  sorry

end max_value_of_expression_l2882_288261


namespace unique_intersection_l2882_288220

/-- Three lines in the 2D plane -/
structure ThreeLines where
  line1 : ℝ → ℝ → ℝ
  line2 : ℝ → ℝ → ℝ
  line3 : ℝ → ℝ → ℝ

/-- The intersection point of three lines -/
def intersection (lines : ThreeLines) (k : ℝ) : Set (ℝ × ℝ) :=
  {p | lines.line1 p.1 p.2 = 0 ∧ lines.line2 p.1 p.2 = 0 ∧ lines.line3 p.1 p.2 = 0}

/-- The theorem stating that k = -1/2 is the unique value for which the given lines intersect at a single point -/
theorem unique_intersection : ∃! k : ℝ, 
  let lines := ThreeLines.mk
    (fun x y => x + k * y)
    (fun x y => 2 * x + 3 * y + 8)
    (fun x y => x - y - 1)
  (∃! p : ℝ × ℝ, p ∈ intersection lines k) ∧ k = -1/2 := by
  sorry

end unique_intersection_l2882_288220


namespace extra_cat_food_l2882_288243

theorem extra_cat_food (food_one_cat food_two_cats : ℝ)
  (h1 : food_one_cat = 0.5)
  (h2 : food_two_cats = 0.9) :
  food_two_cats - food_one_cat = 0.4 := by
  sorry

end extra_cat_food_l2882_288243


namespace p_necessary_not_sufficient_for_q_l2882_288236

/-- A function f: ℝ → ℝ is monotonically increasing -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The function f(x) = x^3 + 2x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The condition p: f(x) is monotonically increasing in (-∞, +∞) -/
def p (m : ℝ) : Prop := MonotonicallyIncreasing (f m)

/-- The condition q: m ≥ 8x / (x^2 + 4) holds for any x > 0 -/
def q (m : ℝ) : Prop := ∀ x > 0, m ≥ 8*x / (x^2 + 4)

/-- p is a necessary but not sufficient condition for q -/
theorem p_necessary_not_sufficient_for_q :
  (∀ m : ℝ, q m → p m) ∧ (∃ m : ℝ, p m ∧ ¬q m) := by sorry

end p_necessary_not_sufficient_for_q_l2882_288236


namespace arithmetic_sequence_ratio_l2882_288249

/-- Arithmetic sequence type -/
structure ArithmeticSequence (α : Type*) [Add α] [Mul α] where
  first : α
  diff : α

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence ℚ) (n : ℕ) : ℚ :=
  n * (2 * seq.first + (n - 1) * seq.diff) / 2

/-- n-th term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence ℚ) (n : ℕ) : ℚ :=
  seq.first + (n - 1) * seq.diff

theorem arithmetic_sequence_ratio 
  (a b : ArithmeticSequence ℚ) 
  (h : ∀ n : ℕ, sum_n a n / sum_n b n = (3 * n - 1) / (n + 3)) :
  nth_term a 8 / (nth_term b 5 + nth_term b 11) = 11 / 9 := by
  sorry

end arithmetic_sequence_ratio_l2882_288249


namespace sum_of_extreme_prime_factors_of_1365_l2882_288213

theorem sum_of_extreme_prime_factors_of_1365 : ∃ (min max : ℕ), 
  (min.Prime ∧ max.Prime ∧ 
   min ∣ 1365 ∧ max ∣ 1365 ∧
   (∀ p : ℕ, p.Prime → p ∣ 1365 → min ≤ p) ∧
   (∀ p : ℕ, p.Prime → p ∣ 1365 → p ≤ max)) ∧
  min + max = 16 :=
sorry

end sum_of_extreme_prime_factors_of_1365_l2882_288213


namespace min_absolute_difference_l2882_288208

theorem min_absolute_difference (a b c d : ℝ) 
  (hab : |a - b| = 5)
  (hbc : |b - c| = 8)
  (hcd : |c - d| = 10) :
  ∃ (m : ℝ), (∀ x, |a - d| ≥ x → m ≤ x) ∧ |a - d| ≥ m ∧ m = 3 :=
sorry

end min_absolute_difference_l2882_288208


namespace m_range_l2882_288204

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 3

-- State the theorem
theorem m_range :
  (∀ x : ℝ, x ≥ 1 → f x + m^2 * f x ≥ f (x - 1) + 3 * f m) ↔
  (m ≤ -1 ∨ m ≥ 0) :=
sorry

end m_range_l2882_288204


namespace tim_dan_balloon_ratio_l2882_288270

theorem tim_dan_balloon_ratio :
  let dan_balloons : ℕ := 29
  let tim_balloons : ℕ := 203
  (tim_balloons / dan_balloons : ℚ) = 7 := by sorry

end tim_dan_balloon_ratio_l2882_288270


namespace fraction_equals_five_l2882_288256

theorem fraction_equals_five (a b k : ℕ+) : 
  (a.val^2 + b.val^2) / (a.val * b.val - 1) = k.val → k = 5 := by
  sorry

end fraction_equals_five_l2882_288256


namespace area_of_right_trapezoid_l2882_288221

/-- 
Given a horizontally placed right trapezoid whose oblique axonometric projection
is an isosceles trapezoid with a bottom angle of 45°, legs of length 1, and 
top base of length 1, the area of the original right trapezoid is 2 + √2.
-/
theorem area_of_right_trapezoid (h : ℝ) (w : ℝ) :
  h = 2 →
  w = 1 + Real.sqrt 2 →
  (1 / 2 : ℝ) * (w + 1) * h = 2 + Real.sqrt 2 := by
  sorry


end area_of_right_trapezoid_l2882_288221


namespace product_of_successive_numbers_l2882_288215

theorem product_of_successive_numbers :
  let n : ℝ := 51.49757275833493
  let product := n * (n + 1)
  ∃ ε > 0, |product - 2703| < ε :=
by
  sorry

end product_of_successive_numbers_l2882_288215


namespace quadratic_inequality_problem_l2882_288258

/-- Given that the inequality ax^2 + 5x - 2 > 0 has the solution set {x|1/2 < x < 2},
    prove the value of a and the solution set of ax^2 - 5x + a^2 - 1 > 0 -/
theorem quadratic_inequality_problem (a : ℝ) :
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) →
  (a = -2 ∧
   ∀ x : ℝ, a*x^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end quadratic_inequality_problem_l2882_288258


namespace condition_relationship_l2882_288260

theorem condition_relationship (x : ℝ) :
  ¬(∀ x, (1 / x ≤ 1 → (1/3)^x ≥ (1/2)^x)) ∧
  ¬(∀ x, ((1/3)^x ≥ (1/2)^x → 1 / x ≤ 1)) :=
by sorry

end condition_relationship_l2882_288260


namespace quadratic_inequality_equivalence_l2882_288214

theorem quadratic_inequality_equivalence (m : ℝ) : 
  (∀ x > 1, x^2 + (m - 2) * x + 3 - m ≥ 0) ↔ m ≥ -1 := by
  sorry

end quadratic_inequality_equivalence_l2882_288214


namespace quadratic_prime_square_l2882_288297

/-- A function that represents the given quadratic expression -/
def f (n : ℕ) : ℤ := 2 * n^2 - 5 * n - 33

/-- Predicate to check if a number is prime -/
def isPrime (p : ℕ) : Prop := Nat.Prime p

/-- The main theorem stating that 6 and 14 are the only natural numbers
    for which f(n) is the square of a prime number -/
theorem quadratic_prime_square : 
  ∀ n : ℕ, (∃ p : ℕ, isPrime p ∧ f n = p^2) ↔ n = 6 ∨ n = 14 :=
sorry

end quadratic_prime_square_l2882_288297


namespace coefficient_m3n5_in_binomial_expansion_l2882_288211

theorem coefficient_m3n5_in_binomial_expansion :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (3 : ℕ)^(8 - k) * (5 : ℕ)^k) = 56 := by
  sorry

end coefficient_m3n5_in_binomial_expansion_l2882_288211


namespace group_five_frequency_l2882_288296

theorem group_five_frequency (total : ℕ) (group1 group2 group3 group4 : ℕ) 
  (h_total : total = 50)
  (h_group1 : group1 = 2)
  (h_group2 : group2 = 8)
  (h_group3 : group3 = 15)
  (h_group4 : group4 = 5) :
  (total - group1 - group2 - group3 - group4 : ℚ) / total = 0.4 := by
sorry

end group_five_frequency_l2882_288296


namespace ceiling_minus_x_equals_half_l2882_288298

theorem ceiling_minus_x_equals_half (x : ℝ) (h : x - ⌊x⌋ = 0.5) : ⌈x⌉ - x = 0.5 := by
  sorry

end ceiling_minus_x_equals_half_l2882_288298


namespace ellipse_equation_through_points_l2882_288228

/-- The standard equation of an ellipse passing through (-3, 0) and (0, -2) -/
theorem ellipse_equation_through_points :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ 
    (x^2 / 9 + y^2 / 4 = 1)) ∧
  (-3^2 / a^2 + 0^2 / b^2 = 1) ∧
  (0^2 / a^2 + (-2)^2 / b^2 = 1) :=
by sorry

end ellipse_equation_through_points_l2882_288228


namespace gathering_handshakes_l2882_288293

/-- Represents the number of handshakes in a gathering with specific conditions -/
def handshakes_in_gathering (total_people : ℕ) (group_a : ℕ) (group_b : ℕ) 
  (group_b_connected : ℕ) (connections : ℕ) : ℕ :=
  let group_b_isolated := group_b - group_b_connected
  let handshakes_isolated_to_a := group_b_isolated * group_a
  let handshakes_connected_to_a := group_b_connected * (group_a - connections)
  let handshakes_within_b := (group_b * (group_b - 1)) / 2
  handshakes_isolated_to_a + handshakes_connected_to_a + handshakes_within_b

theorem gathering_handshakes : 
  handshakes_in_gathering 40 30 10 3 5 = 330 :=
sorry

end gathering_handshakes_l2882_288293


namespace min_same_score_competition_l2882_288264

/-- Represents a math competition with fill-in-the-blank and short-answer questions. -/
structure MathCompetition where
  fill_in_blank_count : Nat
  fill_in_blank_points : Nat
  short_answer_count : Nat
  short_answer_points : Nat
  participant_count : Nat

/-- Calculates the minimum number of participants with the same score. -/
def min_same_score (comp : MathCompetition) : Nat :=
  let max_score := comp.fill_in_blank_count * comp.fill_in_blank_points +
                   comp.short_answer_count * comp.short_answer_points
  let distinct_scores := (comp.fill_in_blank_count + 1) * (comp.short_answer_count + 1)
  (comp.participant_count + distinct_scores - 1) / distinct_scores

/-- Theorem stating the minimum number of participants with the same score
    in the given competition configuration. -/
theorem min_same_score_competition :
  let comp := MathCompetition.mk 8 4 6 7 400
  min_same_score comp = 8 := by
  sorry


end min_same_score_competition_l2882_288264


namespace volume_of_S_l2882_288254

/-- A line in ℝ³ -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Distance from a point to a line in ℝ³ -/
def distPointToLine (p : ℝ × ℝ × ℝ) (l : Line3D) : ℝ :=
  sorry

/-- Distance between two points in ℝ³ -/
def distBetweenPoints (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The set S as described in the problem -/
def S (ℓ : Line3D) (P : ℝ × ℝ × ℝ) : Set (ℝ × ℝ × ℝ) :=
  {X | distPointToLine X ℓ ≥ 2 * distBetweenPoints X P}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (s : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

theorem volume_of_S (ℓ : Line3D) (P : ℝ × ℝ × ℝ) (d : ℝ) 
    (h_d : d > 0) (h_dist : distPointToLine P ℓ = d) :
    volume (S ℓ P) = (16 * Real.pi * d^3) / (27 * Real.sqrt 3) :=
  sorry

end volume_of_S_l2882_288254


namespace negation_of_proposition_l2882_288295

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 2 → x^2 - 1 > 0)) ↔ (∃ x : ℝ, x > 2 ∧ x^2 - 1 ≤ 0) :=
by sorry

end negation_of_proposition_l2882_288295


namespace integral_x_squared_l2882_288276

theorem integral_x_squared : ∫ x in (0:ℝ)..1, x^2 = (1/3 : ℝ) := by sorry

end integral_x_squared_l2882_288276


namespace base_difference_proof_l2882_288272

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The digits of 543210 in base 8 -/
def num1_digits : List ℕ := [5, 4, 3, 2, 1, 0]

/-- The digits of 43210 in base 5 -/
def num2_digits : List ℕ := [4, 3, 2, 1, 0]

theorem base_difference_proof :
  to_base_10 num1_digits 8 - to_base_10 num2_digits 5 = 177966 := by
  sorry

end base_difference_proof_l2882_288272


namespace floor_times_self_eq_90_l2882_288210

theorem floor_times_self_eq_90 :
  ∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 90 ∧ x = 10 := by
  sorry

end floor_times_self_eq_90_l2882_288210


namespace largest_absolute_value_l2882_288255

theorem largest_absolute_value : 
  let numbers : List ℤ := [4, -5, 0, -1]
  ∀ x ∈ numbers, |x| ≤ |-5| :=
by sorry

end largest_absolute_value_l2882_288255


namespace absolute_value_sum_zero_l2882_288250

theorem absolute_value_sum_zero (x y : ℝ) :
  |x - 6| + |y + 5| = 0 → x - y = 11 := by
sorry

end absolute_value_sum_zero_l2882_288250


namespace evaluate_expression_l2882_288262

theorem evaluate_expression : 7 - 5 * (9 - 4^2) * 3 = 112 := by sorry

end evaluate_expression_l2882_288262


namespace shaded_area_value_l2882_288237

theorem shaded_area_value (d : ℝ) : 
  (3 * (2 - (1/2 * π * 1^2))) = 6 + d * π → d = -3/2 := by
  sorry

end shaded_area_value_l2882_288237


namespace m_range_l2882_288244

-- Define the function f on the interval [-2, 2]
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_domain : ∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≠ 0
axiom f_even : ∀ x, -2 ≤ x ∧ x ≤ 2 → f (-x) = f x
axiom f_decreasing : ∀ a b, 0 ≤ a ∧ a ≤ 2 → 0 ≤ b ∧ b ≤ 2 → a ≠ b → (f a - f b) / (a - b) < 0

-- Define the theorem
theorem m_range (m : ℝ) (h : f (1 - m) < f m) : -1 ≤ m ∧ m < 1/2 :=
sorry

end m_range_l2882_288244


namespace relationship_between_abc_l2882_288274

theorem relationship_between_abc (x y a b c : ℝ) 
  (ha : a = x + y) 
  (hb : b = x * y) 
  (hc : c = x^2 + y^2) : 
  a^2 = c + 2*b := by
sorry

end relationship_between_abc_l2882_288274


namespace fraction_problem_l2882_288281

theorem fraction_problem (n : ℕ) : 
  (n : ℚ) / (2 * n + 4) = 3 / 7 → n = 12 := by
  sorry

end fraction_problem_l2882_288281


namespace nine_digit_sum_exists_l2882_288247

def is_nine_digit_permutation (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  ∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → ∃ k : ℕ, n / 10^k % 10 = d

theorem nine_digit_sum_exists : ∃ a b : ℕ, 
  is_nine_digit_permutation a ∧ 
  is_nine_digit_permutation b ∧ 
  a + b = 987654321 :=
sorry

end nine_digit_sum_exists_l2882_288247


namespace student_weight_difference_l2882_288286

/-- Proves the difference in average weights given specific conditions about a group of students --/
theorem student_weight_difference (n : ℕ) (initial_avg : ℝ) (joe_weight : ℝ) (new_avg : ℝ) :
  initial_avg = 30 →
  joe_weight = 42 →
  new_avg = 31 →
  (n * initial_avg + joe_weight) / (n + 1) = new_avg →
  ((n + 1) * new_avg - 2 * initial_avg) / (n - 1) = initial_avg →
  abs (((n + 1) * new_avg - n * initial_avg) / 2 - joe_weight) = 6 := by
  sorry

end student_weight_difference_l2882_288286


namespace bisection_sqrt2_approximation_l2882_288238

theorem bisection_sqrt2_approximation :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2| ≤ 0.1 := by
  sorry

end bisection_sqrt2_approximation_l2882_288238


namespace building_floors_l2882_288217

theorem building_floors (floors_B floors_C : ℕ) : 
  (floors_C = 5 * floors_B - 6) →
  (floors_C = 59) →
  (∃ floors_A : ℕ, floors_A = floors_B - 9 ∧ floors_A = 4) :=
by
  sorry

end building_floors_l2882_288217


namespace power_seven_145_mod_12_l2882_288231

theorem power_seven_145_mod_12 : 7^145 % 12 = 7 := by
  sorry

end power_seven_145_mod_12_l2882_288231


namespace right_triangle_with_inscribed_circle_legs_l2882_288279

/-- Represents a right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of one leg -/
  a : ℝ
  /-- Length of the other leg -/
  b : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Distance from center of inscribed circle to one acute angle vertex -/
  d1 : ℝ
  /-- Distance from center of inscribed circle to other acute angle vertex -/
  d2 : ℝ
  /-- a and b are positive -/
  ha : 0 < a
  hb : 0 < b
  /-- r is positive -/
  hr : 0 < r
  /-- d1 and d2 are positive -/
  hd1 : 0 < d1
  hd2 : 0 < d2
  /-- Relationship between leg length and distance to vertex -/
  h1 : d1^2 = r^2 + (a - r)^2
  h2 : d2^2 = r^2 + (b - r)^2

/-- The main theorem -/
theorem right_triangle_with_inscribed_circle_legs
  (t : RightTriangleWithInscribedCircle)
  (h1 : t.d1 = Real.sqrt 5)
  (h2 : t.d2 = Real.sqrt 10) :
  t.a = 4 ∧ t.b = 3 := by
  sorry


end right_triangle_with_inscribed_circle_legs_l2882_288279


namespace smallest_multiple_of_seven_l2882_288278

theorem smallest_multiple_of_seven (x : ℕ) : 
  (∃ k : ℕ, x = 7 * k) ∧ 
  (x^2 > 150) ∧ 
  (x < 40) → 
  x = 14 :=
sorry

end smallest_multiple_of_seven_l2882_288278


namespace algebraic_expression_value_l2882_288299

theorem algebraic_expression_value : 
  let x : ℝ := -1
  3 * x^2 + 2 * x - 1 = 0 := by
sorry

end algebraic_expression_value_l2882_288299


namespace x_pos_sufficient_not_necessary_for_abs_x_pos_l2882_288245

theorem x_pos_sufficient_not_necessary_for_abs_x_pos :
  (∃ (x : ℝ), |x| > 0 ∧ x ≤ 0) ∧
  (∀ (x : ℝ), x > 0 → |x| > 0) :=
by sorry

end x_pos_sufficient_not_necessary_for_abs_x_pos_l2882_288245


namespace cube_edge_length_from_circumscribed_sphere_volume_l2882_288241

theorem cube_edge_length_from_circumscribed_sphere_volume :
  ∀ (edge_length : ℝ) (sphere_volume : ℝ),
    sphere_volume = 4 * Real.pi / 3 →
    (∃ (sphere_radius : ℝ),
      sphere_volume = 4 / 3 * Real.pi * sphere_radius ^ 3 ∧
      edge_length ^ 2 * 3 = (2 * sphere_radius) ^ 2) →
    edge_length = 2 * Real.sqrt 3 / 3 := by
  sorry

end cube_edge_length_from_circumscribed_sphere_volume_l2882_288241


namespace movie_collection_size_l2882_288271

theorem movie_collection_size :
  ∀ (dvd blu : ℕ),
  dvd > 0 ∧ blu > 0 →
  dvd / blu = 17 / 4 →
  dvd / (blu - 4) = 9 / 2 →
  dvd + blu = 378 := by
sorry

end movie_collection_size_l2882_288271


namespace reciprocal_of_negative_two_l2882_288283

theorem reciprocal_of_negative_two :
  (1 : ℚ) / (-2 : ℚ) = -1/2 := by sorry

end reciprocal_of_negative_two_l2882_288283


namespace probability_multiple_of_seven_l2882_288234

/-- The probability of selecting a page number that is a multiple of 7 from a book with 500 pages -/
theorem probability_multiple_of_seven (total_pages : ℕ) (h : total_pages = 500) :
  (Finset.filter (fun n => n % 7 = 0) (Finset.range total_pages)).card / total_pages = 71 / 500 :=
by sorry

end probability_multiple_of_seven_l2882_288234


namespace expressions_equal_thirty_l2882_288259

theorem expressions_equal_thirty : 
  (6 * 6 - 6 = 30) ∧ 
  (5 * 5 + 5 = 30) ∧ 
  (33 - 3 = 30) ∧ 
  (3^3 + 3 = 30) := by
  sorry

#check expressions_equal_thirty

end expressions_equal_thirty_l2882_288259


namespace polynomial_property_l2882_288226

/-- A polynomial of the form 2x^3 - 30x^2 + cx -/
def P (c : ℤ) (x : ℤ) : ℤ := 2 * x^3 - 30 * x^2 + c * x

/-- The property that P(x) yields consecutive integers for consecutive integer inputs -/
def consecutive_values (c : ℤ) : Prop :=
  ∀ a : ℤ, ∃ k : ℤ, P c (a - 1) = k - 1 ∧ P c a = k ∧ P c (a + 1) = k + 1

theorem polynomial_property :
  ∀ c : ℤ, consecutive_values c → c = 149 := by
  sorry

end polynomial_property_l2882_288226


namespace matt_first_quarter_score_l2882_288290

/-- Calculates the total score in basketball given the number of 2-point and 3-point shots made. -/
def totalScore (twoPointShots threePointShots : ℕ) : ℕ :=
  2 * twoPointShots + 3 * threePointShots

/-- Proves that Matt's score in the first quarter is 14 points. -/
theorem matt_first_quarter_score :
  totalScore 4 2 = 14 := by
  sorry

end matt_first_quarter_score_l2882_288290


namespace log_equation_solution_l2882_288240

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  (Real.log y^3 / Real.log 3) + (Real.log y / Real.log (1/3)) = 6 → y = 27 := by
  sorry

end log_equation_solution_l2882_288240


namespace line_segment_parameter_sum_of_squares_l2882_288246

/-- A line segment parameterized by t, connecting two points in 2D space. -/
structure LineSegment where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The point on the line segment at a given parameter t. -/
def LineSegment.point_at (l : LineSegment) (t : ℝ) : ℝ × ℝ :=
  (l.a * t + l.b, l.c * t + l.d)

theorem line_segment_parameter_sum_of_squares :
  ∀ l : LineSegment,
  (l.point_at 0 = (-3, 5)) →
  (l.point_at 0.5 = (0.5, 7.5)) →
  (l.point_at 1 = (4, 10)) →
  l.a^2 + l.b^2 + l.c^2 + l.d^2 = 108 := by
  sorry

end line_segment_parameter_sum_of_squares_l2882_288246


namespace log_inequality_l2882_288232

theorem log_inequality : 
  let x := Real.log 2 / Real.log 5
  let y := Real.log 2
  let z := Real.sqrt 2
  x < y ∧ y < z := by sorry

end log_inequality_l2882_288232


namespace no_solution_to_equation_l2882_288218

theorem no_solution_to_equation :
  ¬∃ x : ℝ, (1 / (x + 8) + 1 / (x + 5) + 1 / (x + 1) = 1 / (x + 11) + 1 / (x + 2) + 1 / (x + 1)) :=
by sorry

end no_solution_to_equation_l2882_288218


namespace product_of_cosines_l2882_288201

theorem product_of_cosines : 
  (1 + Real.cos (π/8)) * (1 + Real.cos (3*π/8)) * (1 + Real.cos (5*π/8)) * (1 + Real.cos (7*π/8)) = 1/8 := by
  sorry

end product_of_cosines_l2882_288201


namespace expected_value_S_l2882_288288

def num_boys : ℕ := 7
def num_girls : ℕ := 13
def total_people : ℕ := num_boys + num_girls

def prob_boy_girl : ℚ := (num_boys : ℚ) / total_people * (num_girls : ℚ) / (total_people - 1)
def prob_girl_boy : ℚ := (num_girls : ℚ) / total_people * (num_boys : ℚ) / (total_people - 1)

def prob_adjacent_pair : ℚ := prob_boy_girl + prob_girl_boy
def num_adjacent_pairs : ℕ := total_people - 1

theorem expected_value_S : (num_adjacent_pairs : ℚ) * prob_adjacent_pair = 91 / 10 := by
  sorry

end expected_value_S_l2882_288288


namespace tv_price_proof_l2882_288222

theorem tv_price_proof (X : ℝ) : 
  X * (1 + 0.4) * 0.8 - X = 270 → X = 2250 := by
  sorry

end tv_price_proof_l2882_288222


namespace final_result_proof_l2882_288223

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 990) : 
  (chosen_number / 9 : ℚ) - 100 = 10 := by
  sorry

end final_result_proof_l2882_288223


namespace savings_period_is_four_months_l2882_288224

-- Define the savings and stock parameters
def wife_weekly_savings : ℕ := 100
def husband_monthly_savings : ℕ := 225
def stock_price : ℕ := 50
def shares_bought : ℕ := 25

-- Define the function to calculate the number of months saved
def months_saved : ℕ :=
  let total_investment := stock_price * shares_bought
  let total_savings := total_investment * 2
  let monthly_savings := wife_weekly_savings * 4 + husband_monthly_savings
  total_savings / monthly_savings

-- Theorem statement
theorem savings_period_is_four_months :
  months_saved = 4 :=
sorry

end savings_period_is_four_months_l2882_288224


namespace hyperbola_equation_l2882_288242

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ k : ℝ, k * x + y = k * x + 2 → a = b) →
  (∃ c : ℝ, c^2 = 24 - 16 ∧ c^2 = a^2 + b^2) →
  a^2 = 4 ∧ b^2 = 4 :=
by sorry

end hyperbola_equation_l2882_288242


namespace carton_height_calculation_l2882_288212

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of items that can fit along one dimension -/
def maxItemsAlongDimension (containerSize itemSize : ℕ) : ℕ :=
  containerSize / itemSize

/-- Calculates the total number of items that can fit on the base of the container -/
def itemsOnBase (containerBase itemBase : Dimensions) : ℕ :=
  (maxItemsAlongDimension containerBase.length itemBase.length) *
  (maxItemsAlongDimension containerBase.width itemBase.width)

/-- Calculates the number of layers of items that can be stacked in the container -/
def numberOfLayers (maxItems itemsPerLayer : ℕ) : ℕ :=
  maxItems / itemsPerLayer

/-- Calculates the height of the container based on the number of layers and item height -/
def containerHeight (layers itemHeight : ℕ) : ℕ :=
  layers * itemHeight

theorem carton_height_calculation (cartonBase : Dimensions) (soapBox : Dimensions) (maxSoapBoxes : ℕ) :
  cartonBase.length = 25 →
  cartonBase.width = 42 →
  soapBox.length = 7 →
  soapBox.width = 12 →
  soapBox.height = 5 →
  maxSoapBoxes = 150 →
  containerHeight (numberOfLayers maxSoapBoxes (itemsOnBase cartonBase soapBox)) soapBox.height = 80 := by
  sorry

#check carton_height_calculation

end carton_height_calculation_l2882_288212


namespace spring_problem_l2882_288202

/-- Represents the length of a spring as a function of mass -/
def spring_length (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem spring_problem (k : ℝ) :
  spring_length k 6 0 = 6 →
  spring_length k 6 4 = 7.2 →
  spring_length k 6 5 = 7.5 := by
  sorry

end spring_problem_l2882_288202


namespace fill_fraction_in_three_minutes_l2882_288265

/-- Represents the fraction of a cistern filled in a given time -/
def fractionFilled (totalTime minutes : ℚ) : ℚ :=
  minutes / totalTime

theorem fill_fraction_in_three_minutes :
  let totalTime : ℚ := 33
  let minutes : ℚ := 3
  fractionFilled totalTime minutes = 1 / 11 := by
  sorry

end fill_fraction_in_three_minutes_l2882_288265


namespace inverse_A_times_B_l2882_288269

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ := !![0, 1; 2, 3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![2, 0; 1, 8]

theorem inverse_A_times_B :
  A⁻¹ * B = !![-(5/2), 4; 2, 0] := by sorry

end inverse_A_times_B_l2882_288269


namespace exist_special_numbers_l2882_288207

/-- Sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of two numbers satisfying the given conditions -/
theorem exist_special_numbers : 
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ A = 2016 * B ∧ sum_of_digits A = sum_of_digits B / 2016 := by
  sorry

end exist_special_numbers_l2882_288207


namespace negation_equivalence_l2882_288200

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) :=
by sorry

end negation_equivalence_l2882_288200


namespace tub_ratio_is_one_third_l2882_288227

/-- Represents the number of tubs in various categories -/
structure TubCounts where
  total : ℕ
  storage : ℕ
  usual_vendor : ℕ

/-- Calculates the ratio of tubs bought from new vendor to usual vendor -/
def tub_ratio (t : TubCounts) : Rat :=
  let new_vendor := t.total - t.storage - t.usual_vendor
  (new_vendor : Rat) / t.usual_vendor

/-- Theorem stating the ratio of tubs bought from new vendor to usual vendor -/
theorem tub_ratio_is_one_third (t : TubCounts) 
  (h_total : t.total = 100)
  (h_storage : t.storage = 20)
  (h_usual : t.usual_vendor = 60) :
  tub_ratio t = 1 / 3 := by
  sorry

end tub_ratio_is_one_third_l2882_288227


namespace ellipse_eccentricity_l2882_288275

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 (where a > b > 0),
    with foci F₁ and F₂, and points A and B on the ellipse satisfying
    AF₁ = 3F₁B and ∠BAF₂ = 90°, prove that the eccentricity of the ellipse
    is √2/2. -/
theorem ellipse_eccentricity (a b : ℝ) (A B F₁ F₂ : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧
  (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
  (F₁.1^2 + F₁.2^2 = (a^2 - b^2)) ∧
  (F₂.1^2 + F₂.2^2 = (a^2 - b^2)) ∧
  (A - F₁ = 3 • (F₁ - B)) ∧
  ((A - B) • (A - F₂) = 0) →
  Real.sqrt ((a^2 - b^2) / a^2) = Real.sqrt 2 / 2 :=
by sorry

end ellipse_eccentricity_l2882_288275


namespace cylinder_min_surface_area_l2882_288203

/-- For a right circular cylinder with fixed volume, the surface area is minimized when the diameter equals the height -/
theorem cylinder_min_surface_area (V : ℝ) (h V_pos : V > 0) :
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧
  V = π * r^2 * h ∧
  (∀ (r' h' : ℝ), r' > 0 → h' > 0 → V = π * r'^2 * h' →
    2 * π * r^2 + 2 * π * r * h ≤ 2 * π * r'^2 + 2 * π * r' * h') ∧
  h = 2 * r := by
  sorry

#check cylinder_min_surface_area

end cylinder_min_surface_area_l2882_288203


namespace missing_data_point_l2882_288291

def linear_regression (x y : ℝ) := 0.28 * x + 0.16 = y

def data_points : List (ℝ × ℝ) := [(1, 0.5), (3, 1), (4, 1.4), (5, 1.5)]

theorem missing_data_point : 
  ∀ (a : ℝ), 
  (∀ (point : ℝ × ℝ), point ∈ data_points → linear_regression point.1 point.2) →
  linear_regression 2 a →
  linear_regression 3 ((0.5 + a + 1 + 1.4 + 1.5) / 5) →
  a = 0.6 := by sorry

end missing_data_point_l2882_288291


namespace triangle_area_two_solutions_l2882_288253

theorem triangle_area_two_solutions (A B C : ℝ) (AB AC : ℝ) :
  B = π / 6 →  -- 30 degrees in radians
  AB = 2 * Real.sqrt 3 →
  AC = 2 →
  let area := (1 / 2) * AB * AC * Real.sin A
  area = 2 * Real.sqrt 3 ∨ area = Real.sqrt 3 :=
by sorry

end triangle_area_two_solutions_l2882_288253


namespace surjective_sum_iff_constant_l2882_288268

-- Define a surjective function over ℤ
def Surjective (g : ℤ → ℤ) : Prop :=
  ∀ y : ℤ, ∃ x : ℤ, g x = y

-- Define the property that f + g is surjective for all surjective g
def SurjectiveSum (f : ℤ → ℤ) : Prop :=
  ∀ g : ℤ → ℤ, Surjective g → Surjective (fun x ↦ f x + g x)

-- Define a constant function
def ConstantFunction (f : ℤ → ℤ) : Prop :=
  ∃ c : ℤ, ∀ x : ℤ, f x = c

-- Theorem statement
theorem surjective_sum_iff_constant (f : ℤ → ℤ) :
  SurjectiveSum f ↔ ConstantFunction f :=
sorry

end surjective_sum_iff_constant_l2882_288268


namespace quadratic_inequality_solution_set_l2882_288282

theorem quadratic_inequality_solution_set (a x : ℝ) :
  let inequality := a * x^2 + (a - 1) * x - 1 < 0
  (a = 0 → (inequality ↔ x > -1)) ∧
  (a > 0 → (inequality ↔ -1 < x ∧ x < 1/a)) ∧
  (-1 < a ∧ a < 0 → (inequality ↔ x < 1/a ∨ x > -1)) ∧
  (a = -1 → (inequality ↔ x ≠ -1)) ∧
  (a < -1 → (inequality ↔ x < -1 ∨ x > 1/a)) :=
by sorry

end quadratic_inequality_solution_set_l2882_288282


namespace second_train_length_second_train_length_is_100_l2882_288266

/-- Calculates the length of the second train given the speeds of two trains moving in opposite directions, the length of the first train, and the time it takes for them to pass each other completely. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (length1 : ℝ) 
  (pass_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_mps := relative_speed * 1000 / 3600
  let total_distance := relative_speed_mps * pass_time
  total_distance - length1

/-- Proves that the length of the second train is 100 meters under the given conditions. -/
theorem second_train_length_is_100 : 
  second_train_length 80 70 150 5.999520038396928 = 100 := by
  sorry

end second_train_length_second_train_length_is_100_l2882_288266


namespace hyperbola_eccentricity_l2882_288205

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±x is √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 1) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 2 := by
  sorry

end hyperbola_eccentricity_l2882_288205


namespace quadratic_function_k_value_l2882_288263

theorem quadratic_function_k_value (a b c k : ℤ) :
  let g := fun (x : ℤ) => a * x^2 + b * x + c
  g 1 = 0 ∧
  20 < g 5 ∧ g 5 < 30 ∧
  40 < g 6 ∧ g 6 < 50 ∧
  3000 * k < g 100 ∧ g 100 < 3000 * (k + 1) →
  k = 9 := by sorry

end quadratic_function_k_value_l2882_288263


namespace family_movie_night_l2882_288251

/-- Proves the number of adults in a family given ticket prices and payment information -/
theorem family_movie_night (regular_price : ℕ) (child_discount : ℕ) (total_payment : ℕ) (change : ℕ) (num_children : ℕ) : 
  regular_price = 9 →
  child_discount = 2 →
  total_payment = 40 →
  change = 1 →
  num_children = 3 →
  (total_payment - change - num_children * (regular_price - child_discount)) / regular_price = 2 := by
sorry

end family_movie_night_l2882_288251


namespace gumball_difference_l2882_288273

def carl_gumballs : ℕ := 16
def lewis_gumballs : ℕ := 12
def amy_gumballs : ℕ := 20

theorem gumball_difference (x y : ℕ) :
  (18 * 5 ≤ carl_gumballs + lewis_gumballs + amy_gumballs + x + y) ∧
  (carl_gumballs + lewis_gumballs + amy_gumballs + x + y ≤ 27 * 5) →
  (87 : ℕ) - (42 : ℕ) = 45 := by
  sorry

end gumball_difference_l2882_288273


namespace class_size_problem_l2882_288206

theorem class_size_problem (x : ℕ) : 
  (40 * x + 50 * 90) / (x + 50 : ℝ) = 71.25 → x = 30 := by
  sorry

end class_size_problem_l2882_288206


namespace four_carpenters_in_five_hours_l2882_288248

/-- Represents the number of desks built by a given number of carpenters in a specific time -/
def desks_built (carpenters : ℕ) (hours : ℚ) : ℚ :=
  sorry

/-- Two carpenters can build 2 desks in 2.5 hours -/
axiom two_carpenters_rate : desks_built 2 (5/2) = 2

/-- All carpenters work at the same pace -/
axiom same_pace (c₁ c₂ : ℕ) (h : ℚ) :
  c₁ * desks_built c₂ h = c₂ * desks_built c₁ h

theorem four_carpenters_in_five_hours :
  desks_built 4 5 = 8 := by
  sorry

end four_carpenters_in_five_hours_l2882_288248


namespace dice_labeling_exists_l2882_288219

/-- Represents a 6-sided die with integer labels -/
def Die := Fin 6 → ℕ

/-- Checks if a given labeling of two dice produces all sums from 1 to 36 -/
def valid_labeling (d1 d2 : Die) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 36 → ∃ (i j : Fin 6), d1 i + d2 j = n

/-- There exists a labeling for two dice that produces all sums from 1 to 36 with equal probabilities -/
theorem dice_labeling_exists : ∃ (d1 d2 : Die), valid_labeling d1 d2 := by
  sorry

end dice_labeling_exists_l2882_288219
