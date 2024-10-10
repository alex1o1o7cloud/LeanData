import Mathlib

namespace min_sum_squared_distances_l2576_257608

/-- Represents a point in 1D space -/
structure Point1D where
  x : ℝ

/-- Distance between two points in 1D -/
def distance (p q : Point1D) : ℝ := |p.x - q.x|

/-- Sum of squared distances from a point to multiple points -/
def sumSquaredDistances (p : Point1D) (points : List Point1D) : ℝ :=
  points.foldl (fun sum q => sum + (distance p q)^2) 0

/-- The problem statement -/
theorem min_sum_squared_distances :
  ∃ (a b c d e : Point1D),
    distance a b = 2 ∧
    distance b c = 2 ∧
    distance c d = 3 ∧
    distance d e = 7 ∧
    (∀ p : Point1D, sumSquaredDistances p [a, b, c, d, e] ≥ 133.2) ∧
    (∃ q : Point1D, sumSquaredDistances q [a, b, c, d, e] = 133.2) := by
  sorry

end min_sum_squared_distances_l2576_257608


namespace coin_order_correct_l2576_257683

-- Define the type for coins
inductive Coin : Type
  | A | B | C | D | E | F

-- Define a relation for one coin being above another
def IsAbove : Coin → Coin → Prop := sorry

-- Define the correct order of coins
def CorrectOrder : List Coin := [Coin.F, Coin.C, Coin.E, Coin.D, Coin.A, Coin.B]

-- State the theorem
theorem coin_order_correct (coins : List Coin) 
  (h1 : IsAbove Coin.F Coin.C)
  (h2 : IsAbove Coin.F Coin.E)
  (h3 : IsAbove Coin.F Coin.A)
  (h4 : IsAbove Coin.F Coin.D)
  (h5 : IsAbove Coin.F Coin.B)
  (h6 : IsAbove Coin.C Coin.A)
  (h7 : IsAbove Coin.C Coin.D)
  (h8 : IsAbove Coin.C Coin.B)
  (h9 : IsAbove Coin.E Coin.A)
  (h10 : IsAbove Coin.E Coin.B)
  (h11 : IsAbove Coin.D Coin.B)
  (h12 : coins.length = 6)
  (h13 : coins.Nodup)
  (h14 : ∀ c, c ∈ coins ↔ c ∈ [Coin.A, Coin.B, Coin.C, Coin.D, Coin.E, Coin.F]) :
  coins = CorrectOrder := by sorry


end coin_order_correct_l2576_257683


namespace absolute_value_fraction_less_than_one_l2576_257679

theorem absolute_value_fraction_less_than_one (a b : ℝ) 
  (ha : |a| < 1) (hb : |b| < 1) : 
  |((a + b) / (1 + a * b))| < 1 := by
  sorry

end absolute_value_fraction_less_than_one_l2576_257679


namespace max_six_yuan_items_l2576_257689

/-- Represents the number of items bought at each price point -/
structure ItemCounts where
  twoYuan : ℕ
  fourYuan : ℕ
  sixYuan : ℕ

/-- The problem constraints -/
def isValidPurchase (items : ItemCounts) : Prop :=
  items.twoYuan + items.fourYuan + items.sixYuan = 16 ∧
  2 * items.twoYuan + 4 * items.fourYuan + 6 * items.sixYuan = 60

/-- The theorem stating the maximum number of 6-yuan items -/
theorem max_six_yuan_items :
  ∃ (max : ℕ), max = 7 ∧
  (∀ (items : ItemCounts), isValidPurchase items → items.sixYuan ≤ max) ∧
  (∃ (items : ItemCounts), isValidPurchase items ∧ items.sixYuan = max) := by
  sorry

end max_six_yuan_items_l2576_257689


namespace escalator_ride_time_l2576_257673

/-- Represents the scenario of Clea walking on an escalator -/
structure EscalatorScenario where
  /-- Clea's walking speed on stationary escalator (units per second) -/
  walkingSpeed : ℝ
  /-- Total distance of the escalator (units) -/
  escalatorDistance : ℝ
  /-- Speed of the moving escalator (units per second) -/
  escalatorSpeed : ℝ

/-- Time taken for Clea to walk down the stationary escalator -/
def stationaryTime (scenario : EscalatorScenario) : ℝ := 70

/-- Time taken for Clea to walk down the moving escalator -/
def movingTime (scenario : EscalatorScenario) : ℝ := 30

/-- Clea's walking speed increase factor on moving escalator -/
def speedIncreaseFactor : ℝ := 1.5

/-- Theorem stating the time taken for Clea to ride the escalator without walking -/
theorem escalator_ride_time (scenario : EscalatorScenario) :
  scenario.escalatorDistance / scenario.escalatorSpeed = 84 :=
by sorry

end escalator_ride_time_l2576_257673


namespace find_b_find_a_range_l2576_257692

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - b * x

-- Define the derivative of f
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := a / x + (1 - a) * x - b

-- Theorem 1: Find the value of b
theorem find_b (a : ℝ) (h : a ≠ 1) :
  (∃ b : ℝ, f_deriv a b 1 = 0) → (∃ b : ℝ, b = 1) :=
sorry

-- Theorem 2: Find the range of values for a
theorem find_a_range (a : ℝ) (h : a ≠ 1) :
  (∃ x : ℝ, x ≥ 1 ∧ f a 1 x < a / (a - 1)) →
  (a ∈ Set.Ioo (- Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∪ Set.Ioi 1) :=
sorry

end

end find_b_find_a_range_l2576_257692


namespace constant_term_expansion_l2576_257600

theorem constant_term_expansion (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 10) :
  (∃ k : ℕ, ∃ r : ℕ, n = 3 * r ∧ n = 2 * k) ↔ n = 6 := by
  sorry

end constant_term_expansion_l2576_257600


namespace group_size_proof_l2576_257682

theorem group_size_proof (W : ℝ) (N : ℕ) : 
  ((W + 35) / N = W / N + 3.5) → N = 10 := by
  sorry

end group_size_proof_l2576_257682


namespace bankers_gain_l2576_257657

/-- Calculate the banker's gain given present worth, interest rate, and time period -/
theorem bankers_gain (present_worth : ℝ) (interest_rate : ℝ) (time_period : ℕ) : 
  present_worth = 600 → 
  interest_rate = 0.1 → 
  time_period = 2 → 
  present_worth * (1 + interest_rate) ^ time_period - present_worth = 126 := by
sorry

end bankers_gain_l2576_257657


namespace greatest_number_l2576_257672

theorem greatest_number : 
  8^85 > 5^100 ∧ 8^85 > 6^91 ∧ 8^85 > 7^90 := by
  sorry

end greatest_number_l2576_257672


namespace callum_max_score_l2576_257649

/-- Calculates the score for n consecutive wins with a base score and multiplier -/
def consecutiveWinScore (baseScore : ℕ) (n : ℕ) : ℕ :=
  baseScore * 2^(n - 1)

/-- Calculates the total score for a given number of wins -/
def totalScore (wins : ℕ) : ℕ :=
  (List.range wins).map (consecutiveWinScore 10) |> List.sum

theorem callum_max_score (totalMatches : ℕ) (krishnaWins : ℕ) 
    (h1 : totalMatches = 12)
    (h2 : krishnaWins = 2 * totalMatches / 3)
    (h3 : krishnaWins < totalMatches) : 
  totalScore (totalMatches - krishnaWins) = 150 := by
  sorry

end callum_max_score_l2576_257649


namespace symmetric_line_correct_l2576_257619

/-- Given a line with equation ax + by + c = 0, 
    returns the equation of the line symmetric to it with respect to the origin -/
def symmetric_line (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, -c)

theorem symmetric_line_correct (a b c : ℝ) :
  let (a', b', c') := symmetric_line a b c
  ∀ x y : ℝ, (a * x + b * y + c = 0) ↔ (a' * (-x) + b' * (-y) + c' = 0) :=
sorry

end symmetric_line_correct_l2576_257619


namespace geometric_sequence_sum_l2576_257652

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 4 + a 5 + a 6 = 168 := by
sorry

end geometric_sequence_sum_l2576_257652


namespace f_of_g_5_l2576_257695

def g (x : ℝ) : ℝ := 4 * x + 9

def f (x : ℝ) : ℝ := 6 * x - 11

theorem f_of_g_5 : f (g 5) = 163 := by
  sorry

end f_of_g_5_l2576_257695


namespace ways_to_walk_teaching_building_l2576_257615

/-- Represents a building with a given number of floors and staircases per floor -/
structure Building where
  floors : Nat
  staircases_per_floor : Nat

/-- Calculates the number of ways to walk from the first floor to the top floor -/
def ways_to_walk (b : Building) : Nat :=
  b.staircases_per_floor ^ (b.floors - 1)

/-- The teaching building -/
def teaching_building : Building :=
  { floors := 4, staircases_per_floor := 2 }

theorem ways_to_walk_teaching_building :
  ways_to_walk teaching_building = 2^3 := by
  sorry

#eval ways_to_walk teaching_building

end ways_to_walk_teaching_building_l2576_257615


namespace unique_intersection_bounded_difference_l2576_257618

-- Define the set U of functions satisfying the conditions
def U : Set (ℝ → ℝ) :=
  {f | ∃ x, f x = 2 * x ∧ ∀ x, 0 < (deriv f) x ∧ (deriv f) x < 2}

-- Statement 1: For any f in U, f(x) = 2x has exactly one solution
theorem unique_intersection (f : ℝ → ℝ) (hf : f ∈ U) :
  ∃! x, f x = 2 * x :=
sorry

-- Statement 2: For any h in U and x₁, x₂ close to 2023, |h(x₁) - h(x₂)| < 4
theorem bounded_difference (h : ℝ → ℝ) (hh : h ∈ U) :
  ∀ x₁ x₂, |x₁ - 2023| < 1 → |x₂ - 2023| < 1 → |h x₁ - h x₂| < 4 :=
sorry

end unique_intersection_bounded_difference_l2576_257618


namespace linear_function_through_zero_one_l2576_257609

/-- A linear function is a function of the form f(x) = kx + b where k and b are real numbers. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, ∀ x, f x = k * x + b

/-- Theorem: There exists a linear function that passes through the point (0,1). -/
theorem linear_function_through_zero_one : ∃ f : ℝ → ℝ, LinearFunction f ∧ f 0 = 1 := by
  sorry

end linear_function_through_zero_one_l2576_257609


namespace orchids_planted_today_calculation_l2576_257638

/-- The number of orchid bushes planted today in the park. -/
def orchids_planted_today (current : ℕ) (tomorrow : ℕ) (final : ℕ) : ℕ :=
  final - current - tomorrow

/-- Theorem stating the number of orchid bushes planted today. -/
theorem orchids_planted_today_calculation :
  orchids_planted_today 47 25 109 = 37 := by
  sorry

end orchids_planted_today_calculation_l2576_257638


namespace original_number_proof_l2576_257658

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
sorry

end original_number_proof_l2576_257658


namespace line_relationships_l2576_257680

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a simplified representation
  dummy : Unit

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop := sorry

theorem line_relationships (a b c : Line3D) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (¬ ∀ (a b c : Line3D), perpendicular a b → perpendicular a c → parallel b c) ∧ 
  (¬ ∀ (a b c : Line3D), perpendicular a b → perpendicular a c → perpendicular b c) ∧
  (∀ (a b c : Line3D), parallel a b → perpendicular b c → perpendicular a c) := by
  sorry

end line_relationships_l2576_257680


namespace expand_and_simplify_l2576_257623

theorem expand_and_simplify (a b : ℝ) : (a + b) * (a - 4 * b) = a^2 - 3*a*b - 4*b^2 := by
  sorry

end expand_and_simplify_l2576_257623


namespace negation_equivalence_l2576_257617

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 ≥ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end negation_equivalence_l2576_257617


namespace arithmetic_sequence_middle_average_l2576_257606

theorem arithmetic_sequence_middle_average (a : ℕ → ℕ) :
  (∀ i j, i < j → a i < a j) →  -- ascending order
  (∀ i, a (i + 1) - a i = a (i + 2) - a (i + 1)) →  -- arithmetic sequence
  (a 1 + a 2 + a 3) / 3 = 20 →  -- average of first three
  (a 5 + a 6 + a 7) / 3 = 24 →  -- average of last three
  (a 3 + a 4 + a 5) / 3 = 22 :=  -- average of middle three
by sorry

end arithmetic_sequence_middle_average_l2576_257606


namespace age_ratio_proof_l2576_257693

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 22 →  -- sum of ages is 22
  b = 8 →  -- b is 8 years old
  b = 2 * c  -- ratio of b's age to c's age is 2:1
:= by sorry

end age_ratio_proof_l2576_257693


namespace constant_term_binomial_expansion_l2576_257610

theorem constant_term_binomial_expansion :
  ∀ x : ℝ, ∃ t : ℕ → ℝ,
    (∀ r, t r = (-1)^r * (Nat.choose 6 r) * (2^((12 - 3*r) * x))) ∧
    (∃ k, t k = 15 ∧ ∀ r ≠ k, ∃ n : ℤ, t r = 2^(n*x)) :=
by sorry

end constant_term_binomial_expansion_l2576_257610


namespace intersection_of_M_and_N_l2576_257632

-- Define the sets M and N
def M : Set ℝ := {x | (x - 1) * (x - 4) = 0}
def N : Set ℝ := {x | (x + 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {1} := by
  sorry

end intersection_of_M_and_N_l2576_257632


namespace intersection_chord_length_l2576_257636

/-- Circle in 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line passing through origin in polar form -/
structure PolarLine where
  angle : ℝ

/-- Chord formed by intersection of circle and line -/
def chord_length (c : Circle) (l : PolarLine) : ℝ :=
  sorry

theorem intersection_chord_length :
  let c : Circle := { center := (0, -6), radius := 5 }
  let l : PolarLine := { angle := Real.arctan (Real.sqrt 5 / 2) }
  chord_length c l = 6 := by sorry

end intersection_chord_length_l2576_257636


namespace probability_at_least_one_boy_one_girl_l2576_257664

theorem probability_at_least_one_boy_one_girl (p : ℝ) : 
  p = 1/2 → (1 - 2 * p^4) = 7/8 := by sorry

end probability_at_least_one_boy_one_girl_l2576_257664


namespace cubic_sum_of_roots_l2576_257647

theorem cubic_sum_of_roots (a b r s : ℝ) : 
  (r^2 - a*r + b = 0) → (s^2 - a*s + b = 0) → (r^3 + s^3 = a^3 - 3*a*b) := by
  sorry

end cubic_sum_of_roots_l2576_257647


namespace contest_ranking_l2576_257644

theorem contest_ranking (A B C D : ℝ) 
  (eq1 : B + D = 2*(A + C) - 20)
  (ineq1 : A + 2*C < 2*B + D)
  (ineq2 : D > 2*(B + C)) :
  D > B ∧ B > A ∧ A > C :=
sorry

end contest_ranking_l2576_257644


namespace opposite_solutions_imply_a_equals_one_l2576_257605

theorem opposite_solutions_imply_a_equals_one (x y a : ℝ) :
  (x + 3 * y = 4 - a) →
  (x - y = -3 * a) →
  (x = -y) →
  a = 1 := by
sorry

end opposite_solutions_imply_a_equals_one_l2576_257605


namespace parabola_properties_l2576_257633

/-- Parabola passing through a specific point -/
structure Parabola where
  a : ℝ
  passes_through : a * (2 - 3)^2 - 1 = 1

/-- The number of units to move the parabola up for one x-axis intersection -/
def move_up_units (p : Parabola) : ℝ := 1

/-- Theorem stating the properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  p.a = 2 ∧ 
  (∃! x : ℝ, 2 * (x - 3)^2 - 1 + move_up_units p = 0) :=
sorry

end parabola_properties_l2576_257633


namespace difference_between_point_eight_and_half_l2576_257635

theorem difference_between_point_eight_and_half : (0.8 : ℝ) - (1/2 : ℝ) = 0.3 := by
  sorry

end difference_between_point_eight_and_half_l2576_257635


namespace inequality_proof_l2576_257694

theorem inequality_proof (x : ℝ) : 
  (4 * x^2 / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9) → 
  (-1/2 ≤ x ∧ x < 45/8 ∧ x ≠ 0) := by
sorry

end inequality_proof_l2576_257694


namespace divisibility_by_42p_l2576_257691

theorem divisibility_by_42p (p : Nat) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℤ, (3^p - 2^p - 1 : ℤ) = 42 * p * k := by
  sorry

end divisibility_by_42p_l2576_257691


namespace infinite_geometric_series_first_term_l2576_257667

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : a = 12 := by
sorry

end infinite_geometric_series_first_term_l2576_257667


namespace sum_of_squares_l2576_257643

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_seven_eq : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 2/7 := by
  sorry

end sum_of_squares_l2576_257643


namespace union_of_sets_l2576_257684

theorem union_of_sets : 
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | 0 < x ∧ x < 2}
  A ∪ B = {x : ℝ | -2 < x ∧ x < 2} := by
sorry

end union_of_sets_l2576_257684


namespace problems_per_worksheet_l2576_257611

/-- Given a set of worksheets with the following properties:
    - There are 9 worksheets in total
    - 5 worksheets have been graded
    - 16 problems remain to be graded
    This theorem proves that there are 4 problems on each worksheet. -/
theorem problems_per_worksheet (total_worksheets : Nat) (graded_worksheets : Nat) (remaining_problems : Nat)
    (h1 : total_worksheets = 9)
    (h2 : graded_worksheets = 5)
    (h3 : remaining_problems = 16) :
    (remaining_problems / (total_worksheets - graded_worksheets) : ℚ) = 4 := by
  sorry

end problems_per_worksheet_l2576_257611


namespace lottery_probability_l2576_257614

theorem lottery_probability (total_tickets winning_tickets people : ℕ) 
  (h1 : total_tickets = 10)
  (h2 : winning_tickets = 3)
  (h3 : people = 5) :
  let non_winning_tickets := total_tickets - winning_tickets
  1 - (Nat.choose non_winning_tickets people / Nat.choose total_tickets people : ℚ) = 11/12 :=
by sorry

end lottery_probability_l2576_257614


namespace small_bottle_price_l2576_257681

/-- The price of a small bottle given the following conditions:
  * 1375 large bottles were purchased at $1.75 each
  * 690 small bottles were purchased
  * The average price of all bottles is approximately $1.6163438256658595
-/
theorem small_bottle_price :
  let large_bottles : ℕ := 1375
  let small_bottles : ℕ := 690
  let large_price : ℝ := 1.75
  let avg_price : ℝ := 1.6163438256658595
  let total_bottles : ℕ := large_bottles + small_bottles
  let small_price : ℝ := (avg_price * total_bottles - large_price * large_bottles) / small_bottles
  ∃ ε > 0, |small_price - 1.34988436247191| < ε := by
  sorry

end small_bottle_price_l2576_257681


namespace least_four_digit_multiple_of_13_l2576_257669

theorem least_four_digit_multiple_of_13 : ∃ n : ℕ, 
  n % 13 = 0 ∧ 
  n ≥ 1000 ∧ 
  n < 10000 ∧ 
  (∀ m : ℕ, m % 13 = 0 ∧ m ≥ 1000 ∧ m < 10000 → n ≤ m) ∧
  n = 1001 :=
by sorry

end least_four_digit_multiple_of_13_l2576_257669


namespace red_blood_cell_diameter_scientific_notation_l2576_257639

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem red_blood_cell_diameter_scientific_notation :
  toScientificNotation 0.0000077 = ScientificNotation.mk 7.7 (-6) (by sorry) :=
sorry

end red_blood_cell_diameter_scientific_notation_l2576_257639


namespace line_hyperbola_intersection_l2576_257641

/-- Given a line y = ax + 1 and a hyperbola 3x^2 - y^2 = 1 that intersect at points A and B,
    if a circle with AB as its diameter passes through the origin,
    then a = 1 or a = -1 -/
theorem line_hyperbola_intersection (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = a * A.1 + 1 ∧ 3 * A.1^2 - A.2^2 = 1) ∧ 
    (B.2 = a * B.1 + 1 ∧ 3 * B.1^2 - B.2^2 = 1) ∧ 
    A ≠ B ∧
    (A.1 * B.1 + A.2 * B.2 = 0)) →
  a = 1 ∨ a = -1 := by
sorry

end line_hyperbola_intersection_l2576_257641


namespace quadratic_root_sum_l2576_257686

theorem quadratic_root_sum (m n : ℝ) : 
  (∀ x, m * x^2 - n * x - 2023 = 0 → x = -1 ∨ x ≠ -1) →
  (m * (-1)^2 - n * (-1) - 2023 = 0) →
  m + n = 2023 := by
sorry

end quadratic_root_sum_l2576_257686


namespace smallest_nonnegative_solution_congruence_l2576_257660

theorem smallest_nonnegative_solution_congruence :
  ∃ (x : ℕ), x < 15 ∧ (7 * x + 3) % 15 = 6 % 15 ∧
  ∀ (y : ℕ), y < x → (7 * y + 3) % 15 ≠ 6 % 15 :=
by
  -- The proof goes here
  sorry

end smallest_nonnegative_solution_congruence_l2576_257660


namespace money_ratio_problem_l2576_257688

/-- Proves that given the ratios between Ravi, Giri, and Kiran's money, and the fact that Ravi has $36, Kiran must have $105. -/
theorem money_ratio_problem (ravi giri kiran : ℕ) : 
  (ravi : ℚ) / giri = 6 / 7 → 
  (giri : ℚ) / kiran = 6 / 15 → 
  ravi = 36 → 
  kiran = 105 := by
sorry

end money_ratio_problem_l2576_257688


namespace race_average_time_l2576_257648

theorem race_average_time (fastest_time last_three_avg : ℝ) 
  (h1 : fastest_time = 15)
  (h2 : last_three_avg = 35) : 
  (fastest_time + 3 * last_three_avg) / 4 = 30 := by
  sorry

end race_average_time_l2576_257648


namespace smallest_square_from_smaller_squares_l2576_257656

theorem smallest_square_from_smaller_squares :
  ∀ n : ℕ,
  (∃ a : ℕ, a * a = n * (1 * 1 + 2 * 2 + 3 * 3)) →
  (∀ m : ℕ, m < n → ¬∃ b : ℕ, b * b = m * (1 * 1 + 2 * 2 + 3 * 3)) →
  n = 14 :=
by sorry

end smallest_square_from_smaller_squares_l2576_257656


namespace keith_stored_bales_l2576_257624

/-- The number of bales Keith stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Theorem: Keith stored 67 bales in the barn -/
theorem keith_stored_bales :
  bales_stored 22 89 = 67 := by
  sorry

end keith_stored_bales_l2576_257624


namespace age_problem_l2576_257613

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →  -- A is two years older than B
  b = 2 * c →  -- B is twice as old as C
  a + b + c = 47 →  -- Total age of A, B, and C is 47
  b = 18 :=  -- B's age is 18
by sorry

end age_problem_l2576_257613


namespace perfect_square_trinomial_condition_l2576_257622

/-- A polynomial p(x) = x^2 + mx + 9 is a perfect square trinomial if and only if
    there exists a real number a such that p(x) = (x + a)^2 for all x. -/
def IsPerfectSquareTrinomial (m : ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 9 = (x + a)^2

/-- If x^2 + mx + 9 is a perfect square trinomial, then m = 6 or m = -6. -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  IsPerfectSquareTrinomial m → m = 6 ∨ m = -6 := by
  sorry

end perfect_square_trinomial_condition_l2576_257622


namespace sodas_sold_in_afternoon_l2576_257645

theorem sodas_sold_in_afternoon (morning_sodas : ℕ) (total_sodas : ℕ) 
  (h1 : morning_sodas = 77) 
  (h2 : total_sodas = 96) : 
  total_sodas - morning_sodas = 19 := by
  sorry

end sodas_sold_in_afternoon_l2576_257645


namespace rectangular_paper_to_hexagon_l2576_257676

/-- A rectangular sheet of paper with sides a and b can be folded into a regular hexagon
    if and only if the aspect ratio b/a is between 1/2 and 2. -/
theorem rectangular_paper_to_hexagon (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x : ℝ), x > 0 ∧ x < a ∧ x < b ∧ (a - x)^2 + (b - x)^2 = x^2) ↔
  (1/2 < b/a ∧ b/a < 2) :=
sorry

end rectangular_paper_to_hexagon_l2576_257676


namespace athletes_on_second_floor_l2576_257654

/-- Proves that given a hotel with three floors housing 38 athletes, where 26 athletes are on the first and second floors, and 27 athletes are on the second and third floors, the number of athletes on the second floor is 15. -/
theorem athletes_on_second_floor 
  (total_athletes : ℕ) 
  (first_second : ℕ) 
  (second_third : ℕ) 
  (h1 : total_athletes = 38) 
  (h2 : first_second = 26) 
  (h3 : second_third = 27) : 
  ∃ (first second third : ℕ), 
    first + second + third = total_athletes ∧ 
    first + second = first_second ∧ 
    second + third = second_third ∧ 
    second = 15 :=
by sorry

end athletes_on_second_floor_l2576_257654


namespace cross_country_winning_scores_l2576_257616

/-- Represents a cross country race between two teams -/
structure CrossCountryRace where
  num_runners_per_team : ℕ
  total_runners : ℕ
  min_score : ℕ
  max_score : ℕ

/-- Calculates the number of different winning scores in a cross country race -/
def num_winning_scores (race : CrossCountryRace) : ℕ :=
  race.max_score - race.min_score + 1

/-- The specific cross country race described in the problem -/
def specific_race : CrossCountryRace :=
  { num_runners_per_team := 6
  , total_runners := 12
  , min_score := 21
  , max_score := 39 }

theorem cross_country_winning_scores :
  num_winning_scores specific_race = 19 := by
  sorry

end cross_country_winning_scores_l2576_257616


namespace alex_class_size_l2576_257603

/-- In a class, given a student who is both the 30th best and 30th worst, 
    the total number of students in the class is 59. -/
theorem alex_class_size (n : ℕ) 
  (h1 : ∃ (alex : ℕ), alex ≤ n ∧ alex = 30)  -- Alex is 30th best
  (h2 : ∃ (alex : ℕ), alex ≤ n ∧ alex = 30)  -- Alex is 30th worst
  : n = 59 := by
  sorry

end alex_class_size_l2576_257603


namespace illumination_theorem_l2576_257699

/-- Calculates the total number of nights a house can be illuminated given an initial number of candles. -/
def totalNights (initialCandles : ℕ) : ℕ :=
  let rec aux (candles stubs nights : ℕ) : ℕ :=
    if candles = 0 then
      nights + (stubs / 4)
    else
      aux (candles - 1) (stubs + 1) (nights + 1)
  aux initialCandles 0 0

/-- Theorem stating that 43 initial candles result in 57 nights of illumination. -/
theorem illumination_theorem :
  totalNights 43 = 57 := by
  sorry

end illumination_theorem_l2576_257699


namespace regular_polygon_sides_l2576_257601

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end regular_polygon_sides_l2576_257601


namespace cristina_speed_l2576_257626

/-- Cristina's running speed in a race with Nicky -/
theorem cristina_speed (head_start : ℝ) (nicky_speed : ℝ) (catch_up_time : ℝ) 
  (h1 : head_start = 36)
  (h2 : nicky_speed = 3)
  (h3 : catch_up_time = 12) :
  (head_start + nicky_speed * catch_up_time) / catch_up_time = 6 :=
by sorry

end cristina_speed_l2576_257626


namespace chord_intersection_ratio_l2576_257662

theorem chord_intersection_ratio (EQ FQ GQ HQ : ℝ) :
  EQ = 5 →
  GQ = 12 →
  HQ = 3 →
  EQ * FQ = GQ * HQ →
  FQ / HQ = 12 / 5 := by
sorry

end chord_intersection_ratio_l2576_257662


namespace rectangular_box_volume_l2576_257646

theorem rectangular_box_volume (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (∃ (k : ℕ), k > 0 ∧ a = 2 * k ∧ b = 3 * k ∧ c = 5 * k) →
  a * b * c = 240 :=
by sorry

end rectangular_box_volume_l2576_257646


namespace absolute_value_equation_solution_set_l2576_257677

theorem absolute_value_equation_solution_set :
  {x : ℝ | |x / (x - 1)| = x / (x - 1)} = {x : ℝ | x ≤ 0 ∨ x > 1} := by
  sorry

end absolute_value_equation_solution_set_l2576_257677


namespace sqrt_sum_condition_l2576_257690

/-- For distinct positive numbers a, b, c that are not perfect squares,
    √a + √b = √c holds if and only if 2√(ab) = c - (a + b) and ab is a perfect square -/
theorem sqrt_sum_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (hna : ¬ ∃ (n : ℕ), a = n^2)
  (hnb : ¬ ∃ (n : ℕ), b = n^2)
  (hnc : ¬ ∃ (n : ℕ), c = n^2) :
  (Real.sqrt a + Real.sqrt b = Real.sqrt c) ↔ 
  (2 * Real.sqrt (a * b) = c - (a + b) ∧ ∃ (n : ℕ), a * b = n^2) :=
by sorry

end sqrt_sum_condition_l2576_257690


namespace total_boxes_moved_l2576_257675

/-- The number of boxes a truck can hold -/
def boxes_per_truck : ℕ := 4

/-- The number of trips taken to move all boxes -/
def num_trips : ℕ := 218

/-- The total number of boxes moved -/
def total_boxes : ℕ := boxes_per_truck * num_trips

theorem total_boxes_moved :
  total_boxes = 872 := by sorry

end total_boxes_moved_l2576_257675


namespace equation_solution_l2576_257668

theorem equation_solution : ∃ x : ℝ, 35 - (5 + 3) = 7 + x ∧ x = 20 := by
  sorry

end equation_solution_l2576_257668


namespace quadratic_one_solution_sum_l2576_257625

theorem quadratic_one_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x, 3 * x^2 + b₁ * x + 5 * x + 7 = 0 → (∀ y, 3 * y^2 + b₁ * y + 5 * y + 7 = 0 → x = y)) ∧
  (∀ x, 3 * x^2 + b₂ * x + 5 * x + 7 = 0 → (∀ y, 3 * y^2 + b₂ * y + 5 * y + 7 = 0 → x = y)) ∧
  (∀ b, (∀ x, 3 * x^2 + b * x + 5 * x + 7 = 0 → (∀ y, 3 * y^2 + b * y + 5 * y + 7 = 0 → x = y)) → b = b₁ ∨ b = b₂) →
  b₁ + b₂ = -10 :=
sorry

end quadratic_one_solution_sum_l2576_257625


namespace fraction_product_l2576_257661

theorem fraction_product : (2 : ℚ) / 5 * (3 : ℚ) / 5 = (6 : ℚ) / 25 := by
  sorry

end fraction_product_l2576_257661


namespace midpoint_quadrilateral_area_l2576_257665

/-- Given a rectangle with length 15 and width 8, the quadrilateral formed by
    connecting the midpoints of its sides has an area of 30 square units. -/
theorem midpoint_quadrilateral_area (l w : ℝ) (hl : l = 15) (hw : w = 8) :
  let midpoint_quad_area := (l / 2) * (w / 2)
  midpoint_quad_area = 30 := by sorry

end midpoint_quadrilateral_area_l2576_257665


namespace jony_turnaround_block_l2576_257698

/-- Represents the walking scenario of Jony along Sunrise Boulevard -/
structure WalkingScenario where
  start_block : ℕ
  end_block : ℕ
  block_length : ℕ
  walking_speed : ℕ
  walking_time : ℕ

/-- Calculates the block where Jony turns around -/
def turnaround_block (scenario : WalkingScenario) : ℕ :=
  let total_distance := scenario.walking_speed * scenario.walking_time
  let start_to_end_distance := (scenario.end_block - scenario.start_block) * scenario.block_length
  let extra_distance := total_distance - start_to_end_distance
  let extra_blocks := extra_distance / scenario.block_length
  scenario.end_block + extra_blocks

/-- Theorem stating that Jony turns around at block 110 -/
theorem jony_turnaround_block :
  let scenario : WalkingScenario := {
    start_block := 10,
    end_block := 70,
    block_length := 40,
    walking_speed := 100,
    walking_time := 40
  }
  turnaround_block scenario = 110 := by
  sorry

end jony_turnaround_block_l2576_257698


namespace size_relationship_l2576_257678

theorem size_relationship (a b c : ℝ) 
  (ha : a = 4^(1/2 : ℝ)) 
  (hb : b = 2^(1/3 : ℝ)) 
  (hc : c = 5^(1/2 : ℝ)) : 
  b < a ∧ a < c :=
by sorry

end size_relationship_l2576_257678


namespace peters_remaining_money_l2576_257650

/-- Peter's shopping trip to the market -/
theorem peters_remaining_money 
  (initial_amount : ℕ) 
  (potato_price potato_quantity : ℕ)
  (tomato_price tomato_quantity : ℕ)
  (cucumber_price cucumber_quantity : ℕ)
  (banana_price banana_quantity : ℕ)
  (h1 : initial_amount = 500)
  (h2 : potato_price = 2 ∧ potato_quantity = 6)
  (h3 : tomato_price = 3 ∧ tomato_quantity = 9)
  (h4 : cucumber_price = 4 ∧ cucumber_quantity = 5)
  (h5 : banana_price = 5 ∧ banana_quantity = 3) :
  initial_amount - 
  (potato_price * potato_quantity + 
   tomato_price * tomato_quantity + 
   cucumber_price * cucumber_quantity + 
   banana_price * banana_quantity) = 426 := by
  sorry

end peters_remaining_money_l2576_257650


namespace david_airport_distance_l2576_257651

/-- The distance from David's home to the airport --/
def airport_distance : ℝ := by sorry

/-- David's initial speed --/
def initial_speed : ℝ := 35

/-- David's speed increase --/
def speed_increase : ℝ := 15

/-- Time saved by increasing speed --/
def time_saved : ℝ := 1.5

/-- Time early --/
def time_early : ℝ := 0.5

theorem david_airport_distance :
  airport_distance = initial_speed * (airport_distance / initial_speed) +
  (initial_speed + speed_increase) * (time_saved - time_early) ∧
  airport_distance = 210 := by sorry

end david_airport_distance_l2576_257651


namespace box_volume_l2576_257621

theorem box_volume (side1 side2 upper : ℝ) 
  (h1 : side1 = 120)
  (h2 : side2 = 72)
  (h3 : upper = 60) :
  ∃ (l w h : ℝ), 
    l * w = side1 ∧ 
    w * h = side2 ∧ 
    l * h = upper ∧ 
    l * w * h = 720 :=
sorry

end box_volume_l2576_257621


namespace min_product_abc_l2576_257697

theorem min_product_abc (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b + c = 3)
  (h5 : a ≤ 3 * b ∧ a ≤ 3 * c)
  (h6 : b ≤ 3 * a ∧ b ≤ 3 * c)
  (h7 : c ≤ 3 * a ∧ c ≤ 3 * b) :
  81 / 125 ≤ a * b * c :=
by sorry

end min_product_abc_l2576_257697


namespace betty_bead_ratio_l2576_257671

/-- Given that Betty has 30 red beads and 20 blue beads, prove that the ratio of red beads to blue beads is 3:2 -/
theorem betty_bead_ratio :
  let red_beads : ℕ := 30
  let blue_beads : ℕ := 20
  (red_beads : ℚ) / blue_beads = 3 / 2 := by sorry

end betty_bead_ratio_l2576_257671


namespace second_term_is_three_l2576_257653

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

/-- The second term of the sequence is 3 -/
theorem second_term_is_three (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 2) (a 5) →
  a 2 = 3 := by sorry

end second_term_is_three_l2576_257653


namespace fourth_power_one_fourth_equals_decimal_l2576_257642

theorem fourth_power_one_fourth_equals_decimal : (1 / 4 : ℚ) ^ 4 = 390625 / 100000000 := by
  sorry

end fourth_power_one_fourth_equals_decimal_l2576_257642


namespace intersection_of_S_and_T_l2576_257696

def S : Set ℝ := {x | x^2 + 2*x = 0}
def T : Set ℝ := {x | x^2 - 2*x = 0}

theorem intersection_of_S_and_T : S ∩ T = {0} := by sorry

end intersection_of_S_and_T_l2576_257696


namespace cubic_root_sum_l2576_257640

-- Define the cubic polynomial
def cubic_poly (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem cubic_root_sum (a b c d : ℝ) : 
  a ≠ 0 → 
  cubic_poly a b c d 4 = 0 →
  cubic_poly a b c d (-3) = 0 →
  (b + c) / a = -13 := by
sorry

end cubic_root_sum_l2576_257640


namespace abcd_multiplication_l2576_257687

theorem abcd_multiplication (A B C D : ℕ) : 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) →
  (1000 * A + 100 * B + 10 * C + D) * 9 = 1000 * D + 100 * C + 10 * B + A →
  A = 1 ∧ B = 0 ∧ C = 8 ∧ D = 9 := by
sorry

end abcd_multiplication_l2576_257687


namespace percentage_problem_l2576_257670

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.1 * x = 40 := by
  sorry

end percentage_problem_l2576_257670


namespace drawBalls_18_4_l2576_257607

/-- The number of balls in the bin -/
def n : ℕ := 18

/-- The number of balls to be drawn -/
def k : ℕ := 4

/-- The number of ways to draw k balls from n balls, 
    where the first ball is returned and the rest are not -/
def drawBalls (n k : ℕ) : ℕ := n * n * (n - 1) * (n - 2)

/-- Theorem stating that drawing 4 balls from 18 balls, 
    where the first ball is returned and the rest are not, 
    can be done in 87984 ways -/
theorem drawBalls_18_4 : drawBalls n k = 87984 := by sorry

end drawBalls_18_4_l2576_257607


namespace ratio_problem_l2576_257631

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 2) (h2 : c/b = 3) : (a+b)/(b+c) = 3/8 := by
  sorry

end ratio_problem_l2576_257631


namespace find_other_number_l2576_257659

/-- Given two positive integers with known LCM, HCF, and one of the numbers, prove the value of the other number -/
theorem find_other_number (a b : ℕ+) (h1 : Nat.lcm a b = 76176) (h2 : Nat.gcd a b = 116) (h3 : a = 8128) : b = 1087 := by
  sorry

end find_other_number_l2576_257659


namespace sandy_shopping_l2576_257629

def shopping_equation (X Y : ℝ) : Prop :=
  let pie_cost : ℝ := 6
  let sandwich_cost : ℝ := 3
  let book_cost : ℝ := 10
  let book_discount : ℝ := 0.2
  let sales_tax : ℝ := 0.05
  let discounted_book_cost : ℝ := book_cost * (1 - book_discount)
  let subtotal : ℝ := pie_cost + sandwich_cost + discounted_book_cost
  let total_cost : ℝ := subtotal * (1 + sales_tax)
  Y = X - total_cost

theorem sandy_shopping : 
  ∀ X Y : ℝ, shopping_equation X Y ↔ Y = X - 17.85 := by sorry

end sandy_shopping_l2576_257629


namespace segment_endpoint_l2576_257627

/-- Given a line segment from (1, 3) to (x, 7) with length 15 and x < 0, prove x = 1 - √209 -/
theorem segment_endpoint (x : ℝ) : 
  x < 0 → 
  Real.sqrt ((1 - x)^2 + (3 - 7)^2) = 15 → 
  x = 1 - Real.sqrt 209 := by
sorry

end segment_endpoint_l2576_257627


namespace cube_equation_solution_l2576_257628

theorem cube_equation_solution (a d : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * d) : d = 49 := by
  sorry

end cube_equation_solution_l2576_257628


namespace correct_average_l2576_257604

theorem correct_average (n : ℕ) (initial_avg incorrect_num correct_num : ℚ) : 
  n = 10 → 
  initial_avg = 16 → 
  incorrect_num = 26 → 
  correct_num = 46 → 
  (n * initial_avg - incorrect_num + correct_num) / n = 18 := by
  sorry

end correct_average_l2576_257604


namespace midpoint_translated_triangle_l2576_257612

/-- Given triangle BIG with vertices B(0, 0), I(3, 3), and G(6, 0),
    translated 3 units left and 4 units up to form triangle B'I'G',
    the midpoint of segment B'G' is (0, 4). -/
theorem midpoint_translated_triangle (B I G B' I' G' : ℝ × ℝ) :
  B = (0, 0) →
  I = (3, 3) →
  G = (6, 0) →
  B' = (B.1 - 3, B.2 + 4) →
  I' = (I.1 - 3, I.2 + 4) →
  G' = (G.1 - 3, G.2 + 4) →
  ((B'.1 + G'.1) / 2, (B'.2 + G'.2) / 2) = (0, 4) := by
sorry

end midpoint_translated_triangle_l2576_257612


namespace triangle_areas_product_l2576_257655

theorem triangle_areas_product (h₁ h₂ h₃ : ℝ) 
  (h1 : h₁ = 1)
  (h2 : h₂ = 1 + Real.sqrt 3 / 2)
  (h3 : h₃ = 1 - Real.sqrt 3 / 2) :
  (1/2 * 1 * h₁) * (1/2 * 1 * h₂) * (1/2 * 1 * h₃) = 1/32 := by
  sorry

#check triangle_areas_product

end triangle_areas_product_l2576_257655


namespace gcd_1729_1314_l2576_257634

theorem gcd_1729_1314 : Nat.gcd 1729 1314 = 1 := by
  sorry

end gcd_1729_1314_l2576_257634


namespace sample_capacity_proof_l2576_257685

/-- Given a sample divided into groups, prove that if one group has a frequency of 30
    and a frequency rate of 0.25, then the sample capacity is 120. -/
theorem sample_capacity_proof (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ) 
    (h1 : frequency = 30)
    (h2 : frequency_rate = 1/4)
    (h3 : frequency_rate = frequency / n) : n = 120 := by
  sorry

end sample_capacity_proof_l2576_257685


namespace power_division_rule_l2576_257674

theorem power_division_rule (n : ℕ) : 19^11 / 19^6 = 247609 := by
  sorry

end power_division_rule_l2576_257674


namespace bacon_to_eggs_ratio_l2576_257620

/-- Represents a breakfast plate with eggs and bacon strips -/
structure BreakfastPlate where
  eggs : ℕ
  bacon : ℕ

/-- Represents the cafe's breakfast order -/
structure CafeOrder where
  plates : ℕ
  totalBacon : ℕ

theorem bacon_to_eggs_ratio (order : CafeOrder) (plate : BreakfastPlate) :
  order.plates = 14 →
  order.totalBacon = 56 →
  plate.eggs = 2 →
  (order.totalBacon / order.plates : ℚ) / plate.eggs = 2 / 1 := by
  sorry

end bacon_to_eggs_ratio_l2576_257620


namespace seven_point_four_five_repeating_equals_82_11_l2576_257666

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def repeatingDecimalToRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + (x.repeatingPart : ℚ) / (99 : ℚ)

/-- The repeating decimal 7.45̄ -/
def seven_point_four_five_repeating : RepeatingDecimal :=
  { integerPart := 7, repeatingPart := 45 }

theorem seven_point_four_five_repeating_equals_82_11 :
  repeatingDecimalToRational seven_point_four_five_repeating = 82 / 11 := by
  sorry

end seven_point_four_five_repeating_equals_82_11_l2576_257666


namespace andy_wrong_questions_l2576_257630

theorem andy_wrong_questions (a b c d : ℕ) : 
  a + b = c + d →
  a + d = b + c + 6 →
  c = 6 →
  a = 12 :=
by
  sorry

end andy_wrong_questions_l2576_257630


namespace sixth_finger_is_one_l2576_257663

def f : ℕ → ℕ
| 2 => 1
| 1 => 8
| 8 => 7
| 7 => 2
| _ => 0  -- Default case for other inputs

def finger_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 2  -- Start with 2 on the first finger (index 0)
  | n + 1 => f (finger_sequence n)

theorem sixth_finger_is_one : finger_sequence 5 = 1 := by
  sorry

end sixth_finger_is_one_l2576_257663


namespace triangle_area_range_l2576_257637

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -Real.log x
  else if x > 1 then Real.log x
  else 0  -- undefined for x ≤ 0 and x = 1

-- Define the derivative of f(x)
noncomputable def f_deriv (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -1/x
  else if x > 1 then 1/x
  else 0  -- undefined for x ≤ 0 and x = 1

-- Theorem statement
theorem triangle_area_range (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁ ∧ x₁ < 1) (h₂ : x₂ > 1) 
  (h_perp : f_deriv x₁ * f_deriv x₂ = -1) :
  let y₁ := f x₁
  let y₂ := f x₂
  let m₁ := f_deriv x₁
  let m₂ := f_deriv x₂
  let x_int := (y₂ - y₁ + m₁*x₁ - m₂*x₂) / (m₁ - m₂)
  let area := abs ((1 - Real.log x₁ - (-1 + Real.log x₂)) * x_int / 2)
  0 < area ∧ area < 1 :=
sorry

end triangle_area_range_l2576_257637


namespace max_rectangles_in_square_l2576_257602

/-- Given a square with side length 14 cm and rectangles of width 2 cm and length 8 cm,
    the maximum number of whole rectangles that can fit within the square is 12. -/
theorem max_rectangles_in_square : ∀ (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ),
  square_side = 14 →
  rect_width = 2 →
  rect_length = 8 →
  ⌊(square_side * square_side) / (rect_width * rect_length)⌋ = 12 := by
  sorry

end max_rectangles_in_square_l2576_257602
