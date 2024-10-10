import Mathlib

namespace maintenance_check_increase_l4127_412762

theorem maintenance_check_increase (original : ℝ) (new : ℝ) 
  (h1 : original = 30)
  (h2 : new = 60) :
  (new - original) / original * 100 = 100 := by
  sorry

end maintenance_check_increase_l4127_412762


namespace divisibility_property_l4127_412767

theorem divisibility_property (p : ℕ) (h1 : p > 1) (h2 : Odd p) :
  ∃ k : ℤ, (p - 1 : ℤ) ^ ((p - 1) / 2) - 1 = (p - 2) * k := by
  sorry

end divisibility_property_l4127_412767


namespace not_equivalent_polar_points_l4127_412774

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Check if two polar points are equivalent -/
def equivalentPolarPoints (p1 p2 : PolarPoint) : Prop :=
  p1.r = p2.r ∧ ∃ k : ℤ, p1.θ = p2.θ + 2 * k * Real.pi

theorem not_equivalent_polar_points :
  ¬ equivalentPolarPoints ⟨2, 11 * Real.pi / 6⟩ ⟨2, Real.pi / 6⟩ := by
  sorry

end not_equivalent_polar_points_l4127_412774


namespace min_value_theorem_l4127_412783

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 / y = 3) :
  ∀ z, z = 2 / x + y → z ≥ 8 / 3 ∧ ∃ w, w = 2 / x + y ∧ w = 8 / 3 :=
sorry

end min_value_theorem_l4127_412783


namespace ping_pong_probabilities_l4127_412782

/-- Represents the probability of player A winning a point -/
def prob_A_wins (serving : Bool) : ℝ :=
  if serving then 0.5 else 0.4

/-- Represents the probability of player A winning the k-th point after a 10:10 tie -/
def prob_A_k (k : ℕ) : ℝ :=
  prob_A_wins (k % 2 = 1)

/-- The probability of the game ending in exactly 2 points after a 10:10 tie -/
def prob_X_2 : ℝ :=
  prob_A_k 1 * prob_A_k 2 + (1 - prob_A_k 1) * (1 - prob_A_k 2)

/-- The probability of the game ending in exactly 4 points after a 10:10 tie with A winning -/
def prob_X_4_A_wins : ℝ :=
  (1 - prob_A_k 1) * prob_A_k 2 * prob_A_k 3 * prob_A_k 4 +
  prob_A_k 1 * (1 - prob_A_k 2) * prob_A_k 3 * prob_A_k 4

theorem ping_pong_probabilities :
  (prob_X_2 = prob_A_k 1 * prob_A_k 2 + (1 - prob_A_k 1) * (1 - prob_A_k 2)) ∧
  (prob_X_4_A_wins = (1 - prob_A_k 1) * prob_A_k 2 * prob_A_k 3 * prob_A_k 4 +
                     prob_A_k 1 * (1 - prob_A_k 2) * prob_A_k 3 * prob_A_k 4) := by
  sorry

end ping_pong_probabilities_l4127_412782


namespace trig_simplification_l4127_412763

theorem trig_simplification (α : ℝ) :
  Real.sin (α - 4 * Real.pi) * Real.sin (Real.pi - α) -
  2 * (Real.cos ((3 * Real.pi) / 2 + α))^2 -
  Real.sin (α + Real.pi) * Real.cos (Real.pi / 2 + α) =
  -2 * (Real.sin α)^2 := by sorry

end trig_simplification_l4127_412763


namespace quadratic_inequality_range_l4127_412742

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → a ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end quadratic_inequality_range_l4127_412742


namespace angle_B_measure_l4127_412740

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.b * Real.sin t.B - t.c * Real.sin t.C = t.a ∧
  (t.b^2 + t.c^2 - t.a^2) / 4 = 1/2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem angle_B_measure (t : Triangle) :
  satisfies_conditions t → t.B = 77.5 * π / 180 :=
by
  sorry

end angle_B_measure_l4127_412740


namespace forty_seventh_digit_is_six_l4127_412750

def sequence_digit (n : ℕ) : ℕ :=
  let start := 90
  let digit_pos := n - 1
  let num_index := digit_pos / 2
  let in_num_pos := digit_pos % 2
  let current_num := start - num_index
  (current_num / 10^(1 - in_num_pos)) % 10

theorem forty_seventh_digit_is_six :
  sequence_digit 47 = 6 := by
  sorry

end forty_seventh_digit_is_six_l4127_412750


namespace power_of_two_difference_l4127_412781

theorem power_of_two_difference (n : ℕ) (h : n > 0) : 2^n - 2^(n-1) = 2^(n-1) := by
  sorry

end power_of_two_difference_l4127_412781


namespace supplementary_angles_theorem_l4127_412717

theorem supplementary_angles_theorem (A B : ℝ) : 
  A + B = 180 →  -- angles A and B are supplementary
  A = 4 * B →    -- measure of angle A is 4 times angle B
  A = 144 :=     -- measure of angle A is 144 degrees
by sorry

end supplementary_angles_theorem_l4127_412717


namespace boy_speed_around_square_l4127_412784

/-- The speed of a boy running around a square field -/
theorem boy_speed_around_square (side_length : ℝ) (time : ℝ) : 
  side_length = 20 → time = 24 → 
  (4 * side_length) / time * (3600 / 1000) = 12 := by
  sorry

end boy_speed_around_square_l4127_412784


namespace erwin_chocolate_consumption_l4127_412777

/-- Represents Erwin's chocolate consumption pattern and total chocolates eaten --/
structure ChocolateConsumption where
  weekday_consumption : ℕ  -- chocolates eaten per weekday
  weekend_consumption : ℕ  -- chocolates eaten per weekend day
  total_chocolates : ℕ     -- total chocolates eaten

/-- Calculates the number of weeks it took to eat all chocolates --/
def weeks_to_finish (consumption : ChocolateConsumption) : ℚ :=
  consumption.total_chocolates / (5 * consumption.weekday_consumption + 2 * consumption.weekend_consumption)

/-- Theorem stating it took Erwin 2 weeks to finish the chocolates --/
theorem erwin_chocolate_consumption :
  let consumption : ChocolateConsumption := {
    weekday_consumption := 2,
    weekend_consumption := 1,
    total_chocolates := 24
  }
  weeks_to_finish consumption = 2 := by sorry

end erwin_chocolate_consumption_l4127_412777


namespace constant_d_value_l4127_412735

-- Define the problem statement
theorem constant_d_value (a d : ℝ) :
  (∀ x : ℝ, (x + 3) * (x + a) = x^2 + d*x + 12) →
  d = 7 :=
by sorry

end constant_d_value_l4127_412735


namespace difference_p_q_l4127_412732

theorem difference_p_q (p q : ℚ) (hp : 3 / p = 8) (hq : 3 / q = 18) : p - q = 5 / 24 := by
  sorry

end difference_p_q_l4127_412732


namespace carlos_july_reading_l4127_412754

/-- The number of books Carlos read in June -/
def june_books : ℕ := 42

/-- The number of books Carlos read in August -/
def august_books : ℕ := 30

/-- The total number of books Carlos needed to read -/
def total_books : ℕ := 100

/-- The number of books Carlos read in July -/
def july_books : ℕ := total_books - (june_books + august_books)

theorem carlos_july_reading :
  july_books = 28 := by sorry

end carlos_july_reading_l4127_412754


namespace digit_1234_is_4_l4127_412796

/-- The number of digits in the representation of an integer -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The nth digit in the sequence of concatenated integers from 1 to 500 -/
def nthDigit (n : ℕ) : ℕ := sorry

theorem digit_1234_is_4 :
  nthDigit 1234 = 4 := by sorry

end digit_1234_is_4_l4127_412796


namespace platform_length_l4127_412797

/-- The length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : 
  train_length = 120 →
  train_speed_kmph = 72 →
  crossing_time = 25 →
  (train_speed_kmph * 1000 / 3600) * crossing_time - train_length = 380 := by
  sorry


end platform_length_l4127_412797


namespace right_triangle_hypotenuse_l4127_412778

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by sorry

end right_triangle_hypotenuse_l4127_412778


namespace subsets_with_three_adjacent_12_chairs_l4127_412711

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets of n chairs arranged in a circle
    that contain at least three adjacent chairs -/
def subsets_with_three_adjacent (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of subsets of 12 chairs arranged in a circle
    that contain at least three adjacent chairs is 2056 -/
theorem subsets_with_three_adjacent_12_chairs :
  subsets_with_three_adjacent n = 2056 := by sorry

end subsets_with_three_adjacent_12_chairs_l4127_412711


namespace car_ownership_l4127_412751

theorem car_ownership (total : ℕ) (neither : ℕ) (both : ℕ) (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 20)
  (h4 : bike_only = 35) :
  total - neither - bike_only = 44 :=
by sorry

end car_ownership_l4127_412751


namespace james_total_score_l4127_412752

/-- Calculates the total points scored by James in a basketball game -/
def total_points (field_goals three_pointers two_pointers free_throws : ℕ) : ℕ :=
  field_goals * 3 + three_pointers * 2 + two_pointers * 2 + free_throws * 1

theorem james_total_score :
  total_points 13 0 20 5 = 84 := by
  sorry

#eval total_points 13 0 20 5

end james_total_score_l4127_412752


namespace triangle_side_lengths_l4127_412724

/-- Given a triangle with area t, angle α, and angle β, prove that the sides a, b, and c have the specified lengths. -/
theorem triangle_side_lengths 
  (t : ℝ) 
  (α β : Real) 
  (h_t : t = 4920)
  (h_α : α = 43 + 36 / 60 + 10 / 3600)
  (h_β : β = 72 + 23 / 60 + 11 / 3600) :
  ∃ (a b c : ℝ), 
    (abs (a - 89) < 1) ∧ 
    (abs (b - 123) < 1) ∧ 
    (abs (c - 116) < 1) ∧
    (a > 0) ∧ (b > 0) ∧ (c > 0) :=
by
  sorry


end triangle_side_lengths_l4127_412724


namespace paths_through_F_and_H_l4127_412719

/-- Represents a point on the grid -/
structure Point where
  x : Nat
  y : Nat

/-- Calculates the number of paths between two points on a grid -/
def numPaths (start finish : Point) : Nat :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The grid dimensions -/
def gridWidth : Nat := 7
def gridHeight : Nat := 6

/-- The points on the grid -/
def E : Point := ⟨0, 5⟩
def F : Point := ⟨4, 4⟩
def H : Point := ⟨5, 2⟩
def G : Point := ⟨6, 0⟩

/-- Theorem: The number of 12-step paths from E to G passing through F and then H is 135 -/
theorem paths_through_F_and_H : 
  numPaths E F * numPaths F H * numPaths H G = 135 := by
  sorry

end paths_through_F_and_H_l4127_412719


namespace probability_of_colored_ball_l4127_412747

def urn_total : ℕ := 30
def red_balls : ℕ := 10
def blue_balls : ℕ := 5
def white_balls : ℕ := 15

theorem probability_of_colored_ball :
  (red_balls + blue_balls : ℚ) / urn_total = 1 / 2 :=
sorry

end probability_of_colored_ball_l4127_412747


namespace alice_added_nineteen_plates_l4127_412756

/-- The number of plates Alice added before the tower fell -/
def additional_plates (initial : ℕ) (second_addition : ℕ) (total : ℕ) : ℕ :=
  total - (initial + second_addition)

/-- Theorem stating that Alice added 19 more plates before the tower fell -/
theorem alice_added_nineteen_plates : 
  additional_plates 27 37 83 = 19 := by
  sorry

end alice_added_nineteen_plates_l4127_412756


namespace min_value_of_function_min_value_achieved_l4127_412729

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  x + 1 / (x - 2) ≥ 4 := by
  sorry

theorem min_value_achieved (x : ℝ) (h : x > 2) :
  ∃ x₀ > 2, x₀ + 1 / (x₀ - 2) = 4 := by
  sorry

end min_value_of_function_min_value_achieved_l4127_412729


namespace election_votes_theorem_l4127_412712

theorem election_votes_theorem (candidate1_percent : ℝ) (candidate2_percent : ℝ) 
  (candidate3_percent : ℝ) (candidate4_percent : ℝ) (candidate4_votes : ℕ) :
  candidate1_percent = 42 →
  candidate2_percent = 30 →
  candidate3_percent = 20 →
  candidate4_percent = 8 →
  candidate4_votes = 720 →
  (candidate1_percent + candidate2_percent + candidate3_percent + candidate4_percent = 100) →
  ∃ (total_votes : ℕ), total_votes = 9000 ∧ 
    (candidate4_percent / 100 * total_votes : ℝ) = candidate4_votes :=
by sorry

end election_votes_theorem_l4127_412712


namespace triangle_max_area_l4127_412736

/-- Given two positive real numbers a and b representing the lengths of two sides of a triangle,
    the area of the triangle is maximized when these sides are perpendicular. -/
theorem triangle_max_area (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ θ : ℝ, 0 < θ ∧ θ < π → (1/2) * a * b * Real.sin θ ≤ (1/2) * a * b := by
  sorry

end triangle_max_area_l4127_412736


namespace taxi_driver_theorem_l4127_412798

def driving_distances : List Int := [-5, 3, 6, -4, 7, -2]

def base_fare : Nat := 8
def extra_fare_per_km : Nat := 2
def base_distance : Nat := 3

def cumulative_distance (n : Nat) : Int :=
  (driving_distances.take n).sum

def trip_fare (distance : Int) : Nat :=
  base_fare + max 0 (distance.natAbs - base_distance) * extra_fare_per_km

def total_earnings : Nat :=
  (driving_distances.map trip_fare).sum

theorem taxi_driver_theorem :
  (cumulative_distance 4 = 0) ∧
  (cumulative_distance driving_distances.length = 5) ∧
  (total_earnings = 68) := by
  sorry

end taxi_driver_theorem_l4127_412798


namespace train_length_l4127_412772

/-- The length of a train given its crossing times over a post and a platform -/
theorem train_length (post_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  post_time = 15 →
  platform_length = 100 →
  platform_time = 25 →
  ∃ (train_length : ℝ),
    train_length / post_time = (train_length + platform_length) / platform_time ∧
    train_length = 150 := by
  sorry

end train_length_l4127_412772


namespace arithmetic_sequence_length_l4127_412764

/-- 
An arithmetic sequence is defined by its first term, common difference, and last term.
This theorem proves that an arithmetic sequence with first term -6, last term 38,
and common difference 4 has exactly 12 terms.
-/
theorem arithmetic_sequence_length 
  (a : ℤ) (d : ℤ) (l : ℤ) (n : ℕ) 
  (h1 : a = -6) 
  (h2 : d = 4) 
  (h3 : l = 38) 
  (h4 : l = a + (n - 1) * d) : n = 12 :=
sorry

end arithmetic_sequence_length_l4127_412764


namespace symmetry_axis_of_translated_sine_function_l4127_412760

/-- Given a function f(x) = 2sin(2x + π/6), g(x) is obtained by translating
    the graph of f(x) to the right by π/6 units. This theorem states that
    x = π/3 is an equation of one symmetry axis of g(x). -/
theorem symmetry_axis_of_translated_sine_function :
  ∀ (f g : ℝ → ℝ),
  (∀ x, f x = 2 * Real.sin (2 * x + π / 6)) →
  (∀ x, g x = f (x - π / 6)) →
  (∃ k : ℤ, 2 * (π / 3) - π / 6 = π / 2 + k * π) :=
by sorry

end symmetry_axis_of_translated_sine_function_l4127_412760


namespace souvenir_distribution_solution_l4127_412716

/-- Represents the souvenir distribution problem -/
structure SouvenirDistribution where
  total_items : ℕ
  total_cost : ℕ
  type_a_cost : ℕ
  type_a_price : ℕ
  type_b_cost : ℕ
  type_b_price : ℕ

/-- Theorem stating the solution to the souvenir distribution problem -/
theorem souvenir_distribution_solution (sd : SouvenirDistribution)
  (h1 : sd.total_items = 100)
  (h2 : sd.total_cost = 6200)
  (h3 : sd.type_a_cost = 50)
  (h4 : sd.type_a_price = 100)
  (h5 : sd.type_b_cost = 70)
  (h6 : sd.type_b_price = 90) :
  ∃ (type_a type_b : ℕ),
    type_a + type_b = sd.total_items ∧
    type_a * sd.type_a_cost + type_b * sd.type_b_cost = sd.total_cost ∧
    type_a = 40 ∧
    type_b = 60 ∧
    (type_a * (sd.type_a_price - sd.type_a_cost) + type_b * (sd.type_b_price - sd.type_b_cost)) = 3200 :=
by
  sorry

end souvenir_distribution_solution_l4127_412716


namespace f_is_quadratic_l4127_412786

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 3x^2 - 6 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end f_is_quadratic_l4127_412786


namespace wire_length_around_square_field_l4127_412759

theorem wire_length_around_square_field (area : ℝ) (num_rounds : ℕ) : 
  area = 27889 ∧ num_rounds = 11 → 
  Real.sqrt area * 4 * num_rounds = 7348 := by
sorry

end wire_length_around_square_field_l4127_412759


namespace perpendicular_vector_scalar_l4127_412714

/-- Given two vectors a and b in ℝ², prove that if a + xb is perpendicular to b, then x = -2/5 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (x : ℝ) 
    (h1 : a = (3, 4))
    (h2 : b = (2, -1))
    (h3 : (a.1 + x * b.1, a.2 + x * b.2) • b = 0) :
  x = -2/5 := by
  sorry

#check perpendicular_vector_scalar

end perpendicular_vector_scalar_l4127_412714


namespace negation_square_positive_l4127_412793

theorem negation_square_positive :
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := by sorry

end negation_square_positive_l4127_412793


namespace trivia_game_score_l4127_412715

/-- The final score of a trivia game given the scores of three rounds -/
def final_score (round1 : Int) (round2 : Int) (round3 : Int) : Int :=
  round1 + round2 + round3

/-- Theorem: Given the scores from three rounds of a trivia game (16, 33, and -48),
    the final score is equal to 1. -/
theorem trivia_game_score :
  final_score 16 33 (-48) = 1 := by
  sorry

end trivia_game_score_l4127_412715


namespace sin_160_cos_10_plus_cos_20_sin_10_l4127_412730

theorem sin_160_cos_10_plus_cos_20_sin_10 :
  Real.sin (160 * π / 180) * Real.cos (10 * π / 180) +
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end sin_160_cos_10_plus_cos_20_sin_10_l4127_412730


namespace union_of_sets_l4127_412785

theorem union_of_sets : 
  let A : Set ℕ := {2, 3}
  let B : Set ℕ := {1, 2}
  A ∪ B = {1, 2, 3} := by
sorry

end union_of_sets_l4127_412785


namespace complex_conversion_l4127_412746

theorem complex_conversion (z : ℂ) : z = Complex.exp (13 * Real.pi * Complex.I / 4) * (Real.sqrt 3) →
  z = Complex.mk (Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

end complex_conversion_l4127_412746


namespace inequality_solution_and_geometric_mean_l4127_412770

theorem inequality_solution_and_geometric_mean (a b m : ℝ) : 
  (∀ x, (x - 2) / (a * x + b) > 0 ↔ -1 < x ∧ x < 2) →
  m^2 = a * b →
  (3 * m^2 * a) / (a^3 + 2 * b^3) = 1 := by sorry

end inequality_solution_and_geometric_mean_l4127_412770


namespace matias_grade_size_l4127_412758

/-- Given a student's rank from best and worst in a group, calculate the total number of students -/
def totalStudents (rankBest : ℕ) (rankWorst : ℕ) : ℕ :=
  (rankBest - 1) + (rankWorst - 1) + 1

/-- Theorem: In a group where a student is both the 75th best and 75th worst, there are 149 students -/
theorem matias_grade_size :
  totalStudents 75 75 = 149 := by
  sorry

#eval totalStudents 75 75

end matias_grade_size_l4127_412758


namespace division_remainder_l4127_412768

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 190 →
  divisor = 21 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 1 := by
sorry

end division_remainder_l4127_412768


namespace inequality_solution_l4127_412702

theorem inequality_solution :
  ∀ x : ℕ, 1 + x ≥ 2 * x - 1 ↔ x ∈ ({0, 1, 2} : Set ℕ) := by
  sorry

end inequality_solution_l4127_412702


namespace rearrangement_time_l4127_412749

/-- The time required to write all rearrangements of a name -/
theorem rearrangement_time (name_length : ℕ) (writing_speed : ℕ) (h1 : name_length = 8) (h2 : writing_speed = 15) :
  (name_length.factorial / writing_speed : ℚ) / 60 = 44.8 := by
sorry

end rearrangement_time_l4127_412749


namespace computer_operations_l4127_412780

/-- Represents the computer's specifications and performance --/
structure ComputerSpec where
  mult_rate : ℕ  -- multiplications per second
  add_rate : ℕ   -- additions per second
  switch_time : ℕ  -- time in seconds when switching from multiplications to additions
  total_time : ℕ  -- total operation time in seconds

/-- Calculates the total number of operations performed by the computer --/
def total_operations (spec : ComputerSpec) : ℕ :=
  let mult_ops := spec.mult_rate * spec.switch_time
  let add_ops := spec.add_rate * (spec.total_time - spec.switch_time)
  mult_ops + add_ops

/-- Theorem stating that the computer performs 63,000,000 operations in 2 hours --/
theorem computer_operations :
  let spec : ComputerSpec := {
    mult_rate := 5000,
    add_rate := 10000,
    switch_time := 1800,
    total_time := 7200
  }
  total_operations spec = 63000000 := by
  sorry


end computer_operations_l4127_412780


namespace solution_pairs_l4127_412708

/-- Sum of factorials from 1 to k -/
def sumFactorials (k : ℕ) : ℕ :=
  (List.range k).map Nat.factorial |>.sum

/-- Sum of integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The set of pairs (k, n) that satisfy the equation -/
def solutionSet : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ sumFactorials p.1 = sumIntegers p.2}

theorem solution_pairs : solutionSet = {(1, 1), (2, 2), (5, 17)} := by
  sorry


end solution_pairs_l4127_412708


namespace tv_sales_effect_l4127_412773

theorem tv_sales_effect (P Q : ℝ) (h_P : P > 0) (h_Q : Q > 0) : 
  let new_price := 0.82 * P
  let new_quantity := 1.88 * Q
  let original_value := P * Q
  let new_value := new_price * new_quantity
  (new_value / original_value - 1) * 100 = 54.26 := by
sorry

end tv_sales_effect_l4127_412773


namespace complex_equation_solution_l4127_412761

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end complex_equation_solution_l4127_412761


namespace divisible_by_77_l4127_412787

theorem divisible_by_77 (n : ℤ) : ∃ k : ℤ, n^18 - n^12 - n^8 + n^2 = 77 * k := by
  sorry

end divisible_by_77_l4127_412787


namespace solution_satisfies_equation_l4127_412709

theorem solution_satisfies_equation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  let x := (a^2 - b^2) / (2*a)
  (x^2 + b^2 + c^2) = ((a - x)^2 + c^2) :=
by sorry

end solution_satisfies_equation_l4127_412709


namespace replaced_student_weight_l4127_412721

theorem replaced_student_weight 
  (n : ℕ) 
  (new_weight : ℝ) 
  (avg_decrease : ℝ) : 
  n = 8 → 
  new_weight = 46 → 
  avg_decrease = 5 → 
  ∃ (old_weight : ℝ), old_weight = n * avg_decrease + new_weight :=
by
  sorry

end replaced_student_weight_l4127_412721


namespace ranking_sequences_count_l4127_412725

/-- Represents a player in the chess tournament -/
inductive Player : Type
| A : Player
| B : Player
| C : Player
| D : Player

/-- Represents a match between two players -/
structure Match :=
(player1 : Player)
(player2 : Player)

/-- Represents the tournament structure -/
structure Tournament :=
(initial_match1 : Match)
(initial_match2 : Match)
(winners_match : Match)
(losers_match : Match)
(third_place_match : Match)

/-- A function to calculate the number of possible ranking sequences -/
def count_ranking_sequences (t : Tournament) : Nat :=
  sorry

/-- The theorem stating that the number of possible ranking sequences is 8 -/
theorem ranking_sequences_count :
  ∀ t : Tournament, count_ranking_sequences t = 8 :=
sorry

end ranking_sequences_count_l4127_412725


namespace simplify_sqrt_three_l4127_412791

theorem simplify_sqrt_three : 3 * Real.sqrt 3 - 2 * Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end simplify_sqrt_three_l4127_412791


namespace part_one_part_two_range_of_m_l4127_412707

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
sorry

-- Part 2
theorem part_two :
  ∀ x : ℝ, f 1 x + f 1 (x + 5) ≥ 5 :=
sorry

-- Range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, f 1 x + f 1 (x + 5) ≥ m) ↔ m ≤ 5 :=
sorry

end part_one_part_two_range_of_m_l4127_412707


namespace box_makers_l4127_412734

/-- Represents the possible makers of the boxes -/
inductive Maker
| Cellini
| CelliniSon
| Bellini
| BelliniSon

/-- Represents a box with its inscription and actual maker -/
structure Box where
  color : String
  inscription : Prop
  maker : Maker

/-- The setup of the problem with two boxes -/
def boxSetup (goldBox silverBox : Box) : Prop :=
  (goldBox.color = "gold" ∧ silverBox.color = "silver") ∧
  (goldBox.inscription = (goldBox.maker = Maker.Cellini ∨ goldBox.maker = Maker.CelliniSon) ∧
                         (silverBox.maker = Maker.Cellini ∨ silverBox.maker = Maker.CelliniSon)) ∧
  (silverBox.inscription = (goldBox.maker ≠ Maker.CelliniSon ∧ goldBox.maker ≠ Maker.BelliniSon) ∧
                           (silverBox.maker ≠ Maker.CelliniSon ∧ silverBox.maker ≠ Maker.BelliniSon)) ∧
  (goldBox.inscription ≠ silverBox.inscription)

theorem box_makers (goldBox silverBox : Box) :
  boxSetup goldBox silverBox →
  (goldBox.maker = Maker.Cellini ∧ silverBox.maker = Maker.Bellini) :=
by
  sorry

end box_makers_l4127_412734


namespace soccer_ball_max_height_l4127_412779

/-- The height function of a soccer ball's trajectory -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 11

/-- Theorem stating the maximum height of the soccer ball -/
theorem soccer_ball_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 136 :=
sorry

end soccer_ball_max_height_l4127_412779


namespace probability_triangle_or_circle_l4127_412731

def total_figures : ℕ := 10
def triangle_count : ℕ := 3
def circle_count : ℕ := 3

theorem probability_triangle_or_circle :
  (triangle_count + circle_count : ℚ) / total_figures = 3 / 5 :=
sorry

end probability_triangle_or_circle_l4127_412731


namespace geometric_sequence_sufficient_not_necessary_l4127_412706

/-- Defines a geometric sequence -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- Defines the condition b^2 = ac -/
def condition_b_squared_eq_ac (a b c : ℝ) : Prop :=
  b^2 = a * c

/-- Theorem stating that "a, b, c form a geometric sequence" is sufficient 
    but not necessary for "b^2 = ac" -/
theorem geometric_sequence_sufficient_not_necessary :
  (∀ a b c : ℝ, is_geometric_sequence a b c → condition_b_squared_eq_ac a b c) ∧
  ¬(∀ a b c : ℝ, condition_b_squared_eq_ac a b c → is_geometric_sequence a b c) :=
by sorry

end geometric_sequence_sufficient_not_necessary_l4127_412706


namespace sum_of_digits_11_pow_2003_l4127_412722

theorem sum_of_digits_11_pow_2003 : ∃ n : ℕ, 
  11^2003 = 100 * n + 31 ∧ 3 + 1 = 4 := by sorry

end sum_of_digits_11_pow_2003_l4127_412722


namespace cut_scene_length_is_8_minutes_l4127_412737

/-- Calculates the length of a cut scene given the original and final movie lengths -/
def cut_scene_length (original_length final_length : ℕ) : ℕ :=
  original_length - final_length

theorem cut_scene_length_is_8_minutes :
  cut_scene_length 60 52 = 8 := by
  sorry

end cut_scene_length_is_8_minutes_l4127_412737


namespace negative_inequality_l4127_412700

theorem negative_inequality (h : 3.14 < Real.pi) : -3.14 > -Real.pi := by
  sorry

end negative_inequality_l4127_412700


namespace wall_bricks_l4127_412743

/-- The number of bricks in the wall -/
def num_bricks : ℕ := 720

/-- The time it takes Brenda to build the wall alone (in hours) -/
def brenda_time : ℕ := 12

/-- The time it takes Brandon to build the wall alone (in hours) -/
def brandon_time : ℕ := 15

/-- The decrease in combined output when working together (in bricks per hour) -/
def output_decrease : ℕ := 12

/-- The time it takes Brenda and Brandon to build the wall together (in hours) -/
def combined_time : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 720 -/
theorem wall_bricks : 
  (combined_time : ℚ) * ((num_bricks / brenda_time : ℚ) + (num_bricks / brandon_time : ℚ) - output_decrease) = num_bricks := by
  sorry

end wall_bricks_l4127_412743


namespace cafeteria_seats_available_l4127_412720

theorem cafeteria_seats_available 
  (total_tables : ℕ) 
  (seats_per_table : ℕ) 
  (people_dining : ℕ) : 
  total_tables = 40 → 
  seats_per_table = 12 → 
  people_dining = 325 → 
  total_tables * seats_per_table - people_dining = 155 := by
sorry

end cafeteria_seats_available_l4127_412720


namespace slope_inequality_l4127_412753

open Real

theorem slope_inequality (x₁ x₂ : ℝ) (hx : 0 < x₁ ∧ x₁ < x₂) :
  let f := λ x : ℝ => Real.log x
  let k := (f x₂ - f x₁) / (x₂ - x₁)
  1 / x₂ < k ∧ k < 1 / x₁ := by
  sorry

end slope_inequality_l4127_412753


namespace debby_flour_purchase_l4127_412704

/-- Given that Debby initially had 12 pounds of flour and ended up with 16 pounds in total,
    prove that she bought 4 pounds of flour. -/
theorem debby_flour_purchase (initial_flour : ℕ) (total_flour : ℕ) (purchased_flour : ℕ) :
  initial_flour = 12 →
  total_flour = 16 →
  total_flour = initial_flour + purchased_flour →
  purchased_flour = 4 := by
  sorry

end debby_flour_purchase_l4127_412704


namespace line_perp_plane_implies_perp_line_l4127_412718

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_implies_perp_line 
  (α β : Plane) (m n : Line) 
  (h_diff_planes : α ≠ β)
  (h_diff_lines : m ≠ n)
  (h_m_subset_β : subset m β)
  (h_n_subset_β : subset n β)
  (h_m_subset_α : subset m α)
  (h_n_perp_α : perp n α) :
  perpLine n m :=
sorry

end line_perp_plane_implies_perp_line_l4127_412718


namespace lesser_number_l4127_412739

theorem lesser_number (x y : ℝ) (h1 : x + y = 60) (h2 : 3 * (x - y) = 9) : min x y = 28.5 := by
  sorry

end lesser_number_l4127_412739


namespace exam_score_theorem_l4127_412726

/-- Represents an examination scoring system and a student's performance --/
structure ExamScore where
  correct_score : ℕ      -- Marks awarded for each correct answer
  wrong_penalty : ℕ      -- Marks deducted for each wrong answer
  total_score : ℤ        -- Total score achieved
  correct_answers : ℕ    -- Number of correct answers
  total_questions : ℕ    -- Total number of questions attempted

/-- Theorem stating that given the exam conditions, the total questions attempted is 75 --/
theorem exam_score_theorem (exam : ExamScore) 
  (h1 : exam.correct_score = 4)
  (h2 : exam.wrong_penalty = 1)
  (h3 : exam.total_score = 125)
  (h4 : exam.correct_answers = 40) :
  exam.total_questions = 75 := by
  sorry


end exam_score_theorem_l4127_412726


namespace line_through_points_and_midpoint_l4127_412745

/-- Given a line y = ax + b passing through (2, 3) and (10, 19) with their midpoint on the line, a - b = 3 -/
theorem line_through_points_and_midpoint (a b : ℝ) : 
  (3 = a * 2 + b) → 
  (19 = a * 10 + b) → 
  (11 = a * 6 + b) → 
  a - b = 3 := by sorry

end line_through_points_and_midpoint_l4127_412745


namespace complex_square_roots_l4127_412776

theorem complex_square_roots (z : ℂ) : 
  z ^ 2 = -104 + 63 * I ∧ (5 + 8 * I) ^ 2 = -104 + 63 * I → 
  (-5 - 8 * I) ^ 2 = -104 + 63 * I := by
sorry

end complex_square_roots_l4127_412776


namespace max_added_value_max_at_two_thirds_verify_half_a_l4127_412710

/-- The added value function for a car factory's production line -/
def f (a : ℝ) (x : ℝ) : ℝ := 8 * (a - x) * x^2

/-- Theorem stating the maximum value of the added value function -/
theorem max_added_value (a : ℝ) (h_a : a > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (4 * a / 5) ∧
    f a x = (32 / 27) * a^3 ∧
    ∀ (y : ℝ), y ∈ Set.Ioo 0 (4 * a / 5) → f a y ≤ (32 / 27) * a^3 :=
by sorry

/-- Theorem stating that the maximum occurs at x = 2a/3 -/
theorem max_at_two_thirds (a : ℝ) (h_a : a > 0) :
  f a (2 * a / 3) = (32 / 27) * a^3 :=
by sorry

/-- Theorem verifying that f(a/2) = a^3 -/
theorem verify_half_a (a : ℝ) :
  f a (a / 2) = a^3 :=
by sorry

end max_added_value_max_at_two_thirds_verify_half_a_l4127_412710


namespace simple_interest_rate_problem_l4127_412795

/-- The simple interest rate problem -/
theorem simple_interest_rate_problem 
  (simple_interest : ℝ) 
  (principal : ℝ) 
  (time : ℝ) 
  (h1 : simple_interest = 16.32)
  (h2 : principal = 34)
  (h3 : time = 8)
  (h4 : simple_interest = principal * (rate / 100) * time) :
  rate = 6 := by
  sorry

end simple_interest_rate_problem_l4127_412795


namespace find_number_l4127_412775

theorem find_number : ∃ x : ℤ, 4 * x + 100 = 4100 := by
  sorry

end find_number_l4127_412775


namespace boat_distance_along_stream_l4127_412788

/-- A boat travels on a river with a current. -/
structure Boat :=
  (speed : ℝ)  -- Speed of the boat in still water in km/h

/-- A river with a current. -/
structure River :=
  (current : ℝ)  -- Speed of the river current in km/h

/-- The distance traveled by a boat on a river in one hour. -/
def distanceTraveled (b : Boat) (r : River) (withCurrent : Bool) : ℝ :=
  if withCurrent then b.speed + r.current else b.speed - r.current

theorem boat_distance_along_stream 
  (b : Boat) 
  (r : River) 
  (h1 : b.speed = 8) 
  (h2 : distanceTraveled b r false = 5) : 
  distanceTraveled b r true = 11 := by
  sorry

#check boat_distance_along_stream

end boat_distance_along_stream_l4127_412788


namespace midpoint_property_l4127_412755

/-- Given two points P and Q in the plane, their midpoint R satisfies 3x + 2y = 39 --/
theorem midpoint_property (P Q R : ℝ × ℝ) : 
  P = (12, 9) → Q = (4, 6) → R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  3 * R.1 + 2 * R.2 = 39 := by
  sorry

#check midpoint_property

end midpoint_property_l4127_412755


namespace minimum_additional_games_minimum_additional_games_is_146_l4127_412769

theorem minimum_additional_games : ℕ → Prop :=
  fun n =>
    let initial_games : ℕ := 4
    let initial_lions_wins : ℕ := 3
    let initial_eagles_wins : ℕ := 1
    let total_games : ℕ := initial_games + n
    let total_eagles_wins : ℕ := initial_eagles_wins + n
    (total_eagles_wins : ℚ) / (total_games : ℚ) ≥ 98 / 100 ∧
    ∀ m : ℕ, m < n →
      let total_games_m : ℕ := initial_games + m
      let total_eagles_wins_m : ℕ := initial_eagles_wins + m
      (total_eagles_wins_m : ℚ) / (total_games_m : ℚ) < 98 / 100

theorem minimum_additional_games_is_146 : minimum_additional_games 146 := by
  sorry

#check minimum_additional_games_is_146

end minimum_additional_games_minimum_additional_games_is_146_l4127_412769


namespace inequality_proof_l4127_412789

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end inequality_proof_l4127_412789


namespace power_23_2005_mod_36_l4127_412738

theorem power_23_2005_mod_36 : 23^2005 % 36 = 11 := by
  sorry

end power_23_2005_mod_36_l4127_412738


namespace mary_max_earnings_l4127_412765

/-- Calculates the maximum weekly earnings for a worker with the given conditions --/
def maxWeeklyEarnings (maxHours : ℕ) (regularHours : ℕ) (regularRate : ℚ) (overtimeRateIncrease : ℚ) : ℚ :=
  let overtimeHours := maxHours - regularHours
  let overtimeRate := regularRate * (1 + overtimeRateIncrease)
  regularHours * regularRate + overtimeHours * overtimeRate

/-- Theorem stating that Mary's maximum weekly earnings are $410 --/
theorem mary_max_earnings :
  maxWeeklyEarnings 45 20 8 (1/4) = 410 := by
  sorry

#eval maxWeeklyEarnings 45 20 8 (1/4)

end mary_max_earnings_l4127_412765


namespace max_sales_and_profit_l4127_412794

-- Define the sales volume function for the first 4 days
def sales_volume_early (x : ℝ) : ℝ := 20 * x + 80

-- Define the sales volume function for days 6 to 20
def sales_volume_late (x : ℝ) : ℝ := -x^2 + 50*x - 100

-- Define the selling price function for the first 5 days
def selling_price (x : ℝ) : ℝ := 2 * x + 28

-- Define the cost price
def cost_price : ℝ := 22

-- Define the profit function for days 1 to 5
def profit_early (x : ℝ) : ℝ := (selling_price x - cost_price) * sales_volume_early x

-- Define the profit function for days 6 to 20
def profit_late (x : ℝ) : ℝ := (28 - cost_price) * sales_volume_late x

theorem max_sales_and_profit :
  (∀ x ∈ Set.Icc 6 20, sales_volume_late x ≤ sales_volume_late 20) ∧
  sales_volume_late 20 = 500 ∧
  (∀ x ∈ Set.Icc 1 20, profit_early x ≤ profit_late 20 ∧ profit_late x ≤ profit_late 20) ∧
  profit_late 20 = 3000 := by
  sorry

end max_sales_and_profit_l4127_412794


namespace exists_close_points_on_graphs_l4127_412701

open Real

/-- The function f(x) = x^4 -/
def f (x : ℝ) : ℝ := x^4

/-- The function g(x) = x^4 + x^2 + x + 1 -/
def g (x : ℝ) : ℝ := x^4 + x^2 + x + 1

/-- Theorem stating the existence of points A and B on the graphs of f and g with distance < 1/100 -/
theorem exists_close_points_on_graphs :
  ∃ (u v : ℝ), |u - v| < 1/100 ∧ f v = g u := by sorry

end exists_close_points_on_graphs_l4127_412701


namespace two_digit_reverse_pythagoras_sum_l4127_412792

theorem two_digit_reverse_pythagoras_sum : ∃ (x y n : ℕ), 
  (10 ≤ x ∧ x < 100) ∧ 
  (10 ≤ y ∧ y < 100) ∧ 
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a ∧ a < 10 ∧ b < 10) ∧
  x^2 + y^2 = n^2 ∧
  x + y + n = 132 := by
  sorry

end two_digit_reverse_pythagoras_sum_l4127_412792


namespace rational_terms_not_adjacent_probability_l4127_412771

theorem rational_terms_not_adjacent_probability (n : ℕ) (rational_terms : ℕ) :
  n = 9 ∧ rational_terms = 3 →
  (Nat.factorial 6 * (Nat.factorial 7 / (Nat.factorial 4 * Nat.factorial 3))) / Nat.factorial 9 = 5 / 12 := by
  sorry

end rational_terms_not_adjacent_probability_l4127_412771


namespace rabbit_speed_l4127_412728

theorem rabbit_speed (x : ℕ) : ((2 * x + 4) * 2 = 188) ↔ (x = 45) := by
  sorry

end rabbit_speed_l4127_412728


namespace square_to_obtuse_triangle_l4127_412748

/-- Represents a part of a square -/
structure SquarePart where
  -- Add necessary fields to represent a part of a square
  -- This is a placeholder and should be defined more precisely based on the problem requirements

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields to represent a triangle
  -- This is a placeholder and should be defined more precisely based on the problem requirements

/-- Determines if a triangle is obtuse -/
def is_obtuse (t : Triangle) : Prop :=
  -- Add the condition for a triangle to be obtuse
  -- This is a placeholder and should be defined more precisely based on the problem requirements
  sorry

/-- Determines if parts can form a triangle -/
def can_form_triangle (parts : List SquarePart) : Prop :=
  -- Add the condition for parts to be able to form a triangle
  -- This is a placeholder and should be defined more precisely based on the problem requirements
  sorry

/-- Theorem stating that a square can be cut into 3 parts that form an obtuse triangle -/
theorem square_to_obtuse_triangle :
  ∃ (parts : List SquarePart), parts.length = 3 ∧
    ∃ (t : Triangle), can_form_triangle parts ∧ is_obtuse t :=
sorry

end square_to_obtuse_triangle_l4127_412748


namespace fraction_of_108_l4127_412741

theorem fraction_of_108 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 108 = 3 := by
  sorry

end fraction_of_108_l4127_412741


namespace intersection_of_A_and_B_l4127_412757

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l4127_412757


namespace triangle_altitude_length_l4127_412703

theorem triangle_altitude_length (r : ℝ) (h : r > 0) : 
  let square_side : ℝ := 4 * r
  let square_area : ℝ := square_side ^ 2
  let diagonal_length : ℝ := square_side * Real.sqrt 2
  let triangle_area : ℝ := 2 * square_area
  let altitude : ℝ := 2 * triangle_area / diagonal_length
  altitude = 8 * r * Real.sqrt 2 := by sorry

end triangle_altitude_length_l4127_412703


namespace number_puzzle_l4127_412713

theorem number_puzzle : ∃ x : ℚ, x = (3/5) * (2*x) + 238 ∧ x = 1190 := by
  sorry

end number_puzzle_l4127_412713


namespace backpack_price_increase_l4127_412723

theorem backpack_price_increase 
  (original_backpack_price : ℕ)
  (original_binder_price : ℕ)
  (num_binders : ℕ)
  (binder_price_reduction : ℕ)
  (total_spent : ℕ)
  (h1 : original_backpack_price = 50)
  (h2 : original_binder_price = 20)
  (h3 : num_binders = 3)
  (h4 : binder_price_reduction = 2)
  (h5 : total_spent = 109)
  : ∃ (price_increase : ℕ), 
    original_backpack_price + price_increase + 
    num_binders * (original_binder_price - binder_price_reduction) = total_spent ∧
    price_increase = 5 := by
  sorry

end backpack_price_increase_l4127_412723


namespace dance_group_composition_l4127_412727

/-- Represents a dance group --/
structure DanceGroup where
  boy_dancers : ℕ
  girl_dancers : ℕ
  boy_escorts : ℕ
  girl_escorts : ℕ

/-- The problem statement --/
theorem dance_group_composition 
  (group_a group_b : DanceGroup)
  (h1 : group_a.boy_dancers + group_a.girl_dancers = group_b.boy_dancers + group_b.girl_dancers + 1)
  (h2 : group_a.boy_escorts + group_a.girl_escorts = group_b.boy_escorts + group_b.girl_escorts + 1)
  (h3 : group_a.boy_dancers + group_b.boy_dancers = group_a.girl_dancers + group_b.girl_dancers + 1)
  (h4 : (group_a.boy_dancers + group_b.boy_dancers) * (group_a.girl_dancers + group_b.girl_dancers) = 484)
  (h5 : (group_a.boy_dancers + group_a.boy_escorts) * (group_b.girl_dancers + group_b.girl_escorts) +
        (group_b.boy_dancers + group_b.boy_escorts) * (group_a.girl_dancers + group_a.girl_escorts) = 246)
  (h6 : (group_a.boy_dancers + group_b.boy_dancers) * (group_a.girl_dancers + group_b.girl_dancers) = 306)
  (h7 : group_a.boy_dancers * group_a.girl_dancers + group_b.boy_dancers * group_b.girl_dancers = 150)
  (h8 : let total := group_a.boy_dancers + group_a.girl_dancers + group_a.boy_escorts + group_a.girl_escorts +
                     group_b.boy_dancers + group_b.girl_dancers + group_b.boy_escorts + group_b.girl_escorts
        (total * (total - 1)) / 2 = 946) :
  group_a = { boy_dancers := 8, girl_dancers := 10, boy_escorts := 2, girl_escorts := 3 } ∧
  group_b = { boy_dancers := 10, girl_dancers := 7, boy_escorts := 2, girl_escorts := 2 } :=
by sorry

end dance_group_composition_l4127_412727


namespace parallel_tangents_theorem_l4127_412766

-- Define the curve C
def C (a b d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + d

-- Define the derivative of C
def C_derivative (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem parallel_tangents_theorem (a b d : ℝ) :
  C a b d 1 = 1 →  -- Point A(1,1) is on the curve
  C a b d (-1) = -3 →  -- Point B(-1,-3) is on the curve
  C_derivative a b 1 = C_derivative a b (-1) →  -- Tangents at A and B are parallel
  a^3 + b^2 + d = 7 := by
  sorry

end parallel_tangents_theorem_l4127_412766


namespace maintenance_team_journey_l4127_412705

/-- Represents the direction of travel --/
inductive Direction
  | East
  | West

/-- Represents a segment of the journey --/
structure Segment where
  distance : ℝ
  direction : Direction

/-- Calculates the net distance traveled given a list of segments --/
def netDistance (journey : List Segment) : ℝ := sorry

/-- Calculates the total distance traveled given a list of segments --/
def totalDistance (journey : List Segment) : ℝ := sorry

/-- Theorem: The maintenance team's final position and fuel consumption --/
theorem maintenance_team_journey 
  (journey : List Segment)
  (fuel_rate : ℝ)
  (h1 : journey = [
    ⟨12, Direction.East⟩, 
    ⟨6, Direction.West⟩, 
    ⟨4, Direction.East⟩, 
    ⟨2, Direction.West⟩, 
    ⟨8, Direction.West⟩, 
    ⟨13, Direction.East⟩, 
    ⟨2, Direction.West⟩
  ])
  (h2 : fuel_rate = 0.2) :
  netDistance journey = 11 ∧ 
  totalDistance journey * fuel_rate * 2 = 11.6 := by sorry

end maintenance_team_journey_l4127_412705


namespace gcd_128_144_360_l4127_412799

theorem gcd_128_144_360 : Nat.gcd 128 (Nat.gcd 144 360) = 72 := by
  sorry

end gcd_128_144_360_l4127_412799


namespace worker_days_calculation_l4127_412790

theorem worker_days_calculation (wages_group1 wages_group2 : ℚ)
  (workers_group1 workers_group2 : ℕ) (days_group2 : ℕ) :
  wages_group1 = 9450 →
  wages_group2 = 9975 →
  workers_group1 = 15 →
  workers_group2 = 19 →
  days_group2 = 5 →
  ∃ (days_group1 : ℕ),
    (wages_group1 / (workers_group1 * days_group1 : ℚ)) =
    (wages_group2 / (workers_group2 * days_group2 : ℚ)) ∧
    days_group1 = 6 :=
by sorry

end worker_days_calculation_l4127_412790


namespace factors_of_72_l4127_412744

def number_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_72 : number_of_factors 72 = 12 := by
  sorry

end factors_of_72_l4127_412744


namespace square_of_fraction_equals_4088484_l4127_412733

theorem square_of_fraction_equals_4088484 :
  ((2023^2 - 2023) / 2023)^2 = 4088484 := by
  sorry

end square_of_fraction_equals_4088484_l4127_412733
