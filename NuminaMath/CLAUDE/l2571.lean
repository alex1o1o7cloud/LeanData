import Mathlib

namespace geometric_sequence_general_term_l2571_257158

/-- A geometric sequence with given third and tenth terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧ 
  a 3 = 3 ∧ 
  a 10 = 384

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 3 * 2^(n - 3)

/-- Theorem stating that the general term is correct for the given geometric sequence -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  geometric_sequence a → (∀ n : ℕ, a n = general_term n) :=
by sorry

end geometric_sequence_general_term_l2571_257158


namespace max_initial_pieces_is_285_l2571_257199

/-- Represents a Go board with dimensions n x n -/
structure GoBoard (n : ℕ) where
  size : n > 0

/-- Represents a rectangular arrangement of pieces on a Go board -/
structure Rectangle (n : ℕ) where
  width : ℕ
  height : ℕ
  pieces : ℕ
  width_valid : width ≤ n
  height_valid : height ≤ n
  area_eq_pieces : width * height = pieces

/-- The maximum number of pieces in the initial rectangle -/
def max_initial_pieces (board : GoBoard 19) : ℕ := 285

/-- Theorem stating the maximum number of pieces in the initial rectangle -/
theorem max_initial_pieces_is_285 (board : GoBoard 19) :
  ∃ (init final : Rectangle 19),
    init.pieces = max_initial_pieces board ∧
    final.pieces = init.pieces + 45 ∧
    final.width = init.width ∧
    final.height > init.height ∧
    ∀ (other : Rectangle 19),
      (∃ (other_final : Rectangle 19),
        other_final.pieces = other.pieces + 45 ∧
        other_final.width = other.width ∧
        other_final.height > other.height) →
      other.pieces ≤ init.pieces :=
by sorry

end max_initial_pieces_is_285_l2571_257199


namespace hockey_team_boys_percentage_l2571_257161

theorem hockey_team_boys_percentage
  (total_players : ℕ)
  (junior_girls : ℕ)
  (h1 : total_players = 50)
  (h2 : junior_girls = 10)
  (h3 : junior_girls = total_players - junior_girls - (total_players - 2 * junior_girls)) :
  (total_players - 2 * junior_girls : ℚ) / total_players = 3/5 := by
  sorry

end hockey_team_boys_percentage_l2571_257161


namespace cards_distribution_l2571_257176

/-- Given a total number of cards and people, calculates how many people receive fewer than the ceiling of the average number of cards. -/
def people_with_fewer_cards (total_cards : ℕ) (total_people : ℕ) : ℕ :=
  let avg_cards := total_cards / total_people
  let remainder := total_cards % total_people
  let max_cards := avg_cards + 1
  total_people - remainder

theorem cards_distribution (total_cards : ℕ) (total_people : ℕ) 
  (h1 : total_cards = 60) (h2 : total_people = 9) :
  people_with_fewer_cards total_cards total_people = 3 := by
  sorry

#eval people_with_fewer_cards 60 9

end cards_distribution_l2571_257176


namespace multiplication_increase_l2571_257136

theorem multiplication_increase (n : ℕ) (x : ℚ) (h : n = 25) :
  n * x = n + 375 → x = 16 := by
  sorry

end multiplication_increase_l2571_257136


namespace prob_12th_roll_last_correct_l2571_257150

/-- The probability of the 12th roll being the last roll when rolling a standard six-sided die
    until getting the same number on consecutive rolls -/
def prob_12th_roll_last : ℚ := (5^10 : ℚ) / (6^11 : ℚ)

/-- Theorem stating that the probability of the 12th roll being the last roll is correct -/
theorem prob_12th_roll_last_correct :
  prob_12th_roll_last = (5^10 : ℚ) / (6^11 : ℚ) := by sorry

end prob_12th_roll_last_correct_l2571_257150


namespace monotone_cubic_implies_m_bound_l2571_257156

/-- A function f: ℝ → ℝ is monotonically increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The cubic function f(x) = x³ + 2x² + mx - 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x - 5

theorem monotone_cubic_implies_m_bound :
  ∀ m : ℝ, MonotonicallyIncreasing (f m) → m ≥ 4/3 := by
  sorry

end monotone_cubic_implies_m_bound_l2571_257156


namespace bridge_length_calculation_l2571_257189

theorem bridge_length_calculation (train_length : ℝ) (signal_post_time : ℝ) (bridge_cross_time : ℝ) :
  train_length = 600 →
  signal_post_time = 40 →
  bridge_cross_time = 360 →
  let train_speed := train_length / signal_post_time
  let bridge_only_time := bridge_cross_time - signal_post_time
  let bridge_length := train_speed * bridge_only_time
  bridge_length = 4800 := by
  sorry

end bridge_length_calculation_l2571_257189


namespace min_steps_equal_iff_path_without_leaves_l2571_257160

/-- Represents a tree of players and ropes -/
structure PlayerTree where
  players : ℕ
  ropes : ℕ
  is_tree : ropes = players - 1

/-- Minimum steps to form a path in unrestricted scenario -/
def min_steps_unrestricted (t : PlayerTree) : ℕ := sorry

/-- Minimum steps to form a path in neighbor-only scenario -/
def min_steps_neighbor_only (t : PlayerTree) : ℕ := sorry

/-- Checks if the tree without leaves is a path -/
def is_path_without_leaves (t : PlayerTree) : Prop := sorry

/-- Main theorem: equality of minimum steps iff tree without leaves is a path -/
theorem min_steps_equal_iff_path_without_leaves (t : PlayerTree) :
  min_steps_unrestricted t = min_steps_neighbor_only t ↔ is_path_without_leaves t := by
  sorry

end min_steps_equal_iff_path_without_leaves_l2571_257160


namespace x_value_in_set_l2571_257139

theorem x_value_in_set (x : ℝ) : x ∈ ({1, 2, x^2 - x} : Set ℝ) → x = 0 ∨ x = 1 := by
  sorry

end x_value_in_set_l2571_257139


namespace payment_for_remaining_worker_l2571_257196

/-- Given a total payment for a job and the fraction of work done by two workers,
    calculate the payment for the remaining worker. -/
theorem payment_for_remaining_worker
  (total_payment : ℚ)
  (work_fraction_two_workers : ℚ)
  (h1 : total_payment = 529)
  (h2 : work_fraction_two_workers = 19 / 23) :
  (1 - work_fraction_two_workers) * total_payment = 92 := by
sorry

end payment_for_remaining_worker_l2571_257196


namespace triangle_line_equations_l2571_257115

/-- Triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Given triangle -/
def givenTriangle : Triangle :=
  { A := (-5, 0)
    B := (3, -3)
    C := (0, 2) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of line AC -/
def lineAC : LineEquation :=
  { a := 2
    b := -5
    c := 10 }

/-- The equation of the median to side BC -/
def medianBC : LineEquation :=
  { a := 1
    b := 13
    c := 5 }

theorem triangle_line_equations (t : Triangle) :
  t = givenTriangle →
  (lineAC.a * t.A.1 + lineAC.b * t.A.2 + lineAC.c = 0 ∧
   lineAC.a * t.C.1 + lineAC.b * t.C.2 + lineAC.c = 0) ∧
  (medianBC.a * t.A.1 + medianBC.b * t.A.2 + medianBC.c = 0 ∧
   medianBC.a * ((t.B.1 + t.C.1) / 2) + medianBC.b * ((t.B.2 + t.C.2) / 2) + medianBC.c = 0) :=
by sorry

end triangle_line_equations_l2571_257115


namespace red_markers_count_l2571_257177

def total_markers : ℕ := 3343
def blue_markers : ℕ := 1028

theorem red_markers_count : total_markers - blue_markers = 2315 := by
  sorry

end red_markers_count_l2571_257177


namespace bridge_length_proof_l2571_257132

/-- Given a train with length 160 meters, traveling at 45 km/hr, that crosses a bridge in 30 seconds, prove that the length of the bridge is 215 meters. -/
theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 160 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 215 := by
sorry

end bridge_length_proof_l2571_257132


namespace cube_surface_area_equals_volume_l2571_257143

theorem cube_surface_area_equals_volume (a : ℝ) (h : a > 0) :
  6 * a^2 = a^3 → a = 6 := by
sorry

end cube_surface_area_equals_volume_l2571_257143


namespace S_is_infinite_l2571_257124

/-- The largest prime divisor of a positive integer -/
def largest_prime_divisor (n : ℕ+) : ℕ+ :=
  sorry

/-- The set of positive integers n where the largest prime divisor of n^4 + n^2 + 1
    equals the largest prime divisor of (n+1)^4 + (n+1)^2 + 1 -/
def S : Set ℕ+ :=
  {n | largest_prime_divisor (n^4 + n^2 + 1) = largest_prime_divisor ((n+1)^4 + (n+1)^2 + 1)}

/-- The main theorem: S is an infinite set -/
theorem S_is_infinite : Set.Infinite S := by
  sorry

end S_is_infinite_l2571_257124


namespace theresa_final_week_hours_l2571_257109

def hours_worked : List Nat := [9, 12, 6, 13, 11]
def total_weeks : Nat := 6
def required_average : Nat := 9

theorem theresa_final_week_hours :
  ∃ x : Nat, 
    (hours_worked.sum + x) / total_weeks = required_average ∧ 
    x = 3 := by
  sorry

end theresa_final_week_hours_l2571_257109


namespace probability_two_pairs_l2571_257153

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The number of ways to choose which two dice will form each pair -/
def numWaysToFormPairs : ℕ := Nat.choose numDice 2

/-- The probability of rolling exactly two pairs of matching numbers
    when four standard six-sided dice are tossed simultaneously -/
theorem probability_two_pairs :
  (numWaysToFormPairs : ℚ) * (1 : ℚ) * (1 / numSides : ℚ) * ((numSides - 1) / numSides : ℚ) * (1 / numSides : ℚ) = 5 / 36 := by
  sorry


end probability_two_pairs_l2571_257153


namespace hexagon_triangles_count_l2571_257169

/-- The number of unit equilateral triangles needed to form a regular hexagon of side length n -/
def triangles_in_hexagon (n : ℕ) : ℕ := 6 * (n * (n + 1) / 2)

/-- Given that 6 unit equilateral triangles can form a regular hexagon with side length 1,
    prove that 126 unit equilateral triangles are needed to form a regular hexagon with side length 6 -/
theorem hexagon_triangles_count :
  triangles_in_hexagon 1 = 6 →
  triangles_in_hexagon 6 = 126 := by
  sorry

end hexagon_triangles_count_l2571_257169


namespace quadratic_factorization_l2571_257190

theorem quadratic_factorization (x : ℝ) : 
  x^2 - 5*x + 3 = (x + 5/2 + Real.sqrt 13/2) * (x + 5/2 - Real.sqrt 13/2) := by
  sorry

end quadratic_factorization_l2571_257190


namespace not_decreasing_everywhere_l2571_257101

theorem not_decreasing_everywhere (f : ℝ → ℝ) (h : f 1 < f 2) :
  ¬(∀ x y : ℝ, x < y → f x ≥ f y) :=
sorry

end not_decreasing_everywhere_l2571_257101


namespace probability_at_least_one_girl_l2571_257106

theorem probability_at_least_one_girl (total_students : Nat) (boys : Nat) (girls : Nat) 
  (selected : Nat) (h1 : total_students = boys + girls) (h2 : total_students = 5) 
  (h3 : boys = 3) (h4 : girls = 2) (h5 : selected = 3) : 
  (Nat.choose total_students selected - Nat.choose boys selected) / 
  Nat.choose total_students selected = 9 / 10 := by
  sorry

end probability_at_least_one_girl_l2571_257106


namespace repeating_decimal_to_fraction_l2571_257191

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 0.7333333333333333 ∧ x = 11 / 15 := by
  sorry

end repeating_decimal_to_fraction_l2571_257191


namespace cost_price_per_metre_l2571_257102

/-- Given a cloth sale with a loss, calculate the cost price per metre. -/
theorem cost_price_per_metre
  (total_metres : ℕ)
  (total_selling_price : ℕ)
  (loss_per_metre : ℕ)
  (h1 : total_metres = 600)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_metre = 5) :
  (total_selling_price + total_metres * loss_per_metre) / total_metres = 35 := by
  sorry

#check cost_price_per_metre

end cost_price_per_metre_l2571_257102


namespace evaluate_expression_l2571_257195

theorem evaluate_expression : (122^2 - 115^2 + 7) / 14 = 119 := by sorry

end evaluate_expression_l2571_257195


namespace cosine_ratio_comparison_l2571_257198

theorem cosine_ratio_comparison : 
  (Real.cos (2016 * π / 180)) / (Real.cos (2017 * π / 180)) < 
  (Real.cos (2018 * π / 180)) / (Real.cos (2019 * π / 180)) := by
  sorry

end cosine_ratio_comparison_l2571_257198


namespace min_sum_reciprocals_l2571_257171

/-- For positive real numbers a, b, c with abc = 1, the sum S = 1/(2a+1) + 1/(2b+1) + 1/(2c+1) is greater than or equal to 1 -/
theorem min_sum_reciprocals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  1 / (2 * a + 1) + 1 / (2 * b + 1) + 1 / (2 * c + 1) ≥ 1 := by
  sorry

end min_sum_reciprocals_l2571_257171


namespace angle_bisector_implies_line_equation_l2571_257148

/-- Given points A and B, and a line representing the angle bisector of ∠ACB,
    prove that the line AC has the equation x - 2y - 1 = 0 -/
theorem angle_bisector_implies_line_equation 
  (A B : ℝ × ℝ)
  (angle_bisector : ℝ → ℝ)
  (h1 : A = (3, 1))
  (h2 : B = (-1, 2))
  (h3 : ∀ x y, y = angle_bisector x ↔ y = x + 1)
  (h4 : ∃ C : ℝ × ℝ, (angle_bisector (C.1) = C.2) ∧ 
       (∃ t : ℝ, C = (1 - t) • A + t • B) ∧
       (∃ s : ℝ, C = (1 - s) • A' + s • B)) 
  (A' : ℝ × ℝ)
  (h5 : A'.2 - 1 = -(A'.1 - 3))  -- Reflection condition
  (h6 : (A'.2 + 1) / 2 = (A'.1 + 3) / 2 + 1)  -- Reflection condition
  : ∀ x y, x - 2*y - 1 = 0 ↔ ∃ t : ℝ, (x, y) = (1 - t) • A + t • C :=
sorry


end angle_bisector_implies_line_equation_l2571_257148


namespace least_sum_of_four_primes_l2571_257151

theorem least_sum_of_four_primes (n : ℕ) : 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), 
    p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₄.Prime ∧
    p₁ > 10 ∧ p₂ > 10 ∧ p₃ > 10 ∧ p₄ > 10 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n = p₁ + p₂ + p₃ + p₄) →
  n ≥ 60 :=
by
  sorry

#check least_sum_of_four_primes

end least_sum_of_four_primes_l2571_257151


namespace probability_not_red_marble_l2571_257194

theorem probability_not_red_marble (total : ℕ) (red green yellow blue : ℕ) 
  (h1 : total = red + green + yellow + blue)
  (h2 : red = 8)
  (h3 : green = 10)
  (h4 : yellow = 12)
  (h5 : blue = 15) :
  (green + yellow + blue : ℚ) / total = 37 / 45 := by
sorry

end probability_not_red_marble_l2571_257194


namespace amy_started_with_101_seeds_l2571_257135

/-- The number of seeds Amy planted in her garden -/
def amy_garden_problem (big_garden_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  big_garden_seeds + small_gardens * seeds_per_small_garden

/-- Theorem stating that Amy started with 101 seeds -/
theorem amy_started_with_101_seeds :
  amy_garden_problem 47 9 6 = 101 := by
  sorry

end amy_started_with_101_seeds_l2571_257135


namespace exists_partition_count_2007_l2571_257104

/-- Given positive integers N and k, count_partitions N k returns the number of ways
    to write N as a sum of three integers a, b, and c, where 1 ≤ a, b, c ≤ k and the order matters. -/
def count_partitions (N k : ℕ+) : ℕ :=
  (Finset.filter (fun (a, b, c) => a + b + c = N ∧ a ≤ k ∧ b ≤ k ∧ c ≤ k)
    (Finset.product (Finset.range k) (Finset.product (Finset.range k) (Finset.range k)))).card

/-- There exist positive integers N and k such that the number of ways to write N
    as a sum of three integers a, b, and c, where 1 ≤ a, b, c ≤ k and the order matters, is 2007. -/
theorem exists_partition_count_2007 : ∃ (N k : ℕ+), count_partitions N k = 2007 := by
  sorry

end exists_partition_count_2007_l2571_257104


namespace expand_expression_l2571_257180

theorem expand_expression (x y : ℝ) : 
  (5 * x^2 - 3/2 * y) * (-4 * x^3 * y^2) = -20 * x^5 * y^2 + 6 * x^3 * y^3 := by
  sorry

end expand_expression_l2571_257180


namespace initial_seashells_count_l2571_257111

/-- The number of seashells Tim found initially -/
def initial_seashells : ℕ := sorry

/-- The number of starfish Tim found -/
def starfish : ℕ := 110

/-- The number of seashells Tim gave to Sara -/
def seashells_given : ℕ := 172

/-- The number of seashells Tim has now -/
def current_seashells : ℕ := 507

/-- Theorem stating that the initial number of seashells is equal to 
    the current number of seashells plus the number of seashells given away -/
theorem initial_seashells_count : 
  initial_seashells = current_seashells + seashells_given := by sorry

end initial_seashells_count_l2571_257111


namespace correct_product_is_5810_l2571_257147

/-- Reverses the digits of a three-digit number -/
def reverse_digits (n : Nat) : Nat :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a number is three-digit -/
def is_three_digit (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem correct_product_is_5810 (a b : Nat) :
  a > 0 ∧ b > 0 ∧ is_three_digit a ∧ (reverse_digits a - 3) * b = 245 →
  a * b = 5810 := by
  sorry

end correct_product_is_5810_l2571_257147


namespace chicken_admission_combinations_l2571_257155

theorem chicken_admission_combinations : Nat.choose 4 2 = 6 := by
  sorry

end chicken_admission_combinations_l2571_257155


namespace student_pairs_l2571_257184

theorem student_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end student_pairs_l2571_257184


namespace a_investment_value_l2571_257128

/-- Represents the investment and profit distribution in a partnership business -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- Theorem stating that given the conditions of the problem, A's investment is 8000 -/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 4000)
  (hc : p.c_investment = 2000)
  (hprofit : p.total_profit = 252000)
  (hshare : p.c_profit_share = 36000)
  : p.a_investment = 8000 := by
  sorry

end a_investment_value_l2571_257128


namespace value_of_y_l2571_257170

theorem value_of_y (y : ℚ) (h : 2/3 - 3/5 = 5/y) : y = 75 := by
  sorry

end value_of_y_l2571_257170


namespace range_of_fraction_l2571_257152

theorem range_of_fraction (x y : ℝ) (h1 : |x + y| ≤ 2) (h2 : |x - y| ≤ 2) :
  ∃ (z : ℝ), z = y / (x - 4) ∧ -1/2 ≤ z ∧ z ≤ 1/2 :=
by sorry

end range_of_fraction_l2571_257152


namespace audiobook_length_l2571_257140

/-- Proves that if a person listens to audiobooks for a certain amount of time each day
    and completes a certain number of audiobooks in a given number of days,
    then each audiobook has a specific length. -/
theorem audiobook_length
  (daily_listening_hours : ℝ)
  (total_days : ℕ)
  (num_audiobooks : ℕ)
  (h1 : daily_listening_hours = 2)
  (h2 : total_days = 90)
  (h3 : num_audiobooks = 6)
  : (daily_listening_hours * total_days) / num_audiobooks = 30 := by
  sorry

end audiobook_length_l2571_257140


namespace urn_problem_l2571_257142

theorem urn_problem (total : ℕ) (red_percent : ℚ) (new_red_percent : ℚ) 
  (h1 : total = 120)
  (h2 : red_percent = 2/5)
  (h3 : new_red_percent = 4/5) :
  ∃ (removed : ℕ), 
    (red_percent * total : ℚ) / (total - removed : ℚ) = new_red_percent ∧ 
    removed = 60 := by
  sorry

end urn_problem_l2571_257142


namespace probability_at_least_one_correct_l2571_257149

theorem probability_at_least_one_correct (n m : ℕ) (h : n > 0 ∧ m > 0) :
  let p := 1 - (1 - 1 / n) ^ m
  n = 6 ∧ m = 6 → p = 31031 / 46656 := by
  sorry

end probability_at_least_one_correct_l2571_257149


namespace percentage_of_filled_holes_l2571_257100

theorem percentage_of_filled_holes (total_holes : ℕ) (unfilled_holes : ℕ) 
  (h1 : total_holes = 8) 
  (h2 : unfilled_holes = 2) 
  (h3 : unfilled_holes < total_holes) : 
  (((total_holes - unfilled_holes) : ℚ) / total_holes) * 100 = 75 := by
  sorry

end percentage_of_filled_holes_l2571_257100


namespace twelve_team_tournament_matches_l2571_257119

/-- Calculates the total number of matches in a round-robin tournament. -/
def total_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 12 teams, the total number of matches is 66. -/
theorem twelve_team_tournament_matches :
  total_matches 12 = 66 := by
  sorry

#eval total_matches 12  -- This will evaluate to 66

end twelve_team_tournament_matches_l2571_257119


namespace skew_symmetric_times_symmetric_is_zero_l2571_257137

/-- Given a skew-symmetric matrix A and a symmetric matrix B, prove that their product is the zero matrix -/
theorem skew_symmetric_times_symmetric_is_zero (a b c : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, c, -b; -c, 0, a; b, -a, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![a^2, a*b, a*c; a*b, b^2, b*c; a*c, b*c, c^2]
  A * B = 0 := by sorry

end skew_symmetric_times_symmetric_is_zero_l2571_257137


namespace system_solutions_l2571_257108

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := y + Real.sqrt (y - 3 * x) + 3 * x = 12
def equation2 (x y : ℝ) : Prop := y^2 + y - 3 * x - 9 * x^2 = 144

-- State the theorem
theorem system_solutions :
  (∀ x y : ℝ, equation1 x y ∧ equation2 x y ↔ (x = -24 ∧ y = 72) ∨ (x = -4/3 ∧ y = 12)) :=
by sorry


end system_solutions_l2571_257108


namespace tony_age_is_twelve_l2571_257121

/-- Represents Tony's work and payment information -/
structure TonyWork where
  hoursPerDay : ℕ
  payPerHourPerYear : ℚ
  workDays : ℕ
  totalEarnings : ℚ

/-- Calculates Tony's age based on his work information -/
def calculateAge (work : TonyWork) : ℕ :=
  sorry

/-- Theorem stating that Tony's age at the end of the five-month period was 12 -/
theorem tony_age_is_twelve (work : TonyWork) 
  (h1 : work.hoursPerDay = 2)
  (h2 : work.payPerHourPerYear = 1)
  (h3 : work.workDays = 60)
  (h4 : work.totalEarnings = 1140) :
  calculateAge work = 12 :=
sorry

end tony_age_is_twelve_l2571_257121


namespace money_difference_l2571_257157

def derek_initial : ℕ := 40
def derek_expense1 : ℕ := 14
def derek_expense2 : ℕ := 11
def derek_expense3 : ℕ := 5
def dave_initial : ℕ := 50
def dave_expense : ℕ := 7

theorem money_difference :
  dave_initial - dave_expense - (derek_initial - derek_expense1 - derek_expense2 - derek_expense3) = 33 := by
  sorry

end money_difference_l2571_257157


namespace sum_of_terms_l2571_257131

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  roots_property : a 2 + a 16 = 6 ∧ a 2 * a 16 = -1

/-- The sum of specific terms in the arithmetic sequence equals 15 -/
theorem sum_of_terms (seq : ArithmeticSequence) : 
  seq.a 5 + seq.a 6 + seq.a 9 + seq.a 12 + seq.a 13 = 15 := by
  sorry

end sum_of_terms_l2571_257131


namespace a_range_theorem_l2571_257164

/-- Proposition p: The inequality x^2 + 2ax + 4 > 0 holds true for all x ∈ ℝ -/
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

/-- Proposition q: The function f(x) = (3-2a)^x is increasing -/
def prop_q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3-2*a)^x < (3-2*a)^y

/-- The range of values for a satisfying the given conditions -/
def a_range (a : ℝ) : Prop := a ≤ -2 ∨ (1 ≤ a ∧ a < 2)

theorem a_range_theorem (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) → a_range a :=
by sorry

end a_range_theorem_l2571_257164


namespace three_piece_suit_cost_l2571_257129

/-- The cost of a jacket in pounds -/
def jacket_cost : ℝ := sorry

/-- The cost of a pair of trousers in pounds -/
def trousers_cost : ℝ := sorry

/-- The cost of a waistcoat in pounds -/
def waistcoat_cost : ℝ := sorry

/-- Two jackets and three pairs of trousers cost £380 -/
axiom two_jackets_three_trousers : 2 * jacket_cost + 3 * trousers_cost = 380

/-- A pair of trousers costs the same as two waistcoats -/
axiom trousers_equals_two_waistcoats : trousers_cost = 2 * waistcoat_cost

/-- The cost of a three-piece suit is £190 -/
theorem three_piece_suit_cost : jacket_cost + trousers_cost + waistcoat_cost = 190 := by
  sorry

end three_piece_suit_cost_l2571_257129


namespace smallest_divisible_by_million_l2571_257159

/-- Represents a geometric sequence with first term a and common ratio r -/
def GeometricSequence (a : ℚ) (r : ℚ) : ℕ → ℚ := λ n => a * r^(n - 1)

/-- The nth term of the specific geometric sequence in the problem -/
def SpecificSequence : ℕ → ℚ := GeometricSequence (1/2) 60

/-- Predicate to check if a rational number is divisible by one million -/
def DivisibleByMillion (q : ℚ) : Prop := ∃ (k : ℤ), q = (k : ℚ) * 1000000

theorem smallest_divisible_by_million :
  (∀ n < 7, ¬ DivisibleByMillion (SpecificSequence n)) ∧
  DivisibleByMillion (SpecificSequence 7) :=
sorry

end smallest_divisible_by_million_l2571_257159


namespace miles_left_to_run_l2571_257165

/-- Macy's weekly running goal in miles -/
def weekly_goal : ℕ := 24

/-- Macy's daily running distance in miles -/
def daily_distance : ℕ := 3

/-- Number of days Macy has run -/
def days_run : ℕ := 6

/-- Theorem stating the number of miles left for Macy to run after 6 days -/
theorem miles_left_to_run : weekly_goal - (daily_distance * days_run) = 6 := by
  sorry

end miles_left_to_run_l2571_257165


namespace solution_set_of_inequality_l2571_257154

-- Define the quadratic function
def f (x : ℝ) : ℝ := x * (1 - 3 * x)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x > 0} = Set.Ioo 0 (1/3) :=
sorry

end solution_set_of_inequality_l2571_257154


namespace largest_multiple_of_13_less_than_neg_124_l2571_257192

theorem largest_multiple_of_13_less_than_neg_124 :
  ∀ n : ℤ, n * 13 < -124 → n * 13 ≤ -130 :=
by
  sorry

end largest_multiple_of_13_less_than_neg_124_l2571_257192


namespace circle_C_equation_l2571_257146

-- Define the circles
def circle_C (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = r^2}
def circle_other : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 3*p.1 = 0}

-- Define the line passing through (5, -2)
def common_chord_line (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + 2*p.2 - 5 + r^2 = 0}

-- Theorem statement
theorem circle_C_equation :
  ∃ (r : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ circle_C r ∩ circle_other → (x, y) ∈ common_chord_line r) ∧
    (5, -2) ∈ common_chord_line r →
    r = 2 :=
sorry

end circle_C_equation_l2571_257146


namespace biquadratic_equation_with_given_root_l2571_257127

theorem biquadratic_equation_with_given_root (x : ℝ) :
  (2 + Real.sqrt 3 : ℝ) ^ 4 - 14 * (2 + Real.sqrt 3 : ℝ) ^ 2 + 1 = 0 :=
by sorry

end biquadratic_equation_with_given_root_l2571_257127


namespace seven_balls_four_boxes_l2571_257123

/-- The number of ways to partition n indistinguishable objects into at most k parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 11 ways to partition 7 indistinguishable objects into at most 4 parts -/
theorem seven_balls_four_boxes : partition_count 7 4 = 11 := by sorry

end seven_balls_four_boxes_l2571_257123


namespace product_of_real_parts_complex_equation_l2571_257144

theorem product_of_real_parts_complex_equation : 
  let f : ℂ → ℂ := fun x ↦ x^2 - 4*x + 2 - 2*I
  let solutions := {x : ℂ | f x = 0}
  ∃ x₁ x₂ : ℂ, x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
  (x₁.re * x₂.re = 3 - Real.sqrt 2) := by
  sorry

end product_of_real_parts_complex_equation_l2571_257144


namespace max_area_special_quadrilateral_l2571_257182

/-- A quadrilateral with the property that the product of any two adjacent sides is 1 -/
structure SpecialQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  ab_eq_one : a * b = 1
  bc_eq_one : b * c = 1
  cd_eq_one : c * d = 1
  da_eq_one : d * a = 1

/-- The area of a quadrilateral -/
def area (q : SpecialQuadrilateral) : ℝ := sorry

/-- The maximum area of a SpecialQuadrilateral is 1 -/
theorem max_area_special_quadrilateral :
  ∀ q : SpecialQuadrilateral, area q ≤ 1 ∧ ∃ q' : SpecialQuadrilateral, area q' = 1 := by
  sorry

end max_area_special_quadrilateral_l2571_257182


namespace equal_pairs_l2571_257174

theorem equal_pairs : 
  (-2^4 ≠ (-2)^4) ∧ 
  (5^3 ≠ 3^5) ∧ 
  (-(-3) ≠ -|(-3)|) ∧ 
  ((-1)^2 = (-1)^2008) := by
  sorry

end equal_pairs_l2571_257174


namespace closest_point_is_A_l2571_257185

-- Define the points as real numbers
variable (A B C D E : ℝ)

-- Define the conditions
axiom A_range : 0 < A ∧ A < 1
axiom B_range : 0 < B ∧ B < 1
axiom C_range : 0 < C ∧ C < 1
axiom D_range : 0 < D ∧ D < 1
axiom E_range : 1 < E ∧ E < 2

-- Define the order of points
axiom point_order : A < B ∧ B < C ∧ C < D

-- Define a function to calculate the distance between two real numbers
def distance (x y : ℝ) : ℝ := |x - y|

-- State the theorem
theorem closest_point_is_A :
  distance (B * C) A < distance (B * C) B ∧
  distance (B * C) A < distance (B * C) C ∧
  distance (B * C) A < distance (B * C) D ∧
  distance (B * C) A < distance (B * C) E :=
sorry

end closest_point_is_A_l2571_257185


namespace lcm_of_ratio_and_hcf_l2571_257188

theorem lcm_of_ratio_and_hcf (a b : ℕ) : 
  a ≠ 0 → b ≠ 0 → a * 4 = b * 3 → Nat.gcd a b = 8 → Nat.lcm a b = 96 := by
  sorry

end lcm_of_ratio_and_hcf_l2571_257188


namespace solve_for_A_l2571_257110

theorem solve_for_A (A : ℤ) (h : A - 10 = 15) : A = 25 := by
  sorry

end solve_for_A_l2571_257110


namespace tan_alpha_for_point_l2571_257122

/-- If the terminal side of angle α passes through the point (-4, -3), then tan α = 3/4 -/
theorem tan_alpha_for_point (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -4 ∧ t * Real.sin α = -3) → 
  Real.tan α = 3/4 := by
  sorry

end tan_alpha_for_point_l2571_257122


namespace functional_equation_solution_l2571_257133

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y t : ℝ, f (x + t + f y) = f (f x) + f t + y) →
  (∀ x : ℝ, f x = x) := by
  sorry

end functional_equation_solution_l2571_257133


namespace birth_death_rate_interval_birth_death_rate_problem_l2571_257117

theorem birth_death_rate_interval (birth_rate : ℕ) (death_rate : ℕ) (daily_increase : ℕ) : ℕ :=
  let net_rate := birth_rate - death_rate
  let intervals_per_day := daily_increase / net_rate
  let minutes_per_day := 24 * 60
  minutes_per_day / intervals_per_day

theorem birth_death_rate_problem :
  birth_death_rate_interval 10 2 345600 = 48 := by
  sorry

end birth_death_rate_interval_birth_death_rate_problem_l2571_257117


namespace parabola_equation_l2571_257183

/-- A parabola with vertex at the origin, axis of symmetry along the y-axis,
    and distance between vertex and focus equal to 6 -/
structure Parabola where
  vertex : ℝ × ℝ
  axis_of_symmetry : ℝ → ℝ
  focus_distance : ℝ
  h_vertex : vertex = (0, 0)
  h_axis : axis_of_symmetry = fun y => 0
  h_focus : focus_distance = 6

/-- The standard equation of the parabola -/
def standard_equation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (x^2 = 24*y ∨ x^2 = -24*y) ↔ (x, y) ∈ {(x, y) | p.axis_of_symmetry x = y}

/-- Theorem stating that the standard equation holds for the given parabola -/
theorem parabola_equation (p : Parabola) : standard_equation p := by
  sorry

end parabola_equation_l2571_257183


namespace quadratic_roots_and_k_value_l2571_257141

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := x^2 + (2*k + 1)*x + k^2 + 1

-- Theorem statement
theorem quadratic_roots_and_k_value (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0) ↔ k > 3/4 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧ x₁ * x₂ = 5) → k = 2 :=
by sorry

end quadratic_roots_and_k_value_l2571_257141


namespace z_minus_two_purely_imaginary_l2571_257120

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem z_minus_two_purely_imaginary (z : ℂ) (h : z = 2 - I) : 
  is_purely_imaginary (z - 2) := by
  sorry

end z_minus_two_purely_imaginary_l2571_257120


namespace raised_beds_planks_l2571_257186

/-- Calculates the number of planks needed for raised beds -/
def planks_needed (num_beds : ℕ) (bed_height : ℕ) (bed_width : ℕ) (bed_length : ℕ) (plank_width : ℕ) : ℕ :=
  let long_sides := 2 * bed_height
  let short_sides := 2 * bed_height
  let planks_per_bed := long_sides + short_sides
  num_beds * planks_per_bed

/-- Proves that 60 planks are needed for 10 raised beds with given dimensions -/
theorem raised_beds_planks :
  planks_needed 10 2 2 8 1 = 60 := by
  sorry

end raised_beds_planks_l2571_257186


namespace associate_professor_pencils_l2571_257168

theorem associate_professor_pencils :
  ∀ (A B P : ℕ),
    A + B = 7 →
    A + 2 * B = 11 →
    P * A + B = 10 →
    P = 2 := by
  sorry

end associate_professor_pencils_l2571_257168


namespace triangle_angle_measure_l2571_257173

theorem triangle_angle_measure (a b c : ℝ) (A : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  S > 0 →
  (4 * Real.sqrt 3 / 3) * S = b^2 + c^2 - a^2 →
  S = (1/2) * b * c * Real.sin A →
  0 < A → A < π →
  A = π / 3 := by
sorry

end triangle_angle_measure_l2571_257173


namespace chess_club_officers_l2571_257125

def choose_officers (n : ℕ) (k : ℕ) (special_pair : ℕ) : ℕ :=
  (n - special_pair).choose k * (n - special_pair - k + 1).choose (k - 1) +
  special_pair * (special_pair - 1) * (n - special_pair)

theorem chess_club_officers :
  choose_officers 24 3 2 = 9372 :=
sorry

end chess_club_officers_l2571_257125


namespace income_calculation_l2571_257167

theorem income_calculation (income expenditure savings : ℕ) : 
  income = 5 * expenditure / 4 →
  income - expenditure = savings →
  savings = 3600 →
  income = 18000 := by
sorry

end income_calculation_l2571_257167


namespace inequality_system_solution_range_l2571_257197

theorem inequality_system_solution_range (m : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℤ, (x + 5 > 0 ∧ x - m ≤ 1) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) ↔
  (-3 ≤ m ∧ m < -2) :=
by sorry

end inequality_system_solution_range_l2571_257197


namespace number_of_lilies_l2571_257105

theorem number_of_lilies (total_flowers other_flowers : ℕ) 
  (h1 : other_flowers = 120)
  (h2 : total_flowers = 160) :
  total_flowers - other_flowers = 40 := by
sorry

end number_of_lilies_l2571_257105


namespace factor_x_squared_minus_64_l2571_257166

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end factor_x_squared_minus_64_l2571_257166


namespace opposite_sides_iff_a_range_l2571_257107

/-- A point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in the 2D plane of the form ax + by + c = 0 -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Check if a point is on one side of a line -/
def onOneSide (p : Point) (l : Line) : ℝ :=
  l.a * p.x + l.b * p.y + l.c

/-- The main theorem -/
theorem opposite_sides_iff_a_range (a : ℝ) :
  let P : Point := ⟨1, a⟩
  let Q : Point := ⟨a, -2⟩
  let l : Line := ⟨1, -2, 1⟩
  (onOneSide P l) * (onOneSide Q l) < 0 ↔ a < -5 ∨ a > 1 :=
sorry

end opposite_sides_iff_a_range_l2571_257107


namespace square_of_negative_two_a_squared_l2571_257112

theorem square_of_negative_two_a_squared (a : ℝ) : (-2 * a^2)^2 = 4 * a^4 := by
  sorry

end square_of_negative_two_a_squared_l2571_257112


namespace max_attendance_l2571_257145

-- Define the days of the week
inductive Day
| monday
| tuesday
| wednesday
| thursday
| friday

-- Define the team members
inductive Member
| alice
| bob
| charlie
| diana
| edward

-- Define a function that returns whether a member is available on a given day
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.alice, Day.monday => false
  | Member.alice, Day.thursday => false
  | Member.bob, Day.tuesday => false
  | Member.bob, Day.friday => false
  | Member.charlie, Day.monday => false
  | Member.charlie, Day.tuesday => false
  | Member.charlie, Day.thursday => false
  | Member.charlie, Day.friday => false
  | Member.diana, Day.wednesday => false
  | Member.diana, Day.thursday => false
  | Member.edward, Day.wednesday => false
  | _, _ => true

-- Define a function that counts the number of available members on a given day
def countAvailable (d : Day) : Nat :=
  List.length (List.filter (λ m => isAvailable m d) [Member.alice, Member.bob, Member.charlie, Member.diana, Member.edward])

-- State the theorem
theorem max_attendance :
  (∀ d : Day, countAvailable d ≤ 3) ∧
  (countAvailable Day.monday = 3) ∧
  (countAvailable Day.wednesday = 3) ∧
  (countAvailable Day.friday = 3) :=
sorry

end max_attendance_l2571_257145


namespace crosswalk_stripe_distance_l2571_257126

theorem crosswalk_stripe_distance 
  (curb_distance : ℝ) 
  (curb_length : ℝ) 
  (stripe_length : ℝ) 
  (h1 : curb_distance = 30) 
  (h2 : curb_length = 10) 
  (h3 : stripe_length = 60) : 
  ∃ (stripe_distance : ℝ), 
    stripe_distance * stripe_length = curb_length * curb_distance ∧ 
    stripe_distance = 5 := by
  sorry

end crosswalk_stripe_distance_l2571_257126


namespace min_value_of_f_l2571_257179

noncomputable def f (x : ℝ) := (Real.exp x - 1)^2 + (Real.exp 1 - x - 1)^2

theorem min_value_of_f :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end min_value_of_f_l2571_257179


namespace x_greater_y_greater_z_l2571_257187

theorem x_greater_y_greater_z (α b x y z : Real) 
  (h_α : α ∈ Set.Ioo (π / 4) (π / 2))
  (h_b : b ∈ Set.Ioo 0 1)
  (h_x : Real.log x = (Real.log (Real.sin α))^2 / Real.log b)
  (h_y : Real.log y = (Real.log (Real.cos α))^2 / Real.log b)
  (h_z : Real.log z = (Real.log (Real.sin α * Real.cos α))^2 / Real.log b) :
  x > y ∧ y > z := by
  sorry

end x_greater_y_greater_z_l2571_257187


namespace siblings_total_weight_siblings_total_weight_is_88_l2571_257103

/-- The total weight of two siblings, where one weighs 50 kg and the other weighs 12 kg less. -/
theorem siblings_total_weight : ℝ :=
  let antonio_weight : ℝ := 50
  let sister_weight : ℝ := antonio_weight - 12
  antonio_weight + sister_weight
  
/-- Prove that the total weight of the siblings is 88 kg. -/
theorem siblings_total_weight_is_88 : siblings_total_weight = 88 := by
  sorry

end siblings_total_weight_siblings_total_weight_is_88_l2571_257103


namespace frog_arrangement_count_l2571_257113

/-- Represents the number of ways to arrange frogs with given constraints -/
def frog_arrangements (n : ℕ) (green red : ℕ) (blue yellow : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 7 2 3 1 1 = 120 := by
  sorry

end frog_arrangement_count_l2571_257113


namespace speed_against_stream_calculation_mans_speed_is_six_l2571_257175

/-- Calculate the speed against the stream given the rate in still water and speed with the stream -/
def speed_against_stream (rate_still : ℝ) (speed_with : ℝ) : ℝ :=
  |rate_still - 2 * (speed_with - rate_still)|

/-- Theorem: Given a man's rowing rate in still water and his speed with the stream,
    his speed against the stream is the absolute difference between his rate in still water
    and twice the difference of his speed with the stream and his rate in still water -/
theorem speed_against_stream_calculation (rate_still : ℝ) (speed_with : ℝ) :
  speed_against_stream rate_still speed_with = 
  |rate_still - 2 * (speed_with - rate_still)| := by
  sorry

/-- The man's speed against the stream given his rate in still water and speed with the stream -/
def mans_speed_against_stream : ℝ :=
  speed_against_stream 5 16

/-- Theorem: The man's speed against the stream is 6 km/h -/
theorem mans_speed_is_six :
  mans_speed_against_stream = 6 := by
  sorry

end speed_against_stream_calculation_mans_speed_is_six_l2571_257175


namespace f_is_square_iff_n_eq_one_l2571_257138

/-- The number of non-empty subsets of {1, ..., n} with gcd 1 -/
def f (n : ℕ+) : ℕ := sorry

/-- f(n) is a perfect square iff n = 1 -/
theorem f_is_square_iff_n_eq_one (n : ℕ+) : 
  ∃ m : ℕ, f n = m ^ 2 ↔ n = 1 := by sorry

end f_is_square_iff_n_eq_one_l2571_257138


namespace trapezoid_perimeter_is_129_l2571_257193

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithCircle where
  -- The lengths of the parallel sides
  shorterBase : ℝ
  longerBase : ℝ
  -- The length of the leg (equal for both legs in an isosceles trapezoid)
  leg : ℝ
  -- The height of the trapezoid
  height : ℝ
  -- The radius of the inscribed circle
  radius : ℝ
  -- Conditions
  height_positive : height > 0
  radius_positive : radius > 0
  longer_than_shorter : longerBase > shorterBase
  circle_touches_base : shorterBase = 2 * radius
  circle_touches_leg : leg^2 = (longerBase - shorterBase)^2 / 4 + height^2

/-- The perimeter of the trapezoid -/
def perimeter (t : IsoscelesTrapezoidWithCircle) : ℝ :=
  t.shorterBase + t.longerBase + 2 * t.leg

theorem trapezoid_perimeter_is_129
  (t : IsoscelesTrapezoidWithCircle)
  (h₁ : t.height = 36)
  (h₂ : t.radius = 11) :
  perimeter t = 129 :=
sorry

end trapezoid_perimeter_is_129_l2571_257193


namespace log_3897_between_consecutive_integers_l2571_257172

theorem log_3897_between_consecutive_integers : 
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 3897 / Real.log 10 ∧ Real.log 3897 / Real.log 10 < b ∧ a + b = 7 := by
  sorry

end log_3897_between_consecutive_integers_l2571_257172


namespace inequality_proof_minimum_value_proof_l2571_257163

-- Define the variables and conditions
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2)

-- Part I: Prove the inequality
theorem inequality_proof : Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) ≤ 4 := by
  sorry

-- Part II: Prove the minimum value
theorem minimum_value_proof : ∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 1 / (a + 1) + 1 / (b + 1) ≥ 1 := by
  sorry

end inequality_proof_minimum_value_proof_l2571_257163


namespace isosceles_triangle_base_length_l2571_257134

theorem isosceles_triangle_base_length 
  (congruent_side : ℝ) 
  (perimeter : ℝ) 
  (h1 : congruent_side = 6) 
  (h2 : perimeter = 20) :
  perimeter - 2 * congruent_side = 8 :=
by sorry

end isosceles_triangle_base_length_l2571_257134


namespace cube_side_ratio_l2571_257130

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 16 → a / b = 4 := by
sorry

end cube_side_ratio_l2571_257130


namespace sqrt_equation_solution_l2571_257116

theorem sqrt_equation_solution :
  ∀ x : ℝ, (Real.sqrt x + Real.sqrt (x + 3) = 12) → x = 19881 / 576 := by
  sorry

end sqrt_equation_solution_l2571_257116


namespace floor_negative_seven_fourths_cubed_l2571_257162

theorem floor_negative_seven_fourths_cubed : ⌊(-7/4)^3⌋ = -6 := by
  sorry

end floor_negative_seven_fourths_cubed_l2571_257162


namespace rectangle_area_l2571_257118

/-- Given a rectangle with perimeter 100 meters and length three times its width, 
    its area is 468.75 square meters. -/
theorem rectangle_area (w : ℝ) (l : ℝ) : 
  (2 * l + 2 * w = 100) → 
  (l = 3 * w) → 
  (l * w = 468.75) :=
by
  sorry

end rectangle_area_l2571_257118


namespace expression_classification_l2571_257181

/-- Represents an algebraic expression -/
inductive AlgebraicExpr
  | Constant (n : ℚ)
  | Variable (name : String)
  | Product (coef : ℚ) (terms : List (String × ℕ))

/-- Calculates the degree of an algebraic expression -/
def degree (expr : AlgebraicExpr) : ℕ :=
  match expr with
  | AlgebraicExpr.Constant _ => 0
  | AlgebraicExpr.Variable _ => 1
  | AlgebraicExpr.Product _ terms => terms.foldl (fun acc (_, power) => acc + power) 0

/-- Checks if an expression contains variables -/
def hasVariables (expr : AlgebraicExpr) : Bool :=
  match expr with
  | AlgebraicExpr.Constant _ => false
  | AlgebraicExpr.Variable _ => true
  | AlgebraicExpr.Product _ terms => terms.length > 0

def expressions : List AlgebraicExpr := [
  AlgebraicExpr.Product (-2) [("a", 1)],
  AlgebraicExpr.Product 3 [("a", 1), ("b", 2)],
  AlgebraicExpr.Constant (2/3),
  AlgebraicExpr.Product 3 [("a", 2), ("b", 1)],
  AlgebraicExpr.Product (-3) [("a", 3)],
  AlgebraicExpr.Constant 25,
  AlgebraicExpr.Product (-(3/4)) [("b", 1)]
]

theorem expression_classification :
  (expressions.filter hasVariables).length = 5 ∧
  (expressions.filter (fun e => ¬hasVariables e)).length = 2 ∧
  (expressions.filter (fun e => degree e = 0)).length = 2 ∧
  (expressions.filter (fun e => degree e = 1)).length = 2 ∧
  (expressions.filter (fun e => degree e = 3)).length = 3 :=
by sorry

end expression_classification_l2571_257181


namespace set_equalities_l2571_257178

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 < x ∧ x < 1}
def B : Set ℝ := {x | x ≤ -1}
def C : Set ℝ := {x | -2 < x ∧ x ≤ 2}

-- Theorem stating the equalities
theorem set_equalities :
  (A = A ∩ (B ∪ C)) ∧
  (A = A ∪ (B ∩ C)) ∧
  (A = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end set_equalities_l2571_257178


namespace remainder_times_seven_l2571_257114

theorem remainder_times_seven (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = quotient * divisor + remainder →
  dividend = 972345 →
  divisor = 145 →
  remainder < divisor →
  remainder * 7 = 840 := by
sorry

end remainder_times_seven_l2571_257114
