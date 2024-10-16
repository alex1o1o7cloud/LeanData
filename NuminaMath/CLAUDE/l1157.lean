import Mathlib

namespace NUMINAMATH_CALUDE_max_pairs_after_loss_l1157_115769

/-- Given a collection of shoes and a number of lost individual shoes,
    calculate the maximum number of matching pairs remaining. -/
def maxRemainingPairs (totalPairs : ℕ) (lostShoes : ℕ) : ℕ :=
  totalPairs - (lostShoes / 2) - (lostShoes % 2)

/-- Theorem: Given 150 pairs of shoes and a loss of 37 individual shoes,
    the maximum number of matching pairs remaining is 131. -/
theorem max_pairs_after_loss :
  maxRemainingPairs 150 37 = 131 := by
  sorry

#eval maxRemainingPairs 150 37

end NUMINAMATH_CALUDE_max_pairs_after_loss_l1157_115769


namespace NUMINAMATH_CALUDE_speed_ratio_after_meeting_l1157_115748

/-- Represents a car with a speed -/
structure Car where
  speed : ℝ

/-- Represents the scenario of two cars meeting -/
structure CarMeeting where
  carA : Car
  carB : Car
  totalDistance : ℝ
  timeToMeet : ℝ
  timeAAfterMeet : ℝ
  timeBAfterMeet : ℝ

/-- The theorem stating the ratio of speeds given the conditions -/
theorem speed_ratio_after_meeting (m : CarMeeting) 
  (h1 : m.timeAAfterMeet = 4)
  (h2 : m.timeBAfterMeet = 1)
  (h3 : m.totalDistance = m.carA.speed * m.timeToMeet + m.carB.speed * m.timeToMeet)
  (h4 : m.carA.speed * m.timeAAfterMeet = m.totalDistance - m.carA.speed * m.timeToMeet)
  (h5 : m.carB.speed * m.timeBAfterMeet = m.totalDistance - m.carB.speed * m.timeToMeet) :
  m.carA.speed / m.carB.speed = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_after_meeting_l1157_115748


namespace NUMINAMATH_CALUDE_probability_at_least_one_fuse_blows_l1157_115754

/-- The probability that at least one fuse blows in a circuit with two independent fuses -/
theorem probability_at_least_one_fuse_blows 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h_prob_A : prob_A = 0.85) 
  (h_prob_B : prob_B = 0.74) 
  (h_independent : True) -- We don't need to express independence explicitly in this theorem
  : 1 - (1 - prob_A) * (1 - prob_B) = 0.961 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_fuse_blows_l1157_115754


namespace NUMINAMATH_CALUDE_original_deck_size_l1157_115774

/-- Represents a deck of cards with blue and yellow cards -/
structure Deck where
  blue : ℕ
  yellow : ℕ

/-- The probability of drawing a blue card from the deck -/
def blueProbability (d : Deck) : ℚ :=
  d.blue / (d.blue + d.yellow)

/-- Adds yellow cards to the deck -/
def addYellow (d : Deck) (n : ℕ) : Deck :=
  { blue := d.blue, yellow := d.yellow + n }

theorem original_deck_size (d : Deck) :
  blueProbability d = 2/5 ∧ 
  blueProbability (addYellow d 6) = 5/14 →
  d.blue + d.yellow = 50 := by
  sorry

end NUMINAMATH_CALUDE_original_deck_size_l1157_115774


namespace NUMINAMATH_CALUDE_quadratic_sum_l1157_115715

/-- A quadratic function with vertex at (-2, 5) and specific points -/
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  (∀ x, g d e f x = d * (x + 2)^2 + 5) →  -- vertex form
  g d e f 0 = -1 →                       -- g(0) = -1
  g d e f 1 = -4 →                       -- g(1) = -4
  d + e + 3 * f = 14 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1157_115715


namespace NUMINAMATH_CALUDE_value_of_x_l1157_115706

theorem value_of_x (x y z : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 10) * z → 
  z = 100 → 
  x = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l1157_115706


namespace NUMINAMATH_CALUDE_firm_partners_count_l1157_115773

theorem firm_partners_count (partners associates : ℕ) : 
  partners / associates = 2 / 63 →
  partners / (associates + 45) = 1 / 34 →
  partners = 18 := by
sorry

end NUMINAMATH_CALUDE_firm_partners_count_l1157_115773


namespace NUMINAMATH_CALUDE_solve_boat_speed_l1157_115758

def boat_speed_problem (stream_speed : ℝ) (distance : ℝ) (total_time : ℝ) : Prop :=
  ∃ (boat_speed : ℝ),
    boat_speed > stream_speed ∧
    (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = total_time ∧
    boat_speed = 9

theorem solve_boat_speed : boat_speed_problem 1.5 105 24 := by
  sorry

end NUMINAMATH_CALUDE_solve_boat_speed_l1157_115758


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1157_115752

def expression (x : ℝ) : ℝ :=
  5 * (x^2 - 2*x^3 + x) + 2 * (x + 3*x^3 - 4*x^2 + 2*x^5 + 2*x^3) - 7 * (2 + x - 5*x^3 - 2*x^2)

theorem coefficient_of_x_cubed (x : ℝ) :
  ∃ (a b c d : ℝ), expression x = a*x^5 + b*x^4 + 35*x^3 + c*x^2 + d*x + (5*1 - 7*2) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1157_115752


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1157_115713

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 3| > 1} = Set.Iio 1 ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1157_115713


namespace NUMINAMATH_CALUDE_smug_twc_minimum_bouts_l1157_115764

theorem smug_twc_minimum_bouts (n : Nat) (h : n = 2008) :
  let total_edges := n * (n - 1) / 2
  let max_complement_edges := n^2 / 4
  total_edges - max_complement_edges = 999000 := by
  sorry

end NUMINAMATH_CALUDE_smug_twc_minimum_bouts_l1157_115764


namespace NUMINAMATH_CALUDE_fraction_simplification_l1157_115747

theorem fraction_simplification :
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1157_115747


namespace NUMINAMATH_CALUDE_circle_equation_l1157_115718

/-- Prove that the equation (x-1)^2 + (y-1)^2 = 2 represents the circle with center (1,1) passing through the point (2,2). -/
theorem circle_equation (x y : ℝ) : 
  (∀ (x₀ y₀ : ℝ), (x₀ - 1)^2 + (y₀ - 1)^2 = 2 ↔ 
    ((x₀ - 1)^2 + (y₀ - 1)^2 = (x - 1)^2 + (y - 1)^2 ∧ (x - 1)^2 + (y - 1)^2 = 1)) ∧
  (2 - 1)^2 + (2 - 1)^2 = 2 := by
sorry


end NUMINAMATH_CALUDE_circle_equation_l1157_115718


namespace NUMINAMATH_CALUDE_count_valid_m_l1157_115792

theorem count_valid_m : ∃! (s : Finset ℕ), 
  (∀ m ∈ s, m > 0 ∧ (2520 : ℤ) % (m^2 - 2) = 0) ∧
  (∀ m : ℕ, m > 0 ∧ (2520 : ℤ) % (m^2 - 2) = 0 → m ∈ s) ∧
  s.card = 5 := by sorry

end NUMINAMATH_CALUDE_count_valid_m_l1157_115792


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1157_115722

/-- 
A rectangular plot has an area that is 20 times its breadth,
and its length is 10 meters more than its breadth.
This theorem proves that the breadth of such a plot is 10 meters.
-/
theorem rectangular_plot_breadth : 
  ∀ (breadth length area : ℝ),
  area = 20 * breadth →
  length = breadth + 10 →
  area = length * breadth →
  breadth = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1157_115722


namespace NUMINAMATH_CALUDE_student_mistake_difference_l1157_115708

theorem student_mistake_difference : (5/6 : ℚ) * 576 - (5/16 : ℚ) * 576 = 300 := by
  sorry

end NUMINAMATH_CALUDE_student_mistake_difference_l1157_115708


namespace NUMINAMATH_CALUDE_sum_range_l1157_115763

theorem sum_range (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let S := a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)
  1 ≤ S ∧ S ≤ 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_range_l1157_115763


namespace NUMINAMATH_CALUDE_inverse_of_A_l1157_115799

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -1; 2, 3]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![3/14, 1/14; -1/7, 2/7]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l1157_115799


namespace NUMINAMATH_CALUDE_total_paid_is_230_l1157_115778

/-- The cost of an item before tax -/
def cost : ℝ := 200

/-- The tax rate as a decimal -/
def tax_rate : ℝ := 0.15

/-- The total amount paid after tax -/
def total_paid : ℝ := cost + (cost * tax_rate)

/-- Theorem stating that the total amount paid after tax is $230 -/
theorem total_paid_is_230 : total_paid = 230 := by
  sorry

end NUMINAMATH_CALUDE_total_paid_is_230_l1157_115778


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l1157_115771

theorem set_equality_implies_a_equals_three (a : ℝ) : 
  ({0, 1, a^2} : Set ℝ) = ({1, 0, 2*a + 3} : Set ℝ) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l1157_115771


namespace NUMINAMATH_CALUDE_diophantine_approximation_l1157_115790

theorem diophantine_approximation (x : ℝ) (h_irr : Irrational x) (h_pos : x > 0) :
  ∀ n : ℕ, ∃ p q : ℤ, q > n ∧ q > 0 ∧ |x - (p : ℝ) / q| ≤ 1 / q^2 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_approximation_l1157_115790


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1157_115738

theorem possible_values_of_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2014)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2014) :
  ∃! s : Finset ℕ+, s.card = 502 ∧ ∀ x : ℕ+, x ∈ s ↔ 
    ∃ b' c' d' : ℕ+, 
      x > b' ∧ b' > c' ∧ c' > d' ∧
      x + b' + c' + d' = 2014 ∧
      x^2 - b'^2 + c'^2 - d'^2 = 2014 :=
by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1157_115738


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1157_115753

-- Define the functional equation
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y * f (x - y) = x^2 * f y - y^2 * f x

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, satisfies_equation f →
    (∀ x : ℝ, f x = x ∨ f x = -x ∨ f x = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l1157_115753


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l1157_115762

theorem wire_cutting_problem (initial_length second_length num_pieces : ℕ) 
  (h1 : initial_length = 1000)
  (h2 : second_length = 1050)
  (h3 : num_pieces = 14)
  (h4 : ∃ (piece_length : ℕ), 
    piece_length * num_pieces = initial_length ∧ 
    piece_length * num_pieces = second_length) :
  ∃ (piece_length : ℕ), piece_length = 71 ∧ 
    piece_length * num_pieces = initial_length ∧ 
    piece_length * num_pieces = second_length :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l1157_115762


namespace NUMINAMATH_CALUDE_river_round_trip_time_l1157_115701

/-- Calculates the total time for a round trip on a river -/
theorem river_round_trip_time 
  (river_current : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_current = 8) 
  (h2 : boat_speed = 20) 
  (h3 : distance = 84) : 
  (distance / (boat_speed - river_current)) + (distance / (boat_speed + river_current)) = 10 := by
  sorry

#check river_round_trip_time

end NUMINAMATH_CALUDE_river_round_trip_time_l1157_115701


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l1157_115736

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 4*I) * (a + b*I) = y*I) : a/b = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l1157_115736


namespace NUMINAMATH_CALUDE_smallest_b_is_85_l1157_115727

/-- A pair of integers that multiply to give 1764 -/
def ValidPair : Type := { p : ℤ × ℤ // p.1 * p.2 = 1764 }

/-- Predicate to check if a number is a perfect square -/
def IsPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

/-- The sum of a valid pair -/
def PairSum (p : ValidPair) : ℤ := p.val.1 + p.val.2

theorem smallest_b_is_85 :
  (∃ (b : ℕ), 
    (∃ (p : ValidPair), PairSum p = b) ∧ 
    (∃ (p : ValidPair), IsPerfectSquare p.val.1 ∨ IsPerfectSquare p.val.2) ∧
    (∀ (b' : ℕ), b' < b → 
      (∀ (p : ValidPair), PairSum p ≠ b' ∨ 
        (¬ IsPerfectSquare p.val.1 ∧ ¬ IsPerfectSquare p.val.2)))) ∧
  (∀ (b : ℕ), 
    ((∃ (p : ValidPair), PairSum p = b) ∧ 
     (∃ (p : ValidPair), IsPerfectSquare p.val.1 ∨ IsPerfectSquare p.val.2) ∧
     (∀ (b' : ℕ), b' < b → 
       (∀ (p : ValidPair), PairSum p ≠ b' ∨ 
         (¬ IsPerfectSquare p.val.1 ∧ ¬ IsPerfectSquare p.val.2))))
    → b = 85) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_is_85_l1157_115727


namespace NUMINAMATH_CALUDE_g_equals_4_at_2_l1157_115740

/-- The function g(x) = 5x - 6 -/
def g (x : ℝ) : ℝ := 5 * x - 6

/-- Theorem: For the function g(x) = 5x - 6, the value of a that satisfies g(a) = 4 is a = 2 -/
theorem g_equals_4_at_2 : g 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_equals_4_at_2_l1157_115740


namespace NUMINAMATH_CALUDE_election_invalid_votes_l1157_115725

theorem election_invalid_votes 
  (total_polled : ℕ) 
  (vote_difference : ℕ) 
  (losing_percentage : ℚ) :
  total_polled = 850 →
  vote_difference = 500 →
  losing_percentage = 1/5 →
  (∃ (invalid_votes : ℕ), invalid_votes = 17) :=
by sorry

end NUMINAMATH_CALUDE_election_invalid_votes_l1157_115725


namespace NUMINAMATH_CALUDE_money_distribution_l1157_115770

/-- Given three people A, B, and C with a total of 600 Rs between them,
    where B and C together have 450 Rs, and C has 100 Rs,
    prove that A and C together have 250 Rs. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 600 →
  B + C = 450 →
  C = 100 →
  A + C = 250 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1157_115770


namespace NUMINAMATH_CALUDE_fencing_cost_l1157_115723

/-- Calculate the total cost of fencing a rectangular plot -/
theorem fencing_cost (length breadth perimeter cost_per_metre : ℝ) : 
  length = 200 →
  length = breadth + 20 →
  cost_per_metre = 26.5 →
  perimeter = 2 * (length + breadth) →
  perimeter * cost_per_metre = 20140 := by
  sorry

#check fencing_cost

end NUMINAMATH_CALUDE_fencing_cost_l1157_115723


namespace NUMINAMATH_CALUDE_permutation_and_combination_problem_l1157_115772

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem permutation_and_combination_problem :
  ∃ (x : ℕ), x > 0 ∧ 7 * A 6 x = 20 * A 7 (x - 1) ∧ 
  x = 3 ∧
  C 20 (20 - x) + C (17 + x) (x - 1) = 1330 :=
sorry

end NUMINAMATH_CALUDE_permutation_and_combination_problem_l1157_115772


namespace NUMINAMATH_CALUDE_figure_area_theorem_l1157_115775

theorem figure_area_theorem (x : ℝ) :
  let small_square_area := (3 * x)^2
  let large_square_area := (7 * x)^2
  let triangle_area := (1/2) * (3 * x) * (7 * x)
  small_square_area + large_square_area + triangle_area = 2200 →
  x = Real.sqrt (4400 / 137) :=
by sorry

end NUMINAMATH_CALUDE_figure_area_theorem_l1157_115775


namespace NUMINAMATH_CALUDE_expression_evaluation_l1157_115744

theorem expression_evaluation (a b : ℤ) (h1 : a = 1) (h2 : b = -2) :
  (a * b - 3 * a^2) - 2 * b^2 - 5 * a * b - (a^2 - 2 * a * b) = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1157_115744


namespace NUMINAMATH_CALUDE_simplify_expression_l1157_115785

theorem simplify_expression (y : ℝ) : 3*y + 5*y + 6*y + 10 = 14*y + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1157_115785


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1157_115721

theorem smallest_solution_of_equation :
  ∃ x : ℝ, x = 1 - Real.sqrt 10 ∧
  (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 12 ∧
  ∀ y : ℝ, (3 * y) / (y - 3) + (3 * y^2 - 27) / y = 12 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1157_115721


namespace NUMINAMATH_CALUDE_simplify_expression_l1157_115798

theorem simplify_expression (y : ℝ) : 3*y + 4*y^2 + 2 - (7 - 3*y - 4*y^2) = 8*y^2 + 6*y - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1157_115798


namespace NUMINAMATH_CALUDE_selling_price_ratio_l1157_115702

/-- Given an item with cost price c, prove that the ratio of selling prices
    y (at 20% profit) to x (at 10% loss) is 4/3 -/
theorem selling_price_ratio (c x y : ℝ) (hx : x = 0.9 * c) (hy : y = 1.2 * c) :
  y / x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l1157_115702


namespace NUMINAMATH_CALUDE_division_problem_l1157_115717

theorem division_problem (A : ℕ) (h : 59 = 8 * A + 3) : A = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1157_115717


namespace NUMINAMATH_CALUDE_complement_of_A_range_of_m_l1157_115787

-- Define the sets A and B
def A : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}
def B (m : ℝ) : Set ℝ := {x | x ≥ m - 1}

-- Theorem for the complement of A
theorem complement_of_A : 
  (Set.univ \ A) = {x : ℝ | x < 3} := by sorry

-- Theorem for the range of m when A ⊆ B
theorem range_of_m (h : A ⊆ B m) : 
  m ≤ 4 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_range_of_m_l1157_115787


namespace NUMINAMATH_CALUDE_infinite_perfect_square_phi_and_d_l1157_115757

/-- Euler's totient function -/
def phi (n : ℕ+) : ℕ := sorry

/-- Number of positive divisors function -/
def d (n : ℕ+) : ℕ := sorry

/-- A natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The set of positive integers n for which both φ(n) and d(n) are perfect squares -/
def S : Set ℕ+ := {n : ℕ+ | is_perfect_square (phi n) ∧ is_perfect_square (d n)}

theorem infinite_perfect_square_phi_and_d : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_infinite_perfect_square_phi_and_d_l1157_115757


namespace NUMINAMATH_CALUDE_roots_sum_minus_product_l1157_115720

theorem roots_sum_minus_product (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ + 3 = 0) → 
  (x₂^2 - 4*x₂ + 3 = 0) → 
  x₁ + x₂ - x₁*x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_minus_product_l1157_115720


namespace NUMINAMATH_CALUDE_anna_bob_matches_l1157_115776

/-- The number of players in the chess tournament -/
def total_players : ℕ := 12

/-- The number of players in each match -/
def players_per_match : ℕ := 6

/-- The number of players to choose after Anna and Bob are selected -/
def players_to_choose : ℕ := players_per_match - 2

/-- The number of remaining players after Anna and Bob are selected -/
def remaining_players : ℕ := total_players - 2

/-- The number of matches where Anna and Bob play together -/
def matches_together : ℕ := Nat.choose remaining_players players_to_choose

theorem anna_bob_matches :
  matches_together = 210 := by sorry

end NUMINAMATH_CALUDE_anna_bob_matches_l1157_115776


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1157_115780

-- Define variables
variable (a b x y : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : 3*a - 5*b - 2*a + b = a - 4*b := by sorry

-- Theorem for the second expression
theorem simplify_expression_2 : 4*x^2 + 5*x*y - 2*(2*x^2 - x*y) = 7*x*y := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1157_115780


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1157_115703

theorem complex_modulus_problem (z : ℂ) (h : z * (2 + Complex.I) = 5 * Complex.I - 10) : 
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1157_115703


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l1157_115750

theorem congruence_solutions_count : ∃ (S : Finset ℕ), 
  (∀ x ∈ S, x < 100 ∧ x > 0 ∧ (x + 13) % 34 = 55 % 34) ∧ 
  (∀ x < 100, x > 0 → (x + 13) % 34 = 55 % 34 → x ∈ S) ∧
  Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l1157_115750


namespace NUMINAMATH_CALUDE_both_sports_fans_l1157_115714

/-- Represents the number of students who like basketball -/
def basketball_fans : ℕ := 7

/-- Represents the number of students who like cricket -/
def cricket_fans : ℕ := 8

/-- Represents the number of students who like either basketball or cricket or both -/
def total_fans : ℕ := 10

/-- Theorem stating that the number of students who like both basketball and cricket is 5 -/
theorem both_sports_fans : 
  basketball_fans + cricket_fans - total_fans = 5 := by sorry

end NUMINAMATH_CALUDE_both_sports_fans_l1157_115714


namespace NUMINAMATH_CALUDE_vipers_count_l1157_115761

/-- The number of vipers in a swamp area -/
def num_vipers (num_crocodiles num_alligators total_animals : ℕ) : ℕ :=
  total_animals - (num_crocodiles + num_alligators)

/-- Theorem: The number of vipers in the swamp is 5 -/
theorem vipers_count : num_vipers 22 23 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vipers_count_l1157_115761


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l1157_115712

/-- The number of walnut trees in the park after planting -/
def trees_after_planting (initial_trees newly_planted_trees : ℕ) : ℕ :=
  initial_trees + newly_planted_trees

/-- Theorem: The number of walnut trees in the park after planting is 77 -/
theorem walnut_trees_after_planting :
  trees_after_planting 22 55 = 77 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l1157_115712


namespace NUMINAMATH_CALUDE_second_polygon_sides_l1157_115783

/-- Given two regular polygons with the same perimeter, where one polygon has 50 sides
    and each of its sides is three times as long as each side of the other polygon,
    prove that the number of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 →
  50 * (3 * s) = n * s → n = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l1157_115783


namespace NUMINAMATH_CALUDE_room_carpet_cost_l1157_115743

/-- Calculates the total cost of carpeting a rectangular room -/
def carpet_cost (length width cost_per_sq_yard : ℚ) : ℚ :=
  let length_yards := length / 3
  let width_yards := width / 3
  let area_sq_yards := length_yards * width_yards
  area_sq_yards * cost_per_sq_yard

/-- Theorem stating the total cost of carpeting the given room -/
theorem room_carpet_cost :
  carpet_cost 15 12 10 = 200 := by
  sorry

end NUMINAMATH_CALUDE_room_carpet_cost_l1157_115743


namespace NUMINAMATH_CALUDE_opposite_signs_and_greater_magnitude_l1157_115796

theorem opposite_signs_and_greater_magnitude (a b : ℝ) : 
  a * b < 0 → a + b < 0 → 
  ((a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0)) ∧ 
  (max (abs a) (abs b) > min (abs a) (abs b)) :=
by sorry

end NUMINAMATH_CALUDE_opposite_signs_and_greater_magnitude_l1157_115796


namespace NUMINAMATH_CALUDE_box_storage_calculation_l1157_115768

/-- Calculates the total number of boxes stored on a rectangular piece of land over two days -/
theorem box_storage_calculation (land_width land_length : ℕ) 
  (box_dimension : ℕ) (day1_layers day2_layers : ℕ) : 
  land_width = 44 → 
  land_length = 35 → 
  box_dimension = 1 → 
  day1_layers = 7 → 
  day2_layers = 3 → 
  (land_width / box_dimension) * (land_length / box_dimension) * (day1_layers + day2_layers) = 15400 :=
by
  sorry

#check box_storage_calculation

end NUMINAMATH_CALUDE_box_storage_calculation_l1157_115768


namespace NUMINAMATH_CALUDE_line_equations_l1157_115711

-- Define the lines m and n
def line_m (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line_n (x y : ℝ) : Prop := x + y - 3 = 0

-- Define point P as the intersection of m and n
def point_P : ℝ × ℝ := (1, 2)

-- Define points A and B
def point_A : ℝ × ℝ := (1, 3)
def point_B : ℝ × ℝ := (3, 2)

-- Define line l
def line_l (x y : ℝ) : Prop := (x + 2 * y - 4 = 0) ∨ (x = 2)

-- Define line l₁
def line_l1 (x y : ℝ) : Prop := y = -1/2 * x + 2

-- State the theorem
theorem line_equations :
  (∀ x y : ℝ, line_m x y ∧ line_n x y → (x, y) = point_P) →
  (∀ x y : ℝ, line_l x y → (x, y) = point_P) →
  (∀ x y : ℝ, line_l1 x y → (x, y) = point_P) →
  (∀ x y : ℝ, line_l x y → 
    abs ((2*x - 2*point_A.1 + y - point_A.2) / Real.sqrt (5)) = 
    abs ((2*x - 2*point_B.1 + y - point_B.2) / Real.sqrt (5))) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ x y : ℝ, line_l1 x y ↔ x/a + y/b = 1) ∧
    1/2 * a * b = 4) →
  (∀ x y : ℝ, line_l x y ∨ line_l1 x y) :=
sorry

end NUMINAMATH_CALUDE_line_equations_l1157_115711


namespace NUMINAMATH_CALUDE_ball_triangle_ratio_l1157_115793

theorem ball_triangle_ratio (q : ℝ) (hq : q ≠ 1) :
  let r := 2012 / q
  let side1 := r * (1 + q)
  let side2 := 2012 * (1 + q)
  let side3 := 2012 * (1 + q^2) / q
  (side1^2 + side2^2 + side3^2) / (side1 + side2 + side3) = 4024 :=
sorry

end NUMINAMATH_CALUDE_ball_triangle_ratio_l1157_115793


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1157_115751

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (m^2 - 4*m - 1 = 0) → 
  (n^2 - 4*n - 1 = 0) → 
  (m + n = 4) → 
  (m * n = -1) → 
  m + n - m * n = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1157_115751


namespace NUMINAMATH_CALUDE_max_rectangle_area_max_area_condition_l1157_115795

/-- The maximum area of a rectangle with perimeter 40 meters (excluding one side) is 200 square meters. -/
theorem max_rectangle_area (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 40) : x * y ≤ 200 := by
  sorry

/-- The maximum area is achieved when the length is twice the width. -/
theorem max_area_condition (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 40) : 
  x * y = 200 ↔ x = 2*y := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_max_area_condition_l1157_115795


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1157_115726

theorem max_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b + 2 * b * c) / (a^2 + b^2 + c^2) ≤ Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1157_115726


namespace NUMINAMATH_CALUDE_candidate_admission_criterion_l1157_115710

/-- Represents the constructibility of an angle division -/
inductive AngleDivision
  | Constructible
  | NotConstructible

/-- Represents a candidate's response to the angle division questions -/
structure CandidateResponse :=
  (div19 : AngleDivision)
  (div17 : AngleDivision)
  (div18 : AngleDivision)

/-- Determines if an angle of n degrees can be divided into n equal parts -/
def canDivideAngle (n : ℕ) : AngleDivision :=
  if n = 19 ∨ n = 17 then AngleDivision.Constructible
  else AngleDivision.NotConstructible

/-- Determines if a candidate's response is correct -/
def isCorrectResponse (response : CandidateResponse) : Prop :=
  response.div19 = canDivideAngle 19 ∧
  response.div17 = canDivideAngle 17 ∧
  response.div18 = canDivideAngle 18

/-- Determines if a candidate should be admitted based on their response -/
def shouldAdmit (response : CandidateResponse) : Prop :=
  isCorrectResponse response

theorem candidate_admission_criterion (response : CandidateResponse) :
  response.div19 = AngleDivision.Constructible ∧
  response.div17 = AngleDivision.Constructible ∧
  response.div18 = AngleDivision.NotConstructible →
  shouldAdmit response :=
by sorry

end NUMINAMATH_CALUDE_candidate_admission_criterion_l1157_115710


namespace NUMINAMATH_CALUDE_sum_equals_ten_l1157_115765

theorem sum_equals_ten (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 + (x + y)^3 + 30*x*y = 2000) : x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_ten_l1157_115765


namespace NUMINAMATH_CALUDE_slope_angle_of_intersecting_line_l1157_115788

/-- The slope angle of a line intersecting a circle -/
theorem slope_angle_of_intersecting_line (α : Real) : 
  (∃ (A B : ℝ × ℝ), 
    (∀ t : ℝ, (1 + t * Real.cos α, t * Real.sin α) ∈ {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4}) →
    A ∈ {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4} →
    B ∈ {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4} →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14) →
  α = π/4 ∨ α = 3*π/4 := by
sorry

end NUMINAMATH_CALUDE_slope_angle_of_intersecting_line_l1157_115788


namespace NUMINAMATH_CALUDE_square_reciprocal_sum_l1157_115709

theorem square_reciprocal_sum (p : ℝ) (h : p + 1/p = 10) :
  p^2 + 1/p^2 + 6 = 104 := by
  sorry

end NUMINAMATH_CALUDE_square_reciprocal_sum_l1157_115709


namespace NUMINAMATH_CALUDE_ellipse_equation_l1157_115779

/-- Prove that an ellipse passing through (2,0) with focal distance 2√2 has the equation x²/4 + y²/2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → x^2/4 + y^2/2 = 1) ↔
  (4/a^2 + 0^2/b^2 = 1 ∧ a^2 - b^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1157_115779


namespace NUMINAMATH_CALUDE_inverse_proportional_problem_l1157_115749

/-- Given that x and y are inversely proportional, x + y = 36, and x - y = 12,
    prove that when x = 8, y = 36. -/
theorem inverse_proportional_problem (x y : ℝ) (k : ℝ) 
  (h_inverse : x * y = k)
  (h_sum : x + y = 36)
  (h_diff : x - y = 12)
  (h_x : x = 8) : 
  y = 36 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportional_problem_l1157_115749


namespace NUMINAMATH_CALUDE_cubic_is_odd_l1157_115707

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f (x : ℝ) : ℝ := x^3

theorem cubic_is_odd : is_odd_function f := by
  sorry

end NUMINAMATH_CALUDE_cubic_is_odd_l1157_115707


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l1157_115766

def geometric_progression (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∃ r : ℝ, ∀ k : ℕ, a (k + 1) = a k * r

theorem fourth_term_of_geometric_progression (a : ℕ → ℝ) :
  geometric_progression a 3 →
  a 1 = 2^6 →
  a 2 = 2^3 →
  a 3 = 2^(3/2) →
  a 4 = 2^(3/4) :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l1157_115766


namespace NUMINAMATH_CALUDE_boxes_needed_l1157_115733

def initial_games : ℕ := 76
def sold_games : ℕ := 46
def games_per_box : ℕ := 5

theorem boxes_needed : 
  (initial_games - sold_games) / games_per_box = 6 :=
by sorry

end NUMINAMATH_CALUDE_boxes_needed_l1157_115733


namespace NUMINAMATH_CALUDE_symmetric_circle_l1157_115724

/-- The equation of a circle symmetric to x^2 + y^2 = 4 with respect to the line x + y - 1 = 0 -/
theorem symmetric_circle (x y : ℝ) : 
  (∀ x y, x^2 + y^2 = 4 → x + y - 1 = 0 → (x-1)^2 + (y-1)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_l1157_115724


namespace NUMINAMATH_CALUDE_modified_cube_edges_l1157_115735

/-- Represents a cube with a given side length. -/
structure Cube where
  sideLength : ℕ

/-- Represents the structure after removing unit cubes from corners. -/
structure ModifiedCube where
  originalCube : Cube
  removedCubeSize : ℕ

/-- Calculates the number of edges in the modified cube structure. -/
def edgesInModifiedCube (mc : ModifiedCube) : ℕ :=
  12 * 3  -- Each original edge is divided into 3 segments

/-- Theorem stating that a cube of side length 4 with unit cubes removed from corners has 36 edges. -/
theorem modified_cube_edges :
  ∀ (mc : ModifiedCube),
    mc.originalCube.sideLength = 4 →
    mc.removedCubeSize = 1 →
    edgesInModifiedCube mc = 36 := by
  sorry


end NUMINAMATH_CALUDE_modified_cube_edges_l1157_115735


namespace NUMINAMATH_CALUDE_radius_of_larger_circle_l1157_115737

/-- Two concentric circles with radii r and R, where R = 4r -/
structure ConcentricCircles where
  r : ℝ
  R : ℝ
  h : R = 4 * r

/-- A chord BC tangent to the inner circle -/
structure TangentChord (c : ConcentricCircles) where
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Diameter AC of the larger circle -/
structure Diameter (c : ConcentricCircles) where
  A : ℝ × ℝ
  C : ℝ × ℝ
  h : dist A C = 2 * c.R

theorem radius_of_larger_circle 
  (c : ConcentricCircles) 
  (d : Diameter c) 
  (t : TangentChord c) 
  (h : dist d.A t.B = 8) : 
  c.R = 16 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_larger_circle_l1157_115737


namespace NUMINAMATH_CALUDE_inequality_problem_l1157_115716

theorem inequality_problem (x : ℝ) : 
  (x - 1) * |4 - x| < 12 ∧ x - 2 > 0 → 4 < x ∧ x < 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l1157_115716


namespace NUMINAMATH_CALUDE_not_enough_unique_names_l1157_115755

/-- Represents the number of possible occurrences of each letter (a, o, u) in a standardized word -/
def letter_choices : ℕ := 7

/-- Represents the total number of tribe members -/
def tribe_members : ℕ := 400

/-- Represents the number of unique standardized words in the Mumbo-Jumbo language -/
def unique_words : ℕ := letter_choices ^ 3

theorem not_enough_unique_names : unique_words < tribe_members := by
  sorry

end NUMINAMATH_CALUDE_not_enough_unique_names_l1157_115755


namespace NUMINAMATH_CALUDE_a_properties_l1157_115777

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 3 * a n + 2 * (Int.sqrt (2 * (a n)^2 - 1)).toNat

theorem a_properties :
  (∀ n : ℕ, a n > 0) ∧
  (∀ m : ℕ, ¬(2015 ∣ a m)) := by
  sorry

end NUMINAMATH_CALUDE_a_properties_l1157_115777


namespace NUMINAMATH_CALUDE_xy_cube_plus_cube_xy_l1157_115719

theorem xy_cube_plus_cube_xy (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = -4) :
  x * y^3 + x^3 * y = -68 := by
  sorry

end NUMINAMATH_CALUDE_xy_cube_plus_cube_xy_l1157_115719


namespace NUMINAMATH_CALUDE_blackboard_divisibility_l1157_115730

/-- Represents the transformation process on the blackboard -/
def transform (n : ℕ) : ℕ := sorry

/-- The number on the blackboard after n minutes -/
def blackboard_number (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | n+1 => transform (blackboard_number n)

/-- The final number N on the blackboard -/
def N : ℕ := blackboard_number (sorry : ℕ)

theorem blackboard_divisibility :
  (9 ∣ N) → (99 ∣ N) := by sorry

end NUMINAMATH_CALUDE_blackboard_divisibility_l1157_115730


namespace NUMINAMATH_CALUDE_andrea_height_l1157_115742

/-- Given a tree's height and shadow length, and a person's shadow length,
    calculate the person's height assuming the same lighting conditions. -/
theorem andrea_height (tree_height shadow_tree shadow_andrea : ℝ) 
    (h_tree : tree_height = 70)
    (h_shadow_tree : shadow_tree = 14)
    (h_shadow_andrea : shadow_andrea = 3.5) :
  tree_height / shadow_tree * shadow_andrea = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_andrea_height_l1157_115742


namespace NUMINAMATH_CALUDE_prime_divisor_existence_l1157_115700

theorem prime_divisor_existence (p : Nat) (hp : p.Prime ∧ p ≥ 3) :
  ∃ N : Nat, ∀ x ≥ N, ∃ i ∈ Finset.range ((p + 3) / 2), 
    ∃ q : Nat, q.Prime ∧ q > p ∧ q ∣ (x + i + 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_divisor_existence_l1157_115700


namespace NUMINAMATH_CALUDE_max_daily_profit_l1157_115760

/-- The daily profit function for a factory -/
def daily_profit (x : ℕ) : ℚ :=
  -4/3 * (x^3 : ℚ) + 3600 * (x : ℚ)

/-- The maximum daily production capacity -/
def max_production : ℕ := 40

/-- Theorem stating the maximum daily profit and the production quantity that achieves it -/
theorem max_daily_profit :
  ∃ (x : ℕ), x ≤ max_production ∧
    (∀ (y : ℕ), y ≤ max_production → daily_profit y ≤ daily_profit x) ∧
    x = 30 ∧ daily_profit x = 72000 := by
  sorry

end NUMINAMATH_CALUDE_max_daily_profit_l1157_115760


namespace NUMINAMATH_CALUDE_tree_spacing_l1157_115789

/-- Given a yard of length 225 meters with 26 trees planted at equal distances,
    including one tree at each end, the distance between two consecutive trees is 9 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (tree_spacing : ℝ) : 
  yard_length = 225 →
  num_trees = 26 →
  tree_spacing * (num_trees - 1) = yard_length →
  tree_spacing = 9 := by sorry

end NUMINAMATH_CALUDE_tree_spacing_l1157_115789


namespace NUMINAMATH_CALUDE_potato_bag_weight_l1157_115786

theorem potato_bag_weight (weight : ℝ) (fraction : ℝ) : 
  weight = 36 → weight / fraction = 36 → fraction = 1 := by sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l1157_115786


namespace NUMINAMATH_CALUDE_max_value_of_f_l1157_115739

noncomputable def f (x : ℝ) : ℝ := min (3 * x + 1) (min (-1/3 * x + 2) (x + 4))

theorem max_value_of_f :
  ∃ (M : ℝ), M = 5/2 ∧ ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1157_115739


namespace NUMINAMATH_CALUDE_problem_solution_l1157_115759

theorem problem_solution : (0.5 : ℝ)^3 - (0.1 : ℝ)^3 / (0.5 : ℝ)^2 + 0.05 + (0.1 : ℝ)^2 = 0.181 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1157_115759


namespace NUMINAMATH_CALUDE_max_receptivity_and_duration_receptivity_comparison_insufficient_duration_l1157_115756

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 44
  else if 10 < x ∧ x ≤ 15 then 60
  else if 15 < x ∧ x ≤ 25 then -3 * x + 105
  else if 25 < x ∧ x ≤ 40 then 30
  else 0

-- Theorem 1: Maximum receptivity and duration
theorem max_receptivity_and_duration :
  (∀ x, 0 < x → x ≤ 40 → f x ≤ 60) ∧
  (∀ x, 10 ≤ x → x ≤ 15 → f x = 60) :=
sorry

-- Theorem 2: Receptivity comparison
theorem receptivity_comparison :
  f 5 > f 20 ∧ f 20 > f 35 :=
sorry

-- Theorem 3: Insufficient duration for required receptivity
theorem insufficient_duration :
  ¬ ∃ a : ℝ, 0 < a ∧ a + 12 ≤ 40 ∧ ∀ x, a ≤ x → x ≤ a + 12 → f x ≥ 56 :=
sorry

end NUMINAMATH_CALUDE_max_receptivity_and_duration_receptivity_comparison_insufficient_duration_l1157_115756


namespace NUMINAMATH_CALUDE_seeds_solution_l1157_115729

def seeds_problem (wednesday thursday friday : ℕ) : Prop :=
  wednesday = 5 * thursday ∧
  wednesday + thursday = 156 ∧
  friday = 4

theorem seeds_solution :
  ∃ (wednesday thursday friday : ℕ),
    seeds_problem wednesday thursday friday ∧
    wednesday = 130 ∧
    thursday = 26 ∧
    friday = 4 ∧
    wednesday + thursday + friday = 160 := by
  sorry

end NUMINAMATH_CALUDE_seeds_solution_l1157_115729


namespace NUMINAMATH_CALUDE_last_score_is_86_l1157_115734

def scores : List ℕ := [68, 74, 78, 83, 86, 95]

def is_integer_average (subset : List ℕ) : Prop :=
  ∃ n : ℕ, (subset.sum : ℚ) / subset.length = n

def satisfies_conditions (last_score : ℕ) : Prop :=
  last_score ∈ scores ∧
  ∀ k : ℕ, k ∈ (List.range 6) →
    is_integer_average (List.take k ((scores.filter (· ≠ last_score)) ++ [last_score]))

theorem last_score_is_86 :
  ∃ (last_score : ℕ), last_score = 86 ∧ 
    satisfies_conditions last_score ∧
    ∀ (other_score : ℕ), other_score ∈ scores → other_score ≠ 86 →
      satisfies_conditions other_score → False :=
sorry

end NUMINAMATH_CALUDE_last_score_is_86_l1157_115734


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l1157_115746

theorem no_simultaneous_squares : ¬ ∃ (x y : ℕ), 
  ∃ (a b : ℕ), (x^2 + 2*y = a^2) ∧ (y^2 + 2*x = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l1157_115746


namespace NUMINAMATH_CALUDE_sticker_remainder_l1157_115784

theorem sticker_remainder (nina_stickers : Nat) (oliver_stickers : Nat) (patty_stickers : Nat) 
  (package_size : Nat) (h1 : nina_stickers = 53) (h2 : oliver_stickers = 68) 
  (h3 : patty_stickers = 29) (h4 : package_size = 18) : 
  (nina_stickers + oliver_stickers + patty_stickers) % package_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_sticker_remainder_l1157_115784


namespace NUMINAMATH_CALUDE_vector_decomposition_l1157_115728

def x : ℝ × ℝ × ℝ := (-5, -5, 5)
def p : ℝ × ℝ × ℝ := (-2, 0, 1)
def q : ℝ × ℝ × ℝ := (1, 3, -1)
def r : ℝ × ℝ × ℝ := (0, 4, 1)

theorem vector_decomposition :
  x = p + (-3 : ℝ) • q + r := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l1157_115728


namespace NUMINAMATH_CALUDE_camel_height_28_feet_l1157_115782

/-- The height of a camel in feet, given the height of a hare in inches and their relative heights -/
def camel_height_in_feet (hare_height_inches : ℕ) (camel_hare_ratio : ℕ) : ℚ :=
  (hare_height_inches * camel_hare_ratio : ℚ) / 12

/-- Theorem stating the height of a camel in feet given specific measurements -/
theorem camel_height_28_feet :
  camel_height_in_feet 14 24 = 28 := by
  sorry

end NUMINAMATH_CALUDE_camel_height_28_feet_l1157_115782


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1157_115767

/-- The equation x^2 + ax + b = 0 has two distinct positive roots less than 1 -/
def has_two_distinct_positive_roots_less_than_one (a b : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
    x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0

/-- p is a necessary but not sufficient condition for q -/
theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (has_two_distinct_positive_roots_less_than_one a b → -2 < a ∧ a < 0 ∧ 0 < b ∧ b < 1) ∧
  ¬((-2 < a ∧ a < 0 ∧ 0 < b ∧ b < 1) → has_two_distinct_positive_roots_less_than_one a b) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1157_115767


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1157_115741

theorem pure_imaginary_condition (b : ℝ) (i : ℂ) : 
  i * i = -1 →  -- i is the imaginary unit
  (∃ (k : ℝ), i * (b * i + 1) = k * i) →  -- i(bi+1) is a pure imaginary number
  b = 0 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1157_115741


namespace NUMINAMATH_CALUDE_wendy_bill_split_l1157_115704

/-- The cost of a Taco Salad at Wendy's -/
def taco_salad_cost : ℝ := 10

theorem wendy_bill_split (
  num_friends : ℕ)
  (individual_payment : ℝ)
  (num_hamburgers : ℕ)
  (hamburger_cost : ℝ)
  (num_fries : ℕ)
  (fries_cost : ℝ)
  (num_lemonade : ℕ)
  (lemonade_cost : ℝ)
  (h1 : num_friends = 5)
  (h2 : individual_payment = 11)
  (h3 : num_hamburgers = 5)
  (h4 : hamburger_cost = 5)
  (h5 : num_fries = 4)
  (h6 : fries_cost = 2.5)
  (h7 : num_lemonade = 5)
  (h8 : lemonade_cost = 2) :
  taco_salad_cost = num_friends * individual_payment -
    (num_hamburgers * hamburger_cost + num_fries * fries_cost + num_lemonade * lemonade_cost) :=
by sorry

end NUMINAMATH_CALUDE_wendy_bill_split_l1157_115704


namespace NUMINAMATH_CALUDE_sum_equidistant_terms_l1157_115705

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_equidistant_terms 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a7 : a 7 = 12) : 
  a 2 + a 12 = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_equidistant_terms_l1157_115705


namespace NUMINAMATH_CALUDE_intersection_A_not_B_l1157_115732

-- Define the sets A and B
def A : Set ℝ := {x | (1 : ℝ) / |x - 1| < 1}
def B : Set ℝ := {x | x^2 - 5*x + 4 > 0}

-- Define the complement of B
def not_B : Set ℝ := {x | ¬ (x ∈ B)}

-- State the theorem
theorem intersection_A_not_B : A ∩ not_B = {x | 2 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_not_B_l1157_115732


namespace NUMINAMATH_CALUDE_plates_with_parents_is_eight_l1157_115781

/-- The number of plates used when Matt's parents join them -/
def plates_with_parents (total_plates : ℕ) (days_per_week : ℕ) (days_with_son : ℕ) (plates_per_person_with_son : ℕ) : ℕ :=
  (total_plates - days_with_son * plates_per_person_with_son * 2) / (days_per_week - days_with_son)

/-- Proof that the number of plates used when Matt's parents join them is 8 -/
theorem plates_with_parents_is_eight :
  plates_with_parents 38 7 3 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_plates_with_parents_is_eight_l1157_115781


namespace NUMINAMATH_CALUDE_right_triangular_pyramid_volume_l1157_115731

/-- A right triangular pyramid with base edge length 2 and pairwise perpendicular side edges -/
structure RightTriangularPyramid where
  base_edge_length : ℝ
  side_edges_perpendicular : Prop

/-- The volume of a right triangular pyramid -/
def volume (p : RightTriangularPyramid) : ℝ := sorry

/-- Theorem: The volume of a right triangular pyramid with base edge length 2 
    and pairwise perpendicular side edges is √2/3 -/
theorem right_triangular_pyramid_volume :
  ∀ (p : RightTriangularPyramid), 
    p.base_edge_length = 2 ∧ p.side_edges_perpendicular →
    volume p = Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_right_triangular_pyramid_volume_l1157_115731


namespace NUMINAMATH_CALUDE_box_counting_l1157_115794

theorem box_counting (initial_boxes : Nat) (boxes_per_fill : Nat) (non_empty_boxes : Nat) :
  initial_boxes = 7 →
  boxes_per_fill = 7 →
  non_empty_boxes = 10 →
  initial_boxes + (non_empty_boxes - 1) * boxes_per_fill = 77 := by
  sorry

end NUMINAMATH_CALUDE_box_counting_l1157_115794


namespace NUMINAMATH_CALUDE_fundraising_ratio_approx_one_third_l1157_115797

/-- The ratio of Miss Rollin's class contribution to the total school fundraising --/
def fundraising_ratio : ℚ :=
  let johnson_amount := 2300
  let sutton_amount := johnson_amount / 2
  let rollin_amount := sutton_amount * 8
  let total_after_fees := 27048
  let total_before_fees := total_after_fees / 0.98
  rollin_amount / total_before_fees

theorem fundraising_ratio_approx_one_third :
  abs (fundraising_ratio - 1/3) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_ratio_approx_one_third_l1157_115797


namespace NUMINAMATH_CALUDE_rental_days_proof_l1157_115745

/-- Calculates the number of rental days given the daily rate, 14-day rate, and total cost. -/
def rental_days (daily_rate : ℕ) (fourteen_day_rate : ℕ) (total_cost : ℕ) : ℕ :=
  let fourteen_day_periods := total_cost / fourteen_day_rate
  let remaining_cost := total_cost % fourteen_day_rate
  let additional_days := remaining_cost / daily_rate
  fourteen_day_periods * 14 + additional_days

/-- Proves that given the specified rental rates and total cost, the number of rental days is 20. -/
theorem rental_days_proof :
  rental_days 50 500 800 = 20 := by
  sorry

#eval rental_days 50 500 800

end NUMINAMATH_CALUDE_rental_days_proof_l1157_115745


namespace NUMINAMATH_CALUDE_cube_of_neg_cube_l1157_115791

theorem cube_of_neg_cube (x : ℝ) : (-x^3)^3 = -x^9 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_neg_cube_l1157_115791
